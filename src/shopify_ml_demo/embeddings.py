from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from typing import List, Protocol, Sequence

import numpy as np

from shopify_ml_demo.retrieval import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    # Row-normalize so dot products are cosine similarities.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass(frozen=True)
class HashingEmbedder:
    # Offline subword embedder: hashed character n-grams (fastText-style), no model download.
    # Captures misspellings and partial matches; it does not capture synonyms.
    dim: int = 1024
    ngram_range: tuple[int, int] = (3, 5)
    name: str = "hashing"

    def _features(self, text: str) -> List[str]:
        features: List[str] = []
        lo, hi = self.ngram_range
        for token in tokenize(text):
            features.append(f"w:{token}")
            padded = f"<{token}>"
            for n in range(lo, hi + 1):
                features.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return features

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                col, sign = self._bucket(feature)
                matrix[row, col] += sign
        return l2_normalize(matrix)


@dataclass(frozen=True)
class OllamaEmbedder:
    # Neural embeddings from a local Ollama embedding model (e.g. nomic-embed-text).
    model: str = "nomic-embed-text"
    url: str = "http://localhost:11434/api/embed"
    timeout: float = 60.0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError(f"Ollama embedding response malformed: {str(payload)[:200]}")
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def build_embedder(spec: str) -> Embedder:
    # Parse an embedder spec: "hashing" or "ollama:<model>".
    spec_norm = spec.strip()
    if spec_norm.lower() == "hashing":
        return HashingEmbedder()
    if spec_norm.lower().startswith("ollama:"):
        model = spec_norm.split(":", 1)[1].strip()
        if not model:
            raise ValueError("Embedder spec 'ollama:' needs a model name, e.g. ollama:nomic-embed-text")
        from shopify_ml_demo.config import load_ollama_settings

        base = load_ollama_settings().url.rsplit("/api/", 1)[0]
        return OllamaEmbedder(model=model, url=f"{base}/api/embed")
    raise ValueError(f"Unknown embedder: {spec}. Expected 'hashing' or 'ollama:<model>'.")
