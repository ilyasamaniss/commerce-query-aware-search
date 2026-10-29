from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from shopify_ml_demo.embeddings import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    # Tokenize text into lowercase alphanumeric terms.
    return _TOKEN_RE.findall((text or "").lower())


def product_to_text(product: Dict[str, Any]) -> str:
    # Build one searchable text blob from a product record.
    tags = product.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    return " ".join(
        [
            str(product.get("title", "")),
            str(product.get("vendor", "")),
            str(product.get("productType", "")),
            str(product.get("description", "")),
            " ".join(map(str, tags)),
        ]
    )


class Retriever(Protocol):
    def search(self, query: str, k: int) -> List[Dict[str, Any]]: ...


@dataclass(frozen=True)
class OverlapRetriever:
    # Baseline retriever that counts token overlap with each doc.
    products: Sequence[Dict[str, Any]]
    corpus_tokens: Sequence[Sequence[str]]

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_set = set(q_tokens)
        scores: List[tuple[int, int]] = []
        for i, doc_tokens in enumerate(self.corpus_tokens):
            doc_set = set(doc_tokens)
            score = sum(1 for token in q_set if token in doc_set)
            if score > 0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.products[i] for _, i in scores[:k]]


@dataclass(frozen=True)
class BM25Retriever:
    # Strong keyword retriever with IDF-style term weighting.
    products: Sequence[Dict[str, Any]]
    corpus_tokens: Sequence[Sequence[str]]
    bm25: Any

    def score_all(self, query: str) -> np.ndarray:
        # BM25 score for every product, aligned with self.products.
        q_tokens = tokenize(query)
        if not q_tokens:
            return np.zeros(len(self.products))
        return np.asarray(self.bm25.get_scores(q_tokens), dtype=float)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not tokenize(query):
            return []

        scores = self.score_all(query)
        ranked = sorted(range(len(self.products)), key=lambda i: scores[i], reverse=True)

        results: List[Dict[str, Any]] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            results.append(self.products[i])
            if len(results) >= k:
                break
        return results


@dataclass(frozen=True)
class DenseRetriever:
    # Embedding retriever: cosine similarity between query and product vectors.
    products: Sequence[Dict[str, Any]]
    embedder: "Embedder"
    doc_vectors: np.ndarray

    def score_all(self, query: str) -> np.ndarray:
        # Cosine similarity for every product, aligned with self.products.
        if not tokenize(query) or len(self.products) == 0:
            return np.zeros(len(self.products))
        return self.doc_vectors @ self.embedder.embed([query])[0]

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not tokenize(query) or len(self.products) == 0:
            return []
        scores = self.score_all(query)
        ranked = np.argsort(-scores, kind="stable")
        return [self.products[i] for i in ranked[:k] if scores[i] > 0]


def _min_max(scores: np.ndarray) -> np.ndarray:
    # Rescale scores to [0, 1]; a flat score vector carries no ranking signal.
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _ranks(scores: np.ndarray) -> np.ndarray:
    # 1-based rank of every item under descending score.
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


@dataclass(frozen=True)
class HybridRetriever:
    # Fuses sparse (BM25) and dense scores over the same catalog.
    # fusion="convex": alpha * dense + (1 - alpha) * sparse on min-max normalized scores.
    # fusion="rrf": Reciprocal Rank Fusion, sum of 1 / (rrf_k + rank) across both rankings.
    products: Sequence[Dict[str, Any]]
    sparse: BM25Retriever
    dense: DenseRetriever
    fusion: str = "convex"
    alpha: float = 0.5
    rrf_k: int = 60

    def score_all(self, query: str) -> np.ndarray:
        sparse_scores = self.sparse.score_all(query)
        dense_scores = self.dense.score_all(query)
        if self.fusion == "rrf":
            return 1.0 / (self.rrf_k + _ranks(sparse_scores)) + 1.0 / (self.rrf_k + _ranks(dense_scores))
        return self.alpha * _min_max(dense_scores) + (1.0 - self.alpha) * _min_max(sparse_scores)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not tokenize(query) or len(self.products) == 0:
            return []
        scores = self.score_all(query)
        ranked = np.argsort(-scores, kind="stable")
        return [self.products[i] for i in ranked[:k] if scores[i] > 0]


RETRIEVER_KINDS = ("overlap", "bm25", "dense", "hybrid")


def build_retriever(
    kind: str,
    products: List[Dict[str, Any]],
    embedder: Optional["Embedder"] = None,
    fusion: str = "convex",
    alpha: float = 0.5,
) -> Retriever:
    # Build and return a configured retriever over one catalog snapshot.
    corpus_tokens = [tokenize(product_to_text(p)) for p in products]
    kind_norm = kind.strip().lower()

    if kind_norm == "overlap":
        return OverlapRetriever(products=products, corpus_tokens=corpus_tokens)

    if kind_norm in {"bm25", "hybrid"}:
        if BM25Okapi is None:
            raise RuntimeError("BM25 retriever requested but rank-bm25 is not installed")
        bm25 = BM25Okapi(corpus_tokens)
        sparse = BM25Retriever(products=products, corpus_tokens=corpus_tokens, bm25=bm25)
        if kind_norm == "bm25":
            return sparse

    if kind_norm in {"dense", "hybrid"}:
        if embedder is None:
            from shopify_ml_demo.embeddings import HashingEmbedder

            embedder = HashingEmbedder()
        doc_vectors = embedder.embed([product_to_text(p) for p in products])
        dense = DenseRetriever(products=products, embedder=embedder, doc_vectors=doc_vectors)
        if kind_norm == "dense":
            return dense
        if fusion not in {"convex", "rrf"}:
            raise ValueError(f"Unknown fusion: {fusion}. Expected 'convex' or 'rrf'.")
        return HybridRetriever(products=products, sparse=sparse, dense=dense, fusion=fusion, alpha=alpha)

    raise ValueError(f"Unknown retriever kind: {kind}. Expected one of {', '.join(RETRIEVER_KINDS)}.")
