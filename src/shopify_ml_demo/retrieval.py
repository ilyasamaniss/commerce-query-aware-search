from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]


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

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(self.products)), key=lambda i: scores[i], reverse=True)

        results: List[Dict[str, Any]] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            results.append(self.products[i])
            if len(results) >= k:
                break
        return results


def build_retriever(kind: str, products: List[Dict[str, Any]]) -> Retriever:
    # Build and return a configured retriever over one catalog snapshot.
    corpus_tokens = [tokenize(product_to_text(p)) for p in products]
    kind_norm = kind.strip().lower()

    if kind_norm == "overlap":
        return OverlapRetriever(products=products, corpus_tokens=corpus_tokens)

    if kind_norm == "bm25":
        if BM25Okapi is None:
            raise RuntimeError("BM25 retriever requested but rank-bm25 is not installed")
        bm25 = BM25Okapi(corpus_tokens)
        return BM25Retriever(products=products, corpus_tokens=corpus_tokens, bm25=bm25)

    raise ValueError(f"Unknown retriever kind: {kind}. Expected 'overlap' or 'bm25'.")
