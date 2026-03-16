from __future__ import annotations

from typing import Any, Dict, List

import pytest

from shopify_ml_demo.schemas import QueryAnalysis, QueryConstraints
from shopify_ml_demo.search import StructuredSearchEngine


class DummyRetriever:
    # Test double that records the final retrieval query passed by the engine.
    def __init__(self, products: List[Dict[str, Any]]) -> None:
        self.products = products
        self.last_query = ""

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        self.last_query = query
        return self.products[:k]


def _product(title: str) -> Dict[str, Any]:
    # Build minimal product shape for search result rendering.
    return {
        "title": title,
        "handle": title.lower().replace(" ", "-"),
        "priceRange": {
            "minVariantPrice": {"amount": "99.0", "currencyCode": "USD"},
            "maxVariantPrice": {"amount": "99.0", "currencyCode": "USD"},
        },
        "availableForSale": True,
    }


def test_structured_search_uses_candidate_entities_in_retrieval_query() -> None:
    # Candidate entities should be appended to the retrieval query.
    retriever = DummyRetriever([_product("Pokemon Pikachu LEGO Building Set")])

    def fake_analyzer(_query: str) -> QueryAnalysis:
        return QueryAnalysis(
            intent="search",
            query_rewrite="yellow pokemon",
            constraints=QueryConstraints(price_min=None, price_max=None, in_stock=None, color=None),
            candidate_entities=["pikachu", "raichu"],
        )

    engine = StructuredSearchEngine(retriever=retriever, analyzer=fake_analyzer)
    _ = engine.search("yellow pokemon", products=retriever.products, k=1)

    assert retriever.last_query == "yellow pokemon pikachu raichu"


def test_structured_search_raises_when_query_analysis_fails() -> None:
    # Structured search should fail closed instead of silently degrading to lexical behavior.
    retriever = DummyRetriever([_product("Pokemon Pikachu LEGO Building Set")])

    def failing_analyzer(_query: str) -> QueryAnalysis:
        raise RuntimeError("ollama unavailable")

    engine = StructuredSearchEngine(retriever=retriever, analyzer=failing_analyzer)

    with pytest.raises(RuntimeError, match="ollama unavailable"):
        engine.search("yellow pokemon", products=retriever.products, k=1)
