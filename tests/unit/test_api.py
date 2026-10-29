from __future__ import annotations

import json
from pathlib import Path

from shopify_ml_demo.api import SearchService, handle_request
from shopify_ml_demo.catalog import load_catalog_snapshot
from shopify_ml_demo.schemas import QueryAnalysis, QueryConstraints

CATALOG = Path(__file__).resolve().parents[2] / "benchmarks" / "catalog" / "sample_catalog.json"


def _service(**kwargs) -> SearchService:
    return SearchService(products=load_catalog_snapshot(CATALOG), **kwargs)


def test_search_returns_product_summaries() -> None:
    # Baseline BM25 search returns UI-ready product rows.
    status, body = handle_request(_service(), "/api/search?q=pikachu+lego&retriever=bm25&k=3")
    assert status == 200
    assert body["analyzer"] == "none"
    assert body["products"][0]["title"] == "Pokemon Pikachu LEGO Building Set"
    assert set(body["products"][0]) == {"title", "handle", "price", "currency", "available", "imageUrl", "tags"}
    json.dumps(body)


def test_hybrid_search_handles_misspelling() -> None:
    status, body = handle_request(_service(), "/api/search?q=charzard&retriever=hybrid&k=1")
    assert status == 200
    assert body["products"][0]["title"] == "Pokemon Charizard LEGO Building Set"


def test_query_aware_search_applies_constraints() -> None:
    # With an analyzer, deterministic filtering removes out-of-stock products.
    def analyzer(query: str) -> QueryAnalysis:
        return QueryAnalysis(
            intent="search",
            query_rewrite="pokemon lego",
            constraints=QueryConstraints(price_min=None, price_max=None, in_stock=True, color=None),
        )

    status, body = handle_request(_service(analyzer=analyzer), "/api/search?q=in+stock+lego&analyzer=stub&k=20")
    assert status == 200
    assert body["products"] and all(p["available"] for p in body["products"])
    assert any(step["tool"] == "deterministic_filter" for step in body["trace"])


def test_bad_requests_and_backend_failures() -> None:
    service = _service()
    assert handle_request(service, "/api/search?q=")[0] == 400
    assert handle_request(service, "/api/search?q=x&retriever=splade")[0] == 400
    assert handle_request(service, "/api/search?q=x&k=0")[0] == 400
    assert handle_request(service, "/api/nope")[0] == 404

    def broken(query: str) -> QueryAnalysis:
        raise RuntimeError("ollama down")

    status, body = handle_request(_service(analyzer=broken), "/api/search?q=x&analyzer=gemma3:4b")
    assert status == 503 and "ollama down" in body["error"]


def test_config_lists_retrievers() -> None:
    status, body = handle_request(_service(), "/api/config")
    assert status == 200
    assert "hybrid" in body["retrievers"] and body["catalogSize"] == 15
