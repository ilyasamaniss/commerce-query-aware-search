from __future__ import annotations

from typing import Any, Dict

from shopify_ml_demo import catalog


def test_fetch_products_paginates_and_flattens_variants(monkeypatch) -> None:
    # Ensure pagination loop aggregates results and flattens variant edges.
    page1: Dict[str, Any] = {
        "products": {
            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
            "edges": [
                {
                    "node": {
                        "title": "A",
                        "variants": {"edges": [{"node": {"id": "v1"}}]},
                    }
                }
            ],
        }
    }
    page2: Dict[str, Any] = {
        "products": {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "edges": [
                {
                    "node": {
                        "title": "B",
                        "variants": {"edges": [{"node": {"id": "v2"}}]},
                    }
                }
            ],
        }
    }

    calls = []

    def fake_graphql(_query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(variables)
        return page1 if len(calls) == 1 else page2

    monkeypatch.setattr(catalog, "storefront_graphql", fake_graphql)

    products = catalog.fetch_products(limit=2)
    assert len(products) == 2
    assert products[0]["variants"] == [{"id": "v1"}]
    assert products[1]["variants"] == [{"id": "v2"}]
    assert calls[0]["first"] == 2
