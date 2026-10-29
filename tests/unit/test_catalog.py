from __future__ import annotations

from pathlib import Path
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


def test_catalog_snapshot_round_trip(tmp_path: Path) -> None:
    # A written snapshot loads back unchanged and honors the product limit.
    products = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    path = tmp_path / "catalog.json"
    catalog.write_catalog_snapshot(path, products)

    assert catalog.load_catalog_snapshot(path) == products
    assert catalog.load_products(path, limit=2) == products[:2]
    assert catalog.load_products(path, limit=None) == products


def test_load_products_without_snapshot_uses_live_api(monkeypatch) -> None:
    # No snapshot path means the Storefront API client is used.
    monkeypatch.setattr(catalog, "fetch_products", lambda limit: [{"title": "live", "limit": limit}])
    assert catalog.load_products(None, limit=7) == [{"title": "live", "limit": 7}]


def test_bundled_sample_catalog_is_valid() -> None:
    # The committed sample catalog has the fields search and filtering rely on.
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "catalog" / "sample_catalog.json"
    products = catalog.load_catalog_snapshot(path)
    assert len(products) >= 10
    for product in products:
        assert product["title"]
        assert isinstance(product["availableForSale"], bool)
        float(product["priceRange"]["minVariantPrice"]["amount"])
