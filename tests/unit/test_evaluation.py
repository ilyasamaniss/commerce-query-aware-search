from __future__ import annotations

import json
from pathlib import Path

from shopify_ml_demo.metrics import score_results_file


def _product(title: str, price: float, in_stock: bool) -> dict:
    # Build minimal catalog rows for metric validation.
    return {
        "title": title,
        "availableForSale": in_stock,
        "priceRange": {
            "minVariantPrice": {"amount": str(price), "currencyCode": "USD"},
            "maxVariantPrice": {"amount": str(price), "currencyCode": "USD"},
        },
    }


def test_score_results_file_tracks_passes_and_failures(tmp_path: Path) -> None:
    # Verify scorecard counts and failure collection on mixed outcomes.
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "id": "t1",
            "query": "pikachu lego",
            "expect": {"must_include_titles": ["Pikachu"], "top_k": 1},
            "picked_titles": ["Pikachu"],
        },
        {
            "id": "t2",
            "query": "under 50",
            "expect": {"constraints": {"price_max": 50}},
            "picked_titles": ["Charizard"],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    products = [_product("Pikachu", 49.0, True), _product("Charizard", 120.0, True)]
    card = score_results_file(path, products=products, k=5)

    assert card.total == 2
    assert card.passed == 1
    assert len(card.failures) == 1
