from __future__ import annotations

from shopify_ml_demo.query_analysis import _sanitize_analysis


def test_sanitize_analysis_normalizes_payload() -> None:
    # Validate conversion of malformed analyzer output into typed values.
    raw = {
        "intent": "SEARCH",
        "query_rewrite": "pikachu lego",
        "candidate_entities": ["Pikachu", " Raichu ", "", "Pikachu"],
        "constraints": {
            "price_min": "50",
            "price_max": 100,
            "in_stock": "true",
            "color": " Blue ",
        },
    }
    analysis = _sanitize_analysis(raw, "pikchu lego")

    assert analysis.intent == "search"
    assert analysis.query_rewrite == "pikachu lego"
    assert analysis.constraints.price_min == 50.0
    assert analysis.constraints.price_max == 100.0
    assert analysis.constraints.in_stock is True
    assert analysis.constraints.color == "blue"
    assert analysis.candidate_entities == ["pikachu", "raichu"]
