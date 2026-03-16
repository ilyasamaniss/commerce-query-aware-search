from __future__ import annotations

from shopify_ml_demo.filtering import apply_constraints
from shopify_ml_demo.schemas import QueryConstraints


def _product(title: str, price: float, available: bool, tags: list[str]) -> dict:
    # Build minimal product shape required by filtering checks.
    return {
        "title": title,
        "description": "pokemon lego set",
        "tags": tags,
        "availableForSale": available,
        "priceRange": {
            "minVariantPrice": {"amount": str(price), "currencyCode": "USD"},
            "maxVariantPrice": {"amount": str(price), "currencyCode": "USD"},
        },
    }


def test_apply_constraints_enforces_price_and_stock_only() -> None:
    # Color is not a hard constraint; enforce only deterministic fields.
    products = [
        _product("Squirtle", 89.0, True, ["blue"]),
        _product("Lapras", 79.0, False, ["blue"]),
        _product("Charizard", 129.0, True, ["red"]),
    ]
    constraints = QueryConstraints(price_min=None, price_max=100.0, in_stock=True, color="blue")

    filtered = apply_constraints(products, constraints)
    assert [p["title"] for p in filtered] == ["Squirtle"]


def test_apply_constraints_ignores_color_constraint() -> None:
    # Requested behavior: color should not filter out otherwise valid products.
    products = [
        _product("Pikachu", 89.0, True, ["electric"]),
        _product("Lapras", 79.0, False, ["blue"]),
    ]
    constraints = QueryConstraints(price_min=None, price_max=100.0, in_stock=True, color="yellow")

    filtered = apply_constraints(products, constraints)
    assert [p["title"] for p in filtered] == ["Pikachu"]
