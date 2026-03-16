from __future__ import annotations

from typing import Any, Dict, List, Optional

from shopify_ml_demo.schemas import QueryConstraints

def _product_min_price(product: Dict[str, Any]) -> Optional[float]:
    # Extract min variant price from product shape.
    try:
        return float(product["priceRange"]["minVariantPrice"]["amount"])
    except (KeyError, TypeError, ValueError):
        return None


def _passes(product: Dict[str, Any], constraints: QueryConstraints) -> bool:
    # Validate one product against deterministic hard constraints only.
    if constraints.in_stock is True and not bool(product.get("availableForSale", False)):
        return False

    amount = _product_min_price(product)
    if constraints.price_min is not None and (amount is None or amount < constraints.price_min):
        return False
    if constraints.price_max is not None and (amount is None or amount > constraints.price_max):
        return False

    return True


def apply_constraints(products: List[Dict[str, Any]], constraints: QueryConstraints) -> List[Dict[str, Any]]:
    # Filter retrieved candidates using catalog-backed hard checks.
    return [product for product in products if _passes(product, constraints)]
