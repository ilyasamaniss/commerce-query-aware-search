from __future__ import annotations

from typing import Any, Dict, List, Optional

from shopify_ml_demo.config import load_shopify_settings


def _storefront_endpoint() -> str:
    # Build the GraphQL endpoint from configured domain and API version.
    cfg = load_shopify_settings()
    if cfg.store_domain.startswith("http"):
        base = cfg.store_domain.rstrip("/")
    else:
        base = f"https://{cfg.store_domain}"
    return f"{base}/api/{cfg.api_version}/graphql.json"


def _headers() -> Dict[str, str]:
    # Build authenticated Storefront API headers.
    cfg = load_shopify_settings()
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Shopify-Storefront-Access-Token": cfg.storefront_token,
    }


def storefront_graphql(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Execute GraphQL and return the top-level data field.
    import requests

    resp = requests.post(
        _storefront_endpoint(),
        headers=_headers(),
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Storefront API HTTP {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"Storefront API GraphQL errors: {payload['errors']}")

    data = payload.get("data")
    if data is None:
        raise RuntimeError(f"Storefront API missing data: {payload}")
    return data


def fetch_products(limit: Optional[int] = 50) -> List[Dict[str, Any]]:
    # Fetch products with pagination and flatten variants for downstream use.
    # If limit is None, fetch the full catalog.
    query = """
    query Products($first: Int!, $after: String) {
      products(first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        edges {
          node {
            id
            title
            handle
            vendor
            productType
            tags
            availableForSale
            description
            featuredImage { url altText }
            priceRange {
              minVariantPrice { amount currencyCode }
              maxVariantPrice { amount currencyCode }
            }
            variants(first: 50) {
              edges {
                node {
                  id
                  title
                  sku
                  availableForSale
                  price { amount currencyCode }
                }
              }
            }
          }
        }
      }
    }
    """

    results: List[Dict[str, Any]] = []
    after: Optional[str] = None

    while limit is None or len(results) < limit:
        first = 50 if limit is None else min(50, limit - len(results))
        data = storefront_graphql(query, {"first": first, "after": after})
        block = data["products"]

        for edge in block["edges"]:
            node = edge["node"]
            node["variants"] = [ve["node"] for ve in node["variants"]["edges"]]
            results.append(node)

        page = block["pageInfo"]
        if not page["hasNextPage"]:
            break
        after = page["endCursor"]

    return results
