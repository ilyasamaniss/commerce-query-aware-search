from __future__ import annotations

from shopify_ml_demo.retrieval import build_retriever


def _product(title: str) -> dict:
    # Build minimal product shape required by retrieval text construction.
    return {
        "title": title,
        "vendor": "Pokemon",
        "productType": "Building Set",
        "description": "pokemon lego building set",
        "tags": ["lego"],
    }


def test_overlap_retriever_returns_expected_match() -> None:
    # Overlap retriever should rank exact keyword match first.
    products = [_product("Pokemon Pikachu LEGO Building Set"), _product("Pokemon Charizard LEGO Building Set")]
    retriever = build_retriever("overlap", products)
    results = retriever.search("pikachu lego", k=1)
    assert results[0]["title"] == "Pokemon Pikachu LEGO Building Set"
