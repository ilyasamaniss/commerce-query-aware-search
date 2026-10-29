from __future__ import annotations

import pytest

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


def _catalog() -> list[dict]:
    return [
        _product("Pokemon Pikachu LEGO Building Set"),
        _product("Pokemon Charizard LEGO Building Set"),
        _product("Pokemon Squirtle LEGO Building Set"),
    ]


def test_dense_retriever_recovers_misspelled_entity() -> None:
    # BM25 has no exact token for a typo; the subword embedder still finds the product.
    products = _catalog()
    assert build_retriever("bm25", products).search("charzard", k=1) == []
    results = build_retriever("dense", products).search("charzard", k=1)
    assert results[0]["title"] == "Pokemon Charizard LEGO Building Set"


@pytest.mark.parametrize("fusion", ["convex", "rrf"])
def test_hybrid_retriever_ranks_exact_and_fuzzy_matches(fusion: str) -> None:
    # Both fusion modes put the intended product first for exact and misspelled queries.
    retriever = build_retriever("hybrid", _catalog(), fusion=fusion)
    assert retriever.search("squirtle lego", k=1)[0]["title"] == "Pokemon Squirtle LEGO Building Set"
    assert retriever.search("pikchu", k=1)[0]["title"] == "Pokemon Pikachu LEGO Building Set"


def test_build_retriever_rejects_unknown_kind_and_fusion() -> None:
    # Misconfiguration fails at build time instead of silently degrading.
    with pytest.raises(ValueError):
        build_retriever("splade", _catalog())
    with pytest.raises(ValueError):
        build_retriever("hybrid", _catalog(), fusion="max")
