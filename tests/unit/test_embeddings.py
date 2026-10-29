from __future__ import annotations

import numpy as np
import pytest

from shopify_ml_demo.embeddings import HashingEmbedder, OllamaEmbedder, build_embedder


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    # Same text gives the same unit-length vector across calls.
    embedder = HashingEmbedder(dim=256)
    first = embedder.embed(["pikachu lego"])
    second = embedder.embed(["pikachu lego"])
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_hashing_embedder_scores_misspelling_close_to_target() -> None:
    # Character n-grams keep a typo closer to its intended word than to an unrelated one.
    embedder = HashingEmbedder()
    typo, target, other = embedder.embed(["pikchu", "pikachu", "charizard"])
    assert typo @ target > typo @ other


def test_build_embedder_parses_specs() -> None:
    # Known specs build the right embedder; unknown specs fail loudly.
    assert isinstance(build_embedder("hashing"), HashingEmbedder)
    ollama = build_embedder("ollama:nomic-embed-text")
    assert isinstance(ollama, OllamaEmbedder)
    assert ollama.model == "nomic-embed-text"
    assert ollama.url.endswith("/api/embed")
    with pytest.raises(ValueError):
        build_embedder("ollama:")
    with pytest.raises(ValueError):
        build_embedder("word2vec")
