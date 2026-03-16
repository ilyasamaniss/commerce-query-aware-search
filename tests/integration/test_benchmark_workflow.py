from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopify_ml_demo.evaluation import run_benchmark
from shopify_ml_demo.metrics import write_score_artifacts
from shopify_ml_demo.schemas import QueryAnalysis


def _product(title: str, price: float, available: bool) -> dict:
    # Build a minimal catalog fixture for an end-to-end benchmark run.
    return {
        "title": title,
        "handle": title.lower().replace(" ", "-"),
        "vendor": "Pokemon",
        "productType": "Building Set",
        "description": "pokemon lego building set",
        "tags": ["lego"],
        "availableForSale": available,
        "priceRange": {
            "minVariantPrice": {"amount": str(price), "currencyCode": "USD"},
            "maxVariantPrice": {"amount": str(price), "currencyCode": "USD"},
        },
    }


def test_benchmark_then_score_workflow(tmp_path: Path) -> None:
    # Smoke the benchmark and scoring pipeline with local fixtures only.
    tasks_path = tmp_path / "tasks.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    failures_path = tmp_path / "failures.jsonl"

    tasks = [
        {
            "id": "t1",
            "query": "pikachu lego",
            "expect": {"must_include_titles": ["Pokemon Pikachu LEGO Building Set"], "top_k": 1},
        },
        {
            "id": "t2",
            "query": "pokemon lego under 50",
            "expect": {"constraints": {"price_max": 50}, "max_results": 0},
        },
    ]
    with tasks_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")

    products = [
        _product("Pokemon Pikachu LEGO Building Set", 49.0, True),
        _product("Pokemon Charizard LEGO Building Set", 120.0, True),
    ]

    summary = run_benchmark(
        tasks_path=tasks_path,
        out_path=predictions_path,
        retriever_kind="overlap",
        query_analyzer=None,
        product_limit=200,
        k=1,
        products=products,
    )
    assert summary.total_tasks == 2
    assert predictions_path.exists()

    scorecard = write_score_artifacts(
        results_path=predictions_path,
        metrics_out_path=metrics_path,
        failures_out_path=failures_path,
        products=products,
        k=5,
    )
    assert metrics_path.exists()
    assert failures_path.exists()
    assert scorecard.total == 2


def test_structured_benchmark_fails_closed_when_analyzer_is_unavailable(tmp_path: Path) -> None:
    # Structured benchmark runs should fail instead of silently downgrading to lexical retrieval.
    tasks_path = tmp_path / "tasks.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    with tasks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "t1", "query": "yellow pokemon", "expect": {}}) + "\n")

    products = [
        _product("Pokemon Pikachu LEGO Building Set", 49.0, True),
    ]

    def failing_analyzer(_query: str) -> QueryAnalysis:
        raise RuntimeError("ollama unavailable")

    with pytest.raises(RuntimeError, match="ollama unavailable"):
        run_benchmark(
            tasks_path=tasks_path,
            out_path=predictions_path,
            retriever_kind="overlap",
            query_analyzer="gemma3:4b",
            product_limit=200,
            k=1,
            products=products,
            analyzer=failing_analyzer,
        )

    assert not predictions_path.exists()


def test_benchmark_treats_none_analyzer_as_baseline(tmp_path: Path) -> None:
    # Tangle can pass a sentinel analyzer value without triggering structured search.
    tasks_path = tmp_path / "tasks.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    with tasks_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "t1",
                    "query": "pikachu lego",
                    "expect": {"must_include_titles": ["Pokemon Pikachu LEGO Building Set"]},
                }
            )
            + "\n"
        )

    products = [
        _product("Pokemon Pikachu LEGO Building Set", 49.0, True),
    ]

    summary = run_benchmark(
        tasks_path=tasks_path,
        out_path=predictions_path,
        retriever_kind="overlap",
        query_analyzer="none",
        product_limit=200,
        k=1,
        products=products,
    )

    assert summary.query_analyzer == "none"
    assert predictions_path.exists()
