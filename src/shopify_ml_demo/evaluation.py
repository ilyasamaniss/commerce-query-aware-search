from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from shopify_ml_demo.catalog import fetch_products
from shopify_ml_demo.retrieval import build_retriever
from shopify_ml_demo.schemas import BenchmarkPrediction, BenchmarkRunSummary, BenchmarkTask, QueryAnalysis
from shopify_ml_demo.search import BaselineSearchEngine, StructuredSearchEngine


def normalize_query_analyzer(query_analyzer: Optional[str]) -> Optional[str]:
    # Treat empty and sentinel analyzer values as baseline mode.
    if query_analyzer is None:
        return None
    normalized = query_analyzer.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def read_tasks(path: Path) -> List[BenchmarkTask]:
    # Load benchmark tasks from JSONL into typed task objects.
    tasks: List[BenchmarkTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(
                BenchmarkTask(
                    task_id=str(row.get("id", "")),
                    query=str(row.get("query", "")),
                    expect=dict(row.get("expect", {}) or {}),
                )
            )
    return tasks


def write_predictions(path: Path, predictions: Iterable[BenchmarkPrediction]) -> None:
    # Write benchmark predictions as JSONL artifacts.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_row(), ensure_ascii=False) + "\n")


def read_predictions(path: Path) -> List[BenchmarkPrediction]:
    # Load predictions JSONL back into typed rows.
    predictions: List[BenchmarkPrediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            predictions.append(
                BenchmarkPrediction(
                    task_id=str(row.get("id", "")),
                    query=str(row.get("query", "")),
                    expect=dict(row.get("expect", {}) or {}),
                    answer=str(row.get("answer", "")),
                    picked_titles=list(row.get("picked_titles", [])),
                    trace=list(row.get("trace", [])),
                )
            )
    return predictions


def run_benchmark(
    *,
    tasks_path: Path,
    out_path: Path,
    retriever_kind: str = "bm25",
    query_analyzer: Optional[str] = None,
    product_limit: int = 200,
    k: int = 3,
    products: Optional[List[Dict[str, object]]] = None,
    analyzer: Optional[Callable[[str], QueryAnalysis]] = None,
) -> BenchmarkRunSummary:
    # Execute benchmark tasks and persist predictions without scoring them.
    tasks = read_tasks(tasks_path)
    query_analyzer = normalize_query_analyzer(query_analyzer)
    if query_analyzer:
        os.environ["OLLAMA_MODEL"] = query_analyzer

    catalog_products = products if products is not None else fetch_products(limit=product_limit)
    retriever = build_retriever(retriever_kind, catalog_products)

    if query_analyzer:
        engine = StructuredSearchEngine(retriever=retriever, analyzer=analyzer) if analyzer is not None else StructuredSearchEngine(retriever=retriever)
        analyzer_name = query_analyzer
    else:
        engine = BaselineSearchEngine(retriever=retriever)
        analyzer_name = "none"

    predictions: List[BenchmarkPrediction] = []
    for task in tasks:
        result = engine.search(task.query, products=catalog_products, k=k)
        predictions.append(
            BenchmarkPrediction(
                task_id=task.task_id,
                query=task.query,
                expect=task.expect,
                answer=result.answer,
                picked_titles=[product["title"] for product in result.picked_products],
                trace=result.trace,
            )
        )

    write_predictions(out_path, predictions)

    return BenchmarkRunSummary(
        tasks_path=str(tasks_path),
        predictions_path=str(out_path),
        retriever=retriever_kind,
        query_analyzer=analyzer_name,
        total_tasks=len(predictions),
        product_limit=product_limit,
        k=k,
    )
