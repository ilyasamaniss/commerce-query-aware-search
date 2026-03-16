from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Allow direct script execution without requiring PYTHONPATH.
    sys.path.insert(0, str(SRC))

from shopify_ml_demo.evaluation import normalize_query_analyzer, run_benchmark


def main() -> None:
    # Execute benchmark tasks and write predictions only.
    parser = argparse.ArgumentParser(description="Run a benchmark suite and write prediction rows.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retriever", choices=("overlap", "bm25"), default="bm25")
    parser.add_argument(
        "--query-analyzer",
        "--analyzer",
        dest="query_analyzer",
        type=str,
        default=None,
        help="Enable query-aware benchmark with analyzer model (e.g., gemma3:4b)",
    )
    parser.add_argument("--product-limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    query_analyzer = normalize_query_analyzer(args.query_analyzer)
    if query_analyzer:
        os.environ["OLLAMA_MODEL"] = query_analyzer

    try:
        summary = run_benchmark(
            tasks_path=args.tasks,
            out_path=args.out,
            retriever_kind=args.retriever,
            query_analyzer=query_analyzer,
            product_limit=args.product_limit,
            k=args.k,
        )
    except Exception as exc:
        raise SystemExit(f"Benchmark failed: {exc}") from exc
    print(f"Wrote {summary.total_tasks} predictions to: {args.out}")


if __name__ == "__main__":
    main()
