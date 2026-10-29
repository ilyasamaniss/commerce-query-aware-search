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

from shopify_ml_demo.catalog import load_catalog_snapshot
from shopify_ml_demo.evaluation import normalize_query_analyzer, run_benchmark
from shopify_ml_demo.retrieval import RETRIEVER_KINDS


def main() -> None:
    # Execute benchmark tasks and write predictions only.
    parser = argparse.ArgumentParser(description="Run a benchmark suite and write prediction rows.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retriever", choices=RETRIEVER_KINDS, default="bm25")
    parser.add_argument(
        "--embedder",
        type=str,
        default="hashing",
        help="Embedder for dense/hybrid retrieval: 'hashing' (offline) or 'ollama:<model>' (e.g. ollama:nomic-embed-text)",
    )
    parser.add_argument("--fusion", choices=("convex", "rrf"), default="convex", help="Hybrid score fusion method")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense weight for convex hybrid fusion")
    parser.add_argument(
        "--query-analyzer",
        "--analyzer",
        dest="query_analyzer",
        type=str,
        default=None,
        help="Enable query-aware benchmark with analyzer model (e.g., gemma3:4b)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Read products from a catalog snapshot JSON instead of the live Storefront API",
    )
    parser.add_argument("--product-limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    query_analyzer = normalize_query_analyzer(args.query_analyzer)
    if query_analyzer:
        os.environ["OLLAMA_MODEL"] = query_analyzer

    products = None
    if args.catalog is not None:
        products = load_catalog_snapshot(args.catalog)[: args.product_limit]

    try:
        summary = run_benchmark(
            tasks_path=args.tasks,
            out_path=args.out,
            retriever_kind=args.retriever,
            query_analyzer=query_analyzer,
            product_limit=args.product_limit,
            k=args.k,
            products=products,
            embedder=args.embedder,
            fusion=args.fusion,
            alpha=args.alpha,
        )
    except Exception as exc:
        raise SystemExit(f"Benchmark failed: {exc}") from exc
    print(f"Wrote {summary.total_tasks} predictions to: {args.out}")


if __name__ == "__main__":
    main()
