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

from shopify_ml_demo.catalog import fetch_products
from shopify_ml_demo.evaluation import normalize_query_analyzer
from shopify_ml_demo.retrieval import build_retriever
from shopify_ml_demo.search import BaselineSearchEngine, StructuredSearchEngine


def main() -> None:
    # Run a single query for local debugging or ad hoc usage.
    parser = argparse.ArgumentParser(description="Run a single ecommerce search query.")
    parser.add_argument("query", type=str, help="Natural-language shopper query")
    parser.add_argument("--retriever", choices=("overlap", "bm25"), default="bm25")
    parser.add_argument(
        "--query-analyzer",
        "--analyzer",
        dest="query_analyzer",
        type=str,
        default=None,
        help="Enable query-aware search with analyzer model (e.g., gemma3:4b)",
    )
    parser.add_argument("--product-limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--show-trace", action="store_true", help="Print tool trace after the answer")
    args = parser.parse_args()

    query_analyzer = normalize_query_analyzer(args.query_analyzer)
    if query_analyzer:
        os.environ["OLLAMA_MODEL"] = query_analyzer

    products = fetch_products(limit=args.product_limit)
    retriever = build_retriever(args.retriever, products)
    if query_analyzer:
        engine = StructuredSearchEngine(retriever=retriever)
    else:
        engine = BaselineSearchEngine(retriever=retriever)

    try:
        result = engine.search(args.query, products=products, k=args.k)
    except Exception as exc:
        raise SystemExit(f"Search failed: {exc}") from exc
    print(result.answer)
    if args.show_trace:
        print("\nTrace:")
        for step in result.trace:
            print(step)


if __name__ == "__main__":
    main()
