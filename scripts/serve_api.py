from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Allow direct script execution without requiring PYTHONPATH.
    sys.path.insert(0, str(SRC))

from shopify_ml_demo.api import SearchService, serve
from shopify_ml_demo.catalog import load_products
from shopify_ml_demo.embeddings import build_embedder


def main() -> None:
    # Serve the search engines over a small JSON API for the web demo.
    parser = argparse.ArgumentParser(description="Serve the search JSON API used by the web demo.")
    parser.add_argument("--catalog", type=Path, default=None, help="Catalog snapshot JSON; default is the live Storefront API")
    parser.add_argument("--product-limit", type=int, default=200)
    parser.add_argument("--embedder", type=str, default="hashing", help="'hashing' or 'ollama:<model>'")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    products = load_products(args.catalog, limit=args.product_limit)
    service = SearchService(products=products, embedder_factory=lambda: build_embedder(args.embedder))
    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
