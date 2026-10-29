from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Allow direct script execution without requiring PYTHONPATH.
    sys.path.insert(0, str(SRC))

from shopify_ml_demo.catalog import fetch_products, write_catalog_snapshot


def main() -> None:
    # Fetch the live Storefront catalog and save it as a reusable snapshot.
    parser = argparse.ArgumentParser(description="Export the Storefront catalog to a JSON snapshot.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--product-limit", type=int, default=None, help="Default: full catalog")
    args = parser.parse_args()

    products = fetch_products(limit=args.product_limit)
    write_catalog_snapshot(args.out, products)
    print(f"Wrote {len(products)} products to: {args.out}")


if __name__ == "__main__":
    main()
