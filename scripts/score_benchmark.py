from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Allow direct script execution without requiring PYTHONPATH.
    sys.path.insert(0, str(SRC))

from shopify_ml_demo.catalog import fetch_products
from shopify_ml_demo.metrics import write_score_artifacts


def main() -> None:
    # Score prediction rows and emit metrics/failure artifacts against the full catalog.
    parser = argparse.ArgumentParser(description="Score benchmark predictions.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    products = fetch_products(limit=None)
    scorecard = write_score_artifacts(
        results_path=args.results,
        metrics_out_path=args.metrics_out,
        failures_out_path=args.failures_out,
        products=products,
        k=args.k,
    )
    print(json.dumps(scorecard.to_row(), indent=2))


if __name__ == "__main__":
    main()
