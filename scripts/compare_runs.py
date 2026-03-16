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

from shopify_ml_demo.comparison import compare_metrics, write_comparison_json, write_comparison_markdown


def main() -> None:
    # Compare two metrics artifacts and write structured + Markdown reports.
    parser = argparse.ArgumentParser(description="Compare two benchmark metric artifacts.")
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--candidate-label", type=str, default="candidate")
    args = parser.parse_args()

    summary = compare_metrics(
        baseline_metrics_path=args.baseline_metrics,
        candidate_metrics_path=args.candidate_metrics,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    write_comparison_json(args.out_json, summary)
    write_comparison_markdown(
        path=args.out_md,
        summary=summary,
        baseline_metrics_path=args.baseline_metrics,
        candidate_metrics_path=args.candidate_metrics,
    )
    print(json.dumps(summary.to_row(), indent=2))


if __name__ == "__main__":
    main()
