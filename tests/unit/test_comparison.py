from __future__ import annotations

import json
from pathlib import Path

from shopify_ml_demo.comparison import compare_metrics


def test_compare_metrics_computes_deltas(tmp_path: Path) -> None:
    # Comparison should capture deltas between two scored runs.
    baseline_path = tmp_path / "baseline_metrics.json"
    candidate_path = tmp_path / "candidate_metrics.json"

    baseline_path.write_text(
        json.dumps({"pass_rate": 0.4, "mrr_at_k": 0.5, "constraint_failures": 2, "k": 5}),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"pass_rate": 0.7, "mrr_at_k": 0.8, "constraint_failures": 1, "k": 5}),
        encoding="utf-8",
    )

    summary = compare_metrics(
        baseline_metrics_path=baseline_path,
        candidate_metrics_path=candidate_path,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert round(summary.pass_rate_delta, 3) == 0.3
    assert round(summary.mrr_at_k_delta, 3) == 0.3
    assert summary.constraint_failures_delta == -1
