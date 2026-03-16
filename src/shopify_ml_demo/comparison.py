from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from shopify_ml_demo.schemas import ComparisonSummary


def load_metrics(path: Path) -> Dict[str, Any]:
    # Load metrics JSON from disk.
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_metrics(
    *,
    baseline_metrics_path: Path,
    candidate_metrics_path: Path,
    baseline_label: str,
    candidate_label: str,
) -> ComparisonSummary:
    # Compute a compact comparison summary across two metrics files.
    baseline = load_metrics(baseline_metrics_path)
    candidate = load_metrics(candidate_metrics_path)

    return ComparisonSummary(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_metrics_path=str(baseline_metrics_path),
        candidate_metrics_path=str(candidate_metrics_path),
        pass_rate_delta=float(candidate["pass_rate"]) - float(baseline["pass_rate"]),
        mrr_at_k_delta=float(candidate["mrr_at_k"]) - float(baseline["mrr_at_k"]),
        constraint_failures_delta=int(candidate["constraint_failures"]) - int(baseline["constraint_failures"]),
    )


def write_comparison_json(path: Path, summary: ComparisonSummary) -> None:
    # Persist structured comparison summary as JSON.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_row(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def render_comparison_markdown(
    *,
    summary: ComparisonSummary,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
) -> str:
    # Build a concise human-readable comparison report.
    return "\n".join(
        [
            "# Benchmark Comparison",
            "",
            f"- Baseline: `{summary.baseline_label}`",
            f"- Candidate: `{summary.candidate_label}`",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Pass rate | {baseline_metrics['pass_rate']:.3f} | {candidate_metrics['pass_rate']:.3f} | {summary.pass_rate_delta:+.3f} |",
            f"| MRR@{candidate_metrics['k']} | {baseline_metrics['mrr_at_k']:.3f} | {candidate_metrics['mrr_at_k']:.3f} | {summary.mrr_at_k_delta:+.3f} |",
            f"| Constraint failures | {baseline_metrics['constraint_failures']} | {candidate_metrics['constraint_failures']} | {summary.constraint_failures_delta:+d} |",
        ]
    )


def write_comparison_markdown(
    *,
    path: Path,
    summary: ComparisonSummary,
    baseline_metrics_path: Path,
    candidate_metrics_path: Path,
) -> None:
    # Persist comparison report as Markdown.
    baseline_metrics = load_metrics(baseline_metrics_path)
    candidate_metrics = load_metrics(candidate_metrics_path)
    content = render_comparison_markdown(
        summary=summary,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content + "\n")
