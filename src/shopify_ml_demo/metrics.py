from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shopify_ml_demo.schemas import Failure, Scorecard


def _index_products_by_title(products: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Build title index for deterministic constraint validation.
    return {str(product.get("title", "")).strip().lower(): product for product in products}


def _reciprocal_rank(picked_titles: List[str], must_include_titles: List[str], k: int) -> float:
    # Compute reciprocal rank of the first expected title within top-k.
    expected = {title.strip().lower() for title in must_include_titles}
    for idx, title in enumerate(picked_titles[:k], start=1):
        if title.strip().lower() in expected:
            return 1.0 / idx
    return 0.0


def _validate_constraints(
    *,
    picked_titles: List[str],
    constraints: Dict[str, Any],
    by_title: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    # Validate result rows against expected hard constraints.
    in_stock = bool(constraints.get("in_stock", False))
    price_min = constraints.get("price_min", None)
    price_max = constraints.get("price_max", None)

    for title in picked_titles:
        key = title.strip().lower()
        product = by_title.get(key)
        if not product:
            return False, f"Returned title not found in catalog: {title}"

        if in_stock and not bool(product.get("availableForSale", False)):
            return False, f"Returned '{product.get('title')}' is not availableForSale."

        amount = float(product["priceRange"]["minVariantPrice"]["amount"])
        if price_min is not None and amount < float(price_min):
            return False, f"Returned '{product.get('title')}' price {amount} < price_min {float(price_min)}."
        if price_max is not None and amount > float(price_max):
            return False, f"Returned '{product.get('title')}' price {amount} > price_max {float(price_max)}."

    return True, None


def score_results_file(results_path: Path, *, products: List[Dict[str, Any]], k: int = 5) -> Scorecard:
    # Score JSONL evaluation results with per-task expectations.
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    by_title = _index_products_by_title(products)

    total = 0
    passed = 0
    failures: List[Failure] = []

    rr_sum = 0.0
    rr_count = 0
    constraint_tasks = 0
    constraint_failures = 0

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            row: Dict[str, Any] = json.loads(line)
            total += 1

            task_id = str(row.get("id", ""))
            query = str(row.get("query", ""))
            expect: Dict[str, Any] = dict(row.get("expect", {}) or {})
            picked_titles: List[str] = list(row.get("picked_titles", []))

            ok = True
            reason = ""

            max_results = expect.get("max_results", None)
            if max_results is not None and len(picked_titles) > int(max_results):
                ok = False
                reason = f"Expected <= {int(max_results)} results, got {len(picked_titles)}."

            min_results = expect.get("min_results", None)
            if ok and min_results is not None and len(picked_titles) < int(min_results):
                ok = False
                reason = f"Expected >= {int(min_results)} results, got {len(picked_titles)}."

            must_include_titles = expect.get("must_include_titles", None)
            top_k = int(expect.get("top_k", k))
            if isinstance(must_include_titles, list) and must_include_titles:
                rr_sum += _reciprocal_rank(picked_titles, must_include_titles, k=min(k, top_k))
                rr_count += 1

                if ok:
                    expected_set = {title.strip().lower() for title in must_include_titles}
                    in_topk = any(title.strip().lower() in expected_set for title in picked_titles[:top_k])
                    if not in_topk:
                        ok = False
                        reason = f"Expected one of {must_include_titles} in top-{top_k}, got top-{top_k}: {picked_titles[:top_k]}."

            constraints = expect.get("constraints", None)
            if isinstance(constraints, dict) and constraints:
                constraint_tasks += 1
                if ok:
                    valid, message = _validate_constraints(
                        picked_titles=picked_titles,
                        constraints=constraints,
                        by_title=by_title,
                    )
                    if not valid:
                        ok = False
                        constraint_failures += 1
                        reason = message or "Constraint validation failed."

            if ok:
                passed += 1
            else:
                failures.append(Failure(task_id=task_id, query=query, reason=reason))

    mrr_at_k = (rr_sum / rr_count) if rr_count else 0.0

    return Scorecard(
        results_path=str(results_path),
        total=total,
        passed=passed,
        failures=failures,
        mrr_at_k=mrr_at_k,
        k=k,
        ranking_tasks=rr_count,
        constraint_tasks=constraint_tasks,
        constraint_failures=constraint_failures,
    )


def write_score_artifacts(
    *,
    results_path: Path,
    metrics_out_path: Path,
    failures_out_path: Optional[Path],
    products: List[Dict[str, Any]],
    k: int = 5,
) -> Scorecard:
    # Score predictions and persist metrics/failures as separate artifacts.
    scorecard = score_results_file(results_path, products=products, k=k)

    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out_path.open("w", encoding="utf-8") as handle:
        json.dump(scorecard.to_row(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if failures_out_path is not None:
        failures_out_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_out_path.open("w", encoding="utf-8") as handle:
            for failure in scorecard.failures:
                handle.write(json.dumps(failure.to_row(), ensure_ascii=False) + "\n")

    return scorecard
