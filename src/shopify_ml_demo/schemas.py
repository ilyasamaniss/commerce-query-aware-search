from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class QueryConstraints:
    # Hard constraints extracted from user language.
    price_min: Optional[float]
    price_max: Optional[float]
    in_stock: Optional[bool]
    color: Optional[str]


@dataclass(frozen=True)
class QueryAnalysis:
    # Structured analysis produced by query understanding.
    intent: str
    query_rewrite: str
    constraints: QueryConstraints
    candidate_entities: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    # Shared search result object for baseline and structured engines.
    answer: str
    picked_products: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkTask:
    # One benchmark example loaded from JSONL.
    task_id: str
    query: str
    expect: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        # Serialize task into JSON-friendly dict fields.
        return {
            "id": self.task_id,
            "query": self.query,
            "expect": self.expect,
        }


@dataclass(frozen=True)
class BenchmarkPrediction:
    # One prediction row emitted by a benchmark run.
    task_id: str
    query: str
    expect: Dict[str, Any]
    answer: str
    picked_titles: List[str]
    trace: List[Dict[str, Any]]

    def to_row(self) -> Dict[str, Any]:
        # Serialize prediction into JSONL row format.
        return {
            "id": self.task_id,
            "query": self.query,
            "expect": self.expect,
            "answer": self.answer,
            "picked_titles": self.picked_titles,
            "trace": self.trace,
        }


@dataclass(frozen=True)
class BenchmarkRunSummary:
    # Summary metadata for a completed benchmark run.
    tasks_path: str
    predictions_path: str
    retriever: str
    query_analyzer: str
    total_tasks: int
    product_limit: int
    k: int

    def to_row(self) -> Dict[str, Any]:
        # Serialize run summary for manifests or reports.
        return asdict(self)


@dataclass(frozen=True)
class Failure:
    # Single benchmark failure row with deterministic reason.
    task_id: str
    query: str
    reason: str

    def to_row(self) -> Dict[str, Any]:
        # Serialize failure for JSONL emission.
        return asdict(self)


@dataclass(frozen=True)
class Scorecard:
    # Aggregate metrics over an evaluation run.
    results_path: str
    total: int
    passed: int
    failures: List[Failure]
    mrr_at_k: float
    k: int
    ranking_tasks: int
    constraint_tasks: int
    constraint_failures: int

    @property
    def pass_rate(self) -> float:
        # Compute pass rate while handling empty task suites.
        return (self.passed / self.total) if self.total else 0.0

    def to_row(self) -> Dict[str, Any]:
        # Serialize scorecard to a JSON-friendly dictionary.
        return {
            "results_path": self.results_path,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "mrr_at_k": self.mrr_at_k,
            "k": self.k,
            "ranking_tasks": self.ranking_tasks,
            "constraint_tasks": self.constraint_tasks,
            "constraint_failures": self.constraint_failures,
        }


@dataclass(frozen=True)
class ComparisonSummary:
    # High-level comparison between two scored runs.
    baseline_label: str
    candidate_label: str
    baseline_metrics_path: str
    candidate_metrics_path: str
    pass_rate_delta: float
    mrr_at_k_delta: float
    constraint_failures_delta: int

    def to_row(self) -> Dict[str, Any]:
        # Serialize comparison summary for JSON output.
        return asdict(self)
