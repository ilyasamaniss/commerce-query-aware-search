import type { Product, QueryAnalysisSummary, TraceStep } from "./types";

export type RankChange = { kind: "new" } | { kind: "same" } | { kind: "up" | "down"; by: number };

/** How a product's rank in `candidate` compares with its rank in `baseline` (matched by handle). */
export function rankChanges(baseline: Product[], candidate: Product[]): RankChange[] {
  const baselineRank = new Map(baseline.map((p, i) => [p.handle, i]));
  return candidate.map((p, i) => {
    const before = baselineRank.get(p.handle);
    if (before === undefined) return { kind: "new" };
    if (before === i) return { kind: "same" };
    return before > i ? { kind: "up", by: before - i } : { kind: "down", by: i - before };
  });
}

/** Pull the structured query analysis out of a search trace, if the engine ran one. */
export function analysisFromTrace(trace: TraceStep[]): QueryAnalysisSummary | null {
  const step = trace.find((s) => s.tool === "query_analysis" && typeof s.result === "object" && s.result !== null);
  if (!step) return null;
  const result = step.result as Record<string, unknown>;
  return {
    intent: String(result.intent ?? ""),
    queryRewrite: String(result.query_rewrite ?? ""),
    candidateEntities: Array.isArray(result.candidate_entities) ? result.candidate_entities.map(String) : [],
    constraints: (result.constraints as Record<string, unknown>) ?? {},
  };
}

/** Candidate counts before and after deterministic filtering, if filtering ran. */
export function filterCounts(trace: TraceStep[]): { input: number; output: number } | null {
  const step = trace.find((s) => s.tool === "deterministic_filter");
  if (!step) return null;
  return { input: Number(step.input_count), output: Number(step.result_count) };
}

export function formatPrice(product: Product): string {
  if (product.price === null) return "—";
  return `$${product.price.toFixed(2)}${product.currency ? ` ${product.currency}` : ""}`;
}
