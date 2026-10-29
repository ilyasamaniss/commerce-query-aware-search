import { analysisFromTrace, filterCounts, formatPrice, rankChanges } from "./compare";
import type { Product } from "./types";

const p = (handle: string, extra: Partial<Product> = {}): Product => ({
  title: handle,
  handle,
  price: 10,
  currency: "USD",
  available: true,
  imageUrl: null,
  tags: [],
  ...extra,
});

describe("rankChanges", () => {
  it("marks moved, unchanged, and new products", () => {
    const baseline = [p("a"), p("b"), p("c")];
    const candidate = [p("c"), p("b"), p("d")];
    expect(rankChanges(baseline, candidate)).toEqual([{ kind: "up", by: 2 }, { kind: "same" }, { kind: "new" }]);
  });

  it("reports drops", () => {
    expect(rankChanges([p("a"), p("b")], [p("b"), p("a")])).toEqual([
      { kind: "up", by: 1 },
      { kind: "down", by: 1 },
    ]);
  });
});

describe("trace helpers", () => {
  const trace = [
    { tool: "query_analysis", args: { query: "x" } },
    {
      tool: "query_analysis",
      result: {
        intent: "search",
        query_rewrite: "pikachu lego",
        candidate_entities: ["pikachu"],
        constraints: { in_stock: true, price_max: null },
      },
    },
    { tool: "deterministic_filter", input_count: 25, result_count: 9 },
  ];

  it("extracts the query analysis", () => {
    expect(analysisFromTrace(trace)).toEqual({
      intent: "search",
      queryRewrite: "pikachu lego",
      candidateEntities: ["pikachu"],
      constraints: { in_stock: true, price_max: null },
    });
  });

  it("returns null for baseline traces", () => {
    expect(analysisFromTrace([{ tool: "retriever.search" }])).toBeNull();
    expect(filterCounts([{ tool: "retriever.search" }])).toBeNull();
  });

  it("extracts filter counts", () => {
    expect(filterCounts(trace)).toEqual({ input: 25, output: 9 });
  });
});

it("formats prices", () => {
  expect(formatPrice(p("a", { price: 24.5 }))).toBe("$24.50 USD");
  expect(formatPrice(p("a", { price: null }))).toBe("—");
});
