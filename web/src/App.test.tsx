import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "./App";
import type { EngineConfig, Product, SearchResponse } from "./types";

const product = (title: string): Product => ({
  title,
  handle: title.toLowerCase().replace(/\s+/g, "-"),
  price: 24.99,
  currency: "USD",
  available: true,
  imageUrl: null,
  tags: [],
});

function fakeSearch(byRetriever: Record<string, string[]>) {
  return vi.fn(async (query: string, config: EngineConfig): Promise<SearchResponse> => {
    if (!(config.retriever in byRetriever)) throw new Error(`${config.retriever} unavailable`);
    return {
      query,
      retriever: config.retriever,
      analyzer: config.analyzer,
      products: byRetriever[config.retriever].map(product),
      trace: [],
    };
  });
}

it("searches both configurations and shows rank changes", async () => {
  const searchFn = fakeSearch({ bm25: ["Mewtwo", "Pikachu"], hybrid: ["Pikachu", "Eevee"] });
  render(<App searchFn={searchFn} />);

  await userEvent.click(screen.getByRole("button", { name: "Search" }));

  expect(searchFn).toHaveBeenCalledTimes(2);
  expect(searchFn.mock.calls.map((c) => c[1].retriever)).toEqual(["bm25", "hybrid"]);
  const candidate = await screen.findByRole("region", { name: "Candidate" });
  expect(within(candidate).getByText("Pikachu")).toBeInTheDocument();
  expect(within(candidate).getByText("▲ 1")).toBeInTheDocument();
  expect(within(candidate).getByText("new")).toBeInTheDocument();
});

it("shows an error in one pane without hiding the other", async () => {
  const searchFn = fakeSearch({ bm25: ["Pikachu"] });
  render(<App searchFn={searchFn} />);

  await userEvent.click(screen.getByRole("button", { name: "charzard" }));

  const candidate = await screen.findByRole("region", { name: "Candidate" });
  expect(within(candidate).getByRole("alert")).toHaveTextContent("hybrid unavailable");
  const baseline = screen.getByRole("region", { name: "Baseline" });
  expect(within(baseline).getByText("Pikachu")).toBeInTheDocument();
  expect(searchFn.mock.calls[0][0]).toBe("charzard");
});

it("turns on query analysis for a pane", async () => {
  const searchFn = fakeSearch({ bm25: [], hybrid: [] });
  render(<App searchFn={searchFn} />);

  const toggles = screen.getAllByRole("checkbox", { name: /query analysis/i });
  await userEvent.click(toggles[1]);
  await userEvent.click(screen.getByRole("button", { name: "Search" }));

  expect(searchFn.mock.calls[1][1]).toEqual({ retriever: "hybrid", analyzer: "gemma3:4b" });
  expect(await screen.findAllByText("No matching products.")).toHaveLength(2);
});
