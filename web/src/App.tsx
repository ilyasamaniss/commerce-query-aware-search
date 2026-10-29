import { useState, type FormEvent } from "react";
import { search } from "./api";
import { rankChanges } from "./compare";
import { EngineControls } from "./components/EngineControls";
import { ResultsPanel } from "./components/ResultsPanel";
import type { EngineConfig, SearchResponse } from "./types";

const EXAMPLES = [
  "pikchu lego",
  "yellow pokemon in stock between $20 and $30",
  "blue jackal pokemon lego",
  "charzard",
  "sleepy big pokemon lego",
];

interface PaneState {
  response: SearchResponse | null;
  error: string | null;
}

const EMPTY: PaneState = { response: null, error: null };

interface Props {
  searchFn?: typeof search;
}

export function App({ searchFn = search }: Props) {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [k, setK] = useState(5);
  const [left, setLeft] = useState<EngineConfig>({ retriever: "bm25", analyzer: "none" });
  const [right, setRight] = useState<EngineConfig>({ retriever: "hybrid", analyzer: "none" });
  const [results, setResults] = useState<[PaneState, PaneState]>([EMPTY, EMPTY]);
  const [loading, setLoading] = useState(false);

  async function run(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    const settled = await Promise.allSettled([searchFn(q, left, k), searchFn(q, right, k)]);
    setResults(
      settled.map((s) =>
        s.status === "fulfilled" ? { response: s.value, error: null } : { response: null, error: String(s.reason?.message ?? s.reason) },
      ) as [PaneState, PaneState],
    );
    setLoading(false);
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void run(query);
  }

  const [a, b] = results;
  const changes = a.response && b.response ? rankChanges(a.response.products, b.response.products) : undefined;

  return (
    <main>
      <header>
        <h1>Query-Aware Commerce Search</h1>
        <p className="muted">
          Compare two search configurations side by side. The LLM interprets the query; price and stock are enforced in code.
        </p>
      </header>

      <form onSubmit={onSubmit} className="search-form">
        <label htmlFor="query" className="visually-hidden">
          Search query
        </label>
        <input id="query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search the catalog" />
        <label htmlFor="k">Top</label>
        <select id="k" value={k} onChange={(e) => setK(Number(e.target.value))}>
          {[3, 5, 10].map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <button type="submit" disabled={loading}>
          Search
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            className="chip"
            onClick={() => {
              setQuery(ex);
              void run(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="panes">
        <div>
          <EngineControls id="left" config={left} onChange={setLeft} />
          <ResultsPanel title="Baseline" response={a.response} error={a.error} loading={loading} />
        </div>
        <div>
          <EngineControls id="right" config={right} onChange={setRight} />
          <ResultsPanel title="Candidate" response={b.response} error={b.error} loading={loading} changes={changes} />
        </div>
      </div>
    </main>
  );
}
