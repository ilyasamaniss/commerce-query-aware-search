import { analysisFromTrace, filterCounts, formatPrice, type RankChange } from "../compare";
import type { SearchResponse } from "../types";

interface Props {
  title: string;
  response: SearchResponse | null;
  error: string | null;
  loading: boolean;
  changes?: RankChange[];
}

function RankBadge({ change }: { change: RankChange }) {
  switch (change.kind) {
    case "new":
      return <span className="badge badge-new">new</span>;
    case "up":
      return <span className="badge badge-up">▲ {change.by}</span>;
    case "down":
      return <span className="badge badge-down">▼ {change.by}</span>;
    default:
      return null;
  }
}

export function ResultsPanel({ title, response, error, loading, changes }: Props) {
  const analysis = response ? analysisFromTrace(response.trace) : null;
  const filtered = response ? filterCounts(response.trace) : null;

  return (
    <section className="panel" aria-label={title}>
      <h2>{title}</h2>
      {loading && <p className="muted">Searching…</p>}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {response && !loading && !error && (
        <>
          {analysis && (
            <dl className="analysis">
              <dt>Rewrite</dt>
              <dd>{analysis.queryRewrite}</dd>
              {analysis.candidateEntities.length > 0 && (
                <>
                  <dt>Entities</dt>
                  <dd>{analysis.candidateEntities.join(", ")}</dd>
                </>
              )}
              <dt>Constraints</dt>
              <dd>
                {Object.entries(analysis.constraints)
                  .filter(([, v]) => v !== null && v !== undefined)
                  .map(([k, v]) => `${k}=${String(v)}`)
                  .join(", ") || "none"}
              </dd>
              {filtered && (
                <>
                  <dt>Filter</dt>
                  <dd>
                    {filtered.input} candidates → {filtered.output} pass price/stock checks
                  </dd>
                </>
              )}
            </dl>
          )}
          {response.products.length === 0 ? (
            <p className="muted">No matching products.</p>
          ) : (
            <ol className="results">
              {response.products.map((p, i) => (
                <li key={p.handle}>
                  <span className="product-title">{p.title}</span>
                  {changes && changes[i] && <RankBadge change={changes[i]} />}
                  <span className="product-meta">
                    {formatPrice(p)} · {p.available ? "in stock" : <span className="oos">out of stock</span>}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}
