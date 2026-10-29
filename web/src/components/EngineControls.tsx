import type { EngineConfig, RetrieverKind } from "../types";

const RETRIEVERS: { value: RetrieverKind; label: string }[] = [
  { value: "bm25", label: "BM25" },
  { value: "dense", label: "Dense (embeddings)" },
  { value: "hybrid", label: "Hybrid (BM25 + dense)" },
  { value: "overlap", label: "Token overlap" },
];

interface Props {
  id: string;
  config: EngineConfig;
  onChange: (config: EngineConfig) => void;
}

export function EngineControls({ id, config, onChange }: Props) {
  const queryAware = config.analyzer !== "none";
  return (
    <div className="engine-controls">
      <label htmlFor={`${id}-retriever`}>Retriever</label>
      <select
        id={`${id}-retriever`}
        value={config.retriever}
        onChange={(e) => onChange({ ...config, retriever: e.target.value as RetrieverKind })}
      >
        {RETRIEVERS.map((r) => (
          <option key={r.value} value={r.value}>
            {r.label}
          </option>
        ))}
      </select>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={queryAware}
          onChange={(e) => onChange({ ...config, analyzer: e.target.checked ? "gemma3:4b" : "none" })}
        />
        Query analysis (Gemma)
      </label>
    </div>
  );
}
