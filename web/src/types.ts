export type RetrieverKind = "overlap" | "bm25" | "dense" | "hybrid";

export interface Product {
  title: string;
  handle: string;
  price: number | null;
  currency: string | null;
  available: boolean;
  imageUrl: string | null;
  tags: string[];
}

export interface TraceStep {
  tool: string;
  [key: string]: unknown;
}

export interface SearchResponse {
  query: string;
  retriever: RetrieverKind;
  analyzer: string;
  products: Product[];
  trace: TraceStep[];
}

export interface EngineConfig {
  retriever: RetrieverKind;
  /** Ollama model name for query analysis, or "none" for the lexical baseline. */
  analyzer: string;
}

export interface QueryAnalysisSummary {
  intent: string;
  queryRewrite: string;
  candidateEntities: string[];
  constraints: Record<string, unknown>;
}
