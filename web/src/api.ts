import type { EngineConfig, SearchResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export async function search(
  query: string,
  config: EngineConfig,
  k: number,
  fetchImpl: typeof fetch = fetch,
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query, retriever: config.retriever, analyzer: config.analyzer, k: String(k) });
  const resp = await fetchImpl(`${API_BASE}/api/search?${params.toString()}`);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new ApiError(typeof body.error === "string" ? body.error : `HTTP ${resp.status}`, resp.status);
  }
  return body as SearchResponse;
}
