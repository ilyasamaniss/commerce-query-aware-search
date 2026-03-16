# Commerce Query-Aware Search

Applied ML/search project for a Pokemon-themed storefront built on the Shopify Storefront API. The repo is structured to show practical search engineering, deterministic commerce correctness, reproducible benchmarking, and clear orchestration boundaries for Tangle and SkyPilot.

## Overview

The system supports two search modes:

- Baseline lexical search: raw query into overlap or BM25 retrieval.
- Query-aware search: Gemma/Ollama analyzes the query, rewrites it for retrieval, proposes candidate entities, and code enforces hard constraints such as price and stock.

Core principle: the LLM interprets intent and query meaning, but runtime code enforces truth from real catalog fields.
If query-aware search is requested and the analyzer is unavailable, the run fails closed instead of silently degrading to a lexical baseline.

## Example Query

Example query:

```text
yellow pokemon in stock between $20 and $30
```

What the system does with it:

- Baseline lexical search matches literal overlap such as `yellow`, `pokemon`, and `stock`, which is simple but can miss relevant products if the catalog uses names like `Pikachu` instead of the broader theme term `pokemon`.
- Query-aware search asks Gemma/Ollama to interpret the request into structured intent, such as theme=`pokemon`, color=`yellow`, in_stock=`true`, and price range=`20..30`.
- Retrieval uses that interpretation to expand or rewrite the query for better candidate recall.
- Final filtering is deterministic: only products that are actually in stock and whose real catalog price falls between `$20` and `$30` are allowed through.

That example captures the main design goal of the project: let the model interpret the user request, but keep catalog truth and hard constraints in code.

## Architecture

- [catalog.py](src/shopify_ml_demo/catalog.py): Shopify Storefront API client and catalog normalization.
- [query_analysis.py](src/shopify_ml_demo/query_analysis.py): local Ollama/Gemma query interpretation.
- [retrieval.py](src/shopify_ml_demo/retrieval.py): overlap and BM25 retrieval.
- [filtering.py](src/shopify_ml_demo/filtering.py): deterministic hard-constraint enforcement.
- [search.py](src/shopify_ml_demo/search.py): baseline and query-aware search engines.
- [evaluation.py](src/shopify_ml_demo/evaluation.py): benchmark task loading and prediction generation.
- [metrics.py](src/shopify_ml_demo/metrics.py): scoring logic and failure extraction.
- [comparison.py](src/shopify_ml_demo/comparison.py): run comparison JSON and Markdown reporting.

## Repo Structure

```text
benchmarks/
  tasks/
    smoke.jsonl
    full.jsonl
artifacts/
  smoke/
  full/
orchestration/
  skypilot/
    sky_smoke_benchmark.yaml
    sky_full_benchmark.yaml
    sky_smoke_eval.yaml
    sky_full_eval.yaml
    sky_devbox.yaml
  tangle/
    components/
    pipelines/
scripts/
  run_search_query.py
  run_search_benchmark.py
  score_benchmark.py
  compare_runs.py
src/shopify_ml_demo/
tests/
  unit/
  integration/
```

## Local Setup

Create a virtual environment, install dependencies, and configure environment variables:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Required environment variables:

- `SHOPIFY_STORE_DOMAIN`
- `SHOPIFY_STOREFRONT_TOKEN`

Optional query-analysis variables:

- `OLLAMA_URL`
- `OLLAMA_MODEL`

## Local Usage

Single-query debug:

```bash
.venv/bin/python scripts/run_search_query.py "pikchu lego" --retriever bm25 --query-analyzer gemma3:4b --show-trace
```

Smoke benchmark predictions only:

```bash
.venv/bin/python scripts/run_search_benchmark.py \
  --tasks benchmarks/tasks/smoke.jsonl \
  --retriever bm25 \
  --query-analyzer gemma3:4b \
  --out artifacts/smoke/retriever=bm25__analyzer=gemma3_4b/predictions.jsonl
```

Score benchmark predictions:

```bash
.venv/bin/python scripts/score_benchmark.py \
  --results artifacts/smoke/retriever=bm25__analyzer=gemma3_4b/predictions.jsonl \
  --metrics-out artifacts/smoke/retriever=bm25__analyzer=gemma3_4b/metrics.json \
  --failures-out artifacts/smoke/retriever=bm25__analyzer=gemma3_4b/failures.jsonl
```

Compare two runs:

```bash
.venv/bin/python scripts/compare_runs.py \
  --baseline-metrics artifacts/smoke/retriever=bm25__analyzer=none/metrics.json \
  --candidate-metrics artifacts/smoke/retriever=bm25__analyzer=gemma3_4b/metrics.json \
  --out-json artifacts/smoke/comparison.json \
  --out-md artifacts/smoke/comparison.md \
  --baseline-label bm25__analyzer=none \
  --candidate-label bm25__analyzer=gemma3_4b
```

## Benchmark Design

The benchmark suite is split into:

- `smoke.jsonl`: small, fast validation set for development loops and orchestration smoke tests.
- `full.jsonl`: broader benchmark for stronger evaluation and regression tracking.

Each prediction run writes a dedicated artifact directory keyed by orthogonal experiment dimensions:

- `retriever=<name>`
- `analyzer=<name or none>`

Example:

- `artifacts/smoke/retriever=bm25__analyzer=none/`
- `artifacts/smoke/retriever=bm25__analyzer=gemma3_4b/`

Each run directory contains:

- `predictions.jsonl`
- `metrics.json`
- `failures.jsonl`

## Tangle

Tangle is used as a workflow orchestrator over stable CLI boundaries, not as an internal search runtime. Components in [orchestration/tangle/components](orchestration/tangle/components) wrap:

- benchmark execution
- scoring
- run comparison

The smoke pipeline in [pipeline.yaml](orchestration/tangle/pipelines/search_experiment_smoke/pipeline.yaml) expresses the experiment flow:

1. baseline benchmark
2. baseline scoring
3. query-aware benchmark
4. query-aware scoring
5. comparison artifact generation

For local Tangle execution, build the local component image first:

```bash
docker build -t commerce-query-aware-search:tangle-local .
```

And keep Docker running before starting the Tangle backend. If you want the structured branch to execute inside Tangle, run Ollama on the host as well:

```bash
ollama serve
ollama pull gemma3:4b
```

## SkyPilot

SkyPilot workloads live in [orchestration/skypilot](orchestration/skypilot):

- `sky_smoke_benchmark.yaml`: benchmark-only smoke workload for Tangle/SkyPilot integration
- `sky_full_benchmark.yaml`: benchmark-only full workload for Tangle/SkyPilot integration
- `sky_smoke_eval.yaml`: smoke benchmark batch run
- `sky_full_eval.yaml`: full benchmark batch run
- `sky_devbox.yaml`: interactive environment for debugging

SkyPilot configs own execution environment concerns only:

- package installation
- Ollama installation/startup
- env/secrets wiring
- which CLI entrypoint is executed

They do not contain retrieval or scoring logic.
The main working orchestration path is the local Tangle DAG. SkyPilot workloads remain in the repo as separate reproducible batch workloads.

## Limitations

- Query analysis currently relies on a local Ollama-compatible API; query-aware runs fail closed if the analyzer is unavailable.
- Constraint enforcement is intentionally limited to catalog-truth fields like price and stock.
- Entity expansion improves recall for some semantic queries but is still heuristic.

## Best Next ML Step

Add a learned reranking stage or a typed entity linker over the fixed product catalog, then measure gains against the same benchmark and artifact structure instead of changing multiple parts of the pipeline at once.
