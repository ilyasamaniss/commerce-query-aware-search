# Tangle Layout

This repository treats Tangle as an experiment orchestrator over file-based CLIs.

Components in `components/` wrap thin scripts:
- `run_search_benchmark`
- `score_benchmark`
- `compare_runs`

For local execution, Tangle runs the benchmark, scoring, and comparison steps in a local Docker image built from this repo.
This keeps the DAG working end to end without requiring SkyPilot to be in Tangle's execution path.

Local prerequisites:
- Docker daemon must be running
- build the local Tangle image first with `docker build -t commerce-query-aware-search:tangle-local .`
- if you want the structured branch to work, Ollama must be running on the host and reachable as `http://host.docker.internal:11434`

Pipelines in `pipelines/` compose those components over the smoke benchmark to compare:
- baseline lexical retrieval
- query-aware retrieval with Gemma/Ollama

The intended contract is simple:
- inputs arrive as files and CLI flags
- outputs are written as artifacts on disk
- core search logic stays in `src/shopify_ml_demo/`
