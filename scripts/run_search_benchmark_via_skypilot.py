from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Allow direct execution without requiring PYTHONPATH.
    sys.path.insert(0, str(SRC))

from shopify_ml_demo.skypilot_bridge import run_skypilot_benchmark


def main() -> None:
    # Launch the benchmark through SkyPilot and sync the predictions artifact back locally.
    parser = argparse.ArgumentParser(description="Run a search benchmark through SkyPilot.")
    parser.add_argument("--tasks-path", type=str, required=True, help="Repo-relative benchmark task path to sync into SkyPilot.")
    parser.add_argument("--retriever", choices=("overlap", "bm25"), required=True)
    parser.add_argument("--query-analyzer", "--analyzer", dest="query_analyzer", type=str, default="none")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-predictions-path", type=str, required=True, help="Repo-relative output path inside the SkyPilot workdir.")
    parser.add_argument("--sky-yaml", type=Path, required=True)
    parser.add_argument("--cluster-name-prefix", type=str, required=True)
    parser.add_argument("--product-limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--remote-workdir", type=str, default="~/sky_workdir")
    args = parser.parse_args()

    try:
        run_skypilot_benchmark(
            sky_yaml=args.sky_yaml,
            cluster_name_prefix=args.cluster_name_prefix,
            tasks_repo_path=args.tasks_path,
            retriever=args.retriever,
            query_analyzer=args.query_analyzer,
            remote_predictions_path=args.remote_predictions_path,
            local_out=args.out,
            repo_root=ROOT,
            product_limit=args.product_limit,
            k=args.k,
            remote_workdir=args.remote_workdir,
        )
    except Exception as exc:
        raise SystemExit(f"SkyPilot benchmark failed: {exc}") from exc

    print(f"Wrote predictions to: {args.out}")


if __name__ == "__main__":
    main()
