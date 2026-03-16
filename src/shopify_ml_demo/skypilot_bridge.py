from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

from shopify_ml_demo.evaluation import normalize_query_analyzer


def resolve_repo_relative_path(path: Path | str, *, repo_root: Path) -> str:
    # Convert a repo-local path into the relative path SkyPilot will see under workdir sync.
    resolved = (repo_root / path).resolve() if isinstance(path, str) else path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Path must live inside the repo workdir for SkyPilot sync: {resolved}") from exc


def generate_cluster_name(prefix: str) -> str:
    # Generate a run-specific cluster name to avoid stale-cluster collisions across repeated runs.
    sanitized = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in prefix.strip().lower()).strip("-")
    if not sanitized:
        raise ValueError("Cluster name prefix must contain at least one alphanumeric character.")
    return f"{sanitized}-{uuid.uuid4().hex[:8]}"


def _detect_sky_binary() -> str:
    # Prefer the active environment's sky CLI and fall back to PATH lookup.
    candidates = [
        Path(sys_executable).resolve().parent / "sky"
        for sys_executable in [shutil.which("python3"), shutil.which("python")]
        if sys_executable is not None
    ]
    candidates.extend(
        Path(path)
        for path in [
            shutil.which("sky"),
            str(Path(".venv/bin/sky").resolve()) if Path(".venv/bin/sky").exists() else None,
        ]
        if path is not None
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Could not find SkyPilot CLI. Install it or activate the environment that provides `sky`.")


def build_launch_command(
    *,
    sky_binary: str,
    sky_yaml: Path,
    cluster_name: str,
    tasks_repo_path: str,
    retriever: str,
    query_analyzer: Optional[str],
    remote_predictions_path: str,
    repo_root: Path,
    product_limit: int,
    k: int,
) -> List[str]:
    # Build the SkyPilot launch command with explicit env/secret overrides.
    normalized_analyzer = normalize_query_analyzer(query_analyzer)
    query_analyzer_value = normalized_analyzer or "none"

    command = [
        sky_binary,
        "launch",
        "-c",
        cluster_name,
        str(sky_yaml),
        "-y",
        "--env",
        f"TASKS_PATH={resolve_repo_relative_path(tasks_repo_path, repo_root=repo_root)}",
        "--env",
        f"RETRIEVER={retriever}",
        "--env",
        f"QUERY_ANALYZER={query_analyzer_value}",
        "--env",
        f"PREDICTIONS_OUT={resolve_repo_relative_path(remote_predictions_path, repo_root=repo_root)}",
        "--env",
        f"PRODUCT_LIMIT={product_limit}",
        "--env",
        f"K={k}",
    ]

    if normalized_analyzer:
        command.extend(["--env", f"OLLAMA_MODEL={normalized_analyzer}"])

    if os.getenv("SHOPIFY_STORE_DOMAIN"):
        command.extend(["--env", "SHOPIFY_STORE_DOMAIN"])
    if os.getenv("SHOPIFY_STOREFRONT_TOKEN"):
        command.extend(["--secret", "SHOPIFY_STOREFRONT_TOKEN"])

    return command


def build_artifact_copy_command(
    *,
    cluster_name: str,
    remote_workdir: str,
    remote_predictions_path: str,
    local_out: Path,
    repo_root: Path,
) -> List[str]:
    # Copy a generated artifact back from the SkyPilot cluster using the Sky-managed SSH alias.
    remote_relative = resolve_repo_relative_path(remote_predictions_path, repo_root=repo_root)
    remote_path = f"{cluster_name}:{remote_workdir.rstrip('/')}/{remote_relative}"
    return ["scp", remote_path, str(local_out)]


def run_skypilot_benchmark(
    *,
    sky_yaml: Path,
    cluster_name_prefix: str,
    tasks_repo_path: str,
    retriever: str,
    query_analyzer: Optional[str],
    remote_predictions_path: str,
    local_out: Path,
    repo_root: Path,
    product_limit: int,
    k: int,
    remote_workdir: str = "~/sky_workdir",
) -> None:
    # Launch the SkyPilot workload and pull the predictions artifact back locally.
    sky_binary = _detect_sky_binary()
    cluster_name = generate_cluster_name(cluster_name_prefix)
    launch_command = build_launch_command(
        sky_binary=sky_binary,
        sky_yaml=sky_yaml,
        cluster_name=cluster_name,
        tasks_repo_path=tasks_repo_path,
        retriever=retriever,
        query_analyzer=query_analyzer,
        remote_predictions_path=remote_predictions_path,
        repo_root=repo_root,
        product_limit=product_limit,
        k=k,
    )
    subprocess.run(launch_command, check=True)

    # Refresh local SSH config before copying the artifact via the cluster alias.
    subprocess.run([sky_binary, "status", cluster_name], check=True, stdout=subprocess.DEVNULL)

    local_out.parent.mkdir(parents=True, exist_ok=True)
    copy_command = build_artifact_copy_command(
        cluster_name=cluster_name,
        remote_workdir=remote_workdir,
        remote_predictions_path=remote_predictions_path,
        local_out=local_out,
        repo_root=repo_root,
    )
    subprocess.run(copy_command, check=True)
