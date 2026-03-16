from __future__ import annotations

from pathlib import Path

from shopify_ml_demo.skypilot_bridge import (
    build_artifact_copy_command,
    build_launch_command,
    generate_cluster_name,
    resolve_repo_relative_path,
)


def test_resolve_repo_relative_path_requires_repo_member(tmp_path: Path) -> None:
    # Repo-relative path resolution should preserve artifact layout under synced workdir.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifact = repo_root / "artifacts" / "smoke" / "predictions.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("", encoding="utf-8")

    relative = resolve_repo_relative_path(artifact, repo_root=repo_root)

    assert relative == "artifacts/smoke/predictions.jsonl"


def test_build_launch_command_normalizes_none_query_analyzer(tmp_path: Path) -> None:
    # Tangle can pass a sentinel analyzer value while SkyPilot receives an explicit env contract.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sky_yaml = repo_root / "orchestration" / "skypilot" / "sky_smoke_eval.yaml"
    sky_yaml.parent.mkdir(parents=True)
    sky_yaml.write_text("name: test\n", encoding="utf-8")
    tasks = repo_root / "benchmarks" / "tasks" / "smoke.jsonl"
    tasks.parent.mkdir(parents=True)
    tasks.write_text("", encoding="utf-8")
    out = repo_root / "artifacts" / "smoke" / "predictions.jsonl"
    out.parent.mkdir(parents=True)

    command = build_launch_command(
        sky_binary="sky",
        sky_yaml=sky_yaml,
        cluster_name="shopify-smoke",
        tasks_repo_path="benchmarks/tasks/smoke.jsonl",
        retriever="bm25",
        query_analyzer="none",
        remote_predictions_path="artifacts/smoke/predictions.jsonl",
        repo_root=repo_root,
        product_limit=200,
        k=3,
    )

    joined = " ".join(command)
    assert "QUERY_ANALYZER=none" in joined
    assert "OLLAMA_MODEL=" not in joined


def test_build_artifact_copy_command_uses_remote_workdir_layout(tmp_path: Path) -> None:
    # Artifact copy should target the synced repo-relative path under the remote workdir.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    out = repo_root / "artifacts" / "smoke" / "predictions.jsonl"
    out.parent.mkdir(parents=True)

    command = build_artifact_copy_command(
        cluster_name="shopify-smoke",
        remote_workdir="~/sky_workdir",
        remote_predictions_path="artifacts/smoke/predictions.jsonl",
        local_out=out,
        repo_root=repo_root,
    )

    assert command[0] == "scp"
    assert command[1] == "shopify-smoke:~/sky_workdir/artifacts/smoke/predictions.jsonl"


def test_generate_cluster_name_adds_unique_suffix() -> None:
    # Generated cluster names should keep the prefix readable while avoiding collisions.
    name = generate_cluster_name("shopify-smoke")

    assert name.startswith("shopify-smoke-")
    assert len(name) > len("shopify-smoke-")
