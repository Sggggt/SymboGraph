from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from subprocess_env import sanitized_subprocess_env


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _executable_scripts() -> list[Path]:
    result: list[Path] = []
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.If)
            and "__name__" in ast.unparse(node.test)
            and "__main__" in ast.unparse(node.test)
            for node in ast.walk(tree)
        ):
            result.append(path)
    return result


EXECUTABLE_SCRIPTS = _executable_scripts()


def test_every_executable_script_is_documented() -> None:
    readme = (SCRIPTS_ROOT / "README.md").read_text(encoding="utf-8")
    undocumented = [
        path.name for path in EXECUTABLE_SCRIPTS if f"`{path.name}`" not in readme
    ]
    assert undocumented == []


@pytest.mark.parametrize("script_path", EXECUTABLE_SCRIPTS, ids=lambda path: path.name)
def test_every_script_help_is_dependency_failure_safe(script_path: Path) -> None:
    env = sanitized_subprocess_env(
        {
            "DATABASE_URL": (
                "postgresql+psycopg://invalid:invalid@127.0.0.1:1/unavailable"
            ),
            "REDIS_URL": "redis://127.0.0.1:1/15",
            "QDRANT_URL": "http://127.0.0.1:1",
            "ENABLE_DATABASE_FALLBACK": "false",
            "ENABLE_MODEL_FALLBACK": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{script_path.name}\nstdout={completed.stdout}\nstderr={completed.stderr}"
    )
    assert "usage:" in completed.stdout.lower()
    assert "Primary database is unavailable" not in completed.stderr


def test_write_capable_scripts_expose_explicit_execution_gate() -> None:
    write_capable = {
        "cleanup_stale_data.py",
        "cleanup_vector_collection.py",
        "docker_smoke.py",
        "manage_runtime_settings_candidate.py",
        "manage_vector_shadow.py",
        "rebuild_chunk_relation_graph.py",
        "rebuild_chunks.py",
        "rebuild_coarse_concept_graph.py",
        "rebuild_context_graph_all.py",
        "rebuild_mid_concept_graph.py",
        "rebuild_rq_membership_graph.py",
        "rebuild_structure_graph.py",
        "reconcile_ingestion_batch_recoveries.py",
        "reconcile_scoped_rebuild_cache_invalidations.py",
        "reconcile_source_snapshots.py",
        "reconcile_vector_records.py",
        "reconcile_versioned_graph_completion.py",
        "refresh_context_protocol_identity.py",
        "retry_versioned_graph.py",
        "source_snapshot_gc.py",
    }
    existing = {path.name for path in EXECUTABLE_SCRIPTS}
    assert write_capable <= existing
    missing_gate = []
    for name in sorted(write_capable):
        source = (SCRIPTS_ROOT / name).read_text(encoding="utf-8")
        if "--execute" not in source:
            missing_gate.append(name)
    assert missing_gate == []


def test_subprocess_environment_drops_local_credentials_and_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_API_KEY", "must-not-propagate")
    monkeypatch.setenv("CHAT_BASE_URL", "https://private.example.test")
    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "must-not-propagate")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    env = sanitized_subprocess_env({"APP_ENV": "test"})

    assert "CHAT_API_KEY" not in env
    assert "CHAT_BASE_URL" not in env
    assert "MODEL_BRIDGE_ADMIN_TOKEN" not in env
    assert env["APP_ENV"] == "test"
    assert "PATH" in env
