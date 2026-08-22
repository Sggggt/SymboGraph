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
        "cleanup_orphan_mid_rq_memberships.py",
        "cleanup_stale_data.py",
        "cleanup_vector_collection.py",
        "destroy_legacy_derived_data.py",
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


def test_scripts_do_not_reintroduce_retired_terminal_or_private_fixtures() -> None:
    forbidden = (
        "sample_terminal_v5",
        "\u006f\u0075\u0074\u0070\u0075\u0074\u002f\u0066\u0069\u006e\u0061\u006c\u002d\u0073\u0061\u006d\u0070\u006c\u0065",
        "\u0077\u0069\u006e\u0063\u006f\u0064\u0065\u002e\u0077\u0069\u006e\u006e\u0069\u006e\u0067\u002e\u0063\u006f\u006d\u002e\u0063\u006e",
        "\u4f55\u52b2\u71ca",
    )
    matches: list[str] = []
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in source:
                matches.append(f"{path.name}:{marker}")
    assert matches == []


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
