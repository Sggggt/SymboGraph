from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manage_migrations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("manage_migrations_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_runs_preflight_before_alembic_and_stops_when_blocked(monkeypatch):
    module = _load_module()
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, dry_run: bool = False) -> int:
        commands.append(command)
        return 2

    monkeypatch.setattr(module, "run_command", fake_run)

    assert module.main(["upgrade", "head"]) == 2
    assert len(commands) == 1
    assert commands[0][-6:] == [
        "python",
        "-m",
        "app.core.migration_safety",
        "preflight",
        "--target-revision",
        "head",
    ]
    assert "alembic" not in commands[0]


def test_authorized_upgrade_uses_one_shot_preflight_and_alembic_x_argument(monkeypatch):
    module = _load_module()
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, dry_run: bool = False) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(module, "run_command", fake_run)

    assert module.main(["--compose-run", "upgrade", "head", "--allow-destructive"]) == 0
    assert len(commands) == 2
    assert commands[0][-1] == "--allow-destructive"
    assert commands[1][-5:] == ["alembic", "-x", "allow_destructive=true", "upgrade", "head"]
    assert commands[0][commands[0].index("--project-name") + 1] == (
        "knowledgegraph-dev-20260820"
    )
    assert ["run", "--rm", "--no-deps", "api"] == commands[0][commands[0].index("run") : commands[0].index("run") + 4]


def test_executing_downgrade_requires_explicit_authorization():
    module = _load_module()

    with pytest.raises(SystemExit) as caught:
        module.main(["downgrade", "-1"])

    assert caught.value.code == 2
