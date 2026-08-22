from __future__ import annotations

import os
from pathlib import Path


def test_database_engine_fork_reset_replaces_pool_once_per_child(
    monkeypatch,
) -> None:
    from app import db

    dispose_calls: list[bool] = []

    class FakeEngine:
        def dispose(self, *, close: bool = True) -> None:
            dispose_calls.append(close)

    monkeypatch.setattr(db, "engine", FakeEngine())
    monkeypatch.setattr(db, "_engine_process_id", os.getpid() - 1)

    assert db.reset_database_engine_after_fork() is True
    assert dispose_calls == [False]
    assert db._engine_process_id == os.getpid()

    assert db.reset_database_engine_after_fork() is False
    assert dispose_calls == [False]

    assert db.reset_database_engine_after_fork(force=True) is True
    assert dispose_calls == [False, False]


def test_celery_prefork_child_registers_explicit_database_pool_reset() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    celery_source = (
        repository_root
        / "apps"
        / "worker"
        / "worker_app"
        / "celery_app.py"
    ).read_text(encoding="utf-8")
    database_source = (
        repository_root / "apps" / "api" / "app" / "db.py"
    ).read_text(encoding="utf-8")

    assert "from celery.signals import worker_process_init" in celery_source
    assert "@worker_process_init.connect(weak=False)" in celery_source
    assert "reset_database_engine_after_fork(force=True)" in celery_source
    assert "os.register_at_fork(" in database_source
    assert "after_in_child=reset_database_engine_after_fork" in database_source
    assert "engine.dispose(close=False)" in database_source
