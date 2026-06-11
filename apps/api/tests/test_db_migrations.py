from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_alembic_initial_upgrade_creates_evidence_graph_schema(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'alembic-test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    from app.core.config import get_settings

    get_settings.cache_clear()
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")

    engine = create_engine(database_url, future=True)
    try:
        tables = set(inspect(engine).get_table_names())
        assert "knowledge_bases" in tables
        assert "evidence_atoms" in tables
        assert "active_chunks" in tables
        assert "policy_states" in tables
        assert "reward_events" in tables
        assert "alembic_version" in tables
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        assert version == "20260611_0002"
    finally:
        engine.dispose()
        get_settings.cache_clear()
