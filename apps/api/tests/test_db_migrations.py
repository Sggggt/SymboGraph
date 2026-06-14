from __future__ import annotations


def test_alembic_upgrade_creates_four_layer_schema(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect
    from app.core.config import get_settings

    database_path = tmp_path / "migration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{database_path.as_posix()}", future=True)
    config = Config("alembic.ini")
    config.set_main_option("script_location", "migrations")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "head")
    tables = set(inspect(engine).get_table_names())
    assert "chunks" in tables
    assert "chunk_structure_nodes" in tables
    assert "chunk_relation_edges" in tables
    assert "fine_clusters" in tables
    assert "mid_concepts" in tables
    assert "coarse_concepts" in tables
    assert "context_packages" in tables
    assert "evidence_atoms" not in tables
    assert "active_chunks" not in tables
