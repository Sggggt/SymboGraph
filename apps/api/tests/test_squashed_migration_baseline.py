from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import sqlalchemy as sa
import pytest
from alembic import command
from alembic.config import Config


API_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = API_ROOT / "migrations" / "versions"
BASELINE_PATH = (
    VERSIONS_ROOT / "20260820_0041_squashed_schema_baseline.py"
)
SINGLE_ENV_PATH = VERSIONS_ROOT / "20260822_0043_single_root_env.py"
RQ_PRIMARY_SCHEMA_PATH = (
    VERSIONS_ROOT
    / "20260824_0044_align_rq_primary_membership_schema.py"
)


def _baseline_module():
    spec = importlib.util.spec_from_file_location("squashed_baseline", BASELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_recent_squashed_migrations_are_retained() -> None:
    revisions = sorted(VERSIONS_ROOT.glob("*.py"))
    assert revisions == [BASELINE_PATH, SINGLE_ENV_PATH, RQ_PRIMARY_SCHEMA_PATH]

    module = _baseline_module()
    assert module.revision == "20260820_0041"
    assert module.down_revision is None

    spec = importlib.util.spec_from_file_location("single_env", SINGLE_ENV_PATH)
    assert spec is not None and spec.loader is not None
    single_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(single_env)
    assert single_env.revision == "20260822_0043"
    assert single_env.down_revision == "20260820_0041"

    spec = importlib.util.spec_from_file_location(
        "rq_primary_schema", RQ_PRIMARY_SCHEMA_PATH
    )
    assert spec is not None and spec.loader is not None
    rq_primary_schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rq_primary_schema)
    assert rq_primary_schema.revision == "20260824_0044"
    assert rq_primary_schema.down_revision == "20260822_0043"


def test_fresh_sqlite_upgrade_matches_current_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings
    from app.models import Base

    database_path = tmp_path / "fresh-baseline.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    config = Config(str(API_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")

    engine = sa.create_engine(database_url, future=True)
    try:
        observed = set(sa.inspect(engine).get_table_names()) - {"alembic_version"}
        assert observed == set(Base.metadata.tables)
        with engine.connect() as connection:
            assert connection.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "20260824_0044"
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_pre_primary_rq_column_requires_explicit_drop_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings

    database_path = tmp_path / "pre-primary-rq-column.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    config = Config(str(API_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "20260822_0043")
    engine = sa.create_engine(database_url, future=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text(
                    "ALTER TABLE rq_prefix_memberships "
                    "ADD COLUMN top_alternative_prefix_ids_json JSON "
                    "NOT NULL DEFAULT '[]'"
                )
            )

        with pytest.raises(RuntimeError, match="Destructive migration is blocked"):
            command.upgrade(config, "head")
        assert "top_alternative_prefix_ids_json" in {
            column["name"]
            for column in sa.inspect(engine).get_columns("rq_prefix_memberships")
        }

        config.cmd_opts = SimpleNamespace(x=["allow_destructive=true"])
        command.upgrade(config, "head")
        assert "top_alternative_prefix_ids_json" not in {
            column["name"]
            for column in sa.inspect(engine).get_columns("rq_prefix_memberships")
        }
        with engine.connect() as connection:
            assert connection.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "20260824_0044"
    finally:
        engine.dispose()
        get_settings.cache_clear()
