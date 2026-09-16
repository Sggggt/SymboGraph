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
CURRENT_UPGRADE_PATH = (
    VERSIONS_ROOT / "20260916_0045_upgrade_from_v7_1_2.py"
)


def _baseline_module():
    spec = importlib.util.spec_from_file_location("squashed_baseline", BASELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_recent_squashed_migrations_are_retained() -> None:
    revisions = sorted(VERSIONS_ROOT.glob("*.py"))
    assert revisions == [
        BASELINE_PATH,
        SINGLE_ENV_PATH,
        RQ_PRIMARY_SCHEMA_PATH,
        CURRENT_UPGRADE_PATH,
    ]

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

    spec = importlib.util.spec_from_file_location(
        "current_upgrade", CURRENT_UPGRADE_PATH
    )
    assert spec is not None and spec.loader is not None
    current_upgrade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current_upgrade)
    assert current_upgrade.revision == "20260916_0045"
    assert current_upgrade.down_revision == "20260824_0044"


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
        ).scalar_one() == "20260916_0045"
        command.check(config)
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_last_pushed_revision_upgrades_through_one_consolidated_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings
    from app.models import Base

    database_path = tmp_path / "v7-1-2-upgrade.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    config = Config(str(API_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "20260824_0044")
    engine = sa.create_engine(database_url, future=True)
    try:
        before = set(sa.inspect(engine).get_table_names())
        assert "answer_source_bindings" not in before
        assert "lexical_index_states" not in before

        command.upgrade(config, "head")
        command.check(config)
        after = set(sa.inspect(engine).get_table_names()) - {"alembic_version"}
        assert after == set(Base.metadata.tables)
        with engine.connect() as connection:
            assert connection.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "20260916_0045"
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_postgres_fresh_upgrade_and_reversible_convergence_use_isolated_schema():
    from uuid import uuid4
    from app.db import engine
    if engine.dialect.name != 'postgresql':
        pytest.skip('requires Docker PostgreSQL')
    from alembic.migration import MigrationContext
    schema = 'unit_release_' + uuid4().hex
    with engine.connect() as connection:
        transaction = connection.begin()
        original_schema = connection.dialect.default_schema_name
        try:
            connection.execute(sa.schema.CreateSchema(schema))
            connection.execute(sa.text('SET LOCAL search_path TO '+schema))
            connection.dialect.default_schema_name = schema
            config = Config(str(API_ROOT/'alembic.ini'))
            config.attributes['connection'] = connection
            command.upgrade(config,'head')
            command.check(config)
            command.downgrade(config,'20260824_0044')
            assert MigrationContext.configure(connection).get_current_revision() == '20260824_0044'
            command.upgrade(config,'head')
            command.check(config)
        finally:
            connection.dialect.default_schema_name = original_schema
            transaction.rollback()


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
        assert "answer_source_bindings" not in sa.inspect(engine).get_table_names()
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
            ).scalar_one() == "20260916_0045"
    finally:
        engine.dispose()
        get_settings.cache_clear()
