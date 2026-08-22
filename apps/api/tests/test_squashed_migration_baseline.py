from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
import pytest
from alembic import command
from alembic.config import Config


API_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = API_ROOT / "migrations" / "versions"
BASELINE_PATH = (
    VERSIONS_ROOT / "20260820_0041_squashed_schema_baseline.py"
)
RETIREMENT_PATH = (
    VERSIONS_ROOT / "20260821_0042_retire_development_artifacts.py"
)
SINGLE_ENV_PATH = VERSIONS_ROOT / "20260822_0043_single_root_env.py"


def _baseline_module():
    spec = importlib.util.spec_from_file_location("squashed_baseline", BASELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_recent_squashed_migrations_are_retained() -> None:
    revisions = sorted(VERSIONS_ROOT.glob("*.py"))
    assert revisions == [BASELINE_PATH, RETIREMENT_PATH, SINGLE_ENV_PATH]

    module = _baseline_module()
    assert module.revision == "20260820_0041"
    assert module.down_revision is None

    spec = importlib.util.spec_from_file_location("retirement", RETIREMENT_PATH)
    assert spec is not None and spec.loader is not None
    retirement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retirement)
    assert retirement.revision == "20260821_0042"
    assert retirement.down_revision == "20260820_0041"

    spec = importlib.util.spec_from_file_location("single_env", SINGLE_ENV_PATH)
    assert spec is not None and spec.loader is not None
    single_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(single_env)
    assert single_env.revision == "20260822_0043"
    assert single_env.down_revision == "20260821_0042"


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
            ).scalar_one() == "20260822_0043"
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_retired_tables_require_explicit_destructive_authorization() -> None:
    from app.core.migration_safety import (
        RETIRED_DEVELOPMENT_ARTIFACTS_REVISION,
        RETIRED_DEVELOPMENT_TABLES,
        revision_report,
    )

    engine = sa.create_engine("sqlite:///:memory:", future=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text(f"CREATE TABLE {RETIRED_DEVELOPMENT_TABLES[-1]} (id TEXT)")
            )
            blocked = revision_report(
                connection,
                RETIRED_DEVELOPMENT_ARTIFACTS_REVISION,
            )
            assert blocked["status"] == "blocked"
            assert blocked["targets"] == [RETIRED_DEVELOPMENT_TABLES[-1]]

            allowed = revision_report(
                connection,
                RETIRED_DEVELOPMENT_ARTIFACTS_REVISION,
                force_authorized=True,
            )
            assert allowed["status"] == "authorized"
            assert allowed["authorization_source"] == "explicit_cli_flag"
    finally:
        engine.dispose()
