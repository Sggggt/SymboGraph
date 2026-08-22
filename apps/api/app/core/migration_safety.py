from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine


BASELINE_REVISION = "20260820_0041"
RETIRED_DEVELOPMENT_ARTIFACTS_REVISION = "20260821_0042"
DESTRUCTIVE_MIGRATION_ENV = "ALLOW_DESTRUCTIVE_MIGRATIONS"
RETIRED_DEVELOPMENT_TABLES = (
    "runtime_settings_initial_graph_handoff_receipts",
    "runtime_settings_initial_graph_source_closure_amendments",
    "first_import_graph_retry_execution_intents",
    "runtime_settings_initial_graph_activation_intents",
)
DESTRUCTIVE_UPGRADE_REVISIONS = {
    RETIRED_DEVELOPMENT_ARTIFACTS_REVISION: RETIRED_DEVELOPMENT_TABLES,
}


def _normalized_tokens(value: str | None) -> set[str]:
    return {token.strip() for token in (value or "").split(",") if token.strip()}


def destructive_authorization_source(
    revision: str,
    *,
    x_arguments: dict[str, str] | None = None,
    force_authorized: bool = False,
) -> str | None:
    if force_authorized:
        return "explicit_cli_flag"
    x_value = str((x_arguments or {}).get("allow_destructive") or "").strip()
    if x_value.lower() in {"1", "true", "yes", "on"} or revision in _normalized_tokens(
        x_value
    ):
        return "alembic_x_argument"
    if revision in _normalized_tokens(os.getenv(DESTRUCTIVE_MIGRATION_ENV)):
        return f"environment:{DESTRUCTIVE_MIGRATION_ENV}"
    return None


def revision_report(
    bind: Connection,
    revision: str,
    *,
    x_arguments: dict[str, str] | None = None,
    force_authorized: bool = False,
) -> dict[str, Any]:
    configured_targets = DESTRUCTIVE_UPGRADE_REVISIONS.get(revision, ())
    existing = set(inspect(bind).get_table_names()) if configured_targets else set()
    targets = [name for name in configured_targets if name in existing]
    authorization_source = destructive_authorization_source(
        revision,
        x_arguments=x_arguments,
        force_authorized=force_authorized,
    )
    authorized = not targets or authorization_source is not None
    return {
        "revision": revision,
        "destructive": bool(configured_targets),
        "authorization_required": bool(targets),
        "authorized": authorized,
        "authorization_source": authorization_source,
        "status": (
            "safe_no_materialized_targets"
            if not targets
            else "authorized"
            if authorized
            else "blocked"
        ),
        "targets": targets,
    }


def render_revision_report(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "[migration-destructive-preflight] "
            f"revision={report['revision']} status={report['status']}",
            f"authorization_required={str(report['authorization_required']).lower()} "
            f"source={report.get('authorization_source') or 'none'}",
            "targets=" + json.dumps(report["targets"], ensure_ascii=False),
        )
    )


def require_destructive_authorization(
    bind: Connection,
    revision: str,
    *,
    x_arguments: dict[str, str] | None = None,
) -> dict[str, Any]:
    report = revision_report(bind, revision, x_arguments=x_arguments)
    print(render_revision_report(report), flush=True)
    if not report["authorized"]:
        raise RuntimeError(
            "Destructive migration is blocked. Review the printed targets and rerun "
            "with -x allow_destructive=true."
        )
    return report


def pending_revision_ids(
    bind: Connection,
    target_revision: str,
    alembic_ini: Path,
) -> list[str]:
    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(alembic_ini.parent / "migrations"))
    script = ScriptDirectory.from_config(config)
    current_heads = MigrationContext.configure(bind).get_current_heads()
    lower: str | Iterable[str] | None = current_heads if current_heads else None
    revisions = list(script.iterate_revisions(target_revision, lower))
    return [str(item.revision) for item in reversed(revisions)]


def database_preflight_report(
    bind: Connection,
    target_revision: str,
    alembic_ini: Path,
    *,
    force_authorized: bool = False,
) -> dict[str, Any]:
    pending_revisions = pending_revision_ids(bind, target_revision, alembic_ini)
    revision_reports = [
        revision_report(bind, revision, force_authorized=force_authorized)
        for revision in pending_revisions
        if revision in DESTRUCTIVE_UPGRADE_REVISIONS
    ]
    pending_destructive = [
        report["revision"]
        for report in revision_reports
        if report["authorization_required"]
    ]
    authorized = all(report["authorized"] for report in revision_reports)
    return {
        "operation": "alembic_upgrade_preflight",
        "target_revision": target_revision,
        "current_revisions": list(MigrationContext.configure(bind).get_current_heads()),
        "pending_revisions": pending_revisions,
        "pending_destructive_revisions": pending_destructive,
        "authorized": authorized,
        "status": "safe" if authorized else "blocked",
        "revision_reports": revision_reports,
        "baseline_revision": BASELINE_REVISION,
        "force_authorized": bool(force_authorized),
    }


def render_database_report(report: dict[str, Any]) -> str:
    lines = [
        f"[migration-preflight] target={report['target_revision']} status={report['status']}",
        "current_revisions=" + json.dumps(report["current_revisions"], ensure_ascii=False),
        "pending_revisions=" + json.dumps(report["pending_revisions"], ensure_ascii=False),
        "pending_destructive_revisions="
        + json.dumps(report["pending_destructive_revisions"], ensure_ascii=False),
    ]
    lines.extend(render_revision_report(item) for item in report["revision_reports"])
    if not report["authorized"]:
        lines.append("Review targets, then rerun with --allow-destructive if approved.")
    return "\n".join(lines)


def _default_alembic_ini() -> Path:
    return Path(__file__).resolve().parents[2] / "alembic.ini"


def run_preflight(
    engine: Engine,
    *,
    target_revision: str,
    alembic_ini: Path,
    force_authorized: bool = False,
) -> dict[str, Any]:
    with engine.connect() as connection:
        return database_preflight_report(
            connection,
            target_revision,
            alembic_ini,
            force_authorized=force_authorized,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Alembic migration preflight."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="Inspect pending revisions without mutating the database."
    )
    preflight.add_argument("--target-revision", default="head")
    preflight.add_argument("--allow-destructive", action="store_true")
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument("--alembic-ini", type=Path, default=_default_alembic_ini())
    args = parser.parse_args(argv)

    from app.db import engine

    report = run_preflight(
        engine,
        target_revision=args.target_revision,
        alembic_ini=args.alembic_ini,
        force_authorized=args.allow_destructive,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_database_report(report))
    return 0 if report["authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
