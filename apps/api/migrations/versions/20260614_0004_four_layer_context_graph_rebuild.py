"""Destructively replace legacy graph tables with four-layer context graph schema.

Revision ID: 20260614_0004
Revises: 20260613_0003
Create Date: 2026-06-14
"""
from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from alembic import op

from app.db import Base
import app.models  # noqa: F401


revision = "20260614_0004"
down_revision = "20260613_0003"
branch_labels = None
depends_on = None


def _quote(bind, name: str) -> str:
    return bind.dialect.identifier_preparer.quote(name)


def _preserve_knowledge_bases(bind, tables: set[str]) -> list[dict]:
    if "knowledge_bases" not in tables:
        return []
    rows = []
    columns = {column["name"] for column in sa.inspect(bind).get_columns("knowledge_bases")}
    wanted = [column for column in ("id", "name", "description", "source_root") if column in columns]
    if "id" not in wanted or "name" not in wanted:
        return []
    result = bind.execute(sa.text(f"SELECT {', '.join(_quote(bind, column) for column in wanted)} FROM {_quote(bind, 'knowledge_bases')}"))
    for row in result.mappings():
        name = row.get("name")
        if not name:
            continue
        rows.append(
            {
                "id": row.get("id"),
                "name": name,
                "description": row.get("description"),
                "source_root": row.get("source_root") or "",
                "current_chunk_version": 0,
                "active_profile_id": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )
    return rows


def _drop_application_tables(bind, tables: set[str]) -> None:
    dialect = bind.dialect.name
    if dialect == "sqlite":
        bind.execute(sa.text("PRAGMA foreign_keys=OFF"))
    for table_name in sorted(tables - {"alembic_version"}, reverse=True):
        quoted = _quote(bind, table_name)
        if dialect == "postgresql":
            bind.execute(sa.text(f"DROP TABLE IF EXISTS {quoted} CASCADE"))
        else:
            bind.execute(sa.text(f"DROP TABLE IF EXISTS {quoted}"))
    if dialect == "sqlite":
        bind.execute(sa.text("PRAGMA foreign_keys=ON"))


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    preserved_knowledge_bases = _preserve_knowledge_bases(bind, tables)
    _drop_application_tables(bind, tables)
    Base.metadata.create_all(bind=bind)
    if preserved_knowledge_bases:
        op.bulk_insert(
            sa.table(
                "knowledge_bases",
                sa.column("id", sa.String),
                sa.column("name", sa.String),
                sa.column("description", sa.Text),
                sa.column("source_root", sa.Text),
                sa.column("current_chunk_version", sa.Integer),
                sa.column("active_profile_id", sa.String),
                sa.column("created_at", sa.DateTime),
                sa.column("updated_at", sa.DateTime),
            ),
            preserved_knowledge_bases,
        )


def downgrade() -> None:
    raise RuntimeError("Downgrade from four-layer context graph schema is intentionally unsupported.")
