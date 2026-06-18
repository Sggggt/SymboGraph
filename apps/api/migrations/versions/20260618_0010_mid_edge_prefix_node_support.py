"""Add RQ prefix node support to mid concept edges.

Revision ID: 20260618_0010
Revises: 20260618_0009
Create Date: 2026-06-18
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260618_0010"
down_revision = "20260618_0009"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    columns = _columns("mid_concept_edges")
    if "support_rq_prefix_node_ids_json" not in columns:
        with op.batch_alter_table("mid_concept_edges") as batch:
            batch.add_column(sa.Column("support_rq_prefix_node_ids_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    raise RuntimeError("Downgrade from mid edge prefix-node support is intentionally unsupported.")
