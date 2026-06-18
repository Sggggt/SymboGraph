"""Rename active traversal trace fields.

Revision ID: 20260617_0008
Revises: 20260615_0007
Create Date: 2026-06-17
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260617_0008"
down_revision = "20260615_0007"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _rename_or_add(table_name: str, old_name: str, new_column: sa.Column) -> None:
    if table_name not in _tables():
        return
    columns = _columns(table_name)
    with op.batch_alter_table(table_name) as batch:
        if old_name in columns and new_column.name not in columns:
            batch.alter_column(old_name, new_column_name=new_column.name, existing_type=new_column.type)
        elif new_column.name not in columns:
            batch.add_column(new_column)
        if old_name in columns and new_column.name in columns:
            batch.drop_column(old_name)


def upgrade() -> None:
    _rename_or_add("graph_retrieval_steps", "cycle_reward", sa.Column("cycle_distance_reward", sa.Float(), nullable=True))
    _rename_or_add("graph_retrieval_steps", "ambiguous_edge_decisions_json", sa.Column("gray_zone_path_decisions_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    raise RuntimeError("Downgrade from active trace protocol rename is intentionally unsupported.")
