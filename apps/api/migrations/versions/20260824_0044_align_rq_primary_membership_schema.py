"""Align the RQ membership table with the primary-chain schema.

Revision ID: 20260824_0044
Revises: 20260822_0043

The active protocol persists exactly one L1/L2/L3 primary chain per chunk.
Databases created by the preceding schema may contain a NOT NULL alternatives
column that is not part of the primary-chain write contract and blocks current
RQ graph construction.
"""
from __future__ import annotations

from alembic import context, op

from app.core.migration_safety import require_destructive_authorization


revision = "20260824_0044"
down_revision = "20260822_0043"
branch_labels = None
depends_on = None

TARGET = "rq_prefix_memberships.top_alternative_prefix_ids_json"


def upgrade() -> None:
    bind = op.get_bind()
    report = require_destructive_authorization(
        bind,
        revision,
        x_arguments=context.get_x_argument(as_dictionary=True),
    )
    if TARGET in report["targets"]:
        op.drop_column(
            "rq_prefix_memberships",
            "top_alternative_prefix_ids_json",
        )


def downgrade() -> None:
    raise RuntimeError(
        "The pre-primary-chain alternatives cannot be reconstructed; "
        "restore a database backup instead of recreating the column."
    )
