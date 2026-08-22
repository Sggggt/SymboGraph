"""Retire Sample-only first-import development artifact tables.

Revision ID: 20260821_0042
Revises: 20260820_0041
"""

from alembic import context, op

from app.core.migration_safety import (
    RETIRED_DEVELOPMENT_ARTIFACTS_REVISION,
    RETIRED_DEVELOPMENT_TABLES,
    require_destructive_authorization,
)


revision = RETIRED_DEVELOPMENT_ARTIFACTS_REVISION
down_revision = "20260820_0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    report = require_destructive_authorization(
        op.get_bind(),
        revision,
        x_arguments=context.get_x_argument(as_dictionary=True),
    )
    materialized = set(report["targets"])
    for table_name in RETIRED_DEVELOPMENT_TABLES:
        if table_name in materialized:
            op.drop_table(table_name)


def downgrade() -> None:
    raise RuntimeError(
        "20260821_0042 is an explicitly authorized irreversible cleanup; "
        "restore a database backup instead of recreating retired development tables."
    )
