"""Squashed current-schema baseline.

Revision ID: 20260820_0041
Revises: None
Create Date: 2026-08-20

This revision creates the complete current SQLAlchemy schema for a fresh
database.
"""
from __future__ import annotations

from alembic import op


revision = "20260820_0041"
down_revision = None
branch_labels = None
depends_on = None
POST_BASELINE_TABLES = frozenset({"answer_source_bindings", "context_package_source_retentions", "context_package_source_expansions",
    "retrieval_lexical_policies", "retrieval_lexical_rewards", "lexical_index_states", "lexical_documents",
    "lexical_terms", "lexical_postings", "lexical_index_jobs"})


def _baseline_tables():
    from app.models import Base

    # Later revisions own these tables even when current ORM metadata knows them.
    return [table for table in Base.metadata.sorted_tables if table.name not in POST_BASELINE_TABLES]


def upgrade() -> None:
    from app.models import Base

    Base.metadata.create_all(bind=op.get_bind(), tables=_baseline_tables(), checkfirst=True)


def downgrade() -> None:
    from app.models import Base

    Base.metadata.drop_all(bind=op.get_bind(), tables=_baseline_tables(), checkfirst=True)
