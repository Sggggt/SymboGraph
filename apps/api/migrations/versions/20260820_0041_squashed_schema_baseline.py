"""Squashed current-schema baseline.

Revision ID: 20260820_0041
Revises: None
Create Date: 2026-08-20

The repository retains only the current one-week migration baseline. Existing
installations already stamped at ``20260820_0041`` remain compatible; a fresh
database creates the complete current SQLAlchemy schema in this revision.
"""
from __future__ import annotations

from alembic import op


revision = "20260820_0041"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from app.models import Base

    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    from app.models import Base

    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
