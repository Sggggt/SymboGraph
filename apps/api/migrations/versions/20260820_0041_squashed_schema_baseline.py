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


def upgrade() -> None:
    from app.models import Base

    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    from app.models import Base

    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
