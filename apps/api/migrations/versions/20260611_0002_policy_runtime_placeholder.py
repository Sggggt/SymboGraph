"""Retired legacy policy migration placeholder.

Revision ID: 20260611_0002
Revises: 20260611_0001
Create Date: 2026-06-11
"""
from __future__ import annotations


revision = "20260611_0002"
down_revision = "20260611_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    return None


def downgrade() -> None:
    raise RuntimeError("Downgrade from the four-layer context graph migration line is intentionally unsupported.")
