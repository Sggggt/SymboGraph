"""Retired pre-context-graph profile contract placeholder.

Revision ID: 20260613_0003
Revises: 20260611_0002
Create Date: 2026-06-13
"""
from __future__ import annotations


revision = "20260613_0003"
down_revision = "20260611_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    return None


def downgrade() -> None:
    raise RuntimeError("Downgrade from the four-layer context graph migration line is intentionally unsupported.")
