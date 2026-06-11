"""Add policy drift and feedback-driven candidate fields.

Revision ID: 20260611_0002
Revises: 20260611_0001
Create Date: 2026-06-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260611_0002"
down_revision = "20260611_0001"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns(table_name)}


def _check_constraints(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    try:
        return {constraint["name"] for constraint in inspector.get_check_constraints(table_name) if constraint.get("name")}
    except NotImplementedError:
        return set()


def upgrade() -> None:
    chunk_candidate_columns = _columns("chunk_candidates")
    if "feedback_driven" not in chunk_candidate_columns:
        op.add_column(
            "chunk_candidates",
            sa.Column("feedback_driven", sa.Boolean(), nullable=False, server_default=sa.false()),
        )

    policy_state_columns = _columns("policy_states")
    if "drift_detected_at" not in policy_state_columns:
        op.add_column("policy_states", sa.Column("drift_detected_at", sa.DateTime(), nullable=True))

    dialect = op.get_bind().dialect.name
    if dialect != "sqlite":
        constraints = _check_constraints("signal_nodes")
        if "ck_signal_node_support_atoms_present" not in constraints:
            op.create_check_constraint(
                "ck_signal_node_support_atoms_present",
                "signal_nodes",
                "support_atom_ids_json IS NOT NULL",
            )
        if "ck_signal_node_source_span_present" not in constraints:
            op.create_check_constraint(
                "ck_signal_node_source_span_present",
                "signal_nodes",
                "source_span_union_json IS NOT NULL",
            )


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect != "sqlite":
        constraints = _check_constraints("signal_nodes")
        if "ck_signal_node_source_span_present" in constraints:
            op.drop_constraint("ck_signal_node_source_span_present", "signal_nodes", type_="check")
        if "ck_signal_node_support_atoms_present" in constraints:
            op.drop_constraint("ck_signal_node_support_atoms_present", "signal_nodes", type_="check")

    policy_state_columns = _columns("policy_states")
    if "drift_detected_at" in policy_state_columns:
        op.drop_column("policy_states", "drift_detected_at")

    chunk_candidate_columns = _columns("chunk_candidates")
    if "feedback_driven" in chunk_candidate_columns:
        op.drop_column("chunk_candidates", "feedback_driven")
