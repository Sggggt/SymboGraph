"""Add typed Agent and runtime hash closure fields.

Revision ID: 20260614_0006
Revises: 20260614_0005
Create Date: 2026-06-14
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260614_0006"
down_revision = "20260614_0005"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if table_name not in _tables():
        return
    if column.name in _columns(table_name):
        return
    with op.batch_alter_table(table_name) as batch:
        batch.add_column(column)


def _create_index_if_missing(table_name: str, index_name: str, columns: list[str]) -> None:
    if table_name not in _tables():
        return
    indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}
    if index_name in indexes:
        return
    op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    _add_column_if_missing("context_graph_states", sa.Column("runtime_settings_hash", sa.String(length=64), nullable=True))
    _add_column_if_missing("context_graph_states", sa.Column("agent_operating_envelope_hash", sa.String(length=64), nullable=True))
    _add_column_if_missing("retrieval_traces", sa.Column("runtime_settings_hash", sa.String(length=64), nullable=True))
    _add_column_if_missing("retrieval_traces", sa.Column("agent_operating_envelope_hash", sa.String(length=64), nullable=True))
    _add_column_if_missing("context_packages", sa.Column("runtime_settings_hash", sa.String(length=64), nullable=True))
    _add_column_if_missing("context_packages", sa.Column("profile_hash", sa.String(length=64), nullable=True))

    tables = _tables()
    if "agent_plans" not in tables:
        op.create_table(
            "agent_plans",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("knowledge_base_id", sa.String(length=36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
            sa.Column("retrieval_trace_id", sa.String(length=36), sa.ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True),
            sa.Column("plan_index", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("planner_model_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("query_intent_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("envelope_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("typed_actions_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
            sa.Column("validation_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="planned"),
            sa.Column("diagnostics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
    if "agent_actions" not in tables:
        op.create_table(
            "agent_actions",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("plan_id", sa.String(length=36), sa.ForeignKey("agent_plans.id", ondelete="CASCADE"), nullable=True),
            sa.Column("parent_action_id", sa.String(length=36), sa.ForeignKey("agent_actions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("action_index", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("action_type", sa.String(length=96), nullable=False),
            sa.Column("target_ids_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
            sa.Column("reason", sa.Text(), nullable=False, server_default=""),
            sa.Column("budget_request_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("expected_evidence_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("stop_condition_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("validation_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
            sa.Column("output_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("diagnostics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
    if "agent_observations" not in tables:
        op.create_table(
            "agent_observations",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("action_id", sa.String(length=36), sa.ForeignKey("agent_actions.id", ondelete="CASCADE"), nullable=True),
            sa.Column("observation_type", sa.String(length=96), nullable=False),
            sa.Column("observation_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("evidence_chunk_ids_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
            sa.Column("verdict", sa.String(length=32), nullable=False, server_default="observed"),
            sa.Column("diagnostics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )

    for table, columns in {
        "context_graph_states": ["runtime_settings_hash", "agent_operating_envelope_hash"],
        "retrieval_traces": ["runtime_settings_hash", "agent_operating_envelope_hash"],
        "context_packages": ["runtime_settings_hash", "profile_hash"],
        "agent_plans": ["run_id", "knowledge_base_id", "retrieval_trace_id", "status"],
        "agent_actions": ["run_id", "plan_id", "parent_action_id", "action_type", "status"],
        "agent_observations": ["run_id", "action_id", "observation_type", "verdict"],
    }.items():
        for column in columns:
            _create_index_if_missing(table, f"ix_{table}_{column}", [column])


def downgrade() -> None:
    raise RuntimeError("Downgrade from typed Agent runtime closure is intentionally unsupported.")
