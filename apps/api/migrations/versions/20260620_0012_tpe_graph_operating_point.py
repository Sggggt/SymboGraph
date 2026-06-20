"""Replace manual TPE calibration with automatic lightweight operating point runs.

Revision ID: 20260620_0012
Revises: 20260619_0011
Create Date: 2026-06-20
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260620_0012"
down_revision = "20260619_0011"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _tables() -> set[str]:
    return set(_inspector().get_table_names())


def _columns(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {column["name"] for column in _inspector().get_columns(table_name)}


def _indexes(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {index["name"] for index in _inspector().get_indexes(table_name)}


def _create_index_if_missing(table_name: str, index_name: str, columns: list[str]) -> None:
    if table_name in _tables() and index_name not in _indexes(table_name):
        op.create_index(index_name, table_name, columns)


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if column.name not in _columns(table_name):
        op.add_column(table_name, column)


def upgrade() -> None:
    for table_name in ("tpe_trials", "tpe_calibration_jobs"):
        if table_name in _tables():
            op.drop_table(table_name)

    if "chunk_relation_graph_states" in _tables():
        _add_column_if_missing("chunk_relation_graph_states", sa.Column("runtime_settings_hash", sa.String(length=64), nullable=True))
        _add_column_if_missing("chunk_relation_graph_states", sa.Column("auto_tpe_run_id", sa.String(length=36), nullable=True))
        _add_column_if_missing("chunk_relation_graph_states", sa.Column("auto_tpe_best_trial_id", sa.String(length=36), nullable=True))
        _create_index_if_missing("chunk_relation_graph_states", "ix_crgs_runtime_settings_hash", ["runtime_settings_hash"])
        _create_index_if_missing("chunk_relation_graph_states", "ix_crgs_auto_tpe_run_id", ["auto_tpe_run_id"])
        _create_index_if_missing("chunk_relation_graph_states", "ix_crgs_auto_tpe_best_trial_id", ["auto_tpe_best_trial_id"])

    if "auto_tpe_runs" not in _tables():
        op.create_table(
            "auto_tpe_runs",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
            sa.Column("batch_id", sa.String(length=36), nullable=True),
            sa.Column("chunk_relation_graph_state_id", sa.String(length=36), nullable=True),
            sa.Column("chunk_version", sa.Integer(), nullable=False),
            sa.Column("chunk_scope_hash", sa.String(length=64), nullable=False),
            sa.Column("graph_operating_point_protocol", sa.String(length=64), nullable=False),
            sa.Column("protocol_hash", sa.String(length=64), nullable=False),
            sa.Column("chat_model", sa.String(length=128), nullable=False),
            sa.Column("embedding_model", sa.String(length=128), nullable=False),
            sa.Column("embedding_text_version", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("trigger_reason", sa.String(length=128), nullable=False),
            sa.Column("trial_budget", sa.Integer(), nullable=False),
            sa.Column("startup_random_trials", sa.Integer(), nullable=False),
            sa.Column("good_quantile_gamma", sa.Float(), nullable=False),
            sa.Column("probe_query_budget", sa.Integer(), nullable=False),
            sa.Column("candidate_pool_size", sa.Integer(), nullable=False),
            sa.Column("best_trial_id", sa.String(length=36), nullable=True),
            sa.Column("best_objective_score", sa.Float(), nullable=True),
            sa.Column("selected_theta_hash", sa.String(length=64), nullable=True),
            sa.Column("selected_theta_json", sa.JSON(), nullable=True),
            sa.Column("sampler_state_hash", sa.String(length=64), nullable=True),
            sa.Column("probe_set_hash", sa.String(length=64), nullable=True),
            sa.Column("hard_gate_json", sa.JSON(), nullable=True),
            sa.Column("objective_components_json", sa.JSON(), nullable=True),
            sa.Column("blocking_reasons_json", sa.JSON(), nullable=True),
            sa.Column("runtime_settings_hash", sa.String(length=64), nullable=True),
            sa.Column("diagnostics_json", sa.JSON(), nullable=True),
            sa.Column("failure_code", sa.String(length=128), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["batch_id"], ["ingestion_batches.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["chunk_relation_graph_state_id"], ["chunk_relation_graph_states.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if "auto_tpe_trials" not in _tables():
        op.create_table(
            "auto_tpe_trials",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
            sa.Column("trial_index", sa.Integer(), nullable=False),
            sa.Column("sampled_theta_json", sa.JSON(), nullable=True),
            sa.Column("theta_hash", sa.String(length=64), nullable=False),
            sa.Column("sampler_state_hash", sa.String(length=64), nullable=False),
            sa.Column("candidate_adjacency_hash", sa.String(length=64), nullable=True),
            sa.Column("probe_set_hash", sa.String(length=64), nullable=True),
            sa.Column("hard_gate_json", sa.JSON(), nullable=True),
            sa.Column("objective_components_json", sa.JSON(), nullable=True),
            sa.Column("objective_score", sa.Float(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("failure_code", sa.String(length=128), nullable=True),
            sa.Column("diagnostics_json", sa.JSON(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["run_id"], ["auto_tpe_runs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("run_id", "trial_index", name="uq_auto_tpe_trial_run_index"),
        )

    for table_name, index_name, columns in (
        ("auto_tpe_runs", "ix_auto_tpe_runs_kb", ["knowledge_base_id"]),
        ("auto_tpe_runs", "ix_auto_tpe_runs_batch", ["batch_id"]),
        ("auto_tpe_runs", "ix_auto_tpe_runs_relation_state", ["chunk_relation_graph_state_id"]),
        ("auto_tpe_runs", "ix_auto_tpe_runs_scope", ["knowledge_base_id", "chunk_version", "chat_model", "embedding_model", "embedding_text_version"]),
        ("auto_tpe_runs", "ix_auto_tpe_runs_kb_status", ["knowledge_base_id", "status"]),
        ("auto_tpe_runs", "ix_auto_tpe_runs_selected_theta", ["selected_theta_hash"]),
        ("auto_tpe_trials", "ix_auto_tpe_trials_run", ["run_id"]),
        ("auto_tpe_trials", "ix_auto_tpe_trials_kb", ["knowledge_base_id"]),
        ("auto_tpe_trials", "ix_auto_tpe_trials_theta", ["theta_hash"]),
        ("auto_tpe_trials", "ix_auto_tpe_trials_adjacency", ["candidate_adjacency_hash"]),
        ("auto_tpe_trials", "ix_auto_tpe_trials_objective", ["objective_score"]),
        ("auto_tpe_trials", "ix_auto_tpe_trials_run_status", ["run_id", "status"]),
    ):
        _create_index_if_missing(table_name, index_name, columns)


def downgrade() -> None:
    raise RuntimeError("Downgrade from automatic TPE graph operating point calibration is intentionally unsupported.")
