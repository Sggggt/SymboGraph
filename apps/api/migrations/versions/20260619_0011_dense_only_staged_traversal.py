"""Switch active retrieval storage to dense-only staged traversal.

Revision ID: 20260619_0011
Revises: 20260618_0010
Create Date: 2026-06-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260619_0011"
down_revision = "20260618_0010"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _indexes(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _add_columns(table_name: str, columns: list[sa.Column]) -> None:
    if table_name not in _tables():
        return
    existing = _columns(table_name)
    with op.batch_alter_table(table_name) as batch:
        for column in columns:
            if column.name not in existing:
                batch.add_column(column)


def _create_index_if_missing(table_name: str, index_name: str, columns: list[str]) -> None:
    if table_name not in _tables() or index_name in _indexes(table_name):
        return
    op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    if "bm25_records" in _tables():
        op.drop_table("bm25_records")

    _add_columns(
        "chunk_relation_graph_states",
        [
            sa.Column("graph_operating_point_hash", sa.String(length=64), nullable=True),
            sa.Column("graph_operating_point_json", sa.JSON(), nullable=True),
            sa.Column("edge_distance_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("edge_type_calibration_protocol_hash", sa.String(length=64), nullable=True),
        ],
    )
    _add_columns(
        "chunk_relation_edges",
        [
            sa.Column("normalization_stats_json", sa.JSON(), nullable=True),
            sa.Column("edge_distance_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("source_language", sa.String(length=32), nullable=True),
            sa.Column("target_language", sa.String(length=32), nullable=True),
            sa.Column("is_cross_document", sa.Boolean(), nullable=True),
            sa.Column("is_cross_language", sa.Boolean(), nullable=True),
            sa.Column("bridge_quota_reason", sa.String(length=96), nullable=True),
        ],
    )
    _add_columns(
        "retrieval_traces",
        [
            sa.Column("stage_queues_json", sa.JSON(), nullable=True),
            sa.Column("candidate_pools_json", sa.JSON(), nullable=True),
            sa.Column("topk_selection_json", sa.JSON(), nullable=True),
        ],
    )
    _add_columns(
        "graph_retrieval_steps",
        [
            sa.Column("action_type", sa.String(length=96), nullable=True),
            sa.Column("parent_layer", sa.String(length=64), nullable=True),
            sa.Column("parent_node_id", sa.String(length=128), nullable=True),
            sa.Column("candidate_pool_ids_json", sa.JSON(), nullable=True),
            sa.Column("selected_topk_ids_json", sa.JSON(), nullable=True),
            sa.Column("per_parent_budget_status_json", sa.JSON(), nullable=True),
        ],
    )

    index_specs = [
        ("chunk_relation_graph_states", "ix_crgs_operating_hash", ["graph_operating_point_hash"]),
        ("chunk_relation_graph_states", "ix_crgs_edge_distance_hash", ["edge_distance_protocol_hash"]),
        ("chunk_relation_graph_states", "ix_crgs_edge_calibration_hash", ["edge_type_calibration_protocol_hash"]),
        ("chunk_relation_edges", "ix_cre_edge_distance_hash", ["edge_distance_protocol_hash"]),
        ("chunk_relation_edges", "ix_cre_source_language", ["source_language"]),
        ("chunk_relation_edges", "ix_cre_target_language", ["target_language"]),
        ("chunk_relation_edges", "ix_cre_cross_document", ["is_cross_document"]),
        ("chunk_relation_edges", "ix_cre_cross_language", ["is_cross_language"]),
        ("graph_retrieval_steps", "ix_grs_action_type", ["action_type"]),
        ("graph_retrieval_steps", "ix_grs_parent_layer", ["parent_layer"]),
        ("graph_retrieval_steps", "ix_grs_parent_node", ["parent_node_id"]),
    ]
    for table_name, index_name, columns in index_specs:
        _create_index_if_missing(table_name, index_name, columns)


def downgrade() -> None:
    raise RuntimeError("Downgrade from dense-only staged traversal is intentionally unsupported.")
