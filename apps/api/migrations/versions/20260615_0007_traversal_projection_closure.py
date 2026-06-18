"""Add traversal and edge projection closure fields.

Revision ID: 20260615_0007
Revises: 20260614_0006
Create Date: 2026-06-15
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260615_0007"
down_revision = "20260614_0006"
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
    _add_columns(
        "chunk_relation_edges",
        [
            sa.Column("distance", sa.Float(), nullable=True),
            sa.Column("raw_strength", sa.Float(), nullable=True),
            sa.Column("raw_strength_summary_json", sa.JSON(), nullable=True),
            sa.Column("diagnostics_json", sa.JSON(), nullable=True),
        ],
    )
    _add_columns("rq_prefixes", [sa.Column("node_type", sa.String(length=64), nullable=True)])
    _add_columns(
        "rq_prefix_memberships",
        [
            sa.Column("membership_role", sa.String(length=64), nullable=True),
            sa.Column("support_chunk_edge_ids_json", sa.JSON(), nullable=True),
        ],
    )
    _add_columns(
        "mid_concept_edges",
        [
            sa.Column("distance", sa.Float(), nullable=True),
            sa.Column("raw_strength_summary_json", sa.JSON(), nullable=True),
            sa.Column("support_rq_prefix_node_ids_json", sa.JSON(), nullable=True),
        ],
    )
    _add_columns(
        "coarse_concept_edges",
        [
            sa.Column("distance", sa.Float(), nullable=True),
            sa.Column("raw_strength_summary_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("cross_community_weak_ties_json", sa.JSON(), nullable=True),
        ],
    )
    _add_columns(
        "retrieval_traces",
        [
            sa.Column("query_facets_json", sa.JSON(), nullable=True),
            sa.Column("entry_nodes_json", sa.JSON(), nullable=True),
            sa.Column("frontier_json", sa.JSON(), nullable=True),
            sa.Column("path_labels_json", sa.JSON(), nullable=True),
            sa.Column("convergence_json", sa.JSON(), nullable=True),
            sa.Column("edge_distance_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("edge_projection_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("traversal_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("conversation_state_scope_hash", sa.String(length=64), nullable=True),
        ],
    )
    _add_columns(
        "graph_retrieval_steps",
        [
            sa.Column("popped_frontier_state_json", sa.JSON(), nullable=True),
            sa.Column("expanded_edge_ids_json", sa.JSON(), nullable=True),
            sa.Column("dominance_pruned_count", sa.Integer(), nullable=True),
            sa.Column("cycle_reward", sa.Float(), nullable=True),
            sa.Column("ambiguous_edge_decisions_json", sa.JSON(), nullable=True),
            sa.Column("stop_reason", sa.String(length=96), nullable=True),
        ],
    )
    _add_columns(
        "context_packages",
        [
            sa.Column("graph_path_ids_json", sa.JSON(), nullable=True),
            sa.Column("why_selected_json", sa.JSON(), nullable=True),
            sa.Column("cycle_convergence_score", sa.Float(), nullable=True),
            sa.Column("dedupe_keys_json", sa.JSON(), nullable=True),
            sa.Column("covered_facets_json", sa.JSON(), nullable=True),
        ],
    )

    for table_name, columns in {
        "chunk_relation_edges": ["distance"],
        "rq_prefixes": ["node_type"],
        "rq_prefix_memberships": ["membership_role"],
        "mid_concept_edges": ["distance"],
        "coarse_concept_edges": ["distance"],
        "retrieval_traces": [
            "edge_distance_protocol_hash",
            "edge_projection_protocol_hash",
            "traversal_protocol_hash",
            "conversation_state_scope_hash",
        ],
        "graph_retrieval_steps": ["stop_reason"],
    }.items():
        for column in columns:
            _create_index_if_missing(table_name, f"ix_{table_name}_{column}", [column])


def downgrade() -> None:
    raise RuntimeError("Downgrade from traversal projection closure is intentionally unsupported.")
