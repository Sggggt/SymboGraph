"""Add residual quantized KMeans metadata to fine graph tables.

Revision ID: 20260614_0005
Revises: 20260614_0004
Create Date: 2026-06-14
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260614_0005"
down_revision = "20260614_0004"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    return {column["name"] for column in sa.inspect(bind).get_columns(table_name)}


def upgrade() -> None:
    chunk_columns = _columns("chunks")
    with op.batch_alter_table("chunks") as batch:
        if "rq_path" not in chunk_columns:
            batch.add_column(sa.Column("rq_path", sa.JSON(), nullable=True))
        if "rq_residual_norm" not in chunk_columns:
            batch.add_column(sa.Column("rq_residual_norm", sa.Float(), nullable=True))
    relation_edge_columns = _columns("chunk_relation_edges")
    with op.batch_alter_table("chunk_relation_edges") as batch:
        if "source_algorithm" not in relation_edge_columns:
            batch.add_column(sa.Column("source_algorithm", sa.String(length=64), nullable=True))
        if "protocol_version" not in relation_edge_columns:
            batch.add_column(sa.Column("protocol_version", sa.String(length=64), nullable=True))
        if "graph_state_hash" not in relation_edge_columns:
            batch.add_column(sa.Column("graph_state_hash", sa.String(length=64), nullable=True))
    fine_cluster_columns = _columns("fine_clusters")
    with op.batch_alter_table("fine_clusters") as batch:
        if "rq_level" not in fine_cluster_columns:
            batch.add_column(sa.Column("rq_level", sa.Integer(), nullable=True))
        if "rq_path_prefix" not in fine_cluster_columns:
            batch.add_column(sa.Column("rq_path_prefix", sa.JSON(), nullable=True))
        if "centroid_vector_ref" not in fine_cluster_columns:
            batch.add_column(sa.Column("centroid_vector_ref", sa.String(length=128), nullable=True))
    fine_membership_columns = _columns("fine_cluster_memberships")
    with op.batch_alter_table("fine_cluster_memberships") as batch:
        if "rq_path" not in fine_membership_columns:
            batch.add_column(sa.Column("rq_path", sa.JSON(), nullable=True))
        if "residual_norm" not in fine_membership_columns:
            batch.add_column(sa.Column("residual_norm", sa.Float(), nullable=True))
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    indexes = {
        index["name"]
        for table in ("chunks", "chunk_relation_edges", "fine_clusters", "fine_cluster_memberships")
        if table in tables
        for index in sa.inspect(bind).get_indexes(table)
    }
    if "ix_chunks_rq_residual_norm" not in indexes:
        op.create_index("ix_chunks_rq_residual_norm", "chunks", ["rq_residual_norm"])
    if "ix_chunk_relation_edges_source_algorithm" not in indexes:
        op.create_index("ix_chunk_relation_edges_source_algorithm", "chunk_relation_edges", ["source_algorithm"])
    if "ix_chunk_relation_edges_protocol_version" not in indexes:
        op.create_index("ix_chunk_relation_edges_protocol_version", "chunk_relation_edges", ["protocol_version"])
    if "ix_chunk_relation_edges_graph_state_hash" not in indexes:
        op.create_index("ix_chunk_relation_edges_graph_state_hash", "chunk_relation_edges", ["graph_state_hash"])
    if "ix_fine_clusters_rq_level" not in indexes:
        op.create_index("ix_fine_clusters_rq_level", "fine_clusters", ["rq_level"])
    if "ix_fine_clusters_centroid_vector_ref" not in indexes:
        op.create_index("ix_fine_clusters_centroid_vector_ref", "fine_clusters", ["centroid_vector_ref"])
    if "ix_fine_cluster_memberships_residual_norm" not in indexes:
        op.create_index("ix_fine_cluster_memberships_residual_norm", "fine_cluster_memberships", ["residual_norm"])


def downgrade() -> None:
    raise RuntimeError("Downgrade from RQ-KMeans graph fields is intentionally unsupported.")
