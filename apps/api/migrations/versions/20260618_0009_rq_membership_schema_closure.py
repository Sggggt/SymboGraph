"""Close RQ membership schema and remove legacy prefix-edge tables.

Revision ID: 20260618_0009
Revises: 20260617_0008
Create Date: 2026-06-18
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260618_0009"
down_revision = "20260617_0008"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _add_columns(table_name: str, columns: list[sa.Column]) -> None:
    if table_name not in _tables():
        return
    existing = _columns(table_name)
    with op.batch_alter_table(table_name) as batch:
        for column in columns:
            if column.name not in existing:
                batch.add_column(column)


def _drop_columns(table_name: str, column_names: list[str]) -> None:
    if table_name not in _tables():
        return
    existing = _columns(table_name)
    with op.batch_alter_table(table_name) as batch:
        for column_name in column_names:
            if column_name in existing:
                batch.drop_column(column_name)


def _rename_or_add(table_name: str, old_name: str, new_column: sa.Column) -> None:
    if table_name not in _tables():
        return
    existing = _columns(table_name)
    with op.batch_alter_table(table_name) as batch:
        if old_name in existing and new_column.name not in existing:
            batch.alter_column(old_name, new_column_name=new_column.name, existing_type=new_column.type)
        elif new_column.name not in existing:
            batch.add_column(new_column)
        if old_name in existing and new_column.name in existing:
            batch.drop_column(old_name)


def _drop_table_if_present(table_name: str) -> None:
    if table_name in _tables():
        bind = op.get_bind()
        if bind.dialect.name == "postgresql":
            quoted = bind.dialect.identifier_preparer.quote(table_name)
            bind.execute(sa.text(f"DROP TABLE IF EXISTS {quoted} CASCADE"))
        else:
            op.drop_table(table_name)


def upgrade() -> None:
    for table_name in ("rq_prefix_edges", "fine_cluster_edges", "fine_cluster_memberships", "fine_clusters"):
        _drop_table_if_present(table_name)

    if "rq_prefixes" not in _tables():
        op.create_table(
            "rq_prefixes",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("graph_state_id", sa.String(length=36), sa.ForeignKey("chunk_relation_graph_states.id", ondelete="CASCADE"), nullable=True),
            sa.Column("knowledge_base_id", sa.String(length=36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=True),
            sa.Column("rq_prefix_key", sa.String(length=128), nullable=True),
            sa.Column("label", sa.String(length=255), nullable=True),
            sa.Column("node_type", sa.String(length=64), nullable=True),
            sa.Column("centroid_json", sa.JSON(), nullable=True),
            sa.Column("rq_level", sa.Integer(), nullable=True),
            sa.Column("rq_path_prefix", sa.JSON(), nullable=True),
            sa.Column("parent_rq_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("codebook_version", sa.String(length=64), nullable=True),
            sa.Column("centroid_vector_ref", sa.String(length=128), nullable=True),
            sa.Column("representative_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("bridge_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("stats_json", sa.JSON(), nullable=True),
            sa.Column("diagnostics_json", sa.JSON(), nullable=True),
            sa.Column("state", sa.String(length=32), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
    if "rq_prefix_memberships" not in _tables():
        op.create_table(
            "rq_prefix_memberships",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("rq_prefix_id", sa.String(length=36), sa.ForeignKey("rq_prefixes.id", ondelete="CASCADE"), nullable=True),
            sa.Column("chunk_id", sa.String(length=36), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True),
            sa.Column("membership_score", sa.Float(), nullable=True),
            sa.Column("membership_role", sa.String(length=64), nullable=True),
            sa.Column("membership_reason", sa.String(length=64), nullable=True),
            sa.Column("membership_entropy", sa.Float(), nullable=True),
            sa.Column("rq_path", sa.JSON(), nullable=True),
            sa.Column("residual_norm", sa.Float(), nullable=True),
            sa.Column("rank", sa.Integer(), nullable=True),
            sa.Column("top_alternative_prefix_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_edge_ids_json", sa.JSON(), nullable=True),
            sa.Column("diagnostics_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )

    _rename_or_add("context_graph_states", "fine_cluster_hash", sa.Column("rq_membership_hash", sa.String(length=64), nullable=True))
    _rename_or_add("retrieval_traces", "fine_cluster_hash", sa.Column("rq_membership_hash", sa.String(length=64), nullable=True))
    _rename_or_add("rq_prefixes", "cluster_key", sa.Column("rq_prefix_key", sa.String(length=128), nullable=True))
    _rename_or_add("rq_prefix_memberships", "fine_cluster_id", sa.Column("rq_prefix_id", sa.String(length=36), nullable=True))
    _rename_or_add("mid_concept_memberships", "fine_cluster_id", sa.Column("rq_prefix_id", sa.String(length=36), nullable=True))

    _add_columns(
        "rq_prefixes",
        [
            sa.Column("parent_rq_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("codebook_version", sa.String(length=64), nullable=True),
        ],
    )
    _add_columns(
        "rq_prefix_memberships",
        [
            sa.Column("membership_entropy", sa.Float(), nullable=True),
            sa.Column("rank", sa.Integer(), nullable=True),
            sa.Column("top_alternative_prefix_ids_json", sa.JSON(), nullable=True),
        ],
    )
    if "rq_prefix_diagnostics" not in _tables():
        op.create_table(
            "rq_prefix_diagnostics",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("graph_state_id", sa.String(length=36), sa.ForeignKey("chunk_relation_graph_states.id", ondelete="CASCADE"), nullable=True),
            sa.Column("rq_prefix_id", sa.String(length=36), sa.ForeignKey("rq_prefixes.id", ondelete="CASCADE"), nullable=True),
            sa.Column("diagnostic_type", sa.String(length=96), nullable=True),
            sa.Column("diagnostic_strength", sa.Float(), nullable=True),
            sa.Column("support_membership_mass", sa.Float(), nullable=True),
            sa.Column("support_chunk_ids_sample_json", sa.JSON(), nullable=True),
            sa.Column("protocol_version", sa.String(length=64), nullable=True),
            sa.Column("diagnostics_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )

    _add_columns(
        "mid_concepts",
        [
            sa.Column("support_rq_l3_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("parent_rq_l2_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("parent_rq_l1_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("display_terms_json", sa.JSON(), nullable=True),
            sa.Column("internal_state_json", sa.JSON(), nullable=True),
            sa.Column("support_rq_prefix_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_edge_ids_json", sa.JSON(), nullable=True),
            sa.Column("core_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("boundary_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("bridge_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("outlier_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("raw_node_weight", sa.Float(), nullable=True),
            sa.Column("node_weight", sa.Float(), nullable=True),
            sa.Column("node_weight_normalization_scope", sa.String(length=64), nullable=True),
            sa.Column("node_weight_diagnostics_json", sa.JSON(), nullable=True),
        ],
    )
    _drop_columns("mid_concepts", ["support_fine_cluster_ids_json"])
    _add_columns(
        "mid_concept_edges",
        [
            sa.Column("projected_distance_raw", sa.Float(), nullable=True),
            sa.Column("projected_strength_raw", sa.Float(), nullable=True),
            sa.Column("projection_normalization_stats_json", sa.JSON(), nullable=True),
            sa.Column("edge_projection_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("support_rq_prefix_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_edge_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_rq_prefix_node_ids_json", sa.JSON(), nullable=True),
        ],
    )
    _drop_columns("mid_concept_edges", ["support_fine_edge_ids_json", "support_fine_node_ids_json", "support_rq_prefix_edge_ids_json"])

    _add_columns(
        "coarse_concepts",
        [
            sa.Column("support_rq_l2_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("parent_rq_l1_prefix_id", sa.String(length=36), nullable=True),
            sa.Column("child_rq_l3_prefix_ids_json", sa.JSON(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("scope_note", sa.Text(), nullable=True),
            sa.Column("inclusion_criteria_json", sa.JSON(), nullable=True),
            sa.Column("exclusion_criteria_json", sa.JSON(), nullable=True),
            sa.Column("display_terms_json", sa.JSON(), nullable=True),
            sa.Column("internal_state_json", sa.JSON(), nullable=True),
            sa.Column("outlier_mid_concept_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_ids_json", sa.JSON(), nullable=True),
            sa.Column("support_chunk_edge_ids_json", sa.JSON(), nullable=True),
            sa.Column("raw_node_weight", sa.Float(), nullable=True),
            sa.Column("node_weight", sa.Float(), nullable=True),
            sa.Column("node_weight_normalization_scope", sa.String(length=64), nullable=True),
            sa.Column("node_weight_diagnostics_json", sa.JSON(), nullable=True),
        ],
    )
    _add_columns(
        "coarse_concept_edges",
        [
            sa.Column("projected_distance_raw", sa.Float(), nullable=True),
            sa.Column("projected_strength_raw", sa.Float(), nullable=True),
            sa.Column("projection_normalization_stats_json", sa.JSON(), nullable=True),
            sa.Column("edge_projection_protocol_hash", sa.String(length=64), nullable=True),
            sa.Column("support_chunk_edge_ids_json", sa.JSON(), nullable=True),
        ],
    )
    _drop_columns("coarse_concept_edges", ["support_fine_edge_ids_json", "support_rq_prefix_edge_ids_json"])


def downgrade() -> None:
    raise RuntimeError("Downgrade from RQ membership schema closure is intentionally unsupported.")
