"""Upgrade the last pushed v7.1.2 schema to the current release schema.

Revision ID: 20260916_0045
Revises: 20260824_0044
"""
from __future__ import annotations

from collections import defaultdict

from alembic import op
import sqlalchemy as sa


revision = "20260916_0045"
down_revision = "20260824_0044"
branch_labels = None
depends_on = None


POST_PUSH_TABLES = frozenset({
    "answer_source_bindings",
    "context_package_source_retentions",
    "context_package_source_expansions",
    "retrieval_lexical_policies",
    "retrieval_lexical_rewards",
    "lexical_index_states",
    "lexical_documents",
    "lexical_terms",
    "lexical_postings",
    "lexical_index_jobs",
})

INDEX_RENAMES = (
    ("auto_tpe_runs", "ix_auto_tpe_runs_batch", "ix_auto_tpe_runs_batch_id", ("batch_id",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_relation_state", "ix_auto_tpe_runs_chunk_relation_graph_state_id", ("chunk_relation_graph_state_id",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_kb", "ix_auto_tpe_runs_knowledge_base_id", ("knowledge_base_id",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_selected_theta", "ix_auto_tpe_runs_selected_theta_hash", ("selected_theta_hash",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_adjacency", "ix_auto_tpe_trials_candidate_adjacency_hash", ("candidate_adjacency_hash",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_kb", "ix_auto_tpe_trials_knowledge_base_id", ("knowledge_base_id",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_objective", "ix_auto_tpe_trials_objective_score", ("objective_score",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_run", "ix_auto_tpe_trials_run_id", ("run_id",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_theta", "ix_auto_tpe_trials_theta_hash", ("theta_hash",)),
    ("chunk_relation_edges", "ix_cre_edge_distance_hash", "ix_chunk_relation_edges_edge_distance_protocol_hash", ("edge_distance_protocol_hash",)),
    ("chunk_relation_edges", "ix_cre_cross_document", "ix_chunk_relation_edges_is_cross_document", ("is_cross_document",)),
    ("chunk_relation_edges", "ix_cre_cross_language", "ix_chunk_relation_edges_is_cross_language", ("is_cross_language",)),
    ("chunk_relation_edges", "ix_cre_source_language", "ix_chunk_relation_edges_source_language", ("source_language",)),
    ("chunk_relation_edges", "ix_cre_target_language", "ix_chunk_relation_edges_target_language", ("target_language",)),
    ("chunk_relation_graph_states", "ix_crgs_auto_tpe_best_trial_id", "ix_chunk_relation_graph_states_auto_tpe_best_trial_id", ("auto_tpe_best_trial_id",)),
    ("chunk_relation_graph_states", "ix_crgs_auto_tpe_run_id", "ix_chunk_relation_graph_states_auto_tpe_run_id", ("auto_tpe_run_id",)),
    ("chunk_relation_graph_states", "ix_crgs_edge_distance_hash", "ix_chunk_relation_graph_states_edge_distance_protocol_hash", ("edge_distance_protocol_hash",)),
    ("chunk_relation_graph_states", "ix_crgs_edge_calibration_hash", "ix_chunk_relation_graph_states_edge_type_calibration_protocol_hash", ("edge_type_calibration_protocol_hash",)),
    ("chunk_relation_graph_states", "ix_crgs_operating_hash", "ix_chunk_relation_graph_states_graph_operating_point_hash", ("graph_operating_point_hash",)),
    ("chunk_relation_graph_states", "ix_crgs_runtime_settings_hash", "ix_chunk_relation_graph_states_runtime_settings_hash", ("runtime_settings_hash",)),
    ("graph_retrieval_steps", "ix_grs_action_type", "ix_graph_retrieval_steps_action_type", ("action_type",)),
    ("graph_retrieval_steps", "ix_grs_parent_layer", "ix_graph_retrieval_steps_parent_layer", ("parent_layer",)),
    ("graph_retrieval_steps", "ix_grs_parent_node", "ix_graph_retrieval_steps_parent_node_id", ("parent_node_id",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_created_at", "ix_rq_prefix_pair_diagnostics_created_at", ("created_at",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_diagnostic_hash", "ix_rq_prefix_pair_diagnostics_diagnostic_hash", ("diagnostic_hash",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_edge_type", "ix_rq_prefix_pair_diagnostics_edge_type", ("edge_type",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_graph_state_id", "ix_rq_prefix_pair_diagnostics_graph_state_id", ("graph_state_id",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_knowledge_base_id", "ix_rq_prefix_pair_diagnostics_knowledge_base_id", ("knowledge_base_id",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_protocol_version", "ix_rq_prefix_pair_diagnostics_protocol_version", ("protocol_version",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_source_algorithm", "ix_rq_prefix_pair_diagnostics_source_algorithm", ("source_algorithm",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_source_rq_prefix_id", "ix_rq_prefix_pair_diagnostics_source_rq_prefix_id", ("source_rq_prefix_id",)),
    ("rq_prefix_pair_diagnostics", "ix_rq_prefix_pair_target_rq_prefix_id", "ix_rq_prefix_pair_diagnostics_target_rq_prefix_id", ("target_rq_prefix_id",)),
)

NEW_INDEXES = (
    ("auto_tpe_runs", "ix_auto_tpe_runs_best_trial_id", ("best_trial_id",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_chat_model", ("chat_model",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_chunk_scope_hash", ("chunk_scope_hash",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_chunk_version", ("chunk_version",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_created_at", ("created_at",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_embedding_model", ("embedding_model",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_embedding_text_version", ("embedding_text_version",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_failure_code", ("failure_code",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_graph_operating_point_protocol", ("graph_operating_point_protocol",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_probe_set_hash", ("probe_set_hash",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_protocol_hash", ("protocol_hash",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_runtime_settings_hash", ("runtime_settings_hash",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_sampler_state_hash", ("sampler_state_hash",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_status", ("status",)),
    ("auto_tpe_runs", "ix_auto_tpe_runs_trigger_reason", ("trigger_reason",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_failure_code", ("failure_code",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_probe_set_hash", ("probe_set_hash",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_sampler_state_hash", ("sampler_state_hash",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_status", ("status",)),
    ("auto_tpe_trials", "ix_auto_tpe_trials_trial_index", ("trial_index",)),
    ("chunk_relation_edges", "ix_chunk_relation_edges_bridge_quota_reason", ("bridge_quota_reason",)),
    ("ingestion_batch_recoveries", "ix_ingestion_batch_recoveries_created_at", ("created_at",)),
    ("ingestion_file_stages", "ix_ingestion_file_stages_created_at", ("created_at",)),
    ("runtime_settings_audits", "ix_runtime_settings_audits_status", ("status",)),
    ("storage_maintenance_intents", "ix_storage_maintenance_intents_kb_status", ("knowledge_base_id", "status")),
)

NOT_NULL = {
    "auto_tpe_runs": ["selected_theta_json", "hard_gate_json", "objective_components_json", "blocking_reasons_json", "diagnostics_json", "created_at", "updated_at"],
    "auto_tpe_trials": ["sampled_theta_json", "hard_gate_json", "objective_components_json", "diagnostics_json"],
    "chunk_relation_edges": ["normalization_stats_json", "is_cross_document", "is_cross_language"],
    "chunk_relation_graph_states": ["graph_operating_point_json"],
    "graph_retrieval_steps": ["candidate_pool_ids_json", "selected_topk_ids_json", "per_parent_budget_status_json"],
    "retrieval_traces": ["stage_queues_json", "candidate_pools_json", "topk_selection_json"],
}


def _post_push_schema() -> tuple[sa.MetaData, tuple[sa.Table, ...]]:
    metadata = sa.MetaData()
    sa.Table("knowledge_bases", metadata, sa.Column("id", sa.String(36), primary_key=True))
    sa.Table(
        "chunks", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), nullable=False),
        sa.UniqueConstraint("id", "knowledge_base_id"),
    )
    for name in (
        "context_packages", "retrieval_traces", "answer_sessions",
        "agent_observations", "agent_runs", "document_versions",
        "ingestion_batches",
    ):
        sa.Table(name, metadata, sa.Column("id", sa.String(36), primary_key=True))

    answer_bindings = sa.Table(
        "answer_source_bindings", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("answer_session_id", sa.String(36), sa.ForeignKey("answer_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("context_package_id", sa.String(36), sa.ForeignKey("context_packages.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("retrieval_trace_id", sa.String(36), sa.ForeignKey("retrieval_traces.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chunk_id", sa.String(36), nullable=False, index=True),
        sa.Column("unit_id", sa.String(64), nullable=False, index=True),
        sa.Column("unit_index", sa.Integer(), nullable=False),
        sa.Column("unit_text", sa.Text(), nullable=False),
        sa.Column("answer_char_start", sa.Integer(), nullable=False),
        sa.Column("answer_char_end", sa.Integer(), nullable=False),
        sa.Column("answer_hash", sa.String(64), nullable=False, index=True),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("retrieval_gate_observation_id", sa.String(36), sa.ForeignKey("agent_observations.id", name="fk_answer_binding_retrieval_gate", ondelete="RESTRICT"), nullable=True, index=True),
        sa.Column("source_span_json", sa.JSON(), nullable=False),
        sa.Column("binding_hash", sa.String(64), nullable=False),
        sa.Column("diagnostics_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
        sa.CheckConstraint("protocol_version != 'answer_source_binding_v2' OR retrieval_gate_observation_id IS NOT NULL", name="ck_answer_binding_v2_gate_required"),
        sa.CheckConstraint("unit_index >= 0 AND unit_index < 32", name="ck_answer_source_binding_unit_index"),
        sa.CheckConstraint("answer_char_start >= 0 AND answer_char_end > answer_char_start", name="ck_answer_source_binding_answer_span"),
        sa.CheckConstraint("length(unit_id) = 64 AND length(answer_hash) = 64 AND length(binding_hash) = 64", name="ck_answer_source_binding_hash_lengths"),
        sa.CheckConstraint("protocol_version IN ('answer_source_binding_v1','answer_source_binding_v2')", name="ck_answer_source_binding_protocol"),
        sa.UniqueConstraint("answer_session_id", "unit_id", "chunk_id", name="uq_answer_source_binding_unit_chunk"),
        sa.UniqueConstraint("binding_hash", name="uq_answer_source_binding_hash"),
        sa.ForeignKeyConstraint(["chunk_id", "knowledge_base_id"], ["chunks.id", "chunks.knowledge_base_id"], name="fk_answer_source_binding_chunk_scope", ondelete="CASCADE"),
    )

    retentions = sa.Table(
        "context_package_source_retentions", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("target_context_package_id", sa.String(36), sa.ForeignKey("context_packages.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("source_context_package_id", sa.String(36), sa.ForeignKey("context_packages.id", deferrable=True, initially="DEFERRED"), nullable=False, index=True),
        sa.Column("source_retrieval_trace_id", sa.String(36), sa.ForeignKey("retrieval_traces.id", deferrable=True, initially="DEFERRED"), nullable=False, index=True),
        sa.Column("chunk_id", sa.String(36), nullable=False, index=True),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("source_item_hash", sa.String(64), nullable=False),
        sa.Column("retention_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("target_context_package_id", "chunk_id", name="uq_context_source_retention_chunk"),
        sa.ForeignKeyConstraint(["chunk_id", "knowledge_base_id"], ["chunks.id", "chunks.knowledge_base_id"], name="fk_context_source_retention_chunk_scope", ondelete="CASCADE"),
        sa.CheckConstraint("target_context_package_id <> source_context_package_id", name="ck_context_source_retention_not_self"),
        sa.CheckConstraint("protocol_version = 'reflection_bound_source_retention_v1'", name="ck_context_source_retention_protocol"),
        sa.CheckConstraint("length(source_item_hash) = 64 AND length(retention_hash) = 64", name="ck_context_source_retention_hashes"),
    )

    expansions = sa.Table(
        "context_package_source_expansions", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("target_context_package_id", sa.String(36), sa.ForeignKey("context_packages.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("source_context_package_id", sa.String(36), sa.ForeignKey("context_packages.id", deferrable=True, initially="DEFERRED"), nullable=False, index=True),
        sa.Column("source_retrieval_trace_id", sa.String(36), sa.ForeignKey("retrieval_traces.id", deferrable=True, initially="DEFERRED"), nullable=False, index=True),
        sa.Column("anchor_chunk_id", sa.String(36), nullable=False, index=True),
        sa.Column("chunk_id", sa.String(36), nullable=False, index=True),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("anchor_item_hash", sa.String(64), nullable=False),
        sa.Column("target_item_hash", sa.String(64), nullable=False),
        sa.Column("witness_json", sa.JSON(), nullable=False),
        sa.Column("witness_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("target_context_package_id", "chunk_id", name="uq_context_source_expansion_chunk"),
        sa.ForeignKeyConstraint(["chunk_id", "knowledge_base_id"], ["chunks.id", "chunks.knowledge_base_id"], name="fk_context_expansion_chunk_scope", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["anchor_chunk_id", "knowledge_base_id"], ["chunks.id", "chunks.knowledge_base_id"], name="fk_context_expansion_anchor_scope", deferrable=True, initially="DEFERRED"),
        sa.CheckConstraint("target_context_package_id <> source_context_package_id", name="ck_context_expansion_not_self"),
        sa.CheckConstraint("protocol_version = 'reflection_source_structure_expansion_v1'", name="ck_context_expansion_protocol"),
        sa.CheckConstraint("length(anchor_item_hash) = 64 AND length(target_item_hash) = 64 AND length(witness_hash) = 64", name="ck_context_expansion_hashes"),
    )

    lexical_policy = sa.Table(
        "retrieval_lexical_policies", metadata,
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("operation_counts_json", sa.JSON(), nullable=False),
        sa.Column("state_hash", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    lexical_reward = sa.Table(
        "retrieval_lexical_rewards", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("context_package_id", sa.String(36), sa.ForeignKey("context_packages.id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("observation_json", sa.JSON(), nullable=False),
        sa.Column("observation_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("run_id", "attempt_index", name="uq_retrieval_reward_attempt"),
        sa.CheckConstraint("attempt_index >= 0 AND attempt_index <= 2", name="ck_retrieval_reward_attempt"),
    )

    index_states = sa.Table(
        "lexical_index_states", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("predecessor_id", sa.String(36), sa.ForeignKey("lexical_index_states.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("tokenizer_protocol", sa.String(64), nullable=False),
        sa.Column("tokenizer_hash", sa.String(64), nullable=False),
        sa.Column("scoring_protocol", sa.String(64), nullable=False),
        sa.Column("scoring_hash", sa.String(64), nullable=False, index=True),
        sa.Column("bm25_k1", sa.Float(), nullable=False),
        sa.Column("bm25_b", sa.Float(), nullable=False),
        sa.Column("source_scope_hash", sa.String(64), nullable=False, index=True),
        sa.Column("statistics_hash", sa.String(64), nullable=False),
        sa.Column("postings_hash", sa.String(64), nullable=False),
        sa.Column("state_hash", sa.String(64), nullable=False, index=True),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("total_length", sa.Integer(), nullable=False),
        sa.Column("term_count", sa.Integer(), nullable=False),
        sa.Column("posting_count", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("id", "knowledge_base_id", name="uq_lexical_index_state_owner"),
        sa.CheckConstraint("state IN ('candidate','active','stale','failed')", name="ck_lexical_index_state"),
        sa.CheckConstraint("document_count >= 0 AND total_length >= 0 AND posting_count >= 0 AND term_count >= 0", name="ck_lexical_index_counts"),
        sa.Index("uq_lexical_index_active_kb", "knowledge_base_id", unique=True, postgresql_where=sa.text("state = 'active'"), sqlite_where=sa.text("state = 'active'")),
    )
    index_jobs = sa.Table(
        "lexical_index_jobs", metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("target_state_id", sa.String(36), nullable=False, unique=True),
        sa.Column("predecessor_state_id", sa.String(36), sa.ForeignKey("lexical_index_states.id", ondelete="RESTRICT"), nullable=True, index=True),
        sa.Column("ingestion_batch_id", sa.String(36), sa.ForeignKey("ingestion_batches.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("source_scope_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False, index=True),
        sa.Column("publish_intent", sa.Boolean(), nullable=False),
        sa.Column("completed_documents", sa.Integer(), nullable=False),
        sa.Column("completed_postings", sa.Integer(), nullable=False),
        sa.Column("cache_invalidation_pending", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(80), nullable=True),
        sa.Column("diagnostics_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["target_state_id", "knowledge_base_id"], ["lexical_index_states.id", "lexical_index_states.knowledge_base_id"], name="fk_lexical_job_target_owner", ondelete="CASCADE"),
        sa.CheckConstraint("status IN ('prepared','building','ready_to_publish','published','completed','failed','cancel_requested','cancelled')", name="ck_lexical_job_status"),
    )
    terms = sa.Table(
        "lexical_terms", metadata,
        sa.Column("index_state_id", sa.String(36), sa.ForeignKey("lexical_index_states.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("term_key", sa.String(64), primary_key=True),
        sa.Column("term", sa.Text(), nullable=False),
        sa.Column("document_frequency", sa.Integer(), nullable=False),
        sa.CheckConstraint("document_frequency > 0", name="ck_lexical_term_df"),
    )
    documents = sa.Table(
        "lexical_documents", metadata,
        sa.Column("index_state_id", sa.String(36), primary_key=True),
        sa.Column("chunk_id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), nullable=False, index=True),
        sa.Column("document_version_id", sa.String(36), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("raw_text_hash", sa.String(64), nullable=False),
        sa.Column("token_length", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["index_state_id", "knowledge_base_id"], ["lexical_index_states.id", "lexical_index_states.knowledge_base_id"], name="fk_lexical_document_index_owner", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id", "knowledge_base_id"], ["chunks.id", "chunks.knowledge_base_id"], name="fk_lexical_document_chunk_owner", ondelete="CASCADE"),
        sa.CheckConstraint("token_length >= 0 AND char_start >= 0 AND char_end >= char_start", name="ck_lexical_document_shape"),
    )
    postings = sa.Table(
        "lexical_postings", metadata,
        sa.Column("index_state_id", sa.String(36), primary_key=True),
        sa.Column("term_key", sa.String(64), primary_key=True),
        sa.Column("chunk_id", sa.String(36), primary_key=True),
        sa.Column("term_frequency", sa.Integer(), nullable=False),
        sa.Column("positions_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["index_state_id", "chunk_id"], ["lexical_documents.index_state_id", "lexical_documents.chunk_id"], name="fk_lexical_posting_document", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["index_state_id", "term_key"], ["lexical_terms.index_state_id", "lexical_terms.term_key"], name="fk_lexical_posting_term", ondelete="CASCADE"),
        sa.CheckConstraint("term_frequency > 0", name="ck_lexical_posting_tf"),
        sa.Index("ix_lexical_posting_document", "index_state_id", "chunk_id"),
    )
    return metadata, (
        answer_bindings, retentions, expansions, lexical_policy,
        lexical_reward, index_states, index_jobs, terms, documents, postings,
    )


def _index_name(name: str) -> str:
    preparer = op.get_bind().dialect.identifier_preparer
    return preparer.format_index(sa.Index(op.f(name), sa.column("id"))).strip('"')


def _indexes(table: str) -> dict[str, dict]:
    return {
        item["name"]: item
        for item in sa.inspect(op.get_bind()).get_indexes(table)
    }


def _validate_index(item: dict, columns: tuple[str, ...]) -> None:
    if item["unique"] or tuple(item["column_names"]) != tuple(columns):
        raise RuntimeError("release_schema_index_definition_mismatch")


def _rename_index(
    table: str,
    old: str,
    new: str,
    columns: tuple[str, ...],
) -> None:
    indexes = _indexes(table)
    old_name, new_name = _index_name(old), _index_name(new)
    if old_name not in indexes:
        if new_name not in indexes:
            raise RuntimeError("release_schema_expected_index_missing:" + table)
        _validate_index(indexes[new_name], columns)
        return
    _validate_index(indexes[old_name], columns)
    if new_name in indexes:
        raise RuntimeError("release_schema_duplicate_index_identity:" + table)
    if op.get_bind().dialect.name == "postgresql":
        quote = op.get_bind().dialect.identifier_preparer.quote
        op.execute(sa.text(
            "ALTER INDEX " + quote(old_name) + " RENAME TO " + quote(new_name)
        ))
    else:
        op.create_index(op.f(new), table, list(columns))
        op.drop_index(op.f(old), table_name=table)


def _change_nullability(nullable: bool) -> None:
    inspector = sa.inspect(op.get_bind())
    pending: dict[str, list[dict]] = defaultdict(list)
    for table, names in NOT_NULL.items():
        columns = {item["name"]: item for item in inspector.get_columns(table)}
        for name in names:
            column = columns[name]
            if column["nullable"] == nullable:
                continue
            if not nullable:
                relation = sa.table(table, sa.column(name))
                count = op.get_bind().scalar(
                    sa.select(sa.func.count())
                    .select_from(relation)
                    .where(relation.c[name].is_(None))
                )
                if count:
                    raise RuntimeError(
                        "release_schema_null_precondition:" + table + "." + name
                    )
            pending[table].append(column)
    for table, columns in pending.items():
        with op.batch_alter_table(table) as batch:
            for column in columns:
                batch.alter_column(
                    column["name"],
                    existing_type=column["type"],
                    nullable=nullable,
                )


def _converge_existing_schema() -> None:
    _change_nullability(False)
    for table, old, new, columns in INDEX_RENAMES:
        _rename_index(table, old, new, columns)
    for table, name, columns in NEW_INDEXES:
        existing = _indexes(table).get(_index_name(name))
        if existing is None:
            op.create_index(op.f(name), table, list(columns))
        else:
            _validate_index(existing, columns)


def upgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        op.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
        op.execute(sa.text("SET LOCAL statement_timeout = '120s'"))
    existing = set(sa.inspect(connection).get_table_names())
    conflicts = sorted(existing.intersection(POST_PUSH_TABLES))
    if conflicts:
        raise RuntimeError(
            "post_push_schema_tables_already_exist:" + ",".join(conflicts)
        )
    _converge_existing_schema()
    metadata, tables = _post_push_schema()
    metadata.create_all(
        bind=connection,
        tables=tables,
        checkfirst=False,
    )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    for table in sorted(POST_PUSH_TABLES):
        if table not in inspector.get_table_names():
            continue
        relation = sa.table(table)
        if connection.scalar(sa.select(sa.func.count()).select_from(relation)):
            raise RuntimeError("post_push_history_must_be_preserved:" + table)
    metadata, tables = _post_push_schema()
    metadata.drop_all(
        bind=connection,
        tables=tables,
        checkfirst=True,
    )
    for table, name, _columns in reversed(NEW_INDEXES):
        if _index_name(name) in _indexes(table):
            op.drop_index(op.f(name), table_name=table)
    for table, old, new, columns in reversed(INDEX_RENAMES):
        _rename_index(table, new, old, columns)
    _change_nullability(True)
