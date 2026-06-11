from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeBase(TimestampMixin, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_root: Mapped[str] = mapped_column(Text)
    current_chunk_version: Mapped[int] = mapped_column(Integer, default=0)
    active_profile_id: Mapped[str | None] = mapped_column(ForeignKey("strategy_profiles.id"), nullable=True, index=True)

    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base")
    batches: Mapped[list["IngestionBatch"]] = relationship(back_populates="knowledge_base")
    active_profile: Mapped["StrategyProfile | None"] = relationship(foreign_keys=[active_profile_id])


class SourceFile(TimestampMixin, Base):
    __tablename__ = "source_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    source_path: Mapped[str] = mapped_column(Text, index=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    source_type: Mapped[str] = mapped_column(String(50), default="unknown", index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)


class ParseJob(TimestampMixin, Base):
    __tablename__ = "parse_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    document_version_id: Mapped[str | None] = mapped_column(ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    ingestion_job_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    source_file_id: Mapped[str | None] = mapped_column(ForeignKey("source_files.id", ondelete="SET NULL"), nullable=True, index=True)
    parser_protocol_version: Mapped[str] = mapped_column(String(64), default="parser_v1")
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class StrategyProfile(TimestampMixin, Base):
    __tablename__ = "strategy_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    library_type: Mapped[str] = mapped_column(String(64), default="academic", index=True)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    profile_hash: Mapped[str] = mapped_column(String(64), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text, index=True)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(32), default="private")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version", name="uq_document_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    storage_path: Mapped[str] = mapped_column(Text)
    extracted_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="versions")


class EvidenceAtom(Base):
    __tablename__ = "evidence_atoms"
    __table_args__ = (UniqueConstraint("document_version_id", "atom_index", name="uq_evidence_atom_version_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    atom_index: Mapped[int] = mapped_column(Integer, index=True)
    atom_type: Mapped[str] = mapped_column(String(64), default="paragraph", index=True)
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_span_json: Mapped[dict] = mapped_column(JSON, default=dict)
    layout_json: Mapped[dict] = mapped_column(JSON, default=dict)
    parser_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvidenceGraphState(Base):
    __tablename__ = "evidence_graph_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    scope_type: Mapped[str] = mapped_column(String(32), default="document", index=True)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    atom_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    edge_protocol_version: Mapped[str] = mapped_column(String(64), default="deterministic_edges_v1", index=True)
    parser_protocol_version: Mapped[str] = mapped_column(String(64), default="parser_v1", index=True)
    embedding_text_version: Mapped[str] = mapped_column(String(64), default="metadata_enriched_v1", index=True)
    active_document_version_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    active_atom_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    community_state_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    policy_state_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="prompt_protocol_v1")
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvidenceEdge(Base):
    __tablename__ = "evidence_edges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    graph_state_id: Mapped[str] = mapped_column(ForeignKey("evidence_graph_states.id", ondelete="CASCADE"), index=True)
    source_atom_id: Mapped[str] = mapped_column(ForeignKey("evidence_atoms.id", ondelete="CASCADE"), index=True)
    target_atom_id: Mapped[str] = mapped_column(ForeignKey("evidence_atoms.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(64), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalSchemaState(Base):
    __tablename__ = "signal_schema_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    evidence_graph_state_id: Mapped[str] = mapped_column(ForeignKey("evidence_graph_states.id", ondelete="CASCADE"), index=True)
    evidence_community_state_id: Mapped[str | None] = mapped_column(ForeignKey("community_states.id", ondelete="SET NULL"), nullable=True, index=True)
    schema_hash: Mapped[str] = mapped_column(String(64), index=True)
    schema_protocol_version: Mapped[str] = mapped_column(String(64), default="signal_schema_induction_v1", index=True)
    sample_pack_hash: Mapped[str] = mapped_column(String(64), index=True)
    llm_model_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalTypeSpec(Base):
    __tablename__ = "signal_type_specs"
    __table_args__ = (UniqueConstraint("schema_state_id", "name", name="uq_signal_type_spec_schema_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    schema_state_id: Mapped[str] = mapped_column(ForeignKey("signal_schema_states.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(96), index=True)
    description: Mapped[str] = mapped_column(Text)
    evidence_patterns_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    applicable_atom_types_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    risk: Mapped[str] = mapped_column(String(32), default="medium", index=True)
    retrieval_use_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    gate_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalRelationSpec(Base):
    __tablename__ = "signal_relation_specs"
    __table_args__ = (UniqueConstraint("schema_state_id", "name", name="uq_signal_relation_spec_schema_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    schema_state_id: Mapped[str] = mapped_column(ForeignKey("signal_schema_states.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(96), index=True)
    source_signal_types_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    target_signal_types_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    required_evidence_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    risk: Mapped[str] = mapped_column(String(32), default="medium", index=True)
    gate_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalState(Base):
    __tablename__ = "signal_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    evidence_graph_state_id: Mapped[str] = mapped_column(ForeignKey("evidence_graph_states.id", ondelete="CASCADE"), index=True)
    evidence_community_state_id: Mapped[str | None] = mapped_column(ForeignKey("community_states.id", ondelete="SET NULL"), nullable=True, index=True)
    schema_state_id: Mapped[str | None] = mapped_column(ForeignKey("signal_schema_states.id", ondelete="SET NULL"), nullable=True, index=True)
    signal_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    schema_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    evidence_graph_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    evidence_community_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    signal_community_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    active_signal_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    signal_protocol_version: Mapped[str] = mapped_column(String(64), default="evidence_signal_layer_v1", index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    eligible_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    processed_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SignalCandidate(Base):
    __tablename__ = "signal_candidates"
    __table_args__ = (UniqueConstraint("signal_state_id", "normalized_key", "extractor_name", name="uq_signal_candidate_state_key_extractor"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    candidate_type: Mapped[str] = mapped_column(String(96), index=True)
    surface: Mapped[str] = mapped_column(String(255), index=True)
    normalized_key: Mapped[str] = mapped_column(String(320), index=True)
    evidence_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    support_active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    extractor_name: Mapped[str] = mapped_column(String(96), index=True)
    extractor_version: Mapped[str] = mapped_column(String(64), index=True)
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalDecision(Base):
    __tablename__ = "signal_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str | None] = mapped_column(ForeignKey("signal_candidates.id", ondelete="CASCADE"), nullable=True, index=True)
    decision_type: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    reason_code: Mapped[str] = mapped_column(String(96), index=True)
    support_evidence_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    algorithm_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    llm_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalNode(Base):
    __tablename__ = "signal_nodes"
    __table_args__ = (
        UniqueConstraint("signal_state_id", "normalized_key", name="uq_signal_node_state_key"),
        CheckConstraint("support_atom_ids_json IS NOT NULL", name="ck_signal_node_support_atoms_present"),
        CheckConstraint("source_span_union_json IS NOT NULL", name="ck_signal_node_source_span_present"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    signal_type: Mapped[str] = mapped_column(String(96), index=True)
    canonical_label: Mapped[str] = mapped_column(String(255), index=True)
    normalized_key: Mapped[str] = mapped_column(String(320), index=True)
    support_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalEdge(Base):
    __tablename__ = "signal_edges"
    __table_args__ = (UniqueConstraint("signal_state_id", "source_signal_id", "target_signal_id", "edge_type", name="uq_signal_edge_state_pair_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(96), index=True)
    source_signal_id: Mapped[str] = mapped_column(ForeignKey("signal_nodes.id", ondelete="CASCADE"), index=True)
    target_signal_id: Mapped[str] = mapped_column(ForeignKey("signal_nodes.id", ondelete="CASCADE"), index=True)
    support_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    relation_source: Mapped[str] = mapped_column(String(64), default="observed_signal_co_support", index=True)
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("signal_decisions.id", ondelete="SET NULL"), nullable=True, index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalCommunity(Base):
    __tablename__ = "signal_communities"
    __table_args__ = (UniqueConstraint("signal_state_id", "community_id", name="uq_signal_community_state_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    community_id: Mapped[str] = mapped_column(String(64), index=True)
    algorithm: Mapped[str] = mapped_column(String(64), default="support_overlap_components", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignalCommunityMembership(Base):
    __tablename__ = "signal_community_memberships"
    __table_args__ = (UniqueConstraint("signal_community_id", "signal_node_id", name="uq_signal_community_membership_node"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    signal_community_id: Mapped[str] = mapped_column(ForeignKey("signal_communities.id", ondelete="CASCADE"), index=True)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    signal_node_id: Mapped[str] = mapped_column(ForeignKey("signal_nodes.id", ondelete="CASCADE"), index=True)
    community_id: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProjectionState(Base):
    __tablename__ = "projection_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    signal_state_id: Mapped[str] = mapped_column(ForeignKey("signal_states.id", ondelete="CASCADE"), index=True)
    view: Mapped[str] = mapped_column(String(64), default="overview", index=True)
    projection_hash: Mapped[str] = mapped_column(String(64), index=True)
    projection_protocol_version: Mapped[str] = mapped_column(String(64), default="signal_projection_view_v1", index=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProjectionNode(Base):
    __tablename__ = "projection_nodes"
    __table_args__ = (UniqueConstraint("projection_state_id", "source_kind", "source_id", name="uq_projection_node_source"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    projection_state_id: Mapped[str] = mapped_column(ForeignKey("projection_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    source_kind: Mapped[str] = mapped_column(String(64), index=True)
    source_id: Mapped[str] = mapped_column(String(36), index=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    category: Mapped[str] = mapped_column(String(96), index=True)
    support_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProjectionEdge(Base):
    __tablename__ = "projection_edges"
    __table_args__ = (UniqueConstraint("projection_state_id", "source_node_id", "target_node_id", "edge_type", name="uq_projection_edge_pair_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    projection_state_id: Mapped[str] = mapped_column(ForeignKey("projection_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    source_node_id: Mapped[str] = mapped_column(ForeignKey("projection_nodes.id", ondelete="CASCADE"), index=True)
    target_node_id: Mapped[str] = mapped_column(ForeignKey("projection_nodes.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(96), index=True)
    support_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProjectionCommunity(Base):
    __tablename__ = "projection_communities"
    __table_args__ = (UniqueConstraint("projection_state_id", "community_id", name="uq_projection_community_state_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    projection_state_id: Mapped[str] = mapped_column(ForeignKey("projection_states.id", ondelete="CASCADE"), index=True)
    community_id: Mapped[str] = mapped_column(String(64), index=True)
    source_community_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    collapsed_node_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PolicyState(Base):
    __tablename__ = "policy_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    policy_family: Mapped[str] = mapped_column(String(64), default="constrained_linucb", index=True)
    policy_version: Mapped[str] = mapped_column(String(64), default="bandit_policy_v1", index=True)
    profile_objective_hash: Mapped[str] = mapped_column(String(64), index=True)
    posterior_json: Mapped[dict] = mapped_column(JSON, default=dict)
    constraints_json: Mapped[dict] = mapped_column(JSON, default=dict)
    exploration_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    drift_status: Mapped[str] = mapped_column(String(32), default="fresh", index=True)
    drift_detected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChunkCandidate(Base):
    __tablename__ = "chunk_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    graph_state_id: Mapped[str] = mapped_column(ForeignKey("evidence_graph_states.id", ondelete="CASCADE"), index=True)
    generator_name: Mapped[str] = mapped_column(String(128), index=True)
    generator_version: Mapped[str] = mapped_column(String(64), index=True)
    atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    graph_features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    cost_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback_driven: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QualityDecision(Base):
    __tablename__ = "quality_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    candidate_id: Mapped[str] = mapped_column(ForeignKey("chunk_candidates.id", ondelete="CASCADE"), index=True)
    policy_state_id: Mapped[str | None] = mapped_column(ForeignKey("policy_states.id", ondelete="SET NULL"), nullable=True, index=True)
    decision_action: Mapped[str] = mapped_column(String(64), default="answer_candidate", index=True)
    gate_passed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    risk_flags_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward_features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChunkDecision(Base):
    __tablename__ = "chunk_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    graph_state_id: Mapped[str] = mapped_column(ForeignKey("evidence_graph_states.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str] = mapped_column(ForeignKey("chunk_candidates.id", ondelete="CASCADE"), index=True)
    quality_decision_id: Mapped[str] = mapped_column(ForeignKey("quality_decisions.id", ondelete="CASCADE"), index=True)
    policy_state_id: Mapped[str | None] = mapped_column(ForeignKey("policy_states.id", ondelete="SET NULL"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64), default="activate", index=True)
    decision_protocol_version: Mapped[str] = mapped_column(String(64), default="chunk_decision_v1", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ActiveChunk(Base):
    __tablename__ = "active_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_decision_id: Mapped[str] = mapped_column(ForeignKey("chunk_decisions.id", ondelete="CASCADE"), index=True)
    document_version_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    graph_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    text: Mapped[str] = mapped_column(Text)
    source_span_union_json: Mapped[dict] = mapped_column(JSON, default=dict)
    boundary_policy_version: Mapped[str] = mapped_column(String(64), default="chunk_decision_v1")
    quality_decision_id: Mapped[str] = mapped_column(String(36), index=True)
    policy_state_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    community_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    @property
    def content(self) -> str:
        return self.text

    @property
    def snippet(self) -> str:
        metadata = self.metadata_json or {}
        return str(metadata.get("snippet") or self.text[:240])

    @property
    def document_id(self) -> str | None:
        return (self.metadata_json or {}).get("document_id")

    @property
    def document_version_id(self) -> str | None:
        return (self.metadata_json or {}).get("document_version_id")

    @property
    def partition(self) -> str | None:
        return (self.metadata_json or {}).get("partition")

    @property
    def section(self) -> str | None:
        return (self.metadata_json or {}).get("section")

    @property
    def source_type(self) -> str | None:
        return (self.metadata_json or {}).get("source_type")

    @property
    def page_number(self) -> int | None:
        return (self.metadata_json or {}).get("page_number")

    @property
    def parent_chunk_id(self) -> str | None:
        return (self.metadata_json or {}).get("parent_chunk_id")

    @property
    def chunk_version(self) -> int:
        try:
            return int((self.metadata_json or {}).get("chunk_version") or 0)
        except (TypeError, ValueError):
            return 0


class VectorRecord(Base):
    __tablename__ = "vector_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    active_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("active_chunks.id", ondelete="SET NULL"), nullable=True, index=True)
    qdrant_point_id: Mapped[str] = mapped_column(String(64), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_text_version: Mapped[str] = mapped_column(String(64), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64), index=True)
    vector_status: Mapped[str] = mapped_column(String(32), default="ready", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IngestionBatch(TimestampMixin, Base):
    __tablename__ = "ingestion_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    trigger_source: Mapped[str] = mapped_column(String(64), default="sync")
    source_root: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="batches")
    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="batch")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_batches.id"), nullable=True, index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger_source: Mapped[str] = mapped_column(String(64), default="upload")
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)

    batch: Mapped[IngestionBatch | None] = relationship(back_populates="jobs")


class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    batch_id: Mapped[str] = mapped_column(ForeignKey("ingestion_batches.id"), index=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class IngestionCompensationLog(Base):
    __tablename__ = "ingestion_compensation_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_jobs.id"), nullable=True, index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    operation: Mapped[str] = mapped_column(String(32), index=True)
    vector_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QASession(TimestampMixin, Base):
    __tablename__ = "qa_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[list[dict]] = mapped_column(JSON, default=list)


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieval_mode: Mapped[str] = mapped_column(String(64), default="evidence_first", index=True)
    active_chunk_scope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    evidence_graph_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    community_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    policy_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    prompt_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    result_active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    expansion_path_json: Mapped[dict] = mapped_column(JSON, default=dict)
    scores_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnswerSession(Base):
    __tablename__ = "answer_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    qa_session_id: Mapped[str | None] = mapped_column(ForeignKey("qa_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    citation_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="answer_grounding_v1", index=True)
    model_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CitationVerification(Base):
    __tablename__ = "citation_verifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    answer_session_id: Mapped[str | None] = mapped_column(ForeignKey("answer_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    active_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("active_chunks.id", ondelete="SET NULL"), nullable=True, index=True)
    claim_text: Mapped[str] = mapped_column(Text, default="")
    source_span_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    verdict: Mapped[str] = mapped_column(String(32), default="supported", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QualityObservation(Base):
    __tablename__ = "quality_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    quality_decision_id: Mapped[str | None] = mapped_column(ForeignKey("quality_decisions.id", ondelete="SET NULL"), nullable=True, index=True)
    observation_type: Mapped[str] = mapped_column(String(64), index=True)
    observation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CommunityState(Base):
    __tablename__ = "community_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    graph_state_id: Mapped[str | None] = mapped_column(ForeignKey("evidence_graph_states.id", ondelete="SET NULL"), nullable=True, index=True)
    community_protocol_version: Mapped[str] = mapped_column(String(64), default="local_community_v1", index=True)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CommunityMembership(Base):
    __tablename__ = "community_memberships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    community_state_id: Mapped[str] = mapped_column(ForeignKey("community_states.id", ondelete="CASCADE"), index=True)
    community_id: Mapped[str] = mapped_column(String(64), index=True)
    atom_id: Mapped[str] = mapped_column(ForeignKey("evidence_atoms.id", ondelete="CASCADE"), index=True)
    membership_score: Mapped[float] = mapped_column(Float, default=1.0)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CommunitySummary(Base):
    __tablename__ = "community_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    community_state_id: Mapped[str] = mapped_column(ForeignKey("community_states.id", ondelete="CASCADE"), index=True)
    community_id: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    citations_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    evidence_atom_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PolicyObservation(Base):
    __tablename__ = "policy_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    policy_state_id: Mapped[str | None] = mapped_column(ForeignKey("policy_states.id", ondelete="SET NULL"), nullable=True, index=True)
    context_json: Mapped[dict] = mapped_column(JSON, default=dict)
    action_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward_json: Mapped[dict] = mapped_column(JSON, default=dict)
    propensity: Mapped[float] = mapped_column(Float, default=1.0)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RewardEvent(Base):
    __tablename__ = "reward_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    policy_state_id: Mapped[str | None] = mapped_column(ForeignKey("policy_states.id", ondelete="SET NULL"), nullable=True, index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    answer_session_id: Mapped[str | None] = mapped_column(ForeignKey("answer_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    context_json: Mapped[dict] = mapped_column(JSON, default=dict)
    action_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward_json: Mapped[dict] = mapped_column(JSON, default=dict)
    propensity: Mapped[float] = mapped_column(Float, default=1.0)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PromptProtocolVersion(Base):
    __tablename__ = "prompt_protocol_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    protocol_name: Mapped[str] = mapped_column(String(128), index=True)
    protocol_version: Mapped[str] = mapped_column(String(64), index=True)
    protocol_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_pack_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RuntimeSettingsVersion(Base):
    __tablename__ = "runtime_settings_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    version_hash: Mapped[str] = mapped_column(String(64), index=True)
    settings_json: Mapped[dict] = mapped_column(JSON, default=dict)
    changed_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(64), default="api", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("qa_sessions.id"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    route: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    current_node: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentTraceEvent(Base):
    __tablename__ = "agent_trace_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    node: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    scores: Mapped[dict] = mapped_column(JSON, default=dict)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
