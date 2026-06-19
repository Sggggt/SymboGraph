from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeBase(TimestampMixin, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_root: Mapped[str] = mapped_column(Text)
    current_chunk_version: Mapped[int] = mapped_column(Integer, default=0, index=True)
    active_profile_id: Mapped[str | None] = mapped_column(ForeignKey("strategy_profiles.id", ondelete="SET NULL"), nullable=True, index=True)

    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base", cascade="all, delete-orphan")
    batches: Mapped[list["IngestionBatch"]] = relationship(back_populates="knowledge_base", cascade="all, delete-orphan")
    active_profile: Mapped["StrategyProfile | None"] = relationship(foreign_keys=[active_profile_id])


class StrategyProfile(TimestampMixin, Base):
    __tablename__ = "strategy_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    library_type: Mapped[str] = mapped_column(String(64), default="academic", index=True)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    profile_hash: Mapped[str] = mapped_column(String(64), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class SourceFile(TimestampMixin, Base):
    __tablename__ = "source_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    source_path: Mapped[str] = mapped_column(Text)
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
    parser_protocol_version: Mapped[str] = mapped_column(String(64), default="parser_v1", index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_kb_source", "knowledge_base_id", "source_path"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(32), default="private")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version", name="uq_document_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    storage_path: Mapped[str] = mapped_column(Text)
    extracted_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_protocol_version: Mapped[str] = mapped_column(String(64), default="parser_v1", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    document: Mapped["Document"] = relationship(back_populates="versions")


class ChunkVersion(Base):
    __tablename__ = "chunk_versions"
    __table_args__ = (UniqueConstraint("knowledge_base_id", "chunk_version", name="uq_chunk_version_kb_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_version: Mapped[int] = mapped_column(Integer, index=True)
    chunk_schema_version: Mapped[str] = mapped_column(String(64), default="chunk_schema_v1", index=True)
    tokenizer_version: Mapped[str] = mapped_column(String(64), default="symbograph_regex_tokenizer_v1", index=True)
    chunk_size: Mapped[int] = mapped_column(Integer, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=80)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_version", "chunk_index", name="uq_chunk_version_index"),
        Index("ix_chunks_kb_state_version", "knowledge_base_id", "state", "chunk_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    chunk_version: Mapped[int] = mapped_column(Integer, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, index=True)
    token_start: Mapped[int] = mapped_column(Integer, default=0)
    token_end: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    previous_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    next_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    rq_path: Mapped[list[int]] = mapped_column(JSON, default=list)
    rq_residual_norm: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChunkSpan(Base):
    __tablename__ = "chunk_spans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    token_start: Mapped[int] = mapped_column(Integer)
    token_end: Mapped[int] = mapped_column(Integer)
    span_type: Mapped[str] = mapped_column(String(64), default="raw_text", index=True)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ChunkCoordinate(Base):
    __tablename__ = "chunk_coordinates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    page_range_json: Mapped[dict] = mapped_column(JSON, default=dict)
    bbox_json: Mapped[dict] = mapped_column(JSON, default=dict)
    region_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    coordinate_system: Mapped[str] = mapped_column(String(64), default="parser_layout_v1")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)


class ChunkContextText(Base):
    __tablename__ = "chunk_context_texts"
    __table_args__ = (UniqueConstraint("chunk_id", "embedding_text_version", name="uq_chunk_context_text_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    contextual_text: Mapped[str] = mapped_column(Text)
    embedding_text_version: Mapped[str] = mapped_column(String(64), default="contextual_text_v1", index=True)
    context_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="contextual_text_v1", index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChunkStructureNode(Base):
    __tablename__ = "chunk_structure_nodes"
    __table_args__ = (Index("ix_structure_nodes_version_type", "document_version_id", "node_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    node_type: Mapped[str] = mapped_column(String(64), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    previous_sibling_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    next_sibling_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    bbox_json: Mapped[dict] = mapped_column(JSON, default=dict)
    layout_json: Mapped[dict] = mapped_column(JSON, default=dict)
    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChunkStructureEdge(Base):
    __tablename__ = "chunk_structure_edges"
    __table_args__ = (Index("ix_structure_edges_version_type", "document_version_id", "edge_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    source_node_id: Mapped[str] = mapped_column(ForeignKey("chunk_structure_nodes.id", ondelete="CASCADE"), index=True)
    target_node_id: Mapped[str] = mapped_column(ForeignKey("chunk_structure_nodes.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(64), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChunkStructureMapping(Base):
    __tablename__ = "chunk_structure_mappings"
    __table_args__ = (UniqueConstraint("chunk_id", "structure_node_id", name="uq_chunk_structure_mapping"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    structure_node_id: Mapped[str] = mapped_column(ForeignKey("chunk_structure_nodes.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    overlap_chars: Mapped[int] = mapped_column(Integer, default=0)
    overlap_tokens: Mapped[int] = mapped_column(Integer, default=0)
    coverage_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_intersection_json: Mapped[dict] = mapped_column(JSON, default=dict)
    mapping_role: Mapped[str] = mapped_column(String(64), default="overlap", index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ChunkRelationGraphState(Base):
    __tablename__ = "chunk_relation_graph_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_version: Mapped[int] = mapped_column(Integer, index=True)
    scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    graph_operating_point_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    graph_operating_point_json: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding_text_version: Mapped[str] = mapped_column(String(64), index=True)
    relation_protocol_version: Mapped[str] = mapped_column(String(64), default="chunk_relation_graph_v1", index=True)
    edge_distance_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    edge_type_calibration_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    active_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChunkRelationEdge(Base):
    __tablename__ = "chunk_relation_edges"
    __table_args__ = (Index("ix_chunk_relation_edges_graph_type", "graph_state_id", "edge_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    graph_state_id: Mapped[str] = mapped_column(ForeignKey("chunk_relation_graph_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    source_chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    target_chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(64), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    distance: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    raw_strength: Mapped[float] = mapped_column(Float, default=1.0)
    raw_strength_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    normalization_stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    support_json: Mapped[dict] = mapped_column(JSON, default=dict)
    source_algorithm: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    protocol_version: Mapped[str] = mapped_column(String(64), default="chunk_relation_graph_v1", index=True)
    edge_distance_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    graph_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_language: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    target_language: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    is_cross_document: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_cross_language: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    bridge_quota_reason: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    is_bridge: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RQPrefix(Base):
    __tablename__ = "rq_prefixes"
    __table_args__ = (UniqueConstraint("graph_state_id", "rq_prefix_key", name="uq_rq_prefix_state_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    graph_state_id: Mapped[str] = mapped_column(ForeignKey("chunk_relation_graph_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    rq_prefix_key: Mapped[str] = mapped_column(String(128), index=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    node_type: Mapped[str] = mapped_column(String(64), default="rq_prefix", index=True)
    centroid_json: Mapped[list[float]] = mapped_column(JSON, default=list)
    rq_level: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    rq_path_prefix: Mapped[list[int]] = mapped_column(JSON, default=list)
    parent_rq_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="SET NULL"), nullable=True, index=True)
    codebook_version: Mapped[str] = mapped_column(String(64), default="residual_quantized_kmeans_v1", index=True)
    centroid_vector_ref: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    representative_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    bridge_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RQPrefixMembership(Base):
    __tablename__ = "rq_prefix_memberships"
    __table_args__ = (UniqueConstraint("rq_prefix_id", "chunk_id", name="uq_rq_prefix_chunk"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    rq_prefix_id: Mapped[str] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="CASCADE"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    membership_score: Mapped[float] = mapped_column(Float, default=1.0)
    membership_role: Mapped[str] = mapped_column(String(64), default="member", index=True)
    membership_reason: Mapped[str] = mapped_column(String(64), default="centroid", index=True)
    membership_entropy: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    rq_path: Mapped[list[int]] = mapped_column(JSON, default=list)
    residual_norm: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    rank: Mapped[int] = mapped_column(Integer, default=0, index=True)
    top_alternative_prefix_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RQPrefixDiagnostic(Base):
    __tablename__ = "rq_prefix_diagnostics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    graph_state_id: Mapped[str] = mapped_column(ForeignKey("chunk_relation_graph_states.id", ondelete="CASCADE"), index=True)
    rq_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="CASCADE"), nullable=True, index=True)
    diagnostic_type: Mapped[str] = mapped_column(String(96), index=True)
    diagnostic_strength: Mapped[float] = mapped_column(Float, default=0.0)
    support_membership_mass: Mapped[float] = mapped_column(Float, default=0.0)
    support_chunk_ids_sample_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    protocol_version: Mapped[str] = mapped_column(String(64), default="rq_membership_diagnostics_v1", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class MidConceptState(Base):
    __tablename__ = "mid_concept_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_relation_graph_state_id: Mapped[str | None] = mapped_column(ForeignKey("chunk_relation_graph_states.id", ondelete="SET NULL"), nullable=True, index=True)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    grounding_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="mid_concept_definition_v1", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class MidConcept(Base):
    __tablename__ = "mid_concepts"
    __table_args__ = (UniqueConstraint("concept_state_id", "canonical_label", name="uq_mid_concept_state_label"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    concept_state_id: Mapped[str] = mapped_column(ForeignKey("mid_concept_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    canonical_label: Mapped[str] = mapped_column(String(255), index=True)
    aliases_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_rq_l3_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="SET NULL"), nullable=True, index=True)
    parent_rq_l2_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="SET NULL"), nullable=True, index=True)
    parent_rq_l1_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="SET NULL"), nullable=True, index=True)
    definition: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    scope_note: Mapped[str] = mapped_column(Text, default="")
    inclusion_criteria_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    exclusion_criteria_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    display_terms_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    internal_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    representative_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_rq_prefix_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    core_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    boundary_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    bridge_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    outlier_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    raw_node_weight: Mapped[float] = mapped_column(Float, default=0.0)
    node_weight: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    node_weight_normalization_scope: Mapped[str] = mapped_column(String(64), default="mid_concept_state", index=True)
    node_weight_diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    llm_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    grounding_hash: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class MidConceptMembership(Base):
    __tablename__ = "mid_concept_memberships"
    __table_args__ = (UniqueConstraint("mid_concept_id", "rq_prefix_id", name="uq_mid_concept_rq_prefix"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    mid_concept_id: Mapped[str] = mapped_column(ForeignKey("mid_concepts.id", ondelete="CASCADE"), index=True)
    rq_prefix_id: Mapped[str] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="CASCADE"), index=True)
    membership_score: Mapped[float] = mapped_column(Float, default=1.0)
    support_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class MidConceptEdge(Base):
    __tablename__ = "mid_concept_edges"
    __table_args__ = (Index("ix_mid_concept_edges_state_type", "concept_state_id", "edge_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    concept_state_id: Mapped[str] = mapped_column(ForeignKey("mid_concept_states.id", ondelete="CASCADE"), index=True)
    source_concept_id: Mapped[str] = mapped_column(ForeignKey("mid_concepts.id", ondelete="CASCADE"), index=True)
    target_concept_id: Mapped[str] = mapped_column(ForeignKey("mid_concepts.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(64), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    distance: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    projected_distance_raw: Mapped[float] = mapped_column(Float, default=0.0)
    projected_strength_raw: Mapped[float] = mapped_column(Float, default=0.0)
    raw_strength_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    projection_normalization_stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    edge_projection_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    network_evidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    llm_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    support_rq_prefix_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_relation_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_rq_prefix_node_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="")
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class MidConceptDefinition(Base):
    __tablename__ = "mid_concept_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    mid_concept_id: Mapped[str] = mapped_column(ForeignKey("mid_concepts.id", ondelete="CASCADE"), index=True)
    definition_version: Mapped[str] = mapped_column(String(64), default="v1", index=True)
    definition_json: Mapped[dict] = mapped_column(JSON, default=dict)
    support_spans_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    llm_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CoarseConceptState(Base):
    __tablename__ = "coarse_concept_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    mid_concept_state_id: Mapped[str | None] = mapped_column(ForeignKey("mid_concept_states.id", ondelete="SET NULL"), nullable=True, index=True)
    community_protocol_version: Mapped[str] = mapped_column(String(64), default="bridge_aware_community_v1", index=True)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    grounding_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="coarse_concept_definition_v1", index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CoarseConcept(Base):
    __tablename__ = "coarse_concepts"
    __table_args__ = (UniqueConstraint("coarse_state_id", "canonical_label", name="uq_coarse_concept_state_label"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    coarse_state_id: Mapped[str] = mapped_column(ForeignKey("coarse_concept_states.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    canonical_label: Mapped[str] = mapped_column(String(255), index=True)
    aliases_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_rq_l2_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="SET NULL"), nullable=True, index=True)
    parent_rq_l1_prefix_id: Mapped[str | None] = mapped_column(ForeignKey("rq_prefixes.id", ondelete="SET NULL"), nullable=True, index=True)
    child_rq_l3_prefix_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    definition: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    scope_note: Mapped[str] = mapped_column(Text, default="")
    inclusion_criteria_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    exclusion_criteria_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    display_terms_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    internal_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    included_mid_concept_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    boundary_mid_concept_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    bridge_mid_concept_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    outlier_mid_concept_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    cross_community_weak_ties_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    raw_node_weight: Mapped[float] = mapped_column(Float, default=0.0)
    node_weight: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    node_weight_normalization_scope: Mapped[str] = mapped_column(String(64), default="coarse_concept_state", index=True)
    node_weight_diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    llm_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    grounding_hash: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CoarseConceptMembership(Base):
    __tablename__ = "coarse_concept_memberships"
    __table_args__ = (UniqueConstraint("coarse_concept_id", "mid_concept_id", name="uq_coarse_mid_membership"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    coarse_concept_id: Mapped[str] = mapped_column(ForeignKey("coarse_concepts.id", ondelete="CASCADE"), index=True)
    mid_concept_id: Mapped[str] = mapped_column(ForeignKey("mid_concepts.id", ondelete="CASCADE"), index=True)
    membership_score: Mapped[float] = mapped_column(Float, default=1.0)
    role: Mapped[str] = mapped_column(String(64), default="included", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CoarseConceptEdge(Base):
    __tablename__ = "coarse_concept_edges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    coarse_state_id: Mapped[str] = mapped_column(ForeignKey("coarse_concept_states.id", ondelete="CASCADE"), index=True)
    source_concept_id: Mapped[str] = mapped_column(ForeignKey("coarse_concepts.id", ondelete="CASCADE"), index=True)
    target_concept_id: Mapped[str] = mapped_column(ForeignKey("coarse_concepts.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(64), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    distance: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    projected_distance_raw: Mapped[float] = mapped_column(Float, default=0.0)
    projected_strength_raw: Mapped[float] = mapped_column(Float, default=0.0)
    raw_strength_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    projection_normalization_stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    edge_projection_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    support_mid_concept_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_mid_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    cross_community_weak_ties_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="")
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CoarseConceptDefinition(Base):
    __tablename__ = "coarse_concept_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    coarse_concept_id: Mapped[str] = mapped_column(ForeignKey("coarse_concepts.id", ondelete="CASCADE"), index=True)
    definition_version: Mapped[str] = mapped_column(String(64), default="v1", index=True)
    definition_json: Mapped[dict] = mapped_column(JSON, default=dict)
    support_spans_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    llm_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ContextGraphState(Base):
    __tablename__ = "context_graph_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_relation_graph_state_id: Mapped[str | None] = mapped_column(ForeignKey("chunk_relation_graph_states.id", ondelete="SET NULL"), nullable=True, index=True)
    mid_concept_state_id: Mapped[str | None] = mapped_column(ForeignKey("mid_concept_states.id", ondelete="SET NULL"), nullable=True, index=True)
    coarse_concept_state_id: Mapped[str | None] = mapped_column(ForeignKey("coarse_concept_states.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    structure_graph_hash: Mapped[str] = mapped_column(String(64), index=True)
    chunk_relation_graph_hash: Mapped[str] = mapped_column(String(64), index=True)
    rq_membership_hash: Mapped[str] = mapped_column(String(64), index=True)
    mid_concept_hash: Mapped[str] = mapped_column(String(64), index=True)
    coarse_concept_hash: Mapped[str] = mapped_column(String(64), index=True)
    context_graph_hash: Mapped[str] = mapped_column(String(64), index=True)
    runtime_settings_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    agent_operating_envelope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    policy_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    prompt_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ContextGraphFreshness(Base):
    __tablename__ = "context_graph_freshness"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    context_graph_state_id: Mapped[str | None] = mapped_column(ForeignKey("context_graph_states.id", ondelete="CASCADE"), nullable=True, index=True)
    layer: Mapped[str] = mapped_column(String(64), index=True)
    state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    stale_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    checked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)


class VectorRecord(Base):
    __tablename__ = "vector_records"
    __table_args__ = (UniqueConstraint("chunk_id", "embedding_text_version", name="uq_vector_chunk_text_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    qdrant_point_id: Mapped[str] = mapped_column(String(64), index=True)
    collection_name: Mapped[str] = mapped_column(String(255), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=0)
    embedding_text_version: Mapped[str] = mapped_column(String(64), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64), index=True)
    vector_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieval_mode: Mapped[str] = mapped_column(String(64), default="layered_context_graph", index=True)
    chunk_scope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    structure_graph_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chunk_relation_graph_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    rq_membership_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    mid_concept_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    coarse_concept_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    runtime_settings_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    agent_operating_envelope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    policy_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    prompt_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    result_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    concept_path_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    scores_json: Mapped[dict] = mapped_column(JSON, default=dict)
    query_facets_json: Mapped[dict] = mapped_column(JSON, default=dict)
    entry_nodes_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    frontier_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    stage_queues_json: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_pools_json: Mapped[dict] = mapped_column(JSON, default=dict)
    topk_selection_json: Mapped[dict] = mapped_column(JSON, default=dict)
    path_labels_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    convergence_json: Mapped[dict] = mapped_column(JSON, default=dict)
    edge_distance_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    edge_projection_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    traversal_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    conversation_state_scope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class GraphRetrievalStep(Base):
    __tablename__ = "graph_retrieval_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    retrieval_trace_id: Mapped[str] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    step_index: Mapped[int] = mapped_column(Integer, index=True)
    layer: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(96), index=True)
    action_type: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    parent_layer: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    parent_node_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    input_json: Mapped[dict] = mapped_column(JSON, default=dict)
    output_json: Mapped[dict] = mapped_column(JSON, default=dict)
    score_json: Mapped[dict] = mapped_column(JSON, default=dict)
    popped_frontier_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    expanded_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    candidate_pool_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    selected_topk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    dominance_pruned_count: Mapped[int] = mapped_column(Integer, default=0)
    cycle_distance_reward: Mapped[float] = mapped_column(Float, default=0.0)
    gray_zone_path_decisions_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    per_parent_budget_status_json: Mapped[dict] = mapped_column(JSON, default=dict)
    stop_reason: Mapped[str] = mapped_column(String(96), default="", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ContextPackage(Base):
    __tablename__ = "context_packages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text, default="")
    hit_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    restored_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    bridge_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    parent_structure_node_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    concept_path_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    package_json: Mapped[dict] = mapped_column(JSON, default=dict)
    graph_path_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    why_selected_json: Mapped[dict] = mapped_column(JSON, default=dict)
    cycle_convergence_score: Mapped[float] = mapped_column(Float, default=0.0)
    dedupe_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    covered_facets_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    token_budget: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    runtime_settings_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    profile_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    citation_spans_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class QASession(TimestampMixin, Base):
    __tablename__ = "qa_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[list[dict]] = mapped_column(JSON, default=list)


class AnswerSession(Base):
    __tablename__ = "answer_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    context_package_id: Mapped[str | None] = mapped_column(ForeignKey("context_packages.id", ondelete="SET NULL"), nullable=True, index=True)
    qa_session_id: Mapped[str | None] = mapped_column(ForeignKey("qa_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    citation_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="context_graph_answer_v1", index=True)
    model_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CitationVerification(Base):
    __tablename__ = "citation_verifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    answer_session_id: Mapped[str | None] = mapped_column(ForeignKey("answer_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    context_package_id: Mapped[str | None] = mapped_column(ForeignKey("context_packages.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True, index=True)
    claim_text: Mapped[str] = mapped_column(Text, default="")
    source_span_json: Mapped[dict] = mapped_column(JSON, default=dict)
    verdict: Mapped[str] = mapped_column(String(32), default="supported", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class PolicyState(Base):
    __tablename__ = "policy_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    policy_family: Mapped[str] = mapped_column(String(64), default="context_graph_bandit", index=True)
    policy_version: Mapped[str] = mapped_column(String(64), default="context_graph_bandit_v1", index=True)
    profile_objective_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    weights_json: Mapped[dict] = mapped_column(JSON, default=dict)
    constraints_json: Mapped[dict] = mapped_column(JSON, default=dict)
    exploration_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RewardEvent(Base):
    __tablename__ = "reward_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    policy_state_id: Mapped[str | None] = mapped_column(ForeignKey("policy_states.id", ondelete="SET NULL"), nullable=True, index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    answer_session_id: Mapped[str | None] = mapped_column(ForeignKey("answer_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    context_json: Mapped[dict] = mapped_column(JSON, default=dict)
    action_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward_json: Mapped[dict] = mapped_column(JSON, default=dict)
    propensity: Mapped[float] = mapped_column(Float, default=1.0)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class PromptProtocolVersion(Base):
    __tablename__ = "prompt_protocol_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    protocol_name: Mapped[str] = mapped_column(String(128), index=True)
    protocol_version: Mapped[str] = mapped_column(String(64), index=True)
    protocol_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_pack_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RuntimeSettingsVersion(Base):
    __tablename__ = "runtime_settings_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    version_hash: Mapped[str] = mapped_column(String(64), index=True)
    settings_json: Mapped[dict] = mapped_column(JSON, default=dict)
    changed_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(64), default="api", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class IngestionBatch(TimestampMixin, Base):
    __tablename__ = "ingestion_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
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
    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="batch", cascade="all, delete-orphan")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_batches.id", ondelete="SET NULL"), nullable=True, index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger_source: Mapped[str] = mapped_column(String(64), default="upload")
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)

    batch: Mapped[IngestionBatch | None] = relationship(back_populates="jobs")


class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    batch_id: Mapped[str] = mapped_column(ForeignKey("ingestion_batches.id", ondelete="CASCADE"), index=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class IngestionCompensationLog(Base):
    __tablename__ = "ingestion_compensation_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    operation: Mapped[str] = mapped_column(String(64), index=True)
    target_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("qa_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    route: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    current_node: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentTraceEvent(Base):
    __tablename__ = "agent_trace_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    node: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    scores: Mapped[dict] = mapped_column(JSON, default=dict)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class AgentPlan(Base):
    __tablename__ = "agent_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    retrieval_trace_id: Mapped[str | None] = mapped_column(ForeignKey("retrieval_traces.id", ondelete="SET NULL"), nullable=True, index=True)
    plan_index: Mapped[int] = mapped_column(Integer, default=0, index=True)
    planner_model_json: Mapped[dict] = mapped_column(JSON, default=dict)
    query_intent_json: Mapped[dict] = mapped_column(JSON, default=dict)
    envelope_json: Mapped[dict] = mapped_column(JSON, default=dict)
    typed_actions_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    validation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="planned", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    plan_id: Mapped[str | None] = mapped_column(ForeignKey("agent_plans.id", ondelete="CASCADE"), nullable=True, index=True)
    parent_action_id: Mapped[str | None] = mapped_column(ForeignKey("agent_actions.id", ondelete="SET NULL"), nullable=True, index=True)
    action_index: Mapped[int] = mapped_column(Integer, default=0, index=True)
    action_type: Mapped[str] = mapped_column(String(96), index=True)
    target_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    reason: Mapped[str] = mapped_column(Text, default="")
    budget_request_json: Mapped[dict] = mapped_column(JSON, default=dict)
    expected_evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)
    stop_condition_json: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    output_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class AgentObservation(Base):
    __tablename__ = "agent_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    action_id: Mapped[str | None] = mapped_column(ForeignKey("agent_actions.id", ondelete="CASCADE"), nullable=True, index=True)
    observation_type: Mapped[str] = mapped_column(String(96), index=True)
    observation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    verdict: Mapped[str] = mapped_column(String(32), default="observed", index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
