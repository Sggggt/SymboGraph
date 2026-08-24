from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def generate_unmanaged_source_slot_key() -> str:
    """Keep direct ORM fixtures distinct without granting them a managed slot."""

    return f"unmanaged:{uuid.uuid4().hex}"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeBase(TimestampMixin, Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        CheckConstraint(
            "lifecycle_status IN ('active','deleting','delete_manual_review')",
            name="ck_knowledge_bases_lifecycle_status",
        ),
        UniqueConstraint(
            "source_root",
            name="uq_knowledge_bases_source_root",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_root: Mapped[str] = mapped_column(Text)
    lifecycle_status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        server_default="active",
        index=True,
    )
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
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "logical_source_slot_key",
            name="uq_source_files_kb_logical_source_slot",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    source_path: Mapped[str] = mapped_column(Text)
    logical_source_slot_key: Mapped[str] = mapped_column(
        String(1024),
        default=generate_unmanaged_source_slot_key,
    )
    source_slot_protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="unmanaged_opaque_v1",
        index=True,
    )
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
    __table_args__ = (
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_documents_id_knowledge_base_id",
        ),
        CheckConstraint(
            "language_confidence IS NULL OR (language_confidence >= 0 AND language_confidence <= 1)",
            name="ck_documents_language_confidence_range",
        ),
        CheckConstraint(
            "language_source IS NULL OR language_source IN ('explicit_metadata','deterministic_detection','unknown')",
            name="ck_documents_language_source",
        ),
        UniqueConstraint(
            "knowledge_base_id",
            "logical_source_slot_key",
            name="uq_documents_kb_logical_source_slot",
        ),
        Index("ix_documents_kb_source", "knowledge_base_id", "source_path"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text)
    logical_source_slot_key: Mapped[str] = mapped_column(
        String(1024),
        default=generate_unmanaged_source_slot_key,
    )
    source_slot_protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="unmanaged_opaque_v1",
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    language_source: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    language_detection_protocol_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    language_detection_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    language_metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(32), default="private")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    # A selected-file reparse stays on the knowledge-base chunk version.  Keep
    # each parse attempt as a separate row so a failed attempt can
    # reactivate the exact pre-parse version instead of overwriting it.
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_document_versions_version_positive"),
        CheckConstraint(
            "language_confidence IS NULL OR (language_confidence >= 0 AND language_confidence <= 1)",
            name="ck_document_versions_language_confidence_range",
        ),
        CheckConstraint(
            "language_source IS NULL OR language_source IN ('explicit_metadata','deterministic_detection','unknown')",
            name="ck_document_versions_language_source",
        ),
        UniqueConstraint(
            "id",
            "document_id",
            "version",
            name="uq_document_versions_id_document_id_version",
        ),
        Index("ix_document_versions_document_version", "document_id", "version"),
        Index(
            "uq_document_versions_one_active_per_document",
            "document_id",
            unique=True,
            postgresql_where=text("is_active IS TRUE"),
            sqlite_where=text("is_active = 1"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    storage_path: Mapped[str] = mapped_column(Text)
    extracted_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_protocol_version: Mapped[str] = mapped_column(String(64), default="parser_v1", index=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    language_source: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    language_detection_protocol_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    language_detection_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    language_metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    document: Mapped["Document"] = relationship(back_populates="versions")


class ChunkVersion(Base):
    __tablename__ = "chunk_versions"
    __table_args__ = (
        CheckConstraint("chunk_version >= 1", name="ck_chunk_versions_chunk_version_positive"),
        UniqueConstraint("knowledge_base_id", "chunk_version", name="uq_chunk_version_kb_version"),
    )

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
        CheckConstraint("chunk_version >= 1", name="ck_chunks_chunk_version_positive"),
        UniqueConstraint("id", "knowledge_base_id", name="uq_chunks_id_knowledge_base_id"),
        UniqueConstraint("document_version_id", "chunk_version", "chunk_index", name="uq_chunk_version_index"),
        ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_chunks_document_knowledge_base_provenance",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["document_version_id", "document_id", "chunk_version"],
            ["document_versions.id", "document_versions.document_id", "document_versions.version"],
            name="fk_chunks_document_version_provenance",
            ondelete="CASCADE",
        ),
        Index("ix_chunks_kb_state_version", "knowledge_base_id", "state", "chunk_version"),
        Index(
            "uq_chunks_active_document_version_index",
            "document_id",
            "chunk_version",
            "chunk_index",
            unique=True,
            postgresql_where=text("state = 'active'"),
            sqlite_where=text("state = 'active'"),
        ),
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
    embedding_text_version: Mapped[str] = mapped_column(String(64), default="contextual_text_v2", index=True)
    context_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_protocol_version: Mapped[str] = mapped_column(String(64), default="contextual_text_v2", index=True)
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
    span_overlap: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_iou: Mapped[float | None] = mapped_column(Float, nullable=True)
    path_match: Mapped[float | None] = mapped_column(Float, nullable=True)
    mapping_weight: Mapped[float] = mapped_column(Float, default=0.0)
    mapping_protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="structure_mapping_span_bbox_path_v2",
        index=True,
    )
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
    runtime_settings_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    auto_tpe_run_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "auto_tpe_runs.id",
            name="fk_chunk_relation_graph_states_auto_tpe_run_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
    auto_tpe_best_trial_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "auto_tpe_trials.id",
            name="fk_chunk_relation_graph_states_auto_tpe_best_trial_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
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


class RQPrefixPairDiagnostic(Base):
    __tablename__ = "rq_prefix_pair_diagnostics"
    __table_args__ = (
        UniqueConstraint(
            "graph_state_id",
            "source_rq_prefix_id",
            "target_rq_prefix_id",
            "edge_type",
            name="uq_rq_prefix_pair_diagnostic",
        ),
        CheckConstraint(
            "source_rq_prefix_id <> target_rq_prefix_id",
            name="ck_rq_prefix_pair_distinct_endpoints",
        ),
        Index(
            "ix_rq_prefix_pair_state_type",
            "graph_state_id",
            "edge_type",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    graph_state_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_relation_graph_states.id", ondelete="CASCADE"),
        index=True,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        index=True,
    )
    source_rq_prefix_id: Mapped[str] = mapped_column(
        ForeignKey("rq_prefixes.id", ondelete="CASCADE"),
        index=True,
    )
    target_rq_prefix_id: Mapped[str] = mapped_column(
        ForeignKey("rq_prefixes.id", ondelete="CASCADE"),
        index=True,
    )
    edge_type: Mapped[str] = mapped_column(String(64), index=True)
    diagnostic_strength: Mapped[float] = mapped_column(Float, default=0.0)
    support_membership_mass: Mapped[float] = mapped_column(Float, default=0.0)
    support_chunk_ids_sample_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_chunk_edge_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_algorithm: Mapped[str] = mapped_column(String(96), index=True)
    protocol_version: Mapped[str] = mapped_column(String(64), index=True)
    diagnostic_hash: Mapped[str] = mapped_column(String(64), index=True)
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
    source_algorithm: Mapped[str] = mapped_column(
        String(96),
        default="membership_weighted_bottom_edge_projection",
        index=True,
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="membership_q15_layer_type_calibrated_v3",
        index=True,
    )
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
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
    source_algorithm: Mapped[str] = mapped_column(
        String(96),
        default="membership_weighted_bottom_edge_projection",
        index=True,
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="membership_q15_layer_type_calibrated_v3",
        index=True,
    )
    state_hash: Mapped[str] = mapped_column(String(64), index=True)
    support_rq_prefix_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
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
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "embedding_model",
            "embedding_dimension",
            "embedding_text_version",
            "chunk_schema_version",
            name="uq_vector_chunk_model_dimension_text_schema_version",
        ),
        CheckConstraint(
            "embedding_dimension > 0",
            name="ck_vector_records_embedding_dimension_positive",
        ),
        ForeignKeyConstraint(
            ["chunk_id", "knowledge_base_id"],
            ["chunks.id", "chunks.knowledge_base_id"],
            name="fk_vector_records_chunk_knowledge_base_provenance",
            ondelete="CASCADE",
        ),
        Index(
            "ix_vector_records_reconcile_keyset",
            "knowledge_base_id",
            "collection_name",
            "id",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    # Keep the legacy single-column FK as a redundant existence/cascade gate;
    # the composite FK above is the authoritative same-KB provenance gate.
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    qdrant_point_id: Mapped[str] = mapped_column(String(64), index=True)
    collection_name: Mapped[str] = mapped_column(String(255), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    embedding_text_version: Mapped[str] = mapped_column(String(64), index=True)
    chunk_schema_version: Mapped[str] = mapped_column(
        String(64),
        default="chunk_schema_v1",
        index=True,
    )
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
    __table_args__ = (
        CheckConstraint(
            "conversation_state_protocol_version = 'conversation_state_v1'",
            name="ck_qa_sessions_conversation_state_protocol",
        ),
        CheckConstraint(
            "conversation_state_revision >= 0",
            name="ck_qa_sessions_conversation_state_revision",
        ),
        CheckConstraint(
            "length(conversation_state_hash) = 64",
            name="ck_qa_sessions_conversation_state_hash_length",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[list[dict]] = mapped_column(JSON, default=list)
    conversation_state_protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="conversation_state_v1",
        index=True,
    )
    conversation_state_revision: Mapped[int] = mapped_column(Integer, default=0)
    active_user_constraints_json: Mapped[dict] = mapped_column(JSON, default=dict)
    task_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    history_references_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    conversation_state_hash: Mapped[str] = mapped_column(
        String(64),
        default="959a95f2683e19efd10a1296ed82dac31d14c77a74caac4f5e0c12cfd062bd5e",
        index=True,
    )


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
    changed_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(64), default="api", index=True)
    managed_env_identity_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RuntimeSettingsAudit(TimestampMixin, Base):
    """Non-authoritative audit metadata for one root ``.env`` write.

    Runtime values and secret-presence snapshots are deliberately absent.  The
    repository-root ``.env`` is the only configuration authority; this row
    records only what changed, its lifecycle, delivery state and file identity.
    """

    __tablename__ = "runtime_settings_audits"
    __table_args__ = (
        UniqueConstraint(
            "version_hash", name="uq_runtime_settings_audits_hash"
        ),
        CheckConstraint(
            "protocol_version = 'runtime_settings_audit_v1'",
            name="ck_runtime_settings_audits_protocol",
        ),
        CheckConstraint(
            "status IN ('written','applied','pending_lifecycle','failed')",
            name="ck_runtime_settings_audits_status",
        ),
        Index(
            "ix_runtime_settings_audits_status_created",
            "status",
            "created_at",
        ),
        Index("ix_rs_audit_prior_runtime", "prior_runtime_version_hash"),
        Index("ix_rs_audit_env_identity", "env_identity_hash"),
        Index("ix_rs_audit_runtime", "runtime_version_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    protocol_version: Mapped[str] = mapped_column(
        String(64), default="runtime_settings_audit_v1"
    )
    version_hash: Mapped[str] = mapped_column(String(64), index=True)
    prior_runtime_version_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    changed_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    lifecycle_json: Mapped[dict] = mapped_column(JSON, default=dict)
    field_status_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(32), default="written", index=True
    )
    env_identity_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    runtime_version_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    source: Mapped[str] = mapped_column(String(64), default="api", index=True)
    last_error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)



class RuntimeSettingsCandidate(TimestampMixin, Base):
    __tablename__ = "runtime_settings_candidates"
    __table_args__ = (
        UniqueConstraint("candidate_hash", name="uq_runtime_settings_candidates_hash"),
        CheckConstraint(
            "protocol_version IN ('runtime_settings_vector_candidate_v1',"
            "'runtime_settings_candidate_v2')",
            name="ck_runtime_settings_candidates_protocol",
        ),
        CheckConstraint(
            "lifecycle_scope = 'rebuild_required'",
            name="ck_runtime_settings_candidates_lifecycle_scope",
        ),
        CheckConstraint(
            "status IN ('staged','building','evaluating','evaluation_passed','promotion_blocked',"
            "'promoting','promoted','rejected','failed','rolled_back','superseded')",
            name="ck_runtime_settings_candidates_status",
        ),
        Index("ix_runtime_candidates_base_version", "base_runtime_version_hash"),
        Index("ix_runtime_settings_candidates_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="runtime_settings_vector_candidate_v1",
    )
    candidate_hash: Mapped[str] = mapped_column(String(64))
    base_runtime_version_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settings_json: Mapped[dict] = mapped_column(JSON, default=dict)
    changed_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    target_knowledge_base_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    lifecycle_scope: Mapped[str] = mapped_column(String(32), default="rebuild_required")
    status: Mapped[str] = mapped_column(String(32), default="staged")
    source: Mapped[str] = mapped_column(String(64), default="api")
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    blocking_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RuntimeSettingsShadowBuild(TimestampMixin, Base):
    """Durable per-KB build/evaluation facts for a rebuild-required candidate.

    Vector artifacts stay owned by ``VectorShadowBuild``.  This row binds the
    wider runtime candidate to its frozen active before-state, optional shadow
    rechunk scope, four-layer graph result and measured evaluation.
    """

    __tablename__ = "runtime_settings_shadow_builds"
    __table_args__ = (
        UniqueConstraint(
            "runtime_settings_candidate_id",
            "knowledge_base_id",
            name="uq_runtime_settings_shadow_builds_candidate_kb",
        ),
        CheckConstraint(
            "protocol_version = 'runtime_settings_shadow_build_v1'",
            name="ck_runtime_settings_shadow_builds_protocol",
        ),
        CheckConstraint(
            "status IN ('staged','dry_run_passed','building','shadow_ready','evaluating',"
            "'evaluation_passed','promotion_blocked','promoting','promoted','failed',"
            "'rolled_back','superseded')",
            name="ck_runtime_settings_shadow_builds_status",
        ),
        CheckConstraint(
            "base_chunk_version >= 0 AND candidate_chunk_version >= 0",
            name="ck_runtime_settings_shadow_builds_chunk_versions_nonnegative",
        ),
        Index(
            "ix_runtime_settings_shadow_builds_kb_status",
            "knowledge_base_id",
            "status",
        ),
        Index(
            "ix_runtime_settings_shadow_builds_vector_build",
            "vector_shadow_build_id",
        ),
        Index(
            "ix_rs_shadow_candidate_chunk_schema",
            "candidate_chunk_schema_version",
        ),
        Index(
            "uq_runtime_settings_shadow_builds_one_live_per_kb",
            "knowledge_base_id",
            unique=True,
            postgresql_where=text(
                "status IN ('staged','dry_run_passed','building','shadow_ready','evaluating',"
                "'evaluation_passed','promotion_blocked','promoting')"
            ),
            sqlite_where=text(
                "status IN ('staged','dry_run_passed','building','shadow_ready','evaluating',"
                "'evaluation_passed','promotion_blocked','promoting')"
            ),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    runtime_settings_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("runtime_settings_candidates.id", ondelete="CASCADE"),
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
    )
    vector_shadow_build_id: Mapped[str | None] = mapped_column(
        ForeignKey("vector_shadow_builds.id", ondelete="SET NULL"),
        nullable=True,
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64), default="runtime_settings_shadow_build_v1"
    )
    status: Mapped[str] = mapped_column(String(32), default="staged")
    base_runtime_version_hash: Mapped[str] = mapped_column(String(64), index=True)
    base_chunk_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    base_vector_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    base_graph_bundle_hash: Mapped[str] = mapped_column(String(64), index=True)
    base_graph_state_ids_json: Mapped[dict] = mapped_column(JSON, default=dict)
    base_chunk_version: Mapped[int] = mapped_column(Integer, default=0)
    candidate_chunk_version: Mapped[int] = mapped_column(Integer, default=0)
    candidate_chunk_schema_version: Mapped[str] = mapped_column(String(64))
    candidate_chunk_scope_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    candidate_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    candidate_document_version_ids_json: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )
    shadow_context_graph_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("context_graph_states.id", ondelete="SET NULL"), nullable=True
    )
    shadow_chunk_relation_graph_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunk_relation_graph_states.id", ondelete="SET NULL"), nullable=True
    )
    shadow_mid_concept_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("mid_concept_states.id", ondelete="SET NULL"), nullable=True
    )
    shadow_coarse_concept_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("coarse_concept_states.id", ondelete="SET NULL"), nullable=True
    )
    dry_run_json: Mapped[dict] = mapped_column(JSON, default=dict)
    dry_run_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    build_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    build_result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_protocol_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    evaluation_input_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    evaluation_result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluation_result_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    promotion_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    rollback_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    blocking_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    shadow_ready_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RuntimeSettingsActivationIntent(TimestampMixin, Base):
    """Recoverable post-commit shared-env/version activation or rollback intent."""

    __tablename__ = "runtime_settings_activation_intents"
    __table_args__ = (
        UniqueConstraint(
            "runtime_settings_candidate_id",
            "direction",
            name="uq_runtime_settings_activation_candidate_direction",
        ),
        CheckConstraint(
            "protocol_version = 'runtime_settings_activation_intent_v1'",
            name="ck_runtime_settings_activation_intents_protocol",
        ),
        CheckConstraint(
            "direction IN ('promotion','rollback')",
            name="ck_runtime_settings_activation_intents_direction",
        ),
        CheckConstraint(
            "status IN ('pending','applying','applied','failed','superseded')",
            name="ck_runtime_settings_activation_intents_status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_runtime_settings_activation_intents_attempt_nonnegative",
        ),
        Index(
            "ix_runtime_settings_activation_intents_status_created",
            "status",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    runtime_settings_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("runtime_settings_candidates.id", ondelete="CASCADE")
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64), default="runtime_settings_activation_intent_v1"
    )
    direction: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    settings_json: Mapped[dict] = mapped_column(JSON, default=dict)
    settings_hash: Mapped[str] = mapped_column(String(64), index=True)
    changed_keys_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_candidate_status: Mapped[str] = mapped_column(String(32))
    runtime_version_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    last_error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class KnowledgeBaseVectorRuntimeState(TimestampMixin, Base):
    __tablename__ = "knowledge_base_vector_runtime_states"
    __table_args__ = (
        UniqueConstraint("knowledge_base_id", name="uq_kb_vector_runtime_states_kb"),
        CheckConstraint(
            "embedding_dimension > 0",
            name="ck_kb_vector_runtime_states_dimension_positive",
        ),
        CheckConstraint(
            "protocol_version = 'knowledge_base_vector_runtime_state_v1'",
            name="ck_kb_vector_runtime_states_protocol",
        ),
        CheckConstraint(
            "distance_metric = 'cosine'",
            name="ck_kb_vector_runtime_states_distance_metric",
        ),
        CheckConstraint(
            "activation_generation >= 1",
            name="ck_kb_vector_runtime_states_generation_positive",
        ),
        Index("ix_kb_vector_runtime_candidate", "runtime_settings_candidate_id"),
        Index("ix_kb_vector_runtime_collection", "collection_name"),
        Index("ix_kb_vector_runtime_schema_hash", "vector_schema_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
    )
    runtime_settings_candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("runtime_settings_candidates.id", ondelete="SET NULL"),
        nullable=True,
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="knowledge_base_vector_runtime_state_v1",
    )
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    distance_metric: Mapped[str] = mapped_column(String(32), default="cosine")
    embedding_text_version: Mapped[str] = mapped_column(String(64))
    chunk_schema_version: Mapped[str] = mapped_column(String(64))
    collection_identity_protocol_version: Mapped[str] = mapped_column(String(96))
    collection_identity_digest: Mapped[str] = mapped_column(String(64))
    collection_name: Mapped[str] = mapped_column(String(255))
    vector_schema_hash: Mapped[str] = mapped_column(String(64))
    state_hash: Mapped[str] = mapped_column(String(64))
    activation_generation: Mapped[int] = mapped_column(Integer, default=1)
    active_context_graph_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("context_graph_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    active_chunk_relation_graph_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunk_relation_graph_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    active_mid_concept_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("mid_concept_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    active_coarse_concept_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("coarse_concept_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    previous_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    promotion_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)


class VectorShadowBuild(TimestampMixin, Base):
    __tablename__ = "vector_shadow_builds"
    __table_args__ = (
        UniqueConstraint(
            "runtime_settings_candidate_id",
            "knowledge_base_id",
            name="uq_vector_shadow_builds_candidate_kb",
        ),
        CheckConstraint(
            "status IN ('staged','building','shadow_ready','evaluating','evaluation_passed',"
            "'promotion_blocked','promotion_pending','promoted','rejected','failed','rolled_back',"
            "'superseded')",
            name="ck_vector_shadow_builds_status",
        ),
        CheckConstraint(
            "protocol_version = 'vector_shadow_build_v1'",
            name="ck_vector_shadow_builds_protocol",
        ),
        CheckConstraint(
            "embedding_dimension > 0",
            name="ck_vector_shadow_builds_dimension_positive",
        ),
        CheckConstraint(
            "distance_metric = 'cosine'",
            name="ck_vector_shadow_builds_distance_metric",
        ),
        CheckConstraint(
            "expected_point_count >= 0 AND ready_point_count >= 0",
            name="ck_vector_shadow_builds_counts_nonnegative",
        ),
        Index("ix_vector_shadow_builds_schema_hash", "candidate_vector_schema_hash"),
        Index("ix_vector_shadow_builds_collection", "collection_name"),
        Index("ix_vector_shadow_builds_context_state", "shadow_context_graph_state_id"),
        Index("ix_vector_shadow_builds_kb_status", "knowledge_base_id", "status"),
        Index(
            "uq_vector_shadow_builds_one_live_per_kb",
            "knowledge_base_id",
            unique=True,
            postgresql_where=text(
                "status IN ('staged','building','shadow_ready','evaluating','evaluation_passed',"
                "'promotion_blocked','promotion_pending')"
            ),
            sqlite_where=text(
                "status IN ('staged','building','shadow_ready','evaluating','evaluation_passed',"
                "'promotion_blocked','promotion_pending')"
            ),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    runtime_settings_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("runtime_settings_candidates.id", ondelete="CASCADE"),
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64),
        default="vector_shadow_build_v1",
    )
    status: Mapped[str] = mapped_column(String(32), default="staged")
    base_vector_state_hash: Mapped[str] = mapped_column(String(64))
    candidate_vector_schema_json: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_vector_schema_hash: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    distance_metric: Mapped[str] = mapped_column(String(32), default="cosine")
    embedding_text_version: Mapped[str] = mapped_column(String(64))
    chunk_schema_version: Mapped[str] = mapped_column(String(64))
    collection_identity_protocol_version: Mapped[str] = mapped_column(String(96))
    collection_identity_digest: Mapped[str] = mapped_column(String(64))
    collection_name: Mapped[str] = mapped_column(String(255))
    shadow_context_graph_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("context_graph_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    shadow_chunk_relation_graph_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunk_relation_graph_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    shadow_mid_concept_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("mid_concept_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    shadow_coarse_concept_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("coarse_concept_states.id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_scope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_point_count: Mapped[int] = mapped_column(Integer, default=0)
    ready_point_count: Mapped[int] = mapped_column(Integer, default=0)
    vector_record_set_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    qdrant_proof_json: Mapped[dict] = mapped_column(JSON, default=dict)
    qdrant_proof_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_protocol_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluation_result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    promotion_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    rollback_audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    blocking_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    shadow_ready_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AutoTpeRun(TimestampMixin, Base):
    __tablename__ = "auto_tpe_runs"
    __table_args__ = (
        Index("ix_auto_tpe_runs_scope", "knowledge_base_id", "chunk_version", "chat_model", "embedding_model", "embedding_text_version"),
        Index("ix_auto_tpe_runs_kb_status", "knowledge_base_id", "status"),
        CheckConstraint(
            "status IN ('running','selected_pending_graph_commit','completed','failed','cancelled','skipped')",
            name="ck_auto_tpe_runs_status",
        ),
        CheckConstraint(
            "selected_graph_runtime_settings_hash IS NULL "
            "OR length(selected_graph_runtime_settings_hash) = 64",
            name="ck_auto_tpe_runs_graph_runtime_hash_length",
        ),
        CheckConstraint(
            "status NOT IN ('selected_pending_graph_commit','completed') "
            "OR selected_graph_runtime_settings_hash IS NOT NULL",
            name="ck_auto_tpe_runs_selected_graph_runtime_required",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_batches.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_relation_graph_state_id: Mapped[str | None] = mapped_column(ForeignKey("chunk_relation_graph_states.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_version: Mapped[int] = mapped_column(Integer, index=True)
    chunk_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    graph_operating_point_protocol: Mapped[str] = mapped_column(String(64), index=True)
    protocol_hash: Mapped[str] = mapped_column(String(64), index=True)
    tpe_search_space_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chat_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_text_version: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    trigger_reason: Mapped[str] = mapped_column(String(128), default="chunk_version_incremented", index=True)
    trial_budget: Mapped[int] = mapped_column(Integer, default=0)
    startup_random_trials: Mapped[int] = mapped_column(Integer, default=0)
    good_quantile_gamma: Mapped[float] = mapped_column(Float, default=0.25)
    probe_query_budget: Mapped[int] = mapped_column(Integer, default=0)
    candidate_pool_size: Mapped[int] = mapped_column(Integer, default=0)
    best_trial_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "auto_tpe_trials.id",
            name="fk_auto_tpe_runs_best_trial_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
    best_objective_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    selected_theta_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_theta_json: Mapped[dict] = mapped_column(JSON, default=dict)
    selected_edge_distance_protocol: Mapped[str | None] = mapped_column(String(64), nullable=True)
    selected_edge_distance_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_edge_type_calibration_protocol: Mapped[str | None] = mapped_column(String(64), nullable=True)
    selected_edge_type_calibration_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_calibration_params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    selected_calibration_params_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_edge_type_calibration_config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    sampler_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    probe_set_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    hard_gate_json: Mapped[dict] = mapped_column(JSON, default=dict)
    objective_components_json: Mapped[dict] = mapped_column(JSON, default=dict)
    blocking_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    runtime_settings_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_graph_runtime_settings_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    selected_gate_profile_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    selected_gate_profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AutoTpeTrial(Base):
    __tablename__ = "auto_tpe_trials"
    __table_args__ = (
        UniqueConstraint("run_id", "trial_index", name="uq_auto_tpe_trial_run_index"),
        Index("ix_auto_tpe_trials_run_status", "run_id", "status"),
        CheckConstraint(
            "status IN ('queued','running','completed','blocked','failed','cancelled')",
            name="ck_auto_tpe_trials_status",
        ),
        Index(
            "ix_auto_tpe_trials_scope",
            "knowledge_base_id",
            "chunk_scope_hash",
            "embedding_model",
            "embedding_text_version",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("auto_tpe_runs.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    build_batch_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_batches.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    chunk_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_text_version: Mapped[str] = mapped_column(String(64), index=True)
    trial_index: Mapped[int] = mapped_column(Integer, index=True)
    sampled_theta_json: Mapped[dict] = mapped_column(JSON, default=dict)
    theta_hash: Mapped[str] = mapped_column(String(64), index=True)
    tpe_search_space_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    edge_distance_protocol: Mapped[str | None] = mapped_column(String(64), nullable=True)
    edge_distance_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    edge_type_calibration_protocol: Mapped[str | None] = mapped_column(String(64), nullable=True)
    edge_type_calibration_protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    calibration_params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    calibration_params_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    edge_type_calibration_config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    sampler_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    runtime_settings_hash: Mapped[str] = mapped_column(String(64), index=True)
    gate_profile_hash: Mapped[str] = mapped_column(String(64), index=True)
    gate_profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_adjacency_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    probe_set_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    hard_gate_json: Mapped[dict] = mapped_column(JSON, default=dict)
    objective_components_json: Mapped[dict] = mapped_column(JSON, default=dict)
    objective_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class IngestionBatch(TimestampMixin, Base):
    __tablename__ = "ingestion_batches"
    __table_args__ = (
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_ingestion_batches_id_kb",
        ),
    )

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


class IngestionBatchRecovery(TimestampMixin, Base):
    """Durable batch cancellation/restart boundary.

    The row is written before the first file mutation.  It deliberately keeps
    the pre-batch active scope separate from per-file write sets so recovery
    never has to infer an old version with ``target_version - 1``.
    """

    __tablename__ = "ingestion_batch_recoveries"
    __table_args__ = (
        UniqueConstraint("batch_id", name="uq_ingestion_batch_recoveries_batch"),
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_ingestion_batch_recoveries_id_kb",
        ),
        ForeignKeyConstraint(
            ["batch_id", "knowledge_base_id"],
            ["ingestion_batches.id", "ingestion_batches.knowledge_base_id"],
            name="fk_ingestion_batch_recoveries_batch_kb",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "protocol_version = 'ingestion_batch_cancel_compensation_v1'",
            name="ck_ingestion_batch_recoveries_protocol",
        ),
        CheckConstraint(
            "status IN ('prepared','parsing','parse_compensation_pending','parse_compensating',"
            "'parse_compensated','graph_building','graph_compensation_pending',"
            "'graph_compensated','completed','completed_no_writes','manual_review')",
            name="ck_ingestion_batch_recoveries_status",
        ),
        CheckConstraint(
            "v_before_batch >= 0 AND target_version >= v_before_batch",
            name="ck_ingestion_batch_recoveries_versions_ordered",
        ),
        CheckConstraint(
            "((NOT parse_committed) AND status IN ("
            "'prepared','parsing','parse_compensation_pending','parse_compensating',"
            "'parse_compensated','completed_no_writes','manual_review')) OR "
            "(parse_committed AND status IN ("
            "'graph_building','graph_compensation_pending','graph_compensated',"
            "'completed','manual_review'))",
            name="ck_ingestion_batch_recoveries_parse_state",
        ),
        Index(
            "ix_ingestion_batch_recoveries_kb_status",
            "knowledge_base_id",
            "status",
        ),
        Index(
            "uq_ingestion_batch_recovery_active_kb",
            "knowledge_base_id",
            unique=True,
            postgresql_where=text(
                "status IN ('prepared','parsing','parse_compensation_pending',"
                "'parse_compensating','graph_building','graph_compensation_pending',"
                "'manual_review')"
            ),
            sqlite_where=text(
                "status IN ('prepared','parsing','parse_compensation_pending',"
                "'parse_compensating','graph_building','graph_compensation_pending',"
                "'manual_review')"
            ),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    batch_id: Mapped[str] = mapped_column(String(36), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    protocol_version: Mapped[str] = mapped_column(
        String(64), default="ingestion_batch_cancel_compensation_v1", index=True
    )
    status: Mapped[str] = mapped_column(String(40), default="prepared", index=True)
    v_before_batch: Mapped[int] = mapped_column(Integer, default=0)
    target_version: Mapped[int] = mapped_column(Integer, default=0)
    full_reparse: Mapped[bool] = mapped_column(Boolean, default=False)
    parse_committed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    before_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    before_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    graph_before_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    graph_before_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    graph_write_set_json: Mapped[dict] = mapped_column(JSON, default=dict)
    graph_write_set_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    compensation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class IngestionFileStage(TimestampMixin, Base):
    """One immutable before-image plus the exact committed write set per file."""

    __tablename__ = "ingestion_file_stages"
    __table_args__ = (
        UniqueConstraint(
            "batch_recovery_id",
            "source_path",
            name="uq_ingestion_file_stages_recovery_source",
        ),
        UniqueConstraint(
            "batch_recovery_id",
            "sequence_index",
            name="uq_ingestion_file_stages_recovery_sequence",
        ),
        CheckConstraint("sequence_index >= 1", name="ck_ingestion_file_stages_sequence_positive"),
        CheckConstraint(
            "status IN ('prepared','parsing','indexed_committed','failed','cancel_observed',"
            "'compensation_pending','compensating','compensated','retained_after_parse_commit',"
            "'manual_review')",
            name="ck_ingestion_file_stages_status",
        ),
        CheckConstraint(
            "(status = 'prepared' AND phase = 'prepared') OR "
            "(status = 'parsing' AND phase = 'parsing') OR "
            "(status = 'indexed_committed' AND phase = 'indexed') OR "
            "(status = 'failed' AND phase = 'failed') OR "
            "(status = 'cancel_observed' AND phase = 'cancel_observed') OR "
            "(status = 'compensation_pending' AND phase = 'qdrant_compensation') OR "
            "(status = 'compensating' AND phase = 'database_restore') OR "
            "(status = 'compensated' AND phase = 'compensated') OR "
            "(status = 'retained_after_parse_commit' AND phase = 'context_graph') OR "
            "(status = 'manual_review' AND phase = 'manual_review')",
            name="ck_ingestion_file_stages_status_phase",
        ),
        ForeignKeyConstraint(
            ["batch_recovery_id", "knowledge_base_id"],
            [
                "ingestion_batch_recoveries.id",
                "ingestion_batch_recoveries.knowledge_base_id",
            ],
            name="fk_ingestion_file_stages_recovery_kb",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["ingestion_job_id", "knowledge_base_id"],
            ["ingestion_jobs.id", "ingestion_jobs.knowledge_base_id"],
            name="fk_ingestion_file_stages_job_kb",
        ),
        ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_ingestion_file_stages_document_kb",
        ),
        Index(
            "ix_ingestion_file_stages_recovery_status",
            "batch_recovery_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    batch_recovery_id: Mapped[str] = mapped_column(String(36), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    ingestion_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_path: Mapped[str] = mapped_column(Text)
    sequence_index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40), default="prepared", index=True)
    phase: Mapped[str] = mapped_column(String(40), default="prepared", index=True)
    before_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    before_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    write_set_json: Mapped[dict] = mapped_column(JSON, default=dict)
    write_set_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    compensation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_ingestion_jobs_id_kb",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_batches.id", ondelete="SET NULL"), nullable=True, index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    logical_source_slot_key: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        index=True,
    )
    source_slot_protocol_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
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


class StorageMaintenanceIntent(Base):
    """Durable storage/Qdrant work that must outlive knowledge-base facts."""

    __tablename__ = "storage_maintenance_intents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('intent_committed','external_deleting','external_applied',"
            "'facts_deleted','cache_invalidation_pending','completed','manual_review')",
            name="ck_storage_maintenance_intents_status",
        ),
        Index(
            "uq_storage_maintenance_intents_active_operation_scope",
            "operation",
            "scope_key",
            unique=True,
            postgresql_where=text("status <> 'completed'"),
            sqlite_where=text("status <> 'completed'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    # Deliberately no FK: full-KB deletion must retain its tombstone after the
    # knowledge_bases row and every cascading fact have been removed.
    knowledge_base_id: Mapped[str] = mapped_column(String(36), index=True)
    knowledge_base_name: Mapped[str] = mapped_column(String(255))
    operation: Mapped[str] = mapped_column(String(64), index=True)
    protocol_version: Mapped[str] = mapped_column(String(96))
    scope_key: Mapped[str] = mapped_column(String(128))
    target_root: Mapped[str | None] = mapped_column(Text, nullable=True)
    inventory_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(32),
        default="intent_committed",
        server_default="intent_committed",
        index=True,
    )
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
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "sequence_index",
            name="uq_agent_trace_events_run_sequence",
        ),
        CheckConstraint(
            "sequence_index >= 0",
            name="ck_agent_trace_events_sequence_nonnegative",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    sequence_index: Mapped[int] = mapped_column(Integer)
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
