from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Course(TimestampMixin, Base):
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_root: Mapped[str] = mapped_column(Text)

    documents: Mapped[list["Document"]] = relationship(back_populates="course")
    concepts: Mapped[list["Concept"]] = relationship(back_populates="course")
    batches: Mapped[list["IngestionBatch"]] = relationship(back_populates="course")
    model_hyperparameters: Mapped[list["CourseModelHyperparameter"]] = relationship(back_populates="course")


class CourseModelHyperparameter(TimestampMixin, Base):
    __tablename__ = "course_model_hyperparameters"

    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True)
    llm_model_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding_model_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding_text_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_name: Mapped[str | None] = mapped_column(String(384), nullable=True)
    graph_version: Mapped[str] = mapped_column(String(128), default="active")

    min_relation_confidence: Mapped[float] = mapped_column(Float, default=0.72)
    min_accepted_relation_weight: Mapped[float] = mapped_column(Float, default=0.62)
    dijkstra_semantic_threshold: Mapped[float] = mapped_column(Float, default=0.78)

    w_degree: Mapped[float] = mapped_column(Float, default=0.25)
    w_weighted_degree: Mapped[float] = mapped_column(Float, default=0.25)
    w_pagerank: Mapped[float] = mapped_column(Float, default=0.20)
    w_betweenness: Mapped[float] = mapped_column(Float, default=0.20)
    w_closeness: Mapped[float] = mapped_column(Float, default=0.10)

    w_centrality: Mapped[float] = mapped_column(Float, default=0.50)
    w_llm_importance: Mapped[float] = mapped_column(Float, default=0.25)
    w_evidence: Mapped[float] = mapped_column(Float, default=0.25)

    hpo_status: Mapped[str] = mapped_column(String(32), default="pending")
    last_optimized_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    optuna_history: Mapped[dict] = mapped_column(JSON, default=dict)

    course: Mapped["Course"] = relationship(back_populates="model_hyperparameters")


class GraphHpoJudgeSample(TimestampMixin, Base):
    __tablename__ = "graph_hpo_judge_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    llm_model_name: Mapped[str] = mapped_column(String(128), index=True)
    embedding_model_name: Mapped[str] = mapped_column(String(128), index=True)
    embedding_text_version: Mapped[str] = mapped_column(String(32), index=True)
    model_key: Mapped[str] = mapped_column(String(384), index=True)
    payload_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    prompt_version: Mapped[str] = mapped_column(String(128), index=True)
    judge_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    candidate_a_params: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_b_params: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_a_features: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_b_features: Mapped[dict] = mapped_column(JSON, default=dict)
    winner: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    safety_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    raw_response: Mapped[dict] = mapped_column(JSON, default=dict)


class GraphHpoObjectiveModel(TimestampMixin, Base):
    __tablename__ = "graph_hpo_objective_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    llm_model_name: Mapped[str] = mapped_column(String(128), index=True)
    embedding_model_name: Mapped[str] = mapped_column(String(128), index=True)
    embedding_text_version: Mapped[str] = mapped_column(String(32), index=True)
    model_key: Mapped[str] = mapped_column(String(384), index=True)
    objective_version: Mapped[str] = mapped_column(String(128), index=True)
    payload_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    feature_names: Mapped[list[str]] = mapped_column(JSON, default=list)
    weights_json: Mapped[dict] = mapped_column(JSON, default=dict)
    normalization_json: Mapped[dict] = mapped_column(JSON, default=dict)
    label_count: Mapped[int] = mapped_column(Integer, default=0)
    training_audit: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text, index=True)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(32), default="private")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    course: Mapped["Course"] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


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
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document_version")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    snippet: Mapped[str] = mapped_column(Text)
    chapter: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding_status: Mapped[str] = mapped_column(String(32), default="pending")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("chunks.id"), nullable=True, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    embedding_text_version: Mapped[str] = mapped_column(String(32), default="metadata_enriched_v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    document_version: Mapped["DocumentVersion"] = relationship(back_populates="chunks")


class Concept(TimestampMixin, Base):
    __tablename__ = "concepts"
    __table_args__ = (UniqueConstraint("course_id", "normalized_name", name="uq_course_concept_normalized"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), index=True)
    normalized_name: Mapped[str] = mapped_column(String(255), index=True)
    concept_type: Mapped[str] = mapped_column(String(64), default="concept")
    summary: Mapped[str] = mapped_column(Text, default="")
    chapter_refs: Mapped[list[str]] = mapped_column(JSON, default=list)
    importance_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    community_louvain: Mapped[int | None] = mapped_column(Integer, nullable=True)
    community_spectral: Mapped[int | None] = mapped_column(Integer, nullable=True)
    component_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    centrality_json: Mapped[dict] = mapped_column(JSON, default=dict)
    graph_rank_score: Mapped[float] = mapped_column(Float, default=0.0)
    source_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)

    course: Mapped["Course"] = relationship(back_populates="concepts")
    aliases: Mapped[list["ConceptAlias"]] = relationship(back_populates="concept")


class ConceptAlias(Base):
    __tablename__ = "concept_aliases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), index=True)
    alias: Mapped[str] = mapped_column(String(255))
    normalized_alias: Mapped[str] = mapped_column(String(255), index=True)

    concept: Mapped["Concept"] = relationship(back_populates="aliases")


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    __table_args__ = (UniqueConstraint("course_id", "chunk_id", "surface", "entity_type", name="uq_entity_mention_surface_chunk_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    concept_id: Mapped[str | None] = mapped_column(ForeignKey("concepts.id"), nullable=True, index=True)
    surface: Mapped[str] = mapped_column(String(255))
    canonical_name: Mapped[str] = mapped_column(String(255), index=True)
    normalized_key: Mapped[str] = mapped_column(String(320), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_spans: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="staged", index=True)
    decision_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EntityMergeCandidate(Base):
    __tablename__ = "entity_merge_candidates"
    __table_args__ = (UniqueConstraint("course_id", "left_key", "right_key", name="uq_entity_merge_candidate_pair"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    left_key: Mapped[str] = mapped_column(String(320), index=True)
    right_key: Mapped[str] = mapped_column(String(320), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(64), default="lexical")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    verifier_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ConceptRelation(Base):
    __tablename__ = "concept_relations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    source_concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), index=True)
    target_concept_id: Mapped[str | None] = mapped_column(ForeignKey("concepts.id"), nullable=True)
    target_name: Mapped[str] = mapped_column(String(255))
    relation_type: Mapped[str] = mapped_column(String(64), index=True)
    evidence_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id"), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.55)
    extraction_method: Mapped[str] = mapped_column(String(64), default="heuristic")
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    semantic_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    support_count: Mapped[int] = mapped_column(Integer, default=1)
    relation_source: Mapped[str] = mapped_column(String(64), default="llm")
    is_inferred: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    source_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QualityProfile(Base):
    __tablename__ = "quality_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    version: Mapped[str] = mapped_column(String(128), index=True)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GraphRelationCandidate(Base):
    __tablename__ = "graph_relation_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    source_concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), index=True)
    target_concept_id: Mapped[str | None] = mapped_column(ForeignKey("concepts.id"), nullable=True)
    target_name: Mapped[str] = mapped_column(String(255))
    relation_type: Mapped[str] = mapped_column(String(64), index=True)
    relation_source: Mapped[str] = mapped_column(String(64), default="semantic_sparse", index=True)
    evidence_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id"), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    semantic_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    support_count: Mapped[int] = mapped_column(Integer, default=1)
    is_inferred: Mapped[bool] = mapped_column(Boolean, default=True)
    decision_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    source_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GraphCommunitySummary(TimestampMixin, Base):
    __tablename__ = "graph_community_summaries"
    __table_args__ = (UniqueConstraint("course_id", "algorithm", "community_id", "version", name="uq_graph_community_summary_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    algorithm: Mapped[str] = mapped_column(String(64), default="louvain", index=True)
    community_id: Mapped[int] = mapped_column(Integer, index=True)
    version: Mapped[str] = mapped_column(String(128), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    key_concepts_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    representative_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_version: Mapped[str] = mapped_column(String(128), default="community_summary_v1")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class GraphExtractionRun(TimestampMixin, Base):
    __tablename__ = "graph_extraction_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("ingestion_batches.id"), nullable=True, index=True)
    strategy: Mapped[str] = mapped_column(String(64), default="adaptive_best_first", index=True)
    profile_version: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    prompt_version: Mapped[str] = mapped_column(String(128), default="graph_extraction_v1")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="planned", index=True)
    coverage_json: Mapped[dict] = mapped_column(JSON, default=dict)
    budget_json: Mapped[dict] = mapped_column(JSON, default=dict)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GraphExtractionChunkTask(TimestampMixin, Base):
    __tablename__ = "graph_extraction_chunk_tasks"
    __table_args__ = (UniqueConstraint("run_id", "chunk_id", name="uq_graph_extraction_run_chunk"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("graph_extraction_runs.id"), index=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), index=True)
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    selected_reason: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)


class IngestionBatch(TimestampMixin, Base):
    __tablename__ = "ingestion_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
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
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    course: Mapped["Course"] = relationship(back_populates="batches")
    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="batch")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
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
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
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
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[list[dict]] = mapped_column(JSON, default=list)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
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
