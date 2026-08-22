from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    Chunk,
    ChunkVersion,
    ChunkRelationGraphState,
    CoarseConcept,
    CoarseConceptState,
    ContextGraphState,
    DocumentVersion,
    IngestionCompensationLog,
    KnowledgeBase,
    KnowledgeBaseVectorRuntimeState,
    MidConcept,
    MidConceptState,
    RQPrefix,
    RuntimeSettingsCandidate,
    VectorRecord,
    VectorShadowBuild,
)
from app.services.chunking import CHUNK_SCHEMA_VERSION, CURRENT_EMBEDDING_TEXT_VERSION
from app.services.context_graph import (
    CHUNK_SCOPE_HASH_PROTOCOL_VERSION,
    QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
    QDRANT_VECTOR_DISTANCE_METRIC,
    compute_chunk_scope_hash,
    qdrant_collection_identity_digest,
    qdrant_collection_name,
    runtime_settings_state_hash,
)
from app.services.error_sanitizer import external_failure_classification


VECTOR_SCHEMA_PROTOCOL_VERSION = "frozen_vector_schema_v2"
VECTOR_RUNTIME_CANDIDATE_PROTOCOL_VERSION = "runtime_settings_vector_candidate_v1"
VECTOR_RUNTIME_STATE_PROTOCOL_VERSION = "knowledge_base_vector_runtime_state_v1"
VECTOR_SHADOW_BUILD_PROTOCOL_VERSION = "vector_shadow_build_v1"
VECTOR_SHADOW_WRITER_PROTOCOL_VERSION = "vector_shadow_writer_v1"
VECTOR_SHADOW_GRAPH_CONSUMER_PROTOCOL_VERSION = "vector_shadow_graph_consumer_v1"
QDRANT_SHADOW_PROOF_PROTOCOL_VERSION = "qdrant_shadow_scope_proof_v1"
QDRANT_SHADOW_OBSERVER_PROTOCOL_VERSION = "qdrant_shadow_bounded_observer_v1"
VECTOR_SHADOW_EVALUATION_PROTOCOL_VERSION = "vector_shadow_evaluation_v1"
VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION = "vector_shadow_rollback_v1"
VECTOR_SHADOW_CONCEPT_SEMANTIC_REUSE_PROTOCOL_VERSION = (
    "vector_shadow_terminal_concept_semantic_reuse_v2"
)
VECTOR_SHADOW_COMPENSATED_EMBEDDING_RECOVERY_PROTOCOL_VERSION = (
    "vector_shadow_compensated_embedding_recovery_v2"
)
DEFAULT_VECTOR_SHADOW_CONCEPT_PROVIDER_REQUEST_BUDGET = 4

REQUIRED_ACTIVE_VECTOR_RUNTIME_CONSUMERS = frozenset(
    {
        "contextual_index_writer",
        "context_graph_builder",
        "query_embedding",
        "vector_lookup",
    }
)
INTEGRATED_ACTIVE_VECTOR_RUNTIME_CONSUMERS: frozenset[str] = (
    REQUIRED_ACTIVE_VECTOR_RUNTIME_CONSUMERS
)
ATOMIC_ACTIVE_SWITCH_IMPLEMENTED = True
VECTOR_RUNTIME_TARGET_PROTOCOL_VERSION = "vector_runtime_target_v1"
VECTOR_RUNTIME_CACHE_INVALIDATION_PROTOCOL_VERSION = (
    "vector_runtime_cache_invalidation_v1"
)
VECTOR_RUNTIME_CACHE_INVALIDATION_OPERATION = "vector_runtime_cache_invalidation"

REQUIRED_EVALUATION_GATES = frozenset(
    {
        "qdrant_schema_match",
        "vector_record_coverage",
        "structure_recovery",
        "retrieval_quality",
        "citation_quality",
        "latency_budget",
        "resource_budget",
    }
)
REQUIRED_EVALUATION_EVIDENCE = frozenset(
    {
        "retrieval_quality",
        "citation_quality",
        "latency",
        "resource_usage",
    }
)
LIVE_SHADOW_BUILD_STATUSES = frozenset(
    {
        "staged",
        "building",
        "shadow_ready",
        "evaluating",
        "evaluation_passed",
        "promotion_blocked",
        "promotion_pending",
    }
)
HASH64_PATTERN = r"^[0-9a-f]{64}$"
ACTIVE_CHUNK_READ_PAGE_SIZE = 256
MAX_CANDIDATE_KNOWLEDGE_BASES = 64
MAX_CACHE_INVALIDATION_INTENT_SCAN = 4096
MAX_VECTOR_SHADOW_COMPENSATED_RECOVERY_INTENTS = 128
MAX_VECTOR_SHADOW_TERMINAL_CONCEPT_SOURCES = 32


class FrozenVectorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["frozen_vector_schema_v2"] = VECTOR_SCHEMA_PROTOCOL_VERSION
    embedding_api_protocol: Literal["openai"] = "openai"
    embedding_model: str = Field(min_length=1, max_length=128)
    embedding_dimension: int = Field(gt=0)
    distance_metric: Literal["cosine"] = "cosine"
    embedding_text_version: str = Field(min_length=1, max_length=64)
    chunk_schema_version: str = Field(min_length=1, max_length=64)
    collection_identity_protocol_version: Literal[
        "qdrant_collection_identity_u64be_utf8_sha256_v2"
    ]
    collection_identity_digest: str = Field(pattern=HASH64_PATTERN)
    collection_name: str = Field(min_length=1, max_length=255)


class VectorRuntimeTarget(BaseModel):
    """One immutable writer/query/graph-consumer vector identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["vector_runtime_target_v1"] = (
        VECTOR_RUNTIME_TARGET_PROTOCOL_VERSION
    )
    knowledge_base_id: str = Field(min_length=1, max_length=36)
    state_scope: Literal["active", "shadow"]
    schema: FrozenVectorSchema
    vector_schema_hash: str = Field(pattern=HASH64_PATTERN)
    runtime_state_id: str | None = Field(default=None, max_length=36)
    runtime_state_hash: str = Field(pattern=HASH64_PATTERN)
    activation_generation: int = Field(ge=0)
    runtime_settings_candidate_id: str | None = Field(default=None, max_length=36)
    runtime_settings_candidate_hash: str | None = Field(
        default=None,
        pattern=HASH64_PATTERN,
    )
    vector_shadow_build_id: str | None = Field(default=None, max_length=36)
    pending_vector_status: Literal["pending", "shadow_pending"]
    ready_vector_status: Literal["ready", "shadow_ready"]

    @model_validator(mode="after")
    def validate_runtime_target(self) -> "VectorRuntimeTarget":
        if vector_schema_hash(self.schema) != self.vector_schema_hash:
            raise ValueError("vector_schema_hash does not match the frozen schema")
        if self.state_scope == "active":
            if self.vector_shadow_build_id is not None:
                raise ValueError("active target cannot bind a vector shadow build")
            if (self.pending_vector_status, self.ready_vector_status) != (
                "pending",
                "ready",
            ):
                raise ValueError("active target must use pending -> ready")
        else:
            if (
                not self.runtime_settings_candidate_id
                or not self.runtime_settings_candidate_hash
                or not self.vector_shadow_build_id
            ):
                raise ValueError("shadow target requires exact candidate/build bindings")
            if (self.pending_vector_status, self.ready_vector_status) != (
                "shadow_pending",
                "shadow_ready",
            ):
                raise ValueError("shadow target must use shadow_pending -> shadow_ready")
        return self


class QdrantShadowScopeProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["qdrant_shadow_scope_proof_v1"] = (
        QDRANT_SHADOW_PROOF_PROTOCOL_VERSION
    )
    verified: Literal[True]
    collection_name: str = Field(min_length=1, max_length=255)
    collection_identity_protocol_version: Literal[
        "qdrant_collection_identity_u64be_utf8_sha256_v2"
    ]
    collection_identity_digest: str = Field(pattern=HASH64_PATTERN)
    embedding_model: str = Field(min_length=1, max_length=128)
    embedding_dimension: int = Field(gt=0)
    distance_metric: Literal["cosine"]
    embedding_text_version: str = Field(min_length=1, max_length=64)
    chunk_schema_version: str = Field(min_length=1, max_length=64)
    scope_filter_hash: str = Field(pattern=HASH64_PATTERN)
    scoped_point_count: int = Field(ge=0)
    point_set_hash: str = Field(pattern=HASH64_PATTERN)
    observer_input_hash: str = Field(pattern=HASH64_PATTERN)
    observer_output_hash: str = Field(pattern=HASH64_PATTERN)


class VectorShadowBuildAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["vector_shadow_build_attestation_v1"] = (
        "vector_shadow_build_attestation_v1"
    )
    writer_protocol_version: Literal["vector_shadow_writer_v1"]
    graph_consumer_protocol_version: Literal["vector_shadow_graph_consumer_v1"]
    chunk_scope_hash: str = Field(pattern=HASH64_PATTERN)
    shadow_context_graph_state_id: str = Field(min_length=1, max_length=36)
    shadow_chunk_relation_graph_state_id: str = Field(min_length=1, max_length=36)
    shadow_mid_concept_state_id: str = Field(min_length=1, max_length=36)
    shadow_coarse_concept_state_id: str = Field(min_length=1, max_length=36)
    qdrant: QdrantShadowScopeProof


class VectorShadowEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["vector_shadow_evaluation_v1"] = (
        VECTOR_SHADOW_EVALUATION_PROTOCOL_VERSION
    )
    evaluation_input_hash: str = Field(pattern=HASH64_PATTERN)
    hard_gates: dict[str, bool]
    evidence_hashes: dict[str, str]
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    evaluator_diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_gate_contract(self) -> "VectorShadowEvaluation":
        actual_gates = set(self.hard_gates)
        if actual_gates != REQUIRED_EVALUATION_GATES:
            raise ValueError(
                "hard_gates must contain exactly the vector_shadow_evaluation_v1 gate set"
            )
        if set(self.evidence_hashes) != REQUIRED_EVALUATION_EVIDENCE:
            raise ValueError(
                "evidence_hashes must contain exactly the vector_shadow_evaluation_v1 artifact set"
            )
        for key, value in self.evidence_hashes.items():
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"evidence_hashes.{key} must be a lowercase sha256 digest")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash64(value: str | None, *, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return candidate


def _candidate_effective_runtime_hash(candidate: RuntimeSettingsCandidate) -> str:
    if candidate.protocol_version == "runtime_settings_candidate_v2":
        value = str(
            (candidate.diagnostics_json or {}).get(
                "effective_runtime_settings_hash"
            )
            or ""
        )
    else:
        # Vector-only candidates change the frozen vector schema, not
        # the active Runtime Settings slice.  Their promoted graph therefore
        # remains bound to the current active runtime identity.
        value = runtime_settings_state_hash()
    return _require_hash64(value, field_name="effective_runtime_settings_hash")


def frozen_vector_schema(
    *,
    embedding_api_protocol: Literal["openai"] = "openai",
    embedding_model: str,
    embedding_dimension: int,
    embedding_text_version: str = CURRENT_EMBEDDING_TEXT_VERSION,
    chunk_schema_version: str = CHUNK_SCHEMA_VERSION,
) -> FrozenVectorSchema:
    model = str(embedding_model or "").strip()
    text_version = str(embedding_text_version or "").strip()
    schema_version = str(chunk_schema_version or "").strip()
    digest = qdrant_collection_identity_digest(
        embedding_model=model,
        embedding_dimension=embedding_dimension,
        embedding_text_version=text_version,
        chunk_schema_version=schema_version,
    )
    return FrozenVectorSchema(
        embedding_api_protocol=embedding_api_protocol,
        embedding_model=model,
        embedding_dimension=embedding_dimension,
        distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
        embedding_text_version=text_version,
        chunk_schema_version=schema_version,
        collection_identity_protocol_version=QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        collection_identity_digest=digest,
        collection_name=qdrant_collection_name(
            embedding_model=model,
            embedding_dimension=embedding_dimension,
            embedding_text_version=text_version,
            chunk_schema_version=schema_version,
        ),
    )


def vector_schema_hash(schema: FrozenVectorSchema) -> str:
    return _stable_hash(schema.model_dump(mode="json"))


def _active_chunks(db: Session, knowledge_base_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    last_id: str | None = None
    while True:
        statement = select(Chunk).where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Chunk.state == "active",
        )
        if last_id is not None:
            statement = statement.where(Chunk.id > last_id)
        page = list(
            db.scalars(
                statement.order_by(Chunk.id.asc()).limit(ACTIVE_CHUNK_READ_PAGE_SIZE)
            ).all()
        )
        chunks.extend(page)
        if len(page) < ACTIVE_CHUNK_READ_PAGE_SIZE:
            return chunks
        last_id = page[-1].id


def _scope_filter_hash(knowledge_base_id: str, chunk_ids: Sequence[str]) -> str:
    return _stable_hash(
        {
            "protocol_version": "qdrant_knowledge_base_chunk_scope_filter_v1",
            "knowledge_base_id": knowledge_base_id,
            "chunk_ids": list(chunk_ids),
        }
    )


def _schema_from_pointer(pointer: KnowledgeBaseVectorRuntimeState) -> FrozenVectorSchema:
    schema = FrozenVectorSchema(
        embedding_model=pointer.embedding_model,
        embedding_dimension=pointer.embedding_dimension,
        distance_metric=pointer.distance_metric,
        embedding_text_version=pointer.embedding_text_version,
        chunk_schema_version=pointer.chunk_schema_version,
        collection_identity_protocol_version=pointer.collection_identity_protocol_version,
        collection_identity_digest=pointer.collection_identity_digest,
        collection_name=pointer.collection_name,
    )
    if vector_schema_hash(schema) != pointer.vector_schema_hash:
        raise RuntimeError(
            f"Active vector runtime state {pointer.id} has a non-canonical vector_schema_hash"
        )
    return schema


def _graph_state_ids_from_pointer(
    pointer: KnowledgeBaseVectorRuntimeState,
) -> dict[str, str | None]:
    return {
        "context": pointer.active_context_graph_state_id,
        "relation": pointer.active_chunk_relation_graph_state_id,
        "mid": pointer.active_mid_concept_state_id,
        "coarse": pointer.active_coarse_concept_state_id,
    }


def vector_runtime_state_hash(
    *,
    knowledge_base_id: str,
    runtime_settings_candidate_id: str | None,
    activation_generation: int,
    schema: FrozenVectorSchema,
    graph_state_ids: dict[str, str | None],
) -> str:
    if set(graph_state_ids) != {"context", "relation", "mid", "coarse"}:
        raise ValueError("graph_state_ids must contain the exact four-layer key set")
    return _stable_hash(
        {
            "protocol_version": VECTOR_RUNTIME_STATE_PROTOCOL_VERSION,
            "knowledge_base_id": knowledge_base_id,
            "runtime_settings_candidate_id": runtime_settings_candidate_id,
            "activation_generation": int(activation_generation),
            "vector_schema": schema.model_dump(mode="json"),
            "graph_state_ids": graph_state_ids,
        }
    )


def _validate_pointer(pointer: KnowledgeBaseVectorRuntimeState) -> FrozenVectorSchema:
    if pointer.protocol_version != VECTOR_RUNTIME_STATE_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Active vector runtime state {pointer.id} has an unsupported protocol"
        )
    if int(pointer.activation_generation or 0) < 1:
        raise RuntimeError(
            f"Active vector runtime state {pointer.id} has an invalid generation"
        )
    schema = _schema_from_pointer(pointer)
    expected_state_hash = vector_runtime_state_hash(
        knowledge_base_id=str(pointer.knowledge_base_id),
        runtime_settings_candidate_id=pointer.runtime_settings_candidate_id,
        activation_generation=int(pointer.activation_generation),
        schema=schema,
        graph_state_ids=_graph_state_ids_from_pointer(pointer),
    )
    if pointer.state_hash != expected_state_hash:
        raise RuntimeError(
            f"Active vector runtime state {pointer.id} has a non-canonical state_hash"
        )
    return schema


def _latest_active_graph_state_ids(
    db: Session,
    knowledge_base_id: str,
) -> dict[str, str | None]:
    context = db.scalar(
        select(ContextGraphState)
        .where(
            ContextGraphState.knowledge_base_id == knowledge_base_id,
            ContextGraphState.state == "active",
        )
        .order_by(ContextGraphState.created_at.desc())
    )
    if context is None:
        return {"context": None, "relation": None, "mid": None, "coarse": None}
    return {
        "context": context.id,
        "relation": context.chunk_relation_graph_state_id,
        "mid": context.mid_concept_state_id,
        "coarse": context.coarse_concept_state_id,
    }


def resolve_active_vector_runtime_target(
    db: Session,
    knowledge_base_id: str,
    *,
    for_update: bool = False,
) -> VectorRuntimeTarget:
    statement = select(KnowledgeBaseVectorRuntimeState).where(
        KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base_id
    )
    if for_update:
        statement = statement.with_for_update()
    pointer = db.scalar(statement)
    if pointer is None:
        raise RuntimeError(
            "Knowledge base has no active PostgreSQL vector runtime pointer; "
            "run an active contextual-index rebuild or explicit pointer bootstrap"
        )
    schema = _validate_pointer(pointer)
    return VectorRuntimeTarget(
        knowledge_base_id=str(knowledge_base_id),
        state_scope="active",
        schema=schema,
        vector_schema_hash=pointer.vector_schema_hash,
        runtime_state_id=pointer.id,
        runtime_state_hash=pointer.state_hash,
        activation_generation=int(pointer.activation_generation),
        runtime_settings_candidate_id=pointer.runtime_settings_candidate_id,
        vector_shadow_build_id=None,
        pending_vector_status="pending",
        ready_vector_status="ready",
    )


def ensure_active_vector_runtime_target(
    db: Session,
    knowledge_base_id: str,
) -> VectorRuntimeTarget:
    """Create the one-time PG active pointer before an active writer runs.

    The caller owns commit/rollback.  This function has no Qdrant, Redis or
    process-environment side effects.
    """

    knowledge_base = db.scalar(
        select(KnowledgeBase)
        .where(KnowledgeBase.id == knowledge_base_id)
        # A durable vector intent is committed through an independent audit
        # session while this ingest transaction remains open. PostgreSQL FK
        # validation takes KEY SHARE on this row, so NO KEY UPDATE preserves
        # the writer mutex without self-blocking that independent commit.
        .with_for_update(key_share=True)
    )
    if knowledge_base is None:
        raise ValueError(f"Unknown knowledge base: {knowledge_base_id}")
    pointer = db.scalar(
        select(KnowledgeBaseVectorRuntimeState)
        .where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base_id
        )
        .with_for_update()
    )
    if pointer is not None:
        return resolve_active_vector_runtime_target(
            db,
            knowledge_base_id,
            for_update=True,
        )

    settings = get_settings()
    schema = frozen_vector_schema(
        embedding_api_protocol=settings.embedding_api_protocol,
        embedding_model=settings.embedding_model,
        embedding_dimension=int(settings.embedding_dimensions),
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )
    graph_state_ids = _latest_active_graph_state_ids(db, knowledge_base_id)
    schema_hash = vector_schema_hash(schema)
    state_hash = vector_runtime_state_hash(
        knowledge_base_id=knowledge_base_id,
        runtime_settings_candidate_id=None,
        activation_generation=1,
        schema=schema,
        graph_state_ids=graph_state_ids,
    )
    pointer = KnowledgeBaseVectorRuntimeState(
        knowledge_base_id=knowledge_base_id,
        runtime_settings_candidate_id=None,
        protocol_version=VECTOR_RUNTIME_STATE_PROTOCOL_VERSION,
        embedding_model=schema.embedding_model,
        embedding_dimension=schema.embedding_dimension,
        distance_metric=schema.distance_metric,
        embedding_text_version=schema.embedding_text_version,
        chunk_schema_version=schema.chunk_schema_version,
        collection_identity_protocol_version=(
            schema.collection_identity_protocol_version
        ),
        collection_identity_digest=schema.collection_identity_digest,
        collection_name=schema.collection_name,
        vector_schema_hash=schema_hash,
        state_hash=state_hash,
        activation_generation=1,
        active_context_graph_state_id=graph_state_ids["context"],
        active_chunk_relation_graph_state_id=graph_state_ids["relation"],
        active_mid_concept_state_id=graph_state_ids["mid"],
        active_coarse_concept_state_id=graph_state_ids["coarse"],
        previous_state_json={},
        promotion_audit_json={
            "protocol_version": "vector_runtime_pointer_bootstrap_v1",
            "source": "active_contextual_index_writer",
            "active_env_mutated": False,
            "external_side_effects": False,
        },
    )
    db.add(pointer)
    db.flush()
    return resolve_active_vector_runtime_target(
        db,
        knowledge_base_id,
        for_update=True,
    )


def resolve_shadow_vector_runtime_target(
    db: Session,
    build_id: str,
    *,
    for_update: bool = False,
) -> VectorRuntimeTarget:
    statement = select(VectorShadowBuild).where(VectorShadowBuild.id == build_id)
    if for_update:
        statement = statement.with_for_update()
    build = db.scalar(statement)
    if build is None:
        raise ValueError(f"Unknown vector shadow build: {build_id}")
    candidate_statement = select(RuntimeSettingsCandidate).where(
        RuntimeSettingsCandidate.id == build.runtime_settings_candidate_id
    )
    if for_update:
        candidate_statement = candidate_statement.with_for_update()
    candidate = db.scalar(candidate_statement)
    if candidate is None:
        raise RuntimeError(f"Vector shadow build {build.id} has no candidate")
    if build.status not in {
        "building",
        "shadow_ready",
        "evaluating",
        "evaluation_passed",
        "promotion_blocked",
    }:
        raise RuntimeError(
            f"Vector shadow build {build.id} cannot be consumed from status {build.status}"
        )
    if candidate.status not in {
        "building",
        "evaluating",
        "evaluation_passed",
        "promotion_blocked",
    }:
        raise RuntimeError(
            f"Vector shadow candidate {candidate.id} cannot be consumed from status {candidate.status}"
        )
    schema = _build_schema(build)
    runtime_hash = _stable_hash(
        {
            "protocol_version": VECTOR_RUNTIME_TARGET_PROTOCOL_VERSION,
            "state_scope": "shadow",
            "runtime_settings_candidate_id": candidate.id,
            "candidate_hash": candidate.candidate_hash,
            "vector_shadow_build_id": build.id,
            "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
            "base_vector_state_hash": build.base_vector_state_hash,
        }
    )
    return VectorRuntimeTarget(
        knowledge_base_id=str(build.knowledge_base_id),
        state_scope="shadow",
        schema=schema,
        vector_schema_hash=build.candidate_vector_schema_hash,
        runtime_state_id=None,
        runtime_state_hash=runtime_hash,
        activation_generation=0,
        runtime_settings_candidate_id=candidate.id,
        runtime_settings_candidate_hash=_candidate_effective_runtime_hash(candidate),
        vector_shadow_build_id=build.id,
        pending_vector_status="shadow_pending",
        ready_vector_status="shadow_ready",
    )


def vector_runtime_diagnostics(target: VectorRuntimeTarget) -> dict[str, Any]:
    schema = target.schema
    diagnostics = {
        "vector_runtime_target_protocol_version": target.protocol_version,
        "vector_runtime_state_scope": target.state_scope,
        "vector_runtime_state_hash": target.runtime_state_hash,
        "activation_generation": target.activation_generation,
        "candidate_vector_schema_hash": target.vector_schema_hash,
        "embedding_model": schema.embedding_model,
        "embedding_dimension": schema.embedding_dimension,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "collection_name": schema.collection_name,
        "collection_identity_digest": schema.collection_identity_digest,
    }
    if target.runtime_state_id is not None:
        diagnostics["vector_runtime_state_id"] = target.runtime_state_id
    if target.runtime_settings_candidate_id is not None:
        diagnostics["runtime_settings_candidate_id"] = (
            target.runtime_settings_candidate_id
        )
    if target.runtime_settings_candidate_hash is not None:
        diagnostics["runtime_settings_candidate_hash"] = (
            target.runtime_settings_candidate_hash
        )
    if target.vector_shadow_build_id is not None:
        diagnostics["vector_shadow_build_id"] = target.vector_shadow_build_id
    if target.state_scope == "shadow":
        diagnostics.update(
            {
                "vector_shadow_writer_protocol_version": (
                    VECTOR_SHADOW_WRITER_PROTOCOL_VERSION
                ),
                "vector_runtime_consumer_protocol_version": (
                    VECTOR_SHADOW_GRAPH_CONSUMER_PROTOCOL_VERSION
                ),
            }
        )
    return diagnostics


def _bind_context_qdrant_proof_to_active_pointer(
    context_state: ContextGraphState,
    pointer: KnowledgeBaseVectorRuntimeState,
) -> None:
    diagnostics = dict(context_state.diagnostics_json or {})
    maintenance = diagnostics.get("contextual_index_maintenance")
    proof = (
        maintenance.get("qdrant_freshness")
        if isinstance(maintenance, dict)
        else None
    )
    if not isinstance(proof, dict) or proof.get("verified") is not True:
        raise RuntimeError(
            f"Context graph {context_state.id} lacks a verified Qdrant proof"
        )
    if (
        proof.get("collection_name") != pointer.collection_name
        or proof.get("vector_schema_hash") != pointer.vector_schema_hash
    ):
        raise RuntimeError(
            f"Context graph {context_state.id} Qdrant proof does not match the active vector schema"
        )
    rebound_proof = dict(proof)
    rebound_proof.setdefault(
        "observed_vector_runtime_state_scope",
        proof.get("vector_runtime_state_scope"),
    )
    rebound_proof.setdefault(
        "observed_vector_runtime_state_hash",
        proof.get("vector_runtime_state_hash"),
    )
    rebound_proof.update(
        {
            "vector_runtime_state_scope": "active",
            "vector_runtime_state_hash": pointer.state_hash,
            "active_pointer_rebound": True,
            "active_pointer_rebound_at": datetime.utcnow().isoformat(),
        }
    )
    context_state.diagnostics_json = {
        **diagnostics,
        "contextual_index_maintenance": {
            **dict(maintenance),
            "qdrant_freshness": rebound_proof,
        },
    }


def bind_active_vector_runtime_graph(
    db: Session,
    *,
    target: VectorRuntimeTarget,
    context_state: ContextGraphState,
) -> VectorRuntimeTarget:
    """Bind a newly built active four-layer graph to the same PG pointer."""

    if target.state_scope != "active" or target.runtime_state_id is None:
        raise ValueError("Only an active vector runtime target can bind an active graph")
    pointer = db.scalar(
        select(KnowledgeBaseVectorRuntimeState)
        .where(KnowledgeBaseVectorRuntimeState.id == target.runtime_state_id)
        .with_for_update()
    )
    if pointer is None:
        raise RuntimeError("Active vector runtime pointer disappeared during graph build")
    _validate_pointer(pointer)
    if pointer.state_hash != target.runtime_state_hash:
        raise RuntimeError("Active vector runtime pointer changed during graph build")
    if (
        str(context_state.knowledge_base_id) != str(target.knowledge_base_id)
        or context_state.state != "active"
    ):
        raise RuntimeError("Active context graph provenance/state mismatch")
    graph_state_ids = {
        "context": context_state.id,
        "relation": context_state.chunk_relation_graph_state_id,
        "mid": context_state.mid_concept_state_id,
        "coarse": context_state.coarse_concept_state_id,
    }
    pointer.active_context_graph_state_id = graph_state_ids["context"]
    pointer.active_chunk_relation_graph_state_id = graph_state_ids["relation"]
    pointer.active_mid_concept_state_id = graph_state_ids["mid"]
    pointer.active_coarse_concept_state_id = graph_state_ids["coarse"]
    pointer.state_hash = vector_runtime_state_hash(
        knowledge_base_id=pointer.knowledge_base_id,
        runtime_settings_candidate_id=pointer.runtime_settings_candidate_id,
        activation_generation=pointer.activation_generation,
        schema=_schema_from_pointer(pointer),
        graph_state_ids=graph_state_ids,
    )
    audit = {
        "protocol_version": "active_vector_graph_pointer_binding_v1",
        "context_graph_state_id": context_state.id,
        "graph_state_ids": graph_state_ids,
        "vector_schema_hash": pointer.vector_schema_hash,
        "state_hash": pointer.state_hash,
        "bound_at": datetime.utcnow().isoformat(),
    }
    pointer.promotion_audit_json = {
        **dict(pointer.promotion_audit_json or {}),
        "latest_active_graph_binding": audit,
    }
    for model, state_id in (
        (ContextGraphState, graph_state_ids["context"]),
        (ChunkRelationGraphState, graph_state_ids["relation"]),
        (MidConceptState, graph_state_ids["mid"]),
        (CoarseConceptState, graph_state_ids["coarse"]),
    ):
        row = db.get(model, state_id) if state_id else None
        if row is not None:
            row.diagnostics_json = {
                **dict(row.diagnostics_json or {}),
                "active_vector_runtime_state_id": pointer.id,
                "active_vector_runtime_state_hash": pointer.state_hash,
                "active_vector_schema_hash": pointer.vector_schema_hash,
            }
            if isinstance(row, ContextGraphState):
                _bind_context_qdrant_proof_to_active_pointer(row, pointer)
    db.flush()
    return resolve_active_vector_runtime_target(
        db,
        target.knowledge_base_id,
        for_update=True,
    )


def _base_vector_state(
    db: Session,
    knowledge_base_id: str,
) -> tuple[FrozenVectorSchema, str, KnowledgeBaseVectorRuntimeState | None]:
    pointer = db.scalar(
        select(KnowledgeBaseVectorRuntimeState)
        .where(KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base_id)
        .with_for_update()
    )
    if pointer is not None:
        schema = _validate_pointer(pointer)
        return schema, pointer.state_hash, pointer
    settings = get_settings()
    schema = frozen_vector_schema(
        embedding_api_protocol=settings.embedding_api_protocol,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimensions,
    )
    state_hash = _stable_hash(
        {
            "protocol_version": "legacy_global_vector_runtime_state_v1",
            "knowledge_base_id": knowledge_base_id,
            "vector_schema": schema.model_dump(mode="json"),
        }
    )
    return schema, state_hash, None


def stage_vector_runtime_candidate(
    db: Session,
    *,
    knowledge_base_ids: Sequence[str],
    embedding_api_protocol: Literal["openai"] = "openai",
    embedding_model: str,
    embedding_dimension: int,
    embedding_text_version: str = CURRENT_EMBEDDING_TEXT_VERSION,
    chunk_schema_version: str = CHUNK_SCHEMA_VERSION,
    source: str = "api",
    base_runtime_version_hash: str | None = None,
) -> tuple[RuntimeSettingsCandidate, list[VectorShadowBuild]]:
    """Stage immutable vector settings without touching active env/runtime state.

    The caller owns commit/rollback.  No Qdrant or Redis side effect is emitted
    here; one candidate and one build intent per KB are written atomically.
    """

    target_ids = sorted({str(value).strip() for value in knowledge_base_ids if str(value).strip()})
    if not target_ids:
        raise ValueError("knowledge_base_ids must contain at least one id")
    if len(target_ids) > MAX_CANDIDATE_KNOWLEDGE_BASES:
        raise ValueError(
            f"A vector runtime candidate may target at most {MAX_CANDIDATE_KNOWLEDGE_BASES} knowledge bases"
        )
    schema = frozen_vector_schema(
        embedding_api_protocol=embedding_api_protocol,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_text_version=embedding_text_version,
        chunk_schema_version=chunk_schema_version,
    )
    schema_json = schema.model_dump(mode="json")
    schema_hash = vector_schema_hash(schema)
    from app.services.vector_collection_cleanup import (
        assert_vector_collection_not_pending_cleanup,
    )

    assert_vector_collection_not_pending_cleanup(db, schema.collection_name)

    knowledge_bases = list(
        db.scalars(
            select(KnowledgeBase)
            .where(KnowledgeBase.id.in_(target_ids))
            .order_by(KnowledgeBase.id.asc())
            .with_for_update()
        ).all()
    )
    found_ids = {item.id for item in knowledge_bases}
    missing = sorted(set(target_ids) - found_ids)
    if missing:
        raise ValueError("Unknown knowledge_base_ids: " + ", ".join(missing))

    base_states: dict[str, str] = {}
    staged_scope: dict[str, dict[str, Any]] = {}
    changed_keys: set[str] = set()
    for knowledge_base_id in target_ids:
        active_chunks = _active_chunks(db, knowledge_base_id)
        if not active_chunks:
            raise ValueError(
                f"Knowledge base {knowledge_base_id} has no active chunks for a shadow rebuild"
            )
        active_chunk_ids = [chunk.id for chunk in active_chunks]
        base_schema, base_state_hash, _pointer = _base_vector_state(db, knowledge_base_id)
        if vector_schema_hash(base_schema) == schema_hash:
            raise ValueError(
                f"Knowledge base {knowledge_base_id} already uses the requested vector schema"
            )
        base_states[knowledge_base_id] = base_state_hash
        staged_scope[knowledge_base_id] = {
            "chunk_ids": active_chunk_ids,
            "fingerprint": compute_chunk_scope_hash(active_chunks),
            "expected_point_count": len(active_chunk_ids),
        }
        if base_schema.embedding_model != schema.embedding_model:
            changed_keys.add("embedding_model")
        if base_schema.embedding_dimension != schema.embedding_dimension:
            changed_keys.add("embedding_dimensions")
        if base_schema.embedding_text_version != schema.embedding_text_version:
            changed_keys.add("embedding_text_version")
        if base_schema.chunk_schema_version != schema.chunk_schema_version:
            changed_keys.add("chunk_schema_version")

    if base_runtime_version_hash is None:
        from app.services.runtime_settings import model_settings_payload

        base_runtime_version_hash = _stable_hash(
            {
                "protocol_version": "runtime_settings_snapshot_hash_v1",
                "settings": model_settings_payload(include_dynamic_status=False),
            }
        )
    else:
        base_runtime_version_hash = _require_hash64(
            base_runtime_version_hash,
            field_name="base_runtime_version_hash",
        )

    settings_json = {
        "protocol_version": VECTOR_RUNTIME_CANDIDATE_PROTOCOL_VERSION,
        "embedding_model": schema.embedding_model,
        "embedding_dimensions": schema.embedding_dimension,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "vector_schema": schema_json,
    }
    candidate_hash = _stable_hash(
        {
            "protocol_version": VECTOR_RUNTIME_CANDIDATE_PROTOCOL_VERSION,
            "base_runtime_version_hash": base_runtime_version_hash,
            "settings": settings_json,
            "changed_keys": sorted(changed_keys),
            "target_knowledge_base_ids": target_ids,
            "base_vector_state_hashes": base_states,
            "staged_scope_fingerprints": {
                key: value["fingerprint"] for key, value in staged_scope.items()
            },
        }
    )
    existing = db.scalar(
        select(RuntimeSettingsCandidate).where(
            RuntimeSettingsCandidate.candidate_hash == candidate_hash
        )
    )
    if existing is not None:
        if existing.status in {
            "promoted",
            "rejected",
            "failed",
            "rolled_back",
            "superseded",
        }:
            raise RuntimeError(
                f"The exact vector runtime candidate is already terminal as {existing.status}"
            )
        builds = list(
            db.scalars(
                select(VectorShadowBuild)
                .where(VectorShadowBuild.runtime_settings_candidate_id == existing.id)
                .order_by(VectorShadowBuild.knowledge_base_id.asc())
            ).all()
        )
        return existing, builds

    competing_builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(
                VectorShadowBuild.knowledge_base_id.in_(target_ids),
                VectorShadowBuild.status.in_(LIVE_SHADOW_BUILD_STATUSES),
            )
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
            .with_for_update()
        ).all()
    )
    if competing_builds:
        conflicts = ", ".join(
            f"{build.knowledge_base_id}:{build.id}:{build.status}"
            for build in competing_builds
        )
        raise RuntimeError(
            "A live vector shadow build already owns one or more target knowledge bases: "
            + conflicts
        )

    candidate = RuntimeSettingsCandidate(
        protocol_version=VECTOR_RUNTIME_CANDIDATE_PROTOCOL_VERSION,
        candidate_hash=candidate_hash,
        base_runtime_version_hash=base_runtime_version_hash,
        settings_json=settings_json,
        changed_keys_json=sorted(changed_keys),
        target_knowledge_base_ids_json=target_ids,
        lifecycle_scope="rebuild_required",
        status="staged",
        source=str(source or "api")[:64],
        diagnostics_json={
            "protocol_version": VECTOR_RUNTIME_CANDIDATE_PROTOCOL_VERSION,
            "base_vector_state_hashes": base_states,
            "active_env_mutated": False,
            "runtime_version_broadcast": False,
            "transaction_owner": "caller",
            "active_chunk_read_page_size": ACTIVE_CHUNK_READ_PAGE_SIZE,
        },
        blocking_reasons_json=[],
    )
    db.add(candidate)
    db.flush()

    builds: list[VectorShadowBuild] = []
    for knowledge_base_id in target_ids:
        scope = staged_scope[knowledge_base_id]
        build = VectorShadowBuild(
            runtime_settings_candidate_id=candidate.id,
            knowledge_base_id=knowledge_base_id,
            protocol_version=VECTOR_SHADOW_BUILD_PROTOCOL_VERSION,
            status="staged",
            base_vector_state_hash=base_states[knowledge_base_id],
            candidate_vector_schema_json=schema_json,
            candidate_vector_schema_hash=schema_hash,
            embedding_model=schema.embedding_model,
            embedding_dimension=schema.embedding_dimension,
            distance_metric=schema.distance_metric,
            embedding_text_version=schema.embedding_text_version,
            chunk_schema_version=schema.chunk_schema_version,
            collection_identity_protocol_version=(
                schema.collection_identity_protocol_version
            ),
            collection_identity_digest=schema.collection_identity_digest,
            collection_name=schema.collection_name,
            expected_point_count=scope["expected_point_count"],
            ready_point_count=0,
            qdrant_proof_json={},
            evaluation_result_json={},
            promotion_audit_json={},
            rollback_audit_json={},
            diagnostics_json={
                "staged_active_chunk_scope_hash_protocol_version": (
                    CHUNK_SCOPE_HASH_PROTOCOL_VERSION
                ),
                "staged_active_chunk_scope_hash": scope["fingerprint"],
                "staged_scope_filter_hash": _scope_filter_hash(
                    knowledge_base_id,
                    scope["chunk_ids"],
                ),
                "candidate_hash": candidate_hash,
                "active_pointer_mutated": False,
            },
            blocking_reasons_json=[],
        )
        db.add(build)
        builds.append(build)
    db.flush()
    return candidate, builds


def _locked_build_candidate(
    db: Session,
    build_id: str,
) -> tuple[VectorShadowBuild, RuntimeSettingsCandidate]:
    build = db.scalar(
        select(VectorShadowBuild)
        .where(VectorShadowBuild.id == build_id)
        .with_for_update()
    )
    if build is None:
        raise ValueError(f"Unknown vector shadow build: {build_id}")
    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == build.runtime_settings_candidate_id)
        .with_for_update()
    )
    if candidate is None:
        raise RuntimeError(f"Vector shadow build {build.id} has no candidate")
    return build, candidate


def _assert_staged_scope_unchanged(db: Session, build: VectorShadowBuild) -> list[str]:
    active_chunks = _active_chunks(db, build.knowledge_base_id)
    active_chunk_ids = [chunk.id for chunk in active_chunks]
    diagnostics = dict(build.diagnostics_json or {})
    if (
        diagnostics.get("staged_active_chunk_scope_hash_protocol_version")
        != CHUNK_SCOPE_HASH_PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "Vector shadow build lacks the frozen complete-address chunk scope protocol"
        )
    expected_hash = str(diagnostics.get("staged_active_chunk_scope_hash") or "")
    actual_hash = compute_chunk_scope_hash(active_chunks)
    base_expected_count = int(
        diagnostics.get("base_active_expected_point_count", build.expected_point_count)
    )
    if actual_hash != expected_hash or len(active_chunk_ids) != base_expected_count:
        raise RuntimeError(
            "Active chunk scope changed after candidate staging; create a new candidate"
        )
    candidate_chunk_ids = [
        str(value)
        for value in (diagnostics.get("candidate_chunk_ids") or [])
        if str(value)
    ]
    if not candidate_chunk_ids:
        return active_chunk_ids
    candidate_chunks = list(
        db.scalars(
            select(Chunk)
            .where(
                Chunk.knowledge_base_id == build.knowledge_base_id,
                Chunk.id.in_(candidate_chunk_ids),
                Chunk.state == "shadow",
            )
            .order_by(Chunk.id.asc())
        ).all()
    )
    canonical_ids = sorted(candidate_chunk_ids)
    if [str(chunk.id) for chunk in candidate_chunks] != canonical_ids:
        raise RuntimeError("Candidate shadow chunk scope is incomplete or no longer shadow")
    expected_candidate_hash = str(
        diagnostics.get("candidate_chunk_scope_hash") or ""
    )
    if (
        not expected_candidate_hash
        or compute_chunk_scope_hash(candidate_chunks) != expected_candidate_hash
        or len(candidate_chunks) != int(build.expected_point_count)
    ):
        raise RuntimeError("Candidate shadow chunk scope changed after bounded rechunk")
    mismatched_schema = [
        str(chunk.id)
        for chunk in candidate_chunks
        if str((chunk.metadata_json or {}).get("chunk_schema_version") or "")
        != str(build.chunk_schema_version)
    ]
    if mismatched_schema:
        raise RuntimeError(
            "Candidate shadow chunks do not match the vector chunk schema: "
            + ", ".join(mismatched_schema[:8])
        )
    return canonical_ids


def _shadow_build_scope_chunks(db: Session, build: VectorShadowBuild) -> list[Chunk]:
    chunk_ids = _assert_staged_scope_unchanged(db, build)
    chunks = list(
        db.scalars(
            select(Chunk).where(Chunk.id.in_(chunk_ids)).order_by(Chunk.id.asc())
        ).all()
    )
    if [str(chunk.id) for chunk in chunks] != list(chunk_ids):
        raise RuntimeError("Vector shadow build chunk scope disappeared")
    return chunks


def start_vector_shadow_build(db: Session, build_id: str) -> VectorShadowBuild:
    build, candidate = _locked_build_candidate(db, build_id)
    if build.status == "building":
        return build
    if build.status != "staged" or candidate.status not in {"staged", "building"}:
        raise RuntimeError(
            f"Cannot start vector shadow build {build.id} from {build.status}/{candidate.status}"
        )
    _assert_staged_scope_unchanged(db, build)
    now = datetime.utcnow()
    build.status = "building"
    build.started_at = now
    build.error_code = None
    build.last_error = None
    candidate.status = "building"
    db.flush()
    return build


def _active_concept_semantic_reuse_sources(
    db: Session,
    build: VectorShadowBuild,
) -> tuple[MidConceptState | None, CoarseConceptState | None, dict[str, Any]]:
    """Select an exact terminal-shadow source or the staged active pointer.

    The semantic-reuse index performs its own strict packet/profile/protocol
    identity checks. A terminal source is admitted only for the same complete
    candidate vector/chunk scope and makes every semantic packet hit mandatory;
    otherwise the frozen active vector pointer remains the optional source and
    a miss may call the provider within its hard budget.
    """

    chunk_scope_hash = compute_chunk_scope_hash(
        _shadow_build_scope_chunks(db, build)
    )
    terminal_builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(
                VectorShadowBuild.knowledge_base_id
                == build.knowledge_base_id,
                VectorShadowBuild.id != build.id,
                VectorShadowBuild.candidate_vector_schema_hash
                == build.candidate_vector_schema_hash,
                VectorShadowBuild.status.in_({"rejected", "superseded"}),
            )
            .order_by(
                VectorShadowBuild.created_at.desc(),
                VectorShadowBuild.id.desc(),
            )
            .limit(MAX_VECTOR_SHADOW_TERMINAL_CONCEPT_SOURCES + 1)
        ).all()
    )
    if len(terminal_builds) > MAX_VECTOR_SHADOW_TERMINAL_CONCEPT_SOURCES:
        raise RuntimeError(
            "Vector shadow terminal concept reuse refused an unbounded source scan"
        )
    for source_build in terminal_builds:
        source_candidate = db.get(
            RuntimeSettingsCandidate,
            source_build.runtime_settings_candidate_id,
        )
        if (
            source_candidate is None
            or source_candidate.status not in {"rejected", "superseded"}
            or source_candidate.status != source_build.status
            or source_build.chunk_scope_hash != chunk_scope_hash
            or not source_build.qdrant_proof_hash
            or not source_build.shadow_context_graph_state_id
            or not source_build.shadow_mid_concept_state_id
            or not source_build.shadow_coarse_concept_state_id
        ):
            continue
        context_state = db.get(
            ContextGraphState, source_build.shadow_context_graph_state_id
        )
        mid_state = db.get(
            MidConceptState, source_build.shadow_mid_concept_state_id
        )
        coarse_state = db.get(
            CoarseConceptState, source_build.shadow_coarse_concept_state_id
        )
        if (
            context_state is None
            or mid_state is None
            or coarse_state is None
            or context_state.state != "shadow"
            or mid_state.state != "shadow"
            or coarse_state.state != "shadow"
            or str(context_state.knowledge_base_id)
            != str(build.knowledge_base_id)
            or str(mid_state.knowledge_base_id) != str(build.knowledge_base_id)
            or str(coarse_state.knowledge_base_id)
            != str(build.knowledge_base_id)
            or context_state.chunk_scope_hash != chunk_scope_hash
            or str(context_state.mid_concept_state_id) != str(mid_state.id)
            or str(context_state.coarse_concept_state_id)
            != str(coarse_state.id)
        ):
            continue
        return (
            mid_state,
            coarse_state,
            {
                "protocol_version": (
                    VECTOR_SHADOW_CONCEPT_SEMANTIC_REUSE_PROTOCOL_VERSION
                ),
                "source_available": True,
                "source_kind": "terminal_shadow",
                "source_candidate_id": source_candidate.id,
                "source_build_id": source_build.id,
                "source_mid_concept_state_id": mid_state.id,
                "source_coarse_concept_state_id": coarse_state.id,
                "source_context_graph_state_id": context_state.id,
                "source_vector_schema_hash": (
                    source_build.candidate_vector_schema_hash
                ),
                "source_chunk_scope_hash": source_build.chunk_scope_hash,
                "bounded_source_scan_count": len(terminal_builds),
                "strict_semantic_identity_required": True,
                "exact_reuse_required": True,
                "provider_allowed_on_miss": False,
                "provider_response_persisted": False,
            },
        )

    pointer = db.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id
            == build.knowledge_base_id
        )
    )
    mid_state, coarse_state, audit = (
        _active_concept_semantic_reuse_sources_from_pointer(
        db,
        build=build,
        pointer=pointer,
        )
    )
    return mid_state, coarse_state, {
        **audit,
        "source_kind": "active_pointer",
        "bounded_source_scan_count": len(terminal_builds),
        "exact_reuse_required": False,
        "provider_response_persisted": False,
    }


def _compensated_shadow_embedding_recovery(
    db: Session,
    *,
    build: VectorShadowBuild,
    chunks: Sequence[Chunk],
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any]]:
    """Recover exact provider-produced vectors from canonical compensated outboxes."""

    from app.services.qdrant_outbox import (
        QDRANT_UPSERT_OPERATION,
        validated_qdrant_outbox_target_points,
    )
    from app.services.vector_store import canonical_embedding_vector

    schema = _build_schema(build)
    rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.knowledge_base_id
                == build.knowledge_base_id,
                IngestionCompensationLog.operation == QDRANT_UPSERT_OPERATION,
                IngestionCompensationLog.status == "compensated",
                IngestionCompensationLog.payload_json[
                    "collection_name"
                ].as_string()
                == schema.collection_name,
            )
            .order_by(
                IngestionCompensationLog.created_at.desc(),
                IngestionCompensationLog.id.desc(),
            )
            .limit(MAX_VECTOR_SHADOW_COMPENSATED_RECOVERY_INTENTS + 1)
        ).all()
    )
    if len(rows) > MAX_VECTOR_SHADOW_COMPENSATED_RECOVERY_INTENTS:
        raise RuntimeError(
            "Vector shadow compensated recovery refused an unbounded intent scan"
        )
    expected_chunk_ids = {str(chunk.id) for chunk in chunks}
    cards: dict[str, dict[str, Any]] = {}
    contributing_intent_ids: set[str] = set()
    source_bindings: set[tuple[str, str]] = set()
    source_admission_cache: dict[tuple[str, str], bool] = {}
    expected_payload_values = {
        "knowledge_base_id": str(build.knowledge_base_id),
        "embedding_model": schema.embedding_model,
        "embedding_dimension": schema.embedding_dimension,
        "vector_distance_metric": schema.distance_metric,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "collection_identity_protocol_version": (
            schema.collection_identity_protocol_version
        ),
        "collection_identity_digest": schema.collection_identity_digest,
        "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
        "vector_shadow_writer_protocol_version": (
            VECTOR_SHADOW_WRITER_PROTOCOL_VERSION
        ),
    }
    for row in rows:
        payload_json = dict(row.payload_json or {})
        if payload_json.get("collection_name") != schema.collection_name:
            continue
        for point in validated_qdrant_outbox_target_points(row):
            point_id = str(point.get("id") or "")
            payload = point.get("payload")
            if point_id not in expected_chunk_ids or not isinstance(payload, dict):
                continue
            if any(
                payload.get(field_name) != expected
                for field_name, expected in expected_payload_values.items()
            ):
                continue
            source_candidate_id = str(
                payload.get("runtime_settings_candidate_id") or ""
            )
            source_build_id = str(payload.get("vector_shadow_build_id") or "")
            source_binding = (source_candidate_id, source_build_id)
            if not all(source_binding):
                continue
            source_admitted = source_admission_cache.get(source_binding)
            if source_admitted is None:
                source_candidate = db.get(
                    RuntimeSettingsCandidate, source_candidate_id
                )
                source_build = db.get(VectorShadowBuild, source_build_id)
                same_build = source_binding == (
                    str(build.runtime_settings_candidate_id),
                    str(build.id),
                )
                source_admitted = bool(
                    source_candidate is not None
                    and source_build is not None
                    and str(source_build.runtime_settings_candidate_id)
                    == source_candidate_id
                    and str(source_build.knowledge_base_id)
                    == str(build.knowledge_base_id)
                    and str(source_build.candidate_vector_schema_hash)
                    == str(build.candidate_vector_schema_hash)
                    and (
                        same_build
                        or (
                            source_candidate.status
                            in {"rejected", "superseded"}
                            and source_build.status
                            in {"rejected", "superseded"}
                        )
                    )
                )
                source_admission_cache[source_binding] = source_admitted
            if not source_admitted:
                continue
            if payload.get("chunk_id") != point_id:
                raise RuntimeError(
                    "Compensated vector recovery point/chunk identity mismatch"
                )
            vector = canonical_embedding_vector(
                point.get("vector"),
                source=f"Compensated vector shadow point {point_id}",
            )
            if len(vector) != int(schema.embedding_dimension):
                raise RuntimeError(
                    "Compensated vector recovery dimension mismatch"
                )
            vector_payload_hash_value = payload.get("vector_payload_hash")
            if (
                not isinstance(vector_payload_hash_value, str)
                or len(vector_payload_hash_value) != 64
            ):
                raise RuntimeError(
                    "Compensated vector recovery lacks a canonical payload hash"
                )
            card = {
                "knowledge_base_id": str(build.knowledge_base_id),
                "chunk_id": point_id,
                "embedding_model": schema.embedding_model,
                "embedding_dimension": schema.embedding_dimension,
                "vector_distance_metric": schema.distance_metric,
                "embedding_text_version": schema.embedding_text_version,
                "chunk_schema_version": schema.chunk_schema_version,
                "collection_identity_protocol_version": (
                    schema.collection_identity_protocol_version
                ),
                "collection_identity_digest": schema.collection_identity_digest,
                "runtime_settings_candidate_id": build.runtime_settings_candidate_id,
                "vector_shadow_build_id": build.id,
                "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
                "vector_shadow_writer_protocol_version": payload.get(
                    "vector_shadow_writer_protocol_version"
                ),
                "context_hash_protocol_version": payload.get(
                    "context_hash_protocol_version"
                ),
                "context_hash": payload.get("context_hash"),
                "local_hint_protocol_version": payload.get(
                    "local_hint_protocol_version"
                ),
                "local_hint_hash": payload.get("local_hint_hash"),
                "vector_payload_hash": vector_payload_hash_value,
                "vector": vector,
            }
            previous = cards.get(point_id)
            if previous is not None and _stable_hash(previous) != _stable_hash(card):
                raise RuntimeError(
                    "Compensated vector recovery found conflicting exact candidates"
                )
            cards[point_id] = card
            contributing_intent_ids.add(str(row.id))
            source_bindings.add(source_binding)

    recovered_ids = set(cards)
    complete = recovered_ids == expected_chunk_ids
    audit = {
        "protocol_version": (
            VECTOR_SHADOW_COMPENSATED_EMBEDDING_RECOVERY_PROTOCOL_VERSION
        ),
        "bounded_intent_scan_count": len(rows),
        "contributing_intent_count": len(contributing_intent_ids),
        "contributing_intent_set_hash": _stable_hash(
            sorted(contributing_intent_ids)
        ),
        "expected_chunk_count": len(expected_chunk_ids),
        "recovered_chunk_count": len(recovered_ids),
        "complete": complete,
        "embedding_provider_call_count": 0 if complete else None,
        "provider_response_persisted": False,
        "credential_value_persisted": False,
        "source_binding_count": len(source_bindings),
        "source_binding_set_hash": _stable_hash(sorted(source_bindings)),
        "target_runtime_settings_candidate_id": str(
            build.runtime_settings_candidate_id
        ),
        "target_vector_shadow_build_id": str(build.id),
        "cross_candidate_rebound": any(
            binding
            != (str(build.runtime_settings_candidate_id), str(build.id))
            for binding in source_bindings
        ),
    }
    if not complete:
        return None, audit
    audit["recovered_vector_set_hash"] = _stable_hash(
        [
            {
                "chunk_id": chunk_id,
                "vector_payload_hash": cards[chunk_id]["vector_payload_hash"],
            }
            for chunk_id in sorted(cards)
        ]
    )
    return cards, audit


def _active_concept_semantic_reuse_sources_from_pointer(
    db: Session,
    *,
    build: VectorShadowBuild,
    pointer: KnowledgeBaseVectorRuntimeState | None,
) -> tuple[MidConceptState | None, CoarseConceptState | None, dict[str, Any]]:
    if pointer is None:
        return None, None, {
            "protocol_version": (
                VECTOR_SHADOW_CONCEPT_SEMANTIC_REUSE_PROTOCOL_VERSION
            ),
            "source_available": False,
            "source_reason": "active_vector_runtime_pointer_missing",
            "strict_semantic_identity_required": True,
            "provider_allowed_on_miss": True,
        }
    _validate_pointer(pointer)
    if pointer.state_hash != build.base_vector_state_hash:
        raise RuntimeError(
            "Active vector runtime pointer changed after vector candidate staging"
        )

    source_rows: dict[str, MidConceptState | CoarseConceptState | None] = {
        "mid": (
            db.get(MidConceptState, pointer.active_mid_concept_state_id)
            if pointer.active_mid_concept_state_id
            else None
        ),
        "coarse": (
            db.get(CoarseConceptState, pointer.active_coarse_concept_state_id)
            if pointer.active_coarse_concept_state_id
            else None
        ),
    }
    source_ids = {
        "mid": pointer.active_mid_concept_state_id,
        "coarse": pointer.active_coarse_concept_state_id,
    }
    for layer, state in source_rows.items():
        source_id = source_ids[layer]
        if source_id and state is None:
            raise RuntimeError(
                f"Active vector runtime pointer references a missing {layer} concept state"
            )
        if state is not None and (
            str(state.knowledge_base_id) != str(build.knowledge_base_id)
            or state.state != "active"
        ):
            raise RuntimeError(
                f"Active vector runtime pointer {layer} concept source is not active/in-scope"
            )

    return (
        source_rows["mid"],
        source_rows["coarse"],
        {
            "protocol_version": (
                VECTOR_SHADOW_CONCEPT_SEMANTIC_REUSE_PROTOCOL_VERSION
            ),
            "source_available": any(source_rows.values()),
            "source_mid_concept_state_id": source_ids["mid"],
            "source_coarse_concept_state_id": source_ids["coarse"],
            "source_vector_runtime_state_id": pointer.id,
            "source_vector_runtime_state_hash": pointer.state_hash,
            "strict_semantic_identity_required": True,
            "provider_allowed_on_miss": True,
        },
    )


def record_vector_shadow_build_attempt_failure(
    db: Session,
    build_id: str,
    *,
    error_type: str,
    failure: BaseException | None = None,
) -> VectorShadowBuild:
    """Persist a safe retry audit after the caller rolled back partial facts.

    Provider/Qdrant response text is intentionally not accepted.  The build
    remains staged so the exact immutable candidate can be retried without
    rewriting or deleting its lifecycle identity.
    """

    build, candidate = _locked_build_candidate(db, build_id)
    if build.status != "staged":
        raise RuntimeError(
            "Vector shadow build failure audit requires the partial build transaction "
            "to be rolled back to staged first"
        )
    safe_error_type = str(error_type or "Exception").strip()[:128] or "Exception"
    previous = dict((build.diagnostics_json or {}).get("last_failed_attempt") or {})
    audit = {
        "protocol_version": "vector_shadow_build_attempt_failure_v1",
        "attempt_count": int(previous.get("attempt_count") or 0) + 1,
        "error_type": safe_error_type,
        "recorded_at": datetime.utcnow().isoformat(),
        "partial_postgresql_facts_rolled_back": True,
        "retry_boundary": "same_staged_build_id",
        "provider_response_persisted": False,
    }
    if failure is not None:
        audit["safe_failure_diagnostics"] = (
            vector_shadow_safe_failure_diagnostics(failure)
        )
    build.error_code = "vector_shadow_build_attempt_failed"
    build.last_error = safe_error_type
    build.diagnostics_json = {
        **dict(build.diagnostics_json or {}),
        "last_failed_attempt": audit,
    }
    candidate.status = "staged"
    candidate.error_code = "vector_shadow_build_attempt_failed"
    candidate.last_error = safe_error_type
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "last_failed_build_attempt": {
            **audit,
            "build_id": build.id,
            "knowledge_base_id": build.knowledge_base_id,
        },
    }
    db.flush()
    return build


def vector_shadow_safe_failure_diagnostics(exc: BaseException) -> dict[str, Any]:
    """Classify a shadow-build failure without copying exception messages."""

    chain: list[BaseException] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and len(chain) < 8 and id(current) not in visited:
        visited.add(id(current))
        chain.append(current)
        next_error = current.__cause__
        if next_error is None or id(next_error) in visited:
            next_error = current.__context__
        current = next_error

    def safe_type_name(error: BaseException) -> str:
        value = type(error).__name__
        return (
            value
            if value and len(value) <= 128 and value.replace("_", "").isalnum()
            else "Exception"
        )

    error_chain_types = [safe_type_name(error) for error in chain]
    batch_error = next(
        (
            error
            for error in chain
            if safe_type_name(error) == "ConceptProviderBatchError"
        ),
        None,
    )
    budget_error = next(
        (
            error
            for error in chain
            if safe_type_name(error) == "ConceptProviderRequestBudgetExceeded"
        ),
        None,
    )
    classification = (
        "concept_provider_request_budget_exhausted"
        if budget_error is not None
        else (
            "concept_provider_batch_failure"
            if batch_error is not None
            else "shadow_build_failure"
        )
    )
    card: dict[str, Any] = {
        "protocol_version": "vector_shadow_safe_failure_diagnostics_v1",
        "classification": classification,
        "outer_error_type": safe_type_name(exc),
        "error_chain_types": error_chain_types,
        "external_failure": external_failure_classification(exc),
        "provider_response_persisted": False,
        "credential_value_persisted": False,
    }
    if batch_error is not None:
        layer = str(getattr(batch_error, "layer", "unknown"))
        batch_index = getattr(batch_error, "batch_index", None)
        packet_ids = getattr(batch_error, "packet_ids", [])
        card["provider_batch"] = {
            "layer": layer if layer in {"mid", "coarse"} else "unknown",
            "batch_index": (
                batch_index
                if type(batch_index) is int and 0 <= batch_index <= 1_000_000
                else None
            ),
            "packet_count": (
                min(8, len(packet_ids)) if isinstance(packet_ids, list) else None
            ),
        }
    if budget_error is not None:
        source = getattr(budget_error, "diagnostics", None)
        source = source if isinstance(source, dict) else {}
        budget_card: dict[str, Any] = {
            "protocol_version": "concept_provider_request_budget_v1",
            "layer": (
                source.get("layer")
                if source.get("layer") in {"mid", "coarse"}
                else "unknown"
            ),
            "provider_response_persisted": False,
        }
        for field_name in (
            "miss_count",
            "max_requests",
            "reserved_requests",
            "observed_requests",
            "next_group_worst_case_requests",
        ):
            value = source.get(field_name)
            budget_card[field_name] = (
                value if type(value) is int and 0 <= value <= 1_000_000 else None
            )
        diagnostics_hash = source.get("diagnostics_hash")
        budget_card["diagnostics_hash"] = (
            diagnostics_hash
            if isinstance(diagnostics_hash, str)
            and len(diagnostics_hash) == 64
            and all(char in "0123456789abcdef" for char in diagnostics_hash)
            else None
        )
        card["provider_request_budget"] = budget_card
    return card


def abandon_vector_shadow_candidate(
    db: Session,
    candidate_id: str,
    *,
    disposition: Literal["rejected", "superseded"] = "rejected",
    reason: str,
) -> RuntimeSettingsCandidate:
    """Close a non-promoted candidate while retaining all derived facts."""

    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == candidate_id)
        .with_for_update()
    )
    if candidate is None:
        raise ValueError(f"Unknown runtime settings candidate: {candidate_id}")
    if candidate.status in {"promoted", "rolled_back"}:
        raise RuntimeError(
            f"Candidate {candidate.id} is {candidate.status} and cannot be abandoned"
        )
    if candidate.status in {"rejected", "superseded"}:
        if candidate.status != disposition:
            raise RuntimeError(
                f"Candidate {candidate.id} is already terminal as {candidate.status}"
            )
        return candidate
    now = datetime.utcnow().isoformat()
    audit = {
        "protocol_version": "vector_shadow_candidate_abandonment_v1",
        "disposition": disposition,
        "reason": str(reason or "unspecified"),
        "recorded_at": now,
        "derived_facts_retained": True,
    }
    builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(VectorShadowBuild.runtime_settings_candidate_id == candidate.id)
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
            .with_for_update()
        ).all()
    )
    for build in builds:
        if build.status == "promoted":
            raise RuntimeError(f"Promoted build {build.id} cannot be abandoned")
        if build.status not in {"rejected", "failed", "rolled_back", "superseded"}:
            build.status = disposition
        build.diagnostics_json = {
            **dict(build.diagnostics_json or {}),
            "abandonment": audit,
        }
    candidate.status = disposition
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "abandonment": audit,
    }
    candidate.error_code = f"vector_shadow_candidate_{disposition}"
    candidate.last_error = str(reason or "unspecified")
    db.flush()
    return candidate


def _build_schema(build: VectorShadowBuild) -> FrozenVectorSchema:
    schema = FrozenVectorSchema.model_validate(build.candidate_vector_schema_json)
    expected_hash = vector_schema_hash(schema)
    if expected_hash != build.candidate_vector_schema_hash:
        raise RuntimeError(f"Vector shadow build {build.id} has a tampered schema hash")
    duplicate_fields = {
        "embedding_model": build.embedding_model,
        "embedding_dimension": build.embedding_dimension,
        "distance_metric": build.distance_metric,
        "embedding_text_version": build.embedding_text_version,
        "chunk_schema_version": build.chunk_schema_version,
        "collection_identity_protocol_version": build.collection_identity_protocol_version,
        "collection_identity_digest": build.collection_identity_digest,
        "collection_name": build.collection_name,
    }
    for field_name, stored in duplicate_fields.items():
        if getattr(schema, field_name) != stored:
            raise RuntimeError(
                f"Vector shadow build {build.id} has a mismatched {field_name}"
            )
    return schema


def _vector_record_proof(
    db: Session,
    *,
    build: VectorShadowBuild,
    candidate: RuntimeSettingsCandidate,
    schema: FrozenVectorSchema,
    chunk_ids: Sequence[str],
) -> tuple[str, str]:
    records: list[VectorRecord] = []
    for offset in range(0, len(chunk_ids), ACTIVE_CHUNK_READ_PAGE_SIZE):
        chunk_id_page = list(chunk_ids[offset : offset + ACTIVE_CHUNK_READ_PAGE_SIZE])
        records.extend(
            db.scalars(
                select(VectorRecord)
                .where(
                    VectorRecord.knowledge_base_id == build.knowledge_base_id,
                    VectorRecord.chunk_id.in_(chunk_id_page),
                    VectorRecord.embedding_model == schema.embedding_model,
                    VectorRecord.embedding_dimension == schema.embedding_dimension,
                    VectorRecord.embedding_text_version == schema.embedding_text_version,
                    VectorRecord.chunk_schema_version == schema.chunk_schema_version,
                    VectorRecord.collection_name == schema.collection_name,
                )
                .order_by(VectorRecord.chunk_id.asc())
            ).all()
        )
    if len(records) != len(chunk_ids):
        raise RuntimeError(
            f"Vector shadow build {build.id} has {len(records)}/{len(chunk_ids)} vector records"
        )
    if [record.chunk_id for record in records] != list(chunk_ids):
        raise RuntimeError(f"Vector shadow build {build.id} vector record scope is not exact")
    point_ids = [record.qdrant_point_id for record in records]
    if len(set(point_ids)) != len(point_ids):
        raise RuntimeError(f"Vector shadow build {build.id} reuses Qdrant point ids")

    record_payload: list[dict[str, Any]] = []
    point_payload: list[dict[str, Any]] = []
    expected_markers = {
        "runtime_settings_candidate_id": candidate.id,
        "vector_shadow_build_id": build.id,
        "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
        "vector_shadow_writer_protocol_version": VECTOR_SHADOW_WRITER_PROTOCOL_VERSION,
        "collection_identity_protocol_version": schema.collection_identity_protocol_version,
        "collection_identity_digest": schema.collection_identity_digest,
        "chunk_schema_version": schema.chunk_schema_version,
    }
    for record in records:
        if record.vector_status != "shadow_ready":
            raise RuntimeError(
                f"Vector record {record.id} is {record.vector_status}, expected shadow_ready"
            )
        if record.chunk_schema_version != schema.chunk_schema_version:
            raise RuntimeError(
                f"Vector record {record.id} has a mismatched direct chunk schema identity"
            )
        diagnostics = dict(record.diagnostics_json or {})
        for key, expected in expected_markers.items():
            if diagnostics.get(key) != expected:
                raise RuntimeError(f"Vector record {record.id} is missing exact {key} binding")
        item = {
            "id": record.id,
            "chunk_id": record.chunk_id,
            "qdrant_point_id": record.qdrant_point_id,
            "payload_hash": record.payload_hash,
        }
        record_payload.append(item)
        point_payload.append(
            {
                "chunk_id": record.chunk_id,
                "qdrant_point_id": record.qdrant_point_id,
                "payload_hash": record.payload_hash,
            }
        )
    return _stable_hash(record_payload), _stable_hash(point_payload)


def _required_graph_state(db: Session, model: type[Any], state_id: str) -> Any:
    state = db.get(model, state_id)
    if state is None:
        raise RuntimeError(f"Missing shadow graph state {model.__name__}:{state_id}")
    return state


def _assert_shadow_graph_proof(
    db: Session,
    *,
    build: VectorShadowBuild,
    candidate: RuntimeSettingsCandidate,
    schema: FrozenVectorSchema,
    chunk_ids: Sequence[str],
    attestation: VectorShadowBuildAttestation,
) -> ContextGraphState:
    relation = _required_graph_state(
        db,
        ChunkRelationGraphState,
        attestation.shadow_chunk_relation_graph_state_id,
    )
    mid = _required_graph_state(db, MidConceptState, attestation.shadow_mid_concept_state_id)
    coarse = _required_graph_state(
        db,
        CoarseConceptState,
        attestation.shadow_coarse_concept_state_id,
    )
    context = _required_graph_state(
        db,
        ContextGraphState,
        attestation.shadow_context_graph_state_id,
    )
    states = (relation, mid, coarse, context)
    if any(state.knowledge_base_id != build.knowledge_base_id for state in states):
        raise RuntimeError("Shadow graph state knowledge-base provenance mismatch")
    if any(state.state != "shadow" for state in states):
        raise RuntimeError("Every candidate graph state must remain in shadow state")
    if relation.embedding_text_version != schema.embedding_text_version:
        raise RuntimeError("Shadow relation graph uses the wrong embedding text version")
    effective_runtime_hash = _candidate_effective_runtime_hash(candidate)
    if relation.runtime_settings_hash != effective_runtime_hash:
        raise RuntimeError("Shadow relation graph is not bound to the candidate hash")
    if context.runtime_settings_hash != effective_runtime_hash:
        raise RuntimeError("Shadow context graph is not bound to the candidate hash")
    if relation.scope_hash != attestation.chunk_scope_hash:
        raise RuntimeError("Shadow relation graph scope hash mismatch")
    if context.chunk_scope_hash != attestation.chunk_scope_hash:
        raise RuntimeError("Shadow context graph scope hash mismatch")
    if sorted(relation.active_chunk_ids_json or []) != list(chunk_ids):
        raise RuntimeError("Shadow relation graph active chunk set mismatch")
    if mid.chunk_relation_graph_state_id != relation.id:
        raise RuntimeError("Shadow mid graph is not linked to the relation graph")
    if coarse.mid_concept_state_id != mid.id:
        raise RuntimeError("Shadow coarse graph is not linked to the mid graph")
    if (
        context.chunk_relation_graph_state_id != relation.id
        or context.mid_concept_state_id != mid.id
        or context.coarse_concept_state_id != coarse.id
    ):
        raise RuntimeError("Shadow context graph layer linkage mismatch")

    expected_markers = {
        "runtime_settings_candidate_id": candidate.id,
        "vector_shadow_build_id": build.id,
        "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
        "vector_shadow_writer_protocol_version": VECTOR_SHADOW_WRITER_PROTOCOL_VERSION,
        "vector_runtime_consumer_protocol_version": (
            VECTOR_SHADOW_GRAPH_CONSUMER_PROTOCOL_VERSION
        ),
        "embedding_model": schema.embedding_model,
        "embedding_dimension": schema.embedding_dimension,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "collection_name": schema.collection_name,
        "collection_identity_digest": schema.collection_identity_digest,
    }
    for state in states:
        diagnostics = dict(state.diagnostics_json or {})
        for key, expected in expected_markers.items():
            if diagnostics.get(key) != expected:
                raise RuntimeError(
                    f"Shadow graph state {state.id} is missing exact {key} binding"
                )
    from app.services.context_graph import (
        rq_prefix_pair_integrity_proof,
        rq_prefix_pair_persisted_integrity,
    )

    pair_integrity = rq_prefix_pair_persisted_integrity(db, relation)
    if not pair_integrity["valid"]:
        raise RuntimeError(
            "Shadow RQ prefix-pair facts failed full integrity verification: "
            + ", ".join(pair_integrity["reasons"][:8])
        )
    relation_diagnostics = dict(relation.diagnostics_json or {})
    pair_diagnostics = dict(
        relation_diagnostics.get("rq_prefix_pair_diagnostics") or {}
    )
    if pair_integrity["aggregate_hash"] != pair_diagnostics.get("diagnostic_hash"):
        raise RuntimeError("Shadow RQ prefix-pair aggregate hash drifted")
    expected_pair_proof = rq_prefix_pair_integrity_proof(
        relation,
        diagnostic_count=pair_integrity["diagnostic_count"],
        aggregate_hash=pair_integrity["aggregate_hash"],
    )
    if (
        relation_diagnostics.get("rq_prefix_pair_persisted_integrity")
        != expected_pair_proof
    ):
        raise RuntimeError("Shadow RQ prefix-pair durable proof drifted")
    return context


def _assert_qdrant_proof(
    *,
    build: VectorShadowBuild,
    schema: FrozenVectorSchema,
    chunk_ids: Sequence[str],
    expected_point_set_hash: str,
    proof: QdrantShadowScopeProof,
) -> None:
    expected = {
        "collection_name": schema.collection_name,
        "collection_identity_protocol_version": schema.collection_identity_protocol_version,
        "collection_identity_digest": schema.collection_identity_digest,
        "embedding_model": schema.embedding_model,
        "embedding_dimension": schema.embedding_dimension,
        "distance_metric": schema.distance_metric,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "scope_filter_hash": _scope_filter_hash(build.knowledge_base_id, chunk_ids),
        "scoped_point_count": len(chunk_ids),
        "point_set_hash": expected_point_set_hash,
    }
    for field_name, expected_value in expected.items():
        if getattr(proof, field_name) != expected_value:
            raise RuntimeError(f"Qdrant shadow proof mismatch: {field_name}")


def _assert_shadow_build_facts(
    db: Session,
    *,
    build: VectorShadowBuild,
    candidate: RuntimeSettingsCandidate,
    attestation: VectorShadowBuildAttestation,
) -> tuple[str, ContextGraphState]:
    schema = _build_schema(build)
    chunk_ids = _assert_staged_scope_unchanged(db, build)
    record_set_hash, point_set_hash = _vector_record_proof(
        db,
        build=build,
        candidate=candidate,
        schema=schema,
        chunk_ids=chunk_ids,
    )
    _assert_qdrant_proof(
        build=build,
        schema=schema,
        chunk_ids=chunk_ids,
        expected_point_set_hash=point_set_hash,
        proof=attestation.qdrant,
    )
    context = _assert_shadow_graph_proof(
        db,
        build=build,
        candidate=candidate,
        schema=schema,
        chunk_ids=chunk_ids,
        attestation=attestation,
    )
    return record_set_hash, context


def _refresh_candidate_progress(db: Session, candidate: RuntimeSettingsCandidate) -> None:
    db.flush()
    statuses = set(
        db.scalars(
            select(VectorShadowBuild.status).where(
                VectorShadowBuild.runtime_settings_candidate_id == candidate.id
            )
        ).all()
    )
    if statuses and statuses <= {"evaluation_passed"}:
        candidate.status = "evaluation_passed"
    elif "promotion_blocked" in statuses:
        candidate.status = "promotion_blocked"
    elif statuses & {"shadow_ready", "evaluating", "evaluation_passed"}:
        candidate.status = "evaluating"
    elif "building" in statuses:
        candidate.status = "building"
    else:
        candidate.status = "staged"


def bind_vector_shadow_build_attestation(
    db: Session,
    *,
    build_id: str,
    attestation: VectorShadowBuildAttestation | dict[str, Any],
) -> VectorShadowBuild:
    build, candidate = _locked_build_candidate(db, build_id)
    if build.status not in {"building", "shadow_ready"}:
        raise RuntimeError(
            f"Cannot attest vector shadow build {build.id} from status {build.status}"
        )
    parsed = VectorShadowBuildAttestation.model_validate(attestation)
    record_set_hash, context = _assert_shadow_build_facts(
        db,
        build=build,
        candidate=candidate,
        attestation=parsed,
    )
    proof_json = parsed.qdrant.model_dump(mode="json")
    proof_hash = _stable_hash(proof_json)
    if build.status == "shadow_ready":
        if (
            build.vector_record_set_hash != record_set_hash
            or build.qdrant_proof_hash != proof_hash
            or build.shadow_context_graph_state_id
            != parsed.shadow_context_graph_state_id
        ):
            raise RuntimeError("A ready vector shadow build cannot be rebound to different facts")
        return build
    build.shadow_context_graph_state_id = parsed.shadow_context_graph_state_id
    build.shadow_chunk_relation_graph_state_id = (
        parsed.shadow_chunk_relation_graph_state_id
    )
    build.shadow_mid_concept_state_id = parsed.shadow_mid_concept_state_id
    build.shadow_coarse_concept_state_id = parsed.shadow_coarse_concept_state_id
    build.chunk_scope_hash = parsed.chunk_scope_hash
    build.ready_point_count = build.expected_point_count
    build.vector_record_set_hash = record_set_hash
    build.qdrant_proof_json = proof_json
    build.qdrant_proof_hash = proof_hash
    build.status = "shadow_ready"
    build.shadow_ready_at = datetime.utcnow()
    build.diagnostics_json = {
        **dict(build.diagnostics_json or {}),
        "context_graph_hash": context.context_graph_hash,
        "shadow_attestation_hash": _stable_hash(parsed.model_dump(mode="json")),
        "active_pointer_mutated": False,
    }
    _refresh_candidate_progress(db, candidate)
    db.flush()
    return build


def _qdrant_shadow_scope_proof_from_context_state(
    db: Session,
    *,
    build: VectorShadowBuild,
    candidate: RuntimeSettingsCandidate,
    target: VectorRuntimeTarget,
    context_state: ContextGraphState,
) -> QdrantShadowScopeProof:
    """Freeze the bounded real-Qdrant observation emitted by the writer preflight.

    The context graph builder verifies only the exact PostgreSQL-owned point ids;
    it never performs an unbounded collection scan.  This adapter binds that
    observation to the immutable shadow build facts used by promotion.
    """

    if target.state_scope != "shadow" or target.vector_shadow_build_id != build.id:
        raise RuntimeError("Qdrant shadow proof target/build provenance mismatch")
    if target.runtime_settings_candidate_id != candidate.id:
        raise RuntimeError("Qdrant shadow proof target/candidate provenance mismatch")
    chunk_ids = _assert_staged_scope_unchanged(db, build)
    schema = _build_schema(build)
    _record_set_hash, point_set_hash = _vector_record_proof(
        db,
        build=build,
        candidate=candidate,
        schema=schema,
        chunk_ids=chunk_ids,
    )
    diagnostics = dict(context_state.diagnostics_json or {})
    maintenance = diagnostics.get("contextual_index_maintenance")
    observation = (
        maintenance.get("qdrant_freshness")
        if isinstance(maintenance, dict)
        else None
    )
    if not isinstance(observation, dict):
        raise RuntimeError("Shadow context graph lacks a bounded Qdrant observation")
    expected_values = {
        "verified": True,
        "collection_name": schema.collection_name,
        "collection_identity_protocol_version": (
            schema.collection_identity_protocol_version
        ),
        "collection_identity_digest": schema.collection_identity_digest,
        "vector_runtime_state_scope": "shadow",
        "vector_runtime_state_hash": target.runtime_state_hash,
        "vector_schema_hash": target.vector_schema_hash,
        "embedding_model": schema.embedding_model,
        "embedding_dimension": schema.embedding_dimension,
        "vector_distance_metric": schema.distance_metric,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "contextual_index_hash": diagnostics.get("contextual_index_hash"),
        "expected_point_count": len(chunk_ids),
        "postgres_vector_record_count": len(chunk_ids),
        "observed_point_count": len(chunk_ids),
        "verified_point_count": len(chunk_ids),
        "mismatch_count": 0,
        "orphan_scan_performed": False,
    }
    mismatches = sorted(
        field_name
        for field_name, expected in expected_values.items()
        if observation.get(field_name) != expected
    )
    if mismatches:
        raise RuntimeError(
            "Shadow Qdrant observation does not bind the exact candidate scope: "
            + ", ".join(mismatches)
        )
    verification_hash = _require_hash64(
        observation.get("verification_hash"),
        field_name="qdrant_observation.verification_hash",
    )
    expected_scope_hash = _require_hash64(
        observation.get("expected_scope_hash"),
        field_name="qdrant_observation.expected_scope_hash",
    )
    observer_input = {
        "protocol_version": QDRANT_SHADOW_OBSERVER_PROTOCOL_VERSION,
        "build_id": build.id,
        "runtime_settings_candidate_id": candidate.id,
        "runtime_settings_candidate_hash": candidate.candidate_hash,
        "knowledge_base_id": build.knowledge_base_id,
        "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
        "chunk_scope_hash": compute_chunk_scope_hash(
            _shadow_build_scope_chunks(db, build)
        ),
        "contextual_index_hash": diagnostics.get("contextual_index_hash"),
        "point_set_hash": point_set_hash,
        "qdrant_expected_scope_hash": expected_scope_hash,
        "expected_point_count": len(chunk_ids),
    }
    observer_input_hash = _stable_hash(observer_input)
    observer_output_hash = _stable_hash(
        {
            "protocol_version": QDRANT_SHADOW_OBSERVER_PROTOCOL_VERSION,
            "observer_input_hash": observer_input_hash,
            "verified": True,
            "collection_exists": observation.get("collection_exists"),
            "collection_schema_error": observation.get("collection_schema_error"),
            "verification_hash": verification_hash,
            "observed_point_count": observation.get("observed_point_count"),
            "verified_point_count": observation.get("verified_point_count"),
            "mismatch_count": observation.get("mismatch_count"),
            "retrieve_batch_size": observation.get("retrieve_batch_size"),
            "retrieve_batch_count": observation.get("retrieve_batch_count"),
            "orphan_scan_performed": observation.get("orphan_scan_performed"),
        }
    )
    return QdrantShadowScopeProof(
        verified=True,
        collection_name=schema.collection_name,
        collection_identity_protocol_version=(
            schema.collection_identity_protocol_version
        ),
        collection_identity_digest=schema.collection_identity_digest,
        embedding_model=schema.embedding_model,
        embedding_dimension=schema.embedding_dimension,
        distance_metric=schema.distance_metric,
        embedding_text_version=schema.embedding_text_version,
        chunk_schema_version=schema.chunk_schema_version,
        scope_filter_hash=_scope_filter_hash(build.knowledge_base_id, chunk_ids),
        scoped_point_count=len(chunk_ids),
        point_set_hash=point_set_hash,
        observer_input_hash=observer_input_hash,
        observer_output_hash=observer_output_hash,
    )


async def build_vector_shadow_artifacts(
    db: Session,
    *,
    build_id: str,
    batch_id: str | None = None,
    operating_point: dict[str, Any] | None = None,
    emit_heartbeats: bool = True,
    concept_provider_request_budget: int = (
        DEFAULT_VECTOR_SHADOW_CONCEPT_PROVIDER_REQUEST_BUDGET
    ),
) -> VectorShadowBuild:
    """Build and attest one exact candidate collection and four-layer graph.

    PostgreSQL is the lifecycle fact source.  Qdrant writes use the durable
    outbox owned by ``write_contextual_indexes``; an exception leaves promotion
    impossible and lets the caller roll back the transaction/trigger outbox
    compensation.  No active pointer, process environment or Redis key is
    changed here.
    """

    build = start_vector_shadow_build(db, build_id)
    target = resolve_shadow_vector_runtime_target(db, build.id, for_update=True)
    build_chunks = _shadow_build_scope_chunks(db, build)
    (
        semantic_reuse_mid_state,
        semantic_reuse_coarse_state,
        semantic_reuse_audit,
    ) = _active_concept_semantic_reuse_sources(db, build)
    (
        compensated_embedding_recovery_cards,
        compensated_embedding_recovery_audit,
    ) = _compensated_shadow_embedding_recovery(
        db,
        build=build,
        chunks=build_chunks,
    )
    from app.services.context_graph import (
        ConceptProviderRequestBudget,
        rebuild_context_graph,
    )

    shared_provider_request_budget = ConceptProviderRequestBudget(
        max_requests=concept_provider_request_budget
    )

    context_state = await rebuild_context_graph(
        db,
        build.knowledge_base_id,
        batch_id=batch_id,
        state_scope="shadow",
        operating_point=operating_point,
        vector_runtime_target=target,
        emit_heartbeats=emit_heartbeats,
        chunks_override=build_chunks,
        shadow_metadata={
            "vector_shadow_concept_semantic_reuse": semantic_reuse_audit,
            "vector_shadow_compensated_embedding_recovery": (
                compensated_embedding_recovery_audit
            ),
        },
        provider_semantic_reuse_source_mid_state=semantic_reuse_mid_state,
        provider_semantic_reuse_source_coarse_state=semantic_reuse_coarse_state,
        provider_semantic_reuse_source_scope=(
            "shadow"
            if semantic_reuse_audit.get("source_kind") == "terminal_shadow"
            else "active"
        ),
        require_provider_semantic_reuse=bool(
            semantic_reuse_audit.get("exact_reuse_required")
        ),
        concept_provider_request_budget=shared_provider_request_budget,
        compensated_embedding_recovery_cards=(
            compensated_embedding_recovery_cards
        ),
        compensated_embedding_recovery_audit=(
            compensated_embedding_recovery_audit
        ),
    )
    locked_build, candidate = _locked_build_candidate(db, build.id)
    proof = _qdrant_shadow_scope_proof_from_context_state(
        db,
        build=locked_build,
        candidate=candidate,
        target=target,
        context_state=context_state,
    )
    attestation = VectorShadowBuildAttestation(
        writer_protocol_version=VECTOR_SHADOW_WRITER_PROTOCOL_VERSION,
        graph_consumer_protocol_version=VECTOR_SHADOW_GRAPH_CONSUMER_PROTOCOL_VERSION,
        chunk_scope_hash=context_state.chunk_scope_hash,
        shadow_context_graph_state_id=context_state.id,
        shadow_chunk_relation_graph_state_id=(
            context_state.chunk_relation_graph_state_id
        ),
        shadow_mid_concept_state_id=context_state.mid_concept_state_id,
        shadow_coarse_concept_state_id=context_state.coarse_concept_state_id,
        qdrant=proof,
    )
    ready_build = bind_vector_shadow_build_attestation(
        db,
        build_id=locked_build.id,
        attestation=attestation,
    )
    shadow_mid_state = db.get(MidConceptState, context_state.mid_concept_state_id)
    shadow_coarse_state = db.get(
        CoarseConceptState,
        context_state.coarse_concept_state_id,
    )
    mid_stats = dict(shadow_mid_state.stats_json or {}) if shadow_mid_state else {}
    coarse_stats = (
        dict(shadow_coarse_state.stats_json or {}) if shadow_coarse_state else {}
    )
    semantic_reuse_result = {
        **semantic_reuse_audit,
        "mid_hit_count": int(
            mid_stats.get("provider_semantic_reuse_hit_count") or 0
        ),
        "mid_miss_count": int(
            mid_stats.get("provider_semantic_reuse_miss_count") or 0
        ),
        "mid_provider_request_count": int(
            mid_stats.get("provider_request_count") or 0
        ),
        "coarse_hit_count": int(
            coarse_stats.get("provider_semantic_reuse_hit_count") or 0
        ),
        "coarse_miss_count": int(
            coarse_stats.get("provider_semantic_reuse_miss_count") or 0
        ),
        "coarse_provider_request_count": int(
            coarse_stats.get("provider_request_count") or 0
        ),
        "provider_request_budget": shared_provider_request_budget.diagnostics(),
    }
    semantic_reuse_result["provider_request_count"] = (
        semantic_reuse_result["mid_provider_request_count"]
        + semantic_reuse_result["coarse_provider_request_count"]
    )
    ready_build.diagnostics_json = {
        **dict(ready_build.diagnostics_json or {}),
        "build_orchestration_protocol_version": (
            "vector_shadow_artifact_orchestration_v1"
        ),
        "qdrant_observer_protocol_version": (
            QDRANT_SHADOW_OBSERVER_PROTOCOL_VERSION
        ),
        "active_pointer_mutated": False,
        "vector_shadow_concept_semantic_reuse": semantic_reuse_result,
        "vector_shadow_compensated_embedding_recovery": (
            compensated_embedding_recovery_audit
        ),
        "candidate_chunk_scope_consumed": bool(
            (ready_build.diagnostics_json or {}).get("candidate_chunk_ids")
        ),
    }
    db.flush()
    return ready_build


def _stored_attestation(build: VectorShadowBuild) -> VectorShadowBuildAttestation:
    required_ids = {
        "shadow_context_graph_state_id": build.shadow_context_graph_state_id,
        "shadow_chunk_relation_graph_state_id": build.shadow_chunk_relation_graph_state_id,
        "shadow_mid_concept_state_id": build.shadow_mid_concept_state_id,
        "shadow_coarse_concept_state_id": build.shadow_coarse_concept_state_id,
    }
    missing = sorted(key for key, value in required_ids.items() if not value)
    if missing or not build.chunk_scope_hash or not build.qdrant_proof_json:
        raise RuntimeError(
            f"Vector shadow build {build.id} has incomplete stored proof: {', '.join(missing)}"
        )
    return VectorShadowBuildAttestation(
        writer_protocol_version=VECTOR_SHADOW_WRITER_PROTOCOL_VERSION,
        graph_consumer_protocol_version=VECTOR_SHADOW_GRAPH_CONSUMER_PROTOCOL_VERSION,
        chunk_scope_hash=build.chunk_scope_hash,
        shadow_context_graph_state_id=build.shadow_context_graph_state_id,
        shadow_chunk_relation_graph_state_id=build.shadow_chunk_relation_graph_state_id,
        shadow_mid_concept_state_id=build.shadow_mid_concept_state_id,
        shadow_coarse_concept_state_id=build.shadow_coarse_concept_state_id,
        qdrant=QdrantShadowScopeProof.model_validate(build.qdrant_proof_json),
    )


def vector_shadow_evaluation_input_hash(
    build: VectorShadowBuild,
    context_state: ContextGraphState,
) -> str:
    return _stable_hash(
        {
            "protocol_version": VECTOR_SHADOW_EVALUATION_PROTOCOL_VERSION,
            "build_id": build.id,
            "runtime_settings_candidate_id": build.runtime_settings_candidate_id,
            "knowledge_base_id": build.knowledge_base_id,
            "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
            "chunk_scope_hash": build.chunk_scope_hash,
            "vector_record_set_hash": build.vector_record_set_hash,
            "qdrant_proof_hash": build.qdrant_proof_hash,
            "context_graph_state_id": context_state.id,
            "context_graph_hash": context_state.context_graph_hash,
        }
    )


def record_vector_shadow_evaluation(
    db: Session,
    *,
    build_id: str,
    evaluation: VectorShadowEvaluation | dict[str, Any],
) -> VectorShadowBuild:
    build, candidate = _locked_build_candidate(db, build_id)
    if build.status not in {
        "shadow_ready",
        "evaluating",
        "evaluation_passed",
        "promotion_blocked",
    }:
        raise RuntimeError(
            f"Cannot evaluate vector shadow build {build.id} from status {build.status}"
        )
    parsed = VectorShadowEvaluation.model_validate(evaluation)
    attestation = _stored_attestation(build)
    record_set_hash, context = _assert_shadow_build_facts(
        db,
        build=build,
        candidate=candidate,
        attestation=attestation,
    )
    if record_set_hash != build.vector_record_set_hash:
        raise RuntimeError("Vector record facts changed after shadow attestation")
    expected_input_hash = vector_shadow_evaluation_input_hash(build, context)
    if parsed.evaluation_input_hash != expected_input_hash:
        raise RuntimeError("Shadow evaluation input hash does not bind the current artifacts")
    result_json = parsed.model_dump(mode="json")
    result_hash = _stable_hash(result_json)
    failed_gates = sorted(
        key for key, passed in parsed.hard_gates.items() if passed is not True
    )
    target_status = "promotion_blocked" if failed_gates else "evaluation_passed"
    if build.status in {"evaluation_passed", "promotion_blocked"}:
        if build.evaluation_result_hash != result_hash:
            raise RuntimeError("A recorded shadow evaluation cannot be replaced")
        return build
    build.status = target_status
    build.evaluation_protocol_version = parsed.protocol_version
    build.evaluation_input_hash = parsed.evaluation_input_hash
    build.evaluation_result_json = result_json
    build.evaluation_result_hash = result_hash
    build.evaluated_at = datetime.utcnow()
    build.blocking_reasons_json = [
        f"evaluation_gate_failed:{gate}" for gate in failed_gates
    ]
    _refresh_candidate_progress(db, candidate)
    db.flush()
    return build


def _records_for_schema(
    db: Session,
    *,
    knowledge_base_id: str,
    schema: FrozenVectorSchema,
    chunk_ids: Sequence[str],
    for_update: bool = False,
) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for offset in range(0, len(chunk_ids), ACTIVE_CHUNK_READ_PAGE_SIZE):
        page = list(chunk_ids[offset : offset + ACTIVE_CHUNK_READ_PAGE_SIZE])
        statement = (
            select(VectorRecord)
            .where(
                VectorRecord.knowledge_base_id == knowledge_base_id,
                VectorRecord.chunk_id.in_(page),
                VectorRecord.embedding_model == schema.embedding_model,
                VectorRecord.embedding_dimension == schema.embedding_dimension,
                VectorRecord.embedding_text_version == schema.embedding_text_version,
                VectorRecord.chunk_schema_version == schema.chunk_schema_version,
                VectorRecord.collection_name == schema.collection_name,
            )
            .order_by(VectorRecord.chunk_id.asc())
        )
        if for_update:
            statement = statement.with_for_update()
        records.extend(db.scalars(statement).all())
    return records


def _assert_exact_record_scope(
    records: Sequence[VectorRecord],
    *,
    chunk_ids: Sequence[str],
    expected_status: str,
    scope_name: str,
) -> None:
    if [str(record.chunk_id) for record in records] != list(chunk_ids):
        raise RuntimeError(
            f"{scope_name} VectorRecord scope is not an exact active chunk set"
        )
    invalid = [
        str(record.id)
        for record in records
        if record.vector_status != expected_status
    ]
    if invalid:
        raise RuntimeError(
            f"{scope_name} VectorRecords are not {expected_status}: {invalid[:8]}"
        )


def _locked_graph_state(
    db: Session,
    model: type[Any],
    state_id: str | None,
    *,
    knowledge_base_id: str,
    expected_state: str,
) -> Any | None:
    if state_id is None:
        return None
    row = db.scalar(
        select(model).where(model.id == state_id).with_for_update()
    )
    if row is None:
        raise RuntimeError(f"Missing graph state: {model.__name__}:{state_id}")
    if str(row.knowledge_base_id) != str(knowledge_base_id):
        raise RuntimeError(
            f"Graph state provenance mismatch: {model.__name__}:{state_id}"
        )
    if row.state != expected_state:
        raise RuntimeError(
            f"Graph state {model.__name__}:{state_id} is {row.state}, expected {expected_state}"
        )
    return row


def _locked_graph_bundle(
    db: Session,
    *,
    knowledge_base_id: str,
    graph_state_ids: dict[str, str | None],
    expected_state: str,
) -> dict[str, Any | None]:
    if set(graph_state_ids) != {"context", "relation", "mid", "coarse"}:
        raise RuntimeError("Graph bundle does not contain the exact four-layer key set")
    return {
        "context": _locked_graph_state(
            db,
            ContextGraphState,
            graph_state_ids["context"],
            knowledge_base_id=knowledge_base_id,
            expected_state=expected_state,
        ),
        "relation": _locked_graph_state(
            db,
            ChunkRelationGraphState,
            graph_state_ids["relation"],
            knowledge_base_id=knowledge_base_id,
            expected_state=expected_state,
        ),
        "mid": _locked_graph_state(
            db,
            MidConceptState,
            graph_state_ids["mid"],
            knowledge_base_id=knowledge_base_id,
            expected_state=expected_state,
        ),
        "coarse": _locked_graph_state(
            db,
            CoarseConceptState,
            graph_state_ids["coarse"],
            knowledge_base_id=knowledge_base_id,
            expected_state=expected_state,
        ),
    }


def _set_graph_bundle_state(
    db: Session,
    bundle: dict[str, Any | None],
    state: str,
) -> None:
    """Switch a four-layer bundle and its stateful public child rows exactly."""

    for row in bundle.values():
        if row is not None:
            row.state = state
    relation_state = bundle.get("relation")
    if relation_state is not None:
        db.execute(
            update(RQPrefix)
            .where(RQPrefix.graph_state_id == relation_state.id)
            .values(state=state)
        )
    mid_state = bundle.get("mid")
    if mid_state is not None:
        db.execute(
            update(MidConcept)
            .where(MidConcept.concept_state_id == mid_state.id)
            .values(state=state)
        )
    coarse_state = bundle.get("coarse")
    if coarse_state is not None:
        db.execute(
            update(CoarseConcept)
            .where(CoarseConcept.coarse_state_id == coarse_state.id)
            .values(state=state)
        )


def _candidate_graph_state_ids(build: VectorShadowBuild) -> dict[str, str | None]:
    return {
        "context": build.shadow_context_graph_state_id,
        "relation": build.shadow_chunk_relation_graph_state_id,
        "mid": build.shadow_mid_concept_state_id,
        "coarse": build.shadow_coarse_concept_state_id,
    }


def _candidate_chunk_promotion_scope(
    db: Session,
    build: VectorShadowBuild,
) -> dict[str, Any] | None:
    diagnostics = dict(build.diagnostics_json or {})
    candidate_chunk_ids = sorted(
        str(value) for value in (diagnostics.get("candidate_chunk_ids") or [])
    )
    if not candidate_chunk_ids:
        return None
    candidate_document_version_ids = sorted(
        str(value)
        for value in (diagnostics.get("candidate_document_version_ids") or [])
    )
    if not candidate_document_version_ids:
        raise RuntimeError("Candidate rechunk scope lacks document-version provenance")
    candidate_chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(candidate_chunk_ids))
            .order_by(Chunk.id.asc())
            .with_for_update()
        ).all()
    )
    if (
        [str(chunk.id) for chunk in candidate_chunks] != candidate_chunk_ids
        or any(chunk.state != "shadow" for chunk in candidate_chunks)
    ):
        raise RuntimeError("Candidate rechunk rows are not an exact shadow scope")
    candidate_versions = list(
        db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.id.in_(candidate_document_version_ids))
            .order_by(DocumentVersion.id.asc())
            .with_for_update()
        ).all()
    )
    if (
        [str(version.id) for version in candidate_versions]
        != candidate_document_version_ids
        or any(version.is_active for version in candidate_versions)
    ):
        raise RuntimeError("Candidate rechunk document versions are not exact inactive rows")
    active_chunks = _active_chunks(db, build.knowledge_base_id)
    active_version_ids = sorted(
        {str(chunk.document_version_id) for chunk in active_chunks}
    )
    active_versions = list(
        db.scalars(
            select(DocumentVersion)
            .where(
                DocumentVersion.id.in_(active_version_ids),
                DocumentVersion.is_active.is_(True),
            )
            .order_by(DocumentVersion.id.asc())
            .with_for_update()
        ).all()
    )
    if [str(version.id) for version in active_versions] != active_version_ids:
        raise RuntimeError(
            "Candidate rechunk active document-version scope is incomplete"
        )
    knowledge_base = db.scalar(
        select(KnowledgeBase)
        .where(KnowledgeBase.id == build.knowledge_base_id)
        .with_for_update()
    )
    if knowledge_base is None:
        raise RuntimeError("Candidate rechunk knowledge base disappeared")
    candidate_chunk_version = int(
        diagnostics.get("candidate_chunk_version") or 0
    )
    descriptor = db.scalar(
        select(ChunkVersion)
        .where(
            ChunkVersion.knowledge_base_id == build.knowledge_base_id,
            ChunkVersion.chunk_version == candidate_chunk_version,
            ChunkVersion.state == "shadow",
        )
        .with_for_update()
    )
    if descriptor is None:
        raise RuntimeError("Candidate rechunk ChunkVersion shadow descriptor is missing")
    active_descriptors = list(
        db.scalars(
            select(ChunkVersion)
            .where(
                ChunkVersion.knowledge_base_id == build.knowledge_base_id,
                ChunkVersion.state == "active",
            )
            .order_by(ChunkVersion.chunk_version.asc())
            .with_for_update()
        ).all()
    )
    return {
        "active_chunks": active_chunks,
        "active_chunk_ids": sorted(str(chunk.id) for chunk in active_chunks),
        "candidate_chunks": candidate_chunks,
        "candidate_chunk_ids": candidate_chunk_ids,
        "active_versions": active_versions,
        "active_version_ids": sorted(str(version.id) for version in active_versions),
        "candidate_versions": candidate_versions,
        "candidate_document_version_ids": candidate_document_version_ids,
        "knowledge_base": knowledge_base,
        "base_chunk_version": int(knowledge_base.current_chunk_version or 0),
        "candidate_chunk_version": candidate_chunk_version,
        "candidate_chunk_version_descriptor": descriptor,
        "active_chunk_version_descriptors": active_descriptors,
    }


def _promote_candidate_chunk_scope(db: Session, scope: dict[str, Any]) -> None:
    for chunk in scope["active_chunks"]:
        chunk.state = "inactive"
    for version in scope["active_versions"]:
        version.is_active = False
    # DocumentVersion has a partial unique constraint that permits only one
    # active version per document.  SQLAlchemy may reorder UPDATE statements by
    # primary key, so make the hand-off explicitly two phase inside the same
    # transaction instead of relying on unit-of-work ordering.
    db.flush()
    for chunk in scope["candidate_chunks"]:
        chunk.state = "active"
    for version in scope["candidate_versions"]:
        version.is_active = True
    knowledge_base: KnowledgeBase = scope["knowledge_base"]
    knowledge_base.current_chunk_version = int(scope["candidate_chunk_version"])
    for active_descriptor in scope["active_chunk_version_descriptors"]:
        active_descriptor.state = "inactive"
    descriptor: ChunkVersion = scope["candidate_chunk_version_descriptor"]
    descriptor.state = "active"


def _cache_invalidation_intent(
    db: Session,
    *,
    candidate_id: str,
    knowledge_base_id: str,
    action: Literal["promotion", "rollback"],
    pointer: KnowledgeBaseVectorRuntimeState,
) -> IngestionCompensationLog:
    existing = db.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.knowledge_base_id == knowledge_base_id,
            IngestionCompensationLog.operation
            == VECTOR_RUNTIME_CACHE_INVALIDATION_OPERATION,
            IngestionCompensationLog.status == "cache_invalidation_pending",
        )
    )
    if existing is not None:
        payload = dict(existing.payload_json or {})
        if (
            payload.get("candidate_id") != candidate_id
            or payload.get("action") != action
            or payload.get("pointer_state_hash") != pointer.state_hash
        ):
            # Cache invalidation is knowledge-base wide.  A newer pointer switch
            # subsumes the older pending request, so retain the audit row but do
            # not let a Redis outage make rollback impossible.
            superseded_at = datetime.utcnow().isoformat()
            payload.pop("payload_hash", None)
            superseded_payload = {
                **payload,
                "superseded_at": superseded_at,
                "superseded_by_candidate_id": candidate_id,
                "superseded_by_action": action,
                "superseded_by_pointer_state_hash": pointer.state_hash,
            }
            existing.status = "superseded"
            existing.payload_json = {
                **superseded_payload,
                "payload_hash": _stable_hash(superseded_payload),
            }
            existing.error_message = None
        else:
            return existing
    now = datetime.utcnow().isoformat()
    payload = {
        "protocol_version": VECTOR_RUNTIME_CACHE_INVALIDATION_PROTOCOL_VERSION,
        "candidate_id": candidate_id,
        "knowledge_base_id": knowledge_base_id,
        "action": action,
        "phase": "database_committed",
        "pointer_state_hash": pointer.state_hash,
        "activation_generation": int(pointer.activation_generation),
        "attempt_count": 0,
        "last_attempt_at": None,
        "completed_at": None,
        "created_at": now,
    }
    row = IngestionCompensationLog(
        knowledge_base_id=knowledge_base_id,
        operation=VECTOR_RUNTIME_CACHE_INVALIDATION_OPERATION,
        target_ids_json=[pointer.id, candidate_id],
        payload_json={
            **payload,
            "payload_hash": _stable_hash(payload),
        },
        status="cache_invalidation_pending",
    )
    db.add(row)
    db.flush()
    return row


def vector_shadow_promotion_preflight(
    db: Session,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = db.get(RuntimeSettingsCandidate, candidate_id)
    if candidate is None:
        raise ValueError(f"Unknown runtime settings candidate: {candidate_id}")
    builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(VectorShadowBuild.runtime_settings_candidate_id == candidate.id)
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
        ).all()
    )
    blockers: list[str] = []
    if not builds:
        blockers.append("candidate_has_no_shadow_builds")
    for build in builds:
        if build.status != "evaluation_passed":
            blockers.append(
                f"build:{build.id}:status:{build.status}:expected:evaluation_passed"
            )
            continue
        try:
            _assert_shadow_build_facts(
                db,
                build=build,
                candidate=candidate,
                attestation=_stored_attestation(build),
            )
        except Exception as exc:
            blockers.append(f"build:{build.id}:proof_invalid:{type(exc).__name__}")
    missing_consumers = sorted(
        REQUIRED_ACTIVE_VECTOR_RUNTIME_CONSUMERS
        - INTEGRATED_ACTIVE_VECTOR_RUNTIME_CONSUMERS
    )
    blockers.extend(f"active_consumer_not_integrated:{name}" for name in missing_consumers)
    if not ATOMIC_ACTIVE_SWITCH_IMPLEMENTED:
        blockers.append("atomic_active_vector_switch_not_implemented")
    return {
        "protocol_version": "vector_shadow_promotion_preflight_v1",
        "candidate_id": candidate.id,
        "candidate_hash": candidate.candidate_hash,
        "allowed": not blockers,
        "blockers": sorted(set(blockers)),
        "build_ids": [build.id for build in builds],
        "active_pointer_mutated": False,
    }


def _terminal_candidate_builds(
    db: Session,
    candidate: RuntimeSettingsCandidate,
    *,
    expected_status: str,
) -> list[VectorShadowBuild]:
    builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(VectorShadowBuild.runtime_settings_candidate_id == candidate.id)
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
            .limit(MAX_CANDIDATE_KNOWLEDGE_BASES + 1)
            .with_for_update()
        ).all()
    )
    if len(builds) > MAX_CANDIDATE_KNOWLEDGE_BASES:
        raise RuntimeError("Terminal vector candidate exceeds its bounded KB scope")
    expected_knowledge_base_ids = sorted(
        str(value) for value in (candidate.target_knowledge_base_ids_json or [])
    )
    actual_knowledge_base_ids = [str(build.knowledge_base_id) for build in builds]
    if not builds or actual_knowledge_base_ids != expected_knowledge_base_ids:
        raise RuntimeError("Terminal vector candidate build scope does not match its target scope")
    invalid = [
        f"{build.id}:{build.status}"
        for build in builds
        if build.status != expected_status
    ]
    if invalid:
        raise RuntimeError(
            f"Terminal vector candidate has non-{expected_status} builds: "
            + ", ".join(invalid)
        )
    return builds


def _assert_promoted_candidate_replay(
    db: Session,
    candidate: RuntimeSettingsCandidate,
) -> list[KnowledgeBaseVectorRuntimeState]:
    builds = _terminal_candidate_builds(db, candidate, expected_status="promoted")
    from app.services.context_graph import active_graph_admission_gate

    pointers: list[KnowledgeBaseVectorRuntimeState] = []
    for build in builds:
        pointer = db.scalar(
            select(KnowledgeBaseVectorRuntimeState)
            .where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == build.knowledge_base_id
            )
            .with_for_update()
        )
        if pointer is None or pointer.runtime_settings_candidate_id != candidate.id:
            raise RuntimeError(
                f"Promoted candidate {candidate.id} is not active for {build.knowledge_base_id}"
            )
        _validate_pointer(pointer)
        if (
            pointer.vector_schema_hash != build.candidate_vector_schema_hash
            or _graph_state_ids_from_pointer(pointer) != _candidate_graph_state_ids(build)
        ):
            raise RuntimeError("Promoted candidate pointer/schema/graph facts drifted")
        promotion_audit = dict(build.promotion_audit_json or {})
        if promotion_audit.get("candidate_id") != candidate.id:
            raise RuntimeError("Promoted candidate audit no longer identifies the candidate")
        active_graph_admission_gate(db, str(build.knowledge_base_id))
        pointers.append(pointer)
    return pointers


def _assert_rolled_back_candidate_replay(
    db: Session,
    candidate: RuntimeSettingsCandidate,
) -> None:
    builds = _terminal_candidate_builds(db, candidate, expected_status="rolled_back")
    from app.services.context_graph import active_graph_admission_gate

    for build in builds:
        pointer = db.scalar(
            select(KnowledgeBaseVectorRuntimeState)
            .where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == build.knowledge_base_id
            )
            .with_for_update()
        )
        if pointer is None:
            raise RuntimeError("Rolled-back candidate lost its restored active pointer")
        _validate_pointer(pointer)
        rollback_audit = dict(build.rollback_audit_json or {})
        pointer_rollback_audit = dict(
            (pointer.promotion_audit_json or {}).get("rollback") or {}
        )
        if (
            rollback_audit.get("protocol_version")
            != VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION
            or rollback_audit.get("restored_vector_schema_hash")
            != pointer.vector_schema_hash
            or rollback_audit.get("restored_vector_state_hash") != pointer.state_hash
            or pointer_rollback_audit.get("candidate_id") != candidate.id
        ):
            raise RuntimeError("Rolled-back candidate audit no longer matches the restored pointer")
        active_graph_admission_gate(db, str(build.knowledge_base_id))


def promote_vector_shadow_candidate(
    db: Session,
    candidate_id: str,
) -> dict[str, Any]:
    """Atomically switch every target KB to one evaluated vector candidate.

    Qdrant bytes are immutable prebuilt derived facts.  This function only
    changes PostgreSQL graph/vector/pointer facts plus a durable post-commit
    cache-invalidation intent; the caller owns the surrounding commit.
    """

    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == candidate_id)
        .with_for_update()
    )
    if candidate is None:
        raise ValueError(f"Unknown runtime settings candidate: {candidate_id}")
    if candidate.status == "promoted":
        pointers = _assert_promoted_candidate_replay(db, candidate)
        return {
            "protocol_version": "vector_shadow_atomic_promotion_v1",
            "candidate_id": candidate.id,
            "candidate_hash": candidate.candidate_hash,
            "allowed": True,
            "promoted": True,
            "idempotent_replay": True,
            "knowledge_base_ids": sorted(
                str(pointer.knowledge_base_id) for pointer in pointers
            ),
            "active_pointer_mutated": False,
        }
    decision = vector_shadow_promotion_preflight(db, candidate_id)
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "last_promotion_preflight": decision,
    }
    candidate.blocking_reasons_json = list(decision["blockers"])
    if not decision["allowed"]:
        db.flush()
        return decision

    builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(
                VectorShadowBuild.runtime_settings_candidate_id == candidate.id
            )
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
            .with_for_update()
        ).all()
    )
    locked_knowledge_bases = list(
        db.scalars(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.id.in_([build.knowledge_base_id for build in builds])
            )
            .order_by(KnowledgeBase.id.asc())
            .with_for_update()
        ).all()
    )
    if len(locked_knowledge_bases) != len(builds):
        raise RuntimeError("A candidate target knowledge base disappeared before promotion")

    candidate.status = "promoting"
    candidate.error_code = None
    candidate.last_error = None
    promoted_at = datetime.utcnow()
    promotion_rows: list[dict[str, Any]] = []
    for build in builds:
        if build.status != "evaluation_passed":
            raise RuntimeError(
                f"Vector shadow build {build.id} is not evaluation-passed"
            )
        attestation = _stored_attestation(build)
        record_set_hash, _context = _assert_shadow_build_facts(
            db,
            build=build,
            candidate=candidate,
            attestation=attestation,
        )
        if record_set_hash != build.vector_record_set_hash:
            raise RuntimeError("Shadow VectorRecord facts drifted before promotion")
        schema = _build_schema(build)
        chunk_ids = _assert_staged_scope_unchanged(db, build)
        candidate_chunk_scope = _candidate_chunk_promotion_scope(db, build)
        previous_chunk_ids = (
            list(candidate_chunk_scope["active_chunk_ids"])
            if candidate_chunk_scope is not None
            else list(chunk_ids)
        )

        pointer = db.scalar(
            select(KnowledgeBaseVectorRuntimeState)
            .where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == build.knowledge_base_id
            )
            .with_for_update()
        )
        if pointer is None:
            previous_schema, previous_state_hash, _none = _base_vector_state(
                db,
                build.knowledge_base_id,
            )
            previous_graph_ids = _latest_active_graph_state_ids(
                db,
                build.knowledge_base_id,
            )
            previous_generation = 0
            previous_candidate_id = None
        else:
            previous_schema = _validate_pointer(pointer)
            previous_state_hash = pointer.state_hash
            previous_graph_ids = _graph_state_ids_from_pointer(pointer)
            previous_generation = int(pointer.activation_generation)
            previous_candidate_id = pointer.runtime_settings_candidate_id
        if previous_state_hash != build.base_vector_state_hash:
            raise RuntimeError(
                f"Active vector pointer for {build.knowledge_base_id} changed after staging"
            )
        if previous_graph_ids != _latest_active_graph_state_ids(
            db,
            build.knowledge_base_id,
        ):
            raise RuntimeError(
                f"Active graph pointer for {build.knowledge_base_id} is not the latest coherent graph"
            )

        previous_records = _records_for_schema(
            db,
            knowledge_base_id=build.knowledge_base_id,
            schema=previous_schema,
            chunk_ids=previous_chunk_ids,
            for_update=True,
        )
        _assert_exact_record_scope(
            previous_records,
            chunk_ids=previous_chunk_ids,
            expected_status="ready",
            scope_name="pre-promotion active",
        )
        candidate_records = _records_for_schema(
            db,
            knowledge_base_id=build.knowledge_base_id,
            schema=schema,
            chunk_ids=chunk_ids,
            for_update=True,
        )
        _assert_exact_record_scope(
            candidate_records,
            chunk_ids=chunk_ids,
            expected_status="shadow_ready",
            scope_name="candidate shadow",
        )

        previous_bundle = _locked_graph_bundle(
            db,
            knowledge_base_id=build.knowledge_base_id,
            graph_state_ids=previous_graph_ids,
            expected_state="active",
        )
        candidate_graph_ids = _candidate_graph_state_ids(build)
        candidate_bundle = _locked_graph_bundle(
            db,
            knowledge_base_id=build.knowledge_base_id,
            graph_state_ids=candidate_graph_ids,
            expected_state="shadow",
        )
        if candidate_chunk_scope is not None:
            _promote_candidate_chunk_scope(db, candidate_chunk_scope)
        _set_graph_bundle_state(db, previous_bundle, "inactive")
        _set_graph_bundle_state(db, candidate_bundle, "active")
        for record in previous_records:
            record.vector_status = "rollback_retained"
        next_generation = previous_generation + 1
        for record in candidate_records:
            record.vector_status = "ready"
            record.diagnostics_json = {
                **dict(record.diagnostics_json or {}),
                "promoted_activation_generation": next_generation,
                "promoted_at": promoted_at.isoformat(),
            }

        previous_state = {
            "protocol_version": VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION,
            "rollback_eligible": True,
            "runtime_settings_candidate_id": previous_candidate_id,
            "activation_generation": previous_generation,
            "vector_schema": previous_schema.model_dump(mode="json"),
            "vector_schema_hash": vector_schema_hash(previous_schema),
            "state_hash": previous_state_hash,
            "graph_state_ids": previous_graph_ids,
            "vector_record_count": len(previous_records),
            "chunk_scope": (
                {
                    "protocol_version": "runtime_settings_shadow_rechunk_rollback_v1",
                    "base_chunk_ids": previous_chunk_ids,
                    "candidate_chunk_ids": list(chunk_ids),
                    "base_document_version_ids": list(
                        candidate_chunk_scope["active_version_ids"]
                    ),
                    "candidate_document_version_ids": list(
                        candidate_chunk_scope["candidate_document_version_ids"]
                    ),
                    "base_chunk_version": int(
                        candidate_chunk_scope["base_chunk_version"]
                    ),
                    "candidate_chunk_version": int(
                        candidate_chunk_scope["candidate_chunk_version"]
                    ),
                }
                if candidate_chunk_scope is not None
                else None
            ),
            "captured_at": promoted_at.isoformat(),
        }
        if pointer is None:
            pointer = KnowledgeBaseVectorRuntimeState(
                knowledge_base_id=build.knowledge_base_id,
                protocol_version=VECTOR_RUNTIME_STATE_PROTOCOL_VERSION,
            )
            db.add(pointer)
        pointer.runtime_settings_candidate_id = candidate.id
        pointer.embedding_model = schema.embedding_model
        pointer.embedding_dimension = schema.embedding_dimension
        pointer.distance_metric = schema.distance_metric
        pointer.embedding_text_version = schema.embedding_text_version
        pointer.chunk_schema_version = schema.chunk_schema_version
        pointer.collection_identity_protocol_version = (
            schema.collection_identity_protocol_version
        )
        pointer.collection_identity_digest = schema.collection_identity_digest
        pointer.collection_name = schema.collection_name
        pointer.vector_schema_hash = vector_schema_hash(schema)
        pointer.activation_generation = next_generation
        pointer.active_context_graph_state_id = candidate_graph_ids["context"]
        pointer.active_chunk_relation_graph_state_id = candidate_graph_ids["relation"]
        pointer.active_mid_concept_state_id = candidate_graph_ids["mid"]
        pointer.active_coarse_concept_state_id = candidate_graph_ids["coarse"]
        pointer.previous_state_json = previous_state
        pointer.state_hash = vector_runtime_state_hash(
            knowledge_base_id=build.knowledge_base_id,
            runtime_settings_candidate_id=candidate.id,
            activation_generation=next_generation,
            schema=schema,
            graph_state_ids=candidate_graph_ids,
        )
        db.flush()
        for row in candidate_bundle.values():
            if row is not None:
                row.diagnostics_json = {
                    **dict(row.diagnostics_json or {}),
                    "active_vector_runtime_state_id": pointer.id,
                    "active_vector_runtime_state_hash": pointer.state_hash,
                    "active_vector_schema_hash": pointer.vector_schema_hash,
                    "promoted_from_shadow": True,
                }
                if isinstance(row, ContextGraphState):
                    _bind_context_qdrant_proof_to_active_pointer(row, pointer)
        db.flush()
        cache_intent = _cache_invalidation_intent(
            db,
            candidate_id=candidate.id,
            knowledge_base_id=build.knowledge_base_id,
            action="promotion",
            pointer=pointer,
        )
        promotion_audit = {
            "protocol_version": "vector_shadow_atomic_promotion_v1",
            "candidate_id": candidate.id,
            "candidate_hash": candidate.candidate_hash,
            "build_id": build.id,
            "knowledge_base_id": build.knowledge_base_id,
            "base_vector_state_hash": previous_state_hash,
            "promoted_vector_state_hash": pointer.state_hash,
            "activation_generation": next_generation,
            "previous_graph_state_ids": previous_graph_ids,
            "promoted_graph_state_ids": candidate_graph_ids,
            "vector_record_count": len(candidate_records),
            "chunk_scope_promoted": candidate_chunk_scope is not None,
            "base_chunk_count": len(previous_chunk_ids),
            "promoted_chunk_count": len(chunk_ids),
            "cache_invalidation_intent_id": cache_intent.id,
            "cache_invalidation_status": cache_intent.status,
            "qdrant_mutated": False,
            "active_env_mutated": False,
            "promoted_at": promoted_at.isoformat(),
        }
        pointer.promotion_audit_json = promotion_audit
        build.status = "promoted"
        build.promoted_at = promoted_at
        build.blocking_reasons_json = []
        build.error_code = None
        build.last_error = None
        build.promotion_audit_json = promotion_audit
        promotion_rows.append(promotion_audit)

    candidate.status = "promoted"
    candidate.promoted_at = promoted_at
    candidate.blocking_reasons_json = []
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "atomic_promotion": {
            "protocol_version": "vector_shadow_atomic_promotion_v1",
            "promoted_at": promoted_at.isoformat(),
            "knowledge_bases": promotion_rows,
            "transaction_owner": "caller",
            "cache_invalidation_recovery": (
                VECTOR_RUNTIME_CACHE_INVALIDATION_PROTOCOL_VERSION
            ),
        },
    }
    db.flush()
    return {
        "protocol_version": "vector_shadow_atomic_promotion_v1",
        "candidate_id": candidate.id,
        "candidate_hash": candidate.candidate_hash,
        "allowed": True,
        "promoted": True,
        "idempotent_replay": False,
        "knowledge_base_ids": [row["knowledge_base_id"] for row in promotion_rows],
        "cache_invalidation_intent_ids": [
            row["cache_invalidation_intent_id"] for row in promotion_rows
        ],
        "active_pointer_mutated": True,
        "qdrant_mutated": False,
    }


def reconcile_vector_runtime_cache_invalidations(
    db: Session,
    *,
    candidate_id: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Retry durable post-commit cache invalidations without local fallback."""

    statement = (
        select(IngestionCompensationLog)
        .where(
            IngestionCompensationLog.operation
            == VECTOR_RUNTIME_CACHE_INVALIDATION_OPERATION,
            IngestionCompensationLog.status == "cache_invalidation_pending",
        )
        .order_by(IngestionCompensationLog.created_at.asc())
    )
    if candidate_id is not None:
        statement = statement.where(
            IngestionCompensationLog.payload_json["candidate_id"].as_string()
            == candidate_id
        )
    rows = list(
        db.scalars(
            (statement if dry_run else statement.with_for_update()).limit(
                MAX_CACHE_INVALIDATION_INTENT_SCAN + 1
            )
        ).all()
    )
    if len(rows) > MAX_CACHE_INVALIDATION_INTENT_SCAN:
        raise RuntimeError("Vector runtime cache reconciliation refused an unbounded intent scan")
    targets: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        payload = dict(row.payload_json or {})
        stored_hash = str(payload.pop("payload_hash", ""))
        if (
            payload.get("protocol_version")
            != VECTOR_RUNTIME_CACHE_INVALIDATION_PROTOCOL_VERSION
            or stored_hash != _stable_hash(payload)
        ):
            raise RuntimeError(
                f"Vector runtime cache invalidation intent {row.id} failed its payload hash"
            )
        pointer = db.scalar(
            select(KnowledgeBaseVectorRuntimeState).where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == row.knowledge_base_id
            )
        )
        if pointer is None or pointer.state_hash != payload.get("pointer_state_hash"):
            raise RuntimeError(
                f"Vector runtime cache invalidation intent {row.id} no longer matches the active pointer"
            )
        targets.append(
            {
                "intent_id": row.id,
                "knowledge_base_id": row.knowledge_base_id,
                "candidate_id": payload["candidate_id"],
                "action": payload["action"],
                "pointer_state_hash": payload["pointer_state_hash"],
            }
        )
        if dry_run:
            continue
        from app.services.context_graph import invalidate_context_graph_cache_after_commit

        try:
            invalidate_context_graph_cache_after_commit(
                row.knowledge_base_id,
                strict=True,
            )
        except Exception as exc:
            now = datetime.utcnow().isoformat()
            failed_payload = {
                **payload,
                "attempt_count": int(payload.get("attempt_count") or 0) + 1,
                "last_attempt_at": now,
            }
            row.payload_json = {
                **failed_payload,
                "payload_hash": _stable_hash(failed_payload),
            }
            row.error_message = type(exc).__name__
            failures.append(
                {
                    "intent_id": row.id,
                    "knowledge_base_id": row.knowledge_base_id,
                    "error_type": type(exc).__name__,
                    "recovery": "retry_same_pending_intent",
                }
            )
            continue
        now = datetime.utcnow().isoformat()
        completed_payload = {
            **payload,
            "phase": "completed",
            "attempt_count": int(payload.get("attempt_count") or 0) + 1,
            "last_attempt_at": now,
            "completed_at": now,
        }
        row.payload_json = {
            **completed_payload,
            "payload_hash": _stable_hash(completed_payload),
        }
        row.status = "committed"
        row.error_message = None
    if not dry_run:
        db.flush()
    return {
        "protocol_version": VECTOR_RUNTIME_CACHE_INVALIDATION_PROTOCOL_VERSION,
        "dry_run": bool(dry_run),
        "target_count": len(targets),
        "scan_limit": MAX_CACHE_INVALIDATION_INTENT_SCAN,
        "scan_truncated": False,
        "targets": targets,
        "applied": not dry_run and not failures,
        "failed_count": len(failures),
        "failures": failures,
    }


def _set_graph_states(
    db: Session,
    *,
    context_state_id: str | None,
    relation_state_id: str | None,
    mid_state_id: str | None,
    coarse_state_id: str | None,
    state: str,
) -> None:
    for model, state_id in (
        (ContextGraphState, context_state_id),
        (ChunkRelationGraphState, relation_state_id),
        (MidConceptState, mid_state_id),
        (CoarseConceptState, coarse_state_id),
    ):
        if state_id:
            item = db.get(model, state_id)
            if item is None:
                raise RuntimeError(f"Rollback graph state is missing: {model.__name__}:{state_id}")
            item.state = state


def _locked_rechunk_rollback_scope(
    db: Session,
    *,
    build: VectorShadowBuild,
    card: dict[str, Any],
) -> dict[str, Any]:
    if card.get("protocol_version") != "runtime_settings_shadow_rechunk_rollback_v1":
        raise RuntimeError("Rechunk rollback scope protocol is invalid")
    base_chunk_ids = sorted(str(value) for value in card.get("base_chunk_ids") or [])
    candidate_chunk_ids = sorted(
        str(value) for value in card.get("candidate_chunk_ids") or []
    )
    base_version_ids = sorted(
        str(value) for value in card.get("base_document_version_ids") or []
    )
    candidate_version_ids = sorted(
        str(value)
        for value in card.get("candidate_document_version_ids") or []
    )
    if not all((base_chunk_ids, candidate_chunk_ids, base_version_ids, candidate_version_ids)):
        raise RuntimeError("Rechunk rollback scope is incomplete")
    base_chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(base_chunk_ids))
            .order_by(Chunk.id.asc())
            .with_for_update()
        ).all()
    )
    candidate_chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(candidate_chunk_ids))
            .order_by(Chunk.id.asc())
            .with_for_update()
        ).all()
    )
    if (
        [str(row.id) for row in base_chunks] != base_chunk_ids
        or [str(row.id) for row in candidate_chunks] != candidate_chunk_ids
        or any(row.state != "inactive" for row in base_chunks)
        or any(row.state != "active" for row in candidate_chunks)
    ):
        raise RuntimeError("Rechunk rollback chunk state is no longer recoverable")
    base_versions = list(
        db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.id.in_(base_version_ids))
            .order_by(DocumentVersion.id.asc())
            .with_for_update()
        ).all()
    )
    candidate_versions = list(
        db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.id.in_(candidate_version_ids))
            .order_by(DocumentVersion.id.asc())
            .with_for_update()
        ).all()
    )
    if (
        [str(row.id) for row in base_versions] != base_version_ids
        or [str(row.id) for row in candidate_versions] != candidate_version_ids
        or any(row.is_active for row in base_versions)
        or any(not row.is_active for row in candidate_versions)
    ):
        raise RuntimeError("Rechunk rollback document-version state is no longer recoverable")
    knowledge_base = db.scalar(
        select(KnowledgeBase)
        .where(KnowledgeBase.id == build.knowledge_base_id)
        .with_for_update()
    )
    if knowledge_base is None or int(knowledge_base.current_chunk_version or 0) != int(
        card.get("candidate_chunk_version") or 0
    ):
        raise RuntimeError("Rechunk rollback knowledge-base version drifted")
    base_descriptor = db.scalar(
        select(ChunkVersion)
        .where(
            ChunkVersion.knowledge_base_id == build.knowledge_base_id,
            ChunkVersion.chunk_version == int(card["base_chunk_version"]),
        )
        .with_for_update()
    )
    candidate_descriptor = db.scalar(
        select(ChunkVersion)
        .where(
            ChunkVersion.knowledge_base_id == build.knowledge_base_id,
            ChunkVersion.chunk_version == int(card["candidate_chunk_version"]),
        )
        .with_for_update()
    )
    if base_descriptor is None or candidate_descriptor is None:
        raise RuntimeError("Rechunk rollback ChunkVersion descriptor is missing")
    return {
        "base_chunk_ids": base_chunk_ids,
        "candidate_chunk_ids": candidate_chunk_ids,
        "base_chunks": base_chunks,
        "candidate_chunks": candidate_chunks,
        "base_versions": base_versions,
        "candidate_versions": candidate_versions,
        "knowledge_base": knowledge_base,
        "base_descriptor": base_descriptor,
        "candidate_descriptor": candidate_descriptor,
        "base_chunk_version": int(card["base_chunk_version"]),
    }


def _restore_previous_chunk_scope(db: Session, scope: dict[str, Any]) -> None:
    for row in scope["candidate_chunks"]:
        row.state = "rolled_back"
    for row in scope["candidate_versions"]:
        row.is_active = False
    # See the matching promotion hand-off above: force deactivation before
    # reactivating the retained versions so the partial unique constraint is
    # never transiently violated on PostgreSQL or SQLite.
    db.flush()
    for row in scope["base_chunks"]:
        row.state = "active"
    for row in scope["base_versions"]:
        row.is_active = True
    scope["candidate_descriptor"].state = "inactive"
    scope["base_descriptor"].state = "active"
    scope["knowledge_base"].current_chunk_version = scope["base_chunk_version"]


def rollback_vector_shadow_candidate(
    db: Session,
    candidate_id: str,
    *,
    reason: str,
) -> RuntimeSettingsCandidate:
    """Restore the exact retained pre-promotion pointer; caller owns commit."""

    candidate_preview = db.get(RuntimeSettingsCandidate, candidate_id)
    if candidate_preview is None:
        raise ValueError(f"Unknown runtime settings candidate: {candidate_id}")
    if candidate_preview.status == "rolled_back":
        candidate = db.scalar(
            select(RuntimeSettingsCandidate)
            .where(RuntimeSettingsCandidate.id == candidate_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if candidate is None:
            raise RuntimeError("Rolled-back candidate disappeared while locking")
        _assert_rolled_back_candidate_replay(db, candidate)
        return candidate
    if candidate_preview.status != "promoted":
        raise RuntimeError(
            f"Candidate {candidate_preview.id} is {candidate_preview.status}; "
            "only promoted candidates can roll back"
        )

    preview_builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(VectorShadowBuild.runtime_settings_candidate_id == candidate_id)
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
            .limit(MAX_CANDIDATE_KNOWLEDGE_BASES + 1)
        ).all()
    )
    if (
        not preview_builds
        or len(preview_builds) > MAX_CANDIDATE_KNOWLEDGE_BASES
        or any(build.status != "promoted" for build in preview_builds)
    ):
        raise RuntimeError("Rollback preview requires every bounded candidate build to be promoted")
    rollback_collections: set[str] = set()
    for preview_build in preview_builds:
        pointer_preview = db.scalar(
            select(KnowledgeBaseVectorRuntimeState).where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == preview_build.knowledge_base_id
            )
        )
        if (
            pointer_preview is None
            or pointer_preview.runtime_settings_candidate_id != candidate_id
        ):
            raise RuntimeError("Rollback preview no longer owns every active pointer")
        previous_preview = dict(pointer_preview.previous_state_json or {})
        if (
            previous_preview.get("protocol_version")
            != VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION
            or previous_preview.get("rollback_eligible") is not True
        ):
            raise RuntimeError("Rollback preview lacks an eligible frozen previous state")
        previous_schema_preview = FrozenVectorSchema.model_validate(
            previous_preview.get("vector_schema")
        )
        if (
            previous_preview.get("vector_schema_hash")
            != vector_schema_hash(previous_schema_preview)
        ):
            raise RuntimeError("Rollback preview vector schema hash is invalid")
        rollback_collections.add(previous_schema_preview.collection_name)

    # Stage/cleanup take the same exact-collection lock before KB/pointer rows.
    # Acquire every rollback target in canonical order before taking candidate,
    # build, or pointer row locks to avoid a collection<->pointer lock inversion.
    from app.services.vector_collection_cleanup import (
        vector_collection_lifecycle_lock,
    )

    for collection_name in sorted(rollback_collections):
        vector_collection_lifecycle_lock(db, collection_name)

    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == candidate_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if candidate is None:
        raise RuntimeError("Vector runtime candidate disappeared while locking")
    if candidate.status == "rolled_back":
        _assert_rolled_back_candidate_replay(db, candidate)
        return candidate
    if candidate.status != "promoted":
        raise RuntimeError(
            f"Candidate {candidate.id} is {candidate.status}; only promoted candidates can roll back"
        )
    builds = _terminal_candidate_builds(
        db,
        candidate,
        expected_status="promoted",
    )
    now = datetime.utcnow()
    for build in builds:
        pointer = db.scalar(
            select(KnowledgeBaseVectorRuntimeState)
            .where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == build.knowledge_base_id
            )
            .with_for_update()
        )
        if pointer is None or pointer.runtime_settings_candidate_id != candidate.id:
            raise RuntimeError(
                f"Knowledge base {build.knowledge_base_id} no longer points at candidate {candidate.id}"
            )
        _validate_pointer(pointer)
        previous = dict(pointer.previous_state_json or {})
        if (
            previous.get("protocol_version") != VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION
            or previous.get("rollback_eligible") is not True
        ):
            raise RuntimeError("Active pointer does not contain an eligible frozen rollback state")
        previous_schema = FrozenVectorSchema.model_validate(previous.get("vector_schema"))
        previous_schema_hash = vector_schema_hash(previous_schema)
        if previous.get("vector_schema_hash") != previous_schema_hash:
            raise RuntimeError("Frozen rollback vector schema hash is invalid")
        # Exact old-collection cleanup is allowed to relinquish rollback
        # retention.  Serialize against its durable pending intent before any
        # pointer or record mutation so rollback can never reactivate a target
        # whose Qdrant deletion has already been authorized.
        from app.services.vector_collection_cleanup import (
            assert_vector_collection_not_pending_cleanup,
        )

        assert_vector_collection_not_pending_cleanup(
            db,
            previous_schema.collection_name,
        )
        chunk_scope_card = dict(previous.get("chunk_scope") or {})
        rechunk_scope = (
            _locked_rechunk_rollback_scope(
                db,
                build=build,
                card=chunk_scope_card,
            )
            if chunk_scope_card
            else None
        )
        promoted_chunk_ids = (
            list(rechunk_scope["candidate_chunk_ids"])
            if rechunk_scope is not None
            else _assert_staged_scope_unchanged(db, build)
        )
        previous_chunk_ids = (
            list(rechunk_scope["base_chunk_ids"])
            if rechunk_scope is not None
            else list(promoted_chunk_ids)
        )
        promoted_schema = _schema_from_pointer(pointer)
        promoted_records = _records_for_schema(
            db,
            knowledge_base_id=build.knowledge_base_id,
            schema=promoted_schema,
            chunk_ids=promoted_chunk_ids,
            for_update=True,
        )
        _assert_exact_record_scope(
            promoted_records,
            chunk_ids=promoted_chunk_ids,
            expected_status="ready",
            scope_name="promoted active",
        )
        previous_records = _records_for_schema(
            db,
            knowledge_base_id=build.knowledge_base_id,
            schema=previous_schema,
            chunk_ids=previous_chunk_ids,
            for_update=True,
        )
        _assert_exact_record_scope(
            previous_records,
            chunk_ids=previous_chunk_ids,
            expected_status="rollback_retained",
            scope_name="rollback retained",
        )
        current_graph = _graph_state_ids_from_pointer(pointer)
        current_bundle = _locked_graph_bundle(
            db,
            knowledge_base_id=build.knowledge_base_id,
            graph_state_ids=current_graph,
            expected_state="active",
        )
        previous_graph = dict(previous.get("graph_state_ids") or {})
        previous_bundle = _locked_graph_bundle(
            db,
            knowledge_base_id=build.knowledge_base_id,
            graph_state_ids=previous_graph,
            expected_state="inactive",
        )
        if rechunk_scope is not None:
            _restore_previous_chunk_scope(db, rechunk_scope)
        _set_graph_bundle_state(db, current_bundle, "inactive")
        _set_graph_bundle_state(db, previous_bundle, "active")
        for record in promoted_records:
            record.vector_status = "rolled_back_retained"
        for record in previous_records:
            record.vector_status = "ready"

        pointer.runtime_settings_candidate_id = previous.get(
            "runtime_settings_candidate_id"
        )
        pointer.embedding_model = previous_schema.embedding_model
        pointer.embedding_dimension = previous_schema.embedding_dimension
        pointer.distance_metric = previous_schema.distance_metric
        pointer.embedding_text_version = previous_schema.embedding_text_version
        pointer.chunk_schema_version = previous_schema.chunk_schema_version
        pointer.collection_identity_protocol_version = (
            previous_schema.collection_identity_protocol_version
        )
        pointer.collection_identity_digest = previous_schema.collection_identity_digest
        pointer.collection_name = previous_schema.collection_name
        pointer.vector_schema_hash = previous_schema_hash
        pointer.active_context_graph_state_id = previous_graph.get("context")
        pointer.active_chunk_relation_graph_state_id = previous_graph.get("relation")
        pointer.active_mid_concept_state_id = previous_graph.get("mid")
        pointer.active_coarse_concept_state_id = previous_graph.get("coarse")
        pointer.activation_generation += 1
        pointer.previous_state_json = {}
        pointer.promotion_audit_json = {
            **dict(pointer.promotion_audit_json or {}),
            "rollback": {
                "protocol_version": VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION,
                "candidate_id": candidate.id,
                "reason": str(reason or "unspecified"),
                "rolled_back_at": now.isoformat(),
            },
        }
        pointer.state_hash = vector_runtime_state_hash(
            knowledge_base_id=pointer.knowledge_base_id,
            runtime_settings_candidate_id=pointer.runtime_settings_candidate_id,
            activation_generation=pointer.activation_generation,
            schema=previous_schema,
            graph_state_ids=previous_graph,
        )
        for row in previous_bundle.values():
            if row is not None:
                row.diagnostics_json = {
                    **dict(row.diagnostics_json or {}),
                    "active_vector_runtime_state_id": pointer.id,
                    "active_vector_runtime_state_hash": pointer.state_hash,
                    "active_vector_schema_hash": pointer.vector_schema_hash,
                    "restored_by_vector_runtime_rollback": True,
                }
                if isinstance(row, ContextGraphState):
                    _bind_context_qdrant_proof_to_active_pointer(row, pointer)
        db.flush()
        cache_intent = _cache_invalidation_intent(
            db,
            candidate_id=candidate.id,
            knowledge_base_id=build.knowledge_base_id,
            action="rollback",
            pointer=pointer,
        )
        build.status = "rolled_back"
        build.rolled_back_at = now
        build.rollback_audit_json = {
            "protocol_version": VECTOR_SHADOW_ROLLBACK_PROTOCOL_VERSION,
            "reason": str(reason or "unspecified"),
            "restored_vector_schema_hash": previous_schema_hash,
            "restored_vector_state_hash": pointer.state_hash,
            "cache_invalidation_intent_id": cache_intent.id,
            "cache_invalidation_status": cache_intent.status,
            "qdrant_mutated": False,
            "chunk_scope_restored": rechunk_scope is not None,
            "restored_chunk_count": len(previous_chunk_ids),
            "rolled_back_at": now.isoformat(),
        }
    candidate.status = "rolled_back"
    candidate.rolled_back_at = now
    candidate.error_code = None
    candidate.last_error = None
    db.flush()
    return candidate
