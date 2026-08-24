from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from app.core.config import (
    QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX,
    TRAVERSAL_OBSERVATION_BUDGET_MAX,
)
from app.core.sensitive_fields import (
    SENSITIVE_FIELD_KEY_PROTOCOL_VERSION,
    sensitive_field_paths,
)
from app.services.graph_protocols import (
    retrieval_node_contribution_facts,
    retrieval_path_contribution_id,
)


JobState = Literal[
    "queued",
    "parsing",
    "chunking",
    "embedding",
    "extracting_graph",
    "context_graph",
    "processing",
    "cancel_requested",
    "cancelling",
    "compensating",
    "cancelled",
    "cancel_failed",
    "completed",
    "partial_failed",
    "failed",
    "skipped",
]
KnowledgeBaseFileStatus = Literal[
    "pending",
    "parsing",
    "parsed",
    "failed",
    "skipped",
    "active",
]

SourceType = Literal["upload", "storage", "watchdog", "batch", "unknown"]
GraphType = Literal["chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts", "context-graph"]
AgentRoute = Literal[
    "layered_context_graph",
    "direct_answer",
    "multi_hop_research",
    "definition_lookup",
    "formula_table_lookup",
    "cross_document_synthesis",
]
RetrievalGranularity = Literal["mid", "coarse"]
CONVERSATION_CLIENT_HISTORY_MAX_TURNS = 32
CONVERSATION_CLIENT_HISTORY_MAX_TOKENS = 8_192
CONVERSATION_QUESTION_MAX_TOKENS = 2_048


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClosedContractModel(BaseModel):
    """Public audit contract: versioned and fail-closed on renamed fields."""

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    def __getitem__(self, key: str) -> Any:
        """Allow legacy read-only mapping access without accepting open fields."""

        if key not in self.__class__.model_fields:
            raise KeyError(key)
        return getattr(self, key)


def _forbidden_public_response_paths(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> list[str]:
    prefix = ".".join(path)
    return [
        f"{prefix}.{item}" if prefix else item
        for item in sensitive_field_paths(
            value,
            include_public_private=True,
        )
    ]


class PublicResponseModel(ClosedContractModel):
    """Closed public projection with recursive secret/profile leak rejection."""

    @model_validator(mode="before")
    @classmethod
    def reject_private_payloads(cls, value: object) -> object:
        forbidden_paths = sorted(
            set(_forbidden_public_response_paths(value))
        )
        if forbidden_paths:
            raise ValueError(
                "Public response contains forbidden private fields under "
                f"{SENSITIVE_FIELD_KEY_PROTOCOL_VERSION}"
            )
        return value


class SearchFilters(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    document_ids: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    source_type: str | None = None
    partition: str | None = None
    tags: list[str] = Field(default_factory=list)
    page_range: tuple[int | None, int | None] | None = None
    content_kinds: list[str] = Field(default_factory=list)
    chunk_version: int | None = None


class UploadReplacementLockReleaseAudit(APIModel):
    persisted: bool
    intent_id: str
    error: str | None = None


class UploadReplacementLockReleaseDiagnostics(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    resource_key: str
    knowledge_base_id: str
    advisory_key: int
    backend: Literal["postgresql"]
    operation: str
    batch_id: str | None
    protocol_version: str
    release_error: str


class UploadReplacementAudit(APIModel):
    protocol_version: str
    intent_id: str
    status: Literal["completed", "cleanup_pending", "manual_review"]
    phase: Literal["completed", "cleanup_pending", "manual_review"]
    database_committed: bool
    cleanup_pending: bool
    postcommit_lock_release_failure: UploadReplacementLockReleaseDiagnostics | None = None
    lock_release_audit: UploadReplacementLockReleaseAudit | None = None


class LanguageIdentitySummary(APIModel):
    status: Literal["pending", "resolved"]
    language: str | None = None
    source: Literal["explicit_metadata", "deterministic_detection", "unknown"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    protocol_version: str | None = None
    detection_hash: str | None = None
    explicit_language_tag: str | None = None
    decision_reason: str | None = None


class UploadFileResponse(APIModel):
    document_id: str
    job_id: str
    status: str
    source_path: str
    language_identity: LanguageIdentitySummary
    upload_replacement: UploadReplacementAudit


class ParseUploadedFilesRequest(APIModel):
    file_paths: list[str] | None = None
    force: bool = False
    full_reparse: bool = False


class BatchStartResponse(APIModel):
    batch_id: str
    state: str


class JobStatusResponse(APIModel):
    job_id: str
    state: JobState | str
    document_id: str | None = None
    source_path: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class CitationBoundingBox(ClosedContractModel):
    page_number: int | None = None
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    coordinate_system: str | None = None
    synthetic: bool | None = None
    raw_bbox: list[float] | None = None
    raw_coordinate_system: str | None = None
    page_size: list[float] | None = None
    source_bbox_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_native_coordinate_card(self) -> "CitationBoundingBox":
        if self.raw_bbox is not None and len(self.raw_bbox) != 4:
            raise ValueError("citation raw_bbox must contain exactly four coordinates")
        if self.page_size is not None and len(self.page_size) != 2:
            raise ValueError("citation page_size must contain exactly two dimensions")
        return self


class SourceSnapshotVerification(ClosedContractModel):
    protocol_version: str
    final_open_protocol_version: str
    storage_path: str
    checksum: str = Field(min_length=64, max_length=64)
    verified: Literal[True]
    size_bytes: int = Field(ge=0)


class CitationSourceSpan(ClosedContractModel):
    contract_version: Literal["raw_chunk_source_span_v1"] = "raw_chunk_source_span_v1"
    document_version_id: str
    chunk_id: str
    source_path: str
    source_checksum: str = Field(min_length=64, max_length=64)
    logical_source_path: str
    source_snapshot_verification: SourceSnapshotVerification
    chunk_text_hash_protocol_version: str
    chunk_text_hash: str = Field(min_length=64, max_length=64)
    raw_span_text_hash_protocol_version: str
    raw_span_text_hash: str = Field(min_length=64, max_length=64)
    char_span: list[int] | tuple[int, int]
    raw_chunk_char_span: list[int] | tuple[int, int] | None = None
    page_range: list[int | None] | tuple[int | None, int | None]
    section_path: str | list[str] | None = None
    structure_path: str | list[str] | None = None
    structure_node_ids: list[str] = Field(default_factory=list)
    bbox: CitationBoundingBox | None = None
    context_package_id: str | None = None
    retrieval_trace_id: str | None = None
    verification_id: str | None = None
    content_clipped: bool = False
    content_token_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_span_order(self) -> "CitationSourceSpan":
        if self.source_checksum != self.source_snapshot_verification.checksum:
            raise ValueError(
                "citation source checksum must match the verified source snapshot"
            )
        if len(self.char_span) != 2:
            raise ValueError("citation char_span must contain exactly two offsets")
        if len(self.page_range) != 2:
            raise ValueError("citation page_range must contain exactly two page bounds")
        start, end = self.char_span
        if start < 0 or end < start:
            raise ValueError("citation char_span must be an ordered non-negative raw span")
        if self.raw_chunk_char_span is not None:
            if len(self.raw_chunk_char_span) != 2:
                raise ValueError(
                    "citation raw_chunk_char_span must contain exactly two offsets"
                )
            raw_start, raw_end = self.raw_chunk_char_span
            if raw_start < 0 or raw_end < raw_start or start < raw_start or end > raw_end:
                raise ValueError("citation char_span must be contained by raw_chunk_char_span")
        return self


class VerifiedCitationSourceSpan(CitationSourceSpan):
    """Raw address after it has been bound to a persisted verification."""

    context_package_id: str = Field(min_length=1)
    retrieval_trace_id: str = Field(min_length=1)
    verification_id: str = Field(min_length=1)


class CitationVerificationDiagnostics(ClosedContractModel):
    verification_method: str = Field(min_length=1)
    claim_grounded_gate_protocol_version: str = Field(min_length=1)
    claim_id: str | None
    claim_index: int | None = Field(ge=0)
    answer_hash: str | None = Field(min_length=64, max_length=64)
    citation_provenance_protocol_version: str = Field(min_length=1)
    citation_provenance_valid: bool | None
    citation_provenance_hash: str | None
    citation_provenance_reasons: list[str]
    citation_provenance_fail_closed: Literal[True]
    citation_provenance_llm_override_allowed: Literal[False]
    citation_provenance_session_hash: str | None
    citation_provenance_persistence_gate_passed: bool | None
    llm_entailment_judge: str | None = None
    rule_verdict: Literal[
        "supported",
        "unsupported",
        "contradicted",
        "missing_citation",
        "structure_context_missing",
        "formula_table_context_missing",
    ] | None
    llm_entailment_verdict: Literal[
        "supported",
        "unsupported",
        "contradicted",
        "missing_citation",
        "structure_context_missing",
        "formula_table_context_missing",
    ] | None
    llm_entailment_result_present: bool
    llm_entailment_reason: str | None = None
    deterministic_exact_span_entailment: bool = False
    deterministic_exact_span_entailment_protocol_version: str | None = None
    citation_prompt_protocol_hash: str | None
    citation_grounding_envelope_protocol_version: str | None
    citation_grounding_envelope_hash: str | None
    citation_profile_hash: str | None
    citation_verification_microbatch_protocol_version: str | None = None
    citation_verification_microbatch_size: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )
    citation_verification_model_call_count: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )
    reason: str | None = None


class CitationVerificationAudit(ClosedContractModel):
    contract_version: Literal["citation_verification_public_v1"] = "citation_verification_public_v1"
    verdict: Literal[
        "supported",
        "unsupported",
        "contradicted",
        "missing_citation",
        "structure_context_missing",
        "formula_table_context_missing",
    ]
    failure_type: str = Field(min_length=1, max_length=128)
    provenance_status: Literal["valid", "invalid", "missing"]
    structure_context_status: Literal["valid", "invalid", "missing"]
    confidence: float = Field(ge=0.0, le=1.0)
    diagnostics: CitationVerificationDiagnostics

    @model_validator(mode="after")
    def validate_replay_consistency(self) -> "CitationVerificationAudit":
        diagnostics = self.diagnostics
        rule_verdict = diagnostics.rule_verdict
        llm_verdict = diagnostics.llm_entailment_verdict or rule_verdict
        exact_span_supported = (
            diagnostics.deterministic_exact_span_entailment is True
            and diagnostics.deterministic_exact_span_entailment_protocol_version
            == "claim_raw_span_exact_entailment_v1"
            and diagnostics.llm_entailment_result_present is False
            and diagnostics.llm_entailment_judge
            == "skipped_deterministic_exact_span"
            and rule_verdict == "supported"
        )
        if exact_span_supported:
            replayed_verdict = "supported"
        elif rule_verdict in {
            "missing_citation",
            "formula_table_context_missing",
            "structure_context_missing",
        }:
            replayed_verdict = rule_verdict
        elif llm_verdict in {
            "contradicted",
            "unsupported",
            "missing_citation",
            "formula_table_context_missing",
        }:
            replayed_verdict = llm_verdict
        elif (
            rule_verdict == "supported"
            and diagnostics.llm_entailment_result_present
            and llm_verdict == "supported"
        ):
            replayed_verdict = "supported"
        else:
            replayed_verdict = "unsupported"

        provenance_valid = (
            diagnostics.citation_provenance_valid is True
            and diagnostics.citation_provenance_fail_closed is True
            and diagnostics.citation_provenance_llm_override_allowed is False
            and diagnostics.citation_provenance_persistence_gate_passed is True
        )
        if replayed_verdict == "supported" and not provenance_valid:
            replayed_verdict = "structure_context_missing"
        if self.verdict != replayed_verdict:
            raise ValueError(
                "citation verdict does not replay from rule, LLM entailment, "
                "and persisted provenance diagnostics"
            )

        expected_provenance_status = "valid" if provenance_valid else "invalid"
        if self.provenance_status != expected_provenance_status:
            raise ValueError(
                "citation provenance_status conflicts with persisted provenance diagnostics"
            )

        if self.verdict in {
            "structure_context_missing",
            "formula_table_context_missing",
        }:
            expected_structure_status = "missing"
        elif self.provenance_status == "invalid":
            expected_structure_status = "invalid"
        else:
            expected_structure_status = "valid"
        if self.structure_context_status != expected_structure_status:
            raise ValueError(
                "citation structure_context_status conflicts with verdict or provenance"
            )

        if self.verdict == "supported":
            if self.failure_type != "none":
                raise ValueError("supported citation must use failure_type=none")
            llm_supported = (
                diagnostics.llm_entailment_result_present is True
                and diagnostics.llm_entailment_verdict == "supported"
            )
            if rule_verdict != "supported" or not (
                llm_supported or exact_span_supported
            ):
                raise ValueError(
                    "supported citation requires matching rule and an audited "
                    "LLM or deterministic exact-span entailment result"
                )
        elif self.failure_type.strip().casefold() in {"", "none"}:
            raise ValueError("non-supported citation must carry an actual failure_type")

        if self.verdict == "contradicted" and (
            diagnostics.llm_entailment_result_present is not True
            or diagnostics.llm_entailment_verdict != "contradicted"
        ):
            raise ValueError(
                "contradicted citation requires a present contradicted LLM entailment result"
            )
        return self


class Citation(ClosedContractModel):
    contract_version: Literal["citation_public_v1"] = "citation_public_v1"
    chunk_id: str = Field(min_length=1)
    citation_index: int = Field(ge=1)
    claim_id: str | None = Field(min_length=64, max_length=64)
    claim_index: int | None = Field(ge=0)
    claim_text: str | None
    answer_hash: str | None = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    title: str | None = None
    document_title: str | None = None
    source_path: str = Field(min_length=1)
    logical_source_path: str = Field(min_length=1)
    partition: str | None = None
    section: str | list[str] | None = None
    page_number: int | None = None
    context_package_id: str = Field(min_length=1)
    page_range: list[int | None] | tuple[int | None, int | None]
    char_span: list[int] | tuple[int, int]
    bbox: CitationBoundingBox | None = None
    section_path: list[str]
    text: str | None = None
    snippet: str | None = None
    source_span: VerifiedCitationSourceSpan
    retrieval_trace_id: str = Field(min_length=1)
    answer_session_id: str = Field(min_length=1)
    citation_verification_id: str = Field(min_length=1)
    verification: CitationVerificationAudit

    @model_validator(mode="after")
    def validate_public_bindings(self) -> "Citation":
        span = self.source_span
        identity_pairs = (
            ("chunk_id", self.chunk_id, span.chunk_id),
            (
                "document_version_id",
                self.document_version_id,
                span.document_version_id,
            ),
            ("source_path", self.source_path, span.source_path),
            (
                "logical_source_path",
                self.logical_source_path,
                span.logical_source_path,
            ),
            (
                "context_package_id",
                self.context_package_id,
                span.context_package_id,
            ),
            (
                "retrieval_trace_id",
                self.retrieval_trace_id,
                span.retrieval_trace_id,
            ),
            (
                "citation_verification_id",
                self.citation_verification_id,
                span.verification_id,
            ),
        )
        for field_name, outer_value, span_value in identity_pairs:
            if outer_value != span_value:
                raise ValueError(
                    f"citation {field_name} does not match the raw source span"
                )
        if list(self.char_span) != list(span.char_span):
            raise ValueError("citation char_span does not match the raw source span")
        if list(self.page_range) != list(span.page_range):
            raise ValueError("citation page_range does not match the raw source span")

        def normalized_path(value: str | list[str] | None) -> list[str]:
            if isinstance(value, list):
                return value
            return [value] if value else []

        if self.section_path != normalized_path(span.section_path):
            raise ValueError(
                "citation section_path does not match the raw source span"
            )
        if self.bbox is not None and self.bbox != span.bbox:
            raise ValueError("citation bbox does not match the raw source span")

        diagnostics = self.verification.diagnostics
        if self.claim_id != diagnostics.claim_id:
            raise ValueError("citation claim_id does not match verification diagnostics")
        if self.claim_index != diagnostics.claim_index:
            raise ValueError(
                "citation claim_index does not match verification diagnostics"
            )
        if self.answer_hash != diagnostics.answer_hash:
            raise ValueError(
                "citation answer_hash does not match verification diagnostics"
            )
        if self.verification.verdict in {"supported", "contradicted"} and (
            self.claim_id is None
            or self.claim_index is None
            or not str(self.claim_text or "").strip()
            or self.answer_hash is None
        ):
            raise ValueError(
                "supported or contradicted citation requires a complete claim binding"
            )

        has_structure_address = (
            bool(normalized_path(span.section_path))
            and bool(normalized_path(span.structure_path))
            and bool(span.structure_node_ids)
        )
        if self.verification.verdict in {
            "structure_context_missing",
            "formula_table_context_missing",
        }:
            expected_structure_status = "missing"
        elif self.verification.provenance_status == "invalid":
            expected_structure_status = "invalid"
        elif not has_structure_address:
            expected_structure_status = "missing"
        else:
            expected_structure_status = "valid"
        if self.verification.structure_context_status != expected_structure_status:
            raise ValueError(
                "citation structure_context_status conflicts with its raw structure address"
            )
        return self


class SearchCitation(ClosedContractModel):
    """Unverified raw address returned by ordinary layered search."""

    contract_version: Literal["search_citation_public_v1"] = (
        "search_citation_public_v1"
    )
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str | None = None
    source_path: str = Field(min_length=1)
    logical_source_path: str = Field(min_length=1)
    partition: str | None = None
    section: str | list[str] | None = None
    page_number: int | None = None
    snippet: str | None = None
    retrieval_trace_id: str = Field(min_length=1)
    source_span: CitationSourceSpan

    @model_validator(mode="after")
    def validate_search_source_binding(self) -> "SearchCitation":
        span = self.source_span
        if (
            self.chunk_id != span.chunk_id
            or self.source_path != span.source_path
            or self.logical_source_path != span.logical_source_path
            or self.retrieval_trace_id != span.retrieval_trace_id
        ):
            raise ValueError(
                "search citation identity does not match its raw source span"
            )
        return self


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=12_000)
    knowledge_base_id: str | None = None
    session_id: str | None = Field(default=None, max_length=36)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_granularity: RetrievalGranularity = "mid"


class ActiveContextGraphAdmissionIssue(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    code: Literal["active_context_graph_not_admissible"]
    title: str
    message: str
    fix_commands: list[str]


class ActiveContextGraphAdmissionErrorDetail(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    protocol_version: Literal["active_context_graph_admission_error_v1"]
    code: Literal["active_context_graph_rebuild_required"]
    title: str
    message: str
    reason: Literal["active_graph_freshness_gate_rejected"]
    action: Literal["rebuild_context_graph"]
    issues: list[ActiveContextGraphAdmissionIssue]
    fix_commands: list[str]
    retryable: Literal[False]
    retry_after_rebuild: Literal[True]
    rebuild_required: Literal[True]


class ActiveContextGraphAdmissionErrorResponse(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    detail: ActiveContextGraphAdmissionErrorDetail


class SearchResult(PublicResponseModel):
    chunk_id: str
    document_id: str | None = None
    document_version_id: str | None = None
    title: str | None = None
    snippet: str = ""
    text: str = ""
    content: str | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None
    logical_source_path: str | None = None
    page_range: list[int] | tuple[int | None, int | None] | None = None
    char_span: list[int] | tuple[int | None, int | None] | None = None
    section_path: list[str] = Field(default_factory=list)
    graph_path: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[SearchCitation] = Field(default_factory=list)
    document_title: str | None = None
    partition: str | None = None
    source_type: str | None = None


def public_search_result_payload(value: object) -> dict[str, Any]:
    """Project an internal retrieval row onto the closed public result."""

    if not isinstance(value, dict):
        raise ValueError("Search result must be an object")
    projected = {
        key: value[key]
        for key in SearchResult.model_fields
        if key in value
    }
    return SearchResult.model_validate(projected).model_dump(mode="json")


class RetrievalCacheAudit(ClosedContractModel):
    protocol_version: Literal[
        "layered_retrieval_postgresql_strict_replay_v1"
    ]
    status: Literal["hit", "miss", "poison", "unavailable"]
    cache_hit: bool
    cache_miss: bool
    cache_key: str = Field(min_length=64, max_length=64)
    redis_key_digest: str = Field(min_length=64, max_length=64)
    ttl_seconds: int = Field(ge=1)
    ttl_seconds_remaining: int | None = Field(default=None, ge=1)
    reason: str
    deletion_attempted: bool
    deleted: bool
    source_retrieval_trace_id: str | None = None
    source_context_package_id: str | None = None
    postgresql_replay_required: Literal[True]
    redis_payload_used_as_evidence: Literal[False]
    provider_perception_model_call_count: Literal[0] | None = None
    query_embedding_model_call_count: Literal[0] | None = None
    traversal_execution_count: Literal[0] | None = None
    retrieval_fact_insert_count: Literal[0] | None = None
    gray_zone_input_modified: Literal[False]
    gray_zone_model_call_count: Literal[0]
    context_package_reused: bool | None = None
    write_scheduled_after_commit: bool | None = None
    ordinary_query_pointer_write_scheduled_after_commit: (
        bool | None
    ) = None


class OrdinaryQueryReplayPointerAudit(ClosedContractModel):
    status: Literal[
        "hit",
        "miss",
        "poison",
        "unavailable",
        "stale",
    ]
    reason: str
    pointer_key_digest: str = Field(min_length=64, max_length=64)
    ttl_seconds_remaining: int | None = Field(default=None, ge=1)
    deletion_attempted: bool = False
    deleted: bool = False
    source_retrieval_trace_id: str | None = None
    source_context_package_id: str | None = None
    full_cache_probe: RetrievalCacheAudit | None = None


class OrdinaryQueryPerceptionAudit(ClosedContractModel):
    protocol_version: Literal[
        "bounded_query_perception_and_facet_proposal_v1"
    ]
    provider_protocol_hash: str = Field(min_length=64, max_length=64)
    model_call_budget: Literal[2]
    model_call_count: int = Field(ge=1, le=2)
    budget_exhausted: Literal[False]
    intent_schema_validated: Literal[True]
    facet_schema_validated: bool
    query_intent_hash: str = Field(min_length=64, max_length=64)
    conversation_history_hash: str = Field(
        min_length=64,
        max_length=64,
    )
    conversation_history_turn_count: int = Field(ge=0)
    conversation_history_audit_hash: str = Field(
        min_length=64,
        max_length=64,
    )
    conversation_history_is_evidence: Literal[False]
    conversation_history_gray_zone_decision_authority: Literal[False]
    query_facet_packet_hash: str = Field(min_length=64, max_length=64)
    query_facet_packet_is_evidence: Literal[False]
    query_facet_packet_routing_only: Literal[True]
    suggested_strategy_is_executor_authority: Literal[False]
    retrieval_granularity_is_user_or_executor_locked: Literal[True]
    gray_zone_decision_authority: Literal[False]
    gray_zone_rule_inputs_modified: Literal[False]
    gray_zone_model_call_count: Literal[0]
    provider_free_pointer: OrdinaryQueryReplayPointerAudit


class QueryEmbeddingExecutionAudit(ClosedContractModel):
    protocol_version: Literal["request_scoped_query_embedding_memo_v1"]
    request_memo_enabled: bool
    request_memo_hit: bool
    request_memo_key_hash: str = Field(min_length=64, max_length=64)
    query_embedding_model_call_count: Literal[0, 1]
    provider_response_present: Literal[False]
    credentials_present: Literal[False]
    gray_zone_decision_authority: Literal[False]
    gray_zone_model_call_count: Literal[0]


class ModelAudit(ClosedContractModel):
    contract_version: Literal["model_audit_public_v1"] = "model_audit_public_v1"
    provider: str | None = None
    prompt_protocol_version: str | None = None
    prompt_protocol_hash: str | None = None
    grounding_envelope_protocol_version: str | None = None
    grounding_envelope_hash: str | None = None
    profile_hash: str | None = None
    embedding_model: str | None = None
    embedding_text_version: str | None = None
    retrieval_mode: str | None = None
    retrieval_granularity: RetrievalGranularity | None = None
    retrieval_trace_id: str | None = None
    context_package_id: str | None = None
    conversation_state_scope_hash: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    semantic_entry_query_protocol_version: Literal[
        "validated_query_facet_semantic_entry_v1"
    ] | None = None
    semantic_entry_query_hash: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    semantic_entry_query_selection_source: Literal[
        "validated_required_facet",
        "raw_query",
    ] | None = None
    semantic_entry_query_is_evidence: Literal[False] | None = None
    semantic_entry_query_citation_authority: Literal[False] | None = None
    semantic_entry_query_gray_zone_decision_authority: Literal[False] | None = None
    degraded: bool = False
    degraded_mode: bool = False
    fallback_used: bool = False
    fallback_enabled: bool = False
    latency_ms: int | None = None
    route: str | None = None
    retrieval_pipeline: str | None = None
    context_graph_state_id: str | None = None
    result_top_k: int | None = Field(default=None, ge=0)
    coarse_entries: int | None = Field(default=None, ge=0)
    mid_entries: int | None = Field(default=None, ge=0)
    rq_membership_entries: int | None = Field(default=None, ge=0)
    frontier_pops: int | None = Field(default=None, ge=0)
    stage_queue_count: int | None = Field(default=None, ge=0)
    mid_topk_selected: int | None = Field(default=None, ge=0)
    chunk_topk_selected: int | None = Field(default=None, ge=0)
    dominance_pruned_count: int | None = Field(default=None, ge=0)
    hard_stop_pruned_count: int | None = Field(default=None, ge=0)
    red_zone_pruned_count: int | None = Field(default=None, ge=0)
    gray_zone_decision_count: int | None = Field(default=None, ge=0)
    query_rq_path: list[int] = Field(default_factory=list)
    coarse_skipped_reason: str | None = None
    mid_direct_entry_count: int | None = Field(default=None, ge=0)
    repair_protocol_version: str | None = None
    repair_action_type: str | None = None
    repair_executor_mechanism: str | None = None
    repair_directive_hash: str | None = None
    repair_global_top_k_modified: Literal[False] | None = None
    repair_gray_zone_model_call_count: Literal[0] | None = None
    retrieval_cache: RetrievalCacheAudit | None = None
    query_perception_audit: OrdinaryQueryPerceptionAudit | None = None
    query_embedding_execution: QueryEmbeddingExecutionAudit | None = None


class SearchResponse(PublicResponseModel):
    contract_version: Literal["search_public_v1"] = "search_public_v1"
    query: str
    results: list[SearchResult]
    degraded_mode: bool = False
    model_audit: ModelAudit = Field(default_factory=ModelAudit)
    retrieval_trace_id: str | None = None
    context_package_id: str | None = None
    retrieval_granularity: RetrievalGranularity = "mid"
    conversation_state: "ConversationStatePayload | None" = None


class ChatMessage(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message content must not be blank")
        return normalized


class ConversationRetrievalConstraints(SearchFilters):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ConversationUserConstraints(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    instructions: list[str] = Field(default_factory=list, max_length=16)
    retrieval_filters: ConversationRetrievalConstraints = Field(
        default_factory=ConversationRetrievalConstraints
    )

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            if not item or len(item) > 512:
                raise ValueError(
                    "conversation constraint instructions must contain 1-512 characters"
                )
            normalized.append(item)
        return normalized


class ConversationTaskState(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    status: Literal[
        "active",
        "waiting_user",
        "completed",
        "cancelled",
    ] = "active"
    objective: str | None = Field(default=None, max_length=2_000)
    current_step: str | None = Field(default=None, max_length=1_000)

    @field_validator("objective", "current_step")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ConversationStateUpdate(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    active_user_constraints: ConversationUserConstraints | None = None
    task_state: ConversationTaskState | None = None


class ConversationHistoryReference(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    protocol_version: Literal["answer_context_citation_reference_v1"]
    turn_index: int = Field(ge=0)
    run_id: str
    answer_session_id: str
    context_package_id: str
    retrieval_trace_id: str
    citation_verification_ids: list[str] = Field(default_factory=list)


class ConversationStatePayload(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    protocol_version: Literal["conversation_state_v1"]
    scope_protocol_version: Literal["conversation_state_scope_v1"]
    qa_session_id: str | None = None
    knowledge_base_id: str
    revision: int = Field(ge=0)
    state_hash: str = Field(min_length=64, max_length=64)
    scope_hash: str = Field(min_length=64, max_length=64)
    active_user_constraints: ConversationUserConstraints
    task_state: ConversationTaskState
    history_references: list[ConversationHistoryReference] = Field(
        default_factory=list
    )
    transcript_message_count: int = Field(ge=0)
    prompt_history_audit: dict[str, Any] = Field(default_factory=dict)
    evidence_authority: Literal[False]
    gray_zone_decision_authority: Literal[False]


def _validate_conversation_request(
    *,
    question: str,
    history: list[ChatMessage],
) -> None:
    from app.services.chinese_text import estimate_tokens

    if estimate_tokens(question) > CONVERSATION_QUESTION_MAX_TOKENS:
        raise ValueError(
            f"question exceeds {CONVERSATION_QUESTION_MAX_TOKENS} estimated tokens"
        )
    token_count = sum(estimate_tokens(item.content) for item in history)
    if token_count > CONVERSATION_CLIENT_HISTORY_MAX_TOKENS:
        raise ValueError(
            "history exceeds "
            f"{CONVERSATION_CLIENT_HISTORY_MAX_TOKENS} estimated tokens"
        )
    if history and history[0].role != "user":
        raise ValueError("history must start with a user message")
    if history and history[-1].role != "assistant":
        raise ValueError(
            "history must end with an assistant message before a new question"
        )
    if any(
        history[index - 1].role == history[index].role
        for index in range(1, len(history))
    ):
        raise ValueError("history roles must alternate")


class QARequest(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    question: str = Field(min_length=1, max_length=12_000)
    knowledge_base_id: str | None = None
    session_id: str | None = Field(default=None, max_length=36)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=CONVERSATION_CLIENT_HISTORY_MAX_TURNS,
    )
    conversation_state_update: ConversationStateUpdate | None = None
    retrieval_granularity: RetrievalGranularity = "mid"

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_conversation_bounds(self) -> "QARequest":
        _validate_conversation_request(
            question=self.question,
            history=self.history,
        )
        return self


class ExpectedEvidenceAudit(ClosedContractModel):
    source: str | None = None
    requires_chunk_spans: bool | None = None
    required_facets: list[str] = Field(default_factory=list)
    allowed_relation_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    required_restore_modes: list[str] = Field(default_factory=list)
    minimum_independent_support_paths: int | None = Field(default=None, ge=0)
    required_evidence_roles: list[str] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)
    start_layer: str | None = None
    target_layer: str | None = None
    fallback_allowed: bool | None = None
    required_verification_stage: str | None = None
    protocol_version: str | None = None
    executor_mechanism: str | None = None
    failure_card_hashes: list[str] = Field(default_factory=list)
    action_input_hash: str | None = None


class EvidenceEvaluatorVerdictAudit(ClosedContractModel):
    protocol_version: str | None = None
    verdict: Literal[
        "sufficient",
        "need_more_same_node",
        "need_bridge_jump",
        "need_mid_expansion",
        "need_chunk_expansion",
        "need_structure_closure",
        "insufficient_corpus",
        "validator_rejection",
    ]
    reason: str
    target_ids: list[str] = Field(default_factory=list)
    expected_evidence: ExpectedEvidenceAudit = Field(default_factory=ExpectedEvidenceAudit)
    profile_hash: str | None = None
    prompt_protocol_hash: str | None = None
    schema_repair_attempted: bool = False
    decision_hash: str | None = None


class ClaimGroundedClaimAudit(ClosedContractModel):
    claim_id: str
    claim_index: int = Field(ge=0)
    claim_text: str
    answer_hash: str = Field(min_length=64, max_length=64)
    claim_id_protocol_version: str
    supported: bool
    candidate_verification_count: int = Field(ge=0)
    supported_verification_count: int = Field(ge=0)
    supported_citation_indexes: list[int] = Field(default_factory=list)
    supported_chunk_ids: list[str] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)


class ClaimGroundedGateAudit(ClosedContractModel):
    protocol_version: str
    answer_hash: str = Field(min_length=64, max_length=64)
    claim_id_protocol_version: str
    claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    claim_pass_rate: float = Field(ge=0.0, le=1.0)
    all_claims_supported: bool
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    claims: list[ClaimGroundedClaimAudit] = Field(default_factory=list)
    unbound_verification_count: int = Field(ge=0)
    unbound_verification_hash: str = Field(min_length=64, max_length=64)
    require_persistence_replay: bool
    gate_hash: str = Field(min_length=64, max_length=64)
    nonfactual_insufficiency_response: bool | None = None


class EvidenceGapAudit(ClosedContractModel):
    kind: Literal["unsupported_claims_removed", "no_supported_claims"] | None = None
    dropped_claim_count: int | None = Field(default=None, ge=0)
    dropped_claim_ids: list[str] = Field(default_factory=list)
    dropped_claim_texts: list[str] = Field(default_factory=list)
    kept_claim_ids: list[str] = Field(default_factory=list)
    repair_convergence_reason: str | None = None
    repair_round_budget: int | None = Field(default=None, ge=0)
    repair_rounds_used: int | None = Field(default=None, ge=0)
    unsupported_claims_removed: bool | None = None
    original_answer_hash: str | None = None
    original_claim_count: int | None = Field(default=None, ge=0)
    original_supported_claim_count: int | None = Field(default=None, ge=0)
    original_unsupported_claim_count: int | None = Field(default=None, ge=0)
    original_claim_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    pre_guard_gate_hash: str | None = None


class RepairCanonicalTargetRefsAudit(ClosedContractModel):
    claim_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_context_package_id: str
    source_retrieval_trace_id: str | None = None
    mid_concept_ids: list[str] = Field(default_factory=list)
    target_refs_hash: str = Field(min_length=64, max_length=64)


class RepairValidatedTargetsAudit(ClosedContractModel):
    action_target_ids: list[str] = Field(default_factory=list)
    canonical_target_refs: RepairCanonicalTargetRefsAudit
    supported_source_chunk_ids: list[str] = Field(default_factory=list)
    carry_forward_supported_chunk_ids: list[str] = Field(default_factory=list)
    bridge_seed_chunk_ids: list[str] = Field(default_factory=list)
    excluded_mid_ids: list[str] = Field(default_factory=list)
    excluded_result_chunk_ids: list[str] = Field(default_factory=list)


class RepairRebindCandidateAudit(ClosedContractModel):
    claim_id: str
    chunk_id: str
    exact_span_match: bool
    support_score: float
    meaningful_overlap_count: int = Field(ge=0)


class RepairCurrentPackageRebindAudit(ClosedContractModel):
    protocol_version: str | None = None
    candidate_count: int = Field(default=0, ge=0)
    candidates: list[RepairRebindCandidateAudit] = Field(default_factory=list)
    preferred_claim_chunk_ids: dict[str, str] = Field(default_factory=dict)
    gray_zone_decision_authority: Literal[False] | None = None
    gray_zone_model_call_count: Literal[0] | None = None
    rebind_input_hash: str | None = None
    verification_attempted: bool | None = None
    verification_deferred_to_caller: bool | None = None
    supported_claim_gain: list[str] = Field(default_factory=list)
    supported_claim_regression: list[str] = Field(default_factory=list)


class RepairExecutionAudit(ClosedContractModel):
    executor_mechanism: str
    layered_search_called: bool
    source_chunk_ids: list[str] = Field(default_factory=list)
    search_audit: ModelAudit | None = None
    current_package_rebind: RepairCurrentPackageRebindAudit | None = None
    gray_zone_model_call_count: Literal[0]
    gray_zone_decision_authority: Literal["deterministic_executor_only"]
    conversation_state_scope_hash: str = Field(min_length=64, max_length=64)
    retrieval_granularity: RetrievalGranularity
    result_top_k: int = Field(ge=1)
    global_top_k_increased: Literal[False]
    answer_regenerated: Literal[False]
    source_agent_operating_envelope_hash: str = Field(min_length=64, max_length=64)
    repaired_agent_operating_envelope_hash: str = Field(min_length=64, max_length=64)
    source_traversal_protocol_hash: str = Field(min_length=64, max_length=64)
    repaired_traversal_protocol_hash: str = Field(min_length=64, max_length=64)
    source_path_distance_threshold_hash: str = Field(min_length=64, max_length=64)
    repaired_path_distance_threshold_hash: str = Field(min_length=64, max_length=64)
    gray_zone_protocol_and_thresholds_frozen: Literal[True]
    candidate_context_package_id: str | None = None
    candidate_retrieval_trace_id: str | None = None
    candidate_semantic_progress_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    candidate_reverted_to_last_valid_package: Literal[True] | None = None
    supported_claim_regression_rejected: list[str] = Field(default_factory=list)
    regression_fail_closed: Literal[True] | None = None


class RepairProgressSpanAudit(ClosedContractModel):
    chunk_id: str
    document_version_id: str
    char_span: list[int]
    raw_span_text_hash: str


class RepairProgressPayloadAudit(ClosedContractModel):
    result_chunk_ids: list[str] = Field(default_factory=list)
    package_chunk_spans: list[RepairProgressSpanAudit] = Field(
        default_factory=list
    )
    covered_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    graph_path_ids: list[str] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)


class RepairProgressAudit(ClosedContractModel):
    protocol_version: str
    payload: RepairProgressPayloadAudit
    progress_hash: str = Field(min_length=64, max_length=64)


class RepairRoundAudit(ClosedContractModel):
    action_type: Literal[
        "repair_missing_citation",
        "repair_concept_gap",
        "repair_bridge_gap",
        "repair_structure_context",
    ]
    protocol_version: str
    repair_round_index: int = Field(ge=0)
    remaining_repair_budget_before: int = Field(ge=1)
    remaining_repair_budget_after: int = Field(ge=0)
    executor_mechanism: str
    action_input_hash: str = Field(min_length=64, max_length=64)
    action_output_hash: str = Field(min_length=64, max_length=64)
    failure_card_hashes: list[str] = Field(default_factory=list)
    before_failure_types: list[str] = Field(default_factory=list)
    after_failure_types: list[str] = Field(default_factory=list)
    before_context_package_id: str
    repaired_context_package_id: str
    before_retrieval_trace_id: str | None = None
    repaired_retrieval_trace_id: str | None = None
    before_progress: RepairProgressAudit
    after_progress: RepairProgressAudit
    before_progress_hash: str = Field(min_length=64, max_length=64)
    after_progress_hash: str = Field(min_length=64, max_length=64)
    made_semantic_progress: bool
    repair_candidate_reverted: bool
    convergence_reason: str
    retrieval_granularity: RetrievalGranularity
    conversation_state_scope_hash: str = Field(min_length=64, max_length=64)
    query_facets_hash: str = Field(min_length=64, max_length=64)
    result_top_k: int = Field(ge=1)
    global_top_k_increased: Literal[False]
    gray_zone_model_call_count: Literal[0]
    gray_zone_decision_authority: Literal["deterministic_executor_only"]
    repair_audit: RepairExecutionAudit
    validated_targets: RepairValidatedTargetsAudit


class FinalGroundedGateRepairAudit(ClosedContractModel):
    action_type: Literal["claim_level_final_grounded_gate"]
    protocol_version: str
    typed_action_control_hash: str = Field(min_length=64, max_length=64)
    grounding_outcome: Literal["grounded_answer", "insufficient_evidence"]
    exact_answer_hash: str = Field(min_length=64, max_length=64)
    claim_grounded_gate: ClaimGroundedGateAudit
    evidence_gap: EvidenceGapAudit
    deterministic_citation_guard: Literal[True]
    gray_zone_model_call_count: Literal[0]


RepairActionAudit = Annotated[
    RepairRoundAudit | FinalGroundedGateRepairAudit,
    Field(discriminator="action_type"),
]


class ProviderPromptCacheAudit(ClosedContractModel):
    protocol_version: str
    api_protocol: Literal["openai", "anthropic"]
    cache_mode: str
    cacheable_system_prompt_present: bool
    cacheable_system_prompt_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    cacheable_system_prompt_utf8_bytes: int | None = Field(
        default=None,
        ge=0,
    )
    provider_response_persisted: Literal[False]


class ProviderUsageAudit(ClosedContractModel):
    protocol_version: str
    api_protocol: Literal["openai", "anthropic"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_hit: bool
    cache_write: bool
    token_accounting_mode: str
    usage_present: bool
    provider_response_persisted: Literal[False]

    @model_validator(mode="after")
    def validate_provider_reported_cache_flags(self) -> "ProviderUsageAudit":
        if self.cache_hit is not bool((self.cache_read_input_tokens or 0) > 0):
            raise ValueError(
                "cache_hit must equal provider-reported cache_read_input_tokens > 0"
            )
        if self.cache_write is not bool(
            (self.cache_creation_input_tokens or 0) > 0
        ):
            raise ValueError(
                "cache_write must equal provider-reported cache_creation_input_tokens > 0"
            )
        return self


class ProviderCallAudit(ClosedContractModel):
    protocol_version: str
    prompt_cache: ProviderPromptCacheAudit
    usage: ProviderUsageAudit
    provider_response_persisted: Literal[False]


class AnswerModelAudit(ModelAudit):
    contract_version: Literal["answer_model_audit_public_v1"] = (
        "answer_model_audit_public_v1"
    )
    model: str | None = None
    external_called: bool | None = None
    fallback_reason: str | None = None
    chat_model: str | None = None
    agent_plan_id: str | None = None
    agent_plan_index: int | None = Field(default=None, ge=0)
    planning_rounds_used: int | None = Field(default=None, ge=0)
    typed_action_control_hash: str | None = None
    evidence_evaluator: EvidenceEvaluatorVerdictAudit | None = None
    context_package_evidence_gate_passed: bool | None = None
    answer_model_called: bool | None = None
    answer_claim_limit: int | None = Field(default=None, ge=1, le=100)
    output_token_budget: int | None = Field(default=None, ge=1, le=32_768)
    output_token_budget_protocol_version: str | None = None
    provider_call: ProviderCallAudit | None = None
    answer_session_id: str | None = None
    citation_verification_pass_rate: float | None = None
    raw_citation_verification_pass_rate: float | None = None
    repair_protocol_version: str | None = None
    repair_round_budget: int | None = Field(default=None, ge=0)
    repair_rounds_used: int | None = Field(default=None, ge=0)
    repair_convergence_reason: str | None = None
    repair_actions: list[RepairActionAudit] = Field(default_factory=list)
    claim_grounded_gate_protocol_version: str | None = None
    claim_grounded_gate: ClaimGroundedGateAudit | None = None
    exact_answer_hash: str | None = Field(default=None, min_length=64, max_length=64)
    evidence_gap: EvidenceGapAudit = Field(default_factory=EvidenceGapAudit)
    citation_guard_applied: bool = False
    unsupported_claims_removed: bool = False
    insufficient_evidence: bool = False
    grounding_outcome: str | None = None
    returned_citation_count: int | None = Field(default=None, ge=0)


class QAResponse(PublicResponseModel):
    contract_version: Literal["qa_public_v1"] = "qa_public_v1"
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    session_id: str | None = None
    run_id: str | None = None
    context_package_id: str | None = None
    retrieval_trace_id: str | None = None
    retrieval_granularity: RetrievalGranularity = "mid"
    used_chunks: list["ContextItem"] = Field(default_factory=list)
    route: AgentRoute | str | None = None
    trace: list["AgentTraceEventPayload"] = Field(default_factory=list)
    degraded_mode: bool = False
    model_audit: AnswerModelAudit = Field(default_factory=AnswerModelAudit)
    answer_model_audit: AnswerModelAudit | None = None
    conversation_state: ConversationStatePayload | None = None


class AgentRequest(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    question: str = Field(min_length=1, max_length=12_000)
    knowledge_base_id: str | None = None
    session_id: str | None = Field(default=None, max_length=36)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=CONVERSATION_CLIENT_HISTORY_MAX_TURNS,
    )
    conversation_state_update: ConversationStateUpdate | None = None
    retrieval_granularity: RetrievalGranularity = "mid"
    route: AgentRoute = "layered_context_graph"
    stream_trace: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_conversation_bounds(self) -> "AgentRequest":
        _validate_conversation_request(
            question=self.question,
            history=self.history,
        )
        return self


class AgentQueryIntentConversationAudit(ClosedContractModel):
    active_user_constraints: ConversationUserConstraints
    task_state: ConversationTaskState
    history_references: list[ConversationHistoryReference] = Field(default_factory=list)
    conversation_text_is_evidence: Literal[False]
    historical_references_are_evidence: Literal[False]
    gray_zone_decision_authority: Literal[False]


class AgentQueryIntentAudit(ClosedContractModel):
    intent: str
    entities: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    needs_graph: bool
    history_turns: int | None = Field(default=None, ge=0)
    suggested_strategy: str | None = None
    conversation_state: AgentQueryIntentConversationAudit | None = None


class AgentActionValidationResult(ClosedContractModel):
    valid: bool
    schema_checked: bool | None = None
    budget_checked: bool | None = None
    target_ids_checked: bool | None = None
    target_scope_checked: bool | None = None
    target_layers: dict[str, list[str]] = Field(default_factory=dict)
    fallback_disabled_checked: bool | None = None
    bridge_protection_checked: bool | None = None
    required_restore_modes: list[str] = Field(default_factory=list)
    required_verification_stage: str | None = None
    inserted_required_action: bool | None = None


class AgentActionValidationAccepted(ClosedContractModel):
    index: int | None = None
    accepted_index: int = Field(ge=0)
    action_type: str
    validation: AgentActionValidationResult


class AgentActionValidationDetail(ClosedContractModel):
    key: str | None = None
    reason: str | None = None
    requested: int | None = None
    limit: int | None = None
    keys: list[str] = Field(default_factory=list)
    target_id: str | None = None
    layers: list[str] = Field(default_factory=list)


class AgentActionValidationRejected(ClosedContractModel):
    index: int | None = None
    action_type: str | None = None
    reason: str
    missing_fields: list[str] = Field(default_factory=list)
    extra_fields: list[str] = Field(default_factory=list)
    forbidden_mentions: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    details: list[AgentActionValidationDetail | str] = Field(default_factory=list)
    requested_start_layer: str | None = None
    retrieval_granularity: RetrievalGranularity | None = None
    input_action_count: int | None = Field(default=None, ge=0)
    max_typed_actions_per_round: int | None = Field(default=None, ge=0)
    rejected_count: int | None = Field(default=None, ge=0)
    required_action_count: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, ge=0)


class AgentTypedActionValidationAudit(ClosedContractModel):
    typed_action_schema_protocol_version: str
    typed_action_schema_protocol_hash: str
    accepted: list[AgentActionValidationAccepted] = Field(default_factory=list)
    rejected: list[AgentActionValidationRejected] = Field(default_factory=list)
    inserted_required_actions: list[str] = Field(default_factory=list)
    fallback_disabled: bool
    required_restore_modes: list[str] = Field(default_factory=list)
    allowed_relation_types: list[str] = Field(default_factory=list)
    required_actions_enforced: bool
    retrieval_granularity: RetrievalGranularity | None = None
    input_action_count: int = Field(ge=0)
    input_scan_limit: int = Field(ge=0)
    valid: bool
    plan_index: int | None = Field(default=None, ge=0)
    retrieval_granularity_locked: RetrievalGranularity | None = None
    unsupported_retrieval_granularity_rewrites_rejected: Literal[True] | None = None


class AgentStopConditionRequestAudit(ClosedContractModel):
    sufficient_evidence: bool | None = None
    required_action_complete: bool | None = None
    all_required_facets_covered: bool | None = None
    independent_support_paths_at_least: int | None = Field(default=None, ge=0)
    citation_verification_passes: bool | None = None
    frontier_empty: bool | None = None
    all_claims_supported: bool | None = None
    no_semantic_progress: bool | None = None


class AgentStopConditionResultAudit(ClosedContractModel):
    sufficient_evidence: bool | None = None
    required_action_complete: bool | None = None
    all_required_facets_covered: bool | None = None
    independent_support_paths_at_least: bool | None = None
    citation_verification_passes: bool | None = None
    frontier_empty: bool | None = None
    all_claims_supported: bool | None = None
    no_semantic_progress: bool | None = None


class AgentStopConditionEvaluationAudit(ClosedContractModel):
    action_id: str
    action_index: int = Field(ge=0)
    action_type: str
    requested: AgentStopConditionRequestAudit
    results: AgentStopConditionResultAudit
    triggered: bool


class AgentActionStopConditionsAudit(ClosedContractModel):
    evaluations: list[AgentStopConditionEvaluationAudit] = Field(default_factory=list)
    triggered_action_indexes: list[int] = Field(default_factory=list)
    stop_triggered: bool
    stop_condition_hash: str


class AgentReplanProgressAudit(ClosedContractModel):
    protocol_version: Literal["agent_replan_semantic_progress_v1"]
    semantic_progress_signature: str | None = None
    evaluator_directive_hash: str | None = None
    matching_prior_plan_indexes: list[int] = Field(default_factory=list)
    no_progress: bool
    phase: Literal["before_retrieval_execution"] | None = None
    plan_index: int | None = Field(default=None, ge=0)
    typed_action_control_hash: str | None = None
    prior_plan_index: int | None = Field(default=None, ge=0)
    reason: Literal[
        "typed_actions_targets_budgets_and_controls_unchanged"
    ] | None = None
    retrieval_execution_count: Literal[0] | None = None
    evidence_evaluator_model_call_count: Literal[0] | None = None
    gray_zone_decision_authority: Literal[False]
    gray_zone_model_call_count: Literal[0]
    audit_hash: str


class AgentTraceScoresBase(ClosedContractModel):
    contract_version: Literal["agent_trace_scores_public_v1"] = (
        "agent_trace_scores_public_v1"
    )


class QueryUnderstandingTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["query_understanding"]
    top_k: int | None = Field(default=None, ge=0)
    query_intent: AgentQueryIntentAudit | None = None
    retrieval_granularity: RetrievalGranularity | None = None


class QueryFacetsTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["query_facets"]
    query_facets: "QueryFacetPacket | None" = None
    retrieval_granularity: RetrievalGranularity | None = None


class PlannerTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["planner"]
    plan_id: str | None = None
    plan_index: int | None = Field(default=None, ge=0)
    replan: bool | None = None
    agent_operating_envelope_hash: str | None = None
    retrieval_granularity: RetrievalGranularity | None = None


class TypedActionValidationTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["typed_action_validation"]
    plan_id: str | None = None
    plan_index: int | None = Field(default=None, ge=0)
    validation: AgentTypedActionValidationAudit | None = None


class TypedActionExecutorTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["typed_action_executor"]
    plan_id: str | None = None
    plan_index: int | None = Field(default=None, ge=0)
    typed_action_control_hash: str | None = None
    effective_result_top_k: int | None = Field(default=None, ge=0)
    retrieval_trace_id: str | None = None


class EvidenceEvaluatorTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["evidence_evaluator"]
    plan_id: str | None = None
    plan_index: int | None = Field(default=None, ge=0)
    verdict: EvidenceEvaluatorVerdictAudit | None = None
    replan_requested: bool | None = None
    replan_candidate: bool | None = None
    replan_no_progress: bool | None = None
    replan_progress: AgentReplanProgressAudit | None = None
    evaluator_requests_replan: bool | None = None
    insufficient_corpus_terminal_deferred: bool | None = None
    action_stop_condition_triggered: bool | None = None
    action_stop_conditions: AgentActionStopConditionsAudit | None = None
    planning_rounds_remaining: int | None = Field(default=None, ge=0)
    gray_zone_model_call_count: Literal[0] | None = None


class ReplanProgressTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["replan_progress"]
    replan_progress: AgentReplanProgressAudit


class EvidenceGateTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["evidence_gate"]
    answer_model_audit: AnswerModelAudit | None = None


class RetrievalStageTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["retrieval_stage"]
    retrieval_trace_id: str | None = None
    retrieval_granularity: RetrievalGranularity | None = None
    coarse_entries: int | None = Field(default=None, ge=0)
    stage_queue_count: int | None = Field(default=None, ge=0)
    mid_topk_selected: int | None = Field(default=None, ge=0)
    chunk_topk_selected: int | None = Field(default=None, ge=0)
    query_rq_path: list[int] = Field(default_factory=list)
    frontier_pops: int | None = Field(default=None, ge=0)
    dominance_pruned_count: int | None = Field(default=None, ge=0)
    chunk_ids: list[str] = Field(default_factory=list)
    gray_zone_model_call_count: Literal[0] | None = None


class LayeredRetrievalTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["layered_retrieval"]
    retrieval_audit: ModelAudit | None = None


class ContextRestorationTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["context_restoration"]
    context_package_id: str | None = None
    hit_chunks: int | None = Field(default=None, ge=0)
    restored_chunks: int | None = Field(default=None, ge=0)
    bridge_chunks: int | None = Field(default=None, ge=0)
    parent_structure_nodes: int | None = Field(default=None, ge=0)
    graph_path_ids: int | None = Field(default=None, ge=0)


class ContextPackageTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["context_package"]
    context_package_id: str | None = None
    token_count: int | None = Field(default=None, ge=0)


class GroundedAnswerTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["grounded_answer"]
    answer_model_audit: AnswerModelAudit | None = None


class RepairTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["repair"]
    repair_action: RepairActionAudit | None = None


class CitationVerificationTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["citation_verification"]
    citation_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_citation_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    verification_count: int | None = Field(default=None, ge=0)
    returned_citation_count: int | None = Field(default=None, ge=0)
    repair_actions: list[RepairActionAudit] = Field(default_factory=list)


class RewardTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["reward"]
    runtime_settings_hash: str | None = None
    agent_operating_envelope_hash: str | None = None


class StatusTraceScores(AgentTraceScoresBase):
    audit_kind: Literal["status"]
    cancel_requested: bool | None = None
    admission_failure: bool | None = None


AgentTraceScores = Annotated[
    QueryUnderstandingTraceScores
    | QueryFacetsTraceScores
    | PlannerTraceScores
    | TypedActionValidationTraceScores
    | TypedActionExecutorTraceScores
    | EvidenceEvaluatorTraceScores
    | ReplanProgressTraceScores
    | EvidenceGateTraceScores
    | RetrievalStageTraceScores
    | LayeredRetrievalTraceScores
    | ContextRestorationTraceScores
    | ContextPackageTraceScores
    | GroundedAnswerTraceScores
    | RepairTraceScores
    | CitationVerificationTraceScores
    | RewardTraceScores
    | StatusTraceScores,
    Field(discriminator="audit_kind"),
]


class AgentTraceEventPayload(ClosedContractModel):
    contract_version: Literal["agent_trace_event_public_v1"] = (
        "agent_trace_event_public_v1"
    )
    type: Literal["trace"] = "trace"
    id: str | None = None
    run_id: str
    sequence_index: int = Field(strict=True, ge=0)
    node: Literal[
        "query_understanding",
        "query_facet_extraction",
        "agent_planner",
        "typed_action_validation",
        "typed_action_executor",
        "evidence_evaluator",
        "replan_no_progress",
        "evidence_gate",
        "entry_selection",
        "layer_drilldown",
        "frontier_traversal",
        "chunk_recall",
        "layered_retrieval",
        "structure_context_restoration",
        "context_package",
        "grounded_answer",
        "citation_verification",
        "repair_executed",
        "reward_event",
        "cancelled",
        "agent_admission",
        "error",
    ]
    status: str
    input_summary: str = ""
    output_summary: str = ""
    document_ids: list[str] = Field(default_factory=list)
    scores: AgentTraceScores
    duration_ms: int = 0
    error: str | None = None
    created_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def bind_scores_to_trace_node(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        node = str(payload.get("node") or "")
        scores = dict(payload.get("scores") or {})
        kind_by_node = {
            "query_understanding": "query_understanding",
            "query_facet_extraction": "query_facets",
            "agent_planner": "planner",
            "typed_action_validation": "typed_action_validation",
            "typed_action_executor": "typed_action_executor",
            "evidence_evaluator": "evidence_evaluator",
            "replan_no_progress": "replan_progress",
            "evidence_gate": "evidence_gate",
            "entry_selection": "retrieval_stage",
            "layer_drilldown": "retrieval_stage",
            "frontier_traversal": "retrieval_stage",
            "chunk_recall": "retrieval_stage",
            "layered_retrieval": "layered_retrieval",
            "structure_context_restoration": "context_restoration",
            "context_package": "context_package",
            "grounded_answer": "grounded_answer",
            "repair_executed": "repair",
            "citation_verification": "citation_verification",
            "reward_event": "reward",
            "cancelled": "status",
            "agent_admission": "status",
            "error": "status",
        }
        audit_kind = kind_by_node.get(node)
        if audit_kind is None:
            return payload
        supplied_audit_kind = scores.get("audit_kind")
        if supplied_audit_kind is not None and supplied_audit_kind != audit_kind:
            raise ValueError(
                f"trace node {node} cannot carry audit_kind={supplied_audit_kind}"
            )
        if audit_kind == "layered_retrieval":
            scores = {"retrieval_audit": scores}
        elif audit_kind in {"grounded_answer", "evidence_gate"}:
            scores = {"answer_model_audit": scores}
        elif audit_kind == "repair":
            scores = {"repair_action": scores}
        elif audit_kind == "replan_progress":
            scores = {"replan_progress": scores}
        scores["audit_kind"] = audit_kind
        payload["scores"] = scores
        return payload

    @model_validator(mode="after")
    def validate_node_score_consistency(self) -> "AgentTraceEventPayload":
        if self.node != "evidence_gate":
            return self
        scores = self.scores
        if not isinstance(scores, EvidenceGateTraceScores):
            raise ValueError("evidence_gate trace must carry evidence_gate scores")
        audit = scores.answer_model_audit
        if audit is None:
            raise ValueError("evidence_gate trace must carry its answer model audit")
        if audit.answer_model_called is not False:
            raise ValueError(
                "evidence_gate trace cannot claim that the answer model was called"
            )
        if self.status == "completed":
            if (
                audit.context_package_evidence_gate_passed is not True
                or audit.evidence_evaluator is None
                or audit.evidence_evaluator.verdict != "sufficient"
            ):
                raise ValueError(
                    "completed evidence_gate trace requires a sufficient evaluator "
                    "verdict and a passed context-package gate"
                )
        elif self.status == "blocked" and (
            audit.context_package_evidence_gate_passed is not False
            or (
                audit.evidence_evaluator is not None
                and audit.evidence_evaluator.verdict == "sufficient"
            )
        ):
            raise ValueError(
                "blocked evidence_gate trace conflicts with its evaluator or gate result"
            )
        return self


class AgentResponse(QAResponse):
    route: AgentRoute | str = "layered_context_graph"
    trace: list[AgentTraceEventPayload] = Field(default_factory=list)


class TaskStatusResponse(PublicResponseModel):
    run_id: str
    session_id: str | None = None
    state: str | None = None
    status: str
    current_node: str | None = None
    retry_count: int | None = None
    route: str | None = None
    retrieval_granularity: RetrievalGranularity = "mid"
    answer: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    trace: list[AgentTraceEventPayload] = Field(default_factory=list)


AgentPEActionType = Literal[
    "activate_coarse_concepts",
    "route_mid_concepts",
    "route_rq_addresses",
    "select_entry_nodes",
    "walk_graph_frontier",
    "drill_down_layer",
    "jump_bridge",
    "stop_and_collect_chunks",
    "need_more_evidence",
    "recall_chunks",
    "restore_context_package",
    "build_context_package",
    "verify_citations",
    "repair_missing_citation",
    "repair_concept_gap",
    "repair_bridge_gap",
    "repair_structure_context",
]
AgentPEObservationType = Literal[
    "plan_validation_failed",
    "executor_contract_blocked",
    "entry_selection",
    "layer_routing",
    "frontier_traversal",
    "chunk_recall",
    "repair_gate",
    "evidence_evaluator",
    "replan_gate",
    "evidence_gate_blocked",
    "context_restoration",
    "context_package_built",
    "citation_verification",
    "typed_repair_round",
    "claim_level_final_grounded_gate",
]


class AgentPEAuditCounts(ClosedContractModel):
    plans: int = Field(strict=True, ge=0)
    actions: int = Field(strict=True, ge=0)
    observations: int = Field(strict=True, ge=0)


class AgentPEAuditOrdering(ClosedContractModel):
    plans: Literal["plan_index ASC, created_at ASC, id ASC"]
    actions: Literal[
        "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC"
    ]
    observations: Literal["created_at ASC, id ASC"]


class AgentPEJsonPayload(ClosedContractModel):
    encoding: Literal["canonical_json_v1"] = "canonical_json_v1"
    canonical_json: str = Field(strict=True)
    sha256: str = Field(
        strict=True,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    redacted_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_canonical_json_and_hash(self) -> "AgentPEJsonPayload":
        def reject_non_finite(value: str) -> None:
            raise ValueError(f"non-finite JSON constant is forbidden: {value}")

        try:
            decoded = json.loads(
                self.canonical_json,
                parse_constant=reject_non_finite,
            )
            canonical = json.dumps(
                decoded,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "canonical_json must be valid finite canonical JSON"
            ) from exc
        if canonical != self.canonical_json:
            raise ValueError(
                "canonical_json does not use canonical_json_v1 serialization"
            )
        expected_hash = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(expected_hash, self.sha256):
            raise ValueError("sha256 does not match canonical_json bytes")
        return self


class AgentPEActionValidatorAudit(ClosedContractModel):
    valid: bool | None = None
    plan_valid: bool | None = None
    schema_checked: bool | None = None
    budget_checked: bool | None = None
    target_ids_checked: bool | None = None
    target_scope_checked: bool | None = None
    typed_action_schema_protocol_version: str | None = None
    typed_action_schema_protocol_hash: str | None = None
    repair_protocol_version: str | None = None
    repair_budget_checked: bool | None = None
    repair_round_index: int | None = Field(default=None, ge=0)
    remaining_repair_budget_before: int | None = Field(default=None, ge=0)
    action_input_hash: str | None = None
    repair_directive_validator_protocol_version: str | None = None
    repair_directive_validator_result: str | None = None
    repair_directive_hash: str | None = None
    validated_directive_hash: str | None = None
    payload: AgentPEJsonPayload


class AgentPEEvaluatorLinkage(ClosedContractModel):
    plan_id: str
    plan_index: int = Field(strict=True, ge=0)
    protocol_version: str | None = None
    verdict: Literal[
        "sufficient",
        "need_more_same_node",
        "need_bridge_jump",
        "need_mid_expansion",
        "need_chunk_expansion",
        "need_structure_closure",
        "insufficient_corpus",
    ]
    decision_hash: str | None = None
    replan_requested: bool
    gray_zone_model_call_count: Literal[0]
    schema_repair_attempted: bool = False


class AgentPERepairLinkage(ClosedContractModel):
    action_id: str
    parent_action_id: str | None = None
    action_type: Literal[
        "repair_missing_citation",
        "repair_concept_gap",
        "repair_bridge_gap",
        "repair_structure_context",
    ]
    repair_protocol_version: str | None = None
    repair_round_index: int = Field(strict=True, ge=0)
    remaining_repair_budget_before: int = Field(strict=True, ge=0)
    remaining_repair_budget_after: int = Field(strict=True, ge=0)
    action_input_hash: str | None = None
    action_output_hash: str | None = None
    before_context_package_id: str | None = None
    repaired_context_package_id: str | None = None
    before_retrieval_trace_id: str | None = None
    repaired_retrieval_trace_id: str | None = None


class AgentPEPlanAuditRow(ClosedContractModel):
    contract_version: Literal["agent_plan_audit_row_v1"] = (
        "agent_plan_audit_row_v1"
    )
    order_index: int = Field(strict=True, ge=0)
    id: str
    run_id: str
    knowledge_base_id: str
    retrieval_trace_id: str | None = None
    plan_index: int = Field(strict=True, ge=0)
    planner_protocol_version: str | None = None
    typed_action_schema_protocol_version: str | None = None
    typed_action_schema_protocol_hash: str | None = None
    typed_action_executor_protocol_version: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    control_hash: str | None = None
    query_intent: AgentPEJsonPayload
    operating_envelope: AgentPEJsonPayload
    typed_actions: AgentPEJsonPayload
    validation: AgentPEJsonPayload
    planner_model_metadata: AgentPEJsonPayload
    status: Literal[
        "validated",
        "invalid",
        "validator_replan_requested",
        "executor_contract_blocked",
        "replan_requested",
        "evidence_sufficient",
        "insufficient_corpus",
        "planning_budget_exhausted",
    ]
    diagnostics: AgentPEJsonPayload
    action_ids: list[str] = Field(default_factory=list)
    action_count: int = Field(strict=True, ge=0)
    redacted_fields: list[str] = Field(default_factory=list)
    created_at: datetime


class AgentPEActionAuditRow(ClosedContractModel):
    contract_version: Literal["agent_action_audit_row_v1"] = (
        "agent_action_audit_row_v1"
    )
    order_index: int = Field(strict=True, ge=0)
    id: str
    run_id: str
    plan_id: str
    plan_index: int = Field(strict=True, ge=0)
    parent_action_id: str | None = None
    action_index: int = Field(strict=True, ge=0)
    action_type: AgentPEActionType
    target_ids: list[str] = Field(default_factory=list)
    reason: str
    budget_request: AgentPEJsonPayload
    expected_evidence: AgentPEJsonPayload
    stop_condition: AgentPEJsonPayload
    validator: AgentPEActionValidatorAudit
    status: Literal[
        "accepted",
        "completed",
        "rejected",
        "deferred",
        "no_progress",
    ]
    input_hash: str | None = None
    output_hash: str | None = None
    control_hash: str | None = None
    output: AgentPEJsonPayload
    diagnostics: AgentPEJsonPayload
    observation_ids: list[str] = Field(default_factory=list)
    observation_count: int = Field(strict=True, ge=0)
    redacted_fields: list[str] = Field(default_factory=list)
    created_at: datetime


class AgentPEObservationAuditRow(ClosedContractModel):
    contract_version: Literal["agent_observation_audit_row_v1"] = (
        "agent_observation_audit_row_v1"
    )
    order_index: int = Field(strict=True, ge=0)
    id: str
    run_id: str
    plan_id: str
    plan_index: int = Field(strict=True, ge=0)
    action_id: str | None = None
    action_index: int | None = Field(default=None, ge=0)
    parent_action_id: str | None = None
    observation_type: AgentPEObservationType
    protocol_version: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    control_hash: str | None = None
    evaluator_linkage: AgentPEEvaluatorLinkage | None = None
    repair_linkage: AgentPERepairLinkage | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    verdict: str
    observation: AgentPEJsonPayload
    diagnostics: AgentPEJsonPayload
    redacted_fields: list[str] = Field(default_factory=list)
    created_at: datetime


class AgentPEAuditResponse(ClosedContractModel):
    contract_version: Literal["agent_pe_audit_public_v1"] = (
        "agent_pe_audit_public_v1"
    )
    run_id: str
    knowledge_base_id: str
    run_status: str
    counts: AgentPEAuditCounts
    ordering: AgentPEAuditOrdering
    plans: list[AgentPEPlanAuditRow] = Field(default_factory=list)
    actions: list[AgentPEActionAuditRow] = Field(default_factory=list)
    observations: list[AgentPEObservationAuditRow] = Field(default_factory=list)
    redaction_protocol_version: Literal[
        "semantic_sensitive_field_key_segments_v1"
    ] = "semantic_sensitive_field_key_segments_v1"
    provider_raw_response_exposed: Literal[False] = False
    credentials_exposed: Literal[False] = False

    @model_validator(mode="after")
    def validate_public_fidelity(self) -> AgentPEAuditResponse:
        expected_counts = {
            "plans": len(self.plans),
            "actions": len(self.actions),
            "observations": len(self.observations),
        }
        if self.counts.model_dump() != expected_counts:
            raise ValueError("P&E public counts do not match returned rows")
        for name, rows in (
            ("plans", self.plans),
            ("actions", self.actions),
            ("observations", self.observations),
        ):
            if [row.order_index for row in rows] != list(range(len(rows))):
                raise ValueError(
                    f"P&E public {name} order_index must be contiguous"
                )

        def instant(value: datetime) -> datetime:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if self.plans != sorted(
            self.plans,
            key=lambda row: (row.plan_index, instant(row.created_at), row.id),
        ):
            raise ValueError("P&E public plans violate canonical ordering")
        if self.actions != sorted(
            self.actions,
            key=lambda row: (
                row.plan_index,
                row.action_index,
                instant(row.created_at),
                row.id,
            ),
        ):
            raise ValueError("P&E public actions violate canonical ordering")
        if self.observations != sorted(
            self.observations,
            key=lambda row: (instant(row.created_at), row.id),
        ):
            raise ValueError(
                "P&E public observations violate canonical ordering"
            )
        for name, rows in (
            ("plans", self.plans),
            ("actions", self.actions),
            ("observations", self.observations),
        ):
            row_ids = [row.id for row in rows]
            if len(set(row_ids)) != len(row_ids):
                raise ValueError(f"P&E public {name} contain duplicate ids")

        if [plan.plan_index for plan in self.plans] != list(
            range(len(self.plans))
        ):
            raise ValueError(
                "P&E public plan_index must be a complete 0..N-1 sequence"
            )
        plan_by_id = {plan.id: plan for plan in self.plans}
        for plan in self.plans:
            if (
                plan.run_id != self.run_id
                or plan.knowledge_base_id != self.knowledge_base_id
                or plan.action_count != len(plan.action_ids)
            ):
                raise ValueError(
                    "P&E public plan scope or action count conflicts"
                )

        action_by_id = {action.id: action for action in self.actions}
        expected_action_ids = [
            action.id
            for plan in self.plans
            for action in sorted(
                (
                    action
                    for action in self.actions
                    if action.plan_id == plan.id
                ),
                key=lambda row: row.action_index,
            )
        ]
        if expected_action_ids != [action.id for action in self.actions]:
            raise ValueError(
                "P&E public actions violate canonical plan/action scope"
            )
        for action in self.actions:
            plan = plan_by_id.get(action.plan_id)
            parent = (
                action_by_id.get(action.parent_action_id)
                if action.parent_action_id is not None
                else None
            )
            if (
                action.run_id != self.run_id
                or plan is None
                or action.plan_index != plan.plan_index
                or action.observation_count != len(action.observation_ids)
                or (
                    action.parent_action_id is not None
                    and (
                        parent is None
                        or parent.plan_id != action.plan_id
                        or parent.action_index >= action.action_index
                    )
                )
            ):
                raise ValueError(
                    "P&E public action scope or observation count conflicts"
                )
        for plan in self.plans:
            action_indexes = [
                action.action_index
                for action in self.actions
                if action.plan_id == plan.id
            ]
            if action_indexes != list(range(len(action_indexes))):
                raise ValueError(
                    "P&E public action_index must be complete within each plan"
                )

        for observation in self.observations:
            plan = plan_by_id.get(observation.plan_id)
            action = (
                action_by_id.get(observation.action_id)
                if observation.action_id is not None
                else None
            )
            if (
                observation.run_id != self.run_id
                or plan is None
                or observation.plan_index != plan.plan_index
                or (
                    observation.action_id is not None
                    and (
                        action is None
                        or action.plan_id != observation.plan_id
                        or action.action_index != observation.action_index
                        or action.parent_action_id
                        != observation.parent_action_id
                    )
                )
            ):
                raise ValueError(
                    "P&E public observation plan/action scope conflicts"
                )

        for plan in self.plans:
            actual_action_ids = [
                action.id
                for action in self.actions
                if action.plan_id == plan.id
            ]
            if plan.action_ids != actual_action_ids:
                raise ValueError(
                    "P&E public plan action ids conflict with canonical rows"
                )
        for action in self.actions:
            actual_observation_ids = [
                observation.id
                for observation in self.observations
                if observation.action_id == action.id
            ]
            if action.observation_ids != actual_observation_ids:
                raise ValueError(
                    "P&E public action observation ids conflict with canonical rows"
                )

        def decoded_payload_object(
            canonical_json: str,
            *,
            field: str,
        ) -> dict[str, object]:
            decoded = json.loads(canonical_json)
            if not isinstance(decoded, dict):
                raise ValueError(f"P&E public {field} must be a JSON object")
            return decoded

        def decoded_lineage_candidates(
            decoded: dict[str, object],
            *,
            field: str,
        ) -> list[object]:
            candidates: list[object] = [
                decoded.get("action_output_hash"),
                decoded.get("observation_hash"),
            ]
            raw_evaluator = decoded.get("evaluator_verdict")
            if raw_evaluator is not None:
                if not isinstance(raw_evaluator, dict):
                    raise ValueError(
                        f"P&E public {field}.evaluator_verdict must be an object"
                    )
                candidates.append(raw_evaluator.get("decision_hash"))
            return candidates

        raw_evaluator_keys = {
            "action_id",
            "action_index",
            "decision_hash",
            "expected_evidence",
            "parent_action_id",
            "plan_id",
            "plan_index",
            "profile_hash",
            "prompt_protocol_hash",
            "protocol_version",
            "reason",
            "schema_repair_attempted",
            "target_ids",
            "verdict",
        }

        def validate_raw_scope(
            raw: dict[str, object],
            observation: AgentPEObservationAuditRow,
            *,
            field: str,
        ) -> None:
            expected = {
                "plan_id": observation.plan_id,
                "plan_index": observation.plan_index,
                "action_id": observation.action_id,
                "action_index": observation.action_index,
                "parent_action_id": observation.parent_action_id,
            }
            for key, expected_value in expected.items():
                if key not in raw:
                    continue
                value = raw[key]
                if key.endswith("_index"):
                    if value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError(
                            f"P&E public {field}.{key} has invalid type"
                        )
                elif value is not None and (
                    not isinstance(value, str) or not value
                ):
                    raise ValueError(
                        f"P&E public {field}.{key} has invalid type"
                    )
                if value != expected_value:
                    raise ValueError(
                        f"P&E public {field}.{key} conflicts with public scope"
                    )

        def validate_raw_evaluator(
            raw: dict[str, object],
            evaluator: AgentPEEvaluatorLinkage,
            observation: AgentPEObservationAuditRow,
            *,
            field: str,
        ) -> None:
            if set(raw) - raw_evaluator_keys:
                raise ValueError(f"P&E public {field} contains unknown fields")
            expected_core = {
                "decision_hash": evaluator.decision_hash,
                "protocol_version": evaluator.protocol_version,
                "verdict": evaluator.verdict,
            }
            if not set(expected_core).intersection(raw):
                raise ValueError(
                    f"P&E public {field} lacks evaluator semantic fields"
                )
            for key, expected_value in expected_core.items():
                if key not in raw:
                    continue
                value = raw.get(key)
                if (
                    not isinstance(value, str)
                    or not value
                    or value != expected_value
                ):
                    raise ValueError(
                        f"P&E public {field}.{key} conflicts with public evaluator"
                    )
            if "reason" in raw and (
                not isinstance(raw["reason"], str)
                or not raw["reason"].strip()
            ):
                raise ValueError(f"P&E public {field}.reason has invalid type")
            target_ids = raw.get("target_ids")
            if "target_ids" in raw and (
                not isinstance(target_ids, list)
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in target_ids
                )
            ):
                raise ValueError(
                    f"P&E public {field}.target_ids has invalid type"
                )
            if "expected_evidence" in raw and not isinstance(
                raw["expected_evidence"],
                dict,
            ):
                raise ValueError(
                    f"P&E public {field}.expected_evidence has invalid type"
                )
            if "schema_repair_attempted" in raw and (
                not isinstance(raw["schema_repair_attempted"], bool)
                or raw["schema_repair_attempted"]
                != evaluator.schema_repair_attempted
            ):
                raise ValueError(
                    f"P&E public {field}.schema_repair_attempted conflicts "
                    "with public evaluator"
                )
            for key in ("profile_hash", "prompt_protocol_hash"):
                if key in raw and (
                    not isinstance(raw[key], str) or not raw[key]
                ):
                    raise ValueError(
                        f"P&E public {field}.{key} has invalid type"
                    )
            validate_raw_scope(raw, observation, field=field)

        for observation in self.observations:
            evaluator = observation.evaluator_linkage
            if observation.observation_type != "evidence_evaluator":
                if evaluator is not None:
                    raise ValueError(
                        "P&E non-evaluator observation carries evaluator linkage"
                    )
                continue
            if evaluator is None:
                raise ValueError(
                    "P&E evidence-evaluator observation lacks evaluator linkage"
                )
            evaluator_plan = plan_by_id.get(observation.plan_id)
            expected_replan = (
                evaluator_plan is not None
                and evaluator_plan.status == "replan_requested"
            )
            if (
                evaluator.plan_id != observation.plan_id
                or evaluator.plan_index != observation.plan_index
                or evaluator.protocol_version != observation.protocol_version
                or evaluator.verdict != observation.verdict
                or evaluator.replan_requested != expected_replan
                or evaluator.gray_zone_model_call_count != 0
            ):
                raise ValueError(
                    "P&E evaluator linkage conflicts with observation/plan"
                )
            expected_hash = evaluator.decision_hash
            if expected_hash != observation.output_hash:
                raise ValueError(
                    "P&E evaluator decision/output hash lineage conflicts"
                )
            observation_payload = decoded_payload_object(
                observation.observation.canonical_json,
                field="evaluator observation payload",
            )
            validate_raw_scope(
                observation_payload,
                observation,
                field="evaluator observation payload",
            )
            raw_protocol = observation_payload.get("protocol_version")
            if "protocol_version" in observation_payload and (
                not isinstance(raw_protocol, str)
                or not raw_protocol
                or raw_protocol != observation.protocol_version
            ):
                raise ValueError(
                    "P&E public evaluator observation protocol conflicts"
                )
            if "bounded_graph_observation" in observation_payload and not isinstance(
                observation_payload["bounded_graph_observation"],
                dict,
            ):
                raise ValueError(
                    "P&E public bounded graph observation must be an object"
                )
            raw_evaluator = observation_payload.get("evaluator_verdict")
            if "evaluator_verdict" in observation_payload:
                if not isinstance(raw_evaluator, dict):
                    raise ValueError(
                        "P&E public evaluator verdict payload must be an object"
                    )
                validate_raw_evaluator(
                    raw_evaluator,
                    evaluator,
                    observation,
                    field="evaluator observation verdict",
                )
            candidates: list[object] = [
                observation.output_hash,
                evaluator.decision_hash,
                *decoded_lineage_candidates(
                    observation_payload,
                    field="evaluator observation payload",
                ),
            ]
            if observation.action_id is not None:
                action = action_by_id.get(observation.action_id)
                if action is None:
                    raise ValueError(
                        "P&E evaluator linked action is not returned"
                    )
                action_payload = decoded_payload_object(
                    action.output.canonical_json,
                    field="evaluator linked action output",
                )
                validate_raw_scope(
                    action_payload,
                    observation,
                    field="evaluator linked action output",
                )
                action_evaluator = action_payload.get("evaluator_verdict")
                if "evaluator_verdict" in action_payload:
                    if not isinstance(action_evaluator, dict):
                        raise ValueError(
                            "P&E public linked action evaluator must be an object"
                        )
                    validate_raw_evaluator(
                        action_evaluator,
                        evaluator,
                        observation,
                        field="evaluator linked action verdict",
                    )
                candidates.extend(
                    [
                        action.output_hash,
                        *decoded_lineage_candidates(
                            action_payload,
                            field="evaluator linked action output",
                        ),
                    ]
                )
            for candidate in candidates:
                if candidate is None:
                    continue
                if (
                    not isinstance(candidate, str)
                    or not candidate
                    or candidate != expected_hash
                ):
                    raise ValueError(
                        "P&E evaluator raw/public hash lineage conflicts"
                    )
        return self


class SessionMessage(ClosedContractModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    run_id: str | None = None
    route: str | None = None
    retrieval_trace_id: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    citation_replay_status: Literal["not_present", "valid", "unavailable"] = (
        "not_present"
    )
    citation_replay_reason: Literal[
        "persisted_citation_contract_mismatch"
    ] | None = None
    source: Literal["client_history"] | None = None


class SessionSummary(APIModel):
    id: str
    knowledge_base_id: str
    title: str | None = None
    last_question: str | None = None
    last_answer: str | None = None
    transcript: list[SessionMessage] = Field(default_factory=list)
    conversation_state: ConversationStatePayload
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SessionMessagesResponse(APIModel):
    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    conversation_state: ConversationStatePayload


class DeleteResponse(APIModel):
    deleted: bool


class KnowledgeBaseCreateRequest(APIModel):
    name: str
    description: str | None = None


class KnowledgeBaseSummary(PublicResponseModel):
    id: str
    name: str
    description: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    active_chunk_count: int = 0
    current_chunk_version: int = 0
    context_graph_state_id: str | None = None
    context_graph_hash: str | None = None
    stale_reason: str | None = None
    source_root: str | None = None
    storage_root: str | None = None
    has_parsed_chunks: bool = False
    can_full_reparse: bool = False
    degraded_mode: bool = False
    active_profile_id: str | None = None
    active_profile_name: str | None = None
    active_profile_hash: str | None = None


class DeleteKnowledgeBaseResponse(APIModel):
    deleted: bool
    knowledge_base_id: str | None = None
    knowledge_base_name: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class UploadReplacementRecoveryHealth(ClosedContractModel):
    protocol_version: Literal["upload_replacement_recovery_health_v1"]
    status: Literal["not_run", "healthy", "degraded"]
    last_run_at: str | None = None
    knowledge_bases: int = Field(ge=0)
    selected: int = Field(ge=0)
    completed: int = Field(ge=0)
    rolled_back: int = Field(ge=0)
    cleanup_pending: int = Field(ge=0)
    manual_review: int = Field(ge=0)
    failed: int = Field(ge=0)
    retryable: bool


class StorageMaintenanceRecoveryHealth(ClosedContractModel):
    protocol_version: Literal["storage_maintenance_recovery_health_v1"]
    status: Literal["not_run", "healthy", "degraded"]
    last_run_at: str | None = None
    selected: int = Field(ge=0)
    completed: int = Field(ge=0)
    pending: int = Field(ge=0)
    cache_pending: int = Field(ge=0)
    external_pending: int = Field(ge=0)
    manual_review: int = Field(ge=0)
    failed: int = Field(ge=0)
    retryable: bool


class KnowledgeBaseFileSummary(ClosedContractModel):
    id: str
    document_id: str | None = None
    source_path: str
    title: str | None = None
    source_type: str
    partition: str | None = None
    status: KnowledgeBaseFileStatus
    job_state: JobState | str | None = None
    batch_id: str | None = None
    error: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    current_version: int = Field(default=0, ge=0)
    active_chunks: int = Field(default=0, ge=0)
    checksum: str | None = None
    chunk_version: int | None = Field(default=None, ge=1)
    language: str | None = None
    language_source: Literal[
        "explicit_metadata",
        "deterministic_detection",
        "unknown",
    ] | None = None
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language_detection_protocol_version: str | None = None
    language_detection_hash: str | None = None
    language_identity_consistent: bool = False
    updated_at: datetime | None = None
    last_ingested_at: datetime | None = None


class KnowledgeBaseTreeNode(APIModel):
    id: str
    label: str
    type: str
    children: list["KnowledgeBaseTreeNode"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionBatchSummary(APIModel):
    batch_id: str
    knowledge_base_id: str
    state: str
    mode: str | None = None
    total_files: int = 0
    processed_files: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    coverage_by_source_type: dict[str, int] = Field(default_factory=dict)
    coverage_by_language: dict[str, int] = Field(default_factory=dict)
    current_file: str | None = None
    current_phase: str | None = None
    cancel_requested: bool = False
    last_error: str | None = None
    cancellation_status: str | None = None
    cancel_failure_reason: str | None = None
    manual_review_required: bool = False
    batch_recovery_id: str | None = None
    batch_recovery_protocol_version: str | None = None
    v_before_batch: int | None = Field(default=None, ge=0)
    parse_committed: bool = False
    parse_commit_boundary: str | None = None
    celery_task_id: str | None = None
    celery_task_name: str | None = None
    batch_task_ids: list[str] = Field(default_factory=list)
    batch_worker_ids: list[str] = Field(default_factory=list)
    worker_id: str | None = None
    heartbeat_at: datetime | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class BatchLogTokenResponse(APIModel):
    batch_id: str
    token: str
    expires_at: datetime


class RefreshResponse(APIModel):
    knowledge_base_id: str
    refreshed_at: datetime


class CleanupStaleDataResponse(APIModel):
    knowledge_base_id: str
    dry_run: bool = False
    stats: dict[str, Any] = Field(default_factory=dict)


class RebuildGraphRequest(APIModel):
    dry_run: bool = True
    layers: list[GraphType] = Field(default_factory=lambda: ["chunk-relation", "mid-concepts", "coarse-concepts", "context-graph"])


class RebuildGraphResponse(APIModel):
    batch_id: str | None = None
    state: str
    mode: str = "four_layer_context_graph"
    affected_documents: int = 0
    dry_run: bool = True
    stats: dict[str, Any] = Field(default_factory=dict)


class EdgeTypeCalibrationParams(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid", strict=True)

    lower_quantile: float = Field(ge=0.0, le=0.25)
    upper_quantile: float = Field(ge=0.75, le=1.0)
    min_span: float = Field(ge=0.01, le=0.5)
    strength_floor: float = Field(ge=0.000001, le=0.25)

    @model_validator(mode="after")
    def validate_quantile_order(self) -> "EdgeTypeCalibrationParams":
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be below upper_quantile")
        return self


class AutoTpeGraphOperatingPointTheta(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid", strict=True)

    graph_operating_point_protocol: Literal[
        "dense_dynamic_knn_bridge_quota_edge_calibration_v2"
    ]
    optimizer: Literal["auto_tpe_lightweight", "auto_tpe_lightweight_or_default"]
    edge_type_calibration_protocol: Literal["type_local_winsorized_minmax_v1"]
    edge_type_calibration_protocol_hash: str = Field(min_length=64, max_length=64)
    calibration_params: EdgeTypeCalibrationParams
    edge_distance_protocol: Literal["edge_distance_log_calibrated_strength_v2"]
    edge_distance_protocol_hash: str = Field(min_length=64, max_length=64)
    rank_score_protocol_version: Literal["channel_percentile_rank_v1"]
    rank_score_protocol_hash: str = Field(min_length=64, max_length=64)
    raw_strength_protocol_version: Literal["dense_relation_raw_strength_v3"]
    raw_strength_protocol_hash: str = Field(min_length=64, max_length=64)
    chunk_node_quality_protocol: Literal["chunk_node_quality_intrinsic_v2"]
    chunk_node_quality_protocol_hash: str = Field(min_length=64, max_length=64)
    out_evidence_mass_protocol: Literal["relation_out_evidence_mass_v2"]
    out_evidence_mass_protocol_hash: str = Field(min_length=64, max_length=64)
    in_acceptance_capacity_protocol: Literal[
        "relation_in_acceptance_capacity_current_scope_v3"
    ]
    in_acceptance_capacity_protocol_hash: str = Field(min_length=64, max_length=64)
    relation_quota_protocol: Literal["dynamic_knn_reverse_quota_signals_v3"]
    relation_quota_protocol_hash: str = Field(min_length=64, max_length=64)
    quota_signal_scale: float = Field(ge=16.0, le=16.0)
    dense_knn_k_min: int = Field(ge=1, le=32)
    dense_knn_k_max: int = Field(ge=1, le=64)
    dense_reverse_b_min_base: int = Field(ge=1, le=32)
    dense_reverse_b_max_base: int = Field(ge=1, le=64)
    dense_reverse_b_min_doc: int = Field(ge=0, le=32)
    dense_reverse_b_max_doc: int = Field(ge=1, le=64)
    dense_reverse_b_min_lang: int = Field(ge=0, le=32)
    dense_reverse_b_max_lang: int = Field(ge=1, le=64)
    dense_min_cosine: float = Field(ge=0.05, le=0.95)
    dense_strong_cosine: float = Field(ge=0.06, le=0.99)
    cross_doc_out_quota_min: int = Field(ge=0, le=32)
    cross_doc_out_quota_max: int = Field(ge=1, le=64)
    cross_doc_min_cosine: float = Field(ge=0.05, le=0.95)
    cross_language_out_quota_min: int = Field(ge=0, le=32)
    cross_language_out_quota_max: int = Field(ge=1, le=64)
    cross_language_min_cosine: float = Field(ge=0.05, le=0.95)

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "AutoTpeGraphOperatingPointTheta":
        for minimum, maximum in (
            (self.dense_knn_k_min, self.dense_knn_k_max),
            (self.dense_reverse_b_min_base, self.dense_reverse_b_max_base),
            (self.dense_reverse_b_min_doc, self.dense_reverse_b_max_doc),
            (self.dense_reverse_b_min_lang, self.dense_reverse_b_max_lang),
            (self.cross_doc_out_quota_min, self.cross_doc_out_quota_max),
            (
                self.cross_language_out_quota_min,
                self.cross_language_out_quota_max,
            ),
        ):
            if minimum > maximum:
                raise ValueError("each minimum quota must not exceed its maximum")
        if self.dense_strong_cosine <= max(
            self.dense_min_cosine,
            self.cross_doc_min_cosine,
            self.cross_language_min_cosine,
        ):
            raise ValueError("dense_strong_cosine must exceed all typed thresholds")
        return self


class AutoTpeTrialSummary(APIModel):
    trial_id: str
    run_id: str
    knowledge_base_id: str
    build_batch_id: str | None = None
    chunk_scope_hash: str
    embedding_model: str
    embedding_text_version: str
    trial_index: int
    status: str
    sampled_theta_json: AutoTpeGraphOperatingPointTheta | None = None
    theta_hash: str | None = None
    tpe_search_space_hash: str | None = None
    edge_distance_protocol: Literal["edge_distance_log_calibrated_strength_v2"] | None = None
    edge_distance_protocol_hash: str | None = None
    edge_type_calibration_protocol: Literal["type_local_winsorized_minmax_v1"] | None = None
    edge_type_calibration_protocol_hash: str | None = None
    calibration_params: EdgeTypeCalibrationParams | None = None
    calibration_params_hash: str | None = None
    edge_type_calibration_config_hash: str | None = None
    sampler_state_hash: str | None = None
    runtime_settings_hash: str
    gate_profile_hash: str
    gate_profile: dict[str, Any]
    candidate_adjacency_hash: str | None = None
    probe_set_hash: str | None = None
    objective_score: float | None = None
    hard_gate: dict[str, Any] = Field(default_factory=dict)
    objective_components: dict[str, Any] = Field(default_factory=dict)
    failure_code: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AutoTpeRunSummary(APIModel):
    run_id: str
    knowledge_base_id: str
    batch_id: str | None = None
    chunk_relation_graph_state_id: str | None = None
    chunk_version: int
    chunk_scope_hash: str | None = None
    graph_operating_point_protocol: str | None = None
    protocol_hash: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    embedding_text_version: str | None = None
    status: str
    trigger_reason: str | None = None
    trial_budget: int = 0
    startup_random_trials: int = 0
    good_quantile_gamma: float | None = None
    probe_query_budget: int = 0
    candidate_pool_size: int = 0
    best_trial_id: str | None = None
    best_objective_score: float | None = None
    selected_theta_hash: str | None = None
    tpe_search_space_hash: str | None = None
    selected_theta: AutoTpeGraphOperatingPointTheta | None = None
    selected_edge_distance_protocol: Literal[
        "edge_distance_log_calibrated_strength_v2"
    ] | None = None
    selected_edge_distance_protocol_hash: str | None = None
    selected_edge_type_calibration_protocol: Literal[
        "type_local_winsorized_minmax_v1"
    ] | None = None
    selected_edge_type_calibration_protocol_hash: str | None = None
    selected_calibration_params: EdgeTypeCalibrationParams | None = None
    selected_calibration_params_hash: str | None = None
    selected_edge_type_calibration_config_hash: str | None = None
    sampler_state_hash: str | None = None
    probe_set_hash: str | None = None
    hard_gate: dict[str, Any] = Field(default_factory=dict)
    objective_components: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None
    failure_code: str | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    runtime_settings_hash: str | None = None
    selected_graph_runtime_settings_hash: str | None = None
    selected_gate_profile_hash: str | None = None
    selected_gate_profile: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    trials: list[AutoTpeTrialSummary] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AutoTpeStatusResponse(APIModel):
    knowledge_base_id: str
    current_chunk_version: int = 0
    enabled: bool = False
    latest_run: AutoTpeRunSummary | None = None


class GraphNodeMetadata(ClosedContractModel):
    rq_path: list[int] = Field(default_factory=list)
    residual_norm: float | None = None
    rq_prefix_key: str | None = None
    rq_level: int | None = None
    rq_path_prefix: list[int] = Field(default_factory=list)
    representative_chunk_ids: list[str] = Field(default_factory=list)
    support_chunk_ids: list[str] = Field(default_factory=list)
    bridge_chunk_ids: list[str] = Field(default_factory=list)
    residual_norm_mean: float | None = None
    residual_norm_max: float | None = None


class GraphNodeWeightDiagnostics(ClosedContractModel):
    protocol_version: str | None = None
    normalization: str | None = None
    normalization_scope: Literal["mid_concept_state", "coarse_concept_state"] | None = None
    normalization_scope_hash: str | None = None
    normalization_pending: bool | None = None
    layer_local_only: bool | None = None
    cross_layer_comparison_allowed: Literal[False] = False
    query_relevance: Literal[False] = False
    model_call_count: Literal[0] = 0
    component_weights: dict[str, float] = Field(default_factory=dict)
    components: dict[str, float] = Field(default_factory=dict)
    formula: str | None = None
    raw_node_weight: float | None = None
    raw_node_weight_distribution: "GraphDistribution | None" = None
    max_raw_node_weight: float | None = None
    node_weight: float | None = None
    support_chunk_count: int | None = Field(default=None, ge=0)
    support_chunk_edge_count: int | None = Field(default=None, ge=0)
    included_mid_concept_count: int | None = Field(default=None, ge=0)
    membership_mass: float | None = Field(default=None, ge=0.0)
    membership_role_mass_distribution: dict[str, float] = Field(default_factory=dict)
    membership_entropy_distribution: "GraphDistribution | None" = None
    internal_edge_count: int | None = Field(default=None, ge=0)
    cross_edge_count: int | None = Field(default=None, ge=0)
    structure_mapping_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    summary_confidence_source: str | None = None
    summary_grounded: bool | None = None
    summary_grounded_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    card_hash: str | None = None


GRAPH_NODE_COARSE_ROLE_FIELDS = (
    "included_mid_concept_ids",
    "boundary_mid_concept_ids",
    "bridge_mid_concept_ids",
    "outlier_mid_concept_ids",
    "low_confidence_mid_concept_ids",
    "all_mid_concept_ids",
)


class GraphNode(ClosedContractModel):
    contract_kind: Literal[
        "structure_node",
        "chunk_node",
        "rq_prefix_node",
        "mid_concept_node",
        "coarse_concept_node",
    ]
    id: str
    label: str
    type: str
    name: str | None = None
    category: str | None = None
    layer: str | None = None
    value: float | int | None = None
    score: float | None = None
    importance_score: float | None = None
    raw_node_weight: float | None = None
    node_weight: float | None = None
    node_weight_normalization_scope: str | None = None
    node_weight_diagnostics: GraphNodeWeightDiagnostics | None = None
    confidence: float | None = None
    support_count: int | None = None
    support_chunk_ids: list[str] = Field(default_factory=list)
    support_active_chunk_ids: list[str] = Field(default_factory=list)
    support_rq_prefix_ids: list[str] = Field(default_factory=list)
    representative_chunk_ids: list[str] = Field(default_factory=list)
    included_mid_concept_ids: list[str] = Field(default_factory=list)
    boundary_mid_concept_ids: list[str] = Field(default_factory=list)
    bridge_mid_concept_ids: list[str] = Field(default_factory=list)
    outlier_mid_concept_ids: list[str] = Field(default_factory=list)
    low_confidence_mid_concept_ids: list[str] = Field(default_factory=list)
    all_mid_concept_ids: list[str] = Field(default_factory=list)
    source_path: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    summary: str | None = None
    snippet: str | None = None
    text: str | None = None
    page_number: int | None = None
    page_range: list[int | None] | tuple[int | None, int | None] | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: GraphNodeMetadata = Field(default_factory=GraphNodeMetadata)

    @model_validator(mode="before")
    @classmethod
    def validate_coarse_role_field_presence(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        contract_kind = value.get("contract_kind")
        present = {
            field_name
            for field_name in GRAPH_NODE_COARSE_ROLE_FIELDS
            if field_name in value
        }
        if contract_kind == "coarse_concept_node":
            missing = [
                field_name
                for field_name in GRAPH_NODE_COARSE_ROLE_FIELDS
                if field_name not in present
            ]
            if missing:
                raise ValueError(
                    "coarse graph node is missing membership-role fields: "
                    + ", ".join(missing)
                )
        elif present:
            raise ValueError(
                "only coarse graph nodes may declare mid-concept role ids"
            )
        return value

    @model_validator(mode="after")
    def validate_kind_type_pair(self) -> "GraphNode":
        expected = {
            "chunk_node": {"chunk"},
            "rq_prefix_node": {"rq_prefix"},
            "mid_concept_node": {"mid_concept"},
            "coarse_concept_node": {"coarse_concept"},
        }
        if self.contract_kind in expected and self.type not in expected[self.contract_kind]:
            raise ValueError("graph node contract_kind does not match its persisted type")
        if self.contract_kind in {"mid_concept_node", "coarse_concept_node"}:
            missing = [
                name
                for name, value in (
                    ("raw_node_weight", self.raw_node_weight),
                    ("node_weight", self.node_weight),
                    (
                        "node_weight_normalization_scope",
                        self.node_weight_normalization_scope,
                    ),
                    ("node_weight_diagnostics", self.node_weight_diagnostics),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "concept graph node is missing weight audit fields: "
                    + ", ".join(missing)
                )
        mid_role_fields = (
            ("included_mid_concept_ids", self.included_mid_concept_ids),
            ("boundary_mid_concept_ids", self.boundary_mid_concept_ids),
            ("bridge_mid_concept_ids", self.bridge_mid_concept_ids),
            ("outlier_mid_concept_ids", self.outlier_mid_concept_ids),
            (
                "low_confidence_mid_concept_ids",
                self.low_confidence_mid_concept_ids,
            ),
        )
        for field_name, values in (
            *mid_role_fields,
            ("all_mid_concept_ids", self.all_mid_concept_ids),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(
                    f"graph node {field_name} contains a blank identifier"
                )
            if len(values) != len(dict.fromkeys(values)):
                raise ValueError(
                    f"graph node {field_name} contains duplicate identifiers"
                )
        if self.contract_kind == "coarse_concept_node":
            expected_all_mid_concept_ids = list(
                dict.fromkeys(
                    value
                    for _field_name, values in mid_role_fields
                    for value in values
                )
            )
            if self.all_mid_concept_ids != expected_all_mid_concept_ids:
                raise ValueError(
                    "coarse graph node all_mid_concept_ids does not "
                    "replay its persisted membership-role union"
                )
        elif any(values for _field_name, values in mid_role_fields) or (
            self.all_mid_concept_ids
        ):
            raise ValueError(
                "only coarse graph nodes may carry mid-concept role ids"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_coarse_role_fields(
        self,
        handler: Any,
    ) -> dict[str, Any]:
        payload = handler(self)
        if self.contract_kind != "coarse_concept_node":
            for field_name in GRAPH_NODE_COARSE_ROLE_FIELDS:
                payload.pop(field_name, None)
        return payload


class ProjectionCalibrationParams(ClosedContractModel):
    lower_quantile: float = Field(ge=0.0, le=1.0)
    upper_quantile: float = Field(ge=0.0, le=1.0)
    min_span: float = Field(gt=0.0)
    strength_floor: float = Field(gt=0.0, le=1.0)


class ProjectionNormalizationAudit(ClosedContractModel):
    normalization: Literal["layer_edge_type_winsorized_minmax_v1"]
    protocol_version: Literal["layer_edge_type_winsorized_minmax_v1"]
    edge_projection_protocol_version: Literal[
        "membership_q15_layer_type_calibrated_v3"
    ]
    edge_projection_protocol_hash: str = Field(min_length=64, max_length=64)
    layer: Literal["mid", "coarse"]
    edge_type: str
    scope: Literal["layer_plus_edge_type"]
    params: ProjectionCalibrationParams
    sample_count: int = Field(ge=0)
    lower_quantile_value: float
    upper_quantile_value: float
    quantile_span: float = Field(ge=0.0)
    fallback: str | None = None
    calibration_applied: bool
    raw_strength_distribution: "GraphDistribution"
    calibrated_strength_distribution: "GraphDistribution"
    calibrated_distance_distribution: "GraphDistribution"
    cross_type_raw_comparison_allowed: Literal[False]
    model_call_count: Literal[0]
    stats_hash: str = Field(min_length=64, max_length=64)
    support_edge_count: int = Field(ge=0)
    support_mid_edge_count: int = Field(default=0, ge=0)
    support_chunk_edge_count: int = Field(default=0, ge=0)
    support_membership_mass: float = Field(default=0.0, ge=0.0)


class GraphProjectionBottomEdgeTypeMass(ClosedContractModel):
    dense_semantic: float = Field(default=0.0, ge=0.0)
    dense_cross_document_bridge: float = Field(default=0.0, ge=0.0)
    dense_cross_language_bridge: float = Field(default=0.0, ge=0.0)


class GraphProjectionSupportEdgeTypeCounts(ClosedContractModel):
    dense_semantic: int = Field(default=0, ge=0)
    dense_cross_document_bridge: int = Field(default=0, ge=0)
    dense_cross_language_bridge: int = Field(default=0, ge=0)


class GraphProjectionRawStrengthSummary(ClosedContractModel):
    aggregation_protocol_version: Literal[
        "membership_weighted_bottom_support_q15_log_mass_v1"
    ]
    q15_bottom_distance: float = Field(ge=0.0)
    support_membership_mass: float = Field(gt=0.0)
    support_mid_edge_count: int = Field(default=0, ge=0)
    support_chunk_edge_count: int = Field(ge=1)
    bottom_distance_distribution: "GraphDistribution"
    membership_product_distribution: "GraphDistribution"
    dominant_bottom_edge_type: Literal[
        "dense_semantic",
        "dense_cross_document_bridge",
        "dense_cross_language_bridge",
    ]
    bottom_edge_type_membership_mass: GraphProjectionBottomEdgeTypeMass
    contribution_facts_hash: str = Field(min_length=64, max_length=64)
    edge_distance_protocol: Literal["edge_distance_log_calibrated_strength_v2"]


class GraphProjectionSupportContribution(ClosedContractModel):
    bottom_chunk_edge_id: str
    source_chunk_id: str
    target_chunk_id: str
    bottom_edge_type: Literal[
        "dense_semantic",
        "dense_cross_document_bridge",
        "dense_cross_language_bridge",
    ]
    bottom_distance: float = Field(ge=0.0)
    source_membership_score: float = Field(ge=0.0, le=1.0)
    target_membership_score: float = Field(ge=0.0, le=1.0)
    membership_product: float = Field(gt=0.0, le=1.0)
    orientation: Literal["source_scope_to_target_scope"]
    assignment_protocol_version: Literal[
        "scope_key_chunk_business_assignment_v1"
    ]
    bottom_edge_fact_hash: str = Field(min_length=64, max_length=64)


class GraphProjectionGrayPredicates(ClosedContractModel):
    protocol_version: Literal["projected_gray_predicates_support_rollup_v1"]
    protocol_hash: str = Field(min_length=64, max_length=64)
    semantic_uncertain: bool
    crossing_rq_boundary: bool
    support_edge_count: int = Field(ge=1)
    semantic_uncertain_support_count: int = Field(ge=0)
    rq_boundary_support_count: int = Field(ge=0)
    semantic_uncertain_support_edge_ids: list[str] = Field(default_factory=list)
    rq_boundary_support_edge_ids: list[str] = Field(default_factory=list)
    semantic_uncertainty_rollup: Literal[
        "all_bottom_support_edges_uncertain"
    ]
    rq_boundary_rollup: Literal[
        "any_bottom_support_edge_crosses_rq_leaf_path"
    ]
    model_call_count: Literal[0]


class GraphI18nTextMap(ClosedContractModel):
    zh: str
    en: str


class GraphI18nSearchTermsMap(ClosedContractModel):
    zh: list[str] = Field(default_factory=list)
    en: list[str] = Field(default_factory=list)


class GraphEdgeI18n(ClosedContractModel):
    id: str | None = None
    layer: Literal["mid", "coarse"]
    protocol_version: Literal["concept_i18n_bilingual_v1"]
    status: Literal["ok", "original_text_fallback"]
    relation_label_i18n: GraphI18nTextMap
    explanation_i18n: GraphI18nTextMap
    summary_i18n: GraphI18nTextMap
    search_terms_i18n: GraphI18nSearchTermsMap
    fallback_source: Literal["original_text_fallback"] | None = None

    @model_validator(mode="after")
    def validate_fallback_identity(self) -> "GraphEdgeI18n":
        expected = (
            None
            if self.status == "ok"
            else "original_text_fallback"
        )
        if self.fallback_source != expected:
            raise ValueError(
                "edge i18n fallback_source must match the closed status"
            )
        return self


class GraphProjectionEdgeDiagnostics(ClosedContractModel):
    edge_projection_protocol: Literal[
        "membership_q15_layer_type_calibrated_v3"
    ]
    edge_projection_protocol_hash: str = Field(min_length=64, max_length=64)
    aggregation_protocol_version: Literal[
        "membership_weighted_bottom_support_q15_log_mass_v1"
    ]
    calibration_protocol_version: Literal[
        "layer_edge_type_winsorized_minmax_v1"
    ]
    source_algorithm: Literal["membership_weighted_bottom_edge_projection"]
    support_rq_prefix_ids: list[str] = Field(default_factory=list)
    support_mid_edge_count: int = Field(default=0, ge=0)
    support_chunk_edge_count: int = Field(ge=1)
    support_contribution_count: int = Field(ge=1)
    support_membership_mass: float = Field(gt=0.0)
    support_contributions: list[GraphProjectionSupportContribution] = Field(
        min_length=1
    )
    support_contributions_complete: bool = True
    contribution_facts_hash: str = Field(min_length=64, max_length=64)
    dominant_bottom_edge_type: Literal[
        "dense_semantic",
        "dense_cross_document_bridge",
        "dense_cross_language_bridge",
    ]
    support_edge_types: GraphProjectionSupportEdgeTypeCounts
    semantic_uncertain: bool
    crossing_rq_boundary: bool
    gray_predicates: GraphProjectionGrayPredicates
    gray_zone_semantics_changed: Literal[False]
    model_call_count: Literal[0]
    edge_i18n: GraphEdgeI18n | None = None

    @model_validator(mode="after")
    def validate_replay_counts(self) -> "GraphProjectionEdgeDiagnostics":
        sampled_edge_count = len(
            {item.bottom_chunk_edge_id for item in self.support_contributions}
        )
        if self.support_contributions_complete:
            if self.support_contribution_count != len(self.support_contributions):
                raise ValueError(
                    "projection support_contribution_count must match complete contribution cards"
                )
            if self.support_chunk_edge_count != sampled_edge_count:
                raise ValueError(
                    "projection support_chunk_edge_count must match complete unique bottom edges"
                )
        elif (
            self.support_contribution_count < len(self.support_contributions)
            or self.support_chunk_edge_count < sampled_edge_count
        ):
            raise ValueError(
                "projection overview sample cannot exceed its full support counts"
            )
        if self.gray_predicates.support_edge_count != self.support_chunk_edge_count:
            raise ValueError("projection gray rollup must cover every bottom edge")
        return self


class GraphEdgeMetadata(ClosedContractModel):
    source_algorithm: str | None = None
    protocol_version: str | None = None
    graph_state_hash: str | None = None
    is_cross_document: bool | None = None
    is_cross_language: bool | None = None
    semantic_uncertain: bool | None = None
    crossing_rq_boundary: bool | None = None
    candidate_channels: list[str] = Field(default_factory=list)
    diagnostic_only: bool | None = None
    active_relation_edge: bool | None = None
    membership_role: str | None = None
    membership_entropy: float | None = None
    membership_rank: int | None = None
    rq_path: list[int] = Field(default_factory=list)
    residual_norm: float | None = None
    diagnostic_strength: float | None = None
    support_membership_mass: float | None = None
    support_chunk_ids_sample: list[str] = Field(default_factory=list)
    support_chunk_edge_ids_sample: list[str] = Field(default_factory=list)
    support_chunk_edge_ids: list[str] = Field(default_factory=list)
    support_chunk_edge_count: int | None = Field(default=None, ge=0)
    support_chunk_edge_ids_hash: str | None = None
    protocol_hash: str | None = None
    diagnostic_hash: str | None = None
    model_call_count: Literal[0] | None = None


class GraphEdge(ClosedContractModel):
    contract_kind: Literal[
        "structure_edge",
        "chunk_relation_edge",
        "rq_membership_edge",
        "rq_diagnostic_edge",
        "concept_projection_edge",
    ]
    id: str | None = None
    source: str
    target: str
    label: str | None = None
    type: str
    category: str | None = None
    weight: float | None = None
    confidence: float | None = None
    distance: float | None = None
    raw_strength: float | None = None
    projected_distance_raw: float | None = None
    projected_strength_raw: float | None = None
    raw_strength_summary: GraphProjectionRawStrengthSummary | None = None
    edge_projection_protocol_hash: str | None = None
    projection_normalization: ProjectionNormalizationAudit | None = None
    diagnostics: GraphProjectionEdgeDiagnostics | None = None
    source_algorithm: str | None = None
    protocol_version: str | None = None
    state_hash: str | None = None
    score: float | None = None
    support_count: int | None = None
    support_chunk_ids: list[str] = Field(default_factory=list)
    support_chunk_edge_ids: list[str] = Field(default_factory=list)
    support_mid_edge_ids: list[str] = Field(default_factory=list)
    support_mid_concept_ids: list[str] = Field(default_factory=list)
    support_rq_prefix_ids: list[str] = Field(default_factory=list)
    relation_source: str | None = None
    is_bridge: bool | None = None
    is_inferred: bool | None = None
    metadata: GraphEdgeMetadata = Field(default_factory=GraphEdgeMetadata)

    @model_validator(mode="after")
    def validate_projection_audit(self) -> "GraphEdge":
        if self.contract_kind == "rq_diagnostic_edge":
            if self.type != "rq_prefix_pair_diagnostic":
                raise ValueError(
                    "RQ diagnostic edge contract_kind does not match its type"
                )
            if self.metadata.diagnostic_only is not True:
                raise ValueError("RQ diagnostic edge must remain diagnostic-only")
            if self.metadata.active_relation_edge is not False:
                raise ValueError("RQ diagnostic edge cannot become an active relation edge")
            if self.metadata.model_call_count != 0:
                raise ValueError("RQ diagnostic edge must prove zero model calls")
            if self.metadata.support_chunk_edge_ids != self.support_chunk_edge_ids:
                raise ValueError(
                    "RQ diagnostic metadata must replay bounded support edge ids"
                )
            if (
                self.metadata.support_chunk_edge_ids_sample
                != self.metadata.support_chunk_edge_ids
            ):
                raise ValueError(
                    "RQ diagnostic support edge sample aliases must match"
                )
            if (
                self.metadata.support_chunk_edge_count is None
                or self.metadata.support_chunk_edge_count
                < len(self.metadata.support_chunk_edge_ids)
            ):
                raise ValueError(
                    "RQ diagnostic support edge count cannot understate its sample"
                )
            support_ids = self.metadata.support_chunk_edge_ids
            if any(not edge_id.strip() for edge_id in support_ids):
                raise ValueError("RQ diagnostic support edge ids cannot be blank")
            if len(support_ids) != len(dict.fromkeys(support_ids)):
                raise ValueError("RQ diagnostic support edge ids must be unique")
            if not self.metadata.support_chunk_edge_ids_hash or len(
                self.metadata.support_chunk_edge_ids_hash
            ) != 64:
                raise ValueError("RQ diagnostic support edge hash is required")
            if not self.metadata.protocol_hash or len(self.metadata.protocol_hash) != 64:
                raise ValueError("RQ diagnostic protocol hash is required")
            if not self.metadata.diagnostic_hash or len(self.metadata.diagnostic_hash) != 64:
                raise ValueError("RQ diagnostic fact hash is required")
            return self
        if self.contract_kind != "concept_projection_edge":
            return self
        required = {
            "projected_distance_raw": self.projected_distance_raw,
            "projected_strength_raw": self.projected_strength_raw,
            "raw_strength_summary": self.raw_strength_summary,
            "edge_projection_protocol_hash": self.edge_projection_protocol_hash,
            "projection_normalization": self.projection_normalization,
            "diagnostics": self.diagnostics,
            "source_algorithm": self.source_algorithm,
            "protocol_version": self.protocol_version,
            "state_hash": self.state_hash,
        }
        missing = [name for name, value in required.items() if value is None or value == ""]
        if missing:
            raise ValueError(
                "concept projection edge is missing public audit fields: "
                + ", ".join(missing)
            )
        if self.protocol_version != "membership_q15_layer_type_calibrated_v3":
            raise ValueError("concept projection edge uses an inactive protocol version")
        if not self.support_chunk_edge_ids:
            raise ValueError("concept projection edge requires complete bottom-edge support ids")
        if self.raw_strength_summary is None or self.diagnostics is None:
            raise ValueError("concept projection replay audit is required")
        support_ids = [
            contribution.bottom_chunk_edge_id
            for contribution in self.diagnostics.support_contributions
        ]
        if support_ids != self.support_chunk_edge_ids:
            raise ValueError(
                "projection contribution cards must replay ordered support_chunk_edge_ids"
            )
        if self.diagnostics.support_contributions_complete:
            if self.raw_strength_summary.support_chunk_edge_count != len(support_ids):
                raise ValueError("projection raw-strength support count mismatch")
        else:
            full_support_count = self.diagnostics.support_chunk_edge_count
            if (
                self.raw_strength_summary.support_chunk_edge_count
                != full_support_count
                or self.metadata.support_chunk_edge_count != full_support_count
                or not self.metadata.support_chunk_edge_ids_hash
                or len(self.metadata.support_chunk_edge_ids_hash) != 64
                or self.metadata.support_chunk_edge_ids != support_ids
                or self.metadata.support_chunk_edge_ids_sample != support_ids
            ):
                raise ValueError(
                    "projection overview sample is missing its full support count/hash proof"
                )
        if self.raw_strength_summary.support_mid_edge_count != len(
            self.support_mid_edge_ids
        ):
            raise ValueError("projection raw-strength mid-edge support count mismatch")
        if self.diagnostics.support_mid_edge_count != len(self.support_mid_edge_ids):
            raise ValueError("projection diagnostics mid-edge support count mismatch")
        if self.diagnostics.support_rq_prefix_ids != self.support_rq_prefix_ids:
            raise ValueError("projection diagnostics RQ-prefix support mismatch")
        if (
            self.raw_strength_summary.contribution_facts_hash
            != self.diagnostics.contribution_facts_hash
        ):
            raise ValueError("projection contribution facts hash mismatch")
        if not math.isclose(
            self.raw_strength_summary.support_membership_mass,
            self.diagnostics.support_membership_mass,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("projection support membership mass mismatch")
        if self.diagnostics.edge_projection_protocol_hash != self.edge_projection_protocol_hash:
            raise ValueError("projection diagnostics protocol hash mismatch")
        if self.diagnostics.edge_projection_protocol != self.protocol_version:
            raise ValueError("projection diagnostics protocol version mismatch")
        if self.diagnostics.source_algorithm != self.source_algorithm:
            raise ValueError("projection diagnostics source algorithm mismatch")
        if (
            self.diagnostics.semantic_uncertain
            != self.diagnostics.gray_predicates.semantic_uncertain
            or self.diagnostics.crossing_rq_boundary
            != self.diagnostics.gray_predicates.crossing_rq_boundary
        ):
            raise ValueError("projection gray predicate rollup mismatch")
        return self


class GraphCounts(ClosedContractModel):
    chunks: int = 0
    active_chunks: int = 0
    structure_nodes: int = 0
    structure_edges: int = 0
    structure_mappings: int = 0
    chunk_relation_edges: int = 0
    rq_prefixes: int = 0
    rq_prefix_memberships: int = 0
    rq_prefix_pair_diagnostics: int = 0
    rq_relation_edges: int = 0
    mid_concepts: int = 0
    mid_concept_edges: int = 0
    mid_concept_memberships: int = 0
    coarse_concepts: int = 0
    coarse_concept_edges: int = 0
    coarse_concept_memberships: int = 0


class GraphLayerCounts(ClosedContractModel):
    nodes: int = Field(ge=0)
    edges: int = Field(ge=0)


class GraphSampleCountCard(ClosedContractModel):
    sampled: int = Field(ge=0)
    full: int = Field(ge=0)


class GraphFreshnessLayerRow(ClosedContractModel):
    layer: str
    state_hash: str | None = None
    is_stale: bool
    stale_reasons: list[str] = Field(default_factory=list)
    checked_at: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class GraphFreshness(ClosedContractModel):
    protocol_version: str | None = None
    admission_protocol_version: str | None = None
    is_stale: bool
    is_admissible: bool | None = None
    stale_reasons: list[str] = Field(default_factory=list)
    admission_reasons: list[str] = Field(default_factory=list)
    current_chunk_scope_hash: str | None = None
    current_contextual_index_hash: str | None = None
    stored_contextual_index_hash: str | None = None
    current_contextual_index_business_hash: str | None = None
    stored_contextual_index_business_hash: str | None = None
    current_chunk_business_scope_hash: str | None = None
    stored_chunk_business_scope_hash: str | None = None
    canonical_state_hash_protocol_version: str | None = None
    canonical_state_validation_reasons: list[str] = Field(default_factory=list)
    checked_at: str | None = None
    layer_rows: list[GraphFreshnessLayerRow] = Field(default_factory=list)
    model_call_count: Literal[0] = 0
    gray_zone_rule_inputs_modified: Literal[False] = False
    local_hint_protocol_version: str
    context_graph_state_id: str | None = None
    context_graph_hash: str | None = None


class GraphHashes(ClosedContractModel):
    chunk_scope_hash: str | None = None
    chunk_business_scope_hash: str | None = None
    contextual_index_hash: str | None = None
    contextual_index_business_hash: str | None = None
    local_hint_protocol_version: str
    structure_graph_hash: str | None = None
    chunk_relation_graph_hash: str | None = None
    rq_membership_hash: str | None = None
    rq_prefix_pair_aggregate_hash: str | None = None
    rq_prefix_pair_diagnostics_hash: str | None = None
    mid_concept_hash: str | None = None
    coarse_concept_hash: str | None = None
    context_graph_hash: str | None = None
    runtime_settings_hash: str
    agent_operating_envelope_hash: str
    policy_state_hash: str | None = None
    prompt_protocol_hash: str | None = None


class GraphGrounding(ClosedContractModel):
    mid_grounded_rate: float = Field(ge=0.0, le=1.0)
    mid_total: int = Field(ge=0)
    coarse_grounded_rate: float = Field(ge=0.0, le=1.0)
    coarse_total: int = Field(ge=0)


class GraphRetrievalContribution(ClosedContractModel):
    trace_count: int = Field(ge=0)
    has_observations: bool
    frontier_pops: int = Field(ge=0)
    dominance_pruned_count: int = Field(ge=0)
    expanded_edge_contribution: dict[str, float] = Field(default_factory=dict)
    convergence_reasons: dict[str, int] = Field(default_factory=dict)
    scores_json_primary: Literal[False]


class GraphDistribution(ClosedContractModel):
    count: int = Field(ge=0)
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    population_std: float | None = None


class GraphEdgeTypeDistribution(ClosedContractModel):
    raw_strength: GraphDistribution | None = None
    calibrated_strength: GraphDistribution | None = None
    distance: GraphDistribution | None = None
    calibration_stats_hashes: list[str] = Field(default_factory=list)
    calibration_stats_hash_consistent: bool | None = None


class GraphEdgeCalibration(ClosedContractModel):
    protocol_version: str
    protocol_hash: str
    stats_by_edge_type: dict[str, "GraphEdgeCalibrationTypeStats"] = Field(default_factory=dict)
    all_stats_hashes_consistent: bool
    cross_type_raw_comparison_allowed: Literal[False]


class GraphEdgeCalibrationParams(ClosedContractModel):
    lower_quantile: float
    upper_quantile: float
    min_span: float
    strength_floor: float


class GraphEdgeCalibrationTypeStats(ClosedContractModel):
    edge_type: str
    protocol_version: str
    protocol_hash: str
    calibration_params_hash: str
    edge_type_calibration_config_hash: str
    stats_hash: str
    params: GraphEdgeCalibrationParams
    sample_count: int = Field(ge=0)
    lower_quantile_value: float | None = None
    upper_quantile_value: float | None = None
    effective_lower_bound: float | None = None
    effective_upper_bound: float | None = None
    quantile_span: float | None = None
    fallback: str | None = None
    calibration_applied: bool
    monotonic_violation_count: int = Field(ge=0)
    raw_strength_distribution: GraphDistribution
    calibrated_strength_distribution: GraphDistribution
    distance_distribution: GraphDistribution
    cross_type_raw_comparison_allowed: Literal[False]


class GraphEdgeDistanceDiagnostics(ClosedContractModel):
    applicable: bool
    reason: str | None = None
    protocol_version: str | None = None
    protocol_hash: str | None = None
    protocol_hashes: list[str] = Field(default_factory=list)
    distribution: GraphDistribution | None = None
    by_edge_type: dict[str, GraphEdgeTypeDistribution] = Field(default_factory=dict)
    calibration: GraphEdgeCalibration | None = None


class GraphProjectionGroupDiagnostics(ClosedContractModel):
    protocol_hashes: list[str] = Field(default_factory=list)
    protocol_hash_edge_count: int = Field(ge=0)
    protocol_hash_coverage: float = Field(ge=0.0, le=1.0)
    protocol_hash_consistent: bool
    source_algorithm_coverage: float = Field(ge=0.0, le=1.0)
    protocol_version_coverage: float = Field(ge=0.0, le=1.0)
    state_hash_coverage: float = Field(ge=0.0, le=1.0)
    full_edge_count: int = Field(ge=0)
    supported_edge_count: int = Field(ge=0)
    support_coverage: float = Field(ge=0.0, le=1.0)
    raw_projected_distance_coverage: float = Field(ge=0.0, le=1.0)
    calibrated_projected_distance_coverage: float = Field(ge=0.0, le=1.0)
    raw_projected_strength_coverage: float = Field(ge=0.0, le=1.0)
    raw_projected_distance_distribution: GraphDistribution
    calibrated_projected_distance_distribution: GraphDistribution
    raw_projected_strength_distribution: GraphDistribution
    support_membership_mass_distribution: GraphDistribution
    support_contribution_count_distribution: GraphDistribution
    normalization_protocol_counts: dict[str, int] = Field(default_factory=dict)
    normalization_stats_hashes: list[str] = Field(default_factory=list)


class GraphProjectedGrayPredicates(ClosedContractModel):
    protocol_version: str
    protocol_hash: str
    coverage: float = Field(ge=0.0, le=1.0)
    missing_edge_count: int = Field(ge=0)
    semantic_uncertain_edge_count: int = Field(ge=0)
    crossing_rq_boundary_edge_count: int = Field(ge=0)
    model_call_count: Literal[0]


class GraphProjectionDiagnostics(GraphProjectionGroupDiagnostics):
    applicable: Literal[True]
    protocol_version: str
    by_edge_type: dict[str, GraphProjectionGroupDiagnostics] = Field(default_factory=dict)
    gray_predicates: GraphProjectedGrayPredicates
    graph_total_edge_count: int = Field(ge=0)
    non_projection_edge_count: int = Field(ge=0)


class GraphProjectionNotApplicable(ClosedContractModel):
    applicable: Literal[False]
    reason: str


class GraphDiagnostics(ClosedContractModel):
    layer_full_counts: GraphLayerCounts
    sampled_counts: GraphLayerCounts
    layer_hash_key: str
    layer_hash: str | None = None


class GraphResponse(ClosedContractModel):
    contract_version: Literal["four_layer_graph_public_v1"]
    knowledge_base_id: str
    graph_type: Literal["chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"]
    schema_version: Literal["context_graph_v1"]
    view: Literal["overview"]
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    counts: GraphCounts
    full_counts: GraphLayerCounts
    sampled_counts: GraphLayerCounts
    node_counts: GraphSampleCountCard
    edge_counts: GraphSampleCountCard
    freshness: GraphFreshness
    hash: str | None = None
    hashes: GraphHashes
    stale_reason: str | None = None
    grounding: GraphGrounding
    retrieval_contribution: GraphRetrievalContribution
    edge_distance_diagnostics: GraphEdgeDistanceDiagnostics
    projection_diagnostics: GraphProjectionDiagnostics | GraphProjectionNotApplicable
    diagnostics: GraphDiagnostics


class DashboardSnapshot(APIModel):
    knowledge_base: KnowledgeBaseSummary | dict[str, Any]
    tree: list[KnowledgeBaseTreeNode] | list[dict[str, Any]] = Field(default_factory=list)
    graph: GraphResponse | dict[str, Any] = Field(default_factory=dict)
    context_graph: dict[str, Any] = Field(default_factory=dict)
    recent_batches: list[IngestionBatchSummary] | list[dict[str, Any]] = Field(default_factory=list)
    last_refreshed_at: datetime | None = None


class RetrievalEntryTopologyPrior(ClosedContractModel):
    protocol_version: str
    protocol_hash: str = Field(min_length=64, max_length=64)
    centrality: float = Field(ge=0.0, le=1.0)
    betweenness: float = Field(ge=0.0, le=1.0)
    betweenness_mode: str
    k_core: int = Field(ge=0)
    k_core_normalized: float = Field(ge=0.0, le=1.0)
    pagerank_or_closeness: float = Field(ge=0.0, le=1.0)
    incident_edge_count: int = Field(ge=0)
    topology_admission_eligible: bool
    bridge_score: float = Field(ge=0.0, le=1.0)
    boundary_score: float = Field(ge=0.0, le=1.0)
    bridge_role: bool
    boundary_role: bool


class RetrievalEntryDenseSupportFact(ClosedContractModel):
    chunk_id: str
    dense_score: float = Field(gt=0.0, le=1.0)
    active_contextual_vector_business_fact_hash: str = Field(
        min_length=64,
        max_length=64,
    )


class RetrievalEntryReplayProof(ClosedContractModel):
    protocol_version: str
    node_id: str
    layer: Literal["coarse", "mid"]
    support_chunk_ids: list[str] = Field(default_factory=list)
    support_count: int = Field(ge=0)
    semantic_dense_support_facts: list[RetrievalEntryDenseSupportFact] = Field(
        default_factory=list,
        max_length=4,
    )
    semantic_support_chunk_ids: list[str] = Field(default_factory=list)
    semantic_support_match_count: int = Field(ge=0, le=4)
    semantic_score: float = Field(ge=0.0, le=1.0)
    semantic_candidate: bool
    semantic_input_hash: str = Field(min_length=64, max_length=64)
    query_facet_packet_hash: str = Field(min_length=64, max_length=64)
    active_topology_state_identity_hash: str = Field(min_length=64, max_length=64)
    topology_state_hash: str = Field(min_length=64, max_length=64)
    topology_node_business_fact_hash: str = Field(min_length=64, max_length=64)
    topology_prior_hash: str = Field(min_length=64, max_length=64)
    dense_replay_input_hash: str = Field(min_length=64, max_length=64)
    neutral_start_cost_protocol_version: str
    proof_hash: str = Field(min_length=64, max_length=64)


class RetrievalEntryCandidateCard(ClosedContractModel):
    protocol_version: str
    protocol_hash: str = Field(min_length=64, max_length=64)
    node_id: str
    layer: Literal["coarse", "mid"]
    semantic_aggregation_protocol_version: str
    intent_strategy: str
    selection_rank: int | None = Field(default=None, ge=1)
    selected: bool
    selection_reasons: list[str] = Field(default_factory=list)
    label: str
    definition_or_summary: str = Field(max_length=512)
    support_count: int = Field(ge=0)
    matched_query_facets: list[str] = Field(default_factory=list)
    semantic_score: float = Field(ge=0.0, le=1.0)
    semantic_candidate: bool
    semantic_support_chunk_ids: list[str] = Field(default_factory=list)
    semantic_support_match_count: int = Field(ge=0)
    topology: RetrievalEntryTopologyPrior
    replay_proof: RetrievalEntryReplayProof
    entry_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    entry_strength_source: str
    neutral_start_cost_protocol_version: str
    neutral_start_cost_is_query_relevance: Literal[False] = False
    topology_used_for_admission_or_tie_break: bool
    topology_used_as_path_distance: Literal[False] = False
    node_weight_used_as_query_relevance: Literal[False] = False
    lexical_overlap_used_as_query_relevance: Literal[False] = False
    gray_zone_rule_inputs_modified: Literal[False] = False
    model_call_count: Literal[0] = 0
    candidate_card_hash: str = Field(min_length=64, max_length=64)


class RetrievalEntryMetadata(ClosedContractModel):
    label: str | None = None
    node_type: str | None = None
    rq_path_prefix: list[int] = Field(default_factory=list)
    representative_terms: list[str] = Field(default_factory=list)
    candidate_card: RetrievalEntryCandidateCard | None = None


class RetrievalEntryNode(ClosedContractModel):
    layer: str
    node_id: str
    entry_strength: float | None = None
    roles: list[str] = Field(default_factory=list)
    rq_prefix_id: str | None = None
    metadata: RetrievalEntryMetadata = Field(default_factory=RetrievalEntryMetadata)


class RetrievalSupportRefs(ClosedContractModel):
    edge_id: str | None = None
    edge_ids: list[str] = Field(default_factory=list)
    edge_type: str | None = None
    edge_types: list[str] = Field(default_factory=list)
    support_ids: list[str] = Field(default_factory=list)
    support_chunk_ids: list[str] = Field(default_factory=list)
    support_chunk_edge_ids: list[str] = Field(default_factory=list)
    support_relation_edge_ids: list[str] = Field(default_factory=list)
    support_rq_prefix_ids: list[str] = Field(default_factory=list)
    support_rq_prefix_node_ids: list[str] = Field(default_factory=list)
    support_mid_edge_ids: list[str] = Field(default_factory=list)
    support_mid_concept_ids: list[str] = Field(default_factory=list)
    entry_strength: float | None = None
    entry_distance: float | None = None
    entry_strengths: dict[str, float] = Field(default_factory=dict)
    raw_entry_strengths: dict[str, float] = Field(default_factory=dict)
    mid_concept_ids: list[str] = Field(default_factory=list)
    rq_prefix_ids: list[str] = Field(default_factory=list)
    chunk_candidate_source: str | None = None
    chunk_candidate_sources: list[str] = Field(default_factory=list)


class RetrievalEntryParentRef(ClosedContractModel):
    parent_layer: str
    parent_node_id: str
    edge_type: str | None = None
    support_refs: RetrievalSupportRefs = Field(default_factory=RetrievalSupportRefs)


class RetrievalStateSignature(ClosedContractModel):
    layer: str | None = None
    node_id: str | None = None
    covered_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    depth_bucket: int | None = Field(default=None, ge=0)
    path_edge_type_multiset: dict[str, int] = Field(default_factory=dict)


class CycleDistanceRewardAudit(ClosedContractModel):
    protocol_version: Literal["bounded_cycle_distance_reward_replay_v1"]
    cycle_edges: list[str] = Field(min_length=1)
    cycle_distance: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    edge_strength: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    support_delta: Literal[0, 1] = 0
    reward_before_cap: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    reward_after_cap: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    cap_reason: Literal[
        "max_cycle_reward_per_path_exhausted",
        "cycle_distance_above_threshold",
        "within_cap",
        "max_cycle_reward_per_path",
    ]

    @model_validator(mode="after")
    def validate_cycle_reward(self) -> "CycleDistanceRewardAudit":
        if any(not edge_id.strip() for edge_id in self.cycle_edges):
            raise ValueError("cycle reward edge ids must be nonempty")
        if self.reward_after_cap > self.reward_before_cap:
            raise ValueError("cycle reward after-cap value exceeds before-cap")
        if self.cap_reason in {
            "max_cycle_reward_per_path_exhausted",
            "cycle_distance_above_threshold",
        } and (
            self.support_delta != 0
            or self.reward_before_cap != 0.0
            or self.reward_after_cap != 0.0
        ):
            raise ValueError("non-rewarding cycle event claims a reward")
        if self.cap_reason in {
            "within_cap",
            "max_cycle_reward_per_path",
        } and self.support_delta != 1:
            raise ValueError("rewarding cycle event has no support delta")
        return self


class RetrievalTraversalState(ClosedContractModel):
    layer: str | None = None
    parent_layer: str | None = None
    parent_node_id: str | None = None
    node_id: str | None = None
    path: list[str] = Field(default_factory=list)
    path_edge_ids: list[str] = Field(default_factory=list)
    path_edge_distances: list[float] = Field(default_factory=list)
    path_edge_strengths: list[float] = Field(default_factory=list)
    path_edge_types: list[str] = Field(default_factory=list)
    distance_so_far: float | None = None
    reward_so_far: float | None = None
    cycle_reward_so_far: float = Field(default=0.0, ge=0.0)
    distance_zone: str | None = None
    covered_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    depth: int | None = None
    root_node_id: str | None = None
    visit_counts: dict[str, int] = Field(default_factory=dict)
    edge_reuse_counts: dict[str, int] = Field(default_factory=dict)
    support_refs: RetrievalSupportRefs = Field(default_factory=RetrievalSupportRefs)
    entry_parent_refs: list[RetrievalEntryParentRef] = Field(default_factory=list)
    entry_support_refs: RetrievalSupportRefs = Field(default_factory=RetrievalSupportRefs)
    state_signature: RetrievalStateSignature = Field(default_factory=RetrievalStateSignature)
    gray_zone_decision: str | None = None
    gray_zone_terminal_action: str | None = None
    cycle_distance_rewards: list[CycleDistanceRewardAudit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_parallel_path_facts(self) -> "RetrievalTraversalState":
        edge_count = len(self.path_edge_ids)
        if not (
            edge_count
            == len(self.path_edge_distances)
            == len(self.path_edge_strengths)
            == len(self.path_edge_types)
            == max(len(self.path) - 1, 0)
        ):
            raise ValueError("retrieval path edge arrays are not parallel")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.path_edge_distances
        ):
            raise ValueError("retrieval path distances must be finite and non-negative")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.path_edge_strengths
        ):
            raise ValueError("retrieval path strengths must be finite unit values")
        return self


class RetrievalFrontierSnapshot(ClosedContractModel):
    layer: str | None = None
    popped: RetrievalTraversalState = Field(default_factory=RetrievalTraversalState)
    queue_size_after_pop: int = 0
    key: list[int | float | str] = Field(default_factory=list)


class RetrievalStageQueue(ClosedContractModel):
    entry_ids: list[str] = Field(default_factory=list)
    forced_entry_ids: list[str] = Field(default_factory=list)
    forced_downstream_entry_ids: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    accepted_ids: list[str] = Field(default_factory=list)
    top_k: int | None = None
    initial_top_k: int | None = None
    frontier_pop_count: int | None = None
    entry_mode: str | None = None
    skipped_by_granularity: str | None = None
    reason: str | None = None


class CandidatePoolDedupeAudit(ClosedContractModel):
    protocol_version: str
    scope: str
    limit: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    unique_admitted_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    rejected_new_count: int = Field(ge=0)
    budget_hit: bool
    hard_interrupt_count: int = Field(ge=0)
    rejected_candidate_id_samples: list[str] = Field(default_factory=list)
    observation_compacted: bool
    stop_reason: str


class PerParentBudgetStatus(ClosedContractModel):
    budget: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    merged_selected_count: int | None = Field(default=None, ge=0)
    stop_reason: str


class RetrievalRQRouteContribution(ClosedContractModel):
    mid_concept_id: str
    mid_entry_strength: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    mid_membership_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    route_fallback_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )

    @model_validator(mode="after")
    def validate_route_score(self) -> "RetrievalRQRouteContribution":
        expected = round(
            self.mid_entry_strength * self.mid_membership_score,
            6,
        )
        if self.route_fallback_score != expected:
            raise ValueError(
                "RQ route fallback score does not replay its components"
            )
        return self


QUERY_RQ_SEED_PROTOCOL_VERSION = (
    "query_rq_primary_residual_mid_dense_v5"
)
QUERY_RQ_SEED_PROTOCOL_HASH = (
    "0bd925993ad11bdc46cf852cfa17e2e62b014ff6f4cc2f5f3c73a5aecf6190bc"
)
CHUNK_FACET_PRIORITY_PROTOCOL_VERSION = (
    "validated_query_facet_posterior_chunk_priority_v2"
)
CHUNK_FACET_PRIORITY_PROTOCOL_HASH = (
    "d266e9eafef9016518b8d42b2924a598bc1d0abbf15e587736a99893b1fa5bc1"
)
QUERY_FACET_POSTERIOR_PROTOCOL_VERSION = (
    "query_facet_posterior_calibration_v1"
)
QUERY_FACET_POSTERIOR_PROTOCOL_HASH = (
    "cbf34022b2e8983c41499cea7e89ac8730ff7b8a3f56d9354e493531e0e0ce08"
)
QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION = (
    "validated_query_facet_ordered_window_v1"
)
QUERY_FACET_ORDERED_WINDOW_PROTOCOL_HASH = (
    "74db72ba2426efe4242ce1599f5f3fd2ee1c85ac90c7602b8a64e9a02c768ac7"
)
QueryRQStageScoreSource = Literal[
    "typed_action_forced_override",
    "query_rq_relevance",
    "selected_mid_route_fallback",
]
QueryRQChunkScoreSource = Literal[
    "query_rq_primary_base",
    "mid_support_without_rq_membership",
]
QueryRQMembershipRole = Literal[
    "primary_member",
    "boundary_member",
    "outlier_member",
    "noise_candidate",
    "bridge_member",
    "low_confidence_member",
    "mid_support_fallback",
]
QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK = {
    "primary_member": 0,
    "boundary_member": 1,
    "outlier_member": 2,
    "noise_candidate": 3,
    "bridge_member": 5,
    "low_confidence_member": 5,
    "mid_support_fallback": 6,
}


def _writer_stable_hash(value: Any) -> str:
    """Replay the active writer's canonical seed-card hash protocol."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rq_lcp_depth(left: list[int], right: list[int]) -> int:
    depth = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        depth += 1
    return depth


def _validate_seed_card_hashes(payload: dict[str, Any]) -> None:
    input_hash = str(payload.pop("input_hash"))
    card_hash = str(payload.pop("card_hash"))
    expected_input_hash = _writer_stable_hash(payload)
    if not hmac.compare_digest(input_hash, expected_input_hash):
        raise ValueError("RQ seed-card input hash mismatch")
    expected_card_hash = _writer_stable_hash(
        {**payload, "input_hash": input_hash}
    )
    if not hmac.compare_digest(card_hash, expected_card_hash):
        raise ValueError("RQ seed-card hash mismatch")


class RetrievalRQStageSeedCard(ClosedContractModel):
    protocol_version: Literal[QUERY_RQ_SEED_PROTOCOL_VERSION]
    protocol_hash: Literal[QUERY_RQ_SEED_PROTOCOL_HASH]
    rq_prefix_id: str
    rq_path: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=list
    )
    rq_level: int = Field(ge=0, strict=True)
    query_rq_path: list[
        Annotated[int, Field(strict=True, ge=0)]
    ] = Field(default_factory=list)
    rq_lcp_depth: int = Field(ge=0, strict=True)
    residual_distance: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False, strict=True
    )
    query_prefix_membership_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
    )
    requested_query_relevance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
    )
    route_fallback_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    parent_mid_contributions: list[
        RetrievalRQRouteContribution
    ] = Field(default_factory=list)
    score_source: QueryRQStageScoreSource
    effective_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    forced_override: bool = Field(strict=True)
    relation_state_hash: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    is_evidence: Literal[False]
    node_weight_used_as_query_relevance: Literal[False]
    hard_path_lcp_used_as_score: Literal[False]
    gray_zone_decision_authority: Literal[False]
    model_call_count: Literal[0]
    input_hash: str = Field(min_length=64, max_length=64)
    card_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_stage_seed_card(self) -> "RetrievalRQStageSeedCard":
        if not self.rq_prefix_id.strip():
            raise ValueError("RQ stage seed prefix id must be nonempty")
        if self.rq_level != len(self.rq_path):
            raise ValueError("RQ stage level does not match path length")
        if self.rq_lcp_depth != _rq_lcp_depth(
            self.query_rq_path, self.rq_path
        ):
            raise ValueError("RQ stage LCP depth does not replay paths")
        contributions = self.parent_mid_contributions
        if [item.mid_concept_id for item in contributions] != sorted(
            item.mid_concept_id for item in contributions
        ) or len({item.mid_concept_id for item in contributions}) != len(
            contributions
        ):
            raise ValueError(
                "RQ stage route contributions must be unique and ordered"
            )
        expected_route_fallback = max(
            [item.route_fallback_score for item in contributions] or [0.0]
        )
        if self.route_fallback_score != expected_route_fallback:
            raise ValueError(
                "RQ stage route fallback does not replay contributions"
            )
        if self.forced_override != (
            self.score_source == "typed_action_forced_override"
        ):
            raise ValueError("RQ stage forced override authority mismatch")
        if self.score_source == "typed_action_forced_override":
            expected_score = 1.0
        elif self.score_source == "query_rq_relevance":
            if self.requested_query_relevance is None:
                raise ValueError(
                    "query RQ relevance seed is missing requested score"
                )
            expected_score = self.requested_query_relevance
        else:
            if self.requested_query_relevance is not None:
                raise ValueError(
                    "route fallback cannot claim requested query relevance"
                )
            expected_score = self.route_fallback_score
        if self.effective_score != expected_score:
            raise ValueError("RQ stage effective score does not replay source")
        _validate_seed_card_hashes(self.model_dump(mode="json"))
        return self


class RetrievalRQChunkSeedCard(ClosedContractModel):
    protocol_version: Literal[QUERY_RQ_SEED_PROTOCOL_VERSION]
    protocol_hash: Literal[QUERY_RQ_SEED_PROTOCOL_HASH]
    parent_mid_concept_id: str
    chunk_id: str
    rq_l3_prefix_id: str | None = None
    query_rq_path: list[
        Annotated[int, Field(strict=True, ge=0)]
    ] = Field(default_factory=list)
    candidate_rq_path: list[
        Annotated[int, Field(strict=True, ge=0)]
    ] = Field(default_factory=list)
    rq_lcp_depth: int = Field(ge=0, strict=True)
    residual_distance: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False, strict=True
    )
    query_prefix_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    chunk_membership_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    membership_overlap_diagnostic_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    rq_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    residual_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    rq_relevance_component: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    primary_membership: bool = Field(strict=True)
    membership_overlap_used_in_effective_score: Literal[False]
    query_membership_entropy: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    rq_drift_penalty: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False, strict=True
    )
    membership_role: QueryRQMembershipRole
    membership_rank: int = Field(ge=0, strict=True)
    membership_entropy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
    )
    bridge_or_boundary_role: bool = Field(strict=True)
    support_edge_ids: list[str] = Field(default_factory=list)
    mid_entry_component: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    dense_component: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    component_weights: dict[str, float] = Field(default_factory=dict)
    effective_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    score_source: QueryRQChunkScoreSource
    membership_role_tie_break_rank: int = Field(ge=0, strict=True)
    is_evidence: Literal[False]
    node_weight_used_as_query_relevance: Literal[False]
    hard_path_lcp_used_as_score: Literal[False]
    gray_zone_decision_authority: Literal[False]
    model_call_count: Literal[0]
    input_hash: str = Field(min_length=64, max_length=64)
    card_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_chunk_seed_card(self) -> "RetrievalRQChunkSeedCard":
        if not self.parent_mid_concept_id.strip() or not self.chunk_id.strip():
            raise ValueError("RQ chunk seed identities must be nonempty")
        if self.rq_lcp_depth != _rq_lcp_depth(
            self.query_rq_path, self.candidate_rq_path
        ):
            raise ValueError("RQ chunk seed LCP depth does not replay paths")
        if len(set(self.support_edge_ids)) != len(self.support_edge_ids):
            raise ValueError("RQ chunk support edge ids must be unique")
        if self.membership_role_tie_break_rank != QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK[
            self.membership_role
        ]:
            raise ValueError("RQ chunk membership role rank mismatch")
        if self.score_source == "query_rq_primary_base":
            if self.rq_l3_prefix_id is None:
                raise ValueError("RQ membership seed is missing its prefix")
            expected_weights = {
                "rq_relevance": 0.2,
                "mid_entry": 0.5,
                "dense": 0.3,
            }
            expected_overlap = round(
                math.sqrt(
                    self.query_prefix_score
                    * self.chunk_membership_score
                ),
                6,
            )
            expected_relevance = (
                self.residual_score
                if self.primary_membership
                else 0.0
            )
            if self.membership_role == "mid_support_fallback":
                raise ValueError("RQ membership seed cannot use fallback role")
        else:
            expected_weights = {
                "rq_relevance": 0.0,
                "mid_entry": 0.8,
                "dense": 0.2,
            }
            expected_overlap = 0.0
            expected_relevance = 0.0
            if (
                self.rq_l3_prefix_id is not None
                or self.query_rq_path
                or self.candidate_rq_path
                or self.rq_lcp_depth != 0
                or self.residual_distance is not None
                or self.query_prefix_score != 0.0
                or self.chunk_membership_score != 0.0
                or self.rq_score != 0.0
                or self.residual_score != 0.0
                or self.rq_drift_penalty is not None
                or self.primary_membership
                or self.query_membership_entropy != 1.0
                or self.membership_role != "mid_support_fallback"
                or self.membership_rank != 0
                or self.membership_entropy is not None
                or self.bridge_or_boundary_role
                or self.support_edge_ids
            ):
                raise ValueError("RQ fallback seed contains membership facts")
        if self.component_weights != expected_weights:
            raise ValueError("RQ chunk seed component weights mismatch")
        if self.membership_overlap_diagnostic_score != expected_overlap:
            raise ValueError("RQ chunk membership overlap does not replay inputs")
        if self.rq_relevance_component != expected_relevance:
            raise ValueError("RQ relevance component does not replay inputs")
        if self.membership_overlap_used_in_effective_score is not False:
            raise ValueError("membership overlap cannot enter the base seed score")
        expected_effective = round(
            expected_weights["rq_relevance"]
            * self.rq_relevance_component
            + expected_weights["mid_entry"] * self.mid_entry_component
            + expected_weights["dense"] * self.dense_component,
            6,
        )
        if self.effective_score != expected_effective:
            raise ValueError("RQ chunk effective score does not replay inputs")
        _validate_seed_card_hashes(self.model_dump(mode="json"))
        return self


class RetrievalChunkFacetPriorityCard(ClosedContractModel):
    protocol_version: Literal[CHUNK_FACET_PRIORITY_PROTOCOL_VERSION]
    protocol_hash: Literal[CHUNK_FACET_PRIORITY_PROTOCOL_HASH]
    facet_match_protocol_version: Literal[
        QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION
    ]
    facet_match_protocol_hash: Literal[
        QUERY_FACET_ORDERED_WINDOW_PROTOCOL_HASH
    ]
    chunk_id: str
    query_facet_packet_hash: str = Field(min_length=64, max_length=64)
    required_facets: list[str] = Field(default_factory=list)
    matched_required_facets: list[str] = Field(default_factory=list)
    uncovered_required_facets: list[str] = Field(default_factory=list)
    matched_required_facet_count: int = Field(ge=0, strict=True)
    uncovered_required_facet_count: int = Field(ge=0, strict=True)
    priority_prefix: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        min_length=1,
        max_length=1,
    )
    lexical_overlap_used_as_numeric_relevance: Literal[False]
    is_evidence: Literal[False]
    citation_authority: Literal[False]
    gray_zone_decision_authority: Literal[False]
    model_call_count: Literal[0]
    card_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_facet_priority(self) -> "RetrievalChunkFacetPriorityCard":
        if not self.chunk_id.strip():
            raise ValueError("chunk facet priority identity must be nonempty")
        if (
            len(self.required_facets) != len(set(self.required_facets))
            or len(self.matched_required_facets)
            != len(set(self.matched_required_facets))
            or len(self.uncovered_required_facets)
            != len(set(self.uncovered_required_facets))
        ):
            raise ValueError("chunk facet priority lists must be unique")
        required = set(self.required_facets)
        matched = set(self.matched_required_facets)
        uncovered = set(self.uncovered_required_facets)
        if (
            not matched.issubset(required)
            or not uncovered.issubset(required)
            or matched.intersection(uncovered)
            or matched.union(uncovered) != required
        ):
            raise ValueError("chunk facet priority coverage does not partition required facets")
        if (
            self.matched_required_facet_count != len(matched)
            or self.uncovered_required_facet_count != len(uncovered)
            or self.priority_prefix != [len(uncovered)]
        ):
            raise ValueError("chunk facet priority counts do not replay coverage")
        payload = self.model_dump(mode="json")
        card_hash = str(payload.pop("card_hash"))
        if not hmac.compare_digest(card_hash, _writer_stable_hash(payload)):
            raise ValueError("chunk facet priority card hash mismatch")
        return self


class RetrievalCandidatePool(ClosedContractModel):
    parent_layer: str | None = None
    parent_node_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_scores: dict[
        str,
        Annotated[
            float,
            Field(strict=True, allow_inf_nan=False),
        ],
    ] = Field(default_factory=dict)
    rq_seed_cards: dict[str, RetrievalRQStageSeedCard] = Field(
        default_factory=dict
    )
    rq_chunk_seed_cards: dict[
        str, list[RetrievalRQChunkSeedCard]
    ] = Field(default_factory=dict)
    chunk_facet_priority_cards: dict[
        str, RetrievalChunkFacetPriorityCard
    ] = Field(
        default_factory=dict
    )
    query_facet_posterior_snapshot: dict[str, Any] | None = None
    covered_posterior_mass_by_candidate: dict[
        str,
        Annotated[
            float,
            Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
        ],
    ] = Field(default_factory=dict)
    facet_priority_protocol_version: str | None = None
    facet_priority_protocol_hash: str | None = None
    ranking_protocol_version: str | None = None
    ranking_protocol_hash: str | None = None
    forced_candidate_ids: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    ranked_selected_ids: list[str] = Field(default_factory=list)
    candidate_count: int | None = None
    top_k: int | None = None
    stop_reason: str | None = None
    source: str | None = None
    coarse_skipped_reason: str | None = None
    per_parent_budget_status: PerParentBudgetStatus | None = None
    candidate_dedupe_budget_audit: CandidatePoolDedupeAudit | None = None

    @model_validator(mode="after")
    def validate_query_rq_card_coverage(self) -> "RetrievalCandidatePool":
        candidate_ids = self.candidate_ids
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate pool ids must be unique")
        if self.candidate_count is not None and self.candidate_count != len(
            candidate_ids
        ):
            raise ValueError("candidate pool count does not replay ids")
        candidate_scope = set(candidate_ids)
        if any(
            candidate_id not in candidate_scope
            for candidate_id in [
                *self.selected_ids,
                *self.ranked_selected_ids,
                *self.forced_candidate_ids,
            ]
        ):
            raise ValueError("candidate pool selection escapes its id scope")

        facet_cards = self.chunk_facet_priority_cards
        if facet_cards:
            if set(facet_cards) != candidate_scope:
                raise ValueError(
                    "chunk facet priority cards do not cover candidate ids"
                )
            if (
                self.facet_priority_protocol_version
                != CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
                or self.facet_priority_protocol_hash
                != CHUNK_FACET_PRIORITY_PROTOCOL_HASH
            ):
                raise ValueError("chunk facet priority protocol mismatch")
            if any(
                card.chunk_id != candidate_id
                for candidate_id, card in facet_cards.items()
            ):
                raise ValueError(
                    "chunk facet priority card identity does not replay pool"
                )
            posterior_snapshot = dict(
                self.query_facet_posterior_snapshot or {}
            )
            supplied_snapshot_hash = str(
                posterior_snapshot.pop("snapshot_hash", "")
            )
            posterior_values = posterior_snapshot.get("posterior")
            if (
                posterior_snapshot.get("protocol_version")
                != QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
                or posterior_snapshot.get("protocol_hash")
                != QUERY_FACET_POSTERIOR_PROTOCOL_HASH
                or not isinstance(posterior_values, dict)
                or (
                    posterior_values
                    and not math.isclose(
                        sum(float(value) for value in posterior_values.values()),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                )
                or not hmac.compare_digest(
                    supplied_snapshot_hash,
                    _writer_stable_hash(posterior_snapshot),
                )
            ):
                raise ValueError("query facet posterior snapshot mismatch")
            if set(self.covered_posterior_mass_by_candidate) != candidate_scope:
                raise ValueError(
                    "query facet posterior masses do not cover candidate ids"
                )
            if any(
                self.covered_posterior_mass_by_candidate[candidate_id]
                != round(
                    sum(
                        float(posterior_values.get(facet, 0.0))
                        for facet in card.matched_required_facets
                    ),
                    6,
                )
                for candidate_id, card in facet_cards.items()
            ):
                raise ValueError(
                    "query facet posterior masses do not replay candidate coverage"
                )
        elif (
            self.facet_priority_protocol_version is not None
            or self.facet_priority_protocol_hash is not None
            or self.query_facet_posterior_snapshot is not None
            or self.covered_posterior_mass_by_candidate
        ):
            raise ValueError(
                "chunk facet priority protocol cannot exist without cards"
            )

        has_stage_cards = bool(self.rq_seed_cards)
        has_chunk_cards = bool(self.rq_chunk_seed_cards)
        if has_stage_cards and has_chunk_cards:
            raise ValueError("candidate pool mixes stage and chunk RQ cards")
        if not has_stage_cards and not has_chunk_cards:
            if facet_cards:
                expected_order = sorted(
                    candidate_ids,
                    key=lambda candidate_id: (
                        facet_cards[
                            candidate_id
                        ].uncovered_required_facet_count,
                        -self.covered_posterior_mass_by_candidate[
                            candidate_id
                        ],
                        -self.candidate_scores[candidate_id],
                        candidate_id,
                    ),
                )
                if candidate_ids != expected_order:
                    raise ValueError(
                        "chunk facet candidate ids do not replay ranking order"
                    )
            return self
        if (
            self.ranking_protocol_version
            != QUERY_RQ_SEED_PROTOCOL_VERSION
            or self.ranking_protocol_hash != QUERY_RQ_SEED_PROTOCOL_HASH
        ):
            raise ValueError("candidate pool RQ ranking protocol mismatch")
        if set(self.candidate_scores) != candidate_scope:
            raise ValueError("candidate pool scores do not cover candidate ids")

        if has_stage_cards:
            if set(self.rq_seed_cards) != candidate_scope:
                raise ValueError("RQ stage cards do not cover candidate ids")
            for candidate_id, card in self.rq_seed_cards.items():
                if (
                    card.rq_prefix_id != candidate_id
                    or self.candidate_scores[candidate_id]
                    != card.effective_score
                ):
                    raise ValueError(
                        "RQ stage card identity or score does not replay pool"
                    )
            expected_order = sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    -self.candidate_scores[candidate_id],
                    candidate_id,
                ),
            )
        else:
            chunk_card_scope = set(self.rq_chunk_seed_cards)
            if (
                any(not cards for cards in self.rq_chunk_seed_cards.values())
                or not chunk_card_scope.issubset(candidate_scope)
                or (
                    self.parent_layer == "mid"
                    and chunk_card_scope != candidate_scope
                )
            ):
                raise ValueError(
                    "RQ chunk cards do not match their candidate scope"
                )
            tie_breaks: dict[str, int] = {}
            for candidate_id, cards in self.rq_chunk_seed_cards.items():
                card_hashes = [card.card_hash for card in cards]
                if (
                    card_hashes != sorted(card_hashes)
                    or len(card_hashes) != len(set(card_hashes))
                ):
                    raise ValueError(
                        "RQ chunk cards must be unique and canonically ordered"
                    )
                if any(
                    card.chunk_id != candidate_id
                    or (
                        self.parent_layer == "mid"
                        and card.parent_mid_concept_id
                        != self.parent_node_id
                    )
                    for card in cards
                ):
                    raise ValueError(
                        "RQ chunk card identity does not replay pool"
                    )
                expected_score = max(card.effective_score for card in cards)
                if (
                    self.parent_layer == "mid"
                    and self.candidate_scores[candidate_id]
                    != expected_score
                ) or (
                    self.parent_layer != "mid"
                    and self.candidate_scores[candidate_id]
                    < expected_score
                ):
                    raise ValueError(
                        "RQ chunk card score does not replay pool"
                    )
                tie_breaks[candidate_id] = min(
                    card.membership_role_tie_break_rank for card in cards
                )
            if self.parent_layer == "mid":
                if not self.parent_node_id:
                    raise ValueError(
                        "per-Mid RQ chunk pool has no parent identity"
                    )
                expected_order = sorted(
                    candidate_ids,
                    key=lambda candidate_id: (
                        *(
                            [
                                facet_cards[candidate_id].uncovered_required_facet_count
                            ]
                            if facet_cards
                            else []
                        ),
                        *(
                            [
                                -self.covered_posterior_mass_by_candidate[
                                    candidate_id
                                ]
                            ]
                            if facet_cards
                            else []
                        ),
                        -self.candidate_scores[candidate_id],
                        tie_breaks[candidate_id],
                        candidate_id,
                    ),
                )
            else:
                if self.parent_node_id is not None:
                    raise ValueError(
                        "merged RQ chunk pool cannot claim one Mid parent"
                    )
                expected_order = sorted(
                    candidate_ids,
                    key=lambda candidate_id: (
                        *(
                            [
                                facet_cards[candidate_id].uncovered_required_facet_count
                            ]
                            if facet_cards
                            else []
                        ),
                        *(
                            [
                                -self.covered_posterior_mass_by_candidate[
                                    candidate_id
                                ]
                            ]
                            if facet_cards
                            else []
                        ),
                        -self.candidate_scores[candidate_id],
                        candidate_id,
                    ),
                )
        if candidate_ids != expected_order:
            raise ValueError("RQ candidate ids do not replay ranking order")
        return self


class CandidatePoolDedupeSummary(ClosedContractModel):
    protocol_version: str
    limit_per_pool: int = Field(ge=0)
    pool_count: int = Field(ge=0)
    budget_hit_pool_count: int = Field(ge=0)
    hard_interrupt_count: int = Field(ge=0)
    unique_admitted_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    audits: list[CandidatePoolDedupeAudit] = Field(default_factory=list)


class RetrievalCandidatePools(ClosedContractModel):
    mid_by_coarse: list[RetrievalCandidatePool] = Field(default_factory=list)
    chunk_by_mid: list[RetrievalCandidatePool] = Field(default_factory=list)
    mid_direct_entries: RetrievalCandidatePool | None = None
    mid_initial_entries: RetrievalCandidatePool | None = None
    rq_membership_entries: RetrievalCandidatePool | None = None
    chunk_initial_entries: RetrievalCandidatePool | None = None
    candidate_dedupe_budget: CandidatePoolDedupeSummary | None = None


class RetrievalTopKSelection(ClosedContractModel):
    top_k: int | None = None
    candidate_count: int = 0
    selected_ids: list[str] = Field(default_factory=list)
    forced_selected_ids: list[str] = Field(default_factory=list)
    ranking_protocol_version: str | None = None
    candidate_rank_facts: list["RetrievalCandidateRankFact"] = Field(
        default_factory=list
    )
    carry_forward_supported_chunk_ids: list[str] = Field(
        default_factory=list
    )
    global_top_k_increased: bool | None = None
    stop_reason: str | None = None
    entry_mode: str | None = None


class RetrievalCandidateRankFact(ClosedContractModel):
    candidate_id: str
    rank_key: list[int | float | str] = Field(default_factory=list)
    path_identity: str
    repair_evidence_retention_protocol_version: Literal[
        "repair_supported_evidence_carry_forward_v1"
    ] | None = None
    source_context_package_id: str | None = None
    source_retrieval_trace_id: str | None = None
    repair_directive_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )


class RetrievalPathLabel(ClosedContractModel):
    layer: str | None = None
    node_id: str | None = None
    chunk_id: str | None = None
    path: list[str] = Field(default_factory=list)
    path_edge_ids: list[str] = Field(default_factory=list)
    path_edge_types: list[str] = Field(default_factory=list)
    expanded_edge_ids: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    distance_so_far: float | None = None
    reward_so_far: float | None = None
    cycle_reward_so_far: float = Field(default=0.0, ge=0.0)
    root_node_id: str | None = None
    parent_layer: str | None = None
    parent_node_id: str | None = None
    stop_reason: str | None = None
    support_refs: RetrievalSupportRefs = Field(default_factory=RetrievalSupportRefs)
    entry_parent_refs: list[RetrievalEntryParentRef] = Field(default_factory=list)
    path_edge_type_multiset: dict[str, int] = Field(default_factory=dict)
    edge_reuse_counts: dict[str, int] = Field(default_factory=dict)
    repair_evidence_retention_protocol_version: Literal[
        "repair_supported_evidence_carry_forward_v1"
    ] | None = None
    source_context_package_id: str | None = None
    source_retrieval_trace_id: str | None = None
    repair_directive_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )


class RetrievalPathContribution(ClosedContractModel):
    contract_version: Literal["multi_path_contribution_v2"] = (
        "multi_path_contribution_v2"
    )
    contribution_id: str = Field(min_length=64, max_length=64)
    layer: Literal["coarse", "mid", "chunk"]
    node_id: str
    parent_layer: str | None = None
    parent_node_id: str | None = None
    origin_parent_layer: str | None = None
    origin_parent_node_id: str | None = None
    root_node_id: str
    path: list[str] = Field(min_length=1)
    path_edge_ids: list[str] = Field(default_factory=list)
    path_edge_types: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    support_refs: RetrievalSupportRefs = Field(default_factory=RetrievalSupportRefs)
    support_chunk_ids: list[str] = Field(default_factory=list)
    distance_so_far: float = Field(ge=0.0, allow_inf_nan=False)
    reward_so_far: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_path_contribution(self) -> "RetrievalPathContribution":
        required_identifiers = {
            "node_id": self.node_id,
            "root_node_id": self.root_node_id,
        }
        if any(
            not value.strip()
            for value in required_identifiers.values()
        ):
            raise ValueError(
                "path contribution identifiers must be nonempty"
            )
        optional_identifiers = {
            "parent_layer": self.parent_layer,
            "parent_node_id": self.parent_node_id,
            "origin_parent_layer": self.origin_parent_layer,
            "origin_parent_node_id": self.origin_parent_node_id,
        }
        if any(
            value is not None and not value.strip()
            for value in optional_identifiers.values()
        ):
            raise ValueError(
                "optional path contribution identifiers cannot be empty"
            )
        if any(
            not value.strip()
            for value in [
                *self.path,
                *self.path_edge_ids,
                *self.path_edge_types,
            ]
        ):
            raise ValueError(
                "path contribution path facts must be nonempty strings"
            )
        if self.path[-1] != self.node_id:
            raise ValueError("path contribution must terminate at node_id")
        if len(self.path) > 1 and (
            self.parent_node_id != self.path[-2]
        ):
            raise ValueError(
                "path contribution parent_node_id must replay the "
                "physical path predecessor"
            )
        has_origin_layer = self.origin_parent_layer is not None
        has_origin_node = self.origin_parent_node_id is not None
        if has_origin_layer != has_origin_node:
            raise ValueError(
                "path contribution origin parent layer and node must be "
                "present together"
            )
        if has_origin_node and (
            len(self.path) < 2
            or self.path[0] != self.origin_parent_node_id
            or self.path[1] != self.root_node_id
        ):
            raise ValueError(
                "path contribution origin parent must be the trace-bound "
                "prefix immediately before root_node_id"
            )
        if not has_origin_node and self.path[0] != self.root_node_id:
            raise ValueError(
                "path contribution without an origin parent must start at "
                "root_node_id"
            )
        canonical_list_fields = {
            "covered_facets": self.covered_facets,
            "evidence_roles": self.evidence_roles,
            "support_chunk_ids": self.support_chunk_ids,
        }
        for field, values in canonical_list_fields.items():
            if values != sorted(set(values)):
                raise ValueError(
                    f"path contribution {field} must be sorted and unique"
                )
        if any(character not in "0123456789abcdef" for character in self.contribution_id.lower()):
            raise ValueError("path contribution id must be a SHA-256 hex digest")
        expected_id = retrieval_path_contribution_id(self)
        if self.contribution_id != expected_id:
            raise ValueError(
                "path contribution id does not replay contribution "
                "identity"
            )
        return self


class RetrievalNodeContributionSummary(ClosedContractModel):
    contract_version: Literal["multi_path_contribution_union_v2"] = (
        "multi_path_contribution_union_v2"
    )
    layer: Literal["coarse", "mid", "chunk"]
    node_id: str
    node_visit_count: int = Field(ge=0)
    distinct_parent_count: int = Field(ge=0)
    distinct_path_count: int = Field(ge=0)
    distinct_edge_type_count: int = Field(ge=0)
    parent_node_ids: list[str] = Field(default_factory=list)
    path_edge_types: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    support_id_union: list[str] = Field(default_factory=list)
    support_chunk_union: list[str] = Field(default_factory=list)
    cycle_convergence_score: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )
    best_distance: float = Field(ge=0.0, allow_inf_nan=False)
    best_reward: float = Field(ge=0.0, allow_inf_nan=False)
    reached_by_paths: list[RetrievalPathContribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contribution_union(self) -> "RetrievalNodeContributionSummary":
        paths = self.reached_by_paths
        path_ids = [path.contribution_id for path in paths]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("node contribution paths must have unique contribution ids")
        if path_ids != sorted(path_ids):
            raise ValueError(
                "node contribution paths must use canonical contribution-id order"
            )
        if any(path.layer != self.layer or path.node_id != self.node_id for path in paths):
            raise ValueError("node contribution paths must target the summarized node")
        expected = retrieval_node_contribution_facts(paths)
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ValueError(
                    f"{field} does not replay reached_by_paths"
                )
        return self


class GrayTraversalObservationBudgetState(ClosedContractModel):
    protocol_version: str
    scope: str
    limit: int = Field(ge=1)
    local_rule_evaluation_index: int = Field(ge=1)
    layer_local_rule_evaluation_index: int = Field(ge=1)
    cadence_due: bool
    expanded_packet_requested: bool
    expanded_observation_count_before: int = Field(ge=0)
    expanded_observation_count_after: int = Field(ge=0)
    remaining_after: int = Field(ge=0)
    observation_compacted: bool
    compaction_reason: str | None = None
    hard_interrupt_applied: bool
    budget_exhausted_after: bool
    model_call_count: Literal[0]


class GrayHardInterruptState(ClosedContractModel):
    edge_reuse_count: int | None = Field(default=None, ge=0)
    max_edge_reuse: int | None = Field(default=None, ge=0)
    frontier_expansion_count: int | None = Field(default=None, ge=0)
    frontier_expansion_budget: int | None = Field(default=None, ge=0)
    per_entry_expansion_count: int | None = Field(default=None, ge=0)
    per_entry_expansion_budget: int | None = Field(default=None, ge=0)
    path_distance_hard_stop: bool | None = None
    traversal_observation_budget: GrayTraversalObservationBudgetState | None = None


class GrayRQMembershipRoleThresholds(ClosedContractModel):
    noise_membership_score_max: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    outlier_gamma_max: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    outlier_residual_quantile: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    low_confidence_gamma_max: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    low_confidence_membership_score_max: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    boundary_entropy_min: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    boundary_probability_margin_max: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    boundary_distance_max: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )


class GrayRQMembershipRoleInputs(ClosedContractModel):
    membership_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    membership_entropy: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    residual_norm: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    gamma: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    boundary_probability_margin: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    boundary_distance: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    residual_outlier_threshold: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    rank: int = Field(ge=1, strict=True)
    is_primary_prefix: Literal[True]
    is_bridge_chunk: bool = Field(strict=True)


class GrayRQMembershipRoleEvaluation(ClosedContractModel):
    role: Literal[
        "noise_candidate",
        "outlier_member",
        "bridge_member",
        "low_confidence_member",
        "boundary_member",
        "primary_member",
    ]
    matched_flags: list[
        Literal[
            "noise_candidate",
            "outlier_member",
            "bridge_member",
            "low_confidence_member",
            "boundary_member",
            "primary_member",
        ]
    ]
    primary_reason: Literal[
        "noise_candidate",
        "outlier_member",
        "bridge_member",
        "low_confidence_member",
        "boundary_member",
        "primary_member",
    ]
    protocol_version: Literal["rq_membership_role_primary_entropy_boundary_v2"]
    protocol_hash: str = Field(min_length=64, max_length=64)
    thresholds: GrayRQMembershipRoleThresholds
    inputs: GrayRQMembershipRoleInputs
    model_call_count: Literal[0]

    @model_validator(mode="after")
    def validate_role_evaluation(self) -> "GrayRQMembershipRoleEvaluation":
        if self.primary_reason != self.role or self.role not in self.matched_flags:
            raise ValueError("RQ membership role does not replay matched flags")
        if len(self.matched_flags) != len(set(self.matched_flags)):
            raise ValueError("RQ membership role matched flags are duplicated")
        return self


class GrayRQScore(ClosedContractModel):
    query_rq_path: list[Annotated[int, Field(strict=True, ge=1)]]
    candidate_rq_path: list[Annotated[int, Field(strict=True, ge=1)]]
    lcp_depth: int = Field(ge=0, strict=True)
    lcp_ratio_diagnostic_only: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    residual_distance: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    query_residual_norm: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    candidate_residual_norm: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    query_prefix_membership_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    candidate_prefix_membership_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    membership_overlap_diagnostic_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    rq_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    residual_score: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, strict=True
    )
    rq_drift_penalty: float = Field(
        ge=0.0, allow_inf_nan=False, strict=True
    )
    membership_reason: Literal["rq_prefix", "rq_leaf"]
    membership_role: Literal[
        "noise_candidate",
        "outlier_member",
        "bridge_member",
        "low_confidence_member",
        "boundary_member",
        "primary_member",
    ]
    membership_rank: int = Field(ge=1, strict=True)
    membership_entropy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
    )
    boundary_probability_margin: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
    )
    boundary_distance: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False, strict=True
    )
    membership_role_evaluation: GrayRQMembershipRoleEvaluation | None = None
    membership_protocol_version: Literal[
        "rq_primary_chain_v1"
    ] | None = None
    membership_protocol_hash: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    hard_path_lcp_used_as_score: Literal[False]

    @model_validator(mode="after")
    def validate_rq_score_paths(self) -> "GrayRQScore":
        expected_lcp = _rq_lcp_depth(
            self.query_rq_path, self.candidate_rq_path
        )
        expected_ratio = round(
            expected_lcp
            / max(
                len(self.query_rq_path),
                len(self.candidate_rq_path),
                1,
            ),
            6,
        )
        if self.lcp_depth != expected_lcp:
            raise ValueError("RQ score LCP depth does not replay paths")
        if self.lcp_ratio_diagnostic_only != expected_ratio:
            raise ValueError("RQ score LCP ratio does not replay paths")
        if (
            self.membership_role_evaluation is None
            or self.membership_role_evaluation.role
            != self.membership_role
        ):
            raise ValueError("RQ score membership role audit is missing or drifts")
        return self


class GrayProjectedRQDiagnostics(ClosedContractModel):
    rq_score: float | None = None
    rq_drift_penalty: float | None = None
    lcp_depth: int | None = Field(default=None, ge=0)
    residual_distance: float | None = None
    query_prefix_membership_score: float | None = None
    candidate_prefix_membership_score: float | None = None
    membership_overlap_diagnostic_score: float | None = None
    membership_reason: str | None = None
    membership_role: str | None = None
    membership_rank: int | None = Field(default=None, ge=0)
    membership_entropy: float | None = None
    boundary_probability_margin: float | None = None
    boundary_distance: float | None = None
    membership_protocol_version: str | None = None
    membership_protocol_hash: str | None = None
    hard_path_lcp_used_as_score: Literal[False] | None = None


class GrayRQMembershipDiagnostics(ClosedContractModel):
    projection_protocol_version: Literal[
        "gray_rq_membership_observation_projection_v1"
    ] | None = None
    source_present: bool | None = None
    diagnostics: GrayProjectedRQDiagnostics | None = None
    # Closed legacy diagnostic projection.  These fields are audit-only and
    # never participate in the deterministic gray rule.
    membership_role: str | None = None
    matched_flags: list[str] = Field(default_factory=list)
    model_call_count: Literal[0] | None = None


class GrayCandidateChunkSpanSummary(ClosedContractModel):
    chunk_id: str | None = None
    document_version_id: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)


class GrayStructureContextStatus(ClosedContractModel):
    # Compacted deterministic observations persist an empty status card.  The
    # absence of this diagnostic bit is not permission to change the local
    # gray decision and must remain replay-compatible.
    available: bool | None = None
    reason: str | None = None
    mapped_chunk_id: str | None = None


class GrayObservation(ClosedContractModel):
    current_layer: str
    path_distance: float
    distance_zone: Literal["green", "gray"] | None = None
    covered_facets_before: list[str] = Field(default_factory=list)
    covered_facets_after: list[str] = Field(default_factory=list)
    missing_facets_after: list[str] = Field(default_factory=list)
    required_facets: list[str] = Field(default_factory=list)
    candidate_facets: list[str] = Field(default_factory=list)
    evidence_roles_before: list[str] = Field(default_factory=list)
    evidence_roles_after: list[str] = Field(default_factory=list)
    support_ids_before: list[str] = Field(default_factory=list)
    support_ids_after: list[str] = Field(default_factory=list)
    support_ids_before_count: int | None = Field(default=None, ge=0)
    support_ids_after_count: int | None = Field(default=None, ge=0)
    support_ids_before_hash: str | None = None
    support_ids_after_hash: str | None = None
    support_id_gain: bool | None = None
    independent_path_contribution_gain: bool | None = None
    path_contribution_key: str | None = None
    support_refs: RetrievalSupportRefs | None = None
    active_edge_support_gate_pass: bool | None = None
    support_gate_pass: bool | None = None
    support_backed_to_covered_path: bool | None = None
    validated_entry_semantic_anchor: bool | None = None
    facet_gain: bool | None = None
    role_gain: bool | None = None
    support_gain: bool | None = None
    query_anchor_preserved: bool | None = None
    drift_risk_high: bool | None = None
    closure_required: bool | None = None
    semantic_uncertain_edge: bool | None = None
    crossing_rq_boundary: bool | None = None
    bridge_or_boundary_reason: list[str] = Field(default_factory=list)
    bridge_eligible: bool | None = None
    edge_type: str | None = None
    supported_raw_span_hit: bool | None = None
    structure_context_available: bool | None = None
    drilldown_eligible: bool | None = None
    drift_risk: bool | None = None
    rq_membership_diagnostics: GrayRQMembershipDiagnostics | None = None
    candidate_chunk_span_summary: GrayCandidateChunkSpanSummary | None = None
    structure_context_status: GrayStructureContextStatus | None = None
    path_distance_green_threshold: float | None = None
    path_distance_gray_threshold: float | None = None
    path_distance_hard_threshold: float | None = None
    gray_zone_rule_protocol_version: str | None = None
    # v0 audit fixtures persisted this already-derived bit. It remains explicit
    # and closed, but the v1 executor never uses it as an input authority.
    support_progress: bool | None = None


class GrayMinimumAudit(ClosedContractModel):
    input_identity_protocol_version: Literal[
        "gray_zone_minimum_replay_card_v1"
    ] | None = None
    current_layer: str | None = None
    distance_zone: Literal["green", "gray", "red", "hard_stop"] | None = None
    path_distance: float
    predicates: dict[str, bool] = Field(default_factory=dict)
    required_facets_hash: str | None = None
    covered_facets_before_hash: str | None = None
    covered_facets_after_hash: str | None = None
    candidate_facets_hash: str | None = None
    evidence_roles_before_hash: str | None = None
    evidence_roles_after_hash: str | None = None
    bounded_support_ids_before_hash: str | None = None
    bounded_support_ids_after_hash: str | None = None
    support_ids_before_count: int | None = Field(default=None, ge=0)
    support_ids_after_count: int | None = Field(default=None, ge=0)
    support_ids_before_hash: str | None = None
    support_ids_after_hash: str | None = None
    support_ids_after: list[str] = Field(default_factory=list)
    independent_path_contribution_gain: bool | None = None
    thresholds: dict[str, float | None] = Field(default_factory=dict)
    edge_distance_protocol: str | None = None
    path_contribution_key: str | None = None
    support_refs_hash: str | None = None
    bridge_or_boundary_reason_hash: str | None = None
    edge_type: str | None = None
    rq_membership_diagnostics_hash: str | None = None
    candidate_chunk_span_summary_hash: str | None = None
    structure_context_status_hash: str | None = None

    @model_serializer(mode="wrap")
    def serialize_replay_identity_fields(
        self,
        handler: Any,
    ) -> dict[str, Any]:
        payload = handler(self)
        return {
            key: value
            for key, value in payload.items()
            if key in self.__pydantic_fields_set__
        }


class RetrievalGrayZoneDecision(ClosedContractModel):
    layer: str | None = None
    edge_id: str | None = None
    from_node_id: str | None = None
    to_node_id: str | None = None
    from_chunk_id: str | None = None
    to_chunk_id: str | None = None
    path_distance: float
    distance_zone: Literal["green", "gray", "red", "hard_stop"]
    decision: str
    decision_reason: str | None = None
    protocol_version: str
    protocol_hash: str
    input_hash: str
    threshold_hash: str
    traversal_protocol_hash: str
    runtime_settings_hash: str
    agent_operating_envelope_hash: str
    decision_hash: str
    matched_rule: str
    predicates: dict[str, bool] = Field(default_factory=dict)
    minimum_audit: GrayMinimumAudit
    observation: GrayObservation | None = None
    observation_compacted: bool = False
    hard_interrupt_state: GrayHardInterruptState
    model_call_count: Literal[0]
    decision_source: Literal["deterministic_local_rule", "deterministic_distance_partition"]
    support_progress: dict[str, bool | int | float | str] = Field(default_factory=dict)
    support_refs: RetrievalSupportRefs
    covered_facets: list[str] = Field(default_factory=list)
    semantic_uncertain_edge: bool = False
    crossing_rq_boundary: bool = False
    edge_type: str | None = None
    gray_candidate_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_complete_deterministic_audit(self) -> "RetrievalGrayZoneDecision":
        required_strings = {
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "input_hash": self.input_hash,
            "threshold_hash": self.threshold_hash,
            "traversal_protocol_hash": self.traversal_protocol_hash,
            "runtime_settings_hash": self.runtime_settings_hash,
            "agent_operating_envelope_hash": self.agent_operating_envelope_hash,
            "decision_hash": self.decision_hash,
            "matched_rule": self.matched_rule,
        }
        missing = [name for name, value in required_strings.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"gray-zone audit strings cannot be empty: {', '.join(sorted(missing))}")
        hash_fields = {
            name: str(value)
            for name, value in required_strings.items()
            if name.endswith("_hash")
        }
        invalid_hashes = [
            name
            for name, value in hash_fields.items()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower())
        ]
        if invalid_hashes:
            raise ValueError(f"gray-zone audit hashes must be canonical SHA-256 hex: {', '.join(sorted(invalid_hashes))}")
        if not self.minimum_audit:
            raise ValueError("gray-zone minimum_audit must be persisted and non-empty")
        if not self.hard_interrupt_state:
            raise ValueError("gray-zone hard_interrupt_state must be persisted and non-empty")
        if not self.support_refs:
            raise ValueError("gray-zone support_refs must be persisted and non-empty")

        local_outcomes = {
            "1_support_or_drift_stop": "stop_path_irrelevant",
            "2_structure_closure": "request_structure_closure",
            "3_supported_bridge": "follow_as_bridge",
            "4_supported_drilldown": "drill_down_layer",
            "5_support_progress": "continue_path",
            "6_no_progress_stop": "stop_path_irrelevant",
        }
        if self.decision_source == "deterministic_local_rule":
            if self.protocol_version != "deterministic_support_progress_v1":
                raise ValueError("local gray-zone decision has an unsupported persisted protocol")
            if (
                self.minimum_audit.input_identity_protocol_version
                != "gray_zone_minimum_replay_card_v1"
            ):
                raise ValueError(
                    "local gray-zone decision is missing its minimum replay-card protocol"
                )
            if (
                self.minimum_audit.current_layer != self.layer
                or self.minimum_audit.distance_zone != self.distance_zone
                or self.minimum_audit.path_distance != self.path_distance
                or self.minimum_audit.predicates != self.predicates
            ):
                raise ValueError(
                    "local gray-zone minimum replay card drifts from the decision record"
                )
            if self.distance_zone not in {"green", "gray"}:
                raise ValueError("local gray-zone rule may only audit gray candidates in the green/gray distance partitions")
            expected = local_outcomes.get(self.matched_rule)
            if expected is None or self.decision != expected:
                raise ValueError("local gray-zone matched_rule and decision are inconsistent")
            reasons = set(self.gray_candidate_reasons)
            if self.distance_zone == "gray":
                if "distance_gray" not in reasons:
                    raise ValueError("distance-gray local decision is missing its gray candidate reason")
            elif not (
                (self.semantic_uncertain_edge and "semantic_uncertain" in reasons)
                or (self.crossing_rq_boundary and "crossing_rq_boundary" in reasons)
            ):
                raise ValueError("green local decision must be semantic-uncertain or cross an RQ boundary")
        else:
            if self.protocol_version != "deterministic_path_distance_partition_v2":
                raise ValueError("distance partition audit has an unsupported persisted protocol")
            path_contribution_key = str(self.minimum_audit.path_contribution_key or "").lower()
            if (
                len(path_contribution_key) != 64
                or any(character not in "0123456789abcdef" for character in path_contribution_key)
            ):
                raise ValueError("distance partition audit must bind a canonical physical path contribution key")
            expected = {
                "red": ("distance_red_zone", "red_zone_pruned"),
                "hard_stop": ("distance_hard_stop", "hard_stop_pruned"),
            }.get(self.distance_zone)
            if expected is None or (self.matched_rule, self.decision) != expected:
                raise ValueError("distance partition matched_rule and decision are inconsistent")
        return self

    @model_serializer(mode="wrap")
    def serialize_exact_replay_cards(
        self,
        handler: Any,
    ) -> dict[str, Any]:
        payload = handler(self)
        payload["minimum_audit"] = self.minimum_audit.model_dump(
            mode="json",
            exclude_unset=True,
        )
        payload["hard_interrupt_state"] = (
            self.hard_interrupt_state.model_dump(
                mode="json",
                exclude_unset=True,
            )
        )
        payload["support_refs"] = self.support_refs.model_dump(
            mode="json",
            exclude_unset=True,
        )
        if self.observation is not None:
            payload["observation"] = self.observation.model_dump(
                mode="json",
                exclude_unset=True,
            )
        return payload


class GrayAuditFinding(ClosedContractModel):
    code: str
    message: str | None = None
    field: str | None = None
    expected: int | str | None = None
    actual: int | str | None = None
    record_index: int | None = None


class RetrievalGrayZoneDeterminismAudit(ClosedContractModel):
    status: Literal["passed", "incomplete", "failed"]
    checked_record_count: int = Field(ge=0)
    unique_record_count: int = Field(ge=0)
    local_rule_record_count: int = Field(ge=0)
    red_partition_record_count: int = Field(ge=0)
    hard_stop_partition_record_count: int = Field(ge=0)
    duplicate_reference_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    incomplete_record_count: int = Field(ge=0)
    conflicts: list[GrayAuditFinding] = Field(default_factory=list)
    issues: list[GrayAuditFinding] = Field(default_factory=list)


class TraversalObservationLayerCounts(ClosedContractModel):
    local_rule_evaluation_count: int = Field(ge=0)
    expanded_request_count: int = Field(ge=0)
    expanded_observation_count: int = Field(ge=0)
    cadence_compacted_count: int = Field(ge=0)
    budget_compacted_count: int = Field(ge=0)


class TraversalObservationBudgetAudit(ClosedContractModel):
    protocol_version: str
    scope: str
    limit: int = Field(ge=1)
    local_rule_evaluation_count: int = Field(ge=0)
    expanded_request_count: int = Field(ge=0)
    expanded_observation_count: int = Field(ge=0)
    cadence_compacted_count: int = Field(ge=0)
    budget_compacted_count: int = Field(ge=0)
    compacted_observation_count: int = Field(ge=0)
    hard_interrupt_count: int = Field(ge=0)
    budget_hit: bool
    traversal_expanded_observation_count: int = Field(ge=0)
    remaining: int = Field(ge=0)
    model_call_count: Literal[0]
    stop_reason: str
    per_layer: dict[str, TraversalObservationLayerCounts] = Field(default_factory=dict)


class EdgeReuseHardInterrupt(ClosedContractModel):
    protocol_version: str
    hard_interrupt: str
    decision: str
    layer: str
    edge_id: str
    from_node_id: str
    to_node_id: str
    used_count: int = Field(ge=0)
    attempted_count: int = Field(ge=0)
    limit: int = Field(ge=0)
    path_edge_ids: list[str]
    state_signature: RetrievalStateSignature


class RetrievalGranularityAudit(ClosedContractModel):
    retrieval_granularity: RetrievalGranularity
    coarse_skipped_reason: str | None = None
    mid_direct_entry_count: int = Field(ge=0)
    mid_entry_mode: Literal["direct_mid", "coarse_drilldown"]


class RetrievalQueryRQSeedAudit(ClosedContractModel):
    protocol_version: Literal[QUERY_RQ_SEED_PROTOCOL_VERSION]
    protocol_hash: Literal[QUERY_RQ_SEED_PROTOCOL_HASH]
    requested_query_rq_scores: dict[
        str,
        Annotated[
            float,
            Field(
                strict=True,
                ge=0.0,
                le=1.0,
                allow_inf_nan=False,
            ),
        ],
    ] = Field(
        default_factory=dict
    )
    effective_rq_scores: dict[
        str,
        Annotated[
            float,
            Field(
                strict=True,
                ge=0.0,
                le=1.0,
                allow_inf_nan=False,
            ),
        ],
    ] = Field(
        default_factory=dict
    )
    explicit_query_relevance_precedence: Literal[True]
    selected_mid_route_fallback_only_when_missing: Literal[True]
    mid_support_baseline_may_mask_rq_seed: Literal[False]
    membership_overlap_used_in_effective_score: Literal[False]
    node_weight_used_as_query_relevance: Literal[False]
    hard_path_lcp_used_as_score: Literal[False]
    is_evidence: Literal[False]
    gray_zone_decision_authority: Literal[False]
    model_call_count: Literal[0]

    @model_validator(mode="after")
    def validate_score_scope(self) -> "RetrievalQueryRQSeedAudit":
        if set(self.requested_query_rq_scores).difference(
            self.effective_rq_scores
        ):
            raise ValueError(
                "requested Query -> RQ scores escape effective score scope"
            )
        return self


class RetrievalConvergence(ClosedContractModel):
    reason: str | None = None
    convergence_replay_protocol_version: str | None = None
    entry_count: int | None = Field(default=None, ge=0)
    frontier_remaining_count: int | None = Field(default=None, ge=0)
    frontier_budget: int | None = Field(default=None, ge=0)
    frontier_expansion_count: int = 0
    dominance_pruned_count: int = 0
    label_budget_pruned_count: int = 0
    hard_stop_pruned_count: int = 0
    red_zone_pruned_count: int = 0
    path_distance_partition_event_count: int = 0
    gray_zone_decision_count: int = 0
    gray_zone_rule_protocol_version: str | None = None
    gray_zone_rule_protocol_hash: str | None = None
    gray_zone_rule_evaluation_count: int = 0
    gray_zone_rule_stop_count: int = 0
    gray_zone_observation_compacted_count: int = 0
    gray_zone_model_call_count: Literal[0] = 0
    gray_zone_observation_cadence: int | None = None
    traversal_observation_budget: int | None = None
    traversal_observation_expanded_count: int = 0
    traversal_observation_budget_compacted_count: int = 0
    traversal_observation_cadence_compacted_count: int = 0
    traversal_observation_hard_interrupt_count: int = 0
    traversal_observation_budget_hit: bool = False
    traversal_observation_budget_audit: TraversalObservationBudgetAudit | None = None
    query_facet_posterior_protocol_version: Literal[
        "query_facet_posterior_calibration_v1"
    ] | None = None
    query_facet_posterior_protocol_hash: str | None = None
    query_facet_posterior_rounds_used: int = Field(default=0, ge=0, le=2)
    query_facet_posterior_observations_used: int = Field(default=0, ge=0)
    query_facet_posterior_stop_reason: str | None = None
    query_facet_posterior_model_call_count: Literal[0] | None = None
    edge_reuse_pruned_count: int = 0
    duplicate_transition_pruned_count: int = Field(
        default=0,
        ge=0,
    )
    duplicate_transition_protocol_version: str | None = None
    frontier_remaining: int = 0
    accepted_node_count: int | None = None
    accepted_chunk_count: int | None = None
    per_entry_expansion_budget: int | None = None
    expansion_count_by_entry: dict[str, int] = Field(default_factory=dict)
    path_distance_thresholds: dict[str, float | None] = Field(default_factory=dict)
    layers: dict[str, "RetrievalConvergence"] = Field(default_factory=dict)
    max_edge_reuse: int | None = Field(default=None, ge=0)
    edge_reuse_hard_interrupt_protocol_version: str | None = None
    edge_reuse_hard_interrupts: list[EdgeReuseHardInterrupt] = Field(default_factory=list)
    edge_reuse_observation_compacted: bool = False
    cycle_distance_reward_bounded: bool | None = None
    candidate_pool_dedupe_budget_audit: CandidatePoolDedupeAudit | None = None
    candidate_pool_dedupe_budget: CandidatePoolDedupeSummary | None = None
    query_rq_seed_protocol_version: str | None = None
    query_rq_seed_protocol_hash: str | None = None
    query_rq_seed_model_call_count: Literal[0] | None = None
    query_rq_seed_gray_zone_decision_authority: Literal[False] | None = None
    query_rq_seed_node_weight_used_as_query_relevance: (
        Literal[False] | None
    ) = None
    query_rq_seed_hard_lcp_used_as_score: Literal[False] | None = None
    query_relevance_overwritten_by_mid_route_prior: (
        Literal[False] | None
    ) = None
    node_weight_used_as_query_relevance: Literal[False] | None = None
    gray_zone_decision_authority: Literal[False] | None = None
    model_call_count: Literal[0] | None = None
    seed_count: int | None = Field(default=None, ge=0)
    active_traversal_layer: bool | None = None
    skipped_by_granularity: str | None = None
    runtime_settings_hash: str | None = None
    agent_operating_envelope_hash: str | None = None
    traversal_protocol_hash: str | None = None
    repair_protocol_version: str | None = None
    repair_action_type: str | None = None
    repair_executor_mechanism: str | None = None
    repair_directive_hash: str | None = None
    repair_excluded_mid_count: int = Field(default=0, ge=0)
    repair_excluded_result_chunk_count: int = Field(default=0, ge=0)
    repair_bridge_seed_count: int = Field(default=0, ge=0)
    repair_carry_forward_supported_chunk_count: int = Field(default=0, ge=0)
    repair_global_top_k_modified: Literal[False] = False
    repair_gray_zone_rule_inputs_modified: Literal[False] = False
    retrieval_granularity: RetrievalGranularity | None = None
    granularity_audit: RetrievalGranularityAudit | None = None
    allowed_relation_types: list[str] = Field(default_factory=list)
    allowed_relation_types_source: Literal[
        "request_scoped_typed_action_control",
        "frozen_agent_operating_envelope",
    ] | None = None


class QueryFacetGroup(ClosedContractModel):
    facet: str
    role: str
    aliases: list[str]
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class QueryFacetSchemaRejection(ClosedContractModel):
    reason: str
    fields: list[str] = Field(default_factory=list)


class QueryFacetDiagnostics(ClosedContractModel):
    source: str
    schema_validation: Literal["canonical_facet_groups_only"]
    query_facet_protocol_hash: str
    lexical_terms: list[str]
    dropped_query_terms: list[str]
    llm_keys: list[str]
    bilingual_query_facets_enabled: bool | None = None
    output_contract_protocol_version: Literal[
        "query_facet_nonempty_output_contract_v2"
    ] | None = None
    sampling_model_call_count: Literal[1, 2] | None = None
    schema_repair_attempted: bool = False
    schema_repair_protocol_version: Literal[
        "query_facet_empty_group_schema_repair_v1"
    ] | None = None
    llm_schema_rejection: QueryFacetSchemaRejection | None = None
    query_perception_audit: OrdinaryQueryPerceptionAudit | None = None

    @model_validator(mode="after")
    def validate_schema_repair_diagnostics(self) -> "QueryFacetDiagnostics":
        if self.output_contract_protocol_version is None:
            if (
                self.sampling_model_call_count is not None
                or self.schema_repair_attempted
                or self.schema_repair_protocol_version is not None
            ):
                raise ValueError(
                    "query facet schema-repair diagnostics require an output contract version"
                )
            return self
        expected_calls = 2 if self.schema_repair_attempted else 1
        if self.sampling_model_call_count != expected_calls:
            raise ValueError(
                "query facet sampling call count must match schema repair state"
            )
        if self.schema_repair_attempted != (
            self.schema_repair_protocol_version is not None
        ):
            raise ValueError(
                "query facet schema repair protocol must match schema repair state"
            )
        return self


class QueryFacetPacket(ClosedContractModel):
    query: str
    protocol_version: Literal["query_facet_packet_v2"]
    terms: list[str]
    required_facets: list[str]
    facet_groups: list[QueryFacetGroup]
    drop_terms: list[str]
    answer_shape: str
    intent: str
    diagnostics: QueryFacetDiagnostics


class QueryFacetPosteriorObservation(ClosedContractModel):
    protocol_version: Literal["query_facet_posterior_calibration_v1"]
    checkpoint: Literal[
        "dense_entry_candidates",
        "merged_chunk_candidates",
    ]
    layer: Literal["chunk"]
    scope: str
    candidate_id: str
    matched_facets: list[str]
    matched_term_witnesses: dict[str, list[str]]
    query_facet_packet_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    candidate_business_input_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    model_call_count: Literal[0]
    observation_id: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    observation_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )


class QueryFacetPosteriorRound(ClosedContractModel):
    round_index: int = Field(ge=0, le=1)
    checkpoint: Literal[
        "dense_entry_candidates",
        "merged_chunk_candidates",
    ]
    prior: dict[str, float]
    likelihood: dict[str, float]
    posterior: dict[str, float]
    alias_prior: dict[str, dict[str, float]]
    alias_likelihood: dict[str, dict[str, float]]
    alias_posterior: dict[str, dict[str, float]]
    observation_count: int = Field(ge=0)
    facet_match_counts: dict[str, int]
    l1_delta: float = Field(ge=0.0, le=2.0, allow_inf_nan=False)
    model_call_count: Literal[0]
    observation_ids: list[str]
    round_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )


class QueryFacetPosteriorCalibration(ClosedContractModel):
    protocol_version: Literal["query_facet_posterior_calibration_v1"]
    protocol_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    enabled: bool
    query_facet_packet_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    required_facets: list[str]
    prior: dict[str, float]
    posterior: dict[str, float]
    alias_posterior: dict[str, dict[str, float]]
    rounds: list[QueryFacetPosteriorRound]
    observations: list[QueryFacetPosteriorObservation]
    round_budget: int = Field(ge=1, le=QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX)
    rounds_used: int = Field(ge=0, le=QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX)
    observation_budget: int = Field(
        ge=1,
        le=QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    )
    observations_used: int = Field(
        ge=0,
        le=QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    )
    convergence_epsilon: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False
    )
    converged: bool
    stop_reason: Literal[
        "disabled_no_required_facets",
        "disabled_by_runtime_setting",
        "converged",
        "round_budget_exhausted",
        "observation_budget_exhausted",
        "checkpoint_sequence_exhausted",
    ]
    llm_sample_budget: Literal[0]
    model_call_count: Literal[0]
    is_evidence: Literal[False]
    citation_authority: Literal[False]
    graph_mutation_authority: Literal[False]
    gray_zone_decision_authority: Literal[False]
    posterior_used_as_numeric_query_relevance: Literal[False]
    posterior_used_only_within_equal_uncovered_count: Literal[True]
    calibration_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_bounded_calibration(self) -> "QueryFacetPosteriorCalibration":
        if self.rounds_used != len(self.rounds):
            raise ValueError("query facet posterior round usage drifted")
        if self.observations_used != len(self.observations):
            raise ValueError("query facet posterior observation usage drifted")
        if self.required_facets:
            if set(self.posterior) != set(self.required_facets):
                raise ValueError("query facet posterior facet scope drifted")
            if not math.isclose(
                sum(self.posterior.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("query facet posterior mass must sum to one")
        elif self.posterior:
            raise ValueError("empty facet scope cannot carry posterior mass")
        if len({item.observation_id for item in self.observations}) != len(
            self.observations
        ):
            raise ValueError("query facet posterior observation ids repeat")
        return self


class RetrievalRQDiagnostics(ClosedContractModel):
    query_rq_path: list[int] = Field(default_factory=list)
    query_residual_norm: float | None = None
    index_protocol: str | None = None


class RetrievalAgentOperatingEnvelope(ClosedContractModel):
    """Provider-free request-frozen executor envelope exposed for replay."""

    agent_coarse_initial_budget: int = Field(ge=1)
    agent_coarse_total_budget: int = Field(ge=1)
    agent_coarse_top_k: int = Field(ge=1)
    agent_mid_per_coarse_budget: int = Field(ge=1)
    agent_coarse_drilldown_mid_initial_budget: int = Field(ge=1)
    agent_mid_initial_budget: int = Field(ge=1)
    agent_mid_top_k: int = Field(ge=1)
    agent_chunk_per_mid_budget: int = Field(ge=1)
    agent_chunk_initial_budget: int = Field(ge=1)
    agent_chunk_top_k: int = Field(ge=1)
    max_depth_per_layer: int = Field(ge=1)
    max_labels_per_node: int = Field(ge=1)
    max_edge_reuse: int = Field(ge=1)
    max_cycle_reward_per_path: float = Field(ge=0, allow_inf_nan=False)
    cycle_reward_distance_threshold: float = Field(ge=0, allow_inf_nan=False)
    path_distance_green_threshold: float = Field(ge=0, allow_inf_nan=False)
    path_distance_gray_threshold: float = Field(ge=0, allow_inf_nan=False)
    path_distance_hard_threshold: float = Field(ge=0, allow_inf_nan=False)
    gray_zone_rule_protocol_version: Literal[
        "deterministic_support_progress_v1"
    ]
    gray_zone_rule_protocol_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    gray_zone_observation_cadence: int = Field(ge=1, le=16)
    traversal_observation_budget: int = Field(
        ge=1, le=TRAVERSAL_OBSERVATION_BUDGET_MAX
    )
    gray_zone_model_call_budget: Literal[0]
    query_facet_posterior_enabled: bool | None = None
    query_facet_posterior_observation_budget: int | None = Field(
        default=None,
        ge=1,
        le=QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    )
    query_facet_posterior_round_budget: int | None = Field(
        default=None,
        ge=1,
        le=QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX,
    )
    query_facet_posterior_convergence_epsilon: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    query_facet_posterior_llm_sample_budget: Literal[0] | None = None
    candidate_pool_dedupe_budget: int = Field(ge=1)
    structure_restore_per_chunk_budget: int = Field(ge=1)
    structure_restore_budget: int = Field(ge=1)
    context_package_token_budget: int = Field(ge=1)
    context_path_summary_budget: int = Field(ge=1)
    planning_round_budget: int = Field(ge=1)
    max_typed_actions_per_round: int = Field(ge=1)
    repair_round_budget: int = Field(ge=0)
    verification_budget: int = Field(ge=0)
    allowed_relation_types: list[
        Literal[
            "dense_semantic",
            "dense_cross_document_bridge",
            "dense_cross_language_bridge",
        ]
    ]
    required_restore_modes: list[
        Literal["previous_next", "parent_structure", "bridge_chunks"]
    ]

    @model_validator(mode="after")
    def validate_provider_free_executor_envelope(
        self,
    ) -> "RetrievalAgentOperatingEnvelope":
        if not (
            self.path_distance_green_threshold
            <= self.path_distance_gray_threshold
            <= self.path_distance_hard_threshold
        ):
            raise ValueError(
                "frozen path-distance thresholds must satisfy green <= gray <= hard"
            )
        if self.structure_restore_budget != self.structure_restore_per_chunk_budget:
            raise ValueError(
                "frozen structure restore aliases must resolve to one per-chunk budget"
            )
        if self.allowed_relation_types != [
            "dense_semantic",
            "dense_cross_document_bridge",
            "dense_cross_language_bridge",
        ]:
            raise ValueError("frozen allowed relation types must match the executor allowlist")
        if self.required_restore_modes != [
            "previous_next",
            "parent_structure",
            "bridge_chunks",
        ]:
            raise ValueError("frozen restore modes must match the executor allowlist")
        posterior_fields = (
            self.query_facet_posterior_enabled,
            self.query_facet_posterior_observation_budget,
            self.query_facet_posterior_round_budget,
            self.query_facet_posterior_convergence_epsilon,
            self.query_facet_posterior_llm_sample_budget,
        )
        if any(value is not None for value in posterior_fields) and any(
            value is None for value in posterior_fields
        ):
            raise ValueError(
                "query facet posterior envelope fields must be all present or all absent"
            )
        return self


class RetrievalTraceDiagnostics(ClosedContractModel):
    context_graph_state_id: str | None = None
    retrieval_granularity: RetrievalGranularity = "mid"
    active_profile_hash: str | None = None
    canonical_active_profile_hash: str | None = None
    repair_protocol_version: str | None = None
    repair_action_type: str | None = None
    repair_executor_mechanism: str | None = None
    repair_directive_hash: str | None = None
    repair_gray_zone_decision_authority: Literal[False] = False
    repair_gray_zone_model_call_count: Literal[0] = 0
    coarse_skipped_reason: str | None = None
    runtime_settings_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    gray_zone_runtime_settings_identity_protocol_version: Literal[
        "gray_zone_runtime_settings_identity_v1"
    ]
    gray_zone_runtime_settings_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    gray_zone_query_facet_protocol_version: Literal[
        "deterministic_gray_query_tokenizer_v1"
    ]
    gray_zone_query_facet_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    gray_zone_external_routing_packet_used: Literal[False]
    gray_zone_request_scoped_budget_in_identity: Literal[False]
    agent_operating_envelope: RetrievalAgentOperatingEnvelope
    agent_operating_envelope_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    effective_traversal_protocol_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    result_top_k: int | None = Field(default=None, ge=1)
    scores_json_retired_as_primary_audit: Literal[True] = True
    traversal_protocol: str | None = None
    rank_score_protocol_version: str | None = None
    rank_score_protocol_hash: str | None = None
    raw_strength_protocol_version: str | None = None
    raw_strength_protocol_hash: str | None = None
    chunk_node_quality_protocol_version: str | None = None
    chunk_node_quality_protocol_hash: str | None = None
    out_evidence_mass_protocol_version: str | None = None
    out_evidence_mass_protocol_hash: str | None = None
    in_acceptance_capacity_protocol_version: str | None = None
    in_acceptance_capacity_protocol_hash: str | None = None
    relation_quota_protocol_version: str | None = None
    relation_quota_protocol_hash: str | None = None
    edge_type_calibration_protocol_version: str | None = None
    edge_type_calibration_protocol_hash: str | None = None
    graph_operating_point_hash: str | None = None
    calibration_params_hash: str | None = None
    edge_type_calibration_config_hash: str | None = None
    edge_distance_protocol_version: str | None = None
    edge_distance_protocol_hash: str | None = None
    entry_selection_protocol_version: str | None = None
    entry_selection_protocol_hash: str | None = None
    entry_topology_protocol_version: str | None = None
    entry_topology_protocol_hash: str | None = None
    entry_replay_proof_protocol_version: str | None = None
    entry_dense_replay_protocol_version: str | None = None
    entry_dense_replay_input_hash: str | None = None
    entry_neutral_start_cost_protocol_version: str | None = None
    entry_selection_model_call_count: Literal[0] = 0
    entry_selection_lexical_overlap_used_as_query_relevance: Literal[False] = False
    entry_selection_topology_used_as_path_distance: Literal[False] = False
    entry_selection_node_weight_used_as_query_relevance: Literal[False] = False
    entry_neutral_start_cost_is_query_relevance: Literal[False] = False
    entry_selection_gray_zone_rule_inputs_modified: Literal[False] = False
    query_rq_seed_audit: RetrievalQueryRQSeedAudit | None = None
    query_facet_posterior_calibration: QueryFacetPosteriorCalibration | None = None

    @model_validator(mode="after")
    def validate_gray_runtime_identity_is_separate(
        self,
    ) -> "RetrievalTraceDiagnostics":
        if hmac.compare_digest(
            self.runtime_settings_hash,
            self.gray_zone_runtime_settings_hash,
        ):
            raise ValueError(
                "broad runtime settings hash must remain separate from the provider-free gray identity"
            )
        if self.query_facet_posterior_calibration is not None:
            envelope = self.agent_operating_envelope
            required_values = (
                envelope.query_facet_posterior_enabled,
                envelope.query_facet_posterior_observation_budget,
                envelope.query_facet_posterior_round_budget,
                envelope.query_facet_posterior_convergence_epsilon,
                envelope.query_facet_posterior_llm_sample_budget,
            )
            if any(value is None for value in required_values):
                raise ValueError(
                    "current retrieval trace requires the complete query facet posterior envelope"
                )
            calibration = self.query_facet_posterior_calibration
            if (
                calibration.enabled
                is not envelope.query_facet_posterior_enabled
                or calibration.observation_budget
                != envelope.query_facet_posterior_observation_budget
                or calibration.round_budget
                != envelope.query_facet_posterior_round_budget
                or calibration.convergence_epsilon
                != envelope.query_facet_posterior_convergence_epsilon
                or calibration.llm_sample_budget
                != envelope.query_facet_posterior_llm_sample_budget
            ):
                raise ValueError(
                    "query facet posterior calibration does not match the frozen retrieval envelope"
                )
        return self


class RetrievalStepInput(ClosedContractModel):
    entry_node_ids: list[str] = Field(default_factory=list)
    coarse_entry_ids: list[str] = Field(default_factory=list)
    mid_entry_ids: list[str] = Field(default_factory=list)
    rq_membership_entry_ids: list[str] = Field(default_factory=list)
    query_rq_path: list[int] = Field(default_factory=list)
    result_chunk_ids: list[str] = Field(default_factory=list)
    hit_chunk_ids: list[str] = Field(default_factory=list)
    token_budget: int | None = Field(default=None, ge=0)
    query_rq_seed_audit: RetrievalQueryRQSeedAudit | None = None


class RetrievalStepOutput(ClosedContractModel):
    accepted_node_ids: list[str] = Field(default_factory=list)
    selected_node_ids: list[str] = Field(default_factory=list)
    accepted_chunk_ids: list[str] = Field(default_factory=list)
    restored_chunk_count: int | None = Field(default=None, ge=0)
    context_package_id: str | None = None
    source_span_count: int = Field(default=0, ge=0)
    convergence_reason: str | None = None


class RetrievalStepDiagnostics(ClosedContractModel):
    retrieval_granularity: RetrievalGranularity = "mid"
    traversal_protocol: str | None = None
    scores_json_retired_as_primary_audit: Literal[True] = True


class GraphRetrievalStepResponse(ClosedContractModel):
    id: str
    step_index: int
    layer: str
    action: str
    action_type: str
    parent_layer: str | None = None
    parent_node_id: str | None = None
    input: RetrievalStepInput = Field(default_factory=RetrievalStepInput)
    output: RetrievalStepOutput = Field(default_factory=RetrievalStepOutput)
    candidate_pool_ids: list[str] = Field(default_factory=list)
    selected_topk_ids: list[str] = Field(default_factory=list)
    per_parent_budget_status: dict[str, PerParentBudgetStatus] = Field(default_factory=dict)
    popped_frontier_state: RetrievalTraversalState = Field(default_factory=RetrievalTraversalState)
    expanded_edge_ids: list[str] = Field(default_factory=list)
    dominance_pruned_count: int = 0
    cycle_distance_reward: float = 0.0
    gray_zone_path_decisions: list[RetrievalGrayZoneDecision] = Field(default_factory=list)
    stop_reason: str | None = None
    diagnostics: RetrievalStepDiagnostics
    created_at: datetime | None = None


class ContextConceptPathEntry(ClosedContractModel):
    layer: Literal["coarse", "mid", "rq_membership", "chunk"]
    ids: list[str]


class ContextSelectionReason(ClosedContractModel):
    roles: list[str] = Field(default_factory=list)
    path_edge_ids: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    reason: str
    reached_by_paths: list[RetrievalPathContribution] = Field(default_factory=list)
    query_facets: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    graph_paths: list[list[str]] = Field(default_factory=list)
    graph_path_chunks: list[str] = Field(default_factory=list)
    convergence_score: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    node_visit_count: int = Field(default=0, ge=0)
    distinct_parent_count: int = Field(default=0, ge=0)
    distinct_path_count: int = Field(default=0, ge=0)
    distinct_edge_type_count: int = Field(default=0, ge=0)
    parent_node_ids: list[str] = Field(default_factory=list)
    support_chunk_union: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_path_replay(self) -> "ContextSelectionReason":
        paths = self.reached_by_paths
        path_ids = [path.contribution_id for path in paths]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError(
                "why_selected reached_by_paths must be unique"
            )
        if path_ids != sorted(path_ids):
            raise ValueError(
                "why_selected reached_by_paths must use canonical order"
            )
        if self.query_facets != sorted(set(self.query_facets)):
            raise ValueError(
                "why_selected query_facets must be sorted and unique"
            )
        expected = retrieval_node_contribution_facts(paths)
        expected_edge_ids = list(
            dict.fromkeys(
                edge_id
                for path in paths
                for edge_id in path.path_edge_ids
            )
        )
        expected_graph_paths = [
            list(path.path) for path in paths
        ]
        strict_fields = {
            "convergence_score": expected[
                "cycle_convergence_score"
            ],
            "node_visit_count": expected["node_visit_count"],
            "distinct_parent_count": expected[
                "distinct_parent_count"
            ],
            "distinct_path_count": expected[
                "distinct_path_count"
            ],
            "distinct_edge_type_count": expected[
                "distinct_edge_type_count"
            ],
            "parent_node_ids": expected["parent_node_ids"],
            "support_chunk_union": expected[
                "support_chunk_union"
            ],
            "graph_paths": expected_graph_paths,
        }
        for field, expected_value in strict_fields.items():
            if getattr(self, field) != expected_value:
                raise ValueError(
                    f"why_selected {field} does not replay "
                    "reached_by_paths"
                )
        if expected_edge_ids and (
            self.path_edge_ids != expected_edge_ids
        ):
            raise ValueError(
                "why_selected path_edge_ids do not replay "
                "reached_by_paths"
            )
        if expected["covered_facets"] and (
            self.covered_facets
            != expected["covered_facets"]
        ):
            raise ValueError(
                "why_selected covered_facets do not replay "
                "reached_by_paths"
            )
        if expected["evidence_roles"]:
            if (
                self.evidence_roles
                != expected["evidence_roles"]
                or self.roles != expected["evidence_roles"]
            ):
                raise ValueError(
                    "why_selected evidence roles do not replay "
                    "reached_by_paths"
                )
        graph_path_nodes = {
            node_id
            for graph_path in expected_graph_paths
            for node_id in graph_path
        }
        if (
            self.graph_path_chunks
            != sorted(set(self.graph_path_chunks))
            or any(
                node_id not in graph_path_nodes
                for node_id in self.graph_path_chunks
            )
        ):
            raise ValueError(
                "why_selected graph_path_chunks must be a canonical "
                "subset of reached paths"
            )
        return self


class ContextStructurePathDiagnostics(ClosedContractModel):
    chunk_segments: list[str] = Field(default_factory=list)
    structure_segments: list[str] = Field(default_factory=list)
    matched_segments: list[str] = Field(default_factory=list)


class ContextStructureMappingAdmissionDiagnostics(ClosedContractModel):
    protocol_version: str
    decision: Literal["admit", "reject"]
    reason: str
    same_scope: bool
    span_available: bool
    span_positive: bool
    bbox_available: bool
    bbox_positive: bool
    path_available: bool
    path_positive: bool
    exact_canonical_section_path: bool
    exact_section_candidate_count: int = Field(ge=0)


class ContextStructureMappingDiagnostics(ClosedContractModel):
    mapping_protocol_version: str
    mapping_admission_protocol_version: str
    admission: ContextStructureMappingAdmissionDiagnostics
    component_weights: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    available_components: list[str] = Field(default_factory=list)
    unavailable_components: list[str] = Field(default_factory=list)
    chunk_layout_ids: list[str] = Field(default_factory=list)
    structure_layout_ids: list[str] = Field(default_factory=list)
    path_diagnostics: ContextStructurePathDiagnostics


class ContextStructureParserMetadataAudit(ClosedContractModel):
    layout_protocol_version: str | None = None
    flow_block_protocol_version: str | None = None
    block_start_protocol_version: str | None = None
    html_block_protocol_version: str | None = None
    link_reference_protocol_version: str | None = None
    target_spec_version: str | None = None
    source_type: str | None = None
    parser: str | None = None
    native_layout_available: bool | None = None
    page_size: list[float] | tuple[float, float] | None = None
    image_size: list[float] | tuple[float, float] | None = None
    cell_index: int | None = Field(default=None, ge=0)
    layout_item_count: int | None = Field(default=None, ge=0)
    structure_object_count: int | None = Field(default=None, ge=0)


class ContextStructureOCRLayoutItemAudit(ClosedContractModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: CitationBoundingBox


class ContextStructureNativeMetadataAudit(ClosedContractModel):
    native_structure: bool | None = None
    parent_ref: str | None = None
    parser_source: str | None = None
    native_geometry: bool | None = None
    layout_protocol_version: str | None = None
    flow_block_protocol_version: str | None = None
    block_start_protocol_version: str | None = None
    html_block_protocol_version: str | None = None
    link_reference_protocol_version: str | None = None
    target_spec_version: str | None = None
    structure_protocol_version: str | None = None
    row_normalization_protocol_version: str | None = None
    source_span_protocol: str | None = None
    indentation_protocol: str | None = None
    html_block_type: int | None = Field(default=None, ge=1, le=7)
    column_count: int | None = Field(default=None, ge=0)
    data_row_count: int | None = Field(default=None, ge=0)
    column_alignments: list[str] = Field(default_factory=list)
    source_body_column_counts: list[int] = Field(default_factory=list)
    padded_missing_cell_counts: list[int] = Field(default_factory=list)
    ignored_excess_cell_counts: list[int] = Field(default_factory=list)
    source_indent_columns: list[int] = Field(default_factory=list)
    html_tag: str | None = None
    source: str | None = None
    ocr_applied: bool | None = None
    ocr_engine: str | None = None
    ocr_confidence: float | None = None
    ocr_reason: str | None = None
    ocr_layout_items: list[ContextStructureOCRLayoutItemAudit] = Field(
        default_factory=list
    )
    image_size: list[float] | tuple[float, float] | None = None
    source_index: int | None = Field(default=None, ge=0)
    span_remap_method: str | None = None
    original_char_span: list[int] | tuple[int, int] | None = None
    section_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_image_size(self) -> "ContextStructureNativeMetadataAudit":
        if self.image_size is not None and len(self.image_size) != 2:
            raise ValueError("structure image_size must contain exactly two dimensions")
        return self


class ContextStructureLayoutAudit(ClosedContractModel):
    coordinate_system: str | None = None
    structure_id: str | None = None
    layout_id: str | None = None
    block_id: str | None = None
    block_type: str | None = None
    source: str | None = None
    reading_order: int | None = None
    order_index: int | None = None
    synthetic: bool | None = None
    source_path: str | None = None
    region_index: int | None = None
    layout_ids: list[str] = Field(default_factory=list)
    section_index: int | None = None
    block_index: int | None = None
    synthetic_page: bool | None = None
    layout_protocol_version: str | None = None
    content_flags: list[str] = Field(default_factory=list)
    content_kind: str | None = None
    encoding_detected: str | None = None
    encoding_coherence: float | None = Field(default=None, ge=0.0)
    encoding_used: str | None = None
    parser_path: str | None = None
    page_size: list[float] | tuple[float, float] | None = None
    native_layout_block_count: int | None = Field(default=None, ge=0)
    pdf_image_count: int | None = Field(default=None, ge=0)
    ocr_applied: bool | None = None
    ocr_page_count: int | None = Field(default=None, ge=0)
    ocr_reason: str | None = None
    mojibake_repaired: bool | None = None
    mojibake_score_before: float | None = Field(default=None, ge=0.0, le=1.0)
    mojibake_score_after: float | None = Field(default=None, ge=0.0, le=1.0)
    text_cleaning_flags: list[str] = Field(default_factory=list)
    parser_metadata: ContextStructureParserMetadataAudit = Field(
        default_factory=ContextStructureParserMetadataAudit
    )
    metadata: ContextStructureNativeMetadataAudit = Field(
        default_factory=ContextStructureNativeMetadataAudit
    )


class ContextStructureNodeAudit(ClosedContractModel):
    node_id: str
    node_type: str
    title: str | None = None
    path: str | None = None
    depth: int = Field(ge=0)
    page_number: int | None = None
    bbox: CitationBoundingBox = Field(default_factory=CitationBoundingBox)
    layout: ContextStructureLayoutAudit = Field(default_factory=ContextStructureLayoutAudit)
    mapping_role: str
    coverage_ratio: float
    span_overlap: float
    bbox_iou: float | None = None
    path_match: float | None = None
    mapping_weight: float
    mapping_protocol_version: str
    mapping_diagnostics: ContextStructureMappingDiagnostics


class ContextStructureClosure(ClosedContractModel):
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    parent_section_node_id: str | None = None
    same_page_region_node_ids: list[str] = Field(default_factory=list)
    table_formula_caption_node_ids: list[str] = Field(default_factory=list)
    code_block_node_ids: list[str] = Field(default_factory=list)
    bridge_chunk_ids: list[str] = Field(default_factory=list)
    parent_section: ContextStructureNodeAudit | None = None
    same_page_region: list[ContextStructureNodeAudit] = Field(default_factory=list)
    table_formula_caption: list[ContextStructureNodeAudit] = Field(default_factory=list)
    code_blocks: list[ContextStructureNodeAudit] = Field(default_factory=list)


class ContextPackageChunk(ClosedContractModel):
    contract_kind: Literal["context_chunk"] = "context_chunk"
    chunk_id: str
    document_id: str
    document_version_id: str
    document_title: str
    source_path: str
    logical_source_path: str
    content: str
    content_clipped: bool
    content_token_count: int = Field(ge=0)
    original_token_count: int = Field(ge=0)
    raw_chunk_char_span: list[int] | tuple[int, int]
    chunk_text_hash_protocol_version: str
    chunk_text_hash: str
    raw_span_text_hash_protocol_version: str
    raw_span_text_hash: str
    section_path: str | list[str] | None = None
    structure_path: str | list[str] | None = None
    structure_node_ids: list[str] = Field(default_factory=list)
    structure_nodes: list[ContextStructureNodeAudit] = Field(default_factory=list)
    parent_section: ContextStructureNodeAudit | None = None
    page_range: list[int | None] | tuple[int | None, int | None]
    char_span: list[int] | tuple[int, int]
    bbox: CitationBoundingBox | None = None
    source_span: CitationSourceSpan
    structure_closure: ContextStructureClosure
    why_selected: ContextSelectionReason
    dedupe_key: str
    role: Literal["hit", "bridge", "graph_path", "restored_context"]
    context_package_id: str

    @model_validator(mode="after")
    def validate_source_span_projection(self) -> "ContextPackageChunk":
        span = self.source_span
        pairs = (
            ("chunk_id", self.chunk_id, span.chunk_id),
            (
                "document_version_id",
                self.document_version_id,
                span.document_version_id,
            ),
            ("source_path", self.source_path, span.source_path),
            (
                "logical_source_path",
                self.logical_source_path,
                span.logical_source_path,
            ),
            (
                "chunk_text_hash_protocol_version",
                self.chunk_text_hash_protocol_version,
                span.chunk_text_hash_protocol_version,
            ),
            ("chunk_text_hash", self.chunk_text_hash, span.chunk_text_hash),
            (
                "raw_span_text_hash_protocol_version",
                self.raw_span_text_hash_protocol_version,
                span.raw_span_text_hash_protocol_version,
            ),
            (
                "raw_span_text_hash",
                self.raw_span_text_hash,
                span.raw_span_text_hash,
            ),
            ("char_span", list(self.char_span), list(span.char_span)),
            (
                "raw_chunk_char_span",
                list(self.raw_chunk_char_span),
                (
                    list(span.raw_chunk_char_span)
                    if span.raw_chunk_char_span is not None
                    else None
                ),
            ),
            ("page_range", list(self.page_range), list(span.page_range)),
            ("section_path", self.section_path, span.section_path),
            ("structure_path", self.structure_path, span.structure_path),
            (
                "structure_node_ids",
                self.structure_node_ids,
                span.structure_node_ids,
            ),
            ("bbox", self.bbox, span.bbox),
            ("content_clipped", self.content_clipped, span.content_clipped),
            (
                "content_token_count",
                self.content_token_count,
                span.content_token_count,
            ),
            (
                "context_package_id",
                self.context_package_id,
                span.context_package_id,
            ),
        )
        for field_name, outer, nested in pairs:
            if outer != nested:
                raise ValueError(
                    f"context package chunk {field_name} does not match "
                    "its source span"
                )
        return self


class ContextPackageDocument(ClosedContractModel):
    contract_version: Literal["context_package_chunks_v1"] = "context_package_chunks_v1"
    chunks: list[ContextPackageChunk]


class ContextMetadata(ClosedContractModel):
    source_path: str
    logical_source_path: str
    section_path: str | list[str] | None = None
    structure_path: str | list[str] | None = None
    parent_section_node_id: str | None = None
    parent_section: ContextStructureNodeAudit | None = None
    structure_node_ids: list[str] = Field(default_factory=list)
    page_range: list[int | None] | tuple[int | None, int | None]
    char_span: list[int] | tuple[int, int]
    bbox: CitationBoundingBox | None = None
    source_span: CitationSourceSpan
    structure_closure: ContextStructureClosure
    why_selected: ContextSelectionReason
    dedupe_key: str
    role: Literal["hit", "bridge", "graph_path", "restored_context"]
    content_clipped: bool
    content_token_count: int = Field(ge=0)
    original_token_count: int = Field(ge=0)
    raw_chunk_char_span: list[int] | tuple[int, int]
    context_package_id: str


class ContextItem(ClosedContractModel):
    contract_kind: Literal["context_item"] = "context_item"
    chunk_id: str
    document_title: str
    source_path: str
    content: str
    snippet: str
    metadata: ContextMetadata


class ContextCitationSpan(ClosedContractModel):
    contract_kind: Literal["citation_span"] = "citation_span"
    document_id: str
    document_title: str
    source_path: str
    logical_source_path: str
    section_path: str | list[str] | None = None
    structure_path: str | list[str] | None = None
    structure_node_ids: list[str] = Field(default_factory=list)
    structure_closure: ContextStructureClosure
    source_span: CitationSourceSpan

    @model_validator(mode="after")
    def validate_source_span_projection(self) -> "ContextCitationSpan":
        span = self.source_span
        pairs = (
            ("source_path", self.source_path, span.source_path),
            (
                "logical_source_path",
                self.logical_source_path,
                span.logical_source_path,
            ),
            ("section_path", self.section_path, span.section_path),
            ("structure_path", self.structure_path, span.structure_path),
            (
                "structure_node_ids",
                self.structure_node_ids,
                span.structure_node_ids,
            ),
        )
        for field_name, outer, nested in pairs:
            if outer != nested:
                raise ValueError(
                    f"context citation {field_name} does not match its source span"
                )
        return self


class ConceptPathExpansion(ClosedContractModel):
    kind: Literal["concept_path"]
    path: list[ContextConceptPathEntry]


class GraphEdgeExpansion(ClosedContractModel):
    kind: Literal["graph_path_ids"]
    edge_ids: list[str]


class RestoredChunksExpansion(ClosedContractModel):
    kind: Literal["restored_chunks"]
    chunk_ids: list[str]


class BridgeChunksExpansion(ClosedContractModel):
    kind: Literal["bridge_chunks"]
    chunk_ids: list[str]


class ParentStructureNodesExpansion(ClosedContractModel):
    kind: Literal["parent_structure_nodes"]
    node_ids: list[str]


ContextGraphExpansionPath = Annotated[
    ConceptPathExpansion
    | GraphEdgeExpansion
    | RestoredChunksExpansion
    | BridgeChunksExpansion
    | ParentStructureNodesExpansion,
    Field(discriminator="kind"),
]


class ContextPathSummary(ClosedContractModel):
    node_visit_count: int = Field(default=0, ge=0)
    distinct_parent_count: int = Field(default=0, ge=0)
    distinct_path_count: int = Field(ge=0)
    distinct_edge_type_count: int = Field(ge=0)
    covered_facets: list[str] = Field(default_factory=list)
    support_chunk_union: list[str] = Field(default_factory=list)
    reached_by_paths: list[RetrievalPathContribution] = Field(default_factory=list)
    cycle_convergence_score: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_path_summary(self) -> "ContextPathSummary":
        path_ids = [
            path.contribution_id
            for path in self.reached_by_paths
        ]
        if (
            len(path_ids) != len(set(path_ids))
            or path_ids != sorted(path_ids)
        ):
            raise ValueError(
                "context path summary contributions must be unique "
                "and canonically ordered"
            )
        expected = retrieval_node_contribution_facts(
            self.reached_by_paths
        )
        fields = {
            "node_visit_count": expected["node_visit_count"],
            "distinct_parent_count": expected[
                "distinct_parent_count"
            ],
            "distinct_path_count": expected[
                "distinct_path_count"
            ],
            "distinct_edge_type_count": expected[
                "distinct_edge_type_count"
            ],
            "covered_facets": expected["covered_facets"],
            "support_chunk_union": expected[
                "support_chunk_union"
            ],
            "cycle_convergence_score": expected[
                "cycle_convergence_score"
            ],
        }
        for field, expected_value in fields.items():
            if getattr(self, field) != expected_value:
                raise ValueError(
                    f"context path summary {field} does not replay "
                    "reached_by_paths"
                )
        return self


class ContextRestoreCounts(ClosedContractModel):
    hit_chunks: int = Field(ge=0)
    restored_chunks: int = Field(ge=0)
    bridge_chunks: int = Field(ge=0)
    graph_path_chunks: int = Field(ge=0)
    parent_structure_nodes: int = Field(ge=0)
    per_hit_chunk_budget: int = Field(ge=0)


class ContextTokenBudgetAudit(ClosedContractModel):
    token_budget: int = Field(ge=0)
    token_count: int = Field(ge=0)
    within_budget: bool
    clipped_chunk_ids: list[str] = Field(default_factory=list)
    skipped_chunk_ids: list[str] = Field(default_factory=list)
    packing_protocol: str


class ContextSnapshotIntegrityAudit(ClosedContractModel):
    protocol_version: str
    verified_document_version_count: int = Field(ge=0)
    fail_closed: Literal[True]


class ContextPackageDiagnostics(ClosedContractModel):
    context_restoration_protocol: str
    repair_protocol_version: str | None = None
    repair_action_type: str | None = None
    repair_executor_mechanism: str | None = None
    repair_gray_zone_model_call_count: Literal[0]
    repair_gray_zone_decision_authority: Literal[False]
    retrieval_granularity: RetrievalGranularity
    conversation_state_scope_hash: str
    conversation_state_is_evidence: Literal[False]
    runtime_settings_hash: str
    profile_hash: str | None = None
    path_summary: ContextPathSummary
    dedupe_keys: list[str]
    restore_counts: ContextRestoreCounts
    token_budget_audit: ContextTokenBudgetAudit
    snapshot_integrity: ContextSnapshotIntegrityAudit


CONTEXT_PACKAGE_PUBLIC_HASH_FIELDS = (
    "contract_version",
    "id",
    "retrieval_trace_id",
    "knowledge_base_id",
    "query",
    "hit_chunk_ids",
    "restored_chunk_ids",
    "bridge_chunk_ids",
    "parent_structure_node_ids",
    "concept_path",
    "graph_path_ids",
    "reached_by_paths",
    "node_contributions",
    "why_selected",
    "cycle_convergence_score",
    "dedupe_keys",
    "covered_facets",
    "package",
    "contexts",
    "token_budget",
    "token_count",
    "citation_spans",
    "graph_expansion_paths",
    "diagnostics",
)


class ContextPackagePublicHashCard(ClosedContractModel):
    protocol_version: Literal["context_package_public_hash_v1"]
    canonicalization: Literal["json_utf8_sort_keys_compact_v1"]
    hashed_public_fields: list[str]
    public_payload_hash: str = Field(min_length=64, max_length=64)
    public_citation_spans_hash: str = Field(
        min_length=64,
        max_length=64,
    )
    citation_spans_consistency: Literal[
        "persisted_equals_public_projection"
    ]
    chunk_count: int = Field(ge=0)
    citation_span_count: int = Field(ge=0)
    graph_expansion_path_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_hashed_public_fields(
        self,
    ) -> "ContextPackagePublicHashCard":
        if tuple(self.hashed_public_fields) != (
            CONTEXT_PACKAGE_PUBLIC_HASH_FIELDS
        ):
            raise ValueError(
                "context package hash card fields do not match protocol"
            )
        return self


class ContextPackageResponse(ClosedContractModel):
    contract_version: Literal["context_package_public_v1"]
    id: str
    retrieval_trace_id: str | None = None
    knowledge_base_id: str
    package_hash: str = Field(min_length=64, max_length=64)
    package_hash_card: ContextPackagePublicHashCard
    query: str
    hit_chunk_ids: list[str] = Field(default_factory=list)
    restored_chunk_ids: list[str] = Field(default_factory=list)
    bridge_chunk_ids: list[str] = Field(default_factory=list)
    parent_structure_node_ids: list[str] = Field(default_factory=list)
    concept_path: list[ContextConceptPathEntry] = Field(default_factory=list)
    graph_path_ids: list[str] = Field(default_factory=list)
    reached_by_paths: list[RetrievalPathContribution] = Field(default_factory=list)
    node_contributions: list[RetrievalNodeContributionSummary] = Field(default_factory=list)
    why_selected: dict[str, ContextSelectionReason] = Field(default_factory=dict)
    cycle_convergence_score: float | None = None
    dedupe_keys: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    package: ContextPackageDocument
    contexts: list[ContextItem] = Field(default_factory=list)
    token_budget: int = 0
    token_count: int = 0
    citation_spans: list[ContextCitationSpan] = Field(default_factory=list)
    graph_expansion_paths: list[ContextGraphExpansionPath] = Field(default_factory=list)
    diagnostics: ContextPackageDiagnostics
    created_at: datetime | None = None

    @model_validator(mode="after")
    def validate_public_package_hash(
        self,
    ) -> "ContextPackageResponse":
        reached_ids = [
            path.contribution_id
            for path in self.reached_by_paths
        ]
        if (
            len(reached_ids) != len(set(reached_ids))
            or reached_ids != sorted(reached_ids)
        ):
            raise ValueError(
                "context package reached_by_paths must be unique "
                "and canonically ordered"
            )
        reached_by_id = {
            path.contribution_id: path.model_dump(mode="json")
            for path in self.reached_by_paths
        }
        summary_keys: set[tuple[str, str]] = set()
        summary_paths_by_id: dict[str, dict[str, Any]] = {}
        for summary in self.node_contributions:
            summary_key = (summary.layer, summary.node_id)
            if summary_key in summary_keys:
                raise ValueError(
                    "context package node contribution summaries "
                    "must be unique by layer and node"
                )
            summary_keys.add(summary_key)
            for path in summary.reached_by_paths:
                dumped = path.model_dump(mode="json")
                existing = summary_paths_by_id.get(
                    path.contribution_id
                )
                if existing is not None and existing != dumped:
                    raise ValueError(
                        "context package contribution id maps to "
                        "conflicting path facts"
                    )
                summary_paths_by_id[path.contribution_id] = dumped
        if summary_paths_by_id != reached_by_id:
            raise ValueError(
                "context package reached_by_paths do not match the "
                "node contribution union"
            )
        path_summary = self.diagnostics.path_summary
        summary_reached_by_id = {
            path.contribution_id: path.model_dump(mode="json")
            for path in path_summary.reached_by_paths
        }
        if summary_reached_by_id != reached_by_id:
            raise ValueError(
                "context package path summary does not match "
                "reached_by_paths"
            )
        package_path_facts = retrieval_node_contribution_facts(
            self.reached_by_paths
        )
        if (
            self.cycle_convergence_score
            != package_path_facts["cycle_convergence_score"]
        ):
            raise ValueError(
                "context package convergence score does not replay "
                "reached_by_paths"
            )
        if self.covered_facets != package_path_facts[
            "covered_facets"
        ]:
            raise ValueError(
                "context package covered facets do not replay "
                "reached_by_paths"
            )
        why_path_ids: set[str] = set()
        for chunk_id, reason in self.why_selected.items():
            for path in reason.reached_by_paths:
                if chunk_id not in path.path:
                    raise ValueError(
                        "why_selected contribution path does not contain "
                        "its chunk key"
                    )
                dumped = path.model_dump(mode="json")
                if reached_by_id.get(path.contribution_id) != dumped:
                    raise ValueError(
                        "why_selected contribution is outside or "
                        "different from the context package union"
                    )
                why_path_ids.add(path.contribution_id)
            if (
                reason.reached_by_paths
                and chunk_id not in reason.graph_path_chunks
            ):
                raise ValueError(
                    "why_selected graph path chunks do not contain "
                    "their chunk key"
                )
        if why_path_ids != set(reached_by_id):
            raise ValueError(
                "context package contributions must all be explained "
                "by why_selected"
            )
        packaged_chunk_ids: set[str] = set()
        if self.package.chunks and not self.retrieval_trace_id:
            raise ValueError(
                "context package with evidence chunks requires a retrieval trace"
            )
        for chunk in self.package.chunks:
            if chunk.chunk_id in packaged_chunk_ids:
                raise ValueError(
                    "context package chunks must be unique by chunk_id"
                )
            packaged_chunk_ids.add(chunk.chunk_id)
            if (
                chunk.context_package_id != self.id
                or chunk.source_span.context_package_id != self.id
                or chunk.source_span.retrieval_trace_id
                != self.retrieval_trace_id
            ):
                raise ValueError(
                    "context package chunk source span escapes package/trace ownership"
                )
            reason = self.why_selected.get(chunk.chunk_id)
            if reason is None or reason != chunk.why_selected:
                raise ValueError(
                    "context package chunk why_selected does not match "
                    "the top-level explanation"
                )
        if set(self.why_selected) != packaged_chunk_ids:
            raise ValueError(
                "context package why_selected keys must exactly match "
                "the packaged chunks"
            )
        expected_dedupe_keys = [
            chunk.dedupe_key for chunk in self.package.chunks
        ]
        if (
            len(expected_dedupe_keys) != len(set(expected_dedupe_keys))
            or self.dedupe_keys != expected_dedupe_keys
            or self.diagnostics.dedupe_keys != expected_dedupe_keys
        ):
            raise ValueError(
                "context package dedupe keys do not exactly replay chunks"
            )
        chunks_by_id = {
            chunk.chunk_id: chunk for chunk in self.package.chunks
        }
        context_chunk_ids: set[str] = set()
        for context in self.contexts:
            if context.chunk_id in context_chunk_ids:
                raise ValueError(
                    "context items must be unique by chunk_id"
                )
            context_chunk_ids.add(context.chunk_id)
            chunk = chunks_by_id.get(context.chunk_id)
            if chunk is None:
                raise ValueError(
                    "context item is outside the packaged chunk scope"
                )
            reason = self.why_selected.get(context.chunk_id)
            if (
                reason is None
                or reason != context.metadata.why_selected
            ):
                raise ValueError(
                    "context item why_selected does not match the "
                    "top-level explanation"
                )
            chunk_payload = chunk.model_dump(mode="json")
            expected_context = {
                "contract_kind": "context_item",
                "chunk_id": chunk.chunk_id,
                "document_title": chunk.document_title,
                "source_path": chunk.source_path,
                "content": chunk.content,
                "snippet": str(chunk.content)[:280],
                "metadata": {
                    "source_path": chunk.source_path,
                    "logical_source_path": chunk.logical_source_path,
                    "section_path": chunk_payload["section_path"],
                    "structure_path": chunk_payload["structure_path"],
                    "parent_section_node_id": chunk_payload[
                        "structure_closure"
                    ]["parent_section_node_id"],
                    "parent_section": chunk_payload["parent_section"],
                    "structure_node_ids": chunk_payload[
                        "structure_node_ids"
                    ],
                    "page_range": chunk_payload["page_range"],
                    "char_span": chunk_payload["char_span"],
                    "bbox": chunk_payload["bbox"],
                    "source_span": chunk_payload["source_span"],
                    "structure_closure": chunk_payload[
                        "structure_closure"
                    ],
                    "why_selected": chunk_payload["why_selected"],
                    "dedupe_key": chunk.dedupe_key,
                    "role": chunk.role,
                    "content_clipped": chunk.content_clipped,
                    "content_token_count": chunk.content_token_count,
                    "original_token_count": chunk.original_token_count,
                    "raw_chunk_char_span": chunk_payload[
                        "raw_chunk_char_span"
                    ],
                    "context_package_id": self.id,
                },
            }
            if context.model_dump(mode="json") != expected_context:
                raise ValueError(
                    "context item does not exactly replay its package chunk"
                )
        if context_chunk_ids != packaged_chunk_ids:
            raise ValueError(
                "context items and package chunks must describe the "
                "same chunk ids"
            )
        citation_chunk_ids: set[str] = set()
        for citation in self.citation_spans:
            chunk_id = citation.source_span.chunk_id
            if chunk_id in citation_chunk_ids:
                raise ValueError(
                    "context citation spans must be unique by chunk_id"
                )
            citation_chunk_ids.add(chunk_id)
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(
                    "context citation is outside the packaged chunk scope"
                )
            if (
                citation.source_span.context_package_id != self.id
                or citation.source_span.retrieval_trace_id
                != self.retrieval_trace_id
            ):
                raise ValueError(
                    "context citation source span escapes package/trace ownership"
                )
            chunk_payload = chunk.model_dump(mode="json")
            expected_citation = {
                "contract_kind": "citation_span",
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "source_path": chunk.source_path,
                "logical_source_path": chunk.logical_source_path,
                "section_path": chunk_payload["section_path"],
                "structure_path": chunk_payload["structure_path"],
                "structure_node_ids": chunk_payload[
                    "structure_node_ids"
                ],
                "structure_closure": chunk_payload[
                    "structure_closure"
                ],
                "source_span": chunk_payload["source_span"],
            }
            if citation.model_dump(mode="json") != expected_citation:
                raise ValueError(
                    "context citation does not exactly replay its package chunk"
                )
        if citation_chunk_ids != packaged_chunk_ids:
            raise ValueError(
                "context citations and package chunks must describe the same chunk ids"
            )
        public_json = self.model_dump(
            mode="json",
            exclude={
                "package_hash",
                "package_hash_card",
                "created_at",
            },
        )
        public_projection = {
            field: public_json[field]
            for field in CONTEXT_PACKAGE_PUBLIC_HASH_FIELDS
        }

        def digest(value: dict[str, Any]) -> str:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()

        card = self.package_hash_card.model_dump(mode="json")
        if card["public_payload_hash"] != digest(
            public_projection
        ):
            raise ValueError(
                "context package public payload hash mismatch"
            )
        if card["public_citation_spans_hash"] != digest(
            {"citation_spans": public_json["citation_spans"]}
        ):
            raise ValueError(
                "context package public citation hash mismatch"
            )
        if card["chunk_count"] != len(
            public_json["package"]["chunks"]
        ):
            raise ValueError(
                "context package hash card chunk count mismatch"
            )
        if card["citation_span_count"] != len(
            public_json["citation_spans"]
        ):
            raise ValueError(
                "context package hash card citation count mismatch"
            )
        if card["graph_expansion_path_count"] != len(
            public_json["graph_expansion_paths"]
        ):
            raise ValueError(
                "context package hash card graph path count mismatch"
            )
        if self.package_hash != digest(card):
            raise ValueError("context package hash card mismatch")
        return self


class RetrievalConceptPathEntry(ClosedContractModel):
    layer: Literal["coarse", "mid", "rq_membership", "chunk"]
    ids: list[str]


class RetrievalTraceStepsResponse(ClosedContractModel):
    contract_version: Literal["layered_retrieval_trace_public_v1"] = (
        "layered_retrieval_trace_public_v1"
    )
    trace_id: str
    context_package_id: str | None = None
    query: str | None = None
    retrieval_mode: str | None = None
    retrieval_granularity: RetrievalGranularity | None = None
    conversation_state_scope_hash: str = Field(min_length=64, max_length=64)
    concept_path: list[RetrievalConceptPathEntry] = Field(default_factory=list)
    result_chunk_ids: list[str] = Field(default_factory=list)
    query_facets: QueryFacetPacket
    entry_nodes: list[RetrievalEntryNode] = Field(default_factory=list)
    frontier: list[RetrievalFrontierSnapshot] = Field(default_factory=list)
    stage_queues: dict[str, RetrievalStageQueue] = Field(default_factory=dict)
    candidate_pools: RetrievalCandidatePools = Field(default_factory=RetrievalCandidatePools)
    topk_selection: dict[str, RetrievalTopKSelection] = Field(default_factory=dict)
    path_labels: list[RetrievalPathLabel] = Field(default_factory=list)
    node_contributions: list[RetrievalNodeContributionSummary] = Field(default_factory=list)
    convergence: RetrievalConvergence = Field(default_factory=RetrievalConvergence)
    trace_diagnostics: RetrievalTraceDiagnostics
    rq_diagnostics: RetrievalRQDiagnostics
    gray_zone_protocol: str
    gray_zone_decision_authority: Literal["executor_local_deterministic_only"]
    gray_zone_model_call_count: Literal[0]
    gray_zone_determinism: RetrievalGrayZoneDeterminismAudit
    gray_zone_path_decisions: list[RetrievalGrayZoneDecision] = Field(default_factory=list)
    path_distance_threshold_hits: list[RetrievalGrayZoneDecision] = Field(default_factory=list)
    steps: list[GraphRetrievalStepResponse] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_gray_zone_trace_summary(self) -> "RetrievalTraceStepsResponse":
        if not self.gray_zone_protocol.strip():
            raise ValueError("persisted gray-zone protocol is required")
        if self.gray_zone_determinism.status != "passed":
            raise ValueError("retrieval gray-zone audit must pass before the typed trace can be returned")
        local_records = [
            record
            for record in self.gray_zone_path_decisions
            if record.decision_source == "deterministic_local_rule"
        ]
        if len(local_records) != len(self.gray_zone_path_decisions):
            raise ValueError("gray_zone_path_decisions may only contain deterministic local-rule records")
        if any(
            record.decision_source != "deterministic_distance_partition"
            for record in self.path_distance_threshold_hits
        ):
            raise ValueError("path_distance_threshold_hits may only contain deterministic partition records")
        convergence = self.convergence
        if convergence.gray_zone_decision_count != len(local_records):
            raise ValueError("gray-zone convergence decision count does not match returned local-rule records")
        if convergence.gray_zone_rule_evaluation_count != len(local_records):
            raise ValueError("gray-zone evaluation count does not match returned local-rule records")
        if convergence.red_zone_pruned_count != sum(
            record.distance_zone == "red" for record in self.path_distance_threshold_hits
        ):
            raise ValueError("red-zone convergence count does not match partition records")
        if convergence.hard_stop_pruned_count != sum(
            record.distance_zone == "hard_stop" for record in self.path_distance_threshold_hits
        ):
            raise ValueError("hard-stop convergence count does not match partition records")
        contribution_keys: set[tuple[str, str]] = set()
        contribution_ids: set[str] = set()
        for summary in self.node_contributions:
            key = (summary.layer, summary.node_id)
            if key in contribution_keys:
                raise ValueError(
                    "retrieval trace node contributions must be "
                    "unique by layer and node"
                )
            contribution_keys.add(key)
            for path in summary.reached_by_paths:
                if path.contribution_id in contribution_ids:
                    raise ValueError(
                        "retrieval trace contribution ids must be "
                        "globally unique"
                    )
                contribution_ids.add(path.contribution_id)
        return self


class ModelBridgeOperationStatus(ClosedContractModel):
    attempted: bool | None = None
    ok: bool | None = None
    reason: str | None = None
    error: str | None = None
    status_code: int | None = None
    self_target_blocked: bool | None = None
    config_version: str | None = None
    chat_target_hash: str | None = None
    embedding_target_hash: str | None = None


class ModelBridgeStatus(ClosedContractModel):
    enabled: bool = False
    base_url: str | None = None
    reachable: bool | None = None
    admin_available: bool | None = None
    config_matches: bool | None = None
    chat_target_is_bridge: bool | None = None
    embedding_target_is_bridge: bool | None = None
    self_target_blocked: bool | None = None
    config_version: str | None = None
    chat_target_hash: str | None = None
    embedding_target_hash: str | None = None
    desired_chat_target_hash: str | None = None
    desired_embedding_target_hash: str | None = None
    chat_api_protocol: Literal["openai", "anthropic"] | None = None
    desired_chat_api_protocol: Literal["openai", "anthropic"] | None = None
    embedding_api_protocol: Literal["openai"] | None = None
    desired_embedding_api_protocol: Literal["openai"] | None = None
    routes: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    last_reload: ModelBridgeOperationStatus | None = None
    last_sync: ModelBridgeOperationStatus | None = None


class RuntimeSettingsOperatingPointGate(ClosedContractModel):
    required: bool
    stages: list[str] = Field(default_factory=list)
    hard_gates: list[str] = Field(default_factory=list)


class RuntimeSettingsRedaction(ClosedContractModel):
    secret_fields: list[str] = Field(default_factory=list)
    payload_exposes_secret_values: Literal[False]


class RuntimeSettingsLifecycle(ClosedContractModel):
    hot_reloadable: list[str] = Field(default_factory=list)
    rebuild_required: list[str] = Field(default_factory=list)
    service_recreate_required: list[str] = Field(default_factory=list)
    candidate_version_required_for: list[str] = Field(default_factory=list)
    fixed_protocol: dict[str, int] = Field(default_factory=dict)
    operating_point_gate: RuntimeSettingsOperatingPointGate
    redaction: RuntimeSettingsRedaction


class ModelSettingsResponse(PublicResponseModel):
    provider: str | None = None
    chat_api_protocol: Literal["openai", "anthropic"] = "openai"
    graph_api_protocol: Literal["openai", "anthropic"] = "openai"
    embedding_api_protocol: Literal["openai"] = "openai"
    chat_base_url: str | None = None
    graph_base_url: str | None = None
    embedding_base_url: str | None = None
    effective_chat_base_url: str | None = None
    effective_graph_base_url: str | None = None
    effective_embedding_base_url: str | None = None
    chat_resolve_ip: str | None = None
    graph_resolve_ip: str | None = None
    embedding_resolve_ip: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    graph_model: str | None = None
    embedding_dimensions: int | None = None
    embedding_batch_size: int | None = None
    worker_concurrency: int | None = None
    model_request_concurrency: int | None = None
    model_request_timeout_seconds: int | None = None
    chat_json_max_tokens: int | None = Field(default=None, ge=256, le=32768)
    agent_request_concurrency: int | None = None
    source_io_concurrency: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=64,
    )
    agent_request_queue_limit: int | None = None
    agent_request_queue_timeout_seconds: int | None = None
    agent_request_lease_ttl_seconds: int | None = None
    upload_max_bytes: int | None = None
    concept_i18n_enabled: bool | None = None
    query_facet_bilingual_enabled: bool | None = None
    query_facet_posterior_enabled: bool | None = None
    query_facet_posterior_observation_budget: int | None = Field(
        default=None,
        ge=1,
        le=QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    )
    query_facet_posterior_round_budget: int | None = Field(
        default=None,
        ge=1,
        le=QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX,
    )
    query_facet_posterior_convergence_epsilon: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    fixed_chunk_size_tokens: int | None = None
    fixed_chunk_overlap_tokens: int | None = None
    context_package_token_budget: int | None = None
    model_bridge_enabled: bool | None = None
    model_bridge_status: ModelBridgeStatus | None = None
    mid_concept_extraction_max_model_batches: int | None = None
    mid_concept_extraction_max_candidates_per_batch: int | None = None
    mid_concept_extraction_max_tokens_per_batch: int | None = None
    mid_concept_candidate_keep_threshold: float | None = None
    rq_kmeans_levels: Literal[3]
    rq_kmeans_max_k: int | None = None
    rq_residual_tau: float | None = None
    edge_distance_protocol: Literal["edge_distance_log_calibrated_strength_v2"]
    rq_membership_protocol: Literal["rq_primary_chain_v1"]
    edge_projection_protocol: Literal[
        "membership_q15_layer_type_calibrated_v3"
    ]
    edge_type_calibration_protocol: Literal["type_local_winsorized_minmax_v1"]
    rq_membership_temperature: float = Field(gt=0.0, le=10.0)
    dense_knn_k_min: int | None = None
    dense_knn_k_max: int | None = None
    dense_reverse_b_min_base: int | None = None
    dense_reverse_b_max_base: int | None = None
    dense_reverse_b_min_doc: int | None = None
    dense_reverse_b_max_doc: int | None = None
    dense_reverse_b_min_lang: int | None = None
    dense_reverse_b_max_lang: int | None = None
    dense_min_cosine: float | None = None
    dense_strong_cosine: float | None = None
    cross_doc_out_quota_min: int | None = None
    cross_doc_out_quota_max: int | None = None
    cross_doc_min_cosine: float | None = None
    cross_language_out_quota_min: int | None = None
    cross_language_out_quota_max: int | None = None
    cross_language_min_cosine: float | None = None
    enable_auto_tpe: bool | None = None
    tpe_trial_budget: int | None = None
    tpe_startup_random_trials: int | None = None
    tpe_good_quantile_gamma: float | None = None
    tpe_probe_query_budget: int | None = None
    tpe_trial_timeout_seconds: int | None = None
    tpe_candidate_pool_size: int | None = None
    operating_point_hard_gate_max_edge_density: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )
    operating_point_hard_gate_max_isolated_ratio: float | None = None
    operating_point_hard_gate_max_hubness_ratio: float | None = None
    operating_point_hard_gate_min_structure_recovery_rate: float | None = None
    operating_point_hard_gate_max_candidate_latency_p95_ms: int | None = None
    retrieval_result_top_k_default: int | None = None
    agent_coarse_initial_budget: int | None = None
    agent_coarse_total_budget: int | None = None
    agent_coarse_top_k: int | None = None
    agent_mid_per_coarse_budget: int | None = None
    agent_coarse_drilldown_mid_initial_budget: int | None = None
    agent_mid_initial_budget: int | None = None
    agent_mid_top_k: int | None = None
    agent_chunk_per_mid_budget: int | None = None
    agent_chunk_initial_budget: int | None = None
    agent_chunk_top_k: int | None = None
    candidate_pool_dedupe_budget: int | None = None
    agent_max_depth_per_layer: int | None = None
    agent_max_labels_per_node: int | None = None
    agent_max_edge_reuse: int | None = None
    agent_max_cycle_reward_per_path: float | None = None
    agent_cycle_reward_distance_threshold: float | None = None
    agent_path_distance_green_threshold: float | None = None
    agent_path_distance_gray_threshold: float | None = None
    agent_path_distance_hard_threshold: float | None = None
    gray_zone_rule_protocol: Literal["deterministic_support_progress_v1"]
    gray_zone_observation_cadence: int = Field(ge=1, le=16)
    traversal_observation_budget: int = Field(ge=1, le=20_000)
    agent_structure_restore_per_chunk_budget: int | None = None
    agent_structure_restore_budget: int | None = None
    context_path_summary_budget: int | None = None
    agent_planning_round_budget: int | None = None
    agent_max_typed_actions_per_round: int | None = None
    agent_repair_round_budget: int | None = None
    agent_verification_budget: int | None = None
    enable_model_fallback: bool | None = None
    enable_database_fallback: bool | None = None
    has_chat_api_key: bool | None = None
    has_graph_api_key: bool | None = None
    has_embedding_api_key: bool | None = None
    degraded_mode: bool | None = None
    runtime_settings_version: str | None = None
    settings_revision: str | None = None
    setting_statuses: dict[
        str,
        Literal[
            "written_and_applied",
            "written_pending_hot_apply",
            "written_pending_rebuild",
            "written_pending_service_recreate",
        ],
    ] = Field(default_factory=dict)
    pending_rebuild_changes: list[str] = Field(default_factory=list)
    pending_service_recreate_changes: list[str] = Field(default_factory=list)
    pending_hot_changes: list[str] = Field(default_factory=list)
    settings_file_synced: bool = False
    lifecycle: RuntimeSettingsLifecycle | None = None
    requires_service_recreate: bool = False
    service_recreate_changes: list[str] = Field(default_factory=list)
    active_mutated: bool | None = None
    runtime_version_broadcast: bool | None = None
    runtime_version_broadcast_pending: bool | None = None
    runtime_local_refresh_pending: bool | None = None
    apply_error_type: str | None = None


class ModelSettingsUpdate(APIModel):
    model_config = ConfigDict(extra="forbid")

    chat_api_key: str | None = None
    clear_chat_api_key: bool | None = None
    chat_api_protocol: Literal["openai", "anthropic"] | None = None
    chat_base_url: str | None = None
    chat_resolve_ip: str | None = None
    graph_api_key: str | None = None
    clear_graph_api_key: bool | None = None
    graph_api_protocol: Literal["openai", "anthropic"] | None = None
    embedding_api_protocol: Literal["openai"] | None = None
    graph_base_url: str | None = None
    graph_resolve_ip: str | None = None
    embedding_base_url: str | None = None
    embedding_resolve_ip: str | None = None
    embedding_api_key: str | None = None
    clear_embedding_api_key: bool | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    graph_model: str | None = None
    embedding_dimensions: int | None = None
    embedding_batch_size: int | None = None
    model_bridge_enabled: bool | None = None
    worker_concurrency: int | None = None
    model_request_concurrency: int | None = None
    model_request_timeout_seconds: int | None = None
    chat_json_max_tokens: int | None = Field(default=None, ge=256, le=32768)
    agent_request_concurrency: int | None = Field(default=None, ge=1, le=128)
    source_io_concurrency: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=64,
    )
    agent_request_queue_limit: int | None = Field(default=None, ge=0, le=1000)
    agent_request_queue_timeout_seconds: int | None = Field(default=None, ge=1, le=3600)
    agent_request_lease_ttl_seconds: int | None = Field(default=None, ge=5, le=7200)
    upload_max_bytes: int | None = Field(default=None, ge=1, le=10 * 1024 * 1024 * 1024)
    concept_i18n_enabled: bool | None = None
    query_facet_bilingual_enabled: bool | None = None
    query_facet_posterior_enabled: bool | None = None
    query_facet_posterior_observation_budget: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    )
    query_facet_posterior_round_budget: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX,
    )
    query_facet_posterior_convergence_epsilon: float | None = Field(
        default=None,
        strict=True,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    fixed_chunk_size_tokens: int | None = None
    fixed_chunk_overlap_tokens: int | None = None
    context_package_token_budget: int | None = None
    mid_concept_extraction_max_model_batches: int | None = None
    mid_concept_extraction_max_candidates_per_batch: int | None = None
    mid_concept_extraction_max_tokens_per_batch: int | None = None
    mid_concept_candidate_keep_threshold: float | None = None
    rq_kmeans_max_k: int | None = Field(default=None, strict=True, ge=1, le=6)
    rq_residual_tau: float | None = None
    edge_distance_protocol: Literal[
        "edge_distance_log_calibrated_strength_v2"
    ] | None = None
    rq_membership_protocol: Literal[
        "rq_primary_chain_v1"
    ] | None = None
    edge_projection_protocol: Literal[
        "membership_q15_layer_type_calibrated_v3"
    ] | None = None
    edge_type_calibration_protocol: Literal[
        "type_local_winsorized_minmax_v1"
    ] | None = None
    rq_membership_temperature: float | None = Field(
        default=None,
        strict=True,
        gt=0.0,
        le=10.0,
    )
    dense_knn_k_min: int | None = None
    dense_knn_k_max: int | None = None
    dense_reverse_b_min_base: int | None = None
    dense_reverse_b_max_base: int | None = None
    dense_reverse_b_min_doc: int | None = None
    dense_reverse_b_max_doc: int | None = None
    dense_reverse_b_min_lang: int | None = None
    dense_reverse_b_max_lang: int | None = None
    dense_min_cosine: float | None = None
    dense_strong_cosine: float | None = None
    cross_doc_out_quota_min: int | None = None
    cross_doc_out_quota_max: int | None = None
    cross_doc_min_cosine: float | None = None
    cross_language_out_quota_min: int | None = None
    cross_language_out_quota_max: int | None = None
    cross_language_min_cosine: float | None = None
    enable_auto_tpe: bool | None = None
    tpe_trial_budget: int | None = None
    tpe_startup_random_trials: int | None = None
    tpe_good_quantile_gamma: float | None = None
    tpe_probe_query_budget: int | None = None
    tpe_trial_timeout_seconds: int | None = None
    tpe_candidate_pool_size: int | None = None
    operating_point_hard_gate_max_edge_density: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )
    operating_point_hard_gate_max_isolated_ratio: float | None = None
    operating_point_hard_gate_max_hubness_ratio: float | None = None
    operating_point_hard_gate_min_structure_recovery_rate: float | None = None
    operating_point_hard_gate_max_candidate_latency_p95_ms: int | None = None
    retrieval_result_top_k_default: int | None = Field(default=None, ge=1, le=50)
    agent_coarse_initial_budget: int | None = None
    agent_coarse_total_budget: int | None = None
    agent_coarse_top_k: int | None = None
    agent_mid_per_coarse_budget: int | None = None
    agent_coarse_drilldown_mid_initial_budget: int | None = None
    agent_mid_initial_budget: int | None = None
    agent_mid_top_k: int | None = None
    agent_chunk_per_mid_budget: int | None = None
    agent_chunk_initial_budget: int | None = None
    agent_chunk_top_k: int | None = None
    candidate_pool_dedupe_budget: int | None = None
    agent_max_depth_per_layer: int | None = None
    agent_max_labels_per_node: int | None = None
    agent_max_edge_reuse: int | None = None
    agent_max_cycle_reward_per_path: float | None = None
    agent_cycle_reward_distance_threshold: float | None = None
    agent_path_distance_green_threshold: float | None = Field(default=None, ge=0.0, le=20.0)
    agent_path_distance_gray_threshold: float | None = Field(default=None, ge=0.0, le=20.0)
    agent_path_distance_hard_threshold: float | None = Field(default=None, ge=0.0, le=40.0)
    gray_zone_rule_protocol: Literal["deterministic_support_progress_v1"] | None = None
    gray_zone_observation_cadence: int | None = Field(default=None, strict=True, ge=1, le=16)
    traversal_observation_budget: int | None = Field(default=None, strict=True, ge=1, le=20_000)
    agent_structure_restore_per_chunk_budget: int | None = None
    agent_structure_restore_budget: int | None = None
    context_path_summary_budget: int | None = None
    agent_planning_round_budget: int | None = None
    agent_max_typed_actions_per_round: int | None = None
    agent_repair_round_budget: int | None = None
    agent_verification_budget: int | None = None

    @field_validator(
        "edge_distance_protocol",
        "rq_membership_protocol",
        "edge_projection_protocol",
        "edge_type_calibration_protocol",
    )
    @classmethod
    def validate_graph_protocol_is_not_null(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("rebuild-required graph protocol settings cannot be null")
        return value

    @field_validator(
        "rq_membership_temperature",
    )
    @classmethod
    def validate_rq_membership_setting_is_not_null(
        cls,
        value: float | None,
    ) -> float:
        if value is None:
            raise ValueError("RQ membership settings cannot be null")
        return value

    @field_validator("gray_zone_rule_protocol")
    @classmethod
    def validate_gray_zone_rule_protocol_is_not_null(
        cls,
        value: Literal["deterministic_support_progress_v1"] | None,
    ) -> Literal["deterministic_support_progress_v1"]:
        if value is None:
            raise ValueError("gray_zone_rule_protocol cannot be null")
        return value

    @field_validator("gray_zone_observation_cadence")
    @classmethod
    def validate_gray_zone_observation_cadence_is_not_null(cls, value: int | None) -> int:
        if value is None:
            raise ValueError("gray_zone_observation_cadence cannot be null")
        return value

    @field_validator("traversal_observation_budget")
    @classmethod
    def validate_traversal_observation_budget_is_not_null(cls, value: int | None) -> int:
        if value is None:
            raise ValueError("traversal_observation_budget cannot be null")
        return value

    @field_validator(
        "agent_path_distance_green_threshold",
        "agent_path_distance_gray_threshold",
        "agent_path_distance_hard_threshold",
    )
    @classmethod
    def validate_path_distance_threshold_is_not_null(cls, value: float | None) -> float:
        if value is None:
            raise ValueError("path distance thresholds cannot be null")
        return value


class RuntimeSettingsCandidateCreate(APIModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_base_ids: list[str] = Field(min_length=1, max_length=64)
    settings: dict[str, Any] = Field(default_factory=dict)
    dry_run_only: bool = False
    source: str = Field(default="api", min_length=1, max_length=64)


class RuntimeSettingsCandidateActionRequest(APIModel):
    model_config = ConfigDict(extra="forbid")

    build_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class RuntimeSettingsCandidateResponse(APIModel):
    candidate: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None
    action: dict[str, Any] = Field(default_factory=dict)


class RuntimeEnvSyncStatus(ClosedContractModel):
    synced: bool
    settings_file_present: bool = False
    settings_file_schema_synced: bool = False
    missing_keys: list[str] = Field(default_factory=list)
    extra_keys: list[str] = Field(default_factory=list)
    deprecated_keys: list[str] = Field(default_factory=list)
    bom_keys: list[str] = Field(default_factory=list)


class RuntimeInfrastructureStatus(ClosedContractModel):
    postgres: bool
    qdrant: bool
    redis: bool
    model_bridge: bool | None = None


class RuntimeIssue(ClosedContractModel):
    code: str
    title: str
    message: str
    fix_commands: list[str] = Field(default_factory=list)


class RuntimeCheckResponse(PublicResponseModel):
    env_sync: RuntimeEnvSyncStatus | None = None
    infrastructure: RuntimeInfrastructureStatus | None = None
    model_bridge_status: ModelBridgeStatus | None = None
    blocking_issues: list[RuntimeIssue] = Field(default_factory=list)
    warnings: list[RuntimeIssue] = Field(default_factory=list)


class StrategyProfileSummary(ClosedContractModel):
    id: str
    name: str
    library_type: str
    is_builtin: bool = False
    is_active: bool = True
    profile_hash: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class StrategyProfileDetail(StrategyProfileSummary):
    profile_json: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("profile_json", mode="before")
    @classmethod
    def validate_public_profile_json(
        cls,
        value: object,
    ) -> dict[str, Any]:
        from app.services.strategy_profiles import validate_profile_payload

        normalized, _warnings = validate_profile_payload(value)
        if normalized != value:
            raise ValueError(
                "profile_json must be the canonical validated Profile document"
            )
        return normalized


class StrategyProfileCreateRequest(ClosedContractModel):
    name: str
    library_type: str = "general"
    profile_json: dict[str, Any] = Field(default_factory=dict)


class StrategyProfileUpdateRequest(ClosedContractModel):
    name: str | None = None
    library_type: str | None = None
    profile_json: dict[str, Any] | None = None


class StrategyProfileCopyRequest(ClosedContractModel):
    name: str


class StrategyProfileBindRequest(ClosedContractModel):
    knowledge_base_id: str
    profile_id: str


class StrategyProfileMutationResponse(ClosedContractModel):
    profile: StrategyProfileDetail
    warnings: list[str] = Field(default_factory=list)


class StrategyProfileAssistantRequest(APIModel):
    prompt: str
    session_id: str | None = None
    base_profile_id: str | None = None
    base_profile_json: dict[str, Any] | None = None


class StrategyProfileAssistantStateResponse(APIModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    draft: dict[str, Any] | None = None


class QueryContextGraphRequest(APIModel):
    query: str
    chunk_ids: list[str] = Field(default_factory=list)
    knowledge_base_id: str | None = None


KnowledgeBaseTreeNode.model_rebuild()
GraphProjectionRawStrengthSummary.model_rebuild()
GraphEdge.model_rebuild()
QAResponse.model_rebuild()
AgentTraceEventPayload.model_rebuild()
AgentResponse.model_rebuild()
