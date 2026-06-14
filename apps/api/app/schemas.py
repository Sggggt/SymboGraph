from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="allow")


class SearchFilters(APIModel):
    document_ids: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    source_type: str | None = None
    partition: str | None = None
    tags: list[str] = Field(default_factory=list)
    page_range: tuple[int | None, int | None] | None = None
    content_kinds: list[str] = Field(default_factory=list)
    chunk_version: int | None = None


class UploadFileResponse(APIModel):
    document_id: str
    job_id: str
    status: str
    source_path: str


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


class Citation(APIModel):
    chunk_id: str
    document_id: str | None = None
    document_version_id: str | None = None
    title: str | None = None
    source_path: str | None = None
    page_range: list[int] | tuple[int | None, int | None] | None = None
    char_span: list[int] | tuple[int | None, int | None] | None = None
    bbox: dict[str, Any] | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str | None = None
    verification: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(APIModel):
    query: str
    knowledge_base_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=8, ge=1, le=50)


class SearchResult(APIModel):
    chunk_id: str
    document_id: str | None = None
    document_version_id: str | None = None
    title: str | None = None
    text: str = ""
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None
    page_range: list[int] | tuple[int | None, int | None] | None = None
    char_span: list[int] | tuple[int | None, int | None] | None = None
    section_path: list[str] = Field(default_factory=list)
    graph_path: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ModelAudit(APIModel):
    provider: str | None = None
    embedding_model: str | None = None
    embedding_text_version: str | None = None
    retrieval_mode: str | None = None
    retrieval_trace_id: str | None = None
    context_package_id: str | None = None
    degraded: bool = False
    fallback_used: bool = False
    latency_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(APIModel):
    query: str
    results: list[SearchResult]
    degraded_mode: bool = False
    model_audit: ModelAudit | dict[str, Any] = Field(default_factory=ModelAudit)


class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"] | str
    content: str


class QARequest(APIModel):
    question: str
    knowledge_base_id: str | None = None
    session_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=8, ge=1, le=50)
    history: list[ChatMessage] = Field(default_factory=list)


class AnswerModelAudit(ModelAudit):
    chat_model: str | None = None
    answer_session_id: str | None = None
    citation_verification_pass_rate: float | None = None


class QAResponse(APIModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    session_id: str | None = None
    run_id: str | None = None
    context_package_id: str | None = None
    retrieval_trace_id: str | None = None
    used_chunks: list[dict[str, Any]] = Field(default_factory=list)
    route: AgentRoute | str | None = None
    trace: list["AgentTraceEventPayload"] = Field(default_factory=list)
    degraded_mode: bool = False
    model_audit: AnswerModelAudit | dict[str, Any] = Field(default_factory=AnswerModelAudit)
    answer_model_audit: AnswerModelAudit | dict[str, Any] | None = None


class AgentRequest(APIModel):
    question: str
    knowledge_base_id: str | None = None
    session_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=8, ge=1, le=50)
    history: list[ChatMessage] = Field(default_factory=list)
    route: AgentRoute = "layered_context_graph"
    stream_trace: bool = False


class AgentTraceEventPayload(APIModel):
    type: str = "trace"
    run_id: str
    node: str
    status: str
    input_summary: str = ""
    output_summary: str = ""
    document_ids: list[str] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None
    created_at: datetime | None = None


class AgentResponse(QAResponse):
    route: AgentRoute | str = "layered_context_graph"
    trace: list[AgentTraceEventPayload] = Field(default_factory=list)


class TaskStatusResponse(APIModel):
    run_id: str
    status: str
    current_node: str | None = None
    answer: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    trace: list[AgentTraceEventPayload] = Field(default_factory=list)


class SessionSummary(APIModel):
    id: str
    knowledge_base_id: str
    title: str | None = None
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SessionMessagesResponse(APIModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


class DeleteResponse(APIModel):
    deleted: bool


class KnowledgeBaseCreateRequest(APIModel):
    name: str
    description: str | None = None


class KnowledgeBaseSummary(APIModel):
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
    created_at: datetime | None = None
    updated_at: datetime | None = None
    profile_id: str | None = None


class DeleteKnowledgeBaseResponse(APIModel):
    deleted: bool
    knowledge_base_id: str | None = None
    knowledge_base_name: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseFileSummary(APIModel):
    document_id: str
    source_path: str
    title: str | None = None
    status: str = "active"
    current_version: int = 0
    active_chunks: int = 0
    checksum: str | None = None
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
    current_file: str | None = None
    current_phase: str | None = None
    cancel_requested: bool = False
    last_error: str | None = None
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


class GraphNode(APIModel):
    id: str
    label: str
    type: str
    layer: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(APIModel):
    id: str
    source: str
    target: str
    type: str
    weight: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphResponse(APIModel):
    knowledge_base_id: str
    graph_type: GraphType | str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    full_counts: dict[str, int] = Field(default_factory=dict)
    sampled_counts: dict[str, int] = Field(default_factory=dict)
    freshness: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = None
    stale_reason: str | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)
    retrieval_contribution: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class DashboardSnapshot(APIModel):
    knowledge_base: KnowledgeBaseSummary | dict[str, Any]
    tree: list[KnowledgeBaseTreeNode] | list[dict[str, Any]] = Field(default_factory=list)
    graph: GraphResponse | dict[str, Any] = Field(default_factory=dict)
    context_graph: dict[str, Any] = Field(default_factory=dict)
    recent_batches: list[IngestionBatchSummary] | list[dict[str, Any]] = Field(default_factory=list)
    last_refreshed_at: datetime | None = None


class ContextPackageResponse(APIModel):
    id: str
    retrieval_trace_id: str | None = None
    knowledge_base_id: str
    package_hash: str = ""
    query: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    token_budget: int = 0
    citation_spans: list[dict[str, Any]] = Field(default_factory=list)
    graph_expansion_paths: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class RetrievalTraceStepsResponse(APIModel):
    trace_id: str
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ModelSettingsResponse(APIModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    runtime_version: dict[str, Any] = Field(default_factory=dict)


class ModelSettingsUpdate(APIModel):
    openai_api_key: str | None = None
    chat_base_url: str | None = None
    chat_resolve_ip: str | None = None
    embedding_base_url: str | None = None
    embedding_resolve_ip: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    embedding_dimensions: int | None = None
    embedding_batch_size: int | None = None
    model_bridge_enabled: bool | None = None
    worker_concurrency: int | None = None
    model_request_concurrency: int | None = None
    model_request_timeout_seconds: int | None = None
    fixed_chunk_size_tokens: int | None = None
    fixed_chunk_overlap_tokens: int | None = None
    context_package_token_budget: int | None = None
    reranker_enabled: bool | None = None
    reranker_model: str | None = None
    reranker_max_length: int | None = None
    reranker_device: str | None = None
    mid_concept_extraction_max_model_batches: int | None = None
    mid_concept_extraction_max_candidates_per_batch: int | None = None
    mid_concept_extraction_max_tokens_per_batch: int | None = None
    mid_concept_candidate_keep_threshold: float | None = None
    rq_kmeans_levels: int | None = None
    rq_kmeans_max_k: int | None = None
    rq_residual_tau: float | None = None
    agent_coarse_activation_budget: int | None = None
    agent_coarse_jump_budget: int | None = None
    agent_mid_activation_budget: int | None = None
    agent_mid_expansion_radius_cap: int | None = None
    agent_fine_cluster_budget: int | None = None
    agent_chunk_candidate_budget: int | None = None
    agent_structure_restore_budget: int | None = None
    agent_planning_round_budget: int | None = None
    agent_max_typed_actions_per_round: int | None = None
    agent_repair_round_budget: int | None = None
    agent_verification_budget: int | None = None


class RuntimeCheckResponse(APIModel):
    env_sync: dict[str, Any] = Field(default_factory=dict)
    reranker: dict[str, Any] = Field(default_factory=dict)
    infrastructure: dict[str, Any] = Field(default_factory=dict)
    blocking_issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class StrategyProfileSummary(APIModel):
    id: str
    name: str
    library_type: str
    is_builtin: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class StrategyProfileDetail(StrategyProfileSummary):
    profile_json: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class StrategyProfileCreateRequest(APIModel):
    name: str
    library_type: str = "general"
    profile_json: dict[str, Any] = Field(default_factory=dict)


class StrategyProfileUpdateRequest(APIModel):
    name: str | None = None
    library_type: str | None = None
    profile_json: dict[str, Any] | None = None


class StrategyProfileCopyRequest(APIModel):
    name: str


class StrategyProfileBindRequest(APIModel):
    knowledge_base_id: str
    profile_id: str


class StrategyProfileMutationResponse(APIModel):
    profile: StrategyProfileDetail | dict[str, Any]
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
