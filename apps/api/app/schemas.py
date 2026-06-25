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
RetrievalGranularity = Literal["mid", "coarse"]


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
    snippet: str | None = None
    source_span: dict[str, Any] = Field(default_factory=dict)
    retrieval_trace_id: str | None = None
    answer_session_id: str | None = None
    citation_verification_id: str | None = None
    verification: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(APIModel):
    query: str
    knowledge_base_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_granularity: RetrievalGranularity = "mid"


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
    retrieval_granularity: RetrievalGranularity | None = None
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
    retrieval_granularity: RetrievalGranularity = "mid"


class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"] | str
    content: str


class QARequest(APIModel):
    question: str
    knowledge_base_id: str | None = None
    session_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[ChatMessage] = Field(default_factory=list)
    retrieval_granularity: RetrievalGranularity = "mid"


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
    retrieval_granularity: RetrievalGranularity = "mid"
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
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[ChatMessage] = Field(default_factory=list)
    retrieval_granularity: RetrievalGranularity = "mid"
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


class AutoTpeTrialSummary(APIModel):
    trial_id: str
    run_id: str
    trial_index: int
    status: str
    theta_hash: str | None = None
    sampler_state_hash: str | None = None
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
    selected_theta: dict[str, Any] = Field(default_factory=dict)
    sampler_state_hash: str | None = None
    probe_set_hash: str | None = None
    hard_gate: dict[str, Any] = Field(default_factory=dict)
    objective_components: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None
    failure_code: str | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    runtime_settings_hash: str | None = None
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


class GraphNode(APIModel):
    id: str
    label: str
    type: str
    name: str | None = None
    category: str | None = None
    layer: str | None = None
    value: float | int | None = None
    score: float | None = None
    importance_score: float | None = None
    confidence: float | None = None
    support_count: int | None = None
    support_chunk_ids: list[str] = Field(default_factory=list)
    support_active_chunk_ids: list[str] = Field(default_factory=list)
    support_rq_prefix_ids: list[str] = Field(default_factory=list)
    representative_chunk_ids: list[str] = Field(default_factory=list)
    included_mid_concept_ids: list[str] = Field(default_factory=list)
    source_path: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    summary: str | None = None
    snippet: str | None = None
    text: str | None = None
    page_number: int | None = None
    page_range: list[int | None] | tuple[int | None, int | None] | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(APIModel):
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
    score: float | None = None
    support_count: int | None = None
    support_chunk_ids: list[str] = Field(default_factory=list)
    relation_source: str | None = None
    is_bridge: bool | None = None
    is_inferred: bool | None = None
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
    query: str | None = None
    retrieval_mode: str | None = None
    retrieval_granularity: RetrievalGranularity | None = None
    concept_path: list[dict[str, Any]] = Field(default_factory=list)
    result_chunk_ids: list[str] = Field(default_factory=list)
    query_facets: dict[str, Any] = Field(default_factory=dict)
    entry_nodes: list[dict[str, Any]] = Field(default_factory=list)
    frontier: list[dict[str, Any]] = Field(default_factory=list)
    stage_queues: dict[str, Any] = Field(default_factory=dict)
    candidate_pools: dict[str, Any] = Field(default_factory=dict)
    topk_selection: dict[str, Any] = Field(default_factory=dict)
    path_labels: list[dict[str, Any]] = Field(default_factory=list)
    convergence: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ModelSettingsResponse(APIModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    runtime_version: dict[str, Any] = Field(default_factory=dict)


class ModelSettingsUpdate(APIModel):
    model_config = ConfigDict(extra="forbid")

    chat_api_key: str | None = None
    clear_chat_api_key: bool | None = None
    chat_base_url: str | None = None
    chat_resolve_ip: str | None = None
    graph_api_key: str | None = None
    clear_graph_api_key: bool | None = None
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
    concept_i18n_enabled: bool | None = None
    query_facet_bilingual_enabled: bool | None = None
    fixed_chunk_size_tokens: int | None = None
    fixed_chunk_overlap_tokens: int | None = None
    context_package_token_budget: int | None = None
    mid_concept_extraction_max_model_batches: int | None = None
    mid_concept_extraction_max_candidates_per_batch: int | None = None
    mid_concept_extraction_max_tokens_per_batch: int | None = None
    mid_concept_candidate_keep_threshold: float | None = None
    rq_kmeans_levels: int | None = None
    rq_kmeans_max_k: int | None = None
    rq_residual_tau: float | None = None
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
    operating_point_hard_gate_max_edge_density: float | None = None
    operating_point_hard_gate_max_isolated_ratio: float | None = None
    operating_point_hard_gate_max_hubness_ratio: float | None = None
    operating_point_hard_gate_min_structure_recovery_rate: float | None = None
    operating_point_hard_gate_max_candidate_latency_p95_ms: int | None = None
    retrieval_result_top_k_default: int | None = Field(default=None, ge=1, le=50)
    agent_coarse_total_budget: int | None = None
    agent_mid_per_coarse_budget: int | None = None
    agent_mid_top_k: int | None = None
    agent_chunk_per_mid_budget: int | None = None
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
    agent_structure_restore_budget: int | None = None
    context_path_summary_budget: int | None = None
    agent_planning_round_budget: int | None = None
    agent_max_typed_actions_per_round: int | None = None
    agent_repair_round_budget: int | None = None
    agent_verification_budget: int | None = None


class RuntimeCheckResponse(APIModel):
    env_sync: dict[str, Any] = Field(default_factory=dict)
    infrastructure: dict[str, Any] = Field(default_factory=dict)
    blocking_issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class StrategyProfileSummary(APIModel):
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
