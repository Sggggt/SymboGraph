from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


JobState = Literal[
    "queued",
    "parsing",
    "chunking",
    "embedding",
    "extracting_graph",
    "cancel_requested",
    "cancelling",
    "compensating",
    "cancelled",
    "processing",
    "completed",
    "partial_failed",
    "failed",
    "skipped",
]
KnowledgeBaseFileStatus = Literal["pending", "parsing", "parsed", "failed", "skipped"]
SourceType = Literal["pdf", "ppt", "pptx", "docx", "markdown", "text", "image", "notebook", "html", "unknown"]
AgentRoute = Literal["direct_answer", "retrieve_sources", "retrieve_tasks", "retrieve_both", "clarify", "multi_hop_research"]
AgentRunState = Literal["queued", "running", "needs_clarification", "completed", "failed"]
GraphType = Literal["evidence"]
GraphNodeCategory = Literal[
    "knowledge_base",
    "evidence_graph_state",
    "evidence_atom",
    "active_chunk",
    "community_region",
    "signal_node",
    "document",
    "partition",
    "section",
    "evidence_chunk",
    "document_version",
]


class SearchFilters(BaseModel):
    partition: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    source_type: SourceType | None = None


class UploadFileResponse(BaseModel):
    document_id: str
    job_id: str | None = None
    status: JobState
    source_path: str


class ParseUploadedFilesRequest(BaseModel):
    file_paths: list[str] = Field(default_factory=list)
    force: bool = False
    full_reparse: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    state: JobState
    error: str | None = None
    document_id: str | None = None
    source_path: str | None = None
    batch_id: str | None = None
    stats: dict = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source_path: str
    partition: str | None = None
    section: str | None = None
    page_number: int | None = None
    snippet: str
    active_chunk_id: str | None = None
    evidence_atom_ids: list[str] = Field(default_factory=list)
    source_span: dict = Field(default_factory=dict)
    retrieval_trace_id: str | None = None
    citation_verification_id: str | None = None


class SearchRequest(BaseModel):
    query: str
    knowledge_base_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=6, ge=1, le=50)


class QueryEvidenceGraphRequest(BaseModel):
    knowledge_base_id: str | None = None
    query: str | None = None
    chunk_ids: list[str] = Field(default_factory=list, min_length=1, max_length=50)


class SearchResult(BaseModel):
    chunk_id: str
    active_chunk_id: str | None = None
    snippet: str
    score: float
    citations: list[Citation]
    metadata: dict
    content: str | None = None
    child_content: str | None = None
    document_title: str | None = None
    source_path: str | None = None
    partition: str | None = None
    source_type: str | None = None


class ModelAudit(BaseModel):
    embedding_provider: str = "none"
    embedding_model: str | None = None
    embedding_external_called: bool = False
    embedding_fallback_reason: str | None = None
    reranker_enabled: bool = False
    reranker_called: bool = False
    fallback_enabled: bool = False
    degraded_mode: bool = False
    vector_index_warning: str | None = None
    retrieval_pipeline: str | None = None
    signal_state_hash: str | None = None
    signal_node_ids: list[str] = Field(default_factory=list)
    retrieval_cache_scope_hash: str | None = None
    cached: bool = False
    scope_hash: str | None = None
    cache: dict = Field(default_factory=dict)


class AnswerModelAudit(BaseModel):
    provider: str = "none"
    model: str | None = None
    external_called: bool = False
    fallback_reason: str | None = None
    skipped_reason: str | None = None
    signal_state_hash: str | None = None
    signal_node_ids: list[str] = Field(default_factory=list)
    signal_expansion_used: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    degraded_mode: bool = False
    model_audit: ModelAudit = Field(default_factory=ModelAudit)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class QARequest(BaseModel):
    question: str
    session_id: str | None = None
    knowledge_base_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=6, ge=1, le=50)
    history: list[ChatMessage] = Field(default_factory=list)


class QAResponse(BaseModel):
    run_id: str | None = None
    session_id: str | None = None
    answer: str
    citations: list[Citation]
    used_chunks: list[dict]
    route: AgentRoute | None = None
    trace: list["AgentTraceEventPayload"] = Field(default_factory=list)
    degraded_mode: bool = False
    answer_model_audit: AnswerModelAudit = Field(default_factory=AnswerModelAudit)


class AgentRequest(BaseModel):
    question: str
    session_id: str | None = None
    knowledge_base_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=6, ge=1, le=50)
    history: list[ChatMessage] = Field(default_factory=list)
    stream_trace: bool = True


class AgentTraceEventPayload(BaseModel):
    id: str | None = None
    run_id: str | None = None
    node: str
    status: str = "completed"
    input_summary: str | None = None
    output_summary: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    scores: dict = Field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None
    created_at: datetime | None = None


class AgentResponse(BaseModel):
    run_id: str
    session_id: str
    answer: str
    citations: list[Citation]
    used_chunks: list[dict]
    route: AgentRoute
    trace: list[AgentTraceEventPayload]
    degraded_mode: bool = False
    answer_model_audit: AnswerModelAudit = Field(default_factory=AnswerModelAudit)


class TaskStatusResponse(BaseModel):
    run_id: str
    state: AgentRunState
    current_node: str | None = None
    retry_count: int = 0
    route: AgentRoute | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class SessionSummary(BaseModel):
    id: str
    title: str | None = None
    last_question: str | None = None
    last_answer: str | None = None
    transcript: list[dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: bool


class CleanupStaleDataResponse(BaseModel):
    deleted_vectors: int = 0
    deleted_chunks: int = 0
    deleted_document_versions: int = 0
    deleted_documents: int = 0
    removed_vector_records: int = 0
    removed_evidence_atoms: int = 0
    removed_evidence_edges: int = 0
    removed_evidence_graph_states: int = 0
    removed_active_chunks: int = 0
    removed_chunk_candidates: int = 0
    removed_chunk_decisions: int = 0
    removed_quality_decisions: int = 0
    removed_community_states: int = 0
    removed_community_memberships: int = 0
    removed_community_summaries: int = 0


class RebuildGraphRequest(BaseModel):
    mode: Literal["evidence"] = "evidence"
    dry_run: bool = False


class StrategyProfileSummary(BaseModel):
    id: str
    name: str
    library_type: str = "custom"
    is_builtin: bool = False
    is_active: bool = True
    profile_hash: str
    knowledge_base_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None


class StrategyProfileDetail(StrategyProfileSummary):
    profile_json: dict = Field(default_factory=dict)


class StrategyProfileCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    library_type: str = "custom"
    profile_json: dict = Field(default_factory=dict)


class StrategyProfileUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    library_type: str | None = None
    profile_json: dict | None = None


class StrategyProfileCopyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class StrategyProfileBindRequest(BaseModel):
    knowledge_base_id: str
    profile_id: str


class StrategyProfileMutationResponse(BaseModel):
    profile: StrategyProfileDetail
    warnings: list[str] = Field(default_factory=list)


class StrategyProfileDraftRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    base_profile_id: str | None = None
    base_profile_json: dict | None = None


class StrategyProfileDraftResponse(BaseModel):
    profile_json: dict
    warnings: list[str] = Field(default_factory=list)
    profile_hash: str


class StrategyProfileAssistantRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    base_profile_id: str | None = None
    base_profile_json: dict | None = None


class StrategyProfileAssistantStateResponse(BaseModel):
    session_id: str
    base_profile_id: str | None = None
    messages: list[dict] = Field(default_factory=list)
    latest_profile_json: dict | None = None
    latest_profile_hash: str | None = None
    warnings: list[str] = Field(default_factory=list)
    draft_message: str = ""
    created_at: str | None = None
    updated_at: str | None = None


class RebuildGraphResponse(BaseModel):
    batch_id: str | None = None
    state: str
    mode: str = "evidence"
    affected_documents: int = 0
    previous_batch_id: str | None = None
    dry_run: bool = False
    evidence_atoms: int = 0
    evidence_edges: int = 0
    active_chunks: int = 0


class BatchLogTokenResponse(BaseModel):
    token: str
    expires_at: datetime


class DeleteKnowledgeBaseResponse(BaseModel):
    deleted: bool
    deleted_vectors: int = 0
    deleted_vector_records: int = 0
    deleted_active_chunks: int = 0
    deleted_chunk_decisions: int = 0
    deleted_quality_decisions: int = 0
    deleted_chunk_candidates: int = 0
    deleted_evidence_atoms: int = 0
    deleted_evidence_edges: int = 0
    deleted_evidence_graph_states: int = 0
    deleted_community_states: int = 0
    deleted_community_memberships: int = 0
    deleted_community_summaries: int = 0
    deleted_signal_schema_states: int = 0
    deleted_signal_states: int = 0
    deleted_signal_candidates: int = 0
    deleted_signal_decisions: int = 0
    deleted_signal_nodes: int = 0
    deleted_signal_edges: int = 0
    deleted_signal_communities: int = 0
    deleted_signal_community_memberships: int = 0
    deleted_projection_states: int = 0
    deleted_projection_nodes: int = 0
    deleted_projection_edges: int = 0
    deleted_projection_communities: int = 0
    deleted_policy_states: int = 0
    deleted_policy_observations: int = 0
    deleted_quality_observations: int = 0
    deleted_retrieval_traces: int = 0
    deleted_answer_sessions: int = 0
    deleted_citation_verifications: int = 0
    deleted_reward_events: int = 0
    deleted_trace_events: int = 0
    deleted_agent_runs: int = 0
    deleted_sessions: int = 0
    deleted_ingestion_logs: int = 0
    deleted_compensations: int = 0
    deleted_jobs: int = 0
    deleted_batches: int = 0
    deleted_chunks: int = 0
    deleted_document_versions: int = 0
    deleted_documents: int = 0
    deleted_knowledge_bases: int = 0
    deleted_directory: int = 0


class EvidenceAtomOut(BaseModel):
    id: str
    knowledge_base_id: str
    document_id: str
    document_version_id: str
    atom_type: str
    text: str
    source_span_json: dict = Field(default_factory=dict)
    state: str
    created_at: datetime


class SignalNodeOut(BaseModel):
    id: str
    knowledge_base_id: str
    signal_state_id: str
    canonical_label: str
    signal_type: str
    support_atom_ids: list[str] = Field(default_factory=list)
    support_active_chunk_ids: list[str] = Field(default_factory=list)
    source_span_union: dict = Field(default_factory=dict)
    confidence: float
    created_at: datetime


class QualityDecisionOut(BaseModel):
    id: str
    candidate_id: str
    policy_state_id: str | None = None
    gate_passed: bool
    decision_action: str
    confidence: float
    risk_flags_json: list[str] = Field(default_factory=list)
    feedback_json: dict = Field(default_factory=dict)
    diagnostics_json: dict = Field(default_factory=dict)
    created_at: datetime


class ChunkDecisionOut(BaseModel):
    id: str
    knowledge_base_id: str
    graph_state_id: str
    candidate_id: str
    quality_decision_id: str
    policy_state_id: str | None = None
    action: str
    decision_protocol_version: str
    diagnostics_json: dict = Field(default_factory=dict)
    created_at: datetime


class PolicyStateOut(BaseModel):
    id: str
    knowledge_base_id: str
    policy_family: str
    policy_version: str
    profile_objective_hash: str
    constraints_json: dict = Field(default_factory=dict)
    exploration_json: dict = Field(default_factory=dict)
    reward_summary_json: dict = Field(default_factory=dict)
    drift_status: str
    drift_detected_at: datetime | None = None
    state_hash: str
    created_at: datetime


class RefreshResponse(BaseModel):
    knowledge_base_id: str
    refreshed_at: datetime


class ModelSettingsResponse(BaseModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    chat_base_url: str
    embedding_base_url: str
    model_bridge_enabled: bool = False
    chat_resolve_ip: str | None = None
    embedding_resolve_ip: str | None = None
    embedding_model: str
    chat_model: str
    embedding_dimensions: int
    worker_concurrency: int = 2
    ingestion_file_concurrency: int = 2
    model_request_concurrency: int = 2
    model_request_timeout_seconds: int = 180
    chunk_token_budget: int = 2400
    reranker_enabled: bool = False
    reranker_model: str = ""
    reranker_max_length: int = 512
    reranker_device: Literal["cpu", "cuda"] = "cpu"
    reranker_url: str = ""
    semantic_chunking_enabled: bool = False
    semantic_chunking_min_length: int = 2000
    retrieval_layer_enabled: bool = True
    retrieval_cache_ttl_seconds: int = 300
    enable_agentic_reflection: bool = True
    enable_post_generation_reflection: bool = False
    citation_verification_sample_max: int = 3
    reflection_max_retries: int = 2
    enable_graph_community_summaries: bool = True
    signal_extraction_max_model_batches: int = 4
    signal_extraction_max_candidates_per_batch: int = 40
    signal_extraction_max_tokens_per_batch: int = 6000
    signal_candidate_keep_threshold: float = 0.62
    community_louvain_resolution: float = 1.0
    community_min_modularity_warn: float = 0.18
    graph_overview_max_nodes: int = 260
    graph_overview_max_edges: int = 800
    enable_model_fallback: bool = False
    enable_database_fallback: bool = False
    has_api_key: bool
    has_embedding_api_key: bool
    degraded_mode: bool
    runtime_settings_version: str | None = None


class ModelSettingsUpdate(BaseModel):
    api_key: str | None = None
    clear_api_key: bool = False
    chat_base_url: str | None = None
    embedding_base_url: str | None = None
    model_bridge_enabled: bool | None = None
    chat_resolve_ip: str | None = None
    embedding_resolve_ip: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    embedding_dimensions: int | None = Field(default=None, ge=1, le=8192)
    worker_concurrency: int | None = Field(default=None, ge=1, le=32)
    ingestion_file_concurrency: int | None = Field(default=None, ge=1, le=8)
    model_request_concurrency: int | None = Field(default=None, ge=1, le=16)
    model_request_timeout_seconds: int | None = Field(default=None, ge=5, le=600)
    chunk_token_budget: int | None = Field(default=None, ge=256, le=20000)
    reranker_enabled: bool | None = None
    reranker_model: str | None = None
    reranker_max_length: int | None = Field(default=None, ge=64, le=2048)
    reranker_device: Literal["cpu", "cuda"] | None = None
    semantic_chunking_enabled: bool | None = None
    semantic_chunking_min_length: int | None = Field(default=None, ge=500, le=5000)
    retrieval_layer_enabled: bool | None = None
    retrieval_cache_ttl_seconds: int | None = Field(default=None, ge=0, le=86400)
    enable_agentic_reflection: bool | None = None
    enable_post_generation_reflection: bool | None = None
    citation_verification_sample_max: int | None = Field(default=None, ge=0, le=20)
    reflection_max_retries: int | None = Field(default=None, ge=0, le=5)
    enable_graph_community_summaries: bool | None = None
    signal_extraction_max_model_batches: int | None = Field(default=None, ge=0, le=64)
    signal_extraction_max_candidates_per_batch: int | None = Field(default=None, ge=1, le=500)
    signal_extraction_max_tokens_per_batch: int | None = Field(default=None, ge=500, le=50000)
    signal_candidate_keep_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    community_louvain_resolution: float | None = Field(default=None, ge=0.05, le=5.0)
    community_min_modularity_warn: float | None = Field(default=None, ge=-1.0, le=1.0)
    graph_overview_max_nodes: int | None = Field(default=None, ge=20, le=2000)
    graph_overview_max_edges: int | None = Field(default=None, ge=20, le=5000)
    embedding_api_key: str | None = None
    clear_embedding_api_key: bool = False


class RuntimeIssue(BaseModel):
    code: str
    title: str
    message: str
    fix_commands: list[str] = Field(default_factory=list)


class EnvSyncStatus(BaseModel):
    synced: bool
    missing_keys: list[str] = Field(default_factory=list)
    extra_keys: list[str] = Field(default_factory=list)
    deprecated_keys: list[str] = Field(default_factory=list)
    bom_keys: list[str] = Field(default_factory=list)


class InfrastructureStatus(BaseModel):
    postgres: bool
    qdrant: bool
    redis: bool
    model_bridge: bool | None = None


class RuntimeCheckResponse(BaseModel):
    env_sync: EnvSyncStatus
    reranker: dict = Field(default_factory=dict)
    infrastructure: InfrastructureStatus
    blocking_issues: list[RuntimeIssue] = Field(default_factory=list)
    warnings: list[RuntimeIssue] = Field(default_factory=list)


class StructuredApiError(BaseModel):
    code: str
    title: str
    message: str
    issues: list[RuntimeIssue] = Field(default_factory=list)
    fix_commands: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    name: str
    category: GraphNodeCategory | str
    value: int | float | None = None
    partition: str | None = None
    importance_score: float | None = None
    source_type: str | None = None
    entity_type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    support_count: int | None = None
    support_atom_ids: list[str] = Field(default_factory=list)
    support_active_chunk_ids: list[str] = Field(default_factory=list)
    source_span_union: dict | None = None
    confidence: float | None = None
    canonical_key: str | None = None
    summary: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    snippet: str | None = None
    page_number: int | None = None
    evidence_count: int | None = None
    community_louvain: int | None = None
    community_spectral: int | None = None
    component_id: int | None = None
    centrality_score: float | None = None
    graph_rank_score: float | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    confidence: float | None = None
    category: str | None = None
    evidence_chunk_id: str | None = None
    weight: float | None = None
    semantic_similarity: float | None = None
    support_count: int | None = None
    support_atom_ids: list[str] = Field(default_factory=list)
    support_active_chunk_ids: list[str] = Field(default_factory=list)
    source_span_union: dict | None = None
    relation_source: str | None = None
    is_inferred: bool = False


class GraphResponse(BaseModel):
    graph_type: GraphType
    schema_version: str = "typed_graph_v1"
    view: str | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_counts: dict[str, int] = Field(default_factory=dict)
    edge_counts: dict[str, int] = Field(default_factory=dict)
    focus_partition: str | None = None
    freshness: dict = Field(default_factory=dict)
    signal_layer_status: str | None = None
    signal_state_id: str | None = None
    signal_state_hash: str | None = None
    signal_layer_complete: bool = False
    diagnostics: dict = Field(default_factory=dict)


class KnowledgeBaseTreeNode(BaseModel):
    id: str
    title: str
    type: Literal["knowledge_base", "partition", "document", "signal"]
    children: list["KnowledgeBaseTreeNode"] = Field(default_factory=list)


class KnowledgeBaseSummary(BaseModel):
    id: str
    name: str
    description: str | None = None
    source_root: str
    storage_root: str
    document_count: int
    evidence_atom_count: int
    current_chunk_version: int = 0
    has_parsed_chunks: bool = False
    can_full_reparse: bool = False
    degraded_mode: bool = False
    active_profile_id: str | None = None
    active_profile_name: str | None = None
    active_profile_hash: str | None = None


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class IngestionStats(BaseModel):
    active_chunks: int = 0
    evidence_atoms: int = 0
    evidence_edges: int = 0
    signal_nodes: int = 0
    signal_edges: int = 0


class BatchError(BaseModel):
    source_path: str
    message: str


class IngestionBatchSummary(BaseModel):
    batch_id: str
    state: JobState
    trigger_source: str
    source_root: str
    total_files: int
    processed_files: int
    success_count: int
    failure_count: int
    skipped_count: int
    coverage_by_source_type: dict[str, int] = Field(default_factory=dict)
    errors: list[BatchError] = Field(default_factory=list)
    graph_stats: dict = Field(default_factory=dict)
    phase: str | None = None
    parse_committed: bool = False
    cancellation_status: str | None = None
    worker_id: str | None = None
    heartbeat_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class BatchStartResponse(BaseModel):
    batch_id: str
    state: JobState


class DashboardSnapshot(BaseModel):
    knowledge_base: KnowledgeBaseSummary
    tree: list[KnowledgeBaseTreeNode]
    graph: GraphResponse
    batch_status: IngestionBatchSummary | None = None
    ingested_document_count: int = 0
    chunk_count: int = 0
    evidence_atom_count: int = 0
    active_chunk_count: int = 0
    evidence_edge_count: int = 0
    community_region_count: int = 0
    graph_eligible_chunk_count: int = 0
    graph_relation_count: int = 0
    coverage_by_source_type: dict[str, int] = Field(default_factory=dict)
    degraded_mode: bool = False


class ChunkPayload(BaseModel):
    id: str
    document_id: str
    document_title: str
    source_path: str
    source_type: str
    partition: str | None = None
    section: str | None = None
    page_number: int | None = None
    snippet: str
    content: str
    metadata: dict = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    id: str
    title: str
    source_path: str
    source_type: str
    partition: str | None = None
    updated_at: datetime


class KnowledgeBaseFileSummary(BaseModel):
    id: str
    document_id: str | None = None
    title: str
    source_path: str
    source_type: str = "unknown"
    partition: str | None = None
    status: KnowledgeBaseFileStatus
    job_state: JobState | None = None
    batch_id: str | None = None
    error: str | None = None
    chunk_count: int = 0
    chunk_version: int | None = None
    updated_at: datetime | None = None


KnowledgeBaseTreeNode.model_rebuild()
QAResponse.model_rebuild()
