export type Visibility = "private";

export type SourceType =
  | "pdf"
  | "ppt"
  | "pptx"
  | "docx"
  | "markdown"
  | "text"
  | "image"
  | "notebook"
  | "html"
  | "unknown";

export type JobState =
  | "queued"
  | "parsing"
  | "chunking"
  | "embedding"
  | "extracting_graph"
  | "cancel_requested"
  | "cancelling"
  | "compensating"
  | "cancelled"
  | "processing"
  | "completed"
  | "partial_failed"
  | "failed"
  | "skipped";

export type KnowledgeBaseFileStatus = "pending" | "parsing" | "parsed" | "failed" | "skipped";

export type AgentRoute =
  | "direct_answer"
  | "retrieve_sources"
  | "retrieve_tasks"
  | "retrieve_both"
  | "clarify"
  | "multi_hop_research";

export type AgentRunState = "queued" | "running" | "needs_clarification" | "completed" | "failed";

export interface SearchFilters {
  partition?: string;
  tags?: string[];
  difficulty?: string;
  source_type?: SourceType;
}

export interface UploadFileResponse {
  document_id: string;
  job_id?: string | null;
  status: JobState;
  source_path: string;
}

export interface JobStatusResponse {
  job_id: string;
  state: JobState;
  error?: string | null;
  document_id?: string | null;
  source_path?: string | null;
  batch_id?: string | null;
  stats?: Record<string, unknown>;
}

export interface SearchRequest {
  query: string;
  knowledge_base_id?: string | null;
  filters?: SearchFilters;
  top_k?: number;
}

export interface QueryEvidenceGraphRequest {
  knowledge_base_id?: string | null;
  query?: string | null;
  chunk_ids: string[];
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  source_path: string;
  partition?: string | null;
  section?: string | null;
  page_number?: number | null;
  snippet: string;
  active_chunk_id?: string | null;
  evidence_atom_ids?: string[];
  source_span?: Record<string, unknown>;
  retrieval_trace_id?: string | null;
  citation_verification_id?: string | null;
}

export interface SearchResult {
  chunk_id: string;
  active_chunk_id?: string | null;
  snippet: string;
  score: number;
  citations: Citation[];
  metadata: Record<string, unknown>;
  content?: string | null;
  child_content?: string | null;
  document_title?: string | null;
  source_path?: string | null;
  partition?: string | null;
  source_type?: string | null;
}

export interface ModelAudit {
  embedding_provider: string;
  embedding_model?: string | null;
  embedding_external_called: boolean;
  embedding_fallback_reason?: string | null;
  reranker_enabled: boolean;
  reranker_called: boolean;
  fallback_enabled: boolean;
  degraded_mode: boolean;
  vector_index_warning?: string | null;
  retrieval_pipeline?: string | null;
  signal_state_hash?: string | null;
  signal_node_ids?: string[];
  retrieval_cache_scope_hash?: string | null;
  cached?: boolean;
  scope_hash?: string | null;
  cache?: Record<string, unknown>;
}

export interface AnswerModelAudit {
  provider: string;
  model?: string | null;
  external_called: boolean;
  fallback_reason?: string | null;
  skipped_reason?: string | null;
  signal_state_hash?: string | null;
  signal_node_ids?: string[];
  signal_expansion_used?: boolean;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  degraded_mode: boolean;
  model_audit: ModelAudit;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface QARequest {
  question: string;
  session_id?: string | null;
  knowledge_base_id?: string | null;
  filters?: SearchFilters;
  top_k?: number;
  history?: ChatMessage[];
}

export interface QAResponse {
  run_id?: string | null;
  session_id?: string | null;
  answer: string;
  citations: Citation[];
  used_chunks: Array<Record<string, unknown>>;
  route?: AgentRoute | null;
  trace?: AgentTraceEventPayload[];
  degraded_mode: boolean;
  answer_model_audit: AnswerModelAudit;
}

export interface AgentRequest {
  question: string;
  session_id?: string | null;
  knowledge_base_id?: string | null;
  filters?: SearchFilters;
  top_k?: number;
  history?: ChatMessage[];
  stream_trace?: boolean;
}

export type AgentTraceNode =
  | "perception"
  | "retrieval_planner"
  | "base_retrieval"
  | "evidence_anchor_selector"
  | "evidence_chain_planner"
  | "controlled_graph_enhancer"
  | "evidence_assembler"
  | "document_grader"
  | "evidence_evaluator"
  | "context_synthesizer"
  | "answer_generator"
  | "citation_checker"
  | "citation_verifier"
  | "reflection"
  | "self_check"
  | "error";

export interface AgentTraceEventPayload {
  id?: string | null;
  run_id?: string | null;
  node: AgentTraceNode | (string & {});
  status: string;
  input_summary?: string | null;
  output_summary?: string | null;
  document_ids: string[];
  scores: Record<string, unknown>;
  duration_ms: number;
  error?: string | null;
  created_at?: string | null;
}

export interface AgentResponse {
  run_id: string;
  session_id: string;
  answer: string;
  citations: Citation[];
  used_chunks: Array<Record<string, unknown>>;
  route: AgentRoute;
  trace: AgentTraceEventPayload[];
  degraded_mode: boolean;
  answer_model_audit: AnswerModelAudit;
}

export interface TaskStatusResponse {
  run_id: string;
  state: AgentRunState;
  current_node?: string | null;
  retry_count: number;
  route?: AgentRoute | null;
  error?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface SessionSummary {
  id: string;
  title?: string | null;
  last_question?: string | null;
  last_answer?: string | null;
  transcript: Array<Record<string, unknown>>;
  created_at: string;
  updated_at: string;
}

export interface SessionMessagesResponse {
  session_id: string;
  messages: Array<Record<string, unknown>>;
}

export interface DeleteResponse {
  deleted: boolean;
}

export interface CleanupStaleDataResponse {
  deleted_vectors: number;
  deleted_chunks: number;
  deleted_document_versions: number;
  deleted_documents: number;
  removed_vector_records: number;
  removed_evidence_atoms: number;
  removed_evidence_edges: number;
  removed_evidence_graph_states: number;
  removed_active_chunks: number;
  removed_chunk_candidates: number;
  removed_chunk_decisions: number;
  removed_quality_decisions: number;
  removed_community_states: number;
  removed_community_memberships: number;
  removed_community_summaries: number;
}

export interface RebuildGraphRequest {
  mode: "evidence";
  dry_run?: boolean;
}

export interface RebuildGraphResponse {
  batch_id?: string | null;
  state: string;
  mode: string;
  affected_documents: number;
  previous_batch_id?: string | null;
  dry_run?: boolean;
  evidence_atoms?: number;
  evidence_edges?: number;
  active_chunks?: number;
}

export interface BatchLogTokenResponse {
  token: string;
  expires_at: string;
}

export interface DeleteKnowledgeBaseResponse {
  deleted: boolean;
  deleted_vectors: number;
  deleted_vector_records: number;
  deleted_active_chunks: number;
  deleted_chunk_decisions: number;
  deleted_quality_decisions: number;
  deleted_chunk_candidates: number;
  deleted_evidence_atoms: number;
  deleted_evidence_edges: number;
  deleted_evidence_graph_states: number;
  deleted_community_states: number;
  deleted_community_memberships: number;
  deleted_community_summaries: number;
  deleted_signal_schema_states?: number;
  deleted_signal_states?: number;
  deleted_signal_candidates?: number;
  deleted_signal_decisions?: number;
  deleted_signal_nodes?: number;
  deleted_signal_edges?: number;
  deleted_signal_communities?: number;
  deleted_signal_community_memberships?: number;
  deleted_projection_states?: number;
  deleted_projection_nodes?: number;
  deleted_projection_edges?: number;
  deleted_projection_communities?: number;
  deleted_policy_states: number;
  deleted_policy_observations: number;
  deleted_quality_observations: number;
  deleted_retrieval_traces: number;
  deleted_answer_sessions: number;
  deleted_citation_verifications: number;
  deleted_reward_events: number;
  deleted_trace_events: number;
  deleted_agent_runs: number;
  deleted_sessions: number;
  deleted_ingestion_logs: number;
  deleted_compensations: number;
  deleted_jobs: number;
  deleted_batches: number;
  deleted_chunks: number;
  deleted_document_versions: number;
  deleted_documents: number;
  deleted_knowledge_bases: number;
  deleted_directory: number;
}

export interface RefreshResponse {
  knowledge_base_id?: string;
  refreshed_at: string;
}

export interface ModelSettingsResponse {
  provider: "openai_compatible";
  chat_base_url: string;
  embedding_base_url: string;
  chat_resolve_ip?: string | null;
  embedding_resolve_ip?: string | null;
  embedding_model: string;
  chat_model: string;
  embedding_dimensions: number;
  worker_concurrency: number;
  ingestion_file_concurrency: number;
  model_request_concurrency: number;
  model_request_timeout_seconds: number;
  chunk_token_budget: number;
  reranker_enabled: boolean;
  reranker_model: string;
  reranker_max_length: number;
  reranker_device: "cpu" | "cuda";
  reranker_url: string;
  semantic_chunking_enabled: boolean;
  semantic_chunking_min_length: number;
  retrieval_layer_enabled: boolean;
  retrieval_cache_ttl_seconds: number;
  enable_agentic_reflection: boolean;
  enable_post_generation_reflection: boolean;
  citation_verification_sample_max: number;
  reflection_max_retries: number;
  model_bridge_enabled: boolean;
  enable_graph_community_summaries: boolean;
  signal_extraction_max_model_batches: number;
  signal_extraction_max_candidates_per_batch: number;
  signal_extraction_max_tokens_per_batch: number;
  signal_candidate_keep_threshold: number;
  community_louvain_resolution: number;
  community_min_modularity_warn: number;
  graph_overview_max_nodes: number;
  graph_overview_max_edges: number;
  enable_model_fallback: boolean;
  enable_database_fallback: boolean;
  has_api_key: boolean;
  has_embedding_api_key: boolean;
  degraded_mode: boolean;
  runtime_settings_version?: string | null;
}

export interface ModelSettingsUpdate {
  api_key?: string | null;
  clear_api_key?: boolean;
  chat_base_url?: string | null;
  embedding_base_url?: string | null;
  chat_resolve_ip?: string | null;
  embedding_resolve_ip?: string | null;
  embedding_model?: string | null;
  chat_model?: string | null;
  embedding_dimensions?: number | null;
  worker_concurrency?: number | null;
  ingestion_file_concurrency?: number | null;
  model_request_concurrency?: number | null;
  model_request_timeout_seconds?: number | null;
  chunk_token_budget?: number | null;
  reranker_enabled?: boolean | null;
  reranker_model?: string | null;
  reranker_max_length?: number | null;
  reranker_device?: "cpu" | "cuda" | null;
  semantic_chunking_enabled?: boolean | null;
  semantic_chunking_min_length?: number | null;
  retrieval_layer_enabled?: boolean | null;
  retrieval_cache_ttl_seconds?: number | null;
  enable_agentic_reflection?: boolean | null;
  enable_post_generation_reflection?: boolean | null;
  citation_verification_sample_max?: number | null;
  reflection_max_retries?: number | null;
  model_bridge_enabled?: boolean | null;
  enable_graph_community_summaries?: boolean | null;
  signal_extraction_max_model_batches?: number | null;
  signal_extraction_max_candidates_per_batch?: number | null;
  signal_extraction_max_tokens_per_batch?: number | null;
  signal_candidate_keep_threshold?: number | null;
  community_louvain_resolution?: number | null;
  community_min_modularity_warn?: number | null;
  graph_overview_max_nodes?: number | null;
  graph_overview_max_edges?: number | null;
  embedding_api_key?: string | null;
  clear_embedding_api_key?: boolean;
}

export interface IngestionLogEvent {
  log_id?: string | null;
  timestamp: string;
  event: string;
  message: string;
  stage?: string;
  objective_mode?: string;
  source_path?: string;
  state?: string;
  processed_files?: number;
  total_files?: number;
  success_count?: number;
  failure_count?: number;
  skipped_count?: number;
  error?: string;
  provider?: string;
  model?: string;
  external_called?: boolean;
  fallback_reason?: string | null;
  vector_count?: number;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_external_called?: boolean;
  embedding_fallback_reason?: string | null;
  embedding_fallback_method?: string | null;
  graph_runtime?: string;
  graph_state_id?: string | null;
  graph_state_hash?: string | null;
  community_state_id?: string | null;
  community_state_hash?: string | null;
  atom_count?: number;
  evidence_atoms?: number;
  evidence_edges?: number;
  active_chunks?: number;
  chunk_candidates?: number;
  quality_decisions?: number;
  community_summary_count?: number;
  candidate_count?: number;
  signal_candidates?: number;
  signal_candidate_count?: number;
  accepted_signal_candidate_count?: number;
  rejected_signal_candidate_count?: number;
  signal_nodes?: number;
  signal_edges?: number;
  signal_node_count?: number;
  signal_edge_count?: number;
  signal_communities?: number;
  signal_community_count?: number;
  signal_state_id?: string | null;
  signal_state_hash?: string | null;
  signal_layer_status?: string;
  retry_count?: number;
  max_retries?: number;
}

export interface RuntimeIssue {
  code: string;
  title: string;
  message: string;
  fix_commands: string[];
}

export interface EnvSyncStatus {
  synced: boolean;
  missing_keys: string[];
  extra_keys: string[];
  deprecated_keys: string[];
  bom_keys: string[];
}

export interface RerankerRuntimeStatus {
  enabled: boolean;
  device: "cpu" | "cuda" | string;
  model: string;
  url: string;
  reachable: boolean;
  healthy: boolean;
  reported_model?: string | null;
  reported_device?: string | null;
  model_matches?: boolean | null;
  device_matches?: boolean | null;
}

export interface InfrastructureStatus {
  postgres: boolean;
  qdrant: boolean;
  redis: boolean;
  model_bridge?: boolean | null;
}

export interface RuntimeCheckResponse {
  env_sync: EnvSyncStatus;
  reranker: RerankerRuntimeStatus;
  infrastructure: InfrastructureStatus;
  blocking_issues: RuntimeIssue[];
  warnings: RuntimeIssue[];
}

export interface StructuredApiErrorBody {
  code: string;
  title: string;
  message: string;
  issues: RuntimeIssue[];
  fix_commands: string[];
}

export type GraphType = "evidence";
export type EvidenceSignalType = "topic" | "method" | "formula" | "metric" | "algorithm" | "definition" | "theorem" | "observation" | "claim";
export type SemanticEntityType = EvidenceSignalType | (string & {});
export type GraphNodeCategory =
  | "knowledge_base"
  | "evidence_graph_state"
  | "evidence_atom"
  | "active_chunk"
  | "signal_node"
  | "community_region"
  | "document"
  | "partition"
  | "section"
  | "evidence_chunk"
  | "document_version"
  | (string & {});

export interface GraphNode {
  id: string;
  name: string;
  category: GraphNodeCategory | string;
  value?: number;
  partition?: string | null;
  importance_score?: number | null;
  source_type?: string | null;
  entity_type?: SemanticEntityType | string | null;
  aliases?: string[];
  support_count?: number | null;
  support_atom_ids?: string[];
  support_active_chunk_ids?: string[];
  source_span_union?: Record<string, unknown> | null;
  confidence?: number | null;
  canonical_key?: string | null;
  signal_node_id?: string | null;
  summary?: string | null;
  document_id?: string | null;
  document_version_id?: string | null;
  snippet?: string | null;
  page_number?: number | null;
  evidence_count?: number | null;
  community_louvain?: number | null;
  community_spectral?: number | null;
  component_id?: number | null;
  centrality_score?: number | null;
  graph_rank_score?: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  confidence?: number | null;
  category?: string | null;
  evidence_chunk_id?: string | null;
  weight?: number | null;
  semantic_similarity?: number | null;
  support_count?: number | null;
  support_atom_ids?: string[];
  support_active_chunk_ids?: string[];
  source_span_union?: Record<string, unknown> | null;
  relation_source?: string | null;
  is_inferred?: boolean;
}

export interface GraphResponse {
  graph_type: GraphType;
  schema_version: string;
  view?: "overview" | "detail" | "neighborhood" | null;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_counts: Record<string, number>;
  edge_counts: Record<string, number>;
  focus_partition?: string | null;
  signal_layer_status?: string | null;
  signal_state_id?: string | null;
  signal_state_hash?: string | null;
  signal_layer_complete?: boolean;
  diagnostics?: Record<string, unknown>;
  freshness: {
    is_stale: boolean;
    reason?: string | null;
    latest_chunk_version?: string | null;
    active_chunk_versions?: string[];
    graph_chunk_version?: string | null;
    graph_chunk_versions?: string[];
    stale_evidence_chunks?: number;
    missing_evidence_chunks?: number;
    current_document_versions?: string[];
    graph_build_document_versions?: string[];
    uncovered_document_versions?: string[];
    removed_document_versions?: string[];
    graph_active_chunk_count?: number | null;
    current_active_chunk_count?: number;
    graph_build_id?: string | null;
    graph_built_at?: string | null;
    chunk_scope_changed?: boolean;
    strategy_profile_changed?: boolean;
    current_strategy_profile_hash?: string | null;
    graph_strategy_profile_hash?: string | null;
  };
}

export interface StrategyProfileSummary {
  id: string;
  name: string;
  library_type: string;
  is_builtin: boolean;
  profile_hash: string;
  is_active: boolean;
  knowledge_base_ids: string[];
  created_at?: string | null;
  updated_at?: string | null;
}

export interface StrategyProfileDetail extends StrategyProfileSummary {
  profile_json: Record<string, unknown>;
  warnings: string[];
}

export interface StrategyProfileMutationResponse {
  profile: StrategyProfileDetail;
  warnings: string[];
}

export interface StrategyProfileCreateRequest {
  name: string;
  library_type?: string;
  profile_json: Record<string, unknown>;
}

export interface StrategyProfileUpdateRequest {
  name?: string | null;
  library_type?: string | null;
  profile_json?: Record<string, unknown> | null;
}

export interface StrategyProfileCopyRequest {
  name: string;
}

export interface StrategyProfileBindRequest {
  knowledge_base_id?: string;
  profile_id: string;
}

export interface StrategyProfileDraftRequest {
  prompt: string;
  base_profile_id?: string | null;
  base_profile_json?: Record<string, unknown> | null;
}

export interface StrategyProfileDraftResponse {
  profile_json: Record<string, unknown>;
  warnings: string[];
  profile_hash?: string;
}

export interface StrategyProfileAssistantRequest {
  prompt: string;
  session_id?: string | null;
  base_profile_id?: string | null;
  base_profile_json?: Record<string, unknown> | null;
}

export interface StrategyProfileAssistantStateResponse {
  session_id: string;
  base_profile_id?: string | null;
  messages: Array<Record<string, unknown>>;
  latest_profile_json?: Record<string, unknown> | null;
  latest_profile_hash?: string | null;
  warnings: string[];
  draft_message?: string;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface KnowledgeBaseTreeNode {
  id: string;
  title: string;
  type: "knowledge_base" | "partition" | "document" | "signal";
  children?: KnowledgeBaseTreeNode[];
}

export interface KnowledgeBaseSummary {
  id: string;
  name: string;
  description?: string | null;
  source_root: string;
  storage_root: string;
  document_count: number;
  evidence_atom_count: number;
  current_chunk_version: number;
  has_parsed_chunks: boolean;
  can_full_reparse: boolean;
  degraded_mode: boolean;
  active_profile_id?: string | null;
  active_profile_name?: string | null;
  active_profile_hash?: string | null;
}

export interface KnowledgeBaseCreateRequest {
  name: string;
  description?: string | null;
}

export interface BatchError {
  source_path: string;
  message: string;
}

  export interface IngestionBatchSummary {
    batch_id: string;
    state: JobState;
    trigger_source: string;
    source_root: string;
  total_files: number;
  processed_files: number;
  success_count: number;
  failure_count: number;
    skipped_count: number;
    coverage_by_source_type: Record<string, number>;
    errors: BatchError[];
    graph_stats: Record<string, unknown>;
    phase?: string | null;
    parse_committed: boolean;
    cancellation_status?: string | null;
    worker_id?: string | null;
    heartbeat_at?: string | null;
    started_at?: string | null;
    completed_at?: string | null;
  }

export interface BatchStartResponse {
  batch_id: string;
  state: JobState;
}

export interface ParseUploadedFilesRequest {
  file_paths: string[];
  force?: boolean;
  full_reparse?: boolean;
}

export interface DashboardSnapshot {
  knowledge_base: KnowledgeBaseSummary;
  tree: KnowledgeBaseTreeNode[];
  graph: GraphResponse;
  batch_status?: IngestionBatchSummary | null;
  ingested_document_count: number;
  chunk_count?: number;
  evidence_atom_count?: number;
  active_chunk_count?: number;
  evidence_edge_count?: number;
  community_region_count?: number;
  graph_relation_count: number;
  coverage_by_source_type: Record<string, number>;
  degraded_mode: boolean;
}

export interface KnowledgeBaseFileSummary {
  id: string;
  document_id?: string | null;
  title: string;
  source_path: string;
  source_type: string;
  partition?: string | null;
  status: KnowledgeBaseFileStatus;
  job_state?: JobState | null;
  batch_id?: string | null;
  error?: string | null;
  chunk_count: number;
  chunk_version?: number | null;
  updated_at?: string | null;
}
