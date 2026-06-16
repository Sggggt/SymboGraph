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
  | "context_graph"
  | "cancel_requested"
  | "cancelling"
  | "terminating_task"
  | "compensating"
  | "cancelled"
  | "cancel_failed"
  | "processing"
  | "completed"
  | "partial_failed"
  | "failed"
  | "skipped";

export type KnowledgeBaseFileStatus = "pending" | "parsing" | "parsed" | "failed" | "skipped" | "active";

export type AgentRoute =
  | "layered_context_graph"
  | "direct_answer"
  | "multi_hop_research"
  | "definition_lookup"
  | "formula_table_lookup"
  | "cross_document_synthesis"
  | "clarify";

export type AgentRunState = "queued" | "running" | "needs_clarification" | "completed" | "failed";

export interface SearchFilters {
  document_ids?: string[];
  source_paths?: string[];
  partition?: string;
  tags?: string[];
  source_type?: SourceType | string;
  page_range?: [number | null, number | null] | null;
  content_kinds?: string[];
  chunk_version?: number | null;
}

export interface UploadFileResponse {
  document_id: string;
  job_id?: string | null;
  status: JobState | string;
  source_path: string;
}

export interface JobStatusResponse {
  job_id: string;
  state: JobState | string;
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

export interface Citation {
  chunk_id: string;
  document_id?: string | null;
  document_version_id?: string | null;
  title?: string | null;
  document_title?: string | null;
  source_path?: string | null;
  partition?: string | null;
  section?: string | null;
  section_path?: string[];
  page_number?: number | null;
  page_range?: [number | null, number | null] | number[] | null;
  char_span?: [number | null, number | null] | number[] | null;
  bbox?: Record<string, unknown> | null;
  text?: string | null;
  snippet?: string | null;
  source_span?: Record<string, unknown>;
  verification?: Record<string, unknown>;
  retrieval_trace_id?: string | null;
  citation_verification_id?: string | null;
}

export interface SearchResult {
  chunk_id: string;
  document_id?: string | null;
  document_version_id?: string | null;
  title?: string | null;
  snippet: string;
  text?: string | null;
  content?: string | null;
  score: number;
  citations: Citation[];
  metadata: Record<string, unknown>;
  source_path?: string | null;
  page_range?: [number | null, number | null] | number[] | null;
  char_span?: [number | null, number | null] | number[] | null;
  section_path?: string[];
  graph_path?: Array<Record<string, unknown>>;
  document_title?: string | null;
  partition?: string | null;
  source_type?: string | null;
}

export interface ModelAudit {
  provider?: string | null;
  embedding_provider?: string | null;
  embedding_model?: string | null;
  embedding_text_version?: string | null;
  embedding_external_called?: boolean;
  retrieval_mode?: string | null;
  retrieval_pipeline?: string | null;
  retrieval_trace_id?: string | null;
  context_package_id?: string | null;
  degraded?: boolean;
  degraded_mode?: boolean;
  fallback_used?: boolean;
  fallback_enabled?: boolean;
  latency_ms?: number | null;
  details?: Record<string, unknown>;
  cache?: Record<string, unknown>;
  cached?: boolean;
}

export interface AnswerModelAudit extends ModelAudit {
  chat_model?: string | null;
  model?: string | null;
  external_called?: boolean;
  skipped_reason?: string | null;
  answer_session_id?: string | null;
  citation_verification_pass_rate?: number | null;
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
  used_chunks?: Array<Record<string, unknown>>;
  route?: AgentRoute | string | null;
  trace?: AgentTraceEventPayload[];
  degraded_mode: boolean;
  model_audit?: AnswerModelAudit | Record<string, unknown>;
  answer_model_audit?: AnswerModelAudit;
  context_package_id?: string | null;
  retrieval_trace_id?: string | null;
}

export interface AgentRequest extends QARequest {
  stream_trace?: boolean;
}

export type AgentTraceNode =
  | "query_understanding"
  | "agent_planner"
  | "typed_action_validation"
  | "entry_selection"
  | "layer_drilldown"
  | "frontier_traversal"
  | "chunk_recall"
  | "structure_context_restoration"
  | "layered_retrieval"
  | "context_package"
  | "grounded_answer"
  | "citation_verification"
  | "repair_executed"
  | "reward_event"
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

export interface AgentResponse extends QAResponse {
  run_id: string;
  session_id: string;
  route: AgentRoute | string;
  trace: AgentTraceEventPayload[];
  answer_model_audit: AnswerModelAudit;
}

export interface TaskStatusResponse {
  run_id: string;
  state?: AgentRunState | string;
  status?: AgentRunState | string;
  current_node?: string | null;
  retry_count?: number;
  route?: AgentRoute | string | null;
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
  knowledge_base_id?: string | null;
  dry_run?: boolean;
  stats: {
    documents?: number;
    chunks?: number;
    vector_records?: number;
    qdrant_points?: number;
    deleted_vectors?: number;
    removed_vector_records?: number;
    deleted_chunks?: number;
    deleted_document_versions?: number;
    deleted_documents?: number;
    inactive_chunks?: number;
    stale_vector_records?: number;
    stale_qdrant_points?: number;
    collections?: string[];
    applied?: boolean;
    stale_vectors?: number;
    stale_bm25_records?: number;
    [key: string]: unknown;
  };
}

export interface RebuildGraphRequest {
  dry_run?: boolean;
  layers?: GraphType[];
}

export interface RebuildGraphResponse {
  batch_id?: string | null;
  state: string;
  mode: "four_layer_context_graph" | string;
  affected_documents: number;
  previous_batch_id?: string | null;
  dry_run?: boolean;
  stats?: Record<string, unknown>;
}

export interface BatchLogTokenResponse {
  batch_id?: string;
  token: string;
  expires_at: string;
}

export interface DeleteKnowledgeBaseResponse {
  deleted: boolean;
  knowledge_base_id?: string | null;
  knowledge_base_name?: string | null;
  stats: {
    deleted_vectors?: number;
    deleted_vector_records?: number;
    deleted_chunks?: number;
    deleted_document_versions?: number;
    deleted_documents?: number;
    deleted_context_graph_states?: number;
    deleted_retrieval_traces?: number;
    deleted_answer_sessions?: number;
    deleted_citation_verifications?: number;
    deleted_reward_events?: number;
    deleted_trace_events?: number;
    deleted_agent_runs?: number;
    deleted_sessions?: number;
    deleted_ingestion_logs?: number;
    deleted_compensations?: number;
    deleted_jobs?: number;
    deleted_batches?: number;
    deleted_knowledge_bases?: number;
    deleted_directory?: number;
    [key: string]: unknown;
  };
}

export interface RefreshResponse {
  knowledge_base_id?: string;
  refreshed_at: string;
}

export interface ModelSettingsResponse {
  provider?: "openai_compatible" | string;
  chat_base_url?: string;
  embedding_base_url?: string;
  effective_chat_base_url?: string;
  effective_embedding_base_url?: string;
  chat_resolve_ip?: string | null;
  embedding_resolve_ip?: string | null;
  embedding_model?: string;
  chat_model?: string;
  embedding_dimensions?: number;
  embedding_batch_size?: number;
  worker_concurrency?: number;
  model_request_concurrency?: number;
  model_request_timeout_seconds?: number;
  fixed_chunk_size_tokens?: number;
  fixed_chunk_overlap_tokens?: number;
  context_package_token_budget?: number;
  reranker_enabled?: boolean;
  reranker_model?: string;
  reranker_max_length?: number;
  reranker_device?: "cpu" | "cuda" | string;
  reranker_url?: string;
  model_bridge_enabled?: boolean;
  model_bridge_status?: ModelBridgeStatus;
  mid_concept_extraction_max_model_batches?: number;
  mid_concept_extraction_max_candidates_per_batch?: number;
  mid_concept_extraction_max_tokens_per_batch?: number;
  mid_concept_candidate_keep_threshold?: number;
  rq_kmeans_levels?: number;
  rq_kmeans_max_k?: number;
  rq_residual_tau?: number;
  agent_coarse_entry_budget?: number;
  agent_coarse_jump_budget?: number;
  agent_mid_entry_budget?: number;
  agent_mid_expansion_radius_cap?: number;
  agent_fine_entry_budget?: number;
  agent_frontier_expansion_budget?: number;
  agent_max_depth_per_layer?: number;
  agent_max_labels_per_node?: number;
  agent_max_edge_reuse?: number;
  agent_max_cycle_reward_per_path?: number;
  agent_ambiguous_edge_distance_low?: number;
  agent_ambiguous_edge_distance_high?: number;
  agent_drilldown_budget_per_layer?: number;
  agent_chunk_candidate_budget?: number;
  agent_structure_restore_budget?: number;
  context_path_summary_budget?: number;
  agent_planning_round_budget?: number;
  agent_max_typed_actions_per_round?: number;
  agent_repair_round_budget?: number;
  agent_verification_budget?: number;
  enable_model_fallback?: boolean;
  enable_database_fallback?: boolean;
  has_api_key?: boolean;
  has_embedding_api_key?: boolean;
  degraded_mode?: boolean;
  runtime_settings_version?: string | null;
  settings?: Record<string, unknown>;
  runtime_version?: Record<string, unknown>;
}

export interface ModelBridgeStatus {
  enabled?: boolean;
  base_url?: string;
  reachable?: boolean | null;
  admin_available?: boolean | null;
  config_matches?: boolean | null;
  chat_target_is_bridge?: boolean;
  embedding_target_is_bridge?: boolean;
  self_target_blocked?: boolean;
  config_version?: string | null;
  chat_target_hash?: string | null;
  embedding_target_hash?: string | null;
  desired_chat_target_hash?: string | null;
  desired_embedding_target_hash?: string | null;
  routes?: Record<string, string>;
  warnings?: string[];
  last_reload?: {
    attempted?: boolean;
    ok?: boolean;
    reason?: string;
    error?: string;
    status_code?: number;
    config_version?: string;
    chat_target_hash?: string;
    embedding_target_hash?: string;
  };
}

export interface ModelSettingsUpdate {
  api_key?: string | null;
  openai_api_key?: string | null;
  clear_api_key?: boolean;
  chat_base_url?: string | null;
  embedding_base_url?: string | null;
  chat_resolve_ip?: string | null;
  embedding_resolve_ip?: string | null;
  embedding_model?: string | null;
  chat_model?: string | null;
  embedding_dimensions?: number | null;
  embedding_batch_size?: number | null;
  worker_concurrency?: number | null;
  model_request_concurrency?: number | null;
  model_request_timeout_seconds?: number | null;
  fixed_chunk_size_tokens?: number | null;
  fixed_chunk_overlap_tokens?: number | null;
  context_package_token_budget?: number | null;
  reranker_enabled?: boolean | null;
  reranker_model?: string | null;
  reranker_max_length?: number | null;
  reranker_device?: "cpu" | "cuda" | string | null;
  model_bridge_enabled?: boolean | null;
  mid_concept_extraction_max_model_batches?: number | null;
  mid_concept_extraction_max_candidates_per_batch?: number | null;
  mid_concept_extraction_max_tokens_per_batch?: number | null;
  mid_concept_candidate_keep_threshold?: number | null;
  rq_kmeans_levels?: number | null;
  rq_kmeans_max_k?: number | null;
  rq_residual_tau?: number | null;
  agent_coarse_entry_budget?: number | null;
  agent_coarse_jump_budget?: number | null;
  agent_mid_entry_budget?: number | null;
  agent_mid_expansion_radius_cap?: number | null;
  agent_fine_entry_budget?: number | null;
  agent_frontier_expansion_budget?: number | null;
  agent_max_depth_per_layer?: number | null;
  agent_max_labels_per_node?: number | null;
  agent_max_edge_reuse?: number | null;
  agent_max_cycle_reward_per_path?: number | null;
  agent_ambiguous_edge_distance_low?: number | null;
  agent_ambiguous_edge_distance_high?: number | null;
  agent_drilldown_budget_per_layer?: number | null;
  agent_chunk_candidate_budget?: number | null;
  agent_structure_restore_budget?: number | null;
  context_path_summary_budget?: number | null;
  agent_planning_round_budget?: number | null;
  agent_max_typed_actions_per_round?: number | null;
  agent_repair_round_budget?: number | null;
  agent_verification_budget?: number | null;
  embedding_api_key?: string | null;
  clear_embedding_api_key?: boolean;
}

export interface IngestionLogEvent {
  log_id?: string | null;
  timestamp: string;
  event: string;
  message: string;
  stage?: string;
  phase?: string;
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
  embedding_provider?: string;
  embedding_model?: string;
  embedding_external_called?: boolean;
  embedding_fallback_reason?: string | null;
  embedding_fallback_method?: string | null;
  graph_runtime?: string | null;
  vector_count?: number;
  bm25_record_count?: number;
  chunk_count?: number;
  relation_edge_count?: number;
  fine_cluster_count?: number;
  mid_concept_count?: number;
  coarse_concept_count?: number;
  context_graph_hash?: string | null;
  context_graph_state_id?: string | null;
  context_graph_phase?: string | null;
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
  ok?: boolean;
  env_sync?: EnvSyncStatus;
  reranker?: RerankerRuntimeStatus;
  infrastructure?: InfrastructureStatus;
  model_bridge_status?: ModelBridgeStatus;
  blocking_issues?: RuntimeIssue[];
  warnings?: RuntimeIssue[];
  issues?: RuntimeIssue[];
  settings?: Record<string, unknown>;
  runtime_version?: Record<string, unknown>;
}

export interface StructuredApiErrorBody {
  code: string;
  title: string;
  message: string;
  issues: RuntimeIssue[];
  fix_commands: string[];
}

export type GraphType = "chunk-structure" | "chunk-relation" | "mid-concepts" | "coarse-concepts" | "context-graph";

export type GraphNodeCategory =
  | "knowledge_base"
  | "document"
  | "document_version"
  | "chunk"
  | "section"
  | "page"
  | "table"
  | "formula"
  | "caption"
  | "structure_node"
  | "chunk_relation"
  | "fine_cluster"
  | "mid_concept"
  | "coarse_concept"
  | "context_package"
  | (string & {});

export interface GraphNode {
  id: string;
  label?: string;
  name?: string;
  type?: string;
  category?: GraphNodeCategory | string;
  layer?: string | null;
  value?: number;
  score?: number | null;
  importance_score?: number | null;
  confidence?: number | null;
  support_count?: number | null;
  support_chunk_ids?: string[];
  support_fine_cluster_ids?: string[];
  representative_chunk_ids?: string[];
  included_mid_concept_ids?: string[];
  source_path?: string | null;
  document_id?: string | null;
  document_version_id?: string | null;
  snippet?: string | null;
  text?: string | null;
  page_number?: number | null;
  page_range?: [number | null, number | null] | number[] | null;
  section_path?: string[];
  metadata?: Record<string, unknown>;
}

export interface GraphEdge {
  id?: string;
  source: string;
  target: string;
  label?: string;
  type?: string;
  category?: string | null;
  confidence?: number | null;
  weight?: number | null;
  distance?: number | null;
  raw_strength?: number | null;
  score?: number | null;
  support_count?: number | null;
  support_chunk_ids?: string[];
  relation_source?: string | null;
  is_bridge?: boolean;
  is_inferred?: boolean;
  metadata?: Record<string, unknown>;
}

export interface GraphResponse {
  knowledge_base_id?: string;
  graph_type: GraphType | string;
  schema_version?: string;
  view?: "overview" | "detail" | "neighborhood" | null;
  nodes: GraphNode[];
  edges: GraphEdge[];
  counts: Record<string, number>;
  full_counts?: Record<string, number>;
  sampled_counts: Record<string, number>;
  node_counts?: Record<string, number>;
  edge_counts?: Record<string, number>;
  freshness: {
    is_stale?: boolean;
    reason?: string | null;
    [key: string]: unknown;
  };
  hash?: string | null;
  stale_reason?: string | null;
  grounding?: Record<string, unknown>;
  retrieval_contribution?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
}

export interface StrategyProfileSummary {
  id: string;
  name: string;
  library_type: string;
  is_builtin: boolean;
  profile_hash?: string;
  is_active?: boolean;
  knowledge_base_ids?: string[];
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
  title?: string;
  label?: string;
  type: "knowledge_base" | "section" | "document" | "chunk" | string;
  children?: KnowledgeBaseTreeNode[];
  metadata?: Record<string, unknown>;
}

export interface KnowledgeBaseSummary {
  id: string;
  name: string;
  description?: string | null;
  source_root?: string;
  storage_root?: string;
  document_count: number;
  chunk_count?: number;
  active_chunk_count?: number;
  current_chunk_version: number;
  has_parsed_chunks?: boolean;
  can_full_reparse?: boolean;
  degraded_mode?: boolean;
  context_graph_state_id?: string | null;
  context_graph_hash?: string | null;
  stale_reason?: string | null;
  active_profile_id?: string | null;
  active_profile_name?: string | null;
  active_profile_hash?: string | null;
}

export interface KnowledgeBaseCreateRequest {
  name: string;
  description?: string | null;
}

export interface BatchError {
  source_path?: string | null;
  message?: string | null;
}

export interface IngestionBatchSummary {
  batch_id: string;
  knowledge_base_id?: string;
  state: JobState | string;
  mode?: string | null;
  trigger_source?: string;
  source_root?: string;
  total_files: number;
  processed_files: number;
  success_count: number;
  failure_count: number;
  skipped_count: number;
  coverage_by_source_type?: Record<string, number>;
  errors?: BatchError[];
  graph_stats?: Record<string, unknown>;
  stats?: Record<string, unknown>;
  phase?: string | null;
  current_phase?: string | null;
  parse_committed?: boolean;
  cancellation_status?: string | null;
  cancel_failure_reason?: string | null;
  manual_review_required?: boolean;
  celery_task_id?: string | null;
  celery_task_name?: string | null;
  batch_task_ids?: string[];
  batch_worker_ids?: string[];
  worker_id?: string | null;
  heartbeat_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface BatchStartResponse {
  batch_id: string;
  state: JobState | string;
}

export interface ParseUploadedFilesRequest {
  file_paths?: string[] | null;
  force?: boolean;
  full_reparse?: boolean;
}

export interface DashboardSnapshot {
  knowledge_base: KnowledgeBaseSummary;
  tree: KnowledgeBaseTreeNode[];
  graph: GraphResponse;
  context_graph?: Record<string, unknown>;
  batch_status?: IngestionBatchSummary | null;
  recent_batches?: IngestionBatchSummary[];
  ingested_document_count: number;
  chunk_count?: number;
  graph_relation_count?: number;
  fine_cluster_count?: number;
  mid_concept_count?: number;
  coarse_concept_count?: number;
  coverage_by_source_type: Record<string, number>;
  degraded_mode: boolean;
  last_refreshed_at?: string | null;
}

export interface KnowledgeBaseFileSummary {
  id?: string;
  document_id?: string | null;
  title?: string | null;
  source_path: string;
  source_type?: string;
  partition?: string | null;
  status: KnowledgeBaseFileStatus;
  job_state?: JobState | string | null;
  batch_id?: string | null;
  error?: string | null;
  chunk_count?: number;
  current_version?: number | null;
  active_chunks?: number;
  checksum?: string | null;
  chunk_version?: number | null;
  updated_at?: string | null;
  last_ingested_at?: string | null;
}

export interface ContextPackageResponse {
  id: string;
  retrieval_trace_id?: string | null;
  knowledge_base_id: string;
  package_hash: string;
  query: string;
  contexts: Array<Record<string, unknown>>;
  token_budget: number;
  citation_spans: Array<Record<string, unknown>>;
  graph_expansion_paths: Array<Record<string, unknown>>;
  diagnostics: Record<string, unknown>;
  created_at?: string | null;
}

export interface RetrievalTraceStepsResponse {
  trace_id: string;
  steps: Array<Record<string, unknown>>;
}
