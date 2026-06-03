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
  | "cancelled"
  | "processing"
  | "completed"
  | "partial_failed"
  | "failed"
  | "skipped";

export type CourseFileStatus = "pending" | "parsing" | "parsed" | "failed" | "skipped";

export type AgentRoute =
  | "direct_answer"
  | "retrieve_notes"
  | "retrieve_exercises"
  | "retrieve_both"
  | "clarify"
  | "multi_hop_research";

export type AgentRunState = "queued" | "running" | "needs_clarification" | "completed" | "failed";

export interface SearchFilters {
  chapter?: string;
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
  course_id?: string | null;
  filters?: SearchFilters;
  top_k?: number;
}

export interface QuerySemanticGraphRequest {
  course_id?: string | null;
  query?: string | null;
  chunk_ids: string[];
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  source_path: string;
  chapter?: string | null;
  section?: string | null;
  page_number?: number | null;
  snippet: string;
}

export interface SearchResult {
  chunk_id: string;
  snippet: string;
  score: number;
  citations: Citation[];
  metadata: Record<string, unknown>;
  content?: string | null;
  child_content?: string | null;
  document_title?: string | null;
  source_path?: string | null;
  chapter?: string | null;
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
}

export interface AnswerModelAudit {
  provider: string;
  model?: string | null;
  external_called: boolean;
  fallback_reason?: string | null;
  skipped_reason?: string | null;
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
  course_id?: string | null;
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
  course_id?: string | null;
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
  removed_graph_relations: number;
  removed_graph_concepts: number;
}

export interface CleanupStaleGraphResponse {
  removed_relations: number;
  removed_aliases: number;
  removed_concepts: number;
  migrated_relations: number;
}

export interface RebuildGraphRequest {
  mode: "incremental" | "full";
  confirm_destructive?: boolean;
  dry_run?: boolean;
}

export interface RebuildGraphResponse {
  batch_id?: string | null;
  state: string;
  mode: string;
  affected_documents: number;
  previous_batch_id?: string | null;
  dry_run?: boolean;
  semantic_entities?: number;
  semantic_relations?: number;
}

export interface BatchLogTokenResponse {
  token: string;
  expires_at: string;
}

export interface DeleteCourseResponse {
  deleted: boolean;
  deleted_vectors: number;
  deleted_trace_events: number;
  deleted_agent_runs: number;
  deleted_sessions: number;
  deleted_ingestion_logs: number;
  deleted_compensations: number;
  deleted_jobs: number;
  deleted_graph_extraction_tasks: number;
  deleted_graph_extraction_runs: number;
  deleted_batches: number;
  deleted_hpo_judge_samples: number;
  deleted_hpo_objective_models: number;
  deleted_model_hyperparameters: number;
  deleted_quality_profiles: number;
  deleted_community_summaries: number;
  deleted_relation_candidates: number;
  deleted_mentions: number;
  deleted_merge_candidates: number;
  deleted_relations: number;
  deleted_aliases: number;
  deleted_concepts: number;
  deleted_chunks: number;
  deleted_document_versions: number;
  deleted_documents: number;
  deleted_courses: number;
  deleted_directory: number;
}

export interface RefreshResponse {
  course_id: string;
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
  graph_extraction_strategy: string;
  graph_extraction_soft_start_budget?: number | null;
  graph_extraction_max_input_tokens_per_run?: number | null;
  graph_extraction_max_model_calls_per_run?: number | null;
  graph_extraction_min_marginal_gain: number;
  graph_extraction_stall_rounds: number;
  graph_extraction_concurrency: number;
  graph_extraction_resume_batch_size: number;
  reranker_enabled: boolean;
  reranker_model: string;
  reranker_max_length: number;
  reranker_device: "cpu" | "cuda";
  reranker_url: string;
  semantic_chunking_enabled: boolean;
  semantic_chunking_min_length: number;
  model_bridge_enabled: boolean;
  enable_auto_hpo: boolean;
  has_api_key: boolean;
  has_embedding_api_key: boolean;
  degraded_mode: boolean;
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
  graph_extraction_strategy?: string | null;
  graph_extraction_soft_start_budget?: number | null;
  graph_extraction_max_input_tokens_per_run?: number | null;
  graph_extraction_max_model_calls_per_run?: number | null;
  graph_extraction_min_marginal_gain?: number | null;
  graph_extraction_stall_rounds?: number | null;
  graph_extraction_concurrency?: number | null;
  graph_extraction_resume_batch_size?: number | null;
  reranker_enabled?: boolean | null;
  reranker_model?: string | null;
  reranker_max_length?: number | null;
  reranker_device?: "cpu" | "cuda" | null;
  semantic_chunking_enabled?: boolean | null;
  semantic_chunking_min_length?: number | null;
  model_bridge_enabled?: boolean | null;
  enable_auto_hpo?: boolean | null;
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
  graph_extraction_provider?: string;
  graph_extraction_model?: string;
  graph_extraction_completed_chunks?: number;
  graph_llm_success_chunks?: number;
  graph_algorithm_nodes?: number;
  graph_algorithm_edges?: number;
  graph_rejected_concepts?: number;
  concepts?: number;
  relations?: number;
  community_summary_count?: number;
  retry_count?: number;
  max_retries?: number;
  candidate_count?: number;
  pair_count?: number;
  processed_pairs?: number;
  effective_labels?: number;
  min_labels?: number;
  trial_count?: number;
  best_value?: number;
  objective_model_id?: string | null;
  feature_summary?: Record<string, unknown>;
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
  bom_keys: string[];
}

export interface RerankerRuntimeStatus {
  enabled: boolean;
  device: string;
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

export type GraphRelationType =
  | "is_a"
  | "part_of"
  | "prerequisite_of"
  | "used_for"
  | "causes"
  | "derives_from"
  | "compares_with"
  | "example_of"
  | "defined_by"
  | "formula_of"
  | "solves"
  | "implemented_by"
  | "related_to";

export interface RelatedConcept {
  concept_id: string;
  relation_type: GraphRelationType;
  target_name: string;
  confidence?: number | null;
  weight?: number | null;
  relation_source?: string | null;
  is_inferred?: boolean;
}

export interface ConceptCard {
  concept_id: string;
  name: string;
  aliases: string[];
  summary: string;
  chapter_refs: string[];
  concept_type: string;
  importance_score: number;
  related_concepts: RelatedConcept[];
}

export type GraphType = "semantic" | "structural" | "evidence";
export type SemanticEntityType = "concept" | "method" | "formula" | "metric" | "algorithm" | "definition" | "theorem" | "problem_type";
export type GraphNodeCategory = "semantic_entity" | "course" | "document" | "chapter" | "section" | "chunk" | "evidence_chunk" | "document_version";

export interface GraphNode {
  id: string;
  name: string;
  category: GraphNodeCategory | string;
  value?: number;
  chapter?: string | null;
  importance_score?: number | null;
  source_type?: string | null;
  entity_type?: SemanticEntityType | string | null;
  aliases?: string[];
  support_count?: number | null;
  confidence?: number | null;
  canonical_key?: string | null;
  concept_id?: string | null;
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
  relation_source?: string | null;
  is_inferred?: boolean;
}

export interface GraphResponse {
  graph_type: GraphType;
  schema_version: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_counts: Record<string, number>;
  edge_counts: Record<string, number>;
  focus_chapter?: string | null;
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
  };
}

export interface CourseTreeNode {
  id: string;
  title: string;
  type: "course" | "chapter" | "document" | "concept";
  children?: CourseTreeNode[];
}

export interface CourseSummary {
  id: string;
  name: string;
  description?: string | null;
  source_root: string;
  storage_root: string;
  document_count: number;
  concept_count: number;
  degraded_mode: boolean;
}

export interface CourseCreateRequest {
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
  rebuild_graph_mode?: "none" | "incremental" | "full";
  confirm_destructive_graph_rebuild?: boolean;
}

export interface DashboardSnapshot {
  course: CourseSummary;
  tree: CourseTreeNode[];
  graph: GraphResponse;
  batch_status?: IngestionBatchSummary | null;
  ingested_document_count: number;
  graph_relation_count: number;
  coverage_by_source_type: Record<string, number>;
  degraded_mode: boolean;
}

export interface CourseFileSummary {
  id: string;
  document_id?: string | null;
  title: string;
  source_path: string;
  source_type: string;
  chapter?: string | null;
  status: CourseFileStatus;
  job_state?: JobState | null;
  batch_id?: string | null;
  error?: string | null;
  chunk_count: number;
  updated_at?: string | null;
}

export interface GraphNodeRelation {
  relation_id: string;
  relation_type: GraphRelationType;
  target_concept_id?: string | null;
  target_name: string;
  confidence: number;
  weight?: number | null;
  semantic_similarity?: number | null;
  support_count?: number | null;
  relation_source?: string | null;
  is_inferred?: boolean;
  evidence?: Citation | null;
}

export interface GraphNodeDetail {
  concept_id: string;
  name: string;
  normalized_name: string;
  summary: string;
  aliases: string[];
  chapter_refs: string[];
  concept_type: string;
  importance_score: number;
  evidence_count: number;
  community_louvain?: number | null;
  community_spectral?: number | null;
  component_id?: number | null;
  centrality: Record<string, number>;
  graph_rank_score: number;
  relations: GraphNodeRelation[];
}
