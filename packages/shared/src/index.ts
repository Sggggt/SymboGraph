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

export type RetrievalGranularity = "mid" | "coarse";

export type AgentRunState =
  | "queued"
  | "running"
  | "needs_clarification"
  | "completed"
  | "failed"
  | "cancelled";

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

export interface UploadReplacementLockReleaseDiagnostics {
  resource_key: string;
  knowledge_base_id: string;
  advisory_key: number;
  backend: "postgresql";
  operation: string;
  batch_id: string | null;
  protocol_version: string;
  release_error: string;
}

export interface LanguageIdentitySummary {
  status: "pending" | "resolved";
  language?: string | null;
  source?: "explicit_metadata" | "deterministic_detection" | "unknown" | null;
  confidence?: number | null;
  protocol_version?: string | null;
  detection_hash?: string | null;
  explicit_language_tag?: string | null;
  decision_reason?: string | null;
}

export interface UploadFileMetadata {
  language?: string | null;
}

export interface UploadFileResponse {
  document_id: string;
  job_id: string;
  status: JobState | string;
  source_path: string;
  language_identity: LanguageIdentitySummary;
  upload_replacement: {
    protocol_version: string;
    intent_id: string;
    status: "completed" | "cleanup_pending" | "manual_review";
    phase: "completed" | "cleanup_pending" | "manual_review";
    database_committed: boolean;
    cleanup_pending: boolean;
    postcommit_lock_release_failure?: UploadReplacementLockReleaseDiagnostics | null;
    lock_release_audit?: {
      persisted: boolean;
      intent_id: string;
      error?: string | null;
    } | null;
  };
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
  session_id?: string | null;
  filters?: SearchFilters;
  top_k?: number;
  retrieval_granularity?: RetrievalGranularity;
}

export interface ConversationUserConstraints {
  instructions: string[];
  retrieval_filters: SearchFilters;
}

export interface ConversationTaskState {
  status: "active" | "waiting_user" | "completed" | "cancelled";
  objective?: string | null;
  current_step?: string | null;
}

export interface ConversationStateUpdate {
  active_user_constraints?: ConversationUserConstraints | null;
  task_state?: ConversationTaskState | null;
}

export interface ConversationHistoryReference {
  protocol_version: "answer_context_citation_reference_v1";
  turn_index: number;
  run_id: string;
  answer_session_id: string;
  context_package_id: string;
  retrieval_trace_id: string;
  citation_verification_ids: string[];
}

export interface ConversationStatePayload {
  protocol_version: "conversation_state_v1";
  scope_protocol_version: "conversation_state_scope_v1";
  qa_session_id?: string | null;
  knowledge_base_id: string;
  revision: number;
  state_hash: string;
  scope_hash: string;
  active_user_constraints: ConversationUserConstraints;
  task_state: ConversationTaskState;
  history_references: ConversationHistoryReference[];
  transcript_message_count: number;
  prompt_history_audit: Record<string, unknown>;
  evidence_authority: false;
  gray_zone_decision_authority: false;
}

export interface CitationBoundingBox {
  page_number?: number | null;
  x0?: number | null;
  y0?: number | null;
  x1?: number | null;
  y1?: number | null;
  coordinate_system?: string | null;
  synthetic?: boolean | null;
  raw_bbox?: number[] | null;
  raw_coordinate_system?: string | null;
  page_size?: number[] | null;
  source_bbox_count?: number | null;
}

export interface SourceSnapshotVerification {
  protocol_version: string;
  final_open_protocol_version: string;
  storage_path: string;
  checksum: string;
  verified: true;
  size_bytes: number;
}

export interface UploadReplacementRecoveryHealth {
  protocol_version: "upload_replacement_recovery_health_v1";
  status: "not_run" | "healthy" | "degraded";
  last_run_at?: string | null;
  knowledge_bases: number;
  selected: number;
  completed: number;
  rolled_back: number;
  cleanup_pending: number;
  manual_review: number;
  failed: number;
  retryable: boolean;
}

export interface StorageMaintenanceRecoveryHealth {
  protocol_version: "storage_maintenance_recovery_health_v1";
  status: "not_run" | "healthy" | "degraded";
  last_run_at?: string | null;
  selected: number;
  completed: number;
  pending: number;
  cache_pending: number;
  external_pending: number;
  manual_review: number;
  failed: number;
  retryable: boolean;
}

export interface CitationSourceSpan {
  contract_version: "raw_chunk_source_span_v1";
  document_version_id: string;
  chunk_id: string;
  source_path: string;
  source_checksum: string;
  logical_source_path: string;
  source_snapshot_verification: SourceSnapshotVerification;
  chunk_text_hash_protocol_version: string;
  chunk_text_hash: string;
  raw_span_text_hash_protocol_version: string;
  raw_span_text_hash: string;
  char_span: [number, number] | number[];
  raw_chunk_char_span?: [number, number] | number[] | null;
  page_range: [number | null, number | null] | Array<number | null>;
  section_path?: string | string[] | null;
  structure_path?: string | string[] | null;
  structure_node_ids: string[];
  bbox?: CitationBoundingBox | null;
  context_package_id?: string | null;
  retrieval_trace_id?: string | null;
  verification_id?: string | null;
  content_clipped: boolean;
  content_token_count?: number | null;
}

export interface VerifiedCitationSourceSpan extends CitationSourceSpan {
  context_package_id: string;
  retrieval_trace_id: string;
  verification_id: string;
}

export interface CitationVerificationDiagnostics {
  verification_method: string;
  claim_grounded_gate_protocol_version: string;
  claim_id: string | null;
  claim_index: number | null;
  answer_hash: string | null;
  citation_provenance_protocol_version: string;
  citation_provenance_valid: boolean | null;
  citation_provenance_hash: string | null;
  citation_provenance_reasons: string[];
  citation_provenance_fail_closed: true;
  citation_provenance_llm_override_allowed: false;
  citation_provenance_session_hash: string | null;
  citation_provenance_persistence_gate_passed: boolean | null;
  llm_entailment_judge?: string | null;
  rule_verdict:
    | "supported"
    | "unsupported"
    | "contradicted"
    | "missing_citation"
    | "structure_context_missing"
    | "formula_table_context_missing"
    | null;
  llm_entailment_verdict:
    | "supported"
    | "unsupported"
    | "contradicted"
    | "missing_citation"
    | "structure_context_missing"
    | "formula_table_context_missing"
    | null;
  llm_entailment_result_present: boolean;
  llm_entailment_reason?: string | null;
  deterministic_exact_span_entailment?: boolean;
  deterministic_exact_span_entailment_protocol_version?: string | null;
  citation_prompt_protocol_hash: string | null;
  citation_grounding_envelope_protocol_version: string | null;
  citation_grounding_envelope_hash: string | null;
  citation_profile_hash: string | null;
  citation_verification_microbatch_protocol_version?: string | null;
  citation_verification_microbatch_size?: number | null;
  citation_verification_model_call_count?: number | null;
  reason?: string | null;
}

export interface CitationVerificationAudit {
  contract_version: "citation_verification_public_v1";
  verdict:
    | "supported"
    | "unsupported"
    | "contradicted"
    | "missing_citation"
    | "structure_context_missing"
    | "formula_table_context_missing";
  failure_type: string;
  provenance_status: "valid" | "invalid" | "missing";
  structure_context_status: "valid" | "invalid" | "missing";
  confidence: number;
  diagnostics: CitationVerificationDiagnostics;
}

export interface Citation {
  contract_version: "citation_public_v1";
  chunk_id: string;
  citation_index: number;
  claim_id: string | null;
  claim_index: number | null;
  claim_text: string | null;
  answer_hash: string | null;
  document_id: string;
  document_version_id: string;
  title?: string | null;
  document_title?: string | null;
  source_path: string;
  logical_source_path: string;
  partition?: string | null;
  section?: string | string[] | null;
  section_path: string[];
  page_number?: number | null;
  page_range: [number | null, number | null] | Array<number | null>;
  char_span: [number, number] | number[];
  context_package_id: string;
  bbox?: CitationBoundingBox | null;
  text?: string | null;
  snippet?: string | null;
  source_span: VerifiedCitationSourceSpan;
  verification: CitationVerificationAudit;
  retrieval_trace_id: string;
  answer_session_id: string;
  citation_verification_id: string;
}

export interface SearchCitation {
  contract_version: "search_citation_public_v1";
  chunk_id: string;
  document_id: string;
  document_title?: string | null;
  source_path: string;
  logical_source_path: string;
  partition?: string | null;
  section?: string | string[] | null;
  page_number?: number | null;
  snippet?: string | null;
  retrieval_trace_id: string;
  source_span: CitationSourceSpan;
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
  citations: SearchCitation[];
  metadata: Record<string, unknown>;
  source_path?: string | null;
  logical_source_path?: string | null;
  page_range?: [number | null, number | null] | number[] | null;
  char_span?: [number | null, number | null] | number[] | null;
  section_path?: string[];
  graph_path?: Array<Record<string, unknown>>;
  document_title?: string | null;
  partition?: string | null;
  source_type?: string | null;
}

export interface RetrievalCacheAudit {
  protocol_version: "layered_retrieval_postgresql_strict_replay_v1";
  status: "hit" | "miss" | "poison" | "unavailable";
  cache_hit: boolean;
  cache_miss: boolean;
  cache_key: string;
  redis_key_digest: string;
  ttl_seconds: number;
  ttl_seconds_remaining?: number | null;
  reason: string;
  deletion_attempted: boolean;
  deleted: boolean;
  source_retrieval_trace_id?: string | null;
  source_context_package_id?: string | null;
  postgresql_replay_required: true;
  redis_payload_used_as_evidence: false;
  provider_perception_model_call_count?: 0 | null;
  query_embedding_model_call_count?: 0 | null;
  traversal_execution_count?: 0 | null;
  retrieval_fact_insert_count?: 0 | null;
  gray_zone_input_modified: false;
  gray_zone_model_call_count: 0;
  context_package_reused?: boolean | null;
  write_scheduled_after_commit?: boolean | null;
  ordinary_query_pointer_write_scheduled_after_commit?: boolean | null;
}

export interface OrdinaryQueryReplayPointerAudit {
  status: "hit" | "miss" | "poison" | "unavailable" | "stale";
  reason: string;
  pointer_key_digest: string;
  ttl_seconds_remaining?: number | null;
  deletion_attempted?: boolean;
  deleted?: boolean;
  source_retrieval_trace_id?: string | null;
  source_context_package_id?: string | null;
  full_cache_probe?: RetrievalCacheAudit | null;
}

export interface OrdinaryQueryPerceptionAudit {
  protocol_version: "bounded_query_perception_and_facet_proposal_v1";
  provider_protocol_hash: string;
  model_call_budget: 2;
  model_call_count: number;
  budget_exhausted: false;
  intent_schema_validated: true;
  facet_schema_validated: boolean;
  query_intent_hash: string;
  conversation_history_hash: string;
  conversation_history_turn_count: number;
  conversation_history_audit_hash: string;
  conversation_history_is_evidence: false;
  conversation_history_gray_zone_decision_authority: false;
  query_facet_packet_hash: string;
  query_facet_packet_is_evidence: false;
  query_facet_packet_routing_only: true;
  suggested_strategy_is_executor_authority: false;
  retrieval_granularity_is_user_or_executor_locked: true;
  gray_zone_decision_authority: false;
  gray_zone_rule_inputs_modified: false;
  gray_zone_model_call_count: 0;
  provider_free_pointer: OrdinaryQueryReplayPointerAudit;
}

export interface QueryEmbeddingExecutionAudit {
  protocol_version: "request_scoped_query_embedding_memo_v1";
  request_memo_enabled: boolean;
  request_memo_hit: boolean;
  request_memo_key_hash: string;
  query_embedding_model_call_count: 0 | 1;
  provider_response_present: false;
  credentials_present: false;
  gray_zone_decision_authority: false;
  gray_zone_model_call_count: 0;
}

export interface ModelAuditFields {
  provider?: string | null;
  prompt_protocol_version?: string | null;
  prompt_protocol_hash?: string | null;
  grounding_envelope_protocol_version?: string | null;
  grounding_envelope_hash?: string | null;
  profile_hash?: string | null;
  embedding_model?: string | null;
  embedding_text_version?: string | null;
  retrieval_mode?: string | null;
  retrieval_granularity?: RetrievalGranularity | null;
  retrieval_trace_id?: string | null;
  context_package_id?: string | null;
  conversation_state_scope_hash?: string | null;
  semantic_entry_query_protocol_version?: "validated_query_facet_semantic_entry_v1" | null;
  semantic_entry_query_hash?: string | null;
  semantic_entry_query_selection_source?: "validated_required_facet" | "raw_query" | null;
  semantic_entry_query_is_evidence?: false | null;
  semantic_entry_query_citation_authority?: false | null;
  semantic_entry_query_gray_zone_decision_authority?: false | null;
  degraded?: boolean;
  degraded_mode?: boolean;
  fallback_used?: boolean;
  fallback_enabled?: boolean;
  latency_ms?: number | null;
  route?: string | null;
  retrieval_pipeline?: string | null;
  context_graph_state_id?: string | null;
  result_top_k?: number | null;
  coarse_entries?: number;
  mid_entries?: number;
  rq_membership_entries?: number;
  stage_queue_count?: number;
  mid_topk_selected?: number;
  chunk_topk_selected?: number;
  frontier_pops?: number;
  dominance_pruned_count?: number;
  hard_stop_pruned_count?: number;
  red_zone_pruned_count?: number;
  gray_zone_decision_count?: number;
  query_rq_path?: number[];
  coarse_skipped_reason?: string | null;
  mid_direct_entry_count?: number | null;
  repair_protocol_version?: string | null;
  repair_action_type?: string | null;
  repair_executor_mechanism?: string | null;
  repair_directive_hash?: string | null;
  repair_global_top_k_modified?: false | null;
  repair_gray_zone_model_call_count?: 0 | null;
  retrieval_cache?: RetrievalCacheAudit | null;
  query_perception_audit?: OrdinaryQueryPerceptionAudit | null;
  query_embedding_execution?: QueryEmbeddingExecutionAudit | null;
}

export interface ModelAudit extends ModelAuditFields {
  contract_version: "model_audit_public_v1";
}

export interface ExpectedEvidenceAudit {
  source?: string | null;
  requires_chunk_spans?: boolean | null;
  required_facets: string[];
  allowed_relation_types: string[];
  relation_types: string[];
  required_restore_modes: string[];
  minimum_independent_support_paths?: number | null;
  required_evidence_roles: string[];
  failure_types: string[];
  start_layer?: string | null;
  target_layer?: string | null;
  fallback_allowed?: boolean | null;
  required_verification_stage?: string | null;
  protocol_version?: string | null;
  executor_mechanism?: string | null;
  failure_card_hashes: string[];
  action_input_hash?: string | null;
}

export interface EvidenceEvaluatorVerdictAudit {
  protocol_version?: string | null;
  verdict:
    | "sufficient"
    | "need_more_same_node"
    | "need_bridge_jump"
    | "need_mid_expansion"
    | "need_chunk_expansion"
    | "need_structure_closure"
    | "insufficient_corpus"
    | "validator_rejection";
  reason: string;
  target_ids: string[];
  expected_evidence: ExpectedEvidenceAudit;
  profile_hash?: string | null;
  prompt_protocol_hash?: string | null;
  schema_repair_attempted?: boolean;
  decision_hash?: string | null;
}

export interface ClaimGroundedClaimAudit {
  claim_id: string;
  claim_index: number;
  claim_text: string;
  answer_hash: string;
  claim_id_protocol_version: string;
  supported: boolean;
  candidate_verification_count: number;
  supported_verification_count: number;
  supported_citation_indexes: number[];
  supported_chunk_ids: string[];
  failure_types: string[];
}

export interface ClaimGroundedGateAudit {
  protocol_version: string;
  answer_hash: string;
  claim_id_protocol_version: string;
  claim_count: number;
  supported_claim_count: number;
  unsupported_claim_count: number;
  claim_pass_rate: number;
  all_claims_supported: boolean;
  supported_claim_ids: string[];
  unsupported_claim_ids: string[];
  claims: ClaimGroundedClaimAudit[];
  unbound_verification_count: number;
  unbound_verification_hash: string;
  require_persistence_replay: boolean;
  gate_hash: string;
  nonfactual_insufficiency_response?: boolean | null;
}

export interface EvidenceGapAudit {
  kind?: "unsupported_claims_removed" | "no_supported_claims" | null;
  dropped_claim_count?: number | null;
  dropped_claim_ids: string[];
  dropped_claim_texts: string[];
  kept_claim_ids: string[];
  repair_convergence_reason?: string | null;
  repair_round_budget?: number | null;
  repair_rounds_used?: number | null;
  unsupported_claims_removed?: boolean | null;
  original_answer_hash?: string | null;
  original_claim_count?: number | null;
  original_supported_claim_count?: number | null;
  original_unsupported_claim_count?: number | null;
  original_claim_pass_rate?: number | null;
  pre_guard_gate_hash?: string | null;
}

export interface RepairCanonicalTargetRefsAudit {
  claim_ids: string[];
  source_chunk_ids: string[];
  source_context_package_id: string;
  source_retrieval_trace_id?: string | null;
  mid_concept_ids: string[];
  target_refs_hash: string;
}

export interface RepairValidatedTargetsAudit {
  action_target_ids: string[];
  canonical_target_refs: RepairCanonicalTargetRefsAudit;
  supported_source_chunk_ids: string[];
  carry_forward_supported_chunk_ids: string[];
  bridge_seed_chunk_ids: string[];
  excluded_mid_ids: string[];
  excluded_result_chunk_ids: string[];
}

export interface RepairRebindCandidateAudit {
  claim_id: string;
  chunk_id: string;
  exact_span_match: boolean;
  support_score: number;
  meaningful_overlap_count: number;
}

export interface RepairCurrentPackageRebindAudit {
  protocol_version?: string | null;
  candidate_count: number;
  candidates: RepairRebindCandidateAudit[];
  preferred_claim_chunk_ids: Record<string, string>;
  gray_zone_decision_authority?: false | null;
  gray_zone_model_call_count?: 0 | null;
  rebind_input_hash?: string | null;
  verification_attempted?: boolean | null;
  verification_deferred_to_caller?: boolean | null;
  supported_claim_gain: string[];
  supported_claim_regression: string[];
}

export interface RepairExecutionAudit {
  executor_mechanism: string;
  layered_search_called: boolean;
  source_chunk_ids: string[];
  search_audit?: ModelAudit | null;
  current_package_rebind?: RepairCurrentPackageRebindAudit | null;
  gray_zone_model_call_count: 0;
  gray_zone_decision_authority: "deterministic_executor_only";
  conversation_state_scope_hash: string;
  retrieval_granularity: RetrievalGranularity;
  result_top_k: number;
  global_top_k_increased: false;
  answer_regenerated: false;
  source_agent_operating_envelope_hash: string;
  repaired_agent_operating_envelope_hash: string;
  source_traversal_protocol_hash: string;
  repaired_traversal_protocol_hash: string;
  source_path_distance_threshold_hash: string;
  repaired_path_distance_threshold_hash: string;
  gray_zone_protocol_and_thresholds_frozen: true;
  candidate_context_package_id?: string | null;
  candidate_retrieval_trace_id?: string | null;
  supported_claim_regression_rejected: string[];
  regression_fail_closed?: true | null;
}

export interface RepairProgressSpanAudit {
  chunk_id: string;
  document_version_id: string;
  char_span: number[];
  raw_span_text_hash: string;
}

export interface RepairProgressPayloadAudit {
  result_chunk_ids: string[];
  package_chunk_spans: RepairProgressSpanAudit[];
  covered_facets: string[];
  evidence_roles: string[];
  graph_path_ids: string[];
  supported_claim_ids: string[];
  unsupported_claim_ids: string[];
}

export interface RepairProgressAudit {
  protocol_version: string;
  payload: RepairProgressPayloadAudit;
  progress_hash: string;
}

export interface RepairRoundAudit {
  action_type:
    | "repair_missing_citation"
    | "repair_concept_gap"
    | "repair_bridge_gap"
    | "repair_structure_context";
  protocol_version: string;
  repair_round_index: number;
  remaining_repair_budget_before: number;
  remaining_repair_budget_after: number;
  executor_mechanism: string;
  action_input_hash: string;
  action_output_hash: string;
  failure_card_hashes: string[];
  before_failure_types: string[];
  after_failure_types: string[];
  before_context_package_id: string;
  repaired_context_package_id: string;
  before_retrieval_trace_id?: string | null;
  repaired_retrieval_trace_id?: string | null;
  before_progress: RepairProgressAudit;
  after_progress: RepairProgressAudit;
  before_progress_hash: string;
  after_progress_hash: string;
  made_semantic_progress: boolean;
  repair_candidate_reverted: boolean;
  convergence_reason: string;
  retrieval_granularity: RetrievalGranularity;
  conversation_state_scope_hash: string;
  query_facets_hash: string;
  result_top_k: number;
  global_top_k_increased: false;
  gray_zone_model_call_count: 0;
  gray_zone_decision_authority: "deterministic_executor_only";
  repair_audit: RepairExecutionAudit;
  validated_targets: RepairValidatedTargetsAudit;
}

export interface FinalGroundedGateRepairAudit {
  action_type: "claim_level_final_grounded_gate";
  protocol_version: string;
  typed_action_control_hash: string;
  grounding_outcome: "grounded_answer" | "insufficient_evidence";
  exact_answer_hash: string;
  claim_grounded_gate: ClaimGroundedGateAudit;
  evidence_gap: EvidenceGapAudit;
  deterministic_citation_guard: true;
  gray_zone_model_call_count: 0;
}

export type RepairActionAudit = RepairRoundAudit | FinalGroundedGateRepairAudit;

export interface ProviderPromptCacheAudit {
  protocol_version: string;
  api_protocol: "openai" | "anthropic";
  cache_mode: string;
  cacheable_system_prompt_present: boolean;
  cacheable_system_prompt_sha256?: string | null;
  cacheable_system_prompt_utf8_bytes?: number | null;
  provider_response_persisted: false;
}

export interface ProviderUsageAudit {
  protocol_version: string;
  api_protocol: "openai" | "anthropic";
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_hit: boolean;
  cache_write: boolean;
  token_accounting_mode: string;
  usage_present: boolean;
  provider_response_persisted: false;
}

export interface ProviderCallAudit {
  protocol_version: string;
  prompt_cache: ProviderPromptCacheAudit;
  usage: ProviderUsageAudit;
  provider_response_persisted: false;
}

export interface AnswerModelAudit extends ModelAuditFields {
  contract_version: "answer_model_audit_public_v1";
  model?: string | null;
  external_called?: boolean | null;
  fallback_reason?: string | null;
  chat_model?: string | null;
  agent_plan_id?: string | null;
  agent_plan_index?: number | null;
  planning_rounds_used?: number | null;
  typed_action_control_hash?: string | null;
  evidence_evaluator?: EvidenceEvaluatorVerdictAudit | null;
  context_package_evidence_gate_passed?: boolean | null;
  answer_model_called?: boolean | null;
  answer_claim_limit?: number | null;
  output_token_budget?: number | null;
  output_token_budget_protocol_version?: string | null;
  provider_call?: ProviderCallAudit | null;
  answer_session_id?: string | null;
  citation_verification_pass_rate?: number | null;
  raw_citation_verification_pass_rate?: number | null;
  repair_protocol_version?: string | null;
  repair_round_budget?: number | null;
  repair_rounds_used?: number | null;
  repair_convergence_reason?: string | null;
  repair_actions: RepairActionAudit[];
  claim_grounded_gate_protocol_version?: string | null;
  claim_grounded_gate?: ClaimGroundedGateAudit | null;
  exact_answer_hash?: string | null;
  evidence_gap: EvidenceGapAudit;
  citation_guard_applied: boolean;
  unsupported_claims_removed: boolean;
  insufficient_evidence: boolean;
  grounding_outcome?: string | null;
  returned_citation_count?: number | null;
}

export interface SearchResponse {
  contract_version: "search_public_v1";
  query: string;
  results: SearchResult[];
  degraded_mode: boolean;
  model_audit: ModelAudit;
  retrieval_trace_id?: string | null;
  context_package_id?: string | null;
  retrieval_granularity?: RetrievalGranularity;
  conversation_state?: ConversationStatePayload | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface QARequest {
  question: string;
  session_id?: string | null;
  knowledge_base_id?: string | null;
  filters?: SearchFilters;
  top_k?: number;
  history?: ChatMessage[];
  conversation_state_update?: ConversationStateUpdate | null;
  retrieval_granularity?: RetrievalGranularity;
}

export interface QAResponse {
  contract_version: "qa_public_v1";
  run_id?: string | null;
  session_id?: string | null;
  answer: string;
  citations: Citation[];
  used_chunks: ContextItem[];
  route?: AgentRoute | string | null;
  trace?: AgentTraceEventPayload[];
  degraded_mode: boolean;
  model_audit: AnswerModelAudit;
  answer_model_audit?: AnswerModelAudit | null;
  context_package_id?: string | null;
  retrieval_trace_id?: string | null;
  retrieval_granularity?: RetrievalGranularity;
  conversation_state?: ConversationStatePayload | null;
}

export interface AgentRequest extends QARequest {
  stream_trace?: boolean;
}

export interface AgentQueryIntentConversationAudit {
  active_user_constraints: ConversationUserConstraints;
  task_state: ConversationTaskState;
  history_references: ConversationHistoryReference[];
  conversation_text_is_evidence: false;
  historical_references_are_evidence: false;
  gray_zone_decision_authority: false;
}

export interface AgentQueryIntentAudit {
  intent: string;
  entities: string[];
  sub_queries: string[];
  needs_graph: boolean;
  history_turns?: number | null;
  suggested_strategy?: string | null;
  conversation_state?: AgentQueryIntentConversationAudit | null;
}

export interface AgentActionValidationResult {
  valid: boolean;
  schema_checked?: boolean | null;
  budget_checked?: boolean | null;
  target_ids_checked?: boolean | null;
  target_scope_checked?: boolean | null;
  target_layers: Record<string, string[]>;
  fallback_disabled_checked?: boolean | null;
  bridge_protection_checked?: boolean | null;
  required_restore_modes: string[];
  required_verification_stage?: string | null;
  inserted_required_action?: boolean | null;
}

export interface AgentActionValidationAccepted {
  index?: number | null;
  accepted_index: number;
  action_type: string;
  validation: AgentActionValidationResult;
}

export interface AgentActionValidationDetail {
  key?: string | null;
  reason?: string | null;
  requested?: number | null;
  limit?: number | null;
  keys: string[];
  target_id?: string | null;
  layers: string[];
}

export interface AgentActionValidationRejected {
  index?: number | null;
  action_type?: string | null;
  reason: string;
  missing_fields: string[];
  extra_fields: string[];
  forbidden_mentions: string[];
  fields: string[];
  target_ids: string[];
  details: Array<AgentActionValidationDetail | string>;
  requested_start_layer?: string | null;
  retrieval_granularity?: RetrievalGranularity | null;
  input_action_count?: number | null;
  max_typed_actions_per_round?: number | null;
  rejected_count?: number | null;
  required_action_count?: number | null;
  limit?: number | null;
}

export interface AgentTypedActionValidationAudit {
  typed_action_schema_protocol_version: string;
  typed_action_schema_protocol_hash: string;
  accepted: AgentActionValidationAccepted[];
  rejected: AgentActionValidationRejected[];
  inserted_required_actions: string[];
  fallback_disabled: boolean;
  required_restore_modes: string[];
  allowed_relation_types: string[];
  required_actions_enforced: boolean;
  retrieval_granularity?: RetrievalGranularity | null;
  input_action_count: number;
  input_scan_limit: number;
  valid: boolean;
  plan_index?: number | null;
  retrieval_granularity_locked?: RetrievalGranularity | null;
  unsupported_retrieval_granularity_rewrites_rejected?: true | null;
}

export interface AgentStopConditionRequestAudit {
  sufficient_evidence?: boolean | null;
  required_action_complete?: boolean | null;
  all_required_facets_covered?: boolean | null;
  independent_support_paths_at_least?: number | null;
  citation_verification_passes?: boolean | null;
  frontier_empty?: boolean | null;
  all_claims_supported?: boolean | null;
  no_semantic_progress?: boolean | null;
}

export interface AgentStopConditionResultAudit {
  sufficient_evidence?: boolean | null;
  required_action_complete?: boolean | null;
  all_required_facets_covered?: boolean | null;
  independent_support_paths_at_least?: boolean | null;
  citation_verification_passes?: boolean | null;
  frontier_empty?: boolean | null;
  all_claims_supported?: boolean | null;
  no_semantic_progress?: boolean | null;
}

export interface AgentStopConditionEvaluationAudit {
  action_id: string;
  action_index: number;
  action_type: string;
  requested: AgentStopConditionRequestAudit;
  results: AgentStopConditionResultAudit;
  triggered: boolean;
}

export interface AgentActionStopConditionsAudit {
  evaluations: AgentStopConditionEvaluationAudit[];
  triggered_action_indexes: number[];
  stop_triggered: boolean;
  stop_condition_hash: string;
}

export interface AgentReplanProgressAudit {
  protocol_version: "agent_replan_semantic_progress_v1";
  semantic_progress_signature?: string | null;
  evaluator_directive_hash?: string | null;
  matching_prior_plan_indexes: number[];
  no_progress: boolean;
  phase?: "before_retrieval_execution" | null;
  plan_index?: number | null;
  typed_action_control_hash?: string | null;
  prior_plan_index?: number | null;
  reason?: "typed_actions_targets_budgets_and_controls_unchanged" | null;
  retrieval_execution_count?: 0 | null;
  evidence_evaluator_model_call_count?: 0 | null;
  gray_zone_decision_authority: false;
  gray_zone_model_call_count: 0;
  audit_hash: string;
}

export type AgentTraceNode =
  | "query_understanding"
  | "query_facet_extraction"
  | "agent_planner"
  | "typed_action_validation"
  | "typed_action_executor"
  | "evidence_evaluator"
  | "replan_no_progress"
  | "evidence_gate"
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
  | "cancelled"
  | "agent_admission"
  | "error";

export interface AgentTraceScoresFields {
  contract_version: "agent_trace_scores_public_v1";
}

export type AgentTraceScores = AgentTraceScoresFields &
  (
    | {
        audit_kind: "query_understanding";
        top_k?: number | null;
        query_intent?: AgentQueryIntentAudit | null;
        retrieval_granularity?: RetrievalGranularity | null;
      }
    | {
        audit_kind: "query_facets";
        query_facets?: QueryFacetPacket | null;
        retrieval_granularity?: RetrievalGranularity | null;
      }
    | {
        audit_kind: "planner";
        plan_id?: string | null;
        plan_index?: number | null;
        replan?: boolean | null;
        agent_operating_envelope_hash?: string | null;
        retrieval_granularity?: RetrievalGranularity | null;
      }
    | {
        audit_kind: "typed_action_validation";
        plan_id?: string | null;
        plan_index?: number | null;
        validation?: AgentTypedActionValidationAudit | null;
      }
    | {
        audit_kind: "typed_action_executor";
        plan_id?: string | null;
        plan_index?: number | null;
        typed_action_control_hash?: string | null;
        effective_result_top_k?: number | null;
        retrieval_trace_id?: string | null;
      }
    | {
        audit_kind: "evidence_evaluator";
        plan_id?: string | null;
        plan_index?: number | null;
        verdict?: EvidenceEvaluatorVerdictAudit | null;
        replan_requested?: boolean | null;
        replan_candidate?: boolean | null;
        replan_no_progress?: boolean | null;
        replan_progress?: AgentReplanProgressAudit | null;
        evaluator_requests_replan?: boolean | null;
        insufficient_corpus_terminal_deferred?: boolean | null;
        action_stop_condition_triggered?: boolean | null;
        action_stop_conditions?: AgentActionStopConditionsAudit | null;
        planning_rounds_remaining?: number | null;
        gray_zone_model_call_count?: 0 | null;
      }
    | {
        audit_kind: "replan_progress";
        replan_progress: AgentReplanProgressAudit;
      }
    | { audit_kind: "evidence_gate"; answer_model_audit?: AnswerModelAudit | null }
    | {
        audit_kind: "retrieval_stage";
        retrieval_trace_id?: string | null;
        retrieval_granularity?: RetrievalGranularity | null;
        coarse_entries?: number | null;
        stage_queue_count?: number | null;
        mid_topk_selected?: number | null;
        chunk_topk_selected?: number | null;
        query_rq_path: number[];
        frontier_pops?: number | null;
        dominance_pruned_count?: number | null;
        chunk_ids: string[];
        gray_zone_model_call_count?: 0 | null;
      }
    | { audit_kind: "layered_retrieval"; retrieval_audit?: ModelAudit | null }
    | {
        audit_kind: "context_restoration";
        context_package_id?: string | null;
        hit_chunks?: number | null;
        restored_chunks?: number | null;
        bridge_chunks?: number | null;
        parent_structure_nodes?: number | null;
        graph_path_ids?: number | null;
      }
    | {
        audit_kind: "context_package";
        context_package_id?: string | null;
        token_count?: number | null;
      }
    | { audit_kind: "grounded_answer"; answer_model_audit?: AnswerModelAudit | null }
    | { audit_kind: "repair"; repair_action?: RepairActionAudit | null }
    | {
        audit_kind: "citation_verification";
        citation_pass_rate?: number | null;
        raw_citation_pass_rate?: number | null;
        verification_count?: number | null;
        returned_citation_count?: number | null;
        repair_actions: RepairActionAudit[];
      }
    | {
        audit_kind: "reward";
        runtime_settings_hash?: string | null;
        agent_operating_envelope_hash?: string | null;
      }
    | {
        audit_kind: "status";
        cancel_requested?: boolean | null;
        admission_failure?: boolean | null;
      }
  );

export interface AgentTraceEventPayload {
  contract_version: "agent_trace_event_public_v1";
  type: "trace";
  id?: string | null;
  run_id: string;
  sequence_index: number;
  node: AgentTraceNode;
  status: string;
  input_summary: string;
  output_summary: string;
  document_ids: string[];
  scores: AgentTraceScores;
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
  session_id?: string | null;
  state?: AgentRunState | string;
  status?: AgentRunState | string;
  current_node?: string | null;
  retry_count?: number;
  route?: AgentRoute | string | null;
  retrieval_granularity?: RetrievalGranularity;
  answer?: string | null;
  error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  trace?: AgentTraceEventPayload[];
}

export type AgentPEActionType =
  | "activate_coarse_concepts"
  | "route_mid_concepts"
  | "route_rq_addresses"
  | "select_entry_nodes"
  | "walk_graph_frontier"
  | "drill_down_layer"
  | "jump_bridge"
  | "stop_and_collect_chunks"
  | "need_more_evidence"
  | "recall_chunks"
  | "restore_context_package"
  | "build_context_package"
  | "verify_citations"
  | "repair_missing_citation"
  | "repair_concept_gap"
  | "repair_bridge_gap"
  | "repair_structure_context";

export type AgentPEObservationType =
  | "plan_validation_failed"
  | "executor_contract_blocked"
  | "entry_selection"
  | "layer_routing"
  | "frontier_traversal"
  | "chunk_recall"
  | "repair_gate"
  | "evidence_evaluator"
  | "replan_gate"
  | "evidence_gate_blocked"
  | "context_restoration"
  | "context_package_built"
  | "citation_verification"
  | "typed_repair_round"
  | "claim_level_final_grounded_gate";

export interface AgentPEJsonPayload {
  encoding: "canonical_json_v1";
  canonical_json: string;
  sha256: string;
  redacted_fields: string[];
}

export interface AgentPEActionValidatorAudit {
  valid?: boolean | null;
  plan_valid?: boolean | null;
  schema_checked?: boolean | null;
  budget_checked?: boolean | null;
  target_ids_checked?: boolean | null;
  target_scope_checked?: boolean | null;
  typed_action_schema_protocol_version?: string | null;
  typed_action_schema_protocol_hash?: string | null;
  repair_protocol_version?: string | null;
  repair_budget_checked?: boolean | null;
  repair_round_index?: number | null;
  remaining_repair_budget_before?: number | null;
  action_input_hash?: string | null;
  repair_directive_validator_protocol_version?: string | null;
  repair_directive_validator_result?: string | null;
  repair_directive_hash?: string | null;
  validated_directive_hash?: string | null;
  payload: AgentPEJsonPayload;
}

export interface AgentPEEvaluatorLinkage {
  plan_id: string;
  plan_index: number;
  protocol_version?: string | null;
  verdict:
    | "sufficient"
    | "need_more_same_node"
    | "need_bridge_jump"
    | "need_mid_expansion"
    | "need_chunk_expansion"
    | "need_structure_closure"
    | "insufficient_corpus";
  decision_hash?: string | null;
  replan_requested: boolean;
  gray_zone_model_call_count: 0;
  schema_repair_attempted?: boolean;
}

export interface AgentPERepairLinkage {
  action_id: string;
  parent_action_id?: string | null;
  action_type:
    | "repair_missing_citation"
    | "repair_concept_gap"
    | "repair_bridge_gap"
    | "repair_structure_context";
  repair_protocol_version?: string | null;
  repair_round_index: number;
  remaining_repair_budget_before: number;
  remaining_repair_budget_after: number;
  action_input_hash?: string | null;
  action_output_hash?: string | null;
  before_context_package_id?: string | null;
  repaired_context_package_id?: string | null;
  before_retrieval_trace_id?: string | null;
  repaired_retrieval_trace_id?: string | null;
}

export interface AgentPEPlanAuditRow {
  contract_version: "agent_plan_audit_row_v1";
  order_index: number;
  id: string;
  run_id: string;
  knowledge_base_id: string;
  retrieval_trace_id?: string | null;
  plan_index: number;
  planner_protocol_version?: string | null;
  typed_action_schema_protocol_version?: string | null;
  typed_action_schema_protocol_hash?: string | null;
  typed_action_executor_protocol_version?: string | null;
  input_hash?: string | null;
  output_hash?: string | null;
  control_hash?: string | null;
  query_intent: AgentPEJsonPayload;
  operating_envelope: AgentPEJsonPayload;
  typed_actions: AgentPEJsonPayload;
  validation: AgentPEJsonPayload;
  planner_model_metadata: AgentPEJsonPayload;
  status:
    | "validated"
    | "invalid"
    | "validator_replan_requested"
    | "executor_contract_blocked"
    | "replan_requested"
    | "evidence_sufficient"
    | "insufficient_corpus"
    | "planning_budget_exhausted";
  diagnostics: AgentPEJsonPayload;
  action_ids: string[];
  action_count: number;
  redacted_fields: string[];
  created_at: string;
}

export interface AgentPEActionAuditRow {
  contract_version: "agent_action_audit_row_v1";
  order_index: number;
  id: string;
  run_id: string;
  plan_id: string;
  plan_index: number;
  parent_action_id?: string | null;
  action_index: number;
  action_type: AgentPEActionType;
  target_ids: string[];
  reason: string;
  budget_request: AgentPEJsonPayload;
  expected_evidence: AgentPEJsonPayload;
  stop_condition: AgentPEJsonPayload;
  validator: AgentPEActionValidatorAudit;
  status: "accepted" | "completed" | "rejected" | "deferred" | "no_progress";
  input_hash?: string | null;
  output_hash?: string | null;
  control_hash?: string | null;
  output: AgentPEJsonPayload;
  diagnostics: AgentPEJsonPayload;
  observation_ids: string[];
  observation_count: number;
  redacted_fields: string[];
  created_at: string;
}

export interface AgentPEObservationAuditRow {
  contract_version: "agent_observation_audit_row_v1";
  order_index: number;
  id: string;
  run_id: string;
  plan_id: string;
  plan_index: number;
  action_id?: string | null;
  action_index?: number | null;
  parent_action_id?: string | null;
  observation_type: AgentPEObservationType;
  protocol_version?: string | null;
  input_hash?: string | null;
  output_hash?: string | null;
  control_hash?: string | null;
  evaluator_linkage?: AgentPEEvaluatorLinkage | null;
  repair_linkage?: AgentPERepairLinkage | null;
  evidence_chunk_ids: string[];
  verdict: string;
  observation: AgentPEJsonPayload;
  diagnostics: AgentPEJsonPayload;
  redacted_fields: string[];
  created_at: string;
}

export interface AgentPEAuditResponse {
  contract_version: "agent_pe_audit_public_v1";
  run_id: string;
  knowledge_base_id: string;
  run_status: string;
  counts: {
    plans: number;
    actions: number;
    observations: number;
  };
  ordering: {
    plans: "plan_index ASC, created_at ASC, id ASC";
    actions: "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC";
    observations: "created_at ASC, id ASC";
  };
  plans: AgentPEPlanAuditRow[];
  actions: AgentPEActionAuditRow[];
  observations: AgentPEObservationAuditRow[];
  redaction_protocol_version: "semantic_sensitive_field_key_segments_v1";
  provider_raw_response_exposed: false;
  credentials_exposed: false;
}

export interface SessionSummary {
  id: string;
  knowledge_base_id: string;
  title?: string | null;
  last_question?: string | null;
  last_answer?: string | null;
  transcript: SessionMessage[];
  conversation_state: ConversationStatePayload;
  created_at: string;
  updated_at: string;
}

export interface SessionMessage {
  role: "user" | "assistant";
  content: string;
  run_id?: string | null;
  route?: string | null;
  retrieval_trace_id?: string | null;
  citations: Citation[];
  citation_replay_status?: "not_present" | "valid" | "unavailable";
  citation_replay_reason?: "persisted_citation_contract_mismatch" | null;
  source?: "client_history" | null;
}

export interface SessionMessagesResponse {
  session_id: string;
  messages: SessionMessage[];
  conversation_state: ConversationStatePayload;
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

export interface EdgeTypeCalibrationParams {
  lower_quantile: number;
  upper_quantile: number;
  min_span: number;
  strength_floor: number;
}

export interface AutoTpeGraphOperatingPointTheta {
  graph_operating_point_protocol: "dense_dynamic_knn_bridge_quota_edge_calibration_v2";
  optimizer: "auto_tpe_lightweight" | "auto_tpe_lightweight_or_default";
  edge_type_calibration_protocol: "type_local_winsorized_minmax_v1";
  edge_type_calibration_protocol_hash: string;
  calibration_params: EdgeTypeCalibrationParams;
  edge_distance_protocol: "edge_distance_log_calibrated_strength_v2";
  edge_distance_protocol_hash: string;
  rank_score_protocol_version: "channel_percentile_rank_v1";
  rank_score_protocol_hash: string;
  raw_strength_protocol_version: "dense_relation_raw_strength_v3";
  raw_strength_protocol_hash: string;
  chunk_node_quality_protocol: "chunk_node_quality_intrinsic_v2";
  chunk_node_quality_protocol_hash: string;
  out_evidence_mass_protocol: "relation_out_evidence_mass_v2";
  out_evidence_mass_protocol_hash: string;
  in_acceptance_capacity_protocol: "relation_in_acceptance_capacity_current_scope_v3";
  in_acceptance_capacity_protocol_hash: string;
  relation_quota_protocol: "dynamic_knn_reverse_quota_signals_v3";
  relation_quota_protocol_hash: string;
  quota_signal_scale: 16;
  dense_knn_k_min: number;
  dense_knn_k_max: number;
  dense_reverse_b_min_base: number;
  dense_reverse_b_max_base: number;
  dense_reverse_b_min_doc: number;
  dense_reverse_b_max_doc: number;
  dense_reverse_b_min_lang: number;
  dense_reverse_b_max_lang: number;
  dense_min_cosine: number;
  dense_strong_cosine: number;
  cross_doc_out_quota_min: number;
  cross_doc_out_quota_max: number;
  cross_doc_min_cosine: number;
  cross_language_out_quota_min: number;
  cross_language_out_quota_max: number;
  cross_language_min_cosine: number;
}

export interface AutoTpeTrialSummary {
  trial_id: string;
  run_id: string;
  knowledge_base_id: string;
  build_batch_id?: string | null;
  chunk_scope_hash: string;
  embedding_model: string;
  embedding_text_version: string;
  trial_index: number;
  status: string;
  sampled_theta_json: AutoTpeGraphOperatingPointTheta | null;
  theta_hash?: string | null;
  tpe_search_space_hash?: string | null;
  edge_distance_protocol?: "edge_distance_log_calibrated_strength_v2" | null;
  edge_distance_protocol_hash?: string | null;
  edge_type_calibration_protocol?: "type_local_winsorized_minmax_v1" | null;
  edge_type_calibration_protocol_hash?: string | null;
  calibration_params?: EdgeTypeCalibrationParams | null;
  calibration_params_hash?: string | null;
  edge_type_calibration_config_hash?: string | null;
  sampler_state_hash?: string | null;
  runtime_settings_hash: string;
  gate_profile_hash: string;
  gate_profile: Record<string, unknown>;
  candidate_adjacency_hash?: string | null;
  probe_set_hash?: string | null;
  objective_score?: number | null;
  hard_gate?: Record<string, unknown>;
  objective_components?: Record<string, unknown>;
  failure_code?: string | null;
  diagnostics?: Record<string, unknown>;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface AutoTpeRunSummary {
  run_id: string;
  knowledge_base_id: string;
  batch_id?: string | null;
  chunk_relation_graph_state_id?: string | null;
  chunk_version: number;
  chunk_scope_hash?: string | null;
  graph_operating_point_protocol?: string | null;
  protocol_hash?: string | null;
  chat_model?: string | null;
  embedding_model?: string | null;
  embedding_text_version?: string | null;
  status: string;
  trigger_reason?: string | null;
  trial_budget?: number;
  startup_random_trials?: number;
  good_quantile_gamma?: number | null;
  probe_query_budget?: number;
  candidate_pool_size?: number;
  best_trial_id?: string | null;
  best_objective_score?: number | null;
  selected_theta_hash?: string | null;
  tpe_search_space_hash?: string | null;
  selected_theta?: AutoTpeGraphOperatingPointTheta | null;
  selected_edge_distance_protocol?: "edge_distance_log_calibrated_strength_v2" | null;
  selected_edge_distance_protocol_hash?: string | null;
  selected_edge_type_calibration_protocol?: "type_local_winsorized_minmax_v1" | null;
  selected_edge_type_calibration_protocol_hash?: string | null;
  selected_calibration_params?: EdgeTypeCalibrationParams | null;
  selected_calibration_params_hash?: string | null;
  selected_edge_type_calibration_config_hash?: string | null;
  sampler_state_hash?: string | null;
  probe_set_hash?: string | null;
  hard_gate?: Record<string, unknown>;
  objective_components?: Record<string, unknown>;
  last_error?: string | null;
  failure_code?: string | null;
  blocking_reasons?: string[];
  runtime_settings_hash?: string | null;
  selected_graph_runtime_settings_hash?: string | null;
  selected_gate_profile_hash?: string | null;
  selected_gate_profile?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
  trials?: AutoTpeTrialSummary[];
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface AutoTpeStatusResponse {
  knowledge_base_id: string;
  current_chunk_version?: number;
  enabled?: boolean;
  latest_run?: AutoTpeRunSummary | null;
}

export interface BatchLogTokenResponse {
  batch_id: string;
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

export interface RuntimeSettingsOperatingPointGate {
  required: boolean;
  stages: string[];
  hard_gates: string[];
}

export interface RuntimeSettingsRedaction {
  secret_fields: string[];
  payload_exposes_secret_values: false;
}

export interface RuntimeSettingsLifecycle {
  hot_reloadable: string[];
  rebuild_required: string[];
  service_recreate_required: string[];
  candidate_version_required_for: string[];
  fixed_protocol: { rq_kmeans_levels: 3 };
  operating_point_gate: RuntimeSettingsOperatingPointGate;
  redaction: RuntimeSettingsRedaction;
}

export interface ModelSettingsResponse {
  provider?: "multi_protocol" | string;
  chat_api_protocol?: "openai" | "anthropic";
  graph_api_protocol?: "openai" | "anthropic";
  embedding_api_protocol?: "openai";
  chat_base_url?: string;
  graph_base_url?: string;
  embedding_base_url?: string;
  effective_chat_base_url?: string;
  effective_graph_base_url?: string;
  effective_embedding_base_url?: string;
  chat_resolve_ip?: string | null;
  graph_resolve_ip?: string | null;
  embedding_resolve_ip?: string | null;
  embedding_model?: string;
  chat_model?: string;
  graph_model?: string;
  embedding_dimensions?: number;
  embedding_batch_size?: number;
  worker_concurrency?: number;
  model_request_concurrency?: number;
  model_request_timeout_seconds?: number;
  chat_json_max_tokens?: number;
  agent_request_concurrency?: number;
  source_io_concurrency?: number;
  agent_request_queue_limit?: number;
  agent_request_queue_timeout_seconds?: number;
  agent_request_lease_ttl_seconds?: number;
  upload_max_bytes?: number;
  concept_i18n_enabled?: boolean;
  query_facet_bilingual_enabled?: boolean;
  query_facet_posterior_enabled?: boolean;
  query_facet_posterior_observation_budget?: number;
  query_facet_posterior_round_budget?: number;
  query_facet_posterior_convergence_epsilon?: number;
  fixed_chunk_size_tokens?: number;
  fixed_chunk_overlap_tokens?: number;
  context_package_token_budget?: number;
  model_bridge_enabled?: boolean;
  model_bridge_status?: ModelBridgeStatus;
  mid_concept_extraction_max_model_batches?: number;
  mid_concept_extraction_max_candidates_per_batch?: number;
  mid_concept_extraction_max_tokens_per_batch?: number;
  mid_concept_candidate_keep_threshold?: number;
  rq_kmeans_levels?: 3;
  rq_kmeans_max_k?: number;
  rq_residual_tau?: number;
  edge_distance_protocol?: "edge_distance_log_calibrated_strength_v2";
  rq_membership_protocol?: "rq_fuzzy_softmax_gamma_product_v1";
  edge_projection_protocol?: "membership_q15_layer_type_calibrated_v3";
  edge_type_calibration_protocol?: "type_local_winsorized_minmax_v1";
  rq_membership_temperature?: number;
  rq_membership_top_m?: number;
  rq_membership_probability_threshold?: number;
  dense_knn_k_min?: number;
  dense_knn_k_max?: number;
  dense_reverse_b_min_base?: number;
  dense_reverse_b_max_base?: number;
  dense_reverse_b_min_doc?: number;
  dense_reverse_b_max_doc?: number;
  dense_reverse_b_min_lang?: number;
  dense_reverse_b_max_lang?: number;
  dense_min_cosine?: number;
  dense_strong_cosine?: number;
  cross_doc_out_quota_min?: number;
  cross_doc_out_quota_max?: number;
  cross_doc_min_cosine?: number;
  cross_language_out_quota_min?: number;
  cross_language_out_quota_max?: number;
  cross_language_min_cosine?: number;
  enable_auto_tpe?: boolean;
  tpe_trial_budget?: number;
  tpe_startup_random_trials?: number;
  tpe_good_quantile_gamma?: number;
  tpe_probe_query_budget?: number;
  tpe_trial_timeout_seconds?: number;
  tpe_candidate_pool_size?: number;
  operating_point_hard_gate_max_edge_density?: number;
  operating_point_hard_gate_max_isolated_ratio?: number;
  operating_point_hard_gate_max_hubness_ratio?: number;
  operating_point_hard_gate_min_structure_recovery_rate?: number;
  operating_point_hard_gate_max_candidate_latency_p95_ms?: number;
  retrieval_result_top_k_default?: number;
  agent_coarse_initial_budget?: number;
  agent_coarse_total_budget?: number;
  agent_coarse_top_k?: number;
  agent_mid_per_coarse_budget?: number;
  agent_coarse_drilldown_mid_initial_budget?: number;
  agent_mid_initial_budget?: number;
  agent_mid_top_k?: number;
  agent_chunk_per_mid_budget?: number;
  agent_chunk_initial_budget?: number;
  agent_chunk_top_k?: number;
  candidate_pool_dedupe_budget?: number;
  agent_max_depth_per_layer?: number;
  agent_max_labels_per_node?: number;
  agent_max_edge_reuse?: number;
  agent_max_cycle_reward_per_path?: number;
  agent_cycle_reward_distance_threshold?: number;
  agent_path_distance_green_threshold?: number;
  agent_path_distance_gray_threshold?: number;
  agent_path_distance_hard_threshold?: number;
  gray_zone_rule_protocol?: "deterministic_support_progress_v1";
  gray_zone_observation_cadence?: number;
  traversal_observation_budget?: number;
  agent_structure_restore_per_chunk_budget?: number;
  agent_structure_restore_budget?: number;
  context_path_summary_budget?: number;
  agent_planning_round_budget?: number;
  agent_max_typed_actions_per_round?: number;
  agent_repair_round_budget?: number;
  agent_verification_budget?: number;
  enable_model_fallback?: boolean;
  enable_database_fallback?: boolean;
  has_chat_api_key?: boolean;
  has_graph_api_key?: boolean;
  has_embedding_api_key?: boolean;
  degraded_mode?: boolean;
  runtime_settings_version?: string | null;
  settings_revision?: string | null;
  setting_statuses?: Record<
    string,
    | "written_and_applied"
    | "written_pending_hot_apply"
    | "written_pending_rebuild"
    | "written_pending_service_recreate"
  >;
  pending_rebuild_changes?: string[];
  pending_service_recreate_changes?: string[];
  pending_hot_changes?: string[];
  settings_file_synced?: boolean;
  requires_service_recreate?: boolean;
  service_recreate_changes?: string[];
  active_mutated?: boolean | null;
  runtime_version_broadcast?: boolean | null;
  runtime_version_broadcast_pending?: boolean | null;
  runtime_local_refresh_pending?: boolean | null;
  apply_error_type?: string | null;
  lifecycle?: RuntimeSettingsLifecycle;
}

export interface RuntimeSettingsCandidateCreate {
  knowledge_base_ids: string[];
  settings: Record<string, unknown>;
  dry_run_only?: boolean;
  source?: string;
}

export interface RuntimeSettingsCandidateActionRequest {
  build_id?: string | null;
  reason?: string | null;
}

export interface RuntimeSettingsCandidateBuild {
  id: string;
  knowledge_base_id: string;
  status: string;
  vector_shadow_build_id?: string | null;
  base_chunk_scope_hash?: string | null;
  candidate_chunk_scope_hash?: string | null;
  candidate_chunk_version?: number;
  candidate_chunk_schema_version?: string;
  shadow_context_graph_state_id?: string | null;
  build_metrics?: Record<string, unknown>;
  evaluation?: Record<string, unknown>;
  blocking_reasons?: string[];
}

export interface RuntimeSettingsCandidateDetail {
  id: string;
  protocol_version: string;
  candidate_hash: string;
  effective_runtime_settings_hash?: string | null;
  base_runtime_version_hash?: string | null;
  status: string;
  changed_keys: string[];
  target_knowledge_base_ids: string[];
  settings: Record<string, unknown>;
  blocking_reasons: string[];
  diagnostics: Record<string, unknown>;
  builds: RuntimeSettingsCandidateBuild[];
  activation_intents: Array<{
    id: string;
    direction: "promotion" | "rollback" | string;
    status: string;
    attempt_count: number;
    runtime_version_hash?: string | null;
    last_error_type?: string | null;
  }>;
}

export interface RuntimeSettingsCandidateResponse {
  candidate?: RuntimeSettingsCandidateDetail | null;
  preview?: Record<string, unknown> | null;
  action?: Record<string, unknown>;
}

export interface ModelBridgeOperationStatus {
  attempted?: boolean;
  ok?: boolean;
  reason?: string;
  error?: string;
  status_code?: number;
  self_target_blocked?: boolean;
  config_version?: string;
  chat_target_hash?: string;
  embedding_target_hash?: string;
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
  chat_api_protocol?: "openai" | "anthropic" | null;
  desired_chat_api_protocol?: "openai" | "anthropic" | null;
  routes?: Record<string, string>;
  warnings?: string[];
  last_reload?: ModelBridgeOperationStatus;
  last_sync?: ModelBridgeOperationStatus;
}

export interface ModelSettingsUpdate {
  chat_api_key?: string | null;
  clear_chat_api_key?: boolean;
  chat_api_protocol?: "openai" | "anthropic" | null;
  chat_base_url?: string | null;
  graph_base_url?: string | null;
  embedding_base_url?: string | null;
  chat_resolve_ip?: string | null;
  graph_resolve_ip?: string | null;
  embedding_resolve_ip?: string | null;
  embedding_model?: string | null;
  chat_model?: string | null;
  graph_model?: string | null;
  graph_api_key?: string | null;
  clear_graph_api_key?: boolean;
  graph_api_protocol?: "openai" | "anthropic" | null;
  embedding_api_protocol?: "openai" | null;
  embedding_dimensions?: number | null;
  embedding_batch_size?: number | null;
  worker_concurrency?: number | null;
  model_request_concurrency?: number | null;
  model_request_timeout_seconds?: number | null;
  chat_json_max_tokens?: number | null;
  agent_request_concurrency?: number | null;
  source_io_concurrency?: number | null;
  agent_request_queue_limit?: number | null;
  agent_request_queue_timeout_seconds?: number | null;
  agent_request_lease_ttl_seconds?: number | null;
  upload_max_bytes?: number | null;
  concept_i18n_enabled?: boolean | null;
  query_facet_bilingual_enabled?: boolean | null;
  query_facet_posterior_enabled?: boolean | null;
  query_facet_posterior_observation_budget?: number | null;
  query_facet_posterior_round_budget?: number | null;
  query_facet_posterior_convergence_epsilon?: number | null;
  fixed_chunk_size_tokens?: number | null;
  fixed_chunk_overlap_tokens?: number | null;
  context_package_token_budget?: number | null;
  model_bridge_enabled?: boolean | null;
  mid_concept_extraction_max_model_batches?: number | null;
  mid_concept_extraction_max_candidates_per_batch?: number | null;
  mid_concept_extraction_max_tokens_per_batch?: number | null;
  mid_concept_candidate_keep_threshold?: number | null;
  rq_kmeans_max_k?: number | null;
  rq_residual_tau?: number | null;
  edge_distance_protocol?: "edge_distance_log_calibrated_strength_v2";
  rq_membership_protocol?: "rq_fuzzy_softmax_gamma_product_v1";
  edge_projection_protocol?: "membership_q15_layer_type_calibrated_v3";
  edge_type_calibration_protocol?: "type_local_winsorized_minmax_v1";
  rq_membership_temperature?: number;
  rq_membership_top_m?: number;
  rq_membership_probability_threshold?: number;
  dense_knn_k_min?: number | null;
  dense_knn_k_max?: number | null;
  dense_reverse_b_min_base?: number | null;
  dense_reverse_b_max_base?: number | null;
  dense_reverse_b_min_doc?: number | null;
  dense_reverse_b_max_doc?: number | null;
  dense_reverse_b_min_lang?: number | null;
  dense_reverse_b_max_lang?: number | null;
  dense_min_cosine?: number | null;
  dense_strong_cosine?: number | null;
  cross_doc_out_quota_min?: number | null;
  cross_doc_out_quota_max?: number | null;
  cross_doc_min_cosine?: number | null;
  cross_language_out_quota_min?: number | null;
  cross_language_out_quota_max?: number | null;
  cross_language_min_cosine?: number | null;
  enable_auto_tpe?: boolean | null;
  tpe_trial_budget?: number | null;
  tpe_startup_random_trials?: number | null;
  tpe_good_quantile_gamma?: number | null;
  tpe_probe_query_budget?: number | null;
  tpe_trial_timeout_seconds?: number | null;
  tpe_candidate_pool_size?: number | null;
  operating_point_hard_gate_max_edge_density?: number | null;
  operating_point_hard_gate_max_isolated_ratio?: number | null;
  operating_point_hard_gate_max_hubness_ratio?: number | null;
  operating_point_hard_gate_min_structure_recovery_rate?: number | null;
  operating_point_hard_gate_max_candidate_latency_p95_ms?: number | null;
  retrieval_result_top_k_default?: number | null;
  agent_coarse_initial_budget?: number | null;
  agent_coarse_total_budget?: number | null;
  agent_coarse_top_k?: number | null;
  agent_mid_per_coarse_budget?: number | null;
  agent_coarse_drilldown_mid_initial_budget?: number | null;
  agent_mid_initial_budget?: number | null;
  agent_mid_top_k?: number | null;
  agent_chunk_per_mid_budget?: number | null;
  agent_chunk_initial_budget?: number | null;
  agent_chunk_top_k?: number | null;
  candidate_pool_dedupe_budget?: number | null;
  agent_max_depth_per_layer?: number | null;
  agent_max_labels_per_node?: number | null;
  agent_max_edge_reuse?: number | null;
  agent_max_cycle_reward_per_path?: number | null;
  agent_cycle_reward_distance_threshold?: number | null;
  agent_path_distance_green_threshold?: number;
  agent_path_distance_gray_threshold?: number;
  agent_path_distance_hard_threshold?: number;
  gray_zone_rule_protocol?: "deterministic_support_progress_v1";
  gray_zone_observation_cadence?: number;
  traversal_observation_budget?: number;
  agent_structure_restore_per_chunk_budget?: number | null;
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
  chunk_count?: number;
  relation_edge_count?: number;
  rq_prefix_count?: number;
  mid_concept_count?: number;
  coarse_concept_count?: number;
  context_graph_hash?: string | null;
  context_graph_state_id?: string | null;
  context_graph_phase?: string | null;
  context_graph_metrics?: Record<string, unknown>;
  translation_phase?: string | null;
  translation_items?: number;
  translation_enabled?: boolean;
  translation_status?: string | null;
  translated_count?: number;
  fallback_count?: number;
  concept_i18n_translated_count?: number;
  concept_i18n_fallback_count?: number;
  edge_i18n_translated_count?: number;
  edge_i18n_fallback_count?: number;
  retry_count?: number;
  max_retries?: number;
  job_id?: string | null;
  trial_id?: string | null;
  trial_index?: number;
  theta_hash?: string | null;
  objective_score?: number | null;
  hard_gate?: Record<string, unknown>;
  failure_code?: string | null;
  probe_set_hash?: string | null;
  reasons?: string[];
}

export interface RuntimeIssue {
  code: string;
  title: string;
  message: string;
  fix_commands: string[];
}

export interface EnvSyncStatus {
  synced: boolean;
  settings_file_present?: boolean;
  settings_file_schema_synced?: boolean;
  missing_keys: string[];
  extra_keys: string[];
  deprecated_keys: string[];
  bom_keys: string[];
}

export interface InfrastructureStatus {
  postgres: boolean;
  qdrant: boolean;
  redis: boolean;
  model_bridge?: boolean | null;
}

export interface RuntimeCheckResponse {
  env_sync: EnvSyncStatus;
  infrastructure: InfrastructureStatus;
  model_bridge_status?: ModelBridgeStatus;
  blocking_issues: RuntimeIssue[];
  warnings: RuntimeIssue[];
}

export interface StructuredApiErrorBody {
  protocol_version?: string;
  code: string;
  title: string;
  message: string;
  reason?: string;
  action?: string;
  issues: RuntimeIssue[];
  fix_commands: string[];
  retryable?: boolean;
  retry_after_seconds?: number;
  retry_after_rebuild?: boolean;
  rebuild_required?: boolean;
  diagnostics?: Record<string, unknown>;
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
  | "rq_prefix"
  | "mid_concept"
  | "coarse_concept"
  | "context_package"
  | (string & {});

export type GraphNodeContractKind =
  | "structure_node"
  | "chunk_node"
  | "rq_prefix_node"
  | "mid_concept_node"
  | "coarse_concept_node";

export interface GraphDistribution {
  count: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  population_std?: number | null;
}

export interface GraphNodeMetadata {
  rq_path: number[];
  residual_norm?: number | null;
  rq_prefix_key?: string | null;
  rq_level?: number | null;
  rq_path_prefix: number[];
  representative_chunk_ids: string[];
  support_chunk_ids: string[];
  bridge_chunk_ids: string[];
  residual_norm_mean?: number | null;
  residual_norm_max?: number | null;
}

export interface GraphNodeWeightDiagnostics {
  protocol_version?: string | null;
  normalization?: string | null;
  normalization_scope?: "mid_concept_state" | "coarse_concept_state" | null;
  normalization_scope_hash?: string | null;
  normalization_pending?: boolean | null;
  layer_local_only?: boolean | null;
  cross_layer_comparison_allowed: false;
  query_relevance: false;
  model_call_count: 0;
  component_weights: Record<string, number>;
  components: Record<string, number>;
  formula?: string | null;
  raw_node_weight?: number | null;
  raw_node_weight_distribution?: GraphDistribution | null;
  max_raw_node_weight?: number | null;
  node_weight?: number | null;
  support_chunk_count?: number | null;
  support_chunk_edge_count?: number | null;
  included_mid_concept_count?: number | null;
  membership_mass?: number | null;
  membership_role_mass_distribution: Record<string, number>;
  membership_entropy_distribution?: GraphDistribution | null;
  internal_edge_count?: number | null;
  cross_edge_count?: number | null;
  structure_mapping_coverage?: number | null;
  summary_confidence_source?: string | null;
  summary_grounded?: boolean | null;
  summary_grounded_rate?: number | null;
  card_hash?: string | null;
}

interface GraphNodeBase {
  contract_kind: GraphNodeContractKind;
  id: string;
  label: string;
  type: string;
  name?: string | null;
  category?: GraphNodeCategory | string | null;
  layer?: string | null;
  value?: number | null;
  score?: number | null;
  importance_score?: number | null;
  raw_node_weight?: number | null;
  node_weight?: number | null;
  node_weight_normalization_scope?: string | null;
  node_weight_diagnostics?: GraphNodeWeightDiagnostics | null;
  confidence?: number | null;
  support_count?: number | null;
  support_chunk_ids: string[];
  support_active_chunk_ids: string[];
  support_rq_prefix_ids: string[];
  representative_chunk_ids: string[];
  source_path?: string | null;
  document_id?: string | null;
  document_version_id?: string | null;
  summary?: string | null;
  snippet?: string | null;
  text?: string | null;
  page_number?: number | null;
  page_range?: [number | null, number | null] | Array<number | null> | null;
  section_path: string[];
  metadata: GraphNodeMetadata;
}

interface GraphNodeCoarseRoleIds {
  included_mid_concept_ids: string[];
  boundary_mid_concept_ids: string[];
  bridge_mid_concept_ids: string[];
  outlier_mid_concept_ids: string[];
  low_confidence_mid_concept_ids: string[];
  all_mid_concept_ids: string[];
}

export type GraphNode =
  | (GraphNodeBase & { contract_kind: "structure_node" })
  | (GraphNodeBase & {
        contract_kind: "chunk_node";
        type: "chunk";
      })
  | (GraphNodeBase & {
        contract_kind: "rq_prefix_node";
        type: "rq_prefix";
      })
  | (GraphNodeBase & {
      contract_kind: "mid_concept_node";
      type: "mid_concept";
      raw_node_weight: number;
      node_weight: number;
      node_weight_normalization_scope: string;
      node_weight_diagnostics: GraphNodeWeightDiagnostics;
    })
  | (GraphNodeBase &
      GraphNodeCoarseRoleIds & {
      contract_kind: "coarse_concept_node";
      type: "coarse_concept";
      raw_node_weight: number;
      node_weight: number;
      node_weight_normalization_scope: string;
      node_weight_diagnostics: GraphNodeWeightDiagnostics;
    });

export interface ProjectionCalibrationParams {
  lower_quantile: number;
  upper_quantile: number;
  min_span: number;
  strength_floor: number;
}

export interface ProjectionNormalizationAudit {
  normalization: "layer_edge_type_winsorized_minmax_v1";
  protocol_version: "layer_edge_type_winsorized_minmax_v1";
  edge_projection_protocol_version: "membership_q15_layer_type_calibrated_v3";
  edge_projection_protocol_hash: string;
  layer: "mid" | "coarse";
  edge_type: string;
  scope: "layer_plus_edge_type";
  params: ProjectionCalibrationParams;
  sample_count: number;
  lower_quantile_value: number;
  upper_quantile_value: number;
  quantile_span: number;
  fallback?: string | null;
  calibration_applied: boolean;
  raw_strength_distribution: GraphDistribution;
  calibrated_strength_distribution: GraphDistribution;
  calibrated_distance_distribution: GraphDistribution;
  cross_type_raw_comparison_allowed: false;
  model_call_count: 0;
  stats_hash: string;
  support_edge_count: number;
  support_mid_edge_count: number;
  support_chunk_edge_count: number;
  support_membership_mass: number;
}

export interface GraphProjectionBottomEdgeTypeMass {
  dense_semantic: number;
  dense_cross_document_bridge: number;
  dense_cross_language_bridge: number;
}

export interface GraphProjectionSupportEdgeTypeCounts {
  dense_semantic: number;
  dense_cross_document_bridge: number;
  dense_cross_language_bridge: number;
}

export interface GraphProjectionRawStrengthSummary {
  aggregation_protocol_version: "membership_weighted_bottom_support_q15_log_mass_v1";
  q15_bottom_distance: number;
  support_membership_mass: number;
  support_mid_edge_count: number;
  support_chunk_edge_count: number;
  bottom_distance_distribution: GraphDistribution;
  membership_product_distribution: GraphDistribution;
  dominant_bottom_edge_type:
    | "dense_semantic"
    | "dense_cross_document_bridge"
    | "dense_cross_language_bridge";
  bottom_edge_type_membership_mass: GraphProjectionBottomEdgeTypeMass;
  contribution_facts_hash: string;
  edge_distance_protocol: "edge_distance_log_calibrated_strength_v2";
}

export interface GraphProjectionSupportContribution {
  bottom_chunk_edge_id: string;
  source_chunk_id: string;
  target_chunk_id: string;
  bottom_edge_type:
    | "dense_semantic"
    | "dense_cross_document_bridge"
    | "dense_cross_language_bridge";
  bottom_distance: number;
  source_membership_score: number;
  target_membership_score: number;
  membership_product: number;
  orientation: "source_scope_to_target_scope";
  assignment_protocol_version: "scope_key_chunk_business_assignment_v1";
  bottom_edge_fact_hash: string;
}

export interface GraphProjectionGrayPredicates {
  protocol_version: "projected_gray_predicates_support_rollup_v1";
  protocol_hash: string;
  semantic_uncertain: boolean;
  crossing_rq_boundary: boolean;
  support_edge_count: number;
  semantic_uncertain_support_count: number;
  rq_boundary_support_count: number;
  semantic_uncertain_support_edge_ids: string[];
  rq_boundary_support_edge_ids: string[];
  semantic_uncertainty_rollup: "all_bottom_support_edges_uncertain";
  rq_boundary_rollup: "any_bottom_support_edge_crosses_rq_leaf_path";
  model_call_count: 0;
}

export interface GraphProjectionEdgeDiagnostics {
  edge_projection_protocol: "membership_q15_layer_type_calibrated_v3";
  edge_projection_protocol_hash: string;
  aggregation_protocol_version: "membership_weighted_bottom_support_q15_log_mass_v1";
  calibration_protocol_version: "layer_edge_type_winsorized_minmax_v1";
  source_algorithm: "membership_weighted_bottom_edge_projection";
  support_rq_prefix_ids: string[];
  support_mid_edge_count: number;
  support_chunk_edge_count: number;
  support_contribution_count: number;
  support_membership_mass: number;
  support_contributions: GraphProjectionSupportContribution[];
  support_contributions_complete: boolean;
  contribution_facts_hash: string;
  dominant_bottom_edge_type:
    | "dense_semantic"
    | "dense_cross_document_bridge"
    | "dense_cross_language_bridge";
  support_edge_types: GraphProjectionSupportEdgeTypeCounts;
  semantic_uncertain: boolean;
  crossing_rq_boundary: boolean;
  gray_predicates: GraphProjectionGrayPredicates;
  gray_zone_semantics_changed: false;
  model_call_count: 0;
  edge_i18n?: GraphEdgeI18n | null;
}

export interface GraphI18nTextMap {
  zh: string;
  en: string;
}

export interface GraphI18nSearchTermsMap {
  zh: string[];
  en: string[];
}

interface GraphEdgeI18nBase {
  id?: string | null;
  layer: "mid" | "coarse";
  protocol_version: "concept_i18n_bilingual_v1";
  relation_label_i18n: GraphI18nTextMap;
  explanation_i18n: GraphI18nTextMap;
  summary_i18n: GraphI18nTextMap;
  search_terms_i18n: GraphI18nSearchTermsMap;
}

export type GraphEdgeI18n = GraphEdgeI18nBase &
  (
    | { status: "ok"; fallback_source?: null }
    | {
        status: "original_text_fallback";
        fallback_source: "original_text_fallback";
      }
  );

export interface GraphEdgeMetadata {
  source_algorithm?: string | null;
  protocol_version?: string | null;
  graph_state_hash?: string | null;
  is_cross_document?: boolean | null;
  is_cross_language?: boolean | null;
  semantic_uncertain?: boolean | null;
  crossing_rq_boundary?: boolean | null;
  candidate_channels: string[];
  diagnostic_only?: boolean | null;
  active_relation_edge?: boolean | null;
  membership_role?: string | null;
  membership_entropy?: number | null;
  membership_rank?: number | null;
  top_alternative_prefix_ids: string[];
  rq_path: number[];
  residual_norm?: number | null;
  diagnostic_strength?: number | null;
  support_membership_mass?: number | null;
  support_chunk_ids_sample: string[];
  support_chunk_edge_ids_sample: string[];
  support_chunk_edge_ids?: string[];
  support_chunk_edge_count?: number | null;
  support_chunk_edge_ids_hash?: string | null;
  protocol_hash?: string | null;
  diagnostic_hash?: string | null;
  model_call_count?: 0 | null;
}

export interface GraphRQDiagnosticEdgeMetadata extends GraphEdgeMetadata {
  diagnostic_only: true;
  active_relation_edge: false;
  model_call_count: 0;
  support_chunk_edge_ids: string[];
  support_chunk_edge_ids_sample: string[];
  support_chunk_edge_count: number;
  support_chunk_edge_ids_hash: string;
  protocol_hash: string;
  diagnostic_hash: string;
}

interface GraphEdgeBase {
  contract_kind:
    | "structure_edge"
    | "chunk_relation_edge"
    | "rq_membership_edge"
    | "rq_diagnostic_edge"
    | "concept_projection_edge";
  id?: string | null;
  source: string;
  target: string;
  label?: string | null;
  type: string;
  category?: string | null;
  confidence?: number | null;
  weight?: number | null;
  distance?: number | null;
  raw_strength?: number | null;
  projected_distance_raw?: number | null;
  projected_strength_raw?: number | null;
  raw_strength_summary?: GraphProjectionRawStrengthSummary | null;
  edge_projection_protocol_hash?: string | null;
  projection_normalization?: ProjectionNormalizationAudit | null;
  diagnostics?: GraphProjectionEdgeDiagnostics | null;
  source_algorithm?: string | null;
  protocol_version?: string | null;
  state_hash?: string | null;
  score?: number | null;
  support_count?: number | null;
  support_chunk_ids: string[];
  support_chunk_edge_ids: string[];
  support_mid_edge_ids: string[];
  support_mid_concept_ids: string[];
  support_rq_prefix_ids: string[];
  relation_source?: string | null;
  is_bridge?: boolean | null;
  is_inferred?: boolean | null;
  metadata: GraphEdgeMetadata;
}

export type GraphEdge =
  | (GraphEdgeBase & { contract_kind: "structure_edge" })
  | (GraphEdgeBase & { contract_kind: "chunk_relation_edge" })
  | (GraphEdgeBase & { contract_kind: "rq_membership_edge" })
  | (GraphEdgeBase & {
      contract_kind: "rq_diagnostic_edge";
      type: "rq_prefix_pair_diagnostic";
      metadata: GraphRQDiagnosticEdgeMetadata;
    })
  | (GraphEdgeBase & {
      contract_kind: "concept_projection_edge";
      projected_distance_raw: number;
      projected_strength_raw: number;
      raw_strength_summary: GraphProjectionRawStrengthSummary;
      edge_projection_protocol_hash: string;
      projection_normalization: ProjectionNormalizationAudit;
      diagnostics: GraphProjectionEdgeDiagnostics;
      source_algorithm: string;
      protocol_version: "membership_q15_layer_type_calibrated_v3";
      state_hash: string;
    });

export interface GraphLayerCounts { nodes: number; edges: number }
export interface GraphSampleCountCard { sampled: number; full: number }
export interface GraphCounts {
  chunks: number; active_chunks: number; structure_nodes: number; structure_edges: number;
  structure_mappings: number; chunk_relation_edges: number; rq_prefixes: number;
  rq_prefix_memberships: number; rq_prefix_pair_diagnostics: number; rq_relation_edges: number;
  mid_concepts: number; mid_concept_edges: number; mid_concept_memberships: number;
  coarse_concepts: number; coarse_concept_edges: number; coarse_concept_memberships: number;
}
export interface GraphFreshnessLayerRow {
  layer: string; state_hash?: string | null; is_stale: boolean; stale_reasons: string[];
  checked_at?: string | null; diagnostics: Record<string, unknown>;
}
export interface GraphFreshness {
  protocol_version?: string | null; admission_protocol_version?: string | null;
  is_stale: boolean; is_admissible?: boolean | null; stale_reasons: string[]; admission_reasons?: string[];
  current_chunk_scope_hash?: string | null;
  current_contextual_index_hash?: string | null; stored_contextual_index_hash?: string | null;
  current_contextual_index_business_hash?: string | null; stored_contextual_index_business_hash?: string | null;
  current_chunk_business_scope_hash?: string | null; stored_chunk_business_scope_hash?: string | null;
  canonical_state_hash_protocol_version?: string | null; canonical_state_validation_reasons: string[];
  checked_at?: string | null; layer_rows?: GraphFreshnessLayerRow[];
  model_call_count?: 0; gray_zone_rule_inputs_modified?: false;
  local_hint_protocol_version: string; context_graph_state_id?: string | null; context_graph_hash?: string | null;
}
export interface GraphHashes {
  chunk_scope_hash?: string | null; chunk_business_scope_hash?: string | null;
  contextual_index_hash?: string | null; contextual_index_business_hash?: string | null;
  local_hint_protocol_version: string; structure_graph_hash?: string | null;
  chunk_relation_graph_hash?: string | null; rq_membership_hash?: string | null;
  rq_prefix_pair_aggregate_hash?: string | null; rq_prefix_pair_diagnostics_hash?: string | null;
  mid_concept_hash?: string | null; coarse_concept_hash?: string | null; context_graph_hash?: string | null;
  runtime_settings_hash: string; agent_operating_envelope_hash: string;
  policy_state_hash?: string | null; prompt_protocol_hash?: string | null;
}
export interface GraphGrounding { mid_grounded_rate: number; mid_total: number; coarse_grounded_rate: number; coarse_total: number }
export interface GraphRetrievalContribution {
  trace_count: number; has_observations: boolean; frontier_pops: number; dominance_pruned_count: number;
  expanded_edge_contribution: Record<string, number>; convergence_reasons: Record<string, number>; scores_json_primary: false;
}
export interface GraphEdgeTypeDistribution {
  raw_strength?: GraphDistribution | null;
  calibrated_strength?: GraphDistribution | null;
  distance?: GraphDistribution | null;
  calibration_stats_hashes: string[];
  calibration_stats_hash_consistent?: boolean | null;
}
export interface GraphEdgeCalibrationParams {
  lower_quantile: number;
  upper_quantile: number;
  min_span: number;
  strength_floor: number;
}
export interface GraphEdgeCalibrationTypeStats {
  edge_type: string;
  protocol_version: string;
  protocol_hash: string;
  calibration_params_hash: string;
  edge_type_calibration_config_hash: string;
  stats_hash: string;
  params: GraphEdgeCalibrationParams;
  sample_count: number;
  lower_quantile_value?: number | null;
  upper_quantile_value?: number | null;
  effective_lower_bound?: number | null;
  effective_upper_bound?: number | null;
  quantile_span?: number | null;
  fallback?: string | null;
  calibration_applied: boolean;
  monotonic_violation_count: number;
  raw_strength_distribution: GraphDistribution;
  calibrated_strength_distribution: GraphDistribution;
  distance_distribution: GraphDistribution;
  cross_type_raw_comparison_allowed: false;
}
export interface GraphEdgeCalibration {
  protocol_version: string;
  protocol_hash: string;
  stats_by_edge_type: Record<string, GraphEdgeCalibrationTypeStats>;
  all_stats_hashes_consistent: boolean;
  cross_type_raw_comparison_allowed: false;
}
export interface GraphEdgeDistanceDiagnostics {
  applicable: boolean;
  reason?: string | null;
  protocol_version?: string | null;
  protocol_hash?: string | null;
  protocol_hashes: string[];
  distribution?: GraphDistribution | null;
  by_edge_type: Record<string, GraphEdgeTypeDistribution>;
  calibration?: GraphEdgeCalibration | null;
}
export interface GraphProjectionGroupDiagnostics {
  protocol_hashes: string[]; protocol_hash_edge_count: number; protocol_hash_coverage: number;
  protocol_hash_consistent: boolean; source_algorithm_coverage: number; protocol_version_coverage: number;
  state_hash_coverage: number; full_edge_count: number; supported_edge_count: number; support_coverage: number;
  raw_projected_distance_coverage: number; calibrated_projected_distance_coverage: number;
  raw_projected_strength_coverage: number; raw_projected_distance_distribution: GraphDistribution;
  calibrated_projected_distance_distribution: GraphDistribution; raw_projected_strength_distribution: GraphDistribution;
  support_membership_mass_distribution: GraphDistribution; support_contribution_count_distribution: GraphDistribution;
  normalization_protocol_counts: Record<string, number>; normalization_stats_hashes: string[];
}
export interface GraphProjectedGrayPredicates {
  protocol_version: string; protocol_hash: string; coverage: number; missing_edge_count: number;
  semantic_uncertain_edge_count: number; crossing_rq_boundary_edge_count: number; model_call_count: 0;
}
export interface GraphProjectionDiagnostics extends GraphProjectionGroupDiagnostics {
  applicable: true; protocol_version: string; by_edge_type: Record<string, GraphProjectionGroupDiagnostics>;
  gray_predicates: GraphProjectedGrayPredicates; graph_total_edge_count: number; non_projection_edge_count: number;
}
export interface GraphProjectionNotApplicable { applicable: false; reason: string }
export interface GraphResponse {
  contract_version: "four_layer_graph_public_v1";
  knowledge_base_id: string;
  graph_type: Exclude<GraphType, "context-graph">;
  schema_version: "context_graph_v1";
  view: "overview";
  nodes: GraphNode[];
  edges: GraphEdge[];
  counts: GraphCounts;
  full_counts: GraphLayerCounts;
  sampled_counts: GraphLayerCounts;
  node_counts: GraphSampleCountCard;
  edge_counts: GraphSampleCountCard;
  freshness: GraphFreshness;
  hash?: string | null;
  hashes: GraphHashes;
  stale_reason?: string | null;
  grounding: GraphGrounding;
  retrieval_contribution: GraphRetrievalContribution;
  edge_distance_diagnostics: GraphEdgeDistanceDiagnostics;
  projection_diagnostics: GraphProjectionDiagnostics | GraphProjectionNotApplicable;
  diagnostics: { layer_full_counts: GraphLayerCounts; sampled_counts: GraphLayerCounts; layer_hash_key: string; layer_hash?: string | null };
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
  knowledge_base_id: string;
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
  coverage_by_language?: Record<string, number>;
  errors?: BatchError[];
  graph_stats?: Record<string, unknown>;
  stats?: Record<string, unknown>;
  phase?: string | null;
  current_phase?: string | null;
  cancel_requested?: boolean;
  last_error?: string | null;
  parse_committed?: boolean;
  batch_recovery_id?: string | null;
  batch_recovery_protocol_version?: string | null;
  v_before_batch?: number | null;
  parse_commit_boundary?: string | null;
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
  rq_prefix_count?: number;
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
  language?: string | null;
  language_source?: "explicit_metadata" | "deterministic_detection" | "unknown" | null;
  language_confidence?: number | null;
  language_detection_protocol_version?: string | null;
  language_detection_hash?: string | null;
  language_identity_consistent?: boolean;
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

export interface RetrievalEntryTopologyPrior {
  protocol_version: string;
  protocol_hash: string;
  centrality: number;
  betweenness: number;
  betweenness_mode: string;
  k_core: number;
  k_core_normalized: number;
  pagerank_or_closeness: number;
  incident_edge_count: number;
  topology_admission_eligible: boolean;
  bridge_score: number;
  boundary_score: number;
  bridge_role: boolean;
  boundary_role: boolean;
}

export interface RetrievalEntryDenseSupportFact {
  chunk_id: string;
  dense_score: number;
  active_contextual_vector_business_fact_hash: string;
}

export interface RetrievalEntryReplayProof {
  protocol_version: string;
  node_id: string;
  layer: "coarse" | "mid";
  support_chunk_ids: string[];
  support_count: number;
  semantic_dense_support_facts: RetrievalEntryDenseSupportFact[];
  semantic_support_chunk_ids: string[];
  semantic_support_match_count: number;
  semantic_score: number;
  semantic_candidate: boolean;
  semantic_input_hash: string;
  query_facet_packet_hash: string;
  active_topology_state_identity_hash: string;
  topology_state_hash: string;
  topology_node_business_fact_hash: string;
  topology_prior_hash: string;
  dense_replay_input_hash: string;
  neutral_start_cost_protocol_version: string;
  proof_hash: string;
}

export interface RetrievalEntryCandidateCard {
  protocol_version: string;
  protocol_hash: string;
  node_id: string;
  layer: "coarse" | "mid";
  semantic_aggregation_protocol_version: string;
  intent_strategy: string;
  selection_rank?: number | null;
  selected: boolean;
  selection_reasons: string[];
  label: string;
  definition_or_summary: string;
  support_count: number;
  matched_query_facets: string[];
  semantic_score: number;
  semantic_candidate: boolean;
  semantic_support_chunk_ids: string[];
  semantic_support_match_count: number;
  topology: RetrievalEntryTopologyPrior;
  replay_proof: RetrievalEntryReplayProof;
  entry_strength?: number | null;
  entry_strength_source: string;
  neutral_start_cost_protocol_version: string;
  neutral_start_cost_is_query_relevance: false;
  topology_used_for_admission_or_tie_break: boolean;
  topology_used_as_path_distance: false;
  node_weight_used_as_query_relevance: false;
  lexical_overlap_used_as_query_relevance: false;
  gray_zone_rule_inputs_modified: false;
  model_call_count: 0;
  candidate_card_hash: string;
}

export interface RetrievalEntryNode {
  layer: string;
  node_id: string;
  entry_strength?: number | null;
  roles: string[];
  rq_prefix_id?: string | null;
  metadata: {
    label?: string | null;
    node_type?: string | null;
    rq_path_prefix: number[];
    representative_terms: string[];
    candidate_card?: RetrievalEntryCandidateCard | null;
  };
}

export interface RetrievalSupportRefs {
  edge_id?: string | null;
  edge_ids: string[];
  edge_type?: string | null;
  edge_types: string[];
  support_ids: string[];
  support_chunk_ids: string[];
  support_chunk_edge_ids: string[];
  support_relation_edge_ids: string[];
  support_rq_prefix_ids: string[];
  support_rq_prefix_node_ids: string[];
  support_mid_edge_ids: string[];
  support_mid_concept_ids: string[];
  entry_strength?: number | null;
  entry_distance?: number | null;
  entry_strengths: Record<string, number>;
  raw_entry_strengths: Record<string, number>;
  mid_concept_ids: string[];
  rq_prefix_ids: string[];
  chunk_candidate_source?: string | null;
  chunk_candidate_sources: string[];
}

export interface RetrievalEntryParentRef {
  parent_layer: string;
  parent_node_id: string;
  edge_type?: string | null;
  support_refs: RetrievalSupportRefs;
}

export interface RetrievalStateSignature {
  layer?: string | null;
  node_id?: string | null;
  covered_facets: string[];
  evidence_roles: string[];
  depth_bucket?: number | null;
  path_edge_type_multiset: Record<string, number>;
}

export interface CycleDistanceRewardAudit {
  protocol_version: "bounded_cycle_distance_reward_replay_v1";
  cycle_edges: string[];
  cycle_distance: number;
  edge_strength: number;
  support_delta: 0 | 1;
  reward_before_cap: number;
  reward_after_cap: number;
  cap_reason: "max_cycle_reward_per_path_exhausted" | "cycle_distance_above_threshold" | "within_cap" | "max_cycle_reward_per_path";
}

export interface RetrievalTraversalState {
  layer?: string | null;
  parent_layer?: string | null;
  parent_node_id?: string | null;
  node_id?: string | null;
  path: string[];
  path_edge_ids: string[];
  path_edge_distances: number[];
  path_edge_strengths: number[];
  path_edge_types: string[];
  distance_so_far?: number | null;
  reward_so_far?: number | null;
  distance_zone?: string | null;
  covered_facets: string[];
  evidence_roles: string[];
  depth?: number | null;
  root_node_id?: string | null;
  visit_counts: Record<string, number>;
  edge_reuse_counts: Record<string, number>;
  support_refs: RetrievalSupportRefs;
  entry_parent_refs: RetrievalEntryParentRef[];
  entry_support_refs: RetrievalSupportRefs;
  state_signature: RetrievalStateSignature;
  gray_zone_decision?: string | null;
  gray_zone_terminal_action?: string | null;
  cycle_distance_rewards: CycleDistanceRewardAudit[];
}

export interface RetrievalFrontierSnapshot {
  layer?: string | null;
  popped: RetrievalTraversalState;
  queue_size_after_pop: number;
  key: Array<number | string>;
}

export interface RetrievalStageQueue {
  entry_ids: string[];
  forced_entry_ids: string[];
  forced_downstream_entry_ids: string[];
  selected_ids: string[];
  accepted_ids: string[];
  top_k?: number | null;
  initial_top_k?: number | null;
  frontier_pop_count?: number | null;
  entry_mode?: string | null;
  skipped_by_granularity?: string | null;
  reason?: string | null;
}

export interface CandidatePoolDedupeAudit {
  protocol_version: string;
  scope: string;
  limit: number;
  attempt_count: number;
  unique_admitted_count: number;
  duplicate_count: number;
  rejected_new_count: number;
  budget_hit: boolean;
  hard_interrupt_count: number;
  rejected_candidate_id_samples: string[];
  observation_compacted: boolean;
  stop_reason: string;
}

export interface PerParentBudgetStatus {
  budget: number;
  candidate_count: number;
  selected_count: number;
  merged_selected_count?: number | null;
  stop_reason: string;
}

export interface RetrievalRQRouteContribution {
  mid_concept_id: string;
  mid_entry_strength: number;
  mid_membership_score: number;
  route_fallback_score: number;
}

export interface RetrievalRQStageSeedCard {
  protocol_version: "query_rq_fuzzy_membership_chunk_seed_v2";
  protocol_hash: "144d218ea37a70f2aa85730624c5cec0807e20542078adc1574f832cbff017d8";
  rq_prefix_id: string;
  rq_path: number[];
  rq_level: number;
  query_rq_path: number[];
  rq_lcp_depth: number;
  residual_distance: number | null;
  query_prefix_membership_score: number | null;
  requested_query_relevance: number | null;
  route_fallback_score: number;
  parent_mid_contributions: RetrievalRQRouteContribution[];
  score_source: "typed_action_forced_override" | "query_rq_relevance" | "selected_mid_route_fallback";
  effective_score: number;
  forced_override: boolean;
  relation_state_hash: string | null;
  is_evidence: false;
  node_weight_used_as_query_relevance: false;
  hard_path_lcp_used_as_score: false;
  gray_zone_decision_authority: false;
  model_call_count: 0;
  input_hash: string;
  card_hash: string;
}

export interface RetrievalRQChunkSeedCard {
  protocol_version: "query_rq_fuzzy_membership_chunk_seed_v2";
  protocol_hash: "144d218ea37a70f2aa85730624c5cec0807e20542078adc1574f832cbff017d8";
  parent_mid_concept_id: string;
  chunk_id: string;
  rq_l3_prefix_id: string | null;
  query_rq_path: number[];
  candidate_rq_path: number[];
  rq_lcp_depth: number;
  residual_distance: number | null;
  query_prefix_score: number;
  chunk_membership_score: number;
  fuzzy_membership_overlap_score: number;
  rq_score: number;
  rq_relevance_component: number;
  rq_drift_penalty: number | null;
  membership_role: "primary_member" | "fuzzy_member" | "boundary_member" | "outlier_member" | "noise_candidate" | "bridge_member" | "low_confidence_member" | "mid_support_fallback";
  membership_rank: number;
  membership_entropy: number | null;
  bridge_or_boundary_role: boolean;
  support_edge_ids: string[];
  mid_entry_component: number;
  dense_component: number;
  component_weights: Record<string, number>;
  effective_score: number;
  score_source: "query_rq_fuzzy_membership" | "mid_support_without_rq_membership";
  membership_role_tie_break_rank: number;
  is_evidence: false;
  node_weight_used_as_query_relevance: false;
  hard_path_lcp_used_as_score: false;
  gray_zone_decision_authority: false;
  model_call_count: 0;
  input_hash: string;
  card_hash: string;
}

export interface RetrievalCandidatePool {
  parent_layer?: string | null;
  parent_node_id?: string | null;
  candidate_ids: string[];
  candidate_scores: Record<string, number>;
  rq_seed_cards: Record<string, RetrievalRQStageSeedCard>;
  rq_chunk_seed_cards: Record<string, RetrievalRQChunkSeedCard[]>;
  chunk_facet_priority_cards: Record<string, RetrievalChunkFacetPriorityCard>;
  query_facet_posterior_snapshot?: {
    protocol_version: "query_facet_posterior_calibration_v1";
    protocol_hash: string;
    posterior: Record<string, number>;
    alias_posterior: Record<string, Record<string, number>>;
    round_count: number;
    observation_count: number;
    snapshot_hash: string;
  } | null;
  covered_posterior_mass_by_candidate?: Record<string, number>;
  facet_priority_protocol_version?: string | null;
  facet_priority_protocol_hash?: string | null;
  ranking_protocol_version: string | null;
  ranking_protocol_hash: string | null;
  forced_candidate_ids: string[];
  selected_ids: string[];
  ranked_selected_ids?: string[];
  candidate_count?: number | null;
  top_k?: number | null;
  stop_reason?: string | null;
  source?: string | null;
  coarse_skipped_reason?: string | null;
  per_parent_budget_status?: PerParentBudgetStatus | null;
  candidate_dedupe_budget_audit?: CandidatePoolDedupeAudit | null;
}

export interface RetrievalChunkFacetPriorityCard {
  protocol_version: "validated_query_facet_posterior_chunk_priority_v2";
  protocol_hash: string;
  facet_match_protocol_version: "validated_query_facet_ordered_window_v1";
  facet_match_protocol_hash: string;
  chunk_id: string;
  query_facet_packet_hash: string;
  required_facets: string[];
  matched_required_facets: string[];
  uncovered_required_facets: string[];
  matched_required_facet_count: number;
  uncovered_required_facet_count: number;
  priority_prefix: [number];
  lexical_overlap_used_as_numeric_relevance: false;
  is_evidence: false;
  citation_authority: false;
  gray_zone_decision_authority: false;
  model_call_count: 0;
  card_hash: string;
}

export interface CandidatePoolDedupeSummary {
  protocol_version: string;
  limit_per_pool: number;
  pool_count: number;
  budget_hit_pool_count: number;
  hard_interrupt_count: number;
  unique_admitted_count: number;
  duplicate_count: number;
  audits: CandidatePoolDedupeAudit[];
}

export interface RetrievalCandidatePools {
  mid_by_coarse: RetrievalCandidatePool[];
  chunk_by_mid: RetrievalCandidatePool[];
  mid_direct_entries?: RetrievalCandidatePool | null;
  mid_initial_entries?: RetrievalCandidatePool | null;
  rq_membership_entries?: RetrievalCandidatePool | null;
  chunk_initial_entries?: RetrievalCandidatePool | null;
  candidate_dedupe_budget?: CandidatePoolDedupeSummary | null;
}

export interface RetrievalTopKSelection {
  top_k?: number | null;
  candidate_count: number;
  selected_ids: string[];
  forced_selected_ids: string[];
  ranking_protocol_version?: string | null;
  candidate_rank_facts?: RetrievalCandidateRankFact[];
  carry_forward_supported_chunk_ids?: string[];
  global_top_k_increased?: boolean | null;
  stop_reason?: string | null;
  entry_mode?: string | null;
}

export interface RetrievalCandidateRankFact {
  candidate_id: string;
  rank_key: Array<number | string>;
  path_identity: string;
  repair_evidence_retention_protocol_version?:
    | "repair_supported_evidence_carry_forward_v1"
    | null;
  source_context_package_id?: string | null;
  source_retrieval_trace_id?: string | null;
  repair_directive_hash?: string | null;
}

export interface RetrievalPathLabel {
  layer?: string | null;
  node_id?: string | null;
  chunk_id?: string | null;
  path: string[];
  path_edge_ids: string[];
  path_edge_types: string[];
  expanded_edge_ids: string[];
  covered_facets: string[];
  evidence_roles: string[];
  distance_so_far?: number | null;
  reward_so_far?: number | null;
  root_node_id?: string | null;
  parent_layer?: string | null;
  parent_node_id?: string | null;
  stop_reason?: string | null;
  support_refs: RetrievalSupportRefs;
  entry_parent_refs: RetrievalEntryParentRef[];
  path_edge_type_multiset: Record<string, number>;
  edge_reuse_counts: Record<string, number>;
  repair_evidence_retention_protocol_version?:
    | "repair_supported_evidence_carry_forward_v1"
    | null;
  source_context_package_id?: string | null;
  source_retrieval_trace_id?: string | null;
  repair_directive_hash?: string | null;
}

export interface RetrievalPathContribution {
  contract_version: "multi_path_contribution_v2";
  contribution_id: string;
  layer: "coarse" | "mid" | "chunk";
  node_id: string;
  parent_layer?: string | null;
  parent_node_id?: string | null;
  origin_parent_layer?: string | null;
  origin_parent_node_id?: string | null;
  root_node_id: string;
  path: string[];
  path_edge_ids: string[];
  path_edge_types: string[];
  covered_facets: string[];
  evidence_roles: string[];
  support_refs: RetrievalSupportRefs;
  support_chunk_ids: string[];
  distance_so_far: number;
  reward_so_far: number;
}

export interface RetrievalNodeContributionSummary {
  contract_version: "multi_path_contribution_union_v2";
  layer: "coarse" | "mid" | "chunk";
  node_id: string;
  node_visit_count: number;
  distinct_parent_count: number;
  distinct_path_count: number;
  distinct_edge_type_count: number;
  parent_node_ids: string[];
  path_edge_types: string[];
  covered_facets: string[];
  evidence_roles: string[];
  support_id_union: string[];
  support_chunk_union: string[];
  cycle_convergence_score: number;
  best_distance: number;
  best_reward: number;
  reached_by_paths: RetrievalPathContribution[];
}

export interface GrayMinimumAudit {
  input_identity_protocol_version?: "gray_zone_minimum_replay_card_v1" | null;
  current_layer?: string | null;
  distance_zone?: "green" | "gray" | "red" | "hard_stop" | null;
  path_distance: number;
  predicates: Record<string, boolean>;
  required_facets_hash?: string | null;
  covered_facets_before_hash?: string | null;
  covered_facets_after_hash?: string | null;
  candidate_facets_hash?: string | null;
  evidence_roles_before_hash?: string | null;
  evidence_roles_after_hash?: string | null;
  bounded_support_ids_before_hash?: string | null;
  bounded_support_ids_after_hash?: string | null;
  support_ids_before_count?: number | null;
  support_ids_after_count?: number | null;
  support_ids_before_hash?: string | null;
  support_ids_after_hash?: string | null;
  support_ids_after: string[];
  independent_path_contribution_gain?: boolean | null;
  thresholds: Record<string, number | null>;
  edge_distance_protocol?: string | null;
  path_contribution_key?: string | null;
  support_refs_hash?: string | null;
  bridge_or_boundary_reason_hash?: string | null;
  edge_type?: string | null;
  rq_membership_diagnostics_hash?: string | null;
  candidate_chunk_span_summary_hash?: string | null;
  structure_context_status_hash?: string | null;
}

export interface GrayTraversalObservationBudgetState {
  protocol_version: string; scope: string; limit: number; local_rule_evaluation_index: number;
  layer_local_rule_evaluation_index: number; cadence_due: boolean; expanded_packet_requested: boolean;
  expanded_observation_count_before: number; expanded_observation_count_after: number; remaining_after: number;
  observation_compacted: boolean; compaction_reason?: string | null; hard_interrupt_applied: boolean;
  budget_exhausted_after: boolean; model_call_count: 0;
}

export interface GrayHardInterruptState {
  edge_reuse_count?: number | null; max_edge_reuse?: number | null;
  frontier_expansion_count?: number | null; frontier_expansion_budget?: number | null;
  per_entry_expansion_count?: number | null; per_entry_expansion_budget?: number | null;
  path_distance_hard_stop?: boolean | null;
  traversal_observation_budget?: GrayTraversalObservationBudgetState | null;
}

export interface GrayRQMembershipRoleThresholds {
  noise_membership_score_max: number;
  outlier_gamma_max: number;
  low_confidence_gamma_max: number;
  low_confidence_membership_score_max: number;
  boundary_entropy_min: number;
  boundary_probability_margin_max: number;
  boundary_distance_max: number;
}

export interface GrayRQMembershipRoleInputs {
  membership_score: number;
  membership_entropy: number;
  residual_norm: number;
  gamma: number;
  boundary_probability_margin: number;
  boundary_distance: number;
  residual_outlier_threshold: number;
  rank: number;
  is_primary_leaf: boolean;
  is_bridge_chunk: boolean;
}

export interface GrayRQMembershipRoleEvaluation {
  role: string;
  matched_flags: string[];
  primary_reason: string;
  protocol_version: string;
  protocol_hash: string;
  thresholds: GrayRQMembershipRoleThresholds;
  inputs: GrayRQMembershipRoleInputs;
  model_call_count: 0;
}

export interface GrayRQScore {
  query_rq_path: number[];
  candidate_rq_path: number[];
  lcp_depth: number;
  lcp_ratio_diagnostic_only: number;
  residual_distance: number;
  query_residual_norm: number;
  candidate_residual_norm: number;
  query_prefix_membership_score: number;
  candidate_prefix_membership_score: number;
  fuzzy_membership_overlap_score: number;
  rq_score: number;
  rq_drift_penalty: number;
  membership_reason: string;
  membership_role: string;
  membership_rank: number;
  membership_entropy?: number | null;
  boundary_probability_margin?: number | null;
  boundary_distance?: number | null;
  membership_role_evaluation?: GrayRQMembershipRoleEvaluation | null;
  membership_protocol_version?: string | null;
  membership_protocol_hash?: string | null;
  hard_path_lcp_used_as_score: false;
}

export interface GrayProjectedRQDiagnostics {
  rq_score?: number | null;
  rq_drift_penalty?: number | null;
  lcp_depth?: number | null;
  residual_distance?: number | null;
  query_prefix_membership_score?: number | null;
  candidate_prefix_membership_score?: number | null;
  fuzzy_membership_overlap_score?: number | null;
  membership_reason?: string | null;
  membership_role?: string | null;
  membership_rank?: number | null;
  membership_entropy?: number | null;
  boundary_probability_margin?: number | null;
  boundary_distance?: number | null;
  membership_protocol_version?: string | null;
  membership_protocol_hash?: string | null;
  hard_path_lcp_used_as_score?: false | null;
}

export interface GrayRQMembershipDiagnostics {
  projection_protocol_version?: "gray_rq_membership_observation_projection_v1" | null;
  source_present?: boolean | null;
  diagnostics?: GrayProjectedRQDiagnostics | null;
}

export interface GrayCandidateChunkSpanSummary {
  chunk_id?: string | null;
  document_version_id?: string | null;
  char_start?: number | null;
  char_end?: number | null;
}

export interface GrayStructureContextStatus {
  available?: boolean | null;
  reason?: string | null;
  mapped_chunk_id?: string | null;
}

export interface GrayObservation {
  current_layer: string; path_distance: number; distance_zone?: "green" | "gray" | null;
  covered_facets_before: string[]; covered_facets_after: string[]; missing_facets_after: string[];
  required_facets: string[]; candidate_facets: string[]; evidence_roles_before: string[]; evidence_roles_after: string[];
  support_ids_before: string[]; support_ids_after: string[]; support_ids_before_count?: number | null;
  support_ids_after_count?: number | null; support_ids_before_hash?: string | null; support_ids_after_hash?: string | null;
  support_id_gain?: boolean | null; independent_path_contribution_gain?: boolean | null;
  path_contribution_key?: string | null; support_refs?: RetrievalSupportRefs | null;
  active_edge_support_gate_pass?: boolean | null; support_gate_pass?: boolean | null;
  support_backed_to_covered_path?: boolean | null; validated_entry_semantic_anchor?: boolean | null;
  facet_gain?: boolean | null; role_gain?: boolean | null; support_gain?: boolean | null;
  query_anchor_preserved?: boolean | null; drift_risk_high?: boolean | null; closure_required?: boolean | null;
  semantic_uncertain_edge?: boolean | null; crossing_rq_boundary?: boolean | null;
  bridge_or_boundary_reason: string[]; bridge_eligible?: boolean | null; edge_type?: string | null;
  supported_raw_span_hit?: boolean | null; structure_context_available?: boolean | null;
  drilldown_eligible?: boolean | null; drift_risk?: boolean | null;
  rq_membership_diagnostics?: GrayRQMembershipDiagnostics | null;
  candidate_chunk_span_summary?: GrayCandidateChunkSpanSummary | null;
  structure_context_status?: GrayStructureContextStatus | null;
  hard_interrupt_state?: GrayHardInterruptState | null;
  path_distance_green_threshold?: number | null; path_distance_gray_threshold?: number | null;
  path_distance_hard_threshold?: number | null; gray_zone_rule_protocol_version?: string | null;
  support_progress?: boolean | null;
}

export interface RetrievalGrayZoneDecision {
  layer?: string | null;
  edge_id?: string | null;
  from_node_id?: string | null;
  to_node_id?: string | null;
  from_chunk_id?: string | null;
  to_chunk_id?: string | null;
  path_distance: number;
  distance_zone: "green" | "gray" | "red" | "hard_stop";
  decision: string;
  decision_reason?: string | null;
  protocol_version: string;
  protocol_hash: string;
  input_hash: string;
  threshold_hash: string;
  traversal_protocol_hash: string;
  runtime_settings_hash: string;
  agent_operating_envelope_hash: string;
  decision_hash: string;
  matched_rule: string;
  predicates: Record<string, boolean>;
  minimum_audit: GrayMinimumAudit;
  observation?: GrayObservation | null;
  observation_compacted?: boolean;
  hard_interrupt_state: GrayHardInterruptState;
  model_call_count: 0;
  decision_source: "deterministic_local_rule" | "deterministic_distance_partition";
  support_progress: Record<string, boolean | number | string>;
  support_refs: RetrievalSupportRefs;
  covered_facets: string[];
  semantic_uncertain_edge: boolean;
  crossing_rq_boundary: boolean;
  edge_type?: string | null;
  gray_candidate_reasons: string[];
}

export interface GrayAuditFinding {
  code: string; message?: string | null; field?: string | null;
  expected?: number | string | null; actual?: number | string | null; record_index?: number | null;
}

export interface RetrievalGrayZoneDeterminismAudit {
  status: "passed" | "incomplete" | "failed";
  checked_record_count: number;
  unique_record_count: number;
  local_rule_record_count: number;
  red_partition_record_count: number;
  hard_stop_partition_record_count: number;
  duplicate_reference_count: number;
  conflict_count: number;
  incomplete_record_count: number;
  conflicts: GrayAuditFinding[];
  issues: GrayAuditFinding[];
}

export interface RetrievalQueryRQSeedAudit {
  protocol_version: "query_rq_fuzzy_membership_chunk_seed_v2";
  protocol_hash: "144d218ea37a70f2aa85730624c5cec0807e20542078adc1574f832cbff017d8";
  requested_query_rq_scores: Record<string, number>;
  effective_rq_scores: Record<string, number>;
  explicit_query_relevance_precedence: true;
  selected_mid_route_fallback_only_when_missing: true;
  mid_support_baseline_may_mask_rq_seed: false;
  node_weight_used_as_query_relevance: false;
  hard_path_lcp_used_as_score: false;
  is_evidence: false;
  gray_zone_decision_authority: false;
  model_call_count: 0;
}

export interface RetrievalConvergence {
  reason?: string | null;
  convergence_replay_protocol_version?: string | null;
  entry_count?: number | null;
  frontier_remaining_count?: number | null;
  frontier_budget?: number | null;
  frontier_expansion_count?: number;
  dominance_pruned_count?: number;
  label_budget_pruned_count?: number;
  hard_stop_pruned_count?: number;
  red_zone_pruned_count?: number;
  gray_zone_decision_count?: number;
  gray_zone_rule_protocol_version?: string | null;
  gray_zone_rule_evaluation_count?: number;
  gray_zone_rule_stop_count?: number;
  gray_zone_observation_compacted_count?: number;
  gray_zone_model_call_count?: 0;
  gray_zone_observation_cadence?: number | null;
  traversal_observation_budget?: number | null;
  traversal_observation_expanded_count?: number;
  traversal_observation_budget_compacted_count?: number;
  traversal_observation_cadence_compacted_count?: number;
  traversal_observation_hard_interrupt_count?: number;
  traversal_observation_budget_hit?: boolean;
  traversal_observation_budget_audit?: {
    protocol_version: string; scope: string; limit: number; local_rule_evaluation_count: number;
    expanded_request_count: number; expanded_observation_count: number; cadence_compacted_count: number;
    budget_compacted_count: number; compacted_observation_count: number; hard_interrupt_count: number;
    budget_hit: boolean; traversal_expanded_observation_count: number; remaining: number; model_call_count: 0;
    stop_reason: string;
  } | null;
  query_facet_posterior_protocol_version?: "query_facet_posterior_calibration_v1" | null;
  query_facet_posterior_protocol_hash?: string | null;
  query_facet_posterior_rounds_used?: number;
  query_facet_posterior_observations_used?: number;
  query_facet_posterior_stop_reason?: string | null;
  query_facet_posterior_model_call_count?: 0 | null;
  edge_reuse_pruned_count?: number;
  duplicate_transition_pruned_count?: number;
  duplicate_transition_protocol_version?: string | null;
  frontier_remaining?: number;
  accepted_node_count?: number | null;
  accepted_chunk_count?: number | null;
  per_entry_expansion_budget?: number | null;
  expansion_count_by_entry?: Record<string, number>;
  path_distance_thresholds?: Record<string, number | null>;
  layers?: Record<string, RetrievalConvergence>;
  candidate_pool_dedupe_budget_audit?: CandidatePoolDedupeAudit | null;
  candidate_pool_dedupe_budget?: CandidatePoolDedupeSummary | null;
  query_rq_seed_protocol_version?: string | null;
  query_rq_seed_protocol_hash?: string | null;
  query_rq_seed_model_call_count?: 0 | null;
  query_rq_seed_gray_zone_decision_authority?: false | null;
  query_rq_seed_node_weight_used_as_query_relevance?: false | null;
  query_rq_seed_hard_lcp_used_as_score?: false | null;
  query_relevance_overwritten_by_mid_route_prior?: false | null;
  node_weight_used_as_query_relevance?: false | null;
  gray_zone_decision_authority?: false | null;
  model_call_count?: 0 | null;
  seed_count?: number | null;
  active_traversal_layer?: boolean | null;
  gray_zone_rule_protocol_hash?: string | null;
  runtime_settings_hash?: string | null;
  agent_operating_envelope_hash?: string | null;
  traversal_protocol_hash?: string | null;
  retrieval_granularity?: RetrievalGranularity | null;
  allowed_relation_types?: string[];
  allowed_relation_types_source?: "request_scoped_typed_action_control" | "frozen_agent_operating_envelope" | null;
}

export interface RetrievalStepInput {
  entry_node_ids: string[];
  coarse_entry_ids: string[];
  mid_entry_ids: string[];
  rq_membership_entry_ids: string[];
  query_rq_path: number[];
  result_chunk_ids: string[];
  hit_chunk_ids: string[];
  token_budget?: number | null;
  query_rq_seed_audit?: RetrievalQueryRQSeedAudit | null;
}

export interface GraphRetrievalStepResponse {
  id: string;
  step_index: number;
  layer: string;
  action: string;
  action_type: string;
  parent_layer?: string | null;
  parent_node_id?: string | null;
  input: RetrievalStepInput;
  output: { accepted_node_ids: string[]; selected_node_ids: string[]; accepted_chunk_ids: string[]; restored_chunk_count?: number | null; context_package_id?: string | null; source_span_count: number; convergence_reason?: string | null };
  candidate_pool_ids: string[];
  selected_topk_ids: string[];
  per_parent_budget_status: Record<string, PerParentBudgetStatus>;
  popped_frontier_state: RetrievalTraversalState;
  expanded_edge_ids: string[];
  dominance_pruned_count: number;
  cycle_distance_reward: number;
  gray_zone_path_decisions: RetrievalGrayZoneDecision[];
  stop_reason?: string | null;
  diagnostics: { retrieval_granularity: RetrievalGranularity; traversal_protocol?: string | null; scores_json_retired_as_primary_audit: true };
  created_at?: string | null;
}

export interface ContextSelectionReason {
  roles: string[];
  path_edge_ids: string[];
  covered_facets: string[];
  reason: string;
  reached_by_paths: RetrievalPathContribution[];
  query_facets: string[];
  evidence_roles: string[];
  graph_paths: string[][];
  graph_path_chunks: string[];
  convergence_score: number;
  node_visit_count: number;
  distinct_parent_count: number;
  distinct_path_count: number;
  distinct_edge_type_count: number;
  parent_node_ids: string[];
  support_chunk_union: string[];
}

export interface ContextStructurePathDiagnostics {
  chunk_segments: string[];
  structure_segments: string[];
  matched_segments: string[];
}

export interface ContextStructureMappingAdmissionDiagnostics {
  protocol_version: string;
  decision: "admit" | "reject";
  reason: string;
  same_scope: boolean;
  span_available: boolean;
  span_positive: boolean;
  bbox_available: boolean;
  bbox_positive: boolean;
  path_available: boolean;
  path_positive: boolean;
  exact_canonical_section_path: boolean;
  exact_section_candidate_count: number;
}

export interface ContextStructureMappingDiagnostics {
  mapping_protocol_version: string;
  mapping_admission_protocol_version: string;
  admission: ContextStructureMappingAdmissionDiagnostics;
  component_weights: Record<string, number>;
  effective_weights: Record<string, number>;
  available_components: string[];
  unavailable_components: string[];
  chunk_layout_ids: string[];
  structure_layout_ids: string[];
  path_diagnostics: ContextStructurePathDiagnostics;
}

export interface ContextStructureParserMetadataAudit {
  layout_protocol_version?: string | null;
  flow_block_protocol_version?: string | null;
  block_start_protocol_version?: string | null;
  html_block_protocol_version?: string | null;
  link_reference_protocol_version?: string | null;
  target_spec_version?: string | null;
  source_type?: string | null;
  parser?: string | null;
  native_layout_available?: boolean | null;
  page_size?: [number, number] | number[] | null;
  image_size?: [number, number] | number[] | null;
  cell_index?: number | null;
  layout_item_count?: number | null;
  structure_object_count?: number | null;
}

export interface ContextStructureOCRLayoutItemAudit {
  text: string;
  confidence: number;
  bbox: CitationBoundingBox;
}

export interface ContextStructureNativeMetadataAudit {
  native_structure?: boolean | null;
  parent_ref?: string | null;
  parser_source?: string | null;
  native_geometry?: boolean | null;
  layout_protocol_version?: string | null;
  flow_block_protocol_version?: string | null;
  block_start_protocol_version?: string | null;
  html_block_protocol_version?: string | null;
  link_reference_protocol_version?: string | null;
  target_spec_version?: string | null;
  structure_protocol_version?: string | null;
  row_normalization_protocol_version?: string | null;
  source_span_protocol?: string | null;
  indentation_protocol?: string | null;
  html_block_type?: number | null;
  column_count?: number | null;
  data_row_count?: number | null;
  column_alignments: string[];
  source_body_column_counts: number[];
  padded_missing_cell_counts: number[];
  ignored_excess_cell_counts: number[];
  source_indent_columns: number[];
  html_tag?: string | null;
  source?: string | null;
  ocr_applied?: boolean | null;
  ocr_engine?: string | null;
  ocr_confidence?: number | null;
  ocr_reason?: string | null;
  ocr_layout_items: ContextStructureOCRLayoutItemAudit[];
  image_size?: [number, number] | number[] | null;
  source_index?: number | null;
  span_remap_method?: string | null;
  original_char_span?: [number, number] | number[] | null;
  section_index?: number | null;
}

export interface ContextStructureLayoutAudit {
  coordinate_system?: string | null;
  structure_id?: string | null;
  layout_id?: string | null;
  block_id?: string | null;
  block_type?: string | null;
  source?: string | null;
  reading_order?: number | null;
  order_index?: number | null;
  synthetic?: boolean | null;
  source_path?: string | null;
  region_index?: number | null;
  layout_ids: string[];
  section_index?: number | null;
  block_index?: number | null;
  synthetic_page?: boolean | null;
  layout_protocol_version?: string | null;
  content_flags: string[];
  content_kind?: string | null;
  encoding_detected?: string | null;
  encoding_coherence?: number | null;
  encoding_used?: string | null;
  parser_path?: string | null;
  page_size?: [number, number] | number[] | null;
  native_layout_block_count?: number | null;
  pdf_image_count?: number | null;
  ocr_applied?: boolean | null;
  ocr_page_count?: number | null;
  ocr_reason?: string | null;
  mojibake_repaired?: boolean | null;
  mojibake_score_before?: number | null;
  mojibake_score_after?: number | null;
  text_cleaning_flags: string[];
  parser_metadata: ContextStructureParserMetadataAudit;
  metadata: ContextStructureNativeMetadataAudit;
}

export interface ContextStructureNodeAudit {
  node_id: string;
  node_type: string;
  title?: string | null;
  path?: string | null;
  depth: number;
  page_number?: number | null;
  bbox: CitationBoundingBox;
  layout: ContextStructureLayoutAudit;
  mapping_role: string;
  coverage_ratio: number;
  span_overlap: number;
  bbox_iou?: number | null;
  path_match?: number | null;
  mapping_weight: number;
  mapping_protocol_version: string;
  mapping_diagnostics: ContextStructureMappingDiagnostics;
}

export interface ContextStructureClosure {
  previous_chunk_id?: string | null;
  next_chunk_id?: string | null;
  parent_section_node_id?: string | null;
  same_page_region_node_ids: string[];
  table_formula_caption_node_ids: string[];
  code_block_node_ids: string[];
  bridge_chunk_ids: string[];
  parent_section?: ContextStructureNodeAudit | null;
  same_page_region: ContextStructureNodeAudit[];
  table_formula_caption: ContextStructureNodeAudit[];
  code_blocks: ContextStructureNodeAudit[];
}

export interface ContextPackageChunk {
  contract_kind: "context_chunk";
  chunk_id: string;
  document_id: string;
  document_version_id: string;
  document_title: string;
  source_path: string;
  logical_source_path: string;
  content: string;
  content_clipped: boolean;
  content_token_count: number;
  original_token_count: number;
  raw_chunk_char_span: [number, number] | number[];
  chunk_text_hash_protocol_version: string;
  chunk_text_hash: string;
  raw_span_text_hash_protocol_version: string;
  raw_span_text_hash: string;
  section_path?: string | string[] | null;
  structure_path?: string | string[] | null;
  structure_node_ids: string[];
  structure_nodes: ContextStructureNodeAudit[];
  parent_section?: ContextStructureNodeAudit | null;
  page_range: [number | null, number | null] | Array<number | null>;
  char_span: [number, number] | number[];
  bbox?: CitationBoundingBox | null;
  source_span: CitationSourceSpan;
  structure_closure: ContextStructureClosure;
  why_selected: ContextSelectionReason;
  dedupe_key: string;
  role: "hit" | "bridge" | "graph_path" | "restored_context";
  context_package_id: string;
}

export interface ContextMetadata {
  source_path: string;
  logical_source_path: string;
  section_path?: string | string[] | null;
  structure_path?: string | string[] | null;
  parent_section_node_id?: string | null;
  parent_section?: ContextStructureNodeAudit | null;
  structure_node_ids: string[];
  page_range: [number | null, number | null] | Array<number | null>;
  char_span: [number, number] | number[];
  bbox?: CitationBoundingBox | null;
  source_span: CitationSourceSpan;
  structure_closure: ContextStructureClosure;
  why_selected: ContextSelectionReason;
  dedupe_key: string;
  role: "hit" | "bridge" | "graph_path" | "restored_context";
  content_clipped: boolean;
  content_token_count: number;
  original_token_count: number;
  raw_chunk_char_span: [number, number] | number[];
  context_package_id: string;
}

export interface ContextItem {
  contract_kind: "context_item";
  chunk_id: string;
  document_title: string;
  source_path: string;
  content: string;
  snippet: string;
  metadata: ContextMetadata;
}

export interface ContextCitationSpan {
  contract_kind: "citation_span";
  document_id: string;
  document_title: string;
  source_path: string;
  logical_source_path: string;
  section_path?: string | string[] | null;
  structure_path?: string | string[] | null;
  structure_node_ids: string[];
  structure_closure: ContextStructureClosure;
  source_span: CitationSourceSpan;
}

export type ContextGraphExpansionPath =
  | { kind: "concept_path"; path: Array<{ layer: "coarse" | "mid" | "rq_membership" | "chunk"; ids: string[] }> }
  | { kind: "graph_path_ids"; edge_ids: string[] }
  | { kind: "restored_chunks"; chunk_ids: string[] }
  | { kind: "bridge_chunks"; chunk_ids: string[] }
  | { kind: "parent_structure_nodes"; node_ids: string[] };

export interface ContextPackageDiagnostics {
  context_restoration_protocol: string;
  repair_protocol_version?: string | null;
  repair_action_type?: string | null;
  repair_executor_mechanism?: string | null;
  repair_gray_zone_model_call_count: 0;
  repair_gray_zone_decision_authority: false;
  retrieval_granularity: RetrievalGranularity;
  conversation_state_scope_hash: string;
  conversation_state_is_evidence: false;
  runtime_settings_hash: string;
  profile_hash?: string | null;
  path_summary: { node_visit_count: number; distinct_parent_count: number; distinct_path_count: number; distinct_edge_type_count: number; covered_facets: string[]; support_chunk_union: string[]; reached_by_paths: RetrievalPathContribution[]; cycle_convergence_score: number };
  dedupe_keys: string[];
  restore_counts: { hit_chunks: number; restored_chunks: number; bridge_chunks: number; graph_path_chunks: number; parent_structure_nodes: number; per_hit_chunk_budget: number };
  token_budget_audit: { token_budget: number; token_count: number; within_budget: boolean; clipped_chunk_ids: string[]; skipped_chunk_ids: string[]; packing_protocol: string };
  snapshot_integrity: { protocol_version: string; verified_document_version_count: number; fail_closed: true };
}

export interface QueryFacetGroup {
  facet: string;
  role: string;
  aliases: string[];
  source: string;
  confidence: number;
}

export interface QueryFacetSchemaRejection {
  reason: string;
  fields: string[];
}

export interface QueryFacetDiagnostics {
  source: string;
  schema_validation: "canonical_facet_groups_only";
  query_facet_protocol_hash: string;
  lexical_terms: string[];
  dropped_query_terms: string[];
  llm_keys: string[];
  bilingual_query_facets_enabled?: boolean | null;
  output_contract_protocol_version?: "query_facet_nonempty_output_contract_v2" | null;
  sampling_model_call_count?: 1 | 2 | null;
  schema_repair_attempted?: boolean;
  schema_repair_protocol_version?: "query_facet_empty_group_schema_repair_v1" | null;
  llm_schema_rejection?: QueryFacetSchemaRejection | null;
  query_perception_audit?: OrdinaryQueryPerceptionAudit | null;
}

export interface QueryFacetPacket {
  query: string;
  protocol_version: "query_facet_packet_v2";
  terms: string[];
  required_facets: string[];
  facet_groups: QueryFacetGroup[];
  drop_terms: string[];
  answer_shape: string;
  intent: string;
  diagnostics: QueryFacetDiagnostics;
}

export interface QueryFacetPosteriorObservation {
  protocol_version: "query_facet_posterior_calibration_v1";
  checkpoint: "dense_entry_candidates" | "merged_chunk_candidates";
  layer: "chunk";
  scope: string;
  candidate_id: string;
  matched_facets: string[];
  matched_term_witnesses: Record<string, string[]>;
  query_facet_packet_hash: string;
  candidate_business_input_hash: string;
  model_call_count: 0;
  observation_id: string;
  observation_hash: string;
}

export interface QueryFacetPosteriorRound {
  round_index: number;
  checkpoint: "dense_entry_candidates" | "merged_chunk_candidates";
  prior: Record<string, number>;
  likelihood: Record<string, number>;
  posterior: Record<string, number>;
  alias_prior: Record<string, Record<string, number>>;
  alias_likelihood: Record<string, Record<string, number>>;
  alias_posterior: Record<string, Record<string, number>>;
  observation_count: number;
  facet_match_counts: Record<string, number>;
  l1_delta: number;
  model_call_count: 0;
  observation_ids: string[];
  round_hash: string;
}

export interface QueryFacetPosteriorCalibration {
  protocol_version: "query_facet_posterior_calibration_v1";
  protocol_hash: string;
  enabled: boolean;
  query_facet_packet_hash: string;
  required_facets: string[];
  prior: Record<string, number>;
  posterior: Record<string, number>;
  alias_posterior: Record<string, Record<string, number>>;
  rounds: QueryFacetPosteriorRound[];
  observations: QueryFacetPosteriorObservation[];
  round_budget: number;
  rounds_used: number;
  observation_budget: number;
  observations_used: number;
  convergence_epsilon: number;
  converged: boolean;
  stop_reason:
    | "disabled_no_required_facets"
    | "disabled_by_runtime_setting"
    | "converged"
    | "round_budget_exhausted"
    | "observation_budget_exhausted"
    | "checkpoint_sequence_exhausted";
  llm_sample_budget: 0;
  model_call_count: 0;
  is_evidence: false;
  citation_authority: false;
  graph_mutation_authority: false;
  gray_zone_decision_authority: false;
  posterior_used_as_numeric_query_relevance: false;
  posterior_used_only_within_equal_uncovered_count: true;
  calibration_hash: string;
}

export interface RetrievalAgentOperatingEnvelope {
  agent_coarse_initial_budget: number;
  agent_coarse_total_budget: number;
  agent_coarse_top_k: number;
  agent_mid_per_coarse_budget: number;
  agent_coarse_drilldown_mid_initial_budget: number;
  agent_mid_initial_budget: number;
  agent_mid_top_k: number;
  agent_chunk_per_mid_budget: number;
  agent_chunk_initial_budget: number;
  agent_chunk_top_k: number;
  max_depth_per_layer: number;
  max_labels_per_node: number;
  max_edge_reuse: number;
  max_cycle_reward_per_path: number;
  cycle_reward_distance_threshold: number;
  path_distance_green_threshold: number;
  path_distance_gray_threshold: number;
  path_distance_hard_threshold: number;
  gray_zone_rule_protocol_version: "deterministic_support_progress_v1";
  gray_zone_rule_protocol_hash: string;
  gray_zone_observation_cadence: number;
  traversal_observation_budget: number;
  gray_zone_model_call_budget: 0;
  query_facet_posterior_enabled: boolean;
  query_facet_posterior_observation_budget: number;
  query_facet_posterior_round_budget: number;
  query_facet_posterior_convergence_epsilon: number;
  query_facet_posterior_llm_sample_budget: 0;
  candidate_pool_dedupe_budget: number;
  structure_restore_per_chunk_budget: number;
  structure_restore_budget: number;
  context_package_token_budget: number;
  context_path_summary_budget: number;
  planning_round_budget: number;
  max_typed_actions_per_round: number;
  repair_round_budget: number;
  verification_budget: number;
  allowed_relation_types: Array<
    "dense_semantic" | "dense_cross_document_bridge" | "dense_cross_language_bridge"
  >;
  required_restore_modes: Array<"previous_next" | "parent_structure" | "bridge_chunks">;
}

export interface RetrievalTraceDiagnostics {
  context_graph_state_id?: string | null;
  retrieval_granularity: RetrievalGranularity;
  active_profile_hash?: string | null;
  canonical_active_profile_hash?: string | null;
  repair_protocol_version?: string | null;
  repair_action_type?: string | null;
  repair_executor_mechanism?: string | null;
  repair_directive_hash?: string | null;
  repair_gray_zone_decision_authority: false;
  repair_gray_zone_model_call_count: 0;
  coarse_skipped_reason?: string | null;
  runtime_settings_hash: string;
  gray_zone_runtime_settings_identity_protocol_version: "gray_zone_runtime_settings_identity_v1";
  gray_zone_runtime_settings_hash: string;
  gray_zone_query_facet_protocol_version: "deterministic_gray_query_tokenizer_v1";
  gray_zone_query_facet_hash: string;
  gray_zone_external_routing_packet_used: false;
  gray_zone_request_scoped_budget_in_identity: false;
  query_facet_posterior_calibration?: QueryFacetPosteriorCalibration | null;
  agent_operating_envelope: RetrievalAgentOperatingEnvelope;
  agent_operating_envelope_hash: string;
  effective_traversal_protocol_hash: string;
  result_top_k?: number | null;
  scores_json_retired_as_primary_audit: true;
  traversal_protocol?: string | null;
  rank_score_protocol_version?: string | null;
  rank_score_protocol_hash?: string | null;
  raw_strength_protocol_version?: string | null;
  raw_strength_protocol_hash?: string | null;
  chunk_node_quality_protocol_version?: string | null;
  chunk_node_quality_protocol_hash?: string | null;
  out_evidence_mass_protocol_version?: string | null;
  out_evidence_mass_protocol_hash?: string | null;
  in_acceptance_capacity_protocol_version?: string | null;
  in_acceptance_capacity_protocol_hash?: string | null;
  relation_quota_protocol_version?: string | null;
  relation_quota_protocol_hash?: string | null;
  edge_type_calibration_protocol_version?: string | null;
  edge_type_calibration_protocol_hash?: string | null;
  graph_operating_point_hash?: string | null;
  calibration_params_hash?: string | null;
  edge_type_calibration_config_hash?: string | null;
  edge_distance_protocol_version?: string | null;
  edge_distance_protocol_hash?: string | null;
  entry_selection_protocol_version?: string | null;
  entry_selection_protocol_hash?: string | null;
  entry_topology_protocol_version?: string | null;
  entry_topology_protocol_hash?: string | null;
  entry_replay_proof_protocol_version?: string | null;
  entry_dense_replay_protocol_version?: string | null;
  entry_dense_replay_input_hash?: string | null;
  entry_neutral_start_cost_protocol_version?: string | null;
  entry_selection_model_call_count?: 0;
  entry_selection_lexical_overlap_used_as_query_relevance?: false;
  entry_selection_topology_used_as_path_distance?: false;
  entry_selection_node_weight_used_as_query_relevance?: false;
  entry_neutral_start_cost_is_query_relevance?: false;
  entry_selection_gray_zone_rule_inputs_modified?: false;
  query_rq_seed_audit?: RetrievalQueryRQSeedAudit | null;
}

export interface ContextPackagePublicHashCard {
  protocol_version: "context_package_public_hash_v1";
  canonicalization: "json_utf8_sort_keys_compact_v1";
  hashed_public_fields: string[];
  public_payload_hash: string;
  public_citation_spans_hash: string;
  citation_spans_consistency: "persisted_equals_public_projection";
  chunk_count: number;
  citation_span_count: number;
  graph_expansion_path_count: number;
}

export interface ContextPackageResponse {
  contract_version: "context_package_public_v1";
  id: string;
  retrieval_trace_id?: string | null;
  knowledge_base_id: string;
  package_hash: string;
  package_hash_card: ContextPackagePublicHashCard;
  query: string;
  hit_chunk_ids?: string[];
  restored_chunk_ids?: string[];
  bridge_chunk_ids?: string[];
  parent_structure_node_ids?: string[];
  concept_path: Array<{ layer: "coarse" | "mid" | "rq_membership" | "chunk"; ids: string[] }>;
  graph_path_ids: string[];
  reached_by_paths: RetrievalPathContribution[];
  node_contributions: RetrievalNodeContributionSummary[];
  why_selected: Record<string, ContextSelectionReason>;
  cycle_convergence_score?: number | null;
  dedupe_keys?: string[];
  covered_facets?: string[];
  package: { contract_version: "context_package_chunks_v1"; chunks: ContextPackageChunk[] };
  contexts: ContextItem[];
  token_budget: number;
  token_count?: number;
  citation_spans: ContextCitationSpan[];
  graph_expansion_paths: ContextGraphExpansionPath[];
  diagnostics: ContextPackageDiagnostics;
  created_at?: string | null;
}

export interface RetrievalTraceStepsResponse {
  contract_version: "layered_retrieval_trace_public_v1";
  trace_id: string;
  context_package_id?: string | null;
  query?: string;
  retrieval_mode?: string;
  retrieval_granularity?: RetrievalGranularity;
  conversation_state_scope_hash: string;
  concept_path: Array<{ layer: "coarse" | "mid" | "rq_membership" | "chunk"; ids: string[] }>;
  query_facets: QueryFacetPacket;
  entry_nodes?: RetrievalEntryNode[];
  stage_queues?: Record<string, RetrievalStageQueue>;
  candidate_pools?: RetrievalCandidatePools;
  topk_selection?: Record<string, RetrievalTopKSelection>;
  frontier?: RetrievalFrontierSnapshot[];
  path_labels?: RetrievalPathLabel[];
  node_contributions: RetrievalNodeContributionSummary[];
  convergence?: RetrievalConvergence;
  trace_diagnostics: RetrievalTraceDiagnostics;
  rq_diagnostics: { query_rq_path: number[]; query_residual_norm?: number | null; index_protocol?: string | null };
  gray_zone_protocol: string;
  gray_zone_model_call_count: 0;
  gray_zone_determinism: RetrievalGrayZoneDeterminismAudit;
  gray_zone_path_decisions?: RetrievalGrayZoneDecision[];
  path_distance_threshold_hits?: RetrievalGrayZoneDecision[];
  result_chunk_ids?: string[];
  steps: GraphRetrievalStepResponse[];
}
