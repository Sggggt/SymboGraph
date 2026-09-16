/** Shared target contracts. These do not expose retired retrieval modes. */
export type RetrievalIntentKind =
  | "summarize" | "overview" | "define" | "fact_lookup" | "enumerate"
  | "compare" | "explain" | "procedure" | "analyze" | "relationship"
  | "source_lookup" | "system_capability" | "clarify";
export type RetrievalEntryLayer = "coarse" | "mid" | "chunk";
export type RetrievalChannel = "dense" | "rq" | "bm25";
export type RetrievalRequirementId = "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8";
export type RetrievalPlanReason =
  | "broad_scope" | "precise_terms" | "semantic_paraphrase" | "mixed_signal"
  | "source_locality" | "existing_evidence" | "system_request" | "ambiguous_request";

export interface RetrievalIntent {
  primary: RetrievalIntentKind;
  secondary: RetrievalIntentKind[];
}
export interface RetrievalChannelWeights {
  dense: number;
  rq: number;
  bm25: number;
}
export interface RetrievalLayerWeights {
  coarse?: RetrievalChannelWeights | null;
  mid?: RetrievalChannelWeights | null;
  chunk?: RetrievalChannelWeights | null;
}
export interface RetrievalLexicalSurface {
  text: string;
  language: "zh" | "en" | "neutral";
  provenance: "user_text" | "model_query";
}
export interface RetrievalLexicalGroup {
  group_id: string;
  requirement_ids: RetrievalRequirementId[];
  kind: "concept" | "identifier" | "number_unit" | "quoted_literal";
  surfaces: RetrievalLexicalSurface[];
}
export interface RetrievalExecutionBudget {
  dense_candidates: number;
  rq_candidates: number;
  bm25_candidates: number;
  root_entries: number;
  per_parent_entries: number;
  layer_entries: number;
  max_depth: number;
  restore_per_hit: number;
}
export interface RetrievalExecutionStrategy {
  protocol_version: "intent_execution_strategy_v2";
  route: "retrieve" | "verified_context_reuse" | "system_capability" | "clarify";
  entry_layer: RetrievalEntryLayer | null;
  semantic_query: string;
  generate_lexical: boolean;
  lexical_groups: RetrievalLexicalGroup[];
  hybrid: boolean;
  layer_weights: RetrievalLayerWeights;
  selection_scope: "focused" | "broad";
  budget_request: { [K in keyof RetrievalExecutionBudget]?: number | null };
  reason_code: RetrievalPlanReason;
}
export interface RetrievalCapabilityManifest {
  protocol_version: "retrieval_capabilities_v2";
  knowledge_base_id: string;
  available_layers: RetrievalEntryLayer[];
  available_channels: RetrievalChannel[];
  graph_identity: string | null;
  lexical_identity: string | null;
  bilingual_lexical_enabled: boolean;
  budget_limits: RetrievalExecutionBudget;
}
export interface RetrievalChannelContribution {
  channel: RetrievalChannel;
  raw_score: number;
  rank: number;
  weight: number;
  contribution: number;
  witness_ids: string[];
}
export interface RetrievalFusedEntry {
  candidate_id: string;
  business_key: string;
  score: number;
  channels: RetrievalChannelContribution[];
}
