import type {
  AgentTraceScores,
  AgentTraceEventPayload,
  Citation,
  CitationSourceSpan,
  CitationVerificationDiagnostics,
  ContextGraphExpansionPath,
  ContextPackageResponse,
  GraphEdge,
  GraphEdgeI18n,
  GraphProjectionEdgeDiagnostics,
  GraphResponse,
  RepairActionAudit,
  RetrievalTraceStepsResponse,
  SearchCitation,
  VerifiedCitationSourceSpan,
} from "@course-kg/shared";

type Assert<T extends true> = T;
type Not<T extends boolean> = T extends true ? false : true;
type IsRequired<T, K extends keyof T> = object extends Pick<T, K> ? false : true;
type HasOpenStringIndex<T> = string extends keyof T ? true : false;

// Required-field assertions are the frontend negative fixtures: a missing field
// or a renamed legacy field makes tsc fail instead of silently widening to a dict.
export type PublicContractCompileAssertions = [
  Assert<IsRequired<Citation, "source_span">>,
  Assert<IsRequired<Citation, "document_id">>,
  Assert<IsRequired<Citation, "document_version_id">>,
  Assert<IsRequired<Citation, "char_span">>,
  Assert<IsRequired<Citation, "page_range">>,
  Assert<IsRequired<Citation, "section_path">>,
  Assert<IsRequired<Citation, "context_package_id">>,
  Assert<IsRequired<Citation, "retrieval_trace_id">>,
  Assert<IsRequired<Citation, "answer_session_id">>,
  Assert<IsRequired<Citation, "citation_verification_id">>,
  Assert<IsRequired<Citation, "verification">>,
  Assert<IsRequired<CitationSourceSpan, "char_span">>,
  Assert<IsRequired<VerifiedCitationSourceSpan, "context_package_id">>,
  Assert<IsRequired<VerifiedCitationSourceSpan, "retrieval_trace_id">>,
  Assert<IsRequired<VerifiedCitationSourceSpan, "verification_id">>,
  Assert<IsRequired<CitationVerificationDiagnostics, "verification_method">>,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "claim_grounded_gate_protocol_version"
    >
  >,
  Assert<IsRequired<CitationVerificationDiagnostics, "claim_id">>,
  Assert<IsRequired<CitationVerificationDiagnostics, "claim_index">>,
  Assert<IsRequired<CitationVerificationDiagnostics, "answer_hash">>,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_provenance_protocol_version"
    >
  >,
  Assert<
    IsRequired<CitationVerificationDiagnostics, "citation_provenance_valid">
  >,
  Assert<
    IsRequired<CitationVerificationDiagnostics, "citation_provenance_hash">
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_provenance_fail_closed"
    >
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_provenance_llm_override_allowed"
    >
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_provenance_session_hash"
    >
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_provenance_persistence_gate_passed"
    >
  >,
  Assert<IsRequired<CitationVerificationDiagnostics, "rule_verdict">>,
  Assert<
    IsRequired<CitationVerificationDiagnostics, "llm_entailment_verdict">
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "llm_entailment_result_present"
    >
  >,
  Assert<
    "deterministic_exact_span_entailment" extends keyof CitationVerificationDiagnostics
      ? true
      : false
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_prompt_protocol_hash"
    >
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_grounding_envelope_protocol_version"
    >
  >,
  Assert<
    IsRequired<
      CitationVerificationDiagnostics,
      "citation_grounding_envelope_hash"
    >
  >,
  Assert<IsRequired<CitationVerificationDiagnostics, "citation_profile_hash">>,
  Assert<IsRequired<SearchCitation, "source_span">>,
  Assert<
    Not<"verification" extends keyof SearchCitation ? true : false>
  >,
  Assert<Not<"character_span" extends keyof CitationSourceSpan ? true : false>>,
  Assert<IsRequired<GraphResponse, "contract_version">>,
  Assert<
    IsRequired<
      Extract<GraphEdge, { contract_kind: "concept_projection_edge" }>,
      "projected_distance_raw"
    >
  >,
  Assert<
    IsRequired<
      Extract<GraphEdge, { contract_kind: "concept_projection_edge" }>,
      "raw_strength_summary"
    >
  >,
  Assert<
    IsRequired<
      Extract<GraphEdge, { contract_kind: "concept_projection_edge" }>,
      "diagnostics"
    >
  >,
  Assert<
    "edge_i18n" extends keyof GraphProjectionEdgeDiagnostics ? true : false
  >,
  Assert<IsRequired<GraphEdgeI18n, "protocol_version">>,
  Assert<IsRequired<GraphEdgeI18n, "status">>,
  Assert<Not<HasOpenStringIndex<GraphEdgeI18n>>>,
  Assert<Not<"raw_projected_distance" extends keyof GraphEdge ? true : false>>,
  Assert<IsRequired<RetrievalTraceStepsResponse, "contract_version">>,
  Assert<Not<HasOpenStringIndex<RetrievalTraceStepsResponse>>>,
  Assert<IsRequired<AgentTraceScores, "audit_kind">>,
  Assert<IsRequired<AgentTraceEventPayload, "sequence_index">>,
  Assert<Not<HasOpenStringIndex<AgentTraceScores>>>,
  Assert<Not<"gray_llm_calls" extends keyof AgentTraceScores ? true : false>>,
  Assert<
    Not<
      "citation_pass_rate" extends keyof Extract<
        AgentTraceScores,
        { audit_kind: "retrieval_stage" }
      >
        ? true
        : false
    >
  >,
  Assert<
    Not<
      "repair_actions" extends keyof Extract<
        AgentTraceScores,
        { audit_kind: "retrieval_stage" }
      >
        ? true
        : false
    >
  >,
  Assert<
    Not<
      "frontier_pops" extends keyof Extract<
        AgentTraceScores,
        { audit_kind: "citation_verification" }
      >
        ? true
        : false
    >
  >,
  Assert<IsRequired<ContextPackageResponse, "contract_version">>,
  Assert<Not<HasOpenStringIndex<ContextPackageResponse>>>,
  Assert<IsRequired<ContextGraphExpansionPath, "kind">>,
  Assert<IsRequired<RepairActionAudit, "action_type">>,
];
