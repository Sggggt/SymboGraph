// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { ContextPackageResponse, RetrievalGrayZoneDecision, RetrievalTraceStepsResponse } from "@course-kg/shared";
import { SearchTraceDetails } from "@/components/search-workspace";
import {
  CONTRACT_HASH,
  makeContextCitationSpan,
  makeContextItem,
  makeContextPackage,
  makeGrayBudgetState,
  makeGrayDecision,
  makeRetrievalStep,
  makeSourceSpan,
  makeSupportRefs,
  makeTraceDiagnostics,
  makeTraceResponse,
  makeTraversalState,
} from "@/test/public-contract-fixtures";

const hashes = {
  protocol_hash: "a".repeat(64),
  input_hash: "b".repeat(64),
  threshold_hash: "c".repeat(64),
  traversal_protocol_hash: "d".repeat(64),
  runtime_settings_hash: "e".repeat(64),
  agent_operating_envelope_hash: "f".repeat(64),
  decision_hash: "1".repeat(64),
};

function grayDecision(overrides: Partial<RetrievalGrayZoneDecision> = {}): RetrievalGrayZoneDecision {
  return makeGrayDecision({
    layer: "coarse",
    edge_id: "edge:gray",
    path_distance: 0.72,
    distance_zone: "gray",
    decision: "drill_down_layer",
    matched_rule: "4_supported_drilldown",
    protocol_version: "deterministic_support_progress_v1",
    ...hashes,
    predicates: { support_gate_pass: true, progress: true, drilldown_eligible: true },
    hard_interrupt_state: {
      traversal_observation_budget: makeGrayBudgetState({
        protocol_version: "traversal_observation_budget_hard_interrupt_v1",
        limit: 64,
        remaining_after: 63,
      }),
    },
    support_refs: makeSupportRefs({ support_chunk_ids: ["chunk:1"] }),
    observation_compacted: false,
    model_call_count: 0,
    decision_source: "deterministic_local_rule",
    semantic_uncertain_edge: true,
    crossing_rq_boundary: false,
    gray_candidate_reasons: ["distance_gray"],
    ...overrides,
  });
}

const trace: RetrievalTraceStepsResponse = makeTraceResponse({
  trace_id: "trace-complete",
  conversation_state_scope_hash: "9".repeat(64),
  context_package_id: "package-complete",
  query: "compare posterior and likelihood",
  retrieval_mode: "layered_context_graph",
  retrieval_granularity: "coarse",
  query_facets: {
    query: "compare posterior and likelihood",
    protocol_version: "query_facet_packet_v2",
    terms: ["posterior", "likelihood"],
    required_facets: ["posterior", "likelihood"],
    facet_groups: [{ facet: "posterior", role: "comparison", aliases: ["likelihood"], source: "query", confidence: 1 }],
    drop_terms: [],
    answer_shape: "comparison",
    intent: "compare",
    diagnostics: {
      source: "deterministic",
      schema_validation: "canonical_facet_groups_only",
      query_facet_protocol_hash: CONTRACT_HASH,
      lexical_terms: ["posterior", "likelihood"],
      dropped_query_terms: [],
      llm_keys: [],
    },
  },
  entry_nodes: [{
    layer: "coarse", node_id: "coarse:bayes", entry_strength: 0.91, roles: ["coarse_entry"],
    metadata: { rq_path_prefix: [], representative_terms: ["bayes"] },
  }],
  frontier: [
    {
      layer: "coarse",
      popped: makeTraversalState({
        layer: "coarse",
        node_id: "coarse:bayes",
        path: ["coarse:bayes", "mid:posterior"],
        path_edge_ids: ["edge:coarse-mid"],
        path_edge_distances: [0.33],
        path_edge_types: ["projection"],
        distance_so_far: 0.44,
        reward_so_far: 0.12,
        distance_zone: "gray",
        covered_facets: ["posterior"],
        evidence_roles: ["coarse_entry"],
        support_refs: makeSupportRefs({ support_chunk_ids: ["chunk:1"] }),
      }),
      queue_size_after_pop: 2,
      key: [0, 0.44, 1, -1],
    },
  ],
  stage_queues: {
    coarse: { entry_ids: ["coarse:bayes"], forced_entry_ids: [], forced_downstream_entry_ids: [], selected_ids: ["coarse:bayes"], accepted_ids: ["coarse:bayes"], top_k: 2, frontier_pop_count: 1 },
  },
  candidate_pools: {
    mid_by_coarse: [
      {
        parent_layer: "coarse",
        parent_node_id: "coarse:bayes",
        candidate_ids: ["mid:posterior", "mid:likelihood"],
        candidate_scores: { "mid:posterior": 0.9, "mid:likelihood": 0.8 },
        rq_seed_cards: {},
        rq_chunk_seed_cards: {},
        chunk_facet_priority_cards: {},
        ranking_protocol_version: null,
        ranking_protocol_hash: null,
        forced_candidate_ids: [],
        selected_ids: ["mid:posterior"],
        candidate_count: 2,
        top_k: 1,
        per_parent_budget_status: { budget: 2, candidate_count: 2, selected_count: 1, stop_reason: "top_k_selected" },
        candidate_dedupe_budget_audit: {
          protocol_version: "candidate_pool_dedupe_hard_interrupt_v1",
          scope: "mid_by_coarse",
          limit: 2,
          attempt_count: 2,
          unique_admitted_count: 2,
          duplicate_count: 0,
          rejected_new_count: 0,
          budget_hit: false,
          hard_interrupt_count: 0,
          rejected_candidate_id_samples: [],
          observation_compacted: false,
          stop_reason: "complete",
        },
      },
    ],
    chunk_by_mid: [],
    rq_membership_entries: {
      candidate_ids: ["rq:l3:posterior"],
      candidate_scores: { "rq:l3:posterior": 0.91 },
      rq_seed_cards: {
        "rq:l3:posterior": {
          protocol_version: "query_rq_fuzzy_membership_chunk_seed_v2",
          protocol_hash: "144d218ea37a70f2aa85730624c5cec0807e20542078adc1574f832cbff017d8",
          rq_prefix_id: "rq:l3:posterior",
          rq_path: [1, 2, 3],
          rq_level: 3,
          query_rq_path: [1, 2, 3],
          rq_lcp_depth: 3,
          residual_distance: 0.08,
          query_prefix_membership_score: 0.91,
          requested_query_relevance: 0.91,
          route_fallback_score: 0.2,
          parent_mid_contributions: [],
          score_source: "query_rq_relevance",
          effective_score: 0.91,
          forced_override: false,
          relation_state_hash: CONTRACT_HASH,
          is_evidence: false,
          node_weight_used_as_query_relevance: false,
          hard_path_lcp_used_as_score: false,
          gray_zone_decision_authority: false,
          model_call_count: 0,
          input_hash: CONTRACT_HASH,
          card_hash: CONTRACT_HASH,
        },
      },
      chunk_facet_priority_cards: {},
      rq_chunk_seed_cards: {},
      ranking_protocol_version: "query_rq_fuzzy_membership_chunk_seed_v2",
      ranking_protocol_hash: CONTRACT_HASH,
      forced_candidate_ids: [],
      selected_ids: ["rq:l3:posterior"],
      ranked_selected_ids: ["rq:l3:posterior"],
      candidate_count: 1,
    },
    candidate_dedupe_budget: {
      protocol_version: "candidate_pool_dedupe_hard_interrupt_v1",
      limit_per_pool: 2,
      pool_count: 1,
      budget_hit_pool_count: 0,
      hard_interrupt_count: 0,
      unique_admitted_count: 2,
      duplicate_count: 0,
      audits: [],
    },
  },
  topk_selection: { mid: { candidate_count: 2, top_k: 1, selected_ids: ["mid:posterior"], forced_selected_ids: [], stop_reason: "top_k_selected" } },
  path_labels: [{
    layer: "mid", node_id: "mid:posterior", path: ["coarse:bayes", "mid:posterior"],
    path_edge_ids: ["edge:coarse-mid"], path_edge_types: ["projection"], expanded_edge_ids: ["edge:coarse-mid"],
    covered_facets: ["posterior"], evidence_roles: ["projection"], support_refs: makeSupportRefs({ support_chunk_ids: ["chunk:1"] }),
    entry_parent_refs: [], path_edge_type_multiset: { projection: 1 }, edge_reuse_counts: {},
  }],
  convergence: {
    reason: "frontier_exhausted",
    dominance_pruned_count: 3,
    gray_zone_model_call_count: 0,
    gray_zone_rule_protocol_version: "deterministic_support_progress_v1",
    gray_zone_observation_cadence: 1,
    traversal_observation_budget: 64,
    traversal_observation_expanded_count: 1,
    traversal_observation_budget_compacted_count: 0,
    traversal_observation_cadence_compacted_count: 0,
    traversal_observation_hard_interrupt_count: 0,
    traversal_observation_budget_hit: false,
    traversal_observation_budget_audit: {
      protocol_version: "traversal_observation_budget_hard_interrupt_v1",
      scope: "trace",
      limit: 64,
      local_rule_evaluation_count: 1,
      expanded_request_count: 1,
      expanded_observation_count: 1,
      budget_compacted_count: 0,
      cadence_compacted_count: 0,
      compacted_observation_count: 0,
      hard_interrupt_count: 0,
      budget_hit: false,
      traversal_expanded_observation_count: 1,
      remaining: 63,
      model_call_count: 0,
      stop_reason: "within_budget",
    },
  },
  trace_diagnostics: {
    ...makeTraceDiagnostics({
      runtime_settings_hash: "9".repeat(64),
      gray_zone_runtime_settings_hash: hashes.runtime_settings_hash,
      agent_operating_envelope_hash: hashes.agent_operating_envelope_hash,
      effective_traversal_protocol_hash: hashes.traversal_protocol_hash,
    }),
    query_rq_seed_audit: {
      protocol_version: "query_rq_fuzzy_membership_chunk_seed_v2",
      protocol_hash: "144d218ea37a70f2aa85730624c5cec0807e20542078adc1574f832cbff017d8",
      requested_query_rq_scores: { "rq:l3:posterior": 0.91 },
      effective_rq_scores: { "rq:l3:posterior": 0.91 },
      explicit_query_relevance_precedence: true,
      selected_mid_route_fallback_only_when_missing: true,
      mid_support_baseline_may_mask_rq_seed: false,
      node_weight_used_as_query_relevance: false,
      hard_path_lcp_used_as_score: false,
      is_evidence: false,
      gray_zone_decision_authority: false,
      model_call_count: 0,
    },
  },
  gray_zone_protocol: "deterministic_support_progress_v1",
  gray_zone_model_call_count: 0,
  gray_zone_determinism: {
    status: "passed",
    checked_record_count: 2,
    unique_record_count: 2,
    local_rule_record_count: 1,
    red_partition_record_count: 0,
    hard_stop_partition_record_count: 1,
    duplicate_reference_count: 0,
    conflict_count: 0,
    incomplete_record_count: 0,
    conflicts: [],
    issues: [],
  },
  gray_zone_path_decisions: [
    grayDecision(),
  ],
  path_distance_threshold_hits: [
    grayDecision({
      layer: "chunk",
      edge_id: "edge:hard",
      path_distance: 1.8,
      distance_zone: "hard_stop",
      decision: "hard_stop_pruned",
      matched_rule: "distance_hard_stop",
      input_hash: "2".repeat(64),
      decision_hash: "3".repeat(64),
      decision_source: "deterministic_distance_partition",
      support_refs: makeSupportRefs({ support_chunk_ids: ["chunk:hard"] }),
      semantic_uncertain_edge: false,
      gray_candidate_reasons: [],
    }),
  ],
  steps: [
    makeRetrievalStep({
      id: "step:mid",
      step_index: 2,
      layer: "mid",
      action: "walk_graph_frontier",
      action_type: "walk_graph_frontier",
      parent_layer: "coarse",
      parent_node_id: "coarse:bayes",
      candidate_pool_ids: ["mid:posterior", "mid:likelihood"],
      selected_topk_ids: ["mid:posterior"],
      expanded_edge_ids: ["edge:coarse-mid"],
      dominance_pruned_count: 3,
      cycle_distance_reward: 0.12,
      stop_reason: "frontier_exhausted",
    }),
  ],
});

const baseContextPackage = makeContextPackage();
const baseContextItem = makeContextItem();
const chunkSourceSpan = makeSourceSpan({
  chunk_id: "chunk:1",
  document_version_id: "document-version:bayes",
  source_path: "bayes.md",
  logical_source_path: "bayes.md",
  char_span: [0, 100],
  raw_chunk_char_span: [0, 100],
  section_path: ["Bayes", "Posterior"],
  structure_path: ["Bayes", "Posterior"],
  context_package_id: "package-complete",
  retrieval_trace_id: "trace-complete",
});
const reachedPath: ContextPackageResponse["reached_by_paths"][number] = {
  contract_version: "multi_path_contribution_v2",
  contribution_id: "2".repeat(64),
  layer: "chunk",
  node_id: "chunk:1",
  parent_layer: "mid",
  parent_node_id: "mid:posterior",
  origin_parent_layer: "mid",
  origin_parent_node_id: "mid:posterior",
  root_node_id: "coarse:bayes",
  path: ["coarse:bayes", "mid:posterior", "chunk:1"],
  path_edge_ids: ["edge:coarse-mid", "edge:mid-chunk"],
  path_edge_types: ["coarse_support", "mid_support"],
  covered_facets: ["posterior", "likelihood"],
  evidence_roles: ["graph_path", "hit"],
  support_refs: makeSupportRefs({
    edge_ids: ["edge:coarse-mid", "edge:mid-chunk"],
    edge_types: ["coarse_support", "mid_support"],
    support_chunk_ids: ["chunk:1"],
  }),
  support_chunk_ids: ["chunk:1"],
  distance_so_far: 0.34,
  reward_so_far: 0.12,
};
const selectionReason: ContextPackageResponse["why_selected"][string] = {
  roles: ["hit"],
  path_edge_ids: ["edge:coarse-mid", "edge:mid-chunk"],
  covered_facets: ["posterior", "likelihood"],
  reason: "hit",
  reached_by_paths: [reachedPath],
  query_facets: ["posterior", "likelihood"],
  evidence_roles: ["graph_path", "hit"],
  graph_paths: [["coarse:bayes", "mid:posterior", "chunk:1"]],
  graph_path_chunks: ["chunk:1"],
  convergence_score: 0.12,
  node_visit_count: 1,
  distinct_parent_count: 1,
  distinct_path_count: 1,
  distinct_edge_type_count: 2,
  parent_node_ids: ["mid:posterior"],
  support_chunk_union: ["chunk:1"],
};
const contextPackage: ContextPackageResponse = makeContextPackage({
  id: "package-complete",
  retrieval_trace_id: "trace-complete",
  knowledge_base_id: "kb:sample",
  package_hash: CONTRACT_HASH,
  query: "compare posterior and likelihood",
  hit_chunk_ids: ["chunk:1"],
  restored_chunk_ids: ["chunk:0", "chunk:2"],
  bridge_chunk_ids: ["chunk:bridge"],
  parent_structure_node_ids: ["section:bayes"],
  concept_path: [{ layer: "coarse", ids: ["coarse:bayes"] }, { layer: "mid", ids: ["mid:posterior"] }],
  graph_path_ids: ["edge:coarse-mid", "edge:mid-chunk"],
  reached_by_paths: [reachedPath],
  node_contributions: [{
    contract_version: "multi_path_contribution_union_v2",
    layer: "chunk",
    node_id: "chunk:1",
    node_visit_count: 1,
    distinct_parent_count: 1,
    distinct_path_count: 1,
    distinct_edge_type_count: 2,
    parent_node_ids: ["mid:posterior"],
    path_edge_types: ["coarse_support", "mid_support"],
    covered_facets: ["posterior", "likelihood"],
    evidence_roles: ["graph_path", "hit"],
    support_id_union: ["chunk:1"],
    support_chunk_union: ["chunk:1"],
    cycle_convergence_score: 0.12,
    best_distance: 0.34,
    best_reward: 0.12,
    reached_by_paths: [reachedPath],
  }],
  why_selected: { "chunk:1": selectionReason },
  cycle_convergence_score: 0.12,
  dedupe_keys: ["chunk:1:[0,100]", "chunk:0:[0,80]"],
  covered_facets: ["posterior", "likelihood"],
  contexts: [
    {
      ...baseContextItem,
      chunk_id: "chunk:1",
      document_title: "Bayes Notes",
      source_path: "bayes.md",
      content: "Posterior is proportional to likelihood times prior.",
      snippet: "Posterior is proportional to likelihood times prior.",
      metadata: {
        ...baseContextItem.metadata,
        source_path: "bayes.md",
        logical_source_path: "bayes.md",
        role: "hit",
        dedupe_key: "chunk:1:[0,100]",
        structure_path: ["Bayes", "Posterior"],
        structure_closure: {
          previous_chunk_id: "chunk:0", next_chunk_id: "chunk:2",
          same_page_region_node_ids: [], table_formula_caption_node_ids: [], code_block_node_ids: [], bridge_chunk_ids: [],
          same_page_region: [], table_formula_caption: [], code_blocks: [],
        },
        why_selected: selectionReason,
        source_span: chunkSourceSpan,
        char_span: [0, 100],
        raw_chunk_char_span: [0, 100],
        context_package_id: "package-complete",
      },
    },
  ],
  token_budget: 800,
  token_count: 26,
  citation_spans: [makeContextCitationSpan({
    document_id: "document:bayes",
    document_title: "Bayes Notes",
    source_path: "bayes.md",
    logical_source_path: "bayes.md",
    section_path: ["Bayes", "Posterior"],
    structure_path: ["Bayes", "Posterior"],
    source_span: chunkSourceSpan,
  })],
  graph_expansion_paths: [{
    kind: "concept_path",
    path: [{ layer: "coarse", ids: ["coarse:bayes"] }, { layer: "mid", ids: ["mid:posterior"] }],
  }],
  diagnostics: {
    ...baseContextPackage.diagnostics,
    path_summary: {
      node_visit_count: 1,
      distinct_parent_count: 1,
      distinct_path_count: 1,
      distinct_edge_type_count: 2,
      covered_facets: ["posterior", "likelihood"],
      support_chunk_union: ["chunk:1"],
      reached_by_paths: [reachedPath],
      cycle_convergence_score: 0.12,
    },
    dedupe_keys: ["chunk:1:[0,100]", "chunk:0:[0,80]"],
    restore_counts: { hit_chunks: 1, restored_chunks: 2, bridge_chunks: 1, graph_path_chunks: 0, parent_structure_nodes: 1, per_hit_chunk_budget: 2 },
    token_budget_audit: { ...baseContextPackage.diagnostics.token_budget_audit, token_budget: 800, token_count: 26 },
  },
});

describe("SearchTraceDetails", () => {
  afterEach(cleanup);

  it("renders the complete layered trace, deterministic gray audit, and context package dedupe", () => {
    render(
      <SearchTraceDetails
        traceId="trace-complete"
        contextPackageId="package-complete"
        trace={trace}
        contextPackage={contextPackage}
        isLoading={false}
        error={null}
      />,
    );

    expect(screen.getByTestId("search-trace-details")).toBeTruthy();
    expect(screen.getByText("Frontier expansion timeline")).toBeTruthy();
    expect(screen.getByText("Multi-path contribution union")).toBeTruthy();
    expect(screen.getByText("Context reached paths and why-selected union")).toBeTruthy();
    expect(screen.getByText("逐父候选池、合并与去重")).toBeTruthy();
    expect(screen.getAllByText("RQ seed ranking / scores").length).toBeGreaterThan(0);
    expect(screen.getByText("Query → RQ seed cards")).toBeTruthy();
    expect(screen.getByText("Query → RQ seed authority audit")).toBeTruthy();
    expect(screen.getByText("持久化检索步骤")).toBeTruthy();
    expect(screen.getByText("Deterministic Local Gray Rule（路径裁决）")).toBeTruthy();
    expect(screen.getByText("Path distance threshold hits")).toBeTruthy();
    expect(screen.getAllByText(/4_supported_drilldown/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Gray 模型调用 0/)).toBeTruthy();
    expect(screen.getAllByText(hashes.runtime_settings_hash).length).toBeGreaterThan(0);
    expect(screen.getAllByText(hashes.agent_operating_envelope_hash).length).toBeGreaterThan(0);
    expect(screen.getAllByText(hashes.decision_hash).length).toBeGreaterThan(0);
    expect(screen.getByText(/gray_candidate_reasons=distance_gray/)).toBeTruthy();
    expect(screen.getByText(/determinism passed/)).toBeTruthy();
    expect(screen.getByText(/observation budget 64/)).toBeTruthy();
    expect(screen.getByText(/budget compacted 0/)).toBeTruthy();
    expect(screen.getByText(/hard interrupt 0/)).toBeTruthy();
    expect(screen.getByText("Context Package 去重与结构恢复")).toBeTruthy();
    expect(screen.getByText("Public package hash proof")).toBeTruthy();
    expect(screen.getByText(/去重键 2 \/ 唯一 2/)).toBeTruthy();
    expect(screen.getAllByText(/Posterior is proportional/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("edge:coarse-mid").length).toBeGreaterThan(0);
  });

  it("fails closed when a required gray decision hash is missing", () => {
    const invalidTrace = {
      ...trace,
      gray_zone_path_decisions: [
        {
          ...trace.gray_zone_path_decisions?.[0],
          runtime_settings_hash: undefined,
        },
      ],
    };

    render(
      <SearchTraceDetails
        traceId="trace-complete"
        contextPackageId="package-complete"
        trace={invalidTrace as unknown as RetrievalTraceStepsResponse}
        contextPackage={contextPackage}
        isLoading={false}
        error={null}
      />,
    );

    expect(screen.getAllByTestId("gray-zone-audit-failure").length).toBeGreaterThan(0);
    expect(screen.getByText(/runtime_settings_hash 必须是规范的 SHA-256/)).toBeTruthy();
    expect(screen.queryByText(/Gray 模型调用 0/)).toBeNull();
  });

  it("renders retrieval trace HTTP 409 as a red audit conflict", () => {
    const error = Object.assign(new Error("persisted trace mismatch"), { status: 409 });

    render(
      <SearchTraceDetails
        traceId="trace-conflict"
        contextPackageId={null}
        trace={undefined}
        contextPackage={undefined}
        isLoading={false}
        error={error}
      />,
    );

    expect(screen.getByText("Gray-zone 持久化轨迹校验冲突")).toBeTruthy();
    const failure = screen.getByTestId("gray-zone-audit-failure");
    expect(failure.getAttribute("role")).toBe("alert");
    expect(failure.className).toContain("border-rose-300/35");
    expect(failure.textContent).toContain("HTTP 409");
  });
});
