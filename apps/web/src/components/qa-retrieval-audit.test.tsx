// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AgentTraceEventPayload, RetrievalTraceStepsResponse } from "@course-kg/shared";
import { fetchRetrievalTraceSteps } from "@/lib/api";
import { QARetrievalAudit } from "@/components/qa-retrieval-audit";
import {
  makeGrayBudgetState,
  makeGrayDecision,
  makeSupportRefs,
  makeTraceResponse,
} from "@/test/public-contract-fixtures";

vi.mock("@/lib/api", () => ({
  fetchRetrievalTraceSteps: vi.fn(),
}));

const protocolHash = "a".repeat(64);
const inputHash = "b".repeat(64);
const thresholdHash = "c".repeat(64);
const traversalHash = "d".repeat(64);
const runtimeHash = "e".repeat(64);
const envelopeHash = "f".repeat(64);
const decisionHash = "1".repeat(64);

const validTrace: RetrievalTraceStepsResponse = makeTraceResponse({
  trace_id: "trace:qa",
  conversation_state_scope_hash: "9".repeat(64),
  gray_zone_determinism: {
    status: "passed",
    checked_record_count: 1,
    unique_record_count: 1,
    local_rule_record_count: 1,
    red_partition_record_count: 0,
    hard_stop_partition_record_count: 0,
    duplicate_reference_count: 0,
    conflict_count: 0,
    incomplete_record_count: 0,
    conflicts: [],
    issues: [],
  },
  gray_zone_path_decisions: [
    makeGrayDecision({
      layer: "mid",
      edge_id: "edge:qa-gray",
      path_distance: 0.82,
      distance_zone: "gray",
      decision: "continue_path",
      decision_reason: "supported_progress",
      protocol_version: "deterministic_support_progress_v1",
      protocol_hash: protocolHash,
      input_hash: inputHash,
      threshold_hash: thresholdHash,
      traversal_protocol_hash: traversalHash,
      runtime_settings_hash: runtimeHash,
      agent_operating_envelope_hash: envelopeHash,
      decision_hash: decisionHash,
      matched_rule: "2_supported_progress",
      predicates: { support_gate_pass: true, progress: true },
      hard_interrupt_state: {
        traversal_observation_budget: makeGrayBudgetState({
          protocol_version: "traversal_observation_budget_hard_interrupt_v1",
          limit: 64,
          remaining_after: 63,
        }),
      },
      support_refs: makeSupportRefs({ support_chunk_ids: ["chunk:qa"] }),
      semantic_uncertain_edge: true,
      crossing_rq_boundary: false,
      gray_candidate_reasons: ["semantic_uncertain_edge"],
    }),
  ],
  path_distance_threshold_hits: [],
  convergence: {
    gray_zone_model_call_count: 0,
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
  steps: [],
});

const evaluatorTrace: AgentTraceEventPayload[] = [
  {
    contract_version: "agent_trace_event_public_v1",
    type: "trace",
    id: "event:evaluator",
    run_id: "run:qa",
    sequence_index: 0,
    node: "evidence_evaluator",
    status: "completed",
    input_summary: `observation=${inputHash}`,
    output_summary: "sufficient",
    document_ids: ["chunk:qa"],
    scores: {
      contract_version: "agent_trace_scores_public_v1",
      audit_kind: "evidence_evaluator",
      plan_index: 0,
      replan_requested: false,
      verdict: {
        verdict: "sufficient",
        reason: "citable spans present",
        target_ids: [],
        expected_evidence: {
          required_facets: [],
          allowed_relation_types: [],
          relation_types: [],
          required_restore_modes: [],
          required_evidence_roles: [],
          failure_types: [],
          failure_card_hashes: [],
        },
        decision_hash: "2".repeat(64),
        prompt_protocol_hash: "3".repeat(64),
      },
    },
    duration_ms: 14,
  },
];

function renderAudit(
  traceId: string | null = "trace:qa",
  agentTrace: AgentTraceEventPayload[] = evaluatorTrace,
) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <QARetrievalAudit traceId={traceId} agentTrace={agentTrace} />
    </QueryClientProvider>,
  );
}

describe("QARetrievalAudit", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("fetches the persisted trace by retrieval_trace_id and separates deterministic gray decisions from the LLM evaluator", async () => {
    vi.mocked(fetchRetrievalTraceSteps).mockResolvedValue(validTrace);

    renderAudit();

    await waitFor(() => expect(fetchRetrievalTraceSteps).toHaveBeenCalledWith("trace:qa"));
    expect(await screen.findByText("Deterministic Local Gray Rule（路径裁决）")).toBeTruthy();
    expect(screen.getByText("Query Facet Posterior（检索路由校准）")).toBeTruthy();
    expect(screen.getByText("LLM Evidence Evaluator（证据充分性 / 重规划）")).toBeTruthy();
    expect(screen.getAllByText(/model_call_count = 0/).length).toBeGreaterThan(0);
    expect(screen.getByText(runtimeHash)).toBeTruthy();
    expect(screen.getByText(envelopeHash)).toBeTruthy();
    expect(screen.getByText(decisionHash)).toBeTruthy();
    expect(screen.getByText(/gray_candidate_reasons=semantic_uncertain_edge/)).toBeTruthy();
    expect(screen.getByText(/observation budget 64/)).toBeTruthy();
    expect(screen.getByText(/expanded 1/)).toBeTruthy();
    expect(screen.getByText(/budget hit false/)).toBeTruthy();
    expect(screen.getByText(/不得裁决任何 gray path/)).toBeTruthy();
  });

  it("points empty stream evaluator state to the persisted PostgreSQL P&E audit", async () => {
    vi.mocked(fetchRetrievalTraceSteps).mockResolvedValue(validTrace);

    renderAudit("trace:qa", []);

    expect(await screen.findByText(/PostgreSQL P&E audit/)).toBeTruthy();
    expect(screen.getByText(/evaluator observation/)).toBeTruthy();
  });

  it("renders an HTTP 409 retrieval audit error as a red fail-closed block", async () => {
    const error = Object.assign(new Error("decision hash mismatch"), {
      status: 409,
      structured: { code: "retrieval_trace_audit_failed", message: "decision hash mismatch" },
    });
    vi.mocked(fetchRetrievalTraceSteps).mockRejectedValue(error);

    renderAudit();

    expect(await screen.findByText("Gray-zone 持久化轨迹校验冲突")).toBeTruthy();
    const failure = screen.getByTestId("gray-zone-audit-failure");
    expect(failure.getAttribute("role")).toBe("alert");
    expect(failure.className).toContain("border-rose-300/35");
    expect(failure.textContent).toContain("HTTP 409");
  });

  it("fails closed instead of displaying a default zero when the required model count is missing", async () => {
    const invalid = { ...validTrace, gray_zone_model_call_count: undefined };
    vi.mocked(fetchRetrievalTraceSteps).mockResolvedValue(invalid as unknown as RetrievalTraceStepsResponse);

    renderAudit();

    expect(await screen.findByText("Gray-zone 审计失败")).toBeTruthy();
    expect(screen.getByText(/gray_zone_model_call_count 必须显式等于 0/)).toBeTruthy();
    expect(screen.queryByTestId("gray-zone-audit-details")).toBeNull();
    expect(screen.getByTestId("query-facet-posterior-audit")).toBeTruthy();
  });
});
