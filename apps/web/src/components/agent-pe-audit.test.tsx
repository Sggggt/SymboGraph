// @vitest-environment jsdom

import { createHash } from "node:crypto";

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  AgentPEAuditResponse,
  AgentPEJsonPayload,
} from "@course-kg/shared";
import {
  AgentPEAuditPanel,
  peAuditIntegrityIssue,
} from "@/components/agent-pe-audit";
import { fetchAgentPEAudit } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  fetchAgentPEAudit: vi.fn(),
}));

const hash = "a".repeat(64);

function payload(value: string): AgentPEJsonPayload {
  return {
    encoding: "canonical_json_v1",
    canonical_json: value,
    sha256: createHash("sha256").update(value).digest("hex"),
    redacted_fields: [],
  };
}

const repairValidatorPayload = payload(
  `{"action_input_hash":"${hash}","remaining_repair_budget_before":1,"repair_protocol_version":"repair_v1","repair_round_index":0,"valid":true}`,
);
const repairRoundPayload = payload(
  `{"action_input_hash":"${hash}","action_output_hash":"${hash}","action_type":"repair_missing_citation","before_context_package_id":"package:1","before_retrieval_trace_id":null,"protocol_version":"repair_v1","remaining_repair_budget_after":0,"remaining_repair_budget_before":1,"repair_round_index":0,"repaired_context_package_id":"package:2","repaired_retrieval_trace_id":null}`,
);

const audit: AgentPEAuditResponse = {
  contract_version: "agent_pe_audit_public_v1",
  run_id: "run:pe",
  knowledge_base_id: "kb:pe",
  run_status: "completed",
  counts: { plans: 1, actions: 2, observations: 2 },
  ordering: {
    plans: "plan_index ASC, created_at ASC, id ASC",
    actions:
      "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC",
    observations: "created_at ASC, id ASC",
  },
  plans: [
    {
      contract_version: "agent_plan_audit_row_v1",
      order_index: 0,
      id: "plan-z",
      run_id: "run:pe",
      knowledge_base_id: "kb:pe",
      retrieval_trace_id: "trace:pe",
      plan_index: 0,
      planner_protocol_version: "planner_v1",
      typed_action_schema_protocol_version: "typed_action_v1",
      typed_action_schema_protocol_hash: hash,
      typed_action_executor_protocol_version: "executor_v1",
      input_hash: hash,
      output_hash: hash,
      control_hash: hash,
      query_intent: payload('{"intent":"qa"}'),
      operating_envelope: payload('{"budget":2}'),
      typed_actions: payload('["action-z","action-a"]'),
      validation: payload('{"valid":true}'),
      planner_model_metadata: payload('{"provider":"redacted"}'),
      status: "evidence_sufficient",
      diagnostics: payload('{"round":0}'),
      action_ids: ["action-z", "action-a"],
      action_count: 2,
      redacted_fields: [],
      created_at: "2026-07-27T00:00:00Z",
    },
  ],
  actions: [
    {
      contract_version: "agent_action_audit_row_v1",
      order_index: 0,
      id: "action-z",
      run_id: "run:pe",
      plan_id: "plan-z",
      plan_index: 0,
      parent_action_id: null,
      action_index: 0,
      action_type: "recall_chunks",
      target_ids: ["chunk:1"],
      reason: "collect direct evidence",
      budget_request: payload('{"chunk_budget":1}'),
      expected_evidence: payload('{"roles":["direct"]}'),
      stop_condition: payload('{"when":"support_found"}'),
      validator: {
        valid: true,
        plan_valid: true,
        schema_checked: true,
        budget_checked: true,
        target_scope_checked: true,
        typed_action_schema_protocol_version: "typed_action_v1",
        typed_action_schema_protocol_hash: hash,
        payload: payload('{"valid":true}'),
      },
      status: "completed",
      input_hash: hash,
      output_hash: hash,
      control_hash: hash,
      output: payload('{"trace_id":"trace:pe"}'),
      diagnostics: payload('{"bounded":true}'),
      observation_ids: ["observation-z"],
      observation_count: 1,
      redacted_fields: [],
      created_at: "2026-07-27T00:00:01Z",
    },
    {
      contract_version: "agent_action_audit_row_v1",
      order_index: 1,
      id: "action-a",
      run_id: "run:pe",
      plan_id: "plan-z",
      plan_index: 0,
      parent_action_id: "action-z",
      action_index: 1,
      action_type: "repair_missing_citation",
      target_ids: ["chunk:1"],
      reason: "repair citation span",
      budget_request: payload('{"repair_budget":1}'),
      expected_evidence: payload('{"roles":["citation"]}'),
      stop_condition: payload('{"when":"verified"}'),
      validator: {
        valid: true,
        plan_valid: true,
        schema_checked: true,
        budget_checked: true,
        target_scope_checked: true,
        repair_budget_checked: true,
        repair_protocol_version: "repair_v1",
        repair_round_index: 0,
        remaining_repair_budget_before: 1,
        action_input_hash: hash,
        payload: repairValidatorPayload,
      },
      status: "completed",
      input_hash: hash,
      output_hash: hash,
      control_hash: hash,
      output: repairRoundPayload,
      diagnostics: payload('{"repair":true}'),
      observation_ids: ["observation-a"],
      observation_count: 1,
      redacted_fields: [],
      created_at: "2026-07-27T00:00:02Z",
    },
  ],
  observations: [
    {
      contract_version: "agent_observation_audit_row_v1",
      order_index: 0,
      id: "observation-z",
      run_id: "run:pe",
      plan_id: "plan-z",
      plan_index: 0,
      action_id: "action-z",
      action_index: 0,
      parent_action_id: null,
      observation_type: "evidence_evaluator",
      protocol_version: "evaluator_v1",
      input_hash: hash,
      output_hash: hash,
      control_hash: hash,
      evaluator_linkage: {
        plan_id: "plan-z",
        plan_index: 0,
        protocol_version: "evaluator_v1",
        verdict: "need_structure_closure",
        decision_hash: hash,
        replan_requested: true,
        schema_repair_attempted: true,
        gray_zone_model_call_count: 0,
      },
      repair_linkage: null,
      evidence_chunk_ids: ["chunk:1"],
      verdict: "need_structure_closure",
      observation: payload('{"support":"partial"}'),
      diagnostics: payload('{"gray_zone_model_call_count":0}'),
      redacted_fields: [],
      created_at: "2026-07-27T00:00:03Z",
    },
    {
      contract_version: "agent_observation_audit_row_v1",
      order_index: 1,
      id: "observation-a",
      run_id: "run:pe",
      plan_id: "plan-z",
      plan_index: 0,
      action_id: "action-a",
      action_index: 1,
      parent_action_id: "action-z",
      observation_type: "typed_repair_round",
      protocol_version: "repair_v1",
      input_hash: hash,
      output_hash: hash,
      control_hash: hash,
      evaluator_linkage: null,
      repair_linkage: {
        action_id: "action-a",
        parent_action_id: "action-z",
        action_type: "repair_missing_citation",
        repair_protocol_version: "repair_v1",
        repair_round_index: 0,
        remaining_repair_budget_before: 1,
        remaining_repair_budget_after: 0,
        action_input_hash: hash,
        action_output_hash: hash,
        before_context_package_id: "package:1",
        repaired_context_package_id: "package:2",
        before_retrieval_trace_id: null,
        repaired_retrieval_trace_id: null,
      },
      evidence_chunk_ids: ["chunk:1"],
      verdict: "completed",
      observation: repairRoundPayload,
      diagnostics: payload('{"repair":true}'),
      redacted_fields: [],
      created_at: "2026-07-27T00:00:04Z",
    },
  ],
  redaction_protocol_version:
    "semantic_sensitive_field_key_segments_v1",
  provider_raw_response_exposed: false,
  credentials_exposed: false,
};

function renderAudit() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AgentPEAuditPanel runId="run:pe" />
    </QueryClientProvider>,
  );
}

describe("AgentPEAuditPanel", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("renders every persisted row in canonical API order with validator, evaluator, and repair linkage", async () => {
    vi.mocked(fetchAgentPEAudit).mockResolvedValue(audit);

    const { container } = renderAudit();

    await waitFor(() =>
      expect(fetchAgentPEAudit).toHaveBeenCalledWith("run:pe"),
    );
    expect(await screen.findByTestId("agent-pe-audit")).toBeTruthy();

    const rowIds = (testId: string) =>
      Array.from(container.querySelectorAll(`[data-testid="${testId}"]`)).map(
        (node) => node.getAttribute("data-row-id"),
      );
    expect(rowIds("agent-pe-plan-row")).toEqual(["plan-z"]);
    expect(rowIds("agent-pe-action-row")).toEqual(["action-z", "action-a"]);
    expect(rowIds("agent-pe-observation-row")).toEqual([
      "observation-z",
      "observation-a",
    ]);
    expect(
      screen.getAllByText(/validator: plan=true schema=true budget=true/),
    ).toHaveLength(2);
    expect(screen.getByText(/query_intent · sha256/)).toBeTruthy();
    expect(screen.getByText(/planner_model_metadata · sha256/)).toBeTruthy();
    expect(screen.getByText(/evaluator: need_structure_closure/)).toBeTruthy();
    expect(screen.getByText(/schema repair true/)).toBeTruthy();
    expect(screen.getByText(/repair\[0\] repair_missing_citation/)).toBeTruthy();
    expect(screen.getByText(/repaired_package:/)).toBeTruthy();
    expect(screen.getAllByText(hash).length).toBeGreaterThan(0);
  });

  it("fails closed on count or row-order drift", () => {
    expect(
      peAuditIntegrityIssue({
        ...audit,
        counts: { ...audit.counts, actions: 3 },
      }),
    ).not.toBeNull();
    expect(
      peAuditIntegrityIssue({
        ...audit,
        actions: [
          { ...audit.actions[0], order_index: 1 },
          { ...audit.actions[1], order_index: 0 },
        ],
      }),
    ).not.toBeNull();
    expect(
      peAuditIntegrityIssue({
        ...audit,
        observations: [
          { ...audit.observations[0], plan_id: "plan-cross-run" },
          audit.observations[1],
        ],
      }),
    ).not.toBeNull();
  });
});
