import type {
  AgentPEActionAuditRow,
  AgentPEActionValidatorAudit,
  AgentPEAuditResponse,
  AgentPEEvaluatorLinkage,
  AgentPEJsonPayload,
  AgentPEObservationAuditRow,
  AgentPEPlanAuditRow,
  AgentPERepairLinkage,
} from "@course-kg/shared";
import { describe, expect, it } from "vitest";

import {
  AGENT_PE_RUNTIME_CONTRACT_PROTOCOL_VERSION,
  agentPEAuditRuntimeContractIssue,
} from "./agent-pe-runtime-contract";


const emptyPayloadHash =
  "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a";
const sentinel = "RUNTIME_CONTRACT_SECRET_SENTINEL";

function payload(): AgentPEJsonPayload {
  return {
    encoding: "canonical_json_v1",
    canonical_json: "{}",
    sha256: emptyPayloadHash,
    redacted_fields: [],
  };
}

function validAudit(): AgentPEAuditResponse {
  const plan: AgentPEPlanAuditRow = {
    contract_version: "agent_plan_audit_row_v1",
    order_index: 0,
    id: "plan:1",
    run_id: "run:1",
    knowledge_base_id: "kb:1",
    retrieval_trace_id: null,
    plan_index: 0,
    planner_protocol_version: "planner_v1",
    typed_action_schema_protocol_version: null,
    typed_action_schema_protocol_hash: null,
    typed_action_executor_protocol_version: "executor_v1",
    input_hash: null,
    output_hash: null,
    control_hash: null,
    query_intent: payload(),
    operating_envelope: payload(),
    typed_actions: payload(),
    validation: payload(),
    planner_model_metadata: payload(),
    status: "validated",
    diagnostics: payload(),
    action_ids: ["action:1"],
    action_count: 1,
    redacted_fields: [],
    created_at: "2026-08-05T00:00:00Z",
  };
  const validator: AgentPEActionValidatorAudit = {
    valid: true,
    plan_valid: null,
    schema_checked: false,
    budget_checked: true,
    target_ids_checked: null,
    target_scope_checked: true,
    typed_action_schema_protocol_version: null,
    typed_action_schema_protocol_hash: "schema-hash",
    repair_protocol_version: "repair_v1",
    repair_budget_checked: true,
    repair_round_index: 0,
    remaining_repair_budget_before: 1,
    action_input_hash: null,
    repair_directive_validator_protocol_version: null,
    repair_directive_validator_result: "accepted",
    repair_directive_hash: null,
    validated_directive_hash: "directive-hash",
    payload: payload(),
  };
  const action: AgentPEActionAuditRow = {
    contract_version: "agent_action_audit_row_v1",
    order_index: 0,
    id: "action:1",
    run_id: "run:1",
    plan_id: "plan:1",
    plan_index: 0,
    parent_action_id: null,
    action_index: 0,
    action_type: "repair_missing_citation",
    target_ids: ["chunk:1"],
    reason: "repair",
    budget_request: payload(),
    expected_evidence: payload(),
    stop_condition: payload(),
    validator,
    status: "completed",
    input_hash: null,
    output_hash: null,
    control_hash: null,
    output: payload(),
    diagnostics: payload(),
    observation_ids: ["observation:repair"],
    observation_count: 1,
    redacted_fields: [],
    created_at: "2026-08-05T00:00:01+08:00",
  };
  const evaluator: AgentPEEvaluatorLinkage = {
    plan_id: "plan:1",
    plan_index: 0,
    protocol_version: null,
    verdict: "sufficient",
    decision_hash: null,
    replan_requested: false,
    schema_repair_attempted: true,
    gray_zone_model_call_count: 0,
  };
  const repair: AgentPERepairLinkage = {
    action_id: "action:1",
    parent_action_id: null,
    action_type: "repair_missing_citation",
    repair_protocol_version: "repair_v1",
    repair_round_index: 0,
    remaining_repair_budget_before: 1,
    remaining_repair_budget_after: 0,
    action_input_hash: null,
    action_output_hash: null,
    before_context_package_id: null,
    repaired_context_package_id: "package:2",
    before_retrieval_trace_id: null,
    repaired_retrieval_trace_id: "trace:2",
  };
  const evaluatorObservation: AgentPEObservationAuditRow = {
    contract_version: "agent_observation_audit_row_v1",
    order_index: 0,
    id: "observation:evaluator",
    run_id: "run:1",
    plan_id: "plan:1",
    plan_index: 0,
    action_id: null,
    action_index: null,
    parent_action_id: null,
    observation_type: "evidence_evaluator",
    protocol_version: null,
    input_hash: null,
    output_hash: null,
    control_hash: null,
    evaluator_linkage: evaluator,
    repair_linkage: null,
    evidence_chunk_ids: ["chunk:1"],
    verdict: "sufficient",
    observation: payload(),
    diagnostics: payload(),
    redacted_fields: [],
    created_at: "2026-08-05T00:00:02",
  };
  const repairObservation: AgentPEObservationAuditRow = {
    ...evaluatorObservation,
    order_index: 1,
    id: "observation:repair",
    action_id: "action:1",
    action_index: 0,
    observation_type: "typed_repair_round",
    protocol_version: "repair_v1",
    evaluator_linkage: null,
    repair_linkage: repair,
    verdict: "completed",
    created_at: "2026-08-05T00:00:03.123456Z",
  };
  return {
    contract_version: "agent_pe_audit_public_v1",
    run_id: "run:1",
    knowledge_base_id: "kb:1",
    run_status: "completed",
    counts: { plans: 1, actions: 1, observations: 2 },
    ordering: {
      plans: "plan_index ASC, created_at ASC, id ASC",
      actions:
        "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC",
      observations: "created_at ASC, id ASC",
    },
    plans: [plan],
    actions: [action],
    observations: [evaluatorObservation, repairObservation],
    redaction_protocol_version: "semantic_sensitive_field_key_segments_v1",
    provider_raw_response_exposed: false,
    credentials_exposed: false,
  };
}

function overwrite(
  value: object,
  key: PropertyKey,
  replacement: unknown,
): void {
  (value as Record<PropertyKey, unknown>)[key] = replacement;
}

function expectRejected(
  mutate: (audit: AgentPEAuditResponse) => void,
  path: string,
): void {
  const audit = structuredClone(validAudit());
  mutate(audit);
  const issue = agentPEAuditRuntimeContractIssue(audit);
  expect(issue).toContain(AGENT_PE_RUNTIME_CONTRACT_PROTOCOL_VERSION);
  expect(issue).toContain(path);
}

describe("agent_pe_public_runtime_contract_v1", () => {
  it("accepts the full shared/Pydantic-shaped control", () => {
    expect(AGENT_PE_RUNTIME_CONTRACT_PROTOCOL_VERSION).toBe(
      "agent_pe_public_runtime_contract_v1",
    );
    expect(agentPEAuditRuntimeContractIssue(validAudit())).toBeNull();
  });

  it.each(["run_id", "knowledge_base_id", "run_status"])(
    "rejects a nested object in audit.%s before payload trust",
    (key) => {
      expectRejected(
        (audit) => overwrite(audit, key, { providerResponseBlob: sentinel }),
        `audit.${key}`,
      );
    },
  );

  it.each(["plans", "actions", "observations"])(
    "rejects non-array audit.%s",
    (key) => {
      expectRejected((audit) => overwrite(audit, key, {}), `audit.${key}`);
    },
  );

  it.each(["plans", "actions", "observations"])(
    "rejects non-negative finite integer drift in audit.counts.%s",
    (key) => {
      expectRejected(
        (audit) => overwrite(audit.counts, key, Number.POSITIVE_INFINITY),
        `audit.counts.${key}`,
      );
    },
  );

  it("rejects every top-level literal/closed-contract drift", () => {
    const attacks: Array<[
      string,
      (audit: AgentPEAuditResponse) => void,
    ]> = [
      ["audit.contract_version", (audit) => overwrite(audit, "contract_version", "v0")],
      ["audit.ordering.plans", (audit) => overwrite(audit.ordering, "plans", "id DESC")],
      ["audit.redaction_protocol_version", (audit) => overwrite(audit, "redaction_protocol_version", "legacy")],
      ["audit.provider_raw_response_exposed", (audit) => overwrite(audit, "provider_raw_response_exposed", true)],
      ["audit.credentials_exposed", (audit) => overwrite(audit, "credentials_exposed", 0)],
    ];
    for (const [path, mutate] of attacks) expectRejected(mutate, path);
  });

  it("rejects missing, extra, and present-undefined fields", () => {
    expectRejected(
      (audit) => delete (audit as unknown as Record<string, unknown>).run_status,
      "audit closed fields mismatch",
    );
    expectRejected(
      (audit) => overwrite(audit, "providerResponseBlob", sentinel),
      "audit closed fields mismatch",
    );
    expectRejected(
      (audit) => overwrite(audit.plans[0], "retrieval_trace_id", undefined),
      "audit.plans[0].retrieval_trace_id",
    );
  });

  it.each([
    "id",
    "run_id",
    "knowledge_base_id",
    "retrieval_trace_id",
    "planner_protocol_version",
    "typed_action_schema_protocol_version",
    "typed_action_schema_protocol_hash",
    "typed_action_executor_protocol_version",
    "input_hash",
    "output_hash",
    "control_hash",
  ])("rejects non-string/null plan field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.plans[0], key, {}),
      `audit.plans[0].${key}`,
    );
  });

  it.each(["order_index", "plan_index", "action_count"])(
    "rejects non-integer plan field %s",
    (key) => {
      expectRejected(
        (audit) => overwrite(audit.plans[0], key, 0.5),
        `audit.plans[0].${key}`,
      );
    },
  );

  it.each([
    "query_intent",
    "operating_envelope",
    "typed_actions",
    "validation",
    "planner_model_metadata",
    "diagnostics",
  ])("rejects non-payload plan field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.plans[0], key, "{}"),
      `audit.plans[0].${key}`,
    );
  });

  it("rejects plan literals, typed arrays, and datetime drift", () => {
    const attacks: Array<[string, (plan: AgentPEPlanAuditRow) => void]> = [
      ["contract_version", (plan) => overwrite(plan, "contract_version", "v0")],
      ["status", (plan) => overwrite(plan, "status", "completed")],
      ["action_ids[0]", (plan) => overwrite(plan, "action_ids", [1])],
      ["redacted_fields[0]", (plan) => overwrite(plan, "redacted_fields", [{}])],
      ["created_at", (plan) => overwrite(plan, "created_at", "2026-99-99")],
    ];
    for (const [suffix, mutate] of attacks) {
      expectRejected(
        (audit) => mutate(audit.plans[0]),
        `audit.plans[0].${suffix}`,
      );
    }
  });

  it.each([
    "id",
    "run_id",
    "plan_id",
    "parent_action_id",
    "reason",
    "input_hash",
    "output_hash",
    "control_hash",
  ])("rejects non-string/null action field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.actions[0], key, []),
      `audit.actions[0].${key}`,
    );
  });

  it.each([
    "order_index",
    "plan_index",
    "action_index",
    "observation_count",
  ])("rejects non-integer action field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.actions[0], key, Number.NaN),
      `audit.actions[0].${key}`,
    );
  });

  it.each([
    "budget_request",
    "expected_evidence",
    "stop_condition",
    "output",
    "diagnostics",
  ])("rejects non-payload action field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.actions[0], key, []),
      `audit.actions[0].${key}`,
    );
  });

  it("rejects action literals, objects, typed arrays, and datetime drift", () => {
    const attacks: Array<[string, (action: AgentPEActionAuditRow) => void]> = [
      ["contract_version", (action) => overwrite(action, "contract_version", "v0")],
      ["action_type", (action) => overwrite(action, "action_type", "freeform")],
      ["status", (action) => overwrite(action, "status", "validated")],
      ["target_ids[0]", (action) => overwrite(action, "target_ids", [{}])],
      ["validator", (action) => overwrite(action, "validator", "valid")],
      ["observation_ids[0]", (action) => overwrite(action, "observation_ids", [1])],
      ["redacted_fields[0]", (action) => overwrite(action, "redacted_fields", [false])],
      ["created_at", (action) => overwrite(action, "created_at", {})],
    ];
    for (const [suffix, mutate] of attacks) {
      expectRejected(
        (audit) => mutate(audit.actions[0]),
        `audit.actions[0].${suffix}`,
      );
    }
  });

  it.each([
    "valid",
    "plan_valid",
    "schema_checked",
    "budget_checked",
    "target_ids_checked",
    "target_scope_checked",
    "repair_budget_checked",
  ])("rejects non-boolean/null validator field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.actions[0].validator, key, 0),
      `audit.actions[0].validator.${key}`,
    );
  });

  it.each([
    "typed_action_schema_protocol_version",
    "typed_action_schema_protocol_hash",
    "repair_protocol_version",
    "action_input_hash",
    "repair_directive_validator_protocol_version",
    "repair_directive_validator_result",
    "repair_directive_hash",
    "validated_directive_hash",
  ])("rejects non-string/null validator field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.actions[0].validator, key, {}),
      `audit.actions[0].validator.${key}`,
    );
  });

  it.each(["repair_round_index", "remaining_repair_budget_before"])(
    "rejects non-integer/null validator field %s",
    (key) => {
      expectRejected(
        (audit) => overwrite(audit.actions[0].validator, key, -1),
        `audit.actions[0].validator.${key}`,
      );
    },
  );

  it("rejects a non-payload validator payload", () => {
    expectRejected(
      (audit) => overwrite(audit.actions[0].validator, "payload", null),
      "audit.actions[0].validator.payload",
    );
  });

  it.each([
    "id",
    "run_id",
    "plan_id",
    "action_id",
    "parent_action_id",
    "protocol_version",
    "input_hash",
    "output_hash",
    "control_hash",
    "verdict",
  ])("rejects non-string/null observation field %s", (key) => {
    expectRejected(
      (audit) => overwrite(audit.observations[0], key, {}),
      `audit.observations[0].${key}`,
    );
  });

  it.each(["order_index", "plan_index", "action_index"])(
    "rejects non-integer/null observation field %s",
    (key) => {
      expectRejected(
        (audit) => overwrite(audit.observations[0], key, "0"),
        `audit.observations[0].${key}`,
      );
    },
  );

  it("rejects observation literals, linkages, typed arrays, payloads, and datetime drift", () => {
    const attacks: Array<[
      string,
      (observation: AgentPEObservationAuditRow) => void,
    ]> = [
      ["contract_version", (row) => overwrite(row, "contract_version", "v0")],
      ["observation_type", (row) => overwrite(row, "observation_type", "freeform")],
      ["evaluator_linkage", (row) => overwrite(row, "evaluator_linkage", [])],
      ["repair_linkage", (row) => overwrite(row, "repair_linkage", "repair")],
      ["evidence_chunk_ids[0]", (row) => overwrite(row, "evidence_chunk_ids", [1])],
      ["observation", (row) => overwrite(row, "observation", "{}")],
      ["diagnostics", (row) => overwrite(row, "diagnostics", 1)],
      ["redacted_fields[0]", (row) => overwrite(row, "redacted_fields", [null])],
      ["created_at", (row) => overwrite(row, "created_at", "2026-02-30T00:00:00Z")],
    ];
    for (const [suffix, mutate] of attacks) {
      expectRejected(
        (audit) => mutate(audit.observations[0]),
        `audit.observations[0].${suffix}`,
      );
    }
  });

  it("rejects every evaluator scalar/literal drift", () => {
    const attacks: Array<[
      string,
      (linkage: AgentPEEvaluatorLinkage) => void,
    ]> = [
      ["plan_id", (row) => overwrite(row, "plan_id", {})],
      ["plan_index", (row) => overwrite(row, "plan_index", 0.25)],
      ["protocol_version", (row) => overwrite(row, "protocol_version", [])],
      ["verdict", (row) => overwrite(row, "verdict", "maybe")],
      ["decision_hash", (row) => overwrite(row, "decision_hash", 1)],
      ["replan_requested", (row) => overwrite(row, "replan_requested", 0)],
      ["schema_repair_attempted", (row) => overwrite(row, "schema_repair_attempted", "yes")],
      ["gray_zone_model_call_count", (row) => overwrite(row, "gray_zone_model_call_count", 1)],
    ];
    for (const [suffix, mutate] of attacks) {
      expectRejected(
        (audit) =>
          mutate(
            audit.observations[0]
              .evaluator_linkage as AgentPEEvaluatorLinkage,
          ),
        `audit.observations[0].evaluator_linkage.${suffix}`,
      );
    }
  });

  it.each([
    "action_id",
    "parent_action_id",
    "repair_protocol_version",
    "action_input_hash",
    "action_output_hash",
    "before_context_package_id",
    "repaired_context_package_id",
    "before_retrieval_trace_id",
    "repaired_retrieval_trace_id",
  ])("rejects non-string/null repair field %s", (key) => {
    expectRejected(
      (audit) =>
        overwrite(
          audit.observations[1].repair_linkage as AgentPERepairLinkage,
          key,
          {},
        ),
      `audit.observations[1].repair_linkage.${key}`,
    );
  });

  it.each([
    "repair_round_index",
    "remaining_repair_budget_before",
    "remaining_repair_budget_after",
  ])("rejects non-integer repair field %s", (key) => {
    expectRejected(
      (audit) =>
        overwrite(
          audit.observations[1].repair_linkage as AgentPERepairLinkage,
          key,
          Number.NEGATIVE_INFINITY,
        ),
      `audit.observations[1].repair_linkage.${key}`,
    );
  });

  it("rejects an unknown repair action literal", () => {
    expectRejected(
      (audit) =>
        overwrite(
          audit.observations[1].repair_linkage as AgentPERepairLinkage,
          "action_type",
          "repair_anything",
        ),
      "audit.observations[1].repair_linkage.action_type",
    );
  });

  it("rejects every payload envelope scalar/array drift before canonical replay", () => {
    const attacks: Array<[
      string,
      (payloadValue: AgentPEJsonPayload) => void,
    ]> = [
      ["encoding", (value) => overwrite(value, "encoding", "json")],
      ["canonical_json", (value) => overwrite(value, "canonical_json", {})],
      ["sha256", (value) => overwrite(value, "sha256", "A".repeat(64))],
      ["redacted_fields", (value) => overwrite(value, "redacted_fields", [1])],
    ];
    for (const [suffix, mutate] of attacks) {
      expectRejected(
        (audit) => mutate(audit.plans[0].diagnostics),
        `audit.plans[0].diagnostics.${suffix}`,
      );
    }
  });
});
