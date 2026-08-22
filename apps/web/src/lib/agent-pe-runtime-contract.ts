import type {
  AgentPEActionAuditRow,
  AgentPEActionType,
  AgentPEActionValidatorAudit,
  AgentPEAuditResponse,
  AgentPEEvaluatorLinkage,
  AgentPEJsonPayload,
  AgentPEObservationAuditRow,
  AgentPEObservationType,
  AgentPEPlanAuditRow,
  AgentPERepairLinkage,
} from "@course-kg/shared";


export const AGENT_PE_RUNTIME_CONTRACT_PROTOCOL_VERSION =
  "agent_pe_public_runtime_contract_v1" as const;
export const AGENT_PE_CROSS_FIELD_CONTRACT_PROTOCOL_VERSION =
  "agent_pe_public_cross_field_contract_v1" as const;

type RuntimeLiteral = string | number | boolean | null;

type RuntimeRule =
  | { kind: "array"; item: RuntimeRule }
  | { kind: "boolean" }
  | { kind: "integer"; minimum?: number }
  | { kind: "literal"; values: readonly RuntimeLiteral[] }
  | { kind: "nullable"; value: RuntimeRule }
  | { kind: "number" }
  | { kind: "object"; fields: Record<string, RuntimeField> }
  | { kind: "string"; format?: "datetime"; pattern?: RegExp };

interface RuntimeField {
  optional?: true;
  rule: RuntimeRule;
}

type RuntimeFieldMap<T extends object> = {
  [Key in keyof T]-?: RuntimeField;
};

const stringRule = { kind: "string" } as const satisfies RuntimeRule;
const booleanRule = { kind: "boolean" } as const satisfies RuntimeRule;
const nonNegativeIntegerRule = {
  kind: "integer",
  minimum: 0,
} as const satisfies RuntimeRule;
const datetimeRule = {
  kind: "string",
  format: "datetime",
} as const satisfies RuntimeRule;

function required(rule: RuntimeRule): RuntimeField {
  return { rule };
}

function optional(rule: RuntimeRule): RuntimeField {
  return { optional: true, rule };
}

function nullable(value: RuntimeRule): RuntimeRule {
  return { kind: "nullable", value };
}

function array(item: RuntimeRule): RuntimeRule {
  return { item, kind: "array" };
}

function literal(
  ...values: readonly RuntimeLiteral[]
): RuntimeRule {
  return { kind: "literal", values };
}

function closedObject<T extends object>(
  fields: RuntimeFieldMap<T>,
): RuntimeRule {
  return { fields, kind: "object" };
}

const nullableStringRule = nullable(stringRule);
const nullableBooleanRule = nullable(booleanRule);
const nullableNonNegativeIntegerRule = nullable(nonNegativeIntegerRule);
const stringArrayRule = array(stringRule);

const payloadRule = closedObject<AgentPEJsonPayload>({
  encoding: required(literal("canonical_json_v1")),
  canonical_json: required(stringRule),
  sha256: required({ kind: "string", pattern: /^[0-9a-f]{64}$/ }),
  redacted_fields: required(stringArrayRule),
});

const validatorRule = closedObject<AgentPEActionValidatorAudit>({
  valid: optional(nullableBooleanRule),
  plan_valid: optional(nullableBooleanRule),
  schema_checked: optional(nullableBooleanRule),
  budget_checked: optional(nullableBooleanRule),
  target_ids_checked: optional(nullableBooleanRule),
  target_scope_checked: optional(nullableBooleanRule),
  typed_action_schema_protocol_version: optional(nullableStringRule),
  typed_action_schema_protocol_hash: optional(nullableStringRule),
  repair_protocol_version: optional(nullableStringRule),
  repair_budget_checked: optional(nullableBooleanRule),
  repair_round_index: optional(nullableNonNegativeIntegerRule),
  remaining_repair_budget_before: optional(nullableNonNegativeIntegerRule),
  action_input_hash: optional(nullableStringRule),
  repair_directive_validator_protocol_version: optional(nullableStringRule),
  repair_directive_validator_result: optional(nullableStringRule),
  repair_directive_hash: optional(nullableStringRule),
  validated_directive_hash: optional(nullableStringRule),
  payload: required(payloadRule),
});

const evaluatorVerdicts = [
  "sufficient",
  "need_more_same_node",
  "need_bridge_jump",
  "need_mid_expansion",
  "need_chunk_expansion",
  "need_structure_closure",
  "insufficient_corpus",
] as const satisfies readonly AgentPEEvaluatorLinkage["verdict"][];

const evaluatorRule = closedObject<AgentPEEvaluatorLinkage>({
  plan_id: required(stringRule),
  plan_index: required(nonNegativeIntegerRule),
  protocol_version: optional(nullableStringRule),
  verdict: required(literal(...evaluatorVerdicts)),
  decision_hash: optional(nullableStringRule),
  replan_requested: required(booleanRule),
  schema_repair_attempted: optional(booleanRule),
  gray_zone_model_call_count: required(literal(0)),
});

const repairActionTypes = [
  "repair_missing_citation",
  "repair_concept_gap",
  "repair_bridge_gap",
  "repair_structure_context",
] as const satisfies readonly AgentPERepairLinkage["action_type"][];

const repairRule = closedObject<AgentPERepairLinkage>({
  action_id: required(stringRule),
  parent_action_id: optional(nullableStringRule),
  action_type: required(literal(...repairActionTypes)),
  repair_protocol_version: optional(nullableStringRule),
  repair_round_index: required(nonNegativeIntegerRule),
  remaining_repair_budget_before: required(nonNegativeIntegerRule),
  remaining_repair_budget_after: required(nonNegativeIntegerRule),
  action_input_hash: optional(nullableStringRule),
  action_output_hash: optional(nullableStringRule),
  before_context_package_id: optional(nullableStringRule),
  repaired_context_package_id: optional(nullableStringRule),
  before_retrieval_trace_id: optional(nullableStringRule),
  repaired_retrieval_trace_id: optional(nullableStringRule),
});

const planStatuses = [
  "validated",
  "invalid",
  "validator_replan_requested",
  "executor_contract_blocked",
  "replan_requested",
  "evidence_sufficient",
  "insufficient_corpus",
  "planning_budget_exhausted",
] as const satisfies readonly AgentPEPlanAuditRow["status"][];

const planRule = closedObject<AgentPEPlanAuditRow>({
  contract_version: required(literal("agent_plan_audit_row_v1")),
  order_index: required(nonNegativeIntegerRule),
  id: required(stringRule),
  run_id: required(stringRule),
  knowledge_base_id: required(stringRule),
  retrieval_trace_id: optional(nullableStringRule),
  plan_index: required(nonNegativeIntegerRule),
  planner_protocol_version: optional(nullableStringRule),
  typed_action_schema_protocol_version: optional(nullableStringRule),
  typed_action_schema_protocol_hash: optional(nullableStringRule),
  typed_action_executor_protocol_version: optional(nullableStringRule),
  input_hash: optional(nullableStringRule),
  output_hash: optional(nullableStringRule),
  control_hash: optional(nullableStringRule),
  query_intent: required(payloadRule),
  operating_envelope: required(payloadRule),
  typed_actions: required(payloadRule),
  validation: required(payloadRule),
  planner_model_metadata: required(payloadRule),
  status: required(literal(...planStatuses)),
  diagnostics: required(payloadRule),
  action_ids: required(stringArrayRule),
  action_count: required(nonNegativeIntegerRule),
  redacted_fields: required(stringArrayRule),
  created_at: required(datetimeRule),
});

const actionTypes = [
  "activate_coarse_concepts",
  "route_mid_concepts",
  "route_rq_addresses",
  "select_entry_nodes",
  "walk_graph_frontier",
  "drill_down_layer",
  "jump_bridge",
  "stop_and_collect_chunks",
  "need_more_evidence",
  "recall_chunks",
  "restore_context_package",
  "build_context_package",
  "verify_citations",
  ...repairActionTypes,
] as const satisfies readonly AgentPEActionType[];

const actionStatuses = [
  "accepted",
  "completed",
  "rejected",
  "deferred",
  "no_progress",
] as const satisfies readonly AgentPEActionAuditRow["status"][];

const actionRule = closedObject<AgentPEActionAuditRow>({
  contract_version: required(literal("agent_action_audit_row_v1")),
  order_index: required(nonNegativeIntegerRule),
  id: required(stringRule),
  run_id: required(stringRule),
  plan_id: required(stringRule),
  plan_index: required(nonNegativeIntegerRule),
  parent_action_id: optional(nullableStringRule),
  action_index: required(nonNegativeIntegerRule),
  action_type: required(literal(...actionTypes)),
  target_ids: required(stringArrayRule),
  reason: required(stringRule),
  budget_request: required(payloadRule),
  expected_evidence: required(payloadRule),
  stop_condition: required(payloadRule),
  validator: required(validatorRule),
  status: required(literal(...actionStatuses)),
  input_hash: optional(nullableStringRule),
  output_hash: optional(nullableStringRule),
  control_hash: optional(nullableStringRule),
  output: required(payloadRule),
  diagnostics: required(payloadRule),
  observation_ids: required(stringArrayRule),
  observation_count: required(nonNegativeIntegerRule),
  redacted_fields: required(stringArrayRule),
  created_at: required(datetimeRule),
});

const observationTypes = [
  "plan_validation_failed",
  "executor_contract_blocked",
  "entry_selection",
  "layer_routing",
  "frontier_traversal",
  "chunk_recall",
  "repair_gate",
  "evidence_evaluator",
  "replan_gate",
  "evidence_gate_blocked",
  "context_restoration",
  "context_package_built",
  "citation_verification",
  "typed_repair_round",
  "claim_level_final_grounded_gate",
] as const satisfies readonly AgentPEObservationType[];

const observationRule = closedObject<AgentPEObservationAuditRow>({
  contract_version: required(literal("agent_observation_audit_row_v1")),
  order_index: required(nonNegativeIntegerRule),
  id: required(stringRule),
  run_id: required(stringRule),
  plan_id: required(stringRule),
  plan_index: required(nonNegativeIntegerRule),
  action_id: optional(nullableStringRule),
  action_index: optional(nullableNonNegativeIntegerRule),
  parent_action_id: optional(nullableStringRule),
  observation_type: required(literal(...observationTypes)),
  protocol_version: optional(nullableStringRule),
  input_hash: optional(nullableStringRule),
  output_hash: optional(nullableStringRule),
  control_hash: optional(nullableStringRule),
  evaluator_linkage: optional(nullable(evaluatorRule)),
  repair_linkage: optional(nullable(repairRule)),
  evidence_chunk_ids: required(stringArrayRule),
  verdict: required(stringRule),
  observation: required(payloadRule),
  diagnostics: required(payloadRule),
  redacted_fields: required(stringArrayRule),
  created_at: required(datetimeRule),
});

const countsRule = closedObject<AgentPEAuditResponse["counts"]>({
  plans: required(nonNegativeIntegerRule),
  actions: required(nonNegativeIntegerRule),
  observations: required(nonNegativeIntegerRule),
});

const orderingRule = closedObject<AgentPEAuditResponse["ordering"]>({
  plans: required(literal("plan_index ASC, created_at ASC, id ASC")),
  actions: required(
    literal(
      "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC",
    ),
  ),
  observations: required(literal("created_at ASC, id ASC")),
});

const auditRule = closedObject<AgentPEAuditResponse>({
  contract_version: required(literal("agent_pe_audit_public_v1")),
  run_id: required(stringRule),
  knowledge_base_id: required(stringRule),
  run_status: required(stringRule),
  counts: required(countsRule),
  ordering: required(orderingRule),
  plans: required(array(planRule)),
  actions: required(array(actionRule)),
  observations: required(array(observationRule)),
  redaction_protocol_version: required(
    literal("semantic_sensitive_field_key_segments_v1"),
  ),
  provider_raw_response_exposed: required(literal(false)),
  credentials_exposed: required(literal(false)),
});

function plainObject(value: unknown): Record<string, unknown> | null {
  if (!value || Array.isArray(value) || typeof value !== "object") {
    return null;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    return null;
  }
  return value as Record<string, unknown>;
}

function datetimeIssue(value: string): string | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(?:Z|([+-])(\d{2}):(\d{2}))?$/.exec(
    value,
  );
  if (!match) return "expected an ISO datetime string";
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);
  const second = Number(match[6]);
  const offsetHour = match[9] === undefined ? 0 : Number(match[9]);
  const offsetMinute = match[10] === undefined ? 0 : Number(match[10]);
  const daysInMonth =
    month >= 1 && month <= 12
      ? new Date(Date.UTC(year, month, 0)).getUTCDate()
      : 0;
  if (
    year < 1 ||
    day < 1 ||
    day > daysInMonth ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    offsetHour > 23 ||
    offsetMinute > 59
  ) {
    return "expected a valid ISO datetime string";
  }
  return null;
}

function datetimeInstantMicroseconds(value: string): bigint | null {
  if (datetimeIssue(value)) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(?:Z|([+-])(\d{2}):(\d{2}))?$/.exec(
    value,
  );
  if (!match) return null;
  const date = new Date(0);
  date.setUTCFullYear(
    Number(match[1]),
    Number(match[2]) - 1,
    Number(match[3]),
  );
  date.setUTCHours(
    Number(match[4]),
    Number(match[5]),
    Number(match[6]),
    0,
  );
  const epochMilliseconds = date.getTime();
  if (!Number.isFinite(epochMilliseconds)) return null;
  const fractionMicroseconds = BigInt(
    (match[7] ?? "").padEnd(6, "0") || "0",
  );
  const offsetMinutes =
    match[8] === undefined
      ? 0
      : (match[8] === "+" ? 1 : -1) *
        (Number(match[9]) * 60 + Number(match[10]));
  return (
    BigInt(epochMilliseconds) * BigInt(1_000) +
    fractionMicroseconds -
    BigInt(offsetMinutes) * BigInt(60_000_000)
  );
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareDatetimes(left: string, right: string): number | null {
  const leftInstant = datetimeInstantMicroseconds(left);
  const rightInstant = datetimeInstantMicroseconds(right);
  if (leftInstant === null || rightInstant === null) return null;
  return leftInstant < rightInstant ? -1 : leftInstant > rightInstant ? 1 : 0;
}

function canonicalRowsIssue<T>(
  rows: readonly T[],
  path: string,
  compare: (left: T, right: T) => number | null,
): string | null {
  const prefix = AGENT_PE_CROSS_FIELD_CONTRACT_PROTOCOL_VERSION;
  for (let index = 1; index < rows.length; index += 1) {
    const ordering = compare(rows[index - 1], rows[index]);
    if (ordering === null) {
      return `${prefix}: ${path} contains an unparseable datetime`;
    }
    if (ordering > 0) {
      return `${prefix}: ${path} contradicts canonical database ordering at index ${index}`;
    }
  }
  return null;
}

export function agentPEAuditCrossFieldContractIssue(
  audit: AgentPEAuditResponse,
): string | null {
  let issue = canonicalRowsIssue(audit.plans, "audit.plans", (left, right) => {
    if (left.plan_index !== right.plan_index) {
      return left.plan_index < right.plan_index ? -1 : 1;
    }
    const instantOrder = compareDatetimes(left.created_at, right.created_at);
    return instantOrder === 0
      ? compareStrings(left.id, right.id)
      : instantOrder;
  });
  if (issue) return issue;
  issue = canonicalRowsIssue(audit.actions, "audit.actions", (left, right) => {
    if (left.plan_index !== right.plan_index) {
      return left.plan_index < right.plan_index ? -1 : 1;
    }
    if (left.action_index !== right.action_index) {
      return left.action_index < right.action_index ? -1 : 1;
    }
    const instantOrder = compareDatetimes(left.created_at, right.created_at);
    return instantOrder === 0
      ? compareStrings(left.id, right.id)
      : instantOrder;
  });
  if (issue) return issue;
  issue = canonicalRowsIssue(
    audit.observations,
    "audit.observations",
    (left, right) => {
      const instantOrder = compareDatetimes(left.created_at, right.created_at);
      return instantOrder === 0
        ? compareStrings(left.id, right.id)
        : instantOrder;
    },
  );
  if (issue) return issue;

  for (const [index, observation] of audit.observations.entries()) {
    if (observation.observation_type !== "evidence_evaluator") continue;
    const decisionHash = observation.evaluator_linkage?.decision_hash ?? null;
    const observationHash = observation.output_hash ?? null;
    if (decisionHash !== observationHash) {
      return `${AGENT_PE_CROSS_FIELD_CONTRACT_PROTOCOL_VERSION}: audit.observations[${index}] evaluator decision/output hash lineage conflicts`;
    }
  }
  return null;
}

function runtimeContractIssue(
  rule: RuntimeRule,
  value: unknown,
  path: string,
): string | null {
  const prefix = AGENT_PE_RUNTIME_CONTRACT_PROTOCOL_VERSION;
  switch (rule.kind) {
    case "array": {
      if (!Array.isArray(value)) {
        return `${prefix}: ${path} expected array`;
      }
      for (const [index, item] of value.entries()) {
        const issue = runtimeContractIssue(
          rule.item,
          item,
          `${path}[${index}]`,
        );
        if (issue) return issue;
      }
      return null;
    }
    case "boolean":
      return typeof value === "boolean"
        ? null
        : `${prefix}: ${path} expected boolean`;
    case "integer":
      if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        !Number.isInteger(value)
      ) {
        return `${prefix}: ${path} expected finite integer`;
      }
      return rule.minimum !== undefined && value < rule.minimum
        ? `${prefix}: ${path} expected integer >= ${rule.minimum}`
        : null;
    case "literal":
      return rule.values.some((expected) => Object.is(value, expected))
        ? null
        : `${prefix}: ${path} expected literal ${rule.values
            .map((item) => JSON.stringify(item))
            .join(" | ")}`;
    case "nullable":
      return value === null
        ? null
        : runtimeContractIssue(rule.value, value, path);
    case "number":
      return typeof value === "number" && Number.isFinite(value)
        ? null
        : `${prefix}: ${path} expected finite number`;
    case "object": {
      const object = plainObject(value);
      if (!object) return `${prefix}: ${path} is not a closed object`;
      const keys = Object.keys(object);
      const expectedKeys = Object.keys(rule.fields);
      const missing = expectedKeys.filter(
        (key) =>
          !rule.fields[key].optional &&
          !Object.prototype.hasOwnProperty.call(object, key),
      );
      const extra = keys.filter(
        (key) => !Object.prototype.hasOwnProperty.call(rule.fields, key),
      );
      if (missing.length > 0 || extra.length > 0) {
        return `${prefix}: ${path} closed fields mismatch (missing=${
          missing.join(",") || "none"
        }; extra=${extra.join(",") || "none"})`;
      }
      for (const key of expectedKeys) {
        if (!Object.prototype.hasOwnProperty.call(object, key)) continue;
        const issue = runtimeContractIssue(
          rule.fields[key].rule,
          object[key],
          `${path}.${key}`,
        );
        if (issue) return issue;
      }
      return null;
    }
    case "string":
      if (typeof value !== "string") {
        return `${prefix}: ${path} expected string`;
      }
      if (rule.pattern && !rule.pattern.test(value)) {
        return `${prefix}: ${path} string pattern mismatch`;
      }
      if (rule.format === "datetime") {
        const issue = datetimeIssue(value);
        return issue ? `${prefix}: ${path} ${issue}` : null;
      }
      return null;
  }
}

export function agentPEAuditRuntimeContractIssue(
  value: unknown,
): string | null {
  return runtimeContractIssue(auditRule, value, "audit");
}
