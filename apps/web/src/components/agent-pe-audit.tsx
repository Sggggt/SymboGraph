"use client";

import { useQuery } from "@tanstack/react-query";
import type {
  AgentPEEvaluatorLinkage,
  AgentPEAuditResponse,
  AgentPEJsonPayload,
  AgentPEObservationAuditRow,
  AgentPERepairLinkage,
} from "@course-kg/shared";
import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";

import { fetchAgentPEAudit } from "@/lib/api";
import {
  agentPEAuditCrossFieldContractIssue,
  agentPEAuditRuntimeContractIssue,
} from "@/lib/agent-pe-runtime-contract";
import {
  SENSITIVE_FIELD_KEY_PROTOCOL_VERSION,
  peSensitiveKeyKind,
} from "@/lib/sensitive-fields";

interface AgentPEAuditPanelProps {
  runId: string | null;
  isRunning?: boolean;
}

const sha256Pattern = /^[0-9a-f]{64}$/;
const redactedValue = "[REDACTED]";
const repairActionTypes = new Set([
  "repair_missing_citation",
  "repair_concept_gap",
  "repair_bridge_gap",
  "repair_structure_context",
]);
function rotateRight(value: number, shift: number): number {
  return (value >>> shift) | (value << (32 - shift));
}

function sha256Utf8(value: string): string {
  const constants = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
    0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
    0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
    0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  const encoded = new TextEncoder().encode(value);
  const bitLength = encoded.length * 8;
  const paddedLength = Math.ceil((encoded.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(encoded);
  padded[encoded.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000));
  view.setUint32(paddedLength - 4, bitLength >>> 0);

  const hash = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4);
    }
    for (let index = 16; index < 64; index += 1) {
      const left = words[index - 15];
      const right = words[index - 2];
      const sigma0 =
        rotateRight(left, 7) ^ rotateRight(left, 18) ^ (left >>> 3);
      const sigma1 =
        rotateRight(right, 17) ^ rotateRight(right, 19) ^ (right >>> 10);
      words[index] =
        (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
    }

    let [a, b, c, d, e, f, g, h] = hash;
    for (let index = 0; index < 64; index += 1) {
      const sum1 =
        rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choice = (e & f) ^ (~e & g);
      const temporary1 =
        (h + sum1 + choice + constants[index] + words[index]) >>> 0;
      const sum0 =
        rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    hash[0] = (hash[0] + a) >>> 0;
    hash[1] = (hash[1] + b) >>> 0;
    hash[2] = (hash[2] + c) >>> 0;
    hash[3] = (hash[3] + d) >>> 0;
    hash[4] = (hash[4] + e) >>> 0;
    hash[5] = (hash[5] + f) >>> 0;
    hash[6] = (hash[6] + g) >>> 0;
    hash[7] = (hash[7] + h) >>> 0;
  }
  return Array.from(hash)
    .map((part) => part.toString(16).padStart(8, "0"))
    .join("");
}

function compareCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left, (value) => value.codePointAt(0) ?? 0);
  const rightPoints = Array.from(right, (value) => value.codePointAt(0) ?? 0);
  for (
    let index = 0;
    index < Math.min(leftPoints.length, rightPoints.length);
    index += 1
  ) {
    if (leftPoints[index] !== rightPoints[index]) {
      return leftPoints[index] - rightPoints[index];
    }
  }
  return leftPoints.length - rightPoints.length;
}

function pythonFloatRepr(value: number): string | null {
  if (!Number.isFinite(value)) return null;
  if (Object.is(value, -0)) return "-0.0";
  if (value === 0) return "0.0";

  const sign = value < 0 ? "-" : "";
  const source = Math.abs(value).toString().toLowerCase();
  let digits: string;
  let exponent: number;
  if (source.includes("e")) {
    const [mantissa, exponentText] = source.split("e");
    digits = mantissa.replace(".", "").replace(/^0+/, "").replace(/0+$/, "");
    exponent = Number.parseInt(exponentText, 10);
  } else {
    const [integerPart, fractionPart = ""] = source.split(".");
    const combined = `${integerPart}${fractionPart}`;
    const firstNonZero = combined.search(/[1-9]/);
    if (firstNonZero < 0) return `${sign}0.0`;
    exponent = integerPart.length - firstNonZero - 1;
    digits = combined.slice(firstNonZero).replace(/0+$/, "");
  }

  if (exponent < -4 || exponent >= 16) {
    const mantissa =
      digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
    const exponentSign = exponent >= 0 ? "+" : "-";
    const exponentDigits = Math.abs(exponent).toString().padStart(2, "0");
    return `${sign}${mantissa}e${exponentSign}${exponentDigits}`;
  }

  let fixed: string;
  if (exponent >= 0) {
    const integerLength = exponent + 1;
    fixed =
      digits.length <= integerLength
        ? `${digits}${"0".repeat(integerLength - digits.length)}`
        : `${digits.slice(0, integerLength)}.${digits.slice(integerLength)}`;
    if (!fixed.includes(".")) fixed = `${fixed}.0`;
  } else {
    fixed = `0.${"0".repeat(-exponent - 1)}${digits}`;
  }
  return `${sign}${fixed}`;
}

function canonicalPythonNumberToken(token: string): boolean {
  if (/^-?(?:0|[1-9]\d*)$/.test(token)) {
    return token !== "-0";
  }
  if (
    !/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(token) ||
    (!token.includes(".") && !/[eE]/.test(token))
  ) {
    return false;
  }
  const value = Number(token);
  return pythonFloatRepr(value) === token;
}

function canonicalJsonStructureIssue(source: string): string | null {
  let cursor = 0;
  let issue: string | null = null;

  const parseString = (): string | null => {
    if (source[cursor] !== '"') return null;
    const start = cursor;
    cursor += 1;
    while (cursor < source.length) {
      const character = source[cursor];
      if (character === '"') {
        cursor += 1;
        const token = source.slice(start, cursor);
        try {
          const decoded = JSON.parse(token);
          if (
            typeof decoded !== "string" ||
            JSON.stringify(decoded) !== token
          ) {
            issue = "string escaping is not canonical";
            return null;
          }
          return decoded;
        } catch {
          issue = "string is not valid JSON";
          return null;
        }
      }
      if (character === "\\") {
        cursor += 2;
      } else {
        if (character.charCodeAt(0) < 0x20) {
          issue = "unescaped control character";
          return null;
        }
        cursor += 1;
      }
    }
    issue = "unterminated string";
    return null;
  };

  const parseNumber = (): boolean => {
    const match = source
      .slice(cursor)
      .match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
    if (!match) return false;
    const token = match[0];
    if (!canonicalPythonNumberToken(token)) {
      issue = "number is not in canonical Python json.dumps form";
      return false;
    }
    cursor += token.length;
    return true;
  };

  const parseValue = (): boolean => {
    if (cursor >= source.length) return false;
    if (source[cursor] === '"') return parseString() !== null;
    if (source.startsWith("null", cursor)) {
      cursor += 4;
      return true;
    }
    if (source.startsWith("true", cursor)) {
      cursor += 4;
      return true;
    }
    if (source.startsWith("false", cursor)) {
      cursor += 5;
      return true;
    }
    if (source[cursor] === "[") {
      cursor += 1;
      if (source[cursor] === "]") {
        cursor += 1;
        return true;
      }
      while (cursor < source.length) {
        if (!parseValue()) return false;
        if (source[cursor] === "]") {
          cursor += 1;
          return true;
        }
        if (source[cursor] !== ",") return false;
        cursor += 1;
      }
      return false;
    }
    if (source[cursor] === "{") {
      cursor += 1;
      if (source[cursor] === "}") {
        cursor += 1;
        return true;
      }
      let previousKey: string | null = null;
      while (cursor < source.length) {
        const key = parseString();
        if (key === null) return false;
        if (
          previousKey !== null &&
          compareCodePoints(previousKey, key) >= 0
        ) {
          issue = "object keys are duplicated or not sorted";
          return false;
        }
        previousKey = key;
        if (source[cursor] !== ":") return false;
        cursor += 1;
        if (!parseValue()) return false;
        if (source[cursor] === "}") {
          cursor += 1;
          return true;
        }
        if (source[cursor] !== ",") return false;
        cursor += 1;
      }
      return false;
    }
    return parseNumber();
  };

  if (!parseValue() || cursor !== source.length) {
    return issue ?? "JSON is invalid or contains non-canonical whitespace/tokens";
  }
  return issue;
}

interface SensitivePayloadScan {
  exposed: boolean;
  redactedFields: string[];
}

function scanSensitivePayload(
  value: unknown,
  path: string,
): SensitivePayloadScan {
  if (Array.isArray(value)) {
    const scans = value.map((child, index) =>
      scanSensitivePayload(child, `${path}[${index}]`),
    );
    return {
      exposed: scans.some((scan) => scan.exposed),
      redactedFields: scans.flatMap((scan) => scan.redactedFields),
    };
  }
  if (!value || typeof value !== "object") {
    return { exposed: false, redactedFields: [] };
  }

  let exposed = false;
  const redactedFields: string[] = [];
  for (const [key, child] of Object.entries(value)) {
    const childPath = `${path}.${key}`;
    if (peSensitiveKeyKind(key) !== null) {
      redactedFields.push(childPath);
      exposed = exposed || child !== redactedValue;
      continue;
    }
    const childScan = scanSensitivePayload(child, childPath);
    exposed = exposed || childScan.exposed;
    redactedFields.push(...childScan.redactedFields);
  }
  return { exposed, redactedFields };
}

function jsonObject(value: unknown): Record<string, unknown> | null {
  if (!value || Array.isArray(value) || typeof value !== "object") return null;
  return value as Record<string, unknown>;
}

const rawEvaluatorKeys = new Set([
  "action_id",
  "action_index",
  "decision_hash",
  "expected_evidence",
  "parent_action_id",
  "plan_id",
  "plan_index",
  "profile_hash",
  "prompt_protocol_hash",
  "protocol_version",
  "reason",
  "schema_repair_attempted",
  "target_ids",
  "verdict",
]);

function rawScopeFidelityIssue(
  raw: Record<string, unknown>,
  observation: AgentPEObservationAuditRow,
  path: string,
): string | null {
  const expected: Record<string, unknown> = {
    plan_id: observation.plan_id,
    plan_index: observation.plan_index,
    action_id: observation.action_id ?? null,
    action_index: observation.action_index ?? null,
    parent_action_id: observation.parent_action_id ?? null,
  };
  for (const [key, expectedValue] of Object.entries(expected)) {
    if (!Object.prototype.hasOwnProperty.call(raw, key)) continue;
    const value = raw[key];
    if (key.endsWith("_index")) {
      if (
        value !== null &&
        (!Number.isInteger(value) || (value as number) < 0)
      ) {
        return `${path}.${key} 类型不合法`;
      }
    } else if (
      value !== null &&
      (typeof value !== "string" || !value)
    ) {
      return `${path}.${key} 类型不合法`;
    }
    if (value !== expectedValue) return `${path}.${key} 与 public scope 不一致`;
  }
  return null;
}

function rawEvaluatorFidelityIssue(
  raw: Record<string, unknown>,
  evaluator: AgentPEEvaluatorLinkage,
  observation: AgentPEObservationAuditRow,
  path: string,
): string | null {
  const unknownKeys = Object.keys(raw).filter(
    (key) => !rawEvaluatorKeys.has(key),
  );
  if (unknownKeys.length) return `${path} 含未知字段`;
  const expectedCore: Record<string, unknown> = {
    decision_hash: evaluator.decision_hash,
    protocol_version: evaluator.protocol_version,
    verdict: evaluator.verdict,
  };
  if (
    !Object.keys(expectedCore).some((key) =>
      Object.prototype.hasOwnProperty.call(raw, key),
    )
  ) {
    return `${path} 缺少 evaluator semantic field`;
  }
  for (const [key, expectedValue] of Object.entries(expectedCore)) {
    if (!Object.prototype.hasOwnProperty.call(raw, key)) continue;
    const value = raw[key];
    if (typeof value !== "string" || !value || value !== expectedValue) {
      return `${path}.${key} 与 public evaluator 不一致`;
    }
  }
  if (Object.prototype.hasOwnProperty.call(raw, "schema_repair_attempted")) {
    if (
      typeof raw.schema_repair_attempted !== "boolean" ||
      raw.schema_repair_attempted !==
        Boolean(evaluator.schema_repair_attempted)
    ) {
      return `${path}.schema_repair_attempted 与 public evaluator 不一致`;
    }
  }
  if (
    Object.prototype.hasOwnProperty.call(raw, "reason") &&
    (typeof raw.reason !== "string" || !raw.reason.trim())
  ) {
    return `${path}.reason 类型不合法`;
  }
  if (
    Object.prototype.hasOwnProperty.call(raw, "target_ids") &&
    (!Array.isArray(raw.target_ids) ||
      raw.target_ids.some(
        (value) => typeof value !== "string" || !value.trim(),
      ))
  ) {
    return `${path}.target_ids 类型不合法`;
  }
  if (
    Object.prototype.hasOwnProperty.call(raw, "expected_evidence") &&
    !jsonObject(raw.expected_evidence)
  ) {
    return `${path}.expected_evidence 类型不合法`;
  }
  for (const key of ["profile_hash", "prompt_protocol_hash"] as const) {
    if (
      Object.prototype.hasOwnProperty.call(raw, key) &&
      (typeof raw[key] !== "string" || !raw[key])
    ) {
      return `${path}.${key} 类型不合法`;
    }
  }
  return rawScopeFidelityIssue(raw, observation, path);
}

function payloadIntegrityIssue(
  payload: AgentPEJsonPayload,
  path: string,
  redactionRoot: string,
): {
  issue: string | null;
  decoded: unknown;
  redactedFields: string[];
} {
  if (
    !payload ||
    typeof payload !== "object" ||
    Object.keys(payload).sort().join(",") !==
      "canonical_json,encoding,redacted_fields,sha256"
  ) {
    return {
      issue: `${path} 不是 closed AgentPEJsonPayload`,
      decoded: null,
      redactedFields: [],
    };
  }
  if (
    payload.encoding !== "canonical_json_v1" ||
    typeof payload.canonical_json !== "string" ||
    typeof payload.sha256 !== "string" ||
    !sha256Pattern.test(payload.sha256) ||
    !Array.isArray(payload.redacted_fields) ||
    payload.redacted_fields.some((field) => typeof field !== "string")
  ) {
    return {
      issue: `${path} canonical/hash 字段类型无效`,
      decoded: null,
      redactedFields: [],
    };
  }
  const structuralIssue = canonicalJsonStructureIssue(payload.canonical_json);
  if (structuralIssue) {
    return {
      issue: `${path} 不满足 canonical_json_v1: ${structuralIssue}`,
      decoded: null,
      redactedFields: [],
    };
  }
  if (sha256Utf8(payload.canonical_json) !== payload.sha256) {
    return {
      issue: `${path} SHA-256 与 canonical bytes 不一致`,
      decoded: null,
      redactedFields: [],
    };
  }
  const sortedRedactions = [...new Set(payload.redacted_fields)].sort(
    compareCodePoints,
  );
  if (
    sortedRedactions.length !== payload.redacted_fields.length ||
    sortedRedactions.some(
      (field, index) => field !== payload.redacted_fields[index],
    )
  ) {
    return {
      issue: `${path}.redacted_fields 不是稳定去重顺序`,
      decoded: null,
      redactedFields: [],
    };
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(payload.canonical_json);
  } catch {
    return {
      issue: `${path} 不是可解析 JSON`,
      decoded: null,
      redactedFields: [],
    };
  }
  const sensitiveScan = scanSensitivePayload(decoded, redactionRoot);
  const expectedRedactions = [
    ...new Set(sensitiveScan.redactedFields),
  ].sort(compareCodePoints);
  if (sensitiveScan.exposed) {
    return {
      issue: `${path} 含未脱敏凭据或 provider 原始响应`,
      decoded,
      redactedFields: expectedRedactions,
    };
  }
  if (!sameIds(payload.redacted_fields, expectedRedactions)) {
    return {
      issue: `${path}.redacted_fields 与 canonical payload 敏感路径不一致`,
      decoded,
      redactedFields: expectedRedactions,
    };
  }
  return {
    issue: null,
    decoded,
    redactedFields: expectedRedactions,
  };
}

function sameIds(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function exactStringSequence(
  value: unknown,
  expected: string[],
): value is string[] {
  return (
    Array.isArray(value) &&
    value.every((item) => typeof item === "string") &&
    sameIds(value, expected)
  );
}

function mergedRedactedFields(values: string[][]): string[] {
  return [...new Set(values.flat())].sort(compareCodePoints);
}

function closedObjectIssue(
  value: unknown,
  path: string,
  requiredKeys: readonly string[],
  optionalKeys: readonly string[] = [],
): string | null {
  const object = jsonObject(value);
  if (!object) return `${path} is not a closed object`;
  const allowed = new Set([...requiredKeys, ...optionalKeys]);
  const missing = requiredKeys.filter(
    (key) => !Object.prototype.hasOwnProperty.call(object, key),
  );
  const extra = Object.keys(object).filter((key) => !allowed.has(key));
  if (missing.length > 0 || extra.length > 0) {
    return `${path} closed fields mismatch (missing=${missing.join(
      ",",
    ) || "none"}; extra=${extra.join(",") || "none"})`;
  }
  return null;
}

const auditRequiredKeys = [
  "contract_version",
  "run_id",
  "knowledge_base_id",
  "run_status",
  "counts",
  "ordering",
  "plans",
  "actions",
  "observations",
  "redaction_protocol_version",
  "provider_raw_response_exposed",
  "credentials_exposed",
] as const;
const countKeys = ["plans", "actions", "observations"] as const;
const orderingKeys = ["plans", "actions", "observations"] as const;
const planRowKeys = [
  "contract_version",
  "order_index",
  "id",
  "run_id",
  "knowledge_base_id",
  "retrieval_trace_id",
  "plan_index",
  "planner_protocol_version",
  "typed_action_schema_protocol_version",
  "typed_action_schema_protocol_hash",
  "typed_action_executor_protocol_version",
  "input_hash",
  "output_hash",
  "control_hash",
  "query_intent",
  "operating_envelope",
  "typed_actions",
  "validation",
  "planner_model_metadata",
  "status",
  "diagnostics",
  "action_ids",
  "action_count",
  "redacted_fields",
  "created_at",
] as const;
const actionRowKeys = [
  "contract_version",
  "order_index",
  "id",
  "run_id",
  "plan_id",
  "plan_index",
  "parent_action_id",
  "action_index",
  "action_type",
  "target_ids",
  "reason",
  "budget_request",
  "expected_evidence",
  "stop_condition",
  "validator",
  "status",
  "input_hash",
  "output_hash",
  "control_hash",
  "output",
  "diagnostics",
  "observation_ids",
  "observation_count",
  "redacted_fields",
  "created_at",
] as const;
const observationRowKeys = [
  "contract_version",
  "order_index",
  "id",
  "run_id",
  "plan_id",
  "plan_index",
  "action_id",
  "action_index",
  "parent_action_id",
  "observation_type",
  "protocol_version",
  "input_hash",
  "output_hash",
  "control_hash",
  "evaluator_linkage",
  "repair_linkage",
  "evidence_chunk_ids",
  "verdict",
  "observation",
  "diagnostics",
  "redacted_fields",
  "created_at",
] as const;
const validatorRequiredKeys = ["payload"] as const;
const validatorOptionalKeys = [
  "valid",
  "plan_valid",
  "schema_checked",
  "budget_checked",
  "target_ids_checked",
  "target_scope_checked",
  "typed_action_schema_protocol_version",
  "typed_action_schema_protocol_hash",
  "repair_protocol_version",
  "repair_budget_checked",
  "repair_round_index",
  "remaining_repair_budget_before",
  "action_input_hash",
  "repair_directive_validator_protocol_version",
  "repair_directive_validator_result",
  "repair_directive_hash",
  "validated_directive_hash",
] as const;
const evaluatorRequiredKeys = [
  "plan_id",
  "plan_index",
  "verdict",
  "replan_requested",
  "gray_zone_model_call_count",
] as const;
const evaluatorOptionalKeys = [
  "protocol_version",
  "decision_hash",
  "schema_repair_attempted",
] as const;
const repairRequiredKeys = [
  "action_id",
  "action_type",
  "repair_round_index",
  "remaining_repair_budget_before",
  "remaining_repair_budget_after",
] as const;
const repairOptionalKeys = [
  "parent_action_id",
  "repair_protocol_version",
  "action_input_hash",
  "action_output_hash",
  "before_context_package_id",
  "repaired_context_package_id",
  "before_retrieval_trace_id",
  "repaired_retrieval_trace_id",
] as const;

function compactId(value: string | null | undefined): string {
  if (!value) return "—";
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
}

function hashLine(label: string, value: string | null | undefined) {
  return (
    <p className="break-all font-mono text-[10px] text-white/48">
      <span className="text-white/28">{label}: </span>
      {value ?? "not persisted"}
    </p>
  );
}

function PayloadDetails({
  label,
  payload,
}: {
  label: string;
  payload: AgentPEJsonPayload;
}) {
  return (
    <details className="border-l border-white/10 pl-3">
      <summary className="cursor-pointer text-[11px] text-cyan-100/60">
        {label} · sha256 {compactId(payload.sha256)}
        {payload.redacted_fields.length
          ? ` · ${payload.redacted_fields.length} redacted`
          : ""}
      </summary>
      <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-all bg-black/18 p-3 font-mono text-[10px] leading-5 text-white/54 custom-scrollbar">
        {payload.canonical_json}
      </pre>
    </details>
  );
}

export function peAuditIntegrityIssue(
  audit: AgentPEAuditResponse,
): string | null {
  const runtimeContractIssue = agentPEAuditRuntimeContractIssue(audit);
  if (runtimeContractIssue) return runtimeContractIssue;
  const crossFieldContractIssue = agentPEAuditCrossFieldContractIssue(audit);
  if (crossFieldContractIssue) return crossFieldContractIssue;

  let closedIssue = closedObjectIssue(
    audit,
    "audit",
    auditRequiredKeys,
  );
  if (closedIssue) return closedIssue;
  closedIssue = closedObjectIssue(audit.counts, "audit.counts", countKeys);
  if (closedIssue) return closedIssue;
  closedIssue = closedObjectIssue(
    audit.ordering,
    "audit.ordering",
    orderingKeys,
  );
  if (closedIssue) return closedIssue;
  if (
    !Array.isArray(audit.plans) ||
    !Array.isArray(audit.actions) ||
    !Array.isArray(audit.observations)
  ) {
    return "audit plans/actions/observations must be arrays";
  }
  for (const [index, row] of audit.plans.entries()) {
    closedIssue = closedObjectIssue(
      row,
      `plans[${index}]`,
      planRowKeys,
    );
    if (closedIssue) return closedIssue;
  }
  for (const [index, row] of audit.actions.entries()) {
    closedIssue = closedObjectIssue(
      row,
      `actions[${index}]`,
      actionRowKeys,
    );
    if (closedIssue) return closedIssue;
    closedIssue = closedObjectIssue(
      row.validator,
      `actions[${index}].validator`,
      validatorRequiredKeys,
      validatorOptionalKeys,
    );
    if (closedIssue) return closedIssue;
  }
  for (const [index, row] of audit.observations.entries()) {
    closedIssue = closedObjectIssue(
      row,
      `observations[${index}]`,
      observationRowKeys,
    );
    if (closedIssue) return closedIssue;
    if (row.evaluator_linkage !== null && row.evaluator_linkage !== undefined) {
      closedIssue = closedObjectIssue(
        row.evaluator_linkage,
        `observations[${index}].evaluator_linkage`,
        evaluatorRequiredKeys,
        evaluatorOptionalKeys,
      );
      if (closedIssue) return closedIssue;
    }
    if (row.repair_linkage !== null && row.repair_linkage !== undefined) {
      closedIssue = closedObjectIssue(
        row.repair_linkage,
        `observations[${index}].repair_linkage`,
        repairRequiredKeys,
        repairOptionalKeys,
      );
      if (closedIssue) return closedIssue;
    }
  }
  if (
    audit.redaction_protocol_version !==
      SENSITIVE_FIELD_KEY_PROTOCOL_VERSION
  ) {
    return "API P&E redaction protocol_version is unsupported";
  }
  if (audit.contract_version !== "agent_pe_audit_public_v1") {
    return "API P&E audit contract_version 不受支持";
  }
  if (
    audit.ordering.plans !== "plan_index ASC, created_at ASC, id ASC" ||
    audit.ordering.actions !==
      "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC" ||
    audit.ordering.observations !== "created_at ASC, id ASC"
  ) {
    return "API canonical ordering contract 不匹配";
  }
  if (
    audit.counts.plans !== audit.plans.length ||
    audit.counts.actions !== audit.actions.length ||
    audit.counts.observations !== audit.observations.length ||
    !Number.isInteger(audit.counts.plans) ||
    !Number.isInteger(audit.counts.actions) ||
    !Number.isInteger(audit.counts.observations) ||
    audit.counts.plans < 0 ||
    audit.counts.actions < 0 ||
    audit.counts.observations < 0
  ) {
    return "API counts 与返回行数不一致";
  }
  if (
    !audit.plans.every((row, index) => row.order_index === index) ||
    !audit.actions.every((row, index) => row.order_index === index) ||
    !audit.observations.every((row, index) => row.order_index === index)
  ) {
    return "API stable order_index 不连续";
  }
  const planIds = new Set(audit.plans.map((row) => row.id));
  const actionIds = new Set(audit.actions.map((row) => row.id));
  const observationIds = new Set(audit.observations.map((row) => row.id));
  if (
    planIds.size !== audit.plans.length ||
    actionIds.size !== audit.actions.length ||
    observationIds.size !== audit.observations.length
  ) {
    return "API P&E rows 含重复 id";
  }
  if (
    audit.plans.some(
      (row, index) =>
        !Number.isInteger(row.plan_index) || row.plan_index !== index,
    )
  ) {
    return "plan_index 必须是完整的 0..N-1 序列";
  }
  const planById = new Map(audit.plans.map((row) => [row.id, row]));
  if (
    audit.plans.some(
      (row) =>
        row.contract_version !== "agent_plan_audit_row_v1" ||
        row.run_id !== audit.run_id ||
        row.knowledge_base_id !== audit.knowledge_base_id ||
        row.action_count !== row.action_ids.length,
    )
  ) {
    return "plan row scope、contract 或 action count 不一致";
  }
  const actionById = new Map(audit.actions.map((row) => [row.id, row]));
  const expectedActionIds = audit.plans.flatMap((plan) =>
    audit.actions
      .filter((action) => action.plan_id === plan.id)
      .sort((left, right) => left.action_index - right.action_index)
      .map((action) => action.id),
  );
  if (!sameIds(expectedActionIds, audit.actions.map((row) => row.id))) {
    return "action rows 不满足 canonical plan/action 顺序";
  }
  if (
    audit.actions.some((row) => {
      const plan = planById.get(row.plan_id);
      const parent = row.parent_action_id
        ? actionById.get(row.parent_action_id)
        : null;
      return (
        row.contract_version !== "agent_action_audit_row_v1" ||
        row.run_id !== audit.run_id ||
        !plan ||
        row.plan_index !== plan.plan_index ||
        !Number.isInteger(row.action_index) ||
        row.action_index < 0 ||
        row.observation_count !== row.observation_ids.length ||
        (row.parent_action_id !== null &&
          row.parent_action_id !== undefined &&
          (!parent ||
            parent.plan_id !== row.plan_id ||
            parent.action_index >= row.action_index))
      );
    })
  ) {
    return "action 指向未返回或跨 run 的 plan";
  }
  if (
    audit.plans.some((plan) => {
      const indexes = audit.actions
        .filter((action) => action.plan_id === plan.id)
        .map((action) => action.action_index);
      return indexes.some((value, index) => value !== index);
    })
  ) {
    return "action_index 必须是每个 plan 内完整的 0..N-1 序列";
  }
  if (
    audit.observations.some(
      (row) => {
        const plan = planById.get(row.plan_id);
        const action = row.action_id ? actionById.get(row.action_id) : null;
        return (
          row.contract_version !== "agent_observation_audit_row_v1" ||
          row.run_id !== audit.run_id ||
          !plan ||
          row.plan_index !== plan.plan_index ||
          (row.action_id !== null &&
            row.action_id !== undefined &&
            (!action ||
              action.plan_id !== row.plan_id ||
              action.action_index !== row.action_index ||
              action.parent_action_id !== row.parent_action_id))
        );
      },
    )
  ) {
    return "observation 指向未返回或跨 run 的 plan/action";
  }
  if (
    audit.plans.some(
      (plan) =>
        !sameIds(
          plan.action_ids,
          audit.actions
            .filter((action) => action.plan_id === plan.id)
            .map((action) => action.id),
        ),
    ) ||
    audit.actions.some(
      (action) =>
        !sameIds(
          action.observation_ids,
          audit.observations
            .filter((observation) => observation.action_id === action.id)
            .map((observation) => observation.id),
        ),
    )
  ) {
    return "plan/action child id 列表与 canonical rows 不一致";
  }

  const decodedPayloads = new Map<string, unknown>();
  const validatePayloadRow = (
    rowPath: string,
    declaredRedactedFields: unknown,
    entries: Array<[string, AgentPEJsonPayload, string]>,
  ): string | null => {
    const payloadRedactedFields: string[][] = [];
    for (const [path, payload, redactionRoot] of entries) {
      const result = payloadIntegrityIssue(payload, path, redactionRoot);
      if (result.issue) return result.issue;
      decodedPayloads.set(path, result.decoded);
      payloadRedactedFields.push(result.redactedFields);
    }
    const expectedRedactedFields = mergedRedactedFields(
      payloadRedactedFields,
    );
    if (
      !exactStringSequence(declaredRedactedFields, expectedRedactedFields)
    ) {
      return `${rowPath}.redacted_fields 与子 payload 敏感路径并集不一致`;
    }
    return null;
  };

  for (const [index, plan] of audit.plans.entries()) {
    const issue = validatePayloadRow(
      `plans[${index}]`,
      plan.redacted_fields,
      [
        [`plans[${index}].query_intent`, plan.query_intent, "query_intent"],
        [
          `plans[${index}].operating_envelope`,
          plan.operating_envelope,
          "operating_envelope",
        ],
        [`plans[${index}].typed_actions`, plan.typed_actions, "typed_actions"],
        [`plans[${index}].validation`, plan.validation, "validation"],
        [
          `plans[${index}].planner_model_metadata`,
          plan.planner_model_metadata,
          "planner_model_metadata",
        ],
        [`plans[${index}].diagnostics`, plan.diagnostics, "diagnostics"],
      ],
    );
    if (issue) return issue;
  }
  for (const [index, action] of audit.actions.entries()) {
    const issue = validatePayloadRow(
      `actions[${index}]`,
      action.redacted_fields,
      [
        [
          `actions[${index}].budget_request`,
          action.budget_request,
          "budget_request",
        ],
        [
          `actions[${index}].expected_evidence`,
          action.expected_evidence,
          "expected_evidence",
        ],
        [
          `actions[${index}].stop_condition`,
          action.stop_condition,
          "stop_condition",
        ],
        [
          `actions[${index}].validator.payload`,
          action.validator.payload,
          "validation",
        ],
        [`actions[${index}].output`, action.output, "output"],
        [`actions[${index}].diagnostics`, action.diagnostics, "diagnostics"],
      ],
    );
    if (issue) return issue;
  }
  for (const [index, observation] of audit.observations.entries()) {
    const issue = validatePayloadRow(
      `observations[${index}]`,
      observation.redacted_fields,
      [
        [
          `observations[${index}].observation`,
          observation.observation,
          "observation",
        ],
        [
          `observations[${index}].diagnostics`,
          observation.diagnostics,
          "diagnostics",
        ],
      ],
    );
    if (issue) return issue;
  }

  const actionPayloads = new Map<
    string,
    { validator: unknown; output: unknown }
  >();
  for (const [index, action] of audit.actions.entries()) {
    actionPayloads.set(action.id, {
      validator: decodedPayloads.get(
        `actions[${index}].validator.payload`,
      ),
      output: decodedPayloads.get(`actions[${index}].output`),
    });
  }
  const observationPayloads = new Map<string, unknown>();
  for (const [index, observation] of audit.observations.entries()) {
    observationPayloads.set(
      observation.id,
      decodedPayloads.get(`observations[${index}].observation`),
    );
  }

  const repairRows: AgentPERepairLinkage[] = [];
  for (const observation of audit.observations) {
    const evaluator = observation.evaluator_linkage;
    if (observation.observation_type === "evidence_evaluator") {
      if (
        !evaluator ||
        evaluator.plan_id !== observation.plan_id ||
        evaluator.plan_index !== observation.plan_index ||
        evaluator.verdict !== observation.verdict ||
        evaluator.protocol_version !== observation.protocol_version ||
        evaluator.replan_requested !==
          !["sufficient", "insufficient_corpus"].includes(evaluator.verdict) ||
        evaluator.gray_zone_model_call_count !== 0
      ) {
        return "evidence evaluator linkage 与 observation/plan 不一致";
      }
      const observationPayload = jsonObject(
        observationPayloads.get(observation.id),
      );
      if (!observationPayload) {
        return "evidence evaluator raw observation 不是 object";
      }
      const observationScopeIssue = rawScopeFidelityIssue(
        observationPayload,
        observation,
        "evidence evaluator raw observation",
      );
      if (observationScopeIssue) return observationScopeIssue;
      if (
        Object.prototype.hasOwnProperty.call(
          observationPayload,
          "protocol_version",
        ) &&
        (typeof observationPayload.protocol_version !== "string" ||
          !observationPayload.protocol_version ||
          observationPayload.protocol_version !== observation.protocol_version)
      ) {
        return "evidence evaluator raw observation protocol 不一致";
      }
      if (
        Object.prototype.hasOwnProperty.call(
          observationPayload,
          "bounded_graph_observation",
        ) &&
        !jsonObject(observationPayload.bounded_graph_observation)
      ) {
        return "evidence evaluator bounded graph observation 不是 object";
      }
      const rawEvaluatorPresent = Object.prototype.hasOwnProperty.call(
        observationPayload,
        "evaluator_verdict",
      );
      const rawEvaluator = jsonObject(observationPayload.evaluator_verdict);
      if (rawEvaluatorPresent && !rawEvaluator) {
        return "evidence evaluator raw lineage object 不合法";
      }
      if (rawEvaluator) {
        const rawEvaluatorIssue = rawEvaluatorFidelityIssue(
          rawEvaluator,
          evaluator,
          observation,
          "evidence evaluator raw verdict",
        );
        if (rawEvaluatorIssue) return rawEvaluatorIssue;
      }
      const linkedAction = observation.action_id
        ? actionById.get(observation.action_id)
        : null;
      const linkedActionPayload = linkedAction
        ? actionPayloads.get(linkedAction.id)
        : null;
      const linkedActionOutput = linkedAction
        ? jsonObject(linkedActionPayload?.output)
        : null;
      if (linkedAction && !linkedActionOutput) {
        return "evidence evaluator linked action output 不是 object";
      }
      const linkedActionEvaluator = jsonObject(
        linkedActionOutput?.evaluator_verdict,
      );
      const linkedActionEvaluatorPresent = Boolean(
        linkedActionOutput &&
          Object.prototype.hasOwnProperty.call(
            linkedActionOutput,
            "evaluator_verdict",
          ),
      );
      if (linkedActionEvaluatorPresent && !linkedActionEvaluator) {
        return "evidence evaluator linked action lineage object 不合法";
      }
      if (linkedActionOutput) {
        const linkedScopeIssue = rawScopeFidelityIssue(
          linkedActionOutput,
          observation,
          "evidence evaluator linked action output",
        );
        if (linkedScopeIssue) return linkedScopeIssue;
      }
      if (linkedActionEvaluator) {
        const linkedEvaluatorIssue = rawEvaluatorFidelityIssue(
          linkedActionEvaluator,
          evaluator,
          observation,
          "evidence evaluator linked action verdict",
        );
        if (linkedEvaluatorIssue) return linkedEvaluatorIssue;
      }
      const rawLineageCandidates = [
        observation.output_hash,
        evaluator.decision_hash,
        observationPayload?.action_output_hash,
        observationPayload?.observation_hash,
        rawEvaluator?.decision_hash,
        linkedAction?.output_hash,
        linkedActionOutput?.action_output_hash,
        linkedActionOutput?.observation_hash,
        linkedActionEvaluator?.decision_hash,
      ].filter((value) => value !== null && value !== undefined);
      if (
        rawLineageCandidates.some(
          (value) =>
            typeof value !== "string" ||
            !value ||
            value !== (evaluator.decision_hash ?? null),
        )
      ) {
        return "evidence evaluator raw/public hash lineage 不一致";
      }
    } else if (evaluator !== null && evaluator !== undefined) {
      return "非 evaluator observation 不得携带 evaluator_linkage";
    }

    const repair = observation.repair_linkage;
    if (observation.observation_type !== "typed_repair_round") {
      if (repair !== null && repair !== undefined) {
        return "非 repair observation 不得携带 repair_linkage";
      }
      continue;
    }
    const action = observation.action_id
      ? actionById.get(observation.action_id)
      : null;
    const actionPayload = action ? actionPayloads.get(action.id) : null;
    const validatorPayload = jsonObject(actionPayload?.validator);
    const outputPayload = jsonObject(actionPayload?.output);
    const observationPayload = jsonObject(
      observationPayloads.get(observation.id),
    );
    if (
      !repair ||
      !action ||
      !repairActionTypes.has(action.action_type) ||
      repair.action_id !== action.id ||
      repair.parent_action_id !== action.parent_action_id ||
      repair.action_type !== action.action_type ||
      repair.repair_protocol_version !== observation.protocol_version ||
      repair.action_input_hash !== action.input_hash ||
      repair.action_input_hash !== observation.input_hash ||
      repair.action_output_hash !== action.output_hash ||
      repair.action_output_hash !== observation.output_hash ||
      !Number.isInteger(repair.repair_round_index) ||
      !Number.isInteger(repair.remaining_repair_budget_before) ||
      !Number.isInteger(repair.remaining_repair_budget_after) ||
      repair.remaining_repair_budget_before < 1 ||
      repair.remaining_repair_budget_after !==
        repair.remaining_repair_budget_before - 1 ||
      typeof repair.before_context_package_id !== "string" ||
      !repair.before_context_package_id ||
      typeof repair.repaired_context_package_id !== "string" ||
      !repair.repaired_context_package_id ||
      !validatorPayload ||
      !outputPayload ||
      !observationPayload ||
      action.output.canonical_json !== observation.observation.canonical_json ||
      action.validator.repair_protocol_version !==
        repair.repair_protocol_version ||
      action.validator.repair_round_index !== repair.repair_round_index ||
      action.validator.remaining_repair_budget_before !==
        repair.remaining_repair_budget_before ||
      action.validator.action_input_hash !== repair.action_input_hash ||
      validatorPayload.repair_protocol_version !==
        repair.repair_protocol_version ||
      validatorPayload.repair_round_index !== repair.repair_round_index ||
      validatorPayload.remaining_repair_budget_before !==
        repair.remaining_repair_budget_before ||
      validatorPayload.action_input_hash !== repair.action_input_hash ||
      outputPayload.action_type !== repair.action_type ||
      outputPayload.protocol_version !== repair.repair_protocol_version ||
      outputPayload.repair_round_index !== repair.repair_round_index ||
      outputPayload.remaining_repair_budget_before !==
        repair.remaining_repair_budget_before ||
      outputPayload.remaining_repair_budget_after !==
        repair.remaining_repair_budget_after ||
      outputPayload.action_input_hash !== repair.action_input_hash ||
      outputPayload.action_output_hash !== repair.action_output_hash ||
      outputPayload.before_context_package_id !==
        repair.before_context_package_id ||
      outputPayload.repaired_context_package_id !==
        repair.repaired_context_package_id ||
      outputPayload.before_retrieval_trace_id !==
        repair.before_retrieval_trace_id ||
      outputPayload.repaired_retrieval_trace_id !==
        repair.repaired_retrieval_trace_id
    ) {
      return "repair linkage 与 action/validator/observation/hash/budget 不一致";
    }
    repairRows.push(repair);
  }
  if (
    repairRows.some(
      (repair, index) =>
        repair.repair_round_index !== index ||
        (index > 0 &&
          repair.remaining_repair_budget_before !==
            repairRows[index - 1].remaining_repair_budget_after),
    )
  ) {
    return "repair round/budget 序列不闭合";
  }
  if (
    audit.provider_raw_response_exposed !== false ||
    audit.credentials_exposed !== false
  ) {
    return "API 标记存在 provider raw response 或凭据暴露";
  }
  return null;
}

export function AgentPEAuditPanel({
  runId,
  isRunning = false,
}: AgentPEAuditPanelProps) {
  const auditQuery = useQuery({
    queryKey: ["agent-pe-audit", runId],
    queryFn: () => fetchAgentPEAudit(runId as string),
    enabled: Boolean(runId),
    retry: false,
    refetchInterval: runId && isRunning ? 1500 : false,
  });

  if (!runId) return null;
  if (auditQuery.isLoading) {
    return (
      <section className="mb-5 flex items-center gap-2 border-l border-cyan-200/20 bg-cyan-200/[0.025] p-4 text-xs text-white/54">
        <Loader2 className="size-4 animate-spin text-cyan-100/70" />
        正在读取 PostgreSQL P&amp;E audit rows…
      </section>
    );
  }
  if (auditQuery.error) {
    const error = auditQuery.error as Error & { status?: number };
    return (
      <section
        role="alert"
        data-testid="agent-pe-audit-error"
        className="mb-5 border-l border-rose-300/40 bg-rose-300/[0.045] p-4 text-xs text-rose-100/78"
      >
        P&amp;E audit 读取失败{error.status ? `（HTTP ${error.status}）` : ""}：
        {error.message}
      </section>
    );
  }

  const audit = auditQuery.data;
  if (!audit) return null;
  const integrityIssue = peAuditIntegrityIssue(audit);
  if (integrityIssue) {
    return (
      <section
        role="alert"
        data-testid="agent-pe-audit-integrity-error"
        className="mb-5 flex items-start gap-2 border-l border-rose-300/40 bg-rose-300/[0.045] p-4 text-xs text-rose-100/78"
      >
        <AlertTriangle className="mt-0.5 size-4 shrink-0" />
        <span>P&amp;E 审计契约冲突：{integrityIssue}</span>
      </section>
    );
  }

  return (
    <section
      data-testid="agent-pe-audit"
      className="mb-5 grid gap-5 border-l border-cyan-200/20 bg-[rgba(7,18,37,0.62)] p-4 text-white"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">PostgreSQL P&amp;E audit</p>
          <p className="mt-1 text-sm text-white/62">
            {audit.counts.plans} plans · {audit.counts.actions} typed actions ·{" "}
            {audit.counts.observations} observations
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 text-xs text-cyan-100/66">
          <CheckCircle2 className="size-4" />
          canonical rows · {compactId(audit.run_id)}
        </span>
      </div>
      <details className="border-l border-white/10 pl-3">
        <summary className="cursor-pointer text-[11px] text-cyan-100/60">
          Canonical database ordering
        </summary>
        <div className="mt-2 grid gap-1">
          {hashLine("plans", audit.ordering.plans)}
          {hashLine("actions", audit.ordering.actions)}
          {hashLine("observations", audit.ordering.observations)}
        </div>
      </details>

      <div className="grid gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-100/58">
          Agent plans ({audit.plans.length})
        </p>
        {audit.plans.map((plan) => (
          <article
            key={plan.id}
            data-testid="agent-pe-plan-row"
            data-row-id={plan.id}
            data-order-index={plan.order_index}
            className="grid gap-2 border-l border-white/12 bg-white/[0.02] p-3"
          >
            <p className="text-sm text-white/76">
              #{plan.order_index} · plan[{plan.plan_index}] · {plan.status}
            </p>
            <p className="font-mono text-[10px] text-white/42">
              id {plan.id} · trace {plan.retrieval_trace_id ?? "none"} · actions{" "}
              {plan.action_count}
            </p>
            <p className="break-all font-mono text-[10px] text-white/42">
              kb {plan.knowledge_base_id} · created {plan.created_at}
            </p>
            <p className="break-all text-[11px] text-white/52">
              action ids: {plan.action_ids.join(", ") || "none"}
            </p>
            {hashLine("schema", plan.typed_action_schema_protocol_version)}
            {hashLine("schema_hash", plan.typed_action_schema_protocol_hash)}
            {hashLine("planner_protocol", plan.planner_protocol_version)}
            {hashLine(
              "executor_protocol",
              plan.typed_action_executor_protocol_version,
            )}
            {hashLine("input_hash", plan.input_hash)}
            {hashLine("output_hash", plan.output_hash)}
            {hashLine("control_hash", plan.control_hash)}
            <PayloadDetails label="query_intent" payload={plan.query_intent} />
            <PayloadDetails
              label="operating_envelope"
              payload={plan.operating_envelope}
            />
            <PayloadDetails label="typed_actions" payload={plan.typed_actions} />
            <PayloadDetails label="validation" payload={plan.validation} />
            <PayloadDetails
              label="planner_model_metadata"
              payload={plan.planner_model_metadata}
            />
            <PayloadDetails label="diagnostics" payload={plan.diagnostics} />
            <p className="break-all text-[10px] text-rose-100/52">
              redacted fields: {plan.redacted_fields.join(", ") || "none"}
            </p>
          </article>
        ))}
      </div>

      <div className="grid gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-100/58">
          Typed actions ({audit.actions.length})
        </p>
        {audit.actions.map((action) => (
          <article
            key={action.id}
            data-testid="agent-pe-action-row"
            data-row-id={action.id}
            data-order-index={action.order_index}
            className="grid gap-2 border-l border-cyan-200/16 bg-cyan-200/[0.018] p-3"
          >
            <p className="text-sm text-white/78">
              #{action.order_index} · plan[{action.plan_index}] action[
              {action.action_index}] · {action.action_type}
            </p>
            <p className="text-xs text-white/58">{action.reason}</p>
            <p className="break-all font-mono text-[10px] text-white/42">
              id {action.id} · plan {action.plan_id} · parent{" "}
              {action.parent_action_id ?? "none"} · status {action.status}
            </p>
            <p className="break-all text-[11px] text-white/52">
              targets: {action.target_ids.join(", ") || "none"}
            </p>
            <p className="break-all font-mono text-[10px] text-white/42">
              created {action.created_at} · observations{" "}
              {action.observation_count}:{" "}
              {action.observation_ids.join(", ") || "none"}
            </p>
            {hashLine("input_hash", action.input_hash)}
            {hashLine("output_hash", action.output_hash)}
            {hashLine("control_hash", action.control_hash)}
            {hashLine(
              "validator_schema",
              action.validator.typed_action_schema_protocol_version,
            )}
            {hashLine(
              "validator_schema_hash",
              action.validator.typed_action_schema_protocol_hash,
            )}
            {hashLine(
              "repair_protocol",
              action.validator.repair_protocol_version,
            )}
            <p className="text-[11px] text-white/54">
              validator: plan={String(action.validator.plan_valid)} schema=
              {String(action.validator.schema_checked)} budget=
              {String(action.validator.budget_checked)} target=
              {String(action.validator.target_scope_checked)}
            </p>
            <PayloadDetails label="budget_request" payload={action.budget_request} />
            <PayloadDetails label="expected_evidence" payload={action.expected_evidence} />
            <PayloadDetails label="stop_condition" payload={action.stop_condition} />
            <PayloadDetails label="validator payload" payload={action.validator.payload} />
            <PayloadDetails label="output" payload={action.output} />
            <PayloadDetails label="diagnostics" payload={action.diagnostics} />
            <p className="break-all text-[10px] text-rose-100/52">
              redacted fields: {action.redacted_fields.join(", ") || "none"}
            </p>
          </article>
        ))}
      </div>

      <div className="grid gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-100/58">
          Observations ({audit.observations.length})
        </p>
        {audit.observations.map((observation) => (
          <article
            key={observation.id}
            data-testid="agent-pe-observation-row"
            data-row-id={observation.id}
            data-order-index={observation.order_index}
            className="grid gap-2 border-l border-amber-200/18 bg-amber-200/[0.018] p-3"
          >
            <p className="text-sm text-white/78">
              #{observation.order_index} · {observation.observation_type} ·{" "}
              {observation.verdict}
            </p>
            <p className="break-all font-mono text-[10px] text-white/42">
              id {observation.id} · plan {observation.plan_id} · action{" "}
              {observation.action_id ?? "evaluator-only"} · evidence{" "}
              {observation.evidence_chunk_ids.length}
            </p>
            <p className="break-all font-mono text-[10px] text-white/42">
              created {observation.created_at} · plan index{" "}
              {observation.plan_index} · action index{" "}
              {observation.action_index ?? "none"} · parent{" "}
              {observation.parent_action_id ?? "none"}
            </p>
            <p className="break-all text-[11px] text-white/52">
              evidence ids:{" "}
              {observation.evidence_chunk_ids.join(", ") || "none"}
            </p>
            {hashLine("protocol", observation.protocol_version)}
            {hashLine("input_hash", observation.input_hash)}
            {hashLine("output_hash", observation.output_hash)}
            {hashLine("control_hash", observation.control_hash)}
            {observation.evaluator_linkage ? (
              <p className="text-[11px] text-violet-100/68">
                evaluator: {observation.evaluator_linkage.verdict} · replan{" "}
                {String(observation.evaluator_linkage.replan_requested)} · decision{" "}
                {observation.evaluator_linkage.decision_hash ?? "not persisted"} · gray
                model calls{" "}
                {observation.evaluator_linkage.gray_zone_model_call_count} · schema repair{" "}
                {String(
                  observation.evaluator_linkage.schema_repair_attempted ?? false,
                )}
              </p>
            ) : null}
            {observation.repair_linkage ? (
              <p className="text-[11px] text-amber-100/70">
                repair[{observation.repair_linkage.repair_round_index}]{" "}
                {observation.repair_linkage.action_type} · budget{" "}
                {observation.repair_linkage.remaining_repair_budget_before} →{" "}
                {observation.repair_linkage.remaining_repair_budget_after} · parent{" "}
                {observation.repair_linkage.parent_action_id ?? "none"}
              </p>
            ) : null}
            {observation.repair_linkage ? (
              <div className="grid gap-1 border-l border-amber-100/12 pl-3">
                {hashLine(
                  "repair_input_hash",
                  observation.repair_linkage.action_input_hash,
                )}
                {hashLine(
                  "repair_output_hash",
                  observation.repair_linkage.action_output_hash,
                )}
                {hashLine(
                  "before_package",
                  observation.repair_linkage.before_context_package_id,
                )}
                {hashLine(
                  "repaired_package",
                  observation.repair_linkage.repaired_context_package_id,
                )}
                {hashLine(
                  "before_trace",
                  observation.repair_linkage.before_retrieval_trace_id,
                )}
                {hashLine(
                  "repaired_trace",
                  observation.repair_linkage.repaired_retrieval_trace_id,
                )}
              </div>
            ) : null}
            <PayloadDetails label="observation" payload={observation.observation} />
            <PayloadDetails label="diagnostics" payload={observation.diagnostics} />
            <p className="break-all text-[10px] text-rose-100/52">
              redacted fields:{" "}
              {observation.redacted_fields.join(", ") || "none"}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}
