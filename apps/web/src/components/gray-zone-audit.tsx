import type { AgentTraceEventPayload, QueryFacetPosteriorCalibration, RetrievalGrayZoneDecision, RetrievalTraceStepsResponse } from "@course-kg/shared";

import { cn } from "@/lib/utils";

type JsonRecord = Record<string, unknown>;

type GrayZoneAuditInspection = {
  ok: boolean;
  issues: string[];
  modelCallCount?: 0;
  localDecisions: RetrievalGrayZoneDecision[];
  partitionDecisions: RetrievalGrayZoneDecision[];
};

const sha256Pattern = /^[0-9a-f]{64}$/;
const requiredDecisionHashes = [
  "protocol_hash",
  "input_hash",
  "threshold_hash",
  "traversal_protocol_hash",
  "runtime_settings_hash",
  "agent_operating_envelope_hash",
  "decision_hash",
] as const;

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function nonEmptyRecord(value: unknown): value is JsonRecord {
  return isRecord(value) && Object.keys(value).length > 0;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function requireNonEmptyString(record: JsonRecord, field: string, path: string, issues: string[]) {
  if (typeof record[field] !== "string" || !record[field].trim()) {
    issues.push(`${path}.${field} 缺失或为空`);
  }
}

function requireNonNegativeInteger(record: JsonRecord, field: string, path: string, issues: string[]) {
  if (typeof record[field] !== "number" || !Number.isInteger(record[field]) || Number(record[field]) < 0) {
    issues.push(`${path}.${field} 缺失或不是非负整数`);
  }
}

function inspectDecision(value: unknown, path: string, issues: string[]): RetrievalGrayZoneDecision | null {
  if (!isRecord(value)) {
    issues.push(`${path} 不是对象`);
    return null;
  }

  for (const field of ["decision", "protocol_version", "matched_rule"] as const) {
    requireNonEmptyString(value, field, path, issues);
  }
  for (const field of requiredDecisionHashes) {
    const hash = value[field];
    if (typeof hash !== "string" || !sha256Pattern.test(hash)) {
      issues.push(`${path}.${field} 必须是规范的 SHA-256`);
    }
  }
  if (typeof value.path_distance !== "number" || !Number.isFinite(value.path_distance)) {
    issues.push(`${path}.path_distance 缺失或不是有限数值`);
  }
  if (!["green", "gray", "red", "hard_stop"].includes(String(value.distance_zone))) {
    issues.push(`${path}.distance_zone 非法`);
  }
  if (!["deterministic_local_rule", "deterministic_distance_partition"].includes(String(value.decision_source))) {
    issues.push(`${path}.decision_source 非法`);
  }
  if (!Object.prototype.hasOwnProperty.call(value, "model_call_count") || value.model_call_count !== 0) {
    issues.push(`${path}.model_call_count 必须显式等于 0`);
  }
  for (const field of ["minimum_audit", "hard_interrupt_state", "support_refs"] as const) {
    if (!nonEmptyRecord(value[field])) {
      issues.push(`${path}.${field} 缺失或为空`);
    }
  }
  if (value.gray_candidate_reasons !== undefined && !Array.isArray(value.gray_candidate_reasons)) {
    issues.push(`${path}.gray_candidate_reasons 必须是数组`);
  }

  return value as unknown as RetrievalGrayZoneDecision;
}

export function inspectGrayZoneTrace(trace: unknown): GrayZoneAuditInspection {
  const issues: string[] = [];
  if (!isRecord(trace)) {
    return { ok: false, issues: ["检索轨迹 payload 不是对象"], localDecisions: [], partitionDecisions: [] };
  }
  requireNonEmptyString(trace, "trace_id", "trace", issues);
  requireNonEmptyString(trace, "gray_zone_protocol", "trace", issues);
  if (!Object.prototype.hasOwnProperty.call(trace, "gray_zone_model_call_count") || trace.gray_zone_model_call_count !== 0) {
    issues.push("trace.gray_zone_model_call_count 必须显式等于 0");
  }

  const determinism = trace.gray_zone_determinism;
  if (!isRecord(determinism)) {
    issues.push("trace.gray_zone_determinism 缺失");
  } else {
    if (determinism.status !== "passed") {
      issues.push(`trace.gray_zone_determinism.status 必须为 passed，当前为 ${String(determinism.status)}`);
    }
    for (const field of [
      "checked_record_count",
      "unique_record_count",
      "local_rule_record_count",
      "red_partition_record_count",
      "hard_stop_partition_record_count",
      "duplicate_reference_count",
      "conflict_count",
      "incomplete_record_count",
    ]) {
      if (typeof determinism[field] !== "number" || !Number.isInteger(determinism[field]) || Number(determinism[field]) < 0) {
        issues.push(`trace.gray_zone_determinism.${field} 缺失或非法`);
      }
    }
    for (const field of ["conflicts", "issues"]) {
      if (!Array.isArray(determinism[field])) {
        issues.push(`trace.gray_zone_determinism.${field} 缺失或不是数组`);
      }
    }
    if (determinism.conflict_count !== 0 || determinism.incomplete_record_count !== 0) {
      issues.push("determinism audit 存在冲突或不完整记录");
    }
  }

  const rawDecisions = trace.gray_zone_path_decisions;
  if (rawDecisions !== undefined && !Array.isArray(rawDecisions)) {
    issues.push("trace.gray_zone_path_decisions 必须是数组");
  }
  const decisions = (Array.isArray(rawDecisions) ? rawDecisions : [])
    .map((decision, index) => inspectDecision(decision, `gray_zone_path_decisions[${index}]`, issues))
    .filter((decision): decision is RetrievalGrayZoneDecision => decision !== null);
  const rawThresholdHits = trace.path_distance_threshold_hits;
  if (rawThresholdHits !== undefined && !Array.isArray(rawThresholdHits)) {
    issues.push("trace.path_distance_threshold_hits 必须是数组");
  }
  const thresholdHits = (Array.isArray(rawThresholdHits) ? rawThresholdHits : [])
    .map((decision, index) => inspectDecision(decision, `path_distance_threshold_hits[${index}]`, issues))
    .filter((decision): decision is RetrievalGrayZoneDecision => decision !== null);
  const localDecisions = decisions.filter((decision) => decision.decision_source === "deterministic_local_rule");
  const misplacedLocalDecisions = thresholdHits.filter((decision) => decision.decision_source === "deterministic_local_rule");
  const misplacedPartitionDecisions = decisions.filter((decision) => decision.decision_source === "deterministic_distance_partition");
  const partitionDecisions = thresholdHits.filter((decision) => decision.decision_source === "deterministic_distance_partition");
  if (misplacedLocalDecisions.length || misplacedPartitionDecisions.length) {
    issues.push("local rule 与 distance partition 记录出现在错误的响应集合中");
  }
  for (const [index, decision] of localDecisions.entries()) {
    const budgetState = isRecord(decision.hard_interrupt_state)
      ? decision.hard_interrupt_state.traversal_observation_budget
      : undefined;
    if (!nonEmptyRecord(budgetState)) {
      issues.push(`gray_zone_path_decisions[${index}].hard_interrupt_state.traversal_observation_budget 缺失`);
      continue;
    }
    requireNonEmptyString(budgetState, "protocol_version", `gray_zone_path_decisions[${index}].hard_interrupt_state.traversal_observation_budget`, issues);
    for (const field of ["limit", "expanded_observation_count_before", "expanded_observation_count_after", "remaining_after"]) {
      requireNonNegativeInteger(budgetState, field, `gray_zone_path_decisions[${index}].hard_interrupt_state.traversal_observation_budget`, issues);
    }
    for (const field of ["cadence_due", "hard_interrupt_applied", "budget_exhausted_after"]) {
      if (typeof budgetState[field] !== "boolean") {
        issues.push(`gray_zone_path_decisions[${index}].hard_interrupt_state.traversal_observation_budget.${field} 缺失或不是布尔值`);
      }
    }
    if (!Object.prototype.hasOwnProperty.call(budgetState, "model_call_count") || budgetState.model_call_count !== 0) {
      issues.push(`gray_zone_path_decisions[${index}].hard_interrupt_state.traversal_observation_budget.model_call_count 必须显式等于 0`);
    }
  }

  const convergence = trace.convergence;
  if (!isRecord(convergence)) {
    issues.push("trace.convergence 缺失，无法审计 traversal observation budget");
  } else {
    requireNonNegativeInteger(convergence, "gray_zone_observation_cadence", "trace.convergence", issues);
    for (const field of [
      "traversal_observation_budget",
      "traversal_observation_expanded_count",
      "traversal_observation_budget_compacted_count",
      "traversal_observation_cadence_compacted_count",
      "traversal_observation_hard_interrupt_count",
    ]) {
      requireNonNegativeInteger(convergence, field, "trace.convergence", issues);
    }
    if (convergence.traversal_observation_budget === 0) {
      issues.push("trace.convergence.traversal_observation_budget 必须大于 0");
    }
    if (typeof convergence.traversal_observation_budget_hit !== "boolean") {
      issues.push("trace.convergence.traversal_observation_budget_hit 缺失或不是布尔值");
    }
    if (!nonEmptyRecord(convergence.traversal_observation_budget_audit)) {
      issues.push("trace.convergence.traversal_observation_budget_audit 缺失或为空");
    }
    const expanded = Number(convergence.traversal_observation_expanded_count);
    const budgetCompacted = Number(convergence.traversal_observation_budget_compacted_count);
    const cadenceCompacted = Number(convergence.traversal_observation_cadence_compacted_count);
    if (Number.isInteger(expanded) && Number.isInteger(budgetCompacted) && Number.isInteger(cadenceCompacted)
      && expanded + budgetCompacted + cadenceCompacted !== localDecisions.length) {
      issues.push("traversal observation expanded/compacted 计数与 local decision 数不一致");
    }
    if (Number.isInteger(budgetCompacted)
      && convergence.traversal_observation_hard_interrupt_count !== budgetCompacted) {
      issues.push("traversal observation hard interrupt 计数与 budget compacted 计数不一致");
    }
    if (Number.isInteger(budgetCompacted)
      && convergence.traversal_observation_budget_hit !== (budgetCompacted > 0)) {
      issues.push("traversal observation budget_hit 与 budget compacted 计数不一致");
    }
  }

  if (isRecord(determinism)) {
    if (determinism.checked_record_count !== decisions.length + thresholdHits.length) {
      issues.push("determinism checked_record_count 与 decision 记录数不一致");
    }
    if (determinism.local_rule_record_count !== localDecisions.length) {
      issues.push("determinism local_rule_record_count 与本地规则记录数不一致");
    }
    if (determinism.red_partition_record_count !== partitionDecisions.filter((decision) => decision.distance_zone === "red").length) {
      issues.push("determinism red_partition_record_count 与红区记录数不一致");
    }
    if (determinism.hard_stop_partition_record_count !== partitionDecisions.filter((decision) => decision.distance_zone === "hard_stop").length) {
      issues.push("determinism hard_stop_partition_record_count 与硬停记录数不一致");
    }
  }

  return {
    ok: issues.length === 0,
    issues,
    modelCallCount: trace.gray_zone_model_call_count === 0 ? 0 : undefined,
    localDecisions,
    partitionDecisions,
  };
}

export function GrayZoneAuditFailure({ title = "Gray-zone 审计失败", message, issues = [] }: { title?: string; message: string; issues?: string[] }) {
  return (
    <section
      data-testid="gray-zone-audit-failure"
      role="alert"
      className="rounded-2xl border border-rose-300/35 bg-rose-300/[0.08] p-4 text-rose-50 shadow-[0_0_28px_rgba(251,113,133,0.08)]"
    >
      <p className="text-sm font-semibold">{title}</p>
      <p className="mt-1 text-xs leading-5 text-rose-100/82">{message}</p>
      {issues.length ? (
        <ul className="mt-3 list-disc space-y-1 pl-5 font-mono text-[11px] leading-5 text-rose-100/72">
          {issues.map((issue) => <li key={issue}>{issue}</li>)}
        </ul>
      ) : null}
    </section>
  );
}

function AuditJson({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 text-[10px] uppercase tracking-[0.12em] text-white/34">{label}</p>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all border-l border-white/10 bg-black/15 p-2 font-mono text-[10px] leading-5 text-white/58 custom-scrollbar">
        {safeJson(value)}
      </pre>
    </div>
  );
}

export function QueryFacetPosteriorAudit({ calibration }: { calibration?: QueryFacetPosteriorCalibration | null }) {
  if (!calibration) {
    return (
      <GrayZoneAuditFailure
        title="Query facet posterior 审计缺失"
        message="当前 retrieval trace 没有版本化 prior / likelihood / posterior、bounded observations 或 budget/convergence 记录。"
      />
    );
  }
  return (
    <section data-testid="query-facet-posterior-audit" className="rounded-2xl border border-cyan-300/16 bg-cyan-300/[0.035] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-cyan-50">Query Facet Posterior（检索路由校准）</p>
          <p className="mt-1 text-xs leading-5 text-cyan-50/58">
            仅使用确定性有界候选观察；posterior 只在相同未覆盖 facet 数量内做 tie-break，不是事实、引用或 gray-zone authority。
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-[11px] text-cyan-50/72">
          <span className="rounded-full border border-cyan-300/20 px-2.5 py-1">rounds {calibration.rounds_used}/{calibration.round_budget}</span>
          <span className="rounded-full border border-cyan-300/20 px-2.5 py-1">observations {calibration.observations_used}/{calibration.observation_budget}</span>
          <span className="rounded-full border border-cyan-300/20 px-2.5 py-1">stop {calibration.stop_reason}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1 font-semibold text-emerald-100">model_call_count = {calibration.model_call_count}</span>
        </div>
      </div>
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <AuditJson label="prior / posterior / alias posterior" value={{ prior: calibration.prior, posterior: calibration.posterior, alias_posterior: calibration.alias_posterior }} />
        <AuditJson label="likelihood rounds / convergence" value={calibration.rounds} />
        <AuditJson label="bounded graph observations" value={calibration.observations} />
        <AuditJson label="authority / identity" value={{ protocol_version: calibration.protocol_version, protocol_hash: calibration.protocol_hash, calibration_hash: calibration.calibration_hash, llm_sample_budget: calibration.llm_sample_budget, is_evidence: calibration.is_evidence, citation_authority: calibration.citation_authority, graph_mutation_authority: calibration.graph_mutation_authority, gray_zone_decision_authority: calibration.gray_zone_decision_authority }} />
      </div>
    </section>
  );
}

function DecisionCard({ decision, index }: { decision: RetrievalGrayZoneDecision; index: number }) {
  const isLocal = decision.decision_source === "deterministic_local_rule";
  return (
    <article className="rounded-xl border border-white/8 bg-black/15 p-3 text-xs text-white/58">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-white/76">
        <span className="font-mono text-[10px] text-white/36">#{index + 1}</span>
        <span>{isLocal ? "本地规则" : "距离分区"}</span>
        <span>{decision.layer || "未分层"}</span>
        <code>{decision.edge_id || `${decision.from_node_id ?? decision.from_chunk_id} → ${decision.to_node_id ?? decision.to_chunk_id}`}</code>
        <span>zone {decision.distance_zone}</span>
        <span>distance {decision.path_distance}</span>
        <span>decision {decision.decision}</span>
        <span>rule {decision.matched_rule}</span>
        <span className="font-semibold text-emerald-100">model_call_count = {decision.model_call_count}</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-cyan-50/58">
        <span>semantic_uncertain={String(Boolean(decision.semantic_uncertain_edge))}</span>
        <span>crossing_rq_boundary={String(Boolean(decision.crossing_rq_boundary))}</span>
        <span>gray_candidate_reasons={decision.gray_candidate_reasons?.join(" / ") || "[]"}</span>
      </div>
      <dl className="mt-3 grid gap-x-4 gap-y-2 font-mono text-[10px] leading-5 text-white/54 xl:grid-cols-2">
        {[
          ["protocol_hash", decision.protocol_hash],
          ["input_hash", decision.input_hash],
          ["threshold_hash", decision.threshold_hash],
          ["traversal_protocol_hash", decision.traversal_protocol_hash],
          ["runtime_settings_hash", decision.runtime_settings_hash],
          ["agent_operating_envelope_hash", decision.agent_operating_envelope_hash],
          ["decision_hash", decision.decision_hash],
        ].map(([label, value]) => (
          <div key={label} className="min-w-0 border-l border-white/10 pl-2">
            <dt className="text-white/32">{label}</dt>
            <dd className="break-all text-white/62">{value}</dd>
          </div>
        ))}
      </dl>
      <details className="mt-3">
        <summary className="cursor-pointer text-[11px] text-cyan-100/62">hard state / minimum audit / support / observation</summary>
        <div className="mt-2 grid gap-3 xl:grid-cols-2">
          <AuditJson label="hard_interrupt_state" value={decision.hard_interrupt_state} />
          <AuditJson label="minimum_audit" value={decision.minimum_audit} />
          <AuditJson label="support_refs" value={decision.support_refs} />
          <AuditJson label="predicates / observation" value={{ predicates: decision.predicates, observation: decision.observation }} />
        </div>
      </details>
    </article>
  );
}

export function GrayZoneAuditDetails({ trace, className }: { trace: RetrievalTraceStepsResponse | unknown; className?: string }) {
  const inspection = inspectGrayZoneTrace(trace);
  if (!inspection.ok || !isRecord(trace) || !isRecord(trace.gray_zone_determinism)) {
    return (
      <GrayZoneAuditFailure
        message="完整检索轨迹缺少必填审计字段、包含非零模型调用，或 deterministic replay 审计未通过。该轨迹不能作为可验证结果展示。"
        issues={inspection.issues}
      />
    );
  }
  const determinism = trace.gray_zone_determinism;
  const convergence = isRecord(trace.convergence) ? trace.convergence : {};

  return (
    <section data-testid="gray-zone-audit-details" className={cn("rounded-2xl border border-emerald-300/16 bg-emerald-300/[0.045] p-4", className)}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-emerald-50">Deterministic Local Gray Rule（路径裁决）</p>
          <p className="mt-1 text-xs leading-5 text-emerald-50/62">
            Gray path 仅由 executor 的版本化本地规则裁决；LLM、Profile、Policy 单次输出和 Evidence Evaluator 均无此权限。
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-[11px]">
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">protocol {String(trace.gray_zone_protocol)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1 font-semibold">model_call_count = {inspection.modelCallCount}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">determinism {String(determinism.status)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">checked {String(determinism.checked_record_count)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">observation budget {String(convergence.traversal_observation_budget)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">expanded {String(convergence.traversal_observation_expanded_count)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">budget compacted {String(convergence.traversal_observation_budget_compacted_count)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">cadence compacted {String(convergence.traversal_observation_cadence_compacted_count)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">hard interrupt {String(convergence.traversal_observation_hard_interrupt_count)}</span>
          <span className={cn("rounded-full border px-2.5 py-1", convergence.traversal_observation_budget_hit ? "border-amber-300/25 text-amber-100" : "border-emerald-300/20")}>budget hit {String(convergence.traversal_observation_budget_hit)}</span>
          <span className="rounded-full border border-emerald-300/20 px-2.5 py-1">cadence {String(convergence.gray_zone_observation_cadence)}</span>
        </div>
      </div>

      <details className="mt-3">
        <summary className="cursor-pointer text-[11px] text-emerald-100/66">Traversal observation budget audit</summary>
        <div className="mt-2">
          <AuditJson label="traversal_observation_budget_audit" value={convergence.traversal_observation_budget_audit} />
        </div>
      </details>

      <div className="mt-4 grid gap-4">
        <div>
          <p className="mb-2 text-xs font-medium text-white/70">Gray candidates：deterministic_local_rule ({inspection.localDecisions.length})</p>
          <div className="space-y-2">
            {inspection.localDecisions.length
              ? inspection.localDecisions.map((decision, index) => <DecisionCard key={`${decision.decision_hash}:${index}`} decision={decision} index={index} />)
              : <p className="rounded-xl border border-white/7 bg-black/10 p-3 text-xs text-white/42">本次检索没有进入本地规则灰区的路径；零记录同样经过完整 determinism 审计。</p>}
          </div>
        </div>
        {inspection.partitionDecisions.length ? (
          <div>
            <p className="mb-2 text-xs font-medium text-white/70">Red / hard-stop：deterministic_distance_partition ({inspection.partitionDecisions.length})</p>
            <div className="space-y-2">
              {inspection.partitionDecisions.map((decision, index) => <DecisionCard key={`${decision.decision_hash}:${index}`} decision={decision} index={index} />)}
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export function EvidenceEvaluatorAudit({ trace, className }: { trace: AgentTraceEventPayload[]; className?: string }) {
  const events = trace.filter((event) => event.node === "evidence_evaluator");
  return (
    <section data-testid="evidence-evaluator-audit" className={cn("rounded-2xl border border-violet-300/14 bg-violet-300/[0.035] p-4", className)}>
      <p className="text-sm font-semibold text-violet-50">LLM Evidence Evaluator（证据充分性 / 重规划）</p>
      <p className="mt-1 text-xs leading-5 text-violet-50/58">
        Evaluator 只能判断证据是否充分并请求受约束的下一轮计划；不得裁决任何 gray path 的继续、停止、走桥、下钻或结构闭包。
      </p>
      <div className="mt-3 space-y-2">
        {events.length ? events.map((event, index) => {
          const scores = event.scores;
          const evaluatorScores =
            scores.audit_kind === "evidence_evaluator" ? scores : null;
          const verdict = evaluatorScores?.verdict;
          return (
            <article key={event.id ?? `${event.run_id ?? "run"}:${index}`} className="rounded-xl border border-white/7 bg-black/12 p-3 text-xs text-white/58">
              <div className="flex flex-wrap gap-3 text-white/74">
                <span>round {String(evaluatorScores?.plan_index ?? index)}</span>
                <span>verdict {String(verdict?.verdict ?? event.output_summary ?? "未记录")}</span>
                <span>replan {String(Boolean(evaluatorScores?.replan_requested))}</span>
                <span>duration {event.duration_ms} ms</span>
              </div>
              <div className="mt-2">
                <AuditJson label="typed evidence evaluation" value={{ reason: verdict?.reason, target_ids: verdict?.target_ids, expected_evidence: verdict?.expected_evidence, decision_hash: verdict?.decision_hash, prompt_protocol_hash: verdict?.prompt_protocol_hash }} />
              </div>
            </article>
          );
        }) : (
          <p className="rounded-xl border border-white/7 bg-black/10 p-3 text-xs text-white/42">当前流式 Agent trace 未携带 Evidence Evaluator 事件；请以上方 PostgreSQL P&amp;E audit 的 evaluator observation 为持久化事实。这不会授权 LLM 补判 gray path。</p>
        )}
      </div>
    </section>
  );
}
