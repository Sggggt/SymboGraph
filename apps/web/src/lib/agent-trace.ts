import type { AgentTraceEventPayload, AgentTraceNode } from "@course-kg/shared";

export const contextGraphTraceFallbackSteps: AgentTraceNode[] = [
  "query_understanding",
  "query_facet_extraction",
  "agent_planner",
  "typed_action_validation",
  "entry_selection",
  "layer_drilldown",
  "frontier_traversal",
  "chunk_recall",
  "structure_context_restoration",
  "context_package",
  "grounded_answer",
  "citation_verification",
  "reward_event",
];

const traceNodeLabels: Record<AgentTraceNode, string> = {
  query_understanding: "查询意图",
  query_facet_extraction: "查询 facets",
  agent_planner: "智能体规划",
  typed_action_validation: "动作校验",
  entry_selection: "分阶段入口",
  layer_drilldown: "逐父下钻",
  frontier_traversal: "队列遍历",
  chunk_recall: "片段 TopK",
  structure_context_restoration: "结构上下文恢复",
  layered_retrieval: "分层检索",
  context_package: "证据包",
  grounded_answer: "有支撑回答",
  citation_verification: "引用验证",
  repair_executed: "修复执行",
  reward_event: "奖励观测",
  error: "错误",
};

export const traceGroupLabels = {
  entry: "分阶段入口",
  drilldown: "逐父下钻",
  frontier: "队列遍历",
  restoration: "结构恢复",
  package: "证据包",
  verification: "引用验证",
  repair: "修复",
  reward: "奖励观测",
  answer: "回答生成",
  other: "其他",
} as const;

export type TraceGroupKey = keyof typeof traceGroupLabels;

export function traceNodeLabel(node: string): string {
  if (node === "retrievers") {
    return "片段召回";
  }
  return traceNodeLabels[node as AgentTraceNode] ?? node;
}

export function traceGroupForNode(node: string): TraceGroupKey {
  if (node === "query_understanding" || node === "query_facet_extraction" || node === "agent_planner" || node === "typed_action_validation" || node === "entry_selection") {
    return "entry";
  }
  if (node === "layer_drilldown" || node === "layered_retrieval") {
    return "drilldown";
  }
  if (node === "frontier_traversal" || node === "chunk_recall" || node === "retrievers") {
    return "frontier";
  }
  if (node === "structure_context_restoration") {
    return "restoration";
  }
  if (node === "context_package") {
    return "package";
  }
  if (node === "citation_verification") {
    return "verification";
  }
  if (node === "repair_executed") {
    return "repair";
  }
  if (node === "reward_event") {
    return "reward";
  }
  if (node === "grounded_answer") {
    return "answer";
  }
  return "other";
}

export function groupTraceEvents(trace: AgentTraceEventPayload[]): Array<{ key: TraceGroupKey; label: string; events: AgentTraceEventPayload[] }> {
  const order: TraceGroupKey[] = ["entry", "drilldown", "frontier", "restoration", "package", "answer", "verification", "repair", "reward", "other"];
  const grouped = new Map<TraceGroupKey, AgentTraceEventPayload[]>();
  for (const event of trace) {
    const key = traceGroupForNode(event.node);
    const events = grouped.get(key) ?? [];
    events.push(event);
    grouped.set(key, events);
  }
  return order
    .map((key) => ({ key, label: traceGroupLabels[key], events: grouped.get(key) ?? [] }))
    .filter((group) => group.events.length > 0);
}

export function traceNodeVariant(node: string): "success" | "info" | "warning" | "danger" {
  if (node === "error") {
    return "danger";
  }
  if (node === "citation_verification" || node === "reward_event" || node === "typed_action_validation" || node === "repair_executed") {
    return "warning";
  }
  if (
    node === "agent_planner" ||
    node === "entry_selection" ||
    node === "layer_drilldown" ||
    node === "frontier_traversal" ||
    node === "chunk_recall" ||
    node === "structure_context_restoration" ||
    node === "layered_retrieval" ||
    node === "context_package"
  ) {
    return "info";
  }
  return "success";
}

function formatAuditValue(key: string, value: unknown): string {
  if (key === "retrieval_granularity") {
    if (value === "mid") return "普通模式";
    if (value === "coarse") return "摘要模式";
  }
  return Array.isArray(value) ? value.join("/") : String(value);
}

export function traceAuditSummary(scores: Record<string, unknown> | undefined): string[] {
  const audit = scores?.audit && typeof scores.audit === "object" ? (scores.audit as Record<string, unknown>) : undefined;
  const data = audit ?? scores ?? {};
  const entries: Array<[string, string]> = [
    ["retrieval_granularity", "检索模式"],
    ["coarse_concepts", "粗概念"],
    ["coarse_entries", "粗入口"],
    ["mid_concepts", "中概念"],
    ["mid_entries", "中入口"],
    ["stage_queue_count", "Stage 队列"],
    ["mid_topk_selected", "中概念 TopK"],
    ["chunk_topk_selected", "片段 TopK"],
    ["rq_prefixes", "RQ 前缀"],
    ["rq_membership_entries", "RQ 归属"],
    ["frontier_pops", "Frontier pop"],
    ["frontier_expansion_count", "扩展边数"],
    ["gray_zone_decision_count", "灰区观测"],
    ["red_zone_pruned_count", "红区剪枝"],
    ["hard_stop_pruned_count", "硬停剪枝"],
    ["dominance_pruned_count", "支配剪枝"],
    ["convergence_reason", "收敛原因"],
    ["query_rq_path", "RQ 路径"],
    ["base_candidate_count", "基础候选"],
    ["recalled_chunks", "召回片段"],
    ["structure_neighbors", "结构邻居"],
    ["bridge_chunks", "桥接片段"],
    ["context_chunks", "上下文片段"],
    ["citation_count", "引用"],
    ["citation_pass_rate", "引用通过率"],
    ["verification_pass_rate", "验证通过率"],
    ["retrieval_trace_id", "检索轨迹"],
    ["context_package_id", "证据包"],
    ["plan_id", "规划"],
  ];
  return entries
    .filter(([key]) => data[key] !== undefined && data[key] !== null)
    .map(([key, label]) => `${label}: ${formatAuditValue(key, data[key])}`);
}
