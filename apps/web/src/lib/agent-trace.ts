import type { AgentTraceNode } from "@course-kg/shared";

export const contextGraphTraceFallbackSteps: AgentTraceNode[] = [
  "query_understanding",
  "agent_planner",
  "typed_action_validation",
  "coarse_concept_activation",
  "mid_concept_routing",
  "fine_cluster_routing",
  "chunk_recall",
  "structure_context_restoration",
  "context_package",
  "grounded_answer",
  "citation_verification",
  "reward_event",
];

const traceNodeLabels: Record<AgentTraceNode, string> = {
  query_understanding: "查询意图",
  agent_planner: "智能体规划",
  typed_action_validation: "动作校验",
  coarse_concept_activation: "粗粒度概念激活",
  mid_concept_routing: "中粒度概念路由",
  fine_cluster_routing: "细聚类/RQ 路由",
  chunk_recall: "片段召回",
  structure_context_restoration: "结构上下文恢复",
  layered_retrieval: "分层检索",
  context_package: "上下文证据包",
  grounded_answer: "有支撑回答",
  citation_verification: "引用验证",
  repair_executed: "修复执行",
  reward_event: "奖励事件",
  error: "错误",
};

export function traceNodeLabel(node: string): string {
  if (node === "retrievers") {
    return "片段召回";
  }
  return traceNodeLabels[node as AgentTraceNode] ?? node;
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
    node === "coarse_concept_activation" ||
    node === "mid_concept_routing" ||
    node === "fine_cluster_routing" ||
    node === "chunk_recall" ||
    node === "structure_context_restoration" ||
    node === "layered_retrieval" ||
    node === "context_package"
  ) {
    return "info";
  }
  return "success";
}

export function traceAuditSummary(scores: Record<string, unknown> | undefined): string[] {
  const audit = scores?.audit;
  if (!audit || typeof audit !== "object") {
    return [];
  }
  const data = audit as Record<string, unknown>;
  const entries: Array<[string, string]> = [
    ["coarse_concepts", "粗概念"],
    ["mid_concepts", "中概念"],
    ["fine_clusters", "细聚类"],
    ["query_rq_path", "RQ 路径"],
    ["base_candidate_count", "基础候选"],
    ["recalled_chunks", "召回片段"],
    ["structure_neighbors", "结构邻居"],
    ["bridge_chunks", "桥接片段"],
    ["context_chunks", "上下文片段"],
    ["citation_count", "引用"],
    ["verification_pass_rate", "验证通过率"],
    ["retrieval_trace_id", "检索轨迹"],
    ["context_package_id", "证据包"],
    ["plan_id", "规划"],
  ];
  return entries
    .filter(([key]) => data[key] !== undefined && data[key] !== null)
    .map(([key, label]) => `${label}: ${Array.isArray(data[key]) ? data[key].join("/") : String(data[key])}`);
}
