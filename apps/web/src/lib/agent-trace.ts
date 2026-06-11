import type { AgentTraceNode } from "@course-kg/shared";

export const evidenceFirstTraceFallbackSteps: AgentTraceNode[] = [
  "perception",
  "retrieval_planner",
  "base_retrieval",
  "evidence_anchor_selector",
  "evidence_chain_planner",
  "controlled_graph_enhancer",
  "evidence_assembler",
  "document_grader",
  "evidence_evaluator",
  "context_synthesizer",
  "answer_generator",
  "citation_checker",
  "citation_verifier",
  "reflection",
  "self_check",
];

const traceNodeLabels: Record<AgentTraceNode, string> = {
  perception: "感知",
  retrieval_planner: "检索规划",
  base_retrieval: "基础召回",
  evidence_anchor_selector: "锚点选择",
  evidence_chain_planner: "证据链规划",
  controlled_graph_enhancer: "受控信号投影",
  evidence_assembler: "证据装配",
  document_grader: "文档筛选",
  evidence_evaluator: "证据充分性",
  context_synthesizer: "上下文合成",
  answer_generator: "答案生成",
  citation_checker: "引用检查",
  citation_verifier: "引用验证",
  reflection: "反思",
  self_check: "自检",
  error: "错误",
};

export function traceNodeLabel(node: string): string {
  if (node === "retrievers") {
    return "基础召回";
  }
  return traceNodeLabels[node as AgentTraceNode] ?? node;
}

export function traceNodeVariant(node: string): "success" | "info" | "warning" | "danger" {
  if (node === "error") {
    return "danger";
  }
  if (node === "evidence_evaluator" || node === "citation_checker" || node === "citation_verifier" || node === "self_check") {
    return "warning";
  }
  if (node === "base_retrieval" || node === "evidence_anchor_selector" || node === "evidence_chain_planner" || node === "controlled_graph_enhancer" || node === "evidence_assembler") {
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
    ["base_candidate_count", "base"],
    ["anchor_count", "anchors"],
    ["planned_paths", "paths"],
    ["observed_edges", "observed"],
    ["graph_enhanced_chunks", "graph chunks"],
    ["expanded_active_chunks", "expanded active"],
    ["path_evidence_chunks", "path chunks"],
    ["discarded_candidate_edges", "discarded"],
    ["community_summaries", "communities"],
    ["community_id", "community"],
    ["chunk_retained", "retained chunks"],
    ["chunk_discarded", "discarded chunks"],
    ["retrieval_eligible_chunks", "retrieval route"],
    ["evidence_only_chunks", "evidence only"],
  ];
  return entries
    .filter(([key]) => data[key] !== undefined && data[key] !== null)
    .map(([key, label]) => `${label}: ${String(data[key])}`);
}
