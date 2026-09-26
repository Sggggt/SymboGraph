import type { AgentTraceEventPayload, AgentTraceNode, AgentTraceScores } from "@course-kg/shared";

export const contextGraphTraceFallbackSteps: AgentTraceNode[] = [
  "retrieval_control",
];

const traceNodeLabels: Record<AgentTraceNode, string> = {
  retrieval_control: "检索与回答进度",
  query_understanding: "查询意图",
  query_facet_extraction: "查询 facets",
  agent_planner: "智能体规划",
  typed_action_validation: "动作校验",
  typed_action_executor: "动作执行",
  evidence_evaluator: "证据充分性评估",
  replan_no_progress: "停止重复规划",
  evidence_gate: "证据门禁",
  entry_selection: "分阶段入口",
  layer_drilldown: "逐父下钻",
  frontier_traversal: "队列遍历",
  chunk_recall: "片段 TopK",
  structure_context_restoration: "结构上下文恢复",
  layered_retrieval: "分层检索",
  context_package: "证据包",
  grounded_answer: "有支撑回答",
  answer_generation: "生成回答并自评",
  reflection_gate: "判断是否需要反思",
  answer_reflection: "检查回答",
  reflection_backtrack: "补充证据或改写回答",
  reflection_action_validation: "检查后续动作",
  answer_source_binding: "核对原文来源",
  citation_verification: "引用验证",
  repair_executed: "修复执行",
  reward_event: "奖励观测",
  direct_answer_route_gate: "直接回答路由",
  direct_answer_reuse_evaluator: "历史证据复用评估",
  direct_answer_fallback: "直接回答回退",
  direct_answer: "直接回答",
  cancelled: "已取消",
  agent_admission: "Agent 准入",
  error: "错误",
  intent_planning: "理解任务并冻结计划",
  planning_resource_titles: "查看粗节点标题",
  planning_resource_details: "阅读粗节点摘要",
  planning_schema_feedback: "修正规划格式",
  intent_execution_retrieval: "执行分层检索",
  source_integrity_admission: "核对来源完整性",
  verified_context_reuse: "复用已验证证据",
  evidence_directory_ready: "准备证据目录",
  evidence_read: "读取候选原文",
  evidence_finalized: "冻结回答证据",
};

export const traceGroupLabels = {
  entry: "分阶段入口",
  drilldown: "逐父下钻",
  frontier: "队列遍历",
  restoration: "结构恢复",
  package: "证据包",
  verification: "引用验证",
  reflection: "回答反思",
  source_binding: "原文来源",
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
  if (["reflection_gate", "answer_reflection", "reflection_backtrack", "reflection_action_validation"].includes(node)) {
    return "reflection";
  }
  if (node === "answer_source_binding") {
    return "source_binding";
  }
  if (node === "query_understanding" || node === "query_facet_extraction" || node === "agent_planner" || node === "typed_action_validation" || node === "replan_no_progress" || node === "entry_selection" || node === "direct_answer_route_gate" || node === "direct_answer_reuse_evaluator" || node === "direct_answer_fallback" || node === "intent_planning" || node === "planning_resource_titles" || node === "planning_resource_details" || node === "planning_schema_feedback") {
    return "entry";
  }
  if (node === "layer_drilldown" || node === "layered_retrieval" || node === "intent_execution_retrieval") {
    return "drilldown";
  }
  if (node === "frontier_traversal" || node === "chunk_recall" || node === "retrievers") {
    return "frontier";
  }
  if (node === "structure_context_restoration") {
    return "restoration";
  }
  if (node === "context_package" || node === "evidence_directory_ready" || node === "evidence_read" || node === "evidence_finalized") {
    return "package";
  }
  if (node === "citation_verification" || node === "source_integrity_admission") {
    return "verification";
  }
  if (node === "repair_executed") {
    return "repair";
  }
  if (node === "reward_event") {
    return "reward";
  }
  if (node === "grounded_answer" || node === "answer_generation" || node === "direct_answer") {
    return "answer";
  }
  return "other";
}

export function groupTraceEvents(trace: AgentTraceEventPayload[]): Array<{ key: TraceGroupKey; label: string; events: AgentTraceEventPayload[] }> {
  const order: TraceGroupKey[] = ["entry", "drilldown", "frontier", "restoration", "package", "answer", "reflection", "source_binding", "verification", "repair", "reward", "other"];
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
    node === "context_package" ||
    node === "intent_planning" ||
    node === "planning_resource_titles" ||
    node === "planning_resource_details" ||
    node === "planning_schema_feedback" ||
    node === "intent_execution_retrieval" ||
    node === "source_integrity_admission" ||
    node === "verified_context_reuse" ||
    node === "evidence_directory_ready" ||
    node === "evidence_read" ||
    node === "evidence_finalized"
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
  if (key === "entry_layer") {
    return value === "coarse" ? "粗概念" : value === "mid" ? "中概念" : value === "chunk" ? "原文片段" : String(value);
  }
  return Array.isArray(value) ? value.join("/") : String(value);
}

export function traceAuditSummary(scores: AgentTraceScores | undefined): string[] {
  if (!scores) return [];
  let entries: Array<[string, string, unknown]> = [];
  switch (scores.audit_kind) {
    case "query_understanding":
      entries = [
        ["retrieval_granularity", "检索模式", scores.retrieval_granularity],
        ["top_k", "结果上限", scores.top_k],
      ];
      break;
    case "query_facets":
      entries = [
        ["retrieval_granularity", "检索模式", scores.retrieval_granularity],
      ];
      break;
    case "planner":
      entries = [
        ["retrieval_granularity", "检索模式", scores.retrieval_granularity],
        ["plan_id", "规划", scores.plan_id],
        ["plan_index", "规划轮次", scores.plan_index],
      ];
      break;
    case "typed_action_validation":
    case "typed_action_executor":
    case "evidence_evaluator":
      entries = [
        ["plan_id", "规划", scores.plan_id],
        ["plan_index", "规划轮次", scores.plan_index],
        [
          "retrieval_trace_id",
          "检索轨迹",
          scores.audit_kind === "typed_action_executor"
            ? scores.retrieval_trace_id
            : undefined,
        ],
      ];
      break;
    case "retrieval_stage":
      entries = [
        ["retrieval_granularity", "检索模式", scores.retrieval_granularity],
        ["coarse_entries", "粗入口", scores.coarse_entries],
        ["stage_queue_count", "Stage 队列", scores.stage_queue_count],
        ["mid_topk_selected", "中概念 TopK", scores.mid_topk_selected],
        ["chunk_topk_selected", "片段 TopK", scores.chunk_topk_selected],
        ["frontier_pops", "Frontier pop", scores.frontier_pops],
        ["dominance_pruned_count", "支配剪枝", scores.dominance_pruned_count],
        ["query_rq_path", "RQ 路径", scores.query_rq_path],
        ["retrieval_trace_id", "检索轨迹", scores.retrieval_trace_id],
      ];
      break;
    case "layered_retrieval":
      entries = [
        [
          "retrieval_granularity",
          "检索模式",
          scores.retrieval_audit?.retrieval_granularity,
        ],
        [
          "frontier_pops",
          "Frontier pop",
          scores.retrieval_audit?.frontier_pops,
        ],
        [
          "retrieval_trace_id",
          "检索轨迹",
          scores.retrieval_audit?.retrieval_trace_id,
        ],
      ];
      break;
    case "context_restoration":
      entries = [
        ["hit_chunks", "命中片段", scores.hit_chunks],
        ["restored_chunks", "恢复片段", scores.restored_chunks],
        ["bridge_chunks", "桥接片段", scores.bridge_chunks],
        ["context_package_id", "证据包", scores.context_package_id],
      ];
      break;
    case "context_package":
      entries = [
        ["context_package_id", "证据包", scores.context_package_id],
        ["token_count", "Token", scores.token_count],
      ];
      break;
    case "intent_execution":
      entries = [
        ["planning_round", "规划轮次", scores.planning_round],
        ["coarse_node_count", "粗节点", scores.coarse_node_count],
        ["model_duration_ms", "模型等待", scores.model_duration_ms == null ? undefined : `${scores.model_duration_ms} ms`],
        ["local_read_duration_ms", "本地读取", scores.local_read_duration_ms == null ? undefined : `${scores.local_read_duration_ms} ms`],
        ["schema_feedback_error_count", "格式问题", scores.schema_feedback_error_count],
        ["entry_layer", "入口层", scores.entry_layer],
        ["accepted_plan_hash", "计划", scores.accepted_plan_hash],
        ["model_call_count", "模型调用", scores.model_call_count],
        ["source_count", "候选来源", scores.source_count],
        ["read_count", "本轮读取", scores.read_count],
        ["selected_count", "已选来源", scores.selected_count],
        ["remaining_count", "剩余节点", scores.remaining_count],
        ["coverage_count", "已覆盖要求", scores.coverage_count],
        ["decision_call_count", "证据决策", scores.decision_call_count],
        ["read_action_count", "读取动作", scores.read_action_count],
        ["mid_count", "语义节点", scores.mid_count],
        ["mandatory_source_count", "必带来源", scores.mandatory_source_count],
        ["estimated_input_tokens", "估算输入 token", scores.estimated_input_tokens],
        ["input_token_count", "实际输入 token", scores.input_token_count],
      ];
      break;
    case "citation_verification":
      entries = [
        ["citation_pass_rate", "引用通过率", scores.citation_pass_rate],
        [
          "raw_citation_pass_rate",
          "原始引用通过率",
          scores.raw_citation_pass_rate,
        ],
        ["returned_citation_count", "返回引用", scores.returned_citation_count],
      ];
      break;
    default:
      break;
  }
  return entries
    .filter(([, , value]) => value !== undefined && value !== null)
    .map(([key, label, value]) => `${label}: ${formatAuditValue(key, value)}`);
}
