import type { IngestionLogEvent } from "@course-kg/shared";

export type LogVisualTone = "graph" | "warning" | "failure" | "default";

export const logEventLabels: Record<string, string> = {
  batch_started: "批次开始",
  batch_files: "文件扫描完成",
  file_started: "开始解析文件",
  file_indexed: "向量索引已写入",
  file_completed: "文件解析完成",
  file_failed: "文件解析失败",
  file_skipped: "文件跳过",
  batch_progress: "批次进度",
  batch_graph_started: "四层图谱开始",
  batch_graph_selected: "图谱候选选择",
  batch_graph_progress: "四层图谱进度",
  graph_rebuilt: "四层图谱重建完成",
  graph_failed: "四层图谱失败",
  context_graph_started: "四层上下文图谱开始",
  chunk_structure_completed: "结构图完成",
  chunk_relation_completed: "片段关系图完成",
  rq_prefixes_completed: "RQ membership 层完成",
  mid_concepts_completed: "中粒度概念完成",
  coarse_concepts_completed: "粗粒度概念完成",
  context_graph_completed: "上下文图谱完成",
  context_graph_failed: "上下文图谱失败",
  batch_completed: "批次完成",
  batch_partial_failed: "部分失败",
  batch_failed: "批次失败",
  batch_skipped: "批次跳过",
  batch_cancelled: "批次已取消",
  batch_cancel_requested: "请求取消",
  batch_cancel_targeted_task: "终止目标任务",
  batch_cancel_failed: "取消失败",
  batch_missing: "批次丢失",
  cleanup_started: "清理开始",
  cleanup_completed: "清理完成",
  compensation_started: "补偿开始",
  compensation_completed: "补偿完成",
  log_stream_retry: "日志流重连",
  log_stream_recovered: "日志流恢复",
  log_stream_warning: "日志流告警",
  auto_tpe_started: "自动 TPE 开始",
  auto_tpe_trial_started: "自动 TPE Trial 开始",
  auto_tpe_trial_completed: "自动 TPE Trial 完成",
  auto_tpe_trial_blocked: "自动 TPE Trial 阻断",
  auto_tpe_best_theta_selected: "自动 TPE 最佳参数",
  auto_tpe_skipped: "自动 TPE 跳过",
  auto_tpe_failed: "自动 TPE 失败",
};

export function logEventLabel(event: string): string {
  return logEventLabels[event] ?? `未知事件：${event.replaceAll("_", " ")}`;
}

export function logVisualTone(item: Pick<IngestionLogEvent, "event" | "stage" | "phase" | "translation_phase">): LogVisualTone {
  if (item.event.includes("failed") || item.event.includes("cancel_failed")) {
    return "failure";
  }
  if (item.event === "log_stream_retry" || item.event.includes("warning")) {
    return "warning";
  }
  if (
    item.event.includes("context_graph") ||
    item.event.includes("chunk_") ||
    item.event.includes("concept") ||
    item.event.includes("prefix") ||
    item.event.includes("membership") ||
    item.event.includes("graph") ||
    item.event.includes("tpe_") ||
    item.event.includes("compensation") ||
    item.phase?.includes("context_graph") ||
    Boolean(item.translation_phase)
  ) {
    return "graph";
  }
  return "default";
}

export function graphLogSummary(item: IngestionLogEvent): string | null {
  const phase = String(item.phase ?? item.stage ?? "");
  const isGraphEvent =
    item.event.includes("context_graph") ||
    item.event.includes("chunk_") ||
    item.event.includes("concept") ||
    item.event.includes("prefix") ||
    item.event.includes("membership") ||
    item.event.includes("graph") ||
    item.event.includes("tpe_") ||
    item.event === "file_indexed" ||
    phase.includes("context_graph");
  if (!isGraphEvent) {
    return null;
  }

  const phaseLabels: Record<string, string> = {
    parsing: "解析",
    chunking: "固定切块",
    embedding: "向量索引",
    "context_graph:starting": "上下文图谱初始化",
    "context_graph:chunk_relation": "片段关系图",
    "context_graph:chunk_relation:chunk_edges": "关系边生成",
    "context_graph:chunk_relation:rq_prefixes": "RQ 前缀与归属",
    "context_graph:mid_concepts": "中粒度概念",
    "context_graph:coarse_concepts": "粗粒度概念",
    "context_graph:context_state": "Context Graph 状态",
    "context_graph:completed": "图谱闭环完成",
    compensating: "清理/补偿",
  };
  const translationPhaseLabels: Record<string, string> = {
    concept_i18n: "节点双语派生",
    edge_i18n: "关系双语派生",
  };

  const parts: string[] = [];
  const contextGraphPhase = item.context_graph_phase ? `context_graph:${item.context_graph_phase}` : "";
  const phaseLabel = phaseLabels[phase] ?? phaseLabels[contextGraphPhase] ?? null;
  if (phaseLabel) {
    parts.push(`阶段 ${phaseLabel}`);
  }
  const translationPhase = item.translation_phase ?? "";
  const translationLabel = translationPhaseLabels[translationPhase] ?? null;
  if (translationLabel) {
    parts.push(`双语 ${translationLabel}`);
  }
  if (item.translation_status === "disabled" || item.translation_enabled === false) {
    parts.push("状态 未启用");
  }
  const pushNumber = (label: string, value: number | undefined) => {
    if (typeof value === "number") {
      parts.push(`${label} ${value}`);
    }
  };
  pushNumber("片段", item.chunk_count);
  pushNumber("向量", item.vector_count);
  pushNumber("关系边", item.relation_edge_count);
  pushNumber("RQ membership", item.rq_prefix_count);
  pushNumber("中概念", item.mid_concept_count);
  pushNumber("粗概念", item.coarse_concept_count);
  pushNumber("派生项", item.translation_items);
  pushNumber("已翻译", item.translated_count);
  pushNumber("回退", item.fallback_count);
  pushNumber("节点翻译", item.concept_i18n_translated_count);
  pushNumber("节点回退", item.concept_i18n_fallback_count);
  pushNumber("关系翻译", item.edge_i18n_translated_count);
  pushNumber("关系回退", item.edge_i18n_fallback_count);
  if (item.context_graph_hash) {
    parts.push(`哈希 ${item.context_graph_hash.slice(0, 8)}`);
  }
  if (item.event.startsWith("auto_tpe_")) {
    if (typeof item.trial_index === "number") {
      parts.push(`Trial ${item.trial_index}`);
    }
    if (typeof item.objective_score === "number") {
      parts.push(`Objective ${item.objective_score.toFixed(4)}`);
    }
    if (item.theta_hash) {
      parts.push(`参数 ${item.theta_hash.slice(0, 8)}`);
    }
    if (item.probe_set_hash) {
      parts.push(`Probe ${item.probe_set_hash.slice(0, 8)}`);
    }
    if (item.failure_code) {
      parts.push(`阻断 ${item.failure_code}`);
    }
    const hardGate = item.hard_gate && typeof item.hard_gate === "object" ? item.hard_gate : null;
    if (hardGate) {
      const failed = Object.entries(hardGate)
        .filter(([, value]) => typeof value === "object" && value !== null && (value as { passed?: unknown }).passed === false)
        .map(([key]) => key);
      parts.push(failed.length ? `Hard gate 未通过 ${failed.join(", ")}` : "Hard gate 通过");
    }
  }
  return parts.length ? parts.join(" / ") : null;
}
