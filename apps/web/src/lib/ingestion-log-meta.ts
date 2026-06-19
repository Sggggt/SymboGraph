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
  log_stream_warning: "日志流告警",
};

export function logEventLabel(event: string): string {
  return logEventLabels[event] ?? `未知事件：${event.replaceAll("_", " ")}`;
}

export function logVisualTone(item: Pick<IngestionLogEvent, "event" | "stage" | "phase">): LogVisualTone {
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
    item.event.includes("compensation") ||
    item.phase?.includes("context_graph")
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

  const parts: string[] = [];
  const phaseLabel = phaseLabels[phase] ?? phaseLabels[String(item.context_graph_phase ?? "")] ?? null;
  if (phaseLabel) {
    parts.push(`阶段 ${phaseLabel}`);
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
  if (item.context_graph_hash) {
    parts.push(`哈希 ${item.context_graph_hash.slice(0, 8)}`);
  }
  return parts.length ? parts.join(" / ") : null;
}
