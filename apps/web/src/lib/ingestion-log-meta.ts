import type { IngestionLogEvent } from "@course-kg/shared";

export type LogVisualTone = "graph" | "warning" | "failure" | "default";

export const logEventLabels: Record<string, string> = {
  batch_started: "批次开始",
  batch_files: "文件扫描完成",
  file_started: "文件开始解析",
  file_indexed: "上下文索引已写入",
  file_completed: "文件完成",
  file_failed: "文件失败",
  file_skipped: "文件跳过",
  batch_progress: "批次进度",
  context_graph_started: "上下文图谱构建",
  chunk_structure_completed: "片段结构图完成",
  chunk_relation_completed: "片段关系图完成",
  fine_clusters_completed: "细聚类完成",
  mid_concepts_completed: "中粒度概念完成",
  coarse_concepts_completed: "粗粒度概念完成",
  context_graph_completed: "上下文图谱完成",
  batch_completed: "批次完成",
  batch_partial_failed: "部分失败",
  batch_failed: "批次失败",
  batch_skipped: "批次跳过",
  batch_cancelled: "批次已取消",
  batch_cancel_requested: "请求取消",
  batch_cancel_targeted_task: "取消目标任务",
  batch_cancel_failed: "取消失败",
  batch_missing: "批次丢失",
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
  if (item.event.includes("context_graph") || item.event.includes("chunk_") || item.event.includes("concept") || item.event.includes("cluster")) {
    return "graph";
  }
  return "default";
}

export function graphLogSummary(item: IngestionLogEvent): string | null {
  const isGraphEvent =
    item.event.includes("context_graph") ||
    item.event.includes("chunk_") ||
    item.event.includes("concept") ||
    item.event.includes("cluster") ||
    item.event === "file_indexed";
  if (!isGraphEvent) {
    return null;
  }
  const parts: string[] = [];
  const pushNumber = (label: string, value: number | undefined) => {
    if (typeof value === "number") {
      parts.push(`${label} ${value}`);
    }
  };
  pushNumber("片段", item.chunk_count);
  pushNumber("向量", item.vector_count);
  pushNumber("关系边", item.relation_edge_count);
  pushNumber("细聚类", item.fine_cluster_count);
  pushNumber("中粒度概念", item.mid_concept_count);
  pushNumber("粗粒度概念", item.coarse_concept_count);
  if (item.context_graph_hash) {
    parts.push(`哈希 ${item.context_graph_hash.slice(0, 8)}`);
  }
  return parts.length ? parts.join(" / ") : null;
}
