import type { IngestionLogEvent } from "@course-kg/shared";

export type LogVisualTone = "adaptive" | "graph" | "hpo" | "warning" | "failure" | "default";

export const logEventLabels: Record<string, string> = {
  batch_started: "批次开始",
  batch_files: "文件扫描",
  file_started: "开始解析",
  job_state: "任务状态",
  file_skipped: "跳过文件",
  file_completed: "文件完成",
  file_failed: "文件失败",
  batch_graph_started: "图谱生成",
  batch_graph_selected: "图谱片段选择",
  batch_graph_plan_created: "图谱计划",
  batch_graph_probe_started: "图谱探针",
  batch_graph_progress: "图谱抽取进度",
  batch_graph_adaptive_round_started: "图谱抽取轮次",
  batch_graph_coverage_updated: "图谱覆盖更新",
  batch_graph_community_summary: "社区摘要进度",
  graph_upsert_started: "图谱写入开始",
  graph_upsert_completed: "图谱写入完成",
  graph_enrichment_started: "拓扑刷新开始",
  graph_enrichment_completed: "拓扑刷新完成",
  graph_community_started: "社区摘要开始",
  graph_community_completed: "社区摘要完成",
  graph_rebuilt: "图谱完成",
  graph_failed: "图谱失败",
  batch_completed: "批次完成",
  batch_partial_failed: "部分失败",
  batch_failed: "批次失败",
  batch_skipped: "批次跳过",
  batch_missing: "批次丢失",
  log_stream_retry: "日志重连",
  chunk_adaptive: "分块自适应",
  hpo_started: "自动调参开始",
  hpo_objective_features_started: "HPO 特征开始",
  hpo_objective_features_completed: "HPO 特征完成",
  hpo_judge_started: "Judge HPO 开始",
  hpo_judge_progress: "Judge HPO 进度",
  hpo_judge_completed: "Judge HPO 完成",
  hpo_judge_failed: "Judge HPO 失败",
  hpo_objective_training_started: "目标函数训练开始",
  hpo_objective_training_completed: "目标函数训练完成",
  hpo_objective_training_failed: "目标函数训练失败",
  hpo_tpe_started: "TPE 开始",
  hpo_tpe_progress: "TPE 进度",
  hpo_tpe_completed: "TPE 完成",
  hpo_completed: "自动调参完成",
  hpo_failed: "自动调参失败",
};

export function logEventLabel(event: string): string {
  return logEventLabels[event] ?? event.replaceAll("_", " ");
}

export function logVisualTone(item: Pick<IngestionLogEvent, "event" | "stage">): LogVisualTone {
  if (item.event.includes("failed") || item.event === "graph_failed") {
    return "failure";
  }
  if (item.event === "log_stream_retry" || item.event.includes("warning")) {
    return "warning";
  }
  if (item.event === "chunk_adaptive") {
    return "adaptive";
  }
  if (item.event.startsWith("hpo_") || item.stage === "judge" || item.stage === "tpe" || item.stage === "objective_training") {
    return "hpo";
  }
  if (item.event.startsWith("batch_graph_") || item.event.startsWith("graph_")) {
    return "graph";
  }
  return "default";
}

function formatNumber(value: unknown, digits = 3): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

export function hpoLogSummary(item: IngestionLogEvent): string | null {
  if (!item.event.startsWith("hpo_")) {
    return null;
  }
  const parts: string[] = [];
  if (typeof item.candidate_count === "number") {
    parts.push(`候选 ${item.candidate_count}`);
  }
  if (typeof item.processed_pairs === "number" || typeof item.pair_count === "number") {
    parts.push(`Judge ${item.processed_pairs ?? 0}/${item.pair_count ?? "?"}`);
  }
  if (typeof item.effective_labels === "number" || typeof item.min_labels === "number") {
    parts.push(`有效标签 ${item.effective_labels ?? 0}/${item.min_labels ?? "?"}`);
  }
  if (typeof item.trial_count === "number") {
    parts.push(`TPE ${item.trial_count}`);
  }
  const bestValue = formatNumber(item.best_value);
  if (bestValue !== null) {
    parts.push(`best ${bestValue}`);
  }
  if (item.objective_model_id) {
    parts.push(`目标 ${item.objective_model_id.slice(0, 8)}`);
  }
  return parts.length ? parts.join(" · ") : null;
}

export function graphLogSummary(item: IngestionLogEvent): string | null {
  if (!(item.event.startsWith("batch_graph_") || item.event.startsWith("graph_"))) {
    return null;
  }
  const parts: string[] = [];
  if (typeof item.graph_extraction_completed_chunks === "number") {
    parts.push(`chunks ${item.graph_extraction_completed_chunks}`);
  } else if (typeof item.graph_llm_success_chunks === "number") {
    parts.push(`chunks ${item.graph_llm_success_chunks}`);
  }
  if (typeof item.concepts === "number") {
    parts.push(`concepts ${item.concepts}`);
  }
  if (typeof item.relations === "number") {
    parts.push(`relations ${item.relations}`);
  }
  if (typeof item.graph_rejected_concepts === "number") {
    parts.push(`rejected ${item.graph_rejected_concepts}`);
  }
  if (typeof item.graph_algorithm_nodes === "number") {
    parts.push(`nodes ${item.graph_algorithm_nodes}`);
  }
  if (typeof item.graph_algorithm_edges === "number") {
    parts.push(`edges ${item.graph_algorithm_edges}`);
  }
  if (typeof item.community_summary_count === "number") {
    parts.push(`communities ${item.community_summary_count}`);
  }
  return parts.length ? parts.join(" · ") : null;
}
