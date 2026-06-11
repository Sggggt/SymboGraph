import type { IngestionLogEvent } from "@course-kg/shared";

export type LogVisualTone = "graph" | "warning" | "failure" | "default";

export const logEventLabels: Record<string, string> = {
  batch_started: "Batch started",
  batch_files: "Files scanned",
  file_started: "File parsing started",
  job_state: "Job state",
  file_skipped: "File skipped",
  file_completed: "File completed",
  file_failed: "File failed",
  batch_graph_started: "Evidence graph build",
  batch_graph_selected: "Evidence scope selected",
  batch_graph_plan_created: "Evidence graph plan",
  batch_graph_probe_started: "Evidence graph probe",
  batch_graph_progress: "Evidence graph progress",
  batch_graph_adaptive_round_started: "Evidence graph round",
  batch_graph_coverage_updated: "Evidence coverage updated",
  batch_graph_community_summary: "Community summary progress",
  graph_upsert_started: "Evidence graph write started",
  graph_upsert_completed: "Evidence graph write completed",
  graph_enrichment_started: "Evidence topology refresh started",
  graph_enrichment_completed: "Evidence topology refresh completed",
  graph_community_started: "Community summary started",
  graph_community_completed: "Community summary completed",
  graph_rebuilt: "Evidence graph completed",
  graph_failed: "Evidence graph failed",
  global_graph_scanning: "Global evidence graph publish",
  global_graph_active: "Global evidence graph active",
  global_graph_failed: "Global evidence graph failed",
  signal_candidate_scanning: "Signal candidate scan",
  signal_candidate_gate: "Signal candidate gate",
  signal_schema_failed: "Signal schema failed",
  signal_layer_scanning: "Signal layer scanning",
  signal_layer_normalizing: "Signal layer normalizing",
  signal_layer_assembling: "Signal layer assembling",
  signal_layer_validating: "Signal layer validating",
  signal_layer_active: "Signal layer active",
  signal_layer_failed: "Signal layer failed",
  batch_completed: "Batch completed",
  batch_partial_failed: "Partially failed",
  batch_failed: "Batch failed",
  batch_skipped: "Batch skipped",
  batch_missing: "Batch missing",
  log_stream_retry: "Log stream reconnect",
  log_stream_warning: "Log stream warning",
  evidence_section_sizing: "Evidence section sizing",
  chunk_adaptive: "Evidence section sizing",
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
  if (item.event.startsWith("batch_graph_") || item.event.startsWith("graph_") || item.event.startsWith("global_graph_")) {
    return "graph";
  }
  return "default";
}

export function graphLogSummary(item: IngestionLogEvent): string | null {
  if (
    !(
      item.event.startsWith("batch_graph_") ||
      item.event.startsWith("graph_") ||
      item.event.startsWith("global_graph_") ||
      item.event.startsWith("signal_candidate_") ||
      item.event.startsWith("signal_schema_") ||
      item.event.startsWith("signal_layer_")
    )
  ) {
    return null;
  }
  const parts: string[] = [];
  const pushNumber = (label: string, value: number | undefined) => {
    if (typeof value === "number") {
      parts.push(`${label} ${value}`);
    }
  };
  pushNumber("atoms", item.atom_count);
  if (typeof item.evidence_atoms === "number") {
    parts.push(`atoms ${item.evidence_atoms}`);
  }
  pushNumber("edges", item.evidence_edges);
  pushNumber("active chunks", item.active_chunks);
  pushNumber("chunk candidates", item.chunk_candidates);
  pushNumber("communities", item.community_summary_count);
  pushNumber("signal candidates", item.candidate_count);
  pushNumber("signal candidates", item.signal_candidates);
  pushNumber("signal candidates", item.signal_candidate_count);
  pushNumber("accepted signals", item.accepted_signal_candidate_count);
  pushNumber("rejected signals", item.rejected_signal_candidate_count);
  pushNumber("signal nodes", item.signal_nodes ?? item.signal_node_count);
  pushNumber("signal edges", item.signal_edges ?? item.signal_edge_count);
  pushNumber("signal communities", item.signal_communities ?? item.signal_community_count);
  return parts.length ? parts.join(" / ") : null;
}
