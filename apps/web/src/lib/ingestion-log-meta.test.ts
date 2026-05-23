import { describe, expect, it } from "vitest";

import { graphLogSummary, hpoLogSummary, logEventLabel, logVisualTone } from "./ingestion-log-meta";

describe("ingestion log metadata", () => {
  it("labels and classifies judge-learned HPO events", () => {
    expect(logEventLabel("hpo_judge_progress")).toBe("Judge HPO 进度");
    expect(logVisualTone({ event: "hpo_judge_progress", stage: "judge" })).toBe("hpo");
    expect(
      hpoLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "hpo_judge_progress",
        message: "progress",
        candidate_count: 4,
        processed_pairs: 2,
        pair_count: 3,
        effective_labels: 2,
        min_labels: 2,
      }),
    ).toContain("Judge 2/3");
  });

  it("keeps unknown events readable and classifies graph/failure tones", () => {
    expect(logEventLabel("unknown_event")).toBe("unknown event");
    expect(logVisualTone({ event: "batch_graph_progress" })).toBe("graph");
    expect(logVisualTone({ event: "hpo_failed" })).toBe("failure");
  });

  it("summarizes graph phase logs", () => {
    expect(logEventLabel("graph_upsert_started")).toBe("图谱写入开始");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "graph_upsert_completed",
        message: "done",
        graph_llm_success_chunks: 12,
        concepts: 4,
        relations: 6,
        graph_rejected_concepts: 2,
      }),
    ).toContain("relations 6");
  });
});
