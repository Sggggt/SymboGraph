import { describe, expect, it } from "vitest";

import { graphLogSummary, logEventLabel, logVisualTone } from "./ingestion-log-meta";

describe("ingestion log metadata", () => {
  it("keeps unknown events readable and classifies graph/failure tones", () => {
    expect(logEventLabel("unknown_event")).toBe("未知事件：unknown event");
    expect(logVisualTone({ event: "chunk_relation_completed" })).toBe("graph");
    expect(logVisualTone({ event: "graph_failed" })).toBe("failure");
    expect(logEventLabel("log_stream_warning")).toBe("日志流告警");
    expect(logVisualTone({ event: "log_stream_warning" })).toBe("warning");
  });

  it("summarizes four-layer graph phase logs", () => {
    expect(logEventLabel("context_graph_started")).toBe("上下文图谱构建");
    expect(logEventLabel("chunk_relation_completed")).toBe("片段关系图完成");
    expect(logVisualTone({ event: "context_graph_completed" })).toBe("graph");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "context_graph_completed",
        message: "done",
        chunk_count: 12,
        relation_edge_count: 8,
        fine_cluster_count: 4,
        context_graph_hash: "abcdef123456",
      }),
    ).toBe("片段 12 / 关系边 8 / 细聚类 4 / 哈希 abcdef12");
  });

  it("summarizes concept graph events", () => {
    expect(logEventLabel("mid_concepts_completed")).toBe("中粒度概念完成");
    expect(logEventLabel("coarse_concepts_completed")).toBe("粗粒度概念完成");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "mid_concepts_completed",
        message: "done",
        mid_concept_count: 7,
      }),
    ).toBe("中粒度概念 7");
  });
});
