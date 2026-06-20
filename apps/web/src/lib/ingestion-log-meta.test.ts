import { describe, expect, it } from "vitest";

import { graphLogSummary, logEventLabel, logVisualTone } from "./ingestion-log-meta";

describe("ingestion log metadata", () => {
  it("keeps unknown events readable and classifies graph/failure tones", () => {
    expect(logEventLabel("unknown_event")).toBe("未知事件：unknown event");
    expect(logVisualTone({ event: "chunk_relation_completed" })).toBe("graph");
    expect(logVisualTone({ event: "graph_failed" })).toBe("failure");
    expect(logEventLabel("log_stream_warning")).toBe("日志流告警");
    expect(logVisualTone({ event: "log_stream_warning" })).toBe("warning");
    expect(logEventLabel("log_stream_recovered")).toBe("日志流恢复");
    expect(logVisualTone({ event: "log_stream_recovered" })).toBe("default");
  });

  it("summarizes four-layer graph phase logs", () => {
    expect(logEventLabel("context_graph_started")).toBe("四层上下文图谱开始");
    expect(logEventLabel("chunk_relation_completed")).toBe("片段关系图完成");
    expect(logVisualTone({ event: "context_graph_completed" })).toBe("graph");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "context_graph_completed",
        message: "done",
        phase: "context_graph:completed",
        chunk_count: 12,
        relation_edge_count: 8,
        rq_prefix_count: 4,
        context_graph_hash: "abcdef123456",
      }),
    ).toBe("阶段 图谱闭环完成 / 片段 12 / 关系边 8 / RQ membership 4 / 哈希 abcdef12");
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
    ).toBe("中概念 7");
  });

  it("summarizes concept i18n progress and completion counters", () => {
    expect(logVisualTone({ event: "batch_graph_progress", translation_phase: "concept_i18n" })).toBe("graph");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "batch_graph_progress",
        message: "i18n",
        phase: "context_graph:mid_concepts",
        context_graph_phase: "mid_concepts",
        translation_phase: "concept_i18n",
        translation_items: 100,
      }),
    ).toBe("阶段 中粒度概念 / 双语 节点双语派生 / 派生项 100");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "batch_graph_progress",
        message: "i18n disabled",
        phase: "context_graph:coarse_concepts",
        translation_phase: "edge_i18n",
        translation_enabled: false,
        translation_status: "disabled",
        translation_items: 12,
      }),
    ).toBe("阶段 粗粒度概念 / 双语 关系双语派生 / 状态 未启用 / 派生项 12");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "context_graph_completed",
        message: "done",
        phase: "context_graph:completed",
        concept_i18n_translated_count: 129,
        edge_i18n_translated_count: 689,
      }),
    ).toBe("阶段 图谱闭环完成 / 节点翻译 129 / 关系翻译 689");
  });

  it("summarizes TPE trial and hard gate events", () => {
    expect(logEventLabel("auto_tpe_trial_completed")).toBe("自动 TPE Trial 完成");
    expect(logVisualTone({ event: "auto_tpe_trial_completed" })).toBe("graph");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "auto_tpe_trial_completed",
        message: "trial done",
        trial_index: 2,
        theta_hash: "abcdef123456",
        objective_score: 0.81234,
        probe_set_hash: "123456abcdef",
        hard_gate: {
          edge_density: { passed: true },
          isolated_ratio: { passed: false },
        },
      }),
    ).toBe("Trial 2 / Objective 0.8123 / 参数 abcdef12 / Probe 123456ab / Hard gate 未通过 isolated_ratio");
  });
});
