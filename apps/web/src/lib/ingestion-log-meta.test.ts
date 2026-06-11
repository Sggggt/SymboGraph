import { describe, expect, it } from "vitest";

import { graphLogSummary, logEventLabel, logVisualTone } from "./ingestion-log-meta";

describe("ingestion log metadata", () => {
  it("keeps unknown events readable and classifies graph/failure tones", () => {
    expect(logEventLabel("unknown_event")).toBe("unknown event");
    expect(logVisualTone({ event: "batch_graph_progress" })).toBe("graph");
    expect(logVisualTone({ event: "graph_failed" })).toBe("failure");
    expect(logEventLabel("log_stream_warning")).toBe("Log stream warning");
    expect(logVisualTone({ event: "log_stream_warning" })).toBe("warning");
  });

  it("summarizes graph phase logs", () => {
    expect(logEventLabel("batch_graph_progress")).toBe("Evidence graph progress");
    expect(logEventLabel("global_graph_active")).toBe("Global evidence graph active");
    expect(logVisualTone({ event: "global_graph_active" })).toBe("graph");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "graph_rebuilt",
        message: "done",
        evidence_atoms: 12,
        evidence_edges: 8,
        active_chunks: 4,
      }),
    ).toContain("active chunks 4");
  });

  it("summarizes global graph and signal candidate logs from the evidence pipeline", () => {
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "global_graph_scanning",
        message: "started",
        atom_count: 9,
      }),
    ).toBe("atoms 9");
    expect(logEventLabel("signal_candidate_gate")).toBe("Signal candidate gate");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "signal_candidate_gate",
        message: "gate",
        candidate_count: 7,
      }),
    ).toBe("signal candidates 7");
    expect(logVisualTone({ event: "signal_schema_failed" })).toBe("failure");
  });

  it("summarizes signal layer logs outside the evidence graph tone", () => {
    expect(logEventLabel("signal_layer_active")).toBe("Signal layer active");
    expect(logVisualTone({ event: "signal_layer_active" })).toBe("default");
    expect(
      graphLogSummary({
        timestamp: "2026-01-01T00:00:00",
        event: "signal_layer_active",
        message: "published",
        signal_node_count: 5,
        signal_edge_count: 4,
      }),
    ).toBe("signal nodes 5 / signal edges 4");
  });

  it("does not expose adaptive chunking wording", () => {
    expect(logEventLabel("evidence_section_sizing")).toBe("Evidence section sizing");
    expect(logEventLabel("chunk_adaptive")).toBe("Evidence section sizing");
    expect(logVisualTone({ event: "chunk_adaptive" })).toBe("default");
  });
});
