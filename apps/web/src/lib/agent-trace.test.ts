import { describe, expect, it } from "vitest";

import { evidenceFirstTraceFallbackSteps, traceAuditSummary, traceNodeLabel } from "./agent-trace";

describe("agent trace helpers", () => {
  it("uses evidence-first fallback steps", () => {
    expect(evidenceFirstTraceFallbackSteps).toEqual([
      "perception",
      "retrieval_planner",
      "base_retrieval",
      "evidence_anchor_selector",
      "evidence_chain_planner",
      "controlled_graph_enhancer",
      "evidence_assembler",
      "document_grader",
      "evidence_evaluator",
      "context_synthesizer",
      "answer_generator",
      "citation_checker",
      "citation_verifier",
      "reflection",
      "self_check",
    ]);
  });

  it("labels new and legacy nodes", () => {
    expect(traceNodeLabel("base_retrieval")).toBe("基础召回");
    expect(traceNodeLabel("evidence_chain_planner")).toBe("证据链规划");
    expect(traceNodeLabel("controlled_graph_enhancer")).toBe("受控信号投影");
    expect(traceNodeLabel("evidence_evaluator")).toBe("证据充分性");
    expect(traceNodeLabel("retrievers")).toBe("基础召回");
  });

  it("summarizes evidence-first and chunk-route audit scores", () => {
    expect(
      traceAuditSummary({
        audit: {
          anchor_count: 2,
          planned_paths: 3,
          observed_edges: 4,
          expanded_active_chunks: 6,
          chunk_retained: 10,
        },
      }),
    ).toEqual(["anchors: 2", "paths: 3", "observed: 4", "expanded active: 6", "retained chunks: 10"]);
  });
});
