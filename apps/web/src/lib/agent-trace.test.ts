import { describe, expect, it } from "vitest";

import { contextGraphTraceFallbackSteps, groupTraceEvents, traceAuditSummary, traceNodeLabel } from "./agent-trace";

describe("agent trace helpers", () => {
  it("uses four-layer P&E fallback steps", () => {
    expect(contextGraphTraceFallbackSteps).toEqual([
      "query_understanding",
      "agent_planner",
      "typed_action_validation",
      "entry_selection",
      "layer_drilldown",
      "frontier_traversal",
      "chunk_recall",
      "structure_context_restoration",
      "context_package",
      "grounded_answer",
      "citation_verification",
      "reward_event",
    ]);
  });

  it("labels context graph nodes in Chinese", () => {
    expect(traceNodeLabel("agent_planner")).toBe("智能体规划");
    expect(traceNodeLabel("typed_action_validation")).toBe("动作校验");
    expect(traceNodeLabel("entry_selection")).toBe("入口选择");
    expect(traceNodeLabel("layer_drilldown")).toBe("分层下钻");
    expect(traceNodeLabel("frontier_traversal")).toBe("Frontier 遍历");
    expect(traceNodeLabel("structure_context_restoration")).toBe("结构上下文恢复");
    expect(traceNodeLabel("retrievers")).toBe("片段召回");
  });

  it("groups trace events by QA workflow stage", () => {
    expect(
      groupTraceEvents([
        { node: "entry_selection", status: "completed", document_ids: [], scores: {}, duration_ms: 1 },
        { node: "frontier_traversal", status: "completed", document_ids: [], scores: {}, duration_ms: 1 },
        { node: "citation_verification", status: "completed", document_ids: [], scores: {}, duration_ms: 1 },
      ]).map((group) => group.label),
    ).toEqual(["入口选择", "Frontier 遍历", "引用验证"]);
  });

  it("summarizes context graph audit scores including RQ path", () => {
    expect(
      traceAuditSummary({
        audit: {
          coarse_entries: 2,
          mid_entries: 3,
          rq_membership_entries: 4,
          frontier_pops: 5,
          query_rq_path: [1, 2, 3],
          recalled_chunks: 10,
          structure_neighbors: 6,
          context_package_id: "pkg-1",
        },
      }),
    ).toEqual(["粗入口: 2", "中入口: 3", "RQ 归属: 4", "Frontier pop: 5", "RQ 路径: 1/2/3", "召回片段: 10", "结构邻居: 6", "证据包: pkg-1"]);
  });
});
