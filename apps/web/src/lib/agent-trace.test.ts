import { describe, expect, it } from "vitest";

import { contextGraphTraceFallbackSteps, traceAuditSummary, traceNodeLabel } from "./agent-trace";

describe("agent trace helpers", () => {
  it("uses four-layer P&E fallback steps", () => {
    expect(contextGraphTraceFallbackSteps).toEqual([
      "query_understanding",
      "agent_planner",
      "typed_action_validation",
      "coarse_concept_activation",
      "mid_concept_routing",
      "fine_cluster_routing",
      "chunk_recall",
      "structure_context_restoration",
      "context_package",
      "grounded_answer",
      "citation_verification",
      "reward_event",
    ]);
  });

  it("labels context graph nodes", () => {
    expect(traceNodeLabel("agent_planner")).toBe("智能体规划");
    expect(traceNodeLabel("typed_action_validation")).toBe("动作校验");
    expect(traceNodeLabel("coarse_concept_activation")).toBe("粗粒度概念激活");
    expect(traceNodeLabel("mid_concept_routing")).toBe("中粒度概念路由");
    expect(traceNodeLabel("fine_cluster_routing")).toBe("细聚类/RQ 路由");
    expect(traceNodeLabel("structure_context_restoration")).toBe("结构上下文恢复");
    expect(traceNodeLabel("retrievers")).toBe("片段召回");
  });

  it("summarizes context graph audit scores including RQ path", () => {
    expect(
      traceAuditSummary({
        audit: {
          coarse_concepts: 2,
          mid_concepts: 3,
          fine_clusters: 4,
          query_rq_path: [1, 2, 3],
          recalled_chunks: 10,
          structure_neighbors: 6,
          context_package_id: "pkg-1",
        },
      }),
    ).toEqual(["粗概念: 2", "中概念: 3", "细聚类: 4", "RQ 路径: 1/2/3", "召回片段: 10", "结构邻居: 6", "证据包: pkg-1"]);
  });
});
