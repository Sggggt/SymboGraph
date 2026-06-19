// @vitest-environment jsdom

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { GraphResponse } from "@course-kg/shared";
import { GraphNodeSummary, nodeDetailRows } from "./graph-panel";

describe("GraphNodeSummary", () => {
  it("renders natural-language concept details instead of raw metadata JSON", () => {
    const selectedNode: GraphResponse["nodes"][number] = {
      id: "mid:bayes",
      label: "贝叶斯更新",
      name: "贝叶斯更新",
      type: "mid_concept",
      category: "mid_concept",
      summary: "贝叶斯更新描述先验、似然和后验之间的证据更新关系。",
      confidence: 0.87,
      support_active_chunk_ids: ["chunk:prior", "chunk:posterior"],
      metadata: { internal_trace: { hidden: true } },
    };
    const graph: GraphResponse = {
      graph_type: "mid-concepts",
      schema_version: "context_graph_v1",
      counts: { mid_concepts: 2 },
      sampled_counts: { nodes: 2, edges: 1 },
      freshness: { is_stale: false },
      nodes: [
        selectedNode,
        { id: "mid:evidence", label: "证据权重", name: "证据权重", type: "mid_concept", category: "mid_concept" },
      ],
      edges: [{ id: "edge:1", source: "mid:bayes", target: "mid:evidence", label: "concept_relation", type: "concept_relation", weight: 0.73 }],
    };

    const { container } = render(<GraphNodeSummary node={selectedNode} graph={graph} graphType="mid-concepts" />);

    expect(screen.getByText("贝叶斯更新")).toBeTruthy();
    expect(container.textContent).toContain("这是中粒度概念节点");
    expect(container.textContent).toContain("自然语言定义");
    expect(container.textContent).toContain("贝叶斯更新描述先验、似然和后验之间的证据更新关系。");
    expect(container.textContent).toContain("2 个支撑片段");
    expect(container.textContent).toContain("concept_relation");
    expect(container.textContent).not.toContain("internal_trace");
    expect(container.textContent).not.toContain("{");
  });

  it("keeps RQ-prefix layer data readable", () => {
    const node: GraphResponse["nodes"][number] = {
      id: "rq:1-4",
      label: "RQ L2 1/4",
      type: "rq_prefix",
      category: "rq_prefix",
      metadata: {
        rq_prefix_key: "L2:1/4",
        rq_level: 2,
        rq_path_prefix: [1, 4],
        support_chunk_ids: ["chunk:alpha", "chunk:beta", "chunk:gamma", "chunk:delta"],
        representative_chunk_ids: ["chunk:alpha"],
        bridge_chunk_ids: ["chunk:bridge"],
        stats: { residual_norm_mean: 0.1234, residual_norm_max: 0.4567 },
      },
    };

    const rows = nodeDetailRows(node, "chunk-relation", []);

    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "RQ 前缀键", value: "L2:1/4" }),
        expect.objectContaining({ label: "RQ 层级", value: "2" }),
        expect.objectContaining({ label: "前缀路径", value: "1/4" }),
        expect.objectContaining({ label: "支撑片段", value: expect.stringContaining("4 个支撑片段") }),
        expect.objectContaining({ label: "残差均值", value: "0.123" }),
      ]),
    );
  });
});
