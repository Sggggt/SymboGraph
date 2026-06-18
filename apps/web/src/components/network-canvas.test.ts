import { describe, expect, it } from "vitest";

import { buildBaseOption } from "@/components/network-canvas";
import type { GraphResponse } from "@course-kg/shared";

describe("NetworkCanvas option mapping", () => {
  it("styles four-layer context graph nodes and RQ edges", () => {
    const graph: GraphResponse = {
      graph_type: "chunk-relation",
      schema_version: "context_graph_v1",
      counts: { active_chunks: 1, rq_relation_edges: 1 },
      sampled_counts: { nodes: 5, edges: 3 },
      freshness: { is_stale: false },
      nodes: [
        { id: "chunk:c1", name: "Chunk 1", category: "chunk", metadata: { rq_path: [1, 2, 1], residual_norm: 0.12 } },
        { id: "chunk:c2", name: "Chunk 2", category: "chunk", metadata: { rq_path: [1, 2, 2], residual_norm: 0.18 } },
        { id: "rq:p1", name: "RQ L3 1/2/1", category: "rq_prefix", value: 4 },
        { id: "mid:m1", name: "Mid concept", category: "mid_concept", value: 4 },
        { id: "coarse:k1", name: "Coarse concept", category: "coarse_concept", value: 4 },
      ],
      edges: [
        { source: "chunk:c1", target: "chunk:c2", label: "rq_residual_near", category: "rq_residual_near", weight: 0.8 },
        { source: "chunk:c1", target: "rq:p1", label: "rq_leaf", category: "rq_membership", weight: 0.7 },
        { source: "mid:m1", target: "coarse:k1", label: "included_in", category: "concept_relation", weight: 0.7 },
      ],
    };

    const option = buildBaseOption(graph);
    const series = Array.isArray(option.series) ? option.series[0] : undefined;

    expect(series).toMatchObject({ type: "graph" });
    const data = (series as { data: Array<{ category: string; symbolSize: number }> }).data;
    expect(data.map((node) => node.category)).toEqual(["chunk", "chunk", "rq_prefix", "mid_concept", "coarse_concept"]);
    expect(data[3].symbolSize).toBeGreaterThan(data[0].symbolSize);
    const links = (series as { links: Array<{ category: string; lineStyle: { type: string; opacity: number } }> }).links;
    expect(links[0].category).toBe("rq_residual_near");
    expect(links[0].lineStyle.type).toBe("dashed");
    expect(links[0].lineStyle.opacity).toBeGreaterThan(links[2].lineStyle.opacity);
  });
});
