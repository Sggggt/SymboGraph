import { describe, expect, it } from "vitest";

import { buildBaseOption } from "@/components/network-canvas";
import type { GraphResponse } from "@course-kg/shared";

describe("NetworkCanvas option mapping", () => {
  it("styles typed semantic and evidence graph nodes without structural assumptions", () => {
    const graph: GraphResponse = {
      graph_type: "evidence",
      schema_version: "typed_graph_v1",
      node_counts: {},
      edge_counts: {},
      freshness: { is_stale: false },
      nodes: [
        { id: "semantic:1", name: "PageRank", category: "semantic_entity", entity_type: "algorithm", value: 4 },
        { id: "evidence_chunk:c1", name: "Evidence", category: "evidence_chunk", snippet: "PageRank is defined..." },
        { id: "document_version:v1", name: "Lecture 1", category: "document_version" },
      ],
      edges: [
        { source: "semantic:1", target: "evidence_chunk:c1", label: "evidenced_by", category: "evidence" },
        { source: "evidence_chunk:c1", target: "document_version:v1", label: "from_version", category: "evidence" },
      ],
    };

    const option = buildBaseOption(graph);
    const series = Array.isArray(option.series) ? option.series[0] : undefined;

    expect(series).toMatchObject({ type: "graph" });
    expect((series as { data: Array<{ category: string; symbolSize: number }> }).data.map((node) => node.category)).toEqual([
      "semantic_entity",
      "evidence_chunk",
      "document_version",
    ]);
    expect((series as { data: Array<{ symbolSize: number }> }).data[0].symbolSize).toBeGreaterThan((series as { data: Array<{ symbolSize: number }> }).data[1].symbolSize);
  });
});
