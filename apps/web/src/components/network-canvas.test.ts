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
        { id: "signal:s1", name: "PageRank", category: "signal_node", entity_type: "algorithm", support_atom_ids: ["a1"], value: 4 },
        { id: "atom:a1", name: "Evidence atom", category: "evidence_atom", snippet: "PageRank is defined...", value: 2 },
        { id: "active_chunk:c1", name: "Evidence chunk", category: "active_chunk", snippet: "PageRank is defined..." },
        { id: "document_version:v1", name: "Document v1", category: "document_version" },
      ],
      edges: [
        { source: "signal:s1", target: "atom:a1", label: "supported_by", category: "signal_projection", support_atom_ids: ["a1"] },
        { source: "atom:a1", target: "active_chunk:c1", label: "included_in", category: "evidence" },
        { source: "active_chunk:c1", target: "document_version:v1", label: "from_version", category: "evidence" },
      ],
    };

    const option = buildBaseOption(graph);
    const series = Array.isArray(option.series) ? option.series[0] : undefined;

    expect(series).toMatchObject({ type: "graph" });
    expect((series as { data: Array<{ category: string; symbolSize: number }> }).data.map((node) => node.category)).toEqual([
      "signal_node",
      "evidence_atom",
      "active_chunk",
      "document_version",
    ]);
    const data = (series as { data: Array<{ symbolSize: number }> }).data;
    expect(data[0].symbolSize).toBeGreaterThan(data[2].symbolSize);
    expect(data[1].symbolSize).toBeGreaterThan(data[2].symbolSize);
  });
});
