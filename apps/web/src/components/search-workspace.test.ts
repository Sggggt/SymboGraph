import { describe, expect, it } from "vitest";

import type { GraphResponse } from "@course-kg/shared";
import { toLayeredContextGraph } from "@/components/search-workspace";

describe("toLayeredContextGraph", () => {
  it("keeps chunk relation nodes, RQ metadata, and relation edges for the search canvas", () => {
    const graph: GraphResponse = {
      graph_type: "chunk-relation",
      schema_version: "context_graph_v1",
      counts: { active_chunks: 1, rq_edges: 1 },
      sampled_counts: { nodes: 2, edges: 1 },
      node_counts: { chunk: 1, fine_cluster: 1 },
      edge_counts: { rq_residual_near: 1 },
      freshness: { is_stale: false },
      nodes: [
        { id: "chunk:c1", name: "Independence", category: "chunk", metadata: { rq_path: [1, 2, 3], residual_norm: 0.2 } },
        { id: "fine:f1", name: "Fine cluster", category: "fine_cluster" },
      ],
      edges: [{ source: "chunk:c1", target: "fine:f1", label: "rq_residual_near", category: "rq_residual_near" }],
    };

    const contextGraph = toLayeredContextGraph(graph);

    expect(contextGraph?.graph_type).toBe("chunk-relation");
    expect(contextGraph?.nodes.map((node) => node.id)).toEqual(["chunk:c1", "fine:f1"]);
    expect(contextGraph?.nodes[0].metadata).toMatchObject({ rq_path: [1, 2, 3], residual_norm: 0.2 });
    expect(contextGraph?.edges.map((edge) => edge.label)).toEqual(["rq_residual_near"]);
    expect(contextGraph?.node_counts).toEqual({ chunk: 1, fine_cluster: 1 });
    expect(contextGraph?.edge_counts).toEqual({ rq_residual_near: 1 });
  });
});
