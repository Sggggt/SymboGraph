import { describe, expect, it } from "vitest";

import type { GraphResponse, SearchResult } from "@course-kg/shared";
import { pathEdgeSummary, toLayeredContextGraph } from "@/components/search-workspace";

describe("toLayeredContextGraph", () => {
  it("keeps chunk relation nodes, RQ metadata, and relation edges for the search canvas", () => {
    const graph: GraphResponse = {
      graph_type: "chunk-relation",
      schema_version: "context_graph_v1",
      counts: { active_chunks: 1, rq_relation_edges: 1 },
      sampled_counts: { nodes: 2, edges: 1 },
      node_counts: { chunk: 1, rq_prefix: 1 },
      edge_counts: { rq_membership: 1 },
      freshness: { is_stale: false },
      nodes: [
        { id: "chunk:c1", name: "Independence", category: "chunk", metadata: { rq_path: [1, 2, 3], residual_norm: 0.2 } },
        { id: "rq:p1", name: "RQ L3 1/2/3", category: "rq_prefix" },
      ],
      edges: [{ source: "chunk:c1", target: "rq:p1", label: "rq_leaf", category: "rq_membership" }],
    };

    const contextGraph = toLayeredContextGraph(graph);

    expect(contextGraph?.graph_type).toBe("chunk-relation");
    expect(contextGraph?.nodes.map((node) => node.id)).toEqual(["chunk:c1", "rq:p1"]);
    expect(contextGraph?.nodes[0].metadata).toMatchObject({ rq_path: [1, 2, 3], residual_norm: 0.2 });
    expect(contextGraph?.edges.map((edge) => edge.label)).toEqual(["rq_leaf"]);
    expect(contextGraph?.node_counts).toEqual({ chunk: 1, rq_prefix: 1 });
    expect(contextGraph?.edge_counts).toEqual({ rq_membership: 1 });
  });
});

describe("pathEdgeSummary", () => {
  const baseResult: SearchResult = {
    chunk_id: "c1",
    snippet: "Bayesian network evidence",
    score: 0.5,
    citations: [],
    metadata: {},
  };

  it("labels seed hits with evidence roles instead of displaying empty edges as none", () => {
    expect(
      pathEdgeSummary({
        ...baseResult,
        metadata: {
          traversal: {
            path_edge_ids: [],
            evidence_roles: ["rq_membership_entry", "dense_entry", "mid_drilldown_entry", "coarse_to_mid_drilldown_entry"],
          },
        },
      }),
    ).toBe("入口种子：rq_membership_entry / dense_entry / mid_drilldown_entry ...");
  });

  it("summarizes traversed relation edge ids when the result path used graph edges", () => {
    expect(
      pathEdgeSummary({
        ...baseResult,
        metadata: {
          traversal: {
            path_edge_ids: ["edge-1", "edge-2"],
            evidence_roles: ["semantic_similarity"],
          },
        },
      }),
    ).toBe("edge-1 / edge-2");
  });
});
