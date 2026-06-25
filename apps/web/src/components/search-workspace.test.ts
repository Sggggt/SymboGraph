import { describe, expect, it } from "vitest";

import type { GraphResponse, RetrievalTraceStepsResponse, SearchResult } from "@course-kg/shared";
import { exploredMidNodeIdsFromTrace, pathEdgeSummary, toExploredMidConceptGraph } from "@/components/search-workspace";

describe("toExploredMidConceptGraph", () => {
  it("shows only mid concept nodes explored by the current retrieval trace", () => {
    const graph: GraphResponse = {
      graph_type: "mid-concepts",
      schema_version: "context_graph_v1",
      counts: { mid_concepts: 3, mid_concept_edges: 2 },
      sampled_counts: { nodes: 3, edges: 2 },
      node_counts: { mid_concept: 3 },
      edge_counts: { concept_relation: 2 },
      freshness: { is_stale: false },
      nodes: [
        { id: "mid:m1", name: "平面图性质", category: "mid_concept", metadata: { support_active_chunk_ids: ["c1"] } },
        { id: "mid:m2", name: "欧拉公式", category: "mid_concept" },
        { id: "mid:m3", name: "未探索概念", category: "mid_concept" },
      ],
      edges: [
        { source: "mid:m1", target: "mid:m2", label: "related", category: "concept_relation" },
        { source: "mid:m2", target: "mid:m3", label: "unseen", category: "concept_relation" },
      ],
    };
    const trace: RetrievalTraceStepsResponse = {
      trace_id: "trace-1",
      stage_queues: { mid: { accepted_ids: ["mid:m1"] } },
      topk_selection: { mid: { selected_ids: ["mid:m2"] } },
      steps: [
        {
          layer: "mid",
          input: { entry_nodes: [{ node_id: "mid:m1" }] },
          output: { accepted_nodes: ["mid:m1", "mid:m2"] },
          selected_topk_ids: ["mid:m2"],
          popped_frontier_state: { node_id: "mid:m1", path: ["mid:m1", "mid:m2"] },
        },
      ],
    };

    const contextGraph = toExploredMidConceptGraph(graph, trace);

    expect(contextGraph?.graph_type).toBe("mid-concepts");
    expect(contextGraph?.nodes.map((node: GraphResponse["nodes"][number]) => node.id)).toEqual(["mid:m1", "mid:m2"]);
    expect(contextGraph?.nodes[0].metadata).toMatchObject({ support_active_chunk_ids: ["c1"] });
    expect(contextGraph?.edges.map((edge: GraphResponse["edges"][number]) => edge.label)).toEqual(["related"]);
    expect(contextGraph?.node_counts).toEqual({ mid_concept: 2 });
    expect(contextGraph?.edge_counts).toEqual({ concept_relation: 1 });
  });

  it("returns an empty display graph before a retrieval trace is available", () => {
    const graph: GraphResponse = {
      graph_type: "mid-concepts",
      counts: { mid_concepts: 1 },
      sampled_counts: { nodes: 1, edges: 0 },
      node_counts: { mid_concept: 1 },
      edge_counts: { concept_relation: 0 },
      freshness: { is_stale: false },
      nodes: [{ id: "mid:m1", name: "平面图性质", category: "mid_concept" }],
      edges: [],
    };

    const contextGraph = toExploredMidConceptGraph(graph, undefined);

    expect(contextGraph?.nodes).toEqual([]);
    expect(contextGraph?.edges).toEqual([]);
    expect(contextGraph?.sampled_counts).toMatchObject({ nodes: 0, edges: 0 });
    expect(contextGraph?.diagnostics?.filtered_to_explored_mid_nodes).toBe(true);
  });
});

describe("exploredMidNodeIdsFromTrace", () => {
  it("collects explored mid ids from queue, top-k, step and path-label diagnostics", () => {
    const ids = exploredMidNodeIdsFromTrace({
      trace_id: "trace-1",
      entry_nodes: [{ node_id: "mid:entry" }],
      stage_queues: { mid: { selected_ids: ["mid:selected"], accepted_ids: ["mid:accepted"] } },
      topk_selection: { mid: { selected_ids: ["mid:topk"] } },
      path_labels: [{ node_id: "mid:path-node", path: ["mid:path-a", "mid:path-b"] }],
      steps: [
        { layer: "chunk", selected_topk_ids: ["chunk:1"] },
        {
          layer: "mid",
          input: { entry_nodes: [{ node_id: "mid:step-entry" }] },
          output: { accepted_nodes: ["mid:step-accepted"] },
          selected_topk_ids: ["mid:step-topk"],
          popped_frontier_state: { node_id: "mid:popped", path: ["mid:popped", "mid:path-c"] },
          diagnostics: { path_labels: [{ node_id: "mid:diag-node", path: ["mid:diag-path"] }] },
        },
      ],
    } as RetrievalTraceStepsResponse);

    expect([...ids].sort()).toEqual([
      "mid:accepted",
      "mid:diag-node",
      "mid:diag-path",
      "mid:entry",
      "mid:path-a",
      "mid:path-b",
      "mid:path-c",
      "mid:path-node",
      "mid:popped",
      "mid:selected",
      "mid:step-accepted",
      "mid:step-entry",
      "mid:step-topk",
      "mid:topk",
    ]);
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
