import { describe, expect, it } from "vitest";

import type { GraphResponse, ModelAudit, SearchResult } from "@course-kg/shared";
import { exploredMidNodeIdsFromTrace, pathEdgeSummary, semanticEntryAuditLabels, toExploredMidConceptGraph } from "@/components/search-workspace";
import {
  makeGraphResponse,
  makeMidConceptNode,
  makeRelationEdge,
  makeRetrievalStep,
  makeSupportRefs,
  makeTraceResponse,
  makeTraversalState,
} from "@/test/public-contract-fixtures";

describe("semanticEntryAuditLabels", () => {
  it("shows a locally selected semantic entry without granting gray-zone authority", () => {
    expect(semanticEntryAuditLabels({
      semantic_entry_query_selection_source: "validated_required_facet",
      semantic_entry_query_gray_zone_decision_authority: false,
    } as ModelAudit)).toEqual({
      source: "已去除交互指令",
      grayAuthority: "无",
    });
  });

  it("fails visibly when the gray-zone authority audit is absent", () => {
    expect(semanticEntryAuditLabels()).toEqual({
      source: "原始问题",
      grayAuthority: "未审计",
    });
  });
});

describe("toExploredMidConceptGraph", () => {
  it("shows only mid concept nodes explored by the current retrieval trace", () => {
    const graph = makeGraphResponse({
      graph_type: "mid-concepts",
      counts: { mid_concepts: 3, mid_concept_edges: 2 },
      sampled_counts: { nodes: 3, edges: 2 },
      nodes: [
        makeMidConceptNode("mid:m1", { name: "平面图性质", support_active_chunk_ids: ["c1"] }),
        makeMidConceptNode("mid:m2", { name: "欧拉公式" }),
        makeMidConceptNode("mid:m3", { name: "未探索概念" }),
      ],
      edges: [
        makeRelationEdge("mid:m1", "mid:m2", { label: "related" }),
        makeRelationEdge("mid:m2", "mid:m3", { label: "unseen" }),
      ],
    });
    const trace = makeTraceResponse({
      trace_id: "trace-1",
      stage_queues: { mid: { entry_ids: [], forced_entry_ids: [], forced_downstream_entry_ids: [], selected_ids: [], accepted_ids: ["mid:m1"] } },
      topk_selection: { mid: { candidate_count: 1, selected_ids: ["mid:m2"], forced_selected_ids: [] } },
      steps: [
        makeRetrievalStep({
          id: "step-mid",
          step_index: 1,
          layer: "mid",
          input: {
            entry_node_ids: ["mid:m1"], coarse_entry_ids: [], mid_entry_ids: ["mid:m1"],
            rq_membership_entry_ids: [], query_rq_path: [], result_chunk_ids: [], hit_chunk_ids: [],
          },
          output: {
            accepted_node_ids: ["mid:m1", "mid:m2"], selected_node_ids: ["mid:m1", "mid:m2"],
            accepted_chunk_ids: [], source_span_count: 0,
          },
          selected_topk_ids: ["mid:m2"],
          popped_frontier_state: makeTraversalState({ node_id: "mid:m1", path: ["mid:m1", "mid:m2"] }),
        }),
      ],
    });

    const contextGraph = toExploredMidConceptGraph(graph, trace);

    expect(contextGraph?.graph_type).toBe("mid-concepts");
    expect(contextGraph?.nodes.map((node: GraphResponse["nodes"][number]) => node.id)).toEqual(["mid:m1", "mid:m2"]);
    expect(contextGraph?.nodes[0].support_active_chunk_ids).toEqual(["c1"]);
    expect(contextGraph?.edges.map((edge: GraphResponse["edges"][number]) => edge.label)).toEqual(["related"]);
    expect(contextGraph?.node_counts).toEqual({ sampled: 2, full: 3 });
    expect(contextGraph?.edge_counts).toEqual({ sampled: 1, full: 2 });
  });

  it("returns an empty display graph before a retrieval trace is available", () => {
    const graph = makeGraphResponse({
      graph_type: "mid-concepts",
      counts: { mid_concepts: 1 },
      sampled_counts: { nodes: 1, edges: 0 },
      nodes: [makeMidConceptNode("mid:m1", { name: "平面图性质" })],
      edges: [],
    });

    const contextGraph = toExploredMidConceptGraph(graph, undefined);

    expect(contextGraph?.nodes).toEqual([]);
    expect(contextGraph?.edges).toEqual([]);
    expect(contextGraph?.sampled_counts).toMatchObject({ nodes: 0, edges: 0 });
    expect(contextGraph?.node_counts.sampled).toBe(0);
  });
});

describe("exploredMidNodeIdsFromTrace", () => {
  it("collects explored mid ids from queue, top-k, step and path-label diagnostics", () => {
    const ids = exploredMidNodeIdsFromTrace(makeTraceResponse({
      trace_id: "trace-1",
      entry_nodes: [{ layer: "mid", node_id: "mid:entry", roles: [], metadata: { rq_path_prefix: [], representative_terms: [] } }],
      stage_queues: { mid: { entry_ids: [], forced_entry_ids: [], forced_downstream_entry_ids: [], selected_ids: ["mid:selected"], accepted_ids: ["mid:accepted"] } },
      topk_selection: { mid: { candidate_count: 1, selected_ids: ["mid:topk"], forced_selected_ids: [] } },
      path_labels: [{
        layer: "mid", node_id: "mid:path-node", path: ["mid:path-a", "mid:path-b"],
        path_edge_ids: [], path_edge_types: [], expanded_edge_ids: [], covered_facets: [], evidence_roles: [],
        support_refs: makeSupportRefs(), entry_parent_refs: [], path_edge_type_multiset: {}, edge_reuse_counts: {},
      }],
      steps: [
        makeRetrievalStep({ id: "step-chunk", step_index: 0, layer: "chunk", selected_topk_ids: ["chunk:1"] }),
        makeRetrievalStep({
          id: "step-mid",
          step_index: 1,
          layer: "mid",
          input: {
            entry_node_ids: [], coarse_entry_ids: [], mid_entry_ids: ["mid:step-entry"],
            rq_membership_entry_ids: [], query_rq_path: [], result_chunk_ids: [], hit_chunk_ids: [],
          },
          output: {
            accepted_node_ids: ["mid:step-accepted"], selected_node_ids: [], accepted_chunk_ids: [], source_span_count: 0,
          },
          selected_topk_ids: ["mid:step-topk"],
          popped_frontier_state: makeTraversalState({ node_id: "mid:popped", path: ["mid:popped", "mid:path-c"] }),
        }),
      ],
    }));

    expect([...ids].sort()).toEqual([
      "mid:accepted",
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
