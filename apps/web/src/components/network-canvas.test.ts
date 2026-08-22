import { describe, expect, it } from "vitest";

import { buildBaseOption, isUsableChartInstance } from "@/components/network-canvas";
import {
  makeChunkNode,
  makeCoarseConceptNode,
  makeGraphResponse,
  makeMidConceptNode,
  makeRelationEdge,
  makeRqPrefixNode,
} from "@/test/public-contract-fixtures";

describe("NetworkCanvas option mapping", () => {
  it("rejects disposed chart instances before deferred callbacks run", () => {
    expect(isUsableChartInstance(null)).toBe(false);
    expect(isUsableChartInstance({ isDisposed: () => true } as never)).toBe(false);
    expect(isUsableChartInstance({ isDisposed: () => false } as never)).toBe(true);
  });

  it("styles four-layer context graph nodes and dense relation edges", () => {
    const graph = makeGraphResponse({
      graph_type: "chunk-relation",
      counts: { active_chunks: 1, rq_relation_edges: 1 },
      sampled_counts: { nodes: 5, edges: 3 },
      nodes: [
        makeChunkNode("chunk:c1", { name: "Chunk 1", metadata: { rq_path: [1, 2, 1], residual_norm: 0.12, rq_path_prefix: [], representative_chunk_ids: [], support_chunk_ids: [], bridge_chunk_ids: [] } }),
        makeChunkNode("chunk:c2", { name: "Chunk 2", metadata: { rq_path: [1, 2, 2], residual_norm: 0.18, rq_path_prefix: [], representative_chunk_ids: [], support_chunk_ids: [], bridge_chunk_ids: [] } }),
        makeRqPrefixNode("rq:p1", { name: "RQ L3 1/2/1", value: 4 }),
        makeMidConceptNode("mid:m1", { name: "Mid concept", value: 4 }),
        makeCoarseConceptNode("coarse:k1", { name: "Coarse concept", value: 4 }),
      ],
      edges: [
        makeRelationEdge("chunk:c1", "chunk:c2", { label: "dense_semantic", category: "dense_semantic", type: "dense_semantic", weight: 0.8 }),
        makeRelationEdge("chunk:c1", "rq:p1", { label: "rq_leaf", category: "rq_membership", type: "rq_membership", weight: 0.7 }),
        makeRelationEdge("mid:m1", "coarse:k1", { label: "included_in", category: "concept_relation", weight: 0.7 }),
      ],
    });

    const option = buildBaseOption(graph);
    const series = Array.isArray(option.series) ? option.series[0] : undefined;

    expect(series).toMatchObject({ type: "graph" });
    const data = (series as { data: Array<{ category: string; symbolSize: number }> }).data;
    expect(data.map((node) => node.category)).toEqual(["chunk", "chunk", "rq_prefix", "mid_concept", "coarse_concept"]);
    expect(data[3].symbolSize).toBeGreaterThan(data[0].symbolSize);
    const links = (series as { links: Array<{ category: string; lineStyle: { type: string; opacity: number } }> }).links;
    expect(links[0].category).toBe("dense_semantic");
    expect(links[0].lineStyle.type).toBe("solid");
    expect(links[0].lineStyle.opacity).toBeGreaterThan(0);
  });
});
