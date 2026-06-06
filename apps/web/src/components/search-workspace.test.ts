import { describe, expect, it } from "vitest";

import type { GraphResponse } from "@course-kg/shared";
import { toPureSemanticGraph } from "@/components/search-workspace";

describe("toPureSemanticGraph", () => {
  it("keeps only semantic entities and semantic relations for the search canvas", () => {
    const graph: GraphResponse = {
      graph_type: "evidence",
      schema_version: "typed_graph_v1",
      node_counts: { semantic_entity: 2, evidence_chunk: 1, document_version: 1 },
      edge_counts: { semantic: 1, evidence: 2 },
      freshness: { is_stale: false },
      nodes: [
        { id: "semantic:a", name: "Independence", category: "semantic_entity" },
        { id: "semantic:b", name: "Mutual exclusivity", category: "semantic_entity" },
        { id: "evidence_chunk:c1", name: "Evidence", category: "evidence_chunk" },
        { id: "document_version:v1", name: "Lecture", category: "document_version" },
      ],
      edges: [
        { source: "semantic:a", target: "semantic:b", label: "contrasts_with", category: "semantic" },
        { source: "semantic:a", target: "evidence_chunk:c1", label: "evidenced_by", category: "evidence" },
        { source: "evidence_chunk:c1", target: "document_version:v1", label: "from_version", category: "evidence" },
      ],
    };

    const semanticGraph = toPureSemanticGraph(graph);

    expect(semanticGraph?.graph_type).toBe("semantic");
    expect(semanticGraph?.nodes.map((node) => node.id)).toEqual(["semantic:a", "semantic:b"]);
    expect(semanticGraph?.edges.map((edge) => edge.label)).toEqual(["contrasts_with"]);
    expect(semanticGraph?.node_counts).toEqual({ semantic_entity: 2 });
    expect(semanticGraph?.edge_counts).toEqual({ semantic: 1 });
  });
});
