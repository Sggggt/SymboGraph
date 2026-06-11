import { describe, expect, it } from "vitest";

import type { GraphResponse } from "@course-kg/shared";
import { toQueryEvidenceGraph } from "@/components/search-workspace";

describe("toQueryEvidenceGraph", () => {
  it("keeps evidence graph nodes and edges for the search canvas", () => {
    const graph: GraphResponse = {
      graph_type: "evidence",
      schema_version: "typed_graph_v1",
      node_counts: { evidence_atom: 1, active_chunk: 1, document_version: 1 },
      edge_counts: { observation: 1, active_chunk: 1 },
      freshness: { is_stale: false },
      nodes: [
        { id: "atom:a", name: "Independence", category: "evidence_atom" },
        { id: "active_chunk:c1", name: "Evidence", category: "active_chunk" },
        { id: "document_version:v1", name: "Lecture", category: "document_version" },
      ],
      edges: [
        { source: "active_chunk:c1", target: "atom:a", label: "grounded_by", category: "active_chunk" },
        { source: "atom:a", target: "document_version:v1", label: "from_version", category: "traceability" },
      ],
    };

    const evidenceGraph = toQueryEvidenceGraph(graph);

    expect(evidenceGraph?.graph_type).toBe("evidence");
    expect(evidenceGraph?.nodes.map((node) => node.id)).toEqual(["atom:a", "active_chunk:c1", "document_version:v1"]);
    expect(evidenceGraph?.edges.map((edge) => edge.label)).toEqual(["grounded_by", "from_version"]);
    expect(evidenceGraph?.node_counts).toEqual({ evidence_atom: 1, active_chunk: 1, document_version: 1 });
    expect(evidenceGraph?.edge_counts).toEqual({ observation: 1, active_chunk: 1 });
  });
});
