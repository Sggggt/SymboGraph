// @vitest-environment jsdom

import type { GraphNode } from "@course-kg/shared";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  makeCoarseConceptNode,
  makeGraphResponse,
} from "@/test/public-contract-fixtures";
import { GraphNodeSummary, nodeDetailRows } from "./graph-panel";

type CoarseGraphNode = Extract<
  GraphNode,
  { contract_kind: "coarse_concept_node" }
>;

type RequiredCoarseRoleContract = {
  included_mid_concept_ids: string[];
  boundary_mid_concept_ids: string[];
  bridge_mid_concept_ids: string[];
  outlier_mid_concept_ids: string[];
  low_confidence_mid_concept_ids: string[];
  all_mid_concept_ids: string[];
};

type NonCoarseGraphNode = Exclude<
  GraphNode,
  { contract_kind: "coarse_concept_node" }
>;
type LeakedRoleKey = Extract<
  keyof NonCoarseGraphNode,
  keyof RequiredCoarseRoleContract
>;
const NON_COARSE_ROLE_KEYS_ARE_FORBIDDEN: [LeakedRoleKey] extends [
  never,
]
  ? true
  : false = true;

// This assignment is intentionally compile-time significant. It rejects a
// shared contract that omits the fields or weakens them to optional/open data.
function requireStrongCoarseRoleContract(
  node: CoarseGraphNode,
): RequiredCoarseRoleContract {
  return node;
}

const roleFields: RequiredCoarseRoleContract = {
  included_mid_concept_ids: ["mid:included", "mid:shared"],
  boundary_mid_concept_ids: ["mid:boundary", "mid:shared"],
  bridge_mid_concept_ids: ["mid:bridge"],
  outlier_mid_concept_ids: ["mid:outlier"],
  low_confidence_mid_concept_ids: [
    "mid:low-confidence",
    "mid:bridge",
  ],
  all_mid_concept_ids: [
    "mid:included",
    "mid:shared",
    "mid:boundary",
    "mid:bridge",
    "mid:outlier",
    "mid:low-confidence",
  ],
};

describe("coarse GraphNode role public contract", () => {
  it("keeps every role strongly typed and renders each role plus the complete union", () => {
    expect(NON_COARSE_ROLE_KEYS_ARE_FORBIDDEN).toBe(true);
    const node = {
      ...makeCoarseConceptNode("coarse:roles", {
        label: "粗粒度角色契约",
        name: "粗粒度角色契约",
      }),
      ...roleFields,
    };
    const typedRoles = requireStrongCoarseRoleContract(node);
    expect(typedRoles).toMatchObject(roleFields);

    const rows = nodeDetailRows(node, "coarse-concepts", []);
    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "包含中概念",
          value: "2 个中概念",
        }),
        expect.objectContaining({
          label: "边界中概念",
          value: "2 个中概念",
        }),
        expect.objectContaining({
          label: "桥接中概念",
          value: "1 个中概念",
        }),
        expect.objectContaining({
          label: "离群中概念",
          value: "1 个中概念",
        }),
        expect.objectContaining({
          label: "低置信中概念",
          value: "2 个中概念",
        }),
        expect.objectContaining({
          label: "全部中概念",
          value: expect.stringContaining("6 个中概念"),
        }),
      ]),
    );

    const graph = makeGraphResponse({
      graph_type: "coarse-concepts",
      nodes: [node],
    });
    const { container } = render(
      <GraphNodeSummary
        node={node}
        graph={graph}
        graphType="coarse-concepts"
      />,
    );
    for (const text of [
      "包含中概念",
      "边界中概念",
      "桥接中概念",
      "离群中概念",
      "低置信中概念",
      "全部中概念",
    ]) {
      expect(container.textContent).toContain(text);
    }
    expect(container.textContent).not.toContain("mid:");
  });

  it("does not substitute chunk support when included is empty but other roles exist", () => {
    const node = {
      ...makeCoarseConceptNode("coarse:no-included", {
        label: "非 included 角色",
        support_active_chunk_ids: ["chunk:not-a-mid-concept"],
      }),
      included_mid_concept_ids: [],
      boundary_mid_concept_ids: ["mid:boundary-only"],
      bridge_mid_concept_ids: ["mid:bridge-only"],
      outlier_mid_concept_ids: [],
      low_confidence_mid_concept_ids: ["mid:low-only"],
      all_mid_concept_ids: [
        "mid:boundary-only",
        "mid:bridge-only",
        "mid:low-only",
      ],
    };
    requireStrongCoarseRoleContract(node);
    const graph = makeGraphResponse({
      graph_type: "coarse-concepts",
      nodes: [node],
    });

    const { container } = render(
      <GraphNodeSummary
        node={node}
        graph={graph}
        graphType="coarse-concepts"
      />,
    );

    expect(container.textContent).toContain("边界中概念1 个中概念");
    expect(container.textContent).toContain("桥接中概念1 个中概念");
    expect(container.textContent).toContain("低置信中概念1 个中概念");
    expect(container.textContent).toContain("3 个中概念");
    expect(container.textContent).not.toContain("mid:");
    expect(container.textContent).not.toContain(
      "chunk:not-a-mid-concept",
    );
  });
});
