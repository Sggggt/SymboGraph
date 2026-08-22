// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  makeGraphResponse,
  makeCoarseConceptNode,
  makeMidConceptNode,
  makeProjectionGroupDiagnostics,
  makeProjectionEdge,
  makeRelationEdge,
  makeRqPrefixNode,
} from "@/test/public-contract-fixtures";
import { GraphDiagnosticsPanel, GraphNodeSummary, RelatedEdgeList, nodeDetailRows } from "./graph-panel";

afterEach(() => cleanup());

describe("GraphNodeSummary", () => {
  it("renders natural-language concept details instead of raw metadata JSON", () => {
    const selectedNode = makeMidConceptNode("mid:bayes", {
      label: "贝叶斯更新",
      name: "贝叶斯更新",
      summary: "贝叶斯更新描述先验、似然和后验之间的证据更新关系。",
      confidence: 0.87,
      support_active_chunk_ids: ["chunk:prior", "chunk:posterior"],
    });
    const graph = makeGraphResponse({
      graph_type: "mid-concepts",
      counts: { mid_concepts: 2 },
      sampled_counts: { nodes: 2, edges: 1 },
      nodes: [
        selectedNode,
        makeMidConceptNode("mid:evidence", { label: "证据权重", name: "证据权重" }),
      ],
      edges: [makeRelationEdge("mid:bayes", "mid:evidence", { id: "edge:1", label: "concept_relation", weight: 0.73 })],
    });

    const { container } = render(<GraphNodeSummary node={selectedNode} graph={graph} graphType="mid-concepts" />);

    expect(screen.getByText("贝叶斯更新")).toBeTruthy();
    expect(container.textContent).toContain("这是中粒度概念节点");
    expect(container.textContent).toContain("自然语言定义");
    expect(container.textContent).toContain("贝叶斯更新描述先验、似然和后验之间的证据更新关系。");
    expect(container.textContent).toContain("2 个支撑片段");
    expect(container.textContent).toContain("相关概念");
    expect(container.textContent).not.toContain("internal_trace");
    expect(container.textContent).not.toContain("{");
  });

  it("summarizes semantic routing groups without exposing internal keys or paths", () => {
    const node = makeRqPrefixNode("rq:1-4", {
      label: "RQ L2 1/4",
      metadata: {
        rq_path: [],
        rq_prefix_key: "L2:1/4",
        rq_level: 2,
        rq_path_prefix: [1, 4],
        support_chunk_ids: ["chunk:alpha", "chunk:beta", "chunk:gamma", "chunk:delta"],
        representative_chunk_ids: ["chunk:alpha"],
        bridge_chunk_ids: ["chunk:bridge"],
        residual_norm_mean: 0.1234,
        residual_norm_max: 0.4567,
      },
    });

    const rows = nodeDetailRows(node, "chunk-relation", []);

    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "语义分组层级", value: "2" }),
        expect.objectContaining({ label: "支撑片段", value: expect.stringContaining("4 个支撑片段") }),
        expect.objectContaining({ label: "代表片段", value: "1 个代表片段" }),
      ]),
    );
    expect(rows.some((row) => row.value.includes("chunk:"))).toBe(false);
    expect(rows.some((row) => row.label.includes("前缀键") || row.label.includes("路径") || row.label.includes("残差"))).toBe(false);
  });

  it("shows every coarse membership role and the exact full union", () => {
    const node = makeCoarseConceptNode("coarse:bayes", {
      included_mid_concept_ids: ["mid:core"],
      boundary_mid_concept_ids: ["mid:boundary"],
      bridge_mid_concept_ids: ["mid:bridge"],
      outlier_mid_concept_ids: ["mid:outlier"],
      low_confidence_mid_concept_ids: ["mid:low"],
      all_mid_concept_ids: [
        "mid:core",
        "mid:boundary",
        "mid:bridge",
        "mid:outlier",
        "mid:low",
      ],
    });

    const rows = nodeDetailRows(node, "coarse-concepts", []);

    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "包含中概念",
          value: "1 个中概念",
        }),
        expect.objectContaining({
          label: "边界中概念",
          value: "1 个中概念",
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
          value: "1 个中概念",
        }),
        expect.objectContaining({
          label: "全部中概念",
          value: expect.stringContaining("5 个中概念"),
        }),
      ]),
    );
    expect(rows.some((row) => row.value.includes("mid:"))).toBe(false);
  });
});

describe("RelatedEdgeList", () => {
  it("renders a natural relationship without projection replay internals", () => {
    const source = makeMidConceptNode("mid:source");
    const target = makeMidConceptNode("mid:target", {
      label: "目标概念",
      name: "目标概念",
    });
    const edge = makeProjectionEdge(source.id, target.id);
    const graph = makeGraphResponse({
      graph_type: "mid-concepts",
      nodes: [source, target],
      edges: [edge],
    });

    render(<RelatedEdgeList node={source} graph={graph} relatedEdges={[edge]} />);

    expect(screen.getByText(/连接到/)).toBeTruthy();
    expect(screen.queryByTestId("projection-edge-replay")).toBeNull();
    expect(document.body.textContent).not.toContain("Q0.15");
    expect(document.body.textContent).not.toContain("protocol");
    expect(document.body.textContent).not.toContain("model calls");
  });
});

describe("GraphDiagnosticsPanel", () => {
  it("renders full hashes, distance calibration, projection distributions, and honest empty traversal state", () => {
    const rawDistribution = { count: 2, min: 0.1, mean: 0.2, max: 0.3, population_std: 0.1 };
    const calibratedDistribution = { count: 2, min: 0.08, mean: 0.16, max: 0.24, population_std: 0.08 };
    const semanticProjection = makeProjectionGroupDiagnostics({
      protocol_hashes: ["d".repeat(64)],
      protocol_hash_edge_count: 2,
      protocol_hash_coverage: 1,
      source_algorithm_coverage: 1,
      protocol_version_coverage: 1,
      state_hash_coverage: 1,
      full_edge_count: 2,
      supported_edge_count: 2,
      support_coverage: 1,
      raw_projected_distance_coverage: 1,
      calibrated_projected_distance_coverage: 1,
      raw_projected_strength_coverage: 1,
      raw_projected_distance_distribution: rawDistribution,
      calibrated_projected_distance_distribution: calibratedDistribution,
    });
    const graph = makeGraphResponse({
      graph_type: "mid-concepts",
      nodes: [],
      edges: [],
      counts: { mid_concept_edges: 2 },
      full_counts: { nodes: 3, edges: 2 },
      sampled_counts: { nodes: 0, edges: 0 },
      hash: "a".repeat(64),
      hashes: {
        chunk_scope_hash: "b".repeat(64),
        mid_concept_hash: "a".repeat(64),
        context_graph_hash: "c".repeat(64),
        local_hint_protocol_version: "context_graph_freshness_v1",
        runtime_settings_hash: "e".repeat(64),
        agent_operating_envelope_hash: "f".repeat(64),
      },
      edge_distance_diagnostics: {
        applicable: true,
        protocol_version: "edge_distance_log_calibrated_strength_v2",
        protocol_hashes: ["d".repeat(64)],
        distribution: rawDistribution,
        by_edge_type: {
          semantic: {
            distance: rawDistribution,
            calibration_stats_hashes: ["d".repeat(64)],
            calibration_stats_hash_consistent: true,
          },
        },
      },
      projection_diagnostics: {
        ...semanticProjection,
        applicable: true,
        protocol_version: "membership_q15_layer_type_calibrated_v3",
        by_edge_type: { semantic: semanticProjection },
        gray_predicates: {
          protocol_version: "projected_gray_predicates_v1",
          protocol_hash: "d".repeat(64),
          coverage: 1,
          missing_edge_count: 0,
          semantic_uncertain_edge_count: 0,
          crossing_rq_boundary_edge_count: 0,
          model_call_count: 0,
        },
        graph_total_edge_count: 2,
        non_projection_edge_count: 0,
      },
      retrieval_contribution: {
        trace_count: 0,
        has_observations: false,
        frontier_pops: 0,
        dominance_pruned_count: 0,
        expanded_edge_contribution: {},
        convergence_reasons: {},
        scores_json_primary: false,
      },
    });

    const { container } = render(<GraphDiagnosticsPanel graph={graph} />);

    expect(screen.getByText("状态与 freshness hashes")).toBeTruthy();
    expect(container.textContent).toContain("Mid concepts");
    expect(container.textContent).toContain("a".repeat(64));
    expect(container.textContent).toContain("Raw projected distance");
    expect(container.textContent).toContain("Calibrated projected distance");
    expect(container.textContent).toContain("semantic · raw");
    expect(container.textContent).toContain("hash 覆盖 100% · 一致 是");
    expect(container.textContent).toContain("2 / 2（100%）");
    expect(container.textContent).toContain("不把空数据标成“available”");
  });
});
