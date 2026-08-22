"use client";

import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { GraphDistribution, GraphHashes, GraphResponse, GraphType } from "@course-kg/shared";
import { motion } from "framer-motion";
import { Boxes, Expand, GitBranch, Layers3, Lock, Map, Minimize2, Network, RefreshCw, Unlock } from "lucide-react";

import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { NetworkCanvas, type NetworkCanvasHandle } from "@/components/network-canvas";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { fetchDashboard, fetchGraph } from "@/lib/api";
import { useLocalStorage } from "@/hooks/use-local-storage";

type SelectedNode = { id: string; category: string } | null;
type GraphNode = GraphResponse["nodes"][number];
type GraphEdge = GraphResponse["edges"][number];
type NodeDetailRow = { label: string; value: string; hint?: string };

const GRAPH_LAYERS: Array<{ type: GraphType; label: string; icon: typeof Network; description: string }> = [
  { type: "chunk-structure", label: "片段结构图", icon: Map, description: "标题、页面、坐标、表格、公式、图注和前后片段" },
  { type: "chunk-relation", label: "片段关系图", icon: Network, description: "内容相近、跨资料和跨语言的片段关系" },
  { type: "mid-concepts", label: "中粒度概念图", icon: GitBranch, description: "由原文片段支撑的具体概念和关系" },
  { type: "coarse-concepts", label: "粗粒度概念图", icon: Layers3, description: "聚合后的高层主题和跨主题关系" },
];

const EMPTY_VALUE = "无";

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function productDisplayLabel(value: unknown, fallback: string): string {
  const text = String(value ?? "").trim();
  if (
    !text ||
    /^[0-9a-f]{64}$/i.test(text) ||
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(text)
  ) {
    return fallback;
  }
  return text;
}

function graphLayerCounts(graph: GraphResponse) {
  return {
    full: { nodes: graph.node_counts.full, edges: graph.edge_counts.full },
    sampled: { nodes: graph.node_counts.sampled, edges: graph.edge_counts.sampled },
  };
}

function nodeLabel(node: GraphNode): string {
  return productDisplayLabel(node.name ?? node.label, "名称缺失");
}

function nodeCategory(node: GraphNode): string {
  const category = node.category ?? node.type ?? "chunk";
  const labels: Record<string, string> = {
    chunk: "片段",
    rq_prefix: "RQ 前缀",
    mid_concept: "中粒度概念",
    coarse_concept: "粗粒度概念",
    section: "章节",
    page: "页面",
    document: "文档",
    table: "表格",
    formula: "公式",
    caption: "图注",
  };
  return labels[category] ?? category;
}

function snippetForNode(node: GraphNode | null): string {
  if (!node) return "";
  return node.summary ?? node.snippet ?? node.text ?? "";
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item)).filter(Boolean);
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

function formatNumber(value: unknown, digits = 3): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? formatCount(value) : value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  return EMPTY_VALUE;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatPercent(value: unknown): string {
  const numeric = finiteNumber(value);
  return numeric === null ? EMPTY_VALUE : `${(numeric * 100).toFixed(1).replace(/\.0$/, "")}%`;
}

function distributionSummary(distribution: GraphDistribution | null | undefined): string {
  if (!distribution || distribution.count === 0) {
    return "0 项";
  }
  return `${formatCount(distribution.count)} 项 · min ${formatNumber(distribution.min)} · mean ${formatNumber(distribution.mean)} · max ${formatNumber(distribution.max)} · σ ${formatNumber(distribution.population_std)}`;
}

const HASH_LABELS: Partial<Record<keyof GraphHashes, string>> = {
  chunk_scope_hash: "Chunk scope",
  contextual_index_hash: "Contextual index",
  structure_graph_hash: "Structure graph",
  chunk_relation_graph_hash: "Relation graph",
  mid_concept_hash: "Mid concepts",
  coarse_concept_hash: "Coarse concepts",
  context_graph_hash: "Context graph",
};

function hashRows(graph: GraphResponse): Array<{ key: string; label: string; value: string }> {
  return (Object.entries(HASH_LABELS) as Array<[keyof GraphHashes, string]>)
    .map(([key, label]) => ({ key, label, value: String(graph.hashes[key] ?? "") }))
    .filter((item) => item.value.length > 0);
}

function formatPage(node: GraphNode): string {
  if (typeof node.page_number === "number") {
    return String(node.page_number);
  }
  if (Array.isArray(node.page_range) && node.page_range.length) {
    return node.page_range.filter((item) => item !== null && item !== undefined).join("-");
  }
  return EMPTY_VALUE;
}

function shortId(value: unknown): string {
  const text = String(value ?? "");
  if (!text) return EMPTY_VALUE;
  if (text.length <= 22) return text;
  return `${text.slice(0, 10)}...${text.slice(-8)}`;
}

function formatIdList(ids: unknown, noun: string): string {
  const values = stringList(ids);
  if (!values.length) {
    return EMPTY_VALUE;
  }
  return `${formatCount(values.length)} 个${noun}`;
}

function supportIdsForNode(node: GraphNode): string[] {
  return uniqueStrings([
    ...(node.support_chunk_ids ?? []),
    ...(node.support_active_chunk_ids ?? []),
  ].filter(Boolean));
}

function representativeIdsForNode(node: GraphNode): string[] {
  return uniqueStrings(node.representative_chunk_ids.filter(Boolean));
}

type CoarseGraphNode = Extract<
  GraphNode,
  { contract_kind: "coarse_concept_node" }
>;
type CoarseRoleField =
  | "included_mid_concept_ids"
  | "boundary_mid_concept_ids"
  | "bridge_mid_concept_ids"
  | "outlier_mid_concept_ids"
  | "low_confidence_mid_concept_ids"
  | "all_mid_concept_ids";

function coarseRoleIdsForNode(
  node: GraphNode,
  field: CoarseRoleField,
): string[] {
  if (node.contract_kind !== "coarse_concept_node") {
    return [];
  }
  const coarseNode: CoarseGraphNode = node;
  return uniqueStrings(coarseNode[field].filter(Boolean));
}

function includedMidIdsForNode(node: GraphNode): string[] {
  return coarseRoleIdsForNode(node, "included_mid_concept_ids");
}

function allMidIdsForNode(node: GraphNode): string[] {
  return coarseRoleIdsForNode(node, "all_mid_concept_ids");
}

function edgeLabel(edge: GraphEdge): string {
  const raw = edge.label ?? edge.type ?? edge.category ?? "关系边";
  const labels: Record<string, string> = {
    concept_relation: "相关概念",
    projected_dense_semantic: "语义相关",
    dense_semantic: "内容相关",
    dense_cross_document_bridge: "跨资料关联",
    dense_cross_language_bridge: "跨语言关联",
    co_occurs_with: "共同出现",
    bridge_to: "主题桥接",
    contains: "包含",
    previous: "前一片段",
    next: "后一片段",
    parent: "上级结构",
  };
  return labels[raw] ?? raw.replaceAll("_", " ");
}

function relatedEdgesForNode(graph: GraphResponse, nodeId: string): GraphEdge[] {
  return graph.edges.filter((edge) => edge.source === nodeId || edge.target === nodeId);
}

function relatedEdgeTypes(edges: GraphEdge[]): string {
  const types = Array.from(new Set(edges.map(edgeLabel).filter(Boolean)));
  if (!types.length) {
    return EMPTY_VALUE;
  }
  return types.slice(0, 4).join(" / ") + (types.length > 4 ? " ..." : "");
}

function appendRow(rows: NodeDetailRow[], label: string, value: unknown, hint?: string) {
  const text = typeof value === "string" ? value : value === null || value === undefined ? EMPTY_VALUE : String(value);
  if (!text || text === EMPTY_VALUE) {
    return;
  }
  rows.push({ label, value: text, hint });
}

function graphTypeLabel(graphType: GraphType): string {
  return GRAPH_LAYERS.find((layer) => layer.type === graphType)?.label ?? graphType;
}

function GraphStaleBadge({ isStale }: { isStale?: boolean }) {
  if (!isStale) {
    return null;
  }
  return (
    <span title="该图谱已过期，建议重建图谱" className="inline-flex rounded-full border border-rose-200/35 bg-rose-400/15 px-3 py-1 text-xs font-medium text-rose-50">
      已过期
    </span>
  );
}

function MetricCard({ label, value, hint }: { label: string; value: number | string; hint?: string }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-white/34">{label}</p>
      <p className="mt-1 text-lg font-semibold text-white/82">{value}</p>
      {hint ? <p className="mt-1 text-xs leading-5 text-white/42">{hint}</p> : null}
    </div>
  );
}

export function GraphDiagnosticsPanel({ graph }: { graph: GraphResponse }) {
  const distance = graph.edge_distance_diagnostics;
  const projection = graph.projection_diagnostics;
  const contribution = graph.retrieval_contribution;
  const hashes = hashRows(graph);
  const hasTraversal = contribution.has_observations || contribution.trace_count > 0;

  return (
    <section aria-label="图层诊断" className="mb-4 rounded-[22px] border border-white/8 bg-white/[0.025] p-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="section-kicker">可回放诊断</p>
          <p className="mt-2 text-sm text-white/56">完整状态统计与当前画布采样分开显示，避免用采样边推断全图。</p>
        </div>
        <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-white/55">当前层 hash：{shortId(graph.hash)}</span>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <article className="rounded-[18px] border border-white/8 bg-black/10 p-4">
          <h3 className="text-sm font-medium text-white/82">状态与 freshness hashes</h3>
          <div className="mt-3 space-y-2">
            {hashes.length ? (
              hashes.map((item) => (
                <div key={item.key} className="grid gap-1 text-xs sm:grid-cols-[130px_minmax(0,1fr)]">
                  <span className="text-white/42">{item.label}</span>
                  <code className="break-all text-cyan-100/72" title={item.value}>{item.value}</code>
                </div>
              ))
            ) : (
              <p className="text-sm text-amber-100/72">API 未返回 hash 卡，当前图层不能完成 freshness 核对。</p>
            )}
          </div>
        </article>

        <article className="rounded-[18px] border border-white/8 bg-black/10 p-4">
          <h3 className="text-sm font-medium text-white/82">边距离与类型校准</h3>
          {distance.applicable === true ? (
            <div className="mt-3 space-y-2 text-xs text-white/62">
              <p>完整分布：{distributionSummary(distance.distribution)}</p>
              <p>协议：{String(distance.protocol_version ?? EMPTY_VALUE)} · {shortId(distance.protocol_hash ?? (Array.isArray(distance.protocol_hashes) ? distance.protocol_hashes[0] : null))}</p>
              {Object.entries(distance.by_edge_type).length ? (
                <div className="space-y-1 border-t border-white/8 pt-2">
                  {Object.entries(distance.by_edge_type).slice(0, 6).map(([edgeType, payload]) => (
                    <p key={edgeType} className="break-words"><span className="text-white/40">{edgeType}</span> · {distributionSummary(payload.distance)}</p>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 text-sm text-white/48">本层不使用统一图距离：{String(distance.reason ?? "not_applicable")}</p>
          )}
        </article>

        <article className="rounded-[18px] border border-white/8 bg-black/10 p-4">
          <h3 className="text-sm font-medium text-white/82">投影支撑与校准</h3>
          {projection.applicable === true ? (
            <div className="mt-3 space-y-2 text-xs text-white/62">
              <p>支撑覆盖：{formatCount(finiteNumber(projection.supported_edge_count) ?? 0)} / {formatCount(finiteNumber(projection.full_edge_count) ?? 0)}（{formatPercent(projection.support_coverage)}）</p>
              <p>Raw projected distance：{distributionSummary(projection.raw_projected_distance_distribution)}</p>
              <p>Calibrated projected distance：{distributionSummary(projection.calibrated_projected_distance_distribution)}</p>
              <p>协议：{String(projection.protocol_version ?? EMPTY_VALUE)} · hash 覆盖 {formatPercent(projection.protocol_hash_coverage)} · 一致 {projection.protocol_hash_consistent === true ? "是" : "否"}</p>
              {Object.entries(projection.by_edge_type).length ? (
                <div className="space-y-1 border-t border-white/8 pt-2">
                  {Object.entries(projection.by_edge_type).slice(0, 6).map(([edgeType, typeDiagnostics]) => {
                    return (
                      <p key={edgeType} className="break-words">
                        <span className="text-white/40">{edgeType}</span> · raw {distributionSummary(typeDiagnostics.raw_projected_distance_distribution)} · calibrated {distributionSummary(typeDiagnostics.calibrated_projected_distance_distribution)}
                      </p>
                    );
                  })}
                </div>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 text-sm text-white/48">本层没有概念边投影：{String(projection.reason ?? "not_applicable")}</p>
          )}
        </article>

        <article className="rounded-[18px] border border-white/8 bg-black/10 p-4">
          <h3 className="text-sm font-medium text-white/82">最近检索遍历贡献</h3>
          {hasTraversal ? (
            <div className="mt-3 space-y-2 text-xs text-white/62">
              <p>{formatCount(contribution.trace_count)} 条 trace · frontier pops {formatCount(contribution.frontier_pops)} · dominance pruning {formatCount(contribution.dominance_pruned_count)}</p>
              <p>收敛：{Object.entries(contribution.convergence_reasons).map(([reason, count]) => `${reason} ${formatNumber(count)}`).join(" / ") || EMPTY_VALUE}</p>
              <p>Top edge contribution：{Object.entries(contribution.expanded_edge_contribution).slice(0, 3).map(([edgeId, score]) => `${shortId(edgeId)} ${formatPercent(score)}`).join(" / ") || EMPTY_VALUE}</p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-white/48">当前资料库还没有可汇总的检索 trace，不把空数据标成“available”。</p>
          )}
        </article>
      </div>
    </section>
  );
}

function nodeNaturalDescription(node: GraphNode, graphType: GraphType, relatedEdges: GraphEdge[]): string {
  const category = nodeCategory(node);
  const relationText = relatedEdges.length ? `当前采样视图里有 ${formatCount(relatedEdges.length)} 条关联边，关系类型包括 ${relatedEdgeTypes(relatedEdges)}。` : "当前采样视图里还没有返回与它直接相连的边。";
  if (graphType === "chunk-structure") {
    return `这是结构图里的${category}节点，用来把原文位置、章节路径、页面区域和片段上下文串起来。${relationText}`;
  }
  if (graphType === "chunk-relation") {
    if ((node.category ?? node.type) === "rq_prefix") {
      return `这是一个语义分组，汇集内容相近、可继续定位原文的片段。${relationText}`;
    }
    return `这是片段关系图里的${category}节点，系统会沿有证据支撑的关系寻找相邻内容。${relationText}`;
  }
  if (graphType === "mid-concepts") {
    return `这是中粒度概念节点，由原文片段支撑，用于从概念定位到具体证据。${relationText}`;
  }
  if (graphType === "coarse-concepts") {
    return `这是粗粒度概念节点，聚合多个中粒度概念，用于高层入口选择、主题区域定位和跨主题桥接。${relationText}`;
  }
  return `这是${category}节点。${relationText}`;
}

export function nodeDetailRows(node: GraphNode, graphType: GraphType, relatedEdges: GraphEdge[]): NodeDetailRow[] {
  const rows: NodeDetailRow[] = [];
  appendRow(rows, "节点类型", nodeCategory(node));
  appendRow(rows, "关联边", relatedEdges.length ? `${formatCount(relatedEdges.length)} 条` : EMPTY_VALUE, relatedEdgeTypes(relatedEdges));
  appendRow(rows, "页码", formatPage(node));

  if (graphType === "chunk-structure") {
    appendRow(rows, "结构路径", node.snippet ?? node.section_path?.join(" / "));
    appendRow(rows, "结构类别", node.type);
    return rows;
  }

  if (graphType === "chunk-relation") {
    if ((node.category ?? node.type) === "rq_prefix") {
      appendRow(rows, "语义分组层级", node.metadata.rq_level);
      appendRow(rows, "支撑片段", formatIdList(node.metadata.support_chunk_ids, "支撑片段"));
      appendRow(rows, "代表片段", formatIdList(node.metadata.representative_chunk_ids, "代表片段"));
      appendRow(rows, "桥接片段", formatIdList(node.metadata.bridge_chunk_ids, "桥接片段"));
      return rows;
    }
    appendRow(rows, "支撑片段", formatIdList(supportIdsForNode(node), "支撑片段"));
    return rows;
  }

  if (graphType === "mid-concepts") {
    appendRow(rows, "定义置信度", formatNumber(node.confidence));
    appendRow(rows, "支撑片段", formatIdList(supportIdsForNode(node), "支撑片段"));
    appendRow(rows, "代表片段", formatIdList(representativeIdsForNode(node), "代表片段"));
    return rows;
  }

  if (graphType === "coarse-concepts") {
    const includedMidIds = includedMidIdsForNode(node);
    appendRow(rows, "定义置信度", formatNumber(node.confidence));
    appendRow(rows, "包含中概念", formatIdList(includedMidIds, "中概念"));
    appendRow(rows, "边界中概念", formatIdList(coarseRoleIdsForNode(node, "boundary_mid_concept_ids"), "中概念"));
    appendRow(rows, "桥接中概念", formatIdList(coarseRoleIdsForNode(node, "bridge_mid_concept_ids"), "中概念"));
    appendRow(rows, "离群中概念", formatIdList(coarseRoleIdsForNode(node, "outlier_mid_concept_ids"), "中概念"));
    appendRow(rows, "低置信中概念", formatIdList(coarseRoleIdsForNode(node, "low_confidence_mid_concept_ids"), "中概念"));
    appendRow(rows, "全部中概念", formatIdList(allMidIdsForNode(node), "中概念"));
    appendRow(rows, "支撑片段", formatIdList(node.support_chunk_ids, "支撑片段"));
    return rows;
  }

  appendRow(rows, "分数", formatNumber(node.score ?? node.importance_score));
  appendRow(rows, "支撑数", node.support_count ?? node.support_chunk_ids?.length);
  return rows;
}

function DetailMetricGrid({ rows }: { rows: NodeDetailRow[] }) {
  return (
    <div className="mt-4 grid gap-2">
      {rows.map((row) => (
        <div key={`${row.label}:${row.value}`} className="rounded-[18px] border border-white/8 bg-white/[0.035] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.18em] text-white/38">{row.label}</p>
          <p className="mt-1 min-w-0 break-words text-sm font-semibold leading-6 text-white/82">{row.value}</p>
          {row.hint ? <p className="mt-1 break-words text-xs leading-5 text-white/45">{row.hint}</p> : null}
        </div>
      ))}
    </div>
  );
}

function DetailSection({ title, children, description }: { title: string; children: ReactNode; description?: string }) {
  return (
    <section className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
      <p className="text-xs uppercase tracking-[0.22em] text-white/45">{title}</p>
      {description ? <p className="mt-2 break-words text-sm leading-7 text-white/58">{description}</p> : null}
      {children}
    </section>
  );
}

function nodeDetailLeadMetrics(node: GraphNode, graphType: GraphType, relatedEdges: GraphEdge[]): NodeDetailRow[] {
  const supportCount = supportIdsForNode(node).length || node.metadata.support_chunk_ids.length || node.support_count || 0;
  const rows: NodeDetailRow[] = [
    { label: "图层", value: graphTypeLabel(graphType) },
    { label: "类型", value: nodeCategory(node) },
    { label: "关联", value: relatedEdges.length ? `${formatCount(relatedEdges.length)} 条` : "无直接边" },
  ];
  if (supportCount > 0) {
    rows.push({ label: "支撑", value: `${formatCount(supportCount)} 个证据` });
  }
  if (node.confidence !== null && node.confidence !== undefined) {
    rows.push({ label: "置信度", value: formatNumber(node.confidence) });
  }
  const page = formatPage(node);
  if (page !== EMPTY_VALUE) {
    rows.push({ label: "页码", value: page });
  }
  return rows.slice(0, 6);
}

function nodeSupportNarrative(node: GraphNode, graphType: GraphType): string {
  if (graphType === "chunk-structure") {
    const path = node.snippet ?? node.section_path?.join(" / ");
    return path ? `这个结构节点把图谱中的位置恢复到原文路径：${path}。它主要用于定位章节、页面区域、公式、表格、图注和前后片段。` : "这个结构节点用于恢复原文位置，但当前响应没有携带更细的结构路径。";
  }
  if (graphType === "chunk-relation") {
    if ((node.category ?? node.type) === "rq_prefix") {
      const support = formatIdList(node.metadata.support_chunk_ids, "支撑片段");
      const reps = formatIdList(node.metadata.representative_chunk_ids, "代表片段");
      return `这个语义分组帮助系统从主题定位到原文。${support !== EMPTY_VALUE ? support : "当前没有返回支撑片段"}；${reps !== EMPTY_VALUE ? reps : "当前没有返回代表片段"}。`;
    }
    return "这个片段节点来自原文证据，系统会沿有支撑的语义关系寻找相邻内容。";
  }
  if (graphType === "mid-concepts") {
    const support = formatIdList(supportIdsForNode(node), "支撑片段");
    const representatives = formatIdList(representativeIdsForNode(node), "代表片段");
    return `这个中粒度概念可以回到原文证据。${support !== EMPTY_VALUE ? support : "当前没有可展示的支撑片段"}；${representatives !== EMPTY_VALUE ? representatives : "当前没有可展示的代表片段"}。`;
  }
  if (graphType === "coarse-concepts") {
    const mids = formatIdList(allMidIdsForNode(node), "中概念");
    return `这个粗粒度概念负责把多个中粒度概念聚合成高层主题入口。${mids !== EMPTY_VALUE ? mids : "当前响应没有返回包含的中概念列表"}。`;
  }
  return "当前节点没有额外支撑说明。";
}

export function RelatedEdgeList({ node, graph, relatedEdges }: { node: GraphNode; graph: GraphResponse; relatedEdges: GraphEdge[] }) {
  const nodeById = useMemo(() => new globalThis.Map(graph.nodes.map((item) => [item.id, item])), [graph.nodes]);
  if (!relatedEdges.length) {
    return <p className="mt-3 text-sm leading-7 text-white/52">当前采样视图没有返回直接相连的边；可以切换图层或扩大后端采样上限查看更多邻接关系。</p>;
  }
  return (
    <div className="mt-3 space-y-2">
      {relatedEdges.slice(0, 6).map((edge, index) => {
        const neighborId = edge.source === node.id ? edge.target : edge.source;
        const neighbor = nodeById.get(neighborId);
        return (
          <div key={edge.id ?? `${edge.source}:${edge.target}:${index}`} className="rounded-[16px] border border-white/8 bg-white/[0.03] px-3 py-3 text-sm">
            <div className="flex min-w-0 items-center justify-between gap-3">
              <span className="min-w-0 break-words font-medium text-white/78">{edgeLabel(edge)}</span>
            </div>
            <p className="mt-1 break-words text-xs leading-5 text-white/50">连接到 {neighbor ? nodeLabel(neighbor) : "相邻节点"}</p>
          </div>
        );
      })}
      {relatedEdges.length > 6 ? <p className="text-xs text-white/42">还有 {formatCount(relatedEdges.length - 6)} 条关联边未展开。</p> : null}
    </div>
  );
}

function GraphPanelContent({ selectedKnowledgeBaseId }: { selectedKnowledgeBaseId: string | null }) {
  const storageScope = selectedKnowledgeBaseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedKnowledgeBaseId],
    queryFn: () => fetchDashboard(selectedKnowledgeBaseId, { includeGraph: false }),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const [selectedGraphType, setSelectedGraphType] = useLocalStorage<GraphType>(`graph.selectedLayer.${storageScope}`, "chunk-relation");
  const [selectedNode, setSelectedNode] = useLocalStorage<SelectedNode>(`graph.selectedNode.${storageScope}`, null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sidePanelHeight, setSidePanelHeight] = useState<number | null>(null);
  const [isLocked, setIsLocked] = useState(false);
  const graphView: GraphResponse["view"] = "overview";
  const canvasRef = useRef<NetworkCanvasHandle | null>(null);
  const fullscreenRef = useRef<HTMLDivElement | null>(null);
  const mainPanelRef = useRef<HTMLElement | null>(null);

  const graphQuery = useQuery({
    queryKey: ["knowledge-graph", selectedKnowledgeBaseId, selectedGraphType, graphView],
    queryFn: () => fetchGraph(selectedKnowledgeBaseId, selectedGraphType, graphView),
    enabled: Boolean(selectedKnowledgeBaseId),
  });

  const activeLayer = GRAPH_LAYERS.find((layer) => layer.type === selectedGraphType) ?? GRAPH_LAYERS[1];
  const selectedGraphNode = useMemo(
    () => (selectedNode && graphQuery.data ? graphQuery.data.nodes.find((node) => node.id === selectedNode.id) ?? null : null),
    [selectedNode, graphQuery.data],
  );

  useEffect(() => {
    if (selectedNode && graphQuery.data && !graphQuery.data.nodes.some((node) => node.id === selectedNode.id)) {
      setSelectedNode(null);
    }
  }, [selectedNode, setSelectedNode, graphQuery.data]);

  useEffect(() => {
    const handleChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  useEffect(() => {
    if (isFullscreen) {
      return undefined;
    }
    const element = mainPanelRef.current;
    if (!element) {
      return undefined;
    }
    let frame: number | null = null;
    const updateHeight = () => {
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
      }
      frame = window.requestAnimationFrame(() => {
        frame = null;
        setSidePanelHeight(Math.ceil(element.getBoundingClientRect().height));
      });
    };
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(element);
    window.addEventListener("resize", updateHeight);
    return () => {
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
      }
      observer.disconnect();
      window.removeEventListener("resize", updateHeight);
    };
  }, [isFullscreen, selectedGraphType, graphQuery.data]);

  const handleLayerChange = (graphType: GraphType) => {
    setSelectedGraphType(graphType);
    setSelectedNode(null);
    setIsLocked(false);
  };

  const toggleFullscreen = async () => {
    if (!fullscreenRef.current) return;
    if (!document.fullscreenElement) {
      await fullscreenRef.current.requestFullscreen();
      return;
    }
    await document.exitFullscreen();
  };

  if (dashboardQuery.isLoading || graphQuery.isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (dashboardQuery.error || graphQuery.error) {
    return <ErrorBlock message={(dashboardQuery.error as Error | undefined)?.message ?? (graphQuery.error as Error).message} />;
  }
  if (!graphQuery.data || !dashboardQuery.data) {
    return null;
  }

  const graph = graphQuery.data;
  const freshnessReason = graph.stale_reason ?? graph.freshness.stale_reasons[0];
  const grounding = graph.grounding;
  const treeNodes = dashboardQuery.data.tree;
  const layerCounts = graphLayerCounts(graph);
  const sidePanelStyle = !isFullscreen && sidePanelHeight ? { height: sidePanelHeight, maxHeight: sidePanelHeight } : undefined;

  return (
    <div
      ref={fullscreenRef}
      className={`relative grid items-stretch gap-4 ${isFullscreen ? "min-h-screen bg-[rgba(3,8,24,0.98)] p-4" : "kg-page kg-graph-page xl:grid-cols-[280px_minmax(0,1fr)_360px]"}`}
    >
      {!isFullscreen ? (
        <motion.section initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} style={sidePanelStyle} className="glass-panel kg-scroll-panel h-full rounded-[28px] p-5">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="section-kicker">结构索引</p>
              <h2 className="mt-2 break-words text-2xl font-semibold text-white">文档与结构路径</h2>
            </div>
            <Boxes className="size-5 shrink-0 text-cyan-200" />
          </div>
          <div className="mt-6 space-y-4">
            {treeNodes.map((node, nodeIndex) => (
              <div key={node.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] px-4 py-4">
                <p className="break-words text-base font-medium text-white">{productDisplayLabel(node.title ?? node.label, `文档 ${nodeIndex + 1}`)}</p>
                <div className="mt-3 space-y-2">
                  {(node.children ?? []).map((child, childIndex) => (
                    <div key={child.id} className="rounded-[16px] border border-white/8 px-4 py-3 text-sm leading-6 text-white/62 break-words">
                      {productDisplayLabel(child.title ?? child.label, `章节 ${childIndex + 1}`)}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </motion.section>
      ) : null}

      <motion.section
        ref={mainPanelRef}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className={`glass-panel min-w-0 rounded-[30px] ${isFullscreen ? "col-span-full flex min-h-[calc(100vh-2rem)] flex-col p-3" : "flex h-full min-h-0 flex-col p-4 lg:p-5"}`}
      >
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4 px-2">
          <div className="min-w-0">
            <p className="section-kicker">四层上下文图谱</p>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <h2 className="break-words text-3xl font-semibold text-white">{activeLayer.label}</h2>
              <GraphStaleBadge isStale={Boolean(graph.freshness?.is_stale)} />
              {freshnessReason ? <span className="rounded-full border border-amber-200/25 bg-amber-300/[0.08] px-3 py-1.5 text-amber-50">{freshnessReason}</span> : null}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex flex-wrap items-center gap-1 rounded-full border border-white/8 bg-white/[0.04] p-1">
              {GRAPH_LAYERS.map((layer) => {
                const Icon = layer.icon;
                const active = selectedGraphType === layer.type;
                return (
                  <button
                    key={layer.type}
                    type="button"
                    onClick={() => handleLayerChange(layer.type)}
                    className={`inline-flex h-9 items-center gap-2 rounded-full px-3 text-xs font-medium transition ${
                      active ? "bg-cyan-300/15 text-cyan-50 shadow-[0_0_24px_rgba(103,232,249,0.10)]" : "text-white/48 hover:bg-white/[0.06] hover:text-white/78"
                    }`}
                    title={layer.description}
                  >
                    <Icon className="size-4" />
                    <span>{layer.label}</span>
                  </button>
                );
              })}
            </div>
            <motion.button
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.98 }}
              type="button"
              className="action-chip rounded-full px-4 py-2 text-xs uppercase tracking-[0.22em]"
              onClick={() => {
                canvasRef.current?.resetView();
                canvasRef.current?.fitView();
                setIsLocked(false);
              }}
            >
              <RefreshCw className="mr-2 inline size-4" />
              重置视图
            </motion.button>
            <motion.button
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.98 }}
              type="button"
              className="action-chip rounded-full px-4 py-2 text-xs uppercase tracking-[0.22em]"
              onClick={() => setIsLocked(Boolean(canvasRef.current?.toggleLayoutLock()))}
            >
              {isLocked ? <Lock className="mr-2 inline size-4" /> : <Unlock className="mr-2 inline size-4" />}
              {isLocked ? "已锁定" : "锁定布局"}
            </motion.button>
            <motion.button whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }} type="button" className="action-chip rounded-full px-4 py-2 text-xs uppercase tracking-[0.22em]" onClick={toggleFullscreen}>
              {isFullscreen ? <Minimize2 className="mr-2 inline size-4" /> : <Expand className="mr-2 inline size-4" />}
              {isFullscreen ? "退出全屏" : "全屏查看"}
            </motion.button>
          </div>
        </div>

        <div className="mb-4 grid gap-2 px-2 text-xs text-white/52 sm:grid-cols-4">
          <MetricCard label="节点" value={formatCount(layerCounts.full.nodes)} hint={`采样 ${formatCount(layerCounts.sampled.nodes)}`} />
          <MetricCard label="边" value={formatCount(layerCounts.full.edges)} hint={`采样 ${formatCount(layerCounts.sampled.edges)}`} />
          <MetricCard
            label="支撑状态"
            value={graph.graph_type === "coarse-concepts" ? formatPercent(grounding.coarse_grounded_rate) : formatPercent(grounding.mid_grounded_rate)}
            hint="概念可回到原文证据"
          />
          <MetricCard
            label="状态"
            value={graph.freshness?.is_stale ? "需要更新" : "已就绪"}
            hint={graph.freshness?.is_stale ? "请重新构建当前资料库" : "可用于搜索和问答"}
          />
        </div>

        <div className={`grid min-h-0 flex-1 gap-4 ${isFullscreen ? "grid-cols-[minmax(0,1fr)_380px]" : "grid-cols-1"}`}>
          <div className="min-w-0 rounded-[24px] border border-white/8 bg-[rgba(4,9,24,0.36)] p-2">
            <NetworkCanvas
              key={selectedGraphType}
              ref={canvasRef}
              graph={graph}
              height={isFullscreen ? 900 : 760}
              selectedNodeId={selectedNode?.id ?? null}
              onNodeClick={(nodeId, category) => setSelectedNode({ id: nodeId, category })}
              onNodeDoubleClick={(nodeId, category) => setSelectedNode({ id: nodeId, category })}
            />
          </div>

          {isFullscreen && (
            <aside className="glass-panel kg-scroll-panel min-w-0 rounded-[24px] p-5">
              <GraphNodeSummary node={selectedGraphNode} graph={graph} graphType={selectedGraphType} />
            </aside>
          )}
        </div>
      </motion.section>

      {!isFullscreen ? (
        <motion.section initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} style={sidePanelStyle} className="glass-panel kg-scroll-panel h-full min-w-0 rounded-[28px] p-5">
          <GraphNodeSummary node={selectedGraphNode} graph={graph} graphType={selectedGraphType} />
        </motion.section>
      ) : null}
    </div>
  );
}

export function GraphPanel() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  return <GraphPanelContent key={selectedKnowledgeBaseId ?? "unassigned"} selectedKnowledgeBaseId={selectedKnowledgeBaseId} />;
}

export function GraphNodeSummary({ node, graph, graphType }: { node: GraphNode | null; graph: GraphResponse; graphType: GraphType }) {
  if (!node) {
    return (
      <section className="mt-6 rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
        <p className="section-kicker">节点详情</p>
        <h2 className="mt-2 break-words text-2xl font-semibold text-white">图谱节点解读</h2>
        <p className="mt-4 break-words text-sm leading-7 text-white/58">双击图中的节点后，这里会按当前图层展示它的证据来源、邻接关系和检索作用。</p>
      </section>
    );
  }
  const relatedEdges = relatedEdgesForNode(graph, node.id);
  const rows = nodeDetailRows(node, graphType, relatedEdges);
  const leadMetrics = nodeDetailLeadMetrics(node, graphType, relatedEdges);
  const text = snippetForNode(node);
  const textTitle = graphType === "mid-concepts" || graphType === "coarse-concepts" ? "自然语言定义" : graphType === "chunk-structure" ? "结构说明" : "支撑文本";
  return (
    <div className="mt-6 space-y-5">
      <section className="rounded-[24px] border border-cyan-200/10 bg-cyan-300/[0.035] p-5">
        <p className="section-kicker">节点详情</p>
        <p className="mt-3 break-words text-xs uppercase tracking-[0.26em] text-cyan-100/58">{graphTypeLabel(graphType)}</p>
        <h2 className="mt-3 break-words text-3xl font-semibold leading-tight text-white">{nodeLabel(node)}</h2>
        <p className="mt-4 break-words text-sm leading-8 text-white/68">{nodeNaturalDescription(node, graphType, relatedEdges)}</p>
        <div className="mt-5 flex flex-wrap gap-2">
          {leadMetrics.map((metric) => (
            <span key={`${metric.label}:${metric.value}`} className="max-w-full rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-cyan-50/78">
              <span className="text-white/42">{metric.label}</span>
              <span className="ml-2 break-words font-medium text-white/82">{metric.value}</span>
            </span>
          ))}
        </div>
      </section>

      <DetailSection title="关键数据" description="这里只显示帮助理解节点作用的少量信息。">
        <DetailMetricGrid rows={rows} />
      </DetailSection>

      <DetailSection title="证据与定位">
        <p className="mt-4 break-words text-sm leading-8 text-white/66">{nodeSupportNarrative(node, graphType)}</p>
      </DetailSection>

      <DetailSection title="相邻关系" description="显示当前画布中与该节点直接相连的关系。">
        <RelatedEdgeList node={node} graph={graph} relatedEdges={relatedEdges} />
      </DetailSection>

      <DetailSection title={textTitle}>
        {text ? (
          <MarkdownRenderer content={text} compact className="mt-4 break-words text-white/68" />
        ) : (
          <p className="mt-4 break-words text-sm leading-7 text-white/52">当前节点没有返回可展示的定义或原文片段；可以切换图层查看更低层的支撑节点。</p>
        )}
      </DetailSection>
    </div>
  );
}
