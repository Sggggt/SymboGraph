"use client";

import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { GraphResponse, GraphType } from "@course-kg/shared";
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
  { type: "chunk-relation", label: "片段关系图", icon: Network, description: "Dense 语义边、跨文档桥边、跨语言桥边与 RQ membership 诊断" },
  { type: "mid-concepts", label: "中粒度概念图", icon: GitBranch, description: "由片段和RQ 前缀支撑的中粒度概念和关系" },
  { type: "coarse-concepts", label: "粗粒度概念图", icon: Layers3, description: "社区、桥接概念、弱边和主题区域" },
];

const EMPTY_VALUE = "无";

function statNumber(values: Record<string, number> | undefined, key: string) {
  const value = values?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function graphLayerCounts(graph: GraphResponse) {
  const diagnosticFullCounts = graph.diagnostics?.layer_full_counts as Record<string, number> | undefined;
  const fullNodes = statNumber(graph.node_counts, "full") || statNumber(graph.full_counts, "nodes") || statNumber(diagnosticFullCounts, "nodes");
  const fullEdges = statNumber(graph.edge_counts, "full") || statNumber(graph.full_counts, "edges") || statNumber(diagnosticFullCounts, "edges");
  const sampledNodes = statNumber(graph.node_counts, "sampled") || statNumber(graph.sampled_counts, "nodes") || graph.nodes.length;
  const sampledEdges = statNumber(graph.edge_counts, "sampled") || statNumber(graph.sampled_counts, "edges") || graph.edges.length;
  return {
    full: { nodes: fullNodes, edges: fullEdges },
    sampled: { nodes: sampledNodes, edges: sampledEdges },
  };
}

function nodeLabel(node: GraphNode): string {
  return node.name ?? node.label ?? node.id;
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
  return node.summary ?? node.snippet ?? node.text ?? (typeof node.metadata?.text === "string" ? node.metadata.text : "");
}

function metadataNumber(node: GraphNode, key: string): number | null {
  const value = node.metadata?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metadataPath(node: GraphNode, key: string): string {
  const value = node.metadata?.[key];
  return Array.isArray(value) && value.length ? value.join("/") : EMPTY_VALUE;
}

function metadataRecord(node: GraphNode, key: string): Record<string, unknown> {
  const value = node.metadata?.[key];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item)).filter(Boolean);
}

function metadataStringList(node: GraphNode, key: string): string[] {
  return stringList(node.metadata?.[key]);
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
  const preview = values.slice(0, 3).map(shortId).join("、");
  return `${formatCount(values.length)} 个${noun}：${preview}${values.length > 3 ? " 等" : ""}`;
}

function supportIdsForNode(node: GraphNode): string[] {
  return uniqueStrings([
    ...(node.support_chunk_ids ?? []),
    ...(node.support_active_chunk_ids ?? []),
    ...metadataStringList(node, "support_chunk_ids"),
  ].filter(Boolean));
}

function representativeIdsForNode(node: GraphNode): string[] {
  return uniqueStrings([...(node.representative_chunk_ids ?? []), ...metadataStringList(node, "representative_chunk_ids")].filter(Boolean));
}

function includedMidIdsForNode(node: GraphNode): string[] {
  return uniqueStrings([...(node.included_mid_concept_ids ?? []), ...metadataStringList(node, "included_mid_concept_ids")].filter(Boolean));
}

function edgeLabel(edge: GraphEdge): string {
  return edge.label ?? edge.type ?? edge.category ?? "关系边";
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

function statsNumber(node: GraphNode, key: string): string {
  return formatNumber(metadataRecord(node, "stats")[key]);
}

function statusLabel(value: unknown): string {
  const text = typeof value === "string" && value.trim() ? value : "recorded";
  const labels: Record<string, string> = {
    recorded: "已记录",
    available: "可用",
    unavailable: "不可用",
    missing: "缺失",
    stale: "已过期",
    fresh: "最新",
    active: "已激活",
    pending: "等待中",
  };
  return labels[text] ?? text;
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

function nodeNaturalDescription(node: GraphNode, graphType: GraphType, relatedEdges: GraphEdge[]): string {
  const category = nodeCategory(node);
  const relationText = relatedEdges.length ? `当前采样视图里有 ${formatCount(relatedEdges.length)} 条关联边，关系类型包括 ${relatedEdgeTypes(relatedEdges)}。` : "当前采样视图里还没有返回与它直接相连的边。";
  if (graphType === "chunk-structure") {
    return `这是结构图里的${category}节点，用来把原文位置、章节路径、页面区域和片段上下文串起来。${relationText}`;
  }
  if (graphType === "chunk-relation") {
    if ((node.category ?? node.type) === "rq_prefix") {
      return `这是 RQ 残差量化前缀节点，表示一组向量空间位置相近、可作为同层候选入口的片段集合。${relationText}`;
    }
    return `这是片段关系图里的${category}节点，检索会根据 RQ 路径、残差范数、dense 语义边和桥接边从它扩展到相邻证据。${relationText}`;
  }
  if (graphType === "mid-concepts") {
    return `这是中粒度概念节点，由片段和 RQ 前缀证据投影支撑，用于从概念层向原文片段下钻。${relationText}`;
  }
  if (graphType === "coarse-concepts") {
    return `这是粗粒度概念节点，聚合多个中粒度概念，用于高层入口选择、主题区域定位和跨主题桥接。${relationText}`;
  }
  return `这是${category}节点。${relationText}`;
}

export function nodeDetailRows(node: GraphNode, graphType: GraphType, relatedEdges: GraphEdge[]): NodeDetailRow[] {
  const rows: NodeDetailRow[] = [];
  appendRow(rows, "节点类型", nodeCategory(node));
  appendRow(rows, "节点 ID", shortId(node.id));
  appendRow(rows, "关联边", relatedEdges.length ? `${formatCount(relatedEdges.length)} 条` : EMPTY_VALUE, relatedEdgeTypes(relatedEdges));

  if (node.document_id) appendRow(rows, "文档 ID", shortId(node.document_id));
  if (node.document_version_id) appendRow(rows, "文档版本", shortId(node.document_version_id));
  appendRow(rows, "页码", formatPage(node));

  if (graphType === "chunk-structure") {
    appendRow(rows, "结构路径", node.snippet ?? node.section_path?.join(" / "));
    appendRow(rows, "结构类别", node.type ?? node.category);
    return rows;
  }

  if (graphType === "chunk-relation") {
    if ((node.category ?? node.type) === "rq_prefix") {
      appendRow(rows, "RQ 前缀键", node.metadata?.rq_prefix_key);
      appendRow(rows, "RQ 层级", node.metadata?.rq_level);
      appendRow(rows, "前缀路径", metadataPath(node, "rq_path_prefix"));
      appendRow(rows, "支撑片段", formatIdList(node.metadata?.support_chunk_ids, "支撑片段"));
      appendRow(rows, "代表片段", formatIdList(node.metadata?.representative_chunk_ids, "代表片段"));
      appendRow(rows, "桥接片段", formatIdList(node.metadata?.bridge_chunk_ids, "桥接片段"));
      appendRow(rows, "残差均值", statsNumber(node, "residual_norm_mean"));
      appendRow(rows, "残差最大值", statsNumber(node, "residual_norm_max"));
      return rows;
    }
    appendRow(rows, "RQ 路径", metadataPath(node, "rq_path"));
    appendRow(rows, "残差范数", formatNumber(metadataNumber(node, "residual_norm")));
    appendRow(rows, "片段分数", formatNumber(node.score ?? node.importance_score));
    appendRow(rows, "支撑片段", formatIdList(supportIdsForNode(node), "支撑片段"));
    return rows;
  }

  if (graphType === "mid-concepts") {
    appendRow(rows, "定义置信度", formatNumber(node.confidence));
    appendRow(rows, "支撑片段", formatIdList(supportIdsForNode(node), "支撑片段"));
    appendRow(rows, "代表片段", formatIdList(representativeIdsForNode(node), "代表片段"));
    appendRow(rows, "重要分数", formatNumber(node.score ?? node.importance_score));
    return rows;
  }

  if (graphType === "coarse-concepts") {
    const includedMidIds = includedMidIdsForNode(node);
    const fallbackMidIds = includedMidIds.length ? includedMidIds : node.support_active_chunk_ids ?? [];
    appendRow(rows, "定义置信度", formatNumber(node.confidence));
    appendRow(rows, "包含中概念", formatIdList(fallbackMidIds, "中概念"));
    appendRow(rows, "支撑片段", formatIdList(node.support_chunk_ids, "支撑片段"));
    appendRow(rows, "重要分数", formatNumber(node.score ?? node.importance_score));
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
  const supportCount = supportIdsForNode(node).length || stringList(node.metadata?.support_chunk_ids).length || node.support_count || 0;
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
  const rqPath = metadataPath(node, (node.category ?? node.type) === "rq_prefix" ? "rq_path_prefix" : "rq_path");
  if (rqPath !== EMPTY_VALUE) {
    rows.push({ label: "RQ 路径", value: rqPath });
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
      const support = formatIdList(node.metadata?.support_chunk_ids, "支撑片段");
      const reps = formatIdList(node.metadata?.representative_chunk_ids, "代表片段");
      return `这个 RQ 前缀由向量残差量化聚合而来。${support !== EMPTY_VALUE ? support : "当前没有返回支撑片段列表"}；${reps !== EMPTY_VALUE ? reps : "当前没有返回代表片段"}。`;
    }
    const rqPath = metadataPath(node, "rq_path");
    const residual = formatNumber(metadataNumber(node, "residual_norm"));
    return `这个片段节点的 RQ 路径是 ${rqPath}，残差范数是 ${residual}。检索会结合 dense 语义边、桥接边和 RQ membership 判断它是否继续扩展。`;
  }
  if (graphType === "mid-concepts") {
    const support = formatIdList(supportIdsForNode(node), "支撑片段");
    const representatives = formatIdList(representativeIdsForNode(node), "代表片段");
    return `这个中粒度概念必须能回到底层 evidence。${support !== EMPTY_VALUE ? support : "当前响应没有返回支撑片段"}；${representatives !== EMPTY_VALUE ? representatives : "当前响应没有返回代表片段"}。`;
  }
  if (graphType === "coarse-concepts") {
    const mids = formatIdList(includedMidIdsForNode(node).length ? includedMidIdsForNode(node) : node.support_active_chunk_ids, "中概念");
    return `这个粗粒度概念负责把多个中粒度概念聚合成高层主题入口。${mids !== EMPTY_VALUE ? mids : "当前响应没有返回包含的中概念列表"}。`;
  }
  return "当前节点没有额外支撑说明。";
}

function RelatedEdgeList({ node, graph, relatedEdges }: { node: GraphNode; graph: GraphResponse; relatedEdges: GraphEdge[] }) {
  const nodeById = useMemo(() => new globalThis.Map(graph.nodes.map((item) => [item.id, item])), [graph.nodes]);
  if (!relatedEdges.length) {
    return <p className="mt-3 text-sm leading-7 text-white/52">当前采样视图没有返回直接相连的边；可以切换图层或扩大后端采样上限查看更多邻接关系。</p>;
  }
  return (
    <div className="mt-3 space-y-2">
      {relatedEdges.slice(0, 6).map((edge, index) => {
        const neighborId = edge.source === node.id ? edge.target : edge.source;
        const neighbor = nodeById.get(neighborId);
        const metric = edge.distance !== null && edge.distance !== undefined ? `距离 ${formatNumber(edge.distance)}` : edge.weight !== null && edge.weight !== undefined ? `权重 ${formatNumber(edge.weight)}` : null;
        return (
          <div key={edge.id ?? `${edge.source}:${edge.target}:${index}`} className="rounded-[16px] border border-white/8 bg-white/[0.03] px-3 py-3 text-sm">
            <div className="flex min-w-0 items-center justify-between gap-3">
              <span className="min-w-0 break-words font-medium text-white/78">{edgeLabel(edge)}</span>
              {metric ? <span className="shrink-0 text-xs text-white/42">{metric}</span> : null}
            </div>
            <p className="mt-1 break-words text-xs leading-5 text-white/50">连接到 {neighbor ? nodeLabel(neighbor) : shortId(neighborId)}</p>
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
  const freshnessReason = graph.stale_reason ?? graph.freshness?.reason;
  const grounding = graph.grounding ?? {};
  const contribution = graph.retrieval_contribution ?? {};
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
            {treeNodes.map((node) => (
              <div key={node.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] px-4 py-4">
                <p className="break-words text-base font-medium text-white">{node.title ?? node.label ?? node.id}</p>
                <div className="mt-3 space-y-2">
                  {(node.children ?? []).map((child) => (
                    <div key={child.id} className="rounded-[16px] border border-white/8 px-4 py-3 text-sm leading-6 text-white/62 break-words">
                      {child.title ?? child.label ?? child.id}
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
          <MetricCard label="支撑状态" value={statusLabel(grounding.status ?? grounding.support_status)} hint={`片段 ${String(grounding.support_chunks ?? grounding.chunk_count ?? EMPTY_VALUE)}`} />
          {selectedGraphType === "chunk-relation" ? (
            <MetricCard
              label="RQ-KMeans"
              value={statNumber(graph.counts, "rq_prefixes")}
              hint={`片段 RQ 边 ${statNumber(graph.counts, "rq_relation_edges")} / 前缀 ${statNumber(graph.counts, "rq_prefixes")} / 归属 ${statNumber(graph.counts, "rq_prefix_memberships")}`}
            />
          ) : (
            <MetricCard label="检索贡献" value={statusLabel(contribution.role ?? "available")} hint={`权重 ${String(contribution.weight ?? contribution.score ?? EMPTY_VALUE)}`} />
          )}
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

      <DetailSection title="关键数据" description="这些字段来自当前图层的节点载荷，只展示能解释节点作用的核心指标。">
        <DetailMetricGrid rows={rows} />
      </DetailSection>

      <DetailSection title="证据与定位">
        <p className="mt-4 break-words text-sm leading-8 text-white/66">{nodeSupportNarrative(node, graphType)}</p>
      </DetailSection>

      <DetailSection title="相邻关系" description="只展示当前采样画布里直接连到该节点的边；完整图谱仍以数据库和图状态为准。">
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
