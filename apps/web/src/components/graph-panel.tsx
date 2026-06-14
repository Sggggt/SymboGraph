"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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

const GRAPH_LAYERS: Array<{ type: GraphType; label: string; icon: typeof Network; description: string }> = [
  { type: "chunk-structure", label: "片段结构图", icon: Map, description: "标题、页面、坐标、表格、公式、图注和前后片段" },
  { type: "chunk-relation", label: "片段关系图", icon: Network, description: "向量、BM25、结构邻接、共检索、细聚类、RQ-KMeans 和桥边" },
  { type: "mid-concepts", label: "中粒度概念图", icon: GitBranch, description: "由片段和细聚类支撑的中粒度概念和关系" },
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

function nodeLabel(node: GraphResponse["nodes"][number]): string {
  return node.name ?? node.label ?? node.id;
}

function nodeCategory(node: GraphResponse["nodes"][number]): string {
  const category = node.category ?? node.type ?? "chunk";
  const labels: Record<string, string> = {
    chunk: "片段",
    fine_cluster: "细聚类",
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

function snippetForNode(node: GraphResponse["nodes"][number] | null): string {
  if (!node) return "";
  return node.snippet ?? node.text ?? (typeof node.metadata?.text === "string" ? node.metadata.text : "");
}

function metadataNumber(node: GraphResponse["nodes"][number], key: string): number | null {
  const value = node.metadata?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metadataPath(node: GraphResponse["nodes"][number], key: string): string {
  const value = node.metadata?.[key];
  return Array.isArray(value) && value.length ? value.join("/") : EMPTY_VALUE;
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
              value={statNumber(graph.counts, "rq_clusters")}
              hint={`片段边 ${statNumber(graph.counts, "rq_edges")} / 聚类边 ${statNumber(graph.counts, "rq_cluster_edges")} / 成员 ${statNumber(graph.counts, "rq_memberships")}`}
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
              <GraphNodeSummary node={selectedGraphNode} graphType={selectedGraphType} />
            </aside>
          )}
        </div>
      </motion.section>

      {!isFullscreen ? (
        <motion.section initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} style={sidePanelStyle} className="glass-panel kg-scroll-panel h-full min-w-0 rounded-[28px] p-5">
          <GraphNodeSummary node={selectedGraphNode} graphType={selectedGraphType} />
        </motion.section>
      ) : null}
    </div>
  );
}

export function GraphPanel() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  return <GraphPanelContent key={selectedKnowledgeBaseId ?? "unassigned"} selectedKnowledgeBaseId={selectedKnowledgeBaseId} />;
}

function GraphNodeSummary({ node, graphType }: { node: GraphResponse["nodes"][number] | null; graphType: GraphType }) {
  if (!node) {
    return (
      <div className="mt-6 rounded-[24px] border border-white/8 bg-white/[0.03] p-5 text-sm leading-7 text-white/58">
        选择一个节点查看当前图层的支撑信息、结构路径和支持片段。
      </div>
    );
  }
  const rows = [
    ["类型", nodeCategory(node)],
    ["分数", node.score ?? node.importance_score ?? EMPTY_VALUE],
    ["支持数", node.support_count ?? node.support_chunk_ids?.length ?? EMPTY_VALUE],
    ["页码", node.page_number ?? (Array.isArray(node.page_range) ? node.page_range.join("-") : EMPTY_VALUE)],
    ["文档版本", node.document_version_id ?? EMPTY_VALUE],
    ["结构路径", node.section_path?.join(" / ") || EMPTY_VALUE],
    ["RQ 路径", metadataPath(node, "rq_path")],
    ["残差范数", metadataNumber(node, "residual_norm") ?? EMPTY_VALUE],
  ];
  const text = snippetForNode(node);
  return (
    <div className="mt-6 space-y-4">
      <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
        <p className="break-words text-xs uppercase tracking-[0.26em] text-white/45">{graphTypeLabel(graphType)}</p>
        <p className="mt-3 break-words text-2xl font-semibold text-white">{nodeLabel(node)}</p>
        <div className="mt-4 grid grid-cols-1 gap-2 text-sm text-white/62">
          {rows.map(([label, value]) => (
            <div key={label} className="flex min-w-0 items-center justify-between gap-3">
              <span className="shrink-0 text-white/42">{label}</span>
              <span className="min-w-0 break-words text-right">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
        <p className="text-xs uppercase tracking-[0.26em] text-white/45">支撑文本</p>
        {text ? (
          <MarkdownRenderer content={text} compact className="mt-4 break-words text-white/68" />
        ) : (
          <p className="mt-4 break-words text-sm leading-7 text-white/52">当前节点没有返回文本片段；请查看元数据，或切换到片段结构图、片段关系图。</p>
        )}
      </div>
      {node.metadata && Object.keys(node.metadata).length > 0 ? (
        <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
          <p className="text-xs uppercase tracking-[0.26em] text-white/45">元数据</p>
          <pre className="mt-4 max-h-72 overflow-auto whitespace-pre-wrap text-xs leading-5 text-white/56">{JSON.stringify(node.metadata, null, 2)}</pre>
        </div>
      ) : null}
    </div>
  );
}
