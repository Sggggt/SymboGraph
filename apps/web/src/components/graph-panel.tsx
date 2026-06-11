"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { GraphResponse } from "@course-kg/shared";
import { motion } from "framer-motion";
import { Boxes, ChevronDown, Expand, Lock, Minimize2, RefreshCw, Unlock } from "lucide-react";

import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { fetchPartitionGraph, fetchDashboard, fetchGraph } from "@/lib/api";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { NetworkCanvas, type NetworkCanvasHandle } from "@/components/network-canvas";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { useLocalStorage } from "@/hooks/use-local-storage";


type SelectedNode = { id: string; category: string } | null;

type EvidenceSnippet = {
  id: string;
  title: string;
  text: string;
  kind: string;
  page?: number | null;
  documentVersionId?: string | null;
};

function stripGraphPrefix(id: string) {
  const separatorIndex = id.indexOf(":");
  return separatorIndex >= 0 ? id.slice(separatorIndex + 1) : id;
}

function idSet(values?: string[]) {
  return new Set((values ?? []).map((value) => stripGraphPrefix(String(value))));
}

function communityNumberFromGraphId(id: string) {
  const tail = id.split(":").at(-1);
  const parsed = Number(tail);
  return Number.isFinite(parsed) ? parsed : null;
}

function dominantSupportCommunity(data: GraphResponse, node: GraphResponse["nodes"][number]) {
  const supportIds = new Set([...idSet(node.support_atom_ids), ...idSet(node.support_active_chunk_ids)]);
  if (!supportIds.size) {
    return null;
  }
  const communityCounts = new Map<number, number>();
  for (const edge of data.edges) {
    const sourceId = stripGraphPrefix(String(edge.source));
    const targetId = stripGraphPrefix(String(edge.target));
    const source = String(edge.source);
    const target = String(edge.target);
    if (supportIds.has(sourceId) && target.startsWith("community:")) {
      const community = communityNumberFromGraphId(target);
      if (community !== null) {
        communityCounts.set(community, (communityCounts.get(community) ?? 0) + 1);
      }
    }
    if (supportIds.has(targetId) && source.startsWith("community:")) {
      const community = communityNumberFromGraphId(source);
      if (community !== null) {
        communityCounts.set(community, (communityCounts.get(community) ?? 0) + 1);
      }
    }
  }
  for (const supportNode of data.nodes) {
    if (typeof supportNode.community_louvain !== "number") {
      continue;
    }
    if (!supportIds.has(stripGraphPrefix(supportNode.id))) {
      continue;
    }
    communityCounts.set(supportNode.community_louvain, (communityCounts.get(supportNode.community_louvain) ?? 0) + 1);
  }
  let bestCommunity: number | null = null;
  let bestCount = 0;
  for (const [community, count] of communityCounts.entries()) {
    if (count > bestCount) {
      bestCommunity = community;
      bestCount = count;
    }
  }
  return bestCommunity;
}

function assignMissingSignalCommunities(nodes: GraphResponse["nodes"], edges: GraphResponse["edges"]) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const adjacency = new Map<string, Set<string>>();
  for (const node of nodes) {
    adjacency.set(node.id, new Set());
  }
  for (const edge of edges) {
    const source = String(edge.source);
    const target = String(edge.target);
    if (!byId.has(source) || !byId.has(target)) {
      continue;
    }
    adjacency.get(source)?.add(target);
    adjacency.get(target)?.add(source);
  }

  const visited = new Set<string>();
  let fallbackCommunity = 1000;
  for (const node of nodes) {
    if (visited.has(node.id)) {
      continue;
    }
    const component: GraphResponse["nodes"] = [];
    const queue = [node.id];
    visited.add(node.id);
    for (let index = 0; index < queue.length; index += 1) {
      const current = byId.get(queue[index]);
      if (current) {
        component.push(current);
      }
      for (const next of adjacency.get(queue[index]) ?? []) {
        if (!visited.has(next)) {
          visited.add(next);
          queue.push(next);
        }
      }
    }

    const communityCounts = new Map<number, number>();
    for (const item of component) {
      if (typeof item.community_louvain === "number") {
        communityCounts.set(item.community_louvain, (communityCounts.get(item.community_louvain) ?? 0) + 1);
      }
    }
    let componentCommunity: number | null = null;
    let bestCount = 0;
    for (const [community, count] of communityCounts.entries()) {
      if (count > bestCount) {
        componentCommunity = community;
        bestCount = count;
      }
    }
    if (componentCommunity === null) {
      componentCommunity = fallbackCommunity;
      fallbackCommunity += 1;
    }
    for (const item of component) {
      if (typeof item.community_louvain !== "number") {
        item.community_louvain = componentCommunity;
      }
    }
  }
  return nodes;
}

function signalOnlyGraph(data: GraphResponse, signalLayerComplete: boolean): GraphResponse {
  if (!signalLayerComplete) {
    return {
      ...data,
      nodes: [],
      edges: [],
      node_counts: { signal_node: 0 },
      edge_counts: { signal_projection: 0 },
    };
  }
  const nodesWithSupportCommunity = data.nodes
    .filter((node) => node.category === "signal_node")
    .map((node) => ({
      ...node,
      community_louvain: typeof node.community_louvain === "number" ? node.community_louvain : dominantSupportCommunity(data, node),
    }));
  const keptNodeIds = new Set(nodesWithSupportCommunity.map((node) => node.id));
  const edges = data.edges.filter((edge) => keptNodeIds.has(String(edge.source)) && keptNodeIds.has(String(edge.target)));
  const nodes = assignMissingSignalCommunities(nodesWithSupportCommunity, edges);
  return {
    ...data,
    nodes,
    edges,
    node_counts: { signal_node: nodes.length },
    edge_counts: { signal_projection: edges.length },
  };
}

function evidenceSnippetsForNode(data: GraphResponse | undefined, node: GraphResponse["nodes"][number] | null): EvidenceSnippet[] {
  if (!data || !node) {
    return [];
  }
  const supportAtomIds = idSet(node.support_atom_ids);
  const supportChunkIds = idSet(node.support_active_chunk_ids);
  const snippets: EvidenceSnippet[] = [];
  const seen = new Set<string>();

  const ownText = (node.snippet ?? "").trim();
  if (ownText) {
    seen.add(ownText);
    snippets.push({
      id: node.id,
      title: node.name,
      text: ownText,
      kind: node.entity_type ?? node.category,
      page: node.page_number,
      documentVersionId: node.document_version_id,
    });
  }

  for (const supportNode of data.nodes) {
    const normalizedId = stripGraphPrefix(supportNode.id);
    const isSupportAtom = supportNode.category === "evidence_atom" && supportAtomIds.has(normalizedId);
    const isSupportChunk = supportNode.category === "active_chunk" && supportChunkIds.has(normalizedId);
    if (!isSupportAtom && !isSupportChunk) {
      continue;
    }
    const text = (supportNode.snippet ?? "").trim();
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    snippets.push({
      id: supportNode.id,
      title: supportNode.name,
      text,
      kind: supportNode.entity_type ?? supportNode.category,
      page: supportNode.page_number,
      documentVersionId: supportNode.document_version_id,
    });
  }

  return snippets.slice(0, 12);
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

function GraphPanelContent({ selectedKnowledgeBaseId }: { selectedKnowledgeBaseId: string | null }) {
  const storageScope = selectedKnowledgeBaseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedKnowledgeBaseId],
    queryFn: () => fetchDashboard(selectedKnowledgeBaseId),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const [selectedPartition, setSelectedPartition] = useLocalStorage(`graph.selectedPartition.${storageScope}`, "");
  const [selectedNode, setSelectedNode] = useLocalStorage<SelectedNode>(`graph.selectedNode.${storageScope}`, null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [partitionMenuOpen, setPartitionMenuOpen] = useState(false);
  const graphView: GraphResponse["view"] = "overview";
  const canvasRef = useRef<NetworkCanvasHandle | null>(null);
  const fullscreenRef = useRef<HTMLDivElement | null>(null);

  const graphQuery = useQuery({
    queryKey: ["projection-graph", selectedKnowledgeBaseId, selectedPartition, graphView],
    queryFn: () => (selectedPartition ? fetchPartitionGraph(selectedPartition, selectedKnowledgeBaseId, "evidence", graphView) : fetchGraph(selectedKnowledgeBaseId, "evidence", graphView)),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const signalLayerComplete = Boolean(graphQuery.data?.signal_layer_complete);
  const partitionOptions = useMemo(() => dashboardQuery.data?.tree.map((node) => node.title) ?? [], [dashboardQuery.data]);
  const visibleGraph = useMemo<GraphResponse | null>(() => {
    if (!graphQuery.data) {
      return null;
    }
    return signalOnlyGraph(graphQuery.data, signalLayerComplete);
  }, [signalLayerComplete, graphQuery.data]);
  const selectedGraphNode = useMemo(
    () => (selectedNode && visibleGraph ? visibleGraph.nodes.find((node) => node.id === selectedNode.id) ?? null : null),
    [selectedNode, visibleGraph],
  );
  const selectedSnippets = useMemo(() => evidenceSnippetsForNode(graphQuery.data, selectedGraphNode), [graphQuery.data, selectedGraphNode]);

  useEffect(() => {
    if (selectedNode && visibleGraph && !visibleGraph.nodes.some((node) => node.id === selectedNode.id)) {
      setSelectedNode(null);
    }
  }, [selectedNode, setSelectedNode, visibleGraph]);

  useEffect(() => {
    const handleChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  const handlePartitionChange = (partition: string) => {
    setSelectedPartition(partition);
    setSelectedNode(null);
    setIsLocked(false);
    setPartitionMenuOpen(false);
  };

  const openDetail = (nodeId: string, category: string) => {
    setSelectedNode({ id: nodeId, category });
  };

  const toggleFullscreen = async () => {
    if (!fullscreenRef.current) {
      return;
    }
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
  if (!graphQuery.data || !dashboardQuery.data || !visibleGraph) {
    return null;
  }

  return (
    <div
      ref={fullscreenRef}
      className={`relative grid gap-4 ${isFullscreen ? "min-h-screen bg-[rgba(3,8,24,0.98)] p-4" : "kg-page xl:grid-cols-[260px_minmax(0,1fr)_360px]"}`}
    >
      {!isFullscreen ? (
        <motion.section initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass-panel kg-scroll-panel rounded-[28px] p-5">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="section-kicker">目录树</p>
              <h2 className="mt-2 break-words text-2xl font-semibold text-white">目录与文档</h2>
            </div>
            <Boxes className="size-5 shrink-0 text-cyan-200" />
          </div>

          <div className="relative mt-5">
            <button
              type="button"
              onClick={() => setPartitionMenuOpen((open) => !open)}
              className="flex h-11 w-full items-center justify-between gap-3 rounded-full border border-white/10 bg-white/[0.05] px-4 text-left text-sm text-white outline-none transition hover:border-cyan-200/24"
            >
              <span className="min-w-0 truncate">{selectedPartition || "全部目录"}</span>
              <ChevronDown className={`size-4 shrink-0 text-cyan-100/60 transition ${partitionMenuOpen ? "rotate-180" : ""}`} />
            </button>
            {partitionMenuOpen ? (
              <div className="custom-scrollbar absolute left-0 right-0 top-[calc(100%+0.5rem)] z-[80] max-h-72 overflow-y-auto rounded-[1.25rem] border border-white/10 bg-[rgba(4,10,24,0.96)] p-2 shadow-[0_24px_70px_rgba(0,0,0,0.42)] backdrop-blur-2xl">
                <button
                  type="button"
                  onClick={() => handlePartitionChange("")}
                  className="w-full rounded-2xl px-3 py-2.5 text-left text-sm text-white/70 transition hover:bg-cyan-300/[0.08] hover:text-white"
                >
                  全部目录
                </button>
                {partitionOptions.map((partition) => (
                  <button
                    key={partition}
                    type="button"
                    onClick={() => handlePartitionChange(partition)}
                    className="w-full rounded-2xl px-3 py-2.5 text-left text-sm text-white/70 transition hover:bg-cyan-300/[0.08] hover:text-white"
                  >
                    {partition}
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-6 space-y-4">
            {dashboardQuery.data.tree
              .filter((partition) => !selectedPartition || partition.title === selectedPartition)
              .map((partition) => (
                <div key={partition.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] px-4 py-4">
                  <p className="break-words text-base font-medium text-white">{partition.title}</p>
                  <div className="mt-3 space-y-2">
                    {(partition.children ?? []).map((document) => (
                      <div key={document.id} className="rounded-[16px] border border-white/8 px-4 py-3 text-sm leading-6 text-white/62 break-words">
                        {document.title}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        </motion.section>
      ) : null}

      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className={`glass-panel min-w-0 rounded-[30px] ${isFullscreen ? "col-span-full flex min-h-[calc(100vh-2rem)] flex-col p-3" : "flex min-h-0 flex-col p-4 lg:p-5"}`}
      >
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4 px-2">
          <div className="min-w-0">
            <p className="section-kicker">图谱画布</p>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <h2 className="break-words text-3xl font-semibold text-white">{selectedPartition || "全库投影派生层"}</h2>
              <GraphStaleBadge isStale={graphQuery.data.freshness?.is_stale} />
            </div>
            <p className="mt-2 max-w-3xl break-words text-sm leading-7 text-white/50">
              {signalLayerComplete
                ? "画布只渲染完整发布后的投影派生层。证据图不在画布展开，仅用于右侧支撑片段、引用和诊断。"
                : "投影派生层尚未 active；前端不会渲染 evidence atom、active chunk 或其他半成品节点。"}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full border px-3 py-1.5 text-xs ${
                signalLayerComplete
                  ? "border-fuchsia-200/30 bg-fuchsia-300/[0.10] text-fuchsia-50"
                  : "border-white/8 text-white/30"
              }`}
              title={signalLayerComplete ? "Signal layer is active" : `Signal layer ${graphQuery.data.signal_layer_status ?? "pending"}`}
            >
              Signal {signalLayerComplete ? "active" : graphQuery.data.signal_layer_status ?? "pending"}
            </span>
            <span className="rounded-full border border-white/8 px-3 py-1.5 text-xs text-white/45">
              {graphQuery.data.view ?? "overview"}
            </span>
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

        <div className={`grid min-h-0 flex-1 gap-4 ${isFullscreen ? "grid-cols-[minmax(0,1fr)_380px]" : "grid-cols-1"}`}>
          <div className="min-w-0 rounded-[24px] border border-white/8 bg-[rgba(4,9,24,0.36)] p-2">
            <NetworkCanvas
              key={`projection:${selectedPartition || "all"}`}
              ref={canvasRef}
              graph={visibleGraph}
              height={isFullscreen ? 900 : 760}
              selectedNodeId={selectedNode?.id ?? null}
              onNodeClick={(nodeId, category) => setSelectedNode({ id: nodeId, category })}
              onNodeDoubleClick={(nodeId, category) => openDetail(nodeId, category)}
            />
          </div>

          {isFullscreen && (
            <aside className={`glass-panel min-w-0 ${isFullscreen ? "kg-scroll-panel rounded-[24px]" : "hidden"} p-5`}>
              <GraphNodeSummary node={selectedGraphNode} snippets={selectedSnippets} />
            </aside>
          )}
        </div>
      </motion.section>

      {!isFullscreen ? (
        <motion.section initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} className="glass-panel kg-scroll-panel min-w-0 rounded-[28px] p-5">
          <GraphNodeSummary node={selectedGraphNode} snippets={selectedSnippets} />
        </motion.section>
      ) : null}
    </div>
  );
}

export function GraphPanel() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  return <GraphPanelContent key={selectedKnowledgeBaseId ?? "unassigned"} selectedKnowledgeBaseId={selectedKnowledgeBaseId} />;
}

function GraphNodeSummary({ node, snippets }: { node: GraphResponse["nodes"][number] | null; snippets: EvidenceSnippet[] }) {
  if (!node) {
    return (
      <div className="mt-6 rounded-[24px] border border-white/8 bg-white/[0.03] p-5 text-sm leading-7 text-white/58">
        选择一个节点查看当前图层中的基础信息。
      </div>
    );
  }
  const rows = [
    ["类型", node.entity_type ?? node.category],
    ["目录", node.partition ?? "n/a"],
    ["证据", node.evidence_count ?? node.support_count ?? "n/a"],
    ["Support atoms", node.support_atom_ids?.length ?? "n/a"],
    ["页码", node.page_number ?? "n/a"],
    ["文档版本", node.document_version_id ?? "n/a"],
  ];
  return (
    <div className="mt-6 space-y-4">
      <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
        <p className="break-words text-xs uppercase tracking-[0.26em] text-white/45">projection derived layer</p>
        <p className="mt-3 break-words text-2xl font-semibold text-white">{node.name}</p>
        <div className="mt-4 grid grid-cols-1 gap-2 text-sm text-white/62">
          {rows.map(([label, value]) => (
            <div key={label} className="flex min-w-0 items-center justify-between gap-3">
              <span className="shrink-0 text-white/42">{label}</span>
              <span className="min-w-0 break-words text-right">{String(value)}</span>
            </div>
          ))}
        </div>
        {node.snippet ? <p className="mt-4 break-words text-sm leading-7 text-white/68">{node.snippet}</p> : null}
      </div>
      <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs uppercase tracking-[0.26em] text-white/45">Related snippets</p>
          <span className="rounded-full border border-white/8 px-2.5 py-1 text-[11px] text-white/45">{snippets.length}</span>
        </div>
        {snippets.length ? (
          <div className="mt-4 space-y-3">
            {snippets.map((snippet) => (
              <article key={snippet.id} className="rounded-[18px] border border-white/8 bg-black/10 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/38">
                  <span>{snippet.kind}</span>
                  {snippet.page ? <span>page {snippet.page}</span> : null}
                  {snippet.documentVersionId ? <span className="normal-case tracking-normal">version {snippet.documentVersionId}</span> : null}
                </div>
                <p className="mb-2 break-words text-sm font-medium text-white/82">{snippet.title}</p>
                <MarkdownRenderer content={snippet.text} compact className="break-words text-white/68" />
              </article>
            ))}
          </div>
        ) : (
          <p className="mt-4 break-words text-sm leading-7 text-white/52">
            No support snippets were returned in the current overview payload. Support atoms: {node.support_atom_ids?.length ?? 0}; active chunks: {node.support_active_chunk_ids?.length ?? 0}.
          </p>
        )}
      </div>
    </div>
  );
}
