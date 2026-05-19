"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { GraphNodeDetail, GraphResponse, GraphType, SemanticEntityType } from "@course-kg/shared";
import { motion } from "framer-motion";
import { Boxes, ChevronDown, Expand, Lock, Minimize2, Network, RefreshCw, ScanSearch, Unlock } from "lucide-react";

import { useCourseContext } from "@/components/course-context";
import { createBatchLogToken, fetchChapterGraph, fetchDashboard, fetchGraph, fetchGraphNode, getBatchLogUrl, rebuildGraph } from "@/lib/api";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { NetworkCanvas, type NetworkCanvasHandle } from "@/components/network-canvas";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { useLocalStorage } from "@/hooks/use-local-storage";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

type SelectedNode = { id: string; category: string } | null;
const graphTabs: Array<{ type: GraphType; label: string }> = [
  { type: "semantic", label: "语义KG" },
  { type: "structural", label: "结构图" },
  { type: "evidence", label: "证据图" },
];
const entityTypes: SemanticEntityType[] = ["concept", "method", "formula", "metric", "algorithm", "definition", "theorem", "problem_type"];
type RebuildPhase = "extraction" | "community_summary" | "completed";
type RebuildLogEvent = {
  event: string;
  completed_graph_chunks?: number;
  total_graph_chunks?: number;
  community_summary_count?: number;
  total_communities?: number;
  message?: string;
  error?: string | null;
};

const graphLogStreamMaxRetries = 3;
const graphLogStreamRetryDelayMs = 1200;

function GraphPanelContent({ selectedCourseId }: { selectedCourseId: string | null }) {
  const storageScope = selectedCourseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedCourseId],
    queryFn: () => fetchDashboard(selectedCourseId),
    enabled: Boolean(selectedCourseId),
  });
  const [selectedChapter, setSelectedChapter] = useLocalStorage(`graph.selectedChapter.${storageScope}`, "");
  const [selectedNode, setSelectedNode] = useLocalStorage<SelectedNode>(`graph.selectedNode.${storageScope}`, null);
  const [detailNodeId, setDetailNodeId] = useLocalStorage<string | null>(`graph.detailNodeId.${storageScope}`, null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [chapterMenuOpen, setChapterMenuOpen] = useState(false);
  const [rebuildDialogOpen, setRebuildDialogOpen] = useState(false);
  const [rebuildMode, setRebuildMode] = useState<"incremental" | "full">("incremental");
  const [rebuildPhase, setRebuildPhase] = useState<RebuildPhase>("extraction");
  const [chunkProgress, setChunkProgress] = useState(0);
  const [communityProgress, setCommunityProgress] = useState(0);
  const [rebuildBatchId, setRebuildBatchId] = useState<string | null>(null);
  const [rebuildFailureDialog, setRebuildFailureDialog] = useState<{ title: string; message: string; details?: string | null } | null>(null);
  const [dryRunResult, setDryRunResult] = useState<{ affectedDocuments: number; semanticEntities: number; semanticRelations: number } | null>(null);
  const [graphLogRetryCount, setGraphLogRetryCount] = useState(0);
  const [selectedCommunity, setSelectedCommunity] = useState<number | null>(null);
  const [graphType, setGraphType] = useLocalStorage<GraphType>(`graph.type.${storageScope}`, "semantic");
  const [selectedEntityType, setSelectedEntityType] = useLocalStorage<SemanticEntityType | "all">(`graph.entityType.${storageScope}`, "all");
  const canvasRef = useRef<NetworkCanvasHandle | null>(null);
  const fullscreenRef = useRef<HTMLDivElement | null>(null);
  const isRebuildingRef = useRef(false);

  const graphQuery = useQuery({
    queryKey: ["graph", selectedCourseId, graphType, selectedChapter],
    queryFn: () => (selectedChapter && graphType !== "structural" ? fetchChapterGraph(selectedChapter, selectedCourseId, graphType) : fetchGraph(selectedCourseId, graphType)),
    enabled: Boolean(selectedCourseId),
  });
  const refetchGraph = graphQuery.refetch;
  const rebuildMutation = useMutation({
    mutationFn: () => rebuildGraph(selectedCourseId, rebuildMode, false),
    onMutate: () => {
      setRebuildPhase("extraction");
      setChunkProgress(0);
      setCommunityProgress(0);
      setGraphLogRetryCount(0);
      setRebuildFailureDialog(null);
      setDryRunResult(null);
    },
    onSuccess: (data) => {
      if (!data.batch_id) {
        return;
      }
      isRebuildingRef.current = true;
      setRebuildBatchId(data.batch_id);
    },
    onError: (error) => {
      setRebuildPhase("extraction");
      setChunkProgress(0);
      setCommunityProgress(0);
      setRebuildBatchId(null);
      setRebuildDialogOpen(false);
      setRebuildFailureDialog({
        title: "图谱重建启动失败",
        message: error instanceof Error ? error.message : "图谱重建任务启动失败，后端未返回错误详情。",
      });
    },
  });
  const dryRunMutation = useMutation({
    mutationFn: () => rebuildGraph(selectedCourseId, rebuildMode, true),
    onSuccess: (data) => {
      setDryRunResult({
        affectedDocuments: data.affected_documents,
        semanticEntities: data.semantic_entities ?? 0,
        semanticRelations: data.semantic_relations ?? 0,
      });
    },
  });
  const detailQuery = useQuery({
    queryKey: ["graph-node", selectedCourseId, detailNodeId],
    queryFn: () => fetchGraphNode(detailNodeId as string, selectedCourseId),
    enabled: Boolean(selectedCourseId && detailNodeId),
  });

  const chapterOptions = useMemo(() => dashboardQuery.data?.tree.map((node) => node.title) ?? [], [dashboardQuery.data]);
  const visibleGraph = useMemo<GraphResponse | null>(() => {
    if (!graphQuery.data) {
      return null;
    }
    const data = graphQuery.data;
    const keptNodeIds = new Set(
      data.nodes
        .filter((node) => graphType !== "semantic" || selectedCommunity === null || node.category !== "semantic_entity" || node.community_louvain === selectedCommunity)
        .filter((node) => graphType !== "semantic" || selectedEntityType === "all" || node.category !== "semantic_entity" || node.entity_type === selectedEntityType)
        .map((node) => node.id),
    );
    return {
      ...data,
      nodes: data.nodes.filter((node) => keptNodeIds.has(node.id)),
      edges: data.edges.filter((edge) => keptNodeIds.has(String(edge.source)) && keptNodeIds.has(String(edge.target))),
    };
  }, [graphQuery.data, graphType, selectedCommunity, selectedEntityType]);
  const selectedGraphNode = useMemo(
    () => (selectedNode && visibleGraph ? visibleGraph.nodes.find((node) => node.id === selectedNode.id) ?? null : null),
    [selectedNode, visibleGraph],
  );

  useEffect(() => {
    const handleChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  useEffect(() => {
    if (!rebuildBatchId) {
      isRebuildingRef.current = false;
      queueMicrotask(() => setGraphLogRetryCount(0));
      return;
    }
    isRebuildingRef.current = true;
    const streamBatchId = rebuildBatchId;
    let closed = false;
    let retryCount = 0;
    let source: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const closeSource = () => {
      source?.close();
      source = null;
    };

    const connect = async () => {
      if (closed) {
        return;
      }
      closeSource();
      let token: string;
      try {
        token = (await createBatchLogToken(streamBatchId)).token;
      } catch (error) {
        if (closed) {
          return;
        }
        retryCount += 1;
        setGraphLogRetryCount(retryCount);
        if (retryCount <= graphLogStreamMaxRetries) {
          retryTimer = setTimeout(() => {
            void connect();
          }, graphLogStreamRetryDelayMs * retryCount);
          return;
        }
        setRebuildPhase("extraction");
        setChunkProgress(0);
        setCommunityProgress(0);
        setRebuildDialogOpen(false);
        setRebuildBatchId(null);
        setRebuildFailureDialog({
          title: "图谱重建日志流授权失败",
          message: "日志流 token 创建失败，无法继续同步重建进度。请检查 API 鉴权或后端日志。",
          details: error instanceof Error ? error.message : String(error),
        });
        return;
      }
      source = new EventSource(getBatchLogUrl(streamBatchId, token));
      source.onmessage = (event) => {
        if (!isRebuildingRef.current || closed) {
          return;
        }
        retryCount = 0;
        setGraphLogRetryCount(0);
        const item = JSON.parse(event.data) as RebuildLogEvent;
        if (item.event === "batch_graph_progress") {
          const completed = item.completed_graph_chunks ?? 0;
          const total = item.total_graph_chunks ?? 1;
          setChunkProgress(Math.round((completed / total) * 100));
          setRebuildPhase("extraction");
        } else if (item.event === "batch_graph_community_summary") {
          const completed = item.community_summary_count ?? 0;
          const total = item.total_communities ?? 1;
          setCommunityProgress(Math.round((completed / total) * 100));
          setRebuildPhase("community_summary");
        } else if (item.event === "batch_graph_started") {
          setChunkProgress(0);
          setCommunityProgress(0);
          setRebuildPhase("extraction");
        } else if (item.event === "graph_rebuilt" || item.event === "batch_completed") {
          setChunkProgress(100);
          setCommunityProgress(100);
          setRebuildPhase("completed");
          setGraphLogRetryCount(0);
          refetchGraph();
          setRebuildDialogOpen(false);
          setRebuildBatchId(null);
          closeSource();
        } else if (item.event === "graph_failed" || item.event === "batch_failed") {
          setRebuildPhase("extraction");
          setChunkProgress(0);
          setCommunityProgress(0);
          setGraphLogRetryCount(0);
          setRebuildDialogOpen(false);
          setRebuildBatchId(null);
          setRebuildFailureDialog({
            title: "图谱重建失败",
            message: item.message || "图谱重建失败，后端未返回错误详情。",
            details: item.error ?? null,
          });
          closeSource();
        }
      };
      source.onerror = () => {
        closeSource();
        if (closed) {
          return;
        }
        retryCount += 1;
        setGraphLogRetryCount(retryCount);
        if (retryCount <= graphLogStreamMaxRetries) {
          retryTimer = setTimeout(() => {
            void connect();
          }, graphLogStreamRetryDelayMs * retryCount);
          return;
        }
        setRebuildPhase("extraction");
        setChunkProgress(0);
        setCommunityProgress(0);
        setRebuildDialogOpen(false);
        setRebuildBatchId(null);
        setRebuildFailureDialog({
          title: "图谱重建日志流中断",
          message: "日志流重连失败，无法继续同步重建进度。任务可能仍在后端运行，请稍后刷新图谱或重新触发重建。",
        });
      };
    };
    void connect();
    return () => {
      closed = true;
      isRebuildingRef.current = false;
      if (retryTimer) {
        clearTimeout(retryTimer);
      }
      closeSource();
    };
  }, [rebuildBatchId, refetchGraph]);

  const handleChapterChange = (chapter: string) => {
    setSelectedChapter(chapter);
    setSelectedNode(null);
    setDetailNodeId(null);
    setIsLocked(false);
    setSelectedCommunity(null);
    setChapterMenuOpen(false);
  };

  const openDetail = (nodeId: string, category: string) => {
    setSelectedNode({ id: nodeId, category });
    if (graphType === "semantic" && category === "semantic_entity") {
      setDetailNodeId(nodeId);
      return;
    }
    setDetailNodeId(null);
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
              onClick={() => setChapterMenuOpen((open) => !open)}
              className="flex h-11 w-full items-center justify-between gap-3 rounded-full border border-white/10 bg-white/[0.05] px-4 text-left text-sm text-white outline-none transition hover:border-cyan-200/24"
            >
              <span className="min-w-0 truncate">{selectedChapter || "全部目录"}</span>
              <ChevronDown className={`size-4 shrink-0 text-cyan-100/60 transition ${chapterMenuOpen ? "rotate-180" : ""}`} />
            </button>
            {chapterMenuOpen ? (
              <div className="custom-scrollbar absolute left-0 right-0 top-[calc(100%+0.5rem)] z-[80] max-h-72 overflow-y-auto rounded-[1.25rem] border border-white/10 bg-[rgba(4,10,24,0.96)] p-2 shadow-[0_24px_70px_rgba(0,0,0,0.42)] backdrop-blur-2xl">
                <button
                  type="button"
                  onClick={() => handleChapterChange("")}
                  className="w-full rounded-2xl px-3 py-2.5 text-left text-sm text-white/70 transition hover:bg-cyan-300/[0.08] hover:text-white"
                >
                  全部目录
                </button>
                {chapterOptions.map((chapter) => (
                  <button
                    key={chapter}
                    type="button"
                    onClick={() => handleChapterChange(chapter)}
                    className="w-full rounded-2xl px-3 py-2.5 text-left text-sm text-white/70 transition hover:bg-cyan-300/[0.08] hover:text-white"
                  >
                    {chapter}
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-6 space-y-4">
            {dashboardQuery.data.tree
              .filter((chapter) => !selectedChapter || chapter.title === selectedChapter)
              .map((chapter) => (
                <div key={chapter.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] px-4 py-4">
                  <p className="break-words text-base font-medium text-white">{chapter.title}</p>
                  <div className="mt-3 space-y-2">
                    {(chapter.children ?? []).map((document) => (
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
            <h2 className="mt-2 break-words text-3xl font-semibold text-white">{selectedChapter && graphType !== "structural" ? selectedChapter : "全库图谱"}</h2>
            <p className="mt-2 max-w-3xl break-words text-sm leading-7 text-white/50">
              {graphType === "semantic" ? "单击节点只做高亮，双击语义实体打开知识详解。支持拖拽平移与滚轮缩放。" : graphType === "evidence" ? "语义实体、证据片段与文档版本分层展示，单击节点查看基础信息。" : "本地资料结构单独展示，不混入语义实体。"}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex rounded-full border border-white/10 bg-white/[0.035] p-1">
              {graphTabs.map((tab) => (
                <button
                  key={tab.type}
                  type="button"
                  onClick={() => {
                    setSelectedNode(null);
                    setDetailNodeId(null);
                    setSelectedCommunity(null);
                    if (tab.type !== "semantic") {
                      setSelectedEntityType("all");
                    }
                    setGraphType(tab.type);
                  }}
                  className={`rounded-full px-3 py-1.5 text-xs transition ${graphType === tab.type ? "bg-cyan-300/12 text-cyan-50" : "text-white/58 hover:text-white"}`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <motion.button
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.98 }}
              type="button"
              disabled={rebuildMutation.isPending || Boolean(rebuildBatchId)}
              className="action-chip rounded-full px-4 py-2 text-xs uppercase tracking-[0.22em]"
              onClick={() => {
                isRebuildingRef.current = false;
                setRebuildPhase("extraction");
                setChunkProgress(0);
                setCommunityProgress(0);
                setRebuildBatchId(null);
                setGraphLogRetryCount(0);
                setRebuildFailureDialog(null);
                setRebuildDialogOpen(true);
              }}
            >
              {rebuildMutation.isPending || Boolean(rebuildBatchId) ? (
                <RefreshCw className="mr-2 inline size-4 animate-spin" />
              ) : (
                <Network className="mr-2 inline size-4" />
              )}
              {rebuildMutation.isPending || Boolean(rebuildBatchId) ? "重建中..." : "重建图谱"}
            </motion.button>
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

        {graphType === "semantic" ? (
          <div className="mb-4 flex flex-wrap items-center gap-2 px-2">
            <button
              type="button"
              onClick={() => setSelectedEntityType("all")}
              className={`rounded-full border px-3 py-1.5 text-xs transition ${selectedEntityType === "all" ? "border-cyan-200/30 bg-cyan-300/[0.08] text-cyan-50" : "border-white/10 text-white/58 hover:text-white"}`}
            >
              全部实体
            </button>
            {entityTypes.map((entityType) => (
              <button
                key={entityType}
                type="button"
                onClick={() => setSelectedEntityType(entityType)}
                className={`rounded-full border px-3 py-1.5 text-xs transition ${selectedEntityType === entityType ? "border-cyan-200/30 bg-cyan-300/[0.08] text-cyan-50" : "border-white/10 text-white/58 hover:text-white"}`}
              >
                {entityType}
              </button>
            ))}
          </div>
        ) : null}

        <div className={`grid min-h-0 flex-1 gap-4 ${isFullscreen ? "grid-cols-[minmax(0,1fr)_380px]" : "grid-cols-1"}`}>
          <div className="min-w-0 rounded-[24px] border border-white/8 bg-[rgba(4,9,24,0.36)] p-2">
            <NetworkCanvas
              key={`${graphType}:${selectedChapter || "all"}`}
              ref={canvasRef}
              graph={visibleGraph}
              height={isFullscreen ? 900 : 760}
              selectedNodeId={selectedNode?.id ?? null}
              onNodeClick={(nodeId, category) => setSelectedNode({ id: nodeId, category })}
              onNodeDoubleClick={(nodeId, category) => openDetail(nodeId, category)}
            />
          </div>

          {(detailNodeId || isFullscreen) && (
            <aside className={`glass-panel min-w-0 ${isFullscreen ? "kg-scroll-panel rounded-[24px]" : "hidden"} p-5`}>
              {detailNodeId ? (
                <NodeDetail
                  detailQuery={{
                    data: detailQuery.data,
                    isLoading: detailQuery.isLoading,
                    error: (detailQuery.error as Error | null) ?? null,
                  }}
                  onClose={() => setDetailNodeId(null)}
                />
              ) : (
                <GraphNodeSummary node={selectedGraphNode} graphType={graphType} />
              )}
            </aside>
          )}
        </div>
      </motion.section>

      <Dialog
        open={rebuildDialogOpen}
        onOpenChange={(open, eventDetails) => {
          if (!open && eventDetails?.reason === "focus-out") {
            return;
          }
          setRebuildDialogOpen(open);
        }}
      >
        <DialogContent
          className="max-w-md border border-white/10 bg-[rgba(3,7,20,0.92)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.4)] backdrop-blur-2xl"
          showCloseButton={!rebuildMutation.isPending && !rebuildBatchId}
        >
          <DialogHeader className="border-b border-white/8 px-6 py-5">
            <DialogTitle>确认重建图谱</DialogTitle>
            <DialogDescription>全量重建会清空并重建当前资料库的 Semantic KG；结构图和证据图会从持久文档与证据链重新生成。</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 px-6 py-5">
            {rebuildMutation.isPending || rebuildBatchId ? (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between gap-4 text-sm text-white/72">
                    <span className={rebuildPhase === "extraction" ? "text-white" : "text-white/40"}>图谱抽取 {rebuildPhase === "extraction" ? "进行中" : chunkProgress >= 100 ? "已完成" : "等待中"}</span>
                    <span className="min-w-12 text-right font-mono text-cyan-100">{chunkProgress}%</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={chunkProgress}>
                    <div
                      className={`h-full rounded-full transition-[width] duration-700 ${rebuildPhase === "extraction" ? "bg-[linear-gradient(90deg,#64dfff,#7b7cff,#64dfff)]" : "bg-white/20"}`}
                      style={{ width: `${chunkProgress}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between gap-4 text-sm text-white/72">
                    <span className={rebuildPhase === "community_summary" ? "text-white" : "text-white/40"}>社区摘要 {rebuildPhase === "community_summary" ? "生成中" : communityProgress >= 100 ? "已完成" : "等待中"}</span>
                    <span className="min-w-12 text-right font-mono text-cyan-100">{communityProgress}%</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={communityProgress}>
                    <div
                      className={`h-full rounded-full transition-[width] duration-700 ${rebuildPhase === "community_summary" ? "bg-[linear-gradient(90deg,#64dfff,#7b7cff,#64dfff)]" : "bg-white/20"}`}
                      style={{ width: `${communityProgress}%` }}
                    />
                  </div>
                </div>
                {graphLogRetryCount > 0 ? (
                  <p className="mt-1 rounded-full border border-amber-200/14 bg-amber-300/[0.055] px-3 py-2 text-[11px] leading-5 text-amber-50/72">
                    日志流重连 {Math.min(graphLogRetryCount, graphLogStreamMaxRetries)}/{graphLogStreamMaxRetries}
                  </p>
                ) : null}
              </div>
            ) : rebuildMutation.isError ? (
              <p className="rounded-2xl border border-rose-200/16 bg-rose-300/[0.055] px-4 py-3 text-sm leading-6 text-rose-50/78">
                重建失败：{(rebuildMutation.error as Error)?.message || "未知错误"}
              </p>
            ) : (
              <div className="space-y-3">
                {dryRunResult ? (
                  <div className="rounded-2xl border border-cyan-200/18 bg-cyan-300/[0.055] px-4 py-3 text-xs leading-5 text-cyan-50/74">
                    Dry-run：将扫描 {dryRunResult.affectedDocuments} 份文档；当前 Semantic KG 有 {dryRunResult.semanticEntities} 个实体、{dryRunResult.semanticRelations} 条关系。未写入数据库。
                  </div>
                ) : null}
                {dryRunMutation.isError ? (
                  <div className="rounded-2xl border border-rose-200/16 bg-rose-300/[0.055] px-4 py-3 text-xs leading-5 text-rose-50/78">
                    Dry-run 失败：{(dryRunMutation.error as Error)?.message || "未知错误"}
                  </div>
                ) : null}
                <label className={`flex cursor-pointer items-start gap-3 rounded-2xl border p-4 transition ${rebuildMode === "incremental" ? "border-cyan-200/30 bg-cyan-300/[0.06]" : "border-white/10 bg-white/[0.035] hover:bg-white/[0.05]"}`}>
                  <input
                    type="radio"
                    name="rebuild-mode"
                    className="mt-1"
                    checked={rebuildMode === "incremental"}
                    onChange={() => {
                      setRebuildMode("incremental");
                      setDryRunResult(null);
                    }}
                  />
                  <div>
                    <p className="text-sm font-medium text-white">增量更新（推荐）</p>
                    <p className="mt-1 text-xs leading-5 text-white/60">仅更新变更文档相关的图谱节点，速度快，不影响现有结构。</p>
                  </div>
                </label>
                <label className={`flex cursor-pointer items-start gap-3 rounded-2xl border p-4 transition ${rebuildMode === "full" ? "border-cyan-200/30 bg-cyan-300/[0.06]" : "border-white/10 bg-white/[0.035] hover:bg-white/[0.05]"}`}>
                  <input
                    type="radio"
                    name="rebuild-mode"
                    className="mt-1"
                    checked={rebuildMode === "full"}
                    onChange={() => {
                      setRebuildMode("full");
                      setDryRunResult(null);
                    }}
                  />
                  <div>
                    <p className="text-sm font-medium text-white">全量重建</p>
                    <p className="mt-1 text-xs leading-5 text-white/60">完整重建当前资料库的 Semantic KG，耗时较长，会自动保留备份并在失败时回滚。</p>
                  </div>
                </label>
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                disabled={rebuildMutation.isPending || dryRunMutation.isPending || Boolean(rebuildBatchId)}
                onClick={() => setRebuildDialogOpen(false)}
                className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:text-white disabled:pointer-events-none disabled:opacity-45"
              >
                {rebuildMutation.isError ? "关闭" : "取消"}
              </button>
              {!rebuildMutation.isError && !rebuildMutation.isPending && !rebuildBatchId ? (
                <>
                  <button
                    type="button"
                    disabled={!selectedCourseId || dryRunMutation.isPending}
                    onClick={() => dryRunMutation.mutate()}
                    className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:text-white disabled:pointer-events-none disabled:opacity-45"
                  >
                    {dryRunMutation.isPending ? "检查中..." : "先做 dry-run"}
                  </button>
                  <button
                    type="button"
                    disabled={!selectedCourseId || dryRunMutation.isPending}
                    onClick={() => rebuildMutation.mutate()}
                    className="rounded-full border border-cyan-200/24 bg-cyan-300/[0.08] px-4 py-2 text-sm text-cyan-50/82 transition hover:text-white disabled:pointer-events-none disabled:opacity-45"
                  >
                    确认重建
                  </button>
                </>
              ) : null}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={rebuildFailureDialog !== null} onOpenChange={(open) => !open && setRebuildFailureDialog(null)}>
        <DialogContent className="max-w-md border border-rose-200/18 bg-[rgba(18,6,12,0.94)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.4)] backdrop-blur-2xl">
          <DialogHeader className="border-b border-rose-200/12 px-6 py-5">
            <DialogTitle>{rebuildFailureDialog?.title ?? "图谱重建失败"}</DialogTitle>
            <DialogDescription>{rebuildFailureDialog?.message ?? "后端未返回错误详情。"}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 px-6 py-5">
            {rebuildFailureDialog?.details ? (
              <pre className="max-h-52 overflow-auto whitespace-pre-wrap rounded-2xl border border-rose-200/12 bg-black/20 px-4 py-3 text-xs leading-5 text-rose-50/78">
                {rebuildFailureDialog.details}
              </pre>
            ) : null}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setRebuildMode("incremental");
                  setRebuildFailureDialog(null);
                  rebuildMutation.mutate();
                }}
                className="rounded-full border border-cyan-200/24 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-50 transition hover:bg-cyan-300/16"
              >
                继续生成图谱
              </button>
              <button type="button" onClick={() => setRebuildFailureDialog(null)} className="rounded-full border border-rose-200/20 px-4 py-2 text-sm text-rose-50/78 transition hover:text-white">
                关闭
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {!isFullscreen ? (
        <motion.section initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} className="glass-panel kg-scroll-panel min-w-0 rounded-[28px] p-5">
          <NodeDetail
            detailQuery={{
              data: detailQuery.data,
              isLoading: detailQuery.isLoading,
              error: (detailQuery.error as Error | null) ?? null,
            }}
            onClose={() => setDetailNodeId(null)}
          />
          {!detailNodeId ? <GraphNodeSummary node={selectedGraphNode} graphType={graphType} /> : null}
        </motion.section>
      ) : null}
    </div>
  );
}

export function GraphPanel() {
  const { selectedCourseId } = useCourseContext();
  return <GraphPanelContent key={selectedCourseId ?? "unassigned"} selectedCourseId={selectedCourseId} />;
}

function NodeDetail({
  detailQuery,
  onClose,
}: {
  detailQuery: {
    data?: GraphNodeDetail;
    isLoading: boolean;
    error: Error | null;
  };
  onClose: () => void;
}) {
  return (
    <>
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="section-kicker">节点详情</p>
          <h2 className="mt-2 break-words text-2xl font-semibold text-white">知识详解</h2>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <ScanSearch className="size-5 text-cyan-200" />
          {detailQuery.data ? (
            <motion.button whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }} type="button" onClick={onClose} className="action-chip rounded-full px-3 py-2 text-xs uppercase tracking-[0.22em]">
              关闭
            </motion.button>
          ) : null}
        </div>
      </div>

      {detailQuery.isLoading ? (
        <div className="mt-6">
          <LoadingBlock rows={3} />
        </div>
      ) : detailQuery.error ? (
        <div className="mt-6">
          <ErrorBlock message={detailQuery.error.message} />
        </div>
      ) : detailQuery.data ? (
        <div className="mt-6 space-y-5">
          <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
            <p className="break-words text-xs uppercase tracking-[0.26em] text-white/45">{detailQuery.data.concept_type}</p>
            <p className="mt-3 break-words text-3xl font-semibold text-white">{detailQuery.data.name}</p>
            <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-white/58">
              <span>Community {detailQuery.data.community_louvain ?? "n/a"}</span>
              <span>Evidence {detailQuery.data.evidence_count}</span>
              <span>Centrality {(detailQuery.data.centrality?.centrality_score ?? 0).toFixed(3)}</span>
              <span>Rank {detailQuery.data.graph_rank_score.toFixed(3)}</span>
            </div>
            <p className="mt-3 break-words text-sm leading-8 text-white/68">
              {detailQuery.data.summary || "当前节点缺少完整摘要，已展示最小可用信息。"}
            </p>
          </div>

          <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
            <p className="text-xs uppercase tracking-[0.26em] text-white/45">别名</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {detailQuery.data.aliases.length > 0 ? (
                detailQuery.data.aliases.map((alias) => (
                  <span key={alias} className="max-w-full break-words rounded-full border border-white/10 px-3 py-1 text-sm text-cyan-50/76">
                    {alias}
                  </span>
                ))
              ) : (
                <span className="text-sm text-white/58">暂无别名</span>
              )}
            </div>
          </div>

          <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
            <p className="text-xs uppercase tracking-[0.26em] text-white/45">目录引用</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {detailQuery.data.chapter_refs.length > 0 ? (
                detailQuery.data.chapter_refs.map((chapter) => (
                  <span key={chapter} className="max-w-full break-words rounded-full border border-white/10 px-3 py-1 text-sm text-cyan-50/76">
                    {chapter}
                  </span>
                ))
              ) : (
                <span className="text-sm text-white/58">暂无目录引用</span>
              )}
            </div>
          </div>

          <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
            <p className="text-xs uppercase tracking-[0.26em] text-white/45">相关关系与证据</p>
            <div className="mt-4 space-y-3">
              {detailQuery.data.relations.length > 0 ? (
                detailQuery.data.relations.map((relation) => (
                  <div key={relation.relation_id} className="rounded-[18px] border border-white/8 px-4 py-4">
                    <div className="flex items-start justify-between gap-4">
                      <p className="min-w-0 break-words text-sm font-medium leading-6 text-white">{relation.target_name}</p>
                      <span className="shrink-0 text-xs text-white/42">{relation.confidence.toFixed(2)}</span>
                    </div>
                    <p className="mt-1 break-words text-xs uppercase tracking-[0.22em] text-white/45">
                      {relation.relation_type} - w={(relation.weight ?? relation.confidence).toFixed(2)}{relation.is_inferred ? " - inferred" : ""}
                    </p>
                    {relation.evidence ? <MarkdownRenderer content={relation.evidence.snippet} compact className="mt-3 break-words text-white/58" /> : null}
                  </div>
                ))
              ) : (
                <p className="break-words text-sm text-white/58">当前节点暂无可展示的关系。</p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6 rounded-[22px] border border-white/8 bg-white/[0.03] px-5 py-6 text-sm leading-7 text-white/58 break-words">
          双击图谱中的概念节点即可查看知识详解。单击只会高亮节点，不会打断浏览。
        </div>
      )}
    </>
  );
}

function GraphNodeSummary({ node, graphType }: { node: GraphResponse["nodes"][number] | null; graphType: GraphType }) {
  if (!node) {
    return (
      <div className="mt-6 rounded-[24px] border border-white/8 bg-white/[0.03] p-5 text-sm leading-7 text-white/58">
        选择一个节点查看当前图层中的基础信息。
      </div>
    );
  }
  const rows = [
    ["类型", node.entity_type ?? node.category],
    ["目录", node.chapter ?? "n/a"],
    ["证据", node.evidence_count ?? node.support_count ?? "n/a"],
    ["页码", node.page_number ?? "n/a"],
    ["文档版本", node.document_version_id ?? "n/a"],
  ];
  return (
    <div className="mt-6 space-y-4">
      <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
        <p className="break-words text-xs uppercase tracking-[0.26em] text-white/45">{graphType}</p>
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
    </div>
  );
}
