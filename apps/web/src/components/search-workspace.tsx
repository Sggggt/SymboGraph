"use client";

import { createPortal } from "react-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphResponse, SearchResult, SourceType } from "@course-kg/shared";
import { AnimatePresence, motion } from "framer-motion";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowUpRight,
  FileText,
  Filter,
  GitBranch,
  Loader2,
  Radar,
  Search,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";

import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { NetworkCanvas } from "@/components/network-canvas";
import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { fetchDashboard, fetchGraph, searchKnowledge } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

const sourceOptions: SourceType[] = ["pdf", "notebook", "markdown", "text", "image", "docx", "pptx"];
const emptySearchResults: SearchResult[] = [];

export function toLayeredContextGraph(graph: GraphResponse | undefined): GraphResponse | undefined {
  if (!graph) {
    return undefined;
  }
  return {
    ...graph,
    graph_type: "chunk-relation",
  };
}

type HoverPreviewState = {
  result: SearchResult;
  top: number;
  left: number;
  width: number;
};

type HoverPreviewTimer = ReturnType<typeof setTimeout> | null;

function resultTraversal(result: SearchResult): Record<string, unknown> {
  const traversal = result.metadata.traversal;
  return traversal && typeof traversal === "object" && !Array.isArray(traversal) ? (traversal as Record<string, unknown>) : {};
}

function traversalTextList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function traversalMetric(result: SearchResult, key: string): string {
  const value = resultTraversal(result)[key];
  if (typeof value === "number") {
    return value.toFixed(3);
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(" / ") : "无";
  }
  return value === undefined || value === null || value === "" ? "无" : String(value);
}

function resultRq(result: SearchResult): Record<string, unknown> | null {
  const rq = result.metadata.rq;
  return rq && typeof rq === "object" && !Array.isArray(rq) ? (rq as Record<string, unknown>) : null;
}

function rqPath(value: unknown): string {
  return Array.isArray(value) && value.length ? value.join("/") : "无";
}

function sourceTypeLabel(value: string | undefined) {
  const labels: Record<string, string> = {
    pdf: "PDF",
    notebook: "笔记本",
    markdown: "Markdown 文档",
    text: "文本",
    image: "图片",
    docx: "Word 文档",
    pptx: "演示文稿",
  };
  return value ? labels[value] ?? value : "未知";
}

function resultPartition(result: SearchResult | undefined) {
  if (!result) return "通用";
  return result.partition ?? (result.metadata.partition as string | undefined) ?? result.citations[0]?.partition ?? "通用";
}

function resultSourceType(result: SearchResult | undefined) {
  if (!result) return "未知";
  return sourceTypeLabel(result.source_type ?? (result.metadata.source_type as string | undefined));
}

function resultContextSnippet(result: SearchResult) {
  return result.citations[0]?.snippet || result.snippet || result.content || "";
}

function SearchHero({
  query,
  setQuery,
  onSearch,
  isSearching,
  hitCount,
  history,
  onPickHistory,
}: {
  query: string;
  setQuery: (value: string) => void;
  onSearch: () => void;
  isSearching: boolean;
  hitCount: number;
  history: string[];
  onPickHistory: (value: string) => void;
}) {
  const [historyOpen, setHistoryOpen] = useState(false);
  return (
    <section className="relative overflow-hidden rounded-[2rem] px-1 py-2">
      <div className="pointer-events-none absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-cyan-200/50 to-transparent" />
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-5 text-center">
        <div className="kg-micro-chip rounded-full px-3 py-2 text-xs uppercase tracking-[0.22em]">
          <Radar data-icon="inline-start" />
          混合检索链路
        </div>
        <div className="space-y-3">
          <h2 className="glow-text text-4xl font-semibold text-white lg:text-6xl">检索本地上下文图谱</h2>
          <p className="mx-auto max-w-2xl text-sm leading-7 text-cyan-50/58 lg:text-base">
            稠密向量、BM25、细聚类、RQ-KMeans、中粗概念路由与结构恢复会在这里汇总。
          </p>
        </div>

        <div className={cn("kg-glass-line kg-scan-edge relative z-20 w-full rounded-[1.7rem] p-2", isSearching && "shadow-[0_0_42px_rgba(86,217,255,0.12)]")}>
          <div className="flex items-center gap-3 rounded-[1.35rem] bg-black/18 px-4 py-3">
            <Search className="text-cyan-100/70" />
            <input
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setHistoryOpen(true);
              }}
              onFocus={() => setHistoryOpen(true)}
              onBlur={() => window.setTimeout(() => setHistoryOpen(false), 140)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  onSearch();
                }
              }}
              className="h-12 min-w-0 flex-1 bg-transparent text-lg text-white outline-none placeholder:text-white/28"
              placeholder="输入要从本地资料中检索的问题或引用线索..."
            />
            <div className="hidden items-center gap-2 md:flex">
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">{hitCount} 条结果</span>
              {isSearching ? (
                <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">
                  <span className="tech-dot" />
                  检索中
                </span>
              ) : null}
            </div>
            <Button type="button" size="lg" onClick={onSearch} disabled={isSearching || !query.trim()} className="rounded-full">
              {isSearching ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Sparkles data-icon="inline-start" />}
              {isSearching ? "搜索中" : "搜索"}
            </Button>
          </div>
          {historyOpen && history.length > 0 ? (
            <div className="custom-scrollbar absolute left-4 right-4 top-[calc(100%+0.5rem)] z-20 max-h-72 overflow-y-auto rounded-[1.35rem] border border-white/10 bg-[rgba(4,10,24,0.96)] p-2 text-left shadow-[0_24px_70px_rgba(0,0,0,0.42)] backdrop-blur-2xl">
              {history.map((item) => (
                <button
                  key={item}
                  type="button"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => onPickHistory(item)}
                  className="flex w-full items-center gap-3 rounded-2xl px-3 py-2.5 text-sm text-white/70 transition hover:bg-cyan-300/[0.08] hover:text-white"
                >
                  <Search className="size-4 text-cyan-100/52" />
                  <span className="min-w-0 truncate">{item}</span>
                </button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function SearchFilterBar({
  partition,
  sourceType,
  onOpenFilters,
  onClearPartition,
  onClearSource,
  degradedMode,
}: {
  partition: string;
  sourceType: string;
  onOpenFilters: () => void;
  onClearPartition: () => void;
  onClearSource: () => void;
  degradedMode: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={onOpenFilters} className="kg-micro-chip rounded-full px-3 py-2 text-xs uppercase tracking-[0.18em] transition hover:border-cyan-200/30 hover:text-white">
          <SlidersHorizontal />
          筛选
        </button>
        <button type="button" onClick={onClearPartition} className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          目录：{partition || "全部"}
          {partition ? <X /> : null}
        </button>
        <button type="button" onClick={onClearSource} className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          来源：{sourceType ? sourceTypeLabel(sourceType) : "全部"}
          {sourceType ? <X /> : null}
        </button>
      </div>
      <div className="kg-micro-chip rounded-full px-3 py-2 text-xs">
        <Activity data-icon="inline-start" />
        {degradedMode ? "依赖链路不完整" : "向量 + BM25 + 图谱"}
      </div>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      {[0, 1, 2, 3].map((item) => (
        <div key={item} className="kg-shimmer rounded-2xl border border-white/7 bg-white/[0.035] p-5">
          <div className="h-4 w-1/3 rounded-full bg-white/10" />
          <div className="mt-4 h-3 w-4/5 rounded-full bg-white/8" />
          <div className="mt-2 h-3 w-3/5 rounded-full bg-white/8" />
        </div>
      ))}
    </div>
  );
}

function ResultRow({
  result,
  active,
  index,
  onHover,
  onSelect,
}: {
  result: SearchResult;
  active: boolean;
  index: number;
  onHover: (result: SearchResult | null, anchor?: HTMLButtonElement | null) => void;
  onSelect: (result: SearchResult) => void;
}) {
  const rq = resultRq(result);
  const traversal = resultTraversal(result);
  const evidenceRoles = traversalTextList(traversal.evidence_roles).slice(0, 4);
  const coveredFacets = traversalTextList(traversal.covered_facets).slice(0, 4);
  return (
    <motion.button
      type="button"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.035 }}
      onPointerEnter={(event) => onHover(result, event.currentTarget)}
      onPointerLeave={() => onHover(null)}
      onFocus={(event) => onHover(result, event.currentTarget)}
      onBlur={() => onHover(null)}
      onClick={() => onSelect(result)}
      className={cn(
        "group relative w-full overflow-hidden rounded-2xl border px-5 py-4 text-left transition duration-200",
        active
          ? "border-cyan-200/36 bg-cyan-300/[0.075] shadow-[0_0_34px_rgba(86,217,255,0.08)]"
          : "border-white/7 bg-white/[0.025] hover:border-cyan-200/24 hover:bg-white/[0.045]",
      )}
    >
      <div className="absolute inset-y-4 left-0 w-px bg-gradient-to-b from-transparent via-cyan-200/60 to-transparent opacity-0 transition group-hover:opacity-100" />
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-white">{result.document_title ?? result.citations[0]?.document_title ?? "资料来源"}</span>
            <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultSourceType(result)}</span>
            <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultPartition(result)}</span>
          </div>
          <MarkdownRenderer content={result.snippet} compact className="mt-3 line-clamp-2 text-white/62" />
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <span className="font-mono text-sm text-cyan-100">{result.score.toFixed(3)}</span>
          <ArrowUpRight className="text-white/32 transition group-hover:text-cyan-100" />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">距离 {traversalMetric(result, "distance_so_far")}</span>
        <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">Cycle {traversalMetric(result, "reward_so_far")}</span>
        {coveredFacets.length ? (
          <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">覆盖 {coveredFacets.join("/")}</span>
        ) : null}
        {evidenceRoles.map((role) => (
          <span key={role} className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">
            {role}
          </span>
        ))}
        {rq ? (
          <>
            <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">RQ 路径 {rqPath(rq.candidate_rq_path)}</span>
            <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">公共前缀 {String(rq.lcp_depth ?? "无")}</span>
            <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">残差 {String(rq.residual_distance ?? "无")}</span>
          </>
        ) : null}
      </div>
    </motion.button>
  );
}

function ResultStream({
  results,
  selectedChunkId,
  isLoading,
  onHover,
  onSelect,
}: {
  results: SearchResult[];
  selectedChunkId: string | null;
  isLoading: boolean;
  onHover: (result: SearchResult | null, anchor?: HTMLButtonElement | null) => void;
  onSelect: (result: SearchResult) => void;
}) {
  return (
    <section className="kg-glass-line kg-scroll-shell flex min-h-0 min-w-0 flex-col overflow-hidden rounded-[2rem] p-2">
      <div className="flex shrink-0 items-center justify-between gap-3 px-3 pb-4 pt-3">
        <div>
          <p className="section-kicker">结果流</p>
          <h3 className="mt-1 text-xl font-semibold text-white">已排序知识片段</h3>
        </div>
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">{results.length} 条结果</span>
      </div>
      <div className="kg-scroll-body min-h-0 flex-1 px-3 pb-3">
        {isLoading ? (
          <ResultSkeleton />
        ) : results.length === 0 ? (
          <div className="kg-glass-line rounded-3xl px-6 py-10 text-center">
            <div className="mx-auto grid size-14 place-items-center rounded-2xl border border-cyan-200/15 bg-cyan-300/[0.06] text-cyan-100">
              <Search />
            </div>
            <h4 className="mt-5 text-lg font-medium text-white">尚未发起检索</h4>
            <p className="mx-auto mt-2 max-w-md text-sm leading-7 text-white/52">
              发起一次图遍历检索后，这里会展示排序片段、frontier 路径和关联图谱上下文。
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {results.map((result, index) => (
              <ResultRow
                key={result.chunk_id}
                result={result}
                index={index}
                active={selectedChunkId === result.chunk_id}
                onHover={onHover}
                onSelect={onSelect}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function HoverPreviewOverlay({ preview }: { preview: HoverPreviewState | null }) {
  if (typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <AnimatePresence>
      {preview ? (
        <div className="pointer-events-none fixed inset-0 z-[120] overflow-hidden">
          <motion.div
            key={preview.result.chunk_id}
            initial={{ opacity: 0, x: -10, y: 14, scale: 0.94 }}
            animate={{ opacity: 1, x: 0, y: 0, scale: 1 }}
            exit={{ opacity: 0, x: -8, y: 8, scale: 0.96 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
            className="kg-glass-line absolute rounded-[1.5rem] p-4 shadow-[0_28px_80px_rgba(0,0,0,0.34)]"
            style={{ top: preview.top, left: preview.left, width: preview.width }}
          >
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs uppercase tracking-[0.22em] text-cyan-100/55">悬停预览</p>
              <div className="flex flex-wrap gap-2">
                <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultPartition(preview.result)}</span>
                <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultSourceType(preview.result)}</span>
              </div>
            </div>
            <p className="mt-3 text-sm font-medium text-white">
              {preview.result.document_title ?? preview.result.citations[0]?.document_title ?? "资料来源"}
            </p>
            <MarkdownRenderer content={preview.result.snippet} compact className="mt-3 line-clamp-4 text-white/76" />
          </motion.div>
        </div>
      ) : null}
    </AnimatePresence>,
    document.body,
  );
}

function GraphCanvasPanel({
  selectedLabel,
  selectedPartition,
  selectedNodeId,
  graph,
  hasResults,
  isLoading,
  error,
}: {
  selectedLabel: string | null;
  selectedPartition: string;
  selectedNodeId: string | null;
  graph: GraphResponse | undefined;
  hasResults: boolean;
  isLoading: boolean;
  error: Error | null;
}) {
  return (
    <section className="kg-glass-line kg-scroll-shell relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-[2rem] bg-[rgba(4,8,22,0.28)]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_20%,rgba(86,217,255,0.11),transparent_38%),linear-gradient(rgba(120,180,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(120,180,255,0.035)_1px,transparent_1px)] bg-[size:auto,42px_42px,42px_42px]" />
      <div className="relative z-10 flex items-center justify-between gap-3 px-5 py-4">
        <div>
          <p className="section-kicker">知识画布</p>
          <h3 className="mt-1 text-xl font-semibold text-white">{selectedLabel ?? "检索上下文图谱"}</h3>
          <p className="mt-1 text-sm text-white/42">{selectedPartition || "当前检索结果的片段关系邻域"}</p>
        </div>
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          <GitBranch data-icon="inline-start" />
          四层图谱
        </span>
      </div>
      <div className="relative z-10 flex min-h-0 flex-1 px-2 pb-2">
        {isLoading ? (
          <div className="kg-shimmer mx-3 grid h-full min-h-0 w-full flex-1 place-items-center rounded-[1.5rem] border border-white/7 bg-white/[0.025] text-sm text-white/54">
            正在加载关系图谱...
          </div>
        ) : error ? (
          <div className="mx-3 flex min-h-0 w-full flex-1 items-stretch">
            <ErrorBlock message={error.message} />
          </div>
        ) : graph && graph.nodes.length > 0 ? (
          <div className="mx-3 flex h-full min-h-0 w-full flex-1">
            <NetworkCanvas graph={graph} height="100%" selectedNodeId={selectedNodeId} />
          </div>
        ) : (
          <div className="mx-3 grid h-full min-h-0 w-full flex-1 place-items-center rounded-[1.5rem] border border-white/7 bg-white/[0.025] px-6 text-center text-sm leading-7 text-white/54">
            {hasResults ? "当前检索结果暂未返回可展示的关系图谱。" : "发起检索后，这里展示片段关系图谱和命中片段上下文。"}
          </div>
        )}
      </div>
    </section>
  );
}

function DetailDrawer({ result, open, onOpenChange }: { result: SearchResult | null; open: boolean; onOpenChange: (open: boolean) => void }) {
  const rq = result ? resultRq(result) : null;
  const traversal = result ? resultTraversal(result) : {};
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-xl">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>结果详情</SheetTitle>
          <SheetDescription>片段证据、图遍历路径和来源元数据。</SheetDescription>
        </SheetHeader>
        {result ? (
          <div className="custom-scrollbar flex-1 overflow-y-auto p-6">
            <div className="flex flex-wrap gap-2">
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">{resultPartition(result)}</span>
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">{resultSourceType(result)}</span>
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">分数 {result.score.toFixed(3)}</span>
            </div>
            <h3 className="mt-5 text-2xl font-semibold text-white">{result.document_title ?? result.citations[0]?.document_title ?? "资料来源"}</h3>
            <p className="mt-3 flex items-center gap-2 truncate text-sm text-white/42">
              <FileText />
              {result.source_path ?? result.citations[0]?.source_path}
            </p>
            <div className="kg-flow-line my-6" />
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-5">
              <p className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">命中证据</p>
              <MarkdownRenderer content={resultContextSnippet(result)} className="mt-4 text-white/70" />
            </div>
            {result.content && result.content !== resultContextSnippet(result) ? (
              <div className="mt-5 rounded-2xl border border-white/8 bg-white/[0.03] p-5">
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">完整片段</p>
                <MarkdownRenderer content={result.content} className="mt-4 text-white/70" />
              </div>
            ) : null}
            <div className="mt-5 grid grid-cols-2 gap-3">
              {[
                ["distance_so_far", "路径距离"],
                ["reward_so_far", "Cycle reward"],
                ["covered_facets", "覆盖 facets"],
                ["path_edge_ids", "路径边"],
              ].map(([key, label]) => (
                <div key={key} className="rounded-2xl border border-white/8 bg-white/[0.025] p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-white/38">{label}</p>
                  <p className="mt-2 break-words font-mono text-sm text-cyan-100">{traversalMetric(result, key)}</p>
                </div>
              ))}
            </div>
            {Array.isArray(traversal.path) && traversal.path.length ? (
              <div className="mt-5 rounded-2xl border border-white/8 bg-white/[0.03] p-5">
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Graph Path</p>
                <p className="mt-4 break-words font-mono text-xs leading-6 text-white/62">{traversalTextList(traversal.path).join(" -> ")}</p>
              </div>
            ) : null}
            {rq ? (
              <div className="mt-5 rounded-2xl border border-white/8 bg-white/[0.03] p-5">
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">RQ-KMeans 残差路由</p>
                <div className="mt-4 grid gap-2 text-sm text-white/62">
                  <p>查询路径：{rqPath(rq.query_rq_path)}</p>
                  <p>候选路径：{rqPath(rq.candidate_rq_path)}</p>
                  <p>公共前缀深度：{String(rq.lcp_depth ?? "无")}</p>
                  <p>残差距离：{String(rq.residual_distance ?? "无")}</p>
                  <p>RQ 分数：{String(rq.rq_score ?? "无")}</p>
                  <p>漂移惩罚：{String(rq.rq_drift_penalty ?? "无")}</p>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}

function AdvancedFilterDrawer({
  open,
  onOpenChange,
  partition,
  setPartition,
  sourceType,
  setSourceType,
  partitionOptions,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  partition: string;
  setPartition: (value: string) => void;
  sourceType: string;
  setSourceType: (value: string) => void;
  partitionOptions: string[];
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-md">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>高级筛选</SheetTitle>
          <SheetDescription>按目录和来源通道限定检索范围。</SheetDescription>
        </SheetHeader>
        <div className="flex flex-col gap-6 p-6">
          <label className="flex flex-col gap-2">
            <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">目录</span>
            <select value={partition} onChange={(event) => setPartition(event.target.value)} className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-white outline-none">
              <option value="">全部目录</option>
              {partitionOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-2">
            <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">来源</span>
            <select value={sourceType} onChange={(event) => setSourceType(event.target.value)} className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-white outline-none">
              <option value="">全部来源</option>
              {sourceOptions.map((option) => (
                <option key={option} value={option}>
                  {sourceTypeLabel(option)}
                </option>
              ))}
            </select>
          </label>
          <Button type="button" onClick={() => onOpenChange(false)} className="rounded-full">
            <Filter data-icon="inline-start" />
            应用范围
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function SearchWorkspaceContent({ selectedKnowledgeBaseId }: { selectedKnowledgeBaseId: string | null }) {
  const storageScope = selectedKnowledgeBaseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedKnowledgeBaseId],
    queryFn: () => fetchDashboard(selectedKnowledgeBaseId, { includeGraph: false }),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const [query, setQuery] = useLocalStorage(`search.query.${storageScope}`, "");
  const [searchHistory, setSearchHistory] = useLocalStorage<string[]>(`search.history.${storageScope}`, []);
  const [partition, setPartition] = useLocalStorage(`search.partition.${storageScope}`, "");
  const [sourceType, setSourceType] = useLocalStorage(`search.sourceType.${storageScope}`, "");
  const [selectedPartition, setSelectedPartition] = useLocalStorage(`search.selectedPartition.${storageScope}`, "");
  const [selectedChunkId, setSelectedChunkId] = useLocalStorage<string | null>(`search.selectedChunkId.${storageScope}`, null);
  const [searchResults, setSearchResults] = useLocalStorage<{ results: SearchResult[]; degraded_mode: boolean } | null>(`search.results.${storageScope}`, null);
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null);
  const [detailResult, setDetailResult] = useState<SearchResult | null>(null);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const hoverPreviewTimerRef = useRef<HoverPreviewTimer>(null);

  const searchMutation = useMutation({
    mutationFn: (searchText: string) =>
      searchKnowledge({
        knowledge_base_id: selectedKnowledgeBaseId,
        query: searchText,
        top_k: 8,
        filters: {
          partition: partition || undefined,
          source_type: (sourceType || undefined) as never,
        },
      }),
    onSuccess: (data, searchText) => {
      setSearchResults({ results: data.results, degraded_mode: Boolean(data.degraded_mode) });
      setSearchHistory((current) => [searchText, ...current.filter((item) => item !== searchText)].slice(0, 50));
      const firstResult = data.results[0];
      const nextPartition = partition || (firstResult ? resultPartition(firstResult) : undefined) || (dashboardQuery.data?.tree[0]?.title ?? "");
      setSelectedPartition(nextPartition);
      setSelectedChunkId(firstResult?.chunk_id ?? null);
    },
    onError: () => {
      setSearchResults({ results: [], degraded_mode: false });
      setSelectedChunkId(null);
    },
  });

  const partitionOptions = useMemo(() => dashboardQuery.data?.tree.map((node) => node.title).filter((title): title is string => Boolean(title)) ?? [], [dashboardQuery.data]);
  const results = searchResults?.results ?? emptySearchResults;
  const resultChunkIds = useMemo(() => results.map((result) => result.chunk_id), [results]);
  const queryContextGraphQuery = useQuery({
    queryKey: ["search-context-graph", selectedKnowledgeBaseId, resultChunkIds],
    queryFn: () => fetchGraph(selectedKnowledgeBaseId, "chunk-relation", "overview"),
    enabled: resultChunkIds.length > 0,
  });
  const queryContextGraph = useMemo(() => toLayeredContextGraph(queryContextGraphQuery.data), [queryContextGraphQuery.data]);
  const selectedResult = results.find((result) => result.chunk_id === selectedChunkId) ?? null;
  const focusPartition = selectedResult ? resultPartition(selectedResult) : selectedPartition;
  const selectedLabel = selectedResult?.document_title ?? selectedResult?.citations[0]?.document_title ?? null;

  useEffect(() => {
    return () => {
      if (hoverPreviewTimerRef.current) {
        clearTimeout(hoverPreviewTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const dismissPreview = () => {
      if (hoverPreviewTimerRef.current) {
        clearTimeout(hoverPreviewTimerRef.current);
        hoverPreviewTimerRef.current = null;
      }
      setHoverPreview(null);
    };

    window.addEventListener("scroll", dismissPreview, true);
    window.addEventListener("resize", dismissPreview);

    return () => {
      window.removeEventListener("scroll", dismissPreview, true);
      window.removeEventListener("resize", dismissPreview);
    };
  }, []);

  const handleHoverPreview = (result: SearchResult | null, anchor?: HTMLButtonElement | null) => {
    if (hoverPreviewTimerRef.current) {
      clearTimeout(hoverPreviewTimerRef.current);
      hoverPreviewTimerRef.current = null;
    }

    if (!result || !anchor) {
      setHoverPreview(null);
      return;
    }

    const rect = anchor.getBoundingClientRect();
    const viewportWidth = document.documentElement.clientWidth;
    const viewportHeight = window.innerHeight;
    const width = Math.min(360, Math.max(300, viewportWidth - 32));
    const gap = 20;
    const preferredLeft = rect.right + gap;
    const fallbackLeft = rect.left - width - gap;
    const unclampedLeft = preferredLeft + width <= viewportWidth - 16 ? preferredLeft : fallbackLeft;
    const left = Math.max(16, Math.min(unclampedLeft, viewportWidth - width - 16));
    const top = Math.min(Math.max(16, rect.top - 8), Math.max(16, viewportHeight - 260));
    const nextPreview = { result, top, left, width };

    setHoverPreview(null);
    hoverPreviewTimerRef.current = setTimeout(() => {
      setHoverPreview(nextPreview);
      hoverPreviewTimerRef.current = null;
    }, 2000);
  };

  if (dashboardQuery.isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (dashboardQuery.error) {
    return <ErrorBlock message={(dashboardQuery.error as Error).message} />;
  }

  return (
    <div className="kg-page flex flex-col gap-6">
      <SearchHero
        query={query}
        setQuery={setQuery}
        onSearch={() => {
          const searchText = query.trim();
          if (searchText) {
            searchMutation.mutate(searchText);
          }
        }}
        isSearching={searchMutation.isPending}
        hitCount={results.length}
        history={searchHistory}
        onPickHistory={(value) => {
          setQuery(value);
          searchMutation.mutate(value);
        }}
      />
      <SearchFilterBar
        partition={partition}
        sourceType={sourceType}
        onOpenFilters={() => setFiltersOpen(true)}
        onClearPartition={() => setPartition("")}
        onClearSource={() => setSourceType("")}
        degradedMode={Boolean(searchResults?.degraded_mode)}
      />

      {searchMutation.error ? (
        <ErrorBlock message={(searchMutation.error as Error).message || "检索请求失败，请检查模型 API、Qdrant 和后端日志。"} />
      ) : null}

      <section className="grid min-h-0 items-stretch gap-6 xl:grid-cols-[minmax(360px,0.78fr)_minmax(520px,1.22fr)]">
        <ResultStream
          results={results}
          selectedChunkId={selectedChunkId}
          isLoading={searchMutation.isPending}
          onHover={handleHoverPreview}
          onSelect={(result) => {
            if (hoverPreviewTimerRef.current) {
              clearTimeout(hoverPreviewTimerRef.current);
              hoverPreviewTimerRef.current = null;
            }
            setHoverPreview(null);
            setSelectedChunkId(result.chunk_id);
            setSelectedPartition(resultPartition(result));
            setDetailResult(result);
          }}
        />
        <GraphCanvasPanel
          selectedLabel={selectedLabel}
          selectedPartition={focusPartition}
          selectedNodeId={null}
          graph={queryContextGraph}
          hasResults={results.length > 0}
          isLoading={queryContextGraphQuery.isLoading || searchMutation.isPending}
          error={(queryContextGraphQuery.error as Error | null) ?? null}
        />
      </section>

      <DetailDrawer result={detailResult} open={Boolean(detailResult)} onOpenChange={(open) => !open && setDetailResult(null)} />
      <AdvancedFilterDrawer
        open={filtersOpen}
        onOpenChange={setFiltersOpen}
        partition={partition}
        setPartition={setPartition}
        sourceType={sourceType}
        setSourceType={setSourceType}
        partitionOptions={partitionOptions}
      />
      <HoverPreviewOverlay preview={hoverPreview} />
    </div>
  );
}

export function SearchWorkspace() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  return <SearchWorkspaceContent key={selectedKnowledgeBaseId ?? "unassigned"} selectedKnowledgeBaseId={selectedKnowledgeBaseId} />;
}
