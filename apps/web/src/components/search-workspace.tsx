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
import { useCourseContext } from "@/components/course-context";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { fetchDashboard, fetchQuerySemanticGraph, searchKnowledge } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

const sourceOptions: SourceType[] = ["pdf", "notebook", "markdown", "text", "image", "docx", "pptx"];
const emptySearchResults: SearchResult[] = [];

type HoverPreviewState = {
  result: SearchResult;
  top: number;
  left: number;
  width: number;
};

type HoverPreviewTimer = ReturnType<typeof setTimeout> | null;

function scoreValue(result: SearchResult, key: string) {
  const scores = result.metadata.scores as Record<string, number | string | boolean | null> | undefined;
  const value = scores?.[key];
  return typeof value === "number" ? value.toFixed(3) : "未参与";
}

function scoreKeys(result: SearchResult) {
  return result.metadata.scores ? ["dense", "lexical", "fused", "rerank"] : [];
}

function scoreLabel(key: string) {
  const labels: Record<string, string> = {
    dense: "向量",
    lexical: "词面",
    fused: "融合",
    rerank: "重排",
  };
  return labels[key] ?? key;
}

function resultChapter(result: SearchResult | undefined) {
  if (!result) return "通用";
  return result.chapter ?? (result.metadata.chapter as string | undefined) ?? result.citations[0]?.chapter ?? "通用";
}

function resultSourceType(result: SearchResult | undefined) {
  if (!result) return "未知";
  return result.source_type ?? (result.metadata.source_type as string | undefined) ?? "未知";
}

function resultEvidenceSnippet(result: SearchResult) {
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
          <h2 className="glow-text text-4xl font-semibold text-white lg:text-6xl">检索本地知识信号</h2>
          <p className="mx-auto max-w-2xl text-sm leading-7 text-cyan-50/58 lg:text-base">
            Dense 向量召回、BM25 词面召回、WSF 融合排序与图谱上下文会在这里汇总。
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
              placeholder="输入概念、定理、练习或关系问题..."
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
  chapter,
  sourceType,
  onOpenFilters,
  onClearChapter,
  onClearSource,
  degradedMode,
}: {
  chapter: string;
  sourceType: string;
  onOpenFilters: () => void;
  onClearChapter: () => void;
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
        <button type="button" onClick={onClearChapter} className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          目录：{chapter || "全部"}
          {chapter ? <X /> : null}
        </button>
        <button type="button" onClick={onClearSource} className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          来源：{sourceType || "全部"}
          {sourceType ? <X /> : null}
        </button>
      </div>
      <div className="kg-micro-chip rounded-full px-3 py-2 text-xs">
        <Activity data-icon="inline-start" />
        {degradedMode ? "仅词面检索" : "Dense + BM25 + WSF"}
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
            <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultChapter(result)}</span>
          </div>
          <MarkdownRenderer content={result.snippet} compact className="mt-3 line-clamp-2 text-white/62" />
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <span className="font-mono text-sm text-cyan-100">{result.score.toFixed(3)}</span>
          <ArrowUpRight className="text-white/32 transition group-hover:text-cyan-100" />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {scoreKeys(result).map((key) => (
          <span key={key} className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">
            {scoreLabel(key)} {scoreValue(result, key)}
          </span>
        ))}
      </div>
    </motion.button>
  );
}

function ResultStream({
  results,
  activeChunkId,
  isLoading,
  onHover,
  onSelect,
}: {
  results: SearchResult[];
  activeChunkId: string | null;
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
              发起一次混合检索后，这里会展示排序片段、分数通道和关联图谱上下文。
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {results.map((result, index) => (
              <ResultRow
                key={result.chunk_id}
                result={result}
                index={index}
                active={activeChunkId === result.chunk_id}
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
                <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultChapter(preview.result)}</span>
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
  selectedChapter,
  selectedNodeId,
  graph,
  hasResults,
  isLoading,
  error,
}: {
  selectedLabel: string | null;
  selectedChapter: string;
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
          <h3 className="mt-1 text-xl font-semibold text-white">{selectedLabel ?? "查询语义子图"}</h3>
          <p className="mt-1 text-sm text-white/42">{selectedChapter || "当前检索结果的语义邻域"}</p>
        </div>
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          <GitBranch data-icon="inline-start" />
          语义子图
        </span>
      </div>
      <div className="relative z-10 flex min-h-0 flex-1 px-2 pb-2">
        {isLoading ? (
          <div className="kg-shimmer mx-3 grid h-full min-h-0 w-full flex-1 place-items-center rounded-[1.5rem] border border-white/7 bg-white/[0.025] text-sm text-white/54">
            正在生成图谱信号...
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
            {hasResults ? "当前检索结果没有匹配到可审计的语义图关系。" : "发起检索后，这里只展示命中片段相关的语义实体、证据片段和真实关系。"}
          </div>
        )}
      </div>
    </section>
  );
}

function DetailDrawer({ result, open, onOpenChange }: { result: SearchResult | null; open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-xl">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>结果详情</SheetTitle>
          <SheetDescription>片段证据、检索分数和来源元数据。</SheetDescription>
        </SheetHeader>
        {result ? (
          <div className="custom-scrollbar flex-1 overflow-y-auto p-6">
            <div className="flex flex-wrap gap-2">
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">{resultChapter(result)}</span>
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
              <MarkdownRenderer content={resultEvidenceSnippet(result)} className="mt-4 text-white/70" />
            </div>
            {result.content && result.content !== resultEvidenceSnippet(result) ? (
              <div className="mt-5 rounded-2xl border border-white/8 bg-white/[0.03] p-5">
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">完整片段</p>
                <MarkdownRenderer content={result.content} className="mt-4 text-white/70" />
              </div>
            ) : null}
            <div className="mt-5 grid grid-cols-2 gap-3">
              {scoreKeys(result).map((key) => (
                <div key={key} className="rounded-2xl border border-white/8 bg-white/[0.025] p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-white/38">{scoreLabel(key)}</p>
                  <p className="mt-2 font-mono text-lg text-cyan-100">{scoreValue(result, key)}</p>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}

function AdvancedFilterDrawer({
  open,
  onOpenChange,
  chapter,
  setChapter,
  sourceType,
  setSourceType,
  chapterOptions,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chapter: string;
  setChapter: (value: string) => void;
  sourceType: string;
  setSourceType: (value: string) => void;
  chapterOptions: string[];
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
            <select value={chapter} onChange={(event) => setChapter(event.target.value)} className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-white outline-none">
              <option value="">全部目录</option>
              {chapterOptions.map((option) => (
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
                  {option}
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

function SearchWorkspaceContent({ selectedCourseId }: { selectedCourseId: string | null }) {
  const storageScope = selectedCourseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedCourseId],
    queryFn: () => fetchDashboard(selectedCourseId),
    enabled: Boolean(selectedCourseId),
  });
  const [query, setQuery] = useLocalStorage(`search.query.${storageScope}`, "");
  const [searchHistory, setSearchHistory] = useLocalStorage<string[]>(`search.history.${storageScope}`, []);
  const [chapter, setChapter] = useLocalStorage(`search.chapter.${storageScope}`, "");
  const [sourceType, setSourceType] = useLocalStorage(`search.sourceType.${storageScope}`, "");
  const [selectedChapter, setSelectedChapter] = useLocalStorage(`search.selectedChapter.${storageScope}`, "");
  const [activeChunkId, setActiveChunkId] = useLocalStorage<string | null>(`search.activeChunkId.${storageScope}`, null);
  const [searchResults, setSearchResults] = useLocalStorage<{ results: SearchResult[]; degraded_mode: boolean } | null>(`search.results.${storageScope}`, null);
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null);
  const [detailResult, setDetailResult] = useState<SearchResult | null>(null);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const hoverPreviewTimerRef = useRef<HoverPreviewTimer>(null);

  const searchMutation = useMutation({
    mutationFn: (searchText: string) =>
      searchKnowledge({
        course_id: selectedCourseId,
        query: searchText,
        top_k: 8,
        filters: {
          chapter: chapter || undefined,
          source_type: (sourceType || undefined) as never,
        },
      }),
    onSuccess: (data, searchText) => {
      setSearchResults({ results: data.results, degraded_mode: Boolean(data.degraded_mode) });
      setSearchHistory((current) => [searchText, ...current.filter((item) => item !== searchText)].slice(0, 50));
      const firstResult = data.results[0];
      const nextChapter = chapter || (firstResult ? resultChapter(firstResult) : undefined) || (dashboardQuery.data?.tree[0]?.title ?? "");
      setSelectedChapter(nextChapter);
      setActiveChunkId(firstResult?.chunk_id ?? null);
    },
    onError: () => {
      setSearchResults({ results: [], degraded_mode: false });
      setActiveChunkId(null);
    },
  });

  const chapterOptions = useMemo(() => dashboardQuery.data?.tree.map((node) => node.title) ?? [], [dashboardQuery.data]);
  const results = searchResults?.results ?? emptySearchResults;
  const resultChunkIds = useMemo(() => results.map((result) => result.chunk_id), [results]);
  const querySemanticGraphQuery = useQuery({
    queryKey: ["search-semantic-graph", selectedCourseId, query, resultChunkIds],
    queryFn: () => fetchQuerySemanticGraph({ course_id: selectedCourseId, query, chunk_ids: resultChunkIds }),
    enabled: resultChunkIds.length > 0,
  });
  const selectedResult = results.find((result) => result.chunk_id === activeChunkId) ?? null;
  const focusChapter = selectedResult ? resultChapter(selectedResult) : selectedChapter;
  const selectedNodeId = selectedResult?.chunk_id ? `evidence_chunk:${selectedResult.chunk_id}` : null;
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
        chapter={chapter}
        sourceType={sourceType}
        onOpenFilters={() => setFiltersOpen(true)}
        onClearChapter={() => setChapter("")}
        onClearSource={() => setSourceType("")}
        degradedMode={Boolean(searchResults?.degraded_mode)}
      />

      {searchMutation.error ? (
        <ErrorBlock message={(searchMutation.error as Error).message || "检索请求失败，请检查模型 API、Qdrant 和后端日志。"} />
      ) : null}

      <section className="grid min-h-0 items-stretch gap-6 xl:grid-cols-[minmax(360px,0.78fr)_minmax(520px,1.22fr)]">
        <ResultStream
          results={results}
          activeChunkId={activeChunkId}
          isLoading={searchMutation.isPending}
          onHover={handleHoverPreview}
          onSelect={(result) => {
            if (hoverPreviewTimerRef.current) {
              clearTimeout(hoverPreviewTimerRef.current);
              hoverPreviewTimerRef.current = null;
            }
            setHoverPreview(null);
            setActiveChunkId(result.chunk_id);
            setSelectedChapter(resultChapter(result));
            setDetailResult(result);
          }}
        />
        <GraphCanvasPanel
          selectedLabel={selectedLabel}
          selectedChapter={focusChapter}
          selectedNodeId={selectedNodeId}
          graph={querySemanticGraphQuery.data}
          hasResults={results.length > 0}
          isLoading={querySemanticGraphQuery.isLoading || searchMutation.isPending}
          error={(querySemanticGraphQuery.error as Error | null) ?? null}
        />
      </section>

      <DetailDrawer result={detailResult} open={Boolean(detailResult)} onOpenChange={(open) => !open && setDetailResult(null)} />
      <AdvancedFilterDrawer
        open={filtersOpen}
        onOpenChange={setFiltersOpen}
        chapter={chapter}
        setChapter={setChapter}
        sourceType={sourceType}
        setSourceType={setSourceType}
        chapterOptions={chapterOptions}
      />
      <HoverPreviewOverlay preview={hoverPreview} />
    </div>
  );
}

export function SearchWorkspace() {
  const { selectedCourseId } = useCourseContext();
  return <SearchWorkspaceContent key={selectedCourseId ?? "unassigned"} selectedCourseId={selectedCourseId} />;
}
