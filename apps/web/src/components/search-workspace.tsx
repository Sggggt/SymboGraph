"use client";

import { createPortal } from "react-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ContextPackageResponse, GraphResponse, ModelAudit, RetrievalTraceStepsResponse, SearchResult, SourceType } from "@course-kg/shared";
import { AnimatePresence, motion } from "framer-motion";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowUpRight,
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
import { GrayZoneAuditDetails, GrayZoneAuditFailure, inspectGrayZoneTrace, QueryFacetPosteriorAudit } from "@/components/gray-zone-audit";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { NetworkCanvas } from "@/components/network-canvas";
import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { fetchDashboard, fetchGraph, fetchRetrievalTraceSteps, searchKnowledge } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

const sourceOptions: SourceType[] = ["pdf", "notebook", "markdown", "text", "image", "docx", "pptx"];
const emptySearchResults: SearchResult[] = [];

type SearchState = {
  results: SearchResult[];
  degraded_mode: boolean;
  model_audit?: ModelAudit;
  retrieval_trace_id?: string | null;
  context_package_id?: string | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function addStringId(value: unknown, ids: Set<string>) {
  if (typeof value === "string" && value.trim()) {
    ids.add(value);
  }
}

function collectStringIds(value: unknown, ids: Set<string>) {
  if (!Array.isArray(value)) {
    return;
  }
  for (const item of value) {
    addStringId(item, ids);
  }
}

export function exploredMidNodeIdsFromTrace(trace: RetrievalTraceStepsResponse | undefined): Set<string> {
  const ids = new Set<string>();
  if (!trace) {
    return ids;
  }

  collectStringIds(trace.stage_queues?.mid?.selected_ids, ids);
  collectStringIds(trace.stage_queues?.mid?.accepted_ids, ids);
  collectStringIds(trace.topk_selection?.mid?.selected_ids, ids);

  for (const entry of trace.entry_nodes ?? []) {
    if (entry.layer === "mid") addStringId(entry.node_id, ids);
  }
  for (const pathLabel of trace.path_labels ?? []) {
    if (pathLabel.layer !== "mid") continue;
    addStringId(pathLabel.node_id, ids);
    collectStringIds(pathLabel.path, ids);
  }

  for (const step of trace.steps) {
    if (step.layer !== "mid") {
      continue;
    }
    collectStringIds(step.selected_topk_ids, ids);
    collectStringIds(step.input.mid_entry_ids, ids);
    collectStringIds(step.output.accepted_node_ids, ids);
    addStringId(step.popped_frontier_state.node_id, ids);
    collectStringIds(step.popped_frontier_state.path, ids);
  }

  return ids;
}

export function toExploredMidConceptGraph(graph: GraphResponse | undefined, trace: RetrievalTraceStepsResponse | undefined): GraphResponse | undefined {
  if (!graph) {
    return undefined;
  }
  const exploredIds = exploredMidNodeIdsFromTrace(trace);
  const nodes = exploredIds.size > 0 ? graph.nodes.filter((node) => exploredIds.has(node.id)) : [];
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = graph.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));

  return {
    ...graph,
    graph_type: "mid-concepts",
    nodes,
    edges,
    sampled_counts: {
      ...graph.sampled_counts,
      nodes: nodes.length,
      edges: edges.length,
    },
    node_counts: {
      ...graph.node_counts,
      sampled: nodes.length,
    },
    edge_counts: {
      ...graph.edge_counts,
      sampled: edges.length,
    },
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

export function pathEdgeSummary(result: SearchResult): string {
  const traversal = resultTraversal(result);
  const edgeIds = traversalTextList(traversal.path_edge_ids);
  if (edgeIds.length) {
    return edgeIds.length <= 3 ? edgeIds.join(" / ") : `${edgeIds.length} 条：${edgeIds.slice(0, 3).join(" / ")} ...`;
  }
  const roles = traversalTextList(traversal.evidence_roles);
  if (roles.length) {
    const visibleRoles = roles.slice(0, 3).join(" / ");
    return roles.length > 3 ? `入口种子：${visibleRoles} ...` : `入口种子：${visibleRoles}`;
  }
  return "无";
}

function retrievalGranularityLabel(value: unknown): string {
  if (value === "mid") return "普通模式";
  if (value === "coarse") return "摘要模式";
  return "无";
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

function resultSourceType(result: SearchResult | undefined) {
  if (!result) return "未知";
  return sourceTypeLabel(result.source_type ?? (result.metadata.source_type as string | undefined));
}

export function resultDisplayTitle(result: SearchResult): string {
  const candidate = String(
    result.document_title ?? result.citations[0]?.document_title ?? "",
  ).trim();
  if (
    candidate &&
    !/^[0-9a-f]{64}$/i.test(candidate) &&
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(candidate)
  ) {
    return candidate;
  }
  const heading = String(result.snippet ?? result.content ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => /^#{1,6}\s+\S/.test(line));
  if (!heading) return "来源名称缺失";
  const naturalHeading = heading
    .replace(/^#{1,6}\s+/, "")
    .split(/\s+#\s+|。|\s+(?=内容涵盖|消息质控采用)/, 1)[0]
    .trim();
  return naturalHeading.slice(0, 60) || "来源名称缺失";
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
          本地资料检索
        </div>
        <div className="space-y-3">
          <h2 className="glow-text text-4xl font-semibold text-white lg:text-6xl">检索本地资料</h2>
          <p className="mx-auto max-w-2xl text-sm leading-7 text-cyan-50/58 lg:text-base">
            输入问题或引用线索，系统会返回相关片段、来源和必要的上下文。
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
        {degradedMode ? "资料索引暂不可用" : "资料索引已就绪"}
      </div>
    </div>
  );
}

export function semanticEntryAuditLabels(audit?: ModelAudit) {
  return {
    source: audit?.semantic_entry_query_selection_source === "validated_required_facet" ? "已去除交互指令" : "原始问题",
    grayAuthority: audit?.semantic_entry_query_gray_zone_decision_authority === false ? "无" : "未审计",
  } as const;
}

function traceText(value: unknown, fallback = "无"): string {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function traceIds(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function traceJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function TraceIdList({ values, empty = "无" }: { values: unknown; empty?: string }) {
  const ids = traceIds(values);
  if (!ids.length) return <span className="text-white/38">{empty}</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {ids.map((id, index) => (
        <code key={`${id}:${index}`} className="rounded-md border border-white/8 bg-black/20 px-1.5 py-0.5 text-[11px] text-cyan-50/68">
          {id}
        </code>
      ))}
    </div>
  );
}

function TraceJson({ value }: { value: unknown }) {
  return <pre className="custom-scrollbar max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-xl border border-white/8 bg-black/20 p-3 text-[11px] leading-5 text-cyan-50/62">{traceJson(value)}</pre>;
}

function TraceSection({ title, count, children, open = true }: { title: string; count?: number; children: React.ReactNode; open?: boolean }) {
  return (
    <details open={open} className="rounded-2xl border border-white/8 bg-white/[0.025] p-4">
      <summary className="cursor-pointer select-none text-sm font-semibold text-white/82">
        {title}
        {count === undefined ? null : <span className="ml-2 text-xs font-normal text-cyan-100/48">{count}</span>}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}

function CandidatePoolCards({ trace }: { trace: RetrievalTraceStepsResponse }) {
  const pools = asRecord(trace.candidate_pools) ?? {};
  const rows: Array<{ key: string; pool: Record<string, unknown> }> = [];
  for (const [key, value] of Object.entries(pools)) {
    if (Array.isArray(value)) {
      value.forEach((item, index) => {
        const pool = asRecord(item);
        if (pool) rows.push({ key: `${key}[${index}]`, pool });
      });
      continue;
    }
    const pool = asRecord(value);
    if (pool && key !== "candidate_dedupe_budget") rows.push({ key, pool });
  }
  return (
    <div className="space-y-3">
      {rows.length ? (
        rows.map(({ key, pool }) => (
          <div key={key} className="rounded-xl border border-white/7 bg-black/15 p-3 text-xs text-white/58">
            <div className="flex flex-wrap items-center gap-2 text-white/76">
              <span className="font-medium">{key}</span>
              <span>父层 {traceText(pool.parent_layer)}</span>
              <code>{traceText(pool.parent_node_id)}</code>
              <span>候选 {traceText(pool.candidate_count ?? traceIds(pool.candidate_ids).length)}</span>
              <span>Top-K {traceText(pool.top_k)}</span>
              <span>停止 {traceText(pool.stop_reason)}</span>
            </div>
            <div className="mt-2 grid gap-2 lg:grid-cols-2">
              <div><p className="mb-1 text-white/40">候选合并</p><TraceIdList values={pool.candidate_ids} /></div>
              <div><p className="mb-1 text-white/40">入选</p><TraceIdList values={pool.selected_ids} /></div>
            </div>
            <div className="mt-2">
              <p className="mb-1 text-white/40">RQ seed ranking / scores</p>
              <TraceJson value={{
                ranking_protocol_version: pool.ranking_protocol_version,
                ranking_protocol_hash: pool.ranking_protocol_hash,
                candidate_scores: pool.candidate_scores,
              }} />
            </div>
            {Object.keys(asRecord(pool.rq_seed_cards) ?? {}).length ? (
              <div className="mt-2">
                <p className="mb-1 text-white/40">Query → RQ seed cards</p>
                <TraceJson value={pool.rq_seed_cards} />
              </div>
            ) : null}
            {Object.keys(asRecord(pool.rq_chunk_seed_cards) ?? {}).length ? (
              <div className="mt-2">
                <p className="mb-1 text-white/40">RQ → chunk seed cards</p>
                <TraceJson value={pool.rq_chunk_seed_cards} />
              </div>
            ) : null}
            {pool.per_parent_budget_status ? <div className="mt-2"><p className="mb-1 text-white/40">逐父预算</p><TraceJson value={pool.per_parent_budget_status} /></div> : null}
            {pool.candidate_dedupe_budget_audit ? <div className="mt-2"><p className="mb-1 text-white/40">候选去重预算</p><TraceJson value={pool.candidate_dedupe_budget_audit} /></div> : null}
          </div>
        ))
      ) : (
        <p className="text-xs text-white/38">没有候选池记录。</p>
      )}
      {pools.candidate_dedupe_budget ? <div><p className="mb-1 text-xs text-white/40">全局候选池去重 hard interrupt</p><TraceJson value={pools.candidate_dedupe_budget} /></div> : null}
    </div>
  );
}

export function SearchTraceDetails({
  traceId,
  contextPackageId,
  trace,
  contextPackage,
  isLoading,
  error,
}: {
  traceId: string | null;
  contextPackageId: string | null;
  trace?: RetrievalTraceStepsResponse;
  contextPackage?: ContextPackageResponse;
  isLoading: boolean;
  error: Error | null;
}) {
  const [rawPayloadOpen, setRawPayloadOpen] = useState(false);
  if (!traceId) return null;
  if (isLoading && !trace) return <LoadingBlock rows={3} />;
  if (error && !trace) {
    const status = (error as Error & { status?: number }).status;
    return status === 409
      ? <GrayZoneAuditFailure title="Gray-zone 持久化轨迹校验冲突" message={`HTTP 409 · ${error.message}`} />
      : <ErrorBlock message={error.message} />;
  }
  if (!trace) return null;

  const stageQueues = Object.entries(trace.stage_queues ?? {});
  const topKSelections = Object.entries(trace.topk_selection ?? {});
  const thresholdHits = trace.path_distance_threshold_hits ?? [];
  const grayAuditInspection = inspectGrayZoneTrace(trace);
  const dedupeKeys = contextPackage?.dedupe_keys ?? [];
  const packageContexts = contextPackage?.contexts ?? [];

  return (
    <section data-testid="search-trace-details" className="rounded-[1.7rem] border border-white/10 bg-white/[0.025] p-5 shadow-[0_18px_70px_rgba(0,0,0,0.18)]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-white">完整分层检索轨迹与证据包</p>
          <p className="mt-1 text-xs leading-5 text-white/45">可回放入口、frontier、逐父下钻、候选合并、剪枝、确定性灰区规则、收敛和结构恢复。</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="kg-micro-chip rounded-full px-2.5 py-1">Trace {trace.trace_id}</span>
          <span className="kg-micro-chip rounded-full px-2.5 py-1">Package {contextPackage?.id ?? contextPackageId ?? "无"}</span>
          <span className="kg-micro-chip rounded-full px-2.5 py-1">{retrievalGranularityLabel(trace.retrieval_granularity)}</span>
          <span className={cn("rounded-full border px-2.5 py-1", grayAuditInspection.ok ? "border-emerald-300/20 bg-emerald-300/[0.07] text-emerald-100/78" : "border-red-300/30 bg-red-300/10 text-red-100")}>
            {grayAuditInspection.ok ? `Gray 模型调用 ${grayAuditInspection.modelCallCount}` : "Gray 审计失败"}
          </span>
        </div>
      </div>

      <div className="mt-5 grid gap-3 xl:grid-cols-2">
        <TraceSection title="查询面与入口节点" count={trace.entry_nodes?.length ?? 0}>
          <div className="mb-3 grid gap-2 text-xs text-white/58 sm:grid-cols-3">
            <span>模式：{traceText(trace.retrieval_mode)}</span>
            <span>粒度：{traceText(trace.retrieval_granularity)}</span>
            <span>Gray 协议：{traceText(trace.gray_zone_protocol)}</span>
          </div>
          <TraceJson value={trace.query_facets} />
          <div className="mt-3">
            <QueryFacetPosteriorAudit calibration={trace.trace_diagnostics.query_facet_posterior_calibration} />
          </div>
          {trace.trace_diagnostics.query_rq_seed_audit ? (
            <div className="mt-3">
              <p className="mb-1 text-xs text-white/40">Query → RQ seed authority audit</p>
              <TraceJson value={trace.trace_diagnostics.query_rq_seed_audit} />
            </div>
          ) : null}
          <div className="mt-3 space-y-2">
            {(trace.entry_nodes ?? []).map((entry, index) => (
              <div key={`${entry.layer}:${entry.node_id}:${index}`} className="rounded-xl border border-white/7 bg-black/15 p-3 text-xs text-white/58">
                <div className="flex flex-wrap gap-2"><span className="text-cyan-100/74">{entry.layer}</span><code>{entry.node_id}</code><span>强度 {traceText(entry.entry_strength)}</span></div>
                <div className="mt-2"><TraceIdList values={entry.roles} empty="无角色" /></div>
                {entry.rq_prefix_id || Object.keys(entry.metadata ?? {}).length ? <div className="mt-2"><TraceJson value={{ rq_prefix_id: entry.rq_prefix_id, metadata: entry.metadata }} /></div> : null}
              </div>
            ))}
          </div>
        </TraceSection>

        <TraceSection title="Stage 队列" count={stageQueues.length}>
          <div className="space-y-2">
            {stageQueues.map(([layer, queue]) => (
              <div key={layer} className="rounded-xl border border-white/7 bg-black/15 p-3 text-xs text-white/58">
                <div className="flex flex-wrap gap-3 text-white/74"><span className="font-medium">{layer}</span><span>入口上限 {traceText(queue.initial_top_k)}</span><span>输出 Top-K {traceText(queue.top_k)}</span><span>Frontier pops {traceText(queue.frontier_pop_count)}</span><span>原因 {traceText(queue.reason ?? queue.skipped_by_granularity)}</span></div>
                <div className="mt-2 grid gap-2 lg:grid-cols-3">
                  <div><p className="mb-1 text-white/40">入口</p><TraceIdList values={queue.entry_ids} /></div>
                  <div><p className="mb-1 text-white/40">选择</p><TraceIdList values={queue.selected_ids} /></div>
                  <div><p className="mb-1 text-white/40">接受</p><TraceIdList values={queue.accepted_ids} /></div>
                </div>
              </div>
            ))}
          </div>
        </TraceSection>

        <TraceSection title="Frontier expansion timeline" count={trace.frontier?.length ?? 0}>
          <ol className="space-y-2">
            {(trace.frontier ?? []).map((snapshot, index) => {
              const popped = asRecord(snapshot.popped) ?? {};
              return (
                <li key={`${traceText(popped.node_id)}:${index}`} className="rounded-xl border border-white/7 bg-black/15 p-3 text-xs text-white/58">
                  <div className="flex flex-wrap gap-3 text-white/74"><span>#{index + 1}</span><span>{traceText(snapshot.layer ?? popped.layer)}</span><code>{traceText(popped.node_id)}</code><span>队列剩余 {snapshot.queue_size_after_pop}</span><span>深度 {traceText(popped.depth)}</span><span>距离 {traceText(popped.distance_so_far)}</span><span>奖励 {traceText(popped.reward_so_far)}</span><span>分区 {traceText(popped.distance_zone)}</span></div>
                  <div className="mt-2"><p className="mb-1 text-white/40">路径</p><TraceIdList values={popped.path} /></div>
                  <div className="mt-2"><p className="mb-1 text-white/40">展开边 / 距离 / 类型</p><TraceJson value={{ edge_ids: popped.path_edge_ids, distances: popped.path_edge_distances, edge_types: popped.path_edge_types, support_refs: popped.support_refs, queue_key: snapshot.key }} /></div>
                </li>
              );
            })}
          </ol>
        </TraceSection>

        <TraceSection title="逐父候选池、合并与去重"><CandidatePoolCards trace={trace} /></TraceSection>

        <TraceSection title="Top-K selections" count={topKSelections.length}>
          <div className="space-y-2">
            {topKSelections.map(([layer, selection]) => (
              <div key={layer} className="rounded-xl border border-white/7 bg-black/15 p-3 text-xs text-white/58">
                <div className="flex flex-wrap gap-3 text-white/74"><span>{layer}</span><span>候选 {selection.candidate_count ?? 0}</span><span>Top-K {traceText(selection.top_k)}</span><span>入口模式 {traceText(selection.entry_mode)}</span><span>停止 {traceText(selection.stop_reason)}</span></div>
                <div className="mt-2"><TraceIdList values={selection.selected_ids} /></div>
              </div>
            ))}
          </div>
        </TraceSection>

        <TraceSection title="持久化检索步骤" count={trace.steps.length}>
          <ol className="space-y-2">
            {trace.steps.map((step) => (
              <li key={step.id} className="rounded-xl border border-white/7 bg-black/15 p-3 text-xs text-white/58">
                <div className="flex flex-wrap gap-3 text-white/74"><span>#{step.step_index}</span><span>{step.layer}</span><span>{step.action_type}</span><span>父 {traceText(step.parent_layer)} / {traceText(step.parent_node_id)}</span><span>支配剪枝 {step.dominance_pruned_count ?? 0}</span><span>cycle reward {traceText(step.cycle_distance_reward)}</span><span>停止 {traceText(step.stop_reason)}</span></div>
                <div className="mt-2 grid gap-2 lg:grid-cols-3">
                  <div><p className="mb-1 text-white/40">候选</p><TraceIdList values={step.candidate_pool_ids} /></div>
                  <div><p className="mb-1 text-white/40">Top-K</p><TraceIdList values={step.selected_topk_ids} /></div>
                  <div><p className="mb-1 text-white/40">展开边</p><TraceIdList values={step.expanded_edge_ids} /></div>
                </div>
                <details className="mt-2"><summary className="cursor-pointer text-white/48">输入 / 输出 / 逐父预算 / 诊断</summary><div className="mt-2"><TraceJson value={{ input: step.input, output: step.output, per_parent_budget_status: step.per_parent_budget_status, popped_frontier_state: step.popped_frontier_state, diagnostics: step.diagnostics }} /></div></details>
              </li>
            ))}
          </ol>
        </TraceSection>

        <TraceSection
          title="Multi-path contribution union"
          count={(trace.node_contributions?.length ?? 0) + (contextPackage?.reached_by_paths.length ?? 0)}
        >
          <div className="grid gap-3 lg:grid-cols-2">
            <div><p className="mb-1 text-xs text-white/40">Trace node contribution summaries</p><TraceJson value={trace.node_contributions} /></div>
            <div><p className="mb-1 text-xs text-white/40">Context reached paths and why-selected union</p><TraceJson value={{ reached_by_paths: contextPackage?.reached_by_paths ?? [], node_contributions: contextPackage?.node_contributions ?? [], why_selected: contextPackage?.why_selected ?? {} }} /></div>
          </div>
        </TraceSection>

        <GrayZoneAuditDetails trace={trace} />

        <TraceSection title="Path distance threshold hits" count={thresholdHits.length}>
          <div className="space-y-2">
            {thresholdHits.map((decision, index) => (
              <div key={`${decision.edge_id ?? "threshold"}:${index}`} className="rounded-xl border border-red-300/12 bg-red-300/[0.035] p-3 text-xs text-white/58">
                <div className="flex flex-wrap gap-3 text-white/74"><span>{traceText(decision.layer)}</span><code>{traceText(decision.edge_id)}</code><span>分区 {traceText(decision.distance_zone)}</span><span>距离 {traceText(decision.path_distance)}</span><span>结果 {decision.decision}</span></div>
              </div>
            ))}
          </div>
        </TraceSection>

        <TraceSection title="Drilldown path 与 convergence" count={trace.path_labels?.length ?? 0}>
          <div className="grid gap-3">
            <div><p className="mb-1 text-xs text-white/40">coarse → mid → chunk 路径标签</p><TraceJson value={trace.path_labels} /></div>
            <div><p className="mb-1 text-xs text-white/40">收敛、支配剪枝、cycle distance reward 与 hard interrupt</p><TraceJson value={trace.convergence} /></div>
          </div>
        </TraceSection>

        <TraceSection title="Context Package 去重与结构恢复" count={packageContexts.length}>
          {contextPackage ? (
            <div className="space-y-3 text-xs text-white/58">
              <div className="flex flex-wrap gap-3 text-white/74"><span>Token {contextPackage.token_count ?? 0} / {contextPackage.token_budget}</span><span>命中 {(contextPackage.hit_chunk_ids ?? []).length}</span><span>恢复 {(contextPackage.restored_chunk_ids ?? []).length}</span><span>桥接 {(contextPackage.bridge_chunk_ids ?? []).length}</span><span>去重键 {dedupeKeys.length} / 唯一 {new Set(dedupeKeys).size}</span><span>cycle convergence {traceText(contextPackage.cycle_convergence_score)}</span></div>
              <div className="grid gap-2 lg:grid-cols-2"><div><p className="mb-1 text-white/40">Hit / restored / bridge chunks</p><TraceJson value={{ hit: contextPackage.hit_chunk_ids, restored: contextPackage.restored_chunk_ids, bridge: contextPackage.bridge_chunk_ids }} /></div><div><p className="mb-1 text-white/40">结构闭包与 graph path</p><TraceJson value={{ parent_structure_node_ids: contextPackage.parent_structure_node_ids, graph_path_ids: contextPackage.graph_path_ids, graph_expansion_paths: contextPackage.graph_expansion_paths }} /></div></div>
              <div><p className="mb-1 text-white/40">why_selected / covered facets / dedupe keys</p><TraceJson value={{ why_selected: contextPackage.why_selected, covered_facets: contextPackage.covered_facets, dedupe_keys: dedupeKeys }} /></div>
              <div><p className="mb-1 text-white/40">citation-ready raw spans</p><TraceJson value={contextPackage.citation_spans} /></div>
              <div><p className="mb-1 text-white/40">Public package hash proof</p><TraceJson value={{ package_hash: contextPackage.package_hash, package_hash_card: contextPackage.package_hash_card }} /></div>
              <div className="space-y-2">
                {packageContexts.map((context, index) => {
                  const metadata = asRecord(context.metadata) ?? {};
                  return <div key={`${traceText(context.chunk_id)}:${index}`} className="rounded-xl border border-white/7 bg-black/15 p-3"><div className="flex flex-wrap gap-2 text-white/74"><code>{traceText(context.chunk_id)}</code><span>{traceText(context.document_title)}</span><span>角色 {traceText(metadata.role)}</span><span>dedupe {traceText(metadata.dedupe_key)}</span></div><p className="mt-2 whitespace-pre-wrap leading-5 text-white/55">{traceText(context.snippet ?? context.content)}</p><details className="mt-2"><summary className="cursor-pointer text-white/42">结构路径、span 与选择理由</summary><div className="mt-2"><TraceJson value={metadata} /></div></details></div>;
                })}
              </div>
              <TraceJson value={contextPackage.diagnostics} />
            </div>
          ) : isLoading && contextPackageId ? <LoadingBlock rows={2} /> : error ? <ErrorBlock message={error.message} /> : <p className="text-xs text-white/38">当前 trace 没有关联的 context package。</p>}
        </TraceSection>
      </div>

      <details
        className="rounded-2xl border border-white/8 bg-white/[0.025] p-4"
        onToggle={(event) => setRawPayloadOpen(event.currentTarget.open)}
      >
        <summary className="cursor-pointer select-none text-sm font-semibold text-white/82">原始保真 payload</summary>
        <div className="mt-3">
          {rawPayloadOpen ? (
            <div className="grid gap-3 xl:grid-cols-2"><TraceJson value={trace} /><TraceJson value={contextPackage ?? { context_package_id: contextPackageId }} /></div>
          ) : (
            <p className="text-xs text-white/38">展开后渲染完整 API payload。</p>
          )}
        </div>
      </details>
    </section>
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

export function ResultRow({
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
  const traversal = resultTraversal(result);
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
            <span className="text-sm font-medium text-white">{resultDisplayTitle(result)}</span>
            <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultSourceType(result)}</span>
          </div>
          <MarkdownRenderer content={result.snippet} compact className="mt-3 line-clamp-2 text-white/62" />
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">相关证据</span>
          <ArrowUpRight className="text-white/32 transition group-hover:text-cyan-100" />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {coveredFacets.length ? (
          <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">相关主题：{coveredFacets.join(" / ")}</span>
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
              发起检索后，这里会展示相关片段、来源和简要主题关系。
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
                <span className="kg-micro-chip rounded-full px-2 py-1 text-[11px]">{resultSourceType(preview.result)}</span>
              </div>
            </div>
            <p className="mt-3 text-sm font-medium text-white">
              {resultDisplayTitle(preview.result)}
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
  graph,
  isLoading,
  error,
  hasTrace,
}: {
  graph: GraphResponse | undefined;
  isLoading: boolean;
  error: Error | null;
  hasTrace: boolean;
}) {
  const emptyMessage = hasTrace ? "本次检索没有返回可展示的相关主题。" : "发起检索后，这里会显示与结果相关的主题。";

  return (
    <section className="kg-glass-line kg-scroll-shell relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-[2rem] bg-[rgba(4,8,22,0.28)]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_20%,rgba(86,217,255,0.11),transparent_38%),linear-gradient(rgba(120,180,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(120,180,255,0.035)_1px,transparent_1px)] bg-[size:auto,42px_42px,42px_42px]" />
      <div className="relative z-10 flex items-center justify-between gap-3 px-5 py-4">
        <div>
          <p className="section-kicker">知识画布</p>
          <h3 className="mt-1 text-xl font-semibold text-white">中层概念图谱</h3>
          <p className="mt-1 text-sm text-white/42">显示与本次结果相关的主题</p>
        </div>
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          <GitBranch data-icon="inline-start" />
          相关主题
        </span>
      </div>
      <div className="relative z-10 flex min-h-0 flex-1 px-2 pb-2">
        {isLoading ? (
          <div className="kg-shimmer mx-3 grid h-full min-h-0 w-full flex-1 place-items-center rounded-[1.5rem] border border-white/7 bg-white/[0.025] text-sm text-white/54">
            正在加载相关主题...
          </div>
        ) : error ? (
          <div className="mx-3 flex min-h-0 w-full flex-1 items-stretch">
            <ErrorBlock message={error.message} />
          </div>
        ) : graph && graph.nodes.length > 0 ? (
          <div className="mx-3 flex h-full min-h-0 w-full flex-1">
            <NetworkCanvas graph={graph} height="100%" selectedNodeId={null} />
          </div>
        ) : (
          <div className="mx-3 grid h-full min-h-0 w-full flex-1 place-items-center rounded-[1.5rem] border border-white/7 bg-white/[0.025] px-6 text-center text-sm leading-7 text-white/54">
            {emptyMessage}
          </div>
        )}
      </div>
    </section>
  );
}

function DetailDrawer({ result, open, onOpenChange }: { result: SearchResult | null; open: boolean; onOpenChange: (open: boolean) => void }) {
  const traversal = result ? resultTraversal(result) : {};
  const coveredFacets = traversalTextList(traversal.covered_facets).slice(0, 6);
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-xl">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>结果详情</SheetTitle>
          <SheetDescription>命中片段、来源和选择原因。</SheetDescription>
        </SheetHeader>
        {result ? (
          <div className="custom-scrollbar flex-1 overflow-y-auto p-6">
            <div className="flex flex-wrap gap-2">
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">{resultSourceType(result)}</span>
              <span className="kg-micro-chip rounded-full px-3 py-1.5 text-xs">相关证据</span>
            </div>
            <h3 className="mt-5 text-2xl font-semibold text-white">{resultDisplayTitle(result)}</h3>
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
            {coveredFacets.length ? (
              <div className="mt-5 rounded-2xl border border-white/8 bg-white/[0.03] p-5">
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">为什么命中</p>
                {coveredFacets.length ? <p className="mt-4 text-sm leading-7 text-white/62">相关主题：{coveredFacets.join("、")}</p> : null}
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
  const [selectedChunkId, setSelectedChunkId] = useLocalStorage<string | null>(`search.selectedChunkId.${storageScope}`, null);
  const [searchResults, setSearchResults] = useLocalStorage<SearchState | null>(`search.results.${storageScope}`, null);
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null);
  const [detailResult, setDetailResult] = useState<SearchResult | null>(null);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const hoverPreviewTimerRef = useRef<HoverPreviewTimer>(null);

  const searchMutation = useMutation({
    mutationFn: (searchText: string) =>
      searchKnowledge({
        knowledge_base_id: selectedKnowledgeBaseId,
        query: searchText,
        retrieval_granularity: "mid",
        filters: {
          partition: partition || undefined,
          source_type: (sourceType || undefined) as never,
        },
    }),
    onSuccess: (data, searchText) => {
      setSearchResults({
        results: data.results,
        degraded_mode: Boolean(data.degraded_mode),
        model_audit: data.model_audit,
        retrieval_trace_id: data.retrieval_trace_id ?? data.model_audit.retrieval_trace_id ?? null,
        context_package_id: data.context_package_id ?? data.model_audit.context_package_id ?? null,
      });
      setSearchHistory((current) => [searchText, ...current.filter((item) => item !== searchText)].slice(0, 50));
      const firstResult = data.results[0];
      setSelectedChunkId(firstResult?.chunk_id ?? null);
    },
    onError: () => {
      setSearchResults({ results: [], degraded_mode: false });
      setSelectedChunkId(null);
    },
  });

  const partitionOptions = useMemo(() => dashboardQuery.data?.tree.map((node) => node.title).filter((title): title is string => Boolean(title)) ?? [], [dashboardQuery.data]);
  const results = searchResults?.results ?? emptySearchResults;
  const retrievalTraceId = searchResults?.retrieval_trace_id ?? searchResults?.model_audit?.retrieval_trace_id ?? null;
  const midConceptGraphQuery = useQuery({
    queryKey: ["search-mid-concept-graph", selectedKnowledgeBaseId],
    queryFn: () => fetchGraph(selectedKnowledgeBaseId, "mid-concepts", "overview"),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const retrievalTraceQuery = useQuery({
    queryKey: ["search-retrieval-trace-steps", retrievalTraceId],
    queryFn: () => fetchRetrievalTraceSteps(retrievalTraceId as string),
    enabled: Boolean(retrievalTraceId),
  });
  const midConceptGraph = useMemo(() => toExploredMidConceptGraph(midConceptGraphQuery.data, retrievalTraceQuery.data), [midConceptGraphQuery.data, retrievalTraceQuery.data]);

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
            setDetailResult(result);
          }}
        />
        <GraphCanvasPanel
          graph={midConceptGraph}
          isLoading={searchMutation.isPending || midConceptGraphQuery.isLoading || retrievalTraceQuery.isLoading}
          error={(midConceptGraphQuery.error as Error | null) ?? (retrievalTraceQuery.error as Error | null) ?? null}
          hasTrace={Boolean(retrievalTraceId)}
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
