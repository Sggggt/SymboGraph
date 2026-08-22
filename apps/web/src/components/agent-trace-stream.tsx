"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { AgentTraceEventPayload, AgentTraceScores } from "@course-kg/shared";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  Database,
  FileText,
  Gauge,
  ListTree,
  Search,
  XCircle,
} from "lucide-react";
import { useMemo, useState } from "react";

import { MarkdownRenderer } from "@/components/markdown-renderer";
import { traceAuditSummary, traceGroupForNode, traceGroupLabels, traceNodeLabel } from "@/lib/agent-trace";
import { cn } from "@/lib/utils";

interface AgentTraceStreamProps {
  trace: AgentTraceEventPayload[];
  isRunning?: boolean;
  defaultExpanded?: boolean;
  compact?: boolean;
  className?: string;
}

function formatScalar(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join("/");
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/\.?0+$/, "");
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (value === null || value === undefined || value === "") {
    return "无";
  }
  return String(value);
}

const TRACE_UUID_PATTERN = /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/gi;
const TRACE_HASH_PATTERN = /\b[0-9a-f]{32,128}\b/gi;
const TRACE_PATH_PATTERN = /(?:[A-Za-z]:[\\/]|\/(?:app|workspace|home|tmp)\/)[^\s,;]+/g;

export function sanitizeTraceDisplayText(value: unknown): string {
  return String(value ?? "")
    .replace(TRACE_UUID_PATTERN, "内部记录")
    .replace(TRACE_HASH_PATTERN, "内部记录")
    .replace(TRACE_PATH_PATTERN, "内部路径")
    .replace(/\btrace\s*=\s*内部记录/gi, "检索已记录")
    .replace(/\bobservation\s*=\s*内部记录/gi, "证据观察已记录")
    .trim();
}

function metricNumber(...values: Array<number | null | undefined>): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return Math.max(0, Math.floor(value));
    }
  }
  return 0;
}

function eventProgressCount(event: AgentTraceEventPayload): { count: number; noun: string } {
  const scores = event.scores;
  if (scores.audit_kind === "retrieval_stage") {
    if (event.node === "chunk_recall") {
      return { count: metricNumber(scores.chunk_topk_selected, scores.chunk_ids.length), noun: "片段" };
    }
    return {
      count: metricNumber(scores.frontier_pops, scores.stage_queue_count, scores.mid_topk_selected, scores.coarse_entries),
      noun: "节点",
    };
  }
  if (scores.audit_kind === "layered_retrieval") {
    return { count: metricNumber(scores.retrieval_audit?.frontier_pops), noun: "节点" };
  }
  if (scores.audit_kind === "context_restoration") {
    return { count: metricNumber(scores.restored_chunks, scores.bridge_chunks, scores.hit_chunks), noun: "片段" };
  }
  if (scores.audit_kind === "citation_verification") {
    return { count: metricNumber(scores.returned_citation_count), noun: "引用" };
  }
  return { count: event.document_ids.length, noun: "证据" };
}

function statusTone(event: AgentTraceEventPayload): { dot: string; text: string; label: string } {
  if (event.status === "failed" || event.node === "error" || event.error) {
    return { dot: "bg-rose-300 shadow-[0_0_18px_rgba(255,111,145,0.34)]", text: "text-rose-100", label: "失败" };
  }
  if (event.status === "pending") {
    return { dot: "bg-white/36", text: "text-white/48", label: "等待" };
  }
  if (event.node === "citation_verification" || event.node === "repair_executed") {
    return { dot: "bg-amber-200 shadow-[0_0_18px_rgba(255,215,109,0.28)]", text: "text-amber-100", label: "校验" };
  }
  return { dot: "bg-cyan-200 shadow-[0_0_18px_rgba(86,217,255,0.28)]", text: "text-cyan-100", label: "完成" };
}

function renderEventIcon(event: AgentTraceEventPayload) {
  if (event.status === "failed" || event.node === "error") return <XCircle className="size-4" />;
  if (event.status === "pending") return <Activity className="size-4" />;
  if (event.node === "frontier_traversal" || event.node === "layered_retrieval") return <Search className="size-4" />;
  if (event.node === "context_package" || event.node === "structure_context_restoration") return <Database className="size-4" />;
  if (event.node === "citation_verification") return <FileText className="size-4" />;
  return <CheckCircle2 className="size-4" />;
}

function queryFacetRows(scores: AgentTraceScores): Array<{ label: string; value: string }> {
  if (scores.audit_kind !== "query_facets") return [];
  const packet = scores.query_facets;
  if (!packet) return [];
  const aliasValues = packet.facet_groups.flatMap((group) => group.aliases);
  const rows = [
    { label: "问题重点", value: packet.required_facets.join(" / ") },
    { label: "忽略词", value: packet.drop_terms.slice(0, 16).join(" / ") },
    { label: "同义表达", value: aliasValues.slice(0, 16).join(" / ") },
  ];
  return rows.filter((row) => row.value);
}

function eventMediumRows(event: AgentTraceEventPayload): Array<{ label: string; value: string }> {
  const statusLabels: Record<string, string> = {
    completed: "完成",
    pending: "等待",
    running: "进行中",
    failed: "失败",
    cancelled: "已取消",
  };
  const hiddenAuditLabels = new Set(["规划", "检索轨迹", "证据包", "RQ 路径"]);
  const rows: Array<{ label: string; value: unknown }> = [
    { label: "状态", value: statusLabels[event.status] ?? event.status },
    { label: "阶段", value: traceGroupLabels[traceGroupForNode(event.node)] },
    { label: "耗时", value: `${event.duration_ms} ms` },
    { label: "证据片段", value: event.document_ids.length ? `${event.document_ids.length} 个` : "" },
  ];
  for (const item of traceAuditSummary(event.scores).slice(0, 10)) {
    const [label, value] = item.split(/:\s*/, 2);
    if (!hiddenAuditLabels.has(label)) {
      rows.push({ label, value: value ?? item });
    }
  }
  for (const row of queryFacetRows(event.scores)) {
    rows.push({ label: row.label, value: row.value });
  }
  return rows
    .filter((row) => row.value !== undefined && row.value !== null && row.value !== "")
    .map((row) => ({ label: row.label, value: sanitizeTraceDisplayText(formatScalar(row.value)) }));
}

function renderMetricIcon(label: string) {
  if (label === "阶段") return <ListTree className="size-3" />;
  if (label === "耗时") return <Clock3 className="size-3" />;
  if (label === "证据片段") return <FileText className="size-3" />;
  if (label === "问题重点" || label === "忽略词" || label === "同义表达") return <Search className="size-3" />;
  return <Gauge className="size-3" />;
}

function ExplorationTicker({ event, isActive }: { event: AgentTraceEventPayload; isActive: boolean }) {
  const { count, noun } = eventProgressCount(event);
  if (!count) {
    return null;
  }
  const visibleCount = Math.min(count, 28);
  return (
    <div className="mt-3">
      <p className="mb-2 text-[11px] text-white/42">{isActive ? "正在探索" : "已探索"}：{count} 个{noun}</p>
      <div className="flex flex-wrap gap-1.5">
        {Array.from({ length: visibleCount }, (_, index) => (
          <motion.span
            key={index}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(index * 0.018, 0.3) }}
            className="inline-flex h-6 items-center border-l border-cyan-200/28 bg-cyan-200/[0.045] px-2 font-mono text-[10px] text-cyan-100/72"
          >
            {index + 1}
          </motion.span>
        ))}
        {count > visibleCount ? <span className="inline-flex h-6 items-center px-2 font-mono text-[10px] text-white/36">+{count - visibleCount}</span> : null}
      </div>
    </div>
  );
}

function FineTraceDetails({ event }: { event: AgentTraceEventPayload }) {
  const [open, setOpen] = useState(false);
  const inputText = sanitizeTraceDisplayText(event.input_summary) || "无";
  const outputText = sanitizeTraceDisplayText(event.output_summary) || "无";
  return (
    <div className="mt-3 border-t border-white/8 pt-3">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="inline-flex items-center gap-1.5 text-[11px] text-cyan-100/66 transition hover:text-cyan-100"
      >
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        具体信息
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div data-testid="agent-trace-fine-details" className="mt-3 grid gap-3 text-xs text-white/52 lg:grid-cols-2">
              <div className="border-l border-white/10 bg-black/12 p-3">
                <p className="mb-2 text-[11px] uppercase text-white/32">本步输入</p>
                <p className="break-words text-xs leading-6 text-white/58">{inputText}</p>
              </div>
              <div className="border-l border-white/10 bg-black/12 p-3">
                <p className="mb-2 text-[11px] uppercase text-white/32">本步结果</p>
                <p className="break-words text-xs leading-6 text-white/58">{outputText}</p>
              </div>
            </div>
            {event.error ? <p className="mt-3 text-xs text-rose-100/72">{sanitizeTraceDisplayText(event.error)}</p> : null}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function TraceEventItem({
  event,
  index,
  isLatest,
}: {
  event: AgentTraceEventPayload;
  index: number;
  isLatest: boolean;
}) {
  const [open, setOpen] = useState(false);
  const tone = statusTone(event);
  const mediumRows = eventMediumRows(event);
  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className="relative pl-8"
    >
      <span className={cn("absolute left-[7px] top-3 size-2.5", tone.dot, isLatest ? "animate-pulse" : "")} />
      <div className="border-l border-white/10 pb-4 pl-4">
        <button type="button" onClick={() => setOpen((current) => !current)} className="group flex w-full items-start justify-between gap-3 text-left">
            <span className="flex min-w-0 items-start gap-3">
            <span className={cn("mt-0.5 grid size-7 shrink-0 place-items-center border border-white/10 bg-white/[0.025]", tone.text)}>
              {renderEventIcon(event)}
            </span>
            <span className="min-w-0">
              <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="font-mono text-[11px] text-white/38">#{index + 1}</span>
                <span className="text-sm font-medium text-white/82">{traceNodeLabel(event.node)}</span>
                <span className={cn("text-[11px]", tone.text)}>{tone.label}</span>
                {isLatest ? <span className="text-[11px] text-cyan-100/62">实时</span> : null}
              </span>
              {event.output_summary ? <MarkdownRenderer content={sanitizeTraceDisplayText(event.output_summary)} compact className="mt-1 line-clamp-2 text-xs leading-5 text-white/50" /> : null}
            </span>
          </span>
          <span className="mt-1 inline-flex shrink-0 items-center gap-1 text-[11px] text-white/38 transition group-hover:text-white/72">
            步骤信息
            {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          </span>
        </button>
        <AnimatePresence initial={false}>
          {open ? (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="mt-3 border-t border-white/8 pt-3">
                {mediumRows.length ? (
                  <div className="grid gap-x-4 gap-y-2 sm:grid-cols-2 xl:grid-cols-3">
                    {mediumRows.map(({ label, value }) => (
                      <div key={`${label}-${value}`} className="min-w-0 border-l border-white/10 pl-2">
                        <p className="flex items-center gap-1.5 text-[10px] uppercase text-white/32">
                          {renderMetricIcon(label)}
                          {label}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-white/68" title={value}>
                          {value}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : null}
                <ExplorationTicker event={event} isActive={isLatest} />
                <FineTraceDetails event={event} />
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </motion.li>
  );
}

export function AgentTraceStream({ trace, isRunning = false, defaultExpanded = false, compact = false, className }: AgentTraceStreamProps) {
  const [expanded, setExpanded] = useState(defaultExpanded || isRunning);
  const events = useMemo<AgentTraceEventPayload[]>(() => trace, [trace]);
  const latest = events.at(-1);

  if (!events.length && !isRunning) {
    return null;
  }

  return (
    <section data-testid="agent-trace-stream" className={cn("border-l border-cyan-200/18 pl-4 text-white", compact ? "py-2" : "py-4", className)}>
      <button type="button" onClick={() => setExpanded((current) => !current)} className="group flex w-full items-center justify-between gap-4 text-left">
        <span className="min-w-0">
          <span className="flex flex-wrap items-center gap-2">
            {isRunning ? <span className="tech-dot" /> : <CheckCircle2 className="size-4 text-cyan-100/62" />}
            <span className="text-sm font-semibold text-white/82">流式轨迹</span>
            <span className="text-[11px] text-white/38">{events.length} 个步骤</span>
          </span>
          <span className="mt-1 block truncate text-xs text-white/42">
            {latest ? `${isRunning ? "正在执行" : "最新事件"}：${traceNodeLabel(latest.node)}` : "等待事件"}
          </span>
        </span>
        <span className="inline-flex shrink-0 items-center gap-1 text-xs text-cyan-100/62 transition group-hover:text-cyan-100">
          {expanded ? "收起总览" : "总展开"}
          {expanded ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {expanded ? (
          <motion.ol
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22 }}
            className={cn("mt-4 overflow-hidden", compact ? "max-h-[34rem] overflow-y-auto pr-2 custom-scrollbar" : "")}
          >
            {events.length ? (
              events.map((event, index) => (
                <TraceEventItem
                  key={event.id ?? `${event.node}-${index}-${event.created_at ?? "pending"}`}
                  event={event}
                  index={index}
                  isLatest={isRunning && index === events.length - 1}
                />
              ))
            ) : (
              <motion.li
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="border-l border-white/10 py-3 pl-4 text-xs text-white/44"
              >
                正在等待后端推送第一个轨迹事件...
              </motion.li>
            )}
          </motion.ol>
        ) : null}
      </AnimatePresence>

      {latest?.error ? (
        <div className="mt-3 flex items-start gap-2 border-l border-rose-300/35 bg-rose-300/[0.045] px-3 py-2 text-xs text-rose-100/78">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>{sanitizeTraceDisplayText(latest.error)}</span>
        </div>
      ) : null}
    </section>
  );
}
