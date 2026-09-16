"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { AgentResponse, AgentTraceEventPayload, AnswerModelAudit, Citation, ConversationStatePayload, DirectAnswerMode, ModelSettingsResponse, ModelSettingsUpdate, SessionMessage, SessionMessagesResponse, SessionSummary, TaskStatusResponse } from "@course-kg/shared";
import { motion } from "framer-motion";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  BrainCircuit,
  ChevronRight,
  CircleDot,
  FileText,
  History,
  Info,
  Layers3,
  Loader2,
  Plus,
  RotateCcw,
  Save,
  Send,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react";

import { AgentTraceStream } from "@/components/agent-trace-stream";
import { CitationCard } from "@/components/citation-card";
import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { cancelAgentRun, deleteSession, fetchModelSettings, fetchSessionMessages, fetchSessions, fetchTaskStatus, streamAnswer, updateModelSettings } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  run_id?: string | null;
  route?: string | null;
  direct_answer_mode?: DirectAnswerMode | null;
  citations?: Citation[];
  trace?: AgentTraceEventPayload[];
  retrieval_trace_id?: string | null;
  context_package_id?: string | null;
  citation_replay_status?: "not_present" | "valid" | "unavailable";
  citation_replay_reason?: "persisted_citation_contract_mismatch" | null;
};

type ActiveStreamState = {
  runId?: string | null;
  sessionId?: string | null;
  question: string;
  startedAt: string;
};

type AgentSettingsForm = {
  context_package_token_budget: string;
  retrieval_result_top_k_default: string;
  retrieval_v1_dense_candidate_budget: string;
  retrieval_v1_rq_candidate_budget: string;
  retrieval_v1_bm25_candidate_budget: string;
  retrieval_v1_root_entry_budget: string;
  retrieval_v1_per_parent_entry_budget: string;
  retrieval_v1_layer_entry_budget: string;
  retrieval_v1_max_depth: string;
  retrieval_v1_restore_per_hit: string;
  agent_path_distance_green_threshold: string;
  agent_path_distance_gray_threshold: string;
  agent_path_distance_hard_threshold: string;
  agent_answer_unit_limit: string;
  agent_history_summary_max_chars: string;
};

type AgentNumberSettingKey = keyof AgentSettingsForm;

type AgentNumberField = {
  key: AgentNumberSettingKey;
  label: string;
  min: number;
  max: number;
  step?: number;
};

const userCancelledMessage = "已取消当前对话";
const missingSessionMessage = "该会话已不存在，历史列表已刷新。";

function responseStatus(value: unknown): number | null {
  const status = (value as { status?: unknown } | null)?.status;
  return typeof status === "number" ? status : null;
}

export function legacyQaPayloadStorageKeys(scope: string): string[] {
  return [
    "turns",
    "draftAnswer",
    "citations",
    "trace",
    "latestRun",
    "conversationState",
  ].map((field) => `qa.${field}.${scope}`);
}

export function productQaErrorMessage(value: unknown): string {
  const message = value instanceof Error ? value.message : String(value ?? "");
  if (message === "cancelled_by_user" || message === userCancelledMessage) {
    return userCancelledMessage;
  }
  if (/openai-compatible|anthropic|model[_ -]?bridge|provider[_ -]?error|http[_ -]?status|\b(?:429|5\d\d)\b/i.test(message)) {
    return "模型服务暂时不可用，请稍后重试。";
  }
  if (/failed to fetch|networkerror|connection|timeout/i.test(message)) {
    return "问答服务暂时无法连接，请稍后重试。";
  }
  return "本次问答未能完成，请稍后重试。";
}

const agentSettingsInputClass = "h-11 rounded-xl border-white/10 bg-white/[0.045] px-3 text-white placeholder:text-white/28";
const agentParameterNameClass = "text-xs font-medium uppercase tracking-[0.16em] text-cyan-100/52";

export const AGENT_PARAMETER_HELP: Record<string, string> = {
  证据包令牌预算: "一次回答能够接收的原文证据上限。来源范围、结构恢复与命中片段都在同一硬预算内装包。",
  结果保留数量默认值: "请求未指定 top_k 时采用的分层合并命中预算；结构恢复另受独立预算控制。",
  "Dense 候选预算": "每层 Dense 通道独立提名的最大候选数。",
  "RQ 候选预算": "每层基于完整 RQ 前缀重构相关性的独立候选数。",
  "BM25 候选预算": "启用词面时，从当前 active 原文 BM25 快照独立提名的候选数。",
  根入口预算: "LLM 选择 coarse、mid 或 chunk 后，该根层融合保留的入口数。",
  逐父节点预算: "从每个父节点分别下钻时允许保留的子候选数。",
  单层总预算: "逐父节点合并和图遍历后，每一层最多保留的节点数。",
  最大遍历深度: "每层按非负累计距离扩展时允许的最大边深度。",
  每命中恢复预算: "每个最终命中可恢复的前后文、结构对象或桥接片段上限。",
  路径绿色阈值: "路径距离小于该值时视为高置信路径，通常可继续确定性扩展。",
  路径灰区阈值: "路径距离落入灰区时，由 executor 基于有界观测和版本化本地规则确定性裁决继续、下钻、走桥或停止；LLM 与证据评估器不参与、覆盖或补判。",
  路径硬中断阈值: "路径距离超过该值时执行器直接剪枝，不允许模型绕过硬阈值继续扩展。",
  回答单元上限: "一次生成可返回的完整段落或列表项上限。来源准入后只生成一次。",
  前文摘要字数上限: "仅用于理解对话指代的历史摘要上限；历史文字不成为事实证据。",
};

const retrievalBudgetFields: AgentNumberField[] = [
  { key: "context_package_token_budget", label: "证据包令牌预算", min: 256, max: 20000 },
  { key: "retrieval_result_top_k_default", label: "结果保留数量默认值", min: 1, max: 50 },
  { key: "retrieval_v1_dense_candidate_budget", label: "Dense 候选预算", min: 1, max: 4096 },
  { key: "retrieval_v1_rq_candidate_budget", label: "RQ 候选预算", min: 1, max: 4096 },
  { key: "retrieval_v1_bm25_candidate_budget", label: "BM25 候选预算", min: 1, max: 4096 },
  { key: "retrieval_v1_root_entry_budget", label: "根入口预算", min: 1, max: 256 },
  { key: "retrieval_v1_per_parent_entry_budget", label: "逐父节点预算", min: 1, max: 256 },
  { key: "retrieval_v1_layer_entry_budget", label: "单层总预算", min: 1, max: 1024 },
  { key: "retrieval_v1_max_depth", label: "最大遍历深度", min: 0, max: 64 },
  { key: "retrieval_v1_restore_per_hit", label: "每命中恢复预算", min: 0, max: 64 },
];

const agentControlFields: AgentNumberField[] = [
  { key: "agent_path_distance_green_threshold", label: "路径绿色阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_gray_threshold", label: "路径灰区阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_hard_threshold", label: "路径硬中断阈值", min: 0, max: 40, step: 0.01 },
  { key: "agent_answer_unit_limit", label: "回答单元上限", min: 1, max: 32 },
  { key: "agent_history_summary_max_chars", label: "前文摘要字数上限", min: 512, max: 12000 },
];

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function parseIntField(value: string): number | undefined {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseFloatField(value: string): number | undefined {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function stringSetting(value: number | undefined, fallback: number): string {
  return String(value ?? fallback);
}

function agentSettingsFormFromSettings(settings?: ModelSettingsResponse | null): AgentSettingsForm {
  return {
    context_package_token_budget: stringSetting(settings?.context_package_token_budget, 12000),
    retrieval_result_top_k_default: stringSetting(settings?.retrieval_result_top_k_default, 12),
    retrieval_v1_dense_candidate_budget: stringSetting(settings?.retrieval_v1_dense_candidate_budget, 64),
    retrieval_v1_rq_candidate_budget: stringSetting(settings?.retrieval_v1_rq_candidate_budget, 64),
    retrieval_v1_bm25_candidate_budget: stringSetting(settings?.retrieval_v1_bm25_candidate_budget, 64),
    retrieval_v1_root_entry_budget: stringSetting(settings?.retrieval_v1_root_entry_budget, 12),
    retrieval_v1_per_parent_entry_budget: stringSetting(settings?.retrieval_v1_per_parent_entry_budget, 8),
    retrieval_v1_layer_entry_budget: stringSetting(settings?.retrieval_v1_layer_entry_budget, 64),
    retrieval_v1_max_depth: stringSetting(settings?.retrieval_v1_max_depth, 3),
    retrieval_v1_restore_per_hit: stringSetting(settings?.retrieval_v1_restore_per_hit, 8),
    agent_path_distance_green_threshold: stringSetting(settings?.agent_path_distance_green_threshold, 0.45),
    agent_path_distance_gray_threshold: stringSetting(settings?.agent_path_distance_gray_threshold, 1.35),
    agent_path_distance_hard_threshold: stringSetting(settings?.agent_path_distance_hard_threshold, 2.4),
    agent_answer_unit_limit: stringSetting(settings?.agent_answer_unit_limit, 12),
    agent_history_summary_max_chars: stringSetting(settings?.agent_history_summary_max_chars, 4000),
  };
}

function buildAgentSettingsPayload(form: AgentSettingsForm): ModelSettingsUpdate {
  return {
    context_package_token_budget: parseIntField(form.context_package_token_budget),
    retrieval_result_top_k_default: parseIntField(form.retrieval_result_top_k_default),
    retrieval_v1_dense_candidate_budget: parseIntField(form.retrieval_v1_dense_candidate_budget),
    retrieval_v1_rq_candidate_budget: parseIntField(form.retrieval_v1_rq_candidate_budget),
    retrieval_v1_bm25_candidate_budget: parseIntField(form.retrieval_v1_bm25_candidate_budget),
    retrieval_v1_root_entry_budget: parseIntField(form.retrieval_v1_root_entry_budget),
    retrieval_v1_per_parent_entry_budget: parseIntField(form.retrieval_v1_per_parent_entry_budget),
    retrieval_v1_layer_entry_budget: parseIntField(form.retrieval_v1_layer_entry_budget),
    retrieval_v1_max_depth: parseIntField(form.retrieval_v1_max_depth),
    retrieval_v1_restore_per_hit: parseIntField(form.retrieval_v1_restore_per_hit),
    agent_path_distance_green_threshold: parseFloatField(form.agent_path_distance_green_threshold),
    agent_path_distance_gray_threshold: parseFloatField(form.agent_path_distance_gray_threshold),
    agent_path_distance_hard_threshold: parseFloatField(form.agent_path_distance_hard_threshold),
    agent_answer_unit_limit: parseIntField(form.agent_answer_unit_limit),
    agent_history_summary_max_chars: parseIntField(form.agent_history_summary_max_chars),
  };
}

export function answerAuditFromTrace(
  trace: AgentTraceEventPayload[] | undefined,
): AnswerModelAudit | null {
  const grounded = [...(trace ?? [])]
    .reverse()
    .find((event) => event.node === "grounded_answer");
  const scores = grounded?.scores as
    | { answer_model_audit?: AnswerModelAudit | null }
    | undefined;
  return scores?.answer_model_audit ?? null;
}

export function answerUsageLabel(
  audit: AgentResponse["answer_model_audit"] | null | undefined,
): string | null {
  const usage = audit?.provider_call?.usage;
  if (!usage?.usage_present) {
    return null;
  }
  const counters = [
    usage.input_tokens != null ? `输入 ${usage.input_tokens}` : null,
    usage.output_tokens != null ? `输出 ${usage.output_tokens}` : null,
    usage.cache_read_input_tokens != null
      ? `缓存读取 ${usage.cache_read_input_tokens}`
      : null,
  ].filter((value): value is string => Boolean(value));
  if (!counters.length) {
    return null;
  }
  return `Provider tokens：${counters.join(" · ")}${usage.cache_hit ? " · 缓存命中" : ""}`;
}

function answerModelLabel(latestRun: AgentResponse | null, configuredChatModel?: string | null): string {
  const audit = latestRun?.answer_model_audit;
  if (!audit) {
    return configuredChatModel ? `模型：${configuredChatModel}` : "模型：未读取";
  }
  if (audit.external_called) {
    return `模型：${audit.model ?? audit.chat_model ?? audit.provider}`;
  }
  return "模型：未调用";
}

export function normalizeMessages(messages: SessionMessage[] | Array<Record<string, unknown>>, conversationState?: ConversationStatePayload | null): ChatTurn[] {
  const referencesByRunId = new Map(conversationState?.history_references.map((reference) => [reference.run_id, reference]) ?? []);
  return (messages as Array<Record<string, unknown>>)
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item) => {
      const messageCitations = Array.isArray(item.citations) ? (item.citations as Citation[]) : undefined;
      const citationTraceId = messageCitations?.find((citation) => typeof citation.retrieval_trace_id === "string")?.retrieval_trace_id;
      const citationPackageId = messageCitations?.find((citation) => typeof citation.context_package_id === "string")?.context_package_id;
      const runId = typeof item.run_id === "string" ? item.run_id : null;
      const directAnswerMode =
        item.direct_answer_mode === "system_capability" || item.direct_answer_mode === "verified_context_reuse"
          ? item.direct_answer_mode
          : null;
      const historyReference = runId ? referencesByRunId.get(runId) : undefined;
      return {
        role: item.role as "user" | "assistant",
        content: String(item.content ?? ""),
        run_id: runId,
        route: typeof item.route === "string" ? item.route : null,
        direct_answer_mode: directAnswerMode,
        citations: messageCitations,
        trace: Array.isArray(item.trace) ? (item.trace as AgentTraceEventPayload[]) : undefined,
        retrieval_trace_id: typeof item.retrieval_trace_id === "string" ? item.retrieval_trace_id : citationTraceId ?? historyReference?.retrieval_trace_id ?? null,
        context_package_id: typeof item.context_package_id === "string" ? item.context_package_id : citationPackageId ?? historyReference?.context_package_id ?? null,
        citation_replay_status:
          item.citation_replay_status === "valid" || item.citation_replay_status === "unavailable"
            ? item.citation_replay_status
            : "not_present",
        citation_replay_reason:
          item.citation_replay_reason === "persisted_citation_contract_mismatch"
            ? item.citation_replay_reason
            : null,
      };
    });
}

export function preserveTurnTraces(nextTurns: ChatTurn[], currentTurns: ChatTurn[]): ChatTurn[] {
  const traceByRunId = new Map(
    currentTurns
      .filter((turn) => turn.role === "assistant" && turn.run_id && turn.trace?.length)
      .map((turn) => [turn.run_id as string, turn.trace as AgentTraceEventPayload[]]),
  );
  return nextTurns.map((turn) => ({
    ...turn,
    trace: turn.trace?.length
      ? turn.trace
      : turn.run_id
        ? traceByRunId.get(turn.run_id)
        : undefined,
  }));
}

function ChatHeader({
  latestRun,
  configuredChatModel,
  modelLoading,
}: {
  latestRun: AgentResponse | null;
  configuredChatModel?: string | null;
  modelLoading: boolean;
}) {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-1">
      <div className="min-w-0">
        <p className="section-kicker">资料库智能问答</p>
        <h2 className="mt-1 text-2xl font-semibold text-white lg:text-3xl">向智能体提问</h2>
      </div>
      <div className="flex w-full flex-wrap gap-2">
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          {modelLoading ? (
            <>
              <Loader2 data-icon="inline-start" className="animate-spin" />
              正在读取模型配置...
            </>
          ) : (
            <>
              <BrainCircuit data-icon="inline-start" />
              {answerModelLabel(latestRun, configuredChatModel)}
            </>
          )}
        </span>
      </div>
    </div>
  );
}

export function AgentParameterName({
  label,
  description,
  className = agentParameterNameClass,
}: {
  label: string;
  description?: string;
  className?: string;
}) {
  const tooltipId = useId();
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const triggerRef = useRef<HTMLSpanElement | null>(null);
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ left: 0, top: 0 });
  const helpText = description ?? AGENT_PARAMETER_HELP[label];

  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);

  if (!helpText) {
    return <span className={className}>{label}</span>;
  }

  const clearHoverTimer = () => {
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
  };

  const openTooltip = () => {
    const rect = triggerRef.current?.getBoundingClientRect();
    const viewportWidth = typeof window === "undefined" ? 1024 : window.innerWidth;
    const tooltipWidth = 288;
    const gutter = 16;
    setTooltipPosition({
      left: Math.max(gutter, Math.min(rect?.left ?? gutter, viewportWidth - tooltipWidth - gutter)),
      top: (rect?.bottom ?? 0) + 8,
    });
    setIsTooltipVisible(true);
  };

  const handleMouseEnter = () => {
    clearHoverTimer();
    hoverTimerRef.current = setTimeout(openTooltip, 1000);
  };

  const handleMouseLeave = () => {
    clearHoverTimer();
    setIsTooltipVisible(false);
  };

  return (
    <>
      <span
        ref={triggerRef}
        className={`relative inline-flex w-fit cursor-help items-center gap-1 rounded-sm ${className}`}
        aria-describedby={isTooltipVisible ? tooltipId : undefined}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <span>{label}</span>
        <Info className="size-3.5 text-cyan-100/45" aria-hidden="true" />
      </span>
      {isTooltipVisible && typeof document !== "undefined"
        ? createPortal(
            <span
              id={tooltipId}
              role="tooltip"
              data-testid="agent-parameter-tooltip"
              className="pointer-events-none fixed z-[9999] w-72 max-w-[calc(100vw-2rem)] rounded-xl border border-cyan-100/20 bg-[#081322]/95 p-3 text-left text-xs font-normal normal-case leading-5 tracking-normal text-cyan-50/82 opacity-100 shadow-2xl shadow-black/30 backdrop-blur"
              style={{ left: tooltipPosition.left, top: tooltipPosition.top }}
            >
              {helpText}
            </span>,
            document.body,
          )
        : null}
    </>
  );
}

function AgentSettingsField({
  field,
  value,
  onChange,
  disabled,
}: {
  field: AgentNumberField;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="flex min-w-0 flex-col gap-2">
      <AgentParameterName label={field.label} />
      <Input
        type="number"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        className={agentSettingsInputClass}
      />
    </label>
  );
}

function AgentSettingsDialog({
  open,
  onOpenChange,
  form,
  onChange,
  onReset,
  onSave,
  isLoading,
  error,
  isSaving,
  savedMessage,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  form: AgentSettingsForm | null;
  onChange: <K extends keyof AgentSettingsForm>(key: K, value: AgentSettingsForm[K]) => void;
  onReset: () => void;
  onSave: () => void;
  isLoading: boolean;
  error: Error | null;
  isSaving: boolean;
  savedMessage: { kind: "success" | "error"; text: string } | null;
}) {
  const disabled = isLoading || isSaving || !form;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[min(46rem,calc(100dvh-2rem))] max-h-[calc(100dvh-2rem)] w-[min(58rem,calc(100vw-2rem))] flex-col overflow-hidden border border-cyan-200/14 bg-[rgba(3,10,22,0.96)] p-0 text-white shadow-[0_30px_90px_rgba(0,0,0,0.48)] backdrop-blur-2xl sm:!max-w-[58rem]">
        <DialogHeader className="shrink-0 border-b border-cyan-200/10 px-6 py-5 pr-14">
          <DialogTitle className="flex items-center gap-2 text-lg text-white">
            <SlidersHorizontal className="size-5 text-cyan-100/78" />
            智能体参数
          </DialogTitle>
          <DialogDescription className="text-cyan-50/58">模型会为每个问题选择 coarse、mid 或 chunk 入口；这里只设置执行器硬预算。</DialogDescription>
        </DialogHeader>
        <form
          className="flex min-h-0 flex-1 flex-col"
          onSubmit={(event) => {
            event.preventDefault();
            onSave();
          }}
        >
          <ScrollArea className="min-h-0 flex-1 px-6 py-5 pr-4">
            {isLoading ? <LoadingBlock rows={3} /> : null}
            {error ? <ErrorBlock message={error.message} /> : null}
            {form ? (
              <div className="grid gap-6">
                <section className="grid gap-4">
                  <div className="border-b border-cyan-100/12 pb-3">
                    <p className="text-sm font-semibold text-white">候选、入口与证据包</p>
                    <p className="mt-1 text-xs leading-5 text-white/48">Dense、RQ 与 BM25 独立提名；权重和入口由本轮冻结计划决定。</p>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    {retrievalBudgetFields.map((field) => (
                      <AgentSettingsField
                        key={field.key}
                        field={field}
                        value={form[field.key]}
                        onChange={(value) => onChange(field.key, value)}
                        disabled={disabled}
                      />
                    ))}
                  </div>
                </section>

                <section className="grid gap-4">
                  <div className="border-b border-cyan-100/12 pb-3">
                    <p className="text-sm font-semibold text-white">遍历与一次生成</p>
                    <p className="mt-1 text-xs leading-5 text-white/48">灰区由本地确定性规则裁决；来源准入通过后只生成一次。</p>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    {agentControlFields.map((field) => (
                      <AgentSettingsField
                        key={field.key}
                        field={field}
                        value={form[field.key]}
                        onChange={(value) => onChange(field.key, value)}
                        disabled={disabled}
                      />
                    ))}
                  </div>
                </section>
              </div>
            ) : null}
          </ScrollArea>
          <DialogFooter className="shrink-0 border-t border-cyan-200/10 bg-white/[0.025] px-6 pb-6 pt-4">
            <div className="flex w-full flex-wrap items-center justify-between gap-3">
              <div className={cn("text-sm", savedMessage?.kind === "error" ? "text-rose-100/78" : "text-emerald-100/72")}>{savedMessage?.text}</div>
              <div className="flex flex-wrap items-center gap-2">
                <Button type="button" variant="outline" onClick={onReset} disabled={disabled} className="border-white/10 bg-white/[0.03] text-white hover:bg-white/[0.08]">
                  <RotateCcw className="size-4" />
                  重置
                </Button>
                <Button type="submit" disabled={disabled} className="rounded-full bg-cyan-300 text-slate-950 hover:bg-cyan-200">
                  {isSaving ? <Loader2 className="size-4 animate-spin" /> : <Save className="size-4" />}
                  保存
                </Button>
              </div>
            </div>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function ChatActionRail({
  onOpenSessions,
  onOpenCitations,
  onOpenAgentSettings,
  citationsCount,
}: {
  onOpenSessions: () => void;
  onOpenCitations: () => void;
  onOpenAgentSettings: () => void;
  citationsCount: number;
}) {
  const actions = [
    { label: "会话", icon: History, onClick: onOpenSessions },
    { label: `引用 ${citationsCount}`, icon: FileText, onClick: onOpenCitations },
  ];

  return (
    <div className="fixed bottom-[11.5rem] right-4 z-40 flex flex-col items-end gap-2 lg:bottom-auto lg:right-7 lg:top-[10rem]">
      <button
        type="button"
        onClick={onOpenAgentSettings}
        className="group grid size-11 place-items-center rounded-full border border-cyan-200/18 bg-[rgba(7,13,31,0.9)] text-cyan-100/72 shadow-[0_16px_44px_rgba(0,0,0,0.28),0_0_28px_rgba(86,217,255,0.08)] backdrop-blur-2xl transition hover:border-cyan-200/36 hover:bg-cyan-300/[0.09] hover:text-white"
        title="智能体参数"
        aria-label="智能体参数"
      >
        <Settings className="size-4 transition group-hover:rotate-45" />
      </button>
      {actions.map(({ label, icon: Icon, onClick }) => (
        <button
          key={label}
          type="button"
          onClick={onClick}
          className="group flex h-11 items-center justify-end gap-2 rounded-full border border-cyan-200/14 bg-[rgba(7,13,31,0.88)] px-3 text-xs text-white/68 shadow-[0_16px_44px_rgba(0,0,0,0.28),0_0_28px_rgba(86,217,255,0.06)] backdrop-blur-2xl transition hover:border-cyan-200/32 hover:bg-cyan-300/[0.08] hover:text-white"
        >
          <span className="hidden whitespace-nowrap sm:inline">{label}</span>
          <Icon className="size-4 text-cyan-100/72 transition group-hover:text-cyan-100" />
        </button>
      ))}
    </div>
  );
}

function EmptyChatState() {
  return (
    <div className="grid min-h-[calc(100dvh-21rem)] place-items-center px-2 pb-44 pt-12 text-center sm:px-4">
      <div className="-translate-y-16 sm:-translate-y-24">
        <div className="mx-auto grid size-14 place-items-center rounded-3xl border border-cyan-200/14 bg-cyan-300/[0.045] text-cyan-100 shadow-[0_0_42px_rgba(86,217,255,0.08)] sm:size-16">
          <Sparkles />
        </div>
        <h3 className="glow-text mx-auto mt-6 max-w-[16rem] break-words text-xl font-semibold leading-snug text-white sm:max-w-3xl sm:text-3xl">
          开始一轮有证据支撑的资料问答
        </h3>
        <p className="mx-auto mt-3 max-w-[21rem] text-sm leading-7 text-white/56 sm:max-w-2xl">
          系统会从资料中寻找相关内容、补充必要上下文，并生成带来源的回答。
        </p>
      </div>
    </div>
  );
}

export function MessageBubble({
  turn,
  index,
  onOpenCitations,
}: {
  turn: ChatTurn;
  index: number;
  onOpenCitations: (citations: Citation[]) => void;
  defaultContextExpanded?: boolean;
}) {
  const isUser = turn.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.025, 0.18) }}
      className={cn("flex", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "relative max-w-[min(860px,92%)] px-1 py-3",
          isUser
            ? "rounded-[1.25rem] border border-cyan-200/12 bg-cyan-300/[0.045] px-5 shadow-[0_0_24px_rgba(86,217,255,0.035)]"
            : "w-full border-l border-cyan-200/18 pl-5 text-white",
        )}
      >
        <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-white/38">
          {isUser ? <CircleDot /> : <BrainCircuit />}
          {isUser ? "你" : "智能体"}
          {!isUser && turn.direct_answer_mode ? (
            <span className="rounded-full border border-cyan-200/15 px-2 py-1 text-[10px] tracking-[0.12em] text-cyan-50/55">
              {turn.direct_answer_mode === "system_capability" ? "系统能力回答" : "复用已验证证据"}
            </span>
          ) : null}
        </div>
        {!isUser && turn.trace?.length ? <AgentTraceStream trace={turn.trace} compact className="mb-5" /> : null}
        <MarkdownRenderer content={turn.content} className={cn(isUser ? "text-white/78" : "text-white/74")} />
        {!isUser && turn.citation_replay_status === "unavailable" ? (
          <div
            data-testid="citation-replay-unavailable"
            className="mt-4 rounded-xl border border-amber-300/20 bg-amber-300/[0.055] px-4 py-3 text-sm leading-6 text-amber-50/80"
          >
            该历史回答的来源信息已过期，暂不展示。你可以重新提问以获得可核验的新引用。
          </div>
        ) : null}
        {!isUser && turn.citations?.length ? (
          <button type="button" onClick={() => onOpenCitations(turn.citations ?? [])} className="kg-micro-chip mt-4 rounded-full px-3 py-2 text-xs transition hover:border-cyan-200/30 hover:text-white">
            <FileText />
            {turn.citations.length} 条来源 · 查看
          </button>
        ) : null}
      </div>
    </motion.div>
  );
}

export function GeneratingBubble({ content, trace }: { content: string; trace: AgentTraceEventPayload[] }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-start">
      <div className="w-full max-w-[min(860px,92%)] border-l border-cyan-200/18 px-5 py-4 text-white">
        <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-cyan-100/50">
          <Loader2 className="size-4 animate-spin" />
          {content ? "正在输出" : "智能体运行中"}
        </div>
        {!content ? <AgentTraceStream trace={trace} isRunning defaultExpanded compact className="mb-5" /> : null}
        {content ? (
          <div className="relative">
            <MarkdownRenderer content={content} className="pr-3 text-white/76" />
            <span className="stream-cursor">|</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-white/56">
            <Loader2 className="size-5 animate-spin text-cyan-100" />
            正在查找资料并核对来源...
          </div>
        )}
      </div>
    </motion.div>
  );
}

export function MessageList({
  turns,
  isLoading,
  isGenerating,
  draftAnswer,
  trace,
  onOpenCitations,
}: {
  turns: ChatTurn[];
  isLoading: boolean;
  isGenerating: boolean;
  draftAnswer: string;
  trace: AgentTraceEventPayload[];
  onOpenCitations: (citations: Citation[]) => void;
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const previousTurnCountRef = useRef(turns.length);

  useEffect(() => {
    const hasNewTurn = turns.length !== previousTurnCountRef.current;
    previousTurnCountRef.current = turns.length;
    if (!hasNewTurn && !isGenerating) {
      return undefined;
    }
    if (scrollFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollFrameRef.current);
    }
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({
        behavior: isGenerating ? "smooth" : "auto",
        block: "end",
      });
      scrollFrameRef.current = null;
    });
    return () => {
      if (scrollFrameRef.current !== null) {
        window.cancelAnimationFrame(scrollFrameRef.current);
        scrollFrameRef.current = null;
      }
    };
  }, [turns.length, draftAnswer, isGenerating]);

  return (
    <div className="relative min-h-[calc(100dvh-21rem)]">
      {isLoading && turns.length === 0 && !isGenerating ? (
        <LoadingBlock rows={3} />
      ) : turns.length === 0 && !isGenerating ? (
        <EmptyChatState />
      ) : (
        <div className="mx-auto flex max-w-5xl flex-col gap-8 px-1 pb-6 pt-4">
          {turns.map((turn, index) => (
            <MessageBubble
              key={`${turn.role}-${index}-${turn.run_id ?? "local"}`}
              turn={turn}
              index={index}
              onOpenCitations={onOpenCitations}
            />
          ))}
          {isGenerating ? <GeneratingBubble content={draftAnswer} trace={trace} /> : null}
          <div ref={bottomRef} className="h-52 shrink-0 md:h-56" />
        </div>
      )}
    </div>
  );
}

function ChatComposer({
  value,
  onChange,
  onSubmit,
  onCancel,
  isPending,
  activeSessionId,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  isPending: boolean;
  activeSessionId: string | null;
}) {
  const handleSubmit = () => {
    if (isPending || !value.trim()) {
      return;
    }
    onSubmit();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="pointer-events-none fixed inset-x-4 bottom-20 z-[45] lg:bottom-4 lg:left-[calc(76px+1.75rem)] lg:right-7"
    >
      <div className="pointer-events-auto mx-auto w-full max-w-5xl">
        <div
          className={cn(
            "kg-scan-edge rounded-[1.7rem] border border-cyan-200/16 bg-[rgba(7,13,31,0.94)] p-2 shadow-[0_20px_70px_rgba(0,0,0,0.42),0_0_42px_rgba(86,217,255,0.08)] backdrop-blur-2xl",
            isPending && "border-cyan-100/24 shadow-[0_20px_70px_rgba(0,0,0,0.34),0_0_58px_rgba(86,217,255,0.14)]",
          )}
        >
          <div className="flex flex-col gap-3 rounded-[1.35rem] bg-[linear-gradient(135deg,rgba(86,217,255,0.065),rgba(122,95,255,0.035)_55%,rgba(0,0,0,0.12))] p-3">
            <div className="flex flex-wrap items-center gap-2 px-1">
              <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">
                <Layers3 />
                资料库上下文
              </span>
              <span className="kg-micro-chip max-w-full truncate rounded-full px-2.5 py-1 text-[11px]">
                {activeSessionId ? "会话已建立" : "新建会话"}
              </span>
              <span className="hidden text-[11px] text-white/42 sm:inline">入口与通道由本轮意图计划选择</span>
            </div>
            <div className="flex items-end gap-3">
              <Textarea
                value={value}
                onChange={(event) => onChange(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (isPending) {
                      return;
                    }
                    handleSubmit();
                  }
                }}
                className="max-h-44 min-h-[72px] resize-none border-0 bg-transparent px-2 text-base text-white shadow-none placeholder:text-white/30 focus-visible:ring-0"
                placeholder="输入问题，系统会规划检索、核对来源并给出一次有引用的回答..."
              />
              <Button
                type="button"
                size="icon-lg"
                className={cn(
                  isPending
                    ? "rounded-[0.45rem] border-rose-200/40 bg-rose-500 text-white shadow-[0_0_24px_rgba(244,63,94,0.35)] hover:bg-rose-400"
                    : "rounded-full",
                )}
                onClick={isPending ? onCancel : handleSubmit}
                disabled={!isPending && !value.trim()}
                title={isPending ? "取消当前对话" : "发送"}
                aria-label={isPending ? "取消当前对话" : "提问"}
              >
                {isPending ? <Square className="size-4 fill-current stroke-[2.4]" /> : <Send />}
                <span className="sr-only">{isPending ? "取消当前对话" : "提问"}</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function SessionsDrawer({
  open,
  onOpenChange,
  sessions,
  activeSessionId,
  onSelect,
  onDelete,
  onNew,
  isPending,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => Promise<void>;
  onDelete: (sessionId: string) => void | Promise<void>;
  onNew: () => void;
  isPending: boolean;
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-md">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>会话</SheetTitle>
          <SheetDescription>资料库智能体的对话记忆。</SheetDescription>
        </SheetHeader>
        <div className="p-5">
          <Button
            type="button"
            className="w-full rounded-full"
            disabled={isPending}
            onClick={() => {
              if (isPending) {
                return;
              }
              onNew();
              onOpenChange(false);
            }}
          >
            <Plus data-icon="inline-start" />
            新建会话
          </Button>
        </div>
        <ScrollArea className="h-[calc(100dvh-10rem)] px-5 pb-5">
          <div className="flex flex-col gap-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "flex items-start gap-2 rounded-2xl border px-3 py-3 transition",
                  session.id === activeSessionId ? "border-cyan-200/28 bg-cyan-300/[0.075]" : "border-white/7 bg-white/[0.025] hover:border-cyan-200/22",
                  isPending && "pointer-events-none opacity-50",
                )}
              >
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => {
                    if (isPending) {
                      return;
                    }
                    void onSelect(session.id)
                      .finally(() => onOpenChange(false))
                      .catch(() => undefined);
                  }}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="min-w-0 truncate text-sm font-medium text-white">{session.title ?? "未命名会话"}</span>
                    <ChevronRight className="text-white/35" />
                  </div>
                  {session.last_question ? <p className="mt-2 line-clamp-2 text-xs leading-5 text-white/45">{session.last_question}</p> : null}
                </button>
                <button
                  type="button"
                  aria-label="删除会话"
                  disabled={isPending}
                  onClick={() => onDelete(session.id)}
                  className="grid size-8 shrink-0 place-items-center rounded-full border border-white/8 text-white/45 transition hover:border-rose-200/30 hover:bg-rose-300/[0.08] hover:text-rose-100 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

function CitationsDrawer({
  open,
  onOpenChange,
  citations,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  citations: Citation[];
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full min-w-0 overflow-hidden border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-xl">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>引用</SheetTitle>
          <SheetDescription>查看回答所依据的资料片段、页码和章节位置。</SheetDescription>
        </SheetHeader>
        <ScrollArea className="h-[calc(100dvh-8rem)] min-w-0 overflow-hidden p-6">
          <div className="flex min-w-0 max-w-full flex-col gap-3 overflow-hidden">
            {citations.length === 0 ? (
              <div className="kg-glass-line rounded-3xl px-6 py-10 text-center text-sm text-white/55">
                <Archive className="mx-auto mb-4 text-cyan-100/70" />
                有证据回答完成后会显示引用。
              </div>
            ) : (
              citations.map((citation, index) => <CitationCard key={`${citation.chunk_id}-${index}`} citation={citation} index={index} />)
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

export function ConversationStatePanel({ state }: { state: ConversationStatePayload | null }) {
  if (!state) {
    return null;
  }

  const filters = Object.entries(state.active_user_constraints.retrieval_filters).filter(([, value]) => {
    if (Array.isArray(value)) {
      return value.length > 0;
    }
    return value !== null && value !== undefined && value !== "";
  });
  const taskStatusLabels: Record<string, string> = {
    active: "进行中",
    waiting_user: "等待你的下一步",
    completed: "已完成",
    cancelled: "已取消",
    failed: "未完成",
  };
  const taskStepLabels: Record<string, string> = {
    awaiting_user: "等待下一问题",
    retrieving: "查找资料",
    answering: "整理回答",
    verifying: "核对来源",
    cancelled: "本轮已停止",
    failed: "本轮未完成",
  };
  const taskStatusLabel =
    taskStatusLabels[state.task_state.status] ?? "进行中";
  const currentStepLabel = state.task_state.current_step
    ? taskStepLabels[state.task_state.current_step] ?? "继续当前问答"
    : "等待你的问题";
  return (
    <section
      data-testid="conversation-state-panel"
      className="mb-4 flex flex-wrap items-center gap-2 border-l border-cyan-200/18 py-1 pl-4 text-xs text-white/52"
    >
      <span className="kg-micro-chip rounded-full px-3 py-1.5">{taskStatusLabel} · {currentStepLabel}</span>
      {state.task_state.objective ? <span className="max-w-xl truncate">{state.task_state.objective}</span> : null}
      {state.active_user_constraints.instructions.slice(0, 2).map((instruction) => (
        <span key={instruction} className="max-w-sm truncate rounded-full border border-white/8 px-2.5 py-1">{instruction}</span>
      ))}
      {filters.length ? <span>已应用资料范围</span> : null}
      <span>已有 {state.history_references.length} 轮回答</span>
      <span className="text-amber-100/62">事实仍以资料来源为准</span>
    </section>
  );
}

function QAWorkspaceContent({ selectedKnowledgeBaseId }: { selectedKnowledgeBaseId: string | null }) {
  const queryClient = useQueryClient();
  const storageScope = selectedKnowledgeBaseId ?? "unassigned";
  const sessionsQuery = useQuery({
    queryKey: ["sessions", selectedKnowledgeBaseId],
    queryFn: () => fetchSessions(selectedKnowledgeBaseId),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const modelSettingsQuery = useQuery({ queryKey: ["model-settings"], queryFn: fetchModelSettings });
  const [question, setQuestion] = useLocalStorage(`qa.question.${storageScope}`, "");
  const [activeSessionId, setActiveSessionId] = useLocalStorage<string | null>(`qa.sessionId.${storageScope}`, null);
  const [autoResumeSuppressed, setAutoResumeSuppressed] = useLocalStorage(
    `qa.autoResumeSuppressed.${storageScope}`,
    false,
  );
  const [activeStream, setActiveStream] = useLocalStorage<ActiveStreamState | null>(`qa.activeStream.${storageScope}`, null);
  // PostgreSQL session/answer/trace rows are the durable conversation source.
  // Large citations, trace audits and AgentResponse payloads must stay in
  // memory; persisting them redundantly in localStorage exceeds browser quota
  // and can prevent the final turn from rendering.
  const cachedSession = activeSessionId
    ? queryClient.getQueryData<SessionMessagesResponse>(["session-messages", activeSessionId])
    : undefined;
  const cachedTurns = cachedSession
    ? normalizeMessages(cachedSession.messages, cachedSession.conversation_state)
    : activeStream?.question
      ? [{ role: "user" as const, content: activeStream.question }]
      : [];
  const cachedLatestAssistant = [...cachedTurns]
    .reverse()
    .find((turn) => turn.role === "assistant");
  const [turns, setTurns] = useState<ChatTurn[]>(() => cachedTurns);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>(() => cachedLatestAssistant?.citations ?? []);
  const [trace, setTrace] = useState<AgentTraceEventPayload[]>(() => cachedLatestAssistant?.trace ?? []);
  const [latestRun, setLatestRun] = useState<AgentResponse | null>(null);
  const [conversationState, setConversationState] = useState<ConversationStatePayload | null>(
    () => cachedSession?.conversation_state ?? null,
  );
  const [streamError, setStreamError] = useState<string | null>(null);
  const streamAbortControllerRef = useRef<AbortController | null>(null);
  const runViewRevisionRef = useRef(0);
  const [sessionReplayVersion, setSessionReplayVersion] = useState(0);
  const [sessionSelectionPending, setSessionSelectionPending] = useState(false);
  const [hydratingSessionId, setHydratingSessionId] = useState<string | null>(null);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [citationsOpen, setCitationsOpen] = useState(false);
  const [citationDrawerCitations, setCitationDrawerCitations] = useState<Citation[]>([]);
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [agentSettingsForm, setAgentSettingsForm] = useState<AgentSettingsForm | null>(null);
  const [agentSettingsSavedMessage, setAgentSettingsSavedMessage] = useState<{ kind: "success" | "error"; text: string } | null>(null);
  const hydratedSessionIdRef = useRef<string | null>(null);
  const activeRunId = activeStream?.runId ?? null;
  const openCitationDrawer = (items: Citation[]) => {
    setCitationDrawerCitations(items);
    setCitationsOpen(true);
  };
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    for (const key of legacyQaPayloadStorageKeys(storageScope)) {
      window.localStorage.removeItem(key);
    }
  }, [storageScope]);

  useEffect(() => () => {
    const controller = streamAbortControllerRef.current;
    streamAbortControllerRef.current = null;
    controller?.abort();
  }, []);

  useEffect(() => {
    const latestSession = sessionsQuery.data?.[0];
    if (
      activeSessionId
      || activeStream
      || autoResumeSuppressed
      || !latestSession
    ) {
      return;
    }
    hydratedSessionIdRef.current = null;
    setActiveSessionId(latestSession.id);
  }, [activeSessionId, activeStream, autoResumeSuppressed, sessionsQuery.data, setActiveSessionId]);

  useEffect(() => {
    if (
      !activeSessionId
      || activeStream
      || hydratedSessionIdRef.current === activeSessionId
    ) {
      return;
    }
    let cancelled = false;
    hydratedSessionIdRef.current = activeSessionId;
    void (async () => {
      setHydratingSessionId(activeSessionId);
      try {
        const response = await queryClient.fetchQuery({
          queryKey: ["session-messages", activeSessionId],
          queryFn: () => fetchSessionMessages(activeSessionId),
          staleTime: Number.POSITIVE_INFINITY,
        });
        const nextTurns = normalizeMessages(response.messages, response.conversation_state);
        if (cancelled) {
          return;
        }
        setTurns((current) => preserveTurnTraces(nextTurns, current));
        setConversationState(response.conversation_state);
        const latestAssistant = [...nextTurns]
          .reverse()
          .find((turn) => turn.role === "assistant");
        setCitations(latestAssistant?.citations ?? []);
        setTrace(latestAssistant?.trace ?? []);
        setStreamError((current) => current === missingSessionMessage ? null : current);
      } catch (error) {
        if (!cancelled) {
          hydratedSessionIdRef.current = null;
          if (responseStatus(error) === 404) {
            queryClient.setQueryData<SessionSummary[]>(
              ["sessions", selectedKnowledgeBaseId],
              (current) => current?.filter((session) => session.id !== activeSessionId),
            );
            void queryClient.invalidateQueries({
              queryKey: ["sessions", selectedKnowledgeBaseId],
            });
            setActiveSessionId(null);
            setAutoResumeSuppressed(true);
            setTurns([]);
            setConversationState(null);
            setCitations([]);
            setTrace([]);
            setStreamError(missingSessionMessage);
          } else {
            setStreamError(productQaErrorMessage(error));
          }
        }
      } finally {
        if (!cancelled) {
          setHydratingSessionId((current) => current === activeSessionId ? null : current);
        }
      }
    })();
    return () => {
      cancelled = true;
      // React Strict Mode replays effects in development.  Releasing the
      // in-flight marker lets the replayed effect perform the authoritative
      // server hydration instead of treating the cancelled first pass as a
      // completed session load.
      if (hydratedSessionIdRef.current === activeSessionId) {
        hydratedSessionIdRef.current = null;
      }
    };
  }, [
    activeStream,
    activeSessionId,
    queryClient,
    selectedKnowledgeBaseId,
    sessionReplayVersion,
    setActiveSessionId,
    setAutoResumeSuppressed,
  ]);

  const runStatusQuery = useQuery({
    queryKey: ["agent-run", activeRunId],
    queryFn: () => fetchTaskStatus(activeRunId as string),
    enabled: Boolean(activeRunId),
    refetchInterval: activeRunId ? 1500 : false,
    retry: false,
  });
  const saveAgentSettingsMutation = useMutation({
    mutationFn: (payload: ModelSettingsUpdate) => updateModelSettings(payload),
    onSuccess: async (settings) => {
      setAgentSettingsForm(agentSettingsFormFromSettings(settings));
      setAgentSettingsSavedMessage({ kind: "success", text: "已保存" });
      window.setTimeout(() => setAgentSettingsSavedMessage(null), 1800);
      await queryClient.invalidateQueries({ queryKey: ["model-settings"] });
    },
    onError: (error) => {
      setAgentSettingsSavedMessage({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    },
  });

  const finishRunFromStatus = useCallback((status: TaskStatusResponse, sessionId?: string | null) => {
    const controller = streamAbortControllerRef.current;
    // Detach before aborting: closing an already terminal run is not a user cancel.
    streamAbortControllerRef.current = null;
    controller?.abort();
    setDraftAnswer("");
    setActiveStream(null);
    if (status.trace?.length) {
      setTrace(status.trace);
    }
    const runState = status.status ?? status.state;
    if (runState === "completed" || runState === "needs_clarification") {
      setStreamError(null);
      if (status.answer) {
        setTurns((current) => current.some((turn) => turn.role === "assistant" && turn.run_id === status.run_id)
          ? current
          : [...current, {
            role: "assistant",
            content: status.answer ?? "",
            run_id: status.run_id,
            route: status.route,
            direct_answer_mode: status.direct_answer_mode,
            trace: status.trace ?? [],
          }]);
      }
      void queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
    } else {
      setStreamError(runState === "cancelled" ? userCancelledMessage : productQaErrorMessage(status.error));
    }
    if (sessionId) {
      // Rehydrate every terminal state. Failed/cancelled runs do not append an
      // answer, but they still persist the authoritative conversation task state.
      hydratedSessionIdRef.current = null;
      setAutoResumeSuppressed(false);
      setActiveSessionId(sessionId);
      setSessionReplayVersion((current) => current + 1);
      void queryClient.invalidateQueries({ queryKey: ["session-messages", sessionId] });
    }
  }, [queryClient, selectedKnowledgeBaseId, setActiveSessionId, setActiveStream, setAutoResumeSuppressed]);

  const cancelRunMutation = useMutation({
    mutationFn: ({ runId }: { runId: string; viewRevision: number }) => cancelAgentRun(runId),
    onSuccess: (status, { viewRevision }) => {
      if (runViewRevisionRef.current === viewRevision
        && ["completed", "needs_clarification", "failed", "cancelled"].includes(status.status ?? status.state ?? "")) {
        finishRunFromStatus(status, status.session_id ?? activeSessionId);
      }
      void queryClient.invalidateQueries({ queryKey: ["agent-run", status.run_id] });
    },
    onError: (error, { viewRevision }) => {
      if (runViewRevisionRef.current === viewRevision) {
        setStreamError(productQaErrorMessage(error));
      }
    },
  });

  const askMutation = useMutation({
    mutationFn: async () => {
      const nextQuestion = question.trim();
      if (!nextQuestion) {
        return;
      }
      runViewRevisionRef.current += 1;
      setStreamError(null);
      setDraftAnswer("");
      setCitations([]);
      setCitationDrawerCitations([]);
      setTrace([]);
      setLatestRun(null);
      setActiveStream({ question: nextQuestion, startedAt: new Date().toISOString() });
      const controller = new AbortController();
      streamAbortControllerRef.current = controller;
      const isCurrentStream = () => streamAbortControllerRef.current === controller && !controller.signal.aborted;
      let receivedRunId: string | null = null;
      const nextTraceEvents: AgentTraceEventPayload[] = [];
      const recoverPersistedRun = (message: string, transportInterrupted = false) => {
        streamAbortControllerRef.current = null;
        setDraftAnswer("");
        if (receivedRunId) {
          setStreamError(transportInterrupted
            ? "实时连接已中断，正在读取本轮任务状态。"
            : productQaErrorMessage(message));
          setActiveStream((current) => current ? {
            ...current,
            runId: receivedRunId,
          } : current);
          void queryClient.invalidateQueries({ queryKey: ["agent-run", receivedRunId] });
          return;
        }
        setStreamError(productQaErrorMessage(message));
        setActiveStream(null);
      };
      setTurns((current) => [...current, { role: "user", content: nextQuestion }]);
      setQuestion("");
      try {
        await streamAnswer(
          {
            question: nextQuestion,
            session_id: activeSessionId,
            knowledge_base_id: selectedKnowledgeBaseId,
          },
          {
            onTrace: (event) => {
              if (!isCurrentStream()) return;
              nextTraceEvents.push(event);
              setTrace((current) => [...current, event]);
            },
            onToken: (token) => {
              if (isCurrentStream()) setDraftAnswer((current) => `${current}${token}`);
            },
            onAnswerReplace: (answer) => {
              if (isCurrentStream()) setDraftAnswer(answer);
            },
            onCitations: (next) => {
              if (isCurrentStream()) setCitations(next);
            },
            onMeta: (meta) => {
              if (!isCurrentStream()) return;
              receivedRunId = meta.run_id ?? receivedRunId;
              if (meta.session_id) {
                setAutoResumeSuppressed(false);
                setActiveSessionId(meta.session_id);
              }
              if (meta.run_id || meta.session_id) {
                setActiveStream((current) => current ? {
                  ...current,
                  runId: meta.run_id ?? current.runId ?? null,
                  sessionId: meta.session_id ?? current.sessionId ?? null,
                } : current);
              }
            },
            onFinal: (response) => {
              if (!isCurrentStream()) return;
              streamAbortControllerRef.current = null;
              const finalTrace = response.trace.length ? response.trace : nextTraceEvents;
              setLatestRun(response);
              setConversationState(response.conversation_state ?? null);
              setAutoResumeSuppressed(false);
              hydratedSessionIdRef.current = null;
              setActiveSessionId(response.session_id);
              setCitations(response.citations);
              setDraftAnswer("");
              setTrace(finalTrace);
              setTurns((current) => current.some((turn) => turn.role === "assistant" && turn.run_id === response.run_id) ? current : [
                ...current,
                {
                  role: "assistant",
                  content: response.answer,
                  run_id: response.run_id,
                  route: response.route,
                  direct_answer_mode: response.direct_answer_mode,
                  citations: response.citations,
                  trace: finalTrace,
                  retrieval_trace_id: response.retrieval_trace_id,
                  context_package_id: response.context_package_id,
                },
              ]);
              setActiveStream(null);
              void queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
              void queryClient.invalidateQueries({ queryKey: ["session-messages", response.session_id] });
              setSessionReplayVersion((current) => current + 1);
            },
            onError: (message) => {
              if (!isCurrentStream()) return;
              recoverPersistedRun(message);
            },
          },
          { signal: controller.signal },
        );
        if (isCurrentStream()) {
          if (receivedRunId) {
            // EOF without final: recover from durable status instead of guessing success.
            void queryClient.invalidateQueries({ queryKey: ["agent-run", receivedRunId] });
          } else {
            setStreamError(productQaErrorMessage("connection closed before run metadata"));
            setDraftAnswer("");
            setActiveStream(null);
          }
        }
      } catch (error) {
        if (isCurrentStream()) {
          if (isAbortError(error)) {
            // Explicit cancellation detaches the controller ref before abort.
            // Reaching this branch therefore means the observer transport was
            // interrupted; the independently owned run must stay recoverable.
            recoverPersistedRun("stream observer closed", true);
          } else {
            recoverPersistedRun(
              error instanceof Error ? error.message : String(error),
              true,
            );
          }
        }
      } finally {
        if (streamAbortControllerRef.current === controller) {
          streamAbortControllerRef.current = null;
        }
      }
    },
  });

  const handleCancelActiveRun = () => {
    const runId = activeStream?.runId;
    const controller = streamAbortControllerRef.current;
    streamAbortControllerRef.current = null;
    controller?.abort();
    const viewRevision = ++runViewRevisionRef.current;
    setDraftAnswer("");
    setActiveStream(null);
    setStreamError(userCancelledMessage);
    if (runId) {
      cancelRunMutation.mutate({ runId, viewRevision });
    }
  };

  useEffect(() => {
    const status = runStatusQuery.data;
    if (!activeStream || !status || status.run_id !== activeStream.runId) {
      return;
    }
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      const runState = status.status ?? status.state;
      if (["completed", "needs_clarification", "failed", "cancelled"].includes(runState ?? "")) {
        finishRunFromStatus(status, status.session_id ?? activeStream.sessionId);
        return;
      }
      if (status.session_id) {
        setAutoResumeSuppressed(false);
        setActiveSessionId(status.session_id);
        if (status.session_id !== activeStream.sessionId) {
          setActiveStream((current) => (current ? { ...current, sessionId: status.session_id } : current));
        }
      }
      if (status.trace?.length) setTrace(status.trace);
    });
    return () => { cancelled = true; };
  }, [
    activeStream,
    finishRunFromStatus,
    runStatusQuery.data,
    setActiveSessionId,
    setAutoResumeSuppressed,
    setActiveStream,
  ]);

  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onMutate: async (sessionId) => {
      const queryKey = ["sessions", selectedKnowledgeBaseId] as const;
      await queryClient.cancelQueries({ queryKey });
      const previousSessions = queryClient.getQueryData<SessionSummary[]>(queryKey);
      queryClient.setQueryData<SessionSummary[]>(
        queryKey,
        (current) => current?.filter((session) => session.id !== sessionId) ?? [],
      );
      return { previousSessions, queryKey };
    },
    onError: (_error, _sessionId, context) => {
      if (context?.previousSessions) {
        queryClient.setQueryData(context.queryKey, context.previousSessions);
      }
    },
    onSuccess: (_data, sessionId) => {
      queryClient.removeQueries({ queryKey: ["session-messages", sessionId] });
      if (sessionId === activeSessionId) {
        runViewRevisionRef.current += 1;
        hydratedSessionIdRef.current = null;
        setActiveSessionId(null);
        setAutoResumeSuppressed(true);
        setTurns([]);
        setDraftAnswer("");
        setCitations([]);
        setCitationDrawerCitations([]);
        setTrace([]);
        setLatestRun(null);
        setConversationState(null);
        setActiveStream(null);
        setQuestion("");
      }
      void queryClient.invalidateQueries({
        queryKey: ["sessions", selectedKnowledgeBaseId],
      });
    },
  });

  const updateAgentSettingsForm = <K extends keyof AgentSettingsForm>(key: K, value: AgentSettingsForm[K]) => {
    setAgentSettingsSavedMessage(null);
    setAgentSettingsForm((current) => {
      const base = current ?? (modelSettingsQuery.data ? agentSettingsFormFromSettings(modelSettingsQuery.data) : null);
      return base ? { ...base, [key]: value } : current;
    });
  };

  const resetAgentSettingsForm = () => {
    setAgentSettingsSavedMessage(null);
    setAgentSettingsForm(modelSettingsQuery.data ? agentSettingsFormFromSettings(modelSettingsQuery.data) : null);
  };

  const openAgentSettingsDialog = () => {
    setAgentSettingsSavedMessage(null);
    setAgentSettingsForm(null);
    setAgentSettingsOpen(true);
    void modelSettingsQuery.refetch();
  };

  const isGenerating = Boolean(activeStream);
  const activeAgentSettingsForm = agentSettingsForm ?? (modelSettingsQuery.data ? agentSettingsFormFromSettings(modelSettingsQuery.data) : null);

  return (
    <div className="kg-page relative -mx-4 -my-5 min-h-[calc(100dvh-4.25rem)] px-4 pb-52 pt-5 lg:-mx-6 lg:-my-7 lg:px-6 lg:pt-7 xl:-mx-8 xl:px-8 2xl:-mx-10 2xl:px-10">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(86,217,255,0.11),transparent_34%),radial-gradient(circle_at_88%_20%,rgba(124,92,255,0.11),transparent_30%),linear-gradient(rgba(120,180,255,0.026)_1px,transparent_1px),linear-gradient(90deg,rgba(120,180,255,0.023)_1px,transparent_1px)] bg-[size:auto,auto,48px_48px,48px_48px]" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-[#030714] via-[#030714]/88 to-transparent" />
      <div className="relative z-10 flex flex-col gap-7">
        <ChatHeader
          latestRun={latestRun}
          configuredChatModel={modelSettingsQuery.data?.chat_model}
          modelLoading={modelSettingsQuery.isLoading}
        />
        <ChatActionRail
          onOpenSessions={() => setSessionsOpen(true)}
          onOpenCitations={() => openCitationDrawer(citations)}
          onOpenAgentSettings={openAgentSettingsDialog}
          citationsCount={citations.length}
        />

        <main className="mx-auto w-full max-w-6xl">
          {streamError ? <ErrorBlock message={streamError} /> : null}
          <ConversationStatePanel state={conversationState} />
          <MessageList
            turns={turns}
            isLoading={Boolean(selectedKnowledgeBaseId) && (
              sessionsQuery.isLoading || (
                hydratingSessionId !== null &&
                hydratingSessionId === activeSessionId
              )
            )}
            isGenerating={isGenerating}
            draftAnswer={draftAnswer}
            trace={trace}
            onOpenCitations={openCitationDrawer}
          />
        </main>

        <ChatComposer
          value={question}
          onChange={setQuestion}
          onSubmit={() => {
            if (isGenerating || streamAbortControllerRef.current) {
              return;
            }
            askMutation.mutate();
          }}
          onCancel={handleCancelActiveRun}
          isPending={isGenerating}
          activeSessionId={activeSessionId}
        />
      </div>

      <SessionsDrawer
        open={sessionsOpen}
        onOpenChange={setSessionsOpen}
        sessions={sessionsQuery.data ?? []}
        activeSessionId={activeSessionId}
        onDelete={(sessionId) => deleteSessionMutation.mutate(sessionId)}
        onSelect={async (sessionId) => {
          setSessionSelectionPending(true);
          try {
            const response = await queryClient.fetchQuery({
              queryKey: ["session-messages", sessionId],
              queryFn: () => fetchSessionMessages(sessionId),
              staleTime: Number.POSITIVE_INFINITY,
            });
            const nextTurns = normalizeMessages(response.messages, response.conversation_state);
            runViewRevisionRef.current += 1;
            setAutoResumeSuppressed(false);
            hydratedSessionIdRef.current = sessionId;
            setActiveSessionId(sessionId);
            setDraftAnswer("");
            setCitationDrawerCitations([]);
            setTrace([]);
            setLatestRun(null);
            setActiveStream(null);
            setTurns((current) => preserveTurnTraces(nextTurns, current));
            setConversationState(response.conversation_state);
            const latestAssistant = [...nextTurns].reverse().find((turn) => turn.role === "assistant");
            setCitations(latestAssistant?.citations ?? []);
            setStreamError(null);
          } catch (error) {
            if (responseStatus(error) === 404) {
              queryClient.setQueryData<SessionSummary[]>(
                ["sessions", selectedKnowledgeBaseId],
                (current) => current?.filter((session) => session.id !== sessionId),
              );
              await queryClient.invalidateQueries({
                queryKey: ["sessions", selectedKnowledgeBaseId],
              });
              if (activeSessionId === sessionId) {
                hydratedSessionIdRef.current = null;
                setActiveSessionId(null);
                setAutoResumeSuppressed(true);
                setTurns([]);
                setConversationState(null);
                setCitations([]);
                setTrace([]);
              }
              setStreamError(missingSessionMessage);
            } else {
              setStreamError(productQaErrorMessage(error));
            }
          } finally {
            setSessionSelectionPending(false);
          }
        }}
        onNew={() => {
          runViewRevisionRef.current += 1;
          setAutoResumeSuppressed(true);
          hydratedSessionIdRef.current = null;
          setActiveSessionId(null);
          setTurns([]);
          setDraftAnswer("");
          setCitations([]);
          setCitationDrawerCitations([]);
          setTrace([]);
          setLatestRun(null);
          setConversationState(null);
          setActiveStream(null);
          setQuestion("");
        }}
        isPending={isGenerating || sessionSelectionPending || deleteSessionMutation.isPending}
      />
      <AgentSettingsDialog
        open={agentSettingsOpen}
        onOpenChange={setAgentSettingsOpen}
        form={activeAgentSettingsForm}
        onChange={updateAgentSettingsForm}
        onReset={resetAgentSettingsForm}
        onSave={() => {
          if (activeAgentSettingsForm) {
            saveAgentSettingsMutation.mutate(buildAgentSettingsPayload(activeAgentSettingsForm));
          }
        }}
        isLoading={modelSettingsQuery.isLoading}
        error={modelSettingsQuery.error instanceof Error ? modelSettingsQuery.error : null}
        isSaving={saveAgentSettingsMutation.isPending}
        savedMessage={agentSettingsSavedMessage}
      />
      <CitationsDrawer open={citationsOpen} onOpenChange={setCitationsOpen} citations={citationDrawerCitations} />
    </div>
  );
}

export function QAWorkspace() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  return <QAWorkspaceContent key={selectedKnowledgeBaseId ?? "unassigned"} selectedKnowledgeBaseId={selectedKnowledgeBaseId} />;
}
