"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentResponse, AgentTraceEventPayload, Citation, ModelSettingsResponse, ModelSettingsUpdate, RetrievalGranularity, SessionSummary } from "@course-kg/shared";
import { motion } from "framer-motion";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  BrainCircuit,
  ChevronRight,
  CircleDot,
  FileText,
  History,
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
import { cancelAgentRun, deleteSession, fetchDashboard, fetchModelSettings, fetchSessionMessages, fetchSessions, fetchTaskStatus, streamAnswer, updateModelSettings } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  run_id?: string | null;
  route?: string | null;
  citations?: Citation[];
  trace?: AgentTraceEventPayload[];
};

type ActiveStreamState = {
  runId?: string | null;
  sessionId?: string | null;
  retrievalGranularity?: RetrievalGranularity;
  question: string;
  startedAt: string;
};

type AgentSettingsForm = {
  query_facet_bilingual_enabled: boolean;
  context_package_token_budget: string;
  retrieval_result_top_k_default: string;
  agent_coarse_total_budget: string;
  agent_mid_per_coarse_budget: string;
  agent_mid_top_k: string;
  agent_chunk_per_mid_budget: string;
  agent_chunk_top_k: string;
  candidate_pool_dedupe_budget: string;
  agent_max_depth_per_layer: string;
  agent_max_labels_per_node: string;
  agent_max_edge_reuse: string;
  agent_max_cycle_reward_per_path: string;
  agent_cycle_reward_distance_threshold: string;
  agent_path_distance_green_threshold: string;
  agent_path_distance_gray_threshold: string;
  agent_path_distance_hard_threshold: string;
  agent_structure_restore_budget: string;
  context_path_summary_budget: string;
  agent_planning_round_budget: string;
  agent_max_typed_actions_per_round: string;
  agent_repair_round_budget: string;
  agent_verification_budget: string;
};

type AgentNumberSettingKey = Exclude<keyof AgentSettingsForm, "query_facet_bilingual_enabled">;

type AgentNumberField = {
  key: AgentNumberSettingKey;
  label: string;
  min: number;
  max: number;
  step?: number;
};

const userCancelledMessage = "已取消当前对话";

const agentSettingsInputClass = "h-11 rounded-xl border-white/10 bg-white/[0.045] px-3 text-white placeholder:text-white/28";

const agentTraversalFields: AgentNumberField[] = [
  { key: "context_package_token_budget", label: "证据包 token 预算", min: 256, max: 20000 },
  { key: "retrieval_result_top_k_default", label: "结果 Top K 默认值", min: 1, max: 50 },
  { key: "agent_coarse_total_budget", label: "粗概念总预算", min: 1, max: 200 },
  { key: "agent_mid_per_coarse_budget", label: "每个粗概念中概念预算", min: 1, max: 100 },
  { key: "agent_mid_top_k", label: "中概念 Top K", min: 1, max: 500 },
  { key: "agent_chunk_per_mid_budget", label: "每个中概念片段预算", min: 1, max: 200 },
  { key: "agent_chunk_top_k", label: "片段 Top K", min: 1, max: 1000 },
  { key: "candidate_pool_dedupe_budget", label: "候选去重池预算", min: 1, max: 5000 },
];

const agentControlFields: AgentNumberField[] = [
  { key: "agent_max_depth_per_layer", label: "每层最大深度", min: 1, max: 12 },
  { key: "agent_max_labels_per_node", label: "每节点标签上限", min: 1, max: 20 },
  { key: "agent_max_edge_reuse", label: "边复用上限", min: 1, max: 20 },
  { key: "agent_max_cycle_reward_per_path", label: "Cycle reward 上限", min: 0, max: 2, step: 0.01 },
  { key: "agent_cycle_reward_distance_threshold", label: "Cycle reward 距离阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_green_threshold", label: "路径 green 阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_gray_threshold", label: "路径 gray 阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_hard_threshold", label: "路径 hard 阈值", min: 0, max: 40, step: 0.01 },
  { key: "agent_structure_restore_budget", label: "结构恢复预算", min: 1, max: 200 },
  { key: "context_path_summary_budget", label: "路径摘要预算", min: 1, max: 500 },
  { key: "agent_planning_round_budget", label: "规划轮次预算", min: 1, max: 10 },
  { key: "agent_max_typed_actions_per_round", label: "每轮动作上限", min: 1, max: 50 },
  { key: "agent_repair_round_budget", label: "修复轮次预算", min: 0, max: 10 },
  { key: "agent_verification_budget", label: "引用验证预算", min: 1, max: 100 },
];

const retrievalGranularityOptions: Array<{ value: RetrievalGranularity; label: string; description: string }> = [
  {
    value: "mid",
    label: "普通模式",
    description: "从中层概念直接进入检索，适合具体知识点和术语问题",
  },
  {
    value: "coarse",
    label: "摘要模式",
    description: "先从粗层摘要概念进入检索，适合总览和主题性问题",
  },
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
    query_facet_bilingual_enabled: settings?.query_facet_bilingual_enabled ?? false,
    context_package_token_budget: stringSetting(settings?.context_package_token_budget, 12000),
    retrieval_result_top_k_default: stringSetting(settings?.retrieval_result_top_k_default, 12),
    agent_coarse_total_budget: stringSetting(settings?.agent_coarse_total_budget, 5),
    agent_mid_per_coarse_budget: stringSetting(settings?.agent_mid_per_coarse_budget, 6),
    agent_mid_top_k: stringSetting(settings?.agent_mid_top_k, 8),
    agent_chunk_per_mid_budget: stringSetting(settings?.agent_chunk_per_mid_budget, 12),
    agent_chunk_top_k: stringSetting(settings?.agent_chunk_top_k, 16),
    candidate_pool_dedupe_budget: stringSetting(settings?.candidate_pool_dedupe_budget, 80),
    agent_max_depth_per_layer: stringSetting(settings?.agent_max_depth_per_layer, 3),
    agent_max_labels_per_node: stringSetting(settings?.agent_max_labels_per_node, 3),
    agent_max_edge_reuse: stringSetting(settings?.agent_max_edge_reuse, 2),
    agent_max_cycle_reward_per_path: stringSetting(settings?.agent_max_cycle_reward_per_path, 0.18),
    agent_cycle_reward_distance_threshold: stringSetting(settings?.agent_cycle_reward_distance_threshold, 1.2),
    agent_path_distance_green_threshold: stringSetting(settings?.agent_path_distance_green_threshold, 0.45),
    agent_path_distance_gray_threshold: stringSetting(settings?.agent_path_distance_gray_threshold, 1.35),
    agent_path_distance_hard_threshold: stringSetting(settings?.agent_path_distance_hard_threshold, 2.4),
    agent_structure_restore_budget: stringSetting(settings?.agent_structure_restore_budget, 16),
    context_path_summary_budget: stringSetting(settings?.context_path_summary_budget, 32),
    agent_planning_round_budget: stringSetting(settings?.agent_planning_round_budget, 2),
    agent_max_typed_actions_per_round: stringSetting(settings?.agent_max_typed_actions_per_round, 8),
    agent_repair_round_budget: stringSetting(settings?.agent_repair_round_budget, 2),
    agent_verification_budget: stringSetting(settings?.agent_verification_budget, 8),
  };
}

function buildAgentSettingsPayload(form: AgentSettingsForm): ModelSettingsUpdate {
  return {
    query_facet_bilingual_enabled: form.query_facet_bilingual_enabled,
    context_package_token_budget: parseIntField(form.context_package_token_budget),
    retrieval_result_top_k_default: parseIntField(form.retrieval_result_top_k_default),
    agent_coarse_total_budget: parseIntField(form.agent_coarse_total_budget),
    agent_mid_per_coarse_budget: parseIntField(form.agent_mid_per_coarse_budget),
    agent_mid_top_k: parseIntField(form.agent_mid_top_k),
    agent_chunk_per_mid_budget: parseIntField(form.agent_chunk_per_mid_budget),
    agent_chunk_top_k: parseIntField(form.agent_chunk_top_k),
    candidate_pool_dedupe_budget: parseIntField(form.candidate_pool_dedupe_budget),
    agent_max_depth_per_layer: parseIntField(form.agent_max_depth_per_layer),
    agent_max_labels_per_node: parseIntField(form.agent_max_labels_per_node),
    agent_max_edge_reuse: parseIntField(form.agent_max_edge_reuse),
    agent_max_cycle_reward_per_path: parseFloatField(form.agent_max_cycle_reward_per_path),
    agent_cycle_reward_distance_threshold: parseFloatField(form.agent_cycle_reward_distance_threshold),
    agent_path_distance_green_threshold: parseFloatField(form.agent_path_distance_green_threshold),
    agent_path_distance_gray_threshold: parseFloatField(form.agent_path_distance_gray_threshold),
    agent_path_distance_hard_threshold: parseFloatField(form.agent_path_distance_hard_threshold),
    agent_structure_restore_budget: parseIntField(form.agent_structure_restore_budget),
    context_path_summary_budget: parseIntField(form.context_path_summary_budget),
    agent_planning_round_budget: parseIntField(form.agent_planning_round_budget),
    agent_max_typed_actions_per_round: parseIntField(form.agent_max_typed_actions_per_round),
    agent_repair_round_budget: parseIntField(form.agent_repair_round_budget),
    agent_verification_budget: parseIntField(form.agent_verification_budget),
  };
}

const fallbackSuggestions = [
  "总结这批资料最核心的知识结构",
  "结合本地资料解释一个重要概念",
  "找出资料库中容易混淆的概念并比较",
  "基于资料引用给我一份阅读路线",
];

function answerModelLabel(latestRun: AgentResponse | null, configuredChatModel?: string | null): string {
  const audit = latestRun?.answer_model_audit;
  if (!audit) {
    return configuredChatModel ? `模型：${configuredChatModel}` : "模型：未读取";
  }
  if (audit.external_called) {
    return `模型：${audit.model ?? audit.chat_model ?? audit.provider}`;
  }
  if (audit.skipped_reason === "clarify_route") {
    return "模型：澄清分支未调用";
  }
  if (audit.skipped_reason === "direct_answer_route") {
    return "模型：直接回答分支未调用";
  }
  return "模型：未调用";
}

function buildKnowledgeBaseSuggestions(tree: Array<{ title?: string; children?: Array<{ title?: string }> }> | undefined): string[] {
  const partitions = tree?.map((node) => node.title).filter((title): title is string => Boolean(title)) ?? [];
  const documents = tree?.flatMap((node) => node.children?.map((child) => child.title) ?? []).filter((title): title is string => Boolean(title)) ?? [];
  const suggestions = [
    partitions[0] ? `总结 ${partitions[0]} 的核心内容` : "",
    partitions[1] ? `比较 ${partitions[0]} 和 ${partitions[1]} 的联系` : "",
    documents[0] ? `根据 ${documents[0]} 生成整理提纲` : "",
    partitions[0] ? `从本地资料中找出 ${partitions[0]} 的关键概念` : "",
  ].filter(Boolean);
  return suggestions.length ? suggestions.slice(0, 4) : fallbackSuggestions;
}

function normalizeMessages(messages: Array<Record<string, unknown>>): ChatTurn[] {
  return messages
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item) => ({
      role: item.role as "user" | "assistant",
      content: String(item.content ?? ""),
      run_id: typeof item.run_id === "string" ? item.run_id : null,
      citations: Array.isArray(item.citations) ? (item.citations as Citation[]) : undefined,
    }));
}

function ChatHeader({
  latestRun,
  configuredChatModel,
}: {
  latestRun: AgentResponse | null;
  configuredChatModel?: string | null;
}) {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-1">
      <div className="min-w-0">
        <p className="section-kicker">资料库智能问答</p>
        <h2 className="mt-1 text-2xl font-semibold text-white lg:text-3xl">向智能体提问</h2>
      </div>
      <div className="flex w-full flex-wrap gap-2">
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          <BrainCircuit data-icon="inline-start" />
          {answerModelLabel(latestRun, configuredChatModel)}
        </span>
      </div>
    </div>
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
      <span className="text-xs font-medium uppercase tracking-[0.16em] text-cyan-100/52">{field.label}</span>
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
      <DialogContent className="flex max-h-[calc(100vh-2rem)] w-[min(58rem,calc(100vw-2rem))] flex-col overflow-hidden border border-cyan-200/14 bg-[rgba(3,10,22,0.96)] p-0 text-white shadow-[0_30px_90px_rgba(0,0,0,0.48)] backdrop-blur-2xl sm:!max-w-[58rem]">
        <DialogHeader className="shrink-0 border-b border-cyan-200/10 px-6 py-5 pr-14">
          <DialogTitle className="flex items-center gap-2 text-lg text-white">
            <SlidersHorizontal className="size-5 text-cyan-100/78" />
            Agent 参数
          </DialogTitle>
          <DialogDescription className="text-cyan-50/58">保存后通过运行时热加载影响下一次检索、对话和引用验证。</DialogDescription>
        </DialogHeader>
        <form
          className="flex min-h-0 flex-1 flex-col"
          onSubmit={(event) => {
            event.preventDefault();
            onSave();
          }}
        >
          <ScrollArea className="min-h-0 flex-1 px-6 py-5">
            {isLoading ? <LoadingBlock rows={3} /> : null}
            {error ? <ErrorBlock message={error.message} /> : null}
            {form ? (
              <div className="grid gap-6">
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/8 pb-5">
                  <div className="min-w-[14rem] flex-1">
                    <p className="text-sm font-semibold text-white">LLM 双语查询面</p>
                    <p className="mt-1 text-sm leading-6 text-white/55">要求查询面提取为显式概念补充中英双语 aliases。</p>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={form.query_facet_bilingual_enabled}
                    disabled={disabled}
                    onClick={() => onChange("query_facet_bilingual_enabled", !form.query_facet_bilingual_enabled)}
                    className={`relative h-8 w-16 rounded-full border transition ${
                      form.query_facet_bilingual_enabled ? "border-cyan-100/40 bg-cyan-300/70" : "border-white/14 bg-white/10"
                    } disabled:cursor-not-allowed disabled:opacity-60`}
                  >
                    <span className={`absolute top-1 size-6 rounded-full bg-white shadow transition ${form.query_facet_bilingual_enabled ? "left-9" : "left-1"}`} />
                  </button>
                </div>

                <section className="grid gap-4">
                  <p className="text-sm font-semibold text-white">检索与证据包</p>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    {agentTraversalFields.map((field) => (
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
                  <p className="text-sm font-semibold text-white">遍历、修复与验证</p>
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
        title="Agent 参数"
        aria-label="Agent 参数"
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

function SuggestionChips({ suggestions, onPick }: { suggestions: string[]; onPick: (value: string) => void }) {
  return (
    <div className="mx-auto flex max-w-3xl flex-wrap justify-center gap-2 px-1">
      {suggestions.map((suggestion) => (
        <motion.button
          key={suggestion}
          type="button"
          whileHover={{ y: -2 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onPick(suggestion)}
          className="kg-micro-chip max-w-full rounded-full px-3 py-2 text-xs transition hover:border-cyan-200/30 hover:text-white sm:px-4 sm:text-sm"
        >
          {suggestion}
        </motion.button>
      ))}
    </div>
  );
}

function EmptyChatState({ suggestions, onPick }: { suggestions: string[]; onPick: (value: string) => void }) {
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
          系统会按粗概念、中概念、RQ 前缀和片段结构逐层寻址，组装上下文证据包后生成带引用的回答。
        </p>
        <div className="mt-7">
          <SuggestionChips suggestions={suggestions} onPick={onPick} />
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ turn, index, onOpenCitations }: { turn: ChatTurn; index: number; onOpenCitations: () => void }) {
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
          {isUser ? "你" : turn.route ? `智能体 / ${turn.route}` : "智能体"}
        </div>
        {!isUser && turn.trace?.length ? <AgentTraceStream trace={turn.trace} compact className="mb-5" /> : null}
        <MarkdownRenderer content={turn.content} className={cn(isUser ? "text-white/78" : "text-white/74")} />
        {!isUser && turn.citations?.length ? (
          <button type="button" onClick={onOpenCitations} className="kg-micro-chip mt-4 rounded-full px-3 py-2 text-xs transition hover:border-cyan-200/30 hover:text-white">
            <FileText />
            {turn.citations.length} 条已验证来源
          </button>
        ) : null}
      </div>
    </motion.div>
  );
}

function GeneratingBubble({ content, trace }: { content: string; trace: AgentTraceEventPayload[] }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-start">
      <div className="w-full max-w-[min(860px,92%)] border-l border-cyan-200/18 px-5 py-4 text-white">
        <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-cyan-100/50">
          <span className="tech-dot" />
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
            <span className="context-bars">
              <span />
              <span />
              <span />
              <span />
            </span>
            正在路由、检索并验证引用...
          </div>
        )}
      </div>
    </motion.div>
  );
}

function MessageList({
  turns,
  isGenerating,
  draftAnswer,
  trace,
  onPickSuggestion,
  onOpenCitations,
  suggestions,
}: {
  turns: ChatTurn[];
  isGenerating: boolean;
  draftAnswer: string;
  trace: AgentTraceEventPayload[];
  onPickSuggestion: (value: string) => void;
  onOpenCitations: () => void;
  suggestions: string[];
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const previousTurnCountRef = useRef(turns.length);

  useEffect(() => {
    const hasNewTurn = turns.length !== previousTurnCountRef.current;
    previousTurnCountRef.current = turns.length;
    const distanceToBottom = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
    const shouldStickToBottom = distanceToBottom < 280;
    if (!hasNewTurn && (!isGenerating || !shouldStickToBottom)) {
      return undefined;
    }
    if (scrollFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollFrameRef.current);
    }
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
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
      {turns.length === 0 && !isGenerating ? (
        <EmptyChatState suggestions={suggestions} onPick={onPickSuggestion} />
      ) : (
        <div className="mx-auto flex max-w-5xl flex-col gap-8 px-1 pb-6 pt-4">
          {turns.map((turn, index) => (
            <MessageBubble key={`${turn.role}-${index}-${turn.run_id ?? "local"}`} turn={turn} index={index} onOpenCitations={onOpenCitations} />
          ))}
          {isGenerating ? <GeneratingBubble content={draftAnswer} trace={trace} /> : null}
          <div ref={bottomRef} className="h-52 shrink-0 md:h-56" />
        </div>
      )}
    </div>
  );
}

export function RetrievalGranularitySelector({
  value,
  onChange,
  disabled,
}: {
  value: RetrievalGranularity;
  onChange: (value: RetrievalGranularity) => void;
  disabled?: boolean;
}) {
  const [tooltipMode, setTooltipMode] = useState<RetrievalGranularity | null>(null);
  const tooltipTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeOption = retrievalGranularityOptions.find((option) => option.value === value) ?? retrievalGranularityOptions[0];

  const clearTooltipTimer = () => {
    if (tooltipTimerRef.current) {
      clearTimeout(tooltipTimerRef.current);
      tooltipTimerRef.current = null;
    }
  };

  const queueTooltip = (mode: RetrievalGranularity) => {
    clearTooltipTimer();
    tooltipTimerRef.current = setTimeout(() => {
      setTooltipMode(mode);
      tooltipTimerRef.current = null;
    }, 1000);
  };

  const hideTooltip = () => {
    clearTooltipTimer();
    setTooltipMode(null);
  };

  useEffect(() => {
    return () => clearTooltipTimer();
  }, []);

  return (
    <div className="relative flex flex-wrap items-center gap-2" aria-label="检索模式">
      <div className="flex rounded-full border border-white/10 bg-black/16 p-1" role="group" aria-label="检索粒度模式">
        {retrievalGranularityOptions.map((option) => {
          const selected = option.value === value;
          return (
            <button
              key={option.value}
              type="button"
              data-testid={`retrieval-granularity-${option.value}`}
              aria-pressed={selected}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              onMouseEnter={() => queueTooltip(option.value)}
              onMouseLeave={hideTooltip}
              onFocus={() => queueTooltip(option.value)}
              onBlur={hideTooltip}
              className={cn(
                "inline-flex h-8 min-w-0 items-center gap-1.5 rounded-full px-3 text-[11px] font-medium transition disabled:cursor-not-allowed disabled:opacity-55",
                selected ? "bg-cyan-200 text-slate-950 shadow-[0_0_18px_rgba(86,217,255,0.18)]" : "text-white/58 hover:bg-white/8 hover:text-white/82",
              )}
            >
              {option.value === "mid" ? <Layers3 /> : <FileText />}
              <span className="whitespace-nowrap">{option.label}</span>
            </button>
          );
        })}
      </div>
      <span className="min-w-0 truncate text-[11px] text-white/42">{activeOption.value === "mid" ? "中层入口" : "粗层入口"}</span>
      {tooltipMode ? (
        <div
          role="tooltip"
          data-testid="retrieval-granularity-tooltip"
          className="absolute bottom-[calc(100%+0.5rem)] left-0 max-w-[min(22rem,calc(100vw-3rem))] rounded-lg border border-cyan-200/18 bg-[#071124] px-3 py-2 text-xs leading-5 text-white/70 shadow-[0_16px_42px_rgba(0,0,0,0.34)]"
        >
          {retrievalGranularityOptions.find((option) => option.value === tooltipMode)?.description}
        </div>
      ) : null}
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
  retrievalGranularity,
  onRetrievalGranularityChange,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  isPending: boolean;
  activeSessionId: string | null;
  retrievalGranularity: RetrievalGranularity;
  onRetrievalGranularityChange: (value: RetrievalGranularity) => void;
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
      className="pointer-events-none fixed inset-x-4 bottom-4 z-[45] lg:left-[calc(76px+1.75rem)] lg:right-7"
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
                会话 {activeSessionId ? activeSessionId.slice(0, 8) : "新建"}
              </span>
              <RetrievalGranularitySelector value={retrievalGranularity} onChange={onRetrievalGranularityChange} disabled={isPending} />
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
                placeholder="输入问题，系统会检索、评估、回答并给出引用..."
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
  onSelect: (sessionId: string) => void | Promise<void>;
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
                    onSelect(session.id);
                    onOpenChange(false);
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
      <SheetContent className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-xl">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>引用</SheetTitle>
          <SheetDescription>智能体回答使用的已验证证据卡片。</SheetDescription>
        </SheetHeader>
        <ScrollArea className="h-[calc(100dvh-8rem)] p-6">
          <div className="flex flex-col gap-3">
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

function QAWorkspaceContent({ selectedKnowledgeBaseId }: { selectedKnowledgeBaseId: string | null }) {
  const queryClient = useQueryClient();
  const storageScope = selectedKnowledgeBaseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedKnowledgeBaseId],
    queryFn: () => fetchDashboard(selectedKnowledgeBaseId, { includeGraph: false }),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const sessionsQuery = useQuery({
    queryKey: ["sessions", selectedKnowledgeBaseId],
    queryFn: () => fetchSessions(selectedKnowledgeBaseId),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const modelSettingsQuery = useQuery({ queryKey: ["model-settings"], queryFn: fetchModelSettings });
  const [question, setQuestion] = useLocalStorage(`qa.question.${storageScope}`, "");
  const [activeSessionId, setActiveSessionId] = useLocalStorage<string | null>(`qa.sessionId.${storageScope}`, null);
  const [turns, setTurns] = useLocalStorage<ChatTurn[]>(`qa.turns.${storageScope}`, []);
  const [draftAnswer, setDraftAnswer] = useLocalStorage(`qa.draftAnswer.${storageScope}`, "");
  const [citations, setCitations] = useLocalStorage<Citation[]>(`qa.citations.${storageScope}`, []);
  const [trace, setTrace] = useLocalStorage<AgentTraceEventPayload[]>(`qa.trace.${storageScope}`, []);
  const [latestRun, setLatestRun] = useLocalStorage<AgentResponse | null>(`qa.latestRun.${storageScope}`, null);
  const [activeStream, setActiveStream] = useLocalStorage<ActiveStreamState | null>(`qa.activeStream.${storageScope}`, null);
  const [retrievalGranularity, setRetrievalGranularity] = useLocalStorage<RetrievalGranularity>(`qa.retrievalGranularity.${storageScope}`, "mid");
  const [streamError, setStreamError] = useState<string | null>(null);
  const streamAbortControllerRef = useRef<AbortController | null>(null);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [citationsOpen, setCitationsOpen] = useState(false);
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [agentSettingsForm, setAgentSettingsForm] = useState<AgentSettingsForm | null>(null);
  const [agentSettingsSavedMessage, setAgentSettingsSavedMessage] = useState<{ kind: "success" | "error"; text: string } | null>(null);
  const activeRunId = activeStream?.runId ?? null;
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

  const cancelRunMutation = useMutation({
    mutationFn: (runId: string) => cancelAgentRun(runId),
    onSuccess: (status) => {
      if (status.trace?.length) {
        setTrace(status.trace);
      }
      const runState = status.status ?? status.state;
      if (runState === "failed") {
        setStreamError(status.error === "cancelled_by_user" ? userCancelledMessage : status.error ?? "回答生成已停止");
      }
      void queryClient.invalidateQueries({ queryKey: ["agent-run", status.run_id] });
    },
    onError: (error) => {
      setStreamError(error instanceof Error ? error.message : String(error));
    },
  });

  const askMutation = useMutation({
    mutationFn: async () => {
      const nextQuestion = question.trim();
      if (!nextQuestion) {
        return;
      }
      setStreamError(null);
      setDraftAnswer("");
      setCitations([]);
      setTrace([]);
      setLatestRun(null);
      setActiveStream({ question: nextQuestion, retrievalGranularity, startedAt: new Date().toISOString() });
      const controller = new AbortController();
      streamAbortControllerRef.current = controller;
      const nextTraceEvents: AgentTraceEventPayload[] = [];
      setTurns((current) => [...current, { role: "user", content: nextQuestion }]);
      setQuestion("");
      try {
        await streamAnswer(
          {
            question: nextQuestion,
            session_id: activeSessionId,
            knowledge_base_id: selectedKnowledgeBaseId,
            retrieval_granularity: retrievalGranularity,
          },
          {
            onTrace: (event) => {
              nextTraceEvents.push(event);
              setTrace((current) => [...current, event]);
            },
            onToken: (token) => setDraftAnswer((current) => `${current}${token}`),
            onCitations: (next) => setCitations(next),
            onMeta: (meta) => {
              if (meta.session_id) {
                setActiveSessionId(meta.session_id);
              }
              if (meta.run_id || meta.session_id) {
                setActiveStream((current) => ({
                  question: current?.question ?? nextQuestion,
                  startedAt: current?.startedAt ?? new Date().toISOString(),
                  retrievalGranularity: meta.retrieval_granularity ?? current?.retrievalGranularity ?? retrievalGranularity,
                  runId: meta.run_id ?? current?.runId ?? null,
                  sessionId: meta.session_id ?? current?.sessionId ?? null,
                }));
              }
            },
            onFinal: (response) => {
              const finalTrace = response.trace.length ? response.trace : nextTraceEvents;
              setLatestRun(response);
              setActiveSessionId(response.session_id);
              setCitations(response.citations);
              setDraftAnswer("");
              setTrace(finalTrace);
              setTurns((current) => [
                ...current,
                {
                  role: "assistant",
                  content: response.answer,
                  run_id: response.run_id,
                  route: response.route,
                  citations: response.citations,
                  trace: finalTrace,
                },
              ]);
              setActiveStream(null);
              void queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
              void queryClient.invalidateQueries({ queryKey: ["session-messages", response.session_id] });
            },
            onError: (message) => {
              setStreamError(message === "cancelled_by_user" ? userCancelledMessage : message);
              setActiveStream(null);
            },
          },
          { signal: controller.signal },
        );
      } catch (error) {
        setStreamError(isAbortError(error) ? userCancelledMessage : error instanceof Error ? error.message : String(error));
        setActiveStream(null);
      } finally {
        if (streamAbortControllerRef.current === controller) {
          streamAbortControllerRef.current = null;
        }
      }
    },
  });

  const handleCancelActiveRun = () => {
    const runId = activeStream?.runId;
    streamAbortControllerRef.current?.abort();
    setDraftAnswer("");
    setActiveStream(null);
    setStreamError(userCancelledMessage);
    if (runId) {
      cancelRunMutation.mutate(runId);
    }
  };

  useEffect(() => {
    const status = runStatusQuery.data;
    if (!activeStream || !status) {
      return;
    }
    if (status.session_id) {
      setActiveSessionId(status.session_id);
      if (status.session_id !== activeStream.sessionId) {
        setActiveStream((current) => (current ? { ...current, sessionId: status.session_id } : current));
      }
    }
    if (status.trace?.length) {
      setTrace(status.trace);
    }
    const runState = status.status ?? status.state;
    if (runState === "completed") {
      const sessionId = status.session_id ?? activeStream.sessionId;
      setDraftAnswer("");
      setActiveStream(null);
      if (sessionId) {
        void (async () => {
          const response = await fetchSessionMessages(sessionId);
          const nextTurns = normalizeMessages(response.messages);
          setTurns(nextTurns);
          const latestAssistant = [...nextTurns].reverse().find((turn) => turn.role === "assistant");
          setCitations(latestAssistant?.citations ?? []);
          await queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
          await queryClient.invalidateQueries({ queryKey: ["session-messages", sessionId] });
        })();
      } else if (status.answer) {
        setTurns((current) =>
          current.some((turn) => turn.run_id === status.run_id && turn.role === "assistant")
            ? current
            : [
                ...current,
                {
                  role: "assistant",
                  content: status.answer ?? "",
                  run_id: status.run_id,
                  route: status.route,
                  trace: status.trace ?? [],
                },
              ],
        );
      }
    } else if (runState === "failed") {
      window.queueMicrotask(() => {
        if (status.error === "cancelled_by_user") {
          setStreamError(userCancelledMessage);
          setActiveStream(null);
          return;
        }
        setStreamError(status.error ?? "回答生成失败");
        setActiveStream(null);
      });
    }
  }, [
    activeStream,
    queryClient,
    runStatusQuery.data,
    selectedKnowledgeBaseId,
    setActiveSessionId,
    setActiveStream,
    setCitations,
    setDraftAnswer,
    setTrace,
    setTurns,
  ]);

  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onSuccess: async (_data, sessionId) => {
      if (sessionId === activeSessionId) {
        setActiveSessionId(null);
        setTurns([]);
        setDraftAnswer("");
        setCitations([]);
        setTrace([]);
        setLatestRun(null);
        setActiveStream(null);
        setQuestion("");
      }
      await queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
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

  const suggestions = useMemo(() => buildKnowledgeBaseSuggestions(dashboardQuery.data?.tree), [dashboardQuery.data?.tree]);
  const isGenerating = askMutation.isPending || Boolean(activeStream);
  const activeAgentSettingsForm = agentSettingsForm ?? (modelSettingsQuery.data ? agentSettingsFormFromSettings(modelSettingsQuery.data) : null);

  if (dashboardQuery.isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (dashboardQuery.error) {
    return <ErrorBlock message={(dashboardQuery.error as Error).message} />;
  }

  return (
    <div className="kg-page relative -mx-4 -my-5 min-h-[calc(100dvh-4.25rem)] px-4 pb-52 pt-5 lg:-mx-7 lg:-my-7 lg:px-7 lg:pt-7">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(86,217,255,0.11),transparent_34%),radial-gradient(circle_at_88%_20%,rgba(124,92,255,0.11),transparent_30%),linear-gradient(rgba(120,180,255,0.026)_1px,transparent_1px),linear-gradient(90deg,rgba(120,180,255,0.023)_1px,transparent_1px)] bg-[size:auto,auto,48px_48px,48px_48px]" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-[#030714] via-[#030714]/88 to-transparent" />
      <div className="relative z-10 flex flex-col gap-7">
        <ChatHeader latestRun={latestRun} configuredChatModel={modelSettingsQuery.data?.chat_model} />
        <ChatActionRail
          onOpenSessions={() => setSessionsOpen(true)}
          onOpenCitations={() => setCitationsOpen(true)}
          onOpenAgentSettings={openAgentSettingsDialog}
          citationsCount={citations.length}
        />

        <main className="mx-auto w-full max-w-6xl">
          {streamError ? <ErrorBlock message={streamError} /> : null}
          <MessageList
            turns={turns}
            isGenerating={isGenerating}
            draftAnswer={draftAnswer}
            trace={trace}
            onPickSuggestion={setQuestion}
            onOpenCitations={() => setCitationsOpen(true)}
            suggestions={suggestions}
          />
        </main>

        <ChatComposer
          value={question}
          onChange={setQuestion}
          onSubmit={() => {
            if (isGenerating) {
              return;
            }
            askMutation.mutate();
          }}
          onCancel={handleCancelActiveRun}
          isPending={isGenerating}
          activeSessionId={activeSessionId}
          retrievalGranularity={retrievalGranularity}
          onRetrievalGranularityChange={setRetrievalGranularity}
        />
      </div>

      <SessionsDrawer
        open={sessionsOpen}
        onOpenChange={setSessionsOpen}
        sessions={sessionsQuery.data ?? []}
        activeSessionId={activeSessionId}
        onDelete={(sessionId) => deleteSessionMutation.mutate(sessionId)}
        onSelect={async (sessionId) => {
          setActiveSessionId(sessionId);
          setDraftAnswer("");
          setCitations([]);
          setTrace([]);
          setLatestRun(null);
          setActiveStream(null);
          const response = await fetchSessionMessages(sessionId);
          const nextTurns = normalizeMessages(response.messages);
          setTurns(nextTurns);
          const latestAssistant = [...nextTurns].reverse().find((turn) => turn.role === "assistant");
          setCitations(latestAssistant?.citations ?? []);
        }}
        onNew={() => {
          setActiveSessionId(null);
          setTurns([]);
          setDraftAnswer("");
          setCitations([]);
          setTrace([]);
          setLatestRun(null);
          setActiveStream(null);
          setQuestion("");
          setRetrievalGranularity("mid");
        }}
        isPending={isGenerating}
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
      <CitationsDrawer open={citationsOpen} onOpenChange={setCitationsOpen} citations={citations} />
    </div>
  );
}

export function QAWorkspace() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  return <QAWorkspaceContent key={selectedKnowledgeBaseId ?? "unassigned"} selectedKnowledgeBaseId={selectedKnowledgeBaseId} />;
}
