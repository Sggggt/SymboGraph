"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { AgentResponse, AgentTraceEventPayload, AnswerModelAudit, Citation, ConversationStatePayload, ModelSettingsResponse, ModelSettingsUpdate, RetrievalGranularity, SessionMessage, SessionSummary } from "@course-kg/shared";
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
import { cancelAgentRun, deleteSession, fetchDashboard, fetchModelSettings, fetchSessionMessages, fetchSessions, fetchTaskStatus, streamAnswer, updateModelSettings } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  run_id?: string | null;
  route?: string | null;
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
  retrievalGranularity?: RetrievalGranularity;
  question: string;
  startedAt: string;
};

type AgentSettingsForm = {
  query_facet_bilingual_enabled: boolean;
  query_facet_posterior_enabled: boolean;
  query_facet_posterior_observation_budget: string;
  query_facet_posterior_round_budget: string;
  query_facet_posterior_convergence_epsilon: string;
  context_package_token_budget: string;
  retrieval_result_top_k_default: string;
  agent_coarse_initial_budget: string;
  agent_coarse_top_k: string;
  agent_mid_per_coarse_budget: string;
  agent_coarse_drilldown_mid_initial_budget: string;
  agent_mid_initial_budget: string;
  agent_mid_top_k: string;
  agent_chunk_per_mid_budget: string;
  agent_chunk_initial_budget: string;
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
  traversal_observation_budget: string;
  agent_structure_restore_per_chunk_budget: string;
  context_path_summary_budget: string;
  agent_planning_round_budget: string;
  agent_max_typed_actions_per_round: string;
  agent_repair_round_budget: string;
  agent_verification_budget: string;
};

type AgentNumberSettingKey = Exclude<
  keyof AgentSettingsForm,
  "query_facet_bilingual_enabled" | "query_facet_posterior_enabled"
>;

type AgentNumberField = {
  key: AgentNumberSettingKey;
  label: string;
  min: number;
  max: number;
  step?: number;
};

const userCancelledMessage = "已取消当前对话";

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
  模型双语查询词面: "开启后，查询词面提取会要求模型为显式概念补充中英文别名；只影响下一次检索路由，不写入事实证据，也不触发图谱重建。",
  "Query facet posterior 观察预算": "单次检索最多读取多少条确定性候选观察来校准 facet 权重；这是 hot_reloadable hard cap，只影响下一次 search、QA 或 repair retrieval，不调用模型、不触发图谱重建。",
  "Query facet posterior 轮次预算": "当前协议最多使用 dense entry 与 merged chunk 两个固定 checkpoint；达到预算立即停止，不扩大 top-k 或模型预算。",
  "Query facet posterior 收敛阈值": "相邻两轮 posterior 的 L1 变化不超过该值时提前停止。posterior 只在相同未覆盖 facet 数量内做 tie-break，不是事实概率。",
  证据包令牌预算: "上下文证据包可容纳的令牌上限。证据包是回答生成的唯一证据输入，预算不足时会优先保留更强支撑。",
  结果保留数量默认值: "搜索、问答和智能体请求未显式指定结果数量时使用的默认返回上限；它不是裸召回规模。",
  粗概念起点数量: "摘要模式下从全部粗概念候选中选入图探索的起点数量；普通模式不使用这个参数。",
  粗概念保留数量: "摘要模式下粗概念图探索后保留并继续下钻的粗概念数量。",
  每个粗概念中概念预算: "对每个已保留粗概念分别下钻的中概念候选数量上限，保证逐父节点探索。",
  普通模式中概念起点数量: "普通模式下，从全体中概念候选池中选入中概念图探索的起点数量。",
  摘要模式中概念起点数量: "摘要模式下，从粗概念逐父节点下钻合并后的中概念候选池中选入中概念图探索的起点数量。",
  中概念保留数量: "中概念图探索后保留并继续下钻到片段层的中概念数量。",
  每个中概念片段预算: "对每个已保留中概念分别下钻到片段候选的数量上限，用来控制底层候选扩展范围。",
  片段起点数量: "从全部片段候选中选入片段图探索的起点数量。",
  片段最终保留数量: "片段图探索后最终保留进入证据包候选的片段数量。",
  候选去重池预算: "跨路径、跨 RQ 成员关系和跨概念合并候选时保留的候选池规模，防止单次检索过载。",
  每层最大深度: "图遍历在每个层级允许继续扩展的最大深度，用来避免路径无限扩张。",
  每节点标签上限: "同一节点可保留的路径标签数量上限，用于限制 dominance pruning 中的重复路径状态。",
  边复用上限: "同一条边在单条路径中允许复用的次数上限，防止环路反复放大。",
  闭环奖励上限: "单条路径最多可获得的闭环收敛奖励。奖励只辅助短而强的收敛路径，不能替代证据。",
  闭环奖励距离阈值: "只有总距离足够短的闭环路径才会得到收敛奖励，长而弱的环不会提升路径价值。",
  路径绿色阈值: "路径距离小于该值时视为高置信路径，通常可继续确定性扩展。",
  路径灰区阈值: "路径距离落入灰区时，由 executor 基于有界观测和版本化本地规则确定性裁决继续、下钻、走桥或停止；LLM 与证据评估器不参与、覆盖或补判。",
  路径硬中断阈值: "路径距离超过该值时执行器直接剪枝，不允许模型绕过硬阈值继续扩展。",
  扩展观察总预算: "单次图遍历允许持久化的完整 gray-zone 有界观测总量。达到预算后仍逐路径执行同一本地确定性规则，但只保留最小审计包并记录 hard interrupt；该参数 hot_reloadable，模型调用预算始终为 0。",
  每个片段结构恢复数量: "对每个最终命中片段最多追加多少前后文或桥接上下文；不改变片段检索命中数量。",
  路径摘要预算: "证据包中可保留的图路径摘要数量上限，用于解释证据从粗层到中层再到片段的来源。",
  规划轮次预算: "智能体可进行规划和评估的最大轮数，用来控制单次任务内的推理成本。",
  每轮动作上限: "每个规划轮最多允许的类型化动作数量。所有动作仍必须通过验证器和确定性执行器。",
  修复轮次预算: "引用缺失、桥接不足或结构上下文不足时允许的修复轮次；耗尽后只能返回已验证部分或证据不足说明。",
  引用验证预算: "回答后可执行的引用验证次数上限，用于把声明绑定回原始片段范围。",
};

const commonRetrievalFields: AgentNumberField[] = [
  { key: "context_package_token_budget", label: "证据包令牌预算", min: 256, max: 20000 },
  { key: "retrieval_result_top_k_default", label: "结果保留数量默认值", min: 1, max: 50 },
  { key: "agent_mid_top_k", label: "中概念保留数量", min: 1, max: 500 },
  { key: "agent_chunk_per_mid_budget", label: "每个中概念片段预算", min: 1, max: 200 },
  { key: "agent_chunk_initial_budget", label: "片段起点数量", min: 1, max: 1000 },
  { key: "agent_chunk_top_k", label: "片段最终保留数量", min: 1, max: 1000 },
  { key: "agent_structure_restore_per_chunk_budget", label: "每个片段结构恢复数量", min: 1, max: 200 },
  { key: "candidate_pool_dedupe_budget", label: "候选去重池预算", min: 1, max: 5000 },
];

const queryFacetPosteriorFields: AgentNumberField[] = [
  { key: "query_facet_posterior_observation_budget", label: "Query facet posterior 观察预算", min: 1, max: 1024 },
  { key: "query_facet_posterior_round_budget", label: "Query facet posterior 轮次预算", min: 1, max: 2 },
  { key: "query_facet_posterior_convergence_epsilon", label: "Query facet posterior 收敛阈值", min: 0, max: 1, step: 0.001 },
];

const midModeRetrievalFields: AgentNumberField[] = [
  { key: "agent_mid_initial_budget", label: "普通模式中概念起点数量", min: 1, max: 500 },
];

const coarseModeRetrievalFields: AgentNumberField[] = [
  { key: "agent_coarse_initial_budget", label: "粗概念起点数量", min: 1, max: 200 },
  { key: "agent_coarse_top_k", label: "粗概念保留数量", min: 1, max: 200 },
  { key: "agent_mid_per_coarse_budget", label: "每个粗概念中概念预算", min: 1, max: 100 },
  { key: "agent_coarse_drilldown_mid_initial_budget", label: "摘要模式中概念起点数量", min: 1, max: 500 },
];

const agentControlFields: AgentNumberField[] = [
  { key: "agent_max_depth_per_layer", label: "每层最大深度", min: 1, max: 12 },
  { key: "agent_max_labels_per_node", label: "每节点标签上限", min: 1, max: 20 },
  { key: "agent_max_edge_reuse", label: "边复用上限", min: 1, max: 20 },
  { key: "agent_max_cycle_reward_per_path", label: "闭环奖励上限", min: 0, max: 2, step: 0.01 },
  { key: "agent_cycle_reward_distance_threshold", label: "闭环奖励距离阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_green_threshold", label: "路径绿色阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_gray_threshold", label: "路径灰区阈值", min: 0, max: 20, step: 0.01 },
  { key: "agent_path_distance_hard_threshold", label: "路径硬中断阈值", min: 0, max: 40, step: 0.01 },
  { key: "traversal_observation_budget", label: "扩展观察总预算", min: 1, max: 20000 },
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
    query_facet_posterior_enabled: settings?.query_facet_posterior_enabled ?? true,
    query_facet_posterior_observation_budget: stringSetting(settings?.query_facet_posterior_observation_budget, 64),
    query_facet_posterior_round_budget: stringSetting(settings?.query_facet_posterior_round_budget, 2),
    query_facet_posterior_convergence_epsilon: stringSetting(settings?.query_facet_posterior_convergence_epsilon, 0.02),
    context_package_token_budget: stringSetting(settings?.context_package_token_budget, 12000),
    retrieval_result_top_k_default: stringSetting(settings?.retrieval_result_top_k_default, 12),
    agent_coarse_initial_budget: stringSetting(settings?.agent_coarse_initial_budget ?? settings?.agent_coarse_total_budget, 5),
    agent_coarse_top_k: stringSetting(settings?.agent_coarse_top_k ?? settings?.agent_coarse_initial_budget ?? settings?.agent_coarse_total_budget, 5),
    agent_mid_per_coarse_budget: stringSetting(settings?.agent_mid_per_coarse_budget, 6),
    agent_coarse_drilldown_mid_initial_budget: stringSetting(
      settings?.agent_coarse_drilldown_mid_initial_budget ?? settings?.agent_mid_top_k,
      8
    ),
    agent_mid_initial_budget: stringSetting(settings?.agent_mid_initial_budget ?? settings?.agent_mid_top_k, 8),
    agent_mid_top_k: stringSetting(settings?.agent_mid_top_k, 8),
    agent_chunk_per_mid_budget: stringSetting(settings?.agent_chunk_per_mid_budget, 12),
    agent_chunk_initial_budget: stringSetting(settings?.agent_chunk_initial_budget ?? settings?.agent_chunk_top_k, 16),
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
    traversal_observation_budget: stringSetting(settings?.traversal_observation_budget, 64),
    agent_structure_restore_per_chunk_budget: stringSetting(settings?.agent_structure_restore_per_chunk_budget ?? settings?.agent_structure_restore_budget, 16),
    context_path_summary_budget: stringSetting(settings?.context_path_summary_budget, 32),
    agent_planning_round_budget: stringSetting(settings?.agent_planning_round_budget, 2),
    agent_max_typed_actions_per_round: stringSetting(settings?.agent_max_typed_actions_per_round, 8),
    agent_repair_round_budget: stringSetting(settings?.agent_repair_round_budget, 2),
    agent_verification_budget: stringSetting(settings?.agent_verification_budget, 8),
  };
}

function buildAgentSettingsPayload(form: AgentSettingsForm, retrievalGranularity: RetrievalGranularity): ModelSettingsUpdate {
  const payload: ModelSettingsUpdate = {
    query_facet_bilingual_enabled: form.query_facet_bilingual_enabled,
    query_facet_posterior_enabled: form.query_facet_posterior_enabled,
    query_facet_posterior_observation_budget: parseIntField(form.query_facet_posterior_observation_budget),
    query_facet_posterior_round_budget: parseIntField(form.query_facet_posterior_round_budget),
    query_facet_posterior_convergence_epsilon: parseFloatField(form.query_facet_posterior_convergence_epsilon),
    context_package_token_budget: parseIntField(form.context_package_token_budget),
    retrieval_result_top_k_default: parseIntField(form.retrieval_result_top_k_default),
    agent_mid_top_k: parseIntField(form.agent_mid_top_k),
    agent_chunk_per_mid_budget: parseIntField(form.agent_chunk_per_mid_budget),
    agent_chunk_initial_budget: parseIntField(form.agent_chunk_initial_budget),
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
    traversal_observation_budget: parseIntField(form.traversal_observation_budget),
    agent_structure_restore_per_chunk_budget: parseIntField(form.agent_structure_restore_per_chunk_budget),
    context_path_summary_budget: parseIntField(form.context_path_summary_budget),
    agent_planning_round_budget: parseIntField(form.agent_planning_round_budget),
    agent_max_typed_actions_per_round: parseIntField(form.agent_max_typed_actions_per_round),
    agent_repair_round_budget: parseIntField(form.agent_repair_round_budget),
    agent_verification_budget: parseIntField(form.agent_verification_budget),
  };
  if (retrievalGranularity === "coarse") {
    payload.agent_coarse_initial_budget = parseIntField(form.agent_coarse_initial_budget);
    payload.agent_coarse_top_k = parseIntField(form.agent_coarse_top_k);
    payload.agent_mid_per_coarse_budget = parseIntField(form.agent_mid_per_coarse_budget);
    payload.agent_coarse_drilldown_mid_initial_budget = parseIntField(form.agent_coarse_drilldown_mid_initial_budget);
  } else {
    payload.agent_mid_initial_budget = parseIntField(form.agent_mid_initial_budget);
  }
  return payload;
}

const fallbackSuggestions = [
  "总结这批资料最核心的知识结构",
  "结合本地资料解释一个重要概念",
  "找出资料库中容易混淆的概念并比较",
  "基于资料引用给我一份阅读路线",
];

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

function buildKnowledgeBaseSuggestions(tree: Array<{ title?: string; children?: Array<{ title?: string }> }> | undefined): string[] {
  const isProductTitle = (title: string | undefined): title is string =>
    Boolean(
      title &&
        !/^[0-9a-f]{64}$/i.test(title) &&
        !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(title),
    );
  const partitions = tree?.map((node) => node.title).filter(isProductTitle) ?? [];
  const documents = tree?.flatMap((node) => node.children?.map((child) => child.title) ?? []).filter(isProductTitle) ?? [];
  const suggestions = [
    partitions[0] ? `总结 ${partitions[0]} 的核心内容` : "",
    partitions[1] ? `比较 ${partitions[0]} 和 ${partitions[1]} 的联系` : "",
    documents[0] ? `根据 ${documents[0]} 生成整理提纲` : "",
    partitions[0] ? `从本地资料中找出 ${partitions[0]} 的关键概念` : "",
  ].filter(Boolean);
  return suggestions.length ? suggestions.slice(0, 4) : fallbackSuggestions;
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
      const historyReference = runId ? referencesByRunId.get(runId) : undefined;
      return {
        role: item.role as "user" | "assistant",
        content: String(item.content ?? ""),
        run_id: runId,
        route: typeof item.route === "string" ? item.route : null,
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

async function hydrateHistoricalTurnTraces(turns: ChatTurn[]): Promise<ChatTurn[]> {
  const hydrated = turns.map((turn) => ({ ...turn }));
  const candidates = hydrated
    .map((turn, index) => ({ turn, index }))
    .filter(({ turn }) => turn.role === "assistant" && Boolean(turn.run_id) && !turn.trace?.length)
    .slice(-8);
  for (const { turn, index } of candidates) {
    try {
      const status = await fetchTaskStatus(turn.run_id as string);
      if (status.trace?.length) {
        hydrated[index] = { ...turn, trace: status.trace };
      }
    } catch {
      // Historical trace replay is an optional product projection. The
      // persisted answer and citations remain usable when a run is too old.
    }
  }
  return hydrated;
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
  retrievalGranularity,
  onRetrievalGranularityChange,
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
  retrievalGranularity: RetrievalGranularity;
  onRetrievalGranularityChange: (value: RetrievalGranularity) => void;
  isLoading: boolean;
  error: Error | null;
  isSaving: boolean;
  savedMessage: { kind: "success" | "error"; text: string } | null;
}) {
  const disabled = isLoading || isSaving || !form;
  const modeFields = retrievalGranularity === "mid" ? midModeRetrievalFields : coarseModeRetrievalFields;
  const modeTitle = retrievalGranularity === "mid" ? "普通模式入口参数" : "摘要模式入口参数";
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[min(46rem,calc(100dvh-2rem))] max-h-[calc(100dvh-2rem)] w-[min(58rem,calc(100vw-2rem))] flex-col overflow-hidden border border-cyan-200/14 bg-[rgba(3,10,22,0.96)] p-0 text-white shadow-[0_30px_90px_rgba(0,0,0,0.48)] backdrop-blur-2xl sm:!max-w-[58rem]">
        <DialogHeader className="shrink-0 border-b border-cyan-200/10 px-6 py-5 pr-14">
          <DialogTitle className="flex items-center gap-2 text-lg text-white">
            <SlidersHorizontal className="size-5 text-cyan-100/78" />
            智能体参数
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
          <ScrollArea className="min-h-0 flex-1 px-6 py-5 pr-4">
            {isLoading ? <LoadingBlock rows={3} /> : null}
            {error ? <ErrorBlock message={error.message} /> : null}
            {form ? (
              <div className="grid gap-6">
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/8 pb-5">
                  <div className="min-w-[14rem] flex-1">
                    <p className="text-sm font-semibold text-white">
                      <AgentParameterName label="模型双语查询词面" className="text-sm font-semibold normal-case tracking-normal text-white" />
                    </p>
                    <p className="mt-1 text-sm leading-6 text-white/55">要求查询词面提取为显式概念补充中英文别名。</p>
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

                <section className="grid gap-4 rounded-xl border border-cyan-200/12 bg-cyan-200/[0.025] p-4">
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="min-w-[14rem] flex-1">
                      <p className="text-sm font-semibold text-white">Query facet posterior calibration</p>
                      <p className="mt-1 text-xs leading-5 text-white/52">
                        hot_reloadable · 影响下一次 search / QA / repair retrieval · 不触发切块、Qdrant 或图谱重建。仅使用确定性有界图观察，LLM sample budget 固定为 0。
                      </p>
                    </div>
                    <button
                      type="button"
                      role="switch"
                      aria-label="Query facet posterior calibration"
                      aria-checked={form.query_facet_posterior_enabled}
                      disabled={disabled}
                      onClick={() => onChange("query_facet_posterior_enabled", !form.query_facet_posterior_enabled)}
                      className={`relative h-8 w-16 rounded-full border transition ${
                        form.query_facet_posterior_enabled ? "border-cyan-100/40 bg-cyan-300/70" : "border-white/14 bg-white/10"
                      } disabled:cursor-not-allowed disabled:opacity-60`}
                    >
                      <span className={`absolute top-1 size-6 rounded-full bg-white shadow transition ${form.query_facet_posterior_enabled ? "left-9" : "left-1"}`} />
                    </button>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {queryFacetPosteriorFields.map((field) => (
                      <AgentSettingsField
                        key={field.key}
                        field={field}
                        value={form[field.key]}
                        onChange={(value) => onChange(field.key, value)}
                        disabled={disabled || !form.query_facet_posterior_enabled}
                      />
                    ))}
                  </div>
                  <p className="text-xs leading-5 text-amber-100/62">
                    posterior 不是事实证据、引用来源或 gray-zone authority；它只能在未覆盖 facet 数量相同的候选之间做确定性 tie-break。
                  </p>
                </section>

                <section className="grid gap-3 rounded-lg border border-white/8 bg-white/[0.025] p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-[14rem]">
                      <p className="text-sm font-semibold text-white">检索模式</p>
                      <p className="mt-1 text-xs leading-5 text-white/48">这里只显示当前模式会读取的入口预算参数。</p>
                    </div>
                    <RetrievalGranularitySelector value={retrievalGranularity} onChange={onRetrievalGranularityChange} disabled={disabled} />
                  </div>
                </section>

                <section className="grid gap-4">
                  <p className="text-sm font-semibold text-white">检索与证据包</p>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    {commonRetrievalFields.map((field) => (
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

                {modeFields.length ? (
                  <section className="grid gap-4">
                    <p className="text-sm font-semibold text-white">{modeTitle}</p>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                      {modeFields.map((field) => (
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
                ) : null}

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
          系统会从资料中寻找相关内容、补充必要上下文，并生成带来源的回答。
        </p>
        <div className="mt-7">
          <SuggestionChips suggestions={suggestions} onPick={onPick} />
        </div>
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
            正在查找资料并核对来源...
          </div>
        )}
      </div>
    </motion.div>
  );
}

export function MessageList({
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
  onOpenCitations: (citations: Citation[]) => void;
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
      <span className="min-w-0 truncate text-[11px] text-white/42">{activeOption.value === "mid" ? "适合具体问题" : "适合主题总览"}</span>
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
                {activeSessionId ? "会话已建立" : "新建会话"}
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
          <SheetDescription>查看回答所依据的资料片段、页码和章节位置。</SheetDescription>
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
  // PostgreSQL session/answer/trace rows are the durable conversation source.
  // Large citations, trace audits and AgentResponse payloads must stay in
  // memory; persisting them redundantly in localStorage exceeds browser quota
  // and can prevent the final turn from rendering.
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [trace, setTrace] = useState<AgentTraceEventPayload[]>([]);
  const [latestRun, setLatestRun] = useState<AgentResponse | null>(null);
  const [conversationState, setConversationState] = useState<ConversationStatePayload | null>(null);
  const [activeStream, setActiveStream] = useLocalStorage<ActiveStreamState | null>(`qa.activeStream.${storageScope}`, null);
  const [retrievalGranularity, setRetrievalGranularity] = useLocalStorage<RetrievalGranularity>(`qa.retrievalGranularity.${storageScope}`, "mid");
  const [streamError, setStreamError] = useState<string | null>(null);
  const streamAbortControllerRef = useRef<AbortController | null>(null);
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

  useEffect(() => {
    if (
      !activeSessionId
      || activeRunId
      || hydratedSessionIdRef.current === activeSessionId
    ) {
      return;
    }
    let cancelled = false;
    hydratedSessionIdRef.current = activeSessionId;
    void (async () => {
      try {
        const response = await fetchSessionMessages(activeSessionId);
        const nextTurns = await hydrateHistoricalTurnTraces(
          normalizeMessages(response.messages, response.conversation_state),
        );
        if (cancelled) {
          return;
        }
        setTurns(nextTurns);
        setConversationState(response.conversation_state);
        const latestAssistant = [...nextTurns]
          .reverse()
          .find((turn) => turn.role === "assistant");
        setCitations(latestAssistant?.citations ?? []);
        setTrace(latestAssistant?.trace ?? []);
      } catch (error) {
        if (!cancelled) {
          hydratedSessionIdRef.current = null;
          setStreamError(productQaErrorMessage(error));
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
  }, [activeRunId, activeSessionId]);

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
      if (runState === "failed" || runState === "cancelled") {
        setStreamError(productQaErrorMessage(status.error ?? "回答生成已停止"));
      }
      void queryClient.invalidateQueries({ queryKey: ["agent-run", status.run_id] });
      void queryClient.invalidateQueries({ queryKey: ["agent-pe-audit", status.run_id] });
    },
    onError: (error) => {
      setStreamError(productQaErrorMessage(error));
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
      setCitationDrawerCitations([]);
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
              setConversationState(response.conversation_state ?? null);
              hydratedSessionIdRef.current = response.session_id;
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
                  retrieval_trace_id: response.retrieval_trace_id,
                  context_package_id: response.context_package_id,
                },
              ]);
              setActiveStream(null);
              void queryClient.invalidateQueries({ queryKey: ["agent-pe-audit", response.run_id] });
              void queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
              void queryClient.invalidateQueries({ queryKey: ["session-messages", response.session_id] });
            },
            onError: (message) => {
              setStreamError(productQaErrorMessage(message));
              setActiveStream(null);
            },
          },
          { signal: controller.signal },
        );
      } catch (error) {
        setStreamError(isAbortError(error) ? userCancelledMessage : productQaErrorMessage(error));
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
      const statusTrace = status.trace;
      queueMicrotask(() => setTrace(statusTrace));
    }
    const runState = status.status ?? status.state;
    if (runState === "completed") {
      void queryClient.invalidateQueries({ queryKey: ["agent-pe-audit", status.run_id] });
      const sessionId = status.session_id ?? activeStream.sessionId;
      queueMicrotask(() => {
        setDraftAnswer("");
        setActiveStream(null);
        if (!sessionId && status.answer) {
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
      });
      if (sessionId) {
        void (async () => {
          const response = await fetchSessionMessages(sessionId);
          const nextTurns = normalizeMessages(response.messages, response.conversation_state);
          const statusBoundTurns = nextTurns.map((turn) => (
            turn.role === "assistant" && turn.run_id === status.run_id && status.trace?.length
              ? { ...turn, trace: status.trace }
              : turn
          ));
          setTurns((current) => preserveTurnTraces(statusBoundTurns, current));
          hydratedSessionIdRef.current = sessionId;
          setConversationState(response.conversation_state);
          const latestAssistant = [...nextTurns].reverse().find((turn) => turn.role === "assistant");
          setCitations(latestAssistant?.citations ?? []);
          await queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] });
          await queryClient.invalidateQueries({ queryKey: ["session-messages", sessionId] });
        })();
      }
    } else if (runState === "failed" || runState === "cancelled") {
      window.queueMicrotask(() => {
        if (status.error === "cancelled_by_user") {
          setStreamError(userCancelledMessage);
          setActiveStream(null);
          return;
        }
        setStreamError(productQaErrorMessage(status.error ?? "回答生成失败"));
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
    setConversationState,
    setDraftAnswer,
    setTrace,
    setTurns,
  ]);

  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onSuccess: async (_data, sessionId) => {
      if (sessionId === activeSessionId) {
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
          onOpenCitations={() => openCitationDrawer(citations)}
          onOpenAgentSettings={openAgentSettingsDialog}
          citationsCount={citations.length}
        />

        <main className="mx-auto w-full max-w-6xl">
          {streamError ? <ErrorBlock message={streamError} /> : null}
          <ConversationStatePanel state={conversationState} />
          <MessageList
            turns={turns}
            isGenerating={isGenerating}
            draftAnswer={draftAnswer}
            trace={trace}
            onPickSuggestion={setQuestion}
            onOpenCitations={openCitationDrawer}
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
          hydratedSessionIdRef.current = sessionId;
          setActiveSessionId(sessionId);
          setDraftAnswer("");
          setCitations([]);
          setCitationDrawerCitations([]);
          setTrace([]);
          setLatestRun(null);
          setActiveStream(null);
          const response = await fetchSessionMessages(sessionId);
          const nextTurns = await hydrateHistoricalTurnTraces(
            normalizeMessages(response.messages, response.conversation_state),
          );
          setTurns((current) => preserveTurnTraces(nextTurns, current));
          setConversationState(response.conversation_state);
          const latestAssistant = [...nextTurns].reverse().find((turn) => turn.role === "assistant");
          setCitations(latestAssistant?.citations ?? []);
        }}
        onNew={() => {
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
        retrievalGranularity={retrievalGranularity}
        onRetrievalGranularityChange={setRetrievalGranularity}
        onSave={() => {
          if (activeAgentSettingsForm) {
            saveAgentSettingsMutation.mutate(buildAgentSettingsPayload(activeAgentSettingsForm, retrievalGranularity));
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
