"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ModelSettingsUpdate, RuntimeIssue, StrategyProfileDetail, StructuredApiErrorBody } from "@course-kg/shared";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Copy,
  Bot,
  EyeOff,
  FilePlus2,
  Info,
  KeyRound,
  Loader2,
  PencilLine,
  RotateCcw,
  Save,
  Send,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  XCircle,
} from "lucide-react";

import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  bindStrategyProfile,
  copyStrategyProfile,
  createStrategyProfile,
  deleteStrategyProfile,
  fetchModelSettings,
  fetchRuntimeCheck,
  fetchStrategyProfile,
  fetchStrategyProfiles,
  streamProfileAssistant,
  updateModelSettings,
  updateStrategyProfile,
} from "@/lib/api";

type SettingsForm = {
  chat_base_url: string;
  embedding_base_url: string;
  chat_resolve_ip: string;
  embedding_resolve_ip: string;
  embedding_model: string;
  chat_model: string;
  embedding_dimensions: string;
  embedding_batch_size: string;
  worker_concurrency: string;
  model_request_concurrency: string;
  model_request_timeout_seconds: string;
  concept_i18n_enabled: boolean;
  fixed_chunk_size_tokens: string;
  fixed_chunk_overlap_tokens: string;
  context_package_token_budget: string;
  api_key: string;
  clear_api_key: boolean;
  embedding_api_key: string;
  clear_embedding_api_key: boolean;
  model_bridge_enabled: boolean;
  mid_concept_extraction_max_model_batches: string;
  mid_concept_extraction_max_candidates_per_batch: string;
  mid_concept_extraction_max_tokens_per_batch: string;
  mid_concept_candidate_keep_threshold: string;
  rq_kmeans_levels: string;
  rq_kmeans_max_k: string;
  rq_residual_tau: string;
  dense_knn_k_min: string;
  dense_knn_k_max: string;
  dense_reverse_b_min_base: string;
  dense_reverse_b_max_base: string;
  dense_reverse_b_min_doc: string;
  dense_reverse_b_max_doc: string;
  dense_reverse_b_min_lang: string;
  dense_reverse_b_max_lang: string;
  dense_min_cosine: string;
  dense_strong_cosine: string;
  cross_doc_out_quota_min: string;
  cross_doc_out_quota_max: string;
  cross_doc_min_cosine: string;
  cross_language_out_quota_min: string;
  cross_language_out_quota_max: string;
  cross_language_min_cosine: string;
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

type ErrorDialogState = {
  title: string;
  message: string;
  status?: number;
  issues: RuntimeIssue[];
  fixCommands: string[];
};

type FieldProps = {
  label: string;
  description?: string;
  value: string;
  onChange: (value: string) => void;
  type?: "text" | "number" | "password";
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
};

type SwitchRowProps = {
  title: string;
  tooltip?: string;
  description: string;
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  badge?: string;
};

const inputClass = "h-11 rounded-xl border-white/10 bg-white/[0.04] px-3 text-white placeholder:text-white/28";
const sectionClass = "rounded-2xl border border-white/10 bg-white/[0.035] p-5";
const parameterNameClass = "text-xs uppercase tracking-[0.2em] text-cyan-100/46";

export const SETTINGS_PARAMETER_HELP: Record<string, string> = {
  资料库类型: "标记当前配置档适用的资料库类别，只影响提示词、界面标签和对话偏好，不参与切块、构图或检索参数。",
  名称: "配置档在设置页和资料库绑定列表里的显示名称，便于区分不同交互风格。",
  模型桥: "开启后 API 和 worker 容器优先通过本机模型桥访问聊天与向量端点，适合宿主机运行本地模型服务的场景。",
  聊天基础地址: "OpenAI 兼容聊天接口的 base URL；保存后影响下一次回答生成、Planner、Evaluator 和概念命名调用。",
  向量基础地址: "Embedding 接口的 base URL；后续解析、重嵌入和图谱重建会用它生成 contextual embedding。",
  "聊天 DNS 覆盖 IP": "仅对聊天端点使用的 DNS 覆盖；需要固定解析到指定 IP 时填写，留空则使用系统 DNS。",
  "向量 DNS 覆盖 IP": "仅对向量端点使用的 DNS 覆盖；需要固定解析到指定 IP 时填写，留空则使用系统 DNS。",
  聊天模型: "用于回答生成、Agent 规划/判停、概念命名和引用辅助判断的聊天模型名称。",
  向量模型: "用于资料 embedding、dense relation 候选和查询向量的模型名称；改变后已有向量需要显式重解析或重建。",
  聊天接口密钥: "聊天模型端点的访问密钥。留空会保留已有密钥，页面不会回显真实密钥。",
  向量接口密钥: "Embedding 端点的访问密钥。留空会保留已有密钥，页面不会回显真实密钥。",
  清除当前聊天接口密钥: "勾选后保存会删除当前聊天密钥；删除后聊天模型调用会因缺少凭据而失败。",
  清除当前向量接口密钥: "勾选后保存会删除当前向量密钥；删除后解析、重嵌入和检索向量生成会因缺少凭据而失败。",
  模型请求并发: "限制同时发起的模型请求数量，用于控制概念生成、Agent 判断和回答生成的吞吐与外部端点压力。",
  模型超时秒数: "单次模型请求等待上限；超过该时间会快速失败并进入可诊断错误，不做静默降级。",
  "Embedding 批大小": "每批提交给向量端点的文本数量；较大批次提升吞吐，但会增加单次请求体积和失败重试成本。",
  "证据包 token 预算": "Context Package 可容纳的证据 token 上限；它约束进入回答生成的唯一证据输入规模。",
  中粗层双语派生: "开启后，下一次图谱重建会对 mid/coarse 概念节点和高层概念边额外生成中英双语派生 metadata；关闭时不会产生这部分模型调用成本。",
  粗概念总预算: "Layered retrieval 在 coarse 层探索的粗概念节点上限；这是层内 hard interrupt，不是相关性评分。",
  每个粗概念中概念预算: "对每个已接受粗概念分别下钻的 mid candidate 数量上限，保证逐父节点探索而不是全局裸 top-k。",
  "中概念 Top K": "所有 mid candidates 合并去重后的层间输出上限；不会绕过 trace、结构恢复或引用验证。",
  每个中概念片段预算: "对每个已选中概念分别下钻到 chunk candidate 的数量上限，控制底层候选扩展范围。",
  "片段 Top K": "chunk candidates 合并去重后进入 Context Package 候选的输出上限，不等同于裸向量召回 top-k。",
  候选去重池预算: "限制跨路径、跨 RQ membership 和跨概念候选合并去重时保留的候选池规模，防止单次检索过载。",
  每层最大深度: "图遍历在每个层级允许继续扩展的最大深度，避免路径无限扩张。",
  每节点标签上限: "每个节点参与 dominance pruning 的路径标签数量上限，用来控制同一节点上的重复路径状态。",
  边复用上限: "同一条图边在单条路径中可被重复使用的次数上限，防止环路反复放大。",
  "Cycle reward 上限": "同一条路径最多获得的环收敛奖励；奖励只辅助短而强的收敛路径，不能替代证据。",
  "Cycle reward 距离阈值": "只有总距离足够短的环才会得到收敛奖励，长而弱的环不会提升路径价值。",
  "路径 green 阈值": "路径距离小于该值时视为可继续的高置信路径，通常不需要 LLM 灰区判停。",
  "路径 gray 阈值": "路径距离落在 green 与 gray 之间时进入灰区，可由 LLM evaluator 输出 typed decision。",
  "路径 hard 阈值": "路径距离超过该值时 executor 直接剪枝，不允许 LLM 绕过硬阈值继续扩展。",
  结构恢复预算: "命中 chunk 后可追加的 previous/next、section 和 bridge-neighbor 上下文数量上限。",
  路径摘要预算: "Context Package 中可保留的图路径摘要数量上限，用于解释证据从 coarse 到 mid 再到 chunk 的来源。",
  规划轮次预算: "QA Agent 可进行 Planner/Evaluator 规划的轮次数上限，控制单次任务内的推理成本。",
  每轮动作上限: "每个规划轮次最多允许的 typed actions 数量，所有动作仍必须通过 validator 和 deterministic executor。",
  修复轮次预算: "引用缺失、桥接不足或结构上下文不足时允许的 repair loop 次数；耗尽后只能返回已验证部分或证据不足说明。",
  引用验证预算: "回答后可执行的 citation verification 次数上限，用于把 claim 绑定回 raw chunk span。",
  固定切块尺寸: "解析时每个稳定 chunk 的目标 token 大小；chunk 是索引和引用地址单位，不假定是完整语义单元。",
  固定切块重叠: "相邻固定 chunk 之间保留的 token 重叠，用来降低边界截断造成的上下文损失。",
  向量维度: "Embedding 向量维数，必须与向量模型和 Qdrant collection 一致；改变后需要重嵌入或重建派生索引。",
  模型批次诊断上限: "构建 mid concept 时最多抽样多少个 LLM 批次做概念诊断；0 表示关闭这类模型诊断。",
  "每批 L3 前缀数": "每个概念生成批次最多处理的 RQ L3 prefix packet 数量，影响 mid concept 生成吞吐。",
  "每批概念 token 上限": "单个概念生成批次允许传入模型的 token 上限，防止 prompt 过大。",
  候选诊断阈值: "mid concept 候选保留诊断的 membership/质量阈值，用于标记低置信候选而不是直接制造事实。",
  "RQ-KMeans 层数": "残差量化地址树的层数；当前四层图谱用 L3 对齐 mid concept、L2 对齐 coarse concept。",
  "RQ-KMeans 最大 K": "每层 RQ-KMeans 聚类的最大分支数，影响 RQ prefix 地址空间粒度。",
  "RQ 残差 Tau": "控制 RQ fuzzy membership 的残差距离温度；值越小，membership 越集中。",
  "Dense KNN 最小 K": "每个 chunk 生成 dense relation 候选时的最小出边候选数，保障低证据节点仍有基本候选。",
  "Dense KNN 最大 K": "每个 chunk 生成 dense relation 候选时的最大出边候选数，限制高证据节点扩张。",
  基础互近邻下限: "普通 dense relation 的反向接纳下限，避免热门 chunk 吞掉全部入边机会。",
  基础互近邻上限: "普通 dense relation 的反向接纳上限，用于控制同一目标 chunk 的基础入边数量。",
  跨文档互近邻下限: "跨文档 bridge 候选的反向接纳下限，保证不同文档之间保留必要连接机会。",
  跨文档互近邻上限: "跨文档 bridge 候选的反向接纳上限，防止跨文档边过度膨胀。",
  跨语言互近邻下限: "跨语言 bridge 候选的反向接纳下限，保障不同语言资料之间的最小连接机会。",
  跨语言互近邻上限: "跨语言 bridge 候选的反向接纳上限，防止跨语言边过度膨胀。",
  "Dense 最小余弦": "dense relation 候选被接受的最低余弦相似度阈值，低于该值不进入 active relation graph。",
  "Dense 强边余弦": "标记强 dense 语义边的余弦阈值，用于 edge calibration 和路径距离诊断。",
  跨文档桥最小配额: "每个 chunk 额外尝试保留的跨文档 bridge 出边下限，只提供候选机会，不提升边权。",
  跨文档桥最大配额: "每个 chunk 额外保留的跨文档 bridge 出边上限，控制跨文档扩展成本。",
  跨文档桥最小余弦: "跨文档 bridge 候选的最低余弦阈值；达不到阈值不会成为底层关系边。",
  跨语言桥最小配额: "每个 chunk 额外尝试保留的跨语言 bridge 出边下限，只提供候选机会，不提升边权。",
  跨语言桥最大配额: "每个 chunk 额外保留的跨语言 bridge 出边上限，控制跨语言扩展成本。",
  跨语言桥最小余弦: "跨语言 bridge 候选的最低余弦阈值；达不到阈值不会成为底层关系边。",
  工作进程并发: "Celery worker 启动时的进程并发数；这是服务级参数，保存后需要重启或重建 worker 才会生效。",
};

function errorDialogFromUnknown(error: unknown): ErrorDialogState {
  const typed = error as Error & { status?: number; structured?: StructuredApiErrorBody };
  if (typed?.structured) {
    return {
      title: typed.structured.title || "操作失败",
      message: typed.structured.message || typed.message,
      status: typed.status,
      issues: typed.structured.issues ?? [],
      fixCommands: typed.structured.fix_commands ?? [],
    };
  }
  return {
    title: "操作失败",
    message: typed?.message || "请求没有成功完成。",
    status: typed?.status,
    issues: [],
    fixCommands: [],
  };
}

function parseIntField(value: string): number | undefined {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseFloatField(value: string): number | undefined {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function ErrorDialog({ state, onClose }: { state: ErrorDialogState | null; onClose: () => void }) {
  return (
    <Dialog open={Boolean(state)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-xl rounded-3xl border border-white/10 bg-[#101826] p-6 text-white shadow-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-lg">
            <ShieldAlert className="size-5 text-amber-200" />
            {state?.title}
          </DialogTitle>
          <DialogDescription className="text-sm leading-6 text-white/64">
            {state?.message}
            {state?.status ? <span className="ml-2 text-white/40">HTTP {state.status}</span> : null}
          </DialogDescription>
        </DialogHeader>

        {state?.issues.length ? (
          <div className="grid gap-3">
            {state.issues.map((issue) => (
              <div key={`${issue.code}:${issue.title}`} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <p className="text-sm font-semibold text-white">{issue.title}</p>
                <p className="mt-1 text-sm leading-6 text-white/62">{issue.message}</p>
              </div>
            ))}
          </div>
        ) : null}

        {state?.fixCommands.length ? (
          <div className="rounded-2xl border border-cyan-100/10 bg-cyan-100/[0.04] p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-cyan-100/58">修复命令</p>
            <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-cyan-50/78">
              {state.fixCommands.join("\n")}
            </pre>
          </div>
        ) : null}

        <DialogFooter className="border-white/10 bg-white/[0.03]">
          <Button type="button" className="rounded-full" onClick={onClose}>
            关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function ParameterName({
  label,
  description,
  className = parameterNameClass,
}: {
  label: string;
  description?: string;
  className?: string;
}) {
  const tooltipId = useId();
  const helpText = description ?? SETTINGS_PARAMETER_HELP[label];

  if (!helpText) {
    return <span className={className}>{label}</span>;
  }

  return (
    <span
      className={`group relative inline-flex w-fit cursor-help items-center gap-1 rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-cyan-200/40 ${className}`}
      tabIndex={0}
      aria-describedby={tooltipId}
    >
      <span>{label}</span>
      <Info className="size-3.5 text-cyan-100/45" aria-hidden="true" />
      <span
        id={tooltipId}
        role="tooltip"
        className="pointer-events-none absolute left-0 top-full z-50 mt-2 w-72 max-w-[calc(100vw-3rem)] rounded-xl border border-cyan-100/20 bg-[#081322]/95 p-3 text-left text-xs font-normal normal-case leading-5 tracking-normal text-cyan-50/82 opacity-0 shadow-2xl shadow-black/30 backdrop-blur transition duration-150 delay-0 group-hover:delay-500 group-hover:opacity-100 group-focus:delay-500 group-focus:opacity-100"
      >
        {helpText}
      </span>
    </span>
  );
}

function SettingField({
  label,
  description,
  value,
  onChange,
  type = "text",
  min,
  max,
  step,
  placeholder,
  disabled,
  className,
}: FieldProps) {
  return (
    <label className={`flex flex-col gap-2 ${className ?? ""}`}>
      <ParameterName label={label} description={description} />
      <Input
        type={type}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className={inputClass}
      />
    </label>
  );
}

function BoundaryNote({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <p className="mt-2 text-xs leading-5 text-cyan-50/52">
      <span className="font-medium text-cyan-50/70">{title}</span>{" "}
      {children}
    </p>
  );
}

function SwitchRow({ title, tooltip, description, checked, onChange, disabled, badge }: SwitchRowProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/[0.035] p-4">
      <div className="min-w-[240px] flex-1">
        <p className="flex flex-wrap items-center gap-2 text-sm font-semibold text-white">
          <SlidersHorizontal className="size-4 text-cyan-100/70" />
          <ParameterName label={title} description={tooltip} className="text-sm font-semibold normal-case tracking-normal text-white" />
          {badge ? <span className="text-xs font-normal text-cyan-100/45">{badge}</span> : null}
        </p>
        <p className="mt-2 text-sm leading-6 text-white/58">{description}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={onChange}
        className={`relative h-8 w-16 rounded-full border transition ${
          checked ? "border-cyan-100/40 bg-cyan-300/70" : "border-white/14 bg-white/10"
        } disabled:cursor-not-allowed disabled:opacity-60`}
      >
        <span className={`absolute top-1 size-6 rounded-full bg-white shadow transition ${checked ? "left-9" : "left-1"}`} />
      </button>
    </div>
  );
}

function StatusPill({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${ok ? "border-emerald-200/20 text-emerald-100" : "border-amber-200/20 text-amber-100"}`}>
      {ok ? <CheckCircle2 className="size-3.5" /> : <ShieldAlert className="size-3.5" />}
      {children}
    </span>
  );
}

function formatProfileJson(profile: StrategyProfileDetail | null | undefined): string {
  return JSON.stringify(profile?.profile_json ?? {}, null, 2);
}

type JsonDiagnostic = {
  line: number;
  column: number;
  message: string;
  reason: string;
  severity: "error" | "warning";
};

type AssistantMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  profileJson?: Record<string, unknown>;
  warnings?: string[];
  profileHash?: string;
};

function makeLocalId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getLineColumnFromPosition(text: string, position: number): { line: number; column: number } {
  const before = text.slice(0, Math.max(0, position));
  const lines = before.split("\n");
  return { line: lines.length, column: lines[lines.length - 1].length + 1 };
}

function getLineForKey(text: string, key: string): number {
  const lines = text.split("\n");
  const pattern = new RegExp(`"${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"\\s*:`);
  const index = lines.findIndex((line) => pattern.test(line));
  return index >= 0 ? index + 1 : 1;
}

function getProfileJsonDiagnostics(text: string): JsonDiagnostic[] {
  const diagnostics: JsonDiagnostic[] = [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : "JSON 解析失败";
    const positionMatch = message.match(/position\s+(\d+)/i);
    const location = positionMatch ? getLineColumnFromPosition(text, Number(positionMatch[1])) : { line: 1, column: 1 };
    diagnostics.push({
      ...location,
      severity: "error",
      message,
      reason: "JSON 语法不完整或存在多余字符，保存前必须修正。",
    });
    return diagnostics;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    diagnostics.push({
      line: 1,
      column: 1,
      severity: "error",
      message: "配置档 JSON 必须是对象",
      reason: "根节点需要是 user_profile_v1 对象，不能是数组、字符串或空值。",
    });
    return diagnostics;
  }
  const profile = parsed as Record<string, unknown>;
  for (const key of ["schema_version", "library_type", "ui_labels", "prompt_pack", "conversation_preferences"]) {
    if (!(key in profile)) {
      diagnostics.push({
        line: 1,
        column: 1,
        severity: "warning",
        message: `缺少 ${key}`,
        reason: "后端会尝试补默认值，但建议显式保留资料库类型、提示词、界面标签和对话偏好。",
      });
    }
  }
  for (const key of ["schema_pack", "concept_induction_policy", "parsing_strategy", "graph_strategy", "retrieval_strategy", "quality_policy", "signal_induction_policy"]) {
    if (key in profile) {
      diagnostics.push({
        line: getLineForKey(text, key),
        column: 1,
        severity: "warning",
        message: `${key} 已退出活动配置档`,
        reason: "配置档 JSON 只保存 library_type、prompt_pack、ui_labels 和 conversation_preferences；工程参数必须进入运行时设置。",
      });
    }
  }
  const promptPack = profile.prompt_pack;
  if (!promptPack || typeof promptPack !== "object" || Array.isArray(promptPack)) {
    diagnostics.push({
      line: getLineForKey(text, "prompt_pack"),
      column: 1,
      severity: "error",
      message: "prompt_pack 必须是对象",
      reason: "回答风格、引用严格度表达和无上下文提示需要从 prompt_pack 读取。",
    });
  }
  const conversationPreferences = profile.conversation_preferences;
  if (!conversationPreferences || typeof conversationPreferences !== "object" || Array.isArray(conversationPreferences)) {
    diagnostics.push({
      line: getLineForKey(text, "conversation_preferences"),
      column: 1,
      severity: "error",
      message: "conversation_preferences 必须是对象",
      reason: "对话偏好只能影响交互方式，不能保存工程运行参数。",
    });
  }
  return diagnostics;
}

function parseProfileJson(text: string): { value?: Record<string, unknown>; error?: string } {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { error: "配置档 JSON 必须是对象。" };
    }
    return { value: parsed as Record<string, unknown> };
  } catch (error) {
    return { error: error instanceof Error ? error.message : "JSON 解析失败。" };
  }
}

function ProfileSettingsPanel({ onError }: { onError: (error: unknown) => void }) {
  const queryClient = useQueryClient();
  const { selectedKnowledgeBaseId, selectedKnowledgeBase } = useKnowledgeBaseContext();
  const profilesQuery = useQuery({ queryKey: ["strategy-profiles"], queryFn: fetchStrategyProfiles });
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [name, setName] = useState("");
  const [libraryType, setLibraryType] = useState("custom");
  const [jsonText, setJsonText] = useState("{}");
  const [message, setMessage] = useState<string | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantPrompt, setAssistantPrompt] = useState("");
  const [assistantSessionId, setAssistantSessionId] = useState<string | null>(null);
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([]);
  const [assistantDraft, setAssistantDraft] = useState("");
  const [assistantResult, setAssistantResult] = useState<{ profileJson: Record<string, unknown>; warnings: string[]; profileHash?: string } | null>(null);
  const [assistantStreaming, setAssistantStreaming] = useState(false);
  const [assistantError, setAssistantError] = useState<string | null>(null);
  const assistantScrollRef = useRef<HTMLDivElement | null>(null);

  const currentProfile = profilesQuery.data?.find((profile) => profile.id === selectedProfileId) ?? null;
  const activeProfile = profilesQuery.data?.find((profile) => profile.id === selectedKnowledgeBase?.active_profile_id) ?? null;
  const detailQuery = useQuery({
    queryKey: ["strategy-profile", selectedProfileId],
    queryFn: () => fetchStrategyProfile(selectedProfileId),
    enabled: Boolean(selectedProfileId),
  });

  const parsed = useMemo(() => parseProfileJson(jsonText), [jsonText]);
  const jsonDiagnostics = useMemo(() => getProfileJsonDiagnostics(jsonText), [jsonText]);
  const hasJsonErrors = jsonDiagnostics.some((item) => item.severity === "error");
  const firstJsonError = jsonDiagnostics.find((item) => item.severity === "error");
  const jsonErrorLineStyle = firstJsonError
    ? {
        backgroundImage: "linear-gradient(rgba(244,63,94,0.22), rgba(244,63,94,0.22))",
        backgroundPosition: `0 ${16 + Math.max(0, firstJsonError.line - 1) * 20}px`,
        backgroundRepeat: "no-repeat",
        backgroundSize: "100% 20px",
        lineHeight: "20px",
      }
    : { lineHeight: "20px" };
  const validationWarnings = detailQuery.data?.warnings ?? [];

  useEffect(() => {
    assistantScrollRef.current?.scrollTo({ top: assistantScrollRef.current.scrollHeight });
  }, [assistantMessages, assistantDraft, assistantResult, assistantOpen]);

  useEffect(() => {
    if (!profilesQuery.data?.length) {
      return;
    }
    const nextId = selectedKnowledgeBase?.active_profile_id || profilesQuery.data[0]?.id || "";
    if (!selectedProfileId || !profilesQuery.data.some((profile) => profile.id === selectedProfileId)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedProfileId(nextId);
    }
  }, [profilesQuery.data, selectedKnowledgeBase?.active_profile_id, selectedProfileId]);

  useEffect(() => {
    if (!detailQuery.data) {
      return;
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setName(detailQuery.data.name);
    setLibraryType(detailQuery.data.library_type);
    setJsonText(formatProfileJson(detailQuery.data));
  }, [detailQuery.data]);

  const invalidateProfiles = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["strategy-profiles"] }),
      queryClient.invalidateQueries({ queryKey: ["strategy-profile", selectedProfileId] }),
      queryClient.invalidateQueries({ queryKey: ["knowledgeBases"] }),
      queryClient.invalidateQueries({ queryKey: ["dashboard", selectedKnowledgeBaseId] }),
      queryClient.invalidateQueries({ queryKey: ["graph", selectedKnowledgeBaseId] }),
    ]);
  };

  const saveMutation = useMutation({
    mutationFn: async () => {
      if (!selectedProfileId || !parsed.value) {
      throw new Error(parsed.error || "配置档 JSON 无效。");
      }
      return updateStrategyProfile(selectedProfileId, {
        name: name.trim(),
        library_type: libraryType.trim() || "custom",
        profile_json: parsed.value,
      });
    },
    onSuccess: async (data) => {
      setMessage("配置档已保存。");
      setJsonText(JSON.stringify(data.profile.profile_json, null, 2));
      await invalidateProfiles();
    },
    onError,
  });

  const copyMutation = useMutation({
    mutationFn: () => copyStrategyProfile(selectedProfileId, { name: `${name || currentProfile?.name || "配置档"} 副本` }),
    onSuccess: async (data) => {
      setSelectedProfileId(data.profile.id);
      setMessage("已复制为自定义配置档。");
      await invalidateProfiles();
    },
    onError,
  });

  const createMutation = useMutation({
    mutationFn: () => createStrategyProfile({ name: "新配置档", library_type: "custom", profile_json: parsed.value || {} }),
    onSuccess: async (data) => {
      setSelectedProfileId(data.profile.id);
      setMessage("已创建新配置档。");
      await invalidateProfiles();
    },
    onError,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteStrategyProfile(selectedProfileId),
    onSuccess: async () => {
      setDeleteConfirmOpen(false);
      setSelectedProfileId("");
      setMessage("配置档已删除；如有资料库曾绑定它，后端已自动切回默认配置档。");
      await invalidateProfiles();
    },
    onError,
  });

  const bindMutation = useMutation({
    mutationFn: () => {
      if (!selectedKnowledgeBaseId) {
        throw new Error("请先选择资料库。");
      }
      return bindStrategyProfile({ knowledge_base_id: selectedKnowledgeBaseId, profile_id: selectedProfileId });
    },
    onSuccess: async () => {
      setMessage("已设为当前资料库配置档。");
      await invalidateProfiles();
    },
    onError,
  });

  async function runAssistant() {
    const prompt = assistantPrompt.trim();
    if (!prompt || assistantStreaming) {
      return;
    }
    setAssistantPrompt("");
    setAssistantDraft("");
    setAssistantResult(null);
    setAssistantError(null);
    setAssistantStreaming(true);
    setAssistantMessages((items) => [...items, { id: makeLocalId(), role: "user", content: prompt }]);

    let streamedText = "";
    let streamedResult: { profileJson: Record<string, unknown>; warnings: string[]; profileHash?: string } | null = null;
    let streamedError: string | null = null;
    try {
      await streamProfileAssistant(
        {
          prompt,
          session_id: assistantSessionId,
          base_profile_id: selectedProfileId || null,
        },
        {
          onMeta: (meta) => {
            if (meta.session_id) {
              setAssistantSessionId(meta.session_id);
            }
          },
          onToken: (token) => {
            streamedText += token;
            setAssistantDraft(streamedText);
          },
          onProfileJson: (result) => {
            streamedResult = {
              profileJson: result.profile_json,
              warnings: result.warnings,
              profileHash: result.profile_hash,
            };
            setAssistantResult(streamedResult);
          },
          onError: (value) => {
            streamedError = value;
            setAssistantError(value);
          },
        },
      );
      if (streamedError) {
        throw new Error(streamedError);
      }
      setAssistantMessages((items) => [
        ...items,
        {
          id: makeLocalId(),
          role: "assistant",
          content: streamedText || "已生成配置档草案。",
          profileJson: streamedResult?.profileJson,
          warnings: streamedResult?.warnings,
          profileHash: streamedResult?.profileHash,
        },
      ]);
      setAssistantDraft("");
      setAssistantResult(null);
    } catch (error) {
      const messageText = error instanceof Error ? error.message : "配置档助手生成失败";
      setAssistantError(messageText);
      onError(error);
    } finally {
      setAssistantStreaming(false);
    }
  }

  function applyAssistantProfile(profileJson: Record<string, unknown>, warnings: string[] = []) {
    setJsonText(JSON.stringify(profileJson, null, 2));
    setMessage(
      isBuiltin
        ? "草案已填入高级 JSON。内置配置档受保护，请复制后保存。"
        : warnings.length
          ? warnings.join("；")
          : "草案已填入高级 JSON，请检查诊断结果后保存。",
    );
    setAssistantOpen(false);
  }

  if (profilesQuery.isLoading) {
    return <LoadingBlock rows={3} />;
  }
  if (profilesQuery.error) {
    return <ErrorBlock message={(profilesQuery.error as Error).message} />;
  }

  const isBuiltin = Boolean(currentProfile?.is_builtin || detailQuery.data?.is_builtin);
  const selectedProfileKnowledgeBaseIds = currentProfile?.knowledge_base_ids ?? detailQuery.data?.knowledge_base_ids ?? [];
  const deleteBlockedReason = isBuiltin ? "默认内置配置档受保护；请复制后编辑。" : null;
  const deleteImpactMessage =
    selectedProfileKnowledgeBaseIds.length > 0
      ? `该配置档当前绑定 ${selectedProfileKnowledgeBaseIds.length} 个资料库。删除后，这些资料库会自动切回默认配置档；已有片段、图谱、向量和会话不会被改写。`
      : "该配置档当前没有绑定资料库。删除后会从列表中隐藏，已有历史数据不会被改写。";
  const hashMismatch = Boolean(
    selectedKnowledgeBase?.active_profile_hash &&
      activeProfile?.profile_hash &&
      selectedKnowledgeBase.active_profile_hash !== activeProfile.profile_hash,
  );
  const renderAssistantJsonCard = (
    profileJson: Record<string, unknown>,
    warnings: string[] = [],
    profileHash?: string,
  ) => (
    <div className="mt-3 rounded-2xl border border-cyan-200/15 bg-black/25 p-3">
      <div className="mb-2 flex items-center justify-between gap-3 text-xs text-cyan-100/70">
        <span>高级 JSON 结果</span>
        {profileHash ? <span className="break-all">哈希 {profileHash}</span> : null}
      </div>
      <pre className="max-h-64 overflow-auto rounded-xl bg-black/35 p-3 font-mono text-[11px] leading-5 text-cyan-50">
        {JSON.stringify(profileJson, null, 2)}
      </pre>
      {warnings.length ? (
        <div className="mt-2 space-y-1">
          {warnings.map((warning) => (
            <p key={warning} className="text-xs leading-5 text-amber-100">
              {warning}
            </p>
          ))}
        </div>
      ) : null}
      <Button type="button" className="mt-3 w-full rounded-full" onClick={() => applyAssistantProfile(profileJson, warnings)}>
        <Sparkles data-icon="inline-start" />
        自动填充
      </Button>
    </div>
  );

  return (
    <section className="grid gap-6 xl:grid-cols-[minmax(300px,0.7fr)_minmax(560px,1.3fr)]">
      <aside className="space-y-5">
        <div>
          <p className="section-kicker">配置档设置</p>
          <h2 className="glow-text mt-2 text-3xl font-semibold text-white">资料库配置档</h2>
          <p className="mt-4 text-sm leading-7 text-cyan-50/62">
            配置档只影响之后启动的新解析、四层图谱、检索和问答任务；已有片段、向量、图谱和会话不会被自动改写。
          </p>
        </div>
        <div className={sectionClass}>
          <p className="text-sm font-semibold text-white">当前绑定</p>
          <div className="mt-3 space-y-2 text-sm leading-6 text-white/62">
            <p>资料库：{selectedKnowledgeBase?.name ?? "未选择"}</p>
            <p>配置档：{activeProfile?.name ?? selectedKnowledgeBase?.active_profile_name ?? "未绑定"}</p>
            <p className="break-all">哈希：{selectedKnowledgeBase?.active_profile_hash ?? "缺失"}</p>
          </div>
          {hashMismatch ? (
            <p className="mt-4 rounded-xl border border-amber-200/20 bg-amber-200/[0.06] p-3 text-sm leading-6 text-amber-100">
              当前资料库记录的配置档哈希与列表中的配置档哈希不一致。切换或修改后，请显式重新解析或重建图谱。
            </p>
          ) : null}
        </div>
      </aside>

      <div className="grid gap-5">
        <section className={sectionClass}>
          <div className="grid gap-4 md:grid-cols-[1fr_0.7fr]">
            <label className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/46">配置档</span>
              <select
                value={selectedProfileId}
                onChange={(event) => setSelectedProfileId(event.target.value)}
                className={`${inputClass} kg-dark-select outline-none`}
              >
                {(profilesQuery.data ?? []).map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}{profile.is_builtin ? " / 内置" : ""}
                  </option>
                ))}
              </select>
            </label>
            <SettingField label="资料库类型" value={libraryType} onChange={setLibraryType} disabled={isBuiltin} />
            <SettingField label="名称" value={name} onChange={setName} disabled={isBuiltin} className="md:col-span-2" />
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button type="button" variant="outline" className="rounded-full" onClick={() => copyMutation.mutate()} disabled={!selectedProfileId || copyMutation.isPending}>
              <Copy data-icon="inline-start" />
              复制预设
            </Button>
            <Button type="button" variant="outline" className="rounded-full" onClick={() => createMutation.mutate()} disabled={createMutation.isPending || hasJsonErrors}>
              <FilePlus2 data-icon="inline-start" />
              新建
            </Button>
            <Button type="button" variant="outline" className="rounded-full" onClick={() => setAssistantOpen(true)}>
              <Sparkles data-icon="inline-start" />
              AI 设置助手
            </Button>
            <Button type="button" className="rounded-full" onClick={() => bindMutation.mutate()} disabled={!selectedKnowledgeBaseId || !selectedProfileId || bindMutation.isPending}>
              设为当前资料库
            </Button>
            <Button
              type="button"
              variant="outline"
              className="rounded-full border-rose-200/20 text-rose-100 disabled:text-white/35"
              onClick={() => {
                if (deleteBlockedReason) {
                  setMessage(deleteBlockedReason);
                  return;
                }
                deleteMutation.reset();
                setDeleteConfirmOpen(true);
              }}
              disabled={Boolean(deleteBlockedReason) || !selectedProfileId || deleteMutation.isPending}
              title={deleteBlockedReason ?? undefined}
            >
              <Trash2 data-icon="inline-start" />
              删除配置档
            </Button>
          </div>
          {deleteBlockedReason ? <p className="mt-3 text-sm leading-6 text-amber-100/80">{deleteBlockedReason}</p> : null}
        </section>

        <section className={sectionClass}>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-white">高级 JSON</p>
              <p className="mt-1 text-sm text-white/52">结构化字段与高级 JSON 共用同一份交互配置结构。</p>
            </div>
            <Button type="button" className="rounded-full" onClick={() => saveMutation.mutate()} disabled={isBuiltin || hasJsonErrors || saveMutation.isPending}>
              {saveMutation.isPending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Save data-icon="inline-start" />}
              保存配置档
            </Button>
          </div>
          <div className="grid gap-3 lg:grid-cols-[minmax(180px,0.35fr)_minmax(0,1fr)]">
            <div className="h-[460px] max-h-[460px] overflow-y-auto rounded-2xl border border-white/10 bg-black/20 p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100/55">诊断</p>
                <span className={`rounded-full px-2 py-0.5 text-[11px] ${hasJsonErrors ? "bg-rose-400/10 text-rose-100" : "bg-emerald-400/10 text-emerald-100"}`}>
                  {hasJsonErrors ? "error" : "ok"}
                </span>
              </div>
              {jsonDiagnostics.length ? (
                <div className="space-y-2">
                  {jsonDiagnostics.map((item, index) => (
                    <div key={`${item.line}-${item.column}-${index}`} className={`rounded-xl border p-3 text-xs leading-5 ${item.severity === "error" ? "border-rose-200/20 bg-rose-200/[0.06] text-rose-100" : "border-amber-200/20 bg-amber-200/[0.05] text-amber-100"}`}>
                      <p className="font-semibold">第 {item.line} 行，第 {item.column} 列</p>
                      <p className="mt-1">{item.message}</p>
                      <p className="mt-1 text-white/52">原因：{item.reason}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="rounded-xl border border-emerald-200/20 bg-emerald-200/[0.05] p-3 text-xs leading-5 text-emerald-100">
                  未发现格式错误。保存时仍会执行后端配置档结构校验。
                </p>
              )}
            </div>
            <Textarea
              value={jsonText}
              onChange={(event) => setJsonText(event.target.value)}
              disabled={isBuiltin}
              spellCheck={false}
              style={jsonErrorLineStyle}
              className={`h-[460px] max-h-[460px] resize-none overflow-y-auto rounded-2xl border-white/10 bg-black/20 p-4 font-mono text-xs leading-5 text-cyan-50 ${firstJsonError ? "border-rose-300/40" : ""}`}
            />
          </div>
          <div className="mt-4 grid gap-2">
            {hasJsonErrors ? <p className="rounded-xl border border-rose-200/20 bg-rose-200/[0.06] p-3 text-sm text-rose-100">请先修正左侧诊断栏中的 JSON 错误。</p> : <p className="text-sm text-emerald-100">JSON 格式有效。</p>}
            {validationWarnings.map((warning) => (
              <p key={warning} className="rounded-xl border border-amber-200/20 bg-amber-200/[0.05] p-3 text-sm text-amber-100">
                {warning}
              </p>
            ))}
            {isBuiltin ? <p className="text-sm text-white/48">内置配置档受保护；复制后可编辑。</p> : null}
            {message ? <p className="text-sm text-cyan-100">{message}</p> : null}
          </div>
        </section>
      </div>

      {assistantOpen ? (
        <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-white/10 bg-[#07111f]/95 p-5 text-white shadow-2xl backdrop-blur-xl">
          <div className="flex items-start justify-between gap-4 border-b border-white/10 pb-4">
            <div>
              <p className="section-kicker">智能设置助手</p>
              <h3 className="mt-2 text-2xl font-semibold">配置档对话草案</h3>
              {assistantSessionId ? <p className="mt-1 max-w-sm truncate text-xs text-white/45">Redis 会话：{assistantSessionId}</p> : null}
            </div>
            <Button type="button" variant="outline" className="rounded-full" onClick={() => setAssistantOpen(false)}>
              关闭
            </Button>
          </div>
          <div ref={assistantScrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto py-4 pr-1">
            {assistantMessages.length === 0 && !assistantDraft ? (
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-sm leading-7 text-white/62">
                输入资料库类型、界面标签、回答提示词、引用严格度表达、澄清方式和无上下文回复文案。助手会先输出说明，再给出一个只包含交互配置的 JSON 草案。
              </div>
            ) : null}
            {assistantMessages.map((item) => (
              <div key={item.id} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[92%] rounded-2xl border p-3 text-sm leading-7 ${item.role === "user" ? "border-cyan-200/20 bg-cyan-200/[0.08] text-cyan-50" : "border-white/10 bg-white/[0.04] text-white/78"}`}>
                  {item.role === "assistant" ? (
                    <div className="mb-2 flex items-center gap-2 text-xs text-cyan-100/65">
                      <Bot className="size-3.5" />
                      配置档助手
                    </div>
                  ) : null}
                  <p className="whitespace-pre-wrap">{item.content}</p>
                  {item.profileJson ? renderAssistantJsonCard(item.profileJson, item.warnings ?? [], item.profileHash) : null}
                </div>
              </div>
            ))}
            {assistantStreaming ? (
              <div className="flex justify-start">
                <div className="max-w-[92%] rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-sm leading-7 text-white/78">
                  <div className="mb-2 flex items-center gap-2 text-xs text-cyan-100/65">
                    <span className="context-bars">
                      <span />
                      <span />
                      <span />
                      <span />
                    </span>
                    正在生成
                  </div>
                  {assistantDraft ? <p className="whitespace-pre-wrap">{assistantDraft}</p> : null}
                  {assistantResult ? renderAssistantJsonCard(assistantResult.profileJson, assistantResult.warnings, assistantResult.profileHash) : null}
                </div>
              </div>
            ) : null}
            {assistantError ? (
              <p className="rounded-xl border border-rose-200/20 bg-rose-200/[0.06] p-3 text-sm text-rose-100">
                {assistantError}
              </p>
            ) : null}
          </div>
          <div className="border-t border-white/10 pt-4">
            <Textarea
              value={assistantPrompt}
              onChange={(event) => setAssistantPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                  event.preventDefault();
                  void runAssistant();
                }
              }}
              placeholder="描述资料库类型、界面标签、回答风格、引用严格度表达、澄清方式和无上下文回复文案。"
              className="min-h-24 resize-none rounded-2xl border-white/10 bg-white/[0.04] text-white"
            />
            <Button type="button" className="mt-3 w-full rounded-full" onClick={() => void runAssistant()} disabled={!assistantPrompt.trim() || assistantStreaming}>
              {assistantStreaming ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Send data-icon="inline-start" />}
              发送
            </Button>
          </div>
        </div>
      ) : null}

      <Dialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
        <DialogContent className="max-h-[calc(100vh-2rem)] w-[min(42rem,calc(100vw-2rem))] overflow-hidden rounded-3xl border border-white/10 bg-[#101826] p-0 text-white shadow-2xl sm:!max-w-xl">
          <DialogHeader className="border-b border-white/8 px-6 py-5 pr-14">
            <DialogTitle>确认删除配置档</DialogTitle>
            <DialogDescription className="break-words text-cyan-100/70">
              {currentProfile?.name ? `即将删除「${currentProfile.name}」。默认内置配置档不能删除，其他配置档删除后会软删除并从列表隐藏。` : "即将删除当前配置档。"}
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[55vh] space-y-4 overflow-y-auto px-6 py-5">
            <p className="rounded-2xl border border-amber-200/18 bg-amber-200/[0.06] p-4 text-sm leading-6 text-amber-50/85">
              {deleteImpactMessage}
            </p>
            {deleteMutation.error ? <p className="text-sm text-rose-100/80">{(deleteMutation.error as Error).message}</p> : null}
          </div>
          <div className="flex flex-wrap items-center justify-end gap-3 border-t border-white/10 bg-white/[0.03] px-6 py-4">
            <Button type="button" variant="outline" className="rounded-full" onClick={() => setDeleteConfirmOpen(false)} disabled={deleteMutation.isPending}>
              取消
            </Button>
            <Button type="button" className="rounded-full" onClick={() => deleteMutation.mutate()} disabled={!selectedProfileId || deleteMutation.isPending}>
              {deleteMutation.isPending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Trash2 data-icon="inline-start" />}
              删除
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
}

export function SettingsWorkspace() {
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({ queryKey: ["model-settings"], queryFn: fetchModelSettings });
  const runtimeQuery = useQuery({ queryKey: ["runtime-check"], queryFn: () => fetchRuntimeCheck(), retry: false });
  const [form, setForm] = useState<SettingsForm | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [apiKeyEditing, setApiKeyEditing] = useState(false);
  const [embeddingApiKeyEditing, setEmbeddingApiKeyEditing] = useState(false);
  const [errorDialog, setErrorDialog] = useState<ErrorDialogState | null>(null);
  const [activeTab, setActiveTab] = useState<"model" | "profile">("model");

  useEffect(() => {
    if (!settingsQuery.data) {
      return;
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setForm({
      chat_base_url: settingsQuery.data.chat_base_url ?? "",
      embedding_base_url: settingsQuery.data.embedding_base_url ?? "",
      chat_resolve_ip: settingsQuery.data.chat_resolve_ip ?? "",
      embedding_resolve_ip: settingsQuery.data.embedding_resolve_ip ?? "",
      embedding_model: settingsQuery.data.embedding_model ?? "",
      chat_model: settingsQuery.data.chat_model ?? "",
      embedding_dimensions: String(settingsQuery.data.embedding_dimensions ?? 1024),
      embedding_batch_size: String(settingsQuery.data.embedding_batch_size ?? 10),
      worker_concurrency: String(settingsQuery.data.worker_concurrency ?? 3),
      model_request_concurrency: String(settingsQuery.data.model_request_concurrency ?? 3),
      model_request_timeout_seconds: String(settingsQuery.data.model_request_timeout_seconds ?? 240),
      concept_i18n_enabled: settingsQuery.data.concept_i18n_enabled ?? false,
      fixed_chunk_size_tokens: String(settingsQuery.data.fixed_chunk_size_tokens ?? 512),
      fixed_chunk_overlap_tokens: String(settingsQuery.data.fixed_chunk_overlap_tokens ?? 80),
      context_package_token_budget: String(settingsQuery.data.context_package_token_budget ?? 2400),
      api_key: "",
      clear_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: settingsQuery.data.model_bridge_enabled ?? true,
      mid_concept_extraction_max_model_batches: String(settingsQuery.data.mid_concept_extraction_max_model_batches ?? 4),
      mid_concept_extraction_max_candidates_per_batch: String(settingsQuery.data.mid_concept_extraction_max_candidates_per_batch ?? 8),
      mid_concept_extraction_max_tokens_per_batch: String(settingsQuery.data.mid_concept_extraction_max_tokens_per_batch ?? 2400),
      mid_concept_candidate_keep_threshold: String(settingsQuery.data.mid_concept_candidate_keep_threshold ?? 0.62),
      rq_kmeans_levels: String(settingsQuery.data.rq_kmeans_levels ?? 3),
      rq_kmeans_max_k: String(settingsQuery.data.rq_kmeans_max_k ?? 6),
      rq_residual_tau: String(settingsQuery.data.rq_residual_tau ?? 0.65),
      dense_knn_k_min: String(settingsQuery.data.dense_knn_k_min ?? 5),
      dense_knn_k_max: String(settingsQuery.data.dense_knn_k_max ?? 24),
      dense_reverse_b_min_base: String(settingsQuery.data.dense_reverse_b_min_base ?? 2),
      dense_reverse_b_max_base: String(settingsQuery.data.dense_reverse_b_max_base ?? 8),
      dense_reverse_b_min_doc: String(settingsQuery.data.dense_reverse_b_min_doc ?? 1),
      dense_reverse_b_max_doc: String(settingsQuery.data.dense_reverse_b_max_doc ?? 6),
      dense_reverse_b_min_lang: String(settingsQuery.data.dense_reverse_b_min_lang ?? 1),
      dense_reverse_b_max_lang: String(settingsQuery.data.dense_reverse_b_max_lang ?? 4),
      dense_min_cosine: String(settingsQuery.data.dense_min_cosine ?? 0.58),
      dense_strong_cosine: String(settingsQuery.data.dense_strong_cosine ?? 0.72),
      cross_doc_out_quota_min: String(settingsQuery.data.cross_doc_out_quota_min ?? 1),
      cross_doc_out_quota_max: String(settingsQuery.data.cross_doc_out_quota_max ?? 4),
      cross_doc_min_cosine: String(settingsQuery.data.cross_doc_min_cosine ?? 0.62),
      cross_language_out_quota_min: String(settingsQuery.data.cross_language_out_quota_min ?? 0),
      cross_language_out_quota_max: String(settingsQuery.data.cross_language_out_quota_max ?? 3),
      cross_language_min_cosine: String(settingsQuery.data.cross_language_min_cosine ?? 0.65),
      agent_coarse_total_budget: String(settingsQuery.data.agent_coarse_total_budget ?? 8),
      agent_mid_per_coarse_budget: String(settingsQuery.data.agent_mid_per_coarse_budget ?? 6),
      agent_mid_top_k: String(settingsQuery.data.agent_mid_top_k ?? 16),
      agent_chunk_per_mid_budget: String(settingsQuery.data.agent_chunk_per_mid_budget ?? 8),
      agent_chunk_top_k: String(settingsQuery.data.agent_chunk_top_k ?? 40),
      candidate_pool_dedupe_budget: String(settingsQuery.data.candidate_pool_dedupe_budget ?? 160),
      agent_max_depth_per_layer: String(settingsQuery.data.agent_max_depth_per_layer ?? 3),
      agent_max_labels_per_node: String(settingsQuery.data.agent_max_labels_per_node ?? 3),
      agent_max_edge_reuse: String(settingsQuery.data.agent_max_edge_reuse ?? 2),
      agent_max_cycle_reward_per_path: String(settingsQuery.data.agent_max_cycle_reward_per_path ?? 0.18),
      agent_cycle_reward_distance_threshold: String(settingsQuery.data.agent_cycle_reward_distance_threshold ?? 1.2),
      agent_path_distance_green_threshold: String(settingsQuery.data.agent_path_distance_green_threshold ?? 0.45),
      agent_path_distance_gray_threshold: String(settingsQuery.data.agent_path_distance_gray_threshold ?? 1.35),
      agent_path_distance_hard_threshold: String(settingsQuery.data.agent_path_distance_hard_threshold ?? 2.4),
      agent_structure_restore_budget: String(settingsQuery.data.agent_structure_restore_budget ?? 16),
      context_path_summary_budget: String(settingsQuery.data.context_path_summary_budget ?? 32),
      agent_planning_round_budget: String(settingsQuery.data.agent_planning_round_budget ?? 2),
      agent_max_typed_actions_per_round: String(settingsQuery.data.agent_max_typed_actions_per_round ?? 8),
      agent_repair_round_budget: String(settingsQuery.data.agent_repair_round_budget ?? 2),
      agent_verification_budget: String(settingsQuery.data.agent_verification_budget ?? 8),
    });
    setApiKeyEditing(false);
    setEmbeddingApiKeyEditing(false);
  }, [settingsQuery.data]);

  const saveMutation = useMutation({
    mutationFn: (payload: ModelSettingsUpdate) => updateModelSettings(payload),
    onSuccess: async () => {
      setApiKeyEditing(false);
      setEmbeddingApiKeyEditing(false);
      setSavedMessage("已保存");
      window.setTimeout(() => setSavedMessage(null), 1800);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["model-settings"] }),
        queryClient.invalidateQueries({ queryKey: ["runtime-check"] }),
        queryClient.invalidateQueries({ queryKey: ["knowledgeBases"] }),
      ]);
    },
    onError: (error) => setErrorDialog(errorDialogFromUnknown(error)),
  });

  const settings = settingsQuery.data;
  const showApiKeyMask = Boolean(settings?.has_api_key && !apiKeyEditing && !form?.clear_api_key);
  const showEmbeddingApiKeyMask = Boolean(settings?.has_embedding_api_key && !embeddingApiKeyEditing && !form?.clear_embedding_api_key);
  const envSynced = Boolean(runtimeQuery.data?.env_sync?.synced);
  const runtimeWarnings = runtimeQuery.data?.warnings ?? [];
  const bridgeStatus = settings?.model_bridge_status;
  const bridgeHealthy =
    !form?.model_bridge_enabled ||
    Boolean(bridgeStatus?.reachable && bridgeStatus?.admin_available && bridgeStatus?.config_matches && !bridgeStatus?.self_target_blocked);
  const bridgeStatusText = !form?.model_bridge_enabled
    ? "未启用"
    : bridgeStatus?.self_target_blocked
      ? "目标自环"
    : bridgeStatus?.reachable
      ? bridgeStatus?.admin_available
        ? bridgeStatus?.config_matches
          ? "配置已同步"
          : "配置不一致"
        : "管理接口不可用"
      : "不可达";
  const bridgeWarnings = bridgeStatus?.warnings ?? [];

  if (settingsQuery.isLoading || !form) {
    return <LoadingBlock rows={4} />;
  }
  if (settingsQuery.error) {
    return <ErrorBlock message={(settingsQuery.error as Error).message} />;
  }

  const updateForm = <K extends keyof SettingsForm>(key: K, value: SettingsForm[K]) => {
    setForm((current) => (current ? { ...current, [key]: value } : current));
  };

  const buildPayload = (): ModelSettingsUpdate => ({
    chat_base_url: form.chat_base_url.trim(),
    embedding_base_url: form.embedding_base_url.trim(),
    chat_resolve_ip: form.chat_resolve_ip.trim() || null,
    embedding_resolve_ip: form.embedding_resolve_ip.trim() || null,
    embedding_model: form.embedding_model.trim(),
    chat_model: form.chat_model.trim(),
    embedding_dimensions: parseIntField(form.embedding_dimensions),
    embedding_batch_size: parseIntField(form.embedding_batch_size),
    worker_concurrency: parseIntField(form.worker_concurrency),
    model_request_concurrency: parseIntField(form.model_request_concurrency),
    model_request_timeout_seconds: parseIntField(form.model_request_timeout_seconds),
    concept_i18n_enabled: form.concept_i18n_enabled,
    fixed_chunk_size_tokens: parseIntField(form.fixed_chunk_size_tokens),
    fixed_chunk_overlap_tokens: parseIntField(form.fixed_chunk_overlap_tokens),
    context_package_token_budget: parseIntField(form.context_package_token_budget),
    api_key: form.api_key.trim() || null,
    clear_api_key: form.clear_api_key,
    embedding_api_key: form.embedding_api_key.trim() || null,
    clear_embedding_api_key: form.clear_embedding_api_key,
    model_bridge_enabled: form.model_bridge_enabled,
    mid_concept_extraction_max_model_batches: parseIntField(form.mid_concept_extraction_max_model_batches),
    mid_concept_extraction_max_candidates_per_batch: parseIntField(form.mid_concept_extraction_max_candidates_per_batch),
    mid_concept_extraction_max_tokens_per_batch: parseIntField(form.mid_concept_extraction_max_tokens_per_batch),
    mid_concept_candidate_keep_threshold: parseFloatField(form.mid_concept_candidate_keep_threshold),
    rq_kmeans_levels: parseIntField(form.rq_kmeans_levels),
    rq_kmeans_max_k: parseIntField(form.rq_kmeans_max_k),
    rq_residual_tau: parseFloatField(form.rq_residual_tau),
    dense_knn_k_min: parseIntField(form.dense_knn_k_min),
    dense_knn_k_max: parseIntField(form.dense_knn_k_max),
    dense_reverse_b_min_base: parseIntField(form.dense_reverse_b_min_base),
    dense_reverse_b_max_base: parseIntField(form.dense_reverse_b_max_base),
    dense_reverse_b_min_doc: parseIntField(form.dense_reverse_b_min_doc),
    dense_reverse_b_max_doc: parseIntField(form.dense_reverse_b_max_doc),
    dense_reverse_b_min_lang: parseIntField(form.dense_reverse_b_min_lang),
    dense_reverse_b_max_lang: parseIntField(form.dense_reverse_b_max_lang),
    dense_min_cosine: parseFloatField(form.dense_min_cosine),
    dense_strong_cosine: parseFloatField(form.dense_strong_cosine),
    cross_doc_out_quota_min: parseIntField(form.cross_doc_out_quota_min),
    cross_doc_out_quota_max: parseIntField(form.cross_doc_out_quota_max),
    cross_doc_min_cosine: parseFloatField(form.cross_doc_min_cosine),
    cross_language_out_quota_min: parseIntField(form.cross_language_out_quota_min),
    cross_language_out_quota_max: parseIntField(form.cross_language_out_quota_max),
    cross_language_min_cosine: parseFloatField(form.cross_language_min_cosine),
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
  });

  const handleSubmit = async () => {
    saveMutation.mutate(buildPayload());
  };

  const handleRuntimeCheck = async () => {
    const result = await runtimeQuery.refetch();
    if (result.error) {
      setErrorDialog(errorDialogFromUnknown(result.error));
    }
  };

  return (
    <div className="kg-page">
      <section className="glass-panel rounded-[28px] p-6 lg:p-8">
        <div className="mb-6 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setActiveTab("model")}
            className={`rounded-full border px-4 py-2 text-sm transition ${activeTab === "model" ? "border-cyan-200/30 bg-cyan-300/[0.08] text-cyan-50" : "border-white/10 text-white/58 hover:text-white"}`}
          >
            模型与运行配置
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("profile")}
            className={`rounded-full border px-4 py-2 text-sm transition ${activeTab === "profile" ? "border-cyan-200/30 bg-cyan-300/[0.08] text-cyan-50" : "border-white/10 text-white/58 hover:text-white"}`}
          >
            配置档设置
          </button>
        </div>
        {activeTab === "profile" ? <ProfileSettingsPanel onError={(error) => setErrorDialog(errorDialogFromUnknown(error))} /> : null}
        <div className={activeTab === "model" ? "grid gap-7 xl:grid-cols-[minmax(320px,0.72fr)_minmax(560px,1.28fr)]" : "hidden"}>
          <aside className="space-y-6">
            <div>
              <p className="section-kicker">生产参数配置</p>
              <h2 className="glow-text mt-2 text-4xl font-semibold text-white">运行时设置</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-cyan-50/62">
                这里只保留当前 active path 实际消费的参数，并按下一次调用、重建后、重启服务后三类边界标注。
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <StatusPill ok={Boolean(settings?.has_api_key)}>聊天密钥 {settings?.has_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(settings?.has_embedding_api_key)}>向量密钥 {settings?.has_embedding_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={bridgeHealthy}>模型桥 {bridgeStatusText}</StatusPill>
              <StatusPill ok={envSynced}>{envSynced ? ".env 已同步" : ".env 需检查"}</StatusPill>
              <StatusPill ok={!settings?.enable_model_fallback && !settings?.enable_database_fallback}>回退已禁用</StatusPill>
              <StatusPill ok={!settings?.concept_i18n_enabled}>双语派生 {settings?.concept_i18n_enabled ? "已开启" : "已关闭"}</StatusPill>
              <StatusPill ok={Boolean(settings?.lifecycle?.hot_reloadable?.length)}>热加载 {settings?.lifecycle?.hot_reloadable?.length ?? 0}</StatusPill>
              <StatusPill ok={Boolean(settings?.lifecycle?.rebuild_required?.length)}>需重建 {settings?.lifecycle?.rebuild_required?.length ?? 0}</StatusPill>
              <StatusPill ok={Boolean(settings?.runtime_settings_version)}>运行时 {settings?.runtime_settings_version ? settings.runtime_settings_version.slice(0, 12) : "等待中"}</StatusPill>
            </div>

            <div className={sectionClass}>
              <p className="text-sm font-semibold text-white">生产保护</p>
              <div className="mt-3 grid gap-2 text-sm leading-6 text-white/58">
                <p>模型回退：{settings?.enable_model_fallback ? "已开启，生产不推荐" : "已关闭"}</p>
                <p>数据库回退：{settings?.enable_database_fallback ? "已开启，生产不推荐" : "已关闭"}</p>
                <p>增量图谱：由前端触发图谱任务时选择“最小更新/全量重建”，当前不是全局 env 开关。</p>
              </div>
            </div>
          </aside>

          <form
            className="grid gap-5"
            onSubmit={(event) => {
              event.preventDefault();
              void handleSubmit();
            }}
          >
            <section className={sectionClass}>
              <div className="mb-5 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-white">模型连接与密钥</p>
                  <p className="mt-1 text-sm text-white/52">聊天模型、向量模型、连接地址和接口密钥集中在这里维护。</p>
                </div>
              </div>
              <BoundaryNote title="生效边界：下一次请求或下一次模型调用">
                聊天地址、聊天 DNS、聊天模型、向量地址、向量 DNS、模型桥和密钥保存后会广播运行时版本；已经在执行的模型调用不会中途切换。
              </BoundaryNote>
              <BoundaryNote title="向量模型边界：已有 active 向量不会自动改写">
                修改向量模型后只影响后续解析、重嵌入或全量重建任务；已有资料库向量需要显式重新解析或重建。
              </BoundaryNote>
              {form.model_bridge_enabled ? (
                <div className="mt-4 border-l border-cyan-200/20 bg-cyan-300/[0.035] px-4 py-3 text-sm text-cyan-50/70">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-white">模型桥转发状态</span>
                    <StatusPill ok={bridgeHealthy}>{bridgeStatusText}</StatusPill>
                    {bridgeStatus?.config_version ? <span className="text-xs text-cyan-100/50">版本 {bridgeStatus.config_version.slice(0, 12)}</span> : null}
                  </div>
                  <div className="mt-3 grid gap-2 md:grid-cols-2">
                    <p className="min-w-0 break-words">聊天 effective 地址：{settings?.effective_chat_base_url || "未读取"}</p>
                    <p className="min-w-0 break-words">向量 effective 地址：{settings?.effective_embedding_base_url || "未读取"}</p>
                    <p className="min-w-0 break-words">聊天目标 hash：{bridgeStatus?.chat_target_hash?.slice(0, 12) || "未知"}</p>
                    <p className="min-w-0 break-words">向量目标 hash：{bridgeStatus?.embedding_target_hash?.slice(0, 12) || "未知"}</p>
                  </div>
                  {bridgeStatus?.last_reload ? (
                    <p className={bridgeStatus.last_reload.ok ? "mt-3 text-emerald-100/75" : "mt-3 text-rose-100/80"}>
                      最近热加载：{bridgeStatus.last_reload.ok ? "成功" : `失败 ${bridgeStatus.last_reload.error ?? bridgeStatus.last_reload.status_code ?? ""}`}
                    </p>
                  ) : null}
                  {bridgeWarnings.length ? (
                    <ul className="mt-3 space-y-1 text-amber-100/78">
                      {bridgeWarnings.map((warning) => (
                        <li key={warning} className="break-words">
                          {warning}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ) : null}
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <SwitchRow
                  title="模型桥"
                  description="开启后容器通过 host.docker.internal 访问宿主机模型桥。"
                  checked={form.model_bridge_enabled}
                  onChange={() => updateForm("model_bridge_enabled", !form.model_bridge_enabled)}
                  disabled={saveMutation.isPending}
                  badge="下一次调用"
                />
                <SettingField label="聊天基础地址" value={form.chat_base_url} onChange={(value) => updateForm("chat_base_url", value)} className="md:col-span-2" />
                <SettingField label="向量基础地址" value={form.embedding_base_url} onChange={(value) => updateForm("embedding_base_url", value)} className="md:col-span-2" />
                <SettingField label="聊天 DNS 覆盖 IP" value={form.chat_resolve_ip} onChange={(value) => updateForm("chat_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="向量 DNS 覆盖 IP" value={form.embedding_resolve_ip} onChange={(value) => updateForm("embedding_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="聊天模型" value={form.chat_model} onChange={(value) => updateForm("chat_model", value)} />
                <SettingField label="向量模型" value={form.embedding_model} onChange={(value) => updateForm("embedding_model", value)} />
              </div>
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <label className="flex flex-col gap-2">
                  <ParameterName label="聊天接口密钥" />
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3">
                    <KeyRound className="size-4 text-cyan-100/58" />
                    <input
                      type="password"
                      value={showApiKeyMask ? "••••••••••••••••" : form.api_key}
                      readOnly={showApiKeyMask}
                      disabled={form.clear_api_key}
                      onChange={(event) => updateForm("api_key", event.target.value)}
                      placeholder={settings?.has_api_key ? "留空则保留当前密钥" : "输入接口密钥"}
                      className="h-11 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30"
                      autoComplete="off"
                    />
                    {showApiKeyMask ? (
                      <button type="button" onClick={() => setApiKeyEditing(true)} className="inline-flex items-center gap-1 rounded-full border border-white/8 px-2.5 py-1 text-xs text-white/55 transition hover:border-cyan-200/24 hover:text-cyan-100">
                        <PencilLine className="size-3.5" />
                        修改
                      </button>
                    ) : null}
                    <EyeOff className="size-4 text-white/32" />
                  </div>
                </label>

                <label className="flex flex-col gap-2">
                  <ParameterName label="向量接口密钥" />
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3">
                    <KeyRound className="size-4 text-cyan-100/58" />
                    <input
                      type="password"
                      value={showEmbeddingApiKeyMask ? "••••••••••••••••" : form.embedding_api_key}
                      readOnly={showEmbeddingApiKeyMask}
                      disabled={form.clear_embedding_api_key}
                      onChange={(event) => updateForm("embedding_api_key", event.target.value)}
                      placeholder={settings?.has_embedding_api_key ? "留空则保留当前密钥" : "输入向量接口密钥"}
                      className="h-11 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30"
                      autoComplete="off"
                    />
                    {showEmbeddingApiKeyMask ? (
                      <button type="button" onClick={() => setEmbeddingApiKeyEditing(true)} className="inline-flex items-center gap-1 rounded-full border border-white/8 px-2.5 py-1 text-xs text-white/55 transition hover:border-cyan-200/24 hover:text-cyan-100">
                        <PencilLine className="size-3.5" />
                        修改
                      </button>
                    ) : null}
                    <EyeOff className="size-4 text-white/32" />
                  </div>
                </label>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                <label className="flex items-center gap-3 border-l border-white/10 px-4 py-3 text-sm text-white/70">
                  <input
                    type="checkbox"
                    checked={form.clear_api_key}
                    onChange={(event) => {
                      updateForm("clear_api_key", event.target.checked);
                      if (event.target.checked) {
                        setApiKeyEditing(false);
                        updateForm("api_key", "");
                      }
                    }}
                    className="size-4 accent-rose-300"
                  />
                  <ParameterName label="清除当前聊天接口密钥" className="text-sm font-normal normal-case tracking-normal text-white/70" />
                </label>
                <label className="flex items-center gap-3 border-l border-white/10 px-4 py-3 text-sm text-white/70">
                  <input
                    type="checkbox"
                    checked={form.clear_embedding_api_key}
                    onChange={(event) => {
                      updateForm("clear_embedding_api_key", event.target.checked);
                      if (event.target.checked) {
                        setEmbeddingApiKeyEditing(false);
                        updateForm("embedding_api_key", "");
                      }
                    }}
                    className="size-4 accent-rose-300"
                  />
                  <ParameterName label="清除当前向量接口密钥" className="text-sm font-normal normal-case tracking-normal text-white/70" />
                </label>
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">调用与 Agent 预算</p>
              <BoundaryNote title="生效边界：下一次请求、下一次模型调用或下一次 Context Package 构建">
                模型请求并发、超时、embedding 批大小、Context Package 预算和 Agent envelope 会热加载；已开始的请求或批次按启动时快照继续执行。
              </BoundaryNote>
              <div className="mt-5 grid gap-4 md:grid-cols-4">
                <SettingField label="模型请求并发" type="number" min={1} max={16} value={form.model_request_concurrency} onChange={(value) => updateForm("model_request_concurrency", value)} />
                <SettingField label="模型超时秒数" type="number" min={5} max={600} value={form.model_request_timeout_seconds} onChange={(value) => updateForm("model_request_timeout_seconds", value)} />
                <SettingField label="Embedding 批大小" type="number" min={1} max={10} value={form.embedding_batch_size} onChange={(value) => updateForm("embedding_batch_size", value)} />
                <SettingField label="证据包 token 预算" type="number" min={256} max={20000} value={form.context_package_token_budget} onChange={(value) => updateForm("context_package_token_budget", value)} />
                <SettingField label="粗概念总预算" type="number" min={1} max={200} value={form.agent_coarse_total_budget} onChange={(value) => updateForm("agent_coarse_total_budget", value)} />
                <SettingField label="每个粗概念中概念预算" type="number" min={1} max={100} value={form.agent_mid_per_coarse_budget} onChange={(value) => updateForm("agent_mid_per_coarse_budget", value)} />
                <SettingField label="中概念 Top K" type="number" min={1} max={500} value={form.agent_mid_top_k} onChange={(value) => updateForm("agent_mid_top_k", value)} />
                <SettingField label="每个中概念片段预算" type="number" min={1} max={200} value={form.agent_chunk_per_mid_budget} onChange={(value) => updateForm("agent_chunk_per_mid_budget", value)} />
                <SettingField label="片段 Top K" type="number" min={1} max={1000} value={form.agent_chunk_top_k} onChange={(value) => updateForm("agent_chunk_top_k", value)} />
                <SettingField label="候选去重池预算" type="number" min={1} max={5000} value={form.candidate_pool_dedupe_budget} onChange={(value) => updateForm("candidate_pool_dedupe_budget", value)} />
                <SettingField label="每层最大深度" type="number" min={1} max={12} value={form.agent_max_depth_per_layer} onChange={(value) => updateForm("agent_max_depth_per_layer", value)} />
                <SettingField label="每节点标签上限" type="number" min={1} max={20} value={form.agent_max_labels_per_node} onChange={(value) => updateForm("agent_max_labels_per_node", value)} />
                <SettingField label="边复用上限" type="number" min={1} max={20} value={form.agent_max_edge_reuse} onChange={(value) => updateForm("agent_max_edge_reuse", value)} />
                <SettingField label="Cycle reward 上限" type="number" min={0} max={2} step={0.01} value={form.agent_max_cycle_reward_per_path} onChange={(value) => updateForm("agent_max_cycle_reward_per_path", value)} />
                <SettingField label="Cycle reward 距离阈值" type="number" min={0} max={20} step={0.01} value={form.agent_cycle_reward_distance_threshold} onChange={(value) => updateForm("agent_cycle_reward_distance_threshold", value)} />
                <SettingField label="路径 green 阈值" type="number" min={0} max={20} step={0.01} value={form.agent_path_distance_green_threshold} onChange={(value) => updateForm("agent_path_distance_green_threshold", value)} />
                <SettingField label="路径 gray 阈值" type="number" min={0} max={20} step={0.01} value={form.agent_path_distance_gray_threshold} onChange={(value) => updateForm("agent_path_distance_gray_threshold", value)} />
                <SettingField label="路径 hard 阈值" type="number" min={0} max={40} step={0.01} value={form.agent_path_distance_hard_threshold} onChange={(value) => updateForm("agent_path_distance_hard_threshold", value)} />
                <SettingField label="结构恢复预算" type="number" min={1} max={200} value={form.agent_structure_restore_budget} onChange={(value) => updateForm("agent_structure_restore_budget", value)} />
                <SettingField label="路径摘要预算" type="number" min={1} max={500} value={form.context_path_summary_budget} onChange={(value) => updateForm("context_path_summary_budget", value)} />
                <SettingField label="规划轮次预算" type="number" min={1} max={10} value={form.agent_planning_round_budget} onChange={(value) => updateForm("agent_planning_round_budget", value)} />
                <SettingField label="每轮动作上限" type="number" min={1} max={50} value={form.agent_max_typed_actions_per_round} onChange={(value) => updateForm("agent_max_typed_actions_per_round", value)} />
                <SettingField label="修复轮次预算" type="number" min={0} max={10} value={form.agent_repair_round_budget} onChange={(value) => updateForm("agent_repair_round_budget", value)} />
                <SettingField label="引用验证预算" type="number" min={1} max={100} value={form.agent_verification_budget} onChange={(value) => updateForm("agent_verification_budget", value)} />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">重建参数</p>
              <BoundaryNote title="生效边界：新任务会读取，但已有 active 数据不会改变">
                固定切块、向量维度、RQ-KMeans 和概念批处理参数必须通过重解析或图谱重建，才能影响已有资料库的 chunk、向量、关系图和概念图；L3 到中粒度、L2 到粗粒度始终全量投影。
              </BoundaryNote>
              <div className="mt-5">
                <SwitchRow
                  title="中粗层双语派生"
                  tooltip={SETTINGS_PARAMETER_HELP["中粗层双语派生"]}
                  description="默认关闭以避免额外模型成本。开启后下一次图谱重建会生成节点和关系的双语派生 metadata；前端图谱仍展示原字段。"
                  checked={form.concept_i18n_enabled}
                  onChange={() => updateForm("concept_i18n_enabled", !form.concept_i18n_enabled)}
                  disabled={saveMutation.isPending}
                  badge="热加载 / 下一次重建"
                />
              </div>
              <div className="mt-5 grid gap-4 md:grid-cols-4">
                <SettingField label="固定切块尺寸" type="number" min={128} max={4096} value={form.fixed_chunk_size_tokens} onChange={(value) => updateForm("fixed_chunk_size_tokens", value)} />
                <SettingField label="固定切块重叠" type="number" min={0} max={1024} value={form.fixed_chunk_overlap_tokens} onChange={(value) => updateForm("fixed_chunk_overlap_tokens", value)} />
                <SettingField label="向量维度" type="number" min={1} max={8192} value={form.embedding_dimensions} onChange={(value) => updateForm("embedding_dimensions", value)} />
                <SettingField label="模型批次诊断上限" type="number" min={0} max={64} value={form.mid_concept_extraction_max_model_batches} onChange={(value) => updateForm("mid_concept_extraction_max_model_batches", value)} />
                <SettingField label="每批 L3 前缀数" type="number" min={1} max={500} value={form.mid_concept_extraction_max_candidates_per_batch} onChange={(value) => updateForm("mid_concept_extraction_max_candidates_per_batch", value)} />
                <SettingField label="每批概念 token 上限" type="number" min={500} max={50000} value={form.mid_concept_extraction_max_tokens_per_batch} onChange={(value) => updateForm("mid_concept_extraction_max_tokens_per_batch", value)} />
                <SettingField label="候选诊断阈值" type="number" min={0} max={1} step={0.01} value={form.mid_concept_candidate_keep_threshold} onChange={(value) => updateForm("mid_concept_candidate_keep_threshold", value)} />
                <SettingField label="RQ-KMeans 层数" type="number" min={1} max={8} value={form.rq_kmeans_levels} onChange={(value) => updateForm("rq_kmeans_levels", value)} />
                <SettingField label="RQ-KMeans 最大 K" type="number" min={1} max={64} value={form.rq_kmeans_max_k} onChange={(value) => updateForm("rq_kmeans_max_k", value)} />
                <SettingField label="RQ 残差 Tau" type="number" min={0.01} max={10} step={0.01} value={form.rq_residual_tau} onChange={(value) => updateForm("rq_residual_tau", value)} />
                <SettingField label="Dense KNN 最小 K" type="number" min={1} max={200} value={form.dense_knn_k_min} onChange={(value) => updateForm("dense_knn_k_min", value)} />
                <SettingField label="Dense KNN 最大 K" type="number" min={1} max={500} value={form.dense_knn_k_max} onChange={(value) => updateForm("dense_knn_k_max", value)} />
                <SettingField label="基础互近邻下限" type="number" min={0} max={200} value={form.dense_reverse_b_min_base} onChange={(value) => updateForm("dense_reverse_b_min_base", value)} />
                <SettingField label="基础互近邻上限" type="number" min={1} max={500} value={form.dense_reverse_b_max_base} onChange={(value) => updateForm("dense_reverse_b_max_base", value)} />
                <SettingField label="跨文档互近邻下限" type="number" min={0} max={200} value={form.dense_reverse_b_min_doc} onChange={(value) => updateForm("dense_reverse_b_min_doc", value)} />
                <SettingField label="跨文档互近邻上限" type="number" min={0} max={500} value={form.dense_reverse_b_max_doc} onChange={(value) => updateForm("dense_reverse_b_max_doc", value)} />
                <SettingField label="跨语言互近邻下限" type="number" min={0} max={200} value={form.dense_reverse_b_min_lang} onChange={(value) => updateForm("dense_reverse_b_min_lang", value)} />
                <SettingField label="跨语言互近邻上限" type="number" min={0} max={500} value={form.dense_reverse_b_max_lang} onChange={(value) => updateForm("dense_reverse_b_max_lang", value)} />
                <SettingField label="Dense 最小余弦" type="number" min={0} max={1} step={0.01} value={form.dense_min_cosine} onChange={(value) => updateForm("dense_min_cosine", value)} />
                <SettingField label="Dense 强边余弦" type="number" min={0} max={1} step={0.01} value={form.dense_strong_cosine} onChange={(value) => updateForm("dense_strong_cosine", value)} />
                <SettingField label="跨文档桥最小配额" type="number" min={0} max={200} value={form.cross_doc_out_quota_min} onChange={(value) => updateForm("cross_doc_out_quota_min", value)} />
                <SettingField label="跨文档桥最大配额" type="number" min={0} max={500} value={form.cross_doc_out_quota_max} onChange={(value) => updateForm("cross_doc_out_quota_max", value)} />
                <SettingField label="跨文档桥最小余弦" type="number" min={0} max={1} step={0.01} value={form.cross_doc_min_cosine} onChange={(value) => updateForm("cross_doc_min_cosine", value)} />
                <SettingField label="跨语言桥最小配额" type="number" min={0} max={200} value={form.cross_language_out_quota_min} onChange={(value) => updateForm("cross_language_out_quota_min", value)} />
                <SettingField label="跨语言桥最大配额" type="number" min={0} max={500} value={form.cross_language_out_quota_max} onChange={(value) => updateForm("cross_language_out_quota_max", value)} />
                <SettingField label="跨语言桥最小余弦" type="number" min={0} max={1} step={0.01} value={form.cross_language_min_cosine} onChange={(value) => updateForm("cross_language_min_cosine", value)} />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">服务重启参数</p>
              <BoundaryNote title="生效边界：必须重启或重建 worker 服务">
                Docker Compose 里 Celery worker 通过 --concurrency 启动；运行中保存工作进程并发不会改变现有 worker 池。
              </BoundaryNote>
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <SettingField label="工作进程并发" type="number" min={1} max={32} value={form.worker_concurrency} onChange={(value) => updateForm("worker_concurrency", value)} />
              </div>
            </section>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-5">
              <p className="text-xs leading-6 text-white/42">
                保存会先规范化 .env 并广播运行时版本；回退开关不在页面开放开启，生产路径必须保持禁用。
              </p>
              <div className="flex items-center gap-2">
                {savedMessage ? <span className="text-sm text-emerald-100">{savedMessage}</span> : null}
                <Button type="button" variant="outline" className="rounded-full" onClick={() => void handleRuntimeCheck()}>
                  <RotateCcw data-icon="inline-start" />
                  检测
                </Button>
                <Button type="submit" className="rounded-full" disabled={saveMutation.isPending}>
                  {saveMutation.isPending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Save data-icon="inline-start" />}
                  保存设置
                </Button>
              </div>
            </div>
          </form>
        </div>
      </section>

      {runtimeWarnings.length ? (
        <section className="glass-panel rounded-[24px] p-5">
          <p className="flex items-center gap-2 text-sm font-semibold text-amber-100">
            <XCircle className="size-4" />
            运行时警告
          </p>
          <div className="mt-3 grid gap-2">
            {runtimeWarnings.map((issue) => (
              <p key={issue.code} className="text-sm leading-6 text-white/58">
                {issue.title}: {issue.message}
              </p>
            ))}
          </div>
        </section>
      ) : null}

      <ErrorDialog state={errorDialog} onClose={() => setErrorDialog(null)} />
    </div>
  );
}
