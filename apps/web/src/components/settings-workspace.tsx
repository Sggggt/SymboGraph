"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type {
  ModelSettingsResponse,
  ModelSettingsUpdate,
  RuntimeIssue,
  RuntimeSettingsCandidateResponse,
  StrategyProfileDetail,
  StructuredApiErrorBody,
} from "@course-kg/shared";
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
import { cn } from "@/lib/utils";
import {
  bindStrategyProfile,
  copyStrategyProfile,
  createRuntimeSettingsCandidate,
  createStrategyProfile,
  deleteStrategyProfile,
  fetchModelSettings,
  fetchRuntimeCheck,
  fetchRuntimeSettingsCandidate,
  fetchStrategyProfile,
  fetchStrategyProfiles,
  promoteRuntimeSettingsCandidate,
  runRuntimeSettingsCandidateAction,
  streamProfileAssistant,
  updateModelSettings,
  updateStrategyProfile,
} from "@/lib/api";

type SettingsForm = {
  chat_api_protocol: "openai" | "anthropic";
  graph_api_protocol: "openai" | "anthropic";
  embedding_api_protocol: "openai";
  chat_base_url: string;
  graph_base_url: string;
  embedding_base_url: string;
  chat_resolve_ip: string;
  graph_resolve_ip: string;
  embedding_resolve_ip: string;
  embedding_model: string;
  chat_model: string;
  graph_model: string;
  embedding_dimensions: string;
  embedding_batch_size: string;
  worker_concurrency: string;
  model_request_concurrency: string;
  model_request_timeout_seconds: string;
  retrieval_total_timeout_seconds: string;
  retrieval_generation_timeout_seconds: string;
  retrieval_planning_max_tokens: string;
  retrieval_generation_max_tokens: string;
  chat_json_max_tokens: string;
  agent_request_concurrency: string;
  source_io_concurrency: string;
  agent_request_queue_limit: string;
  agent_request_queue_timeout_seconds: string;
  agent_request_lease_ttl_seconds: string;
  upload_max_bytes: string;
  concept_i18n_enabled: boolean;
  query_facet_bilingual_enabled: boolean;
  ingestion_memory_soft_limit_ratio: string;
  ingestion_memory_hard_limit_ratio: string;
  ingestion_memory_critical_limit_ratio: string;
  fixed_chunk_size_tokens: string;
  fixed_chunk_overlap_tokens: string;
  chat_api_key: string;
  clear_chat_api_key: boolean;
  graph_api_key: string;
  clear_graph_api_key: boolean;
  embedding_api_key: string;
  clear_embedding_api_key: boolean;
  model_bridge_enabled: boolean;
  mid_concept_extraction_max_model_batches: string;
  mid_concept_extraction_max_candidates_per_batch: string;
  mid_concept_extraction_max_tokens_per_batch: string;
  mid_concept_candidate_keep_threshold: string;
  rq_kmeans_max_k: string;
  rq_residual_tau: string;
  edge_distance_protocol: "edge_distance_log_calibrated_strength_v2";
  rq_membership_protocol: "rq_primary_chain_v1";
  edge_projection_protocol: "membership_q15_layer_type_calibrated_v3";
  edge_type_calibration_protocol: "type_local_winsorized_minmax_v1";
  rq_membership_temperature: string;
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
  lexical_index_max_documents: string;
  lexical_index_max_postings: string;
  lexical_index_max_characters: string;
  bm25_k1: string;
  bm25_b: string;
};

type ErrorDialogState = {
  title: string;
  message: string;
  status?: number;
  issues: RuntimeIssue[];
  fixCommands: string[];
};

type AgentAdmissionSettingsValues = Pick<
  SettingsForm,
  "agent_request_concurrency" | "agent_request_queue_limit" | "agent_request_queue_timeout_seconds" | "agent_request_lease_ttl_seconds"
>;

type AgentTimeBudgetSettingsValues = Pick<
  SettingsForm,
  "retrieval_total_timeout_seconds" | "retrieval_generation_timeout_seconds"
>;

type AgentTokenBudgetSettingsValues = Pick<
  SettingsForm,
  "retrieval_planning_max_tokens" | "retrieval_generation_max_tokens"
>;

type GraphProtocolSettingsValues = Pick<
  SettingsForm,
  | "edge_distance_protocol"
  | "rq_membership_protocol"
  | "edge_projection_protocol"
  | "edge_type_calibration_protocol"
  | "rq_membership_temperature"
>;

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
const sectionClass = "min-w-0 break-words rounded-2xl border border-white/10 bg-white/[0.035] p-5";

type RuntimeSettingsPage =
  | "connections"
  | "runtime"
  | "retrieval"
  | "graph"
  | "build"
  | "deployment";

const runtimeSettingsPages: Array<{
  id: RuntimeSettingsPage;
  label: string;
  description: string;
}> = [
  { id: "connections", label: "模型连接", description: "协议、地址、模型与密钥" },
  { id: "runtime", label: "运行控制", description: "并发、超时、排队与上传" },
  { id: "retrieval", label: "检索入口", description: "候选、预算与 BM25" },
  { id: "graph", label: "图协议", description: "距离、RQ 与投影协议" },
  { id: "build", label: "重建参数", description: "切块、向量、关系与候选" },
  { id: "deployment", label: "服务参数", description: "重启后生效的工作进程" },
];

export function RuntimeSettingsNavigation({
  activePage,
  onChange,
}: {
  activePage: RuntimeSettingsPage;
  onChange: (page: RuntimeSettingsPage) => void;
}) {
  return (
    <nav
      aria-label="运行设置分类"
      data-testid="runtime-settings-navigation"
      className="min-w-0 max-w-full overflow-hidden rounded-3xl border border-white/10 bg-white/[0.035] p-3 shadow-[0_18px_64px_rgba(0,0,0,0.16)] xl:sticky xl:top-[8.5rem]"
    >
      <p className="px-3 pb-2 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-100/52">
        参数分类
      </p>
      <div className="custom-scrollbar flex w-full min-w-0 gap-2 overflow-x-auto pb-1 xl:grid xl:grid-cols-1 xl:overflow-visible xl:pb-0">
        {runtimeSettingsPages.map((page, index) => {
          const active = page.id === activePage;
          return (
            <button
              key={page.id}
              type="button"
              aria-current={active ? "page" : undefined}
              onClick={() => onChange(page.id)}
              className={cn(
                "flex min-w-[12rem] items-start gap-3 rounded-2xl border px-3 py-3 text-left transition xl:min-w-0",
                active
                  ? "border-cyan-200/28 bg-cyan-300/[0.085] text-white shadow-[0_10px_30px_rgba(41,177,255,0.08)]"
                  : "border-transparent text-white/58 hover:border-white/10 hover:bg-white/[0.035] hover:text-white",
              )}
            >
              <span className="grid size-7 shrink-0 place-items-center rounded-xl bg-white/[0.05] text-xs text-cyan-100/72">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-medium">{page.label}</span>
                <span className="mt-0.5 block text-xs leading-5 text-white/42">{page.description}</span>
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
const parameterNameClass = "text-xs uppercase tracking-[0.2em] text-cyan-100/46";
export const UPLOAD_MAX_BYTES_LIMITS = { defaultValue: 104_857_600, min: 1, max: 10_737_418_240 } as const;
export const RUNTIME_ENV_AUTHORITY_NOTE =
  "根 .env 管理秘密、连接与服务启动参数，根 settings.json 管理非秘密运行与构建参数；每个键只属于一个文件。热加载参数作用于下一次请求，重建参数等待显式重建与晋升，服务参数在重启后生效。";
export const RQ_KMEANS_PROTOCOL_DEPTH = 3 as const;

export const SETTINGS_PARAMETER_HELP: Record<string, string> = {
  资料库类型: "标记当前配置档适用的资料库类别，只影响提示词、界面标签和对话偏好，不参与切块、构图或检索参数。",
  名称: "配置档在设置页和资料库绑定列表里的显示名称，便于区分不同交互风格。",
  模型桥: "开启后 API 和 worker 容器优先通过本机模型桥访问聊天与向量端点，适合宿主机运行本地模型服务的场景。",
  聊天接口协议: "选择 openai 时调用 /chat/completions；选择 anthropic 时调用 /v1/messages。只影响对话模型，不改变 gray-zone 判定。",
  图谱接口协议: "选择 openai 时调用 /chat/completions；选择 anthropic 时调用 /v1/messages。它只决定后续构图模型传输，不参与图检索判定。",
  向量接口协议: "Embedding 当前仅支持 OpenAI-compatible 协议。该字段是独立 rebuild identity，不复用聊天或图谱协议，也不表示已支持 Anthropic embedding。",
  聊天基础地址: "对话接口的 base URL；协议为 anthropic 时填写服务根地址，由系统固定追加 /v1/messages。只影响下一次意图规划、一次回答生成和 Profile 助手调用。",
  图谱基础地址: "图谱构建接口的 base URL；协议为 anthropic 时填写服务根地址，由系统固定追加 /v1/messages。只影响下一次构图中的概念命名、粗概念和双语派生调用。",
  向量基础地址: "Embedding 接口的 base URL；后续解析、重嵌入和图谱重建会用它生成 contextual embedding。",
  "聊天 DNS 覆盖 IP": "仅对对话端点使用的 DNS 覆盖；需要固定解析到指定 IP 时填写，留空则使用系统 DNS。",
  "图谱 DNS 覆盖 IP": "仅对图谱构建端点使用的 DNS 覆盖；需要固定解析到指定 IP 时填写，留空则使用系统 DNS。",
  "向量 DNS 覆盖 IP": "仅对向量端点使用的 DNS 覆盖；需要固定解析到指定 IP 时填写，留空则使用系统 DNS。",
  聊天模型: "用于一次意图与执行策略规划、来源准入后的单次回答生成，以及 Profile 助手。",
  图谱模型: "用于中概念命名、粗概念生成和中粗层双语派生的图谱构建模型名称。",
  向量模型: "用于资料 embedding、dense relation 候选和查询向量的模型名称；改变后已有向量需要显式重解析或重建。",
  聊天接口密钥: "对话模型端点的访问密钥。留空会保留已有密钥，页面不会回显真实密钥。",
  图谱接口密钥: "图谱构建模型端点的访问密钥。它不会复用对话密钥，留空会保留已有图谱密钥。",
  向量接口密钥: "Embedding 端点的访问密钥。留空会保留已有密钥，页面不会回显真实密钥。",
  清除当前聊天接口密钥: "勾选后保存会删除当前对话密钥；删除后对话模型调用会因缺少凭据而失败。",
  清除当前图谱接口密钥: "勾选后保存会删除当前图谱密钥；删除后构图模型调用会因缺少凭据而失败。",
  清除当前向量接口密钥: "勾选后保存会删除当前向量密钥；删除后解析、重嵌入和检索向量生成会因缺少凭据而失败。",
  模型请求并发: "限制同时发起的模型请求数量，用于控制概念生成、意图规划和回答生成的吞吐与外部端点压力。",
  模型超时秒数: "单次模型请求等待上限；超过该时间会快速失败并进入可诊断错误，不做静默降级。",
  "Agent 整链总时限（秒）": "从请求接纳到最终终态的总墙钟上限，覆盖规划、检索、证据读取、最终生成和提交；它不会扩大任一单次模型调用的上限。",
  "最终生成时限（秒）": "一次最终回答生成的阶段上限，允许 10–600 秒；实际等待取该值、单次模型时限和整链剩余时间中的最小值。",
  "规划输出 token 上限": "意图规划和证据决策单次模型响应的最大 token 数；提高它会增加潜在生成时间和成本。",
  "最终生成 token 上限": "最终回答单次模型响应的最大 token 数；这是可用上限，不要求模型必须生成到该长度。",
  "源文件 I/O 并发": "限制解析、校验和持久化源文件时同时运行的阻塞 I/O 数量；通过有界 semaphore 热加载，防止文件线程无界扩张。",
  "Agent 请求并发": "普通 Agent 与 SSE 请求共用的全局并发上限；Redis 租约跨 API 进程协调，不以进程内任务表作为正确性边界。",
  "Agent 等待队列上限": "全局并发已满时允许进入 Redis FIFO 等待队列的请求数；队列满后立即返回可重试的 429 诊断。",
  "Agent 排队超时秒数": "请求在有界队列内允许等待的最长时间；等待期间不会创建数据库会话、Agent 审计记录或后台任务。",
  "Agent 租约 TTL 秒数": "活跃请求的 Redis 租约失效时间；心跳持续续租，进程崩溃后由 TTL 自动回收占用。",
  "Embedding 批大小": "每批提交给向量端点的文本数量；较大批次提升吞吐，但会增加单次请求体积和失败重试成本。",
  "单文件上传上限（字节）": "上传流允许的最大字节数；服务逐块计数并在越界时立即中断、清理临时文件，不会把超限内容注册为资料。",
  "证据包 token 预算": "Context Package 可容纳的证据 token 上限；它约束进入回答生成的唯一证据输入规模。",
  中粗层双语派生: "开启后，下一次图谱重建会对 mid/coarse 概念节点和高层概念边额外生成中英双语派生 metadata；关闭时不会产生这部分模型调用成本。",
  "LLM 双语查询面": "开启后，QA 查询面提取会要求 LLM 为显式领域和过程 facet 生成中英双语 aliases；它只影响下一次检索路由，不写事实证据，也不触发图谱重建。",
  "Dense 候选预算": "每层 Dense 通道独立提名的候选上限，进入本轮 ExecutionStrategy 的冻结预算。",
  "RQ 候选预算": "每层按完整 RQ 前缀重构向量计算相关性后独立提名的候选上限。",
  "BM25 候选预算": "混合策略启用时从 active 原文 BM25 快照读取的候选上限；纯向量策略不依赖该索引。",
  "根入口预算": "LLM 选择 coarse、mid 或 chunk 入口后，根层融合保留的入口数量。",
  "逐父节点预算": "从每个父节点分别下钻时保留的子候选数量，防止全局 top-k 吞掉小主题。",
  "单层总预算": "每一层逐父合并和累计距离遍历后最多保留的节点数量。",
  "最大遍历深度": "每层按非负累计距离扩展的最大深度；循环只用于剪枝，不增加优先级。",
  "每命中恢复预算": "每个命中片段允许追加的结构上下文数量；完整来源范围仍按跨度和证据包总预算处理。",
  "BM25 文档上限": "单个候选原文索引允许物化的 active chunk 文档数量硬上限。",
  "BM25 postings 上限": "单个候选索引允许写入的 postings 总量硬上限。",
  "BM25 原文字符上限": "单个候选索引允许读取并物化的原文字符总量硬上限。",
  "BM25 k1": "BM25 词频饱和参数。改变后必须生成并发布新的 lexical 快照。",
  "BM25 b": "BM25 文档长度归一参数。改变后必须生成并发布新的 lexical 快照。",
  "边距离协议": "本地 allowlist 的关系强度到累计距离转换协议。它改变 active relation graph 语义，只能经 candidate、shadow rebuild、evaluation 和 promotion 生效。",
  "RQ membership 协议": "本地 allowlist 的 RQ 主链归属协议。LLM、prompt 和自由表达式都不能成为协议值；变更必须重建 RQ 与下游概念图。",
  "边投影协议": "底层 chunk relation edge 向 mid/coarse 概念边投影的本地协议；support ids 与 gray predicates 都由确定性实现约束。",
  "边类型校准协议": "按 edge type 独立校准 raw strength 的本地协议；变更必须重新校准并重建 active relation graph。",
  "RQ softmax 温度": "逐层完整 codebook softmax 的温度 τ_l；它改变 primary membership 概率，必须通过 candidate 重建与 promotion 生效。",
  "RQ 每层候选上限": "每层保留的非主 code 稀疏候选上限；主 residual trajectory 始终保留，不受该值裁掉。",
  "RQ 概率裁剪阈值": "仅裁剪非主 code 的原始完整 softmax 概率阈值；membership 不重归一、不设人工下限。",
  固定切块尺寸: "解析时每个稳定 chunk 的目标 token 大小；chunk 是索引和引用地址单位，不假定是完整语义单元。",
  固定切块重叠: "相邻固定 chunk 之间保留的 token 重叠，用来降低边界截断造成的上下文损失。",
  向量维度: "Embedding 向量维数，必须与向量模型和 Qdrant collection 一致；改变后需要重嵌入或重建派生索引。",
  模型批次诊断上限: "构建 mid concept 时最多抽样多少个 LLM 批次做概念诊断；0 表示关闭这类模型诊断。",
  "每批 L3 前缀数": "每个概念生成批次最多处理的 RQ L3 prefix packet 数量，影响 mid concept 生成吞吐。",
  "每批概念 token 上限": "单个概念生成批次允许传入模型的 token 上限，防止 prompt 过大。",
  候选诊断阈值: "mid concept 候选保留诊断的 membership/质量阈值，用于标记低置信候选而不是直接制造事实。",
  "RQ-KMeans 协议深度": "当前 Four-Layer active protocol 固定为 3：L3 对齐 mid concept、L2 对齐 coarse concept。该值不是可调构图参数。",
  "RQ-KMeans 最大 K": "每层 RQ-KMeans 聚类的最大分支数，影响 RQ prefix 地址空间粒度。",
  "RQ 残差 Tau": "控制 RQ primary membership 的残差距离温度；值越小，membership 权重越集中。",
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

export function ModelProtocolSelect({
  label,
  value,
  onChange,
  disabled,
  lifecycle,
}: {
  label: "聊天接口协议" | "图谱接口协议";
  value: "openai" | "anthropic";
  onChange: (value: "openai" | "anthropic") => void;
  disabled?: boolean;
  lifecycle: string;
}) {
  return (
    <label className="flex min-w-0 max-w-full flex-col gap-2">
      <ParameterName label={label} />
      <select
        value={value}
        onChange={(event) =>
          onChange(event.target.value as "openai" | "anthropic")
        }
        disabled={disabled}
        className={`${inputClass} appearance-none`}
      >
        <option value="openai">OpenAI-compatible</option>
        <option value="anthropic">Anthropic Messages</option>
      </select>
      <span className="text-xs leading-5 text-cyan-50/52">{lifecycle}</span>
    </label>
  );
}

export function EmbeddingProtocolSelect({
  value,
  onChange,
  disabled,
}: {
  value: "openai";
  onChange: (value: "openai") => void;
  disabled?: boolean;
}) {
  return (
    <label className="flex min-w-0 max-w-full flex-col gap-2">
      <ParameterName label="向量接口协议" />
      <select
        value={value}
        onChange={() => onChange("openai")}
        disabled={disabled}
        className={`${inputClass} appearance-none`}
      >
        <option value="openai">OpenAI-compatible（当前唯一支持）</option>
      </select>
      <span className="text-xs leading-5 text-cyan-50/52">
        rebuild_required · stage as candidate, then rebuild embedding/graph artifacts and promote
      </span>
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

export function SourceIoConcurrencyField({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <SettingField
      label="源文件 I/O 并发"
      type="number"
      min={1}
      max={64}
      value={value}
      onChange={onChange}
    />
  );
}

export function AgentTimeBudgetFields({
  values,
  onChange,
}: {
  values: AgentTimeBudgetSettingsValues;
  onChange: (key: keyof AgentTimeBudgetSettingsValues, value: string) => void;
}) {
  return (
    <>
      <SettingField label="Agent 整链总时限（秒）" type="number" min={15} max={3600} value={values.retrieval_total_timeout_seconds} onChange={(value) => onChange("retrieval_total_timeout_seconds", value)} />
      <SettingField label="最终生成时限（秒）" type="number" min={10} max={600} value={values.retrieval_generation_timeout_seconds} onChange={(value) => onChange("retrieval_generation_timeout_seconds", value)} />
    </>
  );
}

export function AgentTokenBudgetFields({
  values,
  onChange,
}: {
  values: AgentTokenBudgetSettingsValues;
  onChange: (key: keyof AgentTokenBudgetSettingsValues, value: string) => void;
}) {
  return (
    <>
      <SettingField label="规划输出 token 上限" type="number" min={256} max={8192} value={values.retrieval_planning_max_tokens} onChange={(value) => onChange("retrieval_planning_max_tokens", value)} />
      <SettingField label="最终生成 token 上限" type="number" min={256} max={32768} value={values.retrieval_generation_max_tokens} onChange={(value) => onChange("retrieval_generation_max_tokens", value)} />
    </>
  );
}

export function UploadSecuritySettingsSection({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return (
    <section className={sectionClass}>
      <p className="text-sm font-semibold text-white">上传安全参数</p>
      <BoundaryNote title="生效边界：下一次上传请求">
        单文件上限会热加载；上传服务逐块执行硬限制，超限或读取失败时清理未提交的临时文件，不触发资料库重建。
      </BoundaryNote>
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        <SettingField
          label="单文件上传上限（字节）"
          type="number"
          min={UPLOAD_MAX_BYTES_LIMITS.min}
          max={UPLOAD_MAX_BYTES_LIMITS.max}
          value={value}
          onChange={onChange}
        />
      </div>
    </section>
  );
}

export function RqProtocolDepthField() {
  return (
    <SettingField
      label="RQ-KMeans 协议深度"
      type="number"
      min={RQ_KMEANS_PROTOCOL_DEPTH}
      max={RQ_KMEANS_PROTOCOL_DEPTH}
      value={String(RQ_KMEANS_PROTOCOL_DEPTH)}
      onChange={() => undefined}
      disabled
    />
  );
}

export function GraphProtocolSettingsSection({
  values,
  onChange,
}: {
  values: GraphProtocolSettingsValues;
  onChange: (key: keyof GraphProtocolSettingsValues, value: string) => void;
}) {
  return (
    <section className={sectionClass}>
      <p className="text-sm font-semibold text-white">构图协议与 RQ membership</p>
      <BoundaryNote title="生效边界：candidate → shadow rebuild → evaluation → promotion">
        普通保存会立即写入对应的根配置文件，但不会提前改写已有图；请在下方候选生命周期面板完成 dry-run、真实 shadow build、数值评估和原子 promotion。
      </BoundaryNote>
      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <SettingField label="边距离协议" value={values.edge_distance_protocol} onChange={() => undefined} disabled />
        <SettingField label="RQ membership 协议" value={values.rq_membership_protocol} onChange={() => undefined} disabled />
        <SettingField label="边投影协议" value={values.edge_projection_protocol} onChange={() => undefined} disabled />
        <SettingField label="边类型校准协议" value={values.edge_type_calibration_protocol} onChange={() => undefined} disabled />
        <SettingField label="RQ softmax 温度" type="number" min={0.01} max={10} step={0.01} value={values.rq_membership_temperature} onChange={(value) => onChange("rq_membership_temperature", value)} />
      </div>
    </section>
  );
}

export function AgentAdmissionSettingsSection({
  values,
  onChange,
}: {
  values: AgentAdmissionSettingsValues;
  onChange: (key: keyof AgentAdmissionSettingsValues, value: string) => void;
}) {
  return (
    <section className={sectionClass}>
      <p className="text-sm font-semibold text-white">Agent 请求准入</p>
      <BoundaryNote title="生效边界：下一次普通 Agent 或 SSE 请求">
        四项参数会热加载。Redis 统一执行跨进程并发与有界 FIFO 排队；Redis 不可用时请求会快速失败，不启用本地降级。
      </BoundaryNote>
      <div className="mt-5 grid gap-4 md:grid-cols-4">
        <SettingField label="Agent 请求并发" type="number" min={1} max={128} value={values.agent_request_concurrency} onChange={(value) => onChange("agent_request_concurrency", value)} />
        <SettingField label="Agent 等待队列上限" type="number" min={0} max={1000} value={values.agent_request_queue_limit} onChange={(value) => onChange("agent_request_queue_limit", value)} />
        <SettingField label="Agent 排队超时秒数" type="number" min={1} max={3600} value={values.agent_request_queue_timeout_seconds} onChange={(value) => onChange("agent_request_queue_timeout_seconds", value)} />
        <SettingField label="Agent 租约 TTL 秒数" type="number" min={5} max={7200} value={values.agent_request_lease_ttl_seconds} onChange={(value) => onChange("agent_request_lease_ttl_seconds", value)} />
      </div>
    </section>
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
      reason: "配置档 JSON 保存 library_type、prompt_pack、ui_labels 和 conversation_preferences；工程参数必须进入运行时设置。",
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
      reason: "资料库级系统提示词、回答风格、引用严格度表达和无上下文提示需要从 prompt_pack 读取。",
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
            <label className="flex min-w-0 max-w-full flex-col gap-2">
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
                    <Loader2 className="size-4 animate-spin" />
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
              placeholder="描述资料库类型、系统提示词、界面标签、回答风格、引用严格度表达、澄清方式和无上下文回复文案。"
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

export function bindEmbeddingProtocolToCandidateSettings(
  changedSettings: Record<string, unknown>,
  embeddingApiProtocol: "openai",
): Record<string, unknown> {
  if (!Object.keys(changedSettings).length) {
    return {};
  }
  return {
    ...changedSettings,
    embedding_api_protocol: embeddingApiProtocol,
  };
}

export function candidateChangedKeysForDisplay(
  candidateSettings: Record<string, unknown>,
  activeEmbeddingApiProtocol: "openai" | undefined,
): string[] {
  const activeProtocol = activeEmbeddingApiProtocol ?? "openai";
  return Object.keys(candidateSettings)
    .filter(
      (key) =>
        key !== "embedding_api_protocol" ||
        candidateSettings[key] !== activeProtocol,
    )
    .sort();
}

type HotReloadSettingsForm = Pick<
  SettingsForm,
  | "chat_api_protocol"
  | "chat_base_url"
  | "chat_resolve_ip"
  | "chat_model"
  | "embedding_batch_size"
  | "worker_concurrency"
  | "model_request_concurrency"
  | "model_request_timeout_seconds"
  | "retrieval_total_timeout_seconds"
  | "retrieval_generation_timeout_seconds"
  | "retrieval_planning_max_tokens"
  | "retrieval_generation_max_tokens"
  | "chat_json_max_tokens"
  | "agent_request_concurrency"
  | "source_io_concurrency"
  | "agent_request_queue_limit"
  | "agent_request_queue_timeout_seconds"
  | "agent_request_lease_ttl_seconds"
  | "upload_max_bytes"
  | "concept_i18n_enabled"
  | "query_facet_bilingual_enabled"
  | "ingestion_memory_soft_limit_ratio"
  | "ingestion_memory_hard_limit_ratio"
  | "ingestion_memory_critical_limit_ratio"
  | "chat_api_key"
  | "clear_chat_api_key"
  | "graph_api_key"
  | "clear_graph_api_key"
  | "embedding_api_key"
  | "clear_embedding_api_key"
  | "model_bridge_enabled"
  | "context_package_token_budget"
  | "retrieval_result_top_k_default"
  | "retrieval_v1_dense_candidate_budget"
  | "retrieval_v1_rq_candidate_budget"
  | "retrieval_v1_bm25_candidate_budget"
  | "retrieval_v1_root_entry_budget"
  | "retrieval_v1_per_parent_entry_budget"
  | "retrieval_v1_layer_entry_budget"
  | "retrieval_v1_max_depth"
  | "retrieval_v1_restore_per_hit"
  | "lexical_index_max_documents"
  | "lexical_index_max_postings"
  | "lexical_index_max_characters"
>;

export function buildHotReloadSettingsPayload(
  form: HotReloadSettingsForm,
): ModelSettingsUpdate {
  return {
    chat_api_protocol: form.chat_api_protocol,
    chat_base_url: form.chat_base_url.trim(),
    chat_resolve_ip: form.chat_resolve_ip.trim() || null,
    chat_model: form.chat_model.trim(),
    embedding_batch_size: parseIntField(form.embedding_batch_size),
    worker_concurrency: parseIntField(form.worker_concurrency),
    model_request_concurrency: parseIntField(form.model_request_concurrency),
    model_request_timeout_seconds: parseIntField(form.model_request_timeout_seconds),
    retrieval_total_timeout_seconds: parseIntField(form.retrieval_total_timeout_seconds),
    retrieval_generation_timeout_seconds: parseIntField(form.retrieval_generation_timeout_seconds),
    retrieval_planning_max_tokens: parseIntField(form.retrieval_planning_max_tokens),
    retrieval_generation_max_tokens: parseIntField(form.retrieval_generation_max_tokens),
    chat_json_max_tokens: parseIntField(form.chat_json_max_tokens),
    agent_request_concurrency: parseIntField(form.agent_request_concurrency),
    source_io_concurrency: parseIntField(form.source_io_concurrency),
    agent_request_queue_limit: parseIntField(form.agent_request_queue_limit),
    agent_request_queue_timeout_seconds: parseIntField(form.agent_request_queue_timeout_seconds),
    agent_request_lease_ttl_seconds: parseIntField(form.agent_request_lease_ttl_seconds),
    upload_max_bytes: parseIntField(form.upload_max_bytes),
    concept_i18n_enabled: form.concept_i18n_enabled,
    query_facet_bilingual_enabled: form.query_facet_bilingual_enabled,
    ingestion_memory_soft_limit_ratio: parseFloatField(form.ingestion_memory_soft_limit_ratio),
    ingestion_memory_hard_limit_ratio: parseFloatField(form.ingestion_memory_hard_limit_ratio),
    ingestion_memory_critical_limit_ratio: parseFloatField(form.ingestion_memory_critical_limit_ratio),
    chat_api_key: form.chat_api_key.trim() || null,
    clear_chat_api_key: form.clear_chat_api_key,
    graph_api_key: form.graph_api_key.trim() || null,
    clear_graph_api_key: form.clear_graph_api_key,
    embedding_api_key: form.embedding_api_key.trim() || null,
    clear_embedding_api_key: form.clear_embedding_api_key,
    model_bridge_enabled: form.model_bridge_enabled,
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
    lexical_index_max_documents: parseIntField(form.lexical_index_max_documents),
    lexical_index_max_postings: parseIntField(form.lexical_index_max_postings),
    lexical_index_max_characters: parseIntField(form.lexical_index_max_characters),
  };
}

export function buildRuntimeSettingsPayload(form: SettingsForm): ModelSettingsUpdate {
  return {
    ...buildHotReloadSettingsPayload(form),
    embedding_api_protocol: form.embedding_api_protocol,
    fixed_chunk_size_tokens: parseIntField(form.fixed_chunk_size_tokens),
    fixed_chunk_overlap_tokens: parseIntField(form.fixed_chunk_overlap_tokens),
    embedding_base_url: form.embedding_base_url.trim(),
    embedding_resolve_ip: form.embedding_resolve_ip.trim() || null,
    embedding_model: form.embedding_model.trim(),
    embedding_dimensions: parseIntField(form.embedding_dimensions),
    bm25_k1: parseFloatField(form.bm25_k1),
    bm25_b: parseFloatField(form.bm25_b),
    graph_base_url: form.graph_base_url.trim(),
    graph_api_protocol: form.graph_api_protocol,
    graph_resolve_ip: form.graph_resolve_ip.trim() || null,
    graph_model: form.graph_model.trim(),
    edge_distance_protocol: form.edge_distance_protocol,
    rq_membership_protocol: form.rq_membership_protocol,
    edge_projection_protocol: form.edge_projection_protocol,
    edge_type_calibration_protocol: form.edge_type_calibration_protocol,
    rq_kmeans_max_k: parseIntField(form.rq_kmeans_max_k),
    rq_residual_tau: parseFloatField(form.rq_residual_tau),
    rq_membership_temperature: parseFloatField(form.rq_membership_temperature),
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
    mid_concept_extraction_max_model_batches: parseIntField(form.mid_concept_extraction_max_model_batches),
    mid_concept_extraction_max_candidates_per_batch: parseIntField(form.mid_concept_extraction_max_candidates_per_batch),
    mid_concept_extraction_max_tokens_per_batch: parseIntField(form.mid_concept_extraction_max_tokens_per_batch),
    mid_concept_candidate_keep_threshold: parseFloatField(form.mid_concept_candidate_keep_threshold),
  };
}

function rebuildCandidateSettings(
  form: SettingsForm,
  activeSettings: ModelSettingsResponse,
): Record<string, unknown> {
  const requested: Record<string, unknown> = {
    fixed_chunk_size_tokens: parseIntField(form.fixed_chunk_size_tokens),
    fixed_chunk_overlap_tokens: parseIntField(form.fixed_chunk_overlap_tokens),
    embedding_base_url: form.embedding_base_url.trim(),
    embedding_resolve_ip: form.embedding_resolve_ip.trim() || null,
    embedding_model: form.embedding_model.trim(),
    embedding_dimensions: parseIntField(form.embedding_dimensions),
    bm25_k1: parseFloatField(form.bm25_k1),
    bm25_b: parseFloatField(form.bm25_b),
    graph_base_url: form.graph_base_url.trim(),
    graph_api_protocol: form.graph_api_protocol,
    graph_resolve_ip: form.graph_resolve_ip.trim() || null,
    graph_model: form.graph_model.trim(),
    edge_distance_protocol: form.edge_distance_protocol,
    rq_membership_protocol: form.rq_membership_protocol,
    edge_projection_protocol: form.edge_projection_protocol,
    edge_type_calibration_protocol: form.edge_type_calibration_protocol,
    rq_kmeans_max_k: parseIntField(form.rq_kmeans_max_k),
    rq_residual_tau: parseFloatField(form.rq_residual_tau),
    rq_membership_temperature: parseFloatField(form.rq_membership_temperature),
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
    mid_concept_extraction_max_model_batches: parseIntField(form.mid_concept_extraction_max_model_batches),
    mid_concept_extraction_max_candidates_per_batch: parseIntField(form.mid_concept_extraction_max_candidates_per_batch),
    mid_concept_extraction_max_tokens_per_batch: parseIntField(form.mid_concept_extraction_max_tokens_per_batch),
    mid_concept_candidate_keep_threshold: parseFloatField(form.mid_concept_candidate_keep_threshold),
  };
  const active = activeSettings as unknown as Record<string, unknown>;
  const pendingRebuild = new Set(activeSettings.pending_rebuild_changes ?? []);
  const changedSettings = Object.fromEntries(
    Object.entries(requested).filter(([key, value]) => {
      if (value === undefined) {
        return false;
      }
      if (pendingRebuild.has(key)) {
        return true;
      }
      const activeValue = active[key] === "" ? null : active[key];
      return JSON.stringify(value) !== JSON.stringify(activeValue);
    }),
  );
  return bindEmbeddingProtocolToCandidateSettings(
    changedSettings,
    form.embedding_api_protocol,
  );
}

function RuntimeSettingsCandidatePanel({
  knowledgeBaseId,
  knowledgeBaseName,
  settings,
  form,
  onError,
}: {
  knowledgeBaseId: string | null;
  knowledgeBaseName?: string | null;
  settings: ModelSettingsResponse;
  form: SettingsForm;
  onError: (error: unknown) => void;
}) {
  const queryClient = useQueryClient();
  const [candidateId, setCandidateId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<RuntimeSettingsCandidateResponse | null>(null);
  const candidateSettings = useMemo(
    () => rebuildCandidateSettings(form, settings),
    [form, settings],
  );
  const changedKeys = candidateChangedKeysForDisplay(
    candidateSettings,
    settings.embedding_api_protocol,
  );
  const candidateQuery = useQuery({
    queryKey: ["runtime-settings-candidate", candidateId],
    queryFn: () => fetchRuntimeSettingsCandidate(String(candidateId)),
    enabled: Boolean(candidateId),
    refetchInterval: 3000,
  });
  const current = candidateQuery.data ?? lastResult;
  const candidate = current?.candidate ?? null;
  const build = candidate?.builds?.[0];
  const evaluation = build?.evaluation ?? {};
  const hardGates = (evaluation.hard_gates ?? {}) as Record<string, boolean>;

  const createMutation = useMutation({
    mutationFn: (dryRunOnly: boolean) => {
      if (!knowledgeBaseId) {
        throw new Error("请先选择资料库。");
      }
      if (!changedKeys.length) {
        throw new Error("没有待验证的 rebuild_required 参数变更。");
      }
      return createRuntimeSettingsCandidate({
        knowledge_base_ids: [knowledgeBaseId],
        settings: candidateSettings,
        dry_run_only: dryRunOnly,
        source: "settings_ui",
      });
    },
    onSuccess: async (result) => {
      setLastResult(result);
      if (result.candidate?.id) {
        setCandidateId(result.candidate.id);
      }
      await queryClient.invalidateQueries({ queryKey: ["runtime-settings-candidate"] });
    },
    onError,
  });

  const actionMutation = useMutation({
    mutationFn: ({ action, reason }: { action: "build" | "evaluate" | "promote" | "rollback"; reason?: string }) => {
      if (!candidateId) {
        throw new Error("请先创建 Runtime Settings candidate。");
      }
      return action === "promote"
        ? promoteRuntimeSettingsCandidate(candidateId)
        : runRuntimeSettingsCandidateAction(candidateId, action, {
            reason: reason ?? null,
          });
    },
    onSuccess: async (result) => {
      setLastResult(result);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["runtime-settings-candidate", candidateId] }),
        queryClient.invalidateQueries({ queryKey: ["model-settings"] }),
        queryClient.invalidateQueries({ queryKey: ["runtime-check"] }),
        queryClient.invalidateQueries({ queryKey: ["knowledgeBases"] }),
      ]);
    },
    onError,
  });

  const pending = createMutation.isPending || actionMutation.isPending;
  const preview = current?.preview ?? lastResult?.preview;
  const buildReady = build?.status === "shadow_ready";
  const evaluationPassed = build?.status === "evaluation_passed" && candidate?.status === "evaluation_passed";

  return (
    <section className={sectionClass} aria-label="Runtime Settings candidate 生命周期">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-white">Runtime Settings candidate 生命周期</p>
          <BoundaryNote title="强制流程：dry-run → shadow build → measured evaluation → atomic promotion">
            active PostgreSQL/Qdrant/图指针在 promotion 前保持不变；失败只保留可审计的 shadow 事实。灰区路径继续/停止始终由版本化本地规则判定，模型调用数为 0。
          </BoundaryNote>
        </div>
        <StatusPill ok={Boolean(knowledgeBaseId)}>{knowledgeBaseName || "未选择资料库"}</StatusPill>
      </div>

      <div className="mt-4 rounded-2xl border border-white/10 bg-black/10 p-4 text-sm leading-6 text-white/62">
        <p>待变更 rebuild_required 字段：{changedKeys.length ? changedKeys.join("、") : "无"}</p>
        {candidate ? (
          <p className="mt-1 break-all">
            候选配置状态：{candidate.status}
          </p>
        ) : null}
        {build ? (
          <p className="mt-1 break-all">
            影子重建状态：{build.status} · 候选片段版本 {build.candidate_chunk_version ?? "-"}
          </p>
        ) : null}
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <Button type="button" variant="outline" className="rounded-full" disabled={pending || !knowledgeBaseId || !changedKeys.length} onClick={() => createMutation.mutate(true)}>
          Dry-run
        </Button>
        <Button type="button" variant="outline" className="rounded-full" disabled={pending || !knowledgeBaseId || !changedKeys.length} onClick={() => createMutation.mutate(false)}>
          Stage candidate
        </Button>
        <Button type="button" variant="outline" className="rounded-full" disabled={pending || !candidate || !["staged", "failed"].includes(candidate.status)} onClick={() => actionMutation.mutate({ action: "build" })}>
          Build shadow
        </Button>
        <Button type="button" variant="outline" className="rounded-full" disabled={pending || !buildReady} onClick={() => actionMutation.mutate({ action: "evaluate" })}>
          Evaluate
        </Button>
        <Button type="button" className="rounded-full" disabled={pending || !evaluationPassed} onClick={() => actionMutation.mutate({ action: "promote" })}>
          Promote atomically
        </Button>
        <Button type="button" variant="outline" className="rounded-full" disabled={pending || candidate?.status !== "promoted"} onClick={() => actionMutation.mutate({ action: "rollback", reason: "settings_ui_explicit_rollback" })}>
          Roll back
        </Button>
        {pending ? <Loader2 className="size-5 animate-spin text-cyan-100" /> : null}
      </div>

      {preview ? (
        <div className="mt-4 grid gap-2 text-xs text-white/56 md:grid-cols-3">
          <p>Shadow rechunk: {String(preview.requires_shadow_rechunk ?? false)}</p>
          <p>Vector shadow: {String(preview.requires_vector_shadow ?? false)}</p>
          <p>Gray-zone model calls: {String(preview.gray_zone_rule_decision_model_call_count ?? 0)}</p>
        </div>
      ) : null}
      {Object.keys(hardGates).length ? (
        <div className="mt-4">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100/60">Measured hard gates</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(hardGates).map(([name, passed]) => (
              <StatusPill key={name} ok={passed}>{name}</StatusPill>
            ))}
          </div>
        </div>
      ) : null}
      {candidate?.blocking_reasons?.length ? (
        <p className="mt-4 text-sm leading-6 text-rose-100/80">阻断：{candidate.blocking_reasons.join("；")}</p>
      ) : null}
    </section>
  );
}

export function SettingsWorkspace() {
  const queryClient = useQueryClient();
  const { selectedKnowledgeBaseId, selectedKnowledgeBase } = useKnowledgeBaseContext();
  const settingsQuery = useQuery({ queryKey: ["model-settings"], queryFn: fetchModelSettings });
  const runtimeQuery = useQuery({ queryKey: ["runtime-check"], queryFn: () => fetchRuntimeCheck(), retry: false });
  const [form, setForm] = useState<SettingsForm | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [apiKeyEditing, setApiKeyEditing] = useState(false);
  const [graphApiKeyEditing, setGraphApiKeyEditing] = useState(false);
  const [embeddingApiKeyEditing, setEmbeddingApiKeyEditing] = useState(false);
  const [errorDialog, setErrorDialog] = useState<ErrorDialogState | null>(null);
  const [activeTab, setActiveTab] = useState<"model" | "profile">("model");
  const [activeSettingsPage, setActiveSettingsPage] = useState<RuntimeSettingsPage>("connections");

  useEffect(() => {
    if (!settingsQuery.data) {
      return;
    }
    const displayedSettings = settingsQuery.data;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setForm({
      chat_api_protocol: displayedSettings.chat_api_protocol ?? "openai",
      graph_api_protocol: displayedSettings.graph_api_protocol ?? "openai",
      embedding_api_protocol: displayedSettings.embedding_api_protocol ?? "openai",
      chat_base_url: displayedSettings.chat_base_url ?? "",
      graph_base_url: displayedSettings.graph_base_url ?? "",
      embedding_base_url: displayedSettings.embedding_base_url ?? "",
      chat_resolve_ip: displayedSettings.chat_resolve_ip ?? "",
      graph_resolve_ip: displayedSettings.graph_resolve_ip ?? "",
      embedding_resolve_ip: displayedSettings.embedding_resolve_ip ?? "",
      embedding_model: displayedSettings.embedding_model ?? "",
      chat_model: displayedSettings.chat_model ?? "",
      graph_model: displayedSettings.graph_model ?? "",
      embedding_dimensions: String(displayedSettings.embedding_dimensions ?? 1024),
      embedding_batch_size: String(displayedSettings.embedding_batch_size ?? 10),
      worker_concurrency: String(displayedSettings.worker_concurrency ?? 3),
      model_request_concurrency: String(displayedSettings.model_request_concurrency ?? 3),
      model_request_timeout_seconds: String(displayedSettings.model_request_timeout_seconds ?? 240),
      retrieval_total_timeout_seconds: String(displayedSettings.retrieval_total_timeout_seconds ?? 540),
      retrieval_generation_timeout_seconds: String(displayedSettings.retrieval_generation_timeout_seconds ?? 240),
      retrieval_planning_max_tokens: String(displayedSettings.retrieval_planning_max_tokens ?? 8192),
      retrieval_generation_max_tokens: String(displayedSettings.retrieval_generation_max_tokens ?? 32768),
      chat_json_max_tokens: String(displayedSettings.chat_json_max_tokens ?? 12000),
      agent_request_concurrency: String(displayedSettings.agent_request_concurrency ?? 4),
      source_io_concurrency: String(displayedSettings.source_io_concurrency ?? 4),
      agent_request_queue_limit: String(displayedSettings.agent_request_queue_limit ?? 8),
      agent_request_queue_timeout_seconds: String(displayedSettings.agent_request_queue_timeout_seconds ?? 30),
      agent_request_lease_ttl_seconds: String(displayedSettings.agent_request_lease_ttl_seconds ?? 300),
      upload_max_bytes: String(displayedSettings.upload_max_bytes ?? UPLOAD_MAX_BYTES_LIMITS.defaultValue),
      concept_i18n_enabled: displayedSettings.concept_i18n_enabled ?? false,
      query_facet_bilingual_enabled: displayedSettings.query_facet_bilingual_enabled ?? false,
      ingestion_memory_soft_limit_ratio: String(displayedSettings.ingestion_memory_soft_limit_ratio ?? 0.78),
      ingestion_memory_hard_limit_ratio: String(displayedSettings.ingestion_memory_hard_limit_ratio ?? 0.88),
      ingestion_memory_critical_limit_ratio: String(displayedSettings.ingestion_memory_critical_limit_ratio ?? 0.94),
      fixed_chunk_size_tokens: String(displayedSettings.fixed_chunk_size_tokens ?? 512),
      fixed_chunk_overlap_tokens: String(displayedSettings.fixed_chunk_overlap_tokens ?? 80),
      chat_api_key: "",
      clear_chat_api_key: false,
      graph_api_key: "",
      clear_graph_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: displayedSettings.model_bridge_enabled ?? true,
      mid_concept_extraction_max_model_batches: String(displayedSettings.mid_concept_extraction_max_model_batches ?? 4),
      mid_concept_extraction_max_candidates_per_batch: String(displayedSettings.mid_concept_extraction_max_candidates_per_batch ?? 8),
      mid_concept_extraction_max_tokens_per_batch: String(displayedSettings.mid_concept_extraction_max_tokens_per_batch ?? 2400),
      mid_concept_candidate_keep_threshold: String(displayedSettings.mid_concept_candidate_keep_threshold ?? 0.62),
      rq_kmeans_max_k: String(displayedSettings.rq_kmeans_max_k ?? 6),
      rq_residual_tau: String(displayedSettings.rq_residual_tau ?? 0.65),
      edge_distance_protocol: displayedSettings.edge_distance_protocol ?? "edge_distance_log_calibrated_strength_v2",
      rq_membership_protocol: displayedSettings.rq_membership_protocol ?? "rq_primary_chain_v1",
      edge_projection_protocol: displayedSettings.edge_projection_protocol ?? "membership_q15_layer_type_calibrated_v3",
      edge_type_calibration_protocol: displayedSettings.edge_type_calibration_protocol ?? "type_local_winsorized_minmax_v1",
      rq_membership_temperature: String(displayedSettings.rq_membership_temperature ?? 0.35),
      dense_knn_k_min: String(displayedSettings.dense_knn_k_min ?? 5),
      dense_knn_k_max: String(displayedSettings.dense_knn_k_max ?? 24),
      dense_reverse_b_min_base: String(displayedSettings.dense_reverse_b_min_base ?? 2),
      dense_reverse_b_max_base: String(displayedSettings.dense_reverse_b_max_base ?? 8),
      dense_reverse_b_min_doc: String(displayedSettings.dense_reverse_b_min_doc ?? 1),
      dense_reverse_b_max_doc: String(displayedSettings.dense_reverse_b_max_doc ?? 6),
      dense_reverse_b_min_lang: String(displayedSettings.dense_reverse_b_min_lang ?? 1),
      dense_reverse_b_max_lang: String(displayedSettings.dense_reverse_b_max_lang ?? 4),
      dense_min_cosine: String(displayedSettings.dense_min_cosine ?? 0.58),
      dense_strong_cosine: String(displayedSettings.dense_strong_cosine ?? 0.72),
      cross_doc_out_quota_min: String(displayedSettings.cross_doc_out_quota_min ?? 1),
      cross_doc_out_quota_max: String(displayedSettings.cross_doc_out_quota_max ?? 4),
      cross_doc_min_cosine: String(displayedSettings.cross_doc_min_cosine ?? 0.62),
      cross_language_out_quota_min: String(displayedSettings.cross_language_out_quota_min ?? 0),
      cross_language_out_quota_max: String(displayedSettings.cross_language_out_quota_max ?? 3),
      cross_language_min_cosine: String(displayedSettings.cross_language_min_cosine ?? 0.65),
      context_package_token_budget: String(displayedSettings.context_package_token_budget ?? 2400),
      retrieval_result_top_k_default: String(displayedSettings.retrieval_result_top_k_default ?? 8),
      retrieval_v1_dense_candidate_budget: String(displayedSettings.retrieval_v1_dense_candidate_budget ?? 64),
      retrieval_v1_rq_candidate_budget: String(displayedSettings.retrieval_v1_rq_candidate_budget ?? 64),
      retrieval_v1_bm25_candidate_budget: String(displayedSettings.retrieval_v1_bm25_candidate_budget ?? 64),
      retrieval_v1_root_entry_budget: String(displayedSettings.retrieval_v1_root_entry_budget ?? 8),
      retrieval_v1_per_parent_entry_budget: String(displayedSettings.retrieval_v1_per_parent_entry_budget ?? 8),
      retrieval_v1_layer_entry_budget: String(displayedSettings.retrieval_v1_layer_entry_budget ?? 80),
      retrieval_v1_max_depth: String(displayedSettings.retrieval_v1_max_depth ?? 3),
      retrieval_v1_restore_per_hit: String(displayedSettings.retrieval_v1_restore_per_hit ?? 4),
      lexical_index_max_documents: String(displayedSettings.lexical_index_max_documents ?? 100000),
      lexical_index_max_postings: String(displayedSettings.lexical_index_max_postings ?? 4000000),
      lexical_index_max_characters: String(displayedSettings.lexical_index_max_characters ?? 32000000),
      bm25_k1: String(displayedSettings.bm25_k1 ?? 1.2),
      bm25_b: String(displayedSettings.bm25_b ?? 0.75),
    });
    setApiKeyEditing(false);
    setGraphApiKeyEditing(false);
    setEmbeddingApiKeyEditing(false);
  }, [settingsQuery.data]);

  const saveMutation = useMutation({
    mutationFn: (payload: ModelSettingsUpdate) => updateModelSettings(payload),
    onSuccess: async (result) => {
      setApiKeyEditing(false);
      setGraphApiKeyEditing(false);
      setEmbeddingApiKeyEditing(false);
      const pendingRebuild = result.pending_rebuild_changes ?? [];
      const pendingService = result.pending_service_recreate_changes ?? [];
      setSavedMessage(
        result.apply_error_type
          ? `配置已写入；运行时刷新待重试：${result.apply_error_type}`
          : pendingService.length
            ? `配置已写入；${pendingService.length} 项重启后生效，${pendingRebuild.length} 项待图谱/索引重建`
            : pendingRebuild.length
              ? `配置已写入；${pendingRebuild.length} 项待图谱/索引重建`
              : "配置已写入并生效",
      );
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
  const showApiKeyMask = Boolean(settings?.has_chat_api_key && !apiKeyEditing && !form?.clear_chat_api_key);
  const showGraphApiKeyMask = Boolean(settings?.has_graph_api_key && !graphApiKeyEditing && !form?.clear_graph_api_key);
  const showEmbeddingApiKeyMask = Boolean(settings?.has_embedding_api_key && !embeddingApiKeyEditing && !form?.clear_embedding_api_key);
  const envSynced = Boolean(runtimeQuery.data?.env_sync?.synced);
  const settingsFileSynced = Boolean(settings?.settings_file_synced);
  const pendingRebuildCount = settings?.pending_rebuild_changes?.length ?? 0;
  const pendingServiceCount = settings?.pending_service_recreate_changes?.length ?? 0;
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

  if (settingsQuery.isLoading || !form || !settings) {
    return <LoadingBlock rows={4} />;
  }
  if (settingsQuery.error) {
    return <ErrorBlock message={(settingsQuery.error as Error).message} />;
  }

  const updateForm = <K extends keyof SettingsForm>(key: K, value: SettingsForm[K]) => {
    setForm((current) => (current ? { ...current, [key]: value } : current));
  };

  const buildPayload = (): ModelSettingsUpdate => buildRuntimeSettingsPayload(form);

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
        <div className={activeTab === "model" ? "grid min-w-0 gap-7 xl:grid-cols-[minmax(0,1fr)_280px]" : "hidden"}>
          <aside className="min-w-0 space-y-6 xl:order-2">
            <RuntimeSettingsNavigation
              activePage={activeSettingsPage}
              onChange={setActiveSettingsPage}
            />
            <div className="hidden xl:block">
              <p className="section-kicker">生产参数配置</p>
              <h2 className="glow-text mt-2 text-4xl font-semibold text-white">运行时设置</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-cyan-50/62">
                这里只保留当前 active path 实际消费的参数，并按下一次调用、重建后、重启服务后三类边界标注。
              </p>
              <p className="mt-3 max-w-xl text-xs leading-6 text-white/45">
                {RUNTIME_ENV_AUTHORITY_NOTE}
              </p>
            </div>

            <div className="hidden flex-wrap gap-2 xl:flex">
              <StatusPill ok={Boolean(settings?.has_chat_api_key)}>聊天密钥 {settings?.has_chat_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(settings?.has_graph_api_key)}>图谱密钥 {settings?.has_graph_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(settings?.has_embedding_api_key)}>向量密钥 {settings?.has_embedding_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={bridgeHealthy}>模型桥 {bridgeStatusText}</StatusPill>
              <StatusPill ok={settingsFileSynced && envSynced}>{settingsFileSynced && envSynced ? "根配置已同步" : "根配置需检查"}</StatusPill>
              <StatusPill ok={pendingRebuildCount === 0}>待重建 {pendingRebuildCount}</StatusPill>
              <StatusPill ok={pendingServiceCount === 0}>待重启 {pendingServiceCount}</StatusPill>
              <StatusPill ok={!settings?.enable_model_fallback && !settings?.enable_database_fallback}>回退已禁用</StatusPill>
              <StatusPill ok={!settings?.concept_i18n_enabled}>双语派生 {settings?.concept_i18n_enabled ? "已开启" : "已关闭"}</StatusPill>
              <StatusPill ok={Boolean(settings?.lifecycle?.hot_reloadable?.length)}>热加载 {settings?.lifecycle?.hot_reloadable?.length ?? 0}</StatusPill>
              <StatusPill ok={Boolean(settings?.lifecycle?.rebuild_required?.length)}>需重建 {settings?.lifecycle?.rebuild_required?.length ?? 0}</StatusPill>
              <StatusPill ok={Boolean(settings?.runtime_settings_version)}>运行时 {settings?.runtime_settings_version ? "已同步" : "等待中"}</StatusPill>
            </div>

            <div className={`${sectionClass} hidden xl:block`}>
              <p className="text-sm font-semibold text-white">生产保护</p>
              <div className="mt-3 grid gap-2 text-sm leading-6 text-white/58">
                <p>模型回退：{settings?.enable_model_fallback ? "已开启，生产不推荐" : "已关闭"}</p>
                <p>数据库回退：{settings?.enable_database_fallback ? "已开启，生产不推荐" : "已关闭"}</p>
                <p>增量图谱：由前端触发图谱任务时选择“最小更新/全量重建”，当前不是全局 env 开关。</p>
              </div>
            </div>
          </aside>

          <form
            className="grid min-w-0 gap-5 xl:order-1"
            onSubmit={(event) => {
              event.preventDefault();
              void handleSubmit();
            }}
          >
            {activeSettingsPage === "connections" ? (
            <section className={sectionClass}>
              <div className="mb-5 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-white">模型连接与密钥</p>
                  <p className="mt-1 text-sm text-white/52">对话模型、图谱模型、向量模型、连接地址和接口密钥集中在这里维护。</p>
                </div>
              </div>
              <BoundaryNote title="生效边界：下一次请求、模型调用或构图任务">
                对话地址、对话 DNS、对话模型和对话密钥影响下一次问答、检索规划和引用验证；图谱地址、图谱 DNS、图谱模型和图谱密钥影响下一次构图模型调用；已经在执行的模型调用不会中途切换。
              </BoundaryNote>
              <BoundaryNote title="图谱模型边界：已有 active 图谱不会自动改写">
                修改图谱模型 endpoint 只改变后续构图的概念命名、粗概念和双语派生来源；已有 mid/coarse graph 需要显式重建才会更新。
              </BoundaryNote>
              <BoundaryNote title="向量模型边界：已有 active 向量不会自动改写">
                修改向量模型后只影响后续解析、重嵌入或全量重建任务；已有资料库向量需要显式重新解析或重建。
              </BoundaryNote>
              {form.model_bridge_enabled ? (
                <div className="mt-4 rounded-2xl border border-cyan-200/20 bg-cyan-300/[0.035] px-4 py-3 text-sm text-cyan-50/70">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-white">模型桥转发状态</span>
                    <StatusPill ok={bridgeHealthy}>{bridgeStatusText}</StatusPill>
                  </div>
                  <div className="mt-3 grid gap-2 md:grid-cols-2">
                    <p className="min-w-0 break-words">聊天 effective 地址：{settings?.effective_chat_base_url || "未读取"}</p>
                    <p className="min-w-0 break-words">图谱 effective 地址：{settings?.effective_graph_base_url || "未读取"}</p>
                    <p className="min-w-0 break-words">向量 effective 地址：{settings?.effective_embedding_base_url || "未读取"}</p>
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
                  badge="保存后待重建服务"
                />
                <ModelProtocolSelect
                  label="聊天接口协议"
                  value={form.chat_api_protocol}
                  onChange={(value) => updateForm("chat_api_protocol", value)}
                  disabled={saveMutation.isPending}
                  lifecycle="hot_reloadable · applies to the next model call"
                />
                <ModelProtocolSelect
                  label="图谱接口协议"
                  value={form.graph_api_protocol}
                  onChange={(value) => updateForm("graph_api_protocol", value)}
                  disabled={saveMutation.isPending}
                  lifecycle="rebuild_required · stage as candidate, then shadow rebuild, evaluate and promote"
                />
                <EmbeddingProtocolSelect
                  value={form.embedding_api_protocol}
                  onChange={(value) => updateForm("embedding_api_protocol", value)}
                  disabled={saveMutation.isPending}
                />
                <SettingField label="聊天基础地址" value={form.chat_base_url} onChange={(value) => updateForm("chat_base_url", value)} className="md:col-span-2" />
                <SettingField label="图谱基础地址" value={form.graph_base_url} onChange={(value) => updateForm("graph_base_url", value)} className="md:col-span-2" />
                <SettingField label="向量基础地址" value={form.embedding_base_url} onChange={(value) => updateForm("embedding_base_url", value)} className="md:col-span-2" />
                <SettingField label="聊天 DNS 覆盖 IP" value={form.chat_resolve_ip} onChange={(value) => updateForm("chat_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="图谱 DNS 覆盖 IP" value={form.graph_resolve_ip} onChange={(value) => updateForm("graph_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="向量 DNS 覆盖 IP" value={form.embedding_resolve_ip} onChange={(value) => updateForm("embedding_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="聊天模型" value={form.chat_model} onChange={(value) => updateForm("chat_model", value)} />
                <SettingField label="图谱模型" value={form.graph_model} onChange={(value) => updateForm("graph_model", value)} />
                <SettingField label="向量模型" value={form.embedding_model} onChange={(value) => updateForm("embedding_model", value)} />
              </div>
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <label className="flex min-w-0 max-w-full flex-col gap-2">
                  <ParameterName label="聊天接口密钥" />
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3">
                    <KeyRound className="size-4 text-cyan-100/58" />
                    <input
                      type="password"
                      value={showApiKeyMask ? "••••••••••••••••" : form.chat_api_key}
                      readOnly={showApiKeyMask}
                      disabled={form.clear_chat_api_key}
                      onChange={(event) => updateForm("chat_api_key", event.target.value)}
                      placeholder={settings?.has_chat_api_key ? "留空则保留当前对话密钥" : "输入对话接口密钥"}
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

                <label className="flex min-w-0 max-w-full flex-col gap-2">
                  <ParameterName label="图谱接口密钥" />
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3">
                    <KeyRound className="size-4 text-cyan-100/58" />
                    <input
                      type="password"
                      value={showGraphApiKeyMask ? "••••••••••••••••" : form.graph_api_key}
                      readOnly={showGraphApiKeyMask}
                      disabled={form.clear_graph_api_key}
                      onChange={(event) => updateForm("graph_api_key", event.target.value)}
                      placeholder={settings?.has_graph_api_key ? "留空则保留当前图谱密钥" : "输入图谱接口密钥"}
                      className="h-11 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30"
                      autoComplete="off"
                    />
                    {showGraphApiKeyMask ? (
                      <button type="button" onClick={() => setGraphApiKeyEditing(true)} className="inline-flex items-center gap-1 rounded-full border border-white/8 px-2.5 py-1 text-xs text-white/55 transition hover:border-cyan-200/24 hover:text-cyan-100">
                        <PencilLine className="size-3.5" />
                        修改
                      </button>
                    ) : null}
                    <EyeOff className="size-4 text-white/32" />
                  </div>
                </label>

                <label className="flex min-w-0 max-w-full flex-col gap-2">
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
                <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.025] px-4 py-3 text-sm text-white/70">
                  <input
                    type="checkbox"
                    checked={form.clear_chat_api_key}
                    onChange={(event) => {
                      updateForm("clear_chat_api_key", event.target.checked);
                      if (event.target.checked) {
                        setApiKeyEditing(false);
                        updateForm("chat_api_key", "");
                      }
                    }}
                    className="size-4 accent-rose-300"
                  />
                  <ParameterName label="清除当前聊天接口密钥" className="text-sm font-normal normal-case tracking-normal text-white/70" />
                </label>
                <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.025] px-4 py-3 text-sm text-white/70">
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
                <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.025] px-4 py-3 text-sm text-white/70">
                  <input
                    type="checkbox"
                    checked={form.clear_graph_api_key}
                    onChange={(event) => {
                      updateForm("clear_graph_api_key", event.target.checked);
                      if (event.target.checked) {
                        setGraphApiKeyEditing(false);
                        updateForm("graph_api_key", "");
                      }
                    }}
                    className="size-4 accent-rose-300"
                  />
                  <ParameterName label="清除当前图谱接口密钥" className="text-sm font-normal normal-case tracking-normal text-white/70" />
                </label>
              </div>
            </section>
            ) : null}

            {activeSettingsPage === "runtime" ? (
            <>
            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">模型调用参数</p>
              <BoundaryNote title="生效边界：下一次请求或下一次模型调用">
                模型请求并发、源文件 I/O 并发、超时和 embedding 批大小会热加载；已开始的请求或批次按启动时快照继续执行。
              </BoundaryNote>
              <div className="mt-5 grid gap-4 md:grid-cols-5">
                <SettingField label="模型请求并发" type="number" min={1} max={16} value={form.model_request_concurrency} onChange={(value) => updateForm("model_request_concurrency", value)} />
                <SourceIoConcurrencyField value={form.source_io_concurrency} onChange={(value) => updateForm("source_io_concurrency", value)} />
                <SettingField label="Chat JSON token 上限" type="number" min={256} max={32768} value={form.chat_json_max_tokens} onChange={(value) => updateForm("chat_json_max_tokens", value)} />
                <SettingField label="模型超时秒数" type="number" min={5} max={600} value={form.model_request_timeout_seconds} onChange={(value) => updateForm("model_request_timeout_seconds", value)} />
                <AgentTimeBudgetFields
                  values={{
                    retrieval_total_timeout_seconds: form.retrieval_total_timeout_seconds,
                    retrieval_generation_timeout_seconds: form.retrieval_generation_timeout_seconds,
                  }}
                  onChange={(key, value) => updateForm(key, value)}
                />
                <SettingField label="Embedding 批大小" type="number" min={1} max={10} value={form.embedding_batch_size} onChange={(value) => updateForm("embedding_batch_size", value)} />
                <SettingField label="导入内存软水位" type="number" min={0.01} max={0.98} step={0.01} value={form.ingestion_memory_soft_limit_ratio} onChange={(value) => updateForm("ingestion_memory_soft_limit_ratio", value)} />
                <SettingField label="导入内存硬水位" type="number" min={0.02} max={0.99} step={0.01} value={form.ingestion_memory_hard_limit_ratio} onChange={(value) => updateForm("ingestion_memory_hard_limit_ratio", value)} />
                <SettingField label="导入内存临界水位" type="number" min={0.03} max={1} step={0.01} value={form.ingestion_memory_critical_limit_ratio} onChange={(value) => updateForm("ingestion_memory_critical_limit_ratio", value)} />
              </div>
            </section>

            <AgentAdmissionSettingsSection
              values={{
                agent_request_concurrency: form.agent_request_concurrency,
                agent_request_queue_limit: form.agent_request_queue_limit,
                agent_request_queue_timeout_seconds: form.agent_request_queue_timeout_seconds,
                agent_request_lease_ttl_seconds: form.agent_request_lease_ttl_seconds,
              }}
              onChange={(key, value) => updateForm(key, value)}
            />

            <UploadSecuritySettingsSection value={form.upload_max_bytes} onChange={(value) => updateForm("upload_max_bytes", value)} />
            </>
            ) : null}

            {activeSettingsPage === "retrieval" ? (
            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">意图执行检索</p>
              <BoundaryNote title="生效边界：下一次同步或流式检索问答">
                模型逐请求选择 coarse、mid 或 chunk 入口及合法通道权重；以下数值只限定执行器预算，不创建产品模式。
              </BoundaryNote>
              <div className="mt-5">
                <SwitchRow
                  title="问答双语词面"
                  description="开启后，同一次意图规划会为可翻译概念生成中英文检索面；标识符和编号保持原样，空词面仍可选择 Dense-only。"
                  checked={form.query_facet_bilingual_enabled}
                  onChange={() => updateForm("query_facet_bilingual_enabled", !form.query_facet_bilingual_enabled)}
                  disabled={saveMutation.isPending}
                  badge="热加载"
                />
              </div>
              <div className="mt-5 grid gap-4 md:grid-cols-4">
                <SettingField label="证据包 token 预算" type="number" min={256} max={20000} value={form.context_package_token_budget} onChange={(value) => updateForm("context_package_token_budget", value)} />
                <AgentTokenBudgetFields
                  values={{
                    retrieval_planning_max_tokens: form.retrieval_planning_max_tokens,
                    retrieval_generation_max_tokens: form.retrieval_generation_max_tokens,
                  }}
                  onChange={(key, value) => updateForm(key, value)}
                />
                <SettingField label="结果 Top K 默认值" type="number" min={1} max={50} value={form.retrieval_result_top_k_default} onChange={(value) => updateForm("retrieval_result_top_k_default", value)} />
                <SettingField label="Dense 候选预算" type="number" min={1} max={4096} value={form.retrieval_v1_dense_candidate_budget} onChange={(value) => updateForm("retrieval_v1_dense_candidate_budget", value)} />
                <SettingField label="RQ 候选预算" type="number" min={1} max={4096} value={form.retrieval_v1_rq_candidate_budget} onChange={(value) => updateForm("retrieval_v1_rq_candidate_budget", value)} />
                <SettingField label="BM25 候选预算" type="number" min={1} max={4096} value={form.retrieval_v1_bm25_candidate_budget} onChange={(value) => updateForm("retrieval_v1_bm25_candidate_budget", value)} />
                <SettingField label="根入口预算" type="number" min={1} max={256} value={form.retrieval_v1_root_entry_budget} onChange={(value) => updateForm("retrieval_v1_root_entry_budget", value)} />
                <SettingField label="逐父节点预算" type="number" min={1} max={256} value={form.retrieval_v1_per_parent_entry_budget} onChange={(value) => updateForm("retrieval_v1_per_parent_entry_budget", value)} />
                <SettingField label="单层总预算" type="number" min={1} max={1024} value={form.retrieval_v1_layer_entry_budget} onChange={(value) => updateForm("retrieval_v1_layer_entry_budget", value)} />
                <SettingField label="最大遍历深度" type="number" min={0} max={64} value={form.retrieval_v1_max_depth} onChange={(value) => updateForm("retrieval_v1_max_depth", value)} />
                <SettingField label="每命中恢复预算" type="number" min={0} max={64} value={form.retrieval_v1_restore_per_hit} onChange={(value) => updateForm("retrieval_v1_restore_per_hit", value)} />
              </div>
              <div className="mt-6 rounded-2xl border border-white/8 bg-black/10 p-5">
                <p className="text-xs uppercase tracking-[0.18em] text-cyan-100/48">原文 BM25 生命周期</p>
                <p className="mt-2 text-xs leading-5 text-white/46">容量是构建硬上限；k1/b 变更必须通过 candidate、重建、发布和缓存失效后生效。</p>
                <div className="mt-4 grid gap-4 md:grid-cols-5">
                  <SettingField label="BM25 文档上限" type="number" min={1} max={1000000} value={form.lexical_index_max_documents} onChange={(value) => updateForm("lexical_index_max_documents", value)} />
                  <SettingField label="BM25 postings 上限" type="number" min={1} max={20000000} value={form.lexical_index_max_postings} onChange={(value) => updateForm("lexical_index_max_postings", value)} />
                  <SettingField label="BM25 原文字符上限" type="number" min={1} max={1000000000} value={form.lexical_index_max_characters} onChange={(value) => updateForm("lexical_index_max_characters", value)} />
                  <SettingField label="BM25 k1" type="number" min={0.01} max={10} step={0.01} value={form.bm25_k1} onChange={(value) => updateForm("bm25_k1", value)} />
                  <SettingField label="BM25 b" type="number" min={0} max={1} step={0.01} value={form.bm25_b} onChange={(value) => updateForm("bm25_b", value)} />
                </div>
              </div>
            </section>
            ) : null}

            {activeSettingsPage === "graph" ? (
            <GraphProtocolSettingsSection
              values={{
                edge_distance_protocol: form.edge_distance_protocol,
                rq_membership_protocol: form.rq_membership_protocol,
                edge_projection_protocol: form.edge_projection_protocol,
                edge_type_calibration_protocol: form.edge_type_calibration_protocol,
                rq_membership_temperature: form.rq_membership_temperature,
              }}
              onChange={(key, value) => updateForm(key, value)}
            />
            ) : null}

            {activeSettingsPage === "build" ? (
            <>
            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">重建参数</p>
              <BoundaryNote title="生效边界：新任务会读取，但已有 active 数据不会改变">
                固定切块、向量维度、RQ-KMeans 分支与残差参数和概念批处理参数必须通过重解析或图谱重建，才能影响已有资料库的 chunk、向量、关系图和概念图；RQ 地址深度固定为 3，L3 到中粒度、L2 到粗粒度始终全量投影。
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
                <RqProtocolDepthField />
                <SettingField label="RQ-KMeans 最大 K（精确 pair 域上限 6）" type="number" min={1} max={6} value={form.rq_kmeans_max_k} onChange={(value) => updateForm("rq_kmeans_max_k", value)} />
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

            <RuntimeSettingsCandidatePanel
              knowledgeBaseId={selectedKnowledgeBaseId}
              knowledgeBaseName={selectedKnowledgeBase?.name}
              settings={settings}
              form={form}
              onError={(error) => setErrorDialog(errorDialogFromUnknown(error))}
            />
            </>
            ) : null}

            {activeSettingsPage === "deployment" ? (
            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">服务重启参数</p>
              <BoundaryNote title="生效边界：必须重启或重建 worker 服务">
                Docker Compose 里 Celery worker 通过 --concurrency 启动；运行中保存工作进程并发不会改变现有 worker 池。
              </BoundaryNote>
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <SettingField label="工作进程并发" type="number" min={1} max={32} value={form.worker_concurrency} onChange={(value) => updateForm("worker_concurrency", value)} />
              </div>
            </section>
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/8 bg-white/[0.025] p-4">
              <p className="text-xs leading-6 text-white/42">
                保存会按键归属写入根 .env 或 settings.json；热加载参数马上刷新，重建参数和服务参数分别保持待重建、待重启状态。回退开关不在页面开放开启。
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
