"use client";

import { useEffect, useState } from "react";
import type { ModelSettingsUpdate, RuntimeCheckResponse, RuntimeIssue, StructuredApiErrorBody } from "@course-kg/shared";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  EyeOff,
  KeyRound,
  Loader2,
  PencilLine,
  RotateCcw,
  Save,
  ShieldAlert,
  SlidersHorizontal,
  XCircle,
} from "lucide-react";

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
import { fetchModelSettings, fetchRuntimeCheck, updateModelSettings } from "@/lib/api";

type SettingsForm = {
  chat_base_url: string;
  embedding_base_url: string;
  chat_resolve_ip: string;
  embedding_resolve_ip: string;
  embedding_model: string;
  chat_model: string;
  embedding_dimensions: string;
  graph_extraction_strategy: string;
  graph_extraction_soft_start_budget: string;
  graph_extraction_max_model_calls_per_run: string;
  graph_extraction_min_marginal_gain: string;
  graph_extraction_stall_rounds: string;
  graph_extraction_concurrency: string;
  graph_extraction_resume_batch_size: string;
  worker_concurrency: string;
  ingestion_file_concurrency: string;
  model_request_concurrency: string;
  model_request_timeout_seconds: string;
  hpo_concurrency: string;
  api_key: string;
  clear_api_key: boolean;
  embedding_api_key: string;
  clear_embedding_api_key: boolean;
  model_bridge_enabled: boolean;
  reranker_enabled: boolean;
  reranker_model: string;
  reranker_max_length: string;
  semantic_chunking_enabled: boolean;
  semantic_chunking_min_length: string;
  retrieval_layer_enabled: boolean;
  retrieval_cache_ttl_seconds: string;
  enable_agentic_reflection: boolean;
  enable_post_generation_reflection: boolean;
  citation_verification_sample_max: string;
  reflection_max_retries: string;
  enable_auto_hpo: boolean;
  enable_graph_community_summaries: boolean;
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
  description: string;
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  badge?: string;
};

const inputClass = "h-11 rounded-xl border-white/10 bg-white/[0.04] px-3 text-white placeholder:text-white/28";
const sectionClass = "rounded-2xl border border-white/10 bg-white/[0.035] p-5";

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

function runtimeIssueDialog(check: RuntimeCheckResponse): ErrorDialogState | null {
  if (!check.blocking_issues.length) {
    return null;
  }
  return {
    title: "基础设施检测未通过",
    message: "当前运行环境不满足本次操作的前置条件，请按提示修复后重试。",
    issues: check.blocking_issues,
    fixCommands: Array.from(new Set(check.blocking_issues.flatMap((issue) => issue.fix_commands))),
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

function SettingField({
  label,
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
      <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/46">{label}</span>
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

function SwitchRow({ title, description, checked, onChange, disabled, badge }: SwitchRowProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/[0.035] p-4">
      <div className="min-w-[240px] flex-1">
        <p className="flex flex-wrap items-center gap-2 text-sm font-semibold text-white">
          <SlidersHorizontal className="size-4 text-cyan-100/70" />
          {title}
          {badge ? <span className="rounded-full border border-white/10 px-2 py-0.5 text-xs text-white/52">{badge}</span> : null}
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

export function SettingsWorkspace() {
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({ queryKey: ["model-settings"], queryFn: fetchModelSettings });
  const runtimeQuery = useQuery({ queryKey: ["runtime-check"], queryFn: () => fetchRuntimeCheck(), retry: false });
  const [form, setForm] = useState<SettingsForm | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [apiKeyEditing, setApiKeyEditing] = useState(false);
  const [embeddingApiKeyEditing, setEmbeddingApiKeyEditing] = useState(false);
  const [errorDialog, setErrorDialog] = useState<ErrorDialogState | null>(null);

  useEffect(() => {
    if (!settingsQuery.data) {
      return;
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setForm({
      chat_base_url: settingsQuery.data.chat_base_url,
      embedding_base_url: settingsQuery.data.embedding_base_url ?? "",
      chat_resolve_ip: settingsQuery.data.chat_resolve_ip ?? "",
      embedding_resolve_ip: settingsQuery.data.embedding_resolve_ip ?? "",
      embedding_model: settingsQuery.data.embedding_model,
      chat_model: settingsQuery.data.chat_model,
      embedding_dimensions: String(settingsQuery.data.embedding_dimensions),
      graph_extraction_strategy: settingsQuery.data.graph_extraction_strategy ?? "adaptive_best_first",
      graph_extraction_soft_start_budget: String(settingsQuery.data.graph_extraction_soft_start_budget ?? 24),
      graph_extraction_max_model_calls_per_run: String(settingsQuery.data.graph_extraction_max_model_calls_per_run ?? 24),
      graph_extraction_min_marginal_gain: String(settingsQuery.data.graph_extraction_min_marginal_gain ?? 0.03),
      graph_extraction_stall_rounds: String(settingsQuery.data.graph_extraction_stall_rounds ?? 2),
      graph_extraction_concurrency: String(settingsQuery.data.graph_extraction_concurrency ?? 2),
      graph_extraction_resume_batch_size: String(settingsQuery.data.graph_extraction_resume_batch_size ?? 6),
      worker_concurrency: String(settingsQuery.data.worker_concurrency ?? 3),
      ingestion_file_concurrency: String(settingsQuery.data.ingestion_file_concurrency ?? 3),
      model_request_concurrency: String(settingsQuery.data.model_request_concurrency ?? 3),
      model_request_timeout_seconds: String(settingsQuery.data.model_request_timeout_seconds ?? 240),
      hpo_concurrency: String(settingsQuery.data.hpo_concurrency ?? 1),
      api_key: "",
      clear_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: settingsQuery.data.model_bridge_enabled ?? true,
      reranker_enabled: settingsQuery.data.reranker_enabled ?? false,
      reranker_model: settingsQuery.data.reranker_model ?? "cross-encoder/ms-marco-MiniLM-L-6-v2",
      reranker_max_length: String(settingsQuery.data.reranker_max_length ?? 512),
      semantic_chunking_enabled: settingsQuery.data.semantic_chunking_enabled ?? false,
      semantic_chunking_min_length: String(settingsQuery.data.semantic_chunking_min_length ?? 2000),
      retrieval_layer_enabled: settingsQuery.data.retrieval_layer_enabled ?? true,
      retrieval_cache_ttl_seconds: String(settingsQuery.data.retrieval_cache_ttl_seconds ?? 120),
      enable_agentic_reflection: settingsQuery.data.enable_agentic_reflection ?? true,
      enable_post_generation_reflection: settingsQuery.data.enable_post_generation_reflection ?? false,
      citation_verification_sample_max: String(settingsQuery.data.citation_verification_sample_max ?? 3),
      reflection_max_retries: String(settingsQuery.data.reflection_max_retries ?? 2),
      enable_auto_hpo: settingsQuery.data.enable_auto_hpo ?? true,
      enable_graph_community_summaries: settingsQuery.data.enable_graph_community_summaries ?? true,
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
        queryClient.invalidateQueries({ queryKey: ["courses"] }),
      ]);
    },
    onError: (error) => setErrorDialog(errorDialogFromUnknown(error)),
  });

  const settings = settingsQuery.data;
  const showApiKeyMask = Boolean(settings?.has_api_key && !apiKeyEditing && !form?.clear_api_key);
  const showEmbeddingApiKeyMask = Boolean(settings?.has_embedding_api_key && !embeddingApiKeyEditing && !form?.clear_embedding_api_key);

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
    graph_extraction_strategy: form.graph_extraction_strategy.trim() || "adaptive_best_first",
    graph_extraction_soft_start_budget: parseIntField(form.graph_extraction_soft_start_budget),
    graph_extraction_max_model_calls_per_run: parseIntField(form.graph_extraction_max_model_calls_per_run),
    graph_extraction_min_marginal_gain: parseFloatField(form.graph_extraction_min_marginal_gain),
    graph_extraction_stall_rounds: parseIntField(form.graph_extraction_stall_rounds),
    graph_extraction_concurrency: parseIntField(form.graph_extraction_concurrency),
    graph_extraction_resume_batch_size: parseIntField(form.graph_extraction_resume_batch_size),
    worker_concurrency: parseIntField(form.worker_concurrency),
    ingestion_file_concurrency: parseIntField(form.ingestion_file_concurrency),
    model_request_concurrency: parseIntField(form.model_request_concurrency),
    model_request_timeout_seconds: parseIntField(form.model_request_timeout_seconds),
    hpo_concurrency: parseIntField(form.hpo_concurrency),
    api_key: form.api_key.trim() || null,
    clear_api_key: form.clear_api_key,
    embedding_api_key: form.embedding_api_key.trim() || null,
    clear_embedding_api_key: form.clear_embedding_api_key,
    model_bridge_enabled: form.model_bridge_enabled,
    reranker_enabled: form.reranker_enabled,
    reranker_model: form.reranker_model.trim(),
    reranker_max_length: parseIntField(form.reranker_max_length),
    semantic_chunking_enabled: form.semantic_chunking_enabled,
    semantic_chunking_min_length: parseIntField(form.semantic_chunking_min_length),
    retrieval_layer_enabled: form.retrieval_layer_enabled,
    retrieval_cache_ttl_seconds: parseIntField(form.retrieval_cache_ttl_seconds),
    enable_agentic_reflection: form.enable_agentic_reflection,
    enable_post_generation_reflection: form.enable_post_generation_reflection,
    citation_verification_sample_max: parseIntField(form.citation_verification_sample_max),
    reflection_max_retries: parseIntField(form.reflection_max_retries),
    enable_auto_hpo: form.enable_auto_hpo,
    enable_graph_community_summaries: form.enable_graph_community_summaries,
  });

  const handleSubmit = async () => {
    try {
      const check = await fetchRuntimeCheck();
      const dialog = runtimeIssueDialog(check);
      if (dialog) {
        setErrorDialog(dialog);
        await queryClient.invalidateQueries({ queryKey: ["runtime-check"] });
        return;
      }
      saveMutation.mutate(buildPayload());
    } catch (error) {
      setErrorDialog(errorDialogFromUnknown(error));
    }
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
        <div className="grid gap-7 xl:grid-cols-[minmax(320px,0.72fr)_minmax(560px,1.28fr)]">
          <aside className="space-y-6">
            <div>
              <p className="section-kicker">生产参数配置</p>
              <h2 className="glow-text mt-2 text-4xl font-semibold text-white">运行时设置</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-cyan-50/62">
                这里配置会写入根目录 .env，并通过后端热加载影响新任务。模型名、并发、图谱预算、检索增强和 HPO 都按当前生产参数表组织。
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <StatusPill ok={Boolean(settings?.has_api_key)}>Chat Key {settings?.has_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(settings?.has_embedding_api_key)}>Embedding Key {settings?.has_embedding_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(runtimeQuery.data?.env_sync.synced)}>{runtimeQuery.data?.env_sync.synced ? ".env 已同步" : ".env 需检查"}</StatusPill>
              <StatusPill ok={!settings?.enable_model_fallback && !settings?.enable_database_fallback}>Fallback 已禁用</StatusPill>
            </div>

            <div className={sectionClass}>
              <p className="text-sm font-semibold text-white">生产保护</p>
              <div className="mt-3 grid gap-2 text-sm leading-6 text-white/58">
                <p>模型 fallback：{settings?.enable_model_fallback ? "已开启，生产不推荐" : "已关闭"}</p>
                <p>数据库 fallback：{settings?.enable_database_fallback ? "已开启，生产不推荐" : "已关闭"}</p>
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
                  <p className="text-sm font-semibold text-white">模型接入</p>
                  <p className="mt-1 text-sm text-white/52">模型桥默认开启；模型名和 endpoint 保持可热加载。</p>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <SwitchRow
                  title="模型桥"
                  description="开启后容器通过 host.docker.internal 访问宿主机模型桥。"
                  checked={form.model_bridge_enabled}
                  onChange={() => updateForm("model_bridge_enabled", !form.model_bridge_enabled)}
                  disabled={saveMutation.isPending}
                  badge="MODEL_BRIDGE_ENABLED"
                />
                <SettingField label="Chat Base URL" value={form.chat_base_url} onChange={(value) => updateForm("chat_base_url", value)} className="md:col-span-2" />
                <SettingField label="Embedding Base URL" value={form.embedding_base_url} onChange={(value) => updateForm("embedding_base_url", value)} className="md:col-span-2" />
                <SettingField label="Chat DNS Override IP" value={form.chat_resolve_ip} onChange={(value) => updateForm("chat_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="Embedding DNS Override IP" value={form.embedding_resolve_ip} onChange={(value) => updateForm("embedding_resolve_ip", value)} placeholder="可选，留空使用系统 DNS" />
                <SettingField label="Chat / 图谱模型" value={form.chat_model} onChange={(value) => updateForm("chat_model", value)} />
                <SettingField label="Embedding 模型" value={form.embedding_model} onChange={(value) => updateForm("embedding_model", value)} />
                <SettingField label="Embedding 维度" type="number" min={1} max={8192} value={form.embedding_dimensions} onChange={(value) => updateForm("embedding_dimensions", value)} />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">图谱抽取预算</p>
              <p className="mt-1 text-sm text-white/52">生产推荐值是 24 次模型调用、2 路图谱抽取并发、每次 resume 6 个 chunk。</p>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <SettingField label="抽取策略" value={form.graph_extraction_strategy} onChange={(value) => updateForm("graph_extraction_strategy", value)} />
                <SettingField label="Soft Start Budget" type="number" min={1} value={form.graph_extraction_soft_start_budget} onChange={(value) => updateForm("graph_extraction_soft_start_budget", value)} />
                <SettingField label="Max Model Calls / Run" type="number" min={1} value={form.graph_extraction_max_model_calls_per_run} onChange={(value) => updateForm("graph_extraction_max_model_calls_per_run", value)} />
                <SettingField label="Graph 并发" type="number" min={1} max={8} value={form.graph_extraction_concurrency} onChange={(value) => updateForm("graph_extraction_concurrency", value)} />
                <SettingField label="Resume Batch Size" type="number" min={1} max={100} value={form.graph_extraction_resume_batch_size} onChange={(value) => updateForm("graph_extraction_resume_batch_size", value)} />
                <SettingField label="Min Marginal Gain" type="number" min={0} max={1} step={0.01} value={form.graph_extraction_min_marginal_gain} onChange={(value) => updateForm("graph_extraction_min_marginal_gain", value)} />
                <SettingField label="Stall Rounds" type="number" min={1} max={20} value={form.graph_extraction_stall_rounds} onChange={(value) => updateForm("graph_extraction_stall_rounds", value)} />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">并发与超时</p>
              <p className="mt-1 text-sm text-white/52">这些值控制 worker、文件解析、模型请求和 HPO 的上限，避免无界并发。</p>
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <SettingField label="Worker Concurrency" type="number" min={1} max={32} value={form.worker_concurrency} onChange={(value) => updateForm("worker_concurrency", value)} />
                <SettingField label="文件解析并发" type="number" min={1} max={8} value={form.ingestion_file_concurrency} onChange={(value) => updateForm("ingestion_file_concurrency", value)} />
                <SettingField label="模型请求并发" type="number" min={1} max={16} value={form.model_request_concurrency} onChange={(value) => updateForm("model_request_concurrency", value)} />
                <SettingField label="模型超时秒数" type="number" min={5} max={600} value={form.model_request_timeout_seconds} onChange={(value) => updateForm("model_request_timeout_seconds", value)} />
                <SettingField label="HPO 并发" type="number" min={1} max={8} value={form.hpo_concurrency} onChange={(value) => updateForm("hpo_concurrency", value)} />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">检索与回答质量</p>
              <div className="mt-5 grid gap-4">
                <SwitchRow
                  title="检索分层"
                  description="生产建议开启，用于 evidence-first 检索、父子 chunk 装配和图谱增强前的候选组织。"
                  checked={form.retrieval_layer_enabled}
                  onChange={() => updateForm("retrieval_layer_enabled", !form.retrieval_layer_enabled)}
                  disabled={saveMutation.isPending}
                  badge="RETRIEVAL_LAYER_ENABLED"
                />
                <div className="grid gap-4 md:grid-cols-3">
                  <SettingField label="检索缓存 TTL 秒" type="number" min={0} max={86400} value={form.retrieval_cache_ttl_seconds} onChange={(value) => updateForm("retrieval_cache_ttl_seconds", value)} />
                  <SettingField label="引用验证样本数" type="number" min={0} max={20} value={form.citation_verification_sample_max} onChange={(value) => updateForm("citation_verification_sample_max", value)} />
                  <SettingField label="Reflection 重试数" type="number" min={0} max={5} value={form.reflection_max_retries} onChange={(value) => updateForm("reflection_max_retries", value)} />
                </div>
                <SwitchRow
                  title="Agentic Reflection"
                  description="生产建议开启，用于答案前后的引用一致性检查。"
                  checked={form.enable_agentic_reflection}
                  onChange={() => updateForm("enable_agentic_reflection", !form.enable_agentic_reflection)}
                  disabled={saveMutation.isPending}
                  badge="ENABLE_AGENTIC_REFLECTION"
                />
                <SwitchRow
                  title="Post-generation Reflection"
                  description="默认关闭；开启会增加回答后反思和可能的重试成本。"
                  checked={form.enable_post_generation_reflection}
                  onChange={() => updateForm("enable_post_generation_reflection", !form.enable_post_generation_reflection)}
                  disabled={saveMutation.isPending}
                  badge="ENABLE_POST_GENERATION_REFLECTION"
                />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">图谱质量增强</p>
              <div className="mt-5 grid gap-4">
                <SwitchRow
                  title="Auto HPO"
                  description="生产建议开启；全量重建跑全量 HPO，最小更新只针对受影响子图。"
                  checked={form.enable_auto_hpo}
                  onChange={() => updateForm("enable_auto_hpo", !form.enable_auto_hpo)}
                  disabled={saveMutation.isPending}
                  badge="ENABLE_AUTO_HPO"
                />
                <SwitchRow
                  title="社区摘要"
                  description="生产建议开启；全量重建生成全量摘要，最小更新只重算变化社区。"
                  checked={form.enable_graph_community_summaries}
                  onChange={() => updateForm("enable_graph_community_summaries", !form.enable_graph_community_summaries)}
                  disabled={saveMutation.isPending}
                  badge="ENABLE_GRAPH_COMMUNITY_SUMMARIES"
                />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">Chunk 与重排增强</p>
              <div className="mt-5 grid gap-4">
                <SwitchRow
                  title="Semantic Chunking"
                  description="生产初期建议关闭；开启会改变 chunk 边界，需要单独重做检索和图谱基线。"
                  checked={form.semantic_chunking_enabled}
                  onChange={() => updateForm("semantic_chunking_enabled", !form.semantic_chunking_enabled)}
                  disabled={saveMutation.isPending}
                  badge="SEMANTIC_CHUNKING_ENABLED"
                />
                <SwitchRow
                  title="Cross-Encoder Reranker"
                  description="默认关闭；开启后检索阶段会加载 cross-encoder 精排候选，需额外 CPU/GPU 和延迟预算。"
                  checked={form.reranker_enabled}
                  onChange={() => updateForm("reranker_enabled", !form.reranker_enabled)}
                  disabled={saveMutation.isPending}
                  badge="RERANKER_ENABLED"
                />
                <div className="grid gap-4 md:grid-cols-3">
                  <SettingField label="Semantic 最小长度" type="number" min={500} max={5000} value={form.semantic_chunking_min_length} onChange={(value) => updateForm("semantic_chunking_min_length", value)} />
                  <SettingField label="Reranker 模型" value={form.reranker_model} onChange={(value) => updateForm("reranker_model", value)} className="md:col-span-2" />
                  <SettingField label="Reranker Max Length" type="number" min={64} max={2048} value={form.reranker_max_length} onChange={(value) => updateForm("reranker_max_length", value)} />
                </div>
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">密钥</p>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <label className="flex flex-col gap-2">
                  <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/46">Chat API Key</span>
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3">
                    <KeyRound className="size-4 text-cyan-100/58" />
                    <input
                      type="password"
                      value={showApiKeyMask ? "••••••••••••••••" : form.api_key}
                      readOnly={showApiKeyMask}
                      disabled={form.clear_api_key}
                      onChange={(event) => updateForm("api_key", event.target.value)}
                      placeholder={settings?.has_api_key ? "留空则保留当前 key" : "输入 API key"}
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
                  <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/46">Embedding API Key</span>
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3">
                    <KeyRound className="size-4 text-cyan-100/58" />
                    <input
                      type="password"
                      value={showEmbeddingApiKeyMask ? "••••••••••••••••" : form.embedding_api_key}
                      readOnly={showEmbeddingApiKeyMask}
                      disabled={form.clear_embedding_api_key}
                      onChange={(event) => updateForm("embedding_api_key", event.target.value)}
                      placeholder={settings?.has_embedding_api_key ? "留空则保留当前 key" : "输入 Embedding API key"}
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
                  清除当前 Chat API key
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
                  清除当前 Embedding API key
                </label>
              </div>
            </section>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-5">
              <p className="text-xs leading-6 text-white/42">
                保存前会检查运行时环境；fallback 不在页面开放开启，生产路径必须保持禁用。
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

      {runtimeQuery.data?.warnings.length ? (
        <section className="glass-panel rounded-[24px] p-5">
          <p className="flex items-center gap-2 text-sm font-semibold text-amber-100">
            <XCircle className="size-4" />
            运行时警告
          </p>
          <div className="mt-3 grid gap-2">
            {runtimeQuery.data.warnings.map((issue) => (
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
