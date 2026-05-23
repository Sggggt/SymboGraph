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
  graph_extraction_soft_start_budget: string;
  graph_extraction_concurrency: string;
  graph_extraction_resume_batch_size: string;
  api_key: string;
  clear_api_key: boolean;
  embedding_api_key: string;
  clear_embedding_api_key: boolean;
  model_bridge_enabled: boolean;
  reranker_enabled: boolean;
  reranker_max_length: string;
  semantic_chunking_enabled: boolean;
  semantic_chunking_min_length: string;
  enable_auto_hpo: boolean;
};

type ErrorDialogState = {
  title: string;
  message: string;
  status?: number;
  issues: RuntimeIssue[];
  fixCommands: string[];
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
      graph_extraction_soft_start_budget: String(settingsQuery.data.graph_extraction_soft_start_budget ?? 120),
      graph_extraction_concurrency: String(settingsQuery.data.graph_extraction_concurrency ?? 2),
      graph_extraction_resume_batch_size: String(settingsQuery.data.graph_extraction_resume_batch_size ?? 24),
      api_key: "",
      clear_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: settingsQuery.data.model_bridge_enabled ?? false,
      reranker_enabled: settingsQuery.data.reranker_enabled ?? false,
      reranker_max_length: String(settingsQuery.data.reranker_max_length ?? 512),
      semantic_chunking_enabled: settingsQuery.data.semantic_chunking_enabled ?? true,
      semantic_chunking_min_length: String(settingsQuery.data.semantic_chunking_min_length ?? 2000),
      enable_auto_hpo: settingsQuery.data.enable_auto_hpo ?? false,
    });
    setApiKeyEditing(false);
    setEmbeddingApiKeyEditing(false);
  }, [settingsQuery.data]);

  const saveMutation = useMutation({
    mutationFn: (payload: ModelSettingsUpdate) => updateModelSettings(payload),
    onSuccess: async () => {
      setApiKeyEditing(false);
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

  const buildPayload = (): ModelSettingsUpdate => {
    const dimensions = Number.parseInt(form.embedding_dimensions, 10);
    const graphSoftStartBudget = Number.parseInt(form.graph_extraction_soft_start_budget, 10);
    const graphConcurrency = Number.parseInt(form.graph_extraction_concurrency, 10);
    const graphResumeBatchSize = Number.parseInt(form.graph_extraction_resume_batch_size, 10);
    const rerankerMaxLength = Number.parseInt(form.reranker_max_length, 10);
    return {
      chat_base_url: form.chat_base_url.trim(),
      embedding_base_url: form.embedding_base_url.trim(),
      chat_resolve_ip: form.chat_resolve_ip.trim() || null,
      embedding_resolve_ip: form.embedding_resolve_ip.trim() || null,
      embedding_model: form.embedding_model.trim(),
      chat_model: form.chat_model.trim(),
      embedding_dimensions: Number.isFinite(dimensions) ? dimensions : undefined,
      graph_extraction_soft_start_budget: Number.isFinite(graphSoftStartBudget) ? graphSoftStartBudget : undefined,
      graph_extraction_concurrency: Number.isFinite(graphConcurrency) ? graphConcurrency : undefined,
      graph_extraction_resume_batch_size: Number.isFinite(graphResumeBatchSize) ? graphResumeBatchSize : undefined,
      api_key: form.api_key.trim() || null,
      clear_api_key: form.clear_api_key,
      embedding_api_key: form.embedding_api_key.trim() || null,
      clear_embedding_api_key: form.clear_embedding_api_key,
      model_bridge_enabled: form.model_bridge_enabled,
      reranker_enabled: form.reranker_enabled,
      reranker_max_length: Number.isFinite(rerankerMaxLength) ? rerankerMaxLength : undefined,
      semantic_chunking_enabled: true,
      semantic_chunking_min_length: 2000,
      enable_auto_hpo: form.enable_auto_hpo,
    };
  };

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
      <section className="glass-panel relative overflow-hidden rounded-[34px] p-6 lg:p-8">
        <div className="relative z-10 grid gap-7 xl:grid-cols-[minmax(320px,0.72fr)_minmax(520px,1.28fr)]">
          <div className="space-y-6">
            <div>
              <p className="section-kicker">模型与检索基础设施</p>
              <h2 className="glow-text mt-2 text-4xl font-semibold text-white">运行时设置</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-cyan-50/62">
                配置 OpenAI-compatible 模型接口。检索使用 Dense + BM25 + WSF 轻量精排，零外部模型依赖。
              </p>
            </div>

            <div className="grid gap-3">
              <div className="border-l border-white/10 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.26em] text-white/42">模型 API</p>
                <p className="mt-2 flex items-center gap-2 text-sm text-white/68">
                  {settings?.has_api_key ? <CheckCircle2 className="size-4 text-emerald-200" /> : <ShieldAlert className="size-4 text-amber-200" />}
                  {settings?.has_api_key ? "Chat API Key 已配置" : "Chat API Key 未配置"}
                </p>
                <p className="mt-1 flex items-center gap-2 text-sm text-white/68">
                  {settings?.has_embedding_api_key ? <CheckCircle2 className="size-4 text-emerald-200" /> : <ShieldAlert className="size-4 text-amber-200" />}
                  {settings?.has_embedding_api_key ? "Embedding API Key 已配置" : "Embedding API Key 未配置"}
                </p>
              </div>
              <div className="border-l border-white/10 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.26em] text-white/42">环境参数</p>
                <p className="mt-2 text-sm text-white/68">
                  {runtimeQuery.data?.env_sync.synced ? "与 .env.example 同步" : "需要检查 .env"}
                </p>
              </div>
            </div>
          </div>

          <form
            className="relative z-10 grid gap-5"
            onSubmit={(event) => {
              event.preventDefault();
              void handleSubmit();
            }}
          >
            <div className="grid gap-5 md:grid-cols-2">
              <div className="flex flex-wrap items-center justify-between gap-4 md:col-span-2 rounded-2xl border border-white/10 bg-white/[0.035] p-5">
                <div>
                  <p className="flex items-center gap-2 text-sm font-semibold text-white">
                    <SlidersHorizontal className="size-4 text-cyan-100/70" />
                    Cross-Encoder 重排序
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/58">
                    开启后检索阶段使用 cross-encoder 对候选结果精排（模型：{settings?.reranker_model || "cross-encoder/ms-marco-MiniLM-L-6-v2"}）。
                  </p>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={form.reranker_enabled}
                  disabled={saveMutation.isPending}
                  onClick={() => updateForm("reranker_enabled", !form.reranker_enabled)}
                  className={`relative h-8 w-16 rounded-full border transition ${
                    form.reranker_enabled ? "border-cyan-100/40 bg-cyan-300/70" : "border-white/14 bg-white/10"
                  } disabled:cursor-not-allowed disabled:opacity-60`}
                >
                  <span
                    className={`absolute top-1 size-6 rounded-full bg-white shadow transition ${
                      form.reranker_enabled ? "left-9" : "left-1"
                    }`}
                  />
                </button>
              </div>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Reranker Max Length</span>
                <Input type="number" min={64} max={2048} value={form.reranker_max_length} onChange={(event) => updateForm("reranker_max_length", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <div className="flex flex-wrap items-center justify-between gap-4 md:col-span-2 rounded-2xl border border-white/10 bg-white/[0.035] p-5">
                <div>
                  <p className="flex items-center gap-2 text-sm font-semibold text-white">
                    <SlidersHorizontal className="size-4 text-cyan-100/70" />
                    自适应 HPO 自动调参 (Auto HPO)
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/58">
                    开启后，系统在每一次解析新文件或重建图谱时，都会同步自动运行 Optuna 寻优算法，为当前课程提炼最佳的图谱特征比例与过滤阈值。
                  </p>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={form.enable_auto_hpo}
                  disabled={saveMutation.isPending}
                  onClick={() => updateForm("enable_auto_hpo", !form.enable_auto_hpo)}
                  className={`relative h-8 w-16 rounded-full border transition ${
                    form.enable_auto_hpo ? "border-cyan-100/40 bg-cyan-300/70" : "border-white/14 bg-white/10"
                  } disabled:cursor-not-allowed disabled:opacity-60`}
                >
                  <span
                    className={`absolute top-1 size-6 rounded-full bg-white shadow transition ${
                      form.enable_auto_hpo ? "left-9" : "left-1"
                    }`}
                  />
                </button>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-4 md:col-span-2 rounded-2xl border border-white/10 bg-white/[0.035] p-5">
                <div>
                  <p className="flex items-center gap-2 text-sm font-semibold text-white">
                    <SlidersHorizontal className="size-4 text-cyan-100/70" />
                    宿主机模型桥接 (Model Bridge)
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/58">
                    开启后容器将通过宿主机网络访问模型 (http://host.docker.internal)。更改此项后需重新执行 <code className="rounded bg-black/30 px-1.5 py-0.5 text-xs text-amber-100">.\start-app.ps1</code> 才能生效。
                  </p>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={form.model_bridge_enabled}
                  disabled={saveMutation.isPending}
                  onClick={() => updateForm("model_bridge_enabled", !form.model_bridge_enabled)}
                  className={`relative h-8 w-16 rounded-full border transition ${
                    form.model_bridge_enabled ? "border-cyan-100/40 bg-cyan-300/70" : "border-white/14 bg-white/10"
                  } disabled:cursor-not-allowed disabled:opacity-60`}
                >
                  <span
                    className={`absolute top-1 size-6 rounded-full bg-white shadow transition ${
                      form.model_bridge_enabled ? "left-9" : "left-1"
                    }`}
                  />
                </button>
              </div>

              <label className="flex flex-col gap-2 md:col-span-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Chat Base URL</span>
                <Input value={form.chat_base_url} onChange={(event) => updateForm("chat_base_url", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2 md:col-span-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Embedding Base URL</span>
                <Input value={form.embedding_base_url} onChange={(event) => updateForm("embedding_base_url", event.target.value)} placeholder="输入 Embedding Base URL" className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white placeholder:text-white/28" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Chat DNS Override IP</span>
                <Input value={form.chat_resolve_ip} onChange={(event) => updateForm("chat_resolve_ip", event.target.value)} placeholder="可选；留空使用系统 DNS" className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white placeholder:text-white/28" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Embedding DNS Override IP</span>
                <Input value={form.embedding_resolve_ip} onChange={(event) => updateForm("embedding_resolve_ip", event.target.value)} placeholder="可选；留空使用系统 DNS" className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white placeholder:text-white/28" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Embedding 模型</span>
                <Input value={form.embedding_model} onChange={(event) => updateForm("embedding_model", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Chat / 图谱模型</span>
                <Input value={form.chat_model} onChange={(event) => updateForm("chat_model", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">向量维度</span>
                <Input type="number" min={1} max={8192} value={form.embedding_dimensions} onChange={(event) => updateForm("embedding_dimensions", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">图谱初始抽取预算</span>
                <Input type="number" min={1} value={form.graph_extraction_soft_start_budget} onChange={(event) => updateForm("graph_extraction_soft_start_budget", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">图谱抽取并发数</span>
                <Input type="number" min={1} max={8} value={form.graph_extraction_concurrency} onChange={(event) => updateForm("graph_extraction_concurrency", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">每批抽取片段数</span>
                <Input type="number" min={1} max={100} value={form.graph_extraction_resume_batch_size} onChange={(event) => updateForm("graph_extraction_resume_batch_size", event.target.value)} className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white" />
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Chat API Key</span>
                <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.04] px-4">
                  <KeyRound className="size-4 text-cyan-100/58" />
                  <input
                    type="password"
                    value={showApiKeyMask ? "••••••••••••••••" : form.api_key}
                    readOnly={showApiKeyMask}
                    disabled={form.clear_api_key}
                    onChange={(event) => updateForm("api_key", event.target.value)}
                    placeholder={settings?.has_api_key ? "留空则保留当前 key" : "输入 API key"}
                    className="h-12 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30"
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
                <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">Embedding API Key</span>
                <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.04] px-4">
                  <KeyRound className="size-4 text-cyan-100/58" />
                  <input
                    type="password"
                    value={showEmbeddingApiKeyMask ? "••••••••••••••••" : form.embedding_api_key}
                    readOnly={showEmbeddingApiKeyMask}
                    disabled={form.clear_embedding_api_key}
                    onChange={(event) => updateForm("embedding_api_key", event.target.value)}
                    placeholder={settings?.has_embedding_api_key ? "留空则保留当前 key" : "输入 Embedding API key"}
                    className="h-12 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30"
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

            <div className="grid gap-3 md:grid-cols-2">
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

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-5">
              <p className="text-xs leading-6 text-white/42">
                保存前会检查运行时和环境参数；模型 fallback 与数据库 fallback 默认关闭。
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
