"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelSettingsUpdate, RuntimeCheckResponse, RuntimeIssue, StrategyProfileDetail, StructuredApiErrorBody } from "@course-kg/shared";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Copy,
  Bot,
  EyeOff,
  FilePlus2,
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
  worker_concurrency: string;
  ingestion_file_concurrency: string;
  model_request_concurrency: string;
  model_request_timeout_seconds: string;
  chunk_token_budget: string;
  api_key: string;
  clear_api_key: boolean;
  embedding_api_key: string;
  clear_embedding_api_key: boolean;
  model_bridge_enabled: boolean;
  reranker_enabled: boolean;
  reranker_model: string;
  reranker_max_length: string;
  reranker_device: "cpu" | "cuda";
  semantic_chunking_enabled: boolean;
  semantic_chunking_min_length: string;
  retrieval_layer_enabled: boolean;
  retrieval_cache_ttl_seconds: string;
  enable_agentic_reflection: boolean;
  enable_post_generation_reflection: boolean;
  citation_verification_sample_max: string;
  reflection_max_retries: string;
  enable_graph_community_summaries: boolean;
  signal_extraction_max_model_batches: string;
  signal_extraction_max_candidates_per_batch: string;
  signal_extraction_max_tokens_per_batch: string;
  signal_candidate_keep_threshold: string;
  community_louvain_resolution: string;
  community_min_modularity_warn: string;
  graph_overview_max_nodes: string;
  graph_overview_max_edges: string;
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
      message: "Profile JSON 必须是对象",
      reason: "根节点需要是 strategy_profile_v1 对象，不能是数组、字符串或空值。",
    });
    return diagnostics;
  }
  const profile = parsed as Record<string, unknown>;
  for (const key of ["schema_version", "ui_labels", "prompt_pack", "schema_pack", "parsing_strategy", "graph_strategy", "retrieval_strategy", "quality_policy"]) {
    if (!(key in profile)) {
      diagnostics.push({
        line: 1,
        column: 1,
        severity: "warning",
        message: `缺少 ${key}`,
        reason: "后端会尝试补默认值，但建议显式保留完整结构，避免 Profile 语义不清。",
      });
    }
  }
  const schemaPack = profile.schema_pack;
  if (!schemaPack || typeof schemaPack !== "object" || Array.isArray(schemaPack)) {
      diagnostics.push({
        line: getLineForKey(text, "schema_pack"),
        column: 1,
        severity: "error",
        message: "schema_pack 必须是对象",
        reason: "Evidence atom、观测边类型、别名和禁用标签都需要放在 schema_pack 中。",
      });
    return diagnostics;
  }
  const schema = schemaPack as Record<string, unknown>;
  const entityTypes = schema.entity_types;
  const relationTypes = schema.relation_types;
  if (!Array.isArray(entityTypes) || entityTypes.some((item) => typeof item !== "string" || !item.trim())) {
    diagnostics.push({
      line: getLineForKey(text, "entity_types"),
      column: 1,
      severity: "error",
      message: "schema_pack.entity_types 必须是非空字符串数组",
      reason: "Evidence atom 归一和诊断视图需要至少一个合法类型。",
    });
  }
  if (!Array.isArray(relationTypes) || relationTypes.some((item) => typeof item !== "string" || !item.trim())) {
    diagnostics.push({
      line: getLineForKey(text, "relation_types"),
      column: 1,
      severity: "error",
      message: "schema_pack.relation_types 必须是非空字符串数组",
      reason: "Evidence edge 观测类型需要至少一个合法类型。",
    });
  }
  const promptPack = profile.prompt_pack;
  if (!promptPack || typeof promptPack !== "object" || Array.isArray(promptPack)) {
    diagnostics.push({
      line: getLineForKey(text, "prompt_pack"),
      column: 1,
      severity: "error",
      message: "prompt_pack 必须是对象",
      reason: "答案 grounding、引用验证、反思和社区摘要 prompt 都需要从 prompt_pack 读取。",
    });
  }
  return diagnostics;
}

function parseProfileJson(text: string): { value?: Record<string, unknown>; error?: string } {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { error: "Profile JSON 必须是对象。" };
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
        throw new Error(parsed.error || "Profile JSON 无效。");
      }
      return updateStrategyProfile(selectedProfileId, {
        name: name.trim(),
        library_type: libraryType.trim() || "custom",
        profile_json: parsed.value,
      });
    },
    onSuccess: async (data) => {
      setMessage("Profile 已保存。");
      setJsonText(JSON.stringify(data.profile.profile_json, null, 2));
      await invalidateProfiles();
    },
    onError,
  });

  const copyMutation = useMutation({
    mutationFn: () => copyStrategyProfile(selectedProfileId, { name: `${name || currentProfile?.name || "Profile"} Copy` }),
    onSuccess: async (data) => {
      setSelectedProfileId(data.profile.id);
      setMessage("已复制为自定义 Profile。");
      await invalidateProfiles();
    },
    onError,
  });

  const createMutation = useMutation({
    mutationFn: () => createStrategyProfile({ name: "新 Profile", library_type: "custom", profile_json: parsed.value || {} }),
    onSuccess: async (data) => {
      setSelectedProfileId(data.profile.id);
      setMessage("已创建新 Profile。");
      await invalidateProfiles();
    },
    onError,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteStrategyProfile(selectedProfileId),
    onSuccess: async () => {
      setDeleteConfirmOpen(false);
      setSelectedProfileId("");
      setMessage("Profile 已删除；如有资料库曾绑定它，后端已自动切回默认 Profile。");
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
      setMessage("已设为当前资料库 Profile。");
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
          content: streamedText || "已生成 Profile 草案。",
          profileJson: streamedResult?.profileJson,
          warnings: streamedResult?.warnings,
          profileHash: streamedResult?.profileHash,
        },
      ]);
      setAssistantDraft("");
      setAssistantResult(null);
    } catch (error) {
      const messageText = error instanceof Error ? error.message : "Profile 助手生成失败";
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
        ? "草案已填入高级 JSON。内置 Profile 受保护，请复制后保存。"
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
  const deleteBlockedReason = isBuiltin ? "默认内置 Profile 受保护；请复制后编辑。" : null;
  const deleteImpactMessage =
    selectedProfileKnowledgeBaseIds.length > 0
      ? `该 Profile 当前绑定 ${selectedProfileKnowledgeBaseIds.length} 个资料库。删除后，这些资料库会自动切回默认 Profile；已有 chunks、图谱、向量和会话不会被改写。`
      : "该 Profile 当前没有绑定资料库。删除后会从列表中隐藏，已有历史数据不会被改写。";
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
        {profileHash ? <span className="break-all">hash {profileHash}</span> : null}
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
          <p className="section-kicker">Profile 设置</p>
          <h2 className="glow-text mt-2 text-3xl font-semibold text-white">资料库 Profile</h2>
          <p className="mt-4 text-sm leading-7 text-cyan-50/62">
            Profile 只影响之后启动的新解析、evidence graph、检索和问答任务；已有 chunks、向量、图谱和会话不会被自动改写。
          </p>
        </div>
        <div className={sectionClass}>
          <p className="text-sm font-semibold text-white">当前绑定</p>
          <div className="mt-3 space-y-2 text-sm leading-6 text-white/62">
            <p>资料库：{selectedKnowledgeBase?.name ?? "未选择"}</p>
            <p>Profile：{activeProfile?.name ?? selectedKnowledgeBase?.active_profile_name ?? "未绑定"}</p>
            <p className="break-all">Hash：{selectedKnowledgeBase?.active_profile_hash ?? "missing"}</p>
          </div>
          {hashMismatch ? (
            <p className="mt-4 rounded-xl border border-amber-200/20 bg-amber-200/[0.06] p-3 text-sm leading-6 text-amber-100">
              当前资料库记录的 Profile hash 与列表中的 Profile hash 不一致。切换或修改后，请显式重新解析或重建图谱。
            </p>
          ) : null}
        </div>
      </aside>

      <div className="grid gap-5">
        <section className={sectionClass}>
          <div className="grid gap-4 md:grid-cols-[1fr_0.7fr]">
            <label className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/46">Profile</span>
              <select
                value={selectedProfileId}
                onChange={(event) => setSelectedProfileId(event.target.value)}
                className={`${inputClass} kg-dark-select outline-none`}
              >
                {(profilesQuery.data ?? []).map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}{profile.is_builtin ? " / built-in" : ""}
                  </option>
                ))}
              </select>
            </label>
            <SettingField label="Library Type" value={libraryType} onChange={setLibraryType} disabled={isBuiltin} />
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
              删除 Profile
            </Button>
          </div>
          {deleteBlockedReason ? <p className="mt-3 text-sm leading-6 text-amber-100/80">{deleteBlockedReason}</p> : null}
        </section>

        <section className={sectionClass}>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-white">高级 JSON</p>
              <p className="mt-1 text-sm text-white/52">结构化字段与高级 JSON 共用同一份 Profile schema。</p>
            </div>
            <Button type="button" className="rounded-full" onClick={() => saveMutation.mutate()} disabled={isBuiltin || hasJsonErrors || saveMutation.isPending}>
              {saveMutation.isPending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Save data-icon="inline-start" />}
              保存 Profile
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
                  未发现格式错误。保存时仍会执行后端 Profile schema 校验。
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
            {isBuiltin ? <p className="text-sm text-white/48">内置 Profile 受保护；复制后可编辑。</p> : null}
            {message ? <p className="text-sm text-cyan-100">{message}</p> : null}
          </div>
        </section>
      </div>

      {assistantOpen ? (
        <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-white/10 bg-[#07111f]/95 p-5 text-white shadow-2xl backdrop-blur-xl">
          <div className="flex items-start justify-between gap-4 border-b border-white/10 pb-4">
            <div>
              <p className="section-kicker">AI 设置助手</p>
              <h3 className="mt-2 text-2xl font-semibold">Profile 对话草案</h3>
              {assistantSessionId ? <p className="mt-1 max-w-sm truncate text-xs text-white/45">Redis 会话：{assistantSessionId}</p> : null}
            </div>
            <Button type="button" variant="outline" className="rounded-full" onClick={() => setAssistantOpen(false)}>
              关闭
            </Button>
          </div>
          <div ref={assistantScrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto py-4 pr-1">
            {assistantMessages.length === 0 && !assistantDraft ? (
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-sm leading-7 text-white/62">
                输入资料库类型、实体/关系、章节或条款规则、引用要求和检索偏好。助手会先输出说明，再给出一个可填充的高级 JSON 草案。
              </div>
            ) : null}
            {assistantMessages.map((item) => (
              <div key={item.id} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[92%] rounded-2xl border p-3 text-sm leading-7 ${item.role === "user" ? "border-cyan-200/20 bg-cyan-200/[0.08] text-cyan-50" : "border-white/10 bg-white/[0.04] text-white/78"}`}>
                  {item.role === "assistant" ? (
                    <div className="mb-2 flex items-center gap-2 text-xs text-cyan-100/65">
                      <Bot className="size-3.5" />
                      Profile Assistant
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
                    <span className="signal-bars">
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
              placeholder="描述资料库类型、实体/关系、引用规则、分区规则和检索偏好。"
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
            <DialogTitle>确认删除 Profile</DialogTitle>
            <DialogDescription className="break-words text-cyan-100/70">
              {currentProfile?.name ? `即将删除「${currentProfile.name}」。默认内置 Profile 不能删除，其他 Profile 删除后会软删除并从列表隐藏。` : "即将删除当前 Profile。"}
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
      chat_base_url: settingsQuery.data.chat_base_url,
      embedding_base_url: settingsQuery.data.embedding_base_url ?? "",
      chat_resolve_ip: settingsQuery.data.chat_resolve_ip ?? "",
      embedding_resolve_ip: settingsQuery.data.embedding_resolve_ip ?? "",
      embedding_model: settingsQuery.data.embedding_model,
      chat_model: settingsQuery.data.chat_model,
      embedding_dimensions: String(settingsQuery.data.embedding_dimensions),
      worker_concurrency: String(settingsQuery.data.worker_concurrency ?? 3),
      ingestion_file_concurrency: String(settingsQuery.data.ingestion_file_concurrency ?? 3),
      model_request_concurrency: String(settingsQuery.data.model_request_concurrency ?? 3),
      model_request_timeout_seconds: String(settingsQuery.data.model_request_timeout_seconds ?? 240),
      chunk_token_budget: String(settingsQuery.data.chunk_token_budget ?? 2400),
      api_key: "",
      clear_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: settingsQuery.data.model_bridge_enabled ?? true,
      reranker_enabled: settingsQuery.data.reranker_enabled ?? false,
      reranker_model: settingsQuery.data.reranker_model ?? "cross-encoder/ms-marco-MiniLM-L-6-v2",
      reranker_max_length: String(settingsQuery.data.reranker_max_length ?? 512),
      reranker_device: settingsQuery.data.reranker_device === "cuda" ? "cuda" : "cpu",
      semantic_chunking_enabled: settingsQuery.data.semantic_chunking_enabled ?? false,
      semantic_chunking_min_length: String(settingsQuery.data.semantic_chunking_min_length ?? 2000),
      retrieval_layer_enabled: settingsQuery.data.retrieval_layer_enabled ?? true,
      retrieval_cache_ttl_seconds: String(settingsQuery.data.retrieval_cache_ttl_seconds ?? 120),
      enable_agentic_reflection: settingsQuery.data.enable_agentic_reflection ?? true,
      enable_post_generation_reflection: settingsQuery.data.enable_post_generation_reflection ?? false,
      citation_verification_sample_max: String(settingsQuery.data.citation_verification_sample_max ?? 3),
      reflection_max_retries: String(settingsQuery.data.reflection_max_retries ?? 2),
      enable_graph_community_summaries: settingsQuery.data.enable_graph_community_summaries ?? true,
      signal_extraction_max_model_batches: String(settingsQuery.data.signal_extraction_max_model_batches ?? 4),
      signal_extraction_max_candidates_per_batch: String(settingsQuery.data.signal_extraction_max_candidates_per_batch ?? 40),
      signal_extraction_max_tokens_per_batch: String(settingsQuery.data.signal_extraction_max_tokens_per_batch ?? 6000),
      signal_candidate_keep_threshold: String(settingsQuery.data.signal_candidate_keep_threshold ?? 0.62),
      community_louvain_resolution: String(settingsQuery.data.community_louvain_resolution ?? 1.0),
      community_min_modularity_warn: String(settingsQuery.data.community_min_modularity_warn ?? 0.18),
      graph_overview_max_nodes: String(settingsQuery.data.graph_overview_max_nodes ?? 260),
      graph_overview_max_edges: String(settingsQuery.data.graph_overview_max_edges ?? 800),
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
  const crossEncoderStatus = runtimeQuery.data?.reranker;

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
    worker_concurrency: parseIntField(form.worker_concurrency),
    ingestion_file_concurrency: parseIntField(form.ingestion_file_concurrency),
    model_request_concurrency: parseIntField(form.model_request_concurrency),
    model_request_timeout_seconds: parseIntField(form.model_request_timeout_seconds),
    chunk_token_budget: parseIntField(form.chunk_token_budget),
    api_key: form.api_key.trim() || null,
    clear_api_key: form.clear_api_key,
    embedding_api_key: form.embedding_api_key.trim() || null,
    clear_embedding_api_key: form.clear_embedding_api_key,
    model_bridge_enabled: form.model_bridge_enabled,
    reranker_enabled: form.reranker_enabled,
    reranker_model: form.reranker_model.trim(),
    reranker_max_length: parseIntField(form.reranker_max_length),
    reranker_device: form.reranker_device,
    semantic_chunking_enabled: form.semantic_chunking_enabled,
    semantic_chunking_min_length: parseIntField(form.semantic_chunking_min_length),
    retrieval_layer_enabled: form.retrieval_layer_enabled,
    retrieval_cache_ttl_seconds: parseIntField(form.retrieval_cache_ttl_seconds),
    enable_agentic_reflection: form.enable_agentic_reflection,
    enable_post_generation_reflection: form.enable_post_generation_reflection,
    citation_verification_sample_max: parseIntField(form.citation_verification_sample_max),
    reflection_max_retries: parseIntField(form.reflection_max_retries),
    enable_graph_community_summaries: form.enable_graph_community_summaries,
    signal_extraction_max_model_batches: parseIntField(form.signal_extraction_max_model_batches),
    signal_extraction_max_candidates_per_batch: parseIntField(form.signal_extraction_max_candidates_per_batch),
    signal_extraction_max_tokens_per_batch: parseIntField(form.signal_extraction_max_tokens_per_batch),
    signal_candidate_keep_threshold: parseFloatField(form.signal_candidate_keep_threshold),
    community_louvain_resolution: parseFloatField(form.community_louvain_resolution),
    community_min_modularity_warn: parseFloatField(form.community_min_modularity_warn),
    graph_overview_max_nodes: parseIntField(form.graph_overview_max_nodes),
    graph_overview_max_edges: parseIntField(form.graph_overview_max_edges),
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
            Profile 设置
          </button>
        </div>
        {activeTab === "profile" ? <ProfileSettingsPanel onError={(error) => setErrorDialog(errorDialogFromUnknown(error))} /> : null}
        <div className={activeTab === "model" ? "grid gap-7 xl:grid-cols-[minmax(320px,0.72fr)_minmax(560px,1.28fr)]" : "hidden"}>
          <aside className="space-y-6">
            <div>
              <p className="section-kicker">生产参数配置</p>
              <h2 className="glow-text mt-2 text-4xl font-semibold text-white">运行时设置</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-cyan-50/62">
                这里配置会写入根目录 .env，并通过后端热加载影响新任务。模型名、并发、检索增强和质量参数都按当前生产参数表组织。
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <StatusPill ok={Boolean(settings?.has_api_key)}>Chat Key {settings?.has_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(settings?.has_embedding_api_key)}>Embedding Key {settings?.has_embedding_api_key ? "已配置" : "未配置"}</StatusPill>
              <StatusPill ok={Boolean(runtimeQuery.data?.env_sync.synced)}>{runtimeQuery.data?.env_sync.synced ? ".env 已同步" : ".env 需检查"}</StatusPill>
              <StatusPill ok={!settings?.enable_model_fallback && !settings?.enable_database_fallback}>Fallback 已禁用</StatusPill>
              <StatusPill ok={Boolean(settings?.runtime_settings_version)}>Runtime {settings?.runtime_settings_version ? settings.runtime_settings_version.slice(0, 12) : "pending"}</StatusPill>
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
                <SettingField label="Chat 模型" value={form.chat_model} onChange={(value) => updateForm("chat_model", value)} />
                <SettingField label="Embedding 模型" value={form.embedding_model} onChange={(value) => updateForm("embedding_model", value)} />
                <SettingField label="Embedding 维度" type="number" min={1} max={8192} value={form.embedding_dimensions} onChange={(value) => updateForm("embedding_dimensions", value)} />
              </div>
            </section>

            <section className={sectionClass}>
              <p className="text-sm font-semibold text-white">并发与超时</p>
              <p className="mt-1 text-sm text-white/52">这些值控制 worker、文件解析和模型请求的上限，避免无界并发。</p>
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <SettingField label="Worker Concurrency" type="number" min={1} max={32} value={form.worker_concurrency} onChange={(value) => updateForm("worker_concurrency", value)} />
                <SettingField label="文件解析并发" type="number" min={1} max={8} value={form.ingestion_file_concurrency} onChange={(value) => updateForm("ingestion_file_concurrency", value)} />
                <SettingField label="模型请求并发" type="number" min={1} max={16} value={form.model_request_concurrency} onChange={(value) => updateForm("model_request_concurrency", value)} />
                <SettingField label="模型超时秒数" type="number" min={5} max={600} value={form.model_request_timeout_seconds} onChange={(value) => updateForm("model_request_timeout_seconds", value)} />
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
                  title="社区摘要"
                  description="生产建议开启；全量重建生成全量摘要，最小更新只重算变化社区。"
                  checked={form.enable_graph_community_summaries}
                  onChange={() => updateForm("enable_graph_community_summaries", !form.enable_graph_community_summaries)}
                  disabled={saveMutation.isPending}
                  badge="ENABLE_GRAPH_COMMUNITY_SUMMARIES"
                />
                <div className="grid gap-4 md:grid-cols-4">
                  <SettingField label="Signal Model Batches" type="number" min={0} max={64} value={form.signal_extraction_max_model_batches} onChange={(value) => updateForm("signal_extraction_max_model_batches", value)} />
                  <SettingField label="Signal Batch Size" type="number" min={1} max={500} value={form.signal_extraction_max_candidates_per_batch} onChange={(value) => updateForm("signal_extraction_max_candidates_per_batch", value)} />
                  <SettingField label="Signal Tokens/Batch" type="number" min={500} max={50000} value={form.signal_extraction_max_tokens_per_batch} onChange={(value) => updateForm("signal_extraction_max_tokens_per_batch", value)} />
                  <SettingField label="Signal Keep Threshold" type="number" min={0} max={1} step={0.01} value={form.signal_candidate_keep_threshold} onChange={(value) => updateForm("signal_candidate_keep_threshold", value)} />
                  <SettingField label="Louvain Resolution" type="number" min={0.05} max={5} step={0.05} value={form.community_louvain_resolution} onChange={(value) => updateForm("community_louvain_resolution", value)} />
                  <SettingField label="Modularity Warn" type="number" min={-1} max={1} step={0.01} value={form.community_min_modularity_warn} onChange={(value) => updateForm("community_min_modularity_warn", value)} />
                  <SettingField label="Overview Nodes" type="number" min={20} max={2000} value={form.graph_overview_max_nodes} onChange={(value) => updateForm("graph_overview_max_nodes", value)} />
                  <SettingField label="Overview Edges" type="number" min={20} max={5000} value={form.graph_overview_max_edges} onChange={(value) => updateForm("graph_overview_max_edges", value)} />
                </div>
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
                <div className="rounded-xl border border-fuchsia-200/15 bg-fuchsia-300/[0.055] p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-white">Cross-Encoder 模型配置</p>
                      <p className="mt-1 text-sm leading-6 text-white/56">
                        保存后写入共享 .env，并通过 runtime settings version 广播；API 与 worker 会在任务边界刷新并清理 reranker 单例。
                      </p>
                    </div>
                    <StatusPill ok={Boolean(crossEncoderStatus?.healthy)}>
                      {crossEncoderStatus?.enabled ? (crossEncoderStatus.healthy ? "loaded" : "enabled") : "disabled"}
                    </StatusPill>
                  </div>
                  <div className="mt-4 grid gap-3 text-sm text-white/62 md:grid-cols-2">
                    <div className="rounded-xl border border-white/8 bg-black/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-white/38">Configured Model</p>
                      <p className="mt-2 break-words text-white/78">{form.reranker_model || "n/a"}</p>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-black/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-white/38">Runtime Model</p>
                      <p className="mt-2 break-words text-white/78">{crossEncoderStatus?.reported_model ?? crossEncoderStatus?.model ?? "not loaded"}</p>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-black/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-white/38">Configured Device</p>
                      <p className="mt-2 text-white/78">{form.reranker_device}</p>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-black/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-white/38">Runtime Device</p>
                      <p className="mt-2 text-white/78">{crossEncoderStatus?.reported_device ?? crossEncoderStatus?.device ?? "not loaded"}</p>
                    </div>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-3">
                  <SettingField label="Semantic 最小长度" type="number" min={500} max={5000} value={form.semantic_chunking_min_length} onChange={(value) => updateForm("semantic_chunking_min_length", value)} />
                  <SettingField label="Reranker 模型" value={form.reranker_model} onChange={(value) => updateForm("reranker_model", value)} className="md:col-span-2" />
                  <SettingField label="Cross-Encoder Max Length" type="number" min={64} max={2048} value={form.reranker_max_length} onChange={(value) => updateForm("reranker_max_length", value)} />
                  <label className="flex flex-col gap-2">
                    <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/46">Cross-Encoder Device</span>
                    <select
                      value={form.reranker_device}
                      onChange={(event) => updateForm("reranker_device", event.target.value === "cuda" ? "cuda" : "cpu")}
                      disabled={saveMutation.isPending}
                      className={inputClass}
                    >
                      <option value="cpu" className="bg-[#081126] text-white">cpu</option>
                      <option value="cuda" className="bg-[#081126] text-white">cuda</option>
                    </select>
                  </label>
                  <SettingField label="Chunk Token Budget" type="number" min={256} max={20000} value={form.chunk_token_budget} onChange={(value) => updateForm("chunk_token_budget", value)} />
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
