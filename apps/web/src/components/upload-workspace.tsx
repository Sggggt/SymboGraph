"use client";

import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import type { CourseFileStatus, CourseFileSummary } from "@course-kg/shared";
import { AlertCircle, CheckCircle2, Clock3, Database, FileCheck2, Files, LoaderCircle, Network, PanelRightOpen, RefreshCcw, Trash2, UploadCloud, X } from "lucide-react";

import { cleanupStaleData, cleanupStaleGraph, createBatchLogToken, fetchBatchStatus, fetchCourseFiles, fetchDashboard, getBatchLogUrl, parseUploadedFiles, removeCourseFile, uploadFile } from "@/lib/api";
import { useCourseContext } from "@/components/course-context";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

type UploadedFile = {
  name: string;
  path: string;
};

type IngestionLogEvent = {
  timestamp: string;
  event: string;
  message: string;
  source_path?: string;
  state?: string;
  processed_files?: number;
  total_files?: number;
  success_count?: number;
  failure_count?: number;
  skipped_count?: number;
  error?: string;
  provider?: string;
  model?: string;
  external_called?: boolean;
  fallback_reason?: string | null;
  vector_count?: number;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_external_called?: boolean;
  embedding_fallback_reason?: string | null;
  embedding_fallback_method?: string | null;
  graph_extraction_provider?: string;
  graph_extraction_model?: string;
  retry_count?: number;
  max_retries?: number;
};

const terminalLogEvents = new Set(["batch_completed", "batch_failed", "batch_partial_failed", "batch_skipped", "batch_missing"]);
const failureLogEvents = new Set(["batch_failed", "batch_partial_failed", "graph_failed"]);
const terminalBatchStates = new Set(["completed", "partial_failed", "failed", "skipped"]);
const failureBatchStates = new Set(["partial_failed", "failed"]);
const logStreamMaxRetries = 3;
const logStreamRetryDelayMs = 1200;

const logEventLabels: Record<string, string> = {
  batch_started: "批次开始",
  batch_files: "文件扫描",
  file_started: "开始解析",
  job_state: "任务状态",
  file_skipped: "跳过文件",
  file_completed: "文件完成",
  file_failed: "文件失败",
  batch_graph_started: "图谱生成",
  batch_graph_selected: "图谱片段选择",
  batch_graph_progress: "图谱抽取进度",
  graph_rebuilt: "图谱完成",
  graph_failed: "图谱失败",
  batch_completed: "批次完成",
  batch_partial_failed: "部分失败",
  batch_failed: "批次失败",
  batch_skipped: "批次跳过",
  batch_missing: "批次丢失",
  log_stream_retry: "日志重连",
};

function logEventLabel(event: string): string {
  return logEventLabels[event] ?? event.replaceAll("_", " ");
}

const batchStateLabels: Record<string, string> = {
  queued: "排队中",
  parsing: "解析中",
  chunking: "切块中",
  embedding: "向量化中",
  extracting_graph: "生成图谱中",
  completed: "已完成",
  partial_failed: "部分失败",
  failed: "失败",
  skipped: "已跳过",
};

function batchStateLabel(state?: string | null): string {
  return state ? batchStateLabels[state] ?? state : "未启动";
}

function fallbackMethodLabel(value?: string | null): string {
  if (!value) {
    return "本地模拟";
  }
  if (value === "deterministic_local_hash_embedding") {
    return "本地确定性哈希向量";
  }
  return value;
}

function isBatchNotFoundError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  return error.message.includes("Batch not found") || error.message.includes("Request failed with 404");
}

function formatBatchFailureDetails(errors?: Array<{ source_path?: string | null; message?: string | null }>): string | null {
  if (!errors || errors.length === 0) {
    return null;
  }
  return errors
    .slice(0, 5)
    .map((item) => `${item.source_path ?? "unknown"}: ${item.message ?? "未返回错误信息"}`)
    .join("\n");
}

type FileBrowserItem = CourseFileSummary & {
  localOnly?: boolean;
};

const fileStatusMeta: Record<CourseFileStatus, { label: string; className: string }> = {
  pending: { label: "待解析", className: "border-amber-200/24 bg-amber-300/10 text-amber-100" },
  parsed: { label: "已解析", className: "border-emerald-200/24 bg-emerald-300/10 text-emerald-100" },
  parsing: { label: "解析中", className: "border-cyan-200/28 bg-cyan-300/10 text-cyan-100" },
  failed: { label: "解析失败", className: "border-rose-200/28 bg-rose-300/10 text-rose-100" },
  skipped: { label: "已跳过", className: "border-white/14 bg-white/[0.05] text-white/58" },
};

function FileStatusBadge({ status }: { status: CourseFileStatus }) {
  const meta = fileStatusMeta[status];
  const Icon = status === "parsed" ? CheckCircle2 : status === "failed" ? AlertCircle : status === "parsing" ? LoaderCircle : Clock3;
  return (
    <span className={cn("inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs", meta.className, status === "parsing" && "animate-pulse")}>
      <Icon className={cn("size-3.5", status === "parsing" && "animate-spin")} />
      {meta.label}
    </span>
  );
}

const fileProgressMeta: Record<CourseFileStatus, { value: number; barClassName: string; pulse?: boolean }> = {
  pending: { value: 8, barClassName: "bg-amber-200/60" },
  parsing: { value: 58, barClassName: "bg-[linear-gradient(90deg,#64dfff,#7b7cff,#64dfff)]", pulse: true },
  parsed: { value: 100, barClassName: "bg-emerald-300/80" },
  failed: { value: 100, barClassName: "bg-rose-300/72" },
  skipped: { value: 100, barClassName: "bg-white/28" },
};

function FileProgressBar({ status }: { status: CourseFileStatus }) {
  const meta = fileProgressMeta[status];
  return (
    <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/7">
      <div
        className={cn("h-full rounded-full transition-[width] duration-500", meta.barClassName, meta.pulse && "animate-pulse")}
        style={{ width: `${meta.value}%` }}
      />
    </div>
  );
}

function fileNameFromPath(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

function UploadWorkspaceContent({ selectedCourseId }: { selectedCourseId: string | null }) {
  const queryClient = useQueryClient();
  const storageScope = selectedCourseId ?? "unassigned";
  const [batchId, setBatchId] = useLocalStorage<string | null>(`upload.batchId.${storageScope}`, null);
  const [dismissedBatchId, setDismissedBatchId] = useLocalStorage<string | null>(`upload.dismissedBatchId.${storageScope}`, null);
  const [uploadProgress, setUploadProgress] = useState({ completed: 0, total: 0 });
  const [uploadedFiles, setUploadedFiles] = useLocalStorage<UploadedFile[]>(`upload.uploadedFiles.${storageScope}`, []);
  const [activeLogBatchId, setActiveLogBatchId] = useLocalStorage<string | null>(`upload.activeLogBatchId.${storageScope}`, null);
  const [logOpen, setLogOpen] = useState(false);
  const [logs, setLogs] = useLocalStorage<IngestionLogEvent[]>(`upload.logs.${storageScope}`, []);
  const [selectedFilePathValues, setSelectedFilePathValues] = useLocalStorage<string[]>(`upload.selectedFilePaths.${storageScope}`, []);
  const [cleanupMessage, setCleanupMessage] = useState<string | null>(null);
  const [cleanupDialog, setCleanupDialog] = useState<"data" | "graph" | null>(null);
  const [failureDialog, setFailureDialog] = useState<{ title: string; message: string; details?: string | null } | null>(null);
  const [logStreamRetryCount, setLogStreamRetryCount] = useState(0);

  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedCourseId],
    queryFn: () => fetchDashboard(selectedCourseId),
    enabled: Boolean(selectedCourseId),
  });
  const activeBatchCandidate = batchId ?? dashboardQuery.data?.batch_status?.batch_id ?? null;
  const activeBatchId = activeBatchCandidate && activeBatchCandidate !== dismissedBatchId ? activeBatchCandidate : null;
  const batchQuery = useQuery({
    queryKey: ["batch", selectedCourseId, activeBatchId],
    queryFn: () => fetchBatchStatus(activeBatchId as string),
    enabled: Boolean(activeBatchId),
    retry: (failureCount, error) => !isBatchNotFoundError(error) && failureCount < 2,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state && terminalBatchStates.has(state) ? false : 3000;
    },
  });
  const courseFilesQuery = useQuery({
    queryKey: ["course-files", selectedCourseId],
    queryFn: () => fetchCourseFiles(selectedCourseId),
    enabled: Boolean(selectedCourseId),
    refetchInterval: () => (activeBatchId && !terminalBatchStates.has(batchQuery.data?.state ?? "") ? 3000 : false),
  });
  const visibleBatch = batchQuery.data && !terminalBatchStates.has(batchQuery.data.state) ? batchQuery.data : null;
  const isGraphBuilding = visibleBatch?.state === "extracting_graph";
  const remoteParseablePaths = useMemo(
    () => (courseFilesQuery.data ?? []).filter((file) => file.status !== "parsing").map((file) => file.source_path),
    [courseFilesQuery.data],
  );
  const parseTargetPaths = useMemo(() => {
    return Array.from(new Set([...uploadedFiles.map((file) => file.path), ...remoteParseablePaths]));
  }, [remoteParseablePaths, uploadedFiles]);
  const selectedFilePaths = useMemo(() => new Set(selectedFilePathValues), [selectedFilePathValues]);
  const selectedParseTargetPaths = useMemo(
    () => parseTargetPaths.filter((path) => selectedFilePaths.has(path)),
    [parseTargetPaths, selectedFilePaths],
  );
  const effectiveParseTargetPaths = selectedParseTargetPaths.length > 0 ? selectedParseTargetPaths : parseTargetPaths;
  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      setUploadProgress({ completed: 0, total: files.length });
      const responses = await Promise.all(
        files.map(async (file) => {
          const response = await uploadFile(file, selectedCourseId);
          setUploadProgress((progress) => ({ ...progress, completed: progress.completed + 1 }));
          return response;
        }),
      );
      return responses;
    },
    onSuccess: (data) => {
      setUploadedFiles((current) => [
        ...current,
        ...data.map((item) => ({
          name: fileNameFromPath(item.source_path),
          path: item.source_path,
        })),
      ]);
      void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
    },
    onSettled: () => {
      setUploadProgress({ completed: 0, total: 0 });
    },
  });

  const parseUploadsMutation = useMutation({
    mutationFn: ({ paths, force }: { paths: string[]; force: boolean }) => parseUploadedFiles(paths, selectedCourseId, force),
    onSuccess: (data) => {
      setBatchId(data.batch_id);
      setDismissedBatchId(null);
      setActiveLogBatchId(data.batch_id);
      setLogs([]);
      setLogOpen(true);
      setUploadedFiles([]);
      setSelectedFilePathValues([]);
      void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
    },
    onError: (error) => {
      setFailureDialog({
        title: "解析启动失败",
        message: error instanceof Error ? error.message : "解析任务启动失败，后端未返回错误详情。",
      });
    },
  });

  const removeFileMutation = useMutation({
    mutationFn: (sourcePath: string) => removeCourseFile(sourcePath, selectedCourseId),
    onSuccess: (_data, sourcePath) => {
      setUploadedFiles((current) => current.filter((file) => file.path !== sourcePath));
      void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
      if (activeBatchId) {
        void queryClient.invalidateQueries({ queryKey: ["batch", selectedCourseId, activeBatchId] });
      }
    },
  });
  const cleanupStaleDataMutation = useMutation({
    mutationFn: () => cleanupStaleData(selectedCourseId),
    onSuccess: (stats) => {
      setCleanupMessage(
        `旧数据清理完成：向量 ${stats.deleted_vectors}，片段 ${stats.deleted_chunks}，版本 ${stats.deleted_document_versions}，文档 ${stats.deleted_documents}，图谱关系 ${stats.removed_graph_relations}，图谱概念 ${stats.removed_graph_concepts}`,
      );
      void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["graph", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["concepts", selectedCourseId] });
    },
  });
  const cleanupStaleGraphMutation = useMutation({
    mutationFn: () => cleanupStaleGraph(selectedCourseId),
    onSuccess: (stats) => {
      setCleanupMessage(`旧图谱清理完成：关系 ${stats.removed_relations}，别名 ${stats.removed_aliases}，概念 ${stats.removed_concepts}`);
      void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["graph", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["concepts", selectedCourseId] });
    },
  });

  const uploadPercent = uploadProgress.total > 0 ? (uploadProgress.completed / uploadProgress.total) * 100 : 0;
  const modelAudit = useMemo(() => {
    const latestEmbeddingAudit = [...logs].reverse().find((item) => item.event === "embedding_audit");
    const initialAudit = logs.find((item) => item.event === "model_audit");
    const audit = latestEmbeddingAudit ?? initialAudit;
    if (!audit) {
      return "模型：等待后端返回模型审计";
    }
    const provider = audit.provider ?? audit.embedding_provider ?? "未知";
    const embeddingModel = audit.model ?? audit.embedding_model ?? provider;
    const externalCalled = audit.external_called ?? audit.embedding_external_called ?? false;
    const fallbackReason = audit.fallback_reason ?? audit.embedding_fallback_reason ?? null;
    const fallbackMethod = audit.embedding_fallback_method ?? (provider === "fake" ? "deterministic_local_hash_embedding" : null);
    const graphProvider = audit.graph_extraction_provider;
    const graphModel = audit.graph_extraction_model ?? graphProvider;
    const embeddingText =
      provider === "fake"
        ? `向量模型 ${embeddingModel}（${provider} 降级：${fallbackMethodLabel(fallbackMethod ?? fallbackReason)}）`
        : `向量模型 ${embeddingModel}（${provider}${externalCalled ? "，已调用外部 API" : ""}）`;
    const graphText = graphProvider ? `；图谱抽取 ${graphModel}（${graphProvider}）` : "";
    return `模型：${embeddingText}${graphText}`;
  }, [logs]);
  const fileItems = useMemo<FileBrowserItem[]>(() => {
    const remoteFiles = courseFilesQuery.data ?? [];
    const remotePaths = new Set(remoteFiles.map((file) => file.source_path));
    const pendingUploads = uploadedFiles
      .filter((file) => !remotePaths.has(file.path))
      .map<FileBrowserItem>((file) => ({
        id: `pending:${file.path}`,
        document_id: null,
        title: file.name,
        source_path: file.path,
                source_type: "未知",
        chapter: null,
        status: "pending",
        job_state: null,
        batch_id: null,
        error: null,
        chunk_count: 0,
        updated_at: null,
        localOnly: true,
      }));
    return [...pendingUploads, ...remoteFiles];
  }, [courseFilesQuery.data, uploadedFiles]);

  const handleRemoveFile = (file: FileBrowserItem) => {
    if (file.localOnly) {
      setUploadedFiles((current) => current.filter((item) => item.path !== file.source_path));
      return;
    }
    removeFileMutation.mutate(file.source_path);
  };

  const handleFileRowClick = (event: MouseEvent<HTMLDivElement>, file: FileBrowserItem) => {
    if (!event.shiftKey || file.status === "parsing") {
      return;
    }
    event.preventDefault();
    setSelectedFilePathValues((current) => (current.includes(file.source_path) ? current.filter((path) => path !== file.source_path) : [...current, file.source_path]));
  };

  useEffect(() => {
    queueMicrotask(() => {
      setSelectedFilePathValues((current) => {
        const validPaths = new Set(parseTargetPaths);
        const next = current.filter((path) => validPaths.has(path));
        return next.length === current.length ? current : next;
      });
    });
  }, [parseTargetPaths]);

  useEffect(() => {
    if (!activeBatchId || terminalBatchStates.has(batchQuery.data?.state ?? "")) {
      return;
    }
    queueMicrotask(() => {
      setActiveLogBatchId((current) => {
        if (current === activeBatchId) {
          return current;
        }
        setLogs([]);
        return activeBatchId;
      });
    });
  }, [activeBatchId, batchQuery.data?.state]);

  useEffect(() => {
    if (!activeLogBatchId) {
      queueMicrotask(() => setLogStreamRetryCount(0));
      return;
    }
    const streamBatchId = activeLogBatchId;
    let closed = false;
    let retryCount = 0;
    let source: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const appendLog = (item: IngestionLogEvent) => {
      setLogs((current) => [...current, item].slice(-300));
    };

    const closeSource = () => {
      source?.close();
      source = null;
    };

    const connect = async () => {
      if (closed) {
        return;
      }
      closeSource();
      let token: string;
      try {
        token = (await createBatchLogToken(streamBatchId)).token;
      } catch (error) {
        if (closed) {
          return;
        }
        retryCount += 1;
        setLogStreamRetryCount(retryCount);
        appendLog({
          timestamp: new Date().toISOString(),
          event: "log_stream_retry",
          message:
            retryCount <= logStreamMaxRetries
              ? `日志流授权失败，正在第 ${retryCount}/${logStreamMaxRetries} 次重试。`
              : `日志流授权失败，已重试 ${logStreamMaxRetries} 次仍未恢复。`,
          retry_count: retryCount,
          max_retries: logStreamMaxRetries,
          error: error instanceof Error ? error.message : String(error),
        });
        if (retryCount <= logStreamMaxRetries) {
          retryTimer = setTimeout(() => {
            void connect();
          }, logStreamRetryDelayMs * retryCount);
          return;
        }
        setFailureDialog({
          title: "解析日志流授权失败",
          message: "日志流 token 创建失败，无法继续同步解析进度。请检查 API 鉴权或后端日志。",
          details: error instanceof Error ? error.message : String(error),
        });
        setActiveLogBatchId(null);
        return;
      }
      source = new EventSource(getBatchLogUrl(streamBatchId, token));
      source.onmessage = (event) => {
        if (closed) {
          return;
        }
        retryCount = 0;
        setLogStreamRetryCount(0);
        const item = JSON.parse(event.data) as IngestionLogEvent;
        appendLog(item);
        if (failureLogEvents.has(item.event)) {
          setFailureDialog({
            title: item.event === "graph_failed" ? "图谱重建失败" : "解析失败",
            message: item.message || "任务失败，后端未返回错误详情。",
            details: item.error ?? null,
          });
        }
        if (terminalLogEvents.has(item.event)) {
          closeSource();
          setActiveLogBatchId(null);
          if (item.event === "batch_missing") {
            setBatchId(null);
            setDismissedBatchId(streamBatchId);
          }
          void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
          void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
          void queryClient.invalidateQueries({ queryKey: ["batch", selectedCourseId, streamBatchId] });
        }
      };
      source.onerror = () => {
        closeSource();
        if (closed) {
          return;
        }
        retryCount += 1;
        setLogStreamRetryCount(retryCount);
        appendLog({
          timestamp: new Date().toISOString(),
          event: "log_stream_retry",
          message:
            retryCount <= logStreamMaxRetries
              ? `日志流断开，正在第 ${retryCount}/${logStreamMaxRetries} 次重连。`
              : `日志流断开，已重试 ${logStreamMaxRetries} 次仍未恢复。`,
          retry_count: retryCount,
          max_retries: logStreamMaxRetries,
        });
        if (retryCount <= logStreamMaxRetries) {
          retryTimer = setTimeout(() => {
            void connect();
          }, logStreamRetryDelayMs * retryCount);
          return;
        }
        setFailureDialog({
          title: "解析日志流中断",
          message: "日志流重连失败，任务状态可能仍在后端继续执行。请刷新批次状态或重新打开日志查看最新结果。",
        });
        setActiveLogBatchId(null);
        void queryClient.invalidateQueries({ queryKey: ["batch", selectedCourseId, streamBatchId] });
      };
    };
    void connect();
    return () => {
      closed = true;
      if (retryTimer) {
        clearTimeout(retryTimer);
      }
      closeSource();
    };
  }, [activeLogBatchId, queryClient, selectedCourseId]);

  useEffect(() => {
    if (batchQuery.data?.state && terminalBatchStates.has(batchQuery.data.state)) {
      const terminalBatch = batchQuery.data;
      queueMicrotask(() => {
        if (failureBatchStates.has(terminalBatch.state)) {
          setFailureDialog({
            title: terminalBatch.state === "partial_failed" ? "解析部分失败" : "解析失败",
            message: `${batchStateLabel(terminalBatch.state)}：成功 ${terminalBatch.success_count}，失败 ${terminalBatch.failure_count}，跳过 ${terminalBatch.skipped_count}。`,
            details: formatBatchFailureDetails(terminalBatch.errors),
          });
        }
        setBatchId(null);
        setDismissedBatchId(activeBatchId);
      });
      void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
      void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
    }
  }, [activeBatchId, batchQuery.data?.state, queryClient, selectedCourseId]);

  useEffect(() => {
    if (!activeBatchId || !isBatchNotFoundError(batchQuery.error)) {
      return;
    }
    queueMicrotask(() => {
      setBatchId(null);
      setDismissedBatchId(activeBatchId);
      setActiveLogBatchId((current) => (current === activeBatchId ? null : current));
      setLogs((current) => [
        ...current,
        {
          timestamp: new Date().toISOString(),
          event: "batch_missing",
          message: "旧批次日志已清理，已停止同步该批次。",
          state: "missing",
        },
      ].slice(-300));
    });
    void queryClient.invalidateQueries({ queryKey: ["course-files", selectedCourseId] });
    void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
  }, [activeBatchId, batchQuery.error, queryClient, selectedCourseId]);

  const inclusionRules = useMemo(
    () => [
      "纳入：PDF / ipynb / Markdown / TXT / DOCX / PPTX / 图片 OCR",
      "排除：output / tmp / scripts / .ipynb_checkpoints / LaTeX 中间文件 / xlsx / html 派生文件",
      "去重：同一路径同一 checksum 直接跳过，变更则新建 document version",
    ],
    [],
  );
  const cleanupPending = cleanupStaleDataMutation.isPending || cleanupStaleGraphMutation.isPending;
  const cleanupError = (cleanupStaleDataMutation.error ?? cleanupStaleGraphMutation.error) as Error | null;
  const cleanupTitle = cleanupDialog === "data" ? "清理数据库" : "清理图谱";
  const cleanupDescription =
    cleanupDialog === "data"
      ? "清理当前资料库的 inactive 数据库记录和 Qdrant stale 向量，当前有效数据会保留。"
      : "清理当前资料库的陈旧图谱关系和孤立概念，不会重建图谱。";

  if (dashboardQuery.isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (dashboardQuery.error) {
    return <ErrorBlock message={(dashboardQuery.error as Error).message} />;
  }

  return (
    <div className="kg-page">
      <section className="glass-panel relative grid min-h-[calc(100dvh-8rem)] overflow-hidden rounded-[34px] xl:h-[calc(100dvh-8rem)] xl:min-h-0 xl:grid-cols-[minmax(280px,0.82fr)_minmax(420px,1.32fr)_minmax(320px,0.92fr)]">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_18%_0%,rgba(86,217,255,0.12),transparent_32%),radial-gradient(circle_at_88%_20%,rgba(34,197,94,0.08),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.035),transparent_34%)]" />
        <div className="relative border-b border-white/8 p-6 xl:border-b-0 xl:border-r xl:p-7">
        <div className="grid gap-6">
          <div className="space-y-5">
            <p className="section-kicker">导入控制台</p>
            <h2 className="glow-text text-4xl font-semibold text-white lg:text-5xl">全量导入控制台</h2>
            <p className="max-w-2xl text-base leading-8 text-cyan-50/72">
              文件上传后会进入当前资料库存储文件夹。文件导览中的任意入库文件都可以直接解析、切块、向量化并更新图谱。
            </p>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => parseUploadsMutation.mutate({ paths: effectiveParseTargetPaths, force: false })}
                disabled={parseUploadsMutation.isPending || effectiveParseTargetPaths.length === 0}
                className="rounded-full border border-emerald-300/35 bg-emerald-300/10 px-5 py-3 text-sm uppercase tracking-[0.24em] text-white disabled:opacity-50"
              >
                {parseUploadsMutation.isPending ? <LoaderCircle className="mr-2 inline size-4 animate-spin" /> : <FileCheck2 className="mr-2 inline size-4" />}
                {parseUploadsMutation.isPending ? "解析中" : "解析文件"}
              </button>
              <button
                type="button"
                onClick={() => parseUploadsMutation.mutate({ paths: parseTargetPaths, force: true })}
                disabled={parseUploadsMutation.isPending || parseTargetPaths.length === 0}
                className="rounded-full border border-rose-300/30 bg-rose-300/8 px-4 py-3 text-xs uppercase tracking-[0.2em] text-rose-50/80 transition hover:text-white disabled:opacity-45"
                  title="强制重建当前资料库所有文件的片段、向量、Qdrant 向量记录和图谱"
              >
                {parseUploadsMutation.isPending ? <LoaderCircle className="mr-2 inline size-3.5 animate-spin" /> : <RefreshCcw className="mr-2 inline size-3.5" />}
                全量重新解析
              </button>
              <label
                aria-disabled={uploadMutation.isPending}
                className={`cursor-pointer rounded-full border border-white/12 px-5 py-3 text-sm uppercase tracking-[0.24em] text-white/72 transition hover:text-white ${
                  uploadMutation.isPending ? "pointer-events-none opacity-65" : ""
                }`}
              >
                {uploadMutation.isPending ? <LoaderCircle className="mr-2 inline size-4 animate-spin" /> : <UploadCloud className="mr-2 inline size-4" />}
                {uploadMutation.isPending ? `上传中 ${uploadProgress.completed}/${uploadProgress.total}` : "上传文件"}
                <input
                  type="file"
                  multiple
                  disabled={uploadMutation.isPending}
                  className="hidden"
                  onChange={(event) => {
                    const files = Array.from(event.target.files ?? []);
                    event.target.value = "";
                    if (files.length > 0) {
                      uploadMutation.mutate(files);
                    }
                  }}
                />
              </label>
              <button
                type="button"
                onClick={() => {
                  if (activeBatchId) {
                    setActiveLogBatchId(activeBatchId);
                    setLogOpen(true);
                  }
                }}
                disabled={!activeBatchId}
                className="rounded-full border border-white/12 px-4 py-3 text-white/72 transition hover:text-white disabled:opacity-40"
              >
                <PanelRightOpen className="size-4" />
              </button>
            </div>
            {selectedParseTargetPaths.length > 0 ? (
              <p className="text-xs uppercase tracking-[0.2em] text-cyan-50/58">
                已选择 {selectedParseTargetPaths.length} 个文件；点击解析文件只处理选中文件。再次 Shift + 左键点击可取消选择。
              </p>
            ) : (
              <p className="text-xs uppercase tracking-[0.2em] text-white/36">按住 Shift 并左键点击文件可多选；未选择时解析按钮按原逻辑处理待解析/变更文件。</p>
            )}
            {cleanupMessage ? <p className="text-xs leading-5 text-emerald-100/72">{cleanupMessage}</p> : null}
            {cleanupStaleDataMutation.error || cleanupStaleGraphMutation.error ? (
              <p className="text-xs leading-5 text-rose-100/72">{((cleanupStaleDataMutation.error ?? cleanupStaleGraphMutation.error) as Error).message}</p>
            ) : null}
            {uploadMutation.isPending ? (
              <div className="max-w-md">
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-white/45">
                  <span>上传进度</span>
                  <span>{Math.round(uploadPercent)}%</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8">
                  <div className="h-full rounded-full bg-[linear-gradient(90deg,#64dfff,#7b7cff)] transition-[width] duration-300" style={{ width: `${uploadPercent}%` }} />
                </div>
              </div>
            ) : null}
            {uploadedFiles.length > 0 ? (
              <div className="max-w-2xl border-l border-cyan-200/20 bg-cyan-300/[0.035] py-3 pl-4 pr-2">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs uppercase tracking-[0.24em] text-white/45">已上传，待解析</p>
                  <button type="button" className="text-xs uppercase tracking-[0.2em] text-white/48 hover:text-white" onClick={() => setUploadedFiles([])}>
                    清空
                  </button>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {uploadedFiles.map((file) => (
                    <span key={file.path} className="max-w-full truncate rounded-full border border-white/10 px-3 py-1 text-sm text-cyan-50/72">
                      {file.name}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
            {[
              {
                label: "最新批次",
                value: visibleBatch?.state ?? "idle",
                hint: visibleBatch ? `${visibleBatch.processed_files}/${visibleBatch.total_files}` : "等待启动",
              },
              {
                label: "最近上传",
                value: uploadedFiles.length > 0 ? String(uploadedFiles.length) : "idle",
                hint: uploadedFiles.length > 0 ? "等待解析" : "暂无待解析上传",
              },
              {
                label: "已入库文档",
                value: String(dashboardQuery.data?.ingested_document_count ?? 0),
                hint: "当前资料库有效版本",
              },
              {
                label: "图谱关系",
                value: String(dashboardQuery.data?.graph_relation_count ?? 0),
                hint: "概念关系边总数",
              },
            ].map((item) => (
              <div key={item.label} className="border-l border-white/10 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.3em] text-white/45">{item.label}</p>
                <p className="mt-3 text-3xl font-semibold text-white">{item.value}</p>
                <p className="mt-2 text-sm text-white/55">{item.hint}</p>
              </div>
            ))}
          </div>
        </div>
        </div>

        <div className="relative flex min-h-[540px] max-h-[72dvh] flex-col border-b border-white/8 p-6 xl:h-full xl:min-h-0 xl:max-h-none xl:border-b-0 xl:border-r xl:p-7">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="section-kicker">文件库</p>
            <h3 className="mt-2 text-2xl font-semibold text-white">已入库文件导览</h3>
            <p className="mt-2 text-sm text-white/50">当前资料库存储文件夹中的文件会统一显示在这里。</p>
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setCleanupMessage(null);
                cleanupStaleDataMutation.reset();
                cleanupStaleGraphMutation.reset();
                setCleanupDialog("data");
              }}
              disabled={!selectedCourseId || Boolean(activeBatchId) || cleanupStaleDataMutation.isPending || cleanupStaleGraphMutation.isPending}
              className="inline-flex items-center gap-1.5 rounded-full border border-amber-200/18 bg-amber-300/[0.055] px-3 py-1.5 text-[11px] text-amber-50/72 transition hover:border-amber-200/36 hover:text-white disabled:pointer-events-none disabled:opacity-40"
                title={activeBatchId ? "当前有导入批次运行，暂不能清理" : "清理非活跃数据和失效向量"}
            >
              {cleanupStaleDataMutation.isPending ? <LoaderCircle className="size-3.5 animate-spin" /> : <Database className="size-3.5" />}
              清理数据库
            </button>
            <button
              type="button"
              onClick={() => {
                setCleanupMessage(null);
                cleanupStaleDataMutation.reset();
                cleanupStaleGraphMutation.reset();
                setCleanupDialog("graph");
              }}
              disabled={!selectedCourseId || Boolean(activeBatchId) || cleanupStaleDataMutation.isPending || cleanupStaleGraphMutation.isPending}
              className="inline-flex items-center gap-1.5 rounded-full border border-cyan-200/18 bg-cyan-300/[0.055] px-3 py-1.5 text-[11px] text-cyan-50/72 transition hover:border-cyan-200/36 hover:text-white disabled:pointer-events-none disabled:opacity-40"
                title={activeBatchId ? "当前有导入批次运行，暂不能清理" : "清理失效图谱关系"}
            >
              {cleanupStaleGraphMutation.isPending ? <LoaderCircle className="size-3.5 animate-spin" /> : <Network className="size-3.5" />}
              清理图谱
            </button>
            <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">{fileItems.length} 个文件</span>
          </div>
        </div>

        <div className="custom-scrollbar kg-rounded-scrollbar mt-5 min-h-[18rem] flex-1 overflow-y-auto overscroll-contain rounded-[24px] border border-white/8 bg-black/10 pr-1">
          {courseFilesQuery.isLoading && fileItems.length === 0 ? (
            <div className="kg-shimmer px-5 py-8 text-sm text-white/50">正在加载文件...</div>
          ) : fileItems.length === 0 ? (
            <div className="px-5 py-8 text-sm text-white/50">暂无文件。上传文件后会先显示为待解析。</div>
          ) : (
            fileItems.map((file) => {
              const isSelected = selectedFilePaths.has(file.source_path);
              return (
              <div
                key={`${file.id}-${file.source_path}`}
                onClick={(event) => handleFileRowClick(event, file)}
                className={cn(
                  "border-b border-white/7 px-4 py-4 last:border-b-0 transition hover:bg-white/[0.035]",
                  file.status !== "parsing" && "cursor-default",
                  isSelected && "bg-cyan-300/[0.08] ring-1 ring-inset ring-cyan-200/35",
                )}
                title="按住 Shift 并左键点击可多选文件"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="max-w-full truncate text-sm font-medium text-white">{file.title || fileNameFromPath(file.source_path)}</p>
                      <FileStatusBadge status={file.status} />
                    </div>
                    <p className="mt-2 break-all text-xs leading-5 text-white/42">{file.source_path}</p>
                    {file.error ? <p className="mt-2 break-words text-xs leading-5 text-rose-100/70">{file.error}</p> : null}
                    <FileProgressBar status={file.status} />
                  </div>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      handleRemoveFile(file);
                    }}
                    disabled={file.status === "parsing" || removeFileMutation.isPending}
                    className="inline-flex shrink-0 items-center gap-2 rounded-full border border-white/10 px-3 py-2 text-xs text-white/62 transition hover:border-rose-200/35 hover:text-rose-100 disabled:pointer-events-none disabled:opacity-40"
                    title={file.status === "parsing" ? "解析中，暂不能移除" : "移除文件"}
                  >
                    <Trash2 className="size-3.5" />
                    移除
                  </button>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-white/42">
                  <span className="rounded-full border border-white/8 px-2.5 py-1">{file.source_type || "未知"}</span>
                  {file.chapter ? <span className="rounded-full border border-white/8 px-2.5 py-1">{file.chapter}</span> : null}
                  <span className="rounded-full border border-white/8 px-2.5 py-1">{file.chunk_count} 个片段</span>
                  {isSelected ? <span className="rounded-full border border-cyan-200/30 bg-cyan-300/10 px-2.5 py-1 text-cyan-50">已选择</span> : null}
                </div>
              </div>
              );
            })
          )}
        </div>
        </div>

        <div className="relative min-h-0">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="border-b border-white/8 p-6 xl:p-7">
          <div className="flex items-center justify-between">
            <div>
              <p className="section-kicker">规则</p>
              <h3 className="mt-2 text-2xl font-semibold text-white">纳入与排除策略</h3>
            </div>
            <Files className="size-5 text-cyan-200" />
          </div>
          <div className="mt-6 space-y-3">
            {inclusionRules.map((rule) => (
              <div key={rule} className="border-l border-white/10 px-4 py-3 text-sm leading-7 text-white/70">
                {rule}
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="p-6 xl:p-7">
          <div className="flex items-center justify-between">
            <div>
              <p className="section-kicker">批次状态</p>
              <h3 className="mt-2 text-2xl font-semibold text-white">当前后台批次</h3>
            </div>
            <button
              type="button"
              className="rounded-full border border-white/10 p-2 text-white/65 transition hover:text-white"
              onClick={() => {
                void queryClient.invalidateQueries({ queryKey: ["dashboard", selectedCourseId] });
                if (activeBatchId) {
                  void queryClient.invalidateQueries({ queryKey: ["batch", selectedCourseId, activeBatchId] });
                }
              }}
            >
              <RefreshCcw className="size-4" />
            </button>
          </div>

          <div className="mt-6 space-y-5">
            {isGraphBuilding ? (
              <div className="relative overflow-hidden rounded-[22px] border border-cyan-200/22 bg-cyan-300/[0.055] p-5">
                <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent,rgba(103,232,249,0.12),transparent)] animate-pulse" />
                <div className="relative flex items-start gap-4">
                  <div className="grid size-12 shrink-0 place-items-center rounded-full border border-cyan-100/20 bg-cyan-200/10">
                    <LoaderCircle className="size-6 animate-spin text-cyan-100" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-cyan-50">正在生成资料库图谱</p>
                    <p className="mt-2 text-sm leading-6 text-cyan-50/72">图谱关系抽取和绘制需要等待模型完成，请不要关闭页面、停止后端或重启服务。</p>
                    <div className="mt-4 flex items-center gap-2">
                      {[0, 1, 2, 3].map((item) => (
                        <span key={item} className="size-2 animate-pulse rounded-full bg-cyan-100/80" style={{ animationDelay: `${item * 150}ms` }} />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            <div className="border border-white/8 bg-black/10 p-5">
              <div className="flex items-center justify-between gap-4">
                <p className="text-lg font-medium text-white">{batchStateLabel(visibleBatch?.state)}</p>
                <p className="text-xs uppercase tracking-[0.26em] text-white/45">{visibleBatch ? visibleBatch.batch_id.slice(0, 8) : "无批次"}</p>
              </div>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/6">
                <div
                  className="h-full rounded-full bg-[linear-gradient(90deg,#64dfff,#7b7cff)]"
                  style={{
                    width: `${visibleBatch?.total_files ? (visibleBatch.processed_files / visibleBatch.total_files) * 100 : 0}%`,
                  }}
                />
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {[
                  { label: "成功", value: visibleBatch?.success_count ?? 0 },
                  { label: "跳过", value: visibleBatch?.skipped_count ?? 0 },
                  { label: "失败", value: visibleBatch?.failure_count ?? 0 },
                ].map((item) => (
                  <div key={item.label} className="border-l border-white/10 px-4 py-2">
                    <p className="text-xs uppercase tracking-[0.24em] text-white/45">{item.label}</p>
                    <p className="mt-2 text-2xl font-semibold text-white">{item.value}</p>
                  </div>
                ))}
              </div>
            </div>

            {(visibleBatch?.errors ?? []).length > 0 ? (
              <div className="rounded-[22px] border border-rose-300/20 bg-rose-400/[0.05] p-5">
                <p className="text-xs uppercase tracking-[0.26em] text-rose-100/70">失败项</p>
                <div className="custom-scrollbar kg-rounded-scrollbar mt-4 max-h-64 space-y-3 overflow-y-auto pr-1">
                  {visibleBatch?.errors.map((error) => (
                    <div key={`${error.source_path}-${error.message}`} className="rounded-[18px] border border-white/8 px-4 py-3 text-sm text-white/72">
                      <p className="font-medium text-white">{error.source_path}</p>
                      <p className="mt-1 text-white/58">{error.message}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </motion.div>
        </div>
      </section>

      <Dialog
        open={cleanupDialog !== null}
        onOpenChange={(open) => {
          if (!open && !cleanupPending) {
            setCleanupDialog(null);
          }
        }}
      >
        <DialogContent className="max-w-md border border-white/10 bg-[rgba(3,7,20,0.92)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.4)] backdrop-blur-2xl" showCloseButton={!cleanupPending}>
          <DialogHeader className="border-b border-white/8 px-6 py-5">
            <DialogTitle>{cleanupTitle}</DialogTitle>
            <DialogDescription>{cleanupDescription}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 px-6 py-5">
            {cleanupPending ? (
              <div>
                <p className="text-sm text-white/72">{cleanupTitle}执行中...</p>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/8">
                  <div className="h-full w-2/3 animate-pulse rounded-full bg-[linear-gradient(90deg,#64dfff,#7b7cff,#64dfff)]" />
                </div>
              </div>
            ) : cleanupMessage ? (
              <p className="rounded-2xl border border-emerald-200/16 bg-emerald-300/[0.055] px-4 py-3 text-sm leading-6 text-emerald-50/78">{cleanupMessage}</p>
            ) : (
              <p className="rounded-2xl border border-white/10 bg-white/[0.035] px-4 py-3 text-sm leading-6 text-white/68">确认后会立即执行维护操作。</p>
            )}
            {cleanupError ? <p className="text-sm text-rose-100/78">{cleanupError.message}</p> : null}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                disabled={cleanupPending}
                onClick={() => setCleanupDialog(null)}
                className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:text-white disabled:pointer-events-none disabled:opacity-45"
              >
                {cleanupMessage ? "关闭" : "取消"}
              </button>
              {!cleanupMessage ? (
                <button
                  type="button"
                  disabled={!selectedCourseId || cleanupPending}
                  onClick={() => {
                    setCleanupMessage(null);
                    if (cleanupDialog === "data") {
                      cleanupStaleDataMutation.mutate();
                    } else if (cleanupDialog === "graph") {
                      cleanupStaleGraphMutation.mutate();
                    }
                  }}
                  className="rounded-full border border-cyan-200/24 bg-cyan-300/[0.08] px-4 py-2 text-sm text-cyan-50/82 transition hover:text-white disabled:pointer-events-none disabled:opacity-45"
                >
                  {cleanupPending ? "执行中..." : "确认执行"}
                </button>
              ) : null}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={failureDialog !== null} onOpenChange={(open) => !open && setFailureDialog(null)}>
        <DialogContent className="max-w-md border border-rose-200/18 bg-[rgba(18,6,12,0.94)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.4)] backdrop-blur-2xl">
          <DialogHeader className="border-b border-rose-200/12 px-6 py-5">
            <DialogTitle>{failureDialog?.title ?? "任务失败"}</DialogTitle>
            <DialogDescription>{failureDialog?.message ?? "后端未返回错误详情。"}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 px-6 py-5">
            {failureDialog?.details ? (
              <pre className="max-h-52 overflow-auto whitespace-pre-wrap rounded-2xl border border-rose-200/12 bg-black/20 px-4 py-3 text-xs leading-5 text-rose-50/78">
                {failureDialog.details}
              </pre>
            ) : null}
            <div className="flex justify-end">
              <button type="button" onClick={() => setFailureDialog(null)} className="rounded-full border border-rose-200/20 px-4 py-2 text-sm text-rose-50/78 transition hover:text-white">
                关闭
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {logOpen ? (
        <div className="fixed inset-y-0 right-0 z-50 flex w-full justify-end bg-black/24 backdrop-blur-[2px] sm:w-auto sm:bg-transparent">
          <motion.aside
            initial={{ x: 420, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 420, opacity: 0 }}
            className="h-full w-full max-w-[420px] border-l border-white/10 bg-[rgba(3,7,20,0.94)] p-5 text-white shadow-[0_0_60px_rgba(0,0,0,0.45)]"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="section-kicker">解析日志</p>
                <h3 className="mt-2 text-xl font-semibold">导入日志流</h3>
                <p className="mt-1 text-xs text-white/45">{(activeLogBatchId ?? activeBatchId) ? (activeLogBatchId ?? activeBatchId)?.slice(0, 8) : "无批次"}</p>
              </div>
              <button type="button" onClick={() => setLogOpen(false)} className="rounded-full border border-white/10 p-2 text-white/62 transition hover:text-white">
                <X className="size-4" />
              </button>
            </div>
            <p className="mt-3 rounded-full border border-cyan-200/12 bg-cyan-300/[0.045] px-3 py-2 text-[11px] leading-5 text-cyan-50/62">
              {modelAudit}
            </p>
            {logStreamRetryCount > 0 ? (
              <p className="mt-2 rounded-full border border-amber-200/14 bg-amber-300/[0.055] px-3 py-2 text-[11px] leading-5 text-amber-50/72">
                日志流重连 {Math.min(logStreamRetryCount, logStreamMaxRetries)}/{logStreamMaxRetries}
              </p>
            ) : null}

            <div className="mt-4 h-[calc(100%-132px)] overflow-y-auto pr-1">
              {logs.length > 0 ? (
                <div className="space-y-3">
                  {logs.map((item, index) => (
                    <div key={`${item.timestamp}-${index}`} className="rounded-[18px] border border-white/8 bg-white/[0.03] px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-xs uppercase tracking-[0.2em] text-cyan-100/54">{logEventLabel(item.event)}</span>
                        <span className="text-[11px] text-white/36">{new Date(item.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <p className="mt-2 break-words text-sm leading-6 text-white/72">{item.message}</p>
                      {typeof item.processed_files === "number" && typeof item.total_files === "number" ? (
                        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/8">
                          <div
                            className="h-full rounded-full bg-[linear-gradient(90deg,#64dfff,#7b7cff)] transition-[width] duration-300"
                            style={{ width: `${item.total_files ? (item.processed_files / item.total_files) * 100 : 0}%` }}
                          />
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-[18px] border border-white/8 bg-white/[0.03] px-4 py-5 text-sm text-white/54">等待解析日志...</div>
              )}
            </div>
          </motion.aside>
        </div>
      ) : null}
    </div>
  );
}

export function UploadWorkspace() {
  const { selectedCourseId } = useCourseContext();
  return <UploadWorkspaceContent key={selectedCourseId ?? "unassigned"} selectedCourseId={selectedCourseId} />;
}
