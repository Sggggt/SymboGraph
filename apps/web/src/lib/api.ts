import type {
  AgentResponse,
  AgentTraceEventPayload,
  BatchLogTokenResponse,
  BatchStartResponse,
  CleanupStaleDataResponse,
  KnowledgeBaseFileSummary,
  KnowledgeBaseCreateRequest,
  KnowledgeBaseSummary,
  DashboardSnapshot,
  DeleteKnowledgeBaseResponse,
  DeleteResponse,
  GraphResponse,
  GraphType,
  IngestionBatchSummary,
  JobStatusResponse,
  ModelSettingsResponse,
  ModelSettingsUpdate,
  AutoTpeStatusResponse,
  ParseUploadedFilesRequest,
  QARequest,
  QAResponse,
  RebuildGraphRequest,
  RebuildGraphResponse,
  RefreshResponse,
  ContextPackageResponse,
  RetrievalTraceStepsResponse,
  RuntimeCheckResponse,
  SearchRequest,
  SearchResponse,
  SessionMessagesResponse,
  SessionSummary,
  TaskStatusResponse,
  UploadFileResponse,
  StructuredApiErrorBody,
  RetrievalGranularity,
  StrategyProfileAssistantRequest,
  StrategyProfileAssistantStateResponse,
  StrategyProfileBindRequest,
  StrategyProfileCopyRequest,
  StrategyProfileCreateRequest,
  StrategyProfileDetail,
  StrategyProfileMutationResponse,
  StrategyProfileSummary,
  StrategyProfileUpdateRequest,
} from "@course-kg/shared";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

function jsonHeaders(): HeadersInit {
  return { "Content-Type": "application/json", ...authHeaders() };
}

function buildApiUrl(path: string, params?: Record<string, string | null | undefined>): string {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value) {
        url.searchParams.set(key, value);
      }
    }
  }
  return url.toString();
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    const error = new Error(text || `请求失败，HTTP ${response.status}`) as Error & {
      status?: number;
      structured?: StructuredApiErrorBody;
    };
    error.status = response.status;
    try {
      const parsed = JSON.parse(text) as { detail?: StructuredApiErrorBody | string };
      if (parsed.detail && typeof parsed.detail === "object") {
        error.structured = parsed.detail;
        error.message = parsed.detail.message || parsed.detail.title || error.message;
      } else if (typeof parsed.detail === "string") {
        error.message = parsed.detail;
      }
    } catch {
      // Keep the plain text error.
    }
    throw error;
  }
  return response.json() as Promise<T>;
}

function extractApiErrorMessage(text: string, status: number): string {
  const fallback = text || `请求失败，HTTP ${status}`;
  try {
    const parsed = JSON.parse(text) as {
      detail?: StructuredApiErrorBody | string | Array<{ loc?: unknown[]; msg?: string; type?: string }>;
      message?: string;
      title?: string;
    };
    if (parsed.detail && typeof parsed.detail === "object" && !Array.isArray(parsed.detail)) {
      return parsed.detail.message || parsed.detail.title || fallback;
    }
    if (Array.isArray(parsed.detail)) {
      return parsed.detail
        .map((item) => {
          const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
          return [location, item.msg].filter(Boolean).join(": ") || item.type;
        })
        .filter(Boolean)
        .join("；") || fallback;
    }
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
    return parsed.message || parsed.title || fallback;
  } catch {
    return fallback;
  }
}

export async function fetchKnowledgeBases(): Promise<KnowledgeBaseSummary[]> {
  const response = await fetch(buildApiUrl("/knowledge_bases"), { cache: "no-store", headers: authHeaders() });
  return parseResponse<KnowledgeBaseSummary[]>(response);
}

export async function createKnowledgeBase(payload: KnowledgeBaseCreateRequest): Promise<KnowledgeBaseSummary> {
  const response = await fetch(buildApiUrl("/knowledge_bases"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<KnowledgeBaseSummary>(response);
}

export async function deleteKnowledgeBase(knowledgeBaseId: string): Promise<DeleteKnowledgeBaseResponse> {
  const response = await fetch(buildApiUrl(`/knowledge_bases/${encodeURIComponent(knowledgeBaseId)}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse<DeleteKnowledgeBaseResponse>(response);
}

export async function fetchDashboard(knowledgeBaseId?: string | null, options: { includeGraph?: boolean } = {}): Promise<DashboardSnapshot> {
  const response = await fetch(
    buildApiUrl("/knowledge_bases/current/dashboard", {
      knowledge_base_id: knowledgeBaseId,
      include_graph: options.includeGraph === false ? "false" : undefined,
    }),
    { cache: "no-store", headers: authHeaders() },
  );
  return parseResponse<DashboardSnapshot>(response);
}

export async function refreshKnowledgeBase(knowledgeBaseId?: string | null): Promise<RefreshResponse> {
  const response = await fetch(buildApiUrl("/knowledge_bases/current/refresh", { knowledge_base_id: knowledgeBaseId }), {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<RefreshResponse>(response);
}

export async function fetchModelSettings(): Promise<ModelSettingsResponse> {
  const response = await fetch(buildApiUrl("/settings/model"), { cache: "no-store", headers: authHeaders() });
  return parseResponse<ModelSettingsResponse>(response);
}

export async function fetchRuntimeCheck(): Promise<RuntimeCheckResponse> {
  const response = await fetch(buildApiUrl("/settings/runtime-check"), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return parseResponse<RuntimeCheckResponse>(response);
}

export async function updateModelSettings(payload: ModelSettingsUpdate): Promise<ModelSettingsResponse> {
  const response = await fetch(buildApiUrl("/settings/model"), {
    method: "PUT",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<ModelSettingsResponse>(response);
}

export async function fetchStrategyProfiles(): Promise<StrategyProfileSummary[]> {
  const response = await fetch(buildApiUrl("/settings/profiles"), { cache: "no-store", headers: authHeaders() });
  return parseResponse<StrategyProfileSummary[]>(response);
}

export async function fetchStrategyProfile(profileId: string): Promise<StrategyProfileDetail> {
  const response = await fetch(buildApiUrl(`/settings/profiles/${encodeURIComponent(profileId)}`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<StrategyProfileDetail>(response);
}

export async function createStrategyProfile(payload: StrategyProfileCreateRequest): Promise<StrategyProfileMutationResponse> {
  const response = await fetch(buildApiUrl("/settings/profiles"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<StrategyProfileMutationResponse>(response);
}

export async function updateStrategyProfile(profileId: string, payload: StrategyProfileUpdateRequest): Promise<StrategyProfileMutationResponse> {
  const response = await fetch(buildApiUrl(`/settings/profiles/${encodeURIComponent(profileId)}`), {
    method: "PUT",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<StrategyProfileMutationResponse>(response);
}

export async function copyStrategyProfile(profileId: string, payload: StrategyProfileCopyRequest): Promise<StrategyProfileMutationResponse> {
  const response = await fetch(buildApiUrl(`/settings/profiles/${encodeURIComponent(profileId)}/copy`), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<StrategyProfileMutationResponse>(response);
}

export async function deleteStrategyProfile(profileId: string): Promise<DeleteResponse> {
  const response = await fetch(buildApiUrl(`/settings/profiles/${encodeURIComponent(profileId)}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse<DeleteResponse>(response);
}

export async function bindStrategyProfile(payload: StrategyProfileBindRequest): Promise<KnowledgeBaseSummary> {
  const response = await fetch(buildApiUrl("/settings/profiles/bind"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({ knowledge_base_id: payload.knowledge_base_id, profile_id: payload.profile_id }),
  });
  return parseResponse<KnowledgeBaseSummary>(response);
}

export async function fetchProfileAssistantState(sessionId: string): Promise<StrategyProfileAssistantStateResponse> {
  const response = await fetch(buildApiUrl(`/settings/profile-assistant/${encodeURIComponent(sessionId)}`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return parseResponse<StrategyProfileAssistantStateResponse>(response);
}

export async function streamProfileAssistant(
  payload: StrategyProfileAssistantRequest,
  handlers: {
    onToken: (value: string) => void;
    onProfileJson: (value: { profile_json: Record<string, unknown>; warnings: string[]; profile_hash?: string }) => void;
    onFinal?: (value: StrategyProfileAssistantStateResponse) => void;
    onMeta?: (value: { session_id?: string; cached?: boolean }) => void;
    onError?: (value: string) => void;
  },
): Promise<void> {
  const response = await fetch(buildApiUrl("/settings/profile-assistant/stream"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const message = extractApiErrorMessage(await response.text(), response.status);
    handlers.onError?.(message);
    throw new Error(message);
  }
  if (!response.body) {
    const message = "当前浏览器不支持流式响应";
    handlers.onError?.(message);
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const line = event.replace(/^data:\s*/m, "").trim();
      if (!line || line === "[DONE]") {
        continue;
      }
      let parsed: {
        type?: string;
        token?: string;
        session_id?: string;
        cached?: boolean;
        profile_json?: Record<string, unknown>;
        warnings?: string[];
        profile_hash?: string;
        state?: StrategyProfileAssistantStateResponse;
        error?: string;
      };
      try {
        parsed = JSON.parse(line);
      } catch (error) {
        console.warn("忽略无法解析的配置档助手 SSE 事件", { line, error });
        continue;
      }
      if (parsed.type === "meta") {
        handlers.onMeta?.({ session_id: parsed.session_id, cached: parsed.cached });
      }
      if (parsed.type === "error" && parsed.error) {
        handlers.onError?.(parsed.error);
      }
      if (parsed.token) {
        handlers.onToken(parsed.token);
      }
      if (parsed.type === "profile_json" && parsed.profile_json) {
        handlers.onProfileJson({
          profile_json: parsed.profile_json,
          warnings: parsed.warnings ?? [],
          profile_hash: parsed.profile_hash,
        });
      }
      if (parsed.type === "final" && parsed.state) {
        handlers.onFinal?.(parsed.state);
      }
    }
  }
}

export async function fetchKnowledgeBaseFiles(knowledgeBaseId?: string | null): Promise<KnowledgeBaseFileSummary[]> {
  const response = await fetch(buildApiUrl("/knowledge-base-files", { knowledge_base_id: knowledgeBaseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<KnowledgeBaseFileSummary[]>(response);
}

export async function removeKnowledgeBaseFile(sourcePath: string, knowledgeBaseId?: string | null): Promise<{ removed: boolean }> {
  const response = await fetch(buildApiUrl("/knowledge-base-files", { knowledge_base_id: knowledgeBaseId, source_path: sourcePath }), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse<{ removed: boolean }>(response);
}

export async function cleanupStaleData(knowledgeBaseId?: string | null): Promise<CleanupStaleDataResponse> {
  const response = await fetch(buildApiUrl("/maintenance/cleanup-stale-data", { knowledge_base_id: knowledgeBaseId }), {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<CleanupStaleDataResponse>(response);
}

export async function rebuildGraph(
  knowledgeBaseId?: string | null,
  dryRun = false,
  options: Partial<RebuildGraphRequest> = {},
): Promise<RebuildGraphResponse> {
  const response = await fetch(buildApiUrl("/maintenance/rebuild-graph", { knowledge_base_id: knowledgeBaseId }), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({ dry_run: dryRun, ...options } satisfies RebuildGraphRequest),
  });
  return parseResponse<RebuildGraphResponse>(response);
}

export async function fetchGraph(knowledgeBaseId: string | null | undefined, graphType: GraphType, view: GraphResponse["view"] = "overview"): Promise<GraphResponse> {
  const response = await fetch(buildApiUrl("/knowledge_bases/current/graph", { knowledge_base_id: knowledgeBaseId, graph_type: graphType, view }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<GraphResponse>(response);
}

export async function fetchContextPackage(packageId: string): Promise<ContextPackageResponse> {
  const response = await fetch(buildApiUrl(`/context-packages/${encodeURIComponent(packageId)}`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<ContextPackageResponse>(response);
}

export async function fetchRetrievalTraceSteps(traceId: string): Promise<RetrievalTraceStepsResponse> {
  const response = await fetch(buildApiUrl(`/retrieval-traces/${encodeURIComponent(traceId)}/graph-steps`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<RetrievalTraceStepsResponse>(response);
}

export async function searchKnowledge(payload: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(buildApiUrl("/search/graph-enhanced"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<SearchResponse>(response);
}

export async function askQuestion(payload: QARequest): Promise<QAResponse> {
  const response = await fetch(buildApiUrl("/qa"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<QAResponse>(response);
}

export async function callAgent(payload: QARequest): Promise<AgentResponse> {
  const response = await fetch(buildApiUrl("/agent"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<AgentResponse>(response);
}

export async function uploadFile(file: File, knowledgeBaseId?: string | null, signal?: AbortSignal): Promise<UploadFileResponse> {
  const formData = new FormData();
  formData.append("upload", file);
  const response = await fetch(buildApiUrl("/files/upload", { knowledge_base_id: knowledgeBaseId }), {
    method: "POST",
    headers: authHeaders(),
    body: formData,
    signal,
  });
  return parseResponse<UploadFileResponse>(response);
}

export async function parseUploadedFiles(
  filePaths: string[],
  knowledgeBaseId?: string | null,
  force = false,
  fullReparse = false,
): Promise<BatchStartResponse> {
  const payload: ParseUploadedFilesRequest = {
    file_paths: filePaths,
    force,
    full_reparse: fullReparse,
  };
  const response = await fetch(buildApiUrl("/ingestion/parse-uploaded-files", { knowledge_base_id: knowledgeBaseId }), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<BatchStartResponse>(response);
}

export async function cancelBatch(batchId: string, knowledgeBaseId?: string | null): Promise<IngestionBatchSummary> {
  const response = await fetch(buildApiUrl(`/ingestion/batches/${batchId}/cancel`, { knowledge_base_id: knowledgeBaseId }), {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<IngestionBatchSummary>(response);
}

export async function createBatchLogToken(batchId: string): Promise<BatchLogTokenResponse> {
  const response = await fetch(buildApiUrl(`/ingestion/batches/${batchId}/log-token`), {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<BatchLogTokenResponse>(response);
}

export function getBatchLogUrl(batchId: string, token: string): string {
  return buildApiUrl(`/ingestion/batches/${batchId}/logs`, { token });
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(buildApiUrl(`/jobs/${encodeURIComponent(jobId)}`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<JobStatusResponse>(response);
}

export async function fetchBatchStatus(batchId: string): Promise<IngestionBatchSummary> {
  const response = await fetch(buildApiUrl(`/ingestion/batches/${encodeURIComponent(batchId)}`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<IngestionBatchSummary>(response);
}

export async function fetchAutoTpeStatus(knowledgeBaseId: string): Promise<AutoTpeStatusResponse> {
  const response = await fetch(buildApiUrl(`/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/graph-operating-point/auto-tpe/latest`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return parseResponse<AutoTpeStatusResponse>(response);
}

export async function fetchTaskStatus(runId: string): Promise<TaskStatusResponse> {
  const response = await fetch(buildApiUrl(`/tasks/${encodeURIComponent(runId)}`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<TaskStatusResponse>(response);
}

export async function cancelAgentRun(runId: string): Promise<TaskStatusResponse> {
  const response = await fetch(buildApiUrl(`/agent/runs/${encodeURIComponent(runId)}/cancel`), { method: "POST", headers: authHeaders() });
  return parseResponse<TaskStatusResponse>(response);
}

export async function fetchSessions(knowledgeBaseId?: string | null): Promise<SessionSummary[]> {
  const response = await fetch(buildApiUrl("/sessions", { knowledge_base_id: knowledgeBaseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<SessionSummary[]>(response);
}

export async function fetchSessionMessages(sessionId: string): Promise<SessionMessagesResponse> {
  const response = await fetch(buildApiUrl(`/sessions/${encodeURIComponent(sessionId)}/messages`), { cache: "no-store", headers: authHeaders() });
  return parseResponse<SessionMessagesResponse>(response);
}

export async function deleteSession(sessionId: string): Promise<DeleteResponse> {
  const response = await fetch(buildApiUrl(`/sessions/${encodeURIComponent(sessionId)}`), { method: "DELETE", headers: authHeaders() });
  return parseResponse<DeleteResponse>(response);
}

export async function streamAnswer(
  payload: QARequest,
  handlers: {
    onToken: (value: string) => void;
    onCitations: (value: QAResponse["citations"]) => void;
    onTrace?: (value: AgentTraceEventPayload) => void;
    onFinal?: (value: AgentResponse) => void;
    onMeta?: (value: { degraded_mode?: boolean; run_id?: string; session_id?: string; route?: string; retrieval_granularity?: RetrievalGranularity }) => void;
    onError?: (value: string) => void;
  },
  options?: { signal?: AbortSignal },
): Promise<void> {
  const response = await fetch(buildApiUrl("/qa/stream"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
    signal: options?.signal,
  });
  if (!response.ok) {
    const message = extractApiErrorMessage(await response.text(), response.status);
    handlers.onError?.(message);
    throw new Error(message);
  }
  if (!response.body) {
    const message = "浏览器不支持流式响应";
    handlers.onError?.(message);
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const line = event.replace(/^data:\s*/m, "").trim();
      if (!line || line === "[DONE]") {
        continue;
      }
      let parsed: {
        type?: string;
        token?: string;
        trace?: AgentTraceEventPayload;
        citations?: QAResponse["citations"];
        degraded_mode?: boolean;
        response?: AgentResponse;
        error?: string;
        run_id?: string;
        session_id?: string;
        route?: string;
        retrieval_granularity?: RetrievalGranularity;
      };
      try {
        parsed = JSON.parse(line);
      } catch (error) {
        console.warn("忽略无法解析的 SSE 数据行", { line, error });
        continue;
      }
      if (parsed.type === "meta") {
        handlers.onMeta?.({
          run_id: parsed.run_id,
          session_id: parsed.session_id,
          route: parsed.route,
          retrieval_granularity: parsed.retrieval_granularity,
        });
      }
      if (parsed.type === "trace" && parsed.trace) {
        handlers.onTrace?.(parsed.trace);
      }
      if (parsed.type === "error" && parsed.error) {
        handlers.onError?.(parsed.error);
      }
      if (parsed.token) {
        handlers.onToken(parsed.token);
      }
      if (parsed.citations) {
        handlers.onCitations(parsed.citations);
      }
      if (parsed.type === "final" && parsed.response) {
        const response = parsed.response;
        handlers.onFinal?.(parsed.response);
        handlers.onMeta?.({
          degraded_mode: response.degraded_mode,
          run_id: response.run_id,
          session_id: response.session_id,
          route: response.route,
          retrieval_granularity: response.retrieval_granularity,
        });
      }
      if (typeof parsed.degraded_mode === "boolean") {
        handlers.onMeta?.({ degraded_mode: parsed.degraded_mode });
      }
    }
  }
}
