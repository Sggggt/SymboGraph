import type {
  AgentResponse,
  AgentTraceEventPayload,
  BatchLogTokenResponse,
  BatchStartResponse,
  CleanupStaleDataResponse,
  CleanupStaleGraphResponse,
  ConceptCard,
  CourseFileSummary,
  CourseCreateRequest,
  CourseSummary,
  DashboardSnapshot,
  DeleteCourseResponse,
  DeleteResponse,
  GraphNodeDetail,
  GraphResponse,
  GraphType,
  IngestionBatchSummary,
  JobStatusResponse,
  ModelSettingsResponse,
  ModelSettingsUpdate,
  ParseUploadedFilesRequest,
  QARequest,
  QAResponse,
  QuerySemanticGraphRequest,
  RebuildGraphRequest,
  RebuildGraphResponse,
  RefreshResponse,
  RuntimeCheckResponse,
  SearchRequest,
  SearchResponse,
  SessionMessagesResponse,
  SessionSummary,
  TaskStatusResponse,
  UploadFileResponse,
  StructuredApiErrorBody,
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

export async function fetchCourses(): Promise<CourseSummary[]> {
  const response = await fetch(buildApiUrl("/courses"), { cache: "no-store", headers: authHeaders() });
  return parseResponse<CourseSummary[]>(response);
}

export async function createCourse(payload: CourseCreateRequest): Promise<CourseSummary> {
  const response = await fetch(buildApiUrl("/courses"), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<CourseSummary>(response);
}

export async function deleteCourse(courseId: string): Promise<DeleteCourseResponse> {
  const response = await fetch(buildApiUrl(`/courses/${encodeURIComponent(courseId)}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse<DeleteCourseResponse>(response);
}

export async function fetchDashboard(courseId?: string | null): Promise<DashboardSnapshot> {
  const response = await fetch(buildApiUrl("/courses/current/dashboard", { course_id: courseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<DashboardSnapshot>(response);
}

export async function refreshCourse(courseId?: string | null): Promise<RefreshResponse> {
  const response = await fetch(buildApiUrl("/courses/current/refresh", { course_id: courseId }), {
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

export async function fetchCourseFiles(courseId?: string | null): Promise<CourseFileSummary[]> {
  const response = await fetch(buildApiUrl("/course-files", { course_id: courseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<CourseFileSummary[]>(response);
}

export async function removeCourseFile(sourcePath: string, courseId?: string | null): Promise<{ removed: boolean }> {
  const response = await fetch(buildApiUrl("/course-files", { course_id: courseId, source_path: sourcePath }), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse<{ removed: boolean }>(response);
}

export async function cleanupStaleData(courseId?: string | null): Promise<CleanupStaleDataResponse> {
  const response = await fetch(buildApiUrl("/maintenance/cleanup-stale-data", { course_id: courseId }), {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<CleanupStaleDataResponse>(response);
}

export async function cleanupStaleGraph(courseId?: string | null): Promise<CleanupStaleGraphResponse> {
  const response = await fetch(buildApiUrl("/maintenance/cleanup-stale-graph", { course_id: courseId }), {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<CleanupStaleGraphResponse>(response);
}

export async function rebuildGraph(courseId?: string | null, mode: "incremental" | "full" = "incremental", dryRun = false): Promise<RebuildGraphResponse> {
  const response = await fetch(buildApiUrl("/maintenance/rebuild-graph", { course_id: courseId }), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({ mode, confirm_destructive: mode === "full" && !dryRun, dry_run: dryRun } satisfies RebuildGraphRequest),
  });
  return parseResponse<RebuildGraphResponse>(response);
}

export async function fetchGraph(courseId: string | null | undefined, graphType: GraphType): Promise<GraphResponse> {
  const response = await fetch(buildApiUrl("/courses/current/graph", { course_id: courseId, graph_type: graphType }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<GraphResponse>(response);
}

export async function fetchChapterGraph(chapter: string, courseId: string | null | undefined, graphType: GraphType): Promise<GraphResponse> {
  const response = await fetch(buildApiUrl(`/graph/chapters/${encodeURIComponent(chapter)}`, { course_id: courseId, graph_type: graphType }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<GraphResponse>(response);
}

export async function fetchGraphNode(conceptId: string, courseId?: string | null): Promise<GraphNodeDetail> {
  const response = await fetch(buildApiUrl(`/graph/nodes/${conceptId}`, { course_id: courseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<GraphNodeDetail>(response);
}

export async function fetchQuerySemanticGraph(payload: QuerySemanticGraphRequest): Promise<GraphResponse> {
  const response = await fetch(`${API_BASE_URL}/search/semantic-graph`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<GraphResponse>(response);
}

export async function fetchConcepts(courseId?: string | null): Promise<ConceptCard[]> {
  const response = await fetch(buildApiUrl("/concepts", { course_id: courseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<ConceptCard[]>(response);
}

export async function searchKnowledge(payload: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<SearchResponse>(response);
}

export async function askQuestion(payload: QARequest): Promise<QAResponse> {
  const response = await fetch(`${API_BASE_URL}/qa`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<QAResponse>(response);
}

export async function callAgent(payload: QARequest): Promise<AgentResponse> {
  const response = await fetch(`${API_BASE_URL}/agent`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<AgentResponse>(response);
}

export async function uploadFile(file: File, courseId?: string | null): Promise<UploadFileResponse> {
  const formData = new FormData();
  formData.append("upload", file);
  const response = await fetch(buildApiUrl("/files/upload", { course_id: courseId }), {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });
  return parseResponse<UploadFileResponse>(response);
}

export async function parseUploadedFiles(filePaths: string[], courseId?: string | null, force = false): Promise<BatchStartResponse> {
  const payload: ParseUploadedFilesRequest = { file_paths: filePaths, force };
  const response = await fetch(buildApiUrl("/ingestion/parse-uploaded-files", { course_id: courseId }), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  return parseResponse<BatchStartResponse>(response);
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
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, { cache: "no-store", headers: authHeaders() });
  return parseResponse<JobStatusResponse>(response);
}

export async function fetchBatchStatus(batchId: string): Promise<IngestionBatchSummary> {
  const response = await fetch(`${API_BASE_URL}/ingestion/batches/${batchId}`, { cache: "no-store", headers: authHeaders() });
  return parseResponse<IngestionBatchSummary>(response);
}

export async function fetchTaskStatus(runId: string): Promise<TaskStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/tasks/${runId}`, { cache: "no-store", headers: authHeaders() });
  return parseResponse<TaskStatusResponse>(response);
}

export async function fetchSessions(courseId?: string | null): Promise<SessionSummary[]> {
  const response = await fetch(buildApiUrl("/sessions", { course_id: courseId }), { cache: "no-store", headers: authHeaders() });
  return parseResponse<SessionSummary[]>(response);
}

export async function fetchSessionMessages(sessionId: string): Promise<SessionMessagesResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`, { cache: "no-store", headers: authHeaders() });
  return parseResponse<SessionMessagesResponse>(response);
}

export async function deleteSession(sessionId: string): Promise<DeleteResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, { method: "DELETE", headers: authHeaders() });
  return parseResponse<DeleteResponse>(response);
}

export async function streamAnswer(
  payload: QARequest,
  handlers: {
    onToken: (value: string) => void;
    onCitations: (value: QAResponse["citations"]) => void;
    onTrace?: (value: AgentTraceEventPayload) => void;
    onFinal?: (value: AgentResponse) => void;
    onMeta?: (value: { degraded_mode?: boolean; run_id?: string; session_id?: string; route?: string }) => void;
    onError?: (value: string) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/qa/stream`, {
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
      };
      try {
        parsed = JSON.parse(line);
      } catch (error) {
        console.warn("忽略无法解析的 SSE 数据行", { line, error });
        continue;
      }
      if (parsed.type === "meta") {
        handlers.onMeta?.({ run_id: parsed.run_id, session_id: parsed.session_id, route: parsed.route });
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
        handlers.onFinal?.(parsed.response);
        handlers.onMeta?.({
          degraded_mode: parsed.response.degraded_mode,
          run_id: parsed.response.run_id,
          session_id: parsed.response.session_id,
          route: parsed.response.route,
        });
      }
      if (typeof parsed.degraded_mode === "boolean") {
        handlers.onMeta?.({ degraded_mode: parsed.degraded_mode });
      }
    }
  }
}
