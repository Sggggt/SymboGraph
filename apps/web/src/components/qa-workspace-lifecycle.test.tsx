// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentResponse, AgentTraceEventPayload, Citation, TaskStatusResponse } from "@course-kg/shared";

import * as api from "@/lib/api";
import { QAWorkspace } from "./qa-workspace";

vi.mock("@/lib/api", async (importOriginal) => ({
  ...await importOriginal<typeof api>(),
  fetchDashboard: vi.fn(),
  fetchSessions: vi.fn(),
  fetchSessionMessages: vi.fn(),
  fetchModelSettings: vi.fn(),
  fetchTaskStatus: vi.fn(),
  cancelAgentRun: vi.fn(),
  deleteSession: vi.fn(),
  streamAnswer: vi.fn(),
}));
vi.mock("@/components/knowledge-base-context", () => ({
  useKnowledgeBaseContext: () => ({ selectedKnowledgeBaseId: "unit-test-kb" }),
}));

type StreamHandle = {
  handlers: Parameters<typeof api.streamAnswer>[1];
  signal: AbortSignal;
  resolve: () => void;
  reject: (error: Error) => void;
};
let streams: StreamHandle[];
let queryClient: QueryClient;
const streamKey = "qa.activeStream.unit-test-kb";
const sessionId = "unit-test-session";
const runId = "unit-test-run-1";
const answer = "当前资料证据不足，请补充资料范围。";
const trace: AgentTraceEventPayload = {
  contract_version: "agent_trace_event_public_v1", type: "trace", id: "unit-test-event",
  run_id: runId, sequence_index: 0, node: "evidence_gate", status: "blocked",
  input_summary: "证据检查", output_summary: "需要补充资料", document_ids: [],
  scores: { contract_version: "agent_trace_scores_public_v1", audit_kind: "evidence_gate" }, duration_ms: 0,
};

function finalResponse(overrides: Partial<AgentResponse> = {}): AgentResponse {
  return {
    run_id: runId, session_id: sessionId, answer, citations: [], used_chunks: [],
    route: "layered_context_graph", trace: [], degraded_mode: false,
    ...overrides,
  } as AgentResponse;
}

function terminalStatus(state = "needs_clarification"): TaskStatusResponse {
  return { run_id: runId, session_id: sessionId, status: state, answer, trace: [trace] };
}

function mockTranscript(citations: Citation[] = []) {
  vi.mocked(api.fetchSessionMessages).mockResolvedValue({
    session_id: sessionId,
    messages: [
      { role: "user", content: "示例问题", run_id: runId, citations: [] },
      { role: "assistant", content: answer, run_id: runId, citations },
    ],
    conversation_state: {
      protocol_version: "conversation_state_v1", scope_protocol_version: "conversation_state_scope_v1",
      qa_session_id: sessionId, knowledge_base_id: "unit-test-kb", revision: 1,
      state_hash: "a".repeat(64), scope_hash: "b".repeat(64),
      active_user_constraints: { instructions: [], retrieval_filters: {} },
      task_state: { status: "waiting_user" }, history_references: [], transcript_message_count: 2,
      prompt_history_audit: {}, evidence_authority: false, gray_zone_decision_authority: false,
    },
  });
}

async function mountWorkspace() {
  render(<StrictMode><QueryClientProvider client={queryClient}><QAWorkspace /></QueryClientProvider></StrictMode>);
  await screen.findByRole("textbox");
}

async function submitQuestion(question = "示例问题") {
  const previous = vi.mocked(api.streamAnswer).mock.calls.length;
  fireEvent.change(screen.getByRole("textbox"), { target: { value: question } });
  fireEvent.click(screen.getByRole("button", { name: "提问" }));
  await waitFor(() => expect(api.streamAnswer).toHaveBeenCalledTimes(previous + 1));
}

function expectStopped() {
  expect(screen.queryByRole("button", { name: "取消当前对话" })).toBeNull();
  expect(screen.getByRole("button", { name: "提问" })).toBeTruthy();
  expect(screen.queryByText("智能体运行中")).toBeNull();
  expect(JSON.parse(window.localStorage.getItem(streamKey) ?? "null")).toBeNull();
}

describe("QA run terminal reconciliation", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    window.localStorage.clear();
    vi.stubGlobal("scrollTo", vi.fn());
    Element.prototype.scrollIntoView = vi.fn();
    streams = [];
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    vi.mocked(api.fetchDashboard).mockResolvedValue({ tree: [] } as unknown as Awaited<ReturnType<typeof api.fetchDashboard>>);
    vi.mocked(api.fetchSessions).mockResolvedValue([]);
    vi.mocked(api.fetchModelSettings).mockResolvedValue({ chat_model: "unit-test-chat" } as Awaited<ReturnType<typeof api.fetchModelSettings>>);
    vi.mocked(api.fetchTaskStatus).mockImplementation(async (id) => ({ run_id: id, status: "running" }));
    vi.mocked(api.deleteSession).mockResolvedValue({ deleted: true });
    mockTranscript();
    vi.mocked(api.streamAnswer).mockImplementation((_request, handlers, options) => new Promise<void>((resolve, reject) => {
      const signal = options!.signal!;
      streams.push({ handlers, signal, resolve, reject });
      signal.addEventListener("abort", () => reject(new DOMException("stream closed", "AbortError")), { once: true });
      handlers.onMeta?.({ run_id: `unit-test-run-${streams.length}`, session_id: sessionId });
    }));
  });

  afterEach(async () => {
    cleanup();
    await act(async () => { for (const stream of streams) stream.resolve(); });
    queryClient.clear();
    vi.unstubAllGlobals();
  });

  it("shows spinners instead of a false model error while initial data is loading", async () => {
    vi.mocked(api.fetchSessions).mockImplementation(() => new Promise(() => undefined));
    vi.mocked(api.fetchModelSettings).mockImplementation(() => new Promise(() => undefined));

    await mountWorkspace();

    expect(screen.getByText("正在读取模型配置...")).toBeTruthy();
    expect(screen.getByText("正在加载...")).toBeTruthy();
    expect(screen.queryByText("模型：未读取")).toBeNull();
    expect(screen.queryByText("模型服务暂时不可用，请稍后重试。")).toBeNull();
    expect(screen.getAllByRole("status").every((item) => item.querySelector(".animate-spin"))).toBe(true);
  });

  it("applies provider deltas, accepts an authoritative replacement, and follows the stream", async () => {
    await mountWorkspace();
    await submitQuestion();

    await act(async () => {
      streams[0].handlers.onToken("partial output");
    });
    expect(screen.getByText("partial output")).toBeTruthy();
    await waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "end",
    }));

    await act(async () => {
      streams[0].handlers.onAnswerReplace?.("authoritative output");
    });
    expect(screen.getByText("authoritative output")).toBeTruthy();
    expect(screen.queryByText("partial output")).toBeNull();
  });

  it("stops at final despite a pending transport and ignores late metadata, tokens and duplicate finals", async () => {
    await mountWorkspace();
    await submitQuestion();
    expect(screen.getByRole("button", { name: "取消当前对话" })).toBeTruthy();

    await act(async () => {
      streams[0].handlers.onTrace?.(trace);
      streams[0].handlers.onFinal?.(finalResponse());
      streams[0].handlers.onMeta?.({ run_id: runId, session_id: sessionId });
      streams[0].handlers.onToken("不应显示的迟到内容");
      streams[0].handlers.onFinal?.(finalResponse());
    });

    expectStopped();
    expect(screen.getAllByText(answer)).toHaveLength(1);
    expect(screen.queryByText("不应显示的迟到内容")).toBeNull();
    expect(screen.getAllByTestId("agent-trace-stream")).toHaveLength(1);
    expect(api.cancelAgentRun).not.toHaveBeenCalled();

    await submitQuestion("下一轮问题");
    await act(async () => {
      streams[0].handlers.onError?.("cancelled_by_user");
      streams[0].reject(new Error("old connection failure"));
    });
    expect(screen.queryByText("已取消当前对话")).toBeNull();
    expect(screen.getByRole("button", { name: "取消当前对话" })).toBeTruthy();
  });

  it.each(["completed", "needs_clarification"])("recovers persisted %s and citations after reload without cancelling the run", async (state) => {
    window.localStorage.setItem(streamKey, JSON.stringify({
      question: "示例问题", runId, sessionId, startedAt: new Date().toISOString(),
    }));
    vi.mocked(api.fetchTaskStatus).mockResolvedValue(terminalStatus(state));
    if (state === "completed") mockTranscript([{ chunk_id: "unit-test-chunk" } as Citation]);

    await mountWorkspace();
    await waitFor(expectStopped);
    await waitFor(() => expect(api.fetchSessionMessages).toHaveBeenCalledWith(sessionId));
    expect(screen.getAllByText(answer)).toHaveLength(1);
    if (state === "completed") await screen.findByRole("button", { name: "1 条来源 · 查看" });
    expect(screen.queryByText("已取消当前对话")).toBeNull();
    expect(api.cancelAgentRun).not.toHaveBeenCalled();
    expect(api.streamAnswer).not.toHaveBeenCalled();
  });

  it("automatically restores the newest persisted session on first entry", async () => {
    vi.mocked(api.fetchSessions).mockResolvedValue([
      {
        id: sessionId,
        knowledge_base_id: "unit-test-kb",
        title: "最近会话",
        last_question: "示例问题",
        last_answer: answer,
      } as Awaited<ReturnType<typeof api.fetchSessions>>[number],
    ]);
    mockTranscript();

    await mountWorkspace();

    await waitFor(() => expect(api.fetchSessionMessages).toHaveBeenCalledWith(sessionId));
    expect(JSON.parse(window.localStorage.getItem("qa.sessionId.unit-test-kb") ?? "null")).toBe(sessionId);
    expect(screen.getAllByText(answer)).toHaveLength(1);
  });

  it("reuses the React Query transcript when the user leaves and re-enters QA", async () => {
    vi.mocked(api.fetchSessions).mockResolvedValue([
      {
        id: sessionId,
        knowledge_base_id: "unit-test-kb",
        title: "最近会话",
        last_question: "示例问题",
        last_answer: answer,
      } as Awaited<ReturnType<typeof api.fetchSessions>>[number],
    ]);
    mockTranscript();

    await mountWorkspace();
    await screen.findByText(answer);
    expect(api.fetchSessionMessages).toHaveBeenCalledTimes(1);

    cleanup();
    await mountWorkspace();
    expect(screen.getAllByText(answer)).toHaveLength(1);
    await waitFor(() => expect(api.fetchSessionMessages).toHaveBeenCalledTimes(1));
    expect(api.fetchTaskStatus).not.toHaveBeenCalled();
  });

  it("persists an explicit blank conversation across a full workspace remount", async () => {
    vi.mocked(api.fetchSessions).mockResolvedValue([
      {
        id: sessionId,
        knowledge_base_id: "unit-test-kb",
        title: "最近会话",
        last_question: "示例问题",
        last_answer: answer,
      } as Awaited<ReturnType<typeof api.fetchSessions>>[number],
    ]);
    mockTranscript();

    await mountWorkspace();
    await screen.findByText(answer);
    fireEvent.click(screen.getByRole("button", { name: "会话" }));
    fireEvent.click(await screen.findByRole("button", { name: "新建会话" }));
    expect(JSON.parse(window.localStorage.getItem("qa.autoResumeSuppressed.unit-test-kb") ?? "false")).toBe(true);
    expect(screen.queryByText("正在加载...")).toBeNull();
    expect(screen.getByText("开始一轮有证据支撑的资料问答")).toBeTruthy();

    cleanup();
    await mountWorkspace();
    expect(screen.queryByText(answer)).toBeNull();
    expect(screen.queryByText("正在加载...")).toBeNull();
    expect(screen.getByText("开始一轮有证据支撑的资料问答")).toBeTruthy();
    expect(JSON.parse(window.localStorage.getItem("qa.sessionId.unit-test-kb") ?? "null")).toBeNull();
  });

  it("removes a session optimistically without waiting for the DELETE response", async () => {
    const pendingSessionId = "unit-test-pending-delete";
    window.localStorage.setItem("qa.autoResumeSuppressed.unit-test-kb", "true");
    vi.mocked(api.fetchSessions).mockResolvedValue([
      {
        id: pendingSessionId,
        knowledge_base_id: "unit-test-kb",
        title: "待删除会话",
        last_question: null,
        last_answer: null,
      } as Awaited<ReturnType<typeof api.fetchSessions>>[number],
    ]);
    let resolveDelete!: (value: { deleted: boolean }) => void;
    let responseResolved = false;
    vi.mocked(api.deleteSession).mockReturnValue(new Promise((resolve) => {
      resolveDelete = (value) => {
        responseResolved = true;
        resolve(value);
      };
    }));

    await mountWorkspace();
    fireEvent.click(screen.getByRole("button", { name: "会话" }));
    await screen.findByText("待删除会话");
    fireEvent.click(screen.getByRole("button", { name: "删除会话" }));

    await waitFor(() => expect(screen.queryByText("待删除会话")).toBeNull());
    expect(responseResolved).toBe(false);
    expect(queryClient.getQueryData(["sessions", "unit-test-kb"])).toEqual([]);

    await act(async () => resolveDelete({ deleted: true }));
    await waitFor(() => expect(api.deleteSession).toHaveBeenCalledWith(pendingSessionId));
  });

  it("removes a stale session row without persisting it or surfacing an unhandled rejection", async () => {
    await mountWorkspace();
    fireEvent.click(screen.getByRole("button", { name: "会话" }));
    fireEvent.click(await screen.findByRole("button", { name: "新建会话" }));

    const staleSessionId = "unit-test-stale-session";
    await act(async () => {
      queryClient.setQueryData(["sessions", "unit-test-kb"], [
        {
          id: staleSessionId,
          knowledge_base_id: "unit-test-kb",
          title: "过期会话",
          last_question: null,
          last_answer: null,
        },
      ]);
    });
    vi.mocked(api.fetchSessionMessages).mockRejectedValueOnce(
      Object.assign(new Error("Session not found"), { status: 404 }),
    );

    fireEvent.click(screen.getByRole("button", { name: "会话" }));
    fireEvent.click(await screen.findByRole("button", { name: "过期会话" }));

    await screen.findByText("该会话已不存在，历史列表已刷新。");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(api.fetchSessionMessages).toHaveBeenCalledWith(staleSessionId);
    expect(queryClient.getQueryData(["sessions", "unit-test-kb"])).toEqual([]);
    expect(JSON.parse(window.localStorage.getItem("qa.sessionId.unit-test-kb") ?? "null")).toBeNull();
  });

  it("closes the local stream when polling wins, with no false cancellation or duplicated answer", async () => {
    await mountWorkspace();
    await submitQuestion();
    await act(async () => {
      queryClient.setQueryData(["agent-run", runId], terminalStatus());
    });
    await waitFor(expectStopped);
    expect(streams[0].signal.aborted).toBe(true);
    await act(async () => {
      streams[0].handlers.onFinal?.(finalResponse());
      streams[0].handlers.onMeta?.({ run_id: runId });
    });
    expect(screen.getAllByText(answer)).toHaveLength(1);
    expect(screen.queryByText("已取消当前对话")).toBeNull();
    expect(api.cancelAgentRun).not.toHaveBeenCalled();
  });

  it("continues status recovery on EOF without final, but does not mistake a completed trace for run completion", async () => {
    await mountWorkspace();
    await submitQuestion();
    await act(async () => {
      streams[0].handlers.onTrace?.(trace);
      streams[0].resolve();
    });
    expect(screen.getByRole("button", { name: "取消当前对话" })).toBeTruthy();

    await act(async () => { queryClient.setQueryData(["agent-run", runId], terminalStatus()); });
    await waitFor(expectStopped);
    expect(screen.getAllByText(answer)).toHaveLength(1);
  });

  it("reports EOF without run metadata and stops loading", async () => {
    vi.mocked(api.streamAnswer).mockResolvedValue(undefined);
    await mountWorkspace();
    await submitQuestion();
    await waitFor(expectStopped);
    expect(screen.getByText("问答服务暂时无法连接，请稍后重试。")).toBeTruthy();
  });

  it("keeps polling after a transport failure with run metadata and restores the durable terminal result", async () => {
    await mountWorkspace();
    await submitQuestion();
    await act(async () => {
      streams[0].reject(new TypeError("Failed to fetch"));
    });

    expect(screen.getByRole("button", { name: "取消当前对话" })).toBeTruthy();
    expect(screen.getByText("实时连接已中断，正在读取本轮任务状态。")).toBeTruthy();
    expect(JSON.parse(window.localStorage.getItem(streamKey) ?? "null")).toMatchObject({
      runId,
      sessionId,
    });

    await act(async () => {
      queryClient.setQueryData(["agent-run", runId], terminalStatus("completed"));
    });
    await waitFor(expectStopped);
    expect(screen.getAllByText(answer)).toHaveLength(1);
    expect(screen.queryByText("实时连接已中断，正在读取本轮任务状态。")).toBeNull();
    expect(api.cancelAgentRun).not.toHaveBeenCalled();
  });

  it.each(["failed", "cancelled"])("stops the stream when the server reports %s", async (state) => {
    await mountWorkspace();
    await submitQuestion();
    await act(async () => {
      queryClient.setQueryData(["agent-run", runId], { ...terminalStatus(state), answer: null });
    });
    await waitFor(expectStopped);
    expect(streams[0].signal.aborted).toBe(true);
    expect(screen.getByText(state === "cancelled" ? "已取消当前对话" : "本次问答未能完成，请稍后重试。")).toBeTruthy();
    expect(api.cancelAgentRun).not.toHaveBeenCalled();
  });

  it("keeps explicit user cancellation separate and ignores its delayed response after the next request", async () => {
    let resolveCancel!: (status: TaskStatusResponse) => void;
    vi.mocked(api.cancelAgentRun).mockReturnValue(new Promise((resolve) => { resolveCancel = resolve; }));
    await mountWorkspace();
    await submitQuestion();
    fireEvent.click(screen.getByRole("button", { name: "取消当前对话" }));
    await waitFor(() => expect(api.cancelAgentRun).toHaveBeenCalledExactlyOnceWith(runId));
    expectStopped();
    expect(streams[0].signal.aborted).toBe(true);
    expect(screen.getByText("已取消当前对话")).toBeTruthy();

    await submitQuestion("下一轮问题");
    await act(async () => { resolveCancel(terminalStatus("cancelled")); });
    expect(screen.getByRole("button", { name: "取消当前对话" })).toBeTruthy();
    expect(screen.queryByText("已取消当前对话")).toBeNull();
    expect(streams[1].signal.aborted).toBe(false);
  });

  it("replays the completed result when completion won the server-side user-cancel race", async () => {
    vi.mocked(api.cancelAgentRun).mockResolvedValue(terminalStatus());
    await mountWorkspace();
    await submitQuestion();
    fireEvent.click(screen.getByRole("button", { name: "取消当前对话" }));
    await waitFor(() => expect(screen.queryByText("已取消当前对话")).toBeNull());
    expectStopped();
    expect(screen.getAllByText(answer)).toHaveLength(1);
  });
});
