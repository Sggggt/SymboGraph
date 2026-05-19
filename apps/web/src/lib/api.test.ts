import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api client", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://api.test/api");
    vi.stubEnv("NEXT_PUBLIC_API_KEY", "test-key");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("sends API key headers on JSON requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ has_api_key: true }));
    vi.stubGlobal("fetch", fetchMock);
    const { updateModelSettings } = await import("./api");

    await updateModelSettings({ api_key: "new-key", clear_api_key: false });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/settings/model",
      expect.objectContaining({
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-API-Key": "test-key" },
      }),
    );
  });

  it("fetches runtime checks with reranker requirement", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        env_sync: { synced: true, missing_keys: [], extra_keys: [], bom_keys: [] },
        reranker: { enabled: true, device: "cpu", model: "model", url: "http://reranker:8080/rerank", reachable: true, healthy: true },
        infrastructure: { postgres: true, qdrant: true, redis: true },
        blocking_issues: [],
        warnings: [],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const { fetchRuntimeCheck } = await import("./api");

    await fetchRuntimeCheck();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/settings/runtime-check",
      expect.objectContaining({ cache: "no-store", headers: { "X-API-Key": "test-key" } }),
    );
  });

  it("throws structured API errors", async () => {
    const body = {
      detail: {
        code: "runtime_check_failed",
        title: "Runtime infrastructure check failed",
        message: "Reranker cannot be enabled.",
        issues: [{ code: "reranker_unreachable", title: "Missing", message: "No runtime", fix_commands: [".\\start-app.ps1"] }],
        fix_commands: [".\\start-app.ps1"],
      },
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 409 })));
    const { updateModelSettings } = await import("./api");

    await expect(updateModelSettings({ reranker_enabled: true })).rejects.toMatchObject({
      status: 409,
      structured: body.detail,
    });
  });

  it("uses short-lived tokens for batch log EventSource URLs", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ token: "stream-token", expires_at: "2026-05-08T00:00:00Z" }));
    vi.stubGlobal("fetch", fetchMock);
    const { createBatchLogToken, getBatchLogUrl } = await import("./api");

    await expect(createBatchLogToken("batch-1")).resolves.toMatchObject({ token: "stream-token" });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/ingestion/batches/batch-1/log-token",
      expect.objectContaining({ method: "POST", headers: { "X-API-Key": "test-key" } }),
    );
    expect(getBatchLogUrl("batch-1", "stream-token")).toBe("http://api.test/api/ingestion/batches/batch-1/logs?token=stream-token");
  });

  it("calls stale cleanup endpoints with API key headers", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ deleted_vectors: 1, deleted_chunks: 2, deleted_document_versions: 3, deleted_documents: 4, removed_graph_relations: 5, removed_graph_concepts: 6 }))
      .mockResolvedValueOnce(jsonResponse({ removed_relations: 1, removed_aliases: 2, removed_concepts: 3 }));
    vi.stubGlobal("fetch", fetchMock);
    const { cleanupStaleData, cleanupStaleGraph } = await import("./api");

    await cleanupStaleData("course-1");
    await cleanupStaleGraph("course-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://api.test/api/maintenance/cleanup-stale-data?course_id=course-1",
      expect.objectContaining({ method: "POST", headers: { "X-API-Key": "test-key" } }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://api.test/api/maintenance/cleanup-stale-graph?course_id=course-1",
      expect.objectContaining({ method: "POST", headers: { "X-API-Key": "test-key" } }),
    );
  });

  it("requires graph_type on graph requests and confirms destructive rebuilds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ graph_type: "semantic", schema_version: "typed_graph_v1", nodes: [], edges: [], node_counts: {}, edge_counts: {} }))
      .mockResolvedValueOnce(jsonResponse({ graph_type: "evidence", schema_version: "typed_graph_v1", nodes: [], edges: [], node_counts: {}, edge_counts: {} }))
      .mockResolvedValueOnce(jsonResponse({ batch_id: "batch-1", state: "extracting_graph", mode: "full" }))
      .mockResolvedValueOnce(jsonResponse({ batch_id: null, state: "dry_run", mode: "full", dry_run: true, affected_documents: 3 }));
    vi.stubGlobal("fetch", fetchMock);
    const { fetchGraph, fetchChapterGraph, rebuildGraph } = await import("./api");

    await fetchGraph("course-1", "semantic");
    await fetchChapterGraph("Lecture 1", "course-1", "evidence");
    await rebuildGraph("course-1", "full");
    await rebuildGraph("course-1", "full", true);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://api.test/api/courses/current/graph?course_id=course-1&graph_type=semantic",
      expect.objectContaining({ cache: "no-store", headers: { "X-API-Key": "test-key" } }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://api.test/api/graph/chapters/Lecture%201?course_id=course-1&graph_type=evidence",
      expect.objectContaining({ cache: "no-store", headers: { "X-API-Key": "test-key" } }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://api.test/api/maintenance/rebuild-graph?course_id=course-1",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ mode: "full", confirm_destructive: true, dry_run: false }) }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://api.test/api/maintenance/rebuild-graph?course_id=course-1",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ mode: "full", confirm_destructive: false, dry_run: true }) }),
    );
  });

  it("deletes courses with API key headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ deleted: true }));
    vi.stubGlobal("fetch", fetchMock);
    const { deleteCourse } = await import("./api");

    await deleteCourse("course-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/courses/course-1",
      expect.objectContaining({ method: "DELETE", headers: { "X-API-Key": "test-key" } }),
    );
  });

  it("parses SSE stream chunks", async () => {
    const body = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('data: {"type":"meta","run_id":"run-1","session_id":"session-1"}\n\n'));
        controller.enqueue(encoder.encode('data: {"token":"hello"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"final","response":{"run_id":"run-1","session_id":"session-1","answer":"done","citations":[],"used_chunks":[],"route":"retrieve_notes","trace":[],"degraded_mode":false}}\n\n'));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const { streamAnswer } = await import("./api");
    const tokens: string[] = [];
    const meta: unknown[] = [];

    await streamAnswer(
      { question: "hello", top_k: 3 },
      {
        onToken: (value) => tokens.push(value),
        onCitations: () => undefined,
        onMeta: (value) => meta.push(value),
      },
    );

    expect(tokens).toEqual(["hello"]);
    expect(meta).toContainEqual({ run_id: "run-1", session_id: "session-1", route: undefined });
    expect(meta).toContainEqual({ degraded_mode: false, run_id: "run-1", session_id: "session-1", route: "retrieve_notes" });
  });

  it("passes structured stream errors to handlers before rejecting", async () => {
    const body = {
      detail: {
        code: "runtime_check_failed",
        title: "Runtime infrastructure check failed",
        message: "模型端点不可用。",
      },
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 503 })));
    const { streamAnswer } = await import("./api");
    const onError = vi.fn();

    await expect(
      streamAnswer(
        { question: "hello", top_k: 3 },
        {
          onToken: () => undefined,
          onCitations: () => undefined,
          onError,
        },
      ),
    ).rejects.toThrow("模型端点不可用。");

    expect(onError).toHaveBeenCalledWith("模型端点不可用。");
  });

  it("continues parsing SSE after malformed data lines", async () => {
    const body = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode("data: not-json\n\n"));
        controller.enqueue(encoder.encode('data: {"token":"still works"}\n\n'));
        controller.close();
      },
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const { streamAnswer } = await import("./api");
    const tokens: string[] = [];

    await streamAnswer(
      { question: "hello", top_k: 3 },
      {
        onToken: (value) => tokens.push(value),
        onCitations: () => undefined,
      },
    );

    expect(tokens).toEqual(["still works"]);
    expect(warn).toHaveBeenCalledWith("忽略无法解析的 SSE 数据行", expect.objectContaining({ line: "not-json" }));
  });
});
