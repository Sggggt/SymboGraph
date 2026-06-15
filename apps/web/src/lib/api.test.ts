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

  it("starts parse batches with built-in context graph and cancels batches", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ batch_id: "batch-1", state: "queued" }))
      .mockResolvedValueOnce(jsonResponse({ batch_id: "batch-1", state: "cancel_requested", trigger_source: "upload", source_root: "root", total_files: 1, processed_files: 0, success_count: 0, failure_count: 0, skipped_count: 0, coverage_by_source_type: {}, errors: [], graph_stats: {} }));
    vi.stubGlobal("fetch", fetchMock);
    const { parseUploadedFiles, cancelBatch } = await import("./api");

    await parseUploadedFiles(["/data/knowledge-base/a.pdf"], "knowledge-base-1", true);
    await cancelBatch("batch-1", "knowledge-base-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://api.test/api/ingestion/parse-uploaded-files?knowledge_base_id=knowledge-base-1",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          file_paths: ["/data/knowledge-base/a.pdf"],
          force: true,
          full_reparse: false,
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://api.test/api/ingestion/batches/batch-1/cancel?knowledge_base_id=knowledge-base-1",
      expect.objectContaining({ method: "POST", headers: { "X-API-Key": "test-key" } }),
    );
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

  it("passes production runtime setting fields", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ provider: "openai_compatible" }));
    vi.stubGlobal("fetch", fetchMock);
    const { updateModelSettings } = await import("./api");

    await updateModelSettings({
      worker_concurrency: 3,
      model_request_concurrency: 3,
      model_request_timeout_seconds: 240,
      embedding_batch_size: 10,
      fixed_chunk_size_tokens: 512,
      fixed_chunk_overlap_tokens: 80,
      context_package_token_budget: 2400,
      reranker_enabled: true,
      reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
      reranker_max_length: 512,
      reranker_device: "cpu",
      agent_coarse_entry_budget: 8,
      agent_coarse_jump_budget: 2,
      agent_mid_entry_budget: 16,
      agent_mid_expansion_radius_cap: 2,
      agent_fine_entry_budget: 16,
      agent_frontier_expansion_budget: 80,
      agent_max_depth_per_layer: 3,
      agent_max_labels_per_node: 3,
      agent_max_edge_reuse: 2,
      agent_max_cycle_reward_per_path: 0.18,
      agent_ambiguous_edge_distance_low: 0.45,
      agent_ambiguous_edge_distance_high: 1.35,
      agent_drilldown_budget_per_layer: 16,
      agent_chunk_candidate_budget: 80,
      agent_structure_restore_budget: 16,
      context_path_summary_budget: 32,
      agent_planning_round_budget: 2,
      agent_max_typed_actions_per_round: 8,
      agent_repair_round_budget: 2,
      agent_verification_budget: 8,
    });

    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      worker_concurrency: 3,
      model_request_concurrency: 3,
      model_request_timeout_seconds: 240,
      embedding_batch_size: 10,
      fixed_chunk_size_tokens: 512,
      fixed_chunk_overlap_tokens: 80,
      context_package_token_budget: 2400,
      reranker_enabled: true,
      reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
      reranker_max_length: 512,
      reranker_device: "cpu",
      agent_coarse_entry_budget: 8,
      agent_coarse_jump_budget: 2,
      agent_mid_entry_budget: 16,
      agent_mid_expansion_radius_cap: 2,
      agent_fine_entry_budget: 16,
      agent_frontier_expansion_budget: 80,
      agent_max_depth_per_layer: 3,
      agent_max_labels_per_node: 3,
      agent_max_edge_reuse: 2,
      agent_max_cycle_reward_per_path: 0.18,
      agent_ambiguous_edge_distance_low: 0.45,
      agent_ambiguous_edge_distance_high: 1.35,
      agent_drilldown_budget_per_layer: 16,
      agent_chunk_candidate_budget: 80,
      agent_structure_restore_budget: 16,
      context_path_summary_budget: 32,
      agent_planning_round_budget: 2,
      agent_max_typed_actions_per_round: 8,
      agent_repair_round_budget: 2,
      agent_verification_budget: 8,
    });
  });

  it("requests context graph status refresh", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ batch_id: null, state: "context_graph_active", mode: "four_layer_context_graph", affected_documents: 1 }));
    vi.stubGlobal("fetch", fetchMock);
    const { rebuildGraph } = await import("./api");

    await rebuildGraph("knowledge-base-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/maintenance/rebuild-graph?knowledge_base_id=knowledge-base-1",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          dry_run: false,
        }),
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

  it("calls stale data cleanup endpoint with API key headers", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({
        inactive_chunks: 2,
        stale_vector_records: 5,
        stale_qdrant_points: 5,
        collections: ["knowledge_chunks"],
        applied: false,
      }));
    vi.stubGlobal("fetch", fetchMock);
    const { cleanupStaleData } = await import("./api");

    await cleanupStaleData("knowledge-base-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/maintenance/cleanup-stale-data?knowledge_base_id=knowledge-base-1",
      expect.objectContaining({ method: "POST", headers: { "X-API-Key": "test-key" } }),
    );
  });

  it("requires graph_type on graph requests and confirms destructive rebuilds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ graph_type: "chunk-relation", schema_version: "context_graph_v1", nodes: [], edges: [], counts: {}, sampled_counts: {}, node_counts: {}, edge_counts: {}, freshness: { is_stale: false } }))
      .mockResolvedValueOnce(jsonResponse({ batch_id: null, state: "context_graph_active", mode: "four_layer_context_graph" }))
      .mockResolvedValueOnce(jsonResponse({ batch_id: null, state: "context_graph_active", mode: "four_layer_context_graph", dry_run: true, affected_documents: 3 }));
    vi.stubGlobal("fetch", fetchMock);
    const { fetchGraph, rebuildGraph } = await import("./api");

    await fetchGraph("knowledge-base-1", "chunk-relation");
    await rebuildGraph("knowledge-base-1");
    await rebuildGraph("knowledge-base-1", true);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://api.test/api/knowledge_bases/current/graph?knowledge_base_id=knowledge-base-1&graph_type=chunk-relation&view=overview",
      expect.objectContaining({ cache: "no-store", headers: { "X-API-Key": "test-key" } }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://api.test/api/maintenance/rebuild-graph?knowledge_base_id=knowledge-base-1",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ dry_run: false }) }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://api.test/api/maintenance/rebuild-graph?knowledge_base_id=knowledge-base-1",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ dry_run: true }) }),
    );
  });

  it("deletes knowledge bases with API key headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ deleted: true }));
    vi.stubGlobal("fetch", fetchMock);
    const { deleteKnowledgeBase } = await import("./api");

    await deleteKnowledgeBase("knowledge-base-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/knowledge_bases/knowledge-base-1",
      expect.objectContaining({ method: "DELETE", headers: { "X-API-Key": "test-key" } }),
    );
  });

  it("parses SSE stream chunks", async () => {
    const body = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('data: {"type":"meta","run_id":"run-1","session_id":"session-1"}\n\n'));
        controller.enqueue(encoder.encode('data: {"token":"hello"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"final","response":{"run_id":"run-1","session_id":"session-1","answer":"done","citations":[],"used_chunks":[],"route":"retrieve_sources","trace":[],"degraded_mode":false}}\n\n'));
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
    expect(meta).toContainEqual({ degraded_mode: false, run_id: "run-1", session_id: "session-1", route: "retrieve_sources" });
  });

  it("parses profile assistant SSE chunks", async () => {
    const profileJson = {
      schema_version: "user_profile_v1",
      library_type: "legal",
      ui_labels: {},
      prompt_pack: {},
      conversation_preferences: {},
    };
    const body = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('data: {"type":"meta","session_id":"profile-session"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"token","token":"已生成"}\n\n'));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "profile_json", profile_json: profileJson, warnings: ["check"], profile_hash: "hash-1" })}\n\n`));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "final", state: { session_id: "profile-session", messages: [], latest_profile_json: profileJson, latest_profile_hash: "hash-1", warnings: ["check"] } })}\n\n`));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { streamProfileAssistant } = await import("./api");
    const tokens: string[] = [];
    const profileEvents: unknown[] = [];
    const finalStates: unknown[] = [];
    const meta: unknown[] = [];

    await streamProfileAssistant(
      { prompt: "legal", session_id: "profile-session", base_profile_id: "base" },
      {
        onToken: (value) => tokens.push(value),
        onProfileJson: (value) => profileEvents.push(value),
        onFinal: (value) => finalStates.push(value),
        onMeta: (value) => meta.push(value),
      },
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/settings/profile-assistant/stream",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ prompt: "legal", session_id: "profile-session", base_profile_id: "base" }),
      }),
    );
    expect(meta).toEqual([{ session_id: "profile-session", cached: undefined }]);
    expect(tokens).toEqual(["已生成"]);
    expect(profileEvents).toEqual([{ profile_json: profileJson, warnings: ["check"], profile_hash: "hash-1" }]);
    expect(finalStates).toEqual([{ session_id: "profile-session", messages: [], latest_profile_json: profileJson, latest_profile_hash: "hash-1", warnings: ["check"] }]);
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
