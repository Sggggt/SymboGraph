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

  it("registers a multi-file selection sequentially for one knowledge base", async () => {
    let activeRequests = 0;
    let maxActiveRequests = 0;
    const completionOrder: string[] = [];
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      activeRequests += 1;
      maxActiveRequests = Math.max(maxActiveRequests, activeRequests);
      const formData = init?.body as FormData;
      const file = formData.get("upload") as File;
      await Promise.resolve();
      completionOrder.push(file.name);
      activeRequests -= 1;
      return jsonResponse({
        document_id: `document-${file.name}`,
        job_id: `job-${file.name}`,
        status: "queued",
        source_path: `/data/${file.name}`,
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { uploadFilesSequentially } = await import("./api");
    const progress: number[] = [];

    const responses = await uploadFilesSequentially(
      [new File(["alpha"], "alpha.md"), new File(["beta"], "beta.md")],
      "knowledge-base-1",
      undefined,
      (completed) => progress.push(completed),
    );

    expect(maxActiveRequests).toBe(1);
    expect(completionOrder).toEqual(["alpha.md", "beta.md"]);
    expect(progress).toEqual([1, 2]);
    expect(responses.map((response) => response.source_path)).toEqual([
      "/data/alpha.md",
      "/data/beta.md",
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("cancels agent runs through the control endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ run_id: "run-1", status: "cancelled", error: "cancelled_by_user" }));
    vi.stubGlobal("fetch", fetchMock);
    const { cancelAgentRun } = await import("./api");

    await cancelAgentRun("run-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/agent/runs/run-1/cancel",
      expect.objectContaining({ method: "POST", headers: { "X-API-Key": "test-key" } }),
    );
  });

  it("reads the canonical PostgreSQL P&E audit endpoint without caching", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        contract_version: "agent_pe_audit_public_v1",
        run_id: "run-1",
        knowledge_base_id: "kb-1",
        run_status: "completed",
        counts: { plans: 0, actions: 0, observations: 0 },
        ordering: {
          plans: "plan_index ASC, created_at ASC, id ASC",
          actions:
            "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC",
          observations: "created_at ASC, id ASC",
        },
        plans: [],
        actions: [],
        observations: [],
        provider_raw_response_exposed: false,
        credentials_exposed: false,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const { fetchAgentPEAudit } = await import("./api");

    await fetchAgentPEAudit("run-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/agent/runs/run-1/pe-audit",
      expect.objectContaining({
        cache: "no-store",
        headers: { "X-API-Key": "test-key" },
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("sends API key headers on JSON requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ has_chat_api_key: true }));
    vi.stubGlobal("fetch", fetchMock);
    const { updateModelSettings } = await import("./api");

    await updateModelSettings({ chat_api_key: "new-key", clear_chat_api_key: false });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/settings/model",
      expect.objectContaining({
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-API-Key": "test-key" },
      }),
    );
  });

  it("routes search requests through graph-enhanced retrieval", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        query: "markov blanket",
        results: [],
        degraded_mode: false,
        model_audit: { retrieval_pipeline: "layered_context_graph" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const { searchKnowledge } = await import("./api");

    await searchKnowledge({ knowledge_base_id: "kb-1", query: "markov blanket", top_k: 8, filters: {}, retrieval_granularity: "coarse" });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/search/graph-enhanced",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": "test-key" },
        body: JSON.stringify({ knowledge_base_id: "kb-1", query: "markov blanket", top_k: 8, filters: {}, retrieval_granularity: "coarse" }),
      }),
    );
  });

  it("passes production runtime setting fields", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ provider: "multi_protocol" }));
    vi.stubGlobal("fetch", fetchMock);
    const { updateModelSettings } = await import("./api");

    await updateModelSettings({
      chat_api_protocol: "anthropic",
      graph_api_protocol: "anthropic",
      worker_concurrency: 3,
      model_request_concurrency: 3,
      model_request_timeout_seconds: 240,
      source_io_concurrency: 4,
      upload_max_bytes: 104857600,
      concept_i18n_enabled: true,
      query_facet_bilingual_enabled: true,
      query_facet_posterior_enabled: true,
      query_facet_posterior_observation_budget: 64,
      query_facet_posterior_round_budget: 2,
      query_facet_posterior_convergence_epsilon: 0.02,
      embedding_batch_size: 10,
      fixed_chunk_size_tokens: 512,
      fixed_chunk_overlap_tokens: 80,
      context_package_token_budget: 2400,
      dense_knn_k_min: 5,
      dense_knn_k_max: 24,
      dense_reverse_b_min_base: 2,
      dense_reverse_b_max_base: 8,
      dense_reverse_b_min_doc: 1,
      dense_reverse_b_max_doc: 6,
      dense_reverse_b_min_lang: 1,
      dense_reverse_b_max_lang: 4,
      dense_min_cosine: 0.58,
      dense_strong_cosine: 0.72,
      cross_doc_out_quota_min: 1,
      cross_doc_out_quota_max: 4,
      cross_doc_min_cosine: 0.62,
      cross_language_out_quota_min: 0,
      cross_language_out_quota_max: 3,
      cross_language_min_cosine: 0.65,
      retrieval_result_top_k_default: 8,
      agent_coarse_initial_budget: 8,
      agent_coarse_total_budget: 8,
      agent_coarse_top_k: 5,
      agent_mid_per_coarse_budget: 6,
      agent_coarse_drilldown_mid_initial_budget: 10,
      agent_mid_initial_budget: 8,
      agent_mid_top_k: 16,
      agent_chunk_per_mid_budget: 8,
      agent_chunk_initial_budget: 24,
      agent_chunk_top_k: 40,
      candidate_pool_dedupe_budget: 160,
      agent_max_depth_per_layer: 3,
      agent_max_labels_per_node: 3,
      agent_max_edge_reuse: 2,
      agent_max_cycle_reward_per_path: 0.18,
      agent_cycle_reward_distance_threshold: 1.2,
      agent_path_distance_green_threshold: 0.45,
      agent_path_distance_gray_threshold: 1.35,
      agent_path_distance_hard_threshold: 2.4,
      agent_structure_restore_per_chunk_budget: 4,
      agent_structure_restore_budget: 16,
      context_path_summary_budget: 32,
      agent_planning_round_budget: 2,
      agent_max_typed_actions_per_round: 8,
      agent_repair_round_budget: 2,
      agent_verification_budget: 8,
      enable_auto_tpe: false,
      tpe_trial_budget: 6,
      tpe_startup_random_trials: 3,
      tpe_good_quantile_gamma: 0.25,
      tpe_probe_query_budget: 6,
      tpe_trial_timeout_seconds: 30,
      tpe_candidate_pool_size: 24,
      operating_point_hard_gate_max_edge_density: 0.45,
      operating_point_hard_gate_max_isolated_ratio: 0.35,
      operating_point_hard_gate_max_hubness_ratio: 12,
      operating_point_hard_gate_min_structure_recovery_rate: 0.25,
      operating_point_hard_gate_max_candidate_latency_p95_ms: 30000,
    });

    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      chat_api_protocol: "anthropic",
      graph_api_protocol: "anthropic",
      worker_concurrency: 3,
      model_request_concurrency: 3,
      model_request_timeout_seconds: 240,
      source_io_concurrency: 4,
      upload_max_bytes: 104857600,
      concept_i18n_enabled: true,
      query_facet_bilingual_enabled: true,
      query_facet_posterior_enabled: true,
      query_facet_posterior_observation_budget: 64,
      query_facet_posterior_round_budget: 2,
      query_facet_posterior_convergence_epsilon: 0.02,
      embedding_batch_size: 10,
      fixed_chunk_size_tokens: 512,
      fixed_chunk_overlap_tokens: 80,
      context_package_token_budget: 2400,
      dense_knn_k_min: 5,
      dense_knn_k_max: 24,
      dense_reverse_b_min_base: 2,
      dense_reverse_b_max_base: 8,
      dense_reverse_b_min_doc: 1,
      dense_reverse_b_max_doc: 6,
      dense_reverse_b_min_lang: 1,
      dense_reverse_b_max_lang: 4,
      dense_min_cosine: 0.58,
      dense_strong_cosine: 0.72,
      cross_doc_out_quota_min: 1,
      cross_doc_out_quota_max: 4,
      cross_doc_min_cosine: 0.62,
      cross_language_out_quota_min: 0,
      cross_language_out_quota_max: 3,
      cross_language_min_cosine: 0.65,
      retrieval_result_top_k_default: 8,
      agent_coarse_initial_budget: 8,
      agent_coarse_total_budget: 8,
      agent_coarse_top_k: 5,
      agent_mid_per_coarse_budget: 6,
      agent_coarse_drilldown_mid_initial_budget: 10,
      agent_mid_initial_budget: 8,
      agent_mid_top_k: 16,
      agent_chunk_per_mid_budget: 8,
      agent_chunk_initial_budget: 24,
      agent_chunk_top_k: 40,
      candidate_pool_dedupe_budget: 160,
      agent_max_depth_per_layer: 3,
      agent_max_labels_per_node: 3,
      agent_max_edge_reuse: 2,
      agent_max_cycle_reward_per_path: 0.18,
      agent_cycle_reward_distance_threshold: 1.2,
      agent_path_distance_green_threshold: 0.45,
      agent_path_distance_gray_threshold: 1.35,
      agent_path_distance_hard_threshold: 2.4,
      agent_structure_restore_per_chunk_budget: 4,
      agent_structure_restore_budget: 16,
      context_path_summary_budget: 32,
      agent_planning_round_budget: 2,
      agent_max_typed_actions_per_round: 8,
      agent_repair_round_budget: 2,
      agent_verification_budget: 8,
      enable_auto_tpe: false,
      tpe_trial_budget: 6,
      tpe_startup_random_trials: 3,
      tpe_good_quantile_gamma: 0.25,
      tpe_probe_query_budget: 6,
      tpe_trial_timeout_seconds: 30,
      tpe_candidate_pool_size: 24,
      operating_point_hard_gate_max_edge_density: 0.45,
      operating_point_hard_gate_max_isolated_ratio: 0.35,
      operating_point_hard_gate_max_hubness_ratio: 12,
      operating_point_hard_gate_min_structure_recovery_rate: 0.25,
      operating_point_hard_gate_max_candidate_latency_p95_ms: 30000,
    });
  });

  it("fetches readonly automatic TPE status", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({ knowledge_base_id: "kb-1", enabled: false, latest_run: null }));
    vi.stubGlobal("fetch", fetchMock);
    const { fetchAutoTpeStatus } = await import("./api");

    await fetchAutoTpeStatus("kb-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/knowledge-bases/kb-1/graph-operating-point/auto-tpe/latest",
      expect.objectContaining({ cache: "no-store", headers: { "X-API-Key": "test-key" } }),
    );
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

  it("fetches runtime checks", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        env_sync: { synced: true, missing_keys: [], extra_keys: [], bom_keys: [] },
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

  it("uses the explicit Runtime Settings candidate lifecycle endpoints", async () => {
    const payload = { candidate: { id: "candidate-1", status: "staged" } };
    // A Response body is single-use; each endpoint call must receive a fresh
    // response just as it would from the browser fetch implementation.
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);
    const {
      createRuntimeSettingsCandidate,
      fetchRuntimeSettingsCandidate,
      runRuntimeSettingsCandidateAction,
      promoteRuntimeSettingsCandidate,
    } = await import("./api");

    await createRuntimeSettingsCandidate({
      knowledge_base_ids: ["kb-1"],
      settings: {
        fixed_chunk_size_tokens: 640,
        graph_api_protocol: "anthropic",
        embedding_api_protocol: "openai",
      },
      dry_run_only: false,
      source: "unit_ui",
    });
    await fetchRuntimeSettingsCandidate("candidate-1");
    await runRuntimeSettingsCandidateAction("candidate-1", "build");
    await promoteRuntimeSettingsCandidate("candidate-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://api.test/api/settings/runtime-candidates",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          knowledge_base_ids: ["kb-1"],
          settings: {
            fixed_chunk_size_tokens: 640,
            graph_api_protocol: "anthropic",
            embedding_api_protocol: "openai",
          },
          dry_run_only: false,
          source: "unit_ui",
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://api.test/api/settings/runtime-candidates/candidate-1",
      expect.objectContaining({ cache: "no-store" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://api.test/api/settings/runtime-candidates/candidate-1/build",
      expect.objectContaining({ method: "POST", body: "{}" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://api.test/api/settings/runtime-candidates/candidate-1/promote",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("throws structured API errors", async () => {
    const body = {
      detail: {
        code: "runtime_check_failed",
        title: "Runtime infrastructure check failed",
        message: "Model endpoint cannot be reached.",
        issues: [{ code: "model_endpoint_unreachable", title: "Missing", message: "No runtime", fix_commands: [".\\start-app.ps1"] }],
        fix_commands: [".\\start-app.ps1"],
      },
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 409 })));
    const { updateModelSettings } = await import("./api");

    await expect(updateModelSettings({ chat_model: "unit-test-chat-model" })).rejects.toMatchObject({
      status: 409,
      structured: body.detail,
    });
  });

  it("uses short-lived tokens for batch log EventSource URLs", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ batch_id: "batch-1", token: "stream-token", expires_at: "2026-05-08T00:00:00Z" }));
    vi.stubGlobal("fetch", fetchMock);
    const { createBatchLogToken, getBatchLogUrl } = await import("./api");

    await expect(createBatchLogToken("batch-1")).resolves.toMatchObject({ batch_id: "batch-1", token: "stream-token" });
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
      "http://api.test/api/knowledge_bases/current/graph?knowledge_base_id=knowledge-base-1&graph_type=chunk-relation&view=overview&limit=100",
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
        controller.enqueue(encoder.encode('data: {"type":"meta","run_id":"run-1","session_id":"session-1","retrieval_granularity":"coarse"}\n\n'));
        controller.enqueue(encoder.encode('data: {"token":"hello"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"final","response":{"run_id":"run-1","session_id":"session-1","answer":"done","citations":[],"used_chunks":[],"route":"retrieve_sources","trace":[],"degraded_mode":false,"retrieval_granularity":"coarse","retrieval_trace_id":"trace-1","context_package_id":"package-1"}}\n\n'));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { streamAnswer } = await import("./api");
    const tokens: string[] = [];
    const meta: unknown[] = [];
    const finalResponses: unknown[] = [];
    const controller = new AbortController();

    await streamAnswer(
      { question: "hello", top_k: 3, retrieval_granularity: "coarse" },
      {
        onToken: (value) => tokens.push(value),
        onCitations: () => undefined,
        onMeta: (value) => meta.push(value),
        onFinal: (value) => finalResponses.push(value),
      },
      { signal: controller.signal },
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/qa/stream",
      expect.objectContaining({
        signal: controller.signal,
        body: JSON.stringify({ question: "hello", top_k: 3, retrieval_granularity: "coarse" }),
      }),
    );
    expect(tokens).toEqual(["hello"]);
    expect(finalResponses).toContainEqual(expect.objectContaining({ retrieval_trace_id: "trace-1", context_package_id: "package-1" }));
    expect(meta).toContainEqual({ run_id: "run-1", session_id: "session-1", route: undefined, retrieval_granularity: "coarse" });
    expect(meta).toContainEqual({ degraded_mode: false, run_id: "run-1", session_id: "session-1", route: "retrieve_sources", retrieval_granularity: "coarse" });
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
