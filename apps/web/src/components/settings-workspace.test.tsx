// @vitest-environment jsdom

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ModelSettingsUpdate } from "@course-kg/shared";

import {
  AgentAdmissionSettingsSection,
  bindEmbeddingProtocolToCandidateSettings,
  buildRuntimeSettingsPayload,
  buildHotReloadSettingsPayload,
  candidateChangedKeysForDisplay,
  EmbeddingProtocolSelect,
  GraphProtocolSettingsSection,
  ModelProtocolSelect,
  ParameterName,
  RQ_KMEANS_PROTOCOL_DEPTH,
  RUNTIME_ENV_AUTHORITY_NOTE,
  RqProtocolDepthField,
  SETTINGS_PARAMETER_HELP,
  SourceIoConcurrencyField,
  UPLOAD_MAX_BYTES_LIMITS,
  UploadSecuritySettingsSection,
} from "./settings-workspace";

type AssertFalse<T extends false> = T;
type AssertTrue<T extends true> = T;
const MODEL_SETTINGS_UPDATE_HAS_RQ_DEPTH: AssertFalse<
  "rq_kmeans_levels" extends keyof ModelSettingsUpdate ? true : false
> = false;
const MODEL_SETTINGS_UPDATE_HAS_EMBEDDING_PROTOCOL: AssertTrue<
  "embedding_api_protocol" extends keyof ModelSettingsUpdate ? true : false
> = true;

describe("settings parameter help", () => {
  it("explains the single root env and three lifecycle boundaries", () => {
    expect(RUNTIME_ENV_AUTHORITY_NOTE).toContain("仓库根 .env 是唯一配置来源");
    expect(RUNTIME_ENV_AUTHORITY_NOTE).toContain("所有参数立即写入该文件");
    expect(RUNTIME_ENV_AUTHORITY_NOTE).toContain("服务参数在重启后生效");
  });

  it("keeps the fixed RQ depth out of the writable settings contract", () => {
    expect(MODEL_SETTINGS_UPDATE_HAS_RQ_DEPTH).toBe(false);
    expect(MODEL_SETTINGS_UPDATE_HAS_EMBEDDING_PROTOCOL).toBe(true);
  });

  it("renders embedding as an independent OpenAI-compatible-only protocol", () => {
    const onChange = vi.fn();
    const { unmount } = render(<EmbeddingProtocolSelect value="openai" onChange={onChange} />);

    const select = screen.getByRole("combobox", {
      name: /向量接口协议/,
    }) as HTMLSelectElement;
    expect(Array.from(select.options).map((option) => option.value)).toEqual(["openai"]);
    expect(select.options[0]?.textContent).toBe("OpenAI-compatible（当前唯一支持）");
    expect(screen.queryByRole("option", { name: /Anthropic/i })).toBeNull();
    expect(screen.getByText(/rebuild_required/)).toBeTruthy();
    expect(SETTINGS_PARAMETER_HELP["向量接口协议"]).toContain("不表示已支持 Anthropic embedding");
    unmount();
  });

  it("binds the embedding protocol to every non-empty candidate settings payload", () => {
    expect(
      bindEmbeddingProtocolToCandidateSettings(
        { fixed_chunk_size_tokens: 640, graph_api_protocol: "anthropic" },
        "openai",
      ),
    ).toEqual({
      fixed_chunk_size_tokens: 640,
      graph_api_protocol: "anthropic",
      embedding_api_protocol: "openai",
    });
    expect(bindEmbeddingProtocolToCandidateSettings({}, "openai")).toEqual({});
  });

  it("does not misreport a frozen embedding identity as a user change", () => {
    expect(
      candidateChangedKeysForDisplay(
        {
          fixed_chunk_size_tokens: 640,
          embedding_api_protocol: "openai",
        },
        "openai",
      ),
    ).toEqual(["fixed_chunk_size_tokens"]);
    expect(
      candidateChangedKeysForDisplay(
        { embedding_api_protocol: "openai" },
        undefined,
      ),
    ).toEqual([]);
  });

  it("keeps rebuild identity fields out of the ordinary hot-save payload", () => {
    const payload = buildHotReloadSettingsPayload({
      chat_api_protocol: "anthropic",
      chat_base_url: " https://chat.example.test ",
      chat_resolve_ip: "",
      chat_model: "chat-model",
      embedding_batch_size: "10",
      worker_concurrency: "3",
      model_request_concurrency: "3",
      model_request_timeout_seconds: "240",
      chat_json_max_tokens: "12000",
      agent_request_concurrency: "4",
      source_io_concurrency: "4",
      agent_request_queue_limit: "8",
      agent_request_queue_timeout_seconds: "30",
      agent_request_lease_ttl_seconds: "300",
      upload_max_bytes: "104857600",
      concept_i18n_enabled: true,
      chat_api_key: "",
      clear_chat_api_key: false,
      graph_api_key: "",
      clear_graph_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: true,
    });

    expect(payload).toMatchObject({
      chat_api_protocol: "anthropic",
      chat_base_url: "https://chat.example.test",
      embedding_batch_size: 10,
      chat_json_max_tokens: 12000,
      model_bridge_enabled: true,
    });
    for (const forbiddenField of [
      "embedding_api_protocol",
      "embedding_base_url",
      "embedding_resolve_ip",
      "embedding_model",
      "embedding_dimensions",
      "graph_api_protocol",
      "graph_base_url",
      "graph_resolve_ip",
      "graph_model",
    ]) {
      expect(Object.prototype.hasOwnProperty.call(payload, forbiddenField)).toBe(false);
    }
  });

  it("includes hot, rebuild and service values in one root-env save payload", () => {
    const payload = buildRuntimeSettingsPayload({
      chat_api_protocol: "openai",
      graph_api_protocol: "anthropic",
      embedding_api_protocol: "openai",
      chat_base_url: "https://chat.example.test/v1",
      graph_base_url: " https://graph.example.test/v1 ",
      embedding_base_url: " https://embedding.example.test/v1 ",
      chat_resolve_ip: "",
      graph_resolve_ip: "",
      embedding_resolve_ip: "",
      embedding_model: "embedding-v2",
      chat_model: "chat-v2",
      graph_model: "graph-v2",
      embedding_dimensions: "1024",
      embedding_batch_size: "10",
      worker_concurrency: "4",
      model_request_concurrency: "3",
      model_request_timeout_seconds: "240",
      chat_json_max_tokens: "12000",
      agent_request_concurrency: "4",
      source_io_concurrency: "4",
      agent_request_queue_limit: "8",
      agent_request_queue_timeout_seconds: "30",
      agent_request_lease_ttl_seconds: "300",
      upload_max_bytes: "104857600",
      concept_i18n_enabled: false,
      fixed_chunk_size_tokens: "640",
      fixed_chunk_overlap_tokens: "96",
      chat_api_key: "",
      clear_chat_api_key: false,
      graph_api_key: "",
      clear_graph_api_key: false,
      embedding_api_key: "",
      clear_embedding_api_key: false,
      model_bridge_enabled: true,
      mid_concept_extraction_max_model_batches: "2",
      mid_concept_extraction_max_candidates_per_batch: "8",
      mid_concept_extraction_max_tokens_per_batch: "2400",
      mid_concept_candidate_keep_threshold: "0.62",
      rq_kmeans_max_k: "6",
      rq_residual_tau: "0.65",
      edge_distance_protocol: "edge_distance_log_calibrated_strength_v2",
      rq_membership_protocol: "rq_primary_chain_v1",
      edge_projection_protocol: "membership_q15_layer_type_calibrated_v3",
      edge_type_calibration_protocol: "type_local_winsorized_minmax_v1",
      rq_membership_temperature: "0.4",
      dense_knn_k_min: "3",
      dense_knn_k_max: "12",
      dense_reverse_b_min_base: "1",
      dense_reverse_b_max_base: "6",
      dense_reverse_b_min_doc: "1",
      dense_reverse_b_max_doc: "4",
      dense_reverse_b_min_lang: "0",
      dense_reverse_b_max_lang: "2",
      dense_min_cosine: "0.61",
      dense_strong_cosine: "0.75",
      cross_doc_out_quota_min: "1",
      cross_doc_out_quota_max: "3",
      cross_doc_min_cosine: "0.64",
      cross_language_out_quota_min: "0",
      cross_language_out_quota_max: "2",
      cross_language_min_cosine: "0.68",
    });

    expect(payload).toMatchObject({
      chat_model: "chat-v2",
      graph_base_url: "https://graph.example.test/v1",
      embedding_base_url: "https://embedding.example.test/v1",
      embedding_model: "embedding-v2",
      fixed_chunk_size_tokens: 640,
      worker_concurrency: 4,
      model_bridge_enabled: true,
    });
  });

  it("renders the closed protocol choices with an explicit lifecycle", () => {
    const onChange = vi.fn();
    const { unmount } = render(
      <ModelProtocolSelect
        label="聊天接口协议"
        value="openai"
        onChange={onChange}
        lifecycle="hot_reloadable · applies to the next model call"
      />,
    );

    const select = screen.getByRole("combobox") as HTMLSelectElement;
    expect(Array.from(select.options).map((option) => option.value)).toEqual([
      "openai",
      "anthropic",
    ]);
    expect(screen.getByText(/hot_reloadable/)).toBeTruthy();
    fireEvent.change(select, { target: { value: "anthropic" } });
    expect(onChange).toHaveBeenCalledWith("anthropic");
    unmount();
  });

  it("renders a delayed tooltip for parameter names", () => {
    render(<ParameterName label="RQ-KMeans 协议深度" />);

    const trigger = screen.getByText("RQ-KMeans 协议深度").parentElement;
    const tooltip = screen.getByRole("tooltip");

    expect(trigger?.getAttribute("aria-describedby")).toBe(tooltip.id);
    expect(tooltip.textContent).toContain("L3");
    expect(tooltip.className).toContain("group-hover:delay-500");
    expect(tooltip.className).toContain("group-focus:delay-500");
  });

  it("exposes the active RQ depth as a fixed non-editable protocol value", () => {
    render(<RqProtocolDepthField />);

    const input = screen.getByRole("spinbutton", { name: /RQ-KMeans 协议深度/ }) as HTMLInputElement;
    expect(input.valueAsNumber).toBe(RQ_KMEANS_PROTOCOL_DEPTH);
    expect(input.min).toBe(String(RQ_KMEANS_PROTOCOL_DEPTH));
    expect(input.max).toBe(String(RQ_KMEANS_PROTOCOL_DEPTH));
    expect(input.disabled).toBe(true);
    expect(SETTINGS_PARAMETER_HELP["RQ-KMeans 协议深度"]).toContain("固定为 3");
  });

  it("shows graph protocols and primary-membership controls as rebuild-only identities", () => {
    const onChange = vi.fn();
    render(
      <GraphProtocolSettingsSection
        values={{
          edge_distance_protocol: "edge_distance_log_calibrated_strength_v2",
          rq_membership_protocol: "rq_primary_chain_v1",
          edge_projection_protocol: "membership_q15_layer_type_calibrated_v3",
          edge_type_calibration_protocol: "type_local_winsorized_minmax_v1",
          rq_membership_temperature: "0.35",
        }}
        onChange={onChange}
      />,
    );

    expect(screen.getByText(/candidate → shadow rebuild → evaluation → promotion/)).toBeTruthy();
    for (const label of [
      "边距离协议",
      "RQ membership 协议",
      "边投影协议",
      "边类型校准协议",
    ]) {
      expect((screen.getByRole("textbox", { name: new RegExp(label) }) as HTMLInputElement).disabled).toBe(true);
    }
    for (const label of ["RQ softmax 温度"]) {
      expect((screen.getByRole("spinbutton", { name: new RegExp(label) }) as HTMLInputElement).disabled).toBe(false);
    }
    const temperature = screen.getByRole("spinbutton", {
      name: /RQ softmax 温度/,
    }) as HTMLInputElement;
    expect(temperature.min).toBe("0.01");
    expect(temperature.step).toBe("0.01");
    expect(temperature.checkValidity()).toBe(true);
    expect(SETTINGS_PARAMETER_HELP["RQ membership 协议"]).toContain("LLM");
  });

  it("keeps active runtime and graph parameters documented without legacy BM25 wording", () => {
    for (const label of ["模型请求并发", "源文件 I/O 并发", "Agent 请求并发", "Agent 等待队列上限", "Agent 排队超时秒数", "Agent 租约 TTL 秒数", "单文件上传上限（字节）", "中粗层双语派生", "片段 Top K", "跨文档桥最小配额", "引用验证预算", "工作进程并发"]) {
      expect(SETTINGS_PARAMETER_HELP[label]).toBeTruthy();
    }

    expect(UPLOAD_MAX_BYTES_LIMITS).toEqual({ defaultValue: 104_857_600, min: 1, max: 10_737_418_240 });
    expect(SETTINGS_PARAMETER_HELP["LLM 双语查询面"]).toBeTruthy();
    expect(Object.values(SETTINGS_PARAMETER_HELP).join("\n")).not.toContain("BM25");
  });

  it("documents gray-zone decisions as deterministic and model-free", () => {
    expect(SETTINGS_PARAMETER_HELP["路径 green 阈值"]).toContain("LLM 不参与");
    expect(SETTINGS_PARAMETER_HELP["路径 gray 阈值"]).toContain("deterministic local rule");
    expect(SETTINGS_PARAMETER_HELP["路径 gray 阈值"]).toContain("模型调用数必须为 0");
    expect(SETTINGS_PARAMETER_HELP["路径 gray 阈值"]).not.toContain("LLM evaluator");
    expect(SETTINGS_PARAMETER_HELP["路径 hard 阈值"]).toContain("不能绕过或覆盖");
  });

  it("renders and updates the hot-reloadable upload byte limit", () => {
    const onChange = vi.fn();
    render(<UploadSecuritySettingsSection value="104857600" onChange={onChange} />);

    const input = screen.getByRole("spinbutton", { name: /单文件上传上限/ }) as HTMLInputElement;
    expect(input.valueAsNumber).toBe(104857600);
    expect(input.getAttribute("min")).toBe(String(UPLOAD_MAX_BYTES_LIMITS.min));
    expect(input.getAttribute("max")).toBe(String(UPLOAD_MAX_BYTES_LIMITS.max));
    expect(screen.getByText(/下一次上传请求/)).toBeTruthy();

    fireEvent.change(input, { target: { value: "209715200" } });
    expect(onChange).toHaveBeenCalledWith("209715200");
  });

  it("renders and updates the bounded source I/O concurrency", () => {
    const onChange = vi.fn();
    render(
      <SourceIoConcurrencyField value="4" onChange={onChange} />,
    );

    const input = screen.getByRole("spinbutton", {
      name: /源文件 I\/O 并发/,
    }) as HTMLInputElement;
    expect(input.valueAsNumber).toBe(4);
    expect(input.min).toBe("1");
    expect(input.max).toBe("64");

    fireEvent.change(input, { target: { value: "7" } });
    expect(onChange).toHaveBeenCalledWith("7");
  });

  it("renders and updates the cross-process Agent admission limits", () => {
    const onChange = vi.fn();
    render(
      <AgentAdmissionSettingsSection
        values={{
          agent_request_concurrency: "4",
          agent_request_queue_limit: "8",
          agent_request_queue_timeout_seconds: "30",
          agent_request_lease_ttl_seconds: "300",
        }}
        onChange={onChange}
      />,
    );

    expect(screen.getByText(/Redis 统一执行跨进程并发与有界 FIFO 排队/)).toBeTruthy();
    const concurrency = screen.getByRole("spinbutton", { name: /Agent 请求并发/ }) as HTMLInputElement;
    const queueLimit = screen.getByRole("spinbutton", { name: /Agent 等待队列上限/ }) as HTMLInputElement;
    expect(concurrency.valueAsNumber).toBe(4);
    expect(queueLimit.valueAsNumber).toBe(8);

    fireEvent.change(queueLimit, { target: { value: "12" } });
    expect(onChange).toHaveBeenCalledWith("agent_request_queue_limit", "12");
  });
});
