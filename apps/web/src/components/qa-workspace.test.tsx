// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AnswerModelAudit, Citation, ConversationStatePayload, RetrievalGranularity } from "@course-kg/shared";
import { AGENT_PARAMETER_HELP, AgentParameterName, answerAuditFromTrace, answerUsageLabel, ConversationStatePanel, legacyQaPayloadStorageKeys, MessageBubble, normalizeMessages, preserveTurnTraces, productQaErrorMessage, RetrievalGranularitySelector } from "./qa-workspace";

function SelectorHarness() {
  const [value, setValue] = useState<RetrievalGranularity>("mid");
  return <RetrievalGranularitySelector value={value} onChange={setValue} />;
}

describe("answerUsageLabel", () => {
  it("credits a cache hit only from provider-reported cache-read tokens", () => {
    const audit = {
      provider_call: {
        protocol_version: "provider_prompt_cache_audit_v1",
        prompt_cache: {
          protocol_version: "provider_prompt_cache_audit_v1",
          api_protocol: "anthropic",
          cache_mode: "anthropic_explicit_ephemeral",
          cacheable_system_prompt_present: true,
          cacheable_system_prompt_sha256: "a".repeat(64),
          cacheable_system_prompt_utf8_bytes: 128,
          provider_response_persisted: false,
        },
        usage: {
          protocol_version: "provider_prompt_cache_audit_v1",
          api_protocol: "anthropic",
          input_tokens: 201,
          output_tokens: 157,
          total_tokens: null,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 10240,
          cache_hit: true,
          cache_write: false,
          token_accounting_mode: "provider_reported_anthropic_fields_no_cross_field_inference_v1",
          usage_present: true,
          provider_response_persisted: false,
        },
        provider_response_persisted: false,
      },
    } as AnswerModelAudit;

    expect(answerUsageLabel(audit)).toBe(
      "Provider tokens：输入 201 · 输出 157 · 缓存读取 10240 · 缓存命中",
    );
    expect(
      answerUsageLabel({
        ...audit,
        provider_call: {
          ...audit.provider_call!,
          usage: {
            ...audit.provider_call!.usage,
            cache_read_input_tokens: 0,
            cache_hit: false,
          },
        },
      }),
    ).toBe("Provider tokens：输入 201 · 输出 157 · 缓存读取 0");
  });

  it("recovers provider usage from the persisted grounded-answer trace", () => {
    const audit = {
      answer_model_called: true,
      answer_claim_limit: 6,
    } as AnswerModelAudit;
    const recovered = answerAuditFromTrace([
      {
        id: "event-1",
        run_id: "run-1",
        sequence_index: 0,
        node: "grounded_answer",
        status: "completed",
        document_ids: [],
        scores: {
          audit_kind: "grounded_answer",
          answer_model_audit: audit,
        },
        duration_ms: 1,
      },
    ] as unknown as Parameters<typeof answerAuditFromTrace>[0]);

    expect(recovered).toEqual(audit);
  });
});

describe("preserveTurnTraces", () => {
  it("keeps a completed run trace when session transcript replay omits it", () => {
    const trace = [{ node: "query_understanding", status: "completed" }] as unknown as Parameters<typeof preserveTurnTraces>[1][number]["trace"];
    const current = [
      { role: "assistant" as const, content: "answer", run_id: "run-1", trace },
    ];
    const replayed = [
      { role: "assistant" as const, content: "answer", run_id: "run-1" },
    ];

    expect(preserveTurnTraces(replayed, current)[0].trace).toBe(trace);
  });
});

describe("QA browser persistence boundary", () => {
  it("removes legacy large payload keys while preserving small session controls", () => {
    expect(legacyQaPayloadStorageKeys("kb-1")).toEqual([
      "qa.turns.kb-1",
      "qa.draftAnswer.kb-1",
      "qa.citations.kb-1",
      "qa.trace.kb-1",
      "qa.latestRun.kb-1",
      "qa.conversationState.kb-1",
    ]);
    expect(legacyQaPayloadStorageKeys("kb-1")).not.toContain("qa.sessionId.kb-1");
    expect(legacyQaPayloadStorageKeys("kb-1")).not.toContain(
      "qa.retrievalGranularity.kb-1",
    );
  });
});

describe("productQaErrorMessage", () => {
  it("does not expose provider transport diagnostics in the QA product surface", () => {
    const raw = "OpenAI-compatible request failed after 6 attempts: model_bridge request failed; http_status=502; error_code=provider_error";

    expect(productQaErrorMessage(raw)).toBe("模型服务暂时不可用，请稍后重试。");
    expect(productQaErrorMessage(raw)).not.toContain("502");
    expect(productQaErrorMessage(raw)).not.toContain("model_bridge");
  });
});

describe("RetrievalGranularitySelector", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("defaults to 普通模式 and switches to 摘要模式", () => {
    render(<SelectorHarness />);

    const midButton = screen.getByTestId("retrieval-granularity-mid");
    const coarseButton = screen.getByTestId("retrieval-granularity-coarse");

    expect(midButton.getAttribute("aria-pressed")).toBe("true");
    expect(coarseButton.getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(coarseButton);

    expect(midButton.getAttribute("aria-pressed")).toBe("false");
    expect(coarseButton.getAttribute("aria-pressed")).toBe("true");
  });

  it("shows mode help only after a 1000ms hover delay", () => {
    vi.useFakeTimers();
    render(<SelectorHarness />);

    const midButton = screen.getByTestId("retrieval-granularity-mid");
    fireEvent.mouseEnter(midButton);

    act(() => {
      vi.advanceTimersByTime(999);
    });
    expect(screen.queryByTestId("retrieval-granularity-tooltip")).toBeNull();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByTestId("retrieval-granularity-tooltip").textContent).toContain("从中层概念直接进入检索");

    fireEvent.mouseLeave(midButton);
    expect(screen.queryByTestId("retrieval-granularity-tooltip")).toBeNull();
  });
});

describe("AgentParameterName", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("renders parameter help with a 1000ms hover delay", () => {
    vi.useFakeTimers();
    render(<AgentParameterName label="证据包令牌预算" />);

    const trigger = screen.getByText("证据包令牌预算").parentElement;

    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(trigger?.getAttribute("aria-describedby")).toBeNull();

    fireEvent.mouseEnter(trigger!);
    act(() => {
      vi.advanceTimersByTime(999);
    });
    expect(screen.queryByRole("tooltip")).toBeNull();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    const tooltip = screen.getByRole("tooltip");

    expect(trigger?.getAttribute("aria-describedby")).toBe(tooltip.id);
    expect(tooltip.textContent).toContain("上下文证据包可容纳的令牌上限");
    expect(tooltip.className).toContain("fixed");
    expect(tooltip.className).toContain("z-[9999]");

    fireEvent.mouseLeave(trigger!);
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("documents every parameter in the dialog", () => {
    const labels = [
      "模型双语查询词面",
      "Query facet posterior 观察预算",
      "Query facet posterior 轮次预算",
      "Query facet posterior 收敛阈值",
      "证据包令牌预算",
      "结果保留数量默认值",
      "粗概念起点数量",
      "粗概念保留数量",
      "每个粗概念中概念预算",
      "普通模式中概念起点数量",
      "摘要模式中概念起点数量",
      "中概念保留数量",
      "每个中概念片段预算",
      "片段起点数量",
      "片段最终保留数量",
      "候选去重池预算",
      "每层最大深度",
      "每节点标签上限",
      "边复用上限",
      "闭环奖励上限",
      "闭环奖励距离阈值",
      "路径绿色阈值",
      "路径灰区阈值",
      "路径硬中断阈值",
      "扩展观察总预算",
      "每个片段结构恢复数量",
      "路径摘要预算",
      "规划轮次预算",
      "每轮动作上限",
      "修复轮次预算",
      "引用验证预算",
    ];

    for (const label of labels) {
      expect(AGENT_PARAMETER_HELP[label]).toBeTruthy();
    }
  });

  it("documents gray-zone decisions as deterministic executor rules with no LLM evaluator authority", () => {
    const help = AGENT_PARAMETER_HELP["路径灰区阈值"];

    expect(help).toContain("executor");
    expect(help).toContain("版本化本地规则");
    expect(help).toContain("LLM 与证据评估器不参与");
    expect(help).not.toContain("交给智能体评估器");
  });
});

describe("ConversationStatePanel", () => {
  afterEach(() => cleanup());

  it("shows durable constraints, task state, provenance references, and the no-evidence boundary", () => {
    const state: ConversationStatePayload = {
      protocol_version: "conversation_state_v1",
      scope_protocol_version: "conversation_state_scope_v1",
      qa_session_id: "11111111-1111-4111-8111-111111111111",
      knowledge_base_id: "22222222-2222-4222-8222-222222222222",
      revision: 3,
      state_hash: "a".repeat(64),
      scope_hash: "b".repeat(64),
      active_user_constraints: {
        instructions: ["Prefer the selected document."],
        retrieval_filters: {
          document_ids: ["33333333-3333-4333-8333-333333333333"],
        },
      },
      task_state: {
        status: "active",
        objective: "Explain the selected source",
        current_step: "awaiting_user",
      },
      history_references: [
        {
          protocol_version: "answer_context_citation_reference_v1",
          turn_index: 0,
          run_id: "44444444-4444-4444-8444-444444444444",
          answer_session_id: "55555555-5555-4555-8555-555555555555",
          context_package_id: "66666666-6666-4666-8666-666666666666",
          retrieval_trace_id: "77777777-7777-4777-8777-777777777777",
          citation_verification_ids: ["88888888-8888-4888-8888-888888888888"],
        },
      ],
      transcript_message_count: 8,
      prompt_history_audit: { selected_turn_count: 6 },
      evidence_authority: false,
      gray_zone_decision_authority: false,
    };

    render(<ConversationStatePanel state={state} />);

    expect(screen.getByTestId("conversation-state-panel").textContent).toContain("Prefer the selected document.");
    expect(screen.getByTestId("conversation-state-panel").textContent).toContain("Explain the selected source");
    expect(screen.getByTestId("conversation-state-panel").textContent).toContain("已有 1 轮回答");
    expect(screen.getByTestId("conversation-state-panel").textContent).toContain("事实仍以资料来源为准");
    expect(screen.getByTestId("conversation-state-panel").textContent).toContain("已应用资料范围");
    expect(screen.getByTestId("conversation-state-panel").textContent).not.toContain("document_ids");
    expect(screen.getByTestId("conversation-state-panel").textContent).not.toContain("33333333-3333-4333-8333-333333333333");
    expect(screen.getByTestId("conversation-state-panel").textContent).not.toContain(state.scope_hash);
    expect(screen.getByTestId("conversation-state-panel").textContent).not.toContain(state.history_references[0].context_package_id);
    expect(screen.getByTestId("conversation-state-panel").textContent).not.toContain(state.history_references[0].citation_verification_ids[0]);
  });

  it("maps durable history references back to historical assistant Context Packages", () => {
    const runId = "44444444-4444-4444-8444-444444444444";
    const state: ConversationStatePayload = {
      protocol_version: "conversation_state_v1",
      scope_protocol_version: "conversation_state_scope_v1",
      qa_session_id: "11111111-1111-4111-8111-111111111111",
      knowledge_base_id: "22222222-2222-4222-8222-222222222222",
      revision: 1,
      state_hash: "a".repeat(64),
      scope_hash: "b".repeat(64),
      active_user_constraints: { instructions: [], retrieval_filters: {} },
      task_state: { status: "active" },
      history_references: [
        {
          protocol_version: "answer_context_citation_reference_v1",
          turn_index: 0,
          run_id: runId,
          answer_session_id: "55555555-5555-4555-8555-555555555555",
          context_package_id: "66666666-6666-4666-8666-666666666666",
          retrieval_trace_id: "77777777-7777-4777-8777-777777777777",
          citation_verification_ids: [],
        },
      ],
      transcript_message_count: 2,
      prompt_history_audit: {},
      evidence_authority: false,
      gray_zone_decision_authority: false,
    };

    const turns = normalizeMessages(
      [
        { role: "user", content: "What is posterior probability?", run_id: runId },
        { role: "assistant", content: "A grounded answer.", run_id: runId, route: "layered_context_graph" },
      ],
      state,
    );

    expect(turns[1].context_package_id).toBe(state.history_references[0].context_package_id);
    expect(turns[1].retrieval_trace_id).toBe(state.history_references[0].retrieval_trace_id);
  });

  it("opens the citation set belonging to the clicked current or historical turn", () => {
    const historicalCitation = { chunk_id: "chunk:historical" } as Citation;
    const currentCitation = { chunk_id: "chunk:current" } as Citation;
    const onOpenCitations = vi.fn();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <MessageBubble
          turn={{ role: "assistant", content: "Historical answer", route: "direct_answer", citations: [historicalCitation] }}
          index={0}
          onOpenCitations={onOpenCitations}
          defaultContextExpanded={false}
        />
        <MessageBubble
          turn={{ role: "assistant", content: "Current answer", route: "direct_answer", citations: [currentCitation] }}
          index={1}
          onOpenCitations={onOpenCitations}
          defaultContextExpanded={false}
        />
      </QueryClientProvider>,
    );

    const buttons = screen.getAllByRole("button", { name: "1 条来源 · 查看" });
    fireEvent.click(buttons[0]);
    fireEvent.click(buttons[1]);

    expect(onOpenCitations).toHaveBeenNthCalledWith(1, [historicalCitation]);
    expect(onOpenCitations).toHaveBeenNthCalledWith(2, [currentCitation]);
  });

  it("does not render trace, ids, hashes, protocols, or raw payloads in an answer bubble", () => {
    const internalId = "77777777-7777-4777-8777-777777777777";
    const internalHash = "f".repeat(64);
    render(
      <MessageBubble
        turn={{
          role: "assistant",
          content: "示例数据集成会汇集并统一处理多个公开来源。",
          route: "layered_context_graph",
          retrieval_trace_id: internalId,
          context_package_id: "66666666-6666-4666-8666-666666666666",
          trace: [
            {
              contract_version: "agent_trace_event_public_v1",
              type: "trace",
              id: internalId,
              run_id: internalId,
              sequence_index: 0,
              node: "typed_action_executor",
              status: "completed",
              input_summary: "raw input",
              output_summary: "raw output",
              document_ids: [internalId],
              scores: {
                protocol_version: "internal_protocol_v1",
                state_hash: internalHash,
                raw_payload: { secret: "diagnostic" },
              } as never,
              duration_ms: 1,
              created_at: new Date().toISOString(),
            },
          ],
        }}
        index={0}
        onOpenCitations={vi.fn()}
      />,
    );

    const text = document.body.textContent ?? "";
    expect(text).toContain("示例数据集成");
    expect(text).not.toContain(internalId);
    expect(text).not.toContain(internalHash);
    expect(text).not.toContain("internal_protocol_v1");
    expect(text).not.toContain("raw_payload");
    expect(text).not.toContain("typed_action_executor");
    expect(text).not.toContain("layered_context_graph");
  });

  it("shows a fail-closed warning when a historical citation copy no longer matches the public contract", () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <MessageBubble
          turn={{
            role: "assistant",
            content: "Historical answer",
            route: "layered_context_graph",
            citations: [],
            citation_replay_status: "unavailable",
            citation_replay_reason: "persisted_citation_contract_mismatch",
          }}
          index={0}
          onOpenCitations={vi.fn()}
          defaultContextExpanded={false}
        />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("citation-replay-unavailable").textContent).toContain(
      "来源信息已过期",
    );
  });
});
