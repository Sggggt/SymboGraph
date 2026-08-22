import { describe, expect, it } from "vitest";
import type { AgentTraceEventPayload, AgentTraceScores } from "@course-kg/shared";

import { contextGraphTraceFallbackSteps, groupTraceEvents, traceAuditSummary, traceNodeLabel } from "./agent-trace";

function traceScores(overrides: Partial<AgentTraceScores> = {}): AgentTraceScores {
  return {
    contract_version: "agent_trace_scores_public_v1",
    audit_kind: "retrieval_stage",
    query_rq_path: [],
    chunk_ids: [],
    ...overrides,
  } as AgentTraceScores;
}

function traceEvent(node: AgentTraceEventPayload["node"]): AgentTraceEventPayload {
  return {
    contract_version: "agent_trace_event_public_v1",
    type: "trace",
    run_id: "run:test",
    sequence_index: 0,
    node,
    status: "completed",
    input_summary: "",
    output_summary: "",
    document_ids: [],
    scores: traceScores(),
    duration_ms: 1,
  };
}

describe("agent trace helpers", () => {
  it("uses four-layer P&E fallback steps", () => {
    expect(contextGraphTraceFallbackSteps).toEqual([
      "query_understanding",
      "query_facet_extraction",
      "agent_planner",
      "typed_action_validation",
      "entry_selection",
      "layer_drilldown",
      "frontier_traversal",
      "chunk_recall",
      "structure_context_restoration",
      "context_package",
      "grounded_answer",
      "citation_verification",
      "reward_event",
    ]);
  });

  it("labels context graph nodes in Chinese", () => {
    expect(traceNodeLabel("agent_planner")).toBe("智能体规划");
    expect(traceNodeLabel("query_facet_extraction")).toBe("查询 facets");
    expect(traceNodeLabel("typed_action_validation")).toBe("动作校验");
    expect(traceNodeLabel("entry_selection")).toBe("分阶段入口");
    expect(traceNodeLabel("layer_drilldown")).toBe("逐父下钻");
    expect(traceNodeLabel("frontier_traversal")).toBe("队列遍历");
    expect(traceNodeLabel("structure_context_restoration")).toBe("结构上下文恢复");
    expect(traceNodeLabel("retrievers")).toBe("片段召回");
  });

  it("groups trace events by QA workflow stage", () => {
    expect(
      groupTraceEvents([
        traceEvent("entry_selection"),
        traceEvent("frontier_traversal"),
        traceEvent("citation_verification"),
      ]).map((group) => group.label),
    ).toEqual(["分阶段入口", "队列遍历", "引用验证"]);
  });

  it("summarizes context graph audit scores including RQ path", () => {
    const retrievalSummary = traceAuditSummary({
        contract_version: "agent_trace_scores_public_v1",
        audit_kind: "retrieval_stage",
        coarse_entries: 2,
        stage_queue_count: 3,
        frontier_pops: 5,
        dominance_pruned_count: 2,
        query_rq_path: [1, 2, 3],
        chunk_ids: [],
      });
    const contextSummary = traceAuditSummary({
        contract_version: "agent_trace_scores_public_v1",
        audit_kind: "context_restoration",
        hit_chunks: 10,
        restored_chunks: 6,
        context_package_id: "pkg-1",
      });
    expect([...retrievalSummary, ...contextSummary]).toEqual(["粗入口: 2", "Stage 队列: 3", "Frontier pop: 5", "支配剪枝: 2", "RQ 路径: 1/2/3", "命中片段: 10", "恢复片段: 6", "证据包: pkg-1"]);
  });

  it("renders retrieval granularity in Chinese", () => {
    expect(traceAuditSummary(traceScores({ retrieval_granularity: "mid" }))).toContain("检索模式: 普通模式");
    expect(traceAuditSummary(traceScores({ retrieval_granularity: "coarse" }))).toContain("检索模式: 摘要模式");
  });
});
