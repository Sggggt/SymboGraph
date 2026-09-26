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
  it("uses the retrieval controller for current requests", () => {
    expect(contextGraphTraceFallbackSteps).toEqual(["retrieval_control"]);
    expect(traceNodeLabel("retrieval_control")).toBe("检索与回答进度");
  });

  it("labels context graph nodes in Chinese", () => {
    expect(traceNodeLabel("agent_planner")).toBe("智能体规划");
    expect(traceNodeLabel("query_facet_extraction")).toBe("查询 facets");
    expect(traceNodeLabel("typed_action_validation")).toBe("动作校验");
    expect(traceNodeLabel("entry_selection")).toBe("分阶段入口");
    expect(traceNodeLabel("layer_drilldown")).toBe("逐父下钻");
    expect(traceNodeLabel("frontier_traversal")).toBe("队列遍历");
    expect(traceNodeLabel("structure_context_restoration")).toBe("结构上下文恢复");
    expect(traceNodeLabel("evidence_directory_ready")).toBe("准备证据目录");
    expect(traceNodeLabel("evidence_read")).toBe("读取候选原文");
    expect(traceNodeLabel("evidence_finalized")).toBe("冻结回答证据");
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

  it("renders the model-selected target entry layer", () => {
    const summary = traceAuditSummary({
      contract_version: "agent_trace_scores_public_v1",
      audit_kind: "intent_execution",
      entry_layer: "coarse",
      model_call_count: 1,
      score_fields_used: [],
    });
    expect(summary).toContain("入口层: 粗概念");
    expect(summary).toContain("模型调用: 1");
  });

  it("summarizes finite evidence selection without exposing handles", () => {
    const summary = traceAuditSummary({
      contract_version: "agent_trace_scores_public_v1",
      audit_kind: "intent_execution",
      source_count: 12,
      read_count: 2,
      selected_count: 1,
      remaining_count: 10,
      decision_call_count: 2,
      read_action_count: 1,
      mid_count: 4,
      mandatory_source_count: 1,
      estimated_input_tokens: 900,
      input_token_count: 850,
    });
    expect(summary).toEqual([
      "候选来源: 12",
      "本轮读取: 2",
      "已选来源: 1",
      "剩余节点: 10",
      "证据决策: 2",
      "读取动作: 1",
      "语义节点: 4",
      "必带来源: 1",
      "估算输入 token: 900",
      "实际输入 token: 850",
    ]);
  });
});
