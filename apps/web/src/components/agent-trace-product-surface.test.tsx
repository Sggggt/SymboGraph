// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AgentTraceEventPayload } from "@course-kg/shared";
import { AgentTraceStream, sanitizeTraceDisplayText } from "./agent-trace-stream";
import { GeneratingBubble, MessageBubble } from "./qa-workspace";

const INTERNAL_ID = "77777777-7777-4777-8777-777777777777";
const INTERNAL_HASH = "a".repeat(64);

function makeTrace(): AgentTraceEventPayload[] {
  return [
    {
      id: INTERNAL_ID,
      run_id: INTERNAL_ID,
      sequence_index: 0,
      node: "typed_action_executor",
      status: "completed",
      input_summary: `trace=${INTERNAL_ID}, path=/app/data/source_slots/${INTERNAL_HASH}.md`,
      output_summary: `observation=${INTERNAL_HASH}, chunks=8`,
      document_ids: [INTERNAL_ID],
      scores: {
        audit_kind: "typed_action_executor",
        plan_id: INTERNAL_ID,
        plan_index: 0,
        retrieval_trace_id: INTERNAL_ID,
      },
      duration_ms: 42,
    } as AgentTraceEventPayload,
  ];
}

describe("QA layered trace product surface", () => {
  afterEach(() => cleanup());

  it("keeps the layered interaction while redacting internal identities and raw JSON", () => {
    render(<AgentTraceStream trace={makeTrace()} defaultExpanded compact />);

    const stream = screen.getByTestId("agent-trace-stream");
    expect(stream.textContent).toContain("流式轨迹");
    expect(stream.textContent).toContain("1 个步骤");
    expect(stream.textContent).toContain("动作执行");
    expect(stream.textContent).toContain("证据观察已记录");
    expect(stream.textContent).not.toContain(INTERNAL_ID);
    expect(stream.textContent).not.toContain(INTERNAL_HASH);
    expect(stream.textContent).not.toContain("/app/data");
    expect(stream.textContent).not.toContain("scores");
    expect(stream.textContent).not.toContain("document / chunk ids");

    fireEvent.click(screen.getByRole("button", { name: /步骤信息/ }));
    expect(stream.textContent).toContain("规划轮次");
    expect(stream.textContent).not.toContain(INTERNAL_ID);

    fireEvent.click(screen.getByRole("button", { name: /具体信息/ }));
    expect(screen.getByTestId("agent-trace-fine-details").textContent).toContain("本步输入");
    expect(screen.getByTestId("agent-trace-fine-details").textContent).toContain("本步结果");
    expect(screen.getByTestId("agent-trace-fine-details").textContent).toContain("内部路径");
    expect(screen.getByTestId("agent-trace-fine-details").textContent).not.toContain(INTERNAL_ID);
    expect(screen.getByTestId("agent-trace-fine-details").textContent).not.toContain(INTERNAL_HASH);
  });

  it("renders the trajectory for historical assistant messages", () => {
    render(
      <MessageBubble
        turn={{ role: "assistant", content: "有证据支撑的回答。", trace: makeTrace() }}
        index={0}
        onOpenCitations={vi.fn()}
      />,
    );

    expect(screen.getByTestId("agent-trace-stream")).toBeTruthy();
    expect(screen.getByText("有证据支撑的回答。")).toBeTruthy();
  });

  it("expands the live trajectory while the agent is running", () => {
    render(<GeneratingBubble content="" trace={makeTrace()} />);

    const stream = screen.getByTestId("agent-trace-stream");
    expect(stream.textContent).toContain("已完成：动作执行");
    expect(stream.textContent).not.toContain("实时");
  });

  it("shows the current run phase and elapsed time while a model call has no completed trace event", async () => {
    render(<GeneratingBubble content="" trace={[]} currentNode="intent_planning" startedAt={new Date(Date.now() - 65_000).toISOString()} />);

    const stream = screen.getByTestId("agent-trace-stream");
    expect(stream.textContent).toContain("当前阶段：理解问题并规划检索");
    await waitFor(() => expect(stream.textContent).toContain("已运行 1 分"), { timeout: 2000 });
    expect(stream.textContent).toContain("0 个步骤");
    expect(stream.textContent).toContain("当前阶段完成后会显示第一条轨迹事件");
  });

  it("shows evidence reading as the active phase without inventing a completed step", () => {
    render(<GeneratingBubble content="" trace={[]} currentNode="evidence_read" startedAt={new Date().toISOString()} />);

    const stream = screen.getByTestId("agent-trace-stream");
    expect(stream.textContent).toContain("当前阶段：阅读并筛选原文");
    expect(stream.textContent).toContain("0 个步骤");
  });

  it("shows actual coarse reads and schema feedback before the completed plan", () => {
    const stages = [
      { node: "planning_resource_titles", output_summary: "已读取 3 个粗节点标题", scores: { audit_kind: "intent_execution", planning_round: 1, resource_mode: "titles", coarse_node_count: 3, model_duration_ms: 1200, local_read_duration_ms: 23 } },
      { node: "planning_resource_details", output_summary: "已阅读 2 个粗节点详情", scores: { audit_kind: "intent_execution", planning_round: 2, resource_mode: "details", coarse_node_count: 2, model_duration_ms: 900, local_read_duration_ms: 14 } },
      { node: "planning_schema_feedback", output_summary: "计划格式未通过校验，已请求模型重提", scores: { audit_kind: "intent_execution", planning_round: 3, schema_feedback_error_count: 1 } },
      { node: "intent_planning", output_summary: "规划已冻结", scores: { audit_kind: "intent_execution", model_call_count: 4 } },
    ] as const;
    const events = stages.map((item, index) => ({
      contract_version: "agent_trace_event_public_v1", type: "trace", id: `unit-test-step-${index}`,
      run_id: "unit-test-run", sequence_index: index, status: "completed",
      input_summary: "", document_ids: [], duration_ms: 10,
      ...item,
      scores: { contract_version: "agent_trace_scores_public_v1", ...item.scores },
    })) as AgentTraceEventPayload[];

    render(<AgentTraceStream trace={events} defaultExpanded compact />);
    const stream = screen.getByTestId("agent-trace-stream");
    const visible = stream.querySelector("ol")?.textContent ?? "";
    expect(visible.indexOf("查看粗节点标题")).toBeLessThan(visible.indexOf("阅读粗节点摘要"));
    expect(visible.indexOf("阅读粗节点摘要")).toBeLessThan(visible.indexOf("修正规划格式"));
    expect(visible.indexOf("修正规划格式")).toBeLessThan(visible.indexOf("理解任务并冻结计划"));
    expect(stream.textContent).toContain("4 个步骤");
  });

  it("sanitizes ids, hashes, and storage paths in summaries", () => {
    const text = sanitizeTraceDisplayText(
      `trace=${INTERNAL_ID} observation=${INTERNAL_HASH} /app/data/private.md`,
    );

    expect(text).toContain("检索已记录");
    expect(text).toContain("证据观察已记录");
    expect(text).toContain("内部路径");
    expect(text).not.toContain(INTERNAL_ID);
    expect(text).not.toContain(INTERNAL_HASH);
  });
});
