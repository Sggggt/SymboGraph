// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
    expect(stream.textContent).toContain("正在执行：动作执行");
    expect(stream.textContent).toContain("实时");
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
