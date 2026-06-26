// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RetrievalGranularity } from "@course-kg/shared";
import { AGENT_PARAMETER_HELP, AgentParameterName, RetrievalGranularitySelector } from "./qa-workspace";

function SelectorHarness() {
  const [value, setValue] = useState<RetrievalGranularity>("mid");
  return <RetrievalGranularitySelector value={value} onChange={setValue} />;
}

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
      "证据包令牌预算",
      "结果保留数量默认值",
      "粗概念总预算",
      "每个粗概念中概念预算",
      "中概念保留数量",
      "每个中概念片段预算",
      "片段保留数量",
      "候选去重池预算",
      "每层最大深度",
      "每节点标签上限",
      "边复用上限",
      "闭环奖励上限",
      "闭环奖励距离阈值",
      "路径绿色阈值",
      "路径灰区阈值",
      "路径硬中断阈值",
      "结构恢复预算",
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
});
