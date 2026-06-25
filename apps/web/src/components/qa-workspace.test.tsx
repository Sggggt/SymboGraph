// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RetrievalGranularity } from "@course-kg/shared";
import { RetrievalGranularitySelector } from "./qa-workspace";

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
