// @vitest-environment jsdom

import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { elementHasOverflow, OverflowTooltip } from "./overflow-tooltip";

describe("OverflowTooltip", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows complete text after one second only for visually truncated text", () => {
    vi.useFakeTimers();
    render(
      <div className="kg-text-boundary">
        <div className="glass-panel">
          <p data-testid="truncated">完整的长文本内容</p>
        </div>
        <OverflowTooltip />
      </div>,
    );
    const text = screen.getByTestId("truncated");
    Object.defineProperties(text, {
      clientWidth: { configurable: true, value: 100 },
      scrollWidth: { configurable: true, value: 240 },
      clientHeight: { configurable: true, value: 20 },
      scrollHeight: { configurable: true, value: 20 },
    });
    expect(elementHasOverflow(text)).toBe(true);

    fireEvent.pointerOver(text);
    act(() => vi.advanceTimersByTime(999));
    expect(screen.queryByRole("tooltip")).toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("tooltip").textContent).toBe("完整的长文本内容");
  });

  it("does not replace rich Markdown and LaTeX content with a tooltip", () => {
    vi.useFakeTimers();
    render(
      <div className="kg-text-boundary">
        <div className="markdown-output">
          <p data-testid="markdown">Markdown $E=mc^2$</p>
        </div>
        <OverflowTooltip />
      </div>,
    );
    const text = screen.getByTestId("markdown");
    Object.defineProperties(text, {
      clientWidth: { configurable: true, value: 100 },
      scrollWidth: { configurable: true, value: 240 },
      clientHeight: { configurable: true, value: 20 },
      scrollHeight: { configurable: true, value: 20 },
    });

    fireEvent.pointerOver(text);
    act(() => vi.advanceTimersByTime(1000));
    expect(screen.queryByRole("tooltip")).toBeNull();
  });
});
