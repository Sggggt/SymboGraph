// @vitest-environment jsdom

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ParameterName, SETTINGS_PARAMETER_HELP } from "./settings-workspace";

describe("settings parameter help", () => {
  it("renders a delayed tooltip for parameter names", () => {
    render(<ParameterName label="RQ-KMeans 层数" />);

    const trigger = screen.getByText("RQ-KMeans 层数").parentElement;
    const tooltip = screen.getByRole("tooltip");

    expect(trigger?.getAttribute("aria-describedby")).toBe(tooltip.id);
    expect(tooltip.textContent).toContain("L3");
    expect(tooltip.className).toContain("group-hover:delay-500");
    expect(tooltip.className).toContain("group-focus:delay-500");
  });

  it("keeps active runtime and graph parameters documented without legacy BM25 wording", () => {
    for (const label of ["模型请求并发", "中粗层双语派生", "片段 Top K", "跨文档桥最小配额", "引用验证预算", "工作进程并发"]) {
      expect(SETTINGS_PARAMETER_HELP[label]).toBeTruthy();
    }

    expect(Object.values(SETTINGS_PARAMETER_HELP).join("\n")).not.toContain("BM25");
  });
});
