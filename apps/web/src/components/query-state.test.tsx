// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { LoadingBlock } from "./query-state";

describe("LoadingBlock", () => {
  afterEach(() => cleanup());

  it("uses one spinner instead of a loading card or skeleton", () => {
    const { container } = render(<LoadingBlock rows={4} />);

    expect(screen.getByRole("status").textContent).toContain("正在加载");
    expect(container.querySelector(".animate-spin")).toBeTruthy();
    expect(container.querySelector('[data-slot="card"]')).toBeNull();
    expect(container.querySelector('[data-slot="skeleton"]')).toBeNull();
  });
});
