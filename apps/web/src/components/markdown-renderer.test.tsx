// @vitest-environment jsdom

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownRenderer } from "./markdown-renderer";

describe("MarkdownRenderer", () => {
  it("renders generated GFM structure instead of flattening it into prose", () => {
    const { container } = render(
      <MarkdownRenderer content={"## Method\n\n- First step\n- **Second step**\n\n| Case | Value |\n| --- | --- |\n| A | 1 |"} />,
    );

    expect(container.querySelector("h2")?.textContent).toBe("Method");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("strong")?.textContent).toBe("Second step");
    expect(container.querySelector("table")).not.toBeNull();
  });

  it("renders inline and block LaTeX formulas", () => {
    const { container } = render(<MarkdownRenderer content={"Inline $E=mc^2$.\n\n$$\n\\int_0^1 x^2 dx\n$$"} />);

    expect(container.querySelector(".katex")).not.toBeNull();
    expect(container.querySelector(".katex-display")).not.toBeNull();
    expect(container.textContent).toContain("E");
  });

  it("renders TeX parenthesis and bracket delimiters from retrieved snippets", () => {
    const { container } = render(<MarkdownRenderer content={"Inline \\(a^2+b^2=c^2\\).\n\n\\[\\sum_{i=1}^n i\\]"} />);

    expect(container.querySelectorAll(".katex").length).toBeGreaterThanOrEqual(2);
    expect(container.querySelector(".katex-display")).not.toBeNull();
  });
});
