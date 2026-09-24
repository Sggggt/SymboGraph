// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  cleanupLegacyQaPayloadStorage,
  useLocalStorage,
} from "./use-local-storage";

function StoredInput({ storageKey }: { storageKey: string }) {
  const [value, setValue] = useLocalStorage(storageKey, "");
  return (
    <input
      aria-label="question"
      value={value}
      onChange={(event) => setValue(event.target.value)}
    />
  );
}

describe("useLocalStorage quota recovery", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("cleans obsolete QA payloads across knowledge bases without removing durable controls", () => {
    window.localStorage.setItem("qa.turns.old-kb", "large-turns");
    window.localStorage.setItem("qa.trace.other-kb", "large-trace");
    window.localStorage.setItem("qa.sessionId.old-kb", "session-1");
    window.localStorage.setItem("qa.activeStream.old-kb", "run-1");
    window.localStorage.setItem("qa.question.old-kb", "draft");
    window.localStorage.setItem("knowledgeBase.selectedId", "old-kb");

    const removed = cleanupLegacyQaPayloadStorage();

    expect(new Set(removed)).toEqual(new Set([
      "qa.turns.old-kb",
      "qa.trace.other-kb",
    ]));
    expect(window.localStorage.getItem("qa.sessionId.old-kb")).toBe("session-1");
    expect(window.localStorage.getItem("qa.activeStream.old-kb")).toBe("run-1");
    expect(window.localStorage.getItem("qa.question.old-kb")).toBe("draft");
    expect(window.localStorage.getItem("knowledgeBase.selectedId")).toBe("old-kb");
  });

  it("removes legacy payloads and retries a quota-blocked new knowledge-base draft", () => {
    const storageKey = "qa.question.new-kb-quota-retry";
    const legacyKey = "qa.citations.old-kb";
    const sessionKey = "qa.sessionId.old-kb";
    window.localStorage.setItem(legacyKey, "legacy-payload");
    window.localStorage.setItem(sessionKey, "session-1");
    const originalSetItem = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function setItem(this: Storage, key, value) {
      if (key === storageKey && window.localStorage.getItem(legacyKey) !== null) {
        throw new DOMException("Storage quota exceeded", "QuotaExceededError");
      }
      return originalSetItem.call(this, key, value);
    });

    render(<StoredInput storageKey={storageKey} />);
    fireEvent.change(screen.getByRole("textbox", { name: "question" }), {
      target: { value: "新资料库问题" },
    });

    expect((screen.getByRole("textbox", { name: "question" }) as HTMLInputElement).value).toBe("新资料库问题");
    expect(JSON.parse(window.localStorage.getItem(storageKey) ?? "null")).toBe("新资料库问题");
    expect(window.localStorage.getItem(legacyKey)).toBeNull();
    expect(window.localStorage.getItem(sessionKey)).toBe("session-1");
  });

  it("keeps the controlled input editable when storage remains unavailable", () => {
    const storageKey = "qa.question.new-kb-memory-fallback";
    const originalSetItem = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function setItem(this: Storage, key, value) {
      if (key === storageKey) {
        throw new DOMException("Storage quota exceeded", "QuotaExceededError");
      }
      return originalSetItem.call(this, key, value);
    });
    const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    render(<StoredInput storageKey={storageKey} />);
    const input = screen.getByRole("textbox", { name: "question" });
    fireEvent.change(input, { target: { value: "仍然可以输入" } });
    fireEvent.change(input, { target: { value: "仍然可以继续输入" } });

    expect((input as HTMLInputElement).value).toBe("仍然可以继续输入");
    expect(window.localStorage.getItem(storageKey)).toBeNull();
    expect(warning).toHaveBeenCalledTimes(1);
  });
});
