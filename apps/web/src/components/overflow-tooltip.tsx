"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type TooltipState = {
  text: string;
  left: number;
  top: number;
  width: number;
};

const TEXT_CANDIDATE_SELECTOR = [
  "span:not(:has(*))",
  "div:not(:has(*))",
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "li",
  "td",
  "th",
  "label",
  "button",
  "a",
  "[data-overflow-text]",
  ".truncate",
  '[class*="line-clamp-"]',
].join(",");

export function elementHasOverflow(element: HTMLElement): boolean {
  return (
    element.scrollWidth > element.clientWidth + 1
    || element.scrollHeight > element.clientHeight + 1
  );
}

function overflowCandidate(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof Element)) {
    return null;
  }
  const scope = target.closest(".kg-text-boundary");
  if (!scope) {
    return null;
  }
  let candidate = target.closest(TEXT_CANDIDATE_SELECTOR) as HTMLElement | null;
  while (candidate && scope.contains(candidate)) {
    if (
      !candidate.closest(".markdown-output")
      && !candidate.closest('[role="tooltip"]')
      && candidate.dataset.overflowTooltip !== "off"
      && elementHasOverflow(candidate)
      && candidate.textContent?.trim()
    ) {
      return candidate;
    }
    const parent = candidate.parentElement;
    candidate = parent?.closest(TEXT_CANDIDATE_SELECTOR) as HTMLElement | null;
  }
  return null;
}

export function OverflowTooltip() {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const targetRef = useRef<EventTarget | null>(null);

  useEffect(() => {
    const bodyHadBoundary = document.body.classList.contains("kg-text-boundary");
    // Radix/Base UI surfaces render through document.body portals, outside the
    // AppShell subtree. Give those dialogs and drawers the same text policy.
    document.body.classList.add("kg-text-boundary");
    const cancelHide = () => {
      if (hideTimerRef.current) {
        clearTimeout(hideTimerRef.current);
        hideTimerRef.current = null;
      }
    };
    const clear = () => {
      cancelHide();
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      targetRef.current = null;
      setTooltip(null);
    };
    const handlePointerOver = (event: PointerEvent) => {
      if (event.target instanceof Element && event.target.closest("[data-overflow-tooltip-surface]")) {
        cancelHide();
        return;
      }
      clear();
      targetRef.current = event.target;
      timerRef.current = setTimeout(() => {
        const candidate = overflowCandidate(targetRef.current);
        if (!candidate) {
          return;
        }
        const rect = candidate.getBoundingClientRect();
        const gutter = 16;
        const width = Math.min(520, Math.max(240, window.innerWidth - gutter * 2));
        setTooltip({
          text: candidate.textContent?.trim() ?? "",
          left: Math.max(
            gutter,
            Math.min(rect.left, window.innerWidth - width - gutter),
          ),
          top: Math.min(rect.bottom + 10, window.innerHeight - 120),
          width,
        });
      }, 1000);
    };
    const handlePointerOut = (event: PointerEvent) => {
      const related = event.relatedTarget;
      if (related instanceof Element && related.closest("[data-overflow-tooltip-surface]")) {
        cancelHide();
        return;
      }
      if (related instanceof Node && event.target instanceof Node) {
        const source = (event.target as Element).closest?.(TEXT_CANDIDATE_SELECTOR);
        if (source?.contains(related)) {
          return;
        }
      }
      cancelHide();
      hideTimerRef.current = setTimeout(clear, 120);
    };
    document.addEventListener("pointerover", handlePointerOver);
    document.addEventListener("pointerout", handlePointerOut);
    window.addEventListener("scroll", clear, true);
    window.addEventListener("resize", clear);
    return () => {
      clear();
      if (!bodyHadBoundary) {
        document.body.classList.remove("kg-text-boundary");
      }
      document.removeEventListener("pointerover", handlePointerOver);
      document.removeEventListener("pointerout", handlePointerOut);
      window.removeEventListener("scroll", clear, true);
      window.removeEventListener("resize", clear);
    };
  }, []);

  if (!tooltip || typeof document === "undefined") {
    return null;
  }
  return createPortal(
    <div
      role="tooltip"
      data-overflow-tooltip-surface
      data-testid="overflow-tooltip"
      className="fixed z-[9999] max-h-[70vh] overflow-y-auto whitespace-pre-wrap break-words rounded-2xl border border-cyan-100/20 bg-[#081322]/98 px-4 py-3 text-sm leading-6 text-cyan-50/86 shadow-2xl shadow-black/35 backdrop-blur-xl"
      style={{ left: tooltip.left, top: tooltip.top, width: tooltip.width }}
    >
      {tooltip.text}
    </div>,
    document.body,
  );
}
