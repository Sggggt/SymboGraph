// @vitest-environment jsdom

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SearchResult } from "@course-kg/shared";
import { ResultRow, resultDisplayTitle } from "./search-workspace";

describe("Search product surface", () => {
  afterEach(() => cleanup());

  it("shows source evidence while hiding internal ids, hashes, protocols, and trace metrics", () => {
    const internalId = "77777777-7777-4777-8777-777777777777";
    const internalHash = "f".repeat(64);
    const result: SearchResult = {
      chunk_id: internalId,
      document_title: internalHash,
      partition: internalHash,
      source_path: "C:/unit-test/synthetic-source.md",
      snippet: "# 公开合成知识库\n示例数据集成用于汇集并统一处理多个公开输入。",
      score: 0.987,
      citations: [],
      metadata: {
        traversal: {
          path: [internalId],
          path_edge_ids: [internalId],
          distance_so_far: 0.123,
          reward_so_far: 0.456,
          covered_facets: ["示例数据集成"],
          evidence_roles: ["typed_action_forced_entry"],
          protocol_version: "internal_protocol_v1",
          state_hash: internalHash,
          raw_payload: { diagnostic: true },
        },
        rq: {
          candidate_rq_path: [1, 2, 3],
          residual_distance: 0.1,
        },
      },
    };

    const { container } = render(
      <ResultRow
        result={result}
        active={false}
        index={0}
        onHover={vi.fn()}
        onSelect={vi.fn()}
      />,
    );

    const text = container.textContent ?? "";
    expect(text).toContain("公开合成知识库");
    expect(text).toContain("示例数据集成");
    expect(text).not.toContain(internalId);
    expect(text).not.toContain(internalHash);
    expect(text).not.toContain("internal_protocol_v1");
    expect(text).not.toContain("typed_action_forced_entry");
    expect(text).not.toContain("0.123");
    expect(text).not.toContain("0.456");
    expect(text).not.toContain("C:/unit-test");
  });

  it("does not invent a local-file label when source identity is missing", () => {
    const result = {
      chunk_id: "chunk-1",
      document_title: "f".repeat(64),
      snippet: "没有标题标记的命中片段。",
      score: 0.5,
      citations: [],
      metadata: {},
    } as SearchResult;

    expect(resultDisplayTitle(result)).toBe("来源名称缺失");
    expect(resultDisplayTitle(result)).not.toContain("本地");
  });
});
