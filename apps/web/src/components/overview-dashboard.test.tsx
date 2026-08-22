// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { OverviewDashboard } from "./overview-dashboard";

const fetchDashboard = vi.fn();
const fetchGraph = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchDashboard: (...args: unknown[]) => fetchDashboard(...args),
  fetchGraph: (...args: unknown[]) => fetchGraph(...args),
}));

vi.mock("@/components/knowledge-base-context", () => ({
  useKnowledgeBaseContext: () => ({ selectedKnowledgeBaseId: "sample-kb" }),
}));

vi.mock("@/components/network-canvas", () => ({
  NetworkCanvas: () => <div data-testid="network-canvas" />,
}));

describe("OverviewDashboard", () => {
  beforeEach(() => {
    fetchDashboard.mockReset();
    fetchGraph.mockReset();
  });

  it("renders compact overview facts while the detailed graph is still loading", async () => {
    fetchDashboard.mockResolvedValue({
      knowledge_base: {
        id: "sample-kb",
        name: "Sample",
        document_count: 4,
        chunk_count: 275,
        active_chunk_count: 275,
      },
      tree: [],
      graph: { freshness: { is_stale: false }, nodes: [], edges: [] },
      batch_status: null,
      ingested_document_count: 4,
      chunk_count: 275,
      graph_relation_count: 0,
      coverage_by_source_type: {},
      degraded_mode: false,
      context_graph: {
        counts: { active_chunks: 275, chunk_relation_edges: 3792 },
      },
    });
    fetchGraph.mockReturnValue(new Promise(() => undefined));
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <OverviewDashboard />
      </QueryClientProvider>,
    );

    expect(
      await screen.findByText("本地知识图谱、向量检索与 RAG 联动。"),
    ).toBeTruthy();
    expect(screen.getByText("图谱正在加载，概览数据已就绪")).toBeTruthy();
    expect(screen.getByText("275")).toBeTruthy();
    expect(screen.getByText("3792")).toBeTruthy();
    expect(screen.queryByTestId("network-canvas")).toBeNull();
    await waitFor(() => {
      expect(fetchDashboard).toHaveBeenCalledWith("sample-kb", {
        includeGraph: false,
      });
      expect(fetchGraph).toHaveBeenCalledWith(
        "sample-kb",
        "chunk-relation",
        "overview",
      );
    });
  });
});
