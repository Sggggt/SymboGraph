// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  ContextPackageChunk,
  ContextPackageResponse,
  ContextSelectionReason,
  RetrievalNodeContributionSummary,
  RetrievalPathContribution,
} from "@course-kg/shared";
import { QAContextPackageEvidence } from "@/components/qa-context-package-evidence";
import { fetchContextPackage } from "@/lib/api";
import {
  CONTRACT_HASH,
  makeContextCitationSpan,
  makeContextItem,
  makeContextPackage,
  makeSourceSpan,
  makeSupportRefs,
} from "@/test/public-contract-fixtures";

vi.mock("@/lib/api", () => ({
  fetchContextPackage: vi.fn(),
}));

const packageId = "package:qa-evidence";
const traceId = "trace:qa-evidence";

const pathContribution: RetrievalPathContribution = {
  contract_version: "multi_path_contribution_v2",
  contribution_id: "path-contribution:coarse-mid-chunk",
  layer: "chunk",
  node_id: "chunk:selected",
  parent_layer: "mid",
  parent_node_id: "mid:posterior",
  origin_parent_layer: "coarse",
  origin_parent_node_id: "coarse:bayes",
  root_node_id: "coarse:bayes",
  path: ["coarse:bayes", "mid:posterior", "chunk:selected"],
  path_edge_ids: ["edge:coarse-mid", "edge:mid-chunk"],
  path_edge_types: ["projection", "rq_membership"],
  covered_facets: ["posterior"],
  evidence_roles: ["definition", "example"],
  support_refs: makeSupportRefs({
    support_ids: ["support:raw-span"],
    support_chunk_ids: ["chunk:selected", "chunk:graph-path"],
    edge_ids: ["edge:coarse-mid", "edge:mid-chunk"],
    edge_types: ["projection", "rq_membership"],
  }),
  support_chunk_ids: ["chunk:selected", "chunk:graph-path"],
  distance_so_far: 0.42,
  reward_so_far: 0.08,
};

const whySelected: ContextSelectionReason = {
  roles: ["hit"],
  path_edge_ids: pathContribution.path_edge_ids,
  covered_facets: ["posterior"],
  reason: "accepted_by_priority_queue_graph_traversal",
  reached_by_paths: [pathContribution],
  query_facets: ["posterior", "likelihood"],
  evidence_roles: ["definition", "example"],
  graph_paths: [pathContribution.path],
  graph_path_chunks: ["chunk:graph-path", "chunk:selected"],
  convergence_score: 0.75,
  node_visit_count: 2,
  distinct_parent_count: 1,
  distinct_path_count: 1,
  distinct_edge_type_count: 2,
  parent_node_ids: ["mid:posterior"],
  support_chunk_union: ["chunk:graph-path", "chunk:selected"],
};

const sourceSpan = makeSourceSpan({
  document_version_id: "document-version:qa",
  chunk_id: "chunk:selected",
  source_path: "bayes/chapter.md",
  logical_source_path: "bayes/chapter.md",
  char_span: [110, 188],
  raw_chunk_char_span: [90, 210],
  page_range: [4, 4],
  section_path: ["Bayes", "Posterior"],
  structure_path: ["document", "section:posterior", "paragraph:3"],
  structure_node_ids: ["structure:posterior"],
  context_package_id: packageId,
  retrieval_trace_id: traceId,
  content_token_count: 18,
});

const selectedChunk: ContextPackageChunk = {
  contract_kind: "context_chunk",
  chunk_id: "chunk:selected",
  document_id: "document:qa",
  document_version_id: "document-version:qa",
  document_title: "Bayesian Review",
  source_path: "bayes/chapter.md",
  logical_source_path: "bayes/chapter.md",
  content: "Posterior probability combines prior belief with observed likelihood.",
  content_clipped: false,
  content_token_count: 18,
  original_token_count: 18,
  raw_chunk_char_span: [90, 210],
  chunk_text_hash_protocol_version: "chunk_text_sha256_normalized_v1",
  chunk_text_hash: CONTRACT_HASH,
  raw_span_text_hash_protocol_version: "raw_span_text_sha256_v1",
  raw_span_text_hash: CONTRACT_HASH,
  section_path: ["Bayes", "Posterior"],
  structure_path: ["document", "section:posterior", "paragraph:3"],
  structure_node_ids: ["structure:posterior"],
  structure_nodes: [],
  page_range: [4, 4],
  char_span: [110, 188],
  source_span: sourceSpan,
  structure_closure: {
    previous_chunk_id: "chunk:previous",
    next_chunk_id: "chunk:next",
    parent_section_node_id: "structure:posterior",
    same_page_region_node_ids: ["structure:page-region"],
    table_formula_caption_node_ids: ["structure:formula"],
    code_block_node_ids: [],
    bridge_chunk_ids: ["chunk:bridge"],
    same_page_region: [],
    table_formula_caption: [],
    code_blocks: [],
  },
  why_selected: whySelected,
  dedupe_key: "document-version:qa:[110,188]",
  role: "hit",
  context_package_id: packageId,
};

const nodeContribution: RetrievalNodeContributionSummary = {
  contract_version: "multi_path_contribution_union_v2",
  layer: "chunk",
  node_id: "chunk:selected",
  node_visit_count: 2,
  distinct_parent_count: 1,
  distinct_path_count: 1,
  distinct_edge_type_count: 2,
  parent_node_ids: ["mid:posterior"],
  path_edge_types: ["projection", "rq_membership"],
  covered_facets: ["posterior"],
  evidence_roles: ["definition", "example"],
  support_id_union: ["support:raw-span"],
  support_chunk_union: ["chunk:graph-path", "chunk:selected"],
  cycle_convergence_score: 0.75,
  best_distance: 0.42,
  best_reward: 0.08,
  reached_by_paths: [pathContribution],
};

function evidencePackage(overrides: Partial<ContextPackageResponse> = {}): ContextPackageResponse {
  const selectedContext = makeContextItem({
    chunk_id: selectedChunk.chunk_id,
    document_title: selectedChunk.document_title,
    source_path: selectedChunk.source_path,
    content: selectedChunk.content,
    snippet: selectedChunk.content,
    metadata: {
      source_path: selectedChunk.source_path,
      logical_source_path: selectedChunk.logical_source_path,
      section_path: selectedChunk.section_path,
      structure_path: selectedChunk.structure_path,
      parent_section_node_id: selectedChunk.structure_closure.parent_section_node_id,
      structure_node_ids: selectedChunk.structure_node_ids,
      page_range: selectedChunk.page_range,
      char_span: selectedChunk.char_span,
      source_span: sourceSpan,
      structure_closure: selectedChunk.structure_closure,
      why_selected: selectedChunk.why_selected,
      dedupe_key: selectedChunk.dedupe_key,
      role: selectedChunk.role,
      content_clipped: selectedChunk.content_clipped,
      content_token_count: selectedChunk.content_token_count,
      original_token_count: selectedChunk.original_token_count,
      raw_chunk_char_span: selectedChunk.raw_chunk_char_span,
      context_package_id: packageId,
    },
  });
  return makeContextPackage({
    id: packageId,
    retrieval_trace_id: traceId,
    query: "How does posterior probability use likelihood?",
    hit_chunk_ids: ["chunk:selected"],
    restored_chunk_ids: ["chunk:previous", "chunk:next", "chunk:graph-path"],
    bridge_chunk_ids: ["chunk:bridge"],
    parent_structure_node_ids: ["structure:posterior"],
    concept_path: [
      { layer: "coarse", ids: ["coarse:bayes"] },
      { layer: "mid", ids: ["mid:posterior"] },
      { layer: "chunk", ids: ["chunk:selected"] },
    ],
    graph_path_ids: pathContribution.path_edge_ids,
    reached_by_paths: [pathContribution],
    node_contributions: [nodeContribution],
    why_selected: { "chunk:selected": whySelected },
    cycle_convergence_score: 0.75,
    dedupe_keys: [selectedChunk.dedupe_key],
    covered_facets: ["posterior"],
    package: { contract_version: "context_package_chunks_v1", chunks: [selectedChunk] },
    contexts: [selectedContext],
    token_budget: 512,
    token_count: 18,
    citation_spans: [
      makeContextCitationSpan({
        document_id: "document:qa",
        document_title: "Bayesian Review",
        source_path: "bayes/chapter.md",
        logical_source_path: "bayes/chapter.md",
        section_path: ["Bayes", "Posterior"],
        structure_path: ["document", "section:posterior", "paragraph:3"],
        structure_node_ids: ["structure:posterior"],
        structure_closure: selectedChunk.structure_closure,
        source_span: sourceSpan,
      }),
    ],
    graph_expansion_paths: [
      { kind: "concept_path", path: [{ layer: "coarse", ids: ["coarse:bayes"] }, { layer: "mid", ids: ["mid:posterior"] }, { layer: "chunk", ids: ["chunk:selected"] }] },
      { kind: "graph_path_ids", edge_ids: pathContribution.path_edge_ids },
      { kind: "restored_chunks", chunk_ids: ["chunk:previous", "chunk:next", "chunk:graph-path"] },
      { kind: "bridge_chunks", chunk_ids: ["chunk:bridge"] },
      { kind: "parent_structure_nodes", node_ids: ["structure:posterior"] },
    ],
    package_hash_card: {
      ...makeContextPackage().package_hash_card,
      chunk_count: 1,
      citation_span_count: 1,
      graph_expansion_path_count: 5,
    },
    diagnostics: {
      ...makeContextPackage().diagnostics,
      context_restoration_protocol: "previous_next_structure_bridge_v1",
      conversation_state_scope_hash: "9".repeat(64),
      path_summary: {
        node_visit_count: 2,
        distinct_parent_count: 1,
        distinct_path_count: 1,
        distinct_edge_type_count: 2,
        covered_facets: ["posterior"],
        support_chunk_union: ["chunk:graph-path", "chunk:selected"],
        reached_by_paths: [pathContribution],
        cycle_convergence_score: 0.75,
      },
      dedupe_keys: [selectedChunk.dedupe_key],
      restore_counts: {
        hit_chunks: 1,
        restored_chunks: 3,
        bridge_chunks: 1,
        graph_path_chunks: 1,
        parent_structure_nodes: 1,
        per_hit_chunk_budget: 4,
      },
      token_budget_audit: {
        token_budget: 512,
        token_count: 18,
        within_budget: true,
        clipped_chunk_ids: [],
        skipped_chunk_ids: [],
        packing_protocol: "bounded_context_pack_v1",
      },
      snapshot_integrity: {
        protocol_version: "immutable_context_snapshot_v1",
        verified_document_version_count: 1,
        fail_closed: true,
      },
    },
    ...overrides,
  });
}

function renderEvidence(props: Partial<React.ComponentProps<typeof QAContextPackageEvidence>> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <QAContextPackageEvidence packageId={packageId} traceId={traceId} defaultExpanded {...props} />
    </QueryClientProvider>,
  );
}

describe("QAContextPackageEvidence", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("fetches the answer Context Package and renders chunks, raw spans, structure closure, paths, why-selected, and dedupe audits", async () => {
    vi.mocked(fetchContextPackage).mockResolvedValue(evidencePackage());

    renderEvidence();

    await waitFor(() => expect(fetchContextPackage).toHaveBeenCalledWith(packageId));
    const body = await screen.findByTestId("qa-context-package-body");
    expect(body.textContent).toContain("selected hit chunks");
    expect(body.textContent).toContain("Posterior probability combines prior belief");
    expect(body.textContent).toContain("raw citation span");
    expect(body.textContent).toContain("[110, 188]");
    expect(body.textContent).toContain("chunk:previous");
    expect(body.textContent).toContain("chunk:next");
    expect(body.textContent).toContain("chunk:bridge");
    expect(body.textContent).toContain("accepted_by_priority_queue_graph_traversal");
    expect(body.textContent).toContain("path-contribution:coarse-mid-chunk");
    expect(body.textContent).toContain("coarse:bayes");
    expect(body.textContent).toContain("Context package de-duplication");
    expect(body.textContent).toContain("chunk_id unique 1/1");
    expect(body.textContent).toContain("gray model calls = 0");
    expect(body.textContent).toContain("LLM gray authority = false");
    expect(screen.getAllByTestId("context-source-span").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByTestId("context-reached-by-paths").length).toBeGreaterThanOrEqual(2);
  });

  it("loads older evidence packages only after the user expands the audit", async () => {
    vi.mocked(fetchContextPackage).mockResolvedValue(evidencePackage());

    renderEvidence({ defaultExpanded: false });

    expect(fetchContextPackage).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Context Package 证据视图/i }));
    await waitFor(() => expect(fetchContextPackage).toHaveBeenCalledWith(packageId));
    expect(await screen.findByTestId("qa-context-package-body")).toBeTruthy();
  });

  it("fails closed when a retrieval answer has no context_package_id", () => {
    renderEvidence({ packageId: null, retrievalExpected: true, defaultExpanded: false });

    const failure = screen.getByTestId("qa-context-package-failure");
    expect(failure.textContent).toContain("Context Package 证据视图缺失");
    expect(failure.textContent).toContain("不能以裸检索命中");
    expect(fetchContextPackage).not.toHaveBeenCalled();
  });

  it("rejects a package whose retrieval trace does not belong to the answer", async () => {
    vi.mocked(fetchContextPackage).mockResolvedValue(evidencePackage({ retrieval_trace_id: "trace:foreign" }));

    renderEvidence();

    const failure = await screen.findByTestId("qa-context-package-failure");
    expect(failure.textContent).toContain("Context Package 身份或权限审计失败");
    expect(failure.textContent).toContain("does not match answer trace");
    expect(screen.queryByText(/Posterior probability combines prior belief/)).toBeNull();
  });

  it.each([
    ["source path", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.source_path = "forged.md"; }],
    ["logical source path", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.logical_source_path = "forged.md"; }],
    ["text hash", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.chunk_text_hash = "b".repeat(64); }],
    ["raw span hash", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.raw_span_text_hash = "c".repeat(64); }],
    ["char span", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.char_span = [111, 188]; }],
    ["raw chunk span", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.raw_chunk_char_span = [91, 210]; }],
    ["page range", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.page_range = [5, 5]; }],
    ["section path", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.section_path = ["Forged"]; }],
    ["structure path", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.structure_path = ["document", "forged"]; }],
    ["structure nodes", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.structure_node_ids = ["structure:forged"]; }],
    ["content clipping", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.content_clipped = true; }],
    ["content token count", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.content_token_count = 999; }],
  ])("fails closed on package chunk/source-span %s drift", async (_label, mutate) => {
    const payload = structuredClone(evidencePackage());
    mutate(payload);
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    expect(await screen.findByTestId("qa-context-package-failure")).toBeTruthy();
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });

  it("fails closed when package/source facts are forged together but context and citation projections retain the canonical address", async () => {
    const payload = structuredClone(evidencePackage());
    payload.package.chunks[0].source_path = "forged.md";
    payload.package.chunks[0].source_span.source_path = "forged.md";
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    const failure = await screen.findByTestId("qa-context-package-failure");
    expect(failure.textContent).toContain("context item provenance mismatch");
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });

  it("fails closed when context metadata and its source span are forged together", async () => {
    const payload = structuredClone(evidencePackage());
    payload.contexts[0].source_path = "forged.md";
    payload.contexts[0].metadata.source_path = "forged.md";
    payload.contexts[0].metadata.source_span.source_path = "forged.md";
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    expect(await screen.findByTestId("qa-context-package-failure")).toBeTruthy();
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });

  it("fails closed when citation fields and their nested source span are forged together", async () => {
    const payload = structuredClone(evidencePackage());
    payload.citation_spans[0].source_path = "forged.md";
    payload.citation_spans[0].source_span.source_path = "forged.md";
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    expect(await screen.findByTestId("qa-context-package-failure")).toBeTruthy();
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });

  it("fails closed when all nested trace ids are synchronized to a trace foreign to the answer", async () => {
    const payload = structuredClone(evidencePackage());
    payload.retrieval_trace_id = "trace:foreign";
    payload.package.chunks[0].source_span.retrieval_trace_id = "trace:foreign";
    payload.contexts[0].metadata.source_span.retrieval_trace_id = "trace:foreign";
    payload.citation_spans[0].source_span.retrieval_trace_id = "trace:foreign";
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    const failure = await screen.findByTestId("qa-context-package-failure");
    expect(failure.textContent).toContain("does not match answer trace");
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });

  it("fails closed when package/context/citation projections lose one-to-one chunk coverage", async () => {
    const payload = structuredClone(evidencePackage());
    payload.citation_spans = [];
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    const failure = await screen.findByTestId("qa-context-package-failure");
    expect(failure.textContent).toContain("one-to-one coverage");
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });

  it.each([
    ["context snippet", (payload: ContextPackageResponse) => { payload.contexts[0].snippet = "forged snippet"; }],
    ["context structure closure", (payload: ContextPackageResponse) => { payload.contexts[0].metadata.structure_closure = { ...structuredClone(payload.contexts[0].metadata.structure_closure), bridge_chunk_ids: ["chunk:forged"] }; }],
    ["context why-selected", (payload: ContextPackageResponse) => { payload.contexts[0].metadata.why_selected = { ...structuredClone(payload.contexts[0].metadata.why_selected), reason: "forged" }; }],
    ["context dedupe key", (payload: ContextPackageResponse) => { payload.contexts[0].metadata.dedupe_key = "forged"; }],
    ["context role", (payload: ContextPackageResponse) => { payload.contexts[0].metadata.role = "bridge"; }],
    ["context original token count", (payload: ContextPackageResponse) => { payload.contexts[0].metadata.original_token_count += 1; }],
    ["context parent section identity", (payload: ContextPackageResponse) => { payload.contexts[0].metadata.parent_section_node_id = "structure:forged"; }],
    ["citation structure closure", (payload: ContextPackageResponse) => { payload.citation_spans[0].structure_closure = { ...structuredClone(payload.citation_spans[0].structure_closure), next_chunk_id: "chunk:forged" }; }],
    ["top-level why-selected", (payload: ContextPackageResponse) => { payload.why_selected["chunk:selected"] = { ...structuredClone(payload.why_selected["chunk:selected"]), reason: "forged" }; }],
    ["top-level dedupe key", (payload: ContextPackageResponse) => { payload.dedupe_keys = ["forged"]; }],
    ["diagnostic dedupe key", (payload: ContextPackageResponse) => { payload.diagnostics.dedupe_keys = ["forged"]; }],
    ["source checksum", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.source_checksum = "b".repeat(64); }],
    ["snapshot checksum", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.source_snapshot_verification.checksum = "b".repeat(64); }],
    ["snapshot verification", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.source_snapshot_verification.verified = false as true; }],
    ["snapshot size", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.source_snapshot_verification.size_bytes = -1; }],
    ["source-span contract", (payload: ContextPackageResponse) => { payload.package.chunks[0].source_span.contract_version = "forged" as "raw_chunk_source_span_v1"; }],
    ["hash-card count", (payload: ContextPackageResponse) => { payload.package_hash_card.chunk_count = 99; }],
    ["hash-card public hash", (payload: ContextPackageResponse) => { payload.package_hash_card.public_payload_hash = "not-a-hash"; }],
  ])("fails closed on exact package/context/citation %s projection drift", async (_label, mutate) => {
    const payload = structuredClone(evidencePackage());
    mutate(payload);
    vi.mocked(fetchContextPackage).mockResolvedValue(payload);

    renderEvidence();

    expect(await screen.findByTestId("qa-context-package-failure")).toBeTruthy();
    expect(screen.queryByTestId("qa-context-package-body")).toBeNull();
  });
});
