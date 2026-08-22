// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Citation } from "@course-kg/shared";
import { auditCitation, CitationCard } from "./citation-card";

const HASH = "a".repeat(64);

function makeCitation(overrides: Partial<Citation> = {}): Citation {
  const citation: Citation = {
    contract_version: "citation_public_v1",
    chunk_id: "chunk:citation_surface",
    citation_index: 1,
    claim_id: HASH,
    claim_index: 0,
    claim_text: "A grounded claim.",
    answer_hash: HASH,
    document_id: "document:citation_surface",
    document_version_id: "document-version:citation_surface",
    document_title: "Citation audit fixture",
    source_path: "citation_surface.md",
    logical_source_path: "citation_surface.md",
    section: ["Evidence"],
    section_path: ["Evidence"],
    page_number: 2,
    page_range: [2, 2],
    char_span: [10, 42],
    context_package_id: "package:citation_surface",
    bbox: {
      page_number: 2,
      x0: 1,
      y0: 2,
      x1: 3,
      y1: 4,
      coordinate_system: "pdf_points",
    },
    snippet: "A grounded claim.",
    source_span: {
      contract_version: "raw_chunk_source_span_v1",
      document_version_id: "document-version:citation_surface",
      chunk_id: "chunk:citation_surface",
      source_path: "citation_surface.md",
      source_checksum: HASH,
      logical_source_path: "citation_surface.md",
      source_snapshot_verification: {
        protocol_version: "immutable_source_snapshot_v1",
        final_open_protocol_version: "posix_openat_nofollow_fstat_v1",
        storage_path: "snapshots/citation_surface.md",
        checksum: HASH,
        verified: true,
        size_bytes: 128,
      },
      chunk_text_hash_protocol_version: "chunk_text_sha256_normalized_v1",
      chunk_text_hash: HASH,
      raw_span_text_hash_protocol_version: "raw_span_text_sha256_v1",
      raw_span_text_hash: HASH,
      char_span: [10, 42],
      raw_chunk_char_span: [0, 80],
      page_range: [2, 2],
      section_path: ["Evidence"],
      structure_path: ["Document", "Evidence"],
      structure_node_ids: ["structure:citation_surface"],
      bbox: {
        page_number: 2,
        x0: 1,
        y0: 2,
        x1: 3,
        y1: 4,
        coordinate_system: "pdf_points",
      },
      context_package_id: "package:citation_surface",
      retrieval_trace_id: "trace:citation_surface",
      verification_id: "verification:citation_surface",
      content_clipped: false,
      content_token_count: 8,
    },
    verification: {
      contract_version: "citation_verification_public_v1",
      verdict: "supported",
      failure_type: "none",
      provenance_status: "valid",
      structure_context_status: "valid",
      confidence: 0.97,
      diagnostics: {
        verification_method: "claim_structure_plus_llm_entailment_v2",
        claim_grounded_gate_protocol_version: "claim_level_grounded_gate_v1",
        claim_id: HASH,
        claim_index: 0,
        answer_hash: HASH,
        citation_provenance_protocol_version: "citation_provenance_v1",
        citation_provenance_valid: true,
        citation_provenance_hash: HASH,
        citation_provenance_reasons: [],
        citation_provenance_fail_closed: true,
        citation_provenance_llm_override_allowed: false,
        citation_provenance_session_hash: HASH,
        citation_provenance_persistence_gate_passed: true,
        llm_entailment_judge: "completed",
        rule_verdict: "supported",
        llm_entailment_verdict: "supported",
        llm_entailment_result_present: true,
        llm_entailment_reason: "raw span entails the claim",
        citation_prompt_protocol_hash: HASH,
        citation_grounding_envelope_protocol_version: "raw_span_only_citation_grounding_envelope_v2",
        citation_grounding_envelope_hash: HASH,
        citation_profile_hash: HASH,
        reason: "raw span entails the claim",
      },
    },
    retrieval_trace_id: "trace:citation_surface",
    answer_session_id: "answer:citation_surface",
    citation_verification_id: "verification:citation_surface",
  };
  return { ...citation, ...overrides };
}

describe("CitationCard", () => {
  afterEach(() => cleanup());

  it("shows a concise natural-language source without internal audit identifiers", () => {
    const citation = makeCitation();
    render(<CitationCard citation={citation} index={0} />);

    const card = screen.getByTestId("citation-audit-card");
    expect(card.textContent).toContain("已核验");
    expect(card.textContent).toContain("第 2 页");
    expect(card.textContent).toContain("章节：Evidence");
    expect(card.textContent).toContain("内容与回答一致");
    expect(card.textContent).toContain("A grounded claim.");
    expect(screen.queryByTestId("citation-fail-closed")).toBeNull();
    expect(screen.queryByTestId("citation-verification-audit")).toBeNull();
    expect(screen.queryByTestId("citation-raw-address")).toBeNull();
    expect(card.textContent).not.toContain("chunk:citation_surface");
    expect(card.textContent).not.toContain("trace:citation_surface");
    expect(card.textContent).not.toContain(HASH);
    expect(card.textContent).not.toContain("protocol");
  });

  it("does not label a missing source name as a local file", () => {
    const citation = makeCitation({
      document_title: HASH,
      title: HASH,
      section_path: [],
    });

    render(<CitationCard citation={citation} index={0} />);
    const card = screen.getByTestId("citation-audit-card");
    expect(card.textContent).toContain("来源名称缺失");
    expect(card.textContent).not.toContain("本地文件");
    expect(card.textContent).not.toContain("本地资料");
  });

  it.each([
    ["unsupported", "unsupported_claim"],
    ["contradicted", "contradicted_claim"],
  ] as const)("shows an actual %s verdict and failure without calling it supported", (verdict, failureType) => {
    const citation = makeCitation({
      verification: {
        ...makeCitation().verification!,
        verdict,
        failure_type: failureType,
        confidence: 0.2,
        diagnostics: {
          ...makeCitation().verification!.diagnostics,
          llm_entailment_verdict: verdict,
          reason: "claim is not entailed by the raw span",
        },
      },
    });

    const audit = auditCitation(citation);
    expect(audit.complete).toBe(true);
    expect(audit.supported).toBe(false);

    render(<CitationCard citation={citation} index={0} />);
    expect(screen.getByTestId("citation-audit-card").textContent).toContain(
      verdict === "contradicted" ? "与回答冲突" : "未支持",
    );
    expect(screen.getByTestId("citation-audit-card").textContent).not.toContain(failureType);
    expect(screen.queryByTestId("citation-fail-closed")).toBeNull();
  });

  it("replays deterministic exact-span support without pretending an LLM result exists", () => {
    const base = makeCitation();
    const citation = makeCitation({
      verification: {
        ...base.verification!,
        diagnostics: {
          ...base.verification!.diagnostics,
          llm_entailment_judge: "skipped_deterministic_exact_span",
          llm_entailment_result_present: false,
          llm_entailment_reason: null,
          deterministic_exact_span_entailment: true,
          deterministic_exact_span_entailment_protocol_version:
            "claim_raw_span_exact_entailment_v1",
        },
      },
    });

    expect(auditCitation(citation)).toMatchObject({ complete: true, supported: true });
    render(<CitationCard citation={citation} index={0} />);
    const card = screen.getByTestId("citation-audit-card");
    expect(card.textContent).toContain("已核验");
    expect(card.textContent).not.toContain("claim_raw_span_exact_entailment_v1");

    const tampered = makeCitation({
      verification: {
        ...citation.verification!,
        diagnostics: {
          ...citation.verification!.diagnostics,
          deterministic_exact_span_entailment_protocol_version: "unknown",
        },
      },
    });
    expect(auditCitation(tampered).complete).toBe(false);
  });

  it("accepts a page-less source when the outer citation omits an empty bbox", () => {
    const base = makeCitation();
    const citation = makeCitation({
      bbox: null,
      page_number: null,
      page_range: [null, null],
      source_span: {
        ...base.source_span,
        page_range: [null, null],
        bbox: {
          page_number: null,
          x0: null,
          y0: null,
          x1: null,
          y1: null,
          coordinate_system: null,
        },
      },
    });

    expect(auditCitation(citation)).toMatchObject({
      complete: true,
      supported: true,
    });
  });

  it("fails closed when verification identity or provenance is missing or inconsistent", () => {
    const base = makeCitation();
    const citation = makeCitation({
      citation_verification_id: "verification:forged",
      source_span: {
        ...base.source_span,
        retrieval_trace_id: "trace:other",
      },
      verification: {
        ...base.verification!,
        provenance_status: "missing",
        structure_context_status: "missing",
        diagnostics: {
          ...base.verification!.diagnostics,
          citation_provenance_valid: null,
          citation_provenance_persistence_gate_passed: null,
        },
      },
    });

    const audit = auditCitation(citation);
    expect(audit.complete).toBe(false);
    expect(audit.reasons).toContain("retrieval_trace_id is missing or inconsistent");
    expect(audit.reasons).toContain("citation verification id is missing or inconsistent");

    render(<CitationCard citation={citation} index={0} />);
    const failure = screen.getByTestId("citation-fail-closed");
    expect(failure.textContent).toContain("这条来源暂时无法核验");
    expect(failure.textContent).toContain("不会把它作为可信引用");
  });

  it("fails closed for an unknown legacy verdict even when the rest of the card is complete", () => {
    const base = makeCitation();
    const citation = makeCitation({
      verification: {
        ...base.verification!,
        verdict: "legacy_verified" as NonNullable<Citation["verification"]>["verdict"],
      },
    });

    const audit = auditCitation(citation);
    expect(audit.complete).toBe(false);
    expect(audit.reasons).toContain("verification verdict is missing or unsupported");

    render(<CitationCard citation={citation} index={0} />);
    expect(screen.getByTestId("citation-fail-closed")).toBeTruthy();
    expect(screen.getByTestId("citation-audit-card").textContent).toContain("来源不可用");
    expect(screen.getByTestId("citation-audit-card").textContent).not.toContain("legacy_verified");
  });

  it("fails closed when the outer verdict cannot be replayed from rule and LLM diagnostics", () => {
    const base = makeCitation();
    const citation = makeCitation({
      verification: {
        ...base.verification,
        verdict: "contradicted",
        failure_type: "contradicted_claim",
        diagnostics: {
          ...base.verification.diagnostics,
          rule_verdict: "supported",
          llm_entailment_verdict: "supported",
          llm_entailment_result_present: true,
        },
      },
    });

    const audit = auditCitation(citation);
    expect(audit.complete).toBe(false);
    expect(audit.reasons).toContain(
      "verification verdict does not replay from rule, LLM entailment, and persisted provenance diagnostics",
    );
    expect(audit.reasons).toContain(
      "contradicted verdict lacks a present contradicted LLM entailment result",
    );
  });
});
