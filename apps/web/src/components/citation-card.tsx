"use client";

import type { Citation, CitationBoundingBox } from "@course-kg/shared";
import { ShieldAlert, ShieldCheck } from "lucide-react";

import { MarkdownRenderer } from "@/components/markdown-renderer";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface CitationCardProps {
  citation: Citation;
  index: number;
}

export interface CitationDisplayAudit {
  complete: boolean;
  supported: boolean;
  reasons: string[];
}

const verdictLabels: Record<NonNullable<Citation["verification"]>["verdict"], string> = {
  supported: "supported",
  unsupported: "unsupported",
  contradicted: "contradicted",
  missing_citation: "missing citation",
  structure_context_missing: "structure context missing",
  formula_table_context_missing: "formula / table context missing",
};

function pathText(value?: string | string[] | null): string {
  if (Array.isArray(value)) {
    return value.length ? value.join(" › ") : "root";
  }
  return value || "—";
}

function pairEquals(left?: Array<number | null> | null, right?: Array<number | null> | null): boolean {
  return Array.isArray(left) && Array.isArray(right) && left.length === 2 && right.length === 2 && left[0] === right[0] && left[1] === right[1];
}

function orderedNumericPair(value?: Array<number | null> | null): value is [number, number] {
  return Array.isArray(value) && value.length === 2 && typeof value[0] === "number" && typeof value[1] === "number" && value[0] >= 0 && value[1] >= value[0];
}

function pathEquals(left?: string | string[] | null, right?: string | string[] | null): boolean {
  const normalize = (value?: string | string[] | null) => (Array.isArray(value) ? value : value ? [value] : []);
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}

function bboxEquals(left?: CitationBoundingBox | null, right?: CitationBoundingBox | null): boolean {
  if (!left && !right) {
    return true;
  }
  if (!left || !right) {
    return false;
  }
  return ["page_number", "x0", "y0", "x1", "y1", "coordinate_system", "synthetic", "raw_coordinate_system"].every(
    (key) => left[key as keyof CitationBoundingBox] === right[key as keyof CitationBoundingBox],
  );
}

function nonEmpty(value?: string | null): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function replayVerificationVerdict(
  diagnostics: NonNullable<Citation["verification"]>["diagnostics"],
): NonNullable<Citation["verification"]>["verdict"] {
  const ruleVerdict = diagnostics.rule_verdict;
  const llmVerdict = diagnostics.llm_entailment_verdict ?? ruleVerdict;
  const exactSpanSupported =
    diagnostics.deterministic_exact_span_entailment === true &&
    diagnostics.deterministic_exact_span_entailment_protocol_version ===
      "claim_raw_span_exact_entailment_v1" &&
    diagnostics.llm_entailment_result_present === false &&
    diagnostics.llm_entailment_judge === "skipped_deterministic_exact_span" &&
    ruleVerdict === "supported";
  let replayed: NonNullable<Citation["verification"]>["verdict"];
  if (exactSpanSupported) {
    replayed = "supported";
  } else if (
    ruleVerdict === "missing_citation" ||
    ruleVerdict === "formula_table_context_missing" ||
    ruleVerdict === "structure_context_missing"
  ) {
    replayed = ruleVerdict;
  } else if (
    llmVerdict === "contradicted" ||
    llmVerdict === "unsupported" ||
    llmVerdict === "missing_citation" ||
    llmVerdict === "formula_table_context_missing"
  ) {
    replayed = llmVerdict;
  } else if (
    ruleVerdict === "supported" &&
    diagnostics.llm_entailment_result_present === true &&
    llmVerdict === "supported"
  ) {
    replayed = "supported";
  } else {
    replayed = "unsupported";
  }
  const persistedProvenanceValid =
    diagnostics.citation_provenance_valid === true &&
    diagnostics.citation_provenance_fail_closed === true &&
    diagnostics.citation_provenance_llm_override_allowed === false &&
    diagnostics.citation_provenance_persistence_gate_passed === true;
  return replayed === "supported" && !persistedProvenanceValid
    ? "structure_context_missing"
    : replayed;
}

export function auditCitation(citation: Citation): CitationDisplayAudit {
  const reasons: string[] = [];
  const span = citation.source_span;
  const verification = citation.verification;
  const add = (condition: boolean, reason: string) => {
    if (condition) {
      reasons.push(reason);
    }
  };

  add(citation.contract_version !== "citation_public_v1", "citation contract version is missing or unsupported");
  add(!span || span.contract_version !== "raw_chunk_source_span_v1", "raw source-span contract is missing or unsupported");
  if (!span) {
    return { complete: false, supported: false, reasons };
  }

  add(!nonEmpty(citation.chunk_id) || citation.chunk_id !== span.chunk_id, "citation chunk_id does not match the raw source span");
  add(!nonEmpty(citation.document_id), "document_id is missing");
  add(!nonEmpty(citation.document_version_id), "document_version_id is missing");
  add(Boolean(citation.document_version_id) && citation.document_version_id !== span.document_version_id, "document_version_id does not match the raw source span");
  add(!pairEquals(citation.char_span ?? undefined, span.char_span), "citation char_span is missing or differs from the raw source span");
  add(!pairEquals(citation.page_range ?? undefined, span.page_range), "citation page_range is missing or differs from the raw source span");
  add(!Array.isArray(citation.section_path) || !pathEquals(citation.section_path, span.section_path), "citation section_path is missing or differs from the raw source span");
  add(citation.bbox !== undefined && citation.bbox !== null && !bboxEquals(citation.bbox, span.bbox), "citation bbox differs from the raw source span");
  add(!nonEmpty(citation.source_path) || citation.source_path !== span.source_path, "source_path is missing or differs from the raw source span");
  add(!nonEmpty(citation.logical_source_path) || citation.logical_source_path !== span.logical_source_path, "logical_source_path is missing or differs from the raw source span");
  add(!nonEmpty(citation.context_package_id) || citation.context_package_id !== span.context_package_id, "context_package_id is missing or inconsistent");
  add(!nonEmpty(citation.retrieval_trace_id) || citation.retrieval_trace_id !== span.retrieval_trace_id, "retrieval_trace_id is missing or inconsistent");
  add(!nonEmpty(citation.answer_session_id), "answer_session_id is missing");
  add(!nonEmpty(citation.citation_verification_id) || citation.citation_verification_id !== span.verification_id, "citation verification id is missing or inconsistent");
  add(!Array.isArray(span.char_span) || span.char_span.length !== 2, "raw char_span is missing");
  add(!orderedNumericPair(span.char_span), "raw char_span is not an ordered non-negative pair");
  add(Boolean(span.raw_chunk_char_span) && !orderedNumericPair(span.raw_chunk_char_span ?? undefined), "raw chunk char_span is invalid");
  add(Boolean(span.raw_chunk_char_span) && orderedNumericPair(span.char_span) && orderedNumericPair(span.raw_chunk_char_span ?? undefined) && (span.char_span[0] < span.raw_chunk_char_span![0] || span.char_span[1] > span.raw_chunk_char_span![1]), "raw citation span is outside the raw chunk span");
  add(!Array.isArray(span.page_range) || span.page_range.length !== 2, "raw page_range is missing");
  add(span.section_path === undefined || span.section_path === null, "raw section_path is missing");
  add(span.structure_path === undefined || span.structure_path === null, "raw structure_path is missing");
  add(!Array.isArray(span.structure_node_ids) || span.structure_node_ids.length === 0, "raw structure_node_ids are missing");
  add(span.source_snapshot_verification?.verified !== true, "immutable source snapshot is not verified");
  add(!nonEmpty(span.source_snapshot_verification?.protocol_version) || !nonEmpty(span.source_snapshot_verification?.storage_path), "immutable source snapshot protocol or path is missing");
  add(!nonEmpty(span.source_snapshot_verification?.final_open_protocol_version), "immutable source snapshot final-open protocol is missing");
  add(!nonEmpty(span.source_snapshot_verification?.checksum) || span.source_snapshot_verification?.checksum !== span.source_checksum, "immutable source snapshot checksum is missing or inconsistent");
  add(typeof span.source_snapshot_verification?.size_bytes !== "number" || span.source_snapshot_verification.size_bytes < 0, "immutable source snapshot size is missing or invalid");
  add(!nonEmpty(span.source_checksum) || !nonEmpty(span.chunk_text_hash) || !nonEmpty(span.raw_span_text_hash), "source or span integrity hashes are missing");

  if (!verification) {
    reasons.push("citation verification audit is missing");
    return { complete: false, supported: false, reasons };
  }

  const diagnostics = verification.diagnostics;
  if (!diagnostics || typeof diagnostics !== "object") {
    reasons.push("citation verification diagnostics are missing");
    return { complete: false, supported: false, reasons };
  }
  add(verification.contract_version !== "citation_verification_public_v1", "verification contract version is missing or unsupported");
  add(!Object.prototype.hasOwnProperty.call(verdictLabels, verification.verdict), "verification verdict is missing or unsupported");
  add(!nonEmpty(verification.failure_type) || verification.failure_type === "verification_audit_missing_failure_type", "verification failure_type is missing");
  add(typeof verification.confidence !== "number" || !Number.isFinite(verification.confidence) || verification.confidence < 0 || verification.confidence > 1, "verification confidence is missing or outside [0, 1]");
  add(!["valid", "invalid", "missing"].includes(verification.provenance_status), "provenance status is missing or unsupported");
  add(!["valid", "invalid", "missing"].includes(verification.structure_context_status), "structure-context status is missing or unsupported");
  add(diagnostics.citation_provenance_fail_closed !== true, "provenance audit is not fail-closed");
  add(diagnostics.citation_provenance_llm_override_allowed !== false, "provenance audit allows an LLM override");
  add(diagnostics.citation_provenance_valid !== true, "citation provenance is not valid");
  add(diagnostics.citation_provenance_persistence_gate_passed !== true, "persisted provenance replay did not pass");
  add(!nonEmpty(diagnostics.verification_method), "verification method is missing");
  add(!nonEmpty(diagnostics.claim_grounded_gate_protocol_version), "claim grounded-gate protocol is missing");
  add(!nonEmpty(diagnostics.citation_provenance_protocol_version), "citation provenance protocol is missing");
  add(!nonEmpty(diagnostics.citation_provenance_hash), "citation provenance hash is missing");
  add(!nonEmpty(diagnostics.citation_provenance_session_hash), "citation provenance session hash is missing");
  add(!Array.isArray(diagnostics.citation_provenance_reasons), "citation provenance reasons audit is missing");
  add(
    diagnostics.deterministic_exact_span_entailment === true &&
      diagnostics.deterministic_exact_span_entailment_protocol_version !==
        "claim_raw_span_exact_entailment_v1",
    "deterministic exact-span entailment protocol is missing or unsupported",
  );
  add(
    diagnostics.deterministic_exact_span_entailment === true &&
      (diagnostics.llm_entailment_result_present !== false ||
        diagnostics.llm_entailment_judge !== "skipped_deterministic_exact_span"),
    "deterministic exact-span entailment conflicts with the LLM judge audit",
  );
  add(verification.provenance_status !== "valid", `public provenance status is ${verification.provenance_status}`);

  const expectedStructureStatus =
    verification.verdict === "structure_context_missing" || verification.verdict === "formula_table_context_missing"
      ? "missing"
      : verification.provenance_status === "invalid"
        ? "invalid"
        : verification.provenance_status === "missing"
          ? "missing"
          : "valid";
  add(verification.structure_context_status !== expectedStructureStatus, "structure-context status conflicts with verdict or provenance");
  add(verification.structure_context_status !== "valid", `public structure-context status is ${verification.structure_context_status}`);
  add(verification.verdict === "supported" && verification.failure_type !== "none", "supported verdict carries a non-none failure_type");
  add(verification.verdict !== "supported" && verification.failure_type === "none", "non-supported verdict carries failure_type=none");
  add(
    verification.verdict !== replayVerificationVerdict(diagnostics),
    "verification verdict does not replay from rule, LLM entailment, and persisted provenance diagnostics",
  );
  add(
    verification.verdict === "contradicted" &&
      (diagnostics.llm_entailment_result_present !== true ||
        diagnostics.llm_entailment_verdict !== "contradicted"),
    "contradicted verdict lacks a present contradicted LLM entailment result",
  );
  add(!nonEmpty(citation.claim_id) || !nonEmpty(diagnostics.claim_id), "claim_id binding is missing");
  add(Boolean(citation.claim_id) && Boolean(diagnostics.claim_id) && citation.claim_id !== diagnostics.claim_id, "claim_id differs from verification diagnostics");
  add(!Number.isInteger(citation.claim_index) || !Number.isInteger(diagnostics.claim_index), "claim_index binding is missing");
  add(Number.isInteger(citation.claim_index) && Number.isInteger(diagnostics.claim_index) && citation.claim_index !== diagnostics.claim_index, "claim_index differs from verification diagnostics");
  add(!nonEmpty(citation.claim_text), "claim_text is missing");
  add(!nonEmpty(citation.answer_hash) || !nonEmpty(diagnostics.answer_hash), "answer_hash binding is missing");
  add(Boolean(citation.answer_hash) && Boolean(diagnostics.answer_hash) && citation.answer_hash !== diagnostics.answer_hash, "answer_hash differs from verification diagnostics");

  const complete = reasons.length === 0;
  return { complete, supported: complete && verification.verdict === "supported", reasons };
}

export function CitationCard({ citation, index }: CitationCardProps) {
  const audit = auditCitation(citation);
  const span = citation.source_span;
  const verification = citation.verification;
  const pageRange = span?.page_range ?? citation.page_range;
  const pageLabel =
    Array.isArray(pageRange) &&
    pageRange.length === 2 &&
    typeof pageRange[0] === "number" &&
    typeof pageRange[1] === "number"
      ? pageRange[0] === pageRange[1]
        ? `第 ${pageRange[0]} 页`
        : `第 ${pageRange[0]}–${pageRange[1]} 页`
      : null;
  const displaySnippet = String(
    citation.snippet ?? citation.text ?? "",
  )
    .replace(/^#{1,6}\s+/, "")
    .replace(/\s+#{1,6}\s+/g, "\n\n");
  const sectionLabel = pathText(
    span?.section_path ?? citation.section_path,
  );
  const titleCandidate = String(
    citation.document_title ?? citation.title ?? "",
  ).trim();
  const firstSection = Array.isArray(citation.section_path)
    ? String(citation.section_path[0] ?? "").split(" / ")[0]
    : String(citation.section_path ?? "").split(" / ")[0];
  const sourceTitle =
    titleCandidate &&
    !/^[0-9a-f]{64}$/i.test(titleCandidate) &&
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(titleCandidate)
      ? titleCandidate
      : firstSection || "来源名称缺失";
  const statusLabel = !audit.complete
    ? "来源不可用"
    : audit.supported
      ? "已核验"
      : verification?.verdict === "contradicted"
        ? "与回答冲突"
        : "未支持";

  return (
    <Card data-testid="citation-audit-card" className={cn("border-white/10 bg-white/[0.03] text-white", !audit.complete && "border-rose-300/25 bg-rose-400/[0.035]")}>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-start justify-between gap-3">
          <span className="min-w-0 break-words">{sourceTitle}</span>
          <span className="flex flex-wrap gap-1.5">
            <Badge variant="outline">#{index + 1}</Badge>
            <Badge variant={audit.supported ? "secondary" : "outline"}>{statusLabel}</Badge>
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {!audit.complete ? (
          <div data-testid="citation-fail-closed" role="alert" className="rounded-xl border border-rose-300/30 bg-rose-400/[0.07] p-3 text-xs text-rose-50/80">
            <p className="flex items-center gap-2 font-semibold text-rose-100"><ShieldAlert className="size-4" />这条来源暂时无法核验</p>
            <p className="mt-1.5 leading-5">系统不会把它作为可信引用展示。请重新提问以获取新的来源。</p>
          </div>
        ) : (
          <>
            <MarkdownRenderer content={displaySnippet} className="text-white/68" />
            <div className="flex flex-wrap gap-2 text-xs text-white/52">
              {pageLabel ? <span className="kg-micro-chip rounded-full px-2.5 py-1">{pageLabel}</span> : null}
              {sectionLabel !== "—" ? <span className="kg-micro-chip rounded-full px-2.5 py-1">章节：{sectionLabel}</span> : null}
              {audit.supported ? (
                <span className="inline-flex items-center gap-1 rounded-full border border-emerald-300/20 bg-emerald-300/[0.06] px-2.5 py-1 text-emerald-100/78">
                  <ShieldCheck className="size-3" />
                  内容与回答一致
                </span>
              ) : null}
            </div>
            {citation.claim_text ? <p className="text-xs leading-6 text-white/42">支持的回答内容：{citation.claim_text}</p> : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}
