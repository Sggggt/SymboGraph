"use client";

import { useState } from "react";
import type {
  CitationSourceSpan,
  ContextCitationSpan,
  ContextGraphExpansionPath,
  ContextPackageChunk,
  ContextPackageResponse,
  ContextSelectionReason,
  ContextStructureClosure,
  ContextStructureNodeAudit,
  RetrievalNodeContributionSummary,
  RetrievalPathContribution,
} from "@course-kg/shared";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, FileCheck2, Network, ShieldAlert, ShieldCheck } from "lucide-react";

import { LoadingBlock } from "@/components/query-state";
import { fetchContextPackage } from "@/lib/api";
import { cn } from "@/lib/utils";

type ApiError = Error & {
  status?: number;
  structured?: { code?: string; title?: string; message?: string };
};

const roleLabels: Record<ContextPackageChunk["role"], string> = {
  hit: "selected hit",
  restored_context: "previous / next context",
  bridge: "bridge context",
  graph_path: "accepted graph path",
};

function pathText(value?: string | string[] | null): string {
  if (Array.isArray(value)) {
    return value.length ? value.join(" › ") : "—";
  }
  return value || "—";
}

function spanText(value?: Array<number | null> | null): string {
  return value?.length ? `[${value.map((item) => item ?? "null").join(", ")}]` : "—";
}

function numberText(value?: number | null): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "") : "—";
}

function EvidenceFailure({ title, message }: { title: string; message: string }) {
  return (
    <div
      data-testid="qa-context-package-failure"
      role="alert"
      className="mt-4 rounded-2xl border border-rose-300/35 bg-rose-400/[0.065] p-4 text-sm text-rose-50/88"
    >
      <div className="flex items-center gap-2 font-semibold text-rose-100">
        <ShieldAlert className="size-4" />
        {title}
      </div>
      <p className="mt-2 leading-6 text-rose-100/68">{message}</p>
    </div>
  );
}

function EvidenceIdList({ title, values, empty = "无" }: { title: string; values?: string[]; empty?: string }) {
  return (
    <div className="rounded-xl border border-white/8 bg-black/10 p-3">
      <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">{title}</p>
      {values?.length ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {values.map((value, index) => (
            <code key={`${value}-${index}`} className="rounded-md bg-white/[0.045] px-2 py-1 text-[11px] text-cyan-50/68">
              {value}
            </code>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-white/36">{empty}</p>
      )}
    </div>
  );
}

function AuditBadge({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px]",
        ok ? "border-emerald-300/20 bg-emerald-300/[0.055] text-emerald-100/72" : "border-rose-300/30 bg-rose-400/[0.07] text-rose-100/78",
      )}
    >
      {ok ? <ShieldCheck className="size-3" /> : <ShieldAlert className="size-3" />}
      {children}
    </span>
  );
}

function StructureNodeList({ title, nodes }: { title: string; nodes: ContextStructureNodeAudit[] }) {
  if (!nodes.length) {
    return null;
  }
  return (
    <div>
      <p className="text-[11px] uppercase tracking-[0.15em] text-white/38">{title}</p>
      <div className="mt-2 grid gap-2">
        {nodes.map((node) => (
          <div key={`${title}-${node.node_id}-${node.mapping_role}`} className="rounded-lg border border-white/7 bg-black/10 p-2.5 text-xs text-white/56">
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              <code className="text-cyan-100/66">{node.node_id}</code>
              <span>{node.node_type}</span>
              {node.title ? <span>{node.title}</span> : null}
              <span>page {node.page_number ?? "—"}</span>
              <span>role {node.mapping_role}</span>
              <span>coverage {numberText(node.coverage_ratio)}</span>
              <span>weight {numberText(node.mapping_weight)}</span>
            </div>
            <p className="mt-1 break-all text-white/38">path {node.path || "—"} · mapping {node.mapping_protocol_version}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function StructureClosureView({ closure }: { closure: ContextStructureClosure }) {
  const parentNodes = closure.parent_section ? [closure.parent_section] : [];
  return (
    <div className="grid gap-3 rounded-xl border border-violet-200/10 bg-violet-300/[0.025] p-3" data-testid="context-structure-closure">
      <div className="grid gap-2 sm:grid-cols-3">
        <EvidenceIdList title="previous chunk" values={closure.previous_chunk_id ? [closure.previous_chunk_id] : []} />
        <EvidenceIdList title="next chunk" values={closure.next_chunk_id ? [closure.next_chunk_id] : []} />
        <EvidenceIdList title="parent section" values={closure.parent_section_node_id ? [closure.parent_section_node_id] : []} />
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <EvidenceIdList title="bridge chunks" values={closure.bridge_chunk_ids} />
        <EvidenceIdList title="same-page regions" values={closure.same_page_region_node_ids} />
        <EvidenceIdList title="table / formula / caption" values={closure.table_formula_caption_node_ids} />
        <EvidenceIdList title="code blocks" values={closure.code_block_node_ids} />
      </div>
      <StructureNodeList title="parent section audit" nodes={parentNodes} />
      <StructureNodeList title="same-page region audit" nodes={closure.same_page_region} />
      <StructureNodeList title="table / formula / caption audit" nodes={closure.table_formula_caption} />
      <StructureNodeList title="code block audit" nodes={closure.code_blocks} />
    </div>
  );
}

function SourceSpanView({ sourceSpan }: { sourceSpan: CitationSourceSpan }) {
  const bbox = sourceSpan.bbox;
  return (
    <div className="grid gap-2 rounded-xl border border-white/8 bg-black/10 p-3 text-xs text-white/54" data-testid="context-source-span">
      <div className="grid gap-2 sm:grid-cols-2">
        <p>document version <code className="break-all text-cyan-100/62">{sourceSpan.document_version_id}</code></p>
        <p>chunk <code className="break-all text-cyan-100/62">{sourceSpan.chunk_id}</code></p>
        <p>raw citation span <code className="text-white/72">{spanText(sourceSpan.char_span)}</code></p>
        <p>raw chunk span <code className="text-white/72">{spanText(sourceSpan.raw_chunk_char_span ?? undefined)}</code></p>
        <p>page range <code className="text-white/72">{spanText(sourceSpan.page_range)}</code></p>
        <p>section <code className="text-white/72">{pathText(sourceSpan.section_path)}</code></p>
        <p>structure path <code className="text-white/72">{pathText(sourceSpan.structure_path)}</code></p>
        <p>content {sourceSpan.content_clipped ? "clipped" : "complete"} · {sourceSpan.content_token_count ?? 0} tokens</p>
      </div>
      <p className="break-all">source <code className="text-white/68">{sourceSpan.logical_source_path || sourceSpan.source_path}</code></p>
      <p className="break-all">source checksum <code className="text-white/48">{sourceSpan.source_checksum}</code></p>
      <p className="break-all">chunk text hash <code className="text-white/48">{sourceSpan.chunk_text_hash}</code> · {sourceSpan.chunk_text_hash_protocol_version}</p>
      <p className="break-all">raw span hash <code className="text-white/48">{sourceSpan.raw_span_text_hash}</code> · {sourceSpan.raw_span_text_hash_protocol_version}</p>
      <p className="break-all">
        immutable snapshot {sourceSpan.source_snapshot_verification.verified ? "verified" : "invalid"} · {sourceSpan.source_snapshot_verification.protocol_version} · {sourceSpan.source_snapshot_verification.storage_path} · {sourceSpan.source_snapshot_verification.size_bytes} bytes
      </p>
      <p className="break-all">snapshot checksum <code className="text-white/48">{sourceSpan.source_snapshot_verification.checksum}</code></p>
      <EvidenceIdList title="structure node ids" values={sourceSpan.structure_node_ids} />
      {bbox ? (
        <p>
          bbox page {bbox.page_number ?? "—"} · [{bbox.x0 ?? "—"}, {bbox.y0 ?? "—"}, {bbox.x1 ?? "—"}, {bbox.y1 ?? "—"}] · {bbox.coordinate_system || "coordinate system unavailable"}
        </p>
      ) : (
        <p>bbox —</p>
      )}
      <div className="grid gap-1 sm:grid-cols-3">
        <p>package <code className="break-all">{sourceSpan.context_package_id || "—"}</code></p>
        <p>trace <code className="break-all">{sourceSpan.retrieval_trace_id || "—"}</code></p>
        <p>verification <code className="break-all">{sourceSpan.verification_id || "—"}</code></p>
      </div>
    </div>
  );
}

function PathContributionList({ title, paths }: { title: string; paths: RetrievalPathContribution[] }) {
  return (
    <div data-testid="context-reached-by-paths">
      <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">{title}</p>
      {paths.length ? (
        <div className="mt-2 grid gap-2">
          {paths.map((path) => (
            <div key={path.contribution_id} className="rounded-lg border border-cyan-200/10 bg-cyan-300/[0.025] p-3 text-xs text-white/55">
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                <code className="text-cyan-100/72">{path.contribution_id}</code>
                <span>layer {path.layer} · node {path.node_id}</span>
                <span>root {path.root_node_id}</span>
                <span>parent {path.parent_layer || "—"} · {path.parent_node_id || "—"}</span>
                <span>distance {numberText(path.distance_so_far)}</span>
                <span>reward {numberText(path.reward_so_far)}</span>
              </div>
              <p className="mt-2 break-all text-white/48">path {path.path.join(" → ") || "—"}</p>
              <p className="mt-1 break-all text-white/48">edges {path.path_edge_ids.join(" → ") || "—"}</p>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <EvidenceIdList title="covered facets" values={path.covered_facets} />
                <EvidenceIdList title="evidence roles" values={path.evidence_roles} />
                <EvidenceIdList title="support ids" values={path.support_refs.support_ids} />
                <EvidenceIdList title="support chunks" values={path.support_chunk_ids} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-white/34">无 retained path contribution</p>
      )}
    </div>
  );
}

function SelectionReasonView({ reason }: { reason: ContextSelectionReason }) {
  return (
    <div className="grid gap-3 rounded-xl border border-amber-200/10 bg-amber-300/[0.025] p-3" data-testid="context-why-selected">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-white/55">
        <span>reason <strong className="text-amber-50/72">{reason.reason}</strong></span>
        <span>visits {reason.node_visit_count}</span>
        <span>parents {reason.distinct_parent_count}</span>
        <span>paths {reason.distinct_path_count}</span>
        <span>edge types {reason.distinct_edge_type_count}</span>
        <span>convergence {numberText(reason.convergence_score)}</span>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <EvidenceIdList title="selection roles" values={reason.roles} />
        <EvidenceIdList title="query facets" values={reason.query_facets} />
        <EvidenceIdList title="covered facets" values={reason.covered_facets} />
        <EvidenceIdList title="evidence roles" values={reason.evidence_roles} />
        <EvidenceIdList title="path edge ids" values={reason.path_edge_ids} />
        <EvidenceIdList title="graph path chunks" values={reason.graph_path_chunks} />
        <EvidenceIdList title="parent nodes" values={reason.parent_node_ids} />
        <EvidenceIdList title="support chunk union" values={reason.support_chunk_union} />
      </div>
      <div>
        <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">graph paths</p>
        {reason.graph_paths.length ? reason.graph_paths.map((path, index) => <code key={`${path.join(":")}-${index}`} className="mt-1 block break-all text-xs text-white/52">{path.join(" → ")}</code>) : <p className="mt-1 text-xs text-white/34">无</p>}
      </div>
      <PathContributionList title="reached by paths" paths={reason.reached_by_paths} />
    </div>
  );
}

function ChunkEvidenceCard({ chunk, index }: { chunk: ContextPackageChunk; index: number }) {
  return (
    <article data-testid="context-package-chunk" className="rounded-2xl border border-white/10 bg-white/[0.025] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.17em] text-cyan-100/48">#{index + 1} · {roleLabels[chunk.role]}</p>
          <h5 className="mt-1 text-sm font-semibold text-white/82">{chunk.document_title || "Untitled document"}</h5>
          <code className="mt-1 block break-all text-xs text-cyan-100/58">{chunk.chunk_id}</code>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <AuditBadge ok={!chunk.content_clipped}>{chunk.content_clipped ? "content clipped" : "content complete"}</AuditBadge>
          <span className="rounded-full border border-white/10 px-2.5 py-1 text-[11px] text-white/46">{chunk.content_token_count}/{chunk.original_token_count} tokens</span>
        </div>
      </div>
      <div className="mt-3 rounded-xl border border-white/8 bg-black/15 p-3 text-sm leading-7 text-white/68 whitespace-pre-wrap">{chunk.content}</div>
      <div className="mt-3 grid gap-2 text-xs text-white/48 sm:grid-cols-2">
        <p className="break-all">source <code className="text-white/65">{chunk.logical_source_path || chunk.source_path}</code></p>
        <p>role <code className="text-white/65">{chunk.role}</code></p>
        <p>section <code className="text-white/65">{pathText(chunk.section_path)}</code></p>
        <p>structure <code className="text-white/65">{pathText(chunk.structure_path)}</code></p>
        <p>page <code className="text-white/65">{spanText(chunk.page_range)}</code></p>
        <p>packed span <code className="text-white/65">{spanText(chunk.char_span)}</code></p>
        <p>raw chunk span <code className="text-white/65">{spanText(chunk.raw_chunk_char_span)}</code></p>
        <p className="break-all">dedupe key <code className="text-white/65">{chunk.dedupe_key}</code></p>
      </div>
      <div className="mt-3 grid gap-3">
        <SourceSpanView sourceSpan={chunk.source_span} />
        <StructureNodeList title="mapped structure nodes" nodes={chunk.structure_nodes} />
        <StructureClosureView closure={chunk.structure_closure} />
        <SelectionReasonView reason={chunk.why_selected} />
      </div>
    </article>
  );
}

function CitationSpanView({ citation, index }: { citation: ContextCitationSpan; index: number }) {
  return (
    <article data-testid="context-citation-span" className="rounded-xl border border-emerald-200/10 bg-emerald-300/[0.025] p-3">
      <div className="flex flex-wrap justify-between gap-2 text-xs text-white/54">
        <strong className="text-emerald-50/72">citation span #{index + 1} · {citation.document_title}</strong>
        <code className="break-all">{citation.source_span.chunk_id}</code>
      </div>
      <p className="mt-2 text-xs text-white/45">section {pathText(citation.section_path)} · structure {pathText(citation.structure_path)}</p>
      <div className="mt-3 grid gap-3">
        <SourceSpanView sourceSpan={citation.source_span} />
        <StructureClosureView closure={citation.structure_closure} />
      </div>
    </article>
  );
}

function NodeContributionView({ contribution }: { contribution: RetrievalNodeContributionSummary }) {
  return (
    <article className="rounded-xl border border-cyan-200/10 bg-cyan-300/[0.02] p-3" data-testid="context-node-contribution">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-white/55">
        <strong className="text-cyan-50/74">layer {contribution.layer} · node {contribution.node_id}</strong>
        <span>visits {contribution.node_visit_count}</span>
        <span>parents {contribution.distinct_parent_count}</span>
        <span>paths {contribution.distinct_path_count}</span>
        <span>edge types {contribution.distinct_edge_type_count}</span>
        <span>convergence {numberText(contribution.cycle_convergence_score)}</span>
        <span>best distance {numberText(contribution.best_distance)}</span>
        <span>best reward {numberText(contribution.best_reward)}</span>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <EvidenceIdList title="parent nodes" values={contribution.parent_node_ids} />
        <EvidenceIdList title="path edge types" values={contribution.path_edge_types} />
        <EvidenceIdList title="covered facets" values={contribution.covered_facets} />
        <EvidenceIdList title="evidence roles" values={contribution.evidence_roles} />
        <EvidenceIdList title="support ids" values={contribution.support_id_union} />
        <EvidenceIdList title="support chunks" values={contribution.support_chunk_union} />
      </div>
      <div className="mt-3">
        <PathContributionList title="retained path contributions" paths={contribution.reached_by_paths} />
      </div>
    </article>
  );
}

function GraphExpansionView({ expansion }: { expansion: ContextGraphExpansionPath }) {
  if (expansion.kind === "concept_path") {
    return (
      <div className="rounded-lg border border-white/8 bg-black/10 p-3 text-xs text-white/52">
        <strong className="text-white/68">concept path</strong>
        {expansion.path.map((entry, index) => <code key={`${entry.layer}-${index}`} className="mt-1 block break-all">{entry.layer}: {entry.ids.join(" → ") || "—"}</code>)}
      </div>
    );
  }
  const values = expansion.kind === "graph_path_ids" ? expansion.edge_ids : expansion.kind === "parent_structure_nodes" ? expansion.node_ids : expansion.chunk_ids;
  return <EvidenceIdList title={expansion.kind.replaceAll("_", " ")} values={values} />;
}

function DedupeAudit({ evidencePackage }: { evidencePackage: ContextPackageResponse }) {
  const chunks = evidencePackage.package.chunks;
  const chunkIds = chunks.map((chunk) => chunk.chunk_id);
  const citationAddresses = evidencePackage.citation_spans.map((citation) => `${citation.source_span.document_version_id}:${spanText(citation.source_span.char_span)}`);
  const packageKeys = evidencePackage.dedupe_keys ?? [];
  const diagnosticKeys = evidencePackage.diagnostics.dedupe_keys;
  const sameKeys = JSON.stringify([...packageKeys].sort()) === JSON.stringify([...diagnosticKeys].sort());
  const statuses = [
    { label: `chunk_id unique ${new Set(chunkIds).size}/${chunkIds.length}`, ok: new Set(chunkIds).size === chunkIds.length },
    { label: `citation address unique ${new Set(citationAddresses).size}/${citationAddresses.length}`, ok: new Set(citationAddresses).size === citationAddresses.length },
    { label: `dedupe key unique ${new Set(packageKeys).size}/${packageKeys.length}`, ok: new Set(packageKeys).size === packageKeys.length },
    { label: "package / diagnostics keys agree", ok: sameKeys },
  ];
  return (
    <div data-testid="context-dedupe-audit" className="grid gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div>
        <p className="text-[11px] uppercase tracking-[0.17em] text-white/38">Context package de-duplication</p>
        <p className="mt-1 text-xs leading-5 text-white/44">同一 chunk 只打包一次；citation 以 document version + raw char span 审计，路径贡献保留在 summary 中而不复制正文。</p>
      </div>
      <div className="flex flex-wrap gap-2">
        {statuses.map((status) => <AuditBadge key={status.label} ok={status.ok}>{status.label}</AuditBadge>)}
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <EvidenceIdList title="package dedupe keys" values={packageKeys} />
        <EvidenceIdList title="diagnostics dedupe keys" values={diagnosticKeys} />
      </div>
    </div>
  );
}

function canonicalAuditValue(value: unknown): unknown {
  if (value === undefined) return undefined;
  if (Array.isArray(value)) return value.map(canonicalAuditValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, nested]) => nested !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, canonicalAuditValue(nested)]),
    );
  }
  return value;
}

function sameAuditValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonicalAuditValue(left)) === JSON.stringify(canonicalAuditValue(right));
}

const sha256Pattern = /^[0-9a-f]{64}$/;
const contextPackageHashedFields = [
  "contract_version",
  "id",
  "retrieval_trace_id",
  "knowledge_base_id",
  "query",
  "hit_chunk_ids",
  "restored_chunk_ids",
  "bridge_chunk_ids",
  "parent_structure_node_ids",
  "concept_path",
  "graph_path_ids",
  "reached_by_paths",
  "node_contributions",
  "why_selected",
  "cycle_convergence_score",
  "dedupe_keys",
  "covered_facets",
  "package",
  "contexts",
  "token_budget",
  "token_count",
  "citation_spans",
  "graph_expansion_paths",
  "diagnostics",
];

function sourceSpanIntegrityErrors(span: CitationSourceSpan, path: string): string[] {
  const errors: string[] = [];
  const requiredStrings: Array<[string, string | null | undefined]> = [
    ["contract_version", span.contract_version],
    ["document_version_id", span.document_version_id],
    ["chunk_id", span.chunk_id],
    ["source_path", span.source_path],
    ["logical_source_path", span.logical_source_path],
    ["chunk_text_hash_protocol_version", span.chunk_text_hash_protocol_version],
    ["raw_span_text_hash_protocol_version", span.raw_span_text_hash_protocol_version],
    ["snapshot.protocol_version", span.source_snapshot_verification?.protocol_version],
    ["snapshot.final_open_protocol_version", span.source_snapshot_verification?.final_open_protocol_version],
    ["snapshot.storage_path", span.source_snapshot_verification?.storage_path],
  ];
  if (span.contract_version !== "raw_chunk_source_span_v1") {
    errors.push(`${path}.contract_version is invalid`);
  }
  for (const [field, value] of requiredStrings) {
    if (typeof value !== "string" || !value.length) errors.push(`${path}.${field} is missing`);
  }
  for (const [field, value] of [
    ["source_checksum", span.source_checksum],
    ["chunk_text_hash", span.chunk_text_hash],
    ["raw_span_text_hash", span.raw_span_text_hash],
    ["snapshot.checksum", span.source_snapshot_verification?.checksum],
  ] as Array<[string, string | null | undefined]>) {
    if (typeof value !== "string" || !sha256Pattern.test(value)) errors.push(`${path}.${field} is not canonical SHA-256`);
  }
  if (span.source_checksum !== span.source_snapshot_verification?.checksum) {
    errors.push(`${path}.source_checksum does not match the verified snapshot`);
  }
  if (span.source_snapshot_verification?.verified !== true) {
    errors.push(`${path}.source snapshot is not verified`);
  }
  if (!Number.isInteger(span.source_snapshot_verification?.size_bytes) || span.source_snapshot_verification.size_bytes < 0) {
    errors.push(`${path}.source snapshot size is invalid`);
  }
  const orderedSpan = (value: Array<number | null> | null | undefined, allowNull: boolean): boolean => (
    Array.isArray(value) && value.length === 2 && value.every((item) => (
      (allowNull && item === null) || (Number.isInteger(item) && (item as number) >= 0)
    )) && (
      value[0] === null || value[1] === null || (value[0] as number) <= (value[1] as number)
    )
  );
  if (!orderedSpan(span.char_span, false)) errors.push(`${path}.char_span is invalid`);
  if (span.raw_chunk_char_span != null && !orderedSpan(span.raw_chunk_char_span, false)) errors.push(`${path}.raw_chunk_char_span is invalid`);
  if (
    span.raw_chunk_char_span != null && orderedSpan(span.char_span, false) && orderedSpan(span.raw_chunk_char_span, false) &&
    ((span.char_span[0] as number) < (span.raw_chunk_char_span[0] as number) || (span.char_span[1] as number) > (span.raw_chunk_char_span[1] as number))
  ) errors.push(`${path}.char_span escapes raw_chunk_char_span`);
  if (!orderedSpan(span.page_range, true)) errors.push(`${path}.page_range is invalid`);
  if (typeof span.content_clipped !== "boolean") errors.push(`${path}.content_clipped is invalid`);
  if (span.content_token_count == null || !Number.isInteger(span.content_token_count) || span.content_token_count < 0) {
    errors.push(`${path}.content_token_count is invalid`);
  }
  return errors;
}

function packageAuthorityErrors(data: ContextPackageResponse, packageId: string, traceId?: string | null): string[] {
  const errors: string[] = [];
  if (data.contract_version !== "context_package_public_v1") {
    errors.push("unexpected context package contract version");
  }
  if (data.id !== packageId) {
    errors.push(`requested package ${packageId} but received ${data.id}`);
  }
  if (traceId && data.retrieval_trace_id !== traceId) {
    errors.push(`package retrieval_trace_id ${data.retrieval_trace_id || "missing"} does not match answer trace ${traceId}`);
  }
  const packageTraceId = data.retrieval_trace_id;
  if (!packageTraceId) {
    errors.push("context package retrieval_trace_id is missing");
  }
  const hashCard = data.package_hash_card;
  if (
    !sha256Pattern.test(data.package_hash) ||
    !sha256Pattern.test(hashCard.public_payload_hash) ||
    !sha256Pattern.test(hashCard.public_citation_spans_hash) ||
    hashCard.protocol_version !== "context_package_public_hash_v1" ||
    hashCard.canonicalization !== "json_utf8_sort_keys_compact_v1" ||
    hashCard.citation_spans_consistency !== "persisted_equals_public_projection" ||
    !sameAuditValue(hashCard.hashed_public_fields, contextPackageHashedFields) ||
    hashCard.chunk_count !== data.package.chunks.length ||
    hashCard.citation_span_count !== data.citation_spans.length ||
    hashCard.graph_expansion_path_count !== data.graph_expansion_paths.length
  ) {
    errors.push("context package public hash card is invalid");
  }
  const packageChunksById = new Map(data.package.chunks.map((chunk) => [chunk.chunk_id, chunk]));
  const packageChunkIds = data.package.chunks.map((chunk) => chunk.chunk_id);
  const contextChunkIds = data.contexts.map((context) => context.chunk_id);
  const citationChunkIds = data.citation_spans.map((citation) => citation.source_span.chunk_id);
  if (
    packageChunksById.size !== packageChunkIds.length ||
    new Set(contextChunkIds).size !== contextChunkIds.length ||
    new Set(citationChunkIds).size !== citationChunkIds.length ||
    !sameAuditValue([...packageChunkIds].sort(), [...contextChunkIds].sort()) ||
    !sameAuditValue([...packageChunkIds].sort(), [...citationChunkIds].sort())
  ) {
    errors.push("package/context/citation chunk projections do not have one-to-one coverage");
  }
  const foreignChunks = data.package.chunks
    .filter(
      (chunk) => {
        const span = chunk.source_span;
        errors.push(...sourceSpanIntegrityErrors(span, `package.chunks.${chunk.chunk_id}.source_span`));
        return (
        chunk.context_package_id !== data.id ||
        span.context_package_id !== data.id ||
        span.retrieval_trace_id !== packageTraceId ||
        span.chunk_id !== chunk.chunk_id ||
        span.document_version_id !== chunk.document_version_id ||
        span.source_path !== chunk.source_path ||
        span.logical_source_path !== chunk.logical_source_path ||
        span.chunk_text_hash_protocol_version !== chunk.chunk_text_hash_protocol_version ||
        span.chunk_text_hash !== chunk.chunk_text_hash ||
        span.raw_span_text_hash_protocol_version !== chunk.raw_span_text_hash_protocol_version ||
        span.raw_span_text_hash !== chunk.raw_span_text_hash ||
        !sameAuditValue(span.char_span, chunk.char_span) ||
        !sameAuditValue(span.raw_chunk_char_span, chunk.raw_chunk_char_span) ||
        !sameAuditValue(span.page_range, chunk.page_range) ||
        !sameAuditValue(span.section_path, chunk.section_path) ||
        !sameAuditValue(span.structure_path, chunk.structure_path) ||
        !sameAuditValue(span.structure_node_ids, chunk.structure_node_ids) ||
        !sameAuditValue(span.bbox, chunk.bbox) ||
        span.content_clipped !== chunk.content_clipped ||
        span.content_token_count !== chunk.content_token_count
        );
      },
    )
    .map((chunk) => chunk.chunk_id);
  if (foreignChunks.length) {
    errors.push(`chunk package provenance mismatch: ${foreignChunks.join(", ")}`);
  }
  const chunkDedupeKeys = data.package.chunks.map((chunk) => chunk.dedupe_key);
  if (
    new Set(chunkDedupeKeys).size !== chunkDedupeKeys.length ||
    !sameAuditValue(data.dedupe_keys ?? [], chunkDedupeKeys) ||
    !sameAuditValue(data.diagnostics.dedupe_keys, chunkDedupeKeys) ||
    data.package.chunks.some((chunk) => !sameAuditValue(data.why_selected[chunk.chunk_id], chunk.why_selected)) ||
    !sameAuditValue(Object.keys(data.why_selected).sort(), [...packageChunkIds].sort())
  ) {
    errors.push("package chunk why-selected or dedupe projection mismatch");
  }
  const foreignContexts = data.contexts
    .filter(
      (context) => {
        const canonicalChunk = packageChunksById.get(context.chunk_id);
        const metadata = context.metadata;
        const span = metadata.source_span;
        errors.push(...sourceSpanIntegrityErrors(span, `contexts.${context.chunk_id}.metadata.source_span`));
        const expectedContext = canonicalChunk ? {
          contract_kind: "context_item",
          chunk_id: canonicalChunk.chunk_id,
          document_title: canonicalChunk.document_title,
          source_path: canonicalChunk.source_path,
          content: canonicalChunk.content,
          snippet: canonicalChunk.content.slice(0, 280),
          metadata: {
            source_path: canonicalChunk.source_path,
            logical_source_path: canonicalChunk.logical_source_path,
            section_path: canonicalChunk.section_path,
            structure_path: canonicalChunk.structure_path,
            parent_section_node_id: canonicalChunk.structure_closure.parent_section_node_id,
            parent_section: canonicalChunk.parent_section,
            structure_node_ids: canonicalChunk.structure_node_ids,
            page_range: canonicalChunk.page_range,
            char_span: canonicalChunk.char_span,
            bbox: canonicalChunk.bbox,
            source_span: canonicalChunk.source_span,
            structure_closure: canonicalChunk.structure_closure,
            why_selected: canonicalChunk.why_selected,
            dedupe_key: canonicalChunk.dedupe_key,
            role: canonicalChunk.role,
            content_clipped: canonicalChunk.content_clipped,
            content_token_count: canonicalChunk.content_token_count,
            original_token_count: canonicalChunk.original_token_count,
            raw_chunk_char_span: canonicalChunk.raw_chunk_char_span,
            context_package_id: data.id,
          },
        } : null;
        return (
        !canonicalChunk ||
        !sameAuditValue(context, expectedContext) ||
        context.metadata.context_package_id !== data.id ||
        span.context_package_id !== data.id ||
        span.retrieval_trace_id !== packageTraceId ||
        span.chunk_id !== context.chunk_id ||
        context.document_title !== canonicalChunk.document_title ||
        context.source_path !== metadata.source_path ||
        context.source_path !== canonicalChunk.source_path ||
        context.content !== canonicalChunk.content ||
        metadata.source_path !== span.source_path ||
        metadata.logical_source_path !== span.logical_source_path ||
        !sameAuditValue(metadata.char_span, span.char_span) ||
        !sameAuditValue(metadata.raw_chunk_char_span, span.raw_chunk_char_span) ||
        !sameAuditValue(metadata.page_range, span.page_range) ||
        !sameAuditValue(metadata.section_path, span.section_path) ||
        !sameAuditValue(metadata.structure_path, span.structure_path) ||
        !sameAuditValue(metadata.structure_node_ids, span.structure_node_ids) ||
        !sameAuditValue(metadata.bbox, span.bbox) ||
        metadata.content_clipped !== span.content_clipped ||
        metadata.content_token_count !== span.content_token_count ||
        !sameAuditValue(span, canonicalChunk.source_span)
        );
      },
    )
    .map((context) => context.chunk_id);
  if (foreignContexts.length) {
    errors.push(`context item provenance mismatch: ${foreignContexts.join(", ")}`);
  }
  const foreignCitations = data.citation_spans
    .filter(
      (citation) => {
        const canonicalChunk = packageChunksById.get(citation.source_span.chunk_id);
        const span = citation.source_span;
        errors.push(...sourceSpanIntegrityErrors(span, `citation_spans.${span.chunk_id}.source_span`));
        const expectedCitation = canonicalChunk ? {
          contract_kind: "citation_span",
          document_id: canonicalChunk.document_id,
          document_title: canonicalChunk.document_title,
          source_path: canonicalChunk.source_path,
          logical_source_path: canonicalChunk.logical_source_path,
          section_path: canonicalChunk.section_path,
          structure_path: canonicalChunk.structure_path,
          structure_node_ids: canonicalChunk.structure_node_ids,
          structure_closure: canonicalChunk.structure_closure,
          source_span: canonicalChunk.source_span,
        } : null;
        return (
        !canonicalChunk ||
        !sameAuditValue(citation, expectedCitation) ||
        citation.source_span.context_package_id !== data.id ||
        citation.source_span.retrieval_trace_id !== packageTraceId ||
        citation.document_id !== canonicalChunk.document_id ||
        citation.document_title !== canonicalChunk.document_title ||
        citation.source_path !== span.source_path ||
        citation.logical_source_path !== span.logical_source_path ||
        !sameAuditValue(citation.section_path, span.section_path) ||
        !sameAuditValue(citation.structure_path, span.structure_path) ||
        !sameAuditValue(citation.structure_node_ids, span.structure_node_ids) ||
        !sameAuditValue(span, canonicalChunk.source_span)
        );
      },
    )
    .map((citation) => citation.source_span.chunk_id);
  if (foreignCitations.length) {
    errors.push(`citation span provenance mismatch: ${foreignCitations.join(", ")}`);
  }
  if (data.diagnostics.repair_gray_zone_model_call_count !== 0 || data.diagnostics.repair_gray_zone_decision_authority !== false) {
    errors.push("gray-zone authority audit is not executor-only with model_call_count=0");
  }
  if (data.diagnostics.conversation_state_is_evidence !== false) {
    errors.push("conversation state was incorrectly marked as evidence");
  }
  if (data.diagnostics.snapshot_integrity.fail_closed !== true) {
    errors.push("immutable snapshot audit is not fail-closed");
  }
  return errors;
}

function ContextPackageBody({ evidencePackage, packageId, traceId }: { evidencePackage: ContextPackageResponse; packageId: string; traceId?: string | null }) {
  const diagnostics = evidencePackage.diagnostics;
  const authorityErrors = packageAuthorityErrors(evidencePackage, packageId, traceId);
  if (authorityErrors.length) {
    return <EvidenceFailure title="Context Package 身份或权限审计失败" message={`${authorityErrors.join("；")}。该 payload 不作为可信 QA 证据展示。`} />;
  }
  const tokenAudit = diagnostics.token_budget_audit;
  return (
    <div className="grid gap-4" data-testid="qa-context-package-body">
      <div className="grid gap-3 rounded-2xl border border-cyan-200/12 bg-cyan-300/[0.025] p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-100/48">Context Package identity</p>
            <code className="mt-1 block break-all text-xs text-cyan-50/72">{evidencePackage.id}</code>
            <p className="mt-2 text-sm text-white/65">query: {evidencePackage.query}</p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <AuditBadge ok={tokenAudit.within_budget}>token budget {tokenAudit.token_count}/{tokenAudit.token_budget}</AuditBadge>
            <AuditBadge ok={diagnostics.snapshot_integrity.fail_closed}>snapshot fail-closed</AuditBadge>
            <AuditBadge ok={diagnostics.repair_gray_zone_model_call_count === 0}>gray model calls = 0</AuditBadge>
            <AuditBadge ok={!diagnostics.repair_gray_zone_decision_authority}>LLM gray authority = false</AuditBadge>
          </div>
        </div>
        <div className="grid gap-1 text-xs text-white/42">
          <p className="break-all">package hash <code>{evidencePackage.package_hash}</code></p>
          <p className="break-all">retrieval trace <code>{evidencePackage.retrieval_trace_id || "—"}</code></p>
          <p className="break-all">runtime settings <code>{diagnostics.runtime_settings_hash}</code></p>
          <p className="break-all">conversation scope <code>{diagnostics.conversation_state_scope_hash}</code> · not evidence</p>
          <p>restoration {diagnostics.context_restoration_protocol} · packing {tokenAudit.packing_protocol} · granularity {diagnostics.retrieval_granularity}</p>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <EvidenceIdList title="selected hit chunks" values={evidencePackage.hit_chunk_ids} />
        <EvidenceIdList title="restored chunks" values={evidencePackage.restored_chunk_ids} />
        <EvidenceIdList title="bridge chunks" values={evidencePackage.bridge_chunk_ids} />
        <EvidenceIdList title="graph path edge ids" values={evidencePackage.graph_path_ids} />
        <EvidenceIdList title="parent structure nodes" values={evidencePackage.parent_structure_node_ids} />
        <EvidenceIdList title="covered facets" values={evidencePackage.covered_facets} />
      </div>

      <div className="grid gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-4 text-xs text-white/50">
        <p className="text-[11px] uppercase tracking-[0.17em] text-white/38">Restoration / budget diagnostics</p>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <span>hit {diagnostics.restore_counts.hit_chunks}</span>
          <span>restored {diagnostics.restore_counts.restored_chunks}</span>
          <span>bridge {diagnostics.restore_counts.bridge_chunks}</span>
          <span>graph path {diagnostics.restore_counts.graph_path_chunks}</span>
          <span>parent nodes {diagnostics.restore_counts.parent_structure_nodes}</span>
          <span>per-hit restore budget {diagnostics.restore_counts.per_hit_chunk_budget}</span>
          <span>verified document versions {diagnostics.snapshot_integrity.verified_document_version_count}</span>
          <span>clipped {tokenAudit.clipped_chunk_ids.length}</span>
          <span>skipped by hard token budget {tokenAudit.skipped_chunk_ids.length}</span>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <EvidenceIdList title="clipped chunk ids" values={tokenAudit.clipped_chunk_ids} />
          <EvidenceIdList title="skipped chunk ids" values={tokenAudit.skipped_chunk_ids} />
        </div>
      </div>

      <DedupeAudit evidencePackage={evidencePackage} />

      <section className="grid gap-3" aria-label="Context package selected and restored chunks">
        <div className="flex items-center gap-2 text-sm font-semibold text-white/75">
          <FileCheck2 className="size-4 text-cyan-100/65" />
          Selected / restored evidence chunks ({evidencePackage.package.chunks.length})
        </div>
        {evidencePackage.package.chunks.length ? evidencePackage.package.chunks.map((chunk, index) => <ChunkEvidenceCard key={`${chunk.chunk_id}-${chunk.dedupe_key}`} chunk={chunk} index={index} />) : <EvidenceFailure title="Context Package 没有证据 chunk" message="QA 不得退回裸检索命中或 conversation prose 生成事实性回答。" />}
      </section>

      <section className="grid gap-3" aria-label="Context package citation spans">
        <p className="text-sm font-semibold text-white/75">Citation-ready raw spans ({evidencePackage.citation_spans.length})</p>
        {evidencePackage.citation_spans.length ? evidencePackage.citation_spans.map((citation, index) => <CitationSpanView key={`${citation.source_span.document_version_id}-${spanText(citation.source_span.char_span)}-${index}`} citation={citation} index={index} />) : <EvidenceFailure title="Context Package 没有 citation span" message="当前证据包不能为事实声明提供 raw span 地址。" />}
      </section>

      <section className="grid gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-4" aria-label="Context package path summary">
        <div className="flex items-center gap-2 text-sm font-semibold text-white/75">
          <Network className="size-4 text-cyan-100/65" />
          Graph / reached-by path summary
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-white/50">
          <span>visits {diagnostics.path_summary.node_visit_count}</span>
          <span>parents {diagnostics.path_summary.distinct_parent_count}</span>
          <span>paths {diagnostics.path_summary.distinct_path_count}</span>
          <span>edge types {diagnostics.path_summary.distinct_edge_type_count}</span>
          <span>convergence {numberText(diagnostics.path_summary.cycle_convergence_score)}</span>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <EvidenceIdList title="path covered facets" values={diagnostics.path_summary.covered_facets} />
          <EvidenceIdList title="path support chunk union" values={diagnostics.path_summary.support_chunk_union} />
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {evidencePackage.concept_path.map((entry, index) => <EvidenceIdList key={`${entry.layer}-${index}`} title={`${entry.layer} concept path`} values={entry.ids} />)}
        </div>
        <PathContributionList title="package reached by paths" paths={evidencePackage.reached_by_paths} />
        <div className="grid gap-3">
          {evidencePackage.node_contributions.map((contribution) => <NodeContributionView key={`${contribution.layer}-${contribution.node_id}`} contribution={contribution} />)}
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {evidencePackage.graph_expansion_paths.map((expansion, index) => <GraphExpansionView key={`${expansion.kind}-${index}`} expansion={expansion} />)}
        </div>
      </section>
    </div>
  );
}

function packageError(error: Error): { title: string; message: string } {
  const apiError = error as ApiError;
  const status = apiError.status;
  const detail = apiError.structured?.message || error.message;
  return {
    title: status === 404 ? "Context Package 不存在" : status === 409 ? "Context Package 契约冲突" : "Context Package 读取失败",
    message: [status ? `HTTP ${status}` : "", apiError.structured?.code || "", detail, "不能以裸检索命中或 citation cards 替代证据包。"].filter(Boolean).join(" · "),
  };
}

export function QAContextPackageEvidence({
  packageId,
  traceId,
  retrievalExpected = true,
  defaultExpanded = false,
}: {
  packageId?: string | null;
  traceId?: string | null;
  retrievalExpected?: boolean;
  defaultExpanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const packageQuery = useQuery({
    queryKey: ["qa-context-package", packageId],
    queryFn: () => fetchContextPackage(packageId as string),
    enabled: Boolean(packageId) && expanded,
    retry: false,
  });

  if (!packageId) {
    return retrievalExpected ? (
      <EvidenceFailure title="Context Package 证据视图缺失" message="该 QA 回答没有 context_package_id；不能以裸检索命中、回答正文或 citation cards 替代回答生成的唯一证据输入。" />
    ) : (
      <p className="mt-4 rounded-2xl border border-white/8 bg-white/[0.025] p-4 text-xs text-white/42">本次非检索路线未生成 Context Package。</p>
    );
  }

  return (
    <section data-testid="qa-context-package-evidence" className="mt-4 rounded-2xl border border-cyan-200/12 bg-cyan-300/[0.018] p-3">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
        className="flex w-full items-center justify-between gap-3 rounded-xl px-1 py-1 text-left"
      >
        <span className="min-w-0">
          <span className="block text-xs font-semibold uppercase tracking-[0.17em] text-cyan-100/62">Context Package 证据视图</span>
          <code className="mt-1 block truncate text-[11px] text-white/42">{packageId}</code>
        </span>
        <ChevronDown className={cn("size-4 shrink-0 text-white/45 transition", expanded && "rotate-180")} />
      </button>
      {expanded ? (
        <div className="mt-3">
          {packageQuery.isLoading ? (
            <LoadingBlock rows={3} />
          ) : packageQuery.error instanceof Error ? (
            <EvidenceFailure {...packageError(packageQuery.error)} />
          ) : packageQuery.data ? (
            <ContextPackageBody evidencePackage={packageQuery.data} packageId={packageId} traceId={traceId} />
          ) : (
            <EvidenceFailure title="Context Package payload 缺失" message="请求已结束但没有返回闭合证据包。" />
          )}
        </div>
      ) : null}
    </section>
  );
}
