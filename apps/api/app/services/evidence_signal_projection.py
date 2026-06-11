"""Signal projection layer. See apps/api/README.md > Signal projection and evidence graph diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ActiveChunk,
    CommunityMembership,
    CommunityState,
    EvidenceAtom,
    ProjectionCommunity,
    ProjectionEdge,
    ProjectionNode,
    ProjectionState,
    SignalCandidate,
    SignalCommunity,
    SignalCommunityMembership,
    SignalDecision,
    SignalEdge,
    SignalNode,
    SignalRelationSpec,
    SignalSchemaState,
    SignalState,
    SignalTypeSpec,
)


SIGNAL_LAYER_PROTOCOL_VERSION = "evidence_signal_layer_v1"
SIGNAL_SCHEMA_PROTOCOL_VERSION = "evidence_signal_schema_induction_v1"
SIGNAL_EXTRACTOR_VERSION = "evidence_signal_extractors_v1"
SIGNAL_GATE_VERSION = "evidence_signal_gate_v1"
SIGNAL_RELATION_PROTOCOL_VERSION = "observed_signal_relations_v1"
SIGNAL_COMMUNITY_PROTOCOL_VERSION = "signal_support_overlap_components_v1"
PROJECTION_PROTOCOL_VERSION = "signal_projection_view_v1"


@dataclass(frozen=True)
class SignalDraft:
    atom: EvidenceAtom
    surface: str
    normalized_key: str
    canonical_label: str
    signal_type: str
    extractor_name: str
    confidence: float
    source_span: dict[str, Any]
    source_span_hash: str
    features: dict[str, Any]


@dataclass(frozen=True)
class EvidenceSignalBuildResult:
    state: SignalState
    schema_state: SignalSchemaState
    projection_state: ProjectionState
    nodes: list[SignalNode]
    edges: list[SignalEdge]
    candidates: list[SignalCandidate]
    decisions: list[SignalDecision]
    stats: dict[str, Any]


def stable_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    symbols = max(len(text) - cjk - sum(len(item) for item in re.findall(r"[A-Za-z0-9_]+", text)), 0)
    return max(1, cjk + latin_words + symbols // 4)


def _clean_surface(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip(" \t\r\n-–—:;,.()[]{}<>\"'`")).strip()[:160]


def _normalized_key(surface: str) -> str:
    normalized = _clean_surface(surface).lower()
    normalized = re.sub(r"[\s_/-]+", " ", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff ]+", "", normalized)
    return normalized.strip()


def _canonical_label(surface: str) -> str:
    cleaned = _clean_surface(surface)
    if not cleaned:
        return cleaned
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _/-]*", cleaned):
        words = [word if word.isupper() else word[:1].upper() + word[1:] for word in re.split(r"(\s+)", cleaned.lower())]
        return "".join(words)
    return cleaned


def _token_count(value: str) -> int:
    return len(re.findall(r"[\w\u4e00-\u9fff]+", value or ""))


def _surface_shape(surface: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", surface or "")
    chars = len(compact)
    digits = sum(char.isdigit() for char in compact)
    uppercase = sum(char.isupper() for char in compact)
    cjk = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    symbols = sum(1 for char in compact if not char.isalnum() and not ("\u4e00" <= char <= "\u9fff"))
    return {
        "char_count": chars,
        "token_count": _token_count(surface),
        "digit_ratio": round(digits / max(chars, 1), 6),
        "uppercase_ratio": round(uppercase / max(chars, 1), 6),
        "cjk_ratio": round(cjk / max(chars, 1), 6),
        "symbol_count": symbols,
    }


def _surface_is_measureable(surface: str) -> bool:
    cleaned = _clean_surface(surface)
    shape = _surface_shape(cleaned)
    if not cleaned or shape["char_count"] < 2 or shape["char_count"] > 160:
        return False
    if re.fullmatch(r"[\d\W_]+", cleaned):
        return False
    if shape["token_count"] == 1 and shape["char_count"] < 4 and shape["symbol_count"] == 0 and shape["cjk_ratio"] == 0:
        return False
    if shape["digit_ratio"] >= 0.85:
        return False
    return True


def _span_for_surface(atom: EvidenceAtom, surface: str) -> tuple[dict[str, Any], str]:
    atom_span = atom.source_span_json or {}
    text = atom.text or ""
    offset = text.lower().find(surface.lower())
    if offset < 0:
        offset = 0
    base_start = int(atom_span.get("start") or 0)
    span = {
        **atom_span,
        "evidence_atom_id": atom.id,
        "surface": surface,
        "start": base_start + offset,
        "end": base_start + offset + len(surface),
        "signal_protocol_version": SIGNAL_LAYER_PROTOCOL_VERSION,
    }
    return span, stable_hash(span)


def _first_line(text: str) -> str:
    return next((line.strip() for line in (text or "").splitlines() if line.strip()), "")


def _definition_like_surfaces(text: str) -> list[str]:
    surfaces: list[str] = []
    for match in re.finditer(r"^(.{2,120}?)(?:\s+[-:]\s+|\s+(?:is|are|means|refers to|denotes|represents)\s+)", text or "", flags=re.IGNORECASE | re.MULTILINE):
        surfaces.append(match.group(1))
    for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9 _/-]{1,80})(?:是指|指的是|定义为|表示|用于)", text or ""):
        surfaces.append(match.group(1))
    return surfaces


def _symbol_surfaces(text: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]{1,30}(?:\([^)]{0,60}\))?)\b(?=\s*(?:=|:|->|=>))", text or "")
    ]


def _title_phrase_surfaces(text: str) -> list[str]:
    surfaces: list[str] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,5})\b", text or ""):
        surface = match.group(1)
        if _surface_shape(surface)["token_count"] >= 1:
            surfaces.append(surface)
    return surfaces


def _lexical_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"[\w\u4e00-\u9fff]{3,}", text or ""):
        term = match.group(0)
        shape = _surface_shape(term)
        if shape["digit_ratio"] < 0.65:
            terms.add(term.lower())
    return terms


def _specificity(surface: str) -> float:
    shape = _surface_shape(surface)
    if shape["char_count"] == 0:
        return 0.0
    length_signal = min(0.62, shape["char_count"] / 24.0)
    multi_token_signal = min(0.22, max(shape["token_count"] - 1, 0) * 0.055)
    shape_signal = min(0.16, shape["uppercase_ratio"] * 0.10 + min(shape["symbol_count"], 3) * 0.025)
    digit_penalty = 0.20 if shape["digit_ratio"] > 0.55 else 0.0
    return max(0.0, min(1.0, length_signal + multi_token_signal + shape_signal - digit_penalty))


def _source_span_union_from_spans(spans: list[dict[str, Any]]) -> dict[str, Any]:
    atom_ids = sorted({str(span.get("evidence_atom_id")) for span in spans if span.get("evidence_atom_id")})
    document_version_ids = sorted({str(span.get("document_version_id")) for span in spans if span.get("document_version_id")})
    return {
        "spans": spans,
        "evidence_atom_ids": atom_ids,
        "document_version_ids": document_version_ids,
        "atom_count": len(atom_ids),
    }


def _source_span_union_for_candidates(candidates: list[SignalCandidate | SignalDraft]) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, SignalCandidate):
            spans.extend((candidate.source_span_union_json or {}).get("spans") or [])
        else:
            spans.append(candidate.source_span)
    return _source_span_union_from_spans(spans)


def _draft_candidates(atoms: list[EvidenceAtom]) -> tuple[list[SignalDraft], dict[str, Any]]:
    term_counts = Counter(term for atom in atoms for term in _lexical_terms(atom.text))
    drafts_by_atom_key: dict[tuple[str, str, str], SignalDraft] = {}
    estimated_tokens = 0
    extractor_counts: Counter[str] = Counter()
    for atom in atoms:
        text = atom.text or ""
        estimated_tokens += estimate_tokens(text)
        proposals: list[tuple[str, str, str, float, dict[str, Any]]] = []
        if atom.atom_type == "heading":
            proposals.append((_first_line(text), "heading_anchor_extractor", "section_anchor", 0.78, {"structure_score": 1.0}))
        if atom.atom_type in {"table_block", "formula", "code_block", "list_item"}:
            signal_type = atom.atom_type.replace("_block", "")
            proposals.append((_first_line(text), f"{signal_type}_structure_extractor", f"{signal_type}_anchor", 0.70, {"structure_score": 0.86}))
        for surface in _definition_like_surfaces(text):
            proposals.append((surface, "definition_measurement_extractor", "definition_like_signal", 0.76, {"definition_like_score": 0.92}))
        for surface in _symbol_surfaces(text):
            proposals.append((surface, "symbol_context_extractor", "symbol_anchor", 0.74, {"symbol_context_score": 0.88}))
        for surface in _title_phrase_surfaces(text):
            proposals.append((surface, "capitalized_phrase_extractor", "named_surface", 0.58, {"shape_prominence": 0.54}))
        for term in sorted(_lexical_terms(text)):
            if term_counts[term] >= 2:
                proposals.append((term, "repeated_surface_extractor", "local_repeated_signal", 0.48, {"support_frequency": term_counts[term]}))
        for surface, extractor_name, signal_type, confidence, features in proposals[:14]:
            cleaned = _clean_surface(surface)
            if not _surface_is_measureable(cleaned):
                continue
            key = _normalized_key(cleaned)
            span, span_hash = _span_for_surface(atom, cleaned)
            shape = _surface_shape(cleaned)
            draft = SignalDraft(
                atom=atom,
                surface=cleaned,
                normalized_key=key,
                canonical_label=_canonical_label(cleaned),
                signal_type=signal_type,
                extractor_name=extractor_name,
                confidence=confidence,
                source_span=span,
                source_span_hash=span_hash,
                features={
                    **features,
                    "surface_shape": shape,
                    "specificity": round(_specificity(cleaned), 6),
                    "atom_type": atom.atom_type,
                    "extractor_version": SIGNAL_EXTRACTOR_VERSION,
                },
            )
            draft_key = (atom.id, key, extractor_name)
            current = drafts_by_atom_key.get(draft_key)
            if current is None or current.confidence < draft.confidence:
                drafts_by_atom_key[draft_key] = draft
                extractor_counts[extractor_name] += 1
    return list(drafts_by_atom_key.values()), {
        "eligible_atoms": len(atoms),
        "estimated_tokens": estimated_tokens,
        "extractor_counts": dict(extractor_counts),
        "candidate_protocol_version": SIGNAL_EXTRACTOR_VERSION,
    }


def _sample_pack(
    *,
    atoms: list[EvidenceAtom],
    drafts: list[SignalDraft],
    community_state: CommunityState | None,
    memberships: list[CommunityMembership],
) -> dict[str, Any]:
    community_by_atom: dict[str, list[str]] = defaultdict(list)
    for membership in memberships:
        community_by_atom[membership.atom_id].append(membership.community_id)
    atom_samples = [
        {
            "evidence_atom_id": atom.id,
            "atom_type": atom.atom_type,
            "source_span": atom.source_span_json,
            "text_hash": atom.text_hash,
            "text_preview": (atom.text or "")[:240],
            "community_ids": community_by_atom.get(atom.id, []),
        }
        for atom in atoms[:80]
    ]
    return {
        "atom_samples": atom_samples,
        "evidence_community_state_id": community_state.id if community_state else None,
        "evidence_community_hash": community_state.state_hash if community_state else None,
        "raw_signal_candidate_samples": [
            {
                "surface": draft.surface,
                "candidate_type": draft.signal_type,
                "evidence_atom_ids": [draft.atom.id],
                "source_span": draft.source_span,
                "features": draft.features,
            }
            for draft in drafts[:120]
        ],
        "task": "Infer useful signal and relation types for retrieval expansion. Do not create facts.",
    }


def _schema_spec_from_candidates(sample_pack: dict[str, Any], drafts: list[SignalDraft]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_type: dict[str, list[SignalDraft]] = defaultdict(list)
    for draft in drafts:
        by_type[draft.signal_type].append(draft)
    type_specs: list[dict[str, Any]] = []
    for signal_type, grouped in sorted(by_type.items()):
        atom_types = sorted({draft.atom.atom_type for draft in grouped})
        extractor_names = sorted({draft.extractor_name for draft in grouped})
        support_rate = sum(1 for draft in grouped if draft.source_span.get("evidence_atom_id")) / max(len(grouped), 1)
        type_specs.append(
            {
                "name": signal_type,
                "description": f"Evidence-backed signal measured from {len(grouped)} candidate observations.",
                "evidence_patterns": extractor_names,
                "applicable_atom_types": atom_types,
                "risk": "medium" if support_rate >= 1.0 else "high",
                "retrieval_use": ["anchor_selection", "neighborhood_expansion", "diagnostics"],
                "gate": {
                    "requires_source_span": True,
                    "requires_evidence_atom": True,
                    "min_specificity": 0.18,
                    "min_support_atoms": 1,
                },
                "support_rate": round(support_rate, 6),
            }
        )
    relation_specs = [
        {
            "name": "co_supported_by_atom",
            "source_signal_types": sorted(by_type),
            "target_signal_types": sorted(by_type),
            "required_evidence": ["shared_evidence_atom", "source_span"],
            "risk": "medium",
            "gate": {"requires_shared_support_atom": True, "requires_source_span": True},
        },
        {
            "name": "co_supported_by_active_chunk",
            "source_signal_types": sorted(by_type),
            "target_signal_types": sorted(by_type),
            "required_evidence": ["shared_active_chunk", "source_span"],
            "risk": "medium",
            "gate": {"requires_shared_active_chunk": True, "requires_source_span": True},
        },
    ]
    stats = {
        "sample_pack_hash": stable_hash(sample_pack),
        "signal_type_count": len(type_specs),
        "relation_type_count": len(relation_specs),
        "candidate_type_counts": {key: len(value) for key, value in by_type.items()},
    }
    return type_specs, relation_specs, stats


def _schema_gate(type_specs: list[dict[str, Any]], relation_specs: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for spec in type_specs:
        if not spec.get("name") or not spec.get("retrieval_use"):
            failures.append(f"type_spec_incomplete:{spec.get('name')}")
        gate = spec.get("gate") or {}
        if not gate.get("requires_source_span") or not gate.get("requires_evidence_atom"):
            failures.append(f"type_spec_missing_evidence_gate:{spec.get('name')}")
    for spec in relation_specs:
        if not spec.get("required_evidence"):
            failures.append(f"relation_spec_missing_required_evidence:{spec.get('name')}")
    return not failures, failures


def _candidate_quality(key: str, grouped: list[SignalDraft]) -> dict[str, Any]:
    support_atom_ids = sorted({draft.atom.id for draft in grouped})
    support_docs = sorted({draft.atom.document_id for draft in grouped})
    specificity = max(_specificity(draft.surface) for draft in grouped)
    structural_strength = max(float(draft.features.get("structure_score") or 0.0) for draft in grouped)
    definition_strength = max(float(draft.features.get("definition_like_score") or 0.0) for draft in grouped)
    symbol_strength = max(float(draft.features.get("symbol_context_score") or 0.0) for draft in grouped)
    support_strength = min(1.0, math.log1p(len(support_atom_ids)) / math.log(5))
    fragmentation = len({draft.surface.lower() for draft in grouped}) / max(len(grouped), 1)
    evidence_strength = max(structural_strength, definition_strength, symbol_strength)
    extractor_names = {draft.extractor_name for draft in grouped}
    repeated_only = extractor_names == {"repeated_surface_extractor"}
    confidence = (
        0.32 * specificity
        + 0.26 * support_strength
        + 0.26 * evidence_strength
        + 0.10 * min(1.0, len(support_docs) / 3)
        + 0.06 * (1.0 - min(1.0, fragmentation))
    )
    has_span = all(draft.source_span.get("evidence_atom_id") for draft in grouped)
    support_span_rate = sum(1 for draft in grouped if draft.source_span.get("evidence_atom_id")) / max(len(grouped), 1)
    keep = has_span and support_atom_ids and (confidence >= 0.28 or evidence_strength >= 0.70)
    if repeated_only and evidence_strength < 0.20 and (specificity < 0.42 or len(support_atom_ids) < 3):
        keep = False
    if not keep and evidence_strength >= 0.70 and support_atom_ids:
        action = "keep_isolated"
    elif keep:
        action = "accept"
    else:
        action = "reject_noise"
    return {
        "normalized_key": key,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "action": action,
        "specificity": round(specificity, 6),
        "support_atom_count": len(support_atom_ids),
        "support_document_count": len(support_docs),
        "support_span_rate": round(support_span_rate, 6),
        "evidence_strength": round(evidence_strength, 6),
        "fragmentation": round(fragmentation, 6),
        "gate_protocol_version": SIGNAL_GATE_VERSION,
    }


def _set_signal_state_status(state: SignalState, status: str) -> None:
    state.status = status
    if status in {"active", "failed"}:
        state.completed_at = datetime.utcnow()


def _load_memberships(db: Session, community_state: CommunityState | None) -> list[CommunityMembership]:
    if community_state is None:
        return []
    return db.scalars(select(CommunityMembership).where(CommunityMembership.community_state_id == community_state.id)).all()


def build_evidence_signal_layer(
    db: Session,
    *,
    knowledge_base_id: str,
    graph_state_id: str,
    graph_state_hash: str,
    atom_scope_hash: str,
    atoms: list[EvidenceAtom],
    community_state: CommunityState | None = None,
    batch_id: str | None = None,
) -> EvidenceSignalBuildResult:
    from app.services.ingestion_logs import emit_ingestion_log

    eligible_atoms = [atom for atom in atoms if atom.state == "active" and (atom.text or "").strip()]
    memberships = _load_memberships(db, community_state)
    signal_state = SignalState(
        knowledge_base_id=knowledge_base_id,
        evidence_graph_state_id=graph_state_id,
        evidence_community_state_id=community_state.id if community_state else None,
        signal_state_hash="pending",
        schema_hash=None,
        evidence_graph_state_hash=graph_state_hash,
        evidence_community_state_hash=community_state.state_hash if community_state else None,
        active_signal_scope_hash="pending",
        signal_protocol_version=SIGNAL_LAYER_PROTOCOL_VERSION,
        status="queued",
        eligible_atom_ids_json=[atom.id for atom in eligible_atoms],
        processed_atom_ids_json=[],
        model_audit_json={
            "llm_external_called": False,
            "measurement_mode": "algorithmic_evidence_bound",
            "fallback_used": False,
        },
        stats_json={},
        diagnostics_json={},
    )
    db.add(signal_state)
    db.flush()

    emit_ingestion_log(batch_id, "signal_candidate_scanning", "Evidence signal candidate scan started", signal_state_id=signal_state.id)
    _set_signal_state_status(signal_state, "scanning")
    drafts, scan_stats = _draft_candidates(eligible_atoms)
    signal_state.processed_atom_ids_json = [atom.id for atom in eligible_atoms]
    db.flush()

    sample_pack = _sample_pack(atoms=eligible_atoms, drafts=drafts, community_state=community_state, memberships=memberships)
    type_specs, relation_specs, schema_stats = _schema_spec_from_candidates(sample_pack, drafts)
    schema_passed, schema_failures = _schema_gate(type_specs, relation_specs)
    schema_hash = stable_hash({"types": type_specs, "relations": relation_specs, "protocol": SIGNAL_SCHEMA_PROTOCOL_VERSION})
    schema_state = SignalSchemaState(
        knowledge_base_id=knowledge_base_id,
        evidence_graph_state_id=graph_state_id,
        evidence_community_state_id=community_state.id if community_state else None,
        schema_hash=schema_hash,
        schema_protocol_version=SIGNAL_SCHEMA_PROTOCOL_VERSION,
        sample_pack_hash=schema_stats["sample_pack_hash"],
        llm_model_audit_json={
            "llm_external_called": False,
            "measurement_scope": "candidate_space_only",
            "diagnostic_only_without_evidence_binding": True,
        },
        status="active" if schema_passed else "failed",
        stats_json=schema_stats,
        diagnostics_json={"schema_gate_failures": schema_failures, "sample_pack": sample_pack if len(drafts) <= 40 else {"hash": schema_stats["sample_pack_hash"]}},
    )
    db.add(schema_state)
    db.flush()
    for spec in type_specs:
        db.add(
            SignalTypeSpec(
                schema_state_id=schema_state.id,
                name=spec["name"],
                description=spec["description"],
                evidence_patterns_json=spec["evidence_patterns"],
                applicable_atom_types_json=spec["applicable_atom_types"],
                risk=spec["risk"],
                retrieval_use_json=spec["retrieval_use"],
                gate_json=spec["gate"],
            )
        )
    for spec in relation_specs:
        db.add(
            SignalRelationSpec(
                schema_state_id=schema_state.id,
                name=spec["name"],
                source_signal_types_json=spec["source_signal_types"],
                target_signal_types_json=spec["target_signal_types"],
                required_evidence_json=spec["required_evidence"],
                risk=spec["risk"],
                gate_json=spec["gate"],
            )
        )
    signal_state.schema_state_id = schema_state.id
    signal_state.schema_hash = schema_hash
    if not schema_passed:
        _set_signal_state_status(signal_state, "failed")
        db.flush()
        emit_ingestion_log(batch_id, "signal_schema_failed", "Evidence signal schema gate failed", signal_state_id=signal_state.id, failures=schema_failures)
        raise RuntimeError("Evidence signal schema gate failed")
    db.flush()

    emit_ingestion_log(batch_id, "signal_candidate_gate", "Evidence signal candidate gate started", signal_state_id=signal_state.id, candidate_count=len(drafts))
    _set_signal_state_status(signal_state, "gating")
    drafts_by_key: dict[str, list[SignalDraft]] = defaultdict(list)
    for draft in drafts:
        drafts_by_key[draft.normalized_key].append(draft)
    quality_by_key = {key: _candidate_quality(key, grouped) for key, grouped in drafts_by_key.items()}

    candidates: list[SignalCandidate] = []
    decisions: list[SignalDecision] = []
    for key, grouped in sorted(drafts_by_key.items()):
        quality = quality_by_key[key]
        signal_type = Counter(draft.signal_type for draft in grouped).most_common(1)[0][0]
        canonical_label = Counter(draft.canonical_label for draft in grouped).most_common(1)[0][0]
        support_atom_ids = sorted({draft.atom.id for draft in grouped})
        spans = [draft.source_span for draft in grouped]
        candidate = SignalCandidate(
            signal_state_id=signal_state.id,
            knowledge_base_id=knowledge_base_id,
            candidate_type=signal_type,
            surface=canonical_label,
            normalized_key=key,
            evidence_atom_ids_json=support_atom_ids,
            source_span_union_json=_source_span_union_from_spans(spans),
            support_active_chunk_ids_json=[],
            extractor_name=",".join(sorted({draft.extractor_name for draft in grouped}))[:96],
            extractor_version=SIGNAL_EXTRACTOR_VERSION,
            features_json={
                "candidate_features": quality,
                "extractors": sorted({draft.extractor_name for draft in grouped}),
                "surface_variants": sorted({draft.surface for draft in grouped})[:12],
            },
            confidence=float(quality["confidence"]),
            status="accepted" if quality["action"] in {"accept", "keep_isolated"} else "rejected",
        )
        db.add(candidate)
        db.flush()
        decision = SignalDecision(
            signal_state_id=signal_state.id,
            candidate_id=candidate.id,
            decision_type="candidate_gate",
            action=quality["action"],
            reason_code="evidence_bound_measurement" if candidate.status == "accepted" else "insufficient_evidence_signal",
            support_evidence_atom_ids_json=support_atom_ids,
            source_span_union_json=candidate.source_span_union_json,
            algorithm_audit_json={"gate_protocol_version": SIGNAL_GATE_VERSION, "features": quality},
            llm_audit_json={"llm_external_called": False, "candidate_space_fixed": True},
            quality_json=quality,
        )
        db.add(decision)
        candidates.append(candidate)
        decisions.append(decision)
    db.flush()

    emit_ingestion_log(batch_id, "signal_layer_assembling", "Evidence signal active layer assembly started", signal_state_id=signal_state.id)
    _set_signal_state_status(signal_state, "assembling")
    nodes: list[SignalNode] = []
    node_by_key: dict[str, SignalNode] = {}
    for candidate in candidates:
        if candidate.status != "accepted":
            continue
        quality = (candidate.features_json or {}).get("candidate_features") or {}
        assert candidate.evidence_atom_ids_json, "SignalNode requires evidence atoms"
        assert (candidate.source_span_union_json or {}).get("spans"), "SignalNode requires source spans"
        node = SignalNode(
            signal_state_id=signal_state.id,
            knowledge_base_id=knowledge_base_id,
            signal_type=candidate.candidate_type,
            canonical_label=candidate.surface,
            normalized_key=candidate.normalized_key,
            support_atom_ids_json=candidate.evidence_atom_ids_json,
            support_active_chunk_ids_json=[],
            source_span_union_json=candidate.source_span_union_json,
            confidence=candidate.confidence,
            quality_json={
                "support_span_rate": quality.get("support_span_rate"),
                "support_atom_count": quality.get("support_atom_count"),
                "specificity": quality.get("specificity"),
                "evidence_strength": quality.get("evidence_strength"),
                "candidate_gate_action": quality.get("action"),
            },
            diagnostics_json={
                "candidate_id": candidate.id,
                "extractor_name": candidate.extractor_name,
                "signal_gate_version": SIGNAL_GATE_VERSION,
            },
        )
        db.add(node)
        nodes.append(node)
        node_by_key[candidate.normalized_key] = node
    db.flush()

    atom_to_nodes: dict[str, list[SignalNode]] = defaultdict(list)
    for node in nodes:
        for atom_id in node.support_atom_ids_json or []:
            atom_to_nodes[str(atom_id)].append(node)
    edge_support: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for atom_id, grouped in atom_to_nodes.items():
        ranked = sorted(grouped, key=lambda item: (item.confidence, len(item.support_atom_ids_json or [])), reverse=True)[:10]
        for index, left in enumerate(ranked):
            for right in ranked[index + 1 :]:
                left_id, right_id = sorted([left.id, right.id])
                edge_support[(left_id, right_id, "co_supported_by_atom")].add(atom_id)

    edges: list[SignalEdge] = []
    for (source_id, target_id, edge_type), support_atom_ids in sorted(edge_support.items(), key=lambda item: len(item[1]), reverse=True)[:3000]:
        support_spans: list[dict[str, Any]] = []
        for candidate in candidates:
            if set(candidate.evidence_atom_ids_json or []).intersection(support_atom_ids):
                support_spans.extend((candidate.source_span_union_json or {}).get("spans") or [])
        edge = SignalEdge(
            signal_state_id=signal_state.id,
            knowledge_base_id=knowledge_base_id,
            edge_type=edge_type,
            source_signal_id=source_id,
            target_signal_id=target_id,
            support_atom_ids_json=sorted(support_atom_ids),
            support_active_chunk_ids_json=[],
            source_span_union_json=_source_span_union_from_spans(support_spans),
            confidence=round(min(0.95, 0.52 + 0.06 * len(support_atom_ids)), 4),
            relation_source="observed_signal_co_support",
            diagnostics_json={"relation_protocol_version": SIGNAL_RELATION_PROTOCOL_VERSION},
        )
        db.add(edge)
        edges.append(edge)
    db.flush()

    communities, memberships_created = _create_signal_communities(db, signal_state=signal_state, nodes=nodes, edges=edges)
    signal_community_hash = stable_hash(
        {
            "signal_state_id": signal_state.id,
            "communities": [
                {"community_id": community.community_id, "stats": community.stats_json}
                for community in communities
            ],
        }
    )
    signal_state.signal_community_state_hash = signal_community_hash

    projection_state = _create_projection_view(db, knowledge_base_id=knowledge_base_id, signal_state=signal_state, nodes=nodes, edges=edges, communities=communities)

    active_scope = {
        "nodes": [
            {"id": node.id, "key": node.normalized_key, "atoms": node.support_atom_ids_json}
            for node in nodes
        ],
        "edges": [
            {"source": edge.source_signal_id, "target": edge.target_signal_id, "type": edge.edge_type, "atoms": edge.support_atom_ids_json}
            for edge in edges
        ],
    }
    active_signal_scope_hash = stable_hash(active_scope)
    signal_state.active_signal_scope_hash = active_signal_scope_hash
    signal_state.signal_state_hash = stable_hash(
        {
            "protocol": SIGNAL_LAYER_PROTOCOL_VERSION,
            "schema_hash": schema_hash,
            "evidence_graph_state_hash": graph_state_hash,
            "evidence_community_state_hash": community_state.state_hash if community_state else None,
            "signal_community_state_hash": signal_community_hash,
            "active_signal_scope_hash": active_signal_scope_hash,
            "atom_scope_hash": atom_scope_hash,
        }
    )
    stats = {
        **scan_stats,
        "signal_schema_state_id": schema_state.id,
        "signal_schema_hash": schema_hash,
        "signal_candidate_count": len(candidates),
        "accepted_signal_candidate_count": sum(1 for candidate in candidates if candidate.status == "accepted"),
        "rejected_signal_candidate_count": sum(1 for candidate in candidates if candidate.status != "accepted"),
        "signal_node_count": len(nodes),
        "signal_edge_count": len(edges),
        "signal_community_count": len(communities),
        "signal_community_membership_count": memberships_created,
        "projection_state_id": projection_state.id,
        "projection_hash": projection_state.projection_hash,
        "support_span_rate": _support_span_rate(nodes),
        "support_atom_rate": _support_atom_rate(nodes),
        "relation_support_rate": _relation_support_rate(edges),
        "schema_gate_pass_rate": 1.0 if schema_passed else 0.0,
        "candidate_gate_pass_rate": round(sum(1 for candidate in candidates if candidate.status == "accepted") / max(len(candidates), 1), 6),
        "llm_external_called": False,
        "fallback_used": False,
    }
    signal_state.stats_json = stats
    signal_state.diagnostics_json = {
        "complete_signal_layer": True,
        "evidence_first": True,
        "diagnostic_only_without_evidence_binding": True,
        "schema_gate_failures": schema_failures,
    }
    _set_signal_state_status(signal_state, "active")
    db.flush()
    emit_ingestion_log(batch_id, "signal_layer_active", "Evidence signal layer activated", signal_state_id=signal_state.id, signal_state_hash=signal_state.signal_state_hash, **stats)
    return EvidenceSignalBuildResult(
        state=signal_state,
        schema_state=schema_state,
        projection_state=projection_state,
        nodes=nodes,
        edges=edges,
        candidates=candidates,
        decisions=decisions,
        stats=stats,
    )


def _support_span_rate(nodes: list[SignalNode]) -> float:
    if not nodes:
        return 1.0
    return round(sum(1 for node in nodes if (node.source_span_union_json or {}).get("spans")) / len(nodes), 6)


def _support_atom_rate(nodes: list[SignalNode]) -> float:
    if not nodes:
        return 1.0
    return round(sum(1 for node in nodes if node.support_atom_ids_json) / len(nodes), 6)


def _relation_support_rate(edges: list[SignalEdge]) -> float:
    if not edges:
        return 1.0
    return round(sum(1 for edge in edges if edge.support_atom_ids_json and (edge.source_span_union_json or {}).get("spans")) / len(edges), 6)


def _create_signal_communities(db: Session, *, signal_state: SignalState, nodes: list[SignalNode], edges: list[SignalEdge]) -> tuple[list[SignalCommunity], int]:
    parent: dict[str, str] = {node.id: node.id for node in nodes}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in edges:
        union(edge.source_signal_id, edge.target_signal_id)

    groups: dict[str, list[SignalNode]] = defaultdict(list)
    for node in nodes:
        groups[find(node.id)].append(node)
    communities: list[SignalCommunity] = []
    memberships_created = 0
    for index, grouped in enumerate(sorted(groups.values(), key=lambda items: (-len(items), items[0].canonical_label if items else "")), start=1):
        community_id = f"signal-community-{index}"
        atom_ids = sorted({atom_id for node in grouped for atom_id in (node.support_atom_ids_json or [])})
        community = SignalCommunity(
            signal_state_id=signal_state.id,
            community_id=community_id,
            algorithm=SIGNAL_COMMUNITY_PROTOCOL_VERSION,
            stats_json={"signal_count": len(grouped), "support_atom_count": len(atom_ids)},
            diagnostics_json={"community_protocol_version": SIGNAL_COMMUNITY_PROTOCOL_VERSION},
        )
        db.add(community)
        db.flush()
        communities.append(community)
        for node in grouped:
            db.add(
                SignalCommunityMembership(
                    signal_community_id=community.id,
                    signal_state_id=signal_state.id,
                    signal_node_id=node.id,
                    community_id=community_id,
                    score=1.0,
                )
            )
            memberships_created += 1
    db.flush()
    return communities, memberships_created


def _create_projection_view(
    db: Session,
    *,
    knowledge_base_id: str,
    signal_state: SignalState,
    nodes: list[SignalNode],
    edges: list[SignalEdge],
    communities: list[SignalCommunity],
) -> ProjectionState:
    projection_hash = stable_hash(
        {
            "signal_state_id": signal_state.id,
            "signals": [{"id": node.id, "atoms": node.support_atom_ids_json} for node in nodes],
            "edges": [{"id": edge.id, "atoms": edge.support_atom_ids_json} for edge in edges],
            "protocol": PROJECTION_PROTOCOL_VERSION,
        }
    )
    projection_state = ProjectionState(
        knowledge_base_id=knowledge_base_id,
        signal_state_id=signal_state.id,
        view="overview",
        projection_hash=projection_hash,
        projection_protocol_version=PROJECTION_PROTOCOL_VERSION,
        status="active",
        stats_json={"signal_node_count": len(nodes), "signal_edge_count": len(edges), "projection_community_count": len(communities)},
        diagnostics_json={"derived_view": True, "not_fact_source": True},
    )
    db.add(projection_state)
    db.flush()
    projection_node_by_signal_id: dict[str, ProjectionNode] = {}
    for node in nodes:
        projection_node = ProjectionNode(
            projection_state_id=projection_state.id,
            knowledge_base_id=knowledge_base_id,
            source_kind="signal_node",
            source_id=node.id,
            label=node.canonical_label,
            category=node.signal_type,
            support_atom_ids_json=node.support_atom_ids_json,
            support_active_chunk_ids_json=node.support_active_chunk_ids_json,
            source_span_union_json=node.source_span_union_json,
            diagnostics_json={"signal_state_id": signal_state.id},
        )
        db.add(projection_node)
        projection_node_by_signal_id[node.id] = projection_node
    db.flush()
    for edge in edges:
        source = projection_node_by_signal_id.get(edge.source_signal_id)
        target = projection_node_by_signal_id.get(edge.target_signal_id)
        if source is None or target is None:
            continue
        db.add(
            ProjectionEdge(
                projection_state_id=projection_state.id,
                knowledge_base_id=knowledge_base_id,
                source_node_id=source.id,
                target_node_id=target.id,
                edge_type=edge.edge_type,
                support_atom_ids_json=edge.support_atom_ids_json,
                source_span_union_json=edge.source_span_union_json,
                confidence=edge.confidence,
                diagnostics_json={"signal_edge_id": edge.id, "relation_source": edge.relation_source},
            )
        )
    for community in communities:
        collapsed_node_ids = [
            membership.signal_node_id
            for membership in db.scalars(select(SignalCommunityMembership).where(SignalCommunityMembership.signal_community_id == community.id)).all()
        ]
        db.add(
            ProjectionCommunity(
                projection_state_id=projection_state.id,
                community_id=community.community_id,
                source_community_ids_json=[community.community_id],
                collapsed_node_ids_json=collapsed_node_ids,
                stats_json=community.stats_json,
                diagnostics_json={"source": "signal_community"},
            )
        )
    db.flush()
    return projection_state


def load_active_signal_state(db: Session, *, knowledge_base_id: str, graph_state_id: str) -> SignalState | None:
    return db.scalar(
        select(SignalState)
        .where(
            SignalState.knowledge_base_id == knowledge_base_id,
            SignalState.evidence_graph_state_id == graph_state_id,
            SignalState.status == "active",
        )
        .order_by(SignalState.created_at.desc())
    )


def signal_features_for_atoms(
    db: Session,
    *,
    signal_state: SignalState | None,
    atoms: list[EvidenceAtom],
) -> dict[str, Any]:
    atom_ids = {atom.id for atom in atoms}
    if signal_state is None or not atom_ids:
        return {
            "signal_layer_complete": False,
            "signal_coverage": 0.0,
            "signal_fragmentation": 0.0,
            "signal_boundary_cut_cost": 0.0,
            "signal_support_closure": 1.0,
            "dominant_signal_ids": [],
            "dominant_signal_labels": [],
        }
    nodes = db.scalars(
        select(SignalNode).where(
            SignalNode.signal_state_id == signal_state.id,
        )
    ).all()
    visible_nodes = [node for node in nodes if set(str(item) for item in (node.support_atom_ids_json or [])).intersection(atom_ids)]
    node_ids = {node.id for node in visible_nodes}
    all_edges = db.scalars(
        select(SignalEdge).where(
            SignalEdge.signal_state_id == signal_state.id,
            (
                SignalEdge.source_signal_id.in_(node_ids)
                | SignalEdge.target_signal_id.in_(node_ids)
            ) if node_ids else False,
        )
    ).all() if node_ids else []
    crossing = [
        edge
        for edge in all_edges
        if (edge.source_signal_id in node_ids) != (edge.target_signal_id in node_ids)
        and set(edge.support_atom_ids_json or []).intersection(atom_ids)
    ]
    atom_with_signals = {atom_id for node in visible_nodes for atom_id in (node.support_atom_ids_json or []) if atom_id in atom_ids}
    ranked_nodes = sorted(visible_nodes, key=lambda item: (len(item.support_atom_ids_json or []), item.confidence), reverse=True)[:8]
    return {
        "signal_layer_complete": signal_state.status == "active",
        "signal_state_id": signal_state.id,
        "signal_state_hash": signal_state.signal_state_hash,
        "signal_coverage": round(len(atom_with_signals) / max(len(atom_ids), 1), 6),
        "signal_fragmentation": round(len(visible_nodes) / max(len(atom_ids), 1), 6),
        "signal_boundary_cut_cost": round(sum(edge.confidence for edge in crossing), 6),
        "signal_support_closure": round(1.0 - (len(crossing) / max(len(all_edges), 1)), 6),
        "dominant_signal_ids": [node.id for node in ranked_nodes],
        "dominant_signal_labels": [node.canonical_label for node in ranked_nodes],
    }


def attach_active_chunks_to_signal_layer(
    db: Session,
    *,
    signal_state: SignalState | None,
    active_chunks: dict[str, ActiveChunk],
) -> None:
    if signal_state is None or not active_chunks:
        return
    active_chunks_by_atom: dict[str, set[str]] = defaultdict(set)
    for active_chunk in active_chunks.values():
        for atom_id in active_chunk.atom_ids_json or []:
            active_chunks_by_atom[str(atom_id)].add(active_chunk.id)
    nodes = db.scalars(select(SignalNode).where(SignalNode.signal_state_id == signal_state.id)).all()
    for node in nodes:
        support_chunks = sorted(
            {
                active_chunk_id
                for atom_id in node.support_atom_ids_json or []
                for active_chunk_id in active_chunks_by_atom.get(str(atom_id), set())
            }
        )
        node.support_active_chunk_ids_json = support_chunks
    node_chunks = {node.id: set(node.support_active_chunk_ids_json or []) for node in nodes}
    edges = db.scalars(select(SignalEdge).where(SignalEdge.signal_state_id == signal_state.id)).all()
    for edge in edges:
        edge.support_active_chunk_ids_json = sorted(node_chunks.get(edge.source_signal_id, set()).intersection(node_chunks.get(edge.target_signal_id, set())))
    projection = db.scalar(select(ProjectionState).where(ProjectionState.signal_state_id == signal_state.id).order_by(ProjectionState.created_at.desc()))
    if projection is not None:
        projection_nodes = db.scalars(select(ProjectionNode).where(ProjectionNode.projection_state_id == projection.id)).all()
        for projection_node in projection_nodes:
            if projection_node.source_kind != "signal_node":
                continue
            signal_node = next((node for node in nodes if node.id == projection_node.source_id), None)
            if signal_node is not None:
                projection_node.support_active_chunk_ids_json = signal_node.support_active_chunk_ids_json
    db.flush()
