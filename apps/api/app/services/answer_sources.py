"""Exact answer-unit/source binding for the reflection answer protocol."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentObservation, AgentRun, AnswerSession, AnswerSourceBinding, Chunk, ContextPackage, generate_uuid
from app.reflection_contracts import SOURCE_BINDING_PROTOCOL, PATH_SUPPORT_PROTOCOL, AnswerDraft
from app.retrieval_control_contracts import (
    GroundedAnswerDraft,
    GroundedMarkdownAnswerDraft,
    SourceGateAdmission,
    control_hash,
)
from app.services.agent_reflection import ReflectionContractError, reflection_hash, render_answer_units, source_path_metrics, validate_draft_sources
from app.services.citation_provenance import audit_citation_provenance, replay_citation_provenance_for_persistence


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _manifest_identity(package_id: str, trace_id: str, sources) -> dict[str, Any]:
    return {
        "protocol_version": SOURCE_BINDING_PROTOCOL,
        "package_id": package_id, "retrieval_trace_id": trace_id,
        "sources": [{
            "source_handle": item["source_handle"], "chunk_id": item["chunk_id"],
            "text_hash": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
            "source_span": item["source_span"],
            "source_label": {key: item["package_item"].get(key) for key in (
                "document_id", "document_title", "source_path", "logical_source_path",
            )},
        } for item in sources],
    }


@dataclass(frozen=True)
class AnswerEvidenceManifest:
    package_id: str
    retrieval_trace_id: str
    sources: tuple[dict[str, Any], ...]
    manifest_hash: str

    def verify_integrity(self) -> None:
        if reflection_hash(_manifest_identity(self.package_id, self.retrieval_trace_id, self.sources)) != self.manifest_hash:
            raise ReflectionContractError("evidence_manifest_mutated")

    def model_sources(self) -> list[dict[str, Any]]:
        # Provider input contains opaque handles and complete source text.
        # Lifecycle ids, fingerprints and filesystem addresses stay server-side.
        self.verify_integrity()
        return [{"source_handle": item["source_handle"], "text": item["text"], "source_label": {
            "document_title": str(item["package_item"].get("document_title") or "")[:400],
            "section": str(item["source_span"].get("section_path") or "")[:400],
            "pages": list(item["source_span"].get("page_range") or [])[:2],
        }} for item in self.sources]

    def by_handle(self) -> dict[str, dict[str, Any]]:
        self.verify_integrity()
        return {item["source_handle"]: _clone(item) for item in self.sources}

    def semantic_hash(self) -> str:
        self.verify_integrity()
        facts = [{
            "chunk_id": source["chunk_id"],
            "document_version_id": source["source_span"]["document_version_id"],
            "char_span": source["source_span"]["char_span"],
            "text_hash": hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
        } for source in self.sources]
        return reflection_hash(sorted(facts, key=lambda value: (value["chunk_id"], value["char_span"])))


def project_answer_evidence_manifest(
    evidence: AnswerEvidenceManifest,
    package_source_handles: list[str] | tuple[str, ...],
) -> AnswerEvidenceManifest:
    """Project admitted package sources into a compact generation-local view."""

    sources = evidence.by_handle()
    selected = tuple(package_source_handles)
    if len(set(selected)) != len(selected) or any(handle not in sources for handle in selected):
        raise ReflectionContractError("generation_evidence_selection_invalid")
    projected: list[dict[str, Any]] = []
    for index, package_handle in enumerate(selected, start=1):
        source = _clone(sources[package_handle])
        source["package_source_handle"] = package_handle
        source["source_handle"] = f"src_{index}"
        projected.append(source)
    identity = _manifest_identity(evidence.package_id, evidence.retrieval_trace_id, projected)
    return AnswerEvidenceManifest(
        evidence.package_id,
        evidence.retrieval_trace_id,
        tuple(projected),
        reflection_hash(identity),
    )


def build_generation_evidence_view(
    evidence: AnswerEvidenceManifest,
    package_source_handles: list[str] | tuple[str, ...],
    *,
    coverage: dict[str, list[str]],
    source_integrity_admission_hash: str,
) -> tuple[AnswerEvidenceManifest, dict[str, Any]]:
    """Freeze a replayable projection without copying source text."""

    projected = project_answer_evidence_manifest(evidence, package_source_handles)
    by_package_handle = evidence.by_handle()
    selected = tuple(package_source_handles)
    selected_set = set(selected)
    if any(
        not isinstance(requirement_id, str)
        or not requirement_id
        or len(set(handles)) != len(handles)
        or any(handle not in selected_set for handle in handles)
        for requirement_id, handles in coverage.items()
    ):
        raise ReflectionContractError("generation_evidence_coverage_invalid")
    source_cards = []
    for generation_source, package_handle in zip(projected.sources, selected, strict=True):
        package_source = by_package_handle[package_handle]
        span = package_source["source_span"]
        source_cards.append({
            "package_source_handle": package_handle,
            "generation_source_handle": generation_source["source_handle"],
            "chunk_id": package_source["chunk_id"],
            "document_version_id": span["document_version_id"],
            "char_span": list(span["char_span"]),
            "text_hash": hashlib.sha256(package_source["text"].encode("utf-8")).hexdigest(),
            "source_span_hash": control_hash(span),
        })
    payload = {
        "protocol_version": "generation_evidence_view_v1",
        "context_package_id": evidence.package_id,
        "retrieval_trace_id": evidence.retrieval_trace_id,
        "source_integrity_admission_hash": source_integrity_admission_hash,
        "admitted_evidence_manifest_hash": evidence.manifest_hash,
        "generation_evidence_manifest_hash": projected.manifest_hash,
        "selected_package_handles": list(selected),
        "coverage": {key: list(value) for key, value in sorted(coverage.items())},
        "sources": source_cards,
        "source_text_persisted": False,
    }
    return projected, {**payload, "audit_hash": control_hash(payload)}


def replay_generation_evidence_view(
    admitted_evidence: AnswerEvidenceManifest,
    generation_evidence: AnswerEvidenceManifest,
    view: dict[str, Any],
) -> None:
    """Verify the selected view against the complete admitted manifest."""

    if not isinstance(view, dict):
        raise ReflectionContractError("generation_evidence_view_missing")
    unsigned = {key: value for key, value in view.items() if key != "audit_hash"}
    selected = view.get("selected_package_handles")
    coverage = view.get("coverage")
    if (
        view.get("protocol_version") != "generation_evidence_view_v1"
        or view.get("context_package_id") != admitted_evidence.package_id
        or view.get("retrieval_trace_id") != admitted_evidence.retrieval_trace_id
        or view.get("admitted_evidence_manifest_hash") != admitted_evidence.manifest_hash
        or view.get("generation_evidence_manifest_hash") != generation_evidence.manifest_hash
        or view.get("source_text_persisted") is not False
        or not isinstance(selected, list)
        or not isinstance(coverage, dict)
        or view.get("audit_hash") != control_hash(unsigned)
    ):
        raise ReflectionContractError("generation_evidence_view_identity_invalid")
    replayed, expected = build_generation_evidence_view(
        admitted_evidence,
        selected,
        coverage={str(key): list(value) for key, value in coverage.items()},
        source_integrity_admission_hash=str(view.get("source_integrity_admission_hash") or ""),
    )
    if replayed.manifest_hash != generation_evidence.manifest_hash or expected != view:
        raise ReflectionContractError("generation_evidence_view_replay_failed")


def build_answer_evidence_manifest(package: ContextPackage, contexts: list[dict[str, Any]]) -> AnswerEvidenceManifest:
    if not package.id or not package.retrieval_trace_id:
        raise ReflectionContractError("evidence_package_identity_missing")
    package_chunks = (package.package_json or {}).get("chunks") or []
    by_chunk = {str(item.get("chunk_id") or ""): item for item in contexts}
    package_ids = [str(item.get("chunk_id") or "") for item in package_chunks]
    if (
        not package_ids or "" in package_ids or len(set(package_ids)) != len(package_ids)
        or len(by_chunk) != len(contexts) or set(package_ids) != set(by_chunk)
    ):
        raise ReflectionContractError("evidence_context_scope_mismatch")
    sources: list[dict[str, Any]] = []
    for index, item in enumerate(package_chunks, start=1):
        context = by_chunk[str(item["chunk_id"])]
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or "\x00" in content:
            raise ReflectionContractError("evidence_source_text_invalid")
        if content != context.get("content"):
            raise ReflectionContractError("generation_context_text_mismatch")
        span = _clone(item.get("source_span") or {})
        if (
            not span.get("raw_span_text_hash") or not span.get("document_version_id")
            or str(span.get("chunk_id") or "") != str(item["chunk_id"])
            or str(span.get("context_package_id") or "") != str(package.id)
            or str(span.get("retrieval_trace_id") or "") != str(package.retrieval_trace_id)
        ):
            raise ReflectionContractError("evidence_source_address_incomplete")
        sources.append({
            "source_handle": f"src_{index}", "chunk_id": str(item["chunk_id"]),
            "text": content, "source_span": span, "package_item": _clone(item),
        })
    identity = _manifest_identity(package.id, package.retrieval_trace_id, sources)
    return AnswerEvidenceManifest(str(package.id), str(package.retrieval_trace_id), tuple(sources), reflection_hash(identity))


def draft_source_candidates(
    draft: AnswerDraft,
    evidence: AnswerEvidenceManifest,
    *,
    unit_limit: int,
) -> list[dict[str, Any]]:
    sources = evidence.by_handle()
    validate_draft_sources(draft, list(sources), unit_limit=unit_limit)
    _answer, units = render_answer_units(draft)
    candidates: list[dict[str, Any]] = []
    for unit in units:
        for handle in unit["source_handles"]:
            source = sources[handle]
            item = source["package_item"]
            candidates.append({
                "citation_index": len(candidates) + 1,
                "unit_id": unit["unit_id"], "unit_index": unit["unit_index"], "unit_text": unit["text"],
                "answer_hash": unit["answer_hash"], "answer_char_span": unit["char_span"],
                "source_handle": handle, "chunk_id": item["chunk_id"], "document_id": item["document_id"],
                "document_version_id": item["document_version_id"],
                "context_package_id": evidence.package_id, "retrieval_trace_id": evidence.retrieval_trace_id,
                "source_path": item["source_path"], "logical_source_path": item["logical_source_path"],
                "page_number": (item.get("page_range") or [None])[0],
                "section": item.get("section_path"),
                "source_span": _clone(source["source_span"]),
            })
    return candidates


def audit_answer_sources(
    db: Session, *, knowledge_base_id: str, package: ContextPackage, contexts: list[dict[str, Any]],
    draft: AnswerDraft, evidence: AnswerEvidenceManifest, unit_limit: int,
    admitted_evidence: AnswerEvidenceManifest | None = None,
    generation_evidence_view: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replayed_manifest = build_answer_evidence_manifest(package, contexts)
    if admitted_evidence is None:
        if replayed_manifest.manifest_hash != evidence.manifest_hash:
            raise ReflectionContractError("evidence_manifest_changed")
    else:
        if replayed_manifest.manifest_hash != admitted_evidence.manifest_hash:
            raise ReflectionContractError("admitted_evidence_manifest_changed")
        replay_generation_evidence_view(
            admitted_evidence,
            evidence,
            generation_evidence_view or {},
        )
    candidates = draft_source_candidates(draft, evidence, unit_limit=unit_limit)
    if not candidates:
        return [], {"all_valid": True, "valid_count": 0, "invalid_count": 0, "nonfactual_only": True}
    audit = audit_citation_provenance(
        db, knowledge_base_id=knowledge_base_id, package=package, citations=candidates, contexts=contexts,
    )
    return candidates, audit


def _source_binding_authority(
    *,
    reflection_audit_hash,
    retrieval_gate,
    package,
    source_integrity_admission=None,
):
    if source_integrity_admission is not None:
        if reflection_audit_hash is not None or retrieval_gate is not None:
            raise ReflectionContractError("source_binding_mixed_authority")
        payload = dict(source_integrity_admission.observation_json or {})
        unsigned = {key: value for key, value in payload.items() if key != "audit_hash"}
        if (
            source_integrity_admission.observation_type != "source_integrity_admission"
            or source_integrity_admission.verdict != "passed"
            or payload.get("protocol_version") != "source_integrity_admission_v1"
            or payload.get("outcome") != "passed"
            or payload.get("context_package_id") != package.id
            or payload.get("retrieval_trace_id") != package.retrieval_trace_id
            or payload.get("model_call_count") != 0
            or payload.get("score_fields_used_for_admission") != []
            or payload.get("audit_hash") != control_hash(unsigned)
        ):
            raise ReflectionContractError("source_integrity_admission_invalid")
        return {
            "protocol_version": "answer_source_binding_v2",
            "source_integrity_admission_hash": payload["audit_hash"],
            "source_integrity_admission_observation_id": source_integrity_admission.id,
        }, None
    if retrieval_gate is None:
        if (not isinstance(reflection_audit_hash, str) or len(reflection_audit_hash) != 64
            or any(c not in "0123456789abcdef" for c in reflection_audit_hash)):
            raise ReflectionContractError("reflection_audit_identity_invalid")
        return {"protocol_version": SOURCE_BINDING_PROTOCOL, "reflection_audit_hash": reflection_audit_hash}, None
    if reflection_audit_hash is not None:
        raise ReflectionContractError("source_binding_mixed_authority")
    try:
        admission = SourceGateAdmission.model_validate((retrieval_gate.observation_json or {}).get("source_admission"))
    except ValidationError:
        raise ReflectionContractError("retrieval_source_gate_contract_invalid") from None
    if (not retrieval_gate.id or retrieval_gate.observation_type != "retrieval_gate"
        or retrieval_gate.verdict != admission.outcome or retrieval_gate.run_id != admission.run_id
        or admission.context_package_id != package.id or admission.retrieval_trace_id != package.retrieval_trace_id
        or admission.knowledge_base_id != package.knowledge_base_id
        or retrieval_gate.observation_json.get("source_admission_hash") != admission.identity
        or len(set(admission.source_chunk_ids)) != len(admission.source_chunk_ids)
        or set(admission.source_chunk_ids) != {item["chunk_id"] for item in package.package_json["chunks"]}):
        raise ReflectionContractError("retrieval_source_gate_identity_invalid")
    inputs = retrieval_gate.observation_json.get('feature_input') or {}
    if inputs.get('parameters',{}).get('protocol_version') == 'canonical_task_path_quality_v4':
        from app.retrieval_control_contracts import (
            TaskContract, LexicalStrategy, DecisionPanel, PathFeatureCandidate, PathEvaluationParameters,
        )
        from app.services.retrieval_path_features import compute_path_features
        parameters=PathEvaluationParameters.model_validate(inputs['parameters'])
        if parameters.scope_selection is not None and parameters.scope_selection.run_id!=admission.run_id:
            raise ReflectionContractError('source_location_run_identity_changed')
        replay = compute_path_features(task=TaskContract.model_validate(inputs['task']),
            strategy=LexicalStrategy.model_validate(inputs['strategy']),
            panels=tuple(DecisionPanel.model_validate(item) for item in inputs['panels']),
            packaged_candidates=tuple(PathFeatureCandidate.model_validate(item) for item in inputs['package']),
            parameters=parameters)
        if (replay.model_dump(mode='json') != retrieval_gate.observation_json.get('features')
                or admission.feature_hash != control_hash(replay.model_dump(mode='json'))
                or admission.task_hash != replay.task_hash or admission.strategy_hash != replay.strategy_hash
                or any(status.state != 'satisfied' for status in replay.scope_statuses)):
            raise ReflectionContractError('retrieval_source_scope_gate_replay_failed')
    if admission.evidence_sufficiency_hash is not None:
        from app.retrieval_control_contracts import TaskContract, PathEvaluationParameters, PathFeatureSummary, RetrievalGateDecision
        from app.services.context_graph import context_package_to_contexts
        from app.services.evidence_scope import generation_scope_guidance
        from app.services.retrieval_sufficiency import replay_sufficiency, constrain_gate
        task = TaskContract.model_validate(inputs['task'])
        manifest = build_answer_evidence_manifest(package,context_package_to_contexts(package))
        payload = retrieval_gate.observation_json.get('evidence_sufficiency') or {}
        path_decision = RetrievalGateDecision.model_validate(retrieval_gate.observation_json['path_decision'])
        from app.services.source_addressed_assessment import replay_assessment_guidance
        assessment, guidance = replay_assessment_guidance(payload, task=task,
            features=PathFeatureSummary.model_validate(retrieval_gate.observation_json['features']),
            path_decision=path_decision,replay_input=inputs,evidence=manifest)
        if admission.source_addressed_assessment_hash != (assessment.identity if assessment else None):
            raise ReflectionContractError('source_addressed_admission_identity_changed')
        if assessment and assessment.provenance_session_hash != admission.provenance_session_hash:
            raise ReflectionContractError('source_addressed_provenance_changed')
        result = replay_sufficiency(payload,task=task,strategy_hash=admission.strategy_hash,evidence=manifest,source_scopes=guidance)
        effective = constrain_gate(path_decision,task=task,result=result,remaining_repairs=0,assessment=assessment)
        if (payload['audit_hash'] != admission.evidence_sufficiency_hash
                or effective.outcome != admission.outcome or manifest.manifest_hash != admission.evidence_manifest_hash
                or effective.model_dump(mode='json') != retrieval_gate.observation_json['decision']):
            raise ReflectionContractError('retrieval_sufficiency_admission_replay_failed')
    return {"protocol_version": "answer_source_binding_v2",
            "retrieval_gate_audit_hash": admission.identity,
            "retrieval_gate_observation_id": retrieval_gate.id}, admission


def verify_sufficiency_owner(db, *, owner, retrieval_gate, required=False):
    from app.services.retrieval_sufficiency import PROTOCOL
    payload = retrieval_gate.observation_json or {}
    admission = SourceGateAdmission.model_validate(payload.get('source_admission'))
    from app.services.source_addressed_assessment import verify_assessment_owner
    verify_assessment_owner(owner, payload)
    if owner is not None:
        from app.retrieval_control_contracts import TaskContract
        from app.services.retrieval_models import replay_run_source_reference_roles
        try:
            task_payload=(payload.get('feature_input') or {}).get('task') or (owner.metadata_json or {}).get('retrieval_task')
            if task_payload is not None:
                replay_run_source_reference_roles(owner.metadata_json or {},TaskContract.model_validate(task_payload))
        except (TypeError,ValueError):
            raise ReflectionContractError('source_scope_reference_role_owner_invalid') from None
    if admission.resolved_scope_filter_hash is not None:
        from app.schemas import SearchFilters
        from app.services.context_graph import passes_filters
        filters=(owner.metadata_json or {}).get('resolved_scope_filter') if owner is not None else None
        if (not filters or filters.get('task_hash')!=admission.task_hash
            or filters.get('audit_hash')!=admission.resolved_scope_filter_hash
            or filters['audit_hash']!=control_hash({k:v for k,v in filters.items() if k!='audit_hash'})):
            raise ReflectionContractError('source_scope_filter_audit_changed')
        effective=SearchFilters.model_validate(filters['effective_filters'])
        if any(not passes_filters(db,db.get(Chunk,cid),effective) for cid in admission.source_chunk_ids):
            raise ReflectionContractError('source_scope_filter_source_outside_domain')
    required = required or (owner is not None and (owner.metadata_json or {}).get('evidence_sufficiency_protocol') == PROTOCOL)
    if not required and admission.evidence_sufficiency_hash is None:
        return
    audit = payload.get('evidence_sufficiency') or {}
    row = db.get(AgentObservation,audit.get('observation_id')) if audit.get('observation_id') else None
    if (owner is None or owner.id != admission.run_id or admission.evidence_sufficiency_hash != audit.get('audit_hash')
            or not admission.evidence_sufficiency_hash or row is None or row.run_id != owner.id
            or row.observation_type != 'retrieval_sufficiency' or row.verdict != 'completed'
            or row.observation_json != audit or payload.get('admission_pending')):
        raise ReflectionContractError('retrieval_required_sufficiency_missing')


def persist_answer_source_bindings(
    db: Session, *, answer_session: AnswerSession, package: ContextPackage, contexts: list[dict[str, Any]],
    draft: AnswerDraft | GroundedAnswerDraft | GroundedMarkdownAnswerDraft,
    evidence: AnswerEvidenceManifest, unit_limit: int,
    reflection_audit_hash: str | None = None, retrieval_gate: AgentObservation | None = None,
    source_integrity_admission: AgentObservation | None = None,
    admitted_evidence: AnswerEvidenceManifest | None = None,
    generation_evidence_view: dict[str, Any] | None = None,
) -> list[AnswerSourceBinding]:
    """Flush inside the caller's answer/run transaction; never commit a half-answer."""
    answer, _units = render_answer_units(draft)
    if (
        answer_session.answer != answer or not answer_session.id
        or answer_session.knowledge_base_id != package.knowledge_base_id
        or answer_session.context_package_id != package.id
        or answer_session.retrieval_trace_id != package.retrieval_trace_id
    ):
        raise ReflectionContractError("answer_source_session_binding_mismatch")
    authority, admission = _source_binding_authority(
        reflection_audit_hash=reflection_audit_hash,
        retrieval_gate=retrieval_gate,
        package=package,
        source_integrity_admission=source_integrity_admission,
    )
    generation_identity: dict[str, Any] = {}
    if generation_evidence_view is not None:
        if admitted_evidence is None:
            raise ReflectionContractError("generation_evidence_admitted_manifest_missing")
        replay_generation_evidence_view(
            admitted_evidence,
            evidence,
            generation_evidence_view,
        )
        generation_identity = {
            "generation_evidence_view_hash": generation_evidence_view["audit_hash"],
            "generation_evidence_manifest_hash": evidence.manifest_hash,
        }
    if admission is not None:
        owner = db.get(AgentRun, admission.run_id)
        verify_sufficiency_owner(db,owner=owner,retrieval_gate=retrieval_gate)
        if (owner is None or owner.knowledge_base_id != answer_session.knowledge_base_id
            or owner.status != "running" or owner.question != answer_session.question
            or owner.session_id != answer_session.qa_session_id
            or ((owner.metadata_json or {}).get("retrieval_control") or {}).get("task_hash") != admission.task_hash
            or admission.evidence_manifest_hash != (
                admitted_evidence.manifest_hash if admitted_evidence is not None else evidence.manifest_hash
            )):
            raise ReflectionContractError("retrieval_source_gate_scope_changed")
        inputs = retrieval_gate.observation_json.get('feature_input') or {}
        if inputs.get('parameters',{}).get('protocol_version') == 'canonical_task_path_quality_v4':
            from app.retrieval_control_contracts import PathEvaluationParameters, TaskContract
            from app.services.evidence_scope import replay_scope_inputs
            replay_scope_inputs(db,task=TaskContract.model_validate(inputs['task']),
                parameters=PathEvaluationParameters.model_validate(inputs['parameters']),package=package,for_update=True)
    candidates, preflight = audit_answer_sources(
        db, knowledge_base_id=answer_session.knowledge_base_id, package=package, contexts=contexts,
        draft=draft, evidence=evidence, unit_limit=unit_limit,
        admitted_evidence=admitted_evidence,
        generation_evidence_view=generation_evidence_view,
    )
    if admission is not None:
        from app.services.source_addressed_assessment import validate_bound_source_addresses
        validate_bound_source_addresses(retrieval_gate.observation_json,
            ((item['chunk_id'],item['source_span']['char_span']) for item in candidates))
    if not candidates:
        return []
    if not preflight["all_valid"]:
        raise ReflectionContractError("source_provenance_failed")
    replay = replay_citation_provenance_for_persistence(
        db, knowledge_base_id=answer_session.knowledge_base_id, package=package,
        citations=candidates, contexts=contexts, expected_session_hash=preflight["provenance_session_hash"],
    )
    if not replay["persistence_gate_passed"]:
        raise ReflectionContractError("source_persistence_replay_failed")
    rows: list[AnswerSourceBinding] = []
    for candidate in candidates:
        start, end = candidate["answer_char_span"]
        identity = {
            **authority, **generation_identity,
            "knowledge_base_id": answer_session.knowledge_base_id, "answer_session_id": answer_session.id,
            "question_hash": hashlib.sha256(answer_session.question.encode("utf-8")).hexdigest(),
            "evidence_manifest_hash": evidence.manifest_hash,
            **candidate,
        }
        row = AnswerSourceBinding(
            id=generate_uuid(), knowledge_base_id=answer_session.knowledge_base_id, answer_session_id=answer_session.id,
            context_package_id=package.id, retrieval_trace_id=package.retrieval_trace_id,
            chunk_id=candidate["chunk_id"], unit_id=candidate["unit_id"], unit_index=candidate["unit_index"],
            unit_text=candidate["unit_text"], answer_char_start=start, answer_char_end=end,
            answer_hash=candidate["answer_hash"], protocol_version=authority["protocol_version"],
            retrieval_gate_observation_id=(
                authority.get("retrieval_gate_observation_id")
                or authority.get("source_integrity_admission_observation_id")
            ),
            source_span_json=_clone(candidate["source_span"]), binding_hash=reflection_hash(identity),
            diagnostics_json={
                "status": "source_bound", "semantic_entailment_claimed": False,
                "citation_index": candidate["citation_index"], "source_handle": candidate["source_handle"],
                "evidence_manifest_hash": evidence.manifest_hash, **authority, **generation_identity,
                "provenance_session_hash": replay["provenance_session_hash"],
                "transactional_replay": replay["transactional_replay"], "rows_locked": replay["rows_locked"],
                "binding_identity": identity,
            },
        )
        rows.append(row)
    for offset in range(0, len(rows), 64):
        db.add_all(rows[offset:offset + 64])
        db.flush()
    return rows


def source_binding_citations(
    *, answer_session: AnswerSession, package: ContextPackage,
    rows: list[AnswerSourceBinding], reflection_audit_hash: str | None = None,
    retrieval_gate: AgentObservation | None = None,
    source_integrity_admission: AgentObservation | None = None,
    generation_evidence_view: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from app.schemas import Citation

    authority, _admission = _source_binding_authority(
        reflection_audit_hash=reflection_audit_hash,
        retrieval_gate=retrieval_gate,
        package=package,
        source_integrity_admission=source_integrity_admission,
    )
    if _admission is not None:
        from app.services.source_addressed_assessment import validate_bound_source_addresses
        validate_bound_source_addresses(retrieval_gate.observation_json,
            ((row.chunk_id,row.source_span_json['char_span']) for row in rows))
    generation_identity = (
        {
            "generation_evidence_view_hash": generation_evidence_view["audit_hash"],
            "generation_evidence_manifest_hash": generation_evidence_view[
                "generation_evidence_manifest_hash"
            ],
        }
        if generation_evidence_view is not None
        else {}
    )
    if generation_evidence_view is None and any(
        (row.diagnostics_json or {}).get("generation_evidence_view_hash")
        for row in rows
    ):
        raise ReflectionContractError("generation_evidence_view_required")
    items = {item["chunk_id"]: item for item in (package.package_json or {}).get("chunks", [])}
    citations: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value.diagnostics_json["citation_index"])):
        identity = row.diagnostics_json.get("binding_identity") or {}
        expected_answer_hash = hashlib.sha256(answer_session.answer.encode("utf-8")).hexdigest()
        row_values = {
            "answer_session_id": answer_session.id,
            "knowledge_base_id": answer_session.knowledge_base_id,
            "context_package_id": package.id, "retrieval_trace_id": package.retrieval_trace_id,
            "unit_id": row.unit_id, "unit_index": row.unit_index, "unit_text": row.unit_text,
            "answer_hash": row.answer_hash, "chunk_id": row.chunk_id,
            "answer_char_span": [row.answer_char_start, row.answer_char_end],
            "source_span": row.source_span_json,
            **generation_identity,
        }
        if (
            row.chunk_id not in items or reflection_hash(identity) != row.binding_hash
            or any(identity.get(key) != value for key, value in row_values.items())
            or row.answer_session_id != answer_session.id or row.context_package_id != package.id
            or row.retrieval_trace_id != package.retrieval_trace_id
            or row.knowledge_base_id != answer_session.knowledge_base_id
            or row.answer_hash != expected_answer_hash
            or answer_session.answer[row.answer_char_start:row.answer_char_end] != row.unit_text
            or any((key != "protocol_version" and row.diagnostics_json.get(key) != value) or identity.get(key) != value
                   for key, value in authority.items())
            or any(
                row.diagnostics_json.get(key) != value or identity.get(key) != value
                for key, value in generation_identity.items()
            )
            or row.protocol_version != authority["protocol_version"]
                or row.retrieval_gate_observation_id != (
                    authority.get("retrieval_gate_observation_id")
                    or authority.get("source_integrity_admission_observation_id")
                )
            or row.diagnostics_json.get("status") != "source_bound"
            or row.diagnostics_json.get("semantic_entailment_claimed") is not False
            or row.diagnostics_json.get("transactional_replay") is not True
        ):
            raise ReflectionContractError("persisted_source_binding_identity_mismatch")
        item = items[row.chunk_id]
        span = _clone(row.source_span_json)
        span.update(contract_version="raw_chunk_source_span_v2", verification_id=None, source_binding_id=row.id)
        section_path = span.get("section_path") or []
        citation = Citation.model_validate({
            "contract_version": "citation_public_v2", "citation_index": len(citations) + 1,
            "chunk_id": row.chunk_id, "document_id": item["document_id"],
            "document_version_id": span["document_version_id"], "document_title": item.get("document_title") or "",
            "source_path": span["source_path"], "logical_source_path": span["logical_source_path"],
            "section": span.get("section_path"), "section_path": section_path if isinstance(section_path, list) else [section_path],
            "page_number": span["page_range"][0], "page_range": span["page_range"], "char_span": span["char_span"],
            "bbox": span.get("bbox"), "snippet": str(item.get("content") or "")[:800],
            "source_span": span, "context_package_id": package.id, "retrieval_trace_id": package.retrieval_trace_id,
            "answer_session_id": answer_session.id, "answer_hash": row.answer_hash,
            "claim_id": None, "claim_index": None, "claim_text": None,
            "citation_verification_id": None, "verification": None,
            "unit_id": row.unit_id, "unit_index": row.unit_index, "unit_text": row.unit_text,
            "source_binding_id": row.id,
            "source_binding": {
                "source_binding_id": row.id, "status": "source_bound", "unit_id": row.unit_id, "unit_index": row.unit_index,
                "answer_hash": row.answer_hash, "binding_hash": row.binding_hash,
                **{key: value for key, value in authority.items() if key != "protocol_version"},
                "protocol_version": authority["protocol_version"],
                    "contract_version": (
                        "answer_source_binding_public_v3"
                        if source_integrity_admission is not None
                        else "answer_source_binding_public_v2"
                        if retrieval_gate is not None
                        else "answer_source_binding_public_v1"
                    ),
                "provenance_status": "valid", "structure_context_status": "valid", "transactional_replay": True,
                "semantic_entailment_claimed": False,
            },
        })
        citations.append(citation.model_dump(mode="json"))
    return citations


def answer_source_path_metrics(db: Session, *, package: ContextPackage, draft: AnswerDraft, evidence: AnswerEvidenceManifest,
    source_audit: dict[str, Any], _replay_protocol: str | None = None):
    from app.models import RetrievalTrace
    from app.services.context_graph import aggregate_node_path_contributions
    from app.services.reflection_sources import MAX_SOURCE_ANCESTORS, retention_origin

    protocol = PATH_SUPPORT_PROTOCOL if _replay_protocol is None else _replay_protocol
    if not isinstance(protocol, str) or protocol not in {"answer_path_support_score_v1", "answer_path_support_score_v2"}:
        raise ReflectionContractError("reflection_path_protocol_unknown")
    historical_v1 = protocol == "answer_path_support_score_v1"
    selected = {handle for unit in draft.answer_units for handle in unit.source_handles}
    if not source_audit.get("all_valid"):
        return source_path_metrics(sorted(selected), {}, protocol_version=protocol)
    path_cache = {}
    trace_cache = {}

    def trace_paths(owner):
        if owner.retrieval_trace_id not in trace_cache:
            trace = db.get(RetrievalTrace, owner.retrieval_trace_id)
            if trace is None or trace.knowledge_base_id != owner.knowledge_base_id:
                raise ReflectionContractError("reflection_path_trace_scope_invalid")
            labels = []
            for label in trace.path_labels_json or []:
                if label.get("layer") != "chunk":
                    continue
                values = (label.get("distance_so_far"), label.get("reward_so_far"))
                if any(value is None for value in values):
                    continue
                if any(isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < 0 for value in values):
                    raise ReflectionContractError("path_distance_or_reward_invalid")
                labels.append(label)
            summaries = aggregate_node_path_contributions(labels)
            trace_cache[trace.id] = (trace, {item["node_id"]: item["reached_by_paths"]
                for item in summaries if item["layer"] == "chunk"})
        return trace_cache[owner.retrieval_trace_id]

    def path_records(owner, cid, ancestors=()):
        key = (owner.id, cid)
        if key in path_cache:
            return path_cache[key]
        if owner.id in ancestors or len(ancestors) >= MAX_SOURCE_ANCESTORS:
            raise ReflectionContractError("reflection_retention_path_ancestry_invalid")
        origin = retention_origin(db, package=owner, chunk_id=cid)
        if origin is not None:
            records = path_records(origin[1], cid, (*ancestors, owner.id))
        else:
            if historical_v1:
                # Exact historical reader; active requests always use v2.
                trace = None
                by_chunk = {item["chunk_id"]: [dict(path) for path in
                    (item.get("why_selected") or {}).get("reached_by_paths", [])
                    if path.get("layer") == "chunk" and path.get("node_id") == item["chunk_id"]]
                    for item in owner.package_json["chunks"]}
            else:
                trace, by_chunk = trace_paths(owner)
            records = list(by_chunk.get(cid, []))
            from app.models import ContextPackageSourceExpansion
            from app.services.reflection_expansion import expansion_identity
            expansion = db.scalar(select(ContextPackageSourceExpansion).where(
                ContextPackageSourceExpansion.target_context_package_id == owner.id,
                ContextPackageSourceExpansion.chunk_id == cid))
            if (historical_v1 or not records) and expansion is not None:
                source_package = db.get(ContextPackage, expansion.source_context_package_id)
                if source_package is None or expansion.witness_hash != reflection_hash(expansion_identity(expansion)):
                    raise ReflectionContractError("reflection_expansion_path_origin_invalid")
                records = path_records(source_package, expansion.anchor_chunk_id, (*ancestors, owner.id))
                path_cache[key] = records
                return records
            if not records and cid in set(owner.restored_chunk_ids_json or []):
                source = db.get(Chunk, cid)
                anchor_ids = owner.hit_chunk_ids_json if historical_v1 else trace.result_chunk_ids_json
                for anchor_id in anchor_ids or []:
                    anchor = db.get(Chunk, anchor_id)
                    if source is not None and anchor is not None and anchor.document_version_id == source.document_version_id and (
                        cid in {anchor.previous_chunk_id, anchor.next_chunk_id} or anchor_id in {source.previous_chunk_id, source.next_chunk_id}
                    ):
                        records.extend(by_chunk.get(anchor_id, []))
        path_cache[key] = records
        return records

    by_handle = evidence.by_handle()
    scored_paths: dict[str, list[dict[str, Any]]] = {}
    for handle in selected:
        cid = by_handle[handle]["chunk_id"]
        records = path_records(package, cid)
        scored_paths[handle] = [{
            "distance_so_far": path.get("distance_so_far"),
            "reward_so_far": path.get("reward_so_far"),
            "canonical_replay_passed": bool(source_audit.get("all_valid")),
        } for path in records]
    return source_path_metrics(sorted(selected), scored_paths, protocol_version=protocol)
