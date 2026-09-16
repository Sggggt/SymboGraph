"""Deterministic admission of the actual Context Package used for one answer."""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any

from sqlalchemy import select

from app.intent_contracts import AcceptedPlan
from app.models import AgentObservation, Chunk, ContextPackage, RetrievalTrace, generate_uuid
from app.retrieval_control_contracts import SourceScopeObligation, SourceScopeRequest, control_hash
from app.services.answer_sources import build_answer_evidence_manifest
from app.services.citation_provenance import audit_citation_provenance
from app.services.context_graph import context_package_to_contexts, passes_filters
from app.services.reflection_sources import source_citation


PROTOCOL = "source_integrity_admission_v1"


@dataclass(frozen=True)
class SourceIntegrityAdmission:
    outcome: str
    audit: dict[str, Any]
    observation_id: str

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"


def _normalized(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _scope_items(package: ContextPackage) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in (package.package_json or {}).get("chunks", []))


def _selector_matches(selector, item: dict[str, Any]) -> bool:
    reference = _normalized(selector.reference)
    title = _normalized(item.get("document_title") or item.get("title"))
    section = _normalized(
        " / ".join(item.get("section_path") or [])
        if isinstance(item.get("section_path"), list)
        else item.get("section_path") or item.get("section")
    )
    content = str(item.get("content") or item.get("text") or "")
    if selector.kind == "document":
        return bool(reference and reference in title)
    if selector.kind == "section":
        if selector.match == "role":
            roles = set(item.get("source_roles") or item.get("evidence_roles") or [])
            return selector.role in roles
        return bool(reference and (reference in section or reference in title))
    if selector.kind == "text":
        return bool(reference and reference in _normalized(content))
    flags = set(item.get("content_flags") or [])
    if selector.kind in flags:
        return selector.match == "kind" or bool(reference and (reference in section or reference in _normalized(content)))
    return False


def _scope_request_matches(request: SourceScopeRequest, items: tuple[dict[str, Any], ...]) -> bool:
    if request.op == "scope":
        return any(_selector_matches(request.selector, item) for item in items)
    values = [_scope_request_matches(child, items) for child in request.children]
    return all(values) if request.op == "intersection" else any(values)


def _obligation_status(
    obligation: SourceScopeObligation,
    items: tuple[dict[str, Any], ...],
) -> tuple[bool, str]:
    if obligation.op == "coverage":
        if obligation.mode == "complete":
            return False, "representation_incomplete"
        return (
            (True, "covered")
            if _scope_request_matches(obligation.scope, items)
            else (False, "scope_ambiguous")
        )
    statuses = [_obligation_status(child, items) for child in obligation.children]
    if obligation.op == "all":
        if all(passed for passed, _reason in statuses):
            return True, "covered"
        reasons = [reason for passed, reason in statuses if not passed]
        return False, "representation_incomplete" if "representation_incomplete" in reasons else "scope_ambiguous"
    if any(passed for passed, _reason in statuses):
        return True, "covered"
    reasons = [reason for _passed, reason in statuses]
    return False, "representation_incomplete" if reasons and all(reason == "representation_incomplete" for reason in reasons) else "scope_ambiguous"


def _exact_scope_checks(
    db,
    *,
    plan: AcceptedPlan,
    package: ContextPackage,
    filters,
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    required = [
        item for item in plan.task.requirements if item.source_scope is not None
    ]
    if not required:
        return [], None, {"required": False}
    trace = db.get(RetrievalTrace, package.retrieval_trace_id)
    scope_execution = (
        dict((trace.diagnostics_json or {}).get("source_scope_execution") or {})
        if trace is not None
        else {}
    )
    package_scope_execution = dict(
        (package.diagnostics_json or {}).get("source_scope_execution") or {}
    )
    if (
        trace is None
        or trace.knowledge_base_id != plan.task.knowledge_base_id
        or scope_execution.get("protocol_version")
        != "intent_source_scope_execution_v1"
        or scope_execution.get("task_hash") != plan.task.identity
        or not scope_execution.get("audit_hash")
        or scope_execution != package_scope_execution
        or control_hash(
            {
                key: value
                for key, value in scope_execution.items()
                if key != "audit_hash"
            }
        )
        != scope_execution.get("audit_hash")
    ):
        return [], "technical_failure", {
            "required": True,
            "error_type": "source_scope_execution_audit_invalid",
        }
    source_ids = tuple(
        str(item) for item in scope_execution.get("source_chunk_ids") or []
    )
    if (
        not source_ids
        or len(source_ids) > 32768
        or len(set(source_ids)) != len(source_ids)
    ):
        return [], "technical_failure", {
            "required": True,
            "error_type": "source_scope_inventory_invalid",
        }
    chunks = {
        str(item.id): item
        for item in db.scalars(select(Chunk).where(Chunk.id.in_(source_ids)))
    }
    if (
        set(chunks) != set(source_ids)
        or any(
            item.knowledge_base_id != plan.task.knowledge_base_id
            or item.state != "active"
            or not passes_filters(db, item, filters)
            for item in chunks.values()
        )
    ):
        return [], "technical_failure", {
            "required": True,
            "error_type": "source_scope_inventory_changed",
        }
    try:
        from app.services.evidence_scope import (
            evaluate_scope_obligation,
            load_scope_replay_index,
            package_scope_intervals,
        )

        index = load_scope_replay_index(
            db,
            task=plan.task,
            ids=source_ids,
            source_scope_hash=str(scope_execution.get("source_scope_hash") or ""),
            task_scoped=True,
        )
        if index.identity != scope_execution.get("source_index_identity"):
            raise ValueError("source_scope_index_identity_changed")
        bindings = index.bind(plan.task)
        replayed_bindings = [item.model_dump(mode="json") for item in bindings]
        if replayed_bindings != scope_execution.get("scope_bindings"):
            raise ValueError("source_scope_bindings_changed")
        packed = package_scope_intervals(plan.task, package)
        by_facet = {item.facet_id: item for item in bindings}
        statuses = [
            evaluate_scope_obligation(
                task=plan.task,
                facet=requirement,
                bound=by_facet[requirement.id],
                packed=packed,
            )
            for requirement in required
        ]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return [], "technical_failure", {
            "required": True,
            "error_type": exc.__class__.__name__,
        }
    checks = [
        {
            "requirement_id": item.facet_id,
            "passed": item.state == "satisfied",
            "state": item.state,
            "reason_codes": list(item.reason_codes),
            "input_hash": item.input_hash,
            "coverage": [part.model_dump(mode="json") for part in item.coverage],
        }
        for item in statuses
    ]
    target_plan = dict(scope_execution.get("target_plan") or {})
    target_status = str(target_plan.get("status") or "")
    target_ids = set(str(item) for item in target_plan.get("target_chunk_ids") or [])
    materialized = {
        str(item.get("chunk_id") or "") for item in _scope_items(package)
    }
    if target_status == "over_budget":
        outcome = "context_budget_exhausted"
    elif any(item.state == "unknown" for item in statuses):
        reasons = {
            reason
            for item in statuses
            for reason in item.reason_codes
        }
        outcome = (
            "representation_incomplete"
            if reasons & {"representation_incomplete", "unsupported_representation"}
            else "scope_ambiguous"
        )
    elif any(item.state == "unsatisfied" for item in statuses):
        outcome = "context_budget_exhausted"
    elif target_status == "proposed" and not target_ids <= materialized:
        outcome = "context_budget_exhausted"
    else:
        outcome = None
    return checks, outcome, {
        "required": True,
        "source_index_identity": index.identity,
        "source_inventory_count": len(source_ids),
        "target_plan_status": target_status,
        "target_chunk_count": len(target_ids),
        "target_materialized_count": len(target_ids & materialized),
        "model_call_count": 0,
    }


def admit_context_package(
    db,
    *,
    run,
    plan: AcceptedPlan,
    package: ContextPackage,
    filters,
    remaining_seconds: float,
) -> SourceIntegrityAdmission:
    """Validate package identity, raw spans, filters, source duties and budget."""

    items = _scope_items(package)
    checks: dict[str, Any] = {
        "nonempty_package": bool(items),
        "knowledge_base_matches": package.knowledge_base_id == plan.task.knowledge_base_id,
        "question_matches": (
            package.query == plan.task.question
            or plan.strategy.route == "verified_context_reuse"
        ),
        "retrieval_trace_bound": bool(package.retrieval_trace_id),
        "token_budget_valid": 0 <= int(package.token_count) <= int(package.token_budget),
        "remaining_execution_budget": remaining_seconds > 0,
    }
    source_audit = audit_citation_provenance(
        db,
        knowledge_base_id=package.knowledge_base_id,
        package=package,
        contexts=context_package_to_contexts(package),
        citations=[source_citation(item, package) for item in items],
    )
    checks["raw_source_provenance_valid"] = bool(source_audit.get("all_valid"))
    chunk_ids = [str(item.get("chunk_id") or "") for item in items]
    chunks = {
        str(item.id): item
        for item in db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids)))
    } if chunk_ids else {}
    checks["chunk_inventory_complete"] = (
        bool(chunk_ids)
        and len(chunk_ids) == len(set(chunk_ids))
        and set(chunk_ids) == set(chunks)
    )
    checks["filters_preserved"] = checks["chunk_inventory_complete"] and all(
        passes_filters(db, chunks[chunk_id], filters) for chunk_id in chunk_ids
    )
    scope_checks, scope_outcome, scope_replay = _exact_scope_checks(
        db,
        plan=plan,
        package=package,
        filters=filters,
    )
    checks["source_scope_obligations"] = scope_checks
    if not checks["remaining_execution_budget"]:
        outcome = "context_budget_exhausted"
    elif scope_outcome is not None:
        outcome = scope_outcome
    elif not all(value for key, value in checks.items() if key != "source_scope_obligations"):
        outcome = "technical_failure" if items else "insufficient_evidence"
    else:
        outcome = "passed"
    evidence = build_answer_evidence_manifest(package, context_package_to_contexts(package)) if outcome == "passed" else None
    observation_id = generate_uuid()
    payload = {
        "protocol_version": PROTOCOL,
        "outcome": outcome,
        "run_id": run.id,
        "task_hash": plan.task.identity,
        "strategy_hash": plan.strategy.identity,
        "accepted_plan_hash": plan.identity,
        "retrieval_trace_id": package.retrieval_trace_id,
        "context_package_id": package.id,
        "observation_id": observation_id,
        "package_manifest_hash": control_hash(package.package_json or {}),
        "evidence_manifest_hash": evidence.manifest_hash if evidence is not None else None,
        "provenance_session_hash": source_audit.get("provenance_session_hash"),
        "provenance_audit": source_audit,
        "checks": checks,
        "source_scope_replay": scope_replay,
        "score_fields_used_for_admission": [],
        "model_call_count": 0,
    }
    payload["audit_hash"] = control_hash(payload)
    row = AgentObservation(
        id=observation_id,
        run_id=run.id,
        observation_type="source_integrity_admission",
        verdict=outcome,
        observation_json=payload,
        evidence_chunk_ids_json=chunk_ids,
    )
    db.add(row)
    db.flush()
    return SourceIntegrityAdmission(outcome=outcome, audit=payload, observation_id=observation_id)
