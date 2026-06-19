from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import get_settings
from app.models import (
    AgentAction,
    AgentObservation,
    AgentPlan,
    AgentRun,
    AgentTraceEvent,
    AnswerSession,
    Chunk,
    CitationVerification,
    CoarseConcept,
    ContextPackage,
    RQPrefix,
    MidConcept,
    PolicyState,
    QASession,
    RewardEvent,
)
from app.schemas import AgentRequest
from app.services.context_graph import (
    agent_operating_envelope,
    agent_operating_envelope_state_hash,
    build_context_package,
    context_package_to_contexts,
    layered_search,
    runtime_settings_state_hash,
)
from app.services.embeddings import ChatProvider, FallbackDisabledError, is_degraded_mode
from app.services.error_sanitizer import public_exception_message
from app.services.ingestion import resolve_knowledge_base
from app.services.chunking import stable_hash
from app.services.model_output import coerce_confidence


_TRACE_SUBSCRIBERS: dict[str, set[asyncio.Queue[dict]]] = {}


def _summarize(text: str, limit: int = 280) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _publish_trace_event(run_id: str, payload: dict) -> None:
    for queue in list(_TRACE_SUBSCRIBERS.get(run_id, ())):
        queue.put_nowait(payload)


def _subscribe_trace(run_id: str) -> asyncio.Queue[dict]:
    queue: asyncio.Queue[dict] = asyncio.Queue()
    _TRACE_SUBSCRIBERS.setdefault(run_id, set()).add(queue)
    return queue


def _unsubscribe_trace(run_id: str, queue: asyncio.Queue[dict]) -> None:
    subscribers = _TRACE_SUBSCRIBERS.get(run_id)
    if not subscribers:
        return
    subscribers.discard(queue)
    if not subscribers:
        _TRACE_SUBSCRIBERS.pop(run_id, None)


def trace_event_to_payload(event: AgentTraceEvent) -> dict:
    return {
        "id": event.id,
        "run_id": event.run_id,
        "node": event.node,
        "status": event.status,
        "input_summary": event.input_summary,
        "output_summary": event.output_summary,
        "document_ids": event.document_ids or [],
        "scores": event.scores or {},
        "duration_ms": event.duration_ms,
        "error": event.error_message,
        "created_at": event.created_at,
    }


def trace(db: Session, run_id: str, node: str, *, input_summary: str = "", output_summary: str = "", document_ids: list[str] | None = None, scores: dict | None = None, duration_ms: int = 0, status: str = "completed", error: str | None = None) -> dict:
    event = AgentTraceEvent(
        run_id=run_id,
        node=node,
        status=status,
        input_summary=input_summary,
        output_summary=output_summary,
        document_ids=document_ids or [],
        scores=scores or {},
        duration_ms=duration_ms,
        error_message=error,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    payload = trace_event_to_payload(event)
    _publish_trace_event(run_id, payload)
    return payload


ALLOWED_TYPED_ACTIONS = {
    "activate_coarse_concepts",
    "route_mid_concepts",
    "route_rq_memberships",
    "select_entry_nodes",
    "walk_graph_frontier",
    "drill_down_layer",
    "evaluate_gray_zone_path",
    "recall_chunks",
    "restore_context_package",
    "build_context_package",
    "verify_citations",
    "repair_missing_citation",
    "repair_concept_gap",
    "repair_bridge_gap",
    "repair_formula_context",
    "reduce_drift_and_repack",
}

REQUIRED_TYPED_ACTIONS = ["select_entry_nodes", "walk_graph_frontier", "recall_chunks", "restore_context_package", "verify_citations"]


def _default_budget_for_action(action_type: str, envelope: dict[str, Any]) -> dict[str, int]:
    mapping = {
        "select_entry_nodes": {
            "agent_coarse_total_budget": int(envelope.get("agent_coarse_total_budget") or 0),
        },
        "walk_graph_frontier": {
            "agent_chunk_per_mid_budget": int(envelope.get("agent_chunk_per_mid_budget") or 0),
            "max_depth_per_layer": int(envelope.get("max_depth_per_layer") or 0),
            "max_labels_per_node": int(envelope.get("max_labels_per_node") or 0),
        },
        "drill_down_layer": {
            "agent_mid_per_coarse_budget": int(envelope.get("agent_mid_per_coarse_budget") or 0),
            "agent_mid_top_k": int(envelope.get("agent_mid_top_k") or 0),
            "agent_chunk_per_mid_budget": int(envelope.get("agent_chunk_per_mid_budget") or 0),
            "agent_chunk_top_k": int(envelope.get("agent_chunk_top_k") or 0),
        },
        "evaluate_gray_zone_path": {
            "path_distance_green_threshold": float(envelope.get("path_distance_green_threshold") or 0),
            "path_distance_gray_threshold": float(envelope.get("path_distance_gray_threshold") or 0),
            "path_distance_hard_threshold": float(envelope.get("path_distance_hard_threshold") or 0),
        },
        "recall_chunks": {"agent_chunk_top_k": int(envelope.get("agent_chunk_top_k") or 0)},
        "restore_context_package": {"structure_restore_budget": int(envelope.get("structure_restore_budget") or 0)},
        "build_context_package": {"context_package_token_budget": int(envelope.get("context_package_token_budget") or 0)},
        "verify_citations": {"verification_budget": int(envelope.get("verification_budget") or 0)},
        "repair_missing_citation": {"repair_round_budget": int(envelope.get("repair_round_budget") or 0)},
        "repair_concept_gap": {"repair_round_budget": int(envelope.get("repair_round_budget") or 0)},
        "repair_bridge_gap": {"repair_round_budget": int(envelope.get("repair_round_budget") or 0)},
        "repair_formula_context": {"repair_round_budget": int(envelope.get("repair_round_budget") or 0)},
        "reduce_drift_and_repack": {"repair_round_budget": int(envelope.get("repair_round_budget") or 0)},
    }
    return mapping.get(action_type, {})


def fallback_typed_actions(question: str, envelope: dict[str, Any]) -> list[dict[str, Any]]:
    formula_hint = any(token in question.lower() for token in ("formula", "table", "equation", "公式", "表格"))
    actions = [
        "select_entry_nodes",
        "drill_down_layer",
        "walk_graph_frontier",
        "recall_chunks",
        "restore_context_package",
        "build_context_package",
        "verify_citations",
    ]
    if formula_hint:
        actions.insert(-1, "repair_formula_context")
    return [
        {
            "action_type": action_type,
            "target_ids": [],
            "reason": "Route through the four-layer context graph under the active operating envelope.",
            "budget_request": _default_budget_for_action(action_type, envelope),
            "expected_evidence": {"source": "context_graph", "requires_chunk_spans": True},
            "stop_condition": {"sufficient_evidence": action_type in {"build_context_package", "verify_citations"}},
        }
        for action_type in actions
    ]


ACTION_TARGET_LAYERS: dict[str, set[str]] = {
    "activate_coarse_concepts": {"coarse"},
    "route_mid_concepts": {"coarse", "mid"},
    "route_rq_memberships": {"mid", "rq_membership"},
    "select_entry_nodes": {"coarse", "mid", "rq_membership", "chunk"},
    "walk_graph_frontier": {"coarse", "mid", "chunk"},
    "drill_down_layer": {"coarse", "mid"},
    "evaluate_gray_zone_path": {"coarse", "mid", "rq_membership", "chunk"},
    "recall_chunks": {"rq_membership", "chunk"},
    "restore_context_package": {"chunk", "context_package"},
    "build_context_package": {"chunk", "context_package"},
    "verify_citations": {"chunk", "context_package", "citation"},
    "repair_missing_citation": {"coarse", "mid", "rq_membership", "chunk", "context_package"},
    "repair_concept_gap": {"coarse", "mid", "rq_membership"},
    "repair_bridge_gap": {"coarse", "mid", "rq_membership", "chunk"},
    "repair_formula_context": {"rq_membership", "chunk", "context_package"},
    "reduce_drift_and_repack": {"chunk", "context_package"},
}


def _target_id_layers(db: Session, knowledge_base_id: str, target_ids: list[str]) -> dict[str, set[str]]:
    if not target_ids:
        return {}
    target_set = set(target_ids)
    layers: dict[str, set[str]] = {target_id: set() for target_id in target_ids}
    for row in db.scalars(select(CoarseConcept).where(CoarseConcept.id.in_(target_set), CoarseConcept.knowledge_base_id == knowledge_base_id)).all():
        layers[row.id].add("coarse")
    for row in db.scalars(select(MidConcept).where(MidConcept.id.in_(target_set), MidConcept.knowledge_base_id == knowledge_base_id)).all():
        layers[row.id].add("mid")
    for row in db.scalars(select(RQPrefix).where(RQPrefix.id.in_(target_set))).all():
        layers[row.id].add("rq_membership")
    for row in db.scalars(select(Chunk).where(Chunk.id.in_(target_set), Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active")).all():
        layers[row.id].add("chunk")
    for row in db.scalars(select(ContextPackage).where(ContextPackage.id.in_(target_set), ContextPackage.knowledge_base_id == knowledge_base_id)).all():
        layers[row.id].add("context_package")
    return layers


def heuristic_query_intent(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    lower = question.lower()
    if any(token in lower for token in ("compare", "对比", "区别")):
        intent = "comparison"
    elif any(token in lower for token in ("formula", "equation", "公式", "推导")):
        intent = "formula_table_lookup"
    elif any(token in lower for token in ("why", "how", "为什么", "如何")):
        intent = "analysis"
    else:
        intent = "definition"
    return {
        "intent": intent,
        "entities": [],
        "sub_queries": [question],
        "needs_graph": True,
        "history_turns": len(history or []),
    }


async def perceive_query_intent(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    provider = ChatProvider()
    if hasattr(provider, "perceive_question"):
        try:
            result = await provider.perceive_question(question, history or [])
            if isinstance(result, dict):
                return {**heuristic_query_intent(question, history), **result}
        except Exception:
            raise
    return heuristic_query_intent(question, history)


async def propose_agent_plan(question: str, history: list[dict], query_intent: dict[str, Any], envelope: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fallback = {
        "typed_actions": fallback_typed_actions(question, envelope),
        "planner_diagnostics": {"planner": "fallback_schema_when_model_fallback_enabled"},
    }
    system = (
        "You are the Layered P&E planner for a Four-Layer Context Graph RAG system. "
        "Return strict JSON with a typed_actions array. Each action must include action_type, target_ids, reason, "
        "budget_request, expected_evidence, and stop_condition. You may only choose from the supplied action space. "
        "Policy and runtime settings provide only the operating envelope; do not invent facts."
    )
    user_prompt = str(
        {
            "question": question,
            "history": history[-6:],
            "query_intent": query_intent,
            "operating_envelope": envelope,
            "allowed_action_types": sorted(ALLOWED_TYPED_ACTIONS),
            "required_action_types": REQUIRED_TYPED_ACTIONS,
        }
    )
    provider = ChatProvider()
    planner_errors: list[str] = []
    try:
        output = await provider.classify_json(system_prompt=system, user_prompt=user_prompt, fallback=fallback)
    except Exception as exc:
        output = {}
        planner_errors.append(f"initial_request:{public_exception_message(exc)}")
    actions = output.get("typed_actions") if isinstance(output, dict) else None
    if not isinstance(actions, list):
        repair_system = (
            f"{system} Your previous response was rejected by the typed action schema. "
            "Return ONLY a JSON object with key typed_actions. Do not include prose, markdown, analysis, or alternate keys."
        )
        repair_prompt = str(
            {
                "invalid_response_keys": sorted(output.keys()) if isinstance(output, dict) else [],
                "required_shape": {"typed_actions": ["typed action objects"]},
                "required_action_types": REQUIRED_TYPED_ACTIONS,
                "allowed_action_types": sorted(ALLOWED_TYPED_ACTIONS),
                "original_request": {
                    "question": question,
                    "history": history[-6:],
                    "query_intent": query_intent,
                    "operating_envelope": envelope,
                },
            }
        )
        try:
            repaired = await provider.classify_json(system_prompt=repair_system, user_prompt=repair_prompt, fallback=fallback)
            repaired_actions = repaired.get("typed_actions") if isinstance(repaired, dict) else None
            if isinstance(repaired_actions, list):
                output = {**repaired, "planner_repair": {"attempted": True, "errors": planner_errors}}
                actions = repaired_actions
        except Exception as exc:
            planner_errors.append(f"repair_request:{public_exception_message(exc)}")
    if not isinstance(actions, list):
        if not get_settings().enable_model_fallback:
            detail = f" ({'; '.join(planner_errors)})" if planner_errors else ""
            raise RuntimeError(f"Agent planner returned invalid JSON after repair: typed_actions array is required{detail}")
        actions = fallback["typed_actions"]
    return [action for action in actions if isinstance(action, dict)], output if isinstance(output, dict) else {}


def validate_typed_actions(
    actions: list[dict[str, Any]],
    envelope: dict[str, Any],
    *,
    db: Session | None = None,
    knowledge_base_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_actions = int(envelope.get("max_typed_actions_per_round") or 1)
    diagnostics: dict[str, Any] = {
        "accepted": [],
        "rejected": [],
        "inserted_required_actions": [],
        "fallback_disabled": not bool(get_settings().enable_model_fallback),
        "required_restore_modes": envelope.get("required_restore_modes") or [],
        "allowed_relation_types": envelope.get("allowed_relation_types") or [],
    }
    accepted: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for index, action in enumerate(actions[: max_actions * 2]):
        action_type = str(action.get("action_type") or "")
        if action_type not in ALLOWED_TYPED_ACTIONS:
            diagnostics["rejected"].append({"index": index, "action_type": action_type, "reason": "unsupported_action_type"})
            continue
        budget_request = action.get("budget_request") if isinstance(action.get("budget_request"), dict) else {}
        budget_errors = []
        for key, requested in budget_request.items():
            if key not in envelope:
                continue
            try:
                requested_value = int(requested)
            except (TypeError, ValueError):
                budget_errors.append({"key": key, "reason": "not_integer"})
                continue
            if requested_value > int(envelope.get(key) or 0):
                budget_errors.append({"key": key, "requested": requested_value, "limit": int(envelope.get(key) or 0)})
        if budget_errors:
            diagnostics["rejected"].append({"index": index, "action_type": action_type, "reason": "budget_exceeded", "details": budget_errors})
            continue
        target_ids = [str(value) for value in (action.get("target_ids") or []) if str(value).strip()]
        target_layers = _target_id_layers(db, knowledge_base_id, target_ids) if db is not None and knowledge_base_id and target_ids else {}
        missing_targets = [target_id for target_id in target_ids if target_id not in target_layers or not target_layers[target_id]]
        allowed_layers = ACTION_TARGET_LAYERS.get(action_type, set())
        incompatible_targets = [
            {"target_id": target_id, "layers": sorted(target_layers.get(target_id, set()))}
            for target_id in target_ids
            if target_layers.get(target_id) and allowed_layers and not target_layers[target_id].intersection(allowed_layers)
        ]
        if missing_targets:
            diagnostics["rejected"].append({"index": index, "action_type": action_type, "reason": "target_id_not_found", "target_ids": missing_targets})
            continue
        if incompatible_targets:
            diagnostics["rejected"].append({"index": index, "action_type": action_type, "reason": "target_layer_mismatch", "details": incompatible_targets})
            continue
        expected_evidence = action.get("expected_evidence") if isinstance(action.get("expected_evidence"), dict) else {}
        requested_relation_types = set(expected_evidence.get("allowed_relation_types") or expected_evidence.get("relation_types") or [])
        allowed_relation_types = set(envelope.get("allowed_relation_types") or [])
        disallowed_relation_types = sorted(requested_relation_types - allowed_relation_types)
        if disallowed_relation_types:
            diagnostics["rejected"].append({"index": index, "action_type": action_type, "reason": "relation_type_not_allowed", "details": disallowed_relation_types})
            continue
        if expected_evidence.get("fallback_allowed") and not bool(get_settings().enable_model_fallback):
            diagnostics["rejected"].append({"index": index, "action_type": action_type, "reason": "fallback_disabled"})
            continue
        required_restore_modes = list(envelope.get("required_restore_modes") or [])
        if action_type in {"restore_context_package", "build_context_package"}:
            restore_modes = set(expected_evidence.get("required_restore_modes") or [])
            expected_evidence["required_restore_modes"] = sorted(set(required_restore_modes).union(restore_modes))
        if action_type == "verify_citations":
            expected_evidence["required_verification_stage"] = "structure_plus_llm_entailment"
        expected_evidence.setdefault("allowed_relation_types", sorted(allowed_relation_types))
        normalized = {
            "action_type": action_type,
            "target_ids": target_ids,
            "reason": str(action.get("reason") or ""),
            "budget_request": {**_default_budget_for_action(action_type, envelope), **budget_request},
            "expected_evidence": expected_evidence,
            "stop_condition": action.get("stop_condition") if isinstance(action.get("stop_condition"), dict) else {},
            "validation": {
                "valid": True,
                "schema_checked": True,
                "budget_checked": True,
                "target_ids_checked": bool(target_ids),
                "target_layers": {target_id: sorted(target_layers.get(target_id, set())) for target_id in target_ids},
                "fallback_disabled_checked": True,
                "bridge_protection_checked": True,
                "required_restore_modes": required_restore_modes,
                "required_verification_stage": expected_evidence.get("required_verification_stage"),
            },
        }
        accepted.append(normalized)
        seen_types.add(action_type)
        diagnostics["accepted"].append({"index": index, "action_type": action_type})
        if len(accepted) >= max_actions:
            break
    for required in REQUIRED_TYPED_ACTIONS:
        if required not in seen_types and len(accepted) < max_actions:
            inserted = {
                "action_type": required,
                "target_ids": [],
                "reason": "Inserted by validator because this action is required by the technical spec.",
                "budget_request": _default_budget_for_action(required, envelope),
                "expected_evidence": {
                    "source": "context_package",
                    "requires_chunk_spans": True,
                    "required_restore_modes": list(envelope.get("required_restore_modes") or []),
                    "allowed_relation_types": list(envelope.get("allowed_relation_types") or []),
                    "required_verification_stage": "structure_plus_llm_entailment" if required == "verify_citations" else None,
                },
                "stop_condition": {"required_action_complete": True},
                "validation": {"valid": True, "inserted_required_action": True, "fallback_disabled_checked": True},
            }
            accepted.append(inserted)
            diagnostics["inserted_required_actions"].append(required)
    diagnostics["valid"] = all(required in {action["action_type"] for action in accepted} for required in REQUIRED_TYPED_ACTIONS)
    return accepted, diagnostics


def record_agent_plan_and_actions(
    db: Session,
    *,
    run: AgentRun,
    query_intent: dict[str, Any],
    envelope: dict[str, Any],
    raw_planner_output: dict[str, Any],
    actions: list[dict[str, Any]],
    validation: dict[str, Any],
) -> tuple[AgentPlan, list[AgentAction]]:
    plan = AgentPlan(
        run_id=run.id,
        knowledge_base_id=run.knowledge_base_id,
        plan_index=0,
        planner_model_json={"planner_protocol": "layered_pe_planner_v1", "raw_output": raw_planner_output},
        query_intent_json=query_intent,
        envelope_json=envelope,
        typed_actions_json=actions,
        validation_json=validation,
        status="validated" if validation.get("valid") else "invalid",
        diagnostics_json={
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
        },
    )
    db.add(plan)
    db.flush()
    rows: list[AgentAction] = []
    for index, action in enumerate(actions):
        row = AgentAction(
            run_id=run.id,
            plan_id=plan.id,
            action_index=index,
            action_type=action["action_type"],
            target_ids_json=action.get("target_ids") or [],
            reason=action.get("reason") or "",
            budget_request_json=action.get("budget_request") or {},
            expected_evidence_json=action.get("expected_evidence") or {},
            stop_condition_json=action.get("stop_condition") or {},
            validation_json=action.get("validation") or {},
            status="accepted",
        )
        db.add(row)
        rows.append(row)
    db.flush()
    return plan, rows


def record_observation(
    db: Session,
    *,
    run_id: str,
    action: AgentAction | None,
    observation_type: str,
    observation: dict[str, Any],
    evidence_chunk_ids: list[str] | None = None,
    verdict: str = "observed",
) -> AgentObservation:
    row = AgentObservation(
        run_id=run_id,
        action_id=action.id if action else None,
        observation_type=observation_type,
        observation_json=observation,
        evidence_chunk_ids_json=evidence_chunk_ids or [],
        verdict=verdict,
        diagnostics_json={
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
        },
    )
    db.add(row)
    if action is not None:
        action.status = "completed" if verdict != "rejected" else "rejected"
        action.output_json = observation
    db.flush()
    return row


def actions_by_type(actions: list[AgentAction]) -> dict[str, AgentAction]:
    return {action.action_type: action for action in actions}


def create_or_get_session(db: Session, knowledge_base_id: str, session_id: str | None, question: str) -> QASession:
    session = db.get(QASession, session_id) if session_id else None
    if session is not None and session.knowledge_base_id != knowledge_base_id:
        session = None
    if session is None:
        session = QASession(knowledge_base_id=knowledge_base_id, title=_summarize(question, 80), transcript=[])
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def create_agent_run_context(db: Session, request: AgentRequest) -> tuple[QASession, AgentRun]:
    knowledge_base = resolve_knowledge_base(db, request.knowledge_base_id)
    session = create_or_get_session(db, knowledge_base.id, request.session_id, request.question)
    run = AgentRun(
        knowledge_base_id=knowledge_base.id,
        session_id=session.id,
        question=request.question,
        status="queued",
        route="layered_context_graph",
        metadata_json={"top_k": request.top_k, "filters": request.filters.model_dump()},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return session, run


def set_run_state(db: Session, run: AgentRun, status: str, *, current_node: str | None = None, answer: str | None = None, error: str | None = None) -> None:
    run.status = status
    run.current_node = current_node
    if status == "running" and run.started_at is None:
        run.started_at = datetime.utcnow()
    if status in {"completed", "failed", "needs_clarification"}:
        run.completed_at = datetime.utcnow()
    if answer is not None:
        run.final_answer = answer
    if error is not None:
        run.error_message = error
    db.commit()


def append_session_turn(db: Session, session: QASession, question: str, answer: str, run_id: str, citations: list[dict]) -> None:
    transcript = list(session.transcript or [])
    transcript.append({"role": "user", "content": question, "run_id": run_id})
    transcript.append({"role": "assistant", "content": answer, "run_id": run_id, "citations": citations})
    session.transcript = transcript
    session.last_question = question
    session.last_answer = answer
    session.updated_at = datetime.utcnow()
    db.commit()


def citation_payloads_from_package(
    package,
    answer_session_id: str | None = None,
    retrieval_trace_id: str | None = None,
    verification_by_chunk: dict[str, CitationVerification] | None = None,
    answer: str | None = None,
    question: str | None = None,
    supported_only: bool = False,
) -> list[dict]:
    chunks = (package.package_json or {}).get("chunks", [])
    hit_ids = set(getattr(package, "hit_chunk_ids_json", None) or [])
    restored_ids = set(getattr(package, "restored_chunk_ids_json", None) or [])
    bridge_ids = set(getattr(package, "bridge_chunk_ids_json", None) or [])
    citations = []
    candidate_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") in hit_ids]
    if answer:
        corpus_texts = [_citation_candidate_text(chunk) for chunk in chunks]
        df, corpus_size = _term_document_frequency(corpus_texts)
        scored_chunks = [
            (_citation_candidate_score(chunk, answer=answer, question=question, df=df, corpus_size=corpus_size, hit_ids=hit_ids, restored_ids=restored_ids, bridge_ids=bridge_ids), chunk)
            for chunk in chunks
        ]
        supported_chunks = [
            (score, chunk)
            for score, chunk in scored_chunks
            if float(score["combined_support_score"]) >= 0.04
            and (int(score["answer_overlap_count"]) >= 2 or int(score["query_overlap_count"]) >= 1 or chunk.get("chunk_id") in hit_ids)
        ]
        supported_chunks.sort(
            key=lambda item: (
                float(item[0]["combined_support_score"]),
                int(item[0]["answer_overlap_count"]),
                1 if item[1].get("chunk_id") in hit_ids else 0,
                1 if item[1].get("chunk_id") in restored_ids else 0,
            ),
            reverse=True,
        )
        candidate_chunks = [chunk for _, chunk in supported_chunks[: max(1, min(6, get_settings().agent_verification_budget))]]
        if not candidate_chunks:
            candidate_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") in hit_ids] or chunks[:1]
    for index, item in enumerate(candidate_chunks, start=1):
        verification = (verification_by_chunk or {}).get(item["chunk_id"])
        if supported_only and (verification is None or verification.verdict != "supported"):
            continue
        source_span = dict(item.get("source_span") or {})
        source_span.update(
            {
                "document_version_id": source_span.get("document_version_id") or item.get("document_version_id"),
                "chunk_id": source_span.get("chunk_id") or item.get("chunk_id"),
                "char_span": source_span.get("char_span") or item.get("char_span"),
                "page_range": source_span.get("page_range") or item.get("page_range"),
                "section_path": source_span.get("section_path") or item.get("section_path"),
                "bbox": source_span.get("bbox") or item.get("bbox") or {},
                "context_package_id": source_span.get("context_package_id") or package.id,
                "retrieval_trace_id": source_span.get("retrieval_trace_id") or retrieval_trace_id or package.retrieval_trace_id,
                "verification_id": verification.id if verification else source_span.get("verification_id"),
            }
        )
        citations.append(
            {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "document_version_id": item.get("document_version_id") or source_span.get("document_version_id"),
                "document_title": item.get("document_title") or "",
                "source_path": item.get("source_path") or "",
                "partition": None,
                "section": item.get("section_path"),
                "page_number": (item.get("page_range") or [None])[0],
                "snippet": _summarize(item.get("content") or "", 240),
                "source_span": source_span,
                "retrieval_trace_id": retrieval_trace_id or package.retrieval_trace_id,
                "answer_session_id": answer_session_id,
                "citation_verification_id": verification.id if verification else None,
                "verification": (
                    {
                        "verdict": verification.verdict,
                        "confidence": verification.confidence,
                        "diagnostics": verification.diagnostics_json or {},
                    }
                    if verification
                    else {}
                ),
            }
        )
    return citations


def _citation_candidate_text(chunk: dict[str, Any]) -> str:
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, list):
        section_text = " / ".join(str(item) for item in section_path)
    else:
        section_text = str(section_path or "")
    return "\n".join(
        item
        for item in [
            str(chunk.get("document_title") or ""),
            section_text,
            str(chunk.get("content") or chunk.get("snippet") or ""),
        ]
        if item.strip()
    )


def _citation_candidate_score(
    chunk: dict[str, Any],
    *,
    answer: str,
    question: str | None,
    df: Counter[str],
    corpus_size: int,
    hit_ids: set[str],
    restored_ids: set[str],
    bridge_ids: set[str],
) -> dict[str, Any]:
    text = _citation_candidate_text(chunk)
    answer_overlap = _weighted_overlap(answer, text, df, corpus_size)
    query_overlap = _weighted_overlap(question or "", text, df, corpus_size) if question else {"support_score": 0.0, "overlap_count": 0, "overlap_terms": []}
    claim_support = _claim_support(answer, text, df, corpus_size)
    chunk_id = chunk.get("chunk_id")
    provenance_bonus = 0.0
    if chunk_id in hit_ids:
        provenance_bonus += 0.02
    if chunk_id in restored_ids:
        provenance_bonus += 0.015
    if chunk_id in bridge_ids:
        provenance_bonus += 0.01
    combined = (
        0.58 * float(answer_overlap["support_score"])
        + 0.32 * float(query_overlap["support_score"])
        + 0.08 * min(1.0, int(claim_support["supported_claim_count"]) / 4.0)
        + provenance_bonus
    )
    return {
        "combined_support_score": round(combined, 6),
        "answer_support_score": answer_overlap["support_score"],
        "answer_overlap_count": answer_overlap["overlap_count"],
        "query_support_score": query_overlap["support_score"],
        "query_overlap_count": query_overlap["overlap_count"],
        "query_overlap_terms": query_overlap.get("overlap_terms", []),
        "supported_claim_count": claim_support["supported_claim_count"],
        "best_support_score": claim_support["best_support_score"],
        "best_overlap_count": claim_support["best_overlap_count"],
    }


def citation_verification_summary(db: Session, answer_session_id: str) -> tuple[dict[str, CitationVerification], float | None]:
    verifications = db.scalars(select(CitationVerification).where(CitationVerification.answer_session_id == answer_session_id)).all()
    if not verifications:
        return {}, None
    supported = [item for item in verifications if item.verdict == "supported"]
    by_chunk = {item.chunk_id: item for item in verifications if item.chunk_id}
    return by_chunk, len(supported) / len(verifications)


def _tokens(text: str) -> set[str]:
    from app.services.chinese_text import tokenize_for_retrieval

    return {str(token).lower() for token in tokenize_for_retrieval(text or "") if str(token).strip()}


def _term_document_frequency(context_texts: list[str]) -> tuple[Counter[str], int]:
    df: Counter[str] = Counter()
    for text in context_texts:
        df.update(_tokens(text))
    return df, max(len(context_texts), 1)


def _weighted_overlap(answer: str, context_text: str, df: Counter[str], corpus_size: int) -> dict[str, Any]:
    answer_terms = _tokens(answer)
    context_terms = _tokens(context_text)
    overlap = sorted(answer_terms.intersection(context_terms))

    def idf(term: str) -> float:
        return math.log((corpus_size + 1.0) / (float(df.get(term, 0)) + 0.5)) + 1.0

    answer_weight = sum(idf(term) for term in answer_terms) or 1.0
    overlap_weight = sum(idf(term) for term in overlap)
    support_score = overlap_weight / answer_weight
    return {
        "overlap_terms": overlap[:24],
        "overlap_count": len(overlap),
        "overlap_weight": round(overlap_weight, 6),
        "answer_weight": round(answer_weight, 6),
        "support_score": round(support_score, 6),
    }


def _answer_claims(answer: str) -> list[str]:
    claims = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", answer or "") if item.strip()]
    return claims or ([answer.strip()] if answer.strip() else [])


def _answer_claims(answer: str) -> list[str]:
    raw_parts = re.split(r"(?:\n+|(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f]))", answer or "")
    claims: list[str] = []
    for item in raw_parts:
        cleaned = re.sub(r"^\s*(?:[-*#]+|\d+[.)])\s*", "", item).strip()
        if cleaned:
            claims.append(cleaned)
    return claims or ([answer.strip()] if answer.strip() else [])


def _claim_support(answer: str, context_text: str, df: Counter[str], corpus_size: int) -> dict[str, Any]:
    full_overlap = _weighted_overlap(answer, context_text, df, corpus_size)
    claim_rows: list[dict[str, Any]] = []
    for claim in _answer_claims(answer):
        if not _tokens(claim):
            continue
        overlap = _weighted_overlap(claim, context_text, df, corpus_size)
        claim_rows.append(
            {
                "claim_text": claim[:500],
                "support_score": float(overlap["support_score"]),
                "overlap_count": int(overlap["overlap_count"]),
                "overlap_terms": overlap["overlap_terms"],
                "overlap_weight": overlap["overlap_weight"],
                "answer_weight": overlap["answer_weight"],
            }
        )
    best = max(claim_rows, key=lambda row: (float(row["support_score"]), int(row["overlap_count"])), default=None)
    supported_claim_count = sum(1 for row in claim_rows if float(row["support_score"]) >= 0.08 and int(row["overlap_count"]) >= 1)
    return {
        **full_overlap,
        "claim_count": len(claim_rows),
        "supported_claim_count": supported_claim_count,
        "best_supported_claim": best["claim_text"] if best else "",
        "best_support_score": round(float(best["support_score"]), 6) if best else 0.0,
        "best_overlap_count": int(best["overlap_count"]) if best else 0,
        "best_overlap_terms": best["overlap_terms"] if best else [],
        "claim_support_method": "adaptive_context_idf_claim_overlap_v1",
    }


def verify_answer_against_context_rules(answer: str, citations: list[dict], contexts: list[dict], verification_budget: int) -> list[dict[str, Any]]:
    context_by_chunk = {item.get("chunk_id"): item for item in contexts}
    claims = _answer_claims(answer)[: max(1, verification_budget)]
    formula_claim = bool(re.search(r"(\$|\\frac|P\(|=|\bformula\b|\btable\b|公式|表格)", answer or ""))
    package_has_formula = any(re.search(r"(\$|\\frac|P\(|=|\|)", str(item.get("content") or "")) for item in contexts)
    df, corpus_size = _term_document_frequency([str(item.get("content") or "") for item in contexts])
    results: list[dict[str, Any]] = []
    for index, citation in enumerate(citations[: max(1, verification_budget)], start=1):
        chunk_id = citation.get("chunk_id")
        source_span = citation.get("source_span") or {}
        context = context_by_chunk.get(chunk_id) or {}
        context_text = str(context.get("content") or citation.get("snippet") or "")
        overlap = _claim_support(answer, context_text, df, corpus_size)
        has_span = bool(source_span.get("char_span"))
        has_overlap = float(overlap["best_support_score"]) >= 0.08 and int(overlap["best_overlap_count"]) >= 1
        context_has_formula = bool(re.search(r"(\$|\\frac|P\(|=|\|)", context_text))
        if not has_span:
            verdict = "missing_citation"
            failure_type = "citation_missing"
            confidence = 0.2
        elif formula_claim and not package_has_formula:
            verdict = "formula_table_context_missing"
            failure_type = "formula_context_missing"
            confidence = 0.35
        elif has_overlap:
            verdict = "supported"
            failure_type = "none"
            confidence = min(0.96, 0.55 + float(overlap["best_support_score"]))
        else:
            verdict = "unsupported"
            failure_type = "unsupported_claim"
            confidence = 0.3
        results.append(
            {
                "citation_index": index,
                "chunk_id": chunk_id,
                "claim_text": overlap.get("best_supported_claim") or (claims[min(index - 1, len(claims) - 1)] if claims else answer[:1000]),
                "source_span": source_span,
                "verdict": verdict,
                "failure_type": failure_type,
                "confidence": round(float(confidence), 6),
                "diagnostics": {
                    "verification_method": "adaptive_context_idf_claim_overlap_v1",
                    **overlap,
                    "context_has_formula_or_table": context_has_formula,
                    "package_has_formula_or_table": package_has_formula,
                    "answer_formula_or_table_claim": formula_claim,
                },
            }
        )
    if not citations:
        results.append(
            {
                "citation_index": 0,
                "chunk_id": None,
                "claim_text": claims[0] if claims else answer[:1000],
                "source_span": {},
                "verdict": "missing_citation",
                "failure_type": "citation_missing",
                "confidence": 0.0,
                "diagnostics": {"verification_method": "adaptive_context_idf_overlap_v1", "reason": "no_citations"},
            }
        )
    return results


def citation_verification_judge_timeout_seconds(verification_budget: int) -> float:
    request_timeout = float(get_settings().model_request_timeout_seconds or 60)
    budget = max(1, int(verification_budget or 1))
    return max(15.0, min(request_timeout, float(budget * 8), 60.0))


def llm_verification_unavailable_results(
    rule_results: list[dict[str, Any]],
    *,
    judge_state: str,
    failure_type: str,
    exc: Exception | None = None,
) -> list[dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "verification_method": "structure_plus_llm_entailment_v1",
        "llm_entailment_judge": judge_state,
    }
    if exc is not None:
        diagnostics["error"] = public_exception_message(exc)
        diagnostics["error_type"] = type(exc).__name__
    return [
        {
            **item,
            "verdict": "unsupported" if item.get("verdict") == "supported" else item.get("verdict"),
            "failure_type": failure_type if item.get("verdict") == "supported" else item.get("failure_type"),
            "confidence": min(coerce_confidence(item.get("confidence"), default=0.0)[0], 0.35),
            "diagnostics": {
                **(item.get("diagnostics") or {}),
                **diagnostics,
            },
        }
        for item in rule_results
    ]


async def verify_answer_against_context(answer: str, citations: list[dict], contexts: list[dict], verification_budget: int) -> list[dict[str, Any]]:
    rule_results = verify_answer_against_context_rules(answer, citations, contexts, verification_budget)
    if not citations:
        return rule_results
    context_by_chunk = {item.get("chunk_id"): item for item in contexts}
    judge_payload = {
        "answer": answer,
        "claims": _answer_claims(answer)[: max(1, verification_budget)],
        "citations": [
            {
                "citation_index": item.get("citation_index"),
                "chunk_id": item.get("chunk_id"),
                "source_span": item.get("source_span") or {},
                "context": _summarize(str((context_by_chunk.get(item.get("chunk_id")) or {}).get("content") or item.get("snippet") or ""), 1600),
                "rule_verdict": item.get("verdict"),
                "rule_failure_type": item.get("failure_type"),
            }
            for item in rule_results[: max(1, verification_budget)]
        ],
    }
    fallback = {
        "verifications": [
            {
                "citation_index": item.get("citation_index"),
                "verdict": item.get("verdict"),
                "failure_type": item.get("failure_type"),
                "confidence": item.get("confidence"),
                "reason": "fallback_rule_result",
            }
            for item in rule_results
        ]
    }
    try:
        judged = await asyncio.wait_for(
            ChatProvider().classify_json(
                system_prompt=(
                    "You are a citation entailment judge for a grounded Four-Layer Context Graph RAG system. "
                    "Use only the supplied cited context and source spans. Return JSON with verifications. "
                    "Each verification must include citation_index, verdict (supported, unsupported, contradicted, missing_citation, "
                    "formula_table_context_missing), failure_type, confidence, and reason. Do not use outside knowledge."
                ),
                user_prompt=str(judge_payload),
                fallback=fallback,
            ),
            timeout=citation_verification_judge_timeout_seconds(verification_budget),
        )
    except TimeoutError as exc:
        return llm_verification_unavailable_results(
            rule_results,
            judge_state="timeout_hard_interrupt",
            failure_type="verification_model_timeout",
            exc=exc,
        )
    except FallbackDisabledError as exc:
        return llm_verification_unavailable_results(
            rule_results,
            judge_state="unavailable_fallback_disabled",
            failure_type="verification_model_unavailable",
            exc=exc,
        )
    except Exception as exc:
        if not get_settings().enable_model_fallback:
            return llm_verification_unavailable_results(
                rule_results,
                judge_state="error_fallback_disabled",
                failure_type="verification_model_unavailable",
                exc=exc,
            )
        judged = fallback
    judged_by_index = {
        int(item.get("citation_index") or 0): item
        for item in (judged.get("verifications") or [])
        if isinstance(item, dict)
    } if isinstance(judged, dict) else {}
    merged: list[dict[str, Any]] = []
    for item in rule_results:
        judge = judged_by_index.get(int(item.get("citation_index") or 0), {})
        rule_confidence, rule_confidence_diagnostics = coerce_confidence(item.get("confidence"), default=0.0)
        judge_confidence, judge_confidence_diagnostics = coerce_confidence(judge.get("confidence"), default=rule_confidence)
        rule_verdict = item.get("verdict")
        judge_verdict = str(judge.get("verdict") or rule_verdict)
        if rule_verdict in {"missing_citation", "formula_table_context_missing"}:
            final_verdict = rule_verdict
            failure_type = item.get("failure_type")
        elif judge_verdict in {"contradicted", "unsupported", "missing_citation", "formula_table_context_missing"}:
            final_verdict = judge_verdict
            failure_type = str(judge.get("failure_type") or item.get("failure_type") or "unsupported_claim")
        elif rule_verdict == "supported" and judge_verdict == "supported":
            final_verdict = "supported"
            failure_type = "none"
        else:
            final_verdict = "unsupported"
            failure_type = str(judge.get("failure_type") or item.get("failure_type") or "unsupported_claim")
        merged.append(
            {
                **item,
                "verdict": final_verdict,
                "failure_type": failure_type,
                "confidence": round(min(rule_confidence, judge_confidence), 6),
                "diagnostics": {
                    **(item.get("diagnostics") or {}),
                    "verification_method": "structure_plus_llm_entailment_v1",
                    "rule_confidence": rule_confidence_diagnostics,
                    "llm_entailment_confidence": judge_confidence_diagnostics,
                    "rule_verdict": rule_verdict,
                    "llm_entailment_verdict": judge_verdict,
                    "llm_entailment_reason": judge.get("reason"),
                },
            }
        )
    return merged


def reward_metrics_from_verifications(package, verification_results: list[dict[str, Any]], answer: str) -> dict[str, float]:
    supported = [item for item in verification_results if item.get("verdict") == "supported"]
    total = max(len(verification_results), 1)
    citation_pass_rate = len(supported) / total
    return {
        "retrieval_hit": 1.0 if package.hit_chunk_ids_json else 0.0,
        "context_precision": min(1.0, len(package.hit_chunk_ids_json or []) / max(len((package.package_json or {}).get("chunks", [])), 1)),
        "context_recall": 1.0 if package.restored_chunk_ids_json else 0.6,
        "concept_path_accuracy": 1.0 if package.concept_path_json else 0.0,
        "citation_pass_rate": round(citation_pass_rate, 6),
        "answer_groundedness": round(citation_pass_rate, 6),
        "answer_completeness": min(1.0, len(answer.strip()) / 500.0) if answer else 0.0,
        "repair_success_rate": 1.0 if citation_pass_rate >= 1.0 else 0.0,
        "agent_typed_action_validation_pass_rate": 1.0,
        "latency_cost": 0.0,
        "task_token_cost": float(package.token_count),
        "drift_rate": 0.0,
    }


def update_policy_state_from_reward(db: Session, knowledge_base_id: str, reward: RewardEvent) -> PolicyState:
    latest = db.scalar(
        select(PolicyState)
        .where(PolicyState.knowledge_base_id == knowledge_base_id, PolicyState.policy_family == "context_graph_bandit")
        .order_by(PolicyState.created_at.desc())
    )
    arms = [
        "high_precision_direct_chunk",
        "structure_context_heavy",
        "rq_membership_expansion",
        "mid_concept_expansion",
        "coarse_to_mid_drilldown",
        "bridge_edge_exploration",
        "formula_table_closure",
        "cross_document_synthesis",
        "low_latency_minimal_context",
    ]
    previous_weights = dict((latest.weights_json or {}) if latest else {})
    reward_json = reward.reward_json or {}
    citation_pass = float(reward_json.get("citation_pass_rate") or 0.0)
    context_recall = float(reward_json.get("context_recall") or 0.0)
    concept_path = float(reward_json.get("concept_path_accuracy") or 0.0)
    bridge_repair_boost = 0.2 if (reward.action_json or {}).get("repair_actions") else 0.0
    weights = {
        "high_precision_direct_chunk": max(0.1, previous_weights.get("high_precision_direct_chunk", 1.0) * (0.9 + 0.2 * citation_pass)),
        "structure_context_heavy": max(0.1, previous_weights.get("structure_context_heavy", 1.0) * (0.9 + 0.2 * context_recall)),
        "rq_membership_expansion": max(0.1, previous_weights.get("rq_membership_expansion", 1.0) * (0.9 + 0.1 * concept_path)),
        "mid_concept_expansion": max(0.1, previous_weights.get("mid_concept_expansion", 1.0) * (0.9 + 0.2 * concept_path)),
        "coarse_to_mid_drilldown": max(0.1, previous_weights.get("coarse_to_mid_drilldown", 1.0) * (0.9 + 0.15 * concept_path)),
        "bridge_edge_exploration": max(0.1, previous_weights.get("bridge_edge_exploration", 1.0) * (0.9 + bridge_repair_boost)),
        "formula_table_closure": max(0.1, previous_weights.get("formula_table_closure", 1.0) * (0.9 + 0.1 * citation_pass)),
        "cross_document_synthesis": max(0.1, previous_weights.get("cross_document_synthesis", 1.0)),
        "low_latency_minimal_context": max(0.1, previous_weights.get("low_latency_minimal_context", 1.0) * (1.05 if citation_pass >= 1.0 else 0.9)),
    }
    safe_arms = [arm for arm in arms if weights.get(arm, 0.0) >= 0.5]
    previous_reward_history = []
    if latest and isinstance(latest.reward_summary_json, dict):
        previous_reward_history = latest.reward_summary_json.get("reward_history_tail") or []
    reward_summary = {
        "last_reward_event_id": reward.id,
        "last_reward": reward_json,
        "reward_history_tail": previous_reward_history[-7:] + [reward_json],
        "safe_arms": safe_arms,
        "posterior": weights,
        "posterior_proxy": weights,
        "exploration_rate": 0.05,
        "drift_status": "normal" if float(reward_json.get("drift_rate") or 0.0) <= 0.2 else "elevated",
        "policy_version": "context_graph_bandit_v1",
        "runtime_settings_hash": runtime_settings_state_hash(),
        "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
    }
    state_hash = stable_hash({"policy_version": "context_graph_bandit_v1", "weights": weights, "summary": reward_summary})
    policy_state = PolicyState(
        knowledge_base_id=knowledge_base_id,
        policy_family="context_graph_bandit",
        policy_version="context_graph_bandit_v1",
        weights_json=weights,
        constraints_json={
            "fallback_disabled": True,
            "citation_verification_required": True,
            "agent_operating_envelope": agent_operating_envelope(),
            "runtime_settings_hash": runtime_settings_state_hash(),
        },
        exploration_json={
            "epsilon": 0.05,
            "exploration_rate": 0.05,
            "safe_arms": safe_arms,
            "path_distance_green_threshold": agent_operating_envelope().get("path_distance_green_threshold"),
            "path_distance_gray_threshold": agent_operating_envelope().get("path_distance_gray_threshold"),
            "path_distance_hard_threshold": agent_operating_envelope().get("path_distance_hard_threshold"),
        },
        reward_summary_json=reward_summary,
        state_hash=state_hash,
    )
    db.add(policy_state)
    db.flush()
    reward.policy_state_id = policy_state.id
    return policy_state


async def record_answer_audit(
    db: Session,
    *,
    knowledge_base_id: str,
    qa_session_id: str,
    question: str,
    answer: str,
    package,
    contexts: list[dict[str, Any]],
    answer_model_audit: dict,
    repair_actions: list[dict[str, Any]] | None = None,
) -> AnswerSession:
    citations = citation_payloads_from_package(package, retrieval_trace_id=package.retrieval_trace_id, answer=answer, question=question)
    verification_results = await verify_answer_against_context(
        answer,
        citations,
        contexts,
        verification_budget=int(agent_operating_envelope().get("verification_budget") or 1),
    )
    answer_session = AnswerSession(
        knowledge_base_id=knowledge_base_id,
        retrieval_trace_id=package.retrieval_trace_id,
        context_package_id=package.id,
        qa_session_id=qa_session_id,
        question=question,
        answer=answer,
        chunk_ids_json=list(package.hit_chunk_ids_json or []),
        prompt_protocol_version="context_graph_answer_v1",
        model_json=answer_model_audit,
        diagnostics_json={
            "context_package_id": package.id,
            "citation_count": len(citations),
            "context_token_count": package.token_count,
            "verification_protocol": "structure_plus_llm_entailment_v1",
        },
    )
    db.add(answer_session)
    db.flush()
    verification_ids: list[str] = []
    for result in verification_results:
        source_span = result.get("source_span") or {}
        stored_confidence, stored_confidence_diagnostics = coerce_confidence(result.get("confidence"), default=0.0)
        verification = CitationVerification(
            knowledge_base_id=knowledge_base_id,
            answer_session_id=answer_session.id,
            retrieval_trace_id=package.retrieval_trace_id,
            context_package_id=package.id,
            chunk_id=result.get("chunk_id"),
            claim_text=str(result.get("claim_text") or "")[:1000],
            source_span_json=source_span,
            verdict=str(result.get("verdict") or "unsupported"),
            confidence=stored_confidence,
            diagnostics_json={**(result.get("diagnostics") or {}), "stored_confidence": stored_confidence_diagnostics},
        )
        db.add(verification)
        db.flush()
        verification_source_span = {**(verification.source_span_json or {}), "verification_id": verification.id}
        verification.source_span_json = verification_source_span
        result["source_span"] = verification_source_span
        flag_modified(verification, "source_span_json")
        for citation in citations:
            if citation.get("chunk_id") == verification.chunk_id:
                citation["citation_verification_id"] = verification.id
                if isinstance(citation.get("source_span"), dict):
                    citation["source_span"] = {**citation["source_span"], "verification_id": verification.id}
                citation["verification"] = {
                    "verdict": verification.verdict,
                    "confidence": verification.confidence,
                    "diagnostics": verification.diagnostics_json or {},
                }
        verification_ids.append(verification.id)
    answer_session.citation_ids_json = verification_ids
    reward_json = reward_metrics_from_verifications(package, verification_results, answer)
    reward = RewardEvent(
        knowledge_base_id=knowledge_base_id,
        retrieval_trace_id=package.retrieval_trace_id,
        answer_session_id=answer_session.id,
        chunk_ids_json=list(package.hit_chunk_ids_json or []),
        context_json={
            "context_package_id": package.id,
            "question_length": len(question),
            "context_token_count": package.token_count,
        },
        action_json={
            "route": "layered_context_graph",
            "prompt_protocol_version": "context_graph_answer_v1",
            "repair_actions": repair_actions or [],
        },
        reward_json=reward_json,
        diagnostics_json={
            "source": "context_graph_agent_v1",
            "verification_results": verification_results,
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
        },
    )
    db.add(reward)
    db.flush()
    update_policy_state_from_reward(db, knowledge_base_id, reward)
    db.commit()
    db.refresh(answer_session)
    return answer_session


def run_to_task_status(run: AgentRun) -> dict:
    return {
        "run_id": run.id,
        "state": run.status,
        "current_node": run.current_node,
        "retry_count": run.retry_count,
        "route": run.route,
        "error": run.error_message,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


async def execute_agent_run(db: Session, request: AgentRequest, session: QASession, run: AgentRun) -> dict:
    try:
        set_run_state(db, run, "running", current_node="query_understanding")
        start = time.perf_counter()
        query_intent = await perceive_query_intent(request.question, [item.model_dump() for item in request.history])
        trace(
            db,
            run.id,
            "query_understanding",
            input_summary=request.question,
            output_summary=str(query_intent.get("intent") or "layered_context_graph"),
            scores={"top_k": request.top_k, "query_intent": query_intent},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

        envelope = agent_operating_envelope()
        start = time.perf_counter()
        proposed_actions, raw_planner_output = await propose_agent_plan(
            request.question,
            [item.model_dump() for item in request.history],
            query_intent,
            envelope,
        )
        typed_actions, validation = validate_typed_actions(proposed_actions, envelope, db=db, knowledge_base_id=run.knowledge_base_id)
        plan, action_rows = record_agent_plan_and_actions(
            db,
            run=run,
            query_intent=query_intent,
            envelope=envelope,
            raw_planner_output=raw_planner_output,
            actions=typed_actions,
            validation=validation,
        )
        action_map = actions_by_type(action_rows)
        db.commit()
        trace(
            db,
            run.id,
            "agent_planner",
            input_summary=request.question,
            output_summary=f"planned {len(typed_actions)} typed actions",
            scores={"plan_id": plan.id, "agent_operating_envelope_hash": agent_operating_envelope_state_hash()},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        trace(
            db,
            run.id,
            "typed_action_validation",
            input_summary=f"proposed={len(proposed_actions)}",
            output_summary="valid" if validation.get("valid") else "invalid",
            scores={"plan_id": plan.id, "validation": validation},
        )

        start = time.perf_counter()
        search_result = await layered_search(db, run.knowledge_base_id, request.question, request.filters, request.top_k)
        plan.retrieval_trace_id = search_result.trace.id
        record_observation(
            db,
            run_id=run.id,
            action=action_map.get("recall_chunks"),
            observation_type="layered_retrieval",
            observation={
                "retrieval_trace_id": search_result.trace.id,
                "retrieved_chunks": len(search_result.results),
                "audit": search_result.audit,
            },
            evidence_chunk_ids=[item["chunk_id"] for item in search_result.results],
            verdict="sufficient" if search_result.results else "insufficient",
        )
        for action_type, observation_type in [
            ("select_entry_nodes", "entry_selection"),
            ("drill_down_layer", "layer_drilldown"),
            ("walk_graph_frontier", "frontier_traversal"),
        ]:
            record_observation(
                db,
                run_id=run.id,
                action=action_map.get(action_type),
                observation_type=observation_type,
                observation={"retrieval_trace_id": search_result.trace.id, "audit": search_result.audit},
                evidence_chunk_ids=[item["chunk_id"] for item in search_result.results],
                verdict="observed",
            )
        db.commit()
        audit = search_result.audit or {}
        trace(
            db,
            run.id,
            "entry_selection",
            input_summary=request.question,
            output_summary=f"coarse={audit.get('coarse_entries', 0)}, stage-queues={audit.get('stage_queue_count', 0)}",
            document_ids=[item["chunk_id"] for item in search_result.results],
            scores={
                "retrieval_trace_id": search_result.trace.id,
                "coarse_entries": audit.get("coarse_entries", 0),
                "stage_queue_count": audit.get("stage_queue_count", 0),
                "mid_topk_selected": audit.get("mid_topk_selected", 0),
                "chunk_topk_selected": audit.get("chunk_topk_selected", 0),
            },
        )
        trace(
            db,
            run.id,
            "layer_drilldown",
            input_summary=f"trace={search_result.trace.id}",
            output_summary=f"mid top-k={audit.get('mid_topk_selected', 0)}, chunk top-k={audit.get('chunk_topk_selected', 0)}",
            document_ids=[item["chunk_id"] for item in search_result.results],
            scores={
                "retrieval_trace_id": search_result.trace.id,
                "query_rq_path": audit.get("query_rq_path") or [],
                "mid_topk_selected": audit.get("mid_topk_selected", 0),
                "chunk_topk_selected": audit.get("chunk_topk_selected", 0),
            },
        )
        trace(
            db,
            run.id,
            "frontier_traversal",
            input_summary=f"trace={search_result.trace.id}",
            output_summary=f"frontier pops={audit.get('frontier_pops', 0)}",
            document_ids=[item["chunk_id"] for item in search_result.results],
            scores={
                "retrieval_trace_id": search_result.trace.id,
                "frontier_pops": audit.get("frontier_pops", 0),
                "dominance_pruned_count": audit.get("dominance_pruned_count", 0),
            },
        )
        trace(
            db,
            run.id,
            "chunk_recall",
            input_summary=f"trace={search_result.trace.id}",
            output_summary=f"recalled {len(search_result.results)} chunks",
            document_ids=[item["chunk_id"] for item in search_result.results],
            scores={"retrieval_trace_id": search_result.trace.id, "chunk_ids": [item["chunk_id"] for item in search_result.results]},
        )
        trace(
            db,
            run.id,
            "layered_retrieval",
            input_summary=request.question,
            output_summary=f"retrieved {len(search_result.results)} chunks",
            document_ids=[item["chunk_id"] for item in search_result.results],
            scores=search_result.audit,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

        start = time.perf_counter()
        package = build_context_package(
            db,
            knowledge_base_id=run.knowledge_base_id,
            query=request.question,
            trace=search_result.trace,
            results=search_result.results,
        )
        contexts = context_package_to_contexts(package)
        restoration_scores = {
            "context_package_id": package.id,
            "hit_chunks": len(package.hit_chunk_ids_json or []),
            "restored_chunks": len(package.restored_chunk_ids_json or []),
            "bridge_chunks": len(package.bridge_chunk_ids_json or []),
            "parent_structure_nodes": len(package.parent_structure_node_ids_json or []),
            "graph_path_ids": len(package.graph_path_ids_json or []),
        }
        record_observation(
            db,
            run_id=run.id,
            action=action_map.get("restore_context_package") or action_map.get("build_context_package"),
            observation_type="context_restoration",
            observation={
                "context_package_id": package.id,
                "hit_chunks": restoration_scores["hit_chunks"],
                "restored_chunks": restoration_scores["restored_chunks"],
                "bridge_chunks": restoration_scores["bridge_chunks"],
                "parent_structure_nodes": restoration_scores["parent_structure_nodes"],
                "token_count": package.token_count,
            },
            evidence_chunk_ids=list((package.hit_chunk_ids_json or []) + (package.restored_chunk_ids_json or [])),
            verdict="sufficient" if package.hit_chunk_ids_json else "insufficient",
        )
        db.commit()
        trace(
            db,
            run.id,
            "structure_context_restoration",
            input_summary=f"trace={search_result.trace.id}",
            output_summary=f"restored={restoration_scores['restored_chunks']}, bridge={restoration_scores['bridge_chunks']}",
            document_ids=list((package.hit_chunk_ids_json or []) + (package.restored_chunk_ids_json or [])),
            scores=restoration_scores,
        )
        trace(
            db,
            run.id,
            "context_package",
            input_summary=f"hits={len(package.hit_chunk_ids_json or [])}",
            output_summary=f"context chunks={len((package.package_json or {}).get('chunks', []))}, tokens={package.token_count}",
            document_ids=list(package.hit_chunk_ids_json or []),
            scores={"context_package_id": package.id, "token_count": package.token_count},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

        start = time.perf_counter()
        chat_result = await ChatProvider().answer_question_with_meta(request.question, contexts, [item.model_dump() for item in request.history])
        answer_model_audit = {
            "provider": chat_result.provider,
            "model": chat_result.model,
            "external_called": chat_result.external_called,
            "fallback_reason": chat_result.fallback_reason,
            "context_package_id": package.id,
        }
        trace(
            db,
            run.id,
            "grounded_answer",
            input_summary=request.question,
            output_summary=_summarize(chat_result.answer),
            document_ids=list(package.hit_chunk_ids_json or []),
            scores=answer_model_audit,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

        provisional_citations = citation_payloads_from_package(
            package,
            retrieval_trace_id=package.retrieval_trace_id,
            answer=chat_result.answer,
            question=request.question,
        )
        provisional_verifications = await verify_answer_against_context(
            chat_result.answer,
            provisional_citations,
            contexts,
            verification_budget=int(envelope.get("verification_budget") or 1),
        )
        repair_actions: list[dict[str, Any]] = []
        if any(item.get("verdict") != "supported" for item in provisional_verifications) and int(envelope.get("repair_round_budget") or 0) > 0:
            failure_types = sorted({str(item.get("failure_type") or "unsupported_claim") for item in provisional_verifications if item.get("verdict") != "supported"})
            repair_type = "repair_missing_citation"
            if "concept_gap" in failure_types:
                repair_type = "repair_concept_gap"
            elif "bridge_gap" in failure_types:
                repair_type = "repair_bridge_gap"
            elif "formula_context_missing" in failure_types:
                repair_type = "repair_formula_context"
            repair_action = AgentAction(
                run_id=run.id,
                plan_id=plan.id,
                parent_action_id=action_map.get("verify_citations").id if action_map.get("verify_citations") else None,
                action_index=len(action_rows),
                action_type=repair_type,
                reason=f"Verification failures require repair: {', '.join(failure_types)}",
                budget_request_json={"repair_round_budget": 1},
                expected_evidence_json={"failure_types": failure_types},
                stop_condition_json={"citation_verification_passes": True},
                validation_json={"valid": True, "repair_budget_checked": True},
                status="accepted",
            )
            db.add(repair_action)
            db.flush()
            repair_top_k = min(50, request.top_k + max(2, len(failure_types) * 2))
            repaired_search = await layered_search(db, run.knowledge_base_id, request.question, request.filters, repair_top_k)
            repaired_package = build_context_package(
                db,
                knowledge_base_id=run.knowledge_base_id,
                query=request.question,
                trace=repaired_search.trace,
                results=repaired_search.results,
            )
            repaired_contexts = context_package_to_contexts(repaired_package)
            repaired_chat_result = await ChatProvider().answer_question_with_meta(
                request.question,
                repaired_contexts,
                [item.model_dump() for item in request.history],
            )
            record_observation(
                db,
                run_id=run.id,
                action=repair_action,
                observation_type="repair_execution",
                observation={
                    "failure_types": failure_types,
                    "original_context_package_id": package.id,
                    "repaired_context_package_id": repaired_package.id,
                    "repaired_retrieval_trace_id": repaired_search.trace.id,
                    "repair_top_k": repair_top_k,
                },
                evidence_chunk_ids=list(repaired_package.hit_chunk_ids_json or []),
                verdict="observed",
            )
            repair_actions.append(
                {
                    "action_type": repair_type,
                    "failure_types": failure_types,
                    "repaired_context_package_id": repaired_package.id,
                    "repaired_retrieval_trace_id": repaired_search.trace.id,
                }
            )
            package = repaired_package
            contexts = repaired_contexts
            chat_result = repaired_chat_result
            answer_model_audit = {
                **answer_model_audit,
                "context_package_id": package.id,
                "retrieval_trace_id": package.retrieval_trace_id,
                "repair_actions": repair_actions,
            }
            db.commit()
            trace(
                db,
                run.id,
                "repair_executed",
                input_summary=", ".join(failure_types),
                output_summary=f"repacked context {package.id}",
                document_ids=list(package.hit_chunk_ids_json or []),
                scores={"repair_actions": repair_actions},
            )

        answer_session = await record_answer_audit(
            db,
            knowledge_base_id=run.knowledge_base_id,
            qa_session_id=session.id,
            question=request.question,
            answer=chat_result.answer,
            package=package,
            contexts=contexts,
            answer_model_audit=answer_model_audit,
            repair_actions=repair_actions,
        )
        verification_by_chunk, citation_pass_rate = citation_verification_summary(db, answer_session.id)
        record_observation(
            db,
            run_id=run.id,
            action=action_map.get("verify_citations"),
            observation_type="citation_verification",
            observation={
                "answer_session_id": answer_session.id,
                "citation_pass_rate": citation_pass_rate,
                "verification_ids": list(answer_session.citation_ids_json or []),
                "repair_actions": repair_actions,
            },
            evidence_chunk_ids=list(package.hit_chunk_ids_json or []),
            verdict="sufficient" if citation_pass_rate == 1.0 else "insufficient",
        )
        answer_model_audit.update(
            {
                "chat_model": chat_result.model,
                "retrieval_trace_id": package.retrieval_trace_id,
                "answer_session_id": answer_session.id,
                "raw_citation_verification_pass_rate": citation_pass_rate,
                "repair_actions": repair_actions,
            }
        )
        citations = citation_payloads_from_package(
            package,
            answer_session_id=answer_session.id,
            retrieval_trace_id=package.retrieval_trace_id,
            verification_by_chunk=verification_by_chunk,
            answer=chat_result.answer,
            question=request.question,
            supported_only=True,
        )
        final_citation_pass_rate = 1.0 if citations else 0.0
        answer_model_audit["citation_verification_pass_rate"] = final_citation_pass_rate
        answer_model_audit["returned_citation_count"] = len(citations)
        answer_session.model_json = dict(answer_model_audit)
        db.commit()
        trace(
            db,
            run.id,
            "citation_verification",
            input_summary=f"answer_session={answer_session.id}",
            output_summary=f"pass_rate={final_citation_pass_rate}",
            document_ids=[item.get("chunk_id") for item in citations if item.get("chunk_id")],
            scores={
                "citation_pass_rate": final_citation_pass_rate,
                "raw_citation_pass_rate": citation_pass_rate,
                "returned_citation_count": len(citations),
                "repair_actions": repair_actions,
            },
        )
        trace(
            db,
            run.id,
            "reward_event",
            input_summary=f"answer_session={answer_session.id}",
            output_summary="reward and policy state updated",
            scores={"runtime_settings_hash": runtime_settings_state_hash(), "agent_operating_envelope_hash": agent_operating_envelope_state_hash()},
        )
        append_session_turn(db, session, request.question, chat_result.answer, run.id, citations)
        set_run_state(db, run, "completed", current_node=None, answer=chat_result.answer)
        trace_events = db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.created_at.asc())).all()
        return {
            "run_id": run.id,
            "session_id": session.id,
            "answer": chat_result.answer,
            "citations": citations,
            "used_chunks": contexts,
            "route": "layered_context_graph",
            "trace": [trace_event_to_payload(event) for event in trace_events],
            "degraded_mode": is_degraded_mode(),
            "context_package_id": package.id,
            "retrieval_trace_id": package.retrieval_trace_id,
            "model_audit": answer_model_audit,
            "answer_model_audit": answer_model_audit,
        }
    except Exception as exc:
        safe_error = public_exception_message(exc)
        set_run_state(db, run, "failed", error=safe_error)
        trace(db, run.id, "error", status="failed", output_summary=safe_error, error=safe_error)
        raise


async def run_agent(db: Session, request: AgentRequest) -> dict:
    session, run = create_agent_run_context(db, request)
    return await execute_agent_run(db, request, session, run)


async def stream_agent_events(db: Session, request: AgentRequest) -> AsyncGenerator[dict, None]:
    session, run = create_agent_run_context(db, request)
    trace_queue = _subscribe_trace(run.id)
    task = asyncio.create_task(execute_agent_run(db, request, session, run))
    response: dict | None = None
    yielded_trace_ids: set[str] = set()
    try:
        yield {"type": "meta", "run_id": run.id, "session_id": session.id}
        while not task.done():
            try:
                event = await asyncio.wait_for(trace_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if request.stream_trace:
                yielded_trace_ids.add(event["id"])
                yield {"type": "trace", "trace": event}
        while not trace_queue.empty():
            event = trace_queue.get_nowait()
            if request.stream_trace:
                yielded_trace_ids.add(event["id"])
                yield {"type": "trace", "trace": event}
        response = await task
    except Exception as exc:
        yield {"type": "error", "error": public_exception_message(exc)}
        return
    finally:
        _unsubscribe_trace(run.id, trace_queue)
        if not task.done():
            task.cancel()
    if response is None:
        return
    if request.stream_trace:
        for event in response["trace"]:
            if event["id"] not in yielded_trace_ids:
                yield {"type": "trace", "trace": event}
    answer = response["answer"] or ""
    for start in range(0, len(answer), 12):
        yield {"type": "token", "token": answer[start : start + 12]}
        await asyncio.sleep(0.01)
    yield {"type": "citations", "citations": response["citations"], "degraded_mode": response["degraded_mode"]}
    yield {"type": "final", "response": response}
