"""Active one-plan, one-retrieval, one-generation Agent serving path."""
from __future__ import annotations

import asyncio
from datetime import datetime
import hashlib
import re
import time

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import get_settings
from app.models import AgentObservation, AgentTraceEvent, AnswerSession, Chunk, RetrievalTrace
from app.retrieval_control_contracts import control_hash
from app.services.agent_reflection import history_summary_projection, render_answer_units
from app.services.answer_sources import (
    build_answer_evidence_manifest,
    persist_answer_source_bindings,
    source_binding_citations,
)
from app.services.context_graph import (
    agent_operating_envelope,
    build_context_package,
    context_package_to_contexts,
)
from app.services.intent_planning import (
    plan_intent_execution,
    retrieval_capability_snapshot,
)
from app.services.layered_execution_v1 import (
    execute_layered_retrieval,
    publish_intent_retrieval_cache,
)
from app.services.qa_performance import QAPerformance, current_qa_performance, qa_stage
from app.services.retrieval_models import RetrievalModels
from app.services.source_integrity_admission import admit_context_package


PROTOCOL = "intent_execution_retrieval_v1"
GENERATION_CALL_PROTOCOL = "single_grounded_generation_call_v1"


def _prefers_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _trace_payloads(db, run_id: str, agent_graph) -> list[dict]:
    rows = db.scalars(
        select(AgentTraceEvent)
        .where(AgentTraceEvent.run_id == run_id)
        .order_by(AgentTraceEvent.sequence_index)
    ).all()
    return [agent_graph.trace_event_to_payload(item) for item in rows]


def _response(
    db,
    *,
    agent_graph,
    run,
    session,
    answer: str,
    citations: list[dict],
    used_chunks: list[dict],
    route: str,
    terminal_outcome: str,
    plan=None,
    answer_session=None,
    package=None,
    trace=None,
    model_audit=None,
    conversation_state=None,
) -> dict:
    return {
        "run_id": run.id,
        "session_id": session.id,
        "answer": answer,
        "citations": citations,
        "used_chunks": used_chunks,
        "route": route,
        "direct_answer_mode": "system_capability" if route == "system_capability" else (
            "verified_context_reuse" if route == "verified_context_reuse" else None
        ),
        "trace": _trace_payloads(db, run.id, agent_graph),
        "degraded_mode": agent_graph.is_degraded_mode(),
        "context_package_id": package.id if package is not None else None,
        "retrieval_trace_id": trace.id if trace is not None else (
            package.retrieval_trace_id if package is not None else None
        ),
        "entry_layer": plan.strategy.entry_layer if plan is not None else None,
        "intent": plan.intent.model_dump(mode="json") if plan is not None else None,
        "execution_strategy": plan.strategy.model_dump(mode="json") if plan is not None else None,
        "terminal_outcome": terminal_outcome,
        "model_audit": model_audit or {},
        "answer_model_audit": model_audit or {},
        "conversation_state": conversation_state.public_payload() if conversation_state is not None else None,
        "answer_session_id": answer_session.id if answer_session is not None else None,
    }


def _finish_direct(
    db,
    *,
    agent_graph,
    request,
    session,
    run,
    plan,
    answer: str,
    route: str,
    terminal_outcome: str,
    capability_card=None,
) -> dict:
    row = AnswerSession(
        knowledge_base_id=run.knowledge_base_id,
        qa_session_id=session.id,
        question=request.question,
        answer=answer,
        prompt_protocol_version=(
            "system_capability_card_v4" if route == "system_capability" else "intent_clarification_v1"
        ),
        model_json={
            "protocol_version": PROTOCOL,
            "accepted_plan_hash": plan.identity,
            "planning_model_call_count": 1,
            "generation_model_call_count": 0,
            "answer_model_called": False,
            "tool_call_count": 0,
            "capability_card_hash": (capability_card or {}).get("card_hash"),
            "policy_update_eligible": False,
        },
        diagnostics_json={
            "agent_run_id": run.id,
            "terminal_outcome": terminal_outcome,
            "retrieval_trace_id": None,
            "context_package_id": None,
            "route": route,
            "direct_answer_mode": "system_capability" if route == "system_capability" else None,
            "policy_update_eligible": False,
        },
    )
    db.add(row)
    db.flush()
    final_state = agent_graph.append_session_turn(
        db,
        session,
        request.question,
        answer,
        run.id,
        [],
        answer_session_id=row.id,
        task_status="waiting_user" if terminal_outcome == "scope_ambiguous" else "active",
        route=route,
        direct_answer_mode="system_capability" if route == "system_capability" else None,
        commit=False,
    )
    run.status = "needs_clarification" if terminal_outcome == "scope_ambiguous" else "completed"
    run.current_node = None
    run.completed_at = datetime.utcnow()
    run.final_answer = answer
    run.route = route
    run.metadata_json = {
        **dict(run.metadata_json or {}),
        "answer_session_id": row.id,
        "terminal_outcome": terminal_outcome,
        "direct_answer_mode": "system_capability" if route == "system_capability" else None,
        "policy_update_eligible": False,
    }
    flag_modified(run, "metadata_json")
    db.commit()
    return _response(
        db,
        agent_graph=agent_graph,
        run=run,
        session=session,
        answer=answer,
        citations=[],
        used_chunks=[],
        route=route,
        terminal_outcome=terminal_outcome,
        plan=plan,
        answer_session=row,
        model_audit=row.model_json,
        conversation_state=final_state,
    )


def _gap_text(question: str, outcome: str) -> str:
    chinese = _prefers_chinese(question)
    messages = {
        "insufficient_evidence": (
            "本次已检查的材料不足以支持可靠回答；这不表示整个资料库中不存在相关信息。",
            "The material checked in this request is insufficient for a reliable answer; this does not establish that the whole library lacks it.",
        ),
        "scope_ambiguous": (
            "请进一步明确要核对的对象、文档或章节范围。",
            "Please clarify the object, document, or section scope to check.",
        ),
        "representation_incomplete": (
            "指定来源的当前解析表示不完整，暂时无法按该范围可靠作答。",
            "The current parsed representation of the requested source is incomplete, so that scope cannot yet be answered reliably.",
        ),
        "context_budget_exhausted": (
            "必需的来源范围超出本次上下文容量，请缩小范围后重试。",
            "The required source scope exceeds this request's context capacity; narrow the scope and retry.",
        ),
    }
    zh, en = messages.get(outcome, messages["insufficient_evidence"])
    return zh if chinese else en


def _build_intent_context_package(
    db,
    *,
    request,
    plan,
    execution,
    token_budget: int,
):
    """Pack deterministic source-scope targets before ranked hit context."""

    targets = tuple(dict.fromkeys(execution.scope_target_chunk_ids or ()))
    target_set = set(targets)
    scope_context_chunk_ids: list[str] = []
    adjacency_budget = int(
        (
            (execution.scope_execution_audit or {}).get("target_plan")
            or {}
        ).get("overlap_target_budget")
        or min(4, plan.effective_budget.per_parent_entries)
    )
    if targets and adjacency_budget > 0:
        source_scope_ids = set(
            str(item)
            for item in (execution.scope_execution_audit or {}).get(
                "source_chunk_ids", []
            )
        )
        seed_rows = {
            str(row.id): row
            for row in db.scalars(
                select(Chunk).where(
                    Chunk.id.in_(targets),
                    Chunk.knowledge_base_id == plan.task.knowledge_base_id,
                    Chunk.state == "active",
                )
            )
        }
        if set(seed_rows) != target_set:
            raise ValueError("source_scope_target_chunk_changed")
        document_version_ids = {
            str(row.document_version_id) for row in seed_rows.values()
        }
        local_rows = list(
            db.scalars(
                select(Chunk).where(
                    Chunk.knowledge_base_id == plan.task.knowledge_base_id,
                    Chunk.document_version_id.in_(document_version_ids),
                    Chunk.state == "active",
                )
            )
        )
        rows_by_version: dict[str, list[Chunk]] = {}
        for row in local_rows:
            if source_scope_ids and str(row.id) not in source_scope_ids:
                continue
            rows_by_version.setdefault(str(row.document_version_id), []).append(row)
        positions: dict[str, tuple[list[Chunk], int]] = {}
        for rows in rows_by_version.values():
            rows.sort(key=lambda item: (int(item.chunk_index), str(item.id)))
            for index, row in enumerate(rows):
                positions[str(row.id)] = (rows, index)
        adjacent_ids: list[str] = []
        for target_id in targets:
            rows, index = positions[target_id]
            local_count = 0
            distance = 1
            while local_count < adjacency_budget and (
                index - distance >= 0 or index + distance < len(rows)
            ):
                for neighbor_index in (index - distance, index + distance):
                    if not 0 <= neighbor_index < len(rows):
                        continue
                    adjacent_ids.append(str(rows[neighbor_index].id))
                    local_count += 1
                    if local_count >= adjacency_budget:
                        break
                distance += 1
        scope_context_chunk_ids.extend(
            chunk_id
            for chunk_id in dict.fromkeys(adjacent_ids)
            if chunk_id not in target_set
        )
    package = build_context_package(
        db,
        knowledge_base_id=plan.task.knowledge_base_id,
        query=request.question,
        trace=execution.trace,
        results=execution.results,
        token_budget=token_budget,
        # Required targets are packed first.  Preserve the bounded structural
        # restoration budget because overlap targets identify anchors rather
        # than proving that every adjacent fact is already present.
        restore_per_chunk_budget=plan.effective_budget.restore_per_hit,
        required_scope_chunk_ids=list(targets),
        scope_context_chunk_ids=scope_context_chunk_ids,
    )
    package.diagnostics_json = {
        **dict(package.diagnostics_json or {}),
        "source_scope_execution": execution.scope_execution_audit,
        "source_scope_target_chunk_ids": list(targets),
        "source_scope_required_context_chunk_ids": scope_context_chunk_ids,
        "source_scope_adjacency_protocol_version": (
            "source_scope_chunk_adjacency_v1" if targets else None
        ),
        "source_scope_materialized_chunk_ids": [
            str(item.get("chunk_id") or "")
            for item in (package.package_json or {}).get("chunks", [])
            if str(item.get("chunk_id") or "") in target_set
        ],
    }
    flag_modified(package, "diagnostics_json")
    db.flush()
    return package


def _finish_gap(
    db,
    *,
    agent_graph,
    request,
    session,
    run,
    plan,
    outcome: str,
    package=None,
    trace=None,
    admission=None,
) -> dict:
    answer = _gap_text(request.question, outcome)
    row = AnswerSession(
        knowledge_base_id=run.knowledge_base_id,
        qa_session_id=session.id,
        # A grounding terminal reports the deterministic admission outcome;
        # it does not consume the rejected package as answer facts.  Keep the
        # audit pointers below without creating a reusable conversation source
        # reference for evidence that never passed admission.
        retrieval_trace_id=None,
        context_package_id=None,
        question=request.question,
        answer=answer,
        prompt_protocol_version="deterministic_grounding_terminal_v1",
        model_json={
            "protocol_version": PROTOCOL,
            "accepted_plan_hash": plan.identity,
            "planning_model_call_count": 1,
            "generation_model_call_count": 0,
            "source_admission_model_call_count": 0,
            "terminal_outcome": outcome,
            "policy_update_eligible": False,
        },
        diagnostics_json={
            "agent_run_id": run.id,
            "source_integrity_admission": admission.audit if admission is not None else None,
            "audited_retrieval_trace_id": trace.id if trace is not None else None,
            "audited_context_package_id": package.id if package is not None else None,
        },
    )
    db.add(row)
    db.flush()
    final_state = agent_graph.append_session_turn(
        db,
        session,
        request.question,
        answer,
        run.id,
        [],
        answer_session_id=row.id,
        task_status="waiting_user" if outcome == "scope_ambiguous" else "active",
        route=PROTOCOL,
        commit=False,
    )
    run.status = "needs_clarification" if outcome == "scope_ambiguous" else "completed"
    run.current_node = None
    run.completed_at = datetime.utcnow()
    run.final_answer = answer
    run.route = PROTOCOL
    run.metadata_json = {
        **dict(run.metadata_json or {}),
        "answer_session_id": row.id,
        "terminal_outcome": outcome,
        "policy_update_eligible": False,
    }
    flag_modified(run, "metadata_json")
    db.commit()
    return _response(
        db,
        agent_graph=agent_graph,
        run=run,
        session=session,
        answer=answer,
        citations=[],
        used_chunks=[],
        route=PROTOCOL,
        terminal_outcome=outcome,
        plan=plan,
        answer_session=row,
        package=package,
        trace=trace,
        model_audit=row.model_json,
        conversation_state=final_state,
    )


async def _generate_once(
    db,
    *,
    agent_graph,
    request,
    session,
    run,
    plan,
    package,
    trace,
    admission,
    history_summary: str,
    route: str,
    retrieval_cache: dict | None = None,
) -> dict:
    settings = get_settings()
    with qa_stage("generation_packing"):
        contexts = context_package_to_contexts(package)
        evidence = build_answer_evidence_manifest(package, contexts)
    if admission.audit.get("evidence_manifest_hash") != evidence.manifest_hash:
        raise ValueError("source_admission_evidence_manifest_changed")
    prepared = {
        "protocol_version": GENERATION_CALL_PROTOCOL,
        "status": "prepared",
        "run_id": run.id,
        "accepted_plan_hash": plan.identity,
        "source_integrity_admission_hash": admission.audit["audit_hash"],
        "context_package_id": package.id,
        "evidence_manifest_hash": evidence.manifest_hash,
        "input_hash": control_hash(
            {
                "task": plan.task.model_dump(mode="json"),
                "intent": plan.intent.model_dump(mode="json"),
                "evidence_manifest_hash": evidence.manifest_hash,
                "history_summary_hash": hashlib.sha256(history_summary.encode()).hexdigest(),
            }
        ),
        "provider_response_persisted": False,
    }
    call = AgentObservation(
        run_id=run.id,
        observation_type="single_grounded_generation",
        verdict="prepared",
        observation_json=prepared,
        evidence_chunk_ids_json=list(package.hit_chunk_ids_json or []),
    )
    db.add(call)
    with qa_stage("audit_persistence"):
        db.commit()
    model = RetrievalModels(agent_graph.ChatProvider)
    draft, model_audit = await model.generate(
        task=plan.task,
        evidence=evidence,
        history_summary=history_summary,
        missing_facets=(),
        timeout_seconds=min(
            settings.model_request_timeout_seconds,
            settings.retrieval_generation_timeout_seconds,
        ),
        max_tokens=min(
            settings.chat_json_max_tokens,
            settings.retrieval_generation_max_tokens,
        ),
        unit_limit=settings.agent_answer_unit_limit,
        source_scopes=None,
    )
    agent_graph.ensure_agent_run_not_cancelled(db, run)
    answer, units = render_answer_units(draft)
    completed = {
        **prepared,
        "status": "completed",
        "observation_id": call.id,
        "model_call_count": 1,
        "draft_hash": control_hash(draft.model_dump(mode="json")),
        "model_audit": model_audit,
    }
    completed["audit_hash"] = control_hash(completed)
    call.verdict = "completed"
    call.observation_json = completed
    bound_trace = db.get(RetrievalTrace, package.retrieval_trace_id)
    authoritative_conversation_scope = str(
        bound_trace.conversation_state_scope_hash if bound_trace is not None else ""
    )
    if (
        bound_trace is None
        or bound_trace.knowledge_base_id != run.knowledge_base_id
        or len(authoritative_conversation_scope) != 64
        or str(
            (package.diagnostics_json or {}).get(
                "conversation_state_scope_hash"
            )
            or ""
        )
        != authoritative_conversation_scope
    ):
        raise ValueError("generation_conversation_scope_binding_invalid")
    answer_row = AnswerSession(
        knowledge_base_id=run.knowledge_base_id,
        qa_session_id=session.id,
        retrieval_trace_id=package.retrieval_trace_id,
        context_package_id=package.id,
        question=request.question,
        answer=answer,
        chunk_ids_json=list(package.hit_chunk_ids_json or []),
        prompt_protocol_version="single_grounded_answer_v3",
        model_json={
            "protocol_version": PROTOCOL,
            "accepted_plan_hash": plan.identity,
            "planning_model_call_count": 1,
            "generation_model_call_count": 1,
            "post_generation_model_call_count": 0,
            "source_admission_model_call_count": 0,
            "generation": completed,
            "conversation_state_scope_hash": authoritative_conversation_scope,
            "policy_update_eligible": False,
            "intent_retrieval_cache": retrieval_cache,
        },
        diagnostics_json={
            "agent_run_id": run.id,
            "answer_units": units,
            "source_integrity_admission": admission.audit,
            "conversation_state_scope_hash": authoritative_conversation_scope,
        },
    )
    db.add(answer_row)
    db.flush()
    admission_row = db.get(AgentObservation, admission.observation_id)
    with qa_stage("source_binding"):
        bindings = persist_answer_source_bindings(
            db,
            answer_session=answer_row,
            package=package,
            contexts=contexts,
            draft=draft,
            evidence=evidence,
            unit_limit=settings.agent_answer_unit_limit,
            source_integrity_admission=admission_row,
        )
        citations = source_binding_citations(
            answer_session=answer_row,
            package=package,
            rows=bindings,
            source_integrity_admission=admission_row,
        )
    answer_row.citation_ids_json = [binding.id for binding in bindings]
    answer_row.model_json = {
        **answer_row.model_json,
        "conversation_state_scope_hash": authoritative_conversation_scope,
        "answer_session_id": answer_row.id,
        "source_binding_count": len(bindings),
        "source_binding_pass_rate": 1.0 if bindings else 0.0,
        "returned_citation_count": len(citations),
    }
    answer_row.diagnostics_json = {
        **dict(answer_row.diagnostics_json or {}),
        "agent_run_id": run.id,
        "source_integrity_admission": admission.audit,
        "conversation_state_scope_hash": authoritative_conversation_scope,
        "answer_session_id": answer_row.id,
        "source_binding_count": len(bindings),
    }
    flag_modified(answer_row, "model_json")
    flag_modified(answer_row, "diagnostics_json")
    db.flush()
    persisted_answer_audit = db.execute(
        select(
            AnswerSession.model_json,
            AnswerSession.diagnostics_json,
        ).where(AnswerSession.id == answer_row.id)
    ).one()
    if (
        any(
            not isinstance(value, dict)
            or value.get("conversation_state_scope_hash")
            != authoritative_conversation_scope
            for value in persisted_answer_audit
        )
    ):
        raise ValueError("generation_answer_scope_persistence_failed")
    db.expire(answer_row, ["model_json", "diagnostics_json"])
    if (
        not isinstance(answer_row.model_json, dict)
        or answer_row.model_json.get("conversation_state_scope_hash")
        != authoritative_conversation_scope
        or not isinstance(answer_row.diagnostics_json, dict)
        or answer_row.diagnostics_json.get("conversation_state_scope_hash")
        != authoritative_conversation_scope
    ):
        raise ValueError("generation_answer_scope_reload_failed")
    final_state = agent_graph.append_session_turn(
        db,
        session,
        request.question,
        answer,
        run.id,
        citations,
        answer_session_id=answer_row.id,
        retrieval_trace_id=package.retrieval_trace_id,
        task_status="active",
        route=route,
        direct_answer_mode="verified_context_reuse" if route == "verified_context_reuse" else None,
        commit=False,
    )
    run.status = "completed"
    run.current_node = None
    run.completed_at = datetime.utcnow()
    run.final_answer = answer
    run.route = route
    run.metadata_json = {
        **dict(run.metadata_json or {}),
        "answer_session_id": answer_row.id,
        "terminal_outcome": "completed",
        "generation_model_call_count": 1,
        "post_generation_model_call_count": 0,
        "policy_update_eligible": False,
    }
    flag_modified(run, "metadata_json")
    with qa_stage("database_commit"):
        db.commit()
    return _response(
        db,
        agent_graph=agent_graph,
        run=run,
        session=session,
        answer=answer,
        citations=citations,
        used_chunks=contexts,
        route=route,
        terminal_outcome="completed",
        plan=plan,
        answer_session=answer_row,
        package=package,
        trace=trace,
        model_audit=answer_row.model_json,
        conversation_state=final_state,
    )


async def _execute(db, request, session, run) -> dict:
    from app.services import agent_graph
    from app.services.agent_direct_answer import system_capability_card

    settings = get_settings().model_copy(deep=True)
    run.status = "running"
    run.current_node = "intent_planning"
    run.started_at = run.started_at or datetime.utcnow()
    with qa_stage("database_commit"):
        db.commit()
    with qa_stage("history_projection"):
        history_summary, history_audit = history_summary_projection(
            [item.model_dump() for item in request.history],
            max_characters=settings.agent_history_summary_max_chars,
        )
    with qa_stage("capability_manifest"):
        capabilities, _provisional_context_state = retrieval_capability_snapshot(
            db,
            run.knowledge_base_id,
            admit_graph=False,
        )
    filter_scope_hash = control_hash(request.filters.model_dump(mode="json"))
    conversation_planner_context = (
        (run.metadata_json or {}).get("conversation_state_planner_context") or {}
    )
    reuse_candidate = None
    if conversation_planner_context.get("history_references"):
        reuse_candidate = await agent_graph.run_bounded_source_io(
            agent_graph._latest_verified_context_reuse_candidate,
            db,
            knowledge_base_id=run.knowledge_base_id,
            conversation_planner_context=conversation_planner_context,
        )
    planning_started = time.monotonic()
    with qa_stage("intent_planning"):
        plan, planning_audit = await plan_intent_execution(
            db,
            run=run,
            question=request.question,
            conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
            filter_scope_hash=filter_scope_hash,
            history_summary=history_summary,
            capabilities=capabilities,
            verified_context_reuse_available=reuse_candidate is not None,
            provider_factory=agent_graph.ChatProvider,
        )
    run.metadata_json = {
        **dict(run.metadata_json or {}),
        "history_summary": history_summary,
        "history_summary_audit": history_audit,
        "intent_execution_protocol_version": PROTOCOL,
        "accepted_task": plan.task.model_dump(mode="json"),
        "accepted_intent": plan.intent.model_dump(mode="json"),
        "accepted_execution_strategy": plan.strategy.model_dump(mode="json"),
        "policy_update_eligible": False,
    }
    flag_modified(run, "metadata_json")
    with qa_stage("database_commit"):
        db.commit()
    agent_graph.trace(
        db,
        run.id,
        "intent_planning",
        input_summary=request.question,
        output_summary=f"route={plan.strategy.route}, entry={plan.strategy.entry_layer or 'none'}",
        scores={
            "protocol_version": PROTOCOL,
            "accepted_plan_hash": plan.identity,
            "capability_hash": plan.capability_hash,
            "model_call_count": 1,
        },
        duration_ms=int(round((time.monotonic() - planning_started) * 1000)),
    )
    if plan.strategy.route == "system_capability":
        card = system_capability_card()
        answer = card["localized_answers"]["zh" if _prefers_chinese(request.question) else "en"]["capabilities"]
        return _finish_direct(
            db,
            agent_graph=agent_graph,
            request=request,
            session=session,
            run=run,
            plan=plan,
            answer=answer,
            route="system_capability",
            terminal_outcome="completed",
            capability_card=card,
        )
    if plan.strategy.route == "clarify":
        return _finish_direct(
            db,
            agent_graph=agent_graph,
            request=request,
            session=session,
            run=run,
            plan=plan,
            answer=_gap_text(request.question, "scope_ambiguous"),
            route="clarify",
            terminal_outcome="scope_ambiguous",
        )
    started = time.monotonic()
    if plan.strategy.route == "verified_context_reuse":
        candidate = reuse_candidate
        if candidate is not None:
            package = candidate["context_package"]
            with qa_stage("context_reuse"):
                admission = await agent_graph.run_bounded_source_io(
                    admit_context_package,
                    db,
                    run=run,
                    plan=plan,
                    package=package,
                    filters=request.filters,
                    remaining_seconds=settings.retrieval_total_timeout_seconds - (time.monotonic() - started),
                )
            if admission.passed:
                agent_graph.trace(
                    db,
                    run.id,
                    "verified_context_reuse",
                    output_summary="existing source package replay passed",
                    scores={"source_integrity_admission_hash": admission.audit["audit_hash"]},
                )
                return await _generate_once(
                    db,
                    agent_graph=agent_graph,
                    request=request,
                    session=session,
                    run=run,
                    plan=plan,
                    package=package,
                    trace=candidate["retrieval_trace"],
                    admission=admission,
                    history_summary=history_summary,
                    route="verified_context_reuse",
                )
        agent_graph.trace(
            db,
            run.id,
            "verified_context_reuse",
            output_summary="reuse unavailable; executing the frozen retrieval plan",
            status="completed",
            scores={"fallback_before_retrieval": True, "strategy_hash": plan.strategy.identity},
        )
    run.current_node = "retrieval"
    with qa_stage("database_commit"):
        db.commit()
    admitted_capabilities, admitted_context_state = retrieval_capability_snapshot(
        db,
        run.knowledge_base_id,
        admit_graph=True,
    )
    if admitted_capabilities.identity != plan.capability_hash:
        raise ValueError("strategy_capability_identity_changed")
    retrieval_started = time.monotonic()
    with qa_stage("retrieval"):
        execution = await execute_layered_retrieval(
            db,
            plan=plan,
            filters=request.filters,
            top_k=request.top_k,
            capabilities=admitted_capabilities,
            context_state=admitted_context_state,
        )
    with qa_stage("database_commit"):
        db.commit()
    publish_intent_retrieval_cache(execution)
    agent_graph.trace(
        db,
        run.id,
        "intent_execution_retrieval",
        output_summary=f"retrieved {len(execution.results)} chunks",
        document_ids=[item["chunk_id"] for item in execution.results],
        scores={
            "protocol_version": PROTOCOL,
            "accepted_plan_hash": plan.identity,
            "capability_hash": plan.capability_hash,
            "entry_layer": plan.strategy.entry_layer,
            "retrieval_audit": execution.audit,
        },
        duration_ms=int(round((time.monotonic() - retrieval_started) * 1000)),
    )
    if not execution.results and not execution.scope_target_chunk_ids:
        return _finish_gap(
            db,
            agent_graph=agent_graph,
            request=request,
            session=session,
            run=run,
            plan=plan,
            outcome="insufficient_evidence",
            trace=execution.trace,
        )
    envelope = agent_operating_envelope()
    with qa_stage("context_package"):
        package = await agent_graph.run_bounded_source_io(
            _build_intent_context_package,
            db,
            request=request,
            plan=plan,
            execution=execution,
            token_budget=int(envelope["context_package_token_budget"]),
        )
    admission_started = time.monotonic()
    with qa_stage("source_admission"):
        admission = await agent_graph.run_bounded_source_io(
            admit_context_package,
            db,
            run=run,
            plan=plan,
            package=package,
            filters=request.filters,
            remaining_seconds=settings.retrieval_total_timeout_seconds - (time.monotonic() - started),
        )
    with qa_stage("database_commit"):
        db.commit()
    agent_graph.trace(
        db,
        run.id,
        "source_integrity_admission",
        output_summary=admission.outcome,
        document_ids=list(package.hit_chunk_ids_json or []),
        scores={
            "source_integrity_admission_hash": admission.audit["audit_hash"],
            "model_call_count": 0,
            "score_fields_used": [],
        },
        duration_ms=int(round((time.monotonic() - admission_started) * 1000)),
    )
    if not admission.passed:
        if admission.outcome == "technical_failure":
            raise ValueError("source_integrity_admission_failed")
        return _finish_gap(
            db,
            agent_graph=agent_graph,
            request=request,
            session=session,
            run=run,
            plan=plan,
            outcome=admission.outcome,
            package=package,
            trace=execution.trace,
            admission=admission,
        )
    run.current_node = "generation"
    with qa_stage("database_commit"):
        db.commit()
    return await _generate_once(
        db,
        agent_graph=agent_graph,
        request=request,
        session=session,
        run=run,
        plan=plan,
        package=package,
        trace=execution.trace,
        admission=admission,
        history_summary=history_summary,
        route=PROTOCOL,
        retrieval_cache=execution.cache_audit,
    )


async def execute_intent_execution_agent(db, request, session, run) -> dict:
    """Bound the complete serving request and persist its terminal state."""

    from app.services import agent_graph

    performance = current_qa_performance() or QAPerformance()
    response = None
    with performance.activate():
        try:
            remaining = (
                get_settings().retrieval_total_timeout_seconds
                - performance.snapshot().elapsed_ms / 1000
            )
            async with asyncio.timeout(max(0.001, remaining)):
                with qa_stage("request"):
                    response = await _execute(db, request, session, run)
        except BaseException as exc:
            db.rollback()
            try:
                db.refresh(run)
                if run.status not in agent_graph.TERMINAL_AGENT_RUN_STATUSES:
                    from app.services.reflection_models import AnswerReviewModelError

                    technical_failure = None
                    if isinstance(exc, AnswerReviewModelError):
                        technical_failure = {
                            "stage": exc.stage,
                            "code": exc.code,
                            "cause_type": exc.cause_type,
                            "status_code": exc.status_code,
                            "external_failure": dict(exc.external_failure or {}),
                            "provider_shape": dict(exc.provider_shape or {}),
                        }
                    run.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                    run.current_node = None
                    run.completed_at = datetime.utcnow()
                    run.error_message = (
                        "cancelled_by_user"
                        if isinstance(exc, asyncio.CancelledError)
                        else re.sub(r"[^a-z0-9_]+", "_", exc.__class__.__name__.casefold())[:80]
                    )
                    run.metadata_json = {
                        **dict(run.metadata_json or {}),
                        "terminal_outcome": run.status,
                        "policy_update_eligible": False,
                        **(
                            {"technical_failure": technical_failure}
                            if technical_failure is not None
                            else {}
                        ),
                    }
                    from app.services.conversation_state import (
                        mark_session_task_terminal_for_run,
                    )

                    mark_session_task_terminal_for_run(
                        db,
                        session_id=run.session_id,
                        run_id=run.id,
                        status=run.status,
                    )
                    flag_modified(run, "metadata_json")
                    db.commit()
                    agent_graph.trace(
                        db,
                        run.id,
                        "error" if run.status == "failed" else "cancelled_by_user",
                        status=run.status,
                        output_summary=run.error_message,
                    )
            except Exception:
                db.rollback()
            raise
        finally:
            try:
                db.refresh(run)
                performance_payload = performance.snapshot().model_dump(mode="json")
                run.metadata_json = {
                    **dict(run.metadata_json or {}),
                    "qa_performance": performance_payload,
                }
                flag_modified(run, "metadata_json")
                db.commit()
                if response is not None:
                    response["model_audit"] = {
                        **dict(response.get("model_audit") or {}),
                        "qa_performance": performance_payload,
                    }
                    response["answer_model_audit"] = response["model_audit"]
            except Exception:
                db.rollback()
    return response


async def execute_intent_search(db, request) -> dict:
    """Run the same planner/executor/admission contract without answer generation."""

    from app.schemas import AgentRequest
    from app.services import agent_graph
    from app.services.conversation_state import load_conversation_state

    agent_request = AgentRequest(
        question=request.query,
        knowledge_base_id=request.knowledge_base_id,
        session_id=request.session_id,
        filters=request.filters,
        top_k=request.top_k,
        history=[],
        stream_trace=False,
    )
    session, run = agent_graph.create_agent_run_context(
        db,
        agent_request,
        persist_user_turn=False,
    )
    run.status = "running"
    run.current_node = "intent_planning"
    run.started_at = datetime.utcnow()
    db.commit()
    history_summary, _history_audit = history_summary_projection(
        [item.model_dump() for item in agent_request.history],
        max_characters=get_settings().agent_history_summary_max_chars,
    )
    capabilities, _provisional_context_state = retrieval_capability_snapshot(
        db,
        run.knowledge_base_id,
        admit_graph=False,
    )
    plan, planning_audit = await plan_intent_execution(
        db,
        run=run,
        question=agent_request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(agent_request.filters.model_dump(mode="json")),
        history_summary=history_summary,
        capabilities=capabilities,
        provider_factory=agent_graph.ChatProvider,
    )
    if plan.strategy.route in {"system_capability", "clarify"}:
        run.status = "needs_clarification" if plan.strategy.route == "clarify" else "completed"
        run.current_node = None
        run.completed_at = datetime.utcnow()
        run.route = plan.strategy.route
        db.commit()
        _session, conversation = load_conversation_state(
            db,
            knowledge_base_id=run.knowledge_base_id,
            session_id=session.id,
        )
        return {
            "query": request.query,
            "results": [],
            "degraded_mode": agent_graph.is_degraded_mode(),
            "model_audit": {},
            "retrieval_trace_id": None,
            "context_package_id": None,
            "run_id": run.id,
            "entry_layer": None,
            "intent": plan.intent.model_dump(mode="json"),
            "execution_strategy": plan.strategy.model_dump(mode="json"),
            "terminal_outcome": "scope_ambiguous" if plan.strategy.route == "clarify" else "completed",
            "accepted_plan_hash": plan.identity,
            "conversation_state": conversation.public_payload(),
        }
    admitted_capabilities, admitted_context_state = retrieval_capability_snapshot(
        db,
        run.knowledge_base_id,
        admit_graph=True,
    )
    if admitted_capabilities.identity != plan.capability_hash:
        raise ValueError("strategy_capability_identity_changed")
    execution = await execute_layered_retrieval(
        db,
        plan=plan,
        filters=agent_request.filters,
        top_k=agent_request.top_k,
        capabilities=admitted_capabilities,
        context_state=admitted_context_state,
    )
    package = None
    admission = None
    terminal_outcome = "insufficient_evidence"
    if execution.results or execution.scope_target_chunk_ids:
        envelope = agent_operating_envelope()
        package = await agent_graph.run_bounded_source_io(
            _build_intent_context_package,
            db,
            request=agent_request,
            plan=plan,
            execution=execution,
            token_budget=int(envelope["context_package_token_budget"]),
        )
        admission = await agent_graph.run_bounded_source_io(
            admit_context_package,
            db,
            run=run,
            plan=plan,
            package=package,
            filters=agent_request.filters,
            remaining_seconds=get_settings().retrieval_total_timeout_seconds,
        )
        terminal_outcome = "completed" if admission.passed else admission.outcome
        if admission.outcome == "technical_failure":
            raise ValueError("source_integrity_admission_failed")
    run.status = "needs_clarification" if terminal_outcome == "scope_ambiguous" else "completed"
    run.current_node = None
    run.completed_at = datetime.utcnow()
    run.route = PROTOCOL
    run.metadata_json = {
        **dict(run.metadata_json or {}),
        "terminal_outcome": terminal_outcome,
        "search_only": True,
        "policy_update_eligible": False,
        "intent_retrieval_cache": execution.cache_audit,
    }
    flag_modified(run, "metadata_json")
    db.commit()
    publish_intent_retrieval_cache(execution)
    _session, conversation = load_conversation_state(
        db,
        knowledge_base_id=run.knowledge_base_id,
        session_id=session.id,
    )
    return {
        "query": request.query,
        "results": execution.results if admission is not None and admission.passed else [],
        "degraded_mode": agent_graph.is_degraded_mode(),
        "model_audit": {"intent_retrieval_cache": execution.cache_audit},
        "retrieval_trace_id": execution.trace.id,
        "context_package_id": package.id if package is not None else None,
        "run_id": run.id,
        "entry_layer": plan.strategy.entry_layer,
        "intent": plan.intent.model_dump(mode="json"),
        "execution_strategy": plan.strategy.model_dump(mode="json"),
        "terminal_outcome": terminal_outcome,
        "accepted_plan_hash": plan.identity,
        "conversation_state": conversation.public_payload(),
    }
