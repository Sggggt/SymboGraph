from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError


def _create_session(db_session, knowledge_base_id: str):
    from app.models import QASession
    from app.services.conversation_state import initialize_new_session_state

    session = QASession(
        knowledge_base_id=knowledge_base_id,
        title="Conversation state test",
        transcript=[],
    )
    db_session.add(session)
    initialize_new_session_state(db_session, session)
    db_session.commit()
    db_session.refresh(session)
    return session


def _persist_answer_reference(
    db_session,
    *,
    session,
    question: str = "What is grounded?",
    answer_text: str = "A grounded answer.",
):
    from app.models import (
        AgentRun,
        AnswerSession,
        CitationVerification,
        ContextPackage,
        RetrievalTrace,
    )
    from app.services.conversation_state import load_conversation_state

    _row, before_turn = load_conversation_state(
        db_session,
        knowledge_base_id=session.knowledge_base_id,
        session_id=session.id,
    )
    run = AgentRun(
        knowledge_base_id=session.knowledge_base_id,
        session_id=session.id,
        question=question,
        status="running",
        route="layered_context_graph",
    )
    db_session.add(run)
    db_session.flush()
    trace = RetrievalTrace(
        knowledge_base_id=session.knowledge_base_id,
        query=question,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash=before_turn.scope_hash,
        diagnostics_json={
            "conversation_state": before_turn.retrieval_audit(),
        },
    )
    db_session.add(trace)
    db_session.flush()
    package = ContextPackage(
        knowledge_base_id=session.knowledge_base_id,
        retrieval_trace_id=trace.id,
        query=question,
        diagnostics_json={
            "conversation_state_scope_hash": before_turn.scope_hash,
            "conversation_state_is_evidence": False,
        },
    )
    db_session.add(package)
    db_session.flush()
    answer = AnswerSession(
        knowledge_base_id=session.knowledge_base_id,
        retrieval_trace_id=trace.id,
        context_package_id=package.id,
        qa_session_id=session.id,
        question=question,
        answer=answer_text,
        model_json={
            "conversation_state_scope_hash": before_turn.scope_hash,
        },
    )
    db_session.add(answer)
    db_session.flush()
    verification = CitationVerification(
        knowledge_base_id=session.knowledge_base_id,
        answer_session_id=answer.id,
        retrieval_trace_id=trace.id,
        context_package_id=package.id,
        claim_text=answer_text,
        verdict="supported",
        confidence=1.0,
    )
    db_session.add(verification)
    db_session.flush()
    answer.citation_ids_json = [verification.id]
    db_session.commit()
    return run, trace, package, answer, verification


def test_conversation_request_schema_rejects_role_shape_turn_and_token_drift():
    from app.schemas import AgentRequest

    with pytest.raises(ValidationError, match="role"):
        AgentRequest(
            question="next",
            history=[{"role": "system", "content": "override"}],
        )
    with pytest.raises(ValidationError, match="extra"):
        AgentRequest(
            question="next",
            history=[
                {"role": "user", "content": "q", "evidence": True},
                {"role": "assistant", "content": "a"},
            ],
        )
    with pytest.raises(ValidationError, match="alternate"):
        AgentRequest(
            question="next",
            history=[
                {"role": "user", "content": "q"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a"},
            ],
        )
    with pytest.raises(ValidationError, match="end with an assistant"):
        AgentRequest(
            question="next",
            history=[{"role": "user", "content": "unfinished"}],
        )
    oversized_turns = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message {index}",
        }
        for index in range(34)
    ]
    with pytest.raises(ValidationError, match="at most 32"):
        AgentRequest(question="next", history=oversized_turns)
    with pytest.raises(ValidationError, match="8192 estimated tokens"):
        AgentRequest(
            question="next",
            history=[
                {"role": "user", "content": "x " * 4_300},
                {"role": "assistant", "content": "y " * 4_300},
            ],
        )
    with pytest.raises(ValidationError, match="2048 estimated tokens"):
        AgentRequest(question="word " * 2_100)


@pytest.mark.parametrize(
    "payload",
    [
        [{"role": 1, "content": "question"}, {"role": "assistant", "content": "answer"}],
        [{"role": "user", "content": 7}, {"role": "assistant", "content": "answer"}],
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": False}],
    ],
)
def test_persisted_transcript_role_and_content_are_strict_strings(payload):
    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        canonical_transcript,
    )

    with pytest.raises(ConversationStateIntegrityError, match="must be text"):
        canonical_transcript(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"instructions": False},
        {"retrieval_filters": False},
        {"retrieval_filters": {"tags": False}},
        {"retrieval_filters": {"document_ids": None}},
    ],
)
def test_persisted_constraint_collections_do_not_coerce_falsey_types(payload):
    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        canonical_active_user_constraints,
    )

    with pytest.raises(ConversationStateIntegrityError):
        canonical_active_user_constraints(payload)


def test_persisted_task_status_does_not_coerce_false_to_active():
    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        canonical_task_state,
    )

    with pytest.raises(ConversationStateIntegrityError, match="status must be text"):
        canonical_task_state({"status": False})


def test_persisted_transcript_type_tamper_cannot_replay_unchanged_hash(
    db_session,
    sample_knowledge_base,
):
    from sqlalchemy.orm.attributes import flag_modified

    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        conversation_state_hash_for_session,
        load_conversation_state,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    session.transcript = [
        {"role": "user", "content": "7"},
        {"role": "assistant", "content": "answer"},
    ]
    session.conversation_state_hash = conversation_state_hash_for_session(session)
    db_session.commit()
    unchanged_hash = session.conversation_state_hash

    session.transcript[0]["content"] = 7
    flag_modified(session, "transcript")
    db_session.commit()
    assert session.conversation_state_hash == unchanged_hash

    with pytest.raises(ConversationStateIntegrityError, match="content must be text"):
        load_conversation_state(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            session_id=session.id,
        )


@pytest.mark.parametrize("turn_index", ["0", False, 0.0, None])
def test_persisted_history_turn_index_is_a_strict_non_bool_integer(turn_index):
    from app.services.conversation_state import (
        CONVERSATION_REFERENCE_PROTOCOL_VERSION,
        ConversationStateIntegrityError,
        canonical_history_references,
    )

    with pytest.raises(
        ConversationStateIntegrityError,
        match="contiguous stored order",
    ):
        canonical_history_references(
            [
                {
                    "protocol_version": CONVERSATION_REFERENCE_PROTOCOL_VERSION,
                    "turn_index": turn_index,
                    "run_id": str(uuid4()),
                    "answer_session_id": str(uuid4()),
                    "context_package_id": str(uuid4()),
                    "retrieval_trace_id": str(uuid4()),
                    "citation_verification_ids": [],
                }
            ]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("citation_verification_ids", False),
        ("citation_verification_ids", [123]),
        ("run_id", 11111111111141118111111111111111),
    ],
)
def test_persisted_history_reference_identifiers_do_not_coerce_types(field, value):
    from app.services.conversation_state import (
        CONVERSATION_REFERENCE_PROTOCOL_VERSION,
        ConversationStateIntegrityError,
        canonical_history_references,
    )

    payload = {
        "protocol_version": CONVERSATION_REFERENCE_PROTOCOL_VERSION,
        "turn_index": 0,
        "run_id": str(uuid4()),
        "answer_session_id": str(uuid4()),
        "context_package_id": str(uuid4()),
        "retrieval_trace_id": str(uuid4()),
        "citation_verification_ids": [],
    }
    payload[field] = value
    with pytest.raises(ConversationStateIntegrityError, match="UUID strings|UUID string"):
        canonical_history_references([payload])


def test_canonical_constraints_uuid_order_and_scope_are_deterministic():
    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        canonical_active_user_constraints,
        conversation_state_scope_hash,
    )

    first_id = str(uuid4())
    second_id = str(uuid4())
    first = canonical_active_user_constraints(
        {
            "instructions": ["second", "first", "first"],
            "retrieval_filters": {
                "document_ids": [second_id.upper(), first_id],
                "tags": ["b", "a", "a"],
            },
        }
    )
    second = canonical_active_user_constraints(
        {
            "retrieval_filters": {
                "tags": ["a", "b"],
                "document_ids": [first_id, second_id],
            },
            "instructions": ["first", "second"],
        }
    )
    assert first == second
    assert first["retrieval_filters"]["document_ids"] == sorted(
        [first_id, second_id]
    )
    state_hash = "a" * 64
    knowledge_base_id = str(uuid4())
    session_id = str(uuid4())
    assert conversation_state_scope_hash(
        knowledge_base_id=knowledge_base_id,
        qa_session_id=session_id,
        state_hash=state_hash,
    ) == conversation_state_scope_hash(
        knowledge_base_id=knowledge_base_id.upper(),
        qa_session_id=session_id.upper(),
        state_hash=state_hash,
    )
    with pytest.raises(ConversationStateIntegrityError, match="canonical UUID"):
        canonical_active_user_constraints(
            {"retrieval_filters": {"document_ids": ["not-a-uuid"]}}
        )


def test_full_transcript_is_retained_while_prompt_projection_is_bounded(
    db_session,
    sample_knowledge_base,
):
    from app.services.conversation_state import (
        PROMPT_HISTORY_MAX_TURNS,
        conversation_state_hash_for_session,
        load_conversation_state,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    transcript = []
    for index in range(30):
        transcript.extend(
            [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )
    session.transcript = transcript
    session.conversation_state_revision = 7
    session.conversation_state_hash = conversation_state_hash_for_session(session)
    db_session.commit()

    _row, snapshot = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=session.id,
    )
    assert snapshot.transcript_message_count == 60
    assert len(session.transcript) == 60
    assert len(snapshot.prompt_history) == PROMPT_HISTORY_MAX_TURNS
    assert snapshot.prompt_history[0]["role"] == "user"
    assert snapshot.prompt_history[-1]["role"] == "assistant"
    assert snapshot.prompt_history_audit["persisted_transcript_retained_in_full"] is True
    assert snapshot.prompt_history_audit["transcript_truncated"] is True
    original_hash = snapshot.state_hash
    original_scope = snapshot.scope_hash
    session.conversation_state_revision = 8
    db_session.commit()
    _row, same_facts = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=session.id,
    )
    assert same_facts.state_hash == original_hash
    assert same_facts.scope_hash == original_scope

    other_session = _create_session(db_session, sample_knowledge_base.id)
    other_session.transcript = transcript
    other_session.conversation_state_hash = conversation_state_hash_for_session(
        other_session
    )
    db_session.commit()
    _row, other_snapshot = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=other_session.id,
    )
    assert other_snapshot.state_hash != original_hash
    assert other_snapshot.scope_hash != original_scope


def test_server_reload_merges_history_and_applies_active_filter_constraints(
    db_session,
    sample_knowledge_base,
):
    from app.schemas import ConversationStateUpdate, SearchFilters
    from app.services.conversation_state import (
        load_conversation_state,
        merge_search_filters_with_conversation_constraints,
        prepare_session_for_turn,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    document_id = str(uuid4())
    update = ConversationStateUpdate.model_validate(
        {
            "active_user_constraints": {
                "instructions": ["Prefer definitions before examples."],
                "retrieval_filters": {
                    "document_ids": [document_id],
                    "tags": ["verified"],
                },
            },
            "task_state": {
                "status": "active",
                "objective": "Explain the selected document.",
            },
        }
    )
    prepared = prepare_session_for_turn(
        db_session,
        session=session,
        history=[
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ],
        question="Next question",
        state_update=update,
    )
    db_session.commit()
    _row, reloaded = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=session.id,
    )
    assert reloaded.state_hash == prepared.state_hash
    assert reloaded.transcript_message_count == 2
    assert reloaded.active_user_constraints["instructions"] == [
        "Prefer definitions before examples."
    ]
    assert reloaded.task_state == {
        "status": "active",
        "objective": "Explain the selected document.",
        "current_step": "answering",
    }
    effective = merge_search_filters_with_conversation_constraints(
        SearchFilters(tags=["verified", "ignored"]),
        reloaded.active_user_constraints,
    )
    assert effective.document_ids == [document_id]
    assert effective.tags == ["verified"]


def test_session_reload_rejects_cross_kb_and_state_json_tampering(
    db_session,
    sample_knowledge_base,
):
    from app.models import KnowledgeBase
    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        ConversationStateNotFoundError,
        load_conversation_state,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    other = KnowledgeBase(
        name="Other conversation KB",
        description="isolation",
        source_root="other-conversation-kb",
    )
    db_session.add(other)
    db_session.commit()
    with pytest.raises(ConversationStateNotFoundError, match="does not belong"):
        load_conversation_state(
            db_session,
            knowledge_base_id=other.id,
            session_id=session.id,
        )

    session.task_state_json = {
        "status": "active",
        "objective": None,
        "current_step": None,
        "evidence_override": True,
    }
    db_session.commit()
    with pytest.raises(ConversationStateIntegrityError, match="unsupported fields"):
        load_conversation_state(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            session_id=session.id,
        )


def test_conflicting_turn_rolls_back_state_and_agent_run_atomically(
    db_session,
    sample_knowledge_base,
):
    from sqlalchemy import func, select

    from app.models import AgentRun
    from app.schemas import AgentRequest, ConversationStateUpdate, SearchFilters
    from app.services.agent_graph import create_agent_run_context
    from app.services.conversation_state import (
        ConversationStateConflictError,
        load_conversation_state,
        prepare_session_for_turn,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    allowed_document_id = str(uuid4())
    blocked_document_id = str(uuid4())
    prepare_session_for_turn(
        db_session,
        session=session,
        history=[],
        question="Initial task",
        state_update=ConversationStateUpdate.model_validate(
            {
                "active_user_constraints": {
                    "retrieval_filters": {
                        "document_ids": [allowed_document_id]
                    }
                },
                "task_state": {"objective": "Original objective"},
            }
        ),
    )
    db_session.commit()
    _row, before = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=session.id,
    )

    with pytest.raises(ConversationStateConflictError, match="document_ids"):
        create_agent_run_context(
            db_session,
            AgentRequest(
                knowledge_base_id=sample_knowledge_base.id,
                session_id=session.id,
                question="Conflicting turn",
                filters=SearchFilters(document_ids=[blocked_document_id]),
                conversation_state_update=ConversationStateUpdate.model_validate(
                    {"task_state": {"objective": "Must roll back"}}
                ),
            ),
        )
    db_session.rollback()
    _row, after = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=session.id,
    )
    assert after.state_hash == before.state_hash
    assert after.task_state == before.task_state
    assert db_session.scalar(
        select(func.count(AgentRun.id)).where(AgentRun.session_id == session.id)
    ) == 0


def test_answer_context_citation_references_are_durable_and_tamper_checked(
    db_session,
    sample_knowledge_base,
):
    from app.models import KnowledgeBase
    from app.schemas import ConversationStatePayload, SessionSummary
    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        append_completed_turn,
        load_conversation_state,
        session_summary_payload,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    run, trace, package, answer, verification = _persist_answer_reference(
        db_session,
        session=session,
    )
    completed = append_completed_turn(
        db_session,
        session_id=session.id,
        question=run.question,
        answer=answer.answer,
        run_id=run.id,
        citations=[],
        answer_session_id=answer.id,
    )
    assert completed.transcript_message_count == 2
    assert completed.retrieval_audit()["conversation_text_is_evidence"] is False
    assert completed.retrieval_audit()["gray_zone_decision_authority"] is False
    assert completed.retrieval_audit()["gray_zone_model_call_count"] == 0
    assert completed.history_references == [
        {
            "protocol_version": "answer_context_citation_reference_v1",
            "turn_index": 0,
            "run_id": run.id,
            "answer_session_id": answer.id,
            "context_package_id": package.id,
            "retrieval_trace_id": trace.id,
            "citation_verification_ids": [verification.id],
        }
    ]
    assert session.transcript[-1]["retrieval_trace_id"] == trace.id
    ConversationStatePayload.model_validate(completed.public_payload())
    summary = session_summary_payload(db_session, session)
    SessionSummary.model_validate(summary)
    assert summary["last_question"] == run.question
    assert summary["last_answer"] == answer.answer
    assert summary["conversation_state"]["history_references"] == completed.history_references

    other = KnowledgeBase(
        name="Tamper target KB",
        description="negative provenance",
        source_root="tamper-target-kb",
    )
    db_session.add(other)
    db_session.commit()
    package.knowledge_base_id = other.id
    db_session.commit()
    with pytest.raises(ConversationStateIntegrityError, match="provenance diverged"):
        load_conversation_state(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            session_id=session.id,
        )


def test_public_transcript_projects_latest_plan_trace_without_rewriting_history(
    db_session,
    sample_knowledge_base,
):
    from app.models import AgentPlan
    from app.services.conversation_state import (
        append_completed_turn,
        session_transcript_public_payload,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    run, trace, _package, answer, _verification = _persist_answer_reference(
        db_session,
        session=session,
    )
    append_completed_turn(
        db_session,
        session_id=session.id,
        question=run.question,
        answer=answer.answer,
        run_id=run.id,
        citations=[{"chunk_id": "legacy-only"}],
        task_status="waiting_user",
    )
    plan = AgentPlan(
        run_id=run.id,
        knowledge_base_id=sample_knowledge_base.id,
        retrieval_trace_id=trace.id,
        plan_index=1,
        status="planning_budget_exhausted",
    )
    db_session.add(plan)
    db_session.commit()

    assert "retrieval_trace_id" not in session.transcript[-1]
    projected = session_transcript_public_payload(db_session, session)
    assert projected[-1]["retrieval_trace_id"] == trace.id
    assert projected[-1]["citations"] == []
    assert projected[-1]["citation_replay_status"] == "unavailable"
    assert (
        projected[-1]["citation_replay_reason"]
        == "persisted_citation_contract_mismatch"
    )
    assert "retrieval_trace_id" not in session.transcript[-1]
    assert session.transcript[-1]["citations"] == [{"chunk_id": "legacy-only"}]


def test_persisted_history_turn_index_type_tamper_cannot_replay_unchanged_hash(
    db_session,
    sample_knowledge_base,
):
    from sqlalchemy.orm.attributes import flag_modified

    from app.services.conversation_state import (
        ConversationStateIntegrityError,
        append_completed_turn,
        load_conversation_state,
    )

    session = _create_session(db_session, sample_knowledge_base.id)
    run, _trace, _package, answer, _verification = _persist_answer_reference(
        db_session,
        session=session,
    )
    append_completed_turn(
        db_session,
        session_id=session.id,
        question=run.question,
        answer=answer.answer,
        run_id=run.id,
        citations=[],
        answer_session_id=answer.id,
    )
    db_session.commit()
    unchanged_hash = session.conversation_state_hash

    session.history_references_json[0]["turn_index"] = "0"
    flag_modified(session, "history_references_json")
    db_session.commit()
    assert session.conversation_state_hash == unchanged_hash

    with pytest.raises(
        ConversationStateIntegrityError,
        match="contiguous stored order",
    ):
        load_conversation_state(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            session_id=session.id,
        )


@pytest.mark.asyncio
async def test_ordinary_search_uses_session_scope_and_constraints(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.schemas import ConversationStateUpdate, SearchRequest
    from app.services.conversation_state import prepare_session_for_turn

    search_router = import_module("app.routers.search")
    session = _create_session(db_session, sample_knowledge_base.id)
    document_id = str(uuid4())
    prepare_session_for_turn(
        db_session,
        session=session,
        history=[],
        question="Scoped search setup",
        state_update=ConversationStateUpdate.model_validate(
            {
                "active_user_constraints": {
                    "retrieval_filters": {"document_ids": [document_id]}
                }
            }
        ),
    )
    db_session.commit()
    captured = {}

    async def fake_search(
        db,
        knowledge_base_id,
        query,
        filters,
        top_k,
        **kwargs,
    ):
        captured.update(
            {
                "knowledge_base_id": knowledge_base_id,
                "query": query,
                "filters": filters,
                **kwargs,
            }
        )
        return [], {
            "retrieval_trace_id": None,
            "context_package_id": None,
            "conversation_state_scope_hash": kwargs[
                "conversation_state_scope_hash"
            ],
        }

    monkeypatch.setattr(search_router, "search_chunks_with_audit", fake_search)
    response = await search_router.search(
        SearchRequest(
            query="Find the constrained facts",
            knowledge_base_id=sample_knowledge_base.id,
            session_id=session.id,
        ),
        db_session,
    )
    assert captured["filters"].document_ids == [document_id]
    assert captured["conversation_state_scope_hash"] == response[
        "conversation_state"
    ]["scope_hash"]
    assert captured["conversation_state_audit"][
        "conversation_text_is_evidence"
    ] is False
    assert captured["conversation_state_audit"][
        "gray_zone_model_call_count"
    ] == 0


@pytest.mark.asyncio
async def test_empty_graph_trace_and_cache_are_partitioned_by_conversation_scope(
    db_session,
    sample_knowledge_base,
):
    from app.schemas import SearchFilters
    from app.services.context_graph import (
        context_graph_cache_key_components,
        layered_search,
    )
    from app.services.conversation_state import load_conversation_state

    first_session = _create_session(db_session, sample_knowledge_base.id)
    second_session = _create_session(db_session, sample_knowledge_base.id)
    _row, first = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=first_session.id,
    )
    _row, second = load_conversation_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        session_id=second_session.id,
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        context_graph_cache_key_components(
            knowledge_base_id=sample_knowledge_base.id,
            query="same query",
            filters=SearchFilters(),
            context_state=None,
            retrieval_mode="layered_context_graph",
            conversation_state_scope_hash="fixed-none",
        )
    first_result = await layered_search(
        db_session,
        sample_knowledge_base.id,
        "same query",
        SearchFilters(),
        4,
        conversation_state_scope_hash=first.scope_hash,
        conversation_state_audit=first.retrieval_audit(),
    )
    second_result = await layered_search(
        db_session,
        sample_knowledge_base.id,
        "same query",
        SearchFilters(),
        4,
        conversation_state_scope_hash=second.scope_hash,
        conversation_state_audit=second.retrieval_audit(),
    )
    assert first_result.trace.conversation_state_scope_hash == first.scope_hash
    assert second_result.trace.conversation_state_scope_hash == second.scope_hash
    assert first_result.trace.diagnostics_json["cache_key"] != second_result.trace.diagnostics_json["cache_key"]
    assert first_result.trace.diagnostics_json["cache_key_components"][
        "conversation_state_scope_hash"
    ] == first.scope_hash
    assert first_result.trace.convergence_json["gray_zone_model_call_count"] == 0
    assert first_result.trace.diagnostics_json["conversation_state"][
        "gray_zone_decision_authority"
    ] is False
