import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.models import (
    AgentObservation,
    AgentTraceEvent,
    AnswerSession,
    ContextPackage,
    RetrievalLexicalReward,
    RetrievalTrace,
)
from app.schemas import (
    AgentRequest,
    QAResponse,
    SearchRequest,
    SearchResponse,
    public_search_result_payload,
)
from app.services import agent_graph
from app.services.intent_execution_agent import execute_intent_search


@pytest.mark.asyncio
async def test_prefetched_capability_is_rechecked_before_retrieval(monkeypatch):
    import asyncio
    from app.services import intent_execution_agent

    manifest = SimpleNamespace(identity="unit-test-capability")
    state = SimpleNamespace(id="unit-test-state")
    plan = SimpleNamespace(capability_hash=manifest.identity)
    monkeypatch.setattr(
        intent_execution_agent,
        "retrieval_capability_snapshot",
        lambda _db, _kb, *, admit_graph: (manifest, state),
    )
    task = asyncio.create_task(asyncio.sleep(0, result=(manifest, state.id)))
    result = await intent_execution_agent._admitted_capabilities_after_plan(
        object(), knowledge_base_id="unit-test-kb", plan=plan, task=task,
    )
    assert result == (manifest, state)

    drifted = asyncio.create_task(asyncio.sleep(0, result=(manifest, "unit-test-old-state")))
    with pytest.raises(ValueError, match="strategy_capability_identity_changed"):
        await intent_execution_agent._admitted_capabilities_after_plan(
            object(), knowledge_base_id="unit-test-kb", plan=plan, task=drifted,
        )


def test_public_search_response_rejects_duplicate_chunk_results():
    repeated = {"chunk_id": "unit-test-chunk", "score": 1.0}

    with pytest.raises(ValueError, match="unique chunk_id"):
        SearchResponse(query="unit-test query", results=[repeated, repeated])


@pytest.mark.asyncio
async def test_system_capability_uses_one_plan_and_zero_retrieval_or_sources(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
    monkeypatch,
):
    from agent_test_support import Admission

    calls = []

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            assert "INTENT EXECUTION RETRIEVAL V1" in system_prompt
            calls.append("plan")
            return {
                "intent": {"primary": "system_capability"},
                "execution_strategy": {
                    "route": "system_capability",
                    "entry_layer": None,
                    "generate_lexical": False,
                    "hybrid": False,
                    "reason_code": "system_request",
                },
            }

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=sample_knowledge_base.id,
            question="你能做什么？",
        ),
        admission=Admission(),
    )
    assert calls == ["plan"]
    assert response["direct_answer_mode"] == "system_capability"
    assert response["terminal_outcome"] == "completed"
    assert not response["citations"]
    assert response["context_package_id"] is None
    assert response["retrieval_trace_id"] is None
    assert db_session.scalar(select(func.count()).select_from(RetrievalTrace)) == 0
    assert db_session.scalar(select(func.count()).select_from(ContextPackage)) == 0
    answer = db_session.scalar(select(AnswerSession))
    assert answer.retrieval_trace_id is None and answer.context_package_id is None
    plan = db_session.scalar(
        select(AgentObservation).where(
            AgentObservation.observation_type == "intent_execution_plan"
        )
    )
    assert plan.verdict == "completed"
    assert QAResponse.model_validate(response).contract_version == "qa_public_v2"


@pytest.mark.asyncio
async def test_retrieval_path_has_one_plan_finite_evidence_selection_one_generation_and_no_online_reward(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import layered_execution_v1
    from test_intent_contracts import proposal
    from agent_test_support import Admission

    calls = []

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                calls.append("plan")
                return proposal(layer="chunk")
            if "EVIDENCE TOOL SESSION V1" in system_prompt:
                calls.append("evidence")
            return await super().classify_json(system_prompt, user_prompt, fallback)

        async def complete_text(self, system_prompt, user_prompt, *, max_tokens):
            calls.append("answer")
            assert "SINGLE GROUNDED MARKDOWN ANSWER V3" in system_prompt
            assert "GitHub-Flavored Markdown" in system_prompt
            assert "$...$" in system_prompt and "$$...$$" in system_prompt
            packet = json.loads(user_prompt)
            source = packet["evidence"][0]
            return f"{source['text']}⟦cite:{source['source_handle']}⟧"

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=populated_context_graph["knowledge_base"].id,
            question="Summarize the topics.",
            top_k=4,
        ),
        admission=Admission(),
    )
    assert calls == ["plan", "answer"]
    assert response["route"] == "intent_execution_retrieval_v1"
    assert response["entry_layer"] == "chunk"
    assert response["terminal_outcome"] == "completed"
    assert response["citations"]
    observations = list(db_session.scalars(select(AgentObservation)))
    assert [row.verdict for row in observations if row.observation_type == "intent_execution_plan"] == ["completed"]
    assert [row.verdict for row in observations if row.observation_type == "source_integrity_admission"] == ["passed"]
    assert [row.verdict for row in observations if row.observation_type == "evidence_read_loop"] == ["finalized"]
    loop = next(row for row in observations if row.observation_type == "evidence_read_loop")
    assert loop.observation_json["state"]["deterministic_direct"] is True
    assert loop.observation_json["state"]["decision_call_count"] == 0
    assert [row.verdict for row in observations if row.observation_type == "single_grounded_generation"] == ["completed"]
    assert not [row for row in observations if row.observation_type in {"retrieval_sufficiency", "retrieval_lexical_patch"}]
    assert db_session.scalar(select(func.count()).select_from(RetrievalLexicalReward)) == 0
    trace = db_session.get(RetrievalTrace, response["retrieval_trace_id"])
    assert trace.retrieval_mode == "intent_execution_retrieval_v1"
    assert trace.scores_json["post_retrieval_model_call_count"] == 0
    assert QAResponse.model_validate(response).contract_version == "qa_public_v2"


@pytest.mark.asyncio
async def test_generation_shape_failure_persists_content_free_diagnostics(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.models import AgentRun
    from app.services import layered_execution_v1
    from app.services.reflection_models import AnswerReviewModelError
    from test_intent_contracts import proposal
    from agent_test_support import Admission

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                return proposal(layer="chunk")
            return await super().classify_json(system_prompt, user_prompt, fallback)

        async def complete_text(self, system_prompt, user_prompt, *, max_tokens):
            return ""

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    with pytest.raises(
        AnswerReviewModelError,
        match="generation_answer_stream_invalid",
    ):
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                knowledge_base_id=populated_context_graph["knowledge_base"].id,
                question="Summarize the topics.",
                top_k=4,
            ),
            admission=Admission(),
        )

    run = db_session.scalar(
        select(AgentRun).order_by(AgentRun.created_at.desc())
    )
    failure = run.metadata_json["technical_failure"]
    assert run.status == "failed"
    assert failure["stage"] == "generation"
    assert failure["code"] == "answer_stream_invalid"
    assert failure["provider_shape"] == {
        "error_code": "answer_stream_empty",
        "provider_values_persisted": False,
    }


@pytest.mark.asyncio
async def test_failed_source_admission_terminal_is_not_persisted_as_reusable_evidence(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import intent_execution_agent, layered_execution_v1
    from test_intent_contracts import proposal
    from agent_test_support import Admission

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            assert "INTENT EXECUTION RETRIEVAL V1" in system_prompt
            return proposal(layer="chunk")

    def reject_package(*_args, **_kwargs):
        return SimpleNamespace(
            passed=False,
            outcome="insufficient_evidence",
            audit={
                "protocol_version": "source_integrity_admission_v1",
                "audit_hash": "a" * 64,
            },
        )

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    monkeypatch.setattr(intent_execution_agent, "admit_context_package", reject_package)
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=populated_context_graph["knowledge_base"].id,
            question="Summarize the topics.",
            top_k=4,
        ),
        admission=Admission(),
    )

    answer = db_session.get(AnswerSession, response["answer_session_id"])
    assert response["terminal_outcome"] == "insufficient_evidence"
    assert response["retrieval_trace_id"] is not None
    assert response["context_package_id"] is not None
    assert answer.retrieval_trace_id is None
    assert answer.context_package_id is None
    assert response["conversation_state"]["history_references"] == []
    assert QAResponse.model_validate(response).terminal_outcome == "insufficient_evidence"


@pytest.mark.asyncio
async def test_search_uses_the_same_plan_and_source_admission_without_generation(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import layered_execution_v1
    from test_intent_contracts import proposal

    calls = []

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            assert "INTENT EXECUTION RETRIEVAL V1" in system_prompt
            calls.append("plan")
            return proposal(layer="mid")

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    payload = await execute_intent_search(
        db_session,
        SearchRequest(
            knowledge_base_id=populated_context_graph["knowledge_base"].id,
            query="Summarize the topics.",
            top_k=4,
        ),
    )
    assert calls == ["plan"]
    assert payload["entry_layer"] == "mid"
    assert payload["terminal_outcome"] == "completed"
    assert payload["results"]
    assert [event.node for event in db_session.scalars(
        select(AgentTraceEvent).where(AgentTraceEvent.run_id == payload["run_id"]).order_by(AgentTraceEvent.sequence_index)
    )][:1] == ["intent_planning"]
    assert db_session.scalar(select(func.count()).select_from(AnswerSession)) == 0
    assert not list(
        db_session.scalars(
            select(AgentObservation).where(
                AgentObservation.observation_type == "single_grounded_generation"
            )
        )
    )
    public = {
        **payload,
        "results": [public_search_result_payload(item) for item in payload["results"]],
    }
    assert SearchResponse.model_validate(public).contract_version == "search_public_v2"


@pytest.mark.asyncio
async def test_verified_context_reuse_replays_same_session_sources_without_new_retrieval(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import layered_execution_v1
    from test_intent_contracts import proposal
    from agent_test_support import Admission

    planning_calls = 0
    answer_calls = 0

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            nonlocal planning_calls, answer_calls
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                planning_calls += 1
                raw = proposal(layer="chunk")
                if planning_calls == 2:
                    raw["execution_strategy"]["route"] = "verified_context_reuse"
                return raw
            return await super().classify_json(system_prompt, user_prompt, fallback)

        async def complete_text(self, system_prompt, user_prompt, *, max_tokens):
            nonlocal answer_calls
            answer_calls += 1
            packet = json.loads(user_prompt)
            source = packet["evidence"][0]
            return f"{source['text']}⟦cite:{source['source_handle']}⟧"

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    kb_id = populated_context_graph["knowledge_base"].id
    first = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=kb_id,
            question="Summarize the topics.",
            top_k=4,
        ),
        admission=Admission(),
    )
    from app.services.conversation_state import load_conversation_state
    _session, conversation = load_conversation_state(
        db_session,
        knowledge_base_id=kb_id,
        session_id=first["session_id"],
    )
    reference = conversation.history_references[-1]
    from app.services.retrieval_answer_record import replay_answer_bindings
    from app.services.retrieval import get_context_package
    replay_answer_bindings(
        db_session,
        answer=db_session.get(AnswerSession, reference["answer_session_id"]),
        package=db_session.get(ContextPackage, reference["context_package_id"]),
    )
    assert get_context_package(db_session, reference["context_package_id"]) is not None
    assert agent_graph._latest_verified_context_reuse_candidate(
        db_session,
        knowledge_base_id=kb_id,
        conversation_planner_context={
            "history_references": conversation.history_references,
        },
    ) is not None
    trace_count = db_session.scalar(select(func.count()).select_from(RetrievalTrace))
    second = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=kb_id,
            session_id=first["session_id"],
            question="Explain that material again.",
            top_k=4,
        ),
        admission=Admission(),
    )
    assert planning_calls == 2 and answer_calls == 2
    assert second["direct_answer_mode"] == "verified_context_reuse"
    assert second["context_package_id"] == first["context_package_id"]
    assert second["retrieval_trace_id"] == first["retrieval_trace_id"]
    assert db_session.scalar(select(func.count()).select_from(RetrievalTrace)) == trace_count
    assert QAResponse.model_validate(second).terminal_outcome == "completed"


@pytest.mark.asyncio
async def test_grounded_sse_visible_text_is_the_persisted_answer_and_citations_follow(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import layered_execution_v1
    from agent_test_support import Admission
    from test_intent_contracts import proposal

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                return proposal(layer="chunk")
            return await super().classify_json(system_prompt, user_prompt, fallback)

        async def complete_text_streaming(
            self,
            system_prompt,
            user_prompt,
            *,
            max_tokens,
            on_text_delta,
        ):
            packet = json.loads(user_prompt)
            assert len(packet["evidence"]) >= 2
            raw = (
                "## Result\n\nFirst grounded point.⟦cite:src_1⟧\n\n"
                "- Second grounded point.⟦cite:src_2⟧\n\n"
                "Malformed stays visible: ⟦cite:src_999⟧"
            )
            for start in range(0, len(raw), 4):
                await on_text_delta(raw[start : start + 4])
            return raw

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    events = [
        event
        async for event in agent_graph.stream_agent_events(
            AgentRequest(
                knowledge_base_id=populated_context_graph["knowledge_base"].id,
                question="Summarize the topics.",
                top_k=4,
                stream_trace=True,
            ),
            admission=Admission(),
        )
    ]

    visible = "".join(
        event["token"] for event in events if event["type"] == "token"
    )
    final = next(event["response"] for event in events if event["type"] == "final")
    citation_index = next(
        index for index, event in enumerate(events) if event["type"] == "citations"
    )
    last_token_index = max(
        index for index, event in enumerate(events) if event["type"] == "token"
    )
    persisted = db_session.scalar(
        select(AnswerSession).order_by(AnswerSession.created_at.desc())
    )

    assert visible == final["answer"] == persisted.answer
    assert visible == (
        "## Result\n\nFirst grounded point.[1](#source-1)\n\n"
        "- Second grounded point.[2](#source-2)\n\n"
        "Malformed stays visible: ⟦cite:src_999⟧"
    )
    assert not [event for event in events if event["type"] == "answer_replace"]
    assert citation_index > last_token_index
    assert final["citations"]
    expected_source_count = persisted.model_json["generation"]["model_audit"][
        "answer_stream"
    ]["source_handle_count"]
    assert len(final["citations"]) == expected_source_count
    assert persisted.model_json["source_binding_count"] == expected_source_count
    assert len(persisted.diagnostics_json["answer_units"]) == 1
    assert len(
        persisted.diagnostics_json["answer_units"][0]["source_handles"]
    ) == expected_source_count
    assert persisted.model_json["generation"]["model_audit"]["answer_stream"][
        "replace_count"
    ] == 0
    assert persisted.model_json["generation"]["model_audit"]["answer_stream"][
        "citation_marker_count"
    ] == 2
    assert persisted.model_json["generation"]["model_audit"]["answer_stream"][
        "invalid_marker_count"
    ] == 1


@pytest.mark.asyncio
async def test_sse_and_sync_share_the_same_terminal_response_contract(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
    monkeypatch,
):
    from agent_test_support import Admission

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            assert "INTENT EXECUTION RETRIEVAL V1" in system_prompt
            return {
                "intent": {"primary": "system_capability"},
                "execution_strategy": {
                    "route": "system_capability",
                    "entry_layer": None,
                    "generate_lexical": False,
                    "hybrid": False,
                    "reason_code": "system_request",
                },
            }

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    events = [
        event
        async for event in agent_graph.stream_agent_events(
            AgentRequest(
                knowledge_base_id=sample_knowledge_base.id,
                question="What can you do?",
                stream_trace=True,
            ),
            admission=Admission(),
        )
    ]
    assert not [event for event in events if event["type"] == "error"]
    finals = [event["response"] for event in events if event["type"] == "final"]
    assert len(finals) == 1
    final = QAResponse.model_validate(finals[0])
    assert final.terminal_outcome == "completed"
    assert final.direct_answer_mode == "system_capability"
    terminal_meta = [event for event in events if event["type"] == "meta"][-1]
    assert terminal_meta["terminal_outcome"] == final.terminal_outcome
    assert terminal_meta["entry_layer"] == final.entry_layer
