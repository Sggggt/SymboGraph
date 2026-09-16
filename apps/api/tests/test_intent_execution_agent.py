import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.models import (
    AgentObservation,
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
async def test_retrieval_path_has_one_plan_one_generation_and_no_online_reward_or_sufficiency(
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
            if "SINGLE GROUNDED ANSWER V2" in system_prompt:
                calls.append("answer")
                assert "GitHub-Flavored Markdown" in system_prompt
                assert "$...$" in system_prompt and "$$...$$" in system_prompt
                packet = json.loads(user_prompt)
                source = packet["evidence"][0]
                return {
                    "answer_units": [
                        {
                            "kind": "factual",
                            "text": source["text"],
                            "source_handles": [source["source_handle"]],
                        }
                    ]
                }
            raise AssertionError("unexpected model call")

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
    from app.services.embeddings import ProviderJSONShapeError
    from app.services.reflection_models import AnswerReviewModelError
    from test_intent_contracts import proposal
    from agent_test_support import Admission

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                return proposal(layer="chunk")
            raise ProviderJSONShapeError(
                {
                    "error_code": "json_decode_error",
                    "field_path": "$",
                    "utf8_bytes": 127,
                    "sha256": "f" * 64,
                    "starts_with_object": True,
                    "ends_with_object": False,
                    "contains_code_fence": False,
                }
            )

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    with pytest.raises(AnswerReviewModelError, match="generation_schema_invalid"):
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
    assert failure["code"] == "schema_invalid"
    assert failure["provider_shape"] == {
        "error_code": "json_decode_error",
        "field_path": "$",
        "utf8_bytes": 127,
        "starts_with_object": True,
        "ends_with_object": False,
        "contains_code_fence": False,
    }
    assert "sha256" not in json.dumps(failure)


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
            if "SINGLE GROUNDED ANSWER V2" in system_prompt:
                answer_calls += 1
                packet = json.loads(user_prompt)
                source = packet["evidence"][0]
                return {
                    "answer_units": [
                        {
                            "kind": "factual",
                            "text": source["text"],
                            "source_handles": [source["source_handle"]],
                        }
                    ]
                }
            raise AssertionError("unexpected model call")

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


