from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_reuse_evaluator_output_budget_preserves_full_evidence(monkeypatch):
    import ast
    from app.services import agent_direct_answer as direct
    calls = []
    class Model:
        async def classify_json_bounded(self, **kwargs):
            calls.append(kwargs)
            return {"verdict": "sufficient", "reason": "The supplied source covers the question.",
                    "referenced_chunk_ids": ["unit-test-chunk"], "expected_answer_shape": "explanation"}
    monkeypatch.setattr(direct, "ChatProvider", Model)
    content = "A complete synthetic evidence passage. " * 90 + "Required fact at the end."
    result = await direct.evaluate_verified_context_reuse(question="Explain the required fact.", history=[],
        contexts=[{"chunk_id": "unit-test-chunk", "content": content}])
    assert result["verdict"] == "sufficient" and len(calls) == 1
    assert calls[0]["max_tokens"] == 8192
    packet = ast.literal_eval(calls[0]["user_prompt"])
    assert packet["verified_context_excerpts"][0]["excerpt"] == content
    assert packet["decision_controls"]["component_max_tokens"] == 8192
    assert "at most 240 characters" in calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_reuse_candidate_rejects_invalid_public_package_without_rewriting_it(db_session, populated_context_graph):
    from sqlalchemy import select
    from app.models import AnswerSession, ContextPackage
    from app.schemas import AgentRequest
    from app.services import agent_graph
    kb = populated_context_graph["knowledge_base"]
    first = await agent_graph.run_agent(db_session, AgentRequest(question="What is a Bayesian network?", knowledge_base_id=kb.id))
    answer = db_session.get(AnswerSession, first["answer_model_audit"]["answer_session_id"])
    package = db_session.get(ContextPackage, answer.context_package_id)
    reference = first["conversation_state"]["history_references"][-1]
    from app.services.retrieval_answer_record import replay_answer_bindings
    from app.services.retrieval import get_context_package
    replay_answer_bindings(db_session, answer=answer, package=package)
    assert get_context_package(db_session, package.id) is not None
    arguments = dict(knowledge_base_id=kb.id, conversation_planner_context={"history_references": [reference]})
    assert agent_graph._latest_verified_context_reuse_candidate(db_session, **arguments) is not None
    package.why_selected_json = {**package.why_selected_json, "unit-test-unpackaged": next(iter(package.why_selected_json.values()))}
    db_session.flush()
    assert agent_graph._latest_verified_context_reuse_candidate(db_session, **arguments) is None
    assert "unit-test-unpackaged" in db_session.get(ContextPackage, package.id).why_selected_json


@pytest.mark.asyncio
@pytest.mark.usefixtures('historical_answer_executor')
async def test_historical_planner_output_limit_keeps_actual_stage_and_disables_policy_updates(monkeypatch, db_session, populated_context_graph):
    from sqlalchemy import select, func
    from app.models import AgentRun, RewardEvent
    from app.schemas import AgentRequest
    from app.services import agent_graph
    from app.services.error_sanitizer import ExternalServiceError
    async def failed_plan(*args, **kwargs):
        raise ExternalServiceError(service="model_provider", phase="sdk_messages_completion", error_code="incomplete_max_tokens", retryable=False)
    monkeypatch.setattr(agent_graph, "propose_agent_plan", failed_plan)
    request = AgentRequest(question="Explain Bayesian network factorization.", knowledge_base_id=populated_context_graph["knowledge_base"].id)
    with pytest.raises(ExternalServiceError):
        await agent_graph.run_agent(db_session, request)
    run = db_session.scalar(select(AgentRun).where(AgentRun.question == request.question))
    failure = run.metadata_json["technical_failure"]
    assert failure["stage"] == "agent_planner" and failure["code"] == "output_limit"
    assert run.metadata_json["policy_update_eligible"] is False
    assert db_session.scalar(select(func.count()).select_from(RewardEvent)) == 0


def _direct_query_intent(question: str, kind: str) -> dict:
    return {
        "intent": "direct_answer",
        "direct_answer_kind": kind,
        "entities": [],
        "sub_queries": [question],
        "needs_graph": False,
        "suggested_strategy": "none",
        "history_turns": 0,
    }


def test_system_capability_card_and_validated_model_route_are_deterministic() -> None:
    from app.services.agent_direct_answer import (
        SYSTEM_CAPABILITY_CARD_PROTOCOL_VERSION,
        build_system_direct_answer,
        system_capability_card,
    )

    first_card = system_capability_card()
    second_card = system_capability_card()
    first = build_system_direct_answer(
        question="请介绍一下你自己",
        query_intent=_direct_query_intent("请介绍一下你自己", "identity"),
    )
    second = build_system_direct_answer(
        question="换个方式说说你是谁",
        query_intent=_direct_query_intent("换个方式说说你是谁", "identity"),
    )
    model_identity = build_system_direct_answer(
        question="方便说下底层采用哪类模型吗",
        query_intent=_direct_query_intent(
            "方便说下底层采用哪类模型吗",
            "model_identity",
        ),
    )

    assert first_card == second_card
    assert first_card["protocol_version"] == (
        SYSTEM_CAPABILITY_CARD_PROTOCOL_VERSION
    )
    assert len(first_card["card_hash"]) == 64
    assert first is not None and second is not None
    assert model_identity is not None
    assert model_identity["answer_kind"] == "model_identity"
    assert first["answer"] == second["answer"]
    assert first["decision"]["decision_hash"] == second["decision"][
        "decision_hash"
    ]
    assert first["decision"]["response_mode"] == "system_capability"
    assert first["decision"]["model_call_count"] == 1
    assert first["decision"]["tool_call_count"] == 0
    assert first["decision"]["policy_update_eligible"] is False
    assert first["decision"]["result_cache_enabled"] is False


def test_unified_question_perception_contract_is_closed_and_conflict_safe() -> None:
    from app.services.agent_intent import (
        validate_question_perception_output,
    )

    assert validate_question_perception_output(
        {
            "intent": "definition",
            "direct_answer_kind": "none",
            "entities": ["Bayesian network"],
            "sub_queries": ["What is a Bayesian network?"],
            "needs_graph": True,
            "suggested_strategy": "global_dense",
        },
        question="What is a Bayesian network?",
    )["intent"] == "definition"
    with pytest.raises(ValueError, match="not closed"):
        validate_question_perception_output(
            {
                "intent": "direct_answer",
                "direct_answer_kind": "identity",
                "entities": [],
                "sub_queries": ["Who are you?"],
                "needs_graph": False,
                "suggested_strategy": "none",
                "answer": "forged model answer",
            },
            question="Who are you?",
        )
    with pytest.raises(ValueError, match="non-direct"):
        validate_question_perception_output(
            {
                "intent": "definition",
                "direct_answer_kind": "identity",
                "entities": [],
                "sub_queries": ["Define agent identity in the document."],
                "needs_graph": True,
                "suggested_strategy": "global_dense",
            },
            question="Define agent identity in the document.",
        )
    with pytest.raises(ValueError, match="cannot request graph"):
        validate_question_perception_output(
            {
                "intent": "direct_answer",
                "direct_answer_kind": "identity",
                "entities": [],
                "sub_queries": ["Who are you?"],
                "needs_graph": True,
                "suggested_strategy": "local_graph",
            },
            question="Who are you?",
        )


def test_verified_context_reuse_output_contract_is_closed() -> None:
    from app.services.agent_direct_answer import (
        validate_verified_context_reuse_output,
    )

    valid = validate_verified_context_reuse_output(
        {
            "verdict": "sufficient",
            "reason": "the verified excerpt directly defines the requested term",
            "referenced_chunk_ids": ["chunk-1"],
            "expected_answer_shape": "definition",
        }
    )
    assert valid["verdict"] == "sufficient"

    with pytest.raises(ValueError, match="not closed"):
        validate_verified_context_reuse_output(
            {
                **valid,
                "continue_path": True,
            }
        )

    with pytest.raises(ValueError, match="unsupported"):
        validate_verified_context_reuse_output(
            {
                **valid,
                "verdict": "direct_answer",
            }
        )


def test_second_plan_receives_compact_observation_projection() -> None:
    from app.services import agent_graph

    raw_observation = {
        "bounded_graph_observation": {
            "plan_index": 0,
            "observation_hash": "a" * 64,
            "retrieval_granularity": "mid",
            "typed_action_control_hash": "b" * 64,
            "required_facets": [f"facet-{index}" for index in range(32)],
            "covered_facets": ["facet-0"],
            "evidence_roles": ["definition"],
            "result_chunk_ids": [f"chunk-{index}" for index in range(50)],
            "result_count": 50,
            "citable_span_count": 50,
            "independent_support_path_count": 4,
            "candidate_chunk_span_summaries": [
                {
                    "chunk_id": f"chunk-{index}",
                    "summary_hash": str(index) * 64,
                    "document_title": "document" * 80,
                    "text_excerpt": "evidence" * 2000,
                    "source_span_address": {
                        "chunk_id": f"chunk-{index}",
                        "private_nested_payload": "private" * 2000,
                    },
                }
                for index in range(50)
            ],
            "entry_counts": {"mid": 50},
            "stage_counts": {"frontier_pops": 500},
            "convergence": {"reason": "frontier_empty"},
            "hard_budget": {"effective_result_top_k": 50},
            "full_executor_diagnostics": "must-not-be-forwarded" * 2000,
        },
        "evidence_evaluator": {
            "verdict": "need_chunk_expansion",
            "reason": "missing a specific facet" * 100,
            "target_ids": [f"chunk-{index}" for index in range(50)],
            "expected_evidence": {
                "required_facets": [f"facet-{index}" for index in range(50)]
            },
        },
    }

    packet = agent_graph.planner_observation_projection_packet(
        [raw_observation]
    )
    serialized = str(packet)

    assert packet["protocol_version"] == (
        agent_graph.AGENT_PLANNER_OBSERVATION_PROTOCOL_VERSION
    )
    assert packet["serialized_char_count"] <= (
        agent_graph.AGENT_PLANNER_OBSERVATION_MAX_CHARS
    )
    assert packet["full_graph_observation_forwarded"] is False
    assert packet["raw_provider_response_forwarded"] is False
    assert len(packet["observations"][0]["candidate_span_summaries"]) == 4
    assert len(packet["observations"][0]["result_chunk_ids"]) == 16
    assert "full_executor_diagnostics" not in serialized
    assert "must-not-be-forwarded" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["你是谁？", "你是什么模型"])
async def test_system_capability_route_bypasses_graph_models_tools_and_policy(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
    local_agent_admission,
    question: str,
) -> None:
    from sqlalchemy import func, select

    from app.models import (
        AgentRun,
        AgentObservation,
        AnswerSession,
        CitationVerification,
        ContextPackage,
        QASession,
        RetrievalTrace,
        RewardEvent,
    )
    from app.schemas import AgentRequest, AgentResponse
    from app.services import agent_graph

    def forbidden_graph_admission(*_args, **_kwargs):
        raise AssertionError("system direct answer must not admit the graph")

    planning_calls = []

    class PlanningOnlyChatProvider:
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            assert "INTENT EXECUTION RETRIEVAL V1" in system_prompt
            planning_calls.append(question)
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

    monkeypatch.setattr(
        agent_graph,
        "active_graph_admission_gate",
        forbidden_graph_admission,
    )
    monkeypatch.setattr(agent_graph, "ChatProvider", PlanningOnlyChatProvider)

    result = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            question=question,
            knowledge_base_id=sample_knowledge_base.id,
        ),
    )
    response = AgentResponse.model_validate(result)

    assert planning_calls == [question]
    assert response.route == "system_capability"
    assert response.direct_answer_mode == "system_capability"
    assert response.citations == []
    assert response.used_chunks == []
    assert response.context_package_id is None
    assert response.retrieval_trace_id is None
    assert response.model_audit.answer_model_called is False
    assert response.model_audit.tool_call_count == 0
    assert response.model_audit.policy_update_eligible is False

    run = db_session.get(AgentRun, response.run_id)
    answer_session = db_session.scalar(
        select(AnswerSession).where(AnswerSession.qa_session_id == response.session_id)
    )
    qa_session = db_session.get(QASession, response.session_id)
    assert run is not None and run.route == "system_capability"
    assert (run.metadata_json or {}).get("direct_answer_mode") == (
        "system_capability"
    )
    assert answer_session is not None
    assert answer_session.context_package_id is None
    assert answer_session.retrieval_trace_id is None
    assert (answer_session.diagnostics_json or {}).get(
        "policy_update_eligible"
    ) is False
    assert qa_session is not None
    assert qa_session.history_references_json == []
    assert qa_session.transcript[-1]["route"] == "system_capability"
    assert qa_session.transcript[-1]["direct_answer_mode"] == (
        "system_capability"
    )
    plan = db_session.scalar(select(AgentObservation).where(
        AgentObservation.run_id == response.run_id,
        AgentObservation.observation_type == "intent_execution_plan",
    ))
    assert plan is not None and plan.verdict == "completed"
    assert answer_session.model_json["capability_card_hash"]

    for model in (
        RetrievalTrace,
        ContextPackage,
        CitationVerification,
        RewardEvent,
    ):
        assert db_session.scalar(select(func.count()).select_from(model)) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures('historical_answer_executor')
async def test_historical_verified_context_reuse_creates_new_source_binding_without_policy_update(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
    local_agent_admission,
) -> None:
    from sqlalchemy import func, select

    from app.models import AnswerSession, CitationVerification, ContextPackage, PolicyState, RewardEvent
    from app.schemas import AgentRequest, AgentResponse
    from app.services import agent_graph
    from app.services.chunking import stable_hash

    knowledge_base = populated_context_graph["knowledge_base"]
    first = AgentResponse.model_validate(
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                question="What is a Bayesian network?",
                knowledge_base_id=knowledge_base.id,
            ),
        )
    )
    assert first.route == "layered_context_graph"
    assert first.context_package_id is not None
    assert first.citations
    package = db_session.get(ContextPackage, first.context_package_id)
    assert package is not None and package.hit_chunk_ids_json
    rewards_before = db_session.scalar(select(func.count()).select_from(RewardEvent))
    policies_before = db_session.scalar(select(func.count()).select_from(PolicyState))
    verifications_before = db_session.scalar(
        select(func.count()).select_from(CitationVerification)
    )

    async def sufficient_reuse(**_kwargs):
        decision = {
            "verdict": "sufficient",
            "reason": "the previous verified package directly covers the follow-up",
            "referenced_chunk_ids": [package.hit_chunk_ids_json[0]],
            "expected_answer_shape": "explanation",
        }
        return {
            "protocol_version": "verified_context_reuse_evaluator_v1",
            **decision,
            "model_call_count": 1,
            "input_hash": stable_hash({"reuse": "input"}),
            "output_hash": stable_hash(decision),
        }

    monkeypatch.setattr(
        agent_graph,
        "evaluate_verified_context_reuse",
        sufficient_reuse,
    )

    class ExactSpanDirectChatProvider(fake_model_stack["ChatProvider"]):
        async def answer_question_with_meta(
            self,
            question: str,
            contexts: list[dict],
            history: list[dict] | None = None,
            context_quality: str = "normal",
            **_kwargs,
        ):
            from app.services.embeddings import ChatCallResult, ChatProvider
            from app.services.agent_repair import split_answer_claims

            content = str(contexts[0]["content"])
            answer = next(
                claim
                for claim in split_answer_claims(content)
                if len(claim) >= 20 and not claim.lstrip().startswith("#")
            )
            metadata = dict(
                ChatProvider()._answer_prompt_bundle(
                    question,
                    context_quality=context_quality,
                )["protocol_metadata"]
            )
            return ChatCallResult(
                answer=answer,
                provider="unit_chat",
                model="unit-chat",
                external_called=False,
                prompt_protocol_version=metadata["protocol_version"],
                prompt_protocol_hash=metadata["prompt_protocol_hash"],
                grounding_envelope_protocol_version=metadata[
                    "protocol_version"
                ],
                grounding_envelope_hash=metadata["envelope_hash"],
                profile_hash=metadata["profile_hash"],
            )

    monkeypatch.setattr(
        agent_graph,
        "ChatProvider",
        ExactSpanDirectChatProvider,
    )

    second = AgentResponse.model_validate(
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                question="Explain that definition using the same evidence.",
                knowledge_base_id=knowledge_base.id,
                session_id=first.session_id,
            ),
        )
    )

    assert second.route == "direct_answer", [
        (
            item.node,
            item.output_summary,
            item.scores.model_dump().get("reason"),
            item.scores.model_dump().get("claim_pass_rate"),
        )
        for item in second.trace
        if item.node.startswith("direct_answer")
    ]
    assert second.direct_answer_mode == "verified_context_reuse"
    import hashlib
    from app.models import AgentRun
    second_scope = db_session.get(AgentRun, second.run_id).metadata_json["answer_question_scope"]
    first_scope = db_session.get(AgentRun, first.run_id).metadata_json["answer_question_scope"]
    assert second_scope["available"] is True
    assert second_scope["current_question_hash"] == hashlib.sha256("Explain that definition using the same evidence.".encode()).hexdigest()
    assert second_scope["packet_hash"] != first_scope["packet_hash"]
    assert second.context_package_id == first.context_package_id
    assert second.retrieval_trace_id == first.retrieval_trace_id
    assert second.citations
    direct_answer_session = db_session.get(
        AnswerSession,
        second.answer_model_audit.answer_session_id,
    )
    assert direct_answer_session is not None
    assert (direct_answer_session.diagnostics_json or {}).get(
        "direct_answer_mode"
    ) == "verified_context_reuse"
    assert all(item.source_binding is not None and item.source_binding.status == "source_bound" and item.verification is None for item in second.citations)
    assert second.answer_model_audit.answer_reflection.citation_judge_model_call_count == 0
    assert not {
        "typed_action_executor",
        "layered_retrieval",
        "context_package",
    }.intersection({item.node for item in second.trace})
    assert db_session.scalar(select(func.count()).select_from(RewardEvent)) == rewards_before
    assert db_session.scalar(select(func.count()).select_from(PolicyState)) == policies_before
    assert db_session.scalar(
        select(func.count()).select_from(CitationVerification)
    ) == verifications_before
    from app.models import AnswerSourceBinding
    assert db_session.scalar(select(func.count()).select_from(AnswerSourceBinding).where(AnswerSourceBinding.answer_session_id == direct_answer_session.id)) > 0

    from copy import deepcopy
    import hashlib
    import json
    from pydantic import ValidationError
    from app.models import AgentAction
    from app.schemas import AgentPEAuditResponse
    from app.services.agent_pe_audit import AgentPEAuditIntegrityError, load_agent_pe_audit

    public = load_agent_pe_audit(db_session, second.run_id)
    assert public.plans == []
    assert public.actions and all(row.plan_id is None and row.plan_index is None for row in public.actions)
    review = next(row for row in public.actions if row.action_type == "review_answer")
    assert review.status == "completed"
    assert any(row.observation_type == "answer_reflection_final" and row.action_id == review.id for row in public.observations)
    assert any(json.loads(row.diagnostics.canonical_json).get("provider_response_persisted") is False for row in public.observations)
    assert not public.provider_raw_response_exposed

    for attack in ("missing_marker", "invalid_validator", "non_review_action", "index_gap", "foreign_observation"):
        forged = deepcopy(public.model_dump(mode="json"))
        first_action = forged["actions"][0]
        if attack in {"missing_marker", "invalid_validator"}:
            validation = json.loads(first_action["validator"]["payload"]["canonical_json"])
            if attack == "missing_marker":
                validation.pop("direct_context_reuse")
            else:
                validation["valid"] = False
            canonical = json.dumps(validation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            first_action["validator"]["payload"].update(canonical_json=canonical, sha256=hashlib.sha256(canonical.encode()).hexdigest())
        elif attack == "non_review_action":
            first_action["action_type"] = "recall_chunks"
        elif attack == "index_gap":
            first_action["action_index"] = len(forged["actions"]) + 1
        else:
            forged["observations"][0]["plan_id"] = "unit-test-foreign-plan"
        with pytest.raises(ValidationError):
            AgentPEAuditResponse.model_validate(forged)

    stored_review = db_session.get(AgentAction, review.id)
    stored_review.validation_json = {**stored_review.validation_json, "valid": False}
    db_session.flush()
    with pytest.raises(AgentPEAuditIntegrityError, match="plan linkage"):
        load_agent_pe_audit(db_session, second.run_id)


@pytest.mark.asyncio
async def test_system_capability_sse_terminal_meta_matches_final_route(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
    local_agent_admission,
) -> None:
    from app.schemas import AgentRequest
    from app.services import agent_graph
    class PlanningProvider:
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
    monkeypatch.setattr(agent_graph, "ChatProvider", PlanningProvider)

    def forbidden_graph_admission(*_args, **_kwargs):
        raise AssertionError("system direct SSE must not admit the graph")

    monkeypatch.setattr(
        agent_graph,
        "active_graph_admission_gate",
        forbidden_graph_admission,
    )
    events = [
        event
        async for event in agent_graph.stream_agent_events(
            AgentRequest(
                question="What can you do?",
                knowledge_base_id=sample_knowledge_base.id,
                stream_trace=True,
            )
        )
    ]
    meta_events = [event for event in events if event.get("type") == "meta"]
    final = next(event["response"] for event in events if event.get("type") == "final")

    assert meta_events[-1]["route"] == final["route"] == "system_capability"
    assert meta_events[-1]["direct_answer_mode"] == final[
        "direct_answer_mode"
    ] == "system_capability"
    assert final["citations"] == []
    assert final["context_package_id"] is None
    assert meta_events[-1]["terminal_outcome"] == final["terminal_outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.usefixtures('historical_answer_executor')
async def test_historical_verified_context_reuse_insufficient_falls_back_to_layered_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
    local_agent_admission,
) -> None:
    from app.schemas import AgentRequest, AgentResponse
    from app.services import agent_graph
    from app.services.chunking import stable_hash

    knowledge_base = populated_context_graph["knowledge_base"]
    first = AgentResponse.model_validate(
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                question="What is a Bayesian network?",
                knowledge_base_id=knowledge_base.id,
            ),
        )
    )

    async def insufficient_reuse(**_kwargs):
        decision = {
            "verdict": "insufficient",
            "reason": "the follow-up asks for evidence outside the verified package",
            "referenced_chunk_ids": [],
            "expected_answer_shape": "grounded_answer",
        }
        return {
            "protocol_version": "verified_context_reuse_evaluator_v1",
            **decision,
            "model_call_count": 1,
            "input_hash": stable_hash({"reuse": "outside"}),
            "output_hash": stable_hash(decision),
        }

    monkeypatch.setattr(
        agent_graph,
        "evaluate_verified_context_reuse",
        insufficient_reuse,
    )
    second = AgentResponse.model_validate(
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                question="Compare this with a topic not covered by that evidence.",
                knowledge_base_id=knowledge_base.id,
                session_id=first.session_id,
            ),
        )
    )

    assert second.route == "layered_context_graph"
    assert second.direct_answer_mode is None
    assert any(
        item.node == "direct_answer_fallback"
        and item.scores.model_dump().get("reason")
        == "reuse_evaluator_insufficient"
        for item in second.trace
    )
    assert any(item.node == "typed_action_executor" for item in second.trace)


@pytest.mark.asyncio
@pytest.mark.usefixtures('historical_answer_executor')
async def test_historical_second_plan_no_progress_reviews_full_context_and_emits_final(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
    local_agent_admission,
) -> None:
    from sqlalchemy import select

    from app.models import AgentPlan, AgentRun
    from app.schemas import AgentRequest, AgentResponse
    from app.services import agent_graph

    async def always_insufficient(**_kwargs):
        return {
            "protocol_version": "bounded_graph_evidence_evaluator_v4",
            "verdict": "insufficient_corpus",
            "reason": "unit test forces a second planning round",
            "target_ids": [],
            "expected_evidence": {},
        }

    monkeypatch.setattr(
        agent_graph,
        "evaluate_graph_evidence",
        always_insufficient,
    )

    async def collect_events() -> list[dict]:
        return [
            event
            async for event in agent_graph.stream_agent_events(
                AgentRequest(
                    question="Explain a topic that requires more evidence.",
                    knowledge_base_id=populated_context_graph[
                        "knowledge_base"
                    ].id,
                    stream_trace=True,
                )
            )
        ]

    events = await asyncio.wait_for(collect_events(), timeout=30)
    final = AgentResponse.model_validate(
        next(
            event["response"]
            for event in events
            if event.get("type") == "final"
        )
    )
    streamed_trace = [
        event["trace"]
        for event in events
        if event.get("type") == "trace"
    ]
    nodes = [event["node"] for event in streamed_trace]

    assert nodes.count("agent_planner") == 2
    assert "replan_no_progress" in nodes
    assert "evidence_gate" in nodes
    run = db_session.get(AgentRun, final.run_id)
    assert run is not None
    assert run.status == "completed"
    assert final.context_package_id is not None
    assert final.answer_model_audit.preliminary_evidence_uncertain is True
    assert final.answer_model_audit.answer_reflection.reflection_model_call_count >= 1
    assert final.answer_model_audit.answer_reflection.citation_judge_model_call_count == 0
    plans = list(
        db_session.scalars(
            select(AgentPlan)
            .where(AgentPlan.run_id == final.run_id)
            .order_by(AgentPlan.plan_index)
        ).all()
    )
    assert len(plans) == 2
    second_projection = plans[1].diagnostics_json[
        "planner_observation_projection"
    ]
    assert second_projection["serialized_char_count"] <= (
        agent_graph.AGENT_PLANNER_OBSERVATION_MAX_CHARS
    )
    assert second_projection["full_graph_observation_forwarded"] is False


def test_cancel_does_not_overwrite_needs_clarification_terminal_state(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import AgentRun, QASession
    from app.services.agent_graph import cancel_agent_run

    session = QASession(
        knowledge_base_id=sample_knowledge_base.id,
        title="terminal cancellation regression",
    )
    db_session.add(session)
    db_session.flush()
    run = AgentRun(
        knowledge_base_id=sample_knowledge_base.id,
        session_id=session.id,
        question="unanswerable",
        route="layered_context_graph",
        status="needs_clarification",
    )
    db_session.add(run)
    db_session.commit()

    payload = cancel_agent_run(db_session, run.id)
    db_session.refresh(run)

    assert payload["status"] == "needs_clarification"
    assert run.status == "needs_clarification"
