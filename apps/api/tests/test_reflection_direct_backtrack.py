"""A review-initiated direct-to-retrieval handoff stays one audited run."""
import json

import pytest
pytestmark = pytest.mark.usefixtures('historical_answer_executor')
from sqlalchemy import select

@pytest.mark.asyncio
@pytest.mark.parametrize("round_budget,format_retry", [(1, False), (2, False), (2, True)])
async def test_verified_reuse_fallback_preserves_joint_review_accounting(
    monkeypatch, db_session, populated_context_graph, fake_model_stack, local_agent_admission, round_budget, format_retry,
):
    from app.core.config import get_settings
    from app.models import AgentObservation, AgentRun, AnswerSession, ContextPackage, RewardEvent
    from app.schemas import AgentRequest, AgentResponse
    from app.services import agent_graph
    from app.services.chunking import stable_hash
    from app.services.agent_reflection import ReflectionBudgetExhausted

    monkeypatch.setattr(get_settings(), "agent_reflection_round_budget", round_budget)

    kb = populated_context_graph["knowledge_base"]
    first = await agent_graph.run_agent(db_session, AgentRequest(
        question="What is a Bayesian network?", knowledge_base_id=kb.id, top_k=1,
    ))
    package = db_session.get(ContextPackage, first["context_package_id"])

    async def sufficient_reuse(**_kwargs):
        decision = {"verdict": "sufficient", "reason": "Synthetic initial reuse decision.",
                    "referenced_chunk_ids": [package.hit_chunk_ids_json[0]], "expected_answer_shape": "explanation"}
        return {"protocol_version": "verified_context_reuse_evaluator_v2", **decision,
                "model_call_count": 1, "input_hash": stable_hash({"unit-test": "reuse"}), "output_hash": stable_hash(decision)}
    monkeypatch.setattr(agent_graph, "evaluate_verified_context_reuse", sufficient_reuse)
    calls = {"generation": 0, "reflection": 0, "shape_failures": 0}
    controls = []

    class BacktrackModel(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "IMMUTABLE ANSWER REFLECTION SYSTEM ENVELOPE" not in system_prompt:
                return await super().classify_json(system_prompt, user_prompt, fallback)
            packet = json.loads(user_prompt)
            if packet.get("answer_draft") is None:
                if format_retry and not calls["shape_failures"]:
                    from app.services.embeddings import ProviderJSONShapeError
                    calls["shape_failures"] += 1
                    raise ProviderJSONShapeError({"error_code": "json_decode_error", "field_path": "$"})
                calls["generation"] += 1
                source = packet["evidence"][0]
                return {"protocol_version": "structured_answer_self_assessment_v1",
                    "answer_units": [{"kind": "factual", "text": source["text"], "source_handles": [source["source_handle"]]}],
                    "self_assessment": {"question_relevance": 0.9, "context_relevance": 0.9, "coverage": 0.9,
                        "needs_reflection": True, "issue_types": [], "summary": "Review this synthetic answer."}}
            calls["reflection"] += 1
            controls.append(packet["controls"])
            expand = calls["reflection"] == 1
            return {"protocol_version": "agent_answer_reflection_v1", "action": "replan_retrieval" if expand else "accept",
                "issue_types": ["missing_evidence"] if expand else [], "target_unit_indexes": [], "source_handles": [],
                "missing_facets": ["factorization examples"] if expand else [],
                "correction_instructions": "Retrieve evidence for the current factorization question." if expand else "",
                "clarification_question": None}

    monkeypatch.setattr(agent_graph, "ChatProvider", BacktrackModel)
    request = AgentRequest(
        question="Explain that definition and its factorization examples.", knowledge_base_id=kb.id,
        session_id=first["session_id"], top_k=4,
    )
    if round_budget == 1:
        with pytest.raises(ReflectionBudgetExhausted, match="reflection_round_budget_exhausted"):
            await agent_graph.run_agent(db_session, request)
        run = db_session.scalar(select(AgentRun).where(AgentRun.question == request.question))
        assert run.status == "failed" and run.metadata_json["policy_update_eligible"] is False
        assert calls["reflection"] == 1
        assert len(list(db_session.scalars(select(RewardEvent)))) == 1
        return
    second = AgentResponse.model_validate(await agent_graph.run_agent(db_session, request))
    assert second.route == "layered_context_graph"
    review = second.answer_model_audit.answer_reflection
    assert review.reflection_rounds_used == calls["reflection"] == 2
    assert calls["generation"] == 2
    assert review.generation_model_call_count == calls["generation"] + calls["shape_failures"]
    assert [row["remaining_reflection_rounds"] for row in controls] == [2, 1]
    assert controls[0]["route"] == "verified_context_reuse"
    assert controls[-1]["route"] == "layered_context_graph"
    assert all(type(row["remaining_planning_rounds"]) is int for row in controls)
    answer = db_session.get(AnswerSession, second.answer_model_audit.answer_session_id)
    audit = answer.diagnostics_json["answer_reflection"]
    from app.services.reflection_run import ordered_reflection_events
    observations = ordered_reflection_events(db_session, run_id=second.run_id)
    assert audit["events"] == observations
    assert [row["sequence_index"] for row in observations] == list(range(len(observations)))
