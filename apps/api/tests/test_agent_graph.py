from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_citation_verification_normalizes_label_confidence(monkeypatch):
    from app.services import agent_graph

    class LabelConfidenceChatProvider:
        async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict | None = None) -> dict:
            return {
                "verifications": [
                    {
                        "citation_index": 1,
                        "verdict": "supported",
                        "failure_type": "none",
                        "confidence": "high",
                        "reason": "The cited context entails the claim.",
                    }
                ]
            }

    monkeypatch.setattr(agent_graph, "ChatProvider", LabelConfidenceChatProvider)

    results = await agent_graph.verify_answer_against_context(
        "Bayesian regression uses a normal likelihood.",
        [
            {
                "citation_index": 1,
                "chunk_id": "chunk-1",
                "source_span": {"chunk_id": "chunk-1", "char_span": [0, 42]},
            }
        ],
        [{"chunk_id": "chunk-1", "content": "Bayesian regression uses a normal likelihood."}],
        verification_budget=1,
    )

    assert results[0]["verdict"] == "supported"
    assert results[0]["confidence"] == 0.85
    assert results[0]["diagnostics"]["llm_entailment_confidence"]["confidence_raw"] == "high"
    assert results[0]["diagnostics"]["llm_entailment_confidence"]["confidence_normalized_from"] == "label"


@pytest.mark.asyncio
async def test_agent_answers_from_context_package_and_records_audit(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import AgentAction, AgentObservation, AgentPlan, AnswerSession, CitationVerification, PolicyState, RewardEvent
    from app.schemas import AgentRequest, QAResponse, SearchFilters
    from app.services.agent_graph import run_agent

    kb = populated_context_graph["knowledge_base"]
    response = await run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=kb.id,
            question="Explain Bayesian network factorization.",
            filters=SearchFilters(),
            top_k=4,
        ),
    )
    QAResponse.model_validate(response)
    assert response["answer"].startswith("Grounded answer:")
    assert response["citations"]
    assert response["trace"]
    trace_nodes = [item["node"] for item in response["trace"]]
    assert "agent_planner" in trace_nodes
    assert "typed_action_validation" in trace_nodes
    assert "citation_verification" in trace_nodes
    assert "reward_event" in trace_nodes
    assert response["context_package_id"]
    assert response["retrieval_trace_id"]
    assert response["model_audit"]["context_package_id"] == response["context_package_id"]
    assert response["model_audit"]["retrieval_trace_id"] == response["retrieval_trace_id"]
    assert response["model_audit"]["citation_verification_pass_rate"] == 1.0
    assert response["answer_model_audit"]["context_package_id"]
    assert response["answer_model_audit"]["answer_session_id"]
    assert response["citations"][0]["verification"]["verdict"] == "supported"
    assert db_session.scalar(select(func.count(AnswerSession.id)).where(AnswerSession.knowledge_base_id == kb.id)) == 1
    assert db_session.scalar(select(func.count(CitationVerification.id)).where(CitationVerification.knowledge_base_id == kb.id)) >= 1
    assert db_session.scalar(select(func.count(RewardEvent.id)).where(RewardEvent.knowledge_base_id == kb.id)) == 1
    assert db_session.scalar(select(func.count(AgentPlan.id)).where(AgentPlan.knowledge_base_id == kb.id)) == 1
    assert db_session.scalar(select(func.count(AgentAction.id)).join(AgentPlan, AgentAction.plan_id == AgentPlan.id).where(AgentPlan.knowledge_base_id == kb.id)) >= 3
    assert db_session.scalar(select(func.count(AgentObservation.id))) >= 3
    latest_policy = db_session.scalar(select(PolicyState).where(PolicyState.knowledge_base_id == kb.id).order_by(PolicyState.created_at.desc()))
    assert latest_policy is not None
    assert (latest_policy.reward_summary_json or {}).get("last_reward_event_id")
