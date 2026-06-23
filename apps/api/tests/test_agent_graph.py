from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_propose_query_facets_validates_llm_packet(monkeypatch):
    from app.services import agent_graph

    class FacetChatProvider:
        async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict | None = None) -> dict:
            return {
                "domain_facets": [{"facet": "\u914d\u7f6e\u6a21\u578b", "aliases": ["configuration model"]}],
                "procedure_facets": [{"facet": "\u7b97\u6cd5\u6b65\u9aa4", "aliases": ["stub", "\u534a\u8fb9", "\u968f\u673a\u5339\u914d"]}],
                "drop_terms": ["\u7ed9", "\u6211", "\u7684", "\u5177\u4f53"],
                "answer_shape": "step_by_step_algorithm",
                "chunk_ids": ["must-not-survive"],
                "document_ids": ["must-not-survive"],
            }

    monkeypatch.setattr(agent_graph, "ChatProvider", FacetChatProvider)

    facets = await agent_graph.propose_query_facets(
        "\u7ed9\u6211\u914d\u7f6e\u6a21\u578b\u7684\u5177\u4f53\u7b97\u6cd5\u6b65\u9aa4",
        [],
        {"intent": "procedure"},
    )

    assert facets["protocol_version"] == "query_facet_packet_v1"
    assert facets["intent"] == "procedure"
    assert "\u914d\u7f6e\u6a21\u578b" in facets["required_facets"]
    assert "\u7b97\u6cd5\u6b65\u9aa4" in facets["required_facets"]
    assert "\u7ed9" not in facets["required_facets"]
    assert "chunk_ids" not in facets
    assert "document_ids" not in facets


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
async def test_citation_verification_hard_interrupts_slow_judge(monkeypatch):
    from app.services import agent_graph

    class SlowChatProvider:
        async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict | None = None) -> dict:
            await asyncio.sleep(1)
            return {"verifications": []}

    monkeypatch.setattr(agent_graph, "ChatProvider", SlowChatProvider)
    monkeypatch.setattr(agent_graph, "citation_verification_judge_timeout_seconds", lambda verification_budget: 0.01)

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

    assert results[0]["verdict"] == "unsupported"
    assert results[0]["failure_type"] == "verification_model_timeout"
    assert results[0]["diagnostics"]["llm_entailment_judge"] == "timeout_hard_interrupt"


def test_citation_payloads_supported_only_filters_failed_verifications():
    from app.services.agent_graph import citation_payloads_from_package

    package = SimpleNamespace(
        id="package-1",
        retrieval_trace_id="trace-1",
        hit_chunk_ids_json=["chunk-supported", "chunk-unsupported"],
        package_json={
            "chunks": [
                {
                    "chunk_id": "chunk-supported",
                    "document_id": "doc-1",
                    "document_version_id": "version-1",
                    "document_title": "Supported",
                    "source_path": "/tmp/supported.pdf",
                    "section_path": ["Section"],
                    "page_range": [1],
                    "char_span": [0, 12],
                    "source_span": {"chunk_id": "chunk-supported", "char_span": [0, 12]},
                    "content": "The supported context entails the answer.",
                },
                {
                    "chunk_id": "chunk-unsupported",
                    "document_id": "doc-2",
                    "document_version_id": "version-2",
                    "document_title": "Unsupported",
                    "source_path": "/tmp/unsupported.pdf",
                    "section_path": ["Other"],
                    "page_range": [2],
                    "char_span": [0, 11],
                    "source_span": {"chunk_id": "chunk-unsupported", "char_span": [0, 11]},
                    "content": "Unrelated context.",
                },
            ]
        },
    )
    verification_by_chunk = {
        "chunk-supported": SimpleNamespace(id="verification-1", verdict="supported", confidence=0.9, diagnostics_json={}),
        "chunk-unsupported": SimpleNamespace(id="verification-2", verdict="unsupported", confidence=0.3, diagnostics_json={}),
    }

    citations = citation_payloads_from_package(package, verification_by_chunk=verification_by_chunk, supported_only=True)

    assert [item["chunk_id"] for item in citations] == ["chunk-supported"]
    assert citations[0]["verification"]["verdict"] == "supported"


def test_citation_payloads_can_select_restored_supporting_chunk():
    from app.services.agent_graph import citation_payloads_from_package

    package = SimpleNamespace(
        id="package-1",
        retrieval_trace_id="trace-1",
        hit_chunk_ids_json=["chunk-distribution"],
        restored_chunk_ids_json=["chunk-mh"],
        bridge_chunk_ids_json=[],
        package_json={
            "chunks": [
                {
                    "chunk_id": "chunk-distribution",
                    "document_id": "doc-1",
                    "document_version_id": "version-1",
                    "document_title": "Lecture 2 - Slides",
                    "source_path": "/tmp/distributions.pdf",
                    "section_path": ["Normal distribution"],
                    "page_range": [2],
                    "char_span": [10, 60],
                    "source_span": {"chunk_id": "chunk-distribution", "char_span": [10, 60]},
                    "content": "The normal distribution has density parameters mu and sigma. This section reviews probability distributions.",
                },
                {
                    "chunk_id": "chunk-mh",
                    "document_id": "doc-2",
                    "document_version_id": "version-2",
                    "document_title": "Details of MH Algorithm",
                    "source_path": "/tmp/mh.pdf",
                    "section_path": ["The Random-Walk proposal"],
                    "page_range": [8],
                    "char_span": [100, 260],
                    "source_span": {"chunk_id": "chunk-mh", "char_span": [100, 260]},
                    "content": "The proposal is symmetric: q(theta*|theta i-1) = q(theta i-1|theta*). So the acceptance ratio simplifies to alpha = g(theta*) / g(theta i-1).",
                },
            ]
        },
    )

    citations = citation_payloads_from_package(
        package,
        question="Explain the Metropolis-Hastings acceptance probability using the course material.",
        answer=(
            "For a symmetric Metropolis-Hastings random-walk proposal, the course material says the proposal densities cancel, "
            "so the acceptance ratio simplifies to alpha = g(theta*) / g(theta i-1)."
        ),
    )

    assert citations
    assert citations[0]["chunk_id"] == "chunk-mh"
    assert citations[0]["source_span"]["context_package_id"] == "package-1"


def test_cancel_agent_run_marks_running_run_failed_and_records_trace(db_session, sample_knowledge_base):
    from app.models import AgentRun, AgentTraceEvent
    from app.services.agent_graph import CANCELLED_BY_USER, cancel_agent_run

    run = AgentRun(
        knowledge_base_id=sample_knowledge_base.id,
        question="cancel this run",
        status="running",
        route="layered_context_graph",
        current_node="grounded_answer",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    payload = cancel_agent_run(db_session, run.id)

    db_session.refresh(run)
    events = db_session.query(AgentTraceEvent).filter(AgentTraceEvent.run_id == run.id).all()
    assert payload["status"] == "failed"
    assert payload["error"] == CANCELLED_BY_USER
    assert run.status == "failed"
    assert run.error_message == CANCELLED_BY_USER
    assert run.completed_at is not None
    assert [event.node for event in events] == ["cancelled_by_user"]


@pytest.mark.asyncio
async def test_stream_agent_events_cancels_task_when_stream_closes(monkeypatch, db_session, sample_knowledge_base):
    from app.models import AgentRun
    from app.schemas import AgentRequest
    from app.services import agent_graph

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_execute_agent_run_and_close(db, request, session, run):
        try:
            agent_graph.set_run_state(db, run, "running", current_node="test_wait")
            started.set()
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            db.close()

    monkeypatch.setattr(agent_graph, "_execute_agent_run_and_close", fake_execute_agent_run_and_close)
    stream = agent_graph.stream_agent_events(
        AgentRequest(
            knowledge_base_id=sample_knowledge_base.id,
            question="cancel the stream",
            stream_trace=True,
        )
    )

    meta = await stream.__anext__()
    await asyncio.wait_for(started.wait(), timeout=1)
    await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), timeout=1)

    db_session.expire_all()
    run = db_session.get(AgentRun, meta["run_id"])
    assert run is not None
    assert run.status == "failed"
    assert run.error_message == agent_graph.CANCELLED_BY_USER


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
