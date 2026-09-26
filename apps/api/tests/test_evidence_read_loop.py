import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.models import AgentObservation, AgentTraceEvent, AnswerSession, ContextPackage
from app.schemas import AgentRequest, QAResponse
from app.services import agent_graph
from app.services.agent_context import (
    ContextCapacityError,
    ContextTree,
    ContextTreeNode,
    ContextUnit,
    context_event,
    plan_context,
    stable_priority_order,
    PriorityCandidate,
    validate_tool_event_pairs,
)
from app.services.evidence_read_loop import EvidenceToolCall, MidDirectory
from app.services.intent_execution_agent import _agent_total_timeout_seconds


def test_agent_total_deadline_is_independent_from_the_model_call_deadline():
    assert _agent_total_timeout_seconds(SimpleNamespace(
        retrieval_total_timeout_seconds=540,
        model_request_timeout_seconds=240,
    )) == 540
    assert _agent_total_timeout_seconds(SimpleNamespace(
        retrieval_total_timeout_seconds=180,
        model_request_timeout_seconds=240,
    )) == 180


def test_evidence_tool_contract_rejects_zero_progress_and_duplicates():
    with pytest.raises(ValidationError, match="evidence_read_mid_handles_empty"):
        EvidenceToolCall.model_validate({
            "tool": "evidence.read",
            "arguments": {"mid_handles": []},
        })
    with pytest.raises(ValidationError, match="evidence_tool_handles_duplicate"):
        EvidenceToolCall.model_validate({
            "tool": "evidence.commit",
            "arguments": {"source_handles": ["src_1", "src_1"]},
        })


def test_context_plan_preserves_tool_pairs_and_p0_p1():
    units = [
        ContextUnit("task", "user_task", "P0", "task", set_name="pinned"),
        ContextUnit("call", "tool_call", "P2", "x" * 80, atomic_group="pair"),
        ContextUnit("result", "tool_result", "P2", "y" * 80, atomic_group="pair"),
        ContextUnit("directory", "semantic_directory", "P3", "node"),
    ]
    result = plan_context(
        units,
        input_token_budget=4,
        reserved_output_tokens=256,
        stable_prefix="stable",
    )
    assert result.kept_keys == ("task", "directory")
    assert result.audit["compression_applied"] is True
    assert {item["key"] for item in result.audit["removed_units"]} == {"call", "result"}
    with pytest.raises(ContextCapacityError, match="context_p0_p1_capacity_exceeded"):
        plan_context(
            [
                ContextUnit("task", "user_task", "P0", "x" * 80),
                ContextUnit("source", "raw_source", "P1", "y" * 80),
            ],
            input_token_budget=1,
            reserved_output_tokens=256,
            stable_prefix="stable",
        )


def test_context_event_log_requires_atomic_tool_pairs():
    call = context_event("tool_call", tool="evidence.read", arguments={"mid_handles": ["mid_1"]})
    result = context_event("tool_result_ref", tool="evidence.read", status="ok", source_handles=["src_1"])
    validate_tool_event_pairs([context_event("user_task", task_hash="a" * 64), call, result])
    with pytest.raises(ValueError, match="context_tool_call_without_result"):
        validate_tool_event_pairs([call])


def test_priority_queue_is_stable_and_semantic_first():
    assert stable_priority_order(
        [
            PriorityCandidate("late", retrieval_order=2, estimated_tokens=1),
            PriorityCandidate("explicit", explicit_source_responsibility=True, retrieval_order=9),
            PriorityCandidate("active", active_branch=True, retrieval_order=1),
            PriorityCandidate("cheap", retrieval_order=2, estimated_tokens=0),
        ]
    ) == ("explicit", "active", "cheap", "late")


def _two_mid_directory(evidence, trace) -> MidDirectory:
    handles = [source["source_handle"] for source in evidence.sources]
    assert len(handles) >= 2
    split = max(1, len(handles) // 2)
    mid_to_sources = {
        "mid_1": tuple(handles[:split]),
        "mid_2": tuple(handles[split:]),
    }
    source_to_mids = {
        handle: (("mid_1",) if handle in mid_to_sources["mid_1"] else ("mid_2",))
        for handle in handles
    }
    tree = ContextTree(
        "task",
        (
            ContextTreeNode("task", "task", ("semantic:mid_1", "semantic:mid_2")),
            ContextTreeNode("semantic:mid_1", "semantic_node"),
            ContextTreeNode("semantic:mid_2", "semantic_node"),
        ),
    )
    identity = {
        "active_mid_state_hash": trace.mid_concept_hash,
        "entries": [
            {"mid_handle": key, "source_handles": list(value)}
            for key, value in mid_to_sources.items()
        ],
        "mandatory_sources": [],
        "tree_hash": tree.audit()["tree_hash"],
    }
    return MidDirectory(
        active_state_hash=trace.mid_concept_hash,
        model_entries=(
            {"mid_handle": "mid_1", "title": "Foundations", "summary": "Core definitions."},
            {"mid_handle": "mid_2", "title": "Applications", "summary": "Applied details."},
        ),
        mid_to_sources=mid_to_sources,
        source_to_mids=source_to_mids,
        mandatory_sources=(),
        server_identity=identity,
        tree=tree,
    )


@pytest.mark.asyncio
async def test_single_mid_package_uses_deterministic_direct_path(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from agent_test_support import Admission
    from app.services import layered_execution_v1

    evidence_calls = 0

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            nonlocal evidence_calls
            if "EVIDENCE TOOL SESSION V1" in system_prompt:
                evidence_calls += 1
            return await super().classify_json(system_prompt, user_prompt, fallback)

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(layered_execution_v1, "EmbeddingProvider", fake_model_stack["EmbeddingProvider"])
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=populated_context_graph["knowledge_base"].id,
            question="Summarize the topics.",
            top_k=4,
        ),
        admission=Admission(),
    )
    assert QAResponse.model_validate(response).terminal_outcome == "completed"
    loop = db_session.scalar(select(AgentObservation).where(
        AgentObservation.run_id == response["run_id"],
        AgentObservation.observation_type == "evidence_read_loop",
    ))
    assert loop.observation_json["protocol_version"] == "evidence_read_loop_v2"
    assert loop.observation_json["state"]["deterministic_direct"] is True
    assert loop.observation_json["state"]["decision_call_count"] == 0
    assert evidence_calls == 0

    from app.intent_contracts import AcceptedPlan
    from app.services.answer_sources import build_answer_evidence_manifest
    from app.services.context_graph import context_package_to_contexts
    from app.services.evidence_read_loop import build_mid_directory
    from app.models import RetrievalTrace

    plan_row = db_session.scalar(select(AgentObservation).where(
        AgentObservation.run_id == response["run_id"],
        AgentObservation.observation_type == "intent_execution_plan",
    ))
    package = db_session.get(ContextPackage, response["context_package_id"])
    trace = db_session.get(RetrievalTrace, response["retrieval_trace_id"])
    evidence = build_answer_evidence_manifest(package, context_package_to_contexts(package))
    directory = build_mid_directory(
        db_session,
        plan=AcceptedPlan.model_validate(plan_row.observation_json["accepted_plan"]),
        package=package,
        trace=trace,
        evidence=evidence,
    )
    assert directory.model_entries
    assert all(set(item) == {"mid_handle", "title", "summary"} for item in directory.model_entries)
    assert any(directory.source_to_mids[handle] for handle in directory.source_to_mids)
    assert set(directory.mandatory_sources) == {
        handle for handle, mids in directory.source_to_mids.items() if not mids
    }


@pytest.mark.asyncio
async def test_continuous_mid_read_freezes_only_committed_raw_source(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from agent_test_support import Admission
    from app.services import evidence_read_loop, layered_execution_v1
    from test_intent_contracts import proposal

    decision_calls = 0
    observed_message_counts = []

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            nonlocal decision_calls
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                return proposal(layer="chunk")
            if "EVIDENCE TOOL SESSION V1" in system_prompt:
                decision_calls += 1
                packet = json.loads(user_prompt)
                observed_message_counts.append(len(packet["messages"]))
                if decision_calls == 1:
                    return {
                        "tool": "evidence.read",
                        "arguments": {"mid_handles": ["mid_1"]},
                    }
                latest = json.loads(packet["messages"][-1]["content"])
                selected = latest["groups"][0]["sources"][0]["source_handle"]
                return {
                    "tool": "evidence.commit",
                    "arguments": {"source_handles": [selected]},
                }
            return await super().classify_json(system_prompt, user_prompt, fallback)

        async def complete_text(self, system_prompt, user_prompt, *, max_tokens):
            packet = json.loads(user_prompt)
            assert "history_summary" not in packet
            assert len(packet["evidence"]) == 1
            assert packet["evidence"][0]["source_handle"] == "src_1"
            return "Selected evidence only.⟦cite:src_1⟧"

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(layered_execution_v1, "EmbeddingProvider", fake_model_stack["EmbeddingProvider"])
    monkeypatch.setattr(
        evidence_read_loop,
        "build_mid_directory",
        lambda db, *, plan, package, trace, evidence: _two_mid_directory(evidence, trace),
    )
    monkeypatch.setattr(evidence_read_loop, "DIRECT_CONTEXT_TOKEN_LIMIT", -1)
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=populated_context_graph["knowledge_base"].id,
            question="Summarize the topics.",
            top_k=4,
        ),
        admission=Admission(),
    )

    assert decision_calls == 2
    assert observed_message_counts == [1, 3]
    assert len(response["citations"]) == 1
    assert len(response["used_chunks"]) == 1
    timing = response["model_audit"]["qa_performance"]["stages"]
    assert timing["evidence_directory"]["success_count"] == 1
    assert timing["evidence_read"]["success_count"] == 1
    assert timing["evidence_freeze"]["success_count"] == 1
    loop = db_session.scalar(select(AgentObservation).where(
        AgentObservation.run_id == response["run_id"],
        AgentObservation.observation_type == "evidence_read_loop",
    ))
    state = loop.observation_json["state"]
    assert state["read_action_count"] == 1
    assert len(state["committed_source_handles"]) == 1
    assert len(state["context_plans"]) == 2
    assert all(item["body_persisted"] is False for item in state["context_plans"])
    package = db_session.get(ContextPackage, response["context_package_id"])
    source_text = package.package_json["chunks"][0]["content"]
    assert source_text not in json.dumps(loop.observation_json)
    from app.intent_contracts import AcceptedPlan
    from app.services.answer_sources import build_answer_evidence_manifest
    from app.services.context_graph import context_package_to_contexts
    from app.services.evidence_read_loop import _messages_and_units
    from app.models import RetrievalTrace

    plan_row = db_session.scalar(select(AgentObservation).where(
        AgentObservation.run_id == response["run_id"],
        AgentObservation.observation_type == "intent_execution_plan",
    ))
    trace = db_session.get(RetrievalTrace, response["retrieval_trace_id"])
    admitted = build_answer_evidence_manifest(package, context_package_to_contexts(package))
    rebuilt, _units = _messages_and_units(
        plan=AcceptedPlan.model_validate(plan_row.observation_json["accepted_plan"]),
        directory=_two_mid_directory(admitted, trace),
        evidence=admitted,
        events=state["events"][:-1],
    )
    assert len(rebuilt) == 3
    rebuilt_result = json.loads(rebuilt[-1]["content"])
    rebuilt_texts = {
        source["text"]
        for group in rebuilt_result["groups"]
        for source in group["sources"]
    }
    assert admitted.by_handle()[state["read_source_handles"][0]]["text"] in rebuilt_texts
    assert [event.node for event in db_session.scalars(
        select(AgentTraceEvent)
        .where(AgentTraceEvent.run_id == response["run_id"])
        .order_by(AgentTraceEvent.sequence_index)
    ) if event.node.startswith("evidence_")] == [
        "evidence_directory_ready",
        "evidence_read",
        "evidence_finalized",
    ]
    answer = db_session.get(AnswerSession, response["answer_session_id"])
    assert answer.model_json["generation_evidence_source_count"] == 1
    from app.services.retrieval_answer_record import replay_answer_bindings
    assert len(replay_answer_bindings(db_session, answer=answer, package=package)) == 1
    answer.diagnostics_json = {
        key: value
        for key, value in answer.diagnostics_json.items()
        if key not in {"generation_evidence_view", "evidence_read_loop_observation_id"}
    }
    flag_modified(answer, "diagnostics_json")
    db_session.flush()
    with pytest.raises(ValueError, match="generation_evidence_view_required"):
        replay_answer_bindings(db_session, answer=answer, package=package)


@pytest.mark.asyncio
async def test_invalid_mid_read_returns_tool_error_then_allows_empty_commit(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from agent_test_support import Admission
    from app.services import evidence_read_loop, layered_execution_v1
    from test_intent_contracts import proposal

    calls = 0
    answer_called = False

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            nonlocal calls
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                return proposal(layer="chunk")
            if "EVIDENCE TOOL SESSION V1" in system_prompt:
                calls += 1
                if calls == 1:
                    return {"tool": "evidence.read", "arguments": {"mid_handles": ["mid_999"]}}
                packet = json.loads(user_prompt)
                error = json.loads(packet["messages"][-1]["content"])
                assert error == {
                    "protocol_version": "evidence_tool_result_v1",
                    "tool": "evidence.read",
                    "status": "error",
                    "error": "mid_handle_not_unread",
                    "field": "arguments.mid_handles",
                }
                return {"tool": "evidence.commit", "arguments": {"source_handles": []}}
            return await super().classify_json(system_prompt, user_prompt, fallback)

        async def complete_text(self, system_prompt, user_prompt, *, max_tokens):
            nonlocal answer_called
            answer_called = True
            return "must not be called"

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(layered_execution_v1, "EmbeddingProvider", fake_model_stack["EmbeddingProvider"])
    monkeypatch.setattr(
        evidence_read_loop,
        "build_mid_directory",
        lambda db, *, plan, package, trace, evidence: _two_mid_directory(evidence, trace),
    )
    monkeypatch.setattr(evidence_read_loop, "DIRECT_CONTEXT_TOKEN_LIMIT", -1)
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=populated_context_graph["knowledge_base"].id,
            question="Summarize the topics.",
            top_k=4,
        ),
        admission=Admission(),
    )
    assert calls == 2
    assert answer_called is False
    assert response["terminal_outcome"] == "insufficient_evidence"
    loop = db_session.scalar(select(AgentObservation).where(
        AgentObservation.run_id == response["run_id"],
        AgentObservation.observation_type == "evidence_read_loop",
    ))
    assert loop.verdict == "finalized"
    assert loop.observation_json["state"]["error_count"] == 1
    assert loop.observation_json["state"]["read_action_count"] == 0
    assert loop.observation_json["state"]["committed_source_handles"] == []


@pytest.mark.asyncio
async def test_evidence_provider_failure_persists_body_free_terminal_state(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from agent_test_support import Admission
    from app.services import evidence_read_loop, layered_execution_v1
    from app.services.reflection_models import AnswerReviewModelError
    from test_intent_contracts import proposal

    class Model(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "INTENT EXECUTION RETRIEVAL V1" in system_prompt:
                return proposal(layer="chunk")
            if "EVIDENCE TOOL SESSION V1" in system_prompt:
                raise RuntimeError("unit-test-provider-failure")
            return await super().classify_json(system_prompt, user_prompt, fallback)

    monkeypatch.setattr(agent_graph, "ChatProvider", Model)
    monkeypatch.setattr(layered_execution_v1, "EmbeddingProvider", fake_model_stack["EmbeddingProvider"])
    monkeypatch.setattr(
        evidence_read_loop,
        "build_mid_directory",
        lambda db, *, plan, package, trace, evidence: _two_mid_directory(evidence, trace),
    )
    monkeypatch.setattr(evidence_read_loop, "DIRECT_CONTEXT_TOKEN_LIMIT", -1)
    with pytest.raises(AnswerReviewModelError, match="evidence_decision_unavailable"):
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                knowledge_base_id=populated_context_graph["knowledge_base"].id,
                question="Summarize the topics.",
                top_k=4,
            ),
            admission=Admission(),
        )
    loop = db_session.scalar(select(AgentObservation).where(
        AgentObservation.observation_type == "evidence_read_loop",
    ))
    assert loop.verdict == "failed"
    assert loop.observation_json["state"]["phase"] == "failed"
    payload = json.dumps(loop.observation_json)
    assert "unit-test-provider-failure" not in payload
    assert populated_context_graph["chunks"][0].text not in payload
