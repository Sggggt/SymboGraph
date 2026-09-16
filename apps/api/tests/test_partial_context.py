from copy import deepcopy

import pytest
from sqlalchemy import select

from app.models import AnswerSession, ContextPackage, ContextPackageSourceRetention
from app.schemas import AgentRequest, SearchFilters
from app.services import agent_graph, context_graph
from app.services.chunking import rough_token_count
from app.services.partial_context import prepare_partial_context, partial_context_observation, apply_partial_context
from app.services.agent_reflection import ReflectionContractError
from test_reflection_sources import audit_package


@pytest.mark.asyncio
async def test_useful_partial_sources_survive_full_new_package_with_bounded_repacking(monkeypatch, db_session, populated_context_graph):
    kb = populated_context_graph["knowledge_base"]
    first = await agent_graph.run_agent(db_session, AgentRequest(question="Explain Bayesian networks.", knowledge_base_id=kb.id, top_k=4))
    answer = db_session.get(AnswerSession, first["answer_model_audit"]["answer_session_id"])
    source = db_session.get(ContextPackage, answer.context_package_id)
    request = AgentRequest(question="Compare the definition and its remaining details.", knowledge_base_id=kb.id, session_id=first["session_id"])
    _session, run = agent_graph.create_agent_run_context(db_session, request)
    result = await context_graph.layered_search(db_session, kb.id, request.question, SearchFilters(), 3, allow_cache_read=False)
    candidate = context_graph.build_context_package(db_session, knowledge_base_id=kb.id, query=request.question, trace=result.trace,
        results=result.results, token_budget=64, restore_per_chunk_budget=2)
    candidate_ids = {item["chunk_id"] for item in candidate.package_json["chunks"]}
    useful = next(item for item in source.package_json["chunks"] if item["chunk_id"] not in candidate_ids
        and 64 - candidate.token_count < rough_token_count(item["content"]) < 64)
    original = deepcopy(candidate.package_json)
    prepare_partial_context(db_session, run=run, source=source, source_answer=answer,
        evaluator={"verdict": "insufficient", "referenced_chunk_ids": [useful["chunk_id"]], "input_hash": "a" * 64, "output_hash": "b" * 64})
    hints = partial_context_observation(db_session, run=run, query_facets={"required_facets": ["definition details"]}, granularity="mid")
    assert hints[0]["bounded_graph_observation"]["result_count"] == 0
    assert hints[0]["bounded_graph_observation"]["candidate_chunk_span_summaries"][0]["chunk_id"] == useful["chunk_id"]
    final, contexts = apply_partial_context(db_session, run=run, candidate=candidate, token_budget=64)
    assert final.token_count <= 64 and candidate.package_json == original
    carried = next(item for item in contexts if item["chunk_id"] == useful["chunk_id"])
    assert carried["content"] == useful["content"]
    assert useful["chunk_id"] not in final.hit_chunk_ids_json
    assert db_session.scalar(select(ContextPackageSourceRetention).where(ContextPackageSourceRetention.target_context_package_id == final.id)) is not None
    assert audit_package(db_session, final)["all_valid"]
    assert final.token_budget == 64
    assert final.diagnostics_json["token_budget_audit"]["selection_token_budget"] < final.token_budget
    import importlib
    from test_quality_gate_scripts import SCRIPTS_ROOT, _load_quality_gate
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    checker = importlib.import_module("check_context_package_quality")
    quality = _load_quality_gate().audit_context_package_quality(checker.persisted_context_package_quality_snapshot(db_session, final.id))
    assert quality["pass"], quality["findings"]
    assert run.metadata_json["partial_context_carry"]["status"] == "applied"
    repeated, _ = apply_partial_context(db_session, run=run, candidate=candidate, token_budget=64)
    assert repeated.id == final.id
    run.metadata_json = {**run.metadata_json, "partial_context_carry": {**run.metadata_json["partial_context_carry"], "intent_hash": "f" * 64}}
    with pytest.raises(ReflectionContractError, match="intent_identity_changed"):
        apply_partial_context(db_session, run=run, candidate=candidate, token_budget=64)


@pytest.mark.asyncio
async def test_partial_context_rejects_cross_conversation_source(db_session, populated_context_graph):
    kb = populated_context_graph["knowledge_base"]
    first = await agent_graph.run_agent(db_session, AgentRequest(question="What is a Bayesian network?", knowledge_base_id=kb.id))
    answer = db_session.get(AnswerSession, first["answer_model_audit"]["answer_session_id"])
    source = db_session.get(ContextPackage, answer.context_package_id)
    _session, run = agent_graph.create_agent_run_context(db_session, AgentRequest(question="A new conversation.", knowledge_base_id=kb.id))
    with pytest.raises(ReflectionContractError, match="source_scope_invalid"):
        prepare_partial_context(db_session, run=run, source=source, source_answer=answer,
            evaluator={"verdict": "insufficient", "referenced_chunk_ids": [source.hit_chunk_ids_json[0]], "input_hash": "a" * 64, "output_hash": "b" * 64})


@pytest.mark.asyncio
@pytest.mark.usefixtures('historical_answer_executor')
async def test_historical_insufficient_reuse_carries_partial_evidence_through_agent_and_prompt(monkeypatch, db_session, populated_context_graph, fake_model_stack):
    import json
    from app.schemas import AgentResponse
    kb = populated_context_graph["knowledge_base"]
    first = await agent_graph.run_agent(db_session, AgentRequest(question="Explain Bayesian networks.", knowledge_base_id=kb.id, top_k=4))
    source = db_session.get(ContextPackage, first["context_package_id"])
    required_ids = [item["chunk_id"] for item in source.package_json["chunks"]]
    required_texts = {item["content"] for item in source.package_json["chunks"]}
    async def partial_reuse(**kwargs):
        return {"protocol_version": "verified_context_reuse_evaluator_v3", "verdict": "insufficient",
            "reason": "Useful definition evidence needs additional details.", "referenced_chunk_ids": required_ids,
            "expected_answer_shape": "comparison", "model_call_count": 1, "input_hash": "a" * 64, "output_hash": "b" * 64}
    original_builder = agent_graph.build_context_package
    def small_selection(*args, **kwargs):
        kwargs["reserved_token_budget"] = kwargs["token_budget"] - 5
        return original_builder(*args, **kwargs)
    original_planner = agent_graph.propose_agent_plan
    planner_hints, evidence_packets = [], []
    async def planner(*args, **kwargs):
        planner_hints.extend(kwargs.get("bounded_observations") or [])
        return await original_planner(*args, **kwargs)
    class CaptureModel(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            if "IMMUTABLE ANSWER REFLECTION SYSTEM ENVELOPE" in system_prompt:
                packet = json.loads(user_prompt)
                if packet.get("answer_draft") is None:
                    evidence_packets.append({item["text"] for item in packet["evidence"]})
            return await super().classify_json(system_prompt, user_prompt, fallback)
    monkeypatch.setattr(agent_graph, "evaluate_verified_context_reuse", partial_reuse)
    monkeypatch.setattr(agent_graph, "build_context_package", small_selection)
    monkeypatch.setattr(agent_graph, "propose_agent_plan", planner)
    monkeypatch.setattr(agent_graph, "ChatProvider", CaptureModel)
    result = AgentResponse.model_validate(await agent_graph.run_agent(db_session, AgentRequest(
        question="Compare this definition with the remaining details.", knowledge_base_id=kb.id, session_id=first["session_id"])))
    assert result.route == "layered_context_graph"
    assert planner_hints and any(item.get("candidate_span_summaries") for item in planner_hints)
    assert evidence_packets and required_texts.issubset(evidence_packets[0])
    final = db_session.get(ContextPackage, result.context_package_id)
    assert set(required_ids).issubset({item["chunk_id"] for item in final.package_json["chunks"]})
    assert db_session.scalar(select(ContextPackageSourceRetention).where(ContextPackageSourceRetention.target_context_package_id == final.id)) is not None
    assert audit_package(db_session, final)["all_valid"]
