import pytest
from sqlalchemy import select

from app.models import AgentObservation
from app.schemas import AgentRequest, SearchFilters
from app.services import agent_graph
from app.services.context_graph import build_context_package
from app.services.layered_execution_v1 import execute_layered_retrieval
from app.services.source_integrity_admission import admit_context_package


@pytest.mark.asyncio
async def test_actual_package_is_admitted_without_scores_or_model_calls(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import layered_execution_v1
    from test_layered_execution_v1 import accepted_plan
    from test_intent_contracts import proposal

    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    kb_id = populated_context_graph["knowledge_base"].id
    plan = accepted_plan(db_session, kb_id, proposal(layer="chunk"))
    request = AgentRequest(knowledge_base_id=kb_id, question=plan.task.question)
    session, run = agent_graph.create_agent_run_context(db_session, request)
    execution = await execute_layered_retrieval(
        db_session,
        plan=plan,
        filters=request.filters,
        top_k=4,
    )
    package = build_context_package(
        db_session,
        knowledge_base_id=kb_id,
        query=plan.task.question,
        trace=execution.trace,
        results=execution.results,
        token_budget=15000,
        restore_per_chunk_budget=2,
    )
    admission = admit_context_package(
        db_session,
        run=run,
        plan=plan,
        package=package,
        filters=SearchFilters(),
        remaining_seconds=60,
    )
    assert admission.passed, {
        key: value
        for key, value in {
            **admission.audit["checks"],
            "provenance_reasons": [
                item["reasons"]
                for item in admission.audit["provenance_audit"]["audits"]
                if not item["valid"]
            ],
        }.items()
        if value is not True and value != []
    }
    assert admission.audit["score_fields_used_for_admission"] == []
    assert admission.audit["model_call_count"] == 0
    row = db_session.get(AgentObservation, admission.observation_id)
    assert row.verdict == "passed"
    assert row.observation_json["context_package_id"] == package.id


@pytest.mark.asyncio
async def test_empty_actual_package_is_insufficient_evidence_not_a_model_judgment(
    db_session,
    sample_knowledge_base,
):
    from app.intent_contracts import IntentPlanningOutput, accept_plan
    from app.models import ContextPackage
    from app.services.intent_planning import retrieval_capability_manifest

    request = AgentRequest(
        knowledge_base_id=sample_knowledge_base.id,
        question="What is supported?",
    )
    session, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities = retrieval_capability_manifest(db_session, sample_knowledge_base.id)
    raw = {
        "intent": {"primary": "fact_lookup"},
        "requirements": [{"id": "f1", "text": "Supported fact"}],
        "execution_strategy": {
            "route": "retrieve",
            "entry_layer": "chunk",
            "semantic_query": "supported fact",
            "generate_lexical": False,
            "hybrid": False,
            "layer_weights": {"chunk": {"dense": 1, "rq": 0, "bm25": 0}},
            "reason_code": "semantic_paraphrase",
        },
    }
    # Construct the task against a synthetic capability so this test isolates
    # package admission rather than graph readiness.
    synthetic = capabilities.model_copy(
        update={
            "available_layers": ("chunk",),
            "available_channels": ("dense",),
            "graph_identity": "a" * 64,
        }
    )
    plan = accept_plan(
        IntentPlanningOutput.model_validate(raw),
        question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash="b" * 64,
        capabilities=synthetic,
    )
    package = ContextPackage(
        knowledge_base_id=sample_knowledge_base.id,
        query=request.question,
        package_json={"chunks": []},
        token_budget=100,
        token_count=0,
    )
    db_session.add(package)
    db_session.flush()
    admission = admit_context_package(
        db_session,
        run=run,
        plan=plan,
        package=package,
        filters=SearchFilters(),
        remaining_seconds=60,
    )
    assert admission.outcome == "insufficient_evidence"
    assert admission.audit["model_call_count"] == 0


@pytest.mark.asyncio
async def test_complete_section_scope_is_materialized_and_replayed_from_raw_spans(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.retrieval_control_contracts import (
        SourceScopeObligation,
        SourceScopeRequest,
        SourceScopeSelector,
    )
    from app.services import intent_execution_agent, layered_execution_v1
    from test_intent_contracts import proposal
    from test_layered_execution_v1 import accepted_plan

    monkeypatch.setattr(
        layered_execution_v1,
        "EmbeddingProvider",
        fake_model_stack["EmbeddingProvider"],
    )
    question = "Summarize all of the Bayesian networks section."
    raw = proposal(layer="chunk")
    raw["requirements"][0]["source_scope"] = SourceScopeObligation(
        mode="complete",
        scope=SourceScopeRequest(
            selector=SourceScopeSelector(
                kind="section",
                reference="Bayesian networks",
                match="title",
            )
        ),
    ).model_dump(mode="json")
    kb_id = populated_context_graph["knowledge_base"].id
    plan = accepted_plan(
        db_session,
        kb_id,
        raw,
        question=question,
    )
    request = AgentRequest(knowledge_base_id=kb_id, question=question, top_k=1)
    _session, run = agent_graph.create_agent_run_context(db_session, request)
    execution = await execute_layered_retrieval(
        db_session,
        plan=plan,
        filters=request.filters,
        top_k=1,
    )
    assert execution.scope_target_chunk_ids
    assert (
        execution.scope_execution_audit["target_plan"][
            "entry_selection_protocol"
        ]
        == "scope_affinity_entry_v1"
    )
    package = intent_execution_agent._build_intent_context_package(
        db_session,
        request=request,
        plan=plan,
        execution=execution,
        token_budget=15000,
    )
    from app.schemas import IntentExecutionRetrievalTraceStepsResponse
    from app.services.retrieval import get_retrieval_trace_steps

    trace_payload = get_retrieval_trace_steps(db_session, execution.trace.id)
    validated_trace = IntentExecutionRetrievalTraceStepsResponse.model_validate(
        trace_payload
    )
    assert validated_trace.steps[-1]["layer"] == "structure"
    assert validated_trace.steps[-1]["action"] == "restore_context_package"
    admission = admit_context_package(
        db_session,
        run=run,
        plan=plan,
        package=package,
        filters=request.filters,
        remaining_seconds=60,
    )
    assert admission.passed, {
        "outcome": admission.outcome,
        "failed_checks": [
            key
            for key, value in admission.audit["checks"].items()
            if key != "source_scope_obligations" and value is not True
        ],
        "scope": admission.audit["source_scope_replay"],
        "invalid_reasons": sorted({
            reason
            for item in admission.audit["provenance_audit"]["audits"]
            if not item["valid"]
            for reason in item.get("reasons") or []
        }),
    }
    scope = admission.audit["source_scope_replay"]
    assert scope["required"] is True
    assert scope["target_chunk_count"] == scope["target_materialized_count"]
    assert admission.audit["checks"]["source_scope_obligations"][0]["state"] == "satisfied"
    assert admission.audit["model_call_count"] == 0
    from app.services.retrieval import get_context_package

    public = get_context_package(db_session, package.id)
    assert public is not None
    package_roles = {
        item["role"] for item in public["package"]["chunks"]
    }
    assert package_roles & {"source_scope", "source_scope_context"}
