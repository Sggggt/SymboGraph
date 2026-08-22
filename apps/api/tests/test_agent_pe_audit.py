from __future__ import annotations

import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import case, select


def _client():
    from app.db import SessionLocal, get_db
    from app.routers import search

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(search.router, prefix="/api")
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.mark.asyncio
async def test_real_agent_pe_endpoint_matches_every_db_row_field_and_order(
    db_session,
    populated_context_graph,
):
    from app.models import AgentAction, AgentObservation, AgentPlan
    from app.schemas import AgentRequest, SearchFilters
    from app.services import agent_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    result = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=knowledge_base.id,
            question="Explain Bayesian network factorization.",
            filters=SearchFilters(),
            top_k=4,
        ),
    )
    run_id = result["run_id"]
    plans = list(
        db_session.scalars(
            select(AgentPlan)
            .where(AgentPlan.run_id == run_id)
            .order_by(
                AgentPlan.plan_index.asc(),
                AgentPlan.created_at.asc(),
                AgentPlan.id.asc(),
            )
        ).all()
    )
    actions = list(
        db_session.scalars(
            select(AgentAction)
            .outerjoin(AgentPlan, AgentAction.plan_id == AgentPlan.id)
            .where(AgentAction.run_id == run_id)
            .order_by(
                case((AgentPlan.plan_index.is_(None), 1), else_=0).asc(),
                AgentPlan.plan_index.asc(),
                AgentAction.action_index.asc(),
                AgentAction.created_at.asc(),
                AgentAction.id.asc(),
            )
        ).all()
    )
    observations = list(
        db_session.scalars(
            select(AgentObservation)
            .where(AgentObservation.run_id == run_id)
            .order_by(
                AgentObservation.created_at.asc(),
                AgentObservation.id.asc(),
            )
        ).all()
    )
    action_by_id = {str(row.id): row for row in actions}
    package_observations = [
        row
        for row in observations
        if row.observation_type == "context_package_built"
    ]
    assert len(package_observations) == 1
    assert package_observations[0].action_id is not None
    assert (
        action_by_id[str(package_observations[0].action_id)].action_type
        == "build_context_package"
    )
    plans[0].diagnostics_json = {
        **(plans[0].diagnostics_json or {}),
        "api_key": "must-not-leak",
        "nested": {
            "provider_raw_response": {"secret": "must-not-leak"},
            "session_token": "must-not-leak",
        },
    }
    db_session.commit()

    with _client() as client:
        response = client.get(f"/api/agent/runs/{run_id}/pe-audit")
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()

    assert payload["contract_version"] == "agent_pe_audit_public_v1"
    assert payload["counts"] == {
        "plans": len(plans),
        "actions": len(actions),
        "observations": len(observations),
    }
    assert [row["id"] for row in payload["plans"]] == [
        str(row.id) for row in plans
    ]
    assert [row["id"] for row in payload["actions"]] == [
        str(row.id) for row in actions
    ]
    assert [row["id"] for row in payload["observations"]] == [
        str(row.id) for row in observations
    ]
    assert [row["order_index"] for row in payload["plans"]] == list(
        range(len(plans))
    )
    assert [row["order_index"] for row in payload["actions"]] == list(
        range(len(actions))
    )
    assert [row["order_index"] for row in payload["observations"]] == list(
        range(len(observations))
    )

    for public, row in zip(payload["plans"], plans, strict=True):
        assert public["run_id"] == row.run_id
        assert public["knowledge_base_id"] == row.knowledge_base_id
        assert public["retrieval_trace_id"] == row.retrieval_trace_id
        assert public["plan_index"] == row.plan_index
        assert public["status"] == row.status
        assert json.loads(public["query_intent"]["canonical_json"]) == (
            row.query_intent_json or {}
        )
        assert json.loads(public["operating_envelope"]["canonical_json"]) == (
            row.envelope_json or {}
        )
        assert json.loads(public["typed_actions"]["canonical_json"]) == (
            row.typed_actions_json or []
        )
        diagnostics = json.loads(public["diagnostics"]["canonical_json"])
        assert diagnostics["api_key"] == "[REDACTED]"
        assert diagnostics["nested"]["provider_raw_response"] == "[REDACTED]"
        assert diagnostics["nested"]["session_token"] == "[REDACTED]"
        persisted_planner = row.planner_model_json or {}
        assert "raw_output" not in persisted_planner
        assert persisted_planner["provider_response_recorded"] is False
        planner = json.loads(public["planner_model_metadata"]["canonical_json"])
        assert planner == {
            **persisted_planner,
            "provider_response_recorded": "[REDACTED]",
        }
        assert set(planner) == {
            "planner_protocol",
            "typed_action_schema_protocol",
            "planner_audit_protocol",
            "provider_response_recorded",
            "provider_output_hash",
            "proposed_typed_actions",
        }
        assert "raw_output" not in planner
        assert planner["provider_response_recorded"] == "[REDACTED]"
        assert (
            "planner_model_metadata.provider_response_recorded"
            in public["planner_model_metadata"]["redacted_fields"]
        )
        provider_output_hash = planner["provider_output_hash"]
        assert len(provider_output_hash) == 64
        assert all(character in "0123456789abcdef" for character in provider_output_hash)
        assert isinstance(planner["proposed_typed_actions"], list)
        assert "must-not-leak" not in json.dumps(public)

    plan_by_id = {str(row.id): row for row in plans}
    for public, row in zip(payload["actions"], actions, strict=True):
        assert public["run_id"] == row.run_id
        assert public["plan_id"] == row.plan_id
        assert public["plan_index"] == plan_by_id[str(row.plan_id)].plan_index
        assert public["parent_action_id"] == row.parent_action_id
        assert public["action_index"] == row.action_index
        assert public["action_type"] == row.action_type
        assert public["target_ids"] == (row.target_ids_json or [])
        assert public["reason"] == row.reason
        assert json.loads(public["budget_request"]["canonical_json"]) == (
            row.budget_request_json or {}
        )
        assert json.loads(public["expected_evidence"]["canonical_json"]) == (
            row.expected_evidence_json or {}
        )
        assert json.loads(public["stop_condition"]["canonical_json"]) == (
            row.stop_condition_json or {}
        )
        assert json.loads(public["validator"]["payload"]["canonical_json"]) == (
            row.validation_json or {}
        )
        assert json.loads(public["output"]["canonical_json"]) == (
            row.output_json or {}
        )
        assert json.loads(public["diagnostics"]["canonical_json"]) == (
            row.diagnostics_json or {}
        )

    action_by_id = {str(row.id): row for row in actions}
    for public, row in zip(payload["observations"], observations, strict=True):
        linked_action = (
            action_by_id[str(row.action_id)]
            if row.action_id is not None
            else None
        )
        assert public["action_id"] == row.action_id
        assert public["plan_id"] == (
            linked_action.plan_id
            if linked_action is not None
            else (row.observation_json or {})["plan_id"]
        )
        assert public["observation_type"] == row.observation_type
        assert public["evidence_chunk_ids"] == (
            row.evidence_chunk_ids_json or []
        )
        assert public["verdict"] == row.verdict
        assert json.loads(public["observation"]["canonical_json"]) == (
            row.observation_json or {}
        )
        assert json.loads(public["diagnostics"]["canonical_json"]) == (
            row.diagnostics_json or {}
        )
    evaluator_rows = [
        row
        for row in payload["observations"]
        if row["observation_type"] == "evidence_evaluator"
    ]
    assert evaluator_rows
    assert all(row["evaluator_linkage"] for row in evaluator_rows)
    assert payload["provider_raw_response_exposed"] is False
    assert payload["credentials_exposed"] is False


@pytest.mark.parametrize(
    "broken_linkage",
    ["knowledge_base", "plan", "parent_action", "observation_action"],
)
def test_pe_endpoint_fails_closed_for_cross_scope_or_broken_linkage(
    db_session,
    sample_knowledge_base,
    broken_linkage,
):
    from app.models import (
        AgentAction,
        AgentObservation,
        AgentPlan,
        AgentRun,
        KnowledgeBase,
    )

    foreign_kb = KnowledgeBase(
        name=f"foreign-{broken_linkage}",
        description="foreign",
        source_root=f"/tmp/foreign-{broken_linkage}",
    )
    db_session.add(foreign_kb)
    db_session.flush()
    run = AgentRun(
        knowledge_base_id=sample_knowledge_base.id,
        question="audit linkage",
        status="running",
        route="layered_context_graph",
    )
    foreign_run = AgentRun(
        knowledge_base_id=foreign_kb.id,
        question="foreign run",
        status="running",
        route="layered_context_graph",
    )
    db_session.add_all([run, foreign_run])
    db_session.flush()

    local_plan = AgentPlan(
        run_id=run.id,
        knowledge_base_id=(
            foreign_kb.id
            if broken_linkage == "knowledge_base"
            else sample_knowledge_base.id
        ),
        plan_index=0,
        planner_model_json={},
        query_intent_json={},
        envelope_json={},
        typed_actions_json=[],
        validation_json={},
        status="validated",
        diagnostics_json={},
    )
    foreign_plan = AgentPlan(
        run_id=foreign_run.id,
        knowledge_base_id=foreign_kb.id,
        plan_index=0,
        planner_model_json={},
        query_intent_json={},
        envelope_json={},
        typed_actions_json=[],
        validation_json={},
        status="validated",
        diagnostics_json={},
    )
    db_session.add_all([local_plan, foreign_plan])
    db_session.flush()
    if broken_linkage == "plan":
        db_session.add(
            AgentAction(
                run_id=run.id,
                plan_id=foreign_plan.id,
                action_index=0,
                action_type="recall_chunks",
                reason="cross-run plan",
                status="accepted",
            )
        )
    elif broken_linkage == "parent_action":
        parent = AgentAction(
            run_id=foreign_run.id,
            plan_id=foreign_plan.id,
            action_index=0,
            action_type="verify_citations",
            reason="foreign parent",
            status="completed",
        )
        db_session.add(parent)
        db_session.flush()
        db_session.add(
            AgentAction(
                run_id=run.id,
                plan_id=local_plan.id,
                parent_action_id=parent.id,
                action_index=1,
                action_type="repair_missing_citation",
                reason="cross-run parent",
                status="accepted",
            )
        )
    elif broken_linkage == "observation_action":
        foreign_action = AgentAction(
            run_id=foreign_run.id,
            plan_id=foreign_plan.id,
            action_index=0,
            action_type="recall_chunks",
            reason="foreign action",
            status="completed",
        )
        db_session.add(foreign_action)
        db_session.flush()
        db_session.add(
            AgentObservation(
                run_id=run.id,
                action_id=foreign_action.id,
                observation_type="chunk_recall",
                observation_json={},
                verdict="sufficient",
            )
        )
    db_session.commit()

    with _client() as client:
        response = client.get(f"/api/agent/runs/{run.id}/pe-audit")
    assert response.status_code == 409
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == (
        "agent_pe_audit_integrity_failed"
    )


def test_pe_json_payload_round_trip_and_citation_verification_count_contract():
    from pydantic import ValidationError

    from app.schemas import AgentPEJsonPayload, AgentTraceEventPayload

    canonical_json = '{"action_index":0}'
    payload = AgentPEJsonPayload(
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        redacted_fields=[],
    )
    assert AgentPEJsonPayload.model_validate(
        payload.model_dump()
    ).canonical_json == '{"action_index":0}'

    trace = {
        "run_id": "run-1",
        "sequence_index": 0,
        "node": "citation_verification",
        "status": "completed",
        "scores": {
            "citation_pass_rate": 1.0,
            "raw_citation_pass_rate": 1.0,
            "verification_count": 2,
            "returned_citation_count": 2,
            "repair_actions": [],
        },
    }
    parsed = AgentTraceEventPayload.model_validate(trace)
    assert parsed.scores.verification_count == 2
    trace["scores"]["verification_count"] = -1
    with pytest.raises(ValidationError):
        AgentTraceEventPayload.model_validate(trace)
