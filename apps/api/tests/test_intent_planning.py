import json

import pytest
from sqlalchemy import select

from app.models import AgentObservation, AgentRun
from app.retrieval_control_contracts import control_hash
from app.schemas import AgentRequest
from app.services import agent_graph
from app.services.intent_planning import (
    normalize_planning_output,
    plan_intent_execution,
    retrieval_capability_manifest,
)


def test_local_normalization_removes_only_closed_schema_noise():
    raw = {
        "execution_strategy": {"budget_request": {"rq_candidates": 0}},
        "requirements": [
            {
                "role": "summary",
                "mode": "overlap",
                "source_scope": {
                    "op": "coverage",
                    "scope": {
                        "op": "scope",
                        "selector": {
                            "kind": "section",
                            "reference": "Quoted section",
                            "match": "title",
                            "role": "summary",
                        },
                        "children": [
                            {
                                "op": "scope",
                                "selector": {
                                    "kind": "section",
                                    "reference": "冗余",
                                    "match": "title",
                                },
                                "children": [],
                            }
                        ],
                    },
                    "mode": "overlap",
                    "children": [],
                },
            }
        ]
    }
    normalized, audit = normalize_planning_output(raw)
    requirement = normalized["requirements"][0]
    assert requirement["role"] == "topic"
    assert "role" not in requirement["source_scope"]["scope"]["selector"]
    assert audit["role_aliases_normalized"] == 1
    assert "mode" not in requirement
    assert audit["misplaced_requirement_scope_modes_removed"] == 1
    assert audit["removed_selector_roles"] == 1
    assert requirement["source_scope"]["scope"]["children"] == []
    assert audit["scope_shape_noise_removed"] == 1
    assert normalized["execution_strategy"]["budget_request"] == {}
    assert audit["zero_budget_hints_removed"] == 1
    assert raw["requirements"][0]["role"] == "summary"


def test_local_normalization_maps_generic_section_titles_to_declared_source_roles():
    raw = {
        "requirements": [
            {
                "role": "quantity",
                "source_roles": ["summary"],
                "source_scope": {
                    "op": "coverage",
                    "scope": {
                        "op": "scope",
                        "selector": {
                            "kind": "section",
                            "reference": "摘要",
                            "match": "title",
                        },
                        "children": [],
                    },
                    "mode": "complete",
                    "children": [],
                },
            }
        ]
    }

    normalized, audit = normalize_planning_output(raw)
    selector = normalized["requirements"][0]["source_scope"]["scope"]["selector"]
    assert selector["match"] == "role"
    assert selector["role"] == "summary"
    assert audit["section_role_selectors_normalized"] == 1
    assert raw["requirements"][0]["source_scope"]["scope"]["selector"] == {
        "kind": "section",
        "reference": "摘要",
        "match": "title",
    }

    without_declared_role = {
        **raw,
        "requirements": [
            {
                **raw["requirements"][0],
                "source_roles": [],
            }
        ],
    }
    normalized, audit = normalize_planning_output(without_declared_role)
    selector = normalized["requirements"][0]["source_scope"]["scope"]["selector"]
    assert selector["match"] == "role" and selector["role"] == "summary"
    assert audit["section_role_selectors_normalized"] == 1


def test_local_normalization_removes_obligation_shape_noise():
    valid_request = {
        "op": "scope",
        "selector": {
            "kind": "section",
            "reference": "Named section",
            "match": "title",
        },
        "children": [],
    }
    raw = {
        "requirements": [
            {
                "source_scope": {
                    "op": "coverage",
                    "scope": valid_request,
                    "mode": "overlap",
                    "children": [
                        {
                            "op": "coverage",
                            "scope": valid_request,
                            "mode": "overlap",
                            "children": [],
                        }
                    ],
                }
            }
        ]
    }
    normalized, audit = normalize_planning_output(raw)
    assert normalized["requirements"][0]["source_scope"]["children"] == []
    assert audit["scope_shape_noise_removed"] == 1

    misplaced_selector = {
        "requirements": [
            {
                "source_scope": {
                    "op": "coverage",
                    "scope": None,
                    "mode": "overlap",
                    "children": [
                        {
                            "kind": "section",
                            "reference": "详细说明",
                            "match": "role",
                            "role": "detail",
                        }
                    ],
                }
            }
        ]
    }
    normalized, audit = normalize_planning_output(misplaced_selector)
    scope = normalized["requirements"][0]["source_scope"]
    assert scope["children"] == []
    assert scope["scope"]["selector"]["role"] == "detail"
    assert audit["scope_shape_noise_removed"] == 1


def test_local_normalization_closes_direct_route_noise():
    raw = {
        "intent": {
            "primary": "system_capability",
            "secondary": ["explain"],
        },
        "requirements": [{"id": "f1", "text": "redundant"}],
        "execution_strategy": {
            "route": "system_capability",
            "entry_layer": "chunk",
            "semantic_query": "redundant",
            "generate_lexical": True,
            "lexical_groups": [
                {
                    "group_id": "l1",
                    "requirement_ids": ["f1"],
                    "kind": "concept",
                    "surfaces": [
                        {
                            "text": "redundant",
                            "language": "en",
                            "provenance": "model_query",
                        }
                    ],
                }
            ],
            "hybrid": True,
            "layer_weights": {
                "chunk": {"dense": 0.5, "rq": 0.0, "bm25": 0.5}
            },
            "budget_request": {"root_entries": 1},
            "reason_code": "system_request",
        },
    }
    normalized, audit = normalize_planning_output(raw)
    assert normalized["requirements"] == []
    assert normalized["intent"]["secondary"] == []
    strategy = normalized["execution_strategy"]
    assert strategy["entry_layer"] is None
    assert strategy["semantic_query"] == ""
    assert strategy["generate_lexical"] is False
    assert strategy["lexical_groups"] == []
    assert strategy["hybrid"] is False
    assert strategy["layer_weights"] == {}
    assert strategy["budget_request"] == {}
    assert audit["direct_route_noise_removed"] > 0


def test_capability_manifest_reports_no_fake_graph_or_index(db_session, sample_knowledge_base):
    manifest = retrieval_capability_manifest(db_session, sample_knowledge_base.id)
    assert manifest.available_layers == ()
    assert manifest.available_channels == ()
    assert manifest.graph_identity is None
    assert manifest.lexical_identity is None


def test_capability_manifest_reads_the_admitted_graph_layers(
    db_session,
    populated_context_graph,
):
    manifest = retrieval_capability_manifest(
        db_session,
        populated_context_graph["knowledge_base"].id,
    )
    assert manifest.available_layers == ("coarse", "mid", "chunk")
    assert manifest.available_channels == ("dense", "rq")
    assert manifest.graph_identity == populated_context_graph["state"].context_graph_hash
    assert manifest.lexical_identity is None


@pytest.mark.asyncio
async def test_one_shot_plan_persists_prepared_and_completed_audit(
    db_session,
    sample_knowledge_base,
):
    request = AgentRequest(
        knowledge_base_id=sample_knowledge_base.id,
        question="你能做什么？",
    )
    session, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities = retrieval_capability_manifest(db_session, sample_knowledge_base.id)
    calls = []

    class Provider:
        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            calls.append((system_prompt, user_prompt, fallback))
            prepared = db_session.scalar(
                select(AgentObservation).where(
                    AgentObservation.run_id == run.id,
                    AgentObservation.observation_type == "intent_execution_plan",
                )
            )
            assert prepared is not None and prepared.verdict == "prepared"
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

    accepted, audit = await plan_intent_execution(
        db_session,
        run=run,
        question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
        history_summary="",
        capabilities=capabilities,
        verified_context_reuse_available=True,
        provider_factory=Provider,
    )
    db_session.refresh(run)
    observation = db_session.scalar(
        select(AgentObservation).where(
            AgentObservation.run_id == run.id,
            AgentObservation.observation_type == "intent_execution_plan",
        )
    )
    assert len(calls) == 1 and calls[0][2] is None
    assert json.loads(calls[0][1])["verified_context_reuse_available"] is True
    assert accepted.strategy.route == "system_capability"
    assert observation.verdict == "completed"
    assert observation.observation_json == audit
    assert run.metadata_json["intent_execution_plan"]["accepted_plan_hash"] == accepted.identity
    assert db_session.scalar(select(AgentRun).where(AgentRun.id == run.id)) is run
