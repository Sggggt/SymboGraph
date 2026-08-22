from __future__ import annotations

from types import SimpleNamespace

import pytest


def _walk_envelope(*, budget: int = 1, cadence: int = 1) -> dict[str, object]:
    return {
        "agent_mid_top_k": 4,
        "max_depth_per_layer": 3,
        "max_labels_per_node": 4,
        "max_edge_reuse": 1,
        "max_cycle_reward_per_path": 0.2,
        "cycle_reward_distance_threshold": 2.0,
        "path_distance_green_threshold": 0.2,
        "path_distance_gray_threshold": 1.0,
        "path_distance_hard_threshold": 2.0,
        "gray_zone_observation_cadence": cadence,
        "traversal_observation_budget": budget,
    }


def _gray_edge(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"edge-{index}",
        source_concept_id="node-a",
        target_concept_id=f"node-{index}",
        edge_type="dense_semantic",
        distance=0.6,
        raw_strength=0.55,
        weight=0.55,
        features_json={"calibrated_strength": 0.55},
        diagnostics_json={},
        support_chunk_ids_json=[f"chunk-{index}"],
    )


def _run_walk(*, edge_indices: list[int], budget=None, layer: str = "mid"):
    from app.services.context_graph import execute_layer_priority_walk

    edges = [_gray_edge(index) for index in edge_indices]
    node_text = {"node-a": "alpha", **{f"node-{index}": "beta" for index in edge_indices}}
    adjacency = {"node-a": edges, **{f"node-{index}": [edge] for index, edge in zip(edge_indices, edges)}}
    return execute_layer_priority_walk(
        layer=layer,
        entry_scores={"node-a": 1.0},
        node_text_by_id=node_text,
        adjacency=adjacency,
        query_facets={"required_facets": ["beta"], "facet_aliases": {"beta": ["beta"]}},
        source_attr="source_concept_id",
        target_attr="target_concept_id",
        envelope=_walk_envelope(),
        observation_budget=budget,
    )


def test_budget_exhaustion_compacts_packets_without_skipping_local_rule(monkeypatch):
    from app.services import context_graph

    class ForbiddenChatProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("gray-zone observation budgeting must not construct a model")

    monkeypatch.setattr(context_graph, "ChatProvider", ForbiddenChatProvider)
    result = _run_walk(edge_indices=[0, 1, 2])

    decisions = result["gray_zone_path_decisions"]
    assert len(decisions) == 3
    assert all(decision["model_call_count"] == 0 for decision in decisions)
    assert all(decision["matched_rule"] for decision in decisions)
    assert sum("observation" in decision for decision in decisions) == 1
    assert [
        decision["hard_interrupt_state"]["traversal_observation_budget"]["compaction_reason"]
        for decision in decisions
    ] == [None, "traversal_observation_budget_hit", "traversal_observation_budget_hit"]

    convergence = result["convergence"]
    assert convergence["gray_zone_rule_evaluation_count"] == 3
    assert convergence["gray_zone_observation_compacted_count"] == 2
    assert convergence["traversal_observation_expanded_count"] == 1
    assert convergence["traversal_observation_budget_compacted_count"] == 2
    assert convergence["traversal_observation_cadence_compacted_count"] == 0
    assert convergence["traversal_observation_hard_interrupt_count"] == 2
    assert convergence["traversal_observation_budget_hit"] is True
    assert convergence["gray_zone_model_call_count"] == 0


def test_one_budget_ledger_is_shared_across_layer_walks():
    from app.services.context_graph import TraversalObservationBudget

    shared_budget = TraversalObservationBudget(limit=1)
    mid_result = _run_walk(edge_indices=[0], budget=shared_budget, layer="mid")
    coarse_result = _run_walk(edge_indices=[1], budget=shared_budget, layer="coarse")

    assert "observation" in mid_result["gray_zone_path_decisions"][0]
    second = coarse_result["gray_zone_path_decisions"][0]
    assert "observation" not in second
    assert second["observation_compacted"] is True
    assert (
        second["hard_interrupt_state"]["traversal_observation_budget"]["compaction_reason"]
        == "traversal_observation_budget_hit"
    )
    audit = shared_budget.audit()
    assert audit["local_rule_evaluation_count"] == 2
    assert audit["expanded_observation_count"] == 1
    assert audit["budget_compacted_count"] == 1
    assert audit["hard_interrupt_count"] == 1
    assert audit["model_call_count"] == 0


def test_trace_aggregate_gate_replays_budget_transitions_and_rejects_tampering():
    from app.services.context_graph import (
        GrayZoneTraceInvariantError,
        TraversalObservationBudget,
        validate_gray_zone_trace_aggregate_for_persistence,
    )

    shared_budget = TraversalObservationBudget(limit=1)
    result = _run_walk(edge_indices=[0, 1, 2], budget=shared_budget)
    convergence = dict(result["convergence"])
    global_audit = shared_budget.audit()
    convergence.update(
        {
            "traversal_observation_budget_audit": global_audit,
            "traversal_observation_expanded_count": global_audit["expanded_observation_count"],
            "traversal_observation_budget_compacted_count": global_audit["budget_compacted_count"],
            "traversal_observation_cadence_compacted_count": global_audit["cadence_compacted_count"],
            "traversal_observation_hard_interrupt_count": global_audit["hard_interrupt_count"],
            "traversal_observation_budget_hit": global_audit["budget_hit"],
        }
    )
    records = result["gray_zone_path_decisions"]

    assert validate_gray_zone_trace_aggregate_for_persistence(
        records,
        convergence=convergence,
        operating_envelope=_walk_envelope(),
    ) is convergence

    tampered_convergence = {
        **convergence,
        "traversal_observation_hard_interrupt_count": 0,
    }
    with pytest.raises(GrayZoneTraceInvariantError, match="hard_interrupt_count"):
        validate_gray_zone_trace_aggregate_for_persistence(
            records,
            convergence=tampered_convergence,
            operating_envelope=_walk_envelope(),
        )

    tampered_records = [dict(record) for record in records]
    tampered_records[1] = {
        **tampered_records[1],
        "hard_interrupt_state": {
            **tampered_records[1]["hard_interrupt_state"],
            "traversal_observation_budget": {
                **tampered_records[1]["hard_interrupt_state"]["traversal_observation_budget"],
                "compaction_reason": "gray_zone_observation_cadence",
            },
        },
    }
    with pytest.raises(GrayZoneTraceInvariantError, match="budget transition compaction_reason"):
        validate_gray_zone_trace_aggregate_for_persistence(
            tampered_records,
            convergence=convergence,
            operating_envelope=_walk_envelope(),
        )


@pytest.mark.parametrize("invalid", [0, 20_001, True, 1.5, "2"])
def test_budget_ledger_rejects_invalid_limits(invalid):
    from app.services.context_graph import TraversalObservationBudget

    with pytest.raises(ValueError, match="between 1 and 20000"):
        TraversalObservationBudget(limit=invalid)


def test_budget_changes_envelope_traversal_and_cache_identity():
    from app.services.context_graph import (
        context_graph_cache_key,
        context_graph_cache_key_components,
        normalize_agent_operating_envelope,
        traversal_protocol_hash,
    )

    first = normalize_agent_operating_envelope(_walk_envelope(budget=1))
    second = normalize_agent_operating_envelope(_walk_envelope(budget=2))
    assert first["traversal_observation_budget"] == 1
    assert second["traversal_observation_budget"] == 2
    assert traversal_protocol_hash(first) != traversal_protocol_hash(second)

    common = {
        "knowledge_base_id": "kb-budget",
        "query": "bounded gray observation",
        "filters": {},
        "context_state": None,
        "retrieval_mode": "layered_graph",
        "retrieval_granularity": "mid",
        "result_top_k": 3,
        "conversation_state_scope_hash": "c" * 64,
        "query_facets": {"required_facets": ["gray"]},
        "profile_hash_value": "profile-fixed",
        "runtime_settings_hash_value": "runtime-fixed",
    }
    first_components = context_graph_cache_key_components(
        **common,
        operating_envelope=first,
    )
    second_components = context_graph_cache_key_components(
        **common,
        operating_envelope=second,
    )
    assert first_components["traversal_observation_budget"] == 1
    assert second_components["traversal_observation_budget"] == 2
    assert first_components["agent_operating_envelope_hash"] != second_components["agent_operating_envelope_hash"]
    assert first_components["traversal_protocol_hash"] != second_components["traversal_protocol_hash"]
    assert context_graph_cache_key(**common, operating_envelope=first) != context_graph_cache_key(
        **common,
        operating_envelope=second,
    )
