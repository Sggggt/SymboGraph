from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_candidate_pool_dedupe_budget_counts_unique_ids_and_hard_interrupts():
    from app.services.context_graph import CandidatePoolDedupeBudget

    budget = CandidatePoolDedupeBudget(scope="test_pool", limit=2)

    assert budget.admit("candidate-a") is True
    assert budget.admit("candidate-b") is True
    assert budget.admit("candidate-a") is True
    assert budget.admit("candidate-c") is False

    audit = budget.audit()
    assert audit == {
        "protocol_version": "candidate_pool_dedupe_hard_interrupt_v1",
        "scope": "test_pool",
        "limit": 2,
        "attempt_count": 4,
        "unique_admitted_count": 2,
        "duplicate_count": 1,
        "rejected_new_count": 1,
        "budget_hit": True,
        "hard_interrupt_count": 1,
        "rejected_candidate_id_samples": ["candidate-c"],
        "observation_compacted": False,
        "stop_reason": "candidate_pool_dedupe_budget_hit",
    }


def test_candidate_pool_dedupe_zero_budget_rejects_every_new_id():
    from app.services.context_graph import CandidatePoolDedupeBudget

    budget = CandidatePoolDedupeBudget(scope="zero_pool", limit=0)

    assert budget.admit("candidate-a") is False
    assert budget.audit()["unique_admitted_count"] == 0
    assert budget.audit()["hard_interrupt_count"] == 1


@pytest.mark.parametrize(
    ("max_edge_reuse", "expected_expansions", "expected_pruned", "expected_nodes"),
    [
        (1, 1, 1, ["node-a", "node-b"]),
        (0, 0, 1, ["node-a"]),
    ],
)
def test_layer_walk_enforces_path_local_edge_reuse_hard_interrupt(
    max_edge_reuse,
    expected_expansions,
    expected_pruned,
    expected_nodes,
):
    from app.services.context_graph import execute_layer_priority_walk

    edge = SimpleNamespace(
        id="edge-a-b",
        source_concept_id="node-a",
        target_concept_id="node-b",
        edge_type="dense_semantic",
        distance=0.1,
        raw_strength=0.9,
        support_chunk_ids_json=["chunk-1"],
    )
    envelope = {
        "agent_mid_top_k": 8,
        "max_depth_per_layer": 4,
        "max_labels_per_node": 4,
        "max_edge_reuse": max_edge_reuse,
        "max_cycle_reward_per_path": 0.2,
        "cycle_reward_distance_threshold": 2.0,
        "path_distance_green_threshold": 10.0,
        "path_distance_gray_threshold": 11.0,
        "path_distance_hard_threshold": 12.0,
    }

    result = execute_layer_priority_walk(
        layer="mid",
        entry_scores={"node-a": 1.0},
        node_text_by_id={"node-a": "alpha", "node-b": "beta"},
        adjacency={"node-a": [edge], "node-b": [edge]},
        query_facets={"required_facets": []},
        source_attr="source_concept_id",
        target_attr="target_concept_id",
        envelope=envelope,
    )

    convergence = result["convergence"]
    assert result["accepted_nodes"] == expected_nodes
    assert convergence["frontier_expansion_count"] == expected_expansions
    assert convergence["edge_reuse_pruned_count"] == expected_pruned
    assert convergence["max_edge_reuse"] == max_edge_reuse
    assert convergence["edge_reuse_hard_interrupts"][0]["decision"] == "hard_stop_edge_reuse_budget"
    assert convergence["edge_reuse_hard_interrupts"][0]["limit"] == max_edge_reuse
    assert convergence["edge_reuse_hard_interrupts"][0]["protocol_version"] == "edge_reuse_hard_interrupt_v1"
