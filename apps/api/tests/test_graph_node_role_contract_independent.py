"""Adversarial tests for the public coarse-node role contract.

These tests deliberately exercise the production graph payload and the closed
public Pydantic contract.  They are reviewer-owned acceptance tests; they must
not be satisfied by opening ``extra`` fields or by dropping role identities.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError


COARSE_ROLE_FIELDS = (
    "included_mid_concept_ids",
    "boundary_mid_concept_ids",
    "bridge_mid_concept_ids",
    "outlier_mid_concept_ids",
    "low_confidence_mid_concept_ids",
)
COARSE_ALL_MID_FIELD = "all_mid_concept_ids"
COARSE_PUBLIC_ROLE_FIELDS = COARSE_ROLE_FIELDS + (
    COARSE_ALL_MID_FIELD,
)


def _ordered_role_union(node: dict) -> list[str]:
    return list(
        dict.fromkeys(
            str(mid_id)
            for field in COARSE_ROLE_FIELDS
            for mid_id in node[field]
        )
    )


def _coarse_nodes(payload: dict) -> list[dict]:
    nodes = [
        node
        for node in payload["nodes"]
        if node["contract_kind"] == "coarse_concept_node"
    ]
    assert nodes, "the real coarse graph payload returned no coarse nodes"
    return nodes


def _synthetic_complete_coarse_node(template: dict) -> dict:
    node = deepcopy(template)
    node["included_mid_concept_ids"] = [
        "mid:included",
        "mid:shared",
    ]
    node["boundary_mid_concept_ids"] = [
        "mid:boundary",
        "mid:shared",
    ]
    node["bridge_mid_concept_ids"] = [
        "mid:bridge",
    ]
    node["outlier_mid_concept_ids"] = [
        "mid:outlier",
    ]
    node["low_confidence_mid_concept_ids"] = [
        "mid:low-confidence",
        "mid:bridge",
    ]
    node["all_mid_concept_ids"] = [
        "mid:included",
        "mid:shared",
        "mid:boundary",
        "mid:bridge",
        "mid:outlier",
        "mid:low-confidence",
    ]
    return node


@pytest.mark.asyncio
async def test_real_coarse_graph_payload_round_trips_every_role_through_closed_graph_response(
    db_session,
    populated_context_graph,
):
    """The real payload must validate without losing any of its five roles."""

    from app.schemas import GraphResponse
    from app.services.context_graph import graph_layer_payload

    knowledge_base = populated_context_graph["knowledge_base"]
    payload = graph_layer_payload(
        db_session,
        knowledge_base.id,
        "coarse-concepts",
        limit=100,
    )
    raw_nodes = _coarse_nodes(payload)
    for raw in raw_nodes:
        for field in COARSE_PUBLIC_ROLE_FIELDS:
            assert field in raw
            assert isinstance(raw[field], list)
        assert raw["all_mid_concept_ids"] == _ordered_role_union(raw)

    graph = GraphResponse.model_validate(payload)
    validated_nodes = [
        node
        for node in graph.nodes
        if node.contract_kind == "coarse_concept_node"
    ]
    assert len(validated_nodes) == len(raw_nodes)
    raw_by_id = {node["id"]: node for node in raw_nodes}
    for node in validated_nodes:
        raw = raw_by_id[node.id]
        for field in COARSE_PUBLIC_ROLE_FIELDS:
            assert getattr(node, field) == raw[field]
        assert node.all_mid_concept_ids == _ordered_role_union(raw)

    dumped = graph.model_dump(mode="json")
    dumped_nodes = {
        node["id"]: node
        for node in dumped["nodes"]
        if node["contract_kind"] == "coarse_concept_node"
    }
    assert set(dumped_nodes) == set(raw_by_id)
    for node_id, node in dumped_nodes.items():
        for field in COARSE_PUBLIC_ROLE_FIELDS:
            assert field in node
            assert node[field] == raw_by_id[node_id][field]

    round_trip = GraphResponse.model_validate_json(graph.model_dump_json())
    assert round_trip == graph


@pytest.mark.asyncio
async def test_coarse_graph_node_requires_all_roles_and_exact_ordered_deduplicated_union(
    db_session,
    populated_context_graph,
):
    """All role fields are required and ``all_mid`` is their exact union."""

    from app.schemas import GraphNode
    from app.services.context_graph import graph_layer_payload

    knowledge_base = populated_context_graph["knowledge_base"]
    payload = graph_layer_payload(
        db_session,
        knowledge_base.id,
        "coarse-concepts",
        limit=100,
    )
    candidate = _synthetic_complete_coarse_node(
        _coarse_nodes(payload)[0]
    )

    validated = GraphNode.model_validate(candidate)
    assert validated.all_mid_concept_ids == _ordered_role_union(candidate)
    for field in COARSE_PUBLIC_ROLE_FIELDS:
        assert getattr(validated, field) == candidate[field]

    for missing_field in COARSE_PUBLIC_ROLE_FIELDS:
        missing = deepcopy(candidate)
        missing.pop(missing_field)
        with pytest.raises(ValidationError):
            GraphNode.model_validate(missing)

    invalid_unions = [
        candidate["all_mid_concept_ids"] + ["mid:attacker"],
        candidate["all_mid_concept_ids"][:-1],
        candidate["all_mid_concept_ids"]
        + [candidate["all_mid_concept_ids"][0]],
        list(reversed(candidate["all_mid_concept_ids"])),
    ]
    for invalid_union in invalid_unions:
        tampered = deepcopy(candidate)
        tampered["all_mid_concept_ids"] = invalid_union
        with pytest.raises(ValidationError):
            GraphNode.model_validate(tampered)


@pytest.mark.asyncio
async def test_coarse_role_contract_stays_closed_and_cannot_leak_to_other_node_kinds(
    db_session,
    populated_context_graph,
):
    """Known role fields are coarse-only; arbitrary additions stay forbidden."""

    from app.schemas import GraphNode, GraphResponse
    from app.services.context_graph import graph_layer_payload

    knowledge_base = populated_context_graph["knowledge_base"]
    coarse_payload = graph_layer_payload(
        db_session,
        knowledge_base.id,
        "coarse-concepts",
        limit=100,
    )
    coarse = _synthetic_complete_coarse_node(
        _coarse_nodes(coarse_payload)[0]
    )
    unknown = deepcopy(coarse)
    unknown["unknown_mid_concept_role_ids"] = ["mid:attacker"]
    with pytest.raises(ValidationError):
        GraphNode.model_validate(unknown)

    mid_payload = graph_layer_payload(
        db_session,
        knowledge_base.id,
        "mid-concepts",
        limit=100,
    )
    mid_graph = GraphResponse.model_validate(mid_payload)
    mid_dump = mid_graph.model_dump(mode="json")
    assert mid_dump["nodes"]
    for dumped_node in mid_dump["nodes"]:
        assert dumped_node["contract_kind"] != "coarse_concept_node"
        assert all(
            field not in dumped_node
            for field in COARSE_PUBLIC_ROLE_FIELDS
        )

    mid = next(
        deepcopy(node)
        for node in mid_payload["nodes"]
        if node["contract_kind"] == "mid_concept_node"
    )
    for field in COARSE_ROLE_FIELDS:
        mid[field] = []
    mid["all_mid_concept_ids"] = []
    with pytest.raises(ValidationError):
        GraphNode.model_validate(mid)
