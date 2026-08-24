from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
import math
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_rq_prefix_pair_diagnostics_persist_all_five_grounded_types(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import (
        ChunkRelationEdge,
        ChunkRelationGraphState,
        RQPrefix,
        RQPrefixMembership,
        RQPrefixPairDiagnostic,
    )
    from app.services.context_graph import (
        RQ_PREFIX_PAIR_DIAGNOSTIC_EDGE_TYPES,
        RQ_PREFIX_PAIR_DIAGNOSTIC_PROTOCOL_VERSION,
        _edge_calibrated_strength,
        rq_prefix_pair_diagnostic_protocol_hash,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    relation_state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert relation_state is not None
    prefixes = list(
        db_session.scalars(
            select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    prefix_by_id = {prefix.id: prefix for prefix in prefixes}
    memberships = list(
        db_session.scalars(
            select(RQPrefixMembership)
            .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
            .where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    score_by_chunk_prefix = {
        (str(row.chunk_id), str(row.rq_prefix_id)): float(row.membership_score)
        for row in memberships
    }
    rows = list(
        db_session.scalars(
            select(RQPrefixPairDiagnostic)
            .where(RQPrefixPairDiagnostic.graph_state_id == relation_state.id)
            .order_by(
                RQPrefixPairDiagnostic.edge_type,
                RQPrefixPairDiagnostic.diagnostic_hash,
            )
        ).all()
    )
    assert rows
    assert {row.edge_type for row in rows} == set(
        RQ_PREFIX_PAIR_DIAGNOSTIC_EDGE_TYPES
    )
    assert len(
        {
            (
                row.source_rq_prefix_id,
                row.target_rq_prefix_id,
                row.edge_type,
            )
            for row in rows
        }
    ) == len(rows)
    assert all(
        row.protocol_version == RQ_PREFIX_PAIR_DIAGNOSTIC_PROTOCOL_VERSION
        and len(row.diagnostic_hash) == 64
        and (row.diagnostics_json or {}).get("protocol_hash")
        == rq_prefix_pair_diagnostic_protocol_hash()
        and (row.diagnostics_json or {}).get("model_call_count") == 0
        and (row.diagnostics_json or {}).get("diagnostic_only") is True
        and (row.diagnostics_json or {}).get("active_relation_edge") is False
        for row in rows
    )

    for row in rows:
        source = prefix_by_id[row.source_rq_prefix_id]
        target = prefix_by_id[row.target_rq_prefix_id]
        assert source.id != target.id
        if row.edge_type == "parent_child":
            assert target.parent_rq_prefix_id == source.id
        else:
            assert source.rq_level == target.rq_level
        if row.edge_type == "sibling":
            assert source.parent_rq_prefix_id == target.parent_rq_prefix_id

    projected_rows = [
        row for row in rows if row.edge_type == "projected_chunk_support"
    ]
    assert projected_rows
    for row in projected_rows:
        support_edges = list(
            db_session.scalars(
                select(ChunkRelationEdge).where(
                    ChunkRelationEdge.id.in_(row.support_chunk_edge_ids_json)
                )
            ).all()
        )
        assert len(support_edges) == len(row.support_chunk_edge_ids_json)
        expected_mass = 0.0
        expected_weighted_strength_mass = 0.0
        for edge in support_edges:
            direct = score_by_chunk_prefix.get(
                (str(edge.source_chunk_id), str(row.source_rq_prefix_id)), 0.0
            ) * score_by_chunk_prefix.get(
                (str(edge.target_chunk_id), str(row.target_rq_prefix_id)), 0.0
            )
            reverse = score_by_chunk_prefix.get(
                (str(edge.source_chunk_id), str(row.target_rq_prefix_id)), 0.0
            ) * score_by_chunk_prefix.get(
                (str(edge.target_chunk_id), str(row.source_rq_prefix_id)), 0.0
            )
            membership_mass = direct + reverse
            expected_mass += membership_mass
            expected_weighted_strength_mass += (
                membership_mass * float(_edge_calibrated_strength(edge))
            )
        assert row.support_membership_mass == pytest.approx(
            expected_mass,
            rel=1e-11,
            abs=1e-14,
        )
        assert row.diagnostic_strength == pytest.approx(
            expected_weighted_strength_mass / expected_mass,
            rel=1e-11,
            abs=1e-14,
        )

    active_rq_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id,
                ChunkRelationEdge.edge_type.like("rq_%"),
            )
        ).all()
    )
    assert active_rq_edges == []
    diagnostics = (relation_state.diagnostics_json or {})[
        "rq_prefix_pair_diagnostics"
    ]
    assert diagnostics["diagnostic_count"] == len(rows)
    assert diagnostics["diagnostic_count_by_type"] == dict(
        sorted(Counter(row.edge_type for row in rows).items())
    )
    assert diagnostics["active_relation_edge_count"] == 0
    assert diagnostics["fallback_pair_edges_created"] == 0
    assert diagnostics["model_call_count"] == 0
    assert relation_state.diagnostics_json[
        "rq_prefix_pair_diagnostics_hash"
    ] == diagnostics["diagnostic_hash"]


@pytest.mark.asyncio
async def test_rq_prefix_pair_diagnostic_formulas_match_the_exact_domain(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import (
        ChunkRelationGraphState,
        RQPrefix,
        RQPrefixMembership,
        RQPrefixPairDiagnostic,
    )
    from app.services.context_graph import RQ_PREFIX_CENTROID_NEIGHBORS

    knowledge_base = populated_context_graph["knowledge_base"]
    relation_state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert relation_state is not None
    prefixes = list(
        db_session.scalars(
            select(RQPrefix)
            .where(RQPrefix.graph_state_id == relation_state.id)
            .order_by(RQPrefix.rq_level, RQPrefix.rq_prefix_key)
        ).all()
    )
    prefix_by_id = {str(prefix.id): prefix for prefix in prefixes}
    prefixes_by_level: dict[int, list] = defaultdict(list)
    for prefix in prefixes:
        prefixes_by_level[int(prefix.rq_level or 0)].append(prefix)

    membership_by_prefix: dict[str, dict[str, float]] = defaultdict(dict)
    memberships = list(
        db_session.scalars(
            select(RQPrefixMembership)
            .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
            .where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    for membership in memberships:
        membership_by_prefix[str(membership.rq_prefix_id)][
            str(membership.chunk_id)
        ] = float(membership.membership_score)

    rows = list(
        db_session.scalars(
            select(RQPrefixPairDiagnostic).where(
                RQPrefixPairDiagnostic.graph_state_id == relation_state.id
            )
        ).all()
    )
    rows_by_fact = {
        (
            str(prefix_by_id[str(row.source_rq_prefix_id)].rq_prefix_key),
            str(prefix_by_id[str(row.target_rq_prefix_id)].rq_prefix_key),
            str(row.edge_type),
        ): row
        for row in rows
    }

    def ordered_pair(left, right):
        return (
            (left, right)
            if str(left.rq_prefix_key) < str(right.rq_prefix_key)
            else (right, left)
        )

    def centroid_distance(left, right) -> float:
        left_values = [float(value) for value in (left.centroid_json or [])]
        right_values = [float(value) for value in (right.centroid_json or [])]
        width = max(len(left_values), len(right_values))
        left_values.extend([0.0] * (width - len(left_values)))
        right_values.extend([0.0] * (width - len(right_values)))
        return math.sqrt(
            sum(
                (left_value - right_value) ** 2
                for left_value, right_value in zip(
                    left_values,
                    right_values,
                    strict=True,
                )
            )
        )

    def exact_median(values: list[float]) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 1.0
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    expected_parent_child: set[tuple[str, str]] = set()
    for child in prefixes:
        if int(child.rq_level or 0) <= 1:
            continue
        parent = prefix_by_id[str(child.parent_rq_prefix_id)]
        fact_key = (
            str(parent.rq_prefix_key),
            str(child.rq_prefix_key),
            "parent_child",
        )
        row = rows_by_fact[fact_key]
        expected_parent_child.add(fact_key[:2])
        child_mass = sum(membership_by_prefix[str(child.id)].values())
        parent_mass = sum(membership_by_prefix[str(parent.id)].values())
        expected_strength = child_mass / parent_mass if parent_mass > 0.0 else 0.0
        assert row.support_membership_mass == pytest.approx(
            child_mass,
            rel=1e-12,
            abs=1e-14,
        )
        assert row.diagnostic_strength == pytest.approx(
            max(0.0, min(1.0, expected_strength)),
            rel=1e-12,
            abs=1e-14,
        )
    assert {
        (source_key, target_key)
        for source_key, target_key, edge_type in rows_by_fact
        if edge_type == "parent_child"
    } == expected_parent_child

    expected_siblings: set[tuple[str, str]] = set()
    expected_centroid_near: set[tuple[str, str]] = set()
    for level, level_prefixes in prefixes_by_level.items():
        pair_distances: dict[tuple[str, str], float] = {}
        positive_distances: list[float] = []
        for left_index, left in enumerate(level_prefixes):
            for right in level_prefixes[left_index + 1 :]:
                source, target = ordered_pair(left, right)
                pair_key = (
                    str(source.rq_prefix_key),
                    str(target.rq_prefix_key),
                )
                distance = centroid_distance(source, target)
                pair_distances[pair_key] = distance
                if distance > 0.0:
                    positive_distances.append(distance)
        level_tau = max(exact_median(positive_distances), 1e-12)

        for source in level_prefixes:
            candidates: list[tuple[float, str, object]] = []
            for target in level_prefixes:
                if str(source.id) == str(target.id):
                    continue
                left, right = ordered_pair(source, target)
                pair_key = (
                    str(left.rq_prefix_key),
                    str(right.rq_prefix_key),
                )
                candidates.append(
                    (
                        pair_distances[pair_key],
                        str(target.rq_prefix_key),
                        target,
                    )
                )
            for _distance, _target_key, target in sorted(
                candidates,
                key=lambda item: (item[0], item[1]),
            )[:RQ_PREFIX_CENTROID_NEIGHBORS]:
                left, right = ordered_pair(source, target)
                expected_centroid_near.add(
                    (str(left.rq_prefix_key), str(right.rq_prefix_key))
                )

        for left_index, left in enumerate(level_prefixes):
            for right in level_prefixes[left_index + 1 :]:
                source, target = ordered_pair(left, right)
                pair_key = (
                    str(source.rq_prefix_key),
                    str(target.rq_prefix_key),
                )
                distance = pair_distances[pair_key]
                expected_centroid_strength = math.exp(-distance / level_tau)
                left_members = membership_by_prefix[str(source.id)]
                right_members = membership_by_prefix[str(target.id)]
                shared_chunk_ids = set(left_members).intersection(right_members)
                overlap_mass = sum(
                    min(left_members[chunk_id], right_members[chunk_id])
                    for chunk_id in shared_chunk_ids
                )

                if str(source.parent_rq_prefix_id or "rq:root") == str(
                    target.parent_rq_prefix_id or "rq:root"
                ):
                    expected_siblings.add(pair_key)
                    sibling_row = rows_by_fact[(*pair_key, "sibling")]
                    assert sibling_row.diagnostic_strength == pytest.approx(
                        expected_centroid_strength,
                        rel=1e-12,
                        abs=1e-14,
                    )
                    assert sibling_row.support_membership_mass == pytest.approx(
                        overlap_mass,
                        rel=1e-12,
                        abs=1e-14,
                    )
                    assert (sibling_row.diagnostics_json or {})["formula_inputs"][
                        "level_distance_tau"
                    ] == pytest.approx(level_tau, rel=1e-12, abs=1e-14)

                if pair_key in expected_centroid_near:
                    centroid_row = rows_by_fact[(*pair_key, "centroid_near")]
                    assert centroid_row.diagnostic_strength == pytest.approx(
                        expected_centroid_strength,
                        rel=1e-12,
                        abs=1e-14,
                    )
                    assert (centroid_row.diagnostics_json or {})["formula_inputs"][
                        "rq_level"
                    ] == level

    for edge_type, expected_pairs in (
        ("sibling", expected_siblings),
        ("centroid_near", expected_centroid_near),
    ):
        assert {
            (source_key, target_key)
            for source_key, target_key, actual_type in rows_by_fact
            if actual_type == edge_type
        } == expected_pairs


@pytest.mark.asyncio
async def test_rq_prefix_pair_diagnostics_are_idempotent_queryable_and_in_packets(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import (
        ChunkRelationEdge,
        ChunkRelationGraphState,
        RQPrefix,
        RQPrefixMembership,
        RQPrefixPairDiagnostic,
    )
    from app.services.context_graph import (
        RQ_PREFIX_PAIR_EDGE_ID_SAMPLE_LIMIT,
        build_rq_prefix_pair_diagnostics,
        concept_packet_for_cluster,
        graph_layer_payload,
        stable_hash,
    )
    from app.schemas import GraphResponse

    knowledge_base = populated_context_graph["knowledge_base"]
    relation_state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert relation_state is not None
    prefixes = list(
        db_session.scalars(
            select(RQPrefix)
            .where(RQPrefix.graph_state_id == relation_state.id)
            .order_by(RQPrefix.rq_level, RQPrefix.rq_prefix_key)
        ).all()
    )
    prefixes_by_key = {
        (
            int(prefix.rq_level or 0),
            tuple(int(value) for value in (prefix.rq_path_prefix or [])),
        ): prefix
        for prefix in prefixes
    }
    memberships = list(
        db_session.scalars(
            select(RQPrefixMembership)
            .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
            .where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    relation_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id
            )
        ).all()
    )
    edges = {
        (edge.source_chunk_id, edge.target_chunk_id, edge.edge_type): edge
        for edge in relation_edges
    }
    before = list(
        db_session.scalars(
            select(RQPrefixPairDiagnostic).where(
                RQPrefixPairDiagnostic.graph_state_id == relation_state.id
            )
        ).all()
    )
    before_ids = {row.id for row in before}
    before_hashes = {row.diagnostic_hash for row in before}
    state_hash_before = relation_state.state_hash

    retried = build_rq_prefix_pair_diagnostics(
        db_session,
        relation_state,
        prefixes_by_key,
        memberships,
        edges,
    )

    assert {row.id for row in retried} == before_ids
    assert {row.diagnostic_hash for row in retried} == before_hashes
    assert relation_state.state_hash == state_hash_before
    assert relation_state.diagnostics_json["rq_prefix_pair_diagnostics"][
        "idempotent_reuse"
    ] is True

    l3_prefix = next(prefix for prefix in prefixes if prefix.rq_level == 3)
    packet = concept_packet_for_cluster(db_session, l3_prefix)
    incident_rows = [
        row
        for row in before
        if l3_prefix.id
        in {row.source_rq_prefix_id, row.target_rq_prefix_id}
    ]
    assert packet["rq_prefix_pair_diagnostic_count"] == len(incident_rows)
    assert packet["rq_prefix_pair_diagnostics_protocol_hash"]
    assert packet["rq_prefix_pair_diagnostics_hash"]
    assert all(
        item["diagnostics"]["diagnostic_only"] is True
        for item in packet["rq_prefix_pair_diagnostics"]
    )
    assert "rq_path" not in json.dumps(packet, sort_keys=True)

    graph = graph_layer_payload(
        db_session,
        knowledge_base.id,
        "chunk-relation",
        limit=500,
    )
    pair_edges = [
        edge
        for edge in graph["edges"]
        if edge.get("type") == "rq_prefix_pair_diagnostic"
    ]
    assert pair_edges
    GraphResponse.model_validate(graph)
    assert all(
        edge["metadata"]["diagnostic_only"] is True
        and edge["metadata"]["active_relation_edge"] is False
        and edge["metadata"]["model_call_count"] == 0
        if "model_call_count" in edge["metadata"]
        else edge["metadata"]["diagnostics"]["model_call_count"] == 0
        for edge in pair_edges
    )

    # Consumer contracts expose a fixed-size audit sample while preserving the
    # full cardinality and identity hash.  Full support stays in PostgreSQL.
    sampled_diagnostic_id = str(
        packet["rq_prefix_pair_diagnostics"][0]["diagnostic_id"]
    )
    bounded_row = next(
        row for row in incident_rows if str(row.id) == sampled_diagnostic_id
    )
    original_support_edge_ids = list(
        bounded_row.support_chunk_edge_ids_json or []
    )
    oversized_support_edge_ids = [f"bounded-edge-{index:02d}" for index in range(30)]
    bounded_row.support_chunk_edge_ids_json = oversized_support_edge_ids
    db_session.flush()
    try:
        bounded_packet = concept_packet_for_cluster(db_session, l3_prefix)
        bounded_packet_row = next(
            item
            for item in bounded_packet["rq_prefix_pair_diagnostics"]
            if str(item["diagnostic_id"]) == str(bounded_row.id)
        )
        assert len(bounded_packet_row["support_chunk_edge_ids"]) == (
            RQ_PREFIX_PAIR_EDGE_ID_SAMPLE_LIMIT
        )
        assert bounded_packet_row["support_chunk_edge_ids"] == (
            oversized_support_edge_ids[:RQ_PREFIX_PAIR_EDGE_ID_SAMPLE_LIMIT]
        )
        assert bounded_packet_row["support_chunk_edge_count"] == len(
            oversized_support_edge_ids
        )
        assert bounded_packet_row["support_chunk_edge_ids_hash"] == stable_hash(
            oversized_support_edge_ids
        )

        bounded_graph = graph_layer_payload(
            db_session,
            knowledge_base.id,
            "chunk-relation",
            limit=500,
        )
        bounded_graph_row = next(
            edge
            for edge in bounded_graph["edges"]
            if edge.get("id") == f"rq-prefix-diagnostic:{bounded_row.id}"
        )
        assert len(bounded_graph_row["metadata"]["support_chunk_edge_ids"]) == (
            RQ_PREFIX_PAIR_EDGE_ID_SAMPLE_LIMIT
        )
        assert bounded_graph_row["metadata"]["support_chunk_edge_count"] == len(
            oversized_support_edge_ids
        )
        assert bounded_graph_row["metadata"]["support_chunk_edge_ids_hash"] == (
            stable_hash(oversized_support_edge_ids)
        )
        GraphResponse.model_validate(bounded_graph)

        bad_model_call_count = copy.deepcopy(bounded_graph)
        bad_model_call_count_row = next(
            edge
            for edge in bad_model_call_count["edges"]
            if edge.get("id") == f"rq-prefix-diagnostic:{bounded_row.id}"
        )
        bad_model_call_count_row["metadata"]["model_call_count"] = 1
        with pytest.raises(ValueError, match="model_call_count|zero model calls"):
            GraphResponse.model_validate(bad_model_call_count)

        bad_support_projection = copy.deepcopy(bounded_graph)
        bad_support_projection_row = next(
            edge
            for edge in bad_support_projection["edges"]
            if edge.get("id") == f"rq-prefix-diagnostic:{bounded_row.id}"
        )
        bad_support_projection_row["metadata"]["support_chunk_edge_ids"] = []
        with pytest.raises(ValueError, match="replay bounded support edge ids"):
            GraphResponse.model_validate(bad_support_projection)
    finally:
        bounded_row.support_chunk_edge_ids_json = original_support_edge_ids
        db_session.flush()


@pytest.mark.asyncio
async def test_rq_prefix_pair_deep_integrity_rejects_tamper_without_online_scan(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import (
        ChunkRelationEdge,
        ChunkRelationGraphState,
        RQPrefix,
        RQPrefixMembership,
        RQPrefixPairDiagnostic,
    )
    from app.services import context_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    relation_state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert relation_state is not None
    prefixes = list(
        db_session.scalars(
            select(RQPrefix)
            .where(RQPrefix.graph_state_id == relation_state.id)
            .order_by(RQPrefix.rq_level, RQPrefix.rq_prefix_key)
        ).all()
    )
    prefixes_by_key = {
        (
            int(prefix.rq_level or 0),
            tuple(int(value) for value in (prefix.rq_path_prefix or [])),
        ): prefix
        for prefix in prefixes
    }
    memberships = list(
        db_session.scalars(
            select(RQPrefixMembership)
            .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
            .where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    relation_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id
            )
        ).all()
    )
    edges = {
        (edge.source_chunk_id, edge.target_chunk_id, edge.edge_type): edge
        for edge in relation_edges
    }
    rows = list(
        db_session.scalars(
            select(RQPrefixPairDiagnostic)
            .where(RQPrefixPairDiagnostic.graph_state_id == relation_state.id)
            .order_by(RQPrefixPairDiagnostic.diagnostic_hash)
        ).all()
    )
    assert rows
    assert context_graph.rq_prefix_pair_persisted_integrity(
        db_session,
        relation_state,
        rows,
    )["valid"] is True

    class ForbiddenChatProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "RQ pair diagnostics and gray-zone state must not invoke an LLM"
            )

    monkeypatch.setattr(context_graph, "ChatProvider", ForbiddenChatProvider)
    retried = context_graph.build_rq_prefix_pair_diagnostics(
        db_session,
        relation_state,
        prefixes_by_key,
        memberships,
        edges,
    )
    assert len(retried) == len(rows)
    assert all(
        int((row.diagnostics_json or {}).get("model_call_count") or 0) == 0
        for row in retried
    )

    tampered_row = rows[0]
    original_strength = float(tampered_row.diagnostic_strength)
    tampered_row.diagnostic_strength = (
        original_strength - 0.01
        if original_strength >= 0.01
        else original_strength + 0.01
    )
    db_session.flush()
    tampered_integrity = context_graph.rq_prefix_pair_persisted_integrity(
        db_session,
        relation_state,
        rows,
    )
    assert tampered_integrity["valid"] is False
    assert any(
        reason.startswith("persisted_fact_column_mismatch:")
        for reason in tampered_integrity["reasons"]
    )
    with pytest.raises(RuntimeError, match="persisted fact integrity"):
        context_graph.build_rq_prefix_pair_diagnostics(
            db_session,
            relation_state,
            prefixes_by_key,
            memberships,
            edges,
        )

    tampered_row.diagnostic_strength = original_strength
    db_session.flush()
    assert context_graph.rq_prefix_pair_persisted_integrity(
        db_session,
        relation_state,
        rows,
    )["valid"] is True

    original_diagnostics = dict(tampered_row.diagnostics_json or {})
    for invalid_model_call_count in (None, False, "0", 1):
        tampered_row.diagnostics_json = {
            **original_diagnostics,
            "model_call_count": invalid_model_call_count,
        }
        db_session.flush()
        authority_integrity = context_graph.rq_prefix_pair_persisted_integrity(
            db_session,
            relation_state,
            rows,
        )
        assert authority_integrity["valid"] is False
        assert any(
            reason.startswith("model_call_count_nonzero:")
            for reason in authority_integrity["reasons"]
        )
    tampered_row.diagnostics_json = original_diagnostics
    db_session.flush()
    assert context_graph.rq_prefix_pair_persisted_integrity(
        db_session,
        relation_state,
        rows,
    )["valid"] is True

    def forbidden_deep_scan(*args, **kwargs):
        raise AssertionError(
            "online graph admission must use COUNT plus the durable integrity proof"
        )

    monkeypatch.setattr(
        context_graph,
        "rq_prefix_pair_persisted_integrity",
        forbidden_deep_scan,
    )
    admitted_state = context_graph.active_graph_admission_gate(
        db_session,
        knowledge_base.id,
        chunks=populated_context_graph["chunks"],
    )
    assert admitted_state is not None
    assert str(admitted_state.id) == str(populated_context_graph["state"].id)


def test_rq_prefix_pair_static_domain_and_uuid_free_canonical_facts(monkeypatch):
    from pydantic import ValidationError

    from app.core.config import Settings
    from app.schemas import ModelSettingsUpdate
    from app.services import context_graph

    with pytest.raises(ValidationError):
        Settings(rq_kmeans_max_k=7)
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(rq_kmeans_max_k=7)

    monkeypatch.setattr(
        context_graph,
        "validate_active_graph_protocol_settings",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            rq_kmeans_levels=3,
            rq_kmeans_max_k=7,
            rq_membership_temperature=0.35,
            rq_membership_protocol=context_graph.RQ_MEMBERSHIP_PROTOCOL_VERSION,
            rq_residual_tau=0.65,
        ),
    )
    with pytest.raises(RuntimeError, match="statically bounded exact prefix-pair"):
        context_graph.rq_runtime_config()

    protocol_hash_at_six = context_graph.rq_prefix_pair_diagnostic_protocol_hash()
    monkeypatch.setattr(context_graph, "RQ_MAX_K", 7)
    protocol_hash_at_seven = context_graph.rq_prefix_pair_diagnostic_protocol_hash()
    assert protocol_hash_at_six != protocol_hash_at_seven

    document_fields = {
        "source_path": "library/source.md",
        "source_type": "markdown",
        "checksum": "a" * 64,
        "title": "Stable source",
    }
    chunk_fields = {
        "chunk_version": 4,
        "chunk_index": 9,
        "token_start": 10,
        "token_end": 42,
        "char_start": 100,
        "char_end": 420,
        "section_path": "A > B",
        "page_start": 2,
        "page_end": 3,
        "text_hash": "b" * 64,
    }
    first_document = SimpleNamespace(id="document-uuid-a", **document_fields)
    second_document = SimpleNamespace(id="document-uuid-b", **document_fields)
    first_chunk = SimpleNamespace(
        id="chunk-uuid-a",
        document_id=first_document.id,
        **chunk_fields,
    )
    second_chunk = SimpleNamespace(
        id="chunk-uuid-b",
        document_id=second_document.id,
        **chunk_fields,
    )
    first_fact = context_graph._rq_pair_chunk_business_fact(
        first_chunk,
        first_document,
    )
    second_fact = context_graph._rq_pair_chunk_business_fact(
        second_chunk,
        second_document,
    )
    assert first_fact == second_fact
    serialized_fact = json.dumps(first_fact, sort_keys=True)
    assert "chunk-uuid" not in serialized_fact
    assert "document-uuid" not in serialized_fact

    business_keys = {"chunk-left": "business-a", "chunk-right": "business-b"}
    first_edge = SimpleNamespace(
        id="edge-uuid-a",
        source_chunk_id="chunk-left",
        target_chunk_id="chunk-right",
        edge_type="dense_semantic",
        weight=0.7,
        distance=0.3,
        features_json={"calibrated_strength": 0.7},
    )
    second_edge = SimpleNamespace(**{**vars(first_edge), "id": "edge-uuid-b"})
    assert context_graph._rq_pair_bottom_edge_fact(
        first_edge,
        business_keys,
    ) == context_graph._rq_pair_bottom_edge_fact(second_edge, business_keys)
