from __future__ import annotations

from types import SimpleNamespace

import pytest


def _candidate(source: str, target: str, edge_type: str, raw_strength: float):
    from app.services.context_graph import RelationEdgeCandidate, distance_from_strength

    return RelationEdgeCandidate(
        source_chunk_id=source,
        target_chunk_id=target,
        edge_type=edge_type,
        raw_strength=raw_strength,
        calibrated_strength=raw_strength,
        distance=distance_from_strength(raw_strength),
        features_json={"fixture": True},
        is_bridge=edge_type != "dense_semantic",
    )


def _candidate_map(rows):
    return {
        (*sorted((source, target)), edge_type): _candidate(source, target, edge_type, raw_strength)
        for source, target, edge_type, raw_strength in rows
    }


def test_type_local_calibration_is_monotonic_and_never_uses_a_global_distribution(no_fallback_env):
    from app.services.context_graph import calibrate_relation_candidates, dense_graph_operating_point, distance_from_strength

    candidates = _candidate_map(
        [
            ("s-1", "s-2", "dense_semantic", 0.2),
            ("s-1", "s-3", "dense_semantic", 0.4),
            ("s-1", "s-4", "dense_semantic", 0.8),
            ("d-1", "d-2", "dense_cross_document_bridge", 0.4),
            ("d-1", "d-3", "dense_cross_document_bridge", 0.5),
            ("d-1", "d-4", "dense_cross_document_bridge", 0.6),
        ]
    )

    calibrated, diagnostics = calibrate_relation_candidates(candidates, dense_graph_operating_point())
    by_type = diagnostics["stats_by_edge_type"]

    semantic = sorted(
        (item.raw_strength, item.calibrated_strength)
        for item in calibrated.values()
        if item.edge_type == "dense_semantic"
    )
    cross_document = sorted(
        (item.raw_strength, item.calibrated_strength)
        for item in calibrated.values()
        if item.edge_type == "dense_cross_document_bridge"
    )
    assert semantic == [(0.2, 0.05), (0.4, 0.366667), (0.8, 1.0)]
    assert cross_document == [(0.4, 0.05), (0.5, 0.525), (0.6, 1.0)]
    # Identical raw strength has a different meaning in a different type-local
    # distribution; a global or identity transform would make these equal.
    assert dict(semantic)[0.4] != dict(cross_document)[0.4]

    for edge_type in ("dense_semantic", "dense_cross_document_bridge"):
        typed = sorted(
            (item.raw_strength, item.calibrated_strength, item.distance)
            for item in calibrated.values()
            if item.edge_type == edge_type
        )
        assert all(typed[index][1] <= typed[index + 1][1] for index in range(len(typed) - 1))
        assert all(item[2] == distance_from_strength(item[1]) for item in typed)
        assert by_type[edge_type]["calibration_applied"] is True
        assert by_type[edge_type]["monotonic_violation_count"] == 0
        assert by_type[edge_type]["stats_hash"]
        assert {
            edge.features_json["edge_type_calibration_stats_hash"]
            for edge in calibrated.values()
            if edge.edge_type == edge_type
        } == {by_type[edge_type]["stats_hash"]}

    assert by_type["dense_cross_language_bridge"]["fallback"] == "no_edges"
    assert diagnostics["cross_type_raw_comparison_allowed"] is False


def test_sparse_and_degenerate_type_distributions_fall_back_to_identity_with_reason(no_fallback_env):
    from app.services.context_graph import calibrate_relation_candidates, dense_graph_operating_point

    candidates = _candidate_map(
        [
            ("s-1", "s-2", "dense_semantic", 0.7),
            ("d-1", "d-2", "dense_cross_document_bridge", 0.5),
            ("d-1", "d-3", "dense_cross_document_bridge", 0.51),
        ]
    )
    calibrated, diagnostics = calibrate_relation_candidates(candidates, dense_graph_operating_point())

    semantic = next(item for item in calibrated.values() if item.edge_type == "dense_semantic")
    cross_document = [item for item in calibrated.values() if item.edge_type == "dense_cross_document_bridge"]
    assert semantic.calibrated_strength == semantic.raw_strength == 0.7
    assert diagnostics["stats_by_edge_type"]["dense_semantic"]["fallback"] == "insufficient_type_sample"
    assert diagnostics["stats_by_edge_type"]["dense_semantic"]["calibration_applied"] is False
    assert [item.calibrated_strength for item in cross_document] == [item.raw_strength for item in cross_document]
    assert diagnostics["stats_by_edge_type"]["dense_cross_document_bridge"]["fallback"] == "degenerate_quantile_span"


@pytest.mark.parametrize(
    "override",
    [
        {"edge_type_calibration_protocol": "unknown_calibration_v9"},
        {"edge_type_calibration_protocol_hash": "stale-hash"},
        {"edge_distance_protocol": "edge_distance_log_raw_strength_v1"},
        {"edge_distance_protocol_hash": "stale-hash"},
        {"edge_type_calibration_params": {"lower_quantile": 0.4}},
        {"edge_type_calibration_params": {"unknown": 1.0}},
        {
            "calibration_params": {
                "lower_quantile": 0.4,
                "upper_quantile": 0.95,
                "min_span": 0.05,
                "strength_floor": 0.05,
            }
        },
        {
            "calibration_params": {
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "min_span": 0.05,
                "strength_floor": 0.05,
                "unknown": 1.0,
            }
        },
    ],
)
def test_calibration_preflight_fails_closed_for_protocol_or_parameter_drift(no_fallback_env, override):
    from app.services.context_graph import dense_graph_operating_point, edge_type_calibration_config

    operating_point = {**dense_graph_operating_point(), **override}
    with pytest.raises(ValueError):
        edge_type_calibration_config(operating_point)


def test_calibration_config_requires_all_canonical_fields(no_fallback_env):
    from app.services.context_graph import dense_graph_operating_point, edge_type_calibration_config

    for field in (
        "graph_operating_point_protocol",
        "edge_type_calibration_protocol",
        "edge_type_calibration_protocol_hash",
        "edge_distance_protocol",
        "edge_distance_protocol_hash",
        "calibration_params",
    ):
        operating_point = dense_graph_operating_point()
        operating_point.pop(field)
        with pytest.raises(ValueError, match=field):
            edge_type_calibration_config(operating_point)


def test_calibration_parameters_change_simulated_distances_and_audit_hashes(no_fallback_env):
    from app.services.context_graph import calibrate_relation_candidates, dense_graph_operating_point

    candidates = _candidate_map(
        [
            ("s-1", "s-2", "dense_semantic", 0.2),
            ("s-1", "s-3", "dense_semantic", 0.4),
            ("s-1", "s-4", "dense_semantic", 0.6),
            ("s-1", "s-5", "dense_semantic", 0.8),
        ]
    )
    baseline_theta = dense_graph_operating_point()
    changed_theta = dense_graph_operating_point()
    changed_theta["calibration_params"] = {
        "lower_quantile": 0.2,
        "upper_quantile": 0.8,
        "min_span": 0.1,
        "strength_floor": 0.2,
    }

    baseline, baseline_diagnostics = calibrate_relation_candidates(
        candidates,
        baseline_theta,
    )
    changed, changed_diagnostics = calibrate_relation_candidates(
        candidates,
        changed_theta,
    )
    assert [edge.distance for edge in baseline.values()] != [
        edge.distance for edge in changed.values()
    ]
    assert (
        baseline_diagnostics["calibration_params_hash"]
        != changed_diagnostics["calibration_params_hash"]
    )
    assert (
        baseline_diagnostics["edge_type_calibration_config_hash"]
        != changed_diagnostics["edge_type_calibration_config_hash"]
    )


def test_active_strength_and_distance_fallback_never_prefer_raw_strength(no_fallback_env):
    from app.services.context_graph import _edge_calibrated_strength, _edge_distance, distance_from_strength

    edge = SimpleNamespace(
        raw_strength=0.95,
        weight=0.2,
        distance=None,
        features_json={"calibrated_strength": 0.2},
        raw_strength_summary_json={"max_raw_strength": 0.95, "calibrated_strength": 0.2},
    )
    assert _edge_calibrated_strength(edge) == 0.2
    assert _edge_distance(edge) == distance_from_strength(0.2)


def test_calibration_protocol_hash_propagates_to_operating_point_and_retrieval_cache(no_fallback_env):
    from app.services.context_graph import (
        context_graph_cache_key_components,
        dense_graph_operating_point,
        edge_type_calibration_config,
        edge_type_calibration_protocol_hash,
    )

    operating_point = dense_graph_operating_point()
    cache_components = context_graph_cache_key_components(
        knowledge_base_id="kb-calibration-protocol",
        query="calibration audit",
        filters={},
        context_state=None,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash="a" * 64,
        profile_hash_value="profile-calibration-test",
    )
    expected_hash = edge_type_calibration_protocol_hash()
    expected_config = edge_type_calibration_config(operating_point)
    assert operating_point["edge_type_calibration_protocol_hash"] == expected_hash
    assert cache_components["edge_type_calibration_protocol_hash"] == expected_hash
    assert cache_components["calibration_params_hash"] == expected_config["params_hash"]
    assert (
        cache_components["edge_type_calibration_config_hash"]
        == expected_config["config_hash"]
    )


@pytest.mark.asyncio
async def test_relation_build_persists_replayable_type_local_calibration(db_session, populated_context_graph):
    from sqlalchemy import select

    from app.models import ChunkRelationEdge, ChunkRelationGraphState
    from app.services.context_graph import (
        EDGE_DISTANCE_PROTOCOL_VERSION,
        EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
        distance_from_strength,
        edge_distance_protocol_hash,
        edge_type_calibration_protocol_hash,
        relation_edge_rank_trace_payload,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert state is not None
    edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(ChunkRelationEdge.graph_state_id == state.id)
        ).all()
    )
    assert edges

    hashes_by_type: dict[str, set[str]] = {}
    for edge in edges:
        features = dict(edge.features_json or {})
        normalization = dict((edge.normalization_stats_json or {}).get("edge_type_calibration") or {})
        calibrated_strength = float(features["calibrated_strength"])
        assert float(edge.weight) == pytest.approx(calibrated_strength, abs=1e-9)
        assert float(edge.distance) == pytest.approx(distance_from_strength(calibrated_strength), abs=1e-9)
        assert edge.raw_strength_summary_json["max_raw_strength"] == pytest.approx(float(edge.raw_strength), abs=1e-9)
        assert normalization["edge_type"] == edge.edge_type
        assert normalization["protocol_version"] == EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION
        assert normalization["protocol_hash"] == edge_type_calibration_protocol_hash()
        assert normalization["stats_hash"] == features["edge_type_calibration_stats_hash"]
        hashes_by_type.setdefault(edge.edge_type, set()).add(normalization["stats_hash"])
        trace = relation_edge_rank_trace_payload(edge)
        assert trace["calibrated_strength"] == calibrated_strength
        assert trace["edge_type_calibration_stats_hash"] == normalization["stats_hash"]

    assert all(len(stats_hashes) == 1 for stats_hashes in hashes_by_type.values())
    assert state.edge_type_calibration_protocol_hash == edge_type_calibration_protocol_hash()
    assert state.edge_distance_protocol_hash == edge_distance_protocol_hash()
    assert state.graph_operating_point_json["edge_type_calibration_protocol"] == EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION
    assert state.graph_operating_point_json["edge_distance_protocol"] == EDGE_DISTANCE_PROTOCOL_VERSION
    assert state.diagnostics_json["edge_type_calibration"]["stats_by_edge_type"]
    assert state.stats_json["edge_type_calibration"]["all_stats_hashes_consistent"] is True
    assert state.stats_json["edge_type_distributions"]
