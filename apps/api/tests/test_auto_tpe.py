from __future__ import annotations

from types import SimpleNamespace

import pytest


def _simulated_calibration_diagnostics(theta: dict[str, object]) -> dict[str, object]:
    from app.services.auto_tpe import theta_calibration_audit

    audit = theta_calibration_audit(theta)
    return {
        "relation_quota_signals": {
            "language_identity": {
                "protocol_version": "active_chunk_language_identity_scope_v1",
                "scope_hash": "5" * 64,
            }
        },
        "edge_type_calibration": {
            "protocol_version": audit["edge_type_calibration_protocol"],
            "protocol_hash": audit["edge_type_calibration_protocol_hash"],
            "edge_distance_protocol_version": audit["edge_distance_protocol"],
            "edge_distance_protocol_hash": audit["edge_distance_protocol_hash"],
            "params": dict(audit["calibration_params"]),
            "calibration_params_hash": audit["calibration_params_hash"],
            "edge_type_calibration_config_hash": audit[
                "edge_type_calibration_config_hash"
            ],
            "stats_by_edge_type": {},
        },
    }


def _passing_hard_gate() -> dict[str, dict[str, object]]:
    return {
        "edge_density": {
            "value": 0.0,
            "threshold": 1.0,
            "metric_protocol": (
                "tpe_normalized_undirected_simple_graph_density_v1"
            ),
            "direction": "max",
            "passed": True,
        },
        "sparse_edge_budget_ratio": {
            "value": 0.0,
            "threshold": 1.0,
            "direction": "max",
            "passed": True,
        },
        "isolated_ratio": {
            "value": 0.0,
            "threshold": 1.0,
            "direction": "max",
            "passed": True,
        },
        "hubness_ratio": {
            "value": 1.0,
            "threshold": 1000.0,
            "direction": "max",
            "passed": True,
        },
        "structure_recovery_rate": {
            "value": 1.0,
            "threshold": 0.0,
            "direction": "min",
            "passed": True,
        },
        "candidate_latency_p95_ms": {
            "value": 1.0,
            "threshold": 600000.0,
            "direction": "max",
            "passed": True,
        },
    }


def _quality_candidate(
    source: str,
    target: str,
    *,
    edge_type: str = "dense_semantic",
    candidate_channels: list[str] | None = None,
):
    from app.services import context_graph

    channels = list(candidate_channels or ["base_dense_candidates"])
    node_component_values = {
        name: context_graph._quota_signal_component(
            0.5,
            available=True,
            reason=f"test:{name}",
            diagnostics={},
        )
        for name in context_graph.CHUNK_NODE_QUALITY_COMPONENT_WEIGHTS
    }
    source_node_quality = context_graph._availability_weighted_signal_card(
        protocol_version=context_graph.CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
        protocol_hash=context_graph.chunk_node_quality_protocol_hash(),
        component_weights=context_graph.CHUNK_NODE_QUALITY_COMPONENT_WEIGHTS,
        components=node_component_values,
    )
    target_node_quality = dict(source_node_quality)

    def signal_card(
        *,
        protocol_version: str,
        protocol_hash: str,
        weights: dict[str, float],
        node_quality_card: dict[str, object],
    ) -> dict[str, object]:
        components = {
            name: context_graph._quota_signal_component(
                0.5,
                available=True,
                reason=f"test:{name}",
                diagnostics=(
                    {"card_hash": node_quality_card["card_hash"]}
                    if name == "node_quality"
                    else {}
                ),
            )
            for name in weights
        }
        return context_graph._availability_weighted_signal_card(
            protocol_version=protocol_version,
            protocol_hash=protocol_hash,
            component_weights=weights,
            components=components,
        )

    source_signal = signal_card(
        protocol_version=context_graph.OUT_EVIDENCE_MASS_PROTOCOL_VERSION,
        protocol_hash=context_graph.out_evidence_mass_protocol_hash(),
        weights=context_graph.OUT_EVIDENCE_MASS_COMPONENT_WEIGHTS,
        node_quality_card=source_node_quality,
    )
    target_signal = signal_card(
        protocol_version=context_graph.IN_ACCEPTANCE_CAPACITY_PROTOCOL_VERSION,
        protocol_hash=context_graph.in_acceptance_capacity_protocol_hash(),
        weights=context_graph.IN_ACCEPTANCE_CAPACITY_COMPONENT_WEIGHTS,
        node_quality_card=target_node_quality,
    )
    semantic = 0.8
    reciprocity = 1.0
    rank_score = 1.0
    node_quality_pair = 0.5
    raw_strength = context_graph.relation_raw_strength(
        semantic=semantic,
        reciprocity=reciprocity,
        rank_score=rank_score,
        node_quality_pair=node_quality_pair,
    )
    return context_graph.RelationEdgeCandidate(
        source_chunk_id=source,
        target_chunk_id=target,
        edge_type=edge_type,
        raw_strength=raw_strength,
        calibrated_strength=raw_strength,
        distance=0.25,
        features_json={
            "cosine": 0.9,
            "language_identity_scope_hash": "8" * 64,
            "normalization_stats": {
                "type_threshold": 0.7,
                "strong_threshold": 0.95,
                "mutual": True,
                "reverse_accepted": False,
            },
            "raw_strength_components": {
                "protocol_version": context_graph.RELATION_RAW_STRENGTH_PROTOCOL_VERSION,
                "protocol_hash": context_graph.relation_raw_strength_protocol_hash(),
                "coefficients": {
                    "semantic": 0.75,
                    "reciprocity": 0.15,
                    "rank_score": 0.07,
                    "node_quality_pair": 0.03,
                },
                "semantic": semantic,
                "reciprocity": reciprocity,
                "rank_score": rank_score,
                "node_quality_pair": node_quality_pair,
                "node_quality_pair_protocol_version": context_graph.CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
                "node_quality_pair_protocol_hash": context_graph.chunk_node_quality_protocol_hash(),
                "computed_raw_strength": raw_strength,
            },
            "rank_score": rank_score,
            "rank_score_protocol_version": context_graph.RANK_SCORE_PROTOCOL_VERSION,
            "rank_score_protocol_hash": context_graph.relation_rank_score_protocol_hash(),
            "rank_components": [
                {
                    "candidate_channel": channel,
                    "rank": 1,
                    "ordinal": 1,
                    "candidate_count": 1,
                    "selected_limit": 1,
                    "selected_count": 1,
                    "rank_score": rank_score,
                }
                for channel in channels
            ],
            "reciprocity": reciprocity,
            "node_quality_pair": node_quality_pair,
            "node_quality_pair_protocol_version": context_graph.CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
            "node_quality_pair_protocol_hash": context_graph.chunk_node_quality_protocol_hash(),
            "node_quality_pair_role": "mean_intrinsic_node_quality_graph_build_only_not_query_relevance",
            "node_weight_used_as_query_relevance": False,
            "source_out_evidence_mass": source_signal["value"],
            "target_in_acceptance_capacity": target_signal["value"],
            "source_out_signal_card": source_signal,
            "target_in_acceptance_signal_card": target_signal,
            "source_node_quality_card": source_node_quality,
            "target_node_quality_card": target_node_quality,
            "source_out_quota_cards": {
                channel: context_graph.relation_quota_card(
                    lower=1,
                    upper=1,
                    signal=float(source_signal["value"]),
                    signal_role="out_evidence_mass",
                )
                for channel in channels
            },
            "target_inbound_quota_card": context_graph.relation_quota_card(
                lower=1,
                upper=1,
                signal=float(target_signal["value"]),
                signal_role="in_acceptance_capacity",
            ),
            "quota_signal_scope_hash": "9" * 64,
            "chunk_node_quality_protocol_version": context_graph.CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
            "chunk_node_quality_protocol_hash": context_graph.chunk_node_quality_protocol_hash(),
            "out_evidence_mass_protocol_version": context_graph.OUT_EVIDENCE_MASS_PROTOCOL_VERSION,
            "out_evidence_mass_protocol_hash": context_graph.out_evidence_mass_protocol_hash(),
            "in_acceptance_capacity_protocol_version": context_graph.IN_ACCEPTANCE_CAPACITY_PROTOCOL_VERSION,
            "in_acceptance_capacity_protocol_hash": context_graph.in_acceptance_capacity_protocol_hash(),
            "relation_quota_protocol_version": context_graph.RELATION_QUOTA_PROTOCOL_VERSION,
            "relation_quota_protocol_hash": context_graph.relation_quota_protocol_hash(),
            "candidate_channels": channels,
            "all_candidate_channels": channels,
        },
        is_bridge=edge_type != "dense_semantic",
    )


def test_edge_precision_support_audit_replays_complete_frozen_cards():
    from copy import deepcopy

    from app.services import auto_tpe, context_graph

    valid = _quality_candidate("chunk-a", "chunk-b")
    valid_audit = auto_tpe._candidate_support_feature_audit(valid)
    assert valid_audit["passed"] is True
    assert all(valid_audit["checks"].values())
    assert valid_audit["replay"]["raw_strength"] == valid.raw_strength

    tampered = deepcopy(valid)
    tampered.features_json["raw_strength_components"]["coefficients"][
        "semantic"
    ] = 0.74
    audit = auto_tpe._candidate_support_feature_audit(tampered)
    assert audit["checks"]["raw_strength_protocol"] is False
    assert audit["checks"]["raw_strength_replay"] is False
    assert audit["passed"] is False

    tampered = deepcopy(valid)
    tampered.features_json["raw_strength_components"]["semantic"] = 0.7
    audit = auto_tpe._candidate_support_feature_audit(tampered)
    assert audit["checks"]["raw_strength_inputs"] is False
    assert audit["checks"]["raw_strength_replay"] is False

    tampered = deepcopy(valid)
    tampered.features_json["rank_components"][0]["rank_score"] = 0.5
    audit = auto_tpe._candidate_support_feature_audit(tampered)
    assert audit["checks"]["rank_support"] is False
    assert audit["checks"]["raw_strength_inputs"] is False

    tampered = deepcopy(valid)
    tampered.features_json["source_node_quality_card"]["value"] = 0.4
    audit = auto_tpe._candidate_support_feature_audit(tampered)
    assert audit["checks"]["node_quality_support"] is False
    assert audit["checks"]["raw_strength_inputs"] is False

    tampered = deepcopy(valid)
    tampered.features_json["target_inbound_quota_card"]["signal"] = 0.4
    audit = auto_tpe._candidate_support_feature_audit(tampered)
    assert audit["checks"]["quota_support"] is False
    assert audit["passed"] is False


def test_edge_precision_replay_accepts_only_persisted_quantization_error():
    from dataclasses import replace

    from app.services import auto_tpe, context_graph

    candidate = _quality_candidate("chunk-a", "chunk-b")
    type_threshold = 0.3
    strong_threshold = 0.72
    provider_cosine = 0.50000049
    persisted_cosine = round(provider_cosine, 6)
    semantic = round(
        (provider_cosine - type_threshold)
        / (strong_threshold - type_threshold),
        6,
    )
    raw = candidate.features_json["raw_strength_components"]
    raw["semantic"] = semantic
    raw["computed_raw_strength"] = context_graph.relation_raw_strength(
        semantic=semantic,
        reciprocity=float(raw["reciprocity"]),
        rank_score=float(raw["rank_score"]),
        node_quality_pair=float(raw["node_quality_pair"]),
    )
    candidate = replace(
        candidate,
        raw_strength=float(raw["computed_raw_strength"]),
    )
    candidate.features_json["cosine"] = persisted_cosine
    candidate.features_json["normalization_stats"] = {
        "type_threshold": type_threshold,
        "strong_threshold": strong_threshold,
        "mutual": True,
        "reverse_accepted": False,
    }

    audit = auto_tpe._candidate_support_feature_audit(candidate)

    assert audit["passed"] is True, audit
    assert audit["checks"]["raw_strength_inputs"] is True
    assert audit["checks"]["raw_strength_replay"] is True


def test_tpe_latency_profile_uses_phase_nearest_rank_p95_and_detects_long_tail():
    from app.services import auto_tpe

    profile = auto_tpe._tpe_latency_profile(
        candidate_adjacency_samples_ms=[5.0],
        probe_expansion_samples_ms=([1.0] * 19) + [80.0, 100.0],
        metric_computation_samples_ms=[2.0, 3.0, 4.0],
    )

    assert profile["protocol_version"] == "tpe_local_latency_segment_nearest_rank_p95_v1"
    assert profile["phases"]["probe_expansion"]["sample_count"] == 21
    assert profile["phases"]["probe_expansion"]["p95_ms"] == 80.0
    assert profile["phase_p95_ms"] == {
        "candidate_adjacency": 5.0,
        "probe_expansion": 80.0,
        "metric_computation": 4.0,
    }
    assert profile["candidate_latency_p95_ms"] == 80.0
    assert profile["sample_count"] == 25


@pytest.mark.parametrize(
    ("latency_ms", "budget_ms", "expected_penalty", "expected_excess_ms"),
    [
        (50.0, 100.0, 0.0, 0.0),
        (100.0, 100.0, 0.0, 0.0),
        (150.0, 100.0, 0.5, 50.0),
        (250.0, 100.0, 1.0, 150.0),
    ],
)
def test_tpe_latency_penalty_is_budget_excess_only(
    latency_ms,
    budget_ms,
    expected_penalty,
    expected_excess_ms,
):
    from app.services import auto_tpe

    card = auto_tpe._latency_budget_excess_penalty(
        latency_ms,
        budget_ms,
        probe_hash="1" * 64,
    )

    assert card["value"] == expected_penalty
    assert card["numerator"] == expected_excess_ms
    assert card["excess_ms"] == expected_excess_ms
    assert card["denominator"] == budget_ms
    assert card["protocol"] == auto_tpe.TPE_LATENCY_PENALTY_PROTOCOL_VERSION


@pytest.mark.parametrize("samples", [[], [-1.0], [float("nan")], [float("inf")]])
def test_tpe_latency_distribution_fails_closed_on_invalid_samples(samples):
    from app.services import auto_tpe

    with pytest.raises(ValueError, match="latency phase"):
        auto_tpe._latency_distribution(samples, phase="probe_expansion")


def test_tpe_latency_distribution_enforces_bounded_sample_count():
    from app.services import auto_tpe

    with pytest.raises(ValueError, match="sample limit"):
        auto_tpe._latency_distribution(
            [1.0] * (auto_tpe.TPE_LATENCY_SAMPLE_LIMIT + 1),
            phase="metric_computation",
        )


def test_tpe_hubness_penalty_replays_degree_top_share_and_edge_type_imbalance():
    from app.services import auto_tpe

    chunks = [SimpleNamespace(id=f"chunk-{index}") for index in range(12)]
    edge_types = (
        "dense_semantic",
        "dense_cross_document_bridge",
        "dense_cross_language_bridge",
    )
    balanced = {
        (f"chunk-{index}", f"chunk-{(index + 1) % 12}", edge_types[index % 3]): (
            SimpleNamespace(
                source_chunk_id=f"chunk-{index}",
                target_chunk_id=f"chunk-{(index + 1) % 12}",
                edge_type=edge_types[index % 3],
            )
        )
        for index in range(12)
    }
    star = {
        ("chunk-0", f"chunk-{index}", "dense_semantic"): SimpleNamespace(
            source_chunk_id="chunk-0",
            target_chunk_id=f"chunk-{index}",
            edge_type="dense_semantic",
        )
        for index in range(1, 12)
    }

    balanced_stats = auto_tpe._degree_stats(chunks, balanced)
    star_stats = auto_tpe._degree_stats(chunks, star)
    balanced_card = auto_tpe._hubness_penalty_component(
        balanced_stats,
        max_hubness_ratio=12.0,
        probe_hash="a" * 64,
        eligible_edge_types=set(edge_types),
    )
    star_card = auto_tpe._hubness_penalty_component(
        star_stats,
        max_hubness_ratio=12.0,
        probe_hash="a" * 64,
        eligible_edge_types=set(edge_types),
    )
    single_channel_card = auto_tpe._hubness_penalty_component(
        star_stats,
        max_hubness_ratio=12.0,
        probe_hash="a" * 64,
        eligible_edge_types={"dense_semantic"},
    )

    assert balanced_stats["top_hub_concentration"] == pytest.approx(0.0)
    assert star_stats["top_hub_concentration"] > 0.0
    assert balanced_card["dimensions"]["edge_type_imbalance"][
        "value"
    ] == pytest.approx(0.0)
    assert star_card["dimensions"]["edge_type_imbalance"][
        "value"
    ] == pytest.approx(1.0)
    assert single_channel_card["dimensions"]["edge_type_imbalance"][
        "value"
    ] == pytest.approx(0.0)
    assert star_card["value"] > balanced_card["value"]
    assert star_card["sample_count"] == 12
    assert set(star_card["dimensions"]) == {
        "degree_p95_to_median_pressure",
        "top_hub_concentration",
        "edge_type_imbalance",
    }
    assert star_card["protocol"] == auto_tpe.TPE_HUBNESS_PENALTY_PROTOCOL_VERSION


def test_tpe_density_penalty_uses_soft_target_interval_before_hard_ceiling():
    from app.services import auto_tpe

    below = auto_tpe._density_penalty_component(
        0.5,
        edge_count=50,
        max_edge_density=0.8,
        probe_hash="b" * 64,
    )
    inside_penalty_band = auto_tpe._density_penalty_component(
        0.7,
        edge_count=70,
        max_edge_density=0.8,
        probe_hash="b" * 64,
    )
    at_hard_ceiling = auto_tpe._density_penalty_component(
        0.8,
        edge_count=80,
        max_edge_density=0.8,
        probe_hash="b" * 64,
    )

    assert below["target_interval"] == {"lower": 0.0, "upper": 0.6}
    assert below["value"] == 0.0
    assert inside_penalty_band["value"] == 0.5
    assert at_hard_ceiling["value"] == 1.0
    assert at_hard_ceiling["sample_count"] == 80
    assert at_hard_ceiling["protocol"] == auto_tpe.TPE_DENSITY_PENALTY_PROTOCOL_VERSION


def test_small_scope_density_and_search_bounds_reject_near_complete_graph():
    from itertools import combinations

    from app.services import auto_tpe

    chunks = [SimpleNamespace(id=f"chunk-{index}") for index in range(17)]
    pairs = list(combinations([chunk.id for chunk in chunks], 2))
    candidates = {
        (source, target, "dense_semantic"): SimpleNamespace(
            source_chunk_id=source,
            target_chunk_id=target,
        )
        for source, target in pairs[:132]
    }

    card = auto_tpe._normalized_graph_density_card(chunks, candidates)
    scope = auto_tpe.tpe_scope_search_card(len(chunks))
    anchor = auto_tpe._scope_anchor_theta(
        {
            **auto_tpe.dense_graph_operating_point(),
            "dense_knn_k_min": 16,
            "dense_knn_k_max": 32,
        },
        chunk_count=len(chunks),
    )

    assert card["normalized_edge_density"] == pytest.approx(132 / 136)
    assert card["mean_edges_per_node"] == pytest.approx(132 / 17)
    assert card["sparse_edge_budget"] == 70
    assert card["sparse_edge_budget_ratio"] > 1.0
    assert scope["integer_bounds"]["dense_knn_k_max"]["upper"] == 5
    assert anchor["dense_knn_k_min"] == 1
    assert anchor["dense_knn_k_max"] == 5
    assert scope["model_call_count"] == 0


def test_tpe_quality_input_hashes_ignore_database_uuid_reallocation(
    db_session,
    sample_knowledge_base,
):
    from app.services import auto_tpe

    def chunks_for(prefix: str):
        chunks = []
        for index in range(3):
            target_index = (index + 1) % 3
            chunks.append(
                SimpleNamespace(
                    id=f"{prefix}-chunk-{index}",
                    knowledge_base_id=sample_knowledge_base.id,
                    document_id=f"{prefix}-document-{index}",
                    document_version_id=f"{prefix}-version-{index}",
                    chunk_version=1,
                    chunk_index=index,
                    text_hash=str(index + 1) * 64,
                    text=f"stable text {index}",
                    char_start=index * 20,
                    char_end=index * 20 + 18,
                    token_start=index * 4,
                    token_end=index * 4 + 3,
                    section_path=f"section-{index}",
                    page_start=index + 1,
                    page_end=index + 1,
                    previous_chunk_id=None,
                    next_chunk_id=None,
                    metadata_json={
                        "tpe_expected_support_chunk_ids": [
                            f"{prefix}-chunk-{target_index}"
                        ]
                    },
                )
            )
        return chunks

    old_chunks = chunks_for("old")
    new_chunks = chunks_for("new")
    old_keys = {
        str(chunk.id): str(index + 1) * 64
        for index, chunk in enumerate(old_chunks)
    }
    new_keys = {
        str(chunk.id): str(index + 1) * 64
        for index, chunk in enumerate(new_chunks)
    }
    old_vectors = {
        str(chunk.id): [float(index), 1.0, float(index + 1)]
        for index, chunk in enumerate(old_chunks)
    }
    new_vectors = {
        str(chunk.id): [float(index), 1.0, float(index + 1)]
        for index, chunk in enumerate(new_chunks)
    }

    _old_probes, old_probe_hash = auto_tpe._probe_chunks(
        old_chunks,
        3,
        chunk_business_keys=old_keys,
    )
    _new_probes, new_probe_hash = auto_tpe._probe_chunks(
        new_chunks,
        3,
        chunk_business_keys=new_keys,
    )
    assert old_probe_hash == new_probe_hash

    old_structure = {
        str(chunk.id): {
            "previous_next": set(),
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        }
        for chunk in old_chunks
    }
    new_structure = {
        str(chunk.id): {
            "previous_next": set(),
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        }
        for chunk in new_chunks
    }
    _old_expected, old_expected_diagnostics = auto_tpe._expected_support_context(
        db_session,
        old_chunks,
        old_chunks,
        old_structure,
        chunk_business_keys=old_keys,
    )
    _new_expected, new_expected_diagnostics = auto_tpe._expected_support_context(
        db_session,
        new_chunks,
        new_chunks,
        new_structure,
        chunk_business_keys=new_keys,
    )
    assert (
        old_expected_diagnostics["expected_support_pairs_hash"]
        == new_expected_diagnostics["expected_support_pairs_hash"]
    )

    _old_rq, old_rq_diagnostics = auto_tpe._candidate_rq_prefix_inputs(
        old_chunks,
        old_vectors,
        chunk_business_keys=old_keys,
    )
    _new_rq, new_rq_diagnostics = auto_tpe._candidate_rq_prefix_inputs(
        new_chunks,
        new_vectors,
        chunk_business_keys=new_keys,
    )
    assert old_rq_diagnostics["input_hash"] == new_rq_diagnostics["input_hash"]
    assert old_rq_diagnostics["production_canonical_business_keys"] is True


def test_tpe_manual_expected_support_is_bounded_and_rejects_malformed_metadata(
    db_session,
    sample_knowledge_base,
):
    from app.services import auto_tpe

    chunks = [
        SimpleNamespace(
            id="probe-0",
            knowledge_base_id=sample_knowledge_base.id,
            document_id="document-0",
            document_version_id="version-0",
            chunk_version=1,
            chunk_index=0,
            text_hash="1" * 64,
            char_start=0,
            char_end=10,
            token_start=0,
            token_end=2,
            section_path="section",
            page_start=1,
            page_end=1,
            metadata_json={
                "tpe_expected_support_chunk_ids": [
                    f"target-{index:04d}" for index in range(600)
                ]
            },
        ),
        SimpleNamespace(
            id="probe-1",
            knowledge_base_id=sample_knowledge_base.id,
            document_id="document-1",
            document_version_id="version-1",
            chunk_version=1,
            chunk_index=1,
            text_hash="2" * 64,
            char_start=10,
            char_end=20,
            token_start=2,
            token_end=4,
            section_path="section",
            page_start=1,
            page_end=1,
            metadata_json={"tpe_expected_support_chunk_ids": "not-a-list"},
        ),
    ]
    chunks.extend(
        SimpleNamespace(
            id=f"target-{index:04d}",
            knowledge_base_id=sample_knowledge_base.id,
            document_id=f"target-document-{index:04d}",
            document_version_id=f"target-version-{index:04d}",
            chunk_version=1,
            chunk_index=index + 2,
            text_hash=f"{index:064x}"[-64:],
            char_start=(index + 2) * 10,
            char_end=(index + 3) * 10,
            token_start=(index + 2) * 2,
            token_end=(index + 3) * 2,
            section_path="target-section",
            page_start=index + 2,
            page_end=index + 2,
            metadata_json={},
        )
        for index in range(600)
    )
    empty_structure = {
        str(chunk.id): {
            "previous_next": set(),
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        }
        for chunk in chunks
    }
    _expected, diagnostics = auto_tpe._expected_support_context(
        db_session,
        chunks,
        chunks[:2],
        empty_structure,
        chunk_business_keys={
            "probe-0": "1" * 64,
            "probe-1": "2" * 64,
            **{
                f"target-{index:04d}": f"target-business-{index:04d}"
                for index in range(600)
            },
        },
    )

    assert diagnostics["manual_target_limit_per_probe"] == 512
    assert diagnostics["truncated_manual_target_count"] == 88
    assert diagnostics["invalid_manual_value_count"] == 1
    assert diagnostics["invalid_manual_target_id_count"] == 0
    assert diagnostics["manual_canonical_candidate_count"] == 600
    assert diagnostics["manual_selected_target_count"] == 512
    assert diagnostics["model_call_count"] == 0


@pytest.mark.parametrize(
    ("vectors", "expected_reason", "count_field"),
    [
        (
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]},
            "missing_candidate_vectors",
            "missing_vector_count",
        ),
        (
            {
                "chunk-0": [1.0, 0.0],
                "chunk-1": [0.0, 1.0],
                "chunk-2": [0.5, 0.5],
                "chunk-extra": [0.1, 0.9],
            },
            "extra_candidate_vectors",
            "extra_vector_count",
        ),
        (
            {
                "chunk-0": [1.0, 0.0],
                "chunk-1": [0.0, float("nan")],
                "chunk-2": [0.5, 0.5],
            },
            "non_finite_candidate_vector",
            "invalid_vector_count",
        ),
        (
            {
                "chunk-0": [1.0, 0.0],
                "chunk-1": [0.0, 1.0, 0.0],
                "chunk-2": [0.5, 0.5],
            },
            "candidate_vector_dimension_drift",
            "dimension_drift_vector_count",
        ),
        (
            {
                "chunk-0": [1.0, 0.0],
                "chunk-1": [],
                "chunk-2": [0.5, 0.5],
            },
            "empty_candidate_vector",
            "invalid_vector_count",
        ),
    ],
)
def test_candidate_rq_fails_closed_on_invalid_frozen_vector_scope(
    vectors,
    expected_reason,
    count_field,
    monkeypatch,
):
    from app.services import auto_tpe

    chunks = [
        SimpleNamespace(
            id=f"chunk-{index}",
            chunk_index=index,
            text_hash=f"{index:064x}"[-64:],
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        auto_tpe,
        "train_rq_kmeans",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid frozen vector scope must fail before RQ training"
        ),
    )

    prefixes, diagnostics = auto_tpe._candidate_rq_prefix_inputs(
        chunks,
        vectors,
        chunk_business_keys={
            f"chunk-{index}": f"business-{index}" for index in range(3)
        },
        canonical_business_keys_are_production=True,
    )

    assert prefixes == {}
    assert diagnostics["enabled"] is False
    assert diagnostics["reason"] == "invalid_frozen_candidate_vector_scope"
    assert expected_reason in diagnostics["reasons"]
    assert diagnostics[count_field] > 0
    assert diagnostics["frozen_chunk_count"] == 3
    assert diagnostics["input_hash"] == diagnostics["frozen_vector_input_hash"]
    assert diagnostics["model_call_count"] == 0


def test_candidate_rq_partial_assignment_forces_insufficient_evaluation(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import auto_tpe

    chunks = [
        SimpleNamespace(
            id=f"chunk-{index}",
            knowledge_base_id=sample_knowledge_base.id,
            document_id=f"document-{index}",
            document_version_id=f"version-{index}",
            chunk_version=1,
            chunk_index=index,
            text=f"chunk {index}",
            text_hash=f"{index:064x}"[-64:],
            char_start=index * 20,
            char_end=index * 20 + 18,
            token_start=index * 4,
            token_end=index * 4 + 3,
            section_path=f"section-{index}",
            page_start=index + 1,
            page_end=index + 1,
            previous_chunk_id=None,
            next_chunk_id=None,
            metadata_json={
                "tpe_expected_support_chunk_ids": [f"chunk-{1 - index}"]
            },
        )
        for index in range(2)
    ]
    structure_targets = {
        str(chunk.id): {
            "previous_next": {f"chunk-{1 - index}"},
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        }
        for index, chunk in enumerate(chunks)
    }
    monkeypatch.setattr(
        auto_tpe,
        "_structure_positive_context",
        lambda *_args, **_kwargs: (
            structure_targets,
            {"chunk-0", "chunk-1"},
            {
                "protocol_version": auto_tpe.TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
                "category_positive_counts": {
                    "previous_next": 2,
                    "same_section": 0,
                    "same_page": 0,
                    "special_object_closure": 0,
                },
                "positive_count": 2,
                "model_call_count": 0,
            },
        ),
    )
    monkeypatch.setattr(
        auto_tpe,
        "load_chunk_language_identities",
        lambda *_args, **_kwargs: {
            "chunk-0": {"known": True, "language": "en"},
            "chunk-1": {"known": True, "language": "en"},
        },
    )

    _hard_gate, components, score, failure, _probe_hash = (
        auto_tpe.evaluate_candidate_trial(
            db_session,
            chunks,
            {},
            {},
            [1.0],
            {"tpe_probe_query_budget": 2},
            candidate_rq_prefixes_by_chunk={"chunk-0": {"L3:1.1.1"}},
            candidate_rq_diagnostics={
                "enabled": True,
                "input_hash": "6" * 64,
                "model_call_count": 0,
            },
        )
    )

    assert failure == "insufficient_evaluation"
    assert score == 0.0
    rq_diagnostics = components["evaluation_input"]["candidate_rq_input"]
    assert rq_diagnostics["enabled"] is False
    assert rq_diagnostics["reason"] == "invalid_candidate_rq_assignment_scope"
    assert rq_diagnostics["missing_assignment_count"] == 1
    assert "candidate_rq_input_unavailable" in components["evaluation_input"][
        "blocking_reasons"
    ]
    assert components["evaluation_input"]["gray_zone_rule_decision_model_call_count"] == 0


def test_tpe_search_space_samples_real_calibration_dimensions():
    from app.services import auto_tpe
    from app.services.context_graph import dense_graph_operating_point

    base = dense_graph_operating_point()
    samples = [auto_tpe._random_theta(base, seed) for seed in range(1, 6)]
    assert all(auto_tpe.preflight_theta(sample) == [] for sample in samples)
    assert len(
        {
            tuple(sorted(sample["calibration_params"].items()))
            for sample in samples
        }
    ) > 1
    search_space = auto_tpe.tpe_search_space_spec()
    assert set(search_space["calibration_param_bounds"]) == {
        "lower_quantile",
        "upper_quantile",
        "min_span",
        "strength_floor",
    }
    assert set(auto_tpe.TPE_NUMERIC_THETA_PATHS) >= {
        "calibration_params.lower_quantile",
        "calibration_params.upper_quantile",
        "calibration_params.min_span",
        "calibration_params.strength_floor",
    }
    assert search_space["immutable_identity"]["edge_distance_protocol_hash"]
    assert search_space["immutable_identity"][
        "edge_type_calibration_protocol_hash"
    ]
    assert search_space["immutable_identity"]["relation_quota_protocol_hash"]
    assert set(search_space["theta_fields"]) == set(auto_tpe.TPE_ALLOWED_THETA_KEYS)
    assert "edge_distance_protocol_hash" not in search_space["sampled_fields"]
    assert auto_tpe.tpe_search_space_hash()


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda theta: theta.pop("calibration_params"), "missing:calibration_params"),
        (
            lambda theta: theta.update(
                edge_type_calibration_protocol="identity_calibration_v0"
            ),
            "invalid_edge_calibration:",
        ),
        (
            lambda theta: theta["calibration_params"].update(lower_quantile=0.5),
            "out_of_bounds:calibration_params.lower_quantile",
        ),
        (
            lambda theta: theta["calibration_params"].update(
                lower_quantile="0.05"
            ),
            "invalid_float:calibration_params.lower_quantile",
        ),
        (
            lambda theta: theta.update(edge_distance_protocol_hash="0" * 64),
            "invalid_edge_calibration:",
        ),
    ],
)
def test_tpe_preflight_fails_before_simulation_for_calibration_drift(
    mutation,
    expected_reason,
):
    from app.services.auto_tpe import preflight_theta
    from app.services.context_graph import dense_graph_operating_point

    theta = dense_graph_operating_point()
    mutation(theta)
    assert any(
        reason == expected_reason or reason.startswith(expected_reason)
        for reason in preflight_theta(theta)
    )


def test_tpe_theta_pydantic_contract_is_closed_and_calibration_typed():
    from pydantic import ValidationError

    from app.schemas import AutoTpeGraphOperatingPointTheta
    from app.services.context_graph import dense_graph_operating_point

    theta = dense_graph_operating_point()
    validated = AutoTpeGraphOperatingPointTheta.model_validate(theta)
    assert validated.calibration_params.lower_quantile == 0.05

    with pytest.raises(ValidationError):
        AutoTpeGraphOperatingPointTheta.model_validate(
            {**theta, "placeholder_protocol_hash": "0" * 64}
        )
    invalid = dict(theta)
    invalid["calibration_params"] = {
        **theta["calibration_params"],
        "min_span": 0.0,
    }
    with pytest.raises(ValidationError):
        AutoTpeGraphOperatingPointTheta.model_validate(invalid)


@pytest.fixture(autouse=True)
def _isolate_auto_tpe_unit_tests_from_model_bridge(monkeypatch):
    """Unit graph fixtures must not depend on the host model-bridge process."""

    from app.services import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "sync_model_bridge_runtime_config",
        lambda **_kwargs: {"attempted": False, "reason": "unit_test_isolation"},
    )


def _active_chunks_and_vectors(db_session, knowledge_base_id: str):
    from sqlalchemy import select

    from app.models import Chunk
    from app.services.context_graph import vector_for_chunk

    chunks = list(db_session.scalars(select(Chunk).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active")).all())
    vectors = {chunk.id: vector_for_chunk(db_session, chunk.id) for chunk in chunks}
    return chunks, vectors


def test_tpe_gamma_splits_only_completed_trials_by_descending_objective(monkeypatch):
    from datetime import datetime

    from app.services import auto_tpe
    from app.services.context_graph import dense_graph_operating_point

    settings = SimpleNamespace(
        tpe_startup_random_trials=1,
        tpe_good_quantile_gamma=0.5,
        tpe_candidate_pool_size=1,
    )
    monkeypatch.setattr(auto_tpe, "refresh_runtime_settings_if_needed", lambda **_kwargs: None)
    monkeypatch.setattr(auto_tpe, "get_settings", lambda: settings)
    monkeypatch.setattr(auto_tpe, "preflight_theta", lambda _theta: [])

    run = SimpleNamespace(
        id="run-gamma",
        knowledge_base_id="kb-gamma",
        chunk_version=1,
        chat_model="chat",
        embedding_model="embedding",
        embedding_text_version="embedding-text-v1",
    )
    gate_profile = {
        "protocol": "tpe_hard_gate_profile_v2",
        "tpe_probe_query_budget": 1,
        "tpe_trial_timeout_seconds": 30.0,
        "hard_gate_thresholds": {
            name: {
                "direction": gate["direction"],
                "threshold": gate["threshold"],
            }
            for name, gate in _passing_hard_gate().items()
        },
    }
    gate_profile_hash = auto_tpe.tpe_gate_profile_hash(gate_profile)

    def trial(index: int, objective: float, *, status: str = "completed"):
        sampled_theta = dense_graph_operating_point()
        sampled_theta["dense_min_cosine"] = min(0.9, 0.2 + objective / 1000.0)
        calibration_audit = auto_tpe.theta_calibration_audit(sampled_theta)
        simulated = _simulated_calibration_diagnostics(sampled_theta)[
            "edge_type_calibration"
        ]
        return SimpleNamespace(
            trial_index=index,
            theta_hash=auto_tpe._theta_hash(sampled_theta),
            status=status,
            objective_score=objective,
            failure_code=None,
            runtime_settings_hash="r" * 64,
            gate_profile_hash=gate_profile_hash,
            gate_profile_json=gate_profile,
            chunk_scope_hash="s" * 64,
            embedding_model="embedding",
            embedding_text_version="embedding-text-v1",
            candidate_adjacency_hash=f"candidate-{index}",
            hard_gate_json=_passing_hard_gate(),
            finished_at=datetime.utcnow(),
            sampled_theta_json=sampled_theta,
            tpe_search_space_hash=auto_tpe.tpe_search_space_hash(),
            edge_distance_protocol=calibration_audit["edge_distance_protocol"],
            edge_distance_protocol_hash=calibration_audit[
                "edge_distance_protocol_hash"
            ],
            edge_type_calibration_protocol=calibration_audit[
                "edge_type_calibration_protocol"
            ],
            edge_type_calibration_protocol_hash=calibration_audit[
                "edge_type_calibration_protocol_hash"
            ],
            calibration_params_json=dict(calibration_audit["calibration_params"]),
            calibration_params_hash=calibration_audit["calibration_params_hash"],
            edge_type_calibration_config_hash=calibration_audit[
                "edge_type_calibration_config_hash"
            ],
            diagnostics_json={
                "theta_calibration_audit": calibration_audit,
                "simulated_edge_type_calibration": simulated,
            },
        )

    completed_trials = [
        trial(1, 9.0),
        trial(2, 7.0),
        trial(3, 4.0),
        trial(4, 1.0),
        trial(5, 99.0, status="blocked"),
    ]
    observed_groups: list[tuple[int, ...]] = []
    original_mean_for = auto_tpe._mean_for

    def audited_mean_for(trials, key, base):
        observed_groups.append(tuple(item.trial_index for item in trials))
        return original_mean_for(trials, key, base)

    monkeypatch.setattr(auto_tpe, "_mean_for", audited_mean_for)
    sampled, sampler_hash = auto_tpe._sample_tpe_theta(
        run,
        completed_trials,
        trial_index=6,
        base_theta=dense_graph_operating_point(),
        settings=settings,
        runtime_settings_hash="r" * 64,
        gate_profile_hash=gate_profile_hash,
    )

    assert sampled
    assert sampler_hash
    assert (1, 2) in observed_groups
    assert (3, 4) in observed_groups
    assert all(5 not in group for group in observed_groups)
    incompatible = trial(7, 500.0)
    incompatible.runtime_settings_hash = "x" * 64
    assert sampler_hash == auto_tpe._sampler_state_hash(
        run,
        [*completed_trials, incompatible],
        6,
        runtime_settings_hash="r" * 64,
        gate_profile_hash=gate_profile_hash,
    )
    rejected = trial(8, 900.0, status="blocked")
    assert sampler_hash == auto_tpe._sampler_state_hash(
        run,
        [*completed_trials, rejected],
        6,
        runtime_settings_hash="r" * 64,
        gate_profile_hash=gate_profile_hash,
    )
    late_profile_settings = SimpleNamespace(
        tpe_startup_random_trials=5,
        tpe_good_quantile_gamma=0.5,
        tpe_candidate_pool_size=1,
    )
    monkeypatch.setattr(
        auto_tpe,
        "_random_theta",
        lambda _base, _seed, **_kwargs: {"startup_random": True},
    )
    late_sample, _late_hash = auto_tpe._sample_tpe_theta(
        run,
        completed_trials,
        trial_index=99,
        base_theta=dense_graph_operating_point(),
        settings=late_profile_settings,
        runtime_settings_hash="r" * 64,
        gate_profile_hash=gate_profile_hash,
    )
    assert late_sample == {"startup_random": True}


def test_completed_trial_audit_rejects_calibration_identity_mismatch():
    from datetime import datetime

    from app.models import AutoTpeTrial
    from app.services import auto_tpe
    from app.services.context_graph import dense_graph_operating_point
    from app.services.tpe_audit import tpe_trial_is_valid

    theta = dense_graph_operating_point()
    audit = auto_tpe.theta_calibration_audit(theta)
    gate_profile = {
        "protocol": "tpe_hard_gate_profile_v2",
        "tpe_probe_query_budget": 1,
        "tpe_trial_timeout_seconds": 30.0,
        "hard_gate_thresholds": {
            name: {
                "direction": gate["direction"],
                "threshold": gate["threshold"],
            }
            for name, gate in _passing_hard_gate().items()
        },
    }
    trial = AutoTpeTrial(
        run_id="run-calibration-audit",
        knowledge_base_id="kb-calibration-audit",
        chunk_scope_hash="1" * 64,
        embedding_model="embedding",
        embedding_text_version="embedding-text-v1",
        trial_index=1,
        sampled_theta_json=theta,
        theta_hash=auto_tpe._theta_hash(theta),
        tpe_search_space_hash=auto_tpe.tpe_search_space_hash(),
        edge_distance_protocol=audit["edge_distance_protocol"],
        edge_distance_protocol_hash=audit["edge_distance_protocol_hash"],
        edge_type_calibration_protocol=audit["edge_type_calibration_protocol"],
        edge_type_calibration_protocol_hash=audit[
            "edge_type_calibration_protocol_hash"
        ],
        calibration_params_json=dict(audit["calibration_params"]),
        calibration_params_hash=audit["calibration_params_hash"],
        edge_type_calibration_config_hash=audit[
            "edge_type_calibration_config_hash"
        ],
        sampler_state_hash="2" * 64,
        runtime_settings_hash="3" * 64,
        gate_profile_hash=auto_tpe.tpe_gate_profile_hash(gate_profile),
        gate_profile_json=gate_profile,
        candidate_adjacency_hash="4" * 64,
        hard_gate_json=_passing_hard_gate(),
        objective_score=0.5,
        status="completed",
        diagnostics_json={
            "theta_calibration_audit": audit,
            "simulated_edge_type_calibration": _simulated_calibration_diagnostics(
                theta
            )["edge_type_calibration"],
        },
        finished_at=datetime.utcnow(),
    )
    assert tpe_trial_is_valid(trial) is True
    trial.edge_type_calibration_config_hash = "f" * 64
    assert tpe_trial_is_valid(trial) is False


def test_tpe_runtime_hash_uses_pure_configuration_snapshot(monkeypatch):
    from copy import deepcopy

    from app.services import auto_tpe, runtime_settings

    settings = deepcopy(runtime_settings.get_settings())
    dynamic_calls = 0

    def dynamic_status_must_not_run(*_args, **_kwargs):
        nonlocal dynamic_calls
        dynamic_calls += 1
        raise AssertionError("runtime hash must not probe model bridge status")

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime_settings, "read_env_int", lambda _name, default: default)
    monkeypatch.setattr(runtime_settings, "read_env_float", lambda _name, default: default)
    monkeypatch.setattr(runtime_settings, "read_env_bool", lambda _name, default: default)
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", dynamic_status_must_not_run)
    monkeypatch.setattr(
        runtime_settings,
        "current_runtime_settings_version",
        dynamic_status_must_not_run,
    )

    baseline_hash = auto_tpe._runtime_hash()
    settings.tpe_trial_budget += 1
    changed_hash = auto_tpe._runtime_hash()

    assert dynamic_calls == 0
    assert baseline_hash != changed_hash


def test_candidate_adjacency_hash_binds_theta_and_typed_gate_decision():
    from app.services import auto_tpe
    from app.services.context_graph import RelationEdgeCandidate

    def candidate(
        *,
        mutual: bool,
        reverse_accepted: bool,
        language_scope_hash: str = "3" * 64,
    ) -> RelationEdgeCandidate:
        return RelationEdgeCandidate(
            source_chunk_id="chunk-a",
            target_chunk_id="chunk-b",
            edge_type="dense_semantic",
            raw_strength=0.75,
            calibrated_strength=0.8,
            distance=0.2,
            features_json={
                "cosine": 0.9,
                "language_identity_scope_hash": language_scope_hash,
                "normalization_stats": {
                    "type_threshold": 0.7,
                    "strong_threshold": 0.95,
                    "mutual": mutual,
                    "reverse_accepted": reverse_accepted,
                },
            },
        )

    mutual_candidates = {
        ("chunk-a", "chunk-b", "dense_semantic"): candidate(
            mutual=True,
            reverse_accepted=False,
        )
    }
    reverse_candidates = {
        ("chunk-a", "chunk-b", "dense_semantic"): candidate(
            mutual=False,
            reverse_accepted=True,
        )
    }
    mutual_hash = auto_tpe._candidate_adjacency_hash(
        mutual_candidates,
        theta_hash="1" * 64,
        language_identity_scope_hash="3" * 64,
    )
    assert mutual_hash != auto_tpe._candidate_adjacency_hash(
        mutual_candidates,
        theta_hash="2" * 64,
        language_identity_scope_hash="3" * 64,
    )
    assert mutual_hash != auto_tpe._candidate_adjacency_hash(
        reverse_candidates,
        theta_hash="1" * 64,
        language_identity_scope_hash="3" * 64,
    )
    different_language_scope = {
        ("chunk-a", "chunk-b", "dense_semantic"): candidate(
            mutual=True,
            reverse_accepted=False,
            language_scope_hash="4" * 64,
        )
    }
    assert mutual_hash != auto_tpe._candidate_adjacency_hash(
        different_language_scope,
        theta_hash="1" * 64,
        language_identity_scope_hash="4" * 64,
    )


def test_tpe_quality_proxies_use_expected_support_structure_and_four_dimensions(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import auto_tpe

    def chunk(
        chunk_id: str,
        index: int,
        document_id: str,
        *,
        expected_support_chunk_id: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=chunk_id,
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document_id,
            document_version_id=f"version-{document_id}",
            chunk_index=index,
            text_hash=(chunk_id[-1] * 64),
            char_start=index * 20,
            char_end=index * 20 + 18,
            token_start=index * 4,
            token_end=index * 4 + 3,
            section_path=f"section-{document_id}",
            page_start=index + 1,
            page_end=index + 1,
            previous_chunk_id=None,
            next_chunk_id=None,
            metadata_json={
                "tpe_expected_support_chunk_ids": [expected_support_chunk_id]
            },
        )

    chunks = [
        chunk("chunk-1", 0, "document-1", expected_support_chunk_id="chunk-2"),
        chunk("chunk-2", 1, "document-2", expected_support_chunk_id="chunk-1"),
        chunk("chunk-3", 2, "document-3", expected_support_chunk_id="chunk-2"),
    ]
    structure_targets = {
        "chunk-1": {
            "previous_next": {"chunk-2"},
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        },
        "chunk-2": {
            "previous_next": {"chunk-1"},
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        },
        "chunk-3": {
            "previous_next": {"chunk-2"},
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        },
    }
    monkeypatch.setattr(
        auto_tpe,
        "_structure_positive_context",
        lambda *_args, **_kwargs: (
            structure_targets,
            {"chunk-1", "chunk-2", "chunk-3"},
            {
                "protocol_version": auto_tpe.TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
                "category_positive_counts": {
                    "previous_next": 3,
                    "same_section": 0,
                    "same_page": 0,
                    "special_object_closure": 0,
                },
                "positive_count": 3,
                "model_call_count": 0,
            },
        ),
    )
    languages = {
        "chunk-1": {"known": True, "language": "en"},
        "chunk-2": {"known": True, "language": "zh"},
        "chunk-3": {"known": True, "language": "en"},
    }
    monkeypatch.setattr(
        auto_tpe,
        "load_chunk_language_identities",
        lambda *_args, **_kwargs: languages,
    )
    rq_inputs = {
        "chunk-1": {"L3:1.1.1"},
        "chunk-2": {"L3:2.1.1"},
        "chunk-3": {"L3:3.1.1"},
    }
    rq_diagnostics = {
        "enabled": True,
        "input_hash": "7" * 64,
        "model_call_count": 0,
    }
    gate_profile = {
        "tpe_probe_query_budget": 3,
        "hard_gate_thresholds": {
            name: {
                "direction": card["direction"],
                "threshold": card["threshold"],
            }
            for name, card in _passing_hard_gate().items()
        },
    }

    sparse = {
        ("chunk-1", "chunk-3", "dense_semantic"): _quality_candidate(
            "chunk-1", "chunk-3"
        )
    }
    _hard_gate, sparse_components, _score, _failure, _probe_hash = (
        auto_tpe.evaluate_candidate_trial(
            db_session,
            chunks,
            sparse,
            {},
            [1.0],
            gate_profile,
            candidate_rq_prefixes_by_chunk=rq_inputs,
            candidate_rq_diagnostics=rq_diagnostics,
        )
    )
    # Merely giving a probe any edge cannot count as expected-support recall.
    assert sparse_components["evidence_recall_proxy"]["value"] == 0.0
    assert sparse_components["structure_recovery_rate"]["value"] == 0.0
    assert sparse_components["component_coverage"]["dimensions"] == {
        "active_chunks": {
            "numerator": 2,
            "denominator": 3,
            "coverage": 0.666667,
            "weight": 0.25,
        },
        "documents": {
            "numerator": 2,
            "denominator": 3,
            "coverage": 0.666667,
            "weight": 0.25,
        },
        "language_buckets": {
            "numerator": 1,
            "denominator": 2,
            "coverage": 0.5,
            "weight": 0.25,
        },
        "candidate_rq_prefixes": {
            "numerator": 2,
            "denominator": 3,
            "coverage": 0.666667,
            "weight": 0.25,
        },
    }
    assert sparse_components["component_coverage"]["value"] == 0.625
    assert set(sparse_components["path_diversity"]["dimensions"]) == {
        "documents",
        "languages",
        "edge_types",
        "candidate_rq_prefixes",
    }
    assert sparse_components["edge_precision_proxy"]["value"] == 1.0
    assert sparse_components["evaluation_input"][
        "gray_zone_rule_decision_model_call_count"
    ] == 0

    missing_support = _quality_candidate("chunk-1", "chunk-3")
    missing_support.features_json.pop("raw_strength_components")
    _hard_gate, unsupported_components, _score, _failure, _probe_hash = (
        auto_tpe.evaluate_candidate_trial(
            db_session,
            chunks,
            {
                ("chunk-1", "chunk-3", "dense_semantic"): missing_support
            },
            {},
            [1.0],
            gate_profile,
            candidate_rq_prefixes_by_chunk=rq_inputs,
            candidate_rq_diagnostics=rq_diagnostics,
        )
    )
    assert unsupported_components["edge_precision_proxy"]["value"] == 0.0
    assert unsupported_components["edge_precision_proxy"]["failure_counts"] == {
        "raw_strength_protocol": 1,
        "raw_strength_inputs": 1,
        "raw_strength_replay": 1,
    }

    chain = {
        **sparse,
        ("chunk-2", "chunk-3", "dense_semantic"): _quality_candidate(
            "chunk-2", "chunk-3"
        ),
    }
    _hard_gate, chain_components, _score, _failure, _probe_hash = (
        auto_tpe.evaluate_candidate_trial(
            db_session,
            chunks,
            chain,
            {},
            [1.0],
            gate_profile,
            candidate_rq_prefixes_by_chunk=rq_inputs,
            candidate_rq_diagnostics=rq_diagnostics,
        )
    )
    assert chain_components["evidence_recall_proxy"]["value"] == 1.0
    assert chain_components["evidence_recall_proxy"]["denominator"] == 3.0
    assert chain_components["structure_recovery_rate"]["value"] == 1.0
    assert chain_components["structure_recovery_rate"]["categories"][
        "previous_next"
    ] == {"hits": 3, "denominator": 3, "value": 1.0}


def test_tpe_bridge_opportunity_requires_the_dedicated_typed_quota(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import auto_tpe

    chunks = [
        SimpleNamespace(
            id=f"chunk-{index}",
            knowledge_base_id=sample_knowledge_base.id,
            document_id=f"document-{index}",
            document_version_id=f"version-{index}",
            chunk_index=index,
            text_hash=str(index) * 64,
            char_start=index * 10,
            char_end=index * 10 + 8,
            token_start=index,
            token_end=index + 1,
            section_path=f"section-{index}",
            page_start=index,
            page_end=index,
            previous_chunk_id=None,
            next_chunk_id=None,
            metadata_json={
                "tpe_expected_support_chunk_ids": [
                    f"chunk-{2 if index == 1 else 1}"
                ]
            },
        )
        for index in (1, 2)
    ]
    structure_targets = {
        "chunk-1": {
            "previous_next": {"chunk-2"},
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        },
        "chunk-2": {
            "previous_next": {"chunk-1"},
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        },
    }
    monkeypatch.setattr(
        auto_tpe,
        "_structure_positive_context",
        lambda *_args, **_kwargs: (
            structure_targets,
            {"chunk-1", "chunk-2"},
            {
                "protocol_version": auto_tpe.TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
                "category_positive_counts": {
                    "previous_next": 2,
                    "same_section": 0,
                    "same_page": 0,
                    "special_object_closure": 0,
                },
                "positive_count": 2,
                "model_call_count": 0,
            },
        ),
    )
    monkeypatch.setattr(
        auto_tpe,
        "load_chunk_language_identities",
        lambda *_args, **_kwargs: {
            "chunk-1": {"known": True, "language": "en"},
            "chunk-2": {"known": True, "language": "en"},
        },
    )
    gate_profile = {
        "tpe_probe_query_budget": 2,
        "hard_gate_thresholds": {
            name: {
                "direction": card["direction"],
                "threshold": card["threshold"],
            }
            for name, card in _passing_hard_gate().items()
        },
    }

    def evaluate(channels: list[str]) -> dict[str, object]:
        candidates = {
            (
                "chunk-1",
                "chunk-2",
                "dense_cross_document_bridge",
            ): _quality_candidate(
                "chunk-1",
                "chunk-2",
                edge_type="dense_cross_document_bridge",
                candidate_channels=channels,
            )
        }
        return auto_tpe.evaluate_candidate_trial(
            db_session,
            chunks,
            candidates,
            {},
            [1.0],
            gate_profile,
            candidate_rq_prefixes_by_chunk={
                "chunk-1": {"L3:1.1.1"},
                "chunk-2": {"L3:2.1.1"},
            },
            candidate_rq_diagnostics={"enabled": True, "input_hash": "6" * 64},
        )[1]

    base_only = evaluate(["base_dense_candidates"])
    assert base_only["bridge_opportunity_coverage"]["value"] == 0.0
    dedicated = evaluate(["cross_document_candidates"])
    assert dedicated["bridge_opportunity_coverage"]["value"] == 1.0
    assert dedicated["bridge_opportunity_coverage"][
        "dedicated_quota_edge_count"
    ] == 1


@pytest.mark.asyncio
async def test_tpe_real_structure_positives_and_candidate_rq_inputs_are_auditable(
    db_session,
    populated_context_graph,
):
    from app.services import auto_tpe

    knowledge_base = populated_context_graph["knowledge_base"]
    chunks, vectors = _active_chunks_and_vectors(db_session, knowledge_base.id)
    rq_inputs, rq_diagnostics = auto_tpe._candidate_rq_prefix_inputs(
        chunks, vectors
    )

    assert rq_diagnostics["enabled"] is True
    assert rq_diagnostics["model_call_count"] == 0
    assert rq_diagnostics["input_hash"]
    assert set(rq_inputs) == {chunk.id for chunk in chunks}
    assert all(
        any(prefix.startswith("L3:") for prefix in prefixes)
        for prefixes in rq_inputs.values()
    )

    production_candidates, _candidate_diagnostics = (
        auto_tpe.relation_edge_candidates(
            db_session,
            chunks,
            vectors,
            auto_tpe.dense_graph_operating_point(),
        )
    )
    assert production_candidates
    production_audits = [
        auto_tpe._candidate_support_feature_audit(candidate)
        for candidate in production_candidates.values()
    ]
    assert all(audit["passed"] for audit in production_audits), [
        audit for audit in production_audits if not audit["passed"]
    ]
    assert all(
        audit["replay"]["raw_strength"] == candidate.raw_strength
        for audit, candidate in zip(
            production_audits,
            production_candidates.values(),
            strict=True,
        )
    )

    structure_targets, traceable_ids, structure_diagnostics = (
        auto_tpe._structure_positive_context(db_session, chunks, chunks)
    )
    assert structure_diagnostics["model_call_count"] == 0
    assert structure_diagnostics["category_positive_counts"][
        "previous_next"
    ] > 0
    assert structure_diagnostics["category_positive_counts"]["same_page"] > 0
    assert structure_diagnostics["category_positive_counts"][
        "special_object_closure"
    ] > 0
    assert traceable_ids
    assert traceable_ids.issubset({chunk.id for chunk in chunks})

    expected, expected_diagnostics = auto_tpe._expected_support_context(
        db_session,
        chunks,
        chunks,
        structure_targets,
    )
    assert expected_diagnostics["pair_count"] >= 2
    assert expected_diagnostics["probe_with_positive_count"] >= 2
    assert expected_diagnostics["model_call_count"] == 0
    assert expected_diagnostics["expected_support_pairs_hash"]
    assert any(expected.values())


def test_runtime_disable_stops_before_the_next_trial_boundary(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from copy import deepcopy
    from sqlalchemy import select

    from app.models import AutoTpeRun, AutoTpeTrial
    from app.services import auto_tpe
    from app.services.context_graph import auto_tpe_promotion_identity, dense_graph_operating_point

    settings = deepcopy(auto_tpe.get_settings())
    settings.enable_auto_tpe = True
    settings.tpe_trial_budget = 3
    settings.tpe_startup_random_trials = 1
    settings.tpe_probe_query_budget = 1
    settings.tpe_candidate_pool_size = 1
    settings.tpe_trial_timeout_seconds = 30
    refresh_count = 0

    def refresh(**_kwargs):
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count >= 3:
            settings.enable_auto_tpe = False

    monkeypatch.setattr(auto_tpe, "refresh_runtime_settings_if_needed", refresh)
    monkeypatch.setattr(auto_tpe, "get_settings", lambda: settings)
    monkeypatch.setattr(
        auto_tpe,
        "_sample_tpe_theta",
        lambda _run, _completed, _index, base, **_kwargs: (dict(base), "a" * 64),
    )
    monkeypatch.setattr(auto_tpe, "preflight_theta", lambda _theta: [])
    monkeypatch.setattr(
        auto_tpe,
        "relation_edge_candidates",
        lambda *_args, **_kwargs: (
            {},
            _simulated_calibration_diagnostics(_args[-1]),
        ),
    )
    monkeypatch.setattr(
        auto_tpe,
        "evaluate_candidate_trial",
        lambda *_args, **_kwargs: (
            _passing_hard_gate(),
            {"score": {"value": 0.5}},
            0.5,
            None,
            "b" * 64,
        ),
    )
    chunk = SimpleNamespace(
        knowledge_base_id=sample_knowledge_base.id,
        id="runtime-toggle-chunk",
        document_id="runtime-toggle-document",
        document_version_id="runtime-toggle-version",
        chunk_version=1,
        chunk_index=0,
        text_hash="c" * 64,
        token_start=0,
        token_end=3,
        char_start=0,
        char_end=10,
        section_path="Runtime toggle",
        page_start=1,
        page_end=1,
        metadata_json={
            "chunk_schema_version": "chunk_schema_v1",
            "tokenizer_version": "symbograph_regex_tokenizer_v1",
            "chunk_size": 512,
            "chunk_overlap": 80,
        },
    )

    selected, context = auto_tpe.select_auto_tpe_operating_point(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        batch_id=None,
        chunks=[chunk],
        vectors={chunk.id: [1.0, 0.0]},
        fallback_operating_point=dense_graph_operating_point(),
        chunk_version_incremented=True,
    )

    run = db_session.scalar(
        select(AutoTpeRun).where(
            AutoTpeRun.knowledge_base_id == sample_knowledge_base.id
        )
    )
    trials = list(
        db_session.scalars(
            select(AutoTpeTrial)
            .where(AutoTpeTrial.run_id == run.id)
            .order_by(AutoTpeTrial.trial_index)
        ).all()
    )
    assert selected == dense_graph_operating_point()
    assert context["status"] == "cancelled"
    assert context["reason"] == "auto_tpe_disabled_at_trial_boundary"
    assert auto_tpe_promotion_identity(context) == (None, None)
    assert context["trial_boundary"] == 2
    assert run.status == "cancelled"
    assert run.failure_code == "auto_tpe_disabled_at_trial_boundary"
    assert len(trials) == 1
    assert trials[0].status == "completed"
    assert trials[0].build_batch_id is None
    assert trials[0].chunk_scope_hash == run.chunk_scope_hash
    assert trials[0].embedding_model == run.embedding_model
    assert trials[0].embedding_text_version == run.embedding_text_version


def test_runtime_profile_change_after_trial_uses_audited_fallback(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from copy import deepcopy
    from sqlalchemy import select

    from app.models import AutoTpeRun, AutoTpeTrial
    from app.services import auto_tpe
    from app.services.context_graph import auto_tpe_promotion_identity, dense_graph_operating_point

    settings = deepcopy(auto_tpe.get_settings())
    settings.enable_auto_tpe = True
    settings.tpe_trial_budget = 1
    settings.tpe_startup_random_trials = 1
    settings.tpe_probe_query_budget = 1
    settings.tpe_candidate_pool_size = 1
    settings.tpe_trial_timeout_seconds = 30
    settings.operating_point_hard_gate_max_edge_density = 1.0
    settings.operating_point_hard_gate_max_isolated_ratio = 1.0
    settings.operating_point_hard_gate_max_hubness_ratio = 1000.0
    settings.operating_point_hard_gate_min_structure_recovery_rate = 0.0
    settings.operating_point_hard_gate_max_candidate_latency_p95_ms = 600000.0
    runtime_hashes = iter(["a" * 64, "a" * 64, "b" * 64])

    monkeypatch.setattr(auto_tpe, "refresh_runtime_settings_if_needed", lambda **_kwargs: None)
    monkeypatch.setattr(auto_tpe, "get_settings", lambda: settings)
    monkeypatch.setattr(auto_tpe, "_runtime_hash", lambda: next(runtime_hashes))
    monkeypatch.setattr(
        auto_tpe,
        "_sample_tpe_theta",
        lambda _run, _completed, _index, base, **_kwargs: (dict(base), "c" * 64),
    )
    monkeypatch.setattr(auto_tpe, "preflight_theta", lambda _theta: [])
    monkeypatch.setattr(
        auto_tpe,
        "relation_edge_candidates",
        lambda *_args, **_kwargs: (
            {},
            _simulated_calibration_diagnostics(_args[-1]),
        ),
    )
    monkeypatch.setattr(
        auto_tpe,
        "evaluate_candidate_trial",
        lambda *_args, **_kwargs: (
            _passing_hard_gate(),
            {"score": {"value": 0.5}},
            0.5,
            None,
            "d" * 64,
        ),
    )
    chunk = SimpleNamespace(
        knowledge_base_id=sample_knowledge_base.id,
        id="runtime-profile-chunk",
        document_id="runtime-profile-document",
        document_version_id="runtime-profile-version",
        chunk_version=1,
        chunk_index=0,
        text_hash="e" * 64,
        token_start=0,
        token_end=3,
        char_start=0,
        char_end=10,
        section_path="Runtime profile",
        page_start=1,
        page_end=1,
        metadata_json={
            "chunk_schema_version": "chunk_schema_v1",
            "tokenizer_version": "symbograph_regex_tokenizer_v1",
            "chunk_size": 512,
            "chunk_overlap": 80,
        },
    )
    fallback = dense_graph_operating_point()

    selected, context = auto_tpe.select_auto_tpe_operating_point(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        batch_id=None,
        chunks=[chunk],
        vectors={chunk.id: [1.0, 0.0]},
        fallback_operating_point=fallback,
        chunk_version_incremented=True,
    )

    run = db_session.scalar(
        select(AutoTpeRun).where(
            AutoTpeRun.knowledge_base_id == sample_knowledge_base.id
        )
    )
    trial = db_session.scalar(select(AutoTpeTrial).where(AutoTpeTrial.run_id == run.id))
    assert selected == fallback
    assert context["status"] == "failed"
    assert context["reason"] == "no_trial_matches_active_runtime_profile"
    assert auto_tpe_promotion_identity(context) == (None, None)
    assert run.status == "failed"
    assert run.failure_code == "no_trial_matches_active_runtime_profile"
    assert trial.status == "completed"
    assert trial.runtime_settings_hash == "a" * 64


def test_relation_graph_rejects_runtime_change_after_tpe_selection(
    db_session,
    populated_context_graph,
    monkeypatch,
):
    from sqlalchemy import func, select

    from app.models import ChunkRelationGraphState
    from app.services import auto_tpe, context_graph

    kb = populated_context_graph["knowledge_base"]
    chunks, _vectors = _active_chunks_and_vectors(db_session, kb.id)
    profile = auto_tpe.tpe_gate_profile()
    before_count = db_session.scalar(select(func.count(ChunkRelationGraphState.id)))
    monkeypatch.setattr(
        auto_tpe,
        "select_auto_tpe_operating_point",
        lambda *_args, **_kwargs: (
            context_graph.dense_graph_operating_point(),
            {
                "status": "selected_pending_graph_commit",
                "run_id": "runtime-mismatch-run",
                "best_trial_id": "runtime-mismatch-trial",
                "runtime_settings_hash": "f" * 64,
                "gate_profile_hash": auto_tpe.tpe_gate_profile_hash(profile),
                "gate_profile": profile,
            },
        ),
    )

    with pytest.raises(RuntimeError, match="Runtime settings changed after TPE"):
        context_graph.build_chunk_relation_graph(
            db_session,
            kb.id,
            chunks,
            auto_tpe_enabled_for_version=True,
        )

    assert db_session.scalar(select(func.count(ChunkRelationGraphState.id))) == before_count


def test_auto_tpe_skips_when_disabled_or_same_chunk_version(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import AutoTpeRun
    from app.services.auto_tpe import select_auto_tpe_operating_point
    from app.services.context_graph import dense_graph_operating_point

    kb = populated_context_graph["knowledge_base"]
    chunks, vectors = _active_chunks_and_vectors(db_session, kb.id)
    base_theta = dense_graph_operating_point()

    selected, context = select_auto_tpe_operating_point(
        db_session,
        knowledge_base_id=kb.id,
        batch_id="unit-batch-disabled",
        chunks=chunks,
        vectors=vectors,
        fallback_operating_point=base_theta,
        chunk_version_incremented=True,
    )
    assert selected == base_theta
    assert context["status"] == "skipped"
    assert context["reason"] == "disabled"
    assert db_session.scalar(select(func.count(AutoTpeRun.id))) == 0


@pytest.mark.asyncio
async def test_auto_tpe_runs_once_before_active_relation_graph_write(db_session, populated_context_graph, monkeypatch):
    from sqlalchemy import func, select

    from app.core.config import get_settings
    from app.models import AutoTpeRun, AutoTpeTrial, ChunkRelationGraphState, ContextGraphState
    from app.services import auto_tpe, context_graph
    from app.services.context_graph import rebuild_context_graph

    # The production logger intentionally commits through an independent
    # session.  SQLite serializes writers and would deadlock once the graph
    # transaction has registered its promotion intent; PostgreSQL coverage is
    # provided by the durable TPE audit suite.
    monkeypatch.setattr(auto_tpe, "emit_ingestion_log", lambda *_args, **_kwargs: None)
    # The same SQLite limitation applies to the independent cancellation
    # reader and graph heartbeat writer after this test's outer graph
    # transaction starts writing.  This case proves TPE ordering/promotion,
    # not cancellation/heartbeat durability; PostgreSQL and the dedicated
    # suites retain the independent-session behavior.
    monkeypatch.setattr(
        context_graph,
        "ensure_not_cancelled",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        context_graph,
        "context_graph_batch_heartbeat",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setenv("ENABLE_AUTO_TPE", "true")
    monkeypatch.setenv("TPE_TRIAL_BUDGET", "2")
    monkeypatch.setenv("TPE_STARTUP_RANDOM_TRIALS", "1")
    monkeypatch.setenv("TPE_PROBE_QUERY_BUDGET", "2")
    monkeypatch.setenv("TPE_TRIAL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("TPE_CANDIDATE_POOL_SIZE", "4")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO", "1000")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE", "0")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS", "600000")
    get_settings.cache_clear()

    kb = populated_context_graph["knowledge_base"]
    previous_relation_state_count = db_session.scalar(select(func.count(ChunkRelationGraphState.id)))
    state = await rebuild_context_graph(db_session, kb.id, batch_id="unit-batch-auto-tpe", chunk_version_incremented=True)
    db_session.flush()

    run = db_session.scalar(select(AutoTpeRun).where(AutoTpeRun.knowledge_base_id == kb.id).order_by(AutoTpeRun.created_at.desc()))
    assert run is not None
    assert run.status == "selected_pending_graph_commit"
    assert run.selected_theta_hash
    assert run.selected_graph_runtime_settings_hash
    assert (
        run.selected_graph_runtime_settings_hash
        == run.diagnostics_json["selected_graph_runtime_settings_hash"]
    )
    assert run.runtime_settings_hash == run.diagnostics_json["selected_runtime_settings_hash"]
    assert run.selected_graph_runtime_settings_hash != run.runtime_settings_hash
    assert run.tpe_search_space_hash
    assert run.selected_edge_distance_protocol_hash
    assert run.selected_edge_type_calibration_protocol_hash
    assert run.selected_calibration_params_hash
    assert run.selected_edge_type_calibration_config_hash
    assert run.best_trial_id
    assert run.chunk_relation_graph_state_id is None
    db_session.commit()
    db_session.expire_all()
    run = db_session.scalar(select(AutoTpeRun).where(AutoTpeRun.knowledge_base_id == kb.id).order_by(AutoTpeRun.created_at.desc()))
    assert run is not None
    assert run.status == "completed"
    relation_state = db_session.get(
        ChunkRelationGraphState,
        run.chunk_relation_graph_state_id,
    )
    assert relation_state is not None
    assert relation_state.runtime_settings_hash == run.selected_graph_runtime_settings_hash
    assert relation_state.diagnostics_json["auto_tpe"][
        "optimizer_runtime_settings_hash"
    ] == run.runtime_settings_hash
    assert run.chunk_relation_graph_state_id == state.chunk_relation_graph_state_id
    assert db_session.scalar(select(func.count(AutoTpeTrial.id)).where(AutoTpeTrial.run_id == run.id)) == 2

    relation_state = db_session.get(ChunkRelationGraphState, state.chunk_relation_graph_state_id)
    assert relation_state is not None
    assert relation_state.auto_tpe_run_id == run.id
    assert relation_state.auto_tpe_best_trial_id == run.best_trial_id
    assert relation_state.graph_operating_point_hash == run.selected_theta_hash
    assert (
        relation_state.diagnostics_json["calibration_params_hash"]
        == run.selected_calibration_params_hash
    )
    assert (
        relation_state.diagnostics_json["edge_type_calibration_config_hash"]
        == run.selected_edge_type_calibration_config_hash
    )
    assert db_session.scalar(select(func.count(ChunkRelationGraphState.id))) == previous_relation_state_count + 1
    assert db_session.scalar(select(func.count(ContextGraphState.id)).where(ContextGraphState.state == "shadow")) == 0


def test_auto_tpe_hard_gate_failure_uses_audited_fallback(db_session, populated_context_graph, monkeypatch):
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.models import AutoTpeRun
    from app.services.auto_tpe import select_auto_tpe_operating_point
    from app.services.context_graph import auto_tpe_promotion_identity, dense_graph_operating_point

    monkeypatch.setenv("ENABLE_AUTO_TPE", "true")
    monkeypatch.setenv("TPE_TRIAL_BUDGET", "1")
    monkeypatch.setenv("TPE_STARTUP_RANDOM_TRIALS", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY", "0.0001")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO", "1000")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE", "0")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS", "600000")
    get_settings.cache_clear()

    kb = populated_context_graph["knowledge_base"]
    chunks, vectors = _active_chunks_and_vectors(db_session, kb.id)

    fallback = dense_graph_operating_point()
    selected, context = select_auto_tpe_operating_point(
        db_session,
        knowledge_base_id=kb.id,
        batch_id="unit-batch-auto-tpe-fail",
        chunks=chunks,
        vectors=vectors,
        fallback_operating_point=fallback,
        chunk_version_incremented=True,
    )

    run = db_session.scalar(select(AutoTpeRun).where(AutoTpeRun.knowledge_base_id == kb.id).order_by(AutoTpeRun.created_at.desc()))
    assert selected == fallback
    assert context["status"] == "failed"
    assert context["auto_tpe_status"] == "failed_or_skipped"
    assert context["fallback_theta_hash"]
    assert auto_tpe_promotion_identity(context) == (None, None)
    assert run is not None
    assert run.status == "failed"
    assert run.failure_code == "no_trial_passed_hard_gate"
    assert "no_trial_passed_hard_gate" in (run.blocking_reasons_json or [])


def test_latest_auto_tpe_status_is_a_side_effect_free_read(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import auto_tpe

    monkeypatch.setattr(
        auto_tpe,
        "reconcile_tpe_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("status GET must not reconcile TPE state")
        ),
    )
    sample_knowledge_base.description = "dirty status read sentinel"
    monkeypatch.setattr(
        db_session,
        "flush",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("status GET must not autoflush caller state")
        ),
    )

    payload = auto_tpe.latest_auto_tpe_status(
        db_session,
        sample_knowledge_base.id,
    )

    assert payload["knowledge_base_id"] == sample_knowledge_base.id
    assert payload["current_chunk_version"] == (
        sample_knowledge_base.current_chunk_version or 0
    )
    assert payload["latest_run"] is None
