from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _load_quality_gate():
    spec = importlib.util.spec_from_file_location(
        "test_four_layer_quality_gate", SCRIPTS_ROOT / "_quality_gate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relation_stats(module, edge_type: str, raw: float, calibrated: float) -> dict:
    distance = module._distance_from_strength(calibrated)
    params = dict(module.EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS)
    protocol_hash = module.edge_type_calibration_protocol_hash()
    params_hash = module._relation_calibration_params_hash(params)
    config_hash = module._relation_calibration_config_hash(params)
    stats = {
        "edge_type": edge_type,
        "protocol_version": module.EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "calibration_params_hash": params_hash,
        "edge_type_calibration_config_hash": config_hash,
        "params": params,
        "sample_count": 1,
        "lower_quantile_value": raw,
        "upper_quantile_value": raw,
        "effective_lower_bound": raw,
        "effective_upper_bound": raw,
        "quantile_span": 0.0,
        "fallback": "insufficient_type_sample",
        "calibration_applied": False,
        "monotonic_violation_count": 0,
        "raw_strength_distribution": module._distribution([raw]),
        "calibrated_strength_distribution": module._distribution([calibrated]),
        "distance_distribution": module._distribution([distance]),
        "cross_type_raw_comparison_allowed": False,
    }
    stats["stats_hash"] = module.stable_hash(
        {
            "protocol": module.EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "edge_type": edge_type,
            "params": stats["params"],
            "raw_strengths": [raw],
            "fallback": stats["fallback"],
        }
    )
    return stats


def _projection_stats(module, layer: str, edge_type: str, strength: float) -> dict:
    distance = module._raw_projected_distance(strength)
    stats = {
        "normalization": module.PROJECTION_CALIBRATION_PROTOCOL_VERSION,
        "protocol_version": module.PROJECTION_CALIBRATION_PROTOCOL_VERSION,
        "edge_projection_protocol_version": module.EDGE_PROJECTION_PROTOCOL_VERSION,
        "edge_projection_protocol_hash": module.edge_projection_protocol_hash(),
        "layer": layer,
        "edge_type": edge_type,
        "scope": "layer_plus_edge_type",
        "params": dict(module.PROJECTION_CALIBRATION_PARAMS),
        "sample_count": 1,
        "lower_quantile_value": strength,
        "upper_quantile_value": strength,
        "quantile_span": 0.0,
        "fallback": "insufficient_layer_type_sample",
        "calibration_applied": False,
        "raw_strength_distribution": module._distribution([strength]),
        "calibrated_strength_distribution": module._distribution([strength]),
        "calibrated_distance_distribution": module._distribution([distance]),
        "cross_type_raw_comparison_allowed": False,
        "model_call_count": 0,
    }
    stats["stats_hash"] = module.stable_hash(stats)
    return stats


def _projected_edge_fixture(module, layer: str, relation: dict, chunk_keys: dict) -> tuple[dict, dict]:
    source_scope_key = "rq:source"
    target_scope_key = "rq:target"
    source_rq_prefix_id = "rq-source"
    target_rq_prefix_id = "rq-target"
    source_score = 0.8
    target_score = 0.75
    membership_product = source_score * target_score
    bottom_fact = {
        "endpoint_chunk_business_keys": sorted(
            (
                chunk_keys[relation["source_chunk_id"]],
                chunk_keys[relation["target_chunk_id"]],
            )
        ),
        "edge_type": relation["edge_type"],
        "distance": relation["distance"],
        "calibrated_strength": relation["weight"],
        "protocol_version": relation["protocol_version"],
        "graph_state_hash": relation["graph_state_hash"],
        "edge_distance_protocol_hash": relation["edge_distance_protocol_hash"],
    }
    bottom_fact_hash = module.stable_hash(bottom_fact)
    contribution = {
        "bottom_chunk_edge_id": relation["id"],
        "source_chunk_id": relation["source_chunk_id"],
        "target_chunk_id": relation["target_chunk_id"],
        "bottom_edge_type": relation["edge_type"],
        "bottom_distance": relation["distance"],
        "source_membership_score": source_score,
        "target_membership_score": target_score,
        "membership_product": membership_product,
        "orientation": "source_scope_to_target_scope",
        "assignment_protocol_version": module.ASSIGNMENT_PROTOCOL_VERSION,
        "bottom_edge_fact_hash": bottom_fact_hash,
    }
    canonical_contribution = {
        "bottom_edge_fact_hash": bottom_fact_hash,
        "endpoint_scope_memberships": sorted(
            [
                {
                    "scope_key": source_scope_key,
                    "chunk_business_key": chunk_keys[relation["source_chunk_id"]],
                    "membership_score": source_score,
                },
                {
                    "scope_key": target_scope_key,
                    "chunk_business_key": chunk_keys[relation["target_chunk_id"]],
                    "membership_score": target_score,
                },
            ],
            key=lambda item: (item["scope_key"], item["chunk_business_key"]),
        ),
        "membership_product": membership_product,
        "assignment_protocol_version": module.ASSIGNMENT_PROTOCOL_VERSION,
    }
    contribution_hash = module.stable_hash([canonical_contribution])
    raw_distance = relation["distance"] / (1.0 + math.log1p(membership_product))
    raw_strength = math.exp(-raw_distance)
    stats = _projection_stats(module, layer, "co_occurs_with", raw_strength)
    edge_id = f"{layer}-edge-1"
    support_mid_ids = ["mid-edge-1"] if layer == "coarse" else []
    diagnostics = {
        "edge_projection_protocol": module.EDGE_PROJECTION_PROTOCOL_VERSION,
        "edge_projection_protocol_hash": module.edge_projection_protocol_hash(),
        "aggregation_protocol_version": module.PROJECTION_AGGREGATION_PROTOCOL_VERSION,
        "calibration_protocol_version": module.PROJECTION_CALIBRATION_PROTOCOL_VERSION,
        "source_algorithm": "membership_weighted_bottom_edge_projection",
        "support_chunk_edge_count": 1,
        "support_contribution_count": 1,
        "support_membership_mass": membership_product,
        "support_contributions": [contribution],
        "contribution_facts_hash": contribution_hash,
        "dominant_bottom_edge_type": relation["edge_type"],
        "support_edge_types": {relation["edge_type"]: 1},
        "gray_zone_semantics_changed": False,
        "model_call_count": 0,
    }
    if layer == "coarse":
        diagnostics["support_mid_edge_count"] = len(support_mid_ids)
    summary = {
        "aggregation_protocol_version": module.PROJECTION_AGGREGATION_PROTOCOL_VERSION,
        "q15_bottom_distance": relation["distance"],
        "support_membership_mass": membership_product,
        "support_chunk_edge_count": 1,
        "bottom_distance_distribution": module._distribution([relation["distance"]]),
        "membership_product_distribution": module._distribution([membership_product]),
        "dominant_bottom_edge_type": relation["edge_type"],
        "bottom_edge_type_membership_mass": {relation["edge_type"]: membership_product},
        "contribution_facts_hash": contribution_hash,
        "edge_distance_protocol": module.EDGE_DISTANCE_PROTOCOL_VERSION,
    }
    if layer == "coarse":
        summary["support_mid_edge_count"] = len(support_mid_ids)
    normalization = {
        **stats,
        "support_edge_count": 1,
        "support_chunk_edge_count": 1,
        "support_membership_mass": membership_product,
    }
    if layer == "coarse":
        normalization["support_mid_edge_count"] = len(support_mid_ids)
    edge = {
        "id": edge_id,
        "source_concept_id": f"{layer}-source",
        "target_concept_id": f"{layer}-target",
        "source_scope_key": source_scope_key,
        "target_scope_key": target_scope_key,
        "source_rq_prefix_id": source_rq_prefix_id,
        "target_rq_prefix_id": target_rq_prefix_id,
        "edge_type": "co_occurs_with",
        "weight": raw_strength,
        "distance": raw_distance,
        "projected_strength_raw": raw_strength,
        "projected_distance_raw": raw_distance,
        "raw_strength_summary": summary,
        "support_chunk_edge_ids": [relation["id"]],
        "support_chunk_ids": sorted(
            [relation["source_chunk_id"], relation["target_chunk_id"]],
            key=lambda chunk_id: (chunk_keys[chunk_id], chunk_id),
        ),
        "support_mid_edge_ids": support_mid_ids,
        "edge_projection_protocol_hash": module.edge_projection_protocol_hash(),
        "state_hash": "7" * 64,
        "source_algorithm": "membership_weighted_bottom_edge_projection",
        "protocol_version": module.EDGE_PROJECTION_PROTOCOL_VERSION,
        "projection_normalization_stats": normalization,
        "diagnostics": diagnostics,
    }
    aggregate = {
        "protocol_version": module.PROJECTION_CALIBRATION_PROTOCOL_VERSION,
        "edge_projection_protocol_version": module.EDGE_PROJECTION_PROTOCOL_VERSION,
        "edge_projection_protocol_hash": module.edge_projection_protocol_hash(),
        "layer": layer,
        "stats_by_edge_type": {"co_occurs_with": stats},
        "cross_type_raw_comparison_allowed": False,
        "model_call_count": 0,
    }
    return edge, aggregate


def _graph_fixture(module) -> dict:
    relation_edges = []
    relation_stats = {}
    chunk_business_keys = {
        f"chunk-{index}": module.stable_hash({"chunk_business_fact": index})
        for index in range(4)
    }
    params = dict(module.EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS)
    params_hash = module._relation_calibration_params_hash(params)
    config_hash = module._relation_calibration_config_hash(params)
    protocol_hash = module.edge_type_calibration_protocol_hash()
    distance_hash = module.edge_distance_protocol_hash()
    for index, edge_type in enumerate(module.EDGE_TYPE_CALIBRATION_EDGE_TYPES):
        raw = round(0.55 + index * 0.1, 6)
        distance = module._distance_from_strength(raw)
        stats = _relation_stats(module, edge_type, raw, raw)
        relation_stats[edge_type] = stats
        relation_edges.append(
            {
                "id": f"relation-{index}",
                "source_chunk_id": f"chunk-{index}",
                "target_chunk_id": f"chunk-{index + 1}",
                "edge_type": edge_type,
                "raw_strength": raw,
                "weight": raw,
                "distance": distance,
                "support": {
                    "protocol_version": module.RELATION_SUPPORT_PROTOCOL_VERSION,
                    "kind": "dense_vector_pair",
                    "source": edge_type,
                    "chunk_ids": [
                        f"chunk-{index}",
                        f"chunk-{index + 1}",
                    ],
                    "model_call_count": 0,
                },
                "source_algorithm": "dense_embedding",
                "protocol_version": module.RELATION_PROTOCOL_VERSION,
                "graph_state_hash": "6" * 64,
                "edge_distance_protocol_hash": distance_hash,
                "raw_strength_summary": {
                    "max_raw_strength": raw,
                    "calibrated_strength": raw,
                    "distance": distance,
                    "edge_distance_protocol": module.EDGE_DISTANCE_PROTOCOL_VERSION,
                    "edge_type_calibration_protocol": module.EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
                    "calibration_params_hash": params_hash,
                    "edge_type_calibration_config_hash": config_hash,
                    "edge_type_calibration_stats_hash": stats["stats_hash"],
                },
                "normalization_stats": {"edge_type_calibration": copy.deepcopy(stats)},
                "features": {
                    "calibrated_strength": raw,
                    "edge_type_calibration_protocol_version": module.EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
                    "edge_type_calibration_protocol_hash": protocol_hash,
                    "calibration_params_hash": params_hash,
                    "edge_type_calibration_config_hash": config_hash,
                    "edge_type_calibration_stats_hash": stats["stats_hash"],
                    "edge_distance_protocol_version": module.EDGE_DISTANCE_PROTOCOL_VERSION,
                    "edge_distance_protocol_hash": distance_hash,
                    "normalization_stats": {"edge_type_calibration": copy.deepcopy(stats)},
                },
            }
        )
    def membership(
        membership_id: str,
        prefix_id: str,
        chunk_id: str,
        score: float,
    ) -> dict:
        role_evaluation = {
            "role": "primary_member",
            "matched_flags": ["primary_member"],
            "primary_reason": "primary_member",
            "protocol_version": module.RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION,
            "protocol_hash": module.rq_membership_role_protocol_hash(),
            "thresholds": dict(module.RQ_MEMBERSHIP_ROLE_THRESHOLDS),
            "inputs": {
                "membership_score": score,
                "membership_entropy": 0.1,
                "residual_norm": 0.1,
                "gamma": 0.9,
                "boundary_probability_margin": 0.9,
                "boundary_distance": 0.9,
                "residual_outlier_threshold": 1.0,
                "rank": 1,
                "is_primary_leaf": True,
                "is_bridge_chunk": False,
            },
            "model_call_count": 0,
        }
        return {
            "id": membership_id,
            "rq_prefix_id": prefix_id,
            "chunk_id": chunk_id,
            "membership_score": score,
            "membership_role": "primary_member",
            "membership_entropy": 0.1,
            "residual_norm": 0.1,
            "rank": 1,
            "role_evaluation": role_evaluation,
        }

    memberships = [
        membership("membership-1", "rq-1", "chunk-1", 0.9),
        membership("membership-source", "rq-source", "chunk-0", 0.8),
        membership("membership-target", "rq-target", "chunk-1", 0.75),
    ]
    projected_edges = {}
    projection_calibration = {}
    for layer in ("mid", "coarse"):
        edge, calibration = _projected_edge_fixture(
            module, layer, relation_edges[0], chunk_business_keys
        )
        projected_edges[layer] = [edge]
        projection_calibration[layer] = calibration
    return {
        "declared_counts": {
            "chunk_relation_edges": len(relation_edges),
            "rq_prefix_memberships": len(memberships),
            "mid_concept_edges": 1,
            "coarse_concept_edges": 1,
        },
        "chunk_business_keys": chunk_business_keys,
        "rq_prefix_scope_keys": {
            "rq-1": "rq:other",
            "rq-source": "rq:source",
            "rq-target": "rq:target",
        },
        "relation_edges": relation_edges,
        "relation_calibration": {
            "protocol_version": module.EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "calibration_params_hash": params_hash,
            "edge_type_calibration_config_hash": config_hash,
            "edge_distance_protocol_version": module.EDGE_DISTANCE_PROTOCOL_VERSION,
            "edge_distance_protocol_hash": distance_hash,
            "params": params,
            "candidate_scope": "accepted_typed_gate_reverse_quota_undirected_dedupe",
            "cross_type_raw_comparison_allowed": False,
            "stats_by_edge_type": relation_stats,
        },
        "rq_memberships": memberships,
        "rq_membership_diagnostics": {
            "membership_count": len(memberships),
            "membership_role_counts": {
                "primary_member": len(memberships)
            },
            "membership_score_distribution": module._distribution(
                [0.9, 0.8, 0.75]
            ),
            "membership_entropy_distribution": module._distribution(
                [0.1, 0.1, 0.1]
            ),
            "residual_norm_distribution": module._distribution(
                [0.1, 0.1]
            ),
            "model_call_count": 0,
            "renormalized_after_sparsification": False,
            "artificial_membership_floor": False,
            "rq_pair_edges_active": False,
        },
        "projected_edges": projected_edges,
        "projection_calibration": projection_calibration,
    }


def _set_relation_strength_consistently(
    module,
    fixture: dict,
    *,
    index: int,
    strength: float,
) -> None:
    edge = fixture["relation_edges"][index]
    distance = module._distance_from_strength(strength)
    edge.update(raw_strength=strength, weight=strength, distance=distance)
    stats = _relation_stats(module, edge["edge_type"], strength, strength)
    params_hash = fixture["relation_calibration"]["calibration_params_hash"]
    config_hash = fixture["relation_calibration"]["edge_type_calibration_config_hash"]
    edge["raw_strength_summary"].update(
        max_raw_strength=strength,
        calibrated_strength=strength,
        distance=distance,
        edge_type_calibration_stats_hash=stats["stats_hash"],
    )
    edge["normalization_stats"] = {"edge_type_calibration": copy.deepcopy(stats)}
    edge["features"].update(
        calibrated_strength=strength,
        calibration_params_hash=params_hash,
        edge_type_calibration_config_hash=config_hash,
        edge_type_calibration_stats_hash=stats["stats_hash"],
        normalization_stats={"edge_type_calibration": copy.deepcopy(stats)},
    )
    fixture["relation_calibration"]["stats_by_edge_type"][edge["edge_type"]] = stats
    for layer in ("mid", "coarse"):
        if edge["id"] in fixture["projected_edges"][layer][0]["support_chunk_edge_ids"]:
            projected, calibration = _projected_edge_fixture(
                module, layer, edge, fixture["chunk_business_keys"]
            )
            fixture["projected_edges"][layer] = [projected]
            fixture["projection_calibration"][layer] = calibration


def _set_relation_claimed_distance_consistently(
    fixture: dict,
    *,
    index: int,
    distance: float,
) -> None:
    edge = fixture["relation_edges"][index]
    edge["distance"] = distance
    edge["raw_strength_summary"]["distance"] = distance


def _forge_projected_membership_contribution_consistently(
    module,
    fixture: dict,
    *,
    layer: str,
    source_score: float,
    target_score: float,
) -> None:
    edge = fixture["projected_edges"][layer][0]
    contribution = edge["diagnostics"]["support_contributions"][0]
    relation = next(
        item
        for item in fixture["relation_edges"]
        if item["id"] == contribution["bottom_chunk_edge_id"]
    )
    contribution["source_membership_score"] = source_score
    contribution["target_membership_score"] = target_score
    contribution["membership_product"] = source_score * target_score
    product = contribution["membership_product"]
    canonical = {
        "bottom_edge_fact_hash": contribution["bottom_edge_fact_hash"],
        "endpoint_scope_memberships": sorted(
            [
                {
                    "scope_key": edge["source_scope_key"],
                    "chunk_business_key": fixture[
                        "chunk_business_keys"
                    ][contribution["source_chunk_id"]],
                    "membership_score": source_score,
                },
                {
                    "scope_key": edge["target_scope_key"],
                    "chunk_business_key": fixture[
                        "chunk_business_keys"
                    ][contribution["target_chunk_id"]],
                    "membership_score": target_score,
                },
            ],
            key=lambda item: (
                item["scope_key"],
                item["chunk_business_key"],
            ),
        ),
        "membership_product": product,
        "assignment_protocol_version": module.ASSIGNMENT_PROTOCOL_VERSION,
    }
    contribution_hash = module.stable_hash([canonical])
    raw_distance = relation["distance"] / (1.0 + math.log1p(product))
    raw_strength = math.exp(-raw_distance)
    stats = _projection_stats(
        module, layer, edge["edge_type"], raw_strength
    )
    edge.update(
        weight=raw_strength,
        distance=raw_distance,
        projected_strength_raw=raw_strength,
        projected_distance_raw=raw_distance,
    )
    edge["diagnostics"].update(
        support_membership_mass=product,
        contribution_facts_hash=contribution_hash,
    )
    edge["raw_strength_summary"].update(
        support_membership_mass=product,
        membership_product_distribution=module._distribution([product]),
        bottom_edge_type_membership_mass={
            relation["edge_type"]: product
        },
        contribution_facts_hash=contribution_hash,
    )
    normalization = {
        **copy.deepcopy(stats),
        "support_edge_count": 1,
        "support_chunk_edge_count": 1,
        "support_membership_mass": product,
    }
    if layer == "coarse":
        normalization["support_mid_edge_count"] = 1
    edge["projection_normalization_stats"] = normalization
    fixture["projection_calibration"][layer][
        "stats_by_edge_type"
    ][edge["edge_type"]] = stats


def _chunk_facet_priority_card(
    module,
    *,
    chunk_id: str,
    matched: bool,
) -> dict:
    required = ["facet-1"]
    matched_facets = ["facet-1"] if matched else []
    uncovered = [] if matched else ["facet-1"]
    card = {
        "protocol_version": (
            module.CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
        ),
        "protocol_hash": module.chunk_facet_priority_protocol_hash(),
        "facet_match_protocol_version": (
            module.QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION
        ),
        "facet_match_protocol_hash": (
            module.query_facet_ordered_window_protocol_hash()
        ),
        "chunk_id": chunk_id,
        "query_facet_packet_hash": module.stable_hash(
            {"required_facets": required}
        ),
        "required_facets": required,
        "matched_required_facets": matched_facets,
        "uncovered_required_facets": uncovered,
        "matched_required_facet_count": len(matched_facets),
        "uncovered_required_facet_count": len(uncovered),
        "priority_prefix": [len(uncovered)],
        "lexical_overlap_used_as_numeric_relevance": False,
        "is_evidence": False,
        "citation_authority": False,
        "gray_zone_decision_authority": False,
        "model_call_count": 0,
    }
    card["card_hash"] = module.stable_hash(card)
    return card


def _retrieval_fixture() -> dict:
    module = _load_quality_gate()
    from app.services.context_graph import (
        GRAY_ZONE_QUERY_FACET_PROTOCOL_VERSION,
        GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION,
        QueryFacetPosteriorCalibrator,
        agent_operating_envelope,
        deterministic_gray_query_facets_for_search,
        gray_zone_runtime_settings_hash,
        traversal_protocol_hash,
    )

    # Keep the fixture's existing path labels and dominance observations
    # aligned with the deterministic tokenizer; external query_facets below
    # are deliberately tested as non-authoritative for gray replay.
    raw_query = "facet-1"
    envelope = agent_operating_envelope()
    envelope.update(
        {
            "path_distance_green_threshold": 0.2,
            "path_distance_gray_threshold": 0.5,
            "path_distance_hard_threshold": 1.0,
            "max_cycle_reward_per_path": 0.25,
            "cycle_reward_distance_threshold": 0.8,
            "max_labels_per_node": 2,
            "agent_mid_top_k": 4,
            "agent_chunk_top_k": 4,
        }
    )
    envelope_hash = module.stable_hash(envelope)
    traversal_hash = traversal_protocol_hash(envelope)
    gray_runtime_hash = gray_zone_runtime_settings_hash(envelope)
    gray_query_hash = module.stable_hash(
        deterministic_gray_query_facets_for_search(raw_query)
    )
    query_facets = {
        "required_facets": ["facet-1"],
        "facet_groups": [
            {
                "facet": "facet-1",
                "aliases": [],
            }
        ],
    }
    posterior_calibrator = QueryFacetPosteriorCalibrator(
        query_facets=query_facets,
        enabled=True,
        observation_budget=4,
        round_budget=2,
        convergence_epsilon=0.02,
    )
    posterior_candidates = [
        {
            "candidate_id": "chunk-1",
            "text": "facet-1",
            "scope": "fixture",
            "candidate_business_input_hash": module.stable_hash(
                {"candidate_id": "chunk-1", "text": "facet-1"}
            ),
        },
        {
            "candidate_id": "chunk-2",
            "text": "unrelated",
            "scope": "fixture",
            "candidate_business_input_hash": module.stable_hash(
                {"candidate_id": "chunk-2", "text": "unrelated"}
            ),
        },
    ]
    posterior_calibrator.observe(
        "dense_entry_candidates",
        posterior_candidates,
    )
    posterior_snapshot = posterior_calibrator.snapshot()
    posterior_calibrator.observe(
        "merged_chunk_candidates",
        posterior_candidates,
    )
    posterior_card = posterior_calibrator.card()
    empty_layer = {
        "reason": "skipped_by_granularity=mid",
        "hard_stop_pruned_count": 0,
        "red_zone_pruned_count": 0,
        "dominance_pruned_count": 0,
        "label_budget_pruned_count": 0,
    }
    mid_layer = {
        "reason": "frontier_empty",
        "hard_stop_pruned_count": 0,
        "red_zone_pruned_count": 0,
        "dominance_pruned_count": 0,
        "label_budget_pruned_count": 0,
    }
    chunk_layer = {
        "reason": "frontier_empty",
        "hard_stop_pruned_count": 0,
        "red_zone_pruned_count": 0,
        "dominance_pruned_count": 0,
        "label_budget_pruned_count": 0,
    }
    pool = {
        "parent_layer": "mid",
        "parent_node_id": "mid-1",
        "candidate_ids": ["chunk-1", "chunk-2"],
        "candidate_scores": {"chunk-1": 1.0, "chunk-2": 0.01},
        "chunk_facet_priority_cards": {
            "chunk-1": _chunk_facet_priority_card(
                module, chunk_id="chunk-1", matched=True
            ),
            "chunk-2": _chunk_facet_priority_card(
                module, chunk_id="chunk-2", matched=False
            ),
        },
        "facet_priority_protocol_version": (
            module.CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
        ),
        "facet_priority_protocol_hash": (
            module.chunk_facet_priority_protocol_hash()
        ),
        "query_facet_posterior_snapshot": copy.deepcopy(
            posterior_snapshot
        ),
        "covered_posterior_mass_by_candidate": {
            "chunk-1": 1.0,
            "chunk-2": 0.0,
        },
        "forced_candidate_ids": [],
        "selected_ids": ["chunk-1"],
        "per_parent_budget_status": {
            "budget": 1,
            "candidate_count": 2,
            "selected_count": 1,
            "stop_reason": "per_parent_budget_hit",
        },
    }
    path_label = {
        "layer": "mid",
        "node_id": "mid-1",
        "root_node_id": "mid-1",
        "path": ["mid-1"],
        "path_edge_ids": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "covered_facets": ["facet-1"],
        "evidence_roles": ["entry"],
        "path_edge_type_multiset": {},
        "distance_so_far": 0.1,
        "reward_so_far": 0.0,
        "cycle_distance_rewards": [],
        "expanded_edge_ids": [],
    }
    mid_frontier_state = {
        "layer": "mid",
        "node_id": "mid-1",
        "root_node_id": "mid-1",
        "path": ["mid-1"],
        "path_edge_ids": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "path_edge_types": [],
        "distance_so_far": 0.1,
        "reward_so_far": 0.0,
        "covered_facets": ["facet-1"],
        "evidence_roles": ["entry"],
        "depth": 0,
        "entry_parent_refs": [],
    }
    mid_dominated_state = {
        "layer": "mid",
        "node_id": "mid-1",
        "root_node_id": "mid-1",
        "path": ["mid-1"],
        "path_edge_ids": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "path_edge_types": [],
        "distance_so_far": 0.2,
        "reward_so_far": 0.0,
        "covered_facets": [],
        "evidence_roles": [],
        "depth": 0,
        "entry_parent_refs": [],
    }
    chunk_frontier_states = [
        {
            "layer": "chunk",
            "node_id": "chunk-1",
            "root_node_id": "chunk-1",
            "path": ["chunk-1"],
            "path_edge_ids": [],
            "path_edge_distances": [],
            "path_edge_strengths": [],
            "path_edge_types": [],
            "distance_so_far": 0.1,
            "reward_so_far": 0.0,
            "covered_facets": ["facet-1"],
            "evidence_roles": ["entry"],
            "depth": 0,
            "entry_parent_refs": [],
        },
        {
            "layer": "chunk",
            "node_id": "chunk-2",
            "root_node_id": "chunk-1",
            "path": ["chunk-1", "chunk-2"],
            "path_edge_ids": ["edge-chunk-1-2"],
            "path_edge_distances": [0.8],
            "path_edge_strengths": [0.449329],
            "path_edge_types": ["semantic_affinity"],
            "distance_so_far": 0.9,
            "reward_so_far": 0.0,
            "covered_facets": [],
            "evidence_roles": [],
            "depth": 1,
            "entry_parent_refs": [],
        },
    ]
    dominance_observations = [
        {
            "protocol_version": module.DOMINANCE_REPLAY_PROTOCOL_VERSION,
            "node_id": "mid-1",
            "candidate_state": mid_frontier_state,
            "label": {
                "distance": 0.1,
                "covered_facets": ["facet-1"],
                "evidence_roles": ["entry"],
                "depth": 0,
                "path_edge_type_multiset": {},
            },
            "required_facets": ["facet-1"],
            "queue_key": [0, 0.1, 0, -1],
            "path_identity": module._frontier_state_path_identity(
                mid_frontier_state
            ),
            "frontier_serial": 0,
            "queue_size_before_pop": 1,
            "queue_size_after_pop": 0,
            "max_labels": 2,
            "admitted": True,
            "dominance_pruned_delta": 0,
            "label_budget_pruned_delta": 0,
        },
    ]
    mid_terminal_facts = {
        "entry_count": 1,
        "frontier_remaining_count": 0,
        "frontier_expansion_count": 0,
        "frontier_budget": 4,
    }
    chunk_terminal_facts = {
        **mid_terminal_facts,
        "frontier_expansion_count": 1,
    }
    mid_rank_fact = {
        "candidate_id": "mid-1",
        "rank_key": [0, 0.1, 0, -1],
        "path_identity": module._frontier_state_path_identity(
            mid_frontier_state
        ),
    }
    chunk_rank_facts = [
        {
            "candidate_id": "chunk-1",
            "rank_key": [0, 0.1, 0, -1],
            "path_identity": module._frontier_state_path_identity(
                chunk_frontier_states[0]
            ),
        },
        {
            "candidate_id": "chunk-2",
            "rank_key": [1, 0.9, 1, 0],
            "path_identity": module._frontier_state_path_identity(
                chunk_frontier_states[1]
            ),
        },
    ]
    return {
        "trace_id": "trace-quality-gate-1",
        "query": raw_query,
        "result_chunk_ids": ["chunk-1"],
        "runtime_settings_hash": "8" * 64,
        "traversal_protocol_hash": traversal_hash,
        "edge_distance_protocol_hash": "a" * 64,
        "edge_projection_protocol_hash": "b" * 64,
        "agent_operating_envelope_hash": envelope_hash,
        "gray_zone_model_call_count": 0,
        "trace_diagnostics": {
            "agent_operating_envelope": envelope,
            "agent_operating_envelope_hash": envelope_hash,
            "traversal_protocol_hash": traversal_hash,
            "runtime_settings_hash": "8" * 64,
            "gray_zone_runtime_settings_identity_protocol_version": (
                GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION
            ),
            "gray_zone_runtime_settings_hash": gray_runtime_hash,
            "gray_zone_query_facet_protocol_version": (
                GRAY_ZONE_QUERY_FACET_PROTOCOL_VERSION
            ),
            "gray_zone_query_facet_hash": gray_query_hash,
            "gray_zone_external_routing_packet_used": False,
            "gray_zone_request_scoped_budget_in_identity": False,
            "query_facet_posterior_calibration": copy.deepcopy(
                posterior_card
            ),
        },
        "query_facets": query_facets,
        "retrieval_granularity": "mid",
        "stage_queues": {
            "coarse": {"entry_ids": [], "forced_entry_ids": [], "accepted_ids": [], "selected_ids": [], "skipped_by_granularity": "mid"},
            "mid": {"entry_ids": ["mid-1"], "forced_entry_ids": [], "accepted_ids": ["mid-1"], "selected_ids": ["mid-1"], "top_k": 1},
            "rq_membership": {"entry_ids": ["rq-1"], "forced_entry_ids": [], "accepted_ids": ["rq-1"], "selected_ids": ["rq-1"]},
            "chunk": {"entry_ids": ["chunk-1"], "forced_entry_ids": [], "accepted_ids": ["chunk-1"], "selected_ids": ["chunk-1"], "top_k": 1},
        },
        "candidate_pools": {
            "mid_by_coarse": [],
            "chunk_by_mid": [pool],
            "mid_direct_entries": {
                "candidate_ids": ["mid-1"],
                "candidate_scores": {"mid-1": 0.8},
                "forced_candidate_ids": [],
                "selected_ids": ["mid-1"],
                "top_k": 1,
            },
            "rq_membership_entries": {
                "candidate_ids": ["rq-1"],
                "candidate_scores": {"rq-1": 0.7},
                "forced_candidate_ids": [],
                "selected_ids": ["rq-1"],
            },
            "chunk_initial_entries": {
                "candidate_ids": ["chunk-1", "chunk-2"],
                "candidate_scores": {
                    "chunk-1": 1.0,
                    "chunk-2": 0.01,
                },
                "chunk_facet_priority_cards": copy.deepcopy(
                    pool["chunk_facet_priority_cards"]
                ),
                "facet_priority_protocol_version": (
                    module.CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
                ),
                "facet_priority_protocol_hash": (
                    module.chunk_facet_priority_protocol_hash()
                ),
                "query_facet_posterior_snapshot": copy.deepcopy(
                    posterior_snapshot
                ),
                "covered_posterior_mass_by_candidate": {
                    "chunk-1": 1.0,
                    "chunk-2": 0.0,
                },
                "forced_candidate_ids": [],
                "selected_ids": ["chunk-1"],
                "top_k": 1,
            },
        },
        "topk_selection": {
            "coarse": {"top_k": 0, "candidate_count": 0, "selected_ids": []},
            "mid": {
                "top_k": 1,
                "candidate_count": 1,
                "selected_ids": ["mid-1"],
                "forced_selected_ids": [],
                "ranking_protocol_version": module.TOPK_RANKING_PROTOCOL_VERSION,
                "candidate_rank_facts": [mid_rank_fact],
            },
            "chunk": {
                "top_k": 1,
                "candidate_count": 2,
                "selected_ids": ["chunk-1"],
                "forced_selected_ids": [],
                "ranking_protocol_version": module.TOPK_RANKING_PROTOCOL_VERSION,
                "candidate_rank_facts": chunk_rank_facts,
            },
        },
        "convergence": {
            "reason": "frontier_empty",
            "agent_operating_envelope_hash": envelope_hash,
            "retrieval_granularity": "mid",
            "frontier_expansion_count": 1,
            "dominance_pruned_count": 0,
            "hard_stop_pruned_count": 0,
            "red_zone_pruned_count": 0,
            "path_distance_partition_event_count": 0,
            "cycle_distance_reward_bounded": True,
            "gray_zone_decision_count": 0,
            "gray_zone_rule_evaluation_count": 0,
            "gray_zone_model_call_count": 0,
            "gray_zone_rule_protocol_version": "deterministic_support_progress_v1",
            "query_facet_posterior_protocol_version": (
                module.QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
            ),
            "query_facet_posterior_protocol_hash": (
                module.query_facet_posterior_protocol_hash()
            ),
            "query_facet_posterior_rounds_used": posterior_card[
                "rounds_used"
            ],
            "query_facet_posterior_observations_used": posterior_card[
                "observations_used"
            ],
            "query_facet_posterior_stop_reason": posterior_card[
                "stop_reason"
            ],
            "query_facet_posterior_model_call_count": 0,
            "path_distance_thresholds": {"green": 0.2, "gray": 0.5, "hard": 1.0},
            "traversal_observation_budget": 4,
            "traversal_observation_budget_audit": {"local_rule_evaluation_count": 0, "expanded_observation_count": 0},
            "accepted_chunk_count": 1,
            "layers": {"coarse": empty_layer, "mid": mid_layer, "rq_membership": {}, "chunk": chunk_layer},
        },
        "steps": [
            {
                "step_index": 0,
                "layer": "mid",
                "action": "drill_down_each_coarse_or_direct_mid_entry",
                "dominance_pruned_count": 0,
                "cycle_distance_reward": 0.0,
                "gray_zone_path_decisions": [],
                "stop_reason": "frontier_empty",
                "popped_frontier_state": mid_frontier_state,
                "expanded_edge_ids": [],
                "diagnostics": {
                    "path_labels": [path_label],
                    "dominance_observations": dominance_observations,
                    "frontier_enqueues": [
                        {
                            "protocol_version": (
                                module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
                            ),
                            "enqueue_kind": "initial_entry",
                            "serial": 0,
                            "queue_key": [0.0, 0.1, 0.0, -1.0],
                            "candidate_state": copy.deepcopy(
                                mid_frontier_state
                            ),
                            "edge_id": None,
                            "parent_path_identity": None,
                        }
                    ],
                    "frontier_json": [
                        {
                            "layer": "mid",
                            "popped": mid_frontier_state,
                            "queue_size_before_pop": 1,
                            "queue_size_after_pop": 0,
                            "key": [0, 0.1, 0, -1],
                            "serial": 0,
                        }
                    ],
                    "terminal_facts": mid_terminal_facts,
                    "convergence_replay_protocol_version": module.CONVERGENCE_REPLAY_PROTOCOL_VERSION,
                },
            },
            {
                "step_index": 1,
                "layer": "chunk",
                "action": "select_seeds_from_mid_rq_membership",
                "dominance_pruned_count": 0,
                "cycle_distance_reward": 0.0,
                "gray_zone_path_decisions": [],
                "stop_reason": "rq_membership_seed_selection",
                "diagnostics": {"path_labels": []},
            },
            {
                "step_index": 2,
                "layer": "chunk",
                "action": "walk_graph_frontier",
                "dominance_pruned_count": 0,
                "cycle_distance_reward": 0.0,
                "gray_zone_path_decisions": [],
                "stop_reason": "frontier_empty",
                "popped_frontier_state": chunk_frontier_states[0],
                "expanded_edge_ids": ["edge-chunk-1-2"],
                "diagnostics": {
                    "path_labels": [
                        {
                            **path_label,
                            "layer": "chunk",
                            "node_id": "chunk-1",
                            "chunk_id": "chunk-1",
                            "root_node_id": "chunk-1",
                            "path": ["chunk-1"],
                            "expanded_edge_ids": [
                                "edge-chunk-1-2"
                            ],
                        },
                        {
                            **path_label,
                            "layer": "chunk",
                            "node_id": "chunk-2",
                            "chunk_id": "chunk-2",
                            "root_node_id": "chunk-1",
                            "path": ["chunk-1", "chunk-2"],
                            "path_edge_ids": ["edge-chunk-1-2"],
                            "path_edge_distances": [0.8],
                            "path_edge_strengths": [0.449329],
                            "covered_facets": [],
                            "evidence_roles": [],
                            "path_edge_type_multiset": {
                                "semantic_affinity": 1
                            },
                            "distance_so_far": 0.9,
                            "expanded_edge_ids": [],
                        },
                    ],
                    "dominance_observations": [
                        {
                            "protocol_version": module.DOMINANCE_REPLAY_PROTOCOL_VERSION,
                            "node_id": "chunk-1",
                            "candidate_state": chunk_frontier_states[
                                0
                            ],
                            "label": module._frontier_state_label_fact(
                                chunk_frontier_states[0]
                            ),
                            "required_facets": ["facet-1"],
                            "queue_key": [0, 0.1, 0, -1],
                            "path_identity": module._frontier_state_path_identity(
                                chunk_frontier_states[0]
                            ),
                            "frontier_serial": 0,
                            "queue_size_before_pop": 1,
                            "queue_size_after_pop": 0,
                            "max_labels": 2,
                            "admitted": True,
                            "dominance_pruned_delta": 0,
                            "label_budget_pruned_delta": 0,
                        },
                        {
                            "protocol_version": module.DOMINANCE_REPLAY_PROTOCOL_VERSION,
                            "node_id": "chunk-2",
                            "candidate_state": chunk_frontier_states[
                                1
                            ],
                            "label": module._frontier_state_label_fact(
                                chunk_frontier_states[1]
                            ),
                            "required_facets": ["facet-1"],
                            "queue_key": [1, 0.9, 1, 0],
                            "path_identity": module._frontier_state_path_identity(
                                chunk_frontier_states[1]
                            ),
                            "frontier_serial": 1,
                            "queue_size_before_pop": 1,
                            "queue_size_after_pop": 0,
                            "max_labels": 2,
                            "admitted": True,
                            "dominance_pruned_delta": 0,
                            "label_budget_pruned_delta": 0,
                        }
                    ],
                    "frontier_enqueues": [
                        {
                            "protocol_version": (
                                module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
                            ),
                            "enqueue_kind": "initial_entry",
                            "serial": 0,
                            "queue_key": [0.0, 0.1, 0.0, -1.0],
                            "candidate_state": copy.deepcopy(
                                chunk_frontier_states[0]
                            ),
                            "edge_id": None,
                            "parent_path_identity": None,
                        },
                        {
                            "protocol_version": (
                                module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
                            ),
                            "enqueue_kind": "expanded_edge",
                            "serial": 1,
                            "queue_key": [1.0, 0.9, 1.0, 0.0],
                            "candidate_state": copy.deepcopy(
                                chunk_frontier_states[1]
                            ),
                            "edge_id": "edge-chunk-1-2",
                            "parent_path_identity": (
                                module._frontier_state_path_identity(
                                    chunk_frontier_states[0]
                                )
                            ),
                        },
                    ],
                    "frontier_json": [
                        {
                            "popped": chunk_frontier_states[0],
                            "queue_size_before_pop": 1,
                            "queue_size_after_pop": 0,
                            "key": [0, 0.1, 0, -1],
                            "serial": 0,
                        },
                        {
                            "popped": chunk_frontier_states[1],
                            "queue_size_before_pop": 1,
                            "queue_size_after_pop": 0,
                            "key": [1, 0.9, 1, 0],
                            "serial": 1,
                        },
                    ],
                    "terminal_facts": chunk_terminal_facts,
                    "convergence_replay_protocol_version": module.CONVERGENCE_REPLAY_PROTOCOL_VERSION,
                },
            },
            {
                "step_index": 3,
                "layer": "structure",
                "action": "restore_context_package",
                "dominance_pruned_count": 0,
                "cycle_distance_reward": 0.0,
                "gray_zone_path_decisions": [],
                "stop_reason": "context_package_built",
                "diagnostics": {"path_labels": []},
            },
        ],
    }


def _seal_query_rq_seed_card(module, card: dict) -> dict:
    card = copy.deepcopy(card)
    card.pop("input_hash", None)
    card.pop("card_hash", None)
    card["input_hash"] = module.stable_hash(card)
    card["card_hash"] = module.stable_hash(card)
    return card


def _query_rq_stage_seed_card(module) -> dict:
    route_contribution = {
        "mid_concept_id": "mid-1",
        "mid_entry_strength": 0.45,
        "mid_membership_score": 0.5,
        "route_fallback_score": 0.225,
    }
    return _seal_query_rq_seed_card(
        module,
        {
            "protocol_version": module.QUERY_RQ_SEED_PROTOCOL_VERSION,
            "protocol_hash": module.query_rq_seed_protocol_hash(),
            "rq_prefix_id": "rq-1",
            "rq_path": [1, 2, 3],
            "rq_level": 3,
            "query_rq_path": [1, 2, 4],
            "rq_lcp_depth": 2,
            "residual_distance": 0.2,
            "query_prefix_membership_score": 0.6,
            "requested_query_relevance": 0.7,
            "route_fallback_score": 0.225,
            "parent_mid_contributions": [route_contribution],
            "score_source": "query_rq_relevance",
            "effective_score": 0.7,
            "forced_override": False,
            "relation_state_hash": "c" * 64,
            "is_evidence": False,
            "node_weight_used_as_query_relevance": False,
            "hard_path_lcp_used_as_score": False,
            "gray_zone_decision_authority": False,
            "model_call_count": 0,
        },
    )


def _query_rq_chunk_seed_card(
    module,
    *,
    chunk_id: str,
    score_source: str,
) -> dict:
    if score_source == "query_rq_fuzzy_membership":
        source_fields = {
            "rq_l3_prefix_id": "rq-1",
            "query_rq_path": [],
            "candidate_rq_path": [1, 2, 3],
            "rq_lcp_depth": 0,
            "residual_distance": None,
            "query_prefix_score": 1.0,
            "chunk_membership_score": 1.0,
            "fuzzy_membership_overlap_score": 1.0,
            "rq_score": 1.0,
            "rq_relevance_component": 1.0,
            "rq_drift_penalty": 0.0,
            "membership_role": "fuzzy_member",
            "membership_rank": 1,
            "membership_entropy": 0.1,
            "bridge_or_boundary_role": False,
            "support_edge_ids": ["edge-rq-1"],
            "mid_entry_component": 1.0,
            "dense_component": 1.0,
            "component_weights": dict(
                module.QUERY_RQ_CHUNK_SEED_COMPONENT_WEIGHTS
            ),
            "effective_score": 1.0,
            "membership_role_tie_break_rank": 1,
        }
    else:
        source_fields = {
            "rq_l3_prefix_id": None,
            "query_rq_path": [],
            "candidate_rq_path": [],
            "rq_lcp_depth": 0,
            "residual_distance": None,
            "query_prefix_score": 0.0,
            "chunk_membership_score": 0.0,
            "fuzzy_membership_overlap_score": 0.0,
            "rq_score": 0.0,
            "rq_relevance_component": 0.0,
            "rq_drift_penalty": None,
            "membership_role": "mid_support_fallback",
            "membership_rank": 0,
            "membership_entropy": None,
            "bridge_or_boundary_role": False,
            "support_edge_ids": [],
            "mid_entry_component": 0.0,
            "dense_component": 0.05,
            "component_weights": dict(
                module
                .QUERY_RQ_MID_SUPPORT_FALLBACK_COMPONENT_WEIGHTS
            ),
            "effective_score": 0.01,
            "membership_role_tie_break_rank": 6,
        }
    return _seal_query_rq_seed_card(
        module,
        {
            "protocol_version": module.QUERY_RQ_SEED_PROTOCOL_VERSION,
            "protocol_hash": module.query_rq_seed_protocol_hash(),
            "parent_mid_concept_id": "mid-1",
            "chunk_id": chunk_id,
            **source_fields,
            "score_source": score_source,
            "is_evidence": False,
            "node_weight_used_as_query_relevance": False,
            "hard_path_lcp_used_as_score": False,
            "gray_zone_decision_authority": False,
            "model_call_count": 0,
        },
    )


def _retrieval_fixture_with_query_rq_seed_cards(module) -> dict:
    fixture = _retrieval_fixture()
    protocol_fields = {
        "ranking_protocol_version": (
            module.QUERY_RQ_SEED_PROTOCOL_VERSION
        ),
        "ranking_protocol_hash": module.query_rq_seed_protocol_hash(),
    }
    fixture["candidate_pools"]["rq_membership_entries"].update(
        **protocol_fields,
        rq_seed_cards={
            "rq-1": _query_rq_stage_seed_card(module),
        },
        rq_chunk_seed_cards={},
    )
    chunk_cards = {
        "chunk-1": [
            _query_rq_chunk_seed_card(
                module,
                chunk_id="chunk-1",
                score_source="query_rq_fuzzy_membership",
            )
        ],
        "chunk-2": [
            _query_rq_chunk_seed_card(
                module,
                chunk_id="chunk-2",
                score_source=(
                    "mid_support_without_rq_membership"
                ),
            )
        ],
    }
    for pool in (
        fixture["candidate_pools"]["chunk_by_mid"][0],
        fixture["candidate_pools"]["chunk_initial_entries"],
    ):
        pool.update(
            **protocol_fields,
            rq_seed_cards={},
            rq_chunk_seed_cards=copy.deepcopy(chunk_cards),
        )
    return fixture


def _source_span(package_id: str = "package-1", trace_id: str = "trace-1") -> dict:
    module = _load_quality_gate()
    checksum = module.hashlib.sha256(b"source bytes").hexdigest()
    text = "Alpha beta"
    return {
        "contract_version": "raw_chunk_source_span_v1",
        "document_version_id": "version-1",
        "chunk_id": "chunk-1",
        "source_path": "source.pdf",
        "source_checksum": checksum,
        "logical_source_path": "source.pdf",
        "source_snapshot_verification": {
            "protocol_version": module.SOURCE_SNAPSHOT_PROTOCOL_VERSION,
            "final_open_protocol_version": "posix_openat_nofollow_fstat_v1",
            "storage_path": "source.pdf",
            "checksum": checksum,
            "verified": True,
            "size_bytes": len(b"source bytes"),
        },
        "chunk_text_hash_protocol_version": module.CHUNK_TEXT_HASH_PROTOCOL_VERSION,
        "chunk_text_hash": module._chunk_text_hash(text),
        "raw_span_text_hash_protocol_version": module.RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION,
        "raw_span_text_hash": module._raw_span_text_hash(text),
        "char_span": [0, 10],
        "raw_chunk_char_span": [0, 10],
        "page_range": [1, 1],
        "section_path": ["Section"],
        "structure_path": ["doc", "section"],
        "structure_node_ids": ["node-1"],
        "context_package_id": package_id,
        "retrieval_trace_id": trace_id,
    }


def _context_fixture() -> dict:
    module = _load_quality_gate()
    text = "Alpha beta"
    span = _source_span()
    source_fact = {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "document_version_id": "version-1",
        "stored_chunk_text": text,
        "stored_chunk_text_hash": module._chunk_text_hash(text),
        "stored_char_span": [0, len(text)],
        "document_version_checksum": span["source_checksum"],
        "document_version_storage_path": span["source_path"],
        "logical_source_path": span["logical_source_path"],
        "snapshot_protocol_version": module.SOURCE_SNAPSHOT_PROTOCOL_VERSION,
        "snapshot_observed_checksum": span["source_checksum"],
        "snapshot_size_bytes": len(b"source bytes"),
        "snapshot_verified": True,
    }
    source_fact["fact_hash"] = module._source_fact_hash(source_fact)
    return {
        "id": "package-1",
        "retrieval_trace_id": "trace-1",
        "package_hash": "f" * 64,
        "hit_chunk_ids": ["chunk-1"],
        "restored_chunk_ids": [],
        "bridge_chunk_ids": [],
        "parent_structure_node_ids": ["node-1"],
        "graph_path_ids": ["edge-1"],
        "trace_path_edge_ids": ["edge-1"],
        "dedupe_keys": ["dedupe-1"],
        "token_budget": 8,
        "token_count": 3,
        "source_facts": [source_fact],
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "document_id": "document-1",
                "document_version_id": "version-1",
                "context_package_id": "package-1",
                "content": text,
                "char_span": [0, len(text)],
                "raw_chunk_char_span": [0, len(text)],
                "chunk_text_hash": module._chunk_text_hash(text),
                "raw_span_text_hash": module._raw_span_text_hash(text),
                "role": "hit",
                "dedupe_key": "dedupe-1",
                "content_token_count": 3,
                "original_token_count": 3,
                "source_span": span,
                "structure_closure": {"parent_section": {"id": "node-1"}},
                "structure_node_ids": ["node-1"],
                "why_selected": {"reason": "retrieval_hit"},
            }
        ],
        "citation_spans": [copy.deepcopy(span)],
        "diagnostics": {
            "token_budget_audit": {"token_budget": 8, "token_count": 3, "within_budget": True, "packing_protocol": "bounded_pack_v1"},
            "restore_counts": {"hit_chunks": 1, "restored_chunks": 0, "bridge_chunks": 0, "graph_path_chunks": 0, "parent_structure_nodes": 1, "per_hit_chunk_budget": 2},
            "snapshot_integrity": {
                "protocol_version": module.SOURCE_SNAPSHOT_PROTOCOL_VERSION,
                "verified_document_version_count": 1,
                "fail_closed": True,
            },
            "conversation_state_is_evidence": False,
            "repair_gray_zone_decision_authority": False,
            "repair_gray_zone_model_call_count": 0,
        },
    }


def _agent_fixture() -> dict:
    module = _load_quality_gate()
    retrieval_trace_id = "trace-quality-gate-1"
    envelope = {
        "max_typed_actions_per_round": 8,
        "agent_coarse_initial_budget": 2,
        "agent_coarse_drilldown_mid_initial_budget": 2,
        "agent_mid_initial_budget": 2,
        "agent_chunk_initial_budget": 2,
        "agent_chunk_per_mid_budget": 2,
        "max_depth_per_layer": 2,
        "max_labels_per_node": 2,
        "max_edge_reuse": 1,
        "agent_chunk_top_k": 4,
        "structure_restore_per_chunk_budget": 2,
        "context_package_token_budget": 512,
        "verification_budget": 2,
        "allowed_relation_types": ["dense_semantic"],
        "required_restore_modes": ["previous", "next"],
    }
    validation = {
        "valid": True,
        "schema_checked": True,
        "budget_checked": True,
        "target_ids_checked": True,
        "target_scope_checked": True,
        "target_layers": {},
        "fallback_disabled_checked": True,
        "bridge_protection_checked": True,
        "required_restore_modes": ["previous", "next"],
        "required_verification_stage": None,
    }
    actions = []
    for action_type in module.REQUIRED_TYPED_ACTIONS:
        expected_evidence = {
            "source": "context_graph",
            "requires_chunk_spans": True,
            "allowed_relation_types": ["dense_semantic"],
        }
        if action_type in {
            "restore_context_package",
            "build_context_package",
        }:
            expected_evidence["required_restore_modes"] = [
                "previous",
                "next",
            ]
        if action_type == "verify_citations":
            expected_evidence[
                "required_verification_stage"
            ] = "structure_plus_llm_entailment"
        actions.append(
            {
            "action_type": action_type,
            "target_ids": [],
            "reason": "Use the bounded four-layer executor.",
            "budget_request": {
                key: envelope[key]
                for key in module.TYPED_ACTION_BUDGET_KEYS[
                    action_type
                ]
            },
            "expected_evidence": expected_evidence,
            "stop_condition": {"required_action_complete": True},
            }
        )
    validation_audit = {
        "typed_action_schema_protocol_version": module.TYPED_ACTION_SCHEMA_PROTOCOL_VERSION,
        "typed_action_schema_protocol_hash": module.typed_action_schema_protocol_hash(),
        "accepted": [
            {
                "index": index,
                "accepted_index": index,
                "action_type": action["action_type"],
                "validation": {
                    **copy.deepcopy(validation),
                    "required_verification_stage": (
                        action["expected_evidence"].get(
                            "required_verification_stage"
                        )
                    ),
                },
            }
            for index, action in enumerate(actions)
        ],
        "rejected": [],
        "inserted_required_actions": [],
        "retrieval_granularity": "mid",
        "valid": True,
    }
    execution_controls = {
        "protocol_version": (
            module.TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION
        ),
        "retrieval_granularity": "mid",
        "requested_result_top_k": 4,
        "effective_result_top_k": 4,
        "verification_budget": 2,
        "repair_round_budget": 0,
        "context_package_token_budget": 512,
        "structure_restore_per_chunk_budget": 2,
        "entry_targets_by_layer": {
            "coarse": [],
            "mid": [],
            "rq_membership": [],
            "chunk": [],
        },
        "phase_target_ids_by_action": {},
        "budget_overrides": {},
        "traversal_envelope_overrides": {},
        "allowed_relation_types": ["dense_semantic"],
        "action_effects": [
            {
                "action_index": index,
                "action_type": action["action_type"],
                "target_ids": [],
                "budget_overrides": {},
                "expected_evidence": copy.deepcopy(
                    action["expected_evidence"]
                ),
                "stop_condition": copy.deepcopy(
                    action["stop_condition"]
                ),
            }
            for index, action in enumerate(actions)
        ],
        "unsupported_controls": [],
        "gray_zone_semantics_changed": False,
        "gray_zone_rule_inputs_modified": False,
        "path_distance_thresholds_modified": False,
        "gray_zone_model_call_count": 0,
    }
    execution_controls["control_hash"] = module.stable_hash(
        execution_controls
    )
    control_hash = execution_controls["control_hash"]
    verdict = {
        "protocol_version": "evidence_evaluator_v1",
        "verdict": "sufficient",
        "reason": "The bounded observation contains supported raw spans.",
        "target_ids": [],
        "expected_evidence": {},
    }
    verdict["decision_hash"] = module.stable_hash(verdict)
    conversation_state_scope_hash = "4" * 64
    nodes = [
        *module.AGENT_TRACE_FIXED_PREFIX,
        *module.AGENT_TRACE_PLANNING_STAGE_GROUP,
        *module.AGENT_TRACE_FIXED_SUFFIX_BEFORE_REPAIR,
        *module.AGENT_TRACE_FIXED_SUFFIX_AFTER_REPAIR,
    ]
    trace = []
    for node in nodes:
        scores = {}
        if node == "agent_planner":
            scores.update(
                {
                    "plan_id": "plan-1",
                    "plan_index": 0,
                    "replan": False,
                    "agent_operating_envelope_hash": "5" * 64,
                    "retrieval_granularity": "mid",
                }
            )
        elif node == "typed_action_validation":
            scores.update(
                {
                    "plan_id": "plan-1",
                    "plan_index": 0,
                    "validation": copy.deepcopy(validation_audit),
                }
            )
        elif node == "typed_action_executor":
            scores.update(
                {
                    "plan_id": "plan-1",
                    "plan_index": 0,
                    "typed_action_control_hash": control_hash,
                    "retrieval_trace_id": retrieval_trace_id,
                }
            )
        elif node == "evidence_evaluator":
            scores["plan_id"] = "plan-1"
            scores["plan_index"] = 0
            scores["verdict"] = verdict
            scores["planning_rounds_remaining"] = 1
        elif node == "evidence_gate":
            scores.update(
                {
                    "retrieval_trace_id": retrieval_trace_id,
                    "retrieval_granularity": "mid",
                    "agent_plan_id": "plan-1",
                    "agent_plan_index": 0,
                    "planning_rounds_used": 1,
                    "typed_action_control_hash": control_hash,
                    "evidence_evaluator": copy.deepcopy(verdict),
                    "context_package_evidence_gate_passed": True,
                    "answer_model_called": False,
                    "conversation_state_scope_hash": (
                        conversation_state_scope_hash
                    ),
                }
            )
        elif node == "structure_context_restoration":
            scores["context_package_id"] = "package-1"
        elif node == "context_package":
            scores.update(
                {
                    "context_package_id": "package-1",
                    "token_count": 3,
                }
            )
        elif node == "grounded_answer":
            scores["context_package_id"] = "package-1"
        trace.append(
            {
                "id": f"event-{len(trace)}",
                "run_id": "run-1",
                "sequence_index": len(trace),
                "node": node,
                "status": "completed",
                "input_summary": "",
                "output_summary": "",
                "document_ids": [],
                "scores": scores,
                "duration_ms": 0,
                "error": None,
            }
        )
    span = _source_span(trace_id=retrieval_trace_id)
    span["verification_id"] = "verification-1"
    response = {
        "run_id": "run-1",
        "answer": "Grounded answer.",
        "context_package_id": "package-1",
        "retrieval_trace_id": retrieval_trace_id,
        "trace": trace,
        "model_audit": {
            "context_package_id": "package-1",
            "retrieval_trace_id": retrieval_trace_id,
            "answer_session_id": "answer-session-1",
            "typed_action_control_hash": control_hash,
            "planning_rounds_used": 1,
            "planning_round_budget": 2,
            "repair_round_budget": 2,
            "repair_rounds_used": 0,
            "repair_actions": [],
            "grounding_outcome": "grounded_answer",
            "claim_grounded_gate": {"all_claims_supported": True},
            "citation_verification_pass_rate": 1.0,
            "returned_citation_count": 1,
            "insufficient_evidence": False,
        },
        "citations": [
            {
                "chunk_id": "chunk-1",
                "context_package_id": "package-1",
                "retrieval_trace_id": retrieval_trace_id,
                "citation_verification_id": "verification-1",
                "source_span": span,
                "verification": {
                    "verdict": "supported",
                    "failure_type": "none",
                    "provenance_status": "valid",
                    "structure_context_status": "valid",
                    "diagnostics": {
                        "citation_provenance_valid": True,
                        "citation_provenance_fail_closed": True,
                        "citation_provenance_llm_override_allowed": False,
                        "citation_provenance_reasons": [],
                        "citation_provenance_hash": "2" * 64,
                        "citation_provenance_session_hash": "3" * 64,
                        "citation_provenance_persistence_gate_passed": True,
                    },
                },
            }
        ],
    }
    response["test_typed_action_facts"] = {
        "run_id": "run-1",
        "plans": [
            {
                "plan_id": "plan-1",
                "run_id": "run-1",
                "plan_index": 0,
                "knowledge_base_id": "kb-1",
                "retrieval_trace_id": retrieval_trace_id,
                "envelope": copy.deepcopy(envelope),
                "typed_actions": copy.deepcopy(actions),
                "validation": copy.deepcopy(validation_audit),
                "execution_controls": copy.deepcopy(
                    execution_controls
                ),
                "retrieval_control_binding": {
                    "retrieval_trace_id": retrieval_trace_id,
                    "knowledge_base_id": "kb-1",
                    "agent_plan_id": "plan-1",
                    "agent_plan_index": 0,
                    "typed_action_control_hash": control_hash,
                    "typed_action_executor_protocol_version": (
                        module.TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION
                    ),
                    "typed_action_controls": copy.deepcopy(
                        execution_controls
                    ),
                },
                "actions": [
                    {
                        **copy.deepcopy(action),
                        "run_id": "run-1",
                        "action_index": index,
                        "status": "accepted",
                        "validation": {
                            **copy.deepcopy(
                                validation_audit["accepted"][
                                    index
                                ]["validation"]
                            ),
                            "typed_action_schema_protocol_version": module.TYPED_ACTION_SCHEMA_PROTOCOL_VERSION,
                            "typed_action_schema_protocol_hash": module.typed_action_schema_protocol_hash(),
                        },
                    }
                    for index, action in enumerate(actions)
                ],
            }
        ],
    }
    response["test_persisted_agent_facts"] = {
        "protocol_version": (
            module.PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION
        ),
        "run": {
            "id": "run-1",
            "knowledge_base_id": "kb-1",
            "session_id": "qa-session-1",
            "question": "Grounded question.",
            "status": "completed",
            "route": "layered_context_graph",
            "current_node": None,
            "retry_count": 0,
            "final_answer": "Grounded answer.",
            "error_message": None,
            "metadata": {
                "conversation_state_scope_hash": (
                    conversation_state_scope_hash
                )
            },
        },
        "trace_events": copy.deepcopy(trace),
        "bindings": {
            "initial_retrieval_trace": {
                "id": retrieval_trace_id,
                "knowledge_base_id": "kb-1",
            },
            "initial_context_package": {
                "id": "package-1",
                "knowledge_base_id": "kb-1",
                "retrieval_trace_id": retrieval_trace_id,
            },
            "retrieval_trace": {
                "id": retrieval_trace_id,
                "knowledge_base_id": "kb-1",
            },
            "context_package": {
                "id": "package-1",
                "knowledge_base_id": "kb-1",
                "retrieval_trace_id": retrieval_trace_id,
            },
            "answer_session": {
                "id": "answer-session-1",
                "knowledge_base_id": "kb-1",
                "retrieval_trace_id": retrieval_trace_id,
                "context_package_id": "package-1",
                "qa_session_id": "qa-session-1",
                "question": "Grounded question.",
                "answer": "Grounded answer.",
            },
            "reward_event": {
                "id": "reward-1",
                "knowledge_base_id": "kb-1",
                "agent_run_id": "run-1",
                "retrieval_trace_id": retrieval_trace_id,
                "context_package_id": "package-1",
                "answer_session_id": "answer-session-1",
            },
        },
    }
    response["test_persisted_agent_facts"]["snapshot_hash"] = (
        module.persisted_agent_snapshot_hash(
            response["test_persisted_agent_facts"]
        )
    )
    return response


def _audit_agent_fixture(
    module,
    fixture: dict,
    *,
    retrieval_fixture: dict | None = None,
    retrieval_quality: dict | None = None,
) -> dict:
    return module.audit_agent_quality(
        fixture,
        retrieval_snapshot=retrieval_fixture or _retrieval_fixture(),
        typed_action_facts=fixture.get("test_typed_action_facts"),
        retrieval_quality=retrieval_quality,
        persisted_agent_facts=fixture.get(
            "test_persisted_agent_facts"
        ),
    )


def test_versioned_quality_gate_good_fixtures_pass():
    module = _load_quality_gate()
    retrieval_fixture = _retrieval_fixture()
    complete_gray_audit = module._local_gray_zone_audit(retrieval_fixture)
    assert module.audit_graph_quality(_graph_fixture(module))["pass"] is True
    assert module.audit_retrieval_quality(retrieval_fixture)["pass"] is True
    assert module.audit_retrieval_quality(
        retrieval_fixture, gray_zone_audit=complete_gray_audit
    )["pass"] is True
    assert module.audit_context_package_quality(_context_fixture())["pass"] is True
    agent_fixture = _agent_fixture()
    assert _audit_agent_fixture(module, agent_fixture)["pass"] is True


def _context_package_checker():
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        sys.modules.pop("check_context_package_quality", None)
        return importlib.import_module("check_context_package_quality")
    finally:
        sys.path.remove(str(SCRIPTS_ROOT))


def _public_frontier_trace_fixture(retrieval_granularity: str) -> dict:
    def step(layer: str, action: str, *, popped: bool = True) -> dict:
        return {
            "layer": layer,
            "action": action,
            "popped_frontier_state": (
                {"node_id": f"{layer}-1", "path": [f"{layer}-1"]}
                if popped
                else {}
            ),
            "stop_reason": "frontier_empty",
        }

    steps = [
        step("coarse", "staged_priority_queue_walk"),
        step("mid", "drill_down_each_coarse_or_direct_mid_entry"),
        step("chunk", "walk_graph_frontier"),
    ]
    stage_queues = {"coarse": {}}
    trace_diagnostics = {}
    if retrieval_granularity == "mid":
        steps[0]["popped_frontier_state"] = {}
        steps[0]["stop_reason"] = "skipped_by_granularity=mid"
        stage_queues["coarse"] = {
            "skipped_by_granularity": "mid",
            "reason": "skipped_by_granularity=mid",
        }
        trace_diagnostics["coarse_skipped_reason"] = (
            "skipped_by_granularity=mid"
        )
    return {
        "retrieval_granularity": retrieval_granularity,
        "stage_queues": stage_queues,
        "trace_diagnostics": trace_diagnostics,
        "gray_zone_model_call_count": 0,
        "steps": steps,
    }


def test_context_package_checker_mid_granularity_accepts_skipped_coarse():
    checker = _context_package_checker()
    trace = _public_frontier_trace_fixture("mid")

    audit = checker.audit_retrieval_frontier_convergence(trace)

    assert audit["pass"] is True, audit
    assert audit["participating_layers"] == ["mid", "chunk"]
    assert trace["steps"][0]["popped_frontier_state"] == {}
    assert audit["checks"]["coarse_skip_matches_retrieval_granularity"] is True
    assert audit["checks"]["gray_zone_model_call_count_zero"] is True


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (
            lambda trace: trace["stage_queues"]["coarse"].pop(
                "skipped_by_granularity"
            ),
            "coarse_skip_matches_retrieval_granularity",
        ),
        (
            lambda trace: trace["steps"][1].update(
                {"popped_frontier_state": {}}
            ),
            "each_participating_layer_has_frontier_path_convergence",
        ),
        (
            lambda trace: trace.update({"gray_zone_model_call_count": 1}),
            "gray_zone_model_call_count_zero",
        ),
    ],
)
def test_context_package_checker_mid_granularity_fails_closed(
    mutate,
    failed_check,
):
    checker = _context_package_checker()
    trace = _public_frontier_trace_fixture("mid")
    mutate(trace)

    audit = checker.audit_retrieval_frontier_convergence(trace)

    assert audit["pass"] is False
    assert audit["checks"][failed_check] is False


def test_context_package_checker_coarse_granularity_requires_coarse_frontier():
    checker = _context_package_checker()
    trace = _public_frontier_trace_fixture("coarse")

    passing_audit = checker.audit_retrieval_frontier_convergence(trace)
    trace["steps"][0]["popped_frontier_state"] = {}
    failing_audit = checker.audit_retrieval_frontier_convergence(trace)

    assert passing_audit["pass"] is True, passing_audit
    assert passing_audit["participating_layers"] == [
        "coarse",
        "mid",
        "chunk",
    ]
    assert failing_audit["pass"] is False
    assert (
        failing_audit["checks"][
            "each_participating_layer_has_frontier_path_convergence"
        ]
        is False
    )


def test_gray_replay_uses_raw_query_not_external_facets_or_typed_controls():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    baseline = module._local_gray_zone_audit(fixture)
    assert baseline["pass"] is True, baseline
    baseline_quality = module.audit_retrieval_quality(fixture)
    assert baseline_quality["pass"] is True, baseline_quality

    external_routing_tamper = copy.deepcopy(fixture)
    external_routing_tamper["query_facets"] = {
        "required_facets": ["provider-controlled-facet"],
        "diagnostics": {"source": "llm_validated"},
    }
    external_routing_tamper["typed_action_controls"] = {
        "forced_entry_targets": ["provider-controlled-node"],
    }
    external_audit = module._local_gray_zone_audit(
        external_routing_tamper
    )
    assert external_audit["pass"] is True, external_audit
    assert module.stable_hash(external_audit) == module.stable_hash(baseline)
    external_quality = module.audit_retrieval_quality(
        external_routing_tamper
    )
    assert external_quality["pass"] is False, external_quality
    assert {
        finding["code"]
        for finding in external_quality["findings"]
    }.intersection(
        {
            "dominance_raw_event_replay_mismatch",
            (
                "frontier_candidate_path_label_"
                "edge_distance_closure_mismatch"
            ),
        }
    )

    raw_query_tamper = copy.deepcopy(fixture)
    raw_query_tamper["query"] = "different raw query"
    raw_query_audit = module._local_gray_zone_audit(raw_query_tamper)
    assert raw_query_audit["pass"] is False
    assert any(
        item["type"] == "gray_query_facet_hash_mismatch"
        for item in raw_query_audit["violations"]
    )


def test_v11_quality_gate_protocol_identity_is_frozen():
    from app.services import agent_graph

    module = _load_quality_gate()
    assert (
        module.QUALITY_GATE_PROTOCOL_VERSION
        == "four_layer_acceptance_quality_gate_v11"
    )
    assert module.QUALITY_GATE_PROTOCOL_HASH == (
        "299cf0db93c9852878cee195acde509d1"
        "9b3694f0c52d1aa3df84d890bc0044d"
    )
    assert module.chunk_facet_priority_protocol_hash() == (
        "d266e9eafef9016518b8d42b2924a598"
        "bc1d0abbf15e587736a99893b1fa5bc1"
    )
    assert module.query_facet_posterior_protocol_hash() == (
        "cbf34022b2e8983c41499cea7e89ac87"
        "30ff7b8a3f56d9354e493531e0e0ce08"
    )
    assert module.query_facet_ordered_window_protocol_hash() == (
        "74db72ba2426efe4242ce1599f5f3fd2"
        "ee1c85ac90c7602b8a64e9a02c768ac7"
    )
    assert module.FRONTIER_PATH_LABEL_CLOSURE_PROTOCOL_VERSION == (
        "frontier_min_heap_key_serial_temporal_closure_v3"
    )
    assert module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION == (
        "frontier_heap_enqueue_key_serial_v1"
    )
    assert module.EXPECTED_EVIDENCE_FIELDS == (
        agent_graph.EXPECTED_EVIDENCE_FIELDS
    )
    assert module.typed_action_schema_protocol_hash() == (
        agent_graph.stable_hash(
            {
                "protocol_version": (
                    agent_graph.TYPED_ACTION_SCHEMA_PROTOCOL_VERSION
                ),
                "allowed_actions": sorted(
                    agent_graph.ALLOWED_TYPED_ACTIONS
                ),
                "required_actions": agent_graph.REQUIRED_TYPED_ACTIONS,
                "required_fields": sorted(
                    agent_graph.TYPED_ACTION_REQUIRED_FIELDS
                ),
                "expected_evidence_fields": sorted(
                    agent_graph.EXPECTED_EVIDENCE_FIELDS
                ),
                "stop_condition_fields": sorted(
                    agent_graph.STOP_CONDITION_FIELDS
                ),
            }
        )
    )


def test_v10_executor_control_binding_is_per_plan_and_final_plan():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    facts = copy.deepcopy(fixture["test_typed_action_facts"])
    first = facts["plans"][0]
    second = copy.deepcopy(first)
    second.update(
        {
            "plan_id": "plan-2",
            "plan_index": 1,
            "retrieval_trace_id": "trace-quality-gate-2",
        }
    )
    second_controls = second["execution_controls"]
    second_controls["requested_result_top_k"] = 3
    second_controls["effective_result_top_k"] = 3
    second_controls.pop("control_hash")
    second_controls["control_hash"] = module.stable_hash(
        second_controls
    )
    second_binding = second["retrieval_control_binding"]
    second_binding.update(
        {
            "retrieval_trace_id": "trace-quality-gate-2",
            "agent_plan_id": "plan-2",
            "agent_plan_index": 1,
            "typed_action_control_hash": second_controls[
                "control_hash"
            ],
            "typed_action_controls": copy.deepcopy(
                second_controls
            ),
        }
    )
    facts["plans"].append(second)
    executor_events = [
        {
            "scores": {
                "plan_id": plan["plan_id"],
                "plan_index": plan["plan_index"],
                "retrieval_trace_id": plan["retrieval_trace_id"],
                "typed_action_control_hash": plan[
                    "execution_controls"
                ]["control_hash"],
            }
        }
        for plan in facts["plans"]
    ]
    response = {
        "retrieval_trace_id": "trace-quality-gate-2"
    }

    assert module._executor_plan_control_bindings_valid(
        response,
        typed_action_facts=facts,
        executor_events=executor_events,
        final_control_hash=second_controls["control_hash"],
    ) is True

    executor_events[0]["scores"][
        "typed_action_control_hash"
    ] = second_controls["control_hash"]
    assert module._executor_plan_control_bindings_valid(
        response,
        typed_action_facts=facts,
        executor_events=executor_events,
        final_control_hash=second_controls["control_hash"],
    ) is False


def test_v7_query_rq_seed_cards_replay_schema_hash_formula_score_and_order():
    module = _load_quality_gate()
    fixture = _retrieval_fixture_with_query_rq_seed_cards(module)

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is True, audit


@pytest.mark.parametrize(
    "tamper,expected_code",
    [
        (
            "delete_stage_card",
            "query_rq_stage_seed_card_coverage_mismatch",
        ),
        (
            "delete_chunk_card",
            "query_rq_chunk_seed_card_coverage_mismatch",
        ),
        (
            "resign_low_score_overwrite",
            "query_rq_stage_seed_card_formula_mismatch",
        ),
        (
            "resign_component_formula",
            "query_rq_chunk_seed_card_formula_mismatch",
        ),
        (
            "resign_node_weight_authority",
            "query_rq_seed_authority_violation",
        ),
        (
            "candidate_score",
            "query_rq_seed_candidate_score_mismatch",
        ),
        (
            "candidate_order",
            "query_rq_seed_candidate_order_mismatch",
        ),
        (
            "unsealed_card",
            "query_rq_chunk_seed_card_hash_mismatch",
        ),
    ],
)
def test_v7_query_rq_seed_cards_fail_closed_on_tamper(
    tamper,
    expected_code,
):
    module = _load_quality_gate()
    fixture = _retrieval_fixture_with_query_rq_seed_cards(module)
    rq_pool = fixture["candidate_pools"]["rq_membership_entries"]
    chunk_pool = fixture["candidate_pools"]["chunk_by_mid"][0]

    if tamper == "delete_stage_card":
        rq_pool["rq_seed_cards"].pop("rq-1")
    elif tamper == "delete_chunk_card":
        chunk_pool["rq_chunk_seed_cards"].pop("chunk-2")
    elif tamper == "resign_low_score_overwrite":
        card = rq_pool["rq_seed_cards"]["rq-1"]
        card["requested_query_relevance"] = 0.05
        card["effective_score"] = card["route_fallback_score"]
        rq_pool["candidate_scores"]["rq-1"] = card[
            "effective_score"
        ]
        rq_pool["rq_seed_cards"]["rq-1"] = (
            _seal_query_rq_seed_card(module, card)
        )
    elif tamper == "resign_component_formula":
        card = chunk_pool["rq_chunk_seed_cards"]["chunk-2"][0]
        card["component_weights"]["dense"] = 0.3
        chunk_pool["rq_chunk_seed_cards"]["chunk-2"][0] = (
            _seal_query_rq_seed_card(module, card)
        )
    elif tamper == "resign_node_weight_authority":
        card = chunk_pool["rq_chunk_seed_cards"]["chunk-1"][0]
        card["node_weight_used_as_query_relevance"] = True
        chunk_pool["rq_chunk_seed_cards"]["chunk-1"][0] = (
            _seal_query_rq_seed_card(module, card)
        )
    elif tamper == "candidate_score":
        chunk_pool["candidate_scores"]["chunk-1"] = 0.9
    elif tamper == "candidate_order":
        chunk_pool["candidate_ids"] = ["chunk-2", "chunk-1"]
    elif tamper == "unsealed_card":
        chunk_pool["rq_chunk_seed_cards"]["chunk-1"][0][
            "effective_score"
        ] = 0.9
    else:
        raise AssertionError(f"unknown tamper: {tamper}")

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert expected_code in {
        finding["code"] for finding in audit["findings"]
    }


@pytest.mark.parametrize(
    ("tamper", "expected_code"),
    [
        (
            "delete_card",
            "chunk_facet_priority_card_coverage_mismatch",
        ),
        (
            "count",
            "chunk_facet_priority_card_schema_mismatch",
        ),
        (
            "authority",
            "chunk_facet_priority_authority_violation",
        ),
        (
            "hash",
            "chunk_facet_priority_card_hash_mismatch",
        ),
        (
            "candidate_order",
            "candidate_pool_duplicate_id",
        ),
    ],
)
def test_v9_chunk_facet_priority_cards_fail_closed_on_tamper(
    tamper,
    expected_code,
):
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    pool = fixture["candidate_pools"]["chunk_by_mid"][0]
    card = pool["chunk_facet_priority_cards"]["chunk-1"]

    if tamper == "delete_card":
        pool["chunk_facet_priority_cards"].pop("chunk-2")
    elif tamper == "count":
        card["uncovered_required_facet_count"] = 1
        card["card_hash"] = module.stable_hash(
            {key: value for key, value in card.items() if key != "card_hash"}
        )
    elif tamper == "authority":
        card["citation_authority"] = True
        card["card_hash"] = module.stable_hash(
            {key: value for key, value in card.items() if key != "card_hash"}
        )
    elif tamper == "hash":
        card["card_hash"] = "f" * 64
    elif tamper == "candidate_order":
        pool["candidate_ids"] = ["chunk-2", "chunk-1"]
    else:
        raise AssertionError(f"unknown tamper: {tamper}")

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert expected_code in {
        finding["code"] for finding in audit["findings"]
    }


def test_graph_quality_gate_accepts_formula_valid_zero_distance_only_at_unit_strength():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    _set_relation_strength_consistently(module, fixture, index=0, strength=1.0)
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is True, audit["findings"]


def test_graph_quality_gate_replays_chunk_level_residual_norm_distribution():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    duplicate_chunk_membership = fixture["rq_memberships"][-1]
    duplicate_chunk_membership["residual_norm"] = 0.2
    duplicate_chunk_membership["role_evaluation"]["inputs"]["residual_norm"] = 0.2

    audit = module.audit_graph_quality(fixture)

    assert audit["pass"] is False
    assert "rq_residual_norm_chunk_identity_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_projected_bottom_type_mass_numeric_mapping_is_order_tolerant_but_closed():
    module = _load_quality_gate()

    assert module._strict_numeric_mapping_match(
        {"dense_semantic": 0.1 + 0.2},
        {"dense_semantic": 0.3},
    )
    assert not module._strict_numeric_mapping_match(
        {"dense_semantic": 0.3, "unexpected": 0.0},
        {"dense_semantic": 0.3},
    )
    assert not module._strict_numeric_mapping_match(
        {"dense_semantic": 0.3001},
        {"dense_semantic": 0.3},
    )


@pytest.mark.parametrize(
    "mutate,expected_code",
    [
        (lambda data: data["relation_edges"][0].update(distance=9.0), "relation_edge_distance_mismatch"),
        (
            lambda data: _set_relation_claimed_distance_consistently(
                data, index=0, distance=0.0
            ),
            "relation_edge_distance_mismatch",
        ),
        (
            lambda data: _set_relation_claimed_distance_consistently(
                data, index=0, distance=-0.1
            ),
            "relation_edge_numeric_domain_invalid",
        ),
        (
            lambda data: data["projected_edges"]["mid"][0].update(
                distance=0.0
            ),
            "projected_edge_distance_replay_mismatch",
        ),
        (
            lambda data: data["projected_edges"]["coarse"][0].update(
                distance=-0.1
            ),
            "projected_edge_distance_replay_mismatch",
        ),
        (lambda data: data["relation_calibration"]["stats_by_edge_type"][sorted(_load_quality_gate().ACTIVE_RELATION_EDGE_TYPES)[0]].update(stats_hash="0" * 64), "relation_calibration_stats_or_formula_mismatch"),
        (lambda data: data["rq_memberships"][0].update(membership_role="boundary_member"), "rq_membership_role_replay_mismatch"),
        (lambda data: data["projected_edges"]["mid"][0].update(support_chunk_edge_ids=[]), "projected_edge_support_or_protocol_invalid"),
    ],
)
def test_graph_quality_gate_rejects_tamper(mutate, expected_code):
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    mutate(fixture)
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False
    assert expected_code in {finding["code"] for finding in audit["findings"]}


def test_forged_projected_support_id_is_not_accepted():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    fixture["projected_edges"]["mid"][0]["support_chunk_edge_ids"] = [
        "relation-does-not-exist"
    ]
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "projected_edge_support_or_protocol_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def test_synchronized_relation_protocol_hash_tamper_is_not_accepted():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    forged = "f" * 64
    fixture["relation_calibration"]["edge_distance_protocol_hash"] = forged
    for edge in fixture["relation_edges"]:
        edge["edge_distance_protocol_hash"] = forged
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "relation_calibration_protocol_or_config_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_synchronized_relation_calibration_formula_forgery_is_not_accepted():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    edge = fixture["relation_edges"][0]
    edge_type = edge["edge_type"]
    edge["raw_strength"] = 0.2
    edge["weight"] = 0.9
    edge["distance"] = module._distance_from_strength(0.9)
    edge["raw_strength_summary"]["max_raw_strength"] = 0.2
    edge["raw_strength_summary"]["calibrated_strength"] = 0.9
    edge["raw_strength_summary"]["distance"] = edge["distance"]
    stats = fixture["relation_calibration"]["stats_by_edge_type"][edge_type]
    stats.update(
        sample_count=1,
        lower_quantile_value=0.2,
        upper_quantile_value=0.2,
        effective_lower_bound=0.2,
        effective_upper_bound=0.2,
        quantile_span=0.0,
        fallback="insufficient_type_sample",
        calibration_applied=False,
        raw_strength_distribution=module._distribution([0.2]),
        calibrated_strength_distribution=module._distribution([0.9]),
        distance_distribution=module._distribution([edge["distance"]]),
    )
    stats["stats_hash"] = module.stable_hash(
        {
            "protocol": stats["protocol_version"],
            "protocol_hash": stats["protocol_hash"],
            "edge_type": edge_type,
            "params": stats["params"],
            "raw_strengths": [0.2],
            "fallback": stats["fallback"],
        }
    )
    edge["raw_strength_summary"]["edge_type_calibration_stats_hash"] = stats[
        "stats_hash"
    ]
    edge["normalization_stats"] = {
        "edge_type_calibration": copy.deepcopy(stats)
    }
    edge["features"].update(
        calibrated_strength=0.9,
        edge_type_calibration_stats_hash=stats["stats_hash"],
        normalization_stats={"edge_type_calibration": copy.deepcopy(stats)},
    )
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "relation_calibrated_strength_formula_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_synchronized_projection_calibration_formula_forgery_is_not_accepted():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    edge = fixture["projected_edges"]["mid"][0]
    stats = fixture["projection_calibration"]["mid"]["stats_by_edge_type"][
        edge["edge_type"]
    ]
    edge.update(
        projected_strength_raw=0.2,
        projected_distance_raw=module._raw_projected_distance(0.2),
        weight=0.9,
        distance=module._raw_projected_distance(0.9),
    )
    stats.update(
        sample_count=1,
        lower_quantile_value=0.2,
        upper_quantile_value=0.2,
        quantile_span=0.0,
        fallback="insufficient_layer_type_sample",
        calibration_applied=False,
        raw_strength_distribution=module._distribution([0.2]),
        calibrated_strength_distribution=module._distribution([0.9]),
        calibrated_distance_distribution=module._distribution([edge["distance"]]),
    )
    stats["stats_hash"] = module.stable_hash(
        {key: value for key, value in stats.items() if key != "stats_hash"}
    )
    edge["projection_normalization_stats"] = {
        **copy.deepcopy(stats),
        "support_edge_count": 1,
        "support_chunk_edge_count": 1,
        "support_membership_mass": edge["diagnostics"][
            "support_membership_mass"
        ],
    }
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "projected_calibrated_strength_formula_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_forged_gray_audit_pass_boolean_is_not_accepted():
    module = _load_quality_gate()
    audit = module.audit_retrieval_quality(
        _retrieval_fixture(), gray_zone_audit={"pass": True}
    )
    assert audit["pass"] is False, audit
    assert "gray_zone_persisted_trace_audit_failed" in {
        finding["code"] for finding in audit["findings"]
    }


@pytest.mark.parametrize(
    "mutate,expected_code",
    [
        (lambda data: data["candidate_pools"]["chunk_by_mid"].clear(), "mid_parent_drilldown_coverage_mismatch"),
        (lambda data: data["steps"][0].update(cycle_distance_reward=9.0), "cycle_distance_reward_mismatch"),
        (lambda data: data["convergence"].update(gray_zone_model_call_count=1), "gray_observation_budget_mismatch"),
        (lambda data: data["topk_selection"]["chunk"].update(selected_ids=[]), "final_chunk_selection_mismatch"),
    ],
)
def test_retrieval_quality_gate_rejects_tamper(mutate, expected_code):
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    mutate(fixture)
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False
    assert expected_code in {finding["code"] for finding in audit["findings"]}


@pytest.mark.parametrize(
    "mutate,expected_code",
    [
        (lambda data: data.update(token_count=9), "context_package_token_budget_mismatch"),
        (lambda data: data["citation_spans"][0].update(source_checksum="0" * 64), "citation_span_raw_provenance_mismatch"),
        (lambda data: data.update(graph_path_ids=["forged-edge"]), "context_package_graph_path_scope_mismatch"),
        (lambda data: data.update(dedupe_keys=["forged-dedupe"]), "context_package_dedupe_mismatch"),
    ],
)
def test_context_package_quality_gate_rejects_tamper(mutate, expected_code):
    module = _load_quality_gate()
    fixture = _context_fixture()
    mutate(fixture)
    audit = module.audit_context_package_quality(fixture)
    assert audit["pass"] is False
    assert expected_code in {finding["code"] for finding in audit["findings"]}


@pytest.mark.parametrize(
    "mutate,expected_code",
    [
        (lambda data: next(event for event in data["trace"] if event["node"] == "typed_action_validation")["scores"]["validation"]["accepted"][0]["validation"].update(budget_checked=False), "typed_action_validation_incomplete"),
        (lambda data: next(event for event in data["trace"] if event["node"] == "evidence_evaluator")["scores"]["verdict"].update(expected_evidence={"gray_zone": "continue_path"}), "evidence_evaluator_decision_invalid"),
        (lambda data: data["model_audit"].update(repair_rounds_used=1), "repair_budget_or_action_mismatch"),
        (lambda data: data["citations"][0]["verification"]["diagnostics"].update(citation_provenance_valid=False), "citation_raw_provenance_invalid"),
    ],
)
def test_agent_quality_gate_rejects_tamper(mutate, expected_code):
    module = _load_quality_gate()
    fixture = _agent_fixture()
    mutate(fixture)
    audit = _audit_agent_fixture(module, fixture)
    assert audit["pass"] is False
    assert expected_code in {finding["code"] for finding in audit["findings"]}


def test_v3_projection_contribution_cannot_forge_persisted_membership_scores():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    _forge_projected_membership_contribution_consistently(
        module,
        fixture,
        layer="mid",
        source_score=0.99,
        target_score=0.76,
    )
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "projected_edge_contribution_support_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_relation_support_must_close_over_real_edge_endpoints():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    fixture["relation_edges"][0]["support"]["chunk_ids"] = [
        "chunk-does-not-exist-a",
        "chunk-does-not-exist-b",
    ]
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "relation_edge_type_or_support_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_rq_role_protocol_version_and_hash_are_local_allowlisted():
    module = _load_quality_gate()
    fixture = _graph_fixture(module)
    evaluation = fixture["rq_memberships"][0]["role_evaluation"]
    evaluation["protocol_version"] = "llm_membership_override_v1"
    evaluation["protocol_hash"] = "a" * 64
    audit = module.audit_graph_quality(fixture)
    assert audit["pass"] is False, audit
    assert "rq_membership_role_replay_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_context_hash_copies_cannot_override_postgres_and_source_facts():
    module = _load_quality_gate()
    fixture = _context_fixture()
    forged_checksum = "a" * 64
    forged_chunk_hash = "b" * 64
    forged_span_hash = "c" * 64
    for span in (
        fixture["chunks"][0]["source_span"],
        fixture["citation_spans"][0],
    ):
        span["source_checksum"] = forged_checksum
        span["source_snapshot_verification"]["checksum"] = forged_checksum
        span["chunk_text_hash"] = forged_chunk_hash
        span["raw_span_text_hash"] = forged_span_hash
    fixture["chunks"][0]["chunk_text_hash"] = forged_chunk_hash
    fixture["chunks"][0]["raw_span_text_hash"] = forged_span_hash
    audit = module.audit_context_package_quality(fixture)
    assert audit["pass"] is False, audit
    assert "context_chunk_source_span_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_topk_must_select_frozen_best_candidate_after_tie_break():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    fixture["result_chunk_ids"] = ["chunk-2"]
    fixture["stage_queues"]["chunk"].update(
        entry_ids=["chunk-2"],
        accepted_ids=["chunk-2"],
        selected_ids=["chunk-2"],
    )
    fixture["candidate_pools"]["chunk_by_mid"][0][
        "selected_ids"
    ] = ["chunk-2"]
    fixture["candidate_pools"]["chunk_initial_entries"][
        "selected_ids"
    ] = ["chunk-2"]
    fixture["topk_selection"]["chunk"]["selected_ids"] = ["chunk-2"]
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    codes = {finding["code"] for finding in audit["findings"]}
    assert "topk_frozen_score_order_mismatch" in codes
    assert "candidate_pool_score_order_mismatch" in codes


def test_v3_topk_rank_facts_must_match_raw_frontier_replay():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    fixture["result_chunk_ids"] = ["chunk-2"]
    fixture["stage_queues"]["chunk"].update(
        entry_ids=["chunk-2"],
        accepted_ids=["chunk-2"],
        selected_ids=["chunk-2"],
    )
    for pool in (
        fixture["candidate_pools"]["chunk_by_mid"][0],
        fixture["candidate_pools"]["chunk_initial_entries"],
    ):
        pool["candidate_scores"] = {
            "chunk-1": 0.01,
            "chunk-2": 1.0,
        }
        pool["selected_ids"] = ["chunk-2"]
    selection = fixture["topk_selection"]["chunk"]
    selection["selected_ids"] = ["chunk-2"]
    selection["candidate_rank_facts"][0]["rank_key"] = [
        1,
        1.0,
        1,
        0,
    ]
    selection["candidate_rank_facts"][1]["rank_key"] = [
        0,
        0.01,
        0,
        -1,
    ]
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    assert "topk_frozen_score_order_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v4_synchronized_raw_frontier_forgery_cannot_diverge_from_path_label():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    chunk_step = fixture["steps"][2]
    forged_state = {
        "layer": "chunk",
        "node_id": "chunk-2",
        "root_node_id": "chunk-2",
        "path": ["chunk-2"],
        "path_edge_ids": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "path_edge_types": [],
        "distance_so_far": 0.01,
        "reward_so_far": 0.0,
        "covered_facets": ["facet-1"],
        "evidence_roles": ["entry"],
        "depth": 0,
        "entry_parent_refs": [],
    }
    forged_label = module._frontier_state_label_fact(forged_state)
    forged_queue_key = [0, 0.01, 0, -1]
    forged_path_identity = module._frontier_state_path_identity(
        forged_state
    )
    observation = chunk_step["diagnostics"]["dominance_observations"][1]
    observation.update(
        candidate_state=forged_state,
        label=forged_label,
        queue_key=forged_queue_key,
        path_identity=forged_path_identity,
    )
    chunk_step["diagnostics"]["frontier_json"][1].update(
        popped=forged_state,
        key=forged_queue_key,
    )
    fixture["result_chunk_ids"] = ["chunk-2"]
    fixture["stage_queues"]["chunk"].update(
        entry_ids=["chunk-2"],
        accepted_ids=["chunk-2"],
        selected_ids=["chunk-2"],
    )
    for pool in (
        fixture["candidate_pools"]["chunk_by_mid"][0],
        fixture["candidate_pools"]["chunk_initial_entries"],
    ):
        pool["candidate_scores"] = {
            "chunk-1": 0.01,
            "chunk-2": 1.0,
        }
        pool["selected_ids"] = ["chunk-2"]
    fixture["steps"][1]["selected_topk_ids"] = ["chunk-2"]
    selection = fixture["topk_selection"]["chunk"]
    selection["selected_ids"] = ["chunk-2"]
    selection["candidate_rank_facts"][1].update(
        rank_key=forged_queue_key,
        path_identity=forged_path_identity,
    )

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert (
        "frontier_candidate_path_label_edge_distance_closure_mismatch"
        in {finding["code"] for finding in audit["findings"]}
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda step: step.update(expanded_edge_ids=[]),
        lambda step: step["diagnostics"]["path_labels"][1].update(
            path_edge_distances=[0.7]
        ),
        lambda step: step["diagnostics"]["path_labels"][1].update(
            distance_so_far=0.8
        ),
        lambda step: step.update(
            popped_frontier_state=copy.deepcopy(
                step["diagnostics"]["frontier_json"][1]["popped"]
            )
        ),
        lambda step: step["diagnostics"]["path_labels"][0].update(
            repair_evidence_retention_protocol_version="forged"
        ),
    ],
)
def test_v4_frontier_path_label_edge_distance_closure_fails_closed(
    mutate,
):
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    mutate(fixture["steps"][2])

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert (
        "frontier_candidate_path_label_edge_distance_closure_mismatch"
        in {finding["code"] for finding in audit["findings"]}
    )


def _repair_overlay_directive(module, *, chunk_id="chunk-2"):
    directive = {
        "protocol_version": "typed_repair_loop_v1",
        "source_context_package_id": "package-repair-1",
        "source_retrieval_trace_id": "trace-repair-source-1",
        "carry_forward_supported_chunk_ids": [chunk_id],
    }
    directive["validated_directive_hash"] = module.stable_hash(directive)
    return directive


def _repair_overlay_path_label(module, directive, *, chunk_id="chunk-2"):
    return {
        "layer": "chunk",
        "node_id": chunk_id,
        "chunk_id": chunk_id,
        "root_node_id": chunk_id,
        "path": [chunk_id],
        "path_edge_ids": [],
        "path_edge_types": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "cycle_distance_rewards": [],
        "expanded_edge_ids": [],
        "covered_facets": ["facet-1"],
        "evidence_roles": ["prior_supported_claim_carry_forward"],
        "support_refs": {"support_chunk_ids": [chunk_id]},
        "entry_parent_refs": [],
        "distance_so_far": 0.0,
        "reward_so_far": 0.0,
        "stop_reason": "repair_supported_evidence_carry_forward",
        "repair_evidence_retention_protocol_version": (
            module.REPAIR_EVIDENCE_RETENTION_PROTOCOL_VERSION
        ),
        "source_context_package_id": directive[
            "source_context_package_id"
        ],
        "source_retrieval_trace_id": directive[
            "source_retrieval_trace_id"
        ],
        "repair_directive_hash": directive["validated_directive_hash"],
    }


def _repair_overlay_rank_selection(module):
    directive = _repair_overlay_directive(module)
    overlay_card = {
        "candidate_id": "chunk-2",
        "rank_key": [-1.0],
        "repair_evidence_retention_protocol_version": (
            module.REPAIR_EVIDENCE_RETENTION_PROTOCOL_VERSION
        ),
        "source_context_package_id": directive[
            "source_context_package_id"
        ],
        "source_retrieval_trace_id": directive[
            "source_retrieval_trace_id"
        ],
        "repair_directive_hash": directive["validated_directive_hash"],
    }
    overlay_fact = {
        **overlay_card,
        "path_identity": module.stable_hash(overlay_card),
    }
    raw_fact = {
        "candidate_id": "chunk-1",
        "rank_key": [0.0, 0.1, 0.0, -1.0],
        "path_identity": "raw-path-1",
    }
    return (
        {
            "top_k": 2,
            "candidate_count": 2,
            "selected_ids": ["chunk-2", "chunk-1"],
            "forced_selected_ids": ["chunk-2"],
            "ranking_protocol_version": module.TOPK_RANKING_PROTOCOL_VERSION,
            "candidate_rank_facts": [overlay_fact, raw_fact],
            "carry_forward_supported_chunk_ids": ["chunk-2"],
            "global_top_k_increased": False,
        },
        {"chunk-1": raw_fact},
        directive,
    )


def test_v4_repair_carry_forward_path_overlay_preserves_frontier_closure():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    directive = _repair_overlay_directive(module)
    fixture["trace_diagnostics"]["repair_directive"] = directive
    fixture["steps"][2]["diagnostics"]["path_labels"].insert(
        0,
        _repair_overlay_path_label(module, directive),
    )

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is True, audit


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repair_evidence_retention_protocol_version", "forged"),
        ("source_context_package_id", "package-forged"),
        ("source_retrieval_trace_id", "trace-forged"),
        ("repair_directive_hash", "f" * 64),
    ],
)
def test_v4_repair_carry_forward_path_overlay_binding_fails_closed(
    field,
    value,
):
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    directive = _repair_overlay_directive(module)
    label = _repair_overlay_path_label(module, directive)
    label[field] = value
    fixture["trace_diagnostics"]["repair_directive"] = directive
    fixture["steps"][2]["diagnostics"]["path_labels"].insert(0, label)

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert (
        "frontier_candidate_path_label_edge_distance_closure_mismatch"
        in {finding["code"] for finding in audit["findings"]}
    )


def test_v4_repair_carry_forward_rank_overlay_replays_closed_card():
    module = _load_quality_gate()
    selection, raw_rank_by_id, directive = _repair_overlay_rank_selection(
        module
    )
    audit = module._Audit("repair_rank_overlay")

    module._audit_topk_ranking(
        audit,
        layer="chunk",
        selection=selection,
        raw_rank_by_id=raw_rank_by_id,
        repair_directive=directive,
    )

    assert audit.finish()["pass"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fact: fact.update(
            repair_evidence_retention_protocol_version="forged"
        ),
        lambda fact: fact.update(
            source_context_package_id="package-forged"
        ),
        lambda fact: fact.update(repair_directive_hash="f" * 64),
        lambda fact: fact.update(path_identity="f" * 64),
        lambda fact: fact.update(unexpected_overlay_field=True),
    ],
)
def test_v4_repair_carry_forward_rank_overlay_fails_closed(mutate):
    module = _load_quality_gate()
    selection, raw_rank_by_id, directive = _repair_overlay_rank_selection(
        module
    )
    mutate(selection["candidate_rank_facts"][0])
    audit = module._Audit("repair_rank_overlay")

    module._audit_topk_ranking(
        audit,
        layer="chunk",
        selection=selection,
        raw_rank_by_id=raw_rank_by_id,
        repair_directive=directive,
    )

    result = audit.finish()
    assert result["pass"] is False, result
    assert "topk_frozen_score_order_mismatch" in {
        finding["code"] for finding in result["findings"]
    }


def test_v5_frontier_prefix_expansion_must_precede_descendant_pop():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    chunk_step = fixture["steps"][2]
    diagnostics = chunk_step["diagnostics"]
    diagnostics["dominance_observations"] = list(
        reversed(diagnostics["dominance_observations"])
    )
    diagnostics["frontier_json"] = list(
        reversed(diagnostics["frontier_json"])
    )
    diagnostics["path_labels"] = list(
        reversed(diagnostics["path_labels"])
    )
    chunk_step["popped_frontier_state"] = copy.deepcopy(
        diagnostics["frontier_json"][0]["popped"]
    )

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert (
        "frontier_candidate_path_label_edge_distance_closure_mismatch"
        in {finding["code"] for finding in audit["findings"]}
    )


def _v6_independent_root_priority_inversion_fixture(module):
    fixture = _retrieval_fixture()
    step = fixture["steps"][2]
    diagnostics = step["diagnostics"]
    second_state = {
        "layer": "chunk",
        "node_id": "chunk-2",
        "root_node_id": "chunk-2",
        "path": ["chunk-2"],
        "path_edge_ids": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "path_edge_types": [],
        "distance_so_far": 0.01,
        "reward_so_far": 0.0,
        "covered_facets": ["facet-1"],
        "evidence_roles": ["entry"],
        "depth": 0,
        "entry_parent_refs": [],
    }
    second_key = [0.0, 0.01, 0.0, -1.0]
    second_observation = diagnostics[
        "dominance_observations"
    ][1]
    second_observation.update(
        candidate_state=copy.deepcopy(second_state),
        label=module._frontier_state_label_fact(
            second_state
        ),
        queue_key=second_key,
        path_identity=module._frontier_state_path_identity(
            second_state
        ),
        frontier_serial=1,
        queue_size_before_pop=1,
        queue_size_after_pop=0,
    )
    diagnostics["dominance_observations"][0].update(
        frontier_serial=0,
        queue_size_before_pop=2,
        queue_size_after_pop=1,
    )
    diagnostics["frontier_json"][0].update(
        queue_size_before_pop=2,
        queue_size_after_pop=1,
        serial=0,
    )
    diagnostics["frontier_json"][1].update(
        popped=copy.deepcopy(second_state),
        queue_size_before_pop=1,
        queue_size_after_pop=0,
        key=second_key,
        serial=1,
    )
    diagnostics["path_labels"][0][
        "expanded_edge_ids"
    ] = []
    diagnostics["path_labels"][1] = {
        **copy.deepcopy(diagnostics["path_labels"][0]),
        "node_id": "chunk-2",
        "chunk_id": "chunk-2",
        "root_node_id": "chunk-2",
        "path": ["chunk-2"],
        "distance_so_far": 0.01,
    }
    diagnostics["frontier_enqueues"] = [
        {
            "protocol_version": (
                module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
            ),
            "enqueue_kind": "initial_entry",
            "serial": index,
            "queue_key": key,
            "candidate_state": copy.deepcopy(state),
            "edge_id": None,
            "parent_path_identity": None,
        }
        for index, (key, state) in enumerate(
            (
                (
                    [0.0, 0.1, 0.0, -1.0],
                    diagnostics["dominance_observations"][0][
                        "candidate_state"
                    ],
                ),
                (second_key, second_state),
            )
        )
    ]
    diagnostics["terminal_facts"].update(
        entry_count=2,
        frontier_expansion_count=0,
        frontier_remaining_count=0,
        frontier_budget=8,
    )
    step["expanded_edge_ids"] = []
    fixture["stage_queues"]["chunk"].update(
        entry_ids=["chunk-1", "chunk-2"],
        accepted_ids=["chunk-1", "chunk-2"],
        selected_ids=["chunk-1", "chunk-2"],
        top_k=2,
    )
    for pool in (
        fixture["candidate_pools"]["chunk_by_mid"][0],
        fixture["candidate_pools"]["chunk_initial_entries"],
    ):
        pool.update(
            candidate_ids=["chunk-2", "chunk-1"],
            candidate_scores={
                "chunk-2": 1.0,
                "chunk-1": 0.9,
            },
            selected_ids=["chunk-2", "chunk-1"],
        )
        if "top_k" in pool:
            pool["top_k"] = 2
        if isinstance(
            pool.get("per_parent_budget_status"), dict
        ):
            pool["per_parent_budget_status"].update(
                budget=2,
                candidate_count=2,
                selected_count=2,
                stop_reason="parent_candidates_exhausted",
            )
    fixture["topk_selection"]["chunk"].update(
        selected_ids=["chunk-2"],
        candidate_rank_facts=[
            copy.deepcopy(
                fixture["topk_selection"]["chunk"][
                    "candidate_rank_facts"
                ][0]
            ),
            {
                "candidate_id": "chunk-2",
                "rank_key": second_key,
                "path_identity": (
                    module._frontier_state_path_identity(
                        second_state
                    )
                ),
            },
        ],
    )
    fixture["result_chunk_ids"] = ["chunk-2"]
    fixture["convergence"]["frontier_expansion_count"] = 0
    return fixture


def test_v6_available_minimum_root_must_pop_first():
    module = _load_quality_gate()
    fixture = _v6_independent_root_priority_inversion_fixture(
        module
    )

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert (
        "frontier_candidate_path_label_edge_distance_closure_mismatch"
        in {finding["code"] for finding in audit["findings"]}
    )


def test_v5_frontier_temporal_prefix_replay_enforces_every_hop():
    module = _load_quality_gate()
    states = [
        {
            "layer": "chunk",
            "node_id": "root",
            "root_node_id": "root",
            "path": ["root"],
            "path_edge_ids": [],
            "path_edge_distances": [],
            "path_edge_strengths": [],
            "path_edge_types": [],
            "distance_so_far": 0.1,
            "reward_so_far": 0.0,
            "covered_facets": ["facet-1"],
            "evidence_roles": ["entry"],
            "depth": 0,
            "entry_parent_refs": [],
        },
        {
            "layer": "chunk",
            "node_id": "middle",
            "root_node_id": "root",
            "path": ["root", "middle"],
            "path_edge_ids": ["edge-root-middle"],
            "path_edge_distances": [0.2],
            "path_edge_strengths": [0.8],
            "path_edge_types": ["semantic_affinity"],
            "distance_so_far": 0.3,
            "reward_so_far": 0.0,
            "covered_facets": [],
            "evidence_roles": [],
            "depth": 1,
            "entry_parent_refs": [],
        },
        {
            "layer": "chunk",
            "node_id": "leaf",
            "root_node_id": "root",
            "path": ["root", "middle", "leaf"],
            "path_edge_ids": [
                "edge-root-middle",
                "edge-middle-leaf",
            ],
            "path_edge_distances": [0.2, 0.3],
            "path_edge_strengths": [0.8, 0.7],
            "path_edge_types": [
                "semantic_affinity",
                "semantic_affinity",
            ],
            "distance_so_far": 0.6,
            "reward_so_far": 0.0,
            "covered_facets": [],
            "evidence_roles": [],
            "depth": 2,
            "entry_parent_refs": [],
        },
    ]
    expanded_by_index = [
        ["edge-root-middle"],
        ["edge-middle-leaf"],
        [],
    ]
    labels = [
        {
            "layer": "chunk",
            "node_id": state["node_id"],
            "chunk_id": state["node_id"],
            "root_node_id": state["root_node_id"],
            "path": list(state["path"]),
            "path_edge_ids": list(state["path_edge_ids"]),
            "path_edge_distances": list(
                state["path_edge_distances"]
            ),
            "path_edge_strengths": list(
                state["path_edge_strengths"]
            ),
            "distance_so_far": state["distance_so_far"],
            "reward_so_far": state["reward_so_far"],
            "covered_facets": list(state["covered_facets"]),
            "evidence_roles": list(state["evidence_roles"]),
            "expanded_edge_ids": expanded_by_index[index],
        }
        for index, state in enumerate(states)
    ]

    def replay(order):
        queue_keys = [
            list(
                module._replayed_frontier_queue_key(
                    state, ["facet-1"]
                )
            )
            for state in states
        ]
        return module._replay_frontier_path_label_closure(
            [
                {
                    "candidate_state": copy.deepcopy(states[index]),
                    "admitted": True,
                    "queue_key": queue_keys[index],
                    "frontier_serial": index,
                    "queue_size_before_pop": 1,
                    "queue_size_after_pop": 0,
                }
                for index in order
            ],
            frontier_enqueues=[
                {
                    "protocol_version": (
                        module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
                    ),
                    "enqueue_kind": (
                        "initial_entry"
                        if index == 0
                        else "expanded_edge"
                    ),
                    "serial": index,
                    "queue_key": queue_keys[index],
                    "candidate_state": copy.deepcopy(state),
                    "edge_id": (
                        None
                        if index == 0
                        else state["path_edge_ids"][-1]
                    ),
                    "parent_path_identity": (
                        None
                        if index == 0
                        else module._frontier_state_path_identity(
                            states[index - 1]
                        )
                    ),
                }
                for index, state in enumerate(states)
            ],
            initial_entry_ids=["root"],
            frontier_json=[
                {
                    "popped": copy.deepcopy(states[index]),
                    "key": queue_keys[index],
                    "serial": index,
                    "queue_size_before_pop": 1,
                    "queue_size_after_pop": 0,
                }
                for index in order
            ],
            path_labels=[
                copy.deepcopy(labels[index]) for index in order
            ],
            step_expanded_edge_ids=[
                "edge-root-middle",
                "edge-middle-leaf",
            ],
            popped_frontier_state=copy.deepcopy(
                states[order[0]]
            ),
            terminal_facts={
                "entry_count": 1,
                "frontier_expansion_count": 2,
                "frontier_remaining_count": 0,
            },
            required_facets=["facet-1"],
            layer="chunk",
        )

    assert replay([0, 1, 2]) is True
    assert replay([1, 0, 2]) is False
    assert replay([0, 2, 1]) is False
    assert replay([2, 0, 1]) is False


def _v6_direct_heap_replay(
    module,
    *,
    order,
    rejected_node=None,
    same_key_roots=False,
):
    root_a = {
        "layer": "chunk",
        "node_id": "root-a",
        "root_node_id": "root-a",
        "path": ["root-a"],
        "path_edge_ids": [],
        "path_edge_distances": [],
        "path_edge_strengths": [],
        "path_edge_types": [],
        "distance_so_far": 0.1,
        "reward_so_far": 0.0,
        "covered_facets": ["facet-1"],
        "evidence_roles": ["entry"],
        "depth": 0,
        "entry_parent_refs": [],
    }
    root_b = {
        **copy.deepcopy(root_a),
        "node_id": "root-b",
        "root_node_id": "root-b",
        "path": ["root-b"],
        "distance_so_far": (
            0.1 if same_key_roots else 0.3
        ),
    }
    child = {
        **copy.deepcopy(root_a),
        "node_id": "child",
        "root_node_id": "root-a",
        "path": ["root-a", "child"],
        "path_edge_ids": ["edge-root-a-child"],
        "path_edge_distances": [0.1],
        "path_edge_strengths": [0.9],
        "path_edge_types": ["semantic_affinity"],
        "distance_so_far": 0.2,
        "depth": 1,
    }
    states = {
        "root-a": root_a,
        "root-b": root_b,
        "child": child,
    }
    serials = {"root-a": 0, "root-b": 1, "child": 2}
    labels = {
        node_id: {
            "layer": "chunk",
            "node_id": node_id,
            "chunk_id": node_id,
            "root_node_id": state["root_node_id"],
            "path": list(state["path"]),
            "path_edge_ids": list(state["path_edge_ids"]),
            "path_edge_distances": list(
                state["path_edge_distances"]
            ),
            "path_edge_strengths": list(
                state["path_edge_strengths"]
            ),
            "distance_so_far": state["distance_so_far"],
            "reward_so_far": state["reward_so_far"],
            "covered_facets": list(state["covered_facets"]),
            "evidence_roles": list(state["evidence_roles"]),
            "expanded_edge_ids": (
                ["edge-root-a-child"]
                if node_id == "root-a"
                and not same_key_roots
                else []
            ),
        }
        for node_id, state in states.items()
    }
    event_nodes = (
        ["root-a", "root-b"]
        if same_key_roots
        else ["root-a", "root-b", "child"]
    )
    frontier_enqueues = []
    for node_id in event_nodes:
        state = states[node_id]
        frontier_enqueues.append(
            {
                "protocol_version": (
                    module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
                ),
                "enqueue_kind": (
                    "expanded_edge"
                    if node_id == "child"
                    else "initial_entry"
                ),
                "serial": serials[node_id],
                "queue_key": list(
                    module._replayed_frontier_queue_key(
                        state, ["facet-1"]
                    )
                ),
                "candidate_state": copy.deepcopy(state),
                "edge_id": (
                    "edge-root-a-child"
                    if node_id == "child"
                    else None
                ),
                "parent_path_identity": (
                    module._frontier_state_path_identity(root_a)
                    if node_id == "child"
                    else None
                ),
            }
        )
    observations = []
    frontier_json = []
    active_labels = []
    queue_sizes = (
        [(2, 1), (1, 0)]
        if same_key_roots
        else [(2, 1), (2, 1), (1, 0)]
    )
    for index, node_id in enumerate(order):
        state = states[node_id]
        admitted = node_id != rejected_node
        queue_before, queue_after = queue_sizes[index]
        queue_key = list(
            module._replayed_frontier_queue_key(
                state, ["facet-1"]
            )
        )
        observations.append(
            {
                "candidate_state": copy.deepcopy(state),
                "admitted": admitted,
                "queue_key": queue_key,
                "frontier_serial": serials[node_id],
                "queue_size_before_pop": queue_before,
                "queue_size_after_pop": queue_after,
            }
        )
        if admitted:
            frontier_json.append(
                {
                    "popped": copy.deepcopy(state),
                    "key": queue_key,
                    "serial": serials[node_id],
                    "queue_size_before_pop": queue_before,
                    "queue_size_after_pop": queue_after,
                }
            )
            active_labels.append(
                copy.deepcopy(labels[node_id])
            )
    return module._replay_frontier_path_label_closure(
        observations,
        frontier_enqueues=frontier_enqueues,
        initial_entry_ids=["root-a", "root-b"],
        frontier_json=frontier_json,
        path_labels=active_labels,
        step_expanded_edge_ids=(
            []
            if same_key_roots
            else ["edge-root-a-child"]
        ),
        popped_frontier_state=copy.deepcopy(
            frontier_json[0]["popped"]
        ),
        terminal_facts={
            "entry_count": 2,
            "frontier_expansion_count": (
                0 if same_key_roots else 1
            ),
            "frontier_remaining_count": 0,
        },
        required_facets=["facet-1"],
        layer="chunk",
    )


def test_v6_heap_replays_same_key_serial_and_expansion_priority():
    module = _load_quality_gate()
    assert _v6_direct_heap_replay(
        module,
        order=["root-a", "child", "root-b"],
    ) is True
    assert _v6_direct_heap_replay(
        module,
        order=["root-b", "root-a"],
        same_key_roots=True,
    ) is False
    assert _v6_direct_heap_replay(
        module,
        order=["root-a", "root-b", "child"],
    ) is False
    assert _v6_direct_heap_replay(
        module,
        order=["root-a", "root-b", "child"],
        rejected_node="child",
    ) is False


@pytest.mark.parametrize(
    "attack",
    [
        "extra_tail_enqueue",
        "duplicate_enqueue_event",
        "forged_remaining_count",
    ],
)
def test_v6_heap_tail_and_terminal_count_fail_closed(attack):
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    step = fixture["steps"][2]
    diagnostics = step["diagnostics"]
    if attack == "forged_remaining_count":
        diagnostics["terminal_facts"][
            "frontier_remaining_count"
        ] = 1
    else:
        if attack == "duplicate_enqueue_event":
            duplicate = copy.deepcopy(
                diagnostics["frontier_enqueues"][1]
            )
            duplicate["serial"] = 2
            diagnostics["frontier_enqueues"].append(duplicate)
            diagnostics["terminal_facts"][
                "frontier_expansion_count"
            ] = 2
            tail_state = None
        else:
            tail_state = copy.deepcopy(
                diagnostics["frontier_enqueues"][1][
                    "candidate_state"
                ]
            )
        if tail_state is None:
            audit = module.audit_retrieval_quality(fixture)
            assert audit["pass"] is False, audit
            assert (
                "frontier_candidate_path_label_edge_distance_closure_mismatch"
                in {
                    finding["code"]
                    for finding in audit["findings"]
                }
            )
            return
        tail_state.update(
            node_id="tail",
            path=["chunk-1", "tail"],
            path_edge_ids=["edge-tail"],
            path_edge_distances=[0.7],
            path_edge_strengths=[0.5],
            distance_so_far=0.8,
        )
        diagnostics["frontier_enqueues"].append(
            {
                "protocol_version": (
                    module.FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
                ),
                "enqueue_kind": "expanded_edge",
                "serial": 2,
                "queue_key": list(
                    module._replayed_frontier_queue_key(
                        tail_state, ["facet-1"]
                    )
                ),
                "candidate_state": tail_state,
                "edge_id": "edge-tail",
                "parent_path_identity": (
                    module._frontier_state_path_identity(
                        diagnostics["frontier_json"][0][
                            "popped"
                        ]
                    )
                ),
            }
        )
        diagnostics["terminal_facts"][
            "frontier_expansion_count"
        ] = 2

    audit = module.audit_retrieval_quality(fixture)

    assert audit["pass"] is False, audit
    assert (
        "frontier_candidate_path_label_edge_distance_closure_mismatch"
        in {finding["code"] for finding in audit["findings"]}
    )


def test_v3_dominance_count_replays_raw_label_admission_events():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    fixture["steps"][0]["dominance_pruned_count"] = 999
    fixture["convergence"]["layers"]["mid"][
        "dominance_pruned_count"
    ] = 999
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    assert "dominance_raw_event_replay_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_dominance_observation_cannot_override_raw_candidate_state():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    observation = fixture["steps"][0]["diagnostics"][
        "dominance_observations"
    ][0]
    observation["label"]["distance"] = 0.01
    observation["queue_key"] = [0, 0.01, 0, -1]
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    assert "dominance_raw_event_replay_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_cycle_reward_replays_path_events_and_frozen_envelope_cap():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    fixture["steps"][0]["cycle_distance_reward"] = 9.0
    fixture["steps"][0]["diagnostics"]["path_labels"][0][
        "reward_so_far"
    ] = 9.0
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    assert "cycle_distance_reward_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_cycle_events_are_exactly_derived_from_path_facts():
    module = _load_quality_gate()
    envelope = {
        "max_cycle_reward_per_path": 0.25,
        "cycle_reward_distance_threshold": 0.8,
    }
    edge_strength = 0.7
    cycle_distance = 0.3
    reward = round(
        0.04
        * edge_strength
        * math.exp(
            -cycle_distance
            / envelope["cycle_reward_distance_threshold"]
        ),
        6,
    )
    event = {
        "protocol_version": module.CYCLE_REWARD_REPLAY_PROTOCOL_VERSION,
        "cycle_edges": ["edge-a-b", "edge-b-a"],
        "cycle_distance": cycle_distance,
        "edge_strength": edge_strength,
        "support_delta": 1,
        "reward_before_cap": reward,
        "reward_after_cap": reward,
        "cap_reason": "within_cap",
    }
    label = {
        "path": ["chunk-a", "chunk-b", "chunk-a"],
        "path_edge_ids": ["edge-a-b", "edge-b-a"],
        "path_edge_distances": [0.1, 0.2],
        "path_edge_strengths": [0.8, edge_strength],
        "cycle_distance_rewards": [event],
        "reward_so_far": reward,
    }
    assert module._replay_path_cycle_reward(
        label, envelope
    )[0] is True

    duplicated = copy.deepcopy(label)
    duplicated["cycle_distance_rewards"].append(
        copy.deepcopy(event)
    )
    assert module._replay_path_cycle_reward(
        duplicated, envelope
    )[0] is False

    forged_strength = copy.deepcopy(label)
    forged_strength["cycle_distance_rewards"][0][
        "edge_strength"
    ] = 0.99
    assert module._replay_path_cycle_reward(
        forged_strength, envelope
    )[0] is False


def test_v3_frozen_envelope_hash_cannot_be_rewritten_partially():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    fixture["trace_diagnostics"]["agent_operating_envelope"][
        "max_cycle_reward_per_path"
    ] = 0.99
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    assert "retrieval_operating_envelope_hash_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_convergence_reason_must_match_terminal_facts_and_allowlist():
    module = _load_quality_gate()
    fixture = _retrieval_fixture()
    fixture["steps"][2]["stop_reason"] = "llm_said_stop"
    fixture["convergence"]["layers"]["chunk"]["reason"] = "llm_said_stop"
    fixture["convergence"]["reason"] = "llm_said_stop"
    audit = module.audit_retrieval_quality(fixture)
    assert audit["pass"] is False, audit
    assert "convergence_terminal_fact_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_unknown_agent_action_cannot_pass_with_validation_booleans():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    validation_event = next(
        event
        for event in fixture["trace"]
        if event["node"] == "typed_action_validation"
    )
    validation_event["scores"]["validation"]["accepted"][0][
        "action_type"
    ] = "llm_gray_zone_override"
    audit = _audit_agent_fixture(module, fixture)
    assert audit["pass"] is False, audit
    codes = {finding["code"] for finding in audit["findings"]}
    assert "typed_action_validation_incomplete" in codes
    assert "typed_action_persisted_replay_mismatch" in codes


def test_v3_allowed_typed_action_cannot_add_unknown_budget_key():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    facts = fixture["test_typed_action_facts"]
    action = facts["plans"][0]["typed_actions"][0]
    row = facts["plans"][0]["actions"][0]
    action["budget_request"]["llm_override_budget"] = 1
    row["budget_request"]["llm_override_budget"] = 1
    audit = _audit_agent_fixture(module, fixture)
    assert audit["pass"] is False, audit
    assert "typed_action_persisted_replay_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_agent_trace_sequence_must_be_contiguous_and_run_bound():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    fixture["trace"][1]["sequence_index"] = 0
    fixture["trace"][2]["run_id"] = "different-run"
    audit = _audit_agent_fixture(module, fixture)
    assert audit["pass"] is False, audit
    assert "agent_trace_sequence_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def _resequence_agent_trace(trace: list[dict]) -> None:
    for index, event in enumerate(trace):
        event["sequence_index"] = index


def test_v4_evidence_gate_rejects_blocked_false_model_called_foreign_insufficient():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    event = next(
        event
        for event in fixture["trace"]
        if event["node"] == "evidence_gate"
    )
    event["status"] = "blocked"
    event["scores"]["context_package_evidence_gate_passed"] = False
    event["scores"]["answer_model_called"] = True
    event["scores"]["retrieval_trace_id"] = "foreign-trace"
    verdict = dict(event["scores"]["evidence_evaluator"])
    verdict["verdict"] = "insufficient_corpus"
    verdict["decision_hash"] = module.stable_hash(
        {
            key: value
            for key, value in verdict.items()
            if key != "decision_hash"
        }
    )
    event["scores"]["evidence_evaluator"] = verdict

    audit = _audit_agent_fixture(module, fixture)

    assert audit["pass"] is False, audit
    assert "evidence_gate_payload_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v4_agent_trace_rejects_foreign_node_after_contiguous_resequence():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    insert_at = next(
        index
        for index, event in enumerate(fixture["trace"])
        if event["node"] == "evidence_gate"
    )
    fixture["trace"].insert(
        insert_at,
        {
            "id": "event-llm-gray-override",
            "run_id": fixture["run_id"],
            "sequence_index": -1,
            "node": "llm_gray_override",
            "status": "completed",
            "input_summary": "",
            "output_summary": "",
            "document_ids": [],
            "scores": {
                "decision": "continue_path",
                "model_call_count": 1,
            },
            "duration_ms": 0,
            "error": None,
        },
    )
    _resequence_agent_trace(fixture["trace"])

    audit = _audit_agent_fixture(module, fixture)

    assert audit["pass"] is False, audit
    assert "agent_trace_stage_grammar_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v4_agent_trace_rejects_duplicate_executor_after_contiguous_resequence():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    executor_index = next(
        index
        for index, event in enumerate(fixture["trace"])
        if event["node"] == "typed_action_executor"
    )
    duplicate = copy.deepcopy(fixture["trace"][executor_index])
    duplicate["id"] = "event-duplicate-executor"
    fixture["trace"].insert(executor_index + 1, duplicate)
    _resequence_agent_trace(fixture["trace"])

    audit = _audit_agent_fixture(module, fixture)

    assert audit["pass"] is False, audit
    assert "agent_trace_stage_grammar_invalid" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v4_wholesale_foreign_run_rewrite_cannot_replace_persisted_anchor():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    fixture["run_id"] = "foreign-run"
    for event in fixture["trace"]:
        event["run_id"] = "foreign-run"
    typed_action_facts = fixture["test_typed_action_facts"]
    typed_action_facts["run_id"] = "foreign-run"
    for plan in typed_action_facts["plans"]:
        plan["run_id"] = "foreign-run"
        for action in plan["actions"]:
            action["run_id"] = "foreign-run"

    audit = _audit_agent_fixture(module, fixture)

    assert audit["pass"] is False, audit
    codes = {finding["code"] for finding in audit["findings"]}
    assert "persisted_agent_run_trace_replay_mismatch" in codes
    assert "agent_reciprocal_binding_mismatch" in codes


def _agent_fixture_with_one_repair(module) -> tuple[dict, dict]:
    fixture = _agent_fixture()
    repaired_package_id = "package-2"
    repaired_trace_id = "trace-quality-gate-2"
    repair_record = {
        "protocol_version": "typed_repair_v1",
        "repair_round_index": 0,
        "remaining_repair_budget_before": 2,
        "remaining_repair_budget_after": 1,
        "action_type": "repair_structure_context",
        "before_context_package_id": "package-1",
        "repaired_context_package_id": repaired_package_id,
        "before_retrieval_trace_id": "trace-quality-gate-1",
        "repaired_retrieval_trace_id": repaired_trace_id,
        "gray_zone_model_call_count": 0,
        "gray_zone_decision_authority": (
            "deterministic_executor_only"
        ),
        "global_top_k_increased": False,
        "repair_audit": {
            "gray_zone_model_call_count": 0,
            "gray_zone_decision_authority": (
                "deterministic_executor_only"
            ),
        },
    }
    fixture["model_audit"].update(
        {
            "context_package_id": repaired_package_id,
            "retrieval_trace_id": repaired_trace_id,
            "repair_rounds_used": 1,
            "repair_actions": [copy.deepcopy(repair_record)],
        }
    )
    fixture["context_package_id"] = repaired_package_id
    fixture["retrieval_trace_id"] = repaired_trace_id
    citation = fixture["citations"][0]
    citation["context_package_id"] = repaired_package_id
    citation["retrieval_trace_id"] = repaired_trace_id
    citation["source_span"]["context_package_id"] = (
        repaired_package_id
    )
    citation["source_span"]["retrieval_trace_id"] = repaired_trace_id
    repair_insert_index = next(
        index
        for index, event in enumerate(fixture["trace"])
        if event["node"] == "citation_verification"
    )
    fixture["trace"].insert(
        repair_insert_index,
        {
            "id": "event-repair-0",
            "run_id": fixture["run_id"],
            "sequence_index": -1,
            "node": "repair_executed",
            "status": "completed",
            "input_summary": "",
            "output_summary": "",
            "document_ids": [],
            "scores": copy.deepcopy(repair_record),
            "duration_ms": 0,
            "error": None,
        },
    )
    _resequence_agent_trace(fixture["trace"])
    persisted = fixture["test_persisted_agent_facts"]
    persisted["trace_events"] = copy.deepcopy(fixture["trace"])
    persisted["bindings"]["retrieval_trace"] = {
        "id": repaired_trace_id,
        "knowledge_base_id": "kb-1",
    }
    persisted["bindings"]["context_package"] = {
        "id": repaired_package_id,
        "knowledge_base_id": "kb-1",
        "retrieval_trace_id": repaired_trace_id,
    }
    persisted["bindings"]["answer_session"][
        "retrieval_trace_id"
    ] = repaired_trace_id
    persisted["bindings"]["answer_session"][
        "context_package_id"
    ] = repaired_package_id
    persisted["bindings"]["reward_event"][
        "retrieval_trace_id"
    ] = repaired_trace_id
    persisted["bindings"]["reward_event"][
        "context_package_id"
    ] = repaired_package_id
    persisted["snapshot_hash"] = module.persisted_agent_snapshot_hash(
        persisted
    )
    retrieval_fixture = _retrieval_fixture()
    retrieval_fixture["trace_id"] = repaired_trace_id
    return fixture, retrieval_fixture


def test_v4_repair_address_chain_binds_initial_and_final_persisted_facts():
    module = _load_quality_gate()
    fixture, retrieval_fixture = _agent_fixture_with_one_repair(
        module
    )

    audit = _audit_agent_fixture(
        module,
        fixture,
        retrieval_fixture=retrieval_fixture,
    )

    assert audit["pass"] is True, audit


def test_v4_repair_address_chain_rejects_broken_reciprocal_link():
    module = _load_quality_gate()
    fixture, retrieval_fixture = _agent_fixture_with_one_repair(
        module
    )
    fixture["model_audit"]["repair_actions"][0][
        "before_context_package_id"
    ] = "foreign-package"

    audit = _audit_agent_fixture(
        module,
        fixture,
        retrieval_fixture=retrieval_fixture,
    )

    assert audit["pass"] is False, audit
    assert "agent_reciprocal_binding_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_malformed_repair_without_round_index_is_not_ignored():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    fixture["model_audit"]["repair_actions"].append(
        {
            "action_type": "llm_repair_override",
            "remaining_repair_budget_before": 2,
            "remaining_repair_budget_after": 1,
        }
    )
    audit = _audit_agent_fixture(module, fixture)
    assert audit["pass"] is False, audit
    assert "repair_budget_or_action_mismatch" in {
        finding["code"] for finding in audit["findings"]
    }


def test_v3_agent_rejects_boolean_only_retrieval_quality_without_raw_snapshot():
    module = _load_quality_gate()
    fixture = _agent_fixture()
    audit = module.audit_agent_quality(
        fixture,
        typed_action_facts=fixture["test_typed_action_facts"],
        retrieval_quality={"pass": True},
    )
    assert audit["pass"] is False, audit
    assert "agent_retrieval_quality_gate_failed" in {
        finding["code"] for finding in audit["findings"]
    }


@pytest.mark.asyncio
async def test_v3_persisted_retrieval_trace_replays_raw_quality_facts(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import GraphRetrievalStep
    from app.schemas import SearchFilters
    from app.services.context_graph import layered_search

    module = _load_quality_gate()
    knowledge_base = populated_context_graph["knowledge_base"]
    result = await layered_search(
        db_session,
        knowledge_base.id,
        "Bayes theorem prior posterior",
        SearchFilters(),
        3,
    )
    db_session.flush()
    steps = list(
        db_session.scalars(
            select(GraphRetrievalStep)
            .where(GraphRetrievalStep.retrieval_trace_id == result.trace.id)
            .order_by(GraphRetrievalStep.step_index.asc())
        ).all()
    )
    snapshot = module.retrieval_snapshot_from_records(result.trace, steps)
    audit = module.audit_retrieval_quality(snapshot)

    assert audit["pass"] is True, json.dumps(audit, indent=2, sort_keys=True)
    assert audit["checks"]["gray_zone_raw_trace_full_local_replay"] is True


@pytest.mark.asyncio
async def test_v3_persisted_context_package_replays_db_and_snapshot_facts(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from app.schemas import SearchFilters
    from app.services.context_graph import (
        build_context_package,
        layered_search,
    )
    from app.services.retrieval import get_retrieval_trace_steps

    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    sys.modules.pop("check_context_package_quality", None)
    checker = importlib.import_module(
        "check_context_package_quality"
    )
    module = _load_quality_gate()
    knowledge_base = populated_context_graph["knowledge_base"]
    retrieval = await layered_search(
        db_session,
        knowledge_base.id,
        "Bayes theorem prior posterior",
        SearchFilters(),
        3,
    )
    package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query="Bayes theorem prior posterior",
        trace=retrieval.trace,
        results=retrieval.results,
        token_budget=2400,
    )
    db_session.flush()
    snapshot = checker.persisted_context_package_quality_snapshot(
        db_session,
        package.id,
    )
    audit = module.audit_context_package_quality(snapshot)
    public_trace = get_retrieval_trace_steps(db_session, retrieval.trace.id)
    frontier_audit = checker.audit_retrieval_frontier_convergence(
        public_trace
    )

    assert audit["pass"] is True, json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    assert frontier_audit["pass"] is True, json.dumps(
        frontier_audit,
        indent=2,
        sort_keys=True,
    )
    assert frontier_audit["participating_layers"] == ["mid", "chunk"]
    assert public_trace["stage_queues"]["coarse"] == {
        "entry_ids": [],
        "forced_entry_ids": [],
        "accepted_ids": [],
        "frontier_pop_count": 0,
        "skipped_by_granularity": "mid",
        "reason": "skipped_by_granularity=mid",
    }
    assert public_trace["gray_zone_model_call_count"] == 0

    with pytest.raises(
        SystemExit,
        match="does not belong to the selected knowledge base",
    ):
        checker.audit_persisted_context_package(
            db_session,
            package_id=package.id,
            expected_knowledge_base_id="different-kb",
        )

    reports = []
    monkeypatch.setattr(
        checker,
        "session_scope",
        lambda: nullcontext(db_session),
    )
    monkeypatch.setattr(
        checker,
        "resolve_knowledge_base",
        lambda *_args, **_kwargs: knowledge_base,
    )
    monkeypatch.setattr(
        checker,
        "prepare_runtime_for_model_io",
        lambda: pytest.fail(
            "persisted package audit must not initialize model I/O"
        ),
    )
    monkeypatch.setattr(
        checker,
        "write_report",
        lambda _name, payload: reports.append(payload)
        or Path("output/quality-gate/read-only-package.json"),
    )
    monkeypatch.setattr(
        db_session,
        "commit",
        lambda: pytest.fail(
            "persisted package audit must not commit the database session"
        ),
    )

    await checker.main(
        SimpleNamespace(
            knowledge_base_id=knowledge_base.id,
            knowledge_base_name=None,
            query=None,
            context_package_id=package.id,
            top_k=3,
            execute=False,
        )
    )

    assert reports[0]["mode"] == "persisted_context_package_read_only"
    assert reports[0]["pass"] is True
    assert reports[0]["context_package_id"] == package.id
    assert reports[0]["retrieval_trace_id"] == retrieval.trace.id
    assert reports[0]["targets"]["knowledge_base_id"] == knowledge_base.id
    assert "no retrieval" in reports[0]["impact"]


@pytest.mark.asyncio
async def test_v3_persisted_agent_actions_and_retrieval_replay_from_db(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from app.schemas import AgentRequest, SearchFilters
    from app.services import agent_graph

    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    sys.modules.pop("evaluate_agent_trace", None)
    evaluator = importlib.import_module("evaluate_agent_trace")

    class _BoundSessionScope:
        def __enter__(self):
            return db_session

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    monkeypatch.setattr(
        evaluator,
        "session_scope",
        lambda: _BoundSessionScope(),
    )
    module = _load_quality_gate()
    knowledge_base = populated_context_graph["knowledge_base"]
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=knowledge_base.id,
            question="Explain Bayesian network factorization.",
            filters=SearchFilters(),
            top_k=4,
        ),
    )
    db_session.flush()
    retrieval_snapshot = evaluator.persisted_retrieval_snapshot(
        response["retrieval_trace_id"]
    )
    active_required_facets = sorted(
        retrieval_snapshot["query_facets"]["required_facets"]
    )
    deterministic_gray_required_facets = (
        module._deterministic_gray_required_facets(
            retrieval_snapshot
        )
    )
    assert active_required_facets == [
        "Bayesian network",
        "factorization",
    ]
    assert active_required_facets != deterministic_gray_required_facets
    assert module._active_traversal_required_facets(
        retrieval_snapshot
    ) == active_required_facets
    typed_action_facts = evaluator.persisted_typed_action_facts(
        response["run_id"]
    )
    persisted_agent_facts = (
        evaluator.persisted_agent_quality_snapshot(
            response["run_id"],
            expected_knowledge_base_id=knowledge_base.id,
        )
    )
    persisted_response, persisted_identity, pe_audit = (
        evaluator.persisted_agent_response_bundle(
            response["run_id"],
            expected_knowledge_base_id=knowledge_base.id,
            persisted_agent_facts=persisted_agent_facts,
        )
    )
    gray_zone_audit = module._local_gray_zone_audit(
        retrieval_snapshot
    )
    retrieval_quality = module.audit_retrieval_quality(
        retrieval_snapshot,
        gray_zone_audit=gray_zone_audit,
    )
    audit = module.audit_agent_quality(
        persisted_response,
        retrieval_snapshot=retrieval_snapshot,
        gray_zone_audit=gray_zone_audit,
        typed_action_facts=typed_action_facts,
        retrieval_quality=retrieval_quality,
        persisted_agent_facts=persisted_agent_facts,
    )

    assert retrieval_quality["pass"] is True, json.dumps(
        {
            "findings": retrieval_quality["findings"],
            "chunk_walk_step": next(
                (
                    step
                    for step in retrieval_snapshot["steps"]
                    if step.get("layer") == "chunk"
                    and step.get("action") == "walk_graph_frontier"
                ),
                None,
            ),
            "chunk_topk": (
                retrieval_snapshot.get("topk_selection") or {}
            ).get("chunk"),
            "chunk_stage_queue": (
                retrieval_snapshot.get("stage_queues") or {}
            ).get("chunk"),
        },
        indent=2,
        sort_keys=True,
    )
    assert audit["pass"] is True, json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    assert persisted_response["answer"] == response["answer"]
    assert len(persisted_response["citations"]) == len(response["citations"])
    for replayed_citation, live_citation in zip(
        persisted_response["citations"],
        response["citations"],
        strict=True,
    ):
        for identity_field in (
            "claim_id",
            "chunk_id",
            "document_id",
            "document_version_id",
            "context_package_id",
            "retrieval_trace_id",
            "answer_session_id",
            "citation_verification_id",
        ):
            assert replayed_citation[identity_field] == live_citation[
                identity_field
            ]
        assert replayed_citation["source_span"] == live_citation[
            "source_span"
        ]
        assert replayed_citation["verification"] == live_citation[
            "verification"
        ]
    assert persisted_identity["run"]["knowledge_base_id"] == knowledge_base.id
    assert persisted_identity["citation_verification_ids"]
    assert pe_audit["run_id"] == response["run_id"]
    assert pe_audit["counts"]["plans"] == len(pe_audit["plans"])


def test_quality_gate_cli_returns_zero_for_good_and_nonzero_for_bad(tmp_path):
    module = _load_quality_gate()
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    zero_distance = tmp_path / "strength-one-zero-distance.json"
    invalid_zero_distance = tmp_path / "subunit-strength-zero-distance.json"
    good.write_text(json.dumps({"subject": "graph", "snapshot": _graph_fixture(module)}), encoding="utf-8")
    broken = _graph_fixture(module)
    broken["relation_edges"][0]["distance"] = 99.0
    bad.write_text(json.dumps({"subject": "graph", "snapshot": broken}), encoding="utf-8")
    blocked = _graph_fixture(module)
    _set_relation_strength_consistently(module, blocked, index=0, strength=1.0)
    zero_distance.write_text(
        json.dumps({"subject": "graph", "snapshot": blocked}),
        encoding="utf-8",
    )
    invalid_zero = _graph_fixture(module)
    _set_relation_claimed_distance_consistently(
        invalid_zero, index=0, distance=0.0
    )
    invalid_zero_distance.write_text(
        json.dumps({"subject": "graph", "snapshot": invalid_zero}),
        encoding="utf-8",
    )

    good_result = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "_quality_gate.py"), "--fixture", str(good)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    bad_result = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "_quality_gate.py"), "--fixture", str(bad)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    zero_distance_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "_quality_gate.py"),
            "--fixture",
            str(zero_distance),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    invalid_zero_distance_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "_quality_gate.py"),
            "--fixture",
            str(invalid_zero_distance),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert good_result.returncode == 0, good_result.stdout + good_result.stderr
    assert bad_result.returncode != 0
    assert '"pass": false' in bad_result.stdout.lower()
    assert zero_distance_result.returncode == 0
    assert '"pass": true' in zero_distance_result.stdout.lower()
    assert invalid_zero_distance_result.returncode != 0
    assert "relation_edge_distance_mismatch" in invalid_zero_distance_result.stdout
