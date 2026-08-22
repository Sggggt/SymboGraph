from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DISTANCE_REPLAY_ABS_TOLERANCE = 1e-9
QUALITY_GATE_PROTOCOL_VERSION = "four_layer_acceptance_quality_gate_v11"
GRAPH_OPERATING_POINT_PROTOCOL_VERSION = (
    "dense_dynamic_knn_bridge_quota_edge_calibration_v2"
)
RELATION_PROTOCOL_VERSION = "dense_only_chunk_relation_graph_v9"
EDGE_DISTANCE_PROTOCOL_VERSION = "edge_distance_log_calibrated_strength_v2"
EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION = "type_local_winsorized_minmax_v1"
EDGE_TYPE_CALIBRATION_EDGE_TYPES = (
    "dense_semantic",
    "dense_cross_document_bridge",
    "dense_cross_language_bridge",
)
EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS = {
    "lower_quantile": 0.05,
    "upper_quantile": 0.95,
    "min_span": 0.05,
    "strength_floor": 0.05,
}
EDGE_PROJECTION_PROTOCOL_VERSION = "membership_q15_layer_type_calibrated_v3"
PROJECTION_AGGREGATION_PROTOCOL_VERSION = (
    "membership_weighted_bottom_support_q15_log_mass_v1"
)
PROJECTION_CALIBRATION_PROTOCOL_VERSION = (
    "layer_edge_type_winsorized_minmax_v1"
)
PROJECTION_CALIBRATION_PARAMS = dict(EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS)
PROJECTED_EDGE_TYPE_BY_BOTTOM_TYPE = {
    "dense_semantic": "co_occurs_with",
    "dense_cross_document_bridge": "bridge_to",
    "dense_cross_language_bridge": "bridge_to",
}
EDGE_SEMANTIC_UNCERTAINTY_PROTOCOL_VERSION = (
    "edge_semantic_uncertainty_margin_v1"
)
RQ_BOUNDARY_PROTOCOL_VERSION = "rq_leaf_path_boundary_v1"
PROJECTED_GRAY_PREDICATE_PROTOCOL_VERSION = (
    "projected_gray_predicates_support_rollup_v1"
)
CHUNK_BUSINESS_KEY_PROTOCOL_VERSION = "chunk_business_key_v1"
ASSIGNMENT_PROTOCOL_VERSION = "scope_key_chunk_business_assignment_v1"
RELATION_SUPPORT_PROTOCOL_VERSION = "dense_relation_endpoint_support_v1"
RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION = "rq_membership_role_entropy_boundary_v1"
RQ_MEMBERSHIP_ROLE_THRESHOLDS = {
    "noise_membership_score_max": 1e-8,
    "outlier_gamma_max": 0.25,
    "outlier_residual_quantile": 0.95,
    "low_confidence_gamma_max": 0.35,
    "low_confidence_membership_score_max": 0.01,
    "boundary_entropy_min": 0.65,
    "boundary_probability_margin_max": 0.15,
    "boundary_distance_max": 0.05,
}
QUERY_RQ_SEED_PROTOCOL_VERSION = (
    "query_rq_fuzzy_membership_chunk_seed_v2"
)
CHUNK_FACET_PRIORITY_PROTOCOL_VERSION = (
    "validated_query_facet_posterior_chunk_priority_v2"
)
QUERY_FACET_POSTERIOR_PROTOCOL_VERSION = (
    "query_facet_posterior_calibration_v1"
)
QUERY_FACET_POSTERIOR_CHECKPOINTS = (
    "dense_entry_candidates",
    "merged_chunk_candidates",
)
QUERY_FACET_POSTERIOR_ROUND_DIGITS = 6
QUERY_FACET_POSTERIOR_LLM_SAMPLE_BUDGET = 0
QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION = (
    "validated_query_facet_ordered_window_v1"
)
QUERY_FACET_ORDERED_WINDOW_MIN_SPAN = 12
QUERY_FACET_ORDERED_WINDOW_TOKEN_MULTIPLIER = 4
QUERY_RQ_CHUNK_SEED_COMPONENT_WEIGHTS = {
    "rq_relevance": 0.70,
    "mid_entry": 0.20,
    "dense": 0.10,
}
QUERY_RQ_MID_SUPPORT_FALLBACK_COMPONENT_WEIGHTS = {
    "rq_relevance": 0.0,
    "mid_entry": 0.80,
    "dense": 0.20,
}
QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK = {
    "primary_member": 0,
    "fuzzy_member": 1,
    "boundary_member": 2,
    "outlier_member": 3,
    "noise_candidate": 4,
}
CHUNK_TEXT_HASH_PROTOCOL_VERSION = "chunk_text_sha256_normalized_v1"
RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION = "raw_chunk_span_utf8_sha256_v1"
SOURCE_SNAPSHOT_PROTOCOL_VERSION = (
    "sha256_content_addressed_readonly_source_snapshot_v2"
)
TOPK_RANKING_PROTOCOL_VERSION = (
    "forced_prefix_then_frozen_queue_key_path_identity_v1"
)
DOMINANCE_REPLAY_PROTOCOL_VERSION = "pareto_label_admission_replay_v2"
FRONTIER_PATH_LABEL_CLOSURE_PROTOCOL_VERSION = (
    "frontier_min_heap_key_serial_temporal_closure_v3"
)
FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION = (
    "frontier_heap_enqueue_key_serial_v1"
)
CYCLE_REWARD_REPLAY_PROTOCOL_VERSION = "bounded_cycle_distance_reward_replay_v1"
CONVERGENCE_REPLAY_PROTOCOL_VERSION = "deterministic_terminal_fact_replay_v1"
REPAIR_EVIDENCE_RETENTION_PROTOCOL_VERSION = (
    "repair_supported_evidence_carry_forward_v1"
)
TYPED_ACTION_SCHEMA_PROTOCOL_VERSION = "typed_action_schema_v4"
HISTORICAL_TYPED_ACTION_SCHEMA_REQUIRED_ACTIONS_BY_HASH = {
    "46cf68bf72801b62562fe25715c6578c597804d6412a980907b8ebf63ff46f46": (
        "select_entry_nodes",
        "walk_graph_frontier",
        "recall_chunks",
        "restore_context_package",
        "verify_citations",
    ),
    "606a629895e3e02450708a767f797c228b5790b9ee40bdbf97b8fdf909483455": (
        "select_entry_nodes",
        "walk_graph_frontier",
        "recall_chunks",
        "restore_context_package",
        "build_context_package",
        "verify_citations",
    ),
}
HISTORICAL_TYPED_ACTION_SCHEMA_PROTOCOL_HASHES = {
    "typed_action_schema_v3": frozenset(
        HISTORICAL_TYPED_ACTION_SCHEMA_REQUIRED_ACTIONS_BY_HASH
    ),
}
TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION = "planner_typed_action_executor_v2"
PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION = "postgres_agent_run_trace_replay_v2"
AGENT_TRACE_STAGE_PROTOCOL_VERSION = "completed_agent_trace_stage_grammar_v1"
AGENT_TRACE_PLANNING_STAGE_GROUP = (
    "agent_planner",
    "typed_action_validation",
    "typed_action_executor",
    "evidence_evaluator",
)
AGENT_TRACE_FIXED_PREFIX = (
    "query_understanding",
    "query_facet_extraction",
)
AGENT_TRACE_FIXED_SUFFIX_BEFORE_REPAIR = (
    "evidence_gate",
    "entry_selection",
    "layer_drilldown",
    "frontier_traversal",
    "chunk_recall",
    "layered_retrieval",
    "structure_context_restoration",
    "context_package",
    "grounded_answer",
)
AGENT_TRACE_FIXED_SUFFIX_AFTER_REPAIR = (
    "citation_verification",
    "reward_event",
)
AGENT_TRACE_ALLOWED_COMPLETED_NODES = frozenset(
    AGENT_TRACE_FIXED_PREFIX
    + AGENT_TRACE_PLANNING_STAGE_GROUP
    + AGENT_TRACE_FIXED_SUFFIX_BEFORE_REPAIR
    + ("repair_executed",)
    + AGENT_TRACE_FIXED_SUFFIX_AFTER_REPAIR
)
EVIDENCE_GATE_SCORE_FIELDS = frozenset(
    {
        "retrieval_trace_id",
        "retrieval_granularity",
        "agent_plan_id",
        "agent_plan_index",
        "planning_rounds_used",
        "typed_action_control_hash",
        "evidence_evaluator",
        "context_package_evidence_gate_passed",
        "answer_model_called",
        "conversation_state_scope_hash",
    }
)
ALLOWED_TYPED_ACTIONS = {
    "activate_coarse_concepts",
    "route_mid_concepts",
    "route_rq_addresses",
    "select_entry_nodes",
    "walk_graph_frontier",
    "drill_down_layer",
    "jump_bridge",
    "stop_and_collect_chunks",
    "need_more_evidence",
    "recall_chunks",
    "restore_context_package",
    "build_context_package",
    "verify_citations",
    "repair_missing_citation",
    "repair_concept_gap",
    "repair_bridge_gap",
    "repair_structure_context",
}
REQUIRED_TYPED_ACTIONS = [
    "select_entry_nodes",
    "walk_graph_frontier",
    "recall_chunks",
    "restore_context_package",
    "build_context_package",
    "verify_citations",
]
TYPED_ACTION_REQUIRED_FIELDS = {
    "action_type",
    "target_ids",
    "reason",
    "budget_request",
    "expected_evidence",
    "stop_condition",
}
EXPECTED_EVIDENCE_FIELDS = {
    "source",
    "requires_chunk_spans",
    "required_facets",
    "allowed_relation_types",
    "relation_types",
    "required_restore_modes",
    "minimum_independent_support_paths",
    "required_evidence_roles",
    "failure_types",
    "start_layer",
    "target_layer",
    "fallback_allowed",
    "required_verification_stage",
    "protocol_version",
    "executor_mechanism",
    "failure_card_hashes",
    "action_input_hash",
    "canonical_target_refs",
}
STOP_CONDITION_FIELDS = {
    "sufficient_evidence",
    "required_action_complete",
    "all_required_facets_covered",
    "independent_support_paths_at_least",
    "citation_verification_passes",
    "frontier_empty",
    "all_claims_supported",
    "no_semantic_progress",
}
TYPED_ACTION_BUDGET_KEYS = {
    "activate_coarse_concepts": {
        "agent_coarse_initial_budget",
        "agent_coarse_top_k",
    },
    "route_mid_concepts": {
        "agent_mid_per_coarse_budget",
        "agent_coarse_drilldown_mid_initial_budget",
        "agent_mid_initial_budget",
        "agent_mid_top_k",
    },
    "route_rq_addresses": {
        "agent_chunk_per_mid_budget",
        "agent_chunk_initial_budget",
    },
    "select_entry_nodes": {
        "agent_coarse_initial_budget",
        "agent_coarse_drilldown_mid_initial_budget",
        "agent_mid_initial_budget",
        "agent_chunk_initial_budget",
    },
    "walk_graph_frontier": {
        "agent_chunk_per_mid_budget",
        "max_depth_per_layer",
        "max_labels_per_node",
        "max_edge_reuse",
    },
    "drill_down_layer": {
        "agent_coarse_top_k",
        "agent_mid_per_coarse_budget",
        "agent_mid_top_k",
        "agent_chunk_per_mid_budget",
        "agent_chunk_top_k",
    },
    "jump_bridge": {
        "agent_chunk_per_mid_budget",
        "max_depth_per_layer",
    },
    "stop_and_collect_chunks": {"agent_chunk_top_k"},
    "need_more_evidence": {
        "agent_mid_top_k",
        "agent_chunk_top_k",
    },
    "recall_chunks": {"agent_chunk_top_k"},
    "restore_context_package": {
        "structure_restore_per_chunk_budget"
    },
    "build_context_package": {"context_package_token_budget"},
    "verify_citations": {"verification_budget"},
    "repair_missing_citation": {"repair_round_budget"},
    "repair_concept_gap": {"repair_round_budget"},
    "repair_bridge_gap": {"repair_round_budget"},
    "repair_structure_context": {
        "repair_round_budget",
        "structure_restore_per_chunk_budget",
    },
}
TYPED_ACTION_TARGET_LAYERS = {
    "activate_coarse_concepts": {"coarse"},
    "route_mid_concepts": {"coarse", "mid"},
    "route_rq_addresses": {"mid", "rq_membership"},
    "select_entry_nodes": {
        "coarse",
        "mid",
        "rq_membership",
        "chunk",
    },
    "walk_graph_frontier": {"coarse", "mid", "chunk"},
    "drill_down_layer": {"coarse", "mid"},
    "jump_bridge": {
        "coarse",
        "mid",
        "rq_membership",
        "chunk",
    },
    "stop_and_collect_chunks": {"chunk"},
    "need_more_evidence": {
        "coarse",
        "mid",
        "rq_membership",
        "chunk",
    },
    "recall_chunks": {"rq_membership", "chunk"},
    "restore_context_package": {"chunk"},
    "build_context_package": {"chunk"},
    "verify_citations": {"chunk"},
    "repair_missing_citation": {
        "coarse",
        "mid",
        "rq_membership",
        "chunk",
    },
    "repair_concept_gap": {"coarse", "mid", "rq_membership"},
    "repair_bridge_gap": {
        "coarse",
        "mid",
        "rq_membership",
        "chunk",
    },
    "repair_structure_context": {"rq_membership", "chunk"},
}
TYPED_ACTION_EVIDENCE_LAYERS = {
    "coarse",
    "mid",
    "rq_membership",
    "chunk",
    "context_package",
    "citation",
}
QUALITY_GATE_PROTOCOL = {
    "protocol_version": QUALITY_GATE_PROTOCOL_VERSION,
    "subjects": ["graph", "retrieval", "agent", "context_package"],
    "fail_closed": True,
    "numeric_protocol": {
        "distance_domain": "finite_nonnegative",
        "relation_distance": "round6(-log(max(1e-6, round6(clamp(calibrated_strength, 1e-6, 1)))))",
        "projected_distance": "-log(max(1e-12, clamp(calibrated_strength, 0, 1)))",
        "zero_distance": "allowed_only_when_the_replayed_calibrated_strength_is_approximately_one",
        "absolute_tolerance": DISTANCE_REPLAY_ABS_TOLERANCE,
        "relative_tolerance": 0.0,
    },
    "graph_protocol_allowlist": {
        "relation": RELATION_PROTOCOL_VERSION,
        "edge_distance": EDGE_DISTANCE_PROTOCOL_VERSION,
        "edge_type_calibration": EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
        "edge_projection": EDGE_PROJECTION_PROTOCOL_VERSION,
        "projection_aggregation": PROJECTION_AGGREGATION_PROTOCOL_VERSION,
        "projection_calibration": PROJECTION_CALIBRATION_PROTOCOL_VERSION,
        "relation_support": RELATION_SUPPORT_PROTOCOL_VERSION,
        "rq_membership_role": RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION,
        "query_rq_seed": QUERY_RQ_SEED_PROTOCOL_VERSION,
        "chunk_facet_priority": CHUNK_FACET_PRIORITY_PROTOCOL_VERSION,
        "query_facet_posterior": QUERY_FACET_POSTERIOR_PROTOCOL_VERSION,
        "topk_ranking": TOPK_RANKING_PROTOCOL_VERSION,
        "dominance_replay": DOMINANCE_REPLAY_PROTOCOL_VERSION,
        "frontier_path_label_closure": (
            FRONTIER_PATH_LABEL_CLOSURE_PROTOCOL_VERSION
        ),
        "frontier_heap_enqueue": (
            FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
        ),
        "cycle_reward_replay": CYCLE_REWARD_REPLAY_PROTOCOL_VERSION,
        "convergence_replay": CONVERGENCE_REPLAY_PROTOCOL_VERSION,
        "typed_action_schema": TYPED_ACTION_SCHEMA_PROTOCOL_VERSION,
        "persisted_agent_replay": PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION,
        "agent_trace_stage": AGENT_TRACE_STAGE_PROTOCOL_VERSION,
    },
    "checks": {
        "graph": [
            "edge_distance_and_strength_replay",
            "edge_type_calibration_formula_distribution_and_hash_replay",
            "projected_edge_contribution_support_formula_and_calibration_replay",
            "rq_membership_role_replay",
            "relation_support_endpoint_closure",
            "projection_membership_scope_fk_replay",
        ],
        "retrieval": [
            "per_parent_candidate_coverage",
            "query_rq_seed_card_schema_hash_formula_score_and_order_replay",
            "chunk_facet_priority_card_schema_hash_authority_and_order_replay",
            "query_facet_posterior_prior_likelihood_observation_convergence_replay",
            "dominance_and_cycle_replay",
            "complete_local_gray_audit_recompute_and_zero_model_replay",
            "budget_and_convergence_replay",
            "frozen_topk_order_replay",
            "raw_dominance_event_replay",
            "frontier_candidate_path_label_edge_distance_closure",
            "cycle_reward_envelope_replay",
            "terminal_fact_convergence_replay",
        ],
        "agent": [
            "durable_agent_trace_sequence_replay",
            "typed_action_validation_then_execution",
            "evidence_evaluator_and_replan_accounting",
            "repair_budget_accounting",
            "citation_raw_provenance_replay",
            "persisted_typed_action_schema_and_validator_replay",
            "per_plan_executor_control_card_and_final_binding_replay",
            "persisted_agent_run_and_trace_row_replay",
            "strict_evidence_gate_payload_replay",
            "exact_completed_trace_stage_grammar",
            "run_trace_context_package_answer_reward_reciprocal_binding",
            "raw_retrieval_subgate_recompute",
        ],
        "context_package": [
            "raw_source_span_replay",
            "dedupe_and_role_count_replay",
            "trace_graph_path_scope",
            "token_hard_budget_replay",
            "postgres_chunk_and_source_snapshot_fact_replay",
        ],
    },
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_facet_ordered_window_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol_version": QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION,
            "continuous_normalized_substring": True,
            "single_token_match": "full_token_equality",
            "multi_token_match": "candidate_order_preserved",
            "multi_token_window": {
                "minimum_span": QUERY_FACET_ORDERED_WINDOW_MIN_SPAN,
                "candidate_token_multiplier": (
                    QUERY_FACET_ORDERED_WINDOW_TOKEN_MULTIPLIER
                ),
            },
            "unordered_document_token_subset_match": False,
            "lexical_score": None,
            "is_evidence": False,
            "citation_authority": False,
            "gray_zone_decision_authority": False,
            "model_call_budget": 0,
        }
    )


def chunk_facet_priority_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol_version": CHUNK_FACET_PRIORITY_PROTOCOL_VERSION,
            "facet_match_protocol_version": (
                QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION
            ),
            "facet_match_protocol_hash": (
                query_facet_ordered_window_protocol_hash()
            ),
            "candidate_order": [
                "uncovered_required_facet_count_asc",
                "covered_posterior_mass_desc",
                "effective_score_desc",
                "membership_role_tie_break_rank_asc",
                "chunk_id_asc",
            ],
            "query_facet_posterior_protocol_version": (
                QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
            ),
            "query_facet_posterior_protocol_hash": (
                query_facet_posterior_protocol_hash()
            ),
            "posterior_used_only_within_equal_uncovered_count": True,
            "per_parent_top_k_preserved": True,
            "merged_layer_top_k_preserved": True,
            "lexical_overlap_used_as_numeric_relevance": False,
            "gray_zone_decision_authority": False,
            "model_call_budget": 0,
        }
    )


def query_facet_posterior_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol_version": QUERY_FACET_POSTERIOR_PROTOCOL_VERSION,
            "checkpoints": list(QUERY_FACET_POSTERIOR_CHECKPOINTS),
            "prior": "uniform_required_facets_and_uniform_terms_within_facet",
            "likelihood": "beta_1_1_smoothed_boolean_match",
            "update": "normalized_prior_times_likelihood",
            "round_digits": QUERY_FACET_POSTERIOR_ROUND_DIGITS,
            "facet_match_protocol_version": (
                QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION
            ),
            "facet_match_protocol_hash": (
                query_facet_ordered_window_protocol_hash()
            ),
            "candidate_order": {
                "dense_entry_candidates": "dense_score_desc_chunk_id_asc",
                "merged_chunk_candidates": (
                    "effective_score_desc_chunk_id_asc_without_posterior"
                ),
            },
            "priority_role": (
                "same_uncovered_count_routing_tie_break_only"
            ),
            "llm_sample_budget": QUERY_FACET_POSTERIOR_LLM_SAMPLE_BUDGET,
            "is_evidence": False,
            "citation_authority": False,
            "graph_mutation_authority": False,
            "gray_zone_decision_authority": False,
        }
    )


QUALITY_GATE_PROTOCOL_HASH = stable_hash(QUALITY_GATE_PROTOCOL)
HASH_LENGTH = 64
TOLERANCE = 1e-6
ACTIVE_RELATION_EDGE_TYPES = set(EDGE_TYPE_CALIBRATION_EDGE_TYPES)
RQ_MEMBERSHIP_ROLES = {
    "noise_candidate",
    "outlier_member",
    "bridge_member",
    "low_confidence_member",
    "boundary_member",
    "primary_member",
    "fuzzy_member",
}
REPAIR_ACTION_TYPES = {
    "repair_missing_citation",
    "repair_structure_context",
    "repair_concept_gap",
    "repair_bridge_gap",
}


def edge_distance_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
            "formula": "distance=-log(max(epsilon,calibrated_strength))",
            "epsilon": 1e-6,
            "input_role": "edge_type_local_calibrated_strength",
        }
    )


def edge_type_calibration_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol": EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
            "edge_types": list(EDGE_TYPE_CALIBRATION_EDGE_TYPES),
            "candidate_scope": (
                "accepted_typed_gate_reverse_quota_undirected_dedupe"
            ),
            "quantile": "linear_interpolation_index_(n-1)*q",
            "quantile_round_digits": 6,
            "normalization": "strength_floor_plus_winsorized_minmax",
            "calibrated_strength_round_digits": 6,
            "fallback": "identity_sparse_or_degenerate",
            "default_params": EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS,
            "raw_strength_role": "pre_calibration_audit_only",
            "weight_role": "calibrated_strength_compatibility_copy",
            "distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
        }
    )


def edge_projection_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
            "source_algorithm": "membership_weighted_bottom_edge_projection",
            "support": {
                "existence_gate": (
                    "positive_membership_bottom_chunk_edges_only"
                ),
                "complete_chunk_edge_ids": True,
                "membership_contribution": (
                    "undirected_business_endpoint_assignment_max_product_v2"
                ),
                "endpoint_storage_orientation_is_fact": False,
                "canonical_assignment_protocol": ASSIGNMENT_PROTOCOL_VERSION,
                "chunk_business_key_protocol": (
                    CHUNK_BUSINESS_KEY_PROTOCOL_VERSION
                ),
            },
            "aggregation": {
                "protocol_version": PROJECTION_AGGREGATION_PROTOCOL_VERSION,
                "raw_distance": (
                    "q0.15_bottom_distance_divided_by_1_plus_log1p_membership_mass"
                ),
                "raw_strength": "exp_negative_raw_distance",
            },
            "calibration": {
                "protocol_version": PROJECTION_CALIBRATION_PROTOCOL_VERSION,
                "scope": "layer_plus_projected_edge_type",
                "params": PROJECTION_CALIBRATION_PARAMS,
                "degenerate_fallback": "identity_with_explicit_reason",
            },
            "edge_type": {
                "selection": "dominant_membership_mass_bottom_type",
                "mapping": PROJECTED_EDGE_TYPE_BY_BOTTOM_TYPE,
            },
            "gray_predicates": {
                "semantic_uncertainty": (
                    EDGE_SEMANTIC_UNCERTAINTY_PROTOCOL_VERSION
                ),
                "rq_boundary": RQ_BOUNDARY_PROTOCOL_VERSION,
                "projection": PROJECTED_GRAY_PREDICATE_PROTOCOL_VERSION,
                "semantics_changed": False,
                "model_call_budget": 0,
            },
        }
    )


def rq_membership_role_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol_version": RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION,
            "inputs": [
                "membership_score",
                "rank",
                "normalized_entropy",
                "residual_norm",
                "residual_confidence_gamma",
                "boundary_probability_margin",
                "boundary_distance",
                "primary_leaf",
                "bridge_support",
            ],
            "thresholds": RQ_MEMBERSHIP_ROLE_THRESHOLDS,
            "precedence": [
                "noise_candidate",
                "outlier_member",
                "bridge_member",
                "low_confidence_member",
                "boundary_member",
                "primary_member",
                "fuzzy_member",
            ],
            "model_call_budget": 0,
        }
    )


def query_rq_seed_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol_version": QUERY_RQ_SEED_PROTOCOL_VERSION,
            "rq_stage": {
                "explicit_query_relevance_precedence": True,
                "selected_mid_route_fallback_only_when_missing": True,
                "forced_target_override_is_query_relevance": False,
                "zero_query_relevance_preserved": True,
            },
            "chunk_seed": {
                "component_weights": (
                    QUERY_RQ_CHUNK_SEED_COMPONENT_WEIGHTS
                ),
                "no_query_rq_fuzzy_overlap": (
                    "sqrt(query_prefix_score*chunk_membership_score)"
                ),
                "query_rq_present_score": (
                    "0.75*rq_candidate_score+0.25*fuzzy_overlap"
                ),
                "membership_role_tie_break": (
                    QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK
                ),
                "fallback_without_rq_membership": {
                    "score_source": (
                        "mid_support_without_rq_membership"
                    ),
                    "rq_l3_prefix_id": None,
                    "component_weights": (
                        QUERY_RQ_MID_SUPPORT_FALLBACK_COMPONENT_WEIGHTS
                    ),
                    "rq_path_and_membership_components": 0.0,
                },
                "candidate_card_coverage": (
                    "every_chunk_by_mid_candidate_has_nonempty_cards"
                ),
                "candidate_score_replay": (
                    "max(seed_card.effective_score)"
                ),
                "mid_support_baseline_may_mask_rq_seed": False,
                "hard_lcp_used_as_score": False,
                "node_weight_used_as_query_relevance": False,
            },
            "query_facet_packet_is_evidence": False,
            "gray_zone_decision_authority": False,
            "model_call_budget": 0,
        }
    )


def typed_action_schema_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol_version": TYPED_ACTION_SCHEMA_PROTOCOL_VERSION,
            "allowed_actions": sorted(ALLOWED_TYPED_ACTIONS),
            "required_actions": REQUIRED_TYPED_ACTIONS,
            "required_fields": sorted(TYPED_ACTION_REQUIRED_FIELDS),
            "expected_evidence_fields": sorted(EXPECTED_EVIDENCE_FIELDS),
            "stop_condition_fields": sorted(STOP_CONDITION_FIELDS),
        }
    )


def typed_action_schema_identity_supported(validation: Mapping[str, Any]) -> bool:
    version = str(
        validation.get("typed_action_schema_protocol_version") or ""
    )
    protocol_hash = str(
        validation.get("typed_action_schema_protocol_hash") or ""
    )
    if version == TYPED_ACTION_SCHEMA_PROTOCOL_VERSION:
        return protocol_hash == typed_action_schema_protocol_hash()
    return (
        protocol_hash
        in HISTORICAL_TYPED_ACTION_SCHEMA_PROTOCOL_HASHES.get(
            version,
            frozenset(),
        )
    )


def typed_action_required_actions_for_identity(
    validation: Mapping[str, Any],
) -> tuple[str, ...]:
    protocol_hash = str(
        validation.get("typed_action_schema_protocol_hash") or ""
    )
    return HISTORICAL_TYPED_ACTION_SCHEMA_REQUIRED_ACTIONS_BY_HASH.get(
        protocol_hash,
        tuple(REQUIRED_TYPED_ACTIONS),
    )


def _relation_calibration_params_hash(params: Mapping[str, Any]) -> str:
    return stable_hash(dict(params))


def _relation_calibration_config_hash(params: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "graph_operating_point_protocol": (
                GRAPH_OPERATING_POINT_PROTOCOL_VERSION
            ),
            "edge_type_calibration_protocol": (
                EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION
            ),
            "edge_type_calibration_protocol_hash": (
                edge_type_calibration_protocol_hash()
            ),
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
            "edge_distance_protocol_hash": edge_distance_protocol_hash(),
            "calibration_params": dict(params),
        }
    )


def _is_hash(value: Any) -> bool:
    text = str(value or "")
    return len(text) == HASH_LENGTH and all(character in "0123456789abcdef" for character in text.lower())


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if _finite(value) else default


def _strict_numeric_match(observed: Any, expected: Any) -> bool:
    return (
        _finite(observed)
        and _finite(expected)
        and math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=DISTANCE_REPLAY_ABS_TOLERANCE,
        )
    )


def _strict_numeric_mapping_match(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        return False
    if set(observed) != set(expected):
        return False
    return all(
        _strict_numeric_match(observed[key], expected[key])
        for key in expected
    )


def _optional_numeric_match(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is None and expected is None
    return _strict_numeric_match(observed, expected)


def _normalized_relation_strength(value: Any) -> float:
    return round(max(1e-6, min(1.0, float(value))), 6)


def _linear_quantile(
    values: Sequence[float],
    quantile: float,
    *,
    round_digits: int | None,
) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    fraction = position - lower_index
    result = ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )
    return round(result, round_digits) if round_digits is not None else result


def _normalized_relation_params(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or set(value) != set(
        EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS
    ):
        return None
    if any(not _finite(value.get(key)) for key in EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS):
        return None
    params = {
        key: round(float(value[key]), 6)
        for key in EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS
    }
    if any(not _strict_numeric_match(value[key], params[key]) for key in params):
        return None
    if not 0.0 <= params["lower_quantile"] <= 0.25:
        return None
    if not 0.75 <= params["upper_quantile"] <= 1.0:
        return None
    if params["lower_quantile"] >= params["upper_quantile"]:
        return None
    if not 0.01 <= params["min_span"] <= 0.5:
        return None
    if not 1e-6 <= params["strength_floor"] <= 0.25:
        return None
    return params


def _unique_strings(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    return [str(value) for value in values if str(value)]


def _distribution(values: Iterable[Any]) -> dict[str, int | float | None]:
    numeric = [float(value) for value in values if _finite(value)]
    if not numeric:
        return {"count": 0, "min": None, "max": None, "mean": None, "population_std": None}
    mean = sum(numeric) / len(numeric)
    population_std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / len(numeric))
    return {
        "count": len(numeric),
        "min": round(min(numeric), 6),
        "max": round(max(numeric), 6),
        "mean": round(mean, 6),
        "population_std": round(population_std, 6),
    }


def _distribution_matches(claimed: Any, actual: Mapping[str, Any]) -> bool:
    if not isinstance(claimed, Mapping):
        return False
    for key in ("count", "min", "max", "mean", "population_std"):
        if key not in claimed:
            continue
        expected = actual.get(key)
        observed = claimed.get(key)
        if expected is None or observed is None:
            if expected is not observed:
                return False
        elif key == "count":
            if int(observed) != int(expected):
                return False
        elif not _finite(observed) or not math.isclose(float(observed), float(expected), abs_tol=TOLERANCE):
            return False
    return "count" in claimed


def _distance_from_strength(strength: Any) -> float:
    normalized = _normalized_relation_strength(strength)
    distance = -math.log(max(1e-6, normalized))
    return 0.0 if abs(distance) < 1e-9 else round(distance, 6)


def _raw_projected_distance(strength: Any) -> float:
    return -math.log(max(1e-12, min(1.0, float(strength))))


def _distance_replay_matches(observed: Any, expected: Any) -> bool:
    return _strict_numeric_match(observed, expected)


class _Audit:
    def __init__(self, subject: str) -> None:
        self.subject = subject
        self.checks: dict[str, bool] = {}
        self.findings: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    def check(
        self,
        name: str,
        condition: bool,
        *,
        code: str | None = None,
        scope: str | None = None,
        expected: Any = None,
        actual: Any = None,
        message: str | None = None,
    ) -> bool:
        passed = bool(condition)
        self.checks[name] = self.checks.get(name, True) and passed
        if not passed:
            self.findings.append(
                {
                    "code": code or name,
                    "scope": scope or self.subject,
                    "message": message or name.replace("_", " "),
                    "expected": expected,
                    "actual": actual,
                }
            )
        return passed

    def finish(self) -> dict[str, Any]:
        return {
            "protocol_version": QUALITY_GATE_PROTOCOL_VERSION,
            "protocol_hash": QUALITY_GATE_PROTOCOL_HASH,
            "subject": self.subject,
            "fail_closed": True,
            "checks": self.checks,
            "metrics": self.metrics,
            "findings": self.findings,
            "finding_count": len(self.findings),
            "pass": not self.findings and bool(self.checks) and all(self.checks.values()),
        }


def _check_declared_count(audit: _Audit, snapshot: Mapping[str, Any], name: str, rows: Sequence[Any]) -> None:
    declared = (snapshot.get("declared_counts") or {}).get(name)
    audit.check(
        f"declared_{name}_matches_rows",
        type(declared) is int and declared == len(rows),
        code="declared_count_mismatch",
        scope=name,
        expected=len(rows),
        actual=declared,
    )


def _membership_role_replay(
    evaluation: Mapping[str, Any],
) -> tuple[str | None, list[str]]:
    inputs = evaluation.get("inputs") or {}
    required_inputs = {
        "membership_score",
        "membership_entropy",
        "residual_norm",
        "gamma",
        "boundary_probability_margin",
        "boundary_distance",
        "residual_outlier_threshold",
        "rank",
        "is_primary_leaf",
        "is_bridge_chunk",
    }
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != required_inputs
        or dict(evaluation.get("thresholds") or {})
        != RQ_MEMBERSHIP_ROLE_THRESHOLDS
        or not all(
            _finite(inputs.get(field))
            for field in (
                "membership_score",
                "membership_entropy",
                "residual_norm",
                "gamma",
                "boundary_probability_margin",
                "boundary_distance",
                "residual_outlier_threshold",
            )
        )
        or type(inputs.get("rank")) is not int
        or int(inputs["rank"]) <= 0
        or type(inputs.get("is_primary_leaf")) is not bool
        or type(inputs.get("is_bridge_chunk")) is not bool
        or not 0.0 <= float(inputs["membership_score"]) <= 1.0
        or not 0.0 <= float(inputs["membership_entropy"]) <= 1.0
        or float(inputs["residual_norm"]) < 0.0
        or not 0.0 <= float(inputs["gamma"]) <= 1.0
        or not 0.0 <= float(inputs["boundary_probability_margin"]) <= 1.0
        or float(inputs["boundary_distance"]) < 0.0
        or float(inputs["residual_outlier_threshold"]) < 0.0
    ):
        return None, []
    thresholds = RQ_MEMBERSHIP_ROLE_THRESHOLDS
    flags = {
        "noise_candidate": _float(inputs["membership_score"]) <= _float(thresholds["noise_membership_score_max"]),
        "outlier_member": (
            _float(inputs["residual_norm"]) >= _float(inputs["residual_outlier_threshold"])
            and _float(inputs["gamma"]) <= _float(thresholds["outlier_gamma_max"])
        ),
        "bridge_member": bool(inputs["is_bridge_chunk"]),
        "low_confidence_member": (
            _float(inputs["gamma"]) <= _float(thresholds["low_confidence_gamma_max"])
            or _float(inputs["membership_score"]) <= _float(thresholds["low_confidence_membership_score_max"])
        ),
        "boundary_member": (
            _float(inputs["membership_entropy"]) >= _float(thresholds["boundary_entropy_min"])
            or _float(inputs["boundary_probability_margin"]) <= _float(thresholds["boundary_probability_margin_max"])
            or _float(inputs["boundary_distance"]) <= _float(thresholds["boundary_distance_max"])
        ),
        "primary_member": bool(inputs["is_primary_leaf"]),
        "fuzzy_member": not bool(inputs["is_primary_leaf"]) or int(inputs["rank"]) > 1,
    }
    precedence = (
        "noise_candidate",
        "outlier_member",
        "bridge_member",
        "low_confidence_member",
        "boundary_member",
        "primary_member",
        "fuzzy_member",
    )
    matched = [role for role in precedence if flags[role]]
    return (matched[0] if matched else None), matched


def _expected_membership_role(evaluation: Mapping[str, Any]) -> str | None:
    return _membership_role_replay(evaluation)[0]


def _audit_relation_calibration(
    audit: _Audit,
    edges: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> None:
    stats_by_type = diagnostics.get("stats_by_edge_type")
    stats_by_type = stats_by_type if isinstance(stats_by_type, Mapping) else {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[str(edge.get("edge_type") or "")].append(edge)

    params = _normalized_relation_params(diagnostics.get("params"))
    expected_protocol_hash = edge_type_calibration_protocol_hash()
    expected_distance_hash = edge_distance_protocol_hash()
    expected_params_hash = (
        _relation_calibration_params_hash(params) if params is not None else None
    )
    expected_config_hash = (
        _relation_calibration_config_hash(params) if params is not None else None
    )
    audit.check(
        "relation_calibration_local_protocol_bound",
        params is not None
        and diagnostics.get("protocol_version")
        == EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION
        and diagnostics.get("protocol_hash") == expected_protocol_hash
        and diagnostics.get("calibration_params_hash") == expected_params_hash
        and diagnostics.get("edge_type_calibration_config_hash")
        == expected_config_hash
        and diagnostics.get("edge_distance_protocol_version")
        == EDGE_DISTANCE_PROTOCOL_VERSION
        and diagnostics.get("edge_distance_protocol_hash")
        == expected_distance_hash
        and diagnostics.get("candidate_scope")
        == "accepted_typed_gate_reverse_quota_undirected_dedupe"
        and diagnostics.get("cross_type_raw_comparison_allowed") is False
        and set(stats_by_type) == set(EDGE_TYPE_CALIBRATION_EDGE_TYPES),
        code="relation_calibration_protocol_or_config_mismatch",
        expected={
            "protocol_version": EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
            "protocol_hash": expected_protocol_hash,
            "edge_distance_protocol_version": EDGE_DISTANCE_PROTOCOL_VERSION,
            "edge_distance_protocol_hash": expected_distance_hash,
            "params_hash": expected_params_hash,
            "config_hash": expected_config_hash,
            "edge_types": sorted(EDGE_TYPE_CALIBRATION_EDGE_TYPES),
        },
        actual={
            "protocol_version": diagnostics.get("protocol_version"),
            "protocol_hash": diagnostics.get("protocol_hash"),
            "edge_distance_protocol_version": diagnostics.get(
                "edge_distance_protocol_version"
            ),
            "edge_distance_protocol_hash": diagnostics.get(
                "edge_distance_protocol_hash"
            ),
            "params_hash": diagnostics.get("calibration_params_hash"),
            "config_hash": diagnostics.get(
                "edge_type_calibration_config_hash"
            ),
            "edge_types": sorted(stats_by_type),
        },
    )
    if params is None:
        return

    graph_state_hashes = {
        str(edge.get("graph_state_hash") or "") for edge in edges
    }
    audit.check(
        "relation_rows_share_graph_state_hash",
        len(graph_state_hashes) == 1
        and all(_is_hash(value) for value in graph_state_hashes),
        code="relation_graph_state_hash_mismatch",
        actual=sorted(graph_state_hashes),
    )

    for edge_type in EDGE_TYPE_CALIBRATION_EDGE_TYPES:
        rows = sorted(
            grouped.get(edge_type, []),
            key=lambda row: (
                _float(row.get("raw_strength")),
                str(row.get("source_chunk_id") or ""),
                str(row.get("target_chunk_id") or ""),
            ),
        )
        stats = stats_by_type.get(edge_type)
        audit.check(
            f"relation_{edge_type}_calibration_stats_present",
            isinstance(stats, Mapping),
            code="relation_calibration_stats_missing",
            scope=edge_type,
        )
        if not isinstance(stats, Mapping):
            continue

        raw_values_valid = all(
            _finite(row.get("raw_strength"))
            and 0.0 < float(row.get("raw_strength")) <= 1.0
            for row in rows
        )
        audit.check(
            f"relation_{edge_type}_raw_calibration_inputs_valid",
            raw_values_valid,
            code="relation_calibration_raw_input_invalid",
            scope=edge_type,
        )
        raw_values = [
            _normalized_relation_strength(row.get("raw_strength"))
            for row in rows
            if _finite(row.get("raw_strength"))
        ]
        lower_value = _linear_quantile(
            raw_values, params["lower_quantile"], round_digits=6
        )
        upper_value = _linear_quantile(
            raw_values, params["upper_quantile"], round_digits=6
        )
        quantile_span = (
            round(float(upper_value) - float(lower_value), 6)
            if lower_value is not None and upper_value is not None
            else None
        )
        if not raw_values:
            fallback = "no_edges"
        elif len(raw_values) < 2:
            fallback = "insufficient_type_sample"
        elif quantile_span is None or quantile_span < params["min_span"]:
            fallback = "degenerate_quantile_span"
        else:
            fallback = None

        expected_calibrated: list[float] = []
        for raw_strength in raw_values:
            if fallback:
                calibrated_strength = raw_strength
            else:
                unit_value = (raw_strength - float(lower_value)) / max(
                    float(upper_value) - float(lower_value), 1e-12
                )
                unit_value = max(0.0, min(1.0, unit_value))
                calibrated_strength = _normalized_relation_strength(
                    params["strength_floor"]
                    + (1.0 - params["strength_floor"]) * unit_value
                )
            expected_calibrated.append(calibrated_strength)
        expected_distances = [
            _distance_from_strength(value) for value in expected_calibrated
        ]
        expected_stats_hash = stable_hash(
            {
                "protocol": EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
                "protocol_hash": expected_protocol_hash,
                "edge_type": edge_type,
                "params": params,
                "raw_strengths": raw_values,
                "fallback": fallback,
            }
        )
        expected_stats_fields = {
            "edge_type": edge_type,
            "protocol_version": EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
            "protocol_hash": expected_protocol_hash,
            "calibration_params_hash": expected_params_hash,
            "edge_type_calibration_config_hash": expected_config_hash,
            "stats_hash": expected_stats_hash,
            "params": params,
            "sample_count": len(rows),
            "lower_quantile_value": lower_value,
            "upper_quantile_value": upper_value,
            "effective_lower_bound": lower_value,
            "effective_upper_bound": upper_value,
            "quantile_span": quantile_span,
            "fallback": fallback,
            "calibration_applied": fallback is None,
            "monotonic_violation_count": 0,
            "raw_strength_distribution": _distribution(raw_values),
            "calibrated_strength_distribution": _distribution(
                expected_calibrated
            ),
            "distance_distribution": _distribution(expected_distances),
            "cross_type_raw_comparison_allowed": False,
        }
        stats_fields_match = all(
            (
                _optional_numeric_match(stats.get(field), expected)
                if field
                in {
                    "lower_quantile_value",
                    "upper_quantile_value",
                    "effective_lower_bound",
                    "effective_upper_bound",
                    "quantile_span",
                }
                else stats.get(field) == expected
            )
            for field, expected in expected_stats_fields.items()
        )
        audit.check(
            f"relation_{edge_type}_calibration_stats_replayed",
            stats_fields_match,
            code="relation_calibration_stats_or_formula_mismatch",
            scope=edge_type,
            expected=expected_stats_fields,
            actual=dict(stats),
        )

        for row, expected_weight, expected_distance in zip(
            rows, expected_calibrated, expected_distances
        ):
            row_id = str(row.get("id") or "missing")
            features = row.get("features")
            features = features if isinstance(features, Mapping) else {}
            summary = row.get("raw_strength_summary")
            summary = summary if isinstance(summary, Mapping) else {}
            normalization = row.get("normalization_stats")
            normalization = (
                normalization if isinstance(normalization, Mapping) else {}
            )
            nested_stats = normalization.get("edge_type_calibration")
            feature_normalization = features.get("normalization_stats")
            feature_normalization = (
                feature_normalization
                if isinstance(feature_normalization, Mapping)
                else {}
            )
            feature_nested = feature_normalization.get(
                "edge_type_calibration"
            )
            audit.check(
                f"relation_{row_id}_calibrated_strength_formula_replayed",
                _strict_numeric_match(row.get("weight"), expected_weight)
                and _strict_numeric_match(row.get("distance"), expected_distance),
                code="relation_calibrated_strength_formula_mismatch",
                scope=row_id,
                expected={
                    "weight": expected_weight,
                    "distance": expected_distance,
                },
                actual={
                    "weight": row.get("weight"),
                    "distance": row.get("distance"),
                },
            )
            audit.check(
                f"relation_{row_id}_local_protocol_and_stats_bound",
                row.get("protocol_version") == RELATION_PROTOCOL_VERSION
                and row.get("source_algorithm") == "dense_embedding"
                and row.get("edge_distance_protocol_hash")
                == expected_distance_hash
                and features.get("calibrated_strength") == expected_weight
                and features.get("edge_type_calibration_protocol_version")
                == EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION
                and features.get("edge_type_calibration_protocol_hash")
                == expected_protocol_hash
                and features.get("calibration_params_hash")
                == expected_params_hash
                and features.get("edge_type_calibration_config_hash")
                == expected_config_hash
                and features.get("edge_type_calibration_stats_hash")
                == expected_stats_hash
                and features.get("edge_distance_protocol_version")
                == EDGE_DISTANCE_PROTOCOL_VERSION
                and features.get("edge_distance_protocol_hash")
                == expected_distance_hash
                and isinstance(nested_stats, Mapping)
                and dict(nested_stats) == dict(stats)
                and isinstance(feature_nested, Mapping)
                and dict(feature_nested) == dict(stats)
                and summary.get("edge_distance_protocol")
                == EDGE_DISTANCE_PROTOCOL_VERSION
                and summary.get("edge_type_calibration_protocol")
                == EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION
                and summary.get("calibration_params_hash")
                == expected_params_hash
                and summary.get("edge_type_calibration_config_hash")
                == expected_config_hash
                and summary.get("edge_type_calibration_stats_hash")
                == expected_stats_hash,
                code="relation_row_protocol_or_stats_binding_mismatch",
                scope=row_id,
            )


def _audit_projected_edge_support(
    audit: _Audit,
    *,
    layer: str,
    edge: Mapping[str, Any],
    relation_by_id: Mapping[str, Mapping[str, Any]],
    chunk_business_keys: Mapping[str, str],
    membership_by_scope_chunk: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
    rq_prefix_scope_keys: Mapping[str, str],
    mid_edge_ids: set[str],
) -> None:
    edge_id = str(edge.get("id") or "missing")
    scope = f"{layer}.{edge_id}"
    support_edge_ids = _unique_strings(edge.get("support_chunk_edge_ids"))
    support_chunk_ids = _unique_strings(edge.get("support_chunk_ids"))
    diagnostics = edge.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    summary = edge.get("raw_strength_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    contributions = diagnostics.get("support_contributions")
    contributions = contributions if isinstance(contributions, list) else []
    contribution_ids = [
        str(item.get("bottom_chunk_edge_id") or "")
        for item in contributions
        if isinstance(item, Mapping)
    ]
    support_relations = [
        relation_by_id.get(edge_id_value) for edge_id_value in support_edge_ids
    ]
    audit.check(
        f"{layer}_projection_{edge_id}_support_foreign_keys_closed",
        bool(support_edge_ids)
        and len(support_edge_ids) == len(set(support_edge_ids))
        and all(row is not None for row in support_relations)
        and contribution_ids == support_edge_ids
        and len(contributions) == len(support_edge_ids),
        code="projected_edge_support_or_protocol_invalid",
        scope=scope,
        expected={"relation_edge_ids": sorted(relation_by_id)},
        actual={
            "support_chunk_edge_ids": support_edge_ids,
            "contribution_edge_ids": contribution_ids,
        },
    )
    if (
        not support_edge_ids
        or any(row is None for row in support_relations)
        or len(contributions) != len(support_edge_ids)
    ):
        return

    source_scope_key = str(edge.get("source_scope_key") or "")
    target_scope_key = str(edge.get("target_scope_key") or "")
    source_rq_prefix_id = str(edge.get("source_rq_prefix_id") or "")
    target_rq_prefix_id = str(edge.get("target_rq_prefix_id") or "")
    scope_identity_valid = (
        bool(source_rq_prefix_id)
        and bool(target_rq_prefix_id)
        and source_rq_prefix_id != target_rq_prefix_id
        and rq_prefix_scope_keys.get(source_rq_prefix_id)
        == source_scope_key
        and rq_prefix_scope_keys.get(target_rq_prefix_id)
        == target_scope_key
    )
    audit.check(
        f"{layer}_projection_{edge_id}_scope_membership_identity_bound",
        scope_identity_valid,
        code="projected_edge_membership_scope_binding_invalid",
        scope=scope,
        expected={
            "source_scope_key": rq_prefix_scope_keys.get(
                source_rq_prefix_id
            ),
            "target_scope_key": rq_prefix_scope_keys.get(
                target_rq_prefix_id
            ),
        },
        actual={
            "source_scope_key": source_scope_key,
            "target_scope_key": target_scope_key,
            "source_rq_prefix_id": source_rq_prefix_id,
            "target_rq_prefix_id": target_rq_prefix_id,
        },
    )
    canonical_contributions: list[tuple[str, dict[str, Any]]] = []
    contribution_distances: list[float] = []
    membership_products: list[float] = []
    bottom_type_mass: defaultdict[str, float] = defaultdict(float)
    expected_chunk_ids: set[str] = set()
    contribution_rows_valid = True
    for contribution, relation in zip(contributions, support_relations):
        if not isinstance(contribution, Mapping) or relation is None:
            contribution_rows_valid = False
            continue
        relation_source = str(relation.get("source_chunk_id") or "")
        relation_target = str(relation.get("target_chunk_id") or "")
        contribution_source = str(contribution.get("source_chunk_id") or "")
        contribution_target = str(contribution.get("target_chunk_id") or "")
        source_business_key = str(
            chunk_business_keys.get(contribution_source) or ""
        )
        target_business_key = str(
            chunk_business_keys.get(contribution_target) or ""
        )
        source_score = contribution.get("source_membership_score")
        target_score = contribution.get("target_membership_score")
        persisted_source_membership = membership_by_scope_chunk.get(
            (source_rq_prefix_id, contribution_source)
        )
        persisted_target_membership = membership_by_scope_chunk.get(
            (target_rq_prefix_id, contribution_target)
        )
        persisted_scores_valid = (
            persisted_source_membership is not None
            and persisted_target_membership is not None
            and _strict_numeric_match(
                source_score,
                persisted_source_membership.get("membership_score"),
            )
            and _strict_numeric_match(
                target_score,
                persisted_target_membership.get("membership_score"),
            )
        )
        membership_product = contribution.get("membership_product")
        product_valid = (
            _finite(source_score)
            and 0.0 < float(source_score) <= 1.0
            and _finite(target_score)
            and 0.0 < float(target_score) <= 1.0
            and _finite(membership_product)
            and float(membership_product) > 0.0
            and _strict_numeric_match(
                membership_product, float(source_score) * float(target_score)
            )
            and persisted_scores_valid
        )
        relation_distance = relation.get("distance")
        relation_weight = relation.get("weight")
        relation_numbers_valid = (
            _finite(relation_distance)
            and float(relation_distance) >= 0.0
            and _finite(relation_weight)
            and 0.0 < float(relation_weight) <= 1.0
        )
        bottom_fact = {
            "endpoint_chunk_business_keys": sorted(
                (source_business_key, target_business_key)
            ),
            "edge_type": str(relation.get("edge_type") or ""),
            "distance": (
                float(relation_distance) if _finite(relation_distance) else None
            ),
            "calibrated_strength": (
                float(relation_weight) if _finite(relation_weight) else None
            ),
            "protocol_version": str(relation.get("protocol_version") or ""),
            "graph_state_hash": str(relation.get("graph_state_hash") or ""),
            "edge_distance_protocol_hash": str(
                relation.get("edge_distance_protocol_hash") or ""
            ),
        }
        expected_bottom_hash = stable_hash(bottom_fact)
        row_valid = (
            {contribution_source, contribution_target}
            == {relation_source, relation_target}
            and bool(source_business_key)
            and bool(target_business_key)
            and contribution.get("bottom_edge_type")
            == relation.get("edge_type")
            and relation_numbers_valid
            and _strict_numeric_match(
                contribution.get("bottom_distance"), relation.get("distance")
            )
            and product_valid
            and contribution.get("orientation")
            == "source_scope_to_target_scope"
            and contribution.get("assignment_protocol_version")
            == ASSIGNMENT_PROTOCOL_VERSION
            and contribution.get("bottom_edge_fact_hash")
            == expected_bottom_hash
            and scope_identity_valid
        )
        contribution_rows_valid = contribution_rows_valid and row_valid
        if not product_valid or not relation_numbers_valid:
            continue
        product = float(membership_product)
        distance = float(relation.get("distance"))
        contribution_distances.append(distance)
        membership_products.append(product)
        bottom_type_mass[str(relation.get("edge_type") or "")] += product
        expected_chunk_ids.update((contribution_source, contribution_target))
        endpoint_scope_memberships = sorted(
            [
                {
                    "scope_key": source_scope_key,
                    "chunk_business_key": source_business_key,
                    "membership_score": float(source_score),
                },
                {
                    "scope_key": target_scope_key,
                    "chunk_business_key": target_business_key,
                    "membership_score": float(target_score),
                },
            ],
            key=lambda item: (
                str(item["scope_key"]),
                str(item["chunk_business_key"]),
            ),
        )
        canonical_contributions.append(
            (
                str(contribution.get("bottom_chunk_edge_id") or ""),
                {
                "bottom_edge_fact_hash": expected_bottom_hash,
                "endpoint_scope_memberships": endpoint_scope_memberships,
                "membership_product": product,
                "assignment_protocol_version": ASSIGNMENT_PROTOCOL_VERSION,
                },
            )
        )

    canonical_contributions = sorted(
        canonical_contributions,
        key=lambda pair: (
            str(pair[1]["bottom_edge_fact_hash"]),
            pair[0],
        ),
    )
    expected_contribution_ids = [pair[0] for pair in canonical_contributions]
    canonical_contribution_payload = [pair[1] for pair in canonical_contributions]
    support_mass = sum(membership_products)
    q15_distance = (
        _linear_quantile(
            contribution_distances, 0.15, round_digits=None
        )
        if contribution_distances
        else None
    )
    projected_distance_raw = (
        float(q15_distance) / (1.0 + math.log1p(support_mass))
        if q15_distance is not None and support_mass > 0.0
        else None
    )
    projected_strength_raw = (
        math.exp(-projected_distance_raw)
        if projected_distance_raw is not None
        else None
    )
    dominant_bottom_type = (
        min(
            bottom_type_mass,
            key=lambda edge_type: (-bottom_type_mass[edge_type], edge_type),
        )
        if bottom_type_mass
        else None
    )
    expected_edge_type = (
        PROJECTED_EDGE_TYPE_BY_BOTTOM_TYPE.get(dominant_bottom_type)
        if dominant_bottom_type
        else None
    )
    expected_contribution_hash = stable_hash(canonical_contribution_payload)
    expected_chunk_order = sorted(
        expected_chunk_ids,
        key=lambda chunk_id: (
            str(chunk_business_keys.get(chunk_id) or ""),
            chunk_id,
        ),
    )
    audit.check(
        f"{layer}_projection_{edge_id}_contribution_rows_replayed",
        contribution_rows_valid
        and bool(source_scope_key)
        and bool(target_scope_key)
        and contribution_ids == expected_contribution_ids
        and support_edge_ids == expected_contribution_ids
        and support_chunk_ids == expected_chunk_order
        and diagnostics.get("support_contribution_count")
        == len(contributions)
        and diagnostics.get("support_chunk_edge_count")
        == len(support_edge_ids)
        and _strict_numeric_match(
            diagnostics.get("support_membership_mass"), support_mass
        )
        and diagnostics.get("support_edge_types")
        == dict(Counter(row.get("edge_type") for row in support_relations))
        and diagnostics.get("dominant_bottom_edge_type")
        == dominant_bottom_type
        and diagnostics.get("contribution_facts_hash")
        == expected_contribution_hash
        and summary.get("contribution_facts_hash")
        == expected_contribution_hash,
        code="projected_edge_contribution_support_mismatch",
        scope=scope,
    )
    raw_aggregation_checks = {
        "q15_distance_present": q15_distance is not None,
        "projected_distance_present": projected_distance_raw is not None,
        "projected_strength_present": projected_strength_raw is not None,
        "edge_type_matches": edge.get("edge_type") == expected_edge_type,
        "projected_distance_matches": _strict_numeric_match(
            edge.get("projected_distance_raw"), projected_distance_raw
        ),
        "projected_strength_matches": _strict_numeric_match(
            edge.get("projected_strength_raw"), projected_strength_raw
        ),
        "q15_distance_matches": _strict_numeric_match(
            summary.get("q15_bottom_distance"), q15_distance
        ),
        "support_mass_matches": _strict_numeric_match(
            summary.get("support_membership_mass"), support_mass
        ),
        "support_edge_count_matches": (
            summary.get("support_chunk_edge_count") == len(support_edge_ids)
        ),
        "bottom_distance_distribution_matches": (
            summary.get("bottom_distance_distribution")
            == _distribution(contribution_distances)
        ),
        "membership_product_distribution_matches": (
            summary.get("membership_product_distribution")
            == _distribution(membership_products)
        ),
        "dominant_bottom_type_matches": (
            summary.get("dominant_bottom_edge_type") == dominant_bottom_type
        ),
        "bottom_type_mass_matches": (
            _strict_numeric_mapping_match(
                summary.get("bottom_edge_type_membership_mass"),
                dict(sorted(bottom_type_mass.items())),
            )
        ),
        "distance_protocol_matches": (
            summary.get("edge_distance_protocol")
            == EDGE_DISTANCE_PROTOCOL_VERSION
        ),
    }
    audit.check(
        f"{layer}_projection_{edge_id}_raw_aggregation_replayed",
        all(raw_aggregation_checks.values()),
        code="projected_edge_raw_aggregation_formula_mismatch",
        scope=scope,
        actual={
            "failed_checks": sorted(
                name
                for name, passed in raw_aggregation_checks.items()
                if not passed
            )
        },
    )
    if layer == "coarse":
        support_mid_edge_ids = _unique_strings(edge.get("support_mid_edge_ids"))
        audit.check(
            f"{layer}_projection_{edge_id}_mid_support_foreign_keys_closed",
            len(support_mid_edge_ids) == len(set(support_mid_edge_ids))
            and set(support_mid_edge_ids).issubset(mid_edge_ids)
            and diagnostics.get("support_mid_edge_count")
            == len(support_mid_edge_ids),
            code="projected_mid_edge_support_invalid",
            scope=scope,
        )


def _audit_projection_layer(
    audit: _Audit,
    layer: str,
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> None:
    stats_by_type = calibration.get("stats_by_edge_type")
    stats_by_type = stats_by_type if isinstance(stats_by_type, Mapping) else {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("edge_type") or "")].append(row)
    expected_projection_hash = edge_projection_protocol_hash()
    audit.check(
        f"{layer}_projection_local_protocol_bound",
        calibration.get("protocol_version")
        == PROJECTION_CALIBRATION_PROTOCOL_VERSION
        and calibration.get("edge_projection_protocol_version")
        == EDGE_PROJECTION_PROTOCOL_VERSION
        and calibration.get("edge_projection_protocol_hash")
        == expected_projection_hash
        and calibration.get("layer") == layer
        and calibration.get("cross_type_raw_comparison_allowed") is False
        and calibration.get("model_call_count") == 0
        and set(stats_by_type) == set(grouped),
        code="projection_calibration_protocol_or_scope_mismatch",
        scope=layer,
    )

    for edge_type, typed_rows in grouped.items():
        typed_rows = sorted(
            typed_rows,
            key=lambda row: (
                _float(row.get("projected_strength_raw")),
                str(row.get("id") or ""),
            ),
        )
        stats = stats_by_type.get(edge_type)
        audit.check(
            f"{layer}_{edge_type}_projection_stats_present",
            isinstance(stats, Mapping),
            code="projection_calibration_stats_missing",
            scope=f"{layer}.{edge_type}",
        )
        if not isinstance(stats, Mapping):
            continue
        raw_strengths = [
            max(1e-12, min(1.0, _float(row.get("projected_strength_raw"))))
            for row in typed_rows
        ]
        lower_value = _linear_quantile(
            raw_strengths,
            PROJECTION_CALIBRATION_PARAMS["lower_quantile"],
            round_digits=None,
        )
        upper_value = _linear_quantile(
            raw_strengths,
            PROJECTION_CALIBRATION_PARAMS["upper_quantile"],
            round_digits=None,
        )
        quantile_span = (
            float(upper_value) - float(lower_value)
            if lower_value is not None and upper_value is not None
            else None
        )
        if len(raw_strengths) < 2:
            fallback = "insufficient_layer_type_sample"
        elif (
            quantile_span is None
            or quantile_span < PROJECTION_CALIBRATION_PARAMS["min_span"]
        ):
            fallback = "degenerate_layer_type_quantile_span"
        else:
            fallback = None
        expected_calibrated: list[float] = []
        for raw_strength in raw_strengths:
            if fallback:
                calibrated_strength = raw_strength
            else:
                unit_value = (raw_strength - float(lower_value)) / max(
                    float(upper_value) - float(lower_value), 1e-12
                )
                unit_value = max(0.0, min(1.0, unit_value))
                calibrated_strength = (
                    PROJECTION_CALIBRATION_PARAMS["strength_floor"]
                    + (
                        1.0
                        - PROJECTION_CALIBRATION_PARAMS["strength_floor"]
                    )
                    * unit_value
                )
            expected_calibrated.append(
                max(1e-12, min(1.0, calibrated_strength))
            )
        expected_distances = [
            _raw_projected_distance(value) for value in expected_calibrated
        ]
        expected_stats_without_hash = {
            "normalization": PROJECTION_CALIBRATION_PROTOCOL_VERSION,
            "protocol_version": PROJECTION_CALIBRATION_PROTOCOL_VERSION,
            "edge_projection_protocol_version": (
                EDGE_PROJECTION_PROTOCOL_VERSION
            ),
            "edge_projection_protocol_hash": expected_projection_hash,
            "layer": layer,
            "edge_type": edge_type,
            "scope": "layer_plus_edge_type",
            "params": PROJECTION_CALIBRATION_PARAMS,
            "sample_count": len(raw_strengths),
            "lower_quantile_value": lower_value,
            "upper_quantile_value": upper_value,
            "quantile_span": quantile_span,
            "fallback": fallback,
            "calibration_applied": fallback is None,
            "raw_strength_distribution": _distribution(raw_strengths),
            "calibrated_strength_distribution": _distribution(
                expected_calibrated
            ),
            "calibrated_distance_distribution": _distribution(
                expected_distances
            ),
            "cross_type_raw_comparison_allowed": False,
            "model_call_count": 0,
        }
        expected_hash = stable_hash(expected_stats_without_hash)
        expected_stats = {
            **expected_stats_without_hash,
            "stats_hash": expected_hash,
        }
        stats_match = all(
            (
                _optional_numeric_match(stats.get(field), expected)
                if field
                in {
                    "lower_quantile_value",
                    "upper_quantile_value",
                    "quantile_span",
                }
                else stats.get(field) == expected
            )
            for field, expected in expected_stats.items()
        )
        audit.check(
            f"{layer}_{edge_type}_projection_stats_and_formula_replayed",
            stats_match,
            code="projection_calibration_stats_or_formula_mismatch",
            scope=f"{layer}.{edge_type}",
            expected=expected_stats,
            actual=dict(stats),
        )
        for row, expected_weight, expected_distance in zip(
            typed_rows, expected_calibrated, expected_distances
        ):
            row_id = str(row.get("id") or "missing")
            row_stats = row.get("projection_normalization_stats")
            row_stats = row_stats if isinstance(row_stats, Mapping) else {}
            row_core = {
                key: value
                for key, value in row_stats.items()
                if key
                not in {
                    "support_edge_count",
                    "support_chunk_edge_count",
                    "support_mid_edge_count",
                    "support_membership_mass",
                }
            }
            diagnostics = row.get("diagnostics")
            diagnostics = (
                diagnostics if isinstance(diagnostics, Mapping) else {}
            )
            audit.check(
                f"{layer}_edge_{row_id}_calibration_formula_replayed",
                _strict_numeric_match(row.get("weight"), expected_weight)
                and _strict_numeric_match(row.get("distance"), expected_distance),
                code="projected_calibrated_strength_formula_mismatch",
                scope=f"{layer}.{row_id}",
                expected={
                    "weight": expected_weight,
                    "distance": expected_distance,
                },
                actual={
                    "weight": row.get("weight"),
                    "distance": row.get("distance"),
                },
            )
            audit.check(
                f"{layer}_edge_{row_id}_local_protocol_and_stats_bound",
                row.get("protocol_version")
                == EDGE_PROJECTION_PROTOCOL_VERSION
                and row.get("edge_projection_protocol_hash")
                == expected_projection_hash
                and row.get("source_algorithm")
                == "membership_weighted_bottom_edge_projection"
                and dict(row_core) == expected_stats
                and row_stats.get("support_edge_count")
                == len(_unique_strings(row.get("support_chunk_edge_ids")))
                and row_stats.get("support_chunk_edge_count")
                == len(_unique_strings(row.get("support_chunk_edge_ids")))
                and _strict_numeric_match(
                    row_stats.get("support_membership_mass"),
                    diagnostics.get("support_membership_mass"),
                )
                and (
                    layer != "coarse"
                    or row_stats.get("support_mid_edge_count")
                    == len(_unique_strings(row.get("support_mid_edge_ids")))
                )
                and diagnostics.get("edge_projection_protocol")
                == EDGE_PROJECTION_PROTOCOL_VERSION
                and diagnostics.get("edge_projection_protocol_hash")
                == expected_projection_hash
                and diagnostics.get("aggregation_protocol_version")
                == PROJECTION_AGGREGATION_PROTOCOL_VERSION
                and diagnostics.get("calibration_protocol_version")
                == PROJECTION_CALIBRATION_PROTOCOL_VERSION
                and diagnostics.get("source_algorithm")
                == "membership_weighted_bottom_edge_projection"
                and diagnostics.get("gray_zone_semantics_changed") is False
                and diagnostics.get("model_call_count") == 0,
                code="projection_row_protocol_or_stats_binding_mismatch",
                scope=f"{layer}.{row_id}",
            )


def audit_graph_quality(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    audit = _Audit("graph")
    relation_edges = list(snapshot.get("relation_edges") or [])
    memberships = list(snapshot.get("rq_memberships") or [])
    projections = snapshot.get("projected_edges") or {}
    mid_edges = list(projections.get("mid") or [])
    coarse_edges = list(projections.get("coarse") or [])
    relation_ids = [str(edge.get("id") or "") for edge in relation_edges]
    relation_by_id = {
        str(edge.get("id")): edge
        for edge in relation_edges
        if isinstance(edge, Mapping) and edge.get("id")
    }
    chunk_business_keys = snapshot.get("chunk_business_keys")
    chunk_business_keys = (
        chunk_business_keys
        if isinstance(chunk_business_keys, Mapping)
        else {}
    )
    rq_prefix_scope_keys = snapshot.get("rq_prefix_scope_keys")
    rq_prefix_scope_keys = (
        rq_prefix_scope_keys
        if isinstance(rq_prefix_scope_keys, Mapping)
        else {}
    )
    membership_by_scope_chunk: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}
    duplicate_membership_keys: list[tuple[str, str]] = []
    for membership in memberships:
        if not isinstance(membership, Mapping):
            continue
        key = (
            str(membership.get("rq_prefix_id") or ""),
            str(membership.get("chunk_id") or ""),
        )
        if key in membership_by_scope_chunk:
            duplicate_membership_keys.append(key)
        membership_by_scope_chunk[key] = membership
    referenced_chunk_ids = {
        str(edge.get(field))
        for edge in relation_edges
        for field in ("source_chunk_id", "target_chunk_id")
        if edge.get(field)
    }
    mid_edge_ids = {
        str(edge.get("id")) for edge in mid_edges if edge.get("id")
    }
    _check_declared_count(audit, snapshot, "chunk_relation_edges", relation_edges)
    _check_declared_count(audit, snapshot, "rq_prefix_memberships", memberships)
    _check_declared_count(audit, snapshot, "mid_concept_edges", mid_edges)
    _check_declared_count(audit, snapshot, "coarse_concept_edges", coarse_edges)
    audit.metrics.update(
        {
            "relation_edge_count": len(relation_edges),
            "rq_membership_count": len(memberships),
            "mid_projected_edge_count": len(mid_edges),
            "coarse_projected_edge_count": len(coarse_edges),
        }
    )
    audit.check("relation_edges_present", bool(relation_edges), code="relation_edges_missing")
    audit.check(
        "relation_edge_ids_unique_and_complete",
        bool(relation_ids)
        and all(relation_ids)
        and len(relation_ids) == len(set(relation_ids))
        and len(relation_by_id) == len(relation_edges),
        code="relation_edge_identity_invalid",
        actual=relation_ids,
    )
    audit.check(
        "projection_chunk_business_facts_complete",
        bool(referenced_chunk_ids)
        and referenced_chunk_ids.issubset(set(chunk_business_keys))
        and all(
            _is_hash(chunk_business_keys.get(chunk_id))
            for chunk_id in referenced_chunk_ids
        ),
        code="projection_chunk_business_facts_missing",
        expected=sorted(referenced_chunk_ids),
        actual=sorted(chunk_business_keys),
    )
    audit.check(
        "rq_membership_scope_chunk_facts_unique",
        bool(membership_by_scope_chunk)
        and not duplicate_membership_keys
        and all(prefix_id and chunk_id for prefix_id, chunk_id in membership_by_scope_chunk),
        code="rq_membership_scope_chunk_identity_invalid",
        actual=duplicate_membership_keys,
    )
    for edge in relation_edges:
        edge_id = str(edge.get("id") or "missing")
        raw_strength = edge.get("raw_strength")
        weight = edge.get("weight")
        distance = edge.get("distance")
        summary = edge.get("raw_strength_summary") or {}
        audit.check(
            f"relation_edge_{edge_id}_numeric_domain",
            _finite(raw_strength)
            and 0.0 < float(raw_strength) <= 1.0
            and _finite(weight)
            and 0.0 < float(weight) <= 1.0
            and _finite(distance)
            and float(distance) >= 0.0,
            code="relation_edge_numeric_domain_invalid",
            scope=edge_id,
            actual={"raw_strength": raw_strength, "weight": weight, "distance": distance},
        )
        expected_distance = _distance_from_strength(weight) if _finite(weight) else None
        audit.check(
            f"relation_edge_{edge_id}_distance_replayed",
            expected_distance is not None
            and _distance_replay_matches(distance, expected_distance),
            code="relation_edge_distance_mismatch",
            scope=edge_id,
            expected=expected_distance,
            actual=distance,
        )
        audit.check(
            f"relation_edge_{edge_id}_typed_and_supported",
            edge.get("edge_type") in ACTIVE_RELATION_EDGE_TYPES
            and edge.get("source_chunk_id")
            and edge.get("target_chunk_id")
            and edge.get("source_chunk_id") != edge.get("target_chunk_id")
            and isinstance(edge.get("support"), Mapping)
            and edge["support"].get("protocol_version")
            == RELATION_SUPPORT_PROTOCOL_VERSION
            and edge["support"].get("kind") == "dense_vector_pair"
            and edge["support"].get("source") == edge.get("edge_type")
            and edge["support"].get("model_call_count") == 0
            and _unique_strings(edge["support"].get("chunk_ids"))
            == sorted(
                [
                    str(edge.get("source_chunk_id")),
                    str(edge.get("target_chunk_id")),
                ]
            )
            and len(edge["support"].get("chunk_ids") or []) == 2,
            code="relation_edge_type_or_support_invalid",
            scope=edge_id,
        )
        audit.check(
            f"relation_edge_{edge_id}_summary_replayed",
            _finite(summary.get("calibrated_strength"))
            and math.isclose(_float(summary.get("calibrated_strength")), _float(weight), abs_tol=TOLERANCE)
            and _finite(summary.get("distance"))
            and math.isclose(_float(summary.get("distance")), _float(distance), abs_tol=TOLERANCE)
            and _is_hash(edge.get("edge_distance_protocol_hash"))
            and summary.get("edge_distance_protocol"),
            code="relation_edge_summary_mismatch",
            scope=edge_id,
        )
    _audit_relation_calibration(
        audit,
        relation_edges,
        snapshot.get("relation_calibration") or {},
    )
    role_counts: Counter[str] = Counter()
    entropies: list[float] = []
    residual_norm_by_chunk: dict[str, float] = {}
    residual_norm_chunk_identity_valid = True
    scores: list[float] = []
    for membership in memberships:
        membership_id = str(membership.get("id") or "missing")
        role = str(membership.get("membership_role") or "")
        score = membership.get("membership_score")
        entropy = membership.get("membership_entropy")
        residual = membership.get("residual_norm")
        chunk_id = str(membership.get("chunk_id") or "")
        rank = membership.get("rank")
        evaluation = membership.get("role_evaluation") or {}
        expected_role, expected_matched_flags = _membership_role_replay(
            evaluation
        )
        audit.check(
            f"rq_membership_{membership_id}_numeric_domain",
            _finite(score)
            and 0.0 < float(score) <= 1.0
            and _finite(entropy)
            and 0.0 <= float(entropy) <= 1.0
            and _finite(residual)
            and float(residual) >= 0.0
            and type(rank) is int
            and rank > 0,
            code="rq_membership_numeric_domain_invalid",
            scope=membership_id,
        )
        inputs = evaluation.get("inputs") or {}
        audit.check(
            f"rq_membership_{membership_id}_role_replayed",
            role in RQ_MEMBERSHIP_ROLES
            and expected_role == role
            and evaluation.get("role") == role
            and evaluation.get("primary_reason") == role
            and evaluation.get("matched_flags") == expected_matched_flags
            and evaluation.get("protocol_version")
            == RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION
            and evaluation.get("protocol_hash")
            == rq_membership_role_protocol_hash()
            and dict(evaluation.get("thresholds") or {})
            == RQ_MEMBERSHIP_ROLE_THRESHOLDS
            and evaluation.get("model_call_count") == 0
            and _finite(inputs.get("membership_score"))
            and math.isclose(_float(inputs.get("membership_score")), _float(score), abs_tol=1e-12)
            and _finite(inputs.get("membership_entropy"))
            and math.isclose(_float(inputs.get("membership_entropy")), _float(entropy), abs_tol=1e-12)
            and _finite(inputs.get("residual_norm"))
            and math.isclose(_float(inputs.get("residual_norm")), _float(residual), abs_tol=1e-12)
            and inputs.get("rank") == rank,
            code="rq_membership_role_replay_mismatch",
            scope=membership_id,
            expected=expected_role,
            actual=role,
        )
        role_counts[role] += 1
        if _finite(entropy):
            entropies.append(float(entropy))
        if _finite(residual):
            residual_value = float(residual)
            if not chunk_id:
                residual_norm_chunk_identity_valid = False
            elif chunk_id in residual_norm_by_chunk and not _strict_numeric_match(
                residual_norm_by_chunk[chunk_id], residual_value
            ):
                residual_norm_chunk_identity_valid = False
            else:
                residual_norm_by_chunk[chunk_id] = residual_value
        if _finite(score):
            scores.append(float(score))
    rq_diagnostics = snapshot.get("rq_membership_diagnostics") or {}
    audit.check(
        "rq_residual_norm_chunk_identity_replayed",
        residual_norm_chunk_identity_valid
        and bool(residual_norm_by_chunk),
        code="rq_residual_norm_chunk_identity_mismatch",
        expected="one finite residual_norm per chunk identity",
        actual={
            "membership_count": len(memberships),
            "chunk_residual_count": len(residual_norm_by_chunk),
        },
    )
    audit.check(
        "rq_membership_count_replayed",
        rq_diagnostics.get("membership_count") == len(memberships),
        code="rq_membership_diagnostic_count_mismatch",
        expected=len(memberships),
        actual=rq_diagnostics.get("membership_count"),
    )
    audit.check(
        "rq_membership_role_distribution_replayed",
        dict(sorted(role_counts.items())) == dict(rq_diagnostics.get("membership_role_counts") or {}),
        code="rq_membership_role_distribution_mismatch",
        expected=dict(sorted(role_counts.items())),
        actual=rq_diagnostics.get("membership_role_counts"),
    )
    for field, values in (
        ("membership_score_distribution", scores),
        ("membership_entropy_distribution", entropies),
        (
            "residual_norm_distribution",
            [
                residual_norm_by_chunk[chunk_id]
                for chunk_id in sorted(residual_norm_by_chunk)
            ],
        ),
    ):
        audit.check(
            f"rq_{field}_replayed",
            _distribution_matches(rq_diagnostics.get(field), _distribution(values)),
            code="rq_membership_distribution_mismatch",
            scope=field,
            expected=_distribution(values),
            actual=rq_diagnostics.get(field),
        )
    audit.check(
        "rq_membership_diagnostics_are_deterministic",
        rq_diagnostics.get("model_call_count") == 0
        and rq_diagnostics.get("renormalized_after_sparsification") is False
        and rq_diagnostics.get("artificial_membership_floor") is False
        and rq_diagnostics.get("rq_pair_edges_active") is False,
        code="rq_membership_diagnostic_protocol_invalid",
    )
    for layer, rows in (("mid", mid_edges), ("coarse", coarse_edges)):
        layer_edge_ids = [str(edge.get("id") or "") for edge in rows]
        audit.check(
            f"{layer}_projection_edge_ids_unique_and_complete",
            all(layer_edge_ids)
            and len(layer_edge_ids) == len(set(layer_edge_ids)),
            code="projected_edge_identity_invalid",
            scope=layer,
            actual=layer_edge_ids,
        )
        for edge in rows:
            edge_id = str(edge.get("id") or "missing")
            weight = edge.get("weight")
            distance = edge.get("distance")
            raw_strength = edge.get("projected_strength_raw")
            raw_distance = edge.get("projected_distance_raw")
            support_edge_ids = _unique_strings(edge.get("support_chunk_edge_ids"))
            support_chunk_ids = _unique_strings(edge.get("support_chunk_ids"))
            audit.check(
                f"{layer}_projection_{edge_id}_numeric_replay",
                _finite(weight)
                and 0.0 < float(weight) <= 1.0
                and _finite(distance)
                and float(distance) >= 0.0
                and _finite(raw_strength)
                and 0.0 < float(raw_strength) <= 1.0
                and _finite(raw_distance)
                and float(raw_distance) >= 0.0
                and _distance_replay_matches(
                    distance, _raw_projected_distance(weight)
                )
                and _distance_replay_matches(
                    raw_distance, _raw_projected_distance(raw_strength)
                ),
                code="projected_edge_distance_replay_mismatch",
                scope=f"{layer}.{edge_id}",
            )
            audit.check(
                f"{layer}_projection_{edge_id}_support_bound",
                bool(support_edge_ids)
                and bool(support_chunk_ids)
                and len(support_edge_ids) == len(set(support_edge_ids))
                and edge.get("edge_projection_protocol_hash")
                == edge_projection_protocol_hash()
                and _is_hash(edge.get("state_hash"))
                and edge.get("source_algorithm") == "membership_weighted_bottom_edge_projection",
                code="projected_edge_support_or_protocol_invalid",
                scope=f"{layer}.{edge_id}",
            )
            _audit_projected_edge_support(
                audit,
                layer=layer,
                edge=edge,
                relation_by_id=relation_by_id,
                chunk_business_keys=chunk_business_keys,
                membership_by_scope_chunk=membership_by_scope_chunk,
                rq_prefix_scope_keys=rq_prefix_scope_keys,
                mid_edge_ids=mid_edge_ids,
            )
        _audit_projection_layer(
            audit,
            layer,
            rows,
            (snapshot.get("projection_calibration") or {}).get(layer) or {},
        )
    return audit.finish()


_RQ_ROUTE_CONTRIBUTION_FIELDS = frozenset(
    {
        "mid_concept_id",
        "mid_entry_strength",
        "mid_membership_score",
        "route_fallback_score",
    }
)
_RQ_STAGE_SEED_CARD_FIELDS = frozenset(
    {
        "protocol_version",
        "protocol_hash",
        "rq_prefix_id",
        "rq_path",
        "rq_level",
        "query_rq_path",
        "rq_lcp_depth",
        "residual_distance",
        "query_prefix_membership_score",
        "requested_query_relevance",
        "route_fallback_score",
        "parent_mid_contributions",
        "score_source",
        "effective_score",
        "forced_override",
        "relation_state_hash",
        "is_evidence",
        "node_weight_used_as_query_relevance",
        "hard_path_lcp_used_as_score",
        "gray_zone_decision_authority",
        "model_call_count",
        "input_hash",
        "card_hash",
    }
)
_RQ_CHUNK_SEED_CARD_FIELDS = frozenset(
    {
        "protocol_version",
        "protocol_hash",
        "parent_mid_concept_id",
        "chunk_id",
        "rq_l3_prefix_id",
        "query_rq_path",
        "candidate_rq_path",
        "rq_lcp_depth",
        "residual_distance",
        "query_prefix_score",
        "chunk_membership_score",
        "fuzzy_membership_overlap_score",
        "rq_score",
        "rq_relevance_component",
        "rq_drift_penalty",
        "membership_role",
        "membership_rank",
        "membership_entropy",
        "bridge_or_boundary_role",
        "support_edge_ids",
        "mid_entry_component",
        "dense_component",
        "component_weights",
        "effective_score",
        "score_source",
        "membership_role_tie_break_rank",
        "is_evidence",
        "node_weight_used_as_query_relevance",
        "hard_path_lcp_used_as_score",
        "gray_zone_decision_authority",
        "model_call_count",
        "input_hash",
        "card_hash",
    }
)
_CHUNK_FACET_PRIORITY_CARD_FIELDS = frozenset(
    {
        "protocol_version",
        "protocol_hash",
        "facet_match_protocol_version",
        "facet_match_protocol_hash",
        "chunk_id",
        "query_facet_packet_hash",
        "required_facets",
        "matched_required_facets",
        "uncovered_required_facets",
        "matched_required_facet_count",
        "uncovered_required_facet_count",
        "priority_prefix",
        "lexical_overlap_used_as_numeric_relevance",
        "is_evidence",
        "citation_authority",
        "gray_zone_decision_authority",
        "model_call_count",
        "card_hash",
    }
)


def _strict_unit_score(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and _finite(value)
        and 0.0 <= float(value) <= 1.0
    )


def _strict_nonnegative_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and _finite(value)
        and float(value) >= 0.0
    )


def _strict_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _rq_round6_numeric_match(observed: Any, expected: Any) -> bool:
    return (
        _finite(observed)
        and _finite(expected)
        and math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            # Both multiplicands and the product are independently rounded
            # to six decimals in the persisted card.
            abs_tol=2.0 * TOLERANCE,
        )
    )


def _strict_rq_path(value: Any) -> bool:
    return isinstance(value, list) and all(
        _strict_nonnegative_int(item) for item in value
    )


def _rq_lcp_depth(left: Sequence[int], right: Sequence[int]) -> int:
    depth = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        depth += 1
    return depth


def _rq_card_hashes_match(card: Mapping[str, Any]) -> bool:
    input_hash = card.get("input_hash")
    card_hash = card.get("card_hash")
    if not _is_hash(input_hash) or not _is_hash(card_hash):
        return False
    input_payload = dict(card)
    input_payload.pop("input_hash", None)
    input_payload.pop("card_hash", None)
    if input_hash != stable_hash(input_payload):
        return False
    card_payload = dict(input_payload)
    card_payload["input_hash"] = input_hash
    return card_hash == stable_hash(card_payload)


def _rq_component_weights_match(
    observed: Any,
    expected: Mapping[str, float],
) -> bool:
    return (
        isinstance(observed, Mapping)
        and set(observed) == set(expected)
        and all(
            _strict_numeric_match(observed.get(key), value)
            for key, value in expected.items()
        )
    )


def _audit_rq_stage_seed_card(
    audit: _Audit,
    *,
    scope: str,
    candidate_id: str,
    card: Mapping[str, Any],
    forced_candidate_ids: set[str],
) -> bool:
    card_scope = f"{scope}.rq_seed_cards.{candidate_id}"
    contributions_raw = card.get("parent_mid_contributions")
    contributions = (
        list(contributions_raw)
        if isinstance(contributions_raw, list)
        else []
    )
    rq_path = card.get("rq_path")
    query_rq_path = card.get("query_rq_path")
    schema_valid = (
        set(card) == set(_RQ_STAGE_SEED_CARD_FIELDS)
        and str(card.get("rq_prefix_id") or "") == candidate_id
        and _strict_rq_path(rq_path)
        and _strict_rq_path(query_rq_path)
        and _strict_nonnegative_int(card.get("rq_level"))
        and _strict_nonnegative_int(card.get("rq_lcp_depth"))
        and (
            card.get("residual_distance") is None
            or _strict_nonnegative_number(
                card.get("residual_distance")
            )
        )
        and (
            card.get("query_prefix_membership_score") is None
            or _strict_unit_score(
                card.get("query_prefix_membership_score")
            )
        )
        and (
            card.get("requested_query_relevance") is None
            or _strict_unit_score(
                card.get("requested_query_relevance")
            )
        )
        and _strict_unit_score(card.get("route_fallback_score"))
        and isinstance(contributions_raw, list)
        and all(
            isinstance(item, Mapping)
            and set(item) == set(_RQ_ROUTE_CONTRIBUTION_FIELDS)
            and bool(str(item.get("mid_concept_id") or ""))
            and _strict_unit_score(item.get("mid_entry_strength"))
            and _strict_unit_score(item.get("mid_membership_score"))
            and _strict_unit_score(item.get("route_fallback_score"))
            for item in contributions
        )
        and bool(str(card.get("score_source") or ""))
        and _strict_unit_score(card.get("effective_score"))
        and type(card.get("forced_override")) is bool
        and (
            card.get("relation_state_hash") is None
            or _is_hash(card.get("relation_state_hash"))
        )
        and type(card.get("is_evidence")) is bool
        and type(
            card.get("node_weight_used_as_query_relevance")
        )
        is bool
        and type(card.get("hard_path_lcp_used_as_score")) is bool
        and type(card.get("gray_zone_decision_authority")) is bool
        and _strict_nonnegative_int(card.get("model_call_count"))
    )
    audit.check(
        f"{card_scope}_schema",
        schema_valid,
        code="query_rq_stage_seed_card_schema_mismatch",
        scope=card_scope,
        expected=sorted(_RQ_STAGE_SEED_CARD_FIELDS),
        actual=sorted(str(key) for key in card),
    )
    protocol_valid = (
        card.get("protocol_version")
        == QUERY_RQ_SEED_PROTOCOL_VERSION
        and card.get("protocol_hash")
        == query_rq_seed_protocol_hash()
    )
    audit.check(
        f"{card_scope}_protocol",
        protocol_valid,
        code="query_rq_seed_protocol_mismatch",
        scope=card_scope,
        expected={
            "protocol_version": QUERY_RQ_SEED_PROTOCOL_VERSION,
            "protocol_hash": query_rq_seed_protocol_hash(),
        },
        actual={
            "protocol_version": card.get("protocol_version"),
            "protocol_hash": card.get("protocol_hash"),
        },
    )
    hash_valid = _rq_card_hashes_match(card)
    audit.check(
        f"{card_scope}_hash",
        hash_valid,
        code="query_rq_stage_seed_card_hash_mismatch",
        scope=card_scope,
    )

    contribution_mid_ids = [
        str(item.get("mid_concept_id") or "")
        for item in contributions
        if isinstance(item, Mapping)
    ]
    contributions_valid = (
        contribution_mid_ids == sorted(contribution_mid_ids)
        and all(
            _rq_round6_numeric_match(
                item.get("route_fallback_score"),
                round(
                    float(item.get("mid_entry_strength") or 0.0)
                    * float(
                        item.get("mid_membership_score") or 0.0
                    ),
                    6,
                ),
            )
            for item in contributions
            if isinstance(item, Mapping)
        )
    )
    expected_route_fallback = max(
        [
            float(item.get("route_fallback_score") or 0.0)
            for item in contributions
            if isinstance(item, Mapping)
        ]
        or [0.0]
    )
    forced_override = card.get("forced_override") is True
    requested_score = card.get("requested_query_relevance")
    if forced_override:
        expected_source = "typed_action_forced_override"
        expected_effective_score = 1.0
    elif requested_score is not None:
        expected_source = "query_rq_relevance"
        expected_effective_score = requested_score
    else:
        expected_source = "selected_mid_route_fallback"
        expected_effective_score = expected_route_fallback
    formula_valid = (
        schema_valid
        and contributions_valid
        and card.get("rq_level") == len(rq_path)
        and card.get("rq_lcp_depth")
        == _rq_lcp_depth(query_rq_path, rq_path)
        and _strict_numeric_match(
            card.get("route_fallback_score"),
            expected_route_fallback,
        )
        and forced_override
        == (candidate_id in forced_candidate_ids)
        and card.get("score_source") == expected_source
        and _strict_numeric_match(
            card.get("effective_score"),
            expected_effective_score,
        )
    )
    audit.check(
        f"{card_scope}_formula",
        formula_valid,
        code="query_rq_stage_seed_card_formula_mismatch",
        scope=card_scope,
        expected={
            "score_source": expected_source,
            "effective_score": expected_effective_score,
            "route_fallback_score": expected_route_fallback,
        },
        actual={
            "score_source": card.get("score_source"),
            "effective_score": card.get("effective_score"),
            "route_fallback_score": card.get(
                "route_fallback_score"
            ),
            "parent_mid_contributions_valid": contributions_valid,
            "parent_mid_contributions": contributions,
            "rq_level_matches": (
                card.get("rq_level") == len(rq_path)
            ),
            "rq_lcp_depth_matches": (
                card.get("rq_lcp_depth")
                == _rq_lcp_depth(query_rq_path, rq_path)
            ),
            "forced_override_matches": (
                forced_override
                == (candidate_id in forced_candidate_ids)
            ),
        },
    )
    authority_valid = (
        card.get("is_evidence") is False
        and card.get("node_weight_used_as_query_relevance") is False
        and card.get("hard_path_lcp_used_as_score") is False
        and card.get("gray_zone_decision_authority") is False
        and card.get("model_call_count") == 0
    )
    audit.check(
        f"{card_scope}_authority",
        authority_valid,
        code="query_rq_seed_authority_violation",
        scope=card_scope,
    )
    return (
        schema_valid
        and protocol_valid
        and hash_valid
        and formula_valid
        and authority_valid
    )


def _audit_rq_chunk_seed_card(
    audit: _Audit,
    *,
    scope: str,
    candidate_id: str,
    card_index: int,
    card: Mapping[str, Any],
    expected_parent_mid_id: str | None,
) -> bool:
    card_scope = (
        f"{scope}.rq_chunk_seed_cards."
        f"{candidate_id}.{card_index}"
    )
    query_rq_path = card.get("query_rq_path")
    candidate_rq_path = card.get("candidate_rq_path")
    support_edge_ids = card.get("support_edge_ids")
    schema_valid = (
        set(card) == set(_RQ_CHUNK_SEED_CARD_FIELDS)
        and bool(str(card.get("parent_mid_concept_id") or ""))
        and str(card.get("chunk_id") or "") == candidate_id
        and (
            card.get("rq_l3_prefix_id") is None
            or bool(str(card.get("rq_l3_prefix_id") or ""))
        )
        and _strict_rq_path(query_rq_path)
        and _strict_rq_path(candidate_rq_path)
        and _strict_nonnegative_int(card.get("rq_lcp_depth"))
        and (
            card.get("residual_distance") is None
            or _strict_nonnegative_number(
                card.get("residual_distance")
            )
        )
        and all(
            _strict_unit_score(card.get(field))
            for field in (
                "query_prefix_score",
                "chunk_membership_score",
                "fuzzy_membership_overlap_score",
                "rq_score",
                "rq_relevance_component",
                "mid_entry_component",
                "dense_component",
                "effective_score",
            )
        )
        and (
            card.get("rq_drift_penalty") is None
            or _strict_nonnegative_number(
                card.get("rq_drift_penalty")
            )
        )
        and bool(str(card.get("membership_role") or ""))
        and _strict_nonnegative_int(card.get("membership_rank"))
        and (
            card.get("membership_entropy") is None
            or _strict_unit_score(card.get("membership_entropy"))
        )
        and type(card.get("bridge_or_boundary_role")) is bool
        and isinstance(support_edge_ids, list)
        and all(bool(str(value)) for value in support_edge_ids)
        and len(support_edge_ids)
        == len(set(str(value) for value in support_edge_ids))
        and isinstance(card.get("component_weights"), Mapping)
        and bool(str(card.get("score_source") or ""))
        and _strict_nonnegative_int(
            card.get("membership_role_tie_break_rank")
        )
        and type(card.get("is_evidence")) is bool
        and type(
            card.get("node_weight_used_as_query_relevance")
        )
        is bool
        and type(card.get("hard_path_lcp_used_as_score")) is bool
        and type(card.get("gray_zone_decision_authority")) is bool
        and _strict_nonnegative_int(card.get("model_call_count"))
    )
    if expected_parent_mid_id is not None:
        schema_valid = (
            schema_valid
            and card.get("parent_mid_concept_id")
            == expected_parent_mid_id
        )
    audit.check(
        f"{card_scope}_schema",
        schema_valid,
        code="query_rq_chunk_seed_card_schema_mismatch",
        scope=card_scope,
        expected=sorted(_RQ_CHUNK_SEED_CARD_FIELDS),
        actual=sorted(str(key) for key in card),
    )
    protocol_valid = (
        card.get("protocol_version")
        == QUERY_RQ_SEED_PROTOCOL_VERSION
        and card.get("protocol_hash")
        == query_rq_seed_protocol_hash()
    )
    audit.check(
        f"{card_scope}_protocol",
        protocol_valid,
        code="query_rq_seed_protocol_mismatch",
        scope=card_scope,
    )
    hash_valid = _rq_card_hashes_match(card)
    audit.check(
        f"{card_scope}_hash",
        hash_valid,
        code="query_rq_chunk_seed_card_hash_mismatch",
        scope=card_scope,
    )

    source = str(card.get("score_source") or "")
    if source == "query_rq_fuzzy_membership":
        fuzzy_overlap = round(
            math.sqrt(
                max(
                    0.0,
                    _float(card.get("query_prefix_score"))
                    * _float(card.get("chunk_membership_score")),
                )
            ),
            6,
        )
        if query_rq_path:
            rq_relevance = round(
                0.75 * _float(card.get("rq_score"))
                + 0.25 * fuzzy_overlap,
                6,
            )
            no_query_drift_valid = True
        else:
            rq_relevance = fuzzy_overlap
            no_query_drift_valid = (
                _strict_numeric_match(
                    card.get("rq_score"),
                    fuzzy_overlap,
                )
                and _strict_numeric_match(
                    card.get("rq_drift_penalty"),
                    round(max(0.0, 1.0 - fuzzy_overlap), 6),
                )
            )
        expected_weights = QUERY_RQ_CHUNK_SEED_COMPONENT_WEIGHTS
        expected_effective_score = round(
            expected_weights["rq_relevance"] * rq_relevance
            + expected_weights["mid_entry"]
            * _float(card.get("mid_entry_component"))
            + expected_weights["dense"]
            * _float(card.get("dense_component")),
            6,
        )
        membership_role = str(card.get("membership_role") or "")
        expected_tie_break = (
            QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK.get(
                membership_role,
                len(QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK),
            )
        )
        source_formula_valid = (
            card.get("rq_l3_prefix_id") is not None
            and _rq_component_weights_match(
                card.get("component_weights"),
                expected_weights,
            )
            and _strict_numeric_match(
                card.get("fuzzy_membership_overlap_score"),
                fuzzy_overlap,
            )
            and _strict_numeric_match(
                card.get("rq_relevance_component"),
                rq_relevance,
            )
            and no_query_drift_valid
            and card.get("membership_role_tie_break_rank")
            == expected_tie_break
        )
    elif source == "mid_support_without_rq_membership":
        expected_weights = (
            QUERY_RQ_MID_SUPPORT_FALLBACK_COMPONENT_WEIGHTS
        )
        expected_effective_score = round(
            expected_weights["mid_entry"]
            * _float(card.get("mid_entry_component"))
            + expected_weights["dense"]
            * _float(card.get("dense_component")),
            6,
        )
        source_formula_valid = (
            card.get("rq_l3_prefix_id") is None
            and query_rq_path == []
            and candidate_rq_path == []
            and card.get("rq_lcp_depth") == 0
            and card.get("residual_distance") is None
            and all(
                _strict_numeric_match(card.get(field), 0.0)
                for field in (
                    "query_prefix_score",
                    "chunk_membership_score",
                    "fuzzy_membership_overlap_score",
                    "rq_score",
                    "rq_relevance_component",
                )
            )
            and card.get("rq_drift_penalty") is None
            and card.get("membership_role")
            == "mid_support_fallback"
            and card.get("membership_rank") == 0
            and card.get("membership_entropy") is None
            and card.get("bridge_or_boundary_role") is False
            and support_edge_ids == []
            and _rq_component_weights_match(
                card.get("component_weights"),
                expected_weights,
            )
            and card.get("membership_role_tie_break_rank")
            == len(QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK) + 1
        )
    else:
        expected_effective_score = None
        source_formula_valid = False
    formula_valid = (
        schema_valid
        and card.get("rq_lcp_depth")
        == _rq_lcp_depth(query_rq_path, candidate_rq_path)
        and source_formula_valid
        and _strict_numeric_match(
            card.get("effective_score"),
            expected_effective_score,
        )
    )
    audit.check(
        f"{card_scope}_formula",
        formula_valid,
        code="query_rq_chunk_seed_card_formula_mismatch",
        scope=card_scope,
        expected={
            "score_source": source,
            "effective_score": expected_effective_score,
        },
        actual={
            "score_source": card.get("score_source"),
            "effective_score": card.get("effective_score"),
        },
    )
    authority_valid = (
        card.get("is_evidence") is False
        and card.get("node_weight_used_as_query_relevance") is False
        and card.get("hard_path_lcp_used_as_score") is False
        and card.get("gray_zone_decision_authority") is False
        and card.get("model_call_count") == 0
    )
    audit.check(
        f"{card_scope}_authority",
        authority_valid,
        code="query_rq_seed_authority_violation",
        scope=card_scope,
    )
    return (
        schema_valid
        and protocol_valid
        and hash_valid
        and formula_valid
        and authority_valid
    )


def _audit_query_rq_candidate_pool(
    audit: _Audit,
    *,
    scope: str,
    pool: Mapping[str, Any],
    candidates: Sequence[str],
    candidate_scores: Mapping[str, Any],
    forced_candidate_ids: Sequence[str],
) -> tuple[bool, dict[str, int]]:
    stage_cards_raw = pool.get("rq_seed_cards")
    chunk_cards_raw = pool.get("rq_chunk_seed_cards")
    stage_cards = (
        stage_cards_raw
        if isinstance(stage_cards_raw, Mapping)
        else {}
    )
    chunk_cards = (
        chunk_cards_raw
        if isinstance(chunk_cards_raw, Mapping)
        else {}
    )
    marker_present = (
        pool.get("ranking_protocol_version") is not None
        or pool.get("ranking_protocol_hash") is not None
        or bool(stage_cards)
        or bool(chunk_cards)
    )
    if not marker_present:
        return False, {}
    expected_protocol_hash = query_rq_seed_protocol_hash()
    audit.check(
        f"{scope}_query_rq_protocol",
        pool.get("ranking_protocol_version")
        == QUERY_RQ_SEED_PROTOCOL_VERSION
        and pool.get("ranking_protocol_hash")
        == expected_protocol_hash,
        code="query_rq_seed_protocol_mismatch",
        scope=scope,
        expected={
            "ranking_protocol_version": (
                QUERY_RQ_SEED_PROTOCOL_VERSION
            ),
            "ranking_protocol_hash": expected_protocol_hash,
        },
        actual={
            "ranking_protocol_version": pool.get(
                "ranking_protocol_version"
            ),
            "ranking_protocol_hash": pool.get(
                "ranking_protocol_hash"
            ),
        },
    )
    candidate_set = set(candidates)
    if scope == "rq_membership_entries":
        coverage_valid = (
            isinstance(stage_cards_raw, Mapping)
            and set(str(key) for key in stage_cards)
            == candidate_set
            and not chunk_cards
        )
        audit.check(
            f"{scope}_query_rq_stage_card_coverage",
            coverage_valid,
            code="query_rq_stage_seed_card_coverage_mismatch",
            scope=scope,
            expected=sorted(candidate_set),
            actual=sorted(str(key) for key in stage_cards),
        )
        forced_set = set(forced_candidate_ids)
        for candidate_id in candidates:
            card = stage_cards.get(candidate_id)
            if not isinstance(card, Mapping):
                continue
            _audit_rq_stage_seed_card(
                audit,
                scope=scope,
                candidate_id=candidate_id,
                card=card,
                forced_candidate_ids=forced_set,
            )
            audit.check(
                f"{scope}_{candidate_id}_query_rq_stage_score",
                candidate_id in candidate_scores
                and _strict_numeric_match(
                    candidate_scores.get(candidate_id),
                    card.get("effective_score"),
                ),
                code="query_rq_seed_candidate_score_mismatch",
                scope=f"{scope}.{candidate_id}",
            )
        return False, {}

    is_per_mid_chunk_pool = scope.startswith("chunk_by_mid.")
    coverage_valid = (
        isinstance(chunk_cards_raw, Mapping)
        and set(str(key) for key in chunk_cards).issubset(
            candidate_set
        )
        and all(
            isinstance(cards, list) and bool(cards)
            for cards in chunk_cards.values()
        )
    )
    if is_per_mid_chunk_pool:
        coverage_valid = (
            coverage_valid
            and set(str(key) for key in chunk_cards)
            == candidate_set
            and not stage_cards
        )
    audit.check(
        f"{scope}_query_rq_chunk_card_coverage",
        coverage_valid,
        code="query_rq_chunk_seed_card_coverage_mismatch",
        scope=scope,
        expected=(
            sorted(candidate_set)
            if is_per_mid_chunk_pool
            else "nonempty_card_lists_for_mapped_candidates"
        ),
        actual=sorted(str(key) for key in chunk_cards),
    )
    expected_parent_mid_id = (
        str(pool.get("parent_node_id") or "")
        if is_per_mid_chunk_pool
        else None
    )
    tie_break_by_candidate: dict[str, int] = {}
    for candidate_id, cards_raw in chunk_cards.items():
        candidate_id = str(candidate_id)
        if not isinstance(cards_raw, list):
            continue
        effective_scores: list[float] = []
        tie_break_ranks: list[int] = []
        card_hashes: list[str] = []
        for card_index, card in enumerate(cards_raw):
            if not isinstance(card, Mapping):
                audit.check(
                    (
                        f"{scope}_{candidate_id}_{card_index}"
                        "_query_rq_chunk_card_mapping"
                    ),
                    False,
                    code="query_rq_chunk_seed_card_schema_mismatch",
                    scope=f"{scope}.{candidate_id}.{card_index}",
                )
                continue
            _audit_rq_chunk_seed_card(
                audit,
                scope=scope,
                candidate_id=candidate_id,
                card_index=card_index,
                card=card,
                expected_parent_mid_id=expected_parent_mid_id,
            )
            if _strict_unit_score(card.get("effective_score")):
                effective_scores.append(
                    float(card["effective_score"])
                )
            if _strict_nonnegative_int(
                card.get("membership_role_tie_break_rank")
            ):
                tie_break_ranks.append(
                    int(card["membership_role_tie_break_rank"])
                )
            card_hashes.append(str(card.get("card_hash") or ""))
        audit.check(
            f"{scope}_{candidate_id}_query_rq_card_identity",
            bool(card_hashes)
            and len(card_hashes) == len(set(card_hashes)),
            code="query_rq_chunk_seed_card_duplicate_mismatch",
            scope=f"{scope}.{candidate_id}",
        )
        if effective_scores:
            replayed_card_score = max(effective_scores)
            if is_per_mid_chunk_pool:
                score_valid = (
                    candidate_id in candidate_scores
                    and _strict_numeric_match(
                        candidate_scores.get(candidate_id),
                        replayed_card_score,
                    )
                )
            else:
                score_valid = (
                    candidate_id in candidate_scores
                    and _finite(candidate_scores.get(candidate_id))
                    and float(candidate_scores[candidate_id])
                    + TOLERANCE
                    >= replayed_card_score
                )
            audit.check(
                f"{scope}_{candidate_id}_query_rq_chunk_score",
                score_valid,
                code="query_rq_seed_candidate_score_mismatch",
                scope=f"{scope}.{candidate_id}",
                expected=(
                    replayed_card_score
                    if is_per_mid_chunk_pool
                    else {
                        "candidate_score_gte": replayed_card_score
                    }
                ),
                actual=candidate_scores.get(candidate_id),
            )
        if tie_break_ranks:
            tie_break_by_candidate[candidate_id] = min(
                tie_break_ranks
            )
    return is_per_mid_chunk_pool and coverage_valid, tie_break_by_candidate


def _audit_chunk_facet_priority_candidate_pool(
    audit: _Audit,
    *,
    scope: str,
    pool: Mapping[str, Any],
    candidates: Sequence[str],
) -> tuple[bool, dict[str, int], dict[str, float]]:
    cards_raw = pool.get("chunk_facet_priority_cards")
    cards = cards_raw if isinstance(cards_raw, Mapping) else {}
    posterior_snapshot_raw = pool.get("query_facet_posterior_snapshot")
    posterior_mass_raw = pool.get("covered_posterior_mass_by_candidate")
    required_for_scope = bool(candidates) and (
        scope.startswith("chunk_by_mid.")
        or scope == "chunk_initial_entries"
    )
    marker_present = (
        pool.get("facet_priority_protocol_version") is not None
        or pool.get("facet_priority_protocol_hash") is not None
        or bool(cards)
        or posterior_snapshot_raw is not None
        or posterior_mass_raw is not None
    )
    expected_protocol_hash = chunk_facet_priority_protocol_hash()
    protocol_valid = (
        pool.get("facet_priority_protocol_version")
        == CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
        and pool.get("facet_priority_protocol_hash")
        == expected_protocol_hash
    )
    audit.check(
        f"{scope}_chunk_facet_priority_protocol",
        (not required_for_scope and not marker_present) or protocol_valid,
        code="chunk_facet_priority_protocol_mismatch",
        scope=scope,
        expected=(
            {
                "facet_priority_protocol_version": (
                    CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
                ),
                "facet_priority_protocol_hash": expected_protocol_hash,
            }
            if required_for_scope or marker_present
            else None
        ),
        actual={
            "facet_priority_protocol_version": pool.get(
                "facet_priority_protocol_version"
            ),
            "facet_priority_protocol_hash": pool.get(
                "facet_priority_protocol_hash"
            ),
        },
    )
    candidate_set = set(candidates)
    posterior_snapshot = (
        dict(posterior_snapshot_raw)
        if isinstance(posterior_snapshot_raw, Mapping)
        else {}
    )
    supplied_snapshot_hash = posterior_snapshot.pop("snapshot_hash", None)
    posterior_values = posterior_snapshot.get("posterior")
    posterior_snapshot_valid = (
        posterior_snapshot.get("protocol_version")
        == QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
        and posterior_snapshot.get("protocol_hash")
        == query_facet_posterior_protocol_hash()
        and isinstance(posterior_values, Mapping)
        and all(_finite(value) and 0.0 <= float(value) <= 1.0 for value in posterior_values.values())
        and (
            not posterior_values
            or math.isclose(
                sum(float(value) for value in posterior_values.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        and _is_hash(supplied_snapshot_hash)
        and supplied_snapshot_hash == stable_hash(posterior_snapshot)
    )
    posterior_masses = (
        {
            str(key): float(value)
            for key, value in posterior_mass_raw.items()
        }
        if isinstance(posterior_mass_raw, Mapping)
        and all(
            _finite(value) and 0.0 <= float(value) <= 1.0
            for value in posterior_mass_raw.values()
        )
        else {}
    )
    posterior_mass_valid = set(posterior_masses) == candidate_set
    posterior_mass_replay_valid = posterior_mass_valid and all(
        isinstance(cards.get(candidate_id), Mapping)
        and posterior_masses[candidate_id]
        == round(
            sum(
                float(posterior_values.get(facet, 0.0))
                for facet in (
                    cards[candidate_id].get(
                        "matched_required_facets"
                    )
                    or []
                )
            ),
            6,
        )
        for candidate_id in candidate_set
    )
    audit.check(
        f"{scope}_query_facet_posterior_snapshot",
        (not required_for_scope and not marker_present)
        or posterior_snapshot_valid,
        code="query_facet_posterior_snapshot_mismatch",
        scope=scope,
    )
    audit.check(
        f"{scope}_query_facet_posterior_candidate_mass",
        (not required_for_scope and not marker_present)
        or posterior_mass_replay_valid,
        code="query_facet_posterior_candidate_mass_mismatch",
        scope=scope,
        expected=sorted(candidate_set),
        actual=sorted(posterior_masses),
    )
    coverage_valid = (
        isinstance(cards_raw, Mapping)
        and set(str(key) for key in cards) == candidate_set
    )
    audit.check(
        f"{scope}_chunk_facet_priority_card_coverage",
        (not required_for_scope and not marker_present)
        or coverage_valid,
        code="chunk_facet_priority_card_coverage_mismatch",
        scope=scope,
        expected=sorted(candidate_set),
        actual=sorted(str(key) for key in cards),
    )
    uncovered_counts: dict[str, int] = {}
    all_cards_valid = coverage_valid
    for candidate_id, raw_card in cards.items():
        candidate_id = str(candidate_id)
        card_scope = (
            f"{scope}.chunk_facet_priority_cards.{candidate_id}"
        )
        if not isinstance(raw_card, Mapping):
            audit.check(
                f"{card_scope}_schema",
                False,
                code="chunk_facet_priority_card_schema_mismatch",
                scope=card_scope,
            )
            all_cards_valid = False
            continue
        card = dict(raw_card)
        required = card.get("required_facets")
        matched = card.get("matched_required_facets")
        uncovered = card.get("uncovered_required_facets")
        list_schema_valid = all(
            isinstance(values, list)
            and all(isinstance(value, str) and value.strip() for value in values)
            and len(values) == len(set(values))
            for values in (required, matched, uncovered)
        )
        required_set = set(required or [])
        matched_set = set(matched or [])
        uncovered_set = set(uncovered or [])
        partition_valid = (
            list_schema_valid
            and matched_set.issubset(required_set)
            and uncovered_set.issubset(required_set)
            and not matched_set.intersection(uncovered_set)
            and matched_set.union(uncovered_set) == required_set
        )
        counts_valid = (
            type(card.get("matched_required_facet_count")) is int
            and card.get("matched_required_facet_count")
            == len(matched_set)
            and type(card.get("uncovered_required_facet_count"))
            is int
            and card.get("uncovered_required_facet_count")
            == len(uncovered_set)
            and card.get("priority_prefix") == [len(uncovered_set)]
        )
        schema_valid = (
            set(card) == set(_CHUNK_FACET_PRIORITY_CARD_FIELDS)
            and candidate_id in candidate_set
            and card.get("chunk_id") == candidate_id
            and _is_hash(card.get("query_facet_packet_hash"))
            and partition_valid
            and counts_valid
        )
        audit.check(
            f"{card_scope}_schema",
            schema_valid,
            code="chunk_facet_priority_card_schema_mismatch",
            scope=card_scope,
        )
        card_protocol_valid = (
            card.get("protocol_version")
            == CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
            and card.get("protocol_hash") == expected_protocol_hash
            and card.get("facet_match_protocol_version")
            == QUERY_FACET_ORDERED_WINDOW_PROTOCOL_VERSION
            and card.get("facet_match_protocol_hash")
            == query_facet_ordered_window_protocol_hash()
        )
        audit.check(
            f"{card_scope}_protocol",
            card_protocol_valid,
            code="chunk_facet_priority_card_protocol_mismatch",
            scope=card_scope,
        )
        authority_valid = (
            card.get("lexical_overlap_used_as_numeric_relevance")
            is False
            and card.get("is_evidence") is False
            and card.get("citation_authority") is False
            and card.get("gray_zone_decision_authority") is False
            and card.get("model_call_count") == 0
        )
        audit.check(
            f"{card_scope}_authority",
            authority_valid,
            code="chunk_facet_priority_authority_violation",
            scope=card_scope,
        )
        supplied_hash = card.pop("card_hash", None)
        hash_valid = (
            _is_hash(supplied_hash)
            and supplied_hash == stable_hash(card)
        )
        audit.check(
            f"{card_scope}_hash",
            hash_valid,
            code="chunk_facet_priority_card_hash_mismatch",
            scope=card_scope,
        )
        card_valid = (
            schema_valid
            and card_protocol_valid
            and authority_valid
            and hash_valid
        )
        all_cards_valid = all_cards_valid and card_valid
        if card_valid:
            uncovered_counts[candidate_id] = int(
                card["uncovered_required_facet_count"]
            )
    use_priority = (
        protocol_valid
        and coverage_valid
        and all_cards_valid
        and set(uncovered_counts) == candidate_set
        and posterior_snapshot_valid
        and posterior_mass_replay_valid
    )
    return use_priority, uncovered_counts, posterior_masses


def _audit_candidate_pool(audit: _Audit, scope: str, pool: Mapping[str, Any]) -> None:
    candidates = _unique_strings(pool.get("candidate_ids"))
    selected = _unique_strings(pool.get("selected_ids"))
    forced = _unique_strings(pool.get("forced_candidate_ids"))
    candidate_scores = pool.get("candidate_scores")
    candidate_scores = (
        candidate_scores if isinstance(candidate_scores, Mapping) else {}
    )
    score_facts_valid = (
        set(str(key) for key in candidate_scores) == set(candidates)
        and all(
            _finite(value) for value in candidate_scores.values()
        )
    )
    (
        use_facet_priority,
        facet_uncovered_counts,
        facet_posterior_masses,
    ) = (
        _audit_chunk_facet_priority_candidate_pool(
            audit,
            scope=scope,
            pool=pool,
            candidates=candidates,
        )
    )
    use_rq_role_tie_break, rq_role_tie_breaks = (
        _audit_query_rq_candidate_pool(
            audit,
            scope=scope,
            pool=pool,
            candidates=candidates,
            candidate_scores=candidate_scores,
            forced_candidate_ids=forced,
        )
    )
    expected_candidates = (
        sorted(
            candidates,
            key=lambda candidate_id: (
                *(
                    (facet_uncovered_counts[candidate_id],)
                    if use_facet_priority
                    else ()
                ),
                *(
                    (-facet_posterior_masses[candidate_id],)
                    if use_facet_priority
                    else ()
                ),
                -float(candidate_scores[candidate_id]),
                *(
                    (
                        rq_role_tie_breaks.get(
                            candidate_id,
                            len(
                                QUERY_RQ_MEMBERSHIP_ROLE_TIE_BREAK
                            )
                            + 2,
                        ),
                    )
                    if use_rq_role_tie_break
                    else ()
                ),
                candidate_id,
            ),
        )
        if score_facts_valid
        else []
    )
    if use_rq_role_tie_break:
        audit.check(
            f"{scope}_query_rq_candidate_order_replayed",
            score_facts_valid and candidates == expected_candidates,
            code="query_rq_seed_candidate_order_mismatch",
            scope=scope,
            expected=expected_candidates,
            actual=candidates,
        )
    audit.check(
        f"{scope}_candidate_ids_unique",
        len(candidates) == len(set(candidates))
        and score_facts_valid
        and candidates == expected_candidates,
        code="candidate_pool_duplicate_id",
        scope=scope,
        expected=expected_candidates,
        actual=candidates,
    )
    audit.check(
        f"{scope}_selected_subset_replayed",
        set(selected).issubset(candidates) and set(forced).issubset(candidates) and set(forced).issubset(selected),
        code="candidate_pool_selection_out_of_scope",
        scope=scope,
        expected={"selected_subset": candidates, "forced_selected": forced},
        actual={"selected": selected},
    )
    top_k = pool.get("top_k")
    status = pool.get("per_parent_budget_status")
    status = status if isinstance(status, Mapping) else {}
    selection_budget = (
        top_k
        if top_k is not None
        else (
            status.get("budget")
            if pool.get("per_parent_budget_status") is not None
            else len(candidates)
        )
    )
    ranked_selected = _unique_strings(
        pool.get("ranked_selected_ids")
        if pool.get("ranked_selected_ids") is not None
        else selected
    )
    expected_ranked_selected: list[str] = []
    if type(selection_budget) is int and selection_budget >= 0:
        expected_ranked_selected = [
            *forced,
            *[
                candidate_id
                for candidate_id in expected_candidates
                if candidate_id not in set(forced)
            ][: max(0, int(selection_budget) - len(forced))],
        ]
    audit.check(
        f"{scope}_candidate_score_order_replayed",
        type(selection_budget) is int
        and selection_budget >= 0
        and len(forced) <= selection_budget
        and ranked_selected == expected_ranked_selected
        and set(selected).issubset(set(ranked_selected)),
        code="candidate_pool_score_order_mismatch",
        scope=scope,
        expected=expected_ranked_selected,
        actual={
            "selected_ids": selected,
            "ranked_selected_ids": ranked_selected,
        },
    )
    if top_k is not None:
        audit.check(
            f"{scope}_top_k_budget",
            type(top_k) is int and top_k >= 0 and len(selected) <= top_k,
            code="candidate_pool_top_k_exceeded",
            scope=scope,
            expected=top_k,
            actual=len(selected),
        )
    if pool.get("per_parent_budget_status") is not None:
        budget = status.get("budget")
        expected_stop = (
            "per_parent_budget_hit"
            if len(candidates) > len(ranked_selected)
            else "parent_candidates_exhausted"
        )
        audit.check(
            f"{scope}_per_parent_budget_replayed",
            type(budget) is int
            and budget >= 0
            and len(ranked_selected) <= budget
            and status.get("candidate_count") == len(candidates)
            and status.get("selected_count") == len(ranked_selected)
            and status.get("stop_reason") == expected_stop,
            code="per_parent_budget_status_mismatch",
            scope=scope,
            expected={
                "candidate_count": len(candidates),
                "selected_count": len(ranked_selected),
                "stop_reason": expected_stop,
            },
            actual=status,
        )
    dedupe = pool.get("candidate_dedupe_budget_audit")
    if dedupe is not None:
        dedupe = dedupe if isinstance(dedupe, Mapping) else {}
        attempt_count = dedupe.get("attempt_count")
        admitted_count = dedupe.get("unique_admitted_count")
        duplicate_count = dedupe.get("duplicate_count")
        rejected_count = dedupe.get("rejected_new_count")
        audit.check(
            f"{scope}_dedupe_budget_replayed",
            all(type(value) is int and value >= 0 for value in (attempt_count, admitted_count, duplicate_count, rejected_count))
            and attempt_count == admitted_count + duplicate_count + rejected_count
            and admitted_count == len(candidates),
            code="candidate_dedupe_budget_mismatch",
            scope=scope,
            actual=dedupe,
        )


def _all_candidate_pools(candidate_pools: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    pools: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in candidate_pools.items():
        if key == "candidate_dedupe_budget":
            continue
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    parent = item.get("parent_node_id") or index
                    pools.append((f"{key}.{parent}", item))
        elif isinstance(value, Mapping):
            pools.append((key, value))
    return pools


def retrieval_snapshot_from_records(trace: Any, steps: Sequence[Any]) -> dict[str, Any]:
    """Normalize persisted ORM-like trace records without importing application code."""

    diagnostics = getattr(trace, "diagnostics_json", None) or {}
    return {
        "trace_id": getattr(trace, "id", None),
        # Gray-zone replay is derived from the immutable raw request query.
        # Do not substitute query_facets_json here: that packet may contain
        # LLM/Profile/conversation routing output and is outside gray authority.
        "query": str(getattr(trace, "query", None) or ""),
        "result_chunk_ids": getattr(trace, "result_chunk_ids_json", None) or [],
        "runtime_settings_hash": getattr(trace, "runtime_settings_hash", None),
        "agent_operating_envelope_hash": getattr(
            trace, "agent_operating_envelope_hash", None
        ),
        "edge_distance_protocol_hash": getattr(trace, "edge_distance_protocol_hash", None),
        "edge_projection_protocol_hash": getattr(trace, "edge_projection_protocol_hash", None),
        "traversal_protocol_hash": getattr(trace, "traversal_protocol_hash", None),
        "retrieval_granularity": diagnostics.get("retrieval_granularity"),
        "query_facets": getattr(trace, "query_facets_json", None) or {},
        "stage_queues": getattr(trace, "stage_queues_json", None) or {},
        "candidate_pools": getattr(trace, "candidate_pools_json", None) or {},
        "topk_selection": getattr(trace, "topk_selection_json", None) or {},
        "path_labels": getattr(trace, "path_labels_json", None) or [],
        "trace_diagnostics": diagnostics,
        "convergence": getattr(trace, "convergence_json", None) or {},
        "steps": [
            {
                "step_index": getattr(step, "step_index", None),
                "layer": getattr(step, "layer", None),
                "action": getattr(step, "action", None),
                "action_type": getattr(step, "action_type", None),
                "parent_layer": getattr(step, "parent_layer", None),
                "parent_node_id": getattr(step, "parent_node_id", None),
                "input": getattr(step, "input_json", None) or {},
                "output": getattr(step, "output_json", None) or {},
                "popped_frontier_state": getattr(step, "popped_frontier_state_json", None) or {},
                "expanded_edge_ids": getattr(step, "expanded_edge_ids_json", None) or [],
                "candidate_pool_ids": getattr(step, "candidate_pool_ids_json", None) or [],
                "selected_topk_ids": getattr(step, "selected_topk_ids_json", None) or [],
                "dominance_pruned_count": getattr(step, "dominance_pruned_count", None),
                "cycle_distance_reward": getattr(step, "cycle_distance_reward", None),
                "gray_zone_path_decisions": getattr(step, "gray_zone_path_decisions_json", None) or [],
                "per_parent_budget_status": getattr(step, "per_parent_budget_status_json", None) or {},
                "stop_reason": getattr(step, "stop_reason", None),
                "diagnostics": getattr(step, "diagnostics_json", None) or {},
            }
            for step in steps
        ],
    }


_GRAY_ZONE_AUDIT_MODULE: Any | None = None


def _local_gray_zone_audit(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the authoritative deterministic audit from raw persisted trace facts."""

    global _GRAY_ZONE_AUDIT_MODULE
    if _GRAY_ZONE_AUDIT_MODULE is None:
        script_path = Path(__file__).with_name("_gray_zone_audit.py")
        spec = importlib.util.spec_from_file_location(
            "quality_gate_local_gray_zone_audit", script_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load deterministic gray-zone audit: {script_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GRAY_ZONE_AUDIT_MODULE = module
    result = _GRAY_ZONE_AUDIT_MODULE.audit_gray_zone_trace(
        dict(snapshot), require_gray_coverage=False
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("Deterministic gray-zone audit returned a non-object")
    return dict(result)


def _deterministic_gray_required_facets(
    snapshot: Mapping[str, Any],
) -> list[str]:
    """Replay only the closed gray-zone facets from the raw query."""

    global _GRAY_ZONE_AUDIT_MODULE
    if _GRAY_ZONE_AUDIT_MODULE is None:
        _local_gray_zone_audit(snapshot)
    application_contract = getattr(
        _GRAY_ZONE_AUDIT_MODULE,
        "APPLICATION_CONTRACT",
        None,
    )
    if not isinstance(application_contract, Mapping):
        return []
    facet_builder = application_contract.get("gray_query_facets")
    if not callable(facet_builder):
        return []
    packet = facet_builder(str(snapshot.get("query") or ""))
    if not isinstance(packet, Mapping):
        return []
    return sorted(
        {
            str(value)
            for value in packet.get("required_facets") or []
            if isinstance(value, str) and value
        }
    )


def _active_traversal_required_facets(
    snapshot: Mapping[str, Any],
) -> list[str]:
    """Replay ordinary queue labels from the frozen validated facet packet.

    These facets may influence entry/queue coverage, but they never enter the
    independent deterministic gray-zone observation or decision contract.
    """

    packet = snapshot.get("query_facets")
    if not isinstance(packet, Mapping):
        return []
    required = packet.get("required_facets")
    if not isinstance(required, list):
        return []
    return sorted(
        {
            value
            for value in required
            if isinstance(value, str) and value
        }
    )


def _complete_gray_zone_audit_valid(value: Mapping[str, Any]) -> bool:
    application_contract = value.get("application_contract")
    application_contract = (
        application_contract
        if isinstance(application_contract, Mapping)
        else {}
    )
    count_checks = value.get("count_checks")
    count_checks = count_checks if isinstance(count_checks, list) else []
    hash_checks = value.get("hash_replay_checks")
    hash_checks = hash_checks if isinstance(hash_checks, list) else []
    determinism = value.get("determinism")
    determinism = determinism if isinstance(determinism, Mapping) else {}
    raw_count = value.get("raw_record_count")
    gray_count = value.get("gray_rule_record_count")
    red_count = value.get("red_partition_record_count")
    hard_count = value.get("hard_stop_partition_record_count")
    expected_fields = set(
        getattr(_GRAY_ZONE_AUDIT_MODULE, "COMMON_REQUIRED_FIELDS", ())
    )
    return (
        value.get("pass") is True
        and value.get("status") in {"pass", "coverage_gap"}
        and value.get("require_gray_coverage") is False
        and value.get("trace_count") == 1
        and type(raw_count) is int
        and raw_count >= 0
        and all(
            type(count) is int and count >= 0
            for count in (gray_count, red_count, hard_count)
        )
        and raw_count == gray_count + red_count + hard_count
        and value.get("explicit_zero_model_call_record_count") == raw_count
        and value.get("raw_duplicate_event_count") == 0
        and application_contract.get("status") == "pass"
        and application_contract.get("import_error") is None
        and set(application_contract.get("script_required_fields") or [])
        == expected_fields
        and set(application_contract.get("application_required_fields") or [])
        == expected_fields
        and value.get("incomplete_trace_count") == 0
        and value.get("incomplete_record_count") == 0
        and value.get("incomplete_traces") == []
        and value.get("incomplete_records") == []
        and value.get("violations") == []
        and len(count_checks) == 5
        and all(
            isinstance(item, Mapping) and item.get("pass") is True
            for item in count_checks
        )
        and len(hash_checks) == raw_count
        and all(
            isinstance(item, Mapping)
            and all(
                item.get(field) is True
                for field in (
                    "protocol_hash",
                    "threshold_hash",
                    "input_hash",
                    "decision_hash",
                    "application_contract",
                )
            )
            for item in hash_checks
        )
        and determinism.get("status") == "pass"
        and determinism.get("conflict_count") == 0
        and determinism.get("conflicts") == []
        and determinism.get("raw_duplicate_event_count") == 0
    )


def _audit_topk_ranking(
    audit: _Audit,
    *,
    layer: str,
    selection: Mapping[str, Any],
    raw_rank_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    repair_directive: Mapping[str, Any] | None = None,
) -> None:
    selected = _unique_strings(selection.get("selected_ids"))
    forced = _unique_strings(selection.get("forced_selected_ids"))
    facts = selection.get("candidate_rank_facts")
    facts = facts if isinstance(facts, list) else []
    candidate_ids: list[str] = []
    overlay_ids: list[str] = []
    rank_by_id: dict[str, tuple[tuple[float, ...], str]] = {}
    valid = (
        selection.get("ranking_protocol_version")
        == TOPK_RANKING_PROTOCOL_VERSION
    )
    for fact in facts:
        if not isinstance(fact, Mapping):
            valid = False
            continue
        candidate_id = str(fact.get("candidate_id") or "")
        rank_key = fact.get("rank_key")
        tie_break = fact.get("path_identity")
        if (
            not candidate_id
            or candidate_id in rank_by_id
            or not isinstance(rank_key, (list, tuple))
            or not rank_key
            or not all(_finite(value) for value in rank_key)
            or not isinstance(tie_break, str)
            or not tie_break
        ):
            valid = False
            continue
        repair_fields_present = bool(
            {
                "repair_evidence_retention_protocol_version",
                "source_context_package_id",
                "source_retrieval_trace_id",
                "repair_directive_hash",
            }.intersection(fact)
        )
        if repair_fields_present:
            if not _repair_carry_forward_rank_fact_valid(
                fact,
                repair_directive=repair_directive,
            ):
                valid = False
            overlay_ids.append(candidate_id)
        elif set(fact) != {
            "candidate_id",
            "rank_key",
            "path_identity",
        }:
            valid = False
        candidate_ids.append(candidate_id)
        rank_by_id[candidate_id] = (
            tuple(float(value) for value in rank_key),
            tie_break,
        )
    ranked = (
        sorted(
            candidate_ids,
            key=lambda candidate_id: (
                rank_by_id[candidate_id][0],
                rank_by_id[candidate_id][1],
                candidate_id,
            ),
        )
        if valid
        else []
    )
    top_k = selection.get("top_k")
    declared_carry_ids_raw = selection.get(
        "carry_forward_supported_chunk_ids"
    )
    declared_carry_ids_raw = (
        declared_carry_ids_raw
        if isinstance(declared_carry_ids_raw, list)
        else []
    )
    declared_carry_ids = _unique_strings(declared_carry_ids_raw)
    carry_binding_valid = (
        all(
            isinstance(value, str) and value
            for value in declared_carry_ids_raw
        )
        and declared_carry_ids_raw
        == sorted(set(declared_carry_ids_raw))
        and sorted(overlay_ids) == declared_carry_ids
        and set(declared_carry_ids).issubset(set(forced))
        and (
            not declared_carry_ids
            or selection.get("global_top_k_increased") is False
        )
    )
    expected = (
        [
            *forced,
            *[
                candidate_id
                for candidate_id in ranked
                if candidate_id not in set(forced)
            ][: max(0, int(top_k) - len(forced))],
        ]
        if type(top_k) is int and top_k >= len(forced)
        else []
    )
    audit.check(
        f"{layer}_topk_frozen_rank_replayed",
        valid
        and carry_binding_valid
        and type(top_k) is int
        and top_k >= 0
        and set(forced).issubset(set(candidate_ids))
        and isinstance(raw_rank_by_id, Mapping)
        and all(
            (
                candidate_id in set(overlay_ids)
                and _repair_carry_forward_rank_fact_valid(
                    facts[index],
                    repair_directive=repair_directive,
                )
            )
            or (
                candidate_id not in set(overlay_ids)
                and candidate_id in raw_rank_by_id
                and list(
                    raw_rank_by_id[candidate_id].get("rank_key") or []
                )
                == list(facts[index].get("rank_key") or [])
                and raw_rank_by_id[candidate_id].get("path_identity")
                == facts[index].get("path_identity")
            )
            for index, candidate_id in enumerate(candidate_ids)
        )
        and selection.get("candidate_count") == len(candidate_ids)
        and selected == expected,
        code="topk_frozen_score_order_mismatch",
        scope=layer,
        expected=expected,
        actual={
            "selected_ids": selected,
            "candidate_rank_facts": facts,
        },
    )


def _repair_directive_binding_valid(
    repair_directive: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(repair_directive, Mapping):
        return False
    validated_hash = repair_directive.get("validated_directive_hash")
    source_context_package_id = repair_directive.get(
        "source_context_package_id"
    )
    source_retrieval_trace_id = repair_directive.get(
        "source_retrieval_trace_id"
    )
    carry_ids = repair_directive.get(
        "carry_forward_supported_chunk_ids"
    )
    return (
        repair_directive.get("protocol_version") == "typed_repair_loop_v1"
        and _is_hash(validated_hash)
        and validated_hash
        == stable_hash(
            {
                key: value
                for key, value in repair_directive.items()
                if key != "validated_directive_hash"
            }
        )
        and isinstance(source_context_package_id, str)
        and bool(source_context_package_id)
        and isinstance(source_retrieval_trace_id, str)
        and bool(source_retrieval_trace_id)
        and isinstance(carry_ids, list)
        and all(isinstance(value, str) and value for value in carry_ids)
        and carry_ids == sorted(set(carry_ids))
    )


def _repair_carry_forward_rank_fact_valid(
    fact: Mapping[str, Any],
    *,
    repair_directive: Mapping[str, Any] | None,
) -> bool:
    if not _repair_directive_binding_valid(repair_directive):
        return False
    expected_keys = {
        "candidate_id",
        "rank_key",
        "path_identity",
        "repair_evidence_retention_protocol_version",
        "source_context_package_id",
        "source_retrieval_trace_id",
        "repair_directive_hash",
    }
    card = {
        key: fact.get(key)
        for key in expected_keys
        if key != "path_identity"
    }
    candidate_id = fact.get("candidate_id")
    return (
        set(fact) == expected_keys
        and fact.get("repair_evidence_retention_protocol_version")
        == REPAIR_EVIDENCE_RETENTION_PROTOCOL_VERSION
        and fact.get("rank_key") == [-1.0]
        and isinstance(candidate_id, str)
        and bool(candidate_id)
        and candidate_id
        in set(
            repair_directive.get(
                "carry_forward_supported_chunk_ids"
            )
            or []
        )
        and fact.get("source_context_package_id")
        == repair_directive.get("source_context_package_id")
        and fact.get("source_retrieval_trace_id")
        == repair_directive.get("source_retrieval_trace_id")
        and fact.get("repair_directive_hash")
        == repair_directive.get("validated_directive_hash")
        and fact.get("path_identity") == stable_hash(card)
    )


def _label_fact_valid(label: Mapping[str, Any]) -> bool:
    multiset = label.get("path_edge_type_multiset")
    return (
        _finite(label.get("distance"))
        and float(label["distance"]) >= 0.0
        and isinstance(label.get("covered_facets"), list)
        and isinstance(label.get("evidence_roles"), list)
        and type(label.get("depth")) is int
        and int(label["depth"]) >= 0
        and isinstance(multiset, Mapping)
        and all(
            isinstance(key, str)
            and key
            and type(value) is int
            and value >= 0
            for key, value in multiset.items()
        )
    )


def _label_fact_dominates(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    left_multiset = dict(left.get("path_edge_type_multiset") or {})
    right_multiset = dict(right.get("path_edge_type_multiset") or {})
    left_facets = set(left.get("covered_facets") or [])
    right_facets = set(right.get("covered_facets") or [])
    left_roles = set(left.get("evidence_roles") or [])
    right_roles = set(right.get("evidence_roles") or [])
    weakly_better = (
        float(left["distance"]) <= float(right["distance"])
        and left_facets.issuperset(right_facets)
        and left_roles.issuperset(right_roles)
        and int(left["depth"]) <= int(right["depth"])
        and all(
            int(left_multiset.get(edge_type, 0)) >= int(count)
            for edge_type, count in right_multiset.items()
        )
    )
    strictly_better = (
        float(left["distance"]) < float(right["distance"])
        or left_facets != right_facets
        or left_roles != right_roles
        or int(left["depth"]) < int(right["depth"])
        or left_multiset != right_multiset
    )
    return weakly_better and strictly_better


def _frontier_state_path_identity(state: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "layer": state.get("layer"),
            "node_id": str(state.get("node_id") or ""),
            "root_node_id": state.get("root_node_id"),
            "path": list(state.get("path") or []),
            "path_edge_ids": list(
                state.get("path_edge_ids") or []
            ),
            "entry_parent_refs": list(
                state.get("entry_parent_refs") or []
            ),
        }
    )


def _frontier_state_label_fact(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "distance": round(
            float(state.get("distance_so_far") or 0.0), 6
        ),
        "covered_facets": sorted(
            {
                str(value)
                for value in state.get("covered_facets") or []
            }
        ),
        "evidence_roles": sorted(
            {
                str(value)
                for value in state.get("evidence_roles") or []
            }
        ),
        "depth": int(state.get("depth") or 0),
        "path_edge_type_multiset": dict(
            sorted(
                Counter(
                    str(value)
                    for value in state.get("path_edge_types") or []
                    if str(value)
                ).items()
            )
        ),
    }


def _frontier_state_structure_valid(
    state: Mapping[str, Any],
    *,
    require_edge_metrics: bool,
) -> bool:
    path = state.get("path")
    path_edge_ids = state.get("path_edge_ids")
    path_edge_types = state.get("path_edge_types")
    if not (
        isinstance(state.get("layer"), str)
        and bool(state.get("layer"))
        and isinstance(state.get("node_id"), str)
        and bool(state.get("node_id"))
        and isinstance(state.get("root_node_id"), str)
        and bool(state.get("root_node_id"))
        and isinstance(path, list)
        and bool(path)
        and all(isinstance(value, str) and value for value in path)
        and isinstance(path_edge_ids, list)
        and all(
            isinstance(value, str) and value for value in path_edge_ids
        )
        and isinstance(path_edge_types, list)
        and all(
            isinstance(value, str) and value
            for value in path_edge_types
        )
        and len(path) == len(path_edge_ids) + 1
        and len(path_edge_types) == len(path_edge_ids)
        and path[-1] == state.get("node_id")
        and path[0] == state.get("root_node_id")
        and type(state.get("depth")) is int
        and state.get("depth") == len(path_edge_ids)
        and _finite(state.get("distance_so_far"))
        and float(state["distance_so_far"]) >= 0.0
        and _finite(state.get("reward_so_far"))
        and float(state["reward_so_far"]) >= 0.0
        and isinstance(state.get("covered_facets"), list)
        and isinstance(state.get("evidence_roles"), list)
        and isinstance(state.get("entry_parent_refs"), list)
    ):
        return False
    for field_name in ("covered_facets", "evidence_roles"):
        values = state.get(field_name) or []
        if (
            any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            return False
    for field_name, lower, upper in (
        ("path_edge_distances", 0.0, None),
        ("path_edge_strengths", 0.0, 1.0),
    ):
        values = state.get(field_name)
        if values is None and not require_edge_metrics:
            continue
        if not (
            isinstance(values, list)
            and len(values) == len(path_edge_ids)
            and all(
                _finite(value)
                and float(value) >= lower
                and (upper is None or float(value) <= upper)
                for value in values
            )
        ):
            return False
    return True


def _candidate_state_matches_frontier_state(
    candidate: Mapping[str, Any], popped: Mapping[str, Any]
) -> bool:
    if not (
        _frontier_state_structure_valid(
            candidate, require_edge_metrics=False
        )
        and _frontier_state_structure_valid(
            popped, require_edge_metrics=True
        )
    ):
        return False
    exact_fields = (
        "layer",
        "node_id",
        "root_node_id",
        "path",
        "path_edge_ids",
        "path_edge_types",
        "covered_facets",
        "evidence_roles",
        "depth",
        "entry_parent_refs",
    )
    if any(candidate.get(field) != popped.get(field) for field in exact_fields):
        return False
    if not (
        _strict_numeric_match(
            candidate.get("distance_so_far"),
            popped.get("distance_so_far"),
        )
        and _strict_numeric_match(
            candidate.get("reward_so_far"),
            popped.get("reward_so_far"),
        )
    ):
        return False
    for field_name in ("path_edge_distances", "path_edge_strengths"):
        candidate_values = candidate.get(field_name)
        if candidate_values is None:
            continue
        popped_values = popped.get(field_name)
        if not (
            isinstance(candidate_values, list)
            and isinstance(popped_values, list)
            and len(candidate_values) == len(popped_values)
            and all(
                _strict_numeric_match(left, right)
                for left, right in zip(candidate_values, popped_values)
            )
        ):
            return False
    return True


def _path_label_matches_frontier_state(
    label: Mapping[str, Any],
    popped: Mapping[str, Any],
    *,
    layer: str,
) -> bool:
    path = label.get("path")
    path_edge_ids = label.get("path_edge_ids")
    path_edge_distances = label.get("path_edge_distances")
    path_edge_strengths = label.get("path_edge_strengths")
    target_ids = [
        str(label.get(field_name))
        for field_name in ("node_id", "chunk_id")
        if label.get(field_name)
    ]
    if not (
        target_ids
        and len(set(target_ids)) == 1
        and target_ids[0] == popped.get("node_id")
        and (label.get("layer") in {None, "", layer})
        and isinstance(path, list)
        and path == popped.get("path")
        and isinstance(path_edge_ids, list)
        and path_edge_ids == popped.get("path_edge_ids")
        and isinstance(path_edge_distances, list)
        and isinstance(popped.get("path_edge_distances"), list)
        and len(path_edge_distances)
        == len(popped.get("path_edge_distances") or [])
        and all(
            _strict_numeric_match(left, right)
            for left, right in zip(
                path_edge_distances,
                popped.get("path_edge_distances") or [],
            )
        )
        and isinstance(path_edge_strengths, list)
        and isinstance(popped.get("path_edge_strengths"), list)
        and len(path_edge_strengths)
        == len(popped.get("path_edge_strengths") or [])
        and all(
            _strict_numeric_match(left, right)
            for left, right in zip(
                path_edge_strengths,
                popped.get("path_edge_strengths") or [],
            )
        )
        and _strict_numeric_match(
            label.get("distance_so_far"),
            popped.get("distance_so_far"),
        )
        and _strict_numeric_match(
            label.get("reward_so_far"),
            popped.get("reward_so_far"),
        )
        and sorted(label.get("covered_facets") or [])
        == sorted(popped.get("covered_facets") or [])
        and sorted(label.get("evidence_roles") or [])
        == sorted(popped.get("evidence_roles") or [])
        and (
            label.get("root_node_id") in {None, ""}
            or label.get("root_node_id") == popped.get("root_node_id")
        )
    ):
        return False
    if (
        "path_edge_type_multiset" in label
        and dict(label.get("path_edge_type_multiset") or {})
        != dict(
            sorted(
                Counter(
                    str(value)
                    for value in popped.get("path_edge_types") or []
                    if str(value)
                ).items()
            )
        )
    ):
        return False
    return (
        isinstance(label.get("expanded_edge_ids"), list)
        and all(
            isinstance(value, str) and value
            for value in label.get("expanded_edge_ids") or []
        )
        and len(label.get("expanded_edge_ids") or [])
        == len(set(label.get("expanded_edge_ids") or []))
    )


def _repair_carry_forward_path_label_valid(
    label: Mapping[str, Any],
    *,
    layer: str,
    repair_directive: Mapping[str, Any] | None,
) -> bool:
    chunk_id = label.get("chunk_id")
    return (
        label.get("repair_evidence_retention_protocol_version")
        == REPAIR_EVIDENCE_RETENTION_PROTOCOL_VERSION
        and _repair_directive_binding_valid(repair_directive)
        and layer == "chunk"
        and isinstance(chunk_id, str)
        and bool(chunk_id)
        and chunk_id
        in set(
            repair_directive.get(
                "carry_forward_supported_chunk_ids"
            )
            or []
        )
        and label.get("node_id") == chunk_id
        and label.get("root_node_id") == chunk_id
        and label.get("path") == [chunk_id]
        and label.get("path_edge_ids") == []
        and label.get("path_edge_types") == []
        and label.get("path_edge_distances") == []
        and label.get("path_edge_strengths") == []
        and label.get("cycle_distance_rewards") == []
        and label.get("expanded_edge_ids") == []
        and label.get("entry_parent_refs") == []
        and label.get("support_refs")
        == {"support_chunk_ids": [chunk_id]}
        and label.get("evidence_roles")
        == ["prior_supported_claim_carry_forward"]
        and isinstance(label.get("covered_facets"), list)
        and all(
            isinstance(value, str) and value
            for value in label.get("covered_facets") or []
        )
        and label.get("covered_facets")
        == sorted(set(label.get("covered_facets") or []))
        and _strict_numeric_match(label.get("distance_so_far"), 0.0)
        and _strict_numeric_match(label.get("reward_so_far"), 0.0)
        and label.get("stop_reason")
        == "repair_supported_evidence_carry_forward"
        and label.get("source_context_package_id")
        == repair_directive.get("source_context_package_id")
        and label.get("source_retrieval_trace_id")
        == repair_directive.get("source_retrieval_trace_id")
        and label.get("repair_directive_hash")
        == repair_directive.get("validated_directive_hash")
    )


def _replayed_frontier_queue_key(
    state: Mapping[str, Any],
    required_facets: Sequence[str],
) -> tuple[float, float, float, float]:
    label = _frontier_state_label_fact(state)
    return (
        float(
            len(
                set(required_facets)
                - set(label.get("covered_facets") or [])
            )
        ),
        round(
            float(label.get("distance") or 0.0)
            - float(state.get("reward_so_far") or 0.0),
            6,
        ),
        float(label.get("depth") or 0),
        float(-len(label.get("evidence_roles") or [])),
    )


def _strict_frontier_key(
    value: Any,
) -> tuple[float, float, float, float] | None:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_finite(item) for item in value)
    ):
        return None
    return tuple(float(item) for item in value)


def _replay_frontier_path_label_closure(
    observations: Sequence[Any],
    *,
    frontier_enqueues: Sequence[Any],
    initial_entry_ids: Sequence[Any],
    frontier_json: Sequence[Any],
    path_labels: Sequence[Any],
    step_expanded_edge_ids: Sequence[Any],
    popped_frontier_state: Mapping[str, Any],
    terminal_facts: Mapping[str, Any],
    required_facets: Sequence[str],
    layer: str,
    repair_directive: Mapping[str, Any] | None = None,
) -> bool:
    if not (
        isinstance(layer, str)
        and bool(layer)
        and all(isinstance(item, Mapping) for item in observations)
        and isinstance(frontier_enqueues, list)
        and all(
            isinstance(item, Mapping)
            for item in frontier_enqueues
        )
        and isinstance(initial_entry_ids, list)
        and all(
            isinstance(item, str) and item
            for item in initial_entry_ids
        )
        and len(initial_entry_ids) == len(set(initial_entry_ids))
        and all(isinstance(item, Mapping) for item in frontier_json)
        and all(isinstance(item, Mapping) for item in path_labels)
        and isinstance(step_expanded_edge_ids, list)
        and all(
            isinstance(value, str) and value
            for value in step_expanded_edge_ids
        )
        and len(step_expanded_edge_ids)
        == len(set(step_expanded_edge_ids))
        and isinstance(popped_frontier_state, Mapping)
        and isinstance(terminal_facts, Mapping)
        and isinstance(required_facets, list)
        and required_facets
        == sorted(set(str(item) for item in required_facets))
    ):
        return False
    active_labels: list[Mapping[str, Any]] = []
    for label in path_labels:
        if {
            "repair_evidence_retention_protocol_version",
            "source_context_package_id",
            "source_retrieval_trace_id",
            "repair_directive_hash",
        }.intersection(label):
            if not _repair_carry_forward_path_label_valid(
                label,
                layer=layer,
                repair_directive=repair_directive,
            ):
                return False
            continue
        active_labels.append(label)
    frontier_rows = list(frontier_json)
    if len(active_labels) != len(frontier_rows):
        return False
    if frontier_rows:
        first_popped = frontier_rows[0].get("popped")
        if (
            not isinstance(first_popped, Mapping)
            or dict(popped_frontier_state) != dict(first_popped)
        ):
            return False
    elif dict(popped_frontier_state):
        return False
    enqueue_event_keys = {
        "protocol_version",
        "enqueue_kind",
        "serial",
        "queue_key",
        "candidate_state",
        "edge_id",
        "parent_path_identity",
    }
    enqueue_events = list(frontier_enqueues)
    initial_entry_count = len(initial_entry_ids)
    enqueue_event_identities: set[str] = set()
    for index, event in enumerate(enqueue_events):
        candidate = event.get("candidate_state")
        queue_key = _strict_frontier_key(
            event.get("queue_key")
        )
        kind = event.get("enqueue_kind")
        if not (
            set(event) == enqueue_event_keys
            and event.get("protocol_version")
            == FRONTIER_HEAP_ENQUEUE_PROTOCOL_VERSION
            and type(event.get("serial")) is int
            and event.get("serial") == index
            and isinstance(candidate, Mapping)
            and _frontier_state_structure_valid(
                candidate, require_edge_metrics=False
            )
            and queue_key is not None
            and queue_key
            == _replayed_frontier_queue_key(
                candidate, required_facets
            )
        ):
            return False
        enqueue_identity = stable_hash(
            {
                "enqueue_kind": kind,
                "queue_key": list(queue_key),
                "candidate_state": dict(candidate),
                "edge_id": event.get("edge_id"),
                "parent_path_identity": event.get(
                    "parent_path_identity"
                ),
            }
        )
        if enqueue_identity in enqueue_event_identities:
            return False
        enqueue_event_identities.add(enqueue_identity)
        if index < initial_entry_count:
            if not (
                kind == "initial_entry"
                and event.get("edge_id") is None
                and event.get("parent_path_identity") is None
                and candidate.get("node_id")
                == initial_entry_ids[index]
                and candidate.get("path")
                == [initial_entry_ids[index]]
                and candidate.get("path_edge_ids") == []
                and candidate.get("depth") == 0
            ):
                return False
        elif not (
            kind == "expanded_edge"
            and isinstance(event.get("edge_id"), str)
            and bool(event.get("edge_id"))
            and _is_hash(event.get("parent_path_identity"))
            and list(candidate.get("path_edge_ids") or [])[-1:]
            == [event.get("edge_id")]
        ):
            return False
    if any(
        event.get("enqueue_kind") == "initial_entry"
        for event in enqueue_events[initial_entry_count:]
    ):
        return False
    if len(enqueue_events) < initial_entry_count:
        return False
    if not (
        terminal_facts.get("entry_count") == initial_entry_count
        and terminal_facts.get("frontier_expansion_count")
        == len(enqueue_events) - initial_entry_count
    ):
        return False
    root_entry_distances: dict[str, float] = {}
    declared_label_by_path: defaultdict[
        tuple[tuple[str, ...], tuple[str, ...]],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    available_frontier: list[
        tuple[
            tuple[float, float, float, float],
            int,
            int,
        ]
    ] = []
    for event_index in range(initial_entry_count):
        event = enqueue_events[event_index]
        heapq.heappush(
            available_frontier,
            (
                _strict_frontier_key(event.get("queue_key"))
                or (0.0, 0.0, 0.0, 0.0),
                int(event["serial"]),
                event_index,
            ),
        )
    next_enqueue_index = initial_entry_count
    active_label_index = 0
    for observation in observations:
        candidate = observation.get("candidate_state")
        admitted = observation.get("admitted")
        if (
            not isinstance(candidate, Mapping)
            or type(admitted) is not bool
            or not _frontier_state_structure_valid(
                candidate, require_edge_metrics=False
            )
        ):
            return False
        if not available_frontier:
            return False
        queue_size_before_pop = len(available_frontier)
        (
            expected_queue_key,
            expected_serial,
            expected_event_index,
        ) = heapq.heappop(available_frontier)
        expected_event = enqueue_events[expected_event_index]
        if not (
            dict(candidate)
            == dict(expected_event.get("candidate_state") or {})
            and _strict_frontier_key(
                observation.get("queue_key")
            )
            == expected_queue_key
            and type(observation.get("frontier_serial")) is int
            and observation.get("frontier_serial")
            == expected_serial
            and type(
                observation.get("queue_size_before_pop")
            )
            is int
            and observation.get("queue_size_before_pop")
            == queue_size_before_pop
            and type(
                observation.get("queue_size_after_pop")
            )
            is int
            and observation.get("queue_size_after_pop")
            == len(available_frontier)
        ):
            return False
        candidate_path = tuple(
            str(value) for value in candidate.get("path") or []
        )
        candidate_edge_ids = tuple(
            str(value)
            for value in candidate.get("path_edge_ids") or []
        )
        if candidate_edge_ids:
            prior_prefix_labels = declared_label_by_path.get(
                (
                    candidate_path[:-1],
                    candidate_edge_ids[:-1],
                )
            ) or []
            if not any(
                candidate_edge_ids[-1]
                in (prior.get("expanded_edge_ids") or [])
                for prior in prior_prefix_labels
            ):
                return False
        if not admitted:
            continue
        if active_label_index >= len(active_labels):
            return False
        label = active_labels[active_label_index]
        snapshot = frontier_rows[active_label_index]
        popped = snapshot.get("popped")
        if not (
            isinstance(popped, Mapping)
            and _candidate_state_matches_frontier_state(
                candidate, popped
            )
            and _strict_frontier_key(snapshot.get("key"))
            == expected_queue_key
            and snapshot.get("serial") == expected_serial
            and type(snapshot.get("serial")) is int
            and snapshot.get("queue_size_before_pop")
            == queue_size_before_pop
            and type(snapshot.get("queue_size_before_pop"))
            is int
            and snapshot.get("queue_size_after_pop")
            == len(available_frontier)
            and type(snapshot.get("queue_size_after_pop"))
            is int
            and _path_label_matches_frontier_state(
                label, popped, layer=layer
            )
        ):
            return False
        root_id = str(popped.get("root_node_id") or "")
        path = tuple(
            str(value) for value in popped.get("path") or []
        )
        edge_ids = tuple(
            str(value)
            for value in popped.get("path_edge_ids") or []
        )
        if edge_ids:
            predecessors = declared_label_by_path.get(
                (path[:-1], edge_ids[:-1])
            ) or []
            if not any(
                edge_ids[-1]
                in (item.get("expanded_edge_ids") or [])
                for item in predecessors
            ):
                return False
            expected_distance = root_entry_distances.get(root_id)
            if expected_distance is None:
                return False
        else:
            expected_distance = float(popped["distance_so_far"])
            prior_distance = root_entry_distances.get(root_id)
            if (
                prior_distance is not None
                and not _strict_numeric_match(
                    prior_distance, expected_distance
                )
            ):
                return False
            root_entry_distances[root_id] = expected_distance
        for edge_distance in popped.get("path_edge_distances") or []:
            expected_distance = round(
                expected_distance + float(edge_distance), 6
            )
        if not _strict_numeric_match(
            popped.get("distance_so_far"), expected_distance
        ):
            return False
        parent_path_identity = _frontier_state_path_identity(
            popped
        )
        for edge_id in label.get("expanded_edge_ids") or []:
            if next_enqueue_index >= len(enqueue_events):
                return False
            enqueue_event = enqueue_events[next_enqueue_index]
            enqueue_candidate = enqueue_event.get(
                "candidate_state"
            )
            enqueue_path = list(
                (enqueue_candidate or {}).get("path") or []
            )
            enqueue_edge_ids = list(
                (enqueue_candidate or {}).get(
                    "path_edge_ids"
                )
                or []
            )
            if not (
                enqueue_event.get("enqueue_kind")
                == "expanded_edge"
                and enqueue_event.get("edge_id") == edge_id
                and enqueue_event.get("parent_path_identity")
                == parent_path_identity
                and enqueue_path[:-1] == list(path)
                and enqueue_edge_ids[:-1] == list(edge_ids)
                and enqueue_edge_ids[-1:] == [edge_id]
            ):
                return False
            enqueue_key = _strict_frontier_key(
                enqueue_event.get("queue_key")
            )
            if enqueue_key is None:
                return False
            heapq.heappush(
                available_frontier,
                (
                    enqueue_key,
                    int(enqueue_event["serial"]),
                    next_enqueue_index,
                ),
            )
            next_enqueue_index += 1
        declared_label_by_path[(path, edge_ids)].append(label)
        active_label_index += 1
    if active_label_index != len(active_labels):
        return False
    expanded_edge_ids = list(
        dict.fromkeys(
            str(edge_id)
            for label in active_labels
            for edge_id in label.get("expanded_edge_ids") or []
        )
    )
    if expanded_edge_ids != list(step_expanded_edge_ids):
        return False
    frontier_remaining_count = terminal_facts.get(
        "frontier_remaining_count"
    )
    return (
        type(frontier_remaining_count) is int
        and frontier_remaining_count >= 0
        and next_enqueue_index == len(enqueue_events)
        and frontier_remaining_count == len(available_frontier)
    )


def _replay_dominance_observations(
    observations: Sequence[Any],
    *,
    frontier_json: Sequence[Any],
    required_facets: Sequence[str],
    max_labels: int,
) -> tuple[bool, int, int, dict[str, dict[str, Any]]]:
    configured_required_facets = sorted(
        {str(value) for value in required_facets}
    )
    configured_max_labels = max_labels
    retained: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    dominance_count = 0
    label_budget_count = 0
    all_valid = True
    for raw in observations:
        if not isinstance(raw, Mapping):
            all_valid = False
            continue
        node_id = str(raw.get("node_id") or "")
        label = raw.get("label")
        label = label if isinstance(label, Mapping) else {}
        candidate_state = raw.get("candidate_state")
        candidate_state = (
            candidate_state
            if isinstance(candidate_state, Mapping)
            else {}
        )
        queue_key = raw.get("queue_key")
        path_identity = raw.get("path_identity")
        observation_max_labels = raw.get("max_labels")
        observation_required_facets = raw.get("required_facets")
        if (
            raw.get("protocol_version")
            != DOMINANCE_REPLAY_PROTOCOL_VERSION
            or not node_id
            or not _frontier_state_structure_valid(
                candidate_state, require_edge_metrics=False
            )
            or not _label_fact_valid(label)
            or not isinstance(queue_key, (list, tuple))
            or not all(_finite(value) for value in queue_key)
            or not isinstance(path_identity, str)
            or not path_identity
            or type(observation_max_labels) is not int
            or observation_max_labels <= 0
            or observation_max_labels != configured_max_labels
            or not isinstance(observation_required_facets, list)
            or observation_required_facets
            != configured_required_facets
            or any(
                not isinstance(value, str)
                for value in observation_required_facets
            )
            or raw.get("node_id")
            != candidate_state.get("node_id")
            or dict(label)
            != _frontier_state_label_fact(candidate_state)
            or list(queue_key or [])
            != [
                float(
                    len(
                        set(configured_required_facets)
                        - set(label.get("covered_facets") or [])
                    )
                ),
                round(
                    float(label.get("distance") or 0.0)
                    - float(
                        candidate_state.get("reward_so_far")
                        or 0.0
                    ),
                    6,
                ),
                float(label.get("depth") or 0),
                float(
                    -len(label.get("evidence_roles") or [])
                ),
            ]
            or path_identity
            != _frontier_state_path_identity(candidate_state)
        ):
            all_valid = False
            continue
        entries = retained[node_id]
        if any(
            entry["path_identity"] == path_identity
            for entry in entries
        ):
            expected_admitted = False
            dominance_delta = 1
            label_budget_delta = 0
        elif any(
            _label_fact_dominates(entry["label"], label)
            for entry in entries
        ):
            expected_admitted = False
            dominance_delta = 1
            label_budget_delta = 0
        else:
            dominated = [
                entry
                for entry in entries
                if _label_fact_dominates(label, entry["label"])
            ]
            dominance_delta = len(dominated)
            if dominated:
                entries[:] = [
                    entry for entry in entries if entry not in dominated
                ]
            candidate = {
                "label": dict(label),
                "queue_key": tuple(float(value) for value in queue_key),
                "path_identity": path_identity,
            }
            entries.append(candidate)
            entries.sort(
                key=lambda entry: (
                    len(
                        set(observation_required_facets)
                        - set(
                            entry["label"].get(
                                "covered_facets"
                            )
                            or []
                        )
                    ),
                    float(entry["label"]["distance"]),
                    int(entry["label"]["depth"]),
                    -len(
                        set(
                            entry["label"].get(
                                "evidence_roles"
                            )
                            or []
                        )
                    ),
                    tuple(
                        sorted(
                            (
                                str(edge_type),
                                int(count),
                            )
                            for edge_type, count in dict(
                                entry["label"].get(
                                    "path_edge_type_multiset"
                                )
                                or {}
                            ).items()
                        )
                    ),
                    entry["queue_key"],
                    entry["path_identity"],
                )
            )
            removed = entries[int(observation_max_labels) :]
            del entries[int(observation_max_labels) :]
            label_budget_delta = len(removed)
            expected_admitted = candidate not in removed
        if (
            raw.get("admitted") is not expected_admitted
            or raw.get("dominance_pruned_delta") != dominance_delta
            or raw.get("label_budget_pruned_delta")
            != label_budget_delta
        ):
            all_valid = False
        dominance_count += dominance_delta
        label_budget_count += label_budget_delta
    admitted_observations = [
        item
        for item in observations
        if isinstance(item, Mapping) and item.get("admitted") is True
    ]
    frontier_rows = [
        item for item in frontier_json if isinstance(item, Mapping)
    ]
    if len(frontier_rows) != len(frontier_json):
        all_valid = False
    if len(admitted_observations) != len(frontier_rows):
        all_valid = False
    expected_required_facets = configured_required_facets
    for observation, snapshot in zip(
        admitted_observations, frontier_rows
    ):
        popped = snapshot.get("popped")
        popped = popped if isinstance(popped, Mapping) else {}
        label = _frontier_state_label_fact(popped)
        queue_key = [
            float(value)
            for value in list(snapshot.get("key") or [])
            if _finite(value)
        ]
        computed_queue_key = [
            float(
                len(
                    set(expected_required_facets)
                    - set(label["covered_facets"])
                )
            ),
            round(
                float(label["distance"])
                - float(popped.get("reward_so_far") or 0.0),
                6,
            ),
            float(label["depth"]),
            float(-len(label["evidence_roles"])),
        ]
        if not (
            _candidate_state_matches_frontier_state(
                observation.get("candidate_state") or {}, popped
            )
            and observation.get("node_id") == popped.get("node_id")
            and observation.get("label") == label
            and observation.get("required_facets")
            == expected_required_facets
            and observation.get("max_labels")
            == configured_max_labels
            and queue_key == computed_queue_key
            and list(observation.get("queue_key") or [])
            == computed_queue_key
            and observation.get("path_identity")
            == _frontier_state_path_identity(popped)
        ):
            all_valid = False
    rank_by_id: dict[str, dict[str, Any]] = {}
    for node_id, entries in retained.items():
        if not entries:
            continue
        best = min(
            entries,
            key=lambda entry: (
                entry["queue_key"], entry["path_identity"]
            ),
        )
        rank_by_id[node_id] = {
            "candidate_id": node_id,
            "rank_key": list(best["queue_key"]),
            "path_identity": best["path_identity"],
        }
    return (
        all_valid,
        dominance_count,
        label_budget_count,
        rank_by_id,
    )


def _cycle_event_distance_replayed(
    event: Mapping[str, Any],
    path_edge_ids: Sequence[Any],
    path_edge_distances: Sequence[Any],
) -> bool:
    cycle_edges = _unique_strings(event.get("cycle_edges"))
    if (
        not cycle_edges
        or len(path_edge_ids) != len(path_edge_distances)
        or not all(_finite(value) and float(value) >= 0.0 for value in path_edge_distances)
    ):
        return False
    for start in range(0, len(path_edge_ids) - len(cycle_edges) + 1):
        if [
            str(value)
            for value in path_edge_ids[start : start + len(cycle_edges)]
        ] == cycle_edges:
            expected = sum(
                float(value)
                for value in path_edge_distances[
                    start : start + len(cycle_edges)
                ]
            )
            if _strict_numeric_match(
                event.get("cycle_distance"), round(expected, 6)
            ):
                return True
    return False


def _replay_path_cycle_reward(
    label: Mapping[str, Any], envelope: Mapping[str, Any]
) -> tuple[bool, float]:
    events = label.get("cycle_distance_rewards")
    events = events if isinstance(events, list) else []
    cap = envelope.get("max_cycle_reward_per_path")
    threshold = envelope.get("cycle_reward_distance_threshold")
    if (
        not _finite(cap)
        or float(cap) < 0.0
        or not _finite(threshold)
        or float(threshold) < 0.0
    ):
        return False, 0.0
    total = 0.0
    valid = True
    path = [str(value) for value in label.get("path") or []]
    path_edge_ids = list(label.get("path_edge_ids") or [])
    path_edge_distances = list(label.get("path_edge_distances") or [])
    path_edge_strengths = list(label.get("path_edge_strengths") or [])
    expected_cycles: list[dict[str, Any]] = []
    if (
        len(path) != len(path_edge_ids) + 1
        or len(path_edge_ids) != len(path_edge_distances)
        or len(path_edge_ids) != len(path_edge_strengths)
        or any(
            not _finite(value) or float(value) < 0.0
            for value in path_edge_distances
        )
        or any(
            not _finite(value) or not 0.0 <= float(value) <= 1.0
            for value in path_edge_strengths
        )
    ):
        valid = False
    else:
        for edge_index, edge_id in enumerate(path_edge_ids):
            neighbor_id = path[edge_index + 1]
            previous_path = path[: edge_index + 1]
            if neighbor_id not in previous_path:
                continue
            previous_index = max(
                index
                for index, node_id in enumerate(previous_path)
                if node_id == neighbor_id
            )
            expected_cycles.append(
                {
                    "cycle_edges": [
                        str(value)
                        for value in path_edge_ids[
                            previous_index : edge_index + 1
                        ]
                    ],
                    "cycle_distance": round(
                        sum(
                            float(value)
                            for value in path_edge_distances[
                                previous_index : edge_index + 1
                            ]
                        ),
                        6,
                    ),
                    "current_edge_id": str(edge_id),
                    "current_edge_distance": float(
                        path_edge_distances[edge_index]
                    ),
                    "current_edge_strength": float(
                        path_edge_strengths[edge_index]
                    ),
                }
            )
    if len(events) != len(expected_cycles):
        valid = False
    for event, expected_cycle in zip(events, expected_cycles):
        if not isinstance(event, Mapping):
            valid = False
            continue
        edge_strength = event.get("edge_strength")
        cycle_distance = event.get("cycle_distance")
        before = event.get("reward_before_cap")
        after = event.get("reward_after_cap")
        if (
            event.get("protocol_version")
            != CYCLE_REWARD_REPLAY_PROTOCOL_VERSION
            or not _finite(edge_strength)
            or not 0.0 <= float(edge_strength) <= 1.0
            or not _finite(cycle_distance)
            or float(cycle_distance) < 0.0
            or not _finite(before)
            or float(before) < 0.0
            or not _finite(after)
            or float(after) < 0.0
        ):
            valid = False
            continue
        cap_remaining = max(0.0, float(cap) - total)
        if cap_remaining <= 0.0:
            expected_cycle_edges = [
                expected_cycle["current_edge_id"]
            ]
            expected_cycle_distance = expected_cycle[
                "current_edge_distance"
            ]
            expected_cap_reason = (
                "max_cycle_reward_per_path_exhausted"
            )
            expected_support_delta = 0
        else:
            expected_cycle_edges = expected_cycle["cycle_edges"]
            expected_cycle_distance = expected_cycle[
                "cycle_distance"
            ]
            if (
                float(threshold) <= 0.0
                or float(expected_cycle_distance)
                > float(threshold)
            ):
                expected_cap_reason = (
                    "cycle_distance_above_threshold"
                )
                expected_support_delta = 0
            else:
                expected_cap_reason = None
                expected_support_delta = 1
        expected_before = (
            0.0
            if cap_remaining <= 0.0
            or float(threshold) <= 0.0
            or float(expected_cycle_distance)
            > float(threshold)
            else 0.04
            * min(1.0, max(0.0, float(edge_strength)))
            * math.exp(
                -float(expected_cycle_distance)
                / max(float(threshold), 1e-6)
            )
        )
        expected_after = min(cap_remaining, expected_before)
        if expected_cap_reason is None:
            expected_cap_reason = (
                "within_cap"
                if math.isclose(
                    expected_after,
                    expected_before,
                    rel_tol=0.0,
                    abs_tol=DISTANCE_REPLAY_ABS_TOLERANCE,
                )
                else "max_cycle_reward_per_path"
            )
        expected_strength = float(
            expected_cycle["current_edge_strength"]
        )
        if not (
            list(event.get("cycle_edges") or [])
            == expected_cycle_edges
            and _strict_numeric_match(
                cycle_distance,
                expected_cycle_distance,
            )
            and math.isclose(
                float(edge_strength),
                expected_strength,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
            and
            _strict_numeric_match(before, round(expected_before, 6))
            and _strict_numeric_match(after, round(expected_after, 6))
            and event.get("cap_reason") == expected_cap_reason
            and int(event.get("support_delta") or 0)
            == expected_support_delta
        ):
            valid = False
        total = round(total + max(0.0, float(after)), 6)
    declared = label.get("reward_so_far")
    return (
        valid
        and _finite(declared)
        and math.isclose(
            float(declared),
            total,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        and total <= float(cap) + DISTANCE_REPLAY_ABS_TOLERANCE
    ), total


def _expected_walk_stop_reason(
    terminal_facts: Mapping[str, Any], *, layer: str
) -> str | None:
    skipped = terminal_facts.get("skipped_by_granularity")
    if skipped in {"mid", "coarse"}:
        return f"skipped_by_granularity={skipped}"
    entry_count = terminal_facts.get("entry_count")
    frontier_remaining = terminal_facts.get("frontier_remaining_count")
    expansion_count = terminal_facts.get("frontier_expansion_count")
    budget = terminal_facts.get("frontier_budget")
    if not all(
        type(value) is int and value >= 0
        for value in (
            entry_count,
            frontier_remaining,
            expansion_count,
            budget,
        )
    ):
        return None
    if entry_count == 0:
        return "frontier_empty" if layer == "chunk" else "no_entry_nodes"
    if budget > 0 and expansion_count >= budget:
        return "hard_budget_hit"
    if frontier_remaining == 0:
        return "frontier_empty"
    return None


def _expected_frontier_budget_from_envelope(
    layer: str,
    envelope: Mapping[str, Any],
    *,
    entry_count: int,
) -> int | None:
    key_by_layer = {
        "coarse": "agent_coarse_top_k",
        "mid": "agent_mid_top_k",
        "chunk": "agent_chunk_top_k",
    }
    budget_key = key_by_layer.get(layer)
    if (
        budget_key is None
        or type(envelope.get(budget_key)) is not int
        or int(envelope[budget_key]) < 0
    ):
        return None
    return max(1, entry_count or 1) * max(
        1, int(envelope[budget_key]) or 1
    )


def _audit_query_facet_posterior_calibration(
    audit: _Audit,
    *,
    diagnostics: Mapping[str, Any],
    convergence: Mapping[str, Any],
) -> None:
    raw_card = diagnostics.get("query_facet_posterior_calibration")
    if not isinstance(raw_card, Mapping):
        audit.check(
            "query_facet_posterior_calibration_present",
            False,
            code="query_facet_posterior_calibration_missing",
        )
        return
    card = dict(raw_card)
    supplied_hash = card.pop("calibration_hash", None)
    rounds = card.get("rounds")
    observations = card.get("observations")
    posterior = card.get("posterior")
    required_facets = card.get("required_facets")
    schema_valid = (
        card.get("protocol_version")
        == QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
        and card.get("protocol_hash")
        == query_facet_posterior_protocol_hash()
        and isinstance(rounds, list)
        and isinstance(observations, list)
        and isinstance(posterior, Mapping)
        and isinstance(required_facets, list)
        and card.get("rounds_used") == len(rounds)
        and card.get("observations_used") == len(observations)
        and type(card.get("round_budget")) is int
        and 1 <= int(card["round_budget"]) <= 2
        and len(rounds) <= int(card["round_budget"])
        and type(card.get("observation_budget")) is int
        and int(card["observation_budget"]) >= 1
        and len(observations) <= int(card["observation_budget"])
        and _finite(card.get("convergence_epsilon"))
        and 0.0 <= float(card["convergence_epsilon"]) <= 1.0
        and card.get("llm_sample_budget") == 0
        and card.get("model_call_count") == 0
        and card.get("is_evidence") is False
        and card.get("citation_authority") is False
        and card.get("graph_mutation_authority") is False
        and card.get("gray_zone_decision_authority") is False
        and card.get("posterior_used_as_numeric_query_relevance") is False
        and card.get("posterior_used_only_within_equal_uncovered_count")
        is True
        and _is_hash(card.get("query_facet_packet_hash"))
    )
    posterior_valid = (
        set(str(key) for key in posterior) == set(required_facets)
        and all(
            _finite(value) and 0.0 <= float(value) <= 1.0
            for value in posterior.values()
        )
        and (
            (not required_facets and not posterior)
            or math.isclose(
                sum(float(value) for value in posterior.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
    )
    observation_ids: list[str] = []
    observation_hashes_valid = True
    for raw_observation in observations if isinstance(observations, list) else []:
        if not isinstance(raw_observation, Mapping):
            observation_hashes_valid = False
            continue
        observation = dict(raw_observation)
        observed_hash = observation.pop("observation_hash", None)
        observation_id = observation.get("observation_id")
        if (
            observation.get("protocol_version")
            != QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
            or observation.get("checkpoint")
            not in QUERY_FACET_POSTERIOR_CHECKPOINTS
            or observation.get("model_call_count") != 0
            or not _is_hash(observation_id)
            or not _is_hash(observed_hash)
            or observed_hash != stable_hash(observation)
        ):
            observation_hashes_valid = False
        observation_ids.append(str(observation_id or ""))
    round_hashes_valid = True
    for index, raw_round in enumerate(rounds if isinstance(rounds, list) else []):
        if not isinstance(raw_round, Mapping):
            round_hashes_valid = False
            continue
        round_card = dict(raw_round)
        observed_hash = round_card.pop("round_hash", None)
        if (
            round_card.get("round_index") != index
            or round_card.get("checkpoint")
            != QUERY_FACET_POSTERIOR_CHECKPOINTS[index]
            or round_card.get("model_call_count") != 0
            or not _is_hash(observed_hash)
            or observed_hash != stable_hash(round_card)
        ):
            round_hashes_valid = False
    hash_valid = _is_hash(supplied_hash) and supplied_hash == stable_hash(card)
    convergence_valid = (
        convergence.get("query_facet_posterior_protocol_version")
        == QUERY_FACET_POSTERIOR_PROTOCOL_VERSION
        and convergence.get("query_facet_posterior_protocol_hash")
        == query_facet_posterior_protocol_hash()
        and convergence.get("query_facet_posterior_rounds_used")
        == len(rounds)
        and convergence.get("query_facet_posterior_observations_used")
        == len(observations)
        and convergence.get("query_facet_posterior_stop_reason")
        == card.get("stop_reason")
        and convergence.get("query_facet_posterior_model_call_count") == 0
    )
    audit.check(
        "query_facet_posterior_calibration_schema",
        schema_valid and posterior_valid,
        code="query_facet_posterior_calibration_schema_mismatch",
    )
    audit.check(
        "query_facet_posterior_calibration_hashes",
        hash_valid
        and observation_hashes_valid
        and round_hashes_valid
        and len(observation_ids) == len(set(observation_ids)),
        code="query_facet_posterior_calibration_hash_mismatch",
    )
    audit.check(
        "query_facet_posterior_convergence_binding",
        convergence_valid,
        code="query_facet_posterior_convergence_mismatch",
    )


def audit_retrieval_quality(
    snapshot: Mapping[str, Any],
    *,
    gray_zone_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = _Audit("retrieval")
    result_ids = _unique_strings(snapshot.get("result_chunk_ids"))
    stage_queues = snapshot.get("stage_queues") or {}
    candidate_pools = snapshot.get("candidate_pools") or {}
    topk = snapshot.get("topk_selection") or {}
    convergence = snapshot.get("convergence") or {}
    steps = list(snapshot.get("steps") or [])
    retrieval_granularity = str(
        snapshot.get("retrieval_granularity")
        or convergence.get("retrieval_granularity")
        or ""
    )
    audit.metrics.update(
        {
            "result_count": len(result_ids),
            "step_count": len(steps),
            "retrieval_granularity": retrieval_granularity,
            "dominance_pruned_count": sum(int(step.get("dominance_pruned_count") or 0) for step in steps),
            "cycle_distance_reward_max": max([_float(step.get("cycle_distance_reward")) for step in steps] + [0.0]),
        }
    )
    audit.check(
        "retrieval_protocol_hashes_present",
        _is_hash(snapshot.get("runtime_settings_hash"))
        and _is_hash(snapshot.get("agent_operating_envelope_hash"))
        and _is_hash(snapshot.get("traversal_protocol_hash"))
        and _is_hash(snapshot.get("edge_distance_protocol_hash"))
        and _is_hash(snapshot.get("edge_projection_protocol_hash")),
        code="retrieval_protocol_hash_missing",
    )
    frozen_envelope = (snapshot.get("trace_diagnostics") or {}).get(
        "agent_operating_envelope"
    )
    frozen_envelope = (
        frozen_envelope if isinstance(frozen_envelope, Mapping) else {}
    )
    _audit_query_facet_posterior_calibration(
        audit,
        diagnostics=(snapshot.get("trace_diagnostics") or {}),
        convergence=convergence,
    )
    expected_envelope_hash = stable_hash(dict(frozen_envelope))
    repair_directive = (snapshot.get("trace_diagnostics") or {}).get(
        "repair_directive"
    )
    repair_directive = (
        repair_directive
        if isinstance(repair_directive, Mapping)
        else None
    )
    audit.check(
        "retrieval_frozen_operating_envelope_replayed",
        bool(frozen_envelope)
        and snapshot.get("agent_operating_envelope_hash")
        == expected_envelope_hash
        and (snapshot.get("trace_diagnostics") or {}).get(
            "agent_operating_envelope_hash"
        )
        == expected_envelope_hash
        and convergence.get("agent_operating_envelope_hash")
        == expected_envelope_hash,
        code="retrieval_operating_envelope_hash_mismatch",
        expected=expected_envelope_hash,
        actual={
            "trace": snapshot.get("agent_operating_envelope_hash"),
            "diagnostics": (snapshot.get("trace_diagnostics") or {}).get(
                "agent_operating_envelope_hash"
            ),
            "convergence": convergence.get(
                "agent_operating_envelope_hash"
            ),
        },
    )
    walk_actions = {
        "staged_priority_queue_walk",
        "drill_down_each_coarse_or_direct_mid_entry",
        "walk_graph_frontier",
    }
    # Ordinary queue/dominance facts are bound to the frozen validated query
    # facet packet. Gray-zone replay remains independently bound to the raw
    # request query through _local_gray_zone_audit and must not reuse these
    # facets.
    required_facets = _active_traversal_required_facets(snapshot)
    configured_max_labels = frozen_envelope.get(
        "max_labels_per_node"
    )
    dominance_replay_by_step: dict[
        int, tuple[bool, int, int, dict[str, dict[str, Any]]]
    ] = {}
    frontier_path_label_closure_by_step: dict[int, bool] = {}
    raw_rank_by_layer: dict[
        str, dict[str, dict[str, Any]]
    ] = {}
    for step in steps:
        if str(step.get("action") or "") not in walk_actions:
            continue
        diagnostics = step.get("diagnostics")
        diagnostics = (
            diagnostics if isinstance(diagnostics, Mapping) else {}
        )
        observations = diagnostics.get("dominance_observations")
        observations = (
            observations if isinstance(observations, list) else []
        )
        frontier_enqueues = diagnostics.get(
            "frontier_enqueues"
        )
        frontier_enqueues = (
            frontier_enqueues
            if isinstance(frontier_enqueues, list)
            else []
        )
        frontier_json = diagnostics.get("frontier_json")
        frontier_json = (
            frontier_json if isinstance(frontier_json, list) else []
        )
        replay = _replay_dominance_observations(
            observations,
            frontier_json=frontier_json,
            required_facets=required_facets,
            max_labels=(
                int(configured_max_labels)
                if type(configured_max_labels) is int
                and configured_max_labels > 0
                else -1
            ),
        )
        path_labels = diagnostics.get("path_labels")
        path_labels = (
            path_labels if isinstance(path_labels, list) else []
        )
        terminal_facts = diagnostics.get("terminal_facts")
        terminal_facts = (
            terminal_facts
            if isinstance(terminal_facts, Mapping)
            else {}
        )
        step_expanded_edge_ids = step.get("expanded_edge_ids")
        step_expanded_edge_ids = (
            step_expanded_edge_ids
            if isinstance(step_expanded_edge_ids, list)
            else []
        )
        popped_frontier_state = step.get("popped_frontier_state")
        popped_frontier_state = (
            popped_frontier_state
            if isinstance(popped_frontier_state, Mapping)
            else {}
        )
        closure_valid = _replay_frontier_path_label_closure(
            observations,
            frontier_enqueues=frontier_enqueues,
            initial_entry_ids=list(
                (
                    stage_queues.get(
                        str(step.get("layer") or "")
                    )
                    or {}
                ).get("entry_ids")
                or []
            ),
            frontier_json=frontier_json,
            path_labels=path_labels,
            step_expanded_edge_ids=step_expanded_edge_ids,
            popped_frontier_state=popped_frontier_state,
            terminal_facts=terminal_facts,
            required_facets=required_facets,
            layer=str(step.get("layer") or ""),
            repair_directive=repair_directive,
        )
        step_index = step.get("step_index")
        if type(step_index) is int:
            dominance_replay_by_step[step_index] = replay
            frontier_path_label_closure_by_step[step_index] = (
                closure_valid
            )
        if replay[0] and closure_valid:
            raw_rank_by_layer[str(step.get("layer") or "")] = (
                replay[3]
            )
    audit.check(
        "retrieval_results_present_and_unique",
        bool(result_ids) and len(result_ids) == len(set(result_ids)),
        code="retrieval_results_missing_or_duplicate",
    )
    audit.check(
        "retrieval_granularity_valid",
        retrieval_granularity in {"mid", "coarse"},
        code="retrieval_granularity_invalid",
        actual=retrieval_granularity,
    )
    for scope, pool in _all_candidate_pools(candidate_pools):
        _audit_candidate_pool(audit, scope, pool)
    for layer in ("coarse", "mid", "rq_membership", "chunk"):
        queue = stage_queues.get(layer)
        audit.check(
            f"stage_queue_{layer}_present",
            isinstance(queue, Mapping),
            code="stage_queue_missing",
            scope=layer,
        )
        if not isinstance(queue, Mapping):
            continue
        entry_ids = _unique_strings(queue.get("entry_ids"))
        accepted_ids = _unique_strings(queue.get("accepted_ids"))
        selected_ids = _unique_strings(queue.get("selected_ids"))
        forced_ids = _unique_strings(queue.get("forced_entry_ids"))
        audit.check(
            f"stage_queue_{layer}_ids_unique",
            all(len(values) == len(set(values)) for values in (entry_ids, accepted_ids, selected_ids, forced_ids)),
            code="stage_queue_duplicate_id",
            scope=layer,
        )
        audit.check(
            f"stage_queue_{layer}_forced_admitted",
            set(forced_ids).issubset(entry_ids) and set(forced_ids).issubset(set(accepted_ids) | set(selected_ids)),
            code="forced_stage_entry_not_admitted",
            scope=layer,
        )
        if queue.get("top_k") is not None:
            audit.check(
                f"stage_queue_{layer}_top_k_budget",
                type(queue.get("top_k")) is int and len(selected_ids) <= int(queue["top_k"]),
                code="stage_queue_top_k_exceeded",
                scope=layer,
            )
    chunk_topk = topk.get("chunk") or {}
    selected_chunk_ids = _unique_strings(chunk_topk.get("selected_ids"))
    forced_chunk_ids = _unique_strings(chunk_topk.get("forced_selected_ids"))
    audit.check(
        "final_chunk_selection_matches_results",
        selected_chunk_ids == result_ids,
        code="final_chunk_selection_mismatch",
        expected=result_ids,
        actual=selected_chunk_ids,
    )
    audit.check(
        "final_chunk_top_k_budget",
        type(chunk_topk.get("top_k")) is int
        and int(chunk_topk["top_k"]) >= 0
        and len(result_ids) <= int(chunk_topk["top_k"])
        and int(chunk_topk.get("candidate_count") or 0) >= len(result_ids)
        and set(forced_chunk_ids).issubset(result_ids),
        code="final_chunk_top_k_or_forced_selection_invalid",
    )
    _audit_topk_ranking(
        audit,
        layer="chunk",
        selection=chunk_topk,
        raw_rank_by_id=raw_rank_by_layer.get("chunk"),
        repair_directive=repair_directive,
    )
    selected_mid_ids = _unique_strings((topk.get("mid") or {}).get("selected_ids"))
    _audit_topk_ranking(
        audit,
        layer="mid",
        selection=topk.get("mid") or {},
        raw_rank_by_id=raw_rank_by_layer.get("mid"),
        repair_directive=repair_directive,
    )
    mid_parent_ids = {
        str(pool.get("parent_node_id"))
        for pool in candidate_pools.get("chunk_by_mid") or []
        if isinstance(pool, Mapping) and pool.get("parent_node_id")
    }
    audit.check(
        "each_selected_mid_has_chunk_parent_pool",
        set(selected_mid_ids) == mid_parent_ids,
        code="mid_parent_drilldown_coverage_mismatch",
        expected=selected_mid_ids,
        actual=sorted(mid_parent_ids),
    )
    if retrieval_granularity == "coarse":
        selected_coarse_ids = _unique_strings((topk.get("coarse") or {}).get("selected_ids"))
        _audit_topk_ranking(
            audit,
            layer="coarse",
            selection=topk.get("coarse") or {},
            raw_rank_by_id=raw_rank_by_layer.get("coarse"),
            repair_directive=repair_directive,
        )
        coarse_parent_ids = {
            str(pool.get("parent_node_id"))
            for pool in candidate_pools.get("mid_by_coarse") or []
            if isinstance(pool, Mapping) and pool.get("parent_node_id")
        }
        audit.check(
            "each_selected_coarse_has_mid_parent_pool",
            set(selected_coarse_ids) == coarse_parent_ids,
            code="coarse_parent_drilldown_coverage_mismatch",
            expected=selected_coarse_ids,
            actual=sorted(coarse_parent_ids),
        )
    else:
        coarse_queue = stage_queues.get("coarse") or {}
        audit.check(
            "mid_mode_coarse_layer_skipped",
            coarse_queue.get("skipped_by_granularity") == "mid"
            and not _unique_strings(coarse_queue.get("entry_ids")),
            code="mid_mode_coarse_layer_not_skipped",
        )
    indexes = [step.get("step_index") for step in steps]
    audit.check(
        "retrieval_step_indexes_contiguous",
        indexes == list(range(len(steps))),
        code="retrieval_step_index_gap",
        expected=list(range(len(steps))),
        actual=indexes,
    )
    layer_convergence = convergence.get("layers") or {}
    all_decisions: list[Mapping[str, Any]] = []
    replayed_walk_reasons: dict[str, str] = {}
    for step in steps:
        step_scope = f"step[{step.get('step_index')}].{step.get('layer')}.{step.get('action')}"
        decisions = [item for item in step.get("gray_zone_path_decisions") or [] if isinstance(item, Mapping)]
        all_decisions.extend(decisions)
        dominance = step.get("dominance_pruned_count")
        cycle_reward = step.get("cycle_distance_reward")
        diagnostics = step.get("diagnostics") or {}
        path_labels = diagnostics.get("path_labels") or []
        layer = str(step.get("layer") or "")
        action = str(step.get("action") or "")
        if action in walk_actions:
            (
                dominance_replay_valid,
                expected_dominance,
                expected_label_budget,
                _raw_rank_facts,
            ) = dominance_replay_by_step.get(
                step.get("step_index"),
                (False, 0, 0, {}),
            )
            audit.check(
                f"{step_scope}_dominance_raw_events_replayed",
                dominance_replay_valid
                and type(dominance) is int
                and dominance == expected_dominance
                and (layer_convergence.get(layer) or {}).get(
                    "label_budget_pruned_count"
                )
                == expected_label_budget,
                code="dominance_raw_event_replay_mismatch",
                scope=step_scope,
                expected={
                    "dominance_pruned_count": expected_dominance,
                    "label_budget_pruned_count": expected_label_budget,
                },
                actual={
                    "dominance_pruned_count": dominance,
                    "label_budget_pruned_count": (
                        layer_convergence.get(layer) or {}
                    ).get("label_budget_pruned_count"),
                },
            )
            audit.check(
                (
                    f"{step_scope}_frontier_candidate_path_label_"
                    "edge_distance_closed"
                ),
                frontier_path_label_closure_by_step.get(
                    step.get("step_index"), False
                ),
                code=(
                    "frontier_candidate_path_label_"
                    "edge_distance_closure_mismatch"
                ),
                scope=step_scope,
                expected={
                    "protocol_version": (
                        FRONTIER_PATH_LABEL_CLOSURE_PROTOCOL_VERSION
                    ),
                    "closure": True,
                },
                actual={
                    "closure": frontier_path_label_closure_by_step.get(
                        step.get("step_index"), False
                    )
                },
            )
            envelope = (
                snapshot.get("trace_diagnostics") or {}
            ).get("agent_operating_envelope") or {}
            cycle_replays = [
                _replay_path_cycle_reward(label, envelope)
                for label in path_labels
                if isinstance(label, Mapping)
            ]
            expected_cycle = max(
                [value for valid, value in cycle_replays if valid]
                + [0.0]
            )
            audit.check(
                f"{step_scope}_cycle_reward_replayed",
                len(cycle_replays) == len(path_labels)
                and all(valid for valid, _value in cycle_replays)
                and _finite(cycle_reward)
                and _strict_numeric_match(cycle_reward, expected_cycle),
                code="cycle_distance_reward_mismatch",
                scope=step_scope,
                expected=expected_cycle,
                actual=cycle_reward,
            )
            terminal_facts = diagnostics.get("terminal_facts")
            terminal_facts = (
                terminal_facts
                if isinstance(terminal_facts, Mapping)
                else {}
            )
            expected_stop_reason = _expected_walk_stop_reason(
                terminal_facts, layer=layer
            )
            expected_entry_count = len(
                _unique_strings(
                    (stage_queues.get(layer) or {}).get(
                        "entry_ids"
                    )
                )
            )
            expected_expansion_count = sum(
                len(
                    _unique_strings(
                        label.get("expanded_edge_ids")
                    )
                )
                for label in path_labels
                if isinstance(label, Mapping)
            )
            expected_frontier_budget = (
                0
                if terminal_facts.get("skipped_by_granularity")
                in {"mid", "coarse"}
                else _expected_frontier_budget_from_envelope(
                    layer,
                    frozen_envelope,
                    entry_count=expected_entry_count,
                )
            )
            replayed_walk_reasons[layer] = str(
                expected_stop_reason or ""
            )
            audit.check(
                f"{step_scope}_terminal_reason_replayed",
                diagnostics.get("convergence_replay_protocol_version")
                == CONVERGENCE_REPLAY_PROTOCOL_VERSION
                and expected_stop_reason is not None
                and terminal_facts.get("entry_count")
                == expected_entry_count
                and terminal_facts.get(
                    "frontier_expansion_count"
                )
                == expected_expansion_count
                and terminal_facts.get("frontier_budget")
                == expected_frontier_budget
                and step.get("stop_reason") == expected_stop_reason
                and (layer_convergence.get(layer) or {}).get("reason")
                == expected_stop_reason,
                code="convergence_terminal_fact_mismatch",
                scope=step_scope,
                expected=expected_stop_reason,
                actual=step.get("stop_reason"),
            )
        else:
            audit.check(
                f"{step_scope}_dominance_nonnegative",
                type(dominance) is int and dominance >= 0,
                code="dominance_count_invalid",
                scope=step_scope,
            )
            audit.check(
                f"{step_scope}_cycle_reward_numeric",
                _finite(cycle_reward)
                and float(cycle_reward) >= 0.0,
                code="cycle_distance_reward_mismatch",
                scope=step_scope,
            )
        if layer in layer_convergence and action not in {"restore_context_package", "select_seeds_from_mid_rq_membership"}:
            audit.check(
                f"{step_scope}_dominance_matches_layer_convergence",
                int(dominance or 0) == int((layer_convergence.get(layer) or {}).get("dominance_pruned_count") or 0),
                code="dominance_convergence_mismatch",
                scope=step_scope,
            )
        allowed_non_walk_stop = {
            "select_entry_nodes": {
                "skipped_by_granularity=mid",
                "frontier_empty",
                "hard_budget_hit",
                "no_entry_nodes",
            },
            "select_seeds_from_mid_rq_membership": {
                "seeds_selected",
                "candidate_pool_exhausted",
                "candidate_pool_dedupe_budget_hit",
                "rq_membership_seed_selection",
            },
            "restore_context_package": {
                "structure_restored",
                "context_package_built",
            },
        }
        audit.check(
            f"{step_scope}_stop_reason_allowlisted",
            (
                action in walk_actions
                and bool(replayed_walk_reasons.get(layer))
            )
            or step.get("stop_reason")
            in allowed_non_walk_stop.get(action, set()),
            code="retrieval_step_stop_reason_invalid",
            scope=step_scope,
            actual=step.get("stop_reason"),
        )
    deterministic_decisions = [
        decision
        for decision in all_decisions
        if decision.get("decision_source") == "deterministic_local_rule"
    ]
    audit.check(
        "gray_decision_count_replayed",
        convergence.get("gray_zone_decision_count") == len(deterministic_decisions)
        and convergence.get("gray_zone_rule_evaluation_count") == len(deterministic_decisions),
        code="gray_decision_count_mismatch",
        expected=len(deterministic_decisions),
        actual={
            "decision_count": convergence.get("gray_zone_decision_count"),
            "evaluation_count": convergence.get("gray_zone_rule_evaluation_count"),
        },
    )
    thresholds = convergence.get("path_distance_thresholds") or {}
    audit.check(
        "path_distance_threshold_order",
        all(_finite(thresholds.get(key)) for key in ("green", "gray", "hard"))
        and float(thresholds["green"]) <= float(thresholds["gray"]) <= float(thresholds["hard"]),
        code="path_distance_threshold_order_invalid",
        actual=thresholds,
    )
    observation_audit = convergence.get("traversal_observation_budget_audit") or {}
    audit.check(
        "gray_and_observation_budget_replayed",
        convergence.get("gray_zone_model_call_count") == 0
        and observation_audit.get("local_rule_evaluation_count") == len(deterministic_decisions)
        and type(convergence.get("traversal_observation_budget")) is int
        and int(convergence["traversal_observation_budget"]) >= 0
        and int(observation_audit.get("expanded_observation_count") or 0)
        <= int(convergence["traversal_observation_budget"]),
        code="gray_observation_budget_mismatch",
    )
    hard_count = sum(int((item or {}).get("hard_stop_pruned_count") or 0) for item in layer_convergence.values())
    red_count = sum(int((item or {}).get("red_zone_pruned_count") or 0) for item in layer_convergence.values())
    audit.check(
        "distance_partition_counts_replayed",
        convergence.get("hard_stop_pruned_count") == hard_count
        and convergence.get("red_zone_pruned_count") == red_count
        and convergence.get("path_distance_partition_event_count") == hard_count + red_count,
        code="distance_partition_count_mismatch",
        expected={"hard": hard_count, "red": red_count, "events": hard_count + red_count},
        actual={
            "hard": convergence.get("hard_stop_pruned_count"),
            "red": convergence.get("red_zone_pruned_count"),
            "events": convergence.get("path_distance_partition_event_count"),
        },
    )
    audit.check(
        "retrieval_convergence_is_bounded",
        convergence.get("reason") == replayed_walk_reasons.get("chunk")
        and convergence.get("cycle_distance_reward_bounded") is True
        and convergence.get("accepted_chunk_count") == len(result_ids)
        and type(convergence.get("frontier_expansion_count")) is int
        and int(convergence["frontier_expansion_count"]) >= 0,
        code="retrieval_convergence_invalid",
    )
    try:
        recomputed_gray_zone_audit = _local_gray_zone_audit(snapshot)
        recompute_error = None
    except Exception as exc:
        recomputed_gray_zone_audit = {}
        recompute_error = f"{type(exc).__name__}: {exc}"
    audit.check(
        "gray_zone_raw_trace_full_local_replay",
        recompute_error is None
        and _complete_gray_zone_audit_valid(recomputed_gray_zone_audit),
        code="gray_zone_complete_local_audit_failed",
        actual={
            "error": recompute_error,
            "audit": recomputed_gray_zone_audit,
        },
    )
    if gray_zone_audit is not None:
        audit.check(
            "gray_zone_supplied_audit_matches_local_replay",
            isinstance(gray_zone_audit, Mapping)
            and _complete_gray_zone_audit_valid(gray_zone_audit)
            and stable_hash(dict(gray_zone_audit))
            == stable_hash(recomputed_gray_zone_audit),
            code="gray_zone_persisted_trace_audit_failed",
            expected=recomputed_gray_zone_audit,
            actual=gray_zone_audit,
        )
    return audit.finish()


def _normalize_chunk_text(value: str) -> str:
    normalized = (value or "").replace("\x00", "").replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def _chunk_text_hash(value: str) -> str:
    return hashlib.sha256(
        _normalize_chunk_text(value).encode("utf-8")
    ).hexdigest()


def _raw_span_text_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _source_fact_hash(fact: Mapping[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in fact.items() if key != "fact_hash"}
    )


def _source_span_valid(span: Mapping[str, Any], *, package_id: str, trace_id: str | None) -> bool:
    char_span = span.get("char_span")
    raw_span = span.get("raw_chunk_char_span")
    snapshot = span.get("source_snapshot_verification") or {}
    if not (
        span.get("contract_version") == "raw_chunk_source_span_v1"
        and span.get("document_version_id")
        and span.get("chunk_id")
        and span.get("source_path")
        and span.get("logical_source_path")
        and _is_hash(span.get("source_checksum"))
        and _is_hash(span.get("chunk_text_hash"))
        and _is_hash(span.get("raw_span_text_hash"))
        and span.get("chunk_text_hash_protocol_version")
        == CHUNK_TEXT_HASH_PROTOCOL_VERSION
        and span.get("raw_span_text_hash_protocol_version")
        == RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION
        and span.get("context_package_id") == package_id
        and span.get("retrieval_trace_id") == trace_id
        and snapshot.get("verified") is True
        and snapshot.get("protocol_version")
        == SOURCE_SNAPSHOT_PROTOCOL_VERSION
        and snapshot.get("storage_path")
        and snapshot.get("checksum") == span.get("source_checksum")
        and type(snapshot.get("size_bytes")) is int
        and snapshot.get("size_bytes") >= 0
    ):
        return False
    if not (
        isinstance(char_span, (list, tuple))
        and len(char_span) == 2
        and all(type(value) is int for value in char_span)
        and 0 <= char_span[0] <= char_span[1]
        and isinstance(raw_span, (list, tuple))
        and len(raw_span) == 2
        and all(type(value) is int for value in raw_span)
        and 0 <= raw_span[0] <= char_span[0] <= char_span[1] <= raw_span[1]
    ):
        return False
    page_range = span.get("page_range")
    return (
        isinstance(page_range, (list, tuple))
        and len(page_range) == 2
        and span.get("section_path") is not None
        and span.get("structure_path") is not None
        and isinstance(span.get("structure_node_ids"), list)
    )


def _source_span_matches_fact(
    span: Mapping[str, Any],
    package_chunk: Mapping[str, Any],
    fact: Mapping[str, Any],
) -> bool:
    stored_text = fact.get("stored_chunk_text")
    stored_span = fact.get("stored_char_span")
    char_span = span.get("char_span")
    snapshot = span.get("source_snapshot_verification") or {}
    if (
        not isinstance(stored_text, str)
        or not isinstance(stored_span, (list, tuple))
        or len(stored_span) != 2
        or not all(type(value) is int for value in stored_span)
        or not isinstance(char_span, (list, tuple))
        or len(char_span) != 2
        or not all(type(value) is int for value in char_span)
        or fact.get("fact_hash") != _source_fact_hash(fact)
        or fact.get("chunk_id") != span.get("chunk_id")
        or not fact.get("document_id")
        or fact.get("document_version_id")
        != span.get("document_version_id")
        or fact.get("snapshot_protocol_version")
        != SOURCE_SNAPSHOT_PROTOCOL_VERSION
        or fact.get("snapshot_verified") is not True
        or not _is_hash(fact.get("snapshot_observed_checksum"))
        or fact.get("snapshot_observed_checksum")
        != fact.get("document_version_checksum")
        or fact.get("stored_chunk_text_hash")
        != _chunk_text_hash(stored_text)
        or span.get("chunk_text_hash")
        != fact.get("stored_chunk_text_hash")
        or span.get("source_checksum")
        != fact.get("document_version_checksum")
        or span.get("source_path")
        != fact.get("document_version_storage_path")
        or span.get("logical_source_path")
        != fact.get("logical_source_path")
        or list(span.get("raw_chunk_char_span") or [])
        != list(stored_span)
        or snapshot.get("storage_path")
        != fact.get("document_version_storage_path")
        or snapshot.get("checksum")
        != fact.get("snapshot_observed_checksum")
        or snapshot.get("size_bytes") != fact.get("snapshot_size_bytes")
    ):
        return False
    stored_start, stored_end = int(stored_span[0]), int(stored_span[1])
    span_start, span_end = int(char_span[0]), int(char_span[1])
    if not (
        0 <= stored_start <= span_start <= span_end <= stored_end
        and stored_end - stored_start == len(stored_text)
    ):
        return False
    expected_content = stored_text[
        span_start - stored_start : span_end - stored_start
    ]
    return (
        package_chunk.get("content") == expected_content
        and span.get("raw_span_text_hash")
        == _raw_span_text_hash(expected_content)
        and package_chunk.get("chunk_text_hash")
        == fact.get("stored_chunk_text_hash")
        and package_chunk.get("raw_span_text_hash")
        == _raw_span_text_hash(expected_content)
        and package_chunk.get("document_version_id")
        == fact.get("document_version_id")
        and package_chunk.get("document_id")
        == fact.get("document_id")
        and list(package_chunk.get("char_span") or []) == list(char_span)
        and list(package_chunk.get("raw_chunk_char_span") or [])
        == list(stored_span)
    )


def audit_context_package_quality(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    audit = _Audit("context_package")
    package_id = str(snapshot.get("id") or "")
    trace_id = str(snapshot.get("retrieval_trace_id")) if snapshot.get("retrieval_trace_id") else None
    chunks = list(snapshot.get("chunks") or [])
    citation_spans = list(snapshot.get("citation_spans") or [])
    diagnostics = snapshot.get("diagnostics") or {}
    budget_audit = diagnostics.get("token_budget_audit") or {}
    token_budget = snapshot.get("token_budget")
    token_count = snapshot.get("token_count")
    source_facts = snapshot.get("source_facts")
    source_facts = (
        source_facts if isinstance(source_facts, list) else []
    )
    source_fact_by_chunk: dict[str, Mapping[str, Any]] = {}
    duplicate_source_fact_ids: list[str] = []
    for fact in source_facts:
        if not isinstance(fact, Mapping):
            continue
        chunk_id = str(fact.get("chunk_id") or "")
        if chunk_id in source_fact_by_chunk:
            duplicate_source_fact_ids.append(chunk_id)
        source_fact_by_chunk[chunk_id] = fact
    audit.metrics.update(
        {
            "chunk_count": len(chunks),
            "citation_span_count": len(citation_spans),
            "token_budget": token_budget,
            "token_count": token_count,
        }
    )
    audit.check(
        "context_package_identity_present",
        bool(package_id) and bool(trace_id) and _is_hash(snapshot.get("package_hash")),
        code="context_package_identity_missing",
    )
    audit.check("context_package_chunks_present", bool(chunks), code="context_package_chunks_missing")
    chunk_ids = [str(item.get("chunk_id") or "") for item in chunks]
    audit.check(
        "context_package_postgres_source_facts_complete",
        not duplicate_source_fact_ids
        and set(source_fact_by_chunk) == set(chunk_ids)
        and all(
            fact.get("fact_hash") == _source_fact_hash(fact)
            for fact in source_fact_by_chunk.values()
        ),
        code="context_package_source_fact_scope_invalid",
        expected=sorted(chunk_ids),
        actual={
            "source_fact_chunk_ids": sorted(source_fact_by_chunk),
            "duplicates": duplicate_source_fact_ids,
        },
    )
    dedupe_keys = [str(item.get("dedupe_key") or "") for item in chunks]
    declared_dedupe = _unique_strings(snapshot.get("dedupe_keys"))
    audit.check(
        "context_package_dedupe_replayed",
        all(chunk_ids)
        and len(chunk_ids) == len(set(chunk_ids))
        and all(dedupe_keys)
        and len(dedupe_keys) == len(set(dedupe_keys))
        and declared_dedupe == dedupe_keys,
        code="context_package_dedupe_mismatch",
        expected=dedupe_keys,
        actual=declared_dedupe,
    )
    total_chunk_tokens = 0
    source_by_chunk: dict[str, Mapping[str, Any]] = {}
    role_counts: Counter[str] = Counter()
    for item in chunks:
        chunk_id = str(item.get("chunk_id") or "missing")
        source_span = item.get("source_span") or {}
        content_tokens = item.get("content_token_count")
        original_tokens = item.get("original_token_count")
        audit.check(
            f"context_chunk_{chunk_id}_source_span_replayed",
            isinstance(source_span, Mapping)
            and source_span.get("chunk_id") == chunk_id
            and _source_span_valid(source_span, package_id=package_id, trace_id=trace_id)
            and chunk_id in source_fact_by_chunk
            and _source_span_matches_fact(
                source_span,
                item,
                source_fact_by_chunk[chunk_id],
            ),
            code="context_chunk_source_span_invalid",
            scope=chunk_id,
        )
        audit.check(
            f"context_chunk_{chunk_id}_closure_and_reason",
            item.get("context_package_id") == package_id
            and item.get("role") in {"hit", "bridge", "graph_path", "restored_context"}
            and bool(item.get("structure_closure"))
            and bool(item.get("why_selected"))
            and isinstance(item.get("structure_node_ids"), list)
            and type(content_tokens) is int
            and content_tokens >= 0
            and type(original_tokens) is int
            and original_tokens >= content_tokens,
            code="context_chunk_closure_or_token_invalid",
            scope=chunk_id,
        )
        if type(content_tokens) is int and content_tokens >= 0:
            total_chunk_tokens += content_tokens
        source_by_chunk[chunk_id] = source_span
        role_counts[str(item.get("role") or "")] += 1
    citation_by_chunk: dict[str, Mapping[str, Any]] = {}
    for span in citation_spans:
        chunk_id = str(span.get("chunk_id") or "missing")
        citation_by_chunk[chunk_id] = span
        audit.check(
            f"citation_span_{chunk_id}_raw_provenance",
            _source_span_valid(span, package_id=package_id, trace_id=trace_id)
            and chunk_id in source_by_chunk
            and chunk_id in source_fact_by_chunk
            and _source_span_matches_fact(
                span,
                next(
                    (
                        item
                        for item in chunks
                        if str(item.get("chunk_id") or "") == chunk_id
                    ),
                    {},
                ),
                source_fact_by_chunk[chunk_id],
            )
            and all(span.get(key) == source_by_chunk[chunk_id].get(key) for key in (
                "document_version_id",
                "source_checksum",
                "char_span",
                "raw_chunk_char_span",
                "chunk_text_hash",
                "raw_span_text_hash",
            )),
            code="citation_span_raw_provenance_mismatch",
            scope=chunk_id,
        )
    audit.check(
        "citation_span_chunk_coverage",
        set(citation_by_chunk) == set(chunk_ids),
        code="citation_span_chunk_coverage_mismatch",
        expected=sorted(chunk_ids),
        actual=sorted(citation_by_chunk),
    )
    hit_ids = set(_unique_strings(snapshot.get("hit_chunk_ids")))
    restored_ids = set(_unique_strings(snapshot.get("restored_chunk_ids")))
    bridge_ids = set(_unique_strings(snapshot.get("bridge_chunk_ids")))
    actual_by_role = {
        "hit": {str(item.get("chunk_id")) for item in chunks if item.get("role") == "hit"},
        "restored": {str(item.get("chunk_id")) for item in chunks if item.get("role") == "restored_context"},
        "bridge": {str(item.get("chunk_id")) for item in chunks if item.get("role") == "bridge"},
        "graph_path": {str(item.get("chunk_id")) for item in chunks if item.get("role") == "graph_path"},
    }
    expected_restored_ids = (
        actual_by_role["restored"]
        | actual_by_role["bridge"]
        | actual_by_role["graph_path"]
    )
    restore_counts = diagnostics.get("restore_counts") or {}
    audit.check(
        "context_package_roles_and_counts_replayed",
        hit_ids == actual_by_role["hit"]
        and restored_ids == expected_restored_ids
        and bridge_ids == actual_by_role["bridge"]
        and restore_counts.get("hit_chunks") == len(hit_ids)
        and restore_counts.get("restored_chunks") == len(restored_ids)
        and restore_counts.get("bridge_chunks") == len(bridge_ids)
        and restore_counts.get("graph_path_chunks") == len(actual_by_role["graph_path"])
        and restore_counts.get("parent_structure_nodes") == len(set(_unique_strings(snapshot.get("parent_structure_node_ids")))),
        code="context_package_role_count_mismatch",
        expected={
            **{
                key: sorted(value)
                for key, value in actual_by_role.items()
            },
            "restored_chunk_ids": sorted(expected_restored_ids),
        },
        actual={"hit": sorted(hit_ids), "restored": sorted(restored_ids), "bridge": sorted(bridge_ids), "counts": restore_counts},
    )
    trace_path_edge_ids = set(_unique_strings(snapshot.get("trace_path_edge_ids")))
    graph_path_ids = set(_unique_strings(snapshot.get("graph_path_ids")))
    audit.check(
        "context_package_graph_path_scope_replayed",
        graph_path_ids.issubset(trace_path_edge_ids),
        code="context_package_graph_path_scope_mismatch",
        expected=sorted(trace_path_edge_ids),
        actual=sorted(graph_path_ids),
    )
    audit.check(
        "context_package_token_hard_budget_replayed",
        type(token_budget) is int
        and token_budget > 0
        and type(token_count) is int
        and token_count >= 0
        and token_count <= token_budget
        and token_count == total_chunk_tokens
        and budget_audit.get("token_budget") == token_budget
        and budget_audit.get("token_count") == token_count
        and budget_audit.get("within_budget") is (token_count <= token_budget)
        and budget_audit.get("packing_protocol"),
        code="context_package_token_budget_mismatch",
        expected={"token_budget": token_budget, "token_count": total_chunk_tokens, "within_budget": True},
        actual=budget_audit,
    )
    snapshot_integrity = diagnostics.get("snapshot_integrity") or {}
    audit.check(
        "context_package_snapshot_integrity_gate",
        snapshot_integrity.get("fail_closed") is True
        and snapshot_integrity.get("protocol_version")
        and snapshot_integrity.get("verified_document_version_count")
        == len({str(item.get("document_version_id")) for item in chunks}),
        code="context_package_snapshot_integrity_invalid",
    )
    audit.check(
        "context_package_non_evidence_boundaries",
        diagnostics.get("conversation_state_is_evidence") is False
        and diagnostics.get("repair_gray_zone_decision_authority") is False
        and diagnostics.get("repair_gray_zone_model_call_count") == 0,
        code="context_package_evidence_boundary_invalid",
    )
    return audit.finish()


def _typed_action_record_valid(
    action: Mapping[str, Any],
    *,
    envelope: Mapping[str, Any],
    card_validation: Mapping[str, Any],
    retrieval_granularity: str,
) -> bool:
    if (
        set(action) != TYPED_ACTION_REQUIRED_FIELDS
        or action.get("action_type") not in ALLOWED_TYPED_ACTIONS
        or not isinstance(action.get("target_ids"), list)
        or len(action["target_ids"]) > 64
        or len(action["target_ids"])
        != len(set(action["target_ids"]))
        or any(
            not isinstance(value, str) or not value.strip()
            for value in action["target_ids"]
        )
        or not isinstance(action.get("reason"), str)
        or not action["reason"].strip()
        or len(action["reason"]) > 2000
        or not isinstance(action.get("budget_request"), Mapping)
        or not isinstance(action.get("expected_evidence"), Mapping)
        or not isinstance(action.get("stop_condition"), Mapping)
        or set(action["expected_evidence"]) - EXPECTED_EVIDENCE_FIELDS
        or set(action["stop_condition"]) - STOP_CONDITION_FIELDS
    ):
        return False
    semantic_payload = repr(
        {
            "reason": action.get("reason"),
            "expected_evidence": action.get("expected_evidence"),
            "stop_condition": action.get("stop_condition"),
        }
    ).casefold()
    if any(
        token in semantic_payload
        for token in (
            "evaluate_gray_zone_path",
            "continue_path",
            "stop_path_irrelevant",
            "follow_as_bridge",
            "request_structure_closure",
            "gray_zone",
            "gray-zone",
            "gray path",
        )
    ):
        return False
    if any(
        type(value) is not int or value < 0
        for value in action["budget_request"].values()
    ):
        return False
    action_type = str(action["action_type"])
    budget_keys = TYPED_ACTION_BUDGET_KEYS.get(action_type)
    if (
        budget_keys is None
        or set(action["budget_request"]) != budget_keys
        or any(
            type(envelope.get(key)) is not int
            or int(action["budget_request"][key])
            > int(envelope[key])
            for key in budget_keys
        )
    ):
        return False
    evidence = action["expected_evidence"]
    for key in ("requires_chunk_spans", "fallback_allowed"):
        if key in evidence and type(evidence[key]) is not bool:
            return False
    for key in (
        "allowed_relation_types",
        "relation_types",
        "required_restore_modes",
        "required_evidence_roles",
        "failure_types",
        "failure_card_hashes",
    ):
        value = evidence.get(key)
        if value is not None and not (
            isinstance(value, list)
            and len(value) <= 64
            and all(
                isinstance(item, str) and item.strip()
                for item in value
            )
        ):
            return False
    minimum_paths = evidence.get(
        "minimum_independent_support_paths"
    )
    if minimum_paths is not None and (
        type(minimum_paths) is not int or minimum_paths < 0
    ):
        return False
    for key in (
        "source",
        "start_layer",
        "target_layer",
        "required_verification_stage",
        "protocol_version",
        "executor_mechanism",
        "action_input_hash",
    ):
        value = evidence.get(key)
        if value is not None and not (
            isinstance(value, str)
            and value.strip()
            and len(value) <= 512
        ):
            return False
    if any(
        evidence.get(key) not in TYPED_ACTION_EVIDENCE_LAYERS
        for key in ("start_layer", "target_layer")
        if key in evidence
    ):
        return False
    if (
        evidence.get("start_layer") is not None
        and evidence.get("start_layer")
        != retrieval_granularity
    ):
        return False
    if "required_verification_stage" in evidence and (
        action_type != "verify_citations"
        or evidence["required_verification_stage"]
        != "structure_plus_llm_entailment"
    ):
        return False
    if (
        "action_input_hash" in evidence
        and not _is_hash(evidence["action_input_hash"])
    ) or any(
        not _is_hash(value)
        for value in evidence.get("failure_card_hashes") or []
    ):
        return False
    target_refs = evidence.get("canonical_target_refs")
    if target_refs is not None:
        allowed_target_ref_fields = {
            "claim_ids",
            "source_chunk_ids",
            "source_context_package_id",
            "source_retrieval_trace_id",
            "mid_concept_ids",
            "target_refs_hash",
        }
        if (
            not isinstance(target_refs, Mapping)
            or len(repr(target_refs)) > 10_000
            or set(target_refs) - allowed_target_ref_fields
        ):
            return False
        for key in (
            "claim_ids",
            "source_chunk_ids",
            "mid_concept_ids",
        ):
            values = target_refs.get(key) or []
            if not (
                isinstance(values, list)
                and len(values) <= 64
                and all(
                    isinstance(value, str) and value.strip()
                    for value in values
                )
            ):
                return False
        supplied_hash = target_refs.get("target_refs_hash")
        if (
            not _is_hash(supplied_hash)
            or supplied_hash
            != stable_hash(
                {
                    key: value
                    for key, value in target_refs.items()
                    if key != "target_refs_hash"
                }
            )
        ):
            return False
    allowed_relation_types = set(
        envelope.get("allowed_relation_types") or []
    )
    requested_relation_types = set(
        evidence.get("allowed_relation_types")
        or evidence.get("relation_types")
        or []
    )
    if (
        requested_relation_types - allowed_relation_types
        or evidence.get("fallback_allowed") is True
    ):
        return False
    required_restore_modes = set(
        envelope.get("required_restore_modes") or []
    )
    if action_type in {
        "restore_context_package",
        "build_context_package",
    } and not required_restore_modes.issubset(
        set(evidence.get("required_restore_modes") or [])
    ):
        return False
    if action_type == "verify_citations" and (
        evidence.get("required_verification_stage")
        != "structure_plus_llm_entailment"
    ):
        return False
    target_layers = card_validation.get("target_layers")
    if not isinstance(target_layers, Mapping) or set(
        target_layers
    ) != set(action["target_ids"]):
        return False
    allowed_target_layers = TYPED_ACTION_TARGET_LAYERS[
        action_type
    ]
    for target_id in action["target_ids"]:
        layers = target_layers.get(target_id)
        if not (
            isinstance(layers, list)
            and layers
            and all(
                layer in TYPED_ACTION_EVIDENCE_LAYERS
                for layer in layers
            )
            and allowed_target_layers.intersection(layers)
            and not (
                retrieval_granularity == "mid"
                and set(layers) == {"coarse"}
            )
        ):
            return False
    if (
        retrieval_granularity == "mid"
        and action_type == "activate_coarse_concepts"
    ):
        return False
    for key, value in action["stop_condition"].items():
        if key == "independent_support_paths_at_least":
            if type(value) is not int or value < 0:
                return False
        elif type(value) is not bool:
            return False
    return True


def _action_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_type": row.get("action_type"),
        "target_ids": list(row.get("target_ids") or []),
        "reason": row.get("reason"),
        "budget_request": dict(row.get("budget_request") or {}),
        "expected_evidence": dict(row.get("expected_evidence") or {}),
        "stop_condition": dict(row.get("stop_condition") or {}),
    }


def _plan_execution_control_card_valid(
    plan: Mapping[str, Any],
    *,
    actions: Sequence[Mapping[str, Any]],
    envelope: Mapping[str, Any],
) -> bool:
    controls = plan.get("execution_controls")
    binding = plan.get("retrieval_control_binding")
    if not isinstance(controls, Mapping) or not isinstance(binding, Mapping):
        return False
    control_card = dict(controls)
    supplied_hash = control_card.pop("control_hash", None)
    action_effects = controls.get("action_effects")
    expected_action_effects = [
        {
            "action_index": index,
            "action_type": action.get("action_type"),
            "target_ids": list(action.get("target_ids") or []),
            "budget_overrides": {
                key: int(value)
                for key, value in dict(
                    action.get("budget_request") or {}
                ).items()
                if type(value) is int
                and type(envelope.get(key)) is int
                and int(value) != int(envelope[key])
            },
            "expected_evidence": dict(
                action.get("expected_evidence") or {}
            ),
            "stop_condition": dict(
                action.get("stop_condition") or {}
            ),
        }
        for index, action in enumerate(actions)
    ]
    expected_binding_fields = {
        "retrieval_trace_id",
        "knowledge_base_id",
        "agent_plan_id",
        "agent_plan_index",
        "typed_action_control_hash",
        "typed_action_executor_protocol_version",
        "typed_action_controls",
    }
    requested_result_top_k = controls.get("requested_result_top_k")
    effective_result_top_k = controls.get("effective_result_top_k")
    return (
        controls.get("protocol_version")
        == TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION
        and _is_hash(supplied_hash)
        and supplied_hash == stable_hash(control_card)
        and action_effects == expected_action_effects
        and controls.get("gray_zone_semantics_changed") is False
        and controls.get("gray_zone_rule_inputs_modified") is False
        and controls.get("path_distance_thresholds_modified") is False
        and controls.get("gray_zone_model_call_count") == 0
        and not list(controls.get("unsupported_controls") or [])
        and isinstance(controls.get("allowed_relation_types"), list)
        and set(controls.get("allowed_relation_types") or []).issubset(
            set(envelope.get("allowed_relation_types") or [])
        )
        and type(requested_result_top_k) is int
        and requested_result_top_k > 0
        and type(effective_result_top_k) is int
        and 0 < effective_result_top_k <= requested_result_top_k
        and set(binding) == expected_binding_fields
        and binding.get("retrieval_trace_id")
        == plan.get("retrieval_trace_id")
        and binding.get("knowledge_base_id")
        == plan.get("knowledge_base_id")
        and binding.get("agent_plan_id") == plan.get("plan_id")
        and binding.get("agent_plan_index") == plan.get("plan_index")
        and binding.get("typed_action_control_hash") == supplied_hash
        and binding.get("typed_action_executor_protocol_version")
        == TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION
        and dict(binding.get("typed_action_controls") or {})
        == dict(controls)
    )


def _persisted_typed_action_facts_valid(
    response: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> bool:
    plans = facts.get("plans")
    if not isinstance(plans, list) or not plans:
        return False
    plan_by_id: dict[str, Mapping[str, Any]] = {}
    valid_plan_count = 0
    response_bound_valid_plan_count = 0
    facts_valid = facts.get("run_id") == response.get("run_id")
    executed_retrieval_trace_ids = {
        str((event.get("scores") or {}).get("retrieval_trace_id") or "")
        for event in response.get("trace") or []
        if isinstance(event, Mapping)
        and event.get("node") == "typed_action_executor"
        and (event.get("scores") or {}).get("retrieval_trace_id")
    }
    for plan in plans:
        if not isinstance(plan, Mapping):
            facts_valid = False
            continue
        plan_id = str(plan.get("plan_id") or "")
        if not plan_id or plan_id in plan_by_id:
            facts_valid = False
            continue
        plan_by_id[plan_id] = plan
        actions = plan.get("typed_actions")
        rows = plan.get("actions")
        validation = plan.get("validation")
        envelope = plan.get("envelope")
        envelope = (
            envelope if isinstance(envelope, Mapping) else {}
        )
        retrieval_granularity = str(
            validation.get("retrieval_granularity") or ""
        ) if isinstance(validation, Mapping) else ""
        if (
            not isinstance(actions, list)
            or not isinstance(rows, list)
            or not isinstance(validation, Mapping)
            or not envelope
            or retrieval_granularity not in {"mid", "coarse"}
            or not typed_action_schema_identity_supported(validation)
            or len(rows) != len(actions)
        ):
            facts_valid = False
            continue
        facts_valid = facts_valid and _plan_execution_control_card_valid(
            plan,
            actions=[
                action
                for action in actions
                if isinstance(action, Mapping)
            ],
            envelope=envelope,
        )
        accepted_cards = validation.get("accepted")
        accepted_cards = (
            accepted_cards if isinstance(accepted_cards, list) else []
        )
        max_actions = envelope.get("max_typed_actions_per_round")
        required_actions = typed_action_required_actions_for_identity(
            validation
        )
        if not (
            type(max_actions) is int
            and max_actions > 0
            and len(actions) <= max_actions
            and len(accepted_cards) == len(actions)
            and len(
                {
                    str(action.get("action_type") or "")
                    for action in actions
                    if isinstance(action, Mapping)
                }
            )
            == len(actions)
        ):
            facts_valid = False
        for index, action in enumerate(actions):
            row = rows[index] if index < len(rows) else {}
            card = next(
                (
                    item
                    for item in accepted_cards
                    if isinstance(item, Mapping)
                    and item.get("accepted_index") == index
                ),
                None,
            )
            row_validation = (
                row.get("validation")
                if isinstance(row, Mapping)
                and isinstance(row.get("validation"), Mapping)
                else {}
            )
            card_validation = (
                card.get("validation")
                if isinstance(card, Mapping)
                and isinstance(card.get("validation"), Mapping)
                else {}
            )
            facts_valid = facts_valid and (
                isinstance(action, Mapping)
                and _typed_action_record_valid(
                    action,
                    envelope=envelope,
                    card_validation=card_validation,
                    retrieval_granularity=retrieval_granularity,
                )
                and isinstance(row, Mapping)
                and _action_row_payload(row) == dict(action)
                and row.get("action_index") == index
                and row.get("status")
                in {
                    "accepted",
                    "completed",
                    "deferred",
                    "no_progress",
                    "rejected",
                }
                and card is not None
                and card.get("accepted_index") == index
                and card.get("action_type") == action.get("action_type")
                and card_validation.get("valid") is True
                and card_validation.get("schema_checked") is True
                and card_validation.get("budget_checked") is True
                and card_validation.get("target_ids_checked") is True
                and card_validation.get("target_scope_checked") is True
                and card_validation.get(
                    "fallback_disabled_checked"
                )
                is True
                and card_validation.get(
                    "bridge_protection_checked"
                )
                is True
                and all(
                    row_validation.get(key) == value
                    for key, value in card_validation.items()
                )
                and row_validation.get(
                    "typed_action_schema_protocol_version"
                )
                == validation.get("typed_action_schema_protocol_version")
                and row_validation.get("typed_action_schema_protocol_hash")
                == validation.get("typed_action_schema_protocol_hash")
            )
        if validation.get("valid") is True:
            valid_plan_count += 1
            if (
                plan.get("retrieval_trace_id")
                in executed_retrieval_trace_ids
            ):
                response_bound_valid_plan_count += 1
            facts_valid = facts_valid and (
                not (validation.get("rejected") or [])
                and not (validation.get("inserted_required_actions") or [])
                and all(
                    row.get("status") != "rejected"
                    for row in rows
                    if isinstance(row, Mapping)
                )
                and set(required_actions).issubset(
                    {
                        str(action.get("action_type") or "")
                        for action in actions
                        if isinstance(action, Mapping)
                    }
                )
            )
    validation_events = [
        event
        for event in response.get("trace") or []
        if isinstance(event, Mapping)
        and event.get("node") == "typed_action_validation"
    ]
    if not validation_events:
        return False
    for event in validation_events:
        scores = event.get("scores")
        scores = scores if isinstance(scores, Mapping) else {}
        plan = plan_by_id.get(str(scores.get("plan_id") or ""))
        if (
            plan is None
            or not isinstance(scores.get("validation"), Mapping)
            or stable_hash(dict(scores["validation"]))
            != stable_hash(dict(plan.get("validation") or {}))
        ):
            facts_valid = False
    return (
        facts_valid
        and valid_plan_count >= 1
        and response_bound_valid_plan_count >= 1
    )


def _executor_plan_control_bindings_valid(
    response: Mapping[str, Any],
    *,
    typed_action_facts: Mapping[str, Any] | None,
    executor_events: Sequence[Mapping[str, Any]],
    final_control_hash: Any,
) -> bool:
    if not isinstance(typed_action_facts, Mapping):
        return False
    valid_plans = [
        plan
        for plan in typed_action_facts.get("plans") or []
        if isinstance(plan, Mapping)
        and isinstance(plan.get("validation"), Mapping)
        and (plan.get("validation") or {}).get("valid") is True
        and plan.get("retrieval_trace_id")
    ]
    plan_by_id = {
        str(plan.get("plan_id") or ""): plan
        for plan in valid_plans
        if str(plan.get("plan_id") or "")
    }
    if (
        not valid_plans
        or len(plan_by_id) != len(valid_plans)
        or len(executor_events) != len(valid_plans)
        or not _is_hash(final_control_hash)
    ):
        return False
    executed_plan_ids: list[str] = []
    for event in executor_events:
        scores = event.get("scores")
        if not isinstance(scores, Mapping):
            return False
        plan_id = str(scores.get("plan_id") or "")
        plan = plan_by_id.get(plan_id)
        if plan is None:
            return False
        control_hash = (
            plan.get("execution_controls") or {}
        ).get("control_hash")
        if not (
            scores.get("plan_index") == plan.get("plan_index")
            and scores.get("retrieval_trace_id")
            == plan.get("retrieval_trace_id")
            and scores.get("typed_action_control_hash") == control_hash
        ):
            return False
        executed_plan_ids.append(plan_id)
    if len(executed_plan_ids) != len(set(executed_plan_ids)):
        return False
    final_plan = max(
        valid_plans,
        key=lambda plan: int(plan.get("plan_index") or 0),
    )
    final_plan_hash = (
        final_plan.get("execution_controls") or {}
    ).get("control_hash")
    model_audit = (
        response.get("model_audit")
        or response.get("answer_model_audit")
        or {}
    )
    repair_actions = list(model_audit.get("repair_actions") or [])
    return (
        set(executed_plan_ids) == set(plan_by_id)
        and final_plan_hash == final_control_hash
        and (
            bool(repair_actions)
            or final_plan.get("retrieval_trace_id")
            == response.get("retrieval_trace_id")
        )
    )


def _validation_entries(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for event in response.get("trace") or []:
        if not isinstance(event, Mapping) or event.get("node") != "typed_action_validation":
            continue
        validation = (event.get("scores") or {}).get("validation") or {}
        entries.extend(item for item in validation.get("accepted") or [] if isinstance(item, Mapping))
    return entries


def _raw_agent_trace_scores(event: Mapping[str, Any]) -> dict[str, Any]:
    scores = event.get("scores")
    scores = dict(scores) if isinstance(scores, Mapping) else {}
    node = str(event.get("node") or "")
    audit_kind = scores.get("audit_kind")
    if (
        node in {"evidence_gate", "grounded_answer"}
        and audit_kind in {"evidence_gate", "grounded_answer"}
        and isinstance(scores.get("answer_model_audit"), Mapping)
    ):
        return dict(scores["answer_model_audit"])
    if (
        node == "layered_retrieval"
        and audit_kind == "layered_retrieval"
        and isinstance(scores.get("retrieval_audit"), Mapping)
    ):
        return dict(scores["retrieval_audit"])
    if (
        node == "repair_executed"
        and audit_kind == "repair"
        and isinstance(scores.get("repair_action"), Mapping)
    ):
        return dict(scores["repair_action"])
    scores.pop("audit_kind", None)
    scores.pop("contract_version", None)
    return scores


def agent_trace_event_replay_payload(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the durable fields whose values must equal one PostgreSQL row."""

    return {
        "id": event.get("id"),
        "run_id": event.get("run_id"),
        "sequence_index": event.get("sequence_index"),
        "node": event.get("node"),
        "status": event.get("status"),
        "input_summary": event.get("input_summary") or "",
        "output_summary": event.get("output_summary") or "",
        "document_ids": list(event.get("document_ids") or []),
        "scores": _raw_agent_trace_scores(event),
        "duration_ms": event.get("duration_ms"),
        "error": event.get("error")
        if "error" in event
        else event.get("error_message"),
    }


def persisted_agent_snapshot_hash(
    snapshot: Mapping[str, Any],
) -> str:
    return stable_hash(
        {
            "protocol_version": snapshot.get("protocol_version"),
            "run": snapshot.get("run"),
            "trace_events": snapshot.get("trace_events"),
            "bindings": snapshot.get("bindings"),
        }
    )


def _expected_completed_agent_trace_nodes(
    planning_rounds_used: Any,
    repair_action_count: int,
) -> list[str] | None:
    if (
        type(planning_rounds_used) is not int
        or planning_rounds_used < 1
        or repair_action_count < 0
    ):
        return None
    return [
        *AGENT_TRACE_FIXED_PREFIX,
        *(
            AGENT_TRACE_PLANNING_STAGE_GROUP
            * planning_rounds_used
        ),
        *AGENT_TRACE_FIXED_SUFFIX_BEFORE_REPAIR,
        *(("repair_executed",) * repair_action_count),
        *AGENT_TRACE_FIXED_SUFFIX_AFTER_REPAIR,
    ]


def _persisted_agent_replay_valid(
    response: Mapping[str, Any],
    persisted_agent_facts: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(persisted_agent_facts, Mapping):
        return False
    run = persisted_agent_facts.get("run")
    trace_events = persisted_agent_facts.get("trace_events")
    if (
        persisted_agent_facts.get("protocol_version")
        != PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION
        or not isinstance(run, Mapping)
        or not isinstance(trace_events, list)
        or not trace_events
        or persisted_agent_facts.get("snapshot_hash")
        != persisted_agent_snapshot_hash(persisted_agent_facts)
    ):
        return False
    persisted_rows = [
        agent_trace_event_replay_payload(event)
        for event in trace_events
        if isinstance(event, Mapping)
    ]
    response_rows = [
        agent_trace_event_replay_payload(event)
        for event in response.get("trace") or []
        if isinstance(event, Mapping)
    ]
    run_id = str(run.get("id") or "")
    event_ids = [str(event.get("id") or "") for event in persisted_rows]
    return (
        bool(run_id)
        and run_id == response.get("run_id")
        and run.get("status") == "completed"
        and run.get("route") == "layered_context_graph"
        and run.get("final_answer") == response.get("answer")
        and len(persisted_rows) == len(trace_events)
        and persisted_rows == response_rows
        and event_ids
        and all(event_ids)
        and len(event_ids) == len(set(event_ids))
        and [event.get("sequence_index") for event in persisted_rows]
        == list(range(len(persisted_rows)))
        and all(event.get("run_id") == run_id for event in persisted_rows)
    )


def _agent_reciprocal_bindings_valid(
    response: Mapping[str, Any],
    *,
    persisted_agent_facts: Mapping[str, Any] | None,
    typed_action_facts: Mapping[str, Any] | None,
    retrieval_snapshot: Mapping[str, Any] | None,
) -> bool:
    if (
        not isinstance(persisted_agent_facts, Mapping)
        or not isinstance(typed_action_facts, Mapping)
        or not isinstance(retrieval_snapshot, Mapping)
    ):
        return False
    run = persisted_agent_facts.get("run")
    bindings = persisted_agent_facts.get("bindings")
    if not isinstance(run, Mapping) or not isinstance(bindings, Mapping):
        return False
    initial_retrieval = bindings.get("initial_retrieval_trace")
    initial_package = bindings.get("initial_context_package")
    retrieval = bindings.get("retrieval_trace")
    package = bindings.get("context_package")
    answer_session = bindings.get("answer_session")
    reward = bindings.get("reward_event")
    if not all(
        isinstance(item, Mapping)
        for item in (
            initial_retrieval,
            initial_package,
            retrieval,
            package,
            answer_session,
            reward,
        )
    ):
        return False
    run_id = str(run.get("id") or "")
    knowledge_base_id = str(run.get("knowledge_base_id") or "")
    initial_retrieval_trace_id = str(
        initial_retrieval.get("id") or ""
    )
    initial_context_package_id = str(
        initial_package.get("id") or ""
    )
    retrieval_trace_id = str(retrieval.get("id") or "")
    context_package_id = str(package.get("id") or "")
    answer_session_id = str(answer_session.get("id") or "")
    plans = typed_action_facts.get("plans")
    plans = plans if isinstance(plans, list) else []
    actions = [
        action
        for plan in plans
        if isinstance(plan, Mapping)
        for action in plan.get("actions") or []
        if isinstance(action, Mapping)
    ]
    ordered_plans = sorted(
        (
            plan
            for plan in plans
            if isinstance(plan, Mapping)
            and type(plan.get("plan_index")) is int
        ),
        key=lambda plan: int(plan["plan_index"]),
    )
    final_plan = ordered_plans[-1] if ordered_plans else None
    model_audit = (
        response.get("model_audit")
        or response.get("answer_model_audit")
        or {}
    )
    events = [
        agent_trace_event_replay_payload(event)
        for event in response.get("trace") or []
        if isinstance(event, Mapping)
    ]
    events_by_node: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_node[str(event.get("node") or "")].append(event)
    package_events = events_by_node.get("context_package") or []
    restoration_events = (
        events_by_node.get("structure_context_restoration") or []
    )
    evidence_gate_events = events_by_node.get("evidence_gate") or []
    executor_events = events_by_node.get("typed_action_executor") or []
    planner_events = events_by_node.get("agent_planner") or []
    validation_events = (
        events_by_node.get("typed_action_validation") or []
    )
    evaluator_events = events_by_node.get("evidence_evaluator") or []
    grounded_answer_events = (
        events_by_node.get("grounded_answer") or []
    )
    planning_stage_rows_bound = (
        len(ordered_plans) == len(plans)
        and [plan.get("plan_index") for plan in ordered_plans]
        == list(range(len(ordered_plans)))
        and all(
            len(stage_events) == len(ordered_plans)
            for stage_events in (
                planner_events,
                validation_events,
                executor_events,
                evaluator_events,
            )
        )
        and all(
            all(
                (stage_events[index].get("scores") or {}).get(
                    "plan_id"
                )
                == plan.get("plan_id")
                and (stage_events[index].get("scores") or {}).get(
                    "plan_index"
                )
                == plan.get("plan_index")
                for stage_events in (
                    planner_events,
                    validation_events,
                    executor_events,
                    evaluator_events,
                )
            )
            and (executor_events[index].get("scores") or {}).get(
                "retrieval_trace_id"
            )
            == plan.get("retrieval_trace_id")
            for index, plan in enumerate(ordered_plans)
        )
    )
    citations = [
        citation
        for citation in response.get("citations") or []
        if isinstance(citation, Mapping)
    ]
    raw_repair_actions = model_audit.get("repair_actions")
    raw_repair_actions = (
        raw_repair_actions
        if isinstance(raw_repair_actions, list)
        else []
    )
    repair_rounds = [
        item
        for item in raw_repair_actions
        if isinstance(item, Mapping)
        and item.get("action_type")
        != "claim_level_final_grounded_gate"
    ]
    repair_chain_valid = (
        len(
            [
                item
                for item in raw_repair_actions
                if isinstance(item, Mapping)
            ]
        )
        == len(raw_repair_actions)
        and [
            item.get("repair_round_index")
            for item in repair_rounds
        ]
        == list(range(len(repair_rounds)))
    )
    chained_package_id = initial_context_package_id
    chained_trace_id = initial_retrieval_trace_id
    for repair in repair_rounds:
        repair_chain_valid = repair_chain_valid and (
            repair.get("before_context_package_id")
            == chained_package_id
            and repair.get("before_retrieval_trace_id")
            == chained_trace_id
            and bool(repair.get("repaired_context_package_id"))
            and bool(repair.get("repaired_retrieval_trace_id"))
        )
        chained_package_id = str(
            repair.get("repaired_context_package_id") or ""
        )
        chained_trace_id = str(
            repair.get("repaired_retrieval_trace_id") or ""
        )
    repair_chain_valid = repair_chain_valid and (
        chained_package_id == context_package_id
        and chained_trace_id == retrieval_trace_id
    )
    return (
        bool(run_id)
        and bool(knowledge_base_id)
        and bool(initial_retrieval_trace_id)
        and bool(initial_context_package_id)
        and bool(retrieval_trace_id)
        and bool(context_package_id)
        and bool(answer_session_id)
        and response.get("run_id") == run_id
        and response.get("retrieval_trace_id") == retrieval_trace_id
        and response.get("context_package_id") == context_package_id
        and retrieval_snapshot.get("trace_id") == retrieval_trace_id
        and typed_action_facts.get("run_id") == run_id
        and plans
        and all(plan.get("run_id") == run_id for plan in plans)
        and all(action.get("run_id") == run_id for action in actions)
        and final_plan is not None
        and final_plan.get("retrieval_trace_id")
        == initial_retrieval_trace_id
        and planning_stage_rows_bound
        and repair_chain_valid
        and initial_retrieval.get("knowledge_base_id")
        == knowledge_base_id
        and initial_package.get("knowledge_base_id")
        == knowledge_base_id
        and initial_package.get("retrieval_trace_id")
        == initial_retrieval_trace_id
        and retrieval.get("knowledge_base_id") == knowledge_base_id
        and package.get("knowledge_base_id") == knowledge_base_id
        and package.get("retrieval_trace_id") == retrieval_trace_id
        and answer_session.get("knowledge_base_id")
        == knowledge_base_id
        and answer_session.get("retrieval_trace_id")
        == retrieval_trace_id
        and answer_session.get("context_package_id")
        == context_package_id
        and answer_session.get("qa_session_id")
        == run.get("session_id")
        and answer_session.get("question") == run.get("question")
        and answer_session.get("answer") == run.get("final_answer")
        and reward.get("knowledge_base_id") == knowledge_base_id
        and reward.get("agent_run_id") == run_id
        and reward.get("retrieval_trace_id") == retrieval_trace_id
        and reward.get("context_package_id") == context_package_id
        and reward.get("answer_session_id") == answer_session_id
        and model_audit.get("answer_session_id") == answer_session_id
        and model_audit.get("retrieval_trace_id") == retrieval_trace_id
        and model_audit.get("context_package_id") == context_package_id
        and len(package_events) == 1
        and (package_events[0].get("scores") or {}).get(
            "context_package_id"
        )
        == initial_context_package_id
        and len(restoration_events) == 1
        and (restoration_events[0].get("scores") or {}).get(
            "context_package_id"
        )
        == initial_context_package_id
        and len(grounded_answer_events) == 1
        and (grounded_answer_events[0].get("scores") or {}).get(
            "context_package_id"
        )
        == initial_context_package_id
        and len(evidence_gate_events) == 1
        and (evidence_gate_events[0].get("scores") or {}).get(
            "retrieval_trace_id"
        )
        == initial_retrieval_trace_id
        and executor_events
        and citations
        and all(
            citation.get("context_package_id") == context_package_id
            and citation.get("retrieval_trace_id")
            == retrieval_trace_id
            for citation in citations
        )
    )


def audit_agent_quality(
    response: Mapping[str, Any],
    *,
    retrieval_snapshot: Mapping[str, Any] | None = None,
    gray_zone_audit: Mapping[str, Any] | None = None,
    typed_action_facts: Mapping[str, Any] | None = None,
    retrieval_quality: Mapping[str, Any] | None = None,
    persisted_agent_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = _Audit("agent")
    events = [
        agent_trace_event_replay_payload(event)
        for event in response.get("trace") or []
        if isinstance(event, Mapping)
    ]
    response = {**dict(response), "trace": events}
    nodes = [str(event.get("node") or "") for event in events]
    events_by_node: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_node[str(event.get("node") or "")].append(event)
    required_nodes = {
        "agent_planner",
        "typed_action_validation",
        "typed_action_executor",
        "evidence_evaluator",
        "evidence_gate",
        "entry_selection",
        "layer_drilldown",
        "frontier_traversal",
        "chunk_recall",
        "layered_retrieval",
        "structure_context_restoration",
        "context_package",
        "grounded_answer",
        "citation_verification",
        "reward_event",
    }
    model_audit = response.get("model_audit") or response.get("answer_model_audit") or {}
    raw_repair_action_list = model_audit.get("repair_actions")
    raw_repair_action_list = (
        raw_repair_action_list
        if isinstance(raw_repair_action_list, list)
        else []
    )
    expected_nodes = _expected_completed_agent_trace_nodes(
        model_audit.get("planning_rounds_used"),
        len(raw_repair_action_list),
    )
    package_id = response.get("context_package_id")
    trace_id = response.get("retrieval_trace_id")
    citations = [citation for citation in response.get("citations") or [] if isinstance(citation, Mapping)]
    audit.metrics.update(
        {
            "event_count": len(events),
            "citation_count": len(citations),
            "planning_rounds_used": model_audit.get("planning_rounds_used"),
            "repair_rounds_used": model_audit.get("repair_rounds_used"),
        }
    )
    sequence_indexes = [
        event.get("sequence_index") for event in events
    ]
    audit.check(
        "agent_trace_sequence_contiguous_and_run_bound",
        bool(events)
        and sequence_indexes == list(range(len(events)))
        and all(
            event.get("run_id") == response.get("run_id")
            for event in events
        ),
        code="agent_trace_sequence_invalid",
        expected={
            "run_id": response.get("run_id"),
            "sequence_indexes": list(range(len(events))),
        },
        actual={
            "run_ids": [
                event.get("run_id") for event in events
            ],
            "sequence_indexes": sequence_indexes,
        },
    )
    audit.check(
        "persisted_agent_run_and_trace_rows_replayed",
        _persisted_agent_replay_valid(
            response,
            persisted_agent_facts,
        ),
        code="persisted_agent_run_trace_replay_mismatch",
        expected={
            "protocol_version": PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION,
            "run_id": response.get("run_id"),
            "event_count": len(events),
        },
        actual=persisted_agent_facts,
    )
    audit.check(
        "completed_agent_trace_exact_stage_grammar",
        expected_nodes is not None
        and nodes == expected_nodes
        and all(node in AGENT_TRACE_ALLOWED_COMPLETED_NODES for node in nodes)
        and all(event.get("status") == "completed" for event in events)
        and all(event.get("error") in {None, ""} for event in events),
        code="agent_trace_stage_grammar_invalid",
        expected={
            "protocol_version": AGENT_TRACE_STAGE_PROTOCOL_VERSION,
            "nodes": expected_nodes,
            "allowed_nodes": sorted(
                AGENT_TRACE_ALLOWED_COMPLETED_NODES
            ),
        },
        actual={
            "nodes": nodes,
            "statuses": [event.get("status") for event in events],
            "errors": [event.get("error") for event in events],
        },
    )
    audit.check(
        "agent_required_trace_nodes_present",
        required_nodes.issubset(nodes),
        code="agent_required_trace_node_missing",
        expected=sorted(required_nodes),
        actual=sorted(set(nodes)),
    )
    audit.check(
        "agent_trace_order_valid",
        all(
            nodes.index(first) < nodes.index(second)
            for first, second in (
                ("agent_planner", "typed_action_validation"),
                ("typed_action_validation", "typed_action_executor"),
                ("typed_action_executor", "evidence_evaluator"),
                ("context_package", "grounded_answer"),
                ("grounded_answer", "citation_verification"),
            )
            if first in nodes and second in nodes
        ),
        code="agent_trace_order_invalid",
    )
    validation_entries = _validation_entries(response)
    audit.check(
        "typed_actions_validated",
        bool(validation_entries)
        and all(
            entry.get("action_type") in ALLOWED_TYPED_ACTIONS
            and
            (entry.get("validation") or {}).get("valid") is True
            and (entry.get("validation") or {}).get("schema_checked") is True
            and (entry.get("validation") or {}).get("budget_checked") is True
            and (entry.get("validation") or {}).get("target_ids_checked") is True
            and (entry.get("validation") or {}).get("target_scope_checked") is True
            for entry in validation_entries
        ),
        code="typed_action_validation_incomplete",
        actual=validation_entries,
    )
    audit.check(
        "persisted_typed_action_schema_and_validator_replayed",
        isinstance(typed_action_facts, Mapping)
        and _persisted_typed_action_facts_valid(
            response, typed_action_facts
        ),
        code="typed_action_persisted_replay_mismatch",
        actual=typed_action_facts,
    )
    audit.check(
        "run_trace_context_package_answer_reward_reciprocal_binding",
        _agent_reciprocal_bindings_valid(
            response,
            persisted_agent_facts=persisted_agent_facts,
            typed_action_facts=typed_action_facts,
            retrieval_snapshot=retrieval_snapshot,
        ),
        code="agent_reciprocal_binding_mismatch",
        actual=(
            persisted_agent_facts.get("bindings")
            if isinstance(persisted_agent_facts, Mapping)
            else persisted_agent_facts
        ),
    )
    executor_events = events_by_node.get("typed_action_executor") or []
    control_hash = model_audit.get("typed_action_control_hash")
    audit.check(
        "typed_actions_executed_with_bound_control",
        _executor_plan_control_bindings_valid(
            response,
            typed_action_facts=typed_action_facts,
            executor_events=executor_events,
            final_control_hash=control_hash,
        ),
        code="typed_action_executor_binding_invalid",
        expected={
            "final_control_hash": control_hash,
            "per_plan_control_binding": True,
        },
        actual=[event.get("scores") for event in executor_events],
    )
    evaluator_events = events_by_node.get("evidence_evaluator") or []
    evaluator_verdicts = [
        (event.get("scores") or {}).get("verdict") or {}
        for event in evaluator_events
    ]
    allowed_verdicts = {
        "sufficient",
        "need_more_same_node",
        "need_mid_expansion",
        "need_bridge_jump",
        "need_chunk_expansion",
        "need_structure_closure",
        "insufficient_corpus",
        "validator_rejection",
    }
    forbidden_gray_tokens = {
        "continue_path",
        "stop_path_irrelevant",
        "follow_as_bridge",
        "request_structure_closure",
        "gray_zone",
        "gray-zone",
        "gray path",
    }

    def evaluator_verdict_valid(verdict: Mapping[str, Any]) -> bool:
        base = {key: value for key, value in verdict.items() if key != "decision_hash"}
        payload_text = repr(
            {
                "reason": verdict.get("reason"),
                "target_ids": verdict.get("target_ids"),
                "expected_evidence": verdict.get("expected_evidence"),
            }
        ).casefold()
        return (
            verdict.get("verdict") in allowed_verdicts
            and isinstance(verdict.get("reason"), str)
            and bool(str(verdict.get("reason")).strip())
            and isinstance(verdict.get("target_ids"), list)
            and isinstance(verdict.get("expected_evidence"), Mapping)
            and not any(token in payload_text for token in forbidden_gray_tokens)
            and _is_hash(verdict.get("decision_hash"))
            and verdict.get("decision_hash") == stable_hash(base)
        )

    audit.check(
        "evidence_evaluator_decisions_typed",
        bool(evaluator_verdicts)
        and all(
            isinstance(verdict, Mapping)
            and evaluator_verdict_valid(verdict)
            for verdict in evaluator_verdicts
        ),
        code="evidence_evaluator_decision_invalid",
        actual=evaluator_verdicts,
    )
    evidence_gate_events = events_by_node.get("evidence_gate") or []
    evidence_gate_event = (
        evidence_gate_events[0]
        if len(evidence_gate_events) == 1
        else {}
    )
    evidence_gate_scores = (
        evidence_gate_event.get("scores")
        if isinstance(evidence_gate_event.get("scores"), Mapping)
        else {}
    )
    fact_plans = (
        typed_action_facts.get("plans")
        if isinstance(typed_action_facts, Mapping)
        and isinstance(typed_action_facts.get("plans"), list)
        else []
    )
    final_fact_plan = (
        max(
            (
                plan
                for plan in fact_plans
                if isinstance(plan, Mapping)
                and type(plan.get("plan_index")) is int
            ),
            key=lambda plan: int(plan["plan_index"]),
            default=None,
        )
        if fact_plans
        else None
    )
    persisted_run = (
        persisted_agent_facts.get("run")
        if isinstance(persisted_agent_facts, Mapping)
        and isinstance(persisted_agent_facts.get("run"), Mapping)
        else {}
    )
    persisted_run_metadata = (
        persisted_run.get("metadata")
        if isinstance(persisted_run.get("metadata"), Mapping)
        else {}
    )
    persisted_bindings = (
        persisted_agent_facts.get("bindings")
        if isinstance(persisted_agent_facts, Mapping)
        and isinstance(
            persisted_agent_facts.get("bindings"),
            Mapping,
        )
        else {}
    )
    persisted_initial_retrieval = (
        persisted_bindings.get("initial_retrieval_trace")
        if isinstance(
            persisted_bindings.get("initial_retrieval_trace"),
            Mapping,
        )
        else {}
    )
    evidence_gate_retrieval_trace_id = (
        persisted_initial_retrieval.get("id")
    )
    audit.check(
        "strict_completed_evidence_gate_payload_replayed",
        len(evidence_gate_events) == 1
        and evidence_gate_event.get("status") == "completed"
        and set(evidence_gate_scores) == EVIDENCE_GATE_SCORE_FIELDS
        and evidence_gate_scores.get(
            "context_package_evidence_gate_passed"
        )
        is True
        and evidence_gate_scores.get("answer_model_called") is False
        and evidence_gate_scores.get("retrieval_trace_id")
        == evidence_gate_retrieval_trace_id
        and evidence_gate_scores.get("retrieval_granularity")
        in {"mid", "coarse"}
        and final_fact_plan is not None
        and evidence_gate_scores.get("agent_plan_id")
        == final_fact_plan.get("plan_id")
        and evidence_gate_scores.get("agent_plan_index")
        == final_fact_plan.get("plan_index")
        and evidence_gate_scores.get("planning_rounds_used")
        == model_audit.get("planning_rounds_used")
        == int(final_fact_plan.get("plan_index")) + 1
        and evidence_gate_scores.get("typed_action_control_hash")
        == model_audit.get("typed_action_control_hash")
        and evaluator_verdicts
        and stable_hash(
            evidence_gate_scores.get("evidence_evaluator")
        )
        == stable_hash(evaluator_verdicts[-1])
        and isinstance(
            evidence_gate_scores.get("evidence_evaluator"),
            Mapping,
        )
        and evidence_gate_scores["evidence_evaluator"].get("verdict")
        == "sufficient"
        and evidence_gate_scores.get(
            "conversation_state_scope_hash"
        )
        == persisted_run_metadata.get(
            "conversation_state_scope_hash"
        ),
        code="evidence_gate_payload_invalid",
        expected={
            "status": "completed",
            "score_fields": sorted(EVIDENCE_GATE_SCORE_FIELDS),
            "retrieval_trace_id": (
                evidence_gate_retrieval_trace_id
            ),
            "context_package_evidence_gate_passed": True,
            "answer_model_called": False,
            "evidence_evaluator_verdict": "sufficient",
        },
        actual={
            "status": evidence_gate_event.get("status"),
            "scores": evidence_gate_scores,
        },
    )
    planning_rounds_used = model_audit.get("planning_rounds_used")
    planning_round_budget = model_audit.get("planning_round_budget")
    remaining_by_round = [
        (event.get("scores") or {}).get("planning_rounds_remaining")
        for event in evaluator_events
    ]
    derived_budget = (
        int(planning_rounds_used) + int(remaining_by_round[-1])
        if type(planning_rounds_used) is int
        and remaining_by_round
        and type(remaining_by_round[-1]) is int
        else None
    )
    effective_planning_budget = (
        planning_round_budget
        if type(planning_round_budget) is int
        else derived_budget
    )
    audit.check(
        "planner_evaluator_budget_replayed",
        type(planning_rounds_used) is int
        and planning_rounds_used > 0
        and type(effective_planning_budget) is int
        and effective_planning_budget >= planning_rounds_used
        and len(evaluator_events) == planning_rounds_used,
        code="planner_evaluator_budget_mismatch",
        expected={"evaluator_count": len(evaluator_events), "budget": effective_planning_budget},
        actual=planning_rounds_used,
    )
    raw_repair_actions = model_audit.get("repair_actions")
    raw_repair_actions = (
        raw_repair_actions if isinstance(raw_repair_actions, list) else []
    )
    repair_actions = [
        item for item in raw_repair_actions if isinstance(item, Mapping)
    ]
    repair_records_complete = len(repair_actions) == len(
        raw_repair_actions
    )
    guard_records = [
        item
        for item in repair_actions
        if item.get("action_type")
        == "claim_level_final_grounded_gate"
    ]
    repair_rounds = [
        item for item in repair_actions if item not in guard_records
    ]
    repair_round_budget = model_audit.get("repair_round_budget")
    repair_rounds_used = model_audit.get("repair_rounds_used")
    audit.check(
        "repair_budget_replayed",
        type(repair_round_budget) is int
        and repair_round_budget >= 0
        and type(repair_rounds_used) is int
        and 0 <= repair_rounds_used <= repair_round_budget
        and repair_records_complete
        and len(guard_records) <= 1
        and all(
            item.get("repair_round_index") is None
            and item.get("protocol_version")
            and _is_hash(item.get("typed_action_control_hash"))
            and item.get("deterministic_citation_guard") is True
            and item.get("gray_zone_model_call_count") == 0
            for item in guard_records
        )
        and repair_rounds_used == len(repair_rounds)
        and [item.get("repair_round_index") for item in repair_rounds] == list(range(len(repair_rounds)))
        and all(
            item.get("action_type") in REPAIR_ACTION_TYPES
            and type(item.get("remaining_repair_budget_before")) is int
            and type(item.get("remaining_repair_budget_after")) is int
            and item.get("remaining_repair_budget_after")
            == item.get("remaining_repair_budget_before") - 1
            and item.get("gray_zone_model_call_count") == 0
            and item.get("gray_zone_decision_authority")
            == "deterministic_executor_only"
            and (item.get("repair_audit") or {}).get("gray_zone_model_call_count")
            == 0
            and (item.get("repair_audit") or {}).get(
                "gray_zone_decision_authority"
            )
            == "deterministic_executor_only"
            and item.get("global_top_k_increased") is False
            for item in repair_rounds
        ),
        code="repair_budget_or_action_mismatch",
        actual={
            "budget": repair_round_budget,
            "used": repair_rounds_used,
            "rounds": repair_rounds,
        },
    )
    if repair_rounds:
        audit.check(
            "repair_trace_events_replayed",
            len(events_by_node.get("repair_executed") or []) >= len(repair_rounds),
            code="repair_trace_event_missing",
            expected=len(repair_rounds),
            actual=len(events_by_node.get("repair_executed") or []),
        )
    grounding_outcome = model_audit.get("grounding_outcome")
    grounding_gate = model_audit.get("claim_grounded_gate") or {}
    audit.check(
        "grounded_answer_gate_passed",
        grounding_outcome == "grounded_answer"
        and grounding_gate.get("all_claims_supported") is True
        and bool(package_id)
        and bool(trace_id)
        and bool(citations)
        and model_audit.get("insufficient_evidence") is not True,
        code="grounded_answer_gate_failed",
        actual={"grounding_outcome": grounding_outcome, "claim_grounded_gate": grounding_gate},
    )
    supported_citations = 0
    for index, citation in enumerate(citations):
        scope = f"citation[{index}]"
        source_span = citation.get("source_span") or {}
        verification = citation.get("verification") or {}
        verification_diagnostics = verification.get("diagnostics") or {}
        provenance_valid = (
            verification.get("verdict") == "supported"
            and verification.get("failure_type") == "none"
            and verification.get("provenance_status") == "valid"
            and verification.get("structure_context_status") == "valid"
            and verification_diagnostics.get("citation_provenance_valid") is True
            and verification_diagnostics.get("citation_provenance_fail_closed") is True
            and verification_diagnostics.get("citation_provenance_llm_override_allowed") is False
            and not (verification_diagnostics.get("citation_provenance_reasons") or [])
            and _is_hash(verification_diagnostics.get("citation_provenance_hash"))
            and _is_hash(verification_diagnostics.get("citation_provenance_session_hash"))
            and verification_diagnostics.get("citation_provenance_persistence_gate_passed") is True
        )
        if provenance_valid:
            supported_citations += 1
        audit.check(
            f"{scope}_raw_provenance_replayed",
            citation.get("context_package_id") == package_id
            and citation.get("retrieval_trace_id") == trace_id
            and citation.get("citation_verification_id")
            and source_span.get("verification_id") == citation.get("citation_verification_id")
            and _source_span_valid(source_span, package_id=str(package_id), trace_id=str(trace_id))
            and source_span.get("chunk_id") == citation.get("chunk_id")
            and provenance_valid,
            code="citation_raw_provenance_invalid",
            scope=scope,
        )
    recomputed_rate = supported_citations / len(citations) if citations else 0.0
    audit.check(
        "citation_verification_rate_replayed",
        _finite(model_audit.get("citation_verification_pass_rate"))
        and math.isclose(float(model_audit["citation_verification_pass_rate"]), recomputed_rate, abs_tol=TOLERANCE)
        and model_audit.get("returned_citation_count") == len(citations),
        code="citation_verification_rate_mismatch",
        expected={"pass_rate": recomputed_rate, "returned_count": len(citations)},
        actual={
            "pass_rate": model_audit.get("citation_verification_pass_rate"),
            "returned_count": model_audit.get("returned_citation_count"),
        },
    )
    try:
        recomputed_retrieval_quality = (
            audit_retrieval_quality(
                retrieval_snapshot,
                gray_zone_audit=gray_zone_audit,
            )
            if isinstance(retrieval_snapshot, Mapping)
            else {}
        )
        retrieval_recompute_error = None
    except Exception as exc:
        recomputed_retrieval_quality = {}
        retrieval_recompute_error = f"{type(exc).__name__}: {exc}"
    supplied_matches = (
        retrieval_quality is None
        or (
            isinstance(retrieval_quality, Mapping)
            and stable_hash(dict(retrieval_quality))
            == stable_hash(recomputed_retrieval_quality)
        )
    )
    audit.check(
        "retrieval_quality_gate_bound_to_agent",
        retrieval_recompute_error is None
        and isinstance(retrieval_snapshot, Mapping)
        and recomputed_retrieval_quality.get("protocol_version")
        == QUALITY_GATE_PROTOCOL_VERSION
        and recomputed_retrieval_quality.get("protocol_hash")
        == QUALITY_GATE_PROTOCOL_HASH
        and recomputed_retrieval_quality.get("subject") == "retrieval"
        and recomputed_retrieval_quality.get("pass") is True
        and recomputed_retrieval_quality.get("finding_count") == 0
        and isinstance(recomputed_retrieval_quality.get("checks"), Mapping)
        and bool(recomputed_retrieval_quality.get("checks"))
        and recomputed_retrieval_quality.get("findings") == []
        and retrieval_snapshot.get("trace_id") == trace_id
        and supplied_matches,
        code="agent_retrieval_quality_gate_failed",
        actual={
            "error": retrieval_recompute_error,
            "recomputed": recomputed_retrieval_quality,
            "supplied": retrieval_quality,
        },
    )
    return audit.finish()


def audit_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    subject = payload.get("subject")
    snapshot = payload.get("snapshot") or {}
    if subject == "graph":
        return audit_graph_quality(snapshot)
    if subject == "retrieval":
        return audit_retrieval_quality(
            snapshot,
            gray_zone_audit=payload.get("gray_zone_audit"),
        )
    if subject == "context_package":
        return audit_context_package_quality(snapshot)
    if subject == "agent":
        return audit_agent_quality(
            snapshot,
            retrieval_snapshot=payload.get("retrieval_snapshot"),
            gray_zone_audit=payload.get("gray_zone_audit"),
            typed_action_facts=payload.get("typed_action_facts"),
            retrieval_quality=payload.get("retrieval_quality"),
            persisted_agent_facts=payload.get(
                "persisted_agent_facts"
            ),
        )
    audit = _Audit("unknown")
    audit.check(
        "known_fixture_subject",
        False,
        code="unknown_quality_gate_subject",
        expected=QUALITY_GATE_PROTOCOL["subjects"],
        actual=subject,
    )
    return audit.finish()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one versioned four-layer acceptance quality fixture without external writes."
    )
    parser.add_argument("--fixture", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.fixture.read_text(encoding="utf-8"))
    result = audit_fixture(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
