from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import exists, func, or_, select
from sqlalchemy.orm import Session, aliased

from app.core.config import REBUILD_REQUIRED_SETTINGS, get_settings
from app.models import (
    AutoTpeRun,
    AutoTpeTrial,
    Chunk,
    ChunkSpan,
    ChunkStructureEdge,
    ChunkStructureMapping,
    ChunkStructureNode,
    CitationVerification,
    ContextPackage,
    KnowledgeBase,
)
from app.services.cancellation import IngestionCancelled, ensure_not_cancelled
from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
from app.services.context_graph import (
    CHUNK_NODE_QUALITY_COMPONENT_WEIGHTS,
    CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
    EDGE_DISTANCE_PROTOCOL_VERSION,
    EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS,
    EDGE_TYPE_CALIBRATION_PARAM_BOUNDS,
    EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
    GRAPH_OPERATING_POINT_PROTOCOL_VERSION,
    IN_ACCEPTANCE_CAPACITY_COMPONENT_WEIGHTS,
    IN_ACCEPTANCE_CAPACITY_PROTOCOL_VERSION,
    OUT_EVIDENCE_MASS_COMPONENT_WEIGHTS,
    OUT_EVIDENCE_MASS_PROTOCOL_VERSION,
    RANK_SCORE_PROTOCOL_VERSION,
    RELATION_QUOTA_PROTOCOL_VERSION,
    RELATION_PROTOCOL_VERSION,
    RELATION_RAW_STRENGTH_PROTOCOL_VERSION,
    RQ_LEVELS,
    RelationEdgeCandidate,
    channel_rank_score,
    chunk_node_quality_protocol_hash,
    compute_chunk_scope_hash,
    dense_graph_operating_point,
    edge_distance_protocol_hash,
    edge_type_calibration_config,
    edge_type_calibration_protocol_hash,
    in_acceptance_capacity_protocol_hash,
    out_evidence_mass_protocol_hash,
    relation_edge_candidates,
    relation_quota_card,
    relation_quota_protocol_hash,
    relation_quota_signal_config,
    relation_rank_score_protocol_hash,
    relation_raw_strength,
    relation_raw_strength_protocol_hash,
    encode_rq_vectors_batch,
    rq_runtime_config,
    runtime_settings_snapshot,
    stable_hash,
    train_rq_kmeans,
)
from app.services.ingestion_logs import emit_ingestion_log
from app.services.graph_state_hashes import (
    CHUNK_BUSINESS_KEY_PROTOCOL_VERSION,
    chunk_business_references,
)
from app.services.language_metadata import load_chunk_language_identities
from app.services import runtime_settings as runtime_settings_service
from app.services.runtime_settings import refresh_runtime_settings_if_needed
from app.services.tpe_audit import (
    TPE_GATE_PROFILE_PROTOCOL_VERSION,
    TPE_SELECTED_PENDING_STATUS,
    bind_tpe_graph_promotion_state,
    persist_tpe_run,
    persist_tpe_trial,
    promotion_lease_expires_at,
    reconcile_tpe_audit,
    register_tpe_graph_promotion,
    transition_tpe_run,
    transition_tpe_trial,
    trial_lease_expires_at,
    tpe_run_has_valid_active_promotion,
    tpe_trial_is_valid,
    update_tpe_run,
)


AUTO_TPE_PROTOCOL = "auto_tpe_lightweight_graph_operating_point_v8"
TPE_QUALITY_PROXY_PROTOCOL_VERSION = (
    "tpe_expected_support_structure_coverage_diversity_v4"
)
TPE_EXPECTED_SUPPORT_PROTOCOL_VERSION = (
    "tpe_manual_verified_structure_positive_support_v2"
)
TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION = (
    "tpe_previous_next_section_page_special_closure_v1"
)
TPE_COMPONENT_COVERAGE_PROTOCOL_VERSION = (
    "tpe_chunk_document_language_candidate_rq_equal_weight_v2"
)
TPE_PATH_DIVERSITY_PROTOCOL_VERSION = (
    "tpe_probe_two_hop_document_language_edge_type_candidate_rq_entropy_v1"
)
TPE_EDGE_PRECISION_PROTOCOL_VERSION = (
    "tpe_typed_gate_support_feature_structure_traceability_v2"
)
TPE_BRIDGE_OPPORTUNITY_PROTOCOL_VERSION = (
    "tpe_dedicated_cross_document_language_quota_opportunity_v1"
)
TPE_HUBNESS_PENALTY_PROTOCOL_VERSION = (
    "tpe_degree_ratio_top_five_percent_share_edge_type_imbalance_v1"
)
TPE_DENSITY_PENALTY_PROTOCOL_VERSION = (
    "tpe_normalized_undirected_density_soft_ceiling_v2"
)
TPE_DENSITY_SOFT_MAX_FRACTION = 0.75
TPE_NORMALIZED_DENSITY_PROTOCOL_VERSION = (
    "tpe_normalized_undirected_simple_graph_density_v1"
)
TPE_SCOPE_SPARSE_EDGE_BUDGET_PROTOCOL_VERSION = (
    "tpe_scope_sparse_edge_budget_log2_v1"
)
TPE_SCOPE_SEARCH_BOUNDS_PROTOCOL_VERSION = (
    "tpe_scope_sparse_integer_bounds_v1"
)
TPE_TOP_HUB_FRACTION = 0.05
TPE_EXPECTED_SUPPORT_PER_PROBE_LIMIT = 512
TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT = 512
TPE_VERIFIED_CITATION_ROW_LIMIT = 4096
TPE_PRECISION_SAMPLE_LIMIT = 512
TPE_RUNTIME_ENVELOPE_KEYS = (
    "enable_auto_tpe",
    "tpe_trial_budget",
    "tpe_startup_random_trials",
    "tpe_good_quantile_gamma",
    "tpe_probe_query_budget",
    "tpe_trial_timeout_seconds",
    "tpe_candidate_pool_size",
    "operating_point_hard_gate_max_edge_density",
    "operating_point_hard_gate_max_isolated_ratio",
    "operating_point_hard_gate_max_hubness_ratio",
    "operating_point_hard_gate_min_structure_recovery_rate",
    "operating_point_hard_gate_max_candidate_latency_p95_ms",
)
TPE_SEARCH_SPACE_PROTOCOL_VERSION = "tpe_graph_operating_point_search_space_v3"
TPE_CANDIDATE_ADJACENCY_HASH_PROTOCOL = (
    "tpe_candidate_adjacency_theta_typed_gate_language_scope_v4"
)
TPE_LATENCY_PROTOCOL_VERSION = "tpe_local_latency_segment_nearest_rank_p95_v1"
TPE_LATENCY_PENALTY_PROTOCOL_VERSION = "tpe_latency_budget_excess_ratio_v1"
TPE_LATENCY_SAMPLE_LIMIT = 512
TPE_DIAGNOSTIC_SAMPLE_LIMIT = 64
TPE_INTEGER_SEARCH_BOUNDS = {
    "dense_knn_k_min": (1, 32),
    "dense_knn_k_max": (1, 64),
    "dense_reverse_b_min_base": (1, 32),
    "dense_reverse_b_max_base": (1, 64),
    "dense_reverse_b_min_doc": (0, 32),
    "dense_reverse_b_max_doc": (1, 64),
    "dense_reverse_b_min_lang": (0, 32),
    "dense_reverse_b_max_lang": (1, 64),
    "cross_doc_out_quota_min": (0, 32),
    "cross_doc_out_quota_max": (1, 64),
    "cross_language_out_quota_min": (0, 32),
    "cross_language_out_quota_max": (1, 64),
}
TPE_FLOAT_SEARCH_BOUNDS = {
    "dense_min_cosine": (0.05, 0.95),
    "dense_strong_cosine": (0.06, 0.99),
    "cross_doc_min_cosine": (0.05, 0.95),
    "cross_language_min_cosine": (0.05, 0.95),
}
TPE_NUMERIC_THETA_KEYS = (
    "dense_knn_k_min",
    "dense_knn_k_max",
    "dense_reverse_b_min_base",
    "dense_reverse_b_max_base",
    "dense_reverse_b_min_doc",
    "dense_reverse_b_max_doc",
    "dense_reverse_b_min_lang",
    "dense_reverse_b_max_lang",
    "dense_min_cosine",
    "dense_strong_cosine",
    "cross_doc_out_quota_min",
    "cross_doc_out_quota_max",
    "cross_doc_min_cosine",
    "cross_language_out_quota_min",
    "cross_language_out_quota_max",
    "cross_language_min_cosine",
)
TPE_CALIBRATION_PARAM_KEYS = tuple(EDGE_TYPE_CALIBRATION_DEFAULT_PARAMS)
TPE_NUMERIC_THETA_PATHS = TPE_NUMERIC_THETA_KEYS + tuple(
    f"calibration_params.{key}" for key in TPE_CALIBRATION_PARAM_KEYS
)
TPE_THETA_KEYS = (
    "graph_operating_point_protocol",
    "edge_distance_protocol",
    "edge_distance_protocol_hash",
    "edge_type_calibration_protocol",
    "edge_type_calibration_protocol_hash",
    "calibration_params",
    *TPE_NUMERIC_THETA_KEYS,
)
TPE_IMMUTABLE_THETA_KEYS = (
    "optimizer",
    "rank_score_protocol_version",
    "rank_score_protocol_hash",
    "raw_strength_protocol_version",
    "raw_strength_protocol_hash",
    "chunk_node_quality_protocol",
    "chunk_node_quality_protocol_hash",
    "out_evidence_mass_protocol",
    "out_evidence_mass_protocol_hash",
    "in_acceptance_capacity_protocol",
    "in_acceptance_capacity_protocol_hash",
    "relation_quota_protocol",
    "relation_quota_protocol_hash",
    "quota_signal_scale",
)
TPE_ALLOWED_THETA_KEYS = frozenset((*TPE_THETA_KEYS, *TPE_IMMUTABLE_THETA_KEYS))
TPE_TERMINAL_TRIAL_STATES = {"completed", "blocked", "failed", "cancelled"}
def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.utcnow()


def _protocol_hash() -> str:
    return stable_hash(
        {
            "auto_tpe_protocol": AUTO_TPE_PROTOCOL,
            "candidate_adjacency_hash_protocol": TPE_CANDIDATE_ADJACENCY_HASH_PROTOCOL,
            "tpe_search_space_hash": tpe_search_space_hash(),
            "relation_protocol": RELATION_PROTOCOL_VERSION,
            "edge_distance_protocol_hash": edge_distance_protocol_hash(),
            "edge_type_calibration_protocol_hash": edge_type_calibration_protocol_hash(),
            "rank_score_protocol_hash": relation_rank_score_protocol_hash(),
            "raw_strength_protocol_hash": relation_raw_strength_protocol_hash(),
            "chunk_node_quality_protocol_hash": chunk_node_quality_protocol_hash(),
            "out_evidence_mass_protocol_hash": out_evidence_mass_protocol_hash(),
            "in_acceptance_capacity_protocol_hash": in_acceptance_capacity_protocol_hash(),
            "relation_quota_protocol_hash": relation_quota_protocol_hash(),
            "tpe_latency_protocol_version": TPE_LATENCY_PROTOCOL_VERSION,
            "tpe_latency_penalty_protocol_version": (
                TPE_LATENCY_PENALTY_PROTOCOL_VERSION
            ),
            "tpe_quality_proxy_protocol_version": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
            "tpe_expected_support_protocol_version": TPE_EXPECTED_SUPPORT_PROTOCOL_VERSION,
            "tpe_structure_recovery_protocol_version": TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
            "tpe_component_coverage_protocol_version": TPE_COMPONENT_COVERAGE_PROTOCOL_VERSION,
            "tpe_path_diversity_protocol_version": TPE_PATH_DIVERSITY_PROTOCOL_VERSION,
            "tpe_edge_precision_protocol_version": TPE_EDGE_PRECISION_PROTOCOL_VERSION,
            "tpe_bridge_opportunity_protocol_version": TPE_BRIDGE_OPPORTUNITY_PROTOCOL_VERSION,
            "tpe_hubness_penalty_protocol_version": TPE_HUBNESS_PENALTY_PROTOCOL_VERSION,
            "tpe_density_penalty_protocol_version": TPE_DENSITY_PENALTY_PROTOCOL_VERSION,
            "tpe_normalized_density_protocol_version": (
                TPE_NORMALIZED_DENSITY_PROTOCOL_VERSION
            ),
            "tpe_scope_sparse_edge_budget_protocol_version": (
                TPE_SCOPE_SPARSE_EDGE_BUDGET_PROTOCOL_VERSION
            ),
            "tpe_scope_search_bounds_protocol_version": (
                TPE_SCOPE_SEARCH_BOUNDS_PROTOCOL_VERSION
            ),
        }
    )


def tpe_search_space_spec() -> dict[str, Any]:
    immutable_identity = {
        "edge_distance_protocol_hash": edge_distance_protocol_hash(),
        "edge_type_calibration_protocol_hash": edge_type_calibration_protocol_hash(),
        "rank_score_protocol_version": RANK_SCORE_PROTOCOL_VERSION,
        "rank_score_protocol_hash": relation_rank_score_protocol_hash(),
        "raw_strength_protocol_version": RELATION_RAW_STRENGTH_PROTOCOL_VERSION,
        "raw_strength_protocol_hash": relation_raw_strength_protocol_hash(),
        **relation_quota_signal_config(None),
    }
    return {
        "protocol_version": TPE_SEARCH_SPACE_PROTOCOL_VERSION,
        "categorical": {
            "graph_operating_point_protocol": [GRAPH_OPERATING_POINT_PROTOCOL_VERSION],
            "edge_distance_protocol": [EDGE_DISTANCE_PROTOCOL_VERSION],
            "edge_type_calibration_protocol": [EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION],
        },
        "metadata_allowlists": {
            "optimizer": [
                "auto_tpe_lightweight",
                "auto_tpe_lightweight_or_default",
            ],
        },
        "immutable_identity": immutable_identity,
        "integer_bounds": {
            key: {"lower": lower, "upper": upper}
            for key, (lower, upper) in TPE_INTEGER_SEARCH_BOUNDS.items()
        },
        "float_bounds": {
            key: {"lower": lower, "upper": upper, "round_digits": 4}
            for key, (lower, upper) in TPE_FLOAT_SEARCH_BOUNDS.items()
        },
        "calibration_param_bounds": {
            key: {"lower": lower, "upper": upper, "round_digits": 6}
            for key, (lower, upper) in EDGE_TYPE_CALIBRATION_PARAM_BOUNDS.items()
        },
        "constraints": [
            "all_min_quota_values_must_not_exceed_corresponding_max_values",
            "dense_strong_cosine_must_exceed_every_typed_min_cosine",
            "lower_quantile_must_be_below_upper_quantile",
            "integer_search_bounds_are_clipped_by_chunk_scope_sparse_allowance",
            "all_protocols_and_protocol_hashes_must_match_local_allowlists",
        ],
        "theta_fields": sorted(TPE_ALLOWED_THETA_KEYS),
        "sampled_fields": [
            "graph_operating_point_protocol",
            "edge_distance_protocol",
            "edge_type_calibration_protocol",
            *TPE_NUMERIC_THETA_PATHS,
        ],
        "audit_companion_fields": sorted(immutable_identity),
        "numeric_paths": list(TPE_NUMERIC_THETA_PATHS),
    }


def tpe_search_space_hash() -> str:
    return stable_hash(tpe_search_space_spec())


def theta_calibration_audit(theta: dict[str, Any]) -> dict[str, Any]:
    config = edge_type_calibration_config(theta)
    return {
        "graph_operating_point_protocol": config["graph_operating_point_protocol"],
        "edge_distance_protocol": config["edge_distance_protocol"],
        "edge_distance_protocol_hash": config["edge_distance_protocol_hash"],
        "edge_type_calibration_protocol": config["protocol"],
        "edge_type_calibration_protocol_hash": config["protocol_hash"],
        "calibration_params": dict(config["params"]),
        "calibration_params_hash": config["params_hash"],
        "edge_type_calibration_config_hash": config["config_hash"],
    }


def _relation_quota_protocol_diagnostics() -> dict[str, Any]:
    return {
        "relation_protocol_version": RELATION_PROTOCOL_VERSION,
        "chunk_node_quality_protocol_version": CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
        "out_evidence_mass_protocol_version": OUT_EVIDENCE_MASS_PROTOCOL_VERSION,
        "in_acceptance_capacity_protocol_version": IN_ACCEPTANCE_CAPACITY_PROTOCOL_VERSION,
        "relation_quota_protocol_version": RELATION_QUOTA_PROTOCOL_VERSION,
        **relation_quota_signal_config(None),
        "out_and_in_signals_are_independent": True,
        "quota_signals_used_as_query_relevance": False,
    }


def _runtime_hash() -> str:
    current = runtime_settings_service.get_settings()

    def value_for(key: str) -> Any:
        value = getattr(current, key)
        return str(value) if isinstance(value, Path) else value

    return stable_hash(
        {
            "protocol_version": "auto_tpe_runtime_profile_v2",
            "graph_rebuild_settings": {
                key: value_for(key) for key in sorted(REBUILD_REQUIRED_SETTINGS)
            },
            "optimizer_envelope": {
                key: value_for(key) for key in TPE_RUNTIME_ENVELOPE_KEYS
            },
            "quality_proxy_protocol_version": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
        }
    )


def _graph_runtime_hash() -> str:
    """Identity persisted on the graph; optimizer-envelope settings stay hot."""
    return stable_hash(runtime_settings_snapshot())


def tpe_gate_profile(settings: Any | None = None) -> dict[str, Any]:
    current = settings or get_settings()
    return {
        "protocol": TPE_GATE_PROFILE_PROTOCOL_VERSION,
        "tpe_probe_query_budget": int(current.tpe_probe_query_budget),
        "tpe_trial_timeout_seconds": float(current.tpe_trial_timeout_seconds),
        "hard_gate_thresholds": {
            "edge_density": {
                "direction": "max",
                "threshold": float(current.operating_point_hard_gate_max_edge_density),
                "metric_protocol": TPE_NORMALIZED_DENSITY_PROTOCOL_VERSION,
            },
            "sparse_edge_budget_ratio": {
                "direction": "max",
                "threshold": 1.0,
                "budget_protocol": TPE_SCOPE_SPARSE_EDGE_BUDGET_PROTOCOL_VERSION,
            },
            "isolated_ratio": {
                "direction": "max",
                "threshold": float(current.operating_point_hard_gate_max_isolated_ratio),
            },
            "hubness_ratio": {
                "direction": "max",
                "threshold": float(current.operating_point_hard_gate_max_hubness_ratio),
            },
            "structure_recovery_rate": {
                "direction": "min",
                "threshold": float(
                    current.operating_point_hard_gate_min_structure_recovery_rate
                ),
            },
            "candidate_latency_p95_ms": {
                "direction": "max",
                "threshold": float(
                    current.operating_point_hard_gate_max_candidate_latency_p95_ms
                ),
            },
        },
    }


def tpe_gate_profile_hash(profile: dict[str, Any]) -> str:
    return stable_hash(dict(profile))


def _theta_hash(theta: dict[str, Any]) -> str:
    return stable_hash({key: theta.get(key) for key in sorted(theta)})


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _bounded_float(value: Any, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def normalize_theta(theta: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(theta)
    for key, (lower, upper) in TPE_INTEGER_SEARCH_BOUNDS.items():
        if key in normalized:
            normalized[key] = _bounded_int(normalized[key], lower, upper)
    for key, (lower, upper) in TPE_FLOAT_SEARCH_BOUNDS.items():
        if key in normalized:
            normalized[key] = round(_bounded_float(normalized[key], lower, upper), 4)
    supplied_calibration_params = normalized.get("calibration_params")
    if isinstance(supplied_calibration_params, dict):
        calibration_params = dict(supplied_calibration_params)
        for key, (lower, upper) in EDGE_TYPE_CALIBRATION_PARAM_BOUNDS.items():
            if key in calibration_params:
                calibration_params[key] = round(
                    _bounded_float(calibration_params[key], lower, upper),
                    6,
                )
        normalized["calibration_params"] = calibration_params
    for min_key, max_key in (
        ("dense_knn_k_min", "dense_knn_k_max"),
        ("dense_reverse_b_min_base", "dense_reverse_b_max_base"),
        ("dense_reverse_b_min_doc", "dense_reverse_b_max_doc"),
        ("dense_reverse_b_min_lang", "dense_reverse_b_max_lang"),
        ("cross_doc_out_quota_min", "cross_doc_out_quota_max"),
        ("cross_language_out_quota_min", "cross_language_out_quota_max"),
    ):
        if (
            min_key in normalized
            and max_key in normalized
            and int(normalized[min_key]) > int(normalized[max_key])
        ):
            normalized[max_key] = normalized[min_key]
    threshold_keys = (
        "dense_min_cosine",
        "cross_doc_min_cosine",
        "cross_language_min_cosine",
    )
    if "dense_strong_cosine" in normalized and all(
        key in normalized for key in threshold_keys
    ):
        min_threshold = max(float(normalized[key]) for key in threshold_keys)
    else:
        min_threshold = None
    if (
        min_threshold is not None
        and float(normalized["dense_strong_cosine"]) <= min_threshold
    ):
        normalized["dense_strong_cosine"] = round(min(0.99, min_threshold + 0.08), 4)
    normalized["optimizer"] = "auto_tpe_lightweight"
    normalized["rank_score_protocol_version"] = RANK_SCORE_PROTOCOL_VERSION
    normalized["rank_score_protocol_hash"] = relation_rank_score_protocol_hash()
    normalized["raw_strength_protocol_version"] = RELATION_RAW_STRENGTH_PROTOCOL_VERSION
    normalized["raw_strength_protocol_hash"] = relation_raw_strength_protocol_hash()
    normalized.update(relation_quota_signal_config(None))
    return normalized


def preflight_theta(theta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not isinstance(theta, dict):
        return ["invalid_theta_type"]
    for key in (*TPE_THETA_KEYS, *TPE_IMMUTABLE_THETA_KEYS):
        if key not in theta:
            reasons.append(f"missing:{key}")
    for key in sorted(set(theta) - TPE_ALLOWED_THETA_KEYS):
        reasons.append(f"unknown:{key}")
    for key, (lower, upper) in TPE_INTEGER_SEARCH_BOUNDS.items():
        if key not in theta:
            continue
        try:
            if type(theta[key]) is not int:
                raise ValueError
            value = int(theta[key])
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"invalid_integer:{key}")
            continue
        if value < lower or value > upper:
            reasons.append(f"out_of_bounds:{key}")
    for key, (lower, upper) in TPE_FLOAT_SEARCH_BOUNDS.items():
        if key not in theta:
            continue
        try:
            if isinstance(theta[key], bool) or not isinstance(theta[key], (int, float)):
                raise ValueError
            value = float(theta[key])
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"invalid_float:{key}")
            continue
        if not math.isfinite(value) or value < lower or value > upper:
            reasons.append(f"out_of_bounds:{key}")
    calibration_params = theta.get("calibration_params")
    if "calibration_params" in theta and not isinstance(calibration_params, dict):
        reasons.append("invalid_object:calibration_params")
    elif isinstance(calibration_params, dict):
        for key, (lower, upper) in EDGE_TYPE_CALIBRATION_PARAM_BOUNDS.items():
            if key not in calibration_params:
                continue
            value = calibration_params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append(f"invalid_float:calibration_params.{key}")
                continue
            numeric_value = float(value)
            if (
                not math.isfinite(numeric_value)
                or numeric_value < lower
                or numeric_value > upper
            ):
                reasons.append(f"out_of_bounds:calibration_params.{key}")
    for min_key, max_key in (
        ("dense_knn_k_min", "dense_knn_k_max"),
        ("dense_reverse_b_min_base", "dense_reverse_b_max_base"),
        ("dense_reverse_b_min_doc", "dense_reverse_b_max_doc"),
        ("dense_reverse_b_min_lang", "dense_reverse_b_max_lang"),
        ("cross_doc_out_quota_min", "cross_doc_out_quota_max"),
        ("cross_language_out_quota_min", "cross_language_out_quota_max"),
    ):
        try:
            if int(theta[min_key]) > int(theta[max_key]):
                reasons.append(f"invalid_range:{min_key}:{max_key}")
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    try:
        strong = float(theta["dense_strong_cosine"])
    except (KeyError, TypeError, ValueError, OverflowError):
        strong = None
    if strong is not None and math.isfinite(strong):
        for key in (
            "dense_min_cosine",
            "cross_doc_min_cosine",
            "cross_language_min_cosine",
        ):
            try:
                value = float(theta[key])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value) and strong <= value:
                reasons.append(f"strong_threshold_not_above:{key}")
    if theta.get("optimizer") not in {
        "auto_tpe_lightweight",
        "auto_tpe_lightweight_or_default",
    }:
        reasons.append("invalid_optimizer")
    try:
        edge_type_calibration_config(theta)
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid_edge_calibration:{exc}")
    try:
        expected_quota_config = relation_quota_signal_config(theta)
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid_relation_quota_protocol:{exc}")
    else:
        for key, expected_value in expected_quota_config.items():
            if key not in theta or str(theta.get(key)) != str(expected_value):
                reasons.append(f"invalid_relation_quota_protocol:{key}")
    expected_immutable = {
        "rank_score_protocol_version": RANK_SCORE_PROTOCOL_VERSION,
        "rank_score_protocol_hash": relation_rank_score_protocol_hash(),
        "raw_strength_protocol_version": RELATION_RAW_STRENGTH_PROTOCOL_VERSION,
        "raw_strength_protocol_hash": relation_raw_strength_protocol_hash(),
    }
    for key, expected_value in expected_immutable.items():
        if theta.get(key) != expected_value:
            reasons.append(f"invalid_immutable_protocol:{key}")
    return list(dict.fromkeys(reasons))


def tpe_scope_sparse_edge_budget(chunk_count: int) -> int:
    if type(chunk_count) is not int or chunk_count < 0:
        raise ValueError("TPE chunk scope size must be a non-negative integer")
    if chunk_count <= 1:
        return 0
    complete_edge_count = chunk_count * (chunk_count - 1) // 2
    sparse_edge_count = int(
        math.ceil(chunk_count * max(1.0, math.log2(chunk_count)))
    )
    return min(complete_edge_count, sparse_edge_count)


def tpe_scope_integer_bounds(
    chunk_count: int,
) -> dict[str, tuple[int, int]]:
    if type(chunk_count) is not int or chunk_count < 0:
        raise ValueError("TPE chunk scope size must be a non-negative integer")
    sparse_out_degree_allowance = (
        1
        if chunk_count <= 1
        else max(1, min(chunk_count - 1, int(math.ceil(math.log2(chunk_count)))))
    )
    return {
        key: (
            min(lower, sparse_out_degree_allowance),
            max(
                min(lower, sparse_out_degree_allowance),
                min(upper, sparse_out_degree_allowance),
            ),
        )
        for key, (lower, upper) in TPE_INTEGER_SEARCH_BOUNDS.items()
    }


def tpe_scope_search_card(chunk_count: int) -> dict[str, Any]:
    integer_bounds = tpe_scope_integer_bounds(chunk_count)
    return {
        "protocol_version": TPE_SCOPE_SEARCH_BOUNDS_PROTOCOL_VERSION,
        "chunk_count": int(chunk_count),
        "sparse_edge_budget": tpe_scope_sparse_edge_budget(chunk_count),
        "sparse_edge_budget_protocol_version": (
            TPE_SCOPE_SPARSE_EDGE_BUDGET_PROTOCOL_VERSION
        ),
        "integer_bounds": {
            key: {"lower": lower, "upper": upper}
            for key, (lower, upper) in sorted(integer_bounds.items())
        },
        "model_call_count": 0,
    }


def _normalize_theta_for_scope(
    theta: dict[str, Any],
    *,
    chunk_count: int | None,
) -> dict[str, Any]:
    normalized = normalize_theta(theta)
    if chunk_count is None:
        return normalized
    for key, (lower, upper) in tpe_scope_integer_bounds(chunk_count).items():
        normalized[key] = _bounded_int(normalized[key], lower, upper)
    for min_key, max_key in (
        ("dense_knn_k_min", "dense_knn_k_max"),
        ("dense_reverse_b_min_base", "dense_reverse_b_max_base"),
        ("dense_reverse_b_min_doc", "dense_reverse_b_max_doc"),
        ("dense_reverse_b_min_lang", "dense_reverse_b_max_lang"),
        ("cross_doc_out_quota_min", "cross_doc_out_quota_max"),
        ("cross_language_out_quota_min", "cross_language_out_quota_max"),
    ):
        if int(normalized[min_key]) > int(normalized[max_key]):
            normalized[max_key] = normalized[min_key]
    return normalized


def _scope_anchor_theta(
    base: dict[str, Any],
    *,
    chunk_count: int,
) -> dict[str, Any]:
    bounds = tpe_scope_integer_bounds(chunk_count)
    sparse_allowance = bounds["dense_knn_k_max"][1]
    bridge_allowance = max(1, sparse_allowance // 3)
    theta = {
        **base,
        "dense_knn_k_min": 1,
        "dense_knn_k_max": sparse_allowance,
        "dense_reverse_b_min_base": 1,
        "dense_reverse_b_max_base": sparse_allowance,
        "dense_reverse_b_min_doc": 0,
        "dense_reverse_b_max_doc": bridge_allowance,
        "dense_reverse_b_min_lang": 0,
        "dense_reverse_b_max_lang": bridge_allowance,
        "cross_doc_out_quota_min": 0,
        "cross_doc_out_quota_max": bridge_allowance,
        "cross_language_out_quota_min": 0,
        "cross_language_out_quota_max": bridge_allowance,
    }
    return _normalize_theta_for_scope(theta, chunk_count=chunk_count)


def _random_theta(
    base: dict[str, Any],
    seed: int,
    *,
    chunk_count: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    theta = dict(base)
    integer_bounds = (
        TPE_INTEGER_SEARCH_BOUNDS
        if chunk_count is None
        else tpe_scope_integer_bounds(chunk_count)
    )
    for key, (lower, upper) in integer_bounds.items():
        theta[key] = rng.randint(lower, upper)
    for key, (lower, upper) in TPE_FLOAT_SEARCH_BOUNDS.items():
        theta[key] = round(rng.uniform(lower, upper), 4)
    theta["calibration_params"] = {
        key: round(rng.uniform(lower, upper), 6)
        for key, (lower, upper) in EDGE_TYPE_CALIBRATION_PARAM_BOUNDS.items()
    }
    return _normalize_theta_for_scope(theta, chunk_count=chunk_count)


def _compatible_tpe_observations(
    completed_trials: list[AutoTpeTrial],
    *,
    runtime_settings_hash: str,
    gate_profile_hash: str,
) -> list[AutoTpeTrial]:
    return [
        trial
        for trial in completed_trials
        if tpe_trial_is_valid(trial)
        and trial.runtime_settings_hash == runtime_settings_hash
        and trial.gate_profile_hash == gate_profile_hash
    ]


def _sampler_state_hash(
    run: AutoTpeRun,
    completed_trials: list[AutoTpeTrial],
    next_index: int,
    *,
    runtime_settings_hash: str,
    gate_profile_hash: str,
) -> str:
    compatible_trials = _compatible_tpe_observations(
        completed_trials,
        runtime_settings_hash=runtime_settings_hash,
        gate_profile_hash=gate_profile_hash,
    )
    return stable_hash(
        {
            "run_id": run.id,
            "scope": {
                "knowledge_base_id": run.knowledge_base_id,
                "chunk_version": run.chunk_version,
                "chat_model": run.chat_model,
                "embedding_model": run.embedding_model,
                "embedding_text_version": run.embedding_text_version,
            },
            "next_index": next_index,
            "tpe_search_space_hash": tpe_search_space_hash(),
            "runtime_settings_hash": runtime_settings_hash,
            "gate_profile_hash": gate_profile_hash,
            "completed": [
                {
                    "trial_index": trial.trial_index,
                    "theta_hash": trial.theta_hash,
                    "status": trial.status,
                    "objective_score": trial.objective_score,
                    "failure_code": trial.failure_code,
                    "runtime_settings_hash": getattr(trial, "runtime_settings_hash", None),
                    "gate_profile_hash": getattr(trial, "gate_profile_hash", None),
                }
                for trial in sorted(compatible_trials, key=lambda item: item.trial_index)
            ],
        }
    )


def _theta_numeric_value(theta: dict[str, Any], path: str) -> float:
    if path.startswith("calibration_params."):
        key = path.split(".", 1)[1]
        return float(dict(theta.get("calibration_params") or {})[key])
    return float(theta[path])


def _set_theta_numeric_value(theta: dict[str, Any], path: str, value: float) -> None:
    if path.startswith("calibration_params."):
        key = path.split(".", 1)[1]
        params = dict(theta.get("calibration_params") or {})
        params[key] = value
        theta["calibration_params"] = params
        return
    theta[path] = value


def _mean_for(trials: list[AutoTpeTrial], path: str, base: dict[str, Any]) -> float:
    values: list[float] = []
    for trial in trials:
        theta = dict(trial.sampled_theta_json or {})
        try:
            values.append(_theta_numeric_value(theta, path))
        except (KeyError, TypeError, ValueError, OverflowError):
            values.append(_theta_numeric_value(base, path))
    return sum(values) / max(len(values), 1)


def _sample_tpe_theta(
    run: AutoTpeRun,
    completed_trials: list[AutoTpeTrial],
    trial_index: int,
    base_theta: dict[str, Any],
    *,
    settings: Any,
    runtime_settings_hash: str,
    gate_profile_hash: str,
    chunk_count: int | None = None,
) -> tuple[dict[str, Any], str]:
    sampler_hash = _sampler_state_hash(
        run,
        completed_trials,
        trial_index,
        runtime_settings_hash=runtime_settings_hash,
        gate_profile_hash=gate_profile_hash,
    )
    startup = int(settings.tpe_startup_random_trials)
    valid = _compatible_tpe_observations(
        completed_trials,
        runtime_settings_hash=runtime_settings_hash,
        gate_profile_hash=gate_profile_hash,
    )
    if trial_index == 1 and chunk_count is not None:
        return (
            _scope_anchor_theta(base_theta, chunk_count=chunk_count),
            sampler_hash,
        )
    if len(valid) < startup or not valid:
        return (
            _random_theta(
                base_theta,
                int(stable_hash({"run": run.id, "trial": trial_index})[:8], 16),
                chunk_count=chunk_count,
            ),
            sampler_hash,
        )

    ordered = sorted(valid, key=lambda trial: float(trial.objective_score or 0.0), reverse=True)
    split = max(1, int(math.ceil(len(ordered) * float(settings.tpe_good_quantile_gamma))))
    good = ordered[:split]
    bad = ordered[split:] or ordered[-1:]
    rng = random.Random(int(sampler_hash[:8], 16))
    candidate_pool_size = max(1, int(settings.tpe_candidate_pool_size))
    best_candidate: dict[str, Any] | None = None
    best_ratio = -1.0
    for _ in range(candidate_pool_size):
        candidate = dict(base_theta)
        for path in TPE_NUMERIC_THETA_PATHS:
            good_mean = _mean_for(good, path, base_theta)
            bad_mean = _mean_for(bad, path, base_theta)
            if path.startswith("calibration_params."):
                parameter = path.split(".", 1)[1]
                lower, upper = EDGE_TYPE_CALIBRATION_PARAM_BOUNDS[parameter]
                minimum_spread = max((upper - lower) * 0.08, 1e-6)
            elif "cosine" in path:
                minimum_spread = 0.05
            else:
                minimum_spread = 1.0
            spread = max(abs(good_mean - bad_mean), minimum_spread)
            _set_theta_numeric_value(
                candidate,
                path,
                good_mean + rng.uniform(-0.35, 0.35) * spread,
            )
        candidate = _normalize_theta_for_scope(
            candidate,
            chunk_count=chunk_count,
        )
        if preflight_theta(candidate):
            continue
        ratio = 0.0
        for path in TPE_NUMERIC_THETA_PATHS:
            value = _theta_numeric_value(candidate, path)
            good_distance = abs(value - _mean_for(good, path, base_theta)) + 1e-6
            bad_distance = abs(value - _mean_for(bad, path, base_theta)) + 1e-6
            ratio += bad_distance / good_distance
        if ratio > best_ratio:
            best_candidate = candidate
            best_ratio = ratio
    return (
        best_candidate
        or _random_theta(
            base_theta,
            int(stable_hash({"fallback": sampler_hash})[:8], 16),
            chunk_count=chunk_count,
        ),
        sampler_hash,
    )


def _fallback_tpe_chunk_audit_key(chunk: Any) -> str:
    """A test-only/local audit key; production paths inject full provenance keys."""
    return stable_hash(
        {
            "protocol_version": "tpe_local_chunk_audit_fallback_v1",
            "document_address": str(getattr(chunk, "document_id", "")),
            "chunk_version": int(getattr(chunk, "chunk_version", 0) or 0),
            "chunk_index": int(getattr(chunk, "chunk_index", 0) or 0),
            "token_span": [
                int(getattr(chunk, "token_start", 0) or 0),
                int(getattr(chunk, "token_end", 0) or 0),
            ],
            "char_span": [
                int(getattr(chunk, "char_start", 0) or 0),
                int(getattr(chunk, "char_end", 0) or 0),
            ],
            "section_path": str(getattr(chunk, "section_path", "") or ""),
            "page_range": [
                getattr(chunk, "page_start", None),
                getattr(chunk, "page_end", None),
            ],
            "text_hash": str(getattr(chunk, "text_hash", "") or ""),
        }
    )


def _probe_chunks(
    chunks: list[Chunk],
    budget: int,
    *,
    chunk_business_keys: dict[str, str] | None = None,
) -> tuple[list[Chunk], str]:
    audit_keys = {
        str(chunk.id): str(
            (chunk_business_keys or {}).get(str(chunk.id))
            or _fallback_tpe_chunk_audit_key(chunk)
        )
        for chunk in chunks
    }
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            audit_keys[str(chunk.id)],
            int(chunk.chunk_index),
            str(chunk.text_hash),
        ),
    )
    if len(ordered) <= budget:
        probes = ordered
    else:
        step = max(1, len(ordered) // budget)
        probes = [ordered[index] for index in range(0, len(ordered), step)][:budget]
    return probes, stable_hash(
        [
            {
                "chunk_business_key": audit_keys[str(chunk.id)],
                "chunk_index": chunk.chunk_index,
                "span": [chunk.char_start, chunk.char_end],
            }
            for chunk in probes
        ]
    )


def _candidate_typed_gate_decision(candidate: RelationEdgeCandidate) -> dict[str, Any]:
    features = dict(candidate.features_json or {})
    normalization = dict(features.get("normalization_stats") or {})
    required = {"type_threshold", "strong_threshold", "mutual", "reverse_accepted"}
    missing = sorted(required - set(normalization))
    if "cosine" not in features:
        missing.append("cosine")
    if "language_identity_scope_hash" not in features:
        missing.append("language_identity_scope_hash")
    if missing:
        raise ValueError(
            "TPE candidate is missing typed gate diagnostics required for hashing: "
            + ", ".join(missing)
        )
    cosine = round(float(features["cosine"]), 6)
    language_identity_scope_hash = str(features["language_identity_scope_hash"])
    if (
        len(language_identity_scope_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in language_identity_scope_hash.lower()
        )
    ):
        raise ValueError(
            "TPE candidate language identity scope hash must be a SHA-256 hex digest"
        )
    type_threshold = round(float(normalization["type_threshold"]), 6)
    strong_threshold = round(float(normalization["strong_threshold"]), 6)
    mutual = bool(normalization["mutual"])
    reverse_accepted = bool(normalization["reverse_accepted"])
    type_threshold_passed = cosine >= type_threshold
    strong_threshold_passed = cosine >= strong_threshold
    accepted_reasons = [
        reason
        for reason, passed in (
            ("mutual", mutual),
            ("reverse_quota", reverse_accepted),
            ("strong_threshold", strong_threshold_passed),
        )
        if passed
    ]
    accepted = type_threshold_passed and bool(accepted_reasons)
    if not accepted:
        raise ValueError(
            "TPE candidate adjacency contains an edge that did not pass its persisted typed gate"
        )
    return {
        "accepted": True,
        "cosine": cosine,
        "type_threshold": type_threshold,
        "type_threshold_passed": type_threshold_passed,
        "strong_threshold": strong_threshold,
        "strong_threshold_passed": strong_threshold_passed,
        "mutual": mutual,
        "reverse_accepted": reverse_accepted,
        "accepted_reasons": accepted_reasons,
        "language_identity_scope_hash": language_identity_scope_hash.lower(),
    }


def _candidate_adjacency_hash(
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate],
    *,
    theta_hash: str,
    language_identity_scope_hash: str,
) -> str:
    if not theta_hash:
        raise ValueError("theta_hash is required for TPE candidate adjacency hashing")
    normalized_language_scope_hash = str(language_identity_scope_hash).lower()
    if (
        len(normalized_language_scope_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in normalized_language_scope_hash
        )
    ):
        raise ValueError(
            "language_identity_scope_hash is required for TPE candidate adjacency hashing"
        )
    edges = []
    for candidate in sorted(
        candidates.values(),
        key=lambda item: (item.source_chunk_id, item.target_chunk_id, item.edge_type),
    ):
        edge_identity = {
            "source_chunk_id": candidate.source_chunk_id,
            "target_chunk_id": candidate.target_chunk_id,
            "edge_type": candidate.edge_type,
        }
        typed_gate_decision = _candidate_typed_gate_decision(candidate)
        if (
            typed_gate_decision["language_identity_scope_hash"]
            != normalized_language_scope_hash
        ):
            raise ValueError(
                "TPE candidate language identity scope hash does not match its trial scope"
            )
        edges.append(
            {
                "candidate_edge_id": stable_hash(edge_identity),
                **edge_identity,
                "raw_strength": round(float(candidate.raw_strength), 6),
                "typed_gate_decision": typed_gate_decision,
            }
        )
    return stable_hash(
        {
            "protocol": TPE_CANDIDATE_ADJACENCY_HASH_PROTOCOL,
            "theta_hash": theta_hash,
            "language_identity_scope_hash": normalized_language_scope_hash,
            "edges": edges,
        }
    )


def _adjacency(candidates: dict[tuple[str, str, str], RelationEdgeCandidate]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates.values():
        adjacency[candidate.source_chunk_id].add(candidate.target_chunk_id)
        adjacency[candidate.target_chunk_id].add(candidate.source_chunk_id)
    return adjacency


def _reachable_within_two(source_id: str, target_ids: set[str], adjacency: dict[str, set[str]]) -> bool:
    if not target_ids:
        return False
    seen = {source_id}
    queue: deque[tuple[str, int]] = deque([(source_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= 2:
            continue
        for neighbor_id in adjacency.get(node_id, set()):
            if neighbor_id in target_ids:
                return True
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
    return False


def _bounded_reachable_ids(
    source_id: str,
    adjacency: dict[str, set[str]],
    *,
    max_depth: int = 2,
) -> set[str]:
    """Return the deterministic bounded expansion used by TPE quality probes."""
    if max_depth < 1:
        return set()
    seen = {source_id}
    reached: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(source_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor_id in sorted(adjacency.get(node_id, set())):
            if neighbor_id != source_id:
                reached.add(neighbor_id)
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
    return reached


def _candidate_rq_prefix_inputs(
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    *,
    chunk_business_keys: dict[str, str] | None = None,
    canonical_business_keys_are_production: bool | None = None,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Build the trial-independent RQ input address buckets entirely in memory."""
    if chunk_business_keys is not None:
        missing = {str(chunk.id) for chunk in chunks}.difference(
            {str(value) for value in chunk_business_keys}
        )
        if missing:
            raise RuntimeError(
                "TPE candidate RQ input received an incomplete canonical chunk key map"
            )
    audit_key_by_id = {
        str(chunk.id): str(
            (chunk_business_keys or {}).get(str(chunk.id))
            or _fallback_tpe_chunk_audit_key(chunk)
        )
        for chunk in chunks
    }
    production_canonical_business_keys = (
        bool(canonical_business_keys_are_production)
        if canonical_business_keys_are_production is not None
        else bool(chunk_business_keys)
    )
    key_protocol = (
        CHUNK_BUSINESS_KEY_PROTOCOL_VERSION
        if production_canonical_business_keys
        else "tpe_local_chunk_audit_fallback_v1"
    )
    ordered_chunks = sorted(
        chunks,
        key=lambda item: (
            audit_key_by_id[str(item.id)],
            int(item.chunk_index),
            str(item.text_hash),
        ),
    )
    frozen_ids = {str(chunk.id) for chunk in ordered_chunks}
    vector_by_id: dict[str, Any] = {}
    duplicate_vector_keys: set[str] = set()
    for raw_chunk_id, vector in dict(vectors or {}).items():
        chunk_id = str(raw_chunk_id)
        if chunk_id in vector_by_id:
            duplicate_vector_keys.add(chunk_id)
            continue
        vector_by_id[chunk_id] = vector

    missing_ids = frozen_ids.difference(vector_by_id)
    extra_ids = set(vector_by_id).difference(frozen_ids)
    normalized_by_id: dict[str, list[float]] = {}
    invalid_by_reason: dict[str, set[str]] = defaultdict(set)
    vector_cards: list[dict[str, Any]] = []
    dimension_counts: Counter[int] = Counter()
    for chunk in ordered_chunks:
        chunk_id = str(chunk.id)
        business_key = audit_key_by_id[chunk_id]
        card: dict[str, Any] = {"chunk_business_key": business_key}
        if chunk_id in missing_ids:
            card.update({"status": "missing", "dimension": None})
            vector_cards.append(card)
            continue
        raw_vector = vector_by_id[chunk_id]
        if not isinstance(raw_vector, (list, tuple)):
            invalid_by_reason["invalid_candidate_vector_type"].add(business_key)
            card.update(
                {
                    "status": "invalid_type",
                    "dimension": None,
                    "value_type": type(raw_vector).__name__,
                }
            )
            vector_cards.append(card)
            continue
        if not raw_vector:
            invalid_by_reason["empty_candidate_vector"].add(business_key)
            card.update({"status": "empty", "dimension": 0})
            vector_cards.append(card)
            continue
        normalized_vector: list[float] = []
        invalid_numeric = False
        non_finite = False
        for value in raw_vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid_numeric = True
                break
            numeric = float(value)
            if not math.isfinite(numeric):
                non_finite = True
                break
            normalized_vector.append(numeric)
        if invalid_numeric:
            invalid_by_reason["non_numeric_candidate_vector"].add(business_key)
            card.update(
                {
                    "status": "non_numeric",
                    "dimension": len(raw_vector),
                }
            )
            vector_cards.append(card)
            continue
        if non_finite:
            invalid_by_reason["non_finite_candidate_vector"].add(business_key)
            card.update(
                {
                    "status": "non_finite",
                    "dimension": len(raw_vector),
                }
            )
            vector_cards.append(card)
            continue
        dimension = len(normalized_vector)
        dimension_counts[dimension] += 1
        normalized_by_id[chunk_id] = normalized_vector
        card.update(
            {
                "status": "usable",
                "dimension": dimension,
                "vector_hash": stable_hash(normalized_vector),
            }
        )
        vector_cards.append(card)

    extra_vector_cards: list[dict[str, Any]] = []
    for chunk_id in sorted(extra_ids):
        raw_vector = vector_by_id[chunk_id]
        if isinstance(raw_vector, (list, tuple)):
            numeric_values: list[float] = []
            numeric_and_finite = True
            for value in raw_vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    numeric_and_finite = False
                    break
                numeric = float(value)
                if not math.isfinite(numeric):
                    numeric_and_finite = False
                    break
                numeric_values.append(numeric)
            extra_vector_cards.append(
                {
                    "status": "extra",
                    "dimension": len(raw_vector),
                    "vector_hash": (
                        stable_hash(numeric_values) if numeric_and_finite else None
                    ),
                }
            )
        else:
            extra_vector_cards.append(
                {
                    "status": "extra_invalid_type",
                    "dimension": None,
                    "value_type": type(raw_vector).__name__,
                }
            )
    extra_vector_cards.sort(
        key=lambda item: (
            str(item.get("status") or ""),
            int(item.get("dimension") or -1),
            str(item.get("vector_hash") or ""),
            str(item.get("value_type") or ""),
        )
    )

    frozen_scope_hash = stable_hash(sorted(audit_key_by_id.values()))
    frozen_vector_input_hash = stable_hash(
        {
            "protocol_version": TPE_COMPONENT_COVERAGE_PROTOCOL_VERSION,
            "frozen_scope_hash": frozen_scope_hash,
            "vectors": vector_cards,
            "extra_vectors": extra_vector_cards,
            "duplicate_normalized_key_count": len(duplicate_vector_keys),
        }
    )
    reasons: list[str] = []
    if not ordered_chunks:
        reasons.append("empty_frozen_chunk_scope")
    if missing_ids:
        reasons.append("missing_candidate_vectors")
    if extra_ids:
        reasons.append("extra_candidate_vectors")
    if duplicate_vector_keys:
        reasons.append("duplicate_candidate_vector_keys")
    reasons.extend(sorted(invalid_by_reason))
    if len(dimension_counts) > 1:
        reasons.append("candidate_vector_dimension_drift")

    base_diagnostics = {
        "protocol_version": TPE_COMPONENT_COVERAGE_PROTOCOL_VERSION,
        "frozen_chunk_count": len(ordered_chunks),
        "provided_vector_key_count": len(vector_by_id),
        "usable_vector_count": len(normalized_by_id),
        "missing_vector_count": len(missing_ids),
        "extra_vector_count": len(extra_ids),
        "duplicate_vector_key_count": len(duplicate_vector_keys),
        "invalid_vector_count": sum(len(values) for values in invalid_by_reason.values()),
        "dimension_drift_vector_count": (
            sum(dimension_counts.values()) if len(dimension_counts) > 1 else 0
        ),
        "dimension_distribution": {
            str(dimension): count
            for dimension, count in sorted(dimension_counts.items())
        },
        "expected_dimension": (
            next(iter(dimension_counts)) if len(dimension_counts) == 1 else None
        ),
        "missing_chunk_business_key_sample": sorted(
            audit_key_by_id[chunk_id] for chunk_id in missing_ids
        )[:TPE_DIAGNOSTIC_SAMPLE_LIMIT],
        "extra_chunk_id_sample": sorted(extra_ids)[:TPE_DIAGNOSTIC_SAMPLE_LIMIT],
        "duplicate_chunk_id_sample": sorted(duplicate_vector_keys)[
            :TPE_DIAGNOSTIC_SAMPLE_LIMIT
        ],
        "invalid_chunk_business_key_samples": {
            reason: sorted(values)[:TPE_DIAGNOSTIC_SAMPLE_LIMIT]
            for reason, values in sorted(invalid_by_reason.items())
        },
        "frozen_scope_hash": frozen_scope_hash,
        "frozen_vector_input_hash": frozen_vector_input_hash,
        "chunk_business_key_protocol_version": key_protocol,
        "production_canonical_business_keys": production_canonical_business_keys,
        "model_call_count": 0,
    }
    if reasons:
        return {}, {
            **base_diagnostics,
            "enabled": False,
            "reason": "invalid_frozen_candidate_vector_scope",
            "reasons": reasons,
            "chunk_count": 0,
            "input_hash": frozen_vector_input_hash,
        }

    normalized = [
        (str(chunk.id), normalized_by_id[str(chunk.id)]) for chunk in ordered_chunks
    ]
    rq_config = rq_runtime_config()
    if int(rq_config["levels"]) != RQ_LEVELS:
        raise RuntimeError(
            "TPE candidate RQ input requires the fixed Four-Layer RQ depth"
        )
    rq_model = train_rq_kmeans(
        [vector for _chunk_id, vector in normalized],
        levels=RQ_LEVELS,
        max_k=int(rq_config["max_k"]),
        tau_r=float(rq_config["tau_r"]),
        tau_l=float(rq_config["tau_l"]),
        membership_top_m=int(rq_config["top_m"]),
        membership_probability_threshold=float(
            rq_config["probability_threshold"]
        ),
        membership_protocol=str(rq_config["membership_protocol"]),
    )
    assignments, batch_count = encode_rq_vectors_batch(normalized, rq_model)
    assignments = {str(chunk_id): encoded for chunk_id, encoded in assignments.items()}
    assignment_ids = set(assignments)
    assignment_scope_reasons: list[str] = []
    if assignment_ids != frozen_ids:
        assignment_scope_reasons.append("rq_encoding_scope_mismatch")
    prefixes_by_chunk: dict[str, set[str]] = {}
    canonical_assignments: list[dict[str, Any]] = []
    for chunk_id, encoded in sorted(assignments.items()):
        if chunk_id not in frozen_ids:
            continue
        prefixes = {
            "L{depth}:{path}".format(
                depth=int(item.get("depth") or 0),
                path=".".join(str(int(value)) for value in (item.get("prefix") or [])),
            )
            for item in list(encoded.get("prefix_memberships") or [])
            if item.get("prefix")
        }
        if not prefixes:
            assignment_scope_reasons.append("rq_prefix_assignment_missing")
        prefixes_by_chunk[chunk_id] = prefixes
        canonical_assignments.append(
            {
                "chunk_business_key": audit_key_by_id[chunk_id],
                "prefixes": sorted(prefixes),
                "membership_encoding_hash": str(
                    encoded.get("membership_encoding_hash") or ""
                ),
            }
        )
    canonical_assignments.sort(key=lambda item: item["chunk_business_key"])
    if assignment_scope_reasons:
        assignment_input_hash = stable_hash(canonical_assignments)
        return {}, {
            **base_diagnostics,
            "enabled": False,
            "reason": "invalid_candidate_rq_encoding_scope",
            "reasons": sorted(set(assignment_scope_reasons)),
            "chunk_count": 0,
            "assignment_count": len(assignments),
            "prefix_assignment_count": sum(
                1 for values in prefixes_by_chunk.values() if values
            ),
            "input_hash": assignment_input_hash,
            "rq_index_protocol": rq_model.get("index_protocol"),
            "rq_membership_protocol": rq_model.get("membership_protocol"),
            "rq_membership_protocol_hash": rq_model.get(
                "membership_protocol_hash"
            ),
            "rq_codebook_hash": rq_model.get("codebook_hash"),
            "encoding_batch_count": batch_count,
        }
    return prefixes_by_chunk, {
        **base_diagnostics,
        "enabled": True,
        "reasons": [],
        "chunk_count": len(assignments),
        "assignment_count": len(assignments),
        "prefix_count": len(
            {
                prefix
                for prefixes in prefixes_by_chunk.values()
                for prefix in prefixes
            }
        ),
        "rq_index_protocol": rq_model.get("index_protocol"),
        "rq_membership_protocol": rq_model.get("membership_protocol"),
        "rq_membership_protocol_hash": rq_model.get("membership_protocol_hash"),
        "rq_codebook_hash": rq_model.get("codebook_hash"),
        "encoding_batch_count": batch_count,
        "input_hash": stable_hash(canonical_assignments),
    }


def _valid_verified_support_span(
    verification: CitationVerification,
    chunk: Chunk,
    package: ContextPackage,
) -> bool:
    source_span = dict(verification.source_span_json or {})
    char_span = list(source_span.get("char_span") or [])
    if (
        str(source_span.get("chunk_id") or "") != str(chunk.id)
        or str(source_span.get("context_package_id") or "") != str(package.id)
        or len(char_span) != 2
    ):
        return False
    try:
        char_start, char_end = int(char_span[0]), int(char_span[1])
    except (TypeError, ValueError):
        return False
    if (
        char_start < int(chunk.char_start)
        or char_end > int(chunk.char_end)
        or char_end <= char_start
    ):
        return False
    return any(
        str(item.get("chunk_id") or "") == str(chunk.id)
        and list(item.get("char_span") or []) == [char_start, char_end]
        for item in list(package.citation_spans_json or [])
        if isinstance(item, dict)
    )


def _structure_positive_context(
    db: Session,
    chunks: list[Chunk],
    probes: list[Chunk],
    *,
    chunk_business_keys: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, set[str]]], set[str], dict[str, Any]]:
    chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
    chunk_ids = set(chunk_by_id)
    targets: dict[str, dict[str, set[str]]] = {
        str(probe.id): {
            "previous_next": set(),
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        }
        for probe in probes
    }

    by_version_section: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_version: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        version_key = str(chunk.document_version_id)
        by_version[version_key].append(chunk)
        section = str(chunk.section_path or "").strip()
        if section:
            by_version_section[(version_key, section)].add(str(chunk.id))

    for probe in probes:
        probe_id = str(probe.id)
        for neighbor_id in (
            str(probe.previous_chunk_id or ""),
            str(probe.next_chunk_id or ""),
        ):
            if neighbor_id in chunk_ids and neighbor_id != probe_id:
                targets[probe_id]["previous_next"].add(neighbor_id)
        section = str(probe.section_path or "").strip()
        if section:
            targets[probe_id]["same_section"].update(
                by_version_section.get((str(probe.document_version_id), section), set())
                - {probe_id}
            )
        if probe.page_start is not None and probe.page_end is not None:
            probe_start = int(probe.page_start)
            probe_end = int(probe.page_end)
            for candidate in by_version.get(str(probe.document_version_id), []):
                if (
                    str(candidate.id) == probe_id
                    or candidate.page_start is None
                    or candidate.page_end is None
                ):
                    continue
                if max(probe_start, int(candidate.page_start)) <= min(
                    probe_end, int(candidate.page_end)
                ):
                    targets[probe_id]["same_page"].add(str(candidate.id))

    special_types = {"table", "formula", "caption", "code_block"}
    allowed_closure_edge_types = (
        "prev_next",
        "table_formula_context",
        "same_page_region",
    )
    ordered_chunk_ids = sorted(chunk_ids)
    probe_ids = {
        str(probe.id) for probe in probes if str(probe.id) in chunk_ids
    }

    # Only probe mappings participate in structure-positive closure discovery.
    # Project scalar identities so a large active mapping scope never enters the
    # SQLAlchemy identity map as Mapping/Node ORM instances.
    probe_mapping_rows = (
        list(
            db.execute(
                select(
                    ChunkStructureMapping.chunk_id,
                    ChunkStructureMapping.structure_node_id,
                )
                .where(ChunkStructureMapping.chunk_id.in_(sorted(probe_ids)))
                .distinct()
            ).all()
        )
        if probe_ids
        else []
    )
    probe_ids_by_node: dict[str, set[str]] = defaultdict(set)
    for chunk_id_value, node_id_value in probe_mapping_rows:
        probe_ids_by_node[str(node_id_value)].add(str(chunk_id_value))
    probe_node_ids = set(probe_ids_by_node)

    probe_node_type_rows = (
        list(
            db.execute(
                select(ChunkStructureNode.id, ChunkStructureNode.node_type)
                .where(ChunkStructureNode.id.in_(sorted(probe_node_ids)))
                .distinct()
            ).all()
        )
        if probe_node_ids
        else []
    )
    probe_special_node_ids = {
        str(node_id)
        for node_id, node_type in probe_node_type_rows
        if str(node_type) in special_types
    }

    source_node = aliased(ChunkStructureNode)
    target_node = aliased(ChunkStructureNode)
    incident_edge_rows = (
        list(
            db.execute(
                select(
                    ChunkStructureEdge.id,
                    ChunkStructureEdge.source_node_id,
                    ChunkStructureEdge.target_node_id,
                )
                .join(
                    source_node,
                    source_node.id == ChunkStructureEdge.source_node_id,
                )
                .join(
                    target_node,
                    target_node.id == ChunkStructureEdge.target_node_id,
                )
                .where(
                    ChunkStructureEdge.edge_type.in_(allowed_closure_edge_types),
                    or_(
                        ChunkStructureEdge.source_node_id.in_(
                            sorted(probe_node_ids)
                        ),
                        ChunkStructureEdge.target_node_id.in_(
                            sorted(probe_node_ids)
                        ),
                    ),
                    or_(
                        source_node.node_type.in_(sorted(special_types)),
                        target_node.node_type.in_(sorted(special_types)),
                    ),
                )
                .distinct()
            ).all()
        )
        if probe_node_ids
        else []
    )

    relevant_node_ids_by_probe: dict[str, set[str]] = defaultdict(set)
    for node_id in probe_special_node_ids:
        for probe_id in probe_ids_by_node.get(node_id, set()):
            relevant_node_ids_by_probe[probe_id].add(node_id)
    for _edge_id, source_id_value, target_id_value in incident_edge_rows:
        source_id = str(source_id_value)
        target_id = str(target_id_value)
        for probe_id in probe_ids_by_node.get(source_id, set()):
            relevant_node_ids_by_probe[probe_id].add(target_id)
        for probe_id in probe_ids_by_node.get(target_id, set()):
            relevant_node_ids_by_probe[probe_id].add(source_id)
    relevant_node_ids = {
        node_id
        for node_ids in relevant_node_ids_by_probe.values()
        for node_id in node_ids
    }
    relevant_node_mapping_rows = (
        list(
            db.execute(
                select(
                    ChunkStructureMapping.structure_node_id,
                    ChunkStructureMapping.chunk_id,
                )
                .where(
                    ChunkStructureMapping.structure_node_id.in_(
                        sorted(relevant_node_ids)
                    ),
                    ChunkStructureMapping.chunk_id.in_(ordered_chunk_ids),
                )
                .distinct()
            ).all()
        )
        if relevant_node_ids and chunk_ids
        else []
    )
    chunk_ids_by_relevant_node: dict[str, set[str]] = defaultdict(set)
    for node_id_value, chunk_id_value in relevant_node_mapping_rows:
        chunk_ids_by_relevant_node[str(node_id_value)].add(str(chunk_id_value))
    for probe_id, relevant_nodes in relevant_node_ids_by_probe.items():
        for node_id in relevant_nodes:
            targets[probe_id]["special_object_closure"].update(
                chunk_ids_by_relevant_node.get(node_id, set()) - {probe_id}
            )

    mapped_chunk_count = 0
    special_node_count = 0
    special_closure_edge_count = 0
    if chunk_ids:
        mapped_chunk_count = int(
            db.scalar(
                select(
                    func.count(func.distinct(ChunkStructureMapping.chunk_id))
                ).where(ChunkStructureMapping.chunk_id.in_(ordered_chunk_ids))
            )
            or 0
        )
        special_node_count = int(
            db.scalar(
                select(
                    func.count(
                        func.distinct(ChunkStructureMapping.structure_node_id)
                    )
                )
                .select_from(ChunkStructureMapping)
                .join(
                    ChunkStructureNode,
                    ChunkStructureNode.id
                    == ChunkStructureMapping.structure_node_id,
                )
                .where(
                    ChunkStructureMapping.chunk_id.in_(ordered_chunk_ids),
                    ChunkStructureNode.node_type.in_(sorted(special_types)),
                )
            )
            or 0
        )
        source_mapping = aliased(ChunkStructureMapping)
        target_mapping = aliased(ChunkStructureMapping)
        source_is_mapped = exists(
            select(1).where(
                source_mapping.structure_node_id
                == ChunkStructureEdge.source_node_id,
                source_mapping.chunk_id.in_(ordered_chunk_ids),
            )
        )
        target_is_mapped = exists(
            select(1).where(
                target_mapping.structure_node_id
                == ChunkStructureEdge.target_node_id,
                target_mapping.chunk_id.in_(ordered_chunk_ids),
            )
        )
        special_closure_edge_count = int(
            db.scalar(
                select(func.count(ChunkStructureEdge.id)).where(
                    ChunkStructureEdge.edge_type.in_(
                        allowed_closure_edge_types
                    ),
                    source_is_mapped,
                    target_is_mapped,
                )
            )
            or 0
        )

    audit_key_by_id = {
        str(chunk.id): str(
            (chunk_business_keys or {}).get(str(chunk.id))
            or _fallback_tpe_chunk_audit_key(chunk)
        )
        for chunk in chunks
    }
    truncated_by_category: Counter[str] = Counter()
    for category_card in targets.values():
        for category, target_ids in category_card.items():
            ordered_target_ids = sorted(
                target_ids,
                key=lambda target_id: audit_key_by_id[target_id],
            )
            truncated_by_category[category] += max(
                0,
                len(ordered_target_ids)
                - TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT,
            )
            category_card[category] = set(
                ordered_target_ids[:TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT]
            )

    raw_span_traceability = (
        select(ChunkSpan.chunk_id)
        .join(Chunk, Chunk.id == ChunkSpan.chunk_id)
        .where(
            ChunkSpan.chunk_id.in_(ordered_chunk_ids),
            ChunkSpan.document_version_id == Chunk.document_version_id,
            ChunkSpan.char_start <= Chunk.char_start,
            ChunkSpan.char_end >= Chunk.char_end,
            ChunkSpan.char_end > ChunkSpan.char_start,
        )
        .distinct()
    )
    raw_span_traceable_ids = (
        {str(chunk_id) for chunk_id in db.scalars(raw_span_traceability).all()}
        if chunk_ids
        else set()
    )
    mapped_traceability = exists(
        select(1).where(
            ChunkStructureMapping.chunk_id == ChunkSpan.chunk_id,
        )
    )
    structure_traceable_ids = (
        {
            str(chunk_id)
            for chunk_id in db.scalars(
                raw_span_traceability.where(mapped_traceability)
            ).all()
        }
        if chunk_ids
        else set()
    )
    category_counts = {
        category: sum(len(card[category]) for card in targets.values())
        for category in (
            "previous_next",
            "same_section",
            "same_page",
            "special_object_closure",
        )
    }
    return targets, structure_traceable_ids, {
        "protocol_version": TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
        "category_positive_counts": category_counts,
        "positive_count": sum(category_counts.values()),
        "mapped_chunk_count": mapped_chunk_count,
        "raw_span_traceable_chunk_count": len(raw_span_traceable_ids),
        "structure_traceable_chunk_count": len(structure_traceable_ids),
        "special_node_count": special_node_count,
        "special_closure_edge_count": special_closure_edge_count,
        "bounded_structure_query_rows": {
            "probe_mapping_rows": len(probe_mapping_rows),
            "probe_node_type_rows": len(probe_node_type_rows),
            "incident_special_edge_rows": len(incident_edge_rows),
            "relevant_node_mapping_rows": len(relevant_node_mapping_rows),
            "raw_span_traceable_rows": len(raw_span_traceable_ids),
            "structure_traceable_rows": len(structure_traceable_ids),
        },
        "positive_limit_per_probe_category": (
            TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT
        ),
        "truncated_positive_counts": dict(sorted(truncated_by_category.items())),
        "model_call_count": 0,
    }


def _expected_support_context(
    db: Session,
    chunks: list[Chunk],
    probes: list[Chunk],
    structure_targets: dict[str, dict[str, set[str]]],
    *,
    chunk_business_keys: dict[str, str] | None = None,
    production_canonical_business_keys: bool | None = None,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
    chunk_ids = set(chunk_by_id)
    resolved_production_canonical_business_keys = (
        bool(production_canonical_business_keys)
        if production_canonical_business_keys is not None
        else chunk_business_keys is not None
    )
    if chunk_business_keys is not None and chunk_ids.difference(
        {str(value) for value in chunk_business_keys}
    ):
        raise RuntimeError(
            "TPE expected-support input received an incomplete canonical chunk key map"
        )
    if chunk_business_keys is None and all(isinstance(chunk, Chunk) for chunk in chunks):
        chunk_business_keys = dict(chunk_business_references(db, chunks).key_by_id)
        resolved_production_canonical_business_keys = True
    else:
        chunk_business_keys = {
            str(chunk.id): str(
                (chunk_business_keys or {}).get(str(chunk.id))
                or _fallback_tpe_chunk_audit_key(chunk)
            )
            for chunk in chunks
        }
    expected: dict[str, set[str]] = {str(probe.id): set() for probe in probes}
    source_types: dict[str, dict[str, set[str]]] = {
        str(probe.id): defaultdict(set) for probe in probes
    }
    invalid_manual_ids: set[str] = set()
    invalid_manual_value_count = 0
    truncated_manual_target_count = 0
    manual_canonical_candidate_count = 0
    manual_canonical_duplicate_count = 0
    manual_selected_target_count = 0
    manual_selection_cards: list[dict[str, str]] = []

    for probe in probes:
        probe_id = str(probe.id)
        metadata = dict(probe.metadata_json or {})
        raw_manual_values = (
            metadata.get("tpe_expected_support_chunk_ids")
            or metadata.get("expected_support_chunk_ids")
            or []
        )
        if not isinstance(raw_manual_values, (list, tuple, set)):
            invalid_manual_value_count += 1
            raw_manual_values = []
        valid_manual_values = [
            value
            for value in raw_manual_values
            if isinstance(value, str) and value
        ]
        invalid_manual_value_count += len(raw_manual_values) - len(
            valid_manual_values
        )
        targets_by_business_key: dict[str, str] = {}
        for target_id in set(valid_manual_values):
            if target_id not in chunk_ids or target_id == probe_id:
                invalid_manual_ids.add(target_id)
                continue
            target_business_key = str(chunk_business_keys.get(target_id) or "")
            if not target_business_key:
                invalid_manual_ids.add(target_id)
                continue
            previous_target_id = targets_by_business_key.get(target_business_key)
            if previous_target_id is not None:
                manual_canonical_duplicate_count += 1
                targets_by_business_key[target_business_key] = min(
                    previous_target_id,
                    target_id,
                )
            else:
                targets_by_business_key[target_business_key] = target_id
        ordered_manual_targets = sorted(targets_by_business_key.items())
        manual_canonical_candidate_count += len(ordered_manual_targets)
        truncated_manual_target_count += max(
            0,
            len(ordered_manual_targets) - TPE_EXPECTED_SUPPORT_PER_PROBE_LIMIT,
        )
        for target_business_key, target_id in ordered_manual_targets[
            :TPE_EXPECTED_SUPPORT_PER_PROBE_LIMIT
        ]:
            expected[probe_id].add(target_id)
            source_types[probe_id][target_id].add("manual_probe")
            manual_selected_target_count += 1
            manual_selection_cards.append(
                {
                    "probe_chunk_business_key": chunk_business_keys[probe_id],
                    "expected_support_chunk_business_key": target_business_key,
                }
            )
        for category, target_ids in structure_targets.get(probe_id, {}).items():
            for target_id in target_ids:
                expected[probe_id].add(target_id)
                source_types[probe_id][target_id].add(f"structure:{category}")

    knowledge_base_ids = {str(chunk.knowledge_base_id) for chunk in chunks}
    if len(knowledge_base_ids) > 1:
        raise RuntimeError("TPE evaluation chunks must belong to one knowledge base")
    verification_rows: list[CitationVerification] = []
    verifications: list[CitationVerification] = []
    if chunk_ids and knowledge_base_ids:
        verification_rows = list(
            db.scalars(
                select(CitationVerification).where(
                    CitationVerification.knowledge_base_id
                    == next(iter(knowledge_base_ids)),
                    CitationVerification.verdict == "supported",
                    CitationVerification.chunk_id.in_(sorted(chunk_ids)),
                    CitationVerification.context_package_id.is_not(None),
                ).order_by(
                    CitationVerification.created_at.desc(),
                    CitationVerification.id.desc(),
                ).limit(TPE_VERIFIED_CITATION_ROW_LIMIT + 1)
            ).all()
        )
        verifications = verification_rows[:TPE_VERIFIED_CITATION_ROW_LIMIT]
        package_ids = {
            str(row.context_package_id)
            for row in verifications
            if row.context_package_id
        }
        packages = list(
            db.scalars(
                select(ContextPackage).where(ContextPackage.id.in_(sorted(package_ids)))
            ).all()
        ) if package_ids else []
        package_by_id = {str(row.id): row for row in packages}
        verified_targets_by_package: dict[str, set[str]] = defaultdict(set)
        for verification in verifications:
            package = package_by_id.get(str(verification.context_package_id))
            chunk = chunk_by_id.get(str(verification.chunk_id))
            if (
                package is not None
                and chunk is not None
                and str(package.knowledge_base_id) in knowledge_base_ids
                and _valid_verified_support_span(verification, chunk, package)
            ):
                verified_targets_by_package[str(package.id)].add(str(chunk.id))
        for package_id, target_ids in verified_targets_by_package.items():
            package = package_by_id[package_id]
            package_scope_ids = {
                str(value)
                for value in (
                    list(package.hit_chunk_ids_json or [])
                    + list(package.restored_chunk_ids_json or [])
                    + list(package.bridge_chunk_ids_json or [])
                    + [
                        item.get("chunk_id")
                        for item in list(package.citation_spans_json or [])
                        if isinstance(item, dict)
                    ]
                )
                if value is not None and str(value) in chunk_ids
            }
            for probe_id in expected:
                if probe_id not in package_scope_ids:
                    continue
                for target_id in target_ids - {probe_id}:
                    expected[probe_id].add(target_id)
                    source_types[probe_id][target_id].add(
                        "verified_citation_span"
                    )

    canonical_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for probe_id in expected:
        for target_id in expected[probe_id]:
            canonical_sources[
                (
                    chunk_business_keys[probe_id],
                    chunk_business_keys[target_id],
                )
            ].update(source_types[probe_id][target_id])
    canonical_pairs = [
        {
            "probe_chunk_business_key": probe_business_key,
            "expected_support_chunk_business_key": target_business_key,
            "sources": sorted(sources),
        }
        for (probe_business_key, target_business_key), sources in sorted(
            canonical_sources.items()
        )
    ]
    manual_selection_cards.sort(
        key=lambda item: (
            item["probe_chunk_business_key"],
            item["expected_support_chunk_business_key"],
        )
    )
    source_counts = Counter(
        source
        for probe_card in source_types.values()
        for target_sources in probe_card.values()
        for source in target_sources
    )
    return expected, {
        "protocol_version": TPE_EXPECTED_SUPPORT_PROTOCOL_VERSION,
        "pair_count": len(canonical_pairs),
        "probe_with_positive_count": sum(1 for values in expected.values() if values),
        "source_counts": dict(sorted(source_counts.items())),
        "verified_citation_pair_count": sum(
            1
            for probe_card in source_types.values()
            for target_sources in probe_card.values()
            if "verified_citation_span" in target_sources
        ),
        "invalid_manual_target_ids": sorted(invalid_manual_ids)[
            :TPE_DIAGNOSTIC_SAMPLE_LIMIT
        ],
        "invalid_manual_target_id_count": len(invalid_manual_ids),
        "invalid_manual_target_ids_hash": stable_hash(sorted(invalid_manual_ids)),
        "invalid_manual_target_ids_truncated": max(
            0,
            len(invalid_manual_ids) - TPE_DIAGNOSTIC_SAMPLE_LIMIT,
        ),
        "invalid_manual_value_count": invalid_manual_value_count,
        "manual_canonical_candidate_count": manual_canonical_candidate_count,
        "manual_canonical_duplicate_count": manual_canonical_duplicate_count,
        "manual_selected_target_count": manual_selected_target_count,
        "manual_selection_hash": stable_hash(manual_selection_cards),
        "manual_target_limit_per_probe": TPE_EXPECTED_SUPPORT_PER_PROBE_LIMIT,
        "truncated_manual_target_count": truncated_manual_target_count,
        "verified_citation_row_limit": TPE_VERIFIED_CITATION_ROW_LIMIT,
        "verified_citation_rows_considered": len(verifications),
        "verified_citation_rows_truncated": max(
            0, len(verification_rows) - len(verifications)
        ) if chunk_ids and knowledge_base_ids else 0,
        "expected_support_pairs_hash": stable_hash(canonical_pairs),
        "chunk_business_key_protocol_version": (
            CHUNK_BUSINESS_KEY_PROTOCOL_VERSION
            if resolved_production_canonical_business_keys
            else "tpe_local_chunk_audit_fallback_v1"
        ),
        "production_canonical_business_keys": (
            resolved_production_canonical_business_keys
        ),
        "model_call_count": 0,
    }


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _float_matches(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    left_value = _finite_number(left)
    right_value = _finite_number(right)
    return (
        left_value is not None
        and right_value is not None
        and abs(left_value - right_value) <= tolerance
    )


def _sha256_hex(value: Any) -> bool:
    normalized = str(value or "").lower()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _weighted_signal_card_replays(
    card: dict[str, Any],
    *,
    protocol_version: str,
    protocol_hash: str,
    component_weights: dict[str, float],
) -> bool:
    required_fields = {
        "value",
        "protocol_version",
        "protocol_hash",
        "configured_weights",
        "available_weight",
        "missing_component_protocol",
        "unavailable_components",
        "components",
        "query_relevance",
        "card_hash",
    }
    if not required_fields.issubset(card):
        return False
    if (
        card.get("protocol_version") != protocol_version
        or card.get("protocol_hash") != protocol_hash
        or card.get("missing_component_protocol")
        != "availability_aware_weight_renormalization"
        or card.get("query_relevance") is not False
    ):
        return False
    configured_weights = _mapping_or_empty(card.get("configured_weights"))
    components = _mapping_or_empty(card.get("components"))
    if set(configured_weights) != set(component_weights) or set(components) != set(
        component_weights
    ):
        return False
    for name, expected_weight in component_weights.items():
        if not _float_matches(configured_weights.get(name), expected_weight):
            return False

    available_names: list[str] = []
    weighted_total = 0.0
    available_weight = 0.0
    for name, expected_weight in component_weights.items():
        component = components.get(name)
        if not isinstance(component, dict):
            return False
        if not {
            "value",
            "available",
            "reason",
            "diagnostics",
            "configured_weight",
            "effective_weight",
        }.issubset(component):
            return False
        if type(component.get("available")) is not bool:
            return False
        if not isinstance(component.get("reason"), str) or not component.get(
            "reason"
        ):
            return False
        if not isinstance(component.get("diagnostics"), dict):
            return False
        if not _float_matches(component.get("configured_weight"), expected_weight):
            return False
        if component["available"]:
            value = _finite_number(component.get("value"))
            if value is None or not 0.0 <= value <= 1.0:
                return False
            available_names.append(name)
            weighted_total += float(expected_weight) * value
            available_weight += float(expected_weight)
        elif component.get("value") is not None:
            return False

    if not _float_matches(card.get("available_weight"), round(available_weight, 6)):
        return False
    for name, expected_weight in component_weights.items():
        component = components[name]
        expected_effective_weight = (
            round(float(expected_weight) / available_weight, 6)
            if component["available"] and available_weight > 0.0
            else 0.0
        )
        if not _float_matches(
            component.get("effective_weight"), expected_effective_weight
        ):
            return False
    expected_value = (
        round(weighted_total / available_weight, 6)
        if available_weight > 0.0
        else 0.0
    )
    if not _float_matches(card.get("value"), expected_value):
        return False
    expected_unavailable = sorted(set(component_weights).difference(available_names))
    if list(card.get("unavailable_components") or []) != expected_unavailable:
        return False
    hash_payload = dict(card)
    claimed_hash = hash_payload.pop("card_hash", None)
    return _sha256_hex(claimed_hash) and claimed_hash == stable_hash(hash_payload)


def _quota_card_replays(
    card: dict[str, Any],
    *,
    signal: float,
    signal_role: str,
) -> bool:
    if set(card) != {
        "value",
        "lower",
        "upper",
        "signal_role",
        "signal",
        "quota_signal_scale",
        "scaled_signal",
        "log2_increment",
        "protocol_version",
        "protocol_hash",
        "query_relevance",
    }:
        return False
    try:
        expected = relation_quota_card(
            lower=int(card["lower"]),
            upper=int(card["upper"]),
            signal=signal,
            signal_role=signal_role,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return card == expected


def _candidate_support_feature_audit(
    candidate: RelationEdgeCandidate,
) -> dict[str, Any]:
    features = dict(candidate.features_json or {})
    raw_card = _mapping_or_empty(features.get("raw_strength_components"))
    source_signal = _mapping_or_empty(features.get("source_out_signal_card"))
    target_signal = _mapping_or_empty(
        features.get("target_in_acceptance_signal_card")
    )
    source_node_quality = _mapping_or_empty(
        features.get("source_node_quality_card")
    )
    target_node_quality = _mapping_or_empty(
        features.get("target_node_quality_card")
    )
    try:
        typed_gate = _candidate_typed_gate_decision(candidate)
        typed_gate_passed = bool(typed_gate["accepted"])
    except (TypeError, ValueError, OverflowError):
        typed_gate = {}
        typed_gate_passed = False

    candidate_channels = list(features.get("candidate_channels") or [])
    rank_components = list(features.get("rank_components") or [])
    rank_support = bool(candidate_channels) and len(rank_components) == len(
        set(candidate_channels)
    )
    replayed_rank_scores: list[float] = []
    rank_channels: set[str] = set()
    if rank_support:
        for card in rank_components:
            if not isinstance(card, dict) or set(card) != {
                "candidate_channel",
                "rank",
                "ordinal",
                "candidate_count",
                "selected_limit",
                "selected_count",
                "rank_score",
            }:
                rank_support = False
                break
            channel = card.get("candidate_channel")
            integer_fields = (
                "rank",
                "ordinal",
                "candidate_count",
                "selected_limit",
                "selected_count",
            )
            if (
                not isinstance(channel, str)
                or not channel
                or channel in rank_channels
                or any(type(card.get(field)) is not int for field in integer_fields)
            ):
                rank_support = False
                break
            rank = int(card["rank"])
            ordinal = int(card["ordinal"])
            candidate_count = int(card["candidate_count"])
            selected_limit = int(card["selected_limit"])
            selected_count = int(card["selected_count"])
            if (
                candidate_count < 1
                or selected_limit < 0
                or selected_count
                != min(max(0, selected_limit), candidate_count)
                or not 1 <= rank <= ordinal <= selected_count <= candidate_count
            ):
                rank_support = False
                break
            try:
                replayed_rank_score = channel_rank_score(
                    rank=rank,
                    candidate_count=candidate_count,
                )
            except ValueError:
                rank_support = False
                break
            if not _float_matches(card.get("rank_score"), replayed_rank_score):
                rank_support = False
                break
            rank_channels.add(channel)
            replayed_rank_scores.append(replayed_rank_score)
    replayed_rank_score = max(replayed_rank_scores or [0.0])
    rank_support = bool(
        rank_support
        and rank_channels == set(candidate_channels)
        and features.get("rank_score_protocol_version")
        == RANK_SCORE_PROTOCOL_VERSION
        and features.get("rank_score_protocol_hash")
        == relation_rank_score_protocol_hash()
        and _float_matches(features.get("rank_score"), replayed_rank_score)
    )

    source_node_quality_ok = _weighted_signal_card_replays(
        source_node_quality,
        protocol_version=CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
        protocol_hash=chunk_node_quality_protocol_hash(),
        component_weights=CHUNK_NODE_QUALITY_COMPONENT_WEIGHTS,
    )
    target_node_quality_ok = _weighted_signal_card_replays(
        target_node_quality,
        protocol_version=CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
        protocol_hash=chunk_node_quality_protocol_hash(),
        component_weights=CHUNK_NODE_QUALITY_COMPONENT_WEIGHTS,
    )
    source_node_quality_value = _finite_number(source_node_quality.get("value"))
    target_node_quality_value = _finite_number(target_node_quality.get("value"))
    replayed_node_quality_pair = (
        round((source_node_quality_value + target_node_quality_value) / 2.0, 6)
        if source_node_quality_value is not None
        and target_node_quality_value is not None
        else None
    )
    node_quality_support = bool(
        source_node_quality_ok
        and target_node_quality_ok
        and features.get("chunk_node_quality_protocol_version")
        == CHUNK_NODE_QUALITY_PROTOCOL_VERSION
        and features.get("chunk_node_quality_protocol_hash")
        == chunk_node_quality_protocol_hash()
        and features.get("node_quality_pair_protocol_version")
        == CHUNK_NODE_QUALITY_PROTOCOL_VERSION
        and features.get("node_quality_pair_protocol_hash")
        == chunk_node_quality_protocol_hash()
        and features.get("node_quality_pair_role")
        == "mean_intrinsic_node_quality_graph_build_only_not_query_relevance"
        and features.get("node_weight_used_as_query_relevance") is False
        and replayed_node_quality_pair is not None
        and _float_matches(
            features.get("node_quality_pair"), replayed_node_quality_pair
        )
    )

    source_signal_ok = _weighted_signal_card_replays(
        source_signal,
        protocol_version=OUT_EVIDENCE_MASS_PROTOCOL_VERSION,
        protocol_hash=out_evidence_mass_protocol_hash(),
        component_weights=OUT_EVIDENCE_MASS_COMPONENT_WEIGHTS,
    )
    target_signal_ok = _weighted_signal_card_replays(
        target_signal,
        protocol_version=IN_ACCEPTANCE_CAPACITY_PROTOCOL_VERSION,
        protocol_hash=in_acceptance_capacity_protocol_hash(),
        component_weights=IN_ACCEPTANCE_CAPACITY_COMPONENT_WEIGHTS,
    )
    source_signal_value = _finite_number(source_signal.get("value"))
    target_signal_value = _finite_number(target_signal.get("value"))
    source_node_component = _mapping_or_empty(
        _mapping_or_empty(source_signal.get("components")).get("node_quality")
    )
    target_node_component = _mapping_or_empty(
        _mapping_or_empty(target_signal.get("components")).get("node_quality")
    )
    source_quota_cards = _mapping_or_empty(features.get("source_out_quota_cards"))
    target_quota_card = _mapping_or_empty(
        features.get("target_inbound_quota_card")
    )
    quota_cards_replay = bool(
        source_signal_value is not None
        and target_signal_value is not None
        and set(source_quota_cards) == set(candidate_channels)
        and all(
            _quota_card_replays(
                _mapping_or_empty(source_quota_cards[channel]),
                signal=source_signal_value,
                signal_role="out_evidence_mass",
            )
            for channel in candidate_channels
        )
        and _quota_card_replays(
            target_quota_card,
            signal=target_signal_value,
            signal_role="in_acceptance_capacity",
        )
    )
    quota_support = bool(
        source_signal_ok
        and target_signal_ok
        and source_signal_value is not None
        and target_signal_value is not None
        and _float_matches(
            features.get("source_out_evidence_mass"), source_signal_value
        )
        and _float_matches(
            features.get("target_in_acceptance_capacity"), target_signal_value
        )
        and _float_matches(
            source_node_component.get("value"), source_node_quality_value
        )
        and _mapping_or_empty(source_node_component.get("diagnostics")).get(
            "card_hash"
        )
        == source_node_quality.get("card_hash")
        and _float_matches(
            target_node_component.get("value"), target_node_quality_value
        )
        and _mapping_or_empty(target_node_component.get("diagnostics")).get(
            "card_hash"
        )
        == target_node_quality.get("card_hash")
        and features.get("out_evidence_mass_protocol_version")
        == OUT_EVIDENCE_MASS_PROTOCOL_VERSION
        and features.get("out_evidence_mass_protocol_hash")
        == out_evidence_mass_protocol_hash()
        and features.get("in_acceptance_capacity_protocol_version")
        == IN_ACCEPTANCE_CAPACITY_PROTOCOL_VERSION
        and features.get("in_acceptance_capacity_protocol_hash")
        == in_acceptance_capacity_protocol_hash()
        and features.get("relation_quota_protocol_version")
        == RELATION_QUOTA_PROTOCOL_VERSION
        and features.get("relation_quota_protocol_hash")
        == relation_quota_protocol_hash()
        and _sha256_hex(features.get("quota_signal_scope_hash"))
        and quota_cards_replay
    )

    required_raw_fields = {
        "protocol_version",
        "protocol_hash",
        "coefficients",
        "semantic",
        "reciprocity",
        "rank_score",
        "node_quality_pair",
        "node_quality_pair_protocol_version",
        "node_quality_pair_protocol_hash",
        "computed_raw_strength",
    }
    expected_coefficients = {
        "semantic": 0.75,
        "reciprocity": 0.15,
        "rank_score": 0.07,
        "node_quality_pair": 0.03,
    }
    raw_strength_protocol = bool(
        set(raw_card) == required_raw_fields
        and raw_card.get("protocol_version")
        == RELATION_RAW_STRENGTH_PROTOCOL_VERSION
        and raw_card.get("protocol_hash") == relation_raw_strength_protocol_hash()
        and raw_card.get("coefficients") == expected_coefficients
        and raw_card.get("node_quality_pair_protocol_version")
        == CHUNK_NODE_QUALITY_PROTOCOL_VERSION
        and raw_card.get("node_quality_pair_protocol_hash")
        == chunk_node_quality_protocol_hash()
    )
    semantic = _finite_number(raw_card.get("semantic"))
    reciprocity = _finite_number(raw_card.get("reciprocity"))
    raw_rank_score = _finite_number(raw_card.get("rank_score"))
    raw_node_quality_pair = _finite_number(raw_card.get("node_quality_pair"))
    expected_semantic = None
    expected_reciprocity = None
    semantic_replay_tolerance = 1e-6
    if typed_gate_passed:
        type_threshold = _finite_number(typed_gate.get("type_threshold"))
        strong_threshold = _finite_number(typed_gate.get("strong_threshold"))
        cosine = _finite_number(typed_gate.get("cosine"))
        if (
            type_threshold is not None
            and strong_threshold is not None
            and cosine is not None
            and strong_threshold > type_threshold
        ):
            threshold_span = max(strong_threshold - type_threshold, 1e-6)
            expected_semantic = round(
                max(
                    0.0,
                    min(
                        1.0,
                        (cosine - type_threshold)
                        / threshold_span,
                    ),
                ),
                6,
            )
            # The relation builder computes semantic strength from the
            # provider cosine, then persists both semantic and cosine at six
            # decimal places. Replay therefore has two bounded quantization
            # errors: half a cosine unit propagated through the threshold
            # span, plus half a semantic unit. This is not a free tolerance;
            # it is the exact maximum error implied by the persisted wire
            # precision.
            semantic_replay_tolerance = (
                (0.5e-6 / threshold_span) + 1.0e-6 + 1e-12
            )
        expected_reciprocity = (
            1.0
            if typed_gate.get("mutual")
            else 0.35
            if typed_gate.get("reverse_accepted")
            else 0.0
        )
    raw_strength_inputs = bool(
        semantic is not None
        and 0.0 <= semantic <= 1.0
        and reciprocity is not None
        and reciprocity in {0.0, 0.35, 1.0}
        and raw_rank_score is not None
        and 0.0 <= raw_rank_score <= 1.0
        and raw_node_quality_pair is not None
        and 0.0 <= raw_node_quality_pair <= 1.0
        and expected_semantic is not None
        and expected_reciprocity is not None
        and _float_matches(
            semantic,
            expected_semantic,
            tolerance=semantic_replay_tolerance,
        )
        and _float_matches(reciprocity, expected_reciprocity)
        and rank_support
        and _float_matches(raw_rank_score, replayed_rank_score)
        and node_quality_support
        and _float_matches(raw_node_quality_pair, replayed_node_quality_pair)
        and _float_matches(features.get("reciprocity"), expected_reciprocity)
    )
    replayed_raw_strength = None
    if raw_strength_protocol and raw_strength_inputs:
        replayed_raw_strength = relation_raw_strength(
            semantic=semantic,
            reciprocity=reciprocity,
            rank_score=raw_rank_score,
            node_quality_pair=raw_node_quality_pair,
        )
    raw_strength_replay = bool(
        replayed_raw_strength is not None
        and _float_matches(
            raw_card.get("computed_raw_strength"), replayed_raw_strength
        )
        and _float_matches(candidate.raw_strength, replayed_raw_strength)
    )
    checks = {
        "typed_gate": typed_gate_passed,
        "raw_strength_protocol": raw_strength_protocol,
        "raw_strength_inputs": raw_strength_inputs,
        "raw_strength_replay": raw_strength_replay,
        "rank_support": rank_support,
        "node_quality_support": node_quality_support,
        "quota_support": quota_support,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "replay": {
            "semantic": expected_semantic,
            "reciprocity": expected_reciprocity,
            "rank_score": replayed_rank_score if rank_support else None,
            "node_quality_pair": (
                replayed_node_quality_pair if node_quality_support else None
            ),
            "raw_strength": replayed_raw_strength,
        },
    }


def _entropy_ratio(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = float(len(values))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return max(0.0, min(1.0, entropy / math.log(len(counts))))


def _component(
    name: str,
    value: float,
    numerator: float,
    denominator: float,
    probe_hash: str | None,
    *,
    protocol: str = AUTO_TPE_PROTOCOL,
) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "numerator": round(float(numerator), 6),
        "denominator": round(float(denominator), 6),
        "sample_count": round(float(denominator), 6),
        "probe_set_hash": probe_hash,
        "protocol": protocol,
        "component": name,
    }


def _degree_stats(
    chunks: list[Chunk],
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate],
) -> dict[str, Any]:
    degree_by_id = {str(chunk.id): 0 for chunk in chunks}
    edge_type_counts: Counter[str] = Counter()
    for candidate in candidates.values():
        source_id = str(candidate.source_chunk_id)
        target_id = str(candidate.target_chunk_id)
        degree_by_id[source_id] = degree_by_id.get(source_id, 0) + 1
        degree_by_id[target_id] = degree_by_id.get(target_id, 0) + 1
        edge_type_counts[str(candidate.edge_type)] += 1
    degrees = sorted(degree_by_id.values())
    if not degrees:
        return {
            "isolated": 0.0,
            "isolated_ratio": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "hubness_ratio": 0.0,
            "degree_count": 0,
            "total_degree": 0.0,
            "top_hub_count": 0,
            "top_hub_share": 0.0,
            "uniform_top_hub_share": 0.0,
            "top_hub_concentration": 0.0,
            "edge_type_counts": {},
            "distribution_hash": stable_hash(
                {"degrees": [], "edge_type_counts": {}}
            ),
        }
    p95_index = min(len(degrees) - 1, int(math.ceil(len(degrees) * 0.95)) - 1)
    midpoint = len(degrees) // 2
    median = (
        float(degrees[midpoint])
        if len(degrees) % 2
        else (float(degrees[midpoint - 1]) + float(degrees[midpoint])) / 2.0
    )
    p95 = degrees[p95_index]
    isolated = sum(1 for value in degrees if value <= 0)
    total_degree = float(sum(degrees))
    top_hub_count = max(1, int(math.ceil(len(degrees) * TPE_TOP_HUB_FRACTION)))
    top_hub_share = (
        sum(degrees[-top_hub_count:]) / total_degree if total_degree > 0.0 else 0.0
    )
    uniform_top_hub_share = top_hub_count / len(degrees)
    top_hub_concentration = (
        max(
            0.0,
            min(
                1.0,
                (top_hub_share - uniform_top_hub_share)
                / max(1.0 - uniform_top_hub_share, 1e-12),
            ),
        )
        if total_degree > 0.0 and uniform_top_hub_share < 1.0
        else 0.0
    )
    canonical_edge_type_counts = dict(sorted(edge_type_counts.items()))
    return {
        "isolated": float(isolated),
        "isolated_ratio": isolated / max(len(degrees), 1),
        "median": float(median),
        "p95": float(p95),
        "hubness_ratio": p95 / max(float(median), 1.0),
        "degree_count": len(degrees),
        "total_degree": total_degree,
        "top_hub_count": top_hub_count,
        "top_hub_share": top_hub_share,
        "uniform_top_hub_share": uniform_top_hub_share,
        "top_hub_concentration": top_hub_concentration,
        "edge_type_counts": canonical_edge_type_counts,
        "distribution_hash": stable_hash(
            {
                "degrees": degrees,
                "edge_type_counts": canonical_edge_type_counts,
            }
        ),
    }


def _hubness_penalty_component(
    degree_stats: dict[str, Any],
    *,
    max_hubness_ratio: float,
    probe_hash: str,
    eligible_edge_types: set[str],
) -> dict[str, Any]:
    edge_type_counts = {
        str(edge_type): int(count)
        for edge_type, count in dict(degree_stats["edge_type_counts"]).items()
    }
    normalized_eligible_edge_types = {
        str(edge_type) for edge_type in eligible_edge_types
    }
    entropy_buckets = sorted(
        normalized_eligible_edge_types.union(edge_type_counts)
    )
    edge_type_total = float(sum(edge_type_counts.values()))
    if edge_type_total <= 0.0 or len(entropy_buckets) <= 1:
        edge_type_entropy = 1.0 if edge_type_total > 0.0 else 0.0
        edge_type_imbalance = 0.0
    else:
        raw_entropy = 0.0
        for edge_type in entropy_buckets:
            count = edge_type_counts.get(edge_type, 0)
            if count <= 0:
                continue
            probability = float(count) / edge_type_total
            raw_entropy -= probability * math.log(probability)
        edge_type_entropy = max(
            0.0,
            min(1.0, raw_entropy / math.log(len(entropy_buckets))),
        )
        edge_type_imbalance = 1.0 - edge_type_entropy
    dimensions = {
        "degree_p95_to_median_pressure": {
            "value": min(
                1.0,
                float(degree_stats["hubness_ratio"])
                / max(float(max_hubness_ratio), 1e-6),
            ),
            "weight": 1.0 / 3.0,
            "degree_p95": float(degree_stats["p95"]),
            "degree_median": float(degree_stats["median"]),
            "hard_gate_ratio": float(max_hubness_ratio),
        },
        "top_hub_concentration": {
            "value": float(degree_stats["top_hub_concentration"]),
            "weight": 1.0 / 3.0,
            "top_fraction": TPE_TOP_HUB_FRACTION,
            "top_hub_count": int(degree_stats["top_hub_count"]),
            "top_hub_share": float(degree_stats["top_hub_share"]),
            "uniform_share": float(degree_stats["uniform_top_hub_share"]),
        },
        "edge_type_imbalance": {
            "value": edge_type_imbalance,
            "weight": 1.0 / 3.0,
            "normalized_entropy": edge_type_entropy,
            "normalization_bucket_count": len(entropy_buckets),
            "eligible_edge_types": sorted(normalized_eligible_edge_types),
            "edge_type_counts": edge_type_counts,
        },
    }
    raw_dimension_sum = sum(float(card["value"]) for card in dimensions.values())
    value = sum(
        float(card["value"]) * float(card["weight"])
        for card in dimensions.values()
    )
    card = _component(
        "hubness_penalty",
        value,
        raw_dimension_sum,
        len(dimensions),
        probe_hash,
        protocol=TPE_HUBNESS_PENALTY_PROTOCOL_VERSION,
    )
    card.update(
        {
            "sample_count": int(degree_stats["degree_count"]),
            "dimensions": dimensions,
            "degree_distribution_hash": degree_stats["distribution_hash"],
        }
    )
    return card


def _normalized_graph_density_card(
    chunks: list[Chunk],
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate],
) -> dict[str, Any]:
    chunk_count = len(chunks)
    edge_count = len(candidates)
    unique_pairs = {
        tuple(
            sorted(
                (
                    str(candidate.source_chunk_id),
                    str(candidate.target_chunk_id),
                )
            )
        )
        for candidate in candidates.values()
        if str(candidate.source_chunk_id) != str(candidate.target_chunk_id)
    }
    unique_pair_count = len(unique_pairs)
    maximum_pair_count = (
        chunk_count * (chunk_count - 1) // 2 if chunk_count > 1 else 0
    )
    normalized_density = (
        unique_pair_count / maximum_pair_count
        if maximum_pair_count > 0
        else 0.0
    )
    sparse_edge_budget = tpe_scope_sparse_edge_budget(chunk_count)
    sparse_edge_budget_ratio = (
        edge_count / sparse_edge_budget
        if sparse_edge_budget > 0
        else (0.0 if edge_count == 0 else float(edge_count + 1))
    )
    return {
        "protocol_version": TPE_NORMALIZED_DENSITY_PROTOCOL_VERSION,
        "chunk_count": chunk_count,
        "edge_count": edge_count,
        "unique_undirected_pair_count": unique_pair_count,
        "duplicate_typed_pair_count": max(0, edge_count - unique_pair_count),
        "maximum_undirected_pair_count": maximum_pair_count,
        "normalized_edge_density": normalized_density,
        "mean_edges_per_node": edge_count / max(chunk_count, 1),
        "sparse_edge_budget": sparse_edge_budget,
        "sparse_edge_budget_ratio": sparse_edge_budget_ratio,
        "sparse_edge_budget_protocol_version": (
            TPE_SCOPE_SPARSE_EDGE_BUDGET_PROTOCOL_VERSION
        ),
        "model_call_count": 0,
    }


def _density_penalty_component(
    edge_density: float,
    *,
    edge_count: int,
    max_edge_density: float,
    probe_hash: str,
    density_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    soft_ceiling = float(max_edge_density) * TPE_DENSITY_SOFT_MAX_FRACTION
    penalty_span = max(float(max_edge_density) - soft_ceiling, 1e-6)
    excess = max(0.0, float(edge_density) - soft_ceiling)
    value = min(1.0, excess / penalty_span)
    card = _component(
        "density_penalty",
        value,
        excess,
        penalty_span,
        probe_hash,
        protocol=TPE_DENSITY_PENALTY_PROTOCOL_VERSION,
    )
    card.update(
        {
            "sample_count": int(edge_count),
            "target_interval": {"lower": 0.0, "upper": round(soft_ceiling, 6)},
            "hard_ceiling": round(float(max_edge_density), 6),
            "soft_ceiling_fraction": TPE_DENSITY_SOFT_MAX_FRACTION,
            "observed_edge_density": round(float(edge_density), 6),
            "density_diagnostics": dict(density_diagnostics or {}),
        }
    )
    return card


def _latency_distribution(samples_ms: list[float], *, phase: str) -> dict[str, Any]:
    if not samples_ms:
        raise ValueError(f"TPE latency phase {phase} requires at least one sample")
    if len(samples_ms) > TPE_LATENCY_SAMPLE_LIMIT:
        raise ValueError(
            f"TPE latency phase {phase} exceeds the {TPE_LATENCY_SAMPLE_LIMIT} sample limit"
        )
    samples: list[float] = []
    for value in samples_ms:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"TPE latency phase {phase} contains an invalid sample")
        samples.append(numeric)
    ordered = sorted(samples)

    def nearest_rank(percentile: float) -> float:
        index = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * percentile)) - 1))
        return ordered[index]

    return {
        "phase": phase,
        "sample_count": len(ordered),
        "samples_ms": [round(value, 6) for value in ordered],
        "min_ms": round(ordered[0], 6),
        "max_ms": round(ordered[-1], 6),
        "mean_ms": round(sum(ordered) / len(ordered), 6),
        "p50_ms": round(nearest_rank(0.50), 6),
        "p95_ms": round(nearest_rank(0.95), 6),
    }


def _tpe_latency_profile(
    *,
    candidate_adjacency_samples_ms: list[float],
    probe_expansion_samples_ms: list[float],
    metric_computation_samples_ms: list[float],
) -> dict[str, Any]:
    phases = {
        "candidate_adjacency": _latency_distribution(
            candidate_adjacency_samples_ms,
            phase="candidate_adjacency",
        ),
        "probe_expansion": _latency_distribution(
            probe_expansion_samples_ms,
            phase="probe_expansion",
        ),
        "metric_computation": _latency_distribution(
            metric_computation_samples_ms,
            phase="metric_computation",
        ),
    }
    phase_p95_ms = {
        phase: float(distribution["p95_ms"])
        for phase, distribution in phases.items()
    }
    return {
        "protocol_version": TPE_LATENCY_PROTOCOL_VERSION,
        "percentile_method": "nearest_rank_ceil_n_times_p_v1",
        "sample_limit_per_phase": TPE_LATENCY_SAMPLE_LIMIT,
        "sample_count": sum(int(item["sample_count"]) for item in phases.values()),
        "phase_p95_ms": phase_p95_ms,
        "candidate_latency_p95_ms": round(max(phase_p95_ms.values()), 6),
        "phases": phases,
    }


def _latency_budget_excess_penalty(
    candidate_latency_p95_ms: float,
    budget_ms: float,
    *,
    probe_hash: str | None,
) -> dict[str, Any]:
    latency = _finite_number(candidate_latency_p95_ms)
    budget = _finite_number(budget_ms)
    if latency is None or latency < 0.0:
        raise ValueError("TPE latency penalty requires a finite non-negative p95")
    if budget is None or budget <= 0.0:
        raise ValueError("TPE latency penalty requires a finite positive budget")
    excess_ms = max(0.0, latency - budget)
    excess_ratio = min(1.0, excess_ms / budget)
    card = _component(
        "latency_penalty",
        excess_ratio,
        excess_ms,
        budget,
        probe_hash,
        protocol=TPE_LATENCY_PENALTY_PROTOCOL_VERSION,
    )
    card.update(
        {
            "candidate_latency_p95_ms": round(latency, 6),
            "budget_ms": round(budget, 6),
            "excess_ms": round(excess_ms, 6),
            "normalization": "min(1,max(0,p95_ms-budget_ms)/budget_ms)",
            "penalty_applies_only_above_budget": True,
        }
    )
    return card


def evaluate_candidate_trial(
    db: Session,
    chunks: list[Chunk],
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate],
    diagnostics: dict[str, Any],
    candidate_adjacency_samples_ms: list[float],
    gate_profile: dict[str, Any],
    *,
    vectors: dict[str, list[float]] | None = None,
    candidate_rq_prefixes_by_chunk: dict[str, set[str]] | None = None,
    candidate_rq_diagnostics: dict[str, Any] | None = None,
    chunk_business_keys: dict[str, str] | None = None,
    canonical_business_keys_are_production: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float, str | None, str]:
    profile = dict(gate_profile)
    thresholds = dict(profile.get("hard_gate_thresholds") or {})
    metric_computation_samples_ms: list[float] = []
    metric_started_at = time.perf_counter()
    provided_business_keys = chunk_business_keys is not None
    chunk_ids_for_business_keys = {str(chunk.id) for chunk in chunks}
    if provided_business_keys and chunk_ids_for_business_keys.difference(
        {str(value) for value in (chunk_business_keys or {})}
    ):
        raise RuntimeError(
            "TPE quality evaluation received an incomplete canonical chunk key map"
        )
    production_canonical_business_keys = False
    if chunk_business_keys is None and all(isinstance(chunk, Chunk) for chunk in chunks):
        references = chunk_business_references(db, chunks)
        chunk_business_keys = dict(references.key_by_id)
        chunk_business_scope_hash = references.scope_hash
        production_canonical_business_keys = True
    else:
        chunk_business_keys = {
            str(chunk.id): str(
                (chunk_business_keys or {}).get(str(chunk.id))
                or _fallback_tpe_chunk_audit_key(chunk)
            )
            for chunk in chunks
        }
        chunk_business_scope_hash = stable_hash(
            sorted(chunk_business_keys.values())
        )
        production_canonical_business_keys = (
            bool(canonical_business_keys_are_production)
            if canonical_business_keys_are_production is not None
            else provided_business_keys
        )
    missing_business_keys = {
        str(chunk.id) for chunk in chunks
    }.difference(chunk_business_keys)
    if missing_business_keys:
        raise RuntimeError(
            "TPE quality evaluation requires a business key for every chunk"
        )
    if production_canonical_business_keys and len(
        set(chunk_business_keys.values())
    ) != len(chunk_business_keys):
        raise RuntimeError(
            "TPE quality evaluation detected a canonical chunk business-key collision"
        )
    probes, base_probe_hash = _probe_chunks(
        chunks,
        int(profile["tpe_probe_query_budget"]),
        chunk_business_keys=chunk_business_keys,
    )
    metric_computation_samples_ms.append((time.perf_counter() - metric_started_at) * 1000.0)
    if len(probes) < 2 or len(chunks) < 2:
        return {}, {}, 0.0, "insufficient_evaluation", base_probe_hash
    metric_started_at = time.perf_counter()
    adjacency = _adjacency(candidates)
    degree_stats = _degree_stats(chunks, candidates)
    chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
    unknown_candidate_endpoints = {
        endpoint_id
        for candidate in candidates.values()
        for endpoint_id in (
            str(candidate.source_chunk_id),
            str(candidate.target_chunk_id),
        )
        if endpoint_id not in chunk_by_id
    }
    if unknown_candidate_endpoints:
        raise ValueError(
            "TPE candidate adjacency references chunks outside its frozen scope"
        )
    language_identities = load_chunk_language_identities(db, chunks)

    def language_for(chunk: Chunk) -> str:
        identity = language_identities.get(str(chunk.id), {})
        return (
            str(identity.get("language") or "unknown").lower()
            if identity.get("known")
            else "unknown"
        )

    if candidate_rq_prefixes_by_chunk is None:
        candidate_rq_prefixes_by_chunk, candidate_rq_diagnostics = (
            _candidate_rq_prefix_inputs(
                chunks,
                vectors or {},
                chunk_business_keys=chunk_business_keys,
                canonical_business_keys_are_production=(
                    production_canonical_business_keys
                ),
            )
        )
    normalized_candidate_rq_prefixes: dict[str, set[str]] = {}
    invalid_rq_prefix_value_ids: set[str] = set()
    for chunk_id, values in dict(candidate_rq_prefixes_by_chunk or {}).items():
        normalized_chunk_id = str(chunk_id)
        if not isinstance(values, (list, tuple, set)):
            invalid_rq_prefix_value_ids.add(normalized_chunk_id)
            normalized_candidate_rq_prefixes[normalized_chunk_id] = set()
            continue
        normalized_candidate_rq_prefixes[normalized_chunk_id] = {
            str(value) for value in values if str(value)
        }
    candidate_rq_prefixes_by_chunk = normalized_candidate_rq_prefixes
    candidate_rq_diagnostics = dict(candidate_rq_diagnostics or {})
    rq_assignment_ids = set(candidate_rq_prefixes_by_chunk)
    rq_missing_assignment_ids = chunk_ids_for_business_keys.difference(
        rq_assignment_ids
    )
    rq_extra_assignment_ids = rq_assignment_ids.difference(
        chunk_ids_for_business_keys
    )
    rq_empty_assignment_ids = {
        chunk_id
        for chunk_id in chunk_ids_for_business_keys.intersection(rq_assignment_ids)
        if not candidate_rq_prefixes_by_chunk.get(chunk_id)
    }
    if (
        rq_missing_assignment_ids
        or rq_extra_assignment_ids
        or rq_empty_assignment_ids
    ):
        rq_scope_reasons = list(candidate_rq_diagnostics.get("reasons") or [])
        if rq_missing_assignment_ids:
            rq_scope_reasons.append("rq_prefix_scope_missing_chunks")
        if rq_extra_assignment_ids:
            rq_scope_reasons.append("rq_prefix_scope_extra_chunks")
        if rq_empty_assignment_ids:
            rq_scope_reasons.append("rq_prefix_scope_empty_assignments")
        if invalid_rq_prefix_value_ids:
            rq_scope_reasons.append("rq_prefix_scope_invalid_assignment_type")
        rq_assignment_scope_card = [
            {
                "chunk_business_key": chunk_business_keys[chunk_id],
                "prefixes": sorted(candidate_rq_prefixes_by_chunk.get(chunk_id, set())),
            }
            for chunk_id in sorted(
                chunk_ids_for_business_keys,
                key=lambda value: chunk_business_keys[value],
            )
        ]
        candidate_rq_diagnostics.update(
            {
                "enabled": False,
                "reason": "invalid_candidate_rq_assignment_scope",
                "reasons": sorted(set(rq_scope_reasons)),
                "missing_assignment_count": len(rq_missing_assignment_ids),
                "extra_assignment_count": len(rq_extra_assignment_ids),
                "empty_assignment_count": len(rq_empty_assignment_ids),
                "invalid_assignment_type_count": len(invalid_rq_prefix_value_ids),
                "missing_assignment_business_key_sample": sorted(
                    chunk_business_keys[chunk_id]
                    for chunk_id in rq_missing_assignment_ids
                )[:TPE_DIAGNOSTIC_SAMPLE_LIMIT],
                "extra_assignment_id_sample": sorted(rq_extra_assignment_ids)[
                    :TPE_DIAGNOSTIC_SAMPLE_LIMIT
                ],
                "empty_assignment_business_key_sample": sorted(
                    chunk_business_keys[chunk_id]
                    for chunk_id in rq_empty_assignment_ids
                )[:TPE_DIAGNOSTIC_SAMPLE_LIMIT],
                "assignment_scope_hash": stable_hash(rq_assignment_scope_card),
                "model_call_count": 0,
            }
        )

    structure_targets, structure_traceable_ids, structure_diagnostics = (
        _structure_positive_context(
            db,
            chunks,
            probes,
            chunk_business_keys=chunk_business_keys,
        )
    )
    expected_support, expected_support_diagnostics = _expected_support_context(
        db,
        chunks,
        probes,
        structure_targets,
        chunk_business_keys=chunk_business_keys,
        production_canonical_business_keys=production_canonical_business_keys,
    )
    probe_hash = stable_hash(
        {
            "protocol_version": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
            "base_probe_set_hash": base_probe_hash,
            "expected_support_pairs_hash": expected_support_diagnostics[
                "expected_support_pairs_hash"
            ],
            "structure_protocol_version": TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
            "structure_positive_counts": structure_diagnostics[
                "category_positive_counts"
            ],
            "candidate_rq_input_hash": candidate_rq_diagnostics.get(
                "input_hash"
            ),
        }
    )
    insufficient_reasons = []
    if int(expected_support_diagnostics["pair_count"]) < 2:
        insufficient_reasons.append("expected_support_pairs_below_two")
    if int(structure_diagnostics["positive_count"]) < 1:
        insufficient_reasons.append("structure_positive_set_empty")
    if not candidate_rq_diagnostics.get("enabled"):
        insufficient_reasons.append("candidate_rq_input_unavailable")
    if insufficient_reasons:
        return {}, {
            "evaluation_input": {
                "value": 0.0,
                "protocol": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
                "base_probe_set_hash": base_probe_hash,
                "probe_set_hash": probe_hash,
                "expected_support": expected_support_diagnostics,
                "structure_recovery": structure_diagnostics,
                "candidate_rq_input": candidate_rq_diagnostics,
                "blocking_reasons": insufficient_reasons,
                "chunk_business_key_protocol_version": (
                    CHUNK_BUSINESS_KEY_PROTOCOL_VERSION
                    if production_canonical_business_keys
                    else "tpe_local_chunk_audit_fallback_v1"
                ),
                "chunk_business_scope_hash": chunk_business_scope_hash,
                "production_canonical_business_keys": (
                    production_canonical_business_keys
                ),
                "gray_zone_semantics_changed": False,
                "gray_zone_rule_decision_model_call_count": 0,
                "model_call_count": 0,
            }
        }, 0.0, "insufficient_evaluation", probe_hash

    density_diagnostics = _normalized_graph_density_card(chunks, candidates)
    edge_count = int(density_diagnostics["edge_count"])
    chunk_count = max(1, len(chunks))
    edge_density = float(
        density_diagnostics["normalized_edge_density"]
    )
    metric_computation_samples_ms.append((time.perf_counter() - metric_started_at) * 1000.0)

    evidence_hits = 0
    evidence_denominator = 0
    structure_denominator = 0
    structure_hits = 0
    structure_category_cards = {
        category: {"hits": 0, "denominator": 0}
        for category in (
            "previous_next",
            "same_section",
            "same_page",
            "special_object_closure",
        )
    }
    diversity_documents: list[str] = []
    diversity_languages: list[str] = []
    diversity_edge_types: list[str] = []
    diversity_rq_prefixes: list[str] = []
    reached_ids_by_probe: dict[str, set[str]] = {}
    candidate_rows = list(candidates.values())
    probe_expansion_samples_ms: list[float] = []
    for chunk in probes:
        probe_started_at = time.perf_counter()
        probe_id = str(chunk.id)
        reached = _bounded_reachable_ids(probe_id, adjacency, max_depth=2)
        reached_ids_by_probe[probe_id] = reached
        expected_ids = expected_support.get(probe_id, set())
        evidence_hits += len(reached.intersection(expected_ids))
        evidence_denominator += len(expected_ids)
        for category, target_ids in structure_targets.get(probe_id, {}).items():
            category_hits = len(reached.intersection(target_ids))
            structure_category_cards[category]["hits"] += category_hits
            structure_category_cards[category]["denominator"] += len(target_ids)
            structure_hits += category_hits
            structure_denominator += len(target_ids)

        expanded_scope = reached | {probe_id}
        diversity_documents.extend(
            str(chunk_by_id[target_id].document_id)
            for target_id in sorted(reached)
            if target_id in chunk_by_id
        )
        diversity_languages.extend(
            language_for(chunk_by_id[target_id])
            for target_id in sorted(reached)
            if target_id in chunk_by_id
        )
        diversity_rq_prefixes.extend(
            prefix
            for target_id in sorted(reached)
            for prefix in sorted(
                candidate_rq_prefixes_by_chunk.get(target_id, set())
            )
        )
        diversity_edge_types.extend(
            str(candidate.edge_type)
            for candidate in candidate_rows
            if str(candidate.source_chunk_id) in expanded_scope
            and str(candidate.target_chunk_id) in expanded_scope
        )
        probe_expansion_samples_ms.append(
            (time.perf_counter() - probe_started_at) * 1000.0
        )
    structure_recovery_rate = structure_hits / max(structure_denominator, 1)
    evidence_recall_proxy = evidence_hits / max(evidence_denominator, 1)

    metric_started_at = time.perf_counter()
    non_isolated_ids = {
        str(chunk.id) for chunk in chunks if adjacency.get(str(chunk.id))
    }
    covered_documents = {
        str(chunk_by_id[chunk_id].document_id)
        for chunk_id in non_isolated_ids
        if chunk_id in chunk_by_id
    }
    all_documents = {str(chunk.document_id) for chunk in chunks}
    language_by_chunk = {
        str(chunk.id): language_for(chunk) for chunk in chunks
    }
    all_languages = set(language_by_chunk.values())
    known_languages = {language for language in all_languages if language != "unknown"}
    eligible_edge_types = {"dense_semantic"}
    if len(all_documents) > 1:
        eligible_edge_types.add("dense_cross_document_bridge")
    if len(known_languages) > 1:
        eligible_edge_types.add("dense_cross_language_bridge")
    covered_languages = {
        language_by_chunk[chunk_id]
        for chunk_id in non_isolated_ids
        if chunk_id in language_by_chunk
    }
    all_rq_prefixes = {
        prefix
        for values in candidate_rq_prefixes_by_chunk.values()
        for prefix in values
    }
    covered_rq_prefixes = {
        prefix
        for chunk_id in non_isolated_ids
        for prefix in candidate_rq_prefixes_by_chunk.get(chunk_id, set())
    }
    coverage_dimensions = {
        "active_chunks": {
            "numerator": len(non_isolated_ids),
            "denominator": len(chunks),
        },
        "documents": {
            "numerator": len(covered_documents),
            "denominator": len(all_documents),
        },
        "language_buckets": {
            "numerator": len(covered_languages),
            "denominator": len(all_languages),
        },
        "candidate_rq_prefixes": {
            "numerator": len(covered_rq_prefixes),
            "denominator": len(all_rq_prefixes),
        },
    }
    for card in coverage_dimensions.values():
        card["coverage"] = round(
            float(card["numerator"]) / max(int(card["denominator"]), 1),
            6,
        )
        card["weight"] = 0.25
    component_coverage = sum(
        float(card["coverage"]) * float(card["weight"])
        for card in coverage_dimensions.values()
    )
    metric_computation_samples_ms.append((time.perf_counter() - metric_started_at) * 1000.0)

    metric_started_at = time.perf_counter()
    precision_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            min(
                chunk_business_keys[str(item.source_chunk_id)],
                chunk_business_keys[str(item.target_chunk_id)],
            ),
            max(
                chunk_business_keys[str(item.source_chunk_id)],
                chunk_business_keys[str(item.target_chunk_id)],
            ),
            str(item.edge_type),
        ),
    )[:TPE_PRECISION_SAMPLE_LIMIT]
    precision_failure_counts: Counter[str] = Counter()
    precision_hits = 0
    for candidate in precision_candidates:
        support_audit = _candidate_support_feature_audit(candidate)
        traceable = {
            str(candidate.source_chunk_id),
            str(candidate.target_chunk_id),
        }.issubset(structure_traceable_ids)
        if support_audit["passed"] and traceable:
            precision_hits += 1
        else:
            if not traceable:
                precision_failure_counts["structure_traceability"] += 1
            for check, passed in support_audit["checks"].items():
                if not passed:
                    precision_failure_counts[check] += 1
    precision_denominator = len(precision_candidates)
    edge_precision_proxy = precision_hits / max(precision_denominator, 1)
    metric_computation_samples_ms.append((time.perf_counter() - metric_started_at) * 1000.0)

    metric_started_at = time.perf_counter()
    bridge_edges = [
        candidate
        for candidate in candidates.values()
        if candidate.edge_type
        in {"dense_cross_document_bridge", "dense_cross_language_bridge"}
        and (
            (
                candidate.edge_type == "dense_cross_document_bridge"
                and "cross_document_candidates"
                in set(
                    list(candidate.features_json.get("candidate_channels") or [])
                    + list(candidate.features_json.get("all_candidate_channels") or [])
                )
            )
            or (
                candidate.edge_type == "dense_cross_language_bridge"
                and "cross_language_candidates"
                in set(
                    list(candidate.features_json.get("candidate_channels") or [])
                    + list(candidate.features_json.get("all_candidate_channels") or [])
                )
            )
        )
    ]
    possible_bridge_sources = {
        str(chunk.id)
        for chunk in chunks
        if any(other.document_id != chunk.document_id for other in chunks)
        or any(
            language_for(chunk) != "unknown"
            and language_for(other) != "unknown"
            and language_for(other) != language_for(chunk)
            for other in chunks
        )
    }
    bridge_covered_ids = {
        str(edge.source_chunk_id) for edge in bridge_edges
    } | {str(edge.target_chunk_id) for edge in bridge_edges}
    bridge_opportunity_coverage = len(
        bridge_covered_ids.intersection(possible_bridge_sources)
    ) / max(len(possible_bridge_sources), 1)
    metric_computation_samples_ms.append((time.perf_counter() - metric_started_at) * 1000.0)

    metric_started_at = time.perf_counter()
    diversity_dimensions = {
        "documents": {
            "entropy": _entropy_ratio(diversity_documents),
            "sample_count": len(diversity_documents),
            "bucket_count": len(set(diversity_documents)),
        },
        "languages": {
            "entropy": _entropy_ratio(diversity_languages),
            "sample_count": len(diversity_languages),
            "bucket_count": len(set(diversity_languages)),
        },
        "edge_types": {
            "entropy": _entropy_ratio(diversity_edge_types),
            "sample_count": len(diversity_edge_types),
            "bucket_count": len(set(diversity_edge_types)),
        },
        "candidate_rq_prefixes": {
            "entropy": _entropy_ratio(diversity_rq_prefixes),
            "sample_count": len(diversity_rq_prefixes),
            "bucket_count": len(set(diversity_rq_prefixes)),
        },
    }
    path_diversity = sum(
        float(card["entropy"]) for card in diversity_dimensions.values()
    ) / len(diversity_dimensions)
    metric_computation_samples_ms.append((time.perf_counter() - metric_started_at) * 1000.0)
    latency_profile = _tpe_latency_profile(
        candidate_adjacency_samples_ms=candidate_adjacency_samples_ms,
        probe_expansion_samples_ms=probe_expansion_samples_ms,
        metric_computation_samples_ms=metric_computation_samples_ms,
    )
    candidate_latency_p95_ms = float(latency_profile["candidate_latency_p95_ms"])

    max_edge_density = float(thresholds["edge_density"]["threshold"])
    max_isolated_ratio = float(thresholds["isolated_ratio"]["threshold"])
    max_hubness_ratio = float(thresholds["hubness_ratio"]["threshold"])
    min_structure_recovery = float(thresholds["structure_recovery_rate"]["threshold"])
    max_latency = float(thresholds["candidate_latency_p95_ms"]["threshold"])
    hard_gate = {
        "edge_density": {
            "value": round(edge_density, 6),
            "threshold": max_edge_density,
            "passed": edge_density <= max_edge_density,
            "direction": "max",
            "metric_protocol": TPE_NORMALIZED_DENSITY_PROTOCOL_VERSION,
            "unique_undirected_pair_count": density_diagnostics[
                "unique_undirected_pair_count"
            ],
            "maximum_undirected_pair_count": density_diagnostics[
                "maximum_undirected_pair_count"
            ],
        },
        "sparse_edge_budget_ratio": {
            "value": round(
                float(density_diagnostics["sparse_edge_budget_ratio"]),
                6,
            ),
            "threshold": float(
                thresholds["sparse_edge_budget_ratio"]["threshold"]
            ),
            "passed": float(
                density_diagnostics["sparse_edge_budget_ratio"]
            )
            <= float(thresholds["sparse_edge_budget_ratio"]["threshold"]),
            "direction": "max",
            "edge_count": edge_count,
            "sparse_edge_budget": density_diagnostics[
                "sparse_edge_budget"
            ],
            "budget_protocol": TPE_SCOPE_SPARSE_EDGE_BUDGET_PROTOCOL_VERSION,
        },
        "isolated_ratio": {
            "value": round(float(degree_stats["isolated_ratio"]), 6),
            "threshold": max_isolated_ratio,
            "passed": float(degree_stats["isolated_ratio"]) <= max_isolated_ratio,
            "direction": "max",
        },
        "hubness_ratio": {
            "value": round(float(degree_stats["hubness_ratio"]), 6),
            "threshold": max_hubness_ratio,
            "passed": float(degree_stats["hubness_ratio"]) <= max_hubness_ratio,
            "direction": "max",
        },
        "structure_recovery_rate": {
            "value": round(structure_recovery_rate, 6),
            "threshold": min_structure_recovery,
            "passed": structure_recovery_rate >= min_structure_recovery,
            "direction": "min",
        },
        "candidate_latency_p95_ms": {
            "value": round(candidate_latency_p95_ms, 3),
            "threshold": max_latency,
            "passed": candidate_latency_p95_ms <= max_latency,
            "direction": "max",
            "latency_protocol_version": TPE_LATENCY_PROTOCOL_VERSION,
            "phase_p95_ms": latency_profile["phase_p95_ms"],
            "sample_count": latency_profile["sample_count"],
        },
    }
    failed = [name for name, gate in hard_gate.items() if not gate["passed"]]
    components = {
        "evidence_recall_proxy": _component(
            "evidence_recall_proxy",
            evidence_recall_proxy,
            evidence_hits,
            evidence_denominator,
            probe_hash,
            protocol=TPE_EXPECTED_SUPPORT_PROTOCOL_VERSION,
        ),
        "structure_recovery_rate": _component(
            "structure_recovery_rate",
            structure_recovery_rate,
            structure_hits,
            structure_denominator,
            probe_hash,
            protocol=TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
        ),
        "component_coverage": _component(
            "component_coverage",
            component_coverage,
            sum(float(card["coverage"]) for card in coverage_dimensions.values()),
            len(coverage_dimensions),
            probe_hash,
            protocol=TPE_COMPONENT_COVERAGE_PROTOCOL_VERSION,
        ),
        "edge_precision_proxy": _component(
            "edge_precision_proxy",
            edge_precision_proxy,
            precision_hits,
            precision_denominator,
            probe_hash,
            protocol=TPE_EDGE_PRECISION_PROTOCOL_VERSION,
        ),
        "bridge_opportunity_coverage": _component(
            "bridge_opportunity_coverage",
            bridge_opportunity_coverage,
            len(bridge_covered_ids.intersection(possible_bridge_sources)),
            len(possible_bridge_sources),
            probe_hash,
            protocol=TPE_BRIDGE_OPPORTUNITY_PROTOCOL_VERSION,
        ),
        "path_diversity": _component(
            "path_diversity",
            path_diversity,
            sum(float(card["entropy"]) for card in diversity_dimensions.values()),
            len(diversity_dimensions),
            probe_hash,
            protocol=TPE_PATH_DIVERSITY_PROTOCOL_VERSION,
        ),
        "hubness_penalty": _hubness_penalty_component(
            degree_stats,
            max_hubness_ratio=max_hubness_ratio,
            probe_hash=probe_hash,
            eligible_edge_types=eligible_edge_types,
        ),
        "density_penalty": _density_penalty_component(
            edge_density,
            edge_count=edge_count,
            max_edge_density=max_edge_density,
            probe_hash=probe_hash,
            density_diagnostics=density_diagnostics,
        ),
        "latency_penalty": _latency_budget_excess_penalty(
            candidate_latency_p95_ms,
            max_latency,
            probe_hash=probe_hash,
        ),
    }
    components["evidence_recall_proxy"]["expected_support"] = (
        expected_support_diagnostics
    )
    components["structure_recovery_rate"]["categories"] = {
        category: {
            **card,
            "value": round(
                float(card["hits"]) / max(int(card["denominator"]), 1),
                6,
            ),
        }
        for category, card in structure_category_cards.items()
    }
    components["structure_recovery_rate"]["input_diagnostics"] = (
        structure_diagnostics
    )
    components["component_coverage"]["dimensions"] = coverage_dimensions
    components["component_coverage"]["candidate_rq_input"] = (
        candidate_rq_diagnostics
    )
    components["edge_precision_proxy"]["sample_limit"] = (
        TPE_PRECISION_SAMPLE_LIMIT
    )
    components["edge_precision_proxy"]["candidate_count"] = edge_count
    components["edge_precision_proxy"]["failure_counts"] = dict(
        sorted(precision_failure_counts.items())
    )
    components["edge_precision_proxy"]["sample_hash"] = stable_hash(
        [
            {
                "source_chunk_business_key": min(
                    chunk_business_keys[str(candidate.source_chunk_id)],
                    chunk_business_keys[str(candidate.target_chunk_id)],
                ),
                "target_chunk_business_key": max(
                    chunk_business_keys[str(candidate.source_chunk_id)],
                    chunk_business_keys[str(candidate.target_chunk_id)],
                ),
                "edge_type": str(candidate.edge_type),
                "raw_strength": round(float(candidate.raw_strength), 6),
            }
            for candidate in precision_candidates
        ]
    )
    components["bridge_opportunity_coverage"]["dedicated_quota_edge_count"] = len(
        bridge_edges
    )
    components["path_diversity"]["dimensions"] = diversity_dimensions
    components["path_diversity"]["probe_expansion_count"] = len(
        reached_ids_by_probe
    )
    components["evaluation_input"] = {
        "value": 1.0,
        "protocol": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
        "base_probe_set_hash": base_probe_hash,
        "probe_set_hash": probe_hash,
        "probe_count": len(probes),
        "expected_support_pairs_hash": expected_support_diagnostics[
            "expected_support_pairs_hash"
        ],
        "candidate_rq_input_hash": candidate_rq_diagnostics.get("input_hash"),
        "chunk_business_key_protocol_version": (
            CHUNK_BUSINESS_KEY_PROTOCOL_VERSION
            if production_canonical_business_keys
            else "tpe_local_chunk_audit_fallback_v1"
        ),
        "chunk_business_scope_hash": chunk_business_scope_hash,
        "production_canonical_business_keys": production_canonical_business_keys,
        "gray_zone_semantics_changed": False,
        "gray_zone_rule_decision_model_call_count": 0,
        "model_call_count": 0,
    }
    components["latency_penalty"]["sample_count"] = latency_profile["sample_count"]
    components["latency_penalty"]["latency_profile"] = latency_profile
    score = (
        0.26 * evidence_recall_proxy
        + 0.18 * structure_recovery_rate
        + 0.16 * component_coverage
        + 0.12 * edge_precision_proxy
        + 0.10 * bridge_opportunity_coverage
        + 0.08 * path_diversity
        - 0.12 * components["hubness_penalty"]["value"]
        - 0.10 * components["density_penalty"]["value"]
        - 0.06 * components["latency_penalty"]["value"]
    )
    failure_code = "hard_gate_failed:" + ",".join(failed) if failed else None
    components["candidate_diagnostics"] = {
        "value": 1.0,
        "protocol": AUTO_TPE_PROTOCOL,
        "quality_proxy_protocol": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
        "candidate_count": edge_count,
        "diagnostics": diagnostics,
        "gray_zone_semantics_changed": False,
        "gray_zone_rule_decision_model_call_count": 0,
        "model_call_count": 0,
    }
    return hard_gate, components, round(score, 6), failure_code, probe_hash


def _latest_completed_theta(
    db: Session,
    knowledge_base_id: str,
    chat_model: str,
    embedding_model: str,
    embedding_text_version: str,
    chunk_scope_hash: str,
) -> dict[str, Any] | None:
    runs = list(
        db.scalars(
            select(AutoTpeRun)
            .where(
                AutoTpeRun.knowledge_base_id == knowledge_base_id,
                AutoTpeRun.status == "completed",
                AutoTpeRun.chat_model == chat_model,
                AutoTpeRun.embedding_model == embedding_model,
                AutoTpeRun.embedding_text_version == embedding_text_version,
                AutoTpeRun.chunk_scope_hash == chunk_scope_hash,
                AutoTpeRun.protocol_hash == _protocol_hash(),
                AutoTpeRun.chunk_relation_graph_state_id.is_not(None),
            )
            .order_by(AutoTpeRun.chunk_version.desc(), AutoTpeRun.completed_at.desc())
        ).all()
    )
    for run in runs:
        if run.selected_theta_json and tpe_run_has_valid_active_promotion(db, run):
            return dict(run.selected_theta_json)
    return None


def _persist_trial_terminal(
    db: Session,
    trial: AutoTpeTrial,
    *,
    status: str,
    details: dict[str, Any],
) -> None:
    transition_tpe_trial(
        db,
        trial,
        status,
        details=details,
        candidate_adjacency_hash=trial.candidate_adjacency_hash,
        probe_set_hash=trial.probe_set_hash,
        hard_gate_json=dict(trial.hard_gate_json or {}),
        objective_components_json=dict(trial.objective_components_json or {}),
        objective_score=trial.objective_score,
        failure_code=trial.failure_code,
        diagnostics_json=dict(trial.diagnostics_json or {}),
        finished_at=trial.finished_at or _now(),
    )


def _stop_tpe_after_runtime_disable(
    db: Session,
    run: AutoTpeRun,
    *,
    batch_id: str | None,
    chunk_version: int,
    trial_boundary: int,
    completed_trial_count: int,
    base_theta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reason = "auto_tpe_disabled_at_trial_boundary"
    fallback_theta_hash = _theta_hash(base_theta)
    transition_tpe_run(
        db,
        run,
        "cancelled",
        details={
            "phase": "trial_boundary",
            "trial_boundary": trial_boundary,
            "completed_trial_count": completed_trial_count,
            "blocking_reasons": [reason],
            "retry_boundary": "next_chunk_version_graph_build",
            "fallback_theta_hash": fallback_theta_hash,
        },
        failure_code=reason,
        blocking_reasons_json=[reason],
        last_error="Automatic TPE was disabled at a trial boundary",
        completed_at=_now(),
    )
    emit_ingestion_log(
        batch_id,
        "auto_tpe_skipped",
        "自动 TPE 在 trial 边界检测到运行时开关已关闭，沿用上一版或默认工作点",
        run_id=run.id,
        chunk_version=chunk_version,
        reason=reason,
        trial_boundary=trial_boundary,
        completed_trial_count=completed_trial_count,
        fallback_theta_hash=fallback_theta_hash,
    )
    return dict(base_theta), {
        "status": "cancelled",
        "reason": reason,
        "run_id": run.id,
        "chunk_version": chunk_version,
        "trial_boundary": trial_boundary,
        "completed_trial_count": completed_trial_count,
        "fallback_theta_hash": fallback_theta_hash,
    }


def select_auto_tpe_operating_point(
    db: Session,
    knowledge_base_id: str,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    *,
    fallback_operating_point: dict[str, Any],
    batch_id: str | None,
    chunk_version_incremented: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    refresh_runtime_settings_if_needed(force=True)
    settings = get_settings()
    chunk_version = max(chunk.chunk_version for chunk in chunks)
    chunk_scope_hash = compute_chunk_scope_hash(chunks)
    fallback_theta = dict(fallback_operating_point or dense_graph_operating_point())
    fallback_rejection_reasons = preflight_theta(fallback_theta)
    if fallback_rejection_reasons:
        base_theta = normalize_theta(dense_graph_operating_point())
    else:
        base_theta = normalize_theta(fallback_theta)
        if fallback_theta.get("optimizer"):
            base_theta["optimizer"] = str(fallback_theta["optimizer"])
    base_theta_reasons = preflight_theta(base_theta)
    if base_theta_reasons:
        raise RuntimeError(
            "Versioned default TPE theta failed preflight: "
            + "; ".join(base_theta_reasons)
        )
    reconcile_tpe_audit(db, knowledge_base_id=knowledge_base_id)
    previous_theta = _latest_completed_theta(
        db,
        knowledge_base_id,
        settings.graph_model,
        settings.embedding_model,
        CURRENT_EMBEDDING_TEXT_VERSION,
        chunk_scope_hash,
    )
    if previous_theta and not preflight_theta(previous_theta):
        base_theta = normalize_theta(previous_theta)

    if not settings.enable_auto_tpe:
        emit_ingestion_log(batch_id, "auto_tpe_skipped", "自动 TPE 未开启，沿用上一版或默认工作点", reason="disabled", chunk_version=chunk_version)
        return base_theta, {
            "status": "skipped",
            "reason": "disabled",
            "chunk_version": chunk_version,
            "fallback_prior_rejected": bool(fallback_rejection_reasons),
            "fallback_prior_rejection_reasons": fallback_rejection_reasons,
        }
    if not chunk_version_incremented:
        emit_ingestion_log(batch_id, "auto_tpe_skipped", "chunk 最高版本号未递增，跳过自动 TPE", reason="chunk_version_not_incremented", chunk_version=chunk_version)
        return base_theta, {
            "status": "skipped",
            "reason": "chunk_version_not_incremented",
            "chunk_version": chunk_version,
            "fallback_prior_rejected": bool(fallback_rejection_reasons),
            "fallback_prior_rejection_reasons": fallback_rejection_reasons,
        }

    if all(isinstance(chunk, Chunk) for chunk in chunks):
        chunk_references = chunk_business_references(db, chunks)
        chunk_business_keys = dict(chunk_references.key_by_id)
        chunk_business_scope_hash = chunk_references.scope_hash
        chunk_business_key_protocol_version = CHUNK_BUSINESS_KEY_PROTOCOL_VERSION
        production_canonical_business_keys = True
    else:
        chunk_business_keys = {
            str(chunk.id): _fallback_tpe_chunk_audit_key(chunk) for chunk in chunks
        }
        chunk_business_scope_hash = stable_hash(
            sorted(chunk_business_keys.values())
        )
        chunk_business_key_protocol_version = "tpe_local_chunk_audit_fallback_v1"
        production_canonical_business_keys = False
    run_start_runtime_hash = _runtime_hash()
    run = AutoTpeRun(
        knowledge_base_id=knowledge_base_id,
        batch_id=batch_id,
        chunk_version=chunk_version,
        chunk_scope_hash=chunk_scope_hash,
        graph_operating_point_protocol=str(
            base_theta["graph_operating_point_protocol"]
        ),
        protocol_hash=_protocol_hash(),
        tpe_search_space_hash=tpe_search_space_hash(),
        chat_model=settings.graph_model,
        embedding_model=settings.embedding_model,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        status="running",
        trigger_reason="chunk_version_incremented",
        trial_budget=int(settings.tpe_trial_budget),
        startup_random_trials=int(settings.tpe_startup_random_trials),
        good_quantile_gamma=float(settings.tpe_good_quantile_gamma),
        probe_query_budget=int(settings.tpe_probe_query_budget),
        candidate_pool_size=int(settings.tpe_candidate_pool_size),
        runtime_settings_hash=run_start_runtime_hash,
        diagnostics_json={
            "protocol": AUTO_TPE_PROTOCOL,
            "tpe_search_space_protocol_version": TPE_SEARCH_SPACE_PROTOCOL_VERSION,
            "tpe_search_space_hash": tpe_search_space_hash(),
            "tpe_search_space": tpe_search_space_spec(),
            "fallback_prior_rejected": bool(fallback_rejection_reasons),
            "fallback_prior_rejection_reasons": fallback_rejection_reasons,
            **_relation_quota_protocol_diagnostics(),
            "no_llm": True,
            "no_embedding": True,
            "no_mid_coarse": True,
            "no_shadow_graph": True,
            "run_lease_expires_at": trial_lease_expires_at(
                float(settings.tpe_trial_timeout_seconds) * max(1, int(settings.tpe_trial_budget))
            ),
            "retry_boundary": "next_trial_boundary",
            "run_started_runtime_settings_hash": run_start_runtime_hash,
            "chunk_business_key_protocol_version": (
                chunk_business_key_protocol_version
            ),
            "chunk_business_scope_hash": chunk_business_scope_hash,
            "production_canonical_business_keys": (
                production_canonical_business_keys
            ),
        },
        started_at=_now(),
    )
    persist_tpe_run(db, run)
    emit_ingestion_log(
        batch_id,
        "auto_tpe_started",
        "自动 TPE 轻量调参开始",
        run_id=run.id,
        chunk_version=chunk_version,
        trial_budget=run.trial_budget,
        embedding_model=run.embedding_model,
        chat_model=run.chat_model,
    )

    candidate_rq_started_at = time.perf_counter()
    candidate_rq_prefixes_by_chunk, candidate_rq_diagnostics = (
        _candidate_rq_prefix_inputs(
            chunks,
            vectors,
            chunk_business_keys=chunk_business_keys,
            canonical_business_keys_are_production=(
                production_canonical_business_keys
            ),
        )
    )
    candidate_rq_diagnostics = {
        **candidate_rq_diagnostics,
        "precompute_latency_ms": round(
            (time.perf_counter() - candidate_rq_started_at) * 1000.0,
            6,
        ),
        "precomputed_once_per_run": True,
    }
    run.diagnostics_json = {
        **dict(run.diagnostics_json or {}),
        "tpe_quality_proxy_protocol_version": TPE_QUALITY_PROXY_PROTOCOL_VERSION,
        "candidate_rq_input": candidate_rq_diagnostics,
        "gray_zone_semantics_changed": False,
        "gray_zone_rule_decision_model_call_count": 0,
    }
    db.flush()

    completed_trials: list[AutoTpeTrial] = []
    for trial_index in range(1, int(run.trial_budget) + 1):
        refresh_runtime_settings_if_needed(force=True)
        boundary_settings = get_settings()
        if not boundary_settings.enable_auto_tpe:
            return _stop_tpe_after_runtime_disable(
                db,
                run,
                batch_id=batch_id,
                chunk_version=chunk_version,
                trial_boundary=trial_index,
                completed_trial_count=len(completed_trials),
                base_theta=base_theta,
            )
        trial_runtime_settings_hash = _runtime_hash()
        trial_gate_profile = tpe_gate_profile(boundary_settings)
        trial_gate_profile_hash = tpe_gate_profile_hash(trial_gate_profile)
        try:
            ensure_not_cancelled(db, batch_id)
            theta, sampler_hash = _sample_tpe_theta(
                run,
                completed_trials,
                trial_index,
                base_theta,
                settings=boundary_settings,
                runtime_settings_hash=trial_runtime_settings_hash,
                gate_profile_hash=trial_gate_profile_hash,
                chunk_count=len(chunks),
            )
        except IngestionCancelled:
            transition_tpe_run(
                db,
                run,
                "cancelled",
                details={
                    "phase": "trial_boundary",
                    "trial_index": trial_index,
                    "retry_boundary": "next_graph_build",
                    "blocking_reasons": ["batch_cancelled"],
                },
                failure_code="batch_cancelled",
                blocking_reasons_json=["batch_cancelled"],
                last_error="Ingestion batch cancellation observed at the TPE trial boundary",
                completed_at=_now(),
            )
            raise
        except Exception as exc:
            transition_tpe_run(
                db,
                run,
                "failed",
                details={
                    "phase": "sampler",
                    "trial_index": trial_index,
                    "retry_boundary": "next_graph_build",
                    "blocking_reasons": ["tpe_sampler_failed"],
                },
                failure_code="tpe_sampler_failed",
                blocking_reasons_json=["tpe_sampler_failed"],
                last_error=str(exc),
                completed_at=_now(),
            )
            raise
        invalid = preflight_theta(theta)
        calibration_audit: dict[str, Any] = {}
        if not invalid:
            try:
                calibration_audit = theta_calibration_audit(theta)
            except (TypeError, ValueError) as exc:
                invalid.append(f"invalid_edge_calibration_audit:{exc}")
        trial = AutoTpeTrial(
            run_id=run.id,
            knowledge_base_id=knowledge_base_id,
            build_batch_id=batch_id,
            chunk_scope_hash=chunk_scope_hash,
            embedding_model=run.embedding_model,
            embedding_text_version=run.embedding_text_version,
            trial_index=trial_index,
            sampled_theta_json=theta,
            theta_hash=_theta_hash(theta),
            tpe_search_space_hash=tpe_search_space_hash(),
            edge_distance_protocol=calibration_audit.get("edge_distance_protocol"),
            edge_distance_protocol_hash=calibration_audit.get(
                "edge_distance_protocol_hash"
            ),
            edge_type_calibration_protocol=calibration_audit.get(
                "edge_type_calibration_protocol"
            ),
            edge_type_calibration_protocol_hash=calibration_audit.get(
                "edge_type_calibration_protocol_hash"
            ),
            calibration_params_json=dict(
                calibration_audit.get("calibration_params") or {}
            ),
            calibration_params_hash=calibration_audit.get(
                "calibration_params_hash"
            ),
            edge_type_calibration_config_hash=calibration_audit.get(
                "edge_type_calibration_config_hash"
            ),
            sampler_state_hash=sampler_hash,
            runtime_settings_hash=trial_runtime_settings_hash,
            gate_profile_hash=trial_gate_profile_hash,
            gate_profile_json=trial_gate_profile,
            status="running",
            failure_code=None,
            diagnostics_json={
                "invalid_theta_reasons": invalid,
                "sampler": "tpe_l_over_g_lightweight_v1",
                "tpe_search_space_protocol_version": TPE_SEARCH_SPACE_PROTOCOL_VERSION,
                "tpe_search_space_hash": tpe_search_space_hash(),
                "theta_calibration_audit": calibration_audit,
                "scope_search": tpe_scope_search_card(len(chunks)),
                "trial_lease_expires_at": trial_lease_expires_at(
                    float(trial_gate_profile["tpe_trial_timeout_seconds"])
                ),
                "retry_boundary": "next_trial_boundary",
                **_relation_quota_protocol_diagnostics(),
            },
            started_at=_now(),
        )
        persist_tpe_trial(db, trial)
        update_tpe_run(
            db,
            run,
            sampler_state_hash=sampler_hash,
            diagnostics_json={
                **dict(run.diagnostics_json or {}),
                "run_lease_expires_at": trial_lease_expires_at(
                    float(trial_gate_profile["tpe_trial_timeout_seconds"])
                    * max(1, int(run.trial_budget) - trial_index + 1)
                ),
                "current_trial_index": trial_index,
                "current_trial_runtime_settings_hash": trial_runtime_settings_hash,
                "current_trial_gate_profile_hash": trial_gate_profile_hash,
            },
        )
        emit_ingestion_log(batch_id, "auto_tpe_trial_started", f"自动 TPE trial {trial_index} 开始", run_id=run.id, trial_id=trial.id, trial_index=trial_index)
        if invalid:
            transition_tpe_trial(
                db,
                trial,
                "blocked",
                details={
                    "phase": "theta_preflight",
                    "blocking_reasons": invalid,
                    "retry_boundary": "next_trial_boundary",
                },
                failure_code="invalid_theta_preflight",
                finished_at=_now(),
            )
            emit_ingestion_log(batch_id, "auto_tpe_trial_blocked", f"自动 TPE trial {trial_index} 参数预检失败", run_id=run.id, trial_id=trial.id, reasons=invalid)
            completed_trials.append(trial)
            continue
        started = time.perf_counter()
        try:
            candidates, diagnostics = relation_edge_candidates(db, chunks, vectors, theta)
            simulated_calibration = dict(diagnostics.get("edge_type_calibration") or {})
            if not simulated_calibration:
                raise ValueError(
                    "TPE candidate simulation did not return edge calibration diagnostics"
                )
            if (
                simulated_calibration.get("protocol_hash")
                != calibration_audit.get("edge_type_calibration_protocol_hash")
                or simulated_calibration.get("edge_distance_protocol_hash")
                != calibration_audit.get("edge_distance_protocol_hash")
                or simulated_calibration.get("calibration_params_hash")
                != calibration_audit.get("calibration_params_hash")
                or simulated_calibration.get("edge_type_calibration_config_hash")
                != calibration_audit.get("edge_type_calibration_config_hash")
            ):
                raise ValueError(
                    "TPE candidate simulation calibration identity does not match sampled theta"
                )
            quota_signal_diagnostics = dict(diagnostics.get("relation_quota_signals") or {})
            language_identity_diagnostics = dict(
                quota_signal_diagnostics.get("language_identity") or {}
            )
            candidate_hash = _candidate_adjacency_hash(
                candidates,
                theta_hash=trial.theta_hash,
                language_identity_scope_hash=str(
                    language_identity_diagnostics.get("scope_hash") or ""
                ),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            trial.diagnostics_json = {
                **(trial.diagnostics_json or {}),
                "relation_quota_signal_scope_hash": quota_signal_diagnostics.get("signal_scope_hash"),
                "language_identity": language_identity_diagnostics,
                "out_evidence_mass_distribution": quota_signal_diagnostics.get("out_evidence_mass_distribution"),
                "in_acceptance_capacity_distribution": quota_signal_diagnostics.get("in_acceptance_capacity_distribution"),
                "theta_calibration_audit": calibration_audit,
                "simulated_edge_type_calibration": simulated_calibration,
            }
            if elapsed_ms > float(trial_gate_profile["tpe_trial_timeout_seconds"]) * 1000.0:
                terminal_status = "failed"
                trial.failure_code = "trial_timeout"
                trial.candidate_adjacency_hash = candidate_hash
                trial.diagnostics_json = {
                    **(trial.diagnostics_json or {}),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "candidate_count": len(candidates),
                    "trial_timeout_seconds": float(
                        trial_gate_profile["tpe_trial_timeout_seconds"]
                    ),
                }
            else:
                hard_gate, components, score, failure_code, probe_hash = evaluate_candidate_trial(
                    db,
                    chunks,
                    candidates,
                    diagnostics,
                    [elapsed_ms],
                    trial_gate_profile,
                    vectors=vectors,
                    candidate_rq_prefixes_by_chunk=(
                        candidate_rq_prefixes_by_chunk
                    ),
                    candidate_rq_diagnostics=candidate_rq_diagnostics,
                    chunk_business_keys=chunk_business_keys,
                    canonical_business_keys_are_production=(
                        production_canonical_business_keys
                    ),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                trial.candidate_adjacency_hash = candidate_hash
                trial.probe_set_hash = probe_hash
                trial.hard_gate_json = hard_gate
                trial.objective_components_json = components
                trial.objective_score = score
                latency_profile = dict(
                    (components.get("latency_penalty") or {}).get("latency_profile") or {}
                )
                trial.diagnostics_json = {
                    **(trial.diagnostics_json or {}),
                    "latency_protocol_version": latency_profile.get("protocol_version"),
                    "candidate_latency_p95_ms": latency_profile.get("candidate_latency_p95_ms"),
                    "latency_sample_count": latency_profile.get("sample_count"),
                }
                if elapsed_ms > float(trial_gate_profile["tpe_trial_timeout_seconds"]) * 1000.0:
                    terminal_status = "failed"
                    trial.failure_code = "trial_timeout"
                    trial.diagnostics_json = {
                        **(trial.diagnostics_json or {}),
                        "evaluation_failure_code_before_timeout": failure_code,
                    }
                else:
                    terminal_status = "blocked" if failure_code else "completed"
                    trial.failure_code = failure_code
                trial.diagnostics_json = {
                    **(trial.diagnostics_json or {}),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "candidate_count": len(candidates),
                    "trial_timeout_seconds": float(
                        trial_gate_profile["tpe_trial_timeout_seconds"]
                    ),
                }
            trial.finished_at = _now()
            _persist_trial_terminal(
                db,
                trial,
                status=terminal_status,
                details={
                    "phase": "candidate_evaluation",
                    "blocking_reasons": [trial.failure_code] if trial.failure_code else [],
                    "retry_boundary": "none" if terminal_status == "completed" else "next_trial_boundary",
                },
            )
            event = "auto_tpe_trial_blocked" if trial.status != "completed" else "auto_tpe_trial_completed"
            emit_ingestion_log(
                batch_id,
                event,
                f"自动 TPE trial {trial_index} {'完成' if trial.status == 'completed' else '被阻断'}",
                run_id=run.id,
                trial_id=trial.id,
                trial_index=trial_index,
                objective_score=trial.objective_score,
                failure_code=trial.failure_code,
                hard_gate=trial.hard_gate_json,
            )
        except IngestionCancelled:
            trial.failure_code = "batch_cancelled"
            trial.finished_at = _now()
            trial.diagnostics_json = {
                **(trial.diagnostics_json or {}),
                "blocking_reasons": ["batch_cancelled"],
                "retry_boundary": "next_graph_build",
            }
            _persist_trial_terminal(
                db,
                trial,
                status="cancelled",
                details={
                    "phase": "candidate_evaluation",
                    "blocking_reasons": ["batch_cancelled"],
                    "retry_boundary": "next_graph_build",
                },
            )
            transition_tpe_run(
                db,
                run,
                "cancelled",
                details={
                    "phase": "candidate_evaluation",
                    "trial_index": trial_index,
                    "blocking_reasons": ["batch_cancelled"],
                    "retry_boundary": "next_graph_build",
                },
                failure_code="batch_cancelled",
                blocking_reasons_json=["batch_cancelled"],
                last_error="Ingestion batch cancellation observed during a TPE trial",
                completed_at=_now(),
            )
            raise
        except Exception as exc:
            trial.failure_code = "candidate_simulation_failed"
            trial.finished_at = _now()
            trial.diagnostics_json = {
                **(trial.diagnostics_json or {}),
                "error": str(exc),
                "blocking_reasons": ["candidate_simulation_failed"],
                "retry_boundary": "next_trial_boundary",
            }
            _persist_trial_terminal(
                db,
                trial,
                status="failed",
                details={
                    "phase": "candidate_evaluation",
                    "blocking_reasons": ["candidate_simulation_failed"],
                    "retry_boundary": "next_trial_boundary",
                },
            )
            emit_ingestion_log(batch_id, "auto_tpe_trial_blocked", f"自动 TPE trial {trial_index} 模拟失败：{exc}", run_id=run.id, trial_id=trial.id, error=str(exc))
        completed_trials.append(trial)

    refresh_runtime_settings_if_needed(force=True)
    selection_settings = get_settings()
    if not selection_settings.enable_auto_tpe:
        return _stop_tpe_after_runtime_disable(
            db,
            run,
            batch_id=batch_id,
            chunk_version=chunk_version,
            trial_boundary=int(run.trial_budget) + 1,
            completed_trial_count=len(completed_trials),
            base_theta=base_theta,
        )
    selection_runtime_settings_hash = _runtime_hash()
    selection_graph_runtime_hash = _graph_runtime_hash()
    selection_gate_profile = tpe_gate_profile(selection_settings)
    selection_gate_profile_hash = tpe_gate_profile_hash(selection_gate_profile)
    try:
        ensure_not_cancelled(db, batch_id)
    except IngestionCancelled:
        transition_tpe_run(
            db,
            run,
            "cancelled",
            details={
                "phase": "post_trial_boundary",
                "retry_boundary": "next_graph_build",
                "blocking_reasons": ["batch_cancelled"],
            },
            failure_code="batch_cancelled",
            blocking_reasons_json=["batch_cancelled"],
            last_error="Ingestion batch cancellation observed after the final TPE trial",
            completed_at=_now(),
        )
        raise

    all_valid_trials = [trial for trial in completed_trials if tpe_trial_is_valid(trial)]
    valid_trials = [
        trial
        for trial in all_valid_trials
        if trial.runtime_settings_hash == selection_runtime_settings_hash
        and trial.gate_profile_hash == selection_gate_profile_hash
    ]
    if not valid_trials:
        failure_codes = {trial.failure_code for trial in completed_trials}
        if failure_codes == {"insufficient_evaluation"}:
            base_calibration_audit = theta_calibration_audit(base_theta)
            transition_tpe_run(
                db,
                run,
                "skipped",
                details={
                    "phase": "trial_selection",
                    "blocking_reasons": ["insufficient_evaluation"],
                    "retry_boundary": "next_chunk_version_graph_build",
                },
                completed_at=_now(),
                failure_code="insufficient_evaluation",
                blocking_reasons_json=["insufficient_evaluation"],
                selected_theta_json=base_theta,
                selected_theta_hash=_theta_hash(base_theta),
                selected_edge_distance_protocol=base_calibration_audit[
                    "edge_distance_protocol"
                ],
                selected_edge_distance_protocol_hash=base_calibration_audit[
                    "edge_distance_protocol_hash"
                ],
                selected_edge_type_calibration_protocol=base_calibration_audit[
                    "edge_type_calibration_protocol"
                ],
                selected_edge_type_calibration_protocol_hash=base_calibration_audit[
                    "edge_type_calibration_protocol_hash"
                ],
                selected_calibration_params_json=dict(
                    base_calibration_audit["calibration_params"]
                ),
                selected_calibration_params_hash=base_calibration_audit[
                    "calibration_params_hash"
                ],
                selected_edge_type_calibration_config_hash=base_calibration_audit[
                    "edge_type_calibration_config_hash"
                ],
            )
            emit_ingestion_log(
                batch_id,
                "auto_tpe_skipped",
                "自动 TPE probe 不足，沿用上一版或默认工作点",
                run_id=run.id,
                chunk_version=chunk_version,
                reason="insufficient_evaluation",
            )
            return base_theta, {
                "status": "skipped",
                "auto_tpe_status": "failed_or_skipped",
                "reason": "insufficient_evaluation",
                "run_id": run.id,
                "selected_theta_hash": run.selected_theta_hash,
                "chunk_version": chunk_version,
            }
        failure_reason = (
            "no_trial_matches_active_runtime_profile"
            if all_valid_trials
            else "no_trial_passed_hard_gate"
        )
        transition_tpe_run(
            db,
            run,
            "failed",
            details={
                "phase": "trial_selection",
                "blocking_reasons": [failure_reason],
                "retry_boundary": "next_chunk_version_graph_build",
            },
            completed_at=_now(),
            failure_code=failure_reason,
            blocking_reasons_json=[failure_reason],
            last_error=(
                "自动 TPE 没有 trial 匹配当前 Runtime/Gate Profile；沿用上一版或默认工作点"
                if all_valid_trials
                else "自动 TPE 没有 trial 通过 hard gate；沿用上一版或默认工作点"
            ),
        )
        fallback_theta_hash = _theta_hash(base_theta)
        emit_ingestion_log(
            batch_id,
            "auto_tpe_failed",
            run.last_error,
            run_id=run.id,
            chunk_version=chunk_version,
            blocking_reasons=run.blocking_reasons_json,
            fallback_theta_hash=fallback_theta_hash,
            auto_tpe_status="failed_or_skipped",
        )
        return dict(base_theta), {
            "status": "failed",
            "auto_tpe_status": "failed_or_skipped",
            "reason": failure_reason,
            "run_id": run.id,
            "chunk_version": chunk_version,
            "fallback_theta_hash": fallback_theta_hash,
            "blocking_reasons": list(run.blocking_reasons_json or []),
        }

    best = max(valid_trials, key=lambda trial: float(trial.objective_score or 0.0))
    transition_tpe_run(
        db,
        run,
        TPE_SELECTED_PENDING_STATUS,
        details={
            "phase": "best_theta_selected",
            "promotion_status": "awaiting_active_graph_transaction",
            "retry_boundary": "graph_transaction_reconciliation",
        },
        best_trial_id=best.id,
        best_objective_score=best.objective_score,
        selected_theta_json=dict(best.sampled_theta_json or {}),
        selected_theta_hash=best.theta_hash,
        selected_edge_distance_protocol=best.edge_distance_protocol,
        selected_edge_distance_protocol_hash=best.edge_distance_protocol_hash,
        selected_edge_type_calibration_protocol=best.edge_type_calibration_protocol,
        selected_edge_type_calibration_protocol_hash=best.edge_type_calibration_protocol_hash,
        selected_calibration_params_json=dict(best.calibration_params_json or {}),
        selected_calibration_params_hash=best.calibration_params_hash,
        selected_edge_type_calibration_config_hash=best.edge_type_calibration_config_hash,
        probe_set_hash=best.probe_set_hash,
        hard_gate_json=dict(best.hard_gate_json or {}),
        objective_components_json=dict(best.objective_components_json or {}),
        runtime_settings_hash=best.runtime_settings_hash,
        selected_graph_runtime_settings_hash=selection_graph_runtime_hash,
        selected_gate_profile_hash=best.gate_profile_hash,
        selected_gate_profile_json=dict(best.gate_profile_json or {}),
        completed_at=None,
        failure_code=None,
        blocking_reasons_json=[],
        last_error=None,
        diagnostics_json={
            **dict(run.diagnostics_json or {}),
            "promotion_lease_expires_at": promotion_lease_expires_at(),
            "promotion_status": "awaiting_active_graph_transaction",
            "selected_runtime_settings_hash": best.runtime_settings_hash,
            "selected_graph_runtime_settings_hash": selection_graph_runtime_hash,
            "selected_gate_profile_hash": best.gate_profile_hash,
        },
    )
    register_tpe_graph_promotion(
        db,
        run_id=run.id,
        knowledge_base_id=run.knowledge_base_id,
        best_trial_id=best.id,
        selected_theta_hash=str(run.selected_theta_hash),
        selected_graph_runtime_settings_hash=selection_graph_runtime_hash,
    )
    emit_ingestion_log(
        batch_id,
        "auto_tpe_best_theta_selected",
        "自动 TPE 已选择最佳底层关系图工作点",
        run_id=run.id,
        trial_id=best.id,
        objective_score=best.objective_score,
        theta_hash=best.theta_hash,
        chunk_version=chunk_version,
    )
    return dict(run.selected_theta_json or base_theta), {
        "status": run.status,
        "run_id": run.id,
        "best_trial_id": best.id,
        "selected_theta_hash": run.selected_theta_hash,
        "objective_score": run.best_objective_score,
        "chunk_version": chunk_version,
        "protocol_hash": run.protocol_hash,
        "tpe_search_space_hash": run.tpe_search_space_hash,
        "edge_distance_protocol": best.edge_distance_protocol,
        "edge_distance_protocol_hash": best.edge_distance_protocol_hash,
        "edge_type_calibration_protocol": best.edge_type_calibration_protocol,
        "edge_type_calibration_protocol_hash": best.edge_type_calibration_protocol_hash,
        "calibration_params": dict(best.calibration_params_json or {}),
        "calibration_params_hash": best.calibration_params_hash,
        "edge_type_calibration_config_hash": best.edge_type_calibration_config_hash,
        # The active graph binds only rebuild-required settings.  The TPE
        # optimizer envelope remains hot and is audited separately on the run
        # and trial so changing trial budget does not make an existing graph
        # stale.
        "runtime_settings_hash": selection_graph_runtime_hash,
        "optimizer_runtime_settings_hash": run.runtime_settings_hash,
        "gate_profile_hash": run.selected_gate_profile_hash,
        "gate_profile": dict(run.selected_gate_profile_json or {}),
    }


def mark_auto_tpe_relation_state(db: Session, run_id: str, relation_state_id: str) -> None:
    bind_tpe_graph_promotion_state(
        db,
        run_id=run_id,
        relation_state_id=relation_state_id,
    )


def summarize_auto_tpe_trial(trial: AutoTpeTrial) -> dict[str, Any]:
    sampled_theta = dict(trial.sampled_theta_json or {})
    return {
        "trial_id": trial.id,
        "run_id": trial.run_id,
        "knowledge_base_id": trial.knowledge_base_id,
        "build_batch_id": trial.build_batch_id,
        "chunk_scope_hash": trial.chunk_scope_hash,
        "embedding_model": trial.embedding_model,
        "embedding_text_version": trial.embedding_text_version,
        "trial_index": trial.trial_index,
        "status": trial.status,
        "sampled_theta_json": sampled_theta if not preflight_theta(sampled_theta) else None,
        "theta_hash": trial.theta_hash,
        "tpe_search_space_hash": trial.tpe_search_space_hash,
        "edge_distance_protocol": trial.edge_distance_protocol,
        "edge_distance_protocol_hash": trial.edge_distance_protocol_hash,
        "edge_type_calibration_protocol": trial.edge_type_calibration_protocol,
        "edge_type_calibration_protocol_hash": trial.edge_type_calibration_protocol_hash,
        "calibration_params": trial.calibration_params_json or None,
        "calibration_params_hash": trial.calibration_params_hash,
        "edge_type_calibration_config_hash": trial.edge_type_calibration_config_hash,
        "sampler_state_hash": trial.sampler_state_hash,
        "runtime_settings_hash": trial.runtime_settings_hash,
        "gate_profile_hash": trial.gate_profile_hash,
        "gate_profile": trial.gate_profile_json or {},
        "candidate_adjacency_hash": trial.candidate_adjacency_hash,
        "probe_set_hash": trial.probe_set_hash,
        "objective_score": trial.objective_score,
        "hard_gate": trial.hard_gate_json or {},
        "objective_components": trial.objective_components_json or {},
        "failure_code": trial.failure_code,
        "diagnostics": trial.diagnostics_json or {},
        "started_at": trial.started_at,
        "finished_at": trial.finished_at,
    }


def summarize_auto_tpe_run(db: Session, run: AutoTpeRun) -> dict[str, Any]:
    trials = list(db.scalars(select(AutoTpeTrial).where(AutoTpeTrial.run_id == run.id).order_by(AutoTpeTrial.trial_index.asc())).all())
    selected_theta = dict(run.selected_theta_json or {})
    return {
        "run_id": run.id,
        "knowledge_base_id": run.knowledge_base_id,
        "batch_id": run.batch_id,
        "chunk_relation_graph_state_id": run.chunk_relation_graph_state_id,
        "chunk_version": run.chunk_version,
        "chunk_scope_hash": run.chunk_scope_hash,
        "graph_operating_point_protocol": run.graph_operating_point_protocol,
        "protocol_hash": run.protocol_hash,
        "tpe_search_space_hash": run.tpe_search_space_hash,
        "chat_model": run.chat_model,
        "embedding_model": run.embedding_model,
        "embedding_text_version": run.embedding_text_version,
        "status": run.status,
        "trigger_reason": run.trigger_reason,
        "trial_budget": run.trial_budget,
        "startup_random_trials": run.startup_random_trials,
        "good_quantile_gamma": run.good_quantile_gamma,
        "probe_query_budget": run.probe_query_budget,
        "candidate_pool_size": run.candidate_pool_size,
        "best_trial_id": run.best_trial_id,
        "best_objective_score": run.best_objective_score,
        "selected_theta_hash": run.selected_theta_hash,
        "selected_theta": selected_theta if not preflight_theta(selected_theta) else None,
        "selected_edge_distance_protocol": run.selected_edge_distance_protocol,
        "selected_edge_distance_protocol_hash": run.selected_edge_distance_protocol_hash,
        "selected_edge_type_calibration_protocol": run.selected_edge_type_calibration_protocol,
        "selected_edge_type_calibration_protocol_hash": run.selected_edge_type_calibration_protocol_hash,
        "selected_calibration_params": run.selected_calibration_params_json or None,
        "selected_calibration_params_hash": run.selected_calibration_params_hash,
        "selected_edge_type_calibration_config_hash": run.selected_edge_type_calibration_config_hash,
        "sampler_state_hash": run.sampler_state_hash,
        "probe_set_hash": run.probe_set_hash,
        "hard_gate": run.hard_gate_json or {},
        "objective_components": run.objective_components_json or {},
        "last_error": run.last_error,
        "failure_code": run.failure_code,
        "blocking_reasons": run.blocking_reasons_json or [],
        "runtime_settings_hash": run.runtime_settings_hash,
        "selected_graph_runtime_settings_hash": run.selected_graph_runtime_settings_hash,
        "selected_gate_profile_hash": run.selected_gate_profile_hash,
        "selected_gate_profile": run.selected_gate_profile_json or {},
        "diagnostics": run.diagnostics_json or {},
        "trials": [summarize_auto_tpe_trial(trial) for trial in trials],
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


def latest_auto_tpe_status(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    # GET/status reads must never become an implicit recovery writer.  Orphan
    # reconciliation is an explicit task boundary guarded by durable owner and
    # resource locks; observing status is deliberately side-effect free.
    with db.no_autoflush:
        knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise LookupError("Knowledge base not found")
        run = db.scalar(
            select(AutoTpeRun)
            .where(AutoTpeRun.knowledge_base_id == knowledge_base_id)
            .order_by(AutoTpeRun.created_at.desc())
        )
        return {
            "knowledge_base_id": knowledge_base_id,
            "current_chunk_version": knowledge_base.current_chunk_version or 0,
            "enabled": bool(get_settings().enable_auto_tpe),
            "latest_run": summarize_auto_tpe_run(db, run) if run else None,
        }
