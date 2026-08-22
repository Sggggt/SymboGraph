from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

from app.core.config import (
    EDGE_DISTANCE_PROTOCOL_DEFAULT,
    EDGE_PROJECTION_PROTOCOL_DEFAULT,
    EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT,
    RQ_MEMBERSHIP_PROTOCOL_DEFAULT,
    get_settings,
)


GRAPH_PROTOCOL_RUNTIME_IDENTITY_VERSION = "graph_protocol_runtime_identity_v1"
MULTI_PATH_CONTRIBUTION_PROTOCOL_VERSION = "multi_path_contribution_union_v2"
MULTI_PATH_CONTRIBUTION_ID_CANONICALIZATION = (
    "json_utf8_sort_keys_compact_v1"
)
ACTIVE_GRAPH_PROTOCOLS = {
    "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_DEFAULT,
    "rq_membership_protocol": RQ_MEMBERSHIP_PROTOCOL_DEFAULT,
    "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_DEFAULT,
    "edge_type_calibration_protocol": EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT,
}
RQ_MEMBERSHIP_PARAMETER_NAMES = (
    "rq_membership_temperature",
    "rq_membership_top_m",
    "rq_membership_probability_threshold",
)


class GraphProtocolAdmissionError(RuntimeError):
    """Runtime settings do not match a locally implemented graph protocol."""


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_value(
    record: Mapping[str, Any] | object,
    field: str,
    default: Any = None,
) -> Any:
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


def _record_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _canonical_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = (
        value
        if isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        else [value]
    )
    return sorted(
        {
            str(item)
            for item in values
            if item is not None and str(item)
        }
    )


def retrieval_path_contribution_identity(
    record: Mapping[str, Any] | object,
) -> dict[str, Any]:
    """Return the only public identity packet for one physical retrieval path."""

    return {
        "protocol_version": MULTI_PATH_CONTRIBUTION_PROTOCOL_VERSION,
        "canonicalization": (
            MULTI_PATH_CONTRIBUTION_ID_CANONICALIZATION
        ),
        "layer": str(_record_value(record, "layer") or ""),
        "node_id": str(_record_value(record, "node_id") or ""),
        "parent_layer": (
            str(_record_value(record, "parent_layer"))
            if _record_value(record, "parent_layer") is not None
            else None
        ),
        "parent_node_id": (
            str(_record_value(record, "parent_node_id"))
            if _record_value(record, "parent_node_id") is not None
            else None
        ),
        "origin_parent_layer": (
            str(_record_value(record, "origin_parent_layer"))
            if _record_value(record, "origin_parent_layer")
            is not None
            else None
        ),
        "origin_parent_node_id": (
            str(_record_value(record, "origin_parent_node_id"))
            if _record_value(record, "origin_parent_node_id")
            is not None
            else None
        ),
        "root_node_id": str(
            _record_value(record, "root_node_id") or ""
        ),
        "path": [
            str(value)
            for value in (_record_value(record, "path", []) or [])
        ],
        "path_edge_ids": [
            str(value)
            for value in (
                _record_value(record, "path_edge_ids", []) or []
            )
        ],
        "path_edge_types": [
            str(value)
            for value in (
                _record_value(record, "path_edge_types", []) or []
            )
        ],
    }


def retrieval_path_contribution_id(
    record: Mapping[str, Any] | object,
) -> str:
    """Replay a contribution id without importing traversal implementation."""

    return _canonical_hash(
        retrieval_path_contribution_identity(record)
    )


def retrieval_path_support_ids(
    record: Mapping[str, Any] | object,
) -> list[str]:
    """Replay the support-id union of one public path contribution."""

    support_refs = _record_mapping(
        _record_value(record, "support_refs", {})
    )
    values: set[str] = set(
        _canonical_string_list(
            _record_value(record, "support_chunk_ids", [])
        )
    )
    non_identity_fields = {
        "edge_type",
        "edge_types",
        "chunk_candidate_source",
        "chunk_candidate_sources",
        "entry_strength",
        "entry_distance",
        "entry_strengths",
        "raw_entry_strengths",
    }
    for key, value in support_refs.items():
        if key in non_identity_fields:
            continue
        source = (
            value
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            else [value]
        )
        for item in source:
            if (
                isinstance(item, (str, int))
                and not isinstance(item, bool)
                and str(item)
            ):
                values.add(str(item))
    return sorted(values)


def retrieval_node_contribution_facts(
    paths: Sequence[Mapping[str, Any] | object],
) -> dict[str, Any]:
    """Replay every aggregate exposed by a node contribution summary."""

    parent_node_ids = _canonical_string_list(
        [
            _record_value(path, "parent_node_id")
            for path in paths
            if _record_value(path, "parent_node_id") is not None
        ]
    )
    path_edge_types = _canonical_string_list(
        [
            edge_type
            for path in paths
            for edge_type in (
                _record_value(path, "path_edge_types", []) or []
            )
        ]
    )
    covered_facets = _canonical_string_list(
        [
            facet
            for path in paths
            for facet in (
                _record_value(path, "covered_facets", []) or []
            )
        ]
    )
    evidence_roles = _canonical_string_list(
        [
            role
            for path in paths
            for role in (
                _record_value(path, "evidence_roles", []) or []
            )
        ]
    )
    support_chunk_union = _canonical_string_list(
        [
            chunk_id
            for path in paths
            for chunk_id in (
                _record_value(path, "support_chunk_ids", []) or []
            )
        ]
    )
    support_id_union = _canonical_string_list(
        [
            support_id
            for path in paths
            for support_id in retrieval_path_support_ids(path)
        ]
    )
    distances = [
        float(_record_value(path, "distance_so_far", 0.0) or 0.0)
        for path in paths
    ]
    rewards = [
        float(_record_value(path, "reward_so_far", 0.0) or 0.0)
        for path in paths
    ]
    if any(
        not math.isfinite(value) or value < 0.0
        for value in [*distances, *rewards]
    ):
        raise ValueError(
            "retrieval contribution distance/reward must be finite "
            "and nonnegative"
        )
    return {
        "node_visit_count": len(paths),
        "distinct_parent_count": len(parent_node_ids),
        "distinct_path_count": len(paths),
        "distinct_edge_type_count": len(path_edge_types),
        "parent_node_ids": parent_node_ids,
        "path_edge_types": path_edge_types,
        "covered_facets": covered_facets,
        "evidence_roles": evidence_roles,
        "support_id_union": support_id_union,
        "support_chunk_union": support_chunk_union,
        "cycle_convergence_score": round(sum(rewards), 6),
        "best_distance": round(min(distances), 6)
        if distances
        else 0.0,
        "best_reward": round(max(rewards), 6)
        if rewards
        else 0.0,
    }


def validate_active_graph_protocol_settings(
    settings: Any | None = None,
    *,
    required_protocols: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless selected protocols map to local deterministic implementations."""

    settings = settings or get_settings()
    required = set(required_protocols or ACTIVE_GRAPH_PROTOCOLS)
    unknown = required.difference(ACTIVE_GRAPH_PROTOCOLS)
    if unknown:
        raise GraphProtocolAdmissionError(
            "Unknown graph protocol admission fields: " + ", ".join(sorted(unknown))
        )

    selected_protocols: dict[str, str] = {}
    for setting_key in sorted(required):
        expected = ACTIVE_GRAPH_PROTOCOLS[setting_key]
        selected = getattr(settings, setting_key, None)
        if selected != expected:
            raise GraphProtocolAdmissionError(
                f"{setting_key}={selected!r} does not match the local implementation "
                f"{expected!r}; stage and promote only a locally supported candidate."
            )
        selected_protocols[setting_key] = expected

    temperature = getattr(settings, "rq_membership_temperature", None)
    top_m = getattr(settings, "rq_membership_top_m", None)
    probability_threshold = getattr(
        settings,
        "rq_membership_probability_threshold",
        None,
    )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0.0 < float(temperature) <= 10.0
    ):
        raise GraphProtocolAdmissionError(
            "rq_membership_temperature must be finite and in (0, 10]"
        )
    if type(top_m) is not int or not 1 <= top_m <= 6:
        raise GraphProtocolAdmissionError("rq_membership_top_m must be in 1..6")
    if (
        isinstance(probability_threshold, bool)
        or not isinstance(probability_threshold, (int, float))
        or not math.isfinite(float(probability_threshold))
        or not 0.0 <= float(probability_threshold) <= 1.0
    ):
        raise GraphProtocolAdmissionError(
            "rq_membership_probability_threshold must be finite and in [0, 1]"
        )

    identity = {
        "protocol_version": GRAPH_PROTOCOL_RUNTIME_IDENTITY_VERSION,
        "protocols": selected_protocols,
        "rq_membership_parameters": {
            "temperature": float(temperature),
            "top_m": top_m,
            "probability_threshold": float(probability_threshold),
            "primary_code_forced": True,
            "renormalize_after_sparsification": False,
            "membership_score_floor": None,
        },
        "dynamic_language_inputs": {
            "llm": False,
            "prompt": False,
            "free_expression": False,
        },
    }
    return {
        **identity,
        "identity_hash": _canonical_hash(identity),
    }


def graph_protocol_runtime_identity_hash(settings: Any | None = None) -> str:
    return str(validate_active_graph_protocol_settings(settings)["identity_hash"])
