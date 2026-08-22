from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LOCAL_RULE_SOURCE = "deterministic_local_rule"
DISTANCE_PARTITION_SOURCE = "deterministic_distance_partition"
COMMON_REQUIRED_FIELDS = (
    "path_distance",
    "distance_zone",
    "decision",
    "protocol_version",
    "protocol_hash",
    "input_hash",
    "threshold_hash",
    "traversal_protocol_hash",
    "runtime_settings_hash",
    "agent_operating_envelope_hash",
    "decision_hash",
    "matched_rule",
    "minimum_audit",
    "hard_interrupt_state",
    "model_call_count",
    "decision_source",
    "support_refs",
)
LOCAL_RULE_PROTOCOL_VERSION = "deterministic_support_progress_v1"
DISTANCE_PARTITION_PROTOCOL_VERSION = "deterministic_path_distance_partition_v2"
LOCAL_RULE_OUTCOMES = {
    "1_support_or_drift_stop": "stop_path_irrelevant",
    "2_structure_closure": "request_structure_closure",
    "3_supported_bridge": "follow_as_bridge",
    "4_supported_drilldown": "drill_down_layer",
    "5_support_progress": "continue_path",
    "6_no_progress_stop": "stop_path_irrelevant",
}
PARTITION_OUTCOMES = {
    "red_partition": ("distance_red_zone", "red_zone_pruned"),
    "hard_stop_partition": ("distance_hard_stop", "hard_stop_pruned"),
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS = (
    "protocol_hash",
    "input_hash",
    "threshold_hash",
    "traversal_protocol_hash",
    "runtime_settings_hash",
    "agent_operating_envelope_hash",
    "decision_hash",
)


def _load_application_contract() -> tuple[dict[str, Any] | None, str | None]:
    """Load the authoritative application replay contract in host and container layouts."""

    api_root = Path(__file__).resolve().parents[1] / "apps" / "api"
    inserted = False
    if api_root.is_dir() and str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))
        inserted = True
    try:
        from app.schemas import RetrievalGrayZoneDecision
        from app.services.chunking import stable_hash
        from app.services.context_graph import (
            GRAY_ZONE_QUERY_FACET_PROTOCOL_VERSION,
            GRAY_ZONE_RUNTIME_SETTINGS_FIELDS,
            GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION,
            deterministic_gray_query_facets_for_search,
            gray_zone_decision_identity_hash,
            gray_zone_local_rule_outcome,
            gray_zone_minimum_replay_card,
            gray_zone_runtime_settings_hash,
            gray_zone_runtime_settings_snapshot,
            gray_zone_rule_protocol_hash,
            path_distance_partition_protocol_hash,
            path_distance_threshold_hash,
            traversal_protocol_hash,
            validate_gray_zone_decision_records_for_persistence,
        )
        from app.schemas import RetrievalAgentOperatingEnvelope

        return (
            {
                "schema": RetrievalGrayZoneDecision,
                "envelope_schema": RetrievalAgentOperatingEnvelope,
                "stable_hash": stable_hash,
                "decision_identity_hash": gray_zone_decision_identity_hash,
                "local_rule_outcome": gray_zone_local_rule_outcome,
                "minimum_replay_card": gray_zone_minimum_replay_card,
                "gray_runtime_fields": tuple(GRAY_ZONE_RUNTIME_SETTINGS_FIELDS),
                "gray_runtime_identity_protocol": (
                    GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION
                ),
                "gray_query_facet_protocol": (
                    GRAY_ZONE_QUERY_FACET_PROTOCOL_VERSION
                ),
                "gray_query_facets": deterministic_gray_query_facets_for_search,
                "gray_runtime_hash": gray_zone_runtime_settings_hash,
                "gray_runtime_snapshot": gray_zone_runtime_settings_snapshot,
                "local_protocol_hash": gray_zone_rule_protocol_hash,
                "partition_protocol_hash": path_distance_partition_protocol_hash,
                "threshold_hash": path_distance_threshold_hash,
                "traversal_hash": traversal_protocol_hash,
                "validate_records": validate_gray_zone_decision_records_for_persistence,
            },
            None,
        )
    except Exception as exc:  # pragma: no cover - exercised by deployment smoke, not unit tests.
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if inserted:
            sys.path.remove(str(api_root))


APPLICATION_CONTRACT, APPLICATION_CONTRACT_IMPORT_ERROR = _load_application_contract()


def _application_required_fields() -> tuple[str, ...] | None:
    if APPLICATION_CONTRACT is None:
        return None
    schema = APPLICATION_CONTRACT["schema"]
    return tuple(name for name, field in schema.model_fields.items() if field.is_required())


def _nonempty(value: Any) -> bool:
    return value is not None and value != ""


def _trace_hash(trace: Mapping[str, Any], field: str) -> Any:
    diagnostics = trace.get("trace_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    cache_components = diagnostics.get("cache_key_components")
    if not isinstance(cache_components, Mapping):
        cache_components = {}
    for candidate in (
        trace.get(field),
        diagnostics.get(field),
        (
            diagnostics.get("effective_traversal_protocol_hash")
            if field == "traversal_protocol_hash"
            else None
        ),
        cache_components.get(field),
    ):
        if _nonempty(candidate):
            return candidate
    return None


def _trace_diagnostics(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = trace.get("trace_diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _trace_operating_envelope(trace: Mapping[str, Any]) -> dict[str, Any] | None:
    diagnostics = _trace_diagnostics(trace)
    for candidate in (
        trace.get("agent_operating_envelope"),
        diagnostics.get("agent_operating_envelope"),
        diagnostics.get("effective_agent_operating_envelope"),
    ):
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return None


def _trace_gray_runtime_hash(trace: Mapping[str, Any]) -> Any:
    diagnostics = _trace_diagnostics(trace)
    return diagnostics.get("gray_zone_runtime_settings_hash")


def _trace_gray_runtime_identity_protocol(trace: Mapping[str, Any]) -> Any:
    diagnostics = _trace_diagnostics(trace)
    return diagnostics.get(
        "gray_zone_runtime_settings_identity_protocol_version"
    )


def _trace_execution_identity(trace: Mapping[str, Any], trace_label: str) -> str:
    diagnostics = _trace_diagnostics(trace)
    run_id = trace.get("run_id") or diagnostics.get("run_id")
    trace_id = trace.get("trace_id") or trace_label
    return json.dumps(
        {"trace_id": str(trace_id), "run_id": str(run_id) if _nonempty(run_id) else None},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _is_gray_candidate(record: Mapping[str, Any]) -> bool:
    zone = str(record.get("distance_zone") or "")
    observation = record.get("observation")
    if not isinstance(observation, Mapping):
        observation = {}
    return (
        zone == "gray"
        or bool(record.get("semantic_uncertain_edge"))
        or bool(record.get("crossing_rq_boundary"))
        or bool(observation.get("semantic_uncertain_edge"))
        or bool(observation.get("crossing_rq_boundary"))
    )


def _record_kind(record: Mapping[str, Any]) -> str:
    zone = str(record.get("distance_zone") or "")
    source = str(record.get("decision_source") or "")
    # The deterministic distance partition is evaluated before gray-candidate
    # predicates. A red/hard record cannot relabel itself as a local-rule record.
    if zone == "red":
        return "red_partition"
    if zone == "hard_stop":
        return "hard_stop_partition"
    if _is_gray_candidate(record) or source == LOCAL_RULE_SOURCE:
        return "gray_local_rule"
    return "unknown"


def _record_signature(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _record_event_signature(record: Mapping[str, Any]) -> str:
    decision_hash = record.get("decision_hash")
    return f"decision:{decision_hash}" if _nonempty(decision_hash) else f"raw:{_record_signature(record)}"


def _expected_source(kind: str) -> str | None:
    if kind == "gray_local_rule":
        return LOCAL_RULE_SOURCE
    if kind in {"red_partition", "hard_stop_partition"}:
        return DISTANCE_PARTITION_SOURCE
    return None


def audit_gray_zone_traces(
    traces: Sequence[Mapping[str, Any]],
    *,
    require_gray_coverage: bool = False,
) -> dict[str, Any]:
    """Audit persisted raw step events and independently replay every available hash.

    ``require_gray_coverage`` is the formal replay gate. It is intentionally stronger
    than merely observing a gray event: every local-rule input identity must be seen in
    at least two distinct trace/run identities, with a replayable expanded input in both.
    """

    incomplete_traces: list[dict[str, Any]] = []
    incomplete_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    count_checks: list[dict[str, Any]] = []
    hash_replay_checks: list[dict[str, Any]] = []
    audited_records: list[dict[str, Any]] = []
    per_trace: list[dict[str, Any]] = []
    raw_duplicate_event_count = 0

    application_required_fields = _application_required_fields()
    contract_fields_match = (
        application_required_fields is not None
        and set(application_required_fields) == set(COMMON_REQUIRED_FIELDS)
    )
    if APPLICATION_CONTRACT is None:
        incomplete_traces.append(
            {
                "trace_id": "<audit-contract>",
                "missing_fields": ["application_gray_zone_replay_contract"],
                "error": APPLICATION_CONTRACT_IMPORT_ERROR,
            }
        )
    elif not contract_fields_match:
        violations.append(
            {
                "trace_id": "<audit-contract>",
                "type": "application_contract_required_field_drift",
                "script_fields": sorted(COMMON_REQUIRED_FIELDS),
                "application_fields": sorted(application_required_fields or ()),
            }
        )

    for trace_index, trace in enumerate(traces):
        trace_id = trace.get("trace_id")
        trace_label = str(trace_id or f"trace[{trace_index}]")
        execution_identity = _trace_execution_identity(trace, trace_label)
        trace_missing: list[str] = []
        if not _nonempty(trace_id):
            trace_missing.append("trace_id")

        steps = trace.get("steps")
        if not isinstance(steps, list):
            trace_missing.append("steps")
            steps = []
        convergence = trace.get("convergence")
        if not isinstance(convergence, Mapping):
            trace_missing.append("convergence")
            convergence = {}

        trace_hashes = {
            field: _trace_hash(trace, field)
            for field in (
                "traversal_protocol_hash",
                "runtime_settings_hash",
                "agent_operating_envelope_hash",
            )
        }
        gray_runtime_hash = _trace_gray_runtime_hash(trace)
        gray_runtime_identity_protocol = _trace_gray_runtime_identity_protocol(
            trace
        )
        diagnostics = _trace_diagnostics(trace)
        gray_query_facet_protocol = diagnostics.get(
            "gray_zone_query_facet_protocol_version"
        )
        gray_query_facet_hash = diagnostics.get(
            "gray_zone_query_facet_hash"
        )
        trace_identity_values = {
            **trace_hashes,
            "gray_zone_runtime_settings_hash": gray_runtime_hash,
        }
        for field, value in trace_identity_values.items():
            if not _nonempty(value):
                trace_missing.append(
                    f"trace_diagnostics.{field}"
                    if field == "gray_zone_runtime_settings_hash"
                    else field
                )
            elif not _canonical_sha256(value):
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "noncanonical_trace_sha256",
                        "field": field,
                        "actual": value,
                    }
                )
        if not _nonempty(gray_runtime_identity_protocol):
            trace_missing.append(
                "trace_diagnostics.gray_zone_runtime_settings_identity_protocol_version"
            )
        if not _nonempty(gray_query_facet_protocol):
            trace_missing.append(
                "trace_diagnostics.gray_zone_query_facet_protocol_version"
            )
        if not _nonempty(gray_query_facet_hash):
            trace_missing.append(
                "trace_diagnostics.gray_zone_query_facet_hash"
            )
        elif not _canonical_sha256(gray_query_facet_hash):
            violations.append(
                {
                    "trace_id": trace_label,
                    "type": "noncanonical_gray_query_facet_hash",
                    "actual": gray_query_facet_hash,
                }
            )
        for field in (
            "gray_zone_external_routing_packet_used",
            "gray_zone_request_scoped_budget_in_identity",
        ):
            if field not in diagnostics:
                trace_missing.append(f"trace_diagnostics.{field}")
            elif diagnostics.get(field) is not False:
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "gray_zone_external_authority_enabled",
                        "field": field,
                        "actual": diagnostics.get(field),
                    }
                )
        if APPLICATION_CONTRACT is not None and _nonempty(
            gray_query_facet_protocol
        ):
            expected_gray_query_protocol = APPLICATION_CONTRACT[
                "gray_query_facet_protocol"
            ]
            if gray_query_facet_protocol != expected_gray_query_protocol:
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "gray_query_facet_protocol_mismatch",
                        "expected": expected_gray_query_protocol,
                        "actual": gray_query_facet_protocol,
                    }
                )
            expected_gray_query_hash = APPLICATION_CONTRACT["stable_hash"](
                APPLICATION_CONTRACT["gray_query_facets"](
                    str(trace.get("query") or "")
                )
            )
            if (
                _nonempty(gray_query_facet_hash)
                and gray_query_facet_hash != expected_gray_query_hash
            ):
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "gray_query_facet_hash_mismatch",
                        "expected": expected_gray_query_hash,
                        "actual": gray_query_facet_hash,
                    }
                )

        operating_envelope = _trace_operating_envelope(trace)
        if operating_envelope is None:
            trace_missing.append("trace_diagnostics.agent_operating_envelope")
        elif APPLICATION_CONTRACT is not None:
            try:
                frozen_envelope = APPLICATION_CONTRACT[
                    "envelope_schema"
                ].model_validate(operating_envelope).model_dump(mode="json")
            except Exception as exc:
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "invalid_frozen_operating_envelope",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                operating_envelope = None
            else:
                operating_envelope = frozen_envelope
                expected_envelope_hash = APPLICATION_CONTRACT["stable_hash"](
                    operating_envelope
                )
                if (
                    _nonempty(trace_hashes["agent_operating_envelope_hash"])
                    and trace_hashes["agent_operating_envelope_hash"]
                    != expected_envelope_hash
                ):
                    violations.append(
                        {
                            "trace_id": trace_label,
                            "type": "operating_envelope_hash_mismatch",
                            "expected": expected_envelope_hash,
                            "actual": trace_hashes[
                                "agent_operating_envelope_hash"
                            ],
                        }
                    )
                expected_traversal_hash = APPLICATION_CONTRACT["traversal_hash"](
                    operating_envelope
                )
                if (
                    _nonempty(trace_hashes["traversal_protocol_hash"])
                    and trace_hashes["traversal_protocol_hash"]
                    != expected_traversal_hash
                ):
                    violations.append(
                        {
                            "trace_id": trace_label,
                            "type": "traversal_protocol_hash_mismatch",
                            "expected": expected_traversal_hash,
                            "actual": trace_hashes["traversal_protocol_hash"],
                        }
                    )
                expected_gray_runtime_hash = APPLICATION_CONTRACT[
                    "gray_runtime_hash"
                ](operating_envelope)
                if (
                    _nonempty(gray_runtime_hash)
                    and gray_runtime_hash != expected_gray_runtime_hash
                ):
                    violations.append(
                        {
                            "trace_id": trace_label,
                            "type": "gray_zone_runtime_settings_hash_mismatch",
                            "expected": expected_gray_runtime_hash,
                            "actual": gray_runtime_hash,
                        }
                    )
                expected_gray_runtime_protocol = APPLICATION_CONTRACT[
                    "gray_runtime_identity_protocol"
                ]
                if (
                    _nonempty(gray_runtime_identity_protocol)
                    and gray_runtime_identity_protocol
                    != expected_gray_runtime_protocol
                ):
                    violations.append(
                        {
                            "trace_id": trace_label,
                            "type": "gray_zone_runtime_identity_protocol_mismatch",
                            "expected": expected_gray_runtime_protocol,
                            "actual": gray_runtime_identity_protocol,
                        }
                    )
                gray_runtime_snapshot = APPLICATION_CONTRACT[
                    "gray_runtime_snapshot"
                ](operating_envelope)
                if (
                    set(gray_runtime_snapshot.get("settings") or {})
                    != set(APPLICATION_CONTRACT["gray_runtime_fields"])
                    or gray_runtime_snapshot.get(
                        "provider_configuration_included"
                    )
                    is not False
                    or gray_runtime_snapshot.get("model_call_budget") != 0
                ):
                    violations.append(
                        {
                            "trace_id": trace_label,
                            "type": "provider_free_gray_runtime_contract_drift",
                        }
                    )
                broad_runtime_hash = trace_hashes["runtime_settings_hash"]
                if (
                    _nonempty(broad_runtime_hash)
                    and _nonempty(gray_runtime_hash)
                    and broad_runtime_hash == gray_runtime_hash
                ):
                    violations.append(
                        {
                            "trace_id": trace_label,
                            "type": "broad_runtime_hash_reuses_gray_identity",
                        }
                    )

        persisted_protocol = convergence.get("gray_zone_rule_protocol_version")
        if not _nonempty(persisted_protocol):
            trace_missing.append("convergence.gray_zone_rule_protocol_version")
        elif persisted_protocol != LOCAL_RULE_PROTOCOL_VERSION:
            violations.append(
                {
                    "trace_id": trace_label,
                    "type": "invalid_trace_local_rule_protocol",
                    "expected": LOCAL_RULE_PROTOCOL_VERSION,
                    "actual": persisted_protocol,
                }
            )

        raw_records: list[dict[str, Any]] = []
        event_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for step_position, step in enumerate(steps):
            if not isinstance(step, Mapping):
                incomplete_records.append(
                    {
                        "trace_id": trace_label,
                        "step_index": step_position,
                        "record_index": None,
                        "kind": "unknown",
                        "missing_fields": ["step_object"],
                    }
                )
                continue
            decisions = step.get("gray_zone_path_decisions")
            if decisions is None:
                decisions = []
            if not isinstance(decisions, list):
                incomplete_records.append(
                    {
                        "trace_id": trace_label,
                        "step_index": step.get("step_index", step_position),
                        "record_index": None,
                        "kind": "unknown",
                        "missing_fields": ["gray_zone_path_decisions"],
                    }
                )
                continue
            for record_index, record in enumerate(decisions):
                location = {
                    "trace_id": trace_label,
                    "step_index": step.get("step_index", step_position),
                    "record_index": record_index,
                }
                if not isinstance(record, Mapping):
                    incomplete_records.append(
                        {**location, "kind": "unknown", "missing_fields": ["record_object"]}
                    )
                    continue

                value = dict(record)
                kind = _record_kind(value)
                raw_records.append(value)
                event_locations[_record_event_signature(value)].append(location)
                missing_fields = [
                    field for field in COMMON_REQUIRED_FIELDS if not _nonempty(value.get(field))
                ]
                if "model_call_count" in value and type(value.get("model_call_count")) is not int:
                    missing_fields.append("model_call_count:int")
                for field in ("hard_interrupt_state", "minimum_audit", "support_refs"):
                    if field in value and not isinstance(value.get(field), Mapping):
                        missing_fields.append(f"{field}:object")
                    elif isinstance(value.get(field), Mapping) and not value.get(field):
                        missing_fields.append(f"{field}:nonempty")
                if missing_fields:
                    incomplete_records.append(
                        {
                            **location,
                            "kind": kind,
                            "distance_zone": value.get("distance_zone"),
                            "missing_fields": sorted(set(missing_fields)),
                        }
                    )

                for field in HASH_FIELDS:
                    if _nonempty(value.get(field)) and not _canonical_sha256(value.get(field)):
                        violations.append(
                            {
                                **location,
                                "type": "noncanonical_record_sha256",
                                "field": field,
                                "actual": value.get(field),
                            }
                        )
                record_trace_identities = {
                    "traversal_protocol_hash": trace_hashes[
                        "traversal_protocol_hash"
                    ],
                    "runtime_settings_hash": gray_runtime_hash,
                    "agent_operating_envelope_hash": trace_hashes[
                        "agent_operating_envelope_hash"
                    ],
                }
                identity_mismatches = {
                    field: {"trace": trace_value, "record": value.get(field)}
                    for field, trace_value in record_trace_identities.items()
                    if _nonempty(trace_value)
                    and _nonempty(value.get(field))
                    and value.get(field) != trace_value
                }
                if identity_mismatches:
                    violations.append(
                        {
                            **location,
                            "type": "record_trace_identity_mismatch",
                            "mismatches": identity_mismatches,
                        }
                    )

                expected_source = _expected_source(kind)
                if expected_source is None:
                    violations.append(
                        {
                            **location,
                            "type": "unclassified_raw_decision_record",
                            "distance_zone": value.get("distance_zone"),
                            "decision_source": value.get("decision_source"),
                        }
                    )
                elif _nonempty(value.get("decision_source")) and value.get("decision_source") != expected_source:
                    violations.append(
                        {
                            **location,
                            "type": "invalid_decision_source",
                            "expected": expected_source,
                            "actual": value.get("decision_source"),
                        }
                    )
                if kind == "gray_local_rule" and not _is_gray_candidate(value):
                    violations.append(
                        {
                            **location,
                            "type": "local_rule_outside_gray_candidate",
                            "distance_zone": value.get("distance_zone"),
                        }
                    )
                if kind == "gray_local_rule":
                    matched_rule = str(value.get("matched_rule") or "")
                    expected_decision = LOCAL_RULE_OUTCOMES.get(matched_rule)
                    if value.get("protocol_version") != LOCAL_RULE_PROTOCOL_VERSION:
                        violations.append(
                            {
                                **location,
                                "type": "invalid_local_rule_protocol",
                                "expected": LOCAL_RULE_PROTOCOL_VERSION,
                                "actual": value.get("protocol_version"),
                            }
                        )
                    if expected_decision is None or value.get("decision") != expected_decision:
                        violations.append(
                            {
                                **location,
                                "type": "invalid_local_rule_outcome",
                                "matched_rule": matched_rule,
                                "expected_decision": expected_decision,
                                "actual_decision": value.get("decision"),
                            }
                        )
                elif kind in PARTITION_OUTCOMES:
                    expected_rule, expected_decision = PARTITION_OUTCOMES[kind]
                    if value.get("protocol_version") != DISTANCE_PARTITION_PROTOCOL_VERSION:
                        violations.append(
                            {
                                **location,
                                "type": "invalid_partition_protocol",
                                "expected": DISTANCE_PARTITION_PROTOCOL_VERSION,
                                "actual": value.get("protocol_version"),
                            }
                        )
                    if value.get("matched_rule") != expected_rule or value.get("decision") != expected_decision:
                        violations.append(
                            {
                                **location,
                                "type": "invalid_partition_outcome",
                                "expected_rule": expected_rule,
                                "expected_decision": expected_decision,
                                "actual_rule": value.get("matched_rule"),
                                "actual_decision": value.get("decision"),
                            }
                        )
                if "model_call_count" in value and value.get("model_call_count") != 0:
                    violations.append(
                        {
                            **location,
                            "type": "nonzero_model_call_count",
                            "actual": value.get("model_call_count"),
                        }
                    )

                replay = {
                    **location,
                    "kind": kind,
                    "protocol_hash": None,
                    "threshold_hash": None,
                    "input_hash": None,
                    "decision_hash": None,
                    "application_contract": False,
                }
                input_replayable = False
                contract_valid = False
                can_replay = (
                    APPLICATION_CONTRACT is not None
                    and contract_fields_match
                    and not missing_fields
                    and operating_envelope is not None
                    and all(_nonempty(value.get(field)) for field in HASH_FIELDS)
                    and all(_nonempty(trace_hashes[field]) for field in trace_hashes)
                    and _nonempty(gray_runtime_hash)
                    and _nonempty(gray_runtime_identity_protocol)
                )
                if can_replay:
                    try:
                        expected_protocol_hash = (
                            APPLICATION_CONTRACT["local_protocol_hash"](value.get("protocol_version"))
                            if kind == "gray_local_rule"
                            else APPLICATION_CONTRACT["partition_protocol_hash"]()
                        )
                        replay["protocol_hash"] = value.get("protocol_hash") == expected_protocol_hash
                        expected_threshold_hash = APPLICATION_CONTRACT["threshold_hash"](operating_envelope)
                        replay["threshold_hash"] = value.get("threshold_hash") == expected_threshold_hash
                        minimum_audit = dict(value.get("minimum_audit") or {})
                        replay["input_hash"] = (
                            value.get("input_hash")
                            == APPLICATION_CONTRACT["stable_hash"](minimum_audit)
                        )
                        input_replayable = True
                        observation = value.get("observation")
                        if kind == "gray_local_rule":
                            if isinstance(observation, Mapping):
                                replay["input_hash"] = bool(
                                    replay["input_hash"]
                                    and APPLICATION_CONTRACT["minimum_replay_card"](
                                        dict(observation),
                                        dict(value.get("predicates") or {}),
                                    )
                                    == minimum_audit
                                )
                            expected_rule, expected_decision = APPLICATION_CONTRACT[
                                "local_rule_outcome"
                            ](dict(value.get("predicates") or {}))
                            replay["input_hash"] = bool(
                                replay["input_hash"]
                                and value.get("matched_rule") == expected_rule
                                and value.get("decision") == expected_decision
                            )
                        expected_decision_hash = APPLICATION_CONTRACT["decision_identity_hash"](
                            value,
                            traversal_hash=str(trace_hashes["traversal_protocol_hash"]),
                            runtime_settings_hash=str(gray_runtime_hash),
                            operating_envelope_hash=str(trace_hashes["agent_operating_envelope_hash"]),
                        )
                        replay["decision_hash"] = value.get("decision_hash") == expected_decision_hash
                        for replay_field in ("protocol_hash", "threshold_hash", "input_hash", "decision_hash"):
                            if replay[replay_field] is False:
                                violations.append(
                                    {**location, "type": f"nonreplayable_{replay_field}"}
                                )
                        APPLICATION_CONTRACT["validate_records"](
                            [value],
                            traversal_hash=str(trace_hashes["traversal_protocol_hash"]),
                            runtime_settings_hash=str(gray_runtime_hash),
                            operating_envelope_hash=str(trace_hashes["agent_operating_envelope_hash"]),
                            operating_envelope=operating_envelope,
                        )
                        replay["application_contract"] = True
                        contract_valid = True
                    except Exception as exc:
                        violations.append(
                            {
                                **location,
                                "type": "application_contract_replay_failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                hash_replay_checks.append(replay)
                audited_records.append(
                    {
                        **location,
                        "kind": kind,
                        "record": value,
                        "execution_identity": execution_identity,
                        "input_replayable": input_replayable,
                        "contract_valid": contract_valid,
                    }
                )

        duplicate_groups = {
            signature: locations
            for signature, locations in event_locations.items()
            if len(locations) > 1
        }
        for signature, locations in sorted(duplicate_groups.items()):
            raw_duplicate_event_count += len(locations) - 1
            violations.append(
                {
                    "trace_id": trace_label,
                    "type": "duplicate_gray_zone_decision_event",
                    "event_signature": signature,
                    "locations": locations,
                }
            )

        counts = Counter(_record_kind(record) for record in raw_records)
        expected_counts = {
            "gray_zone_decision_count": counts["gray_local_rule"],
            "gray_zone_rule_evaluation_count": counts["gray_local_rule"],
            "red_zone_pruned_count": counts["red_partition"],
            "hard_stop_pruned_count": counts["hard_stop_partition"],
            "gray_zone_model_call_count": 0,
        }
        for name, expected in expected_counts.items():
            if name not in convergence:
                trace_missing.append(f"convergence.{name}")
                continue
            actual = convergence.get(name)
            passed = type(actual) is int and actual == expected
            count_checks.append(
                {
                    "trace_id": trace_label,
                    "counter": name,
                    "expected": expected,
                    "actual": actual,
                    "pass": passed,
                }
            )
            if not passed:
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "count_mismatch",
                        "counter": name,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        if "gray_zone_model_call_count" in trace:
            top_level_count = trace.get("gray_zone_model_call_count")
            if type(top_level_count) is not int:
                trace_missing.append("gray_zone_model_call_count:int")
            elif top_level_count != 0:
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "nonzero_top_level_model_call_count",
                        "actual": top_level_count,
                    }
                )

        summary_pairs = (
            ("gray_zone_path_decisions", "gray_local_rule"),
            ("path_distance_threshold_hits", "distance_partition"),
        )
        for field, summary_kind in summary_pairs:
            if field not in trace:
                continue
            summary_records = trace.get(field)
            if not isinstance(summary_records, list):
                trace_missing.append(field)
                continue
            source_records = (
                [record for record in raw_records if _record_kind(record) == "gray_local_rule"]
                if summary_kind == "gray_local_rule"
                else [
                    record
                    for record in raw_records
                    if _record_kind(record) in {"red_partition", "hard_stop_partition"}
                ]
            )
            summary_signatures = Counter(
                _record_event_signature(record)
                for record in summary_records
                if isinstance(record, Mapping)
            )
            source_signatures = Counter(_record_event_signature(record) for record in source_records)
            if summary_signatures != source_signatures:
                violations.append(
                    {
                        "trace_id": trace_label,
                        "type": "summary_raw_record_mismatch",
                        "field": field,
                        "summary_count": len(summary_records),
                        "raw_count": len(source_records),
                    }
                )

        if trace_missing:
            incomplete_traces.append(
                {"trace_id": trace_label, "missing_fields": sorted(set(trace_missing))}
            )
        per_trace.append(
            {
                "trace_id": trace_label,
                "execution_identity": execution_identity,
                "broad_runtime_settings_hash": trace_hashes[
                    "runtime_settings_hash"
                ],
                "gray_zone_runtime_settings_hash": gray_runtime_hash,
                "raw_record_count": len(raw_records),
                "gray_rule_record_count": counts["gray_local_rule"],
                "red_partition_record_count": counts["red_partition"],
                "hard_stop_partition_record_count": counts["hard_stop_partition"],
                "raw_duplicate_event_count": sum(len(items) - 1 for items in duplicate_groups.values()),
            }
        )

    outcomes_by_key: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    locations_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    execution_ids_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    replayable_execution_ids_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for item in audited_records:
        if item["kind"] != "gray_local_rule":
            continue
        record = item["record"]
        key_values = (
            record.get("protocol_hash"),
            record.get("runtime_settings_hash"),
            record.get("input_hash"),
        )
        if not all(_nonempty(value) for value in key_values):
            continue
        key = tuple(str(value) for value in key_values)
        outcome = (str(record.get("decision") or ""), str(record.get("matched_rule") or ""))
        outcomes_by_key[key].add(outcome)
        execution_ids_by_key[key].add(item["execution_identity"])
        if item["input_replayable"]:
            replayable_execution_ids_by_key[key].add(item["execution_identity"])
        locations_by_key[key].append(
            {
                "trace_id": item["trace_id"],
                "step_index": item["step_index"],
                "record_index": item["record_index"],
                "execution_identity": item["execution_identity"],
                "input_replayable": item["input_replayable"],
            }
        )
    conflicts = [
        {
            "key": {
                "protocol_hash": key[0],
                "runtime_settings_hash": key[1],
                "input_hash": key[2],
            },
            "outcomes": [
                {"decision": decision, "matched_rule": matched_rule}
                for decision, matched_rule in sorted(outcomes)
            ],
            "locations": locations_by_key[key],
        }
        for key, outcomes in outcomes_by_key.items()
        if len(outcomes) > 1
    ]
    repeated_key_count = sum(
        len(execution_ids_by_key[key]) >= 2 for key in outcomes_by_key
    )
    formal_replay_covered_keys = {
        key
        for key in outcomes_by_key
        if len(execution_ids_by_key[key]) >= 2
        and len(replayable_execution_ids_by_key[key]) >= 2
    }
    formal_replay_coverage = (
        len(formal_replay_covered_keys) / len(outcomes_by_key)
        if outcomes_by_key
        else 0.0
    )
    uncovered_keys = [
        {
            "key": {
                "protocol_hash": key[0],
                "runtime_settings_hash": key[1],
                "input_hash": key[2],
            },
            "distinct_trace_run_count": len(execution_ids_by_key[key]),
            "replayable_distinct_trace_run_count": len(replayable_execution_ids_by_key[key]),
            "locations": locations_by_key[key],
        }
        for key in outcomes_by_key
        if key not in formal_replay_covered_keys
    ]

    kind_counts = Counter(item["kind"] for item in audited_records)
    gray_coverage = kind_counts["gray_local_rule"] > 0
    formal_replay_ready = bool(outcomes_by_key) and formal_replay_coverage == 1.0
    has_incomplete = bool(incomplete_traces or incomplete_records)
    has_failure = bool(violations or conflicts)
    formal_coverage_gap = bool(require_gray_coverage) and not formal_replay_ready
    passed = not has_incomplete and not has_failure and not formal_coverage_gap
    if has_failure:
        status = "failed"
    elif has_incomplete:
        status = "incomplete"
    elif formal_coverage_gap or not gray_coverage:
        status = "coverage_gap"
    else:
        status = "pass"

    return {
        "status": status,
        "pass": passed,
        "require_gray_coverage": bool(require_gray_coverage),
        "gray_zone_coverage": gray_coverage,
        "formal_replay_ready": formal_replay_ready,
        "formal_replay_coverage": formal_replay_coverage,
        "trace_count": len(traces),
        "raw_record_count": len(audited_records),
        "raw_duplicate_event_count": raw_duplicate_event_count,
        "gray_rule_record_count": kind_counts["gray_local_rule"],
        "red_partition_record_count": kind_counts["red_partition"],
        "hard_stop_partition_record_count": kind_counts["hard_stop_partition"],
        "explicit_zero_model_call_record_count": sum(
            1
            for item in audited_records
            if type(item["record"].get("model_call_count")) is int
            and item["record"].get("model_call_count") == 0
        ),
        "application_contract": {
            "status": (
                "unavailable"
                if APPLICATION_CONTRACT is None
                else ("pass" if contract_fields_match else "drift")
            ),
            "import_error": APPLICATION_CONTRACT_IMPORT_ERROR,
            "script_required_fields": list(COMMON_REQUIRED_FIELDS),
            "application_required_fields": list(application_required_fields or ()),
        },
        "incomplete_trace_count": len(incomplete_traces),
        "incomplete_record_count": len(incomplete_records),
        "incomplete_traces": incomplete_traces,
        "incomplete_records": incomplete_records,
        "violations": violations,
        "count_checks": count_checks,
        "hash_replay_checks": hash_replay_checks,
        "determinism": {
            "status": (
                "failed"
                if conflicts or raw_duplicate_event_count
                else ("coverage_gap" if formal_coverage_gap else "pass")
            ),
            "checked_key_count": len(outcomes_by_key),
            "duplicate_key_count": repeated_key_count,
            "formal_replay_covered_key_count": len(formal_replay_covered_keys),
            "formal_replay_coverage": formal_replay_coverage,
            "uncovered_key_count": len(uncovered_keys),
            "uncovered_keys": uncovered_keys,
            "raw_duplicate_event_count": raw_duplicate_event_count,
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
        },
        "per_trace": per_trace,
    }


def audit_gray_zone_trace(
    trace: Mapping[str, Any],
    *,
    require_gray_coverage: bool = False,
) -> dict[str, Any]:
    return audit_gray_zone_traces([trace], require_gray_coverage=require_gray_coverage)
