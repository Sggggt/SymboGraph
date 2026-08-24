from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _load_script(name: str):
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            f"test_gray_script_{name}", SCRIPTS_ROOT / f"{name}.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS_ROOT))


RUNTIME_HASH = "5" * 64


def _envelope(**overrides) -> dict:
    from app.services.context_graph import normalize_agent_operating_envelope

    return normalize_agent_operating_envelope(
        {
            "path_distance_green_threshold": 0.5,
            "path_distance_gray_threshold": 1.0,
            "path_distance_hard_threshold": 2.0,
            **overrides,
        }
    )


def _gray_runtime_hash(envelope: dict | None = None) -> str:
    from app.services.context_graph import gray_zone_runtime_settings_hash

    return gray_zone_runtime_settings_hash(envelope or _envelope())


def _traversal_hash(envelope: dict | None = None) -> str:
    from app.services.context_graph import traversal_protocol_hash

    return traversal_protocol_hash(envelope or _envelope())


def _observation_thresholds(envelope: dict | None = None) -> dict:
    frozen = envelope or _envelope()
    return {
        "path_distance_green_threshold": frozen[
            "path_distance_green_threshold"
        ],
        "path_distance_gray_threshold": frozen["path_distance_gray_threshold"],
        "path_distance_hard_threshold": frozen["path_distance_hard_threshold"],
        "gray_zone_rule_protocol_version": frozen[
            "gray_zone_rule_protocol_version"
        ],
    }


def _local_observation(
    *,
    zone: str,
    semantic_uncertain_edge: bool = False,
    crossing_rq_boundary: bool = False,
) -> dict:
    from app.services.chunking import stable_hash

    path_distance = 0.25 if zone == "green" else 0.75
    support_before: list[str] = []
    support_after = ["support-a"]
    return {
        "current_layer": "mid",
        "path_distance": path_distance,
        "distance_zone": zone,
        "covered_facets_before": [],
        "covered_facets_after": ["facet-a"],
        "required_facets": ["facet-a"],
        "candidate_facets": ["facet-a"],
        "evidence_roles_before": [],
        "evidence_roles_after": ["definition"],
        "support_ids_before": support_before,
        "support_ids_after": support_after,
        "support_ids_before_count": len(support_before),
        "support_ids_after_count": len(support_after),
        "support_ids_before_hash": stable_hash(support_before),
        "support_ids_after_hash": stable_hash(support_after),
        "support_id_gain": True,
        "independent_path_contribution_gain": True,
        "path_contribution_key": "6" * 64,
        "support_refs": {"edge_ids": ["edge-a"], "support_ids": support_after},
        "active_edge_support_gate_pass": True,
        "support_backed_to_covered_path": True,
        "validated_entry_semantic_anchor": True,
        "semantic_uncertain_edge": semantic_uncertain_edge,
        "crossing_rq_boundary": crossing_rq_boundary,
        "bridge_or_boundary_reason": [],
        "edge_type": "dense_semantic",
        "supported_raw_span_hit": False,
        "structure_context_available": True,
        "drilldown_eligible": False,
        "rq_membership_diagnostics": {},
        "candidate_chunk_span_summary": {},
        "structure_context_status": {},
        "hard_interrupt_state": {"frontier_expansion_count": 1, "frontier_expansion_budget": 8},
        **_observation_thresholds(),
    }


def _record(
    *,
    zone: str = "gray",
    source: str = "deterministic_local_rule",
    input_hash: str | None = None,
    decision: str | None = None,
    matched_rule: str | None = None,
    model_call_count: int = 0,
    semantic_uncertain_edge: bool | None = None,
    crossing_rq_boundary: bool = False,
) -> dict:
    from app.services.chunking import stable_hash
    from app.services.context_graph import (
        deterministic_gray_zone_decision,
        gray_zone_decision_identity_hash,
        path_distance_partition_audit,
    )

    envelope = _envelope()
    if source == "deterministic_distance_partition" or zone in {"red", "hard_stop"}:
        record = path_distance_partition_audit(
            path_distance=1.5 if zone == "red" else 2.5,
            distance_zone=zone if zone in {"red", "hard_stop"} else "red",
            envelope=envelope,
            path_contribution_key="7" * 64,
            support_refs={"edge_ids": ["edge-a"], "support_ids": ["support-a"]},
            hard_interrupt_state={"frontier_expansion_count": 1, "frontier_expansion_budget": 8},
        )
        record.update(
            {
                "layer": "mid",
                "edge_id": "edge-a",
                "from_node_id": "from-a",
                "to_node_id": "to-a",
                "gray_candidate_reasons": [],
            }
        )
    else:
        semantic = zone == "green" if semantic_uncertain_edge is None else semantic_uncertain_edge
        observation = _local_observation(
            zone=zone,
            semantic_uncertain_edge=semantic,
            crossing_rq_boundary=crossing_rq_boundary,
        )
        record = deterministic_gray_zone_decision(observation)
        record.update(
            {
                "layer": "mid",
                "edge_id": "edge-a",
                "from_node_id": "from-a",
                "to_node_id": "to-a",
                "semantic_uncertain_edge": semantic,
                "crossing_rq_boundary": crossing_rq_boundary,
                "gray_candidate_reasons": (
                    ["distance_gray"]
                    if zone == "gray"
                    else [
                        reason
                        for enabled, reason in (
                            (semantic, "semantic_uncertain"),
                            (crossing_rq_boundary, "crossing_rq_boundary"),
                        )
                        if enabled
                    ]
                ),
            }
        )
    record["decision_source"] = source
    if input_hash is not None:
        record["input_hash"] = input_hash
    if decision is not None:
        record["decision"] = decision
    if matched_rule is not None:
        record["matched_rule"] = matched_rule
    record["model_call_count"] = model_call_count
    record["traversal_protocol_hash"] = _traversal_hash(envelope)
    record["runtime_settings_hash"] = _gray_runtime_hash(envelope)
    record["agent_operating_envelope_hash"] = stable_hash(envelope)
    record["decision_hash"] = gray_zone_decision_identity_hash(
        record,
        traversal_hash=record["traversal_protocol_hash"],
        runtime_settings_hash=record["runtime_settings_hash"],
        operating_envelope_hash=record["agent_operating_envelope_hash"],
    )
    return record


def _trace(
    records: list[dict],
    *,
    trace_id: str = "trace-a",
    runtime_settings_hash: str = RUNTIME_HASH,
    include_summaries: bool = True,
    run_id: str | None = None,
    envelope: dict | None = None,
) -> dict:
    from app.services.chunking import stable_hash
    from app.services.context_graph import gray_zone_decision_identity_hash
    from app.services.context_graph import (
        GRAY_ZONE_QUERY_FACET_PROTOCOL_VERSION,
        deterministic_gray_query_facets_for_search,
    )

    envelope = dict(envelope or _envelope())
    traversal_hash = _traversal_hash(envelope)
    gray_runtime_hash = _gray_runtime_hash(envelope)
    normalized_records: list[dict] = []
    for source_record in records:
        record = dict(source_record)
        record["traversal_protocol_hash"] = traversal_hash
        record["runtime_settings_hash"] = gray_runtime_hash
        record["agent_operating_envelope_hash"] = stable_hash(envelope)
        record["decision_hash"] = gray_zone_decision_identity_hash(
            record,
            traversal_hash=record["traversal_protocol_hash"],
            runtime_settings_hash=record["runtime_settings_hash"],
            operating_envelope_hash=record["agent_operating_envelope_hash"],
        )
        normalized_records.append(record)
    records = normalized_records
    gray_records = [
        record
        for record in records
        if record.get("decision_source") == "deterministic_local_rule"
    ]
    partition_records = [
        record
        for record in records
        if record.get("decision_source") == "deterministic_distance_partition"
    ]
    payload = {
        "trace_id": trace_id,
        "run_id": run_id,
        "query": "alpha",
        "traversal_protocol_hash": traversal_hash,
        "runtime_settings_hash": runtime_settings_hash,
        "agent_operating_envelope_hash": stable_hash(envelope),
        "trace_diagnostics": {
            "agent_operating_envelope": envelope,
            "agent_operating_envelope_hash": stable_hash(envelope),
            "effective_traversal_protocol_hash": traversal_hash,
            "runtime_settings_hash": runtime_settings_hash,
            "gray_zone_runtime_settings_identity_protocol_version": (
                "gray_zone_runtime_settings_identity_v1"
            ),
            "gray_zone_runtime_settings_hash": gray_runtime_hash,
            "gray_zone_query_facet_protocol_version": (
                GRAY_ZONE_QUERY_FACET_PROTOCOL_VERSION
            ),
            "gray_zone_query_facet_hash": stable_hash(
                deterministic_gray_query_facets_for_search("alpha")
            ),
            "gray_zone_external_routing_packet_used": False,
            "gray_zone_request_scoped_budget_in_identity": False,
        },
        "gray_zone_model_call_count": 0,
        "convergence": {
            "gray_zone_rule_evaluation_count": len(gray_records),
            "gray_zone_decision_count": len(gray_records),
            "red_zone_pruned_count": sum(
                record.get("distance_zone") == "red" for record in partition_records
            ),
            "hard_stop_pruned_count": sum(
                record.get("distance_zone") == "hard_stop"
                for record in partition_records
            ),
            "gray_zone_model_call_count": 0,
            "gray_zone_rule_protocol_version": "deterministic_support_progress_v1",
        },
        "steps": [
            {"step_index": 3, "gray_zone_path_decisions": records},
        ],
    }
    if include_summaries:
        payload["gray_zone_path_decisions"] = gray_records
        payload["path_distance_threshold_hits"] = partition_records
    return payload


def test_raw_trace_audit_accepts_complete_gray_red_hard_records_and_explicit_zero():
    module = _load_script("_gray_zone_audit")
    records = [
        _record(),
        _record(
            zone="red",
            source="deterministic_distance_partition",
            decision="red_zone_pruned",
            matched_rule="distance_red_zone",
        ),
        _record(
            zone="hard_stop",
            source="deterministic_distance_partition",
            decision="hard_stop_pruned",
            matched_rule="distance_hard_stop",
        ),
    ]
    records[1]["protocol_version"] = "deterministic_path_distance_partition_v2"
    records[2]["protocol_version"] = "deterministic_path_distance_partition_v2"

    audit = module.audit_gray_zone_traces(
        [_trace(records, trace_id="trace-a"), _trace(records, trace_id="trace-b")],
        require_gray_coverage=True,
    )
    assert audit["pass"] is True
    assert audit["status"] == "pass"
    assert audit["gray_rule_record_count"] == 2
    assert audit["red_partition_record_count"] == 2
    assert audit["hard_stop_partition_record_count"] == 2
    assert audit["explicit_zero_model_call_record_count"] == 6
    assert audit["determinism"]["conflict_count"] == 0
    assert audit["determinism"]["duplicate_key_count"] == 1
    assert audit["formal_replay_coverage"] == 1.0


@pytest.mark.parametrize("predicate", ["semantic_uncertain_edge", "crossing_rq_boundary"])
def test_green_uncertain_or_crossing_record_is_audited_as_gray_local_rule(predicate):
    module = _load_script("_gray_zone_audit")
    record = _record(
        zone="green",
        semantic_uncertain_edge=predicate == "semantic_uncertain_edge",
        crossing_rq_boundary=predicate == "crossing_rq_boundary",
    )

    audit = module.audit_gray_zone_traces(
        [_trace([record], trace_id="trace-a"), _trace([record], trace_id="trace-b")],
        require_gray_coverage=True,
    )

    assert audit["pass"] is True
    assert audit["gray_rule_record_count"] == 2
    assert audit["gray_zone_coverage"] is True


def test_red_partition_cannot_relabel_itself_as_local_rule():
    module = _load_script("_gray_zone_audit")
    record = _record(zone="red", source="deterministic_local_rule")

    audit = module.audit_gray_zone_trace(_trace([record]))

    assert audit["pass"] is False
    assert any(
        violation["type"] == "invalid_decision_source"
        and violation["expected"] == "deterministic_distance_partition"
        for violation in audit["violations"]
    )


def test_plain_green_record_cannot_invoke_local_gray_rule():
    module = _load_script("_gray_zone_audit")
    record = _record(zone="green")
    record["semantic_uncertain_edge"] = False
    record["crossing_rq_boundary"] = False
    record["gray_candidate_reasons"] = []
    record["observation"]["semantic_uncertain_edge"] = False
    record["observation"]["crossing_rq_boundary"] = False

    audit = module.audit_gray_zone_trace(_trace([record]))

    assert audit["pass"] is False
    assert any(
        violation["type"] == "local_rule_outside_gray_candidate"
        for violation in audit["violations"]
    )


def test_raw_trace_audit_marks_missing_persisted_fields_incomplete_without_defaults():
    module = _load_script("_gray_zone_audit")
    record = _record()
    del record["protocol_hash"]
    del record["model_call_count"]

    audit = module.audit_gray_zone_trace(_trace([record]))

    assert audit["pass"] is False
    assert audit["status"] == "incomplete"
    assert audit["incomplete_record_count"] == 1
    missing = audit["incomplete_records"][0]["missing_fields"]
    assert "protocol_hash" in missing
    assert "model_call_count" in missing
    assert audit["explicit_zero_model_call_record_count"] == 0


@pytest.mark.parametrize("missing_field", ["minimum_audit", "support_refs", "hard_interrupt_state"])
def test_raw_trace_audit_rejects_missing_minimum_audit_evidence(missing_field):
    module = _load_script("_gray_zone_audit")
    record = _record()
    del record[missing_field]

    audit = module.audit_gray_zone_trace(_trace([record]))

    assert audit["pass"] is False
    assert audit["status"] == "incomplete"
    assert missing_field in audit["incomplete_records"][0]["missing_fields"]


def test_raw_trace_audit_rejects_partition_with_wrong_rule_or_decision():
    module = _load_script("_gray_zone_audit")
    record = _record(
        zone="red",
        source="deterministic_distance_partition",
        decision="continue_path",
        matched_rule="distance_red_zone",
    )
    record["protocol_version"] = "deterministic_path_distance_partition_v2"

    audit = module.audit_gray_zone_trace(_trace([record]))

    assert audit["pass"] is False
    assert any(item["type"] == "invalid_partition_outcome" for item in audit["violations"])


def test_raw_trace_audit_fails_nonzero_calls_count_mismatch_and_summary_drift():
    module = _load_script("_gray_zone_audit")
    trace = _trace([_record(model_call_count=1)])
    trace["convergence"]["gray_zone_rule_evaluation_count"] = 0
    trace["gray_zone_path_decisions"] = []

    audit = module.audit_gray_zone_trace(trace)

    assert audit["pass"] is False
    assert audit["status"] == "failed"
    violation_types = {item["type"] for item in audit["violations"]}
    assert "nonzero_model_call_count" in violation_types
    assert "count_mismatch" in violation_types
    assert "summary_raw_record_mismatch" in violation_types


def test_raw_trace_audit_fails_same_identity_with_conflicting_decision_or_rule():
    module = _load_script("_gray_zone_audit")
    first = _trace([_record()], trace_id="trace-a")
    second = _trace(
        [
            _record(
                decision="stop_path_irrelevant",
                matched_rule="6_no_progress_stop",
            )
        ],
        trace_id="trace-b",
    )

    audit = module.audit_gray_zone_traces([first, second], require_gray_coverage=True)

    assert audit["pass"] is False
    assert audit["status"] == "failed"
    assert audit["determinism"]["conflict_count"] == 1
    assert audit["determinism"]["conflicts"][0]["key"] == {
        "protocol_hash": first["steps"][0]["gray_zone_path_decisions"][0]["protocol_hash"],
        "runtime_settings_hash": _gray_runtime_hash(),
        "input_hash": first["steps"][0]["gray_zone_path_decisions"][0]["input_hash"],
    }


def test_same_input_is_isolated_by_provider_free_gray_runtime_settings_hash():
    module = _load_script("_gray_zone_audit")
    envelope_a = _envelope()
    envelope_b = _envelope(
        gray_zone_observation_cadence=(
            envelope_a["gray_zone_observation_cadence"] + 1
        )
    )
    traces = [
        _trace([_record()], trace_id="trace-a1", envelope=envelope_a),
        _trace([_record()], trace_id="trace-a2", envelope=envelope_a),
        _trace([_record()], trace_id="trace-b1", envelope=envelope_b),
        _trace([_record()], trace_id="trace-b2", envelope=envelope_b),
    ]

    audit = module.audit_gray_zone_traces(traces, require_gray_coverage=True)

    assert audit["pass"] is True
    assert audit["determinism"]["checked_key_count"] == 2
    assert audit["determinism"]["duplicate_key_count"] == 2
    assert audit["determinism"]["conflict_count"] == 0


def test_gray_coverage_gap_is_reported_but_only_required_mode_fails():
    module = _load_script("_gray_zone_audit")
    trace = _trace([])

    ordinary = module.audit_gray_zone_trace(trace, require_gray_coverage=False)
    formal = module.audit_gray_zone_trace(trace, require_gray_coverage=True)

    assert ordinary["status"] == "coverage_gap"
    assert ordinary["pass"] is True
    assert ordinary["gray_zone_coverage"] is False
    assert formal["status"] == "coverage_gap"
    assert formal["pass"] is False


def test_formal_replay_rejects_single_observation_even_when_gray_event_exists():
    module = _load_script("_gray_zone_audit")
    trace = _trace([_record()])

    ordinary = module.audit_gray_zone_trace(trace, require_gray_coverage=False)
    formal = module.audit_gray_zone_trace(trace, require_gray_coverage=True)

    assert ordinary["pass"] is True
    assert ordinary["status"] == "pass"
    assert formal["pass"] is False
    assert formal["status"] == "coverage_gap"
    assert formal["gray_zone_coverage"] is True
    assert formal["determinism"]["duplicate_key_count"] == 0
    assert formal["determinism"]["formal_replay_coverage"] == 0.0
    assert formal["determinism"]["uncovered_key_count"] == 1


def test_formal_replay_accepts_compact_replay_card_in_two_distinct_trace_or_run_ids():
    module = _load_script("_gray_zone_audit")
    compact = _record()
    del compact["observation"]
    compact["observation_compacted"] = True

    audit = module.audit_gray_zone_traces(
        [
            _trace([compact], trace_id="trace-a", run_id="run-a"),
            _trace([compact], trace_id="trace-b", run_id="run-b"),
        ],
        require_gray_coverage=True,
    )

    assert audit["pass"] is True
    assert audit["status"] == "pass"
    assert audit["determinism"]["duplicate_key_count"] == 1
    assert audit["determinism"]["formal_replay_covered_key_count"] == 1
    assert audit["determinism"]["uncovered_key_count"] == 0


def test_compact_replay_card_tamper_is_a_hard_input_hash_failure():
    module = _load_script("_gray_zone_audit")
    compact = _record()
    del compact["observation"]
    compact["observation_compacted"] = True
    compact["minimum_audit"]["support_ids_after_count"] += 1

    audit = module.audit_gray_zone_trace(_trace([compact]))

    assert audit["pass"] is False
    assert audit["status"] == "failed"
    assert any(
        item["type"] == "nonreplayable_input_hash"
        for item in audit["violations"]
    )


def test_raw_duplicate_decision_event_is_a_hard_failure_not_replay_coverage():
    module = _load_script("_gray_zone_audit")
    record = _record()
    trace = _trace([record, record])

    audit = module.audit_gray_zone_trace(trace, require_gray_coverage=True)

    assert audit["pass"] is False
    assert audit["status"] == "failed"
    assert audit["raw_duplicate_event_count"] == 1
    assert audit["determinism"]["raw_duplicate_event_count"] == 1
    assert any(
        violation["type"] == "duplicate_gray_zone_decision_event"
        for violation in audit["violations"]
    )


@pytest.mark.parametrize(
    ("field", "violation_type"),
    [
        ("protocol_hash", "nonreplayable_protocol_hash"),
        ("threshold_hash", "nonreplayable_threshold_hash"),
        ("input_hash", "nonreplayable_input_hash"),
        ("decision_hash", "nonreplayable_decision_hash"),
    ],
)
def test_hash_replay_rejects_canonical_but_forged_record_hash(field, violation_type):
    module = _load_script("_gray_zone_audit")
    trace = _trace([_record()], include_summaries=False)
    trace["steps"][0]["gray_zone_path_decisions"][0][field] = "a" * 64

    audit = module.audit_gray_zone_trace(trace)

    assert audit["pass"] is False
    assert audit["status"] == "failed"
    assert any(item["type"] == violation_type for item in audit["violations"])


def test_record_trace_identity_and_canonical_sha_are_strictly_validated():
    module = _load_script("_gray_zone_audit")
    trace = _trace([_record()], include_summaries=False)
    record = trace["steps"][0]["gray_zone_path_decisions"][0]
    record["runtime_settings_hash"] = "A" * 64

    audit = module.audit_gray_zone_trace(trace)

    violation_types = {item["type"] for item in audit["violations"]}
    assert audit["pass"] is False
    assert "noncanonical_record_sha256" in violation_types
    assert "record_trace_identity_mismatch" in violation_types


def test_public_frozen_envelope_schema_rejects_provider_fields_and_nonfinite_values():
    from pydantic import ValidationError

    from app.schemas import RetrievalTraceDiagnostics

    diagnostics = _trace([])["trace_diagnostics"]
    validated = RetrievalTraceDiagnostics.model_validate(diagnostics)
    assert validated.gray_zone_runtime_settings_hash == _gray_runtime_hash()
    assert validated.agent_operating_envelope.gray_zone_model_call_budget == 0

    provider_contaminated = {
        **diagnostics,
        "agent_operating_envelope": {
            **diagnostics["agent_operating_envelope"],
            "chat_base_url": "https://provider.invalid/v1",
        },
    }
    with pytest.raises(ValidationError):
        RetrievalTraceDiagnostics.model_validate(provider_contaminated)

    for field in (
        "max_cycle_reward_per_path",
        "cycle_reward_distance_threshold",
        "path_distance_green_threshold",
        "path_distance_gray_threshold",
        "path_distance_hard_threshold",
    ):
        nonfinite = {
            **diagnostics,
            "agent_operating_envelope": {
                **diagnostics["agent_operating_envelope"],
                field: float("inf"),
            },
        }
        with pytest.raises(ValidationError):
            RetrievalTraceDiagnostics.model_validate(nonfinite)


def test_offline_audit_recomputes_frozen_envelope_and_gray_runtime_hash_fail_closed():
    module = _load_script("_gray_zone_audit")

    tampered_envelope = _trace([_record()])
    tampered_envelope["trace_diagnostics"]["agent_operating_envelope"][
        "candidate_pool_dedupe_budget"
    ] += 1
    envelope_audit = module.audit_gray_zone_trace(tampered_envelope)
    assert envelope_audit["pass"] is False
    envelope_violations = {
        item["type"] for item in envelope_audit["violations"]
    }
    assert "operating_envelope_hash_mismatch" in envelope_violations
    assert "traversal_protocol_hash_mismatch" in envelope_violations
    assert "gray_zone_runtime_settings_hash_mismatch" not in envelope_violations

    tampered_gray_runtime_envelope = _trace([_record()])
    tampered_gray_runtime_envelope["trace_diagnostics"][
        "agent_operating_envelope"
    ]["gray_zone_observation_cadence"] += 1
    gray_runtime_envelope_audit = module.audit_gray_zone_trace(
        tampered_gray_runtime_envelope
    )
    assert any(
        item["type"] == "gray_zone_runtime_settings_hash_mismatch"
        for item in gray_runtime_envelope_audit["violations"]
    )

    tampered_gray_hash = _trace([_record()])
    tampered_gray_hash["trace_diagnostics"][
        "gray_zone_runtime_settings_hash"
    ] = "a" * 64
    gray_hash_audit = module.audit_gray_zone_trace(tampered_gray_hash)
    assert gray_hash_audit["pass"] is False
    assert any(
        item["type"] == "gray_zone_runtime_settings_hash_mismatch"
        for item in gray_hash_audit["violations"]
    )


def test_offline_audit_requires_dedicated_gray_hash_and_rejects_broad_hash_reuse():
    module = _load_script("_gray_zone_audit")

    missing = _trace([_record()])
    del missing["trace_diagnostics"]["gray_zone_runtime_settings_hash"]
    missing_audit = module.audit_gray_zone_trace(missing)
    assert missing_audit["pass"] is False
    assert missing_audit["status"] == "incomplete"
    assert (
        "trace_diagnostics.gray_zone_runtime_settings_hash"
        in missing_audit["incomplete_traces"][0]["missing_fields"]
    )

    conflated = _trace([_record()])
    gray_hash = conflated["trace_diagnostics"][
        "gray_zone_runtime_settings_hash"
    ]
    conflated["runtime_settings_hash"] = gray_hash
    conflated["trace_diagnostics"]["runtime_settings_hash"] = gray_hash
    conflated_audit = module.audit_gray_zone_trace(conflated)
    assert conflated_audit["pass"] is False
    assert any(
        item["type"] == "broad_runtime_hash_reuses_gray_identity"
        for item in conflated_audit["violations"]
    )


def test_script_required_fields_are_exactly_the_application_contract():
    module = _load_script("_gray_zone_audit")

    assert set(module.COMMON_REQUIRED_FIELDS) == {
        name
        for name, field in module.APPLICATION_CONTRACT["schema"].model_fields.items()
        if field.is_required()
    }
    audit = module.audit_gray_zone_trace(_trace([]))
    assert audit["application_contract"]["status"] == "pass"


def test_agent_evaluation_fetches_and_audits_each_returned_retrieval_trace(
    monkeypatch, capsys
):
    module = _load_script("evaluate_agent_trace")
    fetched_paths: list[str] = []
    posted_timeouts: list[int] = []
    fetched_timeouts: list[int] = []
    captured_report: dict = {}
    response_ids = iter([("run-a", "trace-a"), ("run-b", "trace-b")])

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            base_url="http://example.invalid/api",
            knowledge_base_id="kb-1",
            knowledge_base_name="sample",
            question=["audited question", "audited question"],
            top_k=4,
            execute=True,
            require_gray_coverage=True,
        ),
    )
    monkeypatch.setattr(module, "resolve_kb_id", lambda _args: ("kb-1", "sample"))
    def fake_post_json(*_args, **_kwargs):
        posted_timeouts.append(_kwargs["timeout"])
        run_id, trace_id = next(response_ids)
        return {
            "run_id": run_id,
            "context_package_id": "package-1",
            "retrieval_trace_id": trace_id,
            "answer": "Grounded answer.",
            "citations": [{"chunk_id": "chunk-1"}],
            "trace": [{"node": node} for node in module.REQUIRED_TRACE_NODES],
            "model_audit": {"citation_verification_pass_rate": 1.0},
            "degraded_mode": False,
        }

    monkeypatch.setattr(module, "post_json", fake_post_json)

    def fake_get_json(_base_url, path, timeout=300):
        fetched_paths.append(path)
        fetched_timeouts.append(timeout)
        trace_id = path.split("/")[2]
        return _trace([_record()], trace_id=trace_id)

    def fake_write_report(_name, payload):
        captured_report.update(payload)
        return Path("output/agent-gray-audit.json")

    monkeypatch.setattr(module, "get_json", fake_get_json)
    monkeypatch.setattr(
        module,
        "persisted_retrieval_snapshot",
        lambda trace_id: _trace([_record()], trace_id=trace_id),
    )
    monkeypatch.setattr(
        module,
        "persisted_typed_action_facts",
        lambda run_id: {"run_id": run_id, "plans": []},
    )
    monkeypatch.setattr(
        module,
        "persisted_agent_quality_snapshot",
        lambda run_id: {
            "run": {"id": run_id},
            "trace_events": [],
        },
    )
    # This test isolates the persisted gray replay path. The numerical
    # retrieval/Agent gates have adversarial fixtures in test_quality_gate_scripts.
    monkeypatch.setattr(
        module,
        "audit_retrieval_quality",
        lambda *_args, **_kwargs: {"pass": True, "findings": []},
    )
    monkeypatch.setattr(
        module,
        "audit_agent_quality",
        lambda *_args, **_kwargs: {"pass": True, "findings": []},
    )
    monkeypatch.setattr(module, "write_report", fake_write_report)

    module.main()

    assert fetched_paths == [
        "/retrieval-traces/trace-a/graph-steps",
        "/retrieval-traces/trace-b/graph-steps",
    ]
    assert posted_timeouts == [900, 900]
    assert fetched_timeouts == [900, 900]
    assert captured_report["pass"] is True
    assert captured_report["gray_zone_trace_audit"]["gray_zone_coverage"] is True
    assert captured_report["gray_zone_trace_audit"]["determinism"]["status"] == "pass"
    assert captured_report["gray_zone_trace_audit"]["formal_replay_coverage"] == 1.0
    assert json.loads(capsys.readouterr().out)["pass"] is True


def test_layered_evaluation_replays_provider_free_gray_identity(monkeypatch):
    module = _load_script("evaluate_layered_retrieval")
    trace = SimpleNamespace(
        id="trace-layered",
        query="audited query",
        retrieval_mode="layered_context_graph",
        created_at=None,
        result_chunk_ids_json=["chunk-1"],
    )
    raw_trace = _trace([_record()], trace_id="trace-layered")
    captured_gray_audit: dict = {}

    monkeypatch.setattr(module, "_raw_trace_payload", lambda *_args: raw_trace)
    monkeypatch.setattr(
        module,
        "_document_coverage",
        lambda *_args: {
            "chunk_count": 1,
            "document_count": 1,
            "document_ids": ["document-1"],
            "documents": [
                {
                    "document_id": "document-1",
                    "title": "Document 1",
                    "chunk_ids": ["chunk-1"],
                }
            ],
            "missing_chunk_ids": [],
            "missing_document_ids": [],
            "orphan_chunk_ids": [],
            "complete": True,
        },
    )

    def fake_quality(_snapshot, *, gray_zone_audit):
        captured_gray_audit.update(gray_zone_audit)
        return {"pass": bool(gray_zone_audit["pass"]), "findings": []}

    monkeypatch.setattr(module, "audit_retrieval_quality", fake_quality)

    row, replayed = module._trace_row(None, trace, source="persisted_replay")

    assert replayed is raw_trace
    assert row["quality_gate"]["pass"] is True
    assert captured_gray_audit["pass"] is True
    assert captured_gray_audit["per_trace"][0][
        "broad_runtime_settings_hash"
    ] == RUNTIME_HASH
    assert captured_gray_audit["per_trace"][0][
        "gray_zone_runtime_settings_hash"
    ] == _gray_runtime_hash()
    assert (
        captured_gray_audit["per_trace"][0]["broad_runtime_settings_hash"]
        != captured_gray_audit["per_trace"][0][
            "gray_zone_runtime_settings_hash"
        ]
    )


def test_docker_smoke_gray_coverage_gate_is_explicit_cli_option(monkeypatch):
    module = _load_script("docker_smoke")

    monkeypatch.setattr(sys, "argv", ["docker_smoke.py"])
    assert module.parse_args().require_gray_coverage is False
    assert module.parse_args().require_relation_edge_coverage is False
    assert module.parse_args().require_rq_diagnostic_coverage is False
    monkeypatch.setattr(sys, "argv", ["docker_smoke.py", "--require-gray-coverage"])
    assert module.parse_args().require_gray_coverage is True


def _smoke_trace(records: list[dict], trace_id: str) -> dict:
    trace = _trace(records, trace_id=trace_id)
    rq_protocol = "query_rq_primary_residual_mid_dense_v5"
    rq_protocol_hash = "a" * 64
    trace["trace_diagnostics"] = {
        **(trace.get("trace_diagnostics") or {}),
        "query_rq_seed_audit": {
            "protocol_version": rq_protocol,
            "protocol_hash": rq_protocol_hash,
            "model_call_count": 0,
            "gray_zone_decision_authority": False,
            "is_evidence": False,
        },
    }
    trace["candidate_pools"] = {
        **(trace.get("candidate_pools") or {}),
        "rq_membership_entries": {
            "ranking_protocol_version": rq_protocol,
            "ranking_protocol_hash": rq_protocol_hash,
            "rq_seed_cards": {
                "chunk-1": {
                    "model_call_count": 0,
                    "gray_zone_decision_authority": False,
                    "is_evidence": False,
                    "input_hash": "b" * 64,
                    "card_hash": "c" * 64,
                }
            },
        },
    }
    trace["steps"] = [
        {
            "step_index": 1,
            "layer": "chunk",
            "action": "select_seeds_from_mid_rq_membership",
            "input": {"query_rq_path": [1, 2, 3]},
            "output": {"candidate_rq": [{"chunk_id": "chunk-1"}]},
            "gray_zone_path_decisions": [],
        },
        {
            "step_index": 2,
            "layer": "chunk",
            "action": "walk_graph_frontier",
            "input": {},
            "output": {},
            "gray_zone_path_decisions": records,
        },
    ]
    return trace


def test_docker_smoke_fetches_search_and_qa_traces_and_reports_ordinary_coverage_gap(
    monkeypatch, capsys
):
    module = _load_script("docker_smoke")
    fetched_trace_paths: list[str] = []
    captured_report: dict = {}

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def request_json(self, method, path, payload=None, params=None, timeout=None):
            if path == "/health":
                return {"status": "ok"}
            if path == "/knowledge_bases":
                return [{"id": "kb-1", "active_chunk_count": 1}]
            if path == "/knowledge_bases/kb-1/context-graph/stats":
                return {"counts": {"active_chunks": 1, "chunk_relation_edges": 1}}
            if path.startswith("/knowledge_bases/kb-1/graph/"):
                graph_type = path.rsplit("/", 1)[-1]
                response = {"graph_type": graph_type, "counts": {}, "nodes": [], "edges": []}
                if graph_type == "chunk-relation":
                    response["nodes"] = [
                        {
                            "contract_kind": "chunk_node",
                            "id": "chunk-1",
                            "metadata": {"rq_path": [1, 2, 3]},
                        },
                        {
                            "contract_kind": "rq_prefix_node",
                            "id": "rq-1",
                            "category": "rq_prefix",
                        },
                    ]
                    response["edges"] = [
                        {
                            "contract_kind": "chunk_relation_edge",
                            "id": "edge-1",
                            "type": "dense_semantic",
                        },
                        {
                            "contract_kind": "rq_membership_edge",
                            "id": "membership",
                            "category": "rq_membership",
                        },
                        {
                            "contract_kind": "rq_diagnostic_edge",
                            "id": "rq-diagnostic",
                            "type": "rq_prefix_pair_diagnostic",
                            "metadata": {
                                "diagnostic_only": True,
                                "active_relation_edge": False,
                            },
                        },
                    ]
                return response
            if path == "/search":
                return {
                    "results": [{"chunk_id": "chunk-1", "metadata": {"rq": {"distance": 0.1}}}],
                    "retrieval_trace_id": "search-trace",
                    "context_package_id": "search-package",
                    "model_audit": {
                        "retrieval_trace_id": "search-trace",
                        "context_package_id": "search-package",
                        "query_rq_path": [1, 2, 3],
                        "retrieval_cache": {
                            "status": "miss",
                            "gray_zone_input_modified": False,
                            "gray_zone_model_call_count": 0,
                        },
                    },
                }
            if path == "/qa":
                return {
                    "answer": "Grounded answer.",
                    "citations": [{"chunk_id": "chunk-1"}],
                    "context_package_id": "package-1",
                    "retrieval_trace_id": "qa-trace",
                    "model_audit": {"citation_verification_pass_rate": 1.0},
                }
            if path.startswith("/retrieval-traces/"):
                fetched_trace_paths.append(path)
                trace_id = path.split("/")[2]
                return _smoke_trace([], trace_id)
            if path == "/context-packages/search-package":
                return {
                    "retrieval_trace_id": "search-trace",
                    "citation_spans": [
                        {"chunk_id": "chunk-1", "char_start": 0, "char_end": 8}
                    ],
                    "token_count": 8,
                    "token_budget": 128,
                }
            if path == "/context-packages/package-1":
                return {"citation_spans": [{"chunk_id": "chunk-1", "char_start": 0, "char_end": 8}]}
            raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(module, "SmokeClient", FakeClient)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            base_url="http://example.invalid/api",
            knowledge_base_id="kb-1",
            query="audited query",
            execute=True,
            wait_batch_id=None,
            wait_timeout_seconds=30,
            request_timeout_seconds=2.0,
            qa_timeout_seconds=3.0,
            require_gray_coverage=False,
            require_relation_edge_coverage=False,
            require_rq_diagnostic_coverage=False,
        ),
    )

    def fake_write_report(payload):
        captured_report.update(payload)
        return Path("output/docker-gray-audit.json")

    monkeypatch.setattr(module, "write_report", fake_write_report)

    assert module.main() == 0
    assert fetched_trace_paths == [
        "/retrieval-traces/search-trace/graph-steps",
        "/retrieval-traces/qa-trace/graph-steps",
    ]
    assert captured_report["pass"] is True
    assert captured_report["gray_zone_trace_audit"]["status"] == "coverage_gap"
    assert captured_report["gray_zone_trace_audit"]["gray_zone_coverage"] is False
    assert json.loads(capsys.readouterr().out)["pass"] is True


def test_docker_smoke_formal_gray_coverage_gate_fails_empty_coverage(monkeypatch):
    module = _load_script("docker_smoke")
    ordinary_audit = _load_script("_gray_zone_audit").audit_gray_zone_traces(
        [_smoke_trace([], "search-trace"), _smoke_trace([], "qa-trace")],
        require_gray_coverage=True,
    )

    assert ordinary_audit["status"] == "coverage_gap"
    assert ordinary_audit["pass"] is False
    assert ordinary_audit["require_gray_coverage"] is True
