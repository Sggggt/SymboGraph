from __future__ import annotations

import importlib.util
import copy
import hashlib
import io
import json
import sys
from urllib.error import HTTPError
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[3] / "scripts"


def _load_docker_smoke():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location(
        "docker_smoke_under_test",
        SCRIPTS_ROOT / "docker_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_timeouts_cover_sequential_fail_closed_model_calls(
    monkeypatch,
) -> None:
    docker_smoke = _load_docker_smoke()
    monkeypatch.setattr(sys, "argv", ["docker_smoke.py"])

    args = docker_smoke.parse_args()

    assert args.request_timeout_seconds >= 3 * 240
    assert args.qa_timeout_seconds >= 2 * args.request_timeout_seconds


def _valid_graph_payload() -> dict:
    return {
        "graph_type": "chunk-relation",
        "nodes": [
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
        ],
        "edges": [
            {
                "contract_kind": "chunk_relation_edge",
                "id": "edge-1",
                "type": "dense_semantic",
            },
            {
                "contract_kind": "rq_membership_edge",
                "id": "membership-1",
                "type": "rq_prefix_membership",
                "category": "rq_membership",
            },
            {
                "contract_kind": "rq_diagnostic_edge",
                "id": "diagnostic-1",
                "type": "rq_prefix_pair_diagnostic",
                "category": "rq_diagnostic_sibling_overlap",
                "metadata": {
                    "diagnostic_only": True,
                    "active_relation_edge": False,
                },
            },
        ],
    }


def test_docker_smoke_distinguishes_active_relations_from_rq_diagnostics() -> None:
    docker_smoke = _load_docker_smoke()

    counts = docker_smoke.validate_chunk_relation_graph_payload(
        _valid_graph_payload(),
        require_relation_edge_coverage=True,
        require_rq_diagnostic_coverage=True,
    )

    assert counts == {
        "active_relation_edges": 1,
        "rq_chunk_nodes": 1,
        "rq_prefix_nodes": 1,
        "rq_membership_edges": 1,
        "rq_diagnostic_edges": 1,
    }


def test_docker_smoke_default_selection_skips_active_but_graphless_kb() -> None:
    docker_smoke = _load_docker_smoke()
    graphless = {
        "id": "kb-graphless",
        "name": "Graphless",
        "active_chunk_count": 534,
        "context_graph_state_id": None,
        "context_graph_hash": None,
        "stale_reason": "context_graph_state_missing",
    }
    graph_ready = {
        "id": "kb-ready",
        "name": "Ready",
        "active_chunk_count": 273,
        "context_graph_state_id": "context-state-1",
        "context_graph_hash": "a" * 64,
        "stale_reason": None,
    }

    selected, reason = docker_smoke.select_smoke_knowledge_base(
        [graphless, graph_ready],
        requested_id=None,
    )

    assert selected == graph_ready
    assert reason == "first_fresh_graph_ready"


def test_docker_smoke_default_selection_fails_actionably_without_ready_graph() -> None:
    docker_smoke = _load_docker_smoke()

    with pytest.raises(RuntimeError, match="No graph-ready knowledge base"):
        docker_smoke.select_smoke_knowledge_base(
            [
                {
                    "id": "kb-graphless",
                    "active_chunk_count": 1,
                    "context_graph_state_id": None,
                    "context_graph_hash": None,
                    "stale_reason": "context_graph_state_missing",
                }
            ],
            requested_id=None,
        )


def test_smoke_nested_admission_overrides_flat_metadata():
    smoke = _load_docker_smoke()
    items = [{"id": name, "active_chunk_count": 2, "context_graph_state_id": "unit-state",
              "context_graph_hash": "a" * 64, "stale_reason": None} for name in ("stale", "ready")]
    selected, _ = smoke.select_smoke_knowledge_base(items, requested_id=None,
        freshness_loader=lambda kb: {"freshness": {"is_admissible": kb == "ready"}})
    assert selected["id"] == "ready"


def test_smoke_report_drops_private_graph_and_provider_fields():
    smoke = _load_docker_smoke()
    payload = {"script": "docker_smoke", "pass": False, "error_type": "RuntimeError",
        "error": "unit-test-private-error", "checks": [{"name": "context_graph_stats", "pass": True,
        "payload": {"provider": "unit-test-private-provider"}, "counts": {"active_chunks": 12}}]}
    safe = smoke.safe_smoke_report(payload)
    assert "unit-test-private" not in str(safe)
    assert safe["checks"][0]["counts"]["active_chunks"] == 12


def test_docker_smoke_rejects_rq_edge_leaking_into_active_relation_graph() -> None:
    docker_smoke = _load_docker_smoke()
    payload = _valid_graph_payload()
    payload["edges"][0]["type"] = "rq_sibling_overlap"

    with pytest.raises(RuntimeError, match="invalid active edge type"):
        docker_smoke.validate_chunk_relation_graph_payload(payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {"diagnostic_only": False, "active_relation_edge": False},
        {"diagnostic_only": True, "active_relation_edge": True},
        {},
    ],
)
def test_docker_smoke_rejects_diagnostic_edge_without_nonactive_flags(
    metadata: dict,
) -> None:
    docker_smoke = _load_docker_smoke()
    payload = _valid_graph_payload()
    payload["edges"][2]["metadata"] = metadata

    with pytest.raises(RuntimeError, match="explicitly non-active"):
        docker_smoke.validate_chunk_relation_graph_payload(payload)


def test_docker_smoke_coverage_flags_fail_closed() -> None:
    docker_smoke = _load_docker_smoke()
    payload = _valid_graph_payload()
    payload["edges"] = [
        edge
        for edge in payload["edges"]
        if edge["contract_kind"]
        not in {"chunk_relation_edge", "rq_diagnostic_edge"}
    ]

    with pytest.raises(RuntimeError, match="no sampled active relation edge"):
        docker_smoke.validate_chunk_relation_graph_payload(
            payload,
            require_relation_edge_coverage=True,
        )
    with pytest.raises(
        RuntimeError, match="no sampled RQ prefix-pair diagnostic edge"
    ):
        docker_smoke.validate_chunk_relation_graph_payload(
            payload,
            require_rq_diagnostic_coverage=True,
        )


def _rq_seed_trace_fixture() -> dict:
    audit = {
        "protocol_version": "query_rq_primary_residual_mid_dense_v5",
        "protocol_hash": "a" * 64,
        "model_call_count": 0,
        "gray_zone_decision_authority": False,
        "is_evidence": False,
    }
    return {
        "steps": [
            {
                "layer": "chunk",
                "action": "select_seeds_from_mid_rq_membership",
                "input": {
                    "query_rq_path": [1, 2, 3],
                    "query_rq_seed_audit": audit,
                },
                "output": {"accepted_chunk_ids": ["chunk-1"]},
            }
        ],
        "trace_diagnostics": {"query_rq_seed_audit": audit},
        "candidate_pools": {
            "rq_membership_entries": {
                "ranking_protocol_version": audit["protocol_version"],
                "ranking_protocol_hash": audit["protocol_hash"],
                "rq_seed_cards": {
                    "rq-1": {
                        "model_call_count": 0,
                        "gray_zone_decision_authority": False,
                        "is_evidence": False,
                        "input_hash": "b" * 64,
                        "card_hash": "c" * 64,
                    }
                },
            }
        },
    }


def test_docker_smoke_uses_current_typed_rq_seed_contract() -> None:
    docker_smoke = _load_docker_smoke()

    counts = docker_smoke.validate_retrieval_rq_seed_diagnostics(
        _rq_seed_trace_fixture()
    )

    assert counts == {"seed_steps": 1, "rq_seed_cards": 1}


def test_docker_smoke_replays_target_gray_zone_records_without_old_fields() -> None:
    smoke = _load_docker_smoke()
    inputs = {
        "layer": "chunk",
        "distance": 0.3,
        "zone": "gray",
        "support_ids": ["unit-support"],
        "query_anchor_preserved": True,
        "model_call_count": 0,
    }
    canonical = json.dumps(
        inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    audit = smoke.audit_target_gray_zone_traces(
        [
            {
                "retrieval_mode": "intent_execution_retrieval_v1",
                "gray_zone_path_decisions": [
                    {
                        "protocol_version": "deterministic_support_progress_v2",
                        "inputs": inputs,
                        "input_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                        "model_call_count": 0,
                    }
                ],
            }
        ],
        require_gray_coverage=True,
    )
    assert audit["pass"] is True
    assert audit["gray_rule_record_count"] == 1


def test_docker_smoke_rejects_retired_candidate_rq_only_trace() -> None:
    docker_smoke = _load_docker_smoke()
    trace = _rq_seed_trace_fixture()
    trace.pop("trace_diagnostics")
    trace.pop("candidate_pools")
    trace["steps"][0]["input"].pop("query_rq_seed_audit")
    trace["steps"][0]["output"] = {"candidate_rq": [{"id": "rq-1"}]}

    with pytest.raises(RuntimeError, match="typed RQ seed audit"):
        docker_smoke.validate_retrieval_rq_seed_diagnostics(trace)


def test_docker_smoke_accepts_explicit_evidence_insufficiency_without_citations() -> None:
    docker_smoke = _load_docker_smoke()

    audit = docker_smoke.validate_qa_acceptance_payload(
        {
            "answer": "The available evidence is insufficient.",
            "citations": [],
            "model_audit": {
                "grounding_outcome": "insufficient_evidence",
                "insufficient_evidence": True,
                "citation_verification_pass_rate": 0.0,
            },
        }
    )

    assert audit["insufficient_evidence"] is True


def test_docker_smoke_accepts_target_one_shot_source_binding() -> None:
    smoke = _load_docker_smoke()
    audit = smoke.validate_qa_acceptance_payload(
        {
            "answer": "A source-grounded answer.",
            "route": "intent_execution_retrieval_v1",
            "entry_layer": "chunk",
            "terminal_outcome": "completed",
            "retrieval_trace_id": "trace-1",
            "context_package_id": "package-1",
            "citations": [
                {
                    "source_binding_id": "binding-1",
                    "source_span": {"source_binding_id": "binding-1"},
                    "source_binding": {
                        "contract_version": "answer_source_binding_public_v3",
                        "protocol_version": "answer_source_binding_v2",
                        "status": "source_bound",
                        "source_integrity_admission_hash": "a" * 64,
                    },
                }
            ],
            "model_audit": {
                "protocol_version": "intent_execution_retrieval_v1",
                "planning_model_call_count": 1,
                "generation_model_call_count": 1,
                "post_generation_model_call_count": 0,
                "source_admission_model_call_count": 0,
                "policy_update_eligible": False,
                "source_binding_count": 1,
                "qa_performance": {
                    "protocol_version": "qa_stage_timing_v1",
                    "unfinished_span_count": 0,
                },
            },
        }
    )
    assert audit["insufficient_evidence"] is False
    assert audit["entry_layer"] == "chunk"
    assert audit["context_package_required"] is True
    assert audit["returned_citation_count"] == 1


def test_docker_smoke_accepts_pre_package_evidence_gate_block() -> None:
    docker_smoke = _load_docker_smoke()

    audit = docker_smoke.validate_qa_acceptance_payload(
        {
            "answer": "The available corpus is insufficient; narrow the question.",
            "citations": [],
            "context_package_id": None,
            "model_audit": {
                "context_package_evidence_gate_passed": False,
                "answer_model_called": False,
                "evidence_evaluator": {
                    "verdict": "insufficient_corpus"
                },
            },
        }
    )

    assert audit["insufficient_evidence"] is True
    assert audit["evidence_gate_blocked"] is True
    assert audit["context_package_required"] is False
    assert audit["returned_citation_count"] == 0


def test_docker_smoke_requires_citations_for_grounded_answer() -> None:
    docker_smoke = _load_docker_smoke()

    with pytest.raises(RuntimeError, match="Grounded QA returned no citations"):
        docker_smoke.validate_qa_acceptance_payload(
            {
                "answer": "A factual answer.",
                "citations": [],
                "model_audit": {
                    "grounding_outcome": "grounded_answer",
                    "insufficient_evidence": False,
                    "citation_verification_pass_rate": 0.0,
                },
            }
        )


def _pe_envelope(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"encoding": "canonical_json_v1", "canonical_json": raw,
            "sha256": hashlib.sha256(raw.encode()).hexdigest(), "redacted_fields": []}


def _current_gap_fixture(trace_ids=()):
    qa = {"run_id": "unit-test-run", "session_id": "unit-test-session",
          "route": "layered_context_graph", "answer": "No supported answer in the checked scope.",
          "citations": [], "model_audit": {"policy_update_eligible": False,
          "retrieval_control": {"protocol_version": "retrieval_answer_v1", "gate_outcome": "scoped_not_found",
              "generation_call_count": 0, "post_generation_review_count": 0,
              "source_binding_count": 0, "repairs_used": 0},
          "qa_performance": {"protocol_version": "qa_stage_timing_v1", "unfinished_span_count": 0}}}
    task = {**{key: qa[key] for key in ("run_id", "session_id", "route", "answer")},
            "status": "needs_clarification", "state": "needs_clarification", "error": None}
    plans, actions = [], []
    for index, trace_id in enumerate(trace_ids):
        plans.append({"id": f"unit-test-plan-{index}", "run_id": qa['run_id'], "order_index": index,
            "knowledge_base_id": "unit-test-kb", "status": "executed", "retrieval_trace_id": trace_id})
        actions.append({"id": f"unit-test-action-{index}", "run_id": qa['run_id'], "order_index": index,
            "status": "completed", "output": _pe_envelope({"retrieval_trace_id": trace_id})})
    states = ["planning", "task_ready"] + [name for _ in trace_ids for name in ("searching", "packing", "diagnosing")] + ["insufficient"]
    observations = []
    for index in range(len(states) - 1):
        def state(position):
            return {"state": states[position], "sequence_index": position,
                    "generation_started": False, "repairs_used": 0}
        event = {"protocol_version": "retrieval_fsm_transition_v1", "run_id": qa['run_id'],
                 "sequence_index": index + 1, "before": state(index), "after": state(index + 1)}
        event['event_hash'] = _pe_envelope(event)['sha256']
        observations.append({"id": f"unit-test-transition-{index}", "run_id": qa['run_id'], "order_index": index,
            "observation_type": "retrieval_state_transition", "run_control_protocol": "retrieval_fsm_v1",
            "observation": _pe_envelope(event)})
    pe = {"contract_version": "agent_pe_audit_public_v1", "run_id": qa['run_id'],
          "knowledge_base_id": "unit-test-kb", "run_status": "needs_clarification",
          "provider_raw_response_exposed": False, "credentials_exposed": False,
          "plans": plans, "actions": actions, "observations": observations,
          "counts": {"plans": len(plans), "actions": len(actions), "observations": len(observations)}}
    responses = {f"/tasks/{qa['run_id']}": task, f"/agent/runs/{qa['run_id']}/pe-audit": pe}
    responses.update({f"/retrieval-traces/{trace}/graph-steps": {"trace_id": trace, "steps": [{"layer": "chunk"}]}
                      for trace in trace_ids})
    class Client:
        def __init__(self):
            self.calls = []
        def request_json(self, method, path):
            assert method == "GET", "Gap audit must never issue a model or retrieval POST"
            self.calls.append(path)
            return copy.deepcopy(responses[path])
    return qa, responses, Client()


@pytest.mark.parametrize("trace_ids", [(), ("unit-test-trace-a",), ("unit-test-trace-a", "unit-test-trace-b")])
def test_smoke_current_gap_audits_all_executed_traces_or_no_retrieval(trace_ids):
    smoke = _load_docker_smoke()
    qa, _, client = _current_gap_fixture(trace_ids)
    acceptance = smoke.validate_qa_acceptance_payload(qa)
    traces, audit = smoke.load_qa_retrieval_traces(client, qa, acceptance, knowledge_base_id="unit-test-kb")
    assert [trace['trace_id'] for trace in traces] == list(trace_ids)
    assert audit == {"name": "qa_terminal_audit", "pass": True, "insufficient_evidence": True,
                     "trace_count": len(trace_ids)}
    assert len(client.calls) == 2 + len(trace_ids)
    assert not qa.get('context_package_id') and not qa.get('retrieval_trace_id')


@pytest.mark.parametrize("corruption", ["task_run", "task_answer", "task_failed", "kb", "missing_rows",
    "missing_transition", "canonical_hash", "event_hash", "generated", "repairs", "lost_plan_trace",
    "unowned_action_trace", "failed_action", "different_trace", "unknown_gap"])
def test_smoke_gap_does_not_mask_identity_audit_or_technical_failures(corruption):
    smoke = _load_docker_smoke()
    qa, responses, client = _current_gap_fixture(("unit-test-trace-a",))
    task = responses['/tasks/unit-test-run']
    pe = responses['/agent/runs/unit-test-run/pe-audit']
    if corruption == 'task_run': task['run_id'] = 'unit-test-other-run'
    elif corruption == 'task_answer': task['answer'] = 'Different result'
    elif corruption == 'task_failed': task.update(status='failed', state='failed')
    elif corruption == 'kb': pe['knowledge_base_id'] = 'unit-test-other-kb'
    elif corruption == 'missing_rows': pe['counts']['plans'] += 1
    elif corruption == 'missing_transition':
        pe['observations'].pop(0)
        for index, item in enumerate(pe['observations']): item['order_index'] = index
        pe['counts']['observations'] -= 1
    elif corruption == 'canonical_hash': pe['observations'][-1]['observation']['sha256'] = '0' * 64
    elif corruption in {'event_hash', 'generated', 'repairs'}:
        value = json.loads(pe['observations'][-1]['observation']['canonical_json'])
        if corruption == 'generated': value['after']['generation_started'] = True
        elif corruption == 'repairs': value['after']['repairs_used'] = 1
        value['event_hash'] = ('0' * 64 if corruption == 'event_hash' else
            _pe_envelope({key: item for key, item in value.items() if key != 'event_hash'})['sha256'])
        pe['observations'][-1]['observation'] = _pe_envelope(value)
    elif corruption == 'lost_plan_trace': pe['plans'][0]['retrieval_trace_id'] = None
    elif corruption == 'unowned_action_trace':
        pe['actions'][0]['output'] = _pe_envelope({'retrieval_trace_id': 'unit-test-other-trace'})
    elif corruption == 'failed_action': pe['actions'][0]['status'] = 'failed'
    elif corruption == 'different_trace':
        responses['/retrieval-traces/unit-test-trace-a/graph-steps']['trace_id'] = 'unit-test-other-trace'
    elif corruption == 'unknown_gap': qa['model_audit']['retrieval_control']['gate_outcome'] = 'unknown'
    with pytest.raises(RuntimeError):
        smoke.load_qa_retrieval_traces(client, qa, smoke.validate_qa_acceptance_payload(qa), knowledge_base_id="unit-test-kb")


def test_smoke_non_gap_still_requires_the_answer_trace():
    smoke = _load_docker_smoke()
    qa, _, client = _current_gap_fixture()
    with pytest.raises(RuntimeError, match='QA did not return retrieval_trace_id'):
        smoke.load_qa_retrieval_traces(client, qa, {"model_audit": {}, "insufficient_evidence": False},
                                      knowledge_base_id='unit-test-kb')
    assert not client.calls


class _FakeResponse:
    def __init__(self, body: bytes, *, headers: dict[str, str] | None = None) -> None:
        self._body = io.BytesIO(body)
        self.headers = headers or {"Content-Type": "application/json"}
        self.status = 200
        self.read_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._body.read(size)


def test_smoke_client_uses_bounded_json_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    docker_smoke = _load_docker_smoke()
    response = _FakeResponse(b'{"status":"ok"}')
    monkeypatch.setattr(docker_smoke, "urlopen", lambda *_a, **_k: response)

    payload = docker_smoke.SmokeClient("http://127.0.0.1:8000/api").request_json(
        "GET", "/health"
    )

    assert payload == {"status": "ok"}
    assert response.read_sizes
    assert -1 not in response.read_sizes
    assert max(response.read_sizes) <= docker_smoke.HTTP_READ_CHUNK_BYTES


def test_smoke_client_rejects_oversize_before_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_smoke = _load_docker_smoke()
    monkeypatch.setattr(docker_smoke, "MAX_PUBLIC_API_RESPONSE_BYTES", 16)
    response = _FakeResponse(b"x" * 17)
    monkeypatch.setattr(docker_smoke, "urlopen", lambda *_a, **_k: response)
    monkeypatch.setattr(
        docker_smoke.json,
        "loads",
        lambda *_a, **_k: pytest.fail("oversize body must be rejected before JSON parse"),
    )

    with pytest.raises(docker_smoke.SmokeTransportError) as raised:
        docker_smoke.SmokeClient("http://127.0.0.1:8000/api").request_json(
            "GET", "/health"
        )

    assert raised.value.error_code == "response_body_too_large"
    assert raised.value.observed_body_bytes == 17
    assert "x" * 17 not in str(raised.value)


def test_smoke_client_rejects_declared_oversize_and_non_json_without_body_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_smoke = _load_docker_smoke()
    declared = _FakeResponse(
        b"not-read",
        headers={"Content-Type": "application/json", "Content-Length": str(33 * 1024 * 1024)},
    )
    monkeypatch.setattr(docker_smoke, "urlopen", lambda *_a, **_k: declared)
    with pytest.raises(docker_smoke.SmokeTransportError) as raised:
        docker_smoke.SmokeClient("http://127.0.0.1:8000/api").request_json(
            "GET", "/health"
        )
    assert raised.value.error_code == "response_body_too_large"
    assert declared.read_sizes == []

    non_json = _FakeResponse(b"<html>secret</html>", headers={"Content-Type": "text/html"})
    monkeypatch.setattr(docker_smoke, "urlopen", lambda *_a, **_k: non_json)
    with pytest.raises(docker_smoke.SmokeTransportError) as raised:
        docker_smoke.SmokeClient("http://127.0.0.1:8000/api").request_json(
            "GET", "/health"
        )
    assert raised.value.error_code == "non_json_content_type"
    assert non_json.read_sizes == []


def test_smoke_http_error_is_bounded_and_never_logs_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_smoke = _load_docker_smoke()
    body = b'{"detail":"must-not-be-logged"}'
    error = HTTPError(
        "http://127.0.0.1:8000/api/health",
        503,
        "unavailable",
        {"Content-Type": "application/json"},
        io.BytesIO(body),
    )

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(docker_smoke, "urlopen", fail)
    with pytest.raises(docker_smoke.SmokeTransportError) as raised:
        docker_smoke.SmokeClient("http://127.0.0.1:8000/api").request_json(
            "GET", "/health"
        )

    assert raised.value.error_code == "http_error"
    assert raised.value.status_code == 503
    assert raised.value.observed_body_bytes == len(body)
    assert "must-not-be-logged" not in str(raised.value)
