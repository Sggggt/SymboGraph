from __future__ import annotations

import importlib.util
import io
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
        "protocol_version": "query_rq_fuzzy_membership_chunk_seed_v2",
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


def test_docker_smoke_rejects_retired_candidate_rq_only_trace() -> None:
    docker_smoke = _load_docker_smoke()
    trace = _rq_seed_trace_fixture()
    trace.pop("trace_diagnostics")
    trace.pop("candidate_pools")
    trace["steps"][0]["input"].pop("query_rq_seed_audit")
    trace["steps"][0]["output"] = {"candidate_rq": [{"id": "rq-1"}]}

    with pytest.raises(RuntimeError, match="typed RQ seed audit"):
        docker_smoke.validate_retrieval_rq_seed_diagnostics(trace)


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
