from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from _gray_zone_audit import audit_gray_zone_traces


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
ACTIVE_CHUNK_RELATION_EDGE_TYPES = {
    "dense_semantic",
    "dense_cross_document_bridge",
    "dense_cross_language_bridge",
}
MAX_PUBLIC_API_RESPONSE_BYTES = 32 * 1024 * 1024
HTTP_READ_CHUNK_BYTES = 64 * 1024
# A cache-miss Search can make two bounded query-perception attempts and one
# embedding request. The API's default per-model-call timeout is 240 seconds,
# so the smoke client must not abandon a valid fail-closed request after the
# legacy 60-second window. QA contains several sequential bounded stages.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_QA_TIMEOUT_SECONDS = 1800.0


class SmokeTransportError(RuntimeError):
    def __init__(
        self,
        *,
        method: str,
        url: str,
        error_code: str,
        status_code: int | None = None,
        observed_body_bytes: int = 0,
        bounded_body_sha256: str | None = None,
    ) -> None:
        self.method = method
        self.url = url
        self.error_code = error_code
        self.status_code = status_code
        self.observed_body_bytes = observed_body_bytes
        self.bounded_body_sha256 = bounded_body_sha256
        super().__init__(
            f"{method} {url} failed: code={error_code} status={status_code} "
            f"observed_body_bytes={observed_body_bytes}"
        )


def _response_content_type(headers: object) -> str:
    getter = getattr(headers, "get", None)
    raw = getter("Content-Type", "") if callable(getter) else ""
    return str(raw or "").split(";", 1)[0].strip().lower()


def _declared_content_length(
    headers: object, *, method: str, url: str, status_code: int
) -> int | None:
    getter = getattr(headers, "get", None)
    raw = getter("Content-Length", "") if callable(getter) else ""
    if raw in {None, ""}:
        return None
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise SmokeTransportError(
            method=method,
            url=url,
            error_code="invalid_content_length",
            status_code=status_code,
        ) from exc
    if value < 0:
        raise SmokeTransportError(
            method=method,
            url=url,
            error_code="invalid_content_length",
            status_code=status_code,
        )
    return value


def _read_bounded_body(
    stream: object,
    *,
    method: str,
    url: str,
    status_code: int,
    too_large_error_code: str = "response_body_too_large",
) -> bytes:
    parts: list[bytes] = []
    observed = 0
    while observed <= MAX_PUBLIC_API_RESPONSE_BYTES:
        remaining = MAX_PUBLIC_API_RESPONSE_BYTES + 1 - observed
        block = stream.read(min(HTTP_READ_CHUNK_BYTES, remaining))
        if not block:
            break
        raw = bytes(block)
        parts.append(raw)
        observed += len(raw)
    body = b"".join(parts)
    if len(body) > MAX_PUBLIC_API_RESPONSE_BYTES:
        raise SmokeTransportError(
            method=method,
            url=url,
            error_code=too_large_error_code,
            status_code=status_code,
            observed_body_bytes=len(body),
            bounded_body_sha256=hashlib.sha256(body).hexdigest(),
        )
    return body


class SmokeClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> dict | list:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode({key: value for key, value in params.items() if value is not None})}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                content_type = _response_content_type(response.headers)
                if content_type != "application/json":
                    raise SmokeTransportError(
                        method=method,
                        url=url,
                        error_code="non_json_content_type",
                        status_code=int(getattr(response, "status", 200)),
                    )
                declared = _declared_content_length(
                    response.headers,
                    method=method,
                    url=url,
                    status_code=int(getattr(response, "status", 200)),
                )
                if declared is not None and declared > MAX_PUBLIC_API_RESPONSE_BYTES:
                    raise SmokeTransportError(
                        method=method,
                        url=url,
                        error_code="response_body_too_large",
                        status_code=int(getattr(response, "status", 200)),
                        observed_body_bytes=0,
                    )
                raw = _read_bounded_body(
                    response,
                    method=method,
                    url=url,
                    status_code=int(getattr(response, "status", 200)),
                )
                try:
                    decoded = raw.decode("utf-8")
                    parsed = json.loads(decoded) if decoded else {}
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise SmokeTransportError(
                        method=method,
                        url=url,
                        error_code="invalid_json_response",
                        status_code=int(getattr(response, "status", 200)),
                        observed_body_bytes=len(raw),
                        bounded_body_sha256=hashlib.sha256(raw).hexdigest(),
                    ) from exc
                if not isinstance(parsed, (dict, list)):
                    raise SmokeTransportError(
                        method=method,
                        url=url,
                        error_code="non_object_or_list_json",
                        status_code=int(getattr(response, "status", 200)),
                        observed_body_bytes=len(raw),
                        bounded_body_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                return parsed
        except HTTPError as exc:
            content_type = _response_content_type(exc.headers)
            if content_type != "application/json":
                raise SmokeTransportError(
                    method=method,
                    url=url,
                    error_code="http_error_non_json_content_type",
                    status_code=int(exc.code),
                ) from exc
            declared = _declared_content_length(
                exc.headers,
                method=method,
                url=url,
                status_code=int(exc.code),
            )
            if declared is not None and declared > MAX_PUBLIC_API_RESPONSE_BYTES:
                raise SmokeTransportError(
                    method=method,
                    url=url,
                    error_code="http_error_response_body_too_large",
                    status_code=int(exc.code),
                    observed_body_bytes=0,
                ) from exc
            try:
                raw = _read_bounded_body(
                    exc,
                    method=method,
                    url=url,
                    status_code=int(exc.code),
                    too_large_error_code="http_error_response_body_too_large",
                )
            except SmokeTransportError as bounded_exc:
                raise bounded_exc from exc
            raise SmokeTransportError(
                method=method,
                url=url,
                error_code="http_error",
                status_code=int(exc.code),
                observed_body_bytes=len(raw),
                bounded_body_sha256=hashlib.sha256(raw).hexdigest(),
            ) from exc
        except URLError as exc:
            raise SmokeTransportError(
                method=method,
                url=url,
                error_code="url_error",
            ) from exc
        except TimeoutError as exc:
            raise SmokeTransportError(
                method=method,
                url=url,
                error_code="timeout",
            ) from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_target_gray_zone_traces(
    traces: list[dict],
    *,
    require_gray_coverage: bool = False,
) -> dict:
    records = [
        dict(record)
        for trace in traces
        for record in trace.get("gray_zone_path_decisions") or []
    ]
    for record in records:
        inputs = record.get("inputs")
        canonical = (
            json.dumps(
                inputs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if isinstance(inputs, dict)
            else ""
        )
        require(
            record.get("protocol_version")
            == "deterministic_support_progress_v2"
            and record.get("model_call_count") == 0
            and bool(canonical)
            and hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            == record.get("input_hash"),
            "Target gray-zone decision failed deterministic hash replay",
        )
    gray = sum((item.get("inputs") or {}).get("zone") == "gray" for item in records)
    red = sum((item.get("inputs") or {}).get("zone") == "red" for item in records)
    hard = sum((item.get("inputs") or {}).get("zone") == "hard_stop" for item in records)
    if require_gray_coverage:
        require(gray > 0, "Target smoke trace did not exercise a gray decision")
    return {
        "protocol_version": "intent_execution_gray_zone_smoke_v1",
        "status": "passed",
        "pass": True,
        "gray_zone_coverage": gray > 0,
        "trace_count": len(traces),
        "gray_rule_record_count": gray,
        "red_partition_record_count": red,
        "hard_stop_partition_record_count": hard,
        "determinism": {
            "all_input_hashes_replayed": True,
            "model_call_count": 0,
            "cycle_reward": 0,
        },
    }


def validate_qa_acceptance_payload(qa: dict) -> dict:
    require(qa.get("answer"), "QA returned no answer")
    qa_audit = qa.get("model_audit") or {}
    if qa_audit.get("protocol_version") == "intent_execution_retrieval_v1":
        citations = list(qa.get("citations") or [])
        outcome = qa.get("terminal_outcome")
        direct = qa.get("route") == "system_capability"
        accepted = outcome in {"completed", "partial_answer"} and not direct
        require(qa_audit.get("planning_model_call_count") == 1, "Target planning call count differs")
        require(qa_audit.get("post_generation_model_call_count", 0) == 0, "Target path called post-generation review")
        require(qa_audit.get("source_admission_model_call_count", 0) == 0, "Target source admission called a model")
        require(not qa_audit.get("policy_update_eligible"), "Target path enabled online policy updates")
        require(qa_audit.get("generation_model_call_count", 0) == int(accepted), "Target one-shot generation contract failed")
        if direct:
            require(not citations and qa.get("context_package_id") is None and qa.get("retrieval_trace_id") is None,
                    "System capability response has corpus sources")
        elif accepted:
            require(bool(citations), "Completed target factual QA has no sources")
            require(qa.get("context_package_id") and qa.get("retrieval_trace_id"), "Completed target QA lacks package identity")
            require(qa_audit.get("source_binding_count") == len(citations), "Target source count differs from binding audit")
        else:
            require(not citations, "Grounding terminal returned unsupported citations")
        require(all(
            (citation.get("source_binding") or {}).get("contract_version") == "answer_source_binding_public_v3"
            and (citation.get("source_binding") or {}).get("protocol_version") == "answer_source_binding_v2"
            and (citation.get("source_binding") or {}).get("status") == "source_bound"
            and bool((citation.get("source_binding") or {}).get("source_integrity_admission_hash"))
            and citation.get("source_binding_id") == (citation.get("source_span") or {}).get("source_binding_id")
            for citation in citations
        ), "Target source-integrity binding contract is inconsistent")
        timing = qa_audit.get("qa_performance") or {}
        require(timing.get("protocol_version") == "qa_stage_timing_v1"
                and timing.get("unfinished_span_count") == 0, "Target QA timing audit is missing or unfinished")
        return {
            "model_audit": qa_audit,
            "citation_verification_pass_rate": None,
            "source_binding_pass_rate": 1.0 if citations else None,
            "insufficient_evidence": not accepted and not direct,
            "evidence_gate_blocked": False,
            "context_package_required": accepted,
            "returned_citation_count": len(citations),
            "terminal_outcome": outcome,
            "entry_layer": qa.get("entry_layer"),
        }
    current = qa_audit.get("retrieval_control") or {}
    if current.get("protocol_version") == "retrieval_answer_v1":
        citations = list(qa.get("citations") or [])
        accepted = current.get("gate_outcome") in {"ready_full", "ready_partial"}
        require(current.get("post_generation_review_count") == 0, "Post-generation review was called")
        require(current.get("generation_call_count") == int(accepted), "One-shot generation contract failed")
        require(not qa_audit.get("policy_update_eligible"), "Retired answer policy received a retrieval reward")
        require(current.get("source_binding_count") == len(citations), "Source count differs from audit")
        require(bool(citations) == accepted, "Ready/gap response has inconsistent sources")
        require(all((c.get("source_binding") or {}).get("protocol_version") == "answer_source_binding_v2"
            and (c.get("source_binding") or {}).get("status") == "source_bound"
            and (c.get("source_binding") or {}).get("semantic_entailment_claimed") is False
            and c.get("source_binding_id") == (c.get("source_span") or {}).get("source_binding_id")
            for c in citations), "Current source binding contract is inconsistent")
        timing = qa_audit.get("qa_performance") or {}
        require(timing.get("protocol_version") == "qa_stage_timing_v1"
                and timing.get("unfinished_span_count") == 0, "QA timing audit is missing or unfinished")
        return {"model_audit": qa_audit, "citation_verification_pass_rate": None,
            "source_binding_pass_rate": current.get("source_binding_pass_rate"),
            "insufficient_evidence": not accepted, "evidence_gate_blocked": not accepted,
            "context_package_required": accepted, "returned_citation_count": len(citations)}
    reflection = qa_audit.get("answer_reflection") or {}
    if reflection.get("protocol_version") == "agent_answer_reflection_v1":
        citations = list(qa.get("citations") or [])
        accepted = reflection.get("outcome") in {"accepted_without_reflection", "accepted_after_reflection"}
        require(reflection.get("citation_judge_model_call_count") == 0, "Retired citation judge was called")
        require(reflection.get("self_assessment_is_reward_label") is False, "Model self-score was treated as a reward label")
        if accepted:
            require(bool(citations), "Accepted factual QA has no source bindings")
            require(reflection.get("source_binding_pass_rate") == 1.0, "Source binding did not pass")
        require(all(c.get("contract_version") == "citation_public_v2" and c.get("verification") is None
            and (c.get("source_binding") or {}).get("status") == "source_bound"
            and (c.get("source_binding") or {}).get("semantic_entailment_claimed") is False
            and c.get("source_binding_id") == (c.get("source_span") or {}).get("source_binding_id")
            for c in citations), "Source binding contract is inconsistent")
        return {"model_audit": qa_audit, "citation_verification_pass_rate": None,
            "source_binding_pass_rate": reflection.get("source_binding_pass_rate"),
            "insufficient_evidence": not accepted, "evidence_gate_blocked": False,
            "context_package_required": True, "returned_citation_count": len(citations)}
    grounding_outcome = qa_audit.get("grounding_outcome")
    evaluator = qa_audit.get("evidence_evaluator") or {}
    evidence_gate_blocked = bool(
        qa_audit.get("context_package_evidence_gate_passed") is False
        and qa_audit.get("answer_model_called") is False
        and evaluator.get("verdict")
        in {"insufficient_corpus", "need_expansion"}
    )
    insufficient_evidence = bool(
        qa_audit.get("insufficient_evidence")
        or grounding_outcome == "insufficient_evidence"
        or evidence_gate_blocked
    )
    citations = list(qa.get("citations") or [])
    pass_rate = qa_audit.get("citation_verification_pass_rate")
    if not insufficient_evidence:
        require(citations, f"Grounded QA returned no citations: {qa}")
        require(
            pass_rate is None or float(pass_rate) > 0.0,
            f"QA citation verification did not pass: {qa}",
        )
    return {
        "model_audit": qa_audit,
        "citation_verification_pass_rate": pass_rate,
        "insufficient_evidence": insufficient_evidence,
        "evidence_gate_blocked": evidence_gate_blocked,
        "context_package_required": not evidence_gate_blocked,
        "returned_citation_count": len(citations),
    }


def _decode_pe_payload(payload: dict) -> dict:
    """Check the public canonical envelope before consuming control identities."""
    raw = payload.get("canonical_json")
    require(payload.get("encoding") == "canonical_json_v1" and isinstance(raw, str),
            "P&E canonical payload is missing")
    require(hashlib.sha256(raw.encode("utf-8")).hexdigest() == payload.get("sha256"),
            "P&E canonical payload hash differs")
    try:
        value = json.loads(raw)
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise RuntimeError("P&E canonical payload is invalid") from None
    require(isinstance(value, dict) and raw == canonical, "P&E control payload is not canonical object JSON")
    return value


def load_qa_retrieval_traces(client, qa: dict, acceptance: dict, *, knowledge_base_id: str) -> tuple[list[dict], dict]:
    """Read every executed gap trace without inventing a final answer trace."""
    current = acceptance["model_audit"].get("retrieval_control") or {}
    is_gap = current.get("protocol_version") == "retrieval_answer_v1" and acceptance["insufficient_evidence"]
    if not is_gap:
        trace_id = qa.get("retrieval_trace_id")
        require(trace_id, "QA did not return retrieval_trace_id")
        trace_ids = [trace_id]
    else:
        require(current.get("gate_outcome") in {"scoped_not_found", "scope_ambiguous", "source_unresolved",
                "representation_incomplete", "budget_exhausted"}, "Unknown or non-terminal QA gap outcome")
        require(not qa.get("citations") and not qa.get("context_package_id") and not qa.get("retrieval_trace_id")
                and current.get("generation_call_count") == 0 and current.get("source_binding_count") == 0,
                "QA gap contains answer-generation artifacts")
        run_id = qa.get("run_id")
        require(isinstance(run_id, str) and bool(run_id), "QA gap has no persisted run")
        route_id = quote(run_id, safe="")
        task = client.request_json("GET", f"/tasks/{route_id}")
        require(task.get("run_id") == run_id and task.get("session_id") == qa.get("session_id")
                and task.get("status") == task.get("state") == "needs_clarification"
                and task.get("route") == qa.get("route") == "layered_context_graph"
                and task.get("answer") == qa.get("answer") and not task.get("error"),
                "QA gap differs from its persisted terminal task")
        pe = client.request_json("GET", f"/agent/runs/{route_id}/pe-audit")
        require(pe.get("contract_version") == "agent_pe_audit_public_v1" and pe.get("run_id") == run_id
                and pe.get("knowledge_base_id") == knowledge_base_id
                and pe.get("run_status") == task["status"]
                and pe.get("provider_raw_response_exposed") is False and pe.get("credentials_exposed") is False,
                "QA gap P&E identity or public safety differs")
        for name in ("plans", "actions", "observations"):
            rows = pe.get(name)
            require(isinstance(rows, list) and (pe.get("counts") or {}).get(name) == len(rows),
                    "QA gap P&E rows are incomplete")
            require([row.get("order_index") for row in rows] == list(range(len(rows)))
                    and len({row.get("id") for row in rows}) == len(rows)
                    and all(row.get("run_id") == run_id for row in rows), "QA gap P&E row ownership differs")
        trace_ids = []
        for plan in pe["plans"]:
            require(plan.get("knowledge_base_id") == knowledge_base_id, "QA gap plan crosses knowledge bases")
            trace_id = plan.get("retrieval_trace_id")
            require(plan.get("status") != "executed" or trace_id, "Executed QA plan lost its retrieval trace")
            if trace_id and trace_id not in trace_ids:
                trace_ids.append(trace_id)
        for action in pe["actions"]:
            require(action.get("status") not in {"executing", "failed", "cancelled"},
                    "Technical action failure cannot become a successful gap")
            trace_id = _decode_pe_payload(action["output"]).get("retrieval_trace_id")
            require(not trace_id or trace_id in trace_ids, "QA action trace has no owning plan")
        transitions = []
        for observation in pe["observations"]:
            value = _decode_pe_payload(observation["observation"])
            trace_id = value.get("retrieval_trace_id")
            require(not trace_id or trace_id in trace_ids, "QA observation trace has no owning plan")
            if observation.get("observation_type") == "retrieval_state_transition":
                require(observation.get("run_control_protocol") == "retrieval_fsm_v1"
                        and value.get("protocol_version") == "retrieval_fsm_transition_v1"
                        and value.get("run_id") == run_id, "QA gap FSM identity differs")
                event = {key: item for key, item in value.items() if key != "event_hash"}
                canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                require(hashlib.sha256(canonical.encode()).hexdigest() == value.get("event_hash"),
                        "QA gap FSM event hash differs")
                transitions.append(value)
        transitions.sort(key=lambda event: event["sequence_index"])
        require(transitions and [event["sequence_index"] for event in transitions] == list(range(1, len(transitions) + 1)),
                "QA gap FSM transition sequence is incomplete")
        for index, event in enumerate(transitions):
            require(event["before"]["sequence_index"] == index and event["after"]["sequence_index"] == index + 1
                    and (index == 0 or transitions[index - 1]["after"] == event["before"]),
                    "QA gap FSM transitions do not join")
        final = transitions[-1]["after"]
        require(final.get("state") == "insufficient" and final.get("generation_started") is False
                and final.get("repairs_used") == current.get("repairs_used"), "QA gap FSM terminal differs")
    traces = []
    for trace_id in trace_ids:
        require(isinstance(trace_id, str) and bool(trace_id), "QA retrieval trace identity is invalid")
        trace = client.request_json("GET", f"/retrieval-traces/{quote(trace_id, safe='')}/graph-steps")
        require(trace.get("trace_id") == trace_id and trace.get("steps"), "QA retrieval trace identity or steps differ")
        traces.append(trace)
    return traces, {"name": "qa_terminal_audit", "pass": True,
                    "insufficient_evidence": bool(is_gap), "trace_count": len(traces)}


def validate_retrieval_rq_seed_diagnostics(trace: dict) -> dict[str, int]:
    """Validate the current public RQ seed audit without retired fields."""
    seed_steps = [
        step
        for step in trace.get("steps", [])
        if step.get("layer") == "chunk"
        and step.get("action") == "select_seeds_from_mid_rq_membership"
    ]
    require(seed_steps, "Retrieval trace has no rq-prefix-address seed selection step")
    seed_input = seed_steps[0].get("input") or {}
    require(seed_input.get("query_rq_path"), "Retrieval trace has no RQ query path")

    trace_diagnostics = trace.get("trace_diagnostics") or {}
    audit = trace_diagnostics.get("query_rq_seed_audit") or seed_input.get(
        "query_rq_seed_audit"
    )
    require(isinstance(audit, dict), "Retrieval trace has no typed RQ seed audit")
    require(
        audit.get("model_call_count") == 0
        and audit.get("gray_zone_decision_authority") is False
        and audit.get("is_evidence") is False,
        "Retrieval RQ seed audit violated the evidence/gray boundary",
    )

    rq_pool = (trace.get("candidate_pools") or {}).get("rq_membership_entries") or {}
    cards = rq_pool.get("rq_seed_cards")
    require(
        isinstance(cards, dict) and bool(cards),
        "Retrieval trace has no hash-bound RQ seed cards",
    )
    require(
        rq_pool.get("ranking_protocol_version") == audit.get("protocol_version")
        and rq_pool.get("ranking_protocol_hash") == audit.get("protocol_hash"),
        "Retrieval RQ seed pool is not bound to its typed audit",
    )
    require(
        all(
            isinstance(card, dict)
            and card.get("model_call_count") == 0
            and card.get("gray_zone_decision_authority") is False
            and card.get("is_evidence") is False
            and len(str(card.get("input_hash") or "")) == 64
            and len(str(card.get("card_hash") or "")) == 64
            for card in cards.values()
        ),
        "Retrieval RQ seed cards violated the typed audit contract",
    )
    return {"seed_steps": len(seed_steps), "rq_seed_cards": len(cards)}


def select_smoke_knowledge_base(
    knowledge_bases: list[dict],
    *,
    requested_id: str | None,
    freshness_loader=None,
) -> tuple[dict, str]:
    if requested_id:
        selected = next(
            (item for item in knowledge_bases if item.get("id") == requested_id),
            None,
        )
        require(selected is not None, f"Knowledge base not found: {requested_id}")
        return selected, "explicit_id"

    graph_ready = [
        item
        for item in knowledge_bases
        if int(item.get("active_chunk_count") or item.get("chunk_count") or 0) > 0
        and bool(item.get("context_graph_state_id"))
        and bool(item.get("context_graph_hash"))
        and not item.get("stale_reason")
    ]
    if freshness_loader is not None:
        graph_ready = [item for item in graph_ready
                       if (freshness_loader(item["id"]).get("freshness") or {}).get("is_admissible") is True]
    else:
        graph_ready = [item for item in graph_ready if (item.get("freshness") or {}).get("is_admissible") is not False]
    require(
        bool(graph_ready),
        "No graph-ready knowledge base is available; pass --knowledge-base-id "
        "after building and promoting a fresh four-layer context graph",
    )
    return graph_ready[0], "first_fresh_graph_ready"


def safe_smoke_report(payload: dict) -> dict:
    """Reports contain aggregate checks, never provider or graph payloads."""
    report = {key: payload[key] for key in ("script", "mode", "execute", "pass", "error_type", "error_code") if key in payload}
    report["checks"] = []
    for check in payload.get("checks", []):
        summary = {key: value for key, value in check.items() if key in {
            "name", "pass", "count", "nodes", "edges", "result_count", "returned_citation_count", "persisted_citation_span_count",
            "citation_verification_pass_rate", "source_binding_pass_rate", "insufficient_evidence", "evidence_gate_blocked",
            "gray_zone_coverage", "gray_zone_coverage_required", "trace_count", "gray_rule_record_count", "red_partition_record_count", "hard_stop_partition_record_count"}}
        for key in ("contract_counts", "rq_seed_counts", "counts"):
            if isinstance(check.get(key), dict):
                summary[key] = {name: number for name, number in check[key].items() if isinstance(number, (int, float))}
        report["checks"].append(summary)
    report["selection"] = {key: value for key, value in payload.get("knowledge_base_selection", {}).items() if key in {"reason", "active_chunk_count"}}
    return report


def write_report(payload: dict) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"docker_smoke_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(safe_smoke_report(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the four-layer context graph API in Docker.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--query", default="What are the main concepts in this knowledge base?")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Send the write-capable POST /search and POST /qa acceptance "
            "requests. Omit for a read-only GET preflight and exact write plan."
        ),
    )
    parser.add_argument("--wait-batch-id")
    parser.add_argument("--wait-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--qa-timeout-seconds",
        type=float,
        default=DEFAULT_QA_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--require-gray-coverage",
        action="store_true",
        help="Fail unless the search/QA persisted traces include at least one deterministic gray local-rule decision.",
    )
    parser.add_argument(
        "--require-relation-edge-coverage",
        action="store_true",
        help="Fail unless the sampled chunk-relation graph contains at least one active dense relation edge.",
    )
    parser.add_argument(
        "--require-rq-diagnostic-coverage",
        action="store_true",
        help="Fail unless the sampled chunk-relation graph contains at least one explicitly non-active RQ prefix-pair diagnostic edge.",
    )
    return parser.parse_args()


def wait_for_batch(client: SmokeClient, batch_id: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        batch = client.request_json("GET", f"/ingestion/batches/{batch_id}")
        if batch.get("state") in {"completed", "partial_failed", "failed", "skipped", "cancelled", "cancel_failed"}:
            return batch
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for batch {batch_id}")


def validate_chunk_relation_graph_payload(
    graph: dict,
    *,
    require_relation_edge_coverage: bool = False,
    require_rq_diagnostic_coverage: bool = False,
) -> dict[str, int]:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    active_relation_edges = [
        edge
        for edge in edges
        if edge.get("contract_kind") == "chunk_relation_edge"
    ]
    rq_chunk_nodes = [
        node
        for node in nodes
        if node.get("contract_kind") == "chunk_node"
        and ((node.get("metadata") or {}).get("rq_path"))
    ]
    rq_prefix_nodes = [
        node
        for node in nodes
        if node.get("contract_kind") == "rq_prefix_node"
        and node.get("category") == "rq_prefix"
    ]
    rq_membership_edges = [
        edge
        for edge in edges
        if edge.get("contract_kind") == "rq_membership_edge"
        and edge.get("category") == "rq_membership"
    ]
    rq_diagnostic_edges = [
        edge
        for edge in edges
        if edge.get("contract_kind") == "rq_diagnostic_edge"
    ]

    require(
        all(
            edge.get("type") in ACTIVE_CHUNK_RELATION_EDGE_TYPES
            for edge in active_relation_edges
        ),
        f"Chunk relation graph contains an invalid active edge type: {active_relation_edges}",
    )
    require(
        not any(
            str(edge.get("type") or "").startswith("rq_")
            for edge in active_relation_edges
        ),
        f"RQ-derived diagnostics leaked into active chunk relation edges: {active_relation_edges}",
    )
    require(rq_chunk_nodes, f"Chunk relation graph has no chunk RQ path metadata: {graph}")
    require(rq_prefix_nodes, f"Chunk relation graph has no visible RQ prefix nodes: {graph}")
    require(rq_membership_edges, f"Chunk relation graph has no visible RQ membership edges: {graph}")
    require(
        all(
            edge.get("type") == "rq_prefix_pair_diagnostic"
            and (edge.get("metadata") or {}).get("diagnostic_only") is True
            and (edge.get("metadata") or {}).get("active_relation_edge") is False
            for edge in rq_diagnostic_edges
        ),
        f"RQ prefix-pair diagnostics are not explicitly non-active: {rq_diagnostic_edges}",
    )
    if require_relation_edge_coverage:
        require(
            bool(active_relation_edges),
            f"Chunk relation graph has no sampled active relation edge: {graph}",
        )
    if require_rq_diagnostic_coverage:
        require(
            bool(rq_diagnostic_edges),
            f"Chunk relation graph has no sampled RQ prefix-pair diagnostic edge: {graph}",
        )
    return {
        "active_relation_edges": len(active_relation_edges),
        "rq_chunk_nodes": len(rq_chunk_nodes),
        "rq_prefix_nodes": len(rq_prefix_nodes),
        "rq_membership_edges": len(rq_membership_edges),
        "rq_diagnostic_edges": len(rq_diagnostic_edges),
    }


def main() -> int:
    args = parse_args()
    client = SmokeClient(args.base_url, timeout=float(args.request_timeout_seconds))
    payload: dict = {
        "script": "docker_smoke",
        "base_url": args.base_url,
        "mode": "execute" if args.execute else "dry_run",
        "execute": bool(args.execute),
        "checks": [],
    }
    try:
        health = client.request_json("GET", "/health")
        require(health.get("status") == "ok", f"Health check failed: {health}")
        payload["checks"].append({"name": "health", "pass": True, "payload": health})

        knowledge_bases = client.request_json("GET", "/knowledge_bases")
        require(isinstance(knowledge_bases, list) and knowledge_bases, "No knowledge bases returned")
        stats_cache = {}
        def load_stats(kb_id):
            if kb_id not in stats_cache:
                stats_cache[kb_id] = client.request_json("GET", f"/knowledge_bases/{kb_id}/context-graph/stats")
            return stats_cache[kb_id]
        selected, selection_reason = select_smoke_knowledge_base(
            knowledge_bases,
            requested_id=args.knowledge_base_id,
            freshness_loader=load_stats,
        )
        knowledge_base_id = selected["id"]
        payload["knowledge_base_id"] = knowledge_base_id
        payload["knowledge_base_selection"] = {
            "reason": selection_reason,
            "name": selected.get("name"),
            "active_chunk_count": int(
                selected.get("active_chunk_count")
                or selected.get("chunk_count")
                or 0
            ),
            "context_graph_state_id": selected.get("context_graph_state_id"),
            "context_graph_hash": selected.get("context_graph_hash"),
            "stale_reason": selected.get("stale_reason"),
        }
        search_request = {
            "knowledge_base_id": knowledge_base_id,
            "query": args.query,
            "top_k": 5,
            "filters": {},
        }
        qa_request = {
            "knowledge_base_id": knowledge_base_id,
            "question": args.query,
            "top_k": 5,
            "filters": {},
            "history": [],
        }
        payload["write_plan"] = {
            "knowledge_base_id": knowledge_base_id,
            "query": args.query,
            "http_post_targets": [
                {"path": "/search", "payload": search_request},
                {"path": "/qa", "payload": qa_request},
            ],
            "impact": (
                "POST /search may persist retrieval traces and Context "
                "Packages and may mutate shared retrieval cache state; POST "
                "/qa may additionally persist Agent, answer, citation, "
                "reward, policy, and session audit state"
            ),
        }
        payload["checks"].append({"name": "knowledge_bases", "pass": True, "count": len(knowledge_bases)})

        if args.wait_batch_id:
            batch = wait_for_batch(client, args.wait_batch_id, args.wait_timeout_seconds)
            require(batch.get("state") == "completed", f"Batch did not complete cleanly: {batch}")
            payload["checks"].append({"name": "batch_completed", "pass": True, "payload": batch})

        stats = load_stats(knowledge_base_id)
        require((stats.get("freshness") or {}).get("is_admissible") is True,
                "Selected knowledge base is not admissible; rebuild and promote it before Search/QA smoke")
        require((stats.get("counts") or {}).get("active_chunks", 0) > 0, f"No current chunks: {stats}")
        require((stats.get("counts") or {}).get("chunk_relation_edges", 0) >= 0, f"Missing relation stats: {stats}")
        payload["checks"].append({"name": "context_graph_stats", "pass": True, "counts": stats.get("counts") or {}})

        for graph_type in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
            graph = client.request_json("GET", f"/knowledge_bases/{knowledge_base_id}/graph/{graph_type}", params={"limit": 80})
            require(graph.get("graph_type") == graph_type, f"Wrong graph type for {graph_type}: {graph}")
            require("counts" in graph or "node_counts" in graph, f"Missing counts for {graph_type}: {graph}")
            graph_contract_counts = None
            if graph_type == "chunk-relation":
                graph_contract_counts = validate_chunk_relation_graph_payload(
                    graph,
                    require_relation_edge_coverage=bool(
                        args.require_relation_edge_coverage
                    ),
                    require_rq_diagnostic_coverage=bool(
                        args.require_rq_diagnostic_coverage
                    ),
                )
            payload["checks"].append(
                {
                    "name": f"graph_{graph_type}",
                    "pass": True,
                    "nodes": len(graph.get("nodes", [])),
                    "edges": len(graph.get("edges", [])),
                    "contract_counts": graph_contract_counts,
                }
            )

        if not args.execute:
            payload["impact"] = (
                "read-only HTTP GET preflight only; no POST /search or "
                "POST /qa and no production-state mutation"
            )
            payload["pass"] = True
            report = write_report(payload)
            print(
                json.dumps(
                    {"output": str(report), **safe_smoke_report(payload), "write_plan": payload["write_plan"]},
                    ensure_ascii=False,
                    default=str,
                )
            )
            return 0

        search = client.request_json("POST", "/search", search_request)
        require(search.get("results"), f"Search returned no results: {search}")
        trace_id = search.get("retrieval_trace_id") or (
            search.get("model_audit") or {}
        ).get("retrieval_trace_id")
        require(bool(trace_id), f"Search did not record retrieval_trace_id: {search}")
        target_search = (
            (search.get("execution_strategy") or {}).get("protocol_version")
            == "intent_execution_strategy_v2"
            and len(str(search.get("accepted_plan_hash") or "")) == 64
        )
        if target_search:
            require(
                search.get("terminal_outcome") == "completed"
                and search.get("entry_layer") in {"coarse", "mid", "chunk"}
                and (search.get("execution_strategy") or {}).get("entry_layer")
                == search.get("entry_layer"),
                f"Target Search plan or terminal identity is inconsistent: {search}",
            )
        else:
            require(
                (search.get("model_audit") or {}).get("retrieval_trace_id")
                == trace_id,
                f"Search trace identity is inconsistent across the public response: {search}",
            )
        search_context_package_id = search.get("context_package_id")
        require(
            bool(search_context_package_id),
            f"Ordinary search did not create a Context Package: {search}",
        )
        if not target_search:
            require(
                (search.get("model_audit") or {}).get("context_package_id")
                == search_context_package_id,
                f"Search Context Package identity is inconsistent: {search}",
            )
            require(
                (search.get("model_audit") or {}).get("query_rq_path"),
                f"Search audit did not include query RQ path: {search}",
            )
        search_cache_audit = (search.get("model_audit") or {}).get(
            "intent_retrieval_cache" if target_search else "retrieval_cache"
        )
        require(
            isinstance(search_cache_audit, dict),
            f"Search audit did not include the closed retrieval cache card: {search}",
        )
        require(
            search_cache_audit.get("status")
            in {"hit", "miss", "poison", "unavailable"}
            and search_cache_audit.get("gray_zone_input_modified") is False
            and search_cache_audit.get("gray_zone_model_call_count") == 0,
            f"Search retrieval cache card violated the evidence/gray boundary: {search_cache_audit}",
        )
        if not target_search:
            require(
                any(
                    ((item.get("metadata") or {}).get("rq"))
                    for item in search.get("results", [])
                ),
                f"Search results did not include RQ candidate metrics: {search}",
            )
        trace = client.request_json("GET", f"/retrieval-traces/{trace_id}/graph-steps")
        require(trace.get("steps"), f"Retrieval trace has no steps: {trace}")
        if target_search:
            require(
                trace.get("contract_version")
                == "intent_execution_retrieval_trace_public_v1"
                and trace.get("entry_layer") == search.get("entry_layer")
                and trace.get("retrieval_mode")
                == "intent_execution_retrieval_v1",
                f"Target Search trace protocol or entry layer differs: {trace}",
            )
            result_ids = {
                str(item.get("chunk_id") or "")
                for item in search.get("results") or []
            }
            path_ids = {
                str(item.get("chunk_id") or item.get("node_id") or "")
                for item in trace.get("path_labels") or []
                if item.get("path") and item.get("root_node_id")
            }
            floor_ids = set(
                (trace.get("topk_selection") or {}).get(
                    "channel_floor_chunk_ids"
                )
                or []
            )
            require(
                result_ids
                and result_ids <= path_ids
                and floor_ids <= result_ids,
                "Target Search result or channel-floor candidate lacks a graph path label",
            )
            require(
                all(
                    step.get("action")
                    in {"fuse_and_traverse", "restore_context_package"}
                    and step.get("cycle_distance_reward", 0) == 0
                    for step in trace.get("steps") or []
                ),
                "Target Search trace contains a non-graph or rewarded step",
            )
            rq_seed_counts = {
                "graph_path_labels": len(path_ids),
                "channel_floor_chunks": len(floor_ids),
            }
        else:
            require(
                not any(
                    step.get("layer") == "fine"
                    for step in trace.get("steps", [])
                ),
                f"Retrieval trace still exposes RQ prefix as an active traversal layer: {trace}",
            )
            rq_seed_counts = validate_retrieval_rq_seed_diagnostics(trace)
            require(
                any(
                    step.get("layer") == "chunk"
                    and step.get("action") == "walk_graph_frontier"
                    for step in trace.get("steps", [])
                ),
                f"Retrieval trace has no active chunk frontier walk: {trace}",
            )
        search_package = client.request_json(
            "GET", f"/context-packages/{search_context_package_id}"
        )
        require(
            search_package.get("retrieval_trace_id") == trace_id,
            f"Search Context Package is not bound to its retrieval trace: {search_package}",
        )
        require(
            search_package.get("citation_spans"),
            f"Search Context Package has no raw citation spans: {search_package}",
        )
        require(
            int(search_package.get("token_count") or 0)
            <= int(search_package.get("token_budget") or 0),
            f"Search Context Package exceeded its hard token budget: {search_package}",
        )
        payload["checks"].append(
            {
                "name": "layered_search",
                "pass": True,
                "trace_id": trace_id,
                "context_package_id": search_context_package_id,
                "result_count": len(search.get("results", [])),
                "retrieval_cache_status": search_cache_audit.get("status"),
                "rq_seed_counts": rq_seed_counts,
            }
        )

        qa = client.request_json(
            "POST",
            "/qa",
            qa_request,
            timeout=float(args.qa_timeout_seconds),
        )
        qa_acceptance = validate_qa_acceptance_payload(qa)
        if qa_acceptance["context_package_required"]:
            require(qa.get("context_package_id"), f"QA did not return context_package_id: {qa}")
        qa_traces, terminal_audit = load_qa_retrieval_traces(
            client, qa, qa_acceptance, knowledge_base_id=knowledge_base_id,
        )
        payload["checks"].append(terminal_audit)
        smoke_traces = [trace, *qa_traces]
        gray_zone_trace_audit = (
            audit_target_gray_zone_traces(
                smoke_traces,
                require_gray_coverage=bool(args.require_gray_coverage),
            )
            if all(
                item.get("retrieval_mode")
                == "intent_execution_retrieval_v1"
                for item in smoke_traces
            )
            else audit_gray_zone_traces(
                smoke_traces,
                require_gray_coverage=bool(args.require_gray_coverage),
            )
        )
        require(
            bool(gray_zone_trace_audit["pass"]),
            f"Gray-zone persisted trace audit failed: {gray_zone_trace_audit}",
        )
        payload["gray_zone_trace_audit"] = gray_zone_trace_audit
        payload["checks"].append(
            {
                "name": "gray_zone_zero_llm_audit",
                "pass": True,
                "status": gray_zone_trace_audit["status"],
                "gray_zone_coverage": gray_zone_trace_audit["gray_zone_coverage"],
                "gray_zone_coverage_required": bool(args.require_gray_coverage),
                "trace_count": gray_zone_trace_audit["trace_count"],
                "gray_rule_record_count": gray_zone_trace_audit["gray_rule_record_count"],
                "red_partition_record_count": gray_zone_trace_audit["red_partition_record_count"],
                "hard_stop_partition_record_count": gray_zone_trace_audit["hard_stop_partition_record_count"],
                "determinism": gray_zone_trace_audit["determinism"],
            }
        )
        qa_audit = qa_acceptance["model_audit"]
        pass_rate = qa_acceptance["citation_verification_pass_rate"]
        package = {}
        if qa.get("context_package_id"):
            package = client.request_json(
                "GET", f"/context-packages/{qa['context_package_id']}"
            )
            require(
                package.get("citation_spans"),
                f"Context package has no citation spans: {package}",
            )
        payload["checks"].append(
            {
                "name": "qa_context_package",
                "pass": True,
                "run_id": qa.get("run_id"),
                "answer_session_id": qa_audit.get("answer_session_id"),
                "retrieval_trace_id": qa.get("retrieval_trace_id"),
                "context_package_id": qa.get("context_package_id"),
                "returned_citation_count": qa_acceptance[
                    "returned_citation_count"
                ],
                "persisted_citation_span_count": len(
                    package.get("citation_spans") or []
                ),
                "citation_verification_pass_rate": pass_rate,
                "source_binding_pass_rate": qa_acceptance.get("source_binding_pass_rate"),
                "grounding_outcome": qa_audit.get("grounding_outcome"),
                "insufficient_evidence": qa_acceptance[
                    "insufficient_evidence"
                ],
                "evidence_gate_blocked": qa_acceptance[
                    "evidence_gate_blocked"
                ],
            }
        )
        payload["pass"] = True
    except Exception as exc:
        payload["pass"] = False
        payload["error_type"] = type(exc).__name__
        payload["error_code"] = exc.error_code if isinstance(exc, SmokeTransportError) else "preflight_or_acceptance_check_failed"
        report = write_report(payload)
        print(json.dumps({"output": str(report), "pass": False, "error_type": payload["error_type"], "error_code": payload["error_code"]}, ensure_ascii=False), file=sys.stderr)
        return 1
    report = write_report(payload)
    print(json.dumps({"output": str(report), "pass": True, "checks": len(payload["checks"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
