from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
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
    require(
        bool(graph_ready),
        "No graph-ready knowledge base is available; pass --knowledge-base-id "
        "after building and promoting a fresh four-layer context graph",
    )
    return graph_ready[0], "first_fresh_graph_ready"


def write_report(payload: dict) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"docker_smoke_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
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
        selected, selection_reason = select_smoke_knowledge_base(
            knowledge_bases,
            requested_id=args.knowledge_base_id,
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

        stats = client.request_json("GET", f"/knowledge_bases/{knowledge_base_id}/context-graph/stats")
        require((stats.get("counts") or {}).get("active_chunks", 0) > 0, f"No current chunks: {stats}")
        require((stats.get("counts") or {}).get("chunk_relation_edges", 0) >= 0, f"Missing relation stats: {stats}")
        payload["checks"].append({"name": "context_graph_stats", "pass": True, "payload": stats})

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
                    {"output": str(report), **payload},
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
        require(
            (search.get("model_audit") or {}).get("retrieval_trace_id") == trace_id,
            f"Search trace identity is inconsistent across the public response: {search}",
        )
        search_context_package_id = search.get("context_package_id")
        require(
            bool(search_context_package_id),
            f"Ordinary search did not create a Context Package: {search}",
        )
        require(
            (search.get("model_audit") or {}).get("context_package_id")
            == search_context_package_id,
            f"Search Context Package identity is inconsistent: {search}",
        )
        require((search.get("model_audit") or {}).get("query_rq_path"), f"Search audit did not include query RQ path: {search}")
        search_cache_audit = (search.get("model_audit") or {}).get(
            "retrieval_cache"
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
        require(any(((item.get("metadata") or {}).get("rq")) for item in search.get("results", [])), f"Search results did not include RQ candidate metrics: {search}")
        trace = client.request_json("GET", f"/retrieval-traces/{trace_id}/graph-steps")
        require(trace.get("steps"), f"Retrieval trace has no steps: {trace}")
        require(
            not any(step.get("layer") == "fine" for step in trace.get("steps", [])),
            f"Retrieval trace still exposes RQ prefix as an active traversal layer: {trace}",
        )
        rq_seed_counts = validate_retrieval_rq_seed_diagnostics(trace)
        require(
            any(step.get("layer") == "chunk" and step.get("action") == "walk_graph_frontier" for step in trace.get("steps", [])),
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
        require(qa.get("answer"), f"QA returned no answer: {qa}")
        require(qa.get("citations"), f"QA returned no citations: {qa}")
        require(qa.get("context_package_id"), f"QA did not return context_package_id: {qa}")
        require(qa.get("retrieval_trace_id"), f"QA did not return retrieval_trace_id: {qa}")
        qa_trace = client.request_json("GET", f"/retrieval-traces/{qa['retrieval_trace_id']}/graph-steps")
        require(qa_trace.get("steps"), f"QA retrieval trace has no steps: {qa_trace}")
        gray_zone_trace_audit = audit_gray_zone_traces(
            [trace, qa_trace],
            require_gray_coverage=bool(args.require_gray_coverage),
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
        qa_audit = qa.get("model_audit") or {}
        pass_rate = qa_audit.get("citation_verification_pass_rate")
        require(pass_rate is None or float(pass_rate) > 0.0, f"QA citation verification did not pass: {qa}")
        package = client.request_json("GET", f"/context-packages/{qa['context_package_id']}")
        require(package.get("citation_spans"), f"Context package has no citation spans: {package}")
        payload["checks"].append(
            {
                "name": "qa_context_package",
                "pass": True,
                "run_id": qa.get("run_id"),
                "answer_session_id": qa_audit.get("answer_session_id"),
                "retrieval_trace_id": qa["retrieval_trace_id"],
                "context_package_id": qa["context_package_id"],
                "returned_citation_count": len(qa.get("citations") or []),
                "persisted_citation_span_count": len(
                    package.get("citation_spans") or []
                ),
                "citation_verification_pass_rate": pass_rate,
                "grounding_outcome": qa_audit.get("grounding_outcome"),
                "insufficient_evidence": bool(
                    qa_audit.get("insufficient_evidence")
                ),
            }
        )
        payload["pass"] = True
    except Exception as exc:
        payload["pass"] = False
        payload["error"] = str(exc)
        report = write_report(payload)
        print(json.dumps({"output": str(report), "pass": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    report = write_report(payload)
    print(json.dumps({"output": str(report), "pass": True, "checks": len(payload["checks"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
