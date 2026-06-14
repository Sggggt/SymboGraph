from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"


class SmokeClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
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
                data = response.read().decode("utf-8")
                return json.loads(data) if data else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"{method} {url} timed out after {timeout or self.timeout:.1f}s") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    parser.add_argument("--wait-batch-id")
    parser.add_argument("--wait-timeout-seconds", type=int, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--qa-timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def wait_for_batch(client: SmokeClient, batch_id: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        batch = client.request_json("GET", f"/ingestion/batches/{batch_id}")
        if batch.get("state") in {"completed", "partial_failed", "failed", "skipped", "cancelled", "cancel_failed"}:
            return batch
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for batch {batch_id}")


def main() -> int:
    args = parse_args()
    client = SmokeClient(args.base_url, timeout=float(args.request_timeout_seconds))
    payload: dict = {"base_url": args.base_url, "checks": []}
    try:
        health = client.request_json("GET", "/health")
        require(health.get("status") == "ok", f"Health check failed: {health}")
        payload["checks"].append({"name": "health", "pass": True, "payload": health})

        knowledge_bases = client.request_json("GET", "/knowledge_bases")
        require(isinstance(knowledge_bases, list) and knowledge_bases, "No knowledge bases returned")
        selected = None
        if args.knowledge_base_id:
            selected = next((item for item in knowledge_bases if item.get("id") == args.knowledge_base_id), None)
            require(selected is not None, f"Knowledge base not found: {args.knowledge_base_id}")
        else:
            selected = next(
                (
                    item
                    for item in knowledge_bases
                    if int(item.get("active_chunk_count") or item.get("chunk_count") or 0) > 0
                ),
                knowledge_bases[0],
            )
        knowledge_base_id = selected["id"]
        payload["knowledge_base_id"] = knowledge_base_id
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
            if graph_type == "chunk-relation":
                rq_edges = [edge for edge in graph.get("edges", []) if str(edge.get("type") or "").startswith("rq_")]
                rq_chunk_nodes = [node for node in graph.get("nodes", []) if ((node.get("metadata") or {}).get("rq_path"))]
                rq_prefix_nodes = [node for node in graph.get("nodes", []) if node.get("category") == "rq_prefix"]
                rq_membership_edges = [edge for edge in graph.get("edges", []) if edge.get("category") == "rq_membership"]
                rq_cluster_edges = [edge for edge in graph.get("edges", []) if str(edge.get("category") or "").startswith("rq_") and edge.get("category") != "rq_membership"]
                require(rq_edges, f"Chunk relation graph has no RQ edges: {graph}")
                require(rq_chunk_nodes, f"Chunk relation graph has no chunk RQ path metadata: {graph}")
                require(rq_prefix_nodes, f"Chunk relation graph has no visible RQ prefix nodes: {graph}")
                require(rq_membership_edges, f"Chunk relation graph has no visible RQ membership edges: {graph}")
                require(rq_cluster_edges, f"Chunk relation graph has no visible RQ cluster edges: {graph}")
            payload["checks"].append({"name": f"graph_{graph_type}", "pass": True, "nodes": len(graph.get("nodes", [])), "edges": len(graph.get("edges", []))})

        search = client.request_json("POST", "/search", {"knowledge_base_id": knowledge_base_id, "query": args.query, "top_k": 5, "filters": {}})
        require(search.get("results"), f"Search returned no results: {search}")
        trace_id = (search.get("model_audit") or {}).get("retrieval_trace_id")
        require(bool(trace_id), f"Search did not record retrieval_trace_id: {search}")
        require((search.get("model_audit") or {}).get("query_rq_path"), f"Search audit did not include query RQ path: {search}")
        require(any(((item.get("metadata") or {}).get("rq")) for item in search.get("results", [])), f"Search results did not include RQ candidate metrics: {search}")
        trace = client.request_json("GET", f"/retrieval-traces/{trace_id}/graph-steps")
        require(trace.get("steps"), f"Retrieval trace has no steps: {trace}")
        fine_steps = [step for step in trace.get("steps", []) if step.get("layer") == "fine"]
        require(fine_steps and (fine_steps[0].get("input") or {}).get("query_rq_path"), f"Retrieval trace has no RQ query path: {trace}")
        payload["checks"].append({"name": "layered_search", "pass": True, "trace_id": trace_id, "result_count": len(search.get("results", []))})

        qa = client.request_json(
            "POST",
            "/qa",
            {"knowledge_base_id": knowledge_base_id, "question": args.query, "top_k": 5, "filters": {}, "history": []},
            timeout=float(args.qa_timeout_seconds),
        )
        require(qa.get("answer"), f"QA returned no answer: {qa}")
        require(qa.get("citations"), f"QA returned no citations: {qa}")
        require(qa.get("context_package_id"), f"QA did not return context_package_id: {qa}")
        require(qa.get("retrieval_trace_id"), f"QA did not return retrieval_trace_id: {qa}")
        qa_audit = qa.get("model_audit") or {}
        pass_rate = qa_audit.get("citation_verification_pass_rate")
        require(pass_rate is None or float(pass_rate) > 0.0, f"QA citation verification did not pass: {qa}")
        package = client.request_json("GET", f"/context-packages/{qa['context_package_id']}")
        require(package.get("citation_spans"), f"Context package has no citation spans: {package}")
        payload["checks"].append({"name": "qa_context_package", "pass": True, "context_package_id": qa["context_package_id"]})
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
