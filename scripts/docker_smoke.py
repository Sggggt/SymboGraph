from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from urllib import error, parse, request


TERMINAL_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped"}


@dataclass
class ApiClient:
    base_url: str
    api_key: str | None = None

    def url(self, path: str, params: dict[str, str | None] | None = None) -> str:
        url = f"{self.base_url.rstrip('/')}{path}"
        values = {key: value for key, value in (params or {}).items() if value}
        if values:
            url = f"{url}?{parse.urlencode(values)}"
        return url

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | None] | None = None,
        payload: dict | None = None,
        expected_status: int | None = None,
        timeout_seconds: int = 60,
    ) -> dict | list:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = request.Request(self.url(path, params), data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                status = response.status
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} returned {exc.code}: {raw}") from exc
        if expected_status is not None and status != expected_status:
            raise RuntimeError(f"{method} {path} returned {status}, expected {expected_status}: {raw}")
        return json.loads(raw) if raw else {}

    def upload_file(self, path: str, *, knowledge_base_id: str, filename: str, content: bytes) -> dict:
        boundary = f"----KnowledgeBase-kg-smoke-{uuid.uuid4().hex}"
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="upload"; filename="{filename}"\r\n'.encode(),
                b"Content-Type: text/markdown\r\n\r\n",
                content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = request.Request(self.url(path, {"knowledge_base_id": knowledge_base_id}), data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {path} upload returned {exc.code}: {raw}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wait_for_batch(client: ApiClient, batch_id: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        last = client.request_json("GET", f"/ingestion/batches/{batch_id}")
        state = str(last.get("state"))
        if state in TERMINAL_BATCH_STATES:
            return last
        time.sleep(2)
    raise RuntimeError(f"Batch {batch_id} did not finish within {timeout_seconds}s; last={last}")


def check_worker_container(container_name: str) -> dict:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Worker container {container_name} is not inspectable: {result.stderr.strip()}")
    parts = result.stdout.strip().split()
    status = parts[0] if parts else ""
    health = parts[1] if len(parts) > 1 else "unknown"
    require(status == "running", f"Worker container {container_name} is not running: status={status} health={health}")
    require(health in {"healthy", "no-healthcheck"}, f"Worker container {container_name} is not healthy: {health}")
    return {"container": container_name, "status": status, "health": health}


def _docker_exec_api(container_name: str, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", "-w", "/app/apps/api", container_name, *command],
        check=False,
        capture_output=True,
        text=True,
    )


def _alembic_revisions(output: str) -> set[str]:
    revisions: set[str] = set()
    for line in output.splitlines():
        match = re.match(r"^\s*([0-9A-Za-z_]+)", line)
        if match:
            revisions.add(match.group(1))
    return revisions


def check_alembic_current(container_name: str) -> dict:
    current = _docker_exec_api(container_name, "alembic", "current")
    if current.returncode != 0:
        raise RuntimeError(f"alembic current failed in {container_name}: {current.stderr.strip()}")
    heads = _docker_exec_api(container_name, "alembic", "heads")
    if heads.returncode != 0:
        raise RuntimeError(f"alembic heads failed in {container_name}: {heads.stderr.strip()}")
    current_revisions = _alembic_revisions(current.stdout)
    head_revisions = _alembic_revisions(heads.stdout)
    require(head_revisions, f"alembic heads returned no revisions: {heads.stdout!r}")
    require(
        bool(current_revisions.intersection(head_revisions)),
        f"Database migration is not at head: current={current.stdout.strip()!r} heads={heads.stdout.strip()!r}",
    )
    return {
        "container": container_name,
        "current": sorted(current_revisions),
        "heads": sorted(head_revisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Docker API chain with a disposable KnowledgeBase.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--skip-model-calls", action="store_true", help="Only check infrastructure and read-only endpoints.")
    parser.add_argument("--worker-container", default="course-kg-worker")
    parser.add_argument("--skip-worker-check", action="store_true")
    parser.add_argument("--api-container", default="course-kg-api")
    parser.add_argument("--skip-migration-check", action="store_true")
    args = parser.parse_args()

    client = ApiClient(args.base_url, args.api_key or None)
    created_knowledge_base_id: str | None = None
    knowledge_base_name = f"docker-smoke-{uuid.uuid4().hex[:10]}"

    try:
        health = client.request_json("GET", "/health")
        require(health.get("status") == "ok", f"Unexpected health payload: {health}")

        runtime = client.request_json("GET", "/settings/runtime-check")
        require(not runtime.get("blocking_issues"), f"Runtime check has blocking issues: {runtime.get('blocking_issues')}")
        infra = runtime.get("infrastructure") or {}
        for key in ("postgres", "qdrant", "redis"):
            require(infra.get(key) is True, f"{key} is not reachable from the API runtime")
        worker_status = None
        if not args.skip_worker_check:
            worker_status = check_worker_container(args.worker_container)
        migration_status = None
        if not args.skip_migration_check:
            migration_status = check_alembic_current(args.api_container)

        settings = client.request_json("GET", "/settings/model")
        if args.skip_model_calls:
            print(
                json.dumps(
                    {"ok": True, "mode": "infrastructure-only", "runtime": runtime, "worker": worker_status, "migration": migration_status},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        require(settings.get("has_api_key") is True, "Model API key is required for full no-fallback smoke")
        require(settings.get("degraded_mode") is False, f"Model settings are degraded: {settings}")

        KnowledgeBase = client.request_json(
            "POST",
            "/knowledge_bases",
            payload={"name": knowledge_base_name, "description": "temporary docker smoke KnowledgeBase"},
        )
        created_knowledge_base_id = str(KnowledgeBase["id"])

        upload = client.upload_file(
            "/files/upload",
            knowledge_base_id=created_knowledge_base_id,
            filename="centrality-smoke.md",
            content=(
                b"# Centrality smoke test\n\n"
                b"Degree centrality counts incident edges in a graph. "
                b"It is a local network-analysis measure used to compare node prominence.\n"
            ),
        )
        source_path = str(upload["source_path"])

        batch = client.request_json(
            "POST",
            "/ingestion/parse-uploaded-files",
            params={"knowledge_base_id": created_knowledge_base_id},
            payload={"file_paths": [source_path], "force": True},
        )
        batch_status = wait_for_batch(client, str(batch["batch_id"]), args.timeout_seconds)
        require(batch_status.get("state") in {"completed", "partial_failed"}, f"Ingestion failed: {batch_status}")
        require(int(batch_status.get("success_count") or 0) >= 1, f"No file was ingested: {batch_status}")
        graph = client.request_json(
            "GET",
            "/knowledge_bases/current/graph",
            params={"knowledge_base_id": created_knowledge_base_id, "graph_type": "evidence"},
        )
        require(graph.get("signal_layer_complete") is True, f"Signal layer was not complete: {graph}")
        require(graph.get("signal_layer_status") == "active", f"Signal layer was not active: {graph}")
        signal_nodes = [node for node in graph.get("nodes", []) if node.get("category") == "signal_node"]
        require(signal_nodes, f"Signal layer returned no signal nodes: {graph}")
        require(
            all(node.get("support_atom_ids") and (node.get("source_span_union") or {}).get("spans") for node in signal_nodes),
            f"Signal nodes are missing evidence support: {signal_nodes}",
        )

        files = client.request_json("GET", "/knowledge-base-files", params={"knowledge_base_id": created_knowledge_base_id})
        require(isinstance(files, list) and files, "Uploaded file is not visible through /knowledge-base-files")

        search = client.request_json(
            "POST",
            "/search",
            payload={"knowledge_base_id": created_knowledge_base_id, "query": "What is degree centrality?", "top_k": 3},
        )
        require(search.get("results"), f"Search returned no results: {search}")
        first_result = search["results"][0]
        first_metadata = first_result.get("metadata") or {}
        require(first_metadata.get("quality_gate_passed") is True, f"Active chunk quality gate did not pass: {first_result}")
        if first_metadata.get("parent_chunk_id"):
            require(
                first_metadata.get("retrieval_granularity") == "child_with_parent_context",
                f"Child result did not carry parent retrieval granularity: {first_result}",
            )
            require(first_metadata.get("parent_content"), f"Child result did not include parent_content: {first_result}")
            require(first_result.get("child_content"), f"Child result did not preserve child_content: {first_result}")
        audit = search.get("model_audit") or {}
        require(audit.get("embedding_external_called") is True, f"Search did not report a real embedding call: {audit}")
        require(audit.get("embedding_fallback_reason") is None, f"Search used fallback: {audit}")
        require(audit.get("signal_state_hash"), f"Search audit did not include signal state hash: {audit}")
        require(audit.get("retrieval_cache_scope_hash"), f"Search audit did not include retrieval cache scope hash: {audit}")

        qa = client.request_json(
            "POST",
            "/qa",
            payload={"knowledge_base_id": created_knowledge_base_id, "question": "What is degree centrality?", "top_k": 3},
            timeout_seconds=args.timeout_seconds,
        )
        require(qa.get("answer"), f"QA returned no answer: {qa}")
        require(qa.get("citations"), f"QA returned no citations: {qa}")
        qa_audit = qa.get("answer_model_audit") or {}
        require(qa_audit.get("signal_state_hash"), f"QA audit did not include signal state hash: {qa_audit}")

        # ------------------------------------------------------------------
        # Chinese-language smoke path (upload → search → QA)
        # ------------------------------------------------------------------
        upload_cn = client.upload_file(
            "/files/upload",
            knowledge_base_id=created_knowledge_base_id,
            filename="centrality-chinese-smoke.md",
            content=(
                "# 度中心性\n\n"
                "度中心性是图论和网络分析中的一个基本概念。"
                "它通过计算与某个节点直接相连的边的数量来衡量该节点的重要性。"
                "在社会网络中，度中心性高的个体通常拥有更多的直接联系。\n"
            ).encode("utf-8"),
        )
        source_path_cn = str(upload_cn["source_path"])

        batch_cn = client.request_json(
            "POST",
            "/ingestion/parse-uploaded-files",
            params={"knowledge_base_id": created_knowledge_base_id},
            payload={"file_paths": [source_path_cn], "force": True},
        )
        batch_status_cn = wait_for_batch(client, str(batch_cn["batch_id"]), args.timeout_seconds)
        require(batch_status_cn.get("state") in {"completed", "partial_failed"}, f"Chinese ingestion failed: {batch_status_cn}")
        require(int(batch_status_cn.get("success_count") or 0) >= 1, f"No Chinese file was ingested: {batch_status_cn}")
        graph_cn = client.request_json(
            "GET",
            "/knowledge_bases/current/graph",
            params={"knowledge_base_id": created_knowledge_base_id, "graph_type": "evidence"},
        )
        require(graph_cn.get("signal_layer_complete") is True, f"Chinese signal layer was not complete: {graph_cn}")
        require(graph_cn.get("signal_layer_status") == "active", f"Chinese signal layer was not active: {graph_cn}")

        search_cn = client.request_json(
            "POST",
            "/search",
            payload={"knowledge_base_id": created_knowledge_base_id, "query": "什么是度中心性", "top_k": 3},
        )
        require(search_cn.get("results"), f"Chinese search returned no results: {search_cn}")
        audit_cn = search_cn.get("model_audit") or {}
        require(audit_cn.get("embedding_external_called") is True, f"Chinese search did not report a real embedding call: {audit_cn}")
        require(audit_cn.get("embedding_fallback_reason") is None, f"Chinese search used fallback: {audit_cn}")
        require(audit_cn.get("signal_state_hash"), f"Chinese search audit did not include signal state hash: {audit_cn}")

        qa_cn = client.request_json(
            "POST",
            "/qa",
            payload={"knowledge_base_id": created_knowledge_base_id, "question": "什么是度中心性？", "top_k": 3},
            timeout_seconds=args.timeout_seconds,
        )
        require(qa_cn.get("answer"), f"Chinese QA returned no answer: {qa_cn}")
        require(qa_cn.get("citations"), f"Chinese QA returned no citations: {qa_cn}")
        qa_cn_audit = qa_cn.get("answer_model_audit") or {}
        require(qa_cn_audit.get("signal_state_hash"), f"Chinese QA audit did not include signal state hash: {qa_cn_audit}")

        session_id = str(qa["session_id"])
        messages = client.request_json("GET", f"/sessions/{session_id}/messages")
        require(messages.get("messages"), f"Session messages are empty: {messages}")
        client.request_json("DELETE", f"/sessions/{session_id}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "knowledge_base_id": created_knowledge_base_id,
                    "batch": batch_status,
                    "signal_layer": {
                        "status": graph.get("signal_layer_status"),
                        "complete": graph.get("signal_layer_complete"),
                        "nodes": len(signal_nodes),
                        "edges": graph.get("edge_counts", {}).get("signal_projection", 0),
                        "model_batches": (batch_status.get("graph_stats") or {}).get("signal_model_batches_used"),
                        "model_batch_budget": (batch_status.get("graph_stats") or {}).get("signal_model_batch_budget"),
                        "token_estimate": (batch_status.get("graph_stats") or {}).get("signal_estimated_tokens"),
                        "deterministic_fallback_ratio": (batch_status.get("graph_stats") or {}).get(
                            "signal_deterministic_fallback_ratio"
                        ),
                    },
                    "active_chunk_quality": {
                        "active_chunk_id": first_metadata.get("active_chunk_id"),
                        "quality_decision_id": first_metadata.get("quality_decision_id"),
                        "quality_gate_passed": first_metadata.get("quality_gate_passed"),
                        "quality_action": first_metadata.get("quality_action"),
                        "selected_candidate_generator": first_metadata.get("selected_candidate_generator"),
                    },
                    "worker": worker_status,
                    "migration": migration_status,
                    "search_model_audit": audit,
                    "qa_answer_model_audit": qa_audit,
                    "qa_run_id": qa.get("run_id"),
                    "chinese_search_model_audit": audit_cn,
                    "chinese_qa_answer_model_audit": qa_cn_audit,
                    "chinese_qa_run_id": qa_cn.get("run_id"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if created_knowledge_base_id:
            try:
                client.request_json("DELETE", f"/knowledge_bases/{created_knowledge_base_id}")
            except Exception as exc:
                print(f"cleanup_failed: {exc}")


if __name__ == "__main__":
    main()
