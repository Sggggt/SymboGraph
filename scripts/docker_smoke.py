from __future__ import annotations

import argparse
import json
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
    ) -> dict | list:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = request.Request(self.url(path, params), data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=60) as response:
                status = response.status
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} returned {exc.code}: {raw}") from exc
        if expected_status is not None and status != expected_status:
            raise RuntimeError(f"{method} {path} returned {status}, expected {expected_status}: {raw}")
        return json.loads(raw) if raw else {}

    def upload_file(self, path: str, *, course_id: str, filename: str, content: bytes) -> dict:
        boundary = f"----course-kg-smoke-{uuid.uuid4().hex}"
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
        req = request.Request(self.url(path, {"course_id": course_id}), data=body, headers=headers, method="POST")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Docker API chain with a disposable course.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--skip-model-calls", action="store_true", help="Only check infrastructure and read-only endpoints.")
    parser.add_argument("--worker-container", default="course-kg-worker")
    parser.add_argument("--skip-worker-check", action="store_true")
    args = parser.parse_args()

    client = ApiClient(args.base_url, args.api_key or None)
    created_course_id: str | None = None
    course_name = f"docker-smoke-{uuid.uuid4().hex[:10]}"

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

        settings = client.request_json("GET", "/settings/model")
        if args.skip_model_calls:
            print(json.dumps({"ok": True, "mode": "infrastructure-only", "runtime": runtime, "worker": worker_status}, ensure_ascii=False, indent=2))
            return
        require(settings.get("has_api_key") is True, "Model API key is required for full no-fallback smoke")
        require(settings.get("degraded_mode") is False, f"Model settings are degraded: {settings}")

        course = client.request_json(
            "POST",
            "/courses",
            payload={"name": course_name, "description": "temporary docker smoke course"},
        )
        created_course_id = str(course["id"])

        upload = client.upload_file(
            "/files/upload",
            course_id=created_course_id,
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
            params={"course_id": created_course_id},
            payload={"file_paths": [source_path], "force": True},
        )
        batch_status = wait_for_batch(client, str(batch["batch_id"]), args.timeout_seconds)
        require(batch_status.get("state") in {"completed", "partial_failed"}, f"Ingestion failed: {batch_status}")
        require(int(batch_status.get("success_count") or 0) >= 1, f"No file was ingested: {batch_status}")

        files = client.request_json("GET", "/course-files", params={"course_id": created_course_id})
        require(isinstance(files, list) and files, "Uploaded file is not visible through /course-files")

        search = client.request_json(
            "POST",
            "/search",
            payload={"course_id": created_course_id, "query": "What is degree centrality?", "top_k": 3},
        )
        require(search.get("results"), f"Search returned no results: {search}")
        first_result = search["results"][0]
        first_metadata = first_result.get("metadata") or {}
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

        qa = client.request_json(
            "POST",
            "/qa",
            payload={"course_id": created_course_id, "question": "What is degree centrality?", "top_k": 3},
        )
        require(qa.get("answer"), f"QA returned no answer: {qa}")
        require(qa.get("citations"), f"QA returned no citations: {qa}")

        # ------------------------------------------------------------------
        # Chinese-language smoke path (upload → search → QA)
        # ------------------------------------------------------------------
        upload_cn = client.upload_file(
            "/files/upload",
            course_id=created_course_id,
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
            params={"course_id": created_course_id},
            payload={"file_paths": [source_path_cn], "force": True},
        )
        batch_status_cn = wait_for_batch(client, str(batch_cn["batch_id"]), args.timeout_seconds)
        require(batch_status_cn.get("state") in {"completed", "partial_failed"}, f"Chinese ingestion failed: {batch_status_cn}")
        require(int(batch_status_cn.get("success_count") or 0) >= 1, f"No Chinese file was ingested: {batch_status_cn}")

        search_cn = client.request_json(
            "POST",
            "/search",
            payload={"course_id": created_course_id, "query": "什么是度中心性", "top_k": 3},
        )
        require(search_cn.get("results"), f"Chinese search returned no results: {search_cn}")
        audit_cn = search_cn.get("model_audit") or {}
        require(audit_cn.get("embedding_external_called") is True, f"Chinese search did not report a real embedding call: {audit_cn}")
        require(audit_cn.get("embedding_fallback_reason") is None, f"Chinese search used fallback: {audit_cn}")

        qa_cn = client.request_json(
            "POST",
            "/qa",
            payload={"course_id": created_course_id, "question": "什么是度中心性？", "top_k": 3},
        )
        require(qa_cn.get("answer"), f"Chinese QA returned no answer: {qa_cn}")
        require(qa_cn.get("citations"), f"Chinese QA returned no citations: {qa_cn}")

        session_id = str(qa["session_id"])
        messages = client.request_json("GET", f"/sessions/{session_id}/messages")
        require(messages.get("messages"), f"Session messages are empty: {messages}")
        client.request_json("DELETE", f"/sessions/{session_id}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "course_id": created_course_id,
                    "batch": batch_status,
                    "worker": worker_status,
                    "search_model_audit": audit,
                    "qa_run_id": qa.get("run_id"),
                    "chinese_search_model_audit": audit_cn,
                    "chinese_qa_run_id": qa_cn.get("run_id"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if created_course_id:
            try:
                client.request_json("DELETE", f"/courses/{created_course_id}")
            except Exception as exc:
                print(f"cleanup_failed: {exc}")


if __name__ == "__main__":
    main()
