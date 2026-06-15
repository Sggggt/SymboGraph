from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


REQUIRED_TRACE_NODES = {
    "agent_planner",
    "typed_action_validation",
    "entry_selection",
    "layer_drilldown",
    "frontier_traversal",
    "chunk_recall",
    "structure_context_restoration",
    "context_package",
    "citation_verification",
    "reward_event",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real QA requests and verify the Layered P&E Agent trace.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {body}") from exc


def resolve_kb_id(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.knowledge_base_id:
        return args.knowledge_base_id, args.knowledge_base_name
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_name=args.knowledge_base_name)
        return knowledge_base.id, knowledge_base.name


def main() -> None:
    args = parse_args()
    knowledge_base_id, knowledge_base_name = resolve_kb_id(args)
    questions = args.question or [
        "Explain the Metropolis-Hastings acceptance probability using the indexed material.",
        "How do posterior, prior, and likelihood relate in Bayesian inference?",
    ]
    rows: list[dict[str, Any]] = []
    for question in questions:
        response = post_json(
            args.base_url,
            "/qa",
            {"question": question, "knowledge_base_id": knowledge_base_id, "top_k": args.top_k, "filters": {}, "history": []},
        )
        nodes = {str(item.get("node")) for item in response.get("trace") or [] if item.get("node")}
        model_audit = response.get("model_audit") or response.get("answer_model_audit") or {}
        rows.append(
            {
                "question": question,
                "run_id": response.get("run_id"),
                "context_package_id": response.get("context_package_id"),
                "retrieval_trace_id": response.get("retrieval_trace_id"),
                "answer_length": len(response.get("answer") or ""),
                "citation_count": len(response.get("citations") or []),
                "trace_nodes": sorted(nodes),
                "missing_trace_nodes": sorted(REQUIRED_TRACE_NODES - nodes),
                "citation_verification_pass_rate": model_audit.get("citation_verification_pass_rate"),
                "degraded_mode": bool(response.get("degraded_mode")),
            }
        )

    checks = {
        "answers_returned": all(row["answer_length"] > 0 for row in rows),
        "context_packages_returned": all(row["context_package_id"] and row["retrieval_trace_id"] for row in rows),
        "citations_returned": all(row["citation_count"] > 0 for row in rows),
        "required_trace_nodes_present": all(not row["missing_trace_nodes"] for row in rows),
        "citation_verification_positive": all((row["citation_verification_pass_rate"] or 0) > 0 for row in rows),
        "not_degraded": not any(row["degraded_mode"] for row in rows),
    }
    payload = {
        "script": "evaluate_agent_trace",
        "base_url": args.base_url,
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "checks": checks,
        "pass": all(checks.values()),
        "rows": rows,
    }
    report = write_report("evaluate_agent_trace", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "checks": checks}, ensure_ascii=False, default=str))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
