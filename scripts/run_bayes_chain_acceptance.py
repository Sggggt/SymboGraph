from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from _context_graph_maintenance import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real API search and QA acceptance checks for a knowledge base.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--search-query", action="append", default=[])
    parser.add_argument("--qa-question", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {body}") from exc


def trace_nodes(response: dict[str, Any]) -> set[str]:
    return {str(item.get("node")) for item in response.get("trace") or [] if item.get("node")}


def citation_verdicts(response: dict[str, Any]) -> list[str]:
    verdicts: list[str] = []
    for citation in response.get("citations") or []:
        verification = citation.get("verification") or {}
        verdict = verification.get("verdict")
        if verdict:
            verdicts.append(str(verdict))
    return verdicts


def main() -> None:
    args = parse_args()
    search_queries = args.search_query or [
        "What is the Metropolis Hastings acceptance probability?",
        "贝叶斯后验分布和先验似然的关系是什么？",
    ]
    qa_questions = args.qa_question or [
        "Explain the Metropolis-Hastings acceptance probability using the course material.",
        "请用资料库说明贝叶斯后验分布与先验、似然之间的关系，并给出引用。",
    ]

    search_rows: list[dict[str, Any]] = []
    for query in search_queries:
        response = post_json(
            args.base_url,
            "/search/graph-enhanced",
            {"query": query, "knowledge_base_id": args.knowledge_base_id, "top_k": args.top_k, "filters": {}},
        )
        audit = response.get("model_audit") or {}
        details = audit.get("details") or audit
        search_rows.append(
            {
                "query": query,
                "result_count": len(response.get("results") or []),
                "retrieval_trace_id": audit.get("retrieval_trace_id") or details.get("retrieval_trace_id"),
                "retrieval_pipeline": details.get("retrieval_pipeline") or audit.get("retrieval_pipeline"),
                "degraded_mode": bool(response.get("degraded_mode")),
                "top_chunk_ids": [item.get("chunk_id") for item in (response.get("results") or [])[:5]],
                "audit": audit,
            }
        )

    qa_rows: list[dict[str, Any]] = []
    required_nodes = {"agent_planner", "typed_action_validation", "context_package", "citation_verification", "reward_event"}
    for question in qa_questions:
        response = post_json(
            args.base_url,
            "/qa",
            {"question": question, "knowledge_base_id": args.knowledge_base_id, "top_k": args.top_k, "filters": {}, "history": []},
            timeout=300,
        )
        nodes = trace_nodes(response)
        model_audit = response.get("model_audit") or response.get("answer_model_audit") or {}
        qa_rows.append(
            {
                "question": question,
                "answer_length": len(response.get("answer") or ""),
                "citation_count": len(response.get("citations") or []),
                "citation_verdicts": citation_verdicts(response),
                "context_package_id": response.get("context_package_id"),
                "retrieval_trace_id": response.get("retrieval_trace_id"),
                "run_id": response.get("run_id"),
                "trace_nodes": sorted(nodes),
                "missing_required_trace_nodes": sorted(required_nodes - nodes),
                "citation_verification_pass_rate": model_audit.get("citation_verification_pass_rate"),
                "degraded_mode": bool(response.get("degraded_mode")),
            }
        )

    checks = {
        "search_returns_results": all(row["result_count"] > 0 for row in search_rows),
        "search_uses_layered_trace": all(row["retrieval_trace_id"] for row in search_rows),
        "search_not_degraded": not any(row["degraded_mode"] for row in search_rows),
        "qa_returns_answers": all(row["answer_length"] > 0 for row in qa_rows),
        "qa_has_context_packages": all(row["context_package_id"] and row["retrieval_trace_id"] for row in qa_rows),
        "qa_has_citations": all(row["citation_count"] > 0 for row in qa_rows),
        "qa_required_trace_nodes": all(not row["missing_required_trace_nodes"] for row in qa_rows),
        "qa_citation_verification": all((row["citation_verification_pass_rate"] or 0) > 0 for row in qa_rows),
        "qa_not_degraded": not any(row["degraded_mode"] for row in qa_rows),
    }
    payload = {
        "script": "run_bayes_chain_acceptance",
        "base_url": args.base_url,
        "knowledge_base_id": args.knowledge_base_id,
        "checks": checks,
        "pass": all(checks.values()),
        "search": search_rows,
        "qa": qa_rows,
    }
    report = write_report("bayes_chain_acceptance", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "checks": checks}, ensure_ascii=False))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
