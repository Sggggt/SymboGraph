from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from _context_graph_maintenance import write_report


DEFAULT_QUESTIONS = [
    "平面图有哪些性质",
    "在平面网络中边数m和面数f满足什么不等式关系",
    "Gamma指数和alpha指数是什么",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L13 retrieval granularity acceptance against the Docker API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--retrieval-granularity", choices=("mid", "coarse"), default="mid")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--qa-timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def get_json(base_url: str, path: str, timeout: float = 180.0) -> dict[str, Any]:
    with urllib.request.urlopen(api_url(base_url, path), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url(base_url, path),
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


def trace_actions(trace: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(step.get("layer")), str(step.get("action_type") or step.get("action")))
        for step in trace.get("steps") or []
    }


def trace_summary(trace: dict[str, Any], mode: str) -> dict[str, Any]:
    stage_queues = trace.get("stage_queues") or {}
    candidate_pools = trace.get("candidate_pools") or {}
    topk_selection = trace.get("topk_selection") or {}
    actions = trace_actions(trace)
    required_actions = {
        ("coarse", "select_entry_nodes"),
        ("coarse", "staged_priority_queue_walk"),
        ("mid", "drill_down_each_coarse_or_direct_mid_entry"),
        ("chunk", "select_seeds_from_mid_rq_membership"),
        ("chunk", "walk_graph_frontier"),
    }
    return {
        "trace_id": trace.get("trace_id"),
        "retrieval_granularity": trace.get("retrieval_granularity"),
        "result_chunk_count": len(trace.get("result_chunk_ids") or []),
        "coarse_skipped_by_granularity": (stage_queues.get("coarse") or {}).get("skipped_by_granularity"),
        "mid_direct_entry_count": len(((candidate_pools.get("mid_direct_entries") or {}).get("selected_ids") or [])),
        "mid_entry_mode": (topk_selection.get("mid") or {}).get("entry_mode"),
        "missing_actions": sorted(f"{layer}/{action}" for layer, action in required_actions - actions),
        "mode_checks": {
            "trace_mode_matches": trace.get("retrieval_granularity") == mode,
            "mid_skipped_coarse": mode != "mid" or (stage_queues.get("coarse") or {}).get("skipped_by_granularity") == "mid",
            "mid_direct_entries": mode != "mid" or bool(((candidate_pools.get("mid_direct_entries") or {}).get("selected_ids") or [])),
            "mid_entry_mode": (topk_selection.get("mid") or {}).get("entry_mode") == mode,
            "required_actions_present": not (required_actions - actions),
        },
    }


def citation_verdicts(response: dict[str, Any]) -> list[str]:
    verdicts: list[str] = []
    for citation in response.get("citations") or []:
        verification = citation.get("verification") or {}
        if verification.get("verdict"):
            verdicts.append(str(verification["verdict"]))
    return verdicts


def run_question(args: argparse.Namespace, question: str) -> dict[str, Any]:
    payload = {
        "knowledge_base_id": args.knowledge_base_id,
        "query": question,
        "question": question,
        "top_k": args.top_k,
        "filters": {},
        "history": [],
        "retrieval_granularity": args.retrieval_granularity,
    }
    search = post_json(
        args.base_url,
        "/search/graph-enhanced",
        {
            "knowledge_base_id": payload["knowledge_base_id"],
            "query": question,
            "top_k": payload["top_k"],
            "filters": payload["filters"],
            "retrieval_granularity": payload["retrieval_granularity"],
        },
        timeout=300.0,
    )
    search_audit = search.get("model_audit") or {}
    search_trace_id = search_audit.get("retrieval_trace_id")
    search_trace = get_json(args.base_url, f"/retrieval-traces/{urllib.parse.quote(str(search_trace_id))}/graph-steps") if search_trace_id else {}

    qa = post_json(
        args.base_url,
        "/qa",
        {
            "knowledge_base_id": payload["knowledge_base_id"],
            "question": question,
            "top_k": payload["top_k"],
            "filters": payload["filters"],
            "history": payload["history"],
            "retrieval_granularity": payload["retrieval_granularity"],
        },
        timeout=args.qa_timeout_seconds,
    )
    qa_audit = qa.get("model_audit") or qa.get("answer_model_audit") or {}
    qa_trace_id = qa.get("retrieval_trace_id") or qa_audit.get("retrieval_trace_id")
    qa_trace = get_json(args.base_url, f"/retrieval-traces/{urllib.parse.quote(str(qa_trace_id))}/graph-steps") if qa_trace_id else {}
    package_id = qa.get("context_package_id") or qa_audit.get("context_package_id")
    package = get_json(args.base_url, f"/context-packages/{urllib.parse.quote(str(package_id))}") if package_id else {}

    search_summary = trace_summary(search_trace, args.retrieval_granularity)
    qa_summary = trace_summary(qa_trace, args.retrieval_granularity)
    package_diagnostics = package.get("diagnostics") or {}
    row_checks = {
        "search_results": bool(search.get("results")),
        "search_mode": (search.get("retrieval_granularity") or search_audit.get("retrieval_granularity")) == args.retrieval_granularity,
        "search_trace_mode": all(search_summary["mode_checks"].values()),
        "qa_answer": bool(qa.get("answer")),
        "qa_citations": bool(qa.get("citations")),
        "qa_context_package": bool(package_id and package.get("contexts")),
        "qa_mode": (qa.get("retrieval_granularity") or qa_audit.get("retrieval_granularity")) == args.retrieval_granularity,
        "qa_trace_mode": all(qa_summary["mode_checks"].values()),
        "package_mode": package_diagnostics.get("retrieval_granularity") == args.retrieval_granularity,
        "not_degraded": not bool(search.get("degraded_mode")) and not bool(qa.get("degraded_mode")),
    }
    return {
        "question": question,
        "checks": row_checks,
        "pass": all(row_checks.values()),
        "search": {
            "result_count": len(search.get("results") or []),
            "retrieval_granularity": search.get("retrieval_granularity") or search_audit.get("retrieval_granularity"),
            "retrieval_trace_id": search_trace_id,
            "trace_summary": search_summary,
            "top_chunks": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "document_title": item.get("document_title"),
                    "score": item.get("score"),
                    "snippet": item.get("snippet"),
                }
                for item in (search.get("results") or [])[:3]
            ],
        },
        "qa": {
            "answer_length": len(qa.get("answer") or ""),
            "answer_preview": (qa.get("answer") or "")[:500],
            "citation_count": len(qa.get("citations") or []),
            "citation_verdicts": citation_verdicts(qa),
            "retrieval_granularity": qa.get("retrieval_granularity") or qa_audit.get("retrieval_granularity"),
            "retrieval_trace_id": qa_trace_id,
            "context_package_id": package_id,
            "trace_summary": qa_summary,
            "context_count": len(package.get("contexts") or []),
            "citation_span_count": len(package.get("citation_spans") or []),
            "package_diagnostics": package_diagnostics,
        },
    }


def main() -> None:
    args = parse_args()
    questions = args.question or DEFAULT_QUESTIONS
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "script": "run_l13_retrieval_granularity_acceptance",
        "base_url": args.base_url,
        "knowledge_base_id": args.knowledge_base_id,
        "retrieval_granularity": args.retrieval_granularity,
        "questions": questions,
        "rows": rows,
    }
    try:
        rows.extend(run_question(args, question) for question in questions)
        checks = {
            "all_questions_passed": all(row["pass"] for row in rows),
            "all_search_traces_mid_direct": all(row["search"]["trace_summary"]["mode_checks"]["mid_direct_entries"] for row in rows)
            if args.retrieval_granularity == "mid"
            else True,
            "all_qa_context_packages": all(row["qa"]["context_package_id"] and row["qa"]["context_count"] > 0 for row in rows),
            "all_qa_citations": all(row["qa"]["citation_count"] > 0 for row in rows),
            "no_degraded_mode": all(row["checks"]["not_degraded"] for row in rows),
        }
        payload["checks"] = checks
        payload["pass"] = all(checks.values())
    except Exception as exc:
        payload["pass"] = False
        payload["error"] = str(exc)
        report = write_report("l13_retrieval_granularity_acceptance", payload)
        print(json.dumps({"output": str(report), "pass": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1) from exc

    report = write_report("l13_retrieval_granularity_acceptance", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "checks": payload.get("checks")}, ensure_ascii=False), flush=True)
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
