"""Run ten frozen conversations through intent_execution_retrieval_v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


class QAHTTPError(RuntimeError):
    def __init__(self, status_code: int, payload: dict | None, body: str) -> None:
        self.status_code = int(status_code)
        self.payload = dict(payload or {})
        super().__init__(f"HTTP {status_code} for /qa: {body[:500]}")


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_gold(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("conversation_gold_requires_ten_frozen_cases")
    cases = value.get("cases")
    if value.get("frozen_before_execution") is not True or not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("conversation_gold_requires_ten_frozen_cases")
    ids = [str(item.get("id") or "") for item in cases]
    if any(not item or not str(case.get("question") or "").strip() for item, case in zip(ids, cases, strict=True)):
        raise ValueError("conversation_gold_case_invalid")
    if len(set(ids)) != len(ids):
        raise ValueError("conversation_gold_case_ids_must_be_unique")
    seen: set[str] = set()
    for case in cases:
        parent = case.get("session_from")
        if parent is not None and parent not in seen:
            raise ValueError("conversation_session_parent_must_precede_child")
        seen.add(str(case["id"]))
    return value


def _resolve_knowledge_base(*, knowledge_base_id: str | None, name: str | None):
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import KnowledgeBase

    with SessionLocal() as db:
        if knowledge_base_id:
            row = db.get(KnowledgeBase, knowledge_base_id)
        else:
            rows = list(db.scalars(select(KnowledgeBase).where(KnowledgeBase.name == str(name or ""))))
            if len(rows) != 1:
                raise ValueError("conversation_knowledge_base_name_not_unique")
            row = rows[0]
        if row is None or row.lifecycle_status != "active":
            raise ValueError("conversation_knowledge_base_not_active")
        return str(row.id), str(row.name)


def _post_json(base_url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/qa",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = None
        raise QAHTTPError(exc.code, error_payload, body) from exc
    if not isinstance(value, dict):
        raise RuntimeError("conversation_response_is_not_an_object")
    return value


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(__import__("math").ceil(percentile * len(ordered))) - 1)
    return ordered[index]


def _performance_projection(payload: dict | None) -> dict:
    value = dict(payload or {})
    stages = value.get("stages") if isinstance(value.get("stages"), dict) else {}
    elapsed_ms = float(value.get("elapsed_ms") or 0.0)
    provider_ms = float(
        (stages.get("provider_roundtrip") or {}).get("active_wall_ms") or 0.0
    )
    return {
        "protocol_version": value.get("protocol_version"),
        "elapsed_ms": elapsed_ms,
        "provider_roundtrip_active_ms": provider_ms,
        "non_model_wall_ms": max(0.0, elapsed_ms - provider_ms),
        "unfinished_span_count": int(value.get("unfinished_span_count") or 0),
        "stages": stages,
        "spans": value.get("spans") if isinstance(value.get("spans"), list) else [],
    }


def _persisted_audit(
    response: dict,
    *,
    case: dict,
    knowledge_base_id: str,
    parent_response: dict | None,
) -> dict:
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import (
        AgentObservation,
        AgentRun,
        AgentTraceEvent,
        AnswerSession,
        AnswerSourceBinding,
        ContextPackage,
        RetrievalLexicalReward,
        RetrievalTrace,
        RewardEvent,
    )

    with SessionLocal() as db:
        run = db.get(AgentRun, str(response.get("run_id") or ""))
        answer = db.get(AnswerSession, str(response.get("answer_session_id") or ""))
        trace = db.get(RetrievalTrace, str(response.get("retrieval_trace_id") or ""))
        package = db.get(ContextPackage, str(response.get("context_package_id") or ""))
        observations = list(
            db.scalars(
                select(AgentObservation).where(
                    AgentObservation.run_id == str(response.get("run_id") or "")
                )
            )
        )
        events = list(
            db.scalars(
                select(AgentTraceEvent).where(
                    AgentTraceEvent.run_id == str(response.get("run_id") or "")
                )
            )
        )
        bindings = list(
            db.scalars(
                select(AnswerSourceBinding).where(
                    AnswerSourceBinding.answer_session_id
                    == str(response.get("answer_session_id") or "")
                )
            )
        )
        reward_count = int(
            db.scalar(
                select(func.count()).select_from(RewardEvent).where(
                    RewardEvent.answer_session_id == str(response.get("answer_session_id") or "")
                )
            )
            or 0
        )
        lexical_reward_count = int(
            db.scalar(
                select(func.count()).select_from(RetrievalLexicalReward).where(
                    RetrievalLexicalReward.run_id == str(response.get("run_id") or "")
                )
            )
            or 0
        )
        performance = _performance_projection(
            (run.metadata_json or {}).get("qa_performance")
            if run is not None
            else None
        )

    plan_rows = [item for item in observations if item.observation_type == "intent_execution_plan"]
    admission_rows = [item for item in observations if item.observation_type == "source_integrity_admission"]
    generation_rows = [item for item in observations if item.observation_type == "single_grounded_generation"]
    citations = list(response.get("citations") or [])
    category = str(case.get("category") or "")
    direct = category == "system_capability"
    completed_answer = response.get("terminal_outcome") == "completed"
    timed_stages = set(performance["stages"])
    required_timed_stages = {
        "request",
        "conversation_prepare",
        "history_projection",
        "capability_manifest",
        "intent_planning",
        "database_commit",
        "model_call",
        "provider_roundtrip",
    }
    if category == "verified_reuse":
        required_timed_stages.add("context_reuse")
    elif not direct:
        required_timed_stages.update({"retrieval", "dense_entry"})
        if category != "insufficient":
            required_timed_stages.update({"context_package", "source_admission"})
    common = {
        "run_terminal": run is not None and run.status in {"completed", "needs_clarification"},
        "answer_bound": answer is not None
        and answer.knowledge_base_id == knowledge_base_id
        and answer.qa_session_id == response.get("session_id"),
        "one_plan": len(plan_rows) == 1 and plan_rows[0].verdict == "completed",
        "one_planning_model_call": len(plan_rows) == 1
        and plan_rows[0].observation_json.get("model_call_count") == 1,
        "no_online_reward": reward_count == 0 and lexical_reward_count == 0,
        "step_timing_complete": performance["protocol_version"]
        == "qa_stage_timing_v1"
        and performance["unfinished_span_count"] == 0
        and required_timed_stages <= timed_stages,
    }
    if direct:
        checks = {
            **common,
            "capability_route": response.get("route") == "system_capability",
            "capability_mode": response.get("direct_answer_mode") == "system_capability",
            "completed": completed_answer,
            "zero_retrieval": trace is None
            and package is None
            and response.get("retrieval_trace_id") is None
            and response.get("context_package_id") is None,
            "zero_sources": not citations and not bindings,
            "zero_generation": not generation_rows,
            "zero_source_admission": not admission_rows,
            "capability_card_answer": answer is not None
            and answer.prompt_protocol_version == "system_capability_card_v4",
        }
    else:
        source_bound = (
            completed_answer
            and trace is not None
            and package is not None
            and trace.knowledge_base_id == knowledge_base_id
            and package.retrieval_trace_id == trace.id
            and len(admission_rows) == 1
            and admission_rows[0].verdict == "passed"
            and admission_rows[0].observation_json.get("model_call_count") == 0
            and len(generation_rows) == 1
            and generation_rows[0].verdict == "completed"
            and len(bindings) == len(citations) > 0
        )
        gap_terminal = response.get("terminal_outcome") in {
            "insufficient_evidence",
            "scope_ambiguous",
            "representation_incomplete",
            "context_budget_exhausted",
        }
        checks = {
            **common,
            "target_or_reuse_route": response.get("route")
            in {"intent_execution_retrieval_v1", "verified_context_reuse"},
            "expected_grounding_outcome": (
                source_bound or gap_terminal
                if category == "insufficient"
                else source_bound
            ),
        }
        if category == "verified_reuse":
            checks.update(
                reuse_route=response.get("route") == "verified_context_reuse",
                same_session=parent_response is not None
                and response.get("session_id") == parent_response.get("session_id"),
                same_verified_package=parent_response is not None
                and response.get("context_package_id") == parent_response.get("context_package_id")
                and response.get("retrieval_trace_id") == parent_response.get("retrieval_trace_id"),
                no_new_retrieval_event=not any(
                    event.node == "intent_execution_retrieval" for event in events
                ),
            )
        elif category == "reuse_to_retrieval":
            checks.update(
                same_session=parent_response is not None
                and response.get("session_id") == parent_response.get("session_id"),
                incompatible_followup_uses_target_retrieval=response.get("route")
                == "intent_execution_retrieval_v1"
                and any(event.node == "intent_execution_retrieval" for event in events),
                reuse_was_replayed_or_rejected_by_plan=(
                    any(event.node == "verified_context_reuse" for event in events)
                    or (
                        len(plan_rows) == 1
                        and (
                            plan_rows[0].observation_json.get("accepted_plan")
                            or {}
                        ).get("strategy", {}).get("route") == "retrieve"
                    )
                ),
                new_package=parent_response is not None
                and response.get("context_package_id")
                != parent_response.get("context_package_id"),
            )
        elif category == "insufficient":
            checks.update(
                bounded_absence_response=gap_terminal
                or (completed_answer and source_bound),
            )
    return {
        "protocol_version": "intent_execution_conversation_audit_v1",
        "checks": checks,
        "passed": all(checks.values()),
        "safe_counts": {
            "plan_count": len(plan_rows),
            "admission_count": len(admission_rows),
            "generation_count": len(generation_rows),
            "source_binding_count": len(bindings),
            "citation_count": len(citations),
            "reward_count": reward_count,
            "lexical_reward_count": lexical_reward_count,
        },
        "performance": performance,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "intent-conversations.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50 or not 30 <= args.timeout_seconds <= 3600:
        parser.error("top-k must be 1-50 and timeout must be 30-3600 seconds")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "output").resolve()):
        parser.error("output must remain under output/")
    gold = _load_gold(args.gold.resolve())
    knowledge_base_id, knowledge_base_name = _resolve_knowledge_base(
        knowledge_base_id=args.knowledge_base_id,
        name=args.knowledge_base_name,
    )
    plan = {
        "operation": "intent_execution_conversation_acceptance",
        "execute": bool(args.execute),
        "knowledge_base_name": knowledge_base_name,
        "case_count": 10,
        "case_ids": [str(item["id"]) for item in gold["cases"]],
        "gold_hash": _hash(gold),
        "writes": (["10 sequential QA runs", str(output)] if args.execute else []),
    }
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if not args.execute:
        return 0

    records: list[dict] = []
    responses: dict[str, dict] = {}
    started = time.perf_counter()
    for case in gold["cases"]:
        case_started = time.perf_counter()
        case_id = str(case["id"])
        parent = responses.get(str(case.get("session_from") or ""))
        try:
            response = _post_json(
                args.base_url,
                {
                    "knowledge_base_id": knowledge_base_id,
                    "question": case["question"],
                    "session_id": parent.get("session_id") if parent else None,
                    "top_k": args.top_k,
                },
                args.timeout_seconds,
            )
            audit = _persisted_audit(
                response,
                case=case,
                knowledge_base_id=knowledge_base_id,
                parent_response=parent,
            )
            failure = None
            responses[case_id] = response
        except Exception as exc:
            detail = (
                exc.payload.get("detail")
                if isinstance(exc, QAHTTPError)
                and isinstance(exc.payload.get("detail"), dict)
                else {}
            )
            run_id = str(detail.get("run_id") or "")
            response = (
                {
                    "run_id": run_id,
                    "session_id": detail.get("session_id"),
                    "terminal_outcome": "technical_failure",
                    "route": None,
                    "citations": [],
                }
                if run_id
                else None
            )
            audit = (
                _persisted_audit(
                    response,
                    case=case,
                    knowledge_base_id=knowledge_base_id,
                    parent_response=parent,
                )
                if response is not None
                else {
                    "protocol_version": "intent_execution_conversation_audit_v1",
                    "checks": {},
                    "passed": False,
                }
            )
            failure = {
                "status": "technical_failure",
                "error_type": exc.__class__.__name__,
                "detail": str(exc)[:1000],
            }
        record = {
            **case,
            "response": response,
            "persisted_audit": audit,
            "hard_gate_passed": audit["passed"],
            "elapsed_seconds": round(time.perf_counter() - case_started, 3),
            "human_score": None,
            "failure": failure,
        }
        records.append(record)
        _write(
            output,
            {
                "protocol_version": "intent_execution_conversation_report_v1",
                "status": "running",
                "knowledge_base_id": knowledge_base_id,
                "knowledge_base_name": knowledge_base_name,
                "gold_hash": _hash(gold),
                "records": records,
            },
        )
        print(
            json.dumps(
                {
                    "id": case_id,
                    "category": case.get("category"),
                    "hard_gate_passed": audit["passed"],
                    "elapsed_seconds": record["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    stage_samples: dict[str, list[float]] = {}
    non_model_samples: list[float] = []
    provider_samples: list[float] = []
    outcome_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for record in records:
        response = record.get("response") or {}
        outcome = str(
            response.get("terminal_outcome")
            or (record.get("failure") or {}).get("status")
            or "unknown"
        )
        route = str(response.get("route") or "not_completed")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
        performance = (record.get("persisted_audit") or {}).get("performance") or {}
        if performance.get("protocol_version") != "qa_stage_timing_v1":
            continue
        non_model_samples.append(float(performance.get("non_model_wall_ms") or 0.0))
        provider_samples.append(
            float(performance.get("provider_roundtrip_active_ms") or 0.0)
        )
        for stage, values in (performance.get("stages") or {}).items():
            stage_samples.setdefault(stage, []).append(
                float((values or {}).get("active_wall_ms") or 0.0)
            )
    performance_summary = {
        "quantile_method": "nearest_rank",
        "sample_count": len(non_model_samples),
        "non_model_wall_ms": {
            "p50": _nearest_rank(non_model_samples, 0.5),
            "p95": _nearest_rank(non_model_samples, 0.95),
        },
        "provider_roundtrip_active_ms": {
            "p50": _nearest_rank(provider_samples, 0.5),
            "p95": _nearest_rank(provider_samples, 0.95),
        },
        "stages": {
            stage: {
                "n": len(values),
                "p50_ms": _nearest_rank(values, 0.5),
                "p95_ms": _nearest_rank(values, 0.95),
            }
            for stage, values in sorted(stage_samples.items())
        },
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
    }
    report = {
        "protocol_version": "intent_execution_conversation_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "gold_hash": _hash(gold),
        "same_final_configuration": True,
        "case_count": len(records),
        "hard_gate_pass_count": sum(item["hard_gate_passed"] for item in records),
        "all_hard_gates_passed": all(item["hard_gate_passed"] for item in records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "performance_summary": performance_summary,
        "records": records,
    }
    report["report_hash"] = _hash(report)
    _write(output, report)
    return 0 if report["all_hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
