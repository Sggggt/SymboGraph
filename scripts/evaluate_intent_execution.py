"""Run or replay the target RAG acceptance contract against frozen private gold."""

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
    if not isinstance(value, dict) or value.get("frozen_before_execution") is not True:
        raise ValueError("acceptance_gold_must_be_frozen_before_execution")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("acceptance_gold_requires_exactly_five_cases")
    ids = []
    for item in cases:
        if (
            not isinstance(item, dict)
            or not str(item.get("id") or "").strip()
            or not str(item.get("question") or "").strip()
            or not str(item.get("gold_answer") or "").strip()
            or not isinstance(item.get("required_points"), list)
            or not item["required_points"]
            or not item.get("source_reference")
        ):
            raise ValueError("acceptance_gold_case_contract_invalid")
        ids.append(str(item["id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("acceptance_gold_case_ids_must_be_unique")
    return value


def _post_json(base_url: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {body[:500]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("acceptance_response_is_not_an_object")
    return value


def _resolve_knowledge_base(*, knowledge_base_id: str | None, name: str | None):
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import KnowledgeBase

    with SessionLocal() as db:
        if knowledge_base_id:
            row = db.get(KnowledgeBase, knowledge_base_id)
        else:
            rows = list(
                db.scalars(
                    select(KnowledgeBase).where(
                        KnowledgeBase.name == str(name or "")
                    )
                )
            )
            if len(rows) != 1:
                raise ValueError("acceptance_knowledge_base_name_not_unique")
            row = rows[0]
        if row is None or row.lifecycle_status != "active":
            raise ValueError("acceptance_knowledge_base_not_active")
        return str(row.id), str(row.name)


def _persisted_audit(response: dict, *, knowledge_base_id: str) -> dict:
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import (
        AgentObservation,
        AgentRun,
        AnswerSession,
        AnswerSourceBinding,
        ContextPackage,
        RetrievalLexicalReward,
        RetrievalTrace,
        RewardEvent,
    )

    run_id = str(response.get("run_id") or "")
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        answer = db.get(AnswerSession, response.get("answer_session_id"))
        trace = db.get(RetrievalTrace, response.get("retrieval_trace_id"))
        package = db.get(ContextPackage, response.get("context_package_id"))
        observations = list(
            db.scalars(
                select(AgentObservation).where(AgentObservation.run_id == run_id)
            )
        )
        bindings = list(
            db.scalars(
                select(AnswerSourceBinding).where(
                    AnswerSourceBinding.answer_session_id
                    == response.get("answer_session_id")
                )
            )
        ) if answer is not None else []
        reward_count = int(
            db.scalar(
                select(func.count()).select_from(RewardEvent).where(
                    RewardEvent.answer_session_id == response.get("answer_session_id")
                )
            )
            or 0
        )
        lexical_reward_count = int(
            db.scalar(
                select(func.count()).select_from(RetrievalLexicalReward).where(
                    RetrievalLexicalReward.run_id == run_id
                )
            )
            or 0
        )
        admission = [
            item
            for item in observations
            if item.observation_type == "source_integrity_admission"
        ]
        generation = [
            item
            for item in observations
            if item.observation_type == "single_grounded_generation"
        ]
        checks = {
            "run_completed": run is not None and run.status == "completed",
            "target_route": run is not None
            and run.route == "intent_execution_retrieval_v1",
            "terminal_completed": response.get("terminal_outcome") == "completed",
            "answer_bound": answer is not None
            and answer.knowledge_base_id == knowledge_base_id,
            "retrieval_trace_bound": trace is not None
            and trace.knowledge_base_id == knowledge_base_id
            and trace.retrieval_mode == "intent_execution_retrieval_v1",
            "context_package_bound": package is not None
            and package.knowledge_base_id == knowledge_base_id
            and package.retrieval_trace_id == (trace.id if trace else None),
            "source_admission_passed": len(admission) == 1
            and admission[0].verdict == "passed"
            and admission[0].observation_json.get("model_call_count") == 0,
            "single_generation": len(generation) == 1
            and generation[0].verdict == "completed"
            and generation[0].observation_json.get("model_call_count") == 1,
            "citations_present": bool(response.get("citations")),
            "source_bindings_complete": bool(bindings)
            and len(bindings) == len(response.get("citations") or [])
            and all(
                item.protocol_version == "answer_source_binding_v2"
                and bool(
                    (item.diagnostics_json or {}).get(
                        "source_integrity_admission_hash"
                    )
                )
                for item in bindings
            ),
            "no_online_reward": reward_count == 0 and lexical_reward_count == 0,
            "no_result_reflection": bool(
                (trace.diagnostics_json or {}).get("result_reflection_enabled")
                is False
            ) if trace is not None else False,
            "no_generation_sufficiency_model": bool(
                (trace.diagnostics_json or {}).get(
                    "generation_sufficiency_model_enabled"
                )
                is False
            ) if trace is not None else False,
        }
        return {
            "protocol_version": "intent_execution_persisted_acceptance_v1",
            "checks": checks,
            "passed": all(checks.values()),
            "run_id": run_id,
            "retrieval_trace_id": trace.id if trace else None,
            "context_package_id": package.id if package else None,
            "answer_session_id": answer.id if answer else None,
            "source_binding_count": len(bindings),
            "source_admission_hash": (
                admission[0].observation_json.get("audit_hash")
                if len(admission) == 1
                else None
            ),
            "retrieval_cache": (
                (trace.scores_json or {}).get("retrieval_cache")
                if trace is not None
                else None
            ),
        }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate intent_execution_retrieval_v1 against five frozen cases."
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "intent-execution-acceptance.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50 or not 30 <= args.timeout_seconds <= 3600:
        parser.error("top-k must be 1-50 and timeout must be 30-3600 seconds")
    gold = _load_gold(args.gold.resolve())
    cases = [
        item
        for item in gold["cases"]
        if not args.case_id or str(item["id"]) in set(args.case_id)
    ]
    if not cases or (args.case_id and len(cases) != len(set(args.case_id))):
        parser.error("every --case-id must identify one frozen gold case")
    knowledge_base_id, knowledge_base_name = _resolve_knowledge_base(
        knowledge_base_id=args.knowledge_base_id,
        name=args.knowledge_base_name,
    )
    plan = {
        "operation": "intent_execution_retrieval_acceptance",
        "execute": bool(args.execute),
        "knowledge_base_name": knowledge_base_name,
        "case_ids": [str(item["id"]) for item in cases],
        "case_count": len(cases),
        "gold_hash": _hash(gold),
        "writes": (
            [f"{len(cases)} sequential QA runs", str(args.output.resolve())]
            if args.execute
            else []
        ),
    }
    print(json.dumps(plan, ensure_ascii=False))
    if not args.execute:
        return 0
    started = time.perf_counter()
    records = []
    for item in cases:
        case_started = time.perf_counter()
        try:
            response = _post_json(
                args.base_url,
                "/qa",
                {
                    "knowledge_base_id": knowledge_base_id,
                    "question": item["question"],
                    "top_k": args.top_k,
                },
                args.timeout_seconds,
            )
            audit = _persisted_audit(
                response,
                knowledge_base_id=knowledge_base_id,
            )
            failure = None
        except Exception as exc:
            response = None
            audit = {
                "protocol_version": "intent_execution_persisted_acceptance_v1",
                "checks": {},
                "passed": False,
            }
            failure = {
                "status": "technical_failure",
                "error_type": exc.__class__.__name__,
                "detail": str(exc)[:1000],
            }
        record = {
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "required_points": item["required_points"],
            "source_reference": item["source_reference"],
            "response": response,
            "persisted_audit": audit,
            "hard_gate_passed": audit["passed"],
            "elapsed_seconds": round(time.perf_counter() - case_started, 3),
            "human_score": None,
            "failure": failure,
        }
        records.append(record)
        _write(
            args.output.resolve(),
            {
                "protocol_version": "intent_execution_acceptance_report_v1",
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
                    "id": item["id"],
                    "hard_gate_passed": audit["passed"],
                    "elapsed_seconds": records[-1]["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    output = {
        "protocol_version": "intent_execution_acceptance_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "gold_hash": _hash(gold),
        "same_final_configuration": True,
        "case_count": len(records),
        "hard_gate_pass_count": sum(item["hard_gate_passed"] for item in records),
        "all_hard_gates_passed": all(item["hard_gate_passed"] for item in records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "records": records,
    }
    output["report_hash"] = _hash(output)
    _write(args.output.resolve(), output)
    return 0 if output["all_hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
