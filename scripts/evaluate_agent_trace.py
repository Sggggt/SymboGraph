from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report
from _gray_zone_audit import audit_gray_zone_trace, audit_gray_zone_traces
from _quality_gate import (
    PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION,
    audit_agent_quality,
    audit_retrieval_quality,
    persisted_agent_snapshot_hash,
    retrieval_snapshot_from_records,
)


REQUIRED_TRACE_NODES = {
    "agent_planner",
    "typed_action_validation",
    "typed_action_executor",
    "evidence_evaluator",
    "evidence_gate",
    "entry_selection",
    "layer_drilldown",
    "frontier_traversal",
    "chunk_recall",
    "layered_retrieval",
    "structure_context_restoration",
    "context_package",
    "grounded_answer",
    "citation_verification",
    "reward_event",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real QA requests and verify the Layered P&E Agent trace.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        default=[],
        help=(
            "Replay one exact persisted Agent run without HTTP or model calls. "
            "Repeat as needed; cannot be combined with --question or --execute."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=900,
        help=(
            "Bounded HTTP timeout for each explicit --execute QA request and "
            "its trace readback (30-3600 seconds)."
        ),
    )
    parser.add_argument(
        "--require-gray-coverage",
        action="store_true",
        help="Fail unless the persisted retrieval traces include at least one deterministic gray local-rule decision.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send QA requests and persist Agent/answer/reward/session audit. Omit for a read-only request plan.",
    )
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


def get_json(base_url: str, path: str, timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", method="GET")
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


def persisted_retrieval_snapshot(trace_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from app.models import GraphRetrievalStep, RetrievalTrace

    with session_scope() as db:
        trace = db.get(RetrievalTrace, trace_id)
        if trace is None:
            raise RuntimeError(f"persisted retrieval trace not found: {trace_id}")
        steps = list(
            db.scalars(
                select(GraphRetrievalStep)
                .where(GraphRetrievalStep.retrieval_trace_id == trace_id)
                .order_by(GraphRetrievalStep.step_index.asc())
            ).all()
        )
        return retrieval_snapshot_from_records(trace, steps)


def persisted_typed_action_facts(run_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from app.models import AgentAction, AgentPlan, RetrievalTrace

    with session_scope() as db:
        plans = list(
            db.scalars(
                select(AgentPlan)
                .where(AgentPlan.run_id == run_id)
                .order_by(AgentPlan.plan_index.asc(), AgentPlan.id.asc())
            ).all()
        )
        payloads: list[dict[str, Any]] = []
        for plan in plans:
            typed_actions = list(plan.typed_actions_json or [])
            plan_diagnostics = dict(plan.diagnostics_json or {})
            execution_controls = dict(
                plan_diagnostics.get("execution_controls") or {}
            )
            retrieval_trace = (
                db.get(RetrievalTrace, plan.retrieval_trace_id)
                if plan.retrieval_trace_id
                else None
            )
            retrieval_diagnostics = dict(
                (retrieval_trace.diagnostics_json or {})
                if retrieval_trace is not None
                else {}
            )
            action_rows = list(
                db.scalars(
                    select(AgentAction)
                    .where(AgentAction.plan_id == plan.id)
                    .order_by(
                        AgentAction.action_index.asc(),
                        AgentAction.id.asc(),
                    )
                ).all()
            )
            base_rows = [
                row
                for row in action_rows
                if int(row.action_index) < len(typed_actions)
            ]
            payloads.append(
                {
                    "plan_id": plan.id,
                    "run_id": plan.run_id,
                    "plan_index": plan.plan_index,
                    "knowledge_base_id": plan.knowledge_base_id,
                    "retrieval_trace_id": plan.retrieval_trace_id,
                    "envelope": plan.envelope_json or {},
                    "typed_actions": typed_actions,
                    "validation": plan.validation_json or {},
                    "execution_controls": execution_controls,
                    "retrieval_control_binding": {
                        "retrieval_trace_id": (
                            retrieval_trace.id
                            if retrieval_trace is not None
                            else None
                        ),
                        "knowledge_base_id": (
                            retrieval_trace.knowledge_base_id
                            if retrieval_trace is not None
                            else None
                        ),
                        "agent_plan_id": retrieval_diagnostics.get(
                            "agent_plan_id"
                        ),
                        "agent_plan_index": retrieval_diagnostics.get(
                            "agent_plan_index"
                        ),
                        "typed_action_control_hash": (
                            retrieval_diagnostics.get(
                                "typed_action_control_hash"
                            )
                        ),
                        "typed_action_executor_protocol_version": (
                            retrieval_diagnostics.get(
                                "typed_action_executor_protocol_version"
                            )
                        ),
                        "typed_action_controls": dict(
                            retrieval_diagnostics.get(
                                "typed_action_controls"
                            )
                            or {}
                        ),
                    },
                    "actions": [
                        {
                            "run_id": row.run_id,
                            "action_index": row.action_index,
                            "action_type": row.action_type,
                            "target_ids": row.target_ids_json or [],
                            "reason": row.reason,
                            "budget_request": row.budget_request_json or {},
                            "expected_evidence": (
                                row.expected_evidence_json or {}
                            ),
                            "stop_condition": (
                                row.stop_condition_json or {}
                            ),
                            "validation": row.validation_json or {},
                            "status": row.status,
                        }
                        for row in base_rows
                    ],
                }
            )
        return {"run_id": run_id, "plans": payloads}


def persisted_agent_quality_snapshot(
    run_id: str,
    *,
    expected_knowledge_base_id: str | None = None,
) -> dict[str, Any]:
    from sqlalchemy import select

    from app.models import (
        AgentPlan,
        AgentRun,
        AgentTraceEvent,
        AnswerSession,
        ContextPackage,
        RetrievalTrace,
        RewardEvent,
    )

    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            raise RuntimeError(f"persisted Agent run not found: {run_id}")
        if (
            expected_knowledge_base_id is not None
            and str(run.knowledge_base_id)
            != str(expected_knowledge_base_id)
        ):
            raise RuntimeError(
                "persisted Agent run does not belong to the selected knowledge "
                f"base: run_id={run_id}, selected={expected_knowledge_base_id}"
            )
        trace_rows = list(
            db.scalars(
                select(AgentTraceEvent)
                .where(AgentTraceEvent.run_id == run_id)
                .order_by(
                    AgentTraceEvent.sequence_index.asc(),
                    AgentTraceEvent.id.asc(),
                )
            ).all()
        )
        plans = list(
            db.scalars(
                select(AgentPlan)
                .where(AgentPlan.run_id == run_id)
                .order_by(
                    AgentPlan.plan_index.asc(),
                    AgentPlan.id.asc(),
                )
            ).all()
        )
        final_plan = plans[-1] if plans else None
        initial_retrieval_trace_id = (
            str(final_plan.retrieval_trace_id or "")
            if final_plan is not None
            else ""
        )
        package_ids = {
            str((row.scores or {}).get("context_package_id") or "")
            for row in trace_rows
            if row.node == "context_package"
            and (row.scores or {}).get("context_package_id")
        }
        initial_context_package_id = (
            next(iter(package_ids)) if len(package_ids) == 1 else ""
        )
        initial_retrieval_trace = (
            db.get(RetrievalTrace, initial_retrieval_trace_id)
            if initial_retrieval_trace_id
            else None
        )
        initial_context_package = (
            db.get(ContextPackage, initial_context_package_id)
            if initial_context_package_id
            else None
        )
        reward_candidates = list(
            db.scalars(
                select(RewardEvent).where(
                    RewardEvent.knowledge_base_id
                    == run.knowledge_base_id
                )
            ).all()
        )
        reward_rows = [
            row
            for row in reward_candidates
            if str(
                (row.context_json or {}).get("agent_run_id") or ""
            )
            == run_id
        ]
        reward = reward_rows[0] if len(reward_rows) == 1 else None
        answer_session = (
            db.get(AnswerSession, reward.answer_session_id)
            if reward is not None and reward.answer_session_id
            else None
        )
        retrieval_trace = (
            db.get(
                RetrievalTrace,
                str(answer_session.retrieval_trace_id or ""),
            )
            if answer_session is not None
            and answer_session.retrieval_trace_id
            else None
        )
        context_package = (
            db.get(
                ContextPackage,
                str(answer_session.context_package_id or ""),
            )
            if answer_session is not None
            and answer_session.context_package_id
            else None
        )
        snapshot: dict[str, Any] = {
            "protocol_version": (
                PERSISTED_AGENT_REPLAY_PROTOCOL_VERSION
            ),
            "run": {
                "id": run.id,
                "knowledge_base_id": run.knowledge_base_id,
                "session_id": run.session_id,
                "question": run.question,
                "status": run.status,
                "route": run.route,
                "current_node": run.current_node,
                "retry_count": run.retry_count,
                "final_answer": run.final_answer,
                "error_message": run.error_message,
                "metadata": dict(run.metadata_json or {}),
            },
            "trace_events": [
                {
                    "id": row.id,
                    "run_id": row.run_id,
                    "sequence_index": row.sequence_index,
                    "node": row.node,
                    "status": row.status,
                    "input_summary": row.input_summary or "",
                    "output_summary": row.output_summary or "",
                    "document_ids": list(row.document_ids or []),
                    "scores": dict(row.scores or {}),
                    "duration_ms": row.duration_ms,
                    "error": row.error_message,
                }
                for row in trace_rows
            ],
            "bindings": {
                "initial_retrieval_trace": {
                    "id": (
                        initial_retrieval_trace.id
                        if initial_retrieval_trace is not None
                        else None
                    ),
                    "knowledge_base_id": (
                        initial_retrieval_trace.knowledge_base_id
                        if initial_retrieval_trace is not None
                        else None
                    ),
                },
                "initial_context_package": {
                    "id": (
                        initial_context_package.id
                        if initial_context_package is not None
                        else None
                    ),
                    "knowledge_base_id": (
                        initial_context_package.knowledge_base_id
                        if initial_context_package is not None
                        else None
                    ),
                    "retrieval_trace_id": (
                        initial_context_package.retrieval_trace_id
                        if initial_context_package is not None
                        else None
                    ),
                },
                "retrieval_trace": {
                    "id": (
                        retrieval_trace.id
                        if retrieval_trace is not None
                        else None
                    ),
                    "knowledge_base_id": (
                        retrieval_trace.knowledge_base_id
                        if retrieval_trace is not None
                        else None
                    ),
                },
                "context_package": {
                    "id": (
                        context_package.id
                        if context_package is not None
                        else None
                    ),
                    "knowledge_base_id": (
                        context_package.knowledge_base_id
                        if context_package is not None
                        else None
                    ),
                    "retrieval_trace_id": (
                        context_package.retrieval_trace_id
                        if context_package is not None
                        else None
                    ),
                },
                "answer_session": {
                    "id": (
                        answer_session.id
                        if answer_session is not None
                        else None
                    ),
                    "knowledge_base_id": (
                        answer_session.knowledge_base_id
                        if answer_session is not None
                        else None
                    ),
                    "retrieval_trace_id": (
                        answer_session.retrieval_trace_id
                        if answer_session is not None
                        else None
                    ),
                    "context_package_id": (
                        answer_session.context_package_id
                        if answer_session is not None
                        else None
                    ),
                    "qa_session_id": (
                        answer_session.qa_session_id
                        if answer_session is not None
                        else None
                    ),
                    "question": (
                        answer_session.question
                        if answer_session is not None
                        else None
                    ),
                    "answer": (
                        answer_session.answer
                        if answer_session is not None
                        else None
                    ),
                },
                "reward_event": {
                    "id": reward.id if reward is not None else None,
                    "knowledge_base_id": (
                        reward.knowledge_base_id
                        if reward is not None
                        else None
                    ),
                    "agent_run_id": (
                        (reward.context_json or {}).get("agent_run_id")
                        if reward is not None
                        else None
                    ),
                    "retrieval_trace_id": (
                        reward.retrieval_trace_id
                        if reward is not None
                        else None
                    ),
                    "context_package_id": (
                        (reward.context_json or {}).get(
                            "context_package_id"
                        )
                        if reward is not None
                        else None
                    ),
                    "answer_session_id": (
                        reward.answer_session_id
                        if reward is not None
                        else None
                    ),
                    "policy_state_id": (
                        reward.policy_state_id
                        if reward is not None
                        else None
                    ),
                },
            },
        }
        snapshot["snapshot_hash"] = persisted_agent_snapshot_hash(
            snapshot
        )
        return snapshot


def persisted_agent_response_bundle(
    run_id: str,
    *,
    expected_knowledge_base_id: str,
    persisted_agent_facts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild the auditable public response from persisted PostgreSQL facts."""

    from sqlalchemy import select

    from app.models import (
        AgentRun,
        AnswerSession,
        CitationVerification,
        ContextPackage,
    )
    from app.services.agent_graph import citation_verification_public_payload
    from app.services.agent_pe_audit import load_agent_pe_audit

    bindings = dict(persisted_agent_facts.get("bindings") or {})
    answer_binding = dict(bindings.get("answer_session") or {})
    package_binding = dict(bindings.get("context_package") or {})
    retrieval_binding = dict(bindings.get("retrieval_trace") or {})
    answer_session_id = str(answer_binding.get("id") or "")
    context_package_id = str(package_binding.get("id") or "")
    retrieval_trace_id = str(retrieval_binding.get("id") or "")
    if not answer_session_id or not context_package_id or not retrieval_trace_id:
        raise RuntimeError(
            "persisted Agent replay is missing final answer/trace/package bindings"
        )

    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        answer_session = db.get(AnswerSession, answer_session_id)
        context_package = db.get(ContextPackage, context_package_id)
        if run is None or answer_session is None or context_package is None:
            raise RuntimeError(
                "persisted Agent response facts are incomplete for replay"
            )
        if any(
            str(value) != str(expected_knowledge_base_id)
            for value in (
                run.knowledge_base_id,
                answer_session.knowledge_base_id,
                context_package.knowledge_base_id,
            )
        ):
            raise RuntimeError(
                "persisted Agent response facts cross the selected knowledge-base boundary"
            )
        if (
            str(answer_session.context_package_id or "")
            != context_package_id
            or str(answer_session.retrieval_trace_id or "")
            != retrieval_trace_id
            or str(context_package.retrieval_trace_id or "")
            != retrieval_trace_id
        ):
            raise RuntimeError(
                "persisted Agent answer/trace/package bindings are inconsistent"
            )

        verification_rows = list(
            db.scalars(
                select(CitationVerification)
                .where(
                    CitationVerification.answer_session_id
                    == answer_session_id
                )
                .order_by(
                    CitationVerification.created_at.asc(),
                    CitationVerification.id.asc(),
                )
            ).all()
        )
        verification_by_id = {
            str(verification.id): verification
            for verification in verification_rows
        }
        citation_verification_ids = list(
            answer_session.citation_ids_json or []
        )
        if (
            len(citation_verification_ids)
            != len(set(citation_verification_ids))
            or set(citation_verification_ids) != set(verification_by_id)
        ):
            raise RuntimeError(
                "AnswerSession citation ids do not cover the complete persisted "
                "CitationVerification set"
            )
        ordered_verifications = [
            verification_by_id[verification_id]
            for verification_id in citation_verification_ids
        ]
        package_chunks = list(
            (context_package.package_json or {}).get("chunks") or []
        )
        package_chunks_by_id = {
            str(chunk.get("chunk_id") or ""): dict(chunk)
            for chunk in package_chunks
            if isinstance(chunk, dict) and chunk.get("chunk_id")
        }
        citations: list[dict[str, Any]] = []
        for citation_index, verification in enumerate(
            ordered_verifications,
            start=1,
        ):
            if (
                str(verification.verdict or "") != "supported"
                or not verification.chunk_id
            ):
                continue
            chunk_id = str(verification.chunk_id)
            chunk = package_chunks_by_id.get(chunk_id)
            if chunk is None:
                raise RuntimeError(
                    "supported CitationVerification references a chunk outside "
                    "the persisted ContextPackage"
                )
            diagnostics = dict(verification.diagnostics_json or {})
            source_span = dict(verification.source_span_json or {})
            source_span["verification_id"] = str(verification.id)
            section_path = source_span.get("section_path") or chunk.get(
                "section_path"
            )
            citations.append(
                {
                    "citation_index": citation_index,
                    "claim_id": diagnostics.get("claim_id"),
                    "claim_index": diagnostics.get("claim_index"),
                    "claim_text": verification.claim_text,
                    "answer_hash": diagnostics.get("answer_hash"),
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id"),
                    "document_version_id": chunk.get("document_version_id")
                    or source_span.get("document_version_id"),
                    "document_title": chunk.get("document_title") or "",
                    "source_path": chunk.get("source_path") or "",
                    "logical_source_path": chunk.get("logical_source_path")
                    or source_span.get("logical_source_path")
                    or "",
                    "partition": None,
                    "section": chunk.get("section_path"),
                    "page_number": (
                        (chunk.get("page_range") or [None])[0]
                    ),
                    "page_range": source_span.get("page_range")
                    or chunk.get("page_range"),
                    "char_span": source_span.get("char_span")
                    or chunk.get("char_span"),
                    "section_path": (
                        list(section_path)
                        if isinstance(section_path, list)
                        else [str(section_path)]
                        if section_path
                        else []
                    ),
                    "bbox": source_span.get("bbox")
                    or chunk.get("bbox")
                    or None,
                    "snippet": str(chunk.get("content") or "")[:240],
                    "source_span": source_span,
                    "context_package_id": context_package_id,
                    "retrieval_trace_id": retrieval_trace_id,
                    "answer_session_id": answer_session_id,
                    "citation_verification_id": str(verification.id),
                    "verification": citation_verification_public_payload(
                        verification,
                        source_span=source_span,
                    ),
                }
            )

        pe_audit = load_agent_pe_audit(db, run_id).model_dump(mode="json")
        if (
            pe_audit.get("run_id") != run_id
            or pe_audit.get("knowledge_base_id")
            != expected_knowledge_base_id
        ):
            raise RuntimeError(
                "persisted P&E audit does not belong to the selected Agent run"
            )
        model_audit = dict(answer_session.model_json or {})
        response = {
            "run_id": run_id,
            "session_id": run.session_id,
            "answer": answer_session.answer,
            "citations": citations,
            "route": run.route,
            "trace": list(persisted_agent_facts.get("trace_events") or []),
            "context_package_id": context_package_id,
            "retrieval_trace_id": retrieval_trace_id,
            "retrieval_granularity": model_audit.get(
                "retrieval_granularity"
            ),
            "model_audit": model_audit,
            "answer_model_audit": model_audit,
        }
        identity = {
            "run": {
                "id": run_id,
                "knowledge_base_id": str(run.knowledge_base_id),
                "session_id": run.session_id,
                "status": run.status,
            },
            "bindings": bindings,
            "agent_trace_event_ids": [
                str(event.get("id") or "")
                for event in persisted_agent_facts.get("trace_events") or []
            ],
            "citation_verification_ids": citation_verification_ids,
            "context_package": {
                "id": context_package_id,
                "knowledge_base_id": str(
                    context_package.knowledge_base_id
                ),
                "retrieval_trace_id": str(
                    context_package.retrieval_trace_id or ""
                ),
                "hit_chunk_ids": list(
                    context_package.hit_chunk_ids_json or []
                ),
                "restored_chunk_ids": list(
                    context_package.restored_chunk_ids_json or []
                ),
                "bridge_chunk_ids": list(
                    context_package.bridge_chunk_ids_json or []
                ),
                "runtime_settings_hash": (
                    context_package.runtime_settings_hash
                ),
                "profile_hash": context_package.profile_hash,
            },
            "repair_chain": [
                {
                    key: repair.get(key)
                    for key in (
                        "action_type",
                        "repair_round_index",
                        "before_retrieval_trace_id",
                        "repaired_retrieval_trace_id",
                        "before_context_package_id",
                        "repaired_context_package_id",
                    )
                }
                for repair in model_audit.get("repair_actions") or []
                if isinstance(repair, dict)
            ],
            "pe": {
                "contract_version": pe_audit.get("contract_version"),
                "counts": pe_audit.get("counts"),
                "plan_ids": [str(item["id"]) for item in pe_audit["plans"]],
                "action_ids": [
                    str(item["id"]) for item in pe_audit["actions"]
                ],
                "observation_ids": [
                    str(item["id"]) for item in pe_audit["observations"]
                ],
                "provider_raw_response_exposed": pe_audit.get(
                    "provider_raw_response_exposed"
                ),
                "credentials_exposed": pe_audit.get(
                    "credentials_exposed"
                ),
            },
            "persisted_agent_snapshot_hash": persisted_agent_facts.get(
                "snapshot_hash"
            ),
        }
        return response, identity, pe_audit


def main() -> None:
    args = parse_args()
    request_timeout_seconds = int(
        getattr(args, "request_timeout_seconds", 900)
    )
    if not 30 <= request_timeout_seconds <= 3600:
        raise SystemExit(
            "--request-timeout-seconds must be between 30 and 3600"
        )
    run_ids = list(
        dict.fromkeys(
            str(run_id)
            for run_id in (getattr(args, "run_ids", None) or [])
            if str(run_id)
        )
    )
    explicit_questions = list(getattr(args, "question", None) or [])
    if run_ids and bool(args.execute):
        raise SystemExit("--run-id cannot be combined with --execute")
    if run_ids and explicit_questions:
        raise SystemExit("--run-id cannot be combined with --question")
    if len(run_ids) > 100:
        raise SystemExit(
            "At most 100 distinct --run-id targets may be replayed at once"
        )
    knowledge_base_id, knowledge_base_name = resolve_kb_id(args)
    if run_ids:
        replay_rows: list[dict[str, Any]] = []
        replay_raw_traces: list[dict[str, Any]] = []
        for run_id in run_ids:
            persisted_agent_facts = persisted_agent_quality_snapshot(
                run_id,
                expected_knowledge_base_id=knowledge_base_id,
            )
            response, identity, pe_audit = persisted_agent_response_bundle(
                run_id,
                expected_knowledge_base_id=knowledge_base_id,
                persisted_agent_facts=persisted_agent_facts,
            )
            retrieval_trace_id = str(
                response.get("retrieval_trace_id") or ""
            )
            persisted_trace = persisted_retrieval_snapshot(
                retrieval_trace_id
            )
            identity["retrieval_trace"] = {
                "id": persisted_trace.get("trace_id"),
                "runtime_settings_hash": persisted_trace.get(
                    "runtime_settings_hash"
                ),
                "agent_operating_envelope_hash": persisted_trace.get(
                    "agent_operating_envelope_hash"
                ),
                "edge_distance_protocol_hash": persisted_trace.get(
                    "edge_distance_protocol_hash"
                ),
                "edge_projection_protocol_hash": persisted_trace.get(
                    "edge_projection_protocol_hash"
                ),
                "traversal_protocol_hash": persisted_trace.get(
                    "traversal_protocol_hash"
                ),
                "retrieval_granularity": persisted_trace.get(
                    "retrieval_granularity"
                ),
                "result_chunk_ids": list(
                    persisted_trace.get("result_chunk_ids") or []
                ),
            }
            replay_raw_traces.append(persisted_trace)
            row_gray_audit = audit_gray_zone_trace(
                persisted_trace,
                require_gray_coverage=False,
            )
            retrieval_quality = audit_retrieval_quality(
                persisted_trace,
                gray_zone_audit=row_gray_audit,
            )
            typed_action_facts = persisted_typed_action_facts(run_id)
            agent_quality = audit_agent_quality(
                response,
                retrieval_snapshot=persisted_trace,
                gray_zone_audit=row_gray_audit,
                typed_action_facts=typed_action_facts,
                retrieval_quality=retrieval_quality,
                persisted_agent_facts=persisted_agent_facts,
            )
            answer_audit = dict(response.get("answer_model_audit") or {})
            grounding_outcome = str(
                answer_audit.get("grounding_outcome") or ""
            )
            insufficient_evidence = bool(
                answer_audit.get("insufficient_evidence") is True
                or grounding_outcome == "insufficient_evidence"
            )
            citation_count = len(response.get("citations") or [])
            required_binding_names = {
                "initial_retrieval_trace",
                "initial_context_package",
                "retrieval_trace",
                "context_package",
                "answer_session",
                "reward_event",
            }
            bindings = dict(identity.get("bindings") or {})
            identity_complete = (
                required_binding_names.issubset(bindings)
                and all(
                    bool((bindings.get(name) or {}).get("id"))
                    for name in required_binding_names
                )
                and bool(identity.get("agent_trace_event_ids"))
                and all(identity.get("agent_trace_event_ids") or [])
                and all(identity.get("citation_verification_ids") or [])
                and (
                    bool(identity.get("citation_verification_ids"))
                    or insufficient_evidence
                )
                and bool((identity.get("pe") or {}).get("plan_ids"))
                and bool((identity.get("pe") or {}).get("action_ids"))
                and bool((identity.get("pe") or {}).get("observation_ids"))
                and bool(identity.get("persisted_agent_snapshot_hash"))
            )
            replay_rows.append(
                {
                    "run_id": run_id,
                    "question": response.get("answer_model_audit", {}).get(
                        "question"
                    )
                    or (persisted_agent_facts.get("run") or {}).get(
                        "question"
                    ),
                    "answer_length": len(response.get("answer") or ""),
                    "citation_count": citation_count,
                    "grounding_outcome": grounding_outcome or None,
                    "insufficient_evidence": insufficient_evidence,
                    "citation_requirement_satisfied": bool(
                        citation_count > 0 or insufficient_evidence
                    ),
                    "knowledge_base_ownership_verified": (
                        (identity.get("run") or {}).get(
                            "knowledge_base_id"
                        )
                        == knowledge_base_id
                        == pe_audit.get("knowledge_base_id")
                    ),
                    "identity_complete": identity_complete,
                    "identity": identity,
                    "gray_zone_trace_audit": row_gray_audit,
                    "retrieval_quality_gate": retrieval_quality,
                    "agent_quality_gate": agent_quality,
                }
            )

        replay_gray_audit = audit_gray_zone_traces(
            replay_raw_traces,
            require_gray_coverage=bool(
                getattr(args, "require_gray_coverage", False)
            ),
        )
        replay_checks = {
            "requested_runs_replayed": len(replay_rows) == len(run_ids),
            "knowledge_base_ownership_verified": all(
                row["knowledge_base_ownership_verified"]
                for row in replay_rows
            ),
            "complete_persisted_identity_reported": all(
                row["identity_complete"] for row in replay_rows
            ),
            "answers_reconstructed": all(
                row["answer_length"] > 0 for row in replay_rows
            ),
            "citation_requirements_satisfied": all(
                row["citation_requirement_satisfied"]
                for row in replay_rows
            ),
            "gray_zone_trace_audit": bool(
                replay_gray_audit.get("pass")
            )
            and all(
                bool(row["gray_zone_trace_audit"].get("pass"))
                for row in replay_rows
            ),
            "deterministic_gray_model_call_count_zero": (
                type(replay_gray_audit.get("raw_record_count")) is int
                and type(
                    replay_gray_audit.get(
                        "explicit_zero_model_call_record_count"
                    )
                )
                is int
                and replay_gray_audit[
                    "explicit_zero_model_call_record_count"
                ]
                == replay_gray_audit["raw_record_count"]
            ),
            "versioned_retrieval_quality_gate": all(
                row["retrieval_quality_gate"]["pass"]
                for row in replay_rows
            ),
            "versioned_agent_quality_gate": all(
                row["agent_quality_gate"]["pass"]
                for row in replay_rows
            ),
        }
        replay_payload = {
            "script": "evaluate_agent_trace",
            "execute": False,
            "mode": "persisted_replay",
            "base_url": None,
            "knowledge_base_id": knowledge_base_id,
            "knowledge_base_name": knowledge_base_name,
            "run_ids": run_ids,
            "require_gray_coverage": bool(
                getattr(args, "require_gray_coverage", False)
            ),
            "checks": replay_checks,
            "pass": bool(replay_rows) and all(replay_checks.values()),
            "rows": replay_rows,
            "gray_zone_trace_audit": replay_gray_audit,
            "targets": {
                "persisted_run_ids": run_ids,
                "knowledge_base_id": knowledge_base_id,
                "request_count": len(replay_rows),
            },
            "impact": (
                "read only persisted PostgreSQL Agent response/run/trace/"
                "ContextPackage/P&E/citation/reward facts; no HTTP, model-provider, "
                "Qdrant, Redis, cache, or database writes"
            ),
        }
        report = write_report("evaluate_agent_trace", replay_payload)
        print(
            json.dumps(
                {
                    "output": str(report),
                    "mode": replay_payload["mode"],
                    "pass": replay_payload["pass"],
                    "checks": replay_checks,
                    "run_ids": run_ids,
                },
                ensure_ascii=False,
                default=str,
            )
        )
        if not replay_payload["pass"]:
            raise SystemExit(1)
        return

    questions = explicit_questions or [
        "Explain the Metropolis-Hastings acceptance probability using the indexed material.",
        "How do posterior, prior, and likelihood relate in Bayesian inference?",
    ]
    if not args.execute:
        payload = {
            "script": "evaluate_agent_trace",
            "execute": False,
            "mode": "dry_run",
            "base_url": args.base_url,
            "knowledge_base_id": knowledge_base_id,
            "knowledge_base_name": knowledge_base_name,
            "top_k": args.top_k,
            "request_timeout_seconds": request_timeout_seconds,
            "require_gray_coverage": bool(getattr(args, "require_gray_coverage", False)),
            "planned_requests": [
                {
                    "method": "POST",
                    "path": "/qa",
                    "follow_up": "GET /retrieval-traces/{retrieval_trace_id}/graph-steps",
                    "question": question,
                    "knowledge_base_id": knowledge_base_id,
                    "top_k": args.top_k,
                }
                for question in questions
            ],
            "targets": {
                "operations": ["agent_run", "answer_session", "retrieval_trace", "context_package", "reward_event", "policy_state"],
                "knowledge_base_id": knowledge_base_id,
            },
            "impact": "no HTTP POST, PostgreSQL/Qdrant/Redis/model-provider writes or calls",
            "next_step": "rerun with --execute after reviewing every planned QA request",
        }
        report = write_report("evaluate_agent_trace", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))
        return

    rows: list[dict[str, Any]] = []
    raw_traces: list[dict[str, Any]] = []
    for question in questions:
        response = post_json(
            args.base_url,
            "/qa",
            {"question": question, "knowledge_base_id": knowledge_base_id, "top_k": args.top_k, "filters": {}, "history": []},
            timeout=request_timeout_seconds,
        )
        nodes = {str(item.get("node")) for item in response.get("trace") or [] if item.get("node")}
        model_audit = response.get("model_audit") or response.get("answer_model_audit") or {}
        retrieval_trace_id = response.get("retrieval_trace_id")
        trace_payload = (
            get_json(
                args.base_url,
                f"/retrieval-traces/{retrieval_trace_id}/graph-steps",
                timeout=request_timeout_seconds,
            )
            if retrieval_trace_id
            else None
        )
        if trace_payload is not None:
            raw_traces.append(trace_payload)
        row_gray_audit = (
            audit_gray_zone_trace(trace_payload, require_gray_coverage=False)
            if trace_payload is not None
            else {
                "status": "incomplete",
                "pass": False,
                "gray_zone_coverage": False,
                "incomplete_traces": [{"missing_fields": ["retrieval_trace_id"]}],
            }
        )
        persisted_trace = (
            persisted_retrieval_snapshot(str(retrieval_trace_id))
            if retrieval_trace_id
            else {}
        )
        persisted_gray_audit = (
            audit_gray_zone_trace(persisted_trace, require_gray_coverage=False)
            if persisted_trace
            else {"pass": False, "status": "incomplete"}
        )
        retrieval_quality = audit_retrieval_quality(
            persisted_trace,
            gray_zone_audit=persisted_gray_audit,
        )
        typed_action_facts = (
            persisted_typed_action_facts(str(response.get("run_id")))
            if response.get("run_id")
            else {}
        )
        persisted_agent_facts = (
            persisted_agent_quality_snapshot(
                str(response.get("run_id"))
            )
            if response.get("run_id")
            else {}
        )
        persisted_response = {
            **response,
            "trace": list(
                persisted_agent_facts.get("trace_events") or []
            ),
        }
        agent_quality = audit_agent_quality(
            persisted_response,
            retrieval_snapshot=persisted_trace,
            gray_zone_audit=persisted_gray_audit,
            typed_action_facts=typed_action_facts,
            retrieval_quality=retrieval_quality,
            persisted_agent_facts=persisted_agent_facts,
        )
        rows.append(
            {
                "question": question,
                "run_id": response.get("run_id"),
                "context_package_id": response.get("context_package_id"),
                "retrieval_trace_id": retrieval_trace_id,
                "answer_length": len(response.get("answer") or ""),
                "citation_count": len(response.get("citations") or []),
                "trace_nodes": sorted(nodes),
                "missing_trace_nodes": sorted(REQUIRED_TRACE_NODES - nodes),
                "citation_verification_pass_rate": model_audit.get("citation_verification_pass_rate"),
                "degraded_mode": bool(response.get("degraded_mode")),
                "gray_zone_trace_audit": row_gray_audit,
                "retrieval_quality_gate": retrieval_quality,
                "agent_quality_gate": agent_quality,
            }
        )

    gray_zone_trace_audit = audit_gray_zone_traces(
        raw_traces,
        require_gray_coverage=bool(getattr(args, "require_gray_coverage", False)),
    )
    checks = {
        "answers_returned": all(row["answer_length"] > 0 for row in rows),
        "context_packages_returned": all(row["context_package_id"] and row["retrieval_trace_id"] for row in rows),
        "citations_returned": all(row["citation_count"] > 0 for row in rows),
        "required_trace_nodes_present": all(not row["missing_trace_nodes"] for row in rows),
        "citation_verification_positive": all((row["citation_verification_pass_rate"] or 0) > 0 for row in rows),
        "not_degraded": not any(row["degraded_mode"] for row in rows),
        "gray_zone_trace_audit": bool(gray_zone_trace_audit["pass"])
        and all(bool(row["gray_zone_trace_audit"].get("pass")) for row in rows),
        "versioned_retrieval_quality_gate": all(
            row["retrieval_quality_gate"]["pass"] for row in rows
        ),
        "versioned_agent_quality_gate": all(
            row["agent_quality_gate"]["pass"] for row in rows
        ),
    }
    payload = {
        "script": "evaluate_agent_trace",
        "execute": True,
        "mode": "execute",
        "base_url": args.base_url,
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "request_timeout_seconds": request_timeout_seconds,
        "checks": checks,
        "pass": all(checks.values()),
        "rows": rows,
        "gray_zone_trace_audit": gray_zone_trace_audit,
        "targets": {
            "operations": ["agent_run", "answer_session", "retrieval_trace", "context_package", "reward_event", "policy_state"],
            "knowledge_base_id": knowledge_base_id,
            "request_count": len(rows),
        },
        "impact": "persisted QA/Agent trace and downstream audit records",
    }
    report = write_report("evaluate_agent_trace", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "checks": checks}, ensure_ascii=False, default=str))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
