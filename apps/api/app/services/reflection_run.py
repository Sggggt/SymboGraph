"""Durable run identity and continuation of the one answer-reflection budget."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.models import AgentObservation, AgentRun
from app.services.agent_reflection import ReflectionContractError


REFLECTION_RUN_LEDGER_PROTOCOL = "reflection_run_ledger_v1"
MAX_REFLECTION_EVENTS = 128


def ordered_reflection_events(db: Session, *, run_id: str) -> list[dict[str, Any]]:
    """Replay causal order independently of wall-clock adjustment or ties."""
    events = list(db.scalars(select(AgentObservation.observation_json).where(
        AgentObservation.run_id == run_id, AgentObservation.observation_type == "answer_reflection",
    ).limit(MAX_REFLECTION_EVENTS + 1)))
    if len(events) > MAX_REFLECTION_EVENTS or any(
        not isinstance(event, dict) or event.get("protocol_version") != "agent_answer_reflection_v1"
        or type(event.get("sequence_index")) is not int for event in events
    ):
        raise ReflectionContractError("reflection_observation_sequence_invalid")
    events.sort(key=lambda event: event["sequence_index"])
    if any(event["sequence_index"] != index for index, event in enumerate(events)):
        raise ReflectionContractError("reflection_observation_sequence_invalid")
    return deepcopy(events)


def load_reflection_run_ledger(
    db: Session, *, run: AgentRun, requested_limit: int, runtime_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if type(requested_limit) is not int or not 0 <= requested_limit <= 10:
        raise ReflectionContractError("reflection_run_budget_invalid")
    db.scalar(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
    question_hash = hashlib.sha256(run.question.encode("utf-8")).hexdigest()
    previous = ordered_reflection_events(db, run_id=run.id)
    ledger = (run.metadata_json or {}).get("reflection_run_ledger")
    if ledger is None:
        if previous:
            raise ReflectionContractError("reflection_run_ledger_missing")
        ledger = {"protocol_version": REFLECTION_RUN_LEDGER_PROTOCOL, "run_id": run.id,
            "question_hash": question_hash, "runtime_settings_hash": runtime_hash,
            "hard_limit": requested_limit, "started_at": datetime.now(timezone.utc).isoformat()}
        run.metadata_json = {**(run.metadata_json or {}), "reflection_run_ledger": ledger}
        flag_modified(run, "metadata_json")
        db.flush()
        return deepcopy(ledger), []
    if (
        not isinstance(ledger, dict)
        or set(ledger) != {"protocol_version", "run_id", "question_hash", "runtime_settings_hash", "hard_limit", "started_at"}
        or ledger.get("protocol_version") != REFLECTION_RUN_LEDGER_PROTOCOL
        or ledger.get("run_id") != run.id or ledger.get("question_hash") != question_hash
        or ledger.get("runtime_settings_hash") != runtime_hash
        or type(ledger.get("hard_limit")) is not int or not 0 <= ledger["hard_limit"] <= 10
    ):
        raise ReflectionContractError("reflection_run_ledger_identity_changed")
    try:
        started = datetime.fromisoformat(ledger["started_at"])
        if started.tzinfo is None:
            raise ValueError
    except (TypeError, ValueError):
        raise ReflectionContractError("reflection_run_ledger_time_invalid") from None
    if (
        not previous or len(previous) > MAX_REFLECTION_EVENTS
        or any(not isinstance(event, dict) or event.get("protocol_version") != "agent_answer_reflection_v1"
               or type(event.get("sequence_index")) is not int or event["sequence_index"] != index
               for index, event in enumerate(previous))
        or previous[-1].get("stage") != "retrieval_handoff"
        or previous[-1].get("status") != "completed"
        or previous[-1].get("round_budget") != ledger["hard_limit"]
    ):
        raise ReflectionContractError("reflection_run_continuation_not_at_handoff")
    return deepcopy(ledger), deepcopy(previous)


def reflection_run_elapsed_seconds(ledger: dict[str, Any]) -> float:
    return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(ledger["started_at"])).total_seconds())
