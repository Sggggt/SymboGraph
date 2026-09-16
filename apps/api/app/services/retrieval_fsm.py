"""Durable retrieval-first transitions; no edge returns from generation to search."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, model_validator, model_serializer
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.models import AgentObservation, AgentRun
from app.retrieval_control_contracts import ControlContract, LexicalStrategy, TaskContract, control_hash
from app.services.qa_performance import qa_stage


State = Literal["planning", "task_ready", "scope_resolving", "reuse_check", "searching", "packing", "diagnosing", "evaluating",
                "discovering", "patching", "restoring", "ready", "generating", "binding",
                "completed", "insufficient", "failed", "cancelled"]
TERMINAL = frozenset({"completed", "insufficient", "failed", "cancelled"})
ALLOWED = {
    "planning": {"task_ready", "completed", "insufficient", "failed", "cancelled"},
    "task_ready": {"scope_resolving", "reuse_check", "searching", "insufficient", "failed", "cancelled"},
    "scope_resolving": {"task_ready", "failed", "cancelled"},
    "reuse_check": {"diagnosing", "searching", "failed", "cancelled"},
    "searching": {"packing", "failed", "cancelled"},
    "packing": {"diagnosing", "failed", "cancelled"},
    "diagnosing": {"evaluating", "discovering", "restoring", "ready", "searching", "insufficient", "failed", "cancelled"},
    "evaluating": {"diagnosing", "failed", "cancelled"},
    "discovering": {"patching", "restoring", "insufficient", "failed", "cancelled"},
    "patching": {"searching", "insufficient", "failed", "cancelled"},
    "restoring": {"packing", "failed", "cancelled"},
    "ready": {"generating", "failed", "cancelled"},
    "generating": {"binding", "failed", "cancelled"},
    "binding": {"completed", "failed", "cancelled"},
}


class RetrievalControlState(ControlContract):
    protocol_version: Literal["retrieval_fsm_v1"] = "retrieval_fsm_v1"
    state: State = "planning"
    sequence_index: int = Field(default=0, ge=0, le=128)
    runtime_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    strategy_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    repair_limit: int = Field(ge=0, le=2)
    repairs_used: int = Field(default=0, ge=0, le=2)
    generation_started: bool = False
    scope_resolution_started: bool = False
    executed_strategy_hashes: tuple[str, ...] = Field(default=(), max_length=3)

    @model_serializer(mode='wrap')
    def preserve_historical_state(self,handler):
        payload=handler(self)
        if not self.scope_resolution_started:
            payload.pop('scope_resolution_started',None)
        return payload

    @model_validator(mode="after")
    def budget(self):
        if self.repairs_used > self.repair_limit:
            raise ValueError("retrieval_fsm_repair_budget_exceeded")
        if len(set(self.executed_strategy_hashes)) != len(self.executed_strategy_hashes):
            raise ValueError("retrieval_fsm_duplicate_strategy_execution")
        return self


def next_control_state(current: RetrievalControlState, target: State, *,
                       task_hash: str | None = None, strategy_hash: str | None = None,
                       system_capability: bool = False) -> RetrievalControlState:
    if current.state in TERMINAL or target not in ALLOWED.get(current.state, set()):
        raise ValueError("retrieval_fsm_transition_not_allowed")
    if current.task_hash is not None and task_hash is not None and current.task_hash != task_hash:
        raise ValueError("retrieval_fsm_task_identity_changed")
    task = task_hash or current.task_hash
    strategy = strategy_hash or current.strategy_hash
    if target not in {"failed", "cancelled"} and not task and not system_capability:
        raise ValueError("retrieval_fsm_task_not_ready")
    if target == "completed" and current.state == "planning" and not system_capability:
        raise ValueError("retrieval_fsm_only_capability_may_complete_without_source")
    repairs = current.repairs_used
    if target == "restoring" or (current.state == "patching" and target == "searching"):
        repairs += 1
    executed = current.executed_strategy_hashes
    if target == "searching":
        if current.state == "diagnosing" and current.executed_strategy_hashes:
            raise ValueError("retrieval_fsm_direct_search_only_for_reuse_fallback")
        if strategy is None or strategy in executed:
            raise ValueError("retrieval_fsm_strategy_already_executed")
        executed = (*executed, strategy)
    generation_started = current.generation_started
    scope_resolution_started=current.scope_resolution_started
    if target=='scope_resolving':
        if scope_resolution_started or current.executed_strategy_hashes or generation_started:
            raise ValueError('source_location_call_budget_exhausted')
        scope_resolution_started=True
    if target == "generating":
        if generation_started:
            raise ValueError("retrieval_fsm_generation_already_started")
        generation_started = True
    return RetrievalControlState(
        state=target, sequence_index=current.sequence_index + 1, runtime_hash=current.runtime_hash,
        task_hash=task, strategy_hash=strategy, repair_limit=current.repair_limit, repairs_used=repairs,
        generation_started=generation_started,scope_resolution_started=scope_resolution_started, executed_strategy_hashes=executed)


def initialize_control(db, *, run: AgentRun, runtime_hash: str, repair_limit: int):
    row = db.scalar(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
    if row is None or row.status not in {"queued", "running"}:
        raise ValueError("retrieval_fsm_run_not_startable")
    if (row.metadata_json or {}).get("retrieval_control"):
        # Resume must inspect durable phase and outstanding I/O; never reset
        # an existing request's generation or repair budget.
        raise ValueError("retrieval_fsm_existing_run_requires_phase_recovery")
    state = RetrievalControlState(runtime_hash=runtime_hash, repair_limit=repair_limit)
    row.metadata_json = {**(row.metadata_json or {}), "retrieval_control": state.model_dump(mode="json")}
    row.status, row.current_node = "running", "task_planning"
    row.started_at = row.started_at or datetime.now(timezone.utc).replace(tzinfo=None)
    flag_modified(row, "metadata_json")
    with qa_stage("database_commit"):
        db.commit()
    return state


def advance_control(db, *, run: AgentRun, target: State, task: TaskContract | None = None,
                    strategy: LexicalStrategy | None = None, system_capability=False, commit=True):
    row = db.scalar(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
    if row is None:
        raise ValueError("retrieval_fsm_run_missing")
    if row.status == "cancelled" and target != "cancelled":
        raise asyncio.CancelledError("cancelled_by_user")
    current = RetrievalControlState.model_validate((row.metadata_json or {}).get("retrieval_control"))
    updated = next_control_state(current, target, task_hash=task.identity if task else None,
        strategy_hash=strategy.identity if strategy else None, system_capability=system_capability)
    metadata = {**(row.metadata_json or {}), "retrieval_control": updated.model_dump(mode="json")}
    if task is not None:
        metadata["retrieval_task"] = task.model_dump(mode="json")
    if strategy is not None:
        metadata["lexical_strategy"] = strategy.model_dump(mode="json")
    row.metadata_json = metadata
    row.current_node = target
    if target in TERMINAL:
        row.status = {"completed": "completed", "insufficient": "needs_clarification",
                      "failed": "failed", "cancelled": "cancelled"}[target]
        row.current_node = None
        row.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    event = {"protocol_version": "retrieval_fsm_transition_v1", "run_id": row.id,
             "sequence_index": updated.sequence_index, "before": current.model_dump(mode="json"),
             "after": updated.model_dump(mode="json")}
    event["event_hash"] = control_hash(event)
    db.add(AgentObservation(run_id=row.id, observation_type="retrieval_state_transition",
                            verdict=target, observation_json=event))
    flag_modified(row, "metadata_json")
    if commit:
        with qa_stage("database_commit"):
            db.commit()
    else:
        db.flush()
    return updated
