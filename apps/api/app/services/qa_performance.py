"""Bounded request-local timings; no prompts, credentials or provider bodies."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import math
import threading
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Stage = Literal[
    "request", "admission_queue", "conversation_prepare", "task_planning",
    "history_projection", "capability_manifest", "intent_planning", "resource_read", "context_reuse",
    "graph_admission", "embedding", "retrieval", "candidate_discovery",
    "path_features", "feature_preparation", "lexical_repair", "evidence_sufficiency", "packing", "context_package", "source_admission", "generation", "source_binding",
    "database_commit", "audit_persistence", "model_call", "model_queue",
    "provider_roundtrip", "bridge_sync", "source_io_queue", "source_io",
    "graph_dense_scoring", "graph_entry_selection", "graph_traversal", "graph_trace_write",
    "graph_edge_read", "source_scope_resolution", "source_location_model", "generation_packing",
    "dense_entry", "rq_entry", "bm25_entry", "entry_fusion", "structure_restore",
]


class TimingFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["chat", "graph", "embedding"] | None = None
    attempt: int | None = Field(default=None, ge=1, le=32)
    round_index: int | None = Field(default=None, ge=0, le=10)
    input_characters: int | None = Field(default=None, ge=0)
    output_token_budget: int | None = Field(default=None, ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    item_count: int | None = Field(default=None, ge=0)
    http_status: int | None = Field(default=None, ge=100, le=599)
    cache_hit: bool | None = None


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int
    parent_sequence: int | None
    stage: Stage
    start_ms: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    exclusive_ms: float = Field(ge=0, allow_inf_nan=False)
    status: Literal["ok", "error", "cancelled", "running"]
    error_type: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    fields: TimingFields


class QAStageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    count: int
    success_count: int
    error_count: int
    cancelled_count: int
    total_ms: float
    active_wall_ms: float
    exclusive_ms: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


class QAPerformanceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_version: Literal["qa_stage_timing_v1"] = "qa_stage_timing_v1"
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    stages: dict[str, QAStageSummary]
    spans: tuple[StageTiming, ...]
    unfinished_span_count: int
    quantile_method: Literal["nearest_rank"] = "nearest_rank"
    clock: Literal["monotonic"] = "monotonic"
    first_response_ms: float | None = None
    first_token_ms: float | None = None
    provider_compute_ms: float | None = None
    provider_detail_availability: Literal["roundtrip_only"] = "roundtrip_only"


class QAAuditCapacityError(RuntimeError):
    pass


def interval_union(intervals):
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    total = 0.0
    if not ordered:
        return total
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


@dataclass
class _Span:
    sequence: int
    parent: int | None
    stage: Stage
    start: float
    fields: TimingFields = field(default_factory=TimingFields)
    end: float | None = None
    status: str = "running"
    error_type: str | None = None

    def annotate(self, **fields):
        self.fields = TimingFields.model_validate({**self.fields.model_dump(), **fields})


_CURRENT: ContextVar[QAPerformance | None] = ContextVar("qa_performance", default=None)
_PARENT: ContextVar[int | None] = ContextVar("qa_performance_parent", default=None)


class QAPerformance:
    def __init__(self, *, clock=time.perf_counter, span_limit=512):
        self.clock = clock
        self.started = clock()
        self.limit = span_limit
        self._spans: list[_Span] = []
        self._lock = threading.RLock()
        self._first_response_ms: float | None = None
        self._first_token_ms: float | None = None

    def mark_first_response(self, *, token: bool = False) -> None:
        """Record the first answer payload made observable to the client."""

        elapsed_ms = round(max(0.0, self.clock() - self.started) * 1000, 6)
        with self._lock:
            if self._first_response_ms is None:
                self._first_response_ms = elapsed_ms
            if token and self._first_token_ms is None:
                self._first_token_ms = elapsed_ms

    def begin(self, stage: Stage, *, parent=None, **fields):
        # Validate before starting any observed work.
        metadata = TimingFields.model_validate(fields)
        if stage not in Stage.__args__:
            raise ValueError("qa_timing_stage_not_allowlisted")
        with self._lock:
            if len(self._spans) >= self.limit:
                raise QAAuditCapacityError("qa_timing_span_budget_exceeded")
            if parent is not None and not 0 <= parent < len(self._spans):
                raise ValueError("qa_timing_parent_invalid")
            span = _Span(len(self._spans), parent, stage, self.clock(), metadata)
            self._spans.append(span)
        return span

    def finish(self, span, exc=None):
        with self._lock:
            if span.end is not None:
                raise ValueError("qa_timing_span_already_finished")
            span.end = self.clock()
            span.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error" if exc is not None else "ok"
            span.error_type = type(exc).__name__[:80] if exc is not None else None

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        parent = _PARENT.set(None)
        try:
            yield self
        finally:
            _PARENT.reset(parent)
            _CURRENT.reset(token)

    def snapshot(self) -> QAPerformanceSummary:
        with self._lock:
            now = self.clock()
            rows = []
            for span in self._spans:
                end = span.end if span.end is not None else now
                duration = max(0.0, end - span.start)
                child_intervals = [
                    (max(span.start, child.start), min(end, child.end if child.end is not None else now))
                    for child in self._spans if child.parent == span.sequence
                ]
                rows.append(StageTiming(
                    sequence=span.sequence, parent_sequence=span.parent, stage=span.stage,
                    start_ms=round(max(0.0, span.start - self.started) * 1000, 6),
                    duration_ms=round(duration * 1000, 6),
                    exclusive_ms=round(max(0.0, duration - interval_union(child_intervals)) * 1000, 6),
                    status=span.status, error_type=span.error_type, fields=span.fields,
                ))
            stages = {}
            for stage in sorted({row.stage for row in rows}):
                selected = [row for row in rows if row.stage == stage]
                finished = [row.duration_ms for row in selected if row.status != "running"]
                stages[stage] = QAStageSummary(
                    count=len(selected), success_count=sum(row.status == "ok" for row in selected),
                    error_count=sum(row.status == "error" for row in selected),
                    cancelled_count=sum(row.status == "cancelled" for row in selected),
                    total_ms=round(sum(row.duration_ms for row in selected), 6),
                    active_wall_ms=round(interval_union([(row.start_ms, row.start_ms + row.duration_ms) for row in selected]), 6),
                    exclusive_ms=round(sum(row.exclusive_ms for row in selected), 6),
                    p50_ms=nearest_rank(finished, .5), p95_ms=nearest_rank(finished, .95),
                    p99_ms=nearest_rank(finished, .99),
                )
            return QAPerformanceSummary(elapsed_ms=round(max(0.0, now - self.started) * 1000, 6),
                stages=stages, spans=tuple(rows), unfinished_span_count=sum(row.status == "running" for row in rows),
                first_response_ms=self._first_response_ms, first_token_ms=self._first_token_ms)


def current_qa_performance():
    return _CURRENT.get()


@contextmanager
def qa_stage(stage: Stage, **fields):
    recorder = _CURRENT.get()
    if recorder is None:
        yield None
        return
    span = recorder.begin(stage, parent=_PARENT.get(), **fields)
    token = _PARENT.set(span.sequence)
    try:
        yield span
    except BaseException as exc:
        status = getattr(exc, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            span.annotate(http_status=status)
        recorder.finish(span, exc)
        raise
    else:
        recorder.finish(span)
    finally:
        _PARENT.reset(token)


def qa_timed(stage: Stage, **fields):
    def decorator(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            with qa_stage(stage, **fields):
                return await function(*args, **kwargs)
        return wrapped
    return decorator


def qa_request_scope(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        recorder = current_qa_performance() or QAPerformance()
        with recorder.activate():
            return await function(*args, **kwargs)
    return wrapped


def qa_sync_timed(stage: Stage, **fields):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with qa_stage(stage, **fields):
                return function(*args, **kwargs)
        return wrapped
    return decorator
