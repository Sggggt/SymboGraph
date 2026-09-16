"""Task-local, bounded performance observations without source/provider data."""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import event, text
from sqlalchemy.orm import Session

_CURRENT = ContextVar("build_performance", default=None)


class LatencySummary(BaseModel):
    count: int
    sample_count: int
    success_count: int
    failure_count: int
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    total_ms: float
    active_wall_seconds: float
    success_requests_per_second: float
    sampled: bool


class BuildPerformanceSummary(BaseModel):
    protocol_version: Literal["build_performance_v2"] = "build_performance_v2"
    cold: bool
    elapsed_seconds: float
    stages: dict[str, LatencySummary]
    peak_rss_bytes: int
    peak_container_memory_bytes: int
    cpu_seconds: float
    average_cpu_cores: float
    available_cpu_count: int
    average_cpu_percent: float
    resource_sample_count: int
    deadline_seconds: int | None = None
    deadline_exceeded: bool
    provider_cache_hits: int = 0
    provider_cache_observations: int = 0
    provider_cache_unknown_responses: int = 0
    embedding_vector_count: int = 0
    in_progress_ms: dict[str, list[float]] = {}
    numeric_workspace: dict = {}
    process_io_read_bytes: int | None = None
    process_io_write_bytes: int | None = None
    temporary_space_peak_bytes: int = 0
    database_wait_sample_count: int = 0
    database_wait_observations: dict[str,int] = {}


def current_performance():
    return _CURRENT.get()


def cold_build() -> bool:
    return bool(_CURRENT.get() and _CURRENT.get().cold)


def _process_io():
    path=Path("/proc/self/io")
    if not path.exists(): return {}
    return {key:int(value) for line in path.read_text().splitlines() for key,value in [line.split(":",1)] if key in {"read_bytes","write_bytes"}}


@contextmanager
def tracked_temporary_directory(*,prefix,dir=None):
    current=current_performance()
    with tempfile.TemporaryDirectory(prefix=prefix,dir=dir) as directory:
        root=Path(directory)
        if current: current.temporary_roots.add(root)
        try:
            yield directory
        finally:
            if current:
                current.sample_temporary()
                current.temporary_roots.discard(root)


@event.listens_for(Session, "before_flush")
def _before_flush(session, flush_context, instances):
    if _CURRENT.get() is not None:
        session.info["build_flush_timer"] = time.perf_counter()


@event.listens_for(Session, "after_flush")
def _after_flush(session, flush_context):
    started = session.info.pop("build_flush_timer", None)
    if started is not None and _CURRENT.get() is not None:
        _CURRENT.get().record("database_batch", started, True)


@event.listens_for(Session, "after_soft_rollback")
def _failed_flush(session, previous_transaction):
    started = session.info.pop("build_flush_timer", None)
    if started is not None and _CURRENT.get() is not None:
        _CURRENT.get().record("database_batch", started, False)


@event.listens_for(Session, "before_commit")
def _before_commit(session):
    if _CURRENT.get() is not None:
        session.info["build_commit_timer"] = time.perf_counter()


@event.listens_for(Session, "after_commit")
def _after_commit(session):
    started = session.info.pop("build_commit_timer", None)
    if started is not None and _CURRENT.get() is not None:
        _CURRENT.get().record("database_commit", started, True)


class BuildPerformance:
    def __init__(self, *, cold=False, deadline_seconds=None, queued_seconds=0, batch_id=None):
        self.cold, self.deadline_seconds = bool(cold), deadline_seconds
        self.started = time.perf_counter() - queued_seconds
        self.cpu_started = time.process_time()
        self.samples = defaultdict(list)
        self.counts = defaultdict(lambda: [0, 0, 0.0, None, None])
        self.peak_rss = self.peak_container = self.sample_count = 0
        self.provider_cache_hits = self.provider_cache_observations = 0
        self.provider_cache_unknown_responses = 0
        self.embedding_vector_count = 0
        self.stop = threading.Event()
        self.cancel = threading.Event()
        self.batch_id = batch_id
        self.control_error = None
        self.last_progress = 0.0
        self.last_persist = 0.0
        self.active_operations = {}
        self.operation_sequence = 0
        self.numeric_workspace = {}
        self.io_baseline = _process_io()
        self.io_current = dict(self.io_baseline)
        self.temporary_roots = set()
        self.temporary_peak = 0
        self.database_wait_samples = 0
        self.database_wait_observations = defaultdict(int)

    def check(self):
        if self.control_error:
            raise RuntimeError("Build cancellation control read failed") from self.control_error
        if self.cancel.is_set():
            from app.services.cancellation import IngestionCancelled
            raise IngestionCancelled("ingestion batch cancellation requested")
        if self.deadline_seconds and time.perf_counter()-self.started >= self.deadline_seconds:
            from app.services.cancellation import IngestionCancelled
            raise IngestionCancelled("build_deadline_exceeded")

    def progress(self, phase, completed, total):
        self.check()
        if not self.batch_id or time.perf_counter()-self.last_progress < 5:
            return
        from app.db import SessionLocal
        from app.models import IngestionBatch
        from app.services.ingestion_logs import emit_ingestion_log
        with SessionLocal() as db:
            batch = db.get(IngestionBatch, self.batch_id)
            if batch:
                batch.heartbeat_at = datetime.utcnow()
                batch.stats = {**dict(batch.stats or {}), "file_progress": {"phase":phase,"completed":completed,"total":total}}
                db.commit()
        emit_ingestion_log(self.batch_id,"file_parse_progress","文件解析：原文页面处理中",completed=completed,total=total)
        self.last_progress = time.perf_counter()

    def record(self, phase, started, success):
        end = time.perf_counter()
        elapsed = max(0.0, (end-started)*1000)
        row = self.counts[phase]
        row[0 if success else 1] += 1
        row[2] += elapsed
        row[3] = min(row[3] or started, started)
        row[4] = max(row[4] or end, end)
        if len(self.samples[phase]) < 50_000:
            self.samples[phase].append(elapsed)

    def sample(self):
        # /proc and cgroups are read-only and do not expose process arguments.
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    self.peak_rss = max(self.peak_rss, int(line.split()[1])*1024)
            import resource
            self.peak_rss = max(self.peak_rss, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
        memory = Path("/sys/fs/cgroup/memory.current")
        if memory.exists():
            self.peak_container = max(self.peak_container, int(memory.read_text().strip()))
        self.sample_count += 1
        self.io_current = _process_io()
        self.sample_temporary()

    def sample_temporary(self):
        total=0
        for root in list(self.temporary_roots):
            try:
                paths=list(root.iterdir())
            except FileNotFoundError:
                continue
            for path in paths:
                try:
                    if path.is_file(): total += path.stat().st_size
                except FileNotFoundError:
                    continue  # A completed owned temporary run was removed.
        self.temporary_peak=max(self.temporary_peak,total)

    def sampler(self):
        while not self.stop.wait(1.0):
            try:
                self.sample()
                if self.batch_id:
                    from app.db import SessionLocal
                    from app.models import IngestionBatch
                    from app.services.cancellation import CANCELLING_STATES
                    with SessionLocal() as db:
                        batch = db.get(IngestionBatch, self.batch_id)
                        if db.get_bind().dialect.name == "postgresql":
                            waits=db.execute(text("SELECT wait_event_type, wait_event, count(*) FROM pg_stat_activity WHERE datname=current_database() AND state='active' AND wait_event_type IS NOT NULL GROUP BY wait_event_type,wait_event"))
                            for kind,name,count in waits:
                                self.database_wait_observations[f"{kind}:{name}"] += int(count)
                            self.database_wait_samples += 1
                        if batch and (batch.status in CANCELLING_STATES or (batch.stats or {}).get("cancel_requested")):
                            self.cancel.set()
                        if batch and db.get_bind().dialect.name == "postgresql" and time.perf_counter()-self.last_persist >= 5:
                            db.execute(text("UPDATE ingestion_batches SET stats = jsonb_set(COALESCE(stats::jsonb, '{}'::jsonb), '{performance_live}', CAST(:performance AS jsonb), true)::json WHERE id = :batch_id"),
                                       {"performance":json.dumps(self.summary()),"batch_id":self.batch_id})
                            db.commit()
                            self.last_persist = time.perf_counter()
            except Exception as exc:
                self.control_error = exc
                return

    def summary(self):
        elapsed = time.perf_counter()-self.started
        cpu = time.process_time()-self.cpu_started
        stages = {}
        for phase, (good, bad, total, start, end) in list(self.counts.items()):
            values = sorted(self.samples[phase])
            def percentile(q):
                return values[max(0, math.ceil(len(values)*q)-1)] if values else None
            wall = max(1e-9, end-start)
            stages[phase] = LatencySummary(count=good+bad, sample_count=len(values), success_count=good,
                failure_count=bad, total_ms=total, active_wall_seconds=wall, success_requests_per_second=good/wall,
                p50_ms=percentile(.5), p95_ms=percentile(.95), p99_ms=percentile(.99), sampled=len(values)<good+bad)
        active = defaultdict(list)
        now = time.perf_counter()
        for phase, started in list(self.active_operations.values()):
            active[phase].append((now-started)*1000)
        return BuildPerformanceSummary(cold=self.cold, elapsed_seconds=elapsed, stages=stages,
            peak_rss_bytes=self.peak_rss, peak_container_memory_bytes=self.peak_container, cpu_seconds=cpu,
            average_cpu_cores=cpu/max(elapsed,1e-9), available_cpu_count=os.cpu_count() or 1,
            average_cpu_percent=100*cpu/max(elapsed,1e-9)/(os.cpu_count() or 1), resource_sample_count=self.sample_count,
            deadline_seconds=self.deadline_seconds, deadline_exceeded=bool(self.deadline_seconds and elapsed>self.deadline_seconds),
            provider_cache_hits=self.provider_cache_hits, provider_cache_observations=self.provider_cache_observations,
            provider_cache_unknown_responses=self.provider_cache_unknown_responses,
            embedding_vector_count=self.embedding_vector_count,in_progress_ms=dict(active),numeric_workspace=self.numeric_workspace,
            process_io_read_bytes=(self.io_current.get("read_bytes",0)-self.io_baseline.get("read_bytes",0)) if self.io_baseline else None,
            process_io_write_bytes=(self.io_current.get("write_bytes",0)-self.io_baseline.get("write_bytes",0)) if self.io_baseline else None,
            temporary_space_peak_bytes=self.temporary_peak,database_wait_sample_count=self.database_wait_samples,
            database_wait_observations=dict(self.database_wait_observations)).model_dump(mode="json")


@contextmanager
def measure(phase, *, started_at=None):
    current = _CURRENT.get()
    if current is None:
        yield
        return
    current.check()
    start, success = (time.perf_counter() if started_at is None else started_at), False
    current.operation_sequence += 1
    operation = current.operation_sequence
    current.active_operations[operation] = phase, start
    try:
        yield
        success = True
    finally:
        if current:
            current.record(phase, start, success)
            current.active_operations.pop(operation,None)


def measured(phase, *, start_resolver=None):
    def decorate(fn):
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def wrapped(*args, **kwargs):
                with measure(phase,started_at=start_resolver(args,kwargs) if start_resolver else None):
                    current = _CURRENT.get()
                    if current and current.deadline_seconds:
                        remaining = current.deadline_seconds - (time.perf_counter()-current.started)
                        try:
                            async with asyncio.timeout(max(.001, remaining)):
                                return await fn(*args, **kwargs)
                        except TimeoutError:
                            current.check()
                            raise
                    return await fn(*args, **kwargs)
        else:
            @wraps(fn)
            def wrapped(*args, **kwargs):
                with measure(phase):
                    return fn(*args, **kwargs)
        return wrapped
    return decorate


def instrument_build(fn):
    @wraps(fn)
    async def wrapped(batch_id, *args, **kwargs):
        if _CURRENT.get():
            return await fn(batch_id, *args, **kwargs)
        from app.db import SessionLocal
        from app.models import IngestionBatch
        with SessionLocal() as db:
            batch = db.get(IngestionBatch, batch_id)
            control = dict((batch.stats or {}).get("benchmark_control") or {}) if batch else {}
            queued = max(0, (datetime.utcnow()-batch.created_at).total_seconds()) if batch else 0
        performance = BuildPerformance(cold=control.get("cold", False), deadline_seconds=control.get("deadline_seconds"), queued_seconds=queued, batch_id=batch_id)
        token = _CURRENT.set(performance)
        sampler = threading.Thread(target=performance.sampler, daemon=True, name="build-resource-sampler")
        sampler.start()
        try:
            return await fn(batch_id, *args, **kwargs)
        finally:
            performance.stop.set()
            sampler.join(timeout=2)
            performance.sample()
            try:
                with SessionLocal() as db:
                    batch = db.get(IngestionBatch, batch_id)
                    if batch:
                        batch.stats = {**dict(batch.stats or {}), "performance": performance.summary()}
                        db.commit()
            finally:
                _CURRENT.reset(token)
    return wrapped
