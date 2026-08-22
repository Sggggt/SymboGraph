from __future__ import annotations

import asyncio
import time
import uuid

import pytest


def admission_config(*, active: int = 1, queued: int = 2, timeout: int = 10, ttl: int = 30):
    from app.services.agent_admission import AgentAdmissionConfig

    return AgentAdmissionConfig(
        active_limit=active,
        queue_limit=queued,
        queue_timeout_seconds=timeout,
        lease_ttl_seconds=ttl,
    )


async def wait_for_snapshot(adapter, *, active: int, queued: int) -> None:
    for _ in range(100):
        if await adapter.snapshot() == {"active": active, "queued": queued}:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"admission snapshot did not reach active={active}, queued={queued}: {await adapter.snapshot()}")


@pytest.mark.asyncio
async def test_bounded_queue_is_fifo_and_rejects_when_full():
    from app.services.agent_admission import AgentAdmissionError, LocalAgentAdmissionAdapter, acquire_agent_request_slot

    adapter = LocalAgentAdmissionAdapter()
    config = admission_config(queued=2)
    first = await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)
    second_task = asyncio.create_task(acquire_agent_request_slot("sse", adapter=adapter, config=config))
    third_task = asyncio.create_task(acquire_agent_request_slot("ordinary", adapter=adapter, config=config))
    await wait_for_snapshot(adapter, active=1, queued=2)

    with pytest.raises(AgentAdmissionError) as rejected:
        await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)
    assert rejected.value.code == "agent_admission_queue_full"
    assert rejected.value.status_code == 429
    assert rejected.value.payload()["diagnostics"]["queue_limit"] == 2
    assert rejected.value.payload()["issues"][0]["code"] == "agent_admission_queue_full"

    await first.release()
    second = await asyncio.wait_for(second_task, timeout=1)
    assert not third_task.done()
    await second.release()
    third = await asyncio.wait_for(third_task, timeout=1)
    await third.release()
    assert await adapter.snapshot() == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_queue_timeout_removes_waiter_without_starting_work():
    from app.services.agent_admission import AgentAdmissionError, LocalAgentAdmissionAdapter, acquire_agent_request_slot

    adapter = LocalAgentAdmissionAdapter()
    config = admission_config(queued=1, timeout=0)
    active = await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)

    with pytest.raises(AgentAdmissionError) as timed_out:
        await acquire_agent_request_slot("sse", adapter=adapter, config=config)
    assert timed_out.value.code == "agent_admission_queue_timeout"
    assert timed_out.value.status_code == 429
    assert await adapter.snapshot() == {"active": 1, "queued": 0}

    await active.release()


@pytest.mark.asyncio
async def test_cancelled_waiter_and_cancelled_active_work_release_capacity():
    from app.services.agent_admission import LocalAgentAdmissionAdapter, acquire_agent_request_slot

    adapter = LocalAgentAdmissionAdapter()
    config = admission_config(queued=1)
    first = await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)
    waiter = asyncio.create_task(acquire_agent_request_slot("sse", adapter=adapter, config=config))
    await wait_for_snapshot(adapter, active=1, queued=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert await adapter.snapshot() == {"active": 1, "queued": 0}

    work_started = asyncio.Event()

    async def long_work():
        work_started.set()
        await asyncio.sleep(30)

    work = asyncio.create_task(first.run(long_work()))
    await work_started.wait()
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    await first.release()

    replacement = await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)
    await replacement.release()
    assert await adapter.snapshot() == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_execution_exception_releases_capacity_in_caller_finally():
    from app.services.agent_admission import LocalAgentAdmissionAdapter, acquire_agent_request_slot

    adapter = LocalAgentAdmissionAdapter()
    config = admission_config(queued=0)
    lease = await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)

    async def fail():
        raise ValueError("unit failure")

    try:
        with pytest.raises(ValueError, match="unit failure"):
            await lease.run(fail())
    finally:
        await lease.release()

    replacement = await acquire_agent_request_slot("sse", adapter=adapter, config=config)
    await replacement.release()


@pytest.mark.asyncio
async def test_heartbeat_failure_cancels_active_work_and_fails_fast():
    from app.services.agent_admission import AgentAdmissionError, LocalAgentAdmissionAdapter, acquire_agent_request_slot

    class BrokenHeartbeatAdapter(LocalAgentAdmissionAdapter):
        async def heartbeat(self, token, config, *, now_ms):
            raise ConnectionError("redis unavailable")

    adapter = BrokenHeartbeatAdapter()
    config = admission_config(queued=0, ttl=1)
    lease = await acquire_agent_request_slot("ordinary", adapter=adapter, config=config)
    cancelled = asyncio.Event()

    async def long_work():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    started = time.monotonic()
    with pytest.raises(AgentAdmissionError) as unavailable:
        await asyncio.wait_for(lease.run(long_work()), timeout=2)
    assert unavailable.value.code == "agent_admission_unavailable"
    assert cancelled.is_set()
    assert time.monotonic() - started < 2
    await lease.release()


@pytest.mark.asyncio
async def test_redis_unavailable_rejects_without_local_fallback():
    from app.services.agent_admission import AgentAdmissionError, RedisAgentAdmissionAdapter, acquire_agent_request_slot

    class BrokenRedisClient:
        calls = 0

        async def eval(self, *args):
            self.calls += 1
            raise ConnectionError("unit redis outage")

        async def aclose(self):
            return None

    client = BrokenRedisClient()
    adapter = RedisAgentAdmissionAdapter("redis://not-used", client=client)
    with pytest.raises(AgentAdmissionError) as unavailable:
        await acquire_agent_request_slot("ordinary", adapter=adapter, config=admission_config())
    assert unavailable.value.code == "agent_admission_unavailable"
    assert unavailable.value.status_code == 503
    assert "no local fallback" in unavailable.value.message
    assert client.calls >= 1


@pytest.mark.asyncio
async def test_independent_redis_adapters_share_one_global_active_budget():
    import redis.asyncio as redis

    from app.core.config import get_settings
    from app.services.agent_admission import AgentAdmissionError, RedisAgentAdmissionAdapter, acquire_agent_request_slot

    redis_url = get_settings().redis_url
    probe = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    try:
        await probe.ping()
    except Exception:
        await probe.aclose()
        pytest.skip("Redis integration endpoint is unavailable")

    prefix = f"symbograph:test:agent_admission:{uuid.uuid4().hex}"
    keys = [f"{prefix}:active", f"{prefix}:queue", f"{prefix}:deadlines", f"{prefix}:sequence"]
    config = admission_config(queued=0)
    first_adapter = RedisAgentAdmissionAdapter(redis_url, key_prefix=prefix)
    second_adapter = RedisAgentAdmissionAdapter(redis_url, key_prefix=prefix)
    first = None
    replacement = None
    try:
        first = await acquire_agent_request_slot("ordinary", adapter=first_adapter, config=config)
        with pytest.raises(AgentAdmissionError) as rejected:
            await acquire_agent_request_slot("sse", adapter=second_adapter, config=config)
        assert rejected.value.code == "agent_admission_queue_full"

        await first.release()
        first = None
        replacement = await acquire_agent_request_slot(
            "sse",
            adapter=RedisAgentAdmissionAdapter(redis_url, key_prefix=prefix),
            config=config,
        )
    finally:
        if first is not None:
            await first.release()
        if replacement is not None:
            await replacement.release()
        await probe.delete(*keys)
        await probe.aclose()


@pytest.mark.asyncio
async def test_ordinary_and_sse_share_budget_and_reject_before_run_or_db_session(monkeypatch, db_session, sample_knowledge_base):
    from sqlalchemy import func, select

    import app.db as db_module
    from app.models import AgentRun
    from app.schemas import AgentRequest
    from app.services import agent_admission, agent_graph
    from app.services.agent_admission import AgentAdmissionError, LocalAgentAdmissionAdapter, acquire_agent_request_slot, use_agent_admission_adapter

    adapter = LocalAgentAdmissionAdapter()
    config = admission_config(queued=0)
    monkeypatch.setattr(agent_admission.AgentAdmissionConfig, "from_settings", classmethod(lambda cls: config))
    before = db_session.scalar(select(func.count(AgentRun.id)))

    with use_agent_admission_adapter(adapter):
        blocker = await acquire_agent_request_slot("ordinary")
        with pytest.raises(AgentAdmissionError, match="bounded Agent request queue") as ordinary_rejected:
            await agent_graph.run_agent(
                db_session,
                AgentRequest(knowledge_base_id=sample_knowledge_base.id, question="ordinary must not start"),
            )
        assert ordinary_rejected.value.code == "agent_admission_queue_full"
        assert db_session.scalar(select(func.count(AgentRun.id))) == before

        monkeypatch.setattr(db_module, "SessionLocal", lambda: (_ for _ in ()).throw(AssertionError("SSE opened DB before admission")))
        stream = agent_graph.stream_agent_events(
            AgentRequest(knowledge_base_id=sample_knowledge_base.id, question="SSE must not start", stream_trace=True)
        )
        with pytest.raises(AgentAdmissionError) as sse_rejected:
            await stream.__anext__()
        assert sse_rejected.value.code == "agent_admission_queue_full"
        assert db_session.scalar(select(func.count(AgentRun.id))) == before
        await blocker.release()

    assert await adapter.snapshot() == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_cancelled_ordinary_agent_marks_run_terminal_and_releases_slot(
    monkeypatch, db_session, sample_knowledge_base, local_agent_admission
):
    from sqlalchemy import select

    from app.models import AgentRun
    from app.schemas import AgentRequest
    from app.services import agent_graph

    started = asyncio.Event()

    async def slow_execution(db, request, session, run):
        agent_graph.set_run_state(db, run, "running", current_node="unit_wait")
        started.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(agent_graph, "execute_agent_run_with_active_profile", slow_execution)
    task = asyncio.create_task(
        agent_graph.run_agent(
            db_session,
            AgentRequest(knowledge_base_id=sample_knowledge_base.id, question="cancel ordinary Agent"),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    db_session.expire_all()
    run = db_session.scalar(select(AgentRun).order_by(AgentRun.created_at.desc()))
    assert run is not None
    assert run.status == "cancelled"
    assert run.error_message == agent_graph.CANCELLED_BY_USER
    assert await local_agent_admission.snapshot() == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_lost_ordinary_agent_lease_marks_run_failed_and_releases_slot(monkeypatch, db_session, sample_knowledge_base):
    from sqlalchemy import select

    from app.models import AgentRun, AgentTraceEvent
    from app.schemas import AgentRequest
    from app.services import agent_admission, agent_graph
    from app.services.agent_admission import AgentAdmissionError, LocalAgentAdmissionAdapter, use_agent_admission_adapter

    class BrokenHeartbeatAdapter(LocalAgentAdmissionAdapter):
        async def heartbeat(self, token, config, *, now_ms):
            raise ConnectionError("unit Redis heartbeat outage")

    adapter = BrokenHeartbeatAdapter()
    config = admission_config(queued=0, ttl=1)
    monkeypatch.setattr(agent_admission.AgentAdmissionConfig, "from_settings", classmethod(lambda cls: config))

    async def slow_execution(db, request, session, run):
        agent_graph.set_run_state(db, run, "running", current_node="unit_wait")
        await asyncio.sleep(30)

    monkeypatch.setattr(agent_graph, "execute_agent_run_with_active_profile", slow_execution)
    with use_agent_admission_adapter(adapter):
        with pytest.raises(AgentAdmissionError) as lost:
            await agent_graph.run_agent(
                db_session,
                AgentRequest(knowledge_base_id=sample_knowledge_base.id, question="lose admission lease"),
            )

    assert lost.value.code == "agent_admission_unavailable"
    db_session.expire_all()
    run = db_session.scalar(select(AgentRun).order_by(AgentRun.created_at.desc()))
    assert run is not None
    assert run.status == "failed"
    assert run.error_message == "agent_admission_unavailable"
    trace_nodes = db_session.scalars(select(AgentTraceEvent.node).where(AgentTraceEvent.run_id == run.id)).all()
    assert "agent_admission" in trace_nodes
    assert await adapter.snapshot() == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_lost_sse_lease_emits_structured_error_marks_run_failed_and_releases_slot(monkeypatch, db_session, sample_knowledge_base):
    from app.models import AgentRun
    from app.schemas import AgentRequest
    from app.services import agent_admission, agent_graph
    from app.services.agent_admission import LocalAgentAdmissionAdapter, use_agent_admission_adapter

    class BrokenHeartbeatAdapter(LocalAgentAdmissionAdapter):
        async def heartbeat(self, token, config, *, now_ms):
            raise ConnectionError("unit Redis heartbeat outage")

    adapter = BrokenHeartbeatAdapter()
    config = admission_config(queued=0, ttl=1)
    monkeypatch.setattr(agent_admission.AgentAdmissionConfig, "from_settings", classmethod(lambda cls: config))

    async def slow_execution(db, request, session, run):
        try:
            agent_graph.set_run_state(db, run, "running", current_node="unit_sse_wait")
            await asyncio.sleep(30)
        finally:
            db.close()

    monkeypatch.setattr(agent_graph, "_execute_agent_run_and_close", slow_execution)
    with use_agent_admission_adapter(adapter):
        stream = agent_graph.stream_agent_events(
            AgentRequest(
                knowledge_base_id=sample_knowledge_base.id,
                question="lose SSE admission lease",
                stream_trace=True,
            )
        )
        meta = await stream.__anext__()
        error_event = await asyncio.wait_for(stream.__anext__(), timeout=2)
        assert error_event["type"] == "error"
        assert error_event["detail"]["code"] == "agent_admission_unavailable"
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    db_session.expire_all()
    run = db_session.get(AgentRun, meta["run_id"])
    assert run is not None
    assert run.status == "failed"
    assert run.error_message == "agent_admission_unavailable"
    assert await adapter.snapshot() == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_agent_routes_return_structured_rejection_before_opening_db_session(monkeypatch):
    from fastapi import HTTPException

    import app.db as db_module
    from app.routers import search
    from app.schemas import AgentRequest, QARequest
    from app.services.agent_admission import AgentAdmissionError

    error = AgentAdmissionError(
        code="agent_admission_queue_full",
        status_code=429,
        message="bounded queue full",
        config=admission_config(queued=0),
        request_kind="sse",
        active_count=1,
        queue_count=0,
    )

    requested_kinds: list[str] = []

    async def reject_before_db(request_kind):
        requested_kinds.append(request_kind)
        raise error

    monkeypatch.setattr(search, "acquire_agent_request_slot", reject_before_db)
    monkeypatch.setattr(db_module, "SessionLocal", lambda: (_ for _ in ()).throw(AssertionError("DB session opened before admission")))

    with pytest.raises(HTTPException) as rejected:
        await search.qa_stream(QARequest(question="must reject before DB"))
    assert rejected.value.status_code == 429
    assert rejected.value.detail["code"] == "agent_admission_queue_full"
    assert rejected.value.headers == {"Retry-After": "1"}

    with pytest.raises(HTTPException):
        await search.qa(QARequest(question="ordinary QA must reject before DB"))
    with pytest.raises(HTTPException):
        await search.agent_call(AgentRequest(question="ordinary Agent must reject before DB"))
    assert requested_kinds == ["sse", "qa", "agent"]

    error = AgentAdmissionError(
        code="agent_admission_unavailable",
        status_code=503,
        message="Redis admission unavailable",
        config=admission_config(),
        request_kind="agent",
    )
    with pytest.raises(HTTPException) as unavailable:
        await search.agent_call(AgentRequest(question="Redis outage must fail closed"))
    assert unavailable.value.status_code == 503
    assert unavailable.value.detail["code"] == "agent_admission_unavailable"
    assert unavailable.value.detail["retryable"] is True


@pytest.mark.asyncio
async def test_agent_routes_release_admission_when_db_session_creation_fails(monkeypatch):
    from fastapi import HTTPException

    import app.db as db_module
    from app.routers import search
    from app.schemas import AgentRequest, QARequest

    leases = []

    class TrackingLease:
        def __init__(self):
            self.release_calls = 0

        async def release(self):
            self.release_calls += 1

        def raise_if_lost(self):
            return None

    async def admit(_request_kind):
        lease = TrackingLease()
        leases.append(lease)
        return lease

    monkeypatch.setattr(search, "acquire_agent_request_slot", admit)
    monkeypatch.setattr(db_module, "SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("unit DB creation failure")))

    with pytest.raises(HTTPException) as failed_agent:
        await search.agent_call(AgentRequest(question="release ordinary lease"))
    assert failed_agent.value.status_code == 502
    assert failed_agent.value.detail == {
        "code": "agent_request_failed",
        "message": "unit DB creation failure",
    }
    with pytest.raises(RuntimeError, match="unit DB creation failure"):
        await search.qa_stream(QARequest(question="release SSE lease"))

    assert [lease.release_calls for lease in leases] == [1, 1]
