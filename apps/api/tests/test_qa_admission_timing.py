import asyncio
from contextlib import nullcontext

import pytest

from app.schemas import QARequest
from app.services.qa_performance import current_qa_performance


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_router_admission_time_reaches_controller_without_context_leak(no_fallback_env, monkeypatch, streaming):
    import app.db as db
    from app.routers import search
    class Lease:
        def raise_if_lost(self):
            pass
        async def release(self):
            pass
    async def admit(kind):
        await asyncio.sleep(.02)
        return Lease()
    def validate(performance):
        summary = performance.snapshot()
        assert summary.stages["admission_queue"].count == 1
        assert summary.stages["admission_queue"].active_wall_ms >= 10
        assert summary.unfinished_span_count == 0
    async def run(*args, **kwargs):
        validate(current_qa_performance())
        return {"answer": "unit-test"}
    async def stream(*args, performance=None, **kwargs):
        assert current_qa_performance() is None
        validate(performance)
        yield {"type": "meta"}
    monkeypatch.setattr(search, "acquire_agent_request_slot", admit)
    monkeypatch.setattr(db, "SessionLocal", lambda: nullcontext(object()))
    monkeypatch.setattr(search, "get_requested_knowledge_base", lambda *args: None)
    monkeypatch.setattr(search, "run_agent", run)
    monkeypatch.setattr(search, "stream_agent_events", stream)
    if streaming:
        response = await search.qa_stream(QARequest(question="unit-test"))
        assert response.headers["cache-control"] == "no-cache, no-transform"
        assert response.headers["connection"] == "keep-alive"
        assert response.headers["x-accel-buffering"] == "no"
        assert [part async for part in response.body_iterator]
    else:
        assert (await search.qa(QARequest(question="unit-test")))["answer"] == "unit-test"
    assert current_qa_performance() is None


@pytest.mark.asyncio
async def test_sse_framing_emits_transport_heartbeats_without_cancelling_agent_wait():
    from app.routers.search import stream_sse_frames

    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def events():
        yield {"type": "meta", "run_id": "unit-test-run"}
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield {"type": "final", "response": {"answer": "done"}}

    framed = stream_sse_frames(events(), heartbeat_seconds=0.01)
    assert '"type": "meta"' in await framed.__anext__()
    assert await asyncio.wait_for(framed.__anext__(), timeout=0.2) == (
        ": keep-alive\n\n"
    )
    assert cancelled.is_set() is False
    release.set()
    assert '"type": "final"' in await asyncio.wait_for(
        framed.__anext__(), timeout=0.2
    )
    await framed.aclose()
    assert cancelled.is_set() is False
