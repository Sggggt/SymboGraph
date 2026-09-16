import asyncio
from threading import Event, get_ident

import pytest


async def wait_for_thread(event, timeout=3):
    async with asyncio.timeout(timeout):
        while not event.is_set():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancel_keeps_slot_until_worker_exits_even_after_repeat_cancel(monkeypatch, worker_fails):
    from app.core.config import get_settings
    from app.services.storage import run_bounded_source_io
    monkeypatch.setattr(get_settings(), "source_io_concurrency", 1)
    started, release, second_started = Event(), Event(), Event()

    def first_work():
        started.set()
        assert release.wait(3)
        if worker_fails:
            raise RuntimeError("unit-test worker failed during cancellation")
        return 1

    first = asyncio.create_task(run_bounded_source_io(first_work))
    second = None
    try:
        await wait_for_thread(started)
        first.cancel()
        await asyncio.sleep(0.01)
        second = asyncio.create_task(run_bounded_source_io(second_started.set))
        await asyncio.sleep(0.01)
        first.cancel()
        await asyncio.sleep(0.01)
        assert not first.done() and not second_started.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await first
        if worker_fails:
            assert isinstance(cancelled.value.__cause__, RuntimeError)
        await second
        assert second_started.is_set()
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_cooperative_cancel_prevents_later_side_effect_and_does_not_leak_scope():
    from app.services.storage import raise_if_source_io_cancelled, run_bounded_source_io
    started, release = Event(), Event()
    effects = []

    def work():
        started.set()
        assert release.wait(3)
        raise_if_source_io_cancelled()
        effects.append("committed")

    task = asyncio.create_task(run_bounded_source_io(work))
    try:
        await wait_for_thread(started)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert effects == []
        await run_bounded_source_io(raise_if_source_io_cancelled)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_trace_from_worker_is_delivered_on_subscriber_loop(monkeypatch):
    from app.services import agent_graph as ag
    queue = ag._subscribe_trace("unit-test-thread-trace")
    owner_thread = get_ident()
    original = queue.put_nowait

    def checked_put(payload):
        assert get_ident() == owner_thread
        original(payload)

    monkeypatch.setattr(queue, "put_nowait", checked_put)
    try:
        payload = {"id": "unit-test-event", "sequence_index": 1}
        await asyncio.to_thread(ag._publish_trace_event, "unit-test-thread-trace", payload)
        assert await asyncio.wait_for(queue.get(), 1) == payload
    finally:
        ag._unsubscribe_trace("unit-test-thread-trace", queue)
