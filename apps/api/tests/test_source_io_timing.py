import asyncio
from threading import Event

import pytest

from app.services.qa_performance import QAPerformance, qa_stage
from app.services.storage import run_bounded_source_io


@pytest.mark.asyncio
async def test_thread_stage_is_nested_and_exclusive_time_is_not_double_counted(no_fallback_env):
    clock = [0.0]
    timing = QAPerformance(clock=lambda: clock[0])
    def work():
        clock[0] = .01
        with qa_stage("graph_traversal"):
            clock[0] = .03
        clock[0] = .04
    with timing.activate():
        await run_bounded_source_io(work)
    result = timing.snapshot()
    parent = next(span for span in result.spans if span.stage == "source_io")
    child = next(span for span in result.spans if span.stage == "graph_traversal")
    assert child.parent_sequence == parent.sequence
    assert parent.duration_ms == pytest.approx(40)
    assert parent.exclusive_ms == pytest.approx(20)
    assert child.duration_ms == pytest.approx(20)


@pytest.mark.asyncio
async def test_cancelled_parent_span_includes_thread_drain(no_fallback_env):
    clock = [0.0]
    timing = QAPerformance(clock=lambda: clock[0])
    started, release = Event(), Event()
    def work():
        with qa_stage("graph_traversal"):
            started.set()
            assert release.wait(3)
            clock[0] = .05
    with timing.activate():
        task = asyncio.create_task(run_bounded_source_io(work))
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(.001)
            clock[0] = .02
            task.cancel()
            await asyncio.sleep(.01)
            parent = next(span for span in timing.snapshot().spans if span.stage == "source_io")
            assert parent.status == "running" and not task.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    parent = next(span for span in timing.snapshot().spans if span.stage == "source_io")
    assert parent.status == "cancelled"
    assert parent.duration_ms == pytest.approx(50)
    assert timing.snapshot().unfinished_span_count == 0
