import asyncio
import json

import pytest
from pydantic import ValidationError

from app.services.qa_performance import QAPerformance, QAAuditCapacityError, qa_stage


def test_parallel_children_use_interval_union_instead_of_double_subtraction():
    instant = [0.0]
    recorder = QAPerformance(clock=lambda: instant[0])
    parent = recorder.begin("request")
    instant[0] = .010
    first = recorder.begin("provider_roundtrip", parent=parent.sequence, role="chat")
    instant[0] = .030
    second = recorder.begin("provider_roundtrip", parent=parent.sequence, role="embedding")
    instant[0] = .060
    recorder.finish(first)
    instant[0] = .080
    recorder.finish(second)
    instant[0] = .100
    recorder.finish(parent)
    summary = recorder.snapshot()
    assert summary.elapsed_ms == 100
    assert summary.spans[0].exclusive_ms == 30
    assert summary.stages["provider_roundtrip"].total_ms == 100
    assert summary.stages["provider_roundtrip"].active_wall_ms == 70
    assert summary.unfinished_span_count == 0


def test_failed_stage_keeps_elapsed_and_does_not_capture_exception_message():
    instant = [0.0]
    recorder = QAPerformance(clock=lambda: instant[0])
    with recorder.activate():
        with pytest.raises(RuntimeError):
            with qa_stage("generation", output_token_budget=1024):
                instant[0] = .125
                raise RuntimeError("unit-test-secret-provider-body")
    summary = recorder.snapshot()
    assert summary.stages["generation"].error_count == 1
    assert summary.spans[0].duration_ms == 125
    assert summary.spans[0].error_type == "RuntimeError"
    assert "unit-test-secret-provider-body" not in json.dumps(summary.model_dump(mode="json"))
    assert summary.provider_compute_ms is None and summary.first_token_ms is None


def test_first_stream_response_and_token_are_recorded_once():
    instant = [0.0]
    recorder = QAPerformance(clock=lambda: instant[0])
    instant[0] = .025
    recorder.mark_first_response(token=False)
    instant[0] = .040
    recorder.mark_first_response(token=True)
    instant[0] = .090
    recorder.mark_first_response(token=True)

    summary = recorder.snapshot()
    assert summary.first_response_ms == 25
    assert summary.first_token_ms == 40


def test_audit_scope_capacity_and_metadata_are_closed():
    recorder = QAPerformance(span_limit=1)
    with recorder.activate():
        with pytest.raises(ValidationError):
            with qa_stage("generation", prompt="unit-test-secret"):
                pytest.fail("untrusted metadata must fail before execution")
        with qa_stage("generation"):
            pass
        with pytest.raises(QAAuditCapacityError):
            with qa_stage("generation"):
                pytest.fail("span overflow must not silently omit work")
    assert recorder.snapshot().stages["generation"].count == 1
    with qa_stage("generation"):
        pass
    assert recorder.snapshot().stages["generation"].count == 1


@pytest.mark.asyncio
async def test_model_queue_cancellation_does_not_leak_a_slot(monkeypatch, no_fallback_env):
    from app.core.config import get_settings
    from app.core.concurrency import model_request_slot
    monkeypatch.setattr(get_settings(), "model_request_concurrency", 1)
    recorder = QAPerformance()
    entered = asyncio.Event()
    async def wait_for_slot():
        entered.set()
        async with model_request_slot():
            pytest.fail("occupied slot unexpectedly admitted a second caller")
    with recorder.activate():
        async with model_request_slot():
            waiter = asyncio.create_task(wait_for_slot())
            await entered.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        async with model_request_slot():
            pass
    result = recorder.snapshot()
    assert result.stages["model_queue"].count == 3
    assert result.stages["model_queue"].cancelled_count == 1
    assert result.stages["model_queue"].success_count == 2


@pytest.mark.asyncio
async def test_concurrent_task_scopes_are_isolated():
    async def observe(stage):
        recorder = QAPerformance()
        with recorder.activate():
            with qa_stage(stage):
                await asyncio.sleep(0)
        return recorder.snapshot()
    first, second = await asyncio.gather(observe("generation"), observe("retrieval"))
    assert set(first.stages) == {"generation"}
    assert set(second.stages) == {"retrieval"}


def test_nearest_rank_includes_failed_attempt_latency():
    instant = [0.0]
    recorder = QAPerformance(clock=lambda: instant[0])
    for duration, failed in ((.01, False), (.02, False), (.1, True)):
        span = recorder.begin("provider_roundtrip")
        instant[0] += duration
        recorder.finish(span, TimeoutError() if failed else None)
    summary = recorder.snapshot().stages["provider_roundtrip"]
    assert summary.p50_ms == 20 and summary.p95_ms == summary.p99_ms == 100
    assert summary.success_count == 2 and summary.error_count == 1
