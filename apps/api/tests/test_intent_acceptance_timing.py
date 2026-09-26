"""Acceptance latency must count overlapping provider time only once."""
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_intent_execution.py"
    spec = importlib.util.spec_from_file_location("unit_test_intent_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_nonmodel_wall_excludes_provider_interval_union():
    response = {"model_audit": {"qa_performance": {
        "elapsed_ms": 100,
        "spans": [
            {"stage": "provider_roundtrip", "start_ms": 10, "duration_ms": 30},
            {"stage": "provider_roundtrip", "start_ms": 30, "duration_ms": 40},
            {"stage": "resource_read", "start_ms": 3, "duration_ms": 4},
        ],
        "first_response_ms": 75,
        "first_token_ms": 80,
        "stages": {
            "resource_read": {
                "count": 1,
                "success_count": 1,
                "error_count": 0,
                "cancelled_count": 0,
                "total_ms": 4,
                "active_wall_ms": 4,
                "exclusive_ms": 4,
                "p50_ms": 4,
                "p95_ms": 4,
                "p99_ms": 4,
            }
        },
    }}}
    assert _module()._timing_summary(response) == {
        "elapsed_ms": 100.0,
        "provider_roundtrip_ms": 60.0,
        "nonmodel_ms": 40.0,
        "first_response_ms": 75,
        "first_token_ms": 80,
        "stages": {
            "resource_read": {
                "count": 1,
                "success_count": 1,
                "error_count": 0,
                "cancelled_count": 0,
                "total_ms": 4,
                "active_wall_ms": 4,
                "exclusive_ms": 4,
                "p50_ms": 4,
                "p95_ms": 4,
            }
        },
    }
