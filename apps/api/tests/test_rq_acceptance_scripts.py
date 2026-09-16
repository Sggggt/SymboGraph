from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from _rq_acceptance import REFERENCE_PROTOCOL, freeze_reference, read_reference, rq_seconds


def reference(samples):
    return {"protocol": REFERENCE_PROTOCOL, "timing_protocol": "build_performance_v2",
            "created_at": "2026-01-01T00:00:00+00:00", "implementation_hash": "unit-test-code",
            "numeric_config": {"numeric_threads": 2}, "scope": {"n": 256, "d": 128, "L": 3},
            "rq_limit_seconds": max(60., min(samples)),
            "runs": [{"rq_seconds": value, "quality_pass": True, "transaction_rolled_back": True, "shared_preparation_observed": True,
                      "counts": {"rq_trainings": 1, "rq_encodings": 1, "similarity_preparations": 1,
                                 "support_edge_sort_passes": 1}} for value in samples]}


def freeze(report, **kwargs):
    return freeze_reference(report, **{"code_hash": "unit-test-code", "config": {"numeric_threads": 2},
        "n": 256, "d": 128, "admitted_at": datetime(2026, 1, 2, tzinfo=timezone.utc), **kwargs})


@pytest.mark.parametrize("samples,expected", [([59., 50., 58.], 60.), ([83., 71.25, 80.], 71.25)])
def test_rq_limit_is_frozen_from_independent_complete_runs(samples, expected):
    report = reference(samples)
    frozen = freeze(report)
    report["runs"][0]["rq_seconds"] = 500
    assert frozen["rq_limit_seconds"] == expected
    assert frozen["reference_seconds"] == samples
    assert frozen["reference_sample_count"] == 3


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True, "80"])
def test_rq_reference_rejects_invalid_durations(invalid):
    report = reference([70., 80., 90.])
    report["runs"][1]["rq_seconds"] = invalid
    with pytest.raises(ValueError, match="positive and finite"):
        freeze(report)


@pytest.mark.parametrize("changes", [
    {"code_hash": "unit-test-new-code"}, {"config": {"numeric_threads": 4}}, {"n": 257}, {"d": 256},
    {"admitted_at": datetime(2025, 12, 31, tzinfo=timezone.utc)},
])
def test_rq_reference_rejects_drift_or_post_admission_samples(changes):
    with pytest.raises(ValueError):
        freeze(reference([70., 80., 90.]), **changes)


@pytest.mark.parametrize("mutation", ["quality", "rollback", "training", "sample_count", "limit", "timing", "preparation"])
def test_rq_reference_cannot_skip_work_or_inflate_limit(mutation):
    report = reference([70., 80., 90.])
    if mutation == "quality": report["runs"][0]["quality_pass"] = False
    if mutation == "rollback": report["runs"][0]["transaction_rolled_back"] = False
    if mutation == "training": report["runs"][0]["counts"]["rq_trainings"] = 0
    if mutation == "sample_count": report["runs"].pop()
    if mutation == "limit": report["rq_limit_seconds"] = 90.
    if mutation == "timing": report["timing_protocol"] = "build_performance_v1"
    if mutation == "preparation": report["runs"][0]["shared_preparation_observed"] = False
    with pytest.raises(ValueError): freeze(report)


def test_rq_complete_time_includes_shared_preparation_without_double_counting_nested_stages():
    assert rq_seconds({"stages": {"rq_complete": {"total_ms": 50000}, "rq_shared_prepare": {"total_ms": 15000},
        "rq_materialization": {"total_ms": 30000}, "rq_training": {"total_ms": 8000}}}) == 65.


def test_rq_reference_report_must_stay_under_output(tmp_path):
    with pytest.raises(ValueError, match="under output"):
        read_reference("../unit-test-reference.json", output_root=tmp_path, n=256, d=128)


@pytest.mark.parametrize("missing", [None, "start", "end", "gap"])
def test_acceptance_rejects_incomplete_resource_observation(missing):
    from benchmark_build_pipeline import resource_coverage_checks
    start = datetime(2026, 1, 2, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)
    resources = {"first_sample_utc": (start-timedelta(seconds=5)).isoformat(),
        "last_sample_utc": (end-timedelta(seconds=3)).isoformat(), "largest_sample_gap_seconds": .6}
    if missing == "start": resources["first_sample_utc"] = (start+timedelta(seconds=3)).isoformat()
    if missing == "end": resources["last_sample_utc"] = (end-timedelta(seconds=40)).isoformat()
    if missing == "gap": resources["largest_sample_gap_seconds"] = 30.
    assert all(resource_coverage_checks(resources, start, end).values()) is (missing is None)


@pytest.mark.parametrize("skew,expected", [(.08, True), (3., False)])
def test_resource_clock_boundary_allows_only_bounded_cross_host_skew(skew, expected):
    from benchmark_build_pipeline import resource_coverage_checks
    start = datetime(2026, 1, 2, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)
    resources = {"first_sample_utc": (start-timedelta(seconds=10)).isoformat(),
        "last_sample_utc": (end+timedelta(seconds=skew)).isoformat(), "largest_sample_gap_seconds": .5}
    assert all(resource_coverage_checks(resources, start, end).values()) is expected


def resource_samples():
    from monitor_build_resources import SERVICES
    epoch = datetime(2026, 1, 2, tzinfo=timezone.utc)
    rows = []
    for second in range(11):
        services = [{"service": name, "memory_bytes": 1024*(second+1), "cpu_cores": .1} for name in SERVICES]
        rows.append({"observed_at_utc": (epoch+timedelta(seconds=second)).isoformat(), "elapsed_seconds": float(second),
                     "memory_bytes": sum(item["memory_bytes"] for item in services), "services": services})
    return epoch, rows


def test_resource_replay_counts_only_the_complete_requested_window():
    from revalidate_build_resources import summarize_window
    epoch, rows = resource_samples()
    result = summarize_window(iter(rows), start=epoch+timedelta(seconds=2), end=epoch+timedelta(seconds=8), available_cpus=16)
    assert result["sample_count"] == 7
    assert result["largest_sample_gap_seconds"] == 1.
    assert result["peak_service_memory_bytes"]["course-kg-worker"] == 9*1024
    assert result["peak_stack_memory_bytes"] == 8*9*1024


@pytest.mark.parametrize("missing", ["head", "tail", "gap", "service", "clock"])
def test_resource_replay_rejects_missing_coverage_and_clock_discontinuity(missing):
    from revalidate_build_resources import summarize_window
    epoch, rows = resource_samples()
    if missing == "head": rows = rows[3:]
    if missing == "tail": rows = rows[:-3]
    if missing == "gap": rows = rows[:4]+rows[7:]
    if missing == "service": rows[4]["services"].pop()
    if missing == "clock": rows[4]["elapsed_seconds"] += .2
    with pytest.raises(ValueError):
        summarize_window(iter(rows), start=epoch+timedelta(seconds=2), end=epoch+timedelta(seconds=8), available_cpus=16)


def test_resource_replay_cannot_override_a_failed_build_or_incomplete_report():
    from revalidate_build_resources import validate_original, REQUIRED_BUILD_GATES, REPLAYABLE_GATES
    report = {"protocol": "cold_build_acceptance_v2", "gates": {name:True for name in REQUIRED_BUILD_GATES | REPLAYABLE_GATES}}
    report["gates"]["resource_samples_current"] = False
    validate_original(report)
    report["gates"]["tpe_under_60_seconds"] = False
    with pytest.raises(ValueError, match="new complete build"):
        validate_original(report)
    del report["gates"]["tpe_under_60_seconds"]
    with pytest.raises(ValueError, match="complete cold acceptance"):
        validate_original(report)
