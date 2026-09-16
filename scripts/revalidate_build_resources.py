"""Replay a complete resource window without rerunning or altering a build."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _context_graph_maintenance import OUTPUT_ROOT, write_report
from monitor_build_resources import SERVICES, distribution

REPLAYABLE_GATES = {"resource_samples_current", "resource_samples_cover_acceptance"}
REQUIRED_BUILD_GATES = {"all_files_successful", "batch_completed", "cold_observed", "final_performance_available",
    "complete_timing_protocol", "all_files_reparsed", "vectors_regenerated", "deadline", "tpe_completed",
    "tpe_under_60_seconds", "rq_within_frozen_limit", "graph_quality", "freshness", "worker_rss",
    "six_trials_observed", "numeric_block_latency_observed", "trial_nomination_counts_complete",
    "stack_memory", "worker_container_memory", "resource_sampling_cadence"}


def utc(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Resource timestamps require an explicit timezone")
    return result.astimezone(timezone.utc)


def summarize_window(samples, *, start, end, available_cpus):
    selected, previous, first, last = [], None, None, None
    largest_gap = 0.0
    if end <= start or not 1 <= available_cpus <= 4096:
        raise ValueError("Invalid resource audit window")
    for index, sample in enumerate(samples):
        if index >= 100000:
            raise ValueError("Resource sample scope exceeds its bounded allowance")
        stamp = utc(sample["observed_at_utc"])
        elapsed = sample["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed):
            raise ValueError("Invalid collector monotonic time")
        if previous:
            wall_gap = (stamp-previous[0]).total_seconds()
            monotonic_gap = elapsed-previous[1]
            if wall_gap <= 0 or monotonic_gap <= 0 or abs(wall_gap-monotonic_gap) > .1:
                raise ValueError("Resource clocks are discontinuous")
            if stamp >= start and previous[0] <= end:
                largest_gap = max(largest_gap, wall_gap, monotonic_gap)
        first = stamp if first is None else first
        last = stamp
        previous = stamp, elapsed
        if not start <= stamp <= end:
            continue
        rows = sample["services"]
        if len(rows) != len(SERVICES) or {row["service"] for row in rows} != set(SERVICES):
            raise ValueError("Resource frame omits project services")
        if any(type(row["memory_bytes"]) is not int or row["memory_bytes"] < 0
               or not math.isfinite(row["cpu_cores"]) or row["cpu_cores"] < 0 for row in rows):
            raise ValueError("Invalid resource measurements")
        if sample["memory_bytes"] != sum(row["memory_bytes"] for row in rows):
            raise ValueError("Resource stack memory does not match its services")
        selected.append(sample)
    if first is None or first > start or last < end or not selected or largest_gap > 2:
        raise ValueError("Resource samples do not continuously cover the full acceptance window")
    peaks = {name: max(row["memory_bytes"] for sample in selected for row in sample["services"] if row["service"] == name) for name in SERVICES}
    cores = [sum(row["cpu_cores"] for row in sample["services"]) for sample in selected]
    return {"sample_count": len(selected), "available_cpu_count": available_cpus,
            "peak_stack_memory_bytes": max(sample["memory_bytes"] for sample in selected),
            "peak_service_memory_bytes": peaks, "errors": 0, "largest_sample_gap_seconds": largest_gap,
            "stack_cpu_cores": distribution(cores),
            "stack_cpu_percent_of_available": distribution([100*value/available_cpus for value in cores])}


def validate_original(report):
    gates = report.get("gates", {})
    if report.get("protocol") != "cold_build_acceptance_v2" or not (REQUIRED_BUILD_GATES | REPLAYABLE_GATES) <= set(gates):
        raise ValueError("A complete cold acceptance report is required")
    if any(value is not True for key, value in gates.items() if key not in REPLAYABLE_GATES):
        raise ValueError("Non-resource acceptance failure requires a new complete build")


def resolve(name):
    path = (OUTPUT_ROOT/name).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()) or not path.is_file() or path.stat().st_size > 256*1024**2:
        raise ValueError("Resource inputs must be bounded existing artifacts under output")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    original_path, resource_path = resolve(args.report), resolve(args.resources)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    metadata = json.loads(resource_path.read_text(encoding="utf-8"))
    validate_original(original)
    if metadata.get("errors") != 0:
        raise ValueError("Resource collection errors cannot be removed by replay")
    sample_path = resolve(metadata["sample_file"])
    if sample_path != resource_path.with_suffix(".jsonl"):
        raise ValueError("Resource sample file does not belong to its report")
    written_at = datetime.fromtimestamp(original_path.stat().st_mtime, timezone.utc)
    start = min(utc(original["rq_acceptance"]["frozen_at"]), written_at-timedelta(seconds=original["elapsed_seconds"]))-timedelta(seconds=2)
    end = written_at+timedelta(seconds=2)
    plan = {"source_report": original_path.name, "resource_samples": sample_path.name,
            "start_utc": start.isoformat(), "end_utc": end.isoformat(), "clock_padding_seconds": 2,
            "impact": "aggregate resource report only; original build, timings and report are unchanged"}
    print(json.dumps({"plan": plan, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    with sample_path.open(encoding="utf-8") as stream:
        resources = summarize_window((json.loads(line) for line in stream), start=start, end=end,
            available_cpus=metadata["available_cpu_count"])
    gates = {**original["gates"], **{name: True for name in REPLAYABLE_GATES},
             "resource_window_replayed": True,
             "stack_memory": 0 < resources["peak_stack_memory_bytes"] <= 6*1024**3,
             "worker_container_memory": 0 < resources["peak_service_memory_bytes"]["course-kg-worker"] <= 3*1024**3}
    report = {**original, "pass": all(gates.values()), "gates": gates, "stack_resources": resources,
              "resource_revalidation": {"protocol": "resource_window_replay_v1", **plan,
                  "original_failed_gates": [name for name, passed in original["gates"].items() if not passed]}}
    path = write_report("build_performance_revalidated", report)
    print(json.dumps({"report": str(path), "pass": report["pass"], "resource_samples": resources["sample_count"],
                      "elapsed_seconds": report["elapsed_seconds"], "tpe_seconds": report["tpe_seconds"], "rq_seconds": report["rq_seconds"]}), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
