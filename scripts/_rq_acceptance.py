"""Freeze an independently measured complete-RQ limit before cold admission."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from _context_graph_maintenance import REPO_ROOT

REFERENCE_PROTOCOL = "rq_best_complete_time_v2"
REFERENCE_RUNS = 3
IMPLEMENTATION_FILES = (
    "apps/api/app/services/context_graph.py",
    "apps/api/app/services/auto_tpe.py",
    "apps/api/app/services/graph_build_workspace.py",
    "apps/api/app/services/graph_state_hashes.py",
    "apps/api/app/services/rq_numeric_storage.py",
    "apps/api/app/services/relation_signal_storage.py",
    "apps/api/app/services/build_performance.py",
    "apps/api/app/core/json_codec.py",
    "apps/api/app/db.py",
)


def implementation_hash():
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def numeric_config():
    from app.core.config import get_settings
    from app.services.graph_build_workspace import NUMERIC_PROTOCOL
    settings = get_settings()
    return {"numeric_protocol": NUMERIC_PROTOCOL,
            "numeric_threads": settings.graph_compute_threads,
            "memory_budget_bytes": settings.graph_compute_memory_mb * 1024**2}


def rq_seconds(performance):
    stages = performance.get("stages", {})
    return sum(stages.get(name, {}).get("total_ms", 0) for name in ("rq_complete", "rq_shared_prepare")) / 1000


def freeze_reference(report, *, code_hash, config, n, d, admitted_at=None):
    admitted_at = admitted_at or datetime.now(timezone.utc)
    if report.get("protocol") != REFERENCE_PROTOCOL or report.get("timing_protocol") != "build_performance_v2":
        raise ValueError("RQ reference requires the complete timing protocol")
    created = datetime.fromisoformat(report["created_at"])
    if created.tzinfo is None or not created < admitted_at:
        raise ValueError("RQ reference must precede acceptance admission")
    if report.get("implementation_hash") != code_hash or report.get("numeric_config") != config:
        raise ValueError("RQ reference implementation or numeric configuration changed")
    if report.get("scope") != {"n": n, "d": d, "L": 3} or n < 1 or d < 1:
        raise ValueError("RQ reference scale changed")
    runs = report.get("runs", [])
    if len(runs) != REFERENCE_RUNS:
        raise ValueError("RQ reference requires exactly three complete runs")
    samples = []
    for run in runs:
        value = run.get("rq_seconds")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("RQ reference elapsed time must be positive and finite")
        if run.get("quality_pass") is not True or run.get("transaction_rolled_back") is not True:
            raise ValueError("RQ reference quality or rollback validation failed")
        if run.get("shared_preparation_observed") is not True:
            raise ValueError("RQ reference must execute the production shared preparation path")
        if any(run.get("counts", {}).get(name) != 1 for name in (
            "rq_trainings", "rq_encodings", "similarity_preparations", "support_edge_sort_passes",
        )):
            raise ValueError("RQ reference must include fresh complete preparations")
        samples.append(float(value))
    limit = max(60.0, min(samples))
    if report.get("rq_limit_seconds") != limit:
        raise ValueError("RQ reference limit does not match max(60, min(samples))")
    return {"protocol": REFERENCE_PROTOCOL, "rq_limit_seconds": limit,
            "reference_seconds": samples, "reference_sample_count": len(samples),
            "reference_created_at": report["created_at"],
            "reference_workloads": [{"E": run.get("E"), "P": run.get("P")} for run in runs],
            "frozen_at": admitted_at.isoformat(), "implementation_hash": code_hash,
            "numeric_config": config, "scope": report["scope"]}


def read_reference(filename, *, output_root, n, d):
    path = (output_root / filename).resolve()
    if not path.is_relative_to(output_root.resolve()) or path.suffix != ".json":
        raise ValueError("RQ reference must be a JSON report under output")
    if path.stat().st_size > 1024**2:
        raise ValueError("RQ reference exceeds its bounded report size")
    return freeze_reference(json.loads(path.read_text(encoding="utf-8")),
                            code_hash=implementation_hash(), config=numeric_config(), n=n, d=d)
