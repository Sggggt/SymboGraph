"""Explicit cold-build acceptance through the production Celery ingestion path."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from _context_graph_maintenance import resolve_knowledge_base, session_scope, storage_files, write_report
from _rq_acceptance import implementation_hash, numeric_config, read_reference, rq_seconds as complete_rq_seconds

RESOURCE_CLOCK_SKEW_SECONDS = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--knowledge-base-id")
    target.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--full-reparse", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    parser.add_argument("--rq-reference-report", help="Three complete RQ reference runs, as a JSON filename under output/.")
    parser.add_argument("--resource-report", default="build-resources.json", help="Current monitor report filename under output/.")
    return parser.parse_args()


def validate_execution(args):
    if args.deadline_seconds != 1800:
        raise ValueError("Acceptance deadline must remain 1800 seconds")
    if args.execute and not (args.cold and args.full_reparse):
        raise ValueError("Execution requires --cold and --full-reparse")
    if args.execute and not getattr(args, "rq_reference_report", None):
        raise ValueError("Execution requires an independently measured --rq-reference-report")


def latency_distribution(values):
    ordered = sorted(float(value) for value in values)
    return {"sample_count":len(ordered), "method":"nearest_rank", **{
        name: ordered[max(0,math.ceil(len(ordered)*quantile)-1)] if ordered else None
        for name,quantile in (("p50_ms",.5),("p95_ms",.95),("p99_ms",.99))}}


def resource_coverage_checks(resources, started_at, finished_at):
    first = datetime.fromisoformat(resources["first_sample_utc"])
    last = datetime.fromisoformat(resources["last_sample_utc"])
    gap = resources.get("largest_sample_gap_seconds")
    return {
        "resource_samples_cover_acceptance": first <= started_at and -RESOURCE_CLOCK_SKEW_SECONDS <= (finished_at-last).total_seconds() <= 15,
        "resource_sampling_cadence": type(gap) in (int, float) and math.isfinite(gap) and 0 <= gap <= 2,
    }


async def main(args=None):
    args = parse_args() if args is None else args
    validate_execution(args)
    from sqlalchemy import func, select, text
    from app.core.config import get_settings
    from app.models import Chunk, IngestionBatch, AutoTpeRun, AutoTpeTrial, ChunkRelationGraphState
    from app.services.ingestion import create_uploaded_files_batch, request_batch_cancel_control, TERMINAL_STATES
    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise RuntimeError("Cold acceptance requires both fallbacks disabled")
    if not settings.enable_auto_tpe or settings.tpe_trial_budget != 6:
        raise RuntimeError("Cold acceptance requires automatic TPE with six trials")
    if settings.ingestion_execution_mode != "celery":
        raise RuntimeError("Cold acceptance must run through the production worker")
    with session_scope() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        kb = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        kb_id, kb_name, root = kb.id, kb.name, kb.source_root
        files = storage_files(root)
        active = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb_id, Chunk.state == "active")) or 0
        old_version = int(kb.current_chunk_version or 0)
        plan = {"knowledge_base": kb_name, "file_count": len(files), "active_chunks_before": active,
                "version_before": old_version, "target_version": old_version+1 if active else 1,
                "cold": bool(args.cold), "deadline_seconds": args.deadline_seconds,
                "impact": "reparse all sources, generate vectors and all four graph layers" if args.execute else "read-only plan"}
    output_root = Path(__file__).resolve().parents[1]/"output"
    rq_reference = read_reference(args.rq_reference_report, output_root=output_root,
        n=active, d=settings.embedding_dimensions) if getattr(args, "rq_reference_report", None) else None
    plan["rq_limit_seconds"] = rq_reference["rq_limit_seconds"] if rq_reference else None
    plan["rq_limit_formula"] = "max(60 seconds, min(three independent complete RQ reference times))"
    print(json.dumps({"plan": plan}, ensure_ascii=False), flush=True)
    if not args.execute:
        return plan
    if not files:
        raise RuntimeError("No source files to accept")
    worker_root = Path(__file__).resolve().parents[1] / "apps" / "worker"
    if str(worker_root) not in sys.path:
        sys.path.insert(0, str(worker_root))
    import redis
    from worker_app.celery_app import celery_app
    queues = celery_app.control.inspect(timeout=3).active_queues() or {}
    if not any(queue.get("name") == settings.ingestion_task_queue for rows in queues.values() for queue in rows):
        raise RuntimeError("No worker consumes the configured ingestion queue")
    queue_depth = redis.Redis.from_url(settings.redis_url).llen(settings.ingestion_task_queue)
    if queue_depth > 3:
        raise RuntimeError(f"Maintenance queue is not ready ({queue_depth} pending); diagnose/coalesce before admission")
    resource_path = (output_root/args.resource_report).resolve()
    if not resource_path.is_relative_to(output_root.resolve()):
        raise ValueError("Resource report must remain under output")
    initial_resources = json.loads(resource_path.read_text(encoding="utf-8"))
    monitor_age = (datetime.now(timezone.utc)-datetime.fromisoformat(initial_resources["last_sample_utc"])).total_seconds()
    if not -RESOURCE_CLOCK_SKEW_SECONDS <= monitor_age <= 15 or initial_resources.get("errors"):
        raise RuntimeError("Start a healthy resource monitor before cold acceptance")
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    with session_scope() as db:
        batch = create_uploaded_files_batch(db, kb_id, files, force=True, full_reparse=bool(active))
        batch.stats = {**dict(batch.stats or {}), "benchmark_control": {"protocol": "cold_build_acceptance_v2", "cold": True,
            "deadline_seconds": args.deadline_seconds, "rq_acceptance": rq_reference}}
        db.commit()
        batch_id = batch.id
    from app.routers.ingestion import enqueue_uploaded_batch
    await enqueue_uploaded_batch(batch_id, [str(path) for path in files], force=True, full_reparse=bool(active))
    print(json.dumps({"batch_id": batch_id, "state": "enqueued"}), flush=True)
    last_card = None
    cancel_sent = False
    terminal_observed_at = None
    while True:
        await asyncio.sleep(2)
        with session_scope() as db:
            batch = db.get(IngestionBatch, batch_id)
            stats = dict(batch.stats or {})
            card = {"status": batch.status, "files": batch.processed_files, "success": batch.success_count,
                    "failure": batch.failure_count, "phase": stats.get("context_graph_phase") or stats.get("phase")}
            if card != last_card:
                print(json.dumps({"progress": card, "elapsed_seconds": round(time.perf_counter()-started, 2)}, ensure_ascii=False), flush=True)
                last_card = card
            if time.perf_counter()-started > args.deadline_seconds and not cancel_sent:
                request_batch_cancel_control(db, batch_id, kb_id)
                cancel_sent = True
            if batch.status in TERMINAL_STATES:
                terminal_observed_at = terminal_observed_at or time.perf_counter()
                if stats.get("performance") or time.perf_counter()-terminal_observed_at >= 10:
                    break
            if time.perf_counter()-started > args.deadline_seconds+300:
                raise RuntimeError("Acceptance deadline exceeded; compensation still pending")
    quality = {"pass": False, "checks": {}}
    if card["status"] == "completed":
        from diagnose_context_graph import main as diagnose
        quality = diagnose(SimpleNamespace(knowledge_base_id=kb_id, knowledge_base_name=None), emit_report=False)
    with session_scope() as db:
        batch = db.get(IngestionBatch, batch_id)
        final_performance = (batch.stats or {}).get("performance")
        perf = dict(final_performance or (batch.stats or {}).get("performance_live") or {})
        run = db.scalars(select(AutoTpeRun).where(AutoTpeRun.batch_id == batch_id).order_by(AutoTpeRun.created_at.desc()).limit(1)).first()
        relation = db.get(ChunkRelationGraphState, run.chunk_relation_graph_state_id) if run and run.chunk_relation_graph_state_id else None
        workspace = ((relation.diagnostics_json or {}).get("numeric_workspace", {}) if relation else {}) or perf.get("numeric_workspace",{})
        trials = list(db.scalars(select(AutoTpeTrial).where(AutoTpeTrial.run_id == run.id).order_by(AutoTpeTrial.trial_index)).all()) if run else []
        trial_wall = [(trial.finished_at-trial.started_at).total_seconds()*1000 for trial in trials if trial.finished_at and trial.started_at]
        stages = perf.get("stages", {})
        tpe_seconds = stages.get("tpe_trials", {}).get("total_ms", 0)/1000 + workspace.get("prepare_seconds", 0)
        rq_seconds = complete_rq_seconds(perf)
        elapsed = time.perf_counter()-started
        active_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb_id, Chunk.state == "active")) or 0
        target_version = (batch.stats or {}).get("target_version")
        chunks = (db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb_id, Chunk.state == "active", Chunk.chunk_version == target_version)) or 0) if target_version else 0
        edges = (relation.stats_json or {}).get("edge_count", 0) if relation else 0
        gates = {
            "all_files_successful": batch.success_count == len(files) and batch.failure_count == 0,
            "batch_completed": batch.status == "completed", "cold_observed": perf.get("cold") is True,
            "final_performance_available": bool(final_performance),
            "complete_timing_protocol": perf.get("protocol_version") == "build_performance_v2",
            "all_files_reparsed": stages.get("file_parse", {}).get("success_count") == len(files),
            "vectors_regenerated": perf.get("embedding_vector_count", 0) >= chunks > 0,
            "numeric_block_latency_observed": all(stages.get(phase, {}).get("sample_count", 0) > 0
                and stages[phase].get("failure_count") == 0 for phase in ("similarity_block", "rq_distance_block")),
            "deadline": elapsed <= args.deadline_seconds, "tpe_completed": bool(run and run.status == "completed"),
            "tpe_under_60_seconds": bool(tpe_seconds and tpe_seconds <= 60),
            "rq_within_frozen_limit": bool(rq_seconds and rq_seconds <= rq_reference["rq_limit_seconds"]),
            "rq_reference_scope": rq_reference["scope"] == {"n": chunks, "d": workspace.get("d"), "L": 3},
            "rq_reference_configuration": rq_reference["numeric_config"] == numeric_config(),
            "rq_reference_implementation": rq_reference["implementation_hash"] == implementation_hash(),
            "rq_reference_unchanged": (batch.stats or {}).get("benchmark_control", {}).get("rq_acceptance") == rq_reference,
            "graph_quality": bool(quality.get("pass")),
            "freshness": bool((quality.get("stats", {}).get("freshness") or {}).get("is_admissible")),
            "worker_rss": 0 < perf.get("peak_rss_bytes", 0) <= 3*1024**3,
            "six_trials_observed": len(trials) == 6,
            "trial_nomination_counts_complete": len(trials) == 6 and all(
                type((trial.diagnostics_json or {}).get("nomination_count")) is int
                and trial.diagnostics_json["nomination_count"] >= trial.diagnostics_json.get("candidate_count", 0)
                for trial in trials),
            "similarity_prepared_once": workspace.get("counts",{}).get("similarity_preparations") == 1,
            "selected_candidate_reused": workspace.get("counts",{}).get("selected_candidate_reuses") == 1,
            "support_edges_sorted_once": workspace.get("counts",{}).get("support_edge_sort_passes") == 1,
        }
        resources = json.loads(resource_path.read_text(encoding="utf-8"))
        resource_age = (datetime.now(timezone.utc)-datetime.fromisoformat(resources["last_sample_utc"])).total_seconds()
        gates["stack_memory"] = 0 < resources.get("peak_stack_memory_bytes",0) <= 6*1024**3
        gates["worker_container_memory"] = 0 < resources.get("peak_service_memory_bytes",{}).get("course-kg-worker",0) <= 3*1024**3
        gates["resource_samples_current"] = -RESOURCE_CLOCK_SKEW_SECONDS <= resource_age <= 15 and not resources.get("errors")
        gates.update(resource_coverage_checks(resources, started_at, datetime.now(timezone.utc)))
        report = {"protocol": "cold_build_acceptance_v2", "pass": all(gates.values()), "gates": gates,
                  "resource_clock_policy": {"allowed_skew_seconds": RESOURCE_CLOCK_SKEW_SECONDS, "snapshot_age_seconds": resource_age},
                  "rq_acceptance": rq_reference,
                  "rate_definitions": {"request_qps":"successful transport invocations divided by first-start to last-finish observation window; includes intervening pipeline gaps", "construction_throughput":"items produced by this batch divided by full acceptance wall time"},
                  "file_count": len(files), "chunk_count": chunks, "edge_count": edges,
                  "active_chunk_count": active_chunks,
                  "elapsed_seconds": elapsed, "tpe_seconds": tpe_seconds, "rq_seconds": rq_seconds,
                  "files_per_second": batch.success_count/elapsed, "chunks_per_second": chunks/elapsed,
                  "provider_cache_disclosure":{"concept_known_responses":perf.get("provider_cache_observations",0),
                      "concept_hits":perf.get("provider_cache_hits",0),"concept_unknown_responses":perf.get("provider_cache_unknown_responses",0),
                      "embedding":"provider-side cache status not observed; all vectors require fresh transport requests"},
                  "algorithm_parameters":{"n":chunks,"d":workspace.get("d"),"T":len(trials),"E":edges,"L":3,"K_max":workspace.get("counts",{}).get("rq_max_centers"),
                      "I_max_per_level":8,"I_observed_total":workspace.get("counts",{}).get("rq_iterations"),"P":workspace.get("counts",{}).get("rq_prefix_count"),
                      "trial_candidate_counts":[(trial.diagnostics_json or {}).get("candidate_count") for trial in trials]},
                  "algorithm_complexity":{"TPE_time":"O(n^2*d+n^2*log(n)+T*(n^2+M*log(M))) plus one shared RQ preparation",
                      "RQ_time":"O(L*I*K*n*d+L*n*log(n)+E*log(E)+sum(P_l^2*(d+log(P_l)))+L*E*log(E)+H*log(F))",
                      "TPE_space":"O(n^2+n*d+M) plus audit payloads; n^2 arrays may be mapped",
                      "RQ_space":"O(n*d+L*n+P*d+P^2+L*E) plus audit payloads and bounded DB/sort buffers",
                      "audit_storage":"O(H); H counts full serialized business facts, F counts fact rows; packing is lossless",
                      "trial_nomination_counts":[(trial.diagnostics_json or {}).get("nomination_count") for trial in trials]},
                  "edges_per_second": edges/elapsed, "performance": perf, "numeric_workspace": workspace,
                  "trial_wall_latency": latency_distribution(trial_wall),
                  "trial_gate_charged_latency": latency_distribution([(trial.diagnostics_json or {}).get("elapsed_ms",0) for trial in trials]),
                  "stack_resources": {key:resources.get(key) for key in ("sample_count","available_cpu_count","peak_stack_memory_bytes","peak_service_memory_bytes","errors",
                      "largest_sample_gap_seconds","stack_cpu_cores","stack_cpu_percent_of_available")},
                  "quality_checks": quality.get("checks", {}), "compensation_pending": card["status"] == "cancel_failed"}
    path = write_report("build_performance", report)
    print(json.dumps({"report": str(path), **report}, ensure_ascii=False), flush=True)
    if not report["pass"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    asyncio.run(main())
