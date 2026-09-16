"""Measure three production RQ builds using current vectors and rollback writes."""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timezone

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report
from _rq_acceptance import REFERENCE_PROTOCOL, REFERENCE_RUNS, implementation_hash, numeric_config, rq_seconds


def run_reference(kb_id):
    from sqlalchemy import event, func, select, text
    from app.models import AutoTpeRun, Chunk, IngestionBatch, KnowledgeBase, RQPrefix, RQPrefixMembership
    from app.services import context_graph as graph
    from app.services.build_performance import BuildPerformance, _CURRENT
    from app.services.ingestion import TERMINAL_STATES
    from app.services.resource_guard import release_unused_memory

    performance = BuildPerformance(cold=False, deadline_seconds=1800)
    token = _CURRENT.set(performance)
    sampler = threading.Thread(target=performance.sampler, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        with session_scope() as db:
            def reject_commit(session):
                raise RuntimeError("RQ reference transaction must roll back")
            event.listen(db, "before_commit", reject_commit)
            try:
                db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
                db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == kb_id).with_for_update())
                active = db.scalar(select(func.count(IngestionBatch.id)).where(
                    IngestionBatch.knowledge_base_id == kb_id, IngestionBatch.status.not_in(TERMINAL_STATES)))
                if active:
                    raise RuntimeError("RQ reference requires an idle knowledge base")
                selected = db.scalars(select(AutoTpeRun).where(
                    AutoTpeRun.knowledge_base_id == kb_id).order_by(AutoTpeRun.created_at.desc())).first()
                if selected is None or not selected.selected_theta_json:
                    raise RuntimeError("RQ reference requires an audited selected operating point")
                chunks = list(db.scalars(select(Chunk).where(
                    Chunk.knowledge_base_id == kb_id, Chunk.state == "active").order_by(Chunk.id)))
                # Use the same shared RQ input validation/training/encoding
                # entry point as TPE, inside the production workspace and
                # before the complete materialization timer starts. This
                # wrapper is local to this diagnostic process; all candidate
                # generation and graph writes still use the original service.
                original_add = graph.add_relation_edges
                def add_and_prepare(session, state, scope, vectors, edges):
                    original_add(session, state, scope, vectors, edges)
                    from app.services.auto_tpe import _candidate_rq_prefix_inputs
                    from app.services.graph_state_hashes import chunk_business_references
                    references = chunk_business_references(session, scope)
                    _candidate_rq_prefix_inputs(scope, vectors, chunk_business_keys=references.key_by_id,
                        canonical_business_keys_are_production=True)
                graph.add_relation_edges = add_and_prepare
                try:
                    state = graph.build_chunk_relation_graph(db, kb_id, chunks,
                        operating_point=selected.selected_theta_json, emit_heartbeats=False)
                finally:
                    graph.add_relation_edges = original_add
                state_id = state.id
                # The service already verifies support, prefix pairs and canonical
                # persisted facts. Independently verify exactly three primary levels.
                memberships = db.execute(select(RQPrefixMembership.chunk_id, RQPrefix.rq_level)
                    .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
                    .where(RQPrefix.graph_state_id == state_id)).all()
                primary = {}
                for chunk_id, level in memberships:
                    primary.setdefault(chunk_id, []).append(level)
                if set(primary) != {chunk.id for chunk in chunks} or any(sorted(levels) != [1, 2, 3] for levels in primary.values()):
                    raise RuntimeError("RQ reference primary chain is incomplete")
                workspace = state.diagnostics_json["numeric_workspace"]
                identity = (state.scope_hash, state.runtime_settings_hash,
                            state.graph_operating_point_hash, state.state_hash)
                result = {"quality_pass": True, "scope": {"n": len(chunks), "d": workspace["d"], "L": 3},
                          "counts": workspace["counts"], "E": state.stats_json["edge_count"],
                          "P": state.stats_json["rq_prefix_count"]}
            finally:
                db.rollback()
                event.remove(db, "before_commit", reject_commit)
            # Verify the newly constructed graph did not survive the rollback.
            if db.get(graph.ChunkRelationGraphState, state_id) is not None:
                raise RuntimeError("RQ reference graph survived transaction rollback")
            result["transaction_rolled_back"] = True
    finally:
        performance.stop.set()
        sampler.join(timeout=3)
        performance.sample()
        _CURRENT.reset(token)
    summary = performance.summary()
    result.update(rq_seconds=rq_seconds(summary), relation_seconds=time.perf_counter()-started,
                  peak_rss_bytes=summary["peak_rss_bytes"], stages=summary["stages"],
                  shared_preparation_observed=summary["stages"].get("rq_shared_prepare", {}).get("success_count") == 1)
    del state, chunks, selected, memberships, primary
    release_unused_memory()
    return result, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--knowledge-base-id")
    target.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from sqlalchemy import func, select, text
    from app.core.config import get_settings
    from app.models import Chunk
    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise RuntimeError("RQ reference requires disabled fallbacks")
    with session_scope() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        kb = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        kb_id = kb.id
        count = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb_id, Chunk.state == "active"))
        print(json.dumps({"plan": {"knowledge_base": kb.name, "chunks": count, "runs": REFERENCE_RUNS,
              "cold": False, "model_calls": 0, "execute": args.execute,
              "impact": "production relation/RQ builds; all SQL writes rolled back" if args.execute else "read-only plan"}}), flush=True)
    if not args.execute:
        return
    code_hash, config = implementation_hash(), numeric_config()
    runs, identity, scope = [], None, None
    for repetition in range(REFERENCE_RUNS):
        if implementation_hash() != code_hash or numeric_config() != config:
            raise RuntimeError("RQ reference implementation or configuration changed")
        result, observed = run_reference(kb_id)
        if identity is not None and observed != identity:
            raise RuntimeError("RQ reference input, parameters or canonical result changed")
        identity, scope = observed, result.pop("scope")
        runs.append(result)
        print(json.dumps({"reference_run": repetition+1, **{key:result[key] for key in (
            "rq_seconds", "relation_seconds", "quality_pass", "transaction_rolled_back", "E", "P")}}), flush=True)
    if implementation_hash() != code_hash or numeric_config() != config:
        raise RuntimeError("RQ reference implementation or configuration changed")
    report = {"protocol": REFERENCE_PROTOCOL, "timing_protocol": "build_performance_v2",
              "created_at": datetime.now(timezone.utc).isoformat(), "implementation_hash": code_hash,
              "numeric_config": config, "scope": scope, "cold": False, "model_calls": 0,
              "runs": runs, "rq_limit_seconds": max(60.0, min(run["rq_seconds"] for run in runs))}
    path = write_report("rq_complete_reference", report)
    print(json.dumps({"report": str(path), "rq_limit_seconds": report["rq_limit_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
