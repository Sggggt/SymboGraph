from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import (
    prepare_runtime_for_model_io,
    resolve_knowledge_base,
    session_scope,
    write_report,
)


REUSED_LAYERS = [
    "chunk_structure_graph",
    "chunk_relation_graph",
    "rq_membership_graph",
    "mid_concept_graph",
]
REBUILT_LAYERS = [
    "coarse_concept_graph",
    "context_graph_state",
]
WRITTEN_TABLES = [
    "coarse_concept_states",
    "coarse_concepts",
    "coarse_concept_edges",
    "coarse_concept_memberships",
    "context_graph_states",
    "context_graph_freshness",
    "ingestion_compensation_logs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild coarse concepts and the active context state while "
            "reusing the admitted structure, calibrated relation, RQ "
            "membership, and mid-concept layers. Omit --execute for a "
            "read-only report."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Atomically switch only the coarse -> context lineage. Without "
            "this flag the command performs no writes or model calls."
        ),
    )
    return parser.parse_args()


def _rollback_if_supported(db) -> None:
    rollback = getattr(db, "rollback", None)
    if callable(rollback):
        rollback()


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    from app.services import context_graph
    from app.services import scoped_context_graph_rebuild

    model_io_runtime = (
        prepare_runtime_for_model_io() if args.execute else None
    )
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        payload = {
            "script": "rebuild_coarse_concept_graph",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "model_io_runtime": model_io_runtime,
            "execute": args.execute,
            "mode": "execute" if args.execute else "dry_run",
            "scope_semantics": "scoped_dependency_cascade",
            "targets": {
                "requested_scope": "coarse_concept",
                "reused_layers": list(REUSED_LAYERS),
                "rebuilt_layers": list(REBUILT_LAYERS),
                "tables_written_on_execute": list(WRITTEN_TABLES),
                "knowledge_base_id": knowledge_base.id,
            },
            "impact": (
                "reuses structure, calibrated relation, RQ membership, and "
                "mid concepts; rebuilds coarse concepts and the active context "
                "graph state"
                if args.execute
                else "no writes and no model/provider calls"
            ),
        }
        if args.execute:
            try:
                state = (
                    await scoped_context_graph_rebuild.rebuild_coarse_concept_graph(
                        db,
                        knowledge_base.id,
                    )
                )
                db.commit()
            except Exception:
                _rollback_if_supported(db)
                raise
            audit = dict(
                (state.diagnostics_json or {}).get(
                    "scoped_rebuild_audit"
                )
                or {}
            )
            payload.update(
                {
                    "context_graph_state_id": state.id,
                    "stats": state.stats_json or {},
                    "scoped_rebuild_audit": audit,
                    "transaction": {
                        "database_committed": True,
                        "candidate_savepoint": True,
                        "cache_invalidation_queued_after_commit": bool(
                            audit.get(
                                "cache_invalidation_queued_after_commit"
                            )
                        ),
                    },
                }
            )
        else:
            payload["current_stats"] = context_graph.context_graph_stats(
                db,
                knowledge_base.id,
            )
            payload["transaction"] = {
                "database_committed": False,
                "candidate_savepoint": False,
                "cache_invalidation_queued_after_commit": False,
            }
        report = write_report("rebuild_coarse_concept_graph", payload)
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
