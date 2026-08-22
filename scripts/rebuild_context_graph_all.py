from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import active_chunk_count, prepare_runtime_for_model_io, resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the active four-layer context graph for one knowledge base.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--stage",
        choices=("all", "contextual-index", "graph-only"),
        default="all",
        help=(
            "all repairs contextual indexes and rebuilds the graph in one caller transaction; "
            "contextual-index commits only the PostgreSQL/Qdrant contextual index; graph-only "
            "requires that index to be current and performs no embedding/Qdrant repair"
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Write new chunk relation, RQ membership, mid, coarse, and context graph states. Omit for dry-run.")
    return parser.parse_args()


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    from app.services.context_graph import (
        CONTEXTUAL_INDEX_REPAIR_MODE_VERIFY_ONLY,
        active_chunks_query,
        context_graph_stats,
        concept_provider_output_failure_card,
        ensure_contextual_indexes_current,
        rebuild_context_graph,
    )
    from app.services.error_sanitizer import external_failure_classification
    from app.services.ingestion_resource_lock import (
        knowledge_base_ingestion_resource_lock,
    )

    stage = str(getattr(args, "stage", "all") or "all")
    model_io_runtime = (
        prepare_runtime_for_model_io() if args.execute else None
    )
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        chunks = active_chunk_count(db, knowledge_base.id)
        payload = {
            "script": "rebuild_context_graph_all",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "active_chunks": chunks,
            "stage": stage,
            "model_io_runtime": model_io_runtime,
            "execute": args.execute,
            "impact": (
                {
                    "all": "repair contextual indexes if stale and write new active four-layer derived graph states",
                    "contextual-index": "repair and commit only contextual index PostgreSQL/Qdrant state; no concept or graph provider calls",
                    "graph-only": "write new active four-layer derived graph states after read-only contextual-index verification; no embedding or Qdrant repair",
                }[stage]
                if args.execute
                else "no writes and no model/provider calls"
            ),
        }
        try:
            if args.execute:
                if stage == "contextual-index":
                    active_chunks = list(
                        db.scalars(active_chunks_query(knowledge_base.id)).all()
                    )
                    if not active_chunks:
                        raise RuntimeError(
                            "Cannot rebuild contextual indexes without active chunks"
                        )
                    async with knowledge_base_ingestion_resource_lock(
                        db,
                        knowledge_base.id,
                        operation="context_graph_rebuild",
                    ) as resource_lock:
                        maintenance = await ensure_contextual_indexes_current(
                            db,
                            knowledge_base=knowledge_base,
                            chunks=active_chunks,
                        )
                        maintenance = {
                            **maintenance,
                            "ingestion_resource_lock": resource_lock.diagnostics(),
                        }
                        db.commit()
                    payload["contextual_index_maintenance"] = maintenance
                else:
                    rebuild_kwargs = (
                        {
                            "contextual_index_repair_mode": (
                                CONTEXTUAL_INDEX_REPAIR_MODE_VERIFY_ONLY
                            )
                        }
                        if stage == "graph-only"
                        else {}
                    )
                    state = await rebuild_context_graph(
                        db,
                        knowledge_base.id,
                        **rebuild_kwargs,
                    )
                    db.commit()
                    payload.update(
                        {
                            "context_graph_state_id": state.id,
                            "context_graph_hash": state.context_graph_hash,
                            "stats": state.stats_json or {},
                        }
                    )
                payload["status"] = "passed"
                payload["transaction"] = {
                    "postgresql_committed": True,
                    "rolled_back": False,
                }
            else:
                payload["current_stats"] = context_graph_stats(
                    db, knowledge_base.id
                )
                payload["status"] = "dry_run"
                payload["transaction"] = {
                    "postgresql_committed": False,
                    "rolled_back": False,
                }
        except Exception as exc:
            db.rollback()
            payload.update(
                {
                    "status": "failed",
                    "transaction": {
                        "postgresql_committed": False,
                        "rolled_back": True,
                    },
                    "failure_type": type(exc).__name__[:128],
                    "external_failure": external_failure_classification(exc),
                    "concept_provider_failure": (
                        concept_provider_output_failure_card(exc)
                    ),
                    "provider_response_persisted": False,
                }
            )
            report = write_report("rebuild_context_graph_all", payload)
            print(
                json.dumps(
                    {"output": str(report), **payload},
                    ensure_ascii=False,
                    default=str,
                )
            )
            raise SystemExit(1) from None
        report = write_report("rebuild_context_graph_all", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
