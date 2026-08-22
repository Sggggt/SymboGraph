from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import session_scope, write_report


ACTIVE_RECOVERY_STATUSES = {
    "prepared",
    "parsing",
    "parse_compensation_pending",
    "parse_compensating",
    "graph_building",
    "graph_compensation_pending",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose durable ingestion batch recovery rows.  The default is read-only; "
            "--execute requires --all-active-batches because startup reconciliation is global."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--batch-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--all-active-batches",
        action="store_true",
        help=(
            "Explicitly authorize recovery of every released non-terminal batch, "
            "terminal batch with durable compensation pending, and pending cache dispatch."
        ),
    )
    return parser.parse_args()


def snapshot(*, knowledge_base_id: str | None, batch_id: str | None) -> dict:
    from sqlalchemy import select

    from app.models import IngestionBatch, IngestionBatchRecovery, IngestionFileStage
    from app.services.ingestion import _canonical_payload_hash

    with session_scope() as db:
        query = select(IngestionBatchRecovery).order_by(
            IngestionBatchRecovery.created_at.asc(),
            IngestionBatchRecovery.id.asc(),
        )
        if knowledge_base_id:
            query = query.where(
                IngestionBatchRecovery.knowledge_base_id == knowledge_base_id
            )
        if batch_id:
            query = query.where(IngestionBatchRecovery.batch_id == batch_id)
        rows = list(db.scalars(query).all())
        cards = []
        for row in rows:
            batch = db.get(IngestionBatch, row.batch_id)
            stages = list(
                db.scalars(
                    select(IngestionFileStage)
                    .where(IngestionFileStage.batch_recovery_id == row.id)
                    .order_by(IngestionFileStage.sequence_index.asc())
                ).all()
            )
            cache_state = dict(
                (row.compensation_json or {}).get("cache_invalidation") or {}
            )
            cards.append(
                {
                    "recovery_id": row.id,
                    "batch_id": row.batch_id,
                    "knowledge_base_id": row.knowledge_base_id,
                    "batch_status": batch.status if batch else None,
                    "recovery_status": row.status,
                    "protocol_version": row.protocol_version,
                    "v_before_batch": row.v_before_batch,
                    "target_version": row.target_version,
                    "parse_committed": row.parse_committed,
                    "before_state_hash_valid": (
                        _canonical_payload_hash(dict(row.before_state_json or {}))
                        == row.before_state_hash
                    ),
                    "graph_before_state_hash_valid": (
                        _canonical_payload_hash(dict(row.graph_before_state_json or {}))
                        == row.graph_before_state_hash
                    ),
                    "file_stages": [
                        {
                            "id": stage.id,
                            "source_path": stage.source_path,
                            "sequence_index": stage.sequence_index,
                            "status": stage.status,
                            "phase": stage.phase,
                            "before_state_hash_valid": (
                                _canonical_payload_hash(
                                    dict(stage.before_state_json or {})
                                )
                                == stage.before_state_hash
                            ),
                            "write_set_hash_valid": (
                                not stage.write_set_hash
                                or _canonical_payload_hash(
                                    dict(stage.write_set_json or {})
                                )
                                == stage.write_set_hash
                            ),
                            "qdrant_delete_intent_ids": sorted(
                                str(value)
                                for value in (
                                    stage.compensation_json.get(
                                        "qdrant_delete_intents"
                                    )
                                    or {}
                                ).values()
                            ),
                        }
                        for stage in stages
                    ],
                    "cache_invalidation": cache_state,
                    "requires_recovery": row.status in ACTIVE_RECOVERY_STATUSES,
                    "requires_cache_retry": cache_state.get("status") == "pending",
                }
            )
        return {
            "recovery_count": len(cards),
            "active_recovery_count": sum(
                1 for card in cards if card["requires_recovery"]
            ),
            "pending_cache_count": sum(
                1 for card in cards if card["requires_cache_retry"]
            ),
            "recoveries": cards,
        }


def main() -> None:
    args = parse_args()
    if args.execute and not args.all_active_batches:
        raise SystemExit("--execute requires the explicit --all-active-batches flag")
    if args.execute and (args.knowledge_base_id or args.batch_id):
        raise SystemExit(
            "Global startup recovery cannot honor a partial filter; omit filters or use dry-run"
        )
    before = snapshot(
        knowledge_base_id=args.knowledge_base_id,
        batch_id=args.batch_id,
    )
    if args.execute:
        from app.services.ingestion import finalize_interrupted_batches

        finalize_interrupted_batches()
    after = snapshot(
        knowledge_base_id=args.knowledge_base_id,
        batch_id=args.batch_id,
    )
    payload = {
        "script": "reconcile_ingestion_batch_recoveries",
        "mode": "execute" if args.execute else "dry_run",
        "execute": bool(args.execute),
        "scope": {
            "knowledge_base_id": args.knowledge_base_id,
            "batch_id": args.batch_id,
            "all_active_batches": bool(args.all_active_batches),
        },
        "targets": before,
        "after": after,
        "impact": (
            "recover every released non-terminal ingestion batch, retry terminal durable "
            "compensation, and retry durable cache invalidations"
            if args.execute
            else "no PostgreSQL, Qdrant, Redis, file-system, or model writes"
        ),
    }
    report = write_report("reconcile_ingestion_batch_recoveries", payload)
    print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
