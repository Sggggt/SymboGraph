from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from collections.abc import Sequence

from _context_graph_maintenance import (
    resolve_knowledge_base,
    session_scope,
    write_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile PostgreSQL vector_records against Qdrant points."
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply proposed vector_status changes. Omit for read-only dry-run.",
    )
    parser.add_argument(
        "--include-unexpired-intents",
        action="store_true",
        help=(
            "Also reconcile in-flight-lease intents. Use only for an "
            "explicitly confirmed crashed or stopped writer."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PostgreSQL keyset batch size (hard-capped at 256).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from app.services.maintenance import reconcile_vector_store_sync
    from app.services.qdrant_outbox import (
        qdrant_outbox_reconcile_lock,
        reconcile_qdrant_outbox_sync,
    )

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        with qdrant_outbox_reconcile_lock(db, knowledge_base.id):
            outbox_stats = reconcile_qdrant_outbox_sync(
                db,
                knowledge_base_id=knowledge_base.id,
                knowledge_base_name=knowledge_base.name,
                dry_run=not args.execute,
                include_unexpired=args.include_unexpired_intents,
            )
            vector_stats = reconcile_vector_store_sync(
                db,
                knowledge_base_id=knowledge_base.id,
                dry_run=not args.execute,
                batch_size=args.batch_size,
                lock_already_held=True,
            )
            if args.execute:
                db.commit()

        payload = {
            "script": "reconcile_vector_records",
            "execute": args.execute,
            "mode": "execute" if args.execute else "dry_run",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "targets": {
                "postgres_table": "vector_records",
                "qdrant_access": (
                    "reconcile_read_write"
                    if args.execute
                    else "health_check_read_only"
                ),
                "checked_records": vector_stats.get("checked_records", 0),
                "checked_collections": vector_stats.get("checked_collections", 0),
                "checked_outbox_intents": outbox_stats.get("checked_intents", 0),
                "checked_delete_intents": outbox_stats.get(
                    "checked_delete_intents", 0
                ),
                "blocking_delete_intents": outbox_stats.get(
                    "blocking_delete_intents", 0
                ),
                "record_batches": vector_stats.get("processed_record_batches", 0),
                "max_record_batch_size": vector_stats.get(
                    "max_record_batch_size", 0
                ),
            },
            "delete_recovery_actions": outbox_stats.get(
                "delete_recovery_actions", []
            ),
            "transaction": {
                "owner": "reconcile_vector_records.py",
                "committed": bool(args.execute),
                "commit_scope": (
                    "outbox_and_vector_records_for_one_knowledge_base"
                ),
                "durable_job_cursor": False,
                "restart_protocol": vector_stats.get("restart_protocol"),
            },
            "impact": (
                "apply durable Qdrant outbox compensation and PostgreSQL "
                "vector_status reconciliation"
                if args.execute
                else "no PostgreSQL, Qdrant, or Redis writes"
            ),
            "stats": {
                "qdrant_outbox": outbox_stats,
                "vector_records": vector_stats,
            },
        }
        report = write_report("reconcile_vector_records", payload)
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
