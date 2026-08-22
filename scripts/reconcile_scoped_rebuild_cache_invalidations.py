from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import (
    resolve_knowledge_base,
    session_scope,
    write_report,
)


PENDING_SAMPLE_LIMIT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or retry durable Redis invalidation intents created by "
            "RQ/mid/coarse scoped graph rebuilds. The default is read-only."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Retry at most 100 pending invalidations for the selected "
            "knowledge base and persist each idempotent outcome."
        ),
    )
    return parser.parse_args()


def pending_snapshot(db, knowledge_base_id: str) -> dict:
    from sqlalchemy import select

    from app.models import IngestionCompensationLog
    from app.services.scoped_context_graph_rebuild import (
        SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_OPERATION,
    )

    rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.knowledge_base_id
                == knowledge_base_id,
                IngestionCompensationLog.operation
                == SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_OPERATION,
                IngestionCompensationLog.status
                == "cache_invalidation_pending",
            )
            .order_by(IngestionCompensationLog.created_at.asc())
            .limit(PENDING_SAMPLE_LIMIT + 1)
        ).all()
    )
    sampled = rows[:PENDING_SAMPLE_LIMIT]
    return {
        "pending_count_lower_bound": len(rows),
        "sample_truncated": len(rows) > PENDING_SAMPLE_LIMIT,
        "sample": [
            {
                "intent_id": str(row.id),
                "context_graph_state_ids": list(
                    row.target_ids_json or []
                ),
                "status": str(row.status),
                "error_type": (
                    str(row.error_message or "").rsplit(":", 1)[-1].strip()
                    or None
                ),
                "created_at": (
                    row.created_at.isoformat()
                    if row.created_at is not None
                    else None
                ),
            }
            for row in sampled
        ],
    }


def main() -> None:
    args = parse_args()

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        before = pending_snapshot(db, knowledge_base.id)
        if args.execute:
            from app.services.scoped_context_graph_rebuild import (
                reconcile_scoped_rebuild_cache_invalidations,
            )

            result = reconcile_scoped_rebuild_cache_invalidations(
                db,
                knowledge_base_id=knowledge_base.id,
                limit=PENDING_SAMPLE_LIMIT,
            )
            db.commit()
        else:
            result = {
                "checked": 0,
                "completed": 0,
                "pending": before["pending_count_lower_bound"],
                "results": [],
            }
        after = pending_snapshot(db, knowledge_base.id)
        payload = {
            "script": "reconcile_scoped_rebuild_cache_invalidations",
            "mode": "execute" if args.execute else "dry_run",
            "execute": bool(args.execute),
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "targets": before,
            "result": result,
            "after": after,
            "impact": (
                "retry bounded idempotent Redis invalidations and persist "
                "completed/pending outcomes"
                if args.execute
                else "no PostgreSQL, Redis, Qdrant, file-system, or model writes"
            ),
        }
        report = write_report(
            "reconcile_scoped_rebuild_cache_invalidations",
            payload,
        )
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
