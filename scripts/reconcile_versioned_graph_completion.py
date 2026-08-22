from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile one already-committed versioned graph completion: clear "
            "stale current-progress fields and dispatch a graph-commit-bound "
            "Redis cache invalidation. The default is read-only."
        )
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the sealed reconcile plan. This writes only PostgreSQL "
            "completion metadata and the strict Redis cache invalidation."
        ),
    )
    return parser.parse_args()


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    from app.services.ingestion import (
        plan_versioned_graph_completion_reconcile,
        reconcile_versioned_graph_completion,
    )

    with session_scope() as db:
        plan = plan_versioned_graph_completion_reconcile(
            db,
            batch_id=str(args.batch_id),
        )
        db.rollback()
    payload: dict[str, object] = {
        "script": "reconcile_versioned_graph_completion",
        "execute": bool(args.execute),
        "plan": plan,
        "provider_call_count": 0,
        "qdrant_write_count": 0,
        "gray_zone_model_call_count": 0,
        "provider_response_persisted": False,
        "impact": (
            "reconcile committed batch summary metadata and dispatch the exact "
            "graph-commit-bound Redis cache invalidation; no provider or Qdrant calls"
            if args.execute
            else "no PostgreSQL, Redis, Qdrant, or provider writes"
        ),
    }
    if args.execute:
        result = await reconcile_versioned_graph_completion(
            batch_id=str(args.batch_id),
            expected_plan_hash=str(plan["plan_hash"]),
        )
        payload["result"] = result
        cache_state = dict(result.get("cache_invalidation") or {})
        payload["status"] = (
            "passed"
            if cache_state.get("status") == "dispatched"
            else "failed_cache_invalidation_pending"
        )
    else:
        payload["status"] = "dry_run"
    report = write_report("reconcile_versioned_graph_completion", payload)
    print(
        json.dumps(
            {"output": str(report), **payload},
            ensure_ascii=False,
            default=str,
        )
    )
    if payload["status"] == "failed_cache_invalidation_pending":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
