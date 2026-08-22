from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
for candidate in (str(REPOSITORY_ROOT), str(API_ROOT), "/app", "/app/apps/api"):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory immutable source snapshots and optionally delete only "
            "exact, retention-eligible orphans through a durable intent."
        )
    )
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument(
        "--retention-seconds",
        type=int,
        default=7 * 24 * 60 * 60,
    )
    parser.add_argument("--max-entries", type=int, default=100_000)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the exact confirmed deletion; default is read-only dry-run.",
    )
    parser.add_argument("--confirm-knowledge-base-id")
    parser.add_argument("--confirm-inventory-hash")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    from app.db import SessionLocal
    from app.services.ingestion import resolve_knowledge_base
    from app.services.ingestion_resource_lock import (
        knowledge_base_ingestion_resource_lock,
    )
    from app.services.storage_maintenance import (
        SOURCE_SNAPSHOT_GC_OPERATION,
        run_source_snapshot_gc,
    )

    with SessionLocal() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            args.knowledge_base_id,
        )
        if not args.execute:
            return run_source_snapshot_gc(
                db,
                knowledge_base,
                retention_seconds=args.retention_seconds,
                max_entries=args.max_entries,
            )
        async with knowledge_base_ingestion_resource_lock(
            db,
            knowledge_base.id,
            operation=SOURCE_SNAPSHOT_GC_OPERATION,
        ):
            return run_source_snapshot_gc(
                db,
                knowledge_base,
                execute=True,
                confirm_knowledge_base_id=args.confirm_knowledge_base_id,
                confirm_inventory_hash=args.confirm_inventory_hash,
                retention_seconds=args.retention_seconds,
                max_entries=args.max_entries,
            )


def main() -> int:
    args = parse_args()
    if args.execute and (
        not args.confirm_knowledge_base_id
        or not args.confirm_inventory_hash
    ):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": (
                        "--execute requires --confirm-knowledge-base-id and "
                        "--confirm-inventory-hash from the latest dry-run"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
