from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
for candidate in (str(REPOSITORY_ROOT), str(API_ROOT), "/app", "/app/apps/api"):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


PROTOCOL_VERSION = "orphan_mid_rq_membership_cleanup_v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory or explicitly delete only mid_concept_memberships whose "
            "rq_prefix_id has no rq_prefixes row. Dry-run is the default."
        )
    )
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-count", type=int)
    parser.add_argument("--confirm-target-hash")
    parser.add_argument(
        "--output",
        help="Optional new JSON receipt path; existing files are never overwritten.",
    )
    return parser.parse_args()


def _emit(payload: dict[str, Any], *, output: str | None) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if output:
        with Path(output).open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)


def _orphan_rows(
    db,
    *,
    knowledge_base_id: str,
    lock: bool,
) -> list[dict[str, str]]:
    from sqlalchemy import select

    from app.models import MidConcept, MidConceptMembership, RQPrefix

    statement = (
        select(
            MidConceptMembership.id,
            MidConceptMembership.mid_concept_id,
            MidConceptMembership.rq_prefix_id,
        )
        .outerjoin(
            RQPrefix,
            RQPrefix.id == MidConceptMembership.rq_prefix_id,
        )
        .join(
            MidConcept,
            MidConcept.id == MidConceptMembership.mid_concept_id,
        )
        .where(
            MidConcept.knowledge_base_id == knowledge_base_id,
            RQPrefix.id.is_(None),
        )
        .order_by(MidConceptMembership.id.asc())
    )
    if lock:
        statement = statement.with_for_update(
            of=MidConceptMembership,
        )
    return [
        {
            "id": str(row.id),
            "mid_concept_id": str(row.mid_concept_id),
            "rq_prefix_id": str(row.rq_prefix_id),
        }
        for row in db.execute(statement)
    ]


def _inventory(
    db,
    *,
    knowledge_base_id: str,
    lock: bool,
) -> dict[str, Any]:
    targets = _orphan_rows(
        db,
        knowledge_base_id=knowledge_base_id,
        lock=lock,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "knowledge_base_id": knowledge_base_id,
        "target_table": "mid_concept_memberships",
        "orphan_condition": (
            "rq_prefix_id NOT PRESENT IN rq_prefixes.id"
        ),
        "target_count": len(targets),
        "target_hash": _canonical_hash(targets),
        "targets": targets,
        "impact": (
            "Deletes only orphan membership rows; no concept, rq_prefix, "
            "chunk, document, vector, graph-state, or knowledge-base row is "
            "selected."
        ),
    }


async def _run(args: argparse.Namespace) -> int:
    from app.core.config import get_settings
    from app.db import SessionLocal
    from app.models import KnowledgeBase, MidConceptMembership
    from app.services.ingestion_resource_lock import (
        knowledge_base_ingestion_resource_lock,
    )

    settings = get_settings()
    if settings.enable_database_fallback:
        raise RuntimeError(
            "Refusing cleanup while ENABLE_DATABASE_FALLBACK is true"
        )
    with SessionLocal() as db:
        if db.get(KnowledgeBase, args.knowledge_base_id) is None:
            raise RuntimeError("Exact knowledge base does not exist")
        if args.execute:
            lock_context = knowledge_base_ingestion_resource_lock(
                db,
                args.knowledge_base_id,
                operation="cleanup_orphan_mid_rq_memberships",
            )
        else:
            lock_context = None
        if lock_context is not None:
            async with lock_context:
                return _run_with_session(db, args, MidConceptMembership)
        return _run_with_session(db, args, MidConceptMembership)


def _run_with_session(db, args, membership_model) -> int:
    from sqlalchemy import delete

    inventory = _inventory(
        db,
        knowledge_base_id=args.knowledge_base_id,
        lock=bool(args.execute),
    )
    if not args.execute:
        db.rollback()
        _emit(
            {**inventory, "executed": False, "status": "dry_run"},
            output=args.output,
        )
        return 0
    if (
        args.confirm_count != inventory["target_count"]
        or args.confirm_target_hash != inventory["target_hash"]
    ):
        db.rollback()
        raise RuntimeError(
            "Exact orphan cleanup confirmations do not match the locked "
            "target inventory"
        )
    target_ids = [item["id"] for item in inventory["targets"]]
    deleted_count = 0
    if target_ids:
        result = db.execute(
            delete(membership_model).where(
                membership_model.id.in_(target_ids)
            )
        )
        deleted_count = int(result.rowcount or 0)
    if deleted_count != inventory["target_count"]:
        db.rollback()
        raise RuntimeError(
            "Orphan cleanup row count changed inside the transaction"
        )
    remaining = _inventory(
        db,
        knowledge_base_id=args.knowledge_base_id,
        lock=False,
    )
    if remaining["target_count"] != 0:
        db.rollback()
        raise RuntimeError(
            "Orphan cleanup postcondition failed before commit"
        )
    db.commit()
    _emit(
        {
            **inventory,
            "executed": True,
            "status": "completed",
            "deleted_count": deleted_count,
            "remaining_orphan_count": 0,
            "transaction_committed": True,
        },
        output=args.output,
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
