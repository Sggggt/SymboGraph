from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import (
    resolve_knowledge_base,
    session_scope,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or delete one exact obsolete Qdrant collection. "
            "Dry-run is the default and never infers deletion from orphan scans."
        )
    )
    owner = parser.add_mutually_exclusive_group(required=True)
    owner.add_argument("--knowledge-base-id")
    owner.add_argument("--knowledge-base-name")
    parser.add_argument("--collection-name", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Persist an intent, commit it, then delete the exact collection and finalize PostgreSQL.",
    )
    parser.add_argument(
        "--confirm-delete-exact-collection",
        help="Required with --execute and must exactly repeat --collection-name.",
    )
    return parser.parse_args()


def main() -> None:
    # Parse first so ``--help`` remains a dependency-free/read-only operation.
    # Importing the service initializes the configured database engine and must
    # not happen before argparse has had a chance to exit.
    args = parse_args()
    from app.services.vector_collection_cleanup import (
        execute_vector_collection_cleanup,
        prepare_vector_collection_cleanup,
        vector_collection_cleanup_plan,
    )

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        plan = vector_collection_cleanup_plan(
            db,
            collection_name=args.collection_name,
        )
        payload: dict[str, object] = {
            "script": "cleanup_vector_collection",
            "mode": "execute" if args.execute else "dry_run",
            "execute": bool(args.execute),
            "audit_knowledge_base_id": knowledge_base.id,
            "audit_knowledge_base_name": knowledge_base.name,
            "exact_target": {
                "collection_name": plan["collection_name"],
                "qdrant_collection_exists": plan["qdrant_collection_exists"],
            },
            "plan": plan,
            "impact": (
                "delete exactly one named Qdrant collection and mark its retained "
                "PostgreSQL VectorRecords missing; collection bytes are derived and "
                "recoverable only by rebuilding that canonical vector schema"
                if args.execute
                else "read-only PostgreSQL/Qdrant inventory; no deletion and no status writes"
            ),
        }
        if not args.execute:
            report = write_report("cleanup_vector_collection", payload)
            print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))
            return
        if args.confirm_delete_exact_collection != args.collection_name:
            raise SystemExit(
                "--confirm-delete-exact-collection must exactly repeat --collection-name"
            )
        if not plan["allowed"]:
            raise SystemExit(
                "Collection cleanup blocked: " + ", ".join(plan["blockers"])
            )

        intent = prepare_vector_collection_cleanup(
            db,
            audit_knowledge_base_id=knowledge_base.id,
            collection_name=args.collection_name,
            confirmed_collection_name=args.confirm_delete_exact_collection,
        )
        # Durability boundary: the exact intent must exist before Qdrant I/O.
        db.commit()
        intent_id = intent.id
        try:
            result = execute_vector_collection_cleanup(
                db,
                intent_id=intent_id,
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        payload["intent_id"] = intent_id
        payload["result"] = result
        payload["intent_committed_before_qdrant"] = True
        report = write_report("cleanup_vector_collection", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
