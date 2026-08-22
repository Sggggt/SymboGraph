from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import (
    resolve_knowledge_base,
    session_scope,
    write_report,
)
from _destructive_cleanup_guard import (
    build_stale_cleanup_inventory,
    require_exact_confirmation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory stale derived state in read-only mode. Destructive "
            "execution requires exact knowledge-base and complete-inventory "
            "confirmations copied from the latest dry-run."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the exact confirmed cleanup. Omit for dry-run.",
    )
    parser.add_argument(
        "--confirm-knowledge-base-id",
        help="With --execute, exactly repeat the resolved dry-run KB id.",
    )
    parser.add_argument(
        "--confirm-inventory-hash",
        help="With --execute, exactly repeat the complete dry-run inventory hash.",
    )
    parser.add_argument(
        "--delete-inactive-chunks",
        action="store_true",
        help=(
            "Include exact inactive chunks, document versions, chunk "
            "versions, and their dependent rows in the destructive inventory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute and (
        not args.confirm_knowledge_base_id
        or not args.confirm_inventory_hash
    ):
        raise SystemExit(
            "--execute requires --confirm-knowledge-base-id and "
            "--confirm-inventory-hash from the latest dry-run"
        )

    from app.services.maintenance import (
        cleanup_stale_data,
        cleanup_stale_data_lock,
    )

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        preview_stats = None
        if args.execute:
            inventory = build_stale_cleanup_inventory(
                db,
                knowledge_base,
                delete_inactive_chunks=args.delete_inactive_chunks,
            )
        else:
            with cleanup_stale_data_lock(db, knowledge_base.id):
                inventory = build_stale_cleanup_inventory(
                    db,
                    knowledge_base,
                    delete_inactive_chunks=args.delete_inactive_chunks,
                )
                preview_stats = cleanup_stale_data(
                    db,
                    knowledge_base.id,
                    knowledge_base.name,
                    dry_run=True,
                    delete_inactive_chunks=args.delete_inactive_chunks,
                    lock_already_held=True,
                )
        payload = {
            "script": "cleanup_stale_data",
            "mode": "execute" if args.execute else "dry_run",
            "execute": bool(args.execute),
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "delete_inactive_chunks": bool(args.delete_inactive_chunks),
            "inventory": inventory,
            "impact": inventory["impact"],
            "confirmation": {
                "required_on_execute": [
                    "--confirm-knowledge-base-id",
                    "--confirm-inventory-hash",
                ],
                "knowledge_base_id_matches": (
                    args.confirm_knowledge_base_id == knowledge_base.id
                    if args.execute
                    else None
                ),
                "inventory_hash_matches": (
                    args.confirm_inventory_hash
                    == inventory["inventory_hash"]
                    if args.execute
                    else None
                ),
            },
        }
        if not args.execute:
            payload["stats"] = preview_stats
            payload["transaction"] = {
                "database_committed": False,
                "toctou_inventory_revalidated_under_lock": False,
            }
        else:
            require_exact_confirmation(
                actual_knowledge_base_id=knowledge_base.id,
                actual_inventory_hash=inventory["inventory_hash"],
                confirmed_knowledge_base_id=args.confirm_knowledge_base_id,
                confirmed_inventory_hash=args.confirm_inventory_hash,
            )
            authorized_report = write_report(
                "cleanup_stale_data_authorized",
                {
                    **payload,
                    "authorization_state": "confirmed_before_mutation",
                },
            )
            payload["authorized_plan_output"] = str(authorized_report)
            revalidated: dict[str, object] = {}

            def revalidate_under_lock() -> None:
                current = build_stale_cleanup_inventory(
                    db,
                    knowledge_base,
                    delete_inactive_chunks=args.delete_inactive_chunks,
                )
                require_exact_confirmation(
                    actual_knowledge_base_id=knowledge_base.id,
                    actual_inventory_hash=current["inventory_hash"],
                    confirmed_knowledge_base_id=args.confirm_knowledge_base_id,
                    confirmed_inventory_hash=args.confirm_inventory_hash,
                )
                revalidated["inventory_hash"] = current["inventory_hash"]

            payload["stats"] = cleanup_stale_data(
                db,
                knowledge_base.id,
                knowledge_base.name,
                dry_run=False,
                delete_inactive_chunks=args.delete_inactive_chunks,
                pre_mutation_check=revalidate_under_lock,
            )
            payload["transaction"] = {
                "database_committed": True,
                "toctou_inventory_revalidated_under_lock": (
                    revalidated.get("inventory_hash")
                    == args.confirm_inventory_hash
                ),
                "revalidated_inventory_hash": revalidated.get(
                    "inventory_hash"
                ),
            }
        report = write_report("cleanup_stale_data", payload)
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
