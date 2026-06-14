from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean stale derived state and optionally delete inactive chunk versions.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true", help="Apply cleanup. Omit for dry-run.")
    parser.add_argument(
        "--delete-inactive-chunks",
        action="store_true",
        help="With --execute, delete inactive chunks, inactive document versions, and old chunk_versions.",
    )
    return parser.parse_args()


def main() -> None:
    from app.services.maintenance import cleanup_stale_data

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        payload = {
            "script": "cleanup_stale_data",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "execute": args.execute,
            "delete_inactive_chunks": args.delete_inactive_chunks,
            "stats": cleanup_stale_data(
                db,
                knowledge_base.id,
                knowledge_base.name,
                dry_run=not args.execute,
                delete_inactive_chunks=args.delete_inactive_chunks,
            ),
        }
        report = write_report("cleanup_stale_data", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
