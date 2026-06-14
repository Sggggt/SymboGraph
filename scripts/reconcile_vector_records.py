from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile PostgreSQL vector_records against Qdrant points.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    return parser.parse_args()


def main() -> None:
    from app.services.maintenance import reconcile_vector_store_sync

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        payload = {
            "script": "reconcile_vector_records",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "stats": reconcile_vector_store_sync(db, knowledge_base_id=knowledge_base.id),
        }
        report = write_report("reconcile_vector_records", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
