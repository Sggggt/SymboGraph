from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import active_chunk_count, resolve_knowledge_base, session_scope, storage_files, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full source-file reparse into fixed token chunks and all four graph layers.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true", help="Start the reparse. Omit for dry-run.")
    parser.add_argument("--full-reparse", action="store_true", help="Create current_chunk_version + 1 when parsed chunks already exist.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def main() -> None:
    from app.services.ingestion import create_uploaded_files_batch, run_uploaded_files_ingestion

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        files = storage_files(knowledge_base.name)
        chunks = active_chunk_count(db, knowledge_base.id)
        if chunks > 0 and not args.full_reparse:
            raise SystemExit("Existing parsed chunks require --full-reparse so the destructive version bump is explicit.")
        payload = {
            "script": "rebuild_chunks",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "file_count": len(files),
            "active_chunks_before": chunks,
            "full_reparse": args.full_reparse,
            "execute": args.execute,
            "files": [str(path) for path in files],
            "impact": "parse files, write chunks/structure/contextual indexes, rebuild all graph layers" if args.execute else "no writes",
        }
        if args.execute:
            if not files:
                raise SystemExit("No source files found in knowledge base storage.")
            batch = create_uploaded_files_batch(db, knowledge_base.id, files, force=args.force, full_reparse=args.full_reparse)
            result = await run_uploaded_files_ingestion(batch.id, [str(path) for path in files], force=args.force, full_reparse=args.full_reparse, execution_mode="script")
            payload["batch"] = result
        report = write_report("rebuild_chunks", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
