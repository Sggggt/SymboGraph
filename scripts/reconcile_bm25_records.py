from __future__ import annotations

import argparse
import json

from sqlalchemy import select

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile bm25_records against current chunks and contextual texts.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true", help="Create missing BM25 records. Omit for dry-run.")
    parser.add_argument("--delete-stale", action="store_true", help="With --execute, delete BM25 records for stale chunk rows.")
    return parser.parse_args()


def main() -> None:
    from app.models import BM25Record, Chunk, ChunkContextText
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION, TOKENIZER_VERSION, stable_hash
    from app.services.context_graph import tokenize_for_bm25

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        active_chunks = db.scalars(select(Chunk).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")).all()
        active_ids = {chunk.id for chunk in active_chunks}
        records = db.scalars(select(BM25Record).where(BM25Record.knowledge_base_id == knowledge_base.id)).all()
        record_ids = {record.chunk_id for record in records}
        missing = [chunk for chunk in active_chunks if chunk.id not in record_ids]
        stale = [record for record in records if record.chunk_id not in active_ids]
        created = 0
        deleted = 0
        if args.execute:
            for chunk in missing:
                context_text = db.scalar(
                    select(ChunkContextText)
                    .where(
                        ChunkContextText.chunk_id == chunk.id,
                        ChunkContextText.embedding_text_version == CURRENT_EMBEDDING_TEXT_VERSION,
                    )
                    .order_by(ChunkContextText.created_at.desc())
                )
                text = context_text.contextual_text if context_text else chunk.text
                terms = tokenize_for_bm25(text)
                frequencies = {term: terms.count(term) for term in sorted(set(terms))}
                db.add(
                    BM25Record(
                        knowledge_base_id=knowledge_base.id,
                        chunk_id=chunk.id,
                        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
                        tokenizer_version=TOKENIZER_VERSION,
                        text_hash=stable_hash(text),
                        token_count=len(terms),
                        term_frequencies_json=frequencies,
                        document_length=len(terms),
                        state="ready",
                        diagnostics_json={"source": "reconcile_bm25_records"},
                    )
                )
                created += 1
            if args.delete_stale and stale:
                db.query(BM25Record).filter(BM25Record.id.in_([record.id for record in stale])).delete(synchronize_session=False)
                deleted = len(stale)
            db.commit()
        payload = {
            "script": "reconcile_bm25_records",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "execute": args.execute,
            "delete_stale": args.delete_stale,
            "active_chunks": len(active_chunks),
            "bm25_records": len(records),
            "missing_records": len(missing),
            "stale_records": len(stale),
            "created_records": created,
            "deleted_records": deleted,
            "impact": "create missing records and optionally delete stale records" if args.execute else "no writes",
        }
        report = write_report("reconcile_bm25_records", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
