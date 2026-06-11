#!/usr/bin/env python3
"""Repair or refresh active chunk vectors with the configured embedding provider.

Run inside the API container:
    python /app/scripts/reembed_all_chunks.py --knowledge-base-name "Knowledge Base" --dry-run

The script refuses to run when model or database fallback is enabled.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import select

from app.core.config import get_settings
from app.db import SessionLocal
from app.models import ActiveChunk, Document, KnowledgeBase, VectorRecord
from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION, contextual_embedding_text
from app.services.embeddings import EmbeddingProvider, validate_embedding_vectors
from app.services.evidence_graph import stable_hash
from app.services.vector_store import VectorStore


def vector_is_zero(vector: list[float]) -> bool:
    return all(abs(float(value)) < 1e-12 for value in vector)


def metadata(chunk: ActiveChunk) -> dict[str, Any]:
    return dict(chunk.metadata_json or {})


def chunk_summary(chunk: ActiveChunk | None, max_chars: int = 180) -> str | None:
    if chunk is None:
        return None
    meta = metadata(chunk)
    for value in (meta.get("summary"), meta.get("snippet"), chunk.snippet, chunk.text):
        text = str(value or "").strip()
        if text:
            return text[:max_chars]
    return None


def chunk_keywords(chunk: ActiveChunk) -> list[str] | None:
    raw = metadata(chunk).get("keywords")
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return values or None
    return None


def section_sort_key(chunk: ActiveChunk) -> tuple[str, int, int, str]:
    meta = metadata(chunk)
    return (
        str(chunk.document_id or ""),
        int(meta.get("section_index") or 0),
        int(meta.get("chunk_index") or 0),
        chunk.id,
    )


def contextual_text(chunk: ActiveChunk, document: Document | None, active_chunks_by_id: dict[str, ActiveChunk], siblings: list[ActiveChunk]) -> str:
    meta = metadata(chunk)
    sibling_index = next((index for index, item in enumerate(siblings) if item.id == chunk.id), -1)
    prev_chunk = siblings[sibling_index - 1] if sibling_index > 0 else None
    next_chunk = siblings[sibling_index + 1] if sibling_index >= 0 and sibling_index + 1 < len(siblings) else None
    parent_chunk = active_chunks_by_id.get(str(chunk.parent_chunk_id or ""))
    return contextual_embedding_text(
        document_title=document.title if document else "Unknown",
        partition=chunk.partition,
        section=chunk.section,
        source_type=chunk.source_type,
        content_kind=meta.get("content_kind"),
        content=chunk.text or "",
        parent_summary=chunk_summary(parent_chunk, max_chars=220),
        prev_summary=chunk_summary(prev_chunk),
        next_summary=chunk_summary(next_chunk),
        summary=str(meta.get("summary") or "") or None,
        keywords=chunk_keywords(chunk),
        has_table=bool(meta.get("has_table", False)),
        has_formula=bool(meta.get("has_formula", False)),
    )


def vector_payload(chunk: ActiveChunk, document: Document | None, knowledge_base_id: str) -> dict[str, Any]:
    meta = metadata(chunk)
    return {
        "active_chunk_id": chunk.id,
        "chunk_id": chunk.id,
        "knowledge_base_id": knowledge_base_id,
        "document_id": chunk.document_id,
        "document_version_id": chunk.document_version_id,
        "document_title": document.title if document else "Unknown",
        "source_path": document.source_path if document else "",
        "partition": chunk.partition,
        "section": chunk.section,
        "page_number": chunk.page_number,
        "snippet": chunk.snippet,
        "source_type": chunk.source_type,
        "content": chunk.text,
        "content_kind": meta.get("content_kind"),
        "atom_ids": chunk.atom_ids_json or [],
        "community_ids": chunk.community_ids_json or [],
        "graph_state_hash": chunk.graph_state_hash,
        "quality_decision_id": chunk.quality_decision_id,
        "policy_state_id": chunk.policy_state_id,
        "source_span_union": chunk.source_span_union_json or {},
        "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
    }


def upsert_vector_record(db, *, knowledge_base_id: str, chunk: ActiveChunk, point: dict[str, Any], embedding_model: str) -> None:
    payload_hash = stable_hash(point.get("payload") or {})
    db.query(VectorRecord).filter(VectorRecord.active_chunk_id == chunk.id).delete(synchronize_session=False)
    db.add(
        VectorRecord(
            knowledge_base_id=knowledge_base_id,
            active_chunk_id=chunk.id,
            qdrant_point_id=str(point["id"]),
            embedding_model=embedding_model,
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
            payload_hash=payload_hash,
            vector_status="ready",
            diagnostics_json={"payload_hash": payload_hash, "script": "reembed_all_chunks"},
        )
    )


async def reembed_chunks(
    *,
    knowledge_base_name: str | None,
    knowledge_base_id: str | None,
    batch_size: int,
    dry_run: bool,
    refresh_all: bool,
) -> None:
    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise SystemExit("Refusing to re-embed while fallback is enabled.")
    if not settings.openai_api_key and not dry_run:
        raise SystemExit("OPENAI_API_KEY is required for real no-fallback re-embedding.")

    print(f"Embedding model: {settings.embedding_model}")
    print(f"Embedding dimensions: {settings.embedding_dimensions}")
    print(f"Embedding base URL: {settings.embedding_base_url}")
    print(f"Dry run: {dry_run}")
    print(f"Mode: {'refresh all active vectors' if refresh_all else 'repair missing/zero vectors'}\n")

    with SessionLocal() as db:
        query = select(ActiveChunk).where(ActiveChunk.state == "active")
        if knowledge_base_id:
            query = query.where(ActiveChunk.knowledge_base_id == knowledge_base_id)
        if knowledge_base_name:
            matched_id = db.scalars(select(KnowledgeBase.id).where(KnowledgeBase.name == knowledge_base_name)).first()
            if not matched_id:
                raise SystemExit(f"No knowledge base found named {knowledge_base_name!r}.")
            query = query.where(ActiveChunk.knowledge_base_id == matched_id)
        active_chunks = db.scalars(query.order_by(ActiveChunk.knowledge_base_id.asc(), ActiveChunk.created_at.asc())).all()
        if not active_chunks:
            print("No active chunks found.")
            return

        chunks_by_kb: dict[str, list[ActiveChunk]] = defaultdict(list)
        for chunk in active_chunks:
            chunks_by_kb[chunk.knowledge_base_id].append(chunk)

        embedder = None if dry_run else EmbeddingProvider()
        total_selected = 0
        total_updated = 0

        for kb_id, kb_chunks in chunks_by_kb.items():
            knowledge_base = db.get(KnowledgeBase, kb_id)
            if knowledge_base is None:
                print(f"  [SKIP] knowledge base {kb_id} not found")
                continue

            vector_store = VectorStore(knowledge_base_name=knowledge_base.name)
            active_chunk_ids = [chunk.id for chunk in kb_chunks]
            existing_points = {}
            for start in range(0, len(active_chunk_ids), 100):
                for point in vector_store.get_points(active_chunk_ids[start : start + 100]):
                    existing_points[str(point["id"])] = point

            selected = []
            for chunk in kb_chunks:
                point = existing_points.get(chunk.id)
                if refresh_all or point is None or vector_is_zero(point.get("vector") or []):
                    selected.append(chunk)
            total_selected += len(selected)
            print(f"--- Knowledge base: {knowledge_base.name} ---")
            print(f"  active_chunks={len(kb_chunks)} selected_for_reembed={len(selected)}")
            if dry_run or not selected:
                continue

            document_ids = {str(chunk.document_id) for chunk in kb_chunks if chunk.document_id}
            documents = {
                document.id: document
                for document in db.scalars(select(Document).where(Document.id.in_(document_ids or {'__none__'}))).all()
            }
            active_chunks_by_id = {chunk.id: chunk for chunk in kb_chunks}
            siblings_by_section: dict[tuple[str, str], list[ActiveChunk]] = defaultdict(list)
            for chunk in kb_chunks:
                meta = metadata(chunk)
                key = (str(chunk.document_version_id or ""), str(meta.get("section_index") or ""))
                siblings_by_section[key].append(chunk)
            for siblings in siblings_by_section.values():
                siblings.sort(key=section_sort_key)

            assert embedder is not None
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                texts = []
                for chunk in batch:
                    meta = metadata(chunk)
                    section_key = (str(chunk.document_version_id or ""), str(meta.get("section_index") or ""))
                    texts.append(
                        contextual_text(
                            chunk,
                            documents.get(str(chunk.document_id or "")),
                            active_chunks_by_id,
                            siblings_by_section.get(section_key, []),
                        )
                    )

                result = await embedder.embed_texts_with_meta(texts, text_type="document")
                validate_embedding_vectors(
                    result.vectors,
                    expected_count=len(batch),
                    expected_dimensions=settings.embedding_dimensions,
                )

                points = []
                for chunk, vector in zip(batch, result.vectors):
                    document = documents.get(str(chunk.document_id or ""))
                    payload = vector_payload(chunk, document, kb_id)
                    points.append({"id": chunk.id, "vector": vector, "payload": payload})
                    meta = metadata(chunk)
                    meta["embedding_text_version"] = CURRENT_EMBEDDING_TEXT_VERSION
                    chunk.metadata_json = meta

                vector_store.upsert(points)
                for chunk, point in zip(batch, points):
                    upsert_vector_record(
                        db,
                        knowledge_base_id=kb_id,
                        chunk=chunk,
                        point=point,
                        embedding_model=settings.embedding_model,
                    )
                db.commit()
                total_updated += len(batch)
                print(f"  updated {start + len(batch)}/{len(selected)} via {result.provider}")

        print("\n=== Summary ===")
        print(f"Selected active chunks: {total_selected}")
        print(f"Updated vectors: {total_updated}")
        if dry_run:
            print("Dry run: no changes made.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Repair or refresh active chunk vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Report selected chunks without writing")
    parser.add_argument("--refresh-all", action="store_true", help="Re-embed all matching active chunks, not only missing/zero vectors")
    parser.add_argument("--batch-size", type=int, default=10, help="Embedding batch size")
    parser.add_argument("--knowledge-base-id", "--KnowledgeBase-id", dest="knowledge_base_id", default=None)
    parser.add_argument("--knowledge-base-name", "--KnowledgeBase-name", dest="knowledge_base_name", default=None)
    args = parser.parse_args()
    await reembed_chunks(
        knowledge_base_name=args.knowledge_base_name,
        knowledge_base_id=args.knowledge_base_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        refresh_all=args.refresh_all,
    )


if __name__ == "__main__":
    asyncio.run(main())
