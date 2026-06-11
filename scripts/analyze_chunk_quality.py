#!/usr/bin/env python3
"""Analyze active chunk quality from the configured database.

Run inside the API container:
    python /app/scripts/analyze_chunk_quality.py --knowledge-base-name "Knowledge Base"
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import select

from app.db import SessionLocal
from app.models import ActiveChunk, Document, KnowledgeBase, VectorRecord


def _metadata_value(chunk: ActiveChunk, key: str, default=None):
    return (chunk.metadata_json or {}).get(key, default)


def _chunk_text(chunk: ActiveChunk) -> str:
    return chunk.text or ""


def analyze(knowledge_base_name: str | None, include_inactive: bool) -> None:
    with SessionLocal() as db:
        query = select(ActiveChunk, KnowledgeBase).join(KnowledgeBase, ActiveChunk.knowledge_base_id == KnowledgeBase.id)
        if knowledge_base_name:
            query = query.where(KnowledgeBase.name == knowledge_base_name)
        if not include_inactive:
            query = query.where(ActiveChunk.state == "active")
        rows = db.execute(query.order_by(KnowledgeBase.name.asc(), ActiveChunk.created_at.asc())).all()

        document_ids = {
            str(_metadata_value(chunk, "document_id"))
            for chunk, _knowledge_base in rows
            if _metadata_value(chunk, "document_id")
        }
        documents = {
            document.id: document
            for document in db.scalars(select(Document).where(Document.id.in_(document_ids or {"__none__"}))).all()
        }
        vector_records = db.scalars(
            select(VectorRecord).where(VectorRecord.active_chunk_id.in_({chunk.id for chunk, _kb in rows} or {"__none__"}))
        ).all()
        vector_versions = {record.active_chunk_id: record.embedding_text_version for record in vector_records}

    print(f"=== Total active chunks: {len(rows)} ===\n")

    kind_counts = Counter(_metadata_value(chunk, "content_kind", "unknown") for chunk, _knowledge_base in rows)
    print("=== content_kind distribution ===")
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count}")

    print("\n=== has_table distribution ===")
    for value, count in Counter(_metadata_value(chunk, "has_table") for chunk, _kb in rows).most_common():
        print(f"  {value}: {count}")

    print("\n=== has_formula distribution ===")
    for value, count in Counter(_metadata_value(chunk, "has_formula") for chunk, _kb in rows).most_common():
        print(f"  {value}: {count}")

    all_kinds = ["pdf_page", "text", "table", "formula", "mixed", "markdown", "code", "html", "other", "unknown"]
    knowledge_base_stats = defaultdict(
        lambda: {key: 0 for key in all_kinds + ["total", "has_table", "has_formula", "active_atoms", "high_digit_ratio"]}
    )
    for chunk, knowledge_base in rows:
        stats = knowledge_base_stats[knowledge_base.name]
        stats["total"] += 1
        kind = str(_metadata_value(chunk, "content_kind", "unknown") or "unknown")
        stats[kind if kind in stats else "other"] += 1
        if _metadata_value(chunk, "has_table"):
            stats["has_table"] += 1
        if _metadata_value(chunk, "has_formula"):
            stats["has_formula"] += 1
        stats["active_atoms"] += len(chunk.atom_ids_json or [])
        text = _chunk_text(chunk)
        digits_dots = sum(1 for char in text if char.isdigit() or char == ".")
        if text and digits_dots / len(text) > 0.6:
            stats["high_digit_ratio"] += 1

    print("\n=== Per-knowledge-base stats ===")
    for knowledge_base, stats in sorted(knowledge_base_stats.items()):
        print(f"\n  {knowledge_base}:")
        print(f"    active_chunks: {stats['total']}")
        print(
            "    content_kind: "
            f"pdf_page={stats['pdf_page']}, text={stats['text']}, table={stats['table']}, "
            f"formula={stats['formula']}, markdown={stats['markdown']}, code={stats['code']}, "
            f"html={stats['html']}, other={stats['other']}, unknown={stats['unknown']}"
        )
        print(f"    has_table={stats['has_table']}, has_formula={stats['has_formula']}")
        print(f"    atom_refs={stats['active_atoms']}, high_digit_ratio={stats['high_digit_ratio']}")

    print("\n=== High digit ratio (>60%) active chunks ===")
    high_digit_chunks = []
    for chunk, knowledge_base in rows:
        text = _chunk_text(chunk)
        if len(text) <= 50:
            continue
        digits_dots = sum(1 for char in text if char.isdigit() or char == ".")
        ratio = digits_dots / len(text)
        if ratio > 0.6:
            high_digit_chunks.append((ratio, chunk, knowledge_base))
    for ratio, chunk, knowledge_base in sorted(high_digit_chunks, key=lambda item: item[0], reverse=True):
        document = documents.get(str(_metadata_value(chunk, "document_id")))
        preview = _chunk_text(chunk)[:80].replace("\n", " ")
        print(
            f"  [{knowledge_base.name}] {document.title if document else 'unknown document'} | "
            f"kind={_metadata_value(chunk, 'content_kind', 'unknown')} | "
            f"digit_ratio={ratio:.1%} len={len(_chunk_text(chunk))} | {preview}..."
        )

    print("\n=== Active source files ===")
    seen_documents: set[str] = set()
    for chunk, knowledge_base in rows:
        document_id = str(_metadata_value(chunk, "document_id") or "")
        if not document_id or document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        document = documents.get(document_id)
        if document:
            print(f"  [{knowledge_base.name}] {document.title} ({document.source_type})")

    print("\n=== embedding_text_version distribution ===")
    version_counts = Counter(
        vector_versions.get(chunk.id)
        or _metadata_value(chunk, "embedding_text_version")
        or "unknown"
        for chunk, _knowledge_base in rows
    )
    for version, count in version_counts.most_common():
        print(f"  {version}: {count}")

    print("\n=== embedding_quality_score stats ===")
    scores = [
        float(_metadata_value(chunk, "embedding_quality_score"))
        for chunk, _kb in rows
        if _metadata_value(chunk, "embedding_quality_score") is not None
    ]
    if scores:
        print(f"  Count: {len(scores)}")
        print(f"  Min: {min(scores):.3f}")
        print(f"  Max: {max(scores):.3f}")
        print(f"  Mean: {statistics.mean(scores):.3f}")
        print(f"  Median: {statistics.median(scores):.3f}")
    else:
        print("  No scores found")

    print("\n=== Duplicate active chunk text analysis ===")
    content_counts = Counter(_chunk_text(chunk) for chunk, _knowledge_base in rows)
    duplicates = [(content, count) for content, count in content_counts.most_common() if count > 1 and len(content) > 20]
    print(f"  Unique contents: {len(content_counts)}")
    print(f"  Contents with duplicates: {len(duplicates)}")
    print(f"  Total duplicate instances: {sum(count for _content, count in duplicates)}")
    for content, count in duplicates[:10]:
        print(f"    x{count}: {content[:60].replace(chr(10), ' ')}...")

    print("\n=== Active chunk size distribution ===")
    sizes = [len(_chunk_text(chunk)) for chunk, _knowledge_base in rows]
    if sizes:
        print(f"  Count: {len(sizes)}")
        print(f"  Min: {min(sizes)}")
        print(f"  Max: {max(sizes)}")
        print(f"  Mean: {statistics.mean(sizes):.0f}")
        print(f"  Median: {statistics.median(sizes):.0f}")
    else:
        print("  No active chunks found")

    print("\n=== Active chunk size by content_kind ===")
    kind_sizes: dict[str, list[int]] = defaultdict(list)
    for chunk, _knowledge_base in rows:
        kind_sizes[str(_metadata_value(chunk, "content_kind", "unknown") or "unknown")].append(len(_chunk_text(chunk)))
    for kind, values in sorted(kind_sizes.items()):
        print(f"  {kind}: min={min(values)}, max={max(values)}, mean={statistics.mean(values):.0f}, median={statistics.median(values):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze active chunk quality from the configured database.")
    parser.add_argument("--knowledge-base-name", "--KnowledgeBase-name", dest="knowledge_base_name", default=None)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()
    analyze(args.knowledge_base_name, args.include_inactive)
