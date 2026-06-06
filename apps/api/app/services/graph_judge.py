from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Concept, ConceptRelation, Course, Document
from app.services.concept_graph import filter_graph_documents
from app.services.embeddings import ChatProvider
from app.services.parsers import is_invalid_chapter_label
from app.services.strategy_profiles import get_active_profile_record, profile_prompt


def build_graph_judge_evidence(db: Session, course_id: str, sample_limit: int = 20) -> dict[str, Any]:
    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"Course not found: {course_id}")
    strategy_profile = get_active_profile_record(db, course_id)

    all_concepts = db.scalars(select(Concept).where(Concept.course_id == course.id).order_by(Concept.importance_score.desc(), Concept.canonical_name)).all()
    concepts = all_concepts[:sample_limit]
    active_documents = db.scalars(
        select(Document).where(Document.course_id == course.id, Document.is_active.is_(True)).order_by(Document.title)
    ).all()
    documents = filter_graph_documents(course, active_documents)
    graph_document_ids = {document.id for document in documents}
    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.course_id == course.id, Chunk.is_active.is_(True), Chunk.document_id.in_(graph_document_ids))
        .order_by(Chunk.created_at.asc())
        .limit(sample_limit)
    ).all()
    chapter_counter = Counter(chapter for concept in all_concepts for chapter in (concept.chapter_refs or []))
    invalid_refs = sorted(
        {
            chapter
            for concept in all_concepts
            for chapter in (concept.chapter_refs or [])
            if is_invalid_chapter_label(chapter, course_name=course.name)
        }
    )
    concept_count = db.scalar(select(func.count(Concept.id)).where(Concept.course_id == course.id)) or 0
    relation_count = db.scalar(select(func.count(ConceptRelation.id)).where(ConceptRelation.course_id == course.id)) or 0
    chunk_count = (
        db.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.course_id == course.id,
                Chunk.is_active.is_(True),
                Chunk.document_id.in_(graph_document_ids),
            )
        )
        or 0
    )
    document_count = len(documents)
    return {
        "course": course.name,
        "course_id": course.id,
        "strategy_profile_id": strategy_profile.id,
        "strategy_profile_name": strategy_profile.name,
        "strategy_profile_hash": strategy_profile.profile_hash,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "concept_count": concept_count,
        "relation_count": relation_count,
        "concepts_per_100_chunks": round((concept_count / max(chunk_count, 1)) * 100, 2),
        "relations_per_concept": round(relation_count / max(concept_count, 1), 2),
        "community_count": len({item.community_louvain for item in all_concepts if item.community_louvain is not None}),
        "mean_graph_rank_score": round(
            sum(float(getattr(item, "graph_rank_score", 0.0) or 0.0) for item in all_concepts) / max(len(all_concepts), 1),
            4,
        ),
        "chapter_ref_counts": dict(chapter_counter),
        "invalid_chapter_refs": invalid_refs,
        "documents": [{"title": item.title, "tags": item.tags, "source_path": item.source_path} for item in documents[:sample_limit]],
        "concepts": [
            {
                "name": item.canonical_name,
                "chapter_refs": item.chapter_refs,
                "importance_score": item.importance_score,
                "community_louvain": getattr(item, "community_louvain", None),
                "evidence_count": getattr(item, "evidence_count", 0),
                "graph_rank_score": getattr(item, "graph_rank_score", 0.0),
            }
            for item in concepts
        ],
        "sample_chunks": [
            {"chapter": item.chapter, "section": item.section, "page_number": item.page_number, "snippet": item.snippet[:400]}
            for item in chunks
        ],
    }

async def run_graph_judge(db: Session, course_id: str) -> dict[str, Any]:
    evidence = build_graph_judge_evidence(db, course_id)
    strategy_profile = get_active_profile_record(db, course_id)
    system_prompt = profile_prompt(
        strategy_profile.profile_json,
        "graph_judge_system",
        "You are an LLM-as-a-judge for a course knowledge graph pipeline. Return strict JSON.",
    )
    threshold_hint = profile_prompt(
        strategy_profile.profile_json,
        "graph_judge_threshold_hint",
        (
            "Use these acceptance thresholds: invalid_chapter_refs must be empty, "
            "concepts_per_100_chunks >= 5, relations_per_concept >= 2.5, and multi-chapter courses should have at least "
            "5 distinct chapter_ref_counts entries."
        ),
    )
    user_prompt = (
        "Evaluate graph quality and chapter reference correctness. Return JSON with keys: "
        "verdict, severity, concept_density, relation_density, chapter_ref_findings, invalid_chapter_refs, "
        f"recommended_fixes, acceptance_tests. {threshold_hint} Do not reject solely for duplicate source copies if the evidence was "
        "already filtered to current graph documents.\n\nEvidence:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    return await ChatProvider().classify_json(system_prompt, user_prompt, fallback={})


async def _main() -> None:
    from app.db import SessionLocal

    parser = argparse.ArgumentParser(description="Run a read-only LLM-as-Judge graph quality diagnostic.")
    parser.add_argument("course", help="Course id or exact course name")
    args = parser.parse_args()

    with SessionLocal() as db:
        course = db.get(Course, args.course) or db.scalar(select(Course).where(Course.name == args.course))
        if course is None:
            raise SystemExit(f"Course not found: {args.course}")
        result = await run_graph_judge(db, course.id)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
