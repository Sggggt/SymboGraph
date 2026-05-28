"""Rebuild course graphs from existing chunks without reparsing or re-embedding.

Run inside the API container, for example:
    python /app/scripts/rebuild_all_course_graphs.py --confirm-destructive

This script intentionally does not call reingest_all_courses.py and does not
modify chunks, documents, embeddings, or Qdrant points.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1] / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from sqlalchemy import select

from app.core.config import get_settings
from app.db import SessionLocal
from app.models import Chunk, Concept, ConceptRelation, Course, Document, IngestionBatch
from app.services.concept_graph import get_graph_payload
from app.services.ingestion import active_batch_for_course, run_graph_rebuild
from app.services.vector_store import VectorStore


def _json_line(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def _graph_eligible(metadata: dict | None) -> bool:
    route_eligibility = (metadata or {}).get("route_eligibility") or {}
    return route_eligibility.get("graph_extraction") is True


def _course_readiness(db, course: Course, *, skip_vector_check: bool) -> dict:
    documents = db.scalars(
        select(Document).where(Document.course_id == course.id, Document.is_active.is_(True))
    ).all()
    chunks = db.scalars(
        select(Chunk).where(Chunk.course_id == course.id, Chunk.is_active.is_(True))
    ).all()
    graph_eligible_chunks = sum(1 for chunk in chunks if _graph_eligible(chunk.metadata_json))
    concepts = db.query(Concept).filter(Concept.course_id == course.id).count()
    relations = db.query(ConceptRelation).filter(ConceptRelation.course_id == course.id).count()
    graph = get_graph_payload(db, course.id, graph_type="semantic")
    vector_count: int | None = None
    vector_error: str | None = None
    if not skip_vector_check:
        try:
            vector_count = len(VectorStore(course.name).list_ids(course.id))
        except Exception as exc:
            vector_error = f"{type(exc).__name__}: {exc}"
    blocking_issues: list[str] = []
    if not documents:
        blocking_issues.append("no_active_documents")
    if not chunks:
        blocking_issues.append("no_active_chunks")
    if graph_eligible_chunks <= 0:
        blocking_issues.append("no_graph_eligible_chunks")
    if vector_error:
        blocking_issues.append("vector_store_unavailable")
    elif not skip_vector_check and not vector_count:
        blocking_issues.append("no_vectors_for_course")
    return {
        "course": course.name,
        "course_id": course.id,
        "documents": len(documents),
        "chunks": len(chunks),
        "graph_eligible_chunks": graph_eligible_chunks,
        "vectors": vector_count,
        "vector_error": vector_error,
        "nodes_before": len(graph.get("nodes") or []),
        "edges_before": len(graph.get("edges") or []),
        "concepts_before": concepts,
        "relations_before": relations,
        "blocking_issues": blocking_issues,
    }


def _select_courses(db, names: list[str]) -> list[Course]:
    statement = select(Course).order_by(Course.name)
    courses = db.scalars(statement).all()
    if not names:
        return courses
    by_name = {course.name: course for course in courses}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown course(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild all course graphs from existing chunks only.")
    parser.add_argument("--course-name", action="append", default=[], help="Course name to rebuild. Repeat for multiple courses.")
    parser.add_argument("--dry-run", action="store_true", help="Only print readiness; do not rebuild.")
    parser.add_argument("--confirm-destructive", action="store_true", help="Required because full graph rebuild replaces graph tables.")
    parser.add_argument("--skip-vector-check", action="store_true", help="Skip Qdrant/vector presence readiness check.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise SystemExit("Refusing to rebuild graphs while fallback is enabled.")
    if not args.dry_run and not args.confirm_destructive:
        raise SystemExit("--confirm-destructive is required for full graph rebuild.")

    db = SessionLocal()
    try:
        courses = _select_courses(db, args.course_name)
        _json_line({"event": "start", "dry_run": args.dry_run, "course_count": len(courses), "started_at": datetime.utcnow()})
        readiness = [_course_readiness(db, course, skip_vector_check=args.skip_vector_check) for course in courses]
        for item in readiness:
            _json_line({"event": "readiness", **item})
        blocking = [item for item in readiness if item["blocking_issues"]]
        if blocking:
            raise SystemExit("Readiness check failed; graph rebuild was not started.")
        if args.dry_run:
            _json_line({"event": "dry_run_completed"})
            return

        for course in courses:
            active = active_batch_for_course(db, course.id)
            if active is not None:
                raise SystemExit(f"Course {course.name} has active batch {active.id} in state {active.status}")
            batch = IngestionBatch(
                course_id=course.id,
                source_root="graph_rebuild_script",
                trigger_source="rebuild_graph",
                status="extracting_graph",
                stats={"script": "rebuild_all_course_graphs.py", "mode": "full"},
            )
            db.add(batch)
            db.commit()
            db.refresh(batch)
            _json_line({"event": "rebuild_started", "course": course.name, "course_id": course.id, "batch_id": batch.id})
            stats = asyncio.run(run_graph_rebuild(batch.id, course.id, mode="full"))
            db.expire_all()
            graph = get_graph_payload(db, course.id, graph_type="semantic")
            nodes = len(graph.get("nodes") or [])
            edges = len(graph.get("edges") or [])
            if nodes <= 0 or edges <= 0:
                raise SystemExit(f"Graph rebuild produced empty graph for {course.name}: nodes={nodes} edges={edges}")
            _json_line(
                {
                    "event": "rebuild_completed",
                    "course": course.name,
                    "course_id": course.id,
                    "batch_id": batch.id,
                    "nodes_after": nodes,
                    "edges_after": edges,
                    "stats": stats,
                }
            )
        _json_line({"event": "completed", "completed_at": datetime.utcnow()})
    finally:
        db.close()


if __name__ == "__main__":
    main()
