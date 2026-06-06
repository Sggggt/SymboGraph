#!/usr/bin/env python3
"""No-fallback quality gate for DB/Qdrant chunk-vector health.

Run inside the API container:
    python /app/scripts/quality_gate.py --course-name "Course Name"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import select

from app.core.config import get_settings
from app.db import SessionLocal, ensure_schema
from app.models import Chunk, ConceptRelation, Course, GraphCommunitySummary, GraphExtractionChunkTask, GraphExtractionRun, GraphRelationCandidate, QualityProfile
from app.services.strategy_profiles import get_active_profile_record
from app.services.vector_store import VectorStore

LEGACY_ENTRYPOINT_SYMBOLS = (
    "graph_enhanced_search",
    "graph_enhanced_search_v2",
    "layered_search_chunks",
    "local_graph_search",
    "community_search_chunks",
    "RetrievalExecutor",
    "GRAPH_EXTRACTION_CHUNK_LIMIT",
    "GRAPH_EXTRACTION_CHUNKS_PER_DOCUMENT",
    "graph_extraction_chunk_limit",
    "graph_extraction_chunks_per_document",
)


def assert_legacy_entrypoints_removed(repo_root: Path) -> None:
    """Fail if removed compatibility-only entrypoints return to source files."""
    scan_targets = [
        repo_root / "README.md",
        repo_root / "README.en.md",
        repo_root / "todo.md",
        repo_root / "apps" / "api" / "app",
        repo_root / "apps" / "api" / "tests",
        repo_root / "apps" / "web" / "src",
        repo_root / "packages" / "shared" / "src",
        repo_root / "scripts",
    ]
    current_file = Path(__file__).resolve()
    matches: list[str] = []
    for target in scan_targets:
        paths = [target] if target.is_file() else target.rglob("*") if target.exists() else []
        for path in paths:
            if path == current_file or path.suffix.lower() not in {".py", ".ts", ".tsx", ".md", ".env", ".example"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            for symbol in LEGACY_ENTRYPOINT_SYMBOLS:
                if symbol in text:
                    matches.append(f"{path.relative_to(repo_root)}: {symbol}")
    if matches:
        joined = "\n  ".join(matches)
        raise SystemExit(f"Legacy fallback-only entrypoints remain in source:\n  {joined}")


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check active chunk/vector health.")
    parser.add_argument("--course-name", default=None)
    parser.add_argument("--course-id", default=None)
    parser.add_argument("--delete-orphan-zero-vectors", action="store_true")
    parser.add_argument("--allow-flat-chunks", action="store_true", help="Do not fail on active chunks without parent/child metadata.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise SystemExit("Quality gate must run with ENABLE_MODEL_FALLBACK=false and ENABLE_DATABASE_FALLBACK=false.")
    repo_root = Path(__file__).resolve().parents[1]
    assert_legacy_entrypoints_removed(repo_root)

    ensure_schema()
    with SessionLocal() as db:
        course_query = select(Course)
        if args.course_id:
            course_query = course_query.where(Course.id == args.course_id)
        if args.course_name:
            course_query = course_query.where(Course.name == args.course_name)
        courses = db.scalars(course_query.order_by(Course.name.asc())).all()
        if not courses:
            raise SystemExit("No matching courses found.")

        ok = True
        for course in courses:
            chunks = db.scalars(select(Chunk).where(Chunk.course_id == course.id, Chunk.is_active.is_(True))).all()
            active_ids = {chunk.id for chunk in chunks}
            parent_count = sum(1 for chunk in chunks if (chunk.metadata_json or {}).get("is_parent") is True)
            explicit_child_count = sum(1 for chunk in chunks if (chunk.metadata_json or {}).get("is_parent") is False)
            legacy_flat_count = len(chunks) - parent_count - explicit_child_count
            child_without_parent = [
                chunk.id
                for chunk in chunks
                if (chunk.metadata_json or {}).get("is_parent") is False and not chunk.parent_chunk_id
            ]
            quality_policy_counts: dict[str, int] = {}
            chunk_action_counts: dict[str, int] = {}
            chunk_retention_counts = {"retained": 0, "discarded": 0, "missing": 0}
            route_counts = {"embed": 0, "retrieval": 0, "graph_extraction": 0, "summary": 0, "evidence_only": 0, "missing": 0}
            for chunk in chunks:
                metadata = chunk.metadata_json or {}
                policy = str(metadata.get("quality_policy") or "missing")
                quality_policy_counts[policy] = quality_policy_counts.get(policy, 0) + 1
                action = str(metadata.get("quality_action") or "missing")
                chunk_action_counts[action] = chunk_action_counts.get(action, 0) + 1
                if "quality_retain" in metadata:
                    if metadata.get("quality_retain") is False:
                        chunk_retention_counts["discarded"] += 1
                    else:
                        chunk_retention_counts["retained"] += 1
                else:
                    chunk_retention_counts["missing"] += 1
                routes = metadata.get("route_eligibility")
                if isinstance(routes, dict):
                    for route_name in ("embed", "retrieval", "graph_extraction", "summary", "evidence_only"):
                        route_counts[route_name] += int(routes.get(route_name) is True)
                else:
                    route_counts["missing"] += 1
            active_profile = db.scalar(
                select(QualityProfile)
                .where(QualityProfile.course_id == course.id, QualityProfile.is_active.is_(True))
                .order_by(QualityProfile.created_at.desc())
            )
            strategy_profile = get_active_profile_record(db, course.id)
            candidate_count = db.query(GraphRelationCandidate).filter(GraphRelationCandidate.course_id == course.id).count()
            accepted_relation_count = db.query(ConceptRelation).filter(ConceptRelation.course_id == course.id).count()
            forbidden_relation_count = (
                db.query(ConceptRelation)
                .filter(
                    ConceptRelation.course_id == course.id,
                    ConceptRelation.relation_source.in_(["semantic_sparse", "dijkstra_inferred"]),
                )
                .count()
            )
            candidate_only_in_relation_table = [
                relation.id
                for relation in db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course.id)).all()
                if (relation.metadata_json or {}).get("candidate_only")
            ]
            relation_rows = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course.id)).all()
            missing_evidence_relations = [
                relation.id
                for relation in relation_rows
                if relation.relation_source not in {"prerequisite_chain"} and not relation.evidence_chunk_id
            ]
            active_by_id = {chunk.id: chunk for chunk in chunks}
            relation_evidence_inactive = [
                relation.id
                for relation in relation_rows
                if relation.evidence_chunk_id and relation.evidence_chunk_id not in active_ids
            ]
            relation_evidence_not_retrievable = [
                relation.id
                for relation in relation_rows
                if relation.evidence_chunk_id
                and relation.evidence_chunk_id in active_by_id
                and isinstance((active_by_id[relation.evidence_chunk_id].metadata_json or {}).get("route_eligibility"), dict)
                and not (active_by_id[relation.evidence_chunk_id].metadata_json or {}).get("route_eligibility", {}).get("retrieval")
            ]
            graph_chunks_not_retrievable = [
                chunk.id
                for chunk in chunks
                if isinstance((chunk.metadata_json or {}).get("route_eligibility"), dict)
                and (chunk.metadata_json or {}).get("route_eligibility", {}).get("graph_extraction")
                and not (chunk.metadata_json or {}).get("route_eligibility", {}).get("retrieval")
            ]
            active_community_summaries = db.scalars(
                select(GraphCommunitySummary).where(
                    GraphCommunitySummary.course_id == course.id,
                    GraphCommunitySummary.is_active.is_(True),
                )
            ).all()
            community_versions = sorted({summary.version for summary in active_community_summaries})
            latest_extraction_run = db.scalar(
                select(GraphExtractionRun)
                .where(GraphExtractionRun.course_id == course.id)
                .order_by(GraphExtractionRun.created_at.desc())
            )
            extraction_task_counts: dict[str, int] = {}
            if latest_extraction_run:
                for status, count in db.query(GraphExtractionChunkTask.status, GraphExtractionChunkTask.id).filter(
                    GraphExtractionChunkTask.run_id == latest_extraction_run.id
                ).all():
                    extraction_task_counts[str(status)] = extraction_task_counts.get(str(status), 0) + 1

            vector_store = VectorStore(course_name=course.name)
            vector_ids = set(vector_store.list_ids(course.id))
            missing = sorted(active_ids - vector_ids)
            orphan = sorted(vector_ids - active_ids)

            zero_ids: list[str] = []
            checked = 0
            for index in range(0, len(vector_ids), 100):
                batch_ids = sorted(vector_ids)[index : index + 100]
                points = vector_store.get_points(batch_ids)
                checked += len(points)
                for point in points:
                    if vector_norm(point.get("vector") or []) <= 1e-12:
                        zero_ids.append(str(point["id"]))

            orphan_zero_ids = sorted(set(zero_ids).intersection(orphan))
            if args.delete_orphan_zero_vectors and orphan_zero_ids:
                vector_store.delete(orphan_zero_ids)
                orphan = sorted(set(orphan) - set(orphan_zero_ids))
                zero_ids = sorted(set(zero_ids) - set(orphan_zero_ids))

            print(f"\nCourse: {course.name}")
            print(f"  active_chunks={len(chunks)} parent_chunks={parent_count} child_chunks={explicit_child_count} legacy_flat_chunks={legacy_flat_count}")
            print(f"  qdrant_vectors={len(vector_ids)} checked_vectors={checked}")
            print(f"  missing_vectors={len(missing)} orphan_vectors={len(orphan)} zero_vectors={len(zero_ids)}")
            print(f"  child_without_parent={len(child_without_parent)}")
            print(
                "  strategy_profile="
                f"{strategy_profile.name} id={strategy_profile.id} hash={strategy_profile.profile_hash}"
            )
            print(f"  quality_profile={active_profile.version if active_profile else 'missing'}")
            print(f"  quality_policy_counts={quality_policy_counts}")
            print(f"  chunk_action_counts={chunk_action_counts}")
            print(f"  chunk_retention_counts={chunk_retention_counts}")
            print(f"  route_eligibility_counts={route_counts}")
            print(f"  graph_relations={accepted_relation_count} graph_relation_candidates={candidate_count}")
            print(f"  forbidden_candidate_relations={forbidden_relation_count} candidate_only_relation_rows={len(candidate_only_in_relation_table)}")
            print(f"  relation_rows_missing_evidence={len(missing_evidence_relations)}")
            print(f"  relation_evidence_inactive_chunks={len(relation_evidence_inactive)}")
            print(f"  relation_evidence_not_retrievable={len(relation_evidence_not_retrievable)}")
            print(f"  graph_chunks_not_retrievable={len(graph_chunks_not_retrievable)}")
            print(f"  community_summaries={len(active_community_summaries)} community_versions={community_versions or ['missing']}")
            print(
                "  graph_extraction_run="
                f"{latest_extraction_run.id if latest_extraction_run else 'missing'} "
                f"status={latest_extraction_run.status if latest_extraction_run else 'missing'} "
                f"strategy={latest_extraction_run.strategy if latest_extraction_run else 'missing'} "
                f"profile_hash={latest_extraction_run.strategy_profile_hash if latest_extraction_run else 'missing'} "
                f"tasks={extraction_task_counts}"
            )
            if orphan_zero_ids:
                print(f"  orphan_zero_vectors={', '.join(orphan_zero_ids)}")

            if (
                missing
                or orphan
                or zero_ids
                or child_without_parent
                or forbidden_relation_count
                or candidate_only_in_relation_table
                or missing_evidence_relations
                or relation_evidence_inactive
                or relation_evidence_not_retrievable
                or graph_chunks_not_retrievable
                or (legacy_flat_count and not args.allow_flat_chunks)
            ):
                ok = False

        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
