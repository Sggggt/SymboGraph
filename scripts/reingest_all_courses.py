#!/usr/bin/env python3
"""Reingest course storage files through the normal Docker API service code.

Run inside the API container:
    python /app/scripts/reingest_all_courses.py --course-name "Course Name" --cleanup-stale

The script scans DATA_ROOT course directories, creates a sync ingestion batch,
and runs the full parse -> chunk -> embed -> graph pipeline. It refuses to run
when model or database fallback is enabled.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

# 将项目根加入路径
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "apps" / "api"))

from app.db import SessionLocal
from app.models import Course
from app.core.config import get_settings
from app.services.ingestion import active_chunk_count, create_sync_batch, run_batch_ingestion
from app.services.maintenance import cleanup_stale_data

EXCLUDE_DIRS = {"ingestion", "postgres", "qdrant", "redis", "storage", "models"}


def get_all_courses(course_name: str | None = None) -> list[Path]:
    data_root = get_settings().data_root
    if not data_root.exists():
        return []
    courses = [
        d for d in data_root.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DIRS and (d / "storage").exists()
    ]
    if course_name:
        courses = [course for course in courses if course.name == course_name]
    return courses


def ensure_course_record(db, course_name: str) -> Course:
    """Get or create a course record for a DATA_ROOT course directory."""
    course = db.query(Course).filter(Course.name == course_name).first()
    if course is None:
        from app.services.ingestion import create_course_space
        course = create_course_space(db, course_name)
    else:
        paths = get_settings().course_paths_for_name(course.name)
        paths["course_root"].mkdir(parents=True, exist_ok=True)
        paths["storage_root"].mkdir(parents=True, exist_ok=True)
        paths["ingestion_root"].mkdir(parents=True, exist_ok=True)
    return course


async def reingest_course(course_name: str, cleanup_stale: bool) -> dict:
    db = SessionLocal()
    try:
        course = ensure_course_record(db, course_name)
        paths = get_settings().course_paths_for_name(course.name)
        storage_root = paths["storage_root"]

        if not storage_root.exists():
            return {"course": course_name, "status": "skipped", "reason": "no storage directory"}

        batch = create_sync_batch(db, course.id, storage_root, trigger_source="reingest")
        print(f"  [batch {batch.id}] created for course '{course_name}'")

        result = await run_batch_ingestion(batch.id, force=True, full_reparse=active_chunk_count(db, course.id) > 0)
        cleanup_stats = None
        if cleanup_stale and result.get("state") in {"completed", "partial_failed"}:
            cleanup_stats = cleanup_stale_data(db, course.id, course.name)
        return {
            "course": course_name,
            "status": result.get("state", "unknown"),
            "batch_id": batch.id,
            "cleanup": cleanup_stats,
        }
    except Exception as exc:
        return {"course": course_name, "status": "error", "error": str(exc)}
    finally:
        db.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reingest course storage files through the Docker API code path.")
    parser.add_argument("--course-name", default=None, help="Limit to one course directory name.")
    parser.add_argument("--cleanup-stale", action="store_true", help="Delete inactive DB rows and stale Qdrant vectors after each successful course run.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise SystemExit("Refusing to reingest while fallback is enabled.")
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for real no-fallback reingestion.")

    courses = get_all_courses(args.course_name)
    if not courses:
        print("No matching course directories found in DATA_ROOT")
        return

    print(f"Found {len(courses)} course(s) to re-ingest:")
    for c in courses:
        print(f"  - {c.name}")

    for course_dir in courses:
        print(f"\nRe-ingesting: {course_dir.name}")
        result = await reingest_course(course_dir.name, args.cleanup_stale)
        print(f"  -> {result['status']}")
        if result.get("cleanup") is not None:
            print(f"     cleanup={result['cleanup']}")
        if "error" in result:
            print(f"     ERROR: {result['error']}")

    print("\nAll courses re-ingestion completed.")


if __name__ == "__main__":
    asyncio.run(main())
