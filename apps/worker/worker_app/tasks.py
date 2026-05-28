from pathlib import Path
import asyncio

from worker_app.bootstrap import API_ROOT  # noqa: F401
from worker_app.celery_app import celery_app
from app.db import SessionLocal
from app.services.ingestion import ingest_file, run_batch_ingestion, run_graph_rebuild, run_uploaded_files_ingestion


@celery_app.task(name="ingest_path")
def ingest_path(path: str, trigger_source: str = "watchdog", job_id: str | None = None) -> dict:
    session = SessionLocal()
    try:
        return asyncio.run(ingest_file(session, Path(path), trigger_source=trigger_source, existing_job_id=job_id))
    finally:
        session.close()


@celery_app.task(name="ingest_batch")
def ingest_batch(batch_id: str) -> dict:
    return asyncio.run(run_batch_ingestion(batch_id))


@celery_app.task(name="ingest_uploaded_batch")
def ingest_uploaded_batch(batch_id: str, file_paths: list[str], force: bool = False) -> dict:
    return asyncio.run(run_uploaded_files_ingestion(batch_id, file_paths, force=force))


@celery_app.task(name="rebuild_course_graph")
def rebuild_course_graph_task(batch_id: str, course_id: str, mode: str = "full") -> dict:
    return asyncio.run(run_graph_rebuild(batch_id, course_id, mode))
