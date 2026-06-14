from pathlib import Path
import asyncio

from worker_app.bootstrap import API_ROOT  # noqa: F401
from worker_app.celery_app import celery_app
from app.db import SessionLocal
from app.models import KnowledgeBase
from app.services.ingestion import ingest_file, mark_batch_task_started, run_batch_ingestion, run_uploaded_files_ingestion
from app.services.maintenance import reconcile_policy_state, reconcile_vector_store_sync
from app.services.runtime_settings import refresh_runtime_settings_if_needed
from sqlalchemy import select


@celery_app.task(name="ingest_path")
def ingest_path(path: str, trigger_source: str = "watchdog", job_id: str | None = None) -> dict:
    refresh_runtime_settings_if_needed(force=True)
    session = SessionLocal()
    try:
        return asyncio.run(ingest_file(session, Path(path), trigger_source=trigger_source, existing_job_id=job_id))
    finally:
        session.close()


@celery_app.task(name="ingest_batch", bind=True)
def ingest_batch(self, batch_id: str) -> dict:
    refresh_runtime_settings_if_needed(force=True)
    mark_batch_task_started(batch_id, getattr(self.request, "id", None), "ingest_batch")
    return asyncio.run(run_batch_ingestion(batch_id))


@celery_app.task(name="ingest_uploaded_batch", bind=True)
def ingest_uploaded_batch(self, batch_id: str, file_paths: list[str], force: bool = False, full_reparse: bool = False) -> dict:
    refresh_runtime_settings_if_needed(force=True)
    mark_batch_task_started(batch_id, getattr(self.request, "id", None), "ingest_uploaded_batch")
    return asyncio.run(run_uploaded_files_ingestion(batch_id, file_paths, force=force, full_reparse=full_reparse, execution_mode="celery"))


@celery_app.task(name="reconcile_vector_store")
def reconcile_vector_store_task(knowledge_base_id: str | None = None) -> dict:
    refresh_runtime_settings_if_needed(force=True)
    session = SessionLocal()
    try:
        return reconcile_vector_store_sync(session, knowledge_base_id=knowledge_base_id)
    finally:
        session.close()


@celery_app.task(name="update_policy_backfill")
def update_policy_backfill_task(knowledge_base_id: str | None = None) -> dict:
    refresh_runtime_settings_if_needed(force=True)
    session = SessionLocal()
    try:
        knowledge_base_ids = [knowledge_base_id] if knowledge_base_id else list(session.scalars(select(KnowledgeBase.id)).all())
        updated = 0
        reconciled = 0
        for kb_id in knowledge_base_ids:
            reconciled += reconcile_policy_state(session, knowledge_base_id=kb_id)
        session.commit()
        return {"updated_observations": updated, "reconciled_policy_states": reconciled, "knowledge_bases": len(knowledge_base_ids)}
    finally:
        session.close()
