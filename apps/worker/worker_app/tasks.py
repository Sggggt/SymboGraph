import asyncio
from pathlib import Path

from worker_app.bootstrap import API_ROOT  # noqa: F401
from worker_app.celery_app import celery_app
from app.core.config import get_settings
from app.db import SessionLocal
from app.models import KnowledgeBase
from app.services.ingestion import (
    finalize_interrupted_batches,
    ingest_file,
    mark_batch_task_started,
    run_batch_ingestion,
    run_context_graph_rebuild_batch,
    run_uploaded_files_ingestion,
)
from app.services.maintenance import reconcile_policy_state, reconcile_vector_store_sync
from app.services.runtime_settings import (
    refresh_runtime_settings_if_needed,
    sync_model_bridge_runtime_config,
)
from app.services.runtime_settings_lifecycle import (
    build_runtime_settings_shadow,
    record_runtime_settings_build_failure,
)
from app.services.storage import ensure_storage_durability_ready
from app.services.strategy_profiles import (
    reconcile_builtin_default_profile_startup,
    reconcile_pending_profile_lifecycle_events,
)
from sqlalchemy import select


def _refresh_runtime_and_require_durable_storage() -> None:
    before_refresh = get_settings()
    before_root = str(Path(before_refresh.data_root).absolute())
    # Re-open the path no-follow and re-check PID/device/inode/mount signature
    # before Redis/model-bridge refresh or any task-state DB write. A fork,
    # identity/mount change, first use, or bounded TTL expiry causes a complete
    # real probe; hot tasks do not create probe files on every invocation.
    ensure_storage_durability_ready(settings=before_refresh)
    # Reload managed bytes and clear local caches first, but do not mutate the
    # bridge until DB/Redis/current-file/source admission has been replayed
    # against the refreshed process identity.
    refresh_runtime_settings_if_needed(force=True, sync_bridge=False)
    after_refresh = get_settings()
    after_root = str(Path(after_refresh.data_root).absolute())
    if after_root != before_root:
        ensure_storage_durability_ready(settings=after_refresh, force_probe=True)
    sync_model_bridge_runtime_config(settings=after_refresh)
    # A worker may start and consume queued work before API startup or beat.
    # Reconcile code-owned Profile upgrades and any pending cache/version
    # delivery before the task can read a Profile or invoke a model.
    reconcile_builtin_default_profile_startup()
    profile_session = SessionLocal()
    try:
        profile_reconcile = reconcile_pending_profile_lifecycle_events(
            profile_session, limit=128, raise_on_error=False
        )
        if profile_reconcile["failed"]:
            raise RuntimeError(
                "Profile lifecycle side effects remain pending at task boundary"
            )
    finally:
        profile_session.close()
    # A profile reconcile may publish a newer compatible runtime version. The
    # next task boundary reload observes it before any task body runs.


@celery_app.task(name="ingest_path")
def ingest_path(path: str, trigger_source: str = "watchdog", job_id: str | None = None) -> dict:
    _refresh_runtime_and_require_durable_storage()
    session = SessionLocal()
    try:
        return asyncio.run(ingest_file(session, Path(path), trigger_source=trigger_source, existing_job_id=job_id))
    finally:
        session.close()


@celery_app.task(name="ingest_batch", bind=True)
def ingest_batch(self, batch_id: str) -> dict:
    _refresh_runtime_and_require_durable_storage()
    mark_batch_task_started(batch_id, getattr(self.request, "id", None), "ingest_batch")
    return asyncio.run(run_batch_ingestion(batch_id))


@celery_app.task(name="ingest_uploaded_batch", bind=True)
def ingest_uploaded_batch(self, batch_id: str, file_paths: list[str], force: bool = False, full_reparse: bool = False) -> dict:
    _refresh_runtime_and_require_durable_storage()
    mark_batch_task_started(batch_id, getattr(self.request, "id", None), "ingest_uploaded_batch")
    return asyncio.run(run_uploaded_files_ingestion(batch_id, file_paths, force=force, full_reparse=full_reparse, execution_mode="celery"))


@celery_app.task(name="rebuild_context_graph_batch", bind=True)
def rebuild_context_graph_batch(self, batch_id: str) -> dict:
    _refresh_runtime_and_require_durable_storage()
    mark_batch_task_started(batch_id, getattr(self.request, "id", None), "rebuild_context_graph_batch")
    return asyncio.run(run_context_graph_rebuild_batch(batch_id, execution_mode="celery"))


@celery_app.task(name="reconcile_vector_store")
def reconcile_vector_store_task(knowledge_base_id: str | None = None) -> dict:
    _refresh_runtime_and_require_durable_storage()
    session = SessionLocal()
    try:
        return reconcile_vector_store_sync(session, knowledge_base_id=knowledge_base_id)
    finally:
        session.close()


@celery_app.task(name="reconcile_interrupted_ingestion_batches")
def reconcile_interrupted_ingestion_batches_task() -> dict:
    """Retry durable batch recovery without relying on API process startup."""

    _refresh_runtime_and_require_durable_storage()
    finalize_interrupted_batches()
    return {"status": "reconciled"}


@celery_app.task(name="reconcile_profile_lifecycle")
def reconcile_profile_lifecycle_task() -> dict:
    """Retry durable Profile cache/version broadcasts at a bounded cadence."""

    _refresh_runtime_and_require_durable_storage()
    builtin = reconcile_builtin_default_profile_startup()
    session = SessionLocal()
    try:
        result = reconcile_pending_profile_lifecycle_events(
            session, limit=128, raise_on_error=False
        )
        if result["failed"]:
            raise RuntimeError(
                "Profile lifecycle side effects remain pending after worker retry"
            )
        return {**result, "builtin_default_profile": builtin}
    finally:
        session.close()


@celery_app.task(name="update_policy_backfill")
def update_policy_backfill_task(knowledge_base_id: str | None = None) -> dict:
    _refresh_runtime_and_require_durable_storage()
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


@celery_app.task(name="build_runtime_settings_candidate", bind=True)
def build_runtime_settings_candidate_task(
    self,
    candidate_id: str,
    build_ids: list[str],
) -> dict:
    """Run a bounded candidate shadow build through the shared API service.

    The task refreshes Runtime Settings at entry and again at each knowledge-
    base boundary.  Candidate overrides themselves remain ContextVar-local in
    the lifecycle service and never mutate this worker's process environment.
    """

    _refresh_runtime_and_require_durable_storage()
    if not build_ids or len(build_ids) > 64:
        raise ValueError("Runtime Settings candidate build scope must contain 1..64 builds")
    session = SessionLocal()
    completed: list[str] = []
    try:
        for build_id in build_ids:
            refresh_runtime_settings_if_needed(force=True)
            try:
                asyncio.run(
                    build_runtime_settings_shadow(session, build_id=str(build_id))
                )
                session.commit()
                completed.append(str(build_id))
            except Exception as exc:
                session.rollback()
                record_runtime_settings_build_failure(
                    session,
                    build_id=str(build_id),
                    error_type=type(exc).__name__,
                )
                session.commit()
                raise RuntimeError(
                    "Runtime Settings candidate shadow build failed: "
                    f"candidate_id={candidate_id}; build_id={build_id}; "
                    f"error_type={type(exc).__name__}"
                ) from None
        return {
            "candidate_id": candidate_id,
            "completed_build_ids": completed,
            "completed_count": len(completed),
            "task_id": getattr(self.request, "id", None),
        }
    finally:
        session.close()
