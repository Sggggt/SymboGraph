from celery import Celery
from celery.signals import beat_init
from celery.signals import worker_process_init

from worker_app.bootstrap import API_ROOT  # noqa: F401
from app.services.storage import (
    ensure_storage_durability_ready,
    fail_closed_native_windows_production_before_settings,
)

# Reject unsupported native-Windows production before worker bootstrap.
fail_closed_native_windows_production_before_settings()

from app.core.config import get_settings
from app.services.runtime_settings import (
    initialize_runtime_env_from_root_file,
    refresh_runtime_settings_if_needed,
)


WORKER_RUNTIME_ENV_INITIALIZATION = initialize_runtime_env_from_root_file()
settings = get_settings()
WORKER_STORAGE_DURABILITY_CAPABILITY = ensure_storage_durability_ready(
    settings=settings,
    force_probe=True,
)
celery_app = Celery("knowledge_base_worker", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_default_queue = settings.ingestion_task_queue
celery_app.conf.task_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_serializer = "json"
celery_app.conf.task_track_started = True
celery_app.conf.beat_schedule = {
    "reconcile-interrupted-ingestion-every-minute": {
        "task": "reconcile_interrupted_ingestion_batches",
        "schedule": 60,
    },
    "reconcile-vector-store-every-30-minutes": {
        "task": "reconcile_vector_store",
        "schedule": 30 * 60,
    },
    "update-policy-backfill-every-5-minutes": {
        "task": "update_policy_backfill",
        "schedule": 5 * 60,
    },
    "reconcile-profile-lifecycle-every-minute": {
        "task": "reconcile_profile_lifecycle",
        "schedule": 60,
    },
}
celery_app.autodiscover_tasks(["worker_app"])


@worker_process_init.connect(weak=False)
def reset_worker_database_pool_after_fork(**_kwargs) -> None:
    """Reset the child pool and refresh read-side runtime identity."""

    from app.db import reset_database_engine_after_fork

    reset_database_engine_after_fork(force=True)
    refresh_runtime_settings_if_needed(force=True, sync_bridge=False)


@beat_init.connect(weak=False)
def refresh_runtime_before_beat(**_kwargs) -> None:
    """Refresh runtime identity before scheduled maintenance work."""

    refresh_runtime_settings_if_needed(force=True, sync_bridge=False)
