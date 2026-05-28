from celery import Celery

from worker_app.bootstrap import API_ROOT  # noqa: F401
from app.core.config import get_settings


settings = get_settings()
celery_app = Celery("knowledge_base_worker", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_default_queue = settings.ingestion_task_queue
celery_app.conf.task_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_serializer = "json"
celery_app.autodiscover_tasks(["worker_app"])
