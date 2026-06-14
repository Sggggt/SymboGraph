from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import IngestionBatch


CANCEL_REQUESTED = "cancel_requested"
CANCELLED = "cancelled"
CANCEL_FAILED = "cancel_failed"
CANCELLING_STATES = {CANCEL_REQUESTED, "cancelling", "compensating"}


class IngestionCancelled(RuntimeError):
    """Raised when a cooperative ingestion cancellation is observed."""


def is_cancel_requested(db: Session, batch_id: str | None) -> bool:
    if not batch_id:
        return False
    try:
        from app.db import SessionLocal

        with SessionLocal() as session:
            batch = session.get(IngestionBatch, batch_id)
            if batch is None:
                return False
            stats = batch.stats or {}
            return batch.status in CANCELLING_STATES or bool(stats.get("cancel_requested"))
    except Exception:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            return False
        stats = batch.stats or {}
        return batch.status in CANCELLING_STATES or bool(stats.get("cancel_requested"))


def ensure_not_cancelled(db: Session, batch_id: str | None) -> None:
    if is_cancel_requested(db, batch_id):
        raise IngestionCancelled("ingestion batch cancellation requested")
