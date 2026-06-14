from __future__ import annotations

from datetime import datetime


def test_ingestion_logs_are_persisted_and_replayed(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.services.ingestion_logs import emit_ingestion_log, list_ingestion_logs, subscribe_ingestion_logs, unsubscribe_ingestion_logs

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="unit",
        source_root="unit",
        total_files=1,
        status="queued",
        created_at=datetime.utcnow(),
    )
    db_session.add(batch)
    db_session.commit()
    emit_ingestion_log(batch.id, "batch_started", "Parsing", processed_files=0)
    emit_ingestion_log(batch.id, "batch_completed", "Done", processed_files=1)
    persisted = list_ingestion_logs(batch.id)
    assert [item["event"] for item in persisted][-2:] == ["batch_started", "batch_completed"]
    history, subscriber = subscribe_ingestion_logs(batch.id)
    try:
        assert history[-1]["event"] == "batch_completed"
    finally:
        unsubscribe_ingestion_logs(batch.id, subscriber)
