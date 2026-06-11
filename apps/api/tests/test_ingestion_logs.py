from __future__ import annotations


def test_ingestion_logs_are_persisted_and_replayed(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.services.ingestion_logs import emit_ingestion_log, list_ingestion_logs, subscribe_ingestion_logs, unsubscribe_ingestion_logs

    batch = IngestionBatch(knowledge_base_id=sample_knowledge_base.id, source_root="unit-tests", trigger_source="upload", status="queued")
    db_session.add(batch)
    db_session.commit()
    db_session.refresh(batch)

    emit_ingestion_log(batch.id, "batch_started", "Parsing test file", processed_files=0, total_files=1)
    emit_ingestion_log(batch.id, "batch_completed", "Batch completed", processed_files=1, total_files=1)

    persisted = list_ingestion_logs(batch.id)
    assert [item["event"] for item in persisted] == ["batch_started", "batch_completed"]
    assert persisted[0]["processed_files"] == 0
    assert persisted[0]["log_id"]

    history, subscriber = subscribe_ingestion_logs(batch.id)
    try:
        assert [item["event"] for item in history] == ["batch_started", "batch_completed"]
    finally:
        unsubscribe_ingestion_logs(batch.id, subscriber)


def test_log_stream_tokens_are_batch_scoped_and_expire():
    from app.services import ingestion_logs

    issued = ingestion_logs.create_log_stream_token("batch-a", ttl_seconds=60)
    ingestion_logs.validate_log_stream_token("batch-a", issued["token"])

    try:
        ingestion_logs.validate_log_stream_token("batch-b", issued["token"])
    except ValueError as exc:
        assert "not valid for this batch" in str(exc)
    else:
        raise AssertionError("cross-batch log token should be rejected")

    expired = ingestion_logs.create_log_stream_token("batch-a", ttl_seconds=-1)
    try:
        ingestion_logs.validate_log_stream_token("batch-a", expired["token"])
    except ValueError as exc:
        assert "Invalid or expired" in str(exc)
    else:
        raise AssertionError("expired log token should be rejected")


def test_batch_logs_route_accepts_token_and_rejects_query_api_key(db_session, sample_knowledge_base, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app
    from app.models import IngestionBatch
    from app.services.ingestion_logs import create_log_stream_token, emit_ingestion_log

    monkeypatch.setenv("API_KEYS", "secret-key")
    get_settings.cache_clear()

    batch = IngestionBatch(knowledge_base_id=sample_knowledge_base.id, source_root="unit-tests", trigger_source="upload", status="completed")
    db_session.add(batch)
    db_session.commit()
    db_session.refresh(batch)
    emit_ingestion_log(batch.id, "batch_completed", "Done")
    token = create_log_stream_token(batch.id)["token"]

    client = TestClient(app)
    assert client.post(f"/api/ingestion/batches/{batch.id}/log-token?api_key=secret-key").status_code == 401
    assert client.post(f"/api/ingestion/batches/{batch.id}/log-token", headers={"X-API-Key": "secret-key"}).status_code == 200
    with client.stream("GET", f"/api/ingestion/batches/{batch.id}/logs?token={token}") as response:
        assert response.status_code == 200
        assert "batch_completed" in response.read().decode("utf-8")
    assert client.get(f"/api/ingestion/batches/{batch.id}/logs?token=wrong-token").status_code == 401

    monkeypatch.delenv("API_KEYS")
    get_settings.cache_clear()
