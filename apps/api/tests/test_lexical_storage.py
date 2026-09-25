import pytest
from sqlalchemy import func, select

from app.models import IngestionBatch, LexicalDocument, LexicalIndexJob, LexicalIndexState, LexicalPosting, LexicalTermRecord
from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
from app.services.lexical_index import prepare_bm25_snapshot
from app.services.cancellation import IngestionCancelled
from app.services.lexical_storage import (
    active_lexical_sources,
    build_and_publish_lexical_index,
    load_lexical_snapshot,
    verify_lexical_snapshot_streaming,
    materialize_lexical_index_job,
    prepare_lexical_index_job,
    publish_lexical_index,
    reconcile_pending_lexical_index_jobs,
    request_lexical_index_cancel,
    stage_lexical_index,
)


@pytest.fixture
def indexed_material(db_session, sample_knowledge_base):
    from test_relation_quota_signals import _add_document_chunks
    _, _, chunks = _add_document_chunks(db_session, sample_knowledge_base.id, suffix="lexical-storage",
        chunk_ids=["unit-test-lexical-a", "unit-test-lexical-b"], text="Queue limits and 中文检索 are documented.")
    db_session.commit()
    snapshot = prepare_bm25_snapshot(sample_knowledge_base.id, active_lexical_sources(db_session, sample_knowledge_base.id))
    return sample_knowledge_base, chunks, snapshot


def test_candidate_creation_requires_shared_kb_resource_lock(db_session, indexed_material):
    with pytest.raises(ValueError, match="resource_lock_required"):
        stage_lexical_index(db_session, indexed_material[2])
    assert db_session.scalar(select(func.count()).select_from(LexicalIndexState)) == 0


@pytest.mark.asyncio
async def test_streaming_snapshot_verifies_exact_published_identity(db_session, indexed_material):
    import copy

    kb, chunks, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = stage_lexical_index(db_session, snapshot)
        assert verify_lexical_snapshot_streaming(db_session, job.target_state_id) == snapshot.identity
        posting = db_session.scalar(select(LexicalPosting).where(LexicalPosting.index_state_id == job.target_state_id).limit(1))
        original_positions = copy.deepcopy(posting.positions_json)
        posting.positions_json = [[999, 1000]]
        db_session.commit()
        with pytest.raises(ValueError, match="identity_changed"):
            verify_lexical_snapshot_streaming(db_session, job.target_state_id)
        posting.positions_json = original_positions
        db_session.commit()
        term = db_session.scalar(select(LexicalTermRecord).where(LexicalTermRecord.index_state_id == job.target_state_id).limit(1))
        term.document_frequency += 1
        db_session.commit()
        with pytest.raises(ValueError, match="identity_changed"):
            verify_lexical_snapshot_streaming(db_session, job.target_state_id)
        term.document_frequency -= 1
        db_session.commit()
        chunks[0].text = "Source changed outside index"
        chunks[0].char_end = chunks[0].char_start + len(chunks[0].text)
        db_session.commit()
        with pytest.raises(ValueError, match="source_scope_changed"):
            verify_lexical_snapshot_streaming(db_session, job.target_state_id)


@pytest.mark.asyncio
async def test_candidate_replays_fully_and_cache_failure_leaves_retryable_publication(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    calls = []
    def invalidate(kb_id, *, strict):
        calls.append((kb_id, strict))
        if len(calls) == 1: raise RuntimeError("unit-test-cache-offline")
        return True
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = stage_lexical_index(db_session, snapshot)
        assert job.status == "ready_to_publish"
        assert db_session.get(LexicalIndexState, job.target_state_id).state == "candidate"
        assert load_lexical_snapshot(db_session, job.target_state_id, verify_sources=True) == snapshot
        with pytest.raises(RuntimeError, match="cache-offline"):
            publish_lexical_index(db_session, job.id, invalidate=invalidate)
        db_session.expire_all()
        assert db_session.get(LexicalIndexJob, job.id).status == "published"
        assert db_session.get(LexicalIndexJob, job.id).cache_invalidation_pending
        assert db_session.get(LexicalIndexState, job.target_state_id).state == "active"
        published = publish_lexical_index(db_session, job.id, invalidate=invalidate)
        assert job.status == "completed" and not job.cache_invalidation_pending
        assert publish_lexical_index(db_session, job.id, invalidate=invalidate).id == published.id
    assert calls == [(kb.id, True), (kb.id, True)]


@pytest.mark.asyncio
async def test_source_change_blocks_publication_and_makes_the_old_snapshot_ineligible(db_session, indexed_material):
    kb, chunks, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        first = stage_lexical_index(db_session, snapshot)
        publish_lexical_index(db_session, first.id, invalidate=lambda *a, **k: True)
        next_job = stage_lexical_index(db_session, snapshot)
        chunks[0].text = "Changed source"
        chunks[0].char_end = chunks[0].char_start + len(chunks[0].text)
        db_session.commit()
        with pytest.raises(ValueError, match="source_scope_changed"):
            publish_lexical_index(db_session, next_job.id, invalidate=lambda *a, **k: True)
        assert db_session.get(LexicalIndexState, first.target_state_id).state == "stale"
        assert db_session.get(LexicalIndexState, next_job.target_state_id).state == "failed"
        assert db_session.get(LexicalIndexJob, next_job.id).status == "failed"


@pytest.mark.asyncio
async def test_cancelling_a_prepared_candidate_preserves_a_still_valid_active_snapshot(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        active_job = stage_lexical_index(db_session, snapshot)
        publish_lexical_index(db_session, active_job.id, invalidate=lambda *a, **k: True)
        candidate_job = prepare_lexical_index_job(db_session, snapshot)
        cancelled = request_lexical_index_cancel(db_session, candidate_job.id)
        assert cancelled.status == "cancelled"
        assert db_session.get(LexicalIndexState, candidate_job.target_state_id).state == "failed"
        assert db_session.get(LexicalIndexState, active_job.target_state_id).state == "active"


@pytest.mark.asyncio
async def test_materialization_observes_ingestion_cancellation_and_records_a_terminal_job(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    def cancelled():
        raise IngestionCancelled("unit-test-cancel")
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = prepare_lexical_index_job(db_session, snapshot)
        with pytest.raises(IngestionCancelled, match="unit-test-cancel"):
            materialize_lexical_index_job(
                db_session,
                job.id,
                snapshot=snapshot,
                check_cancelled=cancelled,
            )
        assert db_session.get(LexicalIndexJob, job.id).status == "cancelled"
        assert db_session.get(LexicalIndexState, job.target_state_id).state == "failed"


@pytest.mark.asyncio
async def test_full_build_reuses_the_exact_active_snapshot_without_a_second_job(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        first = stage_lexical_index(db_session, snapshot)
        publish_lexical_index(db_session, first.id, invalidate=lambda *a, **k: True)
        result = build_and_publish_lexical_index(
            db_session,
            kb.id,
            invalidate=lambda *a, **k: True,
        )
        assert result["status"] == "reused"
        assert result["index_state_id"] == first.target_state_id
        assert db_session.scalar(select(func.count()).select_from(LexicalIndexJob)) == 1


@pytest.mark.asyncio
async def test_reconciler_replays_a_prepared_job_to_completed(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = prepare_lexical_index_job(db_session, snapshot)
    result = await reconcile_pending_lexical_index_jobs(
        invalidate=lambda *a, **k: True,
        knowledge_base_id=kb.id,
    )
    db_session.expire_all()
    assert result == {
        "attempted": 1,
        "completed_job_ids": [job.id],
        "cancelled_job_ids": [],
        "failed": [],
        "ok": True,
    }
    assert db_session.get(LexicalIndexJob, job.id).status == "completed"
    assert db_session.get(LexicalIndexState, job.target_state_id).state == "active"


@pytest.mark.asyncio
async def test_reconciler_honors_a_cancelled_ingestion_batch(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    batch = IngestionBatch(
        knowledge_base_id=kb.id,
        source_root=kb.source_root,
        status="queued",
    )
    db_session.add(batch)
    db_session.commit()
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = prepare_lexical_index_job(
            db_session,
            snapshot,
            ingestion_batch_id=batch.id,
        )
    batch.status = "cancelled"
    batch.stats = {"cancel_requested": True}
    db_session.commit()
    result = await reconcile_pending_lexical_index_jobs(
        invalidate=lambda *a, **k: True,
        knowledge_base_id=kb.id,
    )
    db_session.expire_all()
    assert result["cancelled_job_ids"] == [job.id]
    assert db_session.get(LexicalIndexJob, job.id).status == "cancelled"
    assert db_session.get(LexicalIndexState, job.target_state_id).state == "failed"


@pytest.mark.asyncio
async def test_materialization_failure_rolls_back_rows_and_preserves_intention(db_session, indexed_material, monkeypatch):
    from app.services import lexical_storage
    kb, _, snapshot = indexed_material
    original = lexical_storage._bulk_insert
    def failing(db, model, records, **kwargs):
        if model is LexicalTermRecord: raise RuntimeError("unit-test-index-write-failure")
        return original(db, model, records, **kwargs)
    monkeypatch.setattr(lexical_storage, "_bulk_insert", failing)
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        with pytest.raises(RuntimeError, match="index-write-failure"):
            stage_lexical_index(db_session, snapshot)
    assert db_session.scalar(select(func.count()).select_from(LexicalDocument)) == 0
    job = db_session.scalar(select(LexicalIndexJob))
    assert job.status == "failed" and job.error_code == "lexical_materialization_failed"
    assert db_session.get(LexicalIndexState, job.target_state_id).state == "failed"


@pytest.mark.asyncio
async def test_persisted_term_corruption_is_not_accepted_as_a_new_valid_snapshot(db_session, indexed_material):
    kb, _, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = stage_lexical_index(db_session, snapshot)
        term = db_session.scalar(select(LexicalTermRecord).limit(1))
        term.document_frequency += 1
        db_session.commit()
        with pytest.raises(ValueError, match="statistics_invalid"):
            load_lexical_snapshot(db_session, job.target_state_id)


@pytest.mark.asyncio
async def test_persisted_query_matches_reference_with_global_stats_and_real_raw_witnesses(db_session, indexed_material):
    from app.services.lexical_index import search_bm25_snapshot
    from app.services.lexical_storage import search_lexical_index, search_lexical_index_many
    kb, chunks, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = stage_lexical_index(db_session, snapshot)
        publish_lexical_index(db_session, job.id, invalidate=lambda *a, **k: True)
    terms = ["limits", "中文"]
    actual = search_lexical_index(db_session, kb.id, terms, expected_identity=snapshot.identity, limit=1)
    expected = search_bm25_snapshot(snapshot, terms, limit=1)
    assert actual.hits == expected
    assert actual.output_truncated and actual.matched_documents == 2
    filtered = search_lexical_index(db_session, kb.id, terms, expected_identity=snapshot.identity,
        eligible_chunk_ids=frozenset({chunks[0].id}), limit=2)
    assert filtered.hits == expected and not filtered.output_truncated
    assert search_lexical_index(db_session, kb.id, ["notmentioned"], expected_identity=snapshot.identity).hits == ()
    batched = search_lexical_index_many(
        db_session,
        kb.id,
        (["limits"], ["中文"], terms),
        expected_identity=snapshot.identity,
        limit=2,
    )
    assert tuple(item.hits for item in batched) == tuple(
        search_bm25_snapshot(snapshot, query, limit=2)
        for query in (["limits"], ["中文"], terms)
    )
    assert batched[2].postings_read == (
        batched[0].postings_read + batched[1].postings_read
    )
    with pytest.raises(ValueError, match="identity_changed"):
        search_lexical_index(db_session, kb.id, terms, expected_identity="f" * 64)
    with pytest.raises(ValueError, match="scan_budget"):
        search_lexical_index(db_session, kb.id, terms, expected_identity=snapshot.identity, posting_budget=1)


@pytest.mark.asyncio
async def test_query_rejects_tampered_document_frequency_even_when_snapshot_header_is_unchanged(db_session, indexed_material):
    from app.services.lexical_storage import search_lexical_index
    kb, _, snapshot = indexed_material
    async with knowledge_base_ingestion_resource_lock(db_session, kb.id, operation="lexical_index_build"):
        job = stage_lexical_index(db_session, snapshot)
        publish_lexical_index(db_session, job.id, invalidate=lambda *a, **k: True)
        term = db_session.scalar(select(LexicalTermRecord).where(LexicalTermRecord.term == "limits"))
        term.document_frequency = 1
        db_session.commit()
    with pytest.raises(ValueError, match="statistics_invalid"):
        search_lexical_index(db_session, kb.id, ["limits"], expected_identity=snapshot.identity)
