"""Durable BM25 candidates, publication and raw-source freshness checks."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import func, insert, select

from app.models import (Chunk, Document, DocumentVersion, IngestionBatch, KnowledgeBase, LexicalDocument,
    LexicalIndexJob, LexicalIndexState, LexicalPosting, LexicalTermRecord)
from app.retrieval_control_contracts import control_hash
from app.services.ingestion_resource_lock import active_ingestion_resource_lease
from app.services.lexical_index import (PROTOCOL, SCORING_PROTOCOL, TOKENIZER_PROTOCOL, BM25Snapshot,
    BM25Hit, IndexedDocument, LexicalSource, Posting, bm25_term_score, prepare_bm25_snapshot,
    query_terms, scoring_identity, tokenize_source, tokenizer_identity)
from app.services.storage import raise_if_source_io_cancelled

BATCH_SIZE = 1024
LEXICAL_PENDING_JOB_STATUSES = frozenset(
    {"prepared", "building", "ready_to_publish", "published", "cancel_requested"}
)


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _require_lease(db, kb_id):
    if active_ingestion_resource_lease(kb_id) is None:
        raise ValueError("lexical_mutation_resource_lock_required")
    kb = db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == kb_id).with_for_update())
    if kb is None or kb.lifecycle_status != "active":
        raise ValueError("lexical_knowledge_base_not_active")


def active_lexical_sources(db, knowledge_base_id, *, max_documents=None, max_characters=None):
    from app.core.config import get_settings

    settings = get_settings()
    max_documents = int(
        settings.lexical_index_max_documents if max_documents is None else max_documents
    )
    max_characters = int(
        settings.lexical_index_max_characters if max_characters is None else max_characters
    )
    if max_documents < 1 or max_characters < 1:
        raise ValueError("lexical_source_read_budget_invalid")
    rows = db.execute(select(Chunk.id, Chunk.document_version_id, Chunk.text, Chunk.char_start, Chunk.char_end)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active",
            Document.is_active.is_(True), DocumentVersion.is_active.is_(True),
            DocumentVersion.document_id == Chunk.document_id)
        .order_by(Chunk.id).execution_options(yield_per=BATCH_SIZE))
    result, characters = [], 0
    for row in rows:
        raise_if_source_io_cancelled()
        characters += len(row.text)
        if len(result) >= max_documents or characters > max_characters:
            raise ValueError("lexical_source_read_budget_exceeded")
        if row.char_end != row.char_start + len(row.text):
            raise ValueError("lexical_source_span_length_invalid")
        result.append(LexicalSource(row.id, row.document_version_id, row.text, row.char_start))
    return tuple(result)


def _verify_current_sources(snapshot, sources):
    if len(sources) != len(snapshot.documents):
        raise ValueError("lexical_source_scope_changed")
    documents = {item.chunk_id: item for item in snapshot.documents}
    for source in sources:
        expected = documents.get(source.chunk_id)
        if (expected is None or expected.document_version_id != source.document_version_id
                or expected.char_start != source.char_start or expected.char_end != source.char_start + len(source.text)
                or expected.raw_text_hash != hashlib.sha256(source.text.encode()).hexdigest()):
            raise ValueError("lexical_source_scope_changed")


def verify_snapshot(snapshot):
    documents = {d.chunk_id: d for d in snapshot.documents}
    if len(documents) != len(snapshot.documents):
        raise ValueError("lexical_snapshot_duplicate_document")
    frequencies, lengths, seen = Counter(), Counter(), set()
    for posting in snapshot.postings:
        document = documents.get(posting.chunk_id)
        key = (posting.term, posting.chunk_id)
        if (not posting.term or key in seen or document is None or not posting.positions
                or tuple(sorted(set(posting.positions))) != posting.positions
                or any(not document.char_start <= left < right <= document.char_end for left, right in posting.positions)):
            raise ValueError("lexical_snapshot_posting_invalid")
        seen.add(key)
        frequencies[posting.term] += 1
        lengths[posting.chunk_id] += posting.tf
    if (tuple(sorted(frequencies.items())) != snapshot.terms
            or any(lengths[d.chunk_id] != d.length for d in snapshot.documents)
            or sum(d.length for d in snapshot.documents) != snapshot.total_length):
        raise ValueError("lexical_snapshot_statistics_invalid")
    facts = [(d.chunk_id, d.document_version_id, d.char_start, d.char_end, d.raw_text_hash, d.length)
             for d in snapshot.documents]
    source_hash = control_hash({"kb": snapshot.knowledge_base_id, "documents": facts})
    stats_hash = control_hash({"N": snapshot.document_count, "total_length": snapshot.total_length, "df": snapshot.terms})
    postings_hash = control_hash([(p.term, p.chunk_id, p.positions) for p in snapshot.postings])
    expected_scoring_hash = scoring_identity(k1=snapshot.k1, b=snapshot.b)
    identity = control_hash({"protocol": PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
        "scoring_hash": expected_scoring_hash,
        "source_scope": source_hash, "tokenizer": snapshot.tokenizer_hash, "statistics": stats_hash, "postings": postings_hash})
    if (source_hash != snapshot.source_scope_hash or stats_hash != snapshot.statistics_hash
            or postings_hash != snapshot.postings_hash or identity != snapshot.identity
            or snapshot.tokenizer_hash != tokenizer_identity()
            or snapshot.scoring_hash != expected_scoring_hash):
        raise ValueError("lexical_snapshot_identity_changed")
    return snapshot


def _bulk_insert(db, model, records, *, check_job):
    for start in range(0, len(records), BATCH_SIZE):
        raise_if_source_io_cancelled()
        check_job()
        db.execute(insert(model), records[start:start + BATCH_SIZE])
    db.flush()


def _snapshot_matches_state(snapshot: BM25Snapshot, state: LexicalIndexState) -> bool:
    return (
        state.knowledge_base_id == snapshot.knowledge_base_id
        and state.protocol_version == PROTOCOL
        and state.tokenizer_protocol == TOKENIZER_PROTOCOL
        and state.tokenizer_hash == snapshot.tokenizer_hash
        and state.scoring_protocol == SCORING_PROTOCOL
        and state.scoring_hash == snapshot.scoring_hash
        and state.bm25_k1 == snapshot.k1
        and state.bm25_b == snapshot.b
        and state.source_scope_hash == snapshot.source_scope_hash
        and state.statistics_hash == snapshot.statistics_hash
        and state.postings_hash == snapshot.postings_hash
        and state.state_hash == snapshot.identity
        and state.document_count == snapshot.document_count
        and state.total_length == snapshot.total_length
        and state.term_count == len(snapshot.terms)
        and state.posting_count == len(snapshot.postings)
    )


def _job_diagnostics(job: LexicalIndexJob, *, phase: str, **details) -> dict:
    return {
        **dict(job.diagnostics_json or {}),
        "phase": phase,
        "updated_at": _now().isoformat(),
        **details,
    }


def prepare_lexical_index_job(
    db,
    snapshot: BM25Snapshot,
    *,
    ingestion_batch_id: str | None = None,
):
    """Freeze a durable candidate and before-state before materialization."""

    _require_lease(db, snapshot.knowledge_base_id)
    verify_snapshot(snapshot)
    _verify_current_sources(snapshot, active_lexical_sources(db, snapshot.knowledge_base_id))
    if ingestion_batch_id is not None:
        batch = db.get(IngestionBatch, ingestion_batch_id)
        if batch is None or batch.knowledge_base_id != snapshot.knowledge_base_id:
            raise ValueError("lexical_ingestion_batch_owner_changed")
    previous = db.scalar(
        select(LexicalIndexState)
        .where(
            LexicalIndexState.knowledge_base_id == snapshot.knowledge_base_id,
            LexicalIndexState.state == "active",
        )
        .with_for_update()
    )
    # Once the source/tokenizer/statistics identity changes, the old pointer is
    # retained for audit but is no longer an eligible active index.
    if previous is not None and previous.state_hash != snapshot.identity:
        previous.state = "stale"
        db.flush()
    state = LexicalIndexState(knowledge_base_id=snapshot.knowledge_base_id,
        predecessor_id=previous.id if previous else None, protocol_version=PROTOCOL,
        tokenizer_protocol=TOKENIZER_PROTOCOL, tokenizer_hash=snapshot.tokenizer_hash,
        scoring_protocol=SCORING_PROTOCOL, scoring_hash=snapshot.scoring_hash,
        bm25_k1=snapshot.k1, bm25_b=snapshot.b,
        source_scope_hash=snapshot.source_scope_hash,
        statistics_hash=snapshot.statistics_hash, postings_hash=snapshot.postings_hash, state_hash=snapshot.identity,
        document_count=snapshot.document_count, total_length=snapshot.total_length,
        term_count=len(snapshot.terms), posting_count=len(snapshot.postings), state="candidate")
    db.add(state)
    db.flush()
    job = LexicalIndexJob(knowledge_base_id=snapshot.knowledge_base_id, target_state_id=state.id,
        predecessor_state_id=previous.id if previous else None,
        ingestion_batch_id=ingestion_batch_id,
        source_scope_hash=snapshot.source_scope_hash, status="prepared",
        publish_intent=False,
        diagnostics_json={
            "phase": "prepared",
            "protocol_version": PROTOCOL,
            "target_state_hash": snapshot.identity,
            "document_count": snapshot.document_count,
            "posting_count": len(snapshot.postings),
            "prepared_at": _now().isoformat(),
        })
    db.add(job)
    db.commit()  # Durable intention precedes materialization and publication.
    return job


def _finish_cancelled_job(db, job: LexicalIndexJob, state: LexicalIndexState) -> LexicalIndexJob:
    if state.state == "candidate":
        state.state = "failed"
    job.status = "cancelled"
    job.error_code = "cancelled"
    job.completed_at = _now()
    job.diagnostics_json = _job_diagnostics(job, phase="cancelled")
    db.commit()
    return job


def materialize_lexical_index_job(
    db,
    job_id: str,
    *,
    snapshot: BM25Snapshot | None = None,
    check_cancelled=lambda: None,
):
    """Materialize or replay one prepared job without changing the active pointer."""

    job = db.scalar(select(LexicalIndexJob).where(LexicalIndexJob.id == job_id).with_for_update())
    if job is None:
        raise ValueError("lexical_job_missing")
    _require_lease(db, job.knowledge_base_id)
    state = db.get(LexicalIndexState, job.target_state_id)
    if state is None or state.knowledge_base_id != job.knowledge_base_id:
        raise ValueError("lexical_job_owner_changed")
    if job.status in {"ready_to_publish", "published", "completed"}:
        return job
    if job.status == "cancel_requested":
        return _finish_cancelled_job(db, job, state)
    if job.status not in {"prepared", "building"} or state.state != "candidate":
        raise ValueError("lexical_job_not_materializable")
    if snapshot is None:
        sources = active_lexical_sources(db, job.knowledge_base_id)
        snapshot = prepare_bm25_snapshot(
            job.knowledge_base_id,
            sources,
            check_cancelled=check_cancelled,
        )
    verify_snapshot(snapshot)
    _verify_current_sources(snapshot, active_lexical_sources(db, job.knowledge_base_id))
    if not _snapshot_matches_state(snapshot, state) or job.source_scope_hash != snapshot.source_scope_hash:
        _fail_candidate_after_source_drift(
            db,
            job=job,
            state=state,
            error_code="lexical_prepared_snapshot_changed",
        )
        raise ValueError("lexical_prepared_snapshot_changed")
    state_id = state.id
    job.status = "building"
    job.diagnostics_json = _job_diagnostics(job, phase="building", started_at=_now().isoformat())
    db.commit()

    def check_job():
        import asyncio
        check_cancelled()
        current = db.scalar(select(LexicalIndexJob.status).where(LexicalIndexJob.id == job.id))
        if current == "cancel_requested":
            raise asyncio.CancelledError("lexical_build_cancelled")
        if current != "building":
            raise ValueError("lexical_job_state_changed")
    try:
        check_job()
        _bulk_insert(db, LexicalDocument, [dict(index_state_id=state.id, chunk_id=d.chunk_id,
            knowledge_base_id=snapshot.knowledge_base_id, document_version_id=d.document_version_id,
            char_start=d.char_start, char_end=d.char_end, raw_text_hash=d.raw_text_hash, token_length=d.length)
            for d in snapshot.documents], check_job=check_job)
        _bulk_insert(db, LexicalTermRecord, [dict(index_state_id=state.id,
            term_key=hashlib.sha256(term.encode()).hexdigest(), term=term, document_frequency=df) for term, df in snapshot.terms], check_job=check_job)
        _bulk_insert(db, LexicalPosting, [dict(index_state_id=state.id, chunk_id=p.chunk_id,
            term_key=hashlib.sha256(p.term.encode()).hexdigest(), term_frequency=p.tf, positions_json=[list(x) for x in p.positions])
            for p in snapshot.postings], check_job=check_job)
        check_job()
        stored = load_lexical_snapshot(db, state.id, verify_sources=True)
        if stored != snapshot:
            raise ValueError("lexical_materialization_changed")
        job.status = "ready_to_publish"
        job.completed_documents, job.completed_postings = len(snapshot.documents), len(snapshot.postings)
        job.diagnostics_json = _job_diagnostics(
            job,
            phase="ready_to_publish",
            materialized_at=_now().isoformat(),
        )
        db.commit()
    except BaseException as exc:
        import asyncio
        from app.services.cancellation import IngestionCancelled
        db.rollback()
        state, job = db.get(LexicalIndexState, state_id), db.get(LexicalIndexJob, job_id)
        if state is None or job is None:
            raise RuntimeError("lexical_failure_audit_scope_missing") from exc
        if state.state == "candidate":
            state.state = "failed"
        cancelled = isinstance(exc, (asyncio.CancelledError, IngestionCancelled))
        job.status = "cancelled" if cancelled else "failed"
        job.error_code = "cancelled" if job.status == "cancelled" else "lexical_materialization_failed"
        job.completed_at = _now()
        job.diagnostics_json = _job_diagnostics(
            job,
            phase=job.status,
            failure_type=exc.__class__.__name__,
        )
        db.commit()
        raise
    return job


def stage_lexical_index(
    db,
    snapshot: BM25Snapshot,
    *,
    ingestion_batch_id: str | None = None,
    check_cancelled=lambda: None,
):
    """Compatibility composition for prepare + materialize; never publishes."""

    job = prepare_lexical_index_job(
        db,
        snapshot,
        ingestion_batch_id=ingestion_batch_id,
    )
    return materialize_lexical_index_job(
        db,
        job.id,
        snapshot=snapshot,
        check_cancelled=check_cancelled,
    )


def load_lexical_snapshot(db, index_id, *, verify_sources=False):
    state = db.get(LexicalIndexState, index_id)
    if state is None or state.protocol_version != PROTOCOL or state.scoring_protocol != SCORING_PROTOCOL or state.tokenizer_protocol != TOKENIZER_PROTOCOL:
        raise ValueError("lexical_index_protocol_invalid")
    documents = tuple(IndexedDocument(d.chunk_id, d.document_version_id, d.char_start, d.char_end, d.raw_text_hash, d.token_length)
        for d in db.scalars(select(LexicalDocument).where(LexicalDocument.index_state_id == index_id).order_by(LexicalDocument.chunk_id)))
    term_rows = list(db.scalars(select(LexicalTermRecord)
        .where(LexicalTermRecord.index_state_id == index_id).order_by(LexicalTermRecord.term_key)))
    terms = tuple(sorted((t.term, t.document_frequency) for t in term_rows))
    term_by_key = {row.term_key: row.term for row in term_rows}
    rows = db.execute(
        select(
            LexicalPosting.term_key,
            LexicalPosting.chunk_id,
            LexicalPosting.term_frequency,
            LexicalPosting.positions_json,
        )
        .where(LexicalPosting.index_state_id == index_id)
        .execution_options(yield_per=BATCH_SIZE)
    )
    postings = []
    for term_key, chunk_id, term_frequency, positions_json in rows:
        raise_if_source_io_cancelled()
        term = term_by_key.get(term_key)
        if (
            term is None
            or term_frequency != len(positions_json)
            or hashlib.sha256(term.encode()).hexdigest() != term_key
        ):
            raise ValueError("lexical_persisted_posting_invalid")
        postings.append(
            Posting(term, chunk_id, tuple(tuple(position) for position in positions_json))
        )
    if (len(documents), len(terms), len(postings)) != (state.document_count, state.term_count, state.posting_count):
        raise ValueError("lexical_persisted_inventory_changed")
    snapshot = BM25Snapshot(state.knowledge_base_id, state.source_scope_hash, state.tokenizer_hash,
        state.scoring_hash, state.bm25_k1, state.bm25_b, documents,
        terms, tuple(sorted(postings, key=lambda p: (p.term, p.chunk_id))), state.total_length, state.statistics_hash, state.postings_hash, state.state_hash)
    verify_snapshot(snapshot)
    if verify_sources:
        _verify_current_sources(snapshot, active_lexical_sources(db, state.knowledge_base_id))
    return snapshot


def request_lexical_index_cancel(db, job_id: str) -> LexicalIndexJob:
    """Persist cancellation before publication; an active pointer is never rolled back."""

    job = db.scalar(select(LexicalIndexJob).where(LexicalIndexJob.id == job_id).with_for_update())
    if job is None:
        raise ValueError("lexical_job_missing")
    _require_lease(db, job.knowledge_base_id)
    state = db.get(LexicalIndexState, job.target_state_id)
    if state is None or state.knowledge_base_id != job.knowledge_base_id:
        raise ValueError("lexical_job_owner_changed")
    if job.status in {"completed", "failed", "cancelled"}:
        return job
    if job.status == "published" or state.state == "active":
        raise ValueError("lexical_job_already_published")
    if job.status == "building":
        job.status = "cancel_requested"
        job.error_code = "cancel_requested"
        job.diagnostics_json = _job_diagnostics(
            job,
            phase="cancel_requested",
            cancel_requested_at=_now().isoformat(),
        )
        db.commit()
        return job
    return _finish_cancelled_job(db, job, state)


def mark_active_lexical_index_stale(
    db,
    knowledge_base_id: str,
    *,
    reason: str,
) -> dict:
    """Remove an invalid source snapshot from active eligibility in the caller transaction."""

    _require_lease(db, knowledge_base_id)
    active = db.scalar(
        select(LexicalIndexState)
        .where(
            LexicalIndexState.knowledge_base_id == knowledge_base_id,
            LexicalIndexState.state == "active",
        )
        .with_for_update()
    )
    if active is None:
        return {"stale_state_id": None, "reason": reason}
    active.state = "stale"
    owning_job = db.scalar(
        select(LexicalIndexJob).where(LexicalIndexJob.target_state_id == active.id).with_for_update()
    )
    if owning_job is not None:
        owning_job.diagnostics_json = _job_diagnostics(
            owning_job,
            phase="stale",
            stale_reason=reason,
            stale_at=_now().isoformat(),
        )
    db.flush()
    return {"stale_state_id": active.id, "reason": reason}


def _fail_candidate_after_source_drift(
    db,
    *,
    job: LexicalIndexJob,
    state: LexicalIndexState,
    error_code: str,
) -> None:
    active = db.scalar(
        select(LexicalIndexState)
        .where(
            LexicalIndexState.knowledge_base_id == job.knowledge_base_id,
            LexicalIndexState.state == "active",
        )
        .with_for_update()
    )
    if active is not None:
        active.state = "stale"
    if state.state == "candidate":
        state.state = "failed"
    job.status = "failed"
    job.error_code = error_code
    job.completed_at = _now()
    job.diagnostics_json = _job_diagnostics(
        job,
        phase="failed",
        failure_type=error_code,
    )
    db.commit()


def publish_lexical_index(db, job_id, *, invalidate):
    """Atomic pointer change followed by retryable cache publication."""
    job = db.scalar(select(LexicalIndexJob).where(LexicalIndexJob.id == job_id).with_for_update())
    if job is None:
        raise ValueError("lexical_job_missing")
    _require_lease(db, job.knowledge_base_id)
    state = db.get(LexicalIndexState, job.target_state_id)
    if (state is None or state.knowledge_base_id != job.knowledge_base_id
            or state.source_scope_hash != job.source_scope_hash):
        raise ValueError("lexical_job_owner_changed")
    if job.status not in {"ready_to_publish", "published", "completed"}:
        raise ValueError("lexical_job_not_publishable")
    try:
        load_lexical_snapshot(db, state.id, verify_sources=True)
    except ValueError as exc:
        if "source_" in str(exc) or "snapshot_identity" in str(exc):
            _fail_candidate_after_source_drift(
                db,
                job=job,
                state=state,
                error_code="lexical_source_scope_changed",
            )
        raise
    active = db.scalar(select(LexicalIndexState).where(LexicalIndexState.knowledge_base_id == job.knowledge_base_id,
        LexicalIndexState.state == "active").with_for_update())
    if job.status == "ready_to_publish":
        predecessor = db.get(LexicalIndexState, state.predecessor_id) if state.predecessor_id else None
        predecessor_matches = (
            (active.id if active else None) == state.predecessor_id
            or (
                active is None
                and predecessor is not None
                and predecessor.state == "stale"
                and predecessor.knowledge_base_id == job.knowledge_base_id
            )
            or (active is None and state.predecessor_id is None)
        )
        if (
            not predecessor_matches
            or job.predecessor_state_id != state.predecessor_id
            or state.state != "candidate"
        ):
            raise ValueError("lexical_predecessor_changed")
        if active:
            active.state = "stale"
            db.flush()
        state.state, state.published_at = "active", _now()
        job.status, job.publish_intent, job.cache_invalidation_pending = "published", True, True
        job.diagnostics_json = _job_diagnostics(
            job,
            phase="published",
            published_at=state.published_at.isoformat(),
        )
        db.commit()
    elif active is None or active.id != state.id:
        raise ValueError("lexical_published_pointer_changed")
    if job.status == "completed":
        if job.cache_invalidation_pending:
            raise ValueError("lexical_completed_job_pending_cache")
        return state
    raise_if_source_io_cancelled()
    if invalidate(job.knowledge_base_id, strict=True) is not True:
        raise ValueError("lexical_cache_publication_failed")
    job.cache_invalidation_pending, job.status, job.completed_at = False, "completed", _now()
    job.diagnostics_json = _job_diagnostics(
        job,
        phase="completed",
        cache_invalidated_at=job.completed_at.isoformat(),
    )
    db.commit()
    return state


def build_and_publish_lexical_index(
    db,
    knowledge_base_id: str,
    *,
    invalidate,
    ingestion_batch_id: str | None = None,
    check_cancelled=lambda: None,
) -> dict:
    """Build the current source snapshot under the existing KB resource lease."""

    _require_lease(db, knowledge_base_id)
    check_cancelled()
    from app.core.config import get_settings

    settings = get_settings()
    sources = active_lexical_sources(db, knowledge_base_id)
    snapshot = prepare_bm25_snapshot(
        knowledge_base_id,
        sources,
        max_documents=settings.lexical_index_max_documents,
        max_postings=settings.lexical_index_max_postings,
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        check_cancelled=check_cancelled,
    )
    active = db.scalar(
        select(LexicalIndexState).where(
            LexicalIndexState.knowledge_base_id == knowledge_base_id,
            LexicalIndexState.state == "active",
        )
    )
    if active is not None and active.state_hash == snapshot.identity:
        load_lexical_snapshot(db, active.id, verify_sources=True)
        pending_publication = db.scalar(
            select(LexicalIndexJob).where(
                LexicalIndexJob.target_state_id == active.id,
                LexicalIndexJob.status == "published",
                LexicalIndexJob.cache_invalidation_pending.is_(True),
            )
        )
        if pending_publication is not None:
            publish_lexical_index(
                db,
                pending_publication.id,
                invalidate=invalidate,
            )
        return {
            "status": "reused",
            "index_state_id": active.id,
            "index_identity": active.state_hash,
            "job_id": None,
            "document_count": active.document_count,
            "posting_count": active.posting_count,
        }
    job = stage_lexical_index(
        db,
        snapshot,
        ingestion_batch_id=ingestion_batch_id,
        check_cancelled=check_cancelled,
    )
    check_cancelled()
    state = publish_lexical_index(db, job.id, invalidate=invalidate)
    return {
        "status": "completed",
        "index_state_id": state.id,
        "index_identity": state.state_hash,
        "job_id": job.id,
        "document_count": state.document_count,
        "posting_count": state.posting_count,
    }


async def reconcile_pending_lexical_index_jobs(
    *,
    invalidate,
    knowledge_base_id: str | None = None,
) -> dict:
    """Replay durable lexical work after a worker/API interruption."""

    from app.db import SessionLocal
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    with SessionLocal() as inventory_db:
        query = select(LexicalIndexJob.id).where(
            LexicalIndexJob.status.in_(LEXICAL_PENDING_JOB_STATUSES)
        )
        if knowledge_base_id is not None:
            query = query.where(LexicalIndexJob.knowledge_base_id == knowledge_base_id)
        job_ids = list(inventory_db.scalars(query.order_by(LexicalIndexJob.created_at, LexicalIndexJob.id)))

    completed: list[str] = []
    cancelled: list[str] = []
    failed: list[dict[str, str]] = []
    for job_id in job_ids:
        with SessionLocal() as db:
            job = db.get(LexicalIndexJob, job_id)
            if job is None or job.status not in LEXICAL_PENDING_JOB_STATUSES:
                continue
            try:
                async with knowledge_base_ingestion_resource_lock(
                    db,
                    job.knowledge_base_id,
                    operation="lexical_index_reconcile",
                    batch_id=job.ingestion_batch_id,
                ):
                    job = db.get(LexicalIndexJob, job_id, populate_existing=True)
                    if job is None:
                        continue
                    state = db.get(LexicalIndexState, job.target_state_id)
                    if state is None:
                        raise ValueError("lexical_job_target_missing")
                    batch = (
                        db.get(IngestionBatch, job.ingestion_batch_id)
                        if job.ingestion_batch_id is not None
                        else None
                    )
                    batch_cancelled = batch is not None and (
                        batch.status in {"cancel_requested", "cancelling", "compensating", "cancelled", "cancel_failed"}
                        or bool((batch.stats or {}).get("cancel_requested"))
                    )
                    if batch_cancelled and job.status not in {"published", "completed"}:
                        _finish_cancelled_job(db, job, state)
                        cancelled.append(job_id)
                        continue
                    if job.status == "cancel_requested":
                        _finish_cancelled_job(db, job, state)
                        cancelled.append(job_id)
                        continue
                    if job.status in {"prepared", "building"}:
                        job = materialize_lexical_index_job(db, job_id)
                    if job.status in {"ready_to_publish", "published"}:
                        publish_lexical_index(db, job_id, invalidate=invalidate)
                    completed.append(job_id)
            except BaseException as exc:
                import asyncio
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                failed.append({"job_id": job_id, "error_type": exc.__class__.__name__})
    return {
        "attempted": len(job_ids),
        "completed_job_ids": completed,
        "cancelled_job_ids": cancelled,
        "failed": failed,
        "ok": not failed,
    }


async def rebuild_lexical_index_for_knowledge_base(
    knowledge_base_id: str,
    *,
    ingestion_batch_id: str | None = None,
    operation: str = "lexical_index_build",
) -> dict:
    """Acquire the shared KB fence and run bounded, cooperatively cancellable work."""

    import asyncio

    from app.db import SessionLocal
    from app.services.cancellation import IngestionCancelled, ensure_not_cancelled
    from app.services.context_graph import invalidate_context_graph_cache_after_commit
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
    from app.services.storage import run_bounded_source_io

    with SessionLocal() as db:
        async with knowledge_base_ingestion_resource_lock(
            db,
            knowledge_base_id,
            operation=operation,
            batch_id=ingestion_batch_id,
        ):
            def check_cancelled() -> None:
                if ingestion_batch_id is not None:
                    ensure_not_cancelled(db, ingestion_batch_id)

            try:
                return await run_bounded_source_io(
                    build_and_publish_lexical_index,
                    db,
                    knowledge_base_id,
                    invalidate=invalidate_context_graph_cache_after_commit,
                    ingestion_batch_id=ingestion_batch_id,
                    check_cancelled=check_cancelled,
                )
            except asyncio.CancelledError as exc:
                raise IngestionCancelled("lexical index build cancelled") from exc


@dataclass(frozen=True)
class LexicalSearchResult:
    index_state_id: str
    index_identity: str
    statistics_hash: str
    hits: tuple[BM25Hit, ...]
    matched_documents: int
    postings_read: int
    output_truncated: bool


def search_lexical_index_many(
    db,
    knowledge_base_id,
    lexical_term_sets: Sequence[Sequence[str]],
    *,
    expected_identity,
    eligible_chunk_ids=None,
    limit=64,
    posting_budget=1000000,
):
    """Score several lexical views with one frozen-source and posting read.

    Every view retains the same corpus statistics and produces the same result as
    an independent search.  The batching only shares deterministic integrity
    checks and a union posting scan; it does not merge query scores or domains.
    """
    if not lexical_term_sets:
        return ()
    term_sets = tuple(frozenset(query_terms(items)) for items in lexical_term_sets)
    terms = frozenset().union(*term_sets)
    if type(limit) is not int or not 1 <= limit <= 4096 or type(posting_budget) is not int or posting_budget < 1:
        raise ValueError("lexical_query_budget_invalid")
    state = db.scalar(select(LexicalIndexState).where(LexicalIndexState.knowledge_base_id == knowledge_base_id,
        LexicalIndexState.state == "active"))
    if state is None:
        raise ValueError("lexical_index_unavailable")
    from app.core.config import get_settings
    settings = get_settings()
    current_scoring_hash = scoring_identity(k1=settings.bm25_k1, b=settings.bm25_b)
    if (state.state_hash != expected_identity or state.protocol_version != PROTOCOL
            or state.scoring_protocol != SCORING_PROTOCOL
            or state.scoring_hash != current_scoring_hash
            or state.tokenizer_hash != tokenizer_identity()):
        raise ValueError("lexical_index_identity_changed")
    source_rows = active_lexical_sources(db, knowledge_base_id)
    source_by_id = {source.chunk_id: source for source in source_rows}
    documents = list(db.scalars(select(LexicalDocument).where(LexicalDocument.index_state_id == state.id)
        .order_by(LexicalDocument.chunk_id)))
    if len(documents) != state.document_count or set(source_by_id) != {d.chunk_id for d in documents}:
        raise ValueError("lexical_source_scope_changed")
    indexed = {d.chunk_id: d for d in documents}
    facts = []
    for document in documents:
        source = source_by_id[document.chunk_id]
        if (document.knowledge_base_id != knowledge_base_id or document.document_version_id != source.document_version_id
                or document.char_start != source.char_start or document.char_end != source.char_start + len(source.text)
                or document.raw_text_hash != hashlib.sha256(source.text.encode()).hexdigest()):
            raise ValueError("lexical_source_scope_changed")
        facts.append((document.chunk_id, document.document_version_id, document.char_start, document.char_end,
            document.raw_text_hash, document.token_length))
    if (control_hash({"kb": knowledge_base_id, "documents": facts}) != state.source_scope_hash
            or sum(d.token_length for d in documents) != state.total_length):
        raise ValueError("lexical_source_inventory_changed")
    if eligible_chunk_ids is not None and not set(eligible_chunk_ids) <= set(indexed):
        raise ValueError("lexical_filter_outside_snapshot")
    keys = {hashlib.sha256(term.encode()).hexdigest(): term for term in terms}
    matched_terms = list(db.scalars(select(LexicalTermRecord).where(LexicalTermRecord.index_state_id == state.id,
        LexicalTermRecord.term_key.in_(keys))))
    term_by_key = {term.term_key: term for term in matched_terms}
    if any(term.term != keys[term.term_key] for term in matched_terms):
        raise ValueError("lexical_query_term_identity_changed")
    actual_df = dict(db.execute(select(LexicalPosting.term_key, func.count())
        .where(LexicalPosting.index_state_id == state.id, LexicalPosting.term_key.in_(term_by_key))
        .group_by(LexicalPosting.term_key)).all())
    if any(term.document_frequency != actual_df.get(key, 0) for key, term in term_by_key.items()):
        raise ValueError("lexical_persisted_statistics_invalid")
    query = select(LexicalPosting).where(LexicalPosting.index_state_id == state.id,
        LexicalPosting.term_key.in_(term_by_key)).order_by(LexicalPosting.term_key, LexicalPosting.chunk_id)
    if eligible_chunk_ids is not None:
        query = query.where(LexicalPosting.chunk_id.in_(sorted(eligible_chunk_ids)))
    contributions = [dict() for _ in term_sets]
    witnesses = [dict() for _ in term_sets]
    posting_counts = [0 for _ in term_sets]
    token_positions, count = {}, 0
    for posting in db.scalars(query.execution_options(yield_per=BATCH_SIZE)):
        raise_if_source_io_cancelled()
        count += 1
        if count > posting_budget:
            raise ValueError("lexical_posting_scan_budget_exhausted")
        term, document = term_by_key[posting.term_key], indexed[posting.chunk_id]
        positions = tuple(tuple(p) for p in posting.positions_json)
        source = source_by_id[posting.chunk_id]
        if posting.term_frequency != len(positions) or tuple(sorted(set(positions))) != positions:
            raise ValueError("lexical_persisted_posting_invalid")
        if posting.chunk_id not in token_positions:
            raw_positions = {}
            for token in tokenize_source(source.text):
                if token.term in terms:
                    raw_positions.setdefault(token.term, []).append((source.char_start + token.start, source.char_start + token.end))
            token_positions[posting.chunk_id] = raw_positions
        if positions != tuple(token_positions[posting.chunk_id].get(term.term, ())):
            raise ValueError("lexical_posting_tokenization_changed")
        for left, right in positions:
            if (not document.char_start <= left < right <= document.char_end
                    or unicodedata.normalize("NFKC", source.text[left-source.char_start:right-source.char_start]).casefold() != term.term):
                raise ValueError("lexical_posting_raw_witness_changed")
        value = bm25_term_score(tf=posting.term_frequency, length=document.token_length,
            document_count=state.document_count, document_frequency=term.document_frequency,
            average_length=state.total_length / state.document_count,
            k1=state.bm25_k1, b=state.bm25_b)
        witness = Posting(term.term, posting.chunk_id, positions)
        for index, query_term_set in enumerate(term_sets):
            if term.term not in query_term_set:
                continue
            posting_counts[index] += 1
            contributions[index].setdefault(posting.chunk_id, []).append(value)
            witnesses[index].setdefault(posting.chunk_id, []).append(witness)
    results = []
    for index in range(len(term_sets)):
        hits = tuple(sorted((BM25Hit(key, math.fsum(values), tuple(sorted(witnesses[index][key], key=lambda p: p.term)))
            for key, values in contributions[index].items()), key=lambda hit: (-hit.score, hit.chunk_id)))
        results.append(LexicalSearchResult(
            state.id,
            state.state_hash,
            state.statistics_hash,
            hits[:limit],
            len(hits),
            posting_counts[index],
            len(hits) > limit,
        ))
    return tuple(results)


def search_lexical_index(db, knowledge_base_id, lexical_terms, *, expected_identity,
                         eligible_chunk_ids=None, limit=64, posting_budget=1000000):
    """Read matched postings from the frozen active index; filters never alter IDF."""
    return search_lexical_index_many(
        db,
        knowledge_base_id,
        (lexical_terms,),
        expected_identity=expected_identity,
        eligible_chunk_ids=eligible_chunk_ids,
        limit=limit,
        posting_budget=posting_budget,
    )[0]
