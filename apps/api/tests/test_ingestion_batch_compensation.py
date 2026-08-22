from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select


def test_versioned_graph_retry_completion_rebinds_cache_and_clears_stale_progress():
    from types import SimpleNamespace

    from app.services import ingestion

    cache_state = ingestion._versioned_graph_commit_cache_invalidation_state(
        knowledge_base_id="kb-1",
        attempt_id="attempt-1",
        plan_hash="a" * 64,
        context_graph_state_id="context-2",
        graph_write_set_hash="b" * 64,
    )
    assert cache_state["status"] == "pending"
    assert cache_state["attempt_count"] == 0
    assert cache_state["reason"] == "versioned_graph_commit"
    assert cache_state["graph_commit_binding"] == {
        "protocol_version": "versioned_graph_commit_cache_binding_v1",
        "attempt_id": "attempt-1",
        "plan_hash": "a" * 64,
        "context_graph_state_id": "context-2",
        "graph_write_set_hash": "b" * 64,
    }
    assert len(cache_state["graph_commit_binding_hash"]) == 64

    stats = ingestion._completed_versioned_graph_retry_stats(
        {
            "phase": "context_graph",
            "context_graph_phase": "mid_concepts",
            "context_graph_heartbeat_at": "stale",
            "context_graph_metrics": {"completed_llm_batches": 38},
            "errors": [{"phase": "context_graph", "message": "old"}],
            "versioned_graph_retry_failure": {"plan_hash": "old"},
            "failure_compensation": {"restored": True},
            "failed_at": "stale",
        },
        context_state=SimpleNamespace(
            id="context-2",
            stats_json={"mid_concepts": 173, "coarse_concepts": 35},
        ),
        graph_write_set_hash="b" * 64,
        attempt_id="attempt-1",
        plan_hash="a" * 64,
    )
    assert stats["phase"] == "completed"
    assert stats["graph_stats"] == {"mid_concepts": 173, "coarse_concepts": 35}
    assert stats["versioned_graph_retry_completed"]["gray_zone_model_call_count"] == 0
    assert stats["ingestion_error_history"] == [
        {"phase": "context_graph", "message": "old"}
    ]
    assert stats["versioned_graph_retry_failure_history"] == [
        {"plan_hash": "old"}
    ]
    for stale_key in (
        "errors",
        "context_graph_phase",
        "context_graph_heartbeat_at",
        "context_graph_metrics",
        "versioned_graph_retry_failure",
        "failure_compensation",
        "failed_at",
    ):
        assert stale_key not in stats


def test_versioned_graph_retry_post_commit_failure_cannot_enter_compensation():
    source = (
        Path(__file__).parents[1]
        / "app"
        / "services"
        / "ingestion.py"
    ).read_text(encoding="utf-8")
    run = source.split("async def run_versioned_graph_retry", 1)[1].split(
        "async def run_uploaded_files_ingestion",
        1,
    )[0]

    commit_boundary = "graph_commit_completed = True"
    dispatch_boundary = "cache_state = _dispatch_batch_recovery_cache_invalidation("
    post_commit_guard = "if graph_commit_completed:"
    lifecycle_guard = "if not lifecycle_started:"
    assert run.index(commit_boundary) < run.index(dispatch_boundary)
    except_branch = run.split("except (Exception, asyncio.CancelledError) as exc:", 1)[1]
    assert except_branch.index(post_commit_guard) < except_branch.index(lifecycle_guard)
    post_commit_branch = except_branch.split(post_commit_guard, 1)[1].split(
        lifecycle_guard,
        1,
    )[0]
    assert "graph_commit_preserved=True" in post_commit_branch
    assert "raise" in post_commit_branch
    assert "_restore_graph_before_scope" not in post_commit_branch
    assert 'batch.status = "failed"' not in post_commit_branch


def test_graph_build_failure_audit_persists_only_bounded_scalar_evidence():
    import json

    from app.services import context_graph, ingestion
    from app.services.error_sanitizer import ExternalServiceError

    provider_error = ExternalServiceError(
        service="model_provider",
        phase="anthropic_messages",
        status_code=503,
        error_code="upstream_unavailable",
        retryable=True,
    )
    batch_error = context_graph.ConceptProviderBatchError(
        layer="mid",
        batch_index=50,
        packet_ids=["0123456789abcdef"],
    )
    batch_error.__cause__ = provider_error

    audit = ingestion.graph_build_failure_audit(batch_error)

    assert audit["protocol_version"] == "graph_build_failure_audit_v1"
    assert audit["provider_response_persisted"] is False
    assert audit["external_failure"] == {
        "protocol_version": "external_failure_classification_v1",
        "classified": True,
        "classification_source": "external_service_error",
        "outer_error_type": "ConceptProviderBatchError",
        "classified_error_type": "ExternalServiceError",
        "cause_depth": 1,
        "service": "model_provider",
        "phase": "anthropic_messages",
        "http_status": 503,
        "error_code": "upstream_unavailable",
        "retryable": True,
    }
    assert audit["concept_provider_failure"]["batch"] == {
        "layer": "mid",
        "batch_index": 50,
        "packet_ids": ["0123456789abcdef"],
    }
    serialized = json.dumps(audit, sort_keys=True)
    assert "authorization" not in serialized.lower()
    assert "provider_response" not in serialized.replace(
        '"provider_response_persisted": false', ""
    )


def _kb_source(knowledge_base, filename: str) -> Path:
    from app.core.config import get_settings

    root = get_settings().knowledge_base_paths_for_name(knowledge_base.name)["storage_root"]
    root.mkdir(parents=True, exist_ok=True)
    return (root / filename).resolve()


async def _seed_document(db, knowledge_base, source: Path) -> dict:
    from app.models import Chunk, DocumentVersion
    from app.services import ingestion

    source.write_text(
        "# Stable baseline\n\nThe exact pre-batch active version must survive cancellation.\n",
        encoding="utf-8",
    )
    result = await ingestion.ingest_file(
        db,
        source,
        knowledge_base_id=knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )
    db.refresh(knowledge_base)
    return {
        "document_id": result["document_id"],
        "chunk_ids": sorted(
            db.scalars(
                select(Chunk.id).where(
                    Chunk.document_id == result["document_id"],
                    Chunk.state == "active",
                )
            ).all()
        ),
        "version_ids": sorted(
            db.scalars(
                select(DocumentVersion.id).where(
                    DocumentVersion.document_id == result["document_id"],
                    DocumentVersion.is_active.is_(True),
                )
            ).all()
        ),
        "kb_version": int(knowledge_base.current_chunk_version or 0),
    }


def _install_candidate_point_compensation(
    monkeypatch,
    fake_model_stack,
    *,
    fail_first_delete: bool = False,
):
    from app.services import ingestion

    fake_store = fake_model_stack["VectorStore"]

    class CompensationVectorStore(fake_store):
        remaining_failures = 1 if fail_first_delete else 0

        def delete_if_payload_matches(self, expected_points):
            if type(self).remaining_failures > 0:
                type(self).remaining_failures -= 1
                raise RuntimeError("synthetic qdrant delete timeout")
            for expected in expected_points:
                point_id = str(expected["id"])
                current = self.points.get(point_id)
                if current is None:
                    continue
                expected_owner = (expected.get("payload") or {}).get(
                    "qdrant_write_intent_id"
                )
                current_owner = (current.get("payload") or {}).get(
                    "qdrant_write_intent_id"
                )
                if expected_owner and current_owner == expected_owner:
                    self.points.pop(point_id, None)

    monkeypatch.setattr(ingestion, "VectorStore", CompensationVectorStore)
    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        lambda _knowledge_base_id, *, strict=False: True,
    )
    return CompensationVectorStore


@pytest.mark.asyncio
async def test_parse_phase_cancel_restores_exact_batch_before_scope(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import (
        Chunk,
        DocumentVersion,
        IngestionBatchRecovery,
        IngestionFileStage,
        IngestionCompensationLog,
        VectorRecord,
    )
    from app.services import ingestion
    from app.services.cancellation import IngestionCancelled

    source = _kb_source(sample_knowledge_base, "cancel-before-graph.md")
    before = await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# Candidate v2\n\nThis committed file must be compensated when cancellation is seen before graph entry.\n",
        encoding="utf-8",
    )
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )
    _install_candidate_point_compensation(monkeypatch, fake_model_stack)
    calls = 0

    def cancel_after_first_file(_db, _batch_id):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise IngestionCancelled("cancel after the first committed file")

    monkeypatch.setattr(ingestion, "ensure_not_cancelled", cancel_after_first_file)
    result = await ingestion.run_uploaded_files_ingestion(
        batch.id,
        [str(source)],
        full_reparse=True,
    )

    db_session.expire_all()
    assert result["state"] == "cancelled"
    assert result["cancellation_status"] == "parse_writes_compensated"
    recovery = db_session.scalar(
        select(IngestionBatchRecovery).where(IngestionBatchRecovery.batch_id == batch.id)
    )
    stage = db_session.scalar(
        select(IngestionFileStage).where(
            IngestionFileStage.batch_recovery_id == recovery.id
        )
    )
    assert recovery.protocol_version == ingestion.INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION
    assert recovery.v_before_batch == before["kb_version"] == 1
    assert recovery.target_version == 2
    assert recovery.parse_committed is False
    assert recovery.status == "parse_compensated"
    assert recovery.compensation_json["verification"]["ok"] is True
    assert recovery.compensation_json["verification"]["version_minus_one_inference_used"] is False
    assert stage.status == "compensated"
    assert stage.write_set_json["protocol_version"] == ingestion.INGESTION_FILE_WRITE_SET_PROTOCOL_VERSION
    assert stage.compensation_json["version_restore_source"] == "file_before_state"
    delete_intent_ids = list(stage.compensation_json["qdrant_delete_intent_ids"])
    assert delete_intent_ids
    assert all(
        db_session.get(IngestionCompensationLog, intent_id).status == "committed"
        for intent_id in delete_intent_ids
    )
    assert sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before["document_id"],
                Chunk.state == "active",
            )
        ).all()
    ) == before["chunk_ids"]
    assert sorted(
        db_session.scalars(
            select(DocumentVersion.id).where(
                DocumentVersion.document_id == before["document_id"],
                DocumentVersion.is_active.is_(True),
            )
        ).all()
    ) == before["version_ids"]
    assert db_session.get(type(sample_knowledge_base), sample_knowledge_base.id).current_chunk_version == 1
    candidate_chunk_ids = list(stage.write_set_json["chunk_ids"])
    assert candidate_chunk_ids
    assert not list(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.id.in_(candidate_chunk_ids),
                Chunk.state == "active",
            )
        ).all()
    )
    assert all(
        status == "compensated"
        for status in db_session.scalars(
            select(VectorRecord.vector_status).where(
                VectorRecord.chunk_id.in_(candidate_chunk_ids)
            )
        ).all()
    )


@pytest.mark.asyncio
async def test_graph_phase_cancel_preserves_committed_parse_scope(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import (
        Chunk,
        ChunkStructureNode,
        CoarseConceptState,
        ContextGraphState,
        IngestionBatchRecovery,
        IngestionFileStage,
        MidConceptState,
        VectorRecord,
    )
    from app.services import ingestion
    from app.services.cancellation import IngestionCancelled

    source = _kb_source(sample_knowledge_base, "cancel-in-concepts.md")
    before = await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# Candidate same-version parse\n\nOnce graph construction begins this parse scope must remain committed.\n",
        encoding="utf-8",
    )
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
    )
    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        lambda _knowledge_base_id, *, strict=False: True,
    )

    candidate_graph_ids: dict[str, str] = {}

    async def cancel_graph(graph_db, knowledge_base_id, *_args, **_kwargs):
        mid = MidConceptState(
            knowledge_base_id=knowledge_base_id,
            state_hash="1" * 64,
            grounding_hash="2" * 64,
            state="active",
        )
        graph_db.add(mid)
        graph_db.flush()
        coarse = CoarseConceptState(
            knowledge_base_id=knowledge_base_id,
            mid_concept_state_id=mid.id,
            state_hash="3" * 64,
            grounding_hash="4" * 64,
            state="active",
        )
        graph_db.add(coarse)
        graph_db.flush()
        context = ContextGraphState(
            knowledge_base_id=knowledge_base_id,
            mid_concept_state_id=mid.id,
            coarse_concept_state_id=coarse.id,
            chunk_scope_hash="5" * 64,
            structure_graph_hash="6" * 64,
            chunk_relation_graph_hash="7" * 64,
            rq_membership_hash="8" * 64,
            mid_concept_hash="9" * 64,
            coarse_concept_hash="a" * 64,
            context_graph_hash="b" * 64,
            state="active",
        )
        graph_db.add(context)
        graph_db.flush()
        candidate_graph_ids.update(mid=mid.id, coarse=coarse.id, context=context.id)
        raise IngestionCancelled("cancel during mid/coarse/context build")

    monkeypatch.setattr(ingestion, "rebuild_context_graph", cancel_graph)
    result = await ingestion.run_uploaded_files_ingestion(batch.id, [str(source)])

    db_session.expire_all()
    recovery = db_session.scalar(
        select(IngestionBatchRecovery).where(IngestionBatchRecovery.batch_id == batch.id)
    )
    stage = db_session.scalar(
        select(IngestionFileStage).where(
            IngestionFileStage.batch_recovery_id == recovery.id
        )
    )
    candidate_chunk_ids = sorted(str(item) for item in stage.write_set_json["chunk_ids"])
    assert result["state"] == "cancelled"
    assert result["cancellation_status"] == "graph_cancelled_parse_scope_preserved"
    assert recovery.parse_committed is True
    assert recovery.status == "graph_compensated"
    assert recovery.compensation_json["parse_scope_preserved"] is True
    assert stage.status == "retained_after_parse_commit"
    assert sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before["document_id"],
                Chunk.state == "active",
            )
        ).all()
    ) == candidate_chunk_ids
    assert candidate_chunk_ids != before["chunk_ids"]
    assert all(
        db_session.get(model, candidate_graph_ids[key]) is None
        for key, model in (
            ("mid", MidConceptState),
            ("coarse", CoarseConceptState),
            ("context", ContextGraphState),
        )
    )
    assert sorted(
        db_session.scalars(
            select(ChunkStructureNode.id).where(
                ChunkStructureNode.document_version_id
                == stage.write_set_json["document_version_id"]
            )
        ).all()
    ) == sorted(stage.write_set_json["structure_node_ids"])
    assert set(
        db_session.scalars(
            select(VectorRecord.vector_status).where(
                VectorRecord.chunk_id.in_(candidate_chunk_ids)
            )
        ).all()
    ) == {"ready"}


def test_batch_recovery_hash_tamper_fails_closed(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionBatchRecovery
    from app.services import ingestion

    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [],
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=1,
        full_reparse=False,
    )
    recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    recovery.before_state_json = {
        **dict(recovery.before_state_json or {}),
        "v_before_batch": 999,
    }
    db_session.commit()
    with pytest.raises(RuntimeError, match="before-state identity is invalid"):
        ingestion.compensate_ingestion_batch_parse_writes(
            db_session,
            recovery=recovery,
            reason="tamper probe",
        )


def test_batch_recovery_service_validators_reject_owner_and_state_drift(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionBatchRecovery, KnowledgeBase
    from app.services import ingestion

    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [],
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=0,
        full_reparse=False,
    )
    stage = ingestion._prepare_file_stage(
        db_session,
        recovery=recovery,
        source_path=_kb_source(sample_knowledge_base, "validator-drift.md"),
        sequence_index=1,
    )
    other = KnowledgeBase(
        name="Ingestion fence service validator cross-KB",
        source_root="ingestion_fence-service-validator",
    )
    db_session.add(other)
    db_session.commit()

    recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    recovery.knowledge_base_id = other.id
    with pytest.raises(RuntimeError, match="batch/knowledge-base ownership is invalid"):
        ingestion._validate_batch_recovery(db_session, recovery)
    db_session.rollback()

    recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    recovery.parse_committed = True
    with pytest.raises(RuntimeError, match="status/parse-commit state is invalid"):
        ingestion._validate_batch_recovery(db_session, recovery)
    db_session.rollback()

    stage = db_session.get(type(stage), stage.id)
    stage.knowledge_base_id = other.id
    with pytest.raises(RuntimeError, match="recovery/knowledge-base ownership is invalid"):
        ingestion._validated_file_stage(
            db_session,
            stage,
            require_write_set=False,
        )
    db_session.rollback()

    stage = db_session.get(type(stage), stage.id)
    stage.phase = "database_restore"
    with pytest.raises(RuntimeError, match="status/phase state is invalid"):
        ingestion._validated_file_stage(
            db_session,
            stage,
            require_write_set=False,
        )
    db_session.rollback()


def test_terminal_recovery_defers_behind_same_kb_resource_fence(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from contextlib import contextmanager

    from app.models import IngestionBatchRecovery
    from app.services import ingestion

    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [],
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=1,
        full_reparse=False,
    )
    batch.status = "cancel_failed"
    batch.stats = {
        **dict(batch.stats or {}),
        "cancel_requested": True,
        "cancellation_status": "durable_compensation_pending_retry",
    }
    db_session.commit()

    @contextmanager
    def busy_fence(_db, *, knowledge_base_id, batch_id):
        assert knowledge_base_id == sample_knowledge_base.id
        assert batch_id == batch.id
        yield {
            "acquired": False,
            "backend": "postgresql",
            "reason": "knowledge_base_resource_busy",
        }

    monkeypatch.setattr(ingestion, "_metadata_recovery_resource_fence", busy_fence)
    monkeypatch.setattr(
        ingestion,
        "_recover_released_batch_metadata_intents",
        lambda *_args, **_kwargs: pytest.fail(
            "recovery must not mutate metadata while the same-KB fence is busy"
        ),
    )

    result = ingestion.reconcile_pending_ingestion_batch_recoveries(db_session)
    db_session.expire_all()
    assert result == {
        "protocol_version": "ingestion_batch_pending_recovery_reconcile_v1",
        "attempted_recovery_ids": [recovery.id],
        "recovered_recovery_ids": [],
        "pending_recovery_ids": [recovery.id],
        "ok": False,
    }
    assert db_session.get(IngestionBatchRecovery, recovery.id).status == "prepared"


@pytest.mark.asyncio
async def test_restart_reconciler_compensates_committed_parse_write_set(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Chunk, IngestionBatch, IngestionBatchRecovery, IngestionFileStage
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "restart-before-graph.md")
    before = await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# Interrupted candidate\n\nThe worker exits after this file commit and before graph entry.\n",
        encoding="utf-8",
    )
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=2,
        full_reparse=True,
    )
    stage = ingestion._prepare_file_stage(
        db_session,
        recovery=recovery,
        source_path=source,
        sequence_index=1,
    )
    await ingestion.ingest_file(
        db_session,
        source,
        batch_id=batch.id,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=2,
        file_stage_id=stage.id,
    )
    interrupted = db_session.get(IngestionBatch, batch.id)
    interrupted.status = "parsing"
    interrupted.success_count = 1
    interrupted.processed_files = 1
    interrupted.stats = {
        **dict(interrupted.stats or {}),
        "phase": "parsing",
        "ingestion_execution_mode": "inline",
        "batch_recovery_id": recovery.id,
        "v_before_batch": 1,
    }
    db_session.commit()
    _install_candidate_point_compensation(monkeypatch, fake_model_stack)

    ingestion.finalize_interrupted_batches()

    db_session.expire_all()
    observed_batch = db_session.get(IngestionBatch, batch.id)
    observed_recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    observed_stage = db_session.get(IngestionFileStage, stage.id)
    assert observed_batch.status == "failed"
    assert observed_batch.stats["retryable"] is True
    assert observed_batch.stats["batch_restart_compensation"]["verification"]["ok"] is True
    assert observed_recovery.status == "parse_compensated"
    assert observed_stage.status == "compensated"
    assert sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before["document_id"],
                Chunk.state == "active",
            )
        ).all()
    ) == before["chunk_ids"]
    delete_intent_ids_before = list(
        observed_stage.compensation_json["qdrant_delete_intent_ids"]
    )
    second = ingestion.compensate_ingestion_batch_parse_writes(
        db_session,
        recovery=observed_recovery,
        reason="idempotent restart replay",
    )
    db_session.expire_all()
    assert second["verification"]["ok"] is True
    assert (
        db_session.get(IngestionFileStage, stage.id)
        .compensation_json["qdrant_delete_intent_ids"]
        == delete_intent_ids_before
    )


@pytest.mark.asyncio
async def test_qdrant_delete_failure_is_retried_by_restart_reconciler(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import IngestionBatch, IngestionBatchRecovery, IngestionCompensationLog, IngestionFileStage
    from app.services import ingestion
    from app.services.cancellation import IngestionCancelled

    source = _kb_source(sample_knowledge_base, "qdrant-delete-retry.md")
    await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text("# Replacement\n\nQdrant deletion fails once.\n", encoding="utf-8")
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )
    _install_candidate_point_compensation(
        monkeypatch,
        fake_model_stack,
        fail_first_delete=True,
    )
    calls = 0

    def cancel_after_file(_db, _batch_id):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise IngestionCancelled("cancel for qdrant retry")

    monkeypatch.setattr(ingestion, "ensure_not_cancelled", cancel_after_file)
    with pytest.raises(RuntimeError, match="synthetic qdrant delete timeout"):
        await ingestion.run_uploaded_files_ingestion(
            batch.id,
            [str(source)],
            full_reparse=True,
        )

    db_session.expire_all()
    failed_batch = db_session.get(IngestionBatch, batch.id)
    recovery = db_session.scalar(
        select(IngestionBatchRecovery).where(IngestionBatchRecovery.batch_id == batch.id)
    )
    stage = db_session.scalar(
        select(IngestionFileStage).where(
            IngestionFileStage.batch_recovery_id == recovery.id
        )
    )
    intent_ids = list(
        (stage.compensation_json.get("qdrant_delete_intents") or {}).values()
    )
    assert failed_batch.status == "cancel_failed"
    assert recovery.status == "parse_compensating"
    assert stage.status == "compensation_pending"
    assert len(intent_ids) == 1
    assert db_session.get(IngestionCompensationLog, intent_ids[0]).status == "external_outcome_unknown"

    # Release this session's read transaction so the restart/beat reconciler can
    # acquire the same-KB resource fence through its own SessionLocal session.
    db_session.rollback()
    ingestion.finalize_interrupted_batches()
    db_session.expire_all()
    retried_batch = db_session.get(IngestionBatch, batch.id)
    retried_recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    assert retried_batch.status == "cancelled"
    assert retried_batch.stats["cancellation_status"] == "durable_compensation_retried"
    assert retried_batch.stats["pending_batch_recovery_reconciled"]["verification"]["ok"] is True
    assert retried_recovery.status == "parse_compensated"
    assert db_session.get(IngestionCompensationLog, intent_ids[0]).status == "committed"
    assert db_session.get(IngestionFileStage, stage.id).status == "compensated"


@pytest.mark.asyncio
async def test_cache_dispatch_failure_remains_pending_then_reconciles(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import IngestionBatchRecovery
    from app.services import ingestion
    from app.services.cancellation import IngestionCancelled

    source = _kb_source(sample_knowledge_base, "cache-dispatch-retry.md")
    await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text("# Replacement\n\nRedis dispatch fails once.\n", encoding="utf-8")
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )
    _install_candidate_point_compensation(monkeypatch, fake_model_stack)
    cache_attempts = 0

    def flaky_cache(_knowledge_base_id, *, strict=False):
        nonlocal cache_attempts
        assert strict is True
        cache_attempts += 1
        if cache_attempts == 1:
            raise RuntimeError("synthetic redis unavailable")
        return True

    monkeypatch.setattr(ingestion, "invalidate_context_graph_cache_after_commit", flaky_cache)
    cancellation_checks = 0

    def cancel_after_file(_db, _batch_id):
        nonlocal cancellation_checks
        cancellation_checks += 1
        if cancellation_checks >= 3:
            raise IngestionCancelled("cancel for cache retry")

    monkeypatch.setattr(ingestion, "ensure_not_cancelled", cancel_after_file)
    result = await ingestion.run_uploaded_files_ingestion(
        batch.id,
        [str(source)],
        full_reparse=True,
    )
    assert result["state"] == "cancelled"
    db_session.expire_all()
    recovery = db_session.scalar(
        select(IngestionBatchRecovery).where(IngestionBatchRecovery.batch_id == batch.id)
    )
    cache_state = recovery.compensation_json["cache_invalidation"]
    assert cache_state["status"] == "pending"
    assert cache_state["retryable"] is True
    assert cache_state["attempt_count"] == 1

    reconcile = ingestion.reconcile_pending_ingestion_recovery_cache_invalidations(
        db_session
    )
    db_session.expire_all()
    assert reconcile["ok"] is True
    assert reconcile["attempted_recovery_ids"] == [recovery.id]
    assert (
        db_session.get(IngestionBatchRecovery, recovery.id)
        .compensation_json["cache_invalidation"]["status"]
        == "dispatched"
    )


@pytest.mark.parametrize(
    "incomplete_receipt",
    [False, {"ok": True}, {"invalidated": True, "partial": True}],
)
def test_cache_dispatch_requires_literal_strict_success(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    incomplete_receipt,
):
    from app.models import IngestionBatchRecovery
    from app.services import ingestion

    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [],
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=0,
        full_reparse=False,
    )
    recovery.status = "parse_compensated"
    recovery.completed_at = ingestion.datetime.utcnow()
    recovery.compensation_json = {
        "cache_invalidation": {
            "protocol_version": "ingestion_batch_cache_invalidation_v1",
            "status": "pending",
            "attempt_count": 0,
            "knowledge_base_id": sample_knowledge_base.id,
        }
    }
    db_session.commit()
    strict_values: list[bool] = []

    def incomplete(_knowledge_base_id, *, strict=False):
        strict_values.append(strict)
        return incomplete_receipt

    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        incomplete,
    )
    result = ingestion._dispatch_batch_recovery_cache_invalidation(
        db_session,
        recovery_id=recovery.id,
    )
    assert strict_values == [True]
    assert result["status"] == "pending"
    assert result["retryable"] is True
    assert result["attempt_count"] == 1
    db_session.expire_all()
    persisted = db_session.get(IngestionBatchRecovery, recovery.id)
    assert persisted.compensation_json["cache_invalidation"]["status"] == "pending"


@pytest.mark.asyncio
async def test_qdrant_delete_intent_crash_window_reuses_durable_source_binding(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import IngestionCompensationLog, IngestionFileStage
    from app.services import ingestion, qdrant_outbox

    source = _kb_source(sample_knowledge_base, "delete-intent-crash-window.md")
    await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# Candidate\n\nThe delete intent is durable before the file-stage binding commit.\n",
        encoding="utf-8",
    )
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=2,
        full_reparse=True,
    )
    stage = ingestion._prepare_file_stage(
        db_session,
        recovery=recovery,
        source_path=source,
        sequence_index=1,
    )
    await ingestion.ingest_file(
        db_session,
        source,
        batch_id=batch.id,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=2,
        file_stage_id=stage.id,
    )
    store_type = _install_candidate_point_compensation(monkeypatch, fake_model_stack)
    stage = db_session.get(IngestionFileStage, stage.id)
    cards = [dict(card) for card in stage.write_set_json["vector_records"]]
    collection_name = str(cards[0]["collection_name"])
    point_ids = [str(card["qdrant_point_id"]) for card in cards]
    store = store_type(sample_knowledge_base.name, collection_name=collection_name)
    intent_id = qdrant_outbox.persist_qdrant_delete_attempt(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        collection_name=collection_name,
        target_ids=point_ids,
        current_points=store.get_points(point_ids),
        reason=qdrant_outbox.QDRANT_INGESTION_PARSE_COMPENSATION_REASON,
        source_binding={
            "protocol_version": qdrant_outbox.QDRANT_DELETE_SOURCE_BINDING_PROTOCOL_VERSION,
            "recovery_id": recovery.id,
            "file_stage_id": stage.id,
            "write_set_hash": stage.write_set_hash,
        },
    )
    db_session.commit()
    db_session.expire_all()
    assert not (
        db_session.get(IngestionFileStage, stage.id).compensation_json.get(
            "qdrant_delete_intents"
        )
    )

    compensation = ingestion.compensate_ingestion_batch_parse_writes(
        db_session,
        recovery=db_session.get(type(recovery), recovery.id),
        reason="recover crash after durable intent persistence",
    )
    db_session.expire_all()
    observed_stage = db_session.get(IngestionFileStage, stage.id)
    assert compensation["verification"]["ok"] is True
    assert observed_stage.compensation_json["qdrant_delete_intents"] == {
        collection_name: intent_id
    }
    matching = [
        row
        for row in db_session.scalars(
            select(IngestionCompensationLog).where(
                IngestionCompensationLog.knowledge_base_id
                == sample_knowledge_base.id,
                IngestionCompensationLog.operation
                == qdrant_outbox.QDRANT_DELETE_OPERATION,
            )
        ).all()
        if (row.payload_json or {}).get("reason")
        == qdrant_outbox.QDRANT_INGESTION_PARSE_COMPENSATION_REASON
    ]
    assert [row.id for row in matching] == [intent_id]


@pytest.mark.asyncio
async def test_qdrant_owner_mutation_enters_manual_review_without_capturing_owner(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import IngestionBatchRecovery, IngestionCompensationLog, IngestionFileStage
    from app.services import ingestion, qdrant_outbox

    source = _kb_source(sample_knowledge_base, "delete-owner-drift.md")
    await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text("# Candidate\n\nOwner mutation must fail closed.\n", encoding="utf-8")
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=2,
        full_reparse=True,
    )
    stage = ingestion._prepare_file_stage(
        db_session,
        recovery=recovery,
        source_path=source,
        sequence_index=1,
    )
    await ingestion.ingest_file(
        db_session,
        source,
        batch_id=batch.id,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=2,
        file_stage_id=stage.id,
    )
    store_type = _install_candidate_point_compensation(monkeypatch, fake_model_stack)
    stage = db_session.get(IngestionFileStage, stage.id)
    card = dict(stage.write_set_json["vector_records"][0])
    point_id = str(card["qdrant_point_id"])
    frozen_owner = str(card["qdrant_write_intent_id"])
    mutated_owner = "mutated-owner-must-not-be-captured"
    store_type.points[point_id]["payload"]["qdrant_write_intent_id"] = mutated_owner

    with pytest.raises(RuntimeError, match="manual Qdrant compensation review"):
        ingestion.compensate_ingestion_batch_parse_writes(
            db_session,
            recovery=db_session.get(IngestionBatchRecovery, recovery.id),
            reason="owner mutation adversarial probe",
        )
    db_session.expire_all()
    observed_recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    observed_stage = db_session.get(IngestionFileStage, stage.id)
    assert observed_recovery.status == "manual_review"
    assert observed_stage.status == "manual_review"
    manual = observed_stage.compensation_json["manual_review"]
    assert manual["current_owner_captured_as_before_image"] is False
    assert manual["automatic_delete_forbidden"] is True
    assert store_type.points[point_id]["payload"]["qdrant_write_intent_id"] == mutated_owner
    assert frozen_owner != mutated_owner
    assert not [
        row
        for row in db_session.scalars(
            select(IngestionCompensationLog).where(
                IngestionCompensationLog.knowledge_base_id
                == sample_knowledge_base.id,
                IngestionCompensationLog.operation
                == qdrant_outbox.QDRANT_DELETE_OPERATION,
            )
        ).all()
        if (row.payload_json or {}).get("reason")
        == qdrant_outbox.QDRANT_INGESTION_PARSE_COMPENSATION_REASON
    ]


@pytest.mark.asyncio
async def test_multi_file_partial_failure_uses_each_exact_before_state(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Chunk, ContextGraphState, IngestionBatchRecovery, IngestionFileStage
    from app.services import ingestion

    first = _kb_source(sample_knowledge_base, "partial-first.md")
    second = _kb_source(sample_knowledge_base, "partial-second.md")
    before_first = await _seed_document(db_session, sample_knowledge_base, first)
    before_second = await _seed_document(db_session, sample_knowledge_base, second)
    first.write_text("# First candidate\n\nThis file succeeds at v2.\n", encoding="utf-8")
    second.write_text("# Second candidate\n\nThis file fails and keeps its old v1 facts.\n", encoding="utf-8")
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [first, second],
        full_reparse=True,
    )
    original_ingest_file = ingestion.ingest_file

    async def one_success_one_failure(db, path, **kwargs):
        if Path(path) == second:
            raise RuntimeError("synthetic second-file parse failure")
        return await original_ingest_file(db, path, **kwargs)

    async def minimal_graph(graph_db, knowledge_base_id, **_kwargs):
        state = ContextGraphState(
            knowledge_base_id=knowledge_base_id,
            chunk_scope_hash="1" * 64,
            structure_graph_hash="2" * 64,
            chunk_relation_graph_hash="3" * 64,
            rq_membership_hash="4" * 64,
            mid_concept_hash="5" * 64,
            coarse_concept_hash="6" * 64,
            context_graph_hash="7" * 64,
            stats_json={"test_scope": "partial_failure"},
            state="active",
        )
        graph_db.add(state)
        graph_db.flush()
        return state

    monkeypatch.setattr(ingestion, "ingest_file", one_success_one_failure)
    monkeypatch.setattr(ingestion, "rebuild_context_graph", minimal_graph)
    result = await ingestion.run_uploaded_files_ingestion(
        batch.id,
        [str(first), str(second)],
        full_reparse=True,
    )

    db_session.expire_all()
    recovery = db_session.scalar(
        select(IngestionBatchRecovery).where(IngestionBatchRecovery.batch_id == batch.id)
    )
    stages = list(
        db_session.scalars(
            select(IngestionFileStage)
            .where(IngestionFileStage.batch_recovery_id == recovery.id)
            .order_by(IngestionFileStage.sequence_index.asc())
        ).all()
    )
    assert result["state"] == "partial_failed"
    assert recovery.status == "completed"
    assert recovery.parse_committed is True
    assert stages[0].status == "retained_after_parse_commit"
    assert stages[1].status == "failed"
    first_active = sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before_first["document_id"],
                Chunk.state == "active",
            )
        ).all()
    )
    second_active = sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before_second["document_id"],
                Chunk.state == "active",
            )
        ).all()
    )
    assert first_active == sorted(stages[0].write_set_json["chunk_ids"])
    assert first_active != before_first["chunk_ids"]
    assert second_active == before_second["chunk_ids"]
    assert db_session.get(type(sample_knowledge_base), sample_knowledge_base.id).current_chunk_version == 2


@pytest.mark.asyncio
async def test_full_rebuild_all_files_fail_does_not_advance_version(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Chunk, IngestionBatchRecovery, IngestionFileStage
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "all-fail.md")
    before = await _seed_document(db_session, sample_knowledge_base, source)
    source.write_text("# Candidate\n\nEvery file fails.\n", encoding="utf-8")
    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [source],
        full_reparse=True,
    )

    async def fail_every_file(*_args, **_kwargs):
        raise RuntimeError("synthetic all-file failure")

    monkeypatch.setattr(ingestion, "ingest_file", fail_every_file)
    result = await ingestion.run_uploaded_files_ingestion(
        batch.id,
        [str(source)],
        full_reparse=True,
    )
    db_session.expire_all()
    recovery = db_session.scalar(
        select(IngestionBatchRecovery).where(IngestionBatchRecovery.batch_id == batch.id)
    )
    stage = db_session.scalar(
        select(IngestionFileStage).where(
            IngestionFileStage.batch_recovery_id == recovery.id
        )
    )
    assert result["state"] == "failed"
    assert recovery.status == "completed_no_writes"
    assert recovery.parse_committed is False
    assert stage.status == "failed"
    assert db_session.get(type(sample_knowledge_base), sample_knowledge_base.id).current_chunk_version == 1
    assert sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before["document_id"],
                Chunk.state == "active",
            )
        ).all()
    ) == before["chunk_ids"]
