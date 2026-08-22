from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select


def test_vector_shadow_safe_failure_diagnostics_preserves_budget_without_message() -> None:
    from app.services.context_graph import (
        ConceptProviderBatchError,
        ConceptProviderRequestBudgetExceeded,
    )
    from app.services.vector_shadow_lifecycle import (
        vector_shadow_safe_failure_diagnostics,
    )

    secret = "authorization=Bearer must-not-persist"
    budget = ConceptProviderRequestBudgetExceeded(
        {
            "protocol_version": "concept_provider_request_budget_v1",
            "layer": "mid",
            "miss_count": 3,
            "max_requests": 16,
            "reserved_requests": 16,
            "observed_requests": 8,
            "next_group_worst_case_requests": 2,
            "provider_response_persisted": False,
            "diagnostics_hash": "a" * 64,
        }
    )
    budget.__cause__ = RuntimeError(secret)
    outer = ConceptProviderBatchError(
        layer="mid",
        batch_index=9,
        packet_ids=["0" * 16, "1" * 16],
    )
    outer.__cause__ = budget

    card = vector_shadow_safe_failure_diagnostics(outer)

    assert card["classification"] == (
        "concept_provider_request_budget_exhausted"
    )
    assert card["provider_batch"] == {
        "layer": "mid",
        "batch_index": 9,
        "packet_count": 2,
    }
    assert card["provider_request_budget"]["max_requests"] == 16
    assert card["provider_request_budget"]["observed_requests"] == 8
    assert card["provider_response_persisted"] is False
    assert card["credential_value_persisted"] is False
    assert secret not in json.dumps(card)


def _compensated_recovery_card(
    *,
    knowledge_base_id: str,
    index_input,
    target,
    vector: list[float],
) -> dict[str, object]:
    from app.services.context_graph import (
        CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        vector_payload_hash,
    )
    from app.services.vector_shadow_lifecycle import (
        VECTOR_SHADOW_WRITER_PROTOCOL_VERSION,
    )

    schema = target.schema
    hint = index_input.local_hint
    payload_hash = vector_payload_hash(
        vector=vector,
        chunk_id=str(index_input.chunk.id),
        embedding_model=schema.embedding_model,
        embedding_dimension=schema.embedding_dimension,
        vector_distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
        embedding_text_version=schema.embedding_text_version,
        chunk_schema_version=schema.chunk_schema_version,
        context_hash_protocol_version=CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
        context_hash=index_input.context_hash,
        local_hint_protocol_version=hint.protocol_version,
        local_hint_hash=hint.hint_hash,
        collection_identity_protocol_version=(
            QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION
        ),
        collection_identity_digest=schema.collection_identity_digest,
    )
    return {
        "knowledge_base_id": str(knowledge_base_id),
        "chunk_id": str(index_input.chunk.id),
        "embedding_model": schema.embedding_model,
        "embedding_dimension": schema.embedding_dimension,
        "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
        "embedding_text_version": schema.embedding_text_version,
        "chunk_schema_version": schema.chunk_schema_version,
        "collection_identity_protocol_version": (
            QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION
        ),
        "collection_identity_digest": schema.collection_identity_digest,
        "runtime_settings_candidate_id": target.runtime_settings_candidate_id,
        "vector_shadow_build_id": target.vector_shadow_build_id,
        "candidate_vector_schema_hash": target.vector_schema_hash,
        "vector_shadow_writer_protocol_version": (
            VECTOR_SHADOW_WRITER_PROTOCOL_VERSION
        ),
        "context_hash_protocol_version": CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
        "context_hash": index_input.context_hash,
        "local_hint_protocol_version": hint.protocol_version,
        "local_hint_hash": hint.hint_hash,
        "vector_payload_hash": payload_hash,
        "vector": vector,
    }


def _complete_recovery_audit(chunk_count: int) -> dict[str, object]:
    return {
        "protocol_version": "vector_shadow_compensated_embedding_recovery_v1",
        "bounded_intent_scan_count": 1,
        "contributing_intent_count": 1,
        "contributing_intent_set_hash": "c" * 64,
        "expected_chunk_count": chunk_count,
        "recovered_chunk_count": chunk_count,
        "complete": True,
        "embedding_provider_call_count": 0,
        "provider_response_persisted": False,
        "credential_value_persisted": False,
        "recovered_vector_set_hash": "d" * 64,
    }


@pytest.mark.asyncio
async def test_compensated_shadow_vectors_skip_embedding_provider(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import context_graph
    from app.services.context_graph import (
        _contextual_index_inputs,
        build_local_context_hints,
        ensure_contextual_indexes_current,
        write_contextual_indexes,
    )
    from app.services.vector_shadow_lifecycle import (
        _compensated_shadow_embedding_recovery,
        _shadow_build_scope_chunks,
        abandon_vector_shadow_candidate,
        resolve_shadow_vector_runtime_target,
        stage_vector_runtime_candidate,
        start_vector_shadow_build,
        vector_runtime_diagnostics,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    source_candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        embedding_model="candidate-recovery-model",
        embedding_dimension=8,
        source="unit_test_compensated_recovery",
    )
    build = start_vector_shadow_build(db_session, builds[0].id)
    target = resolve_shadow_vector_runtime_target(db_session, build.id)
    chunks = _shadow_build_scope_chunks(db_session, build)
    local_hints = build_local_context_hints(db_session, chunks)
    index_inputs = _contextual_index_inputs(
        db_session,
        chunks=chunks,
        local_hints=local_hints,
    )
    cards = {
        str(item.chunk.id): _compensated_recovery_card(
            knowledge_base_id=knowledge_base.id,
            index_input=item,
            target=target,
            vector=[float(position + 1) / 16.0 for position in range(8)],
        )
        for item in index_inputs
    }

    class ForbiddenEmbeddingProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("EmbeddingProvider must not be instantiated")

    monkeypatch.setattr(
        context_graph,
        "EmbeddingProvider",
        ForbiddenEmbeddingProvider,
    )
    stats = await write_contextual_indexes(
        db_session,
        knowledge_base=knowledge_base,
        chunks=chunks,
        local_hints=local_hints,
        vector_runtime_target=target,
        compensated_embedding_recovery_cards=cards,
        compensated_embedding_recovery_audit=_complete_recovery_audit(len(chunks)),
    )

    assert stats["vectors"] == len(chunks)
    assert stats["compensated_embedding_recovery"]["used"] is True
    assert stats["compensated_embedding_recovery"][
        "embedding_provider_call_count"
    ] == 0
    assert stats["compensated_embedding_recovery"][
        "provider_response_persisted"
    ] is False
    from app.models import IngestionCompensationLog
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_PROTOCOL_VERSION,
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        QDRANT_UPSERT_OPERATION,
    )

    intent_ids = stats["qdrant_outbox"]["intent_ids"]
    assert intent_ids
    history = list(db_session.info.get(QDRANT_OUTBOX_TEST_HISTORY_KEY) or [])
    for intent_id in intent_ids:
        payloads = [
            dict(item["payload"])
            for item in history
            if item.get("intent_id") == intent_id
            and isinstance(item.get("payload"), dict)
            and (item.get("payload") or {}).get("protocol_version")
            == QDRANT_OUTBOX_PROTOCOL_VERSION
        ]
        assert len(payloads) == 1
        payload = payloads[0]
        db_session.add(
            IngestionCompensationLog(
                id=intent_id,
                knowledge_base_id=knowledge_base.id,
                operation=QDRANT_UPSERT_OPERATION,
                target_ids_json=[
                    str(point["id"])
                    for point in payload["target_points"]
                ],
                payload_json=payload,
                status="compensated",
            )
        )
    db_session.flush()
    recovered_cards, recovered_audit = _compensated_shadow_embedding_recovery(
        db_session,
        build=build,
        chunks=chunks,
    )
    assert recovered_cards is not None
    assert set(recovered_cards) == {str(chunk.id) for chunk in chunks}
    assert recovered_audit["complete"] is True
    assert recovered_audit["embedding_provider_call_count"] == 0
    assert recovered_audit["protocol_version"] == (
        "vector_shadow_compensated_embedding_recovery_v2"
    )
    assert recovered_audit["cross_candidate_rebound"] is False
    recovered_vectors, recovered_validation = (
        context_graph._validated_compensated_embedding_vectors(
            knowledge_base=knowledge_base,
            index_inputs=index_inputs,
            context_rows=list(
                db_session.scalars(
                    select(context_graph.ChunkContextText).where(
                        context_graph.ChunkContextText.chunk_id.in_(
                            [chunk.id for chunk in chunks]
                        ),
                        context_graph.ChunkContextText.embedding_text_version
                        == target.schema.embedding_text_version,
                    )
                ).all()
            ),
            schema=target.schema,
            target=target,
            runtime_diagnostics=vector_runtime_diagnostics(target),
            recovery_cards=recovered_cards,
            recovery_audit=recovered_audit,
        )
    )
    assert len(recovered_vectors) == len(chunks)
    assert recovered_validation["embedding_provider_call_count"] == 0

    abandon_vector_shadow_candidate(
        db_session,
        source_candidate.id,
        disposition="superseded",
        reason="exercise exact cross-candidate embedding recovery",
    )
    target_candidate, target_builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        embedding_model="candidate-recovery-model",
        embedding_dimension=8,
        source="unit_test_cross_candidate_compensated_recovery",
        base_runtime_version_hash="e" * 64,
    )
    target_build = start_vector_shadow_build(
        db_session, target_builds[0].id
    )
    target = resolve_shadow_vector_runtime_target(
        db_session, target_build.id
    )
    rebound_cards, rebound_audit = _compensated_shadow_embedding_recovery(
        db_session,
        build=target_build,
        chunks=chunks,
    )

    assert rebound_cards is not None
    assert rebound_audit["complete"] is True
    assert rebound_audit["cross_candidate_rebound"] is True
    assert rebound_audit["source_binding_count"] == 1
    assert rebound_audit["target_runtime_settings_candidate_id"] == (
        target_candidate.id
    )
    assert all(
        card["runtime_settings_candidate_id"] == target_candidate.id
        and card["vector_shadow_build_id"] == target_build.id
        for card in rebound_cards.values()
    )
    rebound_vectors, rebound_validation = (
        context_graph._validated_compensated_embedding_vectors(
            knowledge_base=knowledge_base,
            index_inputs=index_inputs,
            context_rows=list(
                db_session.scalars(
                    select(context_graph.ChunkContextText).where(
                        context_graph.ChunkContextText.chunk_id.in_(
                            [chunk.id for chunk in chunks]
                        ),
                        context_graph.ChunkContextText.embedding_text_version
                        == target.schema.embedding_text_version,
                    )
                ).all()
            ),
            schema=target.schema,
            target=target,
            runtime_diagnostics=vector_runtime_diagnostics(target),
            recovery_cards=rebound_cards,
            recovery_audit=rebound_audit,
        )
    )
    assert len(rebound_vectors) == len(chunks)
    assert rebound_validation["embedding_provider_call_count"] == 0
    rebound_repair = await ensure_contextual_indexes_current(
        db_session,
        knowledge_base=knowledge_base,
        chunks=chunks,
        vector_runtime_target=target,
        compensated_embedding_recovery_cards=rebound_cards,
        compensated_embedding_recovery_audit=rebound_audit,
    )
    assert rebound_repair["reindexed_chunks"] == len(chunks)
    assert rebound_repair["write_stats"]["provider_reindex_chunks"] == len(
        chunks
    )
    assert rebound_repair["write_stats"]["provider_reindex"][
        "compensated_embedding_recovery"
    ]["embedding_provider_call_count"] == 0
    assert all(
        not context_graph._active_vector_record_protocol_reasons(
            record,
            vector_runtime_target=target,
        )
        for record in db_session.scalars(
            select(context_graph.VectorRecord).where(
                context_graph.VectorRecord.chunk_id.in_(
                    [chunk.id for chunk in chunks]
                ),
                context_graph.VectorRecord.embedding_model
                == target.schema.embedding_model,
                context_graph.VectorRecord.embedding_dimension
                == target.schema.embedding_dimension,
                context_graph.VectorRecord.embedding_text_version
                == target.schema.embedding_text_version,
                context_graph.VectorRecord.chunk_schema_version
                == target.schema.chunk_schema_version,
            )
        ).all()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper_field", "expected_error"),
    [
        ("context_hash", "binding does not match"),
        ("vector_payload_hash", "payload hash mismatch"),
    ],
)
async def test_compensated_shadow_vector_tamper_fails_closed(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch: pytest.MonkeyPatch,
    tamper_field: str,
    expected_error: str,
) -> None:
    from app.services import context_graph
    from app.services.context_graph import (
        _contextual_index_inputs,
        build_local_context_hints,
        write_contextual_indexes,
    )
    from app.services.vector_shadow_lifecycle import (
        _shadow_build_scope_chunks,
        resolve_shadow_vector_runtime_target,
        stage_vector_runtime_candidate,
        start_vector_shadow_build,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    _candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        embedding_model=f"candidate-recovery-tamper-{tamper_field}",
        embedding_dimension=8,
        source="unit_test_compensated_recovery_tamper",
    )
    build = start_vector_shadow_build(db_session, builds[0].id)
    target = resolve_shadow_vector_runtime_target(db_session, build.id)
    chunks = _shadow_build_scope_chunks(db_session, build)
    local_hints = build_local_context_hints(db_session, chunks)
    index_inputs = _contextual_index_inputs(
        db_session,
        chunks=chunks,
        local_hints=local_hints,
    )
    cards = {
        str(item.chunk.id): _compensated_recovery_card(
            knowledge_base_id=knowledge_base.id,
            index_input=item,
            target=target,
            vector=[float(position + 1) / 16.0 for position in range(8)],
        )
        for item in index_inputs
    }
    first_card = cards[sorted(cards)[0]]
    first_card[tamper_field] = "0" * 64

    class ForbiddenEmbeddingProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("EmbeddingProvider must not be instantiated")

    monkeypatch.setattr(
        context_graph,
        "EmbeddingProvider",
        ForbiddenEmbeddingProvider,
    )
    with pytest.raises(RuntimeError, match=expected_error):
        await write_contextual_indexes(
            db_session,
            knowledge_base=knowledge_base,
            chunks=chunks,
            local_hints=local_hints,
            vector_runtime_target=target,
            compensated_embedding_recovery_cards=cards,
            compensated_embedding_recovery_audit=_complete_recovery_audit(
                len(chunks)
            ),
        )


def _active_chunk(db_session, knowledge_base_id: str):
    from app.models import Chunk, Document, DocumentVersion

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title="Vector shadow lifecycle",
        source_path="vector-shadow.md",
        source_type="markdown",
        checksum="a" * 64,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        text="The active vector pointer must not change before promotion.",
        text_hash="b" * 64,
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_frozen_vector_schema_changes_collection_for_dimension():
    from app.services.vector_shadow_lifecycle import (
        frozen_vector_schema,
        vector_schema_hash,
    )

    first = frozen_vector_schema(
        embedding_model="same/model",
        embedding_dimension=8,
    )
    second = frozen_vector_schema(
        embedding_model="same/model",
        embedding_dimension=16,
    )

    assert first.collection_name != second.collection_name
    assert first.collection_identity_digest != second.collection_identity_digest
    assert vector_schema_hash(first) != vector_schema_hash(second)
    assert first.distance_metric == second.distance_metric == "cosine"


def test_stage_candidate_writes_intent_without_active_pointer(
    db_session,
    sample_knowledge_base,
):
    from app.models import KnowledgeBaseVectorRuntimeState, RuntimeSettingsCandidate
    from app.services.vector_shadow_lifecycle import stage_vector_runtime_candidate

    _active_chunk(db_session, sample_knowledge_base.id)
    candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[sample_knowledge_base.id],
        embedding_model="candidate-shadow-model",
        embedding_dimension=16,
        source="unit_test",
    )
    db_session.commit()

    assert candidate.status == "staged"
    assert candidate.protocol_version == "runtime_settings_vector_candidate_v1"
    assert candidate.lifecycle_scope == "rebuild_required"
    assert candidate.settings_json["embedding_model"] == "candidate-shadow-model"
    assert candidate.diagnostics_json["active_env_mutated"] is False
    assert len(builds) == 1
    assert builds[0].status == "staged"
    assert builds[0].expected_point_count == 1
    assert builds[0].ready_point_count == 0
    assert db_session.scalar(select(KnowledgeBaseVectorRuntimeState)) is None
    assert db_session.scalar(select(RuntimeSettingsCandidate).where(
        RuntimeSettingsCandidate.id == candidate.id
    )) is not None


def test_promotion_fails_closed_until_build_is_evaluated(
    db_session,
    sample_knowledge_base,
):
    from app.models import KnowledgeBaseVectorRuntimeState
    from app.services.vector_shadow_lifecycle import (
        promote_vector_shadow_candidate,
        stage_vector_runtime_candidate,
    )

    _active_chunk(db_session, sample_knowledge_base.id)
    candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[sample_knowledge_base.id],
        embedding_model="candidate-shadow-model",
        embedding_dimension=16,
        source="unit_test",
    )

    decision = promote_vector_shadow_candidate(db_session, candidate.id)
    db_session.commit()

    assert decision["allowed"] is False
    assert any("expected:evaluation_passed" in blocker for blocker in decision["blockers"])
    assert "atomic_active_vector_switch_not_implemented" not in decision["blockers"]
    assert not any(
        blocker.startswith("active_consumer_not_integrated:")
        for blocker in decision["blockers"]
    )
    assert candidate.status == "staged"
    assert candidate.diagnostics_json["last_promotion_preflight"] == decision
    assert builds[0].status == "staged"
    assert db_session.scalar(select(KnowledgeBaseVectorRuntimeState)) is None


@pytest.mark.asyncio
async def test_shadow_build_promotion_and_rollback_switch_exact_postgres_facts(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch: pytest.MonkeyPatch,
):
    import hashlib

    from app.models import (
        CoarseConcept,
        ContextGraphState,
        IngestionCompensationLog,
        KnowledgeBaseVectorRuntimeState,
        MidConcept,
        RQPrefix,
        VectorRecord,
    )
    from app.services.context_graph import active_graph_admission_gate
    from app.services.vector_collection_cleanup import (
        prepare_vector_collection_cleanup,
        vector_collection_cleanup_plan,
    )
    from app.services.vector_shadow_lifecycle import (
        REQUIRED_EVALUATION_EVIDENCE,
        REQUIRED_EVALUATION_GATES,
        VectorShadowEvaluation,
        _active_concept_semantic_reuse_sources,
        abandon_vector_shadow_candidate,
        build_vector_shadow_artifacts,
        promote_vector_shadow_candidate,
        reconcile_vector_runtime_cache_invalidations,
        record_vector_shadow_evaluation,
        rollback_vector_shadow_candidate,
        stage_vector_runtime_candidate,
        start_vector_shadow_build,
        vector_shadow_evaluation_input_hash,
        vector_shadow_promotion_preflight,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    old_context = populated_context_graph["state"]
    active_pointer_before = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert active_pointer_before is not None
    old_schema_hash = active_pointer_before.vector_schema_hash
    old_pointer_state_hash = active_pointer_before.state_hash

    candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        embedding_model="candidate-shadow-model",
        embedding_dimension=8,
        source="unit_test_full_lifecycle",
    )
    build = await build_vector_shadow_artifacts(
        db_session,
        build_id=builds[0].id,
        emit_heartbeats=False,
        concept_provider_request_budget=128,
    )
    shadow_context = db_session.get(ContextGraphState, build.shadow_context_graph_state_id)
    assert shadow_context is not None
    assert build.status == "shadow_ready"
    assert shadow_context.state == "shadow"
    assert build.qdrant_proof_json["verified"] is True
    assert build.qdrant_proof_json["scoped_point_count"] == len(
        populated_context_graph["chunks"]
    )
    assert len(build.qdrant_proof_json["point_set_hash"]) == 64
    assert len(build.qdrant_proof_json["observer_input_hash"]) == 64
    assert len(build.qdrant_proof_json["observer_output_hash"]) == 64
    semantic_reuse = build.diagnostics_json[
        "vector_shadow_concept_semantic_reuse"
    ]
    assert semantic_reuse["protocol_version"] == (
        "vector_shadow_terminal_concept_semantic_reuse_v2"
    )
    assert semantic_reuse["source_kind"] == "active_pointer"
    assert semantic_reuse["exact_reuse_required"] is False
    assert semantic_reuse["source_available"] is True
    assert semantic_reuse["source_mid_concept_state_id"] == (
        old_context.mid_concept_state_id
    )
    assert semantic_reuse["source_coarse_concept_state_id"] == (
        old_context.coarse_concept_state_id
    )
    assert semantic_reuse["strict_semantic_identity_required"] is True
    assert semantic_reuse["provider_allowed_on_miss"] is True
    assert semantic_reuse["provider_request_count"] == (
        semantic_reuse["mid_provider_request_count"]
        + semantic_reuse["coarse_provider_request_count"]
    )
    assert semantic_reuse["provider_request_budget"]["max_requests"] == 128
    assert semantic_reuse["provider_request_budget"][
        "provider_response_persisted"
    ] is False

    terminal_reuse_nested = db_session.begin_nested()
    abandon_vector_shadow_candidate(
        db_session,
        candidate.id,
        disposition="superseded",
        reason="exercise exact terminal concept semantic reuse",
    )
    terminal_target_candidate, terminal_target_builds = (
        stage_vector_runtime_candidate(
            db_session,
            knowledge_base_ids=[knowledge_base.id],
            embedding_model="candidate-shadow-model",
            embedding_dimension=8,
            source="unit_test_terminal_concept_reuse",
            base_runtime_version_hash="f" * 64,
        )
    )
    terminal_target_build = start_vector_shadow_build(
        db_session, terminal_target_builds[0].id
    )
    terminal_mid, terminal_coarse, terminal_audit = (
        _active_concept_semantic_reuse_sources(
            db_session, terminal_target_build
        )
    )
    assert terminal_mid is not None
    assert terminal_coarse is not None
    assert terminal_audit["source_kind"] == "terminal_shadow"
    assert terminal_audit["source_candidate_id"] == candidate.id
    assert terminal_audit["source_build_id"] == build.id
    assert terminal_audit["exact_reuse_required"] is True
    assert terminal_audit["provider_allowed_on_miss"] is False
    assert terminal_audit["provider_response_persisted"] is False
    assert terminal_target_candidate.status == "building"
    terminal_reuse_nested.rollback()
    db_session.expire_all()
    build = db_session.get(type(build), build.id)
    shadow_context = db_session.get(
        ContextGraphState, build.shadow_context_graph_state_id
    )
    candidate = db_session.get(type(candidate), candidate.id)

    candidate_records_before_promotion = list(
        db_session.scalars(
            select(VectorRecord)
            .where(
                VectorRecord.knowledge_base_id == knowledge_base.id,
                VectorRecord.embedding_model == "candidate-shadow-model",
            )
            .order_by(VectorRecord.chunk_id.asc())
        ).all()
    )
    point_cards_before_promotion = [
        {
            "chunk_id": record.chunk_id,
            "qdrant_point_id": record.qdrant_point_id,
            "payload_hash": record.payload_hash,
        }
        for record in candidate_records_before_promotion
    ]
    point_set_hash_before_promotion = hashlib.sha256(
        json.dumps(
            point_cards_before_promotion,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert build.qdrant_proof_json["point_set_hash"] == point_set_hash_before_promotion
    assert fake_model_stack["VectorStore"].unbounded_list_ids_calls == 0

    nested = db_session.begin_nested()
    failed_evaluation = VectorShadowEvaluation(
        evaluation_input_hash=vector_shadow_evaluation_input_hash(
            build, shadow_context
        ),
        hard_gates={
            name: name != "latency_budget"
            for name in REQUIRED_EVALUATION_GATES
        },
        evidence_hashes={
            name: hashlib.sha256(
                f"unit:failed:{name}".encode("utf-8")
            ).hexdigest()
            for name in REQUIRED_EVALUATION_EVIDENCE
        },
        metrics={"build_elapsed_ms": 700_000.0},
        evaluator_diagnostics={"source": "unit_test_failed_gate"},
    )
    record_vector_shadow_evaluation(
        db_session,
        build_id=build.id,
        evaluation=failed_evaluation,
    )
    db_session.flush()
    assert build.status == "promotion_blocked"
    assert candidate.status == "promotion_blocked"
    assert build.blocking_reasons_json == [
        "evaluation_gate_failed:latency_budget"
    ]
    blocked_preflight = vector_shadow_promotion_preflight(
        db_session, candidate.id
    )
    assert blocked_preflight["allowed"] is False
    assert any(
        "status:promotion_blocked:expected:evaluation_passed" in blocker
        for blocker in blocked_preflight["blockers"]
    )
    replay = record_vector_shadow_evaluation(
        db_session,
        build_id=build.id,
        evaluation=failed_evaluation,
    )
    assert replay.evaluation_result_hash == build.evaluation_result_hash
    nested.rollback()
    db_session.expire_all()
    build = db_session.get(type(build), build.id)
    shadow_context = db_session.get(
        ContextGraphState, build.shadow_context_graph_state_id
    )
    candidate = db_session.get(type(candidate), candidate.id)

    evaluation = VectorShadowEvaluation(
        evaluation_input_hash=vector_shadow_evaluation_input_hash(build, shadow_context),
        hard_gates={name: True for name in REQUIRED_EVALUATION_GATES},
        evidence_hashes={
            name: hashlib.sha256(f"unit:{name}".encode("utf-8")).hexdigest()
            for name in REQUIRED_EVALUATION_EVIDENCE
        },
        metrics={"retrieval_recall": 1.0, "citation_precision": 1.0},
        evaluator_diagnostics={"source": "unit_test_artifacts"},
    )
    record_vector_shadow_evaluation(
        db_session,
        build_id=build.id,
        evaluation=evaluation,
    )
    promotion = promote_vector_shadow_candidate(db_session, candidate.id)
    db_session.flush()

    assert promotion["allowed"] is True
    assert promotion["promoted"] is True
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert pointer is not None
    assert pointer.runtime_settings_candidate_id == candidate.id
    assert pointer.vector_schema_hash == build.candidate_vector_schema_hash
    assert pointer.active_context_graph_state_id == shadow_context.id
    assert old_context.state == "inactive"
    assert shadow_context.state == "active"
    assert set(
        db_session.scalars(
            select(RQPrefix.state).where(
                RQPrefix.graph_state_id
                == shadow_context.chunk_relation_graph_state_id
            )
        ).all()
    ) == {"active"}
    assert set(
        db_session.scalars(
            select(MidConcept.state).where(
                MidConcept.concept_state_id == shadow_context.mid_concept_state_id
            )
        ).all()
    ) == {"active"}
    assert set(
        db_session.scalars(
            select(CoarseConcept.state).where(
                CoarseConcept.coarse_state_id
                == shadow_context.coarse_concept_state_id
            )
        ).all()
    ) == {"active"}
    active_graph_admission_gate(db_session, knowledge_base.id)
    promotion_replay = promote_vector_shadow_candidate(db_session, candidate.id)
    assert promotion_replay["idempotent_replay"] is True
    assert promotion_replay["active_pointer_mutated"] is False
    promoted_state_hash = pointer.state_hash
    pointer.state_hash = "0" * 64
    db_session.flush()
    with pytest.raises(RuntimeError, match="non-canonical state_hash"):
        promote_vector_shadow_candidate(db_session, candidate.id)
    pointer.state_hash = promoted_state_hash
    db_session.flush()

    statuses_after_promotion = {
        (record.embedding_model, record.vector_status)
        for record in db_session.scalars(
            select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base.id)
        ).all()
    }
    assert ("candidate-shadow-model", "ready") in statuses_after_promotion
    assert any(status == "rollback_retained" for _model, status in statuses_after_promotion)

    previous_schema = (pointer.previous_state_json or {})["vector_schema"]
    retained_cleanup_plan = vector_collection_cleanup_plan(
        db_session,
        collection_name=previous_schema["collection_name"],
        check_qdrant=False,
    )
    assert retained_cleanup_plan["allowed"] is True
    assert retained_cleanup_plan["rollback_pointer_cards"] == [
        {
            "pointer_id": pointer.id,
            "knowledge_base_id": knowledge_base.id,
            "active_candidate_id": candidate.id,
            "active_pointer_state_hash": pointer.state_hash,
            "rollback_recovery_will_be_relinquished": True,
        }
    ]
    assert any(
        card["vector_status"] == "rollback_retained"
        for card in retained_cleanup_plan["record_owner_cards"]
    )
    cleanup_savepoint = db_session.begin_nested()
    prepare_vector_collection_cleanup(
        db_session,
        audit_knowledge_base_id=knowledge_base.id,
        collection_name=previous_schema["collection_name"],
        confirmed_collection_name=previous_schema["collection_name"],
        allow_sqlite_test_adapter=True,
    )
    db_session.flush()
    with pytest.raises(RuntimeError, match="destructive cleanup intent"):
        rollback_vector_shadow_candidate(
            db_session,
            candidate.id,
            reason="must not race exact retained-collection cleanup",
        )
    cleanup_savepoint.rollback()
    db_session.expire_all()
    candidate = db_session.get(type(candidate), candidate.id)
    pointer = db_session.get(type(pointer), pointer.id)
    assert candidate is not None
    assert pointer is not None

    pending_promotion_intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.knowledge_base_id == knowledge_base.id,
            IngestionCompensationLog.status == "cache_invalidation_pending",
        )
    )
    assert pending_promotion_intent is not None
    dry_run = reconcile_vector_runtime_cache_invalidations(
        db_session,
        candidate_id=candidate.id,
        dry_run=True,
    )
    assert dry_run["target_count"] == 1
    assert dry_run["applied"] is False
    from app.services import context_graph as context_graph_service

    def redis_unavailable(*_args, **_kwargs):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(
        context_graph_service,
        "invalidate_context_graph_cache_after_commit",
        redis_unavailable,
    )
    failed_reconcile = reconcile_vector_runtime_cache_invalidations(
        db_session,
        candidate_id=candidate.id,
        dry_run=False,
    )
    assert failed_reconcile["failed_count"] == 1
    assert failed_reconcile["applied"] is False
    assert pending_promotion_intent.status == "cache_invalidation_pending"
    assert (pending_promotion_intent.payload_json or {})["attempt_count"] == 1

    rollback_vector_shadow_candidate(
        db_session,
        candidate.id,
        reason="unit test rollback",
    )
    db_session.flush()
    db_session.refresh(pointer)

    assert candidate.status == "rolled_back"
    assert pointer.runtime_settings_candidate_id is None
    assert pointer.vector_schema_hash == old_schema_hash
    assert pointer.state_hash != old_pointer_state_hash
    assert pointer.active_context_graph_state_id == old_context.id
    assert old_context.state == "active"
    assert shadow_context.state == "inactive"
    assert set(
        db_session.scalars(
            select(RQPrefix.state).where(
                RQPrefix.graph_state_id
                == shadow_context.chunk_relation_graph_state_id
            )
        ).all()
    ) == {"inactive"}
    assert set(
        db_session.scalars(
            select(MidConcept.state).where(
                MidConcept.concept_state_id == shadow_context.mid_concept_state_id
            )
        ).all()
    ) == {"inactive"}
    assert set(
        db_session.scalars(
            select(CoarseConcept.state).where(
                CoarseConcept.coarse_state_id
                == shadow_context.coarse_concept_state_id
            )
        ).all()
    ) == {"inactive"}
    assert pending_promotion_intent.status == "superseded"
    rollback_intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.knowledge_base_id == knowledge_base.id,
            IngestionCompensationLog.status == "cache_invalidation_pending",
        )
    )
    assert rollback_intent is not None
    assert (rollback_intent.payload_json or {})["action"] == "rollback"
    candidate_records_after_rollback = list(
        db_session.scalars(
            select(VectorRecord)
            .where(
                VectorRecord.knowledge_base_id == knowledge_base.id,
                VectorRecord.embedding_model == "candidate-shadow-model",
            )
            .order_by(VectorRecord.chunk_id.asc())
        ).all()
    )
    assert [record.payload_hash for record in candidate_records_after_rollback] == [
        record.payload_hash for record in candidate_records_before_promotion
    ]
    active_graph_admission_gate(db_session, knowledge_base.id)
    assert rollback_vector_shadow_candidate(
        db_session,
        candidate.id,
        reason="idempotent replay",
    ).status == "rolled_back"
    restored_state_hash = pointer.state_hash
    pointer.state_hash = "f" * 64
    db_session.flush()
    with pytest.raises(RuntimeError, match="non-canonical state_hash"):
        rollback_vector_shadow_candidate(
            db_session,
            candidate.id,
            reason="tampered replay must fail",
        )
    pointer.state_hash = restored_state_hash
    db_session.flush()


@pytest.mark.asyncio
async def test_shadow_build_fails_closed_on_real_qdrant_payload_tamper(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.models import VectorRecord, VectorShadowBuild
    from app.services.vector_shadow_lifecycle import (
        build_vector_shadow_artifacts,
        record_vector_shadow_build_attempt_failure,
        stage_vector_runtime_candidate,
        start_vector_shadow_build,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        embedding_model="candidate-tampered-model",
        embedding_dimension=8,
        source="unit_test_qdrant_tamper",
    )
    build_id = builds[0].id
    db_session.commit()
    fake_store = fake_model_stack["VectorStore"]
    original_snapshot = fake_store.expected_points_snapshot

    def tampered_snapshot(self, expected_point_ids, *, expected_vector_size, batch_size=256):
        snapshot = original_snapshot(
            self,
            expected_point_ids,
            expected_vector_size=expected_vector_size,
            batch_size=batch_size,
        )
        points = [dict(point) for point in list(snapshot.get("points") or [])]
        for point in points:
            payload = dict(point.get("payload") or {})
            if payload.get("embedding_model") == "candidate-tampered-model":
                payload["vector_payload_hash"] = "0" * 64
                point["payload"] = payload
        return {**snapshot, "points": points}

    monkeypatch.setattr(fake_store, "expected_points_snapshot", tampered_snapshot)
    with pytest.raises(RuntimeError, match="Qdrant repair did not converge"):
        await build_vector_shadow_artifacts(
            db_session,
            build_id=build_id,
            emit_heartbeats=False,
        )
    db_session.rollback()

    persisted_build = db_session.get(VectorShadowBuild, build_id)
    assert persisted_build is not None
    assert persisted_build.status == "staged"
    assert persisted_build.qdrant_proof_hash is None
    assert persisted_build.shadow_context_graph_state_id is None
    assert db_session.scalar(
        select(func.count(VectorRecord.id)).where(
            VectorRecord.knowledge_base_id == knowledge_base.id,
            VectorRecord.embedding_model == "candidate-tampered-model",
        )
    ) == 0
    assert candidate.status == "staged"

    failed_attempt = record_vector_shadow_build_attempt_failure(
        db_session,
        build_id,
        error_type="RuntimeError",
    )
    db_session.commit()
    assert failed_attempt.status == "staged"
    assert failed_attempt.error_code == "vector_shadow_build_attempt_failed"
    assert failed_attempt.last_error == "RuntimeError"
    failure_audit = (failed_attempt.diagnostics_json or {})["last_failed_attempt"]
    assert failure_audit["attempt_count"] == 1
    assert failure_audit["partial_postgresql_facts_rolled_back"] is True
    assert failure_audit["provider_response_persisted"] is False

    retry = start_vector_shadow_build(db_session, build_id)
    assert retry.status == "building"
    assert retry.error_code is None
    assert retry.last_error is None
    db_session.rollback()


def test_shadow_candidate_rejects_complete_chunk_address_drift(
    db_session,
    sample_knowledge_base,
):
    from app.services.context_graph import CHUNK_SCOPE_HASH_PROTOCOL_VERSION
    from app.services.vector_shadow_lifecycle import (
        stage_vector_runtime_candidate,
        start_vector_shadow_build,
    )

    chunk = _active_chunk(db_session, sample_knowledge_base.id)
    _candidate, builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[sample_knowledge_base.id],
        embedding_model="candidate-shadow-model",
        embedding_dimension=16,
        source="unit_test",
    )
    diagnostics = dict(builds[0].diagnostics_json or {})
    assert (
        diagnostics["staged_active_chunk_scope_hash_protocol_version"]
        == CHUNK_SCOPE_HASH_PROTOCOL_VERSION
    )

    # Keep the durable chunk identity fixed while changing one field from the
    # complete address card. An ID-only scope fingerprint would miss this.
    chunk.text_hash = "c" * 64
    db_session.flush()

    with pytest.raises(RuntimeError, match="Active chunk scope changed"):
        start_vector_shadow_build(db_session, builds[0].id)


def test_one_live_candidate_per_kb_requires_explicit_abandonment(
    db_session,
    sample_knowledge_base,
):
    from app.services.vector_shadow_lifecycle import (
        abandon_vector_shadow_candidate,
        stage_vector_runtime_candidate,
    )

    _active_chunk(db_session, sample_knowledge_base.id)
    first, _first_builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[sample_knowledge_base.id],
        embedding_model="candidate-shadow-model-a",
        embedding_dimension=16,
        source="unit_test",
    )
    with pytest.raises(RuntimeError, match="live vector shadow build already owns"):
        stage_vector_runtime_candidate(
            db_session,
            knowledge_base_ids=[sample_knowledge_base.id],
            embedding_model="candidate-shadow-model-b",
            embedding_dimension=32,
            source="unit_test",
        )

    abandon_vector_shadow_candidate(
        db_session,
        first.id,
        disposition="superseded",
        reason="replace candidate after review",
    )
    second, second_builds = stage_vector_runtime_candidate(
        db_session,
        knowledge_base_ids=[sample_knowledge_base.id],
        embedding_model="candidate-shadow-model-b",
        embedding_dimension=32,
        source="unit_test",
    )

    assert first.status == "superseded"
    assert second.status == "staged"
    assert second_builds[0].status == "staged"


def test_direct_vector_setting_update_fails_before_env_side_effect(
    no_fallback_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.services import runtime_settings

    normalize_called = False

    def unexpected_normalize() -> bool:
        nonlocal normalize_called
        normalize_called = True
        return False

    monkeypatch.setattr(runtime_settings, "normalize_env_file", unexpected_normalize)
    with pytest.raises(ValueError, match="Direct active mutation is forbidden"):
        runtime_settings.update_model_settings(
            {
                "embedding_model": "candidate-shadow-model",
                "embedding_dimensions": 16,
            }
        )

    assert normalize_called is False
