from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_prebuild_repairs_deleted_qdrant_point_and_reverifies_full_scope(
    monkeypatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from sqlalchemy import select

    from app.models import VectorRecord
    from app.services import context_graph
    from app.services.context_graph import ensure_contextual_indexes_current

    knowledge_base = populated_context_graph["knowledge_base"]
    chunks = populated_context_graph["chunks"]
    target_chunk = chunks[0]
    fake_store = fake_model_stack["VectorStore"]
    fake_store.expected_snapshot_calls.clear()
    fake_store.unbounded_list_ids_calls = 0
    fake_store.points.pop(str(target_chunk.id))
    vector_row = db_session.scalar(
        select(VectorRecord).where(VectorRecord.chunk_id == target_chunk.id)
    )
    assert vector_row is not None
    vector_row.vector_status = "stale"
    db_session.flush()

    class EmbeddingProviderMustNotRun:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "Qdrant-only repair must replay canonical PostgreSQL vectors"
            )

    monkeypatch.setattr(
        context_graph,
        "EmbeddingProvider",
        EmbeddingProviderMustNotRun,
    )

    result = await ensure_contextual_indexes_current(
        db_session,
        knowledge_base=knowledge_base,
        chunks=chunks,
    )

    assert result["reindexed_chunks"] == 1
    assert "qdrant_point_missing" in result["stale_reasons_by_chunk"][str(target_chunk.id)]
    assert "vector_not_ready" in result["stale_reasons_by_chunk"][str(target_chunk.id)]
    assert result["write_stats"]["postgresql_qdrant_replay_chunks"] == 1
    assert result["write_stats"]["provider_reindex_chunks"] == 0
    assert result["write_stats"]["postgresql_qdrant_replay"][
        "embedding_provider_call_count"
    ] == 0
    assert result["qdrant_freshness"]["verified"] is True
    assert result["qdrant_freshness"]["observed_point_count"] == len(chunks)
    assert str(target_chunk.id) in fake_store.points
    db_session.refresh(vector_row)
    assert vector_row.vector_status == "ready"
    assert (vector_row.diagnostics_json or {})[
        "qdrant_replay_protocol_version"
    ] == context_graph.POSTGRES_VECTOR_QDRANT_REPLAY_PROTOCOL_VERSION
    assert len(fake_store.expected_snapshot_calls) == 2
    assert fake_store.unbounded_list_ids_calls == 0


@pytest.mark.asyncio
async def test_prebuild_repairs_qdrant_payload_identity_tamper(
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        ensure_contextual_indexes_current,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    chunks = populated_context_graph["chunks"]
    target_chunk = chunks[0]
    fake_store = fake_model_stack["VectorStore"]
    point = dict(fake_store.points[str(target_chunk.id)])
    point["payload"] = {
        **dict(point["payload"]),
        "collection_identity_digest": "tampered-digest",
        "vector_payload_hash": "tampered-payload-hash",
    }
    fake_store.points[str(target_chunk.id)] = point

    result = await ensure_contextual_indexes_current(
        db_session,
        knowledge_base=knowledge_base,
        chunks=chunks,
    )

    reasons = result["stale_reasons_by_chunk"][str(target_chunk.id)]
    assert "qdrant_collection_identity_digest_mismatch" in reasons
    assert "qdrant_vector_payload_hash_mismatch" in reasons
    repaired_payload = fake_store.points[str(target_chunk.id)]["payload"]
    assert repaired_payload["collection_identity_protocol_version"] == (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION
    )
    assert repaired_payload["collection_identity_digest"] != "tampered-digest"
    assert repaired_payload["vector_payload_hash"] != "tampered-payload-hash"
    assert result["qdrant_freshness"]["verified"] is True


@pytest.mark.asyncio
async def test_prebuild_fails_closed_on_active_collection_schema_mismatch(
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from app.services.context_graph import (
        ActiveQdrantSchemaMismatchError,
        ensure_contextual_indexes_current,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    chunks = populated_context_graph["chunks"]
    fake_store = fake_model_stack["VectorStore"]
    collection_name = next(iter(fake_store.collection_schemas))
    fake_store.collection_schemas[collection_name] = (7, "cosine")

    with pytest.raises(ActiveQdrantSchemaMismatchError, match="schema"):
        await ensure_contextual_indexes_current(
            db_session,
            knowledge_base=knowledge_base,
            chunks=chunks,
        )


@pytest.mark.asyncio
async def test_graph_only_verify_mode_rejects_contextual_drift_without_repair_or_graph_mutation(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from sqlalchemy import func, select

    from app.models import ChunkContextText, ChunkRelationGraphState, ContextGraphState
    from app.services import context_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    chunks = populated_context_graph["chunks"]
    previous_context_state = populated_context_graph["state"]
    previous_relation_count = int(
        db_session.scalar(
            select(func.count(ChunkRelationGraphState.id)).where(
                ChunkRelationGraphState.knowledge_base_id == knowledge_base.id
            )
        )
        or 0
    )
    target_row = db_session.scalar(
        select(ChunkContextText).where(ChunkContextText.chunk_id == chunks[0].id)
    )
    db_session.delete(target_row)
    db_session.flush()
    forbidden_calls = {"write": 0, "embed": 0, "upsert": 0}

    async def forbidden_write(*_args, **_kwargs):
        forbidden_calls["write"] += 1
        raise AssertionError("graph-only verification must not repair contextual indexes")

    async def forbidden_embed(*_args, **_kwargs):
        forbidden_calls["embed"] += 1
        raise AssertionError("graph-only verification must not call embeddings")

    async def forbidden_upsert(*_args, **_kwargs):
        forbidden_calls["upsert"] += 1
        raise AssertionError("graph-only verification must not upsert Qdrant")

    monkeypatch.setattr(context_graph, "write_contextual_indexes", forbidden_write)
    monkeypatch.setattr(context_graph.EmbeddingProvider, "embed_texts", forbidden_embed)
    monkeypatch.setattr(context_graph, "execute_qdrant_upsert_batches", forbidden_upsert)

    with pytest.raises(RuntimeError, match="drift is forbidden during graph-only retry"):
        await context_graph.rebuild_context_graph(
            db_session,
            knowledge_base.id,
            contextual_index_repair_mode=(
                context_graph.CONTEXTUAL_INDEX_REPAIR_MODE_VERIFY_ONLY
            ),
        )

    assert forbidden_calls == {"write": 0, "embed": 0, "upsert": 0}
    assert db_session.get(ContextGraphState, previous_context_state.id).state == "active"
    assert int(
        db_session.scalar(
            select(func.count(ChunkRelationGraphState.id)).where(
                ChunkRelationGraphState.knowledge_base_id == knowledge_base.id
            )
        )
        or 0
    ) == previous_relation_count


@pytest.mark.asyncio
async def test_layered_search_rejects_old_graph_without_active_qdrant_proof_before_query_embedding(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import SearchFilters
    from app.services import context_graph

    state = populated_context_graph["state"]
    diagnostics = dict(state.diagnostics_json or {})
    diagnostics.pop("contextual_index_maintenance", None)
    state.diagnostics_json = diagnostics
    db_session.flush()
    provider_calls = 0

    async def forbidden_embed(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("query embedding must not run before graph admission")

    monkeypatch.setattr(context_graph.EmbeddingProvider, "embed_texts", forbidden_embed)

    with pytest.raises(
        context_graph.ActiveContextGraphAdmissionError,
        match="active_qdrant_freshness_proof_missing",
    ):
        await context_graph.layered_search(
            db_session,
            populated_context_graph["knowledge_base"].id,
            "old graph must be rejected",
            SearchFilters(),
            4,
        )
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_layered_search_rejects_stale_freshness_and_does_no_qdrant_io(
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from sqlalchemy import select

    from app.models import ContextGraphFreshness
    from app.schemas import SearchFilters
    from app.services import context_graph

    state = populated_context_graph["state"]
    freshness = db_session.scalar(
        select(ContextGraphFreshness).where(
            ContextGraphFreshness.context_graph_state_id == state.id,
            ContextGraphFreshness.layer == "contextual_index",
        )
    )
    assert freshness is not None
    freshness.is_stale = True
    freshness.stale_reasons_json = ["test_stale"]
    db_session.flush()
    fake_store = fake_model_stack["VectorStore"]
    fake_store.expected_snapshot_calls.clear()
    fake_store.unbounded_list_ids_calls = 0

    with pytest.raises(
        context_graph.ActiveContextGraphAdmissionError,
        match="context_graph_freshness_stale:contextual_index",
    ):
        await context_graph.layered_search(
            db_session,
            populated_context_graph["knowledge_base"].id,
            "stale graph must be rejected",
            SearchFilters(),
            4,
        )
    assert fake_store.expected_snapshot_calls == []
    assert fake_store.unbounded_list_ids_calls == 0


@pytest.mark.asyncio
async def test_layered_search_does_not_fall_back_to_old_coarse_graph_when_dense_scope_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
) -> None:
    from sqlalchemy import select

    from app.models import VectorRecord
    from app.schemas import SearchFilters
    from app.services import context_graph

    knowledge_base_id = populated_context_graph["knowledge_base"].id
    records = list(
        db_session.scalars(
            select(VectorRecord).where(
                VectorRecord.knowledge_base_id == knowledge_base_id
            )
        ).all()
    )
    assert records
    for record in records:
        record.vector_status = "stale"
    db_session.flush()
    query_embedding_calls = 0

    async def forbidden_embed(*args, **kwargs):
        nonlocal query_embedding_calls
        query_embedding_calls += 1
        raise AssertionError("stale dense scope must stop before query embedding")

    monkeypatch.setattr(context_graph.EmbeddingProvider, "embed_texts", forbidden_embed)

    with pytest.raises(
        context_graph.ActiveContextGraphAdmissionError,
        match="contextual_index_hash_stale",
    ):
        await context_graph.layered_search(
            db_session,
            knowledge_base_id,
            "must not use historical coarse graph",
            SearchFilters(),
            4,
            retrieval_granularity="coarse",
        )
    assert query_embedding_calls == 0


@pytest.mark.asyncio
async def test_agent_rejects_old_graph_before_any_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import AgentRequest, SearchFilters
    from app.services import agent_graph, context_graph

    state = populated_context_graph["state"]
    diagnostics = dict(state.diagnostics_json or {})
    diagnostics.pop("contextual_index_maintenance", None)
    state.diagnostics_json = diagnostics
    db_session.flush()
    provider_calls: list[str] = []

    async def forbidden_provider(*args, **kwargs):
        provider_calls.append("called")
        raise AssertionError("provider must not run before active graph admission")

    monkeypatch.setattr(agent_graph, "perceive_query_intent", forbidden_provider)
    monkeypatch.setattr(agent_graph, "propose_query_facets", forbidden_provider)
    monkeypatch.setattr(agent_graph, "propose_agent_plan", forbidden_provider)

    with pytest.raises(
        context_graph.ActiveContextGraphAdmissionError,
        match="active_qdrant_freshness_proof_missing",
    ):
        await agent_graph.run_agent(
            db_session,
            AgentRequest(
                knowledge_base_id=populated_context_graph["knowledge_base"].id,
                question="old graph agent admission",
                filters=SearchFilters(),
                top_k=4,
            ),
        )
    assert provider_calls == []


@pytest.mark.asyncio
async def test_normal_layered_search_uses_persisted_proof_without_qdrant_read_or_scroll(
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from app.schemas import SearchFilters
    from app.services.context_graph import layered_search

    fake_store = fake_model_stack["VectorStore"]
    fake_store.expected_snapshot_calls.clear()
    fake_store.unbounded_list_ids_calls = 0

    result = await layered_search(
        db_session,
        populated_context_graph["knowledge_base"].id,
        "Bayesian network factorization",
        SearchFilters(),
        4,
    )

    assert result.results
    assert fake_store.expected_snapshot_calls == []
    assert fake_store.unbounded_list_ids_calls == 0
