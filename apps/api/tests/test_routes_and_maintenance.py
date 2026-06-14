from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_context_graph_stats_and_layer_payloads(db_session, populated_context_graph):
    from app.services.context_graph import context_graph_stats, graph_layer_payload
    from app.schemas import GraphResponse

    kb = populated_context_graph["knowledge_base"]
    stats = context_graph_stats(db_session, kb.id)
    assert stats["counts"]["active_chunks"] >= 3
    assert stats["counts"]["mid_concepts"] >= 1
    assert "codebooks" not in (stats["diagnostics"]["relation"].get("rq_kmeans") or {})
    assert "path_by_chunk" not in (stats["diagnostics"]["relation"].get("rq_kmeans") or {})
    for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
        payload = graph_layer_payload(db_session, kb.id, layer, limit=20)
        GraphResponse.model_validate(payload)
        assert payload["knowledge_base_id"] == kb.id
        assert payload["graph_type"] == layer
        assert payload["counts"]
        assert "freshness" in payload
        assert "grounding" in payload


@pytest.mark.asyncio
async def test_knowledge_base_summary_includes_active_graph_state(db_session, populated_context_graph):
    from app.schemas import KnowledgeBaseSummary
    from app.services.ingestion import summarize_knowledge_base

    kb = populated_context_graph["knowledge_base"]
    payload = summarize_knowledge_base(db_session, kb)
    validated = KnowledgeBaseSummary.model_validate(payload)

    assert validated.chunk_count >= 3
    assert validated.active_chunk_count == validated.chunk_count
    assert validated.context_graph_state_id == populated_context_graph["state"].id
    assert validated.context_graph_hash == populated_context_graph["state"].context_graph_hash
    assert validated.stale_reason is None
    assert validated.has_parsed_chunks is True
    assert validated.can_full_reparse is True


@pytest.mark.asyncio
async def test_search_route_commits_retrieval_trace(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import GraphRetrievalStep, RetrievalTrace
    from app.routers.search import search
    from app.schemas import SearchFilters, SearchRequest

    kb = populated_context_graph["knowledge_base"]
    payload = await search(
        SearchRequest(knowledge_base_id=kb.id, query="Bayes theorem prior posterior", filters=SearchFilters(), top_k=3),
        db_session,
    )
    trace_id = payload["model_audit"]["retrieval_trace_id"]

    assert db_session.get(RetrievalTrace, trace_id) is not None
    assert db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == trace_id)) >= 4


def test_policy_reconcile_creates_context_graph_policy(db_session, sample_knowledge_base):
    from sqlalchemy import func, select

    from app.models import PolicyState
    from app.services.maintenance import reconcile_policy_state

    assert reconcile_policy_state(db_session, sample_knowledge_base.id) == 1
    db_session.commit()
    policy = db_session.scalar(select(PolicyState).where(PolicyState.knowledge_base_id == sample_knowledge_base.id))
    assert policy is not None
    assert policy.policy_family == "context_graph_bandit"
    assert "bridge_edge_exploration" in policy.weights_json


def test_runtime_check_payload_matches_response_schema(monkeypatch):
    from app.schemas import RuntimeCheckResponse
    from app.services import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "env_sync_status",
        lambda: {"synced": True, "missing_keys": [], "extra_keys": [], "deprecated_keys": [], "bom_keys": []},
    )
    monkeypatch.setattr(runtime_settings, "_reranker_runtime_status", lambda: {"enabled": False, "healthy": False})
    monkeypatch.setattr(runtime_settings, "_check_postgres", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_qdrant", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_redis", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_model_bridge", lambda: None)

    payload = runtime_settings.runtime_check_payload()
    validated = RuntimeCheckResponse.model_validate(payload)

    assert validated.env_sync["synced"] is True
    assert validated.infrastructure["postgres"] is True
    assert validated.blocking_issues == []


def test_runtime_env_sync_treats_legacy_runtime_keys_as_deprecated(monkeypatch, tmp_path):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text(
        "DATABASE_URL=sqlite:///runtime.db\n"
        "FIXED_CHUNK_SIZE_TOKENS=512\n"
        "FIXED_CHUNK_OVERLAP_TOKENS=80\n"
        "CONTEXT_PACKAGE_TOKEN_BUDGET=2400\n"
        "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES=4\n"
        "CHUNK_TOKEN_BUDGET=2400\n"
        "SEMANTIC_CHUNKING_ENABLED=false\n"
        "COMMUNITY_LOUVAIN_RESOLUTION=1.0\n"
        "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES=4\n"
        "INGESTION_FILE_CONCURRENCY=3\n"
        "RETRIEVAL_LAYER_ENABLED=true\n"
        "RETRIEVAL_CACHE_TTL_SECONDS=120\n"
        "CITATION_VERIFICATION_SAMPLE_MAX=3\n"
        "GRAPH_OVERVIEW_MAX_NODES=260\n",
        encoding="utf-8",
    )
    example_path.write_text(
        "DATABASE_URL=\n"
        "FIXED_CHUNK_SIZE_TOKENS=512\n"
        "FIXED_CHUNK_OVERLAP_TOKENS=80\n"
        "CONTEXT_PACKAGE_TOKEN_BUDGET=2400\n"
        "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES=4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    status = runtime_settings.env_sync_status()
    assert status["synced"] is True
    assert status["deprecated_keys"] == [
        "CHUNK_TOKEN_BUDGET",
        "CITATION_VERIFICATION_SAMPLE_MAX",
        "COMMUNITY_LOUVAIN_RESOLUTION",
        "GRAPH_OVERVIEW_MAX_NODES",
        "INGESTION_FILE_CONCURRENCY",
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "RETRIEVAL_LAYER_ENABLED",
        "SEMANTIC_CHUNKING_ENABLED",
        "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES",
    ]

    assert runtime_settings.normalize_env_file() is True
    cleaned = env_path.read_text(encoding="utf-8")
    assert "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES" not in cleaned
    assert "CHUNK_TOKEN_BUDGET" not in cleaned
    assert "SEMANTIC_CHUNKING_ENABLED" not in cleaned
    assert "COMMUNITY_LOUVAIN_RESOLUTION" not in cleaned
    assert "INGESTION_FILE_CONCURRENCY" not in cleaned
    assert "RETRIEVAL_LAYER_ENABLED" not in cleaned
    assert "RETRIEVAL_CACHE_TTL_SECONDS" not in cleaned
    assert "CITATION_VERIFICATION_SAMPLE_MAX" not in cleaned
    assert "GRAPH_OVERVIEW_MAX_NODES" not in cleaned


def test_model_settings_payload_uses_fixed_chunk_and_context_budget(monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("FIXED_CHUNK_SIZE_TOKENS", "512")
    monkeypatch.setenv("FIXED_CHUNK_OVERLAP_TOKENS", "80")
    monkeypatch.setenv("CONTEXT_PACKAGE_TOKEN_BUDGET", "2400")
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit-version")
    monkeypatch.setattr(runtime_settings, "_env_entries", lambda _path: {})

    payload = runtime_settings.model_settings_payload()

    assert payload["fixed_chunk_size_tokens"] == 512
    assert payload["fixed_chunk_overlap_tokens"] == 80
    assert payload["context_package_token_budget"] == 2400
    assert "chunk_token_budget" not in payload
    assert "semantic_chunking_enabled" not in payload


def test_profile_validation_ignores_legacy_strategy_fields():
    from app.services.strategy_profiles import validate_profile_payload

    payload, warnings = validate_profile_payload(
        {
            "schema_version": "strategy_profile_v3",
            "library_type": "academic",
            "ui_labels": {"knowledge_base": "Library"},
            "prompt_pack": {"no_context_answer_en": "No evidence."},
            "conversation_preferences": {"citation_strictness": "strict"},
            "schema_pack": {"entity_types": ["legacy"], "relation_types": ["legacy"]},
            "retrieval_strategy": {"top_k": 100},
            "quality_policy": {"fallback": True},
        }
    )

    assert set(payload) == {"schema_version", "library_type", "ui_labels", "prompt_pack", "conversation_preferences", "profile_hash"}
    assert payload["schema_version"] == "user_profile_v1"
    assert payload["ui_labels"]["knowledge_base"] == "Library"
    assert "schema_pack" not in payload
    assert any("schema_pack is ignored" in warning for warning in warnings)
    assert any("retrieval_strategy is ignored" in warning for warning in warnings)


def test_batch_summary_matches_response_schema(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.schemas import IngestionBatchSummary
    from app.services.ingestion import summarize_batch

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="parsing",
        total_files=2,
        processed_files=1,
        success_count=1,
        stats={
            "phase": "context_graph",
            "context_graph_phase": "mid_concepts",
            "current_file": "lesson.pdf",
            "ingestion_execution_mode": "celery",
        },
    )
    db_session.add(batch)
    db_session.commit()

    payload = summarize_batch(batch)
    validated = IngestionBatchSummary.model_validate(payload)

    assert validated.knowledge_base_id == sample_knowledge_base.id
    assert validated.current_phase == "context_graph:mid_concepts"
    assert validated.current_file == "lesson.pdf"
    assert validated.stats["context_graph_phase"] == "mid_concepts"


def test_celery_owned_batches_are_not_finalized_on_api_restart():
    from app.services.ingestion import batch_is_worker_owned

    assert batch_is_worker_owned({"ingestion_execution_mode": "celery"}) is True
    assert batch_is_worker_owned({"celery_task_id": "task-1"}) is True
    assert batch_is_worker_owned({"ingestion_execution_mode": "inline"}) is False


def test_context_graph_heartbeat_updates_batch_stats(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.services.context_graph import context_graph_batch_heartbeat

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="extracting_graph",
        total_files=1,
    )
    db_session.add(batch)
    db_session.commit()

    context_graph_batch_heartbeat(batch.id, "mid_concepts", {"relation_edges": 3})
    db_session.refresh(batch)

    assert batch.heartbeat_at is not None
    assert batch.stats["phase"] == "context_graph"
    assert batch.stats["context_graph_phase"] == "mid_concepts"
    assert batch.stats["context_graph_metrics"]["relation_edges"] == 3


def test_cleanup_stale_data_deletes_inactive_chunk_versions_only_with_explicit_flag(db_session, sample_knowledge_base):
    from sqlalchemy import func, select

    from app.models import (
        Chunk,
        ChunkContextText,
        ChunkCoordinate,
        ChunkSpan,
        ChunkStructureMapping,
        ChunkStructureNode,
        ChunkVersion,
        Document,
        DocumentVersion,
    )
    from app.services.maintenance import cleanup_stale_data

    sample_knowledge_base.current_chunk_version = 2
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Versioned document",
        source_path="versioned.md",
        source_type="markdown",
        checksum="v2",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    old_version = DocumentVersion(document_id=document.id, version=1, checksum="v1", storage_path="versioned.md", is_active=False)
    active_version = DocumentVersion(document_id=document.id, version=2, checksum="v2", storage_path="versioned.md", is_active=True)
    db_session.add_all([old_version, active_version])
    db_session.flush()
    old_chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=old_version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=4,
        char_start=0,
        char_end=12,
        text="old content",
        text_hash="old",
        state="inactive",
    )
    active_chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=active_version.id,
        chunk_version=2,
        chunk_index=0,
        token_start=0,
        token_end=4,
        char_start=0,
        char_end=12,
        text="new content",
        text_hash="new",
        state="active",
    )
    db_session.add_all(
        [
            ChunkVersion(knowledge_base_id=sample_knowledge_base.id, chunk_version=1, state_hash="old", state="active"),
            ChunkVersion(knowledge_base_id=sample_knowledge_base.id, chunk_version=2, state_hash="new", state="active"),
            old_chunk,
            active_chunk,
        ]
    )
    db_session.flush()
    old_node = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=old_version.id,
        node_type="section",
        title="Old",
    )
    db_session.add(old_node)
    db_session.flush()
    db_session.add_all(
        [
            ChunkSpan(chunk_id=old_chunk.id, document_version_id=old_version.id, char_start=0, char_end=12, token_start=0, token_end=4),
            ChunkCoordinate(chunk_id=old_chunk.id, document_version_id=old_version.id, page_number=1),
            ChunkContextText(chunk_id=old_chunk.id, raw_text="old content", contextual_text="old content", context_hash="old"),
            ChunkStructureMapping(chunk_id=old_chunk.id, structure_node_id=old_node.id, document_version_id=old_version.id),
        ]
    )
    old_chunk_id = old_chunk.id
    active_chunk_id = active_chunk.id
    old_version_id = old_version.id
    db_session.commit()

    dry_run = cleanup_stale_data(
        db_session,
        sample_knowledge_base.id,
        sample_knowledge_base.name,
        dry_run=True,
        delete_inactive_chunks=True,
    )
    assert dry_run["inactive_chunks"] == 1
    assert dry_run["inactive_document_versions"] == 1
    assert db_session.get(Chunk, old_chunk.id) is not None

    applied = cleanup_stale_data(
        db_session,
        sample_knowledge_base.id,
        sample_knowledge_base.name,
        dry_run=False,
        delete_inactive_chunks=True,
    )

    assert applied["deleted_inactive"]["chunks"] == 1
    assert applied["deleted_inactive"]["document_versions"] == 1
    assert applied["deleted_inactive"]["chunk_versions"] == 1
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.id == old_chunk_id)) == 0
    assert db_session.scalar(select(func.count(DocumentVersion.id)).where(DocumentVersion.id == old_version_id)) == 0
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.id == active_chunk_id)) == 1
    current_chunk_version = db_session.scalar(select(ChunkVersion).where(ChunkVersion.knowledge_base_id == sample_knowledge_base.id))
    assert current_chunk_version is not None
    assert current_chunk_version.chunk_version == 2
    assert current_chunk_version.state == "active"


def test_target_chunk_version_uses_active_max_after_interrupted_batch():
    from app.services.ingestion import target_chunk_version

    assert target_chunk_version(current_version=0, active_max_version=1, full_reparse=True) == 2
    assert target_chunk_version(current_version=0, active_max_version=1, full_reparse=False) == 1
    assert target_chunk_version(current_version=3, active_max_version=1, full_reparse=True) == 4


def test_collect_source_documents_deduplicates_same_named_copies(tmp_path):
    from app.services.ingestion import collect_source_documents

    original = tmp_path / "20260611" / "lesson.pdf"
    copied = tmp_path / "20260613" / "lesson.pdf"
    same_content_other_name = tmp_path / "20260613" / "lesson-copy.pdf"
    original.parent.mkdir()
    copied.parent.mkdir()
    original.write_bytes(b"same pdf bytes")
    copied.write_bytes(b"same pdf bytes")
    same_content_other_name.write_bytes(b"same pdf bytes")

    paths = collect_source_documents(tmp_path)

    assert original in paths
    assert copied not in paths
    assert same_content_other_name in paths
