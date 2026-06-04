from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select


def make_chunk(chunk_id: str, document_id: str, content: str, content_kind: str = "pdf_page", metadata: dict | None = None):
    return SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        content=content,
        chapter="Lecture 1",
        source_type="pdf",
        metadata_json={"content_kind": content_kind, **(metadata or {})},
    )


def write_test_env(monkeypatch, tmp_path, **entries: str):
    from app.core import config

    workspace = tmp_path / "workspace"
    app_dir = workspace / "apps" / "api" / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(config, "APP_DIR", app_dir)
    env_entries = {
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "DATA_ROOT": (tmp_path / "data").as_posix(),
        "GRAPH_EXTRACTION_MAX_MODEL_CALLS_PER_RUN": "1000",
        "GRAPH_EXTRACTION_MAX_INPUT_TOKENS_PER_RUN": "999999",
        "GRAPH_EXTRACTION_MIN_MARGINAL_GAIN": "0",
        "GRAPH_EXTRACTION_STALL_ROUNDS": "20",
        **entries,
    }
    (workspace / ".env").write_text("\n".join(f"{key}={value}" for key, value in env_entries.items()) + "\n", encoding="utf-8")
    config.get_settings.cache_clear()


def test_adaptive_graph_extraction_plan_has_no_fixed_chunk_cap(no_fallback_env, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services.concept_graph import plan_adaptive_graph_extraction_chunks

    write_test_env(monkeypatch, tmp_path, GRAPH_EXTRACTION_SOFT_START_BUDGET="3")
    monkeypatch.delenv("GRAPH_EXTRACTION_MAX_MODEL_CALLS_PER_RUN", raising=False)
    monkeypatch.delenv("GRAPH_EXTRACTION_MAX_INPUT_TOKENS_PER_RUN", raising=False)
    get_settings.cache_clear()
    chunks = [
        make_chunk(f"a-{index}", "doc-a", "Residual Network is defined as remaining capacity and augmenting path evidence. " * 8)
        for index in range(6)
    ] + [
        make_chunk(f"b-{index}", "doc-b", "1 2 3 4 5 6 7 8", metadata={"quality_action": "evidence_only", "quality_retain": True})
        for index in range(2)
    ]
    for chunk in chunks[:6]:
        chunk.chapter = "Dense Chapter"
        chunk.section = f"Dense Section {chunk.id}"
    for chunk in chunks[6:]:
        chunk.chapter = "Sparse Chapter"
        chunk.section = "Sparse"

    plan = plan_adaptive_graph_extraction_chunks(chunks)

    assert len(plan.selected_chunk_ids) > 3
    assert plan.budget["soft_start_budget"] == 3
    assert plan.coverage["chapters"]["covered"] >= 2


@pytest.mark.asyncio
async def test_llm_graph_extraction_respects_model_request_timeout(monkeypatch):
    from app.services import concept_graph

    class SlowProvider:
        async def extract_graph_payload(self, *_args, **_kwargs):
            await asyncio.sleep(1)
            return {"concepts": [], "relations": []}

    monkeypatch.setattr(concept_graph, "ChatProvider", SlowProvider)
    monkeypatch.setattr(
        concept_graph,
        "get_settings",
        lambda: SimpleNamespace(graph_extraction_concurrency=1, model_request_timeout_seconds=0.01),
    )

    payloads, errors = await concept_graph.extract_llm_graph_payloads(
        [make_chunk("chunk-timeout", "doc-1", "centrality and modularity")]
    )

    assert payloads == {}
    assert "chunk-timeout" in errors
    assert "model request exceeded" in errors["chunk-timeout"]


@pytest.mark.asyncio
async def test_llm_graph_extraction_batch_timeout_is_not_applied_above_chunk_timeouts(monkeypatch):
    from app.services import concept_graph

    async def slow_batch(*_args, **_kwargs):
        await asyncio.sleep(1)
        return {}, {}

    monkeypatch.setattr(concept_graph, "run_llm_graph_extraction", slow_batch)

    payloads, errors = await concept_graph.run_llm_graph_extraction_with_timeout(
        [make_chunk("chunk-batch-timeout", "doc-1", "centrality and modularity")],
        timeout_seconds=0.01,
    )

    assert payloads == {}
    assert errors == {}


def test_adaptive_graph_extraction_plan_respects_optional_model_call_budget(no_fallback_env, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services.concept_graph import plan_adaptive_graph_extraction_chunks

    write_test_env(monkeypatch, tmp_path, GRAPH_EXTRACTION_MAX_MODEL_CALLS_PER_RUN="2")
    get_settings.cache_clear()
    chunks = [
        make_chunk(f"doc-{index}", f"doc-{index}", "Maximum Flow is defined by feasible flow and residual network. " * 10)
        for index in range(5)
    ]

    plan = plan_adaptive_graph_extraction_chunks(chunks)

    assert len(plan.selected_chunk_ids) == 2
    assert plan.stop_reason == "model_call_budget_reached"


def test_adaptive_specificity_threshold_uses_distribution_with_fallback():
    from app.services.concept_graph import adaptive_specificity_threshold

    fallback = adaptive_specificity_threshold([0.41, 0.52, 0.66])
    assert fallback["enabled"] is False
    assert fallback["threshold"] == 0.35
    assert fallback["fallback_reason"] == "insufficient_samples"

    profile = adaptive_specificity_threshold([0.42] * 5 + [0.48] * 10 + [0.72] * 10)
    assert profile["enabled"] is True
    assert 0.42 <= profile["threshold"] <= 0.48
    assert profile["p25"] <= profile["p50"] <= profile["p75"]


def test_record_entity_mention_merges_duplicate_pending_mentions(db_session, sample_course, indexed_chunks):
    from app.models import EntityMention
    from app.services.concept_graph import StagedConcept, _record_entity_mention

    _document, chunks = indexed_chunks
    group = StagedConcept(
        key="edgelist::concept",
        name="Edgelist",
        concept_type="concept",
        confidence=0.8,
        evidence_spans={"edge list representation"},
    )

    _record_entity_mention(db_session, course_id=sample_course.id, chunk=chunks[0], group=group, surface="Edgelist")
    group.confidence = 0.92
    group.evidence_spans.add("edgelist format")
    _record_entity_mention(db_session, course_id=sample_course.id, chunk=chunks[0], group=group, surface="Edgelist")

    db_session.flush()
    mentions = db_session.query(EntityMention).filter_by(course_id=sample_course.id, chunk_id=chunks[0].id, surface="Edgelist", entity_type="concept").all()

    assert len(mentions) == 1
    assert mentions[0].confidence == 0.92
    assert set(mentions[0].evidence_spans) == {"edge list representation", "edgelist format"}


@pytest.mark.asyncio
async def test_extract_llm_graph_payloads_uses_configured_concurrency(no_fallback_env, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import concept_graph

    observed_values: list[int] = []

    class RecordingSemaphore:
        def __init__(self, value: int):
            observed_values.append(value)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    write_test_env(monkeypatch, tmp_path, GRAPH_EXTRACTION_CONCURRENCY="4")
    get_settings.cache_clear()
    monkeypatch.setattr(concept_graph.asyncio, "Semaphore", RecordingSemaphore)

    await concept_graph.extract_llm_graph_payloads([])

    assert observed_values == [4]


def test_adaptive_graph_extraction_run_reuses_completed_chunk_payloads(db_session, sample_course, indexed_chunks, no_fallback_env):
    from app.models import GraphExtractionChunkTask
    from app.services.concept_graph import (
        create_graph_extraction_run_from_plan,
        plan_adaptive_graph_extraction_chunks,
    )

    _document, chunks = indexed_chunks
    plan = plan_adaptive_graph_extraction_chunks(chunks)
    first = create_graph_extraction_run_from_plan(
        db_session,
        course_id=sample_course.id,
        batch_id=None,
        chunks=chunks,
        plan=plan,
        profile_version="quality_profile_v1:test",
    )
    db_session.flush()
    first_task = db_session.query(GraphExtractionChunkTask).filter_by(run_id=first.id).first()
    first_task.status = "completed"
    first_task.payload_json = {"concepts": [{"name": "Residual Network", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.9}], "relations": []}
    db_session.commit()

    second = create_graph_extraction_run_from_plan(
        db_session,
        course_id=sample_course.id,
        batch_id=None,
        chunks=chunks,
        plan=plan,
        profile_version="quality_profile_v1:test",
    )
    db_session.flush()
    reused = db_session.query(GraphExtractionChunkTask).filter_by(run_id=second.id, status="completed").all()

    assert reused
    assert second.stats_json["reused_completed_chunks"] >= 1


def test_resumable_graph_extraction_ignores_terminal_batch_runs(db_session, sample_course, indexed_chunks):
    from app.models import GraphExtractionChunkTask, GraphExtractionRun, IngestionBatch
    from app.services.concept_graph import has_resumable_graph_extraction

    _document, chunks = indexed_chunks
    failed_batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="unit",
        trigger_source="rebuild_graph",
        status="failed",
    )
    db_session.add(failed_batch)
    db_session.flush()
    stale_run = GraphExtractionRun(
        course_id=sample_course.id,
        batch_id=failed_batch.id,
        strategy="adaptive_best_first",
        status="running",
    )
    db_session.add(stale_run)
    db_session.flush()
    db_session.add(
        GraphExtractionChunkTask(
            run_id=stale_run.id,
            course_id=sample_course.id,
            chunk_id=chunks[0].id,
            chunk_hash="stale",
            status="pending",
        )
    )
    db_session.commit()

    assert has_resumable_graph_extraction(db_session, sample_course.id) is False

    active_batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="unit",
        trigger_source="rebuild_graph",
        status="extracting_graph",
    )
    db_session.add(active_batch)
    db_session.flush()
    active_run = GraphExtractionRun(
        course_id=sample_course.id,
        batch_id=active_batch.id,
        strategy="adaptive_best_first",
        status="running",
    )
    db_session.add(active_run)
    db_session.flush()
    db_session.add(
        GraphExtractionChunkTask(
            run_id=active_run.id,
            course_id=sample_course.id,
            chunk_id=chunks[1].id,
            chunk_hash="active",
            status="pending",
        )
    )
    db_session.commit()

    assert has_resumable_graph_extraction(db_session, sample_course.id) is True


@pytest.mark.asyncio
async def test_rebuild_course_graph_reports_real_llm_selection_stats(db_session, sample_course, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.models import Chunk, Document, DocumentVersion
    from app.services import concept_graph

    write_test_env(monkeypatch, tmp_path, GRAPH_EXTRACTION_MAX_MODEL_CALLS_PER_RUN="5")
    get_settings.cache_clear()

    for document_index in range(3):
        document = Document(
            course_id=sample_course.id,
            title=f"Lecture {document_index + 1}",
            source_path=f"Lecture {document_index + 1}.pdf",
            source_type="pdf",
            tags=["20260425"] if document_index == 0 else [f"Lecture {document_index + 1}"],
            checksum=f"checksum-{document_index}",
        )
        db_session.add(document)
        db_session.flush()
        version = DocumentVersion(
            document_id=document.id,
            version=1,
            checksum=document.checksum,
            storage_path=document.source_path,
            extracted_path=None,
            is_active=True,
        )
        db_session.add(version)
        db_session.flush()
        for chunk_index in range(3):
            db_session.add(
                Chunk(
                    course_id=sample_course.id,
                    document_id=document.id,
                    document_version_id=version.id,
                    content=f"Bayesian inference posterior prior likelihood {document_index} {chunk_index} " * 20,
                    snippet="Bayesian inference posterior prior likelihood",
                    chapter="20260425" if document_index == 0 else document.tags[0],
                    section="Topic",
                    source_type="pdf",
                    metadata_json={"content_kind": "pdf_page"},
                    embedding_status="ready",
                )
            )
    db_session.commit()

    async def fake_upsert(db, course_id, chunk, use_llm=True, llm_payload=None):
        return (1 if use_llm else 0, 1 if use_llm else 0)

    async def fake_extract_payloads(chunks, concurrency=4):
        return {chunk.id: {"concepts": [{"name": f"Concept {chunk.id}", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.8}], "relations": []} for chunk in chunks}

    async def fake_community_summaries(db, course_id, batch_id=None):
        return {
            "community_summary_count": 0,
            "community_summary_prompt_version": "community_summary_v1",
        }

    monkeypatch.setattr(concept_graph, "upsert_concepts_from_chunk", fake_upsert)
    monkeypatch.setattr(concept_graph, "extract_llm_graph_payloads", fake_extract_payloads)
    monkeypatch.setattr(concept_graph, "rebuild_graph_community_summaries", fake_community_summaries)
    monkeypatch.setattr(concept_graph, "_backup_course_graph_tables", lambda db, cid: None)

    stats = await concept_graph.rebuild_course_graph(db_session, sample_course.id)

    assert stats["graph_extraction_strategy"] == "adaptive_best_first"
    assert stats["graph_extraction_selected_chunks"] == 5
    assert stats["graph_llm_selected_chunks"] == 5
    assert stats["graph_llm_success_chunks"] == 5
    assert stats["graph_llm_source_documents"] == 3
    assert stats["graph_probe_chunks"] == 3
    assert stats["graph_probe_success_chunks"] == 3
    assert stats["graph_probe_failed_chunks"] == 0
    refreshed_document = db_session.scalar(select(Document).where(Document.title == "Lecture 1"))
    refreshed_chunk = db_session.scalar(select(Chunk).where(Chunk.document_id == refreshed_document.id))
    assert refreshed_document.tags == ["Lecture 1"]
    assert refreshed_chunk.chapter == "Lecture 1"


@pytest.mark.asyncio
async def test_auto_hpo_runs_after_graph_upsert_before_enrich(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.core.config import get_settings
    from app.services import concept_graph

    _document, _chunks = indexed_chunks
    monkeypatch.setenv("ENABLE_AUTO_HPO", "true")
    get_settings.cache_clear()
    order: list[str] = []

    async def fake_extract_payloads(chunks, **_kwargs):
        order.append("extract")
        return {
            chunk.id: {
                "concepts": [{"name": f"Concept {chunk.id}", "confidence": 0.9}],
                "relations": [],
            }
            for chunk in chunks
        }

    async def fake_upsert_graph_candidates(db, course_id, chunks, *, llm_payloads, run_llm_merge=False):
        order.append("upsert")
        assert llm_payloads
        return {
            "created_concepts": 2,
            "relations": 0,
            "llm_success_chunks": len(llm_payloads),
            "validation_warnings": {},
            "rejected_concepts": 0,
            "graph_concept_specificity_threshold": None,
            "graph_concept_specificity_threshold_audit": {},
            "llm_verified_merges": 0,
        }

    async def fake_hpo(db, *, course_id, batch_id, probe_chunks, payloads, baseline_context_chunks, run_hpo=None):
        order.append("hpo")
        assert payloads
        assert probe_chunks
        assert baseline_context_chunks
        return {"hpo_auto_enabled": True, "hpo_status": "completed"}

    async def fake_enrich(db, course_id):
        order.append("enrich")
        return {"concepts_analyzed": 2}

    async def fake_community_summaries(db, course_id, batch_id=None):
        return {
            "community_summary_count": 0,
            "community_summary_prompt_version": "community_summary_v1",
        }

    monkeypatch.setattr(concept_graph, "extract_llm_graph_payloads", fake_extract_payloads)
    monkeypatch.setattr(concept_graph, "upsert_graph_candidates_from_chunks", fake_upsert_graph_candidates)
    monkeypatch.setattr(concept_graph, "_run_auto_hpo_after_graph_upsert", fake_hpo)
    monkeypatch.setattr(concept_graph, "enrich_course_graph", fake_enrich)
    monkeypatch.setattr(concept_graph, "rebuild_graph_community_summaries", fake_community_summaries)
    monkeypatch.setattr(concept_graph, "_backup_course_graph_tables", lambda db, cid: None)

    stats = await concept_graph.rebuild_course_graph(db_session, sample_course.id)

    assert order == ["extract", "upsert", "hpo", "enrich"]
    assert stats["hpo_status"] == "completed"


def test_choose_graph_probe_chunks_samples_short_middle_long():
    from app.services.concept_graph import choose_graph_probe_chunks

    chunks = [
        make_chunk("short", "doc-a", "x"),
        make_chunk("middle", "doc-a", "x" * 50),
        make_chunk("long", "doc-a", "x" * 100),
        make_chunk("tiny", "doc-a", "xx"),
        make_chunk("wide", "doc-a", "x" * 75),
    ]

    probes = choose_graph_probe_chunks(chunks)

    assert [chunk.id for chunk in probes] == ["short", "middle", "long"]


@pytest.mark.asyncio
async def test_rebuild_graph_community_summaries_persists_active_summary(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.models import Concept, ConceptRelation, GraphCommunitySummary
    from app.services import concept_graph
    from app.services.embeddings import ChatProvider

    _, chunks = indexed_chunks
    source = Concept(
        course_id=sample_course.id,
        canonical_name="Degree Centrality",
        normalized_name="degree centrality",
        evidence_count=2,
        community_louvain=7,
        graph_rank_score=0.9,
    )
    target = Concept(
        course_id=sample_course.id,
        canonical_name="Shortest Paths",
        normalized_name="shortest paths",
        evidence_count=2,
        community_louvain=7,
        graph_rank_score=0.8,
    )
    db_session.add_all([source, target])
    db_session.flush()
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=source.id,
            target_concept_id=target.id,
            target_name=target.canonical_name,
            relation_type="used_for",
            evidence_chunk_id=chunks[0].id,
            confidence=0.9,
            weight=0.9,
            relation_source="llm",
            is_validated=True,
            metadata_json={"evidence_source_match": True, "evidence_target_match": True},
        )
    )
    db_session.commit()

    async def fake_classify_json(self, system_prompt, user_prompt, fallback=None):
        return {
            "summary": "Centrality community summary.",
            "key_concepts": ["Degree Centrality", "Shortest Paths"],
            "routing_hints": ["centrality"],
            "quality_notes": [],
        }

    monkeypatch.setattr(ChatProvider, "classify_json", fake_classify_json)

    stats = await concept_graph.rebuild_graph_community_summaries(db_session, sample_course.id)
    db_session.commit()

    assert stats["community_summary_count"] == 1
    summary = db_session.query(GraphCommunitySummary).filter_by(course_id=sample_course.id, is_active=True).one()
    assert summary.community_id == 7
    assert summary.summary == "Centrality community summary."
    assert summary.representative_chunk_ids == [chunks[0].id]


@pytest.mark.asyncio
async def test_rebuild_course_graph_fails_fast_when_probe_fails(db_session, sample_course, monkeypatch):
    from app.core.config import get_settings
    from app.models import Chunk, Document, DocumentVersion
    from app.services import concept_graph

    monkeypatch.setenv("GRAPH_EXTRACTION_SOFT_START_BUDGET", "5")
    get_settings.cache_clear()

    document = Document(
        course_id=sample_course.id,
        title="Lecture 1",
        source_path="Lecture 1.pdf",
        source_type="pdf",
        tags=["Lecture 1"],
        checksum="checksum-probe",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        extracted_path=None,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    for index in range(5):
        db_session.add(
            Chunk(
                course_id=sample_course.id,
                document_id=document.id,
                document_version_id=version.id,
                content=f"Bayesian graph probe failure {index} " * 20,
                snippet="Bayesian graph probe failure",
                chapter="Lecture 1",
                section="Topic",
                source_type="pdf",
                metadata_json={"content_kind": "pdf_page"},
                embedding_status="ready",
            )
        )
    db_session.commit()

    calls = 0

    async def fake_extract_payloads(chunks, concurrency=4):
        nonlocal calls
        calls += 1
        return {}, {chunk.id: "ReadTimeout: ReadTimeout('')" for chunk in chunks}

    monkeypatch.setattr(concept_graph, "extract_llm_graph_payloads", fake_extract_payloads)

    with pytest.raises(RuntimeError, match="轻量预检失败"):
        await concept_graph.rebuild_course_graph(db_session, sample_course.id)

    assert calls == 1


@pytest.mark.asyncio
async def test_rebuild_course_graph_rolls_back_failed_backup_before_graph_delete(db_session, sample_course, monkeypatch):
    from sqlalchemy import text

    from app.models import Chunk, Concept, ConceptRelation, Document, DocumentVersion
    from app.services import concept_graph

    document = Document(
        course_id=sample_course.id,
        title="Lecture 1",
        source_path="Lecture 1.pdf",
        source_type="pdf",
        tags=["Lecture 1"],
        checksum="backup-failure-checksum",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        extracted_path=None,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Residual Network is defined as remaining capacity in maximum flow. " * 8,
        snippet="Residual Network is defined as remaining capacity.",
        chapter="Lecture 1",
        section="Residual Network",
        source_type="pdf",
        metadata_json={"content_kind": "pdf_page"},
        embedding_status="ready",
    )
    chunk_two = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Maximum Flow uses residual networks to find augmenting paths. " * 8,
        snippet="Maximum Flow uses residual networks.",
        chapter="Lecture 1",
        section="Maximum Flow",
        source_type="pdf",
        metadata_json={"content_kind": "pdf_page"},
        embedding_status="ready",
    )
    old_source = Concept(course_id=sample_course.id, canonical_name="Old Source", normalized_name="old source")
    old_target = Concept(course_id=sample_course.id, canonical_name="Old Target", normalized_name="old target")
    db_session.add_all([chunk, chunk_two, old_source, old_target])
    db_session.flush()
    old_relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=old_source.id,
        target_concept_id=old_target.id,
        target_name=old_target.canonical_name,
        relation_type="defined_by",
        evidence_chunk_id=chunk.id,
        confidence=0.9,
        relation_source="llm",
    )
    db_session.add(old_relation)
    db_session.commit()

    async def fake_extract_payloads(chunks, concurrency=4, batch_id=None):
        return {
            item.id: {
                "concepts": [
                    {"name": "Residual Network", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.9},
                    {"name": "Maximum Flow", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.9},
                ],
                "relations": [
                    {"source": "Residual Network", "target": "Maximum Flow", "relation_type": "used_for", "confidence": 0.9},
                ],
            }
            for item in chunks
        }, {}

    def failing_backup(db, course_id):
        db.execute(text("SELECT * FROM definitely_missing_graph_backup_table"))

    async def fake_community_summaries(db, course_id, batch_id=None):
        return {
            "community_summary_count": 0,
            "community_summary_prompt_version": "community_summary_v1",
        }

    rollback_count = 0
    original_rollback = db_session.rollback

    def tracked_rollback():
        nonlocal rollback_count
        rollback_count += 1
        original_rollback()

    monkeypatch.setattr(concept_graph, "extract_llm_graph_payloads", fake_extract_payloads)
    monkeypatch.setattr(concept_graph, "_backup_course_graph_tables", failing_backup)
    monkeypatch.setattr(concept_graph, "rebuild_graph_community_summaries", fake_community_summaries)

    with pytest.raises(RuntimeError, match="图谱备份失败"):
        await concept_graph.rebuild_course_graph(db_session, sample_course.id)


def test_invalid_chapter_refs_are_not_added_to_concepts(db_session, sample_course):
    from app.services.concept_graph import get_or_create_concept

    concept, _ = get_or_create_concept(
        db_session,
        sample_course.id,
        name="Posterior Distribution",
        chapter="20260425",
        summary="A distribution after observing evidence.",
        aliases=[],
        concept_type="concept",
        importance_score=0.9,
    )

    assert concept.chapter_refs == []


def test_document_chapter_label_prefers_canonical_filename_over_stale_tags(sample_course):
    from app.models import Document
    from app.services.concept_graph import document_chapter_label

    lab_document = Document(
        course_id=sample_course.id,
        title="Labs solutions",
        source_path="fixtures/course-a/storage/20260425/Labs solutions.pdf",
        source_type="pdf",
        tags=["Labs solutions"],
        checksum="checksum",
    )
    visualizer_document = Document(
        course_id=sample_course.id,
        title="graph_algorithms_visualizer",
        source_path="fixtures/course-a/storage/20260425/graph_algorithms_visualizer.html",
        source_type="html",
        tags=["graph algorithms visualizer"],
        checksum="checksum",
    )

    assert document_chapter_label(lab_document, "Course A") == "Lab Solutions"
    assert document_chapter_label(visualizer_document, "Course A") == "Reference"


def test_merge_graph_candidates_ignores_non_text_model_fields():
    from app.services.concept_graph import merge_graph_candidates

    merged = merge_graph_candidates(
        {
            "concepts": [
                {
                    "name": 123,
                    "aliases": ["Numeric Alias"],
                    "summary": "invalid concept name should be ignored",
                    "concept_type": "concept",
                    "importance_score": 0.9,
                },
                {
                    "name": "Posterior Distribution",
                    "aliases": [456, "Posterior"],
                    "summary": 789,
                    "concept_type": 123,
                    "importance_score": "bad",
                },
            ],
            "relations": [
                {"source": 123, "target": "Posterior Distribution", "relation_type": "related_to", "confidence": 0.9},
                {"source": "Posterior Distribution", "target": "Bayesian Inference", "relation_type": 5, "confidence": "bad"},
                {"source": "Posterior Distribution", "target": "Bayesian Inference", "relation_type": "related_to", "confidence": "bad"},
            ],
        },
        {
            "concepts": [
                {"name": "Bayesian Inference", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.7}
            ],
            "relations": [],
        },
    )

    assert [concept["name"] for concept in merged["concepts"]] == ["Bayesian Inference", "Posterior Distribution"]
    posterior = next(concept for concept in merged["concepts"] if concept["name"] == "Posterior Distribution")
    assert set(posterior["aliases"]) == {"Posterior", "Posterior Distribution"}
    assert posterior["concept_type"] == "concept"
    assert posterior["importance_score"] == 0.0
    assert len(merged["relations"]) == 1
    assert merged["relations"][0]["source"] == "Posterior Distribution"
    assert merged["relations"][0]["target"] == "Bayesian Inference"
    assert merged["relations"][0]["source_key"] == "posterior distribution::concept"
    assert merged["relations"][0]["target_key"] == "bayesian inference::concept"


def test_merge_graph_candidates_treats_non_mapping_payload_as_empty():
    from app.services.concept_graph import merge_graph_candidates

    merged = merge_graph_candidates(
        ["not", "a", "mapping"],
        {
            "concepts": [
                {"name": "Bayesian Inference", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.7}
            ],
            "relations": [],
        },
    )

    assert len(merged["concepts"]) == 1
    assert merged["concepts"][0]["name"] == "Bayesian Inference"
    assert merged["concepts"][0]["aliases"] == ["Bayesian Inference"]
    assert merged["concepts"][0]["concept_type"] == "concept"
    assert merged["concepts"][0]["entity_type"] == "concept"
    assert merged["relations"] == []


def test_graph_extraction_payload_normalizes_model_enum_aliases():
    from app.schemas import GraphExtractionPayload

    payload = GraphExtractionPayload.model_validate(
        {
            "concepts": [
                {"name": "Equilibrium", "concept_type": "Definition", "entity_type": "Definition", "importance_score": 8},
                {"name": "Price of Anarchy", "concept_type": "Metric", "entity_type": "Metric", "importance_score": "7"},
                {"name": "Mechanism Design", "concept_type": "Application", "entity_type": "Application", "importance_score": 0.6},
                {"name": "Auction Problem", "concept_type": "Problem Type", "entity_type": "Problem Type", "importance_score": 0.6},
            ],
            "relations": [
                {"source": "Equilibrium", "target": "Price of Anarchy", "relation_type": "defines", "confidence": 8},
                {"source": "Mechanism Design", "target": "Auction Problem", "relation_type": "relates_to", "confidence": 0.7},
                {"source": "Mechanism Design", "target": "Equilibrium", "relation_type": "extends", "confidence": 0.7},
            ],
        }
    ).model_dump()

    assert [concept["concept_type"] for concept in payload["concepts"]] == ["definition", "metric", "concept", "problem_type"]
    assert [concept["importance_score"] for concept in payload["concepts"][:2]] == [0.8, 0.7]
    assert [relation["relation_type"] for relation in payload["relations"]] == ["defined_by", "related_to", "derives_from"]
    assert payload["relations"][0]["confidence"] == 0.8


def test_concept_quality_filters_structural_noise_generically():
    from app.services.concept_graph import concept_quality, is_valid_concept

    context_terms = {"Unit Test Course", "Lecture 3", "Networks overview.pdf"}
    rejected = [
        "Chapter 1",
        "Lecture 3",
        "Week 2 Solutions",
        "20260425",
        "slides.pdf",
        "data/storage/course/file.md",
        "cafÃ©",
        "Homework",
    ]
    for value in rejected:
        assert concept_quality(value, context_terms)["valid"] is False, value

    accepted = [
        "Bayesian Inference",
        "Menger's Theorem",
        "PageRank",
        "O(n log n)",
        "P vs NP",
        "残差网络",
    ]
    for value in accepted:
        assert is_valid_concept(value, context_terms), value


def test_concept_normalization_handles_general_variants():
    from app.services.concept_graph import normalize_concept_name

    assert normalize_concept_name("Minimum-Spanning_Trees") == "minimum spanning tree"
    assert normalize_concept_name("PageRanks") == "pagerank"
    assert normalize_concept_name("Normal–Normal Conjugacy") == "normal normal conjugacy"


def test_concept_gate_requires_batch_evidence_and_specificity():
    from app.services.concept_graph import StagedConcept, concept_gate_decision

    chunks = [
        make_chunk("c1", "doc-a", "Residual Network is defined as remaining capacity in maximum flow."),
        make_chunk("c2", "doc-b", "Residual Network is used by augmenting path algorithms."),
    ]
    for chunk in chunks:
        chunk.section = "Residual Network"
        chunk.snippet = chunk.content

    accepted_group = StagedConcept(
        key="residual network",
        name="Residual Network",
        importance_score=0.84,
        chunk_ids={"c1", "c2"},
        heading_hits=1,
        definition_hits=1,
    )
    accepted, audit = concept_gate_decision(accepted_group, chunks)

    assert accepted is True
    assert audit["specificity_score"] >= 0.55
    assert audit["evidence_chunk_count"] == 2

    generic_group = StagedConcept(
        key="algorithm",
        name="Algorithm",
        importance_score=0.95,
        chunk_ids={"c1", "c2"},
        heading_hits=1,
        definition_hits=1,
    )
    generic_accepted, generic_audit = concept_gate_decision(generic_group, chunks)

    assert generic_accepted is False
    assert generic_audit["gate_reason"] == "generic_low_specificity"

    singleton_group = StagedConcept(
        key="minimum cut::theorem",
        name="Minimum Cut",
        concept_type="theorem",
        importance_score=0.95,
        confidence=0.95,
        chunk_ids={"c1"},
        heading_hits=0,
        definition_hits=1,
    )
    singleton_accepted, singleton_audit = concept_gate_decision(singleton_group, chunks)

    assert singleton_accepted is True
    assert singleton_audit["gate_reason"] == "strong_singleton_evidence"

    single_chunk_group = StagedConcept(
        key="pagerank",
        name="PageRank",
        concept_type="metric",
        importance_score=0.7,
        confidence=0.7,
        chunk_ids={"c1"},
        heading_hits=0,
        definition_hits=0,
    )
    single_chunk_accepted, single_chunk_audit = concept_gate_decision(single_chunk_group, chunks, specificity_threshold=0.35)

    assert single_chunk_accepted is True
    assert single_chunk_audit["gate_reason"] == "single_chunk_domain_term"

    heading_singleton_group = StagedConcept(
        key="lecture 1::concept",
        name="Lecture 1",
        concept_type="concept",
        importance_score=0.95,
        confidence=0.95,
        chunk_ids={"c1"},
        heading_hits=1,
        definition_hits=1,
    )
    heading_singleton_accepted, _heading_audit = concept_gate_decision(heading_singleton_group, chunks)

    assert heading_singleton_accepted is False


def test_staged_merge_combines_obvious_alias_variants():
    from app.services.concept_graph import StagedConcept, merge_staged_concept_groups

    groups = {
        "maximum flow": StagedConcept(
            key="maximum flow",
            name="Maximum Flow",
            aliases={"Max Flow"},
            chunk_ids={"c1"},
        ),
        "maximum flow problem": StagedConcept(
            key="maximum flow problem",
            name="Maximum-Flow Problem",
            aliases=set(),
            chunk_ids={"c2"},
        ),
    }

    merged, key_map = merge_staged_concept_groups(groups)

    assert len(merged) == 1
    assert key_map["maximum flow problem"] == "maximum flow"
    group = next(iter(merged.values()))
    assert group.chunk_ids == {"c1", "c2"}
    assert "Maximum-Flow Problem" in group.aliases


@pytest.mark.asyncio
async def test_llm_verified_staged_merge_requires_structured_confidence(monkeypatch):
    from app.services import concept_graph
    from app.services.concept_graph import StagedConcept, apply_llm_verified_staged_merges

    class FakeChatProvider:
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            return {
                "should_merge": True,
                "canonical_name": "Posterior Distribution",
                "reason": "same statistical concept",
                "confidence": 0.81,
            }

    class FakeFuzz:
        @staticmethod
        def token_set_ratio(left, right):
            return 30

        @staticmethod
        def token_sort_ratio(left, right):
            return 30

    monkeypatch.setattr(concept_graph, "ChatProvider", FakeChatProvider)
    monkeypatch.setattr(concept_graph, "fuzz", FakeFuzz)
    groups = {
        "posterior distribution": StagedConcept(
            key="posterior distribution",
            name="Posterior Distribution",
            aliases=set(),
            chunk_ids={"c1"},
            chapter_refs={"L1"},
        ),
        "bayesian posterior": StagedConcept(
            key="bayesian posterior",
            name="Bayesian Posterior",
            aliases=set(),
            chunk_ids={"c1", "c2"},
            chapter_refs={"L1"},
        ),
    }

    merged, key_map, verified = await apply_llm_verified_staged_merges(groups)

    assert verified == 1
    assert len(merged) == 1
    assert key_map["bayesian posterior"] == "posterior distribution"
    group = next(iter(merged.values()))
    assert any(item.startswith("llm:") for item in group.merged_from)


def test_get_or_create_concept_merges_alias_variants_within_course(db_session, sample_course):
    from app.services.concept_graph import get_or_create_concept

    concept, created = get_or_create_concept(
        db_session,
        sample_course.id,
        name="Minimum Spanning Tree",
        chapter="Lecture 1",
        summary="A tree that connects all vertices with minimum total edge weight.",
        aliases=["MST"],
        concept_type="concept",
        importance_score=0.8,
    )
    assert created is True

    same_concept, created_again = get_or_create_concept(
        db_session,
        sample_course.id,
        name="minimum-spanning trees",
        chapter="Lecture 2",
        summary="Plural and hyphenated spelling.",
        aliases=[],
        concept_type="concept",
        importance_score=0.7,
    )
    assert created_again is False
    assert same_concept.id == concept.id
    assert sorted(same_concept.chapter_refs) == ["Lecture 1", "Lecture 2"]

    acronym_match, acronym_created = get_or_create_concept(
        db_session,
        sample_course.id,
        name="MST",
        chapter="Lecture 3",
        summary="Common abbreviation.",
        aliases=[],
        concept_type="concept",
        importance_score=0.6,
    )
    assert acronym_created is False
    assert acronym_match.id == concept.id

    separate, separate_created = get_or_create_concept(
        db_session,
        sample_course.id,
        name="Minimum Cut",
        chapter="Lecture 4",
        summary="Different graph optimization concept.",
        aliases=[],
        concept_type="concept",
        importance_score=0.6,
    )
    assert separate_created is True
    assert separate.id != concept.id


def test_validate_graph_payload_synthesizes_relation_endpoint_candidates():
    from app.services.concept_graph import validate_graph_payload

    payload, warnings = validate_graph_payload(
        {
            "concepts": [
                {"name": "Bayesian Inference", "aliases": [], "summary": "", "concept_type": "concept", "importance_score": 0.8}
            ],
            "relations": [
                {
                    "source": "Bayesian Inference",
                    "target": "Missing Concept",
                    "relation_type": "related_to",
                    "confidence": 0.9,
                }
            ],
        }
    )

    assert payload["relations"][0]["target"] == "Missing Concept"
    assert "Missing Concept" in {concept["name"] for concept in payload["concepts"]}
    assert payload["_validation_warnings"] == warnings
    assert "Missing Concept" in warnings[0]


def test_validate_graph_payload_rejects_bad_shape():
    from app.services.concept_graph import validate_graph_payload

    with pytest.raises(ValueError, match="schema validation failed"):
        validate_graph_payload({"concepts": "not-a-list", "relations": []})


@pytest.mark.asyncio
async def test_extract_llm_graph_payloads_isolates_chunk_failures(monkeypatch):
    from app.services import concept_graph

    class FakeChatProvider:
        async def extract_graph_payload(self, text, chapter, source_type):
            if "bad" in text:
                raise RuntimeError("model timeout")
            return {"concepts": [{"name": "Good Concept"}], "relations": []}

    monkeypatch.setattr(concept_graph, "ChatProvider", FakeChatProvider)

    payloads, errors = await concept_graph.extract_llm_graph_payloads(
        [
            make_chunk("ok", "doc-a", "good content"),
            make_chunk("bad", "doc-a", "bad content"),
        ],
        concurrency=2,
    )

    assert set(payloads) == {"ok"}
    assert set(errors) == {"bad"}
    assert "model timeout" in errors["bad"]


@pytest.mark.asyncio
async def test_extract_llm_graph_payloads_records_validation_failures(monkeypatch):
    from app.services import concept_graph

    class FakeChatProvider:
        async def extract_graph_payload(self, text, chapter, source_type):
            return {"concepts": "bad", "relations": []}

    monkeypatch.setattr(concept_graph, "ChatProvider", FakeChatProvider)

    payloads, errors = await concept_graph.extract_llm_graph_payloads([make_chunk("bad-shape", "doc-a", "content")])

    assert payloads == {}
    assert set(errors) == {"bad-shape"}
    assert "schema validation failed" in errors["bad-shape"]


@pytest.mark.asyncio
async def test_extract_llm_graph_payloads_keeps_relation_warning_out_of_failures(monkeypatch):
    from app.services import concept_graph

    class FakeChatProvider:
        async def extract_graph_payload(self, text, chapter, source_type):
            return {
                "concepts": [{"name": "Bayesian Inference"}],
                "relations": [
                    {
                        "source": "Bayesian Inference",
                        "target": "Missing Concept",
                        "relation_type": "related_to",
                        "confidence": 0.9,
                    }
                ],
            }

    monkeypatch.setattr(concept_graph, "ChatProvider", FakeChatProvider)

    payloads, errors = await concept_graph.extract_llm_graph_payloads([make_chunk("warn", "doc-a", "content")])

    assert errors == {}
    assert payloads["warn"]["relations"][0]["target"] == "Missing Concept"
    assert "Missing Concept" in {concept["name"] for concept in payloads["warn"]["concepts"]}
    assert "Missing Concept" in payloads["warn"]["_validation_warnings"][0]


@pytest.mark.asyncio
async def test_recover_existing_graph_algorithm_metrics_commits_success():
    from app.services.ingestion import recover_existing_graph_algorithm_metrics

    class FakeSession:
        def __init__(self):
            self.commits = 0
            self.rollbacks = 0

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    async def fake_enricher(session, course_id):
        assert course_id == "course-a"
        return {"graph_algorithm_nodes": 2}

    session = FakeSession()
    stats = await recover_existing_graph_algorithm_metrics(session, "course-a", enricher=fake_enricher)

    assert stats == {"graph_algorithm_recovered_existing": True, "graph_algorithm_nodes": 2}
    assert session.commits == 1
    assert session.rollbacks == 0


@pytest.mark.asyncio
async def test_recover_existing_graph_algorithm_metrics_reports_failure():
    from app.services.ingestion import recover_existing_graph_algorithm_metrics

    class FakeSession:
        def __init__(self):
            self.commits = 0
            self.rollbacks = 0

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    async def fake_enricher(session, course_id):
        raise RuntimeError("graph unavailable")

    session = FakeSession()
    stats = await recover_existing_graph_algorithm_metrics(session, "course-a", enricher=fake_enricher)

    assert stats["graph_algorithm_recovered_existing"] is False
    assert "graph unavailable" in stats["graph_algorithm_recovery_error"]
    assert session.commits == 0
    assert session.rollbacks == 1


@pytest.mark.asyncio
async def test_chat_provider_repairs_invalid_graph_json(no_fallback_env, monkeypatch):
    from app.services.embeddings import ChatProvider

    calls = []

    async def fake_post_chat_text(self, payload):
        calls.append(payload)
        if len(calls) == 1:
            return "Here is the graph: not-json"
        return '{"concepts":[{"name":"Bayesian Inference","aliases":[],"summary":"","concept_type":"concept","importance_score":0.8}],"relations":[]}'

    monkeypatch.setattr(ChatProvider, "_post_chat_text", fake_post_chat_text)

    payload = await ChatProvider().extract_graph_payload("Bayesian inference updates beliefs.", "Lecture 1", "pdf")

    assert payload["concepts"][0]["name"] == "Bayesian Inference"
    assert len(calls) == 2
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["temperature"] == 0.0


def test_get_or_create_concept_tracks_source_document_ids(db_session, sample_course):
    from app.services.concept_graph import get_or_create_concept

    concept, created = get_or_create_concept(
        db_session,
        sample_course.id,
        "Test Concept",
        "chapter",
        "unit test summary",
        aliases=[],
        concept_type="concept",
        importance_score=0.5,
        document_id="doc-1",
    )
    assert created is True
    assert "doc-1" in concept.source_document_ids

    concept2, created2 = get_or_create_concept(
        db_session,
        sample_course.id,
        "Test Concept",
        "chapter",
        "unit test summary",
        aliases=[],
        concept_type="concept",
        importance_score=0.5,
        document_id="doc-2",
    )
    assert created2 is False
    assert concept2.id == concept.id
    assert sorted(concept2.source_document_ids) == ["doc-1", "doc-2"]


def test_get_or_create_concept_can_allow_valid_entity_matching_chapter_context(db_session, sample_course):
    from app.services.concept_graph import get_or_create_concept

    concept, created = get_or_create_concept(
        db_session,
        sample_course.id,
        "Centrality",
        "Centralities",
        "A graph measure family for node importance.",
        aliases=["Centrality"],
        concept_type="concept",
        importance_score=0.8,
        document_id=None,
        context_terms=set(),
    )

    assert created is True
    assert concept.canonical_name == "Centrality"
    assert concept.normalized_name == "centrality::concept"


def test_sync_graph_chapter_labels_normalizes_existing_refs(db_session, sample_course):
    from app.models import Chunk, Concept, ConceptRelation, Document, DocumentVersion
    from app.services.concept_graph import sync_graph_chapter_labels

    document = Document(
        course_id=sample_course.id,
        title="Labs solutions",
        source_path="data/Unit Test Course/storage/20260425/Labs solutions.pdf",
        source_type="pdf",
        tags=["Labs solutions"],
        checksum="checksum",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="checksum",
        storage_path=document.source_path,
        extracted_path=None,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Lab solution content",
        snippet="Lab solution content",
        chapter="Labs solutions",
        section="Lab",
        source_type="pdf",
        metadata_json={"content_kind": "pdf_page"},
        embedding_status="ready",
    )
    concept = Concept(
        course_id=sample_course.id,
        canonical_name="Minimum Cut",
        normalized_name="minimum cut",
        summary="",
        chapter_refs=["Labs solutions", "20260425"],
        importance_score=0.8,
    )
    empty_ref_concept = Concept(
        course_id=sample_course.id,
        canonical_name="Residual Network",
        normalized_name="residual network",
        summary="",
        chapter_refs=[],
        importance_score=0.8,
    )
    db_session.add_all([chunk, concept, empty_ref_concept])
    db_session.flush()
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=concept.id,
            target_concept_id=empty_ref_concept.id,
            target_name=empty_ref_concept.canonical_name,
            relation_type="related_to",
            evidence_chunk_id=chunk.id,
            confidence=0.9,
            extraction_method="llm",
        )
    )
    db_session.commit()

    stats = sync_graph_chapter_labels(db_session, sample_course.id)

    db_session.refresh(document)
    db_session.refresh(chunk)
    db_session.refresh(concept)
    db_session.refresh(empty_ref_concept)
    assert stats["updated_documents"] == 1
    assert document.tags == ["Lab Solutions"]
    assert chunk.chapter == "Lab Solutions"
    assert concept.chapter_refs == ["Lab Solutions"]
    assert empty_ref_concept.chapter_refs == ["Lab Solutions"]


@pytest.mark.asyncio
async def test_upsert_graph_candidates_accumulates_support_count(db_session, sample_course, monkeypatch):
    """Regression: StagedRelation must merge evidence across chunks so support_count > 1."""
    from app.models import Chunk, ConceptRelation, Document, DocumentVersion
    from app.services.concept_graph import upsert_graph_candidates_from_chunks

    document = Document(
        course_id=sample_course.id,
        title="Test Doc",
        source_path="test.md",
        source_type="markdown",
        checksum="checksum",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="checksum",
        storage_path="test.md",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()

    chunk1 = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Node A is related to Node B.",
        snippet="Node A is related to Node B.",
        chapter="Lecture 1",
        section="Section 1",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=True,
    )
    chunk2 = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Node A and Node B are connected.",
        snippet="Node A and Node B are connected.",
        chapter="Lecture 1",
        section="Section 2",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=True,
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    llm_payloads = {
        str(chunk1.id): {
            "concepts": [
                {"name": "Node A", "aliases": [], "summary": "", "concept_type": "concept", "confidence": 0.8},
                {"name": "Node B", "aliases": [], "summary": "", "concept_type": "concept", "confidence": 0.8},
            ],
            "relations": [
                {"source": "Node A", "target": "Node B", "relation_type": "related_to", "confidence": 0.7},
            ],
        },
        str(chunk2.id): {
            "concepts": [
                {"name": "Node A", "aliases": [], "summary": "", "concept_type": "concept", "confidence": 0.8},
                {"name": "Node B", "aliases": [], "summary": "", "concept_type": "concept", "confidence": 0.8},
            ],
            "relations": [
                {"source": "Node A", "target": "Node B", "relation_type": "related_to", "confidence": 0.75},
            ],
        },
    }

    stats = await upsert_graph_candidates_from_chunks(
        db_session,
        sample_course.id,
        [chunk1, chunk2],
        llm_payloads=llm_payloads,
        run_llm_merge=False,
    )

    relations = db_session.scalars(
        select(ConceptRelation).where(ConceptRelation.course_id == sample_course.id)
    ).all()

    assert len(relations) == 1
    assert relations[0].support_count == 2
    assert set(relations[0].source_document_ids) == {str(document.id)}


@pytest.mark.asyncio
async def test_rebuild_course_graph_aborts_on_backup_failure(db_session, sample_course, monkeypatch):
    """Regression: backup failure must abort rebuild instead of silently deleting the graph."""
    from app.services import concept_graph
    from app.services.concept_graph import rebuild_course_graph

    monkeypatch.setattr(concept_graph, "sync_graph_chapter_labels", lambda db, cid: None)
    monkeypatch.setattr(
        concept_graph,
        "plan_adaptive_graph_extraction_chunks",
        lambda chunks: SimpleNamespace(selected_chunk_ids=[], coverage={}, stop_reason="empty", budget={}),
    )
    import app.services.quality.profiles as quality_profiles
    monkeypatch.setattr(
        quality_profiles,
        "rebuild_domain_quality_profile",
        lambda db, cid: SimpleNamespace(version="v1"),
    )
    monkeypatch.setattr(
        concept_graph,
        "create_graph_extraction_run_from_plan",
        lambda db, **kwargs: SimpleNamespace(id="run-1"),
    )
    async def fake_execute(db, **kwargs):
        return {}, {}, {}

    monkeypatch.setattr(
        concept_graph,
        "execute_graph_extraction_run",
        fake_execute,
    )
    monkeypatch.setattr(
        concept_graph,
        "_backup_course_graph_tables",
        lambda db, cid: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    monkeypatch.setattr(concept_graph, "enrich_course_graph", lambda db, cid: {})
    monkeypatch.setattr(concept_graph, "rebuild_graph_community_summaries", lambda db, cid, **kw: {})

    with pytest.raises(RuntimeError, match="图谱备份失败"):
        await rebuild_course_graph(db_session, sample_course.id, batch_id=None)
