from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select


def _knowledge_base_source(knowledge_base, filename: str) -> Path:
    from app.core.config import get_settings

    root = get_settings().knowledge_base_paths_for_name(knowledge_base.name)[
        "storage_root"
    ]
    root.mkdir(parents=True, exist_ok=True)
    return (root / filename).resolve()


def test_language_detection_is_deterministic_and_keeps_unsupported_or_short_text_unknown():
    from app.services.language_metadata import (
        LANGUAGE_DETECTION_PROTOCOL_VERSION,
        detect_document_language,
    )

    english_sections = [
        SimpleNamespace(
            title="Bayesian inference",
            text=(
                "This document explains the posterior distribution and the evidence "
                "that is used for inference in a probabilistic model."
            ),
        )
    ]
    chinese_sections = [
        SimpleNamespace(
            title="贝叶斯推断",
            text="本文说明如何根据观测证据更新后验概率，并比较不同模型的推断结果。",
        )
    ]
    first = detect_document_language(english_sections)
    second = detect_document_language(english_sections)
    chinese = detect_document_language(chinese_sections)
    unknown = detect_document_language(
        [SimpleNamespace(title="x", text="x = 1")]
    )
    unsupported = detect_document_language(
        [
            SimpleNamespace(
                title="Русский текст",
                text=(
                    "Этот документ содержит достаточно букв для наблюдения, "
                    "но язык не входит в закрытый список детектора."
                ),
            )
        ]
    )
    mixed = detect_document_language(
        [
            SimpleNamespace(
                title="Mixed evidence",
                text=(
                    "中文证据用于解释概率模型和推理结果以及不同观察之间的关系，"
                    "while the English evidence is deliberately mixed in the record."
                ),
            )
        ]
    )

    assert first == second
    assert first["language"] == "en"
    assert first["language_source"] == "deterministic_detection"
    assert first["language_detection_protocol_version"] == LANGUAGE_DETECTION_PROTOCOL_VERSION
    assert len(first["language_detection_hash"]) == 64
    assert chinese["language"] == "zh"
    assert unknown["language"] is None
    assert unknown["language_source"] == "unknown"
    assert unknown["language_detection_hash"]
    assert unsupported["language"] is None
    assert unsupported["language_source"] == "unknown"
    assert mixed["language"] is None
    assert mixed["language_source"] == "unknown"


def test_language_detection_head_tail_sample_obeys_the_total_character_budget():
    from app.services.language_metadata import (
        LANGUAGE_DETECTION_SAMPLE_MAX_CHARS,
        detect_document_language,
    )

    head = "a" * LANGUAGE_DETECTION_SAMPLE_MAX_CHARS
    tail = "z" * LANGUAGE_DETECTION_SAMPLE_MAX_CHARS
    identity = detect_document_language(
        [SimpleNamespace(title="", text=f"{head} {tail}")]
    )
    card = identity["language_metadata_json"]

    assert card["sample_char_count"] == LANGUAGE_DETECTION_SAMPLE_MAX_CHARS
    assert card["sample_char_count"] <= LANGUAGE_DETECTION_SAMPLE_MAX_CHARS


def test_explicit_bcp47_metadata_has_precedence_but_und_remains_unknown():
    from app.services.language_metadata import (
        InvalidExplicitLanguageMetadata,
        detect_document_language,
        normalize_explicit_language_tag,
    )

    english_sections = [
        SimpleNamespace(
            title="English body",
            text="This is an English document with the evidence in the body.",
        )
    ]
    explicit = detect_document_language(
        english_sections,
        explicit_language="ZH_cn",
    )
    undetermined = detect_document_language(
        english_sections,
        explicit_language="und",
    )

    assert explicit["language"] == "zh"
    assert explicit["language_source"] == "explicit_metadata"
    assert explicit["language_metadata_json"]["normalized_language_tag"] == "zh-cn"
    assert undetermined["language"] is None
    assert undetermined["language_source"] == "unknown"
    with pytest.raises(InvalidExplicitLanguageMetadata):
        normalize_explicit_language_tag("not a language tag")


def test_pre_language_identity_pending_intent_fails_closed_on_reregistration(
    db_session,
    sample_knowledge_base,
):
    from app.services import ingestion

    source = _knowledge_base_source(sample_knowledge_base, "stale-language-intent.md")
    source.write_text("# Pending source\n\nEnough text is staged for later parsing.\n", encoding="utf-8")
    _document, queued_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
    )
    stale_intent = dict(queued_job.stats[ingestion.DOCUMENT_METADATA_INTENT_KEY])
    stale_intent["protocol_version"] = "document_metadata_intent_v1"
    queued_job.stats = {
        **queued_job.stats,
        ingestion.DOCUMENT_METADATA_INTENT_KEY: stale_intent,
    }
    db_session.commit()

    with pytest.raises(
        ingestion.DocumentMetadataRestoreError,
        match="unsupported protocol",
    ):
        ingestion.register_uploaded_file(
            db_session,
            sample_knowledge_base,
            source,
        )


@pytest.mark.asyncio
async def test_ingestion_persists_matching_document_version_and_audit_cards(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Document, DocumentVersion, IngestionJob, ParseJob, SourceFile
    from app.services import ingestion
    from app.services.language_metadata import (
        LANGUAGE_DETECTION_PROTOCOL_VERSION,
        language_identity_from_record,
    )

    source = _knowledge_base_source(sample_knowledge_base, "language-english.md")
    source.write_text(
        "# Bayesian inference\n\n"
        "This document explains the posterior distribution and the evidence that is "
        "used for inference in a probabilistic model. The examples are grounded in data.\n",
        encoding="utf-8",
    )
    result = await ingestion.ingest_file(
        db_session,
        source,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )

    document = db_session.get(Document, result["document_id"])
    version = db_session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.is_active.is_(True),
        )
    )
    source_file = db_session.scalar(
        select(SourceFile).where(SourceFile.document_id == document.id)
    )
    parse_job = db_session.scalar(
        select(ParseJob).where(ParseJob.document_version_id == version.id)
    )
    job = db_session.scalar(
        select(IngestionJob)
        .where(IngestionJob.document_id == document.id)
        .order_by(IngestionJob.created_at.desc())
    )
    document_identity = language_identity_from_record(document)
    version_identity = language_identity_from_record(version)

    assert result["language"] == "en"
    assert document_identity["valid"] is True
    assert version_identity["valid"] is True
    assert document_identity["detection_hash"] == version_identity["detection_hash"]
    assert result["language_detection_hash"] == version.language_detection_hash
    assert version.language_detection_protocol_version == LANGUAGE_DETECTION_PROTOCOL_VERSION
    assert source_file.metadata_json["language_detection_hash"] == version.language_detection_hash
    assert parse_job.stats_json["language_identity"]["language"] == "en"
    assert parse_job.diagnostics_json["language_identity"]["status"] == "resolved"
    assert job.stats["language_identity"]["detection_hash"] == version.language_detection_hash
    metadata_intent = job.stats[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert metadata_intent["status"] == "applied"
    assert metadata_intent["protocol_version"] == "document_metadata_intent_v2"


@pytest.mark.asyncio
async def test_uploaded_explicit_language_survives_worker_job_boundary_and_overrides_detection(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Document, DocumentVersion
    from app.services import ingestion
    from app.services.language_metadata import language_identity_from_record

    source = _knowledge_base_source(sample_knowledge_base, "explicit-language.md")
    source.write_text(
        "# English content\n\n"
        "This document is written in English and it contains enough words for the "
        "deterministic detector to identify the language from the content.\n",
        encoding="utf-8",
    )
    document, queued_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
        explicit_language="zh-CN",
    )
    pending = ingestion.ingestion_job_language_identity_summary(queued_job)
    assert pending["status"] == "pending"
    assert pending["explicit_language_tag"] == "zh-cn"

    # This is the same service entry used by the Celery worker: only the durable
    # job id and source path cross the worker boundary.
    result = await ingestion.ingest_file(
        db_session,
        source,
        existing_job_id=queued_job.id,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )
    db_session.refresh(document)
    version = db_session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.is_active.is_(True),
        )
    )

    assert result["language"] == "zh"
    assert document.language_source == "explicit_metadata"
    assert version.language_source == "explicit_metadata"
    assert language_identity_from_record(document)["valid"] is True
    assert document.language_detection_hash == version.language_detection_hash
    assert document.language_metadata_json["normalized_language_tag"] == "zh-cn"


@pytest.mark.asyncio
async def test_normal_bilingual_ingestion_reaches_cross_language_candidate_channel(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Chunk
    from app.services import ingestion
    from app.services.context_graph import (
        dense_graph_operating_point,
        relation_edge_candidates,
    )

    english_source = _knowledge_base_source(
        sample_knowledge_base,
        "normal-ingestion-english.md",
    )
    chinese_source = _knowledge_base_source(
        sample_knowledge_base,
        "normal-ingestion-chinese.md",
    )
    english_source.write_text(
        "# Bayesian inference\n\n"
        "This document explains how the posterior distribution is updated from "
        "observed evidence in a probabilistic model and why the result is useful.\n",
        encoding="utf-8",
    )
    chinese_source.write_text(
        "# 贝叶斯推断\n\n"
        "本文说明如何根据观测证据更新概率模型中的后验分布，并解释这一推断结果为何有用。\n",
        encoding="utf-8",
    )

    english_result = await ingestion.ingest_file(
        db_session,
        english_source,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )
    chinese_result = await ingestion.ingest_file(
        db_session,
        chinese_source,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )
    assert english_result["language"] == "en"
    assert chinese_result["language"] == "zh"

    chunks = list(
        db_session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id.in_(
                    [english_result["document_id"], chinese_result["document_id"]]
                ),
                Chunk.state == "active",
            )
            .order_by(Chunk.document_id, Chunk.chunk_index)
        ).all()
    )
    assert len(chunks) == 2
    vectors = {
        chunks[0].id: [1.0, 0.0],
        chunks[1].id: [0.999, 0.001],
    }
    candidates, diagnostics = relation_edge_candidates(
        db_session,
        chunks,
        vectors,
        dense_graph_operating_point(),
    )

    cross_language_edges = [
        candidate
        for candidate in candidates.values()
        if candidate.edge_type == "dense_cross_language_bridge"
    ]
    assert cross_language_edges
    assert diagnostics["channel_candidate_stats"]["cross_language_candidates"][
        "eligible_candidate_count"
    ] >= 2
    assert diagnostics["relation_quota_signals"]["language_identity"][
        "language_counts"
    ] == {"en": 1, "zh": 1}
    language_scope_hash = diagnostics["relation_quota_signals"][
        "language_identity"
    ]["scope_hash"]
    for edge in cross_language_edges:
        assert edge.features_json["source_language_detection_hash"]
        assert edge.features_json["target_language_detection_hash"]
        assert edge.features_json["source_language_identity_valid"] is True
        assert edge.features_json["target_language_identity_valid"] is True
        assert edge.features_json["language_identity_scope_hash"] == language_scope_hash
