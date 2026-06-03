from __future__ import annotations


def test_graph_payload_marks_inactive_evidence_chunk_stale(db_session, sample_course):
    from app.models import Chunk, Concept, ConceptRelation, Document, DocumentVersion
    from app.services.concept_graph import get_graph_payload
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
    from app.services.concept_graph import record_course_graph_state

    document = Document(
        course_id=sample_course.id,
        title="Freshness Notes",
        source_path="freshness.md",
        source_type="markdown",
        checksum="old",
        tags=[],
    )
    db_session.add(document)
    db_session.flush()
    old_version = DocumentVersion(document_id=document.id, version=1, checksum="old", storage_path="freshness.md", is_active=False)
    new_version = DocumentVersion(document_id=document.id, version=2, checksum="new", storage_path="freshness.md", is_active=True)
    db_session.add_all([old_version, new_version])
    db_session.flush()
    old_chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=old_version.id,
        content="Old graph evidence",
        snippet="Old graph evidence",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=False,
        embedding_text_version="old_text_v1",
    )
    new_chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=new_version.id,
        content="New graph evidence",
        snippet="New graph evidence",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=True,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
    )
    concept = Concept(course_id=sample_course.id, canonical_name="Centrality", normalized_name="centrality", importance_score=0.8)
    db_session.add_all([old_chunk, new_chunk, concept])
    db_session.flush()
    relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=concept.id,
        target_concept_id=concept.id,
        target_name="Centrality",
        relation_type="related_to",
        evidence_chunk_id=old_chunk.id,
        confidence=0.9,
        is_validated=True,
    )
    db_session.add(relation)
    record_course_graph_state(db_session, sample_course.id, build_mode="full")
    db_session.commit()

    payload = get_graph_payload(db_session, sample_course.id, graph_type="semantic")

    assert payload["freshness"]["is_stale"] is True
    assert payload["freshness"]["reason"] == "inactive_evidence_chunks"
    assert payload["freshness"]["stale_evidence_chunks"] == 1

    relation.evidence_chunk_id = new_chunk.id
    db_session.commit()

    refreshed = get_graph_payload(db_session, sample_course.id, graph_type="semantic")
    assert refreshed["freshness"]["is_stale"] is False


def test_graph_payload_marks_new_active_document_without_rebuild_stale(db_session, sample_course):
    from app.models import Chunk, Concept, ConceptRelation, Document, DocumentVersion
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
    from app.services.concept_graph import get_graph_payload, record_course_graph_state

    first_document = Document(
        course_id=sample_course.id,
        title="Built Notes",
        source_path="built.md",
        source_type="markdown",
        checksum="built",
        tags=[],
    )
    db_session.add(first_document)
    db_session.flush()
    first_version = DocumentVersion(document_id=first_document.id, version=1, checksum="built", storage_path="built.md", is_active=True)
    db_session.add(first_version)
    db_session.flush()
    first_chunk = Chunk(
        course_id=sample_course.id,
        document_id=first_document.id,
        document_version_id=first_version.id,
        content="Centrality graph evidence",
        snippet="Centrality graph evidence",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=True,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
    )
    concept = Concept(course_id=sample_course.id, canonical_name="Centrality", normalized_name="centrality", importance_score=0.8)
    db_session.add_all([first_chunk, concept])
    db_session.flush()
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=concept.id,
            target_concept_id=concept.id,
            target_name="Centrality",
            relation_type="related_to",
            evidence_chunk_id=first_chunk.id,
            confidence=0.9,
            is_validated=True,
        )
    )
    record_course_graph_state(db_session, sample_course.id, build_mode="full")
    db_session.commit()

    fresh = get_graph_payload(db_session, sample_course.id, graph_type="semantic")
    assert fresh["freshness"]["is_stale"] is False

    second_document = Document(
        course_id=sample_course.id,
        title="New Notes",
        source_path="new.md",
        source_type="markdown",
        checksum="new",
        tags=[],
    )
    db_session.add(second_document)
    db_session.flush()
    second_version = DocumentVersion(document_id=second_document.id, version=1, checksum="new", storage_path="new.md", is_active=True)
    db_session.add(second_version)
    db_session.flush()
    db_session.add(
        Chunk(
            course_id=sample_course.id,
            document_id=second_document.id,
            document_version_id=second_version.id,
            content="Betweenness centrality was parsed after graph build.",
            snippet="Betweenness centrality",
            source_type="markdown",
            metadata_json={"content_kind": "markdown"},
            embedding_status="ready",
            is_active=True,
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        )
    )
    db_session.commit()

    stale = get_graph_payload(db_session, sample_course.id, graph_type="semantic")

    assert stale["freshness"]["is_stale"] is True
    assert stale["freshness"]["reason"] == "active_documents_changed"
    assert second_version.id in stale["freshness"]["uncovered_document_versions"]


def test_graph_payload_marks_existing_relations_without_watermark_stale(db_session, sample_course):
    from app.models import Chunk, Concept, ConceptRelation, Document, DocumentVersion
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
    from app.services.concept_graph import get_graph_payload

    document = Document(
        course_id=sample_course.id,
        title="Legacy Notes",
        source_path="legacy.md",
        source_type="markdown",
        checksum="legacy",
        tags=[],
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version=1, checksum="legacy", storage_path="legacy.md", is_active=True)
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Legacy evidence",
        snippet="Legacy evidence",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=True,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
    )
    concept = Concept(course_id=sample_course.id, canonical_name="Legacy", normalized_name="legacy", importance_score=0.8)
    db_session.add_all([chunk, concept])
    db_session.flush()
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=concept.id,
            target_concept_id=concept.id,
            target_name="Legacy",
            relation_type="related_to",
            evidence_chunk_id=chunk.id,
            confidence=0.9,
            is_validated=True,
        )
    )
    db_session.commit()

    payload = get_graph_payload(db_session, sample_course.id, graph_type="semantic")

    assert payload["freshness"]["is_stale"] is True
    assert payload["freshness"]["reason"] == "missing_graph_build_watermark"
