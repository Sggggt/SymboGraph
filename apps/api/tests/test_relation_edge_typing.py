from __future__ import annotations

from types import SimpleNamespace


def _add_document_chunks(
    db_session,
    knowledge_base_id: str,
    *,
    suffix: str,
    language: str,
    chunk_count: int,
):
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash
    from app.services.language_metadata import apply_language_identity, detect_document_language

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title=f"Document {suffix}",
        source_path=f"{suffix}.md",
        source_type="markdown",
        language=language,
        checksum=f"checksum-{suffix}",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=f"checksum-{suffix}",
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    language_identity = detect_document_language(
        [SimpleNamespace(title=f"Document {suffix}", text=f"Language fixture {suffix}")],
        explicit_language=language,
    )
    apply_language_identity(document, language_identity)
    apply_language_identity(version, language_identity)
    db_session.flush()
    chunks = []
    for index in range(chunk_count):
        text = f"Semantic content {suffix} {index}"
        chunk = Chunk(
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=index,
            token_start=index * 4,
            token_end=(index + 1) * 4,
            char_start=index * 100,
            char_end=index * 100 + len(text),
            text=text,
            text_hash=text_hash(text),
            section_path="Typing fixture",
            state="active",
        )
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.flush()
    return chunks


def test_dense_candidate_channels_classify_each_pair_to_one_semantic_edge_type(db_session, sample_knowledge_base):
    from app.services.context_graph import dense_graph_operating_point, relation_edge_candidates

    same_document = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="same-document-en",
        language="en",
        chunk_count=2,
    )
    cross_document = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="cross-document-en",
        language="en",
        chunk_count=1,
    )[0]
    cross_language = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="cross-language-zh",
        language="zh",
        chunk_count=1,
    )[0]
    chunks = [*same_document, cross_document, cross_language]
    vectors = {chunk.id: [1.0, 0.0, 0.0] for chunk in chunks}
    operating_point = {
        **dense_graph_operating_point(),
        "dense_knn_k_min": 10,
        "dense_knn_k_max": 10,
        "cross_doc_out_quota_min": 10,
        "cross_doc_out_quota_max": 10,
        "cross_language_out_quota_min": 10,
        "cross_language_out_quota_max": 10,
        "dense_reverse_b_min_base": 10,
        "dense_reverse_b_max_base": 10,
        "dense_reverse_b_min_doc": 10,
        "dense_reverse_b_max_doc": 10,
        "dense_reverse_b_min_lang": 10,
        "dense_reverse_b_max_lang": 10,
        "dense_min_cosine": 0.1,
        "cross_doc_min_cosine": 0.1,
        "cross_language_min_cosine": 0.1,
        "dense_strong_cosine": 0.9,
    }

    candidates, diagnostics = relation_edge_candidates(db_session, chunks, vectors, operating_point)
    edge_types_by_pair: dict[tuple[str, str], set[str]] = {}
    candidates_by_pair = {}
    for candidate in candidates.values():
        pair = tuple(sorted((candidate.source_chunk_id, candidate.target_chunk_id)))
        edge_types_by_pair.setdefault(pair, set()).add(candidate.edge_type)
        candidates_by_pair[pair] = candidate

    same_pair = tuple(sorted((same_document[0].id, same_document[1].id)))
    cross_doc_pairs = {
        tuple(sorted((chunk.id, cross_document.id)))
        for chunk in same_document
    }
    cross_language_pairs = {
        tuple(sorted((chunk.id, cross_language.id)))
        for chunk in [*same_document, cross_document]
    }
    assert edge_types_by_pair[same_pair] == {"dense_semantic"}
    assert all(edge_types_by_pair[pair] == {"dense_cross_document_bridge"} for pair in cross_doc_pairs)
    assert all(edge_types_by_pair[pair] == {"dense_cross_language_bridge"} for pair in cross_language_pairs)
    assert all(len(types) == 1 for types in edge_types_by_pair.values())
    for pair in cross_language_pairs:
        assert candidates_by_pair[pair].features_json["is_cross_document"] is True
        assert "cross_language_candidates" in candidates_by_pair[pair].features_json["candidate_channels"]
        assert candidates_by_pair[pair].features_json["source_language_detection_hash"]
        assert candidates_by_pair[pair].features_json["target_language_detection_hash"]
        assert candidates_by_pair[pair].features_json["source_language_identity_valid"] is True
        assert candidates_by_pair[pair].features_json["target_language_identity_valid"] is True
    assert diagnostics["accepted_edge_types"] == {
        "dense_semantic": 1,
        "dense_cross_document_bridge": 2,
        "dense_cross_language_bridge": 3,
    }
    identity_diagnostics = diagnostics["relation_quota_signals"]["language_identity"]
    assert identity_diagnostics["invalid_identity_count"] == 0
    assert identity_diagnostics["language_counts"] == {"en": 3, "zh": 1}
    assert len(identity_diagnostics["scope_hash"]) == 64
    assert all(
        candidate.features_json["language_identity_scope_hash"]
        == identity_diagnostics["scope_hash"]
        for candidate in candidates.values()
    )


def test_missing_or_inconsistent_version_language_hash_fails_closed_to_cross_document(
    db_session,
    sample_knowledge_base,
):
    from app.models import Document, DocumentVersion
    from app.services.context_graph import dense_graph_operating_point, relation_edge_candidates

    english = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="trusted-en",
        language="en",
        chunk_count=1,
    )[0]
    tampered = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="tampered-zh",
        language="zh",
        chunk_count=1,
    )[0]
    tampered_version = db_session.get(DocumentVersion, tampered.document_version_id)
    tampered_document = db_session.get(Document, tampered.document_id)
    tampered_version.language_detection_hash = None
    tampered_document.language = "zh"
    db_session.flush()

    operating_point = {
        **dense_graph_operating_point(),
        "dense_knn_k_min": 4,
        "dense_knn_k_max": 4,
        "cross_doc_out_quota_min": 4,
        "cross_doc_out_quota_max": 4,
        "cross_language_out_quota_min": 4,
        "cross_language_out_quota_max": 4,
        "dense_reverse_b_min_base": 4,
        "dense_reverse_b_max_base": 4,
        "dense_reverse_b_min_doc": 4,
        "dense_reverse_b_max_doc": 4,
        "dense_reverse_b_min_lang": 4,
        "dense_reverse_b_max_lang": 4,
        "dense_min_cosine": 0.1,
        "cross_doc_min_cosine": 0.1,
        "cross_language_min_cosine": 0.1,
        "dense_strong_cosine": 0.9,
    }
    chunks = [english, tampered]
    candidates, diagnostics = relation_edge_candidates(
        db_session,
        chunks,
        {chunk.id: [1.0, 0.0, 0.0] for chunk in chunks},
        operating_point,
    )

    assert {candidate.edge_type for candidate in candidates.values()} == {
        "dense_cross_document_bridge"
    }
    assert diagnostics["channel_candidate_stats"]["cross_language_candidates"][
        "eligible_candidate_count"
    ] == 0
    identity_diagnostics = diagnostics["relation_quota_signals"]["language_identity"]
    assert identity_diagnostics["known_language_count"] == 1
    assert identity_diagnostics["invalid_identity_count"] == 1
    candidate = next(iter(candidates.values()))
    assert not (
        candidate.features_json["source_language_identity_valid"]
        and candidate.features_json["target_language_identity_valid"]
    )
    assert candidate.features_json["is_cross_language"] is False
