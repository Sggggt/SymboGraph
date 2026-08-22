from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import uuid4

import pytest


def _make_snapshot_writable_for_adversarial_test(path: Path) -> None:
    os.chmod(path, stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def _source_chunk(db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash
    from app.services.storage import snapshot_source_file

    source_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    source_root.mkdir(parents=True, exist_ok=True)
    source = source_root / f"version-bound-{uuid4().hex}.md"
    source.write_text("Evidence belongs to the immutable parse attempt.", encoding="utf-8")
    frozen_snapshot = snapshot_source_file(source, sample_knowledge_base.name)
    snapshot_path = frozen_snapshot.canonical_path
    snapshot_checksum = frozen_snapshot.checksum

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Version-bound citation",
        source_path=str(source),
        source_type="markdown",
        checksum=snapshot_checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=snapshot_checksum,
        storage_path=str(snapshot_path),
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    text = "Evidence belongs to the immutable parse attempt."
    chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=7,
        char_start=0,
        char_end=len(text),
        text=text,
        text_hash=text_hash(text),
        state="active",
    )
    db_session.add(chunk)
    db_session.commit()
    return document, version, chunk


def test_search_citation_uses_document_version_snapshot(db_session, sample_knowledge_base):
    from app.services.context_graph import search_payload_for_chunk

    document, version, chunk = _source_chunk(db_session, sample_knowledge_base)

    payload = search_payload_for_chunk(db_session, chunk, 1.0, {"dense": 1.0}, {})

    assert payload["source_path"] == version.storage_path
    assert payload["logical_source_path"] == document.source_path
    assert payload["citations"][0]["source_path"] == version.storage_path
    assert payload["citations"][0]["logical_source_path"] == document.source_path
    assert payload["citations"][0]["source_span"]["source_checksum"] == version.checksum


def test_chunk_source_span_fails_closed_for_inactive_attempt(db_session, sample_knowledge_base):
    from app.services.context_graph import ChunkSourceProvenanceError, chunk_source_span

    _document, version, chunk = _source_chunk(db_session, sample_knowledge_base)
    version.is_active = False
    db_session.commit()

    with pytest.raises(ChunkSourceProvenanceError, match="document_version_inactive"):
        chunk_source_span(db_session, chunk)


def test_snapshot_integrity_is_verified_once_per_request_scope(db_session, sample_knowledge_base):
    from app.services.context_graph import SnapshotIntegrityVerifier, chunk_source_span

    _document, _version, chunk = _source_chunk(db_session, sample_knowledge_base)
    verifier = SnapshotIntegrityVerifier()

    first = chunk_source_span(db_session, chunk, snapshot_verifier=verifier)
    second = chunk_source_span(db_session, chunk, snapshot_verifier=verifier)

    assert verifier.verification_count == 1
    assert first["source_snapshot_verification"]["verified"] is True
    assert second["source_snapshot_verification"] == first["source_snapshot_verification"]


def test_request_scope_snapshot_cache_rechecks_file_identity(
    db_session,
    sample_knowledge_base,
):
    from app.services.context_graph import (
        ChunkSourceProvenanceError,
        SnapshotIntegrityVerifier,
        chunk_source_span,
    )

    _document, version, chunk = _source_chunk(db_session, sample_knowledge_base)
    verifier = SnapshotIntegrityVerifier()
    chunk_source_span(db_session, chunk, snapshot_verifier=verifier)
    snapshot = Path(version.storage_path)
    _make_snapshot_writable_for_adversarial_test(snapshot)
    snapshot.write_bytes(b"changed after first verification")
    from app.services.storage import protect_immutable_file

    protect_immutable_file(snapshot)

    with pytest.raises(ChunkSourceProvenanceError, match="changed after request-scope"):
        chunk_source_span(db_session, chunk, snapshot_verifier=verifier)


@pytest.mark.parametrize("damage", ["missing", "tampered", "outside"])
def test_chunk_source_span_fails_closed_for_invalid_snapshot(
    db_session,
    sample_knowledge_base,
    damage: str,
):
    from app.services.context_graph import ChunkSourceProvenanceError, chunk_source_span

    document, version, chunk = _source_chunk(db_session, sample_knowledge_base)
    snapshot = Path(version.storage_path)
    if damage == "missing":
        _make_snapshot_writable_for_adversarial_test(snapshot)
        snapshot.unlink()
        expected = "missing or inaccessible"
    elif damage == "tampered":
        _make_snapshot_writable_for_adversarial_test(snapshot)
        snapshot.write_bytes(b"tampered immutable source")
        from app.services.storage import protect_immutable_file

        protect_immutable_file(snapshot)
        expected = "checksum mismatch"
    else:
        version.storage_path = document.source_path
        db_session.commit()
        expected = "outside its knowledge-base snapshot root"

    with pytest.raises(ChunkSourceProvenanceError, match=expected):
        chunk_source_span(db_session, chunk)
