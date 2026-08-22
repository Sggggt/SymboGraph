from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from sqlalchemy import func, select


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


def test_upload_slot_key_is_nfkc_casefold_stable() -> None:
    from app.services.storage import normalize_upload_source_slot_key

    assert normalize_upload_source_slot_key("Ｒｅｐｏｒｔ.MD") == (
        normalize_upload_source_slot_key("report.md")
    )


def test_upload_reparse_recovers_original_title_from_logical_slot(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import Document
    from app.services import ingestion
    from app.services.storage import (
        UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        normalize_upload_source_slot_key,
    )

    digest = "a" * 64
    stored_path = tmp_path / f"{digest}.md"
    stored_path.write_text("# 数据中心整体方案介绍 V6.0\n", encoding="utf-8")
    logical_slot = normalize_upload_source_slot_key("数据中心整体方案介绍V6.0.md")
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title=digest,
        source_path=str(stored_path),
        logical_source_slot_key=logical_slot,
        source_slot_protocol_version=UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        source_type="markdown",
        checksum=digest,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()

    title = ingestion._document_display_title(stored_path, document=document)

    assert title == "数据中心整体方案介绍v6.0"


def test_upload_reparse_preserves_exact_existing_display_title(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import Document
    from app.services import ingestion
    from app.services.storage import (
        UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        normalize_upload_source_slot_key,
    )

    digest = "b" * 64
    stored_path = tmp_path / f"{digest}.md"
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="DataCenter V6.0",
        source_path=str(stored_path),
        logical_source_slot_key=normalize_upload_source_slot_key("DATACENTER V6.0.md"),
        source_slot_protocol_version=UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        source_type="markdown",
        checksum=digest,
        is_active=True,
    )

    assert ingestion._document_display_title(stored_path, document=document) == "DataCenter V6.0"


def test_upload_reparse_recovers_exact_historical_display_title(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import Document, IngestionJob
    from app.services import ingestion
    from app.services.storage import (
        UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        normalize_upload_source_slot_key,
    )

    digest = "c" * 64
    stored_path = tmp_path / f"{digest}.md"
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title=digest,
        source_path=str(stored_path),
        logical_source_slot_key=normalize_upload_source_slot_key("DATACENTER V6.0.md"),
        source_slot_protocol_version=UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        source_type="markdown",
        checksum=digest,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    db_session.add(
        IngestionJob(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            trigger_source="upload",
            source_path=str(stored_path),
            status="completed",
            stats={
                ingestion.DOCUMENT_METADATA_INTENT_KEY: {
                    "candidate_state": {
                        "metadata": {"title": "DataCenter V6.0"}
                    }
                }
            },
        )
    )
    db_session.flush()

    recovered = ingestion._historical_document_display_title(
        db_session,
        document,
    )

    assert recovered == "DataCenter V6.0"
    assert (
        ingestion._document_display_title(
            stored_path,
            document=document,
            display_title=recovered,
        )
        == "DataCenter V6.0"
    )


def test_kb_uuid_roots_do_not_collapse_sanitized_names(no_fallback_env: Path) -> None:
    from app.core.config import get_settings

    settings = get_settings()
    first = settings.knowledge_base_paths_for_id(
        "00000000-0000-0000-0000-000000000001"
    )["knowledge_base_root"]
    second = settings.knowledge_base_paths_for_id(
        "00000000-0000-0000-0000-000000000002"
    )["knowledge_base_root"]

    assert first != second
    assert first.name == "kb_00000000000000000000000000000001"
    assert second.name == "kb_00000000000000000000000000000002"


@pytest.mark.asyncio
async def test_case_and_unicode_equivalent_uploads_reuse_one_document_slot(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import Document
    from app.routers.ingestion import upload_file
    from app.services.storage import (
        UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
        normalize_upload_source_slot_key,
    )

    first = await upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("Ｒｅｐｏｒｔ.MD", b"first"),
        db=db_session,
    )
    second = await upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("report.md", b"second"),
        db=db_session,
    )

    assert first["source_path"] == second["source_path"]
    assert first["document_id"] == second["document_id"]
    assert (
        db_session.scalar(
            select(func.count(Document.id)).where(
                Document.knowledge_base_id == sample_knowledge_base.id
            )
        )
        == 1
    )
    document = db_session.get(Document, first["document_id"])
    assert document is not None
    assert document.logical_source_slot_key == normalize_upload_source_slot_key(
        "report.md"
    )
    assert (
        document.source_slot_protocol_version
        == UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION
    )
    assert document.title == "Report"
    assert document.tags == [document.title]
