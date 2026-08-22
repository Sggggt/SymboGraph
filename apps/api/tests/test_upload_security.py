from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import stat

import pytest
from fastapi import HTTPException, UploadFile


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


def _stored_files(storage_root: Path) -> list[Path]:
    return [path for path in storage_root.rglob("*") if path.is_file()]


def _storage_tree(storage_root: Path) -> list[str]:
    if not storage_root.exists():
        return []
    return sorted(
        str(path.relative_to(storage_root))
        for path in storage_root.rglob("*")
    )


def _durable_side_effect_counts(db_session, knowledge_base_id: str) -> dict[str, int]:
    from app.models import (
        Document,
        IngestionBatch,
        IngestionBatchRecovery,
        IngestionCompensationLog,
        IngestionFileStage,
        IngestionJob,
        SourceFile,
        StorageMaintenanceIntent,
    )

    db_session.expire_all()
    return {
        "documents": db_session.query(Document).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
        "source_files": db_session.query(SourceFile).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
        "jobs": db_session.query(IngestionJob).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
        "batches": db_session.query(IngestionBatch).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
        "batch_recoveries": db_session.query(IngestionBatchRecovery).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
        "file_stages": db_session.query(IngestionFileStage).count(),
        "intents": db_session.query(IngestionCompensationLog).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
        "storage_intents": db_session.query(StorageMaintenanceIntent).filter_by(
            knowledge_base_id=knowledge_base_id
        ).count(),
    }


def _ooxml_bytes(*, presentation: bool) -> bytes:
    from zipfile import ZIP_DEFLATED, ZipFile

    buffer = BytesIO()
    required_part = (
        "ppt/presentation.xml" if presentation else "word/document.xml"
    )
    required_mime = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
        if presentation
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    )
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Types>"
                f'<Override PartName="/{required_part}" '
                f'ContentType="{required_mime}"/>'
                "</Types>"
            ),
        )
        archive.writestr("_rels/.rels", "<Relationships/>")
        archive.writestr(required_part, "<document/>")
    return buffer.getvalue()


def _legacy_ppt_header() -> bytes:
    header = bytearray(512)
    header[:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    header[26:28] = (3).to_bytes(2, "little")
    header[28:30] = b"\xfe\xff"
    header[30:32] = (9).to_bytes(2, "little")
    header[32:34] = (6).to_bytes(2, "little")
    return bytes(header)


def _minimal_bmp() -> bytes:
    content = bytearray(30)
    content[:2] = b"BM"
    content[2:6] = len(content).to_bytes(4, "little")
    content[10:14] = (26).to_bytes(4, "little")
    content[14:18] = (12).to_bytes(4, "little")
    return bytes(content)


def _raw_multipart_body(filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "upload-security-formal-raw-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="upload"; filename="{filename}"\r\n'
        "Content-Type: text/markdown\r\n\r\n"
    ).encode("utf-8")
    body += content
    body += f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, boundary


async def _asgi_client(db_session):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from app.db import get_db
    from app.routers.ingestion import router

    app = FastAPI()
    app.include_router(router, prefix="/api")

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://upload-security.example.test/api/",
    )


def test_upload_replacement_v2_schema_hash_is_frozen_when_global_protocols_expand(monkeypatch):
    from app.services import storage, upload_replacement

    baseline = upload_replacement.upload_source_replacement_schema_hash()
    monkeypatch.setattr(
        storage,
        "NAMESPACE_DURABILITY_PROTOCOLS",
        frozenset({*storage.NAMESPACE_DURABILITY_PROTOCOLS, "future_platform_barrier_v1"}),
    )

    assert upload_replacement.upload_source_replacement_schema_hash() == baseline
    assert (
        "future_platform_barrier_v1"
        not in upload_replacement.UPLOAD_SOURCE_REPLACEMENT_V2_NAMESPACE_DURABILITY_PROTOCOLS
    )


@pytest.mark.parametrize(
    ("status", "phase", "expected"),
    [
        ("pending", "candidate_installed", True),
        ("cleanup_pending", "database_committed", True),
        ("manual_review", "candidate_installed", True),
        ("completed", "completed", True),
        ("rolled_back", "rolled_back", True),
        ("completed", "candidate_installed", False),
        ("rolled_back", "database_committed", False),
        ("manual_review", "completed", False),
    ],
)
def test_upload_replacement_status_phase_contract(status, phase, expected):
    from app.services.upload_replacement import UPLOAD_SOURCE_REPLACEMENT_STATUS_PHASES

    assert (phase in UPLOAD_SOURCE_REPLACEMENT_STATUS_PHASES[status]) is expected


def test_upload_file_response_requires_typed_lock_diagnostics_and_job_id():
    from pydantic import ValidationError

    from app.schemas import UploadFileResponse

    response = {
        "document_id": "document-id",
        "job_id": "job-id",
        "status": "queued",
        "source_path": "/storage/notes.md",
        "language_identity": {
            "status": "pending",
            "language": None,
            "source": None,
            "confidence": None,
            "protocol_version": "document_language_unicode_script_v1",
            "detection_hash": None,
            "explicit_language_tag": None,
            "decision_reason": None,
        },
        "upload_replacement": {
            "protocol_version": "upload_source_replacement_v2",
            "intent_id": "intent-id",
            "status": "completed",
            "phase": "completed",
            "database_committed": True,
            "cleanup_pending": False,
            "postcommit_lock_release_failure": {
                "resource_key": "knowledge_base:kb-id",
                "knowledge_base_id": "kb-id",
                "advisory_key": 123,
                "backend": "postgresql",
                "operation": "upload_registration",
                "batch_id": None,
                "protocol_version": "postgres_advisory_kb_v1",
                "release_error": "OperationalError",
            },
            "lock_release_audit": None,
        },
    }

    validated = UploadFileResponse.model_validate(response)
    assert validated.job_id == "job-id"
    assert validated.upload_replacement.postcommit_lock_release_failure.advisory_key == 123

    without_job = {key: value for key, value in response.items() if key != "job_id"}
    with pytest.raises(ValidationError):
        UploadFileResponse.model_validate(without_job)

    malformed_diagnostics = {
        **response,
        "upload_replacement": {
            **response["upload_replacement"],
            "postcommit_lock_release_failure": {
                "resource_key": "knowledge_base:kb-id",
                "knowledge_base_id": "kb-id",
                "backend": "postgresql",
                "operation": "upload_registration",
                "protocol_version": "postgres_advisory_kb_v1",
                "release_error": "OperationalError",
            },
        },
    }
    with pytest.raises(ValidationError):
        UploadFileResponse.model_validate(malformed_diagnostics)

    diagnostics_with_unknown_field = {
        **response,
        "upload_replacement": {
            **response["upload_replacement"],
            "postcommit_lock_release_failure": {
                **response["upload_replacement"]["postcommit_lock_release_failure"],
                "provider_response": "must not enter the response contract",
            },
        },
    }
    with pytest.raises(ValidationError):
        UploadFileResponse.model_validate(diagnostics_with_unknown_field)


@pytest.mark.parametrize(
    ("filename", "content", "expected_kind"),
    [
        ("notes.md", b"# valid markdown\n", "utf8_text"),
        ("notes.markdown", b"valid markdown\n", "utf8_text"),
        ("notes.txt", b"valid text\n", "utf8_text"),
        (
            "page.html",
            b"<!doctype html><html><body>valid</body></html>",
            "html",
        ),
        (
            "page.htm",
            b"<html><body>valid</body></html>",
            "html",
        ),
        (
            "notebook.ipynb",
            b'{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}',
            "jupyter_notebook",
        ),
        (
            "paper.pdf",
            b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n",
            "pdf",
        ),
        (
            "pixel.png",
            bytes.fromhex(
                "89504e470d0a1a0a0000000d494844520000000100000001"
                "08060000001f15c4890000000d4944415408d763f8cfc0f0"
                "1f00050001ff89993d1d0000000049454e44ae426082"
            ),
            "png",
        ),
        ("photo.jpg", b"\xff\xd8\xff\xe0JFIF\x00\xff\xd9", "jpeg"),
        ("photo.jpeg", b"\xff\xd8\xff\xe0JFIF\x00\xff\xd9", "jpeg"),
        ("bitmap.bmp", _minimal_bmp(), "bmp"),
        ("legacy.ppt", _legacy_ppt_header(), "ole_presentation"),
        ("word.docx", _ooxml_bytes(presentation=False), "ooxml_word"),
        (
            "slides.pptx",
            _ooxml_bytes(presentation=True),
            "ooxml_presentation",
        ),
    ],
)
def test_versioned_content_signature_accepts_each_allowlisted_contract(
    filename,
    content,
    expected_kind,
):
    from app.services.storage import (
        UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION,
        upload_content_signature_protocol_hash,
        upload_filename_validation_protocol_hash,
        validate_upload_admission,
    )

    validated = validate_upload_admission(
        _upload(filename, content),
        max_bytes=len(content),
    )

    assert validated.content_kind == expected_kind
    assert (
        validated.content_signature_protocol_version
        == UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION
    )
    assert len(upload_content_signature_protocol_hash()) == 64
    assert len(upload_filename_validation_protocol_hash()) == 64


def test_all_allowlisted_suffixes_reject_content_outside_their_contract():
    from app.services.storage import (
        ALLOWED_UPLOAD_SUFFIXES,
        UploadValidationError,
        validate_upload_admission,
    )

    invalid_by_suffix = {
        suffix: (
            b"\x00binary"
            if suffix in {".md", ".markdown", ".txt"}
            else b"not the declared content"
        )
        for suffix in ALLOWED_UPLOAD_SUFFIXES
    }
    for suffix, content in sorted(invalid_by_suffix.items()):
        with pytest.raises(UploadValidationError):
            validate_upload_admission(
                _upload(f"invalid{suffix}", content),
                max_bytes=len(content),
            )

    with pytest.raises(UploadValidationError, match="OOXML"):
        validate_upload_admission(
            _upload("word.docx", _ooxml_bytes(presentation=True)),
            max_bytes=len(_ooxml_bytes(presentation=True)),
        )
    with pytest.raises(UploadValidationError, match="OOXML"):
        validate_upload_admission(
            _upload("slides.pptx", _ooxml_bytes(presentation=False)),
            max_bytes=len(_ooxml_bytes(presentation=False)),
        )


@pytest.mark.asyncio
async def test_real_asgi_rejects_raw_path_and_unicode_confusable_filenames_with_zero_side_effects(
    db_session,
    sample_knowledge_base,
):
    from app.services.storage import PATH_SEPARATOR_CONFUSABLES

    attack_filenames = {
        "../escape.md",
        "/tmp/escape.md",
        "C:\\escape.md",
        "C:drive-relative.md",
        "\\\\server\\share\\escape.md",
        "alternate:data.md",
        "fullwidth\uff0fseparator.md",
        "fullwidth-reverse\uff3cseparator.md",
        "fraction\u2044separator.md",
        "division\u2215separator.md",
        "set-minus\u2216separator.md",
        "box-forward\u2571separator.md",
        "box-reverse\u2572separator.md",
        "math-forward\u27cbseparator.md",
        "math-reverse\u27cdseparator.md",
        "reverse-operator\u29f5separator.md",
        "big-forward\u29f8separator.md",
        "big-reverse\u29f9separator.md",
        "triple\u2afbseparator.md",
        "double\u2afdseparator.md",
        "small-reverse\ufe68separator.md",
        "nul\x00name.md",
        " trailing.md",
        "trailing.md ",
        "trailing.md.",
        *(
            f"confusable-{ord(character):x}{character}separator.md"
            for character in PATH_SEPARATOR_CONFUSABLES
        ),
    }
    storage_root = Path(sample_knowledge_base.source_root)
    baseline_counts = _durable_side_effect_counts(
        db_session,
        sample_knowledge_base.id,
    )
    baseline_tree = _storage_tree(storage_root)

    async with await _asgi_client(db_session) as client:
        for filename in sorted(attack_filenames):
            body, boundary = _raw_multipart_body(filename, b"# rejected\n")
            response = await client.post(
                "files/upload",
                params={"knowledge_base_id": sample_knowledge_base.id},
                content=body,
                headers={
                    "Content-Type": (
                        f"multipart/form-data; boundary={boundary}"
                    )
                },
            )
            assert response.status_code == 400, (filename, response.text)
            assert (
                _durable_side_effect_counts(
                    db_session,
                    sample_knowledge_base.id,
                )
                == baseline_counts
            )
            assert _storage_tree(storage_root) == baseline_tree


@pytest.mark.asyncio
async def test_real_asgi_pre_admission_rejections_leave_no_durable_side_effects(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.core.config import get_settings

    storage_root = Path(sample_knowledge_base.source_root)
    baseline_counts = _durable_side_effect_counts(
        db_session,
        sample_knowledge_base.id,
    )
    baseline_tree = _storage_tree(storage_root)

    async with await _asgi_client(db_session) as client:
        cases = [
            {
                "files": {
                    "upload": (
                        "fake.pdf",
                        b"#!/bin/sh\necho not pdf\n",
                        "application/pdf",
                    )
                },
                "expected_status": 400,
            },
            {
                "files": {
                    "upload": (
                        "binary.txt",
                        b"\x00binary",
                        "text/plain",
                    )
                },
                "expected_status": 400,
            },
            {
                "files": {
                    "upload": (
                        "checksum.md",
                        b"# checksum\n",
                        "text/markdown",
                    )
                },
                "data": {"expected_sha256": "0" * 64},
                "expected_status": 400,
            },
            {
                "files": {
                    "upload": (
                        "checksum-format.md",
                        b"# checksum\n",
                        "text/markdown",
                    )
                },
                "data": {"expected_sha256": "not-a-sha256"},
                "expected_status": 400,
            },
            {
                "files": {
                    "upload": (
                        "payload.exe",
                        b"unsafe",
                        "application/octet-stream",
                    )
                },
                "expected_status": 400,
            },
        ]
        for case in cases:
            response = await client.post(
                "files/upload",
                params={"knowledge_base_id": sample_knowledge_base.id},
                files=case["files"],
                data=case.get("data"),
            )
            assert response.status_code == case["expected_status"], response.text
            assert (
                _durable_side_effect_counts(
                    db_session,
                    sample_knowledge_base.id,
                )
                == baseline_counts
            )
            assert _storage_tree(storage_root) == baseline_tree

        monkeypatch.setenv("UPLOAD_MAX_BYTES", "4")
        get_settings.cache_clear()
        response = await client.post(
            "files/upload",
            params={"knowledge_base_id": sample_knowledge_base.id},
            files={
                "upload": (
                    "oversized.md",
                    b"12345",
                    "text/markdown",
                )
            },
        )
        assert response.status_code == 413
        assert (
            _durable_side_effect_counts(
                db_session,
                sample_knowledge_base.id,
            )
            == baseline_counts
        )
        assert _storage_tree(storage_root) == baseline_tree


@pytest.mark.asyncio
async def test_storage_batch_and_executor_entries_replay_shared_content_validator_before_rows(
    db_session,
    sample_knowledge_base,
):
    from app.services import ingestion
    from app.services.storage import UploadValidationError

    storage_root = Path(sample_knowledge_base.source_root)
    fake_pdf = storage_root / "watchdog-batch-fake.pdf"
    fake_pdf.write_bytes(b"#!/bin/sh\necho not pdf\n")
    baseline_counts = _durable_side_effect_counts(
        db_session,
        sample_knowledge_base.id,
    )

    with pytest.raises(UploadValidationError):
        ingestion.collect_source_documents(storage_root)
    with pytest.raises(UploadValidationError):
        ingestion.create_uploaded_files_batch(
            db_session,
            sample_knowledge_base.id,
            [fake_pdf],
        )
    with pytest.raises(UploadValidationError):
        await ingestion.ingest_file(
            db_session,
            fake_pdf,
            trigger_source="watchdog",
            knowledge_base_id=sample_knowledge_base.id,
        )

    assert (
        _durable_side_effect_counts(
            db_session,
            sample_knowledge_base.id,
        )
        == baseline_counts
    )


def test_source_filter_scopes_excluded_parts_to_authorized_root(tmp_path):
    from app.services.ingestion import should_include_file

    storage_root = tmp_path / "scripts" / "authorized-storage"
    storage_root.mkdir(parents=True)
    allowed = storage_root / "allowed.md"
    allowed.write_text("allowed", encoding="utf-8")
    excluded = storage_root / "output" / "excluded.md"
    excluded.parent.mkdir()
    excluded.write_text("excluded", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")

    assert should_include_file(allowed, authorized_root=storage_root) is True
    assert should_include_file(excluded, authorized_root=storage_root) is False
    assert should_include_file(outside, authorized_root=storage_root) is False


def test_watchdog_dispatch_replays_shared_content_validator(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace

    from app.services.storage import UploadValidationError

    delay_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    worker_package = ModuleType("worker_app")
    worker_package.__path__ = []  # type: ignore[attr-defined]
    bootstrap_module = ModuleType("worker_app.bootstrap")
    bootstrap_module.API_ROOT = Path(__file__).resolve().parents[1]  # type: ignore[attr-defined]
    tasks_module = ModuleType("worker_app.tasks")
    tasks_module.ingest_path = SimpleNamespace(  # type: ignore[attr-defined]
        delay=lambda *args, **kwargs: delay_calls.append((args, kwargs))
    )
    watchdog_package = ModuleType("watchdog")
    watchdog_package.__path__ = []  # type: ignore[attr-defined]
    events_module = ModuleType("watchdog.events")
    events_module.FileSystemEventHandler = object  # type: ignore[attr-defined]
    observers_module = ModuleType("watchdog.observers")
    observers_module.Observer = object  # type: ignore[attr-defined]
    for name, module in {
        "worker_app": worker_package,
        "worker_app.bootstrap": bootstrap_module,
        "worker_app.tasks": tasks_module,
        "watchdog": watchdog_package,
        "watchdog.events": events_module,
        "watchdog.observers": observers_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    watcher_path = (
        Path(__file__).resolve().parents[2]
        / "worker"
        / "worker_app"
        / "watcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_upload_security_watcher_probe",
        watcher_path,
    )
    assert spec is not None and spec.loader is not None
    watcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher_module)
    monkeypatch.setattr(
        watcher_module,
        "get_settings",
        lambda: SimpleNamespace(
            storage_root_path=tmp_path,
            upload_max_bytes=1024,
        ),
    )
    handler = watcher_module.KnowledgeBaseEventHandler()

    fake_pdf = tmp_path / "watchdog-fake.pdf"
    fake_pdf.write_bytes(b"#!/bin/sh\necho not pdf\n")
    with pytest.raises(UploadValidationError):
        handler._handle(fake_pdf)
    assert delay_calls == []
    assert handler.cache == {}

    valid_pdf = tmp_path / "watchdog-valid.pdf"
    valid_pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    handler._handle(valid_pdf)
    assert delay_calls == [
        ((str(valid_pdf),), {"trigger_source": "watchdog"})
    ]


@pytest.mark.asyncio
async def test_upload_rejects_parent_traversal_filename_before_writing(db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.routers.ingestion import upload_file

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"].resolve()

    with pytest.raises(HTTPException) as exc_info:
        await upload_file(knowledge_base_id=sample_knowledge_base.id, upload=_upload("../escape.md", b"unsafe"), db=db_session)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_upload"
    assert _stored_files(storage_root) == []
    assert not (storage_root.parent / "escape.md").exists()


@pytest.mark.asyncio
async def test_upload_rejects_disallowed_suffix_before_writing(db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.routers.ingestion import upload_file

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"].resolve()

    with pytest.raises(HTTPException) as exc_info:
        await upload_file(knowledge_base_id=sample_knowledge_base.id, upload=_upload("payload.exe", b"unsafe"), db=db_session)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_upload"
    assert _stored_files(storage_root) == []


@pytest.mark.asyncio
async def test_upload_rejects_invalid_explicit_language_before_writing(
    db_session,
    sample_knowledge_base,
):
    from app.core.config import get_settings
    from app.routers.ingestion import upload_file

    storage_root = get_settings().knowledge_base_paths_for_name(
        sample_knowledge_base.name
    )["storage_root"].resolve()

    with pytest.raises(HTTPException) as exc_info:
        await upload_file(
            knowledge_base_id=sample_knowledge_base.id,
            upload=_upload("notes.md", b"safe"),
            language="not_valid!!!",
            db=db_session,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_language_metadata"
    assert _stored_files(storage_root) == []


@pytest.mark.asyncio
async def test_upload_enforces_streamed_byte_limit_and_cleans_partial_file(monkeypatch, db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.routers.ingestion import upload_file

    monkeypatch.setenv("UPLOAD_MAX_BYTES", "4")
    get_settings.cache_clear()
    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"].resolve()

    with pytest.raises(HTTPException) as exc_info:
        await upload_file(knowledge_base_id=sample_knowledge_base.id, upload=_upload("oversized.md", b"12345"), db=db_session)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == {"code": "upload_too_large", "message": "Upload exceeds the 4-byte limit"}
    assert _stored_files(storage_root) == []


@pytest.mark.asyncio
async def test_normal_allowed_upload_is_committed_and_registered(monkeypatch, db_session, sample_knowledge_base):
    import hashlib

    from app.core.config import get_settings
    from app.models import IngestionJob
    from app.routers.ingestion import upload_file
    from app.schemas import UploadFileResponse
    from app.services.storage import (
        UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION,
        upload_content_signature_protocol_hash,
    )

    content = b"safe"
    monkeypatch.setenv("UPLOAD_MAX_BYTES", str(len(content)))
    get_settings.cache_clear()
    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"].resolve()

    response = await upload_file(knowledge_base_id=sample_knowledge_base.id, upload=_upload("notes.MD", content), db=db_session)
    validated_response = UploadFileResponse.model_validate(response)

    stored_path = Path(response["source_path"]).resolve()
    assert response["status"] == "queued"
    assert response["document_id"]
    assert response["job_id"]
    assert response["language_identity"]["status"] == "pending"
    assert (
        response["language_identity"]["protocol_version"]
        == "document_language_unicode_script_v1"
    )
    assert response["language_identity"]["explicit_language_tag"] is None
    assert response["upload_replacement"]["database_committed"] is True
    assert response["upload_replacement"]["status"] == "completed"
    assert validated_response.upload_replacement.intent_id == response["upload_replacement"]["intent_id"]
    assert stored_path.suffix == ".md"
    assert len(stored_path.stem) == 64
    assert stored_path.parent.parent.name == "source_slots"
    assert stored_path.read_bytes() == content
    assert stored_path != storage_root and storage_root in stored_path.parents
    assert all(not path.name.endswith((".candidate", ".backup")) for path in storage_root.rglob("*"))
    job = db_session.get(IngestionJob, response["job_id"])
    validation = job.stats["upload_content_validation"]
    assert (
        validation["content_signature_protocol_version"]
        == UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION
    )
    assert (
        validation["content_signature_protocol_hash"]
        == upload_content_signature_protocol_hash()
    )
    assert validation["checksum"] == hashlib.sha256(content).hexdigest()
    assert validation["size_bytes"] == len(content)


def test_attempt_source_snapshots_are_immutable_and_checksum_addressed(tmp_path, sample_knowledge_base):
    from app.core.config import get_settings
    from app.services.storage import compute_checksum, snapshot_source_file

    paths = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)
    paths["storage_root"].mkdir(parents=True, exist_ok=True)
    source = paths["storage_root"] / "same-name.md"
    first_bytes = b"first active source"
    second_bytes = b"second candidate source"

    source.write_bytes(first_bytes)
    first_snapshot = snapshot_source_file(source, sample_knowledge_base.name)
    first_path, first_checksum = first_snapshot.canonical_path, first_snapshot.checksum
    source.write_bytes(second_bytes)
    second_snapshot = snapshot_source_file(source, sample_knowledge_base.name)
    second_path, second_checksum = second_snapshot.canonical_path, second_snapshot.checksum

    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes
    assert second_path.read_bytes() == second_bytes
    assert compute_checksum(first_path) == first_checksum
    assert compute_checksum(second_path) == second_checksum
    assert first_checksum in first_path.parts
    assert second_checksum in second_path.parts
    ingestion_root = paths["ingestion_root"].resolve()
    storage_root = paths["storage_root"].resolve()
    assert ingestion_root in first_path.parents
    assert storage_root not in first_path.parents

    reused_snapshot = snapshot_source_file(source, sample_knowledge_base.name)
    reused_path, reused_checksum = reused_snapshot.canonical_path, reused_snapshot.checksum
    assert reused_path == second_path
    assert reused_checksum == second_checksum
    assert not any(paths["ingestion_root"].rglob("*.snapshotting"))


@pytest.mark.asyncio
async def test_same_name_upload_restores_previous_bytes_when_registration_fails(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.routers import ingestion as ingestion_router

    first = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("replaceable.md", b"first registered bytes"),
        db=db_session,
    )
    stored_path = Path(first["source_path"])
    assert stored_path.read_bytes() == b"first registered bytes"

    def fail_registration(*_args, **_kwargs):
        raise RuntimeError("forced upload registration failure")

    monkeypatch.setattr(ingestion_router, "register_uploaded_file", fail_registration)
    with pytest.raises(RuntimeError, match="forced upload registration failure"):
        await ingestion_router.upload_file(
            knowledge_base_id=sample_knowledge_base.id,
            upload=_upload("replaceable.md", b"candidate bytes"),
            db=db_session,
        )

    assert stored_path.read_bytes() == b"first registered bytes"
    assert not any(stored_path.parent.glob(f".{stored_path.name}.*.backup"))
    assert not any(stored_path.parent.glob(f".{stored_path.name}.*.candidate"))


@pytest.mark.asyncio
async def test_same_name_upload_restores_previous_bytes_when_registration_commit_fails(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import Document
    from app.routers import ingestion as ingestion_router

    first = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("commit-failure.md", b"committed bytes"),
        db=db_session,
    )
    stored_path = Path(first["source_path"])
    document_id = first["document_id"]
    before_checksum = db_session.get(Document, document_id).checksum

    original_commit = db_session.commit

    def fail_commit():
        from app.models import IngestionCompensationLog
        from app.services.upload_replacement import UPLOAD_SOURCE_REPLACEMENT_OPERATION

        replacement_intents = db_session.query(IngestionCompensationLog).filter_by(
            operation=UPLOAD_SOURCE_REPLACEMENT_OPERATION,
        ).all()
        if any((row.payload_json or {}).get("phase") == "database_committed" for row in replacement_intents):
            raise RuntimeError("forced registration commit failure")
        original_commit()

    monkeypatch.setattr(db_session, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="forced registration commit failure"):
        await ingestion_router.upload_file(
            knowledge_base_id=sample_knowledge_base.id,
            upload=_upload("commit-failure.md", b"uncommitted candidate"),
            db=db_session,
        )

    assert stored_path.read_bytes() == b"committed bytes"
    db_session.expire_all()
    assert db_session.get(Document, document_id).checksum == before_checksum
    assert not any(stored_path.parent.glob(f".{stored_path.name}.*.backup"))


@pytest.mark.asyncio
async def test_unknown_commit_outcome_uses_durable_phase_instead_of_false_failure(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.routers import ingestion as ingestion_router
    from app.services.upload_replacement import UPLOAD_SOURCE_REPLACEMENT_OPERATION

    original_commit = db_session.commit
    injected = False

    def commit_then_report_connection_failure():
        nonlocal injected
        replacement_intents = db_session.query(IngestionCompensationLog).filter_by(
            operation=UPLOAD_SOURCE_REPLACEMENT_OPERATION,
        ).all()
        database_commit_staged = any(
            (row.payload_json or {}).get("phase") == "database_committed"
            for row in replacement_intents
        )
        original_commit()
        if database_commit_staged and not injected:
            injected = True
            raise RuntimeError("connection dropped after PostgreSQL commit")

    monkeypatch.setattr(db_session, "commit", commit_then_report_connection_failure)
    response = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("unknown-outcome.md", b"committed despite transport failure"),
        db=db_session,
    )

    assert injected is True
    assert response["status"] == "queued"
    assert Path(response["source_path"]).read_bytes() == b"committed despite transport failure"
    assert response["upload_replacement"]["database_committed"] is True
    db_session.expire_all()
    row = db_session.query(IngestionCompensationLog).filter_by(
        operation=UPLOAD_SOURCE_REPLACEMENT_OPERATION,
    ).one()
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_upload_has_no_fallible_refresh_after_registration_commit(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.routers import ingestion as ingestion_router

    def fail_refresh(*_args, **_kwargs):
        raise RuntimeError("post-commit refresh must not run")

    monkeypatch.setattr(db_session, "refresh", fail_refresh)
    response = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("no-post-commit-refresh.md", b"durable candidate"),
        db=db_session,
    )

    assert response["status"] == "queued"
    assert Path(response["source_path"]).read_bytes() == b"durable candidate"


@pytest.mark.asyncio
async def test_upload_intent_is_committed_before_candidate_file_is_created(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import upload_replacement
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    observed: dict[str, object] = {}

    async def inspect_durable_intent(
        _upload,
        candidate_path,
        *,
        max_bytes,
        expected_checksum,
        expected_size_bytes,
    ):
        del max_bytes
        intent_id = candidate_path.name.rsplit(".", 2)[-2]
        row = db_session.get(IngestionCompensationLog, intent_id)
        observed.update(
            {
                "status": row.status,
                "phase": row.payload_json["phase"],
                "candidate_exists": candidate_path.exists(),
                "accepted_checksum": row.payload_json["candidate"]["checksum"],
                "accepted_size_bytes": row.payload_json["candidate"]["size_bytes"],
                "write_expected_checksum": expected_checksum,
                "write_expected_size_bytes": expected_size_bytes,
            }
        )
        raise RuntimeError("fault after durable intent")

    monkeypatch.setattr(upload_replacement, "write_upload_candidate", inspect_durable_intent)
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_intent_fault_test",
    ):
        with pytest.raises(RuntimeError, match="fault after durable intent"):
            await upload_replacement.begin_upload_source_replacement(
                db_session,
                sample_knowledge_base,
                _upload("intent-first.md", b"candidate"),
            )

    assert observed == {
        "status": "pending",
        "phase": "intent_committed",
        "candidate_exists": False,
        "accepted_checksum": observed["write_expected_checksum"],
        "accepted_size_bytes": observed["write_expected_size_bytes"],
        "write_expected_checksum": observed["write_expected_checksum"],
        "write_expected_size_bytes": observed["write_expected_size_bytes"],
    }


@pytest.mark.asyncio
async def test_failed_initial_intent_commit_leaves_no_storage_namespace(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import upload_replacement
    from app.services.ingestion_resource_lock import (
        knowledge_base_ingestion_resource_lock,
    )

    root = Path(sample_knowledge_base.source_root)
    baseline_tree = _storage_tree(root)
    original_commit = db_session.commit
    injected = False

    def fail_initial_intent_commit():
        nonlocal injected
        if not injected:
            injected = True
            raise RuntimeError("forced initial intent commit failure")
        original_commit()

    monkeypatch.setattr(db_session, "commit", fail_initial_intent_commit)
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_initial_intent_commit_failure_test",
    ):
        with pytest.raises(RuntimeError, match="initial intent commit failure"):
            await upload_replacement.begin_upload_source_replacement(
                db_session,
                sample_knowledge_base,
                _upload("intent-commit-failure.md", b"must never reach storage"),
            )

    monkeypatch.setattr(db_session, "commit", original_commit)
    db_session.rollback()
    assert injected is True
    assert _storage_tree(root) == baseline_tree
    assert (
        db_session.query(IngestionCompensationLog)
        .filter_by(
            knowledge_base_id=sample_knowledge_base.id,
            operation=upload_replacement.UPLOAD_SOURCE_REPLACEMENT_OPERATION,
        )
        .count()
        == 0
    )


@pytest.mark.asyncio
async def test_manifest_bound_upload_checksum_mismatch_never_installs_candidate(
    db_session,
    sample_knowledge_base,
):
    import hashlib

    from app.models import IngestionCompensationLog
    from app.services.ingestion_resource_lock import (
        knowledge_base_ingestion_resource_lock,
    )
    from app.services.storage import UploadValidationError
    from app.services.upload_replacement import (
        UPLOAD_SOURCE_REPLACEMENT_OPERATION,
        begin_upload_source_replacement,
    )

    expected_checksum = hashlib.sha256(b"manifest bytes").hexdigest()
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="manifest_bound_upload_test",
    ):
        with pytest.raises(
            UploadValidationError,
            match="manifest-bound expected checksum",
        ):
            await begin_upload_source_replacement(
                db_session,
                sample_knowledge_base,
                _upload("manifest-drift.md", b"changed after manifest"),
                expected_checksum=expected_checksum,
            )

    intent_count = (
        db_session.query(IngestionCompensationLog)
        .filter(
            IngestionCompensationLog.operation
            == UPLOAD_SOURCE_REPLACEMENT_OPERATION
        )
        .count()
    )
    assert intent_count == 0
    storage_root = Path(sample_knowledge_base.source_root)
    assert _stored_files(storage_root) == []


@pytest.mark.asyncio
async def test_candidate_installed_crash_state_reconciles_to_previous_bytes(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
    from app.services.upload_replacement import (
        begin_upload_source_replacement,
        reconcile_upload_source_replacement,
    )
    from app.services.storage import build_storage_path

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_crash_reconcile_test",
    ):
        target = build_storage_path("crash-window.md", sample_knowledge_base.name)
        target.write_bytes(b"prior bytes")

        replacement = await begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("crash-window.md", b"uncommitted candidate"),
        )
        row = db_session.get(IngestionCompensationLog, replacement.intent_id)
        assert row.payload_json["phase"] == "candidate_installed"
        assert replacement.target.read_bytes() == b"uncommitted candidate"
        assert Path(row.payload_json["backup_path"]).read_bytes() == b"prior bytes"

        result = reconcile_upload_source_replacement(db_session, replacement.intent_id)

    assert result["status"] == "rolled_back"
    assert replacement.target.read_bytes() == b"prior bytes"
    assert not Path(row.payload_json["backup_path"]).exists()


@pytest.mark.asyncio
async def test_terminal_status_with_nonterminal_phase_fails_closed_before_recovery(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import upload_replacement
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_invalid_terminal_state_test",
    ):
        target = upload_replacement.build_storage_path(
            "invalid-terminal-state.md",
            sample_knowledge_base.name,
        )
        target.write_bytes(b"prior bytes")
        replacement = await upload_replacement.begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("invalid-terminal-state.md", b"uncommitted candidate"),
        )
        row = db_session.get(IngestionCompensationLog, replacement.intent_id)
        assert row.payload_json["phase"] == "candidate_installed"
        row.status = "completed"
        db_session.commit()

        with pytest.raises(upload_replacement.UploadReplacementRecoveryError):
            upload_replacement.reconcile_upload_source_replacement(
                db_session,
                replacement.intent_id,
            )

        db_session.expire_all()
        failed_closed = db_session.get(IngestionCompensationLog, replacement.intent_id)
        assert failed_closed.status == "manual_review"
        assert failed_closed.payload_json["phase"] == "candidate_installed"

        recovered = upload_replacement.reconcile_upload_source_replacement(
            db_session,
            replacement.intent_id,
        )

    assert recovered["status"] == "rolled_back"
    assert target.read_bytes() == b"prior bytes"


@pytest.mark.asyncio
async def test_pre_phase_commit_crash_state_is_reconciled_from_checksums(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import upload_replacement
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_pre_phase_commit_crash_test",
    ):
        target = upload_replacement.build_storage_path("phase-crash.md", sample_knowledge_base.name)
        target.write_bytes(b"prior phase bytes")
        replacement = await upload_replacement.begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("phase-crash.md", b"candidate phase bytes"),
        )
        row = db_session.get(IngestionCompensationLog, replacement.intent_id)
        candidate = Path(row.payload_json["candidate_path"])
        backup = Path(row.payload_json["backup_path"])

        # Directly construct the observable crash window: target->backup was
        # durable, but PostgreSQL still reports candidate_ready because the
        # next phase commit never returned.
        upload_replacement.durable_replace(target, candidate)
        payload = dict(row.payload_json)
        payload = upload_replacement._transition_payload(payload, "candidate_ready")
        row.payload_json = payload
        row.status = "pending"
        db_session.commit()

        result = upload_replacement.reconcile_upload_source_replacement(
            db_session,
            replacement.intent_id,
        )

    assert result["status"] == "rolled_back"
    assert target.read_bytes() == b"prior phase bytes"
    assert not candidate.exists()
    assert not backup.exists()


@pytest.mark.asyncio
async def test_next_same_path_upload_reconciles_interrupted_predecessor(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
    from app.services.upload_replacement import (
        begin_upload_source_replacement,
        reconcile_upload_source_replacement,
    )
    from app.services.storage import build_storage_path

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_same_path_recovery_test",
    ):
        target = build_storage_path("same-path-recovery.md", sample_knowledge_base.name)
        target.write_bytes(b"prior")
        interrupted = await begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("same-path-recovery.md", b"interrupted"),
        )
        successor = await begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("same-path-recovery.md", b"successor"),
        )
        db_session.expire_all()
        predecessor_row = db_session.get(IngestionCompensationLog, interrupted.intent_id)
        assert predecessor_row.status == "rolled_back"
        assert successor.target.read_bytes() == b"successor"
        reconcile_upload_source_replacement(db_session, successor.intent_id)

    assert successor.target.read_bytes() == b"prior"


@pytest.mark.asyncio
async def test_startup_reconcile_restores_interrupted_upload(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
    from app.services.storage import build_storage_path
    from app.services.upload_replacement import (
        begin_upload_source_replacement,
        reconcile_pending_upload_replacements_startup,
    )

    target = build_storage_path("startup-recovery.md", sample_knowledge_base.name)
    target.write_bytes(b"startup prior")
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_startup_recovery_setup",
    ):
        interrupted = await begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("startup-recovery.md", b"interrupted at process exit"),
        )

    stats = await reconcile_pending_upload_replacements_startup()

    db_session.expire_all()
    row = db_session.get(IngestionCompensationLog, interrupted.intent_id)
    assert stats["rolled_back"] == 1
    assert row.status == "rolled_back"
    assert target.read_bytes() == b"startup prior"


@pytest.mark.asyncio
async def test_rollback_failure_is_durable_and_retryable(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import upload_replacement
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_rollback_failure_test",
    ):
        target = upload_replacement.build_storage_path("rollback-failure.md", sample_knowledge_base.name)
        target.write_bytes(b"prior")
        replacement = await upload_replacement.begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("rollback-failure.md", b"candidate"),
        )
        original_replace = upload_replacement.durable_replace

        def fail_restore(source, destination):
            if destination == replacement.target:
                raise OSError("forced atomic restore failure")
            original_replace(source, destination)

        monkeypatch.setattr(upload_replacement, "durable_replace", fail_restore)
        with pytest.raises(upload_replacement.UploadReplacementRecoveryError):
            upload_replacement.rollback_upload_replacement(db_session, replacement)

        db_session.expire_all()
        failed = db_session.get(IngestionCompensationLog, replacement.intent_id)
        assert failed.status == "manual_review"
        assert failed.payload_json["phase"] == "candidate_installed"
        assert "forced atomic restore failure" in failed.error_message

        monkeypatch.setattr(upload_replacement, "durable_replace", original_replace)
        retried = upload_replacement.reconcile_upload_source_replacement(
            db_session,
            replacement.intent_id,
        )

    assert retried["status"] == "rolled_back"
    assert replacement.target.read_bytes() == b"prior"


@pytest.mark.asyncio
async def test_candidate_symlink_never_deletes_referenced_storage_file(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import upload_replacement
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="upload_candidate_symlink_test",
    ):
        target = upload_replacement.build_storage_path("symlink-guard.md", sample_knowledge_base.name)
        target.write_bytes(b"prior")
        victim = target.with_name("unrelated-victim.md")
        victim.write_bytes(b"must survive")
        replacement = await upload_replacement.begin_upload_source_replacement(
            db_session,
            sample_knowledge_base,
            _upload("symlink-guard.md", b"candidate"),
        )
        row = db_session.get(IngestionCompensationLog, replacement.intent_id)
        candidate = Path(row.payload_json["candidate_path"])
        try:
            candidate.symlink_to(victim)
        except OSError as exc:
            pytest.skip(f"symlink creation is unavailable for this test user: {exc}")

        with pytest.raises(upload_replacement.UploadReplacementRecoveryError):
            upload_replacement.reconcile_upload_source_replacement(
                db_session,
                replacement.intent_id,
            )

    assert victim.read_bytes() == b"must survive"
    assert target.read_bytes() == b"prior"


def test_durable_replace_never_reports_success_when_directory_barrier_fails(
    monkeypatch,
    tmp_path,
):
    from app.services import storage

    source = tmp_path / "rename-source"
    target = tmp_path / "rename-target"
    source.write_bytes(b"candidate")

    def fail_barrier(_directory):
        raise storage.DirectoryDurabilityError("forced directory flush failure")

    monkeypatch.setattr(storage, "durable_sync_directory", fail_barrier)
    with pytest.raises(storage.DirectoryDurabilityError, match="forced directory flush failure"):
        storage.durable_replace(source, target)


@pytest.mark.asyncio
async def test_postcommit_resource_lock_release_failure_returns_success_with_audit(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from contextlib import asynccontextmanager

    from app.models import IngestionCompensationLog
    from app.routers import ingestion as ingestion_router
    from app.services.ingestion_resource_lock import IngestionResourceLockReleaseError
    from app.services.upload_replacement import UPLOAD_SOURCE_REPLACEMENT_OPERATION

    original_lock = ingestion_router.knowledge_base_ingestion_resource_lock
    diagnostics = {
        "resource_key": f"knowledge_base:{sample_knowledge_base.id}",
        "knowledge_base_id": sample_knowledge_base.id,
        "advisory_key": 123,
        "backend": "postgresql",
        "operation": "upload_registration",
        "batch_id": None,
        "protocol_version": "postgres_advisory_kb_v1",
        "release_error": "forced_test_failure",
    }

    @asynccontextmanager
    async def fail_after_release(*args, **kwargs):
        async with original_lock(*args, **kwargs) as lease:
            yield lease
        raise IngestionResourceLockReleaseError(diagnostics)

    monkeypatch.setattr(ingestion_router, "knowledge_base_ingestion_resource_lock", fail_after_release)
    response = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("release-failure.md", b"committed"),
        db=db_session,
    )

    assert response["status"] == "queued"
    assert response["upload_replacement"]["database_committed"] is True
    assert response["upload_replacement"]["postcommit_lock_release_failure"] == diagnostics
    assert response["upload_replacement"]["lock_release_audit"]["persisted"] is True
    db_session.expire_all()
    row = db_session.query(IngestionCompensationLog).filter_by(
        operation=UPLOAD_SOURCE_REPLACEMENT_OPERATION,
    ).one()
    assert "lock release failed" in row.error_message


@pytest.mark.asyncio
async def test_committed_upload_cleanup_failure_does_not_turn_response_into_failure(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.routers import ingestion as ingestion_router
    from app.services import upload_replacement
    from app.services.upload_replacement import UPLOAD_SOURCE_REPLACEMENT_OPERATION

    first = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("cleanup-pending.md", b"prior"),
        db=db_session,
    )
    target = Path(first["source_path"])
    original_unlink = upload_replacement.durable_unlink

    def fail_backup_cleanup(path, *, missing_ok=False):
        del missing_ok
        if path.name.endswith(".backup"):
            raise OSError("forced post-commit cleanup failure")
        original_unlink(path)

    monkeypatch.setattr(upload_replacement, "durable_unlink", fail_backup_cleanup)
    response = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("cleanup-pending.md", b"registered replacement"),
        db=db_session,
    )

    assert response["status"] == "queued"
    assert target.read_bytes() == b"registered replacement"
    assert response["upload_replacement"]["status"] == "cleanup_pending"
    assert response["upload_replacement"]["cleanup_pending"] is True
    db_session.expire_all()
    pending = (
        db_session.query(IngestionCompensationLog)
        .filter_by(operation=UPLOAD_SOURCE_REPLACEMENT_OPERATION, status="cleanup_pending")
        .one()
    )
    assert pending.payload_json["phase"] == "cleanup_pending"
    assert pending.job_id == response["job_id"]
    assert "forced post-commit cleanup failure" in pending.error_message

    with pytest.raises(HTTPException) as blocked:
        await ingestion_router.upload_file(
            knowledge_base_id=sample_knowledge_base.id,
            upload=_upload("cleanup-pending.md", b"must not overtake pending cleanup"),
            db=db_session,
        )
    assert blocked.value.status_code == 409
    assert blocked.value.detail["code"] == "upload_recovery_pending"

    monkeypatch.setattr(upload_replacement, "durable_unlink", original_unlink)
    recovered = await ingestion_router.upload_file(
        knowledge_base_id=sample_knowledge_base.id,
        upload=_upload("cleanup-pending.md", b"successor after cleanup recovery"),
        db=db_session,
    )
    assert recovered["upload_replacement"]["status"] == "completed"
    assert target.read_bytes() == b"successor after cleanup recovery"


def test_source_snapshot_uses_atomic_durable_no_clobber_publish(
    monkeypatch,
    tmp_path,
    sample_knowledge_base,
):
    from app.core.config import get_settings
    from app.services import storage

    source_root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    source_root.mkdir(parents=True, exist_ok=True)
    source = source_root / "durable-snapshot.md"
    source.write_bytes(b"snapshot")
    calls: list[tuple[Path, Path]] = []
    original_publish = storage._durable_publish_noreplace

    def observe_publish(source_path, target_path, **kwargs):
        calls.append((source_path, target_path))
        return original_publish(source_path, target_path, **kwargs)

    monkeypatch.setattr(storage, "_durable_publish_noreplace", observe_publish)
    frozen_snapshot = storage.snapshot_source_file(source, sample_knowledge_base.name)
    snapshot_path, checksum = frozen_snapshot.canonical_path, frozen_snapshot.checksum

    assert len(calls) == 1
    assert calls[0][1] == snapshot_path
    assert checksum == storage.compute_checksum(snapshot_path)


def test_source_snapshot_is_readonly_and_has_no_ungated_delete_api(
    tmp_path,
    sample_knowledge_base,
):
    from app.core.config import get_settings
    from app.services import storage

    source_root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    source_root.mkdir(parents=True, exist_ok=True)
    source = source_root / "readonly-snapshot.md"
    source.write_bytes(b"immutable bytes")
    frozen_snapshot = storage.snapshot_source_file(source, sample_knowledge_base.name)
    snapshot_path, checksum = frozen_snapshot.canonical_path, frozen_snapshot.checksum

    assert stat.S_IMODE(snapshot_path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        with pytest.raises(OSError):
            snapshot_path.write_bytes(b"forbidden overwrite")
    assert checksum == storage.compute_checksum(snapshot_path)
    assert not hasattr(storage, "remove_source_snapshot_for_maintenance")
