from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def no_fallback_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("COURSE_NAME", "Unit Test Course")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://api.openai.test/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "unit-test-embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.openai.test/v1")
    monkeypatch.setenv("ENABLE_AGENTIC_REFLECTION", "true")
    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_DATABASE_FALLBACK", "false")
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield data_root
    get_settings.cache_clear()


@pytest.fixture
def db_session(no_fallback_env: Path):
    from app.core.config import get_settings
    import app.db as db
    import app.models  # noqa: F401

    get_settings.cache_clear()
    db.settings = get_settings()
    db.engine.dispose()
    db.engine = db.build_engine()
    engine_url = db.engine.url
    if engine_url.drivername != "sqlite":
        raise RuntimeError(f"Refusing to run unit-test schema reset against non-sqlite database: {engine_url}")
    db.SessionLocal.configure(bind=db.engine)
    db.Base.metadata.drop_all(bind=db.engine)
    db.Base.metadata.create_all(bind=db.engine)

    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()
        db.engine.dispose()
        get_settings.cache_clear()


@pytest.fixture
def sample_course(db_session):
    from app.models import Course

    course = Course(name="Unit Test Course", description="tests", source_root="unit-tests")
    db_session.add(course)
    db_session.commit()
    db_session.refresh(course)
    return course


@pytest.fixture
def indexed_chunks(db_session, sample_course):
    from app.models import Chunk, Document, DocumentVersion

    document = Document(
        course_id=sample_course.id,
        title="Centrality Notes",
        source_path="centrality.md",
        source_type="markdown",
        tags=["L3"],
        checksum="checksum",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="checksum",
        storage_path="centrality.md",
        extracted_path=None,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunks = [
        Chunk(
            course_id=sample_course.id,
            document_id=document.id,
            document_version_id=version.id,
            content="Degree centrality counts the number of incident edges for a node.",
            snippet="Degree centrality counts incident edges.",
            chapter="L3",
            section="Centrality",
            source_type="markdown",
            metadata_json={"content_kind": "markdown"},
            embedding_status="ready",
        ),
        Chunk(
            course_id=sample_course.id,
            document_id=document.id,
            document_version_id=version.id,
            content="Betweenness centrality measures how often a node lies on shortest paths.",
            snippet="Betweenness centrality uses shortest paths.",
            chapter="L3",
            section="Centrality",
            source_type="markdown",
            metadata_json={"content_kind": "markdown"},
            embedding_status="ready",
        ),
    ]
    db_session.add_all(chunks)
    db_session.commit()
    for item in (document, version, *chunks):
        db_session.refresh(item)
    return document, chunks
