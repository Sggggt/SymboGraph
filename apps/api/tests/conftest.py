from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def no_fallback_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("knowledge_base_name", "Unit Test KnowledgeBase")
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
def sample_knowledge_base(db_session):
    from app.models import KnowledgeBase

    KnowledgeBase = KnowledgeBase(name="Unit Test KnowledgeBase", description="tests", source_root="unit-tests")
    db_session.add(KnowledgeBase)
    db_session.commit()
    db_session.refresh(KnowledgeBase)
    return KnowledgeBase


@pytest.fixture
def indexed_chunks(db_session, sample_knowledge_base):
    from app.models import (
        ActiveChunk,
        ChunkCandidate,
        ChunkDecision,
        Document,
        DocumentVersion,
        EvidenceAtom,
        EvidenceGraphState,
        PolicyState,
        QualityDecision,
    )
    from app.services.evidence_graph import stable_hash

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Centrality sources",
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
    policy_state = PolicyState(
        knowledge_base_id=sample_knowledge_base.id,
        profile_objective_hash="unit-profile",
        posterior_json={},
        constraints_json={},
        exploration_json={},
        reward_summary_json={},
        state_hash="unit-policy",
    )
    db_session.add(policy_state)
    db_session.flush()
    source_texts = [
        "Degree centrality counts the number of incident edges for a node.",
        "Betweenness centrality measures how often a node lies on shortest paths.",
    ]
    atoms = []
    for index, text in enumerate(source_texts):
        atom = EvidenceAtom(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            atom_index=index,
            atom_type="paragraph",
            text=text,
            text_hash=stable_hash({"text": text}),
            source_span_json={"spans": [{"start": 0, "end": len(text), "section": "Centrality"}]},
            layout_json={},
            metadata_json={"section": "Centrality", "section_index": 0},
            state="active",
        )
        db_session.add(atom)
        atoms.append(atom)
    db_session.flush()
    graph_state = EvidenceGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        scope_type="global",
        state_hash="unit-graph",
        atom_scope_hash="unit-atoms",
        active_document_version_ids=[version.id],
        active_atom_ids=[atom.id for atom in atoms],
        policy_state_id=policy_state.id,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db_session.add(graph_state)
    db_session.flush()

    chunks = []
    for index, (text, atom) in enumerate(zip(source_texts, atoms)):
        candidate = ChunkCandidate(
            graph_state_id=graph_state.id,
            generator_name="unit_fixture",
            generator_version="unit_v1",
            atom_ids_json=[atom.id],
            source_span_union_json={"spans": [atom.source_span_json["spans"][0]]},
            token_count=12,
            graph_features_json={"fixture": True},
            cost_json={},
            diagnostics_json={},
        )
        db_session.add(candidate)
        db_session.flush()
        quality = QualityDecision(
            candidate_id=candidate.id,
            policy_state_id=policy_state.id,
            decision_action="answer_candidate",
            gate_passed=True,
            confidence=1.0,
            diagnostics_json={},
            reward_features_json={},
            feedback_json={},
        )
        db_session.add(quality)
        db_session.flush()
        decision = ChunkDecision(
            knowledge_base_id=sample_knowledge_base.id,
            graph_state_id=graph_state.id,
            candidate_id=candidate.id,
            quality_decision_id=quality.id,
            policy_state_id=policy_state.id,
            action="activate",
        )
        db_session.add(decision)
        db_session.flush()
        snippet = "Degree centrality counts incident edges." if index == 0 else "Betweenness centrality uses shortest paths."
        active_chunk = ActiveChunk(
            knowledge_base_id=sample_knowledge_base.id,
            chunk_decision_id=decision.id,
            document_version_scope_hash="unit-version-scope",
            graph_state_hash=graph_state.state_hash,
            atom_ids_json=[atom.id],
            text=text,
            source_span_union_json={"spans": [atom.source_span_json["spans"][0]]},
            boundary_policy_version="unit_fixture_v1",
            quality_decision_id=quality.id,
            policy_state_id=policy_state.id,
            community_ids_json=[],
            metadata_json={
                "document_id": document.id,
                "document_version_id": version.id,
                "chunk_version": 1,
                "partition": "L3",
                "section": "Centrality",
                "source_type": "markdown",
                "content_kind": "markdown",
                "snippet": snippet,
                "is_parent": False,
            },
            state="active",
        )
        db_session.add(active_chunk)
        chunks.append(active_chunk)
    db_session.commit()
    for item in (document, version, *chunks):
        db_session.refresh(item)
    return document, chunks
