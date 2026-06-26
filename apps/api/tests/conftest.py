from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture
def no_fallback_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("KNOWLEDGE_BASE_NAME", "Unit Test KnowledgeBase")
    monkeypatch.setenv("CHAT_API_KEY", "unit-test-chat-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat.invalid/v1")
    monkeypatch.setenv("GRAPH_API_KEY", "unit-test-graph-key")
    monkeypatch.setenv("GRAPH_BASE_URL", "https://graph.invalid/v1")
    monkeypatch.setenv("GRAPH_MODEL", "unit-test-graph-model")
    monkeypatch.setenv("EMBEDDING_API_KEY", "unit-test-embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.invalid/v1")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "8")
    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_DATABASE_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_AUTO_TPE", "false")
    monkeypatch.setenv("TPE_TRIAL_BUDGET", "6")
    monkeypatch.setenv("TPE_STARTUP_RANDOM_TRIALS", "3")
    monkeypatch.setenv("TPE_PROBE_QUERY_BUDGET", "6")
    monkeypatch.setenv("TPE_CANDIDATE_POOL_SIZE", "24")
    from app.core import config as config_module

    def read_test_env() -> dict[str, str]:
        return {key.upper(): value for key, value in os.environ.items()}

    monkeypatch.setattr(config_module, "_read_workspace_env", read_test_env)
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
    if db.engine.url.drivername != "sqlite":
        raise RuntimeError(f"Refusing to reset non-sqlite test database: {db.engine.url}")
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

    knowledge_base = KnowledgeBase(name="Unit Test KnowledgeBase", description="tests", source_root="unit-tests")
    db_session.add(knowledge_base)
    db_session.commit()
    db_session.refresh(knowledge_base)
    return knowledge_base


def _unit_vector(text: str, dimensions: int = 8) -> list[float]:
    values = []
    for index in range(dimensions):
        digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
        values.append((int.from_bytes(digest[:4], "big") % 1000) / 1000.0 + 0.001)
    magnitude = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / magnitude for value in values]


@pytest.fixture
def fake_model_stack(monkeypatch: pytest.MonkeyPatch):
    from app.services import agent_graph, context_graph
    from app.services.embeddings import ChatCallResult

    class FakeEmbeddingProvider:
        async def embed_texts(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
            return [_unit_vector(f"{text_type}:{text}") for text in texts]

    class FakeVectorStore:
        points: dict[str, dict] = {}

        def __init__(self, knowledge_base_name: str | None = None, collection_name: str | None = None) -> None:
            self.collection_name = collection_name or "unit"

        async def async_upsert(self, points: list[dict]) -> None:
            for point in points:
                self.points[point["id"]] = {**point, "collection": self.collection_name}

        def delete(self, ids: list[str]) -> None:
            for point_id in ids:
                self.points.pop(point_id, None)

        def list_ids(self, knowledge_base_id: str | None = None) -> list[str]:
            ids = []
            for point_id, point in self.points.items():
                if knowledge_base_id and (point.get("payload") or {}).get("knowledge_base_id") != knowledge_base_id:
                    continue
                ids.append(point_id)
            return ids

        def health_check(self, knowledge_base_id: str, active_chunk_ids: list[str]) -> dict:
            vector_ids = set(self.list_ids(knowledge_base_id))
            active_ids = set(active_chunk_ids)
            return {"ok": active_ids.issubset(vector_ids), "missing": sorted(active_ids - vector_ids), "stale": sorted(vector_ids - active_ids)}

    class FakeChatProvider:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict | None = None) -> dict:
            if "query facet extractor" in system_prompt:
                return {
                    "domain_facets": [{"facet": "Bayesian network", "aliases": ["Bayesian networks"]}],
                    "procedure_facets": [{"facet": "factorization", "aliases": ["conditional probability factorization"]}],
                    "answer_shape": "grounded_answer",
                }
            return fallback or {"label": "Unit concept", "definition": "Unit definition"}

        async def answer_question_with_meta(self, question: str, contexts: list[dict], history: list[dict] | None = None, context_quality: str = "normal"):
            first = contexts[0]["content"] if contexts else "no context"
            return ChatCallResult(answer=f"Grounded answer: {first[:120]}", provider="unit_chat", model="unit-chat", external_called=False)

    monkeypatch.setattr(context_graph, "EmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(context_graph, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(context_graph, "ChatProvider", FakeChatProvider)
    monkeypatch.setattr(agent_graph, "ChatProvider", FakeChatProvider)
    return {"EmbeddingProvider": FakeEmbeddingProvider, "VectorStore": FakeVectorStore, "ChatProvider": FakeChatProvider}


@pytest_asyncio.fixture
async def populated_context_graph(db_session, sample_knowledge_base, fake_model_stack):
    from app.models import Document, DocumentVersion
    from app.services.context_graph import rebuild_context_graph, write_chunks_and_structure, write_contextual_indexes
    from app.services.parsers import ParsedSection

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Bayesian networks",
        source_path="bayesian-networks.md",
        source_type="markdown",
        tags=["Bayesian"],
        checksum="checksum",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version=1, checksum="checksum", storage_path="bayesian-networks.md", is_active=True)
    db_session.add(version)
    db_session.flush()
    sections = [
        ParsedSection(
            title="Bayesian networks",
            text=(
                "# Bayesian networks\n"
                "Bayesian networks represent variables as nodes and conditional dependence as directed edges. "
                "Inference combines prior probability, likelihood, and observed evidence to update posterior beliefs."
            ),
            page_number=1,
            section="Bayesian networks",
        ),
        ParsedSection(
            title="Markov blanket",
            text=(
                "A Markov blanket contains parents, children, and co-parents. "
                "Given the Markov blanket, a node is conditionally independent from the rest of the graph."
            ),
            page_number=2,
            section="Bayesian networks > Markov blanket",
        ),
        ParsedSection(
            title="Factorization",
            text=(
                "$$P(X_1, X_2, ..., X_n)=\\prod_i P(X_i | Pa(X_i))$$\n"
                "| variable | role |\n| X_i | node |\n| Pa(X_i) | parents |\n"
                "The table and formula must remain addressable as structure neighbors."
            ),
            page_number=3,
            section="Bayesian networks > Factorization",
        ),
    ]
    chunks = write_chunks_and_structure(
        db_session,
        knowledge_base=sample_knowledge_base,
        document=document,
        version=version,
        sections=sections,
        chunk_version=1,
        chunk_size=24,
        chunk_overlap=4,
    )
    await write_contextual_indexes(db_session, knowledge_base=sample_knowledge_base, chunks=chunks)
    state = await rebuild_context_graph(db_session, sample_knowledge_base.id)
    sample_knowledge_base.current_chunk_version = 1
    db_session.commit()
    return {"knowledge_base": sample_knowledge_base, "document": document, "version": version, "chunks": chunks, "state": state}
