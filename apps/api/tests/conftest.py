from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def explicit_test_storage_durability_adapter():
    """Tests opt into a no-op namespace adapter without production env switches."""

    from app.services.storage import _use_explicit_test_namespace_durability_adapter

    with _use_explicit_test_namespace_durability_adapter():
        yield


@pytest.fixture
def no_fallback_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    root_env = tmp_path / ".env"
    # DATA_ROOT is deployment-provisioned; production code must never create
    # it before proving the filesystem capability contract.
    data_root.mkdir(parents=True, exist_ok=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("KNOWLEDGE_BASE_NAME", "Unit Test KnowledgeBase")
    monkeypatch.setenv("CHAT_API_KEY", "unit-test-chat-key")
    monkeypatch.setenv("CHAT_API_PROTOCOL", "openai")
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat.invalid/v1")
    monkeypatch.setenv("GRAPH_API_KEY", "unit-test-graph-key")
    monkeypatch.setenv("GRAPH_API_PROTOCOL", "openai")
    monkeypatch.setenv("GRAPH_BASE_URL", "https://graph.invalid/v1")
    monkeypatch.setenv("GRAPH_MODEL", "unit-test-graph-model")
    monkeypatch.setenv("EMBEDDING_API_KEY", "unit-test-embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.invalid/v1")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "8")
    # Offline unit fixtures must never inherit a developer workstation's
    # enabled host bridge or admin credential. Bridge-specific tests opt in
    # explicitly with their own closed configuration.
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "false")
    monkeypatch.delenv("MODEL_BRIDGE_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_DATABASE_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_AUTO_TPE", "false")
    monkeypatch.setenv("TPE_TRIAL_BUDGET", "6")
    monkeypatch.setenv("TPE_STARTUP_RANDOM_TRIALS", "3")
    monkeypatch.setenv("TPE_PROBE_QUERY_BUDGET", "6")
    monkeypatch.setenv("TPE_CANDIDATE_POOL_SIZE", "24")
    root_env.write_text(
        "\n".join(
            [
                "MODEL_BRIDGE_ENABLED=false",
                "CHAT_API_PROTOCOL=openai",
                "GRAPH_API_PROTOCOL=openai",
                "ENABLE_MODEL_FALLBACK=false",
                "ENABLE_DATABASE_FALLBACK=false",
                "ENABLE_AUTO_TPE=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_ENV_FILE", str(root_env))
    from app.core import config as config_module

    def read_test_env() -> dict[str, str]:
        return {key.upper(): value for key, value in os.environ.items()}

    monkeypatch.setattr(config_module, "_read_workspace_env", read_test_env)
    from app.services import runtime_settings

    monkeypatch.setattr(runtime_settings, "ENV_PATH", root_env)
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
    from app.core.config import get_settings
    from app.models import KnowledgeBase

    name = "Unit Test KnowledgeBase"
    source_root = get_settings().knowledge_base_paths_for_name(name)["storage_root"]
    source_root.mkdir(parents=True, exist_ok=True)
    source_root.parent.joinpath("ingestion").mkdir(parents=True, exist_ok=True)
    knowledge_base = KnowledgeBase(
        name=name,
        description="tests",
        source_root=str(source_root),
    )
    db_session.add(knowledge_base)
    db_session.commit()
    db_session.refresh(knowledge_base)
    return knowledge_base


@pytest.fixture
def fake_profile_lifecycle_side_effects(monkeypatch: pytest.MonkeyPatch):
    """Keep isolated SQLite profile tests off shared Redis while preserving dispatch."""

    from app.services import cache_manager, runtime_settings

    calls = {"cache": [], "publish": []}

    class FakeCacheManager:
        def invalidate_knowledge_base(
            self, knowledge_base_id: str, *, strict: bool = False
        ) -> bool:
            calls["cache"].append((knowledge_base_id, strict))
            return True

    def fake_publish(changed_keys, source="api", *, idempotency_key=None):
        calls["publish"].append(
            {
                "changed_keys": sorted(changed_keys),
                "source": source,
                "idempotency_key": idempotency_key,
            }
        )
        return {"version_hash": "e" * 64}

    monkeypatch.setattr(cache_manager, "get_cache_manager", lambda: FakeCacheManager())
    monkeypatch.setattr(
        runtime_settings, "publish_runtime_settings_version", fake_publish
    )
    return calls


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
        def __init__(self) -> None:
            self.embedding_model = "unit-test-embedding-model"
            self.embedding_dimensions = 8

        def for_embedding_identity(
            self,
            *,
            embedding_model: str,
            embedding_dimensions: int,
        ):
            self.embedding_model = str(embedding_model)
            self.embedding_dimensions = int(embedding_dimensions)
            return self

        async def embed_texts(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
            return [
                _unit_vector(
                    f"{self.embedding_model}:{text_type}:{text}",
                    self.embedding_dimensions,
                )
                for text in texts
            ]

    class FakeVectorStore:
        points: dict[str, dict] = {}
        collection_schemas: dict[str, tuple[int, str]] = {}
        expected_snapshot_calls: list[dict[str, object]] = []
        unbounded_list_ids_calls: int = 0

        def __init__(
            self,
            knowledge_base_name: str | None = None,
            collection_name: str | None = None,
            *,
            create_if_missing: bool = False,
        ) -> None:
            self.collection_name = collection_name or "unit"
            if create_if_missing:
                self.collection_schemas.setdefault(self.collection_name, (8, "cosine"))

        async def async_upsert(self, points: list[dict]) -> None:
            vector_sizes = {len(point.get("vector") or []) for point in points}
            if len(vector_sizes) != 1 or not vector_sizes or 0 in vector_sizes:
                raise ValueError("FakeVectorStore requires one non-empty vector dimension")
            vector_size = next(iter(vector_sizes))
            existing_schema = self.collection_schemas.get(self.collection_name)
            if existing_schema is not None and existing_schema != (vector_size, "cosine"):
                raise RuntimeError("FakeVectorStore collection schema mismatch")
            self.collection_schemas[self.collection_name] = (vector_size, "cosine")
            for point in points:
                self.points[point["id"]] = {**point, "collection": self.collection_name}

        def get_points(self, ids: list[str]) -> list[dict]:
            return [
                {
                    "id": str(self.points[point_id]["id"]),
                    "vector": list(self.points[point_id]["vector"]),
                    "payload": dict(self.points[point_id].get("payload") or {}),
                }
                for point_id in ids
                if point_id in self.points
                and self.points[point_id].get("collection") == self.collection_name
            ]

        def expected_points_snapshot(
            self,
            expected_point_ids: list[str],
            *,
            expected_vector_size: int,
            batch_size: int = 256,
        ) -> dict[str, object]:
            bounded_batch_size = max(1, min(int(batch_size), 256))
            expected_ids = list(dict.fromkeys(str(item) for item in expected_point_ids))
            self.expected_snapshot_calls.append(
                {
                    "collection_name": self.collection_name,
                    "expected_point_ids": list(expected_ids),
                    "expected_vector_size": expected_vector_size,
                    "batch_size": bounded_batch_size,
                }
            )
            schema = self.collection_schemas.get(self.collection_name)
            schema_error = None
            if schema is not None and schema != (expected_vector_size, "cosine"):
                schema_error = (
                    f"FakeVectorStore collection {self.collection_name!r} schema is {schema}, "
                    f"expected {(expected_vector_size, 'cosine')}"
                )
            points = [
                {
                    "id": str(self.points[point_id]["id"]),
                    "vector": list(self.points[point_id]["vector"]),
                    "payload": dict(self.points[point_id].get("payload") or {}),
                }
                for point_id in expected_ids
                if point_id in self.points
                and self.points[point_id].get("collection") == self.collection_name
            ]
            return {
                "collection_exists": schema is not None,
                "collection_schema_error": schema_error,
                "points": points if schema_error is None else [],
                "expected_point_count": len(expected_ids),
                "retrieve_batch_size": bounded_batch_size,
                "retrieve_batch_count": (
                    (len(expected_ids) + bounded_batch_size - 1) // bounded_batch_size
                    if expected_ids and schema_error is None
                    else 0
                ),
                "orphan_scan_performed": False,
            }

        def delete(self, ids: list[str]) -> None:
            for point_id in ids:
                self.points.pop(point_id, None)

        def list_ids(self, knowledge_base_id: str | None = None) -> list[str]:
            type(self).unbounded_list_ids_calls += 1
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
                    "facet_groups": [
                        {"facet": "Bayesian network", "role": "domain", "aliases": ["Bayesian networks"]},
                        {
                            "facet": "factorization",
                            "role": "procedure",
                            "aliases": ["conditional probability factorization"],
                        },
                    ],
                    "answer_shape": "grounded_answer",
                    "drop_terms": [],
                }
            return fallback or {"label": "Unit concept", "definition": "Unit definition"}

        async def answer_question_with_meta(self, question: str, contexts: list[dict], history: list[dict] | None = None, context_quality: str = "normal", **_kwargs):
            from app.services.embeddings import (
                ChatProvider as TrustedPromptProvider,
            )

            first = contexts[0]["content"] if contexts else "no context"
            metadata = dict(
                TrustedPromptProvider()._answer_prompt_bundle(
                    question,
                    context_quality=context_quality,
                )["protocol_metadata"]
            )
            return ChatCallResult(
                answer=f"Grounded answer: {first[:120]}",
                provider="unit_chat",
                model="unit-chat",
                external_called=False,
                prompt_protocol_version=metadata["protocol_version"],
                prompt_protocol_hash=metadata["prompt_protocol_hash"],
                grounding_envelope_protocol_version=metadata[
                    "protocol_version"
                ],
                grounding_envelope_hash=metadata["envelope_hash"],
                profile_hash=metadata["profile_hash"],
            )

    monkeypatch.setattr(context_graph, "EmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(context_graph, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(context_graph, "ChatProvider", FakeChatProvider)
    monkeypatch.setattr(agent_graph, "ChatProvider", FakeChatProvider)
    return {"EmbeddingProvider": FakeEmbeddingProvider, "VectorStore": FakeVectorStore, "ChatProvider": FakeChatProvider}


@pytest.fixture
def local_agent_admission():
    from app.services.agent_admission import LocalAgentAdmissionAdapter, use_agent_admission_adapter

    adapter = LocalAgentAdmissionAdapter()
    with use_agent_admission_adapter(adapter):
        yield adapter


@pytest_asyncio.fixture
async def populated_context_graph(db_session, sample_knowledge_base, fake_model_stack, local_agent_admission, tmp_path):
    from app.core.config import get_settings
    from app.models import Document, DocumentVersion
    from app.services.context_graph import rebuild_context_graph, write_chunks_and_structure, write_contextual_indexes
    from app.services.parsers import ParsedSection
    from app.services.storage import snapshot_source_file

    source_root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    source_root.mkdir(parents=True, exist_ok=True)
    logical_source = source_root / "bayesian-networks.md"
    logical_source.write_text(
        "# Bayesian networks\n"
        "Bayesian networks represent variables as nodes and conditional dependence as directed edges.\n"
        "A Markov blanket contains parents, children, and co-parents.\n"
        "P(X_1, X_2, ..., X_n)=product_i P(X_i | Pa(X_i)).\n",
        encoding="utf-8",
    )
    frozen_snapshot = snapshot_source_file(
        logical_source,
        knowledge_base_source_root=sample_knowledge_base.source_root,
    )
    snapshot_path = frozen_snapshot.canonical_path
    source_checksum = frozen_snapshot.checksum

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Bayesian networks",
        source_path=str(logical_source),
        source_type="markdown",
        tags=["Bayesian"],
        checksum=source_checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=source_checksum,
        storage_path=str(snapshot_path),
        is_active=True,
    )
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
