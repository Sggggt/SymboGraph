from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _SchemaAwareQdrantClient:
    def __init__(self, collections: dict[str, tuple[int, object]] | None = None) -> None:
        self.collections = dict(collections or {})
        self.points: dict[str, dict[str, dict]] = {}
        self.create_calls: list[tuple[str, object]] = []
        self.upsert_calls: list[tuple[str, list[object]]] = []
        self.retrieve_calls: list[tuple[str, list[str]]] = []
        self.scroll_calls = 0

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def get_collection(self, *, collection_name: str):
        size, distance = self.collections[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=size, distance=distance)
                )
            )
        )

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        self.create_calls.append((collection_name, vectors_config))
        self.collections[collection_name] = (
            vectors_config.size,
            vectors_config.distance,
        )

    def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
        assert wait is True
        self.upsert_calls.append((collection_name, points))

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool,
        with_vectors: bool,
    ) -> list[object]:
        assert with_payload is True
        assert with_vectors is True
        self.retrieve_calls.append((collection_name, list(ids)))
        collection_points = self.points.get(collection_name, {})
        return [
            SimpleNamespace(
                id=point_id,
                vector=collection_points[point_id]["vector"],
                payload=collection_points[point_id]["payload"],
            )
            for point_id in ids
            if point_id in collection_points
        ]

    def scroll(self, **kwargs):
        self.scroll_calls += 1
        raise AssertionError("active expected-point snapshot must not scroll")


def _point(
    point_id: str,
    vector: list[float],
    *,
    embedding_dimension: int | None = None,
    vector_distance_metric: str | None = None,
) -> dict:
    payload: dict[str, object] = {
        "knowledge_base_id": "kb-1",
        "chunk_id": point_id,
    }
    if embedding_dimension is not None:
        payload["embedding_dimension"] = embedding_dimension
    if vector_distance_metric is not None:
        payload["vector_distance_metric"] = vector_distance_metric
    return {"id": point_id, "vector": vector, "payload": payload}


def test_constructor_and_missing_collection_reads_or_deletes_do_not_create(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    client = _SchemaAwareQdrantClient()
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)

    store = vector_store.VectorStore(
        "read delete without creation",
        collection_name="missing-collection",
    )
    assert store.get_points(["missing-point"]) == []
    store.delete(["missing-point"])

    assert client.create_calls == []
    assert client.upsert_calls == []


@pytest.mark.parametrize(
    ("existing_size", "existing_distance", "message"),
    [
        (3, "Dot", "distance"),
        (4, "Cosine", "vector size"),
    ],
)
def test_existing_collection_schema_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
    existing_size: int,
    existing_distance: str,
    message: str,
) -> None:
    from app.services import vector_store

    client = _SchemaAwareQdrantClient(
        {"existing": (existing_size, existing_distance)}
    )
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    store = vector_store.VectorStore(
        "schema mismatch",
        collection_name="existing",
        create_if_missing=False,
    )

    with pytest.raises(vector_store.VectorCollectionSchemaError, match=message):
        store.upsert([_point("chunk-1", [1.0, 0.0, 0.0])])

    assert client.create_calls == []
    assert client.upsert_calls == []


def test_enabled_fallback_does_not_hide_existing_collection_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.core.config import get_settings
    from app.services import vector_store

    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "true")
    get_settings.cache_clear()
    knowledge_base_name = "Fallback schema mismatch"
    fallback_parent = get_settings().knowledge_base_paths_for_name(
        knowledge_base_name
    )["ingestion_root"]
    client = _SchemaAwareQdrantClient({"existing": (4, "Cosine")})
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    store = vector_store.VectorStore(
        knowledge_base_name,
        collection_name="existing",
        create_if_missing=False,
    )

    with pytest.raises(vector_store.VectorCollectionSchemaError, match="vector size"):
        store.upsert([_point("chunk-1", [1.0, 0.0, 0.0])])

    assert store.client is client
    assert client.create_calls == []
    assert client.upsert_calls == []
    assert not fallback_parent.exists()


def test_outbox_style_replay_creates_missing_collection_from_historical_vector_dimension(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.core.config import get_settings
    from app.services import vector_store

    assert get_settings().embedding_dimensions == 8
    client = _SchemaAwareQdrantClient()
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    store = vector_store.VectorStore(
        "historical replay",
        collection_name="historical-three-dimensional",
        create_if_missing=False,
    )
    assert client.create_calls == []

    # Older durable target payloads may predate the explicit schema fields. The
    # complete target vector plus fixed cosine protocol is sufficient to replay.
    store.upsert([_point("chunk-old", [1.0, 0.0, 0.0])])

    assert len(client.create_calls) == 1
    collection_name, vectors_config = client.create_calls[0]
    assert collection_name == "historical-three-dimensional"
    assert vectors_config.size == 3
    assert vectors_config.distance == vector_store.rest.Distance.COSINE
    assert len(client.upsert_calls) == 1


def test_mixed_vector_dimensions_are_rejected_before_collection_creation(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    client = _SchemaAwareQdrantClient()
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    store = vector_store.VectorStore(
        "mixed dimensions",
        collection_name="must-not-be-created",
        create_if_missing=False,
    )

    with pytest.raises(ValueError, match="same dimension"):
        store.upsert(
            [
                _point("chunk-1", [1.0, 0.0]),
                _point("chunk-2", [1.0, 0.0, 0.0]),
            ]
        )

    assert client.create_calls == []
    assert client.upsert_calls == []


@pytest.mark.parametrize(
    ("point", "message"),
    [
        (_point("empty", []), "non-empty vector"),
        (
            _point("wrong-dimension", [1.0, 0.0], embedding_dimension=3),
            "embedding_dimension conflicts",
        ),
        (
            _point("wrong-metric", [1.0, 0.0], vector_distance_metric="dot"),
            "must be cosine",
        ),
        (_point("nan", [float("nan"), 0.0]), "finite real number"),
        (_point("infinity", [float("inf"), 0.0]), "finite real number"),
        (_point("bool", [True, 0.0]), "finite real number"),
        (_point("string", ["1.0", 0.0]), "finite real number"),
        (_point("zero-vector", [0.0, 0.0]), "norm greater than"),
    ],
)
def test_invalid_vector_or_payload_schema_is_rejected_before_collection_creation(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
    point: dict,
    message: str,
) -> None:
    from app.services import vector_store

    client = _SchemaAwareQdrantClient()
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    store = vector_store.VectorStore(
        "invalid payload",
        collection_name="must-not-be-created",
        create_if_missing=False,
    )

    with pytest.raises(ValueError, match=message):
        store.upsert([point])

    assert client.create_calls == []
    assert client.upsert_calls == []


def test_active_expected_point_snapshot_batches_only_postgres_scope_without_scroll(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    expected_ids = [f"point-{index:04d}" for index in range(600)]
    client = _SchemaAwareQdrantClient({"active": (3, "Cosine")})
    client.points["active"] = {
        point_id: {
            "vector": [1.0, 0.0, 0.0],
            "payload": {"knowledge_base_id": "kb-1", "chunk_id": point_id},
        }
        for point_id in expected_ids
    }
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    store = vector_store.VectorStore(
        "active freshness",
        collection_name="active",
        create_if_missing=False,
    )

    snapshot = store.expected_points_snapshot(
        expected_ids,
        expected_vector_size=3,
        batch_size=10_000,
    )

    assert snapshot["collection_exists"] is True
    assert snapshot["collection_schema_error"] is None
    assert snapshot["expected_point_count"] == 600
    assert snapshot["retrieve_batch_size"] == 256
    assert snapshot["retrieve_batch_count"] == 3
    assert len(snapshot["points"]) == 600
    assert snapshot["orphan_scan_performed"] is False
    assert [len(ids) for _collection, ids in client.retrieve_calls] == [256, 256, 88]
    assert client.scroll_calls == 0
