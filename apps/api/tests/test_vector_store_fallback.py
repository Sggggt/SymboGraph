from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


class _AvailableQdrantClient:
    def __init__(self, *args, **kwargs) -> None:
        self.collections: set[str] = set()

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        self.collections.add(collection_name)


class _UnavailableQdrantClient:
    def __init__(self, *args, **kwargs) -> None:
        raise OSError("qdrant unavailable")


def _transport_unavailable() -> ResponseHandlingException:
    request = httpx.Request("GET", "http://qdrant.invalid/collections")
    return ResponseHandlingException(
        httpx.ConnectError("qdrant transport unavailable", request=request)
    )


class _FirstProbeUnavailableQdrantClient:
    def __init__(self) -> None:
        self.probe_calls = 0

    def collection_exists(self, *, collection_name: str) -> bool:
        self.probe_calls += 1
        raise _transport_unavailable()


class _FirstSearchUnavailableQdrantClient:
    def __init__(self) -> None:
        self.search_calls = 0

    def query_points(self, **kwargs):
        self.search_calls += 1
        raise _transport_unavailable()


def _enable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "true")
    get_settings.cache_clear()


def _fallback_parent(knowledge_base_name: str) -> Path:
    from app.core.config import get_settings

    return get_settings().knowledge_base_paths_for_name(knowledge_base_name)[
        "ingestion_root"
    ]


def test_qdrant_success_with_fallback_disabled_has_no_fallback_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    knowledge_base_name = "Vector primary only"
    fallback_parent = _fallback_parent(knowledge_base_name)
    assert not fallback_parent.exists()
    monkeypatch.setattr(vector_store, "QdrantClient", _AvailableQdrantClient)

    store = vector_store.VectorStore(knowledge_base_name)

    assert store.client is not None
    assert store._fallback is None
    assert not fallback_parent.exists()


def test_qdrant_failure_with_fallback_disabled_has_no_fallback_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    knowledge_base_name = "Vector unavailable without fallback"
    fallback_parent = _fallback_parent(knowledge_base_name)
    assert not fallback_parent.exists()
    monkeypatch.setattr(vector_store, "QdrantClient", _UnavailableQdrantClient)

    with pytest.raises(OSError, match="qdrant unavailable"):
        vector_store.VectorStore(knowledge_base_name)

    assert not fallback_parent.exists()


def test_enabled_fallback_bootstraps_controlled_directory_only_on_first_write(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    _enable_fallback(monkeypatch)
    knowledge_base_name = "Explicit vector fallback"
    fallback_parent = _fallback_parent(knowledge_base_name)
    fallback_file = fallback_parent / "vector_index.json"
    bootstrap_calls: list[Path] = []
    controlled_bootstrap = vector_store.durable_ensure_directory

    def record_controlled_bootstrap(path: Path) -> Path:
        bootstrap_calls.append(path)
        return controlled_bootstrap(path)

    monkeypatch.setattr(
        vector_store,
        "durable_ensure_directory",
        record_controlled_bootstrap,
    )
    client = _FirstProbeUnavailableQdrantClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )

    store = vector_store.VectorStore(knowledge_base_name)

    assert store.client is client
    assert store._fallback is None
    assert not fallback_parent.exists()

    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0],
                "payload": {
                    "knowledge_base_id": "kb-1",
                    "chunk_id": "chunk-1",
                },
            }
        ]
    )

    assert client.probe_calls == 1
    assert store.client is None
    assert bootstrap_calls == [fallback_parent]
    assert fallback_parent.is_dir()
    assert fallback_file.is_file()
    assert store.list_ids("kb-1") == ["chunk-1"]
    assert bootstrap_calls == [fallback_parent]


def test_first_rpc_read_switches_once_without_bootstrapping_fallback_directory(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    _enable_fallback(monkeypatch)
    knowledge_base_name = "First RPC read fallback"
    fallback_parent = _fallback_parent(knowledge_base_name)
    client = _FirstProbeUnavailableQdrantClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore(knowledge_base_name)

    assert store.get_points(["chunk-1"]) == []

    assert client.probe_calls == 1
    assert store.client is None
    assert not fallback_parent.exists()


def test_first_rpc_delete_switches_once_and_bootstraps_on_fallback_mutation(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    _enable_fallback(monkeypatch)
    knowledge_base_name = "First RPC delete fallback"
    fallback_parent = _fallback_parent(knowledge_base_name)
    client = _FirstProbeUnavailableQdrantClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore(knowledge_base_name)

    store.delete(["chunk-1"])

    assert client.probe_calls == 1
    assert store.client is None
    assert fallback_parent.is_dir()
    assert (fallback_parent / "vector_index.json").is_file()


def test_first_rpc_search_switches_once_without_bootstrapping_fallback_directory(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    _enable_fallback(monkeypatch)
    knowledge_base_name = "First RPC search fallback"
    fallback_parent = _fallback_parent(knowledge_base_name)
    client = _FirstSearchUnavailableQdrantClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore(knowledge_base_name)

    assert store.search([1.0, 0.0], 3) == []

    assert client.search_calls == 1
    assert store.client is None
    assert not fallback_parent.exists()


def test_first_rpc_transport_failure_with_fallback_disabled_is_raised_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    knowledge_base_name = "First RPC no fallback"
    fallback_parent = _fallback_parent(knowledge_base_name)
    client = _FirstProbeUnavailableQdrantClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore(knowledge_base_name)

    with pytest.raises(ResponseHandlingException):
        store.get_points(["chunk-1"])

    assert client.probe_calls == 1
    assert store.client is client
    assert not fallback_parent.exists()


def test_transport_failure_after_qdrant_mutation_attempt_never_switches_backend(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    class AmbiguousUpsertClient:
        def collection_exists(self, *, collection_name: str) -> bool:
            return True

        def get_collection(self, *, collection_name: str):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=2, distance="Cosine")
                    )
                )
            )

        def upsert(self, **kwargs) -> None:
            raise _transport_unavailable()

    _enable_fallback(monkeypatch)
    knowledge_base_name = "Ambiguous upsert must fail closed"
    fallback_parent = _fallback_parent(knowledge_base_name)
    client = AmbiguousUpsertClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore(knowledge_base_name)

    with pytest.raises(ResponseHandlingException):
        store.upsert(
            [
                {
                    "id": "chunk-1",
                    "vector": [1.0, 0.0],
                    "payload": {
                        "knowledge_base_id": "kb-1",
                        "chunk_id": "chunk-1",
                    },
                }
            ]
        )

    assert store.client is client
    assert not fallback_parent.exists()


def test_partial_batched_qdrant_read_is_discarded_before_fallback_replay(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    class PartialReadClient:
        def __init__(self) -> None:
            self.retrieve_calls = 0

        def collection_exists(self, *, collection_name: str) -> bool:
            return True

        def retrieve(self, *, ids: list[str], **kwargs):
            self.retrieve_calls += 1
            if self.retrieve_calls == 2:
                raise _transport_unavailable()
            return [
                SimpleNamespace(
                    id=ids[0],
                    vector=[0.0, 1.0],
                    payload={"source": "qdrant"},
                )
            ]

    _enable_fallback(monkeypatch)
    client = PartialReadClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore("Batched read backend boundary")
    store._fallback_store().upsert(
        [
            {
                "id": "expected-0",
                "vector": [1.0, 0.0],
                "payload": {"source": "fallback"},
            }
        ]
    )

    points = store.get_points_batched(
        [f"expected-{index}" for index in range(257)]
    )

    assert client.retrieve_calls == 2
    assert points == [
        {
            "id": "expected-0",
            "vector": [1.0, 0.0],
            "payload": {"source": "fallback"},
        }
    ]
    assert store.client is None


def test_explicit_503_unavailable_can_switch_but_4xx_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    class StatusClient:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.probe_calls = 0

        def collection_exists(self, *, collection_name: str) -> bool:
            self.probe_calls += 1
            raise UnexpectedResponse(
                self.status_code,
                "unavailable" if self.status_code == 503 else "bad request",
                b"{}",
                httpx.Headers(),
            )

    _enable_fallback(monkeypatch)
    unavailable_client = StatusClient(503)
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: unavailable_client,
    )
    unavailable_store = vector_store.VectorStore("Explicit 503 fallback")
    assert unavailable_store.get_points(["chunk-1"]) == []
    assert unavailable_store.client is None

    validation_client = StatusClient(400)
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: validation_client,
    )
    validation_name = "Explicit 400 fail closed"
    validation_parent = _fallback_parent(validation_name)
    validation_store = vector_store.VectorStore(validation_name)
    with pytest.raises(UnexpectedResponse) as raised:
        validation_store.get_points(["chunk-1"])

    assert raised.value.status_code == 400
    assert validation_store.client is validation_client
    assert not validation_parent.exists()


@pytest.mark.parametrize(
    "invalid_point",
    [
        {
            "id": "chunk-1",
            "vector": [0.0, 0.0],
            "payload": {"knowledge_base_id": "kb-1", "chunk_id": "chunk-1"},
        },
        {
            "id": {},
            "vector": [1.0, 0.0],
            "payload": {"knowledge_base_id": "kb-1", "chunk_id": "chunk-1"},
        },
    ],
)
def test_local_validation_error_makes_zero_rpc_and_zero_fallback_mutation(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
    invalid_point: dict,
) -> None:
    from app.services import vector_store

    _enable_fallback(monkeypatch)
    knowledge_base_name = "Validation must fail closed"
    fallback_parent = _fallback_parent(knowledge_base_name)
    client = _FirstProbeUnavailableQdrantClient()
    monkeypatch.setattr(
        vector_store,
        "QdrantClient",
        lambda *args, **kwargs: client,
    )
    store = vector_store.VectorStore(knowledge_base_name)

    with pytest.raises((ValueError, ValidationError)):
        store.upsert([invalid_point])

    assert client.probe_calls == 0
    assert store.client is client
    assert not fallback_parent.exists()


def test_raw_constructor_oserror_does_not_qualify_for_enabled_fallback(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import vector_store

    _enable_fallback(monkeypatch)
    knowledge_base_name = "Raw constructor error"
    fallback_parent = _fallback_parent(knowledge_base_name)
    monkeypatch.setattr(vector_store, "QdrantClient", _UnavailableQdrantClient)

    with pytest.raises(OSError, match="qdrant unavailable"):
        vector_store.VectorStore(knowledge_base_name)

    assert not fallback_parent.exists()
