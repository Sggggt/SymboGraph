from __future__ import annotations

import asyncio
import json
import math
import os
import re
import struct
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from threading import Lock
from typing import Any, Callable, TypeVar

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http import models as rest

from app.core.config import get_settings
from app.services.storage import durable_ensure_directory, durable_replace

try:
    import portalocker
except ImportError:  # pragma: no cover - optional dependency in fallback-only paths
    portalocker = None


_VECTOR_FILE_LOCKS: OrderedDict[Path, Lock] = OrderedDict()
_VECTOR_FILE_LOCKS_GUARD = Lock()
_MAX_VECTOR_FILE_LOCKS = 128
QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE = 256
QDRANT_RECONCILE_SCROLL_PAGE_SIZE = 256
QDRANT_RECONCILE_MAX_SCANNED_POINTS = 4096
QDRANT_DESTRUCTIVE_MAX_COLLECTIONS = 512
QDRANT_DESTRUCTIVE_MAX_OWNED_POINTS = 1_000_000
MIN_EMBEDDING_VECTOR_NORM = 1e-12
QDRANT_DELETE_FENCE_REQUIRED_FIELDS = (
    "knowledge_base_id",
    "chunk_id",
)
QDRANT_DELETE_FENCE_OPTIONAL_FIELDS = (
    "qdrant_write_intent_id",
    "qdrant_write_protocol_version",
    "vector_payload_hash",
)
QDRANT_UNAVAILABLE_STATUS_CODES = frozenset({502, 503, 504})
_OperationResult = TypeVar("_OperationResult")


def _exception_chain(error: BaseException):
    """Yield a bounded, cycle-safe exception/source chain."""

    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending and len(visited) < 32:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        yield current
        for candidate in (
            getattr(current, "source", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(candidate, BaseException) and id(candidate) not in visited:
                pending.append(candidate)


def _is_explicit_qdrant_unavailable(error: BaseException) -> bool:
    """Accept only Qdrant transport failures or explicit gateway unavailability.

    Qdrant wraps HTTP transport failures in ``ResponseHandlingException``.
    Local validation/schema errors and all other HTTP statuses intentionally do
    not qualify, so an enabled fallback cannot hide a malformed request or an
    incompatible active collection.
    """

    chain = tuple(_exception_chain(error))
    if any(
        isinstance(item, UnexpectedResponse)
        and item.status_code in QDRANT_UNAVAILABLE_STATUS_CODES
        for item in chain
    ):
        return True
    if not isinstance(error, ResponseHandlingException):
        return False
    return any(
        item is not error
        and isinstance(item, (httpx.TransportError, OSError, TimeoutError))
        for item in chain
    )


def _delete_fence(point: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    point_id = str(point.get("id") or "")
    payload = point.get("payload")
    if not point_id or not isinstance(payload, dict):
        raise ValueError("Conditional Qdrant delete requires a point id and payload")
    fence: dict[str, Any] = {}
    for field in QDRANT_DELETE_FENCE_REQUIRED_FIELDS:
        value = payload.get(field)
        if value in {None, ""}:
            raise ValueError(f"Conditional Qdrant delete point {point_id} has no {field} fence")
        fence[field] = value
    for field in QDRANT_DELETE_FENCE_OPTIONAL_FIELDS:
        value = payload.get(field)
        if value not in {None, ""}:
            fence[field] = value
    if str(fence["chunk_id"]) != point_id:
        raise ValueError(f"Conditional Qdrant delete point {point_id} has a mismatched chunk fence")
    return point_id, fence


def _point_matches_delete_fence(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    current_id = str(current.get("id") or "")
    expected_id, fence = _delete_fence(expected)
    if current_id != expected_id:
        return False
    payload = current.get("payload")
    return isinstance(payload, dict) and all(payload.get(key) == value for key, value in fence.items())


def vector_file_lock(path: Path) -> Lock:
    resolved = path.resolve()
    with _VECTOR_FILE_LOCKS_GUARD:
        if resolved not in _VECTOR_FILE_LOCKS:
            _VECTOR_FILE_LOCKS[resolved] = Lock()
        else:
            _VECTOR_FILE_LOCKS.move_to_end(resolved)
        while len(_VECTOR_FILE_LOCKS) > _MAX_VECTOR_FILE_LOCKS:
            _VECTOR_FILE_LOCKS.popitem(last=False)
        return _VECTOR_FILE_LOCKS[resolved]


@contextmanager
def vector_process_lock(path: Path):
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    if not lock_path.parent.is_dir() or lock_path.parent.is_symlink():
        raise RuntimeError(
            "Fallback vector storage must be bootstrapped through the storage durability gate"
        )
    if portalocker is not None:
        with lock_path.open("a+b") as handle:
            portalocker.lock(handle, portalocker.LOCK_EX)
            try:
                yield
            finally:
                portalocker.unlock(handle)
        return
    with vector_file_lock(path):
        yield


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_mag = math.sqrt(sum(a * a for a in left)) or 1.0
    right_mag = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_mag * right_mag)


class FallbackVectorStore:
    def __init__(self, backing_file: Path) -> None:
        self.backing_file = backing_file
        self._bootstrap_lock = Lock()
        self._bootstrapped = False

    def _ensure_ready(self) -> None:
        parent = self.backing_file.parent
        if self._bootstrapped and parent.is_dir() and not parent.is_symlink():
            return
        with self._bootstrap_lock:
            if self._bootstrapped and parent.is_dir() and not parent.is_symlink():
                return
            durable_ensure_directory(parent)
            self._bootstrapped = True

    @contextmanager
    def _locked(self):
        self._ensure_ready()
        with vector_process_lock(self.backing_file):
            yield

    def _read(self) -> list[dict]:
        if not self.backing_file.exists():
            return []
        try:
            text = self.backing_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                data, _ = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                return []
        return data if isinstance(data, list) else []

    def _read_snapshot(self) -> list[dict]:
        """Read without bootstrapping fallback storage.

        Writers publish with an atomic replace, so a process-local read lock is
        sufficient here and, unlike the process lock, does not create a lock
        file. Merely switching to or reading from fallback therefore has no
        filesystem side effect.
        """

        if not self.backing_file.is_file():
            return []
        with vector_file_lock(self.backing_file):
            return self._read()

    def collection_exists(self) -> bool:
        return self.backing_file.is_file()

    def _write(self, data: list[dict]) -> None:
        temporary_file = self.backing_file.with_suffix(f"{self.backing_file.suffix}.tmp")
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        durable_replace(temporary_file, self.backing_file)

    def upsert(self, points: list[dict]) -> None:
        with self._locked():
            current = self._read()
            indexed = {item["id"]: item for item in current}
            for point in points:
                indexed[point["id"]] = point
            self._write(list(indexed.values()))

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        id_set = set(ids)
        with self._locked():
            current = self._read()
            self._write([item for item in current if item.get("id") not in id_set])

    def delete_if_payload_matches(self, expected_points: list[dict[str, Any]]) -> None:
        if not expected_points:
            return
        expected_by_id = {
            str(point.get("id") or ""): dict(point)
            for point in expected_points
        }
        if not all(expected_by_id) or len(expected_by_id) != len(expected_points):
            raise ValueError("Conditional vector delete requires unique non-empty point ids")
        with self._locked():
            current = self._read()
            self._write(
                [
                    point
                    for point in current
                    if not (
                        str(point.get("id") or "") in expected_by_id
                        and _point_matches_delete_fence(
                            point,
                            expected_by_id[str(point.get("id") or "")],
                        )
                    )
                ]
            )

    def get_points(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        id_set = set(ids)
        current = self._read_snapshot()
        return [item for item in current if item.get("id") in id_set]

    def list_ids(self, knowledge_base_id: str | None = None) -> list[str]:
        current = self._read_snapshot()
        ids = []
        for item in current:
            payload = item.get("payload", {})
            if knowledge_base_id and payload.get("knowledge_base_id") != knowledge_base_id:
                continue
            ids.append(item["id"])
        return ids

    def search(self, vector: list[float], limit: int, filters: dict[str, Any]) -> list[dict]:
        points = self._read_snapshot()
        results = []
        for point in points:
            payload = point.get("payload", {})
            if any(filters.get(key) and payload.get(key) != filters[key] for key in filters):
                continue
            score = cosine_similarity(vector, point["vector"])
            results.append({"id": point["id"], "score": score, "payload": payload})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


QDRANT_SINGLE_VECTOR_DISTANCE_METRIC = "cosine"


class VectorCollectionSchemaError(RuntimeError):
    """Raised when a Qdrant collection cannot satisfy the active vector contract."""


def canonical_embedding_vector(vector: Any, *, source: str) -> list[float]:
    """Return finite float values without accepting bool or string coercions."""

    if not isinstance(vector, list) or not vector:
        raise ValueError(f"{source} must be a non-empty vector list")
    normalized: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{source} value at index {index} must be a finite real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{source} value at index {index} must be a finite real number")
        try:
            packed = struct.pack(">f", numeric)
        except OverflowError as exc:
            raise ValueError(
                f"{source} value at index {index} must fit finite IEEE-754 binary32"
            ) from exc
        binary32 = struct.unpack(">f", packed)[0]
        if not math.isfinite(binary32):
            raise ValueError(
                f"{source} value at index {index} must fit finite IEEE-754 binary32"
            )
        normalized.append(0.0 if binary32 == 0.0 else binary32)
    if math.sqrt(math.fsum(value * value for value in normalized)) <= MIN_EMBEDDING_VECTOR_NORM:
        raise ValueError(f"{source} must have norm greater than {MIN_EMBEDDING_VECTOR_NORM}")
    return normalized


def _positive_vector_size(value: Any, *, source: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{source} must be a positive integer")
    return value


def _upsert_vector_size(points: list[dict]) -> int:
    expected_size: int | None = None
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"Vector point at index {index} must be an object")
        vector = canonical_embedding_vector(
            point.get("vector"),
            source=f"Vector point at index {index}",
        )
        point["vector"] = vector
        vector_size = len(vector)
        if expected_size is None:
            expected_size = vector_size
        elif vector_size != expected_size:
            raise ValueError(
                "All vectors in one Qdrant upsert must have the same dimension"
            )

        payload = point.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Vector point at index {index} must contain a payload object"
            )
        declared_dimension = payload.get("embedding_dimension")
        if declared_dimension is not None and (
            type(declared_dimension) is not int or declared_dimension != vector_size
        ):
            raise ValueError(
                f"Vector point at index {index} embedding_dimension conflicts with its vector"
            )
        declared_metric = payload.get("vector_distance_metric")
        if declared_metric is not None and (
            not isinstance(declared_metric, str)
            or declared_metric.strip().lower() != QDRANT_SINGLE_VECTOR_DISTANCE_METRIC
        ):
            raise ValueError(
                f"Vector point at index {index} vector_distance_metric must be cosine"
            )

    if expected_size is None:  # pragma: no cover - guarded by VectorStore.upsert
        raise ValueError("Qdrant upsert requires at least one vector point")
    return expected_size


def _single_vector_schema(collection_info: Any) -> tuple[int, str]:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if vectors is None and isinstance(collection_info, dict):
        vectors = (
            (collection_info.get("config") or {}).get("params") or {}
        ).get("vectors")

    if isinstance(vectors, dict):
        if "size" not in vectors or "distance" not in vectors:
            raise VectorCollectionSchemaError(
                "Qdrant collection uses named or otherwise non-single vector configuration"
            )
        size = vectors.get("size")
        distance = vectors.get("distance")
    else:
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)

    if type(size) is not int or size <= 0:
        raise VectorCollectionSchemaError(
            "Qdrant collection has no valid single-vector dimension"
        )
    distance_value = getattr(distance, "value", distance)
    normalized_distance = str(distance_value or "").strip().lower()
    if not normalized_distance:
        raise VectorCollectionSchemaError(
            "Qdrant collection has no valid single-vector distance metric"
        )
    return size, normalized_distance


class VectorStore:
    def __init__(
        self,
        knowledge_base_name: str | None = None,
        collection_name: str | None = None,
        *,
        knowledge_base_id: str | None = None,
        create_if_missing: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.collection = collection_name or self.settings.qdrant_collection
        knowledge_base_paths = (
            self.settings.knowledge_base_paths_for_id(knowledge_base_id)
            if knowledge_base_id is not None
            else self.settings.knowledge_base_paths_for_name(
                knowledge_base_name or self.settings.knowledge_base_name
            )
        )
        self._fallback_path = knowledge_base_paths["ingestion_root"] / "vector_index.json"
        self._fallback: FallbackVectorStore | None = None
        self._fallback_lock = Lock()
        self._backend_switch_lock = Lock()
        self.client: QdrantClient | None = None
        try:
            self.client = QdrantClient(url=self.settings.qdrant_url, timeout=5.0)
        except Exception as exc:
            if (
                not self.settings.enable_model_fallback
                or not _is_explicit_qdrant_unavailable(exc)
            ):
                raise
            self.client = None
        if create_if_missing:
            self.ensure_collection(vector_size=int(self.settings.embedding_dimensions))

    def _fallback_store(self) -> FallbackVectorStore:
        if not self.settings.enable_model_fallback:
            raise RuntimeError("Vector fallback is disabled")
        if self._fallback is not None:
            return self._fallback
        with self._fallback_lock:
            if self._fallback is None:
                self._fallback = FallbackVectorStore(self._fallback_path)
            return self._fallback

    def _switch_to_fallback(
        self,
        error: BaseException,
        *,
        expected_client: QdrantClient,
    ) -> bool:
        """Disable the primary backend only for an explicit unavailable signal."""

        if (
            not self.settings.enable_model_fallback
            or not _is_explicit_qdrant_unavailable(error)
        ):
            return False
        switch_lock = getattr(self, "_backend_switch_lock", None)
        if switch_lock is None:  # compatibility for narrowly constructed test doubles
            if self.client is expected_client:
                self.client = None
            return self.client is None
        with switch_lock:
            if self.client is expected_client:
                self.client = None
            return self.client is None

    def _run_read_operation(
        self,
        primary: Callable[[QdrantClient], _OperationResult],
        fallback: Callable[[FallbackVectorStore], _OperationResult],
    ) -> _OperationResult:
        """Run one read wholly against Qdrant or wholly against fallback."""

        client = self.client
        if client is None:
            return fallback(self._fallback_store())
        try:
            return primary(client)
        except Exception as exc:
            if not self._switch_to_fallback(exc, expected_client=client):
                raise
        # The primary result is discarded before fallback is read, preventing a
        # multi-page/batch operation from combining two backend snapshots.
        return fallback(self._fallback_store())

    def _client_collection_exists(self, client: QdrantClient) -> bool:
        direct_probe = getattr(client, "collection_exists", None)
        if callable(direct_probe):
            return bool(direct_probe(collection_name=self.collection))
        collections = {item.name for item in client.get_collections().collections}
        return self.collection in collections

    def collection_exists(self) -> bool:
        if self.client is None and not self.settings.enable_model_fallback:
            return False
        return self._run_read_operation(
            self._client_collection_exists,
            lambda fallback: fallback.collection_exists(),
        )

    def list_collection_names_bounded(
        self,
        *,
        max_collections: int = QDRANT_DESTRUCTIVE_MAX_COLLECTIONS,
    ) -> dict[str, Any]:
        """Enumerate the destructive scan surface without creating collections."""

        bounded_max = max(
            1,
            min(int(max_collections), QDRANT_DESTRUCTIVE_MAX_COLLECTIONS),
        )

        def primary(client: QdrantClient) -> dict[str, Any]:
            names = sorted(
                {
                    str(item.name)
                    for item in client.get_collections().collections
                    if str(getattr(item, "name", ""))
                }
            )
            return {
                "collection_names": names[:bounded_max],
                "collection_count": len(names),
                "truncated": len(names) > bounded_max,
                "max_collections": bounded_max,
                "complete_backend_inventory": len(names) <= bounded_max,
            }

        def fallback_inventory(fallback: FallbackVectorStore) -> dict[str, Any]:
            names = [self.collection] if fallback.collection_exists() else []
            return {
                "collection_names": names,
                "collection_count": len(names),
                "truncated": False,
                "max_collections": bounded_max,
                "complete_backend_inventory": False,
            }

        return self._run_read_operation(primary, fallback_inventory)

    def list_owned_ids_complete(
        self,
        knowledge_base_id: str,
        *,
        max_points: int = QDRANT_DESTRUCTIVE_MAX_OWNED_POINTS,
        page_size: int = QDRANT_RECONCILE_SCROLL_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Scroll the complete owner-filtered point scope or fail visibly at a hard cap."""

        bounded_max = max(
            1,
            min(int(max_points), QDRANT_DESTRUCTIVE_MAX_OWNED_POINTS),
        )
        bounded_page = max(
            1,
            min(int(page_size), QDRANT_RECONCILE_SCROLL_PAGE_SIZE),
        )

        def primary(client: QdrantClient) -> dict[str, Any]:
            if not self._client_collection_exists(client):
                return {
                    "ids": [],
                    "point_count": 0,
                    "page_count": 0,
                    "truncated": False,
                    "max_points": bounded_max,
                }
            qdrant_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="knowledge_base_id",
                        match=rest.MatchValue(value=knowledge_base_id),
                    )
                ]
            )
            ids: list[str] = []
            offset = None
            page_count = 0
            while True:
                remaining_with_sentinel = bounded_max + 1 - len(ids)
                if remaining_with_sentinel <= 0:
                    break
                points, offset = client.scroll(
                    collection_name=self.collection,
                    scroll_filter=qdrant_filter,
                    limit=min(bounded_page, remaining_with_sentinel),
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                page_count += 1
                ids.extend(str(point.id) for point in points)
                if offset is None:
                    break
            truncated = len(ids) > bounded_max or offset is not None
            return {
                "ids": ids[:bounded_max],
                "point_count": min(len(ids), bounded_max),
                "page_count": page_count,
                "truncated": truncated,
                "max_points": bounded_max,
            }

        def fallback_inventory(fallback: FallbackVectorStore) -> dict[str, Any]:
            ids = fallback.list_ids(knowledge_base_id)
            return {
                "ids": ids[:bounded_max],
                "point_count": min(len(ids), bounded_max),
                "page_count": 1 if ids else 0,
                "truncated": len(ids) > bounded_max,
                "max_points": bounded_max,
            }

        return self._run_read_operation(primary, fallback_inventory)

    def list_owned_ids_complete_exact(
        self,
        knowledge_base_id: str,
        *,
        max_points: int,
        page_size: int = QDRANT_RECONCILE_SCROLL_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Bounded complete KB-owned id inventory without vectors or fallback."""

        if type(knowledge_base_id) is not str or not knowledge_base_id:
            raise ValueError("Exact Qdrant owner inventory requires a knowledge-base id")
        if (
            type(max_points) is not int
            or max_points <= 0
            or max_points > QDRANT_DESTRUCTIVE_MAX_OWNED_POINTS
        ):
            raise ValueError("Exact Qdrant owner inventory max_points is out of bounds")
        if (
            type(page_size) is not int
            or page_size <= 0
            or page_size > QDRANT_RECONCILE_SCROLL_PAGE_SIZE
        ):
            raise ValueError("Exact Qdrant owner inventory page_size is out of bounds")
        bounded_max = max_points
        bounded_page = page_size
        client = self.client
        if client is None:
            raise RuntimeError("Exact Qdrant owner inventory requires Qdrant; fallback is forbidden")
        if not self._client_collection_exists(client):
            return {
                "ids": [],
                "point_count": 0,
                "page_count": 0,
                "truncated": False,
                "max_points": bounded_max,
            }
        qdrant_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="knowledge_base_id",
                    match=rest.MatchValue(value=knowledge_base_id),
                )
            ]
        )
        ids: list[str] = []
        offset = None
        page_count = 0
        while True:
            remaining_with_sentinel = bounded_max + 1 - len(ids)
            if remaining_with_sentinel <= 0:
                break
            points, offset = client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                limit=min(bounded_page, remaining_with_sentinel),
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            page_count += 1
            ids.extend(str(point.id) for point in points)
            if offset is None:
                break
        truncated = len(ids) > bounded_max or offset is not None
        return {
            "ids": sorted(ids[:bounded_max]),
            "point_count": min(len(ids), bounded_max),
            "page_count": page_count,
            "truncated": truncated,
            "max_points": bounded_max,
        }

    def _validate_collection_schema_on_client(
        self,
        client: QdrantClient,
        *,
        vector_size: int,
    ) -> None:
        collection_info = client.get_collection(collection_name=self.collection)
        actual_size, actual_distance = _single_vector_schema(collection_info)
        if actual_size != vector_size:
            raise VectorCollectionSchemaError(
                f"Qdrant collection {self.collection!r} vector size is {actual_size}, "
                f"expected {vector_size}"
            )
        if actual_distance != QDRANT_SINGLE_VECTOR_DISTANCE_METRIC:
            raise VectorCollectionSchemaError(
                f"Qdrant collection {self.collection!r} distance is {actual_distance!r}, "
                f"expected {QDRANT_SINGLE_VECTOR_DISTANCE_METRIC!r}"
            )

    def _validate_collection_schema(self, *, vector_size: int) -> None:
        client = self.client
        if client is None:
            return
        self._validate_collection_schema_on_client(
            client,
            vector_size=vector_size,
        )

    def _ensure_collection_on_client(
        self,
        client: QdrantClient,
        *,
        vector_size: int,
        mutation_started: list[bool],
    ) -> None:
        if self._client_collection_exists(client):
            self._validate_collection_schema_on_client(
                client,
                vector_size=vector_size,
            )
            return
        mutation_started[0] = True
        try:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )
        except Exception as exc:
            if not _is_explicit_qdrant_unavailable(exc):
                raise
            # An unavailable response may arrive after Qdrant committed the
            # create. Once that RPC was sent its outcome is ambiguous, so this
            # recovery may validate Qdrant but may never switch the operation to
            # filesystem fallback. Validation and every 4xx fail immediately.
            if not self._client_collection_exists(client):
                raise
            self._validate_collection_schema_on_client(
                client,
                vector_size=vector_size,
            )

    def ensure_collection(self, *, vector_size: int | None = None) -> None:
        expected_size = _positive_vector_size(
            int(self.settings.embedding_dimensions) if vector_size is None else vector_size,
            source="Qdrant vector size",
        )
        client = self.client
        if client is None:
            if not self.settings.enable_model_fallback:
                raise RuntimeError(
                    "Qdrant is unavailable and ENABLE_MODEL_FALLBACK is false"
                )
            return
        mutation_started = [False]
        try:
            self._ensure_collection_on_client(
                client,
                vector_size=expected_size,
                mutation_started=mutation_started,
            )
        except Exception as exc:
            if (
                mutation_started[0]
                or not self._switch_to_fallback(exc, expected_client=client)
            ):
                raise

    def upsert(self, points: list[dict]) -> None:
        if not points:
            return
        vector_size = _upsert_vector_size(points)
        qdrant_points = [
            rest.PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"],
            )
            for point in points
        ]
        client = self.client
        if client is None:
            self._fallback_store().upsert(points)
            return
        mutation_started = [False]
        try:
            self._ensure_collection_on_client(
                client,
                vector_size=vector_size,
                mutation_started=mutation_started,
            )
            mutation_started[0] = True
            client.upsert(
                collection_name=self.collection,
                points=qdrant_points,
                wait=True,
            )
        except Exception as exc:
            if (
                mutation_started[0]
                or not self._switch_to_fallback(exc, expected_client=client)
            ):
                raise
            self._fallback_store().upsert(points)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        points_selector = rest.PointIdsList(points=ids)
        client = self.client
        if client is None:
            self._fallback_store().delete(ids)
            return
        mutation_started = False
        try:
            if not self._client_collection_exists(client):
                return
            mutation_started = True
            client.delete(
                collection_name=self.collection,
                points_selector=points_selector,
                wait=True,
            )
        except Exception as exc:
            if mutation_started or not self._switch_to_fallback(
                exc,
                expected_client=client,
            ):
                raise
            self._fallback_store().delete(ids)

    def delete_collection_exact(self, expected_collection_name: str) -> dict[str, Any]:
        """Delete only this explicitly named Qdrant collection, never fallback.

        Collection cleanup is a separate destructive maintenance operation.
        The caller must first persist its PostgreSQL intent; this method merely
        executes and verifies the exact external target.
        """

        expected = str(expected_collection_name or "").strip()
        if (
            not expected
            or len(expected) > 255
            or re.fullmatch(r"[A-Za-z0-9_-]+", expected) is None
        ):
            raise ValueError("Exact Qdrant collection name is invalid")
        if expected != self.collection:
            raise ValueError("Exact Qdrant collection confirmation does not match the store target")
        client = self.client
        if client is None:
            raise RuntimeError(
                "Exact collection deletion requires Qdrant; filesystem fallback is forbidden"
            )
        existed_before = self._client_collection_exists(client)
        if not existed_before:
            return {
                "collection_name": expected,
                "existed_before": False,
                "deleted": False,
                "verified_absent": True,
                "outcome": "already_absent",
            }
        try:
            client.delete_collection(collection_name=expected)
        except Exception:
            # A transport failure can arrive after Qdrant applied the delete.
            # Only a successful exact absence probe may turn it into success.
            if self._client_collection_exists(client):
                raise
            return {
                "collection_name": expected,
                "existed_before": True,
                "deleted": True,
                "verified_absent": True,
                "outcome": "applied_response_uncertain_absence_verified",
            }
        if self._client_collection_exists(client):
            raise RuntimeError(
                f"Qdrant collection {expected!r} still exists after delete acknowledgement"
            )
        return {
            "collection_name": expected,
            "existed_before": True,
            "deleted": True,
            "verified_absent": True,
            "outcome": "applied_and_verified",
        }

    def delete_if_payload_matches(self, expected_points: list[dict[str, Any]]) -> None:
        if not expected_points:
            return
        fences = [_delete_fence(point) for point in expected_points]
        if len({point_id for point_id, _fence in fences}) != len(fences):
            raise ValueError("Conditional Qdrant delete requires unique point ids")
        selectors = [
            rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.HasIdCondition(has_id=[point_id]),
                        *[
                            rest.FieldCondition(
                                key=key,
                                match=rest.MatchValue(value=value),
                            )
                            for key, value in fence.items()
                        ],
                    ]
                )
            )
            for point_id, fence in fences
        ]
        client = self.client
        if client is None:
            self._fallback_store().delete_if_payload_matches(expected_points)
            return
        mutation_started = False
        try:
            if not self._client_collection_exists(client):
                return
            for selector in selectors:
                mutation_started = True
                client.delete(
                    collection_name=self.collection,
                    points_selector=selector,
                    wait=True,
                )
        except Exception as exc:
            if mutation_started or not self._switch_to_fallback(
                exc,
                expected_client=client,
            ):
                raise
            self._fallback_store().delete_if_payload_matches(expected_points)

    async def async_upsert(self, points: list[dict]) -> None:
        """Async wrapper for upsert to avoid blocking the event loop."""
        await asyncio.to_thread(self.upsert, points)

    async def async_delete(self, ids: list[str]) -> None:
        """Async wrapper for delete to avoid blocking the event loop."""
        await asyncio.to_thread(self.delete, ids)

    async def async_delete_if_payload_matches(
        self,
        expected_points: list[dict[str, Any]],
    ) -> None:
        """Async wrapper for fenced delete to avoid blocking the event loop."""
        await asyncio.to_thread(self.delete_if_payload_matches, expected_points)

    def get_points(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        def primary(client: QdrantClient) -> list[dict]:
            if not self._client_collection_exists(client):
                return []
            points = client.retrieve(
                collection_name=self.collection,
                ids=ids,
                with_payload=True,
                with_vectors=True,
            )
            return [
                {"id": str(point.id), "vector": point.vector, "payload": point.payload or {}}
                for point in points
            ]

        return self._run_read_operation(
            primary,
            lambda fallback: fallback.get_points(ids),
        )

    def _get_points_batched_from_client(
        self,
        client: QdrantClient,
        ids: list[str],
        *,
        batch_size: int,
        known_collection_exists: bool | None,
    ) -> list[dict[str, Any]]:
        exists = (
            self._client_collection_exists(client)
            if known_collection_exists is None
            else known_collection_exists
        )
        if not exists:
            return []
        points: list[dict[str, Any]] = []
        for offset in range(0, len(ids), batch_size):
            batch = ids[offset : offset + batch_size]
            retrieved = client.retrieve(
                collection_name=self.collection,
                ids=batch,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(
                {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": point.payload or {},
                }
                for point in retrieved
            )
        return points

    def get_points_batched(
        self,
        ids: list[str],
        *,
        batch_size: int = QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE,
        _known_collection_exists: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Read a PostgreSQL-derived point scope without an unbounded Qdrant request."""

        normalized_ids = list(
            dict.fromkeys(
                str(point_id)
                for point_id in ids
                if point_id is not None and str(point_id)
            )
        )
        bounded_batch_size = max(1, min(int(batch_size), QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE))
        if not normalized_ids:
            return []
        return self._run_read_operation(
            lambda client: self._get_points_batched_from_client(
                client,
                normalized_ids,
                batch_size=bounded_batch_size,
                known_collection_exists=_known_collection_exists,
            ),
            lambda fallback: fallback.get_points(normalized_ids),
        )

    def get_payloads_batched_exact(
        self,
        ids: list[str],
        *,
        expected_vector_size: int,
        batch_size: int = QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """Retrieve one bounded exact payload scope without vectors or fallback.

        This is intentionally narrower than ``get_points_batched``.  Metadata
        maintenance must never read or rewrite embedding bytes, bootstrap a
        collection, or switch to the JSON fallback store.
        """

        if type(ids) is not list:
            raise ValueError("Exact Qdrant payload scope must be a list")
        normalized_ids = list(
            dict.fromkeys(
                str(point_id)
                for point_id in ids
                if type(point_id) is str and point_id
            )
        )
        if len(normalized_ids) != len(ids):
            raise ValueError("Exact Qdrant payload scope requires unique non-empty string ids")
        if not normalized_ids:
            return []
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or batch_size > QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE
        ):
            raise ValueError("Exact Qdrant payload batch_size is out of bounds")
        bounded_batch_size = batch_size
        vector_size = _positive_vector_size(
            expected_vector_size,
            source="Exact Qdrant payload vector size",
        )
        client = self.client
        if client is None:
            raise RuntimeError("Exact Qdrant payload retrieval requires Qdrant; fallback is forbidden")
        if not self._client_collection_exists(client):
            return []
        self._validate_collection_schema_on_client(client, vector_size=vector_size)
        points: list[dict[str, Any]] = []
        for offset in range(0, len(normalized_ids), bounded_batch_size):
            batch = normalized_ids[offset : offset + bounded_batch_size]
            retrieved = client.retrieve(
                collection_name=self.collection,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(
                {
                    "id": str(point.id),
                    "payload": point.payload or {},
                }
                for point in retrieved
            )
        return points

    def patch_vector_schema_metadata_exact(
        self,
        point_ids: list[str],
        *,
        expected_collection_name: str,
        candidate_vector_schema_hash: str,
        vector_runtime_state_hash: str,
    ) -> None:
        """Patch exactly the two frozen-schema runtime metadata fields.

        Callers must perform their own before/after payload fencing.  This
        primitive deliberately accepts no arbitrary payload mapping and never
        sends vector bytes.
        """

        if type(point_ids) is not list:
            raise ValueError("Exact Qdrant metadata patch point scope must be a list")
        normalized_ids = list(
            dict.fromkeys(
                str(point_id)
                for point_id in point_ids
                if type(point_id) is str and point_id
            )
        )
        if len(normalized_ids) != len(point_ids) or not normalized_ids:
            raise ValueError("Exact Qdrant metadata patch requires unique non-empty string ids")
        if type(expected_collection_name) is not str:
            raise ValueError("Exact Qdrant metadata patch collection must be a string")
        expected_collection = expected_collection_name
        if expected_collection != self.collection:
            raise ValueError("Exact Qdrant metadata patch collection confirmation mismatched")
        for label, value in (
            ("candidate_vector_schema_hash", candidate_vector_schema_hash),
            ("vector_runtime_state_hash", vector_runtime_state_hash),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or value.lower() != value
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"Exact Qdrant metadata patch {label} is not lowercase SHA-256")
        client = self.client
        if client is None:
            raise RuntimeError("Exact Qdrant metadata patch requires Qdrant; fallback is forbidden")
        if not self._client_collection_exists(client):
            raise RuntimeError("Exact Qdrant metadata patch collection does not exist")
        client.set_payload(
            collection_name=self.collection,
            payload={
                "candidate_vector_schema_hash": candidate_vector_schema_hash,
                "vector_runtime_state_hash": vector_runtime_state_hash,
            },
            points=normalized_ids,
            wait=True,
        )

    def expected_points_snapshot(
        self,
        expected_point_ids: list[str],
        *,
        expected_vector_size: int,
        batch_size: int = QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Read only the PostgreSQL-authoritative point scope for an active gate.

        This deliberately does not enumerate collections or scroll for orphan
        points.  The number of retrieve calls is bounded by the caller-provided
        PostgreSQL scope and the fixed maximum batch size.
        """

        expected_ids = list(
            dict.fromkeys(
                str(point_id)
                for point_id in expected_point_ids
                if point_id is not None and str(point_id)
            )
        )
        bounded_batch_size = max(
            1,
            min(int(batch_size), QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE),
        )
        expected_size = _positive_vector_size(
            expected_vector_size,
            source="Active Qdrant freshness vector size",
        )
        def primary(client: QdrantClient) -> dict[str, Any]:
            collection_exists = self._client_collection_exists(client)
            if not collection_exists:
                return {
                    "collection_exists": False,
                    "collection_schema_error": None,
                    "points": [],
                    "expected_point_count": len(expected_ids),
                    "retrieve_batch_size": bounded_batch_size,
                    "retrieve_batch_count": 0,
                    "orphan_scan_performed": False,
                }
            try:
                self._validate_collection_schema_on_client(
                    client,
                    vector_size=expected_size,
                )
            except VectorCollectionSchemaError as exc:
                return {
                    "collection_exists": True,
                    "collection_schema_error": str(exc),
                    "points": [],
                    "expected_point_count": len(expected_ids),
                    "retrieve_batch_size": bounded_batch_size,
                    "retrieve_batch_count": 0,
                    "orphan_scan_performed": False,
                }
            points = self._get_points_batched_from_client(
                client,
                expected_ids,
                batch_size=bounded_batch_size,
                known_collection_exists=True,
            )
            return {
                "collection_exists": True,
                "collection_schema_error": None,
                "points": points,
                "expected_point_count": len(expected_ids),
                "retrieve_batch_size": bounded_batch_size,
                "retrieve_batch_count": (
                    (len(expected_ids) + bounded_batch_size - 1) // bounded_batch_size
                    if expected_ids
                    else 0
                ),
                "orphan_scan_performed": False,
            }

        def fallback_snapshot(fallback: FallbackVectorStore) -> dict[str, Any]:
            points = fallback.get_points(expected_ids)
            return {
                "collection_exists": fallback.collection_exists(),
                "collection_schema_error": "active_qdrant_freshness_requires_qdrant_schema",
                "points": points,
                "expected_point_count": len(expected_ids),
                "retrieve_batch_size": bounded_batch_size,
                "retrieve_batch_count": (
                    (len(expected_ids) + bounded_batch_size - 1) // bounded_batch_size
                    if expected_ids
                    else 0
                ),
                "orphan_scan_performed": False,
            }

        return self._run_read_operation(primary, fallback_snapshot)

    def _list_ids_bounded_from_client(
        self,
        client: QdrantClient,
        knowledge_base_id: str,
        *,
        max_points: int,
        page_size: int,
        known_collection_exists: bool | None,
    ) -> dict[str, Any]:
        exists = (
            self._client_collection_exists(client)
            if known_collection_exists is None
            else known_collection_exists
        )
        if not exists:
            return {
                "ids": [],
                "truncated": False,
                "scanned_points": 0,
                "max_points": max_points,
            }
        qdrant_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="knowledge_base_id",
                    match=rest.MatchValue(value=knowledge_base_id),
                )
            ]
        )
        ids: list[str] = []
        offset = None
        max_pages = ((max_points + 1) + page_size - 1) // page_size
        exhausted = False
        for _page in range(max_pages):
            remaining_with_sentinel = max_points + 1 - len(ids)
            if remaining_with_sentinel <= 0:
                break
            points, next_offset = client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                limit=min(page_size, remaining_with_sentinel),
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(point.id) for point in points)
            offset = next_offset
            if offset is None:
                exhausted = True
                break
        truncated = len(ids) > max_points or not exhausted
        return {
            "ids": ids[:max_points],
            "truncated": truncated,
            "scanned_points": min(len(ids), max_points),
            "max_points": max_points,
        }

    def list_ids_bounded(
        self,
        knowledge_base_id: str,
        *,
        max_points: int = QDRANT_RECONCILE_MAX_SCANNED_POINTS,
        page_size: int = QDRANT_RECONCILE_SCROLL_PAGE_SIZE,
        _known_collection_exists: bool | None = None,
    ) -> dict[str, Any]:
        """Return a bounded KB-scoped orphan inventory and an explicit truncation flag."""

        bounded_max = max(1, min(int(max_points), QDRANT_RECONCILE_MAX_SCANNED_POINTS))
        bounded_page = max(1, min(int(page_size), QDRANT_RECONCILE_SCROLL_PAGE_SIZE))
        def fallback_inventory(fallback: FallbackVectorStore) -> dict[str, Any]:
            fallback_ids = fallback.list_ids(knowledge_base_id)
            return {
                "ids": fallback_ids[:bounded_max],
                "truncated": len(fallback_ids) > bounded_max,
                "scanned_points": min(len(fallback_ids), bounded_max),
                "max_points": bounded_max,
            }

        return self._run_read_operation(
            lambda client: self._list_ids_bounded_from_client(
                client,
                knowledge_base_id,
                max_points=bounded_max,
                page_size=bounded_page,
                known_collection_exists=_known_collection_exists,
            ),
            fallback_inventory,
        )

    def reconciliation_snapshot(
        self,
        knowledge_base_id: str,
        expected_point_ids: list[str],
        *,
        expected_vector_size: int | None = None,
        max_scanned_points: int = QDRANT_RECONCILE_MAX_SCANNED_POINTS,
    ) -> dict[str, Any]:
        """Read expected payload/vector state plus a bounded orphan inventory."""

        expected_ids = list(
            dict.fromkeys(
                str(point_id)
                for point_id in expected_point_ids
                if point_id is not None and str(point_id)
            )
        )
        validated_vector_size = (
            None
            if expected_vector_size is None
            else _positive_vector_size(
                expected_vector_size,
                source="Qdrant reconciliation vector size",
            )
        )
        bounded_max = max(
            1,
            min(int(max_scanned_points), QDRANT_RECONCILE_MAX_SCANNED_POINTS),
        )

        def primary(client: QdrantClient) -> dict[str, Any]:
            collection_schema_error: str | None = None
            collection_exists = self._client_collection_exists(client)
            if collection_exists and validated_vector_size is not None:
                try:
                    self._validate_collection_schema_on_client(
                        client,
                        vector_size=validated_vector_size,
                    )
                except VectorCollectionSchemaError as exc:
                    collection_schema_error = str(exc)
            points = self._get_points_batched_from_client(
                client,
                expected_ids,
                batch_size=QDRANT_RECONCILE_RETRIEVE_BATCH_SIZE,
                known_collection_exists=collection_exists,
            )
            inventory = self._list_ids_bounded_from_client(
                client,
                knowledge_base_id,
                max_points=bounded_max,
                page_size=QDRANT_RECONCILE_SCROLL_PAGE_SIZE,
                known_collection_exists=collection_exists,
            )
            return {
                "points": points,
                "scanned_ids": list(inventory["ids"]),
                "scan_truncated": bool(inventory["truncated"]),
                "scanned_points": int(inventory["scanned_points"]),
                "max_scanned_points": int(inventory["max_points"]),
                "collection_schema_error": collection_schema_error,
            }

        def fallback_snapshot(fallback: FallbackVectorStore) -> dict[str, Any]:
            points = fallback.get_points(expected_ids)
            fallback_ids = fallback.list_ids(knowledge_base_id)
            return {
                "points": points,
                "scanned_ids": fallback_ids[:bounded_max],
                "scan_truncated": len(fallback_ids) > bounded_max,
                "scanned_points": min(len(fallback_ids), bounded_max),
                "max_scanned_points": bounded_max,
                "collection_schema_error": None,
            }

        return self._run_read_operation(primary, fallback_snapshot)

    def _list_ids_from_client(
        self,
        client: QdrantClient,
        knowledge_base_id: str | None,
    ) -> list[str]:
        if not self._client_collection_exists(client):
            return []
        qdrant_filter = None
        if knowledge_base_id:
            qdrant_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="knowledge_base_id",
                        match=rest.MatchValue(value=knowledge_base_id),
                    )
                ]
            )
        ids: list[str] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(point.id) for point in points)
            if offset is None:
                break
        return ids

    def list_ids(self, knowledge_base_id: str | None = None) -> list[str]:
        return self._run_read_operation(
            lambda client: self._list_ids_from_client(client, knowledge_base_id),
            lambda fallback: fallback.list_ids(knowledge_base_id),
        )

    def health_check(self, knowledge_base_id: str, active_chunk_ids: list[str]) -> dict:
        vector_ids = set(self.list_ids(knowledge_base_id))
        active_ids = set(active_chunk_ids)
        missing = sorted(active_ids - vector_ids)
        stale = sorted(vector_ids - active_ids)
        return {"ok": not missing, "missing": missing, "stale": stale}

    def search(self, vector: list[float], limit: int, filters: dict[str, Any] | None = None) -> list[dict]:
        vector = canonical_embedding_vector(vector, source="Qdrant search vector")
        limit = _positive_vector_size(limit, source="Qdrant search limit")
        filters = {key: value for key, value in (filters or {}).items() if value not in (None, "", [], {})}
        qdrant_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key=key,
                    match=rest.MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
        ) if filters else None

        def primary(client: QdrantClient) -> list[dict]:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                results = response.points
            else:
                results = client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                )
            return [{"id": item.id, "score": item.score, "payload": item.payload} for item in results]

        return self._run_read_operation(
            primary,
            lambda fallback: fallback.search(
                vector=vector,
                limit=limit,
                filters=filters,
            ),
        )

    async def async_search(self, vector: list[float], limit: int, filters: dict[str, Any] | None = None) -> list[dict]:
        """Async wrapper for search to avoid blocking the event loop."""
        return await asyncio.to_thread(self.search, vector=vector, limit=limit, filters=filters)
