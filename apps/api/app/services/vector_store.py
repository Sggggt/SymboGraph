from __future__ import annotations

import asyncio
import json
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.core.config import get_settings

try:
    import portalocker
except ImportError:  # pragma: no cover - optional dependency in fallback-only paths
    portalocker = None


_VECTOR_FILE_LOCKS: OrderedDict[Path, Lock] = OrderedDict()
_VECTOR_FILE_LOCKS_GUARD = Lock()
_MAX_VECTOR_FILE_LOCKS = 128


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
    lock_path.parent.mkdir(parents=True, exist_ok=True)
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


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_mag = math.sqrt(sum(a * a for a in left)) or 1.0
    right_mag = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_mag * right_mag)


class FallbackVectorStore:
    def __init__(self, backing_file: Path) -> None:
        self.backing_file = backing_file
        self.backing_file.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.backing_file.exists():
            return []
        text = self.backing_file.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                data, _ = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                return []
        return data if isinstance(data, list) else []

    def _write(self, data: list[dict]) -> None:
        temporary_file = self.backing_file.with_suffix(f"{self.backing_file.suffix}.tmp")
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_file, self.backing_file)
        try:
            fsync_directory(self.backing_file.parent)
        except OSError:
            pass

    def upsert(self, points: list[dict]) -> None:
        with vector_process_lock(self.backing_file):
            current = self._read()
            indexed = {item["id"]: item for item in current}
            for point in points:
                indexed[point["id"]] = point
            self._write(list(indexed.values()))

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        id_set = set(ids)
        with vector_process_lock(self.backing_file):
            current = self._read()
            self._write([item for item in current if item.get("id") not in id_set])

    def get_points(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        id_set = set(ids)
        with vector_process_lock(self.backing_file):
            current = self._read()
        return [item for item in current if item.get("id") in id_set]

    def list_ids(self, course_id: str | None = None) -> list[str]:
        with vector_process_lock(self.backing_file):
            current = self._read()
        ids = []
        for item in current:
            payload = item.get("payload", {})
            if course_id and payload.get("course_id") != course_id:
                continue
            ids.append(item["id"])
        return ids

    def search(self, vector: list[float], limit: int, filters: dict[str, Any]) -> list[dict]:
        with vector_process_lock(self.backing_file):
            points = self._read()
        results = []
        for point in points:
            payload = point.get("payload", {})
            if any(filters.get(key) and payload.get(key) != filters[key] for key in filters):
                continue
            score = cosine_similarity(vector, point["vector"])
            results.append({"id": point["id"], "score": score, "payload": payload})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


class VectorStore:
    def __init__(self, course_name: str | None = None) -> None:
        self.settings = get_settings()
        self.collection = self.settings.qdrant_collection
        course_paths = self.settings.course_paths_for_name(course_name or self.settings.course_name)
        self.fallback = FallbackVectorStore(course_paths["ingestion_root"] / "vector_index.json")
        self.client: QdrantClient | None = None
        try:
            self.client = QdrantClient(url=self.settings.qdrant_url, timeout=5.0)
            self._ensure_collection()
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            self.client = None

    def _ensure_collection(self) -> None:
        if not self.client:
            return
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection not in collections:
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=rest.VectorParams(
                        size=self.settings.embedding_dimensions,
                        distance=rest.Distance.COSINE,
                    ),
                )
            except Exception:
                collections = {item.name for item in self.client.get_collections().collections}
                if self.collection not in collections:
                    raise

    def upsert(self, points: list[dict]) -> None:
        if not points:
            return
        if self.client:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    rest.PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
                    for point in points
                ],
                wait=True,
            )
        else:
            if not self.settings.enable_model_fallback:
                raise RuntimeError("Qdrant is unavailable and ENABLE_MODEL_FALLBACK is false")
            self.fallback.upsert(points)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        if self.client:
            self.client.delete(
                collection_name=self.collection,
                points_selector=rest.PointIdsList(points=ids),
                wait=True,
            )
        if self.settings.enable_model_fallback:
            self.fallback.delete(ids)

    async def async_upsert(self, points: list[dict]) -> None:
        """Async wrapper for upsert to avoid blocking the event loop."""
        await asyncio.to_thread(self.upsert, points)

    async def async_delete(self, ids: list[str]) -> None:
        """Async wrapper for delete to avoid blocking the event loop."""
        await asyncio.to_thread(self.delete, ids)

    def get_points(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        if self.client:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=ids,
                with_payload=True,
                with_vectors=True,
            )
            return [
                {"id": str(point.id), "vector": point.vector, "payload": point.payload or {}}
                for point in points
            ]
        if not self.settings.enable_model_fallback:
            raise RuntimeError("Qdrant is unavailable and ENABLE_MODEL_FALLBACK is false")
        return self.fallback.get_points(ids)

    def list_ids(self, course_id: str | None = None) -> list[str]:
        if self.client:
            qdrant_filter = None
            if course_id:
                qdrant_filter = rest.Filter(
                    must=[rest.FieldCondition(key="course_id", match=rest.MatchValue(value=course_id))]
                )
            ids: list[str] = []
            offset = None
            while True:
                points, offset = self.client.scroll(
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
        if not self.settings.enable_model_fallback:
            raise RuntimeError("Qdrant is unavailable and ENABLE_MODEL_FALLBACK is false")
        return self.fallback.list_ids(course_id)

    def health_check(self, course_id: str, active_chunk_ids: list[str]) -> dict:
        vector_ids = set(self.list_ids(course_id))
        active_ids = set(active_chunk_ids)
        missing = sorted(active_ids - vector_ids)
        stale = sorted(vector_ids - active_ids)
        return {"ok": not missing, "missing": missing, "stale": stale}

    def search(self, vector: list[float], limit: int, filters: dict[str, Any] | None = None) -> list[dict]:
        filters = {key: value for key, value in (filters or {}).items() if value not in (None, "", [], {})}
        if self.client:
            qdrant_filter = rest.Filter(
                must=[rest.FieldCondition(key=key, match=rest.MatchValue(value=value)) for key, value in filters.items()]
            ) if filters else None
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                )
            return [{"id": item.id, "score": item.score, "payload": item.payload} for item in results]
        if not self.settings.enable_model_fallback:
            raise RuntimeError("Qdrant is unavailable and ENABLE_MODEL_FALLBACK is false")
        return self.fallback.search(vector=vector, limit=limit, filters=filters)

    async def async_search(self, vector: list[float], limit: int, filters: dict[str, Any] | None = None) -> list[dict]:
        """Async wrapper for search to avoid blocking the event loop."""
        return await asyncio.to_thread(self.search, vector=vector, limit=limit, filters=filters)
