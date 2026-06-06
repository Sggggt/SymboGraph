from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

from app.core.config import get_settings


class CacheManager:
    """Lightweight Redis cache wrapper with in-memory fallback."""

    def __init__(self) -> None:
        self._redis = None
        self._memory: dict[str, Any] = {}
        self._settings = get_settings()
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            import redis as redis_lib

            self._redis = redis_lib.from_url(self._settings.redis_url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
            self._redis.ping()
        except Exception:
            self._redis = None

    def _key(self, namespace: str, *parts: str) -> str:
        safe = ":".join(str(p) for p in parts)
        return f"kg:{namespace}:{safe}"

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_embedding(self, course_id: str, query: str, embedding_version: str) -> list[float] | None:
        key = self._key("emb", course_id, self._hash(query), embedding_version)
        return self._get(key)

    def set_embedding(self, course_id: str, query: str, embedding_version: str, vector: list[float], ttl: int = 600) -> None:
        key = self._key("emb", course_id, self._hash(query), embedding_version)
        self._set(key, vector, ttl)

    def get_search_results(self, course_id: str, query: str, filters_hash: str, embedding_version: str) -> list[dict] | None:
        key = self._key("search", course_id, self._hash(query), filters_hash, embedding_version)
        return self._get(key)

    def set_search_results(self, course_id: str, query: str, filters_hash: str, embedding_version: str, results: list[dict], ttl: int = 300) -> None:
        key = self._key("search", course_id, self._hash(query), filters_hash, embedding_version)
        self._set(key, results, ttl)

    def get_quality_judgment(self, cache_key: str) -> dict | None:
        value = self._get(self._key("quality_judge", cache_key))
        return value if isinstance(value, dict) else None

    def set_quality_judgment(self, cache_key: str, result: dict, ttl: int = 86400) -> None:
        self._set(self._key("quality_judge", cache_key), result, ttl)

    def get_runtime_state(self, namespace: str, state_id: str) -> dict | None:
        value = self._get(self._key("state", namespace, state_id))
        return value if isinstance(value, dict) else None

    def set_runtime_state(self, namespace: str, state_id: str, state: dict, ttl: int = 86400) -> None:
        self._set(self._key("state", namespace, state_id), state, ttl)

    def delete_runtime_state(self, namespace: str, state_id: str) -> None:
        key = self._key("state", namespace, state_id)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._memory.pop(key, None)

    def invalidate_course(self, course_id: str) -> None:
        if self._redis:
            try:
                for key in self._redis.scan_iter(match=f"kg:*:{course_id}:*"):
                    self._redis.delete(key)
            except Exception:
                pass
        # memory fallback: purge keys containing course_id
        self._memory = {k: v for k, v in self._memory.items() if course_id not in k}

    def _get(self, key: str) -> Any | None:
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    return pickle.loads(raw)
            except Exception:
                pass
        return self._memory.get(key)

    def _set(self, key: str, value: Any, ttl: int) -> None:
        if self._redis:
            try:
                self._redis.setex(key, ttl, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                return
            except Exception:
                pass
        self._memory[key] = value


_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_cache_manager() -> None:
    global _cache_manager
    _cache_manager = None
