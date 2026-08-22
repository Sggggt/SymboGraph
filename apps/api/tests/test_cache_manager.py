import pytest

from app.services.cache_manager import (
    ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS,
    ACTIVE_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION,
    CacheInvalidationError,
    CacheManager,
)


def _cache_components(knowledge_base_id: str = "kb-1") -> dict:
    components = {
        field: None for field in ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS
    }
    components.update(
        {
            "cache_key_protocol_version": (
                ACTIVE_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION
            ),
            "knowledge_base_id": knowledge_base_id,
            "query": "query",
            "filters": {},
            "retrieval_mode": "layered_context_graph",
            "retrieval_granularity": "mid",
        }
    )
    return components


def test_cache_manager_disables_cache_without_redis():
    manager = CacheManager()
    manager._redis = None

    manager.set_runtime_state("profile", "session-1", {"value": 1}, ttl=60)
    assert manager.get_runtime_state("profile", "session-1") is None

    components = _cache_components()
    manager.set_search_results(
        "kb-1",
        {"ids": ["c1"]},
        ttl=60,
        cache_components=components,
    )
    assert (
        manager.get_search_results(
            "kb-1",
            cache_components=components,
        )
        is None
    )


def test_active_search_cache_rejects_partial_identity_components():
    manager = CacheManager.__new__(CacheManager)
    manager._redis = None

    with pytest.raises(ValueError, match="complete versioned identity"):
        manager.get_search_results(
            "kb-1",
            cache_components={"knowledge_base_id": "kb-1"},
        )


def test_strict_knowledge_base_invalidation_fails_when_redis_is_unavailable(monkeypatch):
    manager = CacheManager.__new__(CacheManager)
    manager._redis = None
    monkeypatch.setattr(manager, "_try_connect", lambda: None)

    with pytest.raises(CacheInvalidationError, match="Redis is not connected"):
        manager.invalidate_knowledge_base("kb-1", strict=True)

    assert manager.invalidate_knowledge_base("kb-1") is False


def test_strict_knowledge_base_invalidation_exposes_partial_redis_failure():
    class FailingRedis:
        def scan_iter(self, *, match):
            assert match == "kg:*:kb-1:*"
            yield b"kg:search:kb-1:first"
            raise OSError("redis connection dropped")

        def delete(self, key):
            assert key == b"kg:search:kb-1:first"

    manager = CacheManager.__new__(CacheManager)
    manager._redis = FailingRedis()

    with pytest.raises(CacheInvalidationError, match="before completion"):
        manager.invalidate_knowledge_base("kb-1", strict=True)
