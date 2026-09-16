import pytest

from app.services.cache_manager import (
    ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS,
    ACTIVE_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION,
    CacheInvalidationError,
    CacheManager,
    INTENT_RETRIEVAL_CACHE_COMPONENT_FIELDS,
    INTENT_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION,
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
            "source_filter_protocol": "active_source_filters_v2",
            "retrieval_mode": "layered_context_graph",
            "retrieval_granularity": "mid",
        }
    )
    return components


def _intent_cache_components(knowledge_base_id: str = "kb-1") -> dict:
    components = {
        field: "a" * 64 for field in INTENT_RETRIEVAL_CACHE_COMPONENT_FIELDS
    }
    components.update(
        {
            "cache_key_protocol_version": INTENT_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION,
            "knowledge_base_id": knowledge_base_id,
            "entry_layer": "chunk",
            "graph_state_id": "state-1",
            "embedding_identity": {
                "embedding_model": "unit-model",
                "embedding_dimension": 4,
                "embedding_text_version": "unit-v1",
            },
            "lexical_identity": None,
            "generate_lexical": False,
            "hybrid": False,
            "result_top_k": 4,
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


@pytest.mark.parametrize('field,value',[
    ('cache_key_protocol_version','layered_retrieval_full_identity_key_v5'),
    ('source_filter_protocol','active_source_filters_v1'),
])
def test_previous_filter_semantics_cannot_enter_active_cache(field,value):
    from app.services.cache_manager import validate_active_retrieval_cache_components
    components=_cache_components()
    components[field]=value
    with pytest.raises(ValueError,match='protocol mismatch'):
        validate_active_retrieval_cache_components('kb-1',components)


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


def test_knowledge_base_invalidation_batches_redis_round_trips():
    class Pipeline:
        def __init__(self, owner):
            self.owner = owner
            self.keys = []

        def delete(self, *keys):
            self.keys.extend(keys)
            return self

        def execute(self):
            self.owner.round_trips.append(tuple(self.keys))
            return [len(self.keys)]

    class Redis:
        def __init__(self):
            self.round_trips = []

        def scan_iter(self, *, match):
            assert match == "kg:*:kb-1:*"
            yield from (f"kg:search:kb-1:{index}".encode() for index in range(600))

        def pipeline(self, *, transaction):
            assert transaction is False
            return Pipeline(self)

    manager = CacheManager.__new__(CacheManager)
    manager._redis = Redis()
    assert manager.invalidate_knowledge_base("kb-1", strict=True) is True
    assert [len(batch) for batch in manager._redis.round_trips] == [256, 256, 88]


def test_intent_cache_is_strictly_scoped_and_expired_payload_is_deleted():
    class Redis:
        def __init__(self):
            self.values = {}
            self.ttls = {}

        def get(self, key):
            return self.values.get(key)

        def setex(self, key, ttl, value):
            self.values[key] = value
            self.ttls[key] = ttl

        def ttl(self, key):
            return self.ttls.get(key, -2)

        def delete(self, key):
            existed = key in self.values
            self.values.pop(key, None)
            self.ttls.pop(key, None)
            return int(existed)

    manager = CacheManager.__new__(CacheManager)
    manager._redis = Redis()
    components = _intent_cache_components()
    manager.set_intent_retrieval(
        "kb-1",
        {"protocol_version": "unit-cache-payload"},
        ttl=60,
        cache_components=components,
    )
    first = manager.read_intent_retrieval(
        "kb-1",
        cache_components=components,
    )
    assert first.status == "hit"
    with pytest.raises(ValueError, match="knowledge-base identity mismatch"):
        manager.read_intent_retrieval(
            "kb-2",
            cache_components=components,
        )
    key, _digest = manager._intent_retrieval_key("kb-1", components)
    manager._redis.ttls[key] = 0
    expired = manager.read_intent_retrieval(
        "kb-1",
        cache_components=components,
    )
    assert expired.status == "poison"
    assert expired.deletion_attempted and expired.deleted
    assert key not in manager._redis.values
