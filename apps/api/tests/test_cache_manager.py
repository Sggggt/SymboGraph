from app.services.cache_manager import CacheManager


def test_cache_manager_disables_cache_without_redis():
    manager = CacheManager()
    manager._redis = None

    manager.set_runtime_state("profile", "session-1", {"value": 1}, ttl=60)
    assert manager.get_runtime_state("profile", "session-1") is None

    manager.set_search_results("kb-1", "query", "scope", "embedding-v1", {"ids": ["c1"]}, ttl=60)
    assert manager.get_search_results("kb-1", "query", "scope", "embedding-v1") is None
