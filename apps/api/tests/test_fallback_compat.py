from __future__ import annotations

import pytest


@pytest.mark.fallback_compat
def test_fallback_vector_store_round_trips_with_file_lock(tmp_path):
    from app.services.vector_store import FallbackVectorStore

    store = FallbackVectorStore(tmp_path / "vector_index.json")
    points = [
        {
            "id": "chunk-1",
            "vector": [1.0, 0.0, 0.0],
            "payload": {"knowledge_base_id": "KnowledgeBase-1", "partition": "L1"},
        },
        {
            "id": "chunk-2",
            "vector": [0.0, 1.0, 0.0],
            "payload": {"knowledge_base_id": "KnowledgeBase-1", "partition": "L2"},
        },
    ]

    store.upsert(points)
    assert set(store.list_ids("KnowledgeBase-1")) == {"chunk-1", "chunk-2"}
    assert store.get_points(["chunk-1"])[0]["id"] == "chunk-1"
    assert store.search([1.0, 0.0, 0.0], 1, {"knowledge_base_id": "KnowledgeBase-1"})[0]["id"] == "chunk-1"

    store.delete(["chunk-1"])
    assert store.list_ids("KnowledgeBase-1") == ["chunk-2"]
