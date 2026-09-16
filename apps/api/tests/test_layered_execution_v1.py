import pytest
from sqlalchemy import func, select

from app.intent_contracts import IntentPlanningOutput, accept_plan
from app.models import GraphRetrievalStep, RetrievalLexicalReward
from app.schemas import SearchFilters
from app.services import layered_execution_v1 as current
from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
from app.services.intent_planning import retrieval_capability_manifest
from app.services.lexical_index import prepare_bm25_snapshot
from app.services.lexical_storage import (
    active_lexical_sources,
    stage_lexical_index,
    publish_lexical_index,
)


def accepted_plan(db_session, knowledge_base_id, raw, question="Summarize the topics."):
    capabilities = retrieval_capability_manifest(db_session, knowledge_base_id)
    return accept_plan(
        IntentPlanningOutput.model_validate(raw),
        question=question,
        conversation_scope_hash="c" * 64,
        filter_scope_hash="d" * 64,
        capabilities=capabilities,
    )


def test_result_node_deduplication_keeps_the_first_stable_graph_path():
    first = current.TraversedNode(
        node_id="chunk-a",
        root_node_id="root-best",
        distance=0.2,
        depth=1,
        node_path=("root-best", "chunk-a"),
        edge_path=("edge-best",),
        support_ids=("support-best",),
        entry_score=0.8,
        entry_channels=(),
    )
    duplicate = current.TraversedNode(
        node_id="chunk-a",
        root_node_id="root-other",
        distance=0.4,
        depth=1,
        node_path=("root-other", "chunk-a"),
        edge_path=("edge-other",),
        support_ids=("support-other",),
        entry_score=0.7,
        entry_channels=(),
    )
    second = current.TraversedNode(
        node_id="chunk-b",
        root_node_id="root-best",
        distance=0.5,
        depth=2,
        node_path=("root-best", "chunk-a", "chunk-b"),
        edge_path=("edge-best", "edge-next"),
        support_ids=("support-best",),
        entry_score=0.8,
        entry_channels=(),
    )

    unique = current._unique_traversed_nodes((first, duplicate, second))

    assert [item.node_id for item in unique] == ["chunk-a", "chunk-b"]
    assert unique[0] is first


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["coarse", "mid", "chunk"])
async def test_each_llm_selected_root_layer_runs_one_frozen_dense_execution(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
    layer,
):
    from test_intent_contracts import proposal

    monkeypatch.setattr(current, "EmbeddingProvider", fake_model_stack["EmbeddingProvider"])
    kb_id = populated_context_graph["knowledge_base"].id
    plan = accepted_plan(db_session, kb_id, proposal(layer=layer))
    result = await current.execute_layered_retrieval(
        db_session,
        plan=plan,
        filters=SearchFilters(),
        top_k=4,
    )
    assert result.results
    result_ids = [item["chunk_id"] for item in result.results]
    assert len(result_ids) == len(set(result_ids))
    assert result.trace.result_chunk_ids_json == result_ids
    assert result.trace.retrieval_mode == "intent_execution_retrieval_v1"
    assert result.audit["entry_layer"] == layer
    assert result.audit["post_retrieval_model_call_count"] == 0
    assert result.audit["reward_call_count"] == 0
    assert db_session.scalar(
        select(func.count()).select_from(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == result.trace.id
        )
    ) == {"coarse": 3, "mid": 2, "chunk": 1}[layer]
    assert db_session.scalar(select(func.count()).select_from(RetrievalLexicalReward)) == 0
    assert all(
        item["metadata"]["retrieval_protocol_version"] == "intent_execution_retrieval_v1"
        for item in result.results
    )


@pytest.mark.asyncio
async def test_hybrid_chunk_execution_uses_the_published_lexical_snapshot(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from test_intent_contracts import proposal

    monkeypatch.setattr(current, "EmbeddingProvider", fake_model_stack["EmbeddingProvider"])
    kb = populated_context_graph["knowledge_base"]
    snapshot = prepare_bm25_snapshot(
        kb.id,
        active_lexical_sources(db_session, kb.id),
    )
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        kb.id,
        operation="unit_test_lexical_index_build",
    ):
        job = stage_lexical_index(db_session, snapshot)
        publish_lexical_index(db_session, job.id, invalidate=lambda *a, **k: True)
    raw = proposal(layer="chunk", hybrid=True, lexical=True)
    raw["execution_strategy"]["lexical_groups"][0]["surfaces"][0]["text"] = "Bayesian"
    plan = accepted_plan(db_session, kb.id, raw)
    result = await current.execute_layered_retrieval(
        db_session,
        plan=plan,
        filters=SearchFilters(),
        top_k=4,
    )
    assert result.results
    assert result.audit["lexical"]["enabled"] is True
    assert result.audit["lexical"]["index_identity"] == snapshot.identity
    assert {
        contribution["channel"]
        for entry in result.trace.entry_nodes_json
        for contribution in entry["channels"]
    } >= {"dense", "bm25"}
    floor_ids = set(
        result.trace.topk_selection_json["channel_floor_chunk_ids"]
    )
    selected_ids = {item["chunk_id"] for item in result.results}
    assert result.trace.topk_selection_json["channel_floor_protocol_version"] == (
        "entry_channel_floor_v1"
    )
    assert floor_ids <= selected_ids
    path_ids = {
        str(label["chunk_id"])
        for label in result.trace.path_labels_json
        if label.get("path") and label.get("root_node_id")
    }
    assert selected_ids <= path_ids


@pytest.mark.asyncio
async def test_target_cache_replays_postgresql_trace_without_embedding_or_traversal(
    db_session,
    populated_context_graph,
    fake_model_stack,
    monkeypatch,
):
    from app.services import cache_manager as cache_module
    from app.services.cache_manager import CacheManager
    from test_intent_contracts import proposal

    class MemoryRedis:
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
    manager._redis = MemoryRedis()
    monkeypatch.setattr(cache_module, "get_cache_manager", lambda: manager)
    provider = fake_model_stack["EmbeddingProvider"]
    original_embed = provider.embed_texts
    query_calls = []

    async def count_queries(self, texts, text_type="document"):
        if text_type == "query":
            query_calls.append(tuple(texts))
        return await original_embed(self, texts, text_type=text_type)

    monkeypatch.setattr(provider, "embed_texts", count_queries)
    monkeypatch.setattr(current, "EmbeddingProvider", provider)
    kb_id = populated_context_graph["knowledge_base"].id
    plan = accepted_plan(db_session, kb_id, proposal(layer="chunk"))
    first = await current.execute_layered_retrieval(
        db_session,
        plan=plan,
        filters=SearchFilters(),
        top_k=4,
    )
    assert first.cache_audit["status"] == "miss"
    db_session.commit()
    assert current.publish_intent_retrieval_cache(first) is True
    second = await current.execute_layered_retrieval(
        db_session,
        plan=plan,
        filters=SearchFilters(),
        top_k=4,
    )
    assert second.cache_audit["status"] == "hit"
    assert second.trace.id != first.trace.id
    assert second.trace.result_chunk_ids_json == first.trace.result_chunk_ids_json
    assert [item["chunk_id"] for item in second.results] == [
        item["chunk_id"] for item in first.results
    ]
    assert len(query_calls) == 1
    assert db_session.scalar(
        select(func.count()).select_from(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == second.trace.id
        )
    ) == 1
    from app.schemas import IntentExecutionRetrievalTraceStepsResponse
    from app.services.retrieval import get_retrieval_trace_steps

    replay = get_retrieval_trace_steps(db_session, second.trace.id)
    validated = IntentExecutionRetrievalTraceStepsResponse.model_validate(replay)
    assert validated.entry_layer == "chunk"
    assert validated.retrieval_cache is not None
    assert validated.retrieval_cache.cache_hit is True
