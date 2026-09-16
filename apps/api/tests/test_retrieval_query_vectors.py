import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.services import context_graph as cg
from app.services.retrieval_models import TaskPlanningOutput, compile_task_plan, routing_facets
from app.services.retrieval_query_vectors import prepare_query_vectors
from test_retrieval_models import plan_payload


def target_fixture():
    schema = SimpleNamespace(embedding_model="unit-test-embedding", embedding_dimension=2,
        embedding_text_version="unit-test-text", chunk_schema_version="unit-test-chunk",
        collection_name="unit-test-collection", collection_identity_digest="unit-test-identity")
    return SimpleNamespace(schema=schema, vector_schema_hash="unit-test-schema",
        runtime_state_id="unit-test-state", runtime_state_hash="unit-test-runtime",
        activation_generation=1, ready_vector_status="ready")


def task_plan(multiple=False):
    payload = plan_payload()
    if multiple:
        payload['requirements'].append({'facet': 'queue waiting time distribution',
            'lexical_role': 'domain', 'kind': 'definition'})
    _, task, strategy, facets = compile_task_plan(TaskPlanningOutput.model_validate(payload),
        question="Find maximum queue waiting time.", knowledge_base_id="unit-test-kb",
        conversation_scope_hash="a" * 64, retrieval_granularity="mid")
    return dict(task=task, strategy=strategy, initial_facets=facets)


@pytest.mark.asyncio
@pytest.mark.parametrize('multiple', [False, True])
async def test_combined_call_preserves_measurement_order_and_memo_identity(no_fallback_env, multiple):
    plan, target, calls = task_plan(multiple), target_fixture(), []
    class Provider:
        async def embed_texts(self, texts, text_type):
            calls.append((list(texts), text_type))
            return [[float(index + 1), 1.0] for index in range(len(texts))]
    vectors, memo, audit = await prepare_query_vectors(**plan, target=target, provider=Provider())
    task = plan['task']
    semantic = cg.semantic_entry_query_for_search(task.question, routing_facets(**plan))
    expected_texts = ['; '.join(f.text for f in task.requirements),
                      *[f.text for f in task.requirements], semantic['query']]
    unique = list(dict.fromkeys(expected_texts))
    assert calls == [(unique, 'query')]
    assert vectors == [[float(unique.index(text) + 1), 1.0] for text in expected_texts[:-1]]
    key = cg.query_embedding_request_memo_key(task.knowledge_base_id, semantic, target)
    assert key == cg.stable_hash({
        'protocol_version': cg.QUERY_EMBEDDING_REQUEST_MEMO_PROTOCOL_VERSION,
        'knowledge_base_id': task.knowledge_base_id,
        'semantic_entry_query_packet_hash': semantic['packet_hash'],
        'vector_identity': cg._entry_dense_vector_identity_from_target(target)})
    assert memo.get(key, expected_dimension=2) == [float(unique.index(semantic['query']) + 1), 1.0]
    assert audit.input_count == len(expected_texts) and audit.unique_text_count == len(unique)
    for field in ('runtime_state_hash', 'activation_generation', 'vector_schema_hash'):
        changed = deepcopy(target)
        setattr(changed, field, 2 if field == 'activation_generation' else 'unit-test-changed')
        assert memo.get(cg.query_embedding_request_memo_key(task.knowledge_base_id, semantic, changed),
                        expected_dimension=2) is None
    assert memo.get(cg.query_embedding_request_memo_key(task.knowledge_base_id,
        {**semantic, 'packet_hash': 'unit-test-changed-query'}, target), expected_dimension=2) is None
    assert cg.QueryEmbeddingRequestMemo().get(key, expected_dimension=2) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [[], [[1.0]], [[float('nan'), 1.0]], [[0.0, 0.0]]])
async def test_bad_batch_does_not_publish_partial_memo(no_fallback_env, monkeypatch, response):
    writes = []
    monkeypatch.setattr(cg.QueryEmbeddingRequestMemo, 'put', lambda *args: writes.append(args))
    class Provider:
        async def embed_texts(self, texts, text_type):
            return response
    with pytest.raises(RuntimeError):
        await prepare_query_vectors(**task_plan(), target=target_fixture(), provider=Provider())
    assert writes == []


@pytest.mark.asyncio
async def test_cancelled_batch_does_not_publish_vectors(no_fallback_env, monkeypatch):
    entered, released, writes = asyncio.Event(), asyncio.Event(), []
    monkeypatch.setattr(cg.QueryEmbeddingRequestMemo, 'put', lambda *args: writes.append(args))
    class Provider:
        async def embed_texts(self, texts, text_type):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
    operation = asyncio.create_task(prepare_query_vectors(**task_plan(), target=target_fixture(), provider=Provider()))
    await asyncio.wait_for(entered.wait(), 1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert released.is_set() and writes == []
