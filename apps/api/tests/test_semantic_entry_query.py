from __future__ import annotations

from copy import deepcopy

import pytest


RAW_INSTRUCTION_QUERY = (
    "根据资料，用一个完整句子原样回答：技术必须被什么夹在中间？"
    "不要添加引用标记、来源信息、标题或引号。"
)


def _llm_facet_payload(*, specific_facet: str) -> dict:
    return {
        "facet_groups": [
            {
                "facet": "技术",
                "role": "domain",
                "aliases": [],
            },
            {
                "facet": specific_facet,
                "role": "constraint",
                "aliases": [],
            },
        ],
        "answer_shape": "grounded_answer",
        "drop_terms": ["根据资料", "原样回答", "不要添加引用标记"],
    }


def test_semantic_entry_uses_most_specific_validated_facet_without_gray_authority():
    from app.schemas import SearchFilters
    from app.services import context_graph

    facets = context_graph.query_facets_for_search(
        RAW_INSTRUCTION_QUERY,
        _llm_facet_payload(specific_facet="技术必须被什么夹在中间"),
    )
    semantic = context_graph.semantic_entry_query_for_search(
        RAW_INSTRUCTION_QUERY,
        facets,
    )

    assert semantic["protocol_version"] == (
        "validated_query_facet_semantic_entry_v3"
    )
    assert semantic["query"] == "技术必须被什么夹在中间"
    assert semantic["selection_source"] == "validated_required_facet"
    assert semantic["source_packet_model_assisted"] is True
    assert semantic["is_evidence"] is False
    assert semantic["citation_authority"] is False
    assert semantic["gray_zone_decision_authority"] is False
    assert semantic["selection_model_call_count"] == 0
    assert semantic["gray_zone_model_call_count"] == 0

    gray_before = context_graph.deterministic_gray_query_facets_for_search(
        RAW_INSTRUCTION_QUERY
    )
    changed_facets = context_graph.query_facets_for_search(
        RAW_INSTRUCTION_QUERY,
        _llm_facet_payload(specific_facet="需求和验证之间的技术"),
    )
    changed_semantic = context_graph.semantic_entry_query_for_search(
        RAW_INSTRUCTION_QUERY,
        changed_facets,
    )
    gray_after = context_graph.deterministic_gray_query_facets_for_search(
        RAW_INSTRUCTION_QUERY
    )

    assert changed_semantic["packet_hash"] != semantic["packet_hash"]
    assert gray_after == gray_before
    assert gray_after["query"] == RAW_INSTRUCTION_QUERY
    assert gray_after["diagnostics"]["external_routing_packet_used"] is False
    assert gray_after["diagnostics"]["model_call_count"] == 0

    common = {
        "knowledge_base_id": "kb-semantic-entry",
        "query": RAW_INSTRUCTION_QUERY,
        "filters": SearchFilters(),
        "context_state": None,
        "retrieval_mode": "layered_context_graph",
        "conversation_state_scope_hash": "a" * 64,
    }
    components = context_graph.context_graph_cache_key_components(
        **common,
        query_facets=facets,
    )
    changed_components = context_graph.context_graph_cache_key_components(
        **common,
        query_facets=changed_facets,
    )
    assert components["cache_key_protocol_version"] == (
        "layered_retrieval_full_identity_key_v6"
    )
    assert components["query"] == RAW_INSTRUCTION_QUERY
    assert components["semantic_entry_query"] == "技术必须被什么夹在中间"
    assert components["semantic_entry_query_hash"] == semantic["packet_hash"]
    assert changed_components["semantic_entry_query_hash"] != components[
        "semantic_entry_query_hash"
    ]


def test_semantic_entry_keeps_raw_query_for_deterministic_fallback_packet():
    from app.services import context_graph

    facets = context_graph.query_facets_for_search(RAW_INSTRUCTION_QUERY)
    semantic = context_graph.semantic_entry_query_for_search(
        RAW_INSTRUCTION_QUERY,
        facets,
    )

    assert semantic["query"] == RAW_INSTRUCTION_QUERY
    assert semantic["selection_source"] == "raw_query"
    assert semantic["source_packet_model_assisted"] is False
    assert semantic["gray_zone_decision_authority"] is False


def test_semantic_entry_preserves_domain_over_longer_generic_procedure_and_replays_history():
    from app.services import context_graph as graph
    query = "Compare the summary and details of a synthetic photon survey."
    facets = graph.query_facets_for_search(query, {"facet_groups": [
        {"facet": "photon survey", "role": "domain", "aliases": []},
        {"facet": "summary and detailed section consistency comparison", "role": "procedure", "aliases": []},
        {"facet": "area duration and pointing count", "role": "constraint", "aliases": []},
    ], "answer_shape": "comparison", "drop_terms": []})
    current = graph.semantic_entry_query_for_search(query, facets)
    assert current["query"] == "photon survey" and current["selected_required_facet_index"] == 0
    old = graph.semantic_entry_query_for_search(query, facets, _replay_protocol="validated_query_facet_semantic_entry_v1")
    assert old["selected_required_facet_index"] == 1
    assert graph.validate_semantic_entry_query_trace_packet(old, query=query, query_facets=facets) == old
    assert current["packet_hash"] != old["packet_hash"]


def test_public_structure_layout_keeps_parser_formula_flag():
    from app.schemas import ContextStructureLayoutAudit
    assert ContextStructureLayoutAudit.model_validate({"has_formula": True}).has_formula is True


def composite_facets():
    from app.services import context_graph as graph
    query = "List the business categories and counts in Aster platform scope."
    facets = graph.query_facets_for_search(query, {"facet_groups": [
        {"facet": "Aster platform scope", "role": "domain", "aliases": []},
        {"facet": "business category classification and category count", "role": "domain", "aliases": []},
        {"facet": "list only the requested numbers and names", "role": "procedure", "aliases": []},
    ], "answer_shape": "comparison", "drop_terms": []})
    return query, facets


def test_parallel_domain_preservation_and_versioned_history_replay():
    from app.services import context_graph as graph
    query, facets = composite_facets()
    before = graph.deterministic_gray_query_facets_for_search(query)
    current = graph.semantic_entry_query_for_search(query, facets)
    assert current["query"] == "Aster platform scope; business category classification and category count"
    assert current["selection_source"] == "validated_required_domain_composite"
    assert current["selected_required_facet_index"] is None
    assert current["selected_required_facet_indexes"] == [0, 1]
    assert graph.deterministic_gray_query_facets_for_search(query) == before
    for version in ("validated_query_facet_semantic_entry_v1", "validated_query_facet_semantic_entry_v2"):
        old = graph.semantic_entry_query_for_search(query, facets, _replay_protocol=version)
        assert old["query"] == "business category classification and category count"
        assert "selected_required_facet_indexes" not in old
        assert graph.validate_semantic_entry_query_trace_packet(old, query=query, query_facets=facets) == old
        assert old["packet_hash"] != current["packet_hash"]
    forged = {**current, "selected_required_facet_indexes": [1]}
    with pytest.raises(graph.EntrySelectionTraceInvariantError):
        graph.validate_semantic_entry_query_trace_packet(forged, query=query, query_facets=facets)


def test_specific_facet_may_cover_multiple_domains_without_repetition():
    from app.services import context_graph as graph
    query = "Compare the components of the Aster platform."
    facets = graph.query_facets_for_search(query, {"facet_groups": [
        {"facet": "Aster", "role": "domain", "aliases": []},
        {"facet": "Aster platform", "role": "domain", "aliases": []},
        {"facet": "Aster platform components", "role": "constraint", "aliases": []},
    ], "answer_shape": "comparison", "drop_terms": []})
    current = graph.semantic_entry_query_for_search(query, facets)
    assert current["query"] == "Aster platform components"
    assert current["selected_required_facet_indexes"] == [2]


@pytest.mark.asyncio
async def test_composite_domains_reach_embedding_and_public_dense_replay(monkeypatch, db_session, populated_context_graph, fake_model_stack):
    from app.schemas import SearchFilters
    from app.services import context_graph as graph
    from app.services.retrieval import get_retrieval_trace_steps
    query, facets = composite_facets()
    expected = graph.semantic_entry_query_for_search(query, facets)
    provider = fake_model_stack["EmbeddingProvider"]
    original = provider.embed_texts
    observed = []
    async def embed(self, texts, text_type="document"):
        if text_type == "query": observed.extend(texts)
        return await original(self, texts, text_type=text_type)
    monkeypatch.setattr(provider, "embed_texts", embed)
    result = await graph.layered_search(db_session, populated_context_graph["knowledge_base"].id, query, SearchFilters(), 3,
        query_facets=facets, allow_cache_read=False)
    assert observed == [expected["query"]]
    assert result.trace.diagnostics_json["semantic_entry_query"] == expected
    assert result.cache_components["semantic_entry_query_hash"] == expected["packet_hash"]
    assert get_retrieval_trace_steps(db_session, result.trace.id)
    assert result.trace.convergence_json["gray_zone_model_call_count"] == 0


def test_query_embedding_request_memo_is_strictly_bounded():
    from app.services import context_graph

    with pytest.raises(ValueError, match="max_entries"):
        context_graph.QueryEmbeddingRequestMemo(max_entries=5)

    memo = context_graph.QueryEmbeddingRequestMemo(max_entries=2)
    memo.put("first", [1.0])
    memo.put("second", [2.0])
    memo.put("third", [3.0])

    assert memo.get("first", expected_dimension=1) is None
    assert memo.get("second", expected_dimension=1) == [2.0]
    assert memo.get("third", expected_dimension=1) == [3.0]


@pytest.mark.asyncio
async def test_layered_search_embeds_semantic_entry_and_replays_raw_gray_identity(
    monkeypatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
):
    from app.schemas import QueryEmbeddingExecutionAudit, SearchFilters
    from app.services import context_graph
    from app.services.context_graph import EntrySelectionTraceInvariantError
    from app.services.retrieval import get_retrieval_trace_steps

    provider_type = fake_model_stack["EmbeddingProvider"]
    original_embed = provider_type.embed_texts
    captured_query_texts: list[str] = []

    async def capture_embed(self, texts, text_type="document"):
        if text_type == "query":
            captured_query_texts.extend(str(text) for text in texts)
        return await original_embed(self, texts, text_type=text_type)

    monkeypatch.setattr(provider_type, "embed_texts", capture_embed)
    facets = context_graph.query_facets_for_search(
        RAW_INSTRUCTION_QUERY,
        _llm_facet_payload(specific_facet="技术必须被什么夹在中间"),
    )
    knowledge_base = populated_context_graph["knowledge_base"]
    request_memo = context_graph.QueryEmbeddingRequestMemo()
    result = await context_graph.layered_search(
        db_session,
        knowledge_base.id,
        RAW_INSTRUCTION_QUERY,
        SearchFilters(),
        3,
        query_facets=facets,
        retrieval_granularity="mid",
        allow_cache_read=False,
        query_embedding_request_memo=request_memo,
    )

    second_result = await context_graph.layered_search(
        db_session,
        knowledge_base.id,
        RAW_INSTRUCTION_QUERY,
        SearchFilters(),
        3,
        query_facets=facets,
        retrieval_granularity="mid",
        allow_cache_read=False,
        query_embedding_request_memo=request_memo,
    )

    assert captured_query_texts == ["技术必须被什么夹在中间"]
    assert result.trace.id != second_result.trace.id
    first_embedding_audit = result.audit["query_embedding_execution"]
    second_embedding_audit = second_result.audit["query_embedding_execution"]
    assert first_embedding_audit == {
        "protocol_version": "request_scoped_query_embedding_memo_v1",
        "request_memo_enabled": True,
        "request_memo_hit": False,
        "request_memo_key_hash": second_embedding_audit[
            "request_memo_key_hash"
        ],
        "query_embedding_model_call_count": 1,
        "provider_response_present": False,
        "credentials_present": False,
        "gray_zone_decision_authority": False,
        "gray_zone_model_call_count": 0,
    }
    assert second_embedding_audit == {
        **first_embedding_audit,
        "request_memo_hit": True,
        "query_embedding_model_call_count": 0,
    }
    QueryEmbeddingExecutionAudit.model_validate(first_embedding_audit)
    QueryEmbeddingExecutionAudit.model_validate(second_embedding_audit)
    assert len(second_result.trace.frontier_json or []) > 0
    assert result.trace.query == RAW_INSTRUCTION_QUERY
    semantic = result.trace.diagnostics_json["semantic_entry_query"]
    assert semantic["query"] == "技术必须被什么夹在中间"
    assert semantic["gray_zone_decision_authority"] is False
    assert result.trace.diagnostics_json[
        "gray_zone_query_facet_hash"
    ] == context_graph.stable_hash(
        context_graph.deterministic_gray_query_facets_for_search(
            RAW_INSTRUCTION_QUERY
        )
    )
    assert result.trace.diagnostics_json[
        "gray_zone_external_routing_packet_used"
    ] is False
    assert result.trace.convergence_json["gray_zone_model_call_count"] == 0
    assert result.cache_components["query"] == RAW_INSTRUCTION_QUERY
    assert result.cache_components[
        "semantic_entry_query"
    ] == "技术必须被什么夹在中间"
    dense_packet = result.trace.diagnostics_json["entry_dense_replay_input"]
    assert dense_packet["protocol_version"] == "entry_dense_db_vector_replay_v2"
    assert dense_packet["semantic_entry_query_packet_hash"] == semantic[
        "packet_hash"
    ]
    assert get_retrieval_trace_steps(db_session, result.trace.id) is not None

    original_diagnostics = deepcopy(result.trace.diagnostics_json)
    forged_diagnostics = deepcopy(original_diagnostics)
    forged_diagnostics["semantic_entry_query"]["query"] = "forged entry"
    result.trace.diagnostics_json = forged_diagnostics
    db_session.flush()
    with pytest.raises(
        EntrySelectionTraceInvariantError,
        match="semantic entry query",
    ):
        get_retrieval_trace_steps(db_session, result.trace.id)

    result.trace.diagnostics_json = original_diagnostics
    db_session.flush()
    assert get_retrieval_trace_steps(db_session, result.trace.id) is not None
