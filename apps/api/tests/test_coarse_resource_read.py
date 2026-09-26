"""Public synthetic contracts for optional coarse navigation."""
import json

import pytest
from sqlalchemy import select

from app.models import (
    AgentObservation, AgentTraceEvent, Chunk, CoarseConcept, CoarseConceptState,
    ContextGraphState, Document, DocumentVersion,
)
from app.retrieval_control_contracts import control_hash
from app.schemas import AgentRequest, AgentTraceEventPayload, SearchFilters
from app.services import agent_graph
from app.services.coarse_resource_read import (
    read_coarse_details, read_coarse_titles,
)
from app.services.intent_planning import plan_intent_execution, retrieval_capability_snapshot


@pytest.fixture
def coarse_catalog(db_session, sample_knowledge_base):
    kb = sample_knowledge_base
    coarse_state = CoarseConceptState(
        knowledge_base_id=kb.id,
        state_hash="a" * 64,
        grounding_hash="b" * 64,
    )
    db_session.add(coarse_state)
    db_session.flush()
    graph = ContextGraphState(
        knowledge_base_id=kb.id,
        coarse_concept_state_id=coarse_state.id,
        chunk_scope_hash="c" * 64,
        structure_graph_hash="d" * 64,
        chunk_relation_graph_hash="e" * 64,
        rq_membership_hash="f" * 64,
        mid_concept_hash="1" * 64,
        coarse_concept_hash="2" * 64,
        context_graph_hash="3" * 64,
    )
    db_session.add(graph)
    concepts = [
        CoarseConcept(
            knowledge_base_id=kb.id, coarse_state_id=coarse_state.id,
            canonical_label=label, node_weight=weight,
            summary=summary, definition=summary,
            grounding_hash=f"{index}" * 64,
        )
        for index, (label, weight, summary) in enumerate([
            ("Ecosystem dynamics", 0.4, "Relations among species."),
            ("水文循环", 0.9, "降水和蒸发。"),
            ("Energy systems", 0.9, "Storage and conversion."),
        ], 1)
    ]
    db_session.add_all(concepts)
    db_session.commit()
    return kb, graph, concepts


def test_complete_titles_weight_order_details_and_key_gate(db_session, coarse_catalog):
    kb, graph, concepts = coarse_catalog
    titles, keys, audit = read_coarse_titles(
        db_session, knowledge_base_id=kb.id,
        graph_identity=graph.context_graph_hash, filters=SearchFilters(),
    )
    assert titles["complete"] is True
    assert len(titles["nodes"]) == len(concepts)
    assert {node["title"] for node in titles["nodes"][:2]} == {"水文循环", "Energy systems"}
    assert titles["nodes"][2]["title"] == "Ecosystem dynamics"
    assert [node["key"] for node in titles["nodes"]] == ["c1", "c2", "c3"]
    assert all(set(node) == {"key", "title"} for node in titles["nodes"])
    assert "summary" not in titles["nodes"][0]
    assert audit["node_count"] == 3 and audit["local_duration_ms"] >= 0
    details, detail_audit = read_coarse_details(
        db_session, knowledge_base_id=kb.id,
        graph_identity=graph.context_graph_hash, keys=("c3", "c1"), key_to_id=keys,
    )
    assert [item["title"] for item in details["nodes"]] == ["Ecosystem dynamics", titles["nodes"][0]["title"]]
    assert details["nodes"][0]["summary"]["text"] == "Relations among species."
    assert detail_audit["keys"] == ["c3", "c1"]
    with pytest.raises(ValueError, match="outside_directory"):
        read_coarse_details(db_session, knowledge_base_id=kb.id, graph_identity=graph.context_graph_hash, keys=("c99",), key_to_id=keys)


def test_titles_respect_document_scope_and_report_over_budget(db_session, coarse_catalog, monkeypatch):
    from app.services import coarse_resource_read

    kb, graph, concepts = coarse_catalog
    for index, concept in enumerate(concepts):
        document = Document(
            knowledge_base_id=kb.id, title=f"Synthetic {index}",
            source_path=f"/unit-test/{index}.md", source_type="markdown" if index == 0 else "pdf",
            tags=[f"topic-{index}"], checksum=f"{index}" * 64,
        )
        db_session.add(document)
        db_session.flush()
        version = DocumentVersion(
            document_id=document.id, version=1, checksum=f"{index}" * 64,
            storage_path=f"/unit-test/{index}.md",
        )
        db_session.add(version)
        db_session.flush()
        chunk = Chunk(
            knowledge_base_id=kb.id, document_id=document.id,
            document_version_id=version.id, chunk_version=1,
            chunk_index=0, text=f"Synthetic topic {index}", text_hash=f"{index}" * 64,
            page_start=index + 1, page_end=index + 1,
            metadata_json={"content_kind": "paragraph" if index == 0 else "table", "partition": f"p{index}"},
        )
        db_session.add(chunk)
        db_session.flush()
        concept.support_chunk_ids_json = [chunk.id]
        if index == 0:
            selected_document_id = document.id
        if index == 1:
            outside_chunk_id = chunk.id
    db_session.commit()
    titles, _, _ = read_coarse_titles(
        db_session, knowledge_base_id=kb.id,
        graph_identity=graph.context_graph_hash,
        filters=SearchFilters(document_ids=[selected_document_id]),
    )
    assert [node["title"] for node in titles["nodes"]] == [concepts[0].canonical_label]
    for scoped_filters in (
        SearchFilters(source_paths=["/unit-test/0.md"]),
        SearchFilters(source_type="markdown"),
        SearchFilters(tags=["topic-0"]),
        SearchFilters(page_range=(1, 1)),
        SearchFilters(content_kinds=["paragraph"]),
        SearchFilters(partition="p0"),
    ):
        scoped, _, _ = read_coarse_titles(
            db_session, knowledge_base_id=kb.id,
            graph_identity=graph.context_graph_hash, filters=scoped_filters,
        )
        assert [node["title"] for node in scoped["nodes"]] == [concepts[0].canonical_label]
    concepts[0].support_chunk_ids_json = [*concepts[0].support_chunk_ids_json, outside_chunk_id]
    db_session.commit()
    partial, _, _ = read_coarse_titles(
        db_session, knowledge_base_id=kb.id,
        graph_identity=graph.context_graph_hash,
        filters=SearchFilters(document_ids=[selected_document_id]),
    )
    assert partial["nodes"] == []
    monkeypatch.setattr(coarse_resource_read, "MAX_DIRECTORY_CHARACTERS", 10)
    with pytest.raises(ValueError, match="directory_over_budget"):
        read_coarse_titles(db_session, knowledge_base_id=kb.id, graph_identity=graph.context_graph_hash, filters=SearchFilters())


@pytest.mark.asyncio
async def test_model_controls_titles_details_then_plan_with_audited_count(db_session, coarse_catalog):
    from test_intent_contracts import proposal

    kb, graph, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="Explain the topic relationships.")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)
    seen = []

    class Provider:
        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            packet = json.loads(user_prompt)
            navigation = packet["navigation"]
            seen.append(navigation)
            if len(seen) == 1:
                assert navigation["state"] == "initial"
                return {"action": "resource_read", "mode": "titles"}
            if len(seen) == 2:
                assert navigation["state"] == "titles"
                assert len(navigation["observations"][0]["nodes"]) == 3
                return {"action": "resource_read", "mode": "details", "keys": ["c1", "c3"]}
            assert navigation["state"] == "details"
            assert len(navigation["observations"][1]["nodes"]) == 2
            return proposal(layer="chunk")

    plan, audit = await plan_intent_execution(
        db_session, run=run, question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
        history_summary="", capabilities=capabilities, filters=request.filters,
        provider_factory=Provider,
        on_trace=lambda node, kwargs: agent_graph.trace(db_session, run.id, node, **kwargs),
    )
    assert plan.strategy.entry_layer == "chunk"
    assert audit["model_call_count"] == 3
    assert audit["resource_read_count"] == 2
    assert [item["action"] for item in audit["steps"]] == ["resource_read", "resource_read", "plan"]
    events = list(db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.sequence_index)))
    assert [(event.sequence_index, event.node) for event in events] == [
        (0, "planning_resource_titles"), (1, "planning_resource_details"),
    ]
    assert [AgentTraceEventPayload.model_validate(agent_graph.trace_event_to_payload(event)).scores.coarse_node_count for event in events] == [3, 2]
    assert all(not event.document_ids for event in events)
    assert "Relations among species" not in json.dumps([agent_graph.trace_event_to_payload(event) for event in events], default=str)
    assert db_session.scalar(select(AgentObservation).where(AgentObservation.run_id == run.id)).verdict == "completed"


@pytest.mark.asyncio
async def test_production_shape_keeps_continuous_resource_tool_history(db_session, coarse_catalog):
    from test_intent_contracts import proposal

    kb, _, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="Explain the topic relationships.")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)
    seen_messages = []

    class Provider:
        async def classify_json_messages(
            self,
            *,
            system_prompt,
            messages,
            fallback,
            max_tokens,
            compatibility_user_prompt,
            response_schema,
            native_tools,
        ):
            seen_messages.append(messages)
            initial = json.loads(messages[0]["content"])
            assert "knowledge_base_id" not in initial["capabilities"]
            assert "graph_identity" not in initial["capabilities"]
            assert "filter_scope_hash" not in initial
            assert response_schema["additionalProperties"] is False
            assert native_tools[0]["name"] == "planning_action"
            assert native_tools[0]["passthrough"] is True
            if len(messages) == 1:
                return {
                    "protocol_version": "planning_tool_call_v1",
                    "tool": "resource.read",
                    "arguments": {"mode": "titles", "keys": []},
                }
            result = json.loads(messages[-1]["content"])
            assert result["tool"] == "resource.read"
            assert all(set(node) == {"key", "title"} for node in result["observation"]["nodes"])
            return {
                "protocol_version": "planning_tool_call_v1",
                "tool": "plan.commit",
                "arguments": proposal(layer="chunk"),
            }

    plan, audit = await plan_intent_execution(
        db_session,
        run=run,
        question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
        history_summary="",
        capabilities=capabilities,
        filters=request.filters,
        provider_factory=Provider,
    )
    assert plan.strategy.entry_layer == "chunk"
    assert [len(messages) for messages in seen_messages] == [1, 3]
    assert audit["steps"][0]["continuous_messages"] is True
    assert audit["steps"][1]["context_plan"]["atomic_groups_preserved"] is True


@pytest.mark.asyncio
async def test_live_stream_replays_preplan_read_before_final_plan(
    db_session, coarse_catalog, fake_model_stack, monkeypatch,
):
    from agent_test_support import Admission

    kb, _, _ = coarse_catalog

    class Provider(fake_model_stack["ChatProvider"]):
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            state = json.loads(user_prompt)["navigation"]["state"]
            if state == "initial":
                return {"action": "resource_read", "mode": "titles"}
            return {
                "intent": {"primary": "clarify"},
                "execution_strategy": {
                    "route": "clarify", "entry_layer": None,
                    "generate_lexical": False, "hybrid": False,
                    "reason_code": "ambiguous_request",
                },
            }

    monkeypatch.setattr(agent_graph, "ChatProvider", Provider)
    events = [event async for event in agent_graph.stream_agent_events(
        AgentRequest(knowledge_base_id=kb.id, question="Which topic?", stream_trace=True),
        admission=Admission(),
    )]
    assert not [event for event in events if event["type"] == "error"]
    trace_nodes = [event["trace"]["node"] for event in events if event["type"] == "trace"]
    assert trace_nodes == ["planning_resource_titles", "intent_planning"]
    run_id = next(event["run_id"] for event in events if event["type"] == "meta")
    persisted = list(db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run_id).order_by(AgentTraceEvent.sequence_index)))
    assert [event.node for event in persisted] == trace_nodes
    from app.routers.sessions import get_session_messages

    session_id = next(event["session_id"] for event in events if event["type"] == "meta")
    history = get_session_messages(session_id, db_session)
    assistant = next(item for item in history["messages"] if item["role"] == "assistant")
    assert [event["node"] for event in assistant["trace"]] == trace_nodes


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_action", [
    {"action": "resource_read", "mode": "details", "keys": ["c1"]},
    {"action": "resource_read", "mode": "titles", "keys": ["c1"]},
    {"action": "resource_read", "mode": "titles", "unknown": True},
])
async def test_repeated_invalid_read_exhausts_bounded_planning_calls(db_session, coarse_catalog, bad_action):
    kb, _, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="What is covered?")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)

    class Provider:
        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            return bad_action

    with pytest.raises(ValueError):
        await plan_intent_execution(
            db_session, run=run, question=request.question,
            conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
            filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
            history_summary="", capabilities=capabilities,
            provider_factory=Provider,
        )
    observation = db_session.scalar(select(AgentObservation).where(AgentObservation.run_id == run.id))
    assert observation.verdict == "failed"
    assert observation.observation_json["model_call_count"] == 4
    assert all(
        item["action"] == "resource_read_invalid"
        for item in observation.observation_json["steps"]
    )


@pytest.mark.asyncio
async def test_invalid_resource_tool_arguments_are_repaired_in_same_history(db_session, coarse_catalog):
    from test_intent_contracts import proposal

    kb, _, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="Explain the topics.")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)
    seen = []

    class Provider:
        async def classify_json_messages(self, **kwargs):
            messages = kwargs["messages"]
            seen.append(messages)
            if len(seen) == 1:
                return {
                    "tool": "resource.read",
                    "arguments": proposal(layer="chunk"),
                }
            result = json.loads(messages[-1]["content"])
            assert result == {
                "protocol_version": "planning_tool_result_v1",
                "tool": "resource.read",
                "status": "error",
                "error": "resource_read_arguments_invalid",
                "field": "arguments",
            }
            return {"tool": "plan.commit", "arguments": proposal(layer="chunk")}

    plan, audit = await plan_intent_execution(
        db_session,
        run=run,
        question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
        history_summary="",
        capabilities=capabilities,
        provider_factory=Provider,
    )
    assert plan.strategy.entry_layer == "chunk"
    assert [len(messages) for messages in seen] == [1, 3]
    assert [item["action"] for item in audit["steps"]] == [
        "resource_read_invalid",
        "plan",
    ]


@pytest.mark.asyncio
async def test_malformed_provider_tool_output_gets_one_safe_retry(db_session, coarse_catalog):
    from app.services.embeddings import ProviderJSONShapeError
    from test_intent_contracts import proposal

    kb, _, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="Explain the topics.")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)
    seen = []

    class Provider:
        async def classify_json_messages(self, **kwargs):
            messages = kwargs["messages"]
            seen.append(messages)
            if len(seen) == 1:
                raise ProviderJSONShapeError(
                    {"error_code": "invalid_object_key", "field_path": "$"}
                )
            result = json.loads(messages[-1]["content"])
            assert result["validation_feedback"]["raw_response_included"] is False
            assert result["validation_feedback"]["errors"] == [
                {"path": "$", "code": "provider_json_shape"}
            ]
            return {"tool": "plan.commit", "arguments": proposal(layer="chunk")}

    plan, audit = await plan_intent_execution(
        db_session,
        run=run,
        question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
        history_summary="",
        capabilities=capabilities,
        provider_factory=Provider,
    )
    assert plan.strategy.entry_layer == "chunk"
    assert [len(messages) for messages in seen] == [1, 3]
    assert [item["action"] for item in audit["steps"]] == [
        "planning_output_invalid",
        "plan",
    ]


@pytest.mark.asyncio
async def test_graph_change_after_titles_rejects_plan(db_session, coarse_catalog):
    from test_intent_contracts import proposal

    kb, graph, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="Explain the topics.")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)
    calls = 0

    class Provider:
        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"action": "resource_read", "mode": "titles"}
            graph.context_graph_hash = "4" * 64
            db_session.commit()
            return proposal(layer="chunk")

    with pytest.raises(ValueError, match="graph_identity_changed"):
        await plan_intent_execution(
            db_session, run=run, question=request.question,
            conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
            filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
            history_summary="", capabilities=capabilities, filters=request.filters,
            provider_factory=Provider,
        )
    observation = db_session.scalar(select(AgentObservation).where(AgentObservation.run_id == run.id))
    assert observation.verdict == "failed"
    assert observation.observation_json["model_call_count"] == 2
    assert observation.observation_json["steps"][0]["mode"] == "titles"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_schema_feedback_allows_one_model_owned_full_plan_retry(
    db_session, coarse_catalog, repair_succeeds,
):
    from test_intent_contracts import proposal

    kb, _, _ = coarse_catalog
    request = AgentRequest(knowledge_base_id=kb.id, question="Explain the topics.")
    _, run = agent_graph.create_agent_run_context(db_session, request)
    capabilities, _ = retrieval_capability_snapshot(db_session, kb.id, admit_graph=False)
    invalid = proposal(layer="chunk", hybrid=True, lexical=True)
    invalid["execution_strategy"]["lexical_groups"][0]["kind"] = "identifier"
    invalid["execution_strategy"]["lexical_groups"][0]["surfaces"][0] = {
        "text": "unit-test-key", "language": "en", "provenance": "model_query",
    }
    calls = []

    class Provider:
        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            packet = json.loads(user_prompt)
            calls.append(packet)
            if len(calls) == 1:
                return invalid
            feedback = packet["validation_feedback"]
            assert packet["navigation"]["allowed_actions"] == ["plan"]
            assert feedback["raw_response_included"] is False
            assert feedback["errors"][0]["code"] == "strategy_identifier_surface_must_be_neutral"
            assert "unit-test-key" not in user_prompt
            return proposal(layer="chunk") if repair_succeeds else invalid

    kwargs = dict(
        run=run, question=request.question,
        conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        filter_scope_hash=control_hash(request.filters.model_dump(mode="json")),
        history_summary="", capabilities=capabilities, filters=request.filters,
        provider_factory=Provider,
        on_trace=lambda node, trace_kwargs: agent_graph.trace(db_session, run.id, node, **trace_kwargs),
    )
    if repair_succeeds:
        plan, audit = await plan_intent_execution(db_session, **kwargs)
        assert plan.strategy.entry_layer == "chunk"
        assert audit["model_call_count"] == 2
        assert audit["schema_repair_count"] == 1
        assert [step["action"] for step in audit["steps"]] == ["plan_schema_invalid", "plan"]
    else:
        with pytest.raises(ValueError):
            await plan_intent_execution(db_session, **kwargs)
        observation = db_session.scalar(select(AgentObservation).where(AgentObservation.run_id == run.id))
        assert observation.verdict == "failed"
        assert observation.observation_json["model_call_count"] == 2
    assert len(calls) == 2
    feedback_events = list(db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id)))
    assert [event.node for event in feedback_events] == ["planning_schema_feedback"]
    assert AgentTraceEventPayload.model_validate(agent_graph.trace_event_to_payload(feedback_events[0])).scores.schema_feedback_error_count == 1
