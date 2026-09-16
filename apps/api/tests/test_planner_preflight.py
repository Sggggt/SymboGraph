import ast
from copy import deepcopy

import pytest

from app.services import agent_graph as ag
from app.services.context_graph import validate_typed_action_traversal_controls
from app.services.error_sanitizer import ExternalServiceError


def proposal(envelope, chunk_ids):
    actions = ag.fallback_typed_actions("Synthetic topic", envelope)
    for action in actions:
        if action["action_type"] == "recall_chunks":
            action["target_ids"] = chunk_ids[:1]
        elif action["action_type"] == "restore_context_package":
            action["target_ids"] = chunk_ids[1:]
    return {"typed_actions": actions}


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["valid", "invalid", "unavailable"])
async def test_cross_action_target_union_uses_one_existing_repair(monkeypatch, db_session, populated_context_graph, repair):
    envelope = {**ag.agent_operating_envelope(), "agent_chunk_initial_budget": 1}
    original = deepcopy(envelope)
    ids = [chunk.id for chunk in populated_context_graph["chunks"][:2]]
    assert len(ids) == 2
    invalid = proposal(envelope, ids)
    accepted, validation = ag.validate_typed_actions(invalid["typed_actions"], envelope, db=db_session,
        knowledge_base_id=populated_context_graph["knowledge_base"].id, retrieval_granularity="mid")
    assert validation["valid"]  # Each action is valid; the combined executable domain is not.
    controls = ag.compile_typed_action_execution_controls(accepted, envelope, requested_result_top_k=4,
        retrieval_granularity="mid", validation_diagnostics=validation)
    with pytest.raises(ValueError, match="layer budget"):
        validate_typed_action_traversal_controls(controls, base_envelope=envelope, retrieval_granularity="mid", result_top_k=4)
    prompts = []

    class Provider:
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            prompts.append(ast.literal_eval(user_prompt))
            if len(prompts) == 1 or repair == "invalid":
                return deepcopy(invalid)
            if repair == "unavailable":
                raise ExternalServiceError(service="model_provider", phase="sdk_messages_completion",
                    error_code="incomplete_max_tokens", retryable=False)
            return proposal(envelope, ids[:1])

    monkeypatch.setattr(ag, "ChatProvider", Provider)
    kwargs = dict(validation_db=db_session, policy_knowledge_base_id=populated_context_graph["knowledge_base"].id,
        requested_result_top_k=4)
    if repair == "valid":
        actions, output = await ag.propose_agent_plan("Synthetic topic", [], {"intent": "definition"}, envelope, **kwargs)
        assert output["planner_sampling"] == ag.planner_sampling_audit(model_call_count=2, preflight_rejection_count=1)
        assert {target for action in actions for target in action["target_ids"]} == {ids[0]}
        assert not ag.planner_execution_preflight(actions, envelope, db=db_session,
            knowledge_base_id=kwargs["policy_knowledge_base_id"], retrieval_granularity="mid", requested_result_top_k=4)
    elif repair == "invalid":
        with pytest.raises(ag.TypedActionValidationError) as failure:
            await ag.propose_agent_plan("Synthetic topic", [], {"intent": "definition"}, envelope, **kwargs)
        assert failure.value.diagnostics["planner_sampling"] == ag.planner_sampling_audit(model_call_count=2, preflight_rejection_count=2)
    else:
        with pytest.raises(ExternalServiceError, match="incomplete_max_tokens") as failure:
            await ag.propose_agent_plan("Synthetic topic", [], {"intent": "definition"}, envelope, **kwargs)
        assert failure.value.planner_sampling["model_call_count"] == 2
    assert len(prompts) == 2
    assert prompts[0]["typed_action_output_contract"]["output_size_contract"]["distinct_target_limits_by_layer"]["chunk"] == 1
    assert prompts[1]["execution_preflight"]["distinct_target_counts"]["chunk"] == 2
    assert prompts[1]["execution_preflight"]["distinct_target_limits"]["chunk"] == 1
    assert envelope == original


@pytest.mark.asyncio
async def test_repeated_targets_count_once_across_actions(db_session, populated_context_graph):
    envelope = {**ag.agent_operating_envelope(), "agent_chunk_initial_budget": 1}
    chunk_id = populated_context_graph["chunks"][0].id
    actions = proposal(envelope, [chunk_id, chunk_id])["typed_actions"]
    assert not ag.planner_execution_preflight(actions, envelope, db=db_session,
        knowledge_base_id=populated_context_graph["knowledge_base"].id, retrieval_granularity="mid", requested_result_top_k=4)
