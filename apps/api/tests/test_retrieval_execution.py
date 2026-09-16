import pytest

from app.services import agent_graph as ag
from app.services.context_graph import validate_typed_action_traversal_controls
from app.services.retrieval_execution import compile_retrieval_execution
from test_retrieval_path_features import strategy, task_fixture


def test_retrieval_only_plan_has_no_post_generation_action(no_fallback_env):
    task = task_fixture()
    lexical = strategy(task, ("waiting",))
    envelope = ag.agent_operating_envelope()
    actions, validation, controls = compile_retrieval_execution(None, task=task, strategy=lexical,
        envelope=envelope, top_k=4, lexical_policy_hash="e" * 64)
    assert validation["valid"] and validation["required_actions_enforced"]
    assert validation["typed_action_schema_protocol_version"] == "retrieval_typed_action_schema_v1"
    assert {action["action_type"] for action in actions} == {
        "select_entry_nodes", "walk_graph_frontier", "recall_chunks",
        "restore_context_package", "build_context_package",
    }
    assert controls["reflection_round_budget"] == 0
    assert controls["task_hash"] == task.identity and controls["lexical_strategy_hash"] == lexical.identity
    assert controls["gray_zone_model_call_count"] == 0
    assert controls["path_distance_thresholds_modified"] is False
    forged = {**controls, "reflection_round_budget": 1}
    forged["control_hash"] = ag.stable_hash({key: value for key, value in forged.items() if key != "control_hash"})
    with pytest.raises(ValueError, match="cannot_authorize_post_generation"):
        validate_typed_action_traversal_controls(forged, base_envelope=envelope,
            retrieval_granularity="mid", result_top_k=4)


def test_retrieval_only_validator_rejects_review_actions_and_historical_overrides(no_fallback_env):
    envelope = ag.agent_operating_envelope()
    proposal = ag.fallback_typed_actions("Synthetic question", envelope)
    _, validation = ag.validate_typed_actions(proposal, envelope, execution_scope="retrieval_only",
                                               retrieval_granularity="mid")
    assert not validation["valid"]
    assert any(item.get("action_type") == "review_answer" for item in validation["rejected"])
    with pytest.raises(ValueError, match="historical_action_overrides"):
        ag.validate_typed_actions([], envelope, execution_scope="retrieval_only",
            required_actions_override=("select_entry_nodes",))
