from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _load_compliance_module():
    scripts_path = str(SCRIPTS_ROOT)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "test_check_technical_spec_compliance_reward",
        SCRIPTS_ROOT / "check_technical_spec_compliance.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent_run(db_session, knowledge_base, *, suffix: str = ""):
    from app.models import AgentRun

    run = AgentRun(
        knowledge_base_id=knowledge_base.id,
        question=f"grounded question {suffix}",
        status="completed",
        route="layered_context_graph",
    )
    db_session.add(run)
    db_session.flush()
    return run


def _reward(
    db_session,
    knowledge_base,
    *,
    context: dict[str, Any],
    eligible: Any,
    ineligible_reason: Any,
    source: Any = "context_graph_agent_v1",
    include_eligible: bool = True,
    include_reason: bool = True,
):
    from app.models import RewardEvent

    diagnostics: dict[str, Any] = {"source": source}
    if include_eligible:
        diagnostics["policy_reward_training_eligible"] = eligible
    if include_reason:
        diagnostics["policy_reward_training_ineligible_reason"] = (
            ineligible_reason
        )
    reward = RewardEvent(
        knowledge_base_id=knowledge_base.id,
        context_json=context,
        action_json={},
        reward_json={},
        diagnostics_json=diagnostics,
    )
    db_session.add(reward)
    db_session.flush()
    return reward


def _consuming_policy_state(db_session, knowledge_base, reward):
    from app.models import PolicyState

    state = PolicyState(
        knowledge_base_id=knowledge_base.id,
        policy_family="context_graph_bandit",
        policy_version="context_graph_bandit_v1",
        weights_json={},
        constraints_json={},
        exploration_json={},
        reward_summary_json={
            "origin": "reward_update",
            "last_reward_event_id": reward.id,
        },
        state_hash="a" * 64,
    )
    db_session.add(state)
    db_session.flush()
    reward.policy_state_id = state.id
    db_session.flush()
    return state


def _codes(issues: list[dict[str, Any]]) -> set[str]:
    return {str(issue["code"]) for issue in issues}


@pytest.mark.parametrize("include_null_field", [False, True])
def test_audit_only_reward_does_not_require_policy_update(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
    include_null_field: bool,
) -> None:
    from app.services import policy_reward

    replay_calls: list[str] = []
    monkeypatch.setattr(
        policy_reward,
        "replay_policy_reward_event",
        lambda _db, row: replay_calls.append(str(row.id)),
    )
    context = {"context_package_id": "audit-package"}
    if include_null_field:
        context["agent_run_id"] = None
    reward = _reward(
        db_session,
        sample_knowledge_base,
        context=context,
        eligible=False,
        ineligible_reason="missing_agent_run",
    )
    module = _load_compliance_module()
    issues: list[dict[str, Any]] = []

    summary = module._check_policy_reward_consumption(
        db_session,
        sample_knowledge_base.id,
        issues,
    )

    assert not issues
    assert replay_calls == []
    assert summary["reward_event_count"] == 1
    assert summary["audit_only_reward_count"] == 1
    assert summary["audit_only_reward_event_ids"] == [reward.id]
    assert summary["training_eligible_reward_count"] == 0
    assert summary["unconsumed_training_reward_count"] == 0


def test_forged_or_contradictory_eligibility_is_a_corruption_blocker(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import policy_reward

    replay_calls: list[str] = []
    monkeypatch.setattr(
        policy_reward,
        "replay_policy_reward_event",
        lambda _db, row: replay_calls.append(str(row.id)),
    )
    valid_run = _agent_run(db_session, sample_knowledge_base)
    forged_without_run = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": None},
        eligible=True,
        ineligible_reason=None,
    )
    contradictory_with_run = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": valid_run.id},
        eligible=False,
        ineligible_reason="missing_agent_run",
    )
    dangling_run = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": "missing-agent-run"},
        eligible=True,
        ineligible_reason=None,
    )
    malformed_run = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": ""},
        eligible=True,
        ineligible_reason=None,
    )
    missing_diagnostics = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": None},
        eligible=False,
        ineligible_reason="missing_agent_run",
        include_eligible=False,
    )
    audit_bound_to_policy = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": None},
        eligible=False,
        ineligible_reason="missing_agent_run",
    )
    _consuming_policy_state(
        db_session,
        sample_knowledge_base,
        audit_bound_to_policy,
    )
    module = _load_compliance_module()
    issues: list[dict[str, Any]] = []

    summary = module._check_policy_reward_consumption(
        db_session,
        sample_knowledge_base.id,
        issues,
    )

    assert replay_calls == []
    assert _codes(issues) == {
        "policy_reward_eligibility_or_binding_corrupt"
    }
    assert summary["corrupt_reward_count"] == 6
    assert summary["training_eligible_reward_count"] == 0
    assert summary["audit_only_reward_count"] == 0
    detail_ids = {
        row["reward_event_id"]
        for row in issues[0]["evidence"]["records"]
    }
    assert detail_ids == {
        forged_without_run.id,
        contradictory_with_run.id,
        dangling_run.id,
        malformed_run.id,
        missing_diagnostics.id,
        audit_bound_to_policy.id,
    }
    assert "policy_not_updated_from_reward" not in _codes(issues)


def test_only_replay_valid_agent_reward_requires_policy_consumption(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import policy_reward

    run = _agent_run(db_session, sample_knowledge_base)
    reward = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": run.id},
        eligible=True,
        ineligible_reason=None,
    )
    replay_calls: list[str] = []

    def replay(_db, row):
        replay_calls.append(str(row.id))
        return {"validated": True}

    monkeypatch.setattr(policy_reward, "replay_policy_reward_event", replay)
    module = _load_compliance_module()
    issues: list[dict[str, Any]] = []

    summary = module._check_policy_reward_consumption(
        db_session,
        sample_knowledge_base.id,
        issues,
    )

    assert replay_calls == [reward.id]
    assert _codes(issues) == {"policy_not_updated_from_reward"}
    assert summary["training_eligible_reward_count"] == 1
    assert summary["unconsumed_training_reward_count"] == 1
    assert summary["corrupt_reward_count"] == 0
    assert issues[0]["evidence"]["reward_event_ids"] == [reward.id]


def test_consumed_agent_reward_requires_exact_reciprocal_policy_binding(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import policy_reward

    run = _agent_run(db_session, sample_knowledge_base)
    reward = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": run.id},
        eligible=True,
        ineligible_reason=None,
    )
    state = _consuming_policy_state(
        db_session,
        sample_knowledge_base,
        reward,
    )
    monkeypatch.setattr(
        policy_reward,
        "replay_policy_reward_event",
        lambda _db, _row: {"validated": True},
    )
    module = _load_compliance_module()
    issues: list[dict[str, Any]] = []

    summary = module._check_policy_reward_consumption(
        db_session,
        sample_knowledge_base.id,
        issues,
    )

    assert not issues
    assert summary["training_eligible_reward_count"] == 1
    assert summary["consumed_training_reward_count"] == 1
    assert summary["consumed_training_reward_event_ids"] == [reward.id]

    state.reward_summary_json = {
        "origin": "reward_update",
        "last_reward_event_id": "another-reward",
    }
    db_session.flush()
    issues = []
    summary = module._check_policy_reward_consumption(
        db_session,
        sample_knowledge_base.id,
        issues,
    )
    assert _codes(issues) == {
        "policy_reward_eligibility_or_binding_corrupt"
    }
    assert summary["corrupt_reward_count"] == 1
    assert summary["consumed_training_reward_count"] == 0


def test_agent_reward_replay_failure_cannot_be_counted_as_trainable(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import policy_reward

    run = _agent_run(db_session, sample_knowledge_base)
    reward = _reward(
        db_session,
        sample_knowledge_base,
        context={"agent_run_id": run.id},
        eligible=True,
        ineligible_reason=None,
    )

    def reject_replay(_db, _row):
        raise policy_reward.PolicyRewardReplayError("forged replay packet")

    monkeypatch.setattr(
        policy_reward,
        "replay_policy_reward_event",
        reject_replay,
    )
    module = _load_compliance_module()
    issues: list[dict[str, Any]] = []

    summary = module._check_policy_reward_consumption(
        db_session,
        sample_knowledge_base.id,
        issues,
    )

    assert _codes(issues) == {
        "policy_reward_eligibility_or_binding_corrupt"
    }
    assert summary["training_eligible_reward_count"] == 0
    assert summary["corrupt_reward_count"] == 1
    record = issues[0]["evidence"]["records"][0]
    assert record["reward_event_id"] == reward.id
    assert record["error_type"] == "PolicyRewardReplayError"
    assert "forged replay packet" in record["error"]
    assert "policy_not_updated_from_reward" not in _codes(issues)
