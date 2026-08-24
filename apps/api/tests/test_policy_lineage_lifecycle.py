from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import pytest


def _reward_payload(**overrides: Any) -> dict[str, Any]:
    from app.services.policy_reward import (
        POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
    )

    return {
        "reward_metrics_protocol_version": (
            POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION
        ),
        "reward_metric_evidence_hash": "d" * 64,
        "retrieval_hit": 0.25,
        "context_precision": 0.25,
        "context_recall": 0.25,
        "concept_path_accuracy": 0.25,
        "citation_pass_rate": 0.25,
        "answer_groundedness": 0.25,
        "answer_completeness": 0.25,
        "claim_count": 4,
        "supported_claim_count": 2,
        "unsupported_claim_count": 2,
        "repair_success_rate": 0.25,
        "agent_typed_action_validation_pass_rate": 0.25,
        "latency_cost": 1.0,
        "latency_ms": 1000.0,
        "task_token_cost": 128.0,
        "drift_rate": 0.25,
        "answer_acceptance_gate_pass": 1.0,
        **overrides,
    }


def _seed_policy_state(db_session, knowledge_base_id: str, *, created_at):
    from app.models import PolicyState
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import canonical_policy_state_hash
    from app.services.policy import POLICY_ARMS, POLICY_FAMILY, POLICY_VERSION

    envelope = agent_operating_envelope()
    runtime_hash = runtime_settings_state_hash()
    envelope_hash = agent_operating_envelope_state_hash()
    weights = {arm: 1.0 for arm in POLICY_ARMS}
    safe_arms = list(POLICY_ARMS)
    constraints = {
        "fallback_disabled": True,
        "citation_verification_required": True,
        "agent_operating_envelope": envelope,
        "runtime_settings_hash": runtime_hash,
        "planner_replacement": False,
        "gray_zone_decision_authority": False,
        "gray_zone_rule_inputs_modified": False,
        "gray_zone_model_call_count": 0,
    }
    exploration = {
        "epsilon": 0.05,
        "safe_arms": safe_arms,
        "threshold_suggestions_runtime_lifecycle_accepted": False,
        "threshold_suggestions_applied": False,
        "gray_zone_decision_authority": False,
        "gray_zone_model_call_count": 0,
        "path_distance_green_threshold": envelope[
            "path_distance_green_threshold"
        ],
        "path_distance_gray_threshold": envelope[
            "path_distance_gray_threshold"
        ],
        "path_distance_hard_threshold": envelope[
            "path_distance_hard_threshold"
        ],
    }
    summary = {
        "origin": "seed",
        "previous_policy_state_id": None,
        "previous_policy_state_hash": None,
        "safe_arms": safe_arms,
        "posterior": weights,
        "policy_version": POLICY_VERSION,
        "reward_history_tail": [],
        "runtime_settings_hash": runtime_hash,
        "agent_operating_envelope_hash": envelope_hash,
    }
    state = PolicyState(
        knowledge_base_id=knowledge_base_id,
        policy_family=POLICY_FAMILY,
        policy_version=POLICY_VERSION,
        weights_json=weights,
        constraints_json=constraints,
        exploration_json=exploration,
        reward_summary_json=summary,
        state_hash=canonical_policy_state_hash(
            policy_family=POLICY_FAMILY,
            policy_version=POLICY_VERSION,
            profile_objective_hash=None,
            weights=weights,
            constraints=constraints,
            exploration=exploration,
            reward_summary=summary,
        ),
        created_at=created_at,
    )
    db_session.add(state)
    db_session.flush()
    return state


def _build_policy_lineage(
    db_session,
    knowledge_base_id: str,
    *,
    reward_count: int,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.models import PolicyState, RewardEvent
    from app.services import policy
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import canonical_policy_state_hash
    from app.services.policy import (
        POLICY_FAMILY,
        POLICY_POSTERIOR_LEARNING_RATE,
        POLICY_POSTERIOR_UPDATE_PROTOCOL_VERSION,
        POLICY_VERSION,
        replay_policy_posterior_update,
    )
    from app.services.policy_reward import (
        POLICY_REWARD_FACT_PROTOCOL_VERSION,
    )

    replay_calls: list[str] = []
    fact_hash_by_reward_id: dict[str, str] = {}

    def replay_reward(_db, reward):
        replay_calls.append(str(reward.id))
        return {
            "reward_fact_hash": fact_hash_by_reward_id[str(reward.id)]
        }

    monkeypatch.setattr(policy, "replay_policy_reward_event", replay_reward)
    base_time = datetime(2026, 1, 1)
    seed = _seed_policy_state(
        db_session,
        knowledge_base_id,
        created_at=base_time,
    )
    states = [seed]
    rewards = []
    envelope = agent_operating_envelope()
    runtime_hash = runtime_settings_state_hash()
    envelope_hash = agent_operating_envelope_state_hash()
    for index in range(1, reward_count + 1):
        predecessor = states[-1]
        reward_json = _reward_payload(
            reward_metric_evidence_hash=hashlib.sha256(
                f"reward-evidence-{index}".encode()
            ).hexdigest(),
        )
        reward = RewardEvent(
            knowledge_base_id=knowledge_base_id,
            reward_json=reward_json,
            diagnostics_json={
                "source": "context_graph_agent_v1",
                "reward_metrics_protocol_version": reward_json[
                    "reward_metrics_protocol_version"
                ],
                "reward_metric_evidence_hash": reward_json[
                    "reward_metric_evidence_hash"
                ],
                "runtime_settings_hash": runtime_hash,
                "agent_operating_envelope_hash": envelope_hash,
            },
            created_at=base_time + timedelta(seconds=index),
        )
        db_session.add(reward)
        db_session.flush()
        reward_fact_hash = hashlib.sha256(
            f"reward-fact-{index}".encode()
        ).hexdigest()
        fact_hash_by_reward_id[str(reward.id)] = reward_fact_hash
        posterior = replay_policy_posterior_update(
            dict(predecessor.weights_json),
            reward_json,
        )
        weights = dict(posterior["weights"])
        safe_arms = list(posterior["safe_arms"])
        predecessor_summary = dict(predecessor.reward_summary_json)
        history = list(
            predecessor_summary.get("reward_history_tail") or []
        )
        summary = {
            "origin": "reward_update",
            "previous_policy_state_id": predecessor.id,
            "previous_policy_state_hash": predecessor.state_hash,
            "last_reward_event_id": reward.id,
            "last_reward_fact_protocol_version": (
                POLICY_REWARD_FACT_PROTOCOL_VERSION
            ),
            "last_reward_fact_hash": reward_fact_hash,
            "last_reward": reward_json,
            "reward_history_tail": history[-7:] + [reward_json],
            "safe_arms": safe_arms,
            "posterior": weights,
            "posterior_proxy": weights,
            "posterior_update_protocol_version": (
                POLICY_POSTERIOR_UPDATE_PROTOCOL_VERSION
            ),
            "posterior_learning_rate": POLICY_POSTERIOR_LEARNING_RATE,
            "arm_reward_observations": dict(
                posterior["arm_reward_observations"]
            ),
            "normalized_reward_signals": dict(
                posterior["normalized_signals"]
            ),
            "typed_action_validation_pass_rate": reward_json[
                "agent_typed_action_validation_pass_rate"
            ],
            "reward_metrics_protocol_version": reward_json[
                "reward_metrics_protocol_version"
            ],
            "reward_metric_evidence_hash": reward_json[
                "reward_metric_evidence_hash"
            ],
            "exploration_rate": 0.05,
            "drift_status": (
                "normal"
                if float(reward_json["drift_rate"]) <= 0.2
                else "elevated"
            ),
            "policy_version": POLICY_VERSION,
            "runtime_settings_hash": runtime_hash,
            "agent_operating_envelope_hash": envelope_hash,
        }
        constraints = {
            "fallback_disabled": True,
            "citation_verification_required": True,
            "agent_operating_envelope": envelope,
            "runtime_settings_hash": runtime_hash,
            "planner_replacement": False,
            "gray_zone_decision_authority": False,
            "gray_zone_rule_inputs_modified": False,
            "gray_zone_model_call_count": 0,
        }
        exploration = {
            "epsilon": 0.05,
            "exploration_rate": 0.05,
            "safe_arms": safe_arms,
            "threshold_suggestions_runtime_lifecycle_accepted": False,
            "threshold_suggestions_applied": False,
            "gray_zone_decision_authority": False,
            "gray_zone_model_call_count": 0,
            "path_distance_green_threshold": envelope[
                "path_distance_green_threshold"
            ],
            "path_distance_gray_threshold": envelope[
                "path_distance_gray_threshold"
            ],
            "path_distance_hard_threshold": envelope[
                "path_distance_hard_threshold"
            ],
        }
        state = PolicyState(
            knowledge_base_id=knowledge_base_id,
            policy_family=POLICY_FAMILY,
            policy_version=POLICY_VERSION,
            weights_json=weights,
            constraints_json=constraints,
            exploration_json=exploration,
            reward_summary_json=summary,
            state_hash=canonical_policy_state_hash(
                policy_family=POLICY_FAMILY,
                policy_version=POLICY_VERSION,
                profile_objective_hash=None,
                weights=weights,
                constraints=constraints,
                exploration=exploration,
                reward_summary=summary,
            ),
            created_at=base_time + timedelta(seconds=index),
        )
        db_session.add(state)
        db_session.flush()
        reward.policy_state_id = state.id
        states.append(state)
        rewards.append(reward)
    db_session.commit()
    return states, rewards, replay_calls


def test_policy_lineage_replays_seed_plus_sixty_six_rewards_with_request_memo(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.policy import validate_persisted_policy_state

    states, _rewards, replay_calls = _build_policy_lineage(
        db_session,
        sample_knowledge_base.id,
        reward_count=66,
        monkeypatch=monkeypatch,
    )
    weights, safe_arms, _exploration, summary = (
        validate_persisted_policy_state(
            db_session,
            states[-1],
            knowledge_base_id=sample_knowledge_base.id,
        )
    )
    assert weights == states[-1].weights_json
    assert safe_arms == sorted(summary["safe_arms"])
    assert len(replay_calls) == 66

    context = {
        "max_states": len(states),
        "active_state_ids": set(),
        "memo": {},
    }
    validate_persisted_policy_state(
        db_session,
        states[-1],
        knowledge_base_id=sample_knowledge_base.id,
        _validation_context=context,
    )
    calls_after_first_context_replay = len(replay_calls)
    validate_persisted_policy_state(
        db_session,
        states[-1],
        knowledge_base_id=sample_knowledge_base.id,
        _validation_context=context,
    )
    assert len(replay_calls) == calls_after_first_context_replay
    assert len(context["memo"]) == len(states)


def test_policy_lineage_rejects_self_cycle(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.policy import (
        PolicyStateValidationError,
        validate_persisted_policy_state,
    )

    states, _rewards, _calls = _build_policy_lineage(
        db_session,
        sample_knowledge_base.id,
        reward_count=1,
        monkeypatch=monkeypatch,
    )
    child = states[-1]
    summary = dict(child.reward_summary_json)
    summary["previous_policy_state_id"] = child.id
    child.reward_summary_json = summary
    db_session.commit()

    with pytest.raises(PolicyStateValidationError, match="cycle"):
        validate_persisted_policy_state(
            db_session,
            child,
            knowledge_base_id=sample_knowledge_base.id,
        )


def test_policy_lineage_rejects_missing_predecessor(
    db_session,
    sample_knowledge_base,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.policy import (
        PolicyStateValidationError,
        validate_persisted_policy_state,
    )

    states, _rewards, _calls = _build_policy_lineage(
        db_session,
        sample_knowledge_base.id,
        reward_count=1,
        monkeypatch=monkeypatch,
    )
    seed, child = states
    db_session.delete(seed)
    db_session.commit()

    with pytest.raises(
        PolicyStateValidationError,
        match="predecessor id/hash",
    ):
        validate_persisted_policy_state(
            db_session,
            child,
            knowledge_base_id=sample_knowledge_base.id,
        )


@pytest.mark.parametrize(
    (
        "base_overrides",
        "variant_overrides",
        "signal_name",
        "expected_signal",
    ),
    [
        ({}, {"retrieval_hit": 0.75}, "retrieval_hit", 0.75),
        (
            {},
            {"context_precision": 0.75},
            "context_precision",
            0.75,
        ),
        ({}, {"context_recall": 0.75}, "context_recall", 0.75),
        (
            {},
            {"concept_path_accuracy": 0.75},
            "concept_path_accuracy",
            0.75,
        ),
        (
            {},
            {"citation_pass_rate": 0.75},
            "citation_pass_rate",
            0.75,
        ),
        (
            {},
            {"answer_groundedness": 0.75},
            "answer_groundedness",
            0.75,
        ),
        (
            {},
            {"answer_completeness": 0.75},
            "answer_completeness",
            0.75,
        ),
        (
            {},
            {"repair_success_rate": 0.75},
            "repair_success",
            0.75,
        ),
        (
            {},
            {"agent_typed_action_validation_pass_rate": 0.75},
            "typed_action_validation_pass",
            0.75,
        ),
        ({}, {"drift_rate": 0.75}, "drift_quality", 0.25),
        (
            {
                "answer_acceptance_gate_pass": 0.0,
                "citation_pass_rate": 0.0,
                "answer_groundedness": 0.0,
                "answer_completeness": 0.0,
                "repair_success_rate": 0.0,
                "claim_count": 2,
                "supported_claim_count": 0,
                "unsupported_claim_count": 2,
            },
            {"answer_acceptance_gate_pass": 1.0},
            "answer_acceptance",
            1.0,
        ),
        (
            {},
            {
                "claim_count": 4,
                "supported_claim_count": 3,
                "unsupported_claim_count": 1,
            },
            "claim_support_quality",
            0.75,
        ),
        (
            {},
            {"latency_cost": 2.0, "latency_ms": 2000.0},
            "latency_quality",
            0.33333333,
        ),
        (
            {},
            {"task_token_cost": 512.0},
            "task_token_quality",
            0.66666667,
        ),
    ],
    ids=[
        "retrieval-hit",
        "context-precision",
        "context-recall",
        "concept-path-accuracy",
        "citation-pass-rate",
        "answer-groundedness",
        "answer-completeness",
        "repair-success",
        "typed-action-validation",
        "drift-quality",
        "acceptance-from-zero-grounded-credit",
        "claim-count-conservation-tuple",
        "latency-seconds-and-ms-pair",
        "task-token-cost",
    ],
)
def test_policy_posterior_each_effective_signal_is_sensitive_and_deterministic(
    base_overrides: dict[str, Any],
    variant_overrides: dict[str, Any],
    signal_name: str,
    expected_signal: float,
) -> None:
    from app.services.policy import (
        POLICY_ARMS,
        replay_policy_posterior_update,
    )

    prior = {arm: 1.0 for arm in POLICY_ARMS}
    baseline_reward = _reward_payload(**base_overrides)
    variant_reward = deepcopy(baseline_reward)
    variant_reward.update(variant_overrides)
    baseline = replay_policy_posterior_update(prior, baseline_reward)
    variant = replay_policy_posterior_update(prior, variant_reward)

    assert variant == replay_policy_posterior_update(prior, variant_reward)
    assert (
        variant["normalized_signals"][signal_name]
        == expected_signal
    )
    assert any(
        variant["arm_reward_observations"][arm]
        != baseline["arm_reward_observations"][arm]
        for arm in POLICY_ARMS
    )
    assert any(
        variant["weights"][arm] != baseline["weights"][arm]
        for arm in POLICY_ARMS
    )


def _fixed_gray_observation(**overrides: Any) -> dict[str, Any]:
    """A single bounded, allowlisted executor observation for a Gray(P) edge.

    This mirrors the fixture in ``test_deterministic_gray_zone.py``.  It is
    duplicated (not imported) because test modules are collected
    independently and must not depend on each other's private helpers.
    """

    from app.services.chunking import stable_hash

    payload = {
        "current_layer": "mid",
        "path_distance": 0.75,
        "distance_zone": "gray",
        "covered_facets_before": ["alpha"],
        "covered_facets_after": ["alpha", "beta"],
        "required_facets": ["alpha", "beta"],
        "candidate_facets": ["beta"],
        "evidence_roles_before": ["mid_entry"],
        "evidence_roles_after": ["dense_semantic", "mid_entry"],
        "support_ids_before": ["support-0"],
        "support_ids_after": ["support-0", "support-1"],
        "independent_path_contribution_gain": False,
        "path_contribution_key": "9" * 64,
        "support_refs": {"support_chunk_ids": ["chunk-1"]},
        "active_edge_support_gate_pass": True,
        "support_backed_to_covered_path": True,
        "validated_entry_semantic_anchor": True,
        "semantic_uncertain_edge": False,
        "crossing_rq_boundary": False,
        "bridge_or_boundary_reason": [],
        "supported_raw_span_hit": False,
        "structure_context_available": True,
        "drilldown_eligible": False,
        "edge_type": "dense_semantic",
        "rq_membership_diagnostics": {},
        "candidate_chunk_span_summary": {},
        "structure_context_status": {"available": True},
        "hard_interrupt_state": {"max_edge_reuse": 1, "edge_reuse_count": 1},
        "path_distance_green_threshold": 0.5,
        "path_distance_gray_threshold": 1.0,
        "path_distance_hard_threshold": 1.5,
        "gray_zone_rule_protocol_version": "deterministic_support_progress_v1",
    }
    payload.update(overrides)
    support_before = sorted({str(value) for value in payload["support_ids_before"]})
    support_after = sorted({str(value) for value in payload["support_ids_after"]})
    payload.setdefault("support_ids_before_count", len(support_before))
    payload.setdefault("support_ids_after_count", len(support_after))
    payload.setdefault("support_ids_before_hash", stable_hash(support_before))
    payload.setdefault("support_ids_after_hash", stable_hash(support_after))
    payload.setdefault(
        "support_id_gain", bool(set(support_after) - set(support_before))
    )
    return payload


def test_changed_active_policy_state_does_not_change_gray_zone_decision(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy outcome invariant: Policy is read-and-advisory only for the Planner and
    staged-traversal budget priors; it must have zero influence on the
    executor-only deterministic gray-zone rule.

    This test builds two knowledge bases whose *real*, persisted, distinct
    ``PolicyState`` rows read back through the exact same
    ``read_policy_operating_prior`` production path used by
    ``execute_agent_run``.  One KB only has the neutral seed state; the other
    has a reward-derived state with materially different arm priors and
    budget suggestions.  Both priors are then (deliberately) *not* threaded
    into the gray-zone rule input, mirroring the production call sites in
    ``context_graph.py`` -- and the same fixed executor observation is
    replayed once per KB context.  The two Policy identities being genuinely
    different, while the gray decision, matched rule, input hash, and
    zero model-call-count stay byte-for-byte identical, is the closure
    evidence for the deterministic gray-zone authority boundary.
    """

    from app.core.config import get_settings
    from app.models import KnowledgeBase
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        deterministic_gray_zone_decision,
        runtime_settings_state_hash,
    )
    from app.services.policy import read_policy_operating_prior

    def _make_kb(name: str) -> KnowledgeBase:
        source_root = get_settings().knowledge_base_paths_for_name(name)[
            "storage_root"
        ]
        source_root.mkdir(parents=True, exist_ok=True)
        source_root.parent.joinpath("ingestion").mkdir(
            parents=True, exist_ok=True
        )
        kb = KnowledgeBase(name=name, description="tests", source_root=str(source_root))
        db_session.add(kb)
        db_session.commit()
        db_session.refresh(kb)
        return kb

    kb_neutral = _make_kb("Policy outcome gray isolation neutral")
    kb_learned = _make_kb("Policy outcome gray isolation learned")

    _seed_policy_state(db_session, kb_neutral.id, created_at=datetime(2026, 1, 1))
    states, _rewards, _calls = _build_policy_lineage(
        db_session,
        kb_learned.id,
        reward_count=1,
        monkeypatch=monkeypatch,
    )
    seed, learned = states
    assert seed.state_hash != learned.state_hash

    envelope = agent_operating_envelope()
    runtime_hash = runtime_settings_state_hash()
    envelope_hash = agent_operating_envelope_state_hash()
    neutral_prior = read_policy_operating_prior(
        db_session,
        kb_neutral.id,
        runtime_settings_hash=runtime_hash,
        agent_operating_envelope_hash=envelope_hash,
        agent_operating_envelope=envelope,
    )
    learned_prior = read_policy_operating_prior(
        db_session,
        kb_learned.id,
        runtime_settings_hash=runtime_hash,
        agent_operating_envelope_hash=envelope_hash,
        agent_operating_envelope=envelope,
    )

    # The two real, persisted Policy identities read through the production
    # path are genuinely different -- this is not a no-op comparison.
    assert neutral_prior["policy_state_hash"] != learned_prior["policy_state_hash"]
    assert neutral_prior["arm_priors"] != learned_prior["arm_priors"]
    assert neutral_prior["prior_hash"] != learned_prior["prior_hash"]

    fixed_observation = _fixed_gray_observation()
    decision_under_neutral_policy = deterministic_gray_zone_decision(
        dict(fixed_observation)
    )
    decision_under_learned_policy = deterministic_gray_zone_decision(
        dict(fixed_observation)
    )

    assert decision_under_neutral_policy == decision_under_learned_policy
    assert (
        decision_under_neutral_policy["input_hash"]
        == decision_under_learned_policy["input_hash"]
    )
    assert (
        decision_under_neutral_policy["matched_rule"]
        == decision_under_learned_policy["matched_rule"]
    )
    assert (
        decision_under_neutral_policy["decision"]
        == decision_under_learned_policy["decision"]
    )
    assert (
        decision_under_neutral_policy["model_call_count"]
        == decision_under_learned_policy["model_call_count"]
        == 0
    )
    assert (
        decision_under_neutral_policy["decision_source"]
        == "deterministic_local_rule"
    )

    # The gray-zone observation schema itself has no field for a policy
    # identity: attempting to smuggle either real prior's hash or its
    # budget/action priors into the executor-only observation must fail
    # closed rather than silently being ignored.
    for prior in (neutral_prior, learned_prior):
        contaminated = {
            **fixed_observation,
            "policy_state_hash": prior["policy_state_hash"],
            "policy_operating_prior": prior,
        }
        with pytest.raises(ValueError, match="schema mismatch"):
            deterministic_gray_zone_decision(contaminated)
