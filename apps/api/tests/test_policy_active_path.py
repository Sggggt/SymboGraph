from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


def _policy_row(
    db_session,
    knowledge_base_id: str,
    *,
    runtime_hash: str | None = None,
    created_at: datetime | None = None,
):
    from app.models import PolicyState
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import canonical_policy_state_hash
    from app.services.policy import POLICY_ARMS, POLICY_FAMILY, POLICY_VERSION

    effective_runtime_hash = runtime_hash or runtime_settings_state_hash()
    weights = {arm: 1.0 for arm in POLICY_ARMS}
    safe_arms = list(POLICY_ARMS)
    constraints = {
        "fallback_disabled": True,
        "citation_verification_required": True,
        "agent_operating_envelope": agent_operating_envelope(),
        "runtime_settings_hash": effective_runtime_hash,
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
        "path_distance_green_threshold": agent_operating_envelope()[
            "path_distance_green_threshold"
        ],
        "path_distance_gray_threshold": agent_operating_envelope()[
            "path_distance_gray_threshold"
        ],
        "path_distance_hard_threshold": agent_operating_envelope()[
            "path_distance_hard_threshold"
        ],
    }
    reward_summary = {
        "origin": "seed",
        "previous_policy_state_hash": None,
        "safe_arms": safe_arms,
        "posterior": weights,
        "policy_version": POLICY_VERSION,
        "reward_history_tail": [],
        "runtime_settings_hash": effective_runtime_hash,
        "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
    }
    state_hash = canonical_policy_state_hash(
        policy_family=POLICY_FAMILY,
        policy_version=POLICY_VERSION,
        profile_objective_hash=None,
        weights=weights,
        constraints=constraints,
        exploration=exploration,
        reward_summary=reward_summary,
    )
    row = PolicyState(
        knowledge_base_id=knowledge_base_id,
        policy_family=POLICY_FAMILY,
        policy_version=POLICY_VERSION,
        weights_json=weights,
        constraints_json=constraints,
        exploration_json=exploration,
        reward_summary_json=reward_summary,
        state_hash=state_hash,
        created_at=created_at or datetime.utcnow(),
    )
    db_session.add(row)
    db_session.commit()
    return row


def _reward_row(
    db_session,
    knowledge_base_id: str,
    *,
    metric_overrides: dict | None = None,
):
    from app.models import PolicyState, RewardEvent
    from app.services.context_graph import (
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )

    reward_metrics = {
        "reward_metrics_protocol_version": "trace_grounded_reward_metrics_v2",
        "reward_metric_evidence_hash": "d" * 64,
        "retrieval_hit": 1.0,
        "context_precision": 1.0,
        "context_recall": 0.25,
        "concept_path_accuracy": 0.25,
        "citation_pass_rate": 1.0,
        "answer_groundedness": 1.0,
        "answer_completeness": 0.5,
        "claim_count": 2.0,
        "supported_claim_count": 2.0,
        "unsupported_claim_count": 0.0,
        "repair_success_rate": 0.0,
        "agent_typed_action_validation_pass_rate": 1.0,
        "latency_cost": 2.0,
        "latency_ms": 2000.0,
        "task_token_cost": 128.0,
        "drift_rate": 0.0,
        "answer_acceptance_gate_pass": 1.0,
        **(metric_overrides or {}),
    }
    reward = RewardEvent(
        knowledge_base_id=knowledge_base_id,
        reward_json=reward_metrics,
        diagnostics_json={
            "source": "context_graph_agent_v1",
            "reward_metrics_protocol_version": (
                reward_metrics["reward_metrics_protocol_version"]
            ),
            "reward_metric_evidence_hash": reward_metrics[
                "reward_metric_evidence_hash"
            ],
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope_hash": (
                agent_operating_envelope_state_hash()
            ),
        },
    )
    db_session.add(reward)
    db_session.flush()
    return reward


def test_policy_reader_returns_only_bounded_advisory_prior(
    db_session, sample_knowledge_base
) -> None:
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.policy import (
        PolicyStateValidationError,
        read_policy_operating_prior,
    )

    row = _policy_row(db_session, sample_knowledge_base.id)
    envelope = agent_operating_envelope()
    prior = read_policy_operating_prior(
        db_session,
        sample_knowledge_base.id,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        agent_operating_envelope=envelope,
    )

    assert prior["policy_state_id"] == row.id
    assert prior["policy_state_hash"] == row.state_hash
    assert prior["applied"] is True
    assert prior["action_priors"]
    assert prior["budget_suggestions"]
    for field, suggestion in prior["budget_suggestions"].items():
        assert 0 <= suggestion <= int(envelope[field])
    assert prior["runtime_hard_interrupts_unchanged"] is True
    assert prior["threshold_suggestions_runtime_lifecycle_accepted"] is False
    assert prior["threshold_suggestions_applied"] is False
    assert prior["planner_replacement"] is False
    assert prior["gray_zone_decision_authority"] is False
    assert prior["gray_zone_rule_inputs_modified"] is False
    assert prior["gray_zone_model_call_count"] == 0


def test_policy_reader_fails_closed_on_state_hash_drift(
    db_session, sample_knowledge_base
) -> None:
    from sqlalchemy.orm.attributes import flag_modified

    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.policy import (
        PolicyStateValidationError,
        read_policy_operating_prior,
    )

    row = _policy_row(db_session, sample_knowledge_base.id)
    weights = dict(row.weights_json)
    weights["bridge_edge_exploration"] = 999.0
    row.weights_json = weights
    flag_modified(row, "weights_json")
    db_session.commit()

    with pytest.raises(PolicyStateValidationError, match="state-hash replay"):
        read_policy_operating_prior(
            db_session,
            sample_knowledge_base.id,
            runtime_settings_hash=runtime_settings_state_hash(),
            agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
            agent_operating_envelope=agent_operating_envelope(),
        )


def test_policy_runtime_identity_change_disables_application_not_gray_rule(
    db_session, sample_knowledge_base
) -> None:
    from app.services.context_graph import agent_operating_envelope
    from app.services.policy import read_policy_operating_prior

    _policy_row(db_session, sample_knowledge_base.id)
    envelope = agent_operating_envelope()
    prior = read_policy_operating_prior(
        db_session,
        sample_knowledge_base.id,
        runtime_settings_hash="0" * 64,
        agent_operating_envelope_hash="1" * 64,
        agent_operating_envelope=envelope,
    )

    assert prior["applied"] is False
    assert prior["inactive_reason"] == "runtime_identity_changed"
    assert prior["budget_suggestions"] == {
        field: int(envelope.get(field) or 0)
        for field in prior["budget_suggestions"]
    }
    assert prior["threshold_suggestions_applied"] is False
    assert prior["gray_zone_model_call_count"] == 0


def test_unknown_policy_family_cannot_mask_the_latest_active_family(
    db_session, sample_knowledge_base
) -> None:
    from app.models import PolicyState
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        latest_policy_state_content_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import canonical_policy_state_hash
    from app.services.policy import read_policy_operating_prior

    admitted = _policy_row(
        db_session,
        sample_knowledge_base.id,
        created_at=datetime.utcnow() - timedelta(seconds=1),
    )
    unknown_family = "untrusted_policy_family"
    unknown_version = "untrusted_policy_v999"
    ignored = PolicyState(
        knowledge_base_id=sample_knowledge_base.id,
        policy_family=unknown_family,
        policy_version=unknown_version,
        weights_json={"continue_every_gray_path": 1e30},
        constraints_json={"gray_zone_decision_authority": True},
        exploration_json={"gray_zone_model_call_count": 1_000_000},
        reward_summary_json={"origin": "untrusted"},
        created_at=datetime.utcnow(),
    )
    ignored.state_hash = canonical_policy_state_hash(
        policy_family=unknown_family,
        policy_version=unknown_version,
        profile_objective_hash=None,
        weights=ignored.weights_json,
        constraints=ignored.constraints_json,
        exploration=ignored.exploration_json,
        reward_summary=ignored.reward_summary_json,
    )
    db_session.add(ignored)
    db_session.commit()

    prior = read_policy_operating_prior(
        db_session,
        sample_knowledge_base.id,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        agent_operating_envelope=agent_operating_envelope(),
    )
    assert prior["policy_state_id"] == admitted.id
    assert prior["policy_state_hash"] == admitted.state_hash
    assert latest_policy_state_content_hash(
        db_session, sample_knowledge_base.id
    ) == admitted.state_hash
    assert prior["gray_zone_decision_authority"] is False
    assert prior["gray_zone_model_call_count"] == 0


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (
            lambda card: card["action_priors"].update(
                {"continue_path": 1.0}
            ),
            "action priors",
        ),
        (
            lambda card: card["budget_suggestions"].update(
                {"agent_chunk_top_k": 1_000_001}
            ),
            "budget suggestion",
        ),
    ],
)
def test_policy_prior_closed_contract_rejects_self_hashed_authority_injection(
    db_session,
    sample_knowledge_base,
    mutation,
    expected_error: str,
) -> None:
    from app.services.chunking import stable_hash
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.policy import (
        PolicyStateValidationError,
        read_policy_operating_prior,
        validate_policy_operating_prior_card,
    )

    envelope = agent_operating_envelope()
    prior = read_policy_operating_prior(
        db_session,
        sample_knowledge_base.id,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        agent_operating_envelope=envelope,
    )
    forged = deepcopy(prior)
    mutation(forged)
    forged.pop("prior_hash")
    forged["prior_hash"] = stable_hash(forged)

    with pytest.raises(PolicyStateValidationError, match=expected_error):
        validate_policy_operating_prior_card(
            forged,
            knowledge_base_id=sample_knowledge_base.id,
            agent_operating_envelope=envelope,
            runtime_settings_hash=runtime_settings_state_hash(),
            agent_operating_envelope_hash=(
                agent_operating_envelope_state_hash()
            ),
        )


@pytest.mark.parametrize("forgery", ["gray_authority", "posterior", "reward_link"])
def test_policy_reader_rejects_synchronized_state_forgery(
    db_session, sample_knowledge_base, forgery: str
) -> None:
    from sqlalchemy.orm.attributes import flag_modified

    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import canonical_policy_state_hash_for_row
    from app.services.policy import (
        PolicyStateValidationError,
        read_policy_operating_prior,
    )

    row = _policy_row(db_session, sample_knowledge_base.id)
    if forgery == "gray_authority":
        constraints = dict(row.constraints_json)
        constraints["gray_zone_decision_authority"] = True
        row.constraints_json = constraints
        flag_modified(row, "constraints_json")
    elif forgery == "posterior":
        summary = dict(row.reward_summary_json)
        posterior = dict(summary["posterior"])
        posterior["bridge_edge_exploration"] = 999.0
        summary["posterior"] = posterior
        row.reward_summary_json = summary
        flag_modified(row, "reward_summary_json")
    else:
        summary = dict(row.reward_summary_json)
        summary.update(
            {
                "origin": "reward_update",
                "last_reward_event_id": "missing-reward-event",
                "last_reward": {"citation_pass_rate": 1.0},
                "reward_history_tail": [{"citation_pass_rate": 1.0}],
            }
        )
        row.reward_summary_json = summary
        flag_modified(row, "reward_summary_json")
    row.state_hash = canonical_policy_state_hash_for_row(row)
    db_session.commit()

    with pytest.raises(PolicyStateValidationError):
        read_policy_operating_prior(
            db_session,
            sample_knowledge_base.id,
            runtime_settings_hash=runtime_settings_state_hash(),
            agent_operating_envelope_hash=(
                agent_operating_envelope_state_hash()
            ),
            agent_operating_envelope=agent_operating_envelope(),
        )


def test_reward_metrics_replay_real_trace_package_and_repair_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import agent_graph

    monkeypatch.setattr(
        agent_graph,
        "claim_grounding_gate",
        lambda *_args, **_kwargs: {
            "claim_pass_rate": 0.5,
            "claim_count": 2,
            "supported_claim_count": 1,
            "unsupported_claim_count": 1,
            "gate_hash": "g" * 64,
        },
    )
    package = SimpleNamespace(
        id="package-1",
        retrieval_trace_id="trace-1",
        package_json={
            "chunks": [
                {"chunk_id": "chunk-1"},
                {"chunk_id": "chunk-2"},
            ]
        },
        hit_chunk_ids_json=["chunk-1", "chunk-2", "chunk-3"],
        concept_path_json=[
            {"layer": "mid", "ids": ["mid-1"]},
            {"layer": "chunk", "ids": []},
        ],
        covered_facets_json=["facet-a"],
        token_count=42,
    )
    retrieval_trace = SimpleNamespace(
        query_facets_json={"required_facets": ["facet-a", "facet-b"]},
        concept_path_json=[
            {"layer": "mid", "ids": ["mid-1"]},
            {"layer": "chunk", "ids": ["chunk-1"]},
        ],
        result_chunk_ids_json=["chunk-1", "chunk-2", "chunk-3"],
    )
    plans = [
        SimpleNamespace(id="plan-1", validation_json={"valid": True}),
        SimpleNamespace(id="plan-2", validation_json={"valid": False}),
    ]
    events = [
        SimpleNamespace(id="event-1", duration_ms=250),
        SimpleNamespace(id="event-2", duration_ms=750),
    ]
    metrics = agent_graph.reward_metrics_from_verifications(
        package,
        [
            {"verdict": "supported", "chunk_id": "chunk-1"},
            {"verdict": "unsupported", "chunk_id": "chunk-2"},
        ],
        "answer",
        retrieval_trace=retrieval_trace,
        agent_plans=plans,
        trace_events=events,
        repair_actions=[
            {
                "repair_round_index": 0,
                "made_semantic_progress": True,
                "action_output_hash": "a" * 64,
            },
            {
                "repair_round_index": 1,
                "made_semantic_progress": False,
                "action_output_hash": "b" * 64,
            },
        ],
    )

    assert metrics["context_precision"] == 0.5
    assert metrics["context_recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["concept_path_accuracy"] == 0.5
    assert metrics["agent_typed_action_validation_pass_rate"] == 0.5
    assert metrics["latency_ms"] == 1000.0
    assert metrics["latency_cost"] == 1.0
    assert metrics["drift_rate"] == 0.5
    assert metrics["repair_success_rate"] == 0.5
    assert len(metrics["reward_metric_evidence_hash"]) == 64


def test_reward_metrics_vary_with_real_citation_pass_rate_not_a_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy outcome: reward components must be real outcome signals, not a fixed
    existence/constant proxy.  Two otherwise-identical answer sessions that
    differ only in how many claims were actually verified as supported must
    receive different, and correctly ordered, recorded reward -- never the
    same constant regardless of citation outcome."""

    from app.services import agent_graph

    package = SimpleNamespace(
        id="package-1",
        retrieval_trace_id="trace-1",
        package_json={
            "chunks": [
                {"chunk_id": "chunk-1"},
                {"chunk_id": "chunk-2"},
                {"chunk_id": "chunk-3"},
                {"chunk_id": "chunk-4"},
            ]
        },
        hit_chunk_ids_json=["chunk-1", "chunk-2", "chunk-3", "chunk-4"],
        concept_path_json=[],
        covered_facets_json=[],
        token_count=10,
    )

    def _gate_for(pass_rate: float, count: int) -> dict:
        supported = round(pass_rate * count)
        return {
            "claim_pass_rate": pass_rate,
            "claim_count": count,
            "supported_claim_count": supported,
            "unsupported_claim_count": count - supported,
            "gate_hash": "g" * 64,
        }

    def _metrics_for(pass_rate: float, verification_results: list[dict]) -> dict:
        monkeypatch.setattr(
            agent_graph,
            "claim_grounding_gate",
            lambda *_a, **_k: _gate_for(pass_rate, len(verification_results)),
        )
        return agent_graph.reward_metrics_from_verifications(
            package,
            verification_results,
            "answer",
        )

    low_pass_metrics = _metrics_for(
        0.25,
        [
            {"verdict": "supported", "chunk_id": "chunk-1"},
            {"verdict": "unsupported", "chunk_id": "chunk-2"},
            {"verdict": "unsupported", "chunk_id": "chunk-3"},
            {"verdict": "unsupported", "chunk_id": "chunk-4"},
        ],
    )
    high_pass_metrics = _metrics_for(
        1.0,
        [
            {"verdict": "supported", "chunk_id": "chunk-1"},
            {"verdict": "supported", "chunk_id": "chunk-2"},
            {"verdict": "supported", "chunk_id": "chunk-3"},
            {"verdict": "supported", "chunk_id": "chunk-4"},
        ],
    )

    # The two runs must not collapse onto the same constant/existence proxy:
    # citation_pass_rate, groundedness, and context_precision must all move
    # in the correct direction with the real supported/unsupported outcome.
    assert low_pass_metrics["citation_pass_rate"] == 0.25
    assert high_pass_metrics["citation_pass_rate"] == 1.0
    assert low_pass_metrics["citation_pass_rate"] != high_pass_metrics["citation_pass_rate"]
    assert low_pass_metrics["answer_groundedness"] != high_pass_metrics["answer_groundedness"]
    assert low_pass_metrics["context_precision"] < high_pass_metrics["context_precision"]
    assert (
        low_pass_metrics["reward_metric_evidence_hash"]
        != high_pass_metrics["reward_metric_evidence_hash"]
    )

    # Feeding the resulting reward through the real posterior update also
    # moves the arm weights differently -- the difference is not absorbed
    # away before it reaches the Policy update.
    from app.services.policy import POLICY_ARMS, replay_policy_posterior_update

    prior = {arm: 1.0 for arm in POLICY_ARMS}

    def _full_reward_json(metrics: dict) -> dict:
        return {
            "retrieval_hit": 1.0,
            "context_precision": metrics["context_precision"],
            "context_recall": 1.0,
            "concept_path_accuracy": 0.0,
            "citation_pass_rate": metrics["citation_pass_rate"],
            "answer_groundedness": metrics["answer_groundedness"],
            "answer_completeness": metrics["answer_completeness"] or 0.0,
            "claim_count": metrics["claim_count"],
            "supported_claim_count": metrics["supported_claim_count"],
            "unsupported_claim_count": metrics[
                "unsupported_claim_count"
            ],
            "repair_success_rate": 0.0,
            "agent_typed_action_validation_pass_rate": 1.0,
            "latency_cost": metrics["latency_cost"],
            "latency_ms": metrics["latency_ms"],
            "task_token_cost": metrics["task_token_cost"],
            "drift_rate": 0.0,
            "answer_acceptance_gate_pass": 1.0,
        }

    low_posterior = replay_policy_posterior_update(prior, _full_reward_json(low_pass_metrics))
    high_posterior = replay_policy_posterior_update(prior, _full_reward_json(high_pass_metrics))
    assert any(
        low_posterior["weights"][arm] != high_posterior["weights"][arm]
        for arm in POLICY_ARMS
    )


@pytest.mark.asyncio
async def test_policy_prior_is_visible_to_planner_only_as_advisory_input(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.services import agent_graph
    from app.services.context_graph import (
        agent_operating_envelope_state_hash,
        runtime_settings_state_hash,
    )
    from app.services.policy import read_policy_operating_prior

    envelope = agent_graph.agent_operating_envelope()
    prior = read_policy_operating_prior(
        db_session,
        sample_knowledge_base.id,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        agent_operating_envelope=envelope,
    )

    class CapturingPlanner:
        def __init__(self) -> None:
            self.user_prompt = ""

        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            self.user_prompt = user_prompt
            return {
                "typed_actions": agent_graph.fallback_typed_actions(
                    "query", envelope
                )
            }

    provider = CapturingPlanner()
    monkeypatch.setattr(agent_graph, "ChatProvider", lambda: provider)
    actions, _raw = await agent_graph.propose_agent_plan(
        "query",
        [],
        {"intent": "definition"},
        envelope,
        policy_operating_prior=prior,
        policy_knowledge_base_id=sample_knowledge_base.id,
    )

    assert prior["prior_hash"] in provider.user_prompt
    assert "Policy is advisory only" in provider.user_prompt
    assert "cannot replace this planner" in provider.user_prompt
    assert "or decide any gray-zone path" in provider.user_prompt
    assert all(
        action.get("action_type") != "evaluate_gray_zone_path"
        for action in actions
    )


@pytest.mark.asyncio
async def test_real_agent_reward_changes_next_admitted_policy_prior_and_cache_identity(
    db_session, populated_context_graph
) -> None:
    from sqlalchemy import select
    from sqlalchemy.orm.attributes import flag_modified

    from app.models import PolicyState, RewardEvent
    from app.schemas import AgentRequest, SearchFilters
    from app.services import agent_graph
    from app.services.agent_graph import update_policy_state_from_reward
    from app.services.context_graph import (
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        latest_policy_state_content_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import canonical_policy_state_hash_for_row
    from app.services.policy import (
        PolicyStateValidationError,
        read_policy_operating_prior,
    )
    from app.services.policy_reward import replay_policy_reward_event

    kb = populated_context_graph["knowledge_base"]
    seed = _policy_row(db_session, kb.id)
    envelope = agent_operating_envelope()
    response = await agent_graph.run_agent(
        db_session,
        AgentRequest(
            knowledge_base_id=kb.id,
            question="Explain Bayesian network factorization.",
            filters=SearchFilters(),
            top_k=4,
        ),
    )
    # Prove that every UUID-free reward/policy witness survives the actual
    # transaction boundary, rather than succeeding only against dirty ORM
    # objects in the creating session.
    db_session.commit()
    db_session.expire_all()
    reward = db_session.scalar(
        select(RewardEvent).where(
            RewardEvent.answer_session_id
            == response["answer_model_audit"]["answer_session_id"]
        )
    )
    assert reward is not None
    assert reward.policy_state_id is not None
    replay = replay_policy_reward_event(db_session, reward)
    assert reward.reward_json["reward_metric_evidence_hash"] == replay[
        "evidence_hash"
    ]
    state = db_session.get(PolicyState, reward.policy_state_id)
    assert state is not None
    prior = read_policy_operating_prior(
        db_session,
        kb.id,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        agent_operating_envelope=envelope,
    )

    assert reward.policy_state_id == state.id
    assert prior["policy_state_id"] == state.id
    assert prior["applied"] is True
    assert len(set(prior["arm_priors"].values())) > 1
    assert all(
        0 <= suggestion <= int(envelope[field])
        for field, suggestion in prior["budget_suggestions"].items()
    )
    assert latest_policy_state_content_hash(
        db_session, kb.id
    ) == state.state_hash
    assert prior["gray_zone_decision_authority"] is False
    assert prior["gray_zone_rule_inputs_modified"] is False
    assert prior["gray_zone_model_call_count"] == 0

    with pytest.raises(
        PolicyStateValidationError,
        match="already been consumed",
    ):
        update_policy_state_from_reward(db_session, kb.id, reward)

    # A synchronized rewrite of the predecessor gets a new canonical hash;
    # the learned state's immutable predecessor witness must not be laundered.
    seed_weights = dict(seed.weights_json)
    seed_weights[next(iter(seed_weights))] = 2.0
    seed.weights_json = seed_weights
    seed_summary = dict(seed.reward_summary_json)
    seed_summary["posterior"] = dict(seed_weights)
    seed.reward_summary_json = seed_summary
    flag_modified(seed, "weights_json")
    flag_modified(seed, "reward_summary_json")
    seed.state_hash = canonical_policy_state_hash_for_row(seed)
    db_session.commit()
    with pytest.raises(PolicyStateValidationError, match="predecessor|evidence"):
        read_policy_operating_prior(
            db_session,
            kb.id,
            runtime_settings_hash=runtime_settings_state_hash(),
            agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
            agent_operating_envelope=envelope,
        )


def test_policy_update_refuses_to_launder_corrupt_latest_state(
    db_session, sample_knowledge_base
) -> None:
    from sqlalchemy import func, select
    from sqlalchemy.orm.attributes import flag_modified

    from app.models import PolicyState
    from app.services.agent_graph import update_policy_state_from_reward
    from app.services.policy import PolicyStateValidationError

    state = _policy_row(db_session, sample_knowledge_base.id)
    weights = dict(state.weights_json)
    weights["bridge_edge_exploration"] = 999.0
    state.weights_json = weights
    flag_modified(state, "weights_json")
    db_session.commit()
    reward = _reward_row(db_session, sample_knowledge_base.id)
    count_before = db_session.scalar(select(func.count(PolicyState.id)))

    with pytest.raises(PolicyStateValidationError, match="state-hash replay"):
        update_policy_state_from_reward(
            db_session,
            sample_knowledge_base.id,
            reward,
        )

    assert db_session.scalar(select(func.count(PolicyState.id))) == count_before
    assert reward.policy_state_id is None


@pytest.mark.asyncio
async def test_layered_search_freezes_policy_identity_for_whole_agent_snapshot(
    db_session, populated_context_graph
) -> None:
    from app.schemas import SearchFilters
    from app.services.context_graph import layered_search

    kb = populated_context_graph["knowledge_base"]
    first = _policy_row(
        db_session,
        kb.id,
        created_at=datetime.utcnow() - timedelta(seconds=2),
    )
    second = _policy_row(
        db_session,
        kb.id,
        runtime_hash="f" * 64,
        created_at=datetime.utcnow() - timedelta(seconds=1),
    )
    assert first.state_hash != second.state_hash

    frozen = await layered_search(
        db_session,
        kb.id,
        "Bayesian network factorization",
        SearchFilters(),
        2,
        policy_identity_frozen=True,
        frozen_policy_state_hash=first.state_hash,
    )
    latest = await layered_search(
        db_session,
        kb.id,
        "Bayesian network factorization",
        SearchFilters(),
        2,
    )
    frozen_none = await layered_search(
        db_session,
        kb.id,
        "Bayesian network factorization",
        SearchFilters(),
        2,
        policy_identity_frozen=True,
        frozen_policy_state_hash=None,
    )

    assert frozen.trace.policy_state_hash == first.state_hash
    assert (
        frozen.trace.diagnostics_json["cache_key_components"][
            "policy_state_hash"
        ]
        == first.state_hash
    )
    assert frozen.trace.diagnostics_json["policy_identity_frozen"] is True
    assert latest.trace.policy_state_hash == second.state_hash
    assert (
        latest.trace.diagnostics_json["cache_key_components"][
            "policy_state_hash"
        ]
        == second.state_hash
    )
    assert frozen_none.trace.policy_state_hash is None
    assert (
        frozen_none.trace.diagnostics_json["cache_key_components"][
            "policy_state_hash"
        ]
        is None
    )


@pytest.mark.asyncio
async def test_no_active_chunks_trace_persists_frozen_policy_identity(
    db_session, sample_knowledge_base
) -> None:
    from app.schemas import SearchFilters
    from app.services.context_graph import layered_search

    policy = _policy_row(db_session, sample_knowledge_base.id)
    result = await layered_search(
        db_session,
        sample_knowledge_base.id,
        "empty knowledge base",
        SearchFilters(),
        2,
        policy_identity_frozen=True,
        frozen_policy_state_hash=policy.state_hash,
    )

    assert result.results == []
    assert result.trace.policy_state_hash == policy.state_hash
    assert (
        result.trace.diagnostics_json["cache_key_components"][
            "policy_state_hash"
        ]
        == policy.state_hash
    )
    assert result.trace.diagnostics_json["policy_identity_frozen"] is True
