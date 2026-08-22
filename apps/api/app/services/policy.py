from __future__ import annotations

import math
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import PolicyState, RewardEvent
from app.services.chunking import stable_hash
from app.services.graph_state_hashes import canonical_policy_state_hash_for_row
from app.services.policy_reward import (
    POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
    POLICY_REWARD_FACT_PROTOCOL_VERSION,
    replay_policy_reward_event,
)


POLICY_FAMILY = "context_graph_bandit"
POLICY_VERSION = "context_graph_bandit_v1"
POLICY_OPERATING_PRIOR_PROTOCOL_VERSION = "policy_operating_prior_v1"
POLICY_ARMS = (
    "high_precision_direct_chunk",
    "structure_context_heavy",
    "rq_membership_expansion",
    "mid_concept_expansion",
    "coarse_to_mid_drilldown",
    "bridge_edge_exploration",
    "formula_table_closure",
    "cross_document_synthesis",
    "low_latency_minimal_context",
)
POLICY_ARM_ACTIONS: dict[str, tuple[str, ...]] = {
    "high_precision_direct_chunk": ("select_entry_nodes", "recall_chunks"),
    "structure_context_heavy": (
        "restore_context_package",
        "build_context_package",
    ),
    "rq_membership_expansion": ("route_rq_addresses", "walk_graph_frontier"),
    "mid_concept_expansion": ("route_mid_concepts", "drill_down_layer"),
    "coarse_to_mid_drilldown": (
        "activate_coarse_concepts",
        "drill_down_layer",
    ),
    "bridge_edge_exploration": ("jump_bridge", "walk_graph_frontier"),
    "formula_table_closure": (
        "restore_context_package",
        "repair_structure_context",
    ),
    "cross_document_synthesis": (
        "walk_graph_frontier",
        "build_context_package",
    ),
    "low_latency_minimal_context": ("recall_chunks", "verify_citations"),
}
POLICY_BUDGET_ARM_MAP: dict[str, tuple[str, ...]] = {
    "agent_coarse_initial_budget": (
        "coarse_to_mid_drilldown",
        "cross_document_synthesis",
    ),
    "agent_coarse_top_k": ("coarse_to_mid_drilldown",),
    "agent_mid_per_coarse_budget": (
        "coarse_to_mid_drilldown",
        "mid_concept_expansion",
    ),
    "agent_coarse_drilldown_mid_initial_budget": (
        "coarse_to_mid_drilldown",
        "mid_concept_expansion",
    ),
    "agent_mid_initial_budget": (
        "mid_concept_expansion",
        "rq_membership_expansion",
    ),
    "agent_mid_top_k": ("mid_concept_expansion",),
    "agent_chunk_per_mid_budget": (
        "high_precision_direct_chunk",
        "rq_membership_expansion",
    ),
    "agent_chunk_initial_budget": (
        "high_precision_direct_chunk",
        "bridge_edge_exploration",
    ),
    "agent_chunk_top_k": (
        "high_precision_direct_chunk",
        "low_latency_minimal_context",
    ),
    "structure_restore_per_chunk_budget": (
        "structure_context_heavy",
        "formula_table_closure",
    ),
    "context_package_token_budget": (
        "structure_context_heavy",
        "cross_document_synthesis",
    ),
}
POLICY_THRESHOLD_FIELDS = (
    "path_distance_green_threshold",
    "path_distance_gray_threshold",
    "path_distance_hard_threshold",
)
POLICY_POSTERIOR_UPDATE_PROTOCOL_VERSION = "bounded_metric_exp_update_v3"
POLICY_POSTERIOR_LEARNING_RATE = 0.1
POLICY_TASK_TOKEN_NORMALIZATION_SCALE = 1024.0

# Every required reward signal is consumed by at least one arm.  Costs and
# drift are normalized into quality values before this fixed mapping is used.
POLICY_ARM_REWARD_SIGNALS: dict[str, tuple[str, ...]] = {
    "high_precision_direct_chunk": (
        "retrieval_hit",
        "context_precision",
        "citation_pass_rate",
        "answer_groundedness",
        "typed_action_validation_pass",
        "answer_acceptance",
    ),
    "structure_context_heavy": (
        "context_recall",
        "answer_completeness",
        "citation_pass_rate",
        "repair_success",
        "answer_groundedness",
    ),
    "rq_membership_expansion": (
        "retrieval_hit",
        "concept_path_accuracy",
        "context_recall",
        "drift_quality",
        "typed_action_validation_pass",
    ),
    "mid_concept_expansion": (
        "concept_path_accuracy",
        "context_recall",
        "answer_completeness",
        "answer_groundedness",
        "drift_quality",
    ),
    "coarse_to_mid_drilldown": (
        "retrieval_hit",
        "concept_path_accuracy",
        "context_recall",
        "answer_completeness",
        "drift_quality",
    ),
    "bridge_edge_exploration": (
        "retrieval_hit",
        "repair_success",
        "context_recall",
        "drift_quality",
        "typed_action_validation_pass",
    ),
    "formula_table_closure": (
        "repair_success",
        "citation_pass_rate",
        "answer_groundedness",
        "answer_completeness",
        "answer_acceptance",
    ),
    "cross_document_synthesis": (
        "context_recall",
        "concept_path_accuracy",
        "answer_groundedness",
        "answer_completeness",
        "claim_support_quality",
    ),
    "low_latency_minimal_context": (
        "latency_quality",
        "task_token_quality",
        "context_precision",
        "citation_pass_rate",
        "typed_action_validation_pass",
        "answer_acceptance",
    ),
}


class PolicyStateValidationError(RuntimeError):
    """The latest persisted PolicyState cannot be used as an active prior."""


_POLICY_PRIOR_FIELDS = frozenset(
    {
        "protocol_version",
        "knowledge_base_id",
        "policy_state_id",
        "policy_state_hash",
        "policy_version",
        "applied",
        "inactive_reason",
        "safe_arms",
        "arm_priors",
        "action_priors",
        "budget_suggestions",
        "budget_suggestions_are_soft",
        "runtime_hard_interrupts_unchanged",
        "threshold_suggestions",
        "threshold_suggestions_runtime_lifecycle_accepted",
        "threshold_suggestions_applied",
        "planner_replacement",
        "gray_zone_decision_authority",
        "gray_zone_rule_inputs_modified",
        "gray_zone_model_call_count",
        "runtime_settings_hash",
        "agent_operating_envelope_hash",
        "prior_hash",
    }
)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_policy_operating_prior_card(
    card: dict[str, Any],
    *,
    knowledge_base_id: str,
    agent_operating_envelope: dict[str, Any] | None = None,
    runtime_settings_hash: str | None = None,
    agent_operating_envelope_hash: str | None = None,
) -> dict[str, Any]:
    payload = dict(card or {})
    if set(payload) != _POLICY_PRIOR_FIELDS:
        raise PolicyStateValidationError(
            "Policy operating-prior card is not a closed contract"
        )
    prior_hash = str(payload.pop("prior_hash", ""))
    if (
        payload.get("protocol_version")
        != POLICY_OPERATING_PRIOR_PROTOCOL_VERSION
        or payload.get("knowledge_base_id") != str(knowledge_base_id)
        or payload.get("policy_version") != POLICY_VERSION
        or prior_hash != stable_hash(payload)
        or type(payload.get("applied")) is not bool
        or type(payload.get("budget_suggestions_are_soft")) is not bool
        or payload.get("budget_suggestions_are_soft") is not True
        or payload.get("runtime_hard_interrupts_unchanged") is not True
        or payload.get("planner_replacement") is not False
        or payload.get("gray_zone_decision_authority") is not False
        or payload.get("gray_zone_rule_inputs_modified") is not False
        or type(payload.get("gray_zone_model_call_count")) is not int
        or payload.get("gray_zone_model_call_count") != 0
        or payload.get("threshold_suggestions_applied") is not False
        or payload.get(
            "threshold_suggestions_runtime_lifecycle_accepted"
        )
        is not False
    ):
        raise PolicyStateValidationError(
            "Policy operating-prior card failed immutable boundary replay"
        )
    applied = payload["applied"]
    state_id = payload.get("policy_state_id")
    state_hash = payload.get("policy_state_hash")
    inactive_reason = payload.get("inactive_reason")
    if applied:
        if not str(state_id or "") or not _is_sha256(state_hash) or inactive_reason is not None:
            raise PolicyStateValidationError(
                "Applied Policy prior is missing its persisted state identity"
            )
    elif inactive_reason not in {"no_policy_state", "runtime_identity_changed"}:
        raise PolicyStateValidationError(
            "Inactive Policy prior has an unknown reason"
        )
    elif inactive_reason == "no_policy_state" and (
        state_id is not None or state_hash is not None
    ):
        raise PolicyStateValidationError(
            "No-state Policy prior unexpectedly names persisted state"
        )
    elif inactive_reason == "runtime_identity_changed" and (
        not str(state_id or "") or not _is_sha256(state_hash)
    ):
        raise PolicyStateValidationError(
            "Runtime-inactive Policy prior is missing its persisted state identity"
        )

    if not _is_sha256(payload.get("runtime_settings_hash")) or not _is_sha256(
        payload.get("agent_operating_envelope_hash")
    ):
        raise PolicyStateValidationError(
            "Policy operating-prior runtime identities are invalid"
        )
    if runtime_settings_hash is not None and payload.get(
        "runtime_settings_hash"
    ) != runtime_settings_hash:
        raise PolicyStateValidationError(
            "Policy operating-prior runtime identity changed"
        )
    if agent_operating_envelope_hash is not None and payload.get(
        "agent_operating_envelope_hash"
    ) != agent_operating_envelope_hash:
        raise PolicyStateValidationError(
            "Policy operating-prior envelope identity changed"
        )

    safe_arm_values = payload.get("safe_arms")
    if not isinstance(safe_arm_values, list):
        raise PolicyStateValidationError("Policy safe arms must be a list")
    safe_arms = [str(arm) for arm in safe_arm_values]
    if (
        not safe_arms
        or len(safe_arms) != len(set(safe_arms))
        or any(arm not in POLICY_ARMS for arm in safe_arms)
    ):
        raise PolicyStateValidationError(
            "Policy safe arms are empty, duplicated, or unknown"
        )
    raw_arm_priors = payload.get("arm_priors")
    if not isinstance(raw_arm_priors, dict) or set(raw_arm_priors) != set(
        POLICY_ARMS
    ):
        raise PolicyStateValidationError(
            "Policy arm-prior set does not match the local allowlist"
        )
    arm_priors = {
        arm: _finite_nonnegative(
            raw_arm_priors[arm], field=f"arm_priors.{arm}"
        )
        for arm in POLICY_ARMS
    }
    expected_action_priors = _normalized_action_priors(
        arm_priors, safe_arms
    )
    if payload.get("action_priors") != expected_action_priors:
        raise PolicyStateValidationError(
            "Policy action priors do not replay from safe arms and arm priors"
        )

    budget_suggestions = payload.get("budget_suggestions")
    if not isinstance(budget_suggestions, dict) or set(
        budget_suggestions
    ) != set(POLICY_BUDGET_ARM_MAP):
        raise PolicyStateValidationError(
            "Policy budget suggestions do not match the local allowlist"
        )
    for field, raw_value in budget_suggestions.items():
        if type(raw_value) is not int or raw_value < 0 or raw_value > 1_000_000:
            raise PolicyStateValidationError(
                f"Policy budget suggestion {field} is not a bounded integer"
            )
        if agent_operating_envelope is not None and raw_value > int(
            agent_operating_envelope.get(field) or 0
        ):
            raise PolicyStateValidationError(
                f"Policy budget suggestion {field} exceeds its hard interrupt"
            )

    threshold_suggestions = payload.get("threshold_suggestions")
    if not isinstance(threshold_suggestions, dict) or any(
        field not in POLICY_THRESHOLD_FIELDS for field in threshold_suggestions
    ):
        raise PolicyStateValidationError(
            "Policy threshold suggestions do not match the local allowlist"
        )
    for field, value in threshold_suggestions.items():
        _finite_nonnegative(value, field=f"threshold_suggestions.{field}")
    return {**payload, "prior_hash": prior_hash}


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise PolicyStateValidationError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyStateValidationError(
            f"{field} must be a finite number"
        ) from exc
    if not math.isfinite(number) or number < 0.0 or number > 1_000_000.0:
        raise PolicyStateValidationError(
            f"{field} must be finite and within the policy weight bound"
        )
    return number


def replay_policy_posterior_update(
    previous_weights: dict[str, Any],
    reward_json: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically replay the bounded nine-arm posterior update."""

    if set(previous_weights) != set(POLICY_ARMS):
        raise PolicyStateValidationError(
            "Policy predecessor arm set does not match the local allowlist"
        )
    prior = {
        arm: _finite_nonnegative(
            previous_weights[arm], field=f"previous_weights.{arm}"
        )
        for arm in POLICY_ARMS
    }
    ratio_fields = (
        "retrieval_hit",
        "context_precision",
        "context_recall",
        "concept_path_accuracy",
        "citation_pass_rate",
        "answer_groundedness",
        "answer_completeness",
        "repair_success_rate",
        "agent_typed_action_validation_pass_rate",
        "drift_rate",
        "answer_acceptance_gate_pass",
    )
    ratios: dict[str, float] = {}
    for field in ratio_fields:
        raw = reward_json.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PolicyStateValidationError(
                f"Policy reward metric {field} must be numeric"
            )
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise PolicyStateValidationError(
                f"Policy reward metric {field} must be within [0,1]"
            )
        ratios[field] = value

    counts: dict[str, int] = {}
    for field in (
        "claim_count",
        "supported_claim_count",
        "unsupported_claim_count",
    ):
        raw = reward_json.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PolicyStateValidationError(
                f"Policy reward count {field} must be a nonnegative integer"
            )
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            raise PolicyStateValidationError(
                f"Policy reward count {field} must be a nonnegative integer"
            )
        counts[field] = int(value)
    if (
        counts["supported_claim_count"]
        + counts["unsupported_claim_count"]
        != counts["claim_count"]
    ):
        raise PolicyStateValidationError(
            "Policy reward claim counts do not conserve the audited claim set"
        )

    costs: dict[str, float] = {}
    for field in ("latency_cost", "latency_ms", "task_token_cost"):
        raw = reward_json.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PolicyStateValidationError(
                f"Policy reward cost {field} must be finite and nonnegative"
            )
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise PolicyStateValidationError(
                f"Policy reward cost {field} must be finite and nonnegative"
            )
        costs[field] = value
    if not math.isclose(
        costs["latency_cost"],
        costs["latency_ms"] / 1000.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise PolicyStateValidationError(
            "Policy reward latency seconds do not replay from latency_ms"
        )
    if ratios["answer_acceptance_gate_pass"] == 0.0 and any(
        ratios[field] != 0.0
        for field in (
            "citation_pass_rate",
            "answer_groundedness",
            "answer_completeness",
            "repair_success_rate",
        )
    ):
        raise PolicyStateValidationError(
            "Rejected answer reward retained grounded-answer credit"
        )

    signals = {
        "retrieval_hit": ratios["retrieval_hit"],
        "context_precision": ratios["context_precision"],
        "context_recall": ratios["context_recall"],
        "concept_path_accuracy": ratios["concept_path_accuracy"],
        "citation_pass_rate": ratios["citation_pass_rate"],
        "answer_groundedness": ratios["answer_groundedness"],
        "answer_completeness": ratios["answer_completeness"],
        "repair_success": ratios["repair_success_rate"],
        "typed_action_validation_pass": ratios[
            "agent_typed_action_validation_pass_rate"
        ],
        "drift_quality": 1.0 - ratios["drift_rate"],
        "answer_acceptance": ratios["answer_acceptance_gate_pass"],
        "claim_support_quality": (
            counts["supported_claim_count"] / counts["claim_count"]
            if counts["claim_count"]
            else 0.0
        ),
        "latency_quality": 1.0 / (1.0 + costs["latency_cost"]),
        "task_token_quality": 1.0
        / (
            1.0
            + costs["task_token_cost"]
            / POLICY_TASK_TOKEN_NORMALIZATION_SCALE
        ),
    }
    arm_rewards = {
        arm: round(
            sum(signals[name] for name in POLICY_ARM_REWARD_SIGNALS[arm])
            / len(POLICY_ARM_REWARD_SIGNALS[arm]),
            8,
        )
        for arm in POLICY_ARMS
    }
    unnormalized = {
        arm: max(
            0.1,
            prior[arm]
            * math.exp(
                POLICY_POSTERIOR_LEARNING_RATE
                * (arm_rewards[arm] - 0.5)
            ),
        )
        for arm in POLICY_ARMS
    }
    mean_weight = sum(unnormalized.values()) / len(POLICY_ARMS)
    weights = {
        arm: round(unnormalized[arm] / max(mean_weight, 1e-12), 8)
        for arm in POLICY_ARMS
    }
    safe_arms = [arm for arm in POLICY_ARMS if weights[arm] >= 0.5]
    if not safe_arms:
        safe_arms = [
            "high_precision_direct_chunk",
            "low_latency_minimal_context",
        ]
    return {
        "protocol_version": POLICY_POSTERIOR_UPDATE_PROTOCOL_VERSION,
        "learning_rate": POLICY_POSTERIOR_LEARNING_RATE,
        "normalized_signals": {
            key: round(value, 8) for key, value in sorted(signals.items())
        },
        "arm_reward_observations": arm_rewards,
        "weights": weights,
        "safe_arms": safe_arms,
    }


def _normalized_action_priors(
    weights: dict[str, float], safe_arms: list[str]
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for arm in safe_arms:
        for action in POLICY_ARM_ACTIONS[arm]:
            raw[action] = raw.get(action, 0.0) + weights[arm]
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {
        action: round(value / total, 8)
        for action, value in sorted(raw.items())
    }


def _budget_suggestions(
    envelope: dict[str, Any],
    weights: dict[str, float],
    *,
    policy_applied: bool,
) -> dict[str, int]:
    max_weight = max(weights.values(), default=1.0) or 1.0
    suggestions: dict[str, int] = {}
    for field, arms in POLICY_BUDGET_ARM_MAP.items():
        hard_limit = int(envelope.get(field) or 0)
        if hard_limit <= 0:
            suggestions[field] = 0
            continue
        if not policy_applied:
            suggestions[field] = hard_limit
            continue
        arm_ratio = sum(weights[arm] / max_weight for arm in arms) / len(arms)
        # Policy may suggest a smaller soft operating point, but never exceed
        # or redefine the Runtime Settings hard interrupt.
        suggestions[field] = max(
            1,
            min(hard_limit, int(round(hard_limit * (0.5 + 0.5 * arm_ratio)))),
        )
    return suggestions


def validate_persisted_policy_state(
    db: Session,
    row: PolicyState,
    *,
    knowledge_base_id: str,
    _validation_context: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[str], dict[str, Any], dict[str, Any]]:
    """Replay the latest state and its reward linkage before active use.

    A content hash alone is insufficient because the UUID-free canonical hash
    intentionally excludes ``last_reward_event_id``.  The reciprocal
    RewardEvent link is therefore checked separately for learned states.
    """

    row_id = str(row.id or "")
    if not row_id:
        raise PolicyStateValidationError(
            "PolicyState predecessor chain is missing an id"
        )
    if str(row.knowledge_base_id) != str(knowledge_base_id):
        raise PolicyStateValidationError(
            "latest PolicyState belongs to another knowledge base"
        )
    validation_context = _validation_context
    if validation_context is None:
        persisted_state_count = int(
            db.scalar(
                select(func.count(PolicyState.id)).where(
                    PolicyState.knowledge_base_id == knowledge_base_id,
                    PolicyState.policy_family == POLICY_FAMILY,
                    PolicyState.policy_version == POLICY_VERSION,
                )
            )
            or 0
        )
        if persisted_state_count < 1:
            raise PolicyStateValidationError(
                "PolicyState predecessor chain has no persisted replay scope"
            )
        validation_context = {
            "max_states": persisted_state_count,
            "active_state_ids": set(),
            "memo": {},
        }
    active_state_ids = validation_context["active_state_ids"]
    memo = validation_context["memo"]
    if row_id in active_state_ids:
        raise PolicyStateValidationError(
            "PolicyState predecessor chain contains a cycle"
        )
    memoized = memo.get(row_id)
    if memoized is not None:
        return memoized
    if len(active_state_ids) >= int(validation_context["max_states"]):
        raise PolicyStateValidationError(
            "PolicyState predecessor chain exceeds its persisted replay scope"
        )
    active_state_ids.add(row_id)
    canonical_hash = canonical_policy_state_hash_for_row(row)
    if row.state_hash != canonical_hash:
        raise PolicyStateValidationError(
            "latest PolicyState failed canonical state-hash replay"
        )
    if row.policy_family != POLICY_FAMILY or row.policy_version != POLICY_VERSION:
        raise PolicyStateValidationError(
            "latest PolicyState uses an unsupported family or policy version"
        )

    raw_weights = dict(row.weights_json or {})
    if set(raw_weights) != set(POLICY_ARMS):
        raise PolicyStateValidationError(
            "latest PolicyState arm set does not match the local allowlist"
        )
    weights = {
        arm: _finite_nonnegative(raw_weights[arm], field=f"weights.{arm}")
        for arm in POLICY_ARMS
    }
    exploration = dict(row.exploration_json or {})
    reward_summary = dict(row.reward_summary_json or {})
    exploration_safe_arms = exploration.get("safe_arms")
    summary_safe_arms = reward_summary.get("safe_arms")
    if not isinstance(exploration_safe_arms, list) or not isinstance(
        summary_safe_arms, list
    ):
        raise PolicyStateValidationError(
            "latest PolicyState is missing its safe-arm witnesses"
        )
    safe_arms = sorted({str(arm) for arm in exploration_safe_arms})
    if (
        not safe_arms
        or len(safe_arms) != len(exploration_safe_arms)
        or safe_arms != sorted({str(arm) for arm in summary_safe_arms})
        or len(safe_arms) != len(summary_safe_arms)
        or any(arm not in POLICY_ARMS for arm in safe_arms)
    ):
        raise PolicyStateValidationError(
            "latest PolicyState safe arms are empty, divergent, duplicated, or unknown"
        )
    posterior = reward_summary.get("posterior")
    if not isinstance(posterior, dict) or posterior != row.weights_json:
        raise PolicyStateValidationError(
            "latest PolicyState posterior does not match its frozen weights"
        )

    constraints = dict(row.constraints_json or {})
    if (
        constraints.get("fallback_disabled") is not True
        or constraints.get("citation_verification_required") is not True
        or constraints.get("planner_replacement") is not False
        or constraints.get("gray_zone_decision_authority") is not False
        or constraints.get("gray_zone_rule_inputs_modified") is not False
        or type(constraints.get("gray_zone_model_call_count")) is not int
        or constraints.get("gray_zone_model_call_count") != 0
        or exploration.get("threshold_suggestions_runtime_lifecycle_accepted")
        is not False
        or exploration.get("threshold_suggestions_applied") is not False
        or exploration.get("gray_zone_decision_authority") is not False
        or type(exploration.get("gray_zone_model_call_count")) is not int
        or exploration.get("gray_zone_model_call_count") != 0
    ):
        raise PolicyStateValidationError(
            "latest PolicyState violates immutable planner/gray-zone constraints"
        )
    recorded_runtime_hash = str(
        constraints.get("runtime_settings_hash") or ""
    )
    summary_runtime_hash = str(
        reward_summary.get("runtime_settings_hash") or ""
    )
    recorded_envelope_hash = str(
        reward_summary.get("agent_operating_envelope_hash") or ""
    )
    if (
        not _is_sha256(recorded_runtime_hash)
        or recorded_runtime_hash != summary_runtime_hash
        or not _is_sha256(recorded_envelope_hash)
        or stable_hash(dict(constraints.get("agent_operating_envelope") or {}))
        != recorded_envelope_hash
        or reward_summary.get("policy_version") != POLICY_VERSION
    ):
        raise PolicyStateValidationError(
            "latest PolicyState runtime/envelope identity witnesses diverge"
        )

    last_reward_event_id = reward_summary.get("last_reward_event_id")
    reward_history = reward_summary.get("reward_history_tail")
    origin = reward_summary.get("origin")
    previous_policy_state_id = reward_summary.get(
        "previous_policy_state_id"
    )
    previous_policy_state_hash = reward_summary.get(
        "previous_policy_state_hash"
    )
    if not isinstance(reward_history, list) or len(reward_history) > 8:
        raise PolicyStateValidationError(
            "latest PolicyState reward history is malformed"
        )
    if last_reward_event_id:
        if (
            origin != "reward_update"
            or not str(previous_policy_state_id or "")
            or not _is_sha256(previous_policy_state_hash)
        ):
            raise PolicyStateValidationError(
                "learned PolicyState has invalid origin or predecessor identity"
            )
        predecessor = db.get(PolicyState, str(previous_policy_state_id))
        if predecessor is not None and str(predecessor.id) == row_id:
            raise PolicyStateValidationError(
                "PolicyState predecessor chain contains a cycle"
            )
        if (
            predecessor is None
            or str(predecessor.knowledge_base_id) != str(knowledge_base_id)
            or predecessor.policy_family != POLICY_FAMILY
            or predecessor.policy_version != POLICY_VERSION
            or predecessor.state_hash != str(previous_policy_state_hash)
        ):
            raise PolicyStateValidationError(
                "learned PolicyState predecessor id/hash cannot be replayed"
            )
        predecessor_weights, _predecessor_safe_arms, _predecessor_exploration, predecessor_summary = (
            validate_persisted_policy_state(
                db,
                predecessor,
                knowledge_base_id=knowledge_base_id,
                _validation_context=validation_context,
            )
        )
        reward = db.get(RewardEvent, str(last_reward_event_id))
        if (
            reward is None
            or str(reward.knowledge_base_id) != str(knowledge_base_id)
            or str(reward.policy_state_id or "") != str(row.id)
            or dict(reward.reward_json or {})
            != dict(reward_summary.get("last_reward") or {})
            or not reward_history
            or dict(reward_history[-1] or {}) != dict(reward.reward_json or {})
        ):
            raise PolicyStateValidationError(
                "latest PolicyState failed reciprocal RewardEvent replay"
            )
        diagnostics = dict(reward.diagnostics_json or {})
        reward_json = dict(reward.reward_json or {})
        try:
            reward_replay = replay_policy_reward_event(db, reward)
        except Exception as exc:
            raise PolicyStateValidationError(
                "latest PolicyState reward evidence failed persisted replay"
            ) from exc
        if (
            diagnostics.get("source") != "context_graph_agent_v1"
            or diagnostics.get("runtime_settings_hash")
            != recorded_runtime_hash
            or diagnostics.get("agent_operating_envelope_hash")
            != recorded_envelope_hash
            or diagnostics.get("reward_metrics_protocol_version")
            != reward_json.get("reward_metrics_protocol_version")
            or diagnostics.get("reward_metric_evidence_hash")
            != reward_json.get("reward_metric_evidence_hash")
            or reward_json.get("reward_metrics_protocol_version")
            != POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION
            or reward_summary.get("last_reward_fact_protocol_version")
            != POLICY_REWARD_FACT_PROTOCOL_VERSION
            or reward_summary.get("last_reward_fact_hash")
            != reward_replay.get("reward_fact_hash")
        ):
            raise PolicyStateValidationError(
                "latest PolicyState reward audit witnesses diverge"
            )
        predecessor_constraints = dict(predecessor.constraints_json or {})
        effective_predecessor_weights = (
            predecessor_weights
            if predecessor_constraints.get("runtime_settings_hash")
            == recorded_runtime_hash
            and predecessor_summary.get("agent_operating_envelope_hash")
            == recorded_envelope_hash
            else {arm: 1.0 for arm in POLICY_ARMS}
        )
        posterior_replay = replay_policy_posterior_update(
            effective_predecessor_weights,
            reward_json,
        )
        expected_weights = dict(posterior_replay["weights"])
        expected_safe_arms = list(posterior_replay["safe_arms"])
        predecessor_history = list(
            predecessor_summary.get("reward_history_tail") or []
        )
        expected_history = predecessor_history[-7:] + [reward_json]
        expected_drift_status = (
            "normal"
            if float(reward_json.get("drift_rate") or 0.0) <= 0.2
            else "elevated"
        )
        if (
            weights != expected_weights
            or reward_summary.get("posterior") != expected_weights
            or reward_summary.get("posterior_proxy") != expected_weights
            or safe_arms != sorted(expected_safe_arms)
            or sorted(reward_summary.get("safe_arms") or [])
            != sorted(expected_safe_arms)
            or reward_summary.get("posterior_update_protocol_version")
            != POLICY_POSTERIOR_UPDATE_PROTOCOL_VERSION
            or reward_summary.get("posterior_learning_rate")
            != POLICY_POSTERIOR_LEARNING_RATE
            or reward_summary.get("arm_reward_observations")
            != posterior_replay["arm_reward_observations"]
            or reward_summary.get("normalized_reward_signals")
            != posterior_replay["normalized_signals"]
            or reward_summary.get("typed_action_validation_pass_rate")
            != reward_json.get("agent_typed_action_validation_pass_rate")
            or reward_summary.get("reward_metrics_protocol_version")
            != reward_json.get("reward_metrics_protocol_version")
            or reward_summary.get("reward_metric_evidence_hash")
            != reward_json.get("reward_metric_evidence_hash")
            or reward_summary.get("reward_history_tail") != expected_history
            or reward_summary.get("drift_status") != expected_drift_status
            or exploration.get("epsilon") != 0.05
            or exploration.get("exploration_rate") != 0.05
            or reward_summary.get("exploration_rate") != 0.05
        ):
            raise PolicyStateValidationError(
                "learned PolicyState posterior/update/history failed deterministic reward replay"
            )
        if (
            predecessor.created_at is not None
            and row.created_at is not None
            and predecessor.created_at > row.created_at
        ):
            raise PolicyStateValidationError(
                "learned PolicyState predecessor was created after its child"
            )
    elif (
        origin != "seed"
        or previous_policy_state_id is not None
        or previous_policy_state_hash is not None
        or reward_history
        or reward_summary.get("last_reward") is not None
        or any(abs(weights[arm] - 1.0) > 1e-12 for arm in POLICY_ARMS)
        or safe_arms != sorted(POLICY_ARMS)
    ):
        raise PolicyStateValidationError(
            "non-neutral PolicyState has no reciprocal RewardEvent"
        )
    result = weights, safe_arms, exploration, reward_summary
    active_state_ids.remove(row_id)
    memo[row_id] = result
    return result


def replay_policy_operating_prior_card(
    db: Session,
    card: dict[str, Any],
    *,
    knowledge_base_id: str,
    runtime_settings_hash: str,
    agent_operating_envelope_hash: str,
    agent_operating_envelope: dict[str, Any],
) -> tuple[dict[str, Any], PolicyState | None]:
    """Rebuild a frozen prior from its exact referenced PolicyState.

    Unlike the live reader this does not select the latest row: reward/trace
    replay must remain stable after a newer posterior has been committed.
    """

    validated = validate_policy_operating_prior_card(
        card,
        knowledge_base_id=knowledge_base_id,
        agent_operating_envelope=agent_operating_envelope,
        runtime_settings_hash=runtime_settings_hash,
        agent_operating_envelope_hash=agent_operating_envelope_hash,
    )
    state_id = validated.get("policy_state_id")
    row: PolicyState | None = None
    exploration: dict[str, Any] = {}
    if state_id is None:
        weights = {arm: 1.0 for arm in POLICY_ARMS}
        safe_arms = list(POLICY_ARMS)
        identity_matches = False
        inactive_reason = "no_policy_state"
        state_hash = None
    else:
        row = db.get(PolicyState, str(state_id))
        if row is None or str(row.knowledge_base_id) != str(knowledge_base_id):
            raise PolicyStateValidationError(
                "Policy operating-prior references a missing or cross-KB state"
            )
        weights, safe_arms, exploration, reward_summary = (
            validate_persisted_policy_state(
                db,
                row,
                knowledge_base_id=knowledge_base_id,
            )
        )
        if str(row.state_hash or "") != str(
            validated.get("policy_state_hash") or ""
        ):
            raise PolicyStateValidationError(
                "Policy operating-prior state id/hash witnesses diverge"
            )
        constraints = dict(row.constraints_json or {})
        identity_matches = (
            {
                str(constraints.get("runtime_settings_hash") or ""),
                str(reward_summary.get("runtime_settings_hash") or ""),
            }
            == {runtime_settings_hash}
            and str(
                reward_summary.get("agent_operating_envelope_hash") or ""
            )
            == agent_operating_envelope_hash
        )
        inactive_reason = None if identity_matches else "runtime_identity_changed"
        state_hash = row.state_hash

    effective_weights = (
        weights
        if identity_matches
        else {arm: 1.0 for arm in POLICY_ARMS}
    )
    effective_safe_arms = (
        safe_arms if identity_matches else list(POLICY_ARMS)
    )
    threshold_suggestions: dict[str, float] = {}
    if identity_matches:
        for field in POLICY_THRESHOLD_FIELDS:
            if field in exploration:
                threshold_suggestions[field] = _finite_nonnegative(
                    exploration[field], field=f"exploration.{field}"
                )
    expected = {
        "protocol_version": POLICY_OPERATING_PRIOR_PROTOCOL_VERSION,
        "knowledge_base_id": str(knowledge_base_id),
        "policy_state_id": str(row.id) if row is not None else None,
        "policy_state_hash": state_hash,
        "policy_version": POLICY_VERSION,
        "applied": bool(identity_matches),
        "inactive_reason": inactive_reason,
        "safe_arms": effective_safe_arms,
        "arm_priors": {
            arm: round(effective_weights[arm], 8) for arm in POLICY_ARMS
        },
        "action_priors": _normalized_action_priors(
            effective_weights, effective_safe_arms
        ),
        "budget_suggestions": _budget_suggestions(
            agent_operating_envelope,
            effective_weights,
            policy_applied=identity_matches,
        ),
        "budget_suggestions_are_soft": True,
        "runtime_hard_interrupts_unchanged": True,
        "threshold_suggestions": threshold_suggestions,
        "threshold_suggestions_runtime_lifecycle_accepted": False,
        "threshold_suggestions_applied": False,
        "planner_replacement": False,
        "gray_zone_decision_authority": False,
        "gray_zone_rule_inputs_modified": False,
        "gray_zone_model_call_count": 0,
        "runtime_settings_hash": runtime_settings_hash,
        "agent_operating_envelope_hash": agent_operating_envelope_hash,
    }
    expected["prior_hash"] = stable_hash(expected)
    if validated != expected:
        raise PolicyStateValidationError(
            "Policy operating-prior does not replay from its exact state"
        )
    return validated, row


def read_policy_operating_prior(
    db: Session,
    knowledge_base_id: str,
    *,
    runtime_settings_hash: str,
    agent_operating_envelope_hash: str,
    agent_operating_envelope: dict[str, Any],
) -> dict[str, Any]:
    """Return a bounded advisory prior; never return gray-zone authority."""

    row = db.scalar(
        select(PolicyState)
        .where(
            PolicyState.knowledge_base_id == knowledge_base_id,
            PolicyState.policy_family == POLICY_FAMILY,
        )
        .order_by(PolicyState.created_at.desc(), PolicyState.id.desc())
        .limit(1)
    )
    if row is None:
        weights = {arm: 1.0 for arm in POLICY_ARMS}
        safe_arms = list(POLICY_ARMS)
        identity_matches = False
        inactive_reason = "no_policy_state"
        policy_state_hash = None
        policy_state_id = None
        exploration: dict[str, Any] = {}
    else:
        weights, safe_arms, exploration, reward_summary = (
            validate_persisted_policy_state(
                db,
                row,
                knowledge_base_id=knowledge_base_id,
            )
        )
        constraints = dict(row.constraints_json or {})
        recorded_runtime_hashes = {
            str(constraints.get("runtime_settings_hash") or ""),
            str(reward_summary.get("runtime_settings_hash") or ""),
        }
        recorded_envelope_hash = str(
            reward_summary.get("agent_operating_envelope_hash") or ""
        )
        identity_matches = (
            recorded_runtime_hashes == {runtime_settings_hash}
            and recorded_envelope_hash == agent_operating_envelope_hash
        )
        inactive_reason = None if identity_matches else "runtime_identity_changed"
        policy_state_hash = row.state_hash
        policy_state_id = row.id

    effective_weights = (
        weights
        if identity_matches
        else {arm: 1.0 for arm in POLICY_ARMS}
    )
    effective_safe_arms = (
        safe_arms if identity_matches else list(POLICY_ARMS)
    )
    action_priors = _normalized_action_priors(
        effective_weights, effective_safe_arms
    )
    suggestions = _budget_suggestions(
        agent_operating_envelope,
        effective_weights,
        policy_applied=identity_matches,
    )
    threshold_suggestions: dict[str, float] = {}
    for field in POLICY_THRESHOLD_FIELDS:
        if identity_matches and field in exploration:
            threshold_suggestions[field] = _finite_nonnegative(
                exploration[field], field=f"exploration.{field}"
            )
    card = {
        "protocol_version": POLICY_OPERATING_PRIOR_PROTOCOL_VERSION,
        "knowledge_base_id": str(knowledge_base_id),
        "policy_state_id": policy_state_id,
        "policy_state_hash": policy_state_hash,
        "policy_version": POLICY_VERSION,
        "applied": bool(identity_matches),
        "inactive_reason": inactive_reason,
        "safe_arms": effective_safe_arms,
        "arm_priors": {
            arm: round(effective_weights[arm], 8) for arm in POLICY_ARMS
        },
        "action_priors": action_priors,
        "budget_suggestions": suggestions,
        "budget_suggestions_are_soft": True,
        "runtime_hard_interrupts_unchanged": True,
        "threshold_suggestions": threshold_suggestions,
        "threshold_suggestions_runtime_lifecycle_accepted": False,
        "threshold_suggestions_applied": False,
        "planner_replacement": False,
        "gray_zone_decision_authority": False,
        "gray_zone_rule_inputs_modified": False,
        "gray_zone_model_call_count": 0,
        "runtime_settings_hash": runtime_settings_hash,
        "agent_operating_envelope_hash": agent_operating_envelope_hash,
    }
    card["prior_hash"] = stable_hash(card)
    return validate_policy_operating_prior_card(
        card,
        knowledge_base_id=knowledge_base_id,
        agent_operating_envelope=agent_operating_envelope,
        runtime_settings_hash=runtime_settings_hash,
        agent_operating_envelope_hash=agent_operating_envelope_hash,
    )
