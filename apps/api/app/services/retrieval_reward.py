"""Audited pre-generation feedback; never consumes model self-scores."""
from copy import deepcopy
from datetime import datetime, timezone
import math

from sqlalchemy import select

from app.models import KnowledgeBase, RetrievalLexicalPolicy, RetrievalLexicalReward
from app.retrieval_control_contracts import control_hash
from app.core.config import get_settings

OPERATIONS = frozenset({"replace_surface", "add_attested_alias", "qualify", "locator_probe", "restore_context"})


def operation_arm(operations):
    if not operations or len(operations) > 2 or set(operations) - OPERATIONS:
        raise ValueError("retrieval_reward_operation_scope_invalid")
    return "+".join(sorted(operations))


def build_reward_observation(*, features, baseline, operations, extra_elapsed_seconds,
                             source_integrity, time_weight, work_weight, time_scale_seconds):
    if not math.isfinite(extra_elapsed_seconds) or extra_elapsed_seconds < 0:
        raise ValueError("retrieval_reward_elapsed_invalid")
    if not 0 <= time_weight <= 1 or not 0 <= work_weight <= 1 or not 1 <= time_scale_seconds <= 600:
        raise ValueError("retrieval_reward_cost_protocol_invalid")
    same = bool(baseline and baseline.task_hash == features.task_hash
                and baseline.evaluation_protocol_hash == features.evaluation_protocol_hash)
    delta_lower = features.utility.lower - baseline.utility.upper if same else 0
    delta_upper = features.utility.upper - baseline.utility.lower if same else 0
    previous = {item.facet_id: item for item in baseline.facets} if same else {}
    regressed = same and any(item.coverage.lower + 1e-9 < previous[item.facet_id].coverage.lower
                            for item in features.facets)
    known = same and abs(delta_upper - delta_lower) < 1e-6
    arm = operation_arm(operations) if operations else None
    cost = time_weight * extra_elapsed_seconds / time_scale_seconds + work_weight * bool(operations)
    return {"protocol_version": "retrieval_lexical_reward_v1", "task_hash": features.task_hash,
        "before_feature_hash": control_hash(baseline.model_dump(mode="json")) if baseline else None,
        "after_feature_hash": control_hash(features.model_dump(mode="json")),
        "evaluation_protocol_hash": features.evaluation_protocol_hash,
        "delta_lower": delta_lower, "delta_upper": delta_upper,
        "extra_elapsed_seconds": extra_elapsed_seconds,
        "additional_retrieval_count": int(any(operation != "restore_context" for operation in operations)),
        "cost_protocol": {"time_weight": time_weight, "work_weight": work_weight,
                          "time_scale_seconds": time_scale_seconds},
        "cost": cost, "reward_lower": max(-1, min(1, delta_lower - cost)),
        "reward_upper": max(-1, min(1, delta_upper - cost)),
        "operations": sorted(operations), "operation_arm": arm, "credit_scope": "whole_patch",
        "required_coverage_regressed": bool(regressed),
        "training_eligible": bool(source_integrity and operations and known and not regressed),
        "model_self_score_weight": 0, "answer_correctness_claimed": False, "phase": "before_answer_generation"}


def policy_identity(kb, revision, counts):
    return control_hash({"protocol_version": "retrieval_lexical_policy_v1", "knowledge_base_id": kb,
                         "revision": revision, "operation_counts": counts})


def read_lexical_policy(db, knowledge_base_id):
    row = db.get(RetrievalLexicalPolicy, knowledge_base_id)
    if row is None:
        return {"state_hash": policy_identity(knowledge_base_id, 0, {}), "operation_priors": {}}
    if (row.protocol_version != "retrieval_lexical_policy_v1"
        or any(operation_arm(key.split("+")) != key for key in row.operation_counts_json)
        or row.state_hash != policy_identity(knowledge_base_id, row.revision, row.operation_counts_json)):
        raise ValueError("retrieval_lexical_policy_identity_invalid")
    priors = {}
    for operation, counts in row.operation_counts_json.items():
        if set(counts) != {"improved", "not_improved"} or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("retrieval_lexical_policy_counts_invalid")
        priors[operation] = (1 + counts["improved"]) / (2 + sum(counts.values()))
    return {"state_hash": row.state_hash, "operation_priors": priors}


def record_retrieval_reward(db, *, run, package, features, baseline, attempt_index, operations,
                            extra_elapsed_seconds, source_integrity):
    if set(operations) - OPERATIONS or package.knowledge_base_id != run.knowledge_base_id:
        raise ValueError("retrieval_reward_scope_invalid")
    db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == run.knowledge_base_id).with_for_update())
    old = db.scalar(select(RetrievalLexicalReward).where(
        RetrievalLexicalReward.run_id == run.id, RetrievalLexicalReward.attempt_index == attempt_index))
    settings = get_settings()
    observation = build_reward_observation(features=features, baseline=baseline, operations=operations,
        extra_elapsed_seconds=extra_elapsed_seconds, source_integrity=source_integrity,
        time_weight=settings.retrieval_reward_time_weight, work_weight=settings.retrieval_reward_work_weight,
        time_scale_seconds=settings.retrieval_reward_time_scale_seconds)
    digest = control_hash(observation)
    if old:
        # Timing is measured once. Replaying the same completed attempt neither
        # charges additional elapsed time nor updates the policy a second time.
        variable = {"extra_elapsed_seconds", "cost", "reward_lower", "reward_upper"}
        if (old.observation_hash != control_hash(old.observation_json) or
            {k: v for k, v in old.observation_json.items() if k not in variable} !=
            {k: v for k, v in observation.items() if k not in variable}):
            raise ValueError("retrieval_reward_attempt_replay_changed")
        return old
    row = RetrievalLexicalReward(run_id=run.id, knowledge_base_id=run.knowledge_base_id,
        context_package_id=package.id, attempt_index=attempt_index,
        observation_json=observation, observation_hash=digest)
    db.add(row)
    if observation["training_eligible"]:
        read_lexical_policy(db, run.knowledge_base_id)
        policy = db.get(RetrievalLexicalPolicy, run.knowledge_base_id)
        if policy is None:
            policy = RetrievalLexicalPolicy(knowledge_base_id=run.knowledge_base_id,
                revision=0, operation_counts_json={}, state_hash=policy_identity(run.knowledge_base_id, 0, {}))
            db.add(policy)
        counts = deepcopy(policy.operation_counts_json)
        outcome = "improved" if observation["reward_lower"] > 1e-6 else "not_improved"
        counts.setdefault(observation["operation_arm"], {"improved": 0, "not_improved": 0})[outcome] += 1
        policy.revision += 1
        policy.operation_counts_json = counts
        policy.state_hash = policy_identity(run.knowledge_base_id, policy.revision, counts)
        policy.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.flush()
    return row
