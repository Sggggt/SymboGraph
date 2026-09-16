"""Replayable structural proxy rewards; model self-scores have zero weight."""
from __future__ import annotations

import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentObservation, AgentPlan, AgentRun, AnswerSession, AnswerSourceBinding, ContextPackage, KnowledgeBase, PolicyState, RewardEvent
from app.reflection_contracts import AnswerDraft, PathSupportMetrics
from app.services.agent_reflection import ReflectionContractError, reflection_hash, render_answer_units, generation_calls_from_events
from app.services.answer_sources import answer_source_path_metrics, audit_answer_sources, build_answer_evidence_manifest, source_binding_citations
from app.services.context_graph import context_package_to_contexts
from app.services.chunking import stable_hash
from app.services.graph_state_hashes import canonical_policy_state_hash, canonical_policy_state_hash_for_row
from app.services.storage import raise_if_source_io_cancelled


PROTOCOL = "answer_reflection_reward_v1"
POLICY_ORIGIN = "answer_reflection_reward_update_v1"


def recorded_reward_path_protocol(events: list[dict[str, Any]], run_metadata: dict[str, Any]) -> str:
    try:
        protocols = set()
        for event in events:
            if event.get("stage") != "reflection_gate":
                continue
            payload = event["gate"]["path_metrics"]
            if not isinstance(payload, dict) or payload.get("protocol_version") not in {"answer_path_support_score_v1", "answer_path_support_score_v2"}:
                raise ValueError("recorded path protocol missing or unknown")
            protocols.add(PathSupportMetrics.model_validate(payload).protocol_version)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReflectionContractError("reflection_reward_path_protocol_invalid") from exc
    recorded = run_metadata.get("path_support_protocol", "answer_path_support_score_v1")
    if len(protocols) != 1 or recorded not in protocols:
        raise ReflectionContractError("reflection_reward_path_protocol_mismatch")
    return recorded


def replay_reflection_reward(db: Session, reward: RewardEvent) -> dict[str, Any]:
    raise_if_source_io_cancelled()
    answer = db.get(AnswerSession, reward.answer_session_id)
    package = db.get(ContextPackage, (reward.context_json or {}).get("context_package_id"))
    run = db.get(AgentRun, (reward.context_json or {}).get("agent_run_id"))
    if answer is None or package is None or run is None or len({answer.knowledge_base_id, package.knowledge_base_id, run.knowledge_base_id, reward.knowledge_base_id}) != 1:
        raise ReflectionContractError("reflection_reward_scope_mismatch")
    if (answer.context_package_id != package.id or reward.retrieval_trace_id != package.retrieval_trace_id
        or answer.retrieval_trace_id != package.retrieval_trace_id or answer.qa_session_id != run.session_id
        or answer.question != run.question or (run.metadata_json or {}).get("answer_session_id") != answer.id):
        raise ReflectionContractError("reflection_reward_answer_scope_mismatch")
    audit = answer.diagnostics_json.get("answer_reflection") or {}
    if audit.get("audit_hash") != reflection_hash({key: value for key, value in audit.items() if key != "audit_hash"}):
        raise ReflectionContractError("reflection_reward_audit_hash_mismatch")
    ledger = (run.metadata_json or {}).get("reflection_run_ledger") or {}
    if (ledger.get("protocol_version") != "reflection_run_ledger_v1" or audit.get("run_ledger_hash") != reflection_hash(ledger)
        or audit.get("round_budget") != ledger.get("hard_limit")):
        raise ReflectionContractError("reflection_reward_run_ledger_mismatch")
    from app.services.reflection_run import ordered_reflection_events
    recorded_events = ordered_reflection_events(db, run_id=run.id)
    if recorded_events != audit.get("events"):
        raise ReflectionContractError("reflection_reward_observation_replay_mismatch")
    path_protocol = recorded_reward_path_protocol(recorded_events, run.metadata_json or {})
    draft = AnswerDraft.model_validate(answer.diagnostics_json["structured_answer"])
    rendered, units = render_answer_units(draft)
    if rendered != answer.answer:
        raise ReflectionContractError("reflection_reward_answer_changed")
    rows = list(db.scalars(select(AnswerSourceBinding).where(AnswerSourceBinding.answer_session_id == answer.id)))
    source_binding_citations(answer_session=answer, package=package, rows=rows, reflection_audit_hash=audit["audit_hash"])
    contexts = context_package_to_contexts(package)
    evidence = build_answer_evidence_manifest(package, contexts)
    raise_if_source_io_cancelled()
    _candidates, source_audit = audit_answer_sources(db, knowledge_base_id=run.knowledge_base_id,
        package=package, contexts=contexts, draft=draft, evidence=evidence, unit_limit=32)
    if not source_audit["all_valid"]:
        raise ReflectionContractError("reflection_reward_source_replay_failed")
    metrics = answer_source_path_metrics(db, package=package, draft=draft, evidence=evidence,
        source_audit=source_audit, _replay_protocol=path_protocol)
    factual_ids = {unit["unit_id"] for unit in units if unit["kind"] == "factual"}
    bound_ids = {row.unit_id for row in rows}
    if bound_ids != factual_ids:
        raise ReflectionContractError("reflection_reward_unit_coverage_mismatch")
    accepted = audit.get("outcome") in {"accepted_without_reflection", "accepted_after_reflection"}
    plans = list(db.scalars(select(AgentPlan).where(AgentPlan.run_id == run.id).order_by(AgentPlan.plan_index.asc(), AgentPlan.id.asc())))
    source_ids = {row.chunk_id for row in rows}
    source_items = [item for item in package.package_json["chunks"] if item["chunk_id"] in source_ids]
    result = {
        "protocol_version": PROTOCOL,
        "source_binding_rate": 1.0 if factual_ids else 0.0,
        "path_support_score": metrics.path_score,
        "path_coverage": metrics.coverage,
        "restored_source_fraction": len(source_ids.intersection(package.restored_chunk_ids_json or [])) / max(len(source_ids), 1),
        "bridge_source_fraction": len(source_ids.intersection(package.bridge_chunk_ids_json or [])) / max(len(source_ids), 1),
        "source_document_count": len({item["document_id"] for item in source_items}),
        "action_validation_rate": sum(plan.validation_json.get("valid") is True for plan in plans) / max(len(plans), 1),
        "generation_calls": generation_calls_from_events(audit.get("events", [])),
        "reflection_calls": sum(event.get("stage") == "answer_reflection" and event.get("status") == "completed" for event in audit.get("events", [])),
        "self_assessment_reward_weight": 0.0,
        "training_eligible": bool(accepted and factual_ids and answer.model_json.get("policy_update_eligible")),
    }
    result["evidence_hash"] = reflection_hash({
        "protocol": PROTOCOL, "metrics": result, "bindings": sorted((row.id, row.binding_hash) for row in rows),
        "answer_hash": audit["delivered_answer_hash"], "plan_validations": [plan.validation_json for plan in plans],
    })
    return result


def reflection_posterior(previous: dict[str, float], metrics: dict[str, Any]) -> dict[str, float]:
    from app.services.policy import POLICY_ARMS
    if set(previous) != set(POLICY_ARMS) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in previous.values()):
        raise ReflectionContractError("reflection_policy_predecessor_invalid")
    if not metrics["training_eligible"]:
        return dict(previous)
    quality = metrics["source_binding_rate"] * metrics["action_validation_rate"]
    precision = quality * (metrics["path_support_score"] or 0.0) * metrics["path_coverage"]
    extra = max(0, metrics["generation_calls"] + metrics["reflection_calls"] - 1)
    signals = {
        "high_precision_direct_chunk": precision,
        "structure_context_heavy": quality * (0.5 + 0.5 * metrics["restored_source_fraction"]),
        "rq_membership_expansion": quality * metrics["path_coverage"],
        "mid_concept_expansion": quality * metrics["path_coverage"],
        "coarse_to_mid_drilldown": quality * metrics["path_coverage"],
        "bridge_edge_exploration": quality * metrics["bridge_source_fraction"],
        "formula_table_closure": quality * metrics["restored_source_fraction"],
        "cross_document_synthesis": quality * min(1.0, max(0, metrics["source_document_count"] - 1) / 2.0),
        "low_latency_minimal_context": quality / (1.0 + extra),
    }
    updated = {arm: float(previous[arm]) * math.exp(0.1 * max(-1.0, signals[arm] - min(extra, 10) * 0.05)) for arm in POLICY_ARMS}
    scale = sum(updated.values()) / len(updated) or 1.0
    return {arm: round(value / scale, 8) for arm, value in updated.items()}


def validate_reflection_policy_state(db: Session, row: PolicyState, *, knowledge_base_id: str):
    from app.services.policy import POLICY_ARMS, POLICY_FAMILY, POLICY_VERSION
    chain = []
    seen = set()
    current = row
    while current is not None:
        raise_if_source_io_cancelled()
        if current.id in seen or len(seen) > 10000:
            raise ReflectionContractError("reflection_policy_chain_invalid")
        seen.add(current.id)
        if current.knowledge_base_id != knowledge_base_id or current.policy_family != POLICY_FAMILY or current.policy_version != POLICY_VERSION or current.state_hash != canonical_policy_state_hash_for_row(current):
            raise ReflectionContractError("reflection_policy_state_identity_invalid")
        summary = current.reward_summary_json or {}
        if summary.get("origin") != POLICY_ORIGIN or summary.get("posterior") != current.weights_json:
            raise ReflectionContractError("reflection_policy_summary_invalid")
        reward = db.get(RewardEvent, summary.get("last_reward_event_id"))
        if reward is None or reward.policy_state_id != current.id:
            raise ReflectionContractError("reflection_policy_reward_link_invalid")
        metrics = replay_reflection_reward(db, reward)
        if metrics != reward.reward_json or metrics != summary.get("last_reward"):
            raise ReflectionContractError("reflection_policy_reward_replay_mismatch")
        constraints = current.constraints_json or {}
        exploration = current.exploration_json or {}
        if (constraints.get("answer_reflection_protocol") != "agent_answer_reflection_v1"
            or constraints.get("fallback_disabled") is not True or constraints.get("source_binding_required") is not True
            or constraints.get("planner_replacement") is not False or constraints.get("gray_zone_decision_authority") is not False
            or constraints.get("gray_zone_rule_inputs_modified") is not False or type(constraints.get("gray_zone_model_call_count")) is not int or constraints.get("gray_zone_model_call_count") != 0
            or exploration.get("gray_zone_decision_authority") is not False or type(exploration.get("gray_zone_model_call_count")) is not int or exploration.get("gray_zone_model_call_count") != 0
            or exploration.get("threshold_suggestions_applied") is not False
            or exploration.get("threshold_suggestions_runtime_lifecycle_accepted") is not False
            or exploration.get("safe_arms") != list(POLICY_ARMS) or summary.get("safe_arms") != list(POLICY_ARMS)
            or summary.get("self_assessment_is_reward_label") is not False
            or summary.get("runtime_settings_hash") != constraints.get("runtime_settings_hash")
            or summary.get("agent_operating_envelope_hash") != stable_hash(constraints.get("agent_operating_envelope") or {})):
            raise ReflectionContractError("reflection_policy_authority_invalid")
        chain.append((current, metrics))
        predecessor_id = summary.get("previous_policy_state_id")
        current = db.get(PolicyState, predecessor_id) if predecessor_id else None
        if predecessor_id and current is None:
            raise ReflectionContractError("reflection_policy_predecessor_missing")
    previous = {arm: 1.0 for arm in POLICY_ARMS}
    previous_row = None
    for current, metrics in reversed(chain):
        raise_if_source_io_cancelled()
        if current.reward_summary_json.get("previous_policy_state_hash") != (previous_row.state_hash if previous_row else None):
            raise ReflectionContractError("reflection_policy_predecessor_hash_mismatch")
        expected_reset = previous_row is None or any(
            previous_row.constraints_json.get(key) != current.constraints_json.get(key)
            for key in ("runtime_settings_hash", "agent_operating_envelope")
        )
        if current.reward_summary_json.get("reset_runtime_identity") is not expected_reset:
            raise ReflectionContractError("reflection_policy_reset_identity_mismatch")
        if expected_reset:
            previous = {arm: 1.0 for arm in POLICY_ARMS}
        expected = reflection_posterior(previous, metrics)
        if expected != current.weights_json:
            raise ReflectionContractError("reflection_policy_posterior_mismatch")
        previous, previous_row = expected, current
    return dict(row.weights_json), list(POLICY_ARMS), dict(row.exploration_json), dict(row.reward_summary_json)


def consume_reflection_reward(db: Session, reward: RewardEvent, *, envelope: dict[str, Any], runtime_hash: str) -> PolicyState | None:
    raise_if_source_io_cancelled()
    from app.services.policy import POLICY_ARMS, POLICY_FAMILY, POLICY_VERSION
    db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == reward.knowledge_base_id).with_for_update())
    if reward.policy_state_id is not None:
        raise ReflectionContractError("reflection_reward_already_consumed")
    metrics = replay_reflection_reward(db, reward)
    reward.reward_json = metrics
    reward.diagnostics_json = {**(reward.diagnostics_json or {}), "policy_consumption_status": "ineligible"}
    if not metrics["training_eligible"]:
        return None
    previous_row = db.scalar(select(PolicyState).where(PolicyState.knowledge_base_id == reward.knowledge_base_id,
        PolicyState.policy_family == POLICY_FAMILY, PolicyState.reward_summary_json["origin"].as_string() == POLICY_ORIGIN)
        .order_by(PolicyState.created_at.desc(), PolicyState.id.desc()).limit(1))
    previous = {arm: 1.0 for arm in POLICY_ARMS}
    reset = True
    if previous_row:
        weights, _, _, _ = validate_reflection_policy_state(db, previous_row, knowledge_base_id=reward.knowledge_base_id)
        reset = previous_row.constraints_json.get("runtime_settings_hash") != runtime_hash or previous_row.constraints_json.get("agent_operating_envelope") != envelope
        if not reset:
            previous = weights
    weights = reflection_posterior(previous, metrics)
    raise_if_source_io_cancelled()
    constraints = {"answer_reflection_protocol": "agent_answer_reflection_v1", "fallback_disabled": True,
        "source_binding_required": True, "planner_replacement": False, "gray_zone_decision_authority": False,
        "gray_zone_rule_inputs_modified": False, "gray_zone_model_call_count": 0,
        "runtime_settings_hash": runtime_hash, "agent_operating_envelope": envelope}
    exploration = {"epsilon": 0.05, "safe_arms": list(POLICY_ARMS), "threshold_suggestions_runtime_lifecycle_accepted": False,
        "threshold_suggestions_applied": False, "gray_zone_decision_authority": False, "gray_zone_model_call_count": 0}
    summary = {"origin": POLICY_ORIGIN, "policy_version": POLICY_VERSION, "posterior": weights, "safe_arms": list(POLICY_ARMS),
        "previous_policy_state_id": previous_row.id if previous_row else None,
        "previous_policy_state_hash": previous_row.state_hash if previous_row else None,
        "last_reward_event_id": reward.id, "last_reward": metrics, "runtime_settings_hash": runtime_hash,
        "agent_operating_envelope_hash": stable_hash(envelope), "reset_runtime_identity": reset,
        "self_assessment_is_reward_label": False}
    row = PolicyState(knowledge_base_id=reward.knowledge_base_id, policy_family=POLICY_FAMILY, policy_version=POLICY_VERSION,
        weights_json=weights, constraints_json=constraints, exploration_json=exploration, reward_summary_json=summary,
        state_hash=canonical_policy_state_hash(policy_family=POLICY_FAMILY, policy_version=POLICY_VERSION,
            profile_objective_hash=None, weights=weights, constraints=constraints, exploration=exploration, reward_summary=summary))
    db.add(row)
    db.flush()
    reward.policy_state_id = row.id
    reward.diagnostics_json = {**reward.diagnostics_json, "policy_consumption_status": "consumed"}
    db.flush()
    return row
