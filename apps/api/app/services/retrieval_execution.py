"""Compile retrieval-only actions through the existing deterministic executor."""
from __future__ import annotations

from app.retrieval_control_contracts import LexicalStrategy, TaskContract, control_hash


def compile_retrieval_execution(db, *, task: TaskContract, strategy: LexicalStrategy,
                                envelope: dict, top_k: int, lexical_policy_hash: str,
                                chunk_targets: tuple[str, ...] = (), mid_targets: tuple[str, ...] = ()):
    from app.services import agent_graph as ag
    from app.services.context_graph import RETRIEVAL_CONTROL_PROTOCOL, validate_typed_action_traversal_controls

    strategy.validate_task(task)
    kinds = [kind for kind in ag.REQUIRED_TYPED_ACTIONS if kind != "review_answer"]
    actions = []
    for kind in kinds:
        targets = list(chunk_targets) if kind == "recall_chunks" else list(mid_targets) if kind == "select_entry_nodes" else []
        actions.append({"action_type": kind, "target_ids": targets,
            "reason": "Execute the frozen retrieval-only task and collect source-backed context.",
            "budget_request": ag._default_budget_for_action(kind, envelope),
            "expected_evidence": {"source": "context_graph", "requires_chunk_spans": True},
            "stop_condition": {"required_action_complete": True}})
    validated, validation = ag.validate_typed_actions(
        actions, envelope, db=db, knowledge_base_id=task.knowledge_base_id,
        retrieval_granularity=task.retrieval_granularity, execution_scope="retrieval_only")
    if not validation["valid"] or validation["inserted_required_actions"]:
        raise ValueError("retrieval_control_action_validation_failed")
    controls = ag.compile_typed_action_execution_controls(validated, envelope,
        requested_result_top_k=top_k, retrieval_granularity=task.retrieval_granularity,
        validation_diagnostics=validation)
    controls.update(protocol_version=RETRIEVAL_CONTROL_PROTOCOL, reflection_round_budget=0,
                    task_hash=task.identity, lexical_strategy_hash=strategy.identity,
                    lexical_policy_hash=lexical_policy_hash)
    controls["control_hash"] = ag.stable_hash({key: value for key, value in controls.items() if key != "control_hash"})
    controls, _ = validate_typed_action_traversal_controls(controls, base_envelope=envelope,
        retrieval_granularity=task.retrieval_granularity, result_top_k=controls["effective_result_top_k"])
    return validated, validation, controls


def persist_retrieval_execution(db, *, run, task, strategy, envelope, actions, validation, controls):
    from app.models import AgentAction, AgentPlan
    from app.services.chunking import stable_hash
    if run.knowledge_base_id != task.knowledge_base_id or run.question != task.question:
        raise ValueError("retrieval_plan_run_scope_mismatch")
    plan = AgentPlan(run_id=run.id, knowledge_base_id=run.knowledge_base_id,
        plan_index=strategy.revision, query_intent_json={"task_hash": task.identity},
        planner_model_json={"protocol_version": "deterministic_retrieval_compiler_v1", "model_call_count": 0,
            'provider_response_recorded':False},
        envelope_json=envelope, typed_actions_json=actions, validation_json=validation,
        status="validated", diagnostics_json={"retrieval_control": {
            "task_hash": task.identity, "strategy_hash": strategy.identity, "control_hash": controls["control_hash"]},
            "agent_operating_envelope_hash": stable_hash(envelope)})
    db.add(plan)
    db.flush()
    rows = []
    for index, action in enumerate(actions):
        accepted = next(item for item in validation["accepted"] if item["accepted_index"] == index)
        row = AgentAction(run_id=run.id, plan_id=plan.id, action_index=index,
            action_type=action["action_type"], target_ids_json=action["target_ids"], reason=action["reason"],
            budget_request_json=action["budget_request"], expected_evidence_json=action["expected_evidence"],
            stop_condition_json=action["stop_condition"], validation_json={
                **accepted["validation"], "plan_valid": True,
                "typed_action_schema_protocol_version": validation["typed_action_schema_protocol_version"],
                "typed_action_schema_protocol_hash": validation["typed_action_schema_protocol_hash"]},
            status="accepted")
        db.add(row)
        rows.append(row)
    db.flush()
    return plan, rows
