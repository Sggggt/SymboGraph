"""Production adapter for the single answer-level reflection workflow."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import asyncio
import hashlib
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import get_settings
from app.models import AgentAction, AgentObservation, AgentPlan, AgentRun, AgentTraceEvent, AnswerSession, Chunk, ContextPackage, QASession, RetrievalTrace, RewardEvent
from app.reflection_contracts import ANSWER_REFLECTION_PROTOCOL, PATH_SUPPORT_PROTOCOL, AnswerDraft, AnswerReflectionSummary, ReflectionDecision, ReflectionGate, ReflectionThresholds, ReflectionTransition
from app.services.agent_reflection import ReflectionBudgetExhausted, ReflectionContractError, ReflectionNoProgress, ReflectionRetrievalHandoff, decide_reflection, history_summary_projection, reflection_hash, render_answer_units, source_path_metrics
from app.services.answer_reflection_loop import run_answer_reflection_loop
from app.services.answer_sources import AnswerEvidenceManifest, audit_answer_sources, build_answer_evidence_manifest, persist_answer_source_bindings, source_binding_citations
from app.services.context_graph import build_context_package, context_package_to_contexts, runtime_settings_state_hash
from app.services.reflection_context import restore_reflection_context
from app.services.reflection_models import AnswerReflectionModels, StructuredAnswerResult, WholeAnswerReflectionResult
from app.services.reflection_run import load_reflection_run_ledger, reflection_run_elapsed_seconds
from app.services.reflection_sources import pack_and_retain_reflection_sources, retain_reflection_sources


class DirectReuseRequiresRetrieval(ReflectionRetrievalHandoff):
    pass


ANSWER_GENERATION_JSON_MAX_TOKENS = 32768
ANSWER_REFLECTION_JSON_MAX_TOKENS = 12000


class ReflectionAgentExecutor:
    def __init__(self, db: Session, *, request, run: AgentRun, package: ContextPackage,
                 contexts: list[dict[str, Any]], history: list[dict[str, Any]], envelope: dict[str, Any],
                 controls: dict[str, Any], plan: AgentPlan | None, query_intent: dict[str, Any],
                 query_facets: dict[str, Any], policy_prior: dict[str, Any] | None, query_embedding_memo=None,
                 direct_reuse: bool = False):
        from app.services import agent_graph as ag
        self.db, self.request, self.run = db, request, run
        self.package, self.contexts, self.plan = package, contexts, plan
        self.envelope, self.controls = deepcopy(envelope), deepcopy(controls)
        self.query_intent, self.query_facets = deepcopy(query_intent), deepcopy(query_facets)
        prior_path_protocol = (run.metadata_json or {}).get("path_support_protocol")
        if (prior_path_protocol not in {None, PATH_SUPPORT_PROTOCOL}
            or (prior_path_protocol is None and (run.metadata_json or {}).get("reflection_run_ledger"))):
            raise ReflectionContractError("reflection_path_protocol_changed")
        run.metadata_json = {**(run.metadata_json or {}), "path_support_protocol": PATH_SUPPORT_PROTOCOL}
        from app.services.agent_intent import current_question_scope_projection
        self.question_scope = current_question_scope_projection(request.question, self.query_intent)
        previous_scope = (run.metadata_json or {}).get("answer_question_scope")
        if previous_scope is not None and previous_scope != self.question_scope:
            raise ReflectionContractError("answer_question_scope_identity_changed")
        run.metadata_json = {**(run.metadata_json or {}), "answer_question_scope": deepcopy(self.question_scope)}
        flag_modified(run, "metadata_json")
        self.policy_prior, self.query_embedding_memo = policy_prior, query_embedding_memo
        self.direct_reuse = direct_reuse
        self.settings = get_settings().model_copy(deep=True)
        self.runtime_reflection_limit = self.settings.agent_reflection_round_budget
        self.settings.agent_answer_unit_limit = min(self.settings.agent_answer_unit_limit, int(controls.get("answer_unit_limit", self.settings.agent_answer_unit_limit)))
        self.settings.agent_reflection_round_budget = min(self.settings.agent_reflection_round_budget, int(controls.get("reflection_round_budget", self.settings.agent_reflection_round_budget)))
        self.runtime_hash = runtime_settings_state_hash()
        self.history_summary, self.history_audit = history_summary_projection(history, max_characters=self.settings.agent_history_summary_max_chars)
        if isinstance(run.metadata_json.get("history_summary"), str):
            self.history_summary = run.metadata_json["history_summary"]
            self.history_audit = dict(run.metadata_json.get("history_summary_audit") or self.history_audit)
        self.models = AnswerReflectionModels(lambda: ag.ChatProvider())
        self.last_draft: AnswerDraft | None = None
        self.last_model_audit: dict[str, Any] = {}
        self.actions: list[AgentAction] = []
        self.source_audit: dict[str, Any] = {}
        self.ledger, self.prior_review_events = load_reflection_run_ledger(
            db, run=run, requested_limit=self.settings.agent_reflection_round_budget, runtime_hash=self.runtime_hash)
        if self.ledger["hard_limit"] > self.runtime_reflection_limit:
            raise ReflectionContractError("reflection_run_ledger_exceeds_runtime_budget")
        self.phase_reflection_limit = self.settings.agent_reflection_round_budget
        self.phase_reflection_calls = 0
        self.review_action = db.scalar(select(AgentAction).where(AgentAction.plan_id == plan.id, AgentAction.action_type == "review_answer")) if plan else None
        if self.review_action is None:
            self.review_action = AgentAction(run_id=run.id, plan_id=plan.id if plan else None,
                action_type="review_answer", reason="Run-scoped answer reflection",
                action_index=int(db.scalar(select(func.count()).select_from(AgentAction).where(AgentAction.run_id == run.id, AgentAction.plan_id == (plan.id if plan else None))) or 0),
                validation_json={"valid": True, "protocol_version": ANSWER_REFLECTION_PROTOCOL, "direct_context_reuse": direct_reuse}, status="accepted")
            db.add(self.review_action)
            db.flush()

    async def boundary(self, stage: str) -> None:
        from app.services import agent_graph as ag
        ag.ensure_agent_run_not_cancelled(self.db, self.run)
        if runtime_settings_state_hash() != self.runtime_hash:
            raise ReflectionContractError("reflection_runtime_identity_changed")
        ag.set_run_state(self.db, self.run, "running", current_node=stage)

    def model_arguments(self, evidence: AnswerEvidenceManifest) -> dict[str, Any]:
        last_plan_index = self.db.scalar(select(func.max(AgentPlan.plan_index)).where(AgentPlan.run_id == self.run.id))
        planning_remaining = max(0, int(self.envelope["planning_round_budget"]) - (last_plan_index + 1 if last_plan_index is not None else 0))
        available_actions = ["accept", "revise_answer", "clarify_user", "insufficient_evidence"]
        if int(self.envelope["structure_restore_per_chunk_budget"]) > 0 and (not self.direct_reuse or planning_remaining):
            available_actions.append("restore_context")
        if planning_remaining:
            available_actions.append("replan_retrieval")
        return {
            "question": self.request.question, "evidence": evidence, "history_summary": self.history_summary,
            "controls": {"controls_hash": reflection_hash(self.controls), "reflection_round_budget": self.ledger["hard_limit"],
                         "path_support_protocol": PATH_SUPPORT_PROTOCOL,
                         "question_scope": self.question_scope,
                         "evidence_scope_protocol": "bounded_context_negative_claims_v1",
                         "evidence_scope": "provided_context_package_only", "corpus_absence_proven": False,
                         "instruction_priority": "system_current_user_history", "gray_zone_model_call_budget": 0,
                         "remaining_planning_rounds": planning_remaining, "available_actions": available_actions,
                         "route": "verified_context_reuse" if self.direct_reuse else "layered_context_graph",
                         "context_package_token_budget": int(self.package.token_budget),
                         "structure_restore_per_chunk_budget": int(self.envelope["structure_restore_per_chunk_budget"])},
            "unit_limit": self.settings.agent_answer_unit_limit,
            "timeout_seconds": float(self.settings.model_request_timeout_seconds),
            "max_tokens": min(int(self.settings.chat_json_max_tokens), ANSWER_GENERATION_JSON_MAX_TOKENS),
            "max_evidence_characters": max(int(self.package.token_budget) * 8, 1),
        }

    async def generate(self, evidence: AnswerEvidenceManifest, feedback: ReflectionDecision | None) -> StructuredAnswerResult:
        result = await self.models.generate(**self.model_arguments(evidence), feedback=feedback)
        self.last_draft, self.last_model_audit = result.draft, result.model_audit
        return result

    async def gate(self, evidence: AnswerEvidenceManifest, draft: AnswerDraft) -> ReflectionGate:
        from app.services import agent_graph as ag
        return await ag.run_bounded_source_io(self._gate, evidence, draft)

    def _gate(self, evidence: AnswerEvidenceManifest, draft: AnswerDraft) -> ReflectionGate:
        from app.services.storage import raise_if_source_io_cancelled
        raise_if_source_io_cancelled()
        _candidates, self.source_audit = audit_answer_sources(
            self.db, knowledge_base_id=self.run.knowledge_base_id, package=self.package, contexts=self.contexts,
            draft=draft, evidence=evidence, unit_limit=self.settings.agent_answer_unit_limit,
        )
        from app.services.answer_sources import answer_source_path_metrics
        metrics = answer_source_path_metrics(self.db, package=self.package, draft=draft, evidence=evidence, source_audit=self.source_audit)
        gate = decide_reflection(
            draft, metrics, source_binding_valid=bool(self.source_audit.get("all_valid")),
            evidence_manifest_hash=evidence.manifest_hash,
            thresholds=ReflectionThresholds(path_support=self.settings.agent_reflection_path_threshold,
                question_relevance=self.settings.agent_reflection_question_threshold,
                context_relevance=self.settings.agent_reflection_context_threshold),
        )
        if (self.plan is not None and self.plan.diagnostics_json.get("preliminary_evidence_uncertain")
            and self.phase_reflection_calls == 0 and gate.decision != "source_integrity_failed"):
            payload = gate.model_dump(mode="json", exclude={"decision_hash"})
            payload["decision"] = "reflect"
            payload["reasons"] = [*payload["reasons"], "preliminary_evidence_requires_full_context_review"]
            gate = ReflectionGate(**payload, decision_hash=reflection_hash(payload))
        return gate

    async def reflect(self, evidence: AnswerEvidenceManifest, draft: AnswerDraft, gate: ReflectionGate, remaining_rounds: int) -> WholeAnswerReflectionResult:
        self.phase_reflection_calls += 1
        arguments = self.model_arguments(evidence)
        arguments["timeout_seconds"] = min(arguments["timeout_seconds"], float(self.settings.agent_reflection_timeout_seconds))
        arguments["max_tokens"] = min(arguments["max_tokens"], ANSWER_REFLECTION_JSON_MAX_TOKENS)
        arguments["controls"] = {**arguments["controls"], "remaining_reflection_rounds": remaining_rounds}
        return await self.models.reflect(**arguments, draft=draft, gate=gate)

    def remaining_reflection_rounds(self, hard_remaining: int) -> int:
        return max(0, min(hard_remaining, self.phase_reflection_limit - self.phase_reflection_calls))

    def validate_reflection_action(self, evidence: AnswerEvidenceManifest, decision: ReflectionDecision) -> None:
        if decision.action not in self.model_arguments(evidence)["controls"]["available_actions"]:
            raise ReflectionContractError("reflection_action_unavailable_with_remaining_budget")

    async def backtrack(self, evidence: AnswerEvidenceManifest, decision: ReflectionDecision, transition: ReflectionTransition) -> AnswerEvidenceManifest:
        from app.services import agent_graph as ag
        source_package = self.package
        source_map = evidence.by_handle()
        preserve = sorted({source_map[handle]["chunk_id"] for unit in self.last_draft.answer_units for handle in unit.source_handles})
        if self.direct_reuse:
            raise DirectReuseRequiresRetrieval({
                "source_context_package_id": self.package.id, "source_retrieval_trace_id": self.package.retrieval_trace_id,
                "source_evidence_manifest_hash": evidence.manifest_hash,
                "preserve_chunk_ids": preserve,
            })
        if decision.action == "restore_context":
            by_handle = evidence.by_handle()
            targets = [by_handle[handle]["chunk_id"] for handle in decision.source_handles]
            self.package, self.contexts = await ag.run_bounded_source_io(
                restore_reflection_context, self.db, source_package=self.package,
                target_chunk_ids=targets, preserve_chunk_ids=preserve,
                token_budget=int(self.envelope["context_package_token_budget"]),
                restore_per_chunk_budget=int(self.envelope["structure_restore_per_chunk_budget"]),
                query_facets=self.query_facets, restoration_focus=list(decision.missing_facets),
            )
        elif decision.action == "replan_retrieval":
            last_index = self.db.scalar(select(func.max(AgentPlan.plan_index)).where(AgentPlan.run_id == self.run.id))
            index = (last_index if last_index is not None else -1) + 1
            if index >= int(self.envelope["planning_round_budget"]):
                raise ReflectionBudgetExhausted("reflection_planning_budget_exhausted")
            directive = {"verdict": "need_more_same_node", "reason": decision.correction_instructions,
                         "missing_facets": list(decision.missing_facets), "reflection_protocol": ANSWER_REFLECTION_PROTOCOL}
            diverse_sources = []
            seen_documents = set()
            for item in self.package.package_json["chunks"]:
                if item["document_id"] in seen_documents:
                    continue
                seen_documents.add(item["document_id"])
                diverse_sources.append({"chunk_id": item["chunk_id"], "document_title": item["document_title"],
                    "text_excerpt": item["content"][:360], "summary_hash": reflection_hash(item),
                    "source_span_address": {key: item["source_span"].get(key) for key in ("chunk_id", "document_version_id", "char_span", "page_range")}})
            observation = {"bounded_graph_observation": {
                "plan_index": self.plan.plan_index if self.plan else 0, "retrieval_granularity": self.request.retrieval_granularity,
                "typed_action_control_hash": self.controls.get("control_hash"),
                "required_facets": self.query_facets.get("required_facets") or decision.missing_facets,
                "covered_facets": self.package.covered_facets_json or [], "result_chunk_ids": self.package.hit_chunk_ids_json or [],
                "result_count": len(self.package.hit_chunk_ids_json or []), "citable_span_count": len(self.package.package_json["chunks"]),
                "candidate_chunk_span_summaries": diverse_sources,
                "observation_hash": reflection_hash({"protocol": "reflection_context_projection_v1", "manifest": evidence.manifest_hash}),
            }, "evidence_evaluator": directive}
            projection = ag.planner_observation_projection_packet([observation])
            proposed, raw_planner_output = await ag.propose_agent_plan(
                self.request.question, [{"role": "assistant", "content": self.history_summary}], self.query_intent,
                self.envelope, self.request.retrieval_granularity, plan_index=index,
                evaluator_directive=directive, policy_operating_prior=self.policy_prior,
                policy_knowledge_base_id=self.run.knowledge_base_id,
                validation_db=self.db, requested_result_top_k=int(self.controls.get("requested_result_top_k") or self.settings.retrieval_result_top_k_default),
                bounded_observations=projection["observations"],
            )
            planner_audit = ag._planner_model_audit(raw_planner_output, proposed)
            actions, validation = ag.validate_typed_actions(proposed, self.envelope, db=self.db,
                knowledge_base_id=self.run.knowledge_base_id, retrieval_granularity=self.request.retrieval_granularity)
            if not validation["valid"]:
                raise ag.TypedActionValidationError(validation)
            controls = ag.compile_typed_action_execution_controls(actions, self.envelope,
                requested_result_top_k=int(self.controls.get("requested_result_top_k") or self.settings.retrieval_result_top_k_default),
                retrieval_granularity=self.request.retrieval_granularity, validation_diagnostics=validation)
            semantic_keys = ("effective_result_top_k", "entry_targets_by_layer", "phase_target_ids_by_action", "traversal_envelope_overrides", "allowed_relation_types")
            if {key: controls.get(key) for key in semantic_keys} == {key: self.controls.get(key) for key in semantic_keys}:
                raise ReflectionNoProgress("reflection_replan_controls_unchanged")
            self.plan, action_rows = ag.record_agent_plan_and_actions(
                self.db, run=self.run, query_intent=self.query_intent, envelope=self.envelope,
                planner_model_audit=planner_audit, actions=actions, validation=validation, plan_index=index,
                evaluator_input=directive, policy_operating_prior=self.policy_prior,
            )
            self.plan.diagnostics_json = {**self.plan.diagnostics_json,
                "planner_sampling": raw_planner_output.get("planner_sampling") or ag.planner_sampling_audit(),
                "planner_observation_projection": {
                key: value for key, value in projection.items() if key != "observations"}}
            self.review_action.status = "completed"
            self.review_action = next(action for action in action_rows if action.action_type == "review_answer")
            self.db.commit()
            result = await ag.execute_typed_retrieval_plan(self.db, knowledge_base_id=self.run.knowledge_base_id,
                query=self.request.question, filters=self.request.filters, query_facets=self.query_facets, controls=controls,
                conversation_state_scope_hash=str(self.run.metadata_json.get("conversation_state_scope_hash") or ""),
                conversation_state_audit=dict(self.run.metadata_json.get("conversation_state") or {}),
                policy_identity_frozen=True, frozen_policy_state_hash=(self.policy_prior or {}).get("policy_state_hash"),
                query_embedding_request_memo=self.query_embedding_memo)
            if result is None:
                raise ReflectionContractError("reflection_retrieval_result_missing")
            self.plan.retrieval_trace_id = result.trace.id
            self.plan.diagnostics_json = {**self.plan.diagnostics_json, "execution_controls": controls, "reflection_transition": transition.model_dump(mode="json")}
            graph_observation = ag.bounded_graph_observation(search_result=result, query_facets=self.query_facets,
                controls=controls, plan_index=index)
            ag.record_typed_retrieval_observations(self.db, run_id=self.run.id, action_rows=action_rows, search_result=result,
                controls=controls, graph_observation=graph_observation)
            self.package = result.context_package or await ag.run_bounded_source_io(build_context_package, self.db,
                knowledge_base_id=self.run.knowledge_base_id, query=self.request.question, trace=result.trace,
                results=result.results, token_budget=int(controls["context_package_token_budget"]),
                restore_per_chunk_budget=int(controls["structure_restore_per_chunk_budget"]), snapshot_verifier=result.snapshot_verifier)
            self.package, self.contexts = await ag.run_bounded_source_io(
                pack_and_retain_reflection_sources, self.db, candidate_package=self.package,
                source_package=source_package, preserve_chunk_ids=preserve,
                token_budget=int(controls["context_package_token_budget"]),
            )
            self.controls = controls
            self.phase_reflection_calls = 0
            self.phase_reflection_limit = min(self.runtime_reflection_limit, int(controls["reflection_round_budget"]))
        else:
            raise ReflectionContractError("reflection_backtrack_action_invalid")
        self.db.commit()
        return build_answer_evidence_manifest(self.package, self.contexts)

    async def record(self, event: dict[str, Any]) -> None:
        from app.services import agent_graph as ag
        stage = event["stage"]
        node = stage if stage in {"answer_generation", "reflection_gate", "answer_reflection", "reflection_action_validation"} else "reflection_backtrack"
        labels = {"answer_generation": "回答与自评已生成", "reflection_gate": "已判断是否需要检查回答",
                  "answer_reflection": "回答检查已完成", "reflection_action_validation": "后续动作检查未通过",
                  "reflection_backtrack": "正在补充证据或改写回答"}
        if self.actions and stage in {"answer_generation", "context_restoration", "planner", "retrieval_handoff"}:
            action = self.actions[-1]
            if action.status == "accepted" and event.get("status") in {"completed", "no_progress", "failed"}:
                action.status = "rejected" if event["status"] == "failed" else event["status"]
        if stage == "retrieval_handoff":
            self.review_action.status = "completed"
            self.review_action.output_json = {**(self.review_action.output_json or {}), "retrieval_handoff": True}
        row = AgentObservation(run_id=self.run.id, action_id=self.review_action.id, observation_type="answer_reflection", observation_json=deepcopy(event),
            evidence_chunk_ids_json=list(self.package.hit_chunk_ids_json or []), verdict=str(event.get("status") or "observed"),
            diagnostics_json={"protocol_version": ANSWER_REFLECTION_PROTOCOL, "provider_response_persisted": False})
        self.db.add(row)
        if event.get("decision") and stage == "answer_reflection":
            action = AgentAction(run_id=self.run.id, plan_id=self.plan.id if self.plan else None,
                parent_action_id=self.review_action.id,
                action_index=int(self.db.scalar(select(func.count()).select_from(AgentAction).where(AgentAction.run_id == self.run.id, AgentAction.plan_id == (self.plan.id if self.plan else None))) or 0),
                action_type=event["decision"]["action"], reason=event["decision"]["correction_instructions"],
                validation_json={"valid": True, "protocol_version": ANSWER_REFLECTION_PROTOCOL, "direct_context_reuse": self.direct_reuse, "reflection_transition": event.get("transition")},
                output_json={"decision": event["decision"]},
                status="completed" if event["decision"]["action"] in {"accept", "clarify_user", "insufficient_evidence"} else "accepted")
            self.db.add(action)
            self.db.flush()
            row.action_id = action.id
            self.actions.append(action)
        self.db.flush()
        scores = {"stage": stage, **{key: event[key] for key in ("gate", "decision", "error_code", "executed_action_count") if key in event}}
        ag.trace(self.db, self.run.id, node, input_summary="当前问题与完整原文证据",
            output_summary=labels[node], status=str(event.get("status") or "completed"), scores=scores,
            duration_ms=int(event.get("duration_ms") or 0))


async def execute_reflection_answer(
    db: Session, *, request, run: AgentRun, session: QASession, package: ContextPackage,
    contexts: list[dict[str, Any]], history: list[dict[str, Any]], envelope: dict[str, Any], controls: dict[str, Any],
    plan: AgentPlan | None = None, query_intent: dict[str, Any] | None = None, query_facets: dict[str, Any] | None = None,
    policy_prior: dict[str, Any] | None = None, query_embedding_memo=None,
    direct_answer_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.services import agent_graph as ag
    executor = ReflectionAgentExecutor(db, request=request, run=run, package=package, contexts=contexts, history=history,
        envelope=envelope, controls=controls, plan=plan, query_intent=query_intent or {}, query_facets=query_facets or {},
        policy_prior=policy_prior, query_embedding_memo=query_embedding_memo, direct_reuse=direct_answer_audit is not None)
    if direct_answer_audit is None and (run.metadata_json or {}).get("partial_context_carry"):
        from app.services.partial_context import apply_partial_context
        executor.package, executor.contexts = await ag.run_bounded_source_io(apply_partial_context, db, run=run,
            candidate=package, token_budget=int(controls["context_package_token_budget"]))
        package, contexts = executor.package, executor.contexts
    if executor.prior_review_events:
        handoff = executor.prior_review_events[-1]
        source = db.get(ContextPackage, handoff.get("source_context_package_id"))
        if (source is None or source.knowledge_base_id != run.knowledge_base_id
            or source.retrieval_trace_id != handoff.get("source_retrieval_trace_id")
            or build_answer_evidence_manifest(source, context_package_to_contexts(source)).manifest_hash
                != handoff.get("source_evidence_manifest_hash")):
            raise ReflectionContractError("reflection_handoff_source_identity_changed")
        executor.package, executor.contexts = await ag.run_bounded_source_io(
            retain_reflection_sources, db, candidate_package=package, source_package=source,
            preserve_chunk_ids=handoff.get("preserve_chunk_ids") or [],
            token_budget=int(controls["context_package_token_budget"]),
        )
        package, contexts = executor.package, executor.contexts
    evidence = build_answer_evidence_manifest(package, contexts)
    result = await run_answer_reflection_loop(question=request.question, evidence=evidence, executor=executor,
        controls_hash=reflection_hash(controls), round_budget=executor.ledger["hard_limit"], prior_events=executor.prior_review_events)
    return await ag.run_bounded_source_io(_persist_reflection_answer, executor, result, request, session,
        direct_answer_audit=direct_answer_audit)


def _persist_reflection_answer(executor, result, request, session, *, direct_answer_audit):
    from app.services import agent_graph as ag
    from app.services.storage import raise_if_source_io_cancelled
    db, run = executor.db, executor.run
    raise_if_source_io_cancelled()
    package, contexts = executor.package, executor.contexts
    accepted = result.outcome in {"accepted_without_reflection", "accepted_after_reflection"}
    delivered = result.draft
    if not accepted:
        terminal = result.terminal_decision
        message = terminal.clarification_question if terminal.action == "clarify_user" else "当前证据尚未覆盖：" + "、".join(terminal.missing_facets) + "。请补充范围或资料。"
        delivered = AnswerDraft.model_validate({"protocol_version": result.draft.protocol_version,
            "answer_units": [{"kind": "clarification", "text": message, "source_handles": []}],
            "self_assessment": result.draft.self_assessment.model_dump(mode="json")})
    answer, units = render_answer_units(delivered)
    route = "direct_answer" if direct_answer_audit is not None else "layered_context_graph"
    run = db.scalar(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
    if run.status == "cancelled":
        raise asyncio.CancelledError("cancelled_by_user")
    audit = {**result.reflection_audit, "delivered_draft_hash": reflection_hash(delivered.model_dump(mode="json")), "delivered_answer_hash": hashlib.sha256(answer.encode()).hexdigest(),
             "run_ledger_protocol_version": executor.ledger["protocol_version"], "run_ledger_hash": reflection_hash(executor.ledger),
             "elapsed_seconds": reflection_run_elapsed_seconds(executor.ledger)}
    audit["audit_hash"] = reflection_hash({key: value for key, value in audit.items() if key != "audit_hash"})
    model_audit = {
        "provider": result.model_audit["provider"], "model": result.model_audit["model"], "external_called": True,
        "answer_model_called": True, "provider_call": result.model_audit.get("provider_call"),
        "prompt_protocol_version": result.model_audit["protocol_version"], "prompt_protocol_hash": result.model_audit["prompt_protocol_hash"],
        "profile_hash": result.model_audit["profile_hash"], "context_package_id": package.id,
        "retrieval_trace_id": package.retrieval_trace_id, "retrieval_granularity": request.retrieval_granularity,
        "conversation_state_scope_hash": str((package.diagnostics_json or {}).get("conversation_state_scope_hash") or ""),
        "exact_answer_hash": hashlib.sha256(answer.encode()).hexdigest(), "returned_citation_count": 0,
        "grounding_outcome": result.outcome, "policy_update_eligible": accepted and route != "direct_answer",
        "direct_answer_mode": "verified_context_reuse" if route == "direct_answer" else None,
        "direct_answer_protocol_version": (direct_answer_audit or {}).get("protocol_version"),
        "tool_call_count": 0 if route == "direct_answer" else None,
        "agent_plan_id": executor.plan.id if executor.plan else None,
        "agent_plan_index": executor.plan.plan_index if executor.plan else None,
        "planning_rounds_used": executor.plan.plan_index + 1 if executor.plan else 0,
        "typed_action_control_hash": executor.controls.get("control_hash"),
        "evidence_evaluator": (executor.plan.diagnostics_json.get("evidence_evaluator") or None) if executor.plan else None,
        "preliminary_evidence_uncertain": bool(executor.plan and executor.plan.diagnostics_json.get("preliminary_evidence_uncertain")),
    }
    answer_row = AnswerSession(knowledge_base_id=run.knowledge_base_id, qa_session_id=session.id, question=request.question,
        answer=answer, context_package_id=package.id, retrieval_trace_id=package.retrieval_trace_id,
        prompt_protocol_version=result.model_audit["protocol_version"], chunk_ids_json=list(package.hit_chunk_ids_json or []),
        model_json=model_audit, diagnostics_json={"answer_reflection": audit, "structured_answer": delivered.model_dump(mode="json"),
            "answer_units": units, "route": route, "direct_answer_audit": direct_answer_audit or {},
            "direct_answer_mode": "verified_context_reuse" if route == "direct_answer" else None,
            "history_summary_audit": executor.history_audit, "source_binding_protocol_version": "answer_source_binding_v1"})
    db.add(answer_row)
    db.flush()
    bindings = persist_answer_source_bindings(db, answer_session=answer_row, package=package, contexts=contexts,
        draft=delivered, evidence=result.evidence, unit_limit=executor.settings.agent_answer_unit_limit, reflection_audit_hash=audit["audit_hash"])
    raise_if_source_io_cancelled()
    citations = source_binding_citations(answer_session=answer_row, package=package, rows=bindings, reflection_audit_hash=audit["audit_hash"])
    answer_row.citation_ids_json = [row.id for row in bindings]
    summary = AnswerReflectionSummary(outcome=result.outcome, reflection_rounds_used=audit["reflection_rounds_used"],
        generation_model_call_count=audit["generation_model_call_count"], reflection_model_call_count=audit["reflection_model_call_count"],
        source_binding_count=len(bindings), source_binding_pass_rate=1.0 if bindings else 0.0,
        self_assessment=result.draft.self_assessment, path_score=result.gate.path_metrics.path_score,
        path_coverage=result.gate.path_metrics.coverage, reflection_audit_hash=audit["audit_hash"])
    model_audit.update(answer_session_id=answer_row.id, answer_reflection=summary.model_dump(mode="json"),
                       source_binding_pass_rate=summary.source_binding_pass_rate, returned_citation_count=len(citations))
    answer_row.model_json = model_audit
    run.metadata_json = {**run.metadata_json, "answer_session_id": answer_row.id}
    flag_modified(run, "metadata_json")
    review_action = executor.review_action
    review_action.status = "completed"
    review_action.output_json = {"answer_session_id": answer_row.id, "reflection_audit_hash": audit["audit_hash"], "source_binding_count": len(bindings)}
    db.add(AgentObservation(run_id=run.id, action_id=review_action.id, observation_type="answer_reflection_final",
        observation_json=summary.model_dump(mode="json"), evidence_chunk_ids_json=list(package.hit_chunk_ids_json or []),
        verdict="sufficient" if accepted else "insufficient"))
    flag_modified(answer_row, "model_json")
    ag.trace(db, run.id, "answer_source_binding", input_summary="回答与原文地址",
        output_summary=f"已核对 {len(bindings)} 条原文来源", scores={"stage": "final", "source_binding_count": len(bindings),
        "summary": summary.model_dump(mode="json")}, commit=False)
    # Reward facts are durable and explicitly distinct from model confidence.
    if route != "direct_answer":
        reward_metrics = {"protocol_version": "answer_reflection_reward_v1", "source_binding_rate": summary.source_binding_pass_rate,
            "path_support_score": summary.path_score, "path_coverage": summary.path_coverage,
            "generation_calls": summary.generation_model_call_count, "reflection_calls": summary.reflection_model_call_count,
            "self_assessment_reward_weight": 0.0, "training_eligible": bool(accepted and bindings)}
        reward = RewardEvent(knowledge_base_id=run.knowledge_base_id, answer_session_id=answer_row.id,
            retrieval_trace_id=package.retrieval_trace_id, chunk_ids_json=[row.chunk_id for row in bindings],
            reward_json=reward_metrics, context_json={"agent_run_id": run.id, "context_package_id": package.id},
            action_json={"route": route}, diagnostics_json={"protocol_version": "answer_reflection_reward_v1",
                "reflection_audit_hash": audit["audit_hash"], "policy_consumption_status": "pending"})
        db.add(reward)
        db.flush()
        from app.services.reflection_reward import consume_reflection_reward
        consume_reflection_reward(db, reward, envelope=executor.envelope, runtime_hash=executor.runtime_hash)
        ag.trace(db, run.id, "reward_event", input_summary="来源与路径的确定性指标",
            output_summary="已记录反思流程的奖励观测；模型自评分不计入奖励",
            scores={"runtime_settings_hash": executor.runtime_hash, "agent_operating_envelope_hash": ag.stable_hash(executor.envelope)}, commit=False)
    db.flush()
    final_state = ag.append_session_turn(db, session, request.question, answer, run.id, citations,
        answer_session_id=answer_row.id, retrieval_trace_id=package.retrieval_trace_id,
        task_status="active" if accepted else "waiting_user", route=route,
        direct_answer_mode="verified_context_reuse" if route == "direct_answer" else None, commit=False)
    run.status = "completed" if accepted else "needs_clarification"
    run.current_node = None
    run.completed_at = datetime.utcnow()
    run.final_answer, run.route = answer, route
    run.metadata_json = {**run.metadata_json, "answer_reflection_protocol": ANSWER_REFLECTION_PROTOCOL,
                         "direct_answer_mode": model_audit["direct_answer_mode"], "policy_update_eligible": model_audit["policy_update_eligible"],
                         "answer_session_id": answer_row.id}
    if direct_answer_audit is not None:
        run.metadata_json.update(direct_answer_decision=direct_answer_audit, direct_answer_protocol_version=direct_answer_audit.get("protocol_version"), tool_call_count=0)
    flag_modified(run, "metadata_json")
    raise_if_source_io_cancelled()
    db.commit()
    events = list(db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.sequence_index)))
    return {"run_id": run.id, "session_id": session.id, "answer": answer, "citations": citations, "used_chunks": contexts,
        "route": route, "direct_answer_mode": model_audit["direct_answer_mode"], "trace": [ag.trace_event_to_payload(event) for event in events],
        "degraded_mode": ag.is_degraded_mode(), "context_package_id": package.id, "retrieval_trace_id": package.retrieval_trace_id,
        "retrieval_granularity": request.retrieval_granularity, "model_audit": model_audit, "answer_model_audit": model_audit,
        "conversation_state": final_state.public_payload()}
