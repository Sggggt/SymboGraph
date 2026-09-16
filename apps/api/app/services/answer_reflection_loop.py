"""The single answer/reflection loop; executors retain all tool authority."""
from __future__ import annotations

import asyncio
import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from app.reflection_contracts import ANSWER_REFLECTION_PROTOCOL, AnswerDraft, ReflectionDecision, ReflectionGate, ReflectionTransition
from app.services.agent_reflection import ReflectionBudgetExhausted, ReflectionContractError, ReflectionLoopBudget, ReflectionNoProgress, ReflectionRetrievalHandoff, reflection_hash, generation_calls_from_events
from app.services.answer_sources import AnswerEvidenceManifest
from app.services.reflection_models import StructuredAnswerResult, WholeAnswerReflectionResult


class AnswerReviewExecutor(Protocol):
    def remaining_reflection_rounds(self, hard_remaining: int) -> int: ...
    def validate_reflection_action(self, evidence: AnswerEvidenceManifest, decision: ReflectionDecision) -> None: ...
    async def boundary(self, stage: str) -> None: ...
    async def generate(self, evidence: AnswerEvidenceManifest, feedback: ReflectionDecision | None) -> StructuredAnswerResult: ...
    async def gate(self, evidence: AnswerEvidenceManifest, draft: AnswerDraft) -> ReflectionGate: ...
    async def reflect(self, evidence: AnswerEvidenceManifest, draft: AnswerDraft, gate: ReflectionGate, remaining_rounds: int) -> WholeAnswerReflectionResult: ...
    async def backtrack(self, evidence: AnswerEvidenceManifest, decision: ReflectionDecision, transition: ReflectionTransition) -> AnswerEvidenceManifest: ...
    async def record(self, event: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class AnswerReviewResult:
    outcome: str
    draft: AnswerDraft
    evidence: AnswerEvidenceManifest
    gate: ReflectionGate
    terminal_decision: ReflectionDecision | None
    model_audit: dict[str, Any]
    reflection_audit: dict[str, Any]


def _draft_semantic_hash(draft: AnswerDraft, evidence: AnswerEvidenceManifest) -> str:
    sources = evidence.by_handle()
    return reflection_hash([{
        "kind": unit.kind, "text": unit.text,
        "sources": sorted([
            (sources[handle]["chunk_id"], sources[handle]["source_span"]["char_span"])
            for handle in unit.source_handles
        ]),
    } for unit in draft.answer_units])


def post_backtrack_feedback(decision: ReflectionDecision) -> ReflectionDecision:
    """The completed tool request stays in the ledger, not in new user scope."""
    if decision.action not in {"restore_context", "replan_retrieval"}:
        raise ReflectionContractError("post_backtrack_feedback_action_invalid")
    return ReflectionDecision.model_validate({
        "protocol_version": ANSWER_REFLECTION_PROTOCOL, "action": "revise_answer",
        "issue_types": [], "target_unit_indexes": [], "source_handles": [], "missing_facets": [],
        "clarification_question": None,
        "correction_instructions": (
            "Under post_backtrack_resynthesis_v1, the requested source recovery has completed. "
            "Answer the original current user question again using its question_scope and the complete CURRENT evidence. "
            "Reassess required facts and gaps from this evidence. Earlier suggested retrieval locations, tables, sections "
            "or corroboration steps are not additional user requirements; other verified passages may fully satisfy the question. "
            "Do not repeat an already executed recovery command or keep an old gap open when the requested fact is now available. "
            "Retain correct supported facts, label their actual source locations, and distinguish optional follow-ups from "
            "defects that prevent answering this user's question."
        ),
    })


async def run_answer_reflection_loop(
    *, question: str, evidence: AnswerEvidenceManifest, executor: AnswerReviewExecutor,
    controls_hash: str, round_budget: int, prior_events: list[dict[str, Any]] | None = None,
) -> AnswerReviewResult:
    budget = ReflectionLoopBudget(round_budget)
    feedback: ReflectionDecision | None = None
    previous_semantic_draft: str | None = None
    records = deepcopy(prior_events or [])
    if records and (records[-1].get("stage") != "retrieval_handoff"
                    or any(event.get("sequence_index") != index or event.get("status") != "completed" for index, event in enumerate(records))):
        raise ReflectionContractError("reflection_continuation_events_invalid")
    generation_calls = generation_calls_from_events(records)
    reflection_calls = 0
    for event in records:
        if event.get("stage") != "answer_reflection":
            continue
        try:
            previous_decision = ReflectionDecision.model_validate(event["decision"])
            previous_transition = ReflectionTransition.model_validate(event["transition"])
            signature = event["semantic_input_hash"]
            if (previous_transition.round_index != reflection_calls
                or previous_transition.remaining_rounds != round_budget - reflection_calls - 1
                or previous_transition.decision_hash != reflection_hash(previous_decision.model_dump(mode="json"))
                or not isinstance(signature, str) or len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature)
                or signature in budget.attempted_inputs):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ReflectionContractError("reflection_continuation_budget_replay_invalid") from None
        budget.attempted_inputs.add(signature)
        reflection_calls += 1
    budget.used = reflection_calls
    if budget.used > budget.limit:
        raise ReflectionContractError("reflection_continuation_exceeds_budget")
    if records and (not reflection_calls or records[-1].get("reflection_rounds_used") != budget.used
                    or records[-1].get("round_budget") != budget.limit):
        raise ReflectionContractError("reflection_continuation_handoff_count_invalid")
    started = time.monotonic()

    async def record(event: dict[str, Any]) -> None:
        packet = {"protocol_version": ANSWER_REFLECTION_PROTOCOL, "sequence_index": len(records), **event}
        await executor.record(packet)
        records.append(packet)

    def finish(outcome: str, generated: StructuredAnswerResult, gate: ReflectionGate, terminal: ReflectionDecision | None) -> AnswerReviewResult:
        audit = {
            "protocol_version": ANSWER_REFLECTION_PROTOCOL, "outcome": outcome,
            "draft_hash": reflection_hash(generated.draft.model_dump(mode="json")),
            "evidence_manifest_hash": evidence.manifest_hash, "gate": gate.model_dump(mode="json"),
            "round_budget": round_budget, "reflection_rounds_used": budget.used,
            "generation_model_call_count": generation_calls, "reflection_model_call_count": reflection_calls,
            "citation_judge_model_call_count": 0, "events": records,
            "elapsed_seconds": time.monotonic() - started,
            "self_assessment_is_reward_label": False,
        }
        audit["audit_hash"] = reflection_hash(audit)
        return AnswerReviewResult(outcome, generated.draft, evidence, gate, terminal, generated.model_audit, audit)

    if records:
        handoff = records[-1]
        if handoff.get("evidence_semantic_hash") == evidence.semantic_hash():
            await record({"stage": "planner", "status": "no_progress", "reason": "retrieval_handoff_evidence_unchanged"})
            raise ReflectionNoProgress("reflection_handoff_evidence_unchanged")
        feedback = post_backtrack_feedback(ReflectionDecision.model_validate(handoff["decision"]))

    while True:
        await executor.boundary("answer_generation")
        generation_calls += 1
        step_started = time.monotonic()
        try:
            generated = await executor.generate(evidence, feedback)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = getattr(exc, "external_failure", {})
            await record({"stage": "answer_generation", "status": "failed", "error_type": type(exc).__name__, "error_code": getattr(exc, "code", "execution_failed"), "cause_type": getattr(exc, "cause_type", None),
                          "cause_code": failure.get("error_code"), "cause_phase": failure.get("phase"), "http_status": failure.get("http_status"),
                          "model_call_count": getattr(exc, "model_call_count", 0), "schema_repair": getattr(exc, "schema_repair", None), "provider_shape": getattr(exc, "provider_shape", None)})
            raise
        semantic_draft = _draft_semantic_hash(generated.draft, evidence)
        if previous_semantic_draft is not None and semantic_draft == previous_semantic_draft:
            await record({"stage": "answer_generation", "status": "no_progress", "reason": "answer_units_unchanged"})
            raise ReflectionNoProgress("reflection_answer_unchanged")
        generation_event = {
            "stage": "answer_generation", "status": "completed",
            "duration_ms": round((time.monotonic() - step_started) * 1000),
            "draft_hash": reflection_hash(generated.draft.model_dump(mode="json")),
            "self_assessment": generated.draft.self_assessment.model_dump(mode="json"),
            "model_call_count": generated.model_audit.get("model_call_count", 1),
            "schema_repair": generated.model_audit.get("schema_repair"),
        }
        generation_calls += generation_calls_from_events([generation_event]) - 1
        await record(generation_event)
        await executor.boundary("reflection_gate")
        gate = await executor.gate(evidence, generated.draft)
        gate_payload = gate.model_dump(mode="json", exclude={"decision_hash"})
        if (
            gate.draft_hash != reflection_hash(generated.draft.model_dump(mode="json"))
            or gate.evidence_manifest_hash != evidence.manifest_hash
            or gate.decision_hash != reflection_hash(gate_payload)
        ):
            raise ReflectionContractError("reflection_gate_identity_mismatch")
        await record({"stage": "reflection_gate", "status": "completed", "gate": gate.model_dump(mode="json")})
        if gate.decision == "source_integrity_failed":
            raise ReflectionContractError("source_integrity_failed")
        if gate.decision == "skip_reflection":
            return finish("accepted_after_reflection" if reflection_calls else "accepted_without_reflection", generated, gate, None)
        remaining = executor.remaining_reflection_rounds(budget.limit - budget.used)
        if type(remaining) is not int or not 0 <= remaining <= budget.limit - budget.used:
            raise ReflectionContractError("reflection_remaining_budget_invalid")
        if remaining <= 0:
            await record({"stage": "answer_reflection", "status": "budget_exhausted"})
            raise ReflectionBudgetExhausted("reflection_round_budget_exhausted")
        await executor.boundary("answer_reflection")
        reflection_calls += 1
        step_started = time.monotonic()
        try:
            reviewed = await executor.reflect(evidence, generated.draft, gate, remaining)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = getattr(exc, "external_failure", {})
            await record({"stage": "answer_reflection", "status": "failed", "error_type": type(exc).__name__, "error_code": getattr(exc, "code", "execution_failed"), "cause_type": getattr(exc, "cause_type", None),
                          "cause_code": failure.get("error_code"), "cause_phase": failure.get("phase"), "http_status": failure.get("http_status"),
                          "provider_shape": getattr(exc, "provider_shape", None)})
            raise
        try:
            executor.validate_reflection_action(evidence, reviewed.decision)
            transition = budget.admit(
                reviewed.decision, question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest(),
                draft=generated.draft, evidence_manifest_hash=evidence.manifest_hash,
                evidence_semantic_hash=evidence.semantic_hash(), controls_hash=controls_hash,
                available_source_handles=list(evidence.by_handle()), source_binding_valid=gate.source_binding_valid,
            )
        except ReflectionContractError as exc:
            await record({"stage": "reflection_action_validation", "status": "rejected", "reason": str(exc), "executed_action_count": 0})
            raise
        await record({
            "stage": "answer_reflection", "status": "completed",
            "duration_ms": round((time.monotonic() - step_started) * 1000),
            "decision": reviewed.decision.model_dump(mode="json"), "transition": transition.model_dump(mode="json"),
            "semantic_input_hash": budget.last_semantic_input_hash,
            "model_audit": reviewed.model_audit,
        })
        if reviewed.decision.action == "accept":
            return finish("accepted_after_reflection", generated, gate, reviewed.decision)
        if reviewed.decision.action in {"clarify_user", "insufficient_evidence"}:
            return finish(reviewed.decision.action, generated, gate, reviewed.decision)
        feedback = reviewed.decision
        if feedback.action == "revise_answer":
            previous_semantic_draft = semantic_draft
            continue
        await executor.boundary(transition.destination)
        try:
            next_evidence = await executor.backtrack(evidence, feedback, transition)
        except ReflectionRetrievalHandoff as handoff:
            await record({**handoff.payload, "stage": "retrieval_handoff", "status": "completed",
                "round_budget": budget.limit, "reflection_rounds_used": budget.used,
                "decision": feedback.model_dump(mode="json"), "evidence_semantic_hash": evidence.semantic_hash()})
            raise
        if next_evidence.semantic_hash() == evidence.semantic_hash():
            await record({"stage": transition.destination, "status": "no_progress", "reason": "evidence_unchanged"})
            raise ReflectionNoProgress("reflection_evidence_unchanged")
        await record({
            "stage": transition.destination, "status": "completed",
            "before_evidence_manifest_hash": evidence.manifest_hash,
            "after_evidence_manifest_hash": next_evidence.manifest_hash,
        })
        evidence = next_evidence
        feedback = post_backtrack_feedback(feedback)
        previous_semantic_draft = None
