import asyncio
import copy

import pytest

from app.models import ContextPackage
from app.reflection_contracts import ReflectionThresholds
from app.services.agent_reflection import (
    ReflectionBudgetExhausted, ReflectionContractError, ReflectionNoProgress, decide_reflection, source_path_metrics,
)
from app.services.answer_reflection_loop import run_answer_reflection_loop
from app.services.answer_sources import build_answer_evidence_manifest
from app.services.reflection_models import AnswerReviewModelError, StructuredAnswerResult, WholeAnswerReflectionResult
from test_agent_reflection import decision, draft, path


def manifest(text="Complete unit-test evidence.", package_id="unit-test-package"):
    span = {
        "chunk_id": "unit-test-chunk", "document_version_id": "unit-test-version",
        "raw_span_text_hash": "a" * 64, "char_span": [0, len(text)],
        "context_package_id": package_id, "retrieval_trace_id": "unit-test-trace",
    }
    item = {"chunk_id": "unit-test-chunk", "content": text, "source_span": span, "document_id": "unit-test-doc"}
    package = ContextPackage(id=package_id, retrieval_trace_id="unit-test-trace", package_json={"chunks": [item]})
    return build_answer_evidence_manifest(package, [{"chunk_id": "unit-test-chunk", "content": text}])


class Executor:
    def __init__(self, drafts, decisions=(), *, distance=0.1, next_evidence=None):
        self.drafts = iter(drafts)
        self.decisions = iter(decisions)
        self.distance = distance
        self.next_evidence = next_evidence
        self.events = []
        self.boundaries = []
        self.backtracks = []
        self.reflect_calls = 0
        self.reflect_error = None
        self.generation_feedback = []

    async def boundary(self, stage):
        self.boundaries.append(stage)

    def remaining_reflection_rounds(self, hard_remaining):
        return hard_remaining

    def validate_reflection_action(self, evidence, decision):
        pass

    async def generate(self, evidence, feedback):
        self.generation_feedback.append(copy.deepcopy(feedback))
        return StructuredAnswerResult(next(self.drafts), {"provider_response_persisted": False})

    async def gate(self, evidence, answer):
        return decide_reflection(
            answer, source_path_metrics(["src_1"], {"src_1": [path(self.distance)]}),
            source_binding_valid=True, evidence_manifest_hash=evidence.manifest_hash,
            thresholds=ReflectionThresholds(),
        )

    async def reflect(self, evidence, answer, gate, remaining_rounds):
        self.reflect_calls += 1
        if self.reflect_error:
            raise self.reflect_error
        return WholeAnswerReflectionResult(next(self.decisions), {"provider_response_persisted": False})

    async def backtrack(self, evidence, requested, transition):
        self.backtracks.append(transition.destination)
        return self.next_evidence

    async def record(self, event):
        self.events.append(copy.deepcopy(event))


async def run(executor, *, evidence=None, budget=2):
    return await run_answer_reflection_loop(
        question="unit-test question", evidence=evidence or manifest(), executor=executor,
        controls_hash="b" * 64, round_budget=budget,
    )


@pytest.mark.asyncio
async def test_high_quality_draft_skips_review_entirely():
    executor = Executor([draft()])
    result = await run(executor)
    assert result.outcome == "accepted_without_reflection"
    assert executor.reflect_calls == 0
    assert result.reflection_audit["citation_judge_model_call_count"] == 0
    assert result.reflection_audit["self_assessment_is_reward_label"] is False


@pytest.mark.asyncio
async def test_review_accept_ends_weak_path_review_without_repeating_gate():
    executor = Executor([draft()], [decision()], distance=4)
    result = await run(executor)
    assert result.outcome == "accepted_after_reflection"
    assert executor.reflect_calls == 1
    assert result.gate.path_metrics.path_score == 0.2


@pytest.mark.asyncio
async def test_revision_returns_to_generation_and_recomputes_gate():
    revised = draft()
    revised.answer_units[1] = revised.answer_units[1].model_copy(update={"text": "The corrected answer preserves 25.5%."})
    executor = Executor([draft(needs_reflection=True), revised], [decision("revise_answer", correction_instructions="Clarify the factual wording.")])
    result = await run(executor)
    assert result.outcome == "accepted_after_reflection"
    assert result.reflection_audit["generation_model_call_count"] == 2
    assert result.reflection_audit["reflection_rounds_used"] == 1
    assert executor.generation_feedback[-1].correction_instructions == "Clarify the factual wording."


@pytest.mark.asyncio
@pytest.mark.parametrize("action,destination", [("restore_context", "context_restoration"), ("replan_retrieval", "planner")])
async def test_backtrack_executes_specific_module_with_new_evidence(action, destination):
    requested = decision(action, source_handles=["src_1"], correction_instructions="Read an optional table after locating the definition.",
        missing_facets=["Optional table location"])
    executor = Executor(
        [draft(needs_reflection=True), draft()], [requested],
        next_evidence=manifest("New complete unit-test source paragraph.", "unit-test-new-package"),
    )
    result = await run(executor)
    assert executor.backtracks == [destination]
    assert result.evidence.package_id == "unit-test-new-package"
    assert result.reflection_audit["generation_model_call_count"] == 2
    feedback = executor.generation_feedback[-1]
    assert feedback.missing_facets == [] and feedback.issue_types == [] and feedback.source_handles == []
    assert "post_backtrack_resynthesis_v1" in feedback.correction_instructions
    assert "Optional table location" not in feedback.correction_instructions
    review_event = next(event for event in executor.events if event.get("stage") == "answer_reflection")
    assert review_event["decision"] == requested.model_dump(mode="json")


@pytest.mark.asyncio
async def test_new_package_id_without_new_evidence_is_no_progress():
    executor = Executor([draft(needs_reflection=True)], [decision("restore_context", source_handles=["src_1"])], next_evidence=manifest(package_id="unit-test-other-id"))
    with pytest.raises(ReflectionNoProgress, match="evidence_unchanged"):
        await run(executor)
    assert executor.events[-1]["status"] == "no_progress"


@pytest.mark.asyncio
async def test_self_score_change_without_answer_revision_cannot_skip_review():
    executor = Executor([draft(needs_reflection=True), draft(question_relevance=1.0)], [decision("revise_answer", correction_instructions="Correct the answer.")])
    with pytest.raises(ReflectionNoProgress, match="answer_unchanged"):
        await run(executor)


@pytest.mark.asyncio
@pytest.mark.parametrize("action,fields", [
    ("clarify_user", {"clarification_question": "Which definition do you mean?"}),
    ("insufficient_evidence", {"missing_facets": ["the requested comparison"]}),
])
async def test_semantic_terminal_preserves_explicit_reason(action, fields):
    executor = Executor([draft(needs_reflection=True)], [decision(action, **fields)])
    result = await run(executor)
    assert result.outcome == action
    assert result.terminal_decision.action == action


@pytest.mark.asyncio
async def test_timeout_is_a_technical_failure_not_corpus_absence():
    executor = Executor([draft(needs_reflection=True)])
    executor.reflect_error = AnswerReviewModelError("answer_reflection", "timeout")
    with pytest.raises(AnswerReviewModelError, match="answer_reflection_timeout"):
        await run(executor)
    assert executor.events[-1]["error_code"] == "timeout"
    assert all(event.get("outcome") != "insufficient_evidence" for event in executor.events)


@pytest.mark.asyncio
async def test_exhausted_budget_starts_no_reflection_request():
    executor = Executor([draft(needs_reflection=True)])
    with pytest.raises(ReflectionBudgetExhausted):
        await run(executor, budget=0)
    assert executor.reflect_calls == 0


@pytest.mark.asyncio
async def test_rejected_source_target_is_audited_without_execution():
    executor = Executor([draft(needs_reflection=True)], [decision("restore_context", source_handles=["src_99"])])
    with pytest.raises(ReflectionContractError, match="source_target_unknown"):
        await run(executor)
    assert executor.backtracks == []
    assert executor.events[-1]["executed_action_count"] == 0


@pytest.mark.asyncio
async def test_critical_audit_failure_is_not_swallowed():
    executor = Executor([draft()])
    async def broken_record(event):
        raise RuntimeError("unit-test audit failure")
    executor.record = broken_record
    with pytest.raises(RuntimeError, match="audit failure"):
        await run(executor)


@pytest.mark.asyncio
async def test_cancellation_at_reflection_boundary_starts_no_model_work():
    executor = Executor([draft(needs_reflection=True)])
    async def cancelled_boundary(stage):
        if stage == "answer_reflection":
            raise asyncio.CancelledError
    executor.boundary = cancelled_boundary
    with pytest.raises(asyncio.CancelledError):
        await run(executor)
    assert executor.reflect_calls == 0
