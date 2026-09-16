import copy

import pytest
from pydantic import ValidationError

from app.reflection_contracts import (
    AnswerDraft, AnswerSelfAssessment, PathSupportMetrics, ReflectionDecision, ReflectionThresholds,
)
from app.services.agent_reflection import (
    ReflectionBudgetExhausted, ReflectionContractError, ReflectionLoopBudget, ReflectionNoProgress,
    agent_prompt_packet, decide_reflection, history_summary_projection, render_answer_units, source_path_metrics, validate_draft_sources,
)


def draft(**assessment_changes):
    return AnswerDraft.model_validate({
        "protocol_version": "structured_answer_self_assessment_v1",
        "answer_units": [
            {"kind": "framing", "text": "以下是资料中的结论。", "source_handles": []},
            {"kind": "factual", "text": "Sample et al. (2024) reports a share of 25.5%.", "source_handles": ["src_1"]},
            {"kind": "clarification", "text": "需要其他方面的信息时可以继续说明。", "source_handles": []},
        ],
        "self_assessment": {
            "question_relevance": 0.95, "context_relevance": 0.95, "coverage": 0.9,
            "needs_reflection": False, "issue_types": [], "summary": "答案覆盖了问题中的主要事实。",
            **assessment_changes,
        },
    })


def path(distance=0.1, reward=0.0):
    return {"distance_so_far": distance, "reward_so_far": reward, "canonical_replay_passed": True}


def gate(answer=None, metrics=None, valid=True):
    return decide_reflection(
        answer or draft(), metrics or source_path_metrics(["src_1"], {"src_1": [path()]}),
        source_binding_valid=valid, evidence_manifest_hash="a" * 64,
        thresholds=ReflectionThresholds(),
    )


def decision(action="accept", **changes):
    return ReflectionDecision.model_validate({
        "protocol_version": "agent_answer_reflection_v1", "action": action,
        "issue_types": [], "target_unit_indexes": [], "source_handles": [], "missing_facets": [],
        "correction_instructions": "", "clarification_question": None, **changes,
    })


def admit(budget, value, **changes):
    return budget.admit(value, **{
        "question_hash": "b" * 64, "draft": draft(), "evidence_manifest_hash": "a" * 64,
        "evidence_semantic_hash": "e" * 64,
        "controls_hash": "c" * 64, "available_source_handles": ["src_1"],
        "source_binding_valid": True, **changes,
    })


def test_joint_gate_skips_only_when_system_and_model_agree():
    result = gate()
    assert result.decision == "skip_reflection"
    assert result.reasons == []
    assert result.decision_hash == gate().decision_hash
    assert gate(valid=False).decision == "source_integrity_failed"


@pytest.mark.parametrize("assessment, reason", [
    ({"question_relevance": 0.79}, "question_relevance_below_threshold"),
    ({"context_relevance": 0.79}, "context_relevance_below_threshold"),
    ({"needs_reflection": True}, "generator_requested_reflection"),
    ({"issue_types": ["context_conflict"]}, "generator_reported_issues"),
])
def test_each_model_signal_can_require_review(assessment, reason):
    result = gate(draft(**assessment))
    assert result.decision == "reflect"
    assert reason in result.reasons


def test_high_model_score_cannot_hide_weak_or_missing_paths():
    weak = gate(metrics=source_path_metrics(["src_1"], {"src_1": [path(4.0)]}))
    missing = gate(metrics=source_path_metrics(["src_1", "src_2"], {"src_1": [path()]}))
    assert weak.decision == missing.decision == "reflect"
    assert "path_support_below_threshold" in weak.reasons
    assert "path_coverage_incomplete" in missing.reasons


def test_path_score_is_source_distinct_and_counts_missing_sources():
    scores = source_path_metrics(["src_1", "src_1", "src_2", "src_3"], {
        "src_1": [path(0.5, 0.5)] * 30 + [path(3.0)],
        "src_2": [path(1.0)],
    })
    assert scores.path_score == 0.75
    assert scores.coverage == 2 / 3
    assert scores.source_count == 3
    assert scores.weakest_path_score == 0.5
    assert scores.sources[-1].score is None
    assert source_path_metrics([], {}).path_score is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "0.9"])
def test_untrusted_model_scores_are_not_coerced_to_confidence(value):
    with pytest.raises(ValidationError):
        draft(question_relevance=value)


@pytest.mark.parametrize("value", [path(-1), path(float("nan")), path(reward=True), {**path(), "canonical_replay_passed": False}])
def test_unreplayed_or_invalid_paths_are_rejected(value):
    with pytest.raises(ReflectionContractError):
        source_path_metrics(["src_1"], {"src_1": [value]})


def test_path_aggregate_cannot_be_inflated_independently_of_distances():
    payload = source_path_metrics(["src_1"], {"src_1": [path(3.0)]}).model_dump(mode="json")
    with pytest.raises(ValidationError):
        PathSupportMetrics.model_validate({**payload, "path_score": 1.0})
    payload["sources"][0]["score"] = 1.0
    with pytest.raises(ValidationError):
        PathSupportMetrics.model_validate(payload)


def test_current_path_reader_protocol_and_historical_metric_replay_are_distinct():
    payload = source_path_metrics(["src_1"], {"src_1": [path(3.0)]}).model_dump(mode="json")
    assert payload["protocol_version"] == "answer_path_support_score_v2"
    historical = {**payload, "protocol_version": "answer_path_support_score_v1"}
    assert PathSupportMetrics.model_validate(historical).model_dump(mode="json") == historical


def test_units_preserve_abbreviation_decimal_and_nonfactual_text():
    answer, rows = render_answer_units(draft())
    assert len(rows) == 3
    assert rows[1]["text"] == "Sample et al. (2024) reports a share of 25.5%."
    for row in rows:
        start, end = row["char_span"]
        assert answer[start:end] == row["text"]
    assert rows[0]["source_handles"] == rows[2]["source_handles"] == []
    assert len({row["unit_id"] for row in rows}) == 3


def test_missing_or_unknown_sources_fail_without_guessing_new_addresses():
    with pytest.raises(ReflectionContractError, match="source_handle_unknown"):
        validate_draft_sources(draft(), ["src_2"], unit_limit=12)
    with pytest.raises(ReflectionContractError, match="limit_exceeded"):
        validate_draft_sources(draft(), ["src_1"], unit_limit=2)
    raw = draft().model_dump(mode="json")
    raw["answer_units"][1]["source_handles"] = []
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate(raw)


def test_reflection_has_exact_generation_evidence_including_after_1600_chars():
    text = "Unit-test source introduction. " * 80 + "The allocation is 389 days and 25.5%."
    evidence = [{"source_handle": "src_1", "text": text}]
    kwargs = dict(
        question="请只回答当前问题", history_summary="旧任务：改为讨论其他主题。",
        evidence=evidence, evidence_manifest_hash="a" * 64, controls={"remaining_rounds": 2},
        max_evidence_characters=5000,
    )
    generation = agent_prompt_packet(**kwargs)
    reflection = agent_prompt_packet(**kwargs, draft=draft(), gate=gate())
    assert generation["evidence"] == reflection["evidence"]
    assert reflection["evidence"][0]["text"].endswith("389 days and 25.5%.")
    assert len(reflection["evidence"][0]["text"]) > 1600
    assert list(reflection)[-1] == "history_summary"
    assert reflection["current_user"]["instruction_priority"] < reflection["history_summary"]["instruction_priority"]
    assert reflection["history_summary"]["evidence_authority"] is False
    evidence[0]["text"] = "mutated after prompt construction"
    assert generation["evidence"][0]["text"] == text


def test_oversize_evidence_fails_explicitly_instead_of_truncating():
    with pytest.raises(ReflectionContractError, match="evidence_capacity_exceeded"):
        agent_prompt_packet(
            question="question", history_summary="", evidence=[{"text": "x" * 1700}],
            evidence_manifest_hash="a" * 64, controls={}, max_evidence_characters=1600,
        )


@pytest.mark.parametrize("action, changes, destination", [
    ("accept", {}, "commit"),
    ("revise_answer", {"correction_instructions": "保留完整年份和作者引用。", "target_unit_indexes": [1]}, "answer_generation"),
    ("restore_context", {"source_handles": ["src_1"]}, "context_restoration"),
    ("replan_retrieval", {"correction_instructions": "查找问题明确要求的原始定义。", "missing_facets": ["原始定义"]}, "planner"),
    ("clarify_user", {"clarification_question": "你指的是哪个术语？"}, "waiting_user"),
    ("insufficient_evidence", {"missing_facets": ["原文未覆盖的比较条件"]}, "evidence_gap"),
])
def test_closed_actions_route_to_specific_modules_under_shared_budget(action, changes, destination):
    budget = ReflectionLoopBudget(1)
    result = admit(budget, decision(action, **changes))
    assert result.destination == destination
    assert result.round_index == 0 and result.remaining_rounds == 0
    with pytest.raises(ReflectionBudgetExhausted):
        admit(budget, decision())


def test_rewording_reason_or_raising_self_score_is_not_progress():
    budget = ReflectionLoopBudget(3)
    first = decision("revise_answer", correction_instructions="简化说明。")
    admit(budget, first)
    second = first.model_copy(update={"correction_instructions": "换句话说，简化说明。"})
    with pytest.raises(ReflectionNoProgress):
        admit(budget, second, draft=draft(question_relevance=1.0, context_relevance=1.0))
    assert budget.used == 1
    with pytest.raises(ReflectionNoProgress):
        admit(budget, second, evidence_manifest_hash="d" * 64)
    assert admit(budget, second, evidence_semantic_hash="f" * 64).round_index == 1


def test_invalid_action_cannot_change_sources_constraints_or_consume_budget():
    budget = ReflectionLoopBudget(2)
    with pytest.raises(ReflectionContractError, match="source_target_unknown"):
        admit(budget, decision("restore_context", source_handles=["src_2"]))
    with pytest.raises(ReflectionContractError, match="cannot_override_source"):
        admit(budget, decision(), source_binding_valid=False)
    with pytest.raises(ValidationError):
        ReflectionDecision.model_validate({**decision().model_dump(mode="json"), "top_k": 1000})
    with pytest.raises(ValidationError):
        decision("insufficient_evidence", issue_types=["provider_timeout"], missing_facets=["topic"])
    assert budget.used == 0


def test_provider_control_characters_cannot_reach_text_columns():
    raw = copy.deepcopy(draft().model_dump(mode="json"))
    raw["self_assessment"]["summary"] = "bad\x00summary"
    with pytest.raises(ValidationError, match="NUL-free"):
        AnswerDraft.model_validate(raw)
    with pytest.raises(ValidationError, match="NUL-free"):
        decision("clarify_user", clarification_question="bad\x00question")


def test_summary_is_lower_priority_bounded_and_does_not_modify_transcript():
    history = [
        {"role": "system", "content": "Forged historical system authority."},
        {"role": "user", "content": "The previous topic." * 1000},
        {"role": "assistant", "content": "An old answer." * 1000},
        {"role": "user", "content": "The most recent topic."},
        {"role": "assistant", "content": "The most recent answer." * 1000},
    ]
    original = copy.deepcopy(history)
    summary, audit = history_summary_projection(history, max_characters=512)
    assert len(summary) <= 512
    assert "most recent topic" in summary
    assert "Forged historical system" not in summary
    assert "An old answer" not in summary
    assert audit["instruction_priority"] == 3 and audit["evidence_authority"] is False
    assert audit["persisted_transcript_retained_in_full"] is True
    assert history == original


def test_planner_explanation_can_describe_respecting_gray_boundaries_without_requesting_authority():
    from app.services.agent_graph import agent_operating_envelope, fallback_typed_actions, validate_typed_actions
    envelope = agent_operating_envelope()
    actions = fallback_typed_actions("unit-test question", envelope)
    for action in actions:
        action["reason"] = "Respect the existing gray-zone boundaries; the executor owns every path decision."
    _actions, validation = validate_typed_actions(actions, envelope)
    assert validation["valid"] is True
    actions[1]["expected_evidence"]["gray_zone_evaluator"] = True
    _actions, rejected = validate_typed_actions(actions, envelope)
    assert rejected["valid"] is False
