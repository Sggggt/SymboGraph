from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.intent_contracts import (
    CapabilityManifest, ChannelWeights, ExecutionBudget, ExecutionStrategy,
    IntentPlanningOutput, accept_plan,
)


def budget():
    return ExecutionBudget(dense_candidates=64, rq_candidates=64, bm25_candidates=64,
        root_entries=6, per_parent_entries=8, layer_entries=16, max_depth=4, restore_per_hit=2)


def capabilities(channels=("dense", "rq", "bm25"), layers=("coarse", "mid", "chunk"), *, bilingual=False):
    return CapabilityManifest(knowledge_base_id="unit-test-kb", available_layers=layers,
        available_channels=channels, graph_identity="a" * 64 if layers else None,
        lexical_identity="b" * 64 if "bm25" in channels else None,
        bilingual_lexical_enabled=bilingual, budget_limits=budget())


def proposal(layer="coarse", hybrid=False, lexical=False, intent="overview"):
    layers = {"coarse": ("coarse", "mid", "chunk"), "mid": ("mid", "chunk"), "chunk": ("chunk",)}[layer]
    return {"intent": {"primary": intent}, "requirements": [{"id": "f1", "text": "Topics and their relationships"}],
        "execution_strategy": {"route": "retrieve", "entry_layer": layer,
            "semantic_query": "Overview of the collection", "generate_lexical": lexical,
            "lexical_groups": [{"group_id": "l1", "requirement_ids": ["f1"], "kind": "concept",
                "surfaces": [{"text": "topics", "language": "en", "provenance": "model_query"}]}] if lexical else [],
            "hybrid": hybrid, "layer_weights": {item: {"dense": .45, "rq": .15, "bm25": .4} if hybrid
                else {"dense": 1, "rq": 0, "bm25": 0} for item in layers},
            "selection_scope": "broad", "reason_code": "broad_scope"}}


def accept(raw, **kwargs):
    return accept_plan(IntentPlanningOutput.model_validate(raw), question=kwargs.get("question", "Summarize the topics."),
        conversation_scope_hash="c" * 64, filter_scope_hash="d" * 64,
        capabilities=kwargs.get("capabilities", capabilities()))


@pytest.mark.parametrize("layer", ["coarse", "mid", "chunk"])
@pytest.mark.parametrize("intent", ["overview", "summarize", "fact_lookup", "compare"])
def test_intent_does_not_force_a_layer_and_empty_lexical_preserves_the_user_task(layer, intent):
    accepted = accept(proposal(layer=layer, intent=intent), capabilities=capabilities(channels=("dense",)))
    assert accepted.task.question == "Summarize the topics."
    assert len(accepted.task.requirements) == 1
    assert accepted.strategy.entry_layer == layer
    assert accepted.strategy.lexical_terms == () and not accepted.strategy.hybrid
    assert "retrieval_granularity" not in accepted.task.model_dump()
    assert "weight" not in accepted.task.requirements[0].model_dump()
    assert accepted.intent.primary == intent


@pytest.mark.parametrize("layer", ["coarse", "mid", "chunk"])
def test_weights_are_frozen_per_layer_and_not_a_shared_mutable_dictionary(layer):
    accepted = accept(proposal(layer=layer, hybrid=True, lexical=True))
    with pytest.raises(ValidationError):
        accepted.strategy.layer_weights.chunk.bm25 = .8
    with pytest.raises(ValidationError):
        accepted.strategy.entry_layer = "mid"
    assert accepted.effective_weights("chunk") == {"dense": .45, "rq": .15, "bm25": .4}


@pytest.mark.parametrize("value", [-.1, 1.1, float("inf"), float("nan"), True, "0.5"])
def test_invalid_weight_is_not_silently_clamped_or_coerced(value):
    with pytest.raises(ValidationError):
        ChannelWeights(dense=value, rq=0, bm25=.5)


def test_roundoff_normalization_retains_raw_values_and_rejects_a_different_plan():
    weights = ChannelWeights(dense=.4000001, rq=.1, bm25=.5)
    assert weights.dense == .4000001
    assert sum(weights.effective().values()) == pytest.approx(1)
    with pytest.raises(ValidationError, match="must_sum_to_one"):
        ChannelWeights(dense=.45, rq=.1, bm25=.5)


@pytest.mark.parametrize("attack", ["hybrid_empty", "flag_mismatch", "bad_pure_weight", "extra_layer", "missing_layer",
    "foreign_requirement", "duplicate_requirement", "duplicate_term", "unknown_field", "old_mode"])
def test_contradictory_or_legacy_plan_is_rejected_before_execution(attack):
    raw = proposal(layer="mid")
    strategy = raw["execution_strategy"]
    if attack == "hybrid_empty": strategy["hybrid"] = True
    elif attack == "flag_mismatch": strategy["generate_lexical"] = True
    elif attack == "bad_pure_weight": strategy["layer_weights"]["mid"] = {"dense": .8, "rq": .2, "bm25": 0}
    elif attack == "extra_layer": strategy["layer_weights"]["coarse"] = {"dense": 1, "rq": 0, "bm25": 0}
    elif attack == "missing_layer": del strategy["layer_weights"]["chunk"]
    elif attack == "foreign_requirement":
        strategy["generate_lexical"] = True
        strategy["lexical_groups"] = [{"group_id": "l1", "requirement_ids": ["f2"], "kind": "concept",
            "surfaces": [{"text": "unknown", "language": "en", "provenance": "model_query"}]}]
    elif attack == "duplicate_requirement": raw["requirements"] *= 2
    elif attack == "duplicate_term":
        raw = proposal(lexical=True)
        raw["execution_strategy"]["lexical_groups"] *= 2
    elif attack == "unknown_field": strategy["path_threshold"] = 99
    elif attack == "old_mode": strategy["retrieval_granularity"] = "coarse"
    with pytest.raises(ValidationError): accept(raw)


def test_no_index_failure_is_disguised_as_healthy_empty_bm25():
    with pytest.raises(ValueError, match="index_unavailable"):
        accept(proposal(hybrid=True, lexical=True), capabilities=capabilities(channels=("dense", "rq")))
    assert accept(proposal(), capabilities=capabilities(channels=("dense",))).strategy.entry_layer == "coarse"


def test_capability_bounds_do_not_change_the_requested_entry_or_budget():
    with pytest.raises(ValueError, match="entry_unavailable"):
        accept(proposal(), capabilities=capabilities(layers=("mid", "chunk")))
    raw = proposal(layer="chunk")
    raw["execution_strategy"]["budget_request"] = {"root_entries": 7}
    with pytest.raises(ValueError, match="budget_exceeds"):
        accept(raw)
    raw["execution_strategy"]["budget_request"] = {"root_entries": 2}
    assert accept(raw).effective_budget.root_entries == 2


def test_user_term_and_protected_literal_need_actual_current_user_text():
    raw = proposal(lexical=True)
    surface = raw["execution_strategy"]["lexical_groups"][0]["surfaces"][0]
    surface["provenance"] = "user_text"
    accepted = accept(raw, question="Explain the materials.")
    assert accepted.strategy.lexical_terms[0].provenance == "model_query"
    surface["provenance"] = "model_query"
    raw["requirements"][0]["protected_literals"] = ["never supplied"]
    accepted = accept(raw)
    assert accepted.task.requirements[0].protected_literals == ()


def test_question_and_weights_independently_change_the_accepted_identity():
    raw = proposal(hybrid=True, lexical=True)
    first = accept(raw)
    second = accept(raw, question="Compare the topics.")
    changed = deepcopy(raw)
    changed["execution_strategy"]["layer_weights"]["mid"] = {"dense": .1, "rq": .4, "bm25": .5}
    third = accept(changed)
    assert len({first.identity, second.identity, third.identity}) == 3
    assert first.task.identity == third.task.identity != second.task.identity


@pytest.mark.parametrize("route", ["system_capability", "clarify"])
def test_direct_routes_have_no_graph_or_lexical_dependencies(route):
    raw = {"intent": {"primary": route}, "execution_strategy": {"route": route, "entry_layer": None,
        "generate_lexical": False, "hybrid": False, "reason_code": "system_request" if route == "system_capability" else "ambiguous_request"}}
    accepted = accept(raw, capabilities=capabilities(channels=(), layers=()))
    assert accepted.task.requirements == ()
    assert accepted.strategy.layer_weights.enabled_layers() == ()
    raw["execution_strategy"]["semantic_query"] = "unrequested corpus work"
    with pytest.raises(ValidationError): accept(raw)


def test_source_scope_requires_quoted_authority_not_guessed_location():
    from test_evidence_scope import obligation, request
    raw = proposal()
    raw["requirements"][0]["source_scope"] = obligation(request("section", "Invented section")).model_dump(mode="json")
    accepted = accept(raw)
    assert accepted.task.requirements[0].source_scope is None


def test_topic_words_are_not_promoted_to_section_addresses_without_structure_cues():
    from test_evidence_scope import obligation, request

    raw = proposal(layer="chunk")
    raw["requirements"][0]["source_scope"] = obligation(
        request("section", "光谱"),
        "overlap",
    ).model_dump(mode="json")
    accepted = accept(raw, question="比较深场/光谱部分的面积和时长。")
    assert accepted.task.requirements[0].source_scope is None

    accepted = accept(raw, question="请核对第光谱节的面积和时长。")
    assert accepted.task.requirements[0].source_scope is not None


def test_any_scope_keeps_valid_structure_role_when_text_title_noise_is_removed():
    from app.retrieval_control_contracts import SourceScopeObligation
    from test_evidence_scope import obligation, request

    raw = proposal(layer="chunk")
    raw["requirements"][0]["source_scope"] = SourceScopeObligation(
        op="any",
        children=(
            obligation(request("section", "摘要"), "overlap"),
            obligation(request("text", "深场"), "overlap"),
        ),
    ).model_dump(mode="json")
    accepted = accept(raw, question="比较摘要与深场部分。")
    scope = accepted.task.requirements[0].source_scope
    assert scope.op == "coverage"
    assert scope.scope.selector.kind == "section"
    assert scope.scope.selector.reference == "摘要"


@pytest.mark.parametrize(
    ("question", "expected_mode"),
    [
        ("根据《Unit report》比较摘要和详细说明。", "overlap"),
        ("请总结《Unit report》的全文。", "complete"),
    ],
)
def test_complete_document_scope_requires_explicit_whole_document_authority(
    question,
    expected_mode,
):
    from test_evidence_scope import obligation, request

    raw = proposal(layer="chunk")
    raw["requirements"][0]["source_scope"] = obligation(
        request("document", "Unit report"),
        "complete",
    ).model_dump(mode="json")
    accepted = accept(raw, question=question)
    assert accepted.task.requirements[0].source_scope.mode == expected_mode


def test_complete_section_scope_requires_explicit_whole_section_authority():
    from test_evidence_scope import obligation, request

    raw = proposal(layer="chunk")
    raw["requirements"][0]["source_scope"] = obligation(
        request("section", "摘要"),
        "complete",
    ).model_dump(mode="json")
    accepted = accept(raw, question="比较摘要中的面积和时长。")
    assert accepted.task.requirements[0].source_scope.mode == "overlap"

    accepted = accept(raw, question="请读取完整摘要并比较摘要中的面积和时长。")
    assert accepted.task.requirements[0].source_scope.mode == "complete"


def test_nonempty_but_untokenizable_lexical_plan_fails_before_retrieval():
    raw = proposal(hybrid=True, lexical=True)
    raw["execution_strategy"]["lexical_groups"][0]["surfaces"][0]["text"] = "..."
    raw["execution_strategy"]["lexical_groups"][0]["surfaces"][0]["language"] = "neutral"
    with pytest.raises(ValueError, match="no_tokens"):
        accept(raw)


def test_bilingual_concept_groups_are_required_only_when_the_capability_is_enabled():
    raw = proposal(layer="chunk", hybrid=True, lexical=True)
    with pytest.raises(ValueError, match="bilingual_concept_surfaces_missing"):
        accept(raw, capabilities=capabilities(bilingual=True))

    raw["execution_strategy"]["lexical_groups"][0]["surfaces"].insert(0, {
        "text": "主题",
        "language": "zh",
        "provenance": "model_query",
    })
    accepted = accept(raw, capabilities=capabilities(bilingual=True))
    assert [(term.text, term.language, term.group_id) for term in accepted.strategy.lexical_terms] == [
        ("主题", "zh", "l1"),
        ("topics", "en", "l1"),
    ]


def test_bilingual_mode_does_not_force_translation_for_language_neutral_identifier():
    raw = proposal(layer="chunk", hybrid=True, lexical=True)
    raw["execution_strategy"]["lexical_groups"] = [{
        "group_id": "l1",
        "requirement_ids": ["f1"],
        "kind": "identifier",
        "surfaces": [{
            "text": "queue_limit",
            "language": "neutral",
            "provenance": "model_query",
        }],
    }]
    accepted = accept(raw, capabilities=capabilities(bilingual=True))
    assert accepted.strategy.lexical_terms[0].text == "queue_limit"


def test_model_cannot_invent_response_constraint_text_or_offsets():
    raw = proposal(layer="chunk")
    raw["response_constraints"] = [
        {"kind": "brevity", "text": "invented", "char_span": [0, 8]}
    ]
    question = "请用列表回答数据中心的作用。"
    accepted = accept(raw, question=question)
    assert [item.kind for item in accepted.task.response_constraints] == [
        "output_format"
    ]
    constraint = accepted.task.response_constraints[0]
    assert question[slice(*constraint.char_span)] == constraint.text
