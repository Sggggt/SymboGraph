import pytest
from pydantic import ValidationError

from app.retrieval_control_contracts import (
    DecisionPanel, FacetOpportunity, GateThresholds, LexicalStrategy, LexicalTerm,
    PathEvaluationIdentity, PathEvaluationParameters, PathFeatureCandidate, Requirement,
    ScoreBounds, TaskContract,
)
from app.services.retrieval_path_features import (
    compute_path_features, decide_retrieval_gate, estimate_strategy_gain, select_panel, utility,
)


def task_fixture():
    return TaskContract(knowledge_base_id="unit-test-kb", conversation_scope_hash="a" * 64,
        question="Find maximum queue waiting time.",
        requirements=(Requirement(id="f1", text="maximum queue waiting time", weight=1),))


def parameters():
    return PathEvaluationParameters(identity=PathEvaluationIdentity(graph_scope_hash="a" * 64,
        vector_runtime_hash="b" * 64, source_scope_hash="c" * 64, canonical_vectors_hash="d" * 64,
        match_protocol_hash="e" * 64))


def strategy(task, surfaces):
    return LexicalStrategy(task_hash=task.identity, revision=0,
        terms=tuple(LexicalTerm(id=value, facet_id="f1", surface=value) for value in surfaces),
        routing_text=task.requirements[0].text)


def candidate(name, matched=(), *, value=.9, upper=None, distance=.1, **kwargs):
    return PathFeatureCandidate(id=name, source_id=name, topic_group=name, source_valid=True,
        opportunities=(FacetOpportunity(facet_id="f1", value=ScoreBounds(lower=value, upper=value if upper is None else upper)),),
        canonical_entry_distance=distance, matched_term_ids=matched, routing_cost=distance, **kwargs)


def features(task, lexical, pool, *, packaged=None):
    panel = DecisionPanel(id="panel", candidates=pool, selected_ids=(), limit=1)
    panel = panel.model_copy(update={"selected_ids": tuple(item.id for item in select_panel(task, lexical, panel))})
    result = compute_path_features(task=task, strategy=lexical, panels=(panel,),
        packaged_candidates=packaged or tuple(item for item in pool if item.id in panel.selected_ids),
        parameters=parameters())
    return result, panel


def test_absent_surface_zero_route_damage_and_attested_replacement_gain():
    task = task_fixture()
    lexical = strategy(task, ("absent",))
    bad = candidate("bad", value=.1)
    good = candidate("good", value=.95, distance=.3)
    result, panel = features(task, lexical, (bad, good))
    assert result.terms[0].diagnosis == "not_observed"
    assert result.terms[0].routing_damage.lower == result.terms[0].routing_damage.upper == 0
    assert result.terms[0].corpus_absence_proven is False
    proposed = strategy(task, ("attested",)).model_copy(update={"revision": 1})
    rematched = panel.model_copy(update={"candidates": (bad, good.model_copy(update={"matched_term_ids": ("attested",)}))})
    gain = estimate_strategy_gain(task=task, before=lexical, after=proposed,
        original_panels=(panel,), rematched_panels=(rematched,), parameters=parameters())
    assert gain.gain.lower == pytest.approx(.613293567, abs=1e-6)
    assert gain.actual_retrieval_gain is False
    forged = rematched.model_copy(update={"candidates": (bad, candidate("good", ("attested",), value=1, distance=.3))})
    with pytest.raises(ValueError, match="evaluation_or_eligibility_changed"):
        estimate_strategy_gain(task=task, before=lexical, after=proposed,
            original_panels=(panel,), rematched_panels=(forged,), parameters=parameters())


def test_redundant_bad_alias_pair_is_detected_without_removing_useful_alias():
    task = task_fixture()
    lexical = strategy(task, ("broad", "broad-copy", "precise"))
    result, _ = features(task, lexical, (
        candidate("bad", ("broad", "broad-copy"), value=.1),
        candidate("good", ("precise",), value=.95, distance=.3)))
    singles = {item.term_id: item for item in result.terms}
    assert singles["broad"].routing_damage.lower == singles["broad-copy"].routing_damage.lower == 0
    joint = next(item for item in result.interactions if set(item.term_ids) == {"broad", "broad-copy"})
    assert joint.routing_damage.lower > 0
    assert result.model_call_count == 0


def test_rare_helpful_term_is_not_penalized_for_low_frequency():
    task = task_fixture()
    lexical = strategy(task, ("rare",))
    result, _ = features(task, lexical, (
        candidate("bad", value=.1), candidate("good", ("rare",), value=.95, distance=.3)))
    assert result.terms[0].observed_source_hits == 1
    assert result.terms[0].diagnosis == "routing_help"


def test_unknown_and_invalid_sources_cannot_silently_pass_the_gate():
    task = task_fixture()
    lexical = strategy(task, ("term",))
    uncertain, _ = features(task, lexical, (candidate("source", ("term",), value=0, upper=1),))
    thresholds = GateThresholds(coverage=.5, path_quality=.3, calibration_id="unit-test-calibration")
    decision = decide_retrieval_gate(task=task, features=uncertain, thresholds=thresholds,
                                    source_integrity=True, remaining_repairs=0)
    assert decision.outcome == "scoped_not_found" and decision.corpus_absence_proven is False
    invalid, _ = features(task, lexical, (candidate("source", ("term",)),),
        packaged=(candidate("invalid", ("term",)).model_copy(update={"source_valid": False}),))
    assert decide_retrieval_gate(task=task, features=invalid, thresholds=thresholds,
        source_integrity=True, remaining_repairs=0).outcome == "source_incomplete"


def test_repeating_a_path_or_adding_a_cycle_cannot_raise_utility():
    task = task_fixture()
    source = candidate("source", value=.9)
    once = utility(task, (source,), parameters())
    assert utility(task, (source, source, source), parameters()) == once
    cycle = source.model_copy(update={"physical_distances": (0, 0, 1)})
    assert utility(task, (cycle,), parameters()).upper < once.lower


def test_forged_baseline_or_different_task_cannot_be_used_for_attribution():
    task = task_fixture()
    lexical = strategy(task, ("term",))
    result, panel = features(task, lexical, (
        candidate("first", ("term",)), candidate("second", distance=.2)))
    forged = panel.model_copy(update={"selected_ids": ("second",)})
    with pytest.raises(ValueError, match="baseline_selection_mismatch"):
        compute_path_features(task=task, strategy=lexical, panels=(forged,),
            packaged_candidates=panel.candidates, parameters=parameters())
    changed_task = task.model_copy(update={"question": "A different task"})
    with pytest.raises(ValueError, match="task_identity_changed"):
        decide_retrieval_gate(task=changed_task, features=result,
            thresholds=GateThresholds(coverage=.5, path_quality=.3, calibration_id="unit-test"),
            source_integrity=True, remaining_repairs=1)


def test_unchanged_selection_cancels_shared_unknown_exactly():
    task = task_fixture()
    lexical = strategy(task, ("absent",))
    result, _ = features(task, lexical, (candidate("unknown", value=0, upper=1),))
    assert result.terms[0].routing_damage.lower == result.terms[0].routing_damage.upper == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1])
def test_non_finite_or_negative_distance_is_rejected(value):
    with pytest.raises(ValidationError):
        candidate("bad", distance=value)
