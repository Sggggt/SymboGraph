"""Deterministic fixed-task coverage and bounded local lexical attribution."""
from __future__ import annotations

from collections import Counter
import math

from app.retrieval_control_contracts import (
    DecisionPanel, FacetCoverage, GainBounds, GateThresholds, LexicalStrategy,
    PathEvaluationParameters, PathFeatureCandidate, PathFeatureSummary, RetrievalGateDecision,
    ScoreBounds, TaskContract, TermInteraction, TermPathFeature, control_hash,
    StrategyGainEstimate,
)
from app.services.qa_performance import qa_stage


def path_quality(candidate: PathFeatureCandidate, parameters: PathEvaluationParameters) -> float:
    if not candidate.source_valid:
        return 0.0
    energy = candidate.canonical_entry_distance / parameters.root_scale
    energy += sum(value / scale for value, scale in zip(candidate.physical_distances, parameters.physical_scales))
    return math.exp(-energy)


def _opportunity(candidate, facet_id):
    return next((item.value for item in candidate.opportunities if item.facet_id == facet_id),
                ScoreBounds(lower=0, upper=1))


def coverage(task, candidates, parameters):
    result = []
    for facet in task.requirements:
        values = [(_opportunity(item, facet.id), path_quality(item, parameters), item)
                  for item in candidates if item.source_valid]
        supported = [item for item in candidates if item.source_valid and item.path_observed]
        best = max(supported, key=lambda item: (_opportunity(item, facet.id).lower *
                   path_quality(item, parameters), item.source_id), default=None)
        result.append(FacetCoverage(facet_id=facet.id,
            coverage=ScoreBounds(lower=max((value.lower for value, _, _ in values), default=0),
                                 upper=max((value.upper for value, _, _ in values), default=0)),
            path_quality=ScoreBounds(lower=max((value.lower * quality if item.path_observed else 0
                                               for value, quality, item in values), default=0),
                                     upper=max((value.upper * quality if item.path_observed else value.upper
                                               for value, quality, item in values), default=0)),
            best_source_id=best.source_id if best is not None and _opportunity(best, facet.id).lower > 0 else None))
    return tuple(result)


def utility(task, candidates, parameters):
    by_id = {item.facet_id: item for item in coverage(task, candidates, parameters)}
    return ScoreBounds(
        lower=min(1.0, sum(f.weight * by_id[f.id].path_quality.lower for f in task.requirements)),
        upper=min(1.0, sum(f.weight * by_id[f.id].path_quality.upper for f in task.requirements)),
    )


def select_panel(task, strategy, panel, *, masked_term_ids=frozenset()):
    active = {term.id: term.facet_id for term in strategy.terms
              if term.id not in masked_term_ids and term.source != "hypothesis"}
    weights = panel.routing_facet_weights or {facet.id: facet.weight for facet in task.requirements}
    def key(candidate):
        matched = {active[term_id] for term_id in candidate.matched_term_ids if term_id in active}
        mass = sum(weights[facet_id] for facet_id in matched)
        if panel.routing_facet_weights:
            mass = round(mass, 6)
        return (not candidate.mandatory, len(weights) - len(matched),
                -mass,
                candidate.routing_cost, candidate.depth, candidate.role_rank, candidate.id)
    return tuple(sorted(panel.candidates, key=key)[:panel.limit])


def _panel_damage(task, strategy, panels, parameters, masked):
    if not panels:
        return GainBounds(lower=-1, upper=1), 0
    lower, upper, changed = 0.0, 0.0, 0
    for panel in panels:
        original = tuple(item for item in panel.candidates if item.id in panel.selected_ids)
        alternative = select_panel(task, strategy, panel, masked_term_ids=masked)
        if {item.id for item in original} == {item.id for item in alternative}:
            # Shared unknown facts cancel exactly when the selected set is unchanged.
            continue
        changed += 1
        old, new = utility(task, original, parameters), utility(task, alternative, parameters)
        lower += new.lower - old.upper
        upper += new.upper - old.lower
    return GainBounds(lower=max(-1, lower / len(panels)), upper=min(1, upper / len(panels))), changed


def compute_path_features(*, task: TaskContract, strategy: LexicalStrategy,
                          panels: tuple[DecisionPanel, ...],
                          packaged_candidates: tuple[PathFeatureCandidate, ...],
                          parameters: PathEvaluationParameters,
                          interaction_facet_limit: int = 2) -> PathFeatureSummary:
    strategy.validate_task(task)
    scope_facets = {facet.id: facet for facet in task.requirements if facet.source_scope is not None}
    if set(scope_facets) != {item.facet_id for item in parameters.scope_inputs}:
        raise ValueError('path_feature_source_scope_missing_or_changed')
    from app.services.evidence_scope import evaluate_scope_obligation
    scope_statuses = tuple(evaluate_scope_obligation(task=task, facet=scope_facets[item.facet_id],
        bound=item, packed=parameters.packed_scope_intervals) for item in parameters.scope_inputs)
    if not 0 <= interaction_facet_limit <= 2 or len(panels) > 64:
        raise ValueError("path_feature_observation_budget_exceeded")
    if len(packaged_candidates) > 256:
        raise ValueError("path_feature_package_budget_exceeded")
    if len({panel.id for panel in panels}) != len(panels):
        raise ValueError("path_feature_duplicate_decision_panel")
    known_facets = {facet.id for facet in task.requirements}
    known_terms = {term.id for term in strategy.terms}
    all_candidates = [*packaged_candidates, *(candidate for panel in panels for candidate in panel.candidates)]
    for candidate in all_candidates:
        if not {item.facet_id for item in candidate.opportunities}.issubset(known_facets):
            raise ValueError("path_feature_candidate_outside_task")
        if not set(candidate.matched_term_ids).issubset(known_terms):
            raise ValueError("path_feature_unknown_matched_term")
    for panel in panels:
        if panel.routing_facet_weights and set(panel.routing_facet_weights) != known_facets:
            raise ValueError('path_feature_routing_weight_scope_invalid')
        if tuple(item.id for item in select_panel(task, strategy, panel)) != panel.selected_ids:
            raise ValueError("path_feature_baseline_selection_mismatch")
    # One source may be reached by several paths; its source facts must agree.
    source_rows = {}
    for candidate in all_candidates:
        if not candidate.source_valid:
            continue
        identity = (candidate.opportunities, candidate.matched_term_ids, candidate.topic_group)
        previous = source_rows.get(candidate.source_id)
        if previous is not None and previous[0] != identity:
            raise ValueError("path_feature_source_facts_conflict")
        source_rows[candidate.source_id] = (identity, candidate)
    with qa_stage("path_features", item_count=len(all_candidates)):
        terms = []
        for term in strategy.terms:
            damage, pivotal = _panel_damage(task, strategy, panels, parameters, {term.id})
            hits = [entry[1] for entry in source_rows.values() if term.id in entry[1].matched_term_ids]
            leak, entropy = None, None
            if hits:
                values = [_opportunity(item, term.facet_id) for item in hits]
                leak = ScoreBounds(lower=max(0, sum(1 - item.upper for item in values) / len(values)),
                                   upper=min(1, sum(1 - item.lower for item in values) / len(values)))
                groups = Counter(item.topic_group for item in hits)
                entropy = (-sum((count / len(hits)) * math.log(count / len(hits)) for count in groups.values())
                           / math.log(len(groups))) if len(groups) > 1 else 0.0
            diagnosis = ("routing_harm" if damage.lower > 1e-9 else
                         "routing_help" if damage.upper < -1e-9 else
                         "not_observed" if not hits else
                         "no_observed_routing_effect" if not pivotal and panels else "uncertain")
            terms.append(TermPathFeature(term_id=term.id, facet_id=term.facet_id,
                observed_source_hits=len(hits), observed_sources=len(source_rows), leak=leak,
                topic_entropy=entropy, routing_damage=damage, pivotal_panel_count=pivotal,
                panel_count=len(panels), diagnosis=diagnosis))
        # Group masks expose redundant aliases which single deletions cannot identify.
        facet_coverage = coverage(task, packaged_candidates, parameters)
        gaps = {item.facet_id: 1 - item.coverage.lower for item in facet_coverage}
        groups = sorted(task.requirements,
            key=lambda item: (-gaps[item.id], -sum(t.routing_damage.lower for t in terms if t.facet_id == item.id), item.id))
        interactions = []
        checked_facets = 0
        for facet in groups:
            ids = tuple(term.id for term in strategy.terms if term.facet_id == facet.id)
            if len(ids) < 2:
                continue
            if checked_facets >= interaction_facet_limit:
                break
            checked_facets += 1
            checks = [ids]
            if len(ids) > 2:
                suspicious = sorted((item for item in terms if item.facet_id == facet.id),
                    key=lambda item: (-(item.leak.lower if item.leak else 0), -item.routing_damage.lower, item.term_id))
                pair = tuple(item.term_id for item in suspicious[:2])
                checks.append(pair)
            for mask in checks:
                damage, changed = _panel_damage(task, strategy, panels, parameters, set(mask))
                interactions.append(TermInteraction(facet_id=facet.id, term_ids=mask,
                    routing_damage=damage, selected_sets_changed=bool(changed)))
        payload = {"task": task.model_dump(mode="json"), "strategy": strategy.model_dump(mode="json"),
                   "panels": [item.model_dump(mode="json") for item in panels],
                   "package": [item.model_dump(mode="json") for item in packaged_candidates],
                   "parameters": parameters.model_dump(mode="json")}
        return PathFeatureSummary(task_hash=task.identity, strategy_hash=strategy.identity,
            evaluation_protocol_hash=control_hash({key:value for key,value in parameters.model_dump(mode="json").items()
                if key != 'packed_scope_intervals'}), input_hash=control_hash(payload),
            facets=facet_coverage, utility=utility(task, packaged_candidates, parameters), terms=tuple(terms),
            interactions=tuple(interactions), observed_panel_count=len(panels),
            candidate_evaluation_count=len(all_candidates),
            counterfactual_selection_count=(len(terms) + len(interactions)) * len(panels),
            invalid_packaged_source_count=sum(not item.source_valid for item in packaged_candidates), scope_statuses=scope_statuses)


def decide_retrieval_gate(*, task: TaskContract, features: PathFeatureSummary, thresholds: GateThresholds,
                          source_integrity: bool, actionable_ids: tuple[str, ...] = (),
                          remaining_repairs: int, package_complete: bool = True,
                          technical_error: bool = False) -> RetrievalGateDecision:
    if task.identity != features.task_hash:
        raise ValueError("retrieval_gate_task_identity_changed")
    if remaining_repairs < 0 or remaining_repairs > 2:
        raise ValueError("retrieval_gate_budget_invalid")
    by_id = {item.facet_id: item for item in features.facets}
    if set(by_id) != {item.id for item in task.requirements}:
        raise ValueError("retrieval_gate_facet_scope_invalid")
    if {facet.id for facet in task.requirements if facet.source_scope is not None} != {item.facet_id for item in features.scope_statuses}:
        raise ValueError('retrieval_gate_source_obligation_missing')
    missing = tuple(f.id for f in task.requirements
                    if by_id[f.id].coverage.lower < thresholds.coverage
                    or by_id[f.id].path_quality.lower < thresholds.path_quality
                    or any(item.facet_id == f.id and item.state != 'satisfied' for item in features.scope_statuses))
    if technical_error:
        outcome, reasons = "technical_failure", ("dependency_failure",)
    elif not source_integrity or not package_complete or features.invalid_packaged_source_count:
        outcome, reasons = "source_incomplete", ("source_scope_not_verified",)
    elif not missing:
        outcome, reasons = "ready_full", ("fixed_task_coverage_passed",)
    elif actionable_ids and remaining_repairs:
        outcome, reasons = "repairable", ("attested_repair_direction_available",)
    elif any(item.state == 'unknown' for item in features.scope_statuses):
        ambiguous = any('ambiguous' in item.reason_codes for item in features.scope_statuses)
        incomplete = any(set(item.reason_codes) & {'representation_incomplete','unsupported_representation'} for item in features.scope_statuses)
        outcome, reasons = ('scope_ambiguous' if ambiguous else 'representation_incomplete' if incomplete else 'source_unresolved'), ('source_location_not_resolved',)
    elif task.allow_partial and any(f.id not in missing for f in task.requirements):
        outcome, reasons = "ready_partial", ("explicit_partial_scope",)
    elif actionable_ids and not remaining_repairs:
        outcome, reasons = "budget_exhausted", ("repair_budget_exhausted",)
    else:
        outcome, reasons = "scoped_not_found", ("no_supported_repair_direction",)
    return RetrievalGateDecision(outcome=outcome, missing_facet_ids=missing, reason_codes=reasons,
        feature_hash=control_hash(features.model_dump(mode="json")),
        threshold_hash=control_hash(thresholds.model_dump(mode="json")),
        proposed_action_ids=actionable_ids if outcome == "repairable" else ())


def estimate_strategy_gain(*, task: TaskContract, before: LexicalStrategy, after: LexicalStrategy,
                           original_panels: tuple[DecisionPanel, ...],
                           rematched_panels: tuple[DecisionPanel, ...],
                           parameters: PathEvaluationParameters) -> StrategyGainEstimate:
    before.validate_task(task)
    after.validate_task(task)
    if len(original_panels) > 64 or len(original_panels) != len(rematched_panels):
        raise ValueError("counterfactual_panel_scope_changed")
    lower, upper, changed = 0.0, 0.0, 0
    for old_panel, new_panel in zip(original_panels, rematched_panels):
        if old_panel.model_dump(exclude={"candidates"}) != new_panel.model_dump(exclude={"candidates"}):
            raise ValueError("counterfactual_panel_scope_changed")
        old_facts = [item.model_dump(exclude={"matched_term_ids"}) for item in old_panel.candidates]
        new_facts = [item.model_dump(exclude={"matched_term_ids"}) for item in new_panel.candidates]
        if old_facts != new_facts:
            raise ValueError("counterfactual_evaluation_or_eligibility_changed")
        previous = select_panel(task, before, old_panel)
        if tuple(item.id for item in previous) != old_panel.selected_ids:
            raise ValueError("counterfactual_baseline_selection_mismatch")
        proposed = select_panel(task, after, new_panel)
        if {item.id for item in previous} == {item.id for item in proposed}:
            continue
        changed += 1
        baseline, candidate = utility(task, previous, parameters), utility(task, proposed, parameters)
        lower += candidate.lower - baseline.upper
        upper += candidate.upper - baseline.lower
    count = len(original_panels)
    gain = GainBounds(lower=max(-1, lower / count), upper=min(1, upper / count)) if count else GainBounds(lower=-1, upper=1)
    return StrategyGainEstimate(task_hash=task.identity, before_strategy_hash=before.identity,
        after_strategy_hash=after.identity, gain=gain, changed_panel_count=changed)
