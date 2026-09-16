"""Local admission to semantic assessment, never direct generation authority.

Uses only frozen measurements and the actual manifest. Database source/trace
replay remains required at the controller and final source-binding boundaries.
"""
from __future__ import annotations

from app.retrieval_control_contracts import (
    EvidenceInterval, EvidenceScopeExpression, GenerationSourceScopeGuidance,
    PathEvaluationParameters, PathFeatureCandidate, SourceAddressedAssessment,
    SourceAddressedFacet, control_hash,
)
from app.services.evidence_scope import generation_scope_guidance
from app.services.retrieval_path_features import path_quality
from app.services.source_use import evaluate_evidence_scope, _merge_scope_intervals, _intersect_scope_intervals
from app.services.storage import raise_if_source_io_cancelled


PROTOCOL = 'source_addressed_assessment_v1'
ELIGIBLE_OUTCOMES = frozenset({'scoped_not_found', 'repairable', 'budget_exhausted', 'ready_partial'})


def build_assessment(*, task, features, decision, replay_input, evidence, source_audit, thresholds):
    """Return an exact proof, or None when this closed branch is inapplicable."""
    raise_if_source_io_cancelled()
    if (decision.outcome not in ELIGIBLE_OUTCOMES or not source_audit['all_valid']
            or features.invalid_packaged_source_count
            or not any(f.coverage.lower < thresholds.coverage for f in features.facets)
            or any(f.path_quality.lower < thresholds.path_quality for f in features.facets)
            or any(f.source_scope is None for f in task.requirements)
            or any(s.state != 'satisfied' for s in features.scope_statuses)):
        return None
    if (features.task_hash != task.identity or control_hash(replay_input) != features.input_hash
            or decision.feature_hash != control_hash(features.model_dump(mode='json'))
            or decision.threshold_hash != control_hash(thresholds.model_dump(mode='json'))):
        raise ValueError('source_addressed_feature_identity_changed')
    evidence.verify_integrity()
    parameters = PathEvaluationParameters.model_validate(replay_input['parameters'])
    if parameters.protocol_version != 'canonical_task_path_quality_v4':
        return None
    groups = {item.facet_id: item for item in parameters.scope_inputs}
    if set(groups) != {f.id for f in task.requirements}:
        return None
    candidates = tuple(PathFeatureCandidate.model_validate(item) for item in replay_input['package'])
    by_source = {}
    for source in evidence.sources:
        start, end = source['source_span']['char_span']
        if end - start != len(source['text']):
            return None
        key = f"{source['chunk_id']}:{start}:{end}"
        by_source[key] = source
    if {c.source_id for c in candidates} != set(by_source):
        raise ValueError('source_addressed_package_measurement_changed')
    records = []
    for facet in task.requirements:
        raise_if_source_io_cancelled()
        metric = next(f for f in features.facets if f.facet_id == facet.id)
        # Every covered range must be witnessed by an actually observed path
        # with the unchanged per-facet path-quality lower bound.
        eligible = []
        for candidate in candidates:
            value = next((o.value.lower for o in candidate.opportunities if o.facet_id == facet.id), 0)
            if candidate.source_valid and candidate.path_observed and value * path_quality(candidate, parameters) >= thresholds.path_quality:
                eligible.append(candidate.source_id)
        packed = tuple(EvidenceInterval(knowledge_base_id=task.knowledge_base_id,
            document_version_id=by_source[key]['source_span']['document_version_id'],
            start=by_source[key]['source_span']['char_span'][0], end=by_source[key]['source_span']['char_span'][1])
            for key in eligible)
        bindings = {b.request_hash: b for b in groups[facet.id].bindings}
        proofs = []

        def complete(node):
            raise_if_source_io_cancelled()
            if node.op != 'coverage':
                choices = [complete(child) for child in node.children]
                if node.op == 'any':
                    return next((choice for choice in choices if choice is not None), None)
                return tuple(key for choice in choices for key in choice) if all(c is not None for c in choices) else None
            key = control_hash(node.scope.model_dump(mode='json'))
            binding = bindings.get(key)
            if binding is None:
                raise ValueError('source_addressed_binding_missing')
            proof = evaluate_evidence_scope(knowledge_base_id=task.knowledge_base_id,
                expression=EvidenceScopeExpression(op='scope', scope_id=binding.fact.id),
                scopes=(binding.fact,), packed=packed, mode='complete')
            proofs.append(proof)
            if (binding.reason != 'resolved' or proof.state != 'satisfied'
                    or not proof.scope_extent_known or not proof.usable_intervals):
                return None
            return (key,)

        selected = complete(facet.source_scope)
        if selected is None:
            return None
        selected = tuple(dict.fromkeys(selected))
        ranges = _merge_scope_intervals((p.knowledge_base_id, p.document_version_id, p.start, p.end)
            for key in selected for p in bindings[key].fact.intervals)
        sources, used_paths = [], []
        for key in eligible:
            source = by_source[key]
            span = source['source_span']
            start, end = span['char_span']
            intersections = _intersect_scope_intervals(((task.knowledge_base_id, span['document_version_id'], start, end),), ranges)
            if intersections:
                used_paths.append(key)
                sources.append({'source_handle': source['source_handle'],
                    'text_char_spans': [(left-start, right-start) for _, _, left, right in intersections]})
        records.append(SourceAddressedFacet(facet_id=facet.id, coverage=metric.coverage,
            path_quality=metric.path_quality, complete_coverage=tuple(proofs), selected_request_hashes=selected,
            usable_intervals=tuple(EvidenceInterval(knowledge_base_id=k, document_version_id=v, start=l, end=r) for k,v,l,r in ranges),
            path_source_ids=tuple(used_paths), sources=tuple(sources)))
    return SourceAddressedAssessment(task_hash=task.identity, strategy_hash=features.strategy_hash,
        feature_input_hash=features.input_hash, feature_hash=decision.feature_hash, thresholds=thresholds,
        context_package_id=evidence.package_id, retrieval_trace_id=evidence.retrieval_trace_id,
        evidence_manifest_hash=evidence.manifest_hash, provenance_session_hash=source_audit['provenance_session_hash'],
        facets=tuple(records))


def assessment_guidance(assessment):
    return GenerationSourceScopeGuidance(requirements=[{'requirement_id':f.facet_id, 'sources':f.sources}
        for f in assessment.facets])


def replay_assessment(payload, *, task, features, path_decision, replay_input, evidence, source_audit=None):
    """Validate proof by recomputation; checking only a supplied hash is insufficient."""
    raw = payload.get('source_addressed_assessment')
    if raw is None:
        if payload.get('source_addressed_assessment_hash') is not None:
            raise ValueError('source_addressed_assessment_missing')
        return None
    assessment = SourceAddressedAssessment.model_validate(raw)
    audit = source_audit or {'all_valid':True, 'provenance_session_hash':assessment.provenance_session_hash}
    expected = build_assessment(task=task, features=features, decision=path_decision,
        replay_input=replay_input, evidence=evidence, source_audit=audit, thresholds=assessment.thresholds)
    if expected != assessment or payload.get('source_addressed_assessment_hash') != assessment.identity:
        raise ValueError('source_addressed_assessment_replay_changed')
    return assessment


def replay_assessment_guidance(payload, *, task, features, path_decision, replay_input, evidence, source_audit=None):
    assessment = replay_assessment(payload, task=task, features=features, path_decision=path_decision,
        replay_input=replay_input, evidence=evidence, source_audit=source_audit)
    guidance = assessment_guidance(assessment) if assessment else generation_scope_guidance(task=task,
        parameters=PathEvaluationParameters.model_validate(replay_input['parameters']), evidence=evidence)
    return assessment, guidance


def verify_assessment_owner(owner, card):
    """A legacy-shaped admission must not erase a new run's local proof."""
    from app.retrieval_control_contracts import GateThresholds, PathFeatureSummary, TaskContract
    from app.services.retrieval_path_features import decide_retrieval_gate
    call = card.get('evidence_sufficiency') or {}
    raw = call.get('source_addressed_assessment')
    admission = card.get('source_admission') or {}
    metadata = (owner.metadata_json or {}) if owner is not None else {}
    if raw is not None:
        proof = SourceAddressedAssessment.model_validate(raw)
        if (metadata.get('source_addressed_assessment_protocol') != PROTOCOL
                or metadata.get('retrieval_gate_thresholds') != proof.thresholds.model_dump(mode='json')
                or call.get('protocol_version') != 'retrieval_sufficiency_call_v2'
                or call.get('source_addressed_assessment_hash') != proof.identity
                or (admission and admission.get('source_addressed_assessment_hash') != proof.identity)):
            raise ValueError('source_addressed_owner_identity_changed')
    elif (call.get('protocol_version') == 'retrieval_sufficiency_call_v2'
            or admission.get('source_addressed_assessment_hash') is not None):
        raise ValueError('source_addressed_assessment_missing')
    if metadata.get('source_addressed_assessment_protocol') == PROTOCOL and admission:
        features = PathFeatureSummary.model_validate(card['features'])
        task = TaskContract.model_validate(card['feature_input']['task'])
        thresholds = GateThresholds.model_validate(metadata['retrieval_gate_thresholds'])
        numeric = decide_retrieval_gate(task=task, features=features, thresholds=thresholds,
            source_integrity=True, remaining_repairs=0)
        if (admission['outcome'] == 'ready_full' and numeric.outcome != 'ready_full' and raw is None):
            raise ValueError('source_addressed_assessment_missing')


def replay_persisted_assessment(db, *, owner, card):
    """P&E/acceptance replay includes durable scopes and actual source provenance."""
    from app.models import ContextPackage
    from app.retrieval_control_contracts import TaskContract, LexicalStrategy, DecisionPanel, RetrievalGateDecision
    from app.services.answer_sources import build_answer_evidence_manifest
    from app.services.context_graph import context_package_to_contexts
    from app.services.citation_provenance import audit_citation_provenance
    from app.services.reflection_sources import source_citation
    from app.services.evidence_scope import replay_scope_inputs
    from app.services.retrieval_path_features import compute_path_features
    verify_assessment_owner(owner, card)
    call = card.get('evidence_sufficiency') or {}
    if not call.get('source_addressed_assessment'):
        return None
    inputs = card['feature_input']
    task = TaskContract.model_validate(inputs['task'])
    parameters = PathEvaluationParameters.model_validate(inputs['parameters'])
    features = compute_path_features(task=task, strategy=LexicalStrategy.model_validate(inputs['strategy']),
        panels=tuple(DecisionPanel.model_validate(item) for item in inputs['panels']),
        packaged_candidates=tuple(PathFeatureCandidate.model_validate(item) for item in inputs['package']), parameters=parameters)
    package = db.get(ContextPackage, card['context_package_id'])
    if package is None or features.model_dump(mode='json') != card['features']:
        raise ValueError('source_addressed_persisted_input_changed')
    replay_scope_inputs(db, task=task, parameters=parameters, package=package)
    contexts = context_package_to_contexts(package)
    evidence = build_answer_evidence_manifest(package, contexts)
    source_audit = audit_citation_provenance(db, knowledge_base_id=task.knowledge_base_id,
        package=package, contexts=contexts, citations=[source_citation(item,package) for item in package.package_json['chunks']])
    return replay_assessment(call, task=task, features=features,
        path_decision=RetrievalGateDecision.model_validate(card['path_decision']),
        replay_input=inputs, evidence=evidence, source_audit=source_audit)


def replay_assessment_call(db, *, owner, call):
    from app.models import AgentObservation
    row = db.get(AgentObservation, call.get('source_addressed_gate_observation_id'))
    if (row is None or row.run_id != owner.id or row.observation_type != 'retrieval_gate'
            or row.observation_json['control_sequence_index'] >= call['control_sequence_index']):
        raise ValueError('source_addressed_call_gate_missing')
    card = row.observation_json
    if card['context_package_id'] != call['context_package_id'] or card.get('source_admission'):
        raise ValueError('source_addressed_call_gate_changed')
    return replay_persisted_assessment(db, owner=owner,
        card={**card, 'path_decision':card['decision'], 'evidence_sufficiency':call})


def validate_bound_source_addresses(card, addresses):
    """Generation may cite only addresses admitted for at least one requirement."""
    raw = (card.get('evidence_sufficiency') or {}).get('source_addressed_assessment')
    if raw is None:
        return
    proof = SourceAddressedAssessment.model_validate(raw)
    allowed = {key for facet in proof.facets for key in facet.path_source_ids}
    if any(f'{chunk_id}:{span[0]}:{span[1]}' not in allowed for chunk_id, span in addresses):
        raise ValueError('source_addressed_generated_source_outside_scope')
