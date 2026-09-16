"""Bounded pre-generation interpretation; never a source fact or reward label."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.retrieval_control_contracts import ControlContract, control_hash


PROTOCOL = 'retrieval_evidence_sufficiency_v1'
READY = frozenset({'ready_full', 'ready_partial'})


def sufficiency_model_limits(settings):
    """Separate long evidence assessment from short lexical-patch work."""
    return (min(settings.model_request_timeout_seconds, settings.retrieval_sufficiency_timeout_seconds),
            min(settings.chat_json_max_tokens, settings.retrieval_sufficiency_max_tokens))


class RequirementSufficiency(ControlContract):
    facet_id: str = Field(min_length=1, max_length=160)
    status: Literal['covered', 'missing', 'uncertain']
    reason: Literal['supported', 'missing_attribute', 'incomplete_set', 'wrong_source',
                    'conflicting_scope', 'ambiguous_evidence', 'no_usable_evidence']
    source_handles: tuple[str, ...] = Field(default=(), max_length=16)
    gap: str = Field(default='', max_length=240)

    @model_validator(mode='after')
    def consistent_result(self):
        if (self.status == 'covered') != (self.reason == 'supported'):
            raise ValueError('sufficiency_status_reason_conflict')
        if self.status == 'covered' and (not self.source_handles or self.gap):
            raise ValueError('sufficiency_coverage_needs_sources_without_gap')
        if self.status != 'covered' and not self.gap.strip():
            raise ValueError('sufficiency_gap_required')
        if len(set(self.source_handles)) != len(self.source_handles):
            raise ValueError('sufficiency_duplicate_source')
        return self


class EvidenceSufficiency(ControlContract):
    protocol_version: Literal['retrieval_evidence_sufficiency_v1'] = PROTOCOL
    requirements: tuple[RequirementSufficiency, ...] = Field(min_length=1, max_length=8)
    question_complete: bool
    unrepresented_question_span: str = Field(default='', max_length=400)
    affected_facet_ids: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode='after')
    def question_responsibility(self):
        if self.question_complete:
            if self.unrepresented_question_span or self.affected_facet_ids:
                raise ValueError('sufficiency_complete_question_has_gap')
        elif not self.unrepresented_question_span.strip() or not self.affected_facet_ids:
            raise ValueError('sufficiency_question_gap_needs_user_span_and_facets')
        return self


def sufficiency_packet(*, task, evidence, source_scopes):
    return {'current_user': {'question': task.question,
                'response_constraints': [item.model_dump(mode='json') for item in task.response_constraints]},
            'requirements': [item.model_dump(mode='json') for item in task.requirements],
            'evidence': evidence.model_sources(),
            'source_scopes': source_scopes.model_dump(mode='json') if source_scopes else None}


def validate_sufficiency(result, *, task, evidence, source_scopes):
    ids = {item.id for item in task.requirements}
    if len(result.requirements) != len(ids) or {item.facet_id for item in result.requirements} != ids:
        raise ValueError('sufficiency_required_facets_changed')
    if (not set(result.affected_facet_ids) <= ids
            or len(set(result.affected_facet_ids)) != len(result.affected_facet_ids)
            or (not result.question_complete and result.unrepresented_question_span not in task.question)):
        raise ValueError('sufficiency_original_question_changed')
    handles = set(evidence.by_handle())
    scope_handles = {}
    if source_scopes:
        for item in source_scopes.requirements:
            scope_handles[item.requirement_id] = {part.source_handle for part in item.sources}
    for item in result.requirements:
        if not set(item.source_handles) <= handles:
            raise ValueError('sufficiency_source_outside_current_package')
        if (item.status == 'covered' and item.facet_id in scope_handles
                and not set(item.source_handles) <= scope_handles[item.facet_id]):
            raise ValueError('sufficiency_source_outside_required_scope')
    return result


def constrain_gate(decision, *, task, result, remaining_repairs, actionable_ids=(), assessment=None):
    """Semantic readiness needs either numeric readiness or a replayed local proof."""
    if decision.outcome not in READY and decision.outcome not in {'scoped_not_found', 'repairable', 'budget_exhausted'}:
        return decision
    if assessment is not None:
        if (assessment.task_hash != task.identity or assessment.feature_hash != decision.feature_hash
                or control_hash(assessment.thresholds.model_dump(mode='json')) != decision.threshold_hash):
            raise ValueError('source_addressed_gate_identity_changed')
    missing = set(decision.missing_facet_ids) if assessment is None else set()
    missing.update(item.facet_id for item in result.requirements if item.status != 'covered')
    missing.update(result.affected_facet_ids)
    if not missing:
        if assessment is not None:
            return decision.model_copy(update={'outcome':'ready_full', 'missing_facet_ids':(),
                'reason_codes':('source_addressed_evidence_sufficient',), 'proposed_action_ids':()})
        return decision
    ordered = tuple(item.id for item in task.requirements if item.id in missing)
    if actionable_ids and remaining_repairs:
        outcome = 'repairable'
    elif assessment is None and task.allow_partial and result.question_complete and len(missing) < len(task.requirements) and decision.outcome in READY:
        outcome = 'ready_partial'
    else:
        outcome = 'budget_exhausted' if not remaining_repairs else 'scoped_not_found'
    return decision.model_copy(update={'outcome': outcome, 'missing_facet_ids': ordered,
        'reason_codes': ('pre_generation_evidence_gap',),
        'proposed_action_ids': tuple(actionable_ids) if outcome == 'repairable' else ()})


def repair_feedback(result, missing_facet_ids):
    """Only short gaps reach the patch model, never the full evidence packet."""
    missing = set(missing_facet_ids)
    return {'protocol_version': PROTOCOL, 'is_evidence': False,
            'requirements': [item.model_dump(mode='json', exclude={'source_handles'})
                             for item in result.requirements if item.facet_id in missing],
            'unrepresented_question_span': result.unrepresented_question_span,
            'new_terms_must_come_from_supplied_candidates': True}


def replay_sufficiency(payload, *, task, strategy_hash, evidence, source_scopes):
    protocol = payload.get('protocol_version')
    if ((protocol == 'retrieval_sufficiency_call_v2') != bool(payload.get('source_addressed_assessment'))
            or (protocol == 'retrieval_sufficiency_call_v2') != bool(payload.get('source_addressed_assessment_hash'))):
        raise ValueError('source_addressed_call_protocol_changed')
    if (protocol not in {'retrieval_sufficiency_call_v1','retrieval_sufficiency_call_v2'}
            or payload.get('status') != 'completed' or payload.get('task_hash') != task.identity
            or payload.get('strategy_hash') != strategy_hash
            or payload.get('context_package_id') != evidence.package_id
            or payload.get('evidence_manifest_hash') != evidence.manifest_hash
            or payload.get('input_hash') != control_hash(sufficiency_packet(task=task, evidence=evidence, source_scopes=source_scopes))
            or payload.get('audit_hash') != control_hash({k: v for k, v in payload.items() if k != 'audit_hash'})):
        raise ValueError('sufficiency_input_or_audit_changed')
    return validate_sufficiency(EvidenceSufficiency.model_validate(payload['result']),
        task=task, evidence=evidence, source_scopes=source_scopes)
