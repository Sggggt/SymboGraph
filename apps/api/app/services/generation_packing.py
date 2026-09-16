"""Preserve the gate's source witnesses while avoiding a capacity-filled prompt."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.retrieval_control_contracts import ControlContract, PathEvaluationParameters, PathFeatureCandidate, control_hash
from app.services.retrieval_path_features import path_quality
from app.services.chunking import rough_token_count


class GenerationPackingPlan(ControlContract):
    protocol_version: Literal['feature_preserving_generation_packing_v1'] = 'feature_preserving_generation_packing_v1'
    task_hash: str
    feature_input_hash: str
    source_context_package_id: str
    witness_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    neighbor_chunk_ids: tuple[str, ...] = Field(max_length=80)
    selection_token_budget: int = Field(gt=0)
    reserved_token_budget: int = Field(ge=0)
    original_source_count: int = Field(gt=0)
    selected_source_count: int = Field(gt=0)
    original_text_characters: int = Field(gt=0)
    selected_text_characters: int = Field(gt=0)
    model_call_count: Literal[0] = 0


class ScopeGenerationPackingPlan(ControlContract):
    protocol_version: Literal['scope_preserving_generation_packing_v1'] = 'scope_preserving_generation_packing_v1'
    task_hash: str
    feature_input_hash: str
    source_context_package_id: str
    scope_input_hash: str
    source_items_hash: str
    selected_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    selection_token_budget: int = Field(gt=0)
    reserved_token_budget: int = Field(ge=0)
    original_source_count: int = Field(gt=0, le=256)
    selected_source_count: int = Field(gt=0, le=256)
    original_text_characters: int = Field(gt=0)
    selected_text_characters: int = Field(gt=0)
    all_intersecting_sources_kept: Literal[True] = True
    source_text_modified: Literal[False] = False
    model_call_count: Literal[0] = 0


def plan_scope_generation_packing(*, task, package, features, replay_input):
    """Preserve every source in the declared domain, not just numeric maxima.

    This is a plan only; materialization and durable source replay are required
    before the resulting package can be used by an evaluator or generator.
    """
    from app.services.source_use import _intersect_scope_intervals, _merge_scope_intervals
    from app.services.storage import raise_if_source_io_cancelled
    raise_if_source_io_cancelled()
    if (task.identity != features.task_hash or control_hash(replay_input) != features.input_hash
            or replay_input['task'] != task.model_dump(mode='json')):
        raise ValueError('scope_generation_packing_input_changed')
    if any(f.source_scope is None for f in task.requirements):
        return None
    parameters=PathEvaluationParameters.model_validate(replay_input['parameters'])
    if (parameters.protocol_version!='canonical_task_path_quality_v4'
            or {f.id for f in task.requirements}!={f.facet_id for f in parameters.scope_inputs}
            or any(b.reason!='resolved' or b.fact.resolution!='verified' or not b.fact.extent_complete
                   for f in parameters.scope_inputs for b in f.bindings)):
        return None
    items=package.package_json['chunks']
    if not items or len(items)>256 or features.invalid_packaged_source_count:
        return None
    ids=[item['chunk_id'] for item in items]
    from app.services.evidence_scope import package_scope_intervals
    if (len(set(ids))!=len(ids) or set(ids)!={item['id'] for item in replay_input['package']}
            or len(ids)!=len(replay_input['package'])
            or package_scope_intervals(task,package)!=parameters.packed_scope_intervals):
        raise ValueError('scope_generation_packing_package_changed')
    intervals=_merge_scope_intervals((p.knowledge_base_id,p.document_version_id,p.start,p.end)
        for f in parameters.scope_inputs for b in f.bindings for p in b.fact.intervals)
    selected=[]
    for item in items:
        raise_if_source_io_cancelled()
        current=(task.knowledge_base_id,item['document_version_id'],*item['char_span'])
        if _intersect_scope_intervals((current,),intervals):
            if item.get('content_clipped') or item.get('role')=='preserved_source':
                return None
            selected.append(item)
    if not selected or len(selected)==len(items):
        return None
    tokens=sum(rough_token_count(item['content']) for item in selected)
    if not 0<tokens<package.token_budget:
        return None
    return ScopeGenerationPackingPlan(task_hash=task.identity,feature_input_hash=features.input_hash,
        source_context_package_id=package.id,scope_input_hash=control_hash([f.model_dump(mode='json') for f in parameters.scope_inputs]),
        source_items_hash=control_hash(items),
        selected_chunk_ids=tuple(item['chunk_id'] for item in selected),selection_token_budget=tokens,
        reserved_token_budget=package.token_budget-tokens,original_source_count=len(items),selected_source_count=len(selected),
        original_text_characters=sum(len(item['content']) for item in items),
        selected_text_characters=sum(len(item['content']) for item in selected))


def plan_generation_packing(*, task, package, features, replay_input):
    if task.identity != features.task_hash or control_hash(replay_input) != features.input_hash:
        raise ValueError('retrieval_generation_packing_input_changed')
    items = {item['chunk_id']: item for item in package.package_json['chunks']}
    candidates = tuple(PathFeatureCandidate.model_validate(item) for item in replay_input['package'])
    if set(items) != {item.id for item in candidates}:
        raise ValueError('retrieval_generation_packing_source_scope_changed')
    parameters = PathEvaluationParameters.model_validate(replay_input['parameters'])
    witnesses = set()
    for facet in task.requirements:
        rows = []
        for candidate in candidates:
            opportunity = next(item.value for item in candidate.opportunities if item.facet_id == facet.id)
            quality = path_quality(candidate, parameters)
            if candidate.source_valid:
                rows.append((candidate, (opportunity.lower, opportunity.upper,
                    opportunity.lower * quality if candidate.path_observed else 0,
                    opportunity.upper * quality if candidate.path_observed else opportunity.upper)))
        for index in range(4):
            winner = max(rows, key=lambda row: (row[1][index], row[0].source_id), default=None)
            if winner is not None and winner[1][index] > 0:
                witnesses.add(winner[0].id)
    if features.scope_statuses:
        from app.services.source_use import _intersect_scope_intervals, _merge_scope_intervals
        needed = _merge_scope_intervals((part.knowledge_base_id,part.document_version_id,part.start,part.end)
            for status in features.scope_statuses for coverage in status.coverage for part in coverage.usable_intervals)
        witnesses.update(cid for cid,item in items.items() if _intersect_scope_intervals((
            (task.knowledge_base_id,item['document_version_id'],*item['char_span']),),needed))
    if len(witnesses) > 40:
        return None
    if not witnesses or any(items[cid].get('content_clipped') for cid in witnesses):
        return None
    neighbors = {cid for wid in witnesses for cid in (
        items[wid]['structure_closure'].get('previous_chunk_id'),
        items[wid]['structure_closure'].get('next_chunk_id')) if cid in items
        and not items[cid].get('content_clipped')} - witnesses
    selected = witnesses | neighbors
    if len(selected) == len(items):
        return None
    tokens = sum(rough_token_count(items[cid]['content']) for cid in selected)
    if tokens >= package.token_budget:
        return None
    return GenerationPackingPlan(task_hash=task.identity, feature_input_hash=features.input_hash,
        source_context_package_id=package.id, witness_chunk_ids=tuple(cid for cid in items if cid in witnesses),
        neighbor_chunk_ids=tuple(cid for cid in items if cid in neighbors), selection_token_budget=tokens,
        reserved_token_budget=package.token_budget - tokens, original_source_count=len(items), selected_source_count=len(selected),
        original_text_characters=sum(len(item['content']) for item in items.values()),
        selected_text_characters=sum(len(items[cid]['content']) for cid in selected))


def verify_generation_packing(before, after):
    if before.task_hash != after.task_hash or before.evaluation_protocol_hash != after.evaluation_protocol_hash:
        raise ValueError('retrieval_generation_packing_evaluation_changed')
    if {item.facet_id:item.state for item in before.scope_statuses} != {item.facet_id:item.state for item in after.scope_statuses}:
        raise ValueError('retrieval_generation_packing_source_obligation_changed')
    old = {item.facet_id: item for item in before.facets}
    new = {item.facet_id: item for item in after.facets}
    if set(old) != set(new) or after.invalid_packaged_source_count:
        raise ValueError('retrieval_generation_packing_coverage_changed')
    for facet_id in old:
        for metric in ('coverage', 'path_quality'):
            for bound in ('lower', 'upper'):
                if abs(getattr(getattr(old[facet_id], metric), bound) - getattr(getattr(new[facet_id], metric), bound)) > 1e-12:
                    raise ValueError('retrieval_generation_packing_coverage_changed')


def replay_scope_generation_packing(db, *, run, observation):
    """Replay both prepared intent and the atomically committed projection."""
    from app.models import AgentObservation, ContextPackage
    from app.retrieval_control_contracts import TaskContract, PathFeatureSummary
    from app.services.evidence_scope import replay_scope_inputs
    from sqlalchemy import select
    card=observation.observation_json
    if (observation.run_id!=run.id or observation.verdict not in {'prepared','completed'}
            or card.get('audit_hash')!=control_hash({k:v for k,v in card.items() if k!='audit_hash'})):
        raise ValueError('scope_generation_packing_audit_changed')
    before_gate=db.get(AgentObservation,card.get('before_gate_observation_id'))
    source=db.get(ContextPackage,card['source_context_package_id'])
    if (before_gate is None or before_gate.run_id!=run.id or before_gate.observation_type!='retrieval_gate'
            or source is None or source.knowledge_base_id!=run.knowledge_base_id):
        raise ValueError('scope_generation_packing_owner_changed')
    before=before_gate.observation_json
    inputs=before['feature_input']
    fixed=TaskContract.model_validate(inputs['task'])
    parameters=PathEvaluationParameters.model_validate(inputs['parameters'])
    replay_scope_inputs(db,task=fixed,parameters=parameters,package=source)
    plan=plan_scope_generation_packing(task=fixed,package=source,
        features=PathFeatureSummary.model_validate(before['features']),replay_input=inputs)
    if plan is None or plan.model_dump(mode='json')!={k:card[k] for k in ScopeGenerationPackingPlan.model_fields}:
        raise ValueError('scope_generation_packing_plan_changed')
    if observation.verdict=='prepared':
        if 'target_context_package_id' in card:
            raise ValueError('scope_generation_packing_prepared_has_target')
        return plan
    target=db.get(ContextPackage,card['target_context_package_id'])
    after_gate=db.scalar(select(AgentObservation).where(AgentObservation.run_id==run.id,
        AgentObservation.observation_type=='retrieval_gate',
        AgentObservation.observation_json['features']['input_hash'].as_string()==card['feature_input_hash_after'])
        .order_by(AgentObservation.created_at,AgentObservation.id))
    if target is None or target.id==source.id or target.knowledge_base_id!=source.knowledge_base_id or after_gate is None:
        raise ValueError('scope_generation_packing_target_missing')
    actual=target.package_json['chunks']
    old={item['chunk_id']:item for item in source.package_json['chunks']}
    if (tuple(item['chunk_id'] for item in actual)!=plan.selected_chunk_ids or any(
            any(item.get(k)!=old[item['chunk_id']].get(k) for k in
                ('content','document_version_id','char_span','raw_chunk_char_span','content_clipped')) for item in actual)
            or card.get('actual_source_count')!=len(actual)
            or card.get('actual_text_characters')!=sum(len(item['content']) for item in actual)
            or not all(card.get(k) is True for k in ('coverage_bounds_preserved','source_integrity_passed','repair_count_unchanged'))):
        raise ValueError('scope_generation_packing_source_changed')
    after=after_gate.observation_json
    if after['context_package_id']!=target.id or before['decision']['outcome']!=after['decision']['outcome']:
        raise ValueError('scope_generation_packing_gate_changed')
    verify_generation_packing(PathFeatureSummary.model_validate(before['features']),PathFeatureSummary.model_validate(after['features']))
    return plan
