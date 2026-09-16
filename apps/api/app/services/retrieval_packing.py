"""Plan a bounded repair of observed packing loss without another model call."""
from app.retrieval_control_contracts import ControlContract, PathFeatureCandidate, PathEvaluationParameters, control_hash
from app.services.retrieval_path_features import path_quality
from pydantic import Field
from typing import Literal


def required_scope_retention(*,task,package,features,replay_input,scope_index):
    """Keep already satisfied source duties even when their path score is weak."""
    from app.retrieval_control_contracts import EvidenceInterval,EvidenceScopeExpression,EvidenceIntervalCandidate
    from app.services.evidence_scope import package_scope_intervals
    from app.services.source_use import evaluate_evidence_scope,_intersect_scope_intervals
    from app.services.context_packing import plan_interval_scope_completion
    from app.services.chunking import rough_token_count
    if task.identity!=features.task_hash or control_hash(replay_input)!=features.input_hash:
        raise ValueError('scope_retention_input_identity_changed')
    if scope_index is None or not features.scope_statuses:
        return (),None
    items={item['chunk_id']:item for item in package.package_json['chunks']}
    packed=package_scope_intervals(task,package)
    costs={cid:rough_token_count(item['content']) for cid,item in items.items()}
    candidates={raw['id']:PathFeatureCandidate.model_validate(raw) for raw in replay_input['package']}
    bound={item.facet_id:item for item in scope_index.bind(task)}
    preserved=set()
    def cost(ids):
        return sum(costs[cid] for cid in ids)
    for facet in task.requirements:
        if facet.source_scope is None:
            continue
        scopes={item.request_hash:item for item in bound[facet.id].bindings}
        def choose(obligation):
            if obligation.op!='coverage':
                branches=[choose(child) for child in obligation.children]
                if obligation.op=='all':
                    return set().union(*branches) if all(branch is not None for branch in branches) else None
                feasible=[branch for branch in branches if branch is not None]
                return min(feasible,key=lambda ids:(cost(ids),len(ids),tuple(sorted(ids)))) if feasible else None
            binding=scopes[control_hash(obligation.scope.model_dump(mode='json'))]
            coverage=evaluate_evidence_scope(knowledge_base_id=task.knowledge_base_id,
                expression=EvidenceScopeExpression(op='scope',scope_id=binding.fact.id),scopes=(binding.fact,),
                packed=packed,mode=obligation.mode)
            if coverage.state!='satisfied':
                return None
            ranges=[(part.knowledge_base_id,part.document_version_id,part.start,part.end) for part in coverage.usable_intervals]
            eligible=[cid for cid,item in items.items() if candidates[cid].source_valid and _intersect_scope_intervals((
                (task.knowledge_base_id,item['document_version_id'],*item['char_span']),),ranges)]
            if obligation.mode=='overlap':
                def key(cid):
                    value=next(item.value.lower for item in candidates[cid].opportunities if item.facet_id==facet.id)
                    return (-value,costs[cid],cid)
                return {min(eligible,key=key)} if eligible else None
            empty=evaluate_evidence_scope(knowledge_base_id=task.knowledge_base_id,
                expression=EvidenceScopeExpression(op='scope',scope_id=binding.fact.id),scopes=(binding.fact,),packed=(),mode='complete')
            intervals=tuple(EvidenceIntervalCandidate(id=cid,
                interval=EvidenceInterval(knowledge_base_id=task.knowledge_base_id,document_version_id=items[cid]['document_version_id'],
                    start=items[cid]['char_span'][0],end=items[cid]['char_span'][1]),cost=costs[cid],witness_ids=(binding.request_hash,)) for cid in eligible)
            plan=plan_interval_scope_completion(knowledge_base_id=task.knowledge_base_id,coverage=empty,candidates=intervals,budget=package.token_budget)
            return set(plan.selected_ids) if plan.status in {'ready','already_covered'} else None
        chosen=choose(facet.source_scope)
        if chosen is not None:
            preserved.update(chosen)
    record={'protocol_version':'required_scope_retention_v1','task_hash':task.identity,'context_package_id':package.id,
        'feature_input_hash':features.input_hash,'source_identity_hash':scope_index.identity,
        'preserve_chunk_ids':sorted(preserved),'token_cost':cost(preserved),'path_scores_unchanged':True,'model_call_count':0}
    record['audit_hash']=control_hash(record)
    return tuple(sorted(preserved)),record


class PackingRepairPlan(ControlContract):
    protocol_version: Literal['retrieval_packing_repair_v1', 'scope_interval_repacking_v1'] = 'retrieval_packing_repair_v1'
    task_hash: str
    feature_hash: str
    context_package_id: str
    input_signature: str
    target_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    preserve_chunk_ids: tuple[str, ...] = Field(max_length=256)
    focus: tuple[str, ...] = Field(max_length=2)
    model_call_count: Literal[0] = 0
    priority_chunk_ids: tuple[str, ...] = Field(default=(), max_length=256)
    scope_completion_hash: str | None = None


def plan_scope_packing_repair(*, task, package, trace, features, scope_index):
    """Restore already authorized graph hits using the common interval solver."""
    if (scope_index is None or not features.scope_statuses
            or all(item.state == 'satisfied' for item in features.scope_statuses)):
        return None
    from app.services.evidence_scope import scope_target_plan
    from app.services.chunking import rough_token_count
    from app.services.source_use import _merge_scope_intervals, _intersect_scope_intervals
    items = {item['chunk_id']: item for item in package.package_json['chunks']}
    if not items:
        return None
    allowed = set(trace.result_chunk_ids_json or ()) | set(items)
    targets, audit = scope_target_plan(index=scope_index,task=task,token_budget=package.token_budget,
        target_limit=256,candidate_ids=allowed)
    if not targets or all(cid in items and not items[cid].get('content_clipped') for cid in targets):
        return None
    satisfied = {item.facet_id for item in features.scope_statuses if item.state == 'satisfied'}
    needed = _merge_scope_intervals((part.knowledge_base_id,part.document_version_id,part.start,part.end)
        for status in features.scope_statuses if status.facet_id in satisfied
        for coverage in status.coverage for part in coverage.usable_intervals)
    preserve = {cid for cid,item in items.items() if _intersect_scope_intervals((
        (task.knowledge_base_id,item['document_version_id'],*item['char_span']),),needed)}
    preserve.update(facet.best_source_id.split(':',1)[0] for facet in features.facets if facet.best_source_id)
    preserve &= set(items)
    total = sum(rough_token_count(scope_index.corpus.by_id[cid].text) for cid in set(targets)|preserve)
    if total > package.token_budget:
        return None
    signature = control_hash({'task':task.identity,'package':package.id,'scope_plan':audit['audit_hash'],
        'targets':targets,'preserve':sorted(preserve)})
    return PackingRepairPlan(protocol_version='scope_interval_repacking_v1',task_hash=task.identity,
        feature_hash=control_hash(features.model_dump(mode='json')),context_package_id=package.id,input_signature=signature,
        target_chunk_ids=(next(iter(items)),),preserve_chunk_ids=tuple(sorted(preserve)),focus=(),
        priority_chunk_ids=targets,scope_completion_hash=audit['audit_hash'])


def plan_packing_repair(*, task, package, features, replay_input, thresholds):
    if task.identity != features.task_hash or control_hash(replay_input) != features.input_hash:
        raise ValueError('packing_repair_input_identity_changed')
    missing = {item.facet_id for item in features.facets if item.coverage.lower < thresholds.coverage
               or item.path_quality.lower < thresholds.path_quality}
    if not missing:
        return None
    items = {item['chunk_id']: item for item in package.package_json['chunks']}
    parameters = PathEvaluationParameters.model_validate(replay_input['parameters'])
    choices = []
    for raw in replay_input['package']:
        candidate = PathFeatureCandidate.model_validate(raw)
        cid = candidate.id
        if (cid not in items or not items[cid].get('content_clipped')
            or not candidate.source_valid or not candidate.path_observed):
            continue
        potential = max((value.value.upper * path_quality(candidate, parameters)
            for value in candidate.opportunities if value.facet_id in missing
            and value.value.upper >= thresholds.coverage), default=0)
        if potential >= thresholds.path_quality:
            choices.append((potential, cid))
    targets = tuple(cid for _, cid in sorted(choices, key=lambda item: (-item[0], item[1]))[:2])
    if not targets:
        return None
    preserve = tuple(sorted({item.best_source_id.split(':', 1)[0] for item in features.facets
        if item.best_source_id and item.facet_id not in missing}))
    signature = control_hash({'task_hash': task.identity, 'targets': [
        {'chunk_id': cid, 'char_span': items[cid].get('char_span'),
         'content_hash': control_hash(items[cid]['content'])} for cid in targets], 'preserve': preserve})
    return PackingRepairPlan(task_hash=task.identity, feature_hash=control_hash(features.model_dump(mode='json')),
        context_package_id=package.id, input_signature=signature, target_chunk_ids=targets,
        preserve_chunk_ids=preserve, focus=tuple(facet.text for facet in task.requirements if facet.id in missing)[:2])
