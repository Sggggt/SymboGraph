"""Bounded semantic location choices over attested structure, never graph facts."""
from __future__ import annotations

from bisect import bisect_left,bisect_right
from collections import defaultdict
import json
import math
import heapq
import re
from typing import Annotated,Literal,Union

from pydantic import Field,create_model
from sqlalchemy import select

from app.models import AgentObservation
from app.retrieval_control_contracts import (
    ControlContract,SemanticScopeItem,SemanticScopeSelection,SourceLocationCandidate,
    SourceLocationRequest,SourceLocationSelector,control_hash,
)
from app.services.evidence_scope import _NODE_KINDS,_normal,coverage_requests
from app.services.source_use import _intersect_scope_intervals,_merge_scope_intervals
from app.services.storage import raise_if_source_io_cancelled
from app.services.source_reference_alignment import PROTOCOL as ALIGNMENT_PROTOCOL, align_reference


class _RangeMaximum:
    def __init__(self,sources,scores):
        self.points=sorted({position for source in sources for position in (source.char_start,source.char_end)})
        ordered=sorted(sources,key=lambda source:(source.char_start,source.char_end,source.chunk_id))
        heap=[]
        values=[]
        cursor=0
        for ordinal,position in enumerate(self.points[:-1]):
            if ordinal%256==0:
                raise_if_source_io_cancelled()
            while cursor<len(ordered) and ordered[cursor].char_start<=position:
                source=ordered[cursor]
                heapq.heappush(heap,(-float(scores[source.chunk_id]),source.char_end,source.chunk_id))
                cursor+=1
            while heap and heap[0][1]<=position:
                heapq.heappop(heap)
            values.append(-heap[0][0] if heap else 0.)
        self.size=1
        while self.size<len(values):
            self.size*=2
        self.tree=[0.]*(2*self.size)
        self.tree[self.size:self.size+len(values)]=values
        for index in range(self.size-1,0,-1):
            self.tree[index]=max(self.tree[index*2],self.tree[index*2+1])

    def score(self,start,end):
        left=max(0,bisect_right(self.points,start)-1)+self.size
        right=min(max(0,len(self.points)-1),bisect_left(self.points,end))+self.size
        result=0.
        while left<right:
            if left%2:
                result=max(result,self.tree[left]); left+=1
            if right%2:
                right-=1; result=max(result,self.tree[right])
            left//=2; right//=2
        return result


def _tokens(value):
    return set(re.findall(r'[^\W_]+',_normal(value)))


def selector_loci(index,task):
    """Preserve conjunctive context; equal labels in different scopes are distinct."""
    roots={control_hash(request.scope.model_dump(mode='json')):request.scope
        for facet in task.requirements if facet.source_scope for request in coverage_requests(facet.source_scope)}
    result={}
    def known(node):
        if node.op=='scope':
            matches=index._matches(node.selector)
            if not matches or (len(matches)>1 and node.selector.match not in {'kind','role'}):
                return None
            return _merge_scope_intervals(span for item,_ in matches for span in index.selector_ranges(node.selector,item))
        values=[known(child) for child in node.children]
        if node.op=='union':
            return None if any(value is None for value in values) else _merge_scope_intervals(part for value in values for part in value)
        values=[value for value in values if value is not None]
        if not values:
            return None
        merged=values[0]
        for value in values[1:]:
            merged=_intersect_scope_intervals(merged,value)
        return merged
    def walk(node,root_hash,path=(),permission=None):
        if node.op=='scope':
            key=control_hash({'request':root_hash,'path':path,'selector':node.selector.model_dump(mode='json')})
            result[key]=(node.selector,root_hash,path,permission)
            return
        for ordinal,child in enumerate(node.children):
            allowed=permission
            if node.op=='intersection':
                for other_index,other in enumerate(node.children):
                    value=known(other) if other_index!=ordinal else None
                    if value is not None:
                        allowed=value if allowed is None else _intersect_scope_intervals(allowed,value)
            walk(child,root_hash,(*path,ordinal),allowed)
    for root_hash,root in sorted(roots.items()):
        walk(root,root_hash)
    return result


def _card(index,node,identifier,*,navigation=True):
    _,version,start,end=index.node_ranges[node.id]
    sources=index.by_version[version]
    offset=bisect_left(index.ends[version],start+1)
    excerpt=''
    extent=None
    if offset<len(sources):
        left=max(start,sources[offset].char_start)
        right=min(end,left+160,sources[offset].char_end)
        if left<right:
            value=index.raw_slice(version,left,right)
            if value is not None:
                excerpt,extent=value,(left,right)
    parents=[]
    cursor=index.by_id.get(node.parent_id)
    seen={node.id}
    while navigation and cursor is not None and len(parents)<3:
        if cursor.id in seen or cursor.document_version_id!=version:
            raise ValueError('source_location_parent_scope_invalid')
        seen.add(cursor.id)
        parents.append(str(cursor.title or '')[:80])
        cursor=index.by_id.get(cursor.parent_id)
    return SourceLocationCandidate(id=identifier,node_id=node.id,document_version_id=version,
        kind=_NODE_KINDS[node.node_type],document_title=sources[0].title[:160],
        title=str(node.title or '')[:160],excerpt=excerpt,excerpt_span=extent,
        navigation_protocol='structure_location_labels_v1' if navigation else None,
        source_order=start if navigation else None,parent_titles=tuple(reversed(parents)) if navigation else None)


def _scope_requirements(task):
    owners=defaultdict(list)
    for facet in task.requirements:
        if facet.source_scope is not None:
            for request in coverage_requests(facet.source_scope):
                key=control_hash(request.scope.model_dump(mode='json'))
                if facet.id not in owners[key]:
                    owners[key].append(facet.id)
    return owners


def build_location_request(*,index,task,query_scores,facet_scores=None):
    if index.task_hash!=task.identity or index.semantic_selection is not None:
        raise ValueError('source_location_request_identity_invalid')
    if all(binding.reason in {'resolved','representation_incomplete','unsupported_representation'}
           for facet in index.bind(task) for binding in facet.bindings):
        return None
    if len(query_scores)!=len(index.corpus.sources) or any(not math.isfinite(float(value)) or not 0<=float(value)<=1 for value in query_scores):
        raise ValueError('source_location_query_scores_invalid')
    source_ids=tuple(source.chunk_id for source in index.corpus.sources)
    if set(source_ids)!=set(index.corpus.by_id):
        raise ValueError('source_location_score_scope_invalid')
    score_rows={'__whole__':query_scores} if facet_scores is None else facet_scores
    if facet_scores is not None and set(facet_scores)!={facet.id for facet in task.requirements}:
        raise ValueError('source_location_facet_score_scope_invalid')
    maxima={}
    normalized={}
    for facet_id,values in score_rows.items():
        if len(values)!=len(source_ids) or any(not math.isfinite(float(value)) or not 0<=float(value)<=1 for value in values):
            raise ValueError('source_location_facet_scores_invalid')
        normalized[facet_id]=[float(value) for value in values]
        scores=dict(zip(source_ids,normalized[facet_id]))
        maxima[facet_id]={version:_RangeMaximum(sources,scores) for version,sources in index.by_version.items()}
    owners=_scope_requirements(task)
    question_tokens=_tokens(task.question)
    requests=[]
    for key,(selector,root_hash,path,permission) in selector_loci(index,task).items():
        raise_if_source_io_cancelled()
        if selector.match!='title':
            continue
        literal=index._matches(selector)
        if permission is not None:
            literal=tuple((node,ids) for node,ids in literal if _intersect_scope_intervals((index.node_ranges[node.id],),permission))
        if len({index.node_ranges[node.id] for node,_ in literal})==1:
            continue
        reference_tokens=_tokens(selector.reference)
        associated=owners[root_hash] if facet_scores is not None else ['__whole__']
        inventory=[]
        for ordinal,node in enumerate(index.nodes):
            if ordinal%256==0:
                raise_if_source_io_cancelled()
            span=index.node_ranges.get(node.id)
            if not span or _NODE_KINDS[node.node_type]!=selector.kind:
                continue
            if permission is not None and not _intersect_scope_intervals((span,),permission):
                continue
            source_title=index.by_version[span[1]][0].title
            inventory.append((node,len(reference_tokens&_tokens(str(node.title or '')[:512])),
                len(question_tokens&_tokens(source_title)),tuple(maxima[facet_id][span[1]].score(span[2],span[3]) for facet_id in associated)))
        # Independent bounded rails, with deterministic identities as tie breaks.
        lexical=sorted((r for r in inventory if r[1]),key=lambda r:(-r[1],-r[2],r[0].id))
        document=sorted((r for r in inventory if r[2]),key=lambda r:(-r[2],r[0].char_start,r[0].id))
        semantic=[sorted(inventory,key=lambda r:(-r[3][ordinal],r[0].id)) for ordinal in range(len(associated))]
        rails=[lexical,document,*semantic] if facet_scores is None else [lexical,*semantic,document]
        selected=[]
        for rank in range(4):
            for rail in rails:
                if rank<len(rail) and rail[rank][0].id not in {node.id for node in selected}:
                    selected.append(rail[rank][0])
                    if len(selected)==4:
                        break
            if len(selected)==4:
                break
        if not selected:
            continue
        identifier=f's{len(requests)}'
        requests.append(SourceLocationSelector(id=identifier,selector=selector,selector_hash=key,
            request_hash=root_hash,path=path,eligible_candidate_count=len(inventory),
            candidates=tuple(_card(index,node,f'{identifier}_c{pos}') for pos,node in enumerate(selected))))
        if len(requests)>8:
            raise ValueError('source_location_selector_capacity_exceeded')
    return SourceLocationRequest(task_hash=task.identity,source_index_hash=index.structural_identity,
        source_scope_hash=index.corpus.scope_hash,source_chunk_ids=tuple(sorted(index.corpus.by_id)),
        candidate_protocol='fixed_facet_location_candidates_v1' if facet_scores is not None else 'whole_query_location_candidates_v1',
        candidate_input_hash=control_hash({'source_ids':source_ids,'facet_scores':normalized}) if facet_scores is not None else None,
        reference_alignment_protocol=ALIGNMENT_PROTOCOL,
        selectors=tuple(requests)) if requests else None


class _NoLocation(ControlContract):
    status: Literal['unresolved','ambiguous']


def location_output_type(request):
    fields={}
    for ordinal,item in enumerate(request.selectors):
        identifiers=tuple(card.id for card in item.candidates)
        selected=create_model(f'LocationChoice{ordinal}',__base__=ControlContract,
            status=(Literal['selected'],...),candidate_id=(Literal[identifiers],...))
        fields[f'choice_{ordinal}']=(Annotated[Union[_NoLocation,selected],Field(discriminator='status')],Field(alias=item.id))
    choices=create_model('SourceLocationChoices',__base__=ControlContract,**fields)
    return create_model('SourceLocationOutput',__base__=ControlContract,
        protocol_version=(Literal['source_location_choices_v1'],'source_location_choices_v1'),choices=(choices,...))


def location_packet(task,request):
    roots={control_hash(item.scope.model_dump(mode='json')):item.scope for facet in task.requirements
        if facet.source_scope for item in coverage_requests(facet.source_scope)}
    groups={key:f'g{index}' for index,key in enumerate(sorted({item.request_hash for item in request.selectors}))}
    packet={'current_user':{'question':task.question},
        'scope_groups':[{'id':identifier,'scope':roots[key].model_dump(mode='json')} for key,identifier in groups.items()],
        'location_requests':[
        {'id':item.id,'reference':item.selector.reference,'kind':item.selector.kind,
         'scope_group':groups[item.request_hash],
         'candidates':[card.model_dump(mode='json',exclude={'node_id','document_version_id','excerpt_span'}) for card in item.candidates],
         'candidate_inventory_complete':item.eligible_candidate_count==len(item.candidates)} for item in request.selectors],
        'locator_cards_are_answer_evidence':False,'original_scope_may_not_be_relaxed':True}
    if request.candidate_protocol!='whole_query_location_candidates_v1':
        owners=_scope_requirements(task)
        by_id={facet.id:facet.text for facet in task.requirements}
        for row,item in zip(packet['location_requests'],request.selectors):
            row['fixed_requirements']=[by_id[facet_id] for facet_id in owners[item.request_hash]]
    if request.reference_alignment_protocol == ALIGNMENT_PROTOCOL:
        # This is optional locator information. Preserve every original card and
        # the existing packet limit if the complete diagnostic would not fit.
        diagnostics = []
        for item in request.selectors:
            for card in item.candidates:
                raise_if_source_io_cancelled()
                for field in ('document_title', 'title', 'excerpt'):
                    result = align_reference(item.selector.reference, getattr(card, field))
                    if result.status != 'no_alignment':
                        diagnostics.append({'request_id': item.id, 'candidate_id': card.id,
                                            'field': field, **result.model_dump(mode='json', exclude_none=True)})
        projection = {'protocol_version': ALIGNMENT_PROTOCOL, 'semantic_identity_proven': False,
                      'omitted_for_capacity': False, 'correspondences': diagnostics}
        if len(json.dumps({**packet, 'reference_alignment': projection}, ensure_ascii=False)) > 48000:
            projection = {**projection, 'omitted_for_capacity': True, 'correspondences': []}
        if len(json.dumps({**packet, 'reference_alignment': projection}, ensure_ascii=False)) <= 48000:
            packet['reference_alignment'] = projection
    if len(json.dumps(packet,ensure_ascii=False))>48000:
        raise ValueError('source_location_packet_capacity_exceeded')
    return packet


def complete_location_call(db,*,row,task,index,output,model_audit):
    request=SourceLocationRequest.model_validate(row.observation_json['request'])
    result=location_output_type(request).model_validate(output).model_dump(mode='json',by_alias=True)
    if model_audit['input_hash']!=row.observation_json['input_hash']:
        raise ValueError('source_location_model_input_changed')
    payload={k:v for k,v in row.observation_json.items() if k!='audit_hash'}
    payload.update(status='completed',result=result,model_audit=model_audit)
    payload['audit_hash']=control_hash(payload)
    selection=selection_from_record(row.id,row.run_id,payload)
    apply_scope_selection(index=index,task=task,selection=selection,payload=payload)
    row.verdict,row.observation_json='completed',payload
    db.commit()
    return selection


def selection_from_record(observation_id,run_id,payload):
    request=SourceLocationRequest.model_validate(payload['request'])
    result=location_output_type(request).model_validate(payload['result']).model_dump(mode='json',by_alias=True)
    items=[]
    for item in request.selectors:
        choice=result['choices'][item.id]
        if choice['status']=='selected':
            candidate=next(card for card in item.candidates if card.id==choice['candidate_id'])
            items.append(SemanticScopeItem(selector_hash=item.selector_hash,node_id=candidate.node_id))
    return SemanticScopeSelection(observation_id=observation_id,run_id=run_id,ledger_hash=payload['audit_hash'],
        task_hash=request.task_hash,source_index_hash=request.source_index_hash,items=tuple(items))


def apply_scope_selection(*,index,task,selection,payload):
    if index.semantic_selection is not None:
        if index.semantic_selection!=selection:
            raise ValueError('source_location_selection_already_frozen')
    if (selection.task_hash!=task.identity or selection.source_index_hash!=index.structural_identity
        or payload.get('protocol_version')!='source_location_call_v1' or payload.get('status')!='completed'
        or payload.get('run_id')!=selection.run_id
        or payload.get('audit_hash')!=selection.ledger_hash
        or payload['audit_hash']!=control_hash({k:v for k,v in payload.items() if k!='audit_hash'})):
        raise ValueError('source_location_ledger_identity_changed')
    request=SourceLocationRequest.model_validate(payload['request'])
    loci=selector_loci(index,task)
    if request.task_hash!=task.identity or request.source_index_hash!=index.structural_identity:
        raise ValueError('source_location_request_identity_changed')
    if request.source_scope_hash!=index.corpus.scope_hash or set(request.source_chunk_ids)!=set(index.corpus.by_id):
        raise ValueError('source_location_corpus_identity_changed')
    if payload.get('input_hash')!=control_hash(location_packet(task,request)):
        raise ValueError('source_location_input_changed')
    output=location_output_type(request).model_validate(payload['result']).model_dump(mode='json',by_alias=True)
    items=[]
    for item in request.selectors:
        locus=loci.get(item.selector_hash)
        if not locus or locus[:3]!=(item.selector,item.request_hash,item.path) or item.selector.match!='title':
            raise ValueError('source_location_user_declaration_changed')
        literal=index._matches(item.selector)
        permission=locus[3]
        if permission is not None:
            literal=tuple((node,ids) for node,ids in literal if _intersect_scope_intervals((index.node_ranges[node.id],),permission))
        if len({index.node_ranges[node.id] for node,_ in literal})==1:
            raise ValueError('source_location_cannot_override_unique_literal')
        for card in item.candidates:
            node=index.by_id.get(card.node_id)
            if (node is None or node.id not in index.node_ranges or _NODE_KINDS[node.node_type]!=item.selector.kind
                or (permission is not None and not _intersect_scope_intervals((index.node_ranges[node.id],),permission))
                or _card(index,node,card.id,navigation=card.navigation_protocol is not None)!=card):
                raise ValueError('source_location_candidate_provenance_changed')
        choice=output['choices'][item.id]
        if choice['status']=='selected':
            card=next(card for card in item.candidates if card.id==choice['candidate_id'])
            items.append(SemanticScopeItem(selector_hash=item.selector_hash,node_id=card.node_id))
    if tuple(items)!=selection.items:
        raise ValueError('source_location_choice_projection_changed')
    index.semantic_selection=selection
    index.semantic_matches={item.selector_hash:item.node_id for item in items}
    index.identity=control_hash({'structural_identity':index.structural_identity,'selection':selection.model_dump(mode='json')})


def replay_scope_selection(db,*,task,index,selection,for_update=False):
    statement=select(AgentObservation).where(AgentObservation.id==selection.observation_id)
    if for_update:
        statement=statement.with_for_update().execution_options(populate_existing=True)
    row=db.scalar(statement)
    if row is None or row.run_id!=selection.run_id or row.observation_type!='retrieval_scope_resolution' or row.verdict!='completed':
        raise ValueError('source_location_ledger_missing')
    apply_scope_selection(index=index,task=task,selection=selection,payload=row.observation_json)
