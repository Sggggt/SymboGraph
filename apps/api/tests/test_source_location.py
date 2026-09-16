from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.retrieval_control_contracts import (
    Requirement,SemanticScopeItem,SemanticScopeSelection,SourceScopeRequest,control_hash,
)
from app.services.evidence_scope import StructureScopeIndex
from app.services.source_location import (
    _RangeMaximum,apply_scope_selection,build_location_request,location_output_type,location_packet,
)
from test_evidence_scope import index_fixture,obligation,request,task


def fixture(reference='résumé',printed='Executive Summary'):
    text=printed+'\nThe interface accepts a bounded request.\nImplementation\nThe queue is processed in order.'
    cut=text.index('Implementation')
    base=index_fixture(text,[('document','Unit manual',0,len(text)),('section',printed,0,cut),
        ('section','Implementation',cut,len(text))],cuts=(cut,))
    scope=SourceScopeRequest(op='intersection',children=(request('document','Unit manual'),request('section',reference)))
    fixed=task(obligation(scope),f'Use {reference} in Unit manual')
    return fixed,StructureScopeIndex(corpus=base.corpus,nodes=base.nodes,task_hash=fixed.identity)


def completed(index,fixed,request_card,choices=None):
    if choices is None:
        choices={item.id:{'status':'selected','candidate_id':next(card.id for card in item.candidates if card.title=='Executive Summary')}
                 for item in request_card.selectors}
    output=location_output_type(request_card).model_validate({'choices':choices}).model_dump(mode='json',by_alias=True)
    payload={'protocol_version':'source_location_call_v1','status':'completed','run_id':'unit-run','task_hash':fixed.identity,
        'request':request_card.model_dump(mode='json'),'result':output,'input_hash':control_hash(location_packet(fixed,request_card))}
    payload['audit_hash']=control_hash(payload)
    items=[]
    for item in request_card.selectors:
        choice=output['choices'][item.id]
        if choice['status']=='selected':
            card=next(card for card in item.candidates if card.id==choice['candidate_id'])
            items.append(SemanticScopeItem(selector_hash=item.selector_hash,node_id=card.node_id))
    selection=SemanticScopeSelection(observation_id='unit-observation',run_id='unit-run',task_hash=fixed.identity,
        ledger_hash=payload['audit_hash'],source_index_hash=index.structural_identity,items=tuple(items))
    return payload,selection


@pytest.mark.parametrize('reference',['résumé','概述','overview'])
def test_semantic_choice_preserves_original_description_and_resolves_raw_location(reference):
    fixed,index=fixture(reference)
    before=fixed.model_dump(mode='json')
    assert index.bind(fixed)[0].bindings[0].reason=='no_verified_match'
    card=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    assert len(card.selectors)==1 and card.selectors[0].selector.reference==reference
    packet=location_packet(fixed,card)
    assert packet['locator_cards_are_answer_evidence'] is False
    assert all('node_id' not in item and 'document_version_id' not in item for group in packet['location_requests'] for item in group['candidates'])
    payload,selection=completed(index,fixed,card)
    apply_scope_selection(index=index,task=fixed,selection=selection,payload=payload)
    binding=index.bind(fixed)[0].bindings[0]
    assert binding.reason=='resolved' and selection.semantic_identity_proven is False
    assert fixed.model_dump(mode='json')==before
    assert binding.fact.intervals[0].end==index.by_id['unit-n1'].char_end
    assert index.identity!=index.structural_identity


def test_numbered_labels_never_become_model_location_choices():
    text='Table 2\nvalue 3'
    base=index_fixture(text,[('table','Table 2',0,len(text))])
    fixed=task(obligation(request('table','Table 9','label')),'Use Table 9')
    index=StructureScopeIndex(corpus=base.corpus,nodes=base.nodes,task_hash=fixed.identity)
    assert build_location_request(index=index,task=fixed,query_scores=[.9]) is None


def test_unique_literal_location_does_not_call_model():
    fixed,index=fixture('Executive Summary')
    assert build_location_request(index=index,task=fixed,query_scores=[.9,.2]) is None


@pytest.mark.parametrize('status',['unresolved','ambiguous'])
def test_unresolved_model_choice_never_manufactures_a_scope(status):
    fixed,index=fixture()
    card=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    payload,selection=completed(index,fixed,card,{item.id:{'status':status} for item in card.selectors})
    apply_scope_selection(index=index,task=fixed,selection=selection,payload=payload)
    assert not selection.items and index.bind(fixed)[0].bindings[0].fact.resolution=='unresolved'


@pytest.mark.parametrize('attack',['task','index','candidate','input','choice','run'])
def test_stale_or_foreign_choice_fails_replay(attack):
    fixed,index=fixture()
    card=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    payload,selection=completed(index,fixed,card)
    if attack=='task':
        fixed=fixed.model_copy(update={'question':fixed.question+' changed'})
    elif attack=='index':
        index.structural_identity='f'*64
    elif attack=='run':
        selection=selection.model_copy(update={'run_id':'foreign'})
    else:
        payload=deepcopy(payload)
        if attack=='candidate':
            payload['request']['selectors'][0]['candidates'][0]['title']='invented label'
        elif attack=='input':
            payload['input_hash']='f'*64
        else:
            payload['result']['choices'][card.selectors[0].id]['candidate_id']='foreign'
        payload['audit_hash']=control_hash({key:value for key,value in payload.items() if key!='audit_hash'})
        selection=selection.model_copy(update={'ledger_hash':payload['audit_hash']})
    with pytest.raises(ValueError):
        apply_scope_selection(index=index,task=fixed,selection=selection,payload=payload)


def test_range_maximum_handles_nested_overlapping_chunks_and_gaps():
    sources=[SimpleNamespace(chunk_id=str(i),char_start=start,char_end=end) for i,(start,end) in enumerate([(0,10),(2,3),(5,12),(20,25)])]
    scores={'0':.2,'1':.9,'2':.4,'3':.7}
    index=_RangeMaximum(sources,scores)
    for start in range(27):
        for end in range(start+1,28):
            expected=max((scores[source.chunk_id] for source in sources if source.char_start<end and source.char_end>start),default=0.)
            assert index.score(start,end)==expected


def test_same_word_in_two_document_scopes_has_distinct_selector_identity():
    from app.services.source_location import selector_loci
    fixed,index=fixture('overview')
    first=fixed.requirements[0].source_scope
    other=obligation(SourceScopeRequest(op='intersection',children=(request('document','Other manual'),request('section','overview'))))
    fixed=fixed.model_copy(update={'question':'Compare overview in Unit manual and Other manual',
        'requirements':(fixed.requirements[0].model_copy(update={'weight':.5}),Requirement(id='f2',text='Other description',weight=.5,source_scope=other))})
    loci=selector_loci(index,fixed)
    keys=[key for key,(selector,*_) in loci.items() if selector.reference=='overview']
    assert len(keys)==2 and keys[0]!=keys[1]


def test_shared_document_scope_is_inherited_without_losing_source_roles(no_fallback_env):
    from app.services.retrieval_models import TaskPlanningOutput,compile_task_plan
    raw={'perception':{'intent':'comparison','direct_answer_kind':'none','entities':[]},
        'answer_shape':'comparison','shared_source_scope':request('document','Unit manual').model_dump(mode='json'),
        'requirements':[{'facet':'summary limit','lexical_role':'domain','source_roles':['summary']},
                        {'facet':'body limit','lexical_role':'domain','source_roles':['detail']}]}
    _,fixed,_,_=compile_task_plan(TaskPlanningOutput.model_validate(raw),question='Compare the limit in the summary and main text of Unit manual',
        knowledge_base_id='unit-kb',conversation_scope_hash='a'*64,retrieval_granularity='mid')
    assert [item.source_roles for item in fixed.requirements]==[('summary',),('detail',)]
    assert all(item.source_scope.scope.selector.reference=='Unit manual' for item in fixed.requirements)


def test_locator_navigation_comes_from_same_version_structure_and_replays():
    fixed,base=fixture()
    base.by_id['unit-n1'].parent_id='unit-n0'
    index=StructureScopeIndex(corpus=base.corpus,nodes=base.nodes,task_hash=fixed.identity)
    card=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    target=next(item for item in card.selectors[0].candidates if item.title=='Executive Summary')
    assert target.parent_titles==('Unit manual',) and target.source_order==0
    payload,selection=completed(index,fixed,card)
    index.by_id['unit-n0'].title='Changed parent label'
    with pytest.raises(ValueError,match='candidate_provenance_changed'):
        apply_scope_selection(index=index,task=fixed,selection=selection,payload=payload)


def test_earlier_locator_cards_keep_their_original_packet_shape():
    from app.services.source_location import _card
    fixed,index=fixture()
    request_card=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    original=request_card.selectors[0]
    legacy=original.model_copy(update={'candidates':tuple(_card(index,index.by_id[item.node_id],item.id,navigation=False)
        for item in original.candidates)})
    request_card=request_card.model_copy(update={'selectors':(legacy,)})
    encoded=location_packet(fixed,request_card)
    assert all('navigation_protocol' not in item and 'source_order' not in item and 'parent_titles' not in item
        for item in encoded['location_requests'][0]['candidates'])
    payload,selection=completed(index,fixed,request_card)
    apply_scope_selection(index=index,task=fixed,selection=selection,payload=payload)
    assert index.bind(fixed)[0].bindings[0].reason=='resolved'


def test_each_scope_uses_its_fixed_requirement_instead_of_the_shared_query():
    texts=[f'Region {index}\nControl description {index}.\n' for index in range(9)]
    raw=''.join(texts)
    boundaries=[sum(len(part) for part in texts[:index]) for index in range(10)]
    base=index_fixture(raw,[('section',f'Region {index}',boundaries[index],boundaries[index+1]) for index in range(9)],cuts=tuple(boundaries[1:-1]))
    fixed=task(obligation(request('section','premier réglage')),'Comparer premier réglage et dernier réglage')
    fixed=fixed.model_copy(update={'requirements':(
        fixed.requirements[0].model_copy(update={'weight':.5,'text':'Initial control'}),
        Requirement(id='f2',text='Final control',weight=.5,source_scope=obligation(request('section','dernier réglage'))))})
    index=StructureScopeIndex(corpus=base.corpus,nodes=base.nodes,task_hash=fixed.identity)
    global_scores=[0.,0.,0.,0.,1.,0.,0.,0.,0.]
    facets={'f1':[1.,0.,0.,0.,0.,0.,0.,0.,0.],'f2':[0.,0.,0.,0.,0.,0.,0.,0.,1.]}
    card=build_location_request(index=index,task=fixed,query_scores=global_scores,facet_scores=facets)
    by_ref={item.selector.reference:item for item in card.selectors}
    assert by_ref['premier réglage'].candidates[0].node_id=='unit-n0'
    assert by_ref['dernier réglage'].candidates[0].node_id=='unit-n8'
    assert card.candidate_protocol=='fixed_facet_location_candidates_v1' and card.candidate_input_hash
    packet=location_packet(fixed,card)
    assert {tuple(row['fixed_requirements']) for row in packet['location_requests']}=={('Initial control',),('Final control',)}
    opposite=[1.-value for value in global_scores]
    unchanged=build_location_request(index=index,task=fixed,query_scores=opposite,facet_scores=facets)
    assert unchanged==card
    with pytest.raises(ValueError,match='facet_score_scope_invalid'):
        build_location_request(index=index,task=fixed,query_scores=global_scores,facet_scores={'f1':facets['f1']})


def test_earlier_candidate_protocol_keeps_its_request_and_packet_identity():
    fixed,index=fixture()
    card=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    assert 'candidate_protocol' not in card.model_dump(mode='json') and 'candidate_input_hash' not in card.model_dump(mode='json')
    assert all('fixed_requirements' not in row for row in location_packet(fixed,card)['location_requests'])
