"""Keep all declared-domain facts, including weak-scoring complementary facts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.retrieval_control_contracts import EvidenceInterval,PathFeatureCandidate,SourceScopeObligation
from app.services.generation_packing import plan_scope_generation_packing,verify_generation_packing
from app.services.retrieval_path_features import compute_path_features
from test_evidence_scope import index_fixture,task,request,obligation
from test_retrieval_path_features import parameters,strategy


def fixture(kind='section',node_type='section',op=None,incomplete=False,scoped=True):
    segments=('Area A\nCooling has a duration of 18 minutes.\n','Heating takes 27 minutes.\n','Area B\nAn unrelated policy concerns logging.\n')
    raw=''.join(segments)
    first=len(segments[0]); boundary=first+len(segments[1])
    index=index_fixture(raw,[(node_type,'Area A',0,boundary),('section','Area B',boundary,len(raw))],
        cuts=(first,boundary),incomplete=incomplete)
    scope=obligation(request(kind,'Area A')) if scoped else None
    if op: scope=SourceScopeObligation(op=op,children=(scope,obligation(request('section','Area B'))))
    fixed=task(scope,'Use Area A and Area B only as requested.')
    lexical=strategy(fixed,('duration',))
    intervals=tuple(EvidenceInterval(knowledge_base_id=fixed.knowledge_base_id,document_version_id=source.document_version_id,
        start=source.char_start,end=source.char_end) for source in index.corpus.sources)
    params=parameters().model_copy(update={'protocol_version':'canonical_task_path_quality_v4',
        'scope_inputs':index.bind(fixed),'packed_scope_intervals':intervals})
    candidates=tuple(PathFeatureCandidate(id=source.chunk_id,source_id=f'{source.chunk_id}:{source.char_start}:{source.char_end}',
        topic_group='unit-doc',source_valid=True,canonical_entry_distance=.1,routing_cost=0,
        opportunities=({'facet_id':'f1','value':{'lower':value,'upper':value}},))
        for source,value in zip(index.corpus.sources,(.2,.8,0)))
    features=compute_path_features(task=fixed,strategy=lexical,panels=(),packaged_candidates=candidates,parameters=params)
    inputs={'task':fixed.model_dump(mode='json'),'strategy':lexical.model_dump(mode='json'),'panels':[],
        'package':[c.model_dump(mode='json') for c in candidates],'parameters':params.model_dump(mode='json')}
    package=SimpleNamespace(id='unit-package',token_budget=2000,package_json={'chunks':[
        {'chunk_id':source.chunk_id,'document_version_id':source.document_version_id,'char_span':[source.char_start,source.char_end],
         'content':source.text,'content_clipped':False,'role':'hit'} for source in index.corpus.sources]})
    return dict(task=fixed,package=package,features=features,replay_input=inputs),params,lexical,candidates


@pytest.mark.parametrize('kind,node_type',[('section','section'),('table','table'),('formula','formula'),('code','code_block'),('text','paragraph')])
def test_all_overlapping_complete_sources_survive_and_all_four_bounds_remain(kind,node_type):
    args,params,lexical,candidates=fixture(kind,node_type)
    original=deepcopy(args['package'].package_json)
    plan=plan_scope_generation_packing(**args)
    assert plan.selected_chunk_ids==('unit-c0','unit-c1')
    assert plan.model_call_count==0 and plan.source_text_modified is False
    assert plan.selection_token_budget+plan.reserved_token_budget==args['package'].token_budget
    kept=candidates[:2]
    projected=params.model_copy(update={'packed_scope_intervals':params.packed_scope_intervals[:2]})
    after=compute_path_features(task=args['task'],strategy=lexical,panels=(),packaged_candidates=kept,parameters=projected)
    verify_generation_packing(args['features'],after)
    assert args['package'].package_json==original
    assert candidates[0].opportunities[0].value.lower < candidates[1].opportunities[0].value.lower


@pytest.mark.parametrize('op',['all','any'])
def test_all_and_any_keep_every_located_alternative_instead_of_choosing_a_shorter_one(op):
    args,*_=fixture(op=op)
    # Both declared alternatives cover the whole current package: no reduction.
    assert plan_scope_generation_packing(**args) is None


@pytest.mark.parametrize('reason',['unscoped','unknown','clipped','retained','no_reduction','over_budget'])
def test_unsupported_projection_leaves_current_package_intact(reason):
    args,*_=fixture(scoped=reason!='unscoped',incomplete=reason=='unknown')
    if reason=='clipped': args['package'].package_json['chunks'][0]['content_clipped']=True
    elif reason=='retained': args['package'].package_json['chunks'][0]['role']='preserved_source'
    elif reason=='no_reduction': args,*_=fixture(op='all')
    elif reason=='over_budget': args['package'].token_budget=1
    before=deepcopy(args['package'].package_json)
    assert plan_scope_generation_packing(**args) is None
    assert args['package'].package_json==before


@pytest.mark.parametrize('attack',['version','span','duplicate','input'])
def test_changed_source_identity_fails_before_selection(attack):
    args,*_=fixture()
    if attack=='version': args['package'].package_json['chunks'][0]['document_version_id']='foreign-version'
    elif attack=='span': args['package'].package_json['chunks'][0]['char_span'][0]+=1
    elif attack=='duplicate': args['package'].package_json['chunks'].append(args['package'].package_json['chunks'][0])
    else: args['replay_input']['panels']=[{}]
    with pytest.raises(ValueError,match='changed'): plan_scope_generation_packing(**args)


def test_cancellation_is_checked_before_planning(monkeypatch):
    import asyncio
    from app.services import storage
    def cancelled(): raise asyncio.CancelledError()
    monkeypatch.setattr(storage,'raise_if_source_io_cancelled',cancelled)
    args,*_=fixture()
    with pytest.raises(asyncio.CancelledError): plan_scope_generation_packing(**args)
