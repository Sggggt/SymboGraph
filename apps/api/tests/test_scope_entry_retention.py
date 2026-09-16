from types import SimpleNamespace

import pytest

from app.retrieval_control_contracts import SourceScopeObligation,SourceScopeRequest,control_hash
from app.services.evidence_scope import scope_target_plan
from app.services.retrieval_packing import required_scope_retention
from test_evidence_scope import index_fixture,obligation,request,task
from test_retrieval_path_features import candidate


def fixture(mode='overlap'):
    header='Overview\n'
    body='The interface queue has a maximum waiting time of 47 seconds.\n'
    raw=header+body
    index=index_fixture(raw,[('section','Overview',0,len(raw))],cuts=(len(header),))
    fixed=task(obligation(request('section','Overview'),mode),'Use Overview')
    items=[{'chunk_id':source.chunk_id,'document_version_id':source.document_version_id,
        'content':source.text,'char_span':[source.char_start,source.char_end]} for source in index.corpus.sources]
    package=SimpleNamespace(id='unit-package',package_json={'chunks':items},token_budget=1000)
    replay={'package':[candidate('unit-c0',value=.2).model_dump(mode='json'),
        candidate('unit-c1',value=.8,distance=10).model_dump(mode='json')]}
    features=SimpleNamespace(task_hash=fixed.identity,input_hash=control_hash(replay),scope_statuses=(True,))
    return fixed,index,package,replay,features


def test_overlap_entry_uses_fixed_requirement_within_scope_and_budget():
    fixed,index,_,_,_=fixture()
    cheapest,_=scope_target_plan(index=index,task=fixed,token_budget=1000,target_limit=2)
    assert cheapest==('unit-c0',)
    targets,audit=scope_target_plan(index=index,task=fixed,token_budget=1000,target_limit=2,
        affinities={'f1':{'unit-c0':.2,'unit-c1':.8}})
    assert targets==('unit-c1',) and audit['entry_selection_protocol']=='scope_affinity_entry_v1'
    affordable,_=scope_target_plan(index=index,task=fixed,token_budget=3,target_limit=2,
        affinities={'f1':{'unit-c0':.2,'unit-c1':.8}})
    assert affordable==('unit-c0',)
    with pytest.raises(ValueError,match='source_mismatch'):
        scope_target_plan(index=index,task=fixed,token_budget=1000,target_limit=2,affinities={'f1':{'foreign':1.}})


def test_located_scope_survives_weak_path_score():
    fixed,index,package,replay,features=fixture()
    ids,audit=required_scope_retention(task=fixed,package=package,features=features,replay_input=replay,scope_index=index)
    assert ids==('unit-c1',) and audit['path_scores_unchanged'] is True
    assert audit['model_call_count']==0


def test_complete_scope_keeps_all_needed_existing_intervals():
    fixed,index,package,replay,features=fixture('complete')
    ids,_=required_scope_retention(task=fixed,package=package,features=features,replay_input=replay,scope_index=index)
    assert set(ids)=={'unit-c0','unit-c1'}


def test_any_keeps_only_one_satisfied_alternative():
    fixed,index,package,replay,features=fixture()
    alternative=SourceScopeObligation(op='any',children=(fixed.requirements[0].source_scope,
        obligation(request('section','Absent section'))))
    fixed=fixed.model_copy(update={'question':'Use Overview or Absent section',
        'requirements':(fixed.requirements[0].model_copy(update={'source_scope':alternative}),)})
    features.task_hash=fixed.identity
    ids,_=required_scope_retention(task=fixed,package=package,features=features,replay_input=replay,scope_index=index)
    assert ids==('unit-c1',)


def test_retention_rejects_input_change():
    fixed,index,package,replay,features=fixture()
    with pytest.raises(ValueError,match='identity_changed'):
        required_scope_retention(task=fixed,package=package,features=features,replay_input={**replay,'changed':True},scope_index=index)


def test_resolved_document_scope_intersects_original_filters_without_new_inference():
    from app.services.evidence_scope import resolved_scope_filters
    from app.schemas import SearchFilters
    fixed,index,_,_,_=fixture()
    original=SearchFilters(document_ids=['unit-doc','other'],tags=['allowed'],page_range=(1,5))
    effective,audit=resolved_scope_filters(task=fixed,index=index,filters=original)
    assert effective.document_ids==['unit-doc'] and effective.tags==original.tags and effective.page_range==original.page_range
    assert original.document_ids==['unit-doc','other'] and audit['model_call_count']==0
    unscoped=fixed.model_copy(update={'requirements':(fixed.requirements[0].model_copy(update={'source_scope':None}),)})
    unchanged,audit=resolved_scope_filters(task=unscoped,index=index,filters=original)
    assert unchanged==original and audit is None
    unknown=fixed.model_copy(update={'question':'Use Absent','requirements':(
        fixed.requirements[0].model_copy(update={'source_scope':obligation(request('section','Absent'))}),)})
    unchanged,audit=resolved_scope_filters(task=unknown,index=index,filters=original)
    assert unchanged==original and audit is None
