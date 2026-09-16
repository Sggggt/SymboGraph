"""Source identity and source-family intent use independent synthetic examples."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.services.retrieval_models import (TaskPlanningOutput,TaskPlanningOutputV2,RetrievalModels,
    compile_task_plan,source_reference_role_audit,replay_source_reference_roles)
from app.retrieval_control_contracts import control_hash
from app.services.reflection_models import AnswerReviewModelError
from test_evidence_scope import request,obligation


def payload(references=(),scope=None):
    return {'perception':{'intent':'analysis','direct_answer_kind':'none','entities':['Orion']},
            'requirements':[{'facet':'Orion control architecture','lexical_role':'domain'}],
            'answer_shape':'grounded_answer','source_references':list(references),'shared_source_scope':scope}


def compile_plan(raw,question):
    plan=TaskPlanningOutputV2.model_validate(raw)
    _,task,_,_=compile_task_plan(plan,question=question,knowledge_base_id='unit-test-kb',
        conversation_scope_hash='a'*64,retrieval_granularity='mid')
    audit=source_reference_role_audit(plan,question)
    assert replay_source_reference_roles(audit,task)
    return plan,task,audit


def test_family_context_cannot_become_a_single_document_restriction():
    ref={'reference':'Orion project reports','role':'source_family'}
    raw=payload((ref,))
    question='Explain the architecture described in Orion project reports.'
    _,task,audit=compile_plan(raw,question)
    assert task.question==question and task.requirements[0].source_scope is None
    assert audit['references']==[ref] and audit['classification_is_evidence'] is False
    assert audit['additional_model_calls']==0
    raw['shared_source_scope']=request('document',ref['reference']).model_dump(mode='json')
    with pytest.raises(ValidationError,match='reference_role_conflict'):
        TaskPlanningOutputV2.model_validate(raw)


def test_named_document_must_be_used_without_silently_dropping_its_limit():
    name='Orion Safety Report'
    ref={'reference':name,'role':'named_document'}
    raw=payload((ref,),request('document',name).model_dump(mode='json'))
    _,task,_=compile_plan(raw,'According to Orion Safety Report, explain the architecture.')
    assert task.requirements[0].source_scope.scope.selector.reference==name
    raw['shared_source_scope']=None
    with pytest.raises(ValidationError,match='reference_role_conflict'):
        TaskPlanningOutputV2.model_validate(raw)


def test_mixed_family_and_named_local_scope_remain_separate():
    family='Orion project reports'; name='Delta Operations Manual'
    raw=payload(({'reference':family,'role':'source_family'},{'reference':name,'role':'named_document'}))
    raw['requirements'].append({'facet':'Delta operating constraints','lexical_role':'constraint',
        'source_scope':obligation(request('document',name)).model_dump(mode='json')})
    _,task,_=compile_plan(raw,f'Compare {family} with {name}.')
    assert task.requirements[0].source_scope is None
    assert task.requirements[1].source_scope.scope.selector.reference==name


@pytest.mark.parametrize('mutation',['missing','duplicate','role_conflict','invented'])
def test_reference_role_rejections(mutation):
    name='Delta Operations Manual'
    raw=payload(({'reference':name,'role':'named_document'},),request('document',name).model_dump(mode='json'))
    question=f'Use {name}.'
    if mutation=='missing': raw.pop('source_references')
    elif mutation=='duplicate': raw['source_references']*=2
    elif mutation=='role_conflict': raw['source_references'][0]['role']='source_family'
    elif mutation=='invented': question='Explain Orion architecture.'
    with pytest.raises((ValueError,ValidationError)):
        compile_plan(raw,question)


def test_role_audit_replays_against_frozen_question_and_scope():
    name='Delta Operations Manual'
    _,task,audit=compile_plan(payload(({'reference':name,'role':'named_document'},),
        request('document',name).model_dump(mode='json')),f'Use {name}.')
    changed=deepcopy(audit)
    changed['references'][0]['role']='source_family'
    with pytest.raises(ValueError,match='role_audit_invalid'): replay_source_reference_roles(changed,task)
    changed['audit_hash']=control_hash({k:v for k,v in changed.items() if k!='audit_hash'})
    with pytest.raises(ValueError,match='role_audit_invalid'): replay_source_reference_roles(changed,task)
    changed_task=task.model_copy(update={'source_reference_roles_hash':changed['audit_hash']})
    with pytest.raises(ValueError,match='role_conflict'): replay_source_reference_roles(changed,changed_task)
    with pytest.raises(ValueError,match='role_audit_invalid'):
        replay_source_reference_roles(audit,task.model_copy(update={'question':'Changed question'}))


def test_legacy_planning_remains_readable_but_is_not_the_live_model_schema():
    raw=payload();raw.pop('source_references')
    legacy=TaskPlanningOutput.model_validate(raw)
    assert legacy.protocol_version=='retrieval_task_planning_v1'
    assert source_reference_role_audit(legacy,'Explain Orion architecture.') is None
    _,old_task,_,_=compile_task_plan(legacy,question='Explain Orion architecture.',knowledge_base_id='unit-test-kb',
        conversation_scope_hash='a'*64,retrieval_granularity='mid')
    assert 'source_reference_roles_hash' not in old_task.model_dump(mode='json')
    with pytest.raises(ValidationError): TaskPlanningOutputV2.model_validate(legacy.model_dump(mode='json'))


@pytest.mark.asyncio
@pytest.mark.parametrize('direct',[False,True])
async def test_live_schema_uses_the_same_single_call_for_source_roles_and_intent(no_fallback_env,direct):
    calls=[]
    raw=payload(({'reference':'Orion reports','role':'source_family'},))
    if direct:
        raw.update(perception={'intent':'direct_answer','direct_answer_kind':'capabilities','entities':[]},
                   requirements=[],source_references=[])
    class Provider:
        api_protocol,model='anthropic','unit-test-model'
        async def classify_json_bounded(self,system_prompt,user_prompt,fallback,*,max_tokens):
            calls.append(1)
            assert 'RETRIEVAL TASK PLANNING V2' in system_prompt
            return raw
        def provider_call_audit(self): return {}
    question='Describe your capabilities.' if direct else 'Summarize Orion reports.'
    plan,audit=await RetrievalModels(Provider).plan(question=question,history_summary='',timeout_seconds=1,max_tokens=1024)
    assert len(calls)==audit['model_call_count']==1
    assert isinstance(plan,TaskPlanningOutputV2)
    assert audit['protocol_version']=='retrieval_task_planning_v2'


@pytest.mark.asyncio
async def test_live_planning_cannot_fall_back_to_unclassified_legacy_output(no_fallback_env):
    class Provider:
        api_protocol,model='anthropic','unit-test-model'
        async def classify_json_bounded(self,*args,**kwargs):
            raw=payload();raw.pop('source_references');return raw
        def provider_call_audit(self):return {}
    with pytest.raises(AnswerReviewModelError) as caught:
        await RetrievalModels(Provider).plan(question='Explain Orion reports.',history_summary='',timeout_seconds=1,max_tokens=1024)
    assert caught.value.code=='schema_invalid' and caught.value.model_call_count==1


def test_new_task_requires_its_role_audit_even_if_planning_metadata_is_removed():
    from app.services.retrieval_models import replay_run_source_reference_roles
    _,task,audit=compile_plan(payload(({'reference':'Orion reports','role':'source_family'},)),
        'Summarize Orion reports.')
    assert task.source_reference_roles_hash==audit['audit_hash']
    with pytest.raises(ValueError,match='role_audit_invalid'):
        replay_run_source_reference_roles({},task)
    assert replay_run_source_reference_roles({'retrieval_planning_audit':{'source_reference_roles':audit}},task)


def test_whitespace_is_not_a_source_expression():
    with pytest.raises(ValidationError):
        TaskPlanningOutputV2.model_validate(payload(({'reference':' ','role':'source_family'},)))


@pytest.mark.asyncio
async def test_user_referent_is_preserved_without_local_deletion_of_factual_facets(no_fallback_env):
    import json
    question='Mercury (the programming language): compare its documented concurrency and scheduling mechanisms.'
    raw=payload()
    raw['perception']['entities']=['Mercury','programming language']
    raw['requirements']=[{'facet':'Mercury concurrency mechanisms','lexical_role':'domain'},
        {'facet':'Mercury scheduling mechanisms','lexical_role':'domain'}]
    packets=[]
    class Provider:
        api_protocol,model='anthropic','unit-test-model'
        async def classify_json_bounded(self,system_prompt,user_prompt,fallback,*,max_tokens):
            packets.append(json.loads(user_prompt))
            assert 'do not add a separate evidence requirement to prove the supplied disambiguation' in system_prompt
            assert "This does not make the user's assertions verified facts" in system_prompt
            assert 'keep those as evidence requirements' in system_prompt
            return raw
        def provider_call_audit(self):return {}
    plan,audit=await RetrievalModels(Provider).plan(question=question,history_summary='',timeout_seconds=1,max_tokens=2048)
    _,task,_,_=compile_task_plan(plan,question=question,knowledge_base_id='unit-test-kb',
        conversation_scope_hash='a'*64,retrieval_granularity='mid')
    assert task.question==question and len(task.requirements)==2
    assert packets[0]['current_user']['question']==question and audit['model_call_count']==1

