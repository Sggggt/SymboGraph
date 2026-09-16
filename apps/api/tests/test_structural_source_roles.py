import pytest

from app.retrieval_control_contracts import SourceScopeSelector,SourceScopeRequest,control_hash
from app.services.evidence_scope import StructureScopeIndex
from app.services.source_location import build_location_request
from test_evidence_scope import index_fixture,obligation,task,request,evaluate


def role(role_name,reference):
    return SourceScopeRequest(selector=SourceScopeSelector(kind='section',reference=reference,match='role',role=role_name))


@pytest.mark.parametrize('reference',['摘要','overview','résumé'])
def test_declared_summary_role_uses_structural_heading_not_reference_language(reference):
    raw='Executive Summary\nLimit 24.\nImplementation\nLimit 36.'
    split=raw.index('Implementation')
    index=index_fixture(raw,[('document','Unit manual',0,len(raw)),('section','Executive Summary',0,split),
        ('section','Implementation',split,len(raw))])
    fixed=task(obligation(role('summary',reference)),f'Read {reference}')
    assert index.bind(fixed)[0].bindings[0].reason=='resolved'
    assert evaluate(index,fixed,[(0,split)]).state=='satisfied'
    assert evaluate(index,fixed,[(split,len(raw))]).state=='unsatisfied'
    index=StructureScopeIndex(corpus=index.corpus,nodes=index.nodes,task_hash=fixed.identity)
    assert build_location_request(index=index,task=fixed,query_scores=[.7]) is None


def test_body_role_subtracts_summary_and_contents_even_from_broad_parent():
    raw='Unit manual\nExecutive Summary\nLimit 24.\nContents\nSection 1\nImplementation\nLimit 36.'
    summary=raw.index('Executive Summary')
    contents=raw.index('Contents')
    detail=raw.index('Implementation')
    index=index_fixture(raw,[('document','Unit manual',0,len(raw)),('section','Unit manual',0,len(raw)),
        ('section','Executive Summary',summary,contents),('section','Contents',contents,detail),
        ('section','Implementation',detail,len(raw))])
    fixed=task(obligation(role('detail','正文')),'Read 正文')
    binding=index.bind(fixed)[0].bindings[0]
    spans=[(part.start,part.end) for part in binding.fact.intervals]
    assert spans==[(0,summary),(detail,len(raw))]
    assert evaluate(index,fixed,[(summary,detail)]).state=='unsatisfied'
    assert evaluate(index,fixed,[(detail,len(raw))]).state=='satisfied'
    assert {'unit-n2','unit-n3'}<=set(binding.node_ids)


def test_role_still_intersects_the_named_document_scope():
    raw='Executive Summary\nFirst source content.'
    index=index_fixture(raw,[('document','Unit manual',0,len(raw)),('section','Executive Summary',0,len(raw))])
    scope=SourceScopeRequest(op='intersection',children=(request('document','Different manual'),role('summary','摘要')))
    fixed=task(obligation(scope),'Read 摘要 in Different manual')
    assert index.bind(fixed)[0].bindings[0].reason=='no_verified_match'


def test_metadata_title_without_raw_declaration_cannot_supply_summary_role():
    raw='A catalogue mentions an abstract but has no such section.'
    index=index_fixture(raw,[('section','Executive Summary',0,len(raw))])
    fixed=task(obligation(role('summary','摘要')),'Read 摘要')
    assert index.bind(fixed)[0].bindings[0].fact.resolution=='unresolved'


def test_literal_title_does_not_expand_to_role_and_old_hash_shape_stays_unchanged():
    literal=SourceScopeSelector(kind='section',reference='Summary')
    assert literal.model_dump(mode='json')=={'kind':'section','reference':'Summary','match':'title'}
    with pytest.raises(ValueError,match='role_shape'):
        SourceScopeSelector(kind='section',reference='Summary',role='summary')
    with pytest.raises(ValueError,match='role_shape'):
        SourceScopeSelector(kind='table',reference='Table 2',match='role',role='detail')


def test_many_section_role_projects_local_witnesses_without_losing_global_audit():
    from app.services.evidence_scope import scope_target_plan
    parts=[f'Section {index}\nValue {index}.\n' for index in range(80)]
    raw=''.join(parts)
    boundaries=[sum(map(len,parts[:index])) for index in range(81)]
    index=index_fixture(raw,[('section',f'Section {i}',boundaries[i],boundaries[i+1]) for i in range(80)],cuts=tuple(boundaries[1:-1]))
    fixed=task(obligation(role('detail','正文')),'Read 正文')
    before=index.bind(fixed)[0].bindings[0]
    assert len(before.fact.witness_ids)==80
    targets,plan=scope_target_plan(index=index,task=fixed,token_budget=1000,target_limit=4)
    assert targets and plan['status']=='proposed'
    assert index.bind(fixed)[0].bindings[0]==before


@pytest.mark.asyncio
async def test_planning_prompt_distinguishes_shared_expression_and_local_obligation(no_fallback_env):
    import json
    from app.services.retrieval_models import RetrievalModels,compile_task_plan
    class Provider:
        api_protocol,model='anthropic','unit-test-model'
        async def classify_json_bounded(self,system_prompt,user_prompt,fallback,*,max_tokens):
            assert 'must NOT contain a scope or mode field' in system_prompt
            assert 'requirement.source_scope is a COVERAGE obligation' in system_prompt
            return {'source_references':[{'reference': 'Unit manual', 'role': 'named_document'}], 'perception':{'intent':'comparison','direct_answer_kind':'none','entities':[]},
                'answer_shape':'comparison','shared_source_scope':request('document','Unit manual').model_dump(mode='json'),
                'requirements':[{'facet':'summary limit','lexical_role':'domain','source_scope':obligation(role('summary','摘要')).model_dump(mode='json')},
                    {'facet':'body limit','lexical_role':'domain','source_scope':obligation(role('detail','正文')).model_dump(mode='json')}]}
        def provider_call_audit(self):
            return {}
    question='Compare 摘要 and 正文 in Unit manual.'
    plan,_=await RetrievalModels(Provider).plan(question=question,history_summary='',timeout_seconds=1,max_tokens=2048)
    _,fixed,_,_=compile_task_plan(plan,question=question,knowledge_base_id='unit-kb',conversation_scope_hash='a'*64,retrieval_granularity='mid')
    assert all(item.source_scope.scope.op=='intersection' for item in fixed.requirements)
