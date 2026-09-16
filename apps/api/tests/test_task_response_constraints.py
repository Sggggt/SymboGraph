from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.retrieval_control_contracts import TaskContract
from app.services.retrieval_models import RequirementProposal, TaskPlanningOutput, compile_task_plan
from app.services.task_constraints import response_constraints, split_response_requirements
from test_retrieval_path_features import task_fixture


def requirement(text, **kwargs):
    return RequirementProposal(facet=text, lexical_role='domain', **kwargs)


def test_response_only_facet_is_removed_before_weights_freeze_and_instruction_remains():
    question = '资料是否给出系统的工程计划？没有就明确说明，不要推测工程细节。'
    proposed = (requirement('系统工程计划存在性'), requirement('无证据时明确否定且不推测工程细节'))
    kept, constraints, audit = split_response_requirements(question, proposed)
    assert [item.facet for item in kept] == ['系统工程计划存在性']
    assert {item.kind for item in constraints} == {'no_speculation', 'insufficiency_notice'}
    assert all(question[item.char_span[0]:item.char_span[1]] == item.text for item in constraints)
    plan = TaskPlanningOutput.model_validate({'perception': {'intent':'analysis','direct_answer_kind':'none',
        'entities':[], 'sub_queries':[question], 'needs_graph':True,'suggested_strategy':'local_graph'},
        'requirements':[item.model_dump(mode='json') for item in proposed], 'answer_shape':'grounded_answer'})
    _, task, lexical, facets = compile_task_plan(plan, question=question, knowledge_base_id='unit-test-kb',
        conversation_scope_hash='a'*64, retrieval_granularity='mid')
    assert task.requirements[0].weight == 1 and task.response_constraints == constraints
    assert facets['required_facets'] == ['系统工程计划存在性']
    assert all('推测' not in term.surface for term in lexical.terms)
    assert audit['response_only_facets_removed'] == 1


@pytest.mark.parametrize('question, facet', [
    ('文档是否规定不得推测缺失值？不要推测。', '不得推测缺失值的规定'),
    ('系统压力不能超过5 bar时如何处理？', '系统压力不超过5 bar的条件'),
    ('请解释模型不依赖独立性假设的原理。', '模型不依赖独立性假设的原理'),
])
def test_factual_negation_numeric_limit_and_document_policy_are_not_removed(question, facet):
    kept, _, audit = split_response_requirements(question, (requirement(facet),))
    assert kept[0].facet == facet and audit['response_only_facets_removed'] == 0


def test_protected_fact_and_source_roles_cannot_be_hidden_as_output_control():
    q='有什么结果？不要编造参数。'
    for item in (requirement('不要编造参数', protected_literals=('参数',)),
                 requirement('不要编造参数', source_roles=('table',))):
        kept, _, audit=split_response_requirements(q,(item,))
        assert kept and audit['response_only_facets_removed']==0


def test_english_conditional_and_format_instruction_have_exact_spans():
    q='What is the limit? If no evidence is available, state that clearly. Do not speculate.'
    constraints=response_constraints(q)
    assert {'insufficiency_notice','no_speculation'} <= {c.kind for c in constraints}
    assert all(q[c.char_span[0]:c.char_span[1]]==c.text for c in constraints)
    assert response_constraints('请用表格列出结果。')[0].kind=='output_format'


def test_legacy_task_hash_roundtrip_and_constraint_span_tampering():
    old=task_fixture().model_dump(mode='json')
    assert 'response_constraints' not in old
    assert TaskContract.model_validate(old).model_dump(mode='json')==old
    changed=deepcopy(old)
    changed['response_constraints']=[{'kind':'no_speculation','text':'invented','char_span':[0,3]}]
    with pytest.raises(ValidationError):TaskContract.model_validate(changed)
