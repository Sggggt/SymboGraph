import pytest
from app.services.retrieval_models import RequirementProposal, normalize_source_roles


def requirement(**kwargs):
    return RequirementProposal(facet='Mission allocation', lexical_role='domain', kind='quantity', **kwargs)


def test_model_cannot_add_table_and_section_requirements_to_an_ordinary_quantity_question():
    items, audit = normalize_source_roles('How many days were allocated to Mission A?',
                                         (requirement(source_roles=('table', 'summary')),))
    assert items[0].source_roles == ()
    assert audit['rejected_additional_role_count'] == 2


def test_explicit_source_comparison_keeps_each_role_as_an_atomic_requirement():
    items, audit = normalize_source_roles('请核对报告摘要中的数值与后面的详细说明。',
                                         (requirement(source_roles=('summary', 'detail')),))
    assert [item.source_roles for item in items] == [('summary',), ('detail',)]
    assert audit['output_requirement_count'] == 2
    with pytest.raises(ValueError, match='not_represented'):
        normalize_source_roles('比较报告摘要和正文。', (requirement(),))


def test_output_table_format_does_not_become_an_input_source_constraint():
    items, _ = normalize_source_roles('请用表格列出结果并详细说明。', (requirement(source_roles=('table', 'detail')),))
    assert items[0].source_roles == ()
    items, _ = normalize_source_roles('请解释报告表1中的数值。', (requirement(),))
    assert items[0].source_roles == ('table',)


def test_split_source_roles_compile_through_the_real_facet_contract():
    from app.services.retrieval_models import TaskPlanningOutput, compile_task_plan
    question = '请核对报告摘要中的数值与后面的详细说明。'
    plan = TaskPlanningOutput.model_validate({'perception': {'intent': 'comparison', 'direct_answer_kind': 'none',
        'entities': [], 'sub_queries': [question], 'needs_graph': True, 'suggested_strategy': 'local_graph'},
        'requirements': [{'facet': '报告数值', 'lexical_role': 'domain', 'kind': 'comparison',
                          'source_roles': ['summary', 'detail']}], 'answer_shape': 'comparison'})
    _, task, lexical, packet = compile_task_plan(plan, question=question, knowledge_base_id='unit-test-kb',
        conversation_scope_hash='a' * 64, retrieval_granularity='mid')
    assert len(task.requirements) == len(packet['required_facets']) == 2
    assert {role for item in task.requirements for role in item.source_roles} == {'summary', 'detail'}
    lexical.validate_task(task)
