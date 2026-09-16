from copy import deepcopy

import pytest

from app.services.retrieval_models import TaskPlanningOutput, project_task_perception
from test_retrieval_models import plan_payload


@pytest.mark.parametrize('direct', [False, True])
def test_compact_and_full_perception_have_same_validated_execution_fields(direct):
    question='Find maximum queue waiting time.'
    full=plan_payload(direct)
    compact=deepcopy(full)
    for key in ('needs_graph','suggested_strategy','sub_queries'):
        compact['perception'].pop(key)
    expected,_=project_task_perception(TaskPlanningOutput.model_validate(full),question)
    actual,audit=project_task_perception(TaskPlanningOutput.model_validate(compact),question)
    assert actual==expected
    assert set(audit['derived_fields'])=={'needs_graph','suggested_strategy','sub_queries'}
    assert audit['subquery_source']=='original_question'


def test_explicit_conflict_is_not_silently_replaced():
    payload=plan_payload(True)
    payload['perception']['needs_graph']=True
    with pytest.raises(ValueError,match='cannot request graph'):
        project_task_perception(TaskPlanningOutput.model_validate(payload),'Who are you?')


def test_long_question_remains_intact_with_bounded_subquery_projection():
    payload=plan_payload()
    payload['perception'].pop('sub_queries')
    question='context '*700
    actual,audit=project_task_perception(TaskPlanningOutput.model_validate(payload),question)
    assert actual['sub_queries']==[payload['requirements'][0]['facet']]
    assert audit['original_question_preserved'] is True
    assert question=='context '*700
