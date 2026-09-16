import math
import pytest

from app.retrieval_control_contracts import DecisionPanel, Requirement, LexicalStrategy, LexicalTerm
from app.services.retrieval_path_features import select_panel, utility
from test_retrieval_path_features import task_fixture, candidate, parameters


def test_observed_rounding_residual_controls_routing_not_task_utility():
    task = task_fixture().model_copy(update={'requirements': tuple(
        Requirement(id=f'f{i}', text=f'Unit test facet {i}', weight=1/3) for i in range(3))})
    strategy = LexicalStrategy(task_hash=task.identity, revision=0, routing_text='Unit test',
        terms=tuple(LexicalTerm(id=f'a{i}', facet_id=f'f{i}', surface=f'term{i}') for i in range(3)))
    pool = (candidate('lower_dense', matched=('a0',)).model_copy(update={'routing_cost':-.340481}),
            candidate('higher_dense', matched=('a1',)).model_copy(update={'routing_cost':-.346307}))
    panel = DecisionPanel(id='unit-test-panel', candidates=pool, selected_ids=('lower_dense',), limit=1,
        routing_facet_weights={'f0':.333334, 'f1':.333333, 'f2':.333333})
    assert select_panel(task, strategy, panel)[0].id == 'lower_dense'
    legacy = panel.model_copy(update={'routing_facet_weights': {}})
    assert select_panel(task, strategy, legacy)[0].id == 'higher_dense'
    assert [f.weight for f in task.requirements] == [1/3]*3
    assert utility(task, pool, parameters()).lower == pytest.approx(.9 * math.exp(-.1) / 3)


def test_legacy_panel_serialization_does_not_gain_new_hash_fields():
    panel = DecisionPanel(id='unit-test-panel', candidates=(), selected_ids=(), limit=1)
    payload = panel.model_dump(mode='json')
    assert 'routing_facet_weights' not in payload
    assert DecisionPanel.model_validate(payload).model_dump(mode='json') == payload
