from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.retrieval_control_contracts import GateThresholds, control_hash
from app.services.retrieval_path_features import compute_path_features
from app.services.retrieval_packing import plan_packing_repair
from test_retrieval_path_features import task_fixture, strategy, candidate, parameters


def packing_fixture(*, clipped=True, supported=True, upper=.9):
    task = task_fixture()
    lexical = strategy(task, ('queue',))
    item = candidate('unit-test-source', value=0 if clipped else upper, upper=upper, path_observed=supported)
    params = parameters()
    feature = compute_path_features(task=task, strategy=lexical, panels=(), packaged_candidates=(item,), parameters=params)
    replay = {'task': task.model_dump(mode='json'), 'strategy': lexical.model_dump(mode='json'),
        'panels': [], 'package': [item.model_dump(mode='json')], 'parameters': params.model_dump(mode='json')}
    package = SimpleNamespace(id='unit-test-package', package_json={'chunks': [
        {'chunk_id': item.id, 'content': 'The maximum queue waiting', 'char_span': [0, 25], 'content_clipped': clipped}]})
    return dict(task=task, package=package, features=feature, replay_input=replay,
                thresholds=GateThresholds(coverage=.35, path_quality=.1, calibration_id='unit-test'))


def test_packing_plan_requires_observed_loss_and_possible_fixed_task_gain():
    args = packing_fixture()
    plan = plan_packing_repair(**args)
    assert plan.target_chunk_ids == ('unit-test-source',) and plan.model_call_count == 0
    assert plan.focus == ('maximum queue waiting time',)
    assert plan_packing_repair(**packing_fixture(clipped=False)) is None
    assert plan_packing_repair(**packing_fixture(supported=False)) is None
    assert plan_packing_repair(**packing_fixture(upper=.1)) is None


def test_packing_plan_signature_ignores_cloned_package_id_but_rejects_input_tampering():
    args = packing_fixture()
    before = plan_packing_repair(**args)
    args['package'].id = 'unit-test-clone'
    assert plan_packing_repair(**args).input_signature == before.input_signature
    changed = deepcopy(args['replay_input'])
    changed['package'][0]['canonical_entry_distance'] = 0
    with pytest.raises(ValueError, match='identity_changed'):
        plan_packing_repair(**{**args, 'replay_input': changed})


def test_restoration_cost_does_not_claim_a_second_retrieval():
    from app.services.retrieval_reward import build_reward_observation
    args = packing_fixture()
    observation = build_reward_observation(features=args['features'], baseline=None, operations=('restore_context',),
        extra_elapsed_seconds=1, source_integrity=True, time_weight=.05, work_weight=.01, time_scale_seconds=40)
    assert observation['additional_retrieval_count'] == 0
    assert observation['cost'] > 0
