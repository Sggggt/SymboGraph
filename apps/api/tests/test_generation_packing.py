from types import SimpleNamespace

import pytest

from app.services.generation_packing import plan_generation_packing, verify_generation_packing
from app.services.retrieval_path_features import compute_path_features
from test_retrieval_path_features import task_fixture, strategy, candidate, parameters


def fixture():
    task = task_fixture()
    lexical = strategy(task, ('queue',))
    candidates = (candidate('a', value=.9, distance=2), candidate('b', value=.6, distance=.1),
        candidate('c', value=.2, distance=1), candidate('d', value=.85, path_observed=False),
        candidate('unused', value=.1, distance=1))
    params = parameters()
    feats = compute_path_features(task=task, strategy=lexical, panels=(), packaged_candidates=candidates, parameters=params)
    inputs = {'task': task.model_dump(mode='json'), 'strategy': lexical.model_dump(mode='json'), 'panels': [],
        'package': [c.model_dump(mode='json') for c in candidates], 'parameters': params.model_dump(mode='json')}
    package = SimpleNamespace(id='unit-test-package', token_budget=1000, package_json={'chunks': [
        {'chunk_id': c.id, 'content': f'Unit test complete source {c.id}.', 'content_clipped': False,
         'structure_closure': {'previous_chunk_id': None, 'next_chunk_id': 'c' if c.id == 'b' else None}}
        for c in candidates]})
    return task, lexical, candidates, params, feats, inputs, package


def test_compaction_preserves_four_bounds_and_existing_neighbor():
    task, lexical, candidates, params, old, inputs, package = fixture()
    plan = plan_generation_packing(task=task, package=package, features=old, replay_input=inputs)
    assert set(plan.witness_chunk_ids) == {'a', 'b', 'd'}
    assert plan.neighbor_chunk_ids == ('c',)
    assert plan.selected_source_count == 4 and plan.selected_text_characters < plan.original_text_characters
    kept = tuple(c for c in candidates if c.id != 'unused')
    after = compute_path_features(task=task, strategy=lexical, panels=(), packaged_candidates=kept, parameters=params)
    verify_generation_packing(old, after)
    for lost in ('a', 'b', 'd'):
        changed = compute_path_features(task=task, strategy=lexical, panels=(),
            packaged_candidates=tuple(c for c in kept if c.id != lost), parameters=params)
        with pytest.raises(ValueError, match='coverage_changed'):
            verify_generation_packing(old, changed)


def test_compaction_declines_clipped_witness_and_rejects_changed_input():
    task, _, _, _, feats, inputs, package = fixture()
    package.package_json['chunks'][0]['content_clipped'] = True
    assert plan_generation_packing(task=task, package=package, features=feats, replay_input=inputs) is None
    with pytest.raises(ValueError, match='input_changed'):
        plan_generation_packing(task=task, package=package, features=feats, replay_input={**inputs, 'panels': [{}]})
