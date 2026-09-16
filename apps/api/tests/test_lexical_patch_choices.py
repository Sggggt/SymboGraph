import json

import pytest

from app.retrieval_control_contracts import Requirement,LexicalTerm,LexicalStrategy,control_hash
from app.services.lexical_patch import compile_lexical_patch
from app.services.lexical_patch_choices import selection_model,project_selection,replay_choice_projection
from test_lexical_patch import repair_candidate
from test_retrieval_path_features import task_fixture,strategy


def inputs(two=False):
    task=task_fixture()
    before=strategy(task,('old',))
    candidates=(repair_candidate(operations=('replace_surface','add_attested_alias')),)
    if two:
        task=task.model_copy(update={'requirements':(
            task.requirements[0].model_copy(update={'weight':0.5}),
            Requirement(id='f2',text='controller display',weight=0.5))})
        before=LexicalStrategy(task_hash=task.identity,revision=0,routing_text=task.question,
            terms=(*before.terms,LexicalTerm(id='other',facet_id='f2',surface='display')))
        candidates=(*candidates,repair_candidate(surface='monitor',relation='related_locator',
            operations=('locator_probe',)).model_copy(update={'id':'c2','facet_id':'f2'}))
    return task,before,candidates


def test_provider_choice_projects_to_the_existing_validated_patch_and_replays():
    task,before,candidates=inputs()
    output=selection_model(task=task,strategy=before,candidates=candidates)
    selection=output.model_validate({'choices':{'f1':{'operation':'replace_surface','candidate_ids':['c1'],'remove_term_ids':['old']}}})
    patch,audit=project_selection(selection)
    after=compile_lexical_patch(task=task,before=before,patch=patch,candidates=candidates)
    assert after.task_hash==before.task_hash and after.terms[0].surface=='queueing delay'
    assert replay_choice_projection(audit,task=task,strategy=before,candidates=candidates)==patch
    changed={**audit,'canonical_patch_hash':'a'*64}
    with pytest.raises(ValueError,match='projection_changed'):
        replay_choice_projection(changed,task=task,strategy=before,candidates=candidates)
    assert audit['additional_model_calls']==0


@pytest.mark.parametrize('bad',[
    {'outcome':'patch','patches':[]},
    {'choices':{}},
    {'choices':{'f1':{'operation':'none_supported'},'invented':{'operation':'none_supported'}}},
    {'choices':[{'facet_id':'f1','operation':'none_supported'},{'facet_id':'f1','operation':'none_supported'}]},
    {'choices':{'f1':{'operation':'none_supported','candidate_ids':['c1']}}},
    {'choices':{'f1':{'operation':'replace_surface','candidate_ids':['foreign'],'remove_term_ids':['old']}}},
    {'choices':{'f1':{'operation':'replace_surface','candidate_ids':['c1'],'remove_term_ids':['foreign']}}},
    {'choices':{'f1':{'operation':'add_attested_alias','candidate_ids':['c1'],'remove_term_ids':['old']}}},
    {'choices':{'f1':{'operation':'locator_probe','candidate_ids':['c1']}}},
])
def test_schema_rejects_invalid_branch_ownership_and_reference_shapes(bad):
    task,before,candidates=inputs()
    with pytest.raises(ValueError):
        selection_model(task=task,strategy=before,candidates=candidates).model_validate(bad)


def test_each_facet_has_one_choice_and_clarification_stops_the_whole_patch():
    task,before,candidates=inputs(two=True)
    output=selection_model(task=task,strategy=before,candidates=candidates)
    data={'choices':{'f1':{'operation':'none_supported'},'f2':{'operation':'locator_probe','candidate_ids':['c2']}}}
    patch,_=project_selection(output.model_validate(data))
    assert len(patch.patches)==1 and patch.patches[0].facet_id=='f2'
    data['choices']['f1']={'operation':'need_scope_clarification'}
    stopped,_=project_selection(output.model_validate(data))
    assert stopped.outcome=='need_scope_clarification' and not stopped.patches
    data['choices']['f1']={'operation':'replace_surface','candidate_ids':['c2']}
    with pytest.raises(ValueError):
        output.model_validate(data)


def test_related_locator_never_receives_synonym_operation_even_if_candidate_permission_is_inconsistent():
    task,before,candidates=inputs()
    candidate=candidates[0].model_copy(update={'relation':'related_locator','permitted_operations':('replace_surface','locator_probe')})
    output=selection_model(task=task,strategy=before,candidates=(candidate,))
    with pytest.raises(ValueError):
        output.model_validate({'choices':{'f1':{'operation':'replace_surface','candidate_ids':['c1']}}})
    patch,_=project_selection(output.model_validate({'choices':{'f1':{'operation':'locator_probe','candidate_ids':['c1']}}}))
    assert compile_lexical_patch(task=task,before=before,patch=patch,candidates=(candidate,)).locator_ids==(candidate.witness_id,)


def test_schema_is_bounded_and_selection_never_enumerates_candidate_combinations():
    task,before,candidates=inputs(two=True)
    candidates=tuple(candidates[index%2].model_copy(update={'id':f'c{index}'}) for index in range(6))
    output=selection_model(task=task,strategy=before,candidates=candidates)
    schema=output.model_json_schema()
    assert len(schema.get('$defs',{}))<=11
    assert len(json.dumps(schema))<16000
    assert control_hash(schema)==control_hash(selection_model(task=task,strategy=before,candidates=candidates).model_json_schema())
    with pytest.raises(ValueError,match='scope_invalid'):
        selection_model(task=task,strategy=before,candidates=(*candidates,candidates[0]))
