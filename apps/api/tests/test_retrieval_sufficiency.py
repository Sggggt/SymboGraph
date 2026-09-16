import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.retrieval_control_contracts import (
    GenerationSourceScopeGuidance, RetrievalGateDecision, SourceGateAdmission, control_hash,
)
from app.services.retrieval_sufficiency import (
    EvidenceSufficiency, constrain_gate, repair_feedback, replay_sufficiency,
    sufficiency_packet, validate_sufficiency,
)
from app.services.retrieval_models import RetrievalModels
from app.services.reflection_models import AnswerReviewModelError
from test_retrieval_path_features import task_fixture


def evidence_fixture():
    sources = [{'source_handle':'src_1','text':'The approved specification defines two operating modes.'},
               {'source_handle':'src_2','text':'The earlier draft lists one example mode.'}]
    return SimpleNamespace(model_sources=lambda:sources, by_handle=lambda:{s['source_handle']:s for s in sources},
        manifest_hash='e'*64,package_id='unit-test-package')


def result_fixture(task, *, status='covered', reason='supported'):
    return EvidenceSufficiency(question_complete=True,requirements=[
        {'facet_id':item.id,'status':status,'reason':reason,'source_handles':['src_1'],
         'gap':'' if status=='covered' else 'The requested responsibility is not fully supported.'}
        for item in task.requirements])


def gate_fixture(outcome='ready_full'):
    return RetrievalGateDecision(outcome=outcome,missing_facet_ids=(),reason_codes=('fixed_task_coverage_passed',),
        feature_hash='f'*64,threshold_hash='a'*64)


@pytest.mark.parametrize('reason',['missing_attribute','incomplete_set','wrong_source','conflicting_scope','ambiguous_evidence','no_usable_evidence'])
def test_generic_gap_vetoes_similarity_readiness_and_uses_existing_repair_budget(reason):
    task=task_fixture()
    result=result_fixture(task,status='uncertain',reason=reason)
    blocked=constrain_gate(gate_fixture(),task=task,result=result,remaining_repairs=1)
    assert blocked.outcome=='scoped_not_found' and blocked.corpus_absence_proven is False
    repaired=constrain_gate(gate_fixture(),task=task,result=result,remaining_repairs=1,actionable_ids=('c1',))
    assert repaired.outcome=='repairable' and repaired.proposed_action_ids==('c1',)
    exhausted=constrain_gate(gate_fixture(),task=task,result=result,remaining_repairs=0,actionable_ids=('c1',))
    assert exhausted.outcome=='budget_exhausted'
    feedback=repair_feedback(result,blocked.missing_facet_ids)
    assert feedback['is_evidence'] is False and 'source_handles' not in json.dumps(feedback)


@pytest.mark.parametrize('outcome',['technical_failure','source_incomplete','source_unresolved','scope_ambiguous','representation_incomplete'])
def test_model_cannot_override_deterministic_failure(outcome):
    task=task_fixture()
    original=gate_fixture(outcome)
    assert constrain_gate(original,task=task,result=result_fixture(task),remaining_repairs=1)==original


@pytest.mark.parametrize('attack',['missing_facet','duplicate_facet','foreign_handle','foreign_question','foreign_facet'])
def test_closed_question_and_current_evidence_validator(attack):
    task,evidence=task_fixture(),evidence_fixture()
    data=result_fixture(task).model_dump(mode='json')
    if attack=='missing_facet':
        data['requirements']=data['requirements'][:-1]
    elif attack=='duplicate_facet':
        data['requirements'].append(data['requirements'][0])
    elif attack=='foreign_handle':
        data['requirements'][0]['source_handles']=['src_999']
    else:
        data.update(question_complete=False,unrepresented_question_span='invented original question' if attack=='foreign_question' else task.question[:8],
            affected_facet_ids=['foreign'] if attack=='foreign_facet' else [task.requirements[0].id])
    with pytest.raises(ValueError):
        validate_sufficiency(EvidenceSufficiency.model_validate(data),task=task,evidence=evidence,source_scopes=None)


def test_scoped_requirement_cannot_claim_another_source():
    task,evidence=task_fixture(),evidence_fixture()
    guidance=GenerationSourceScopeGuidance(requirements=[{'requirement_id':task.requirements[0].id,
        'sources':[{'source_handle':'src_2','text_char_spans':[[0,8]]}]}])
    with pytest.raises(ValueError,match='outside_required_scope'):
        validate_sufficiency(result_fixture(task),task=task,evidence=evidence,source_scopes=guidance)


def test_omitted_original_responsibility_cannot_pass_even_with_covered_facets():
    task=task_fixture()
    result=result_fixture(task).model_copy(update={'question_complete':False,
        'unrepresented_question_span':task.question[:8],'affected_facet_ids':(task.requirements[0].id,)})
    validate_sufficiency(result,task=task,evidence=evidence_fixture(),source_scopes=None)
    assert constrain_gate(gate_fixture(),task=task,result=result,remaining_repairs=1).outcome=='scoped_not_found'


@pytest.mark.parametrize('change',['task_hash','strategy_hash','evidence_manifest_hash','context_package_id','result','input_hash'])
def test_sufficiency_replay_binds_actual_evidence_task_and_result(change):
    task,evidence=task_fixture(),evidence_fixture()
    audit={'protocol_version':'retrieval_sufficiency_call_v1','status':'completed','task_hash':task.identity,
        'strategy_hash':'s'*64,'context_package_id':evidence.package_id,'evidence_manifest_hash':evidence.manifest_hash,
        'input_hash':control_hash(sufficiency_packet(task=task,evidence=evidence,source_scopes=None)),
        'result':result_fixture(task).model_dump(mode='json')}
    audit['audit_hash']=control_hash(audit)
    assert replay_sufficiency(audit,task=task,strategy_hash='s'*64,evidence=evidence,source_scopes=None)
    audit[change]='changed'
    with pytest.raises(ValueError,match='changed'):
        replay_sufficiency(audit,task=task,strategy_hash='s'*64,evidence=evidence,source_scopes=None)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',[None,'timeout','invalid_schema'])
async def test_bounded_model_decision_uses_original_evidence_once_no_draft(no_fallback_env,failure):
    task,evidence=task_fixture(),evidence_fixture()
    calls=[]
    class Provider:
        api_protocol,model='anthropic','unit-test-model'
        async def classify_json_bounded(self,system_prompt,user_prompt,fallback,*,max_tokens):
            packet=json.loads(user_prompt)
            calls.append(packet)
            assert packet==sufficiency_packet(task=task,evidence=evidence,source_scopes=None)
            assert 'answer_draft' not in packet and max_tokens==1024 and fallback is None
            if failure=='timeout':
                raise TimeoutError()
            if failure=='invalid_schema':
                return {'answer':'not allowed'}
            return result_fixture(task).model_dump(mode='json')
        def provider_call_audit(self):
            return {}
    operation=RetrievalModels(Provider).assess_evidence(task=task,evidence=evidence,source_scopes=None,timeout_seconds=1,max_tokens=1024)
    if failure:
        with pytest.raises(AnswerReviewModelError) as error:
            await operation
        assert error.value.stage=='evidence_sufficiency'
    else:
        result,audit=await operation
        assert result.question_complete and audit['model_call_count']==1
    assert len(calls)==1


def test_legacy_admission_hash_omits_new_empty_authority():
    payload=dict(run_id='r',knowledge_base_id='k',context_package_id='p',retrieval_trace_id='t',task_hash='a'*64,
        strategy_hash='b'*64,feature_hash='c'*64,outcome='ready_full',evidence_manifest_hash='d'*64,
        provenance_session_hash='e'*64,source_chunk_ids=('c',))
    admission=SourceGateAdmission(**payload)
    assert 'evidence_sufficiency_hash' not in admission.model_dump(mode='json')
    assert 'source_addressed_assessment_hash' not in admission.model_dump(mode='json')
    old={**payload,'source_chunk_ids':['c'],'protocol_version':'retrieval_source_admission_v1',
        'post_generation_model_review_count':0}
    assert admission.model_dump(mode='json')==old and admission.identity==control_hash(old)
