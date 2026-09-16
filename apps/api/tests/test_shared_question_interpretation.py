"""Question interpretation is one shared prompt contract, never a local veto rewrite."""
import json

import pytest

from app.services.retrieval_models import RetrievalModels,QUESTION_INTERPRETATION_RULES
from app.services.retrieval_sufficiency import constrain_gate
from test_retrieval_models import plan_payload
from test_retrieval_path_features import task_fixture
from test_retrieval_sufficiency import evidence_fixture,result_fixture,gate_fixture


@pytest.mark.asyncio
async def test_planner_and_evaluator_receive_same_interpretation_and_preserve_model_result(no_fallback_env):
    task=task_fixture()
    calls=[]
    proposed=result_fixture(task).model_copy(update={'question_complete':False,
        'unrepresented_question_span':task.question[:8],'affected_facet_ids':('f1',)})
    class Provider:
        api_protocol,model='anthropic','unit-test-model'
        async def classify_json_bounded(self,system_prompt,user_prompt,fallback,*,max_tokens):
            calls.append((system_prompt,json.loads(user_prompt)))
            if 'RETRIEVAL TASK PLANNING V2' in system_prompt:
                return {**plan_payload(),'source_references':[]}
            return proposed.model_dump(mode='json')
        def provider_call_audit(self): return {}
    model=RetrievalModels(Provider)
    _,planning_audit=await model.plan(question=task.question,history_summary='',timeout_seconds=1,max_tokens=1024)
    assessed,assessment_audit=await model.assess_evidence(task=task,evidence=evidence_fixture(),
        source_scopes=None,timeout_seconds=1,max_tokens=1024)
    assert len(calls)==2 and all(QUESTION_INTERPRETATION_RULES in system for system,_ in calls)
    assert all(packet['current_user']['question']==task.question for _,packet in calls)
    assert assessed==proposed  # no phrase-specific postprocessing of an evaluator's verdict
    assert constrain_gate(gate_fixture(),task=task,result=assessed,remaining_repairs=1).outcome=='scoped_not_found'
    assert planning_audit['prompt_protocol_hash']!=assessment_audit['prompt_protocol_hash']
    assert planning_audit['model_call_count']==assessment_audit['model_call_count']==1
