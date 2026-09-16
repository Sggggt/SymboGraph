"""Generic location/coverage/path matrix, independent of final acceptance data."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.retrieval_control_contracts import (
    EvidenceInterval, GateThresholds, PathFeatureCandidate, SourceScopeObligation, control_hash,
)
from app.services.answer_sources import AnswerEvidenceManifest, _manifest_identity
from app.services.retrieval_path_features import compute_path_features, decide_retrieval_gate
from app.services.retrieval_sufficiency import constrain_gate, validate_sufficiency
from app.services.source_addressed_assessment import (
    build_assessment, assessment_guidance, replay_assessment, verify_assessment_owner,
)
from test_evidence_scope import index_fixture, task, request, obligation
from test_retrieval_path_features import parameters, strategy
from test_retrieval_sufficiency import result_fixture


def fixture(*, kind='table', node_type='table', score=.3, path_observed=True, distance=.1,
            clipped=False, incomplete=False, foreign_version=False, scoped=True, op=None):
    text = 'Operating modes\nCooling uses 18 units. Heating uses 27 units.'
    cut = text.index('Heating')
    specs = [(node_type,'Operating modes',0,len(text))]
    declaration = obligation(request(kind,'Operating modes'))
    if op:
        specs = [('section','Cooling',0,cut), ('section','Heating',cut,len(text))]
        declaration = SourceScopeObligation(op=op, children=tuple(obligation(request('section',t)) for t in ('Cooling','Heating')))
    index = index_fixture(text,specs,incomplete=incomplete)
    fixed = task(declaration if scoped else None, 'Use Operating modes Cooling and Heating.')
    lexical = strategy(fixed, ('operating',))
    end = cut if clipped else len(text)
    version = 'unit-v2' if foreign_version else 'unit-v1'
    interval = EvidenceInterval(knowledge_base_id=fixed.knowledge_base_id,document_version_id=version,start=0,end=end)
    params = parameters().model_copy(update={'protocol_version':'canonical_task_path_quality_v4',
        'scope_inputs':index.bind(fixed), 'packed_scope_intervals':(interval,), 'scope_source_chunk_ids':('unit-c0',)})
    candidate = PathFeatureCandidate(id='unit-c0',source_id=f'unit-c0:0:{end}',topic_group='unit-doc',
        source_valid=True,path_observed=path_observed,opportunities=({'facet_id':'f1','value':{'lower':score,'upper':score}},),
        canonical_entry_distance=distance,routing_cost=0)
    inputs = {'task':fixed.model_dump(mode='json'),'strategy':lexical.model_dump(mode='json'),
        'panels':[], 'package':[candidate.model_dump(mode='json')], 'parameters':params.model_dump(mode='json')}
    features = compute_path_features(task=fixed,strategy=lexical,panels=(),packaged_candidates=(candidate,),parameters=params)
    thresholds = GateThresholds(coverage=.35,path_quality=.1,calibration_id='unit-thresholds')
    decision = decide_retrieval_gate(task=fixed,features=features,thresholds=thresholds,source_integrity=True,remaining_repairs=1)
    sources = ({'source_handle':'src_1','chunk_id':'unit-c0','text':text[:end],
        'source_span':{'document_version_id':version,'char_span':[0,end]},'package_item':{}},)
    evidence = AnswerEvidenceManifest('unit-package','unit-trace',sources,control_hash(_manifest_identity('unit-package','unit-trace',sources)))
    return dict(task=fixed,features=features,decision=decision,replay_input=inputs,evidence=evidence,
        source_audit={'all_valid':True,'provenance_session_hash':'e'*64},thresholds=thresholds)


@pytest.mark.parametrize('kind,node_type',[('table','table'),('formula','formula'),('code','code_block'),('section','section'),('text','paragraph')])
def test_complete_low_similarity_source_can_be_assessed_without_altering_scores(kind,node_type):
    args = fixture(kind=kind,node_type=node_type)
    original = deepcopy(args['replay_input'])
    proof = build_assessment(**args)
    assert proof and proof.generation_authorized is False and proof.model_call_count == 0
    assert proof.facets[0].coverage.lower == .3
    assert all(p.mode == 'complete' and p.state == 'satisfied' for p in proof.facets[0].complete_coverage)
    result = result_fixture(args['task'])
    validate_sufficiency(result,task=args['task'],evidence=args['evidence'],source_scopes=assessment_guidance(proof))
    final = constrain_gate(args['decision'],task=args['task'],result=result,remaining_repairs=1,assessment=proof)
    assert final.outcome == 'ready_full' and args['decision'].outcome == 'scoped_not_found'
    assert args['replay_input'] == original


@pytest.mark.parametrize('changes',[{'clipped':True},{'incomplete':True},{'foreign_version':True},
    {'path_observed':False},{'distance':4},{'scoped':False},{'score':.08},{'score':.8}])
def test_ineligible_sources_do_not_start_this_branch(changes):
    assert build_assessment(**fixture(**changes)) is None


@pytest.mark.parametrize('failure',['source_incomplete','technical_failure','source_unresolved','scope_ambiguous','representation_incomplete'])
def test_technical_and_source_failure_cannot_be_promoted(failure):
    args=fixture()
    args['decision']=args['decision'].model_copy(update={'outcome':failure})
    assert build_assessment(**args) is None


@pytest.mark.parametrize('op,expected',[('any',True),('all',False)])
def test_any_selects_complete_alternative_and_all_keeps_every_responsibility(op,expected):
    proof=build_assessment(**fixture(op=op,clipped=True))
    assert bool(proof) is expected
    if proof:
        assert len(proof.facets[0].selected_request_hashes)==1
        assert len(proof.facets[0].complete_coverage)==2


@pytest.mark.parametrize('reason',['missing_attribute','conflicting_scope','incomplete_set','ambiguous_evidence'])
def test_location_completeness_never_overrides_semantic_gap_even_when_partial_allowed(reason):
    args=fixture()
    proof=build_assessment(**args)
    result=result_fixture(args['task'],status='uncertain',reason=reason)
    assert constrain_gate(args['decision'],task=args['task'],result=result,remaining_repairs=1,assessment=proof).outcome=='scoped_not_found'
    assert constrain_gate(args['decision'],task=args['task'],result=result,remaining_repairs=1,
        actionable_ids=('unit-direction',),assessment=proof).outcome=='repairable'
    assert constrain_gate(args['decision'],task=args['task'],result=result,remaining_repairs=0,assessment=proof).outcome=='budget_exhausted'


@pytest.mark.parametrize('attack',['outside_package','outside_scope','missing_question'])
def test_evaluator_misclassification_cannot_invent_source_authority(attack):
    args=fixture()
    proof=build_assessment(**args)
    result=result_fixture(args['task'])
    guidance=assessment_guidance(proof)
    if attack=='outside_package':
        result=result.model_copy(update={'requirements':(result.requirements[0].model_copy(update={'source_handles':('foreign',)}),)})
    elif attack=='outside_scope':
        guidance=guidance.model_copy(update={'requirements':(guidance.requirements[0].model_copy(update={'sources':()}),)})
    else:
        result=result.model_copy(update={'question_complete':False,'unrepresented_question_span':'Operating modes','affected_facet_ids':('f1',)})
        validate_sufficiency(result,task=args['task'],evidence=args['evidence'],source_scopes=guidance)
        assert constrain_gate(args['decision'],task=args['task'],result=result,remaining_repairs=1,assessment=proof).outcome=='scoped_not_found'
        return
    with pytest.raises(ValueError,match='outside'):
        validate_sufficiency(result,task=args['task'],evidence=args['evidence'],source_scopes=guidance)


@pytest.mark.parametrize('attack',['scope','score','threshold','manifest','provenance','missing'])
def test_proof_replay_rejects_tampering_even_after_rehash(attack):
    args=fixture()
    proof=build_assessment(**args)
    raw=proof.model_dump(mode='json')
    if attack=='scope': raw['facets'][0]['usable_intervals'][0]['end']-=1
    elif attack=='score': raw['facets'][0]['coverage']['lower']=.29
    elif attack=='threshold': raw['thresholds']['path_quality']=.01
    elif attack=='manifest': raw['evidence_manifest_hash']='f'*64
    elif attack=='provenance': raw['provenance_session_hash']='f'*64
    payload={'source_addressed_assessment':raw,'source_addressed_assessment_hash':control_hash(raw)}
    if attack=='missing': payload.pop('source_addressed_assessment')
    with pytest.raises(ValueError,match='source_addressed'):
        replay_assessment(payload,task=args['task'],features=args['features'],path_decision=args['decision'],
            replay_input=args['replay_input'],evidence=args['evidence'],source_audit=args['source_audit'])


def test_cancellation_and_legacy_identity_are_preserved(monkeypatch):
    import asyncio
    from app.services import source_addressed_assessment as module
    def cancelled(): raise asyncio.CancelledError()
    monkeypatch.setattr(module,'raise_if_source_io_cancelled',cancelled)
    with pytest.raises(asyncio.CancelledError): build_assessment(**fixture())
    assert replay_assessment({},task=None,features=None,path_decision=None,replay_input=None,evidence=None) is None
    verify_assessment_owner(SimpleNamespace(metadata_json={}),{})


def test_generation_source_addresses_cannot_expand_assessed_scope():
    from app.services.source_addressed_assessment import validate_bound_source_addresses
    args=fixture()
    proof=build_assessment(**args)
    card={'evidence_sufficiency':{'source_addressed_assessment':proof.model_dump(mode='json')}}
    span=args['evidence'].sources[0]['source_span']['char_span']
    validate_bound_source_addresses(card,[('unit-c0',span)])
    with pytest.raises(ValueError,match='outside_scope'):
        validate_bound_source_addresses(card,[('unit-c0',(0,span[1]+1))])
    with pytest.raises(ValueError,match='outside_scope'):
        validate_bound_source_addresses(card,[('other-chunk',span)])
