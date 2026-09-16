import pytest

from app.retrieval_control_contracts import Requirement, PathEvaluationParameters
from app.services.source_use import analyze_source_use, source_use_decision
from test_retrieval_path_features import task_fixture, parameters


def _interval(start, end, version='document-a-v1', kb='unit-kb'):
    from app.retrieval_control_contracts import EvidenceInterval
    return EvidenceInterval(knowledge_base_id=kb,document_version_id=version,start=start,end=end)


def _scope(sid, intervals=(), *, kind='text', resolution='verified', complete=True):
    from app.retrieval_control_contracts import EvidenceScopeFact
    return EvidenceScopeFact(id=sid,knowledge_base_id='unit-kb',kind=kind,resolution=resolution,
        extent_complete=complete if resolution=='verified' else False,intervals=tuple(intervals),
        witness_ids=('unit-witness-'+sid,) if resolution=='verified' else ())


def _scope_expr(op='scope', sid='a', children=()):
    from app.retrieval_control_contracts import EvidenceScopeExpression
    return EvidenceScopeExpression(op=op,scope_id=sid if op=='scope' else None,children=tuple(children))


@pytest.mark.parametrize('kind',['document','section','text','table','formula','code','figure','caption'])
def test_scope_interval_algebra_is_independent_of_source_kind(kind):
    from app.services.source_use import evaluate_evidence_scope
    scope = _scope('a',[_interval(10,40)],kind=kind)
    result = evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=_scope_expr(),scopes=(scope,),
        packed=(_interval(0,25),_interval(20,40),_interval(20,40)),mode='complete')
    assert result.state=='satisfied' and result.usable_intervals==(_interval(10,40),)
    assert result.missing_intervals==() and not result.semantic_sufficiency_claimed


def test_nested_scopes_intersect_before_evaluating_coverage():
    from app.services.source_use import evaluate_evidence_scope
    scopes = (_scope('chapter',[_interval(0,50)],kind='section'),_scope('object',[_interval(20,70)],kind='code'))
    expression = _scope_expr('intersection',children=(_scope_expr(sid='chapter'),_scope_expr(sid='object')))
    result = evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=expression,scopes=scopes,
        packed=(_interval(0,40),),mode='complete')
    assert result.state=='unsatisfied' and result.usable_intervals==(_interval(20,40),)
    assert result.missing_intervals==(_interval(40,50),)


def test_scope_intersection_never_borrows_another_document_version():
    from app.services.source_use import evaluate_evidence_scope
    scopes = (_scope('a',[_interval(0,20)]),_scope('b',[_interval(0,20,version='document-a-v2')]))
    expression = _scope_expr('intersection',children=(_scope_expr(),_scope_expr(sid='b')))
    result = evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=expression,scopes=scopes,
        packed=(_interval(0,20),),mode='complete')
    assert result.state=='unsatisfied' and result.reason=='empty_resolved_scope'


@pytest.mark.parametrize('resolution',['unresolved','unsupported'])
def test_unknown_scope_cannot_be_declared_complete_by_packing_a_larger_document(resolution):
    from app.services.source_use import evaluate_evidence_scope
    expression = _scope_expr('intersection',children=(_scope_expr(),_scope_expr(sid='b')))
    scopes = (_scope('a',[_interval(0,100)]),_scope('b',resolution=resolution))
    result = evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=expression,scopes=scopes,
        packed=(_interval(0,100),),mode='complete')
    assert result.state=='unknown' and result.usable_intervals==()


def test_partial_representation_and_optional_full_sources_have_distinct_semantics():
    from app.services.source_use import evaluate_evidence_scope, combine_scope_states
    known = _scope('a',[_interval(0,20)])
    partial = _scope('b',[_interval(30,40)],complete=False)
    def evaluate(expr, mode='complete'):
        return evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=expr,scopes=(known,partial),
            packed=(_interval(0,40),),mode=mode)
    assert evaluate(_scope_expr(sid='b')).state=='unknown'
    assert evaluate(_scope_expr(sid='b'),mode='overlap').state=='satisfied'
    union = _scope_expr('union',children=(_scope_expr(),_scope_expr(sid='b')))
    assert evaluate(union).state=='unknown'
    states = (evaluate(_scope_expr()).state,evaluate(_scope_expr(sid='b')).state)
    assert combine_scope_states(states,op='all')=='unknown'
    assert combine_scope_states(states,op='any')=='satisfied'


def test_scope_facts_require_witnesses_and_reject_cross_knowledge_base_inputs():
    from pydantic import ValidationError
    from app.retrieval_control_contracts import EvidenceScopeFact
    from app.services.source_use import evaluate_evidence_scope
    with pytest.raises(ValidationError,match='requires_ranges_and_witnesses'):
        EvidenceScopeFact(id='a',knowledge_base_id='unit-kb',kind='text',resolution='verified',intervals=(_interval(0,10),))
    with pytest.raises(ValueError,match='cross_knowledge_base'):
        evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=_scope_expr(),scopes=(_scope('a',[_interval(0,10)]),),
            packed=(_interval(0,10,kb='another-kb'),))


def test_scope_difference_matches_a_discrete_reference_on_independent_random_ranges():
    import random
    from app.services.source_use import evaluate_evidence_scope
    rng = random.Random(701)
    for _ in range(40):
        required = [_interval(start, start+rng.randint(1,12)) for start in rng.sample(range(35),5)]
        packed = [_interval(start, start+rng.randint(1,12)) for start in rng.sample(range(35),5)]
        scope = _scope('a',required)
        result = evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=_scope_expr(),scopes=(scope,),packed=tuple(packed),mode='complete')
        points = lambda spans: {position for span in spans for position in range(span.start,span.end)}
        assert points(result.usable_intervals)==points(required)&points(packed)
        assert points(result.missing_intervals)==points(required)-points(packed)


REFERENCES = '\n\n'.join(f'[{i}] A. Reader, "Reliable graphs and algorithms", IEEE Trans., vol. 2, pp. 10-20, 2020.' for i in range(1, 5))


def check(text, question='采用什么方法进行计算？', facet='计算方法', roles=()):
    task = task_fixture().model_copy(update={'question': question,
        'requirements': (Requirement(id='f1', text=facet, weight=1),)})
    return source_use_decision(task, task.requirements[0], analyze_source_use(text), roles=roles)


def test_reference_keywords_do_not_establish_a_method():
    result = check(REFERENCES + '\n\nThe paper introduces a new system, motivated by')
    assert result.metadata_dominant and not result.allowed
    assert result.reason == 'metadata_without_requested_form'


@pytest.mark.parametrize('statement', [
    'The algorithm uses bounded elimination to evaluate the system.',
    'This method is based on polynomial factorization.',
    '本文采用有界消元算法进行计算。',
])
def test_short_but_useful_statement_survives_reference_dominance(statement):
    assert check(REFERENCES + '\n\n' + statement).allowed


@pytest.mark.parametrize('mixed', [
    'Her research interests include computing. The algorithm uses elimination.',
    'The algorithm uses elimination. Her research interests include computing.',
])
def test_mixed_biography_keeps_the_actual_technical_statement(mixed):
    assert check(REFERENCES + '\n\n' + mixed).allowed


def test_bibliography_and_author_questions_keep_requested_metadata():
    assert check(REFERENCES, '有哪些参考文献？', '参考文献列表').allowed
    assert check('Her research interests include computing. She is currently a professor.',
        '作者是谁？', '作者信息').allowed


def test_cited_narrative_and_numbered_instruction_are_not_bibliography():
    narrative = 'Smith, A. (2020) showed that the algorithm terminates.'
    assert analyze_source_use(narrative).metadata_characters == 0
    assert check('[1] Use the elimination method to solve this equation.').allowed


def test_structured_proof_and_ordinary_text_are_preserved():
    assert check(REFERENCES + '\n\nx = a + b', roles={'formula'}).allowed
    assert check(REFERENCES + '\n\ndef solve(): return 1', roles={'code'}).allowed
    assert check(REFERENCES + '\n\nMethod | Count\nElimination | 10', roles={'table'}).allowed
    assert check('The method uses a fixed number of steps.').allowed


def test_evaluator_version_retains_old_input_roundtrip():
    old = parameters().model_dump(mode='json')
    assert old['protocol_version'] == 'canonical_task_path_quality_v1'
    assert PathEvaluationParameters.model_validate(old).model_dump(mode='json') == old
    assert PathEvaluationParameters.model_validate({**old, 'protocol_version': 'canonical_task_path_quality_v2'})


def test_discovery_skips_reference_only_candidate_and_observes_cancellation(monkeypatch):
    import numpy as np
    from app.services.retrieval_corpus import CorpusSource, RetrievalCorpus
    from test_retrieval_corpus import corpus
    from app.services import storage
    texts = (REFERENCES, 'The maximum queueing delay is bounded at 20 ms.')
    sources = [CorpusSource(f'unit-test-{i}', f'doc-{i}', f'version-{i}', 'Unit test',
        text, 0, len(text), 'a' * 64) for i, text in enumerate(texts)]
    view = RetrievalCorpus(knowledge_base_id='unit-test-kb', sources=sources,
        vectors=np.array([[1., 0.], [.8, .6]]), scope_hash='b' * 64, threads=1, target=corpus().target)
    candidates = view.discover(task=task_fixture(), missing_facet_ids=('f1',), facet_scores=view.cosine_scores([1., 0.]))
    assert candidates and all(view.locators[c.witness_id].chunk_id == 'unit-test-1' for c in candidates)
    def cancelled():
        raise InterruptedError('unit-test cancellation')
    monkeypatch.setattr(storage, 'raise_if_source_io_cancelled', cancelled)
    with pytest.raises(InterruptedError):
        view.discover(task=task_fixture(), missing_facet_ids=('f1',), facet_scores=view.cosine_scores([1., 0.]))
