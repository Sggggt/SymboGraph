from app.retrieval_control_contracts import Requirement
from app.services.retrieval_constraints import formal_literal, formal_literals_match
from test_retrieval_corpus import corpus
from test_retrieval_path_features import task_fixture


def test_natural_title_and_alternative_words_are_not_literal_identifiers():
    assert not formal_literal('Public Calibration Field Recommendations')
    assert not formal_literal('nominal') and not formal_literal('overguide')
    for value in ('HLWAS', '2035', 'L2', 'event_id'):
        assert formal_literal(value)
    requirement = Requirement(id='f1', text='calibration report', weight=1,
                              protected_literals=('Public Calibration Field Recommendations',))
    assert formal_literals_match(requirement, 'Calibration recommendations', 'Relevant source text')


def test_missing_required_year_does_not_produce_unrelated_repair_candidates():
    view = corpus()
    task = task_fixture()
    task = task.model_copy(update={'requirements': (task.requirements[0].model_copy(
        update={'protected_literals': ('2035',)}),)})
    assert view.discover(task=task, missing_facet_ids=('f1',), facet_scores=view.cosine_scores([0, 1])) == ()
    assert view.literal_lookup('2035')['corpus_fact_absence_proven'] is False
