from types import SimpleNamespace

from app.retrieval_control_contracts import Requirement
from app.services.source_use import analyze_source_use, source_use_decision


def test_numbered_publication_records_are_metadata_without_blacklisting_numbered_instructions():
    references = ('pdf.\n76. Example A, Sample B (2022) Tensor optimization techniques. Journal of Computing 34: 683–713.\n'
        '77. Reader C, Writer D (2023) Numerical modeling software. Bioinformatics 29: 140–142.')
    analysis = analyze_source_use(references)
    facet = Requirement(id='f',text='Explain the numerical method',weight=1,role='procedure')
    assert analysis.metadata_dominant
    assert not source_use_decision(SimpleNamespace(question=facet.text,requirements=(facet,)),facet,analysis).allowed
    instructions = '1. Create the workspace.\n2. In the 2024 release, run the import method before computing the report.'
    assert not analyze_source_use(instructions).metadata_dominant


def test_biography_continuation_is_not_a_standalone_technical_claim():
    text = ('of computer systems which are designed to tolerate faults. Her research interests include numerical modeling.\n\n'
        'Dr. Sample received the Ph.D. degree and is currently a professor.\n\nPhase 1 Phase 2\n0.23 0.47')
    analysis = analyze_source_use(text)
    assert analysis.metadata_dominant
    assert 'of computer systems' not in ' '.join(analysis.prose)
    facet = Requirement(id='f',text='Describe the capabilities',weight=1)
    assert not source_use_decision(SimpleNamespace(question=facet.text,requirements=(facet,)),facet,analysis).allowed


def test_mixed_real_assertion_before_biography_is_preserved():
    text = 'The method computes the distribution in linear time. Her research interests include numerical modeling.'
    analysis = analyze_source_use(text)
    assert any('method computes the distribution' in part for part in analysis.prose)
