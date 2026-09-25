"""Independent source-scope development matrix; no acceptance documents or answers."""
from types import SimpleNamespace

import pytest

from app.retrieval_control_contracts import (
    EvidenceInterval, Requirement, SourceScopeObligation, SourceScopeRequest,
    SourceScopeSelector, TaskContract, control_hash,
)
from app.services.evidence_scope import (
    StructureScopeIndex,
    _document_alias_matches,
    evaluate_scope_obligation,
    validate_scope_declarations,
)
from app.services.retrieval_corpus import CorpusSource


def request(kind, reference, match='title'):
    return SourceScopeRequest(selector=SourceScopeSelector(kind=kind, reference=reference, match=match))


def obligation(scope, mode='overlap'):
    return SourceScopeObligation(scope=scope, mode=mode)


def task(scope, question='Use the requested source'):
    return TaskContract(knowledge_base_id='unit-kb', conversation_scope_hash='a'*64, question=question,
        requirements=(Requirement(id='f1', text='Describe the configuration', weight=1, source_scope=scope),))


def index_fixture(text, specs, *, cuts=(), version='unit-v1', incomplete=False):
    boundaries = (0, *cuts, len(text))
    sources = tuple(CorpusSource(f'unit-c{i}', 'unit-doc', version, 'Unit manual', text[left:right], left, right,
        control_hash(text[left:right])) for i, (left, right) in enumerate(zip(boundaries, boundaries[1:])))
    corpus = SimpleNamespace(knowledge_base_id='unit-kb', sources=sources, scope_hash=control_hash(text),
        by_id={source.chunk_id: source for source in sources})
    nodes = []
    for i, (kind, title, start, end) in enumerate(specs):
        nodes.append(SimpleNamespace(id=f'unit-n{i}', knowledge_base_id='unit-kb', document_id='unit-doc',
            document_version_id=version, node_type=kind, title=title, char_start=start, char_end=end,
            parent_id='unit-parent', previous_sibling_id=f'unit-n{i-1}' if i else None,
            next_sibling_id=f'unit-n{i+1}' if i+1 < len(specs) else None,
            layout_json={'metadata': {'incomplete': incomplete}}))
    return StructureScopeIndex(corpus=corpus, nodes=nodes)


def evaluate(index, fixed, spans):
    bound = index.bind(fixed)[0]
    intervals = tuple(EvidenceInterval(knowledge_base_id='unit-kb', document_version_id='unit-v1', start=start, end=end)
                      for start, end in spans)
    return evaluate_scope_obligation(task=fixed, facet=fixed.requirements[0], bound=bound, packed=intervals)


@pytest.mark.parametrize('kind,node_type', [('text','paragraph'), ('section','section'), ('table','table'),
    ('formula','formula'), ('code','code_block'), ('caption','caption')])
def test_native_textual_locations_share_whole_scope_and_chunk_boundary_rules(kind, node_type):
    raw = 'Alpha object\nvalue = 47\nnext value = 89'
    index = index_fixture(raw, [(node_type,'Alpha object',0,len(raw))], cuts=(11,23))
    fixed = task(obligation(request(kind,'Alpha object'),'complete'), 'Read all of Alpha object')
    assert evaluate(index,fixed,[(0,11),(11,23),(23,len(raw))]).state == 'satisfied'
    partial = evaluate(index,fixed,[(0,23)])
    assert partial.state == 'unsatisfied'
    assert [(part.start,part.end) for part in partial.coverage[0].missing_intervals] == [(23,len(raw))]


def test_document_version_and_foreign_scope_never_supply_missing_content():
    raw = 'Unit manual\nconfiguration'
    index = index_fixture(raw,[('document','Unit manual',0,len(raw))])
    fixed = task(obligation(request('document','Unit manual'),'complete'),'Read Unit manual')
    bound = index.bind(fixed)[0]
    other = EvidenceInterval(knowledge_base_id='unit-kb',document_version_id='unit-v2',start=0,end=len(raw))
    result = evaluate_scope_obligation(task=fixed,facet=fixed.requirements[0],bound=bound,packed=(other,))
    assert result.state == 'unsatisfied'
    with pytest.raises(ValueError,match='cross_knowledge_base'):
        evaluate_scope_obligation(task=fixed,facet=fixed.requirements[0],bound=bound,
            packed=(other.model_copy(update={'knowledge_base_id':'foreign-kb'}),))


def test_document_display_and_file_titles_share_a_bounded_metadata_alias_match():
    assert _document_alias_matches(
        "WFI Calibration Touchstone Field Recommendations",
        "Roman_WFI_Touchstone_Fields_2025",
    )
    assert _document_alias_matches("Roman", "Roman_WFI_Touchstone_Fields_2025")
    assert not _document_alias_matches("report", "Roman_ROTAC_Final_Report_2025")
    assert not _document_alias_matches(
        "Unrelated Calibration Recommendations",
        "Roman_WFI_Touchstone_Fields_2025",
    )


def test_document_title_selector_can_resolve_a_collection_of_matching_documents():
    sources = (
        CorpusSource("unit-c1", "unit-d1", "unit-v1", "Roman Survey A", "alpha", 0, 5, control_hash("alpha")),
        CorpusSource("unit-c2", "unit-d2", "unit-v2", "Roman Survey B", "bravo", 0, 5, control_hash("bravo")),
    )
    corpus = SimpleNamespace(
        knowledge_base_id="unit-kb",
        sources=sources,
        scope_hash=control_hash("roman-collection"),
        by_id={source.chunk_id: source for source in sources},
    )
    nodes = tuple(
        SimpleNamespace(
            id=f"unit-n{index}",
            knowledge_base_id="unit-kb",
            document_id=source.document_id,
            document_version_id=source.document_version_id,
            node_type="document",
            title=source.title,
            char_start=0,
            char_end=5,
            parent_id=None,
            previous_sibling_id=None,
            next_sibling_id=None,
            layout_json={},
        )
        for index, source in enumerate(sources)
    )
    binding = StructureScopeIndex(corpus=corpus, nodes=nodes).resolve(
        request("document", "Roman")
    )
    assert binding.reason == "resolved"
    assert len(binding.fact.intervals) == 2


@pytest.mark.parametrize('kind,node_type,label', [('table','table','Table 7'),('formula','formula','Equation 7'),
    ('code','code_block','Algorithm 7')])
def test_object_label_requires_declaration_not_body_mention(kind,node_type,label):
    caption = label + ': Configuration\n'
    raw = caption + 'setting value\nA 47\n'
    index = index_fixture(raw,[('caption',caption.rstrip(),0,len(caption)),(node_type,'Object',len(caption),len(raw))])
    fixed = task(obligation(request(kind,label,'label'),'complete'),'Read '+label)
    bound = index.bind(fixed)[0]
    assert bound.bindings[0].node_ids == ('unit-n0','unit-n1')
    assert evaluate(index,fixed,[(len(caption),len(raw))]).state == 'satisfied'
    assert evaluate(index,fixed,[(0,len(caption))]).state == 'unsatisfied'
    index.nodes[1].previous_sibling_id = None
    assert evaluate(index,fixed,[(0,len(raw))]).state == 'unknown'


def test_duplicate_labels_need_parent_intersection_and_body_reference_cannot_bind():
    raw = 'First section\nTable 2 Item\nvalue 11\nSecond section\nTable 2 Item\nvalue 99\n'
    middle = raw.index('Second')
    first, second = raw.index('Table 2'),raw.rindex('Table 2')
    index = index_fixture(raw,[('section','First section',0,middle),('table','Table 2 Item',first,middle),
        ('section','Second section',middle,len(raw)),('table','Table 2 Item',second,len(raw))])
    ambiguous = task(obligation(request('table','2','label')),'Read 2')
    assert evaluate(index,ambiguous,[(0,len(raw))]).state == 'unknown'
    scope = SourceScopeRequest(op='intersection',children=(request('section','First section'),request('table','2','label')))
    fixed = task(obligation(scope,'complete'),'Read 2 in First section')
    assert evaluate(index,fixed,[(first,middle)]).state == 'satisfied'
    assert evaluate(index,fixed,[(second,len(raw))]).state == 'unsatisfied'


def test_any_and_all_keep_separate_obligations_and_never_treat_overlap_as_complete():
    raw = 'First area\nsetting 10\nSecond area\nsetting 20'
    split = raw.index('Second')
    index = index_fixture(raw,[('section','First area',0,split),('section','Second area',split,len(raw))])
    children = tuple(obligation(request('section',title),'complete') for title in ('First area','Second area'))
    for op,expected in [('any','satisfied'),('all','unsatisfied')]:
        fixed = task(SourceScopeObligation(op=op,children=children),'Use First area and Second area')
        assert evaluate(index,fixed,[(0,split)]).state == expected


@pytest.mark.parametrize('reason', ['incomplete','figure','missing'])
def test_unresolved_representation_stays_unknown_even_when_document_is_packed(reason):
    raw = 'Alpha object\nsetting 14'
    kind = 'figure' if reason == 'figure' else 'table'
    index = index_fixture(raw,[(kind,'Alpha object',0,len(raw))], incomplete=reason=='incomplete')
    reference = 'Missing object' if reason=='missing' else 'Alpha object'
    fixed = task(obligation(request(kind,reference),'complete'),'Read '+reference)
    assert evaluate(index,fixed,[(0,len(raw))]).state == 'unknown'


def test_known_readable_part_can_satisfy_overlap_but_not_whole_document():
    raw = 'Alpha object\nsetting 14'
    index = index_fixture(raw,[('document','Alpha object',0,len(raw))],incomplete=True)
    for mode, expected in [('overlap','satisfied'),('complete','unknown')]:
        fixed = task(obligation(request('document','Alpha object'),mode),'Read Alpha object')
        assert evaluate(index,fixed,[(0,len(raw))]).state == expected


def test_large_resolved_document_scope_is_bounded_by_corpus_limit_not_256():
    from app.retrieval_control_contracts import EvidenceScopeFact

    intervals = tuple(
        EvidenceInterval(
            knowledge_base_id="unit-kb",
            document_version_id="unit-v1",
            start=index,
            end=index + 1,
        )
        for index in range(1205)
    )
    fact = EvidenceScopeFact(
        id="large-document",
        knowledge_base_id="unit-kb",
        kind="document",
        resolution="verified",
        extent_complete=True,
        intervals=intervals,
        witness_ids=("document-node",),
    )
    assert len(fact.intervals) == 1205


def test_declarations_cannot_invent_user_scope_and_empty_legacy_field_keeps_hash():
    with pytest.raises(ValueError,match='must_quote_current_question'):
        task(obligation(request('section','Invented scope')))
    with pytest.raises(ValueError,match='kind_must_be_explicit'):
        validate_scope_declarations(obligation(request('table','the duration','kind')))
    fixed = task(None)
    assert 'source_scope' not in fixed.model_dump(mode='json')['requirements'][0]
    assert TaskContract.model_validate(fixed.model_dump(mode='json')).identity == fixed.identity


def test_conflicting_overlaps_and_cancellation_fail_before_binding(monkeypatch):
    index = index_fixture('Alpha object',[('section','Alpha object',0,12)])
    original = index.corpus.sources[0]
    index.corpus.sources = (original, CorpusSource('second','unit-doc','unit-v1','Unit manual','Wrong',3,8,'x'))
    with pytest.raises(ValueError,match='overlapping_text_disagrees'):
        StructureScopeIndex(corpus=index.corpus,nodes=index.nodes)
    def cancel():
        raise InterruptedError('cancelled')
    monkeypatch.setattr('app.services.evidence_scope.raise_if_source_io_cancelled',cancel)
    with pytest.raises(InterruptedError):
        index_fixture('Alpha object',[('section','Alpha object',0,12)])


def test_scope_target_cover_uses_interval_planner_and_respects_execution_budget():
    from app.services.evidence_scope import scope_target_plan
    raw = 'Table 8: settings\n' + 'setting 47\n' * 20
    index = index_fixture(raw,[('table','Table 8: settings',0,len(raw))],cuts=(70,150))
    fixed = task(obligation(request('table','Table 8','label'),'complete'),'Read all of Table 8')
    ids, audit = scope_target_plan(index=index,task=fixed,token_budget=1000,target_limit=4)
    assert set(ids) == set(index.corpus.by_id)
    assert audit['interval_plans'][0]['status'] == 'ready'
    assert audit['executor_validation_required'] and audit['model_call_count'] == 0
    ids, audit = scope_target_plan(index=index,task=fixed,token_budget=1000,target_limit=2)
    assert ids == () and audit['status'] == 'over_budget'


def test_scope_targets_keep_version_boundaries_and_share_chunk_costs(monkeypatch):
    from app.services import chunking
    from app.services.evidence_scope import scope_target_plan

    sources = (
        CorpusSource('unit-target', 'unit-doc-a', 'unit-v1', 'Atlas manual', 'alpha', 0, 5, control_hash('alpha')),
        CorpusSource('unit-other', 'unit-doc-b', 'unit-v2', 'Birch handbook', 'bravo', 0, 5, control_hash('bravo')),
    )
    corpus = SimpleNamespace(knowledge_base_id='unit-kb', sources=sources,
        scope_hash=control_hash('two-public-guides'), by_id={source.chunk_id: source for source in sources})
    nodes = tuple(SimpleNamespace(id=f'unit-node-{index}', knowledge_base_id='unit-kb',
        document_id=source.document_id, document_version_id=source.document_version_id,
        node_type='document', title=source.title, char_start=0, char_end=5,
        parent_id=None, previous_sibling_id=None, next_sibling_id=None, layout_json={})
        for index, source in enumerate(sources))
    index = StructureScopeIndex(corpus=corpus, nodes=nodes)
    target_scope = obligation(request('document', 'Atlas manual'))
    fixed = TaskContract(knowledge_base_id='unit-kb', conversation_scope_hash='a'*64,
        question='Summarize Atlas manual twice for two independent duties.',
        requirements=(Requirement(id='f1', text='First duty', weight=.5, source_scope=target_scope),
                      Requirement(id='f2', text='Second duty', weight=.5, source_scope=target_scope)))
    calls = []
    original = chunking.rough_token_count
    def counted(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(chunking, 'rough_token_count', counted)
    ids, audit = scope_target_plan(index=index, task=fixed, token_budget=1000, target_limit=2)
    assert ids == ('unit-target',)
    assert audit['status'] == 'proposed'
    assert len(calls) == len(sources)


def test_cell_number_and_wrong_caption_type_are_not_object_declarations():
    raw = 'Equation 3: unrelated\n3 17 28\n'
    split = raw.index('3 17')
    index = index_fixture(raw,[('caption','Equation 3',0,split),('table','Table',split,len(raw))])
    fixed = task(obligation(request('table','Table 3','label')),'Read Table 3')
    assert evaluate(index,fixed,[(0,len(raw))]).state == 'unknown'


def test_document_identity_can_use_declared_opening_title_and_not_infix_names():
    raw = 'Alpha Reference Manual\nConfiguration description'
    index = index_fixture(raw,[('document','unit-archive-2025',0,len(raw)),
        ('section','Alpha Reference Manual',0,len(raw))])
    index.nodes[1].parent_id = index.nodes[0].id
    index = StructureScopeIndex(corpus=index.corpus,nodes=index.nodes)
    binding = index.resolve(request('document','Alpha Reference Manual'))
    assert binding.reason == 'resolved' and len(binding.node_ids)==2
    assert index.resolve(request('document','lpha Reference')).reason == 'no_verified_match'


def test_document_identity_can_use_a_nested_declared_bundle_heading():
    raw = 'Bundle index\nStep one\n'
    start = raw.index('Step one')
    index = index_fixture(
        raw,
        [
            ('document', 'training-bundle-2026', 0, len(raw)),
            ('section', 'Data Type Extension Tool Guide', start, len(raw)),
        ],
    )
    index.nodes[1].parent_id = index.nodes[0].id
    index = StructureScopeIndex(corpus=index.corpus, nodes=index.nodes)
    binding = index.resolve(
        request('document', 'Data Type Extension Tool Guide')
    )
    assert binding.reason == 'resolved'
    assert binding.node_ids == ('unit-n0', 'unit-n1')
