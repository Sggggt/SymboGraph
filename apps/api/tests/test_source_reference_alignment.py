"""Domain-independent naming correspondence; no private corpus fixtures."""
import json
import random

import pytest

from app.services.source_reference_alignment import align_reference


@pytest.mark.parametrize('reference,candidate,expanded', [
    ('TLS deployment guide', 'Transport Layer Security deployment guide', True),
    ('Remote Procedure Call settings', 'RPC settings', True),
    ('ABC measurement protocol', 'Alpha Beta Channel measurement protocol', True),
    ('CPU power limits', 'Central Processing Unit power limits', True),
    ('Public application programming interface guide', 'Public API guide', True),
    ('ＴＬＳ configuration', 'Transport Layer Security configuration', True),
    ('Index catalogue', 'INDEX_CATALOGUE', False),
    ('配置指南', '配置指南', False),
    ('Version v1.2', 'Version v1.2', False),
])
def test_complete_correspondence_has_raw_spans_without_semantic_authority(reference,candidate,expanded):
    result = align_reference(reference,candidate)
    assert result.status == 'aligned'
    assert result.semantic_identity_proven is False
    assert bool(result.initialisms) is expanded
    assert candidate[result.candidate_span[0]:result.candidate_span[1]]
    for mapping in result.initialisms:
        assert mapping.reference_text == reference[slice(*mapping.reference_span)]
        assert mapping.candidate_text == candidate[slice(*mapping.candidate_span)]


@pytest.mark.parametrize('reference,candidate', [
    ('TLS guide 2025','Transport Layer Security guide 2026'),
    ('TLS private guide','Transport Layer Security guide'),
    ('TLS guide','Transport secure Layer Security guide'),
    ('TLS guide','Transport Layer Security summary guide'),
    ('tls guide','Transport Layer Security guide'),
    ('Version 1.2','Version 1 2'),
    ('Temperature -5','Temperature 5'),
    ('Version v1.2','Version v1 2'),
    ('Net interface','Network interface'),
])
def test_complete_match_cannot_drop_qualifiers_digits_signs_or_internal_words(reference,candidate):
    assert align_reference(reference,candidate).status == 'no_alignment'


def test_least_abbreviation_then_shortest_span_then_earliest_offset():
    text = 'Transport Layer Security guide. TLS guide. TLS guide'
    result = align_reference('TLS guide',text)
    assert result.initialisms == ()
    assert result.candidate_span == (32,41)
    short = 'A very long B term C. A B C'
    # Internal words cannot be skipped while expanding initials.
    result = align_reference('ABC',short)
    assert short[slice(*result.candidate_span)] == 'A B C'


def test_capacity_is_not_reported_as_a_negative_match():
    assert align_reference('word '*129,'word').status == 'not_evaluated'
    assert align_reference('word','word '*41).status == 'not_evaluated'
    assert align_reference('', 'plain title').status == 'no_alignment'


def test_multiple_surface_paths_use_one_deterministic_result():
    reference = 'API RPC configuration'
    candidate = 'Application Programming Interface Remote Procedure Call configuration'
    result = align_reference(reference,candidate)
    assert result.status == 'aligned' and len(result.initialisms) == 2
    assert result == align_reference(reference,candidate)


def test_location_packet_protocol_is_optional_and_does_not_grant_scope():
    from test_source_location import fixture
    from app.services.source_location import build_location_request,location_packet
    from app.retrieval_control_contracts import SourceLocationRequest,control_hash
    fixed,index=fixture('TLS guidance','Transport Layer Security guidance')
    request=build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    packet=location_packet(fixed,request)
    assert packet['reference_alignment']['semantic_identity_proven'] is False
    assert packet['reference_alignment']['correspondences']
    assert all(row['semantic_identity_proven'] is False for row in packet['reference_alignment']['correspondences'])
    assert index.semantic_selection is None
    legacy=request.model_dump(mode='json')
    legacy.pop('reference_alignment_protocol')
    old=SourceLocationRequest.model_validate(legacy)
    old_packet=location_packet(fixed,old)
    assert 'reference_alignment_protocol' not in old.model_dump(mode='json')
    assert 'reference_alignment' not in old_packet
    assert control_hash(old_packet) != control_hash(packet)
    assert json.loads(json.dumps(packet,ensure_ascii=False)) == packet


def test_dynamic_program_matches_bounded_exhaustive_paths():
    # The oracle enumerates only tiny test lattices; production never enumerates.
    randomizer = random.Random(418)
    vocabulary = ('Alpha', 'Beta', 'Gamma', 'AB', 'BG', 'guide', '2028')
    for _ in range(80):
        left = [randomizer.choice(vocabulary) for _ in range(randomizer.randint(1,3))]
        right = [randomizer.choice(vocabulary) for _ in range(randomizer.randint(1,6))]
        if randomizer.randrange(2):
            right = list(left)
            right = [part for word in right for part in
                     (('Alpha','Beta') if word == 'AB' else ('Beta','Gamma') if word == 'BG' else (word,))]
        reference, candidate = ' '.join(left), ' '.join(right)
        starts = [sum(len(word)+1 for word in right[:j]) for j in range(len(right))]
        paths = []
        def visit(i,j,start,cost):
            if i == len(left):
                end = starts[j-1] + len(right[j-1])
                paths.append((cost,end-starts[start],starts[start],end))
                return
            if j == len(right): return
            if left[i].casefold() == right[j].casefold(): visit(i+1,j+1,start,cost)
            for word, tokens, offset, forward in ((left[i],right,j,True),(right[j],left,i,False)):
                if not word.isascii() or not word.isalpha() or not word.isupper() or not 2<=len(word)<=12: continue
                stop=offset+len(word)
                if stop<=len(tokens) and ''.join(part[0].casefold() for part in tokens[offset:stop])==word.casefold():
                    visit(i+1,stop,start,cost+1) if forward else visit(stop,j+1,start,cost+1)
        for start in range(len(right)): visit(0,start,start,0)
        result=align_reference(reference,candidate)
        assert (result.status=='aligned') == bool(paths)
        if paths:
            best=min(paths)
            assert len(result.initialisms)==best[0] and result.candidate_span==best[2:]


def test_old_selection_replay_and_alignment_protocol_tampering():
    from test_source_location import fixture,completed
    from app.services.source_location import build_location_request,apply_scope_selection
    from app.retrieval_control_contracts import control_hash
    fixed,index=fixture()
    request=build_location_request(index=index,task=fixed,query_scores=[.8,.3]).model_copy(
        update={'reference_alignment_protocol':None})
    payload,selection=completed(index,fixed,request)
    apply_scope_selection(index=index,task=fixed,selection=selection,payload=payload)
    fixed,other=fixture()
    payload['request']['reference_alignment_protocol']='source_reference_alignment_v1'
    payload['audit_hash']=control_hash({k:v for k,v in payload.items() if k!='audit_hash'})
    selection=selection.model_copy(update={'ledger_hash':payload['audit_hash']})
    with pytest.raises(ValueError,match='source_location_input_changed'):
        apply_scope_selection(index=other,task=fixed,selection=selection,payload=payload)


def test_optional_correspondences_never_displace_original_candidate_cards():
    from test_evidence_scope import index_fixture,request,obligation
    from app.retrieval_control_contracts import TaskContract,Requirement
    from app.services.evidence_scope import StructureScopeIndex
    from app.services.source_location import build_location_request,location_packet
    names=['A B '*25+str(i) for i in range(8)]
    references=['AB '*25+str(i) for i in range(8)]
    pieces=[name+'\n' for name in names]
    starts=[sum(map(len,pieces[:i])) for i in range(9)]
    base=index_fixture(''.join(pieces),[('section',name,starts[i],starts[i+1]) for i,name in enumerate(names)],cuts=tuple(starts[1:-1]))
    fixed=TaskContract(knowledge_base_id='unit-kb',conversation_scope_hash='a'*64,question='; '.join(references),
        requirements=tuple(Requirement(id=f'f{i}',text=f'Locate configuration {i}',weight=1/8,
            source_scope=obligation(request('section',reference))) for i,reference in enumerate(references)))
    index=StructureScopeIndex(corpus=base.corpus,nodes=base.nodes,task_hash=fixed.identity)
    current=build_location_request(index=index,task=fixed,query_scores=[.5]*8)
    original=location_packet(fixed,current.model_copy(update={'reference_alignment_protocol':None}))
    projected=location_packet(fixed,current)
    assert projected['reference_alignment']['omitted_for_capacity'] is True
    assert len(json.dumps(projected,ensure_ascii=False))<=48000
    assert {key:value for key,value in projected.items() if key!='reference_alignment'}==original


def test_alignment_packet_observes_source_cancellation(monkeypatch):
    from test_source_location import fixture
    from app.services import source_location
    fixed,index=fixture()
    request=source_location.build_location_request(index=index,task=fixed,query_scores=[.8,.3])
    def cancel(): raise RuntimeError('unit-test-cancelled')
    monkeypatch.setattr(source_location,'raise_if_source_io_cancelled',cancel)
    with pytest.raises(RuntimeError,match='unit-test-cancelled'):
        source_location.location_packet(fixed,request)
