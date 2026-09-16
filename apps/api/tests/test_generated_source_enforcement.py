from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.retrieval_control_contracts import GroundedAnswerDraft
from app.services.retrieval_models import enforce_generated_source_units
from app.services.agent_reflection import validate_draft_sources, ReflectionContractError


def test_framing_with_sources_keeps_text_and_enforces_binding():
    raw = {'answer_units': [{'kind': 'framing', 'text': 'The documented values follow.', 'source_handles': ['src_1']}]}
    original = deepcopy(raw)
    normalized, audit = enforce_generated_source_units(raw)
    draft = GroundedAnswerDraft.model_validate(normalized)
    assert raw == original and draft.answer_units[0].text == original['answer_units'][0]['text']
    assert draft.answer_units[0].kind == 'factual' and draft.answer_units[0].source_handles == ['src_1']
    validate_draft_sources(draft, ['src_1'], unit_limit=4)
    assert audit['normalized_unit_count'] == 1 and audit['source_handles_removed'] == 0


@pytest.mark.parametrize('kind', ['framing', 'clarification', 'factual'])
def test_model_tag_cannot_bypass_required_source(kind):
    normalized, _ = enforce_generated_source_units({'answer_units': [{'kind': kind, 'text': 'A value is claimed.', 'source_handles': []}]})
    with pytest.raises(ValidationError):
        GroundedAnswerDraft.model_validate(normalized)


@pytest.mark.parametrize('kind', ['unknown', [], {}])
def test_unknown_kind_and_unknown_source_still_fail(kind):
    raw = {'answer_units': [{'kind': kind, 'text': 'A claim.', 'source_handles': ['src_1']}]}
    with pytest.raises(ValidationError):
        GroundedAnswerDraft.model_validate(enforce_generated_source_units(raw)[0])
    raw['answer_units'][0]['kind'] = 'framing'
    draft = GroundedAnswerDraft.model_validate(enforce_generated_source_units(raw)[0])
    with pytest.raises(ReflectionContractError, match='handle_unknown'):
        validate_draft_sources(draft, ['src_2'], unit_limit=4)
