from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas import ModelSettingsUpdate
from app.services.retrieval_sufficiency import sufficiency_model_limits


@pytest.mark.parametrize('field,value',[('retrieval_sufficiency_timeout_seconds',0),('retrieval_sufficiency_timeout_seconds',241),
    ('retrieval_sufficiency_timeout_seconds',True),('retrieval_sufficiency_timeout_seconds','120'),
    ('retrieval_sufficiency_max_tokens',255),('retrieval_sufficiency_max_tokens',32769),
    ('retrieval_sufficiency_max_tokens',True),('retrieval_sufficiency_max_tokens','16384')])
def test_api_rejects_invalid_or_coerced_evaluation_limits(field,value):
    with pytest.raises(ValidationError): ModelSettingsUpdate(**{field:value})


def test_evaluation_keeps_provider_caps_and_does_not_borrow_repair_budget():
    settings=SimpleNamespace(model_request_timeout_seconds=90,chat_json_max_tokens=12000,
        retrieval_sufficiency_timeout_seconds=120,retrieval_sufficiency_max_tokens=16384,
        retrieval_repair_timeout_seconds=5,retrieval_repair_max_tokens=256)
    assert sufficiency_model_limits(settings)==(90,12000)
    settings.model_request_timeout_seconds=240
    settings.chat_json_max_tokens=24000
    assert sufficiency_model_limits(settings)==(120,16384)
