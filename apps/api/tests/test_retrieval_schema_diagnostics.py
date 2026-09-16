import json
import pytest
from pydantic import ValidationError
from app.services.retrieval_models import TaskPlanningOutput, safe_task_schema_errors


def test_task_schema_diagnostics_keep_known_fields_and_exclude_values():
    private = 'unit-test-secret-provider-value'
    with pytest.raises(ValidationError) as caught:
        TaskPlanningOutput.model_validate({'perception': {'intent': private}, private: private})
    result = safe_task_schema_errors(caught.value, TaskPlanningOutput)
    encoded = json.dumps(result)
    assert private not in encoded
    assert any(item['location'] == ['perception', 'intent'] for item in result['field_errors'])
    assert result['provider_values_persisted'] is False
