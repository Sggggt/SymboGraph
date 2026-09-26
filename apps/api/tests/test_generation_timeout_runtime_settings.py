import pytest
from pydantic import ValidationError

from app.core.config import RUNTIME_JSON_SETTINGS, Settings
from app.schemas import ModelSettingsResponse, ModelSettingsUpdate


REQUIRED_RESPONSE_SETTINGS = {
    "rq_kmeans_levels": 3,
    "edge_distance_protocol": "edge_distance_log_calibrated_strength_v2",
    "rq_membership_protocol": "rq_primary_chain_v1",
    "edge_projection_protocol": "membership_q15_layer_type_calibrated_v3",
    "edge_type_calibration_protocol": "type_local_winsorized_minmax_v1",
    "rq_membership_temperature": 0.35,
    "gray_zone_rule_protocol": "deterministic_support_progress_v1",
    "gray_zone_observation_cadence": 1,
    "traversal_observation_budget": 64,
}


def test_planning_timeout_setting_is_retired_from_public_contracts() -> None:
    field = "retrieval_planning_timeout_seconds"
    assert field not in RUNTIME_JSON_SETTINGS
    assert field not in Settings.model_fields
    assert field not in ModelSettingsResponse.model_fields
    assert field not in ModelSettingsUpdate.model_fields
    with pytest.raises(ValidationError):
        ModelSettingsUpdate.model_validate({field: 120})


def test_generation_timeout_accepts_six_hundred_seconds_across_contracts() -> None:
    assert Settings(
        _env_file=None,
        retrieval_generation_timeout_seconds=600,
    ).retrieval_generation_timeout_seconds == 600
    assert ModelSettingsResponse.model_validate(
        {
            **REQUIRED_RESPONSE_SETTINGS,
            "retrieval_generation_timeout_seconds": 600,
        }
    ).retrieval_generation_timeout_seconds == 600
    assert ModelSettingsUpdate(
        retrieval_generation_timeout_seconds=600,
    ).retrieval_generation_timeout_seconds == 600


@pytest.mark.parametrize("value", [601, 3_600])
def test_generation_timeout_rejects_values_above_six_hundred(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_generation_timeout_seconds=value)
    with pytest.raises(ValidationError):
        ModelSettingsResponse.model_validate(
            {
                **REQUIRED_RESPONSE_SETTINGS,
                "retrieval_generation_timeout_seconds": value,
            }
        )
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(retrieval_generation_timeout_seconds=value)
