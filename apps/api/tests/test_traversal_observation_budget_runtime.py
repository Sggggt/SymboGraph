from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError


class _SettingsReader:
    def __init__(self, settings):
        self.settings = settings
        self.cache_clear_calls = 0

    def __call__(self):
        return self.settings

    def cache_clear(self):
        self.cache_clear_calls += 1


def _active_settings():
    return SimpleNamespace(
        embedding_model="unit-test-embedding-model",
        embedding_dimensions=1024,
        agent_path_distance_green_threshold=0.45,
        agent_path_distance_gray_threshold=1.35,
        agent_path_distance_hard_threshold=2.4,
        traversal_observation_budget=64,
        model_bridge_enabled=False,
    )


def test_settings_env_and_api_schema_bound_traversal_observation_budget():
    from app.core.config import HOT_RELOAD_SETTINGS, Settings, _validate_fixed_protocol_env
    from app.schemas import ModelSettingsResponse, ModelSettingsUpdate

    assert Settings().traversal_observation_budget == 64
    assert "traversal_observation_budget" in HOT_RELOAD_SETTINGS
    _validate_fixed_protocol_env({"TRAVERSAL_OBSERVATION_BUDGET": "20000"})
    for invalid_env in ("0", "20001", "not-an-integer"):
        with pytest.raises(ValueError, match="TRAVERSAL_OBSERVATION_BUDGET"):
            _validate_fixed_protocol_env({"TRAVERSAL_OBSERVATION_BUDGET": invalid_env})

    assert ModelSettingsUpdate(
        traversal_observation_budget=64
    ).traversal_observation_budget == 64
    for invalid in (None, 0, 20_001, True, 1.5, "64"):
        with pytest.raises(ValidationError):
            ModelSettingsUpdate(traversal_observation_budget=invalid)

    response = ModelSettingsResponse.model_validate(
        {
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
    )
    assert response.traversal_observation_budget == 64


def test_runtime_payload_and_lifecycle_expose_traversal_observation_budget(monkeypatch):
    from app.core.config import Settings
    from app.services import runtime_settings

    settings = Settings(traversal_observation_budget=64)
    monkeypatch.setattr(runtime_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {"TRAVERSAL_OBSERVATION_BUDGET": "7"},
    )
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", lambda **_kwargs: {})
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit")

    payload = runtime_settings.model_settings_payload()

    assert payload["traversal_observation_budget"] == 7
    assert (
        "traversal_observation_budget"
        in runtime_settings.runtime_lifecycle_payload()["hot_reloadable"]
    )


@pytest.mark.parametrize("invalid", [None, 0, 20_001, True, 1.5, "64"])
def test_runtime_update_rejects_invalid_budget_before_side_effects(monkeypatch, invalid):
    from app.services import runtime_settings

    settings_reader = _SettingsReader(_active_settings())
    normalize_calls: list[bool] = []
    env_write_calls: list[dict] = []
    publish_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "get_settings", settings_reader)
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: normalize_calls.append(True))
    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: env_write_calls.append(dict(updates)))
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": publish_calls.append(list(changed_keys)),
    )

    with pytest.raises(ValueError, match="traversal_observation_budget must be an integer"):
        runtime_settings.update_model_settings({"traversal_observation_budget": invalid})

    assert normalize_calls == []
    assert env_write_calls == []
    assert settings_reader.cache_clear_calls == 0
    assert publish_calls == []


def test_runtime_update_hot_reloads_budget_and_invalidates_cache(monkeypatch):
    from app.services import cache_manager, runtime_settings

    settings_reader = _SettingsReader(_active_settings())
    env_write_calls: list[dict] = []
    env_apply_calls: list[dict] = []
    cache_clear_calls: list[bool] = []
    publish_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "get_settings", settings_reader)
    monkeypatch.setattr(runtime_settings, "_env_entries", lambda _path: {})
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: False)
    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: env_write_calls.append(dict(updates)))
    monkeypatch.setattr(runtime_settings, "_apply_runtime_env", lambda updates: env_apply_calls.append(dict(updates)))
    monkeypatch.setattr(runtime_settings, "model_settings_payload", lambda: {"status": "ok"})
    monkeypatch.setattr(cache_manager, "clear_cache_manager", lambda: cache_clear_calls.append(True))
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": publish_calls.append(list(changed_keys)),
    )

    result = runtime_settings.update_model_settings({"traversal_observation_budget": 7})

    assert result == {"status": "ok"}
    assert env_write_calls == [{"traversal_observation_budget": 7}]
    assert env_apply_calls == [{"traversal_observation_budget": 7}]
    assert settings_reader.cache_clear_calls == 1
    assert cache_clear_calls == [True]
    assert publish_calls == [["TRAVERSAL_OBSERVATION_BUDGET"]]
