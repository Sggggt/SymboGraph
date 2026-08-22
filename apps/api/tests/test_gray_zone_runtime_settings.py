from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def _active_settings(**overrides):
    values = {
        "embedding_model": "unit-test-embedding-model",
        "embedding_dimensions": 1024,
        "gray_zone_rule_protocol": "deterministic_support_progress_v1",
        "gray_zone_observation_cadence": 1,
        "agent_path_distance_green_threshold": 0.45,
        "agent_path_distance_gray_threshold": 1.35,
        "agent_path_distance_hard_threshold": 2.4,
        "query_facet_posterior_enabled": True,
        "query_facet_posterior_observation_budget": 64,
        "query_facet_posterior_round_budget": 2,
        "query_facet_posterior_convergence_epsilon": 0.02,
        "model_bridge_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _SettingsReader:
    def __init__(self, settings):
        self.settings = settings
        self.cache_clear_calls = 0

    def __call__(self):
        return self.settings

    def cache_clear(self):
        self.cache_clear_calls += 1


@pytest.mark.parametrize(
    "invalid_protocol",
    [
        "",
        "gpt-5",
        "follow this prompt",
        "deterministic_support_progress_v2",
        None,
        ["deterministic_support_progress_v1"],
    ],
)
def test_gray_zone_protocol_is_allowlisted_at_schema_and_service_before_side_effects(
    monkeypatch,
    invalid_protocol,
):
    from app.schemas import ModelSettingsUpdate
    from app.services import runtime_settings

    with pytest.raises(ValidationError):
        ModelSettingsUpdate(gray_zone_rule_protocol=invalid_protocol)

    original_env = (
        runtime_settings.ENV_PATH.read_bytes()
        if runtime_settings.ENV_PATH.exists()
        else None
    )
    settings_reader = _SettingsReader(_active_settings())
    normalize_calls: list[bool] = []
    env_write_calls: list[dict] = []
    env_apply_calls: list[dict] = []
    publish_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "get_settings", settings_reader)
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: normalize_calls.append(True))
    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: env_write_calls.append(dict(updates)))
    monkeypatch.setattr(runtime_settings, "_apply_runtime_env", lambda updates: env_apply_calls.append(dict(updates)))
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": publish_calls.append(list(changed_keys)),
    )

    with pytest.raises(ValueError, match="allowlisted deterministic local protocol"):
        runtime_settings.update_model_settings({"gray_zone_rule_protocol": invalid_protocol})

    current_env = (
        runtime_settings.ENV_PATH.read_bytes()
        if runtime_settings.ENV_PATH.exists()
        else None
    )
    assert current_env == original_env
    assert normalize_calls == []
    assert env_write_calls == []
    assert env_apply_calls == []
    assert settings_reader.cache_clear_calls == 0
    assert publish_calls == []


def test_gray_zone_protocol_contract_is_hot_reloadable_and_in_payload(monkeypatch):
    from app.core.config import HOT_RELOAD_SETTINGS, Settings, _validate_fixed_protocol_env
    from app.schemas import ModelSettingsResponse, ModelSettingsUpdate
    from app.services import runtime_settings

    settings = Settings(gray_zone_rule_protocol="deterministic_support_progress_v1")
    with pytest.raises(ValidationError):
        Settings(gray_zone_rule_protocol="gpt-5")
    _validate_fixed_protocol_env(
        {
            "GRAY_ZONE_RULE_PROTOCOL": "deterministic_support_progress_v1",
            "GRAY_ZONE_OBSERVATION_CADENCE": "16",
        }
    )
    for invalid_env in (
        {"GRAY_ZONE_RULE_PROTOCOL": "gpt-5"},
        {"GRAY_ZONE_OBSERVATION_CADENCE": "0"},
        {"GRAY_ZONE_OBSERVATION_CADENCE": "17"},
        {"GRAY_ZONE_OBSERVATION_CADENCE": "not-an-integer"},
    ):
        with pytest.raises(ValueError):
            _validate_fixed_protocol_env(invalid_env)
    assert settings.gray_zone_rule_protocol == "deterministic_support_progress_v1"
    assert ModelSettingsUpdate(
        gray_zone_rule_protocol="deterministic_support_progress_v1"
    ).gray_zone_rule_protocol == "deterministic_support_progress_v1"
    assert "gray_zone_rule_protocol" in HOT_RELOAD_SETTINGS
    assert "gray_zone_rule_protocol" in runtime_settings.runtime_lifecycle_payload()["hot_reloadable"]

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {"GRAY_ZONE_RULE_PROTOCOL": "deterministic_support_progress_v1"},
    )
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", lambda **_kwargs: {})
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit")
    payload = runtime_settings.model_settings_payload()
    assert payload["gray_zone_rule_protocol"] == "deterministic_support_progress_v1"
    validated_payload = ModelSettingsResponse.model_validate(payload)
    assert validated_payload.gray_zone_rule_protocol == "deterministic_support_progress_v1"
    assert validated_payload.gray_zone_observation_cadence == 1


@pytest.mark.parametrize("invalid_cadence", [None, 0, 17, True, 1.5, "3"])
def test_gray_zone_observation_cadence_is_bounded_before_side_effects(
    monkeypatch,
    invalid_cadence,
):
    from app.schemas import ModelSettingsUpdate
    from app.services import runtime_settings

    with pytest.raises(ValidationError):
        ModelSettingsUpdate(gray_zone_observation_cadence=invalid_cadence)

    settings_reader = _SettingsReader(_active_settings())
    normalize_calls: list[bool] = []
    env_write_calls: list[dict] = []
    env_apply_calls: list[dict] = []
    publish_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "get_settings", settings_reader)
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: normalize_calls.append(True))
    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: env_write_calls.append(dict(updates)))
    monkeypatch.setattr(runtime_settings, "_apply_runtime_env", lambda updates: env_apply_calls.append(dict(updates)))
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": publish_calls.append(list(changed_keys)),
    )

    with pytest.raises(ValueError, match="must be an integer between 1 and 16"):
        runtime_settings.update_model_settings(
            {"gray_zone_observation_cadence": invalid_cadence}
        )

    assert normalize_calls == []
    assert env_write_calls == []
    assert env_apply_calls == []
    assert settings_reader.cache_clear_calls == 0
    assert publish_calls == []


def test_gray_zone_observation_cadence_hot_reload_contract(monkeypatch):
    from app.core.config import HOT_RELOAD_SETTINGS
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

    result = runtime_settings.update_model_settings({"gray_zone_observation_cadence": 3})

    assert result == {"status": "ok"}
    assert "gray_zone_observation_cadence" in HOT_RELOAD_SETTINGS
    assert "gray_zone_observation_cadence" in runtime_settings.runtime_lifecycle_payload()["hot_reloadable"]
    assert env_write_calls == [{"gray_zone_observation_cadence": 3}]
    assert env_apply_calls == [{"gray_zone_observation_cadence": 3}]
    assert settings_reader.cache_clear_calls == 1
    assert cache_clear_calls == [True]
    assert publish_calls == [["GRAY_ZONE_OBSERVATION_CADENCE"]]


def test_query_facet_posterior_settings_are_hot_reloadable_and_gray_identity_isolated(
    monkeypatch,
):
    from app.core.config import HOT_RELOAD_SETTINGS, Settings
    from app.schemas import ModelSettingsUpdate
    from app.services import cache_manager, context_graph, runtime_settings

    fields = {
        "query_facet_posterior_enabled",
        "query_facet_posterior_observation_budget",
        "query_facet_posterior_round_budget",
        "query_facet_posterior_convergence_epsilon",
    }
    settings = Settings()
    assert settings.query_facet_posterior_enabled is True
    assert settings.query_facet_posterior_observation_budget == 64
    assert settings.query_facet_posterior_round_budget == 2
    assert settings.query_facet_posterior_convergence_epsilon == 0.02
    assert fields.issubset(HOT_RELOAD_SETTINGS)
    assert fields.issubset(
        set(runtime_settings.runtime_lifecycle_payload()["hot_reloadable"])
    )
    for invalid in (
        {"query_facet_posterior_observation_budget": 0},
        {"query_facet_posterior_observation_budget": 1025},
        {"query_facet_posterior_round_budget": 0},
        {"query_facet_posterior_round_budget": 3},
        {"query_facet_posterior_convergence_epsilon": -0.01},
        {"query_facet_posterior_convergence_epsilon": 1.01},
    ):
        with pytest.raises(ValidationError):
            ModelSettingsUpdate(**invalid)

    base = context_graph.agent_operating_envelope(settings)
    changed = {
        **base,
        "query_facet_posterior_observation_budget": 12,
        "query_facet_posterior_round_budget": 1,
        "query_facet_posterior_convergence_epsilon": 0.1,
    }
    assert (
        context_graph.gray_zone_runtime_settings_hash(base)
        == context_graph.gray_zone_runtime_settings_hash(changed)
    )
    assert (
        context_graph.traversal_protocol_hash(base)
        != context_graph.traversal_protocol_hash(changed)
    )

    settings_reader = _SettingsReader(_active_settings())
    env_write_calls: list[dict] = []
    env_apply_calls: list[dict] = []
    cache_clear_calls: list[bool] = []
    publish_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "get_settings", settings_reader)
    monkeypatch.setattr(runtime_settings, "_env_entries", lambda _path: {})
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: False)
    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda updates: env_write_calls.append(dict(updates)),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_apply_runtime_env",
        lambda updates: env_apply_calls.append(dict(updates)),
    )
    monkeypatch.setattr(
        runtime_settings,
        "model_settings_payload",
        lambda: {"status": "ok"},
    )
    monkeypatch.setattr(
        cache_manager,
        "clear_cache_manager",
        lambda: cache_clear_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": publish_calls.append(
            list(changed_keys)
        ),
    )
    patch = {
        "query_facet_posterior_enabled": True,
        "query_facet_posterior_observation_budget": 12,
        "query_facet_posterior_round_budget": 1,
        "query_facet_posterior_convergence_epsilon": 0.1,
    }
    assert runtime_settings.update_model_settings(patch) == {"status": "ok"}
    assert env_write_calls == [patch]
    assert env_apply_calls == [patch]
    assert settings_reader.cache_clear_calls == 1
    assert cache_clear_calls == [True]
    assert publish_calls == [
        [
            "QUERY_FACET_POSTERIOR_ENABLED",
            "QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET",
            "QUERY_FACET_POSTERIOR_ROUND_BUDGET",
            "QUERY_FACET_POSTERIOR_CONVERGENCE_EPSILON",
        ]
    ]


@pytest.mark.parametrize(
    "settings_patch",
    [
        {
            "agent_path_distance_green_threshold": 1.4,
            "agent_path_distance_gray_threshold": 1.3,
            "agent_path_distance_hard_threshold": 2.4,
        },
        {
            "agent_path_distance_green_threshold": 0.45,
            "agent_path_distance_gray_threshold": 2.5,
            "agent_path_distance_hard_threshold": 2.4,
        },
    ],
)
def test_settings_startup_rejects_unordered_path_distance_thresholds(settings_patch):
    from app.core.config import Settings

    with pytest.raises(ValidationError, match="green_threshold <="):
        Settings(**settings_patch)


def test_settings_startup_accepts_equal_path_distance_threshold_boundaries():
    from app.core.config import Settings

    settings = Settings(
        agent_path_distance_green_threshold=0.5,
        agent_path_distance_gray_threshold=0.5,
        agent_path_distance_hard_threshold=0.5,
    )
    assert settings.agent_path_distance_green_threshold == 0.5
    assert settings.agent_path_distance_gray_threshold == 0.5
    assert settings.agent_path_distance_hard_threshold == 0.5


@pytest.mark.parametrize(
    "schema_patch",
    [
        {"agent_path_distance_green_threshold": -0.1},
        {"agent_path_distance_gray_threshold": 20.1},
        {"agent_path_distance_hard_threshold": 40.1},
        {"agent_path_distance_green_threshold": None},
    ],
)
def test_model_settings_update_bounds_path_distance_thresholds(schema_patch):
    from app.schemas import ModelSettingsUpdate

    with pytest.raises(ValidationError):
        ModelSettingsUpdate(**schema_patch)


@pytest.mark.parametrize(
    "invalid_patch",
    [
        {"agent_path_distance_green_threshold": 1.4},
        {"agent_path_distance_gray_threshold": 0.4},
        {"agent_path_distance_gray_threshold": 2.5},
        {"agent_path_distance_hard_threshold": 1.0},
        {
            "agent_path_distance_green_threshold": 1.5,
            "agent_path_distance_gray_threshold": 1.4,
        },
        {"agent_path_distance_green_threshold": None},
        {"agent_path_distance_green_threshold": True},
        {"agent_path_distance_green_threshold": "0.2"},
    ],
)
def test_path_distance_patch_is_validated_against_active_values_before_side_effects(
    monkeypatch,
    invalid_patch,
):
    from app.services import runtime_settings

    original_env = (
        runtime_settings.ENV_PATH.read_bytes()
        if runtime_settings.ENV_PATH.exists()
        else None
    )
    tracked_env_keys = (
        "AGENT_PATH_DISTANCE_GREEN_THRESHOLD",
        "AGENT_PATH_DISTANCE_GRAY_THRESHOLD",
        "AGENT_PATH_DISTANCE_HARD_THRESHOLD",
    )
    original_process_env = {
        key: runtime_settings.os.environ.get(key) for key in tracked_env_keys
    }
    settings_reader = _SettingsReader(_active_settings())
    normalize_calls: list[bool] = []
    env_write_calls: list[dict] = []
    env_apply_calls: list[dict] = []
    publish_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "get_settings", settings_reader)
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: normalize_calls.append(True))
    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: env_write_calls.append(dict(updates)))
    monkeypatch.setattr(runtime_settings, "_apply_runtime_env", lambda updates: env_apply_calls.append(dict(updates)))
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": publish_calls.append(list(changed_keys)),
    )

    with pytest.raises(ValueError, match="path distance thresholds|must be a finite number"):
        runtime_settings.update_model_settings(invalid_patch)

    current_env = (
        runtime_settings.ENV_PATH.read_bytes()
        if runtime_settings.ENV_PATH.exists()
        else None
    )
    assert current_env == original_env
    assert {
        key: runtime_settings.os.environ.get(key) for key in tracked_env_keys
    } == original_process_env
    assert normalize_calls == []
    assert env_write_calls == []
    assert env_apply_calls == []
    assert settings_reader.cache_clear_calls == 0
    assert publish_calls == []


def test_equal_path_distance_threshold_patch_is_applied_and_broadcast(monkeypatch):
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
    patch = {
        "agent_path_distance_green_threshold": 0.5,
        "agent_path_distance_gray_threshold": 0.5,
        "agent_path_distance_hard_threshold": 0.5,
    }

    result = runtime_settings.update_model_settings(patch)

    assert result == {"status": "ok"}
    assert env_write_calls == [patch]
    assert env_apply_calls == [patch]
    assert settings_reader.cache_clear_calls == 1
    assert cache_clear_calls == [True]
    assert publish_calls == [[
        "AGENT_PATH_DISTANCE_GREEN_THRESHOLD",
        "AGENT_PATH_DISTANCE_GRAY_THRESHOLD",
        "AGENT_PATH_DISTANCE_HARD_THRESHOLD",
    ]]
