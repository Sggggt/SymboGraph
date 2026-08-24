from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError


PROTOCOL_SETTINGS = {
    "edge_distance_protocol": "edge_distance_log_calibrated_strength_v2",
    "rq_membership_protocol": "rq_primary_chain_v1",
    "edge_projection_protocol": "membership_q15_layer_type_calibrated_v3",
    "edge_type_calibration_protocol": "type_local_winsorized_minmax_v1",
}

RQ_MEMBERSHIP_SETTINGS = {
    "rq_membership_temperature": 0.35,
}


def _active_settings(**overrides):
    values = {**PROTOCOL_SETTINGS, **RQ_MEMBERSHIP_SETTINGS, **overrides}
    return SimpleNamespace(**values)


def test_settings_use_closed_local_protocol_literals_and_bounded_rq_parameters():
    from app.core.config import HOT_RELOAD_SETTINGS, Settings

    settings = Settings(_env_file=None)
    for key, expected in PROTOCOL_SETTINGS.items():
        assert getattr(settings, key) == expected
        assert key not in HOT_RELOAD_SETTINGS
    for key, expected in RQ_MEMBERSHIP_SETTINGS.items():
        assert getattr(settings, key) == expected
        assert key not in HOT_RELOAD_SETTINGS

    with pytest.raises(ValidationError):
        Settings(_env_file=None, rq_membership_protocol="llm_prompt_decides_v9")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rq_membership_temperature=0.0)


def test_workspace_hot_reload_preserves_numeric_rq_and_source_io_types():
    from app.core.config import Settings, _apply_hot_reload_env

    settings = Settings(_env_file=None)
    _apply_hot_reload_env(
        settings,
        {
            "RQ_MEMBERSHIP_TEMPERATURE": "0.41",
            "SOURCE_IO_CONCURRENCY": "7",
        },
    )

    assert type(settings.rq_membership_temperature) is float
    assert settings.rq_membership_temperature == 0.41
    assert type(settings.source_io_concurrency) is int
    assert settings.source_io_concurrency == 7


def test_source_io_runtime_contract_is_strict_and_bounded():
    from app.schemas import ModelSettingsResponse, ModelSettingsUpdate

    required_response = {
        "rq_kmeans_levels": 3,
        **PROTOCOL_SETTINGS,
        **RQ_MEMBERSHIP_SETTINGS,
        "gray_zone_rule_protocol": "deterministic_support_progress_v1",
        "gray_zone_observation_cadence": 1,
        "traversal_observation_budget": 64,
    }
    response = ModelSettingsResponse.model_validate(
        {
            **required_response,
            "source_io_concurrency": 7,
        }
    )
    assert type(response.source_io_concurrency) is int
    assert response.source_io_concurrency == 7

    for invalid in ("7", True, 0, 65):
        with pytest.raises(ValidationError):
            ModelSettingsResponse.model_validate(
                {
                    **required_response,
                    "source_io_concurrency": invalid,
                }
            )
        with pytest.raises(ValidationError):
            ModelSettingsUpdate.model_validate(
                {"source_io_concurrency": invalid}
            )


@pytest.mark.parametrize(
    ("env_key", "invalid_value"),
    [
        ("EDGE_DISTANCE_PROTOCOL", "free_expression_v1"),
        ("RQ_MEMBERSHIP_PROTOCOL", "llm_prompt_membership_v1"),
        ("EDGE_PROJECTION_PROTOCOL", "model_projection_v1"),
        ("EDGE_TYPE_CALIBRATION_PROTOCOL", "dynamic_eval_v1"),
    ],
)
def test_workspace_env_rejects_non_allowlisted_graph_protocols(env_key, invalid_value):
    from app.core.config import _validate_fixed_protocol_env

    with pytest.raises(ValueError, match="locally allowlisted graph protocol"):
        _validate_fixed_protocol_env({env_key: invalid_value})


def test_api_and_shared_contract_expose_rebuild_protocol_identity():
    from app.schemas import ModelSettingsResponse, ModelSettingsUpdate

    response = ModelSettingsResponse.model_validate(
        {
            "rq_kmeans_levels": 3,
            **PROTOCOL_SETTINGS,
            **RQ_MEMBERSHIP_SETTINGS,
            "gray_zone_rule_protocol": "deterministic_support_progress_v1",
            "gray_zone_observation_cadence": 1,
            "traversal_observation_budget": 64,
        }
    )
    assert response.rq_membership_protocol == PROTOCOL_SETTINGS["rq_membership_protocol"]
    assert response.rq_membership_temperature == 0.35

    update = ModelSettingsUpdate.model_validate(
        {**PROTOCOL_SETTINGS, **RQ_MEMBERSHIP_SETTINGS}
    )
    assert update.model_dump(exclude_unset=True) == {
        **PROTOCOL_SETTINGS,
        **RQ_MEMBERSHIP_SETTINGS,
    }
    with pytest.raises(ValidationError):
        ModelSettingsUpdate.model_validate({"edge_projection_protocol": "prompt: choose projection"})


def test_runtime_payload_classifies_protocol_and_membership_settings_rebuild_required():
    from app.services.runtime_settings import model_settings_payload

    payload = model_settings_payload(include_dynamic_status=False)
    rebuild_required = set(payload["lifecycle"]["rebuild_required"])
    hot_reloadable = set(payload["lifecycle"]["hot_reloadable"])
    expected_rebuild = set(PROTOCOL_SETTINGS) | set(RQ_MEMBERSHIP_SETTINGS)

    assert expected_rebuild <= rebuild_required
    assert not expected_rebuild & hot_reloadable
    for key, expected in {**PROTOCOL_SETTINGS, **RQ_MEMBERSHIP_SETTINGS}.items():
        assert payload[key] == expected


def test_graph_protocol_admission_is_deterministic_local_and_hashes_rq_parameters():
    from app.services.graph_protocols import validate_active_graph_protocol_settings

    first = validate_active_graph_protocol_settings(_active_settings())
    second = validate_active_graph_protocol_settings(_active_settings())
    changed = validate_active_graph_protocol_settings(
        _active_settings(rq_membership_temperature=0.4)
    )

    assert first == second
    assert len(first["identity_hash"]) == 64
    assert first["identity_hash"] != changed["identity_hash"]
    assert first["dynamic_language_inputs"] == {
        "llm": False,
        "prompt": False,
        "free_expression": False,
    }
    assert first["rq_membership_parameters"]["membership_score_floor"] is None
    assert first["rq_membership_parameters"]["renormalize_after_primary_selection"] is False


def test_context_graph_protocol_constants_and_cache_identity_share_config_authority():
    from app.core.config import (
        EDGE_DISTANCE_PROTOCOL_DEFAULT,
        EDGE_PROJECTION_PROTOCOL_DEFAULT,
        EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT,
        RQ_MEMBERSHIP_PROTOCOL_DEFAULT,
    )
    from app.schemas import SearchFilters
    from app.services.context_graph import (
        EDGE_DISTANCE_PROTOCOL_VERSION,
        EDGE_PROJECTION_PROTOCOL_VERSION,
        EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION,
        RQ_MEMBERSHIP_PROTOCOL_VERSION,
        context_graph_cache_key_components,
        rq_membership_protocol_hash,
    )
    from app.services.graph_protocols import graph_protocol_runtime_identity_hash

    assert EDGE_DISTANCE_PROTOCOL_VERSION == EDGE_DISTANCE_PROTOCOL_DEFAULT
    assert EDGE_PROJECTION_PROTOCOL_VERSION == EDGE_PROJECTION_PROTOCOL_DEFAULT
    assert EDGE_TYPE_CALIBRATION_PROTOCOL_VERSION == EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT
    assert RQ_MEMBERSHIP_PROTOCOL_VERSION == RQ_MEMBERSHIP_PROTOCOL_DEFAULT

    components = context_graph_cache_key_components(
        knowledge_base_id="kb-protocol",
        query="protocol identity",
        filters=SearchFilters(),
        context_state=None,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash="a" * 64,
    )
    assert components["graph_protocol_runtime_identity_hash"] == graph_protocol_runtime_identity_hash()
    assert components["rq_membership_protocol_hash"] == rq_membership_protocol_hash()


def test_edge_protocol_hashes_fail_closed_when_runtime_setting_drifts(monkeypatch):
    from app.services import graph_protocols
    from app.services.context_graph import edge_distance_protocol_hash
    from app.services.graph_protocols import GraphProtocolAdmissionError

    monkeypatch.setattr(
        graph_protocols,
        "get_settings",
        lambda: _active_settings(edge_distance_protocol="legacy_distance_v0"),
    )

    with pytest.raises(GraphProtocolAdmissionError):
        edge_distance_protocol_hash()


def test_rq_prefix_state_facts_do_not_depend_on_database_generated_ids():
    from app.services.context_graph import canonical_rq_prefix_facts, stable_hash

    first_root = SimpleNamespace(
        id="uuid-root-a",
        rq_prefix_key="rq:L1:1",
        rq_level=1,
        rq_path_prefix=[1],
        parent_rq_prefix_id=None,
        codebook_version="residual_quantized_kmeans_primary_v3",
    )
    first_child = SimpleNamespace(
        id="uuid-child-a",
        rq_prefix_key="rq:L2:1-2",
        rq_level=2,
        rq_path_prefix=[1, 2],
        parent_rq_prefix_id=first_root.id,
        codebook_version="residual_quantized_kmeans_primary_v3",
    )
    second_root = SimpleNamespace(**{**first_root.__dict__, "id": "uuid-root-b"})
    second_child = SimpleNamespace(
        **{
            **first_child.__dict__,
            "id": "uuid-child-b",
            "parent_rq_prefix_id": second_root.id,
        }
    )

    first_facts = canonical_rq_prefix_facts([first_child, first_root])
    second_facts = canonical_rq_prefix_facts([second_root, second_child])

    assert first_facts == second_facts
    assert stable_hash(first_facts) == stable_hash(second_facts)
    assert first_facts[1]["parent_rq_prefix_key"] == "rq:L1:1"


@pytest.mark.parametrize(
    ("setting_key", "value"),
    [
        ("edge_distance_protocol", "legacy_distance_v0"),
        ("rq_membership_protocol", "legacy_hard_path_v0"),
        ("edge_projection_protocol", "free_expression_v1"),
        ("edge_type_calibration_protocol", "llm_calibration_v1"),
        ("rq_membership_temperature", float("nan")),
    ],
)
def test_graph_protocol_admission_fails_closed_on_mismatch(setting_key, value):
    from app.services.graph_protocols import (
        GraphProtocolAdmissionError,
        validate_active_graph_protocol_settings,
    )

    with pytest.raises(GraphProtocolAdmissionError):
        validate_active_graph_protocol_settings(
            _active_settings(**{setting_key: value})
        )


def test_active_settings_endpoint_treats_exact_rebuild_identity_as_side_effect_free_noop(monkeypatch):
    from app.services import runtime_settings

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: _active_settings())
    monkeypatch.setattr(
        runtime_settings,
        "normalize_env_file",
        lambda: pytest.fail("rebuild identity echo must not normalize or write the active env"),
    )
    monkeypatch.setattr(runtime_settings, "model_settings_payload", lambda: {"status": "unchanged"})

    result = runtime_settings.update_model_settings(
        {**PROTOCOL_SETTINGS, **RQ_MEMBERSHIP_SETTINGS}
    )

    assert result == {"status": "unchanged"}


@pytest.mark.parametrize(
    "patch",
    [
        {"edge_distance_protocol": "llm_prompt_distance_v9"},
        {"rq_membership_protocol": "free_expression_v9"},
        {"edge_projection_protocol": None},
        {"edge_type_calibration_protocol": 42},
        {"rq_membership_temperature": 0.0},
    ],
)
def test_active_settings_endpoint_rejects_invalid_graph_protocol_patch_before_side_effects(
    monkeypatch,
    patch,
):
    from app.services import runtime_settings

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: _active_settings())
    monkeypatch.setattr(
        runtime_settings,
        "normalize_env_file",
        lambda: pytest.fail("invalid rebuild patch must fail before side effects"),
    )

    with pytest.raises(ValueError):
        runtime_settings.update_model_settings(patch)


@pytest.mark.parametrize(
    ("setting_key", "active_value", "requested_value"),
    [
        (
            "edge_distance_protocol",
            "legacy_distance_v0",
            "edge_distance_log_calibrated_strength_v2",
        ),
        ("rq_membership_temperature", 0.35, 0.4),
    ],
)
def test_active_settings_endpoint_requires_candidate_promotion_for_valid_changes(
    monkeypatch,
    setting_key,
    active_value,
    requested_value,
):
    from app.services import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "get_settings",
        lambda: _active_settings(**{setting_key: active_value}),
    )
    monkeypatch.setattr(
        runtime_settings,
        "normalize_env_file",
        lambda: pytest.fail("rebuild change must fail before active env mutation"),
    )

    with pytest.raises(ValueError, match="RuntimeSettingsCandidate"):
        runtime_settings.update_model_settings({setting_key: requested_value})
