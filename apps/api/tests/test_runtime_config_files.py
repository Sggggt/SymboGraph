import json
from pathlib import Path

import pytest

from app.core.config import (
    ENV_AUTHORITY_SETTINGS,
    HOT_RELOAD_SETTINGS,
    REBUILD_REQUIRED_SETTINGS,
    RUNTIME_JSON_SETTINGS,
    SERVICE_RECREATE_REQUIRED_SETTINGS,
)
from app.core.runtime_config import (
    RuntimeSettingsFileError,
    atomic_replace_runtime_settings,
    parse_runtime_settings_bytes,
    read_runtime_settings_values,
    runtime_settings_bytes,
    runtime_settings_file_identity,
)


def _example_path() -> Path:
    candidate = Path(__file__).resolve().parents[3] / "settings.example.json"
    return candidate if candidate.exists() else Path("/workspace/settings.example.json")


def test_runtime_authority_sets_are_disjoint_and_example_is_complete():
    assert ENV_AUTHORITY_SETTINGS.isdisjoint(RUNTIME_JSON_SETTINGS)
    assert RUNTIME_JSON_SETTINGS == (
        HOT_RELOAD_SETTINGS
        | REBUILD_REQUIRED_SETTINGS
        | SERVICE_RECREATE_REQUIRED_SETTINGS
    ) - ENV_AUTHORITY_SETTINGS
    values = read_runtime_settings_values(
        _example_path(),
        allowed_keys=RUNTIME_JSON_SETTINGS,
        allow_missing=False,
    )
    assert set(values) == set(RUNTIME_JSON_SETTINGS)
    assert not {
        "retrieval_repair_round_limit",
        "retrieval_sufficiency_timeout_seconds",
        "retrieval_reward_time_weight",
        "query_facet_posterior_enabled",
        "agent_reflection_round_budget",
    }.intersection(RUNTIME_JSON_SETTINGS)


def test_runtime_settings_json_rejects_duplicate_unknown_and_nonfinite_values():
    with pytest.raises(RuntimeSettingsFileError, match="invalid"):
        parse_runtime_settings_bytes(
            b'{"protocol_version":"symbograph_runtime_settings_v1","settings":{"upload_max_bytes":1,"upload_max_bytes":2}}',
            allowed_keys={"upload_max_bytes"},
        )
    with pytest.raises(RuntimeSettingsFileError, match="unsupported"):
        parse_runtime_settings_bytes(
            json.dumps({
                "protocol_version": "symbograph_runtime_settings_v1",
                "settings": {"unknown": 1},
            }).encode(),
            allowed_keys={"upload_max_bytes"},
        )
    with pytest.raises(RuntimeSettingsFileError, match="finite"):
        runtime_settings_bytes(
            {"upload_max_bytes": float("nan")},
            allowed_keys={"upload_max_bytes"},
        )


def test_runtime_settings_atomic_publication_replays_exact_bytes(tmp_path):
    path = tmp_path / "settings.json"
    content = runtime_settings_bytes(
        {"upload_max_bytes": 1024},
        allowed_keys={"upload_max_bytes"},
    )
    atomic_replace_runtime_settings(path, content)
    assert path.read_bytes() == content
    identity = runtime_settings_file_identity(path)
    assert identity["exists"] is True
    assert identity["size"] == len(content)
    assert len(identity["sha256"]) == 64
