from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def test_embedding_protocol_is_closed_and_rebuild_required() -> None:
    from app.core.config import (
        EMBEDDING_API_PROTOCOL_ALLOWLIST,
        HOT_RELOAD_SETTINGS,
        REBUILD_REQUIRED_SETTINGS,
        Settings,
        _validate_fixed_protocol_env,
    )
    from app.schemas import ModelSettingsResponse, ModelSettingsUpdate

    settings = Settings(_env_file=None)
    assert EMBEDDING_API_PROTOCOL_ALLOWLIST == frozenset({"openai"})
    assert settings.embedding_api_protocol == "openai"
    assert "embedding_api_protocol" in REBUILD_REQUIRED_SETTINGS
    assert "embedding_api_protocol" not in HOT_RELOAD_SETTINGS
    assert ModelSettingsUpdate(embedding_api_protocol="openai").embedding_api_protocol == "openai"
    assert ModelSettingsResponse.model_fields["embedding_api_protocol"].default == "openai"

    for invalid in ("anthropic", "OpenAI", " openai", "openai "):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, embedding_api_protocol=invalid)
        with pytest.raises(ValidationError):
            ModelSettingsUpdate.model_validate({"embedding_api_protocol": invalid})
        with pytest.raises(ValueError, match="EMBEDDING_API_PROTOCOL"):
            _validate_fixed_protocol_env({"EMBEDDING_API_PROTOCOL": invalid})


def test_embedding_protocol_candidate_identity_and_frozen_schema_are_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import Settings
    from app.services import runtime_settings_lifecycle as lifecycle
    from app.services.vector_shadow_lifecycle import (
        VECTOR_SCHEMA_PROTOCOL_VERSION,
        FrozenVectorSchema,
        frozen_vector_schema,
        vector_schema_hash,
    )

    active = Settings(
        _env_file=None,
        model_bridge_enabled=False,
        chat_api_protocol="openai",
        chat_base_url="https://chat.example.test/v1",
        graph_api_protocol="openai",
        graph_base_url="https://graph.example.test/v1",
        embedding_api_protocol="openai",
        embedding_base_url="https://embedding.example.test/v1",
    )
    monkeypatch.setattr(lifecycle, "get_settings", lambda: active)
    candidate, base, snapshot, changed, bridge = lifecycle._validated_candidate_settings(
        {
            "embedding_api_protocol": "openai",
            "embedding_model": "next-embedding-model",
        }
    )

    assert base["embedding_api_protocol"] == "openai"
    assert snapshot["embedding_api_protocol"] == "openai"
    assert candidate.embedding_api_protocol == "openai"
    assert changed == ["embedding_model"]
    assert "embedding_api_protocol" in lifecycle.VECTOR_REBUILD_KEYS
    assert bridge["embedding_api_protocol"] == "openai"
    assert bridge["model_call_count"] == 0

    text_version = lifecycle._candidate_embedding_text_version(
        candidate,
        changed,
        "chunk_schema_v1",
    )
    schema = frozen_vector_schema(
        embedding_api_protocol=candidate.embedding_api_protocol,
        embedding_model=candidate.embedding_model,
        embedding_dimension=candidate.embedding_dimensions,
        embedding_text_version=text_version,
    )
    assert VECTOR_SCHEMA_PROTOCOL_VERSION == "frozen_vector_schema_v2"
    assert schema.embedding_api_protocol == "openai"
    assert len(vector_schema_hash(schema)) == 64

    legacy = schema.model_dump(mode="json")
    legacy["protocol_version"] = "frozen_vector_schema_v1"
    legacy.pop("embedding_api_protocol")
    with pytest.raises(ValidationError):
        FrozenVectorSchema.model_validate(legacy)


@pytest.mark.parametrize(
    "invalid",
    ["anthropic", "OpenAI", " openai", "openai ", None, True],
)
def test_embedding_protocol_candidate_rejects_attacks_before_identity_io(
    invalid: object,
) -> None:
    from app.services.runtime_settings_lifecycle import _validated_candidate_settings

    with pytest.raises(ValueError, match="embedding_api_protocol"):
        _validated_candidate_settings({"embedding_api_protocol": invalid})


def test_embedding_provider_rejects_anthropic_before_fallback_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import embeddings

    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_api_protocol="anthropic",
            embedding_model="model",
            embedding_dimensions=8,
            enable_model_fallback=True,
        ),
    )
    with pytest.raises(RuntimeError, match="no embedding request/response contract"):
        embeddings.EmbeddingProvider()


def test_embedding_protocol_env_contract_is_explicit() -> None:
    workspace = Path(__file__).resolve().parents[3]
    runtime_env = Path(os.environ.get("RUNTIME_ENV_FILE") or workspace / ".env")
    assert "EMBEDDING_API_PROTOCOL=openai" in runtime_env.with_name(".env.example").read_text(
        encoding="utf-8"
    )
    compose = (workspace / "infra" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert "EMBEDDING_API_PROTOCOL: ${EMBEDDING_API_PROTOCOL:-openai}" not in compose
    assert compose.count("RUNTIME_ENV_FILE: /workspace/.env") == 3
    assert compose.count("env_file:\n      - ../.env") >= 4
    launcher = (workspace / "start-app.ps1").read_text(encoding="utf-8")
    assert '"--embedding-api-protocol", $embeddingApiProtocol' in launcher
    assert 'if ($embeddingApiProtocol -cne "openai")' in launcher
    contract_script = (workspace / "scripts" / "check_runtime_settings_contract.py").read_text(
        encoding="utf-8"
    )
    assert 'EMBEDDING_API_PROTOCOL_ALLOWLIST != frozenset({"openai"})' in contract_script
    assert '("embedding_api_protocol", "rebuild_required")' in contract_script
