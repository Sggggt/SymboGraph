from __future__ import annotations

import hashlib
from datetime import datetime

import pytest
from sqlalchemy import func, select, update


def _bind_current_bridge_identity(candidate, overrides: dict) -> None:
    """Give manually constructed activation rows the real immutable card."""

    from app.core.config import get_settings
    from app.services.runtime_settings_lifecycle import (
        RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
        _stable_hash,
        _validated_candidate_bridge_identity,
        _validated_closed_bridge_identity,
    )

    card = _validated_closed_bridge_identity(
        _validated_candidate_bridge_identity(get_settings(), overrides)
    )
    candidate_identity = {
        "protocol_version": RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
        "candidate_overrides": dict(overrides),
        "model_bridge_identity": card,
    }
    candidate.candidate_hash = _stable_hash(candidate_identity)
    candidate.settings_json = {
        "candidate_overrides": dict(overrides),
        "model_bridge_identity": card,
    }
    candidate.diagnostics_json = {
        "candidate_identity": candidate_identity,
        "model_bridge_identity": card,
    }


@pytest.fixture
def isolated_bridge_candidate_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from app.core import config
    from app.core.config import get_settings
    from app.services import runtime_settings

    managed_env = tmp_path / ".env"

    def configure(
        *,
        enabled: bool = True,
        chat_target: str = "https://chat.example.test/v1",
        graph_target: str = "https://graph.example.test/v1",
        embedding_target: str = "https://embedding.example.test/v1",
    ):
        managed_env.write_text(
            "\n".join(
                [
                    f"MODEL_BRIDGE_ENABLED={'true' if enabled else 'false'}",
                    "MODEL_BRIDGE_PORT=8765",
                    "CHAT_API_PROTOCOL=openai",
                    f"CHAT_BASE_URL={chat_target}",
                    "CHAT_RESOLVE_IP=1.1.1.1",
                    "GRAPH_API_PROTOCOL=openai",
                    f"GRAPH_BASE_URL={graph_target}",
                    "GRAPH_RESOLVE_IP=3.3.3.3",
                    "EMBEDDING_API_PROTOCOL=openai",
                    f"EMBEDDING_BASE_URL={embedding_target}",
                    "EMBEDDING_RESOLVE_IP=2.2.2.2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("RUNTIME_ENV_FILE", str(managed_env))
        monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true" if enabled else "false")
        monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
        monkeypatch.setenv("CHAT_API_PROTOCOL", "openai")
        monkeypatch.setenv("CHAT_BASE_URL", chat_target)
        monkeypatch.setenv("GRAPH_API_PROTOCOL", "openai")
        monkeypatch.setenv("GRAPH_BASE_URL", graph_target)
        monkeypatch.setenv("EMBEDDING_API_PROTOCOL", "openai")
        monkeypatch.setenv("EMBEDDING_BASE_URL", embedding_target)
        monkeypatch.delenv("API_CHAT_BASE_URL", raising=False)
        monkeypatch.delenv("API_CHAT_RESOLVE_IP", raising=False)
        monkeypatch.setattr(runtime_settings, "ENV_PATH", managed_env)
        monkeypatch.setattr(
            config,
            "_read_workspace_env",
            lambda: runtime_settings._env_entries(managed_env),
        )
        get_settings.cache_clear()
        return managed_env

    yield configure
    get_settings.cache_clear()


def test_lifecycle_sets_are_disjoint_and_direct_writes_enforce_all_three_classes(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env,
):
    from app.core.config import (
        HOT_RELOAD_SETTINGS,
        REBUILD_REQUIRED_SETTINGS,
        SERVICE_RECREATE_REQUIRED_SETTINGS,
        get_settings,
    )
    from app.services import runtime_settings

    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "false")
    get_settings.cache_clear()

    assert HOT_RELOAD_SETTINGS.isdisjoint(REBUILD_REQUIRED_SETTINGS)
    assert HOT_RELOAD_SETTINGS.isdisjoint(SERVICE_RECREATE_REQUIRED_SETTINGS)
    assert REBUILD_REQUIRED_SETTINGS.isdisjoint(SERVICE_RECREATE_REQUIRED_SETTINGS)
    assert "chat_api_protocol" in HOT_RELOAD_SETTINGS
    assert "chat_json_max_tokens" in HOT_RELOAD_SETTINGS
    assert "chat_api_protocol" not in REBUILD_REQUIRED_SETTINGS
    assert "graph_api_protocol" in REBUILD_REQUIRED_SETTINGS
    assert "graph_api_protocol" not in HOT_RELOAD_SETTINGS

    side_effects: list[str] = []
    monkeypatch.setattr(
        runtime_settings,
        "normalize_env_file",
        lambda *args, **kwargs: side_effects.append("normalize") or False,
    )
    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda *args, **kwargs: side_effects.append("env"),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_apply_runtime_env",
        lambda *args, **kwargs: side_effects.append("process"),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda *args, **kwargs: side_effects.append("broadcast") or {},
    )
    monkeypatch.setattr(
        runtime_settings,
        "model_settings_payload",
        lambda *args, **kwargs: {"lifecycle": "unit"},
    )

    settings = get_settings()
    with pytest.raises(ValueError, match="Direct active mutation is forbidden"):
        runtime_settings.update_model_settings(
            {"fixed_chunk_size_tokens": settings.fixed_chunk_size_tokens + 1}
        )
    assert side_effects == []

    next_graph_protocol = (
        "anthropic" if settings.graph_api_protocol == "openai" else "openai"
    )
    with pytest.raises(ValueError, match="RuntimeSettingsCandidate"):
        runtime_settings.update_model_settings(
            {"graph_api_protocol": next_graph_protocol}
        )
    assert side_effects == []

    for invalid_protocol in (None, "", "OpenAI", " anthropic", "unknown"):
        with pytest.raises(ValueError, match="allowlisted model API protocol"):
            runtime_settings.update_model_settings(
                {"chat_api_protocol": invalid_protocol}
            )
        assert side_effects == []

    result = runtime_settings.update_model_settings(
        {"worker_concurrency": settings.worker_concurrency + 1}
    )
    assert result["requires_service_recreate"] is True
    assert result["service_recreate_changes"] == ["worker_concurrency"]
    assert result["active_mutated"] is False
    assert result["runtime_version_broadcast"] is False
    assert side_effects == []

    next_cadence = 2 if settings.gray_zone_observation_cadence != 2 else 3
    runtime_settings.update_model_settings(
        {"gray_zone_observation_cadence": next_cadence}
    )
    assert side_effects == ["normalize", "env", "process", "broadcast"]


def test_single_root_env_save_persists_all_lifecycles_and_only_hot_applies(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    no_fallback_env,
):
    from app.core.config import get_settings
    from app.models import RuntimeSettingsAudit
    from app.services import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda **_kwargs: {
            "version_hash": "b" * 64,
            "runtime_version_broadcast": True,
            "broadcast_pending": False,
            "local_refresh_pending": False,
        },
    )

    settings = get_settings()
    next_chat_model = f"{settings.chat_model}-next"
    next_chunk_size = settings.fixed_chunk_size_tokens + 16
    next_worker_concurrency = settings.worker_concurrency + 1
    result = runtime_settings.save_model_settings_to_root_env(
        db_session,
        {
            "chat_model": next_chat_model,
            # The settings form submits a complete snapshot. Exact echoes of
            # unchanged hot fields must not be replayed into the active env or
            # trigger model-bridge preflight/reload.
            "chat_api_protocol": settings.chat_api_protocol,
            "chat_base_url": settings.chat_base_url,
            "chat_resolve_ip": settings.chat_resolve_ip,
            "model_bridge_enabled": settings.model_bridge_enabled,
            "fixed_chunk_size_tokens": next_chunk_size,
            "worker_concurrency": next_worker_concurrency,
        },
    )

    assert result["active_mutated"] is True
    assert result["setting_statuses"]["chat_model"] == "written_and_applied"
    assert result["setting_statuses"]["fixed_chunk_size_tokens"] == (
        "written_pending_rebuild"
    )
    assert result["setting_statuses"]["worker_concurrency"] == (
        "written_pending_service_recreate"
    )
    assert result["pending_rebuild_changes"] == ["fixed_chunk_size_tokens"]
    assert result["pending_service_recreate_changes"] == ["worker_concurrency"]

    root_entries = runtime_settings._env_entries(runtime_settings.ENV_PATH)
    assert root_entries["CHAT_MODEL"] == next_chat_model
    assert root_entries["FIXED_CHUNK_SIZE_TOKENS"] == str(next_chunk_size)
    assert root_entries["WORKER_CONCURRENCY"] == str(next_worker_concurrency)
    assert get_settings().chat_model == next_chat_model
    assert get_settings().fixed_chunk_size_tokens == settings.fixed_chunk_size_tokens
    assert get_settings().worker_concurrency == settings.worker_concurrency

    row = db_session.scalar(select(RuntimeSettingsAudit))
    assert row is not None
    assert row.status == "pending_lifecycle"
    assert row.changed_keys_json == [
        "chat_model",
        "fixed_chunk_size_tokens",
        "worker_concurrency",
    ]
    assert not hasattr(row, "settings_json")
    assert not hasattr(row, "secret_presence_json")


def test_service_setting_is_written_once_and_becomes_effective_after_restart_load(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    no_fallback_env,
):
    from app.core.config import get_settings
    from app.services import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda **_kwargs: {
            "version_hash": "b" * 64,
            "runtime_version_broadcast": True,
            "broadcast_pending": False,
        },
    )
    before = get_settings().worker_concurrency
    result = runtime_settings.save_model_settings_to_root_env(
        db_session,
        {"worker_concurrency": before + 1},
    )
    assert result["pending_service_recreate_changes"] == ["worker_concurrency"]
    assert get_settings().worker_concurrency == before
    assert runtime_settings._env_entries(runtime_settings.ENV_PATH)["WORKER_CONCURRENCY"] == str(before + 1)

    runtime_settings._apply_runtime_env_file_to_process_environment()
    get_settings.cache_clear()
    restarted = runtime_settings.model_settings_payload()
    assert get_settings().worker_concurrency == before + 1
    assert restarted["pending_service_recreate_changes"] == []


def test_graph_protocol_candidate_is_frozen_as_four_layer_rebuild_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    no_fallback_env,
):
    from app.core.config import get_settings
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import (
        _validated_candidate_settings,
        _validated_closed_bridge_identity,
    )

    managed_env = tmp_path / ".env"
    managed_env.write_text(
        "\n".join(
            [
                "MODEL_BRIDGE_ENABLED=false",
                "GRAPH_API_PROTOCOL=openai",
                "GRAPH_BASE_URL=https://graph.example.test",
                "GRAPH_RESOLVE_IP=1.1.1.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_ENV_FILE", str(managed_env))
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "false")
    monkeypatch.setenv("GRAPH_API_PROTOCOL", "openai")
    monkeypatch.setenv("GRAPH_BASE_URL", "https://graph.example.test")
    monkeypatch.setenv("GRAPH_RESOLVE_IP", "1.1.1.1")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", managed_env)
    get_settings.cache_clear()

    candidate, base, snapshot, changed, bridge_identity = (
        _validated_candidate_settings({"graph_api_protocol": "anthropic"})
    )

    assert base["graph_api_protocol"] == "openai"
    assert snapshot["graph_api_protocol"] == "anthropic"
    assert candidate.graph_api_protocol == "anthropic"
    assert changed == ["graph_api_protocol"]
    closed = _validated_closed_bridge_identity(bridge_identity)
    assert closed["bridge_enabled"] is False
    assert len(closed["direct_graph_identity_hash"]) == 64
    assert closed["model_call_count"] == 0
    assert closed["provider_response_used_as_fact"] is False
    get_settings.cache_clear()


def test_pending_rebuild_audit_survives_process_restart_without_value_mirror(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    no_fallback_env,
):
    from app.core.config import get_settings
    from app.models import RuntimeSettingsAudit
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import _validated_candidate_settings

    current = get_settings().dense_knn_k_min
    requested = min(get_settings().dense_knn_k_max, current + 1)
    if requested == current:
        requested = max(1, current - 1)
    runtime_settings._update_env_file({"dense_knn_k_min": requested})
    monkeypatch.setenv("DENSE_KNN_K_MIN", str(requested))
    get_settings.cache_clear()
    identity = runtime_settings.runtime_env_file_identity(runtime_settings.ENV_PATH)
    row = RuntimeSettingsAudit(
        protocol_version="runtime_settings_audit_v1",
        version_hash="e" * 64,
        prior_runtime_version_hash=None,
        changed_keys_json=["dense_knn_k_min"],
        lifecycle_json={
            "hot_reloadable": [],
            "rebuild_required": ["dense_knn_k_min"],
            "service_recreate_required": [],
        },
        field_status_json={
            "dense_knn_k_min": "written_pending_rebuild",
        },
        status="pending_lifecycle",
        env_identity_hash=identity["identity_hash"],
        source="unit-test",
    )
    db_session.add(row)
    db_session.commit()

    _candidate, base, candidate, changed, _bridge = _validated_candidate_settings(
        {"dense_knn_k_min": requested}
    )

    assert base["dense_knn_k_min"] == requested
    assert candidate["dense_knn_k_min"] == requested
    assert changed == ["dense_knn_k_min"]
    assert not hasattr(row, "settings_json")
    get_settings.cache_clear()

def test_candidate_rejects_embedding_bridge_self_target_before_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    no_fallback_env,
):
    from app.core.config import get_settings
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import _validated_candidate_settings

    managed_env = tmp_path / ".env"
    managed_env.write_text(
        "MODEL_BRIDGE_ENABLED=true\nMODEL_BRIDGE_PORT=8765\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_ENV_FILE", str(managed_env))
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "unit-test-bridge-token")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", managed_env)
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        _validated_candidate_settings(
            {"embedding_base_url": "http://host.docker.internal:8765"}
        )
    get_settings.cache_clear()


def test_bridge_enabled_candidates_use_upstream_provider_identity_not_serving_url(
    isolated_bridge_candidate_env,
):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services.runtime_settings_lifecycle import (
        RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
        _validated_candidate_settings,
    )

    isolated_bridge_candidate_env()
    settings = get_settings()
    bridge_serving_url = model_bridge_client_base_url(8765)
    assert settings.chat_base_url == bridge_serving_url
    assert settings.embedding_base_url == bridge_serving_url

    graph_requests = {
        "dense_knn_k_min": min(
            settings.dense_knn_k_max,
            settings.dense_knn_k_min + 1,
        ),
        "rq_membership_temperature": (
            settings.rq_membership_temperature + 0.05
            if settings.rq_membership_temperature <= 9.95
            else settings.rq_membership_temperature - 0.05
        ),
    }
    bridge_hashes: set[tuple[str, str]] = set()
    for key, value in graph_requests.items():
        _candidate, _base, _candidate_snapshot, changed, identity = (
            _validated_candidate_settings({key: value})
        )
        assert changed == [key]
        assert identity["protocol_version"] == (
            RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION
        )
        assert identity["bridge_enabled"] is True
        assert identity["serving_chat_is_bridge_endpoint"] is True
        assert identity["serving_embedding_is_bridge_endpoint"] is True
        assert identity["candidate_upstream_changed"] is False
        assert identity["base_upstream_config_hash"] == identity[
            "candidate_upstream_config_hash"
        ]
        assert identity["provider_response_used_as_fact"] is False
        assert identity["model_call_count"] == 0
        bridge_hashes.add(
            (
                identity["serving_identity_hash"],
                identity["candidate_upstream_config_hash"],
            )
        )
    assert len(bridge_hashes) == 1

    requested_chunk_size = 256 if settings.fixed_chunk_size_tokens != 256 else 384
    _candidate, _base, _candidate_snapshot, changed, identity = (
        _validated_candidate_settings(
            {"fixed_chunk_size_tokens": requested_chunk_size}
        )
    )
    assert changed == ["fixed_chunk_size_tokens"]
    assert identity["candidate_upstream_changed"] is False
    serving_hash, upstream_hash = next(iter(bridge_hashes))
    assert identity["serving_identity_hash"] == serving_hash
    assert identity["base_upstream_config_hash"] == upstream_hash
    assert identity["candidate_upstream_config_hash"] == upstream_hash


def test_candidate_explicit_bridge_self_targets_fail_before_any_side_effect(
    isolated_bridge_candidate_env,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.core.config import model_bridge_client_base_url
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import _validated_candidate_settings

    isolated_bridge_candidate_env()
    effects: list[str] = []
    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda *args, **kwargs: effects.append("env"),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda *args, **kwargs: effects.append("cache") or {},
    )
    monkeypatch.setattr(
        runtime_settings,
        "reload_model_bridge",
        lambda *args, **kwargs: effects.append("bridge") or {},
    )
    self_target = model_bridge_client_base_url(8765)
    for field in ("chat_base_url", "embedding_base_url"):
        with pytest.raises(ValueError, match=field.upper()):
            _validated_candidate_settings({field: self_target})
    assert effects == []


def test_candidate_rejects_existing_upstream_self_target_and_invalid_identity(
    isolated_bridge_candidate_env,
):
    from app.core.config import get_settings
    from app.services.runtime_settings_lifecycle import _validated_candidate_settings

    managed_env = isolated_bridge_candidate_env(
        embedding_target="http://localhost:8765"
    )
    settings = get_settings()
    requested = min(settings.dense_knn_k_max, settings.dense_knn_k_min + 1)
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        _validated_candidate_settings({"dense_knn_k_min": requested})

    managed_env.write_text(
        managed_env.read_text(encoding="utf-8")
        .replace(
            "CHAT_BASE_URL=https://chat.example.test/v1",
            "CHAT_BASE_URL=",
        )
        .replace(
            "EMBEDDING_BASE_URL=http://localhost:8765",
            "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CHAT_BASE_URL"):
        _validated_candidate_settings({"dense_knn_k_min": requested})

    managed_env.write_text(
        managed_env.read_text(encoding="utf-8").replace(
            "CHAT_BASE_URL=",
            "CHAT_BASE_URL=ambiguous-provider-target",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"HTTPS/public.*CHAT_BASE_URL"):
        _validated_candidate_settings({"dense_knn_k_min": requested})


def test_bridge_disabled_candidate_still_rejects_unsafe_direct_provider_target(
    isolated_bridge_candidate_env,
):
    from app.services.runtime_settings_lifecycle import _validated_candidate_settings

    isolated_bridge_candidate_env(enabled=False)
    with pytest.raises(ValueError, match="must use HTTPS"):
        _validated_candidate_settings(
            {"embedding_base_url": "http://127.0.0.1:8765"}
        )


def test_runtime_graph_hash_excludes_hot_and_service_values_but_tracks_rebuild_values():
    from app.core.config import get_settings, use_runtime_settings_override
    from app.services.context_graph import runtime_settings_state_hash

    settings = get_settings()
    baseline = runtime_settings_state_hash()
    with use_runtime_settings_override(
        {"chat_model": settings.chat_model + "-hot-only"}
    ):
        assert runtime_settings_state_hash() == baseline
    with use_runtime_settings_override(
        {"worker_concurrency": settings.worker_concurrency + 1}
    ):
        assert runtime_settings_state_hash() == baseline
    changed_dense = min(0.99, settings.dense_min_cosine + 0.01)
    with use_runtime_settings_override({"dense_min_cosine": changed_dense}):
        assert runtime_settings_state_hash() != baseline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "graph_setting",
    [
        "dense_knn_k_min",
        "rq_membership_temperature",
        "mid_concept_extraction_max_model_batches",
    ],
)
async def test_graph_only_candidate_build_evaluate_promote_activate_and_rollback(
    db_session,
    populated_context_graph,
    fake_model_stack,
    isolated_bridge_candidate_env,
    monkeypatch: pytest.MonkeyPatch,
    graph_setting: str,
):
    from app.core.config import get_settings
    from app.models import (
        CoarseConcept,
        ContextGraphState,
        ChunkRelationGraphState,
        KnowledgeBaseVectorRuntimeState,
        MidConcept,
        RuntimeSettingsActivationIntent,
    )
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import (
        apply_runtime_settings_activation_intent,
        build_runtime_settings_shadow,
        evaluate_runtime_settings_shadow,
        promote_runtime_settings_candidate,
        rollback_runtime_settings_candidate,
        stage_runtime_settings_candidate,
    )

    isolated_bridge_candidate_env(enabled=False)
    knowledge_base = populated_context_graph["knowledge_base"]
    old_context = populated_context_graph["state"]
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert pointer is not None
    old_pointer_state_hash = pointer.state_hash
    old_context_id = pointer.active_context_graph_state_id
    old_mid_id = pointer.active_mid_concept_state_id
    old_coarse_id = pointer.active_coarse_concept_state_id
    settings = get_settings()
    if graph_setting == "dense_knn_k_min":
        requested_value = min(
            settings.dense_knn_k_max,
            settings.dense_knn_k_min + 1,
        )
    elif graph_setting == "rq_membership_temperature":
        current_temperature = float(settings.rq_membership_temperature)
        requested_value = (
            current_temperature + 0.05
            if current_temperature <= 9.95
            else current_temperature - 0.05
        )
    else:
        current_max_batches = int(
            settings.mid_concept_extraction_max_model_batches
        )
        requested_value = (
            current_max_batches + 1
            if current_max_batches < 64
            else current_max_batches - 1
        )

    candidate, builds = stage_runtime_settings_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        requested_settings={graph_setting: requested_value},
        source=f"unit_test_graph_only_{graph_setting}",
    )
    assert builds[0].status == "dry_run_passed"
    assert pointer.state_hash == old_pointer_state_hash
    assert old_context.state == "active"

    build = await build_runtime_settings_shadow(
        db_session,
        build_id=builds[0].id,
        emit_heartbeats=False,
    )
    assert build.status == "shadow_ready"
    assert build.vector_shadow_build_id is None
    shadow_context = db_session.get(
        ContextGraphState, build.shadow_context_graph_state_id
    )
    assert shadow_context is not None and shadow_context.state == "shadow"
    assert pointer.state_hash == old_pointer_state_hash
    assert old_context.state == "active"
    provider_evidence = build.build_metrics_json["concept_provider_evidence"]
    assert provider_evidence["provider_response_persisted"] is False
    operating_point_evidence = build.build_metrics_json[
        "relation_operating_point_evidence"
    ]
    assert operating_point_evidence["source_relation_state_id"] == (
        str(old_context.chunk_relation_graph_state_id)
        if graph_setting != "dense_knn_k_min"
        else None
    )
    assert operating_point_evidence["active_operating_point_reused"] is (
        graph_setting != "dense_knn_k_min"
    )
    if graph_setting == "mid_concept_extraction_max_model_batches":
        assert build.build_metrics_json["scoped_shadow_rebuild"] is True
        assert build.build_metrics_json["reused_active_graph_layers"] == [
            "relation"
        ]
        assert build.shadow_chunk_relation_graph_state_id == (
            old_context.chunk_relation_graph_state_id
        )
        assert provider_evidence["semantic_reuse_required"] is True
        assert provider_evidence["semantic_reuse_hit_count"] > 0
        assert provider_evidence["semantic_reuse_miss_count"] == 0
        assert provider_evidence["provider_request_count"] == 0
    else:
        assert provider_evidence["semantic_reuse_required"] is False

    evaluated = evaluate_runtime_settings_shadow(db_session, build_id=build.id)
    assert evaluated.status == "evaluation_passed"
    assert evaluated.evaluation_result_json["passed"] is True
    assert all(evaluated.evaluation_result_json["hard_gates"].values())
    assert (
        evaluated.evaluation_result_json[
            "gray_zone_rule_decision_model_call_count"
        ]
        == 0
    )

    promotion = promote_runtime_settings_candidate(db_session, candidate.id)
    assert promotion["promoted"] is True
    assert promotion["active_env_mutated"] is False
    intent_id = str(promotion["activation_intent_id"])
    db_session.commit()
    db_session.expire_all()
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert pointer is not None
    assert pointer.active_context_graph_state_id == shadow_context.id
    assert db_session.get(ContextGraphState, old_context_id).state == "inactive"
    promoted_mid_id = pointer.active_mid_concept_state_id
    promoted_coarse_id = pointer.active_coarse_concept_state_id
    assert db_session.scalar(
        select(func.count(MidConcept.id)).where(
            MidConcept.concept_state_id == promoted_mid_id,
            MidConcept.state != "active",
        )
    ) == 0
    assert db_session.scalar(
        select(func.count(CoarseConcept.id)).where(
            CoarseConcept.coarse_state_id == promoted_coarse_id,
            CoarseConcept.state != "active",
        )
    ) == 0
    assert db_session.scalar(
        select(func.count(MidConcept.id)).where(
            MidConcept.concept_state_id == old_mid_id,
            MidConcept.state != "inactive",
        )
    ) == 0
    assert db_session.scalar(
        select(func.count(CoarseConcept.id)).where(
            CoarseConcept.coarse_state_id == old_coarse_id,
            CoarseConcept.state != "inactive",
        )
    ) == 0
    if graph_setting == "mid_concept_extraction_max_model_batches":
        assert pointer.active_chunk_relation_graph_state_id == (
            old_context.chunk_relation_graph_state_id
        )
        assert db_session.get(
            ChunkRelationGraphState,
            old_context.chunk_relation_graph_state_id,
        ).state == "active"
        db_session.execute(
            update(MidConcept)
            .where(MidConcept.concept_state_id == promoted_mid_id)
            .values(state="shadow")
        )
        db_session.execute(
            update(CoarseConcept)
            .where(CoarseConcept.coarse_state_id == promoted_coarse_id)
            .values(state="shadow")
        )
        db_session.execute(
            update(MidConcept)
            .where(MidConcept.concept_state_id == old_mid_id)
            .values(state="active")
        )
        db_session.execute(
            update(CoarseConcept)
            .where(CoarseConcept.coarse_state_id == old_coarse_id)
            .values(state="active")
        )
        replay = promote_runtime_settings_candidate(db_session, candidate.id)
        assert replay["promoted"] is True
        assert replay["idempotent_replay"] is True
        assert db_session.scalar(
            select(func.count(MidConcept.id)).where(
                MidConcept.concept_state_id == promoted_mid_id,
                MidConcept.state != "active",
            )
        ) == 0
        assert db_session.scalar(
            select(func.count(CoarseConcept.id)).where(
                CoarseConcept.coarse_state_id == promoted_coarse_id,
                CoarseConcept.state != "active",
            )
        ) == 0
        assert db_session.scalar(
            select(func.count(MidConcept.id)).where(
                MidConcept.concept_state_id == old_mid_id,
                MidConcept.state != "inactive",
            )
        ) == 0
        assert db_session.scalar(
            select(func.count(CoarseConcept.id)).where(
                CoarseConcept.coarse_state_id == old_coarse_id,
                CoarseConcept.state != "inactive",
            )
        ) == 0
        db_session.commit()
        db_session.expire_all()

    env_writes: list[dict] = []
    process_writes: list[dict] = []
    broadcasts: list[dict] = []
    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda updates: env_writes.append(dict(updates)),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_apply_runtime_env",
        lambda updates: process_writes.append(dict(updates)),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda **kwargs: broadcasts.append(dict(kwargs))
        or {
            "version_hash": hashlib.sha256(b"promotion").hexdigest(),
            "runtime_version_broadcast": True,
            "broadcast_pending": False,
            "local_refresh_pending": False,
        },
    )
    activation = apply_runtime_settings_activation_intent(intent_id)
    assert activation["status"] == "applied"
    assert env_writes == [{graph_setting: requested_value}]
    assert process_writes == env_writes
    assert len(broadcasts) == 1

    db_session.expire_all()
    rolled_back = rollback_runtime_settings_candidate(
        db_session,
        candidate.id,
        reason="unit_test_exact_restore",
    )
    assert rolled_back.status == "rolled_back"
    db_session.commit()
    rollback_intent = db_session.scalar(
        select(RuntimeSettingsActivationIntent).where(
            RuntimeSettingsActivationIntent.runtime_settings_candidate_id
            == candidate.id,
            RuntimeSettingsActivationIntent.direction == "rollback",
        )
    )
    assert rollback_intent is not None and rollback_intent.status == "pending"
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert pointer is not None
    assert pointer.state_hash == old_pointer_state_hash
    assert pointer.active_context_graph_state_id == old_context_id
    assert db_session.get(ContextGraphState, old_context_id).state == "active"
    assert db_session.get(
        ChunkRelationGraphState,
        old_context.chunk_relation_graph_state_id,
    ).state == "active"
    assert db_session.scalar(
        select(func.count(MidConcept.id)).where(
            MidConcept.concept_state_id == old_mid_id,
            MidConcept.state != "active",
        )
    ) == 0
    assert db_session.scalar(
        select(func.count(CoarseConcept.id)).where(
            CoarseConcept.coarse_state_id == old_coarse_id,
            CoarseConcept.state != "active",
        )
    ) == 0
    assert db_session.scalar(
        select(func.count(MidConcept.id)).where(
            MidConcept.concept_state_id == promoted_mid_id,
            MidConcept.state != "inactive",
        )
    ) == 0
    assert db_session.scalar(
        select(func.count(CoarseConcept.id)).where(
            CoarseConcept.coarse_state_id == promoted_coarse_id,
            CoarseConcept.state != "inactive",
        )
    ) == 0


@pytest.mark.asyncio
async def test_unpromoted_graph_candidate_can_be_abandoned_without_activation(
    db_session,
    populated_context_graph,
    fake_model_stack,
):
    from app.core.config import get_settings
    from app.models import (
        CoarseConceptState,
        ContextGraphState,
        KnowledgeBaseVectorRuntimeState,
        MidConceptState,
        RuntimeSettingsActivationIntent,
    )
    from app.services.runtime_settings_lifecycle import (
        build_runtime_settings_shadow,
        evaluate_runtime_settings_shadow,
        rollback_runtime_settings_candidate,
        stage_runtime_settings_candidate,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert pointer is not None
    base_pointer_hash = str(pointer.state_hash)
    base_graph_ids = {
        "context": str(pointer.active_context_graph_state_id),
        "relation": str(pointer.active_chunk_relation_graph_state_id),
        "mid": str(pointer.active_mid_concept_state_id),
        "coarse": str(pointer.active_coarse_concept_state_id),
    }
    current = int(get_settings().mid_concept_extraction_max_model_batches)
    requested = current + 1 if current < 64 else current - 1
    candidate, builds = stage_runtime_settings_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        requested_settings={
            "mid_concept_extraction_max_model_batches": requested,
        },
        source="unit_test_unpromoted_abandon",
    )
    build = await build_runtime_settings_shadow(
        db_session,
        build_id=builds[0].id,
        emit_heartbeats=False,
    )
    build.build_metrics_json = {
        **dict(build.build_metrics_json or {}),
        "elapsed_ms": float(
            get_settings().operating_point_hard_gate_max_candidate_latency_p95_ms
        )
        + 1.0,
    }
    evaluated = evaluate_runtime_settings_shadow(db_session, build_id=build.id)
    assert evaluated.status == "promotion_blocked"

    rolled_back = rollback_runtime_settings_candidate(
        db_session,
        candidate.id,
        reason="unit_test_abandon_failed_evaluation",
    )
    assert rolled_back.status == "rolled_back"
    assert rolled_back.diagnostics_json["unpromoted_abandon"] is True
    assert db_session.scalar(
        select(RuntimeSettingsActivationIntent).where(
            RuntimeSettingsActivationIntent.runtime_settings_candidate_id
            == candidate.id
        )
    ) is None
    db_session.refresh(pointer)
    assert pointer.state_hash == base_pointer_hash
    assert {
        "context": str(pointer.active_context_graph_state_id),
        "relation": str(pointer.active_chunk_relation_graph_state_id),
        "mid": str(pointer.active_mid_concept_state_id),
        "coarse": str(pointer.active_coarse_concept_state_id),
    } == base_graph_ids
    assert db_session.get(
        ContextGraphState, build.shadow_context_graph_state_id
    ).state == "inactive"
    assert db_session.get(
        MidConceptState, build.shadow_mid_concept_state_id
    ).state == "inactive"
    assert db_session.get(
        CoarseConceptState, build.shadow_coarse_concept_state_id
    ).state == "inactive"
    assert build.shadow_chunk_relation_graph_state_id == base_graph_ids["relation"]


@pytest.mark.asyncio
async def test_chunk_setting_build_creates_real_shadow_rechunk_and_reuses_vector_lifecycle(
    db_session,
    populated_context_graph,
    fake_model_stack,
):
    from app.core.config import get_settings
    from app.models import Chunk, DocumentVersion, KnowledgeBaseVectorRuntimeState
    from app.services.runtime_settings_lifecycle import (
        build_runtime_settings_shadow,
        evaluate_runtime_settings_shadow,
        promote_runtime_settings_candidate,
        rollback_runtime_settings_candidate,
        stage_runtime_settings_candidate,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    active_chunk_ids = sorted(
        str(chunk.id) for chunk in populated_context_graph["chunks"]
    )
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base.id
        )
    )
    assert pointer is not None
    active_pointer_hash = pointer.state_hash
    active_vector_schema_hash = pointer.vector_schema_hash
    active_graph_state_ids = (
        pointer.active_context_graph_state_id,
        pointer.active_chunk_relation_graph_state_id,
        pointer.active_mid_concept_state_id,
        pointer.active_coarse_concept_state_id,
    )
    requested_size = (
        256 if get_settings().fixed_chunk_size_tokens != 256 else 384
    )

    candidate, builds = stage_runtime_settings_candidate(
        db_session,
        knowledge_base_ids=[knowledge_base.id],
        requested_settings={"fixed_chunk_size_tokens": requested_size},
        source="unit_test_rechunk",
    )
    assert builds[0].vector_shadow_build_id is not None
    build = await build_runtime_settings_shadow(
        db_session,
        build_id=builds[0].id,
        emit_heartbeats=False,
    )
    shadow_ids = sorted(str(value) for value in build.candidate_chunk_ids_json)
    assert shadow_ids
    assert set(shadow_ids).isdisjoint(active_chunk_ids)
    shadow_chunks = list(
        db_session.scalars(
            select(Chunk).where(Chunk.id.in_(shadow_ids)).order_by(Chunk.id.asc())
        ).all()
    )
    assert all(chunk.state == "shadow" for chunk in shadow_chunks)
    assert all(
        (chunk.metadata_json or {}).get("chunk_schema_version")
        == build.candidate_chunk_schema_version
        for chunk in shadow_chunks
    )
    candidate_versions = list(
        db_session.scalars(
            select(DocumentVersion).where(
                DocumentVersion.id.in_(build.candidate_document_version_ids_json)
            )
        ).all()
    )
    assert candidate_versions and all(not row.is_active for row in candidate_versions)
    assert pointer.state_hash == active_pointer_hash
    assert all(chunk.state == "active" for chunk in populated_context_graph["chunks"])
    assert candidate.status in {"building", "evaluating"}

    evaluated = evaluate_runtime_settings_shadow(db_session, build_id=build.id)
    assert evaluated.status == "evaluation_passed"
    promotion = promote_runtime_settings_candidate(db_session, candidate.id)
    assert promotion["promoted"] is True
    db_session.flush()
    assert sorted(
        str(chunk.id)
        for chunk in db_session.scalars(
            select(Chunk).where(
                Chunk.knowledge_base_id == knowledge_base.id,
                Chunk.state == "active",
            )
        ).all()
    ) == shadow_ids
    assert all(chunk.state == "inactive" for chunk in populated_context_graph["chunks"])
    assert all(row.is_active for row in candidate_versions)
    assert knowledge_base.current_chunk_version == build.candidate_chunk_version

    rollback_runtime_settings_candidate(
        db_session,
        candidate.id,
        reason="unit_test_rechunk_exact_restore",
    )
    db_session.flush()
    assert sorted(
        str(chunk.id)
        for chunk in db_session.scalars(
            select(Chunk).where(
                Chunk.knowledge_base_id == knowledge_base.id,
                Chunk.state == "active",
            )
        ).all()
    ) == active_chunk_ids
    assert all(chunk.state == "active" for chunk in populated_context_graph["chunks"])
    assert all(not row.is_active for row in candidate_versions)
    assert knowledge_base.current_chunk_version == 1
    # Rollback restores the exact serving scope/schema while advancing the
    # activation generation, so the cache-facing pointer hash must change.
    assert pointer.state_hash != active_pointer_hash
    assert pointer.runtime_settings_candidate_id is None
    assert pointer.vector_schema_hash == active_vector_schema_hash
    assert (
        pointer.active_context_graph_state_id,
        pointer.active_chunk_relation_graph_state_id,
        pointer.active_mid_concept_state_id,
        pointer.active_coarse_concept_state_id,
    ) == active_graph_state_ids


def test_activation_failure_is_durable_and_retryable(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.models import RuntimeSettingsActivationIntent, RuntimeSettingsCandidate
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import (
        RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION,
        apply_runtime_settings_activation_intent,
    )

    candidate = RuntimeSettingsCandidate(
        protocol_version="runtime_settings_candidate_v2",
        candidate_hash="a" * 64,
        base_runtime_version_hash="b" * 64,
        settings_json={},
        changed_keys_json=["dense_knn_k_min"],
        target_knowledge_base_ids_json=[],
        lifecycle_scope="rebuild_required",
        status="promoted",
        diagnostics_json={},
        blocking_reasons_json=[],
    )
    _bind_current_bridge_identity(candidate, {"dense_knn_k_min": 6})
    db_session.add(candidate)
    db_session.flush()
    intent = RuntimeSettingsActivationIntent(
        runtime_settings_candidate_id=candidate.id,
        protocol_version=RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION,
        direction="promotion",
        status="pending",
        settings_json={"dense_knn_k_min": 6},
        settings_hash="c" * 64,
        changed_keys_json=["dense_knn_k_min"],
        expected_candidate_status="promoted",
        attempt_count=0,
        audit_json={},
    )
    db_session.add(intent)
    db_session.commit()

    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda updates: (_ for _ in ()).throw(OSError("unit failure")),
    )
    with pytest.raises(RuntimeError, match="retry the intent"):
        apply_runtime_settings_activation_intent(intent.id)
    db_session.expire_all()
    failed = db_session.get(RuntimeSettingsActivationIntent, intent.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.attempt_count == 1
    assert failed.last_error_type == "OSError"

    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: None)
    monkeypatch.setattr(runtime_settings, "_apply_runtime_env", lambda updates: None)
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda **kwargs: {
            "version_hash": "d" * 64,
            "runtime_version_broadcast": True,
            "broadcast_pending": False,
            "local_refresh_pending": False,
        },
    )
    result = apply_runtime_settings_activation_intent(intent.id)
    assert result["status"] == "applied"
    db_session.expire_all()
    applied = db_session.get(RuntimeSettingsActivationIntent, intent.id)
    assert applied is not None
    assert applied.status == "applied"
    assert applied.attempt_count == 2


def test_activation_fresh_apply_lease_blocks_concurrent_side_effects(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.models import RuntimeSettingsActivationIntent, RuntimeSettingsCandidate
    from app.services import runtime_settings
    from app.services.runtime_settings_lifecycle import (
        RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION,
        apply_runtime_settings_activation_intent,
    )

    candidate = RuntimeSettingsCandidate(
        protocol_version="runtime_settings_candidate_v2",
        candidate_hash="e" * 64,
        base_runtime_version_hash="f" * 64,
        settings_json={},
        changed_keys_json=["dense_knn_k_min"],
        target_knowledge_base_ids_json=[],
        lifecycle_scope="rebuild_required",
        status="promoted",
        diagnostics_json={},
        blocking_reasons_json=[],
    )
    _bind_current_bridge_identity(candidate, {"dense_knn_k_min": 6})
    db_session.add(candidate)
    db_session.flush()
    intent = RuntimeSettingsActivationIntent(
        runtime_settings_candidate_id=candidate.id,
        protocol_version=RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION,
        direction="promotion",
        status="applying",
        settings_json={"dense_knn_k_min": 6},
        settings_hash="1" * 64,
        changed_keys_json=["dense_knn_k_min"],
        expected_candidate_status="promoted",
        attempt_count=1,
        audit_json={},
        updated_at=datetime.utcnow(),
    )
    db_session.add(intent)
    db_session.commit()

    side_effects: list[str] = []
    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda updates: side_effects.append("env"),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_apply_runtime_env",
        lambda updates: side_effects.append("process"),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda **kwargs: side_effects.append("publish")
        or {"version_hash": "2" * 64},
    )

    with pytest.raises(RuntimeError, match="already applying"):
        apply_runtime_settings_activation_intent(intent.id)
    assert side_effects == []
    db_session.expire_all()
    unchanged = db_session.get(RuntimeSettingsActivationIntent, intent.id)
    assert unchanged is not None
    assert unchanged.status == "applying"
    assert unchanged.attempt_count == 1


def test_runtime_version_publish_reuses_intent_identity_on_retry(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    from app.models import RuntimeSettingsVersion
    from app.services import runtime_settings

    managed = tmp_path / ".env"
    managed.write_text("CACHE_TTL_SECONDS=300\n", encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", managed)

    redis_calls: list[tuple[str, str]] = []

    class FakeRedis:
        def set(self, key, value):
            redis_calls.append(("set", str(value)))

        def publish(self, channel, value):
            redis_calls.append(("publish", str(value)))

    monkeypatch.setattr(runtime_settings, "_redis_client", lambda: FakeRedis())
    monkeypatch.setattr(
        runtime_settings,
        "_local_runtime_refresh",
        lambda *args, **kwargs: None,
    )

    first = runtime_settings.publish_runtime_settings_version(
        ["FIXED_CHUNK_SIZE_TOKENS"],
        source="runtime_settings_promotion",
        idempotency_key="intent-1",
    )
    second = runtime_settings.publish_runtime_settings_version(
        ["FIXED_CHUNK_SIZE_TOKENS"],
        source="runtime_settings_promotion",
        idempotency_key="intent-1",
    )

    assert first == second
    rows = list(
        db_session.scalars(
            select(RuntimeSettingsVersion).where(
                RuntimeSettingsVersion.source
                == "runtime_settings_promotion:intent-1"
            )
        ).all()
    )
    assert len(rows) == 1
    assert rows[0].managed_env_identity_hash == (
        runtime_settings.runtime_env_file_identity(managed)["identity_hash"]
    )
    assert [name for name, _value in redis_calls] == [
        "set",
        "publish",
        "set",
        "publish",
    ]

    managed.write_text("CACHE_TTL_SECONDS=301\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="different managed env identity",
    ):
        runtime_settings.publish_runtime_settings_version(
            ["FIXED_CHUNK_SIZE_TOKENS"],
            source="runtime_settings_promotion",
            idempotency_key="intent-1",
        )


def test_runtime_version_postgres_lock_is_independent_from_retired_activation():
    import inspect

    from app.services import runtime_settings

    assert runtime_settings.RUNTIME_SETTINGS_PUBLICATION_ADVISORY_SCOPE == (
        "runtime_settings_version_publication_v1"
    )
    source = inspect.getsource(runtime_settings.publish_runtime_settings_version)
    assert "RUNTIME_SETTINGS_PUBLICATION_ADVISORY_SCOPE" in source
    assert "RUNTIME_SETTINGS_ACTIVATION_ADVISORY_SCOPE" not in source


@pytest.mark.asyncio
async def test_runtime_settings_candidate_api_exposes_dry_run_stage_and_status(
    db_session,
    populated_context_graph,
    fake_model_stack,
):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app
    from app.models import RuntimeSettingsCandidate

    knowledge_base = populated_context_graph["knowledge_base"]
    requested = min(
        get_settings().dense_knn_k_max,
        get_settings().dense_knn_k_min + 1,
    )
    request = {
        "knowledge_base_ids": [knowledge_base.id],
        "settings": {"dense_knn_k_min": requested},
        "source": "route_test",
    }
    with TestClient(app) as client:
        dry_run = client.post(
            "/api/settings/runtime-candidates",
            json={**request, "dry_run_only": True},
        )
        assert dry_run.status_code == 200
        assert dry_run.json()["preview"]["requires_four_layer_shadow"] is True
        assert dry_run.json()["preview"][
            "gray_zone_rule_decision_model_call_count"
        ] == 0
        assert db_session.scalar(select(RuntimeSettingsCandidate.id)) is None
        db_session.rollback()

        staged = client.post(
            "/api/settings/runtime-candidates",
            json={**request, "dry_run_only": False},
        )
        assert staged.status_code == 200
        candidate = staged.json()["candidate"]
        assert candidate["status"] == "staged"
        assert candidate["builds"][0]["status"] == "dry_run_passed"

        status = client.get(
            f"/api/settings/runtime-candidates/{candidate['id']}"
        )
        assert status.status_code == 200
        assert status.json()["candidate"]["candidate_hash"] == candidate[
            "candidate_hash"
        ]

        recreate = client.put(
            "/api/settings/model",
            json={"worker_concurrency": get_settings().worker_concurrency + 1},
        )
        assert recreate.status_code == 200
        assert recreate.json()["requires_service_recreate"] is True
        assert recreate.json()["active_mutated"] is False
