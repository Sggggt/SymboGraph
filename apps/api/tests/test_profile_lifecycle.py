from __future__ import annotations

import copy

import pytest
from sqlalchemy import select


def _install_profile_side_effect_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_error: Exception | None = None,
) -> dict[str, list]:
    from app.services import cache_manager, runtime_settings

    calls: dict[str, list] = {"cache": [], "publish": []}

    class FakeCacheManager:
        def invalidate_knowledge_base(self, knowledge_base_id: str, *, strict: bool = False):
            calls["cache"].append((knowledge_base_id, strict))
            if cache_error is not None:
                raise cache_error
            return True

    def fake_publish(changed_keys, source="api", *, idempotency_key=None):
        calls["publish"].append(
            {
                "changed_keys": sorted(changed_keys),
                "source": source,
                "idempotency_key": idempotency_key,
            }
        )
        return {"version_hash": "f" * 64}

    monkeypatch.setattr(cache_manager, "get_cache_manager", lambda: FakeCacheManager())
    monkeypatch.setattr(runtime_settings, "publish_runtime_settings_version", fake_publish)
    return calls


def test_profile_lifecycle_diff_classifies_hot_rebuild_ui_and_preferences() -> None:
    from app.services.strategy_profiles import (
        default_profile_payload,
        profile_lifecycle_diff,
        validate_profile_payload,
    )

    before = default_profile_payload()
    after = copy.deepcopy(before)
    after["prompt_pack"]["answer_system_prefix"] = "Domain answer style."
    after["prompt_pack"]["mid_concept_definition_system"] = "Domain mid definition."
    after["ui_labels"]["knowledge_base"] = "Corpus"
    after["conversation_preferences"]["default_language"] = "zh"
    diff = profile_lifecycle_diff(before, after)

    assert diff["hot_prompt_keys"] == ["answer_system_prefix"]
    assert diff["concept_prompt_keys"] == ["mid_concept_definition_system"]
    assert diff["ui_label_keys"] == ["knowledge_base"]
    assert diff["conversation_preference_keys"] == ["default_language"]
    assert diff["hot_reload_required"] is True
    assert diff["concept_rebuild_required"] is True
    assert diff["cache_invalidation_required"] is True
    assert diff["active_graph_mutated"] is False
    assert diff["gray_zone_rule_inputs_modified"] is False
    assert diff["gray_zone_model_call_count"] == 0

    normalized, warnings = validate_profile_payload(
        {
            **after,
            "conversation_preferences": {
                "default_language": "invalid",
                "citation_strictness": "compact",
                "clarification_style": "detailed",
                "gray_rule_override": "continue",
            },
        }
    )
    assert normalized["conversation_preferences"] == {
        "default_language": "auto",
        "citation_strictness": "compact",
        "clarification_style": "detailed",
    }
    assert any("gray_rule_override" in warning for warning in warnings)
    assert any("default_language" in warning for warning in warnings)


def test_bound_profile_lifecycle_invalidates_cache_broadcasts_and_marks_only_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import ContextGraphFreshness, ContextGraphState, PromptProtocolVersion
    from app.services.strategy_profiles import (
        PROFILE_LIFECYCLE_PROTOCOL_VERSION,
        bind_profile_to_knowledge_base,
        create_profile,
        default_profile_payload,
        update_profile,
    )

    calls = _install_profile_side_effect_fakes(monkeypatch)
    state = ContextGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_scope_hash="1" * 64,
        structure_graph_hash="2" * 64,
        chunk_relation_graph_hash="3" * 64,
        rq_membership_hash="4" * 64,
        mid_concept_hash="5" * 64,
        coarse_concept_hash="6" * 64,
        context_graph_hash="7" * 64,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db_session.add(state)
    db_session.flush()
    freshness = ContextGraphFreshness(
        knowledge_base_id=sample_knowledge_base.id,
        context_graph_state_id=state.id,
        layer="mid_concepts",
        state_hash=state.mid_concept_hash,
        is_stale=False,
        stale_reasons_json=[],
        diagnostics_json={},
    )
    db_session.add(freshness)
    db_session.commit()

    profile_json = default_profile_payload()
    profile_json["prompt_pack"]["answer_system_prefix"] = "Hot profile guidance."
    profile, _warnings = create_profile(
        db_session,
        name="Lifecycle profile",
        library_type="custom",
        profile_json=profile_json,
    )
    bind_profile_to_knowledge_base(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        profile_id=profile.id,
    )

    events = list(
        db_session.scalars(
            select(PromptProtocolVersion)
            .where(
                PromptProtocolVersion.protocol_version
                == PROFILE_LIFECYCLE_PROTOCOL_VERSION
            )
            .order_by(PromptProtocolVersion.created_at.asc())
        ).all()
    )
    assert len(events) == 1
    assert events[0].state == "active"
    bind_lifecycle = events[0].prompt_pack_json["lifecycle"]
    assert bind_lifecycle["hot_reload_required"] is True
    assert bind_lifecycle["concept_rebuild_required"] is False
    db_session.refresh(state)
    assert "profile_concept_rebuild" not in state.diagnostics_json

    updated_json = copy.deepcopy(profile.profile_json)
    updated_json["prompt_pack"]["mid_concept_definition_system"] = (
        "Rebuild-only concept definition."
    )
    update_profile(db_session, profile.id, profile_json=updated_json)
    db_session.refresh(state)
    db_session.refresh(freshness)
    events = list(
        db_session.scalars(
            select(PromptProtocolVersion)
            .where(
                PromptProtocolVersion.protocol_version
                == PROFILE_LIFECYCLE_PROTOCOL_VERSION
            )
            .order_by(PromptProtocolVersion.created_at.asc())
        ).all()
    )
    assert len(events) == 2
    rebuild_event = events[-1]
    assert rebuild_event.state == "active"
    lifecycle = rebuild_event.prompt_pack_json["lifecycle"]
    assert lifecycle["concept_prompt_keys"] == ["mid_concept_definition_system"]
    assert lifecycle["active_graph_mutated"] is False
    marker = state.diagnostics_json["profile_concept_rebuild"]
    assert marker["status"] == "rebuild_required"
    assert marker["active_graph_state_id"] == state.id
    assert marker["lifecycle_event_id"] == rebuild_event.id
    assert state.context_graph_hash == "7" * 64
    assert state.state == "active"
    assert freshness.is_stale is False
    assert freshness.stale_reasons_json == []
    assert calls["cache"] == [
        (sample_knowledge_base.id, True),
        (sample_knowledge_base.id, True),
    ]
    assert all(call["source"] == "profile_lifecycle" for call in calls["publish"])
    assert calls["publish"][-1]["idempotency_key"] == rebuild_event.id


def test_profile_lifecycle_failure_is_durable_and_exact_retry_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import PromptProtocolVersion
    from app.services.strategy_profiles import (
        PROFILE_LIFECYCLE_PROTOCOL_VERSION,
        bind_profile_to_knowledge_base,
        create_profile,
        default_profile_payload,
        reconcile_pending_profile_lifecycle_events,
    )

    _install_profile_side_effect_fakes(
        monkeypatch, cache_error=ConnectionError("redis unavailable")
    )
    profile_json = default_profile_payload()
    profile_json["conversation_preferences"]["default_language"] = "zh"
    profile, _warnings = create_profile(
        db_session,
        name="Pending lifecycle profile",
        library_type="custom",
        profile_json=profile_json,
    )
    with pytest.raises(RuntimeError, match="remain pending"):
        bind_profile_to_knowledge_base(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            profile_id=profile.id,
        )

    db_session.expire_all()
    persisted_kb = db_session.get(type(sample_knowledge_base), sample_knowledge_base.id)
    event = db_session.scalar(
        select(PromptProtocolVersion).where(
            PromptProtocolVersion.protocol_version
            == PROFILE_LIFECYCLE_PROTOCOL_VERSION
        )
    )
    assert persisted_kb.active_profile_id == profile.id
    assert event is not None
    assert event.state == "pending_dispatch"
    immutable_hash = event.protocol_hash
    assert event.prompt_pack_json["delivery"]["last_error_type"] == "ConnectionError"

    calls = _install_profile_side_effect_fakes(monkeypatch)
    result = reconcile_pending_profile_lifecycle_events(
        db_session, limit=8, raise_on_error=True
    )
    db_session.refresh(event)
    assert result["dispatched_event_ids"] == [event.id]
    assert event.state == "active"
    assert event.protocol_hash == immutable_hash
    assert event.prompt_pack_json["delivery"]["attempt_count"] == 2
    assert calls["publish"][0]["idempotency_key"] == event.id

    second = reconcile_pending_profile_lifecycle_events(
        db_session, limit=8, raise_on_error=True
    )
    assert second["dispatched_event_ids"] == []
    assert len(calls["cache"]) == 1
    assert len(calls["publish"]) == 1


def test_older_pending_concept_event_accepts_exact_newer_persisted_marker(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import ContextGraphState, PromptProtocolVersion
    from app.services.strategy_profiles import (
        PROFILE_LIFECYCLE_PROTOCOL_VERSION,
        bind_profile_to_knowledge_base,
        create_profile,
        default_profile_payload,
        reconcile_pending_profile_lifecycle_events,
        update_profile,
    )

    state = ContextGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_scope_hash="1" * 64,
        structure_graph_hash="2" * 64,
        chunk_relation_graph_hash="3" * 64,
        rq_membership_hash="4" * 64,
        mid_concept_hash="5" * 64,
        coarse_concept_hash="6" * 64,
        context_graph_hash="7" * 64,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db_session.add(state)
    db_session.commit()
    _install_profile_side_effect_fakes(monkeypatch)
    profile, _warnings = create_profile(
        db_session,
        name="Superseding concept marker profile",
        library_type="general",
        profile_json=default_profile_payload(),
    )
    bind_profile_to_knowledge_base(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        profile_id=profile.id,
    )

    first_payload = copy.deepcopy(profile.profile_json)
    first_payload["prompt_pack"]["mid_concept_definition_system"] = (
        "First pending concept prompt."
    )
    _install_profile_side_effect_fakes(
        monkeypatch, cache_error=ConnectionError("redis unavailable")
    )
    with pytest.raises(RuntimeError, match="remain pending"):
        update_profile(db_session, profile.id, profile_json=first_payload)

    first_event = db_session.scalar(
        select(PromptProtocolVersion)
        .where(
            PromptProtocolVersion.protocol_version
            == PROFILE_LIFECYCLE_PROTOCOL_VERSION,
            PromptProtocolVersion.state == "pending_dispatch",
        )
        .order_by(PromptProtocolVersion.created_at.asc())
    )
    assert first_event is not None

    second_payload = copy.deepcopy(first_payload)
    second_payload["prompt_pack"]["mid_concept_definition_system"] = (
        "Second superseding concept prompt."
    )
    _install_profile_side_effect_fakes(monkeypatch)
    update_profile(db_session, profile.id, profile_json=second_payload)
    db_session.refresh(state)
    persisted_marker = state.diagnostics_json["profile_concept_rebuild"]
    assert persisted_marker["lifecycle_event_id"] != first_event.id

    result = reconcile_pending_profile_lifecycle_events(
        db_session, limit=8, raise_on_error=True
    )
    db_session.refresh(first_event)
    assert result["dispatched_event_ids"] == [first_event.id]
    assert first_event.state == "active"


def test_library_type_update_is_part_of_bound_profile_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import PromptProtocolVersion
    from app.services.strategy_profiles import (
        PROFILE_LIFECYCLE_PROTOCOL_VERSION,
        bind_profile_to_knowledge_base,
        create_profile,
        default_profile_payload,
        update_profile,
    )

    calls = _install_profile_side_effect_fakes(monkeypatch)
    profile, _warnings = create_profile(
        db_session,
        name="Library type lifecycle profile",
        library_type="general",
        profile_json=default_profile_payload(),
    )
    assert profile.library_type == "general"
    assert profile.profile_json["library_type"] == "general"
    bind_profile_to_knowledge_base(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        profile_id=profile.id,
    )
    previous_hash = profile.profile_hash

    updated, _warnings = update_profile(
        db_session,
        profile.id,
        library_type="legal",
    )
    assert updated.library_type == "legal"
    assert updated.profile_json["library_type"] == "legal"
    assert updated.profile_hash != previous_hash

    events = list(
        db_session.scalars(
            select(PromptProtocolVersion)
            .where(
                PromptProtocolVersion.protocol_version
                == PROFILE_LIFECYCLE_PROTOCOL_VERSION
            )
            .order_by(PromptProtocolVersion.created_at.asc())
        ).all()
    )
    assert len(events) == 2
    lifecycle = events[-1].prompt_pack_json["lifecycle"]
    assert lifecycle["changed_paths"] == ["library_type"]
    assert lifecycle["library_type_changed"] is True
    assert lifecycle["hot_reload_required"] is True
    assert lifecycle["concept_rebuild_required"] is False
    assert calls["cache"] == [
        (sample_knowledge_base.id, True),
        (sample_knowledge_base.id, True),
    ]
    assert calls["publish"][-1]["idempotency_key"] == events[-1].id


def test_profile_lifecycle_rejects_synchronized_card_hash_forgery(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from sqlalchemy.orm.attributes import flag_modified

    from app.models import PromptProtocolVersion
    from app.services import strategy_profiles

    _install_profile_side_effect_fakes(monkeypatch)
    profile, _warnings = strategy_profiles.create_profile(
        db_session,
        name="Lifecycle tamper profile",
        library_type="general",
        profile_json=strategy_profiles.default_profile_payload(),
    )
    strategy_profiles.bind_profile_to_knowledge_base(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        profile_id=profile.id,
    )
    event = db_session.scalar(
        select(PromptProtocolVersion).where(
            PromptProtocolVersion.protocol_version
            == strategy_profiles.PROFILE_LIFECYCLE_PROTOCOL_VERSION
        )
    )
    assert event is not None

    payload = copy.deepcopy(event.prompt_pack_json)
    payload["lifecycle"]["changed_paths"] = ["prompt_pack.forged"]
    forged_hash = strategy_profiles._canonical_hash(
        {
            "lifecycle": payload["lifecycle"],
            "replay_inputs": payload["replay_inputs"],
        }
    )
    payload["lifecycle_hash"] = forged_hash
    event.protocol_hash = forged_hash
    event.prompt_pack_json = payload
    flag_modified(event, "prompt_pack_json")
    db_session.flush()

    with pytest.raises(RuntimeError, match="frozen-input replay"):
        strategy_profiles._validate_profile_lifecycle_event(db_session, event)


def test_builtin_default_prompt_upgrade_uses_same_bound_kb_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import PromptProtocolVersion
    from app.services import strategy_profiles

    calls = _install_profile_side_effect_fakes(monkeypatch)
    builtin = strategy_profiles.ensure_builtin_default_profile(db_session)
    sample_knowledge_base.active_profile_id = builtin.id
    db_session.commit()
    monkeypatch.setitem(
        strategy_profiles.DEFAULT_PROFILE["prompt_pack"],
        "answer_system_prefix",
        "Upgraded code-owned default answer guidance.",
    )

    result = strategy_profiles.reconcile_builtin_default_profile_startup()
    event = db_session.scalar(
        select(PromptProtocolVersion)
        .where(
            PromptProtocolVersion.protocol_version
            == strategy_profiles.PROFILE_LIFECYCLE_PROTOCOL_VERSION
        )
        .order_by(PromptProtocolVersion.created_at.desc())
    )
    assert event is not None
    assert result["lifecycle_event_ids"] == [event.id]
    assert event.state == "active"
    lifecycle = event.prompt_pack_json["lifecycle"]
    assert lifecycle["mutation"] == "builtin_default_profile_upgraded"
    assert lifecycle["hot_prompt_keys"] == ["answer_system_prefix"]
    assert lifecycle["concept_rebuild_required"] is False
    assert calls["cache"] == [(sample_knowledge_base.id, True)]


@pytest.mark.asyncio
async def test_concept_prompt_marker_keeps_active_graph_immutable_and_searchable(
    monkeypatch: pytest.MonkeyPatch,
    db_session,
    populated_context_graph,
    fake_model_stack,
) -> None:
    from app.schemas import SearchFilters
    from app.services.context_graph import layered_search
    from app.services.strategy_profiles import (
        bind_profile_to_knowledge_base,
        create_profile,
        default_profile_payload,
    )

    _install_profile_side_effect_fakes(monkeypatch)
    knowledge_base = populated_context_graph["knowledge_base"]
    state = populated_context_graph["state"]
    original_state_id = state.id
    original_context_hash = state.context_graph_hash
    profile_json = default_profile_payload()
    profile_json["prompt_pack"]["coarse_concept_definition_system"] = (
        "Future shadow-build coarse prompt."
    )
    profile, _warnings = create_profile(
        db_session,
        name="Concept marker profile",
        library_type="custom",
        profile_json=profile_json,
    )
    bind_profile_to_knowledge_base(
        db_session,
        knowledge_base_id=knowledge_base.id,
        profile_id=profile.id,
    )
    db_session.refresh(state)
    marker = state.diagnostics_json["profile_concept_rebuild"]
    assert marker["status"] == "rebuild_required"
    assert state.id == original_state_id
    assert state.context_graph_hash == original_context_hash

    result = await layered_search(
        db_session,
        knowledge_base.id,
        "How does a Markov blanket support conditional independence?",
        SearchFilters(),
        4,
    )
    assert result.results
    assert result.trace.diagnostics_json["active_profile_hash"] == profile.profile_hash
    assert result.trace.convergence_json["gray_zone_model_call_count"] == 0
    assert db_session.get(type(state), original_state_id).context_graph_hash == (
        original_context_hash
    )


def test_conversation_preferences_change_language_and_safe_prompt_guidance() -> None:
    from app.services.agent_graph import evidence_insufficient_answer
    from app.services.embeddings import (
        ChatProvider,
        no_context_answer,
        prefers_chinese_answer,
    )
    from app.services.strategy_profiles import (
        default_profile_payload,
        use_strategy_profile,
    )

    profile = default_profile_payload()
    profile["conversation_preferences"] = {
        "default_language": "zh",
        "citation_strictness": "compact",
        "clarification_style": "detailed",
    }
    with use_strategy_profile(profile):
        assert prefers_chinese_answer("English-only question") is True
        assert no_context_answer("English-only question") == profile["prompt_pack"][
            "no_context_answer_zh"
        ]
        clarification = evidence_insufficient_answer(
            "English-only question", "insufficient_corpus"
        )
        bundle = ChatProvider()._answer_prompt_bundle(
            "English-only question", context_quality="normal"
        )
    assert "资料来源、章节范围、比较对象" in clarification
    assert bundle["target_language"] == "Chinese"
    system = bundle["system_content"]
    assert "keep citation wording compact" in system
    assert "make clarification requests detailed and actionable" in system
    assert "cannot relax the immutable grounding envelope" in system
