from __future__ import annotations

import json
import uuid

import pytest

from app.models import GraphExtractionRun
from app.services.concept_graph import GraphExtractionPlan, create_graph_extraction_run_from_plan, normalize_entity_type, normalize_relation_type
from app.services.embeddings import ChatProvider
from app.services.ingestion import create_course_space
from app.services.strategy_profiles import (
    copy_profile,
    create_profile,
    default_course_profile_payload,
    delete_profile,
    ensure_courses_have_profiles,
    get_active_profile_record,
    bind_profile_to_course,
    profile_hash,
    update_profile,
    use_strategy_profile,
)
from app.services.profile_assistant import get_profile_assistant_state, stream_profile_assistant_events
from app.services.cache_manager import clear_cache_manager


def custom_profile_payload() -> dict:
    payload = default_course_profile_payload()
    payload["library_type"] = "legal"
    payload["prompt_pack"]["graph_extraction_system"] = "Extract legal clauses only. Return JSON with concepts and relations."
    payload["schema_pack"] = {
        "entity_types": ["clause", "party", "obligation"],
        "relation_types": ["cites", "binds", "related_to"],
        "entity_aliases": {"section": "clause"},
        "relation_aliases": {"references": "cites"},
        "default_entity_type": "clause",
        "default_relation_type": "related_to",
        "disabled_entity_types": ["algorithm", "theorem", "code"],
        "disabled_relation_types": [],
    }
    return payload


def test_builtin_profile_auto_binds_existing_course(db_session, sample_course):
    builtin = ensure_courses_have_profiles(db_session)
    db_session.commit()
    db_session.refresh(sample_course)

    assert builtin.is_builtin is True
    assert sample_course.active_profile_id == builtin.id

    active = get_active_profile_record(db_session, sample_course.id)
    assert active.id == builtin.id
    assert active.profile_hash == profile_hash(default_course_profile_payload())


def test_custom_profile_crud_binding_and_builtin_protection(db_session, sample_course):
    builtin = ensure_courses_have_profiles(db_session)
    custom, warnings = create_profile(db_session, name="Legal Profile", library_type="legal", profile_json=custom_profile_payload())
    assert any("does not include concept" in warning for warning in warnings)

    with pytest.raises(ValueError):
      create_profile(db_session, name="Legal Profile", library_type="legal", profile_json=custom_profile_payload())

    copied = copy_profile(db_session, builtin.id, name="Editable Course Copy")
    assert copied.is_builtin is False

    with pytest.raises(ValueError):
        update_profile(db_session, builtin.id, name="mutated")
    with pytest.raises(ValueError):
        delete_profile(db_session, builtin.id)

    course = bind_profile_to_course(db_session, course_id=sample_course.id, profile_id=custom.id)
    db_session.commit()
    assert course.active_profile_id == custom.id

    delete_profile(db_session, copied.id)
    db_session.commit()


def test_deleting_bound_custom_profile_rebinds_course_to_default(db_session, sample_course):
    builtin = ensure_courses_have_profiles(db_session)
    custom, _warnings = create_profile(db_session, name="Deletable Bound Profile", library_type="legal", profile_json=custom_profile_payload())
    course = bind_profile_to_course(db_session, course_id=sample_course.id, profile_id=custom.id)
    db_session.commit()
    assert course.active_profile_id == custom.id

    delete_profile(db_session, custom.id)
    db_session.refresh(sample_course)
    db_session.refresh(custom)

    assert sample_course.active_profile_id == builtin.id
    assert custom.is_active is False


def test_new_course_space_uses_default_profile(db_session):
    builtin = ensure_courses_have_profiles(db_session)
    course = create_course_space(db_session, name="Profile Default Course")

    assert course.active_profile_id == builtin.id


def test_custom_schema_drives_entity_and_relation_normalization(db_session, sample_course):
    custom, _warnings = create_profile(db_session, name="Legal Schema", library_type="legal", profile_json=custom_profile_payload())
    bind_profile_to_course(db_session, course_id=sample_course.id, profile_id=custom.id)
    db_session.commit()

    with use_strategy_profile(custom.profile_json):
        assert normalize_entity_type("algorithm") == "clause"
        assert normalize_entity_type("section") == "clause"
        assert normalize_relation_type("references") == "cites"
        assert normalize_relation_type("solves") == "related_to"


@pytest.mark.asyncio
async def test_custom_graph_prompt_replaces_course_extraction_prompt(db_session, sample_course, monkeypatch):
    custom, _warnings = create_profile(db_session, name="Legal Prompt", library_type="legal", profile_json=custom_profile_payload())
    bind_profile_to_course(db_session, course_id=sample_course.id, profile_id=custom.id)
    db_session.commit()
    captured: dict[str, object] = {}

    async def fake_post(self, payload):
        captured["payload"] = payload
        return json.dumps({"concepts": [], "relations": []})

    monkeypatch.setattr(ChatProvider, "_post_chat_text_with_response_format_fallback", fake_post)

    with use_strategy_profile(custom.profile_json):
        await ChatProvider().extract_graph_payload("Section 1. Party A must notify Party B.", chapter="S1", source_type="pdf")

    messages = captured["payload"]["messages"]  # type: ignore[index]
    system_prompt = messages[0]["content"]
    assert "Extract legal clauses only" in system_prompt
    assert "algorithms, theorems" not in system_prompt


def test_graph_extraction_run_records_strategy_profile_hash(db_session, sample_course, indexed_chunks):
    custom, _warnings = create_profile(db_session, name="Legal Run", library_type="legal", profile_json=custom_profile_payload())
    bind_profile_to_course(db_session, course_id=sample_course.id, profile_id=custom.id)
    _document, chunks = indexed_chunks
    plan = GraphExtractionPlan(
        selected_chunk_ids=[chunks[0].id],
        selected_reasons={chunks[0].id: {"reason": "test"}},
        skipped_reasons={},
        coverage={},
        budget={},
        stop_reason="test",
    )

    run = create_graph_extraction_run_from_plan(
        db_session,
        course_id=sample_course.id,
        batch_id=None,
        chunks=[chunks[0]],
        plan=plan,
        profile_version=None,
        strategy_profile_id=custom.id,
        strategy_profile_hash=custom.profile_hash,
    )
    db_session.commit()

    stored = db_session.get(GraphExtractionRun, run.id)
    assert stored.strategy_profile_id == custom.id
    assert stored.strategy_profile_hash == custom.profile_hash


@pytest.mark.asyncio
async def test_profile_assistant_streams_profile_json_and_caches_state(monkeypatch):
    clear_cache_manager()
    draft = custom_profile_payload()
    session_id = f"unit-profile-assistant-{uuid.uuid4()}"

    async def fake_classify_json(self, system_prompt, user_prompt, fallback=None):
        assert "explanation" in system_prompt
        assert "profile_json" in user_prompt
        return {"explanation": "已调整为法律资料库 Profile。", "profile_json": draft}

    monkeypatch.setattr(ChatProvider, "classify_json", fake_classify_json)

    events = []
    async for event in stream_profile_assistant_events(
        prompt="改成法律条款资料库",
        session_id=session_id,
        base_profile_id="base-profile",
        base_profile=default_course_profile_payload(),
    ):
        events.append(event)

    assert events[0] == {"type": "meta", "session_id": session_id, "cached": True}
    assert any(event.get("type") == "token" for event in events)
    profile_event = next(event for event in events if event.get("type") == "profile_json")
    assert profile_event["profile_json"]["library_type"] == "legal"
    assert "clause" in profile_event["profile_json"]["schema_pack"]["entity_types"]

    state = get_profile_assistant_state(session_id)
    assert state["base_profile_id"] == "base-profile"
    assert [message["role"] for message in state["messages"]] == ["user", "assistant"]
    assert state["latest_profile_hash"] == profile_event["profile_hash"]
    assert state["latest_profile_json"]["library_type"] == "legal"
    clear_cache_manager()
