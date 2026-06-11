from __future__ import annotations

import uuid

import pytest

from app.services.cache_manager import clear_cache_manager
from app.services.embeddings import ChatProvider
from app.services.ingestion import create_knowledge_base_space
from app.services.profile_assistant import get_profile_assistant_state, stream_profile_assistant_events
from app.services.strategy_profiles import (
    bind_profile_to_knowledge_base,
    copy_profile,
    create_profile,
    default_profile_payload,
    delete_profile,
    ensure_knowledge_bases_have_profiles,
    get_active_profile_record,
    profile_hash,
    update_profile,
    use_strategy_profile,
    validate_profile_payload,
)


def custom_profile_payload() -> dict:
    payload = default_profile_payload()
    payload["library_type"] = "legal"
    payload["prompt_pack"]["answer_grounding_system"] = "Answer only from cited legal clauses."
    payload["schema_pack"] = {
        "entity_types": ["topic", "clause", "party", "obligation"],
        "relation_types": ["cites", "binds", "related_observation"],
        "entity_aliases": {"section": "clause", "concept": "topic"},
        "relation_aliases": {"references": "cites"},
        "default_entity_type": "topic",
        "default_relation_type": "related_observation",
        "disabled_entity_types": ["algorithm", "theorem", "code"],
        "disabled_relation_types": [],
    }
    return payload


def test_builtin_profile_auto_binds_existing_knowledge_base(db_session, sample_knowledge_base):
    builtin = ensure_knowledge_bases_have_profiles(db_session)
    db_session.commit()
    db_session.refresh(sample_knowledge_base)

    assert builtin.is_builtin is True
    assert sample_knowledge_base.active_profile_id == builtin.id

    active = get_active_profile_record(db_session, sample_knowledge_base.id)
    assert active.id == builtin.id
    assert active.profile_hash == profile_hash(default_profile_payload())
    assert "topic" in active.profile_json["schema_pack"]["entity_types"]


def test_custom_profile_crud_binding_and_builtin_protection(db_session, sample_knowledge_base):
    builtin = ensure_knowledge_bases_have_profiles(db_session)
    custom, warnings = create_profile(db_session, name="Legal Profile", library_type="legal", profile_json=custom_profile_payload())
    assert warnings == []

    with pytest.raises(ValueError):
        create_profile(db_session, name="Legal Profile", library_type="legal", profile_json=custom_profile_payload())

    copied = copy_profile(db_session, builtin.id, name="Editable KnowledgeBase Copy")
    assert copied.is_builtin is False

    with pytest.raises(ValueError):
        update_profile(db_session, builtin.id, name="mutated")
    with pytest.raises(ValueError):
        delete_profile(db_session, builtin.id)

    knowledge_base = bind_profile_to_knowledge_base(db_session, knowledge_base_id=sample_knowledge_base.id, profile_id=custom.id)
    db_session.commit()
    assert knowledge_base.active_profile_id == custom.id

    delete_profile(db_session, copied.id)
    db_session.commit()


def test_deleting_bound_custom_profile_rebinds_knowledge_base_to_default(db_session, sample_knowledge_base):
    builtin = ensure_knowledge_bases_have_profiles(db_session)
    custom, _warnings = create_profile(db_session, name="Deletable Bound Profile", library_type="legal", profile_json=custom_profile_payload())
    knowledge_base = bind_profile_to_knowledge_base(db_session, knowledge_base_id=sample_knowledge_base.id, profile_id=custom.id)
    db_session.commit()
    assert knowledge_base.active_profile_id == custom.id

    delete_profile(db_session, custom.id)
    db_session.refresh(sample_knowledge_base)
    db_session.refresh(custom)

    assert sample_knowledge_base.active_profile_id == builtin.id
    assert custom.is_active is False


def test_new_knowledge_base_space_uses_default_profile(db_session):
    builtin = ensure_knowledge_bases_have_profiles(db_session)
    knowledge_base = create_knowledge_base_space(db_session, name="Profile Default KnowledgeBase")

    assert knowledge_base.active_profile_id == builtin.id


def test_custom_schema_is_normalized_by_profile_validation(db_session, sample_knowledge_base):
    payload = custom_profile_payload()
    payload["schema_pack"]["entity_types"].append("Evidence Signal")
    payload["schema_pack"]["relation_types"].append("Observed Edge")
    validated, warnings = validate_profile_payload(payload)

    assert warnings == []
    assert "evidence_signal" in validated["schema_pack"]["entity_types"]
    assert "observed_edge" in validated["schema_pack"]["relation_types"]
    assert validated["schema_pack"]["entity_aliases"]["concept"] == "topic"
    assert validated["schema_pack"]["relation_aliases"]["references"] == "cites"

    custom, _warnings = create_profile(db_session, name="Legal Schema", library_type="legal", profile_json=payload)
    bind_profile_to_knowledge_base(db_session, knowledge_base_id=sample_knowledge_base.id, profile_id=custom.id)
    db_session.commit()

    with use_strategy_profile(custom.profile_json):
        active_schema = custom.profile_json["schema_pack"]
        assert active_schema["default_entity_type"] == "topic"
        assert active_schema["default_relation_type"] == "related_observation"


@pytest.mark.asyncio
async def test_profile_assistant_streams_profile_json_and_caches_state(monkeypatch):
    clear_cache_manager()
    draft = custom_profile_payload()
    session_id = f"unit-profile-assistant-{uuid.uuid4()}"

    async def fake_classify_json(self, system_prompt, user_prompt, fallback=None):
        assert "explanation" in system_prompt
        assert "profile_json" in user_prompt
        return {"explanation": "Updated for legal evidence sources.", "profile_json": draft}

    monkeypatch.setattr(ChatProvider, "classify_json", fake_classify_json)

    events = []
    async for event in stream_profile_assistant_events(
        prompt="Use legal evidence clauses.",
        session_id=session_id,
        base_profile_id="base-profile",
        base_profile=default_profile_payload(),
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
