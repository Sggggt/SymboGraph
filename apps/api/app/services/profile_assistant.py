from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from app.services.cache_manager import get_cache_manager
from app.services.embeddings import ChatProvider
from app.services.strategy_profiles import default_profile_payload, profile_hash, validate_profile_payload


PROFILE_ASSISTANT_CACHE_NAMESPACE = "profile_assistant"
PROFILE_ASSISTANT_CACHE_TTL_SECONDS = 86400


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_state(session_id: str, base_profile_id: str | None = None) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "base_profile_id": base_profile_id,
        "messages": [],
        "latest_profile_json": None,
        "latest_profile_hash": None,
        "warnings": [],
        "draft_message": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def get_profile_assistant_state(session_id: str) -> dict[str, Any]:
    cached = get_cache_manager().get_runtime_state(PROFILE_ASSISTANT_CACHE_NAMESPACE, session_id)
    if cached:
        return cached
    return _new_state(session_id)


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    get_cache_manager().set_runtime_state(
        PROFILE_ASSISTANT_CACHE_NAMESPACE,
        str(state["session_id"]),
        state,
        ttl=PROFILE_ASSISTANT_CACHE_TTL_SECONDS,
    )


def _stringify_profile(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, sort_keys=True)


def _assistant_system_prompt() -> str:
    return (
        "You are a user-profile interaction-configuration assistant for a local context-graph knowledge-base system. "
        "Return strict JSON only, with keys explanation and profile_json. "
        "explanation must be concise natural language describing prompt, UI-label, or conversation-preference changes and any boundary risks. "
        "profile_json must be a complete user_profile_v1 object using only schema_version, library_type, ui_labels, prompt_pack, and conversation_preferences. "
        "Profiles only affect interaction wording, answer style, clarification style, citation strictness expression, and no-context response text. "
        "Do not generate chunking, embedding, BM25, graph build, clustering, retrieval scoring, context-package budget, agent envelope, repair/verification budget, quality gate, policy, ontology, fallback, model, cache, database, vector-store, or runtime controls. "
        "If the user asks for engineering controls, mention in explanation that those belong in Runtime Settings and keep profile_json limited to user_profile_v1 interaction fields. "
        "Do not include markdown fences, API keys, secrets, or instructions to save automatically. "
        "Prefer the user's language for explanation."
    )


async def generate_profile_assistant_response(prompt: str, base_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    base = base_profile or default_profile_payload()
    user_prompt = (
        f"User request:\n{prompt.strip()}\n\n"
        "Base profile JSON:\n"
        f"{_stringify_profile(base)}\n\n"
        "Return JSON in this exact shape:\n"
        '{"explanation":"...","profile_json":{...}}'
    )
    fallback = {
        "explanation": "已基于当前 Profile 生成草案。请检查提示词、界面标签和对话偏好后再保存。",
        "profile_json": base,
    }
    result = await ChatProvider().classify_json(_assistant_system_prompt(), user_prompt, fallback=fallback)
    profile_candidate = result.get("profile_json") if isinstance(result, dict) else None
    if not isinstance(profile_candidate, dict):
        profile_candidate = base
    validated, warnings = validate_profile_payload(profile_candidate)
    explanation = result.get("explanation") if isinstance(result, dict) else None
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "已基于当前 Profile 生成草案。请检查提示词、界面标签和对话偏好后再保存。"
    return {
        "explanation": explanation.strip(),
        "profile_json": validated,
        "warnings": warnings,
        "profile_hash": profile_hash(validated),
    }


async def stream_profile_assistant_events(
    *,
    prompt: str,
    session_id: str | None = None,
    base_profile_id: str | None = None,
    base_profile: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    resolved_session_id = session_id or str(uuid.uuid4())
    state = get_profile_assistant_state(resolved_session_id)
    if base_profile_id:
        state["base_profile_id"] = base_profile_id
    state["messages"].append({"role": "user", "content": prompt.strip(), "created_at": _now_iso()})
    state["draft_message"] = ""
    _save_state(state)

    yield {"type": "meta", "session_id": resolved_session_id, "cached": True}
    result = await generate_profile_assistant_response(prompt, base_profile=base_profile)

    accumulated = ""
    for start in range(0, len(result["explanation"]), 8):
        token = result["explanation"][start : start + 8]
        accumulated += token
        state["draft_message"] = accumulated
        _save_state(state)
        yield {"type": "token", "token": token}
        await asyncio.sleep(0.012)

    state["latest_profile_json"] = result["profile_json"]
    state["latest_profile_hash"] = result["profile_hash"]
    state["warnings"] = result["warnings"]
    state["messages"].append(
        {
            "role": "assistant",
            "content": result["explanation"],
            "profile_json": result["profile_json"],
            "warnings": result["warnings"],
            "profile_hash": result["profile_hash"],
            "created_at": _now_iso(),
        }
    )
    state["draft_message"] = ""
    _save_state(state)

    yield {
        "type": "profile_json",
        "profile_json": result["profile_json"],
        "warnings": result["warnings"],
        "profile_hash": result["profile_hash"],
    }
    yield {"type": "final", "session_id": resolved_session_id, "state": state}
