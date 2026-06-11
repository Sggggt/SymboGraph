from __future__ import annotations

import hashlib
import json
from typing import Any

from app.services.cache_manager import get_cache_manager
from app.services.embeddings import ChatProvider


QUALITY_JUDGE_PROMPT_VERSION = "quality_judge_v1"


def quality_judge_cache_key(*, knowledge_base_id: str, profile_version: str | None, target_type: str, candidate: dict[str, Any], model: str) -> str:
    payload = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return ":".join([knowledge_base_id, profile_version or "no-profile", target_type, model, QUALITY_JUDGE_PROMPT_VERSION, digest])


class QualityJudge:
    def __init__(self, provider: ChatProvider | None = None) -> None:
        self.provider = provider or ChatProvider()

    async def judge(self, *, knowledge_base_id: str, profile: dict[str, Any] | None, target_type: str, candidate: dict[str, Any]) -> dict[str, Any]:
        cache = get_cache_manager()
        profile_version = (profile or {}).get("schema_version") or (profile or {}).get("version")
        cache_key = quality_judge_cache_key(
            knowledge_base_id=knowledge_base_id,
            profile_version=profile_version,
            target_type=target_type,
            candidate=candidate,
            model=self.provider.settings.chat_model,
        )
        cached = cache.get_quality_judgment(cache_key)
        if isinstance(cached, dict):
            return {**cached, "cache_key": cache_key, "cached": True}

        system_prompt = "You are a strict structured quality judge for a KnowledgeBase knowledge graph pipeline. Return JSON only."
        user_prompt = (
            "Judge whether the candidate should be accepted, rejected, or kept as candidate_only. "
            "Return keys: action, score, reasons. Use action accept/reject/candidate_only/defer.\n\n"
            f"target_type={target_type}\n"
            f"profile={json.dumps(profile or {}, ensure_ascii=False)[:4000]}\n"
            f"candidate={json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)[:6000]}"
        )
        result = await self.provider.classify_json(system_prompt, user_prompt, fallback={})
        if not isinstance(result, dict):
            result = {"action": "defer", "score": 0.0, "reasons": ["invalid_judge_response"]}
        cache.set_quality_judgment(cache_key, result)
        return {**result, "cache_key": cache_key, "cached": False}
