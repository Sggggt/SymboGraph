from __future__ import annotations

import hashlib
import asyncio
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.concurrency import model_request_slot
from app.core.config import get_settings
from app.services.strategy_profiles import (
    DEFAULT_ANSWER_SYSTEM_PREFIX,
    DEFAULT_CONTEXT_LABEL,
    DEFAULT_NO_CONTEXT_EN,
    DEFAULT_NO_CONTEXT_ZH,
    active_profile_json,
    profile_prompt,
)


class FallbackDisabledError(RuntimeError):
    pass


def prefers_chinese_answer(text: str) -> bool:
    from app.services.chinese_text import contains_chinese

    return contains_chinese(text)


def answer_language_name(text: str) -> str:
    return "Chinese" if prefers_chinese_answer(text) else "English"


def no_context_answer(question: str) -> str:
    profile = active_profile_json()
    if prefers_chinese_answer(question):
        return profile_prompt(profile, "no_context_answer_zh", DEFAULT_NO_CONTEXT_ZH)
    return profile_prompt(profile, "no_context_answer_en", DEFAULT_NO_CONTEXT_EN)


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def validate_embedding_vectors(vectors: list[list[float]], *, expected_count: int, expected_dimensions: int) -> None:
    if len(vectors) != expected_count:
        raise RuntimeError(f"Embedding response returned {len(vectors)} vector(s), expected {expected_count}")
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimensions:
            raise RuntimeError(f"Embedding vector {index} has dimension {len(vector)}, expected {expected_dimensions}")
        if not all(math.isfinite(float(value)) for value in vector):
            raise RuntimeError(f"Embedding vector {index} contains non-finite values")
        if vector_norm(vector) <= 1e-12:
            raise RuntimeError(f"Embedding vector {index} is all zeros")


def is_degraded_mode() -> bool:
    settings = get_settings()
    return not settings.openai_api_key or not settings.embedding_api_key or not settings.embedding_base_url


def _exception_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.text or str(exc)
        except Exception:
            return str(exc)
    return str(exc)


def _is_unsupported_parameter_error(exc: Exception, parameter_name: str) -> bool:
    message = _exception_message(exc).lower()
    if parameter_name.lower() not in message:
        return False
    return any(
        marker in message
        for marker in (
            "invalidparameter",
            "invalid parameter",
            "unsupported",
            "not support",
            "not supported",
            "unknown parameter",
            "unrecognized",
            "extra inputs",
        )
    )


class EmbeddingProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def embed_texts(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        return (await self.embed_texts_with_meta(texts, text_type=text_type)).vectors

    async def embed_texts_with_meta(self, texts: list[str], text_type: str = "document") -> "EmbeddingCallResult":
        if not texts:
            return EmbeddingCallResult(vectors=[], provider="none", external_called=False, fallback_reason=None)
        if not self.settings.embedding_base_url:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("EMBEDDING_BASE_URL is required because ENABLE_MODEL_FALLBACK is false")
            return EmbeddingCallResult(
                vectors=[self._fake_embedding(text) for text in texts],
                provider="fake",
                external_called=False,
                fallback_reason="missing_embedding_base_url",
            )
        if not self.settings.embedding_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("EMBEDDING_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return EmbeddingCallResult(
                vectors=[self._fake_embedding(text) for text in texts],
                provider="fake",
                external_called=False,
                fallback_reason="missing_embedding_api_key",
            )
        try:
            vectors = await self._openai_compatible_embeddings(texts, text_type=text_type)
            validate_embedding_vectors(vectors, expected_count=len(texts), expected_dimensions=self.settings.embedding_dimensions)
            return EmbeddingCallResult(vectors=vectors, provider="openai_compatible", external_called=True, fallback_reason=None)
        except Exception as exc:
            if not self.settings.enable_model_fallback:
                raise
            return EmbeddingCallResult(
                vectors=[self._fake_embedding(text) for text in texts],
                provider="fake",
                external_called=True,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

    async def _openai_compatible_embeddings(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        batch_size = self.settings.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self._openai_compatible_embeddings_batch(texts[start : start + batch_size], text_type=text_type))
        return vectors

    async def _openai_compatible_embeddings_batch(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self.settings.embedding_dimensions,
        }
        base_url = self.settings.embedding_base_url.rstrip("/")
        api_key = self.settings.embedding_api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            data = await post_openai_compatible_json(
                f"{base_url}/embeddings",
                payload,
                headers,
                timeout=60.0,
                resolve_ip=self.settings.embedding_resolve_ip,
            )
        except Exception as exc:
            if not _is_unsupported_parameter_error(exc, "dimensions"):
                raise
            payload.pop("dimensions", None)
            data = await post_openai_compatible_json(
                f"{base_url}/embeddings",
                payload,
                headers,
                timeout=60.0,
                resolve_ip=self.settings.embedding_resolve_ip,
            )
        return [item["embedding"] for item in data["data"]]

    def _fake_embedding(self, text: str) -> list[float]:
        vector = []
        for idx in range(self.settings.embedding_dimensions):
            digest = hashlib.sha256(f"{idx}:{text}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:4], "big") / 2**32
            vector.append((value * 2.0) - 1.0)
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]


@dataclass(frozen=True)
class EmbeddingCallResult:
    vectors: list[list[float]]
    provider: str
    external_called: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ChatCallResult:
    answer: str
    provider: str
    model: str
    external_called: bool
    fallback_reason: str | None = None


class ChatProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def answer_question(self, question: str, contexts: list[dict], history: list[dict] | None = None) -> str:
        return (await self.answer_question_with_meta(question, contexts, history)).answer

    async def answer_question_with_meta(self, question: str, contexts: list[dict], history: list[dict] | None = None, evidence_quality: str = "normal") -> ChatCallResult:
        if not contexts:
            return ChatCallResult(
                answer=no_context_answer(question),
                provider="none",
                model=self.settings.chat_model,
                external_called=False,
                fallback_reason="no_contexts",
            )
        if not self.settings.openai_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("OPENAI_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return ChatCallResult(
                answer=self._extractive_answer(question, contexts),
                provider="extractive_fallback",
                model="local_extractive_template",
                external_called=False,
                fallback_reason="missing_openai_api_key",
            )
        try:
            answer = await self._openai_compatible_chat(question, contexts, history or [], evidence_quality=evidence_quality)
            return ChatCallResult(
                answer=answer,
                provider="openai_compatible_chat",
                model=self.settings.chat_model,
                external_called=True,
                fallback_reason=None,
            )
        except Exception as exc:
            if not self.settings.enable_model_fallback:
                raise
            return ChatCallResult(
                answer=self._extractive_answer(question, contexts),
                provider="extractive_fallback",
                model="local_extractive_template",
                external_called=True,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

    async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("OPENAI_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            if fallback is None:
                raise FallbackDisabledError("OPENAI_API_KEY is required (no fallback provided)")
            return fallback
        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            return await self._post_chat_json_with_response_format_fallback(payload)
        except Exception as exc:
            if not self.settings.enable_model_fallback:
                raise
            if fallback is None:
                raise
            return fallback

    async def rewrite_question(self, question: str, history: list[dict] | None = None) -> str:
        if not self.settings.openai_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("OPENAI_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return question
        history_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in (history or [])[-6:])
        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Rewrite the user question as a concise standalone retrieval query. Return only the rewritten query.",
                },
                {"role": "user", "content": f"History:\n{history_text}\n\nQuestion:\n{question}"},
            ],
            "temperature": 0.0,
        }
        try:
            return (await self._post_chat_text(payload)).strip() or question
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return question

    async def reflect_answer(self, question: str, answer: str, contexts: list[dict]) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("OPENAI_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return {"has_issue": False, "issue_type": "none", "suggestion": ""}
        context_text = "\n\n".join(
            f"[{i+1}] {ctx.get('document_title', '')}\n{ctx.get('content', '')[:600]}"
            for i, ctx in enumerate(contexts)
        )
        profile = active_profile_json()
        reflection_domain = profile_prompt(profile, "reflection_domain", "KnowledgeBase knowledge-base assistant")
        citation_domain = profile_prompt(profile, "citation_domain", "KnowledgeBase excerpts")
        system_prompt = (
            f"You are a strict quality reviewer for a {reflection_domain}. "
            f"Evaluate whether the assistant's answer is fully supported by the provided {citation_domain}. "
            "Return ONLY a JSON object with keys: has_issue (boolean), issue_type (one of: none, hallucination, insufficient_coverage, contradiction), suggestion (string)."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"{citation_domain}:\n{context_text}\n\n"
            "Check: 1) Does the answer contain claims not found in the excerpts? 2) Is the question fully answered? 3) Are there contradictions between the answer and excerpts?"
        )
        payload = {
            "model": self.settings.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            result = await self._post_chat_json_with_response_format_fallback(payload)
            return {
                "has_issue": bool(result.get("has_issue")),
                "issue_type": str(result.get("issue_type", "none")),
                "suggestion": str(result.get("suggestion", "")),
            }
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return {"has_issue": False, "issue_type": "none", "suggestion": ""}

    async def verify_citations(self, answer: str, citations: list[dict], contexts: list[dict]) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("OPENAI_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return {"verified": True, "unverified_indices": []}
        if not citations or not contexts:
            return {"verified": True, "unverified_indices": []}
        context_map = {ctx.get("chunk_id"): ctx for ctx in contexts}
        claims: list[str] = []
        for sentence in re.split(r'(?<=[.!?。！？])\s+', answer):
            if any(str(c.get("index", idx+1)) in sentence for idx, c in enumerate(citations)):
                claims.append(sentence.strip())
        if not claims:
            return {"verified": True, "unverified_indices": []}
        # Sample at most N claims
        sample = claims[: self.settings.citation_verification_sample_max]
        context_text = "\n\n".join(
            f"[{c.get('index', i+1)}] {context_map.get(c.get('chunk_id'), {}).get('document_title', '')}\n{context_map.get(c.get('chunk_id'), {}).get('content', '')[:400]}"
            for i, c in enumerate(citations)
            if c.get("chunk_id") in context_map
        )
        profile = active_profile_json()
        citation_domain = profile_prompt(profile, "citation_domain", "KnowledgeBase excerpts")
        system_prompt = (
            f"You verify whether specific claims in an answer are supported by cited {citation_domain}. "
            "Return ONLY a JSON object with keys: verified (boolean), unverified_indices (list of citation numbers that are NOT supported)."
        )
        user_prompt = (
            f"Claims to verify:\n" + "\n".join(f"- {claim}" for claim in sample) + "\n\n"
            f"Cited excerpts:\n{context_text}\n\n"
            "For each claim, check if it is directly supported by the cited excerpt. List unsupported citation numbers."
        )
        payload = {
            "model": self.settings.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            result = await self._post_chat_json_with_response_format_fallback(payload)
            return {
                "verified": bool(result.get("verified", True)),
                "unverified_indices": list(result.get("unverified_indices", [])),
            }
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return {"verified": True, "unverified_indices": []}

    async def perceive_question(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        """Perceive user intent, extract entities, and decompose the question.

        Returns a dict with keys:
        - intent: one of definition, comparison, application, procedure, analysis, unknown
        - entities: list of source-grounded evidence signals found in the question
        - sub_queries: list of sub-questions if multi-hop
        - needs_graph: whether graph search is likely helpful
        - suggested_strategy: one of global_dense, local_graph, hybrid, community
        """
        if not self.settings.openai_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("OPENAI_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return {
                "intent": "unknown",
                "entities": [],
                "sub_queries": [question],
                "needs_graph": False,
                "suggested_strategy": "hybrid",
            }
        history_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in (history or [])[-4:])
        profile = active_profile_json()
        perception_domain = profile_prompt(profile, "perception_domain", "evidence-grounded knowledge-base agent")
        entity_label = profile_prompt(profile, "entity_label", "source-grounded evidence signals")
        system_prompt = (
            f"You are a perception module for a {perception_domain}. "
            "Analyze the user's question and return ONLY a JSON object with these exact keys:\n"
            "- intent: one of [definition, comparison, application, procedure, analysis, unknown]\n"
            f"- entities: list of {entity_label} explicitly mentioned or implied in the question\n"
            "- sub_queries: list of simpler sub-questions if the original is complex/multi-hop; otherwise [original_question]\n"
            "- needs_graph: boolean, true if the question asks about observed connections, comparisons, dependencies, or derivations across evidence\n"
            "- suggested_strategy: one of [global_dense, local_graph, hybrid, community]\n"
            "  * global_dense: simple definition, formula, or single-fact lookup\n"
            "  * local_graph: question centers around specific evidence signals and observed connections\n"
            "  * hybrid: multi-aspect or comparison questions\n"
            "  * community: broad summary or overview questions\n"
        )
        user_prompt = (
            f"History:\n{history_text}\n\nQuestion:\n{question}\n\n"
            "Analyze this question and output the JSON perception result."
        )
        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        default_result = {
            "intent": "unknown",
            "entities": [],
            "sub_queries": [question],
            "needs_graph": False,
            "suggested_strategy": "hybrid",
        }
        try:
            result = await self._post_chat_json_with_response_format_fallback(payload)
            return {
                "intent": str(result.get("intent", "unknown")).lower(),
                "entities": list(result.get("entities", [])),
                "sub_queries": list(result.get("sub_queries", [question])),
                "needs_graph": bool(result.get("needs_graph", False)),
                "suggested_strategy": str(result.get("suggested_strategy", "hybrid")).lower(),
            }
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return default_result

    async def _openai_compatible_chat(self, question: str, contexts: list[dict], history: list[dict], evidence_quality: str = "normal") -> str:
        target_language = answer_language_name(question)
        citations = "\n\n".join(
            f"[{idx + 1}] {item['document_title']} / {item.get('partition') or 'General'}\n{item['content']}"
            for idx, item in enumerate(contexts)
        )
        profile = active_profile_json()
        context_label = profile_prompt(profile, "context_label", DEFAULT_CONTEXT_LABEL)
        context_label_lower = context_label[:1].lower() + context_label[1:] if context_label else "excerpts"
        coverage_label = profile_prompt(profile, "coverage_label", "KnowledgeBase materials")
        indexed_coverage_label = profile_prompt(profile, "indexed_coverage_label", "indexed KnowledgeBase materials")
        messages = [
            {
                "role": "system",
                "content": (
                    profile_prompt(profile, "answer_system_prefix", DEFAULT_ANSWER_SYSTEM_PREFIX)
                    + f"Answer only from the supplied {context_label_lower} and do not invent unsupported facts. "
                    "Keep the answer direct, concise, and say when the evidence is insufficient. "
                    "Always follow the required answer language below. "
                    "Do not infer the answer language from the retrieved excerpts. "
                    f"Required answer language: {target_language}. "
                    "The supplied excerpts may be Chinese, English, or mixed; use them as evidence, "
                    "but do not switch the answer language to match the excerpt language. "
                    "Format the answer as clean GitHub-flavored Markdown. "
                    "When writing mathematical notation, use valid LaTeX only: inline variables and short expressions "
                    "must be wrapped in single dollar delimiters like $k_i$ and $n - 1$; important equations must be "
                    "placed in display math blocks using double dollar delimiters on their own lines. "
                    "Never write formulas as glued plain text such as n-1ki, k_iin, or C(i)=n-1ki. "
                    "Use LaTeX commands and braces for fractions, superscripts, subscripts, and named variants, for example "
                    "$$C_D(i) = \\frac{k_i}{n - 1}$$, $k_i^{\\text{in}}$, and $k_i^{\\text{out}}$. "
                    "Do not repeat the same formula in both prose and math form; write the equation once, then explain "
                    "each symbol in separate bullets or sentences. "
                    + (
                        "IMPORTANT: The retrieved excerpts may have low relevance to the question. "
                        f"If they do not contain information that directly answers the question, clearly state that the {coverage_label} "
                        "do not cover this topic, and do NOT force citations from irrelevant excerpts. "
                        "You may provide a brief conceptual answer based on general knowledge, but explicitly note that it is not "
                        f"supported by the {indexed_coverage_label}."
                        if evidence_quality == "low"
                        else "If the supplied excerpts do not contain information that directly answers the question, "
                        f"clearly state that the {coverage_label} do not cover this topic and do NOT force citations."
                    )
                ),
            },
            *history,
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Required answer language: {target_language}\n\n"
                    f"{context_label}:\n"
                    f"{citations}\n\n"
                    + (
                        "Note: the above excerpts have been assessed as potentially irrelevant. "
                        "Only cite them if they truly support a specific claim in your answer. "
                        f"If none are relevant, answer without citations and note the lack of {coverage_label} coverage.\n\n"
                        if evidence_quality == "low"
                        else ""
                    )
                    + "Before finalizing, check that every formula is either inline LaTeX or display LaTeX, "
                    "and that variables are not attached to neighboring words."
                ),
            },
        ]
        payload = {"model": self.settings.chat_model, "messages": messages, "temperature": 0.2}
        return await self._post_chat_text(payload)

    async def _post_chat_text(self, payload: dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        data = await post_openai_compatible_json(
            f"{self.settings.chat_base_url.rstrip('/')}/chat/completions",
            payload,
            headers,
            timeout=float(self.settings.model_request_timeout_seconds),
            resolve_ip=self.settings.chat_resolve_ip,
        )
        return self._normalize_chat_content(data)

    async def _post_chat_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._parse_json_object(await self._post_chat_text(payload))

    async def _post_chat_json_with_response_format_fallback(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for candidate in self._response_format_candidates(payload.get("response_format")):
            candidate_payload = self._payload_with_response_format(payload, candidate)
            try:
                return await self._post_chat_json(candidate_payload)
            except Exception as exc:
                last_error = exc
                if not self._is_unsupported_response_format_error(exc):
                    raise
        raise RuntimeError(f"Chat JSON request failed after response_format fallback: {last_error}")

    def _response_format_candidates(self, response_format: dict[str, Any] | None) -> list[dict[str, Any] | None]:
        if not response_format:
            return [None]
        response_type = response_format.get("type")
        if response_type == "json_schema":
            return [response_format, {"type": "json_object"}, None]
        if response_type == "json_object":
            return [response_format, None]
        return [response_format, None]

    def _payload_with_response_format(self, payload: dict[str, Any], response_format: dict[str, Any] | None) -> dict[str, Any]:
        candidate = dict(payload)
        if response_format is None:
            candidate.pop("response_format", None)
            messages = [dict(item) for item in candidate.get("messages", [])]
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0].get('content', '')}\nReturn only a valid JSON object. Do not use markdown fences."
            else:
                messages.insert(0, {"role": "system", "content": "Return only a valid JSON object. Do not use markdown fences."})
            candidate["messages"] = messages
        else:
            candidate["response_format"] = response_format
        return candidate

    def _normalize_chat_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Chat response did not contain choices")
        choice = choices[0]
        if isinstance(choice.get("text"), str):
            return choice["text"]
        message = choice.get("message") or {}
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise RuntimeError(f"Chat response refusal: {refusal.strip()}")
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                return content
            raise RuntimeError("Chat response content is empty")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(text, dict) and isinstance(text.get("value"), str):
                        parts.append(text["value"])
                    elif isinstance(part.get("content"), str):
                        parts.append(part["content"])
            normalized = "".join(parts).strip()
            if normalized:
                return normalized
        raise RuntimeError("Chat response did not contain text content")

    def _is_unsupported_structured_output_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "json_schema" in message and any(
            marker in message
            for marker in (
                "response_format",
                "invalid",
                "unsupported",
                "not support",
                "not supported",
                "invalidparameter",
                "invalid parameter",
            )
        )

    def _is_unsupported_response_format_error(self, exc: Exception) -> bool:
        message = _exception_message(exc).lower()
        if "response_format" not in message and "json_schema" not in message and "json_object" not in message:
            return False
        return any(
            marker in message
            for marker in (
                "invalid",
                "unsupported",
                "not support",
                "not supported",
                "unknown parameter",
                "unrecognized",
                "invalidparameter",
                "invalid parameter",
                "extra inputs",
            )
        )

    def _extractive_answer(self, question: str, contexts: list[dict]) -> str:
        lead = next((item for item in contexts if item.get("metadata", {}).get("content_kind") != "code"), contexts[0])
        profile = active_profile_json()
        strong_source_label = profile_prompt(profile, "strongest_source_label", "knowledge-base source")
        strong_source_label_zh = profile_prompt(profile, "strongest_source_label_zh", "source")
        relevant_section_label = profile_prompt(profile, "relevant_section_label", "the relevant section")
        relevant_section_label_zh = profile_prompt(profile, "relevant_section_label_zh", "section")
        if prefers_chinese_answer(question):
            lines = [
                f"{strong_source_label_zh}: {lead['document_title']} / {lead.get('partition') or relevant_section_label_zh}.",
                lead["snippet"],
            ]
            if len(contexts) > 1:
                lines.append("Other retrieved excerpts provide related background; use the citations to inspect the source material.")
            lines.append(f"Question: {question}")
            return "\n".join(lines)
        lines = [
            f"The strongest {strong_source_label} is {lead['document_title']} in {lead.get('partition') or relevant_section_label}.",
            lead["snippet"],
        ]
        if len(contexts) > 1:
            lines.append("Other retrieved excerpts provide related background; use the citations to inspect the source material.")
        lines.append(f"Question: {question}")
        return "\n".join(lines)

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}


async def post_openai_compatible_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    resolve_ip: str | None = None,
) -> dict[str, Any]:
    normalized_resolve_ip = (resolve_ip or "").strip()
    if normalized_resolve_ip.lower() in {"", "none", "null", "__none__"}:
        normalized_resolve_ip = ""
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with model_request_slot():
                if normalized_resolve_ip:
                    return await asyncio.to_thread(_post_json_with_curl_resolve, url, payload, headers, timeout, normalized_resolve_ip)
                async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code >= 400:
                        raise httpx.HTTPStatusError(
                            f"OpenAI-compatible request failed with HTTP {response.status_code}: {response.text}",
                            request=response.request,
                            response=response,
                        )
                    return response.json()
        except Exception as exc:
            last_error = exc
            if attempt >= 3 or not _is_retryable_openai_error(exc):
                raise
            await asyncio.sleep(float(attempt))
    raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    message = str(exc).lower()
    non_retryable_markers = ("invalidparameter", "invalid parameter", "unauthorized", "forbidden", "401", "403")
    if any(marker in message for marker in non_retryable_markers):
        return False
    retryable_markers = (
        "timeout",
        "timed out",
        "operation timed out",
        "failed to connect",
        "could not connect",
        "connection reset",
        "handshake",
        "schannel",
        "ssl/tls",
        "temporarily unavailable",
        "curl: (28)",
        "curl: (7)",
        "curl: (35)",
        "curl: (52)",
        "empty reply",
    )
    return any(marker in message for marker in retryable_markers)


def _post_json_with_curl_resolve(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    resolve_ip: str,
) -> dict[str, Any]:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"Invalid OpenAI-compatible URL: {url}")
    _cleanup_private_temp_dirs("symbograph-openai-", max_age_seconds=3600)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    temp_dir = tempfile.mkdtemp(prefix="symbograph-openai-")
    try:
        os.chmod(temp_dir, 0o700)
    except OSError:
        pass
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json", dir=temp_dir) as payload_file:
            json.dump(payload, payload_file, ensure_ascii=False)
            payload_path = payload_file.name
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".curl", dir=temp_dir) as config_file:
            payload_ref = payload_path.replace("\\", "/")
            config_file.write(f'url = "{url}"\n')
            config_file.write("request = POST\n")
            config_file.write(f'connect-timeout = {min(int(timeout), 30)}\n')
            config_file.write(f'max-time = {int(timeout)}\n')
            config_file.write(f'resolve = "{parsed.hostname}:{port}:{resolve_ip}"\n')
            config_file.write(f'header = "Authorization: {headers["Authorization"]}"\n')
            config_file.write('header = "Content-Type: application/json"\n')
            config_file.write(f'data-binary = "@{payload_ref}"\n')
            config_path = config_file.name
        curl_binary = shutil.which("curl.exe") or shutil.which("curl")
        if not curl_binary:
            raise RuntimeError("curl is required for model RESOLVE_IP requests but was not found")
        result = subprocess.run(
            [curl_binary, "-sS", "-K", config_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 10,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"curl exited with {result.returncode}")
        data = json.loads(result.stdout)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
        return data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _cleanup_private_temp_dirs(prefix: str, *, max_age_seconds: int) -> None:
    temp_root = tempfile.gettempdir()
    cutoff = time.time() - max_age_seconds
    try:
        names = os.listdir(temp_root)
    except OSError:
        return
    for name in names:
        if not name.startswith(prefix):
            continue
        path = os.path.join(temp_root, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue
