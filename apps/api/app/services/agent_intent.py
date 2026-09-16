from __future__ import annotations

from typing import Any
import hashlib
import json


QUESTION_PERCEPTION_PROTOCOL_VERSION = "agent_question_perception_v3"


def current_question_scope_projection(question: str, query_intent: dict[str, Any]) -> dict[str, Any]:
    """Carry current interpretation across modules, without factual authority."""
    available = bool(query_intent and QUESTION_PERCEPTION_FIELDS.issubset(query_intent))
    validated = validate_question_perception_output({key: query_intent[key] for key in QUESTION_PERCEPTION_FIELDS}, question=question) if available else {}
    entities, sub_queries = validated.get("entities", []), validated.get("sub_queries", [])
    if any("\x00" in value for value in [*entities, *sub_queries]):
        raise ValueError("question_scope_control_text_invalid")
    packet = {"protocol_version": "current_question_scope_v1", "available": available,
        "current_question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "intent": validated.get("intent", "unknown"),
        "entities": [value[:128] for value in entities[:16]],
        "sub_queries": [value[:512] for value in sub_queries[:8]],
        "original_entity_count": len(entities), "original_sub_query_count": len(sub_queries),
        "projection_clipped": len(entities) > 16 or len(sub_queries) > 8 or any(len(value) > 128 for value in entities[:16])
            or any(len(value) > 512 for value in sub_queries[:8]),
        "current_user_overrides_projection": True, "is_evidence": False, "gray_zone_decision_authority": False}
    packet["packet_hash"] = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return packet

QUESTION_PERCEPTION_IMMUTABLE_CONTRACT = (
    "Immutable Agent question-perception contract: return exactly one JSON "
    "object with exactly these keys: intent, direct_answer_kind, entities, "
    "sub_queries, needs_graph, suggested_strategy. intent must be one of "
    "direct_answer, definition, comparison, application, procedure, analysis, "
    "formula_table_lookup, unknown. direct_answer_kind must be one of identity, "
    "model_identity, capabilities, evidence, usage, none. Use direct_answer only "
    "for this Agent's own identity, configured model identity, capabilities, "
    "evidence policy, or usage; then needs_graph must be false and "
    "suggested_strategy must be none. Every non-direct intent must use "
    "direct_answer_kind=none. Do not answer the question or propose tools."
    " Keep this object compact: list only necessary short entity names and short sub-queries. "
    "A single follow-up normally needs one short sub-query. Do not copy a previous answer or "
    "history summary into any field, and do not output private reasoning. "
    "identity describes the Agent's role; model_identity covers the underlying model, its configuration, "
    "or whether a model name is the same as the Agent's system identity. Prefer model_identity "
    "when the question concerns those model attributes or that relationship."
)

AGENT_QUERY_INTENTS = frozenset(
    {
        "direct_answer",
        "definition",
        "comparison",
        "application",
        "procedure",
        "analysis",
        "formula_table_lookup",
        "unknown",
    }
)

DIRECT_ANSWER_KINDS = frozenset(
    {
        "identity",
        "model_identity",
        "capabilities",
        "evidence",
        "usage",
    }
)

QUESTION_PERCEPTION_STRATEGIES = frozenset(
    {
        "none",
        "global_dense",
        "local_graph",
        "hybrid",
        "community",
    }
)

QUESTION_PERCEPTION_FIELDS = frozenset(
    {
        "intent",
        "direct_answer_kind",
        "entities",
        "sub_queries",
        "needs_graph",
        "suggested_strategy",
    }
)


def validate_question_perception_output(
    raw: Any,
    *,
    question: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("question perception output must be an object")
    if set(raw) != QUESTION_PERCEPTION_FIELDS:
        raise ValueError("question perception output is not closed")

    intent = str(raw.get("intent") or "").strip().lower()
    direct_answer_kind = str(
        raw.get("direct_answer_kind") or ""
    ).strip().lower()
    suggested_strategy = str(
        raw.get("suggested_strategy") or ""
    ).strip().lower()
    entities = raw.get("entities")
    sub_queries = raw.get("sub_queries")
    needs_graph = raw.get("needs_graph")

    if intent not in AGENT_QUERY_INTENTS:
        raise ValueError("question perception intent is unsupported")
    if suggested_strategy not in QUESTION_PERCEPTION_STRATEGIES:
        raise ValueError("question perception strategy is unsupported")
    if (
        not isinstance(entities, list)
        or len(entities) > 64
        or any(
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 512
            for value in entities
        )
    ):
        raise ValueError("question perception entities are invalid")
    if (
        not isinstance(sub_queries, list)
        or not sub_queries
        or len(sub_queries) > 32
        or any(
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 4000
            for value in sub_queries
        )
    ):
        raise ValueError("question perception sub-queries are invalid")
    if type(needs_graph) is not bool:
        raise ValueError("question perception needs_graph must be boolean")

    if intent == "direct_answer":
        if direct_answer_kind not in DIRECT_ANSWER_KINDS:
            raise ValueError("direct-answer intent kind is unsupported")
        if needs_graph is not False or suggested_strategy != "none":
            raise ValueError(
                "direct-answer intent cannot request graph retrieval"
            )
    elif direct_answer_kind != "none":
        raise ValueError(
            "non-direct intent must use direct_answer_kind=none"
        )

    return {
        "intent": intent,
        "direct_answer_kind": direct_answer_kind,
        "entities": [str(value).strip() for value in entities],
        "sub_queries": [str(value).strip() for value in sub_queries],
        "needs_graph": needs_graph,
        "suggested_strategy": suggested_strategy,
    }


def validate_direct_answer_intent(
    query_intent: Any,
) -> dict[str, Any]:
    if not isinstance(query_intent, dict):
        raise ValueError("query intent must be an object")
    intent = str(query_intent.get("intent") or "").strip().lower()
    direct_answer_kind = str(
        query_intent.get("direct_answer_kind") or ""
    ).strip().lower()
    if intent != "direct_answer":
        raise ValueError("query intent is not direct_answer")
    if direct_answer_kind not in DIRECT_ANSWER_KINDS:
        raise ValueError("direct-answer intent kind is unsupported")
    if query_intent.get("needs_graph") is not False:
        raise ValueError("direct-answer intent cannot request graph retrieval")
    if query_intent.get("suggested_strategy") != "none":
        raise ValueError("direct-answer intent strategy must be none")
    return {
        "intent": intent,
        "direct_answer_kind": direct_answer_kind,
        "needs_graph": False,
        "suggested_strategy": "none",
    }
