from __future__ import annotations

import contextvars
import copy
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KnowledgeBase, StrategyProfile


BUILTIN_DEFAULT_PROFILE_NAME = "Default Evidence Profile"
BUILTIN_LIBRARY_TYPE = "general"

DEFAULT_SIGNAL_TYPES = [
    "claim",
    "definition",
    "method",
    "metric",
    "formula",
    "evidence_marker",
    "topic",
    "constraint",
]
DEFAULT_SIGNAL_RELATION_TYPES = [
    "supports",
    "refines",
    "depends_on",
    "contrasts",
    "continues",
    "contains",
    "cites",
    "same_topic",
    "related_observation",
]
DEFAULT_SIGNAL_TYPE_ALIASES = {
    "algorithm": "method",
    "concept": "topic",
    "entity": "topic",
    "term": "topic",
    "theorem": "claim",
}
DEFAULT_SIGNAL_RELATION_ALIASES = {
    "defined_by": "supports",
    "mentions": "same_topic",
    "references": "cites",
    "related_to": "related_observation",
}

GENERIC_ANSWER_SYSTEM_PREFIX = "You are an evidence-grounded knowledge-base assistant. "
GENERIC_CONTEXT_LABEL = "Indexed evidence excerpts"
GENERIC_NO_CONTEXT_EN = "I could not find enough reliable indexed evidence to answer this question with citations."
GENERIC_NO_CONTEXT_ZH = "索引资料中没有找到足够可靠的证据来回答这个问题并提供引用。"

DEFAULT_ANSWER_SYSTEM_PREFIX = GENERIC_ANSWER_SYSTEM_PREFIX
DEFAULT_CONTEXT_LABEL = GENERIC_CONTEXT_LABEL
DEFAULT_NO_CONTEXT_EN = GENERIC_NO_CONTEXT_EN
DEFAULT_NO_CONTEXT_ZH = GENERIC_NO_CONTEXT_ZH

DEFAULT_PROFILE: dict[str, Any] = {
    "schema_version": "strategy_profile_v3",
    "library_type": BUILTIN_LIBRARY_TYPE,
    "ui_labels": {
        "library": "Knowledge Base",
        "knowledge_base": "Knowledge Base",
        "partition": "partition",
        "partition_fallback": "General",
        "entity": "signal",
        "relation": "observed edge",
        "source_route": "retrieve_sources",
        "task_route": "retrieve_tasks",
    },
    "prompt_pack": {
        "answer_system_prefix": GENERIC_ANSWER_SYSTEM_PREFIX,
        "context_label": GENERIC_CONTEXT_LABEL,
        "no_context_answer_en": GENERIC_NO_CONTEXT_EN,
        "no_context_answer_zh": GENERIC_NO_CONTEXT_ZH,
        "reflection_domain": "evidence-grounded knowledge-base assistant",
        "citation_domain": "indexed evidence excerpts",
        "query_translation_domain": "knowledge-base evidence search",
        "community_summary_system": "Summarize an evidence graph community for retrieval routing. Return strict JSON.",
        "quality_judge_system": "You are an LLM-as-a-judge for an evidence-first knowledge-base pipeline. Return strict JSON.",
        "perception_domain": "evidence-grounded knowledge-base agent",
        "entity_label": "source-grounded evidence signals",
        "coverage_label": "indexed materials",
        "indexed_coverage_label": "indexed materials",
        "strongest_source_label": "knowledge-base source",
        "strongest_source_label_zh": "资料来源",
        "relevant_section_label": "the relevant partition",
        "relevant_section_label_zh": "相关分区",
        "agent_direct_answer_en": "I can answer questions about indexed knowledge-base materials, show citations, and explain how retrieval reached the answer.",
        "agent_direct_answer_zh": "我可以回答已索引资料中的问题，提供引用，并说明检索如何得到答案。",
        "agent_clarify_answer_en": "Please clarify the source, partition, task, or comparison you want me to retrieve.",
        "agent_clarify_answer_zh": "请进一步说明你要检索的来源、分区、任务或比较问题。",
        "agent_no_context_answer_en": "I could not find enough relevant indexed evidence to answer this question. If you want me to try with limited retrieved material, please tell me.",
        "agent_no_context_answer_zh": "索引资料中没有找到足够相关的证据来回答这个问题。如果你希望我基于有限检索材料尝试回答，请告诉我。",
        "retry_query_suffix": "knowledge-base evidence excerpts examples",
    },
    "schema_pack": {
        "entity_types": DEFAULT_SIGNAL_TYPES,
        "relation_types": DEFAULT_SIGNAL_RELATION_TYPES,
        "entity_aliases": DEFAULT_SIGNAL_TYPE_ALIASES,
        "relation_aliases": DEFAULT_SIGNAL_RELATION_ALIASES,
        "disabled_entity_types": [],
        "disabled_relation_types": [],
        "default_entity_type": "topic",
        "default_relation_type": "related_observation",
    },
    "parsing_strategy": {
        "partition_label": "partition",
        "section_label": "Section",
        "invalid_partition_labels": ["data", "storage", "reviewmarkdown"],
        "code_keep_markers": ["evidence", "graph", "community", "parser", "retrieval"],
    },
    "graph_strategy": {
        "min_signal_evidence_atoms": 1,
        "min_signal_confidence": 0.5,
        "min_observed_edge_confidence": 0.55,
    },
    "retrieval_strategy": {
        "query_type_markers": {
            "definition": ["what is", "define", "definition", "meaning", "term", "什么是", "定义"],
            "formula": ["formula", "proof", "derive", "equation", "公式", "证明", "推导"],
            "example": ["example", "instance", "case", "举例", "例子"],
            "comparison": ["compare", "versus", "vs", "difference", "relationship", "relate", "区别", "比较", "关系"],
            "procedure": ["procedure", "steps", "how to", "workflow", "流程", "步骤", "如何"],
        },
        "agent_route_markers": {
            "multi_hop_research": ["compare", "relationship", "related to", "relation between", "difference between", "connect", "derive", "prove", "比较", "关系", "区别", "联系", "推导", "证明"],
            "retrieve_tasks": ["task", "requirement", "question", "problem", "todo", "checklist"],
            "retrieve_sources": ["source", "document", "excerpt", "definition", "section", "partition", "reference"],
        },
        "route_terms": {
            "tasks": ["task", "requirement", "question", "problem", "todo", "checklist"],
            "sources": ["source", "document", "excerpt", "definition", "section", "partition", "reference"],
        },
    },
    "quality_policy": {
        "structural_role_terms": ["partition", "section", "unit", "module", "page", "outline", "summary", "appendix", "reference"],
        "generic_signal_terms": ["data", "model", "result", "example", "system", "approach", "process", "value", "function", "feature", "task", "step"],
        "definition_markers": [" is ", " are ", " refers to ", " defined as ", " means ", " denotes ", " definition ", " 定义 ", " 是 ", " 指 "],
    },
}

_active_profile_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("active_strategy_profile", default=None)


def profile_hash(profile_json: dict[str, Any]) -> str:
    payload = json.dumps(profile_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_profile_payload() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_PROFILE)


def migrate_profile_payload(profile: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(profile)
    migrated["schema_version"] = "strategy_profile_v3"
    ui_labels = migrated.setdefault("ui_labels", {})
    if isinstance(ui_labels, dict):
        if "KnowledgeBase" in ui_labels and "knowledge_base" not in ui_labels:
            ui_labels["knowledge_base"] = ui_labels.pop("KnowledgeBase")
        if "notes_route" in ui_labels and "source_route" not in ui_labels:
            ui_labels["source_route"] = ui_labels.pop("notes_route")
        if "exercise_route" in ui_labels and "task_route" not in ui_labels:
            ui_labels["task_route"] = ui_labels.pop("exercise_route")
    retrieval_strategy = migrated.setdefault("retrieval_strategy", {})
    if isinstance(retrieval_strategy, dict):
        route_terms = retrieval_strategy.setdefault("route_terms", {})
        if isinstance(route_terms, dict):
            if "notes" in route_terms and "sources" not in route_terms:
                route_terms["sources"] = route_terms.pop("notes")
            if "exercise" in route_terms and "tasks" not in route_terms:
                route_terms["tasks"] = route_terms.pop("exercise")
        markers = retrieval_strategy.setdefault("agent_route_markers", {})
        if isinstance(markers, dict):
            if "retrieve_notes" in markers and "retrieve_sources" not in markers:
                markers["retrieve_sources"] = markers.pop("retrieve_notes")
            if "retrieve_exercises" in markers and "retrieve_tasks" not in markers:
                markers["retrieve_tasks"] = markers.pop("retrieve_exercises")
    return migrated


def _normalize_schema_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def validate_profile_payload(payload: object) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Profile JSON must be an object")
    profile = migrate_profile_payload(payload)
    warnings: list[str] = []
    profile.setdefault("schema_version", "strategy_profile_v3")
    profile.setdefault("library_type", "custom")
    for key in ("ui_labels", "prompt_pack", "schema_pack", "parsing_strategy", "graph_strategy", "retrieval_strategy", "quality_policy"):
        value = profile.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
    schema_pack = profile["schema_pack"]
    entity_types = schema_pack.get("entity_types")
    relation_types = schema_pack.get("relation_types")
    if not isinstance(entity_types, list) or not all(isinstance(item, str) and item.strip() for item in entity_types):
        raise ValueError("schema_pack.entity_types must be a non-empty list of signal type strings")
    if not isinstance(relation_types, list) or not all(isinstance(item, str) and item.strip() for item in relation_types):
        raise ValueError("schema_pack.relation_types must be a non-empty list of observed edge type strings")
    schema_pack["entity_types"] = list(dict.fromkeys(_normalize_schema_name(item) for item in entity_types))
    schema_pack["relation_types"] = list(dict.fromkeys(_normalize_schema_name(item) for item in relation_types))
    if "topic" not in schema_pack["entity_types"]:
        warnings.append("schema_pack.entity_types does not include topic; unknown signal types will use the configured default")
    if "related_observation" not in schema_pack["relation_types"]:
        warnings.append("schema_pack.relation_types does not include related_observation; unknown observed edge types will use the configured default")
    schema_pack.setdefault("default_entity_type", schema_pack["entity_types"][0])
    schema_pack.setdefault("default_relation_type", schema_pack["relation_types"][0])
    for map_key in ("entity_aliases", "relation_aliases"):
        aliases = schema_pack.get(map_key, {})
        if not isinstance(aliases, dict):
            raise ValueError(f"schema_pack.{map_key} must be an object")
        schema_pack[map_key] = {
            _normalize_schema_name(key): _normalize_schema_name(value)
            for key, value in aliases.items()
            if str(key).strip() and str(value).strip()
        }
    profile["profile_hash"] = profile_hash(profile)
    return profile, warnings


def ensure_builtin_default_profile(db: Session) -> StrategyProfile:
    profile = db.scalar(select(StrategyProfile).where(StrategyProfile.name == BUILTIN_DEFAULT_PROFILE_NAME))
    payload = default_profile_payload()
    digest = profile_hash(payload)
    if profile is None:
        profile = StrategyProfile(
            name=BUILTIN_DEFAULT_PROFILE_NAME,
            library_type=BUILTIN_LIBRARY_TYPE,
            is_builtin=True,
            profile_json=payload,
            profile_hash=digest,
            is_active=True,
        )
        db.add(profile)
        db.flush()
    elif profile.is_builtin and profile.profile_hash != digest:
        profile.library_type = BUILTIN_LIBRARY_TYPE
        profile.profile_json = payload
        profile.profile_hash = digest
        profile.is_active = True
        profile.updated_at = datetime.utcnow()
        db.flush()
    return profile


def ensure_knowledge_bases_have_profiles(db: Session) -> StrategyProfile:
    profile = ensure_builtin_default_profile(db)
    knowledge_bases = db.scalars(select(KnowledgeBase).where(KnowledgeBase.active_profile_id.is_(None))).all()
    for knowledge_base in knowledge_bases:
        knowledge_base.active_profile_id = profile.id
    if knowledge_bases:
        db.flush()
    return profile


def get_active_profile_record(db: Session, knowledge_base_id: str | None) -> StrategyProfile:
    ensure_knowledge_bases_have_profiles(db)
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id) if knowledge_base_id else None
    profile = db.get(StrategyProfile, knowledge_base.active_profile_id) if knowledge_base and knowledge_base.active_profile_id else None
    if profile is None or not profile.is_active:
        profile = ensure_builtin_default_profile(db)
        if knowledge_base is not None:
            knowledge_base.active_profile_id = profile.id
            db.flush()
    return profile


def active_profile_json(db: Session | None = None, knowledge_base_id: str | None = None) -> dict[str, Any]:
    current = _active_profile_var.get()
    if current is not None:
        return current
    if db is not None:
        return copy.deepcopy(get_active_profile_record(db, knowledge_base_id).profile_json)
    return default_profile_payload()


@contextmanager
def use_strategy_profile(profile_json: dict[str, Any] | None) -> Iterator[None]:
    token = _active_profile_var.set(copy.deepcopy(profile_json) if profile_json else None)
    try:
        yield
    finally:
        _active_profile_var.reset(token)


def profile_to_payload(profile: StrategyProfile, *, knowledge_base_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "library_type": profile.library_type,
        "is_builtin": profile.is_builtin,
        "is_active": profile.is_active,
        "profile_hash": profile.profile_hash,
        "profile_json": profile.profile_json,
        "knowledge_base_ids": knowledge_base_ids or [],
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def list_profiles(db: Session) -> list[dict[str, Any]]:
    ensure_knowledge_bases_have_profiles(db)
    knowledge_bases_by_profile: dict[str, list[str]] = {}
    for knowledge_base in db.scalars(select(KnowledgeBase)).all():
        if knowledge_base.active_profile_id:
            knowledge_bases_by_profile.setdefault(knowledge_base.active_profile_id, []).append(knowledge_base.id)
    profiles = db.scalars(select(StrategyProfile).where(StrategyProfile.is_active.is_(True)).order_by(StrategyProfile.is_builtin.desc(), StrategyProfile.name.asc())).all()
    return [profile_to_payload(profile, knowledge_base_ids=sorted(knowledge_bases_by_profile.get(profile.id, []))) for profile in profiles]


def get_profile_or_raise(db: Session, profile_id: str) -> StrategyProfile:
    profile = db.get(StrategyProfile, profile_id)
    if profile is None or not profile.is_active:
        raise LookupError(f"Strategy profile not found: {profile_id}")
    return profile


def create_profile(db: Session, *, name: str, library_type: str, profile_json: dict[str, Any]) -> tuple[StrategyProfile, list[str]]:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Profile name is required")
    if db.scalar(select(StrategyProfile).where(StrategyProfile.name == normalized, StrategyProfile.is_active.is_(True))) is not None:
        raise ValueError(f"Profile name already exists: {normalized}")
    payload, warnings = validate_profile_payload(profile_json)
    profile = StrategyProfile(
        name=normalized,
        library_type=library_type.strip() or str(payload.get("library_type") or "custom"),
        is_builtin=False,
        profile_json=payload,
        profile_hash=profile_hash(payload),
        is_active=True,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile, warnings


def update_profile(db: Session, profile_id: str, *, name: str | None = None, library_type: str | None = None, profile_json: dict[str, Any] | None = None) -> tuple[StrategyProfile, list[str]]:
    profile = get_profile_or_raise(db, profile_id)
    if profile.is_builtin:
        raise ValueError("Builtin profiles cannot be edited; copy it first")
    if name is not None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Profile name is required")
        existing = db.scalar(select(StrategyProfile).where(StrategyProfile.name == normalized, StrategyProfile.id != profile_id, StrategyProfile.is_active.is_(True)))
        if existing is not None:
            raise ValueError(f"Profile name already exists: {normalized}")
        profile.name = normalized
    if library_type is not None:
        profile.library_type = library_type.strip() or "custom"
    warnings: list[str] = []
    if profile_json is not None:
        payload, warnings = validate_profile_payload(profile_json)
        profile.profile_json = payload
        profile.profile_hash = profile_hash(payload)
    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile, warnings


def copy_profile(db: Session, source_profile_id: str, *, name: str) -> StrategyProfile:
    source = get_profile_or_raise(db, source_profile_id)
    profile, _warnings = create_profile(
        db,
        name=name,
        library_type=source.library_type,
        profile_json=copy.deepcopy(source.profile_json),
    )
    return profile


def delete_profile(db: Session, profile_id: str) -> None:
    profile = get_profile_or_raise(db, profile_id)
    if profile.is_builtin:
        raise ValueError("Builtin profiles cannot be deleted")
    default_profile = ensure_builtin_default_profile(db)
    bound_knowledge_bases = db.scalars(select(KnowledgeBase).where(KnowledgeBase.active_profile_id == profile_id)).all()
    for knowledge_base in bound_knowledge_bases:
        knowledge_base.active_profile_id = default_profile.id
    profile.is_active = False
    profile.updated_at = datetime.utcnow()
    db.commit()


def bind_profile_to_knowledge_base(db: Session, *, knowledge_base_id: str, profile_id: str) -> KnowledgeBase:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise LookupError(f"KnowledgeBase not found: {knowledge_base_id}")
    profile = get_profile_or_raise(db, profile_id)
    knowledge_base.active_profile_id = profile.id
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


async def generate_profile_draft(prompt: str, base_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    from app.services.embeddings import ChatProvider

    base = base_profile or default_profile_payload()
    system_prompt = (
        "You generate a strict JSON strategy profile draft for a local evidence-first knowledge-base system. "
        "Return JSON only. Preserve the top-level keys from the provided base profile. "
        "Do not include secrets, API keys, markdown, or commentary."
    )
    user_prompt = (
        f"User request:\n{prompt.strip()}\n\n"
        "Base profile JSON:\n"
        f"{json.dumps(base, ensure_ascii=False)}\n\n"
        "Return an edited profile_json object only."
    )
    draft = await ChatProvider().classify_json(system_prompt, user_prompt, fallback=base)
    validated, warnings = validate_profile_payload(draft or base)
    return {"profile_json": validated, "warnings": warnings, "profile_hash": profile_hash(validated)}


def profile_label(profile_json: dict[str, Any], key: str, default: str) -> str:
    value = ((profile_json or {}).get("ui_labels") or {}).get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def profile_prompt(profile_json: dict[str, Any], key: str, default: str) -> str:
    value = ((profile_json or {}).get("prompt_pack") or {}).get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def profile_schema(profile_json: dict[str, Any]) -> dict[str, Any]:
    return (profile_json or {}).get("schema_pack") if isinstance((profile_json or {}).get("schema_pack"), dict) else DEFAULT_PROFILE["schema_pack"]
