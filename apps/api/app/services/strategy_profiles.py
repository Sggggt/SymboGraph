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

from app.models import Course, StrategyProfile


BUILTIN_COURSE_PROFILE_NAME = "课程资料库默认 Profile"
BUILTIN_COURSE_LIBRARY_TYPE = "academic"
BUILTIN_COURSE_PROFILE_NAME = "课程资料库默认 Profile"

DEFAULT_ENTITY_TYPES = ["concept", "method", "formula", "metric", "algorithm", "definition", "theorem", "problem_type"]
DEFAULT_RELATION_TYPES = [
    "is_a",
    "part_of",
    "prerequisite_of",
    "used_for",
    "causes",
    "derives_from",
    "compares_with",
    "example_of",
    "defined_by",
    "formula_of",
    "solves",
    "implemented_by",
    "related_to",
]

DEFAULT_RELATION_ALIASES = {
    "defines": "defined_by",
    "defined by": "defined_by",
    "relates_to": "related_to",
    "relates to": "related_to",
    "mentions": "related_to",
    "compares": "compares_with",
    "compares with": "compares_with",
    "extends": "derives_from",
}

DEFAULT_ENTITY_ALIASES = {
    "named_algorithm": "algorithm",
    "algo": "algorithm",
    "measure": "metric",
    "definition_term": "definition",
    "problem": "problem_type",
    "problem type": "problem_type",
    "application": "concept",
}

COURSE_GRAPH_EXTRACTION_PROMPT = (
    "You extract a course knowledge graph from teaching material. "
    "Return JSON only with keys concepts and relations. "
    "Each concept must contain name, aliases, summary, concept_type, importance_score. "
    "Each relation must contain source, target, relation_type, confidence. "
    "Allowed relation_type values: defines, relates_to, prerequisite_of, example_of, solves, compares, extends, mentions. "
    "Extract 6 to 12 specific course concepts when the excerpt has enough substance, including algorithms, theorems, "
    "definitions, problem types, complexity classes, graph structures, and proof techniques. "
    "Extract 6 to 16 useful relations between those concepts. "
    "Every relation source and target must exactly match a concept name included in concepts. "
    "Prefer specific names like Breadth-First Search, Dijkstra Algorithm, Spanning Tree, Flow Network, NP-Complete, "
    "Matching, Cut, Planar Graph, Eulerian Tour, Hamiltonian Cycle, and Matrix Tree Theorem over generic words. "
    "Skip formatting artifacts, page headers, exercise labels, and generic words."
)

COURSE_ANSWER_SYSTEM_PREFIX = "You are a course knowledge-base assistant. "
COURSE_CONTEXT_LABEL = "Course excerpts"
COURSE_NO_CONTEXT_EN = "I could not find enough reliable course context to answer this question with citations."
COURSE_NO_CONTEXT_ZH = "课程材料中没有找到足够可靠的上下文来回答这个问题并提供引用。"

DEFAULT_COURSE_PROFILE: dict[str, Any] = {
    "schema_version": "strategy_profile_v1",
    "library_type": BUILTIN_COURSE_LIBRARY_TYPE,
    "ui_labels": {
        "library": "资料库",
        "course": "课程",
        "partition": "章节",
        "partition_fallback": "General",
        "entity": "概念",
        "relation": "关系",
        "notes_route": "retrieve_notes",
        "exercise_route": "retrieve_exercises",
    },
    "prompt_pack": {
        "graph_extraction_system": COURSE_GRAPH_EXTRACTION_PROMPT,
        "answer_system_prefix": COURSE_ANSWER_SYSTEM_PREFIX,
        "context_label": COURSE_CONTEXT_LABEL,
        "no_context_answer_en": COURSE_NO_CONTEXT_EN,
        "no_context_answer_zh": COURSE_NO_CONTEXT_ZH,
        "reflection_domain": "course knowledge-base assistant",
        "citation_domain": "course excerpts",
        "query_translation_domain": "academic course search",
        "community_summary_system": "Summarize a course knowledge graph community for retrieval routing. Return strict JSON.",
        "graph_judge_system": "You are an LLM-as-a-judge for a course knowledge graph pipeline. Return strict JSON.",
        "graph_judge_threshold_hint": "Use these acceptance thresholds: invalid_chapter_refs must be empty, concepts_per_100_chunks >= 5, relations_per_concept >= 2.5, and multi-chapter courses should have at least 5 distinct chapter_ref_counts entries.",
        "perception_domain": "course knowledge-base agent",
        "entity_label": "course-concept-like terms",
        "coverage_label": "course materials",
        "indexed_coverage_label": "indexed course materials",
        "strongest_source_label": "course source",
        "strongest_source_label_zh": "课程来源",
        "relevant_section_label": "the relevant section",
        "relevant_section_label_zh": "相关章节",
        "agent_direct_answer_en": "I can answer questions about the indexed course materials, show citations, and explain how the retrieval agent reached its answer.",
        "agent_direct_answer_zh": "我可以回答已索引课程材料中的问题，提供引用，并说明检索智能体如何得到答案。",
        "agent_clarify_answer_en": "Please clarify the course concept, chapter, exercise, or comparison you want me to retrieve.",
        "agent_clarify_answer_zh": "请进一步说明你要检索的课程概念、章节、习题或比较问题。",
        "agent_no_context_answer_en": "I could not find enough relevant course material to answer this question. If you want me to try answering from the limited retrieved material, which may involve inference, please tell me.",
        "agent_no_context_answer_zh": "课程材料中没有找到足够相关内容来回答这个问题。如果你希望我基于已经检索到的有限材料尝试回答（可能包含推测），请告诉我。",
        "retry_query_suffix": "course lecture notes examples",
    },
    "schema_pack": {
        "entity_types": DEFAULT_ENTITY_TYPES,
        "relation_types": DEFAULT_RELATION_TYPES,
        "entity_aliases": DEFAULT_ENTITY_ALIASES,
        "relation_aliases": DEFAULT_RELATION_ALIASES,
        "disabled_entity_types": [],
        "disabled_relation_types": [],
        "default_entity_type": "concept",
        "default_relation_type": "related_to",
    },
    "parsing_strategy": {
        "partition_label": "Chapter",
        "section_label": "Section",
        "invalid_partition_labels": ["data", "storage", "reviewmarkdown"],
        "code_keep_markers": ["centrality", "community", "random network", "configuration model"],
    },
    "graph_strategy": {
        "min_concept_evidence_chunks": 2,
        "min_concept_specificity": 0.35,
        "min_relation_confidence": 0.72,
        "min_accepted_relation_weight": 0.62,
    },
    "retrieval_strategy": {
        "query_type_markers": {
            "definition": ["what is", "define", "definition", "meaning", "concept", "什么是", "定义", "概念"],
            "formula": ["formula", "theorem", "proof", "derive", "equation", "complexity", "公式", "定理", "证明"],
            "example": ["example", "instance", "case", "举例", "例子"],
            "comparison": ["compare", "versus", "vs", "difference", "relationship", "relate", "区别", "比较", "关系"],
            "procedure": ["algorithm", "procedure", "steps", "how to", "流程", "步骤", "算法", "如何"],
        },
        "agent_route_markers": {
            "multi_hop_research": ["compare", "relationship", "related to", "relation between", "difference between", "connect", "derive", "prove", "比较", "关系", "区别", "联系", "推导", "证明"],
            "retrieve_exercises": ["exercise", "homework", "problem", "assignment", "quiz", "exam"],
            "retrieve_notes": ["note", "slide", "definition", "concept", "chapter"],
        },
        "route_terms": {
            "exercise": ["exercise", "homework", "problem", "assignment", "quiz", "exam"],
            "notes": ["note", "slide", "definition", "concept", "chapter"],
        },
    },
    "quality_policy": {
        "structural_role_terms": ["chapter", "section", "unit", "module", "lecture", "slide", "page", "course", "syllabus", "outline", "agenda", "summary", "appendix", "reference", "solution", "homework", "assignment", "quiz", "exam", "lab", "worksheet"],
        "generic_concept_terms": ["algorithm", "method", "data", "model", "result", "example", "problem", "system", "approach", "process", "value", "function", "feature", "task", "step"],
        "definition_markers": [" is ", " are ", " refers to ", " defined as ", " means ", " denotes ", " definition ", " 定义 ", " 是 ", " 指 ", " 称为 "],
    },
}

_active_profile_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("active_strategy_profile", default=None)


def profile_hash(profile_json: dict[str, Any]) -> str:
    payload = json.dumps(profile_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_course_profile_payload() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_COURSE_PROFILE)


def validate_profile_payload(payload: object) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Profile JSON must be an object")
    profile = copy.deepcopy(payload)
    warnings: list[str] = []
    profile.setdefault("schema_version", "strategy_profile_v1")
    profile.setdefault("library_type", "custom")
    for key in ("ui_labels", "prompt_pack", "schema_pack", "parsing_strategy", "graph_strategy", "retrieval_strategy", "quality_policy"):
        value = profile.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
    schema_pack = profile["schema_pack"]
    entity_types = schema_pack.get("entity_types")
    relation_types = schema_pack.get("relation_types")
    if not isinstance(entity_types, list) or not all(isinstance(item, str) and item.strip() for item in entity_types):
        raise ValueError("schema_pack.entity_types must be a non-empty list of strings")
    if not isinstance(relation_types, list) or not all(isinstance(item, str) and item.strip() for item in relation_types):
        raise ValueError("schema_pack.relation_types must be a non-empty list of strings")
    schema_pack["entity_types"] = list(dict.fromkeys(item.strip().lower().replace("-", "_").replace(" ", "_") for item in entity_types))
    schema_pack["relation_types"] = list(dict.fromkeys(item.strip().lower().replace("-", "_").replace(" ", "_") for item in relation_types))
    if "concept" not in schema_pack["entity_types"]:
        warnings.append("schema_pack.entity_types does not include concept; unknown entity types will use the configured default")
    if "related_to" not in schema_pack["relation_types"]:
        warnings.append("schema_pack.relation_types does not include related_to; unknown relation types will use the configured default")
    schema_pack.setdefault("default_entity_type", schema_pack["entity_types"][0])
    schema_pack.setdefault("default_relation_type", schema_pack["relation_types"][0])
    for map_key in ("entity_aliases", "relation_aliases"):
        aliases = schema_pack.get(map_key, {})
        if not isinstance(aliases, dict):
            raise ValueError(f"schema_pack.{map_key} must be an object")
        schema_pack[map_key] = {str(key).strip().lower().replace("-", "_").replace(" ", "_"): str(value).strip().lower().replace("-", "_").replace(" ", "_") for key, value in aliases.items() if str(key).strip() and str(value).strip()}
    profile["profile_hash"] = profile_hash(profile)
    return profile, warnings


def ensure_builtin_course_profile(db: Session) -> StrategyProfile:
    profile = db.scalar(select(StrategyProfile).where(StrategyProfile.name == BUILTIN_COURSE_PROFILE_NAME))
    payload = default_course_profile_payload()
    digest = profile_hash(payload)
    if profile is None:
        profile = StrategyProfile(
            name=BUILTIN_COURSE_PROFILE_NAME,
            library_type=BUILTIN_COURSE_LIBRARY_TYPE,
            is_builtin=True,
            profile_json=payload,
            profile_hash=digest,
            is_active=True,
        )
        db.add(profile)
        db.flush()
    elif profile.is_builtin and profile.profile_hash != digest:
        profile.library_type = BUILTIN_COURSE_LIBRARY_TYPE
        profile.profile_json = payload
        profile.profile_hash = digest
        profile.is_active = True
        profile.updated_at = datetime.utcnow()
        db.flush()
    return profile


def ensure_courses_have_profiles(db: Session) -> StrategyProfile:
    profile = ensure_builtin_course_profile(db)
    courses = db.scalars(select(Course).where(Course.active_profile_id.is_(None))).all()
    for course in courses:
        course.active_profile_id = profile.id
    if courses:
        db.flush()
    return profile


def get_active_profile_record(db: Session, course_id: str | None) -> StrategyProfile:
    ensure_courses_have_profiles(db)
    course = db.get(Course, course_id) if course_id else None
    profile = db.get(StrategyProfile, course.active_profile_id) if course and course.active_profile_id else None
    if profile is None or not profile.is_active:
        profile = ensure_builtin_course_profile(db)
        if course is not None:
            course.active_profile_id = profile.id
            db.flush()
    return profile


def active_profile_json(db: Session | None = None, course_id: str | None = None) -> dict[str, Any]:
    current = _active_profile_var.get()
    if current is not None:
        return current
    if db is not None:
        return copy.deepcopy(get_active_profile_record(db, course_id).profile_json)
    return default_course_profile_payload()


@contextmanager
def use_strategy_profile(profile_json: dict[str, Any] | None) -> Iterator[None]:
    token = _active_profile_var.set(copy.deepcopy(profile_json) if profile_json else None)
    try:
        yield
    finally:
        _active_profile_var.reset(token)


def profile_to_payload(profile: StrategyProfile, *, course_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "library_type": profile.library_type,
        "is_builtin": profile.is_builtin,
        "is_active": profile.is_active,
        "profile_hash": profile.profile_hash,
        "profile_json": profile.profile_json,
        "course_ids": course_ids or [],
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def list_profiles(db: Session) -> list[dict[str, Any]]:
    ensure_courses_have_profiles(db)
    courses_by_profile: dict[str, list[str]] = {}
    for course in db.scalars(select(Course)).all():
        if course.active_profile_id:
            courses_by_profile.setdefault(course.active_profile_id, []).append(course.id)
    profiles = db.scalars(select(StrategyProfile).where(StrategyProfile.is_active.is_(True)).order_by(StrategyProfile.is_builtin.desc(), StrategyProfile.name.asc())).all()
    return [profile_to_payload(profile, course_ids=sorted(courses_by_profile.get(profile.id, []))) for profile in profiles]


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
    default_profile = ensure_builtin_course_profile(db)
    bound_courses = db.scalars(select(Course).where(Course.active_profile_id == profile_id)).all()
    for course in bound_courses:
        course.active_profile_id = default_profile.id
    profile.is_active = False
    profile.updated_at = datetime.utcnow()
    db.commit()


def bind_profile_to_course(db: Session, *, course_id: str, profile_id: str) -> Course:
    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"Course not found: {course_id}")
    profile = get_profile_or_raise(db, profile_id)
    course.active_profile_id = profile.id
    db.commit()
    db.refresh(course)
    return course


async def generate_profile_draft(prompt: str, base_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    from app.services.embeddings import ChatProvider

    base = base_profile or default_course_profile_payload()
    system_prompt = (
        "You generate a strict JSON strategy profile draft for a local knowledge-base system. "
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
    return (profile_json or {}).get("schema_pack") if isinstance((profile_json or {}).get("schema_pack"), dict) else DEFAULT_COURSE_PROFILE["schema_pack"]
