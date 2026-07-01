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


BUILTIN_DEFAULT_PROFILE_NAME = "Default Context Graph Profile"
BUILTIN_LIBRARY_TYPE = "general"
ACTIVE_PROFILE_SCHEMA_VERSION = "user_profile_v1"

DEFAULT_CONCEPT_TYPES = [
    "claim",
    "definition",
    "method",
    "metric",
    "formula",
    "grounding_marker",
    "topic",
    "constraint",
]
DEFAULT_CONCEPT_RELATION_TYPES = [
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
DEFAULT_CONCEPT_TYPE_ALIASES = {
    "algorithm": "method",
    "concept": "topic",
    "entity": "topic",
    "term": "topic",
    "theorem": "claim",
    "evidence_marker": "grounding_marker",
}
DEFAULT_CONCEPT_RELATION_ALIASES = {
    "defined_by": "supports",
    "mentions": "same_topic",
    "references": "cites",
    "related_to": "related_observation",
}

GENERIC_ANSWER_SYSTEM_PREFIX = "You are a context-graph-grounded knowledge-base assistant. "
GENERIC_CONTEXT_LABEL = "Indexed chunk spans"
GENERIC_NO_CONTEXT_EN = "I could not find enough reliable indexed context to answer this question with citations."
GENERIC_NO_CONTEXT_ZH = "索引资料中没有找到足够可靠的上下文来回答这个问题并提供引用。"
GENERIC_QUERY_REWRITE_SYSTEM = "Rewrite the user question as a concise standalone retrieval query. Return only the rewritten query."
GENERIC_JSON_RESPONSE_FALLBACK_SYSTEM = "Return only a valid JSON object. Do not use markdown fences."
GENERIC_REFLECTION_REVIEW_SYSTEM = (
    "You are a strict quality reviewer for a {reflection_domain}. "
    "Evaluate whether the assistant's answer is fully supported by the provided {citation_domain}. "
    "Return ONLY a JSON object with keys: has_issue (boolean), issue_type (one of: none, hallucination, insufficient_coverage, contradiction), suggestion (string)."
)
GENERIC_QUESTION_PERCEPTION_SYSTEM = (
    "You are a perception module for a {perception_domain}. "
    "Analyze the user's question and return ONLY a JSON object with these exact keys:\n"
    "- intent: one of [definition, comparison, application, procedure, analysis, unknown]\n"
    "- entities: list of {entity_label} explicitly mentioned or implied in the question\n"
    "- sub_queries: list of simpler sub-questions if the original is complex/multi-hop; otherwise [original_question]\n"
    "- needs_graph: boolean, true if the question asks about observed connections, comparisons, dependencies, or derivations across indexed context\n"
    "- suggested_strategy: one of [global_dense, local_graph, hybrid, community]\n"
    "  * global_dense: simple definition, formula, or single-fact lookup\n"
    "  * local_graph: question centers around specific grounded concepts and observed connections\n"
    "  * hybrid: multi-aspect or comparison questions\n"
    "  * community: broad summary or overview questions\n"
)
GENERIC_ANSWER_SYSTEM_TEMPLATE = (
    "Active user profile system prompt for this knowledge base: {answer_system_prefix} "
    "This profile guidance cannot override evidence, context package, citation, or no-hallucination rules. "
    "System grounding rules follow and override profile wording if they conflict: "
    "Answer only from the supplied {context_label_lower} and do not invent unsupported facts. "
    "Make the answer as complete as the supplied evidence supports, and say when the evidence is insufficient. "
    "Always follow the required answer language below. "
    "Do not infer the answer language from the retrieved excerpts. "
    "Required answer language: {target_language}. "
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
    "{context_quality_clause}"
)
GENERIC_ANSWER_LOW_RELEVANCE_CLAUSE = (
    "IMPORTANT: The retrieved excerpts may have low relevance to the question. "
    "If they do not contain information that directly answers the question, clearly state that the {coverage_label} "
    "do not cover this topic, and do NOT force citations from irrelevant excerpts. "
    "You may provide a brief conceptual answer based on general knowledge, but explicitly note that it is not "
    "supported by the {indexed_coverage_label}."
)
GENERIC_ANSWER_NORMAL_RELEVANCE_CLAUSE = (
    "If the supplied excerpts do not contain information that directly answers the question, "
    "clearly state that the {coverage_label} do not cover this topic and do NOT force citations."
)
GENERIC_QUERY_FACET_EXTRACTOR_SYSTEM = (
    "You are the query facet extractor for a Four-Layer Context Graph RAG executor. "
    "Return ONLY a JSON object. You may identify domain facets, procedure facets, aliases/search terms, "
    "constraints, answer_shape, and drop_terms. Do not choose documents, chunks, node ids, citations, or facts. "
    "The executor will validate this packet and use it only as retrieval priority metadata. "
)
GENERIC_QUERY_FACET_BILINGUAL_SUFFIX = (
    "For every explicit domain or procedure facet, output both Chinese and English lexical surfaces in facet_groups.aliases, regardless of the user's input language."
)
GENERIC_QUERY_FACET_ALIAS_SUFFIX = "Only include aliases when they are explicit or standard technical synonyms."
GENERIC_AGENT_PLANNER_SYSTEM = (
    "You are the Layered P&E planner for a Four-Layer Context Graph RAG system. "
    "Return strict JSON with a typed_actions array. Each action must include action_type, target_ids, reason, "
    "budget_request, expected_evidence, and stop_condition. You may only choose from the supplied action space. "
    "Policy and runtime settings provide only the operating envelope; do not invent facts."
)
GENERIC_AGENT_PLANNER_REPAIR_SUFFIX = (
    "Your previous response was rejected by the typed action schema. "
    "Return ONLY a JSON object with key typed_actions. Do not include prose, markdown, analysis, or alternate keys."
)
GENERIC_CITATION_ENTAILMENT_JUDGE_SYSTEM = (
    "You are a citation entailment judge for a grounded Four-Layer Context Graph RAG system. "
    "Use only the supplied cited context and source spans. Return JSON with verifications. "
    "Each verification must include citation_index, verdict (supported, unsupported, contradicted, missing_citation, "
    "formula_table_context_missing), failure_type, confidence, and reason. Do not use outside knowledge."
)
GENERIC_MID_CONCEPT_DEFINITION_SYSTEM = (
    "You define mid-level concepts for a Four-Layer Context Graph RAG system. "
    "Use only the supplied concept packets. Return strict JSON with a concepts array. "
    "Each item must include packet_id, canonical_label, aliases, definition, scope_note, "
    "inclusion_criteria, exclusion_criteria, representative_chunk_ids, support_chunk_ids, "
    "confidence, and why_this_concept_exists."
)
GENERIC_COARSE_CONCEPT_DEFINITION_SYSTEM = (
    "You define coarse topic areas for a Four-Layer Context Graph RAG system. "
    "Use only the supplied mid concept community. Return strict JSON with coarse_label, definition, "
    "included_mid_concepts, boundary_concepts, bridge_concepts, cross_community_weak_ties, and confidence."
)
GENERIC_CONCEPT_I18N_SYSTEM = (
    "You translate derived concept metadata for a grounded Four-Layer Context Graph RAG system. "
    "Return strict JSON with an items array. For every input item, preserve id and provide: "
    "label_i18n {zh,en}, aliases_i18n {zh,en arrays}, definition_i18n {zh,en}, summary_i18n {zh,en}, "
    "scope_note_i18n {zh,en}, search_terms_i18n {zh,en arrays}. "
    "Translate technical terms accurately, keep formulas/symbols unchanged, and do not add facts beyond the source text."
)
GENERIC_CONCEPT_EDGE_I18N_SYSTEM = (
    "You translate derived concept-edge metadata for a grounded Four-Layer Context Graph RAG system. "
    "Return strict JSON with an items array. For every input item, preserve id and provide: "
    "relation_label_i18n {zh,en}, explanation_i18n {zh,en}, summary_i18n {zh,en}, search_terms_i18n {zh,en arrays}. "
    "Translate only the relationship wording; keep evidence meaning, formulas, and technical symbols unchanged."
)
GENERIC_PROFILE_ASSISTANT_SYSTEM = (
    "You are a user-profile interaction-configuration assistant for a local context-graph knowledge-base system. "
    "Return strict JSON only, with keys explanation and profile_json. "
    "explanation must be concise natural language describing prompt, UI-label, or conversation-preference changes and any boundary risks. "
    "profile_json must be a complete user_profile_v1 object using only schema_version, library_type, ui_labels, prompt_pack, and conversation_preferences. "
    "profile_json must not include profile_hash; the server will calculate and return profile_hash separately. "
    "Profiles can tune knowledge-base system prompts, interaction wording, answer style, clarification style, citation strictness expression, and no-context response text. "
    "Do not generate chunking, embedding, legacy lexical index, clustering, retrieval scoring, context-package budget, agent envelope, repair/verification budget, quality gate, policy, ontology, fallback, model, cache, database, vector-store, or runtime controls. "
    "If the user asks for engineering controls, mention in explanation that those belong in Runtime Settings and keep profile_json limited to user_profile_v1 fields. "
    "Do not include markdown fences, API keys, secrets, or instructions to save automatically. "
    "Prefer the user's language for explanation."
)

DEFAULT_ANSWER_SYSTEM_PREFIX = GENERIC_ANSWER_SYSTEM_PREFIX
DEFAULT_CONTEXT_LABEL = GENERIC_CONTEXT_LABEL
DEFAULT_NO_CONTEXT_EN = GENERIC_NO_CONTEXT_EN
DEFAULT_NO_CONTEXT_ZH = GENERIC_NO_CONTEXT_ZH
DEFAULT_QUERY_REWRITE_SYSTEM = GENERIC_QUERY_REWRITE_SYSTEM
DEFAULT_JSON_RESPONSE_FALLBACK_SYSTEM = GENERIC_JSON_RESPONSE_FALLBACK_SYSTEM
DEFAULT_REFLECTION_REVIEW_SYSTEM = GENERIC_REFLECTION_REVIEW_SYSTEM
DEFAULT_QUESTION_PERCEPTION_SYSTEM = GENERIC_QUESTION_PERCEPTION_SYSTEM
DEFAULT_ANSWER_SYSTEM_TEMPLATE = GENERIC_ANSWER_SYSTEM_TEMPLATE
DEFAULT_ANSWER_LOW_RELEVANCE_CLAUSE = GENERIC_ANSWER_LOW_RELEVANCE_CLAUSE
DEFAULT_ANSWER_NORMAL_RELEVANCE_CLAUSE = GENERIC_ANSWER_NORMAL_RELEVANCE_CLAUSE
DEFAULT_QUERY_FACET_EXTRACTOR_SYSTEM = GENERIC_QUERY_FACET_EXTRACTOR_SYSTEM
DEFAULT_QUERY_FACET_BILINGUAL_SUFFIX = GENERIC_QUERY_FACET_BILINGUAL_SUFFIX
DEFAULT_QUERY_FACET_ALIAS_SUFFIX = GENERIC_QUERY_FACET_ALIAS_SUFFIX
DEFAULT_AGENT_PLANNER_SYSTEM = GENERIC_AGENT_PLANNER_SYSTEM
DEFAULT_AGENT_PLANNER_REPAIR_SUFFIX = GENERIC_AGENT_PLANNER_REPAIR_SUFFIX
DEFAULT_CITATION_ENTAILMENT_JUDGE_SYSTEM = GENERIC_CITATION_ENTAILMENT_JUDGE_SYSTEM
DEFAULT_MID_CONCEPT_DEFINITION_SYSTEM = GENERIC_MID_CONCEPT_DEFINITION_SYSTEM
DEFAULT_COARSE_CONCEPT_DEFINITION_SYSTEM = GENERIC_COARSE_CONCEPT_DEFINITION_SYSTEM
DEFAULT_CONCEPT_I18N_SYSTEM = GENERIC_CONCEPT_I18N_SYSTEM
DEFAULT_CONCEPT_EDGE_I18N_SYSTEM = GENERIC_CONCEPT_EDGE_I18N_SYSTEM
DEFAULT_PROFILE_ASSISTANT_SYSTEM = GENERIC_PROFILE_ASSISTANT_SYSTEM

DEFAULT_PROFILE: dict[str, Any] = {
    "schema_version": ACTIVE_PROFILE_SCHEMA_VERSION,
    "library_type": BUILTIN_LIBRARY_TYPE,
    "ui_labels": {
        "library": "Knowledge Base",
        "knowledge_base": "Knowledge Base",
        "partition": "partition",
        "partition_fallback": "General",
        "entity": "concept",
        "relation": "concept relation",
        "source_route": "retrieve_sources",
        "task_route": "retrieve_tasks",
    },
    "prompt_pack": {
        "answer_system_prefix": GENERIC_ANSWER_SYSTEM_PREFIX,
        "answer_system_template": GENERIC_ANSWER_SYSTEM_TEMPLATE,
        "answer_low_relevance_clause": GENERIC_ANSWER_LOW_RELEVANCE_CLAUSE,
        "answer_normal_relevance_clause": GENERIC_ANSWER_NORMAL_RELEVANCE_CLAUSE,
        "context_label": GENERIC_CONTEXT_LABEL,
        "no_context_answer_en": GENERIC_NO_CONTEXT_EN,
        "no_context_answer_zh": GENERIC_NO_CONTEXT_ZH,
        "query_rewrite_system": GENERIC_QUERY_REWRITE_SYSTEM,
        "json_response_fallback_system": GENERIC_JSON_RESPONSE_FALLBACK_SYSTEM,
        "reflection_review_system": GENERIC_REFLECTION_REVIEW_SYSTEM,
        "question_perception_system": GENERIC_QUESTION_PERCEPTION_SYSTEM,
        "query_facet_extractor_system": GENERIC_QUERY_FACET_EXTRACTOR_SYSTEM,
        "query_facet_bilingual_suffix": GENERIC_QUERY_FACET_BILINGUAL_SUFFIX,
        "query_facet_alias_suffix": GENERIC_QUERY_FACET_ALIAS_SUFFIX,
        "agent_planner_system": GENERIC_AGENT_PLANNER_SYSTEM,
        "agent_planner_repair_suffix": GENERIC_AGENT_PLANNER_REPAIR_SUFFIX,
        "citation_entailment_judge_system": GENERIC_CITATION_ENTAILMENT_JUDGE_SYSTEM,
        "mid_concept_definition_system": GENERIC_MID_CONCEPT_DEFINITION_SYSTEM,
        "coarse_concept_definition_system": GENERIC_COARSE_CONCEPT_DEFINITION_SYSTEM,
        "concept_i18n_system": GENERIC_CONCEPT_I18N_SYSTEM,
        "concept_edge_i18n_system": GENERIC_CONCEPT_EDGE_I18N_SYSTEM,
        "profile_assistant_system": GENERIC_PROFILE_ASSISTANT_SYSTEM,
        "reflection_domain": "context-graph-grounded knowledge-base assistant",
        "citation_domain": "indexed chunk spans",
        "perception_domain": "context-graph-grounded knowledge-base agent",
        "entity_label": "grounded mid-level concepts",
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
        "agent_no_context_answer_en": "I could not find enough relevant indexed context to answer this question. If you want me to try with limited retrieved material, please tell me.",
        "agent_no_context_answer_zh": "索引资料中没有找到足够相关的上下文来回答这个问题。如果你希望我基于有限检索材料尝试回答，请告诉我。",
    },
    "conversation_preferences": {
        "default_language": "auto",
        "citation_strictness": "strict",
        "clarification_style": "concise",
    },
}

ACTIVE_PROFILE_KEYS = {"schema_version", "library_type", "ui_labels", "prompt_pack", "conversation_preferences"}
ALLOWED_PROMPT_PACK_KEYS = frozenset(DEFAULT_PROFILE["prompt_pack"].keys())
LEGACY_PROFILE_KEYS = {
    "schema_pack",
    "concept_induction_policy",
    "parsing_strategy",
    "graph_strategy",
    "retrieval_strategy",
    "quality_policy",
    "signal_induction_policy",
}

_active_profile_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("active_strategy_profile", default=None)


def profile_hash(profile_json: dict[str, Any]) -> str:
    normalized = copy.deepcopy(profile_json)
    if isinstance(normalized, dict):
        normalized.pop("profile_hash", None)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_profile_payload() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_PROFILE)


def migrate_profile_payload(profile: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(profile)
    migrated: dict[str, Any] = {
        "schema_version": ACTIVE_PROFILE_SCHEMA_VERSION,
        "library_type": source.get("library_type") if isinstance(source.get("library_type"), str) else "custom",
        "ui_labels": source.get("ui_labels") if isinstance(source.get("ui_labels"), dict) else {},
        "prompt_pack": source.get("prompt_pack") if isinstance(source.get("prompt_pack"), dict) else {},
        "conversation_preferences": source.get("conversation_preferences") if isinstance(source.get("conversation_preferences"), dict) else {},
    }
    ui_labels = migrated.setdefault("ui_labels", {})
    if isinstance(ui_labels, dict):
        if "KnowledgeBase" in ui_labels and "knowledge_base" not in ui_labels:
            ui_labels["knowledge_base"] = ui_labels.pop("KnowledgeBase")
        if "notes_route" in ui_labels and "source_route" not in ui_labels:
            ui_labels["source_route"] = ui_labels.pop("notes_route")
        if "exercise_route" in ui_labels and "task_route" not in ui_labels:
            ui_labels["task_route"] = ui_labels.pop("exercise_route")
    return migrated


def validate_profile_payload(payload: object) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Profile JSON must be an object")
    legacy_keys = sorted(key for key in payload if key in LEGACY_PROFILE_KEYS)
    profile = migrate_profile_payload(payload)
    warnings: list[str] = [
        f"{key} is ignored on the active profile path; move engineering controls to runtime settings."
        for key in legacy_keys
    ]
    if "profile_hash" in payload:
        warnings.append("profile_hash is generated by the server and ignored inside profile_json.")
    if payload.get("schema_version") and payload.get("schema_version") != ACTIVE_PROFILE_SCHEMA_VERSION:
        warnings.append(f"schema_version {payload.get('schema_version')} was migrated to {ACTIVE_PROFILE_SCHEMA_VERSION}.")
    profile["schema_version"] = ACTIVE_PROFILE_SCHEMA_VERSION
    profile.setdefault("library_type", "custom")
    for key in ("ui_labels", "prompt_pack", "conversation_preferences"):
        value = profile.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
    for key in ("ui_labels", "prompt_pack", "conversation_preferences"):
        mapping = profile[key]
        profile[key] = {
            str(item_key): item_value
            for item_key, item_value in mapping.items()
            if str(item_key).strip()
        }
    unknown_prompt_keys = sorted(key for key in profile["prompt_pack"] if key not in ALLOWED_PROMPT_PACK_KEYS)
    for key in unknown_prompt_keys:
        profile["prompt_pack"].pop(key, None)
    if unknown_prompt_keys:
        warnings.append(
            "Unsupported prompt_pack keys are ignored on the active profile path: "
            + ", ".join(unknown_prompt_keys)
            + "."
        )
    for key in list(profile):
        if key not in ACTIVE_PROFILE_KEYS:
            profile.pop(key, None)
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


def active_profile_hash(db: Session, knowledge_base_id: str | None) -> str:
    return get_active_profile_record(db, knowledge_base_id).profile_hash


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


def profile_label(profile_json: dict[str, Any], key: str, default: str) -> str:
    value = ((profile_json or {}).get("ui_labels") or {}).get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def profile_prompt(profile_json: dict[str, Any], key: str, default: str) -> str:
    value = ((profile_json or {}).get("prompt_pack") or {}).get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def render_prompt_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def profile_prompt_template(profile_json: dict[str, Any], key: str, default: str, values: dict[str, Any]) -> str:
    return render_prompt_template(profile_prompt(profile_json, key, default), values)


def profile_schema(profile_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_types": DEFAULT_CONCEPT_TYPES,
        "relation_types": DEFAULT_CONCEPT_RELATION_TYPES,
        "entity_aliases": DEFAULT_CONCEPT_TYPE_ALIASES,
        "relation_aliases": DEFAULT_CONCEPT_RELATION_ALIASES,
        "default_entity_type": "topic",
        "default_relation_type": "related_observation",
    }
