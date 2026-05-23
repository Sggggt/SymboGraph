from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from pydantic import ValidationError

from app.core.config import get_settings
from app.models import (
    Chunk,
    Concept,
    ConceptAlias,
    ConceptRelation,
    Course,
    Document,
    DocumentVersion,
    EntityMention,
    EntityMergeCandidate,
    GraphCommunitySummary,
    GraphExtractionChunkTask,
    GraphExtractionRun,
    GraphRelationCandidate,
    IngestionBatch,
)
from app.schemas import Citation, GraphExtractionPayload
from app.services.embeddings import ChatProvider
from app.services.graph_algorithms import enrich_course_graph
from app.services.ingestion_logs import emit_ingestion_log
from app.services.parsers import canonical_chapter_label, derive_chapter, is_invalid_chapter_label


def _safe_emit_log(batch_id: str | None, event: str, message: str, **payload: Any) -> None:
    """Defensive wrapper for emit_ingestion_log to guard against UnboundLocalError in async contexts."""
    import app.services.ingestion_logs as _il
    return _il.emit_ingestion_log(batch_id, event, message, **payload)
from app.services.quality.policies import ConceptQualityPolicy
from app.services.quality.signals import build_quality_signals

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - dependency is declared, fallback keeps imports robust.
    fuzz = None


ALLOWED_RELATIONS = {
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
}
GRAPH_TYPES = {"semantic", "structural", "evidence"}
SEMANTIC_ENTITY_TYPES = {"concept", "method", "formula", "metric", "algorithm", "definition", "theorem", "problem_type"}
SINGLETON_ENTITY_TYPES = {"formula", "theorem", "algorithm", "metric"}
LEGACY_RELATION_TYPE_MAP = {
    "defines": "defined_by",
    "relates_to": "related_to",
    "mentions": "related_to",
    "compares": "compares_with",
    "extends": "derives_from",
}
MIN_CONCEPT_EVIDENCE_CHUNKS = 2
MIN_CONCEPT_SPECIFICITY = 0.55
ADAPTIVE_SPECIFICITY_MIN_SAMPLE = 20
ADAPTIVE_SPECIFICITY_FLOOR = 0.35
ADAPTIVE_SPECIFICITY_CEILING = 0.60
KEYWORD_TERMS = [
    "eigenvector centrality",
    "betweenness centrality",
    "harmonic centrality",
    "closeness centrality",
    "adjacency matrix",
    "degree matrix",
    "laplacian matrix",
    "algebraic connectivity",
    "bipartite network",
    "random network",
    "community detection",
    "modularity",
    "percolation",
    "clustering coefficient",
    "connected component",
    "pagerank",
    "spectral graph theory",
]
MOJIBAKE_MARKERS = ("\ufffd", "\u00c3", "\u00c2", "\u00e2", "\u9208", "\u9365", "\u9429", "\u95b3", "\u951f", "\u7d34", "\u6d93", "\u934f")
MATH_SYMBOL_RE = re.compile(
    r"[\u2211\u222b\u2202\u221a\u221e\u2248\u2260\u2264\u2265\u00b1\u00d7\u00f7"
    r"\u2208\u2209\u2282\u2286\u222a\u2229\u2192\u2190\u2194\u2200\u2203\u2207=<>^]"
)
CONCEPT_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4}\b")


def graph_extraction_provider() -> str:
    settings = get_settings()
    if settings.openai_api_key:
        return "openai_compatible_chat"
    return "heuristic" if settings.enable_model_fallback else "unavailable"


def _text_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _text_items(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _graph_payload(value: object) -> dict:
    return value if isinstance(value, dict) else {"concepts": [], "relations": []}


def normalize_relation_type(value: object) -> str:
    relation_type = _text_value(value).lower()
    relation_type = LEGACY_RELATION_TYPE_MAP.get(relation_type, relation_type)
    return relation_type if relation_type in ALLOWED_RELATIONS else ""


def _has_graph_concepts(value: object) -> bool:
    payload = _graph_payload(value)
    concepts = payload.get("concepts", [])
    return isinstance(concepts, list) and bool(concepts)


def validate_graph_payload(value: object) -> tuple[dict, list[str]]:
    try:
        payload = GraphExtractionPayload.model_validate(value).model_dump()
    except ValidationError as exc:
        raise ValueError(f"graph payload schema validation failed: {exc}") from exc

    warnings = []
    concept_names = {concept["name"] for concept in payload["concepts"] if concept.get("name")}
    for concept in payload["concepts"]:
        concept["concept_type"] = normalize_entity_type(concept.get("entity_type") or concept.get("concept_type"))
        concept["entity_type"] = concept["concept_type"]
        concept["canonical_name"] = clean_display_name(concept.get("canonical_name") or concept.get("name"))
        concept["surface"] = clean_display_name(concept.get("surface") or concept.get("name"))
        concept["name"] = concept["canonical_name"] or concept["surface"]
        if not concept["name"]:
            warnings.append("dropped graph concept with empty canonical name")
            continue
        quality = concept_quality(concept.get("name"))
        if not quality["valid"]:
            warnings.append(f"staged concept with quality risk: {concept.get('name')!r} ({quality['reason']})")
        concept["_quality"] = quality

    for relation in payload["relations"]:
        source = relation["source"]
        target = relation["target"]
        source_quality = concept_quality(source)
        target_quality = concept_quality(target)
        if not source_quality["valid"] or not target_quality["valid"]:
            warnings.append(
                "staged relation with endpoint quality risk: "
                f"{source!r} -> {target!r} ({relation['relation_type']})"
            )
        if source not in concept_names or target not in concept_names:
            warnings.append(
                "synthesized missing relation endpoint concepts: "
                f"{source!r} -> {target!r} ({relation['relation_type']})"
            )
            for endpoint in (source, target):
                if endpoint not in concept_names:
                    payload["concepts"].append(
                        {
                            "name": endpoint,
                            "surface": endpoint,
                            "canonical_name": endpoint,
                            "aliases": [endpoint],
                            "summary": "",
                            "concept_type": "concept",
                            "entity_type": "concept",
                            "importance_score": 0.45,
                            "confidence": 0.45,
                            "evidence_spans": [],
                            "_quality": concept_quality(endpoint),
                            "_synthesized_from_relation": True,
                        }
                    )
                    concept_names.add(endpoint)
    if warnings:
        payload["_validation_warnings"] = warnings
    return payload, warnings


def normalize_concept_name(name: object) -> str:
    name = _text_value(name)
    if not name:
        return ""
    value = unicodedata.normalize("NFKC", name)
    value = re.sub(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", lambda match: match.group(1) or match.group(2) or "", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[`~*#\[\]{}]", " ", value)
    value = re.sub(r"[’‘]", "'", value)
    value = re.sub(r"\b([A-Za-z])'s\b", r"\1s", value)
    value = re.sub(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\u0370-\u03ff+\-/*^=<>()\s']", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    tokens = []
    for token in value.split():
        if token == "max":
            token = "maximum"
        elif token == "min":
            token = "minimum"
        if re.fullmatch(r"[a-z]{4,}ies", token):
            token = token[:-3] + "y"
        elif re.fullmatch(r"[a-z]{4,}s", token) and not token.endswith(("ss", "ics")):
            token = token[:-1]
        tokens.append(token)
    value = " ".join(tokens)
    return value


def normalize_entity_type(value: object) -> str:
    entity_type = _text_value(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "named_algorithm": "algorithm",
        "algo": "algorithm",
        "measure": "metric",
        "definition_term": "definition",
        "problem": "problem_type",
    }
    entity_type = aliases.get(entity_type, entity_type)
    return entity_type if entity_type in SEMANTIC_ENTITY_TYPES else "concept"


def canonical_entity_key(name: object, entity_type: object = "concept") -> str:
    normalized = normalize_concept_name(name)
    if not normalized:
        return ""
    return f"{normalized}::{normalize_entity_type(entity_type)}"


def canonical_concept_key(name: object, entity_type: object | None = None) -> str:
    if entity_type is None:
        return normalize_concept_name(name)
    return canonical_entity_key(name, entity_type)


def clean_display_name(name: object) -> str:
    name = _text_value(name)
    if not name:
        return ""
    raw = unicodedata.normalize("NFKC", name)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if raw.isupper():
        return raw
    return raw[:1].upper() + raw[1:]


def _normalized_context_terms(context_terms: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    return {normalized for item in context_terms or [] for normalized in [normalize_concept_name(item)] if normalized}


def _is_mojibake_like(raw_name: str) -> bool:
    if not raw_name:
        return False
    marker_count = sum(raw_name.count(marker) for marker in MOJIBAKE_MARKERS)
    return raw_name.count("\ufffd") > 0 or marker_count / max(len(raw_name), 1) > 0.04


def _token_count(normalized: str) -> int:
    return len([token for token in normalized.split() if token])


def _is_acronym(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,7}", value.strip())) and value.strip().upper() == value.strip()


def _acronym(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", normalize_concept_name(value))
    if len(parts) < 2:
        return ""
    return "".join(part[0] for part in parts if part).upper()


def concept_quality(name: object, context_terms: set[str] | list[str] | tuple[str, ...] | None = None) -> dict[str, object]:
    raw_name = _text_value(name)
    normalized = normalize_concept_name(name)
    if _is_acronym(raw_name) and len(raw_name.strip()) >= 2:
        return {"valid": True, "reason": "policy_passed", "normalized": normalized, "decision": {"action": "accept"}}
    signals = build_quality_signals(target_type="concept", text=raw_name, version="concept_quality_v1")
    context_blocklist = _normalized_context_terms(context_terms)
    if normalized in context_blocklist:
        return {
            "valid": False,
            "reason": "context_container",
            "normalized": normalized,
            "decision": {"action": "reject", "reasons": ["context_container"], "signals": signals.model_dump()},
        }
    decision = ConceptQualityPolicy().decide(signals, evidence_count=2, existing=False, specificity_score=0.6)
    valid = decision.action == "accept"
    reason = "policy_passed" if valid else decision.reasons[0]
    return {"valid": valid, "reason": reason, "normalized": normalized, "decision": decision.model_dump()}


HARD_REJECT_CONCEPT_REASONS = {"empty", "mojibake_noise", "path_or_filename", "context_container", "structural_container"}


def is_stageable_concept(name: object, context_terms: set[str] | list[str] | tuple[str, ...] | None = None) -> bool:
    quality = concept_quality(name, context_terms)
    return bool(quality["normalized"]) and (quality["valid"] or quality["reason"] not in HARD_REJECT_CONCEPT_REASONS)


def is_valid_concept(name: object, context_terms: set[str] | list[str] | tuple[str, ...] | None = None) -> bool:
    return bool(concept_quality(name, context_terms)["valid"])


def heuristic_extract_graph(text: str) -> dict:
    candidates: dict[str, dict] = {}
    for term in KEYWORD_TERMS:
        if term in text.lower():
            display = clean_display_name(term.title())
            candidates[normalize_concept_name(display)] = {
                "name": display,
                "surface": display,
                "canonical_name": display,
                "aliases": [display],
                "summary": "",
                "concept_type": "concept",
                "entity_type": "concept",
                "importance_score": 0.72,
                "confidence": 0.72,
            }
    for match in CONCEPT_PATTERN.findall(text):
        if not is_valid_concept(match):
            continue
        display = clean_display_name(match)
        candidates.setdefault(
            normalize_concept_name(display),
            {
                "name": display,
                "surface": display,
                "canonical_name": display,
                "aliases": [display],
                "summary": "",
                "concept_type": "concept",
                "entity_type": "concept",
                "importance_score": 0.55,
                "confidence": 0.55,
            },
        )

    names = list(candidates.values())
    relations = []
    for left, right in zip(names, names[1:]):
        relations.append(
            {
                "source": left["name"],
                "target": right["name"],
                "relation_type": "related_to",
                "confidence": 0.4,
            }
        )
    return {"concepts": list(candidates.values())[:12], "relations": relations[:18]}


def merge_graph_candidates(primary: dict, fallback: dict, context_terms: set[str] | list[str] | tuple[str, ...] | None = None) -> dict:
    merged_concepts: dict[str, dict] = {}
    endpoint_key_by_name: dict[str, str] = {}
    for source in (_graph_payload(fallback), _graph_payload(primary)):
        for concept in source.get("concepts", []):
            if not isinstance(concept, dict):
                continue
            name = _text_value(concept.get("canonical_name") or concept.get("name") or concept.get("surface"))
            if not is_stageable_concept(name, context_terms):
                continue
            concept_type = normalize_entity_type(concept.get("entity_type") or concept.get("concept_type"))
            normalized = canonical_entity_key(name, concept_type)
            for endpoint in {name, _text_value(concept.get("name")), _text_value(concept.get("surface")), *_text_items(concept.get("aliases", []))}:
                endpoint_key_by_name[normalize_concept_name(endpoint)] = normalized
            current = merged_concepts.get(normalized, {})
            aliases = {
                clean_display_name(alias)
                for alias in [
                    current.get("name", ""),
                    name,
                    _text_value(concept.get("surface")),
                    *_text_items(current.get("aliases", [])),
                    *_text_items(concept.get("aliases", [])),
                ]
                if is_stageable_concept(alias, context_terms)
            }
            if not aliases:
                aliases = {clean_display_name(name)}
            merged_concepts[normalized] = {
                "name": current.get("name") or clean_display_name(name),
                "surface": clean_display_name(_text_value(concept.get("surface")) or name),
                "canonical_name": current.get("canonical_name") or clean_display_name(name),
                "aliases": sorted(aliases),
                "summary": _text_value(concept.get("definition") or concept.get("summary", "")) or current.get("summary", ""),
                "definition": _text_value(concept.get("definition") or concept.get("summary", "")) or current.get("definition", ""),
                "concept_type": concept_type,
                "entity_type": concept_type,
                "importance_score": max(
                    _safe_float(current.get("importance_score", 0.0), 0.0),
                    _safe_float(concept.get("importance_score", 0.0), 0.0),
                ),
                "confidence": max(
                    _safe_float(current.get("confidence", 0.0), 0.0),
                    _safe_float(concept.get("confidence", concept.get("importance_score", 0.5)), 0.5),
                ),
                "evidence_spans": [
                    *(_text_items(current.get("evidence_spans", []))),
                    *(_text_items(concept.get("evidence_spans", []))),
                ][:8],
            }

    relations = []
    seen_relations: set[tuple[str, str, str]] = set()
    for source in (_graph_payload(fallback), _graph_payload(primary)):
        for relation in source.get("relations", []):
            if not isinstance(relation, dict):
                continue
            relation_type_value = relation.get("relation_type", "related_to")
            if relation_type_value is not None and not isinstance(relation_type_value, str):
                continue
            relation_type = normalize_relation_type(relation_type_value) or "related_to"
            if relation_type not in ALLOWED_RELATIONS:
                continue
            source_name = _text_value(relation.get("source", ""))
            target_name = _text_value(relation.get("target", ""))
            if not is_stageable_concept(source_name, context_terms) or not is_stageable_concept(target_name, context_terms):
                continue
            source_key = endpoint_key_by_name.get(normalize_concept_name(source_name), canonical_entity_key(source_name, "concept"))
            target_key = endpoint_key_by_name.get(normalize_concept_name(target_name), canonical_entity_key(target_name, "concept"))
            key = (source_key, target_key, relation_type)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append(
                {
                    "source": clean_display_name(source_name),
                    "target": clean_display_name(target_name),
                    "source_key": source_key,
                    "target_key": target_key,
                    "relation_type": relation_type,
                    "confidence": _safe_float(relation.get("confidence", 0.5), 0.5),
                }
            )
    return {"concepts": list(merged_concepts.values()), "relations": relations}


@dataclass
class StagedConcept:
    key: str
    name: str
    aliases: set[str] = field(default_factory=set)
    summaries: list[str] = field(default_factory=list)
    concept_type: str = "concept"
    importance_score: float = 0.0
    confidence: float = 0.0
    evidence_spans: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    document_ids: set[str] = field(default_factory=set)
    chapter_refs: set[str] = field(default_factory=set)
    heading_hits: int = 0
    definition_hits: int = 0
    merged_from: set[str] = field(default_factory=set)


@dataclass
class StagedRelation:
    source_key: str
    target_key: str
    relation_type: str
    confidence: float
    chunk_ids: set[str] = field(default_factory=set)
    document_ids: set[str] = field(default_factory=set)
    extraction_method: str = "llm"


def _record_entity_mention(
    db: Session,
    *,
    course_id: str,
    chunk: Chunk,
    group: StagedConcept,
    surface: str,
    status: str = "staged",
    decision: dict | None = None,
) -> None:
    if not surface:
        return
    mention_key = (course_id, chunk.id, surface, group.concept_type)
    for pending in db.new:
        if not isinstance(pending, EntityMention):
            continue
        pending_key = (pending.course_id, pending.chunk_id, pending.surface, pending.entity_type)
        if pending_key != mention_key:
            continue
        pending.normalized_key = group.key
        pending.canonical_name = group.name
        pending.confidence = max(float(pending.confidence or 0.0), group.confidence)
        pending.evidence_spans = sorted({*(pending.evidence_spans or []), *group.evidence_spans})
        pending.status = status
        pending.decision_json = decision or pending.decision_json
        return
    exists = db.scalar(
        select(EntityMention).where(
            EntityMention.course_id == course_id,
            EntityMention.chunk_id == chunk.id,
            EntityMention.surface == surface,
            EntityMention.entity_type == group.concept_type,
        )
    )
    if exists is not None:
        exists.normalized_key = group.key
        exists.canonical_name = group.name
        exists.confidence = max(float(exists.confidence or 0.0), group.confidence)
        exists.evidence_spans = sorted({*(exists.evidence_spans or []), *group.evidence_spans})
        exists.status = status
        exists.decision_json = decision or exists.decision_json
        return
    db.add(
        EntityMention(
            course_id=course_id,
            chunk_id=chunk.id,
            document_id=str(chunk.document_id) if chunk.document_id else None,
            surface=surface,
            canonical_name=group.name,
            normalized_key=group.key,
            entity_type=group.concept_type,
            confidence=group.confidence,
            evidence_spans=sorted(group.evidence_spans),
            status=status,
            decision_json=decision or {},
        )
    )


def semantic_relation_allowed(source: Concept, target: Concept, relation_type: str) -> bool:
    source_type = normalize_entity_type(getattr(source, "concept_type", "concept"))
    target_type = normalize_entity_type(getattr(target, "concept_type", "concept"))
    if source_type not in SEMANTIC_ENTITY_TYPES or target_type not in SEMANTIC_ENTITY_TYPES:
        return False
    if relation_type == "formula_of":
        return source_type == "formula" or target_type == "formula"
    if relation_type == "defined_by":
        return target_type in {"definition", "concept", "theorem"}
    if relation_type in {"implemented_by", "solves"}:
        return "method" in {source_type, target_type} or "algorithm" in {source_type, target_type}
    return relation_type in ALLOWED_RELATIONS


def chunk_text_for_specificity(chunk: Chunk) -> str:
    return "\n".join(
        str(value)
        for value in (
            getattr(chunk, "chapter", None),
            getattr(chunk, "section", None),
            getattr(chunk, "snippet", None),
            (getattr(chunk, "content", "") or "")[:2400],
        )
        if value
    )


def heading_presence_score(name: str, chunk: Chunk) -> float:
    normalized = canonical_concept_key(name)
    if not normalized:
        return 0.0
    headings = normalize_concept_name(f"{getattr(chunk, 'chapter', '') or ''}\n{getattr(chunk, 'section', '') or ''}")
    return 1.0 if normalized and normalized in headings else 0.0


def definition_pattern_score(name: str, chunk: Chunk) -> float:
    normalized = canonical_concept_key(name)
    if not normalized:
        return 0.0
    text = normalize_concept_name(chunk_text_for_specificity(chunk))
    index = text.find(normalized)
    if index < 0:
        return 0.0
    window = text[index : index + max(len(normalized) + 100, 120)]
    definition_markers = (
        " is ",
        " are ",
        " refers to ",
        " defined as ",
        " means ",
        " denotes ",
        " definition ",
        " 定义 ",
        " 定义为 ",
        " 是 ",
        " 指 ",
        " 称为 ",
    )
    return 1.0 if any(marker in window for marker in definition_markers) else 0.0


def idf_score_for_key(key: str, chunks: list[Chunk]) -> float:
    if not chunks or not key:
        return 0.0
    document_frequency = 0
    for chunk in chunks:
        if key in normalize_concept_name(chunk_text_for_specificity(chunk)):
            document_frequency += 1
    total = len(chunks)
    if total <= 1:
        return 1.0
    return max(0.0, min(1.0, math.log((total + 1) / (document_frequency + 1)) / math.log(total + 1)))


def concept_specificity_score(group: StagedConcept, chunks: list[Chunk]) -> tuple[float, dict]:
    idf = idf_score_for_key(group.key, chunks)
    heading = 1.0 if group.heading_hits > 0 else 0.0
    definition = 1.0 if group.definition_hits > 0 else 0.0
    importance = max(0.0, min(1.0, float(group.importance_score or 0.0)))
    score = 0.35 * idf + 0.25 * heading + 0.20 * definition + 0.20 * importance
    audit = {
        "idf_score": round(idf, 4),
        "title_or_heading_presence": heading,
        "definition_pattern_score": definition,
        "llm_concept_importance": round(importance, 4),
        "specificity_score": round(score, 4),
        "evidence_chunk_count": len(group.chunk_ids),
        "merge_provenance": sorted({group.key, *group.merged_from}),
    }
    return score, audit


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    ratio = position - lower
    return float(ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio)


def adaptive_specificity_threshold(scores: list[float]) -> dict:
    usable = [float(score) for score in scores if score is not None and float(score) > 0.0]
    base = {
        "name": "concept_specificity",
        "default_threshold": MIN_CONCEPT_SPECIFICITY,
        "sample_count": len(usable),
        "min_sample_count": ADAPTIVE_SPECIFICITY_MIN_SAMPLE,
        "floor": ADAPTIVE_SPECIFICITY_FLOOR,
        "ceiling": ADAPTIVE_SPECIFICITY_CEILING,
    }
    if len(usable) < ADAPTIVE_SPECIFICITY_MIN_SAMPLE:
        return {
            **base,
            "enabled": False,
            "threshold": MIN_CONCEPT_SPECIFICITY,
            "fallback_reason": "insufficient_samples",
        }
    p25 = _percentile(usable, 25)
    p50 = _percentile(usable, 50)
    p75 = _percentile(usable, 75)
    threshold = max(ADAPTIVE_SPECIFICITY_FLOOR, min(ADAPTIVE_SPECIFICITY_CEILING, p25))
    if p50 >= 0.70:
        threshold = max(threshold, 0.50)
    return {
        **base,
        "enabled": True,
        "threshold": round(threshold, 4),
        "p25": round(p25, 4),
        "p50": round(p50, 4),
        "p75": round(p75, 4),
        "fallback_reason": None,
    }


def concept_gate_decision(
    group: StagedConcept,
    chunks: list[Chunk],
    existing: Concept | None = None,
    *,
    specificity_threshold: float | None = None,
    threshold_audit: dict | None = None,
    precomputed_specificity: tuple[float, dict] | None = None,
) -> tuple[bool, dict]:
    score, audit = precomputed_specificity or concept_specificity_score(group, chunks)
    effective_specificity_threshold = float(specificity_threshold if specificity_threshold is not None else MIN_CONCEPT_SPECIFICITY)
    signals = build_quality_signals(
        target_type="concept",
        text=group.name,
        corpus_texts=[chunk_text_for_specificity(chunk) for chunk in chunks],
        evidence_text="\n".join(chunk_text_for_specificity(chunk) for chunk in chunks if chunk.id in group.chunk_ids),
        support_count=len(group.chunk_ids),
        version="concept_quality_v1",
    )
    decision = ConceptQualityPolicy().decide(
        signals,
        evidence_count=len(group.chunk_ids),
        existing=existing is not None,
        specificity_score=score,
    )
    accepted = decision.action == "accept"
    strong_singleton_evidence = (
        existing is None
        and len(group.chunk_ids) == 1
        and normalize_entity_type(group.concept_type) in SINGLETON_ENTITY_TYPES
        and score >= 0.60
        and group.definition_hits > 0
        and group.heading_hits == 0
        and group.confidence >= 0.72
        and not set(decision.reasons).intersection(HARD_REJECT_CONCEPT_REASONS)
    )
    if not accepted and decision.reasons == ["insufficient_evidence"] and strong_singleton_evidence:
        accepted = True
        decision_audit = decision.model_dump()
        return True, {
            **audit,
            **decision_audit,
            "accepted": True,
            "gate_reason": "strong_singleton_evidence",
            "specificity_threshold": round(effective_specificity_threshold, 4),
            "adaptive_threshold": threshold_audit,
        }
    if existing is None and accepted and score < effective_specificity_threshold:
        decision_audit = decision.model_dump()
        return False, {
            **audit,
            **decision_audit,
            "accepted": False,
            "gate_reason": "low_specificity",
            "specificity_threshold": round(effective_specificity_threshold, 4),
            "adaptive_threshold": threshold_audit,
        }
    reason = "accepted" if accepted else decision.reasons[0]
    return accepted, {
        **audit,
        **decision.model_dump(),
        "accepted": accepted,
        "gate_reason": reason,
        "specificity_threshold": round(effective_specificity_threshold, 4),
        "adaptive_threshold": threshold_audit,
    }


def _merge_quality_audit(concept: Concept, audit: dict) -> None:
    current = dict(getattr(concept, "centrality_json", None) or {})
    current["quality_gate"] = audit
    concept.centrality_json = current
    concept.quality_json = {"concept_gate": audit}


def _staged_acronyms(group: StagedConcept) -> set[str]:
    values = {group.name, *group.aliases}
    acronyms = {_acronym(value) for value in values}
    acronyms |= {value.strip().upper() for value in values if _is_acronym(value)}
    acronyms.discard("")
    return acronyms


def _staged_groups_should_merge(left: StagedConcept, right: StagedConcept) -> bool:
    if normalize_entity_type(left.concept_type) != normalize_entity_type(right.concept_type):
        return False
    left_keys = {left.key, *(canonical_entity_key(alias, left.concept_type) for alias in left.aliases)}
    right_keys = {right.key, *(canonical_entity_key(alias, right.concept_type) for alias in right.aliases)}
    if left_keys.intersection(right_keys):
        return True
    if _staged_acronyms(left).intersection(_staged_acronyms(right)):
        return True
    left_tokens = set(left.key.split("::", 1)[0].split())
    right_tokens = set(right.key.split("::", 1)[0].split())
    generic_suffix_tokens = {"problem", "model", "method", "approach", "process", "system", "task"}
    if len(left_tokens) >= 2 and left_tokens.issubset(right_tokens) and right_tokens - left_tokens <= generic_suffix_tokens:
        return True
    if len(right_tokens) >= 2 and right_tokens.issubset(left_tokens) and left_tokens - right_tokens <= generic_suffix_tokens:
        return True
    if fuzz is None:
        return False
    if min(len(left.key), len(right.key)) < 8 or MATH_SYMBOL_RE.search(f"{left.name} {right.name}"):
        return False
    token_set_ratio = fuzz.token_set_ratio(left.key, right.key)
    token_sort_ratio = fuzz.token_sort_ratio(left.key, right.key)
    if len(left_tokens) >= 2 and len(right_tokens) >= 2 and token_set_ratio >= 98:
        return True
    if abs(len(left.key) - len(right.key)) <= max(4, int(max(len(left.key), len(right.key)) * 0.25)) and token_sort_ratio >= 97:
        return True
    return False


def _is_relevant_alias(alias: str, target_name: str) -> bool:
    """Check whether an alias is semantically related to the target concept name."""
    alias_norm = normalize_concept_name(alias)
    target_norm = normalize_concept_name(target_name)
    if not alias_norm or not target_norm:
        return False
    # Direct substring match
    if alias_norm in target_norm or target_norm in alias_norm:
        return True
    # Token overlap check (at least 50% of target tokens must appear in alias)
    target_tokens = set(target_norm.split())
    alias_tokens = set(alias_norm.split())
    if not target_tokens:
        return False
    overlap = len(target_tokens & alias_tokens)
    return overlap >= max(1, len(target_tokens) * 0.5)


def _merge_staged_group(target: StagedConcept, source: StagedConcept) -> None:
    source_tokens = set(source.key.split("::", 1)[0].split())
    target_tokens = set(target.key.split("::", 1)[0].split())
    if source_tokens and source_tokens.issubset(target_tokens) and len(source.key) < len(target.key):
        target.name = source.name
        target.key = source.key
    elif len(source.chunk_ids) > len(target.chunk_ids) and len(source.name) <= len(target.name) + 12:
        target.name = source.name
        target.key = source.key
    # Filter aliases: only keep those relevant to the target concept
    relevant_aliases = {a for a in source.aliases if _is_relevant_alias(a, target.name)}
    if _is_relevant_alias(source.name, target.name):
        relevant_aliases.add(source.name)
    target.aliases.update(relevant_aliases)
    target.summaries.extend(source.summaries)
    target.importance_score = max(target.importance_score, source.importance_score)
    target.confidence = max(target.confidence, source.confidence)
    target.evidence_spans.update(source.evidence_spans)
    target.concept_type = normalize_entity_type(target.concept_type)
    target.chunk_ids.update(source.chunk_ids)
    target.document_ids.update(source.document_ids)
    target.chapter_refs.update(source.chapter_refs)
    target.heading_hits += source.heading_hits
    target.definition_hits += source.definition_hits
    target.merged_from.update(source.merged_from)
    target.merged_from.add(source.key)
    # Cap total aliases to prevent alias explosion
    MAX_ALIASES = 50
    if len(target.aliases) > MAX_ALIASES:
        # Prioritize: shortest aliases first (more likely to be core synonyms)
        target.aliases = set(sorted(target.aliases, key=lambda a: (len(a), a))[:MAX_ALIASES])


def merge_staged_concept_groups(groups: dict[str, StagedConcept]) -> tuple[dict[str, StagedConcept], dict[str, str]]:
    merged: list[StagedConcept] = []
    key_map: dict[str, str] = {}
    for group in sorted(groups.values(), key=lambda item: (-len(item.chunk_ids), item.key)):
        destination = next((candidate for candidate in merged if _staged_groups_should_merge(candidate, group)), None)
        if destination is None:
            merged.append(group)
            key_map[group.key] = group.key
            continue
        original_key = destination.key
        _merge_staged_group(destination, group)
        key_map[group.key] = destination.key
        if destination.key != original_key:
            for source_key, target_key in list(key_map.items()):
                if target_key == original_key:
                    key_map[source_key] = destination.key
    return {group.key: group for group in merged}, key_map


def llm_merge_candidate_pairs(groups: dict[str, StagedConcept], limit: int = 24) -> list[tuple[str, str]]:
    items = sorted(groups.values(), key=lambda item: (-len(item.chunk_ids), item.key))
    pairs: list[tuple[float, str, str]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if _staged_groups_should_merge(left, right):
                continue
            evidence_overlap = bool(left.chunk_ids.intersection(right.chunk_ids))
            chapter_overlap = bool(left.chapter_refs.intersection(right.chapter_refs))
            score = 0.0
            if evidence_overlap:
                score += 0.35
            if chapter_overlap:
                score += 0.20
            if _staged_acronyms(left).intersection(_staged_acronyms(right)):
                score += 0.25
            if fuzz is not None and min(len(left.key), len(right.key)) >= 8:
                score += max(fuzz.token_set_ratio(left.key, right.key), fuzz.token_sort_ratio(left.key, right.key)) / 100.0
            if score >= 0.75:
                pairs.append((score, left.key, right.key))
    return [(left, right) for _score, left, right in sorted(pairs, reverse=True)[:limit]]


async def apply_llm_verified_staged_merges(groups: dict[str, StagedConcept], limit: int = 24) -> tuple[dict[str, StagedConcept], dict[str, str], int]:
    pairs = llm_merge_candidate_pairs(groups, limit=limit)
    if not pairs:
        return groups, {key: key for key in groups}, 0
    provider = ChatProvider()
    key_map = {key: key for key in groups}
    verified = 0
    for left_key, right_key in pairs:
        left = groups.get(key_map.get(left_key, left_key))
        right = groups.get(key_map.get(right_key, right_key))
        if left is None or right is None or left.key == right.key:
            continue
        system_prompt = "Verify whether two course knowledge graph concepts are aliases of the same concept. Return strict JSON only."
        user_prompt = (
            "Return JSON with keys should_merge (boolean), canonical_name (string), reason (string), confidence (number 0-1). "
            "Merge only clear aliases, abbreviations, translations, plural/case/hyphen variants, or the same mathematical concept. "
            "Do not merge merely related concepts.\n\n"
            f"Concept A: {left.name}; aliases={sorted(left.aliases)}; chapters={sorted(left.chapter_refs)}\n"
            f"Concept B: {right.name}; aliases={sorted(right.aliases)}; chapters={sorted(right.chapter_refs)}"
        )
        result = await provider.classify_json(system_prompt, user_prompt)
        should_merge = bool(result.get("should_merge"))
        confidence = _safe_float(result.get("confidence", 0.0), 0.0)
        if not should_merge or confidence < 0.75:
            continue
        canonical_name = clean_display_name(result.get("canonical_name") or left.name)
        _merge_staged_group(left, right)
        if canonical_name and is_valid_concept(canonical_name):
            left.name = canonical_name
            left.key = canonical_entity_key(canonical_name, left.concept_type) if "::" in left.key else canonical_concept_key(canonical_name)
        left.merged_from.add(f"llm:{right.key}")
        groups.pop(right.key, None)
        key_map[left_key] = left.key
        key_map[right_key] = left.key
        key_map[right.key] = left.key
        verified += 1
    normalized_groups = {group.key: group for group in groups.values()}
    for source_key, target_key in list(key_map.items()):
        key_map[source_key] = target_key if "::" in target_key else canonical_concept_key(target_key)
    return normalized_groups, key_map, verified


def _find_existing_concept_for_group(db: Session, course_id: str, group: StagedConcept) -> Concept | None:
    concept = _find_exact_alias_match(db, course_id, group.key)
    if concept is None:
        concept = db.scalar(select(Concept).where(Concept.course_id == course_id, Concept.normalized_name == group.key))
    if concept is None:
        concept = _find_alias_variant_match(db, course_id, group.name, group.key, sorted(group.aliases), group.concept_type)
    return concept


async def upsert_graph_candidates_from_chunks(
    db: Session,
    course_id: str,
    chunks: list[Chunk],
    llm_payloads: dict[str, dict] | None = None,
    run_llm_merge: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    llm_payloads = llm_payloads or {}
    staged_concepts: dict[str, StagedConcept] = {}
    staged_relations: dict[tuple[str, str, str], StagedRelation] = {}
    llm_success_chunks = 0

    for chunk in chunks:
        doc_id = str(chunk.document_id) if chunk.document_id else None
        context_terms = concept_context_terms(db, course_id, chunk.chapter, doc_id)
        fallback = heuristic_extract_graph(chunk.content) if settings.enable_model_fallback else {"concepts": [], "relations": []}
        llm_payload = llm_payloads.get(chunk.id)
        has_llm_concepts = _has_graph_concepts(llm_payload)
        if chunk.id in llm_payloads:
            llm_success_chunks += 1
        extracted = merge_graph_candidates(llm_payload or {"concepts": [], "relations": []}, fallback, context_terms=context_terms)
        extraction_method = "llm+rules" if settings.enable_model_fallback and has_llm_concepts else "llm" if has_llm_concepts else "heuristic"

        for concept_data in extracted["concepts"]:
            name = clean_display_name(concept_data.get("canonical_name") or concept_data["name"])
            concept_type = normalize_entity_type(concept_data.get("entity_type") or concept_data.get("concept_type"))
            key = canonical_entity_key(name, concept_type)
            if not key:
                continue
            group = staged_concepts.get(key)
            if group is None:
                group = StagedConcept(key=key, name=name, concept_type=concept_type)
                staged_concepts[key] = group
            group.aliases.update(
                clean_display_name(alias)
                for alias in [name, concept_data.get("surface", ""), *concept_data.get("aliases", [])]
                if is_valid_concept(alias, context_terms)
            )
            summary = _text_value(concept_data.get("definition") or concept_data.get("summary", ""))
            if summary:
                group.summaries.append(summary)
            group.concept_type = concept_type
            group.importance_score = max(group.importance_score, _safe_float(concept_data.get("importance_score", 0.5), 0.5))
            group.confidence = max(group.confidence, _safe_float(concept_data.get("confidence", concept_data.get("importance_score", 0.5)), 0.5))
            group.evidence_spans.update(_text_items(concept_data.get("evidence_spans", [])))
            group.chunk_ids.add(chunk.id)
            if doc_id:
                group.document_ids.add(doc_id)
            if chunk.chapter and not is_invalid_chapter_label(chunk.chapter):
                group.chapter_refs.add(chunk.chapter)
            group.heading_hits += int(heading_presence_score(name, chunk) > 0)
            group.definition_hits += int(definition_pattern_score(name, chunk) > 0)
            _record_entity_mention(
                db,
                course_id=course_id,
                chunk=chunk,
                group=group,
                surface=clean_display_name(concept_data.get("surface") or name),
            )

        for relation_data in extracted["relations"]:
            source_key = _text_value(relation_data.get("source_key")) or canonical_concept_key(relation_data["source"])
            target_key = _text_value(relation_data.get("target_key")) or canonical_concept_key(relation_data["target"])
            relation_type = normalize_relation_type(relation_data.get("relation_type")) or "related_to"
            if not source_key or not target_key or source_key == target_key:
                continue
            relation_key = (source_key, target_key, relation_type)
            existing_relation = staged_relations.get(relation_key)
            if existing_relation is None:
                staged_relations[relation_key] = StagedRelation(
                    source_key=source_key,
                    target_key=target_key,
                    relation_type=relation_type,
                    confidence=_safe_float(relation_data.get("confidence", 0.55), 0.55),
                    extraction_method=extraction_method,
                )
                existing_relation = staged_relations[relation_key]
            existing_relation.chunk_ids.add(chunk.id)
            if doc_id:
                existing_relation.document_ids.add(doc_id)
            existing_relation.confidence = max(
                existing_relation.confidence,
                _safe_float(relation_data.get("confidence", 0.55), 0.55),
            )

    staged_concepts, concept_key_map = merge_staged_concept_groups(staged_concepts)
    for left_key, right_key in llm_merge_candidate_pairs(staged_concepts):
        left = staged_concepts.get(left_key)
        right = staged_concepts.get(right_key)
        if left is None or right is None:
            continue
        existing_candidate = db.scalar(
            select(EntityMergeCandidate).where(
                EntityMergeCandidate.course_id == course_id,
                EntityMergeCandidate.left_key == min(left.key, right.key),
                EntityMergeCandidate.right_key == max(left.key, right.key),
            )
        )
        if existing_candidate is None:
            db.add(
                EntityMergeCandidate(
                    course_id=course_id,
                    left_key=min(left.key, right.key),
                    right_key=max(left.key, right.key),
                    entity_type=normalize_entity_type(left.concept_type),
                    source="lexical_embedding_candidate",
                    score=0.0,
                    status="pending",
                )
            )
    llm_verified_merges = 0
    if run_llm_merge and settings.openai_api_key and len(staged_concepts) <= 160:
        staged_concepts, llm_key_map, llm_verified_merges = await apply_llm_verified_staged_merges(staged_concepts)
        for source_key, target_key in llm_key_map.items():
            concept_key_map[source_key] = target_key
    for relation_data in staged_relations.values():
        relation_data.source_key = concept_key_map.get(relation_data.source_key, relation_data.source_key)
        relation_data.target_key = concept_key_map.get(relation_data.target_key, relation_data.target_key)

    specificity_by_key = {
        key: concept_specificity_score(group, chunks)
        for key, group in staged_concepts.items()
    }
    specificity_threshold_audit = adaptive_specificity_threshold([score for score, _audit in specificity_by_key.values()])
    specificity_threshold = float(specificity_threshold_audit["threshold"])
    concept_map: dict[str, Concept] = {}
    created_count = 0
    rejected_concepts = 0
    for key, group in staged_concepts.items():
        existing = _find_existing_concept_for_group(db, course_id, group)
        accepted, audit = concept_gate_decision(
            group,
            chunks,
            existing=existing,
            specificity_threshold=specificity_threshold,
            threshold_audit=specificity_threshold_audit,
            precomputed_specificity=specificity_by_key.get(key),
        )
        if not accepted:
            rejected_concepts += 1
            continue
        try:
            concept, created = get_or_create_concept(
                db=db,
                course_id=course_id,
                name=group.name,
                chapter=sorted(group.chapter_refs)[0] if group.chapter_refs else None,
                summary=max(group.summaries, key=len) if group.summaries else "",
                aliases=sorted(group.aliases),
                concept_type=group.concept_type,
                importance_score=group.importance_score,
                document_id=sorted(group.document_ids)[0] if group.document_ids else None,
                # Candidate extraction and admission already used chunk/document context.
                # Re-applying chapter labels here would reject legitimate entities such as
                # "Centrality" in a "Centralities" chapter.
                context_terms=set(),
            )
        except ValueError:
            rejected_concepts += 1
            continue
        concept.source_document_ids = sorted({*(concept.source_document_ids or []), *group.document_ids})
        concept.chapter_refs = sorted({*(concept.chapter_refs or []), *group.chapter_refs})
        concept.evidence_count = max(int(concept.evidence_count or 0), len(group.chunk_ids))
        _merge_quality_audit(concept, audit)
        for alias in group.aliases:
            alias_key = canonical_entity_key(alias, group.concept_type)
            if alias_key:
                concept_map[alias_key] = concept
        concept_map[key] = concept
        if created:
            created_count += 1

    relation_count = 0
    for relation_data in staged_relations.values():
        source = concept_map.get(relation_data.source_key)
        target = concept_map.get(relation_data.target_key)
        if source is None or target is None or source.id == target.id:
            continue
        if relation_data.confidence < 0.55 or not semantic_relation_allowed(source, target, relation_data.relation_type):
            for chunk_id in relation_data.chunk_ids:
                db.add(
                    GraphRelationCandidate(
                        course_id=course_id,
                        source_concept_id=source.id,
                        target_concept_id=target.id,
                        target_name=target.canonical_name,
                        relation_type=relation_data.relation_type,
                        relation_source="schema_rejected",
                        evidence_chunk_id=chunk_id,
                        confidence=relation_data.confidence,
                        weight=relation_data.confidence,
                        is_inferred=False,
                        decision_json={"action": "candidate_only", "reason": "semantic_relation_schema_rejected"},
                        metadata_json={"source_entity_type": source.concept_type, "target_entity_type": target.concept_type},
                        source_document_ids=sorted(relation_data.document_ids),
                    )
                )
            continue
        existing = db.scalar(
            select(ConceptRelation).where(
                ConceptRelation.course_id == course_id,
                ConceptRelation.source_concept_id == source.id,
                ConceptRelation.target_concept_id == target.id,
                ConceptRelation.relation_type == relation_data.relation_type,
            )
        )
        if existing:
            existing.confidence = max(float(existing.confidence or 0.0), relation_data.confidence)
            existing.weight = max(float(getattr(existing, "weight", 0.0) or 0.0), relation_data.confidence)
            existing.support_count = int(getattr(existing, "support_count", 1) or 1) + len(relation_data.chunk_ids)
            existing.relation_source = "llm"
            existing.extraction_method = "llm+rules"
            if relation_data.confidence >= float(existing.confidence or 0.0) and relation_data.chunk_ids:
                existing.evidence_chunk_id = max(relation_data.chunk_ids)
            merged_docs = set(existing.source_document_ids or [])
            merged_docs.update(relation_data.document_ids)
            existing.source_document_ids = sorted(merged_docs)
            continue
        db.add(
            ConceptRelation(
                course_id=course_id,
                source_concept_id=source.id,
                target_concept_id=target.id,
                target_name=target.canonical_name,
                relation_type=relation_data.relation_type,
                evidence_chunk_id=max(relation_data.chunk_ids) if relation_data.chunk_ids else None,
                confidence=relation_data.confidence,
                extraction_method=relation_data.extraction_method,
                is_validated=False,
                weight=relation_data.confidence,
                semantic_similarity=0.0,
                support_count=len(relation_data.chunk_ids),
                relation_source="llm",
                is_inferred=False,
                metadata_json={"hard_gate": "pending"},
                source_document_ids=sorted(relation_data.document_ids),
            )
        )
        relation_count += 1

    db.flush()
    return {
        "created_concepts": created_count,
        "relations": relation_count,
        "rejected_concepts": rejected_concepts,
        "llm_success_chunks": llm_success_chunks,
        "llm_verified_merges": llm_verified_merges,
        "graph_concept_specificity_threshold": specificity_threshold,
        "graph_concept_specificity_threshold_audit": specificity_threshold_audit,
    }


def concept_context_terms(db: Session, course_id: str, chapter: str | None = None, document_id: str | None = None) -> set[str]:
    terms: set[str] = set()
    course = db.get(Course, course_id)
    if course and course.name:
        terms.add(course.name)
    if chapter:
        terms.add(chapter)
        canonical = canonical_chapter_label(chapter, course.name if course else None)
        if canonical:
            terms.add(canonical)
    if document_id:
        document = db.get(Document, document_id)
        if document:
            if document.title:
                terms.add(document.title)
            if document.source_path:
                source_path = Path(document.source_path)
                terms.add(source_path.name)
                terms.add(source_path.stem)
            for tag in document.tags or []:
                terms.add(tag)
    return terms


def _find_exact_alias_match(db: Session, course_id: str, normalized: str) -> Concept | None:
    alias_matches = db.scalars(select(ConceptAlias).where(ConceptAlias.normalized_alias == normalized)).all()
    for alias_match in alias_matches:
        if alias_match.concept and alias_match.concept.course_id == course_id:
            return alias_match.concept
    return None


def _course_concepts(db: Session, course_id: str) -> list[Concept]:
    return db.scalars(select(Concept).where(Concept.course_id == course_id)).all()


def _find_alias_variant_match(db: Session, course_id: str, name: str, normalized: str, aliases: list[str], concept_type: str = "concept") -> Concept | None:
    candidates = [name, *aliases]
    candidate_acronyms = {_acronym(candidate) for candidate in candidates}
    candidate_acronyms |= {candidate.strip().upper() for candidate in candidates if _is_acronym(candidate)}
    candidate_acronyms.discard("")

    course_concepts = _course_concepts(db, course_id)
    for concept in course_concepts:
        if normalize_entity_type(concept.concept_type) != normalize_entity_type(concept_type):
            continue
        concept_names = [concept.canonical_name, *(alias.alias for alias in concept.aliases)]
        concept_acronyms = {_acronym(value) for value in concept_names}
        concept_acronyms |= {value.strip().upper() for value in concept_names if _is_acronym(value)}
        concept_acronyms.discard("")
        if candidate_acronyms and concept_acronyms and candidate_acronyms.intersection(concept_acronyms):
            return concept

    if fuzz is None or len(normalized) < 8 or MATH_SYMBOL_RE.search(name):
        return None
    for concept in course_concepts:
        if normalize_entity_type(concept.concept_type) != normalize_entity_type(concept_type):
            continue
        existing_values = {concept.normalized_name, normalize_concept_name(concept.canonical_name)}
        existing_values.update(normalize_concept_name(alias.alias) for alias in concept.aliases)
        for existing in existing_values:
            if not existing or abs(len(existing) - len(normalized)) > max(4, int(max(len(existing), len(normalized)) * 0.25)):
                continue
            ratio = max(fuzz.ratio(normalized, existing), fuzz.token_sort_ratio(normalized, existing))
            if ratio >= 96:
                return concept
    return None


def get_or_create_concept(
    db: Session,
    course_id: str,
    name: str,
    chapter: str | None,
    summary: str,
    aliases: list[str],
    concept_type: str,
    importance_score: float,
    document_id: str | None = None,
    context_terms: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[Concept, bool]:
    context_terms = set(concept_context_terms(db, course_id, chapter, document_id) if context_terms is None else context_terms)
    quality = concept_quality(name, context_terms)
    if not is_stageable_concept(name, context_terms):
        raise ValueError(f"invalid concept name {name!r}: {quality['reason']}")
    entity_type = normalize_entity_type(concept_type)
    normalized = canonical_entity_key(name, entity_type)
    chapter_ref = None if is_invalid_chapter_label(chapter) else chapter
    concept = _find_exact_alias_match(db, course_id, normalized)
    if concept is None:
        concept = db.scalar(select(Concept).where(Concept.course_id == course_id, Concept.normalized_name == normalized))
    if concept is None:
        concept = _find_alias_variant_match(db, course_id, name, normalized, aliases, entity_type)

    created = False
    if concept is None:
        concept = Concept(
            course_id=course_id,
            canonical_name=clean_display_name(name),
            normalized_name=normalized,
            summary=summary[:800],
            concept_type=entity_type,
            importance_score=importance_score,
            chapter_refs=[chapter_ref] if chapter_ref else [],
            source_document_ids=[document_id] if document_id else [],
        )
        db.add(concept)
        db.flush()
        created = True
    else:
        if summary and len(summary) > len(concept.summary or ""):
            concept.summary = summary[:800]
        concept.importance_score = max(float(concept.importance_score or 0.0), float(importance_score or 0.0))
        concept.concept_type = normalize_entity_type(concept.concept_type or entity_type)
        if chapter_ref and chapter_ref not in concept.chapter_refs:
            concept.chapter_refs = sorted({*concept.chapter_refs, chapter_ref})
        if document_id and document_id not in (concept.source_document_ids or []):
            concept.source_document_ids = sorted({*(concept.source_document_ids or []), document_id})

    all_aliases = {
        clean_display_name(concept.canonical_name),
        *[clean_display_name(alias) for alias in aliases if is_stageable_concept(alias, context_terms)],
    }
    if is_stageable_concept(name, context_terms):
        all_aliases.add(clean_display_name(name))
    for alias in all_aliases:
        normalized_alias = canonical_entity_key(alias, entity_type)
        if not normalized_alias:
            continue
        exists = db.scalar(
            select(ConceptAlias).where(
                ConceptAlias.concept_id == concept.id,
                ConceptAlias.normalized_alias == normalized_alias,
            )
        )
        if exists is None:
            db.add(ConceptAlias(concept_id=concept.id, alias=alias, normalized_alias=normalized_alias))
    return concept, created


async def upsert_concepts_from_chunk(
    db: Session,
    course_id: str,
    chunk: Chunk,
    use_llm: bool = True,
    llm_payload: dict | None = None,
) -> tuple[int, int]:
    settings = get_settings()
    doc_id = str(chunk.document_id) if chunk.document_id else None
    context_terms = concept_context_terms(db, course_id, chunk.chapter, doc_id)
    fallback = heuristic_extract_graph(chunk.content) if settings.enable_model_fallback else {"concepts": [], "relations": []}
    if use_llm:
        llm_payload = llm_payload if llm_payload is not None else await ChatProvider().extract_graph_payload(chunk.content, chunk.chapter, chunk.source_type)
        llm_payload, _warnings = validate_graph_payload(llm_payload)
    else:
        llm_payload = {"concepts": [], "relations": []}
    extracted = merge_graph_candidates(llm_payload, fallback, context_terms=context_terms)
    has_llm_concepts = _has_graph_concepts(llm_payload)
    extraction_method = "llm+rules" if settings.enable_model_fallback and has_llm_concepts else "llm" if has_llm_concepts else "heuristic"

    concept_map: dict[str, Concept] = {}
    created_count = 0
    for concept_data in extracted["concepts"]:
        entity_type = normalize_entity_type(concept_data.get("entity_type") or concept_data.get("concept_type"))
        concept, created = get_or_create_concept(
            db=db,
            course_id=course_id,
            name=concept_data.get("canonical_name") or concept_data["name"],
            chapter=chunk.chapter,
            summary=concept_data.get("definition") or concept_data.get("summary") or chunk.snippet,
            aliases=[concept_data.get("surface", ""), *concept_data.get("aliases", [])],
            concept_type=entity_type,
            importance_score=_safe_float(concept_data.get("importance_score", 0.5), 0.5),
            document_id=doc_id,
            context_terms=context_terms,
        )
        for key_source in [concept_data["name"], *concept_data.get("aliases", [])]:
            key = canonical_entity_key(key_source, entity_type)
            if key:
                concept_map[key] = concept
        if created:
            created_count += 1

    relation_count = 0
    relation_keys: set[tuple[str, str, str]] = set()
    for relation_data in extracted["relations"]:
        source = concept_map.get(_text_value(relation_data.get("source_key")) or canonical_concept_key(relation_data["source"]))
        target = concept_map.get(_text_value(relation_data.get("target_key")) or canonical_concept_key(relation_data["target"]))
        if source is None or target is None or source.id == target.id:
            continue
        relation_key = (source.id, target.id, relation_data["relation_type"])
        if relation_key in relation_keys:
            continue
        relation_keys.add(relation_key)
        confidence = _safe_float(relation_data.get("confidence", 0.55), 0.55)
        existing = db.scalar(
            select(ConceptRelation).where(
                ConceptRelation.course_id == course_id,
                ConceptRelation.source_concept_id == source.id,
                ConceptRelation.target_concept_id == target.id,
                ConceptRelation.relation_type == relation_data["relation_type"],
            )
        )
        if existing:
            existing.confidence = max(existing.confidence, confidence)
            existing.weight = max(float(getattr(existing, "weight", 0.0) or 0.0), confidence)
            existing.support_count = int(getattr(existing, "support_count", 1) or 1) + 1
            existing.relation_source = "llm"
            if confidence >= (existing.confidence or 0):
                existing.evidence_chunk_id = chunk.id
            existing.extraction_method = "llm+rules"
            if doc_id and doc_id not in (existing.source_document_ids or []):
                existing.source_document_ids = sorted({*(existing.source_document_ids or []), doc_id})
            continue
        db.add(
            ConceptRelation(
                course_id=course_id,
                source_concept_id=source.id,
                target_concept_id=target.id,
                target_name=target.canonical_name,
                relation_type=relation_data["relation_type"],
                evidence_chunk_id=chunk.id,
                confidence=confidence,
                extraction_method=extraction_method,
                is_validated=confidence >= 0.82,
                weight=confidence,
                semantic_similarity=0.0,
                support_count=1,
                relation_source="llm",
                is_inferred=False,
                metadata_json={},
                source_document_ids=[doc_id] if doc_id else [],
            )
        )
        relation_count += 1
    return created_count, relation_count


def graph_topic_score(chunk: Chunk) -> int:
    metadata = getattr(chunk, "metadata_json", None) or {}
    routes = metadata.get("route_eligibility") or {}
    if metadata.get("quality_retain") is False or metadata.get("quality_action") == "discard":
        return -100
    signals = build_quality_signals(
        target_type="chunk",
        text=getattr(chunk, "content", "") or "",
        title=getattr(chunk, "section", None),
        section=getattr(chunk, "section", None),
        content_kind=metadata.get("content_kind"),
        metadata=metadata,
        course_id=getattr(chunk, "course_id", None),
        document_id=getattr(chunk, "document_id", None),
        document_version_id=getattr(chunk, "document_version_id", None),
        chunk_id=getattr(chunk, "id", None),
        version="graph_chunk_selection_v1",
    )
    score = 0
    if routes.get("graph_extraction") is True:
        score += 4
    elif metadata.get("quality_action") == "graph_candidate":
        score += 3
    if routes.get("retrieval") is True:
        score += 2
    if routes.get("evidence_only") is True:
        score -= 2
    score += int(signals.semantic_density.definition_score)
    score += int(signals.semantic_density.term_density >= 0.2)
    score += int(signals.semantic_density.entity_density >= 0.08)
    domain = signals.domain_specificity
    score += int((domain.local_idf or domain.specificity_score or 0.0) >= 0.45)
    score += int((domain.mutual_information or 0.0) >= 0.25)
    score += int((domain.entropy or 0.0) >= 0.45)
    if signals.text_quality.toc_like or signals.text_quality.mojibake_ratio > 0.01:
        score -= 4
    return score


def graph_chunk_rank(chunk: Chunk) -> tuple[int, int, int, int]:
    priority = {"markdown": 4, "pdf_page": 4, "doc_section": 4, "slide": 4, "text": 3, "html": 3, "ocr": 3, "code": 0, "output": 0}
    return (
        graph_topic_score(chunk),
        priority.get((chunk.metadata_json or {}).get("content_kind", "text"), 2),
        len(chunk.content),
        1 if chunk.source_type == "notebook" else 2,
    )


@dataclass(frozen=True)
class GraphExtractionPlan:
    selected_chunk_ids: list[str]
    selected_reasons: dict[str, dict]
    skipped_reasons: dict[str, str]
    coverage: dict
    budget: dict
    stop_reason: str


def _soft_start_budget() -> int:
    settings = get_settings()
    return int(settings.graph_extraction_soft_start_budget or 120)


def _chunk_token_estimate(chunk: Chunk) -> int:
    return max(1, int(getattr(chunk, "token_count", 0) or math.ceil(len(getattr(chunk, "content", "") or "") / 4)))


def _chunk_hash(chunk: Chunk) -> str:
    metadata = getattr(chunk, "metadata_json", None) or {}
    identity = {
        "content": getattr(chunk, "content", "") or "",
        "quality_policy": metadata.get("quality_policy"),
        "quality_action": metadata.get("quality_action"),
        "route_eligibility": metadata.get("route_eligibility"),
    }
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _chunk_terms(chunk: Chunk) -> set[str]:
    text = normalize_concept_name(getattr(chunk, "content", "") or "")
    return {token for token in re.findall(r"[0-9a-z\u3400-\u4dbf\u4e00-\u9fff\u0370-\u03ff]{2,}", text) if token}


def _chunk_cluster_key(chunk: Chunk) -> str:
    metadata = getattr(chunk, "metadata_json", None) or {}
    content_kind = str(metadata.get("content_kind") or getattr(chunk, "source_type", "") or "text")
    length = len(getattr(chunk, "content", "") or "")
    length_bucket = "short" if length < 400 else "medium" if length < 1400 else "long"
    term_sample = "-".join(sorted(_chunk_terms(chunk))[:3])
    digest = hashlib.sha1(f"{content_kind}:{length_bucket}:{term_sample}".encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{content_kind}:{length_bucket}:{digest}"


def _eligible_graph_extraction_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [
        chunk
        for chunk in chunks
        if ((chunk.metadata_json or {}).get("quality_retain") is not False)
        and ((chunk.metadata_json or {}).get("quality_action") != "discard")
    ]


def _document_frequency(chunks: list[Chunk]) -> Counter[str]:
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(_chunk_terms(chunk))
    return df


def _chunk_feature_scores(chunk: Chunk, df: Counter[str], total_chunks: int) -> dict[str, float]:
    metadata = getattr(chunk, "metadata_json", None) or {}
    signals = build_quality_signals(
        target_type="chunk",
        text=getattr(chunk, "content", "") or "",
        title=getattr(chunk, "section", None),
        section=getattr(chunk, "section", None),
        content_kind=metadata.get("content_kind"),
        metadata=metadata,
        course_id=getattr(chunk, "course_id", None),
        document_id=getattr(chunk, "document_id", None),
        document_version_id=getattr(chunk, "document_version_id", None),
        chunk_id=getattr(chunk, "id", None),
        version="adaptive_graph_extraction_v1",
    )
    terms = _chunk_terms(chunk)
    if terms and total_chunks > 1:
        idf_gain = sum(math.log((total_chunks + 1) / (df.get(term, 0) + 1)) / math.log(total_chunks + 1) for term in terms) / len(terms)
    else:
        idf_gain = 0.0
    routes = metadata.get("route_eligibility") or {}
    semantic = signals.semantic_density
    formula_table_definition = max(float(semantic.has_formula), float(semantic.has_table), semantic.definition_score)
    semantic_density = min(1.0, 0.45 * semantic.term_density + 0.25 * semantic.entity_density + 0.20 * semantic.unique_token_ratio + 0.10 * formula_table_definition)
    mi_gain = min(1.0, (len(terms) * max(semantic.term_density, 0.01)) / 30.0)
    structural_penalty = 0.18 * float(signals.text_quality.toc_like) + 0.30 * min(1.0, signals.text_quality.mojibake_ratio * 20)
    if routes.get("evidence_only"):
        structural_penalty += 0.08
    return {
        "semantic_density_gain": round(semantic_density, 4),
        "definition_formula_table_gain": round(formula_table_definition, 4),
        "low_frequency_idf_gain": round(max(0.0, min(1.0, idf_gain)), 4),
        "mi_cooccurrence_gain": round(mi_gain, 4),
        "structural_noise_penalty": round(structural_penalty, 4),
        "graph_topic_score": float(graph_topic_score(chunk)),
    }


def plan_adaptive_graph_extraction_chunks(chunks: list[Chunk], *, existing_concept_gaps: dict | None = None) -> GraphExtractionPlan:
    settings = get_settings()
    eligible = _eligible_graph_extraction_chunks(chunks)
    if not eligible:
        return GraphExtractionPlan([], {}, {}, {"eligible_chunks": 0}, {}, "no_eligible_chunks")

    soft_start_budget = _soft_start_budget()
    max_tokens = settings.graph_extraction_max_input_tokens_per_run
    max_calls = settings.graph_extraction_max_model_calls_per_run
    min_gain = float(settings.graph_extraction_min_marginal_gain)
    stall_rounds = int(settings.graph_extraction_stall_rounds)
    df = _document_frequency(eligible)
    total_chunks = len(eligible)
    all_documents = {str(chunk.document_id) for chunk in eligible if chunk.document_id}
    all_chapters = {str(getattr(chunk, "chapter", None) or "unknown") for chunk in eligible}
    all_sections = {str(getattr(chunk, "section", None) or getattr(chunk, "chapter", None) or "unknown") for chunk in eligible}
    all_kinds = {str((getattr(chunk, "metadata_json", None) or {}).get("content_kind") or getattr(chunk, "source_type", None) or "text") for chunk in eligible}
    cluster_by_chunk = {chunk.id: _chunk_cluster_key(chunk) for chunk in eligible}
    all_clusters = set(cluster_by_chunk.values())
    all_terms = {term for term, count in df.items() if count <= max(3, int(total_chunks * 0.08))}
    terms_by_chunk = {str(chunk.id): _chunk_terms(chunk) for chunk in eligible}
    token_by_chunk = {str(chunk.id): _chunk_token_estimate(chunk) for chunk in eligible}
    features_by_chunk = {str(chunk.id): _chunk_feature_scores(chunk, df, total_chunks) for chunk in eligible}
    static_by_chunk = {
        str(chunk.id): {
            "document": str(getattr(chunk, "document_id", "") or ""),
            "chapter": str(getattr(chunk, "chapter", None) or "unknown"),
            "section": str(getattr(chunk, "section", None) or getattr(chunk, "chapter", None) or "unknown"),
            "kind": str((getattr(chunk, "metadata_json", None) or {}).get("content_kind") or getattr(chunk, "source_type", None) or "text"),
            "cluster": cluster_by_chunk.get(str(chunk.id), "unknown"),
        }
        for chunk in eligible
    }

    selected: list[Chunk] = []
    selected_ids: set[str] = set()
    selected_reasons: dict[str, dict] = {}
    covered_documents: set[str] = set()
    covered_chapters: set[str] = set()
    covered_sections: set[str] = set()
    covered_kinds: set[str] = set()
    covered_clusters: set[str] = set()
    covered_terms: set[str] = set()
    token_total = 0
    stall_count = 0
    stop_reason = "exhausted_candidates"

    remaining = list(eligible)
    while remaining:
        best_chunk: Chunk | None = None
        best_reason: dict | None = None
        best_priority = -999.0
        for chunk in remaining:
            chunk_id = str(chunk.id)
            terms = terms_by_chunk.get(chunk_id, set())
            static = static_by_chunk[chunk_id]
            doc = static["document"]
            chapter = static["chapter"]
            section = static["section"]
            kind = static["kind"]
            cluster = static["cluster"]
            coverage_gain = 0.0
            coverage_parts: list[str] = []
            if doc and doc not in covered_documents:
                coverage_gain += 0.18
                coverage_parts.append("document")
            if chapter not in covered_chapters:
                coverage_gain += 0.16
                coverage_parts.append("chapter")
            if section not in covered_sections:
                coverage_gain += 0.10
                coverage_parts.append("section")
            if kind not in covered_kinds:
                coverage_gain += 0.08
                coverage_parts.append("content_kind")
            if cluster not in covered_clusters:
                coverage_gain += 0.12
                coverage_parts.append("embedding_cluster_proxy")
            rare_new_terms = terms.intersection(all_terms) - covered_terms
            if rare_new_terms:
                coverage_gain += min(0.18, 0.015 * len(rare_new_terms))
                coverage_parts.append("low_frequency_terms")

            features = features_by_chunk[chunk_id]
            token_cost = token_by_chunk[chunk_id]
            token_penalty = min(0.12, token_cost / 12000.0)
            graph_gap_gain = 0.0
            if existing_concept_gaps and chapter in set(existing_concept_gaps.get("chapters", [])):
                graph_gap_gain = 0.12
                coverage_parts.append("graph_gap")

            priority = (
                coverage_gain
                + 0.24 * features["semantic_density_gain"]
                + 0.18 * features["definition_formula_table_gain"]
                + 0.16 * features["low_frequency_idf_gain"]
                + 0.10 * features["mi_cooccurrence_gain"]
                + graph_gap_gain
                + max(0.0, features["graph_topic_score"]) * 0.015
                - token_penalty
                - features["structural_noise_penalty"]
            )
            reason = {
                **features,
                "marginal_coverage_gain": round(coverage_gain, 4),
                "graph_gap_gain": round(graph_gap_gain, 4),
                "estimated_tokens": token_cost,
                "coverage_parts": coverage_parts,
                "priority": round(priority, 4),
            }
            if priority > best_priority:
                best_priority = priority
                best_chunk = chunk
                best_reason = reason

        if best_chunk is None or best_reason is None:
            break
        token_estimate = token_by_chunk[str(best_chunk.id)]
        if max_tokens is not None and token_total + token_estimate > max_tokens and selected:
            stop_reason = "token_budget_reached"
            break
        if max_calls is not None and len(selected) >= max_calls:
            stop_reason = "model_call_budget_reached"
            break

        marginal_gain = float(best_reason["marginal_coverage_gain"])
        if len(selected) >= soft_start_budget and marginal_gain < min_gain:
            stall_count += 1
            if stall_count >= stall_rounds:
                stop_reason = "marginal_gain_stalled"
                break
        else:
            stall_count = 0

        selected.append(best_chunk)
        selected_ids.add(str(best_chunk.id))
        selected_reasons[str(best_chunk.id)] = best_reason
        token_total += token_estimate
        best_static = static_by_chunk[str(best_chunk.id)]
        covered_documents.add(best_static["document"])
        covered_chapters.add(best_static["chapter"])
        covered_sections.add(best_static["section"])
        covered_kinds.add(best_static["kind"])
        covered_clusters.add(best_static["cluster"])
        covered_terms.update(terms_by_chunk[str(best_chunk.id)].intersection(all_terms))
        remaining = [chunk for chunk in remaining if str(chunk.id) not in selected_ids]

    skipped_reasons = {
        str(chunk.id): "not_selected_by_adaptive_marginal_gain"
        for chunk in eligible
        if str(chunk.id) not in selected_ids
    }
    coverage = {
        "eligible_chunks": len(eligible),
        "selected_chunks": len(selected),
        "documents": {"covered": len(covered_documents), "total": len(all_documents)},
        "chapters": {"covered": len(covered_chapters), "total": len(all_chapters), "uncovered": sorted(all_chapters - covered_chapters)[:50]},
        "sections": {"covered": len(covered_sections), "total": len(all_sections)},
        "content_kinds": {"covered": len(covered_kinds), "total": len(all_kinds)},
        "embedding_cluster_proxy": {"covered": len(covered_clusters), "total": len(all_clusters)},
        "low_frequency_terms": {"covered": len(covered_terms), "total": len(all_terms)},
    }
    budget = {
        "strategy": settings.graph_extraction_strategy,
        "soft_start_budget": soft_start_budget,
        "max_input_tokens_per_run": max_tokens,
        "max_model_calls_per_run": max_calls,
        "min_marginal_gain": min_gain,
        "stall_rounds": stall_rounds,
        "estimated_input_tokens": token_total,
    }
    return GraphExtractionPlan(
        selected_chunk_ids=[str(chunk.id) for chunk in selected],
        selected_reasons=selected_reasons,
        skipped_reasons=skipped_reasons,
        coverage=coverage,
        budget=budget,
        stop_reason=stop_reason,
    )


def choose_graph_probe_chunks(chunks: list[Chunk], limit: int = 3) -> list[Chunk]:
    if len(chunks) <= limit:
        return chunks[:]
    ranked_by_length = sorted(chunks, key=lambda chunk: len(chunk.content or ""))
    indexes = {0, len(ranked_by_length) // 2, len(ranked_by_length) - 1}
    return [ranked_by_length[index] for index in sorted(indexes)][:limit]


async def run_llm_graph_extraction(chunks: list[Chunk], batch_id: str | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    try:
        extraction_result = await extract_llm_graph_payloads(chunks, batch_id=batch_id)
    except TypeError as exc:
        if "unexpected keyword argument 'batch_id'" not in str(exc):
            raise
        extraction_result = await extract_llm_graph_payloads(chunks)
    if isinstance(extraction_result, tuple):
        return extraction_result
    return extraction_result, {}


TERMINAL_GRAPH_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped"}


def has_resumable_graph_extraction(db: Session, course_id: str) -> bool:
    runs = db.scalars(
        select(GraphExtractionRun)
        .where(
            GraphExtractionRun.course_id == course_id,
            GraphExtractionRun.strategy == "adaptive_best_first",
            GraphExtractionRun.status.in_(["planned", "running", "partial_failed", "failed"]),
        )
        .order_by(GraphExtractionRun.created_at.desc())
        .limit(10)
    ).all()
    for run in runs:
        if run.batch_id:
            batch_status = db.scalar(select(IngestionBatch.status).where(IngestionBatch.id == run.batch_id))
            if batch_status is None or batch_status in TERMINAL_GRAPH_BATCH_STATES:
                continue
        if db.scalar(
            select(GraphExtractionChunkTask.id)
            .where(
                GraphExtractionChunkTask.run_id == run.id,
                GraphExtractionChunkTask.status.in_(["pending", "failed"]),
            )
            .limit(1)
        ):
            return True
    return False


def _latest_reusable_task_payloads(
    db: Session,
    course_id: str,
    *,
    profile_version: str | None,
    prompt_version: str,
    model: str | None,
) -> dict[str, GraphExtractionChunkTask]:
    runs = db.scalars(
        select(GraphExtractionRun)
        .where(
            GraphExtractionRun.course_id == course_id,
            GraphExtractionRun.strategy == "adaptive_best_first",
            GraphExtractionRun.profile_version == profile_version,
            GraphExtractionRun.prompt_version == prompt_version,
            GraphExtractionRun.model == model,
        )
        .order_by(GraphExtractionRun.created_at.desc())
        .limit(5)
    ).all()
    if not runs:
        return {}
    tasks = db.scalars(
        select(GraphExtractionChunkTask)
        .where(
            GraphExtractionChunkTask.run_id.in_([run.id for run in runs]),
            GraphExtractionChunkTask.status == "completed",
        )
        .order_by(GraphExtractionChunkTask.updated_at.desc())
    ).all()
    reusable: dict[str, GraphExtractionChunkTask] = {}
    for task in tasks:
        reusable.setdefault(task.chunk_id, task)
    return reusable


def create_graph_extraction_run_from_plan(
    db: Session,
    *,
    course_id: str,
    batch_id: str | None,
    chunks: list[Chunk],
    plan: GraphExtractionPlan,
    profile_version: str | None,
) -> GraphExtractionRun:
    settings = get_settings()
    prompt_version = "graph_extraction_v1"
    model = settings.chat_model
    reusable = _latest_reusable_task_payloads(
        db,
        course_id,
        profile_version=profile_version,
        prompt_version=prompt_version,
        model=model,
    )
    chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
    run = GraphExtractionRun(
        course_id=course_id,
        batch_id=batch_id,
        strategy="adaptive_best_first",
        profile_version=profile_version,
        prompt_version=prompt_version,
        model=model,
        status="planned",
        coverage_json=plan.coverage,
        budget_json=plan.budget,
        stats_json={
            "selected_chunks": len(plan.selected_chunk_ids),
            "reused_completed_chunks": 0,
            "pending_chunks": 0,
            "stop_reason": plan.stop_reason,
        },
    )
    db.add(run)
    db.flush()
    reused_count = 0
    pending_count = 0
    for index, chunk_id in enumerate(plan.selected_chunk_ids):
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        chunk_hash = _chunk_hash(chunk)
        previous = reusable.get(chunk_id)
        status = "pending"
        payload = None
        attempts = 0
        error_message = None
        if previous and previous.chunk_hash == chunk_hash and previous.payload_json:
            status = "completed"
            payload = previous.payload_json
            attempts = previous.attempts
            reused_count += 1
        else:
            pending_count += 1
        db.add(
            GraphExtractionChunkTask(
                run_id=run.id,
                course_id=course_id,
                chunk_id=chunk_id,
                chunk_hash=chunk_hash,
                priority=float(plan.selected_reasons.get(chunk_id, {}).get("priority", 0.0)),
                selected_reason=plan.selected_reasons.get(chunk_id, {}),
                status=status,
                attempts=attempts,
                payload_json=payload,
                error_message=error_message,
                token_estimate=_chunk_token_estimate(chunk),
            )
        )
    run.stats_json = {
        **(run.stats_json or {}),
        "reused_completed_chunks": reused_count,
        "pending_chunks": pending_count,
        "selected_chunks": reused_count + pending_count,
    }
    db.flush()
    return run


async def execute_graph_extraction_run(
    db: Session,
    *,
    run: GraphExtractionRun,
    chunks: list[Chunk],
    batch_id: str | None = None,
) -> tuple[dict[str, dict], dict[str, str], dict]:
    settings = get_settings()
    chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
    try:
        return await _execute_graph_extraction_run_inner(db, run=run, chunks=chunks, batch_id=batch_id, settings=settings, chunk_by_id=chunk_by_id)
    except BaseException:
        # P1-10: Ensure run never stays stuck in 'running' on unhandled exceptions.
        try:
            db.rollback()
            run_check = db.get(GraphExtractionRun, run.id)
            if run_check is not None and run_check.status == "running":
                pending_count = db.query(GraphExtractionChunkTask).filter(
                    GraphExtractionChunkTask.run_id == run.id,
                    GraphExtractionChunkTask.status.in_(["pending", "failed"]),
                ).count()
                run_check.status = "failed"
                run_check.error_message = "execute_graph_extraction_run crashed with unhandled exception"
                run_check.completed_at = datetime.utcnow()
                run_check.stats_json = {
                    **(run_check.stats_json or {}),
                    "crashed": True,
                    "remaining_pending_or_failed": pending_count,
                    "resume_available": pending_count > 0,
                }
                db.commit()
        except Exception:
            pass  # Best-effort cleanup; original exception is re-raised below.
        raise


async def _execute_graph_extraction_run_inner(
    db: Session,
    *,
    run: GraphExtractionRun,
    chunks: list[Chunk],
    batch_id: str | None = None,
    settings: Any = None,
    chunk_by_id: dict[str, Chunk] | None = None,
) -> tuple[dict[str, dict], dict[str, str], dict]:
    chunk_by_id = chunk_by_id or {str(chunk.id): chunk for chunk in chunks}
    tasks = db.scalars(
        select(GraphExtractionChunkTask)
        .where(GraphExtractionChunkTask.run_id == run.id)
        .order_by(GraphExtractionChunkTask.priority.desc(), GraphExtractionChunkTask.created_at.asc())
    ).all()
    pending_task_refs = [(task.id, task.chunk_id) for task in tasks if task.status in {"pending", "failed"}]
    run.status = "running" if pending_task_refs else "completed"
    run.started_at = run.started_at or datetime.utcnow()
    db.commit()

    if pending_task_refs:
        probe_refs = pending_task_refs[: min(3, len(pending_task_refs))]
        probe_chunks = [chunk_by_id[chunk_id] for _task_id, chunk_id in probe_refs if chunk_id in chunk_by_id]
        if batch_id and probe_chunks:
            emit_ingestion_log(
                batch_id,
                "batch_graph_probe_started",
                f"正在用 {len(probe_chunks)} 个自适应片段进行图谱抽取轻量预检",
                graph_extraction_run_id=run.id,
                probe_chunks=len(probe_chunks),
                probe_chunk_ids=[chunk.id for chunk in probe_chunks],
            )
        db.commit()
        probe_payloads, probe_errors = await run_llm_graph_extraction(probe_chunks, batch_id=batch_id)
        probe_task_ids = [task_id for task_id, _chunk_id in probe_refs]
        probe_tasks = db.scalars(
            select(GraphExtractionChunkTask).where(GraphExtractionChunkTask.id.in_(probe_task_ids))
        ).all()
        for task in probe_tasks:
            task.attempts += 1
            if task.chunk_id in probe_payloads:
                task.status = "completed"
                task.payload_json = probe_payloads[task.chunk_id]
                task.error_message = None
            elif task.chunk_id in probe_errors:
                task.status = "failed"
                task.error_message = probe_errors[task.chunk_id]
        db.commit()
        if probe_chunks and not probe_payloads:
            run.status = "failed"
            run.error_message = next(iter(probe_errors.values()), "model did not return graph extraction probe results")
            run.completed_at = datetime.utcnow()
            run.stats_json = {**(run.stats_json or {}), "failed_chunks": len(probe_errors), "resume_available": True}
            db.commit()
            if batch_id:
                emit_ingestion_log(
                    batch_id,
                    "batch_graph_resume_available",
                    "图谱抽取预检失败，已保留任务状态，可重试继续生成",
                    graph_extraction_run_id=run.id,
                    pending_graph_chunks=len(pending_task_refs),
                    error=run.error_message,
                )
            raise RuntimeError(f"图谱抽取轻量预检失败：{run.error_message}")

    remaining_tasks = db.scalars(
        select(GraphExtractionChunkTask)
        .where(GraphExtractionChunkTask.run_id == run.id, GraphExtractionChunkTask.status.in_(["pending", "failed"]))
        .order_by(GraphExtractionChunkTask.priority.desc(), GraphExtractionChunkTask.created_at.asc())
    ).all()
    batch_size = int(settings.graph_extraction_resume_batch_size)
    for start in range(0, len(remaining_tasks), batch_size):
        batch_refs = [(task.id, task.chunk_id) for task in remaining_tasks[start : start + batch_size]]
        batch_chunks = [chunk_by_id[chunk_id] for _task_id, chunk_id in batch_refs if chunk_id in chunk_by_id]
        if batch_id:
            emit_ingestion_log(
                batch_id,
                "batch_graph_adaptive_round_started",
                f"自适应图谱抽取批次 {start // batch_size + 1}，片段 {len(batch_chunks)} 个",
                graph_extraction_run_id=run.id,
                batch_chunk_count=len(batch_chunks),
            )
        db.commit()
        payloads, errors = await run_llm_graph_extraction(batch_chunks, batch_id=batch_id)
        batch_task_ids = [task_id for task_id, _chunk_id in batch_refs]
        batch_tasks = db.scalars(
            select(GraphExtractionChunkTask).where(GraphExtractionChunkTask.id.in_(batch_task_ids))
        ).all()
        for task in batch_tasks:
            task.attempts += 1
            if task.chunk_id in payloads:
                task.status = "completed"
                task.payload_json = payloads[task.chunk_id]
                task.error_message = None
            elif task.chunk_id in errors:
                task.status = "failed"
                task.error_message = errors[task.chunk_id]
        db.flush()
        completed = db.query(GraphExtractionChunkTask).filter(GraphExtractionChunkTask.run_id == run.id, GraphExtractionChunkTask.status == "completed").count()
        failed = db.query(GraphExtractionChunkTask).filter(GraphExtractionChunkTask.run_id == run.id, GraphExtractionChunkTask.status == "failed").count()
        pending = db.query(GraphExtractionChunkTask).filter(GraphExtractionChunkTask.run_id == run.id, GraphExtractionChunkTask.status == "pending").count()
        run.stats_json = {
            **(run.stats_json or {}),
            "completed_chunks": completed,
            "failed_chunks": failed,
            "pending_chunks": pending,
            "resume_available": bool(failed or pending),
        }
        db.commit()
        if batch_id:
            emit_ingestion_log(
                batch_id,
                "batch_graph_coverage_updated",
                f"自适应图谱抽取进度：完成 {completed}，待处理 {pending}，失败 {failed}",
                graph_extraction_run_id=run.id,
                completed_graph_chunks=completed,
                pending_graph_chunks=pending,
                failed_graph_chunks=failed,
                total_graph_chunks=completed + pending + failed,
                graph_extraction_coverage=run.coverage_json,
            )

    tasks = db.scalars(select(GraphExtractionChunkTask).where(GraphExtractionChunkTask.run_id == run.id)).all()
    payloads = {task.chunk_id: task.payload_json for task in tasks if task.status == "completed" and task.payload_json}
    errors = {task.chunk_id: task.error_message or "graph extraction failed" for task in tasks if task.status == "failed"}
    pending = [task for task in tasks if task.status == "pending"]
    run.status = "partial_failed" if errors or pending else "completed"
    run.completed_at = datetime.utcnow()
    run.stats_json = {
        **(run.stats_json or {}),
        "completed_chunks": len(payloads),
        "failed_chunks": len(errors),
        "pending_chunks": len(pending),
        "resume_available": bool(errors or pending),
    }
    db.commit()
    if batch_id and (errors or pending):
        emit_ingestion_log(
            batch_id,
            "batch_graph_resume_available",
            "图谱抽取部分失败，已保留成功结果，可重试继续生成",
            graph_extraction_run_id=run.id,
            completed_graph_chunks=len(payloads),
            pending_graph_chunks=len(pending),
            failed_graph_chunks=len(errors),
        )
    return payloads, errors, run.stats_json or {}


async def extract_llm_graph_payloads(
    chunks: list[Chunk],
    concurrency: int | None = None,
    batch_id: str | None = None,
) -> tuple[dict[str, dict], dict[str, str]]:
    provider = ChatProvider()
    settings = get_settings()
    effective_concurrency = int(concurrency or settings.graph_extraction_concurrency)
    semaphore = asyncio.Semaphore(effective_concurrency)
    total = len(chunks)
    completed = 0
    payloads: dict[str, dict] = {}
    errors: dict[str, str] = {}

    async def extract(chunk: Chunk) -> None:
        nonlocal completed
        async with semaphore:
            try:
                payload, _warnings = validate_graph_payload(await provider.extract_graph_payload(chunk.content, chunk.chapter, chunk.source_type))
                payloads[chunk.id] = payload
            except Exception as exc:
                errors[chunk.id] = f"{type(exc).__name__}: {exc}"
            finally:
                completed += 1
                if batch_id and (completed == total or completed % 5 == 0 or chunk.id in errors):
                    emit_ingestion_log(
                        batch_id,
                        "batch_graph_progress",
                        f"图谱抽取进度 {completed}/{total} 个片段",
                        completed_graph_chunks=completed,
                        total_graph_chunks=total,
                        successful_extractions=len(payloads),
                        failed_extractions=len(errors),
                    )

    if not chunks:
        return {}, {}
    await asyncio.gather(*(extract(chunk) for chunk in chunks))
    return payloads, errors


COMMUNITY_SUMMARY_PROMPT_VERSION = "community_summary_v1"


def _community_summary_version(course_id: str) -> str:
    return f"{COMMUNITY_SUMMARY_PROMPT_VERSION}:{course_id}"


def _concept_rank_value(concept: Concept) -> float:
    return max(
        _safe_float(getattr(concept, "graph_rank_score", None), 0.0),
        _safe_float(getattr(concept, "importance_score", None), 0.0),
        _safe_float(getattr(concept, "evidence_count", None), 0.0) / 10.0,
    )


def _representative_chunks_for_community(
    db: Session,
    course_id: str,
    concept_ids: list[str],
    *,
    limit: int = 6,
) -> tuple[list[str], list[str], list[dict]]:
    if not concept_ids:
        return [], [], []
    relations = db.scalars(
        select(ConceptRelation).where(
            ConceptRelation.course_id == course_id,
            ConceptRelation.evidence_chunk_id.is_not(None),
            or_(
                ConceptRelation.source_concept_id.in_(concept_ids),
                ConceptRelation.target_concept_id.in_(concept_ids),
            ),
        )
    ).all()
    scored_chunk_ids: dict[str, float] = {}
    relation_samples: list[dict] = []
    for relation in relations:
        source = db.get(Concept, relation.source_concept_id)
        target = db.get(Concept, relation.target_concept_id) if relation.target_concept_id else None
        chunk_id = str(relation.evidence_chunk_id)
        score = _safe_float(relation.weight, 0.0) or _safe_float(relation.confidence, 0.0)
        scored_chunk_ids[chunk_id] = max(scored_chunk_ids.get(chunk_id, 0.0), score)
        if len(relation_samples) < 12:
            relation_samples.append(
                {
                    "source": source.canonical_name if source else relation.source_concept_id,
                    "target": target.canonical_name if target else relation.target_name,
                    "relation_type": relation.relation_type,
                    "evidence_chunk_id": chunk_id,
                    "weight": round(score, 4),
                }
            )
    ranked_chunk_ids = [
        chunk_id
        for chunk_id, _score in sorted(scored_chunk_ids.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    if not ranked_chunk_ids:
        return [], [], relation_samples
    chunks = db.scalars(
        select(Chunk).where(
            Chunk.course_id == course_id,
            Chunk.id.in_(ranked_chunk_ids),
            Chunk.is_active.is_(True),
        )
    ).all()
    by_id = {str(chunk.id): chunk for chunk in chunks}
    ordered_chunk_ids = [chunk_id for chunk_id in ranked_chunk_ids if chunk_id in by_id]
    source_document_ids = sorted({str(by_id[chunk_id].document_id) for chunk_id in ordered_chunk_ids})
    return ordered_chunk_ids, source_document_ids, relation_samples


async def rebuild_graph_community_summaries(db: Session, course_id: str, *, batch_id: str | None = None) -> dict:
    """Generate active Louvain community summaries for routing/planning."""

    concepts = db.scalars(
        select(Concept).where(
            Concept.course_id == course_id,
            Concept.community_louvain.is_not(None),
        )
    ).all()
    communities: dict[int, list[Concept]] = defaultdict(list)
    for concept in concepts:
        if concept.community_louvain is not None:
            communities[int(concept.community_louvain)].append(concept)
    if not communities:
        db.query(GraphCommunitySummary).filter(GraphCommunitySummary.course_id == course_id).update(
            {"is_active": False},
            synchronize_session=False,
        )
        return {"community_summary_count": 0, "community_summary_prompt_version": COMMUNITY_SUMMARY_PROMPT_VERSION}

    chat = ChatProvider()
    version = _community_summary_version(course_id)
    db.query(GraphCommunitySummary).filter(GraphCommunitySummary.course_id == course_id).update(
        {"is_active": False},
        synchronize_session=False,
    )
    db.query(GraphCommunitySummary).filter(
        GraphCommunitySummary.course_id == course_id,
        GraphCommunitySummary.algorithm == "louvain",
        GraphCommunitySummary.version == version,
    ).delete(synchronize_session=False)
    created = 0
    for community_id, community_concepts in sorted(communities.items(), key=lambda item: item[0]):
        ranked = sorted(community_concepts, key=_concept_rank_value, reverse=True)
        top_concepts = [
                {
                    "id": concept.id,
                    "name": concept.canonical_name,
                    "type": concept.concept_type,
                    "rank": round(_concept_rank_value(concept), 4),
                    "evidence_count": concept.evidence_count,
            }
            for concept in ranked[:12]
        ]
        representative_chunk_ids, source_document_ids, relation_samples = _representative_chunks_for_community(
            db,
            course_id,
            [concept.id for concept in ranked[:20]],
        )
        sample_chunks = db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(representative_chunk_ids), Chunk.course_id == course_id)
            .limit(6)
        ).all()
        chapter_distribution: dict[str, int] = defaultdict(int)
        chunk_samples = []
        for chunk in sample_chunks:
            chapter_distribution[chunk.chapter or "unknown"] += 1
            chunk_samples.append(
                {
                    "chunk_id": chunk.id,
                    "chapter": chunk.chapter,
                    "snippet": (chunk.snippet or chunk.content or "")[:700],
                }
            )
        user_prompt = json.dumps(
            {
                "course_id": course_id,
                "algorithm": "louvain",
                "community_id": community_id,
                "top_concepts": top_concepts,
                "relation_samples": relation_samples,
                "chapter_distribution": dict(chapter_distribution),
                "representative_chunks": chunk_samples,
            },
            ensure_ascii=False,
        )
        response = await chat.classify_json(
            (
                "You summarize graph communities for evidence-first RAG routing. "
                "Return JSON with keys summary, key_concepts, routing_hints, quality_notes. "
                "Do not invent facts outside the supplied concepts, relations, and chunk snippets."
            ),
            user_prompt,
            fallback={"summary": "", "key_concepts": [], "routing_hints": [], "quality_notes": []},
        )
        summary_text = _text_value(response.get("summary")) or "; ".join(item["name"] for item in top_concepts[:5])
        db.add(
            GraphCommunitySummary(
                course_id=course_id,
                algorithm="louvain",
                community_id=community_id,
                version=version,
                summary=summary_text,
                key_concepts_json=response.get("key_concepts") if isinstance(response.get("key_concepts"), list) else top_concepts,
                representative_chunk_ids=representative_chunk_ids,
                source_document_ids=source_document_ids,
                prompt_version=COMMUNITY_SUMMARY_PROMPT_VERSION,
                model=chat.settings.chat_model,
                quality_json={
                    "routing_hints": response.get("routing_hints", []),
                    "quality_notes": response.get("quality_notes", []),
                    "concept_count": len(community_concepts),
                    "representative_chunk_count": len(representative_chunk_ids),
                },
                is_active=True,
            )
        )
        created += 1
        if batch_id:
            _safe_emit_log(
                batch_id,
                "batch_graph_community_summary",
                f"社区摘要 {created}/{len(communities)} 已生成",
                community_id=community_id,
                community_summary_count=created,
                total_communities=len(communities),
            )
    return {
        "community_summary_count": created,
        "community_summary_prompt_version": COMMUNITY_SUMMARY_PROMPT_VERSION,
        "community_summary_version": version,
    }


async def _run_auto_hpo_after_graph_upsert(
    db: Session,
    *,
    course_id: str,
    batch_id: str | None,
    probe_chunks: list[Chunk],
    payloads: dict[str, dict],
    baseline_context_chunks: list[Chunk],
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.enable_auto_hpo or not batch_id:
        return {"hpo_auto_enabled": False}
    from app.services.hpo_engine import HyperparameterTuningService

    _safe_emit_log(batch_id, "hpo_started", "正在基于本轮图谱抽取结果优化图谱超参数 (Auto HPO)...")
    try:
        hpo_results = await HyperparameterTuningService.tune_corpus_parameters(
            db=db,
            course_id=course_id,
            llm_model_name=settings.chat_model,
            probe_chunks=probe_chunks,
            payloads=payloads,
            baseline_context_chunks=baseline_context_chunks,
            batch_id=batch_id,
            commit=False,
        )
        params = hpo_results.get("hyperparameters", {})
        _safe_emit_log(
            batch_id,
            "hpo_completed",
            f"超参优化完成：Dijkstra={params.get('dijkstra_semantic_threshold')}, relation={params.get('min_relation_confidence')}",
            **hpo_results,
        )
        return {"hpo_auto_enabled": True, "hpo_status": "completed", "hpo": hpo_results}
    except Exception as exc:
        _safe_emit_log(batch_id, "hpo_failed", f"自动调参失败，使用上次成功参数或默认值: {exc}")
        return {"hpo_auto_enabled": True, "hpo_status": "failed", "hpo_error": str(exc)}


async def rebuild_course_graph(db: Session, course_id: str, batch_id: str | None = None) -> dict:
    course = db.get(Course, course_id)
    sync_graph_chapter_labels(db, course_id)
    active_documents = db.scalars(select(Document).where(Document.course_id == course_id, Document.is_active.is_(True))).all()
    graph_documents = filter_graph_documents(course, active_documents)
    graph_document_ids = {document.id for document in graph_documents}
    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.course_id == course_id, Chunk.is_active.is_(True), Chunk.document_id.in_(graph_document_ids))
        .order_by(Chunk.created_at.asc())
    ).all()
    from app.services.quality.profiles import rebuild_domain_quality_profile

    quality_profile = rebuild_domain_quality_profile(db, course_id)
    plan = plan_adaptive_graph_extraction_chunks(chunks)
    extraction_run = create_graph_extraction_run_from_plan(
        db,
        course_id=course_id,
        batch_id=batch_id,
        chunks=chunks,
        plan=plan,
        profile_version=quality_profile.version,
    )
    db.commit()
    llm_chunk_ids = set(plan.selected_chunk_ids)
    selected_llm_chunks = [chunk for chunk in chunks if chunk.id in llm_chunk_ids]
    if batch_id:
        _safe_emit_log(
            batch_id,
            "batch_graph_plan_created",
            f"自适应图谱抽取计划已创建：选择 {len(selected_llm_chunks)} / {len(chunks)} 个片段",
            graph_extraction_run_id=extraction_run.id,
            graph_extraction_strategy="adaptive_best_first",
            graph_extraction_soft_start_budget=plan.budget.get("soft_start_budget"),
            selected_llm_chunks=len(selected_llm_chunks),
            graph_source_documents=len(graph_document_ids),
            total_active_chunks=len(chunks),
            graph_extraction_coverage=plan.coverage,
            graph_extraction_stop_reason=plan.stop_reason,
        )
    llm_payloads, llm_errors, extraction_stats = await execute_graph_extraction_run(
        db,
        run=extraction_run,
        chunks=chunks,
        batch_id=batch_id,
    )
    llm_validation_warnings = {
        chunk_id: payload.get("_validation_warnings", [])
        for chunk_id, payload in llm_payloads.items()
        if payload.get("_validation_warnings")
    }
    if llm_chunk_ids and not llm_payloads:
        sample_error = next(iter(llm_errors.values()), "模型没有返回图谱抽取结果")
        raise RuntimeError(f"所有已选片段的图谱抽取均失败：{sample_error}")
    llm_document_ids = {chunk.document_id for chunk in chunks if chunk.id in llm_chunk_ids}

    try:
        _backup_course_graph_tables(db, course_id)
    except Exception as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("图谱备份失败，course_id=%s", course_id)
        raise RuntimeError(f"图谱备份失败，中止重建: {exc}") from exc
    db.query(ConceptRelation).filter(ConceptRelation.course_id == course_id).delete(synchronize_session=False)
    db.query(GraphRelationCandidate).filter(GraphRelationCandidate.course_id == course_id).delete(synchronize_session=False)
    db.query(EntityMention).filter(EntityMention.course_id == course_id).delete(synchronize_session=False)
    db.query(EntityMergeCandidate).filter(EntityMergeCandidate.course_id == course_id).delete(synchronize_session=False)
    concept_ids = [concept.id for concept in db.scalars(select(Concept).where(Concept.course_id == course_id)).all()]
    if concept_ids:
        db.query(ConceptAlias).filter(ConceptAlias.concept_id.in_(concept_ids)).delete(synchronize_session=False)
        db.query(Concept).filter(Concept.id.in_(concept_ids)).delete(synchronize_session=False)

    _safe_emit_log(
        batch_id,
        "graph_upsert_started",
        "正在写入本轮图谱候选和实体证据",
        stage="graph_upsert",
        graph_extraction_completed_chunks=extraction_stats.get("completed_chunks", len(llm_payloads)),
        graph_llm_success_chunks=len(llm_payloads),
    )
    upsert_stats = await upsert_graph_candidates_from_chunks(
        db,
        course_id,
        chunks,
        llm_payloads=llm_payloads,
        run_llm_merge=batch_id is not None,
    )
    _safe_emit_log(
        batch_id,
        "graph_upsert_completed",
        "图谱候选和实体证据已写入",
        stage="graph_upsert",
        concepts=upsert_stats["created_concepts"],
        relations=upsert_stats["relations"],
        graph_llm_success_chunks=upsert_stats["llm_success_chunks"],
        graph_rejected_concepts=upsert_stats["rejected_concepts"],
    )
    concept_count = upsert_stats["created_concepts"]
    relation_count = upsert_stats["relations"]
    llm_success_chunks = upsert_stats["llm_success_chunks"]
    hpo_stats = await _run_auto_hpo_after_graph_upsert(
        db,
        course_id=course_id,
        batch_id=batch_id,
        probe_chunks=selected_llm_chunks,
        payloads=llm_payloads,
        baseline_context_chunks=chunks,
    )
    _safe_emit_log(batch_id, "graph_enrichment_started", "正在刷新图谱拓扑指标", stage="graph_enrichment")
    graph_algorithm_stats = await enrich_course_graph(db, course_id)
    _safe_emit_log(
        batch_id,
        "graph_enrichment_completed",
        "图谱拓扑指标已刷新",
        stage="graph_enrichment",
        **graph_algorithm_stats,
    )
    _safe_emit_log(batch_id, "graph_community_started", "正在生成图谱社区摘要", stage="graph_community")
    community_summary_stats = await rebuild_graph_community_summaries(db, course_id, batch_id=batch_id)
    _safe_emit_log(
        batch_id,
        "graph_community_completed",
        "图谱社区摘要已生成",
        stage="graph_community",
        **community_summary_stats,
    )
    db.commit()
    graph = get_graph_payload(db, course_id, graph_type="semantic")
    return {
        "graph_rebuilt": True,
        "concepts": concept_count,
        "relations": relation_count,
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "graph_extraction_provider": graph_extraction_provider(),
        "graph_extraction_strategy": "adaptive_best_first",
        "graph_extraction_soft_start_budget": plan.budget.get("soft_start_budget"),
        "graph_extraction_selected_chunks": len(llm_chunk_ids),
        "graph_extraction_completed_chunks": extraction_stats.get("completed_chunks", len(llm_payloads)),
        "graph_extraction_pending_chunks": extraction_stats.get("pending_chunks", 0),
        "graph_extraction_coverage": plan.coverage,
        "graph_extraction_stop_reason": plan.stop_reason,
        "graph_extraction_resume_available": bool(extraction_stats.get("resume_available")),
        "graph_extraction_run_id": extraction_run.id,
        "graph_llm_selected_chunks": len(llm_chunk_ids),
        "graph_llm_source_documents": len(llm_document_ids),
        "graph_llm_success_chunks": llm_success_chunks,
        "graph_llm_failed_chunks": len(llm_errors),
        "graph_llm_errors": llm_errors,
        "graph_llm_validation_warnings": llm_validation_warnings,
        "graph_rejected_concepts": upsert_stats["rejected_concepts"],
        "graph_concept_specificity_threshold": upsert_stats.get("graph_concept_specificity_threshold"),
        "graph_concept_specificity_threshold_audit": upsert_stats.get("graph_concept_specificity_threshold_audit"),
        "graph_llm_verified_merges": upsert_stats["llm_verified_merges"],
        "graph_probe_chunks": min(3, len(selected_llm_chunks)),
        "graph_probe_success_chunks": min(3, len([chunk for chunk in selected_llm_chunks if chunk.id in llm_payloads])),
        "graph_probe_failed_chunks": len([chunk for chunk in selected_llm_chunks[:3] if chunk.id in llm_errors]),
        **hpo_stats,
        **graph_algorithm_stats,
        **community_summary_stats,
        "graph_total_active_chunks": len(chunks),
        "graph_source_documents": len(graph_document_ids),
        "quality_profile_version": quality_profile.version,
    }


def _ensure_graph_backup_tables(db: Session) -> None:
    from sqlalchemy import text

    # If the source table gained new columns since the backup table was created,
    # drop the stale backup table so it gets recreated with the correct schema.
    db.execute(text("""
        DO $$
        BEGIN
            IF to_regclass('public.concepts_backup') IS NOT NULL THEN
                IF (SELECT COUNT(*) FROM pg_attribute
                    WHERE attrelid = 'public.concepts'::regclass AND attnum > 0 AND NOT attisdropped)
                   != (SELECT COUNT(*) FROM pg_attribute
                       WHERE attrelid = 'public.concepts_backup'::regclass AND attnum > 0 AND NOT attisdropped) THEN
                    DROP TABLE concepts_backup;
                END IF;
            END IF;

            IF to_regclass('public.concept_relations_backup') IS NOT NULL THEN
                IF (SELECT COUNT(*) FROM pg_attribute
                    WHERE attrelid = 'public.concept_relations'::regclass AND attnum > 0 AND NOT attisdropped)
                   != (SELECT COUNT(*) FROM pg_attribute
                       WHERE attrelid = 'public.concept_relations_backup'::regclass AND attnum > 0 AND NOT attisdropped) THEN
                    DROP TABLE concept_relations_backup;
                END IF;
            END IF;

            IF to_regclass('public.concept_aliases_backup') IS NOT NULL THEN
                IF (SELECT COUNT(*) FROM pg_attribute
                    WHERE attrelid = 'public.concept_aliases'::regclass AND attnum > 0 AND NOT attisdropped)
                   != (SELECT COUNT(*) FROM pg_attribute
                       WHERE attrelid = 'public.concept_aliases_backup'::regclass AND attnum > 0 AND NOT attisdropped) THEN
                    DROP TABLE concept_aliases_backup;
                END IF;
            END IF;
        END $$;
    """))

    db.execute(text(
        "CREATE TABLE IF NOT EXISTS concepts_backup AS SELECT * FROM concepts WHERE 1=0"
    ))
    db.execute(text(
        "CREATE TABLE IF NOT EXISTS concept_relations_backup AS SELECT * FROM concept_relations WHERE 1=0"
    ))
    db.execute(text(
        "CREATE TABLE IF NOT EXISTS concept_aliases_backup AS SELECT * FROM concept_aliases WHERE 1=0"
    ))


def _backup_course_graph_tables(db: Session, course_id: str) -> None:
    """Create backup copies of graph tables for atomic rollback on failure."""
    from sqlalchemy import text

    _ensure_graph_backup_tables(db)
    db.execute(text("DELETE FROM concepts_backup WHERE course_id = :course_id"), {"course_id": course_id})
    db.execute(text("DELETE FROM concept_relations_backup WHERE course_id = :course_id"), {"course_id": course_id})
    db.execute(text("DELETE FROM concept_aliases_backup WHERE concept_id IN (SELECT id FROM concepts WHERE course_id = :course_id)"), {"course_id": course_id})
    db.execute(text(
        "INSERT INTO concepts_backup SELECT * FROM concepts WHERE course_id = :course_id"
    ), {"course_id": course_id})
    db.execute(text(
        "INSERT INTO concept_relations_backup SELECT * FROM concept_relations WHERE course_id = :course_id"
    ), {"course_id": course_id})
    db.execute(text(
        "INSERT INTO concept_aliases_backup SELECT ca.* FROM concept_aliases ca JOIN concepts c ON ca.concept_id = c.id WHERE c.course_id = :course_id"
    ), {"course_id": course_id})


def _restore_course_graph_from_backup(db: Session, course_id: str) -> None:
    """Restore graph tables from backup copies."""
    from sqlalchemy import text

    db.execute(text("DELETE FROM concept_relations WHERE course_id = :course_id"), {"course_id": course_id})
    db.execute(text("DELETE FROM concept_aliases WHERE concept_id IN (SELECT id FROM concepts WHERE course_id = :course_id)"), {"course_id": course_id})
    db.execute(text("DELETE FROM concepts WHERE course_id = :course_id"), {"course_id": course_id})
    db.execute(text(
        "INSERT INTO concepts SELECT * FROM concepts_backup WHERE course_id = :course_id"
    ), {"course_id": course_id})
    db.execute(text(
        "INSERT INTO concept_relations SELECT * FROM concept_relations_backup WHERE course_id = :course_id"
    ), {"course_id": course_id})
    db.execute(text(
        "INSERT INTO concept_aliases SELECT ca.* FROM concept_aliases_backup ca JOIN concepts_backup c ON ca.concept_id = c.id WHERE c.course_id = :course_id"
    ), {"course_id": course_id})


async def incremental_update_course_graph(
    db: Session,
    course_id: str,
    changed_document_ids: list[str],
    batch_id: str | None = None,
) -> dict:
    course = db.get(Course, course_id)
    if not changed_document_ids:
        return {"graph_rebuilt": False, "reason": "no_changed_documents"}

    sync_graph_chapter_labels(db, course_id)

    # 1. Re-extract from changed documents' active chunks before pruning old graph
    # evidence, so the visible graph does not spend model latency in a half-deleted
    # state.
    active_documents = db.scalars(
        select(Document).where(Document.id.in_(changed_document_ids), Document.is_active.is_(True))
    ).all()
    graph_documents = filter_graph_documents(course, active_documents)
    graph_document_ids = {document.id for document in graph_documents}
    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.course_id == course_id, Chunk.is_active.is_(True), Chunk.document_id.in_(graph_document_ids))
        .order_by(Chunk.created_at.asc())
    ).all()

    if not chunks:
        from app.services.maintenance import delete_document_graph_incremental

        graph_gc_stats: dict[str, int] = {}
        for document_id in changed_document_ids:
            stats = delete_document_graph_incremental(db, course_id, document_id)
            graph_gc_stats = {key: graph_gc_stats.get(key, 0) + value for key, value in stats.items()}
        from app.services.graph_algorithms import enrich_course_graph
        graph_algorithm_stats = await enrich_course_graph(db, course_id, run_relation_completion=False, run_dijkstra=False)
        db.commit()
        return {
            "graph_rebuilt": True,
            "mode": "incremental",
            "concepts": 0,
            "relations": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "graph_extraction_provider": graph_extraction_provider(),
            "reason": "no_active_chunks_for_changed_documents",
            **graph_gc_stats,
            **graph_algorithm_stats,
        }

    from app.services.quality.profiles import rebuild_domain_quality_profile

    quality_profile = rebuild_domain_quality_profile(db, course_id)
    plan = plan_adaptive_graph_extraction_chunks(chunks)
    extraction_run = create_graph_extraction_run_from_plan(
        db,
        course_id=course_id,
        batch_id=batch_id,
        chunks=chunks,
        plan=plan,
        profile_version=quality_profile.version,
    )
    db.commit()
    llm_chunk_ids = set(plan.selected_chunk_ids)
    selected_llm_chunks = [chunk for chunk in chunks if chunk.id in llm_chunk_ids]

    if batch_id:
        _safe_emit_log(
            batch_id,
            "batch_graph_plan_created",
            f"增量自适应图谱抽取计划已创建：选择 {len(selected_llm_chunks)} / {len(chunks)} 个片段",
            graph_extraction_run_id=extraction_run.id,
            graph_extraction_strategy="adaptive_best_first",
            graph_extraction_soft_start_budget=plan.budget.get("soft_start_budget"),
            changed_documents=len(changed_document_ids),
            selected_llm_chunks=len(selected_llm_chunks),
            graph_extraction_coverage=plan.coverage,
            graph_extraction_stop_reason=plan.stop_reason,
        )

    llm_payloads, llm_errors, extraction_stats = await execute_graph_extraction_run(
        db,
        run=extraction_run,
        chunks=chunks,
        batch_id=batch_id,
    )

    from app.services.maintenance import delete_document_graph_incremental

    graph_gc_stats: dict[str, int] = {}
    for document_id in changed_document_ids:
        stats = delete_document_graph_incremental(db, course_id, document_id)
        graph_gc_stats = {key: graph_gc_stats.get(key, 0) + value for key, value in stats.items()}

    _safe_emit_log(
        batch_id,
        "graph_upsert_started",
        "正在写入本轮最小更新图谱证据",
        stage="graph_upsert",
        changed_documents=len(changed_document_ids),
        graph_llm_success_chunks=len(llm_payloads),
    )
    upsert_stats = await upsert_graph_candidates_from_chunks(db, course_id, chunks, llm_payloads=llm_payloads, run_llm_merge=False)
    _safe_emit_log(
        batch_id,
        "graph_upsert_completed",
        "最小更新图谱证据已写入",
        stage="graph_upsert",
        concepts=upsert_stats["created_concepts"],
        relations=upsert_stats["relations"],
        graph_llm_success_chunks=upsert_stats["llm_success_chunks"],
        graph_rejected_concepts=upsert_stats["rejected_concepts"],
    )
    concept_count = upsert_stats["created_concepts"]
    relation_count = upsert_stats["relations"]
    llm_success_chunks = upsert_stats["llm_success_chunks"]
    hpo_stats = await _run_auto_hpo_after_graph_upsert(
        db,
        course_id=course_id,
        batch_id=batch_id,
        probe_chunks=selected_llm_chunks,
        payloads=llm_payloads,
        baseline_context_chunks=chunks,
    )

    # 3. Enrich: run full algorithms but skip expensive LLM completion and Dijkstra for speed
    from app.services.graph_algorithms import enrich_course_graph
    _safe_emit_log(batch_id, "graph_enrichment_started", "正在刷新最小更新后的图谱拓扑指标", stage="graph_enrichment")
    graph_algorithm_stats = await enrich_course_graph(db, course_id, run_relation_completion=False, run_dijkstra=False)
    _safe_emit_log(
        batch_id,
        "graph_enrichment_completed",
        "最小更新图谱拓扑指标已刷新",
        stage="graph_enrichment",
        **graph_algorithm_stats,
    )
    _safe_emit_log(batch_id, "graph_community_started", "正在生成图谱社区摘要", stage="graph_community")
    community_summary_stats = await rebuild_graph_community_summaries(db, course_id, batch_id=batch_id)
    _safe_emit_log(
        batch_id,
        "graph_community_completed",
        "图谱社区摘要已生成",
        stage="graph_community",
        **community_summary_stats,
    )
    db.commit()
    graph = get_graph_payload(db, course_id, graph_type="semantic")
    return {
        "graph_rebuilt": True,
        "mode": "incremental",
        "concepts": concept_count,
        "relations": relation_count,
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "graph_extraction_provider": graph_extraction_provider(),
        "graph_extraction_strategy": "adaptive_best_first",
        "graph_extraction_soft_start_budget": plan.budget.get("soft_start_budget"),
        "graph_extraction_selected_chunks": len(llm_chunk_ids),
        "graph_extraction_completed_chunks": extraction_stats.get("completed_chunks", len(llm_payloads)),
        "graph_extraction_pending_chunks": extraction_stats.get("pending_chunks", 0),
        "graph_extraction_coverage": plan.coverage,
        "graph_extraction_stop_reason": plan.stop_reason,
        "graph_extraction_resume_available": bool(extraction_stats.get("resume_available")),
        "graph_extraction_run_id": extraction_run.id,
        "graph_llm_success_chunks": llm_success_chunks,
        "graph_llm_failed_chunks": len(llm_errors),
        "graph_rejected_concepts": upsert_stats["rejected_concepts"],
        **graph_gc_stats,
        **hpo_stats,
        "graph_concept_specificity_threshold": upsert_stats.get("graph_concept_specificity_threshold"),
        "graph_concept_specificity_threshold_audit": upsert_stats.get("graph_concept_specificity_threshold_audit"),
        "graph_llm_verified_merges": upsert_stats["llm_verified_merges"],
        "graph_total_active_chunks": len(chunks),
        "quality_profile_version": quality_profile.version,
        **graph_algorithm_stats,
        **community_summary_stats,
    }


def document_chapter_label(document: Document, course_name: str | None = None) -> str:
    path_label = derive_chapter(Path(document.source_path), course_name=course_name)
    if not is_invalid_chapter_label(path_label, course_name=course_name):
        return path_label
    for tag in document.tags or []:
        if not is_invalid_chapter_label(tag, course_name=course_name):
            return tag
    return (document.title[:80] if document.title else document.source_type) or document.source_type


def _is_under_path(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def filter_graph_documents(course: Course | None, documents: list[Document]) -> list[Document]:
    if course is None:
        return documents
    storage_root = get_settings().course_paths_for_name(course.name)["storage_root"]
    storage_documents = [document for document in documents if _is_under_path(Path(document.source_path), storage_root)]
    return storage_documents or documents


def normalize_chapter_ref(value: str, course_name: str | None = None) -> str | None:
    label = canonical_chapter_label(value, course_name=course_name)
    if label and not is_invalid_chapter_label(label, course_name=course_name):
        return label
    if is_invalid_chapter_label(value, course_name=course_name):
        return None
    return value


def sync_graph_chapter_labels(db: Session, course_id: str) -> dict:
    course = db.get(Course, course_id)
    documents = db.scalars(select(Document).where(Document.course_id == course_id, Document.is_active.is_(True))).all()
    graph_documents = filter_graph_documents(course, documents)
    document_chapters = {document.id: document_chapter_label(document, course.name if course else None) for document in graph_documents}
    updated_documents = 0
    updated_chunks = 0
    updated_concepts = 0
    for document in graph_documents:
        chapter = document_chapters[document.id]
        if document.tags != [chapter]:
            document.tags = [chapter]
            updated_documents += 1
    chunks = db.scalars(
        select(Chunk).where(Chunk.course_id == course_id, Chunk.is_active.is_(True), Chunk.document_id.in_(document_chapters))
    ).all()
    for chunk in chunks:
        chapter = document_chapters.get(chunk.document_id)
        if chapter and chunk.chapter != chapter:
            chunk.chapter = chapter
            updated_chunks += 1
    relation_refs: dict[str, set[str]] = defaultdict(set)
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    evidence_chunk_ids = {relation.evidence_chunk_id for relation in relations if relation.evidence_chunk_id}
    evidence_chunks = db.scalars(select(Chunk).where(Chunk.id.in_(evidence_chunk_ids))).all() if evidence_chunk_ids else []
    evidence_documents = {
        document.id: document
        for document in db.scalars(
            select(Document).where(Document.id.in_({chunk.document_id for chunk in evidence_chunks}))
        ).all()
    }
    chunk_chapters = {
        chunk.id: document_chapter_label(evidence_documents[chunk.document_id], course.name if course else None)
        for chunk in evidence_chunks
        if chunk.document_id in evidence_documents
    }
    for relation in relations:
        chapter = chunk_chapters.get(relation.evidence_chunk_id or "")
        if not chapter:
            continue
        relation_refs[relation.source_concept_id].add(chapter)
        if relation.target_concept_id:
            relation_refs[relation.target_concept_id].add(chapter)
    concepts = db.scalars(select(Concept).where(Concept.course_id == course_id)).all()
    for concept in concepts:
        normalized_refs = sorted(
            {
                normalized
                for ref in [*(concept.chapter_refs or []), *relation_refs.get(concept.id, set())]
                for normalized in [normalize_chapter_ref(ref, course.name if course else None)]
                if normalized
            }
        )
        if normalized_refs != (concept.chapter_refs or []):
            concept.chapter_refs = normalized_refs
            updated_concepts += 1
    db.commit()
    return {
        "updated_documents": updated_documents,
        "updated_chunks": updated_chunks,
        "updated_concepts": updated_concepts,
    }


def is_default_visible_relation(relation: ConceptRelation) -> bool:
    relation_type = normalize_relation_type(relation.relation_type)
    if not relation_type:
        return False
    metadata = getattr(relation, "metadata_json", None) or {}
    if metadata.get("candidate_only"):
        return False
    if relation_type == "related_to":
        return bool(
            not getattr(relation, "is_inferred", False)
            and float(getattr(relation, "weight", 0.0) or 0.0) >= 0.75
            and bool(getattr(relation, "is_validated", False))
        )
    return True


def get_concept_cards(db: Session, course_id: str) -> list[dict]:
    concepts = db.scalars(select(Concept).where(Concept.course_id == course_id).order_by(Concept.importance_score.desc(), Concept.canonical_name)).all()
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    relation_index: dict[str, list[dict]] = defaultdict(list)
    for relation in relations:
        if not is_default_visible_relation(relation):
            continue
        relation_index[relation.source_concept_id].append(
            {
                "concept_id": relation.target_concept_id or relation.id,
                "relation_type": relation.relation_type,
                "target_name": relation.target_name,
                "confidence": relation.confidence,
                "weight": getattr(relation, "weight", None),
                "relation_source": getattr(relation, "relation_source", None),
                "is_inferred": bool(getattr(relation, "is_inferred", False)),
            }
        )
    return [
        {
            "concept_id": concept.id,
            "name": concept.canonical_name,
            "aliases": sorted({alias.alias for alias in concept.aliases}),
            "summary": concept.summary or "",
            "chapter_refs": concept.chapter_refs,
            "concept_type": concept.concept_type,
            "importance_score": concept.importance_score,
            "related_concepts": relation_index.get(concept.id, []),
        }
        for concept in concepts
    ]


def build_citation(db: Session, chunk_id: str | None) -> dict | None:
    if not chunk_id:
        return None
    chunk = db.get(Chunk, chunk_id)
    if chunk is None:
        return None
    document = db.get(Document, chunk.document_id)
    if document is None:
        return None
    return Citation(
        chunk_id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        source_path=document.source_path,
        chapter=chunk.chapter,
        section=chunk.section,
        page_number=chunk.page_number,
        snippet=chunk.snippet,
    ).model_dump()


def get_graph_node_detail(db: Session, course_id: str, concept_id: str) -> dict | None:
    concept = db.scalar(select(Concept).where(Concept.course_id == course_id, Concept.id == concept_id))
    if concept is None:
        return None
    relations = db.scalars(
        select(ConceptRelation).where(ConceptRelation.course_id == course_id, ConceptRelation.source_concept_id == concept_id)
    ).all()
    return {
        "concept_id": concept.id,
        "name": concept.canonical_name,
        "normalized_name": concept.normalized_name,
        "summary": concept.summary,
        "aliases": sorted({alias.alias for alias in concept.aliases}),
        "chapter_refs": concept.chapter_refs,
        "concept_type": concept.concept_type,
        "importance_score": concept.importance_score,
        "evidence_count": getattr(concept, "evidence_count", 0),
        "community_louvain": getattr(concept, "community_louvain", None),
        "community_spectral": getattr(concept, "community_spectral", None),
        "component_id": getattr(concept, "component_id", None),
        "centrality": getattr(concept, "centrality_json", {}) or {},
        "graph_rank_score": getattr(concept, "graph_rank_score", 0.0),
        "relations": [
            {
                "relation_id": relation.id,
                "relation_type": relation.relation_type,
                "target_concept_id": relation.target_concept_id,
                "target_name": relation.target_name,
                "confidence": relation.confidence,
                "weight": getattr(relation, "weight", None),
                "semantic_similarity": getattr(relation, "semantic_similarity", None),
                "support_count": getattr(relation, "support_count", None),
                "relation_source": getattr(relation, "relation_source", None),
                "is_inferred": bool(getattr(relation, "is_inferred", False)),
                "evidence": build_citation(db, relation.evidence_chunk_id),
            }
            for relation in relations
            if is_default_visible_relation(relation)
        ],
    }


def cleanup_graph(db: Session, course_id: str) -> dict:
    concepts = db.scalars(select(Concept).where(Concept.course_id == course_id)).all()
    invalid_ids = {concept.id for concept in concepts if not is_valid_concept(concept.canonical_name)}
    removed_concepts = 0
    removed_relations = 0
    migrated_relations = 0
    invalid_relation_ids: list[str] = []
    for relation in db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all():
        normalized_type = normalize_relation_type(relation.relation_type)
        if not normalized_type:
            invalid_relation_ids.append(relation.id)
            continue
        if normalized_type != relation.relation_type:
            relation.relation_type = normalized_type
            metadata = dict(getattr(relation, "metadata_json", None) or {})
            metadata["legacy_relation_type_migrated"] = True
            relation.metadata_json = metadata
            migrated_relations += 1
    if invalid_relation_ids:
        removed_relations += db.query(ConceptRelation).filter(ConceptRelation.id.in_(invalid_relation_ids)).delete(synchronize_session=False)
    if invalid_ids:
        removed_relations += db.query(ConceptRelation).filter(
            ConceptRelation.course_id == course_id,
            (ConceptRelation.source_concept_id.in_(invalid_ids) | ConceptRelation.target_concept_id.in_(invalid_ids)),
        ).delete(synchronize_session=False)
        removed_concepts += db.query(ConceptAlias).filter(ConceptAlias.concept_id.in_(invalid_ids)).delete(synchronize_session=False)
        removed_concepts += db.query(Concept).filter(Concept.id.in_(invalid_ids)).delete(synchronize_session=False)
        db.commit()
    elif migrated_relations or invalid_relation_ids:
        db.commit()
    return {"removed_concepts": removed_concepts, "removed_relations": removed_relations, "migrated_relations": migrated_relations}


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[str(item.get(key) or "unknown")] += 1
    return dict(counts)


def _graph_response(graph_type: str, nodes: list[dict], edges: list[dict], chapter: str | None = None) -> dict:
    return {
        "graph_type": graph_type,
        "schema_version": "typed_graph_v1",
        "nodes": nodes,
        "edges": edges,
        "node_counts": _count_by(nodes, "category"),
        "edge_counts": _count_by(edges, "category"),
        "focus_chapter": chapter,
    }


def _load_graph_documents(db: Session, course_id: str, chapter: str | None = None) -> tuple[Course | None, list[Document]]:
    course = db.get(Course, course_id)
    documents = db.scalars(
        select(Document).where(Document.course_id == course_id, Document.is_active.is_(True)).order_by(Document.title)
    ).all()
    documents = filter_graph_documents(course, documents)
    if chapter:
        documents = [document for document in documents if chapter == document_chapter_label(document, course.name if course else None)]
    return course, documents


def get_graph_payload(db: Session, course_id: str, chapter: str | None = None, graph_type: str = "semantic") -> dict:
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"invalid graph_type {graph_type!r}")
    if graph_type == "structural":
        return get_structural_graph_payload(db, course_id)
    if graph_type == "evidence":
        return get_evidence_graph_payload(db, course_id, chapter=chapter)
    return get_semantic_graph_payload(db, course_id, chapter=chapter)


def get_structural_graph_payload(db: Session, course_id: str) -> dict:
    course, documents = _load_graph_documents(db, course_id)
    course_label = course.name if course is not None else "Course Workspace"
    nodes: list[dict] = [{"id": f"course:{course_id}", "name": course_label, "category": "course", "value": 1}]
    edges: list[dict] = []
    for document in documents:
        document_node_id = f"document:{document.id}"
        nodes.append(
            {
                "id": document_node_id,
                "name": document.title,
                "category": "document",
                "value": 2,
                "document_id": document.id,
                "source_type": document.source_type,
            }
        )
        edges.append({"source": f"course:{course_id}", "target": document_node_id, "label": "contains", "category": "structure"})
        chunks = db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document.id, Chunk.is_active.is_(True))
            .order_by(Chunk.page_number.asc(), Chunk.created_at.asc())
        ).all()
        chapter_names = sorted({chunk.chapter or document_chapter_label(document, course.name if course else None) or "General" for chunk in chunks})
        for chapter_name in chapter_names:
            chapter_id = f"chapter:{document.id}:{chapter_name}"
            nodes.append({"id": chapter_id, "name": chapter_name, "category": "chapter", "value": 2, "chapter": chapter_name, "document_id": document.id})
            edges.append({"source": document_node_id, "target": chapter_id, "label": "contains", "category": "structure"})
            section_names = sorted({chunk.section or "General" for chunk in chunks if (chunk.chapter or document_chapter_label(document, course.name if course else None) or "General") == chapter_name})
            for section_name in section_names:
                section_id = f"section:{document.id}:{chapter_name}:{section_name}"
                nodes.append({"id": section_id, "name": section_name, "category": "section", "value": 1.5, "chapter": chapter_name, "document_id": document.id})
                edges.append({"source": chapter_id, "target": section_id, "label": "contains", "category": "structure"})
                # Structural graph stops at section level; chunks are content units, not structure nodes
    return _graph_response("structural", nodes, edges)


def get_semantic_graph_payload(db: Session, course_id: str, chapter: str | None = None) -> dict:
    _course, _documents = _load_graph_documents(db, course_id, chapter=chapter)
    concepts = db.scalars(select(Concept).where(Concept.course_id == course_id)).all()
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    nodes: list[dict] = []
    edges: list[dict] = []

    filtered_concepts = [concept for concept in concepts if not chapter or chapter in concept.chapter_refs]
    if not chapter:
        filtered_concepts = concepts
    relation_counts: dict[str, int] = defaultdict(int)
    for relation in relations:
        if not is_default_visible_relation(relation):
            continue
        relation_counts[relation.source_concept_id] += 1
        if relation.target_concept_id:
            relation_counts[relation.target_concept_id] += 1
    filtered_concepts = [
        concept
        for concept in filtered_concepts
        if concept.importance_score >= 0.55
        or relation_counts.get(concept.id, 0) >= 1
        or len(concept.chapter_refs) >= 1
        or float(getattr(concept, "graph_rank_score", 0.0) or 0.0) > 0
    ]
    filtered_concepts = sorted(
        filtered_concepts,
        key=lambda concept: (
            float(getattr(concept, "graph_rank_score", 0.0) or 0.0),
            float((getattr(concept, "centrality_json", {}) or {}).get("centrality_score", 0.0)),
            concept.importance_score,
            relation_counts.get(concept.id, 0),
        ),
        reverse=True,
    )[:360]
    concept_ids = {concept.id for concept in filtered_concepts}
    for concept in filtered_concepts:
        nodes.append(
            {
                "id": concept.id,
                "concept_id": concept.id,
                "name": concept.canonical_name,
                "category": "semantic_entity",
                "entity_type": normalize_entity_type(concept.concept_type),
                "aliases": sorted({alias.alias for alias in concept.aliases}),
                "support_count": getattr(concept, "evidence_count", 0),
                "confidence": concept.importance_score,
                "canonical_key": concept.normalized_name,
                "summary": concept.summary,
                "value": max(2.0, concept.importance_score * 12),
                "chapter": concept.chapter_refs[0] if concept.chapter_refs else None,
                "importance_score": concept.importance_score,
                "evidence_count": getattr(concept, "evidence_count", 0),
                "community_louvain": getattr(concept, "community_louvain", None),
                "community_spectral": getattr(concept, "community_spectral", None),
                "component_id": getattr(concept, "component_id", None),
                "centrality_score": float((getattr(concept, "centrality_json", {}) or {}).get("centrality_score", 0.0)),
                "graph_rank_score": getattr(concept, "graph_rank_score", 0.0),
            }
        )

    for relation in relations:
        if not is_default_visible_relation(relation):
            continue
        if relation.target_concept_id and relation.source_concept_id in concept_ids and relation.target_concept_id in concept_ids:
            edges.append(
                {
                    "source": relation.source_concept_id,
                    "target": relation.target_concept_id,
                    "label": relation.relation_type,
                    "confidence": relation.confidence,
                    "category": "semantic",
                    "evidence_chunk_id": relation.evidence_chunk_id,
                    "weight": getattr(relation, "weight", None),
                    "semantic_similarity": getattr(relation, "semantic_similarity", None),
                    "support_count": getattr(relation, "support_count", None),
                    "relation_source": getattr(relation, "relation_source", None),
                    "is_inferred": bool(getattr(relation, "is_inferred", False)),
                }
            )
    return _graph_response("semantic", nodes, edges, chapter=chapter)


def get_evidence_graph_payload(db: Session, course_id: str, chapter: str | None = None) -> dict:
    concepts = db.scalars(select(Concept).where(Concept.course_id == course_id)).all()
    if chapter:
        concepts = [concept for concept in concepts if chapter in (concept.chapter_refs or [])]
    concept_by_id = {concept.id: concept for concept in concepts}
    concept_ids = set(concept_by_id)
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    evidence_chunk_ids = {
        relation.evidence_chunk_id
        for relation in relations
        if relation.evidence_chunk_id and (relation.source_concept_id in concept_ids or relation.target_concept_id in concept_ids)
    }
    chunks = db.scalars(select(Chunk).where(Chunk.id.in_(list(evidence_chunk_ids)))).all() if evidence_chunk_ids else []
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    version_ids = {chunk.document_version_id for chunk in chunks if chunk.document_version_id}
    versions = db.scalars(select(DocumentVersion).where(DocumentVersion.id.in_(list(version_ids)))).all() if version_ids else []
    documents = {document.id: document for document in db.scalars(select(Document).where(Document.course_id == course_id)).all()}
    nodes: list[dict] = []
    edges: list[dict] = []
    for concept in concepts:
        nodes.append(
            {
                "id": f"semantic:{concept.id}",
                "concept_id": concept.id,
                "name": concept.canonical_name,
                "category": "semantic_entity",
                "entity_type": normalize_entity_type(concept.concept_type),
                "aliases": sorted({alias.alias for alias in concept.aliases}),
                "support_count": getattr(concept, "evidence_count", 0),
                "confidence": concept.importance_score,
                "canonical_key": concept.normalized_name,
                "summary": concept.summary,
                "chapter": concept.chapter_refs[0] if concept.chapter_refs else None,
                "value": max(2.0, concept.importance_score * 10),
            }
        )
    for chunk in chunks:
        document = documents.get(chunk.document_id)
        chunk_id = f"evidence_chunk:{chunk.id}"
        nodes.append(
            {
                "id": chunk_id,
                "name": chunk.snippet[:80] if chunk.snippet else chunk.id,
                "category": "evidence_chunk",
                "value": 1.5,
                "chapter": chunk.chapter,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "snippet": chunk.snippet,
                "page_number": chunk.page_number,
                "source_type": chunk.source_type,
            }
        )
        if chunk.document_version_id:
            version_id = f"document_version:{chunk.document_version_id}"
            if not any(node["id"] == version_id for node in nodes):
                version = next((item for item in versions if item.id == chunk.document_version_id), None)
                nodes.append(
                    {
                        "id": version_id,
                        "name": document.title if document else (version.storage_path if version else chunk.document_version_id),
                        "category": "document_version",
                        "value": 2,
                        "document_id": chunk.document_id,
                        "document_version_id": chunk.document_version_id,
                        "source_type": document.source_type if document else None,
                    }
                )
            edges.append({"source": chunk_id, "target": version_id, "label": "from_version", "category": "evidence"})
    seen_edges: set[tuple[str, str, str]] = set()
    for relation in relations:
        if not relation.evidence_chunk_id or relation.evidence_chunk_id not in chunks_by_id:
            continue
        for concept_id in (relation.source_concept_id, relation.target_concept_id):
            if not concept_id or concept_id not in concept_ids:
                continue
            edge = (f"semantic:{concept_id}", f"evidence_chunk:{relation.evidence_chunk_id}", "evidenced_by")
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append({"source": edge[0], "target": edge[1], "label": edge[2], "category": "evidence", "evidence_chunk_id": relation.evidence_chunk_id})
    return _graph_response("evidence", nodes, edges, chapter=chapter)


def get_query_semantic_graph_payload(db: Session, course_id: str, chunk_ids: list[str], query: str | None = None) -> dict:
    requested_chunk_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids if chunk_id))
    if not requested_chunk_ids:
        return _graph_response("evidence", [], [])

    chunks = db.scalars(
        select(Chunk).where(
            Chunk.course_id == course_id,
            Chunk.id.in_(requested_chunk_ids),
            Chunk.is_active.is_(True),
        )
    ).all()
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    if not chunks_by_id:
        return _graph_response("evidence", [], [])

    relations = db.scalars(
        select(ConceptRelation).where(
            ConceptRelation.course_id == course_id,
            ConceptRelation.evidence_chunk_id.in_(list(chunks_by_id)),
        )
    ).all()
    query_key = normalize_concept_name(query) if query else ""
    if query_key:
        seed_concept_ids = {
            concept.id
            for concept in db.scalars(
                select(Concept).where(
                    Concept.course_id == course_id,
                    or_(
                        Concept.normalized_name.contains(query_key),
                        Concept.canonical_name.ilike(f"%{query}%"),
                    ),
                )
            ).all()
        }
        seed_concept_ids.update(
            alias.concept_id
            for alias in db.scalars(
                select(ConceptAlias)
                .join(Concept, ConceptAlias.concept_id == Concept.id)
                .where(
                    Concept.course_id == course_id,
                    ConceptAlias.normalized_alias.contains(query_key),
                )
            ).all()
        )
        if seed_concept_ids:
            query_relations = db.scalars(
                select(ConceptRelation).where(
                    ConceptRelation.course_id == course_id,
                    or_(
                        ConceptRelation.source_concept_id.in_(seed_concept_ids),
                        ConceptRelation.target_concept_id.in_(seed_concept_ids),
                    ),
                )
            ).all()
            seen_relation_ids = {relation.id for relation in relations}
            relations.extend(relation for relation in query_relations if relation.id not in seen_relation_ids)
    relations = [relation for relation in relations if not (relation.metadata_json or {}).get("candidate_only")]
    if not relations:
        return _graph_response("evidence", [], [], chapter=chunks[0].chapter if len({chunk.chapter for chunk in chunks}) == 1 else None)

    missing_relation_chunk_ids = {
        relation.evidence_chunk_id
        for relation in relations
        if relation.evidence_chunk_id and relation.evidence_chunk_id not in chunks_by_id
    }
    if missing_relation_chunk_ids:
        relation_chunks = db.scalars(
            select(Chunk).where(
                Chunk.course_id == course_id,
                Chunk.id.in_(list(missing_relation_chunk_ids)),
                Chunk.is_active.is_(True),
            )
        ).all()
        chunks.extend(relation_chunks)
        chunks_by_id.update({chunk.id: chunk for chunk in relation_chunks})

    concept_ids = {
        concept_id
        for relation in relations
        for concept_id in (relation.source_concept_id, relation.target_concept_id)
        if concept_id
    }
    concepts = {
        concept.id: concept
        for concept in db.scalars(
            select(Concept).where(
                Concept.course_id == course_id,
                Concept.id.in_(list(concept_ids)),
            )
        ).all()
    }
    documents = {
        document.id: document
        for document in db.scalars(
            select(Document).where(
                Document.course_id == course_id,
                Document.id.in_({chunk.document_id for chunk in chunks if chunk.document_id}),
            )
        ).all()
    }

    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str, str | None]] = set()

    def add_node(node: dict) -> None:
        if node["id"] in node_ids:
            return
        node_ids.add(node["id"])
        nodes.append(node)

    for concept in concepts.values():
        add_node(
            {
                "id": f"semantic:{concept.id}",
                "concept_id": concept.id,
                "name": concept.canonical_name,
                "category": "semantic_entity",
                "entity_type": normalize_entity_type(concept.concept_type),
                "aliases": sorted({alias.alias for alias in concept.aliases}),
                "support_count": getattr(concept, "evidence_count", 0),
                "confidence": concept.importance_score,
                "canonical_key": concept.normalized_name,
                "summary": concept.summary,
                "chapter": concept.chapter_refs[0] if concept.chapter_refs else None,
                "evidence_count": concept.evidence_count,
                "community_louvain": concept.community_louvain,
                "community_spectral": concept.community_spectral,
                "component_id": concept.component_id,
                "centrality_score": (concept.centrality_json or {}).get("degree"),
                "graph_rank_score": concept.graph_rank_score,
                "value": max(2.0, concept.importance_score * 10, concept.evidence_count),
            }
        )

    for chunk in chunks:
        document = documents.get(chunk.document_id)
        chunk_node_id = f"evidence_chunk:{chunk.id}"
        add_node(
            {
                "id": chunk_node_id,
                "name": chunk.snippet[:80] if chunk.snippet else chunk.id,
                "category": "evidence_chunk",
                "value": 1.5,
                "chapter": chunk.chapter,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "snippet": chunk.snippet,
                "page_number": chunk.page_number,
                "source_type": chunk.source_type,
            }
        )
        if chunk.document_version_id:
            version_node_id = f"document_version:{chunk.document_version_id}"
            add_node(
                {
                    "id": version_node_id,
                    "name": document.title if document else chunk.document_version_id,
                    "category": "document_version",
                    "value": 2,
                    "document_id": chunk.document_id,
                    "document_version_id": chunk.document_version_id,
                    "source_type": document.source_type if document else None,
                }
            )
            edge_key = (chunk_node_id, version_node_id, "from_version", None)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({"source": chunk_node_id, "target": version_node_id, "label": "from_version", "category": "evidence"})

    for relation in relations:
        source_node_id = f"semantic:{relation.source_concept_id}"
        target_node_id = f"semantic:{relation.target_concept_id}" if relation.target_concept_id else None
        chunk_node_id = f"evidence_chunk:{relation.evidence_chunk_id}"
        if source_node_id not in node_ids or chunk_node_id not in node_ids:
            continue
        evidence_edge_key = (source_node_id, chunk_node_id, "evidenced_by", relation.evidence_chunk_id)
        if evidence_edge_key not in seen_edges:
            seen_edges.add(evidence_edge_key)
            edges.append(
                {
                    "source": source_node_id,
                    "target": chunk_node_id,
                    "label": "evidenced_by",
                    "category": "evidence",
                    "evidence_chunk_id": relation.evidence_chunk_id,
                    "confidence": relation.confidence,
                    "weight": relation.weight,
                    "relation_source": relation.relation_source,
                    "is_inferred": relation.is_inferred,
                }
            )
        if not target_node_id or target_node_id not in node_ids:
            continue
        semantic_edge_key = (source_node_id, target_node_id, relation.relation_type, relation.evidence_chunk_id)
        if semantic_edge_key in seen_edges:
            continue
        seen_edges.add(semantic_edge_key)
        edges.append(
            {
                "source": source_node_id,
                "target": target_node_id,
                "label": relation.relation_type,
                "category": "semantic",
                "evidence_chunk_id": relation.evidence_chunk_id,
                "confidence": relation.confidence,
                "weight": relation.weight,
                "semantic_similarity": relation.semantic_similarity,
                "support_count": relation.support_count,
                "relation_source": relation.relation_source,
                "is_inferred": relation.is_inferred,
            }
        )

    chapters = {chunk.chapter for chunk in chunks if chunk.chapter}
    return _graph_response("evidence", nodes, edges, chapter=next(iter(chapters)) if len(chapters) == 1 else None)
