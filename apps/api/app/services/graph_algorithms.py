from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.embeddings import ChatProvider
from app.services.quality.policies import RelationQualityPolicy
from app.services.quality.signals import build_quality_signals
from app.services.vector_store import VectorStore, cosine_similarity


MIN_GRAPH_CONCEPTS = 160
MAX_GRAPH_CONCEPTS = 360
MAX_ADAPTIVE_GRAPH_CONCEPTS = 800
SEMANTIC_EDGE_THRESHOLD = 0.62
DIJKSTRA_SEMANTIC_THRESHOLD = 0.78
MIN_RELATION_CONFIDENCE = 0.72
ADAPTIVE_RELATION_CONFIDENCE_MIN_SAMPLE = 20
ADAPTIVE_RELATION_CONFIDENCE_FLOOR = 0.62
ADAPTIVE_RELATION_CONFIDENCE_CEILING = 0.78
MIN_ACCEPTED_RELATION_WEIGHT = 0.62
MIN_RELATED_TO_DISPLAY_WEIGHT = 0.75
MIN_WEAK_RELATION_SEMANTIC = 0.55
MIN_COOCCURRENCE_STRENGTH = 0.30
ALLOWED_COMPLETION_RELATIONS = {
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
LEGACY_RELATION_TYPE_MAP = {
    "defines": "defined_by",
    "relates_to": "related_to",
    "mentions": "related_to",
    "compares": "compares_with",
    "extends": "derives_from",
}


@dataclass(frozen=True)
class ConceptSignal:
    concept_id: str
    importance: float
    evidence_count: int
    chapter_refs: tuple[str, ...]
    vector: tuple[float, ...] | None


@dataclass(frozen=True)
class RelationSignal:
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    support_count: int = 1
    evidence_chunk_id: str | None = None
    relation_source: str = "llm"
    is_inferred: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class WeightedEdge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    semantic_similarity: float
    support_count: int
    confidence: float
    evidence_chunk_id: str | None
    relation_source: str
    is_inferred: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphQualityThresholds:
    relation_confidence: float
    min_accepted_relation_weight: float
    dijkstra_semantic_threshold: float
    concept_keep_min: int
    concept_keep_max: int
    audit: dict[str, Any]


@dataclass(frozen=True)
class GraphHyperparameters:
    min_relation_confidence: float = MIN_RELATION_CONFIDENCE
    min_accepted_relation_weight: float = MIN_ACCEPTED_RELATION_WEIGHT
    dijkstra_semantic_threshold: float = DIJKSTRA_SEMANTIC_THRESHOLD
    w_degree: float = 0.25
    w_weighted_degree: float = 0.25
    w_pagerank: float = 0.20
    w_betweenness: float = 0.20
    w_closeness: float = 0.10
    w_centrality: float = 0.50
    w_llm_importance: float = 0.25
    w_evidence: float = 0.25
    source: str = "defaults"

    @classmethod
    def from_record(cls, record: Any | None) -> "GraphHyperparameters":
        if record is None:
            return cls()
        return cls(
            min_relation_confidence=float(getattr(record, "min_relation_confidence", MIN_RELATION_CONFIDENCE) or MIN_RELATION_CONFIDENCE),
            min_accepted_relation_weight=float(getattr(record, "min_accepted_relation_weight", MIN_ACCEPTED_RELATION_WEIGHT) or MIN_ACCEPTED_RELATION_WEIGHT),
            dijkstra_semantic_threshold=float(getattr(record, "dijkstra_semantic_threshold", DIJKSTRA_SEMANTIC_THRESHOLD) or DIJKSTRA_SEMANTIC_THRESHOLD),
            w_degree=float(getattr(record, "w_degree", 0.25) or 0.25),
            w_weighted_degree=float(getattr(record, "w_weighted_degree", 0.25) or 0.25),
            w_pagerank=float(getattr(record, "w_pagerank", 0.20) or 0.20),
            w_betweenness=float(getattr(record, "w_betweenness", 0.20) or 0.20),
            w_closeness=float(getattr(record, "w_closeness", 0.10) or 0.10),
            w_centrality=float(getattr(record, "w_centrality", 0.50) or 0.50),
            w_llm_importance=float(getattr(record, "w_llm_importance", 0.25) or 0.25),
            w_evidence=float(getattr(record, "w_evidence", 0.25) or 0.25),
            source=f"course_model_hyperparameters:{course_model_hyperparameter_key(record)}",
        ).normalized()

    def normalized(self) -> "GraphHyperparameters":
        centrality = _normalize_weights(
            [self.w_degree, self.w_weighted_degree, self.w_pagerank, self.w_betweenness, self.w_closeness],
            [0.25, 0.25, 0.20, 0.20, 0.10],
        )
        rank = _normalize_weights([self.w_centrality, self.w_llm_importance, self.w_evidence], [0.50, 0.25, 0.25])
        return GraphHyperparameters(
            min_relation_confidence=clamp(float(self.min_relation_confidence), 0.0, 1.0),
            min_accepted_relation_weight=clamp(float(self.min_accepted_relation_weight), 0.0, 1.0),
            dijkstra_semantic_threshold=clamp(float(self.dijkstra_semantic_threshold), -1.0, 1.0),
            w_degree=centrality[0],
            w_weighted_degree=centrality[1],
            w_pagerank=centrality[2],
            w_betweenness=centrality[3],
            w_closeness=centrality[4],
            w_centrality=rank[0],
            w_llm_importance=rank[1],
            w_evidence=rank[2],
            source=self.source,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "min_relation_confidence": round(self.min_relation_confidence, 4),
            "min_accepted_relation_weight": round(self.min_accepted_relation_weight, 4),
            "dijkstra_semantic_threshold": round(self.dijkstra_semantic_threshold, 4),
            "centrality_weights": {
                "degree": round(self.w_degree, 4),
                "weighted_degree": round(self.w_weighted_degree, 4),
                "pagerank": round(self.w_pagerank, 4),
                "betweenness": round(self.w_betweenness, 4),
                "closeness": round(self.w_closeness, 4),
            },
            "rank_weights": {
                "centrality": round(self.w_centrality, 4),
                "llm_importance": round(self.w_llm_importance, 4),
                "evidence": round(self.w_evidence, 4),
            },
        }


def _normalize_weights(values: list[float], defaults: list[float]) -> tuple[float, ...]:
    cleaned: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        cleaned.append(number if math.isfinite(number) and number >= 0 else 0.0)
    total = sum(cleaned)
    if total <= 1e-12:
        cleaned = defaults[:]
        total = sum(cleaned)
    return tuple(value / total for value in cleaned)


def build_course_model_hyperparameter_key(
    llm_model_name: str,
    embedding_model_name: str,
    embedding_text_version: str,
) -> str:
    return "|".join(
        [
            f"llm:{llm_model_name}",
            f"embedding:{embedding_model_name}",
            f"text:{embedding_text_version}",
        ]
    )


def course_model_hyperparameter_key(record: Any) -> str:
    legacy = getattr(record, "model_name", None)
    llm_model_name = getattr(record, "llm_model_name", None)
    embedding_model_name = getattr(record, "embedding_model_name", None)
    embedding_text_version = getattr(record, "embedding_text_version", None)
    if llm_model_name and embedding_model_name and embedding_text_version:
        return build_course_model_hyperparameter_key(llm_model_name, embedding_model_name, embedding_text_version)
    return str(legacy or "unknown")


def load_course_graph_hyperparameters(
    db: Session,
    course_id: str,
    llm_model_name: str | None = None,
    embedding_model_name: str | None = None,
    embedding_text_version: str | None = None,
) -> GraphHyperparameters:
    from app.core.config import get_settings
    from app.models import CourseModelHyperparameter
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION

    settings = get_settings()
    selected_llm = llm_model_name or settings.chat_model
    selected_embedding = embedding_model_name or settings.embedding_model
    selected_text_version = embedding_text_version or CURRENT_EMBEDDING_TEXT_VERSION
    model_key = build_course_model_hyperparameter_key(selected_llm, selected_embedding, selected_text_version)
    record = db.get(
        CourseModelHyperparameter,
        {
            "course_id": course_id,
            "llm_model_name": selected_llm,
            "embedding_model_name": selected_embedding,
            "embedding_text_version": selected_text_version,
        },
    )
    if record is None:
        record = db.scalar(
            select(CourseModelHyperparameter).where(
                CourseModelHyperparameter.course_id == course_id,
                CourseModelHyperparameter.model_name == model_key,
            )
        )
    return GraphHyperparameters.from_record(record)


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), percentile))


def adaptive_relation_confidence_threshold(
    relations: list[RelationSignal],
    hyperparameters: GraphHyperparameters | None = None,
) -> dict[str, Any]:
    hyperparameters = (hyperparameters or GraphHyperparameters()).normalized()
    usable = [
        float(relation.confidence)
        for relation in relations
        if relation.confidence
        and relation.confidence > 0.0
        and not relation.is_inferred
        and normalize_relation_type(relation.relation_type) != "related_to"
        and relation_has_direct_evidence(relation)
    ]
    base = {
        "name": "relation_confidence",
        "default_threshold": hyperparameters.min_relation_confidence,
        "sample_count": len(usable),
        "min_sample_count": ADAPTIVE_RELATION_CONFIDENCE_MIN_SAMPLE,
        "floor": ADAPTIVE_RELATION_CONFIDENCE_FLOOR,
        "ceiling": ADAPTIVE_RELATION_CONFIDENCE_CEILING,
    }
    if len(usable) < ADAPTIVE_RELATION_CONFIDENCE_MIN_SAMPLE:
        return {
            **base,
            "enabled": False,
            "threshold": hyperparameters.min_relation_confidence,
            "fallback_reason": "insufficient_samples",
        }
    p25 = _percentile(usable, 25)
    p50 = _percentile(usable, 50)
    p75 = _percentile(usable, 75)
    threshold = clamp(p25, ADAPTIVE_RELATION_CONFIDENCE_FLOOR, ADAPTIVE_RELATION_CONFIDENCE_CEILING)
    if p50 >= 0.82:
        threshold = max(threshold, 0.70)
    return {
        **base,
        "enabled": True,
        "threshold": round(threshold, 4),
        "p25": round(p25, 4),
        "p50": round(p50, 4),
        "p75": round(p75, 4),
        "fallback_reason": None,
    }


def adaptive_concept_keep_bounds(concepts: list[ConceptSignal]) -> dict[str, Any]:
    available = len(concepts)
    chapter_count = len({ref for concept in concepts for ref in concept.chapter_refs if ref})
    if available == 0:
        return {
            "name": "concept_keep_bounds",
            "enabled": False,
            "available_concepts": 0,
            "chapter_count": 0,
            "keep_min": 0,
            "keep_max": 0,
            "fallback_reason": "no_concepts",
        }
    scale_cap = int(max(MAX_GRAPH_CONCEPTS, math.sqrt(available) * 30, chapter_count * 35))
    keep_max = min(available, MAX_ADAPTIVE_GRAPH_CONCEPTS, max(120, scale_cap))
    keep_min = min(keep_max, max(min(MIN_GRAPH_CONCEPTS, available), min(available, chapter_count * 8)))
    return {
        "name": "concept_keep_bounds",
        "enabled": True,
        "available_concepts": available,
        "chapter_count": chapter_count,
        "default_keep_min": MIN_GRAPH_CONCEPTS,
        "default_keep_max": MAX_GRAPH_CONCEPTS,
        "hard_keep_max": MAX_ADAPTIVE_GRAPH_CONCEPTS,
        "keep_min": int(keep_min),
        "keep_max": int(keep_max),
        "fallback_reason": None,
    }


def adaptive_graph_quality_thresholds(
    concepts: list[ConceptSignal],
    relations: list[RelationSignal],
    hyperparameters: GraphHyperparameters | None = None,
) -> GraphQualityThresholds:
    hyperparameters = (hyperparameters or GraphHyperparameters()).normalized()
    relation_audit = adaptive_relation_confidence_threshold(relations, hyperparameters)
    concept_audit = adaptive_concept_keep_bounds(concepts)
    audit = {
        "relation_confidence": relation_audit,
        "concept_keep_bounds": concept_audit,
        "hyperparameters": hyperparameters.audit(),
    }
    return GraphQualityThresholds(
        relation_confidence=float(relation_audit["threshold"]),
        min_accepted_relation_weight=hyperparameters.min_accepted_relation_weight,
        dijkstra_semantic_threshold=hyperparameters.dijkstra_semantic_threshold,
        concept_keep_min=int(concept_audit["keep_min"]),
        concept_keep_max=int(concept_audit["keep_max"]),
        audit=audit,
    )


def dynamic_k(evidence_count: int) -> int:
    return int(clamp(4 + math.floor(math.log2(1 + max(evidence_count, 0))), 4, 12))


def dynamic_r(chapter_ref_count: int) -> int:
    return int(clamp(2 + math.floor(math.log2(1 + max(chapter_ref_count, 0))), 2, 8))


def edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def normalized_vector(values: list[float]) -> tuple[float, ...] | None:
    floats = []
    for value in values:
        f = float(value)
        if not math.isfinite(f):
            return None
        floats.append(f)
    magnitude = math.sqrt(sum(f * f for f in floats))
    if magnitude <= 1e-12:
        return None
    return tuple(f / magnitude for f in floats)


def average_vectors(vectors: list[list[float]]) -> tuple[float, ...] | None:
    if not vectors:
        return None
    width = len(vectors[0])
    usable = [vector for vector in vectors if len(vector) == width]
    if not usable:
        return None
    averaged = [sum(vector[index] for vector in usable) / len(usable) for index in range(width)]
    return normalized_vector(averaged)


def safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_relation_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    relation_type = value.strip().lower()
    relation_type = LEGACY_RELATION_TYPE_MAP.get(relation_type, relation_type)
    return relation_type if relation_type in ALLOWED_COMPLETION_RELATIONS else ""


def exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {exc!r}"


def similarity(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(cosine_similarity(list(left), list(right)))


def normalize_evidence_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\u0370-\u03ff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def evidence_mentions_name(haystack: str, name: str | None) -> bool:
    normalized = normalize_evidence_text(name)
    if len(normalized) < 3:
        return False
    return normalized in haystack


def concept_name_candidates(concept: Any | None) -> set[str]:
    if concept is None:
        return set()
    candidates = {
        value
        for value in (
            getattr(concept, "canonical_name", None),
            getattr(concept, "normalized_name", None),
        )
        if isinstance(value, str) and value.strip()
    }
    for alias in getattr(concept, "aliases", []) or []:
        alias_value = getattr(alias, "alias", None)
        if isinstance(alias_value, str) and alias_value.strip():
            candidates.add(alias_value)
    return candidates


def relation_evidence_metadata(relation: Any, concepts_by_id: dict[str, Any], chunks_by_id: dict[str, Any]) -> dict[str, Any]:
    evidence_chunk_id = getattr(relation, "evidence_chunk_id", None)
    chunk = chunks_by_id.get(evidence_chunk_id) if evidence_chunk_id else None
    if chunk is None:
        return {}
    haystack = normalize_evidence_text(
        "\n".join(
            str(value)
            for value in (
                getattr(chunk, "section", None),
                getattr(chunk, "snippet", None),
                getattr(chunk, "content", None),
            )
            if value
        )
    )
    source = concepts_by_id.get(getattr(relation, "source_concept_id", None))
    target = concepts_by_id.get(getattr(relation, "target_concept_id", None))
    source_match = any(evidence_mentions_name(haystack, name) for name in concept_name_candidates(source))
    target_match = any(evidence_mentions_name(haystack, name) for name in concept_name_candidates(target))
    return {
        "hard_gate_evidence_chunk_id": evidence_chunk_id,
        "evidence_source_match": source_match,
        "evidence_target_match": target_match,
        "evidence_endpoint_match_count": int(source_match) + int(target_match),
    }


def relation_has_direct_evidence(relation: RelationSignal) -> bool:
    metadata = relation.metadata or {}
    return bool(
        relation.evidence_chunk_id
        or metadata.get("completion")
        or metadata.get("evidence_support")
        or (metadata.get("evidence_source_match") and metadata.get("evidence_target_match"))
    )


def relation_evidence_support_score(relation: RelationSignal) -> float:
    metadata = relation.metadata or {}
    if metadata.get("evidence_source_match") and metadata.get("evidence_target_match"):
        base = 1.0
    elif relation.evidence_chunk_id or metadata.get("completion") or metadata.get("evidence_support"):
        base = 0.72
    else:
        base = 0.0
    return clamp(max(base, math.log1p(max(relation.support_count, 0)) / math.log1p(4)), 0.0, 1.0)


def cooccurrence_strength(left: ConceptSignal, right: ConceptSignal, relation: RelationSignal, semantic: float) -> float:
    same_chapter = bool(set(left.chapter_refs).intersection(set(right.chapter_refs)))
    support = clamp(math.log1p(max(relation.support_count, 0)) / math.log1p(4), 0.0, 1.0)
    semantic_component = clamp((semantic + 1.0) / 2.0, 0.0, 1.0)
    return clamp(0.45 * support + 0.35 * float(same_chapter) + 0.20 * semantic_component, 0.0, 1.0)


def relation_hard_gate(
    relation: RelationSignal,
    left: ConceptSignal,
    right: ConceptSignal,
    semantic: float,
    weight: float | None = None,
    cooccurrence: float | None = None,
    thresholds: GraphQualityThresholds | None = None,
) -> tuple[bool, str]:
    thresholds = thresholds or adaptive_graph_quality_thresholds([left, right], [relation])
    min_relation_confidence = thresholds.relation_confidence
    relation_type = normalize_relation_type(relation.relation_type)
    if relation_type not in ALLOWED_COMPLETION_RELATIONS:
        return False, "relation_type_not_allowed"
    if relation.source_id == relation.target_id:
        return False, "self_relation"
    if relation.source_id not in {left.concept_id, right.concept_id} or relation.target_id not in {left.concept_id, right.concept_id}:
        return False, "endpoint_mismatch"
    if not math.isfinite(relation.confidence) or relation.confidence < min_relation_confidence:
        return False, "confidence_too_low"

    direct_evidence = relation_has_direct_evidence(relation)
    if relation.is_inferred:
        return False, "inferred_relation_not_validated"
    if not direct_evidence or relation.support_count < 1:
        return False, "missing_direct_evidence"
    cooccurrence = cooccurrence_strength(left, right, relation, semantic) if cooccurrence is None else cooccurrence
    if semantic < MIN_WEAK_RELATION_SEMANTIC and cooccurrence < MIN_COOCCURRENCE_STRENGTH:
        return False, "weak_semantic_and_cooccurrence"
    if weight is not None:
        if not math.isfinite(weight) or weight < 0:
            return False, "invalid_weight"
        if relation_type == "related_to" and weight < MIN_RELATED_TO_DISPLAY_WEIGHT:
            return False, "related_to_below_display_threshold"
        if weight < thresholds.min_accepted_relation_weight:
            return False, "weight_below_threshold"
    return True, "passed"


def structure_score(left: ConceptSignal, right: ConceptSignal) -> float:
    left_refs = set(left.chapter_refs)
    right_refs = set(right.chapter_refs)
    if left_refs and right_refs and left_refs.intersection(right_refs):
        return 1.0
    if left_refs or right_refs:
        return 0.35
    return 0.0


def evidence_score(left: ConceptSignal, right: ConceptSignal) -> float:
    return clamp(math.log1p(left.evidence_count + right.evidence_count) / math.log1p(10), 0.0, 1.0)


def weighted_score(
    left: ConceptSignal,
    right: ConceptSignal,
    llm_confidence: float,
    semantic: float,
    *,
    evidence_support: float | None = None,
    cooccurrence: float | None = None,
    structure_consistency: float | None = None,
) -> float:
    relation_type_confidence = clamp(llm_confidence, 0.0, 1.0)
    evidence_support = evidence_score(left, right) if evidence_support is None else clamp(evidence_support, 0.0, 1.0)
    cooccurrence = 0.0 if cooccurrence is None else clamp(cooccurrence, 0.0, 1.0)
    structure_consistency = structure_score(left, right) if structure_consistency is None else clamp(structure_consistency, 0.0, 1.0)
    raw = (
        0.30 * evidence_support
        + 0.25 * relation_type_confidence
        + 0.20 * clamp((semantic + 1.0) / 2.0, 0.0, 1.0)
        + 0.15 * cooccurrence
        + 0.10 * structure_consistency
    )
    return clamp(raw, 0.0, 1.0)


def build_sparse_edges(
    concepts: list[ConceptSignal],
    relations: list[RelationSignal],
    thresholds: GraphQualityThresholds | None = None,
    hyperparameters: GraphHyperparameters | None = None,
) -> list[WeightedEdge]:
    thresholds = thresholds or adaptive_graph_quality_thresholds(concepts, relations, hyperparameters)
    concept_index = {concept.concept_id: concept for concept in concepts}
    relation_candidates: dict[tuple[str, str], list[RelationSignal]] = defaultdict(list)
    for relation in relations:
        relation_type = normalize_relation_type(relation.relation_type)
        if (
            relation.source_id not in concept_index
            or relation.target_id not in concept_index
            or relation.source_id == relation.target_id
            or relation_type not in ALLOWED_COMPLETION_RELATIONS
        ):
            continue
        if relation_type != relation.relation_type:
            relation = RelationSignal(
                relation.source_id,
                relation.target_id,
                relation_type,
                relation.confidence,
                support_count=relation.support_count,
                evidence_chunk_id=relation.evidence_chunk_id,
                relation_source=relation.relation_source,
                is_inferred=relation.is_inferred,
                metadata=relation.metadata,
            )
        key = edge_key(relation.source_id, relation.target_id)
        relation_candidates[key].append(relation)

    similarities: dict[tuple[str, str], float] = {}
    outgoing: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for index, left in enumerate(concepts):
        for right in concepts[index + 1 :]:
            semantic = similarity(left.vector, right.vector)
            similarities[edge_key(left.concept_id, right.concept_id)] = semantic
            outgoing[left.concept_id].append((right.concept_id, semantic))
            outgoing[right.concept_id].append((left.concept_id, semantic))

    llm_by_pair: dict[tuple[str, str], RelationSignal] = {}
    gate_metadata_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, candidates in relation_candidates.items():
        source_id, target_id = pair
        left = concept_index[source_id]
        right = concept_index[target_id]
        semantic = similarities.get(pair, 0.0)
        accepted: list[tuple[RelationSignal, str]] = []
        rejected_reasons: list[str] = []
        for relation in candidates:
            evidence_support = relation_evidence_support_score(relation)
            cooccurrence = cooccurrence_strength(left, right, relation, semantic)
            weight = weighted_score(
                left,
                right,
                relation.confidence,
                semantic,
                evidence_support=evidence_support,
                cooccurrence=cooccurrence,
            )
            passed, reason = relation_hard_gate(
                relation,
                left,
                right,
                semantic,
                weight=weight,
                cooccurrence=cooccurrence,
                thresholds=thresholds,
            )
            if passed:
                accepted.append((relation, reason))
            else:
                rejected_reasons.append(reason)
        if not accepted:
            gate_metadata_by_pair[pair] = {"hard_gate": "rejected", "hard_gate_reasons": sorted(set(rejected_reasons))}
            continue
        chosen, _reason = max(accepted, key=lambda item: (item[0].confidence, item[0].support_count))
        llm_by_pair[pair] = chosen
        gate_metadata_by_pair[pair] = {
            "hard_gate": "accepted",
            "hard_gate_rejected_candidates": len(rejected_reasons),
        }

    selected_out: dict[str, set[str]] = defaultdict(set)
    inbound_candidates: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for concept in concepts:
        ranked = sorted(outgoing[concept.concept_id], key=lambda item: item[1], reverse=True)
        for target_id, semantic in ranked[: dynamic_k(concept.evidence_count)]:
            if semantic < SEMANTIC_EDGE_THRESHOLD:
                continue
            selected_out[concept.concept_id].add(target_id)
            inbound_candidates[target_id].append((concept.concept_id, semantic))

    accepted_in: dict[str, set[str]] = defaultdict(set)
    for concept in concepts:
        ranked_in = sorted(inbound_candidates[concept.concept_id], key=lambda item: item[1], reverse=True)
        for source_id, _semantic in ranked_in[: dynamic_r(len(concept.chapter_refs))]:
            accepted_in[concept.concept_id].add(source_id)

    edge_pairs: set[tuple[str, str]] = set(llm_by_pair)
    for source_id, targets in selected_out.items():
        for target_id in targets:
            mutual = source_id in selected_out.get(target_id, set())
            reciprocal_accept = source_id in accepted_in.get(target_id, set()) or target_id in accepted_in.get(source_id, set())
            if mutual or reciprocal_accept:
                edge_pairs.add(edge_key(source_id, target_id))

    weighted_edges: list[WeightedEdge] = []
    for pair in sorted(edge_pairs):
        source_id, target_id = pair
        left = concept_index[source_id]
        right = concept_index[target_id]
        llm_relation = llm_by_pair.get(pair)
        semantic = similarities.get(pair, 0.0)
        confidence = llm_relation.confidence if llm_relation else 0.0
        evidence_support = relation_evidence_support_score(llm_relation) if llm_relation else evidence_score(left, right)
        cooccurrence = cooccurrence_strength(left, right, llm_relation, semantic) if llm_relation else 0.0
        weight = weighted_score(
            left,
            right,
            confidence,
            semantic,
            evidence_support=evidence_support,
            cooccurrence=cooccurrence,
        )
        if not llm_relation and weight < 0.42:
            continue
        weighted_edges.append(
            WeightedEdge(
                source_id=llm_relation.source_id if llm_relation else source_id,
                target_id=llm_relation.target_id if llm_relation else target_id,
                relation_type=llm_relation.relation_type if llm_relation else "related_to",
                weight=weight,
                semantic_similarity=semantic,
                support_count=llm_relation.support_count if llm_relation else min(left.evidence_count, right.evidence_count),
                confidence=confidence if llm_relation else round(weight, 4),
                evidence_chunk_id=llm_relation.evidence_chunk_id if llm_relation else None,
                relation_source=llm_relation.relation_source if llm_relation else "semantic_sparse",
                is_inferred=llm_relation.is_inferred if llm_relation else True,
                metadata={
                    **((llm_relation.metadata or {}) if llm_relation else {"semantic_sparse": True, "candidate_only": True}),
                    "evidence_support": round(evidence_support, 4),
                    "cooccurrence_strength": round(cooccurrence, 4),
                    "relation_confidence_threshold": round(thresholds.relation_confidence, 4),
                    "adaptive_thresholds": thresholds.audit,
                    **gate_metadata_by_pair.get(pair, {"hard_gate": "semantic_sparse"}),
                },
            )
        )
    return weighted_edges


def graph_from_edges(concepts: list[ConceptSignal], edges: list[WeightedEdge]) -> nx.Graph:
    graph = nx.Graph()
    for concept in concepts:
        graph.add_node(concept.concept_id, importance=concept.importance, evidence_count=concept.evidence_count)
    for edge in edges:
        if edge.relation_type == "related_to" or edge.metadata.get("candidate_only"):
            continue
        if not math.isfinite(edge.weight) or edge.weight < 0:
            continue
        graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight, semantic_similarity=edge.semantic_similarity)
    return graph


def _deterministic_kmeans(matrix: np.ndarray, k: int, iterations: int = 16) -> list[int]:
    if len(matrix) == 0:
        return []
    if k <= 1:
        return [0 for _ in range(len(matrix))]
    centers = matrix[np.linspace(0, len(matrix) - 1, k, dtype=int)].copy()
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        for center_index in range(k):
            members = matrix[labels == center_index]
            if len(members):
                centers[center_index] = members.mean(axis=0)
    return [int(label) for label in labels]


def spectral_labels_for_graph(graph: nx.Graph) -> dict[str, int]:
    labels: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    large_components = [component for component in nx.connected_components(graph) if len(component) >= 8]
    next_label = 0
    for component in sorted(large_components, key=len, reverse=True):
        subgraph = graph.subgraph(component)
        nodes = list(subgraph.nodes)
        k = int(clamp(round(math.sqrt(len(nodes)) / 2), 2, min(8, len(nodes) - 1)))
        try:
            laplacian = nx.normalized_laplacian_matrix(subgraph, weight="weight")
            from scipy.sparse.linalg import eigsh

            _values, vectors = eigsh(laplacian, k=k, which="SM")
            component_labels = _deterministic_kmeans(np.asarray(vectors), k)
        except Exception:
            component_labels = [0 if index < len(nodes) / 2 else 1 for index in range(len(nodes))]
        label_offset = next_label
        for node_id, label in zip(nodes, component_labels):
            labels[node_id] = label_offset + label
        next_label += max(component_labels, default=0) + 1
    return labels


def analyze_graph(
    concepts: list[ConceptSignal],
    edges: list[WeightedEdge],
    hyperparameters: GraphHyperparameters | None = None,
) -> tuple[dict[str, dict[str, Any]], nx.Graph]:
    hyperparameters = (hyperparameters or GraphHyperparameters()).normalized()
    graph = graph_from_edges(concepts, edges)
    if graph.number_of_nodes() == 0:
        return {}, graph

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_by_node = {node_id: index for index, component in enumerate(components) for node_id in component}
    try:
        communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=42)
    except Exception:
        communities = components
    louvain_by_node = {node_id: index for index, community in enumerate(communities) for node_id in community}
    spectral_by_node = spectral_labels_for_graph(graph)
    degree = nx.degree_centrality(graph)
    weighted_degree = {
        node_id: sum(float(data.get("weight", 0.0)) for _source, _target, data in graph.edges(node_id, data=True))
        for node_id in graph.nodes
    }
    if graph.number_of_edges():
        pagerank = nx.pagerank(graph, weight="weight")
        betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
        closeness = nx.closeness_centrality(graph, distance=lambda _u, _v, data: 1 / (0.05 + float(data.get("weight", 0.0))))
    else:
        pagerank = {node_id: 0.0 for node_id in graph.nodes}
        betweenness = {node_id: 0.0 for node_id in graph.nodes}
        closeness = {node_id: 0.0 for node_id in graph.nodes}
    max_weighted_degree = max(weighted_degree.values(), default=1.0) or 1.0

    metrics: dict[str, dict[str, Any]] = {}
    concept_index = {concept.concept_id: concept for concept in concepts}
    for node_id in graph.nodes:
        signal = concept_index[node_id]
        weighted_degree_norm = weighted_degree[node_id] / max_weighted_degree
        centrality_score = clamp(
            hyperparameters.w_degree * degree.get(node_id, 0.0)
            + hyperparameters.w_weighted_degree * weighted_degree_norm
            + hyperparameters.w_pagerank * pagerank.get(node_id, 0.0)
            + hyperparameters.w_betweenness * betweenness.get(node_id, 0.0)
            + hyperparameters.w_closeness * closeness.get(node_id, 0.0),
            0.0,
            1.0,
        )
        graph_rank_score = clamp(
            hyperparameters.w_centrality * centrality_score
            + hyperparameters.w_llm_importance * signal.importance
            + hyperparameters.w_evidence * clamp(math.log1p(signal.evidence_count) / math.log1p(10), 0.0, 1.0),
            0.0,
            1.0,
        )
        def _safe_jsonb_float(v: float) -> float:
            return v if math.isfinite(v) else 0.0

        metrics[node_id] = {
            "component_id": component_by_node.get(node_id, 0),
            "community_louvain": louvain_by_node.get(node_id, 0),
            "community_spectral": spectral_by_node.get(node_id, 0),
            "centrality": {
                "degree": _safe_jsonb_float(degree.get(node_id, 0.0)),
                "weighted_degree": _safe_jsonb_float(weighted_degree.get(node_id, 0.0)),
                "pagerank": _safe_jsonb_float(pagerank.get(node_id, 0.0)),
                "betweenness": _safe_jsonb_float(betweenness.get(node_id, 0.0)),
                "closeness": _safe_jsonb_float(closeness.get(node_id, 0.0)),
                "centrality_score": _safe_jsonb_float(centrality_score),
                "hyperparameter_source": hyperparameters.source,
            },
            "graph_rank_score": _safe_jsonb_float(graph_rank_score),
        }
    return metrics, graph


def infer_dijkstra_edges(
    graph: nx.Graph,
    concepts: list[ConceptSignal],
    existing_edges: list[WeightedEdge],
    hyperparameters: GraphHyperparameters | None = None,
) -> list[WeightedEdge]:
    hyperparameters = (hyperparameters or GraphHyperparameters()).normalized()
    concept_index = {concept.concept_id: concept for concept in concepts}
    existing_pairs = {edge_key(edge.source_id, edge.target_id) for edge in existing_edges}
    inferred: list[WeightedEdge] = []
    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return inferred

    def cost(_source: str, _target: str, data: dict[str, Any]) -> float:
        return 1 / (0.05 + float(data.get("weight", 0.0)))

    nodes = list(graph.nodes)
    for source_id in nodes:
        lengths, paths = nx.single_source_dijkstra(graph, source_id, cutoff=7.5, weight=cost)
        for target_id, path in paths.items():
            if source_id >= target_id or len(path) not in {3, 4}:
                continue
            pair = edge_key(source_id, target_id)
            if pair in existing_pairs:
                continue
            left = concept_index[source_id]
            right = concept_index[target_id]
            semantic = similarity(left.vector, right.vector)
            if semantic < hyperparameters.dijkstra_semantic_threshold:
                continue
            path_weight = 1 / (1 + float(lengths[target_id]))
            edge_weight = weighted_score(left, right, path_weight, semantic)
            if edge_weight < 0.48:
                continue
            inferred.append(
                WeightedEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type="related_to",
                    weight=edge_weight,
                    semantic_similarity=semantic,
                    support_count=min(left.evidence_count, right.evidence_count),
                    confidence=round(edge_weight, 4),
                    evidence_chunk_id=None,
                    relation_source="dijkstra_inferred",
                    is_inferred=True,
                    metadata={"path": path, "path_cost": float(lengths[target_id])},
                )
            )
    return sorted(inferred, key=lambda edge: edge.weight, reverse=True)[: max(20, len(concepts) // 5)]


def select_concepts_to_keep(
    concepts: list[ConceptSignal],
    metrics: dict[str, dict[str, Any]],
    graph: nx.Graph,
    thresholds: GraphQualityThresholds | None = None,
) -> set[str]:
    thresholds = thresholds or adaptive_graph_quality_thresholds(concepts, [])
    available = len(concepts)
    if available <= thresholds.concept_keep_max:
        floor = min(thresholds.concept_keep_min, available)
    else:
        floor = thresholds.concept_keep_min
    keep_limit = min(thresholds.concept_keep_max, available)
    noisy = {
        concept.concept_id
        for concept in concepts
        if graph.degree(concept.concept_id) == 0 and concept.evidence_count <= 1 and concept.importance < 0.55
    }
    ranked = sorted(
        concepts,
        key=lambda concept: (
            metrics.get(concept.concept_id, {}).get("graph_rank_score", 0.0),
            concept.importance,
            concept.evidence_count,
        ),
        reverse=True,
    )
    keep: list[str] = [concept.concept_id for concept in ranked if concept.concept_id not in noisy][:keep_limit]
    if len(keep) < floor:
        for concept in ranked:
            if concept.concept_id not in keep:
                keep.append(concept.concept_id)
            if len(keep) >= floor:
                break
    return set(keep)


def relation_signals_from_db(
    relations: list[ConceptRelation],
    concepts: list[Any] | None = None,
    chunks_by_id: dict[str, Any] | None = None,
) -> list[RelationSignal]:
    from app.models import ConceptRelation

    by_key: dict[tuple[str, str, str], RelationSignal] = {}
    concepts_by_id = {concept.id: concept for concept in concepts or []}
    chunks_by_id = chunks_by_id or {}
    for relation in relations:
        if not relation.target_concept_id:
            continue
        relation_metadata = getattr(relation, "metadata_json", None) or {}
        if relation_metadata.get("candidate_only") or getattr(relation, "relation_source", None) in {"semantic_sparse", "dijkstra_inferred"}:
            continue
        relation_type = normalize_relation_type(relation.relation_type)
        if not relation_type:
            continue
        key = (relation.source_concept_id, relation.target_concept_id, relation_type)
        current = by_key.get(key)
        support = int(getattr(relation, "support_count", 1) or 1)
        metadata = {
            **(getattr(relation, "metadata_json", None) or {}),
            **relation_evidence_metadata(relation, concepts_by_id, chunks_by_id),
        }
        signal = RelationSignal(
            source_id=relation.source_concept_id,
            target_id=relation.target_concept_id,
            relation_type=relation_type,
            confidence=float(relation.confidence or 0.0),
            support_count=support,
            evidence_chunk_id=relation.evidence_chunk_id,
            relation_source=getattr(relation, "relation_source", None) or relation.extraction_method or "llm",
            is_inferred=bool(getattr(relation, "is_inferred", False)),
            metadata=metadata,
        )
        if current is None or signal.confidence > current.confidence:
            by_key[key] = signal
    return list(by_key.values())


def completion_relation_signals(payload: Any, by_name: dict[str, Any]) -> list[RelationSignal]:
    if not isinstance(payload, dict):
        return []
    raw_relations = payload.get("relations", [])
    if not isinstance(raw_relations, list):
        return []

    signals: list[RelationSignal] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        source_name = relation.get("source")
        target_name = relation.get("target")
        relation_type = normalize_relation_type(relation.get("relation_type", "related_to"))
        if not isinstance(source_name, str) or not isinstance(target_name, str) or relation_type not in ALLOWED_COMPLETION_RELATIONS:
            continue
        source = by_name.get(source_name.strip())
        target = by_name.get(target_name.strip())
        if not source or not target or source.id == target.id:
            continue
        key = (source.id, target.id, relation_type)
        if key in seen:
            continue
        seen.add(key)
        signals.append(
            RelationSignal(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                confidence=clamp(safe_float(relation.get("confidence"), 0.5), 0.0, 1.0),
                relation_source="llm_traversal_completion",
                metadata={"completion": True},
            )
        )
    return signals


def child_chunk_filter(chunk: Chunk) -> bool:
    return bool(chunk.is_active) and not bool((chunk.metadata_json or {}).get("is_parent"))


def collect_concept_signals(db: Session, course: Course, concepts: list[Concept], relations: list[ConceptRelation]) -> list[ConceptSignal]:
    from app.models import Chunk

    active_chunks = db.scalars(select(Chunk).where(Chunk.course_id == course.id, Chunk.is_active.is_(True))).all()
    child_chunks = [chunk for chunk in active_chunks if child_chunk_filter(chunk)]
    evidence_by_concept: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.evidence_chunk_id:
            evidence_by_concept[relation.source_concept_id].add(relation.evidence_chunk_id)
            if relation.target_concept_id:
                evidence_by_concept[relation.target_concept_id].add(relation.evidence_chunk_id)

    searchable_chunks = child_chunks[:]
    for concept in concepts:
        if evidence_by_concept.get(concept.id):
            continue
        normalized = (concept.normalized_name or "").lower()
        if not normalized:
            continue
        for chunk in searchable_chunks:
            haystack = f"{chunk.section or ''}\n{chunk.snippet or ''}\n{chunk.content[:1200]}".lower()
            if normalized in haystack:
                evidence_by_concept[concept.id].add(chunk.id)
                if len(evidence_by_concept[concept.id]) >= 4:
                    break

    vector_ids = sorted({chunk_id for ids in evidence_by_concept.values() for chunk_id in ids})
    point_vectors: dict[str, list[float]] = {}
    if vector_ids:
        vector_store = VectorStore(course_name=course.name)
        for index in range(0, len(vector_ids), 128):
            for point in vector_store.get_points(vector_ids[index : index + 128]):
                vector = point.get("vector")
                if isinstance(vector, list):
                    point_vectors[str(point["id"])] = vector

    signals: list[ConceptSignal] = []
    for concept in concepts:
        ids = evidence_by_concept.get(concept.id, set())
        vectors = [point_vectors[chunk_id] for chunk_id in ids if chunk_id in point_vectors]
        signals.append(
            ConceptSignal(
                concept_id=concept.id,
                importance=float(concept.importance_score or 0.0),
                evidence_count=len(ids),
                chapter_refs=tuple(concept.chapter_refs or []),
                vector=average_vectors(vectors),
            )
        )
    return signals


async def complete_relations_with_llm(
    db: Session,
    course_id: str,
    concepts_by_id: dict[str, Concept],
    graph: nx.Graph,
    metrics: dict[str, dict[str, Any]],
    limit: int = 8,
) -> list[RelationSignal]:
    from app.models import Chunk, ConceptRelation

    ranked_ids = sorted(metrics, key=lambda node_id: metrics[node_id]["graph_rank_score"], reverse=True)[:limit]
    if not ranked_ids:
        return []
    evidence_chunk_ids = {
        relation.evidence_chunk_id
        for relation in db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
        if relation.evidence_chunk_id
    }
    chunks = {
        chunk.id: chunk
        for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(evidence_chunk_ids))).all()
    } if evidence_chunk_ids else {}
    candidates: list[dict[str, Any]] = []
    for node_id in ranked_ids:
        neighbors = set(nx.single_source_shortest_path_length(graph, node_id, cutoff=2).keys())
        names = [concepts_by_id[item].canonical_name for item in neighbors if item in concepts_by_id]
        snippets = []
        for relation in db.scalars(
            select(ConceptRelation).where(
                ConceptRelation.course_id == course_id,
                ConceptRelation.source_concept_id.in_(neighbors),
            )
        ).all():
            chunk = chunks.get(relation.evidence_chunk_id or "")
            if chunk is not None:
                snippets.append(chunk.snippet[:300])
            if len(snippets) >= 5:
                break
        candidates.append({"center": concepts_by_id[node_id].canonical_name, "concepts": names[:18], "evidence": snippets})

    system_prompt = "You complete course graph relations from supplied evidence only. Return strict JSON."
    user_prompt = (
        "Return JSON with keys concepts and relations. Reuse only concept names shown below. "
        "Each relation must be directly supported by the evidence snippets; do not infer unsupported facts. "
        f"Allowed relation_type values: {sorted(ALLOWED_COMPLETION_RELATIONS)}.\n\n"
        f"{candidates}"
    )
    payload = await ChatProvider().classify_json(system_prompt, user_prompt, fallback={})
    by_name = {concept.canonical_name: concept for concept in concepts_by_id.values()}
    return completion_relation_signals(payload, by_name)


def upsert_weighted_edges(db: Session, course_id: str, edges: list[WeightedEdge]) -> int:
    from app.models import Concept, ConceptRelation, GraphRelationCandidate

    existing = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    by_exact = {
        (relation.source_concept_id, relation.target_concept_id, relation.relation_type): relation
        for relation in existing
        if relation.target_concept_id
    }
    changed = 0
    for edge in edges:
        relation_type = normalize_relation_type(edge.relation_type)
        if not relation_type:
            continue
        candidate_only = bool(
            edge.metadata.get("candidate_only")
            or edge.relation_source in {"semantic_sparse", "dijkstra_inferred"}
            or (relation_type == "related_to" and edge.weight < MIN_RELATED_TO_DISPLAY_WEIGHT)
        )
        source = db.get(Concept, edge.source_id)
        target = db.get(Concept, edge.target_id)
        if source is None or target is None:
            continue
        evidence_text = f"{source.canonical_name} {target.canonical_name}" if edge.metadata.get("evidence_source_match") and edge.metadata.get("evidence_target_match") else ""
        signals = build_quality_signals(
            target_type="relation",
            text=f"{source.canonical_name} {relation_type} {target.canonical_name}",
            course_id=course_id,
            chunk_id=edge.evidence_chunk_id,
            evidence_text=evidence_text,
            source_name=source.canonical_name,
            target_name=target.canonical_name,
            support_count=edge.support_count,
            extractor=edge.relation_source,
            version="relation_quality_v1",
        )
        decision = RelationQualityPolicy().decide(
            signals,
            relation_type=relation_type,
            confidence=edge.confidence,
            weight=edge.weight,
            inferred=edge.is_inferred,
            min_confidence=float(edge.metadata.get("relation_confidence_threshold", MIN_RELATION_CONFIDENCE)),
        )
        candidate_only = candidate_only or decision.action == "candidate_only"
        if candidate_only:
            candidate = db.scalar(
                select(GraphRelationCandidate).where(
                    GraphRelationCandidate.course_id == course_id,
                    GraphRelationCandidate.source_concept_id == edge.source_id,
                    GraphRelationCandidate.target_concept_id == edge.target_id,
                    GraphRelationCandidate.relation_type == relation_type,
                    GraphRelationCandidate.relation_source == edge.relation_source,
                )
            )
            if candidate is None:
                candidate = GraphRelationCandidate(
                    course_id=course_id,
                    source_concept_id=edge.source_id,
                    target_concept_id=edge.target_id,
                    target_name=target.canonical_name,
                    relation_type=relation_type,
                    relation_source=edge.relation_source,
                )
                db.add(candidate)
            candidate.evidence_chunk_id = edge.evidence_chunk_id
            candidate.confidence = edge.confidence
            candidate.weight = edge.weight
            candidate.semantic_similarity = edge.semantic_similarity
            candidate.support_count = edge.support_count
            candidate.is_inferred = edge.is_inferred
            candidate.decision_json = decision.model_dump()
            candidate.metadata_json = {**(getattr(candidate, "metadata_json", None) or {}), **edge.metadata}
            changed += 1
            continue
        key = (edge.source_id, edge.target_id, relation_type)
        relation = by_exact.get(key)
        if relation is None:
            relation = ConceptRelation(
                course_id=course_id,
                source_concept_id=edge.source_id,
                target_concept_id=edge.target_id,
                target_name=target.canonical_name,
                relation_type=relation_type,
                evidence_chunk_id=edge.evidence_chunk_id,
                confidence=edge.confidence,
                extraction_method=edge.relation_source,
                is_validated=False,
            )
            db.add(relation)
            by_exact[key] = relation
        direct_evidence = bool(edge.metadata.get("evidence_source_match") and edge.metadata.get("evidence_target_match"))
        relation.weight = max(float(getattr(relation, "weight", 0.0) or 0.0), edge.weight)
        relation.semantic_similarity = max(float(getattr(relation, "semantic_similarity", 0.0) or 0.0), edge.semantic_similarity)
        relation.support_count = max(int(getattr(relation, "support_count", 1) or 1), int(edge.support_count or 1))
        relation.relation_source = edge.relation_source
        relation.is_inferred = bool(edge.is_inferred)
        relation.metadata_json = {
            **(getattr(relation, "metadata_json", None) or {}),
            **edge.metadata,
            "candidate_only": candidate_only,
            "quality_decision": decision.model_dump(),
        }
        relation.is_validated = bool(
            not edge.is_inferred
            and direct_evidence
            and edge.confidence >= float(edge.metadata.get("relation_confidence_threshold", MIN_RELATION_CONFIDENCE))
            and not candidate_only
            and edge.metadata.get("hard_gate") == "accepted"
        )
        if not relation.evidence_chunk_id and edge.evidence_chunk_id:
            relation.evidence_chunk_id = edge.evidence_chunk_id
        changed += 1
    db.flush()
    return changed


def write_concept_metrics(
    db: Session,
    concepts: list[Concept],
    signals: list[ConceptSignal],
    metrics: dict[str, dict[str, Any]],
    keep_ids: set[str],
) -> int:
    from app.models import Concept, ConceptAlias, ConceptRelation, GraphRelationCandidate

    signals_by_id = {signal.concept_id: signal for signal in signals}
    updated = 0
    for concept in concepts:
        if concept.id not in keep_ids:
            continue
        signal = signals_by_id.get(concept.id)
        metric = metrics.get(concept.id, {})
        concept.evidence_count = signal.evidence_count if signal else 0
        concept.community_louvain = metric.get("community_louvain")
        concept.community_spectral = metric.get("community_spectral")
        concept.component_id = metric.get("component_id")
        previous_quality_gate = (getattr(concept, "centrality_json", None) or {}).get("quality_gate")
        centrality = dict(metric.get("centrality", {}) or {})
        if previous_quality_gate:
            centrality["quality_gate"] = previous_quality_gate
        concept.centrality_json = centrality
        concept.graph_rank_score = float(metric.get("graph_rank_score", 0.0))
        updated += 1
    pruned_ids = {concept.id for concept in concepts} - keep_ids
    if pruned_ids:
        db.query(ConceptRelation).filter(
            ConceptRelation.course_id == concepts[0].course_id,
            (ConceptRelation.source_concept_id.in_(pruned_ids) | ConceptRelation.target_concept_id.in_(pruned_ids)),
        ).delete(synchronize_session=False)
        db.query(GraphRelationCandidate).filter(
            GraphRelationCandidate.course_id == concepts[0].course_id,
            (GraphRelationCandidate.source_concept_id.in_(pruned_ids) | GraphRelationCandidate.target_concept_id.in_(pruned_ids)),
        ).delete(synchronize_session=False)
        db.query(ConceptAlias).filter(ConceptAlias.concept_id.in_(pruned_ids)).delete(synchronize_session=False)
        db.query(Concept).filter(Concept.id.in_(pruned_ids)).delete(synchronize_session=False)
    db.flush()
    return updated


async def enrich_course_graph(db: Session, course_id: str, *, run_relation_completion: bool = True, run_dijkstra: bool = True) -> dict[str, Any]:
    from app.models import Chunk, Concept, ConceptRelation, Course

    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"Course not found: {course_id}")
    concepts = db.scalars(select(Concept).where(Concept.course_id == course_id)).all()
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    if not concepts:
        return {"graph_algorithm_nodes": 0, "graph_algorithm_edges": 0}
    hyperparameters = load_course_graph_hyperparameters(db, course_id)
    evidence_chunk_ids = {relation.evidence_chunk_id for relation in relations if relation.evidence_chunk_id}
    chunks_by_id = {
        chunk.id: chunk
        for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(evidence_chunk_ids))).all()
    } if evidence_chunk_ids else {}

    signals = collect_concept_signals(db, course, concepts, relations)
    relation_signals = relation_signals_from_db(relations, concepts, chunks_by_id)
    thresholds = adaptive_graph_quality_thresholds(signals, relation_signals, hyperparameters)
    edges = build_sparse_edges(signals, relation_signals, thresholds=thresholds, hyperparameters=hyperparameters)
    metrics, graph = analyze_graph(signals, edges, hyperparameters=hyperparameters)
    completion_error: str | None = None
    completion_count = 0
    if run_relation_completion:
        try:
            completion_signals = await complete_relations_with_llm(
                db,
                course_id,
                {concept.id: concept for concept in concepts},
                graph,
                metrics,
            )
        except Exception as exc:
            completion_signals = []
            completion_error = exception_summary(exc)
        completion_count = len(completion_signals)
        if completion_signals:
            relation_signals = [*relation_signals, *completion_signals]
            thresholds = adaptive_graph_quality_thresholds(signals, relation_signals, hyperparameters)
            edges = build_sparse_edges(signals, relation_signals, thresholds=thresholds, hyperparameters=hyperparameters)
            metrics, graph = analyze_graph(signals, edges, hyperparameters=hyperparameters)

    inferred_edges: list[WeightedEdge] = []
    if run_dijkstra:
        inferred_edges = infer_dijkstra_edges(graph, signals, edges, hyperparameters=hyperparameters)
        if inferred_edges:
            edges = [*edges, *inferred_edges]
            metrics, graph = analyze_graph(signals, edges, hyperparameters=hyperparameters)
    keep_ids = select_concepts_to_keep(signals, metrics, graph, thresholds=thresholds)
    edge_updates = upsert_weighted_edges(db, course_id, edges)
    concept_updates = write_concept_metrics(db, concepts, signals, metrics, keep_ids)
    return {
        "graph_algorithm_nodes": len(keep_ids),
        "graph_algorithm_edges": graph.number_of_edges(),
        "graph_algorithm_edge_updates": edge_updates,
        "graph_algorithm_concept_updates": concept_updates,
        "graph_algorithm_inferred_edges": len(inferred_edges),
        "graph_algorithm_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "graph_relation_completion_edges": completion_count,
        "graph_relation_completion_error": completion_error,
        "graph_quality_thresholds": thresholds.audit,
        "graph_hyperparameter_source": hyperparameters.source,
    }


async def enrich_course_graph_without_completion(db: Session, course_id: str) -> dict[str, Any]:
    return await enrich_course_graph(db, course_id, run_relation_completion=False)
