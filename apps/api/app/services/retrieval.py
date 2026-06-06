from __future__ import annotations

import asyncio
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.utils import source_type_from_path
from app.core.config import get_settings
from app.models import Chunk, Concept, ConceptRelation, Course, Document, DocumentVersion, GraphCommunitySummary, IngestionBatch, IngestionJob
from app.schemas import Citation, SearchFilters
from app.services.concept_graph import get_graph_payload
from app.services.cache_manager import get_cache_manager
from app.services.embeddings import ChatProvider, EmbeddingProvider, is_degraded_mode
from app.services.parsers import derive_chapter, is_invalid_chapter_label
from app.services.reranker import get_reranker
from app.services.runtime_settings import read_env_bool
from app.services.strategy_profiles import active_profile_json, get_active_profile_record, use_strategy_profile
from app.services.vector_store import VectorStore


STORAGE_ALLOWED_SUFFIXES = {
    ".pdf",
    ".ipynb",
    ".md",
    ".markdown",
    ".txt",
    ".docx",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".html",
    ".htm",
}
STORAGE_EXCLUDED_PARTS = {"output", "scripts", ".ipynb_checkpoints", "__pycache__"}
STORAGE_IGNORED_NAMES = {".ds_store"}
TERMINAL_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled"}
QUERY_TYPE_CONFIG = {
    "definition": {"alpha": 0.85, "recall_k": 60},
    "formula": {"alpha": 0.30, "recall_k": 80},
    "example": {"alpha": 0.70, "recall_k": 60},
    "comparison": {"alpha": 0.75, "recall_k": 80},
    "procedure": {"alpha": 0.75, "recall_k": 60},
    "default": {"alpha": 0.72, "recall_k": 64},
}
PRIMARY_SCORE_KEYS = ("dense", "lexical", "fused", "rerank", "lightweight_rerank", "term_overlap_ratio")
UNVERIFIED_GRAPH_SOURCES = {"semantic_sparse", "dijkstra_inferred"}
EVIDENCE_FIRST_MAX_ANCHORS = 4
EVIDENCE_FIRST_MAX_PATHS = 8


def should_include_storage_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() in STORAGE_IGNORED_NAMES or path.name.startswith("~$"):
        return False
    if path.suffix.lower() not in STORAGE_ALLOWED_SUFFIXES:
        return False
    return not any(part.lower() in STORAGE_EXCLUDED_PARTS for part in path.parts)


def collect_course_storage_paths(course: Course) -> list[Path]:
    root = get_settings().course_paths_for_name(course.name)["storage_root"]
    if not root.exists():
        return []
    return sorted((path for path in root.rglob("*") if should_include_storage_file(path)), key=lambda item: str(item).lower())


def score_chunk_bonus(chunk: Chunk, document: Document, query: str) -> float:
    kind = (chunk.metadata_json or {}).get("content_kind")
    title_text = f"{document.title}\n{chunk.section or ''}".lower()
    bonus = 0.0
    if kind in {"markdown", "text", "pdf_page", "slide", "doc_section"}:
        bonus += 1.1
    if kind == "code":
        bonus -= 1.8
    if kind == "output":
        bonus -= 0.8
    if query.lower() in title_text:
        bonus += 1.4
    if chunk.section and query.lower() in chunk.section.lower():
        bonus += 0.7
    return bonus


def tokenize_for_retrieval(text: str) -> list[str]:
    from app.services.chinese_text import tokenize_for_retrieval as _cn_tokenize

    return _cn_tokenize(text)


def classify_query_type(query: str) -> str:
    lower = query.lower()
    retrieval_strategy = active_profile_json().get("retrieval_strategy") or {}
    markers = retrieval_strategy.get("query_type_markers") if isinstance(retrieval_strategy, dict) else None

    def _markers(key: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
        values = markers.get(key) if isinstance(markers, dict) else None
        if isinstance(values, list):
            normalized = tuple(str(item).lower() for item in values if str(item).strip())
            return normalized or defaults
        return defaults

    if any(marker in lower for marker in _markers("definition", ("what is", "define", "definition", "meaning", "concept", "什么是", "定义", "概念"))):
        return "definition"
    if (
        any(marker in lower for marker in _markers("formula", ("formula", "theorem", "proof", "derive", "equation", "complexity", "o(", "公式", "定理", "证明")))
        or re.search(r"[=∑≤≥∞αβγλμ]|p\(|q\(|\\", query)
    ):
        return "formula"
    if any(marker in lower for marker in _markers("example", ("example", "instance", "case", "举例", "例子"))):
        return "example"
    if any(marker in lower for marker in _markers("comparison", ("compare", "versus", "vs", "difference", "relationship", "relate", "区别", "比较", "关系"))):
        return "comparison"
    if any(marker in lower for marker in _markers("procedure", ("algorithm", "procedure", "steps", "how to", "流程", "步骤", "算法", "如何"))):
        return "procedure"
    if any(marker in lower for marker in ("what is", "define", "definition", "meaning", "concept", "什么是", "定义", "概念")):
        return "definition"
    if (
        any(marker in lower for marker in ("formula", "theorem", "proof", "derive", "equation", "complexity", "o(", "公式", "定理", "证明"))
        or re.search(r"[=∑∫√λθπσμ]|p\(|q\(|\\", query)
    ):
        return "formula"
    if any(marker in lower for marker in ("example", "instance", "case", "举例", "例子")):
        return "example"
    if any(marker in lower for marker in ("compare", "versus", "vs", "difference", "relationship", "relate", "区别", "比较", "关系")):
        return "comparison"
    if any(marker in lower for marker in ("algorithm", "procedure", "steps", "how to", "流程", "步骤", "算法", "如何")):
        return "procedure"
    return "default"


def query_type_config(query: str) -> dict:
    settings = get_settings()
    query_type = classify_query_type(query)
    config = dict(QUERY_TYPE_CONFIG[query_type])
    if query_type == "formula":
        config["recall_k"] = settings.retrieval_recall_k_formula
    elif query_type == "default":
        config["recall_k"] = settings.retrieval_recall_k_default
    config["query_type"] = query_type
    return config


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def clone_for_fusion(item: dict) -> dict:
    clone = item.copy()
    clone["metadata"] = dict(item.get("metadata") or {})
    clone["metadata"]["scores"] = dict(clone["metadata"].get("scores") or {})
    clone["score"] = 0.0
    return clone


def default_model_audit() -> dict:
    settings = get_settings()
    return {
        "embedding_provider": "none",
        "embedding_model": settings.embedding_model,
        "embedding_external_called": False,
        "embedding_fallback_reason": None,
        "reranker_enabled": read_env_bool("RERANKER_ENABLED", settings.reranker_enabled),
        "reranker_called": False,
        "fallback_enabled": settings.enable_model_fallback,
        "degraded_mode": is_degraded_mode(),
        "vector_index_warning": None,
    }


def did_rerank(results: list[dict]) -> bool:
    return any(
        item.get("metadata", {}).get("scores", {}).get("rerank") is not None
        or item.get("metadata", {}).get("scores", {}).get("cross_encoder") is not None
        for item in results
    )


def attach_model_audit(results: list[dict], audit: dict) -> list[dict]:
    for item in results:
        metadata = item.setdefault("metadata", {})
        scores = metadata.setdefault("scores", {})
        for key in PRIMARY_SCORE_KEYS:
            scores.setdefault(key, None)
        metadata["model_audit"] = dict(audit)
    return results


def is_parent_chunk(chunk: Chunk) -> bool:
    return bool((chunk.metadata_json or {}).get("is_parent"))


def is_child_retrieval_candidate(chunk: Chunk, db: Session) -> bool:
    if not is_parent_chunk(chunk):
        return True
    # Parent chunks with no children are valid retrieval candidates themselves
    # (they are the finest-grained unit for that document segment).
    has_children = db.scalar(
        select(1).where(Chunk.parent_chunk_id == chunk.id).limit(1)
    ) is not None
    return not has_children


def expand_results_with_parent_context(db: Session, course_id: str, results: list[dict]) -> list[dict]:
    if not results:
        return results
    parent_ids = {
        str(item.get("metadata", {}).get("parent_chunk_id") or "")
        for item in results
        if item.get("metadata", {}).get("parent_chunk_id")
    }
    if not parent_ids:
        return results
    parents = {
        chunk.id: chunk
        for chunk in db.scalars(
            select(Chunk).where(
                Chunk.id.in_(parent_ids),
                Chunk.course_id == course_id,
                Chunk.is_active.is_(True),
            )
        ).all()
    }
    for item in results:
        metadata = item.setdefault("metadata", {})
        parent_id = metadata.get("parent_chunk_id")
        parent = parents.get(parent_id)
        if parent is None:
            continue
        metadata["parent_content"] = parent.content
        metadata["parent_snippet"] = parent.snippet
        metadata["parent_section"] = parent.section
        metadata["retrieval_granularity"] = "child_with_parent_context"
        item["child_content"] = item.get("content")
        item["content"] = parent.content
    return results


def build_search_payload(chunk: Chunk, document: Document, query: str, score: float, scores: dict | None = None) -> dict:
    citation = Citation(
        chunk_id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        source_path=document.source_path,
        chapter=chunk.chapter,
        section=chunk.section,
        page_number=chunk.page_number,
        snippet=chunk.snippet,
    )
    metadata = (chunk.metadata_json or {}) | {"chapter": chunk.chapter, "source_type": chunk.source_type}
    metadata["is_parent"] = is_parent_chunk(chunk)
    if chunk.parent_chunk_id:
        metadata["parent_chunk_id"] = str(chunk.parent_chunk_id)
        metadata["retrieval_granularity"] = "child"
    if scores:
        metadata["scores"] = scores
    return {
        "chunk_id": chunk.id,
        "snippet": chunk.snippet,
        "score": score,
        "citations": [citation.model_dump()],
        "metadata": metadata,
        "content": chunk.content,
        "child_content": chunk.content if chunk.parent_chunk_id else None,
        "document_title": document.title,
        "source_path": document.source_path,
        "chapter": chunk.chapter,
        "source_type": chunk.source_type,
    }


async def dense_search_chunks(db: Session, course_id: str, query: str, filters: SearchFilters, top_k: int, model_audit: dict | None = None) -> list[dict]:
    course = await asyncio.to_thread(db.get, Course, course_id)
    if course is None:
        return []
    embedder = EmbeddingProvider()
    cache = get_cache_manager()
    embedding_version = "contextual_enriched_v2"
    cached_vector = cache.get_embedding(course_id, query, embedding_version)
    if cached_vector is not None:
        embedding_result = type("obj", (object,), {"vectors": [cached_vector], "provider": "cache", "external_called": False, "fallback_reason": None})()
    else:
        embedding_result = await embedder.embed_texts_with_meta([query], text_type="query")
        if embedding_result.vectors and embedding_result.external_called:
            cache.set_embedding(course_id, query, embedding_version, embedding_result.vectors[0])
    if model_audit is not None:
        model_audit.update(
            {
                "embedding_provider": embedding_result.provider,
                "embedding_external_called": embedding_result.external_called,
                "embedding_fallback_reason": embedding_result.fallback_reason,
            }
        )
    vector_store = VectorStore(course_name=course.name)
    results = await vector_store.async_search(
        vector=embedding_result.vectors[0],
        limit=max(top_k * 3, top_k),
        filters={
            "course_id": course_id,
            "chapter": filters.chapter,
            "difficulty": filters.difficulty,
            "source_type": filters.source_type,
        },
    )
    payloads = []
    dense_scores: list[float] = []
    for result in results:
        chunk = await asyncio.to_thread(db.get, Chunk, result["id"])
        if chunk is None or chunk.course_id != course_id or not chunk.is_active or not await asyncio.to_thread(is_child_retrieval_candidate, chunk, db):
            continue
        document = await asyncio.to_thread(db.get, Document, chunk.document_id)
        if document is None or document.course_id != course_id:
            continue
        if filters.tags and not set(filters.tags).intersection(set(document.tags or [])):
            continue
        dense_score = float(result["score"])
        dense_scores.append(dense_score)
        score = dense_score + score_chunk_bonus(chunk, document, query)
        payloads.append(build_search_payload(chunk, document, query, score, {"dense": dense_score}))
    if payloads and dense_scores and max(abs(score) for score in dense_scores) <= 1e-12:
        if model_audit is not None:
            model_audit["vector_index_warning"] = "qdrant_returned_only_zero_scores"
        return []
    payloads.sort(key=lambda item: item["score"], reverse=True)
    return attach_model_audit(payloads[:top_k], model_audit) if model_audit is not None else payloads[:top_k]


async def search_chunks(db: Session, course_id: str, query: str, filters: SearchFilters, top_k: int) -> list[dict]:
    results, _audit = await hybrid_search_chunks_with_audit(db, course_id, query, filters, top_k)
    return results


async def search_chunks_with_audit(db: Session, course_id: str, query: str, filters: SearchFilters, top_k: int) -> tuple[list[dict], dict]:
    return await hybrid_search_chunks_with_audit(db, course_id, query, filters, top_k)


async def hybrid_search_chunks(db: Session, course_id: str, query: str, filters: SearchFilters, top_k: int) -> list[dict]:
    results, _audit = await hybrid_search_chunks_with_audit(db, course_id, query, filters, top_k)
    return results


async def hybrid_search_chunks_with_audit(db: Session, course_id: str, query: str, filters: SearchFilters, top_k: int) -> tuple[list[dict], dict]:
    settings = get_settings()
    config = query_type_config(query)
    recall_k = max(int(config["recall_k"]), top_k)
    dense_results: list[dict] = []
    model_audit = default_model_audit()
    if is_degraded_mode() and not settings.enable_model_fallback:
        raise RuntimeError("OPENAI_API_KEY is required for search because ENABLE_MODEL_FALLBACK is false")
    if not is_degraded_mode():
        try:
            dense_results = await dense_search_chunks(db, course_id, query, filters, recall_k, model_audit)
        except Exception:
            if not settings.enable_model_fallback:
                raise
            model_audit["embedding_fallback_reason"] = "dense_embedding_failed"
            dense_results = []
    lexical_results = lexical_search_chunks(db, course_id, query, filters, recall_k)
    if not dense_results:
        results = rerank_or_return(query, lexical_results, top_k) if lexical_results else []
        model_audit["reranker_called"] = did_rerank(results)
        results = expand_results_with_parent_context(db, course_id, results)
        return attach_model_audit(results, model_audit), model_audit
    if not lexical_results:
        results = rerank_or_return(query, dense_results, top_k)
        model_audit["reranker_called"] = did_rerank(results)
        results = expand_results_with_parent_context(db, course_id, results)
        return attach_model_audit(results, model_audit), model_audit

    candidates = weighted_score_fusion(
        dense_results,
        lexical_results,
        alpha=float(config["alpha"]),
        top_n=max(recall_k, top_k),
    )
    for item in candidates:
        item.setdefault("metadata", {}).setdefault("scores", {})["query_type"] = config["query_type"]
    results = rerank_or_return(query, candidates, top_k)
    model_audit["reranker_called"] = did_rerank(results)
    results = expand_results_with_parent_context(db, course_id, results)
    return attach_model_audit(results, model_audit), model_audit


def lightweight_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """轻量精排：零外部模型，纯规则 + 统计信号。"""
    if not candidates:
        return []

    query_terms = set(tokenize_for_retrieval(query))
    query_len = len(query_terms) or 1

    scored = []
    for item in candidates:
        haystack = f"{item.get('document_title', '')} {item.get('snippet', '')} {item.get('content', '')}"
        doc_terms = set(tokenize_for_retrieval(haystack))
        overlap = query_terms.intersection(doc_terms)
        overlap_ratio = len(overlap) / query_len

        fused_score = _result_score(item)

        query_type = item.get("metadata", {}).get("scores", {}).get("query_type", "default")
        alpha = 0.65 if query_type in ("definition", "formula") else 0.75

        final_score = alpha * fused_score + (1.0 - alpha) * overlap_ratio

        scores = item.setdefault("metadata", {}).setdefault("scores", {})
        scores["lightweight_rerank"] = round(final_score, 4)
        scores["term_overlap_ratio"] = round(overlap_ratio, 4)
        scores["rerank"] = round(final_score, 4)
        item["score"] = final_score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_or_return(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    settings = get_settings()
    if not settings.reranker_enabled:
        return lightweight_rerank(query, candidates, top_k)
    flag = read_env_bool("RERANKER_ENABLED", settings.reranker_enabled)
    if flag:
        reranker = get_reranker()
        return reranker.rerank(query, candidates, top_k)
    return lightweight_rerank(query, candidates, top_k)


def weighted_score_fusion(dense_results: list[dict], lexical_results: list[dict], alpha: float, top_n: int) -> list[dict]:
    fused: dict[str, dict] = {}
    dense_values = [float(item.get("metadata", {}).get("scores", {}).get("dense", item["score"])) for item in dense_results]
    lexical_values = [float(item.get("metadata", {}).get("scores", {}).get("bm25", item["score"])) for item in lexical_results]
    dense_norm = normalize_scores(dense_values)
    lexical_norm = normalize_scores(lexical_values)
    for item, normalized_score in zip(dense_results, dense_norm):
        chunk_id = item["chunk_id"]
        fused.setdefault(chunk_id, clone_for_fusion(item))
        scores = fused[chunk_id].setdefault("metadata", {}).setdefault("scores", {})
        scores["dense"] = item.get("metadata", {}).get("scores", {}).get("dense", item["score"])
        scores["dense_norm"] = normalized_score
        scores["fusion_alpha"] = alpha
        fused[chunk_id]["score"] = float(fused[chunk_id].get("score", 0.0)) + (alpha * normalized_score)
    for item, normalized_score in zip(lexical_results, lexical_norm):
        chunk_id = item["chunk_id"]
        fused.setdefault(chunk_id, clone_for_fusion(item))
        scores = fused[chunk_id].setdefault("metadata", {}).setdefault("scores", {})
        lexical_score = item.get("metadata", {}).get("scores", {}).get("bm25", item["score"])
        scores["bm25"] = lexical_score
        scores["lexical"] = item.get("metadata", {}).get("scores", {}).get("lexical", item["score"])
        scores["bm25_norm"] = normalized_score
        scores["fusion_alpha"] = alpha
        fused[chunk_id]["score"] = float(fused[chunk_id].get("score", 0.0)) + ((1.0 - alpha) * normalized_score)

    for item in fused.values():
        scores = item.setdefault("metadata", {}).setdefault("scores", {})
        scores["fused"] = float(item["score"])
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_n]


def is_verified_graph_relation(relation: ConceptRelation) -> bool:
    metadata = relation.metadata_json or {}
    if metadata.get("candidate_only"):
        return False
    if relation.relation_source in UNVERIFIED_GRAPH_SOURCES:
        return False
    if relation.relation_type == "related_to" and (
        not relation.is_validated or float(relation.weight or 0.0) < 0.75
    ):
        return False
    if not relation.evidence_chunk_id:
        return False
    endpoint_match = bool(metadata.get("evidence_source_match") and metadata.get("evidence_target_match"))
    return endpoint_match or bool(relation.is_validated)


def _result_score(item: dict) -> float:
    scores = item.get("metadata", {}).get("scores", {})
    for key in ("rerank", "cross_encoder", "fused", "dense", "bm25"):
        value = scores.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    try:
        return float(item.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mark_result(item: dict, **metadata: Any) -> dict:
    result = dict(item)
    result["metadata"] = dict(item.get("metadata") or {})
    result["metadata"]["scores"] = dict(result["metadata"].get("scores") or {})
    result["metadata"].update({key: value for key, value in metadata.items() if value is not None})
    return result


def select_evidence_anchors(db: Session, course_id: str, base_results: list[dict], *, max_anchors: int = EVIDENCE_FIRST_MAX_ANCHORS) -> tuple[list[dict], dict]:
    if not base_results:
        return [], {"anchor_count": 0, "candidate_count": 0}
    chunk_ids = [str(item["chunk_id"]) for item in base_results if item.get("chunk_id")]
    relations = db.scalars(
        select(ConceptRelation).where(
            ConceptRelation.course_id == course_id,
            ConceptRelation.evidence_chunk_id.in_(chunk_ids),
        )
    ).all() if chunk_ids else []
    verified_by_chunk: dict[str, list[ConceptRelation]] = defaultdict(list)
    for relation in relations:
        if is_verified_graph_relation(relation):
            verified_by_chunk[str(relation.evidence_chunk_id)].append(relation)

    scored: list[tuple[float, dict]] = []
    for item in base_results:
        chunk_id = str(item.get("chunk_id") or "")
        metadata = item.get("metadata") or {}
        quality_action = metadata.get("quality_action")
        routes = metadata.get("route_eligibility") or {}
        relations_for_chunk = verified_by_chunk.get(chunk_id, [])
        base_score = _result_score(item)
        quality_bonus = 0.0
        if routes.get("graph_extraction") or quality_action == "graph_candidate":
            quality_bonus = 0.08
        elif routes.get("retrieval") or quality_action == "retrieval_candidate":
            quality_bonus = 0.05
        elif routes.get("evidence_only") or quality_action == "evidence_only":
            quality_bonus = 0.02
        relation_bonus = min(0.18, 0.06 * len(relations_for_chunk))
        anchor_score = base_score + quality_bonus + relation_bonus
        if base_score <= 0 and not relations_for_chunk:
            continue
        concept_ids = sorted(
            {
                concept_id
                for relation in relations_for_chunk
                for concept_id in (relation.source_concept_id, relation.target_concept_id)
                if concept_id
            }
        )
        scored.append(
            (
                anchor_score,
                _mark_result(
                    item,
                    retrieval_stage="evidence_anchor_selector",
                    evidence_role="base_anchor",
                    anchor_score=round(anchor_score, 4),
                    anchor_concept_ids=concept_ids,
                    graph_verified=bool(relations_for_chunk),
                    evidence_support_reason="base_retrieval_high_score" if not relations_for_chunk else "base_retrieval_verified_relation",
                ),
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    anchors = [item for _score, item in scored[:max_anchors]]
    return anchors, {
        "anchor_count": len(anchors),
        "candidate_count": len(base_results),
        "verified_anchor_relations": sum(len(verified_by_chunk.get(str(item.get("chunk_id")), [])) for item in anchors),
    }


def _load_verified_relations(db: Session, course_id: str, concept_ids: set[str], depth: int) -> list[ConceptRelation]:
    if not concept_ids:
        return []
    frontier = set(concept_ids)
    seen = set(concept_ids)
    relations: list[ConceptRelation] = []
    for _ in range(max(1, depth)):
        batch = db.scalars(
            select(ConceptRelation).where(
                ConceptRelation.course_id == course_id,
                or_(
                    ConceptRelation.source_concept_id.in_(frontier),
                    ConceptRelation.target_concept_id.in_(frontier),
                ),
            )
        ).all()
        next_frontier: set[str] = set()
        for relation in batch:
            if not is_verified_graph_relation(relation):
                continue
            relations.append(relation)
            for concept_id in (relation.source_concept_id, relation.target_concept_id):
                if concept_id and concept_id not in seen:
                    seen.add(concept_id)
                    next_frontier.add(concept_id)
        if not next_frontier:
            break
        frontier = next_frontier
    unique: dict[str, ConceptRelation] = {}
    for relation in relations:
        unique[relation.id] = relation
    return list(unique.values())


def plan_evidence_chains(
    db: Session,
    course_id: str,
    anchors: list[dict],
    *,
    query_type: str,
    community_ids: list[int] | None = None,
) -> tuple[list[dict], dict]:
    anchor_concepts = {
        concept_id
        for anchor in anchors
        for concept_id in (anchor.get("metadata", {}).get("anchor_concept_ids") or [])
        if isinstance(concept_id, str)
    }
    max_depth = 3 if query_type in {"comparison", "procedure", "formula"} else 2
    relations = _load_verified_relations(db, course_id, anchor_concepts, max_depth) if anchor_concepts else []
    adjacency: dict[str, list[ConceptRelation]] = defaultdict(list)
    for relation in relations:
        adjacency[relation.source_concept_id].append(relation)
        if relation.target_concept_id:
            adjacency[relation.target_concept_id].append(relation)

    paths: list[dict] = []
    seen_relation_paths: set[tuple[str, ...]] = set()
    for seed in sorted(anchor_concepts):
        queue: list[tuple[str, list[ConceptRelation]]] = [(seed, [])]
        visited = {seed}
        while queue and len(paths) < EVIDENCE_FIRST_MAX_PATHS:
            current, path_relations = queue.pop(0)
            if len(path_relations) >= max_depth:
                continue
            for relation in sorted(adjacency.get(current, []), key=lambda item: float(item.weight or item.confidence or 0.0), reverse=True):
                other = relation.target_concept_id if relation.source_concept_id == current else relation.source_concept_id
                if not other or other in visited:
                    continue
                next_path = [*path_relations, relation]
                visited.add(other)
                queue.append((other, next_path))
                if next_path:
                    relation_key = tuple(sorted(item.id for item in next_path))
                    if relation_key in seen_relation_paths:
                        continue
                    seen_relation_paths.add(relation_key)
                    path_id = f"path-{len(paths) + 1}"
                    paths.append(
                        {
                            "path_id": path_id,
                            "concept_ids": [seed, other],
                            "relation_ids": [item.id for item in next_path],
                            "evidence_chunk_ids": [str(item.evidence_chunk_id) for item in next_path if item.evidence_chunk_id],
                            "relation_types": [item.relation_type for item in next_path],
                            "score": round(sum(float(item.weight or item.confidence or 0.0) for item in next_path) / len(next_path), 4),
                        }
                    )
                if len(paths) >= EVIDENCE_FIRST_MAX_PATHS:
                    break

    community_ids = [int(item) for item in community_ids or [] if item is not None]
    community_summaries = []
    if community_ids:
        community_summaries = db.scalars(
            select(GraphCommunitySummary).where(
                GraphCommunitySummary.course_id == course_id,
                GraphCommunitySummary.algorithm == "louvain",
                GraphCommunitySummary.community_id.in_(community_ids),
                GraphCommunitySummary.is_active.is_(True),
            )
        ).all()
    for summary in community_summaries:
        paths.append(
            {
                "path_id": f"community-{summary.community_id}",
                "community_id": summary.community_id,
                "relation_ids": [],
                "evidence_chunk_ids": list(summary.representative_chunk_ids or [])[:3],
                "relation_types": ["community_summary"],
                "score": 0.55,
            }
        )

    return paths[:EVIDENCE_FIRST_MAX_PATHS], {
        "planned_paths": min(len(paths), EVIDENCE_FIRST_MAX_PATHS),
        "verified_edges": len(relations),
        "community_summaries": len(community_summaries),
        "max_depth": max_depth,
        "skipped_reason": None if paths else "no_anchor_concepts_or_communities",
    }


def controlled_graph_enhancement(
    db: Session,
    course_id: str,
    query: str,
    filters: SearchFilters,
    base_chunk_ids: set[str],
    paths: list[dict],
) -> tuple[list[dict], dict]:
    evidence_by_chunk: dict[str, dict] = {}
    for path in paths:
        for chunk_id in path.get("evidence_chunk_ids", []):
            if not chunk_id or chunk_id in base_chunk_ids:
                continue
            evidence_by_chunk.setdefault(str(chunk_id), path)
    if not evidence_by_chunk:
        return [], {"graph_enhanced_chunks": 0, "path_evidence_chunks": 0}

    chunks = db.scalars(
        select(Chunk).where(
            Chunk.id.in_(list(evidence_by_chunk)),
            Chunk.course_id == course_id,
            Chunk.is_active.is_(True),
        )
    ).all()
    enhanced: list[dict] = []
    for chunk in chunks:
        document = db.get(Document, chunk.document_id)
        if document is None or document.course_id != course_id or not document.is_active:
            continue
        if filters.chapter and chunk.chapter != filters.chapter:
            continue
        if filters.source_type and chunk.source_type != filters.source_type:
            continue
        if filters.tags and not set(filters.tags).intersection(set(document.tags or [])):
            continue
        path = evidence_by_chunk[str(chunk.id)]
        item = build_search_payload(
            chunk,
            document,
            query,
            float(path.get("score", 0.0)),
            {"graph_path": float(path.get("score", 0.0))},
        )
        enhanced.append(
            _mark_result(
                item,
                retrieval_stage="controlled_graph_enhancer",
                evidence_role="path_edge" if path.get("relation_ids") else "community_summary",
                path_id=path.get("path_id"),
                relation_id=(path.get("relation_ids") or [None])[0],
                relation_type=(path.get("relation_types") or [None])[0],
                graph_verified=bool(path.get("relation_ids")),
                community_id=path.get("community_id"),
                evidence_support_reason="verified_graph_path" if path.get("relation_ids") else "community_summary_representative_chunk",
            )
        )
    return expand_results_with_parent_context(db, course_id, enhanced), {
        "graph_enhanced_chunks": len(enhanced),
        "path_evidence_chunks": len(evidence_by_chunk),
    }


def assemble_evidence_documents(base_results: list[dict], anchors: list[dict], graph_results: list[dict], top_k: int) -> tuple[list[dict], dict]:
    anchor_by_id = {str(item["chunk_id"]): item for item in anchors}
    merged: dict[str, dict] = {}
    for item in base_results:
        chunk_id = str(item["chunk_id"])
        role = "base_anchor" if chunk_id in anchor_by_id else "base_candidate"
        source_item = anchor_by_id.get(chunk_id, item)
        merged[chunk_id] = _mark_result(
            source_item,
            retrieval_stage="evidence_assembler",
            evidence_role=source_item.get("metadata", {}).get("evidence_role") or role,
            graph_verified=bool(source_item.get("metadata", {}).get("graph_verified")),
            evidence_support_reason=source_item.get("metadata", {}).get("evidence_support_reason") or "base_retrieval",
        )
    for item in graph_results:
        chunk_id = str(item["chunk_id"])
        current = merged.get(chunk_id)
        if current is None or _result_score(item) > _result_score(current):
            merged[chunk_id] = _mark_result(item, retrieval_stage="evidence_assembler")
    documents = sorted(merged.values(), key=_result_score, reverse=True)[: max(top_k * 2, top_k)]
    return documents, {
        "assembled_documents": len(documents),
        "base_documents": len(base_results),
        "anchor_documents": len(anchors),
        "graph_documents": len(graph_results),
    }


async def evidence_first_search_chunks_with_audit(
    db: Session,
    course_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int,
    route: str = "retrieve_notes",
    community_ids: list[int] | None = None,
) -> tuple[list[dict], dict]:
    config = query_type_config(query)
    recall_k = max(top_k * 3, int(config["recall_k"]))
    base_results, model_audit = await hybrid_search_chunks_with_audit(db, course_id, query, filters, recall_k)
    base_results = [
        _mark_result(item, retrieval_stage="base_retrieval", evidence_role="base_candidate", graph_verified=False)
        for item in base_results
    ]
    anchors, anchor_audit = select_evidence_anchors(db, course_id, base_results)
    should_plan_graph = route == "multi_hop_research" or config["query_type"] in {"comparison", "procedure", "formula"} or bool(community_ids)
    paths: list[dict] = []
    path_audit = {"planned_paths": 0, "verified_edges": 0, "skipped_reason": "simple_query"}
    graph_results: list[dict] = []
    graph_audit = {"graph_enhanced_chunks": 0, "path_evidence_chunks": 0}
    if should_plan_graph and anchors:
        paths, path_audit = plan_evidence_chains(
            db,
            course_id,
            anchors,
            query_type=config["query_type"],
            community_ids=community_ids,
        )
        graph_results, graph_audit = controlled_graph_enhancement(
            db,
            course_id,
            query,
            filters,
            {str(item["chunk_id"]) for item in base_results},
            paths,
        )
    documents, assembly_audit = assemble_evidence_documents(base_results, anchors, graph_results, top_k)
    evidence_audit = {
        **model_audit,
        "retrieval_pipeline": "evidence_first_v1",
        "query_type": config["query_type"],
        "route": route,
        "base_candidate_count": len(base_results),
        "anchors": anchor_audit,
        "paths": path_audit,
        "graph": graph_audit,
        "assembly": assembly_audit,
        "community_ids": community_ids or [],
    }
    for item in documents:
        item.setdefault("metadata", {})["evidence_first_audit"] = {
            "pipeline": "evidence_first_v1",
            "route": route,
            "query_type": config["query_type"],
        }
        item.setdefault("metadata", {}).setdefault("model_audit", dict(model_audit))
    return documents, evidence_audit


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_search_chunks(db: Session, course_id: str, query: str, filters: SearchFilters, top_k: int) -> list[dict]:
    query_terms = tokenize_for_retrieval(query)
    rows = db.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.course_id == course_id, Chunk.is_active.is_(True), Document.is_active.is_(True))
        .order_by(Chunk.created_at.desc())
    ).all()
    corpus: list[list[str]] = []
    chunk_documents: list[tuple[Chunk, Document]] = []
    for chunk, document in rows:
        if not is_child_retrieval_candidate(chunk, db):
            continue
        if filters.chapter and chunk.chapter != filters.chapter:
            continue
        if filters.source_type and chunk.source_type != filters.source_type:
            continue
        if filters.tags and not set(filters.tags).intersection(set(document.tags or [])):
            continue
        corpus.append(tokenize_for_retrieval(f"{document.title}\n{chunk.section or ''}\n{chunk.content}"))
        chunk_documents.append((chunk, document))
    if not query_terms or not corpus:
        return []
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query_terms)
    scored: list[dict] = []
    for idx, (chunk, document) in enumerate(chunk_documents):
        bm25_score = float(bm25_scores[idx])
        overlap = sum(corpus[idx].count(term) for term in query_terms)
        if bm25_score <= 0 and overlap <= 0:
            continue
        score = bm25_score + (0.05 * overlap) + score_chunk_bonus(chunk, document, query)
        scored.append(build_search_payload(chunk, document, query, score, {"bm25": bm25_score, "lexical_overlap": overlap, "lexical": score}))
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


async def answer_question(db: Session, course_id: str, question: str, filters: SearchFilters, top_k: int, history: list[dict]) -> dict:
    strategy_profile = get_active_profile_record(db, course_id)
    with use_strategy_profile(strategy_profile.profile_json):
        results = await search_chunks(db, course_id, question, filters, top_k)
        chat = ChatProvider()
        answer = await chat.answer_question(question, results, history)
    return {
        "answer": answer,
        "citations": [citation for result in results for citation in result["citations"]],
        "used_chunks": results,
        "degraded_mode": is_degraded_mode(),
    }


def get_dashboard_snapshot(db: Session, course_id: str) -> dict:
    course = db.get(Course, course_id)
    if course is None:
        return {
            "course": {
                "id": "empty",
                "name": "Course Workspace",
                "description": None,
                "source_root": "",
                "storage_root": "",
                "document_count": 0,
                "concept_count": 0,
                "current_chunk_version": 0,
                "has_parsed_chunks": False,
                "can_full_reparse": False,
                "degraded_mode": is_degraded_mode(),
                "active_profile_id": None,
                "active_profile_name": None,
                "active_profile_hash": None,
            },
            "tree": [],
            "graph": {"nodes": [], "edges": [], "focus_chapter": None},
            "batch_status": None,
            "ingested_document_count": 0,
            "chunk_count": 0,
            "graph_eligible_chunk_count": 0,
            "graph_relation_count": 0,
            "coverage_by_source_type": {},
            "degraded_mode": is_degraded_mode(),
        }

    documents = db.scalars(select(Document).where(Document.course_id == course.id, Document.is_active.is_(True))).all()
    file_items = list_course_files(db, course.id)
    concepts = db.scalars(select(Concept).where(Concept.course_id == course.id)).all()
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course.id)).all()
    chunks = db.scalars(select(Chunk).where(Chunk.course_id == course.id, Chunk.is_active.is_(True))).all()
    chunk_count = len(chunks)
    current_chunk_version = max([chunk.chunk_version or 0 for chunk in chunks] + [course.current_chunk_version or 0])
    if (course.current_chunk_version or 0) < current_chunk_version:
        course.current_chunk_version = current_chunk_version
        db.flush()
    graph_eligible_chunk_count = sum(
        1
        for chunk in chunks
        if ((chunk.metadata_json or {}).get("route_eligibility") or {}).get("graph_extraction") is True
    )
    batches = db.scalars(select(IngestionBatch).where(IngestionBatch.course_id == course.id).order_by(IngestionBatch.created_at.desc())).all()

    chapter_map: dict[str, list[dict]] = defaultdict(list)
    source_coverage = Counter()
    for item in file_items:
        chapter = item.get("chapter") or "General"
        chapter_map[chapter].append(item)
        source_coverage[item.get("source_type") or "unknown"] += 1

    tree = [
        {
            "id": f"chapter:{chapter}",
            "title": chapter,
            "type": "chapter",
            "children": [
                {"id": item["document_id"] or item["id"], "title": item["title"], "type": "document", "children": []}
                for item in sorted(entries, key=lambda item: item["title"])
            ],
        }
        for chapter, entries in sorted(chapter_map.items())
    ]
    latest_batch = next((batch for batch in batches if batch.status not in TERMINAL_BATCH_STATES), None)
    graph_payload = get_graph_payload(db, course.id, graph_type="semantic")
    active_profile = get_active_profile_record(db, course.id)
    return {
        "course": {
            "id": course.id,
            "name": course.name,
            "description": course.description,
            "source_root": str(get_settings().course_paths_for_name(course.name)["storage_root"]),
            "storage_root": str(get_settings().course_paths_for_name(course.name)["storage_root"]),
            "document_count": len(file_items),
            "concept_count": len(concepts),
            "current_chunk_version": current_chunk_version,
            "has_parsed_chunks": chunk_count > 0,
            "can_full_reparse": chunk_count > 0,
            "degraded_mode": is_degraded_mode(),
            "active_profile_id": active_profile.id,
            "active_profile_name": active_profile.name,
            "active_profile_hash": active_profile.profile_hash,
        },
        "tree": tree,
        "graph": graph_payload,
        "batch_status": None
        if latest_batch is None
        else {
            "batch_id": latest_batch.id,
            "state": latest_batch.status,
            "trigger_source": latest_batch.trigger_source,
            "source_root": latest_batch.source_root,
            "total_files": latest_batch.total_files,
            "processed_files": latest_batch.processed_files,
            "success_count": latest_batch.success_count,
            "failure_count": latest_batch.failure_count,
            "skipped_count": latest_batch.skipped_count,
            "coverage_by_source_type": (latest_batch.stats or {}).get("coverage_by_source_type", {}),
            "errors": (latest_batch.stats or {}).get("errors", []),
            "graph_stats": {
                key: value
                for key, value in (latest_batch.stats or {}).items()
                if key.startswith("graph_") or key in {"concepts", "relations"}
            },
            "started_at": latest_batch.started_at,
            "completed_at": latest_batch.completed_at,
        },
        "ingested_document_count": len(file_items),
        "chunk_count": chunk_count,
        "graph_eligible_chunk_count": graph_eligible_chunk_count,
        "graph_relation_count": len(relations),
        "coverage_by_source_type": dict(source_coverage),
        "degraded_mode": is_degraded_mode(),
    }


ACTIVE_FILE_STATES = {"parsing", "chunking", "embedding", "extracting_graph", "processing"}


def file_status_from_job(job: IngestionJob | None, has_parsed_chunks: bool) -> str:
    if job is None:
        return "parsed" if has_parsed_chunks else "pending"
    if job.status in ACTIVE_FILE_STATES:
        return "parsing"
    if job.status == "cancel_requested":
        return "parsing"
    if job.status == "cancelled":
        return "pending"
    if job.status == "queued":
        if (job.stats or {}).get("force_reparse"):
            return "pending"
        return "parsed" if has_parsed_chunks else "pending"
    if job.status == "failed":
        return "failed"
    if job.status == "skipped":
        return "parsed" if has_parsed_chunks else "skipped"
    if job.status == "completed":
        return "parsed" if has_parsed_chunks else "pending"
    return "parsed" if has_parsed_chunks else "pending"


def list_course_files(db: Session, course_id: str) -> list[dict]:
    course = db.get(Course, course_id)
    documents = db.scalars(select(Document).where(Document.course_id == course_id, Document.is_active.is_(True))).all()
    storage_root = get_settings().course_paths_for_name(course.name)["storage_root"] if course is not None else None
    storage_paths = {str(path) for path in collect_course_storage_paths(course)} if course is not None else set()
    document_versions = db.scalars(
        select(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.course_id == course_id, Document.is_active.is_(True), DocumentVersion.is_active.is_(True))
    ).all()
    documents_by_id = {document.id: document for document in documents}
    documents_by_storage_path = {
        version.storage_path: documents_by_id[version.document_id]
        for version in document_versions
        if version.document_id in documents_by_id and version.storage_path
    }
    jobs = db.scalars(select(IngestionJob).where(IngestionJob.course_id == course_id).order_by(IngestionJob.updated_at.desc())).all()
    latest_jobs: dict[str, IngestionJob] = {}
    removed_paths: set[str] = set()
    for job in jobs:
        is_removed = (job.error_message or "").startswith("Removed by user") or (job.trigger_source == "remove" and (job.stats or {}).get("removed"))
        if is_removed:
            if job.source_path:
                removed_paths.add(job.source_path)
            continue
        if job.source_path and job.source_path not in latest_jobs:
            latest_jobs[job.source_path] = job

    items: dict[str, dict] = {}
    if course is not None:
        for path in sorted((Path(path_string) for path_string in storage_paths), key=lambda item: str(item).lower()):
            path_string = str(path)
            if path_string in removed_paths:
                continue
            if path_string in items:
                continue
            job = latest_jobs.get(path_string)
            document = documents_by_storage_path.get(path_string)
            chunk_count = db.query(Chunk).filter(Chunk.document_id == document.id, Chunk.is_active.is_(True)).count() if document else 0
            chunk_version = (
                db.scalar(select(Chunk.chunk_version).where(Chunk.document_id == document.id, Chunk.is_active.is_(True)).order_by(Chunk.chunk_version.desc()))
                if document
                else None
            )
            items[path_string] = {
                "id": document.id if document else path_string,
                "document_id": document.id if document else None,
                "title": document.title if document else path.stem or path.name,
                "source_path": path_string,
                "source_type": document.source_type if document else source_type_from_path(path_string),
                "chapter": document.tags[0]
                if document and document.tags and not is_invalid_chapter_label(document.tags[0], course_name=course.name if course else None)
                else derive_chapter(path, course_name=course.name if course else None),
                "status": file_status_from_job(job, has_parsed_chunks=chunk_count > 0),
                "job_state": job.status if job else None,
                "batch_id": job.batch_id if job else None,
                "error": job.error_message if job and job.status == "failed" else None,
                "chunk_count": chunk_count,
                "chunk_version": chunk_version,
                "updated_at": document.updated_at if document else job.updated_at if job else None,
            }

    for path, job in latest_jobs.items():
        if path in removed_paths:
            continue
        if path in items:
            continue
        if storage_root is not None:
            continue
        items[path] = {
            "id": job.id,
            "document_id": job.document_id,
            "title": Path(path).stem or Path(path).name,
            "source_path": path,
            "source_type": source_type_from_path(path),
            "chapter": None,
            "status": file_status_from_job(job, has_parsed_chunks=False),
            "job_state": job.status,
            "batch_id": job.batch_id,
            "error": job.error_message,
            "chunk_count": 0,
            "chunk_version": None,
            "updated_at": job.updated_at,
        }

    latest_batch = db.scalar(select(IngestionBatch).where(IngestionBatch.course_id == course_id).order_by(IngestionBatch.created_at.desc()))
    uploaded_paths = (latest_batch.stats or {}).get("uploaded_files", []) if latest_batch else []
    for path in uploaded_paths:
        if path in removed_paths:
            continue
        if path in items:
            continue
        if storage_root is not None:
            continue
        items[path] = {
            "id": path,
            "document_id": None,
            "title": Path(path).stem or Path(path).name,
            "source_path": path,
            "source_type": source_type_from_path(path),
            "chapter": None,
            "status": "pending",
            "job_state": None,
            "batch_id": latest_batch.id,
            "error": None,
            "chunk_count": 0,
            "chunk_version": None,
            "updated_at": latest_batch.created_at,
        }

    status_rank = {"parsing": 0, "pending": 1, "failed": 2, "parsed": 3, "skipped": 4}
    return sorted(items.values(), key=lambda item: (status_rank.get(item["status"], 9), item["title"].lower()))


def get_job_status(db: Session, job_id: str) -> dict | None:
    job = db.get(IngestionJob, job_id)
    if job is None:
        return None
    return {
        "job_id": job.id,
        "state": job.status,
        "error": job.error_message,
        "document_id": job.document_id,
        "source_path": job.source_path,
        "batch_id": job.batch_id,
        "stats": job.stats,
    }
