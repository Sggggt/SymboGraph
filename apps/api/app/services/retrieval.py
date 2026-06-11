from __future__ import annotations

import asyncio
import copy
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.utils import source_type_from_path
from app.core.config import get_settings
from app.models import (
    ActiveChunk,
    CommunityState,
    EvidenceAtom,
    EvidenceEdge,
    EvidenceGraphState,
    KnowledgeBase,
    Document,
    DocumentVersion,
    IngestionBatch,
    IngestionJob,
    PolicyState,
    RetrievalTrace,
    SignalEdge,
    SignalNode,
    SignalState,
)
KnowledgeBaseModel = KnowledgeBase
from app.schemas import Citation, SearchFilters
from app.services.evidence_graph_payload import get_graph_payload
from app.services.cache_manager import get_cache_manager
from app.services.embeddings import ChatProvider, EmbeddingProvider, is_degraded_mode
from app.services.evidence_graph import active_chunk_scope_hash, stable_hash
from app.services.evidence_signal_projection import load_active_signal_state
from app.services.parsers import derive_partition, is_invalid_partition_label
from app.services.reranker import get_reranker
from app.services.runtime_settings import current_runtime_settings_version, read_env_bool
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
PRIMARY_SCORE_KEYS = (
    "dense",
    "lexical",
    "fused",
    "graph_path",
    "signal_projection",
    "community",
    "uncertainty",
    "unified",
    "rerank",
    "lightweight_rerank",
    "term_overlap_ratio",
)
EVIDENCE_FIRST_MAX_ANCHORS = 4
EVIDENCE_FIRST_MAX_PATHS = 8
UNIFIED_SCORE_PROTOCOL_VERSION = "retrieval_unified_score_v1"
UNIFIED_SCORE_WEIGHTS = {
    "definition": {"dense": 0.42, "lexical": 0.24, "graph": 0.16, "community": 0.12, "uncertainty": 0.06},
    "formula": {"dense": 0.28, "lexical": 0.36, "graph": 0.18, "community": 0.10, "uncertainty": 0.08},
    "example": {"dense": 0.38, "lexical": 0.28, "graph": 0.14, "community": 0.12, "uncertainty": 0.08},
    "comparison": {"dense": 0.34, "lexical": 0.20, "graph": 0.24, "community": 0.16, "uncertainty": 0.06},
    "procedure": {"dense": 0.34, "lexical": 0.24, "graph": 0.22, "community": 0.12, "uncertainty": 0.08},
    "default": {"dense": 0.40, "lexical": 0.26, "graph": 0.16, "community": 0.12, "uncertainty": 0.06},
}


def should_include_storage_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() in STORAGE_IGNORED_NAMES or path.name.startswith("~$"):
        return False
    if path.suffix.lower() not in STORAGE_ALLOWED_SUFFIXES:
        return False
    return not any(part.lower() in STORAGE_EXCLUDED_PARTS for part in path.parts)


def collect_knowledge_base_storage_paths(KnowledgeBase: KnowledgeBase) -> list[Path]:
    root = get_settings().knowledge_base_paths_for_name(KnowledgeBase.name)["storage_root"]
    if not root.exists():
        return []
    return sorted((path for path in root.rglob("*") if should_include_storage_file(path)), key=lambda item: str(item).lower())


def score_chunk_bonus(chunk: ActiveChunk, document: Document, query: str) -> float:
    metadata = chunk.metadata_json or {}
    kind = metadata.get("content_kind")
    section = str(metadata.get("section") or "")
    title_text = f"{document.title}\n{section}".lower()
    bonus = 0.0
    if kind in {"markdown", "text", "pdf_page", "slide", "doc_section"}:
        bonus += 1.1
    if kind == "code":
        bonus -= 1.8
    if kind == "output":
        bonus -= 0.8
    if query.lower() in title_text:
        bonus += 1.4
    if section and query.lower() in section.lower():
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


def _score_value(scores: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = scores.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _clamp_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def unified_score_weights(query_type: str) -> dict[str, float]:
    return dict(UNIFIED_SCORE_WEIGHTS.get(query_type) or UNIFIED_SCORE_WEIGHTS["default"])


def _community_score(item: dict) -> float:
    metadata = item.get("metadata") or {}
    if metadata.get("community_id") is not None:
        return 1.0
    community_ids = metadata.get("community_ids") or []
    if community_ids:
        return min(1.0, 0.45 + 0.1 * len(community_ids))
    evidence_role = str(metadata.get("evidence_role") or "")
    if "community" in evidence_role:
        return 0.65
    return 0.0


def _uncertainty_score(scores: dict[str, Any], metadata: dict[str, Any]) -> float:
    dense = _score_value(scores, "dense_norm")
    lexical = _score_value(scores, "bm25_norm")
    graph = _score_value(scores, "graph_path", "signal_projection")
    channels = [value for value in (dense, lexical, graph) if value is not None]
    if not channels:
        return 1.0
    disagreement = 0.0
    if dense is not None and lexical is not None:
        disagreement = abs(dense - lexical)
    sparse_penalty = max(0.0, (3 - len(channels)) / 3)
    trace_penalty = 0.0 if metadata.get("evidence_atom_ids") or metadata.get("graph_verified") else 0.25
    return max(0.0, min(1.0, 0.5 * disagreement + 0.35 * sparse_penalty + trace_penalty))


def compute_unified_score(item: dict, *, query_type: str) -> tuple[float, dict[str, Any]]:
    metadata = item.setdefault("metadata", {})
    scores = metadata.setdefault("scores", {})
    weights = unified_score_weights(query_type)
    dense_component = _score_value(scores, "dense_norm", "dense")
    lexical_component = _score_value(scores, "bm25_norm", "lexical", "bm25")
    graph_component = _score_value(scores, "graph_path", "signal_projection")
    community_component = _community_score(item)
    uncertainty = _uncertainty_score(scores, metadata)
    unified = (
        weights["dense"] * _clamp_score(dense_component)
        + weights["lexical"] * _clamp_score(lexical_component)
        + weights["graph"] * _clamp_score(graph_component)
        + weights["community"] * _clamp_score(community_component)
        - weights["uncertainty"] * _clamp_score(uncertainty)
    )
    return unified, {
        "community": round(community_component, 6),
        "uncertainty": round(uncertainty, 6),
        "unified": round(unified, 6),
        "unified_protocol": UNIFIED_SCORE_PROTOCOL_VERSION,
        "unified_weights": dict(weights),
    }


def apply_unified_retrieval_scores(
    results: list[dict],
    *,
    query_type: str,
    top_n: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    weights = unified_score_weights(query_type)
    scored: list[dict] = []
    for item in results:
        metadata = item.setdefault("metadata", {})
        scores = metadata.setdefault("scores", {})
        unified, unified_components = compute_unified_score(item, query_type=query_type)
        scores.update(unified_components)
        item["score"] = unified
        scored.append(item)
    ranked = sorted(scored, key=lambda item: (item.get("metadata") or {}).get("scores", {}).get("unified", item.get("score", 0.0)), reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked, {
        "protocol_version": UNIFIED_SCORE_PROTOCOL_VERSION,
        "query_type": query_type,
        "weights": weights,
        "scored_documents": len(scored),
    }


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


def is_parent_chunk(chunk: ActiveChunk) -> bool:
    return bool((chunk.metadata_json or {}).get("is_parent"))


def is_child_retrieval_candidate(chunk: ActiveChunk, db: Session) -> bool:
    return True


def expand_results_with_parent_context(db: Session, knowledge_base_id: str, results: list[dict]) -> list[dict]:
    return results


def build_search_payload(chunk: ActiveChunk, document: Document, query: str, score: float, scores: dict | None = None) -> dict:
    chunk_metadata = chunk.metadata_json or {}
    partition = chunk_metadata.get("partition")
    section = chunk_metadata.get("section")
    page_number = chunk_metadata.get("page_number")
    source_type = chunk_metadata.get("source_type") or document.source_type
    snippet = chunk_metadata.get("snippet") or chunk.text[:240]
    citation = Citation(
        chunk_id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        source_path=document.source_path,
        partition=partition,
        section=section,
        page_number=page_number,
        snippet=snippet,
        active_chunk_id=chunk.id,
        evidence_atom_ids=list(chunk.atom_ids_json or []),
        source_span=dict(chunk.source_span_union_json or {}),
    )
    metadata = chunk_metadata | {"partition": partition, "source_type": source_type}
    metadata["is_parent"] = is_parent_chunk(chunk)
    metadata["active_chunk_id"] = chunk.id
    metadata["evidence_atom_ids"] = list(chunk.atom_ids_json or [])
    metadata["source_span_union"] = dict(chunk.source_span_union_json or {})
    metadata["graph_state_hash"] = chunk.graph_state_hash
    if scores:
        metadata["scores"] = scores
    return {
        "chunk_id": chunk.id,
        "active_chunk_id": chunk.id,
        "snippet": snippet,
        "score": score,
        "citations": [citation.model_dump()],
        "metadata": metadata,
        "content": chunk.text,
        "child_content": None,
        "document_title": document.title,
        "source_path": document.source_path,
        "partition": partition,
        "source_type": source_type,
    }


def query_embedding_cache_version(settings: Any | None = None) -> str:
    settings = settings or get_settings()
    runtime_version = current_runtime_settings_version() or "unversioned"
    return "query_embedding:" + stable_hash(
        {
            "text_version": "contextual_enriched_v2",
            "embedding_model": getattr(settings, "embedding_model", ""),
            "embedding_dimensions": int(getattr(settings, "embedding_dimensions", 0) or 0),
            "runtime_settings_version": runtime_version,
        }
    )[:16]


async def dense_search_chunks(db: Session, knowledge_base_id: str, query: str, filters: SearchFilters, top_k: int, model_audit: dict | None = None) -> list[dict]:
    KnowledgeBase = await asyncio.to_thread(db.get, KnowledgeBaseModel, knowledge_base_id)
    if KnowledgeBase is None:
        return []
    settings = get_settings()
    embedder = EmbeddingProvider()
    cache = get_cache_manager()
    embedding_version = query_embedding_cache_version(settings)
    cached_vector = cache.get_embedding(knowledge_base_id, query, embedding_version)
    if cached_vector is not None:
        embedding_result = type("obj", (object,), {"vectors": [cached_vector], "provider": "cache", "external_called": False, "fallback_reason": None})()
    else:
        embedding_result = await embedder.embed_texts_with_meta([query], text_type="query")
        if embedding_result.vectors and embedding_result.external_called:
            cache.set_embedding(knowledge_base_id, query, embedding_version, embedding_result.vectors[0])
    if model_audit is not None:
        model_audit.update(
            {
                "embedding_provider": embedding_result.provider,
                "embedding_external_called": embedding_result.external_called,
                "embedding_fallback_reason": embedding_result.fallback_reason,
            }
        )
    vector_store = VectorStore(knowledge_base_name=KnowledgeBase.name)
    results = await vector_store.async_search(
        vector=embedding_result.vectors[0],
        limit=max(top_k * 3, top_k),
        filters={
            "knowledge_base_id": knowledge_base_id,
            "partition": filters.partition,
            "difficulty": filters.difficulty,
            "source_type": filters.source_type,
        },
    )
    payloads = []
    dense_scores: list[float] = []
    for result in results:
        chunk = await asyncio.to_thread(db.get, ActiveChunk, str(result["id"]))
        if chunk is None or chunk.knowledge_base_id != knowledge_base_id or chunk.state != "active" or not await asyncio.to_thread(is_child_retrieval_candidate, chunk, db):
            continue
        document_id = (chunk.metadata_json or {}).get("document_id")
        document = await asyncio.to_thread(db.get, Document, document_id) if document_id else None
        if document is None or document.knowledge_base_id != knowledge_base_id:
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


async def search_chunks(db: Session, knowledge_base_id: str, query: str, filters: SearchFilters, top_k: int) -> list[dict]:
    results, _audit = await search_chunks_with_audit(db, knowledge_base_id, query, filters, top_k)
    return results


async def search_chunks_with_audit(db: Session, knowledge_base_id: str, query: str, filters: SearchFilters, top_k: int) -> tuple[list[dict], dict]:
    return await evidence_first_search_chunks_with_audit(db, knowledge_base_id, query, filters, top_k, route="retrieve_sources")


async def hybrid_search_chunks(db: Session, knowledge_base_id: str, query: str, filters: SearchFilters, top_k: int) -> list[dict]:
    results, _audit = await hybrid_search_chunks_with_audit(db, knowledge_base_id, query, filters, top_k)
    return results


async def hybrid_search_chunks_with_audit(db: Session, knowledge_base_id: str, query: str, filters: SearchFilters, top_k: int) -> tuple[list[dict], dict]:
    settings = get_settings()
    config = query_type_config(query)
    recall_k = max(int(config["recall_k"]), top_k)
    dense_results: list[dict] = []
    model_audit = default_model_audit()
    if is_degraded_mode() and not settings.enable_model_fallback:
        raise RuntimeError("OPENAI_API_KEY is required for search because ENABLE_MODEL_FALLBACK is false")
    if not is_degraded_mode():
        try:
            dense_results = await dense_search_chunks(db, knowledge_base_id, query, filters, recall_k, model_audit)
        except Exception:
            if not settings.enable_model_fallback:
                raise
            model_audit["embedding_fallback_reason"] = "dense_embedding_failed"
            dense_results = []
    lexical_results = lexical_search_chunks(db, knowledge_base_id, query, filters, recall_k)
    if not dense_results:
        results = rerank_or_return(query, lexical_results, top_k) if lexical_results else []
        results, unified_audit = apply_unified_retrieval_scores(results, query_type=config["query_type"], top_n=top_k)
        model_audit["unified_score"] = unified_audit
        model_audit["reranker_called"] = did_rerank(results)
        results = expand_results_with_parent_context(db, knowledge_base_id, results)
        return attach_model_audit(results, model_audit), model_audit
    if not lexical_results:
        results = rerank_or_return(query, dense_results, top_k)
        results, unified_audit = apply_unified_retrieval_scores(results, query_type=config["query_type"], top_n=top_k)
        model_audit["unified_score"] = unified_audit
        model_audit["reranker_called"] = did_rerank(results)
        results = expand_results_with_parent_context(db, knowledge_base_id, results)
        return attach_model_audit(results, model_audit), model_audit

    candidates = weighted_score_fusion(
        dense_results,
        lexical_results,
        alpha=float(config["alpha"]),
        top_n=max(recall_k, top_k),
        query_type=config["query_type"],
    )
    for item in candidates:
        item.setdefault("metadata", {}).setdefault("scores", {})["query_type"] = config["query_type"]
    results = rerank_or_return(query, candidates, top_k)
    results, unified_audit = apply_unified_retrieval_scores(results, query_type=config["query_type"], top_n=top_k)
    model_audit["unified_score"] = unified_audit
    model_audit["reranker_called"] = did_rerank(results)
    results = expand_results_with_parent_context(db, knowledge_base_id, results)
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


def weighted_score_fusion(
    dense_results: list[dict],
    lexical_results: list[dict],
    alpha: float,
    top_n: int,
    *,
    query_type: str = "default",
) -> list[dict]:
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
        unified, unified_components = compute_unified_score(item, query_type=query_type)
        scores.update(unified_components)
        scores["fusion_ranking_protocol"] = UNIFIED_SCORE_PROTOCOL_VERSION
        item["score"] = unified
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_n]


def _result_score(item: dict) -> float:
    scores = item.get("metadata", {}).get("scores", {})
    for key in ("unified", "rerank", "cross_encoder", "fused", "dense", "bm25", "graph_path", "signal_projection"):
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


def _merge_result_channels(current: dict, incoming: dict) -> dict:
    current_metadata = current.setdefault("metadata", {})
    incoming_metadata = incoming.get("metadata") or {}
    current_scores = current_metadata.setdefault("scores", {})
    current_scores.update(incoming_metadata.get("scores") or {})
    for key in ("signal_node_ids", "retrieval_signal_node_ids", "signal_edge_ids", "evidence_edge_ids", "evidence_atom_ids"):
        merged_values = {
            str(value)
            for value in [*(current_metadata.get(key) or []), *(incoming_metadata.get(key) or [])]
            if value
        }
        if merged_values:
            current_metadata[key] = sorted(merged_values)
    for key in ("signal_state_hash", "community_id", "path_id", "relation_type"):
        if current_metadata.get(key) is None and incoming_metadata.get(key) is not None:
            current_metadata[key] = incoming_metadata.get(key)
    current_metadata["graph_verified"] = bool(current_metadata.get("graph_verified") or incoming_metadata.get("graph_verified"))
    if incoming_metadata.get("signal_verified"):
        current_metadata["signal_verified"] = True
    if incoming_metadata.get("evidence_support_reason"):
        reasons = {
            str(value)
            for value in [
                current_metadata.get("evidence_support_reason"),
                incoming_metadata.get("evidence_support_reason"),
            ]
            if value
        }
        current_metadata["evidence_support_reason"] = "+".join(sorted(reasons))
    current["score"] = max(float(current.get("score", 0.0) or 0.0), float(incoming.get("score", 0.0) or 0.0))
    return current


def active_signal_layer_for_retrieval(db: Session, knowledge_base_id: str) -> tuple[SignalState | None, dict[str, Any]]:
    graph_state = db.scalar(
        select(EvidenceGraphState)
        .where(
            EvidenceGraphState.knowledge_base_id == knowledge_base_id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
        )
        .order_by(EvidenceGraphState.created_at.desc())
    )
    if graph_state is None:
        return None, {
            "signal_layer_complete": False,
            "signal_layer_status": None,
            "signal_state_hash": None,
            "skipped_reason": "missing_evidence_graph_state",
        }
    latest_signal_state = db.scalar(
        select(SignalState)
        .where(
            SignalState.knowledge_base_id == knowledge_base_id,
            SignalState.evidence_graph_state_id == graph_state.id,
        )
        .order_by(SignalState.created_at.desc())
    )
    active_signal = load_active_signal_state(db, knowledge_base_id=knowledge_base_id, graph_state_id=graph_state.id)
    signal = active_signal or latest_signal_state
    return active_signal, {
        "signal_layer_complete": active_signal is not None,
        "signal_layer_status": signal.status if signal else None,
        "signal_state_id": signal.id if signal else None,
        "signal_state_hash": active_signal.signal_state_hash if active_signal else None,
        "signal_schema_hash": active_signal.schema_hash if active_signal else None,
        "evidence_graph_state_hash": graph_state.state_hash,
        "skipped_reason": None if active_signal else "signal_layer_not_active",
    }


def retrieval_cache_scope_hash(
    *,
    filters: SearchFilters,
    route: str,
    top_k: int,
    query_type: str,
    embedding_version: str,
    graph_state_hash: str | None,
    community_state_hash: str | None,
    policy_state_hash: str | None,
    signal_state_hash: str | None,
    community_ids: list[int] | None,
) -> str:
    filters_json = filters.model_dump() if hasattr(filters, "model_dump") else {}
    return stable_hash(
        {
            "filters": filters_json,
            "route": route,
            "top_k": top_k,
            "query_type": query_type,
            "embedding_version": embedding_version,
            "graph_state_hash": graph_state_hash,
            "community_state_hash": community_state_hash,
            "policy_state_hash": policy_state_hash,
            "signal_state_hash": signal_state_hash,
            "community_ids": sorted(str(item) for item in (community_ids or [])),
            "prompt_protocol_hash": stable_hash({"prompt_protocol_version": "answer_grounding_v1"}),
        }
    )


def current_retrieval_scope_components(db: Session, knowledge_base_id: str) -> dict[str, str | None]:
    graph_state = db.scalar(
        select(EvidenceGraphState)
        .where(
            EvidenceGraphState.knowledge_base_id == knowledge_base_id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
        )
        .order_by(EvidenceGraphState.created_at.desc())
    )
    community_state_hash = None
    if graph_state is not None and graph_state.community_state_id:
        community_state_hash = db.scalar(
            select(CommunityState.state_hash).where(CommunityState.id == graph_state.community_state_id)
        )
    policy_state_hash = db.scalar(
        select(PolicyState.state_hash)
        .where(PolicyState.knowledge_base_id == knowledge_base_id)
        .order_by(PolicyState.created_at.desc())
    )
    signal_state = load_active_signal_state(db, knowledge_base_id=knowledge_base_id, graph_state_id=graph_state.id) if graph_state is not None else None
    return {
        "graph_state_hash": graph_state.state_hash if graph_state else None,
        "community_state_hash": community_state_hash,
        "policy_state_hash": policy_state_hash,
        "signal_state_hash": signal_state.signal_state_hash if signal_state else None,
    }


def _signal_node_ids_from_results(results: list[dict]) -> list[str]:
    return sorted(
        {
            str(node_id)
            for item in results
            for node_id in (item.get("metadata") or {}).get("signal_node_ids", [])
            if node_id
        }
    )


def plan_signal_projection_paths(
    db: Session,
    knowledge_base_id: str,
    anchors: list[dict],
    *,
    query: str,
    signal_state: SignalState | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    signal_state = signal_state or active_signal_layer_for_retrieval(db, knowledge_base_id)[0]
    if signal_state is None:
        return [], {
            "planned_paths": 0,
            "signal_edges": 0,
            "expanded_active_chunks": 0,
            "signal_layer_complete": False,
            "skipped_reason": "signal_layer_not_active",
        }

    query_terms = set(tokenize_for_retrieval(query))
    query_lower = query.lower()
    seed_node_ids = {
        str(node_id)
        for anchor in anchors
        for node_id in (anchor.get("metadata") or {}).get("signal_node_ids", [])
        if node_id
    }
    all_nodes = db.scalars(
        select(SignalNode).where(SignalNode.signal_state_id == signal_state.id)
    ).all()
    for node in all_nodes:
        name = f"{node.canonical_label} {node.normalized_key}".lower()
        node_terms = set(tokenize_for_retrieval(name))
        if node.normalized_key and node.normalized_key in query_lower:
            seed_node_ids.add(node.id)
        elif query_terms and query_terms.intersection(node_terms):
            seed_node_ids.add(node.id)
    if not seed_node_ids:
        return [], {
            "planned_paths": 0,
            "signal_edges": 0,
            "expanded_active_chunks": 0,
            "signal_layer_complete": True,
            "signal_state_hash": signal_state.signal_state_hash,
            "skipped_reason": "no_seed_signals",
        }

    signal_edges = db.scalars(
        select(SignalEdge).where(
            SignalEdge.signal_state_id == signal_state.id,
            SignalEdge.source_signal_id.in_(seed_node_ids) | SignalEdge.target_signal_id.in_(seed_node_ids),
        )
    ).all()
    expanded_node_ids = set(seed_node_ids)
    for edge in signal_edges:
        if edge.source_signal_id in seed_node_ids:
            expanded_node_ids.add(edge.target_signal_id)
        if edge.target_signal_id in seed_node_ids:
            expanded_node_ids.add(edge.source_signal_id)
    nodes_by_id = {node.id: node for node in all_nodes if node.id in expanded_node_ids}
    active_chunk_ids = sorted(
        {
            str(active_chunk_id)
            for node in nodes_by_id.values()
            for active_chunk_id in (node.support_active_chunk_ids_json or [])
            if active_chunk_id
        }
    )
    if not active_chunk_ids:
        return [], {
            "planned_paths": 0,
            "signal_edges": len(signal_edges),
            "expanded_active_chunks": 0,
            "signal_layer_complete": True,
            "signal_state_hash": signal_state.signal_state_hash,
            "seed_signal_node_ids": sorted(seed_node_ids),
            "skipped_reason": "signals_have_no_active_chunks",
        }

    active_chunks = db.scalars(
        select(ActiveChunk).where(
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.id.in_(active_chunk_ids),
            ActiveChunk.state == "active",
        )
    ).all()
    anchor_active_chunk_ids = {
        str((anchor.get("metadata") or {}).get("active_chunk_id"))
        for anchor in anchors
        if (anchor.get("metadata") or {}).get("active_chunk_id")
    }
    paths: list[dict] = []
    for active_chunk in active_chunks:
        if active_chunk.id in anchor_active_chunk_ids:
            continue
        chunk_node_ids = [
            node.id
            for node in nodes_by_id.values()
            if active_chunk.id in {str(item) for item in (node.support_active_chunk_ids_json or [])}
        ]
        support_atom_ids = sorted(
            {
                str(atom_id)
                for node_id in chunk_node_ids
                for atom_id in (nodes_by_id[node_id].support_atom_ids_json or [])
                if atom_id
            }
        )
        paths.append(
            {
                "path_id": f"signal-neighbor-{len(paths) + 1}",
                "signal_state_hash": signal_state.signal_state_hash,
                "signal_node_ids": sorted(chunk_node_ids),
                "signal_edge_ids": [edge.id for edge in signal_edges[:8]],
                "active_chunk_ids": [active_chunk.id],
                "evidence_atom_ids": support_atom_ids,
                "evidence_chunk_ids": [active_chunk.id],
                "relation_types": ["signal_projection_neighbor"],
                "score": round(min(0.82, 0.42 + 0.05 * len(chunk_node_ids) + 0.02 * len(support_atom_ids)), 4),
            }
        )
        if len(paths) >= EVIDENCE_FIRST_MAX_PATHS:
            break
    return paths, {
        "planned_paths": len(paths),
        "signal_edges": len(signal_edges),
        "expanded_active_chunks": len(paths),
        "signal_layer_complete": True,
        "signal_state_hash": signal_state.signal_state_hash,
        "seed_signal_node_ids": sorted(seed_node_ids),
        "expanded_signal_node_ids": sorted(expanded_node_ids),
        "skipped_reason": None if paths else "no_non_anchor_active_chunks",
    }


def select_evidence_anchors(db: Session, knowledge_base_id: str, base_results: list[dict], *, max_anchors: int = EVIDENCE_FIRST_MAX_ANCHORS) -> tuple[list[dict], dict]:
    if not base_results:
        return [], {"anchor_count": 0, "candidate_count": 0}
    scored: list[tuple[float, dict]] = []
    for item in base_results:
        metadata = item.get("metadata") or {}
        quality_action = metadata.get("quality_action")
        routes = metadata.get("route_eligibility") or {}
        base_score = _result_score(item)
        quality_bonus = 0.0
        if routes.get("evidence_graph") or quality_action == "graph_candidate":
            quality_bonus = 0.08
        elif routes.get("retrieval") or quality_action == "retrieval_candidate":
            quality_bonus = 0.05
        elif routes.get("evidence_only") or quality_action == "evidence_only":
            quality_bonus = 0.02
        evidence_atom_ids = [str(atom_id) for atom_id in metadata.get("evidence_atom_ids") or [] if atom_id]
        traceability_bonus = 0.08 if metadata.get("active_chunk_id") and evidence_atom_ids else 0.0
        span_bonus = 0.04 if metadata.get("source_span_union") else 0.0
        signal_node_ids = [str(node_id) for node_id in metadata.get("signal_node_ids") or [] if node_id]
        signal_bonus = min(0.05, 0.015 * len(signal_node_ids)) if metadata.get("signal_state_hash") else 0.0
        anchor_score = base_score + quality_bonus + traceability_bonus + span_bonus + signal_bonus
        if base_score <= 0 and not evidence_atom_ids:
            continue
        scored.append(
            (
                anchor_score,
                _mark_result(
                    item,
                    retrieval_stage="evidence_anchor_selector",
                    evidence_role="base_anchor",
                    anchor_score=round(anchor_score, 4),
                    anchor_atom_ids=evidence_atom_ids,
                    anchor_signal_node_ids=signal_node_ids,
                    graph_verified=bool(evidence_atom_ids),
                    evidence_support_reason="base_retrieval_traceable_atom" if evidence_atom_ids else "base_retrieval_high_score",
                ),
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    anchors = [item for _score, item in scored[:max_anchors]]
    return anchors, {
        "anchor_count": len(anchors),
        "candidate_count": len(base_results),
        "traceable_anchor_atoms": sum(len((item.get("metadata") or {}).get("anchor_atom_ids") or []) for item in anchors),
        "traceable_anchor_signals": sum(len((item.get("metadata") or {}).get("anchor_signal_node_ids") or []) for item in anchors),
    }


def plan_evidence_chains(
    db: Session,
    knowledge_base_id: str,
    anchors: list[dict],
    *,
    query_type: str,
    community_ids: list[int] | None = None,
) -> tuple[list[dict], dict]:
    anchor_atom_ids = {
        str(atom_id)
        for anchor in anchors
        for atom_id in (anchor.get("metadata", {}).get("anchor_atom_ids") or anchor.get("metadata", {}).get("evidence_atom_ids") or [])
        if atom_id
    }
    if not anchor_atom_ids:
        return [], {
            "planned_paths": 0,
            "observed_edges": 0,
            "expanded_active_chunks": 0,
            "skipped_reason": "no_anchor_atoms",
        }
    latest_graph_state = db.scalar(
        select(EvidenceGraphState)
        .where(
            EvidenceGraphState.knowledge_base_id == knowledge_base_id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
        )
        .order_by(EvidenceGraphState.created_at.desc())
    )
    if latest_graph_state is None:
        return [], {
            "planned_paths": 0,
            "observed_edges": 0,
            "expanded_active_chunks": 0,
            "skipped_reason": "missing_evidence_graph_state",
        }
    edges = db.scalars(
        select(EvidenceEdge).where(
            EvidenceEdge.graph_state_id == latest_graph_state.id,
            EvidenceEdge.source_atom_id.in_(anchor_atom_ids) | EvidenceEdge.target_atom_id.in_(anchor_atom_ids),
        )
    ).all()
    neighbor_atom_ids = {
        str(edge.target_atom_id if str(edge.source_atom_id) in anchor_atom_ids else edge.source_atom_id)
        for edge in edges
    }
    if community_ids:
        anchor_active_chunk_ids = {
            str((anchor.get("metadata") or {}).get("active_chunk_id"))
            for anchor in anchors
            if (anchor.get("metadata") or {}).get("active_chunk_id")
        }
        community_ids_as_text = {str(item) for item in community_ids if item is not None}
        for active_chunk in db.scalars(
            select(ActiveChunk).where(
                ActiveChunk.knowledge_base_id == knowledge_base_id,
                ActiveChunk.state == "active",
            )
        ).all():
            if active_chunk.id in anchor_active_chunk_ids:
                continue
            if community_ids_as_text.intersection({str(item) for item in (active_chunk.community_ids_json or [])}):
                neighbor_atom_ids.update(str(atom_id) for atom_id in active_chunk.atom_ids_json or [])
    if not neighbor_atom_ids:
        return [], {
            "planned_paths": 0,
            "observed_edges": len(edges),
            "expanded_active_chunks": 0,
            "skipped_reason": "no_neighbor_atoms",
        }
    active_chunks = db.scalars(
        select(ActiveChunk).where(
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.state == "active",
        )
    ).all()
    paths: list[dict] = []
    seen_chunk_ids: set[str] = set()
    for active_chunk in active_chunks:
        if active_chunk.id in seen_chunk_ids:
            continue
        overlap = neighbor_atom_ids.intersection({str(atom_id) for atom_id in active_chunk.atom_ids_json or []})
        if not overlap:
            continue
        seen_chunk_ids.add(active_chunk.id)
        paths.append(
            {
                "path_id": f"evidence-neighbor-{len(paths) + 1}",
                "evidence_edge_ids": [edge.id for edge in edges[:8]],
                "evidence_atom_ids": sorted(overlap),
                "evidence_chunk_ids": [active_chunk.id],
                "relation_types": ["evidence_neighbor"],
                "score": round(min(0.85, 0.45 + 0.05 * len(overlap)), 4),
            }
        )
        if len(paths) >= EVIDENCE_FIRST_MAX_PATHS:
            break
    return paths[:EVIDENCE_FIRST_MAX_PATHS], {
        "planned_paths": min(len(paths), EVIDENCE_FIRST_MAX_PATHS),
        "observed_edges": len(edges),
        "expanded_active_chunks": len(paths),
        "skipped_reason": None if paths else "no_evidence_neighbors",
    }


def controlled_graph_enhancement(
    db: Session,
    knowledge_base_id: str,
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
        select(ActiveChunk).where(
            ActiveChunk.id.in_(list(evidence_by_chunk)),
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.state == "active",
        )
    ).all()
    enhanced: list[dict] = []
    for chunk in chunks:
        metadata = chunk.metadata_json or {}
        document = db.get(Document, metadata.get("document_id"))
        if document is None or document.knowledge_base_id != knowledge_base_id or not document.is_active:
            continue
        if filters.partition and metadata.get("partition") != filters.partition:
            continue
        if filters.source_type and (metadata.get("source_type") or document.source_type) != filters.source_type:
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
                evidence_role="evidence_neighbor",
                path_id=path.get("path_id"),
                evidence_edge_ids=path.get("evidence_edge_ids") or [],
                evidence_atom_ids=path.get("evidence_atom_ids") or [],
                relation_type=(path.get("relation_types") or [None])[0],
                graph_verified=bool(path.get("evidence_edge_ids")),
                community_id=path.get("community_id"),
                evidence_support_reason="evidence_graph_neighbor",
            )
        )
    return expand_results_with_parent_context(db, knowledge_base_id, enhanced), {
        "graph_enhanced_chunks": len(enhanced),
        "path_evidence_chunks": len(evidence_by_chunk),
    }


def controlled_signal_enhancement(
    db: Session,
    knowledge_base_id: str,
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
        return [], {"signal_enhanced_chunks": 0, "path_evidence_chunks": 0}

    chunks = db.scalars(
        select(ActiveChunk).where(
            ActiveChunk.id.in_(list(evidence_by_chunk)),
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.state == "active",
        )
    ).all()
    enhanced: list[dict] = []
    for chunk in chunks:
        metadata = chunk.metadata_json or {}
        document = db.get(Document, metadata.get("document_id"))
        if document is None or document.knowledge_base_id != knowledge_base_id or not document.is_active:
            continue
        if filters.partition and metadata.get("partition") != filters.partition:
            continue
        if filters.source_type and (metadata.get("source_type") or document.source_type) != filters.source_type:
            continue
        if filters.tags and not set(filters.tags).intersection(set(document.tags or [])):
            continue
        path = evidence_by_chunk[str(chunk.id)]
        item = build_search_payload(
            chunk,
            document,
            query,
            float(path.get("score", 0.0)),
            {"signal_projection": float(path.get("score", 0.0))},
        )
        enhanced.append(
            _mark_result(
                item,
                retrieval_stage="signal_projection_enhancer",
                evidence_role="signal_neighbor",
                path_id=path.get("path_id"),
                signal_state_hash=path.get("signal_state_hash"),
                signal_node_ids=path.get("signal_node_ids") or [],
                retrieval_signal_node_ids=path.get("signal_node_ids") or [],
                signal_edge_ids=path.get("signal_edge_ids") or [],
                evidence_atom_ids=path.get("evidence_atom_ids") or [],
                relation_type=(path.get("relation_types") or [None])[0],
                graph_verified=bool(path.get("evidence_atom_ids")),
                signal_verified=True,
                evidence_support_reason="signal_projection_supported_active_chunk",
            )
        )
    return expand_results_with_parent_context(db, knowledge_base_id, enhanced), {
        "signal_enhanced_chunks": len(enhanced),
        "path_evidence_chunks": len(evidence_by_chunk),
    }


def assemble_evidence_documents(
    base_results: list[dict],
    anchors: list[dict],
    graph_results: list[dict],
    top_k: int,
    signal_results: list[dict] | None = None,
) -> tuple[list[dict], dict]:
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
        if current is None:
            merged[chunk_id] = _mark_result(item, retrieval_stage="evidence_assembler")
        else:
            _merge_result_channels(current, item)
    for item in signal_results or []:
        chunk_id = str(item["chunk_id"])
        current = merged.get(chunk_id)
        if current is None:
            merged[chunk_id] = _mark_result(item, retrieval_stage="evidence_assembler")
        else:
            _merge_result_channels(current, item)
    documents = sorted(merged.values(), key=_result_score, reverse=True)[: max(top_k * 2, top_k)]
    return documents, {
        "assembled_documents": len(documents),
        "base_documents": len(base_results),
        "anchor_documents": len(anchors),
        "graph_documents": len(graph_results),
        "signal_documents": len(signal_results or []),
    }


def record_retrieval_trace(
    db: Session,
    *,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    retrieval_mode: str,
    results: list[dict],
    audit: dict,
) -> RetrievalTrace:
    active_chunk_ids = [
        str(item.get("active_chunk_id") or (item.get("metadata") or {}).get("active_chunk_id"))
        for item in results
        if item.get("active_chunk_id") or (item.get("metadata") or {}).get("active_chunk_id")
    ]
    metadata_items = [item.get("metadata") or {} for item in results]
    graph_state_hash = next((str(meta.get("graph_state_hash")) for meta in metadata_items if meta.get("graph_state_hash")), None)
    policy_state_hash = next((str(meta.get("policy_state_hash")) for meta in metadata_items if meta.get("policy_state_hash")), None)
    community_state_hash = next((str(meta.get("community_state_hash")) for meta in metadata_items if meta.get("community_state_hash")), None)
    signal_state_hash = (
        next((str(meta.get("signal_state_hash")) for meta in metadata_items if meta.get("signal_state_hash")), None)
        or audit.get("signal_state_hash")
        or (audit.get("signal") or {}).get("signal_state_hash")
    )
    signal_node_ids = sorted(
        {
            str(node_id)
            for meta in metadata_items
            for node_id in [*(meta.get("signal_node_ids") or []), *(meta.get("retrieval_signal_node_ids") or [])]
            if node_id
        }
    )
    scope_hash = active_chunk_scope_hash(active_chunk_ids, graph_state_hash, policy_state_hash, signal_state_hash) if active_chunk_ids else None
    trace = RetrievalTrace(
        knowledge_base_id=knowledge_base_id,
        query=query,
        filters_json=filters.model_dump() if hasattr(filters, "model_dump") else {},
        retrieval_mode=retrieval_mode,
        active_chunk_scope_hash=scope_hash,
        evidence_graph_state_hash=graph_state_hash,
        community_state_hash=community_state_hash,
        policy_state_hash=policy_state_hash,
        prompt_protocol_hash=stable_hash({"prompt_protocol_version": "answer_grounding_v1"}),
        result_active_chunk_ids_json=active_chunk_ids,
        expansion_path_json={
            "anchors": audit.get("anchors", {}),
            "paths": audit.get("paths", {}),
            "graph": audit.get("graph", {}),
            "signal": audit.get("signal", {}),
            "assembly": audit.get("assembly", {}),
        },
        scores_json={item["chunk_id"]: (item.get("metadata") or {}).get("scores", {}) for item in results if item.get("chunk_id")},
        diagnostics_json=audit,
    )
    db.add(trace)
    db.commit()
    db.refresh(trace)
    for item in results:
        metadata = item.setdefault("metadata", {})
        metadata["retrieval_trace_id"] = trace.id
        metadata["retrieval_signal_node_ids"] = sorted(
            {
                str(node_id)
                for node_id in [*(metadata.get("retrieval_signal_node_ids") or []), *(metadata.get("signal_node_ids") or [])]
                if node_id
            }
        )
        for citation in item.get("citations", []):
            if isinstance(citation, dict):
                citation["retrieval_trace_id"] = trace.id
    trace.diagnostics_json = {
        **(trace.diagnostics_json or {}),
        "signal_state_hash": signal_state_hash,
        "retrieval_signal_node_ids": signal_node_ids,
    }
    db.commit()
    db.refresh(trace)
    return trace


async def evidence_first_search_chunks_with_audit(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int,
    route: str = "retrieve_sources",
    community_ids: list[int] | None = None,
) -> tuple[list[dict], dict]:
    config = query_type_config(query)
    recall_k = max(top_k * 3, int(config["recall_k"]))
    embedding_version = query_embedding_cache_version()
    scope_components = current_retrieval_scope_components(db, knowledge_base_id)
    cache_scope_hash = retrieval_cache_scope_hash(
        filters=filters,
        route=route,
        top_k=top_k,
        query_type=config["query_type"],
        embedding_version=embedding_version,
        graph_state_hash=scope_components.get("graph_state_hash"),
        community_state_hash=scope_components.get("community_state_hash"),
        policy_state_hash=scope_components.get("policy_state_hash"),
        signal_state_hash=scope_components.get("signal_state_hash"),
        community_ids=community_ids,
    )
    cache = get_cache_manager()
    cached_payload = cache.get_search_results(knowledge_base_id, query, cache_scope_hash, embedding_version)
    if isinstance(cached_payload, dict):
        cached_documents = copy.deepcopy(cached_payload.get("results") or [])
        cached_audit = copy.deepcopy(cached_payload.get("audit") or {})
        if cached_documents:
            cached_audit = {
                **cached_audit,
                "cached": True,
                "retrieval_cache_scope_hash": cache_scope_hash,
                "cache": {"hit": True, "scope_hash": cache_scope_hash},
            }
            for item in cached_documents:
                metadata = item.setdefault("metadata", {})
                metadata["cached"] = True
                metadata["retrieval_cache_scope_hash"] = cache_scope_hash
                metadata["model_audit"] = {**dict(metadata.get("model_audit") or {}), "cached": True, "scope_hash": cache_scope_hash}
            trace = record_retrieval_trace(
                db,
                knowledge_base_id=knowledge_base_id,
                query=query,
                filters=filters,
                retrieval_mode="evidence_first_v1",
                results=cached_documents,
                audit=cached_audit,
            )
            cached_audit["retrieval_trace_id"] = trace.id
            return cached_documents[:top_k], cached_audit
    base_results, model_audit = await hybrid_search_chunks_with_audit(db, knowledge_base_id, query, filters, recall_k)
    base_results = [
        _mark_result(item, retrieval_stage="base_retrieval", evidence_role="base_candidate", graph_verified=False)
        for item in base_results
    ]
    anchors, anchor_audit = select_evidence_anchors(db, knowledge_base_id, base_results)
    signal_state, signal_layer_audit = active_signal_layer_for_retrieval(db, knowledge_base_id)
    should_plan_graph = route == "multi_hop_research" or config["query_type"] in {"comparison", "procedure", "formula"} or bool(community_ids)
    paths: list[dict] = []
    path_audit = {"planned_paths": 0, "observed_edges": 0, "skipped_reason": "simple_query"}
    graph_results: list[dict] = []
    graph_audit = {"graph_enhanced_chunks": 0, "path_evidence_chunks": 0}
    signal_paths: list[dict] = []
    signal_audit: dict[str, Any] = {
        **signal_layer_audit,
        "planned_paths": 0,
        "expanded_active_chunks": 0,
    }
    signal_results: list[dict] = []
    signal_result_audit = {"signal_enhanced_chunks": 0, "path_evidence_chunks": 0}
    if should_plan_graph and anchors:
        paths, path_audit = plan_evidence_chains(
            db,
            knowledge_base_id,
            anchors,
            query_type=config["query_type"],
            community_ids=community_ids,
        )
        graph_results, graph_audit = controlled_graph_enhancement(
            db,
            knowledge_base_id,
            query,
            filters,
            {str(item["chunk_id"]) for item in base_results},
            paths,
        )
    if anchors and signal_state is not None:
        signal_paths, signal_audit = plan_signal_projection_paths(
            db,
            knowledge_base_id,
            anchors,
            query=query,
            signal_state=signal_state,
        )
        signal_results, signal_result_audit = controlled_signal_enhancement(
            db,
            knowledge_base_id,
            query,
            filters,
            {str(item["chunk_id"]) for item in base_results},
            signal_paths,
        )
        signal_audit = {**signal_layer_audit, **signal_audit, **signal_result_audit}
    documents, assembly_audit = assemble_evidence_documents(base_results, anchors, graph_results, top_k, signal_results)
    documents, unified_audit = apply_unified_retrieval_scores(
        documents,
        query_type=config["query_type"],
        top_n=max(top_k * 2, top_k),
    )
    graph_hash = next(((item.get("metadata") or {}).get("graph_state_hash") for item in documents if (item.get("metadata") or {}).get("graph_state_hash")), None)
    policy_hash = next(((item.get("metadata") or {}).get("policy_state_hash") for item in documents if (item.get("metadata") or {}).get("policy_state_hash")), None)
    community_hash = next(((item.get("metadata") or {}).get("community_state_hash") for item in documents if (item.get("metadata") or {}).get("community_state_hash")), None)
    signal_hash = signal_audit.get("signal_state_hash") or next(
        ((item.get("metadata") or {}).get("signal_state_hash") for item in documents if (item.get("metadata") or {}).get("signal_state_hash")),
        None,
    )
    evidence_audit = {
        **model_audit,
        "retrieval_pipeline": "evidence_first_v1",
        "query_type": config["query_type"],
        "route": route,
        "base_candidate_count": len(base_results),
        "anchors": anchor_audit,
        "paths": path_audit,
        "graph": graph_audit,
        "signal": signal_audit,
        "assembly": assembly_audit,
        "unified_score": unified_audit,
        "community_ids": community_ids or [],
        "signal_state_hash": signal_hash,
        "signal_node_ids": _signal_node_ids_from_results(documents),
        "retrieval_cache_scope_hash": cache_scope_hash,
        "cache": {"hit": False, "scope_hash": cache_scope_hash},
    }
    for item in documents:
        metadata = item.setdefault("metadata", {})
        metadata.setdefault("signal_state_hash", signal_hash)
        metadata["retrieval_signal_node_ids"] = sorted(
            {
                str(node_id)
                for node_id in [*(metadata.get("retrieval_signal_node_ids") or []), *(metadata.get("signal_node_ids") or [])]
                if node_id
            }
        )
        metadata["evidence_first_audit"] = {
            "pipeline": "evidence_first_v1",
            "route": route,
            "query_type": config["query_type"],
            "signal_state_hash": signal_hash,
            "signal_layer_complete": bool(signal_audit.get("signal_layer_complete")),
            "signal_node_ids": metadata.get("retrieval_signal_node_ids") or [],
        }
        metadata.setdefault("model_audit", dict(model_audit))
    cache_documents = copy.deepcopy(documents)
    cache_audit = copy.deepcopy(evidence_audit)
    trace = record_retrieval_trace(
        db,
        knowledge_base_id=knowledge_base_id,
        query=query,
        filters=filters,
        retrieval_mode="evidence_first_v1",
        results=documents,
        audit=evidence_audit,
    )
    evidence_audit["retrieval_trace_id"] = trace.id
    ttl = int(get_settings().retrieval_cache_ttl_seconds or 0)
    if ttl > 0:
        cache.set_search_results(
            knowledge_base_id,
            query,
            cache_scope_hash,
            embedding_version,
            {"results": cache_documents, "audit": cache_audit},
            ttl=ttl,
        )
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


def lexical_search_chunks(db: Session, knowledge_base_id: str, query: str, filters: SearchFilters, top_k: int) -> list[dict]:
    query_terms = tokenize_for_retrieval(query)
    active_chunks = db.scalars(
        select(ActiveChunk)
        .where(ActiveChunk.knowledge_base_id == knowledge_base_id, ActiveChunk.state == "active")
        .order_by(ActiveChunk.created_at.desc())
    ).all()
    corpus: list[list[str]] = []
    chunk_documents: list[tuple[ActiveChunk, Document]] = []
    for chunk in active_chunks:
        if not is_child_retrieval_candidate(chunk, db):
            continue
        metadata = chunk.metadata_json or {}
        document = db.get(Document, metadata.get("document_id"))
        if document is None or not document.is_active:
            continue
        if filters.partition and metadata.get("partition") != filters.partition:
            continue
        if filters.source_type and (metadata.get("source_type") or document.source_type) != filters.source_type:
            continue
        if filters.tags and not set(filters.tags).intersection(set(document.tags or [])):
            continue
        corpus.append(tokenize_for_retrieval(f"{document.title}\n{metadata.get('section') or ''}\n{chunk.text}"))
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


async def answer_question(db: Session, knowledge_base_id: str, question: str, filters: SearchFilters, top_k: int, history: list[dict]) -> dict:
    strategy_profile = get_active_profile_record(db, knowledge_base_id)
    with use_strategy_profile(strategy_profile.profile_json):
        results = await search_chunks(db, knowledge_base_id, question, filters, top_k)
        chat = ChatProvider()
        answer = await chat.answer_question(question, results, history)
    return {
        "answer": answer,
        "citations": [citation for result in results for citation in result["citations"]],
        "used_chunks": results,
        "degraded_mode": is_degraded_mode(),
    }


def get_dashboard_snapshot(db: Session, knowledge_base_id: str) -> dict:
    KnowledgeBase = db.get(KnowledgeBaseModel, knowledge_base_id)
    if KnowledgeBase is None:
        return {
            "knowledge_base": {
                "id": "empty",
                "name": "KnowledgeBase Workspace",
                "description": None,
                "source_root": "",
                "storage_root": "",
                "document_count": 0,
                "evidence_atom_count": 0,
                "current_chunk_version": 0,
                "has_parsed_chunks": False,
                "can_full_reparse": False,
                "degraded_mode": is_degraded_mode(),
                "active_profile_id": None,
                "active_profile_name": None,
                "active_profile_hash": None,
            },
            "tree": [],
            "graph": {"nodes": [], "edges": [], "focus_partition": None},
            "batch_status": None,
            "ingested_document_count": 0,
            "chunk_count": 0,
            "evidence_atom_count": 0,
            "active_chunk_count": 0,
            "evidence_edge_count": 0,
            "community_region_count": 0,
            "graph_eligible_chunk_count": 0,
            "graph_relation_count": 0,
            "coverage_by_source_type": {},
            "degraded_mode": is_degraded_mode(),
        }

    documents = db.scalars(select(Document).where(Document.knowledge_base_id == KnowledgeBase.id, Document.is_active.is_(True))).all()
    file_items = list_knowledge_base_files(db, KnowledgeBase.id)
    chunks = db.scalars(select(ActiveChunk).where(ActiveChunk.knowledge_base_id == KnowledgeBase.id, ActiveChunk.state == "active")).all()
    chunk_count = len(chunks)
    evidence_atom_count = db.scalar(select(func.count(EvidenceAtom.id)).where(EvidenceAtom.knowledge_base_id == KnowledgeBase.id, EvidenceAtom.state == "active")) or 0
    active_chunk_count = db.scalar(select(func.count(ActiveChunk.id)).where(ActiveChunk.knowledge_base_id == KnowledgeBase.id, ActiveChunk.state == "active")) or 0
    latest_graph_state = db.scalar(
        select(EvidenceGraphState)
        .where(
            EvidenceGraphState.knowledge_base_id == KnowledgeBase.id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
        )
        .order_by(EvidenceGraphState.created_at.desc())
    )
    evidence_edge_count = (
        db.scalar(select(func.count(EvidenceEdge.id)).where(EvidenceEdge.graph_state_id == latest_graph_state.id)) or 0
        if latest_graph_state is not None
        else 0
    )
    community_region_count = (
        db.scalar(select(func.count(CommunityState.id)).where(CommunityState.knowledge_base_id == KnowledgeBase.id, CommunityState.state == "active")) or 0
    )
    current_chunk_version = max([int((chunk.metadata_json or {}).get("chunk_version") or 0) for chunk in chunks] + [KnowledgeBase.current_chunk_version or 0])
    if (KnowledgeBase.current_chunk_version or 0) < current_chunk_version:
        KnowledgeBase.current_chunk_version = current_chunk_version
        db.flush()
    graph_eligible_chunk_count = sum(
        1
        for chunk in chunks
        if ((chunk.metadata_json or {}).get("route_eligibility") or {}).get("evidence_graph") is True
    )
    batches = db.scalars(select(IngestionBatch).where(IngestionBatch.knowledge_base_id == KnowledgeBase.id).order_by(IngestionBatch.created_at.desc())).all()

    partition_map: dict[str, list[dict]] = defaultdict(list)
    source_coverage = Counter()
    for item in file_items:
        partition = item.get("partition") or "General"
        partition_map[partition].append(item)
        source_coverage[item.get("source_type") or "unknown"] += 1

    tree = [
        {
            "id": f"partition:{partition}",
            "title": partition,
            "type": "partition",
            "children": [
                {"id": item["document_id"] or item["id"], "title": item["title"], "type": "document", "children": []}
                for item in sorted(entries, key=lambda item: item["title"])
            ],
        }
        for partition, entries in sorted(partition_map.items())
    ]
    latest_batch = next((batch for batch in batches if batch.status not in TERMINAL_BATCH_STATES), None)
    graph_payload = get_graph_payload(db, KnowledgeBase.id, graph_type="evidence")
    active_profile = get_active_profile_record(db, KnowledgeBase.id)
    return {
        "knowledge_base": {
            "id": KnowledgeBase.id,
            "name": KnowledgeBase.name,
            "description": KnowledgeBase.description,
            "source_root": str(get_settings().knowledge_base_paths_for_name(KnowledgeBase.name)["storage_root"]),
            "storage_root": str(get_settings().knowledge_base_paths_for_name(KnowledgeBase.name)["storage_root"]),
            "document_count": len(file_items),
            "evidence_atom_count": evidence_atom_count,
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
                if key.startswith("graph_") or key.startswith("evidence_") or key in {"active_chunks", "community_regions"}
            },
            "started_at": latest_batch.started_at,
            "completed_at": latest_batch.completed_at,
        },
        "ingested_document_count": len(file_items),
        "chunk_count": chunk_count,
        "evidence_atom_count": evidence_atom_count,
        "active_chunk_count": active_chunk_count,
        "evidence_edge_count": evidence_edge_count,
        "community_region_count": community_region_count,
        "graph_eligible_chunk_count": graph_eligible_chunk_count,
        "graph_relation_count": evidence_edge_count,
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


def list_knowledge_base_files(db: Session, knowledge_base_id: str) -> list[dict]:
    KnowledgeBase = db.get(KnowledgeBaseModel, knowledge_base_id)
    documents = db.scalars(select(Document).where(Document.knowledge_base_id == knowledge_base_id, Document.is_active.is_(True))).all()
    storage_root = get_settings().knowledge_base_paths_for_name(KnowledgeBase.name)["storage_root"] if KnowledgeBase is not None else None
    storage_paths = {str(path) for path in collect_knowledge_base_storage_paths(KnowledgeBase)} if KnowledgeBase is not None else set()
    document_versions = db.scalars(
        select(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.knowledge_base_id == knowledge_base_id, Document.is_active.is_(True), DocumentVersion.is_active.is_(True))
    ).all()
    documents_by_id = {document.id: document for document in documents}
    documents_by_storage_path = {
        version.storage_path: documents_by_id[version.document_id]
        for version in document_versions
        if version.document_id in documents_by_id and version.storage_path
    }
    jobs = db.scalars(select(IngestionJob).where(IngestionJob.knowledge_base_id == knowledge_base_id).order_by(IngestionJob.updated_at.desc())).all()
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
    if KnowledgeBase is not None:
        for path in sorted((Path(path_string) for path_string in storage_paths), key=lambda item: str(item).lower()):
            path_string = str(path)
            if path_string in removed_paths:
                continue
            if path_string in items:
                continue
            job = latest_jobs.get(path_string)
            document = documents_by_storage_path.get(path_string)
            document_chunks = [
                chunk
                for chunk in db.scalars(
                    select(ActiveChunk).where(ActiveChunk.knowledge_base_id == knowledge_base_id, ActiveChunk.state == "active")
                ).all()
                if document and (chunk.metadata_json or {}).get("document_id") == document.id
            ]
            chunk_count = len(document_chunks)
            chunk_version = max([int((chunk.metadata_json or {}).get("chunk_version") or 0) for chunk in document_chunks] or [0]) or None
            items[path_string] = {
                "id": document.id if document else path_string,
                "document_id": document.id if document else None,
                "title": document.title if document else path.stem or path.name,
                "source_path": path_string,
                "source_type": document.source_type if document else source_type_from_path(path_string),
                "partition": document.tags[0]
                if document and document.tags and not is_invalid_partition_label(document.tags[0], knowledge_base_name=KnowledgeBase.name if KnowledgeBase else None)
                else derive_partition(path, knowledge_base_name=KnowledgeBase.name if KnowledgeBase else None),
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
            "partition": None,
            "status": file_status_from_job(job, has_parsed_chunks=False),
            "job_state": job.status,
            "batch_id": job.batch_id,
            "error": job.error_message,
            "chunk_count": 0,
            "chunk_version": None,
            "updated_at": job.updated_at,
        }

    latest_batch = db.scalar(select(IngestionBatch).where(IngestionBatch.knowledge_base_id == knowledge_base_id).order_by(IngestionBatch.created_at.desc()))
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
            "partition": None,
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
