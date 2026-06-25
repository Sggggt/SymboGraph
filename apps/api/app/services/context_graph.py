from __future__ import annotations

import asyncio
import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    Chunk,
    ChunkContextText,
    ChunkCoordinate,
    ChunkRelationEdge,
    ChunkRelationGraphState,
    ChunkSpan,
    ChunkStructureEdge,
    ChunkStructureMapping,
    ChunkStructureNode,
    ChunkVersion,
    CoarseConcept,
    CoarseConceptDefinition,
    CoarseConceptEdge,
    CoarseConceptMembership,
    CoarseConceptState,
    ContextGraphFreshness,
    ContextGraphState,
    ContextPackage,
    Document,
    DocumentVersion,
    RQPrefix,
    RQPrefixDiagnostic,
    RQPrefixMembership,
    GraphRetrievalStep,
    IngestionCompensationLog,
    KnowledgeBase,
    MidConcept,
    MidConceptDefinition,
    MidConceptEdge,
    MidConceptMembership,
    MidConceptState,
    RetrievalTrace,
    VectorRecord,
)
from app.schemas import RetrievalGranularity, SearchFilters
from app.services.chunking import (
    CHUNK_SCHEMA_VERSION,
    CURRENT_EMBEDDING_TEXT_VERSION,
    FixedChunk,
    FixedTokenChunker,
    PreparedDocument,
    build_contextual_text,
    contextual_text_hash,
    rough_token_count,
    stable_hash,
)
from app.services.cancellation import ensure_not_cancelled
from app.services.embeddings import ChatProvider, EmbeddingProvider, FallbackDisabledError, is_degraded_mode
from app.services.model_output import coerce_confidence
from app.services.parsers import ParsedSection
from app.services.vector_store import VectorStore


RELATION_PROTOCOL_VERSION = "dense_only_chunk_relation_graph_v3"
MID_CONCEPT_PROMPT_VERSION = "mid_concept_definition_v1"
COARSE_CONCEPT_PROMPT_VERSION = "coarse_concept_definition_v1"
CONTEXT_GRAPH_PROTOCOL_VERSION = "context_graph_v1"
ANSWER_PROMPT_PROTOCOL_VERSION = "context_graph_answer_v1"
EDGE_DISTANCE_PROTOCOL_VERSION = "edge_distance_log_raw_strength_v1"
EDGE_PROJECTION_PROTOCOL_VERSION = "edge_projection_support_ids_v1"
TRAVERSAL_PROTOCOL_VERSION = "staged_layered_traversal_v2"
CONCEPT_I18N_PROTOCOL_VERSION = "concept_i18n_bilingual_v1"
QUERY_FACET_PROTOCOL_VERSION = "query_facet_packet_v1"

QUERY_FACET_MAX_GROUPS = 12
QUERY_FACET_MAX_ALIASES = 8
QUERY_FACET_MAX_REQUIRED = 8
QUERY_FACET_MAX_TERMS = 48
QUERY_FACET_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "why",
    "\u4e00\u4e0b",
    "\u4e2d",
    "\u4e3a\u4ec0\u4e48",
    "\u4e48",
    "\u4e86",
    "\u4ec0\u4e48",
    "\u4ed6",
    "\u4f60",
    "\u4f60\u4eec",
    "\u5177\u4f53",
    "\u5185",
    "\u5417",
    "\u5462",
    "\u548c",
    "\u54ea",
    "\u54ea\u4e2a",
    "\u5728",
    "\u5e2e",
    "\u5e2e\u6211",
    "\u600e\u4e48",
    "\u6211",
    "\u6211\u4eec",
    "\u628a",
    "\u6750\u6599",
    "\u6837",
    "\u8bb2",
    "\u8bf4",
    "\u8bf7",
    "\u8ddf",
    "\u8fd9",
    "\u8fd9\u4e2a",
    "\u91cc",
    "\u91cc\u9762",
    "\u7684",
    "\u7ed9",
}


@dataclass(frozen=True)
class RelationEdgeCandidate:
    source_chunk_id: str
    target_chunk_id: str
    edge_type: str
    raw_strength: float
    distance: float
    features_json: dict[str, Any]
    is_bridge: bool = False


async def gather_bounded(items: list[Any], limit: int, fn: Any, on_result: Any | None = None) -> list[Any]:
    if not items:
        return []
    worker_count = min(len(items), max(1, int(limit or 1)))
    queue: asyncio.Queue[tuple[int, Any] | None] = asyncio.Queue()
    results: list[Any] = [None] * len(items)
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def run_worker() -> None:
        while True:
            payload = await queue.get()
            try:
                if payload is None:
                    return
                index, item = payload
                result = await fn(item)
                results[index] = result
                if on_result is not None:
                    callback_result = on_result(result)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
            finally:
                queue.task_done()

    workers = [asyncio.create_task(run_worker()) for _ in range(worker_count)]
    await queue.join()
    for worker in workers:
        await worker
    return results


def runtime_settings_snapshot() -> dict[str, Any]:
    from app.services.runtime_settings import model_settings_payload

    snapshot = dict(model_settings_payload())
    snapshot.pop("runtime_settings_version", None)
    return snapshot


def runtime_settings_state_hash() -> str:
    return stable_hash(runtime_settings_snapshot())


def agent_operating_envelope(settings: Any | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return {
        "agent_coarse_total_budget": int(settings.agent_coarse_total_budget),
        "agent_mid_per_coarse_budget": int(settings.agent_mid_per_coarse_budget),
        "agent_mid_top_k": int(settings.agent_mid_top_k),
        "agent_chunk_per_mid_budget": int(settings.agent_chunk_per_mid_budget),
        "agent_chunk_top_k": int(settings.agent_chunk_top_k),
        "max_depth_per_layer": int(settings.agent_max_depth_per_layer),
        "max_labels_per_node": int(settings.agent_max_labels_per_node),
        "max_edge_reuse": int(settings.agent_max_edge_reuse),
        "max_cycle_reward_per_path": float(settings.agent_max_cycle_reward_per_path),
        "cycle_reward_distance_threshold": float(settings.agent_cycle_reward_distance_threshold),
        "path_distance_green_threshold": float(settings.agent_path_distance_green_threshold),
        "path_distance_gray_threshold": float(settings.agent_path_distance_gray_threshold),
        "path_distance_hard_threshold": float(settings.agent_path_distance_hard_threshold),
        "candidate_pool_dedupe_budget": int(settings.candidate_pool_dedupe_budget),
        "structure_restore_budget": int(settings.agent_structure_restore_budget),
        "context_package_token_budget": int(settings.context_package_token_budget),
        "context_path_summary_budget": int(settings.context_path_summary_budget),
        "planning_round_budget": int(settings.agent_planning_round_budget),
        "max_typed_actions_per_round": int(settings.agent_max_typed_actions_per_round),
        "repair_round_budget": int(settings.agent_repair_round_budget),
        "verification_budget": int(settings.agent_verification_budget),
        "allowed_relation_types": [
            "dense_semantic",
            "dense_cross_document_bridge",
            "dense_cross_language_bridge",
        ],
        "required_restore_modes": ["previous_next", "parent_structure", "bridge_chunks"],
    }


def agent_operating_envelope_state_hash() -> str:
    return stable_hash(agent_operating_envelope())


RESULT_TOP_K_MAX = 50


def resolve_result_top_k(top_k: int | None = None) -> int:
    value = int(top_k if top_k is not None else get_settings().retrieval_result_top_k_default)
    if value < 1 or value > RESULT_TOP_K_MAX:
        raise ValueError(f"result top_k must be between 1 and {RESULT_TOP_K_MAX}")
    return value


def qdrant_collection_name(*, embedding_model: str, embedding_text_version: str, chunk_schema_version: str) -> str:
    raw = f"symbograph_{embedding_model}_{embedding_text_version}_{chunk_schema_version}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_").lower()[:180]


def tokenize_for_search_terms(text: str) -> list[str]:
    from app.services.chinese_text import tokenize_for_retrieval

    return tokenize_for_retrieval(text or "")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left)) or 1.0
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


def normalized_strength(value: float) -> float:
    return round(max(1e-6, min(1.0, float(value))), 6)


def distance_from_strength(raw_strength: float) -> float:
    distance = -math.log(max(1e-6, normalized_strength(raw_strength)))
    return 0.0 if abs(distance) < 1e-9 else round(distance, 6)


ENTRY_SEED_STRENGTH_CAPS = {
    "dense_entry": 0.97,
    "rq_membership_entry": 0.88,
    "mid_drilldown_entry": 0.82,
    "coarse_to_mid_drilldown_entry": 0.72,
}


def calibrated_entry_seed_strength(raw_strength: float, role: str) -> float:
    strength = normalized_strength(raw_strength)
    cap = ENTRY_SEED_STRENGTH_CAPS.get(role, 0.9)
    return normalized_strength(min(strength, cap))


def edge_distance_protocol_hash() -> str:
    return stable_hash({"protocol": EDGE_DISTANCE_PROTOCOL_VERSION, "formula": "distance=-log(max(epsilon,raw_strength))"})


def edge_projection_protocol_hash() -> str:
    return stable_hash({"protocol": EDGE_PROJECTION_PROTOCOL_VERSION, "support": ["chunk_edges", "mid_edges"]})


def traversal_protocol_hash() -> str:
    return stable_hash({"protocol": TRAVERSAL_PROTOCOL_VERSION, "queue_key": ["uncovered_facets", "distance_minus_reward", "depth", "negative_evidence_roles"]})


def query_facet_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol": QUERY_FACET_PROTOCOL_VERSION,
            "fields": ["terms", "required_facets", "facet_groups", "drop_terms", "answer_shape", "intent"],
        }
    )


def context_graph_cache_key_components(
    *,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters | dict[str, Any] | None,
    context_state: ContextGraphState | None,
    retrieval_mode: str,
    retrieval_granularity: RetrievalGranularity = "mid",
    result_top_k: int | None = None,
    conversation_state_scope_hash: str | None = None,
    query_facets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filter_payload = filters.model_dump() if isinstance(filters, SearchFilters) else dict(filters or {})
    return {
        "knowledge_base_id": knowledge_base_id,
        "query": query,
        "filters": filter_payload,
        "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
        "chunk_scope_hash": context_state.chunk_scope_hash if context_state else None,
        "structure_graph_hash": context_state.structure_graph_hash if context_state else None,
        "chunk_relation_graph_hash": context_state.chunk_relation_graph_hash if context_state else None,
        "rq_membership_hash": context_state.rq_membership_hash if context_state else None,
        "mid_concept_hash": context_state.mid_concept_hash if context_state else None,
        "coarse_concept_hash": context_state.coarse_concept_hash if context_state else None,
        "edge_distance_protocol_hash": edge_distance_protocol_hash(),
        "edge_projection_protocol_hash": edge_projection_protocol_hash(),
        "traversal_protocol_hash": traversal_protocol_hash(),
        "query_facet_protocol_hash": query_facet_protocol_hash(),
        "query_facets_hash": stable_hash(query_facets or {}),
        "runtime_settings_hash": runtime_settings_state_hash(),
        "policy_state_hash": context_state.policy_state_hash if context_state else None,
        "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
        "conversation_state_scope_hash": conversation_state_scope_hash or stable_hash({"conversation_state": "none"}),
        "prompt_protocol_hash": context_state.prompt_protocol_hash if context_state else None,
        "retrieval_mode": retrieval_mode,
        "retrieval_granularity": retrieval_granularity,
        "result_top_k": result_top_k,
    }


def context_graph_cache_key(
    *,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters | dict[str, Any] | None,
    context_state: ContextGraphState | None,
    retrieval_mode: str,
    retrieval_granularity: RetrievalGranularity = "mid",
    result_top_k: int | None = None,
    conversation_state_scope_hash: str | None = None,
    query_facets: dict[str, Any] | None = None,
) -> str:
    return stable_hash(
        context_graph_cache_key_components(
            knowledge_base_id=knowledge_base_id,
            query=query,
            filters=filters,
            context_state=context_state,
            retrieval_mode=retrieval_mode,
            retrieval_granularity=retrieval_granularity,
            result_top_k=result_top_k,
            conversation_state_scope_hash=conversation_state_scope_hash,
            query_facets=query_facets,
        )
    )


def active_chunks_query(knowledge_base_id: str):
    return (
        select(Chunk)
        .where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active")
        .order_by(Chunk.document_id.asc(), Chunk.chunk_index.asc())
    )


def compute_chunk_scope_hash(chunks: list[Chunk]) -> str:
    return stable_hash(
        [
            {
                "id": chunk.id,
                "document_version_id": chunk.document_version_id,
                "chunk_version": chunk.chunk_version,
                "text_hash": chunk.text_hash,
                "span": [chunk.char_start, chunk.char_end],
            }
            for chunk in chunks
        ]
    )


def latest_context_graph_state(db: Session, knowledge_base_id: str) -> ContextGraphState | None:
    return db.scalar(
        select(ContextGraphState)
        .where(ContextGraphState.knowledge_base_id == knowledge_base_id, ContextGraphState.state == "active")
        .order_by(ContextGraphState.created_at.desc())
    )


def latest_relation_state(db: Session, knowledge_base_id: str) -> ChunkRelationGraphState | None:
    return db.scalar(
        select(ChunkRelationGraphState)
        .where(ChunkRelationGraphState.knowledge_base_id == knowledge_base_id, ChunkRelationGraphState.state == "active")
        .order_by(ChunkRelationGraphState.created_at.desc())
    )


def latest_mid_state(db: Session, knowledge_base_id: str) -> MidConceptState | None:
    return db.scalar(
        select(MidConceptState)
        .where(MidConceptState.knowledge_base_id == knowledge_base_id, MidConceptState.state == "active")
        .order_by(MidConceptState.created_at.desc())
    )


def latest_coarse_state(db: Session, knowledge_base_id: str) -> CoarseConceptState | None:
    return db.scalar(
        select(CoarseConceptState)
        .where(CoarseConceptState.knowledge_base_id == knowledge_base_id, CoarseConceptState.state == "active")
        .order_by(CoarseConceptState.created_at.desc())
    )


def deactivate_derived_states(db: Session, knowledge_base_id: str) -> None:
    for model in (ChunkRelationGraphState, MidConceptState, CoarseConceptState, ContextGraphState):
        db.execute(update(model).where(model.knowledge_base_id == knowledge_base_id, model.state == "active").values(state="inactive"))


def deactivate_document_chunks(db: Session, document_id: str) -> None:
    db.execute(update(Chunk).where(Chunk.document_id == document_id, Chunk.state == "active").values(state="inactive"))


def write_chunk_version_state(
    db: Session,
    *,
    knowledge_base_id: str,
    chunk_version: int,
    chunks: list[Chunk],
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkVersion:
    state_hash = stable_hash([chunk.id for chunk in chunks] + [chunk_version, chunk_size, chunk_overlap])
    db.execute(
        update(ChunkVersion)
        .where(ChunkVersion.knowledge_base_id == knowledge_base_id, ChunkVersion.chunk_version != chunk_version, ChunkVersion.state == "active")
        .values(state="inactive")
    )
    existing = db.scalar(select(ChunkVersion).where(ChunkVersion.knowledge_base_id == knowledge_base_id, ChunkVersion.chunk_version == chunk_version))
    if existing is not None:
        existing.state_hash = state_hash
        existing.stats_json = chunk_distribution_stats(chunks)
        existing.state = "active"
        return existing
    state = ChunkVersion(
        knowledge_base_id=knowledge_base_id,
        chunk_version=chunk_version,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        state_hash=state_hash,
        stats_json=chunk_distribution_stats(chunks),
        diagnostics_json={},
    )
    db.add(state)
    return state


def chunk_distribution_stats(chunks: list[Chunk]) -> dict[str, Any]:
    token_counts = [max(0, (chunk.token_end or 0) - (chunk.token_start or 0)) for chunk in chunks]
    if not token_counts:
        return {"chunk_count": 0, "token_count": {"min": 0, "max": 0, "avg": 0}}
    return {
        "chunk_count": len(chunks),
        "token_count": {
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": round(sum(token_counts) / len(token_counts), 3),
        },
        "page_mapped_chunks": sum(1 for chunk in chunks if chunk.page_start is not None),
        "section_mapped_chunks": sum(1 for chunk in chunks if chunk.section_path),
    }


def write_chunks_and_structure(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    document: Document,
    version: DocumentVersion,
    sections: list[ParsedSection],
    chunk_version: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    chunker = FixedTokenChunker(
        chunk_size=chunk_size or settings.fixed_chunk_size_tokens,
        overlap=chunk_overlap if chunk_overlap is not None else settings.fixed_chunk_overlap_tokens,
    )
    fixed_chunks, prepared = chunker.split_sections(sections, title=document.title)
    deactivate_document_chunks(db, document.id)
    db.flush()

    nodes = write_structure_graph(db, knowledge_base=knowledge_base, document=document, version=version, prepared=prepared)
    chunks: list[Chunk] = []
    for fixed in fixed_chunks:
        chunk = Chunk(
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=chunk_version,
            chunk_index=fixed.chunk_index,
            token_start=fixed.token_start,
            token_end=fixed.token_end,
            char_start=fixed.char_start,
            char_end=fixed.char_end,
            text=fixed.text,
            text_hash=fixed.text_hash,
            section_path=fixed.section_path,
            page_start=fixed.page_start,
            page_end=fixed.page_end,
            metadata_json=fixed.metadata,
            state="active",
        )
        db.add(chunk)
        chunks.append(chunk)
    db.flush()
    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk.previous_chunk_id = chunks[index - 1].id
        if index + 1 < len(chunks):
            chunk.next_chunk_id = chunks[index + 1].id
        db.add(
            ChunkSpan(
                chunk_id=chunk.id,
                document_version_id=version.id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                section_path=chunk.section_path,
                metadata_json={"text_hash": chunk.text_hash},
            )
        )
        db.add(
            ChunkCoordinate(
                chunk_id=chunk.id,
                document_version_id=version.id,
                page_number=chunk.page_start,
                page_range_json={"start": chunk.page_start, "end": chunk.page_end},
                region_type="chunk",
                confidence=1.0 if chunk.page_start is not None else 0.7,
            )
        )
        write_structure_mappings_for_chunk(db, chunk=chunk, nodes=nodes)
    write_chunk_version_state(
        db,
        knowledge_base_id=knowledge_base.id,
        chunk_version=chunk_version,
        chunks=chunks,
        chunk_size=chunker.chunk_size,
        chunk_overlap=chunker.overlap,
    )
    return chunks


def write_structure_graph(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    document: Document,
    version: DocumentVersion,
    prepared: PreparedDocument,
) -> list[ChunkStructureNode]:
    def add_node(
        *,
        node_type: str,
        parent_id: str | None,
        depth: int,
        title: str | None,
        char_start: int | None,
        char_end: int | None,
        page_number: int | None,
        path: str | None,
        bbox: dict[str, Any] | None = None,
        layout: dict[str, Any] | None = None,
        previous_sibling_id: str | None = None,
    ) -> ChunkStructureNode:
        node = ChunkStructureNode(
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type=node_type,
            parent_id=parent_id,
            previous_sibling_id=previous_sibling_id,
            depth=depth,
            title=title,
            char_start=char_start,
            char_end=char_end,
            page_number=page_number,
            bbox_json=bbox or {},
            path=path,
            layout_json=layout or {},
        )
        db.add(node)
        db.flush()
        return node

    def add_edge(
        source: ChunkStructureNode,
        target: ChunkStructureNode,
        edge_type: str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        db.add(
            ChunkStructureEdge(
                knowledge_base_id=knowledge_base.id,
                document_version_id=version.id,
                source_node_id=source.id,
                target_node_id=target.id,
                edge_type=edge_type,
                weight=weight,
                metadata_json=metadata or {},
            )
        )

    document_node = ChunkStructureNode(
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="document",
        title=document.title,
        char_start=0,
        char_end=len(prepared.text),
        depth=0,
        path=document.title,
        layout_json={"source_path": document.source_path},
    )
    db.add(document_node)
    db.flush()
    nodes = [document_node]
    section_items = [item for item in prepared.section_offsets if int(item.get("char_end") or 0) > int(item.get("char_start") or 0)]
    page_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in section_items:
        page_number = item.get("page_number")
        page_groups[int(page_number) if page_number is not None else 1].append(item)
    page_nodes: dict[int, ChunkStructureNode] = {}
    for page_number, items in sorted(page_groups.items()):
        synthetic_page = all(item.get("page_number") is None for item in items)
        char_start = min(int(item["char_start"]) for item in items)
        char_end = max(int(item["char_end"]) for item in items)
        page_node = add_node(
            node_type="page",
            parent_id=document_node.id,
            depth=1,
            title=f"Page {page_number}" if not synthetic_page else "Source text page",
            char_start=char_start,
            char_end=char_end,
            page_number=None if synthetic_page else page_number,
            path=f"{document.title} / page:{page_number}",
            bbox=_normalized_bbox(page_number=page_number, region_index=0, region_count=1),
            layout={"synthetic_page": synthetic_page, "source": "parser_page_number_or_text_fallback"},
        )
        add_edge(document_node, page_node, "parent_child", metadata={"node_type": "page", "synthetic_page": synthetic_page})
        nodes.append(page_node)
        page_nodes[page_number] = page_node

    previous_section: ChunkStructureNode | None = None
    previous_block_by_section: ChunkStructureNode | None = None
    sections_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in section_items:
        page_number = item.get("page_number")
        sections_by_page[int(page_number) if page_number is not None else 1].append(item)
    for item in section_items:
        item_page_number = item.get("page_number")
        page_key = int(item_page_number) if item_page_number is not None else 1
        page_node = page_nodes[page_key]
        page_sections = sections_by_page[page_key]
        region_index = max(0, page_sections.index(item))
        region_node = add_node(
            node_type="region",
            parent_id=page_node.id,
            depth=2,
            title=str(item.get("section_path") or item.get("title") or "Region"),
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            page_number=item_page_number,
            path=f"{page_node.path} / region:{region_index + 1}",
            bbox=_normalized_bbox(page_number=page_key, region_index=region_index, region_count=max(len(page_sections), 1)),
            layout={"region_index": region_index, "region_count": len(page_sections), "source": "section_span_projection"},
        )
        add_edge(page_node, region_node, "parent_child", metadata={"node_type": "region"})
        nodes.append(region_node)
        section_node = add_node(
            node_type="section",
            parent_id=document_node.id,
            previous_sibling_id=previous_section.id if previous_section else None,
            depth=2,
            title=str(item.get("section_path") or item.get("title") or document.title),
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            page_number=item.get("page_number"),
            path=f"{document.title} / {item.get('section_path') or item.get('title')}",
            layout=item.get("metadata") or {},
        )
        if previous_section is not None:
            previous_section.next_sibling_id = section_node.id
            add_edge(previous_section, section_node, "prev_next")
        add_edge(document_node, section_node, "parent_child", metadata={"node_type": "section"})
        add_edge(region_node, section_node, "contains_section", metadata={"page_number": item.get("page_number")})
        add_edge(section_node, region_node, "located_in_region", metadata={"page_number": item.get("page_number")})
        nodes.append(section_node)
        section_text = prepared.text[int(item["char_start"]) : int(item["char_end"])]
        block_nodes: list[ChunkStructureNode] = []
        previous_block_by_section = None
        for block_index, block in enumerate(_structure_blocks_for_section(section_text, base_offset=int(item["char_start"]), metadata=item.get("metadata") or {})):
            block_node = add_node(
                node_type=block["node_type"],
                parent_id=section_node.id,
                depth=3,
                title=block["title"],
                char_start=block["char_start"],
                char_end=block["char_end"],
                page_number=item.get("page_number"),
                path=f"{section_node.path} / {block['node_type']}:{block_index + 1}",
                bbox=_normalized_bbox(page_number=page_key, region_index=region_index, region_count=max(len(page_sections), 1), block_index=block_index, block_count=8),
                layout={
                    "source": block["source"],
                    "section_index": item.get("index"),
                    "block_index": block_index,
                    "content_flags": block.get("content_flags") or [],
                },
                previous_sibling_id=previous_block_by_section.id if previous_block_by_section else None,
            )
            if previous_block_by_section is not None:
                previous_block_by_section.next_sibling_id = block_node.id
                add_edge(previous_block_by_section, block_node, "prev_next", metadata={"scope": "section_blocks"})
            add_edge(section_node, block_node, "parent_child", metadata={"node_type": block["node_type"]})
            add_edge(region_node, block_node, "contains_block", metadata={"node_type": block["node_type"]})
            if block["node_type"] in {"table", "formula", "caption", "code_block"}:
                add_edge(page_node, block_node, "same_page_region", metadata={"node_type": block["node_type"]})
            nodes.append(block_node)
            block_nodes.append(block_node)
            previous_block_by_section = block_node
        special_nodes = [node for node in block_nodes if node.node_type in {"table", "formula", "caption"}]
        for left_index, left in enumerate(special_nodes):
            for right in special_nodes[left_index + 1 : left_index + 4]:
                add_edge(left, right, "table_formula_context", weight=0.92, metadata={"section_node_id": section_node.id})
        previous_section = section_node
    return nodes


def write_structure_mappings_for_chunk(db: Session, *, chunk: Chunk, nodes: list[ChunkStructureNode]) -> None:
    for node in nodes:
        node_start = node.char_start if node.char_start is not None else 0
        node_end = node.char_end if node.char_end is not None else 0
        overlap = max(0, min(chunk.char_end, node_end) - max(chunk.char_start, node_start))
        if overlap <= 0 and node.node_type != "document":
            continue
        denominator = max(1, chunk.char_end - chunk.char_start)
        db.add(
            ChunkStructureMapping(
                chunk_id=chunk.id,
                structure_node_id=node.id,
                document_version_id=chunk.document_version_id,
                overlap_chars=overlap,
                overlap_tokens=max(0, round(((chunk.token_end - chunk.token_start) or 0) * (overlap / denominator))),
                coverage_ratio=round(overlap / denominator, 6),
                mapping_role=_structure_mapping_role(node.node_type),
            )
        )


def _normalized_bbox(
    *,
    page_number: int | None,
    region_index: int,
    region_count: int,
    block_index: int | None = None,
    block_count: int | None = None,
) -> dict[str, Any]:
    y0 = region_index / max(region_count, 1)
    y1 = (region_index + 1) / max(region_count, 1)
    if block_index is not None and block_count:
        block_height = (y1 - y0) / max(block_count, 1)
        y0 = min(0.98, y0 + block_height * min(block_index, block_count - 1))
        y1 = min(1.0, y0 + block_height)
    return {
        "page_number": page_number,
        "x0": 0.0,
        "y0": round(y0, 6),
        "x1": 1.0,
        "y1": round(max(y1, y0 + 0.01), 6),
        "coordinate_system": "normalized_page_region_v1",
        "synthetic": True,
    }


def _structure_mapping_role(node_type: str) -> str:
    return {
        "document": "document",
        "section": "parent_section",
        "page": "page",
        "region": "region",
        "paragraph": "paragraph",
        "list": "list_context",
        "table": "table_formula_context",
        "formula": "table_formula_context",
        "caption": "caption_context",
        "code_block": "code_context",
    }.get(node_type, "overlap")


def _structure_blocks_for_section(text: str, *, base_offset: int, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(node_type: str, start: int, end: int, source: str, sample: str) -> None:
        if end <= start:
            return
        title = re.sub(r"\s+", " ", sample or "").strip()[:96] or node_type
        candidates.append(
            {
                "node_type": node_type,
                "char_start": base_offset + start,
                "char_end": base_offset + end,
                "title": title,
                "source": source,
                "content_flags": sorted(_content_flags_for_text(sample)),
            }
        )

    for match in re.finditer(r"```.*?```", text or "", flags=re.DOTALL):
        add("code_block", match.start(), match.end(), "fenced_code", match.group(0))
    for match in re.finditer(r"(?m)^(?:\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\(|class\s+\w+\(|print\(|[A-Za-z_][\w_]*\s*=).*(?:\n|$)){2,}", text or ""):
        add("code_block", match.start(), match.end(), "code_line_heuristic", match.group(0))
    for match in re.finditer(r"\$\$.*?\$\$", text or "", flags=re.DOTALL):
        add("formula", match.start(), match.end(), "display_math", match.group(0))
    for match in re.finditer(r"(?m)^\s*(?:[A-Za-z][\w\s()]*\s*[=:]\s*[^:\n]{0,180}(?:\\frac|\\sum|\\prod|P\(|\^|_[A-Za-z0-9]|\d+\s*[+\-*/=]\s*\d+).*)$", text or ""):
        add("formula", match.start(), match.end(), "formula_line_heuristic", match.group(0))
    for match in re.finditer(r"(?m)^(?:\|.*\|\s*\n?){2,}", text or ""):
        add("table", match.start(), match.end(), "markdown_table", match.group(0))
    for match in re.finditer(r"(?m)^(?:[^\n\t]+\t[^\n]+\n?){2,}", text or ""):
        add("table", match.start(), match.end(), "tabular_text", match.group(0))
    for match in re.finditer(r"(?im)^\s*(?:figure|fig\.|table|caption|\u56fe|\u8868)\s*[:\d\uff1a.-].+$", text or ""):
        add("caption", match.start(), match.end(), "caption_line", match.group(0))
    for match in re.finditer(r"(?m)^(?:\s*(?:[-*+]|\d+[.)])\s+.+(?:\n|$)){1,}", text or ""):
        add("list", match.start(), match.end(), "list_line_group", match.group(0))
    for match in re.finditer(r"(?s)(?:^|\n{2,})([^\n].*?)(?=\n{2,}|$)", text or ""):
        start, end = match.start(1), match.end(1)
        paragraph = text[start:end]
        if paragraph.strip():
            add("paragraph", start, end, "paragraph_split", paragraph)

    if metadata.get("has_table"):
        add("table", 0, len(text or ""), "parser_metadata_table", text or "")
    if metadata.get("has_formula"):
        add("formula", 0, len(text or ""), "parser_metadata_formula", text or "")
    if metadata.get("has_caption"):
        add("caption", 0, len(text or ""), "parser_metadata_caption", text or "")
    if metadata.get("content_kind") == "code":
        add("code_block", 0, len(text or ""), "parser_metadata_code", text or "")

    candidates.sort(key=lambda item: (item["char_start"], item["char_end"], item["node_type"]))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in candidates:
        key = (str(item["node_type"]), int(item["char_start"]), int(item["char_end"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped or [
        {
            "node_type": "paragraph",
            "char_start": base_offset,
            "char_end": base_offset + len(text or ""),
            "title": re.sub(r"\s+", " ", text or "").strip()[:96] or "paragraph",
            "source": "section_fallback_paragraph",
            "content_flags": sorted(_content_flags_for_text(text or "")),
        }
    ]


def _content_flags_for_text(text: str) -> set[str]:
    flags: set[str] = set()
    if re.search(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", text or ""):
        flags.add("list")
    if "|" in (text or "") or "\t" in (text or ""):
        flags.add("table")
    if re.search(r"(?:\\frac|\\sum|\\prod|P\(|\^|=|[+\-*/]\s*\d)", text or ""):
        flags.add("formula")
    if re.search(r"(?im)^\s*(?:figure|fig\.|table|caption|\u56fe|\u8868)\s*[:\d\uff1a.-]", text or ""):
        flags.add("caption")
    if re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\(|class\s+\w+\(|```)", text or ""):
        flags.add("code")
    return flags


async def write_contextual_indexes(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    chunks: list[Chunk],
    local_hints: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not chunks:
        return {"chunks": 0, "vectors": 0}
    settings = get_settings()
    local_hints = local_hints or {}
    documents = {doc.id: doc for doc in db.scalars(select(Document).where(Document.id.in_({chunk.document_id for chunk in chunks}))).all()}
    context_texts: list[str] = []
    context_rows: list[ChunkContextText] = []
    for chunk in chunks:
        document = documents[chunk.document_id]
        contextual = build_contextual_text(
            document_title=document.title,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            raw_text=chunk.text,
            local_hint=local_hints.get(chunk.id),
            metadata=chunk.metadata_json or {},
        )
        row = ChunkContextText(
            chunk_id=chunk.id,
            raw_text=chunk.text,
            contextual_text=contextual,
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
            context_hash=contextual_text_hash(contextual),
            metadata_json={"document_title": document.title, "source_path": document.source_path},
        )
        db.add(row)
        context_rows.append(row)
        context_texts.append(contextual)
    db.flush()

    vectors = await EmbeddingProvider().embed_texts(context_texts, text_type="document")
    collection_name = qdrant_collection_name(
        embedding_model=settings.embedding_model,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )
    points = []
    vector_records: list[VectorRecord] = []
    for chunk, context_row, vector in zip(chunks, context_rows, vectors):
        document = documents[chunk.document_id]
        payload = {
            "knowledge_base_id": knowledge_base.id,
            "chunk_id": chunk.id,
            "document_id": document.id,
            "document_version_id": chunk.document_version_id,
            "chunk_version": chunk.chunk_version,
            "section_path": chunk.section_path,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "text_hash": chunk.text_hash,
        }
        payload_hash = stable_hash(payload)
        record = VectorRecord(
            knowledge_base_id=knowledge_base.id,
            chunk_id=chunk.id,
            qdrant_point_id=chunk.id,
            collection_name=collection_name,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimensions,
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
            payload_hash=payload_hash,
            vector_status="pending",
            diagnostics_json={"embedding_vector": vector, "context_hash": context_row.context_hash},
        )
        db.add(record)
        vector_records.append(record)
        points.append({"id": chunk.id, "vector": vector, "payload": payload})
    db.flush()
    try:
        await VectorStore(knowledge_base_name=knowledge_base.name, collection_name=collection_name).async_upsert(points)
    except Exception as exc:
        db.add(
            IngestionCompensationLog(
                job_id=None,
                knowledge_base_id=knowledge_base.id,
                operation="qdrant_upsert",
                target_ids_json=[point["id"] for point in points],
                payload_json={
                    "collection_name": collection_name,
                    "embedding_model": settings.embedding_model,
                    "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
                    "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                },
                status="failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )
        db.flush()
        raise
    for record in vector_records:
        record.vector_status = "ready"

    db.flush()
    return {"chunks": len(chunks), "vectors": len(points), "collection_name": collection_name}


def vector_for_chunk(db: Session, chunk_id: str) -> list[float]:
    record = db.scalar(
        select(VectorRecord)
        .where(VectorRecord.chunk_id == chunk_id, VectorRecord.vector_status == "ready")
        .order_by(VectorRecord.created_at.desc())
    )
    if record is None:
        return []
    vector = (record.diagnostics_json or {}).get("embedding_vector") or []
    return [float(value) for value in vector if isinstance(value, (int, float))]


async def rebuild_context_graph(
    db: Session,
    knowledge_base_id: str,
    *,
    batch_id: str | None = None,
    state_scope: str = "active",
    operating_point: dict[str, Any] | None = None,
    shadow_metadata: dict[str, Any] | None = None,
    emit_heartbeats: bool = True,
    chunk_version_incremented: bool = False,
) -> ContextGraphState:
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    if not chunks:
        raise RuntimeError("Cannot rebuild context graph without current chunks")
    if state_scope not in {"active", "shadow"}:
        raise ValueError(f"Unsupported context graph state scope: {state_scope}")
    heartbeat_batch_id = batch_id if emit_heartbeats else None
    previous_relation_state = latest_relation_state(db, knowledge_base_id) if state_scope == "active" else None
    if emit_heartbeats:
        context_graph_batch_heartbeat(batch_id, "starting", {"chunks": len(chunks)})
    if state_scope == "active":
        deactivate_derived_states(db, knowledge_base_id)
    if emit_heartbeats:
        context_graph_batch_heartbeat(batch_id, "chunk_relation", {"chunks": len(chunks)})
    relation_state = build_chunk_relation_graph(
        db,
        knowledge_base_id,
        chunks,
        batch_id=heartbeat_batch_id,
        state_scope=state_scope,
        operating_point=operating_point,
        previous_operating_point=dict(previous_relation_state.graph_operating_point_json or {}) if previous_relation_state else None,
        auto_tpe_enabled_for_version=state_scope == "active" and chunk_version_incremented,
        shadow_metadata=shadow_metadata,
    )
    if emit_heartbeats:
        context_graph_batch_heartbeat(batch_id, "mid_concepts", dict(relation_state.stats_json or {}))
    mid_state = await build_mid_concept_graph(db, knowledge_base_id, relation_state, batch_id=heartbeat_batch_id, state_scope=state_scope, shadow_metadata=shadow_metadata)
    if emit_heartbeats:
        context_graph_batch_heartbeat(
            batch_id,
            "coarse_concepts",
            {
                "mid_concepts": (mid_state.stats_json or {}).get("mid_concept_count"),
                "projected_rq_l3_prefixes": (mid_state.stats_json or {}).get("projected_rq_l3_prefixes"),
                "mid_edge_count": (mid_state.stats_json or {}).get("mid_edge_count"),
            },
        )
    coarse_state = await build_coarse_concept_graph(db, knowledge_base_id, mid_state, batch_id=heartbeat_batch_id, state_scope=state_scope, shadow_metadata=shadow_metadata)
    if emit_heartbeats:
        context_graph_batch_heartbeat(batch_id, "context_state", dict(coarse_state.diagnostics_json or {}))
    context_state = write_context_graph_state(db, knowledge_base_id, relation_state, mid_state, coarse_state, chunks, state_scope=state_scope, shadow_metadata=shadow_metadata)
    db.flush()
    if emit_heartbeats:
        context_graph_batch_heartbeat(batch_id, "completed", dict(context_state.stats_json or {}))
    return context_state


def context_graph_batch_heartbeat(batch_id: str | None, phase: str, metrics: dict[str, Any] | None = None) -> None:
    if not batch_id:
        return
    from app.db import SessionLocal
    from app.models import IngestionBatch
    from app.services.ingestion_logs import emit_ingestion_log

    metrics_payload = dict(metrics or {})
    try:
        with SessionLocal() as session:
            batch = session.get(IngestionBatch, batch_id)
            if batch is None:
                return
            now = datetime.utcnow()
            stats = dict(batch.stats or {})
            stats["phase"] = "context_graph"
            stats["context_graph_phase"] = phase
            stats["context_graph_heartbeat_at"] = now.isoformat()
            if metrics is not None:
                stats["context_graph_metrics"] = metrics_payload
            batch.stats = stats
            batch.heartbeat_at = now
            session.commit()
    except Exception:
        pass
    reserved_log_keys = {"batch_id", "event", "log_id", "message", "phase", "timestamp", "context_graph_phase"}
    log_payload = {key: value for key, value in metrics_payload.items() if key not in reserved_log_keys}
    log_payload["phase"] = f"context_graph:{phase}"
    log_payload["context_graph_phase"] = phase
    emit_ingestion_log(batch_id, "batch_graph_progress", context_graph_heartbeat_message(phase, metrics_payload), **log_payload)


def context_graph_heartbeat_message(phase: str, metrics: dict[str, Any]) -> str:
    phase_labels = {
        "starting": "上下文图谱初始化",
        "chunk_relation": "片段关系图",
        "chunk_relation:chunk_edges": "关系边生成",
        "chunk_relation:rq_prefixes": "RQ 前缀与归属",
        "mid_concepts": "中粒度概念",
        "coarse_concepts": "粗粒度概念",
        "context_state": "Context Graph 状态",
        "completed": "图谱闭环完成",
    }
    translation_labels = {
        "concept_i18n": "节点双语派生",
        "edge_i18n": "关系双语派生",
    }
    phase_label = phase_labels.get(phase, phase.replace("_", " "))
    translation_phase = metrics.get("translation_phase")
    if isinstance(translation_phase, str) and translation_phase:
        translation_label = translation_labels.get(translation_phase, translation_phase.replace("_", " "))
        item_count = metrics.get("translation_items")
        item_suffix = f"，{item_count} 项" if isinstance(item_count, (int, float)) else ""
        if metrics.get("translation_status") == "disabled" or metrics.get("translation_enabled") is False:
            skip_suffix = f" {item_count} 项" if isinstance(item_count, (int, float)) else ""
            return f"{phase_label}：{translation_label}已关闭，跳过{skip_suffix}"
        return f"{phase_label}：{translation_label}{item_suffix}"
    return f"{phase_label}：进度更新"


def build_chunk_relation_graph(
    db: Session,
    knowledge_base_id: str,
    chunks: list[Chunk],
    *,
    batch_id: str | None = None,
    state_scope: str = "active",
    operating_point: dict[str, Any] | None = None,
    previous_operating_point: dict[str, Any] | None = None,
    auto_tpe_enabled_for_version: bool = False,
    shadow_metadata: dict[str, Any] | None = None,
) -> ChunkRelationGraphState:
    scope_hash = compute_chunk_scope_hash(chunks)
    vectors = {chunk.id: vector_for_chunk(db, chunk.id) for chunk in chunks}
    tpe_context: dict[str, Any] = {}
    if operating_point is None:
        fallback_operating_point = previous_operating_point or dense_graph_operating_point()
        if state_scope == "active":
            from app.services.auto_tpe import select_auto_tpe_operating_point

            operating_point, tpe_context = select_auto_tpe_operating_point(
                db,
                knowledge_base_id,
                chunks,
                vectors,
                fallback_operating_point=fallback_operating_point,
                batch_id=batch_id,
                chunk_version_incremented=auto_tpe_enabled_for_version,
            )
        else:
            operating_point = fallback_operating_point
    state_diagnostics = dict(shadow_metadata or {})
    if tpe_context:
        state_diagnostics["auto_tpe"] = tpe_context
    if state_scope == "shadow":
        state_diagnostics["shadow_scope"] = True
    runtime_hash = stable_hash(runtime_settings_snapshot())
    graph_state = ChunkRelationGraphState(
        knowledge_base_id=knowledge_base_id,
        chunk_version=max(chunk.chunk_version for chunk in chunks),
        scope_hash=scope_hash,
        state_hash=stable_hash({"scope": scope_hash, "protocol": RELATION_PROTOCOL_VERSION, "operating_point": operating_point}),
        graph_operating_point_hash=stable_hash(operating_point),
        graph_operating_point_json=operating_point,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        relation_protocol_version=RELATION_PROTOCOL_VERSION,
        edge_distance_protocol_hash=edge_distance_protocol_hash(),
        edge_type_calibration_protocol_hash=edge_type_calibration_protocol_hash(),
        runtime_settings_hash=runtime_hash,
        auto_tpe_run_id=tpe_context.get("run_id"),
        auto_tpe_best_trial_id=tpe_context.get("best_trial_id"),
        active_chunk_ids_json=[chunk.id for chunk in chunks],
        stats_json={},
        diagnostics_json=state_diagnostics,
        state=state_scope,
    )
    db.add(graph_state)
    db.flush()

    edges: dict[tuple[str, str, str], ChunkRelationEdge] = {}
    ensure_not_cancelled(db, batch_id)
    context_graph_batch_heartbeat(batch_id, "chunk_relation:chunk_edges", {"chunks": len(chunks)})
    add_relation_edges(db, graph_state, chunks, vectors, edges)
    ensure_not_cancelled(db, batch_id)
    context_graph_batch_heartbeat(batch_id, "chunk_relation:rq_prefixes", {"chunk_edges": len(edges)})
    rq_prefixes = build_rq_prefixes(db, graph_state, chunks, vectors, edges)
    ensure_not_cancelled(db, batch_id)
    stats = relation_graph_stats(chunks, list(edges.values()), rq_prefixes)
    graph_state.stats_json = stats
    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        "orphan_chunk_rate": stats["orphan_chunk_rate"],
        "singleton_rate": stats["singleton_rate"],
        "bridge_edge_count": stats["bridge_edges"],
        "protocol": RELATION_PROTOCOL_VERSION,
    }
    graph_state.state_hash = stable_hash({"scope": scope_hash, "stats": stats, "rq_prefixes": [prefix.id for prefix in rq_prefixes], "operating_point": operating_point})
    for edge in edges.values():
        edge.graph_state_hash = graph_state.state_hash
    if tpe_context.get("run_id"):
        from app.services.auto_tpe import mark_auto_tpe_relation_state

        mark_auto_tpe_relation_state(db, str(tpe_context["run_id"]), graph_state.id)
    return graph_state


def relation_edge_source_algorithm(edge_type: str) -> str:
    if edge_type.startswith("rq_"):
        raise RuntimeError(f"RQ pair diagnostics are not allowed as active chunk relation edges: {edge_type}")
    if edge_type.startswith("dense"):
        return "dense_embedding"
    if edge_type.startswith("structure") or edge_type.startswith("same_"):
        raise RuntimeError(f"Structure-derived relation edges are not allowed in the active chunk relation graph: {edge_type}")
    return edge_type


def edge_type_calibration_protocol_hash() -> str:
    return stable_hash(
        {
            "protocol": "dense_edge_type_calibration_v1",
            "edge_types": ["dense_semantic", "dense_cross_document_bridge", "dense_cross_language_bridge"],
            "normalization": "type_local_min_max_then_distance",
        }
    )


def dense_graph_operating_point() -> dict[str, Any]:
    settings = get_settings()
    return {
        "protocol": "dense_dynamic_knn_bridge_quota_v1",
        "optimizer": "auto_tpe_lightweight_or_default",
        "dense_knn_k_min": int(settings.dense_knn_k_min),
        "dense_knn_k_max": int(settings.dense_knn_k_max),
        "dense_reverse_b_min_base": int(settings.dense_reverse_b_min_base),
        "dense_reverse_b_max_base": int(settings.dense_reverse_b_max_base),
        "dense_reverse_b_min_doc": int(settings.dense_reverse_b_min_doc),
        "dense_reverse_b_max_doc": int(settings.dense_reverse_b_max_doc),
        "dense_reverse_b_min_lang": int(settings.dense_reverse_b_min_lang),
        "dense_reverse_b_max_lang": int(settings.dense_reverse_b_max_lang),
        "dense_min_cosine": float(settings.dense_min_cosine),
        "dense_strong_cosine": float(settings.dense_strong_cosine),
        "cross_doc_out_quota_min": int(settings.cross_doc_out_quota_min),
        "cross_doc_out_quota_max": int(settings.cross_doc_out_quota_max),
        "cross_doc_min_cosine": float(settings.cross_doc_min_cosine),
        "cross_language_out_quota_min": int(settings.cross_language_out_quota_min),
        "cross_language_out_quota_max": int(settings.cross_language_out_quota_max),
        "cross_language_min_cosine": float(settings.cross_language_min_cosine),
    }


def add_chunk_relation_edge(
    db: Session,
    graph_state: ChunkRelationGraphState,
    source_chunk_id: str,
    target_chunk_id: str,
    edge_type: str,
    weight: float,
    features: dict | None,
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
    *,
    is_bridge: bool = False,
) -> ChunkRelationEdge | None:
    if source_chunk_id == target_chunk_id:
        return None
    left, right = sorted([source_chunk_id, target_chunk_id])
    key = (left, right, edge_type)
    raw_strength = normalized_strength(weight)
    distance = distance_from_strength(raw_strength)
    if key in edges:
        edge = edges[key]
        edge.raw_strength = max(edge.raw_strength, raw_strength)
        edge.distance = distance_from_strength(edge.raw_strength)
        edge.weight = edge.raw_strength
        edge.confidence = max(edge.confidence, edge.raw_strength)
        edge.is_bridge = edge.is_bridge or is_bridge
        edge.features_json = {**(edge.features_json or {}), **(features or {})}
        edge.normalization_stats_json = {
            **(edge.normalization_stats_json or {}),
            **((features or {}).get("normalization_stats") or {}),
        }
        edge.raw_strength_summary_json = {
            "max_raw_strength": edge.raw_strength,
            "distance": edge.distance,
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
        }
        edge.protocol_version = graph_state.relation_protocol_version
        edge.source_algorithm = relation_edge_source_algorithm(edge_type)
        edge.edge_distance_protocol_hash = edge_distance_protocol_hash()
        edge.graph_state_hash = graph_state.state_hash
        edge.source_language = (features or {}).get("source_language") or edge.source_language
        edge.target_language = (features or {}).get("target_language") or edge.target_language
        edge.is_cross_document = edge.is_cross_document or bool((features or {}).get("is_cross_document"))
        edge.is_cross_language = edge.is_cross_language or bool((features or {}).get("is_cross_language"))
        edge.bridge_quota_reason = (features or {}).get("bridge_quota_reason") or edge.bridge_quota_reason
        edge.diagnostics_json = {
            **(edge.diagnostics_json or {}),
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
            "weight_is_compatibility_copy": True,
        }
        return edge
    edge = ChunkRelationEdge(
        graph_state_id=graph_state.id,
        knowledge_base_id=graph_state.knowledge_base_id,
        source_chunk_id=left,
        target_chunk_id=right,
        edge_type=edge_type,
        weight=raw_strength,
        distance=distance,
        raw_strength=raw_strength,
        raw_strength_summary_json={
            "max_raw_strength": raw_strength,
            "distance": distance,
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
        },
        normalization_stats_json=(features or {}).get("normalization_stats") or {},
        confidence=raw_strength,
        features_json=features or {},
        support_json={"source": edge_type},
        source_algorithm=relation_edge_source_algorithm(edge_type),
        protocol_version=graph_state.relation_protocol_version,
        edge_distance_protocol_hash=edge_distance_protocol_hash(),
        graph_state_hash=graph_state.state_hash,
        source_language=(features or {}).get("source_language"),
        target_language=(features or {}).get("target_language"),
        is_cross_document=bool((features or {}).get("is_cross_document")),
        is_cross_language=bool((features or {}).get("is_cross_language")),
        bridge_quota_reason=(features or {}).get("bridge_quota_reason"),
        is_bridge=is_bridge,
        diagnostics_json={
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
            "weight_is_compatibility_copy": True,
        },
    )
    db.add(edge)
    edges[key] = edge
    return edge


def _upsert_relation_candidate(
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate],
    source_chunk_id: str,
    target_chunk_id: str,
    edge_type: str,
    raw_strength: float,
    features: dict[str, Any],
    *,
    is_bridge: bool = False,
) -> None:
    if source_chunk_id == target_chunk_id:
        return
    left, right = sorted([source_chunk_id, target_chunk_id])
    key = (left, right, edge_type)
    strength = normalized_strength(raw_strength)
    existing = candidates.get(key)
    if existing is not None:
        merged_strength = max(existing.raw_strength, strength)
        candidates[key] = RelationEdgeCandidate(
            source_chunk_id=left,
            target_chunk_id=right,
            edge_type=edge_type,
            raw_strength=merged_strength,
            distance=distance_from_strength(merged_strength),
            features_json={**existing.features_json, **features},
            is_bridge=existing.is_bridge or is_bridge,
        )
        return
    candidates[key] = RelationEdgeCandidate(
        source_chunk_id=left,
        target_chunk_id=right,
        edge_type=edge_type,
        raw_strength=strength,
        distance=distance_from_strength(strength),
        features_json=features,
        is_bridge=is_bridge,
    )


def relation_edge_candidates(
    db: Session,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    operating_point: dict[str, Any],
) -> tuple[dict[tuple[str, str, str], RelationEdgeCandidate], dict[str, Any]]:
    documents = {doc.id: doc for doc in db.scalars(select(Document).where(Document.id.in_({chunk.document_id for chunk in chunks}))).all()}
    strong_threshold = float(operating_point.get("dense_strong_cosine") or 0.72)
    type_thresholds = {
        "dense_semantic": float(operating_point.get("dense_min_cosine") or 0.30),
        "dense_cross_document_bridge": float(operating_point.get("cross_doc_min_cosine") or 0.36),
        "dense_cross_language_bridge": float(operating_point.get("cross_language_min_cosine") or 0.34),
    }

    def language_for(chunk: Chunk) -> str:
        document = documents.get(chunk.document_id)
        return (document.language if document and document.language else "unknown").lower()

    def node_mass(chunk: Chunk) -> float:
        token_mass = min(1.0, math.log2(1 + max(1, rough_token_count(chunk.text))) / 10.0)
        span_mass = 1.0 if chunk.char_end > chunk.char_start else 0.25
        structure_mass = 1.0 if chunk.section_path else 0.65
        return max(0.0, min(1.0, 0.5 * token_mass + 0.25 * span_mass + 0.25 * structure_mass))

    def quota(min_key: str, max_key: str, mass: float) -> int:
        lower = int(operating_point.get(min_key) or 1)
        upper = int(operating_point.get(max_key) or lower)
        value = lower + int(math.log2(1 + max(0.0, mass * 16.0)))
        return max(lower, min(upper, value))

    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    mass_by_id = {chunk.id: node_mass(chunk) for chunk in chunks}
    candidate_intents: dict[tuple[str, str, str], dict[str, Any]] = {}
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate] = {}

    def channel_for(source: Chunk, target: Chunk, *, requested_type: str | None = None) -> str:
        source_language = language_for(source)
        target_language = language_for(target)
        if requested_type:
            return requested_type
        if source_language != "unknown" and target_language != "unknown" and source_language != target_language:
            return "dense_cross_language_bridge"
        if source.document_id != target.document_id:
            return "dense_cross_document_bridge"
        return "dense_semantic"

    for source in chunks:
        source_vector = vectors.get(source.id) or []
        if not source_vector:
            continue
        scored: list[tuple[float, Chunk]] = []
        for target in chunks:
            if source.id == target.id:
                continue
            target_vector = vectors.get(target.id) or []
            if target_vector:
                scored.append((cosine_similarity(source_vector, target_vector), target))
        scored.sort(key=lambda item: item[0], reverse=True)
        channels = [
            (
                "dense_semantic",
                quota("dense_knn_k_min", "dense_knn_k_max", mass_by_id[source.id]),
                [item for item in scored if item[0] >= type_thresholds["dense_semantic"]],
            ),
            (
                "dense_cross_document_bridge",
                quota("cross_doc_out_quota_min", "cross_doc_out_quota_max", mass_by_id[source.id]),
                [item for item in scored if item[1].document_id != source.document_id and item[0] >= type_thresholds["dense_cross_document_bridge"]],
            ),
            (
                "dense_cross_language_bridge",
                quota("cross_language_out_quota_min", "cross_language_out_quota_max", mass_by_id[source.id]),
                [
                    item
                    for item in scored
                    if language_for(source) != "unknown"
                    and language_for(item[1]) != "unknown"
                    and language_for(source) != language_for(item[1])
                    and item[0] >= type_thresholds["dense_cross_language_bridge"]
                ],
            ),
        ]
        for requested_type, limit, channel_candidates in channels:
            for score, target in channel_candidates[:limit]:
                edge_type = channel_for(source, target, requested_type=requested_type)
                key = (source.id, target.id, edge_type)
                existing = candidate_intents.get(key)
                if existing is None or score > float(existing["score"]):
                    candidate_intents[key] = {"score": score, "source_id": source.id, "target_id": target.id, "edge_type": edge_type}

    inbound_counts: dict[tuple[str, str], int] = defaultdict(int)
    accepted_types: Counter[str] = Counter()
    sorted_intents = sorted(candidate_intents.values(), key=lambda item: item["score"], reverse=True)
    for intent in sorted_intents:
        source = chunk_by_id[intent["source_id"]]
        target = chunk_by_id[intent["target_id"]]
        edge_type = str(intent["edge_type"])
        score = float(intent["score"])
        reverse_key = (target.id, source.id, edge_type)
        mutual = reverse_key in candidate_intents
        reverse_limit_keys = {
            "dense_semantic": ("dense_reverse_b_min_base", "dense_reverse_b_max_base"),
            "dense_cross_document_bridge": ("dense_reverse_b_min_doc", "dense_reverse_b_max_doc"),
            "dense_cross_language_bridge": ("dense_reverse_b_min_lang", "dense_reverse_b_max_lang"),
        }[edge_type]
        inbound_key = (target.id, edge_type)
        inbound_limit = quota(reverse_limit_keys[0], reverse_limit_keys[1], mass_by_id[target.id])
        reverse_accepted = inbound_counts[inbound_key] < inbound_limit
        if score < type_thresholds[edge_type]:
            continue
        if not (mutual or reverse_accepted or score >= strong_threshold):
            continue
        semantic = (score - type_thresholds[edge_type]) / max(strong_threshold - type_thresholds[edge_type], 1e-6)
        semantic = max(0.0, min(1.0, semantic))
        rank_score = max(0.0, min(1.0, score))
        reciprocity = 1.0 if mutual else 0.35 if reverse_accepted else 0.0
        node_quality_pair = (mass_by_id[source.id] + mass_by_id[target.id]) / 2.0
        raw_strength = normalized_strength(0.75 * semantic + 0.15 * reciprocity + 0.07 * rank_score + 0.03 * node_quality_pair)
        is_cross_document = source.document_id != target.document_id
        is_cross_language = language_for(source) != "unknown" and language_for(target) != "unknown" and language_for(source) != language_for(target)
        bridge_reason = "cross_language_dense_quota" if edge_type == "dense_cross_language_bridge" else "cross_document_dense_quota" if edge_type == "dense_cross_document_bridge" else None
        _upsert_relation_candidate(
            candidates,
            source.id,
            target.id,
            edge_type,
            raw_strength,
            {
                "cosine": round(score, 6),
                "rank_score": round(rank_score, 6),
                "reciprocity": reciprocity,
                "node_quality_pair": round(node_quality_pair, 6),
                "source_language": language_for(source),
                "target_language": language_for(target),
                "is_cross_document": is_cross_document,
                "is_cross_language": is_cross_language,
                "bridge_quota_reason": bridge_reason,
                "normalization_stats": {
                    "type_threshold": type_thresholds[edge_type],
                    "strong_threshold": strong_threshold,
                    "semantic_strength": round(semantic, 6),
                    "source_mass": round(mass_by_id[source.id], 6),
                    "target_mass": round(mass_by_id[target.id], 6),
                    "mutual": mutual,
                    "reverse_accepted": reverse_accepted,
                },
            },
            is_bridge=edge_type in {"dense_cross_document_bridge", "dense_cross_language_bridge"},
        )
        inbound_counts[inbound_key] += 1
        accepted_types[edge_type] += 1

    diagnostics: dict[str, Any] = {}
    if chunks and not accepted_types:
        diagnostics["dense_relation_graph_blocked"] = {
            "reason": "no_dense_candidate_passed_thresholds",
            "candidate_count": len(candidate_intents),
            "thresholds": type_thresholds,
        }
    final_edge_types = Counter(candidate.edge_type for candidate in candidates.values())
    diagnostics.update(
        {
            "structure_edges_active_in_relation_graph": False,
            "structure_context_restore_source": "chunk_structure_graph",
            "relation_edge_protocol": "dense_only",
            "rq_pair_edges_active": False,
            "rq_pair_diagnostics_only": True,
            "accepted_edge_types": dict(final_edge_types),
            "accepted_edge_attempts": dict(accepted_types),
            "candidate_intent_count": len(candidate_intents),
        }
    )
    return candidates, diagnostics


def add_relation_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> None:
    operating_point = dict(graph_state.graph_operating_point_json or dense_graph_operating_point())
    candidates, diagnostics = relation_edge_candidates(db, chunks, vectors, operating_point)
    for candidate in candidates.values():
        add_chunk_relation_edge(
            db,
            graph_state,
            candidate.source_chunk_id,
            candidate.target_chunk_id,
            candidate.edge_type,
            candidate.raw_strength,
            candidate.features_json,
            edges,
            is_bridge=candidate.is_bridge,
        )
    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        **diagnostics,
    }
    db.flush()


def support_chunk_edge_ids_for_chunks(chunk_ids: set[str] | list[str], edges: dict[tuple[str, str, str], ChunkRelationEdge]) -> list[str]:
    chunk_set = set(chunk_ids)
    return [
        edge.id
        for edge in edges.values()
        if edge.id and (edge.source_chunk_id in chunk_set or edge.target_chunk_id in chunk_set)
    ]


def support_chunk_edge_ids_between(
    left_chunk_ids: set[str],
    right_chunk_ids: set[str],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> list[str]:
    direct = [
        edge.id
        for edge in edges.values()
        if edge.id
        and (
            (edge.source_chunk_id in left_chunk_ids and edge.target_chunk_id in right_chunk_ids)
            or (edge.source_chunk_id in right_chunk_ids and edge.target_chunk_id in left_chunk_ids)
        )
    ]
    return list(dict.fromkeys(direct))


def rq_prefix_community_groups(chunks: list[Chunk], edges: dict[tuple[str, str, str], ChunkRelationEdge]) -> list[tuple[str, list[Chunk], dict[str, Any]]]:
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    try:
        import networkx as nx

        graph = nx.Graph()
        for chunk in chunks:
            graph.add_node(chunk.id)
        for edge in edges.values():
            if edge.source_chunk_id not in chunk_by_id or edge.target_chunk_id not in chunk_by_id:
                continue
            strength = float(edge.raw_strength or edge.weight or 0.0)
            if strength <= 0:
                continue
            graph.add_edge(edge.source_chunk_id, edge.target_chunk_id, weight=strength, edge_type=edge.edge_type)
        if graph.number_of_edges() > 0:
            communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        else:
            communities = []
    except Exception:
        communities = []

    groups: list[list[Chunk]] = []
    for community in communities:
        members = [chunk_by_id[chunk_id] for chunk_id in community if chunk_id in chunk_by_id]
        if members:
            groups.append(sorted(members, key=lambda chunk: chunk.chunk_index))

    if not groups:
        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_index)
        groups = [ordered[index : index + 4] for index in range(0, len(ordered), 4)]

    labelled: list[tuple[str, list[Chunk], dict[str, Any]]] = []
    for index, members in enumerate(sorted(groups, key=lambda group: min(chunk.chunk_index for chunk in group))):
        representative_text = "\n".join(chunk.section_path or chunk.text[:600] for chunk in members[:8])
        label = concept_label_from_text(representative_text)
        labelled.append(
            (
                label,
                members,
                {
                    "community_index": index,
                    "community_detection": "chunk_relation_greedy_modularity_v1" if communities else "chunk_relation_sequential_fallback_v1",
                    "member_count": len(members),
                    "label_source": "diagnostic_representative_terms",
                },
            )
        )
    return labelled


def build_rq_prefixes(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> list[RQPrefix]:
    edge_bridge_chunks = {edge.source_chunk_id for edge in edges.values() if edge.is_bridge} | {edge.target_chunk_id for edge in edges.values() if edge.is_bridge}
    clusters: list[RQPrefix] = []
    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        "rq_membership": {
            "active_prefix_source": "residual_quantized_kmeans",
            "community_seed_active": False,
            "rq_prefix_edge_active": False,
        },
    }
    build_rq_kmeans_clusters_and_edges(db, graph_state, chunks, vectors, edge_bridge_chunks, clusters, edges)
    db.flush()
    return clusters


RQ_LEVELS = 3
RQ_MAX_K = 6
RQ_TAU_R = 0.65


def rq_runtime_config() -> dict[str, float | int]:
    settings = get_settings()
    return {
        "levels": max(1, int(getattr(settings, "rq_kmeans_levels", RQ_LEVELS) or RQ_LEVELS)),
        "max_k": max(1, int(getattr(settings, "rq_kmeans_max_k", RQ_MAX_K) or RQ_MAX_K)),
        "tau_r": max(1e-6, float(getattr(settings, "rq_residual_tau", RQ_TAU_R) or RQ_TAU_R)),
    }


def rq_tau() -> float:
    return float(rq_runtime_config()["tau_r"])


def build_rq_kmeans_clusters_and_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    bridge_chunk_ids: set[str],
    clusters: list[RQPrefix],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> None:
    for chunk in chunks:
        chunk.rq_path = []
        chunk.rq_residual_norm = None
    vector_items = [(chunk, vectors.get(chunk.id) or []) for chunk in chunks if vectors.get(chunk.id)]
    if not vector_items:
        graph_state.diagnostics_json = {**(graph_state.diagnostics_json or {}), "rq_kmeans": {"enabled": False, "reason": "no_vectors"}}
        return
    rq_config = rq_runtime_config()
    width = min(len(vector) for _chunk, vector in vector_items)
    normalized_items = [(chunk, _fit_width(vector, width)) for chunk, vector in vector_items]
    rq_model = train_rq_kmeans(
        [vector for _chunk, vector in normalized_items],
        levels=min(int(rq_config["levels"]), max(1, len(normalized_items))),
        max_k=int(rq_config["max_k"]),
        tau_r=float(rq_config["tau_r"]),
    )
    if not rq_model.get("codebooks"):
        graph_state.diagnostics_json = {**(graph_state.diagnostics_json or {}), "rq_kmeans": {"enabled": False, "reason": "empty_codebooks"}}
        return

    assignments: dict[str, dict[str, Any]] = {}
    for chunk, vector in normalized_items:
        encoded = encode_rq_vector(vector, rq_model)
        assignments[chunk.id] = encoded
        chunk.rq_path = encoded["rq_path"]
        chunk.rq_residual_norm = float(encoded["residual_norm"])

    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        "rq_kmeans": {
            **rq_model,
            "enabled": True,
            "chunk_count": len(assignments),
            "tau_r": float(rq_config["tau_r"]),
            "path_by_chunk": {chunk_id: data["rq_path"] for chunk_id, data in assignments.items()},
        },
    }

    rq_prefixes_by_key: dict[tuple[int, tuple[int, ...]], RQPrefix] = {}
    prefix_groups: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for chunk_id, encoded in assignments.items():
        path = encoded["rq_path"]
        for level in range(1, len(path) + 1):
            prefix_groups[(level, tuple(path[:level]))].append(chunk_id)

    for (level, prefix), member_ids in sorted(prefix_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        prefix_vector = rq_prefix_vector(rq_model, list(prefix))
        parent_prefix = rq_prefixes_by_key.get((level - 1, tuple(prefix[:-1]))) if level > 1 else None
        representatives = sorted(
            member_ids,
            key=lambda chunk_id: _sq_distance(_fit_width(vectors.get(chunk_id) or [], len(prefix_vector)), prefix_vector),
        )[:3]
        residual_norms = [float(assignments[chunk_id]["residual_norm"]) for chunk_id in member_ids]
        residual_mean = _centroid([assignments[chunk_id]["residual_vector"] for chunk_id in member_ids])
        cluster = RQPrefix(
            graph_state_id=graph_state.id,
            knowledge_base_id=graph_state.knowledge_base_id,
            rq_prefix_key=f"rq:L{level}:{'-'.join(str(item) for item in prefix)}",
            label=f"RQ L{level} {'/'.join(str(item) for item in prefix)}",
            node_type="rq_prefix",
            centroid_json=prefix_vector,
            rq_level=level,
            rq_path_prefix=list(prefix),
            parent_rq_prefix_id=parent_prefix.id if parent_prefix else None,
            codebook_version=str(rq_model.get("index_protocol") or "residual_quantized_kmeans_v1"),
            centroid_vector_ref=f"rq:{graph_state.id}:L{level}:{'-'.join(str(item) for item in prefix)}",
            representative_chunk_ids_json=representatives,
            support_chunk_ids_json=member_ids,
            bridge_chunk_ids_json=[chunk_id for chunk_id in member_ids if chunk_id in bridge_chunk_ids],
            state=graph_state.state,
            stats_json={
                "member_count": len(member_ids),
                "residual_norm_mean": round(sum(residual_norms) / max(len(residual_norms), 1), 6),
                "residual_norm_max": round(max(residual_norms or [0.0]), 6),
            },
            diagnostics_json={"source": "rq_kmeans", "residual_mean_vector": residual_mean, **({"shadow_scope": True} if graph_state.state == "shadow" else {})},
        )
        db.add(cluster)
        db.flush()
        clusters.append(cluster)
        rq_prefixes_by_key[(level, tuple(prefix))] = cluster
        for chunk_id in member_ids:
            encoded = assignments[chunk_id]
            db.add(
                RQPrefixMembership(
                    rq_prefix_id=cluster.id,
                    chunk_id=chunk_id,
                    membership_score=rq_membership_score(float(encoded["residual_norm"]), tau_r=float(rq_config["tau_r"])),
                    membership_role="primary_member" if level == len(encoded["rq_path"]) else "fuzzy_member",
                    membership_reason="rq_prefix" if level < len(encoded["rq_path"]) else "rq_leaf",
                    membership_entropy=float((encoded.get("membership_entropy") or 0.0)),
                    rq_path=encoded["rq_path"],
                    residual_norm=float(encoded["residual_norm"]),
                    rank=level,
                    support_chunk_edge_ids_json=support_chunk_edge_ids_for_chunks({chunk_id}, edges)[:16],
                    diagnostics_json={
                        "residual_vector": encoded["residual_vector"],
                        "reconstructed_vector": encoded["reconstructed_vector"],
                        "rq_level": level,
                        "rq_path_prefix": list(prefix),
                    },
                )
            )

    db.flush()
    for cluster in clusters:
        db.add(
            RQPrefixDiagnostic(
                graph_state_id=graph_state.id,
                rq_prefix_id=cluster.id,
                diagnostic_type="membership_mass",
                diagnostic_strength=float(len(cluster.support_chunk_ids_json or [])),
                support_membership_mass=float(len(cluster.support_chunk_ids_json or [])),
                support_chunk_ids_sample_json=list(cluster.support_chunk_ids_json or [])[:24],
                diagnostics_json={
                    "rq_level": cluster.rq_level,
                    "rq_path_prefix": cluster.rq_path_prefix or [],
                    "hard_parent_tree": True,
                },
            )
        )
    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        "rq_membership": {
            **((graph_state.diagnostics_json or {}).get("rq_membership") or {}),
            "prefix_count": len(rq_prefixes_by_key),
            "path_availability": round(len(assignments) / max(len(chunks), 1), 6),
            "hard_parent_tree": True,
            "rq_pair_edges_active": False,
            "rq_pair_diagnostics_only": True,
        },
    }

def train_rq_kmeans(vectors: list[list[float]], *, levels: int, max_k: int, tau_r: float | None = None) -> dict[str, Any]:
    residuals = [list(vector) for vector in vectors]
    codebooks: list[list[list[float]]] = []
    for level in range(levels):
        if not residuals:
            break
        k = min(max_k, max(1, int(math.sqrt(len(residuals))) + 1), len(residuals))
        codebook = train_kmeans_codebook(residuals, k=k, iterations=8)
        if not codebook:
            break
        codebooks.append(codebook)
        next_residuals: list[list[float]] = []
        for residual in residuals:
            index, center = nearest_code(residual, codebook)
            _ = index
            next_residuals.append(_vector_sub(residual, center))
        residuals = next_residuals
    return {
        "levels": len(codebooks),
        "codebooks": codebooks,
        "codebook_sizes": [len(codebook) for codebook in codebooks],
        "embedding_dimensions": len(vectors[0]) if vectors else 0,
        "tau_r": float(tau_r or rq_tau()),
        "index_protocol": "residual_quantized_kmeans_v1",
    }


def train_kmeans_codebook(vectors: list[list[float]], *, k: int, iterations: int) -> list[list[float]]:
    if not vectors:
        return []
    ordered = sorted(vectors, key=lambda vector: stable_hash([round(value, 6) for value in vector]))
    if k <= 1:
        return [_centroid(ordered)]
    step = max(1, len(ordered) // k)
    centers = [ordered[min(index * step, len(ordered) - 1)] for index in range(k)]
    for _iteration in range(iterations):
        groups: list[list[list[float]]] = [[] for _ in centers]
        for vector in ordered:
            index, _center = nearest_code(vector, centers)
            groups[index].append(vector)
        next_centers = [(_centroid(group) if group else centers[index]) for index, group in enumerate(groups)]
        if all(_sq_distance(left, right) < 1e-12 for left, right in zip(centers, next_centers)):
            break
        centers = next_centers
    return centers


def encode_rq_vector(vector: list[float], rq_model: dict[str, Any] | None) -> dict[str, Any]:
    codebooks = list((rq_model or {}).get("codebooks") or [])
    width = int((rq_model or {}).get("embedding_dimensions") or len(vector))
    tau = float((rq_model or {}).get("tau_r") or rq_tau())
    residual = _fit_width(vector, width)
    reconstructed = [0.0 for _ in range(width)]
    path: list[int] = []
    for codebook in codebooks:
        fitted_codebook = [_fit_width([float(value) for value in center], width) for center in codebook]
        if not fitted_codebook:
            continue
        index, center = nearest_code(residual, fitted_codebook)
        path.append(index + 1)
        reconstructed = _vector_add(reconstructed, center)
        residual = _vector_sub(residual, center)
    return {
        "rq_path": path,
        "residual_vector": residual,
        "residual_norm": _vector_norm(residual),
        "reconstructed_vector": reconstructed,
        "tau_r": tau,
    }


def rq_prefix_vector(rq_model: dict[str, Any], prefix: list[int]) -> list[float]:
    width = int(rq_model.get("embedding_dimensions") or 0)
    vector = [0.0 for _ in range(width)]
    for level, code_index in enumerate(prefix):
        codebook = (rq_model.get("codebooks") or [])[level] if level < len(rq_model.get("codebooks") or []) else []
        if not codebook:
            continue
        center = codebook[max(0, min(int(code_index) - 1, len(codebook) - 1))]
        vector = _vector_add(vector, _fit_width([float(value) for value in center], width))
    return vector


def rq_membership_score(residual_norm_value: float, *, tau_r: float | None = None) -> float:
    return round(max(0.2, min(1.0, math.exp(-residual_norm_value / max(float(tau_r or rq_tau()), 1e-6)))), 6)


def lcp_depth(left_path: list[int], right_path: list[int]) -> int:
    depth = 0
    for left, right in zip(left_path, right_path):
        if left != right:
            break
        depth += 1
    return depth


def rq_model_from_relation_state(relation_state: ChunkRelationGraphState | None) -> dict[str, Any] | None:
    if relation_state is None:
        return None
    model = (relation_state.diagnostics_json or {}).get("rq_kmeans")
    if not isinstance(model, dict) or not model.get("enabled") or not model.get("codebooks"):
        return None
    return model


def encode_query_rq(relation_state: ChunkRelationGraphState | None, query_vector: list[float]) -> dict[str, Any] | None:
    model = rq_model_from_relation_state(relation_state)
    if model is None or not query_vector:
        return None
    encoded = encode_rq_vector(query_vector, model)
    return {
        "rq_path": encoded["rq_path"],
        "residual_vector": encoded["residual_vector"],
        "residual_norm": round(float(encoded["residual_norm"]), 6),
        "reconstructed_vector": encoded["reconstructed_vector"],
        "tau_r": float(model.get("tau_r") or rq_tau()),
        "index_protocol": model.get("index_protocol") or "residual_quantized_kmeans_v1",
    }


def rq_candidate_score(query_rq: dict[str, Any], membership: RQPrefixMembership) -> dict[str, Any]:
    candidate_path = [int(value) for value in (membership.rq_path or [])]
    query_path = [int(value) for value in (query_rq.get("rq_path") or [])]
    levels = max(len(query_path), len(candidate_path), 1)
    candidate_residual = (membership.diagnostics_json or {}).get("residual_vector") or []
    query_residual = query_rq.get("residual_vector") or []
    distance = _vector_norm(_vector_sub(_fit_width(query_residual, max(len(query_residual), len(candidate_residual))), _fit_width(candidate_residual, max(len(query_residual), len(candidate_residual)))))
    lcp = lcp_depth(query_path, candidate_path)
    tau = max(float(query_rq.get("tau_r") or rq_tau()), 1e-6)
    prefix_score = lcp / levels
    residual_score = math.exp(-distance / tau)
    membership_score = max(0.0, min(1.0, float(membership.membership_score or 0.0)))
    rq_score = max(0.0, min(1.0, 0.55 * prefix_score + 0.35 * residual_score + 0.10 * membership_score))
    rq_drift_penalty = max(0.0, (1.0 - prefix_score) + 0.15 * min(distance, 10.0))
    return {
        "query_rq_path": query_path,
        "candidate_rq_path": candidate_path,
        "lcp_depth": lcp,
        "residual_distance": round(distance, 6),
        "query_residual_norm": round(float(query_rq.get("residual_norm") or 0.0), 6),
        "candidate_residual_norm": round(float(membership.residual_norm or 0.0), 6),
        "rq_score": round(rq_score, 6),
        "rq_drift_penalty": round(rq_drift_penalty, 6),
        "membership_reason": membership.membership_reason,
    }


def nearest_code(vector: list[float], codebook: list[list[float]]) -> tuple[int, list[float]]:
    best_index = 0
    best_distance = float("inf")
    for index, center in enumerate(codebook):
        distance = _sq_distance(vector, center)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, codebook[best_index]


def _fit_width(vector: list[float], width: int) -> list[float]:
    if width <= 0:
        return []
    fitted = [float(value) for value in vector[:width]]
    if len(fitted) < width:
        fitted.extend([0.0] * (width - len(fitted)))
    return fitted


def _vector_add(left: list[float], right: list[float]) -> list[float]:
    width = min(len(left), len(right))
    return [float(left[index]) + float(right[index]) for index in range(width)]


def _vector_sub(left: list[float], right: list[float]) -> list[float]:
    width = min(len(left), len(right))
    return [float(left[index]) - float(right[index]) for index in range(width)]


def _sq_distance(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    return sum((float(left[index]) - float(right[index])) ** 2 for index in range(width))


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors)
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]



def relation_graph_stats(chunks: list[Chunk], edges: list[ChunkRelationEdge], clusters: list[RQPrefix]) -> dict[str, Any]:
    connected = Counter()
    edge_type_counts = Counter()
    distances: list[float] = []
    for edge in edges:
        connected[edge.source_chunk_id] += 1
        connected[edge.target_chunk_id] += 1
        edge_type_counts[edge.edge_type] += 1
        if edge.distance is not None:
            distances.append(float(edge.distance))
    orphan_count = sum(1 for chunk in chunks if connected[chunk.id] == 0)
    singleton_count = sum(1 for cluster in clusters if len(cluster.support_chunk_ids_json or []) <= 1)
    bridge_edges = sum(1 for edge in edges if edge.is_bridge)
    return {
        "chunk_count": len(chunks),
        "edge_count": len(edges),
        "rq_prefix_count": len(clusters),
        "bridge_edges": bridge_edges,
        "bridge_ratio": round(bridge_edges / max(len(edges), 1), 6),
        "edge_type_counts": dict(edge_type_counts),
        "distance_distribution": {
            "min": round(min(distances), 6) if distances else None,
            "max": round(max(distances), 6) if distances else None,
            "mean": round(sum(distances) / len(distances), 6) if distances else None,
        },
        "degree_distribution": {
            "min": min(connected.values()) if connected else 0,
            "max": max(connected.values()) if connected else 0,
            "mean": round(sum(connected.values()) / max(len(connected), 1), 6) if connected else 0,
        },
        "orphan_chunk_rate": round(orphan_count / max(len(chunks), 1), 6),
        "singleton_rate": round(singleton_count / max(len(clusters), 1), 6),
    }


def concept_label_from_text(text: str) -> str:
    terms = [term for term in tokenize_for_search_terms(text.lower()) if len(term) > 2 and term not in STOP_TERMS]
    if not terms:
        return "General context"
    counts = Counter(terms)
    return " ".join(term for term, _ in counts.most_common(3)).title()


def unique_concept_label(base_label: str | None, fallback_label: str, used_labels: set[str]) -> str:
    label = " ".join(str(base_label or fallback_label or "Grounded Concept").split())[:255]
    if label and label not in used_labels:
        used_labels.add(label)
        return label
    suffix_base = " ".join(str(fallback_label or "RQ prefix").split())[:80]
    counter = 1
    while True:
        suffix = f" ({suffix_base})" if counter == 1 else f" ({suffix_base} #{counter})"
        candidate = f"{label[: max(1, 255 - len(suffix))].rstrip()}{suffix}"[:255]
        if candidate not in used_labels:
            used_labels.add(candidate)
            return candidate
        counter += 1


STOP_TERMS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "have",
    "has",
    "can",
    "into",
    "about",
    "section",
    "chapter",
    "page",
}


async def build_mid_concept_graph(
    db: Session,
    knowledge_base_id: str,
    relation_state: ChunkRelationGraphState,
    *,
    batch_id: str | None = None,
    state_scope: str = "active",
    shadow_metadata: dict[str, Any] | None = None,
) -> MidConceptState:
    settings = get_settings()
    all_prefixes = list(db.scalars(select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id, RQPrefix.state == state_scope)).all())
    clusters = [prefix for prefix in all_prefixes if int(prefix.rq_level or 0) == RQ_LEVELS]
    if all_prefixes and not clusters:
        available_levels = sorted({int(prefix.rq_level or 0) for prefix in all_prefixes})
        raise RuntimeError(f"Mid concept graph requires active RQ L3 prefixes; available RQ levels: {available_levels}.")
    target_leaf_level = RQ_LEVELS
    grounding_hash = stable_hash([cluster.id for cluster in clusters] + [relation_state.state_hash])
    state = MidConceptState(
        knowledge_base_id=knowledge_base_id,
        chunk_relation_graph_state_id=relation_state.id,
        state_hash=stable_hash({"grounding": grounding_hash, "prompt": MID_CONCEPT_PROMPT_VERSION}),
        grounding_hash=grounding_hash,
        prompt_protocol_version=MID_CONCEPT_PROMPT_VERSION,
        stats_json={},
        diagnostics_json={**(shadow_metadata or {}), **({"shadow_scope": True} if state_scope == "shadow" else {})},
        state=state_scope,
    )
    db.add(state)
    db.flush()
    concepts: list[MidConcept] = []
    used_mid_labels: set[str] = set()
    packet_batches = mid_concept_packet_batches(db, clusters, settings)
    total_batches = len(packet_batches)
    context_graph_batch_heartbeat(
        batch_id,
        "mid_concepts",
        {"llm_batches": total_batches, "projected_rq_l3_prefixes": len(clusters), "model_concurrency": settings.model_request_concurrency},
    )

    async def define_mid_packet_batch(item: tuple[int, list[tuple[RQPrefix, dict[str, Any]]]]) -> tuple[int, list[tuple[RQPrefix, dict[str, Any]]], list[dict[str, Any]]]:
        batch_index, packet_batch = item
        outputs = await define_mid_concepts_batch([packet for _, packet in packet_batch])
        return batch_index, packet_batch, outputs

    completed_mid_batches = 0

    def mid_batch_progress(_result: tuple[int, list[tuple[RQPrefix, dict[str, Any]]], list[dict[str, Any]]]) -> None:
        nonlocal completed_mid_batches
        completed_mid_batches += 1
        if completed_mid_batches == 1 or completed_mid_batches == total_batches or completed_mid_batches % 5 == 0:
            context_graph_batch_heartbeat(
                batch_id,
                "mid_concepts",
                {
                    "completed_llm_batches": completed_mid_batches,
                    "llm_batches": total_batches,
                    "projected_rq_l3_prefixes": len(clusters),
                    "model_concurrency": settings.model_request_concurrency,
                },
            )

    batch_results = await gather_bounded(
        [(batch_index, packet_batch) for batch_index, packet_batch in enumerate(packet_batches, start=1)],
        settings.model_request_concurrency,
        define_mid_packet_batch,
        on_result=mid_batch_progress,
    )
    for batch_index, packet_batch, outputs in sorted(batch_results, key=lambda item: item[0]):
        for (cluster, packet), output in zip(packet_batch, outputs, strict=False):
            concept = write_mid_concept_from_output(db, state, knowledge_base_id, cluster, packet, output, used_mid_labels)
            concepts.append(concept)
        context_graph_batch_heartbeat(
            batch_id,
            "mid_concepts",
            {"completed_llm_batches": batch_index, "llm_batches": total_batches, "created_mid_concepts": len(concepts)},
        )
    concept_i18n_metrics = {
        "translation_phase": "concept_i18n",
        "translation_items": len(concepts),
        "translation_enabled": settings.concept_i18n_enabled,
    }
    if settings.concept_i18n_enabled:
        context_graph_batch_heartbeat(batch_id, "mid_concepts", concept_i18n_metrics)
        concept_i18n_stats = await enrich_concepts_i18n(concepts, layer="mid")
    else:
        context_graph_batch_heartbeat(batch_id, "mid_concepts", {**concept_i18n_metrics, "translation_status": "disabled"})
        concept_i18n_stats = disabled_concept_i18n_stats(len(concepts), layer="mid", target="concept")
    normalize_concept_node_weights(concepts, "mid_concept_state")
    build_mid_concept_edges(db, state, concepts)
    db.flush()
    mid_edges = list(db.scalars(select(MidConceptEdge).where(MidConceptEdge.concept_state_id == state.id)).all())
    edge_i18n_metrics = {
        "translation_phase": "edge_i18n",
        "translation_items": len(mid_edges),
        "translation_enabled": settings.concept_i18n_enabled,
    }
    if settings.concept_i18n_enabled:
        context_graph_batch_heartbeat(batch_id, "mid_concepts", edge_i18n_metrics)
        edge_i18n_stats = await enrich_concept_edges_i18n(mid_edges, {concept.id: concept for concept in concepts}, layer="mid")
    else:
        context_graph_batch_heartbeat(batch_id, "mid_concepts", {**edge_i18n_metrics, "translation_status": "disabled"})
        edge_i18n_stats = disabled_concept_i18n_stats(len(mid_edges), layer="mid", target="edge")
    db.flush()
    supported_mid_edges = sum(1 for edge in mid_edges if edge.support_chunk_edge_ids_json or edge.support_relation_edge_ids_json)
    stats = {
        "mid_concept_count": len(concepts),
        "grounded_concept_rate": 1.0 if concepts else 0.0,
        "mid_edge_count": len(mid_edges),
        "mid_edge_support_chunk_edge_coverage": round(supported_mid_edges / max(len(mid_edges), 1), 6) if mid_edges else 1.0,
        "rq_prefix_candidates": len(all_prefixes),
        "rq_leaf_prefix_candidates": len(clusters),
        "projected_rq_l3_prefixes": len(concepts),
        "rq_l3_to_mid_projection_coverage": round(len(concepts) / max(len(clusters), 1), 6) if clusters else 1.0,
        "llm_batches": len(packet_batches),
        "concept_i18n_enabled": bool(settings.concept_i18n_enabled),
        "concept_i18n_translated_count": concept_i18n_stats.get("translated_count", 0),
        "edge_i18n_translated_count": edge_i18n_stats.get("translated_count", 0),
    }
    state.stats_json = stats
    state.diagnostics_json = {
        **(state.diagnostics_json or {}),
        "candidate_keep_threshold_ignored_for_coverage": settings.mid_concept_candidate_keep_threshold,
        "configured_max_model_batches_ignored_for_coverage": settings.mid_concept_extraction_max_model_batches,
        "max_candidates_per_batch": settings.mid_concept_extraction_max_candidates_per_batch,
        "max_tokens_per_batch": settings.mid_concept_extraction_max_tokens_per_batch,
        "projected_rq_l3_prefix_ids": [cluster.id for batch in packet_batches for cluster, _ in batch],
        "target_leaf_level": target_leaf_level,
        "l3_available": bool(clusters) or not all_prefixes,
        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
        "concept_i18n": concept_i18n_stats,
        "edge_i18n": edge_i18n_stats,
    }
    state.state_hash = stable_hash({"concepts": [concept.id for concept in concepts], "stats": stats, "diagnostics": state.diagnostics_json})
    return state


def write_mid_concept_from_output(
    db: Session,
    state: MidConceptState,
    knowledge_base_id: str,
    cluster: RQPrefix,
    packet: dict[str, Any],
    output: dict[str, Any],
    used_labels: set[str] | None = None,
) -> MidConcept:
    support_chunks = [chunk_id for chunk_id in output.get("support_chunk_ids", []) if chunk_id in set(cluster.support_chunk_ids_json or [])]
    if not support_chunks:
        support_chunks = list(cluster.support_chunk_ids_json or [])[:5]
    support_chunk_edge_ids = list(dict.fromkeys(packet.get("support_chunk_edge_ids") or []))[:60]
    parent_l2 = db.get(RQPrefix, cluster.parent_rq_prefix_id) if cluster.parent_rq_prefix_id else None
    parent_l1 = db.get(RQPrefix, parent_l2.parent_rq_prefix_id) if parent_l2 and parent_l2.parent_rq_prefix_id else None
    raw_node_weight = float(len(cluster.support_chunk_ids_json or [])) + 0.25 * float(len(support_chunk_edge_ids))
    confidence, confidence_diagnostics = coerce_confidence(output.get("confidence"), default=0.72)
    label = unique_concept_label(str(output.get("canonical_label") or cluster.label), cluster.label, used_labels if used_labels is not None else set())
    concept = MidConcept(
        concept_state_id=state.id,
        knowledge_base_id=knowledge_base_id,
        canonical_label=label,
        aliases_json=list(output.get("aliases") or []),
        support_rq_l3_prefix_id=cluster.id if int(cluster.rq_level or 0) == RQ_LEVELS else None,
        parent_rq_l2_prefix_id=parent_l2.id if parent_l2 else None,
        parent_rq_l1_prefix_id=parent_l1.id if parent_l1 else None,
        definition=str(output.get("definition") or f"Grounded concept around {cluster.label}."),
        summary=str(output.get("summary") or output.get("definition") or f"Grounded concept around {cluster.label}."),
        scope_note=str(output.get("scope_note") or ""),
        inclusion_criteria_json=list(output.get("inclusion_criteria") or []),
        exclusion_criteria_json=list(output.get("exclusion_criteria") or []),
        display_terms_json=list(dict.fromkeys([cluster.label, *list(output.get("aliases") or [])]))[:12],
        internal_state_json={
            "rq_prefix_key": cluster.rq_prefix_key,
            "rq_level": cluster.rq_level,
            "rq_path_prefix": cluster.rq_path_prefix or [],
            "parent_rq_l2_prefix_id": parent_l2.id if parent_l2 else None,
            "parent_rq_l1_prefix_id": parent_l1.id if parent_l1 else None,
        },
        representative_chunk_ids_json=list(output.get("representative_chunk_ids") or cluster.representative_chunk_ids_json or [])[:5],
        support_rq_prefix_ids_json=[cluster.id],
        support_chunk_ids_json=support_chunks,
        support_chunk_edge_ids_json=support_chunk_edge_ids,
        core_chunk_ids_json=support_chunks[: min(5, len(support_chunks))],
        boundary_chunk_ids_json=list(dict.fromkeys(packet.get("boundary_chunk_ids") or []))[:20],
        bridge_chunk_ids_json=list(cluster.bridge_chunk_ids_json or [])[:20],
        outlier_chunk_ids_json=list(dict.fromkeys(packet.get("residual_outlier_chunk_ids") or []))[:10],
        raw_node_weight=raw_node_weight,
        node_weight=raw_node_weight,
        node_weight_diagnostics_json={
            "normalization_pending": True,
            "support_chunk_count": len(cluster.support_chunk_ids_json or []),
            "support_chunk_edge_count": len(support_chunk_edge_ids),
        },
        confidence=confidence,
        llm_audit_json={
            "prompt_protocol_version": MID_CONCEPT_PROMPT_VERSION,
            "packet": packet,
            "raw_output": output,
            "confidence": confidence_diagnostics,
        },
        grounding_hash=stable_hash({"cluster": cluster.id, "chunks": support_chunks}),
        state=state.state,
    )
    db.add(concept)
    db.flush()
    db.add(
        MidConceptMembership(
            mid_concept_id=concept.id,
            rq_prefix_id=cluster.id,
            membership_score=1.0,
            support_chunk_ids_json=support_chunks,
        )
    )
    db.add(
        MidConceptDefinition(
            mid_concept_id=concept.id,
            definition_json={
                "canonical_label": concept.canonical_label,
                "definition": concept.definition,
                "inclusion_criteria": concept.inclusion_criteria_json,
                "exclusion_criteria": concept.exclusion_criteria_json,
            },
            support_spans_json=support_spans_for_chunks(db, support_chunks),
            llm_audit_json=concept.llm_audit_json,
        )
    )
    return concept


def normalize_concept_node_weights(concepts: list[Any], scope: str) -> None:
    if not concepts:
        return
    max_raw = max((float(getattr(concept, "raw_node_weight", 0.0) or 0.0) for concept in concepts), default=0.0)
    for concept in concepts:
        raw = float(getattr(concept, "raw_node_weight", 0.0) or 0.0)
        normalized = round(raw / max(max_raw, 1e-9), 6) if max_raw > 0 else 0.0
        concept.node_weight = normalized
        concept.node_weight_normalization_scope = scope
        concept.node_weight_diagnostics_json = {
            **(getattr(concept, "node_weight_diagnostics_json", None) or {}),
            "normalization": "max_raw_within_state_v1",
            "raw_node_weight": raw,
            "max_raw_node_weight": max_raw,
            "node_weight": normalized,
        }


def mid_concept_packet_batches(db: Session, clusters: list[RQPrefix], settings: Any) -> list[list[tuple[RQPrefix, dict[str, Any]]]]:
    if not clusters:
        return []
    max_candidates = max(1, int(settings.mid_concept_extraction_max_candidates_per_batch))
    max_tokens = max(500, int(settings.mid_concept_extraction_max_tokens_per_batch))
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            tuple(int(part) for part in (cluster.rq_path_prefix or [])),
            cluster.rq_prefix_key or "",
            cluster.label or "",
            cluster.id,
        ),
    )
    batches: list[list[tuple[RQPrefix, dict[str, Any]]]] = []
    current: list[tuple[RQPrefix, dict[str, Any]]] = []
    current_tokens = 0
    for cluster in ordered:
        packet = concept_packet_for_cluster(db, cluster)
        packet_tokens = max(1, rough_token_count(str(packet)))
        if current and (len(current) >= max_candidates or current_tokens + packet_tokens > max_tokens):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((cluster, packet))
        current_tokens += packet_tokens
    if current:
        batches.append(current)
    return batches


def mid_concept_fallback(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet.get("packet_id"),
        "canonical_label": (packet.get("candidate_labels") or ["Grounded Concept"])[0],
        "aliases": [],
        "definition": f"A grounded concept supported by RQ membership evidence for {(packet.get('candidate_labels') or ['this topic'])[0]}.",
        "scope_note": "Generated from chunk relation evidence and RQ membership support.",
        "inclusion_criteria": ["Supported by the listed chunks."],
        "exclusion_criteria": ["Claims without supporting chunks."],
        "representative_chunk_ids": packet.get("representative_chunk_ids") or [],
        "support_chunk_ids": packet.get("support_chunk_ids") or [],
        "confidence": 0.72,
        "why_this_concept_exists": "The supporting chunks share a residual-quantized membership prefix and bottom chunk-edge support.",
    }


async def define_mid_concepts_batch(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback_concepts = [mid_concept_fallback(packet) for packet in packets]
    if not packets:
        return []
    system = (
        "You define mid-level concepts for a Four-Layer Context Graph RAG system. "
        "Use only the supplied concept packets. Return strict JSON with a concepts array. "
        "Each item must include packet_id, canonical_label, aliases, definition, scope_note, "
        "inclusion_criteria, exclusion_criteria, representative_chunk_ids, support_chunk_ids, "
        "confidence, and why_this_concept_exists."
    )
    output = await ChatProvider(purpose="graph").classify_json(system_prompt=system, user_prompt=str({"concept_packets": packets}), fallback={"concepts": fallback_concepts})
    raw_concepts = output.get("concepts") if isinstance(output, dict) else None
    if not isinstance(raw_concepts, list):
        if get_settings().enable_model_fallback:
            return fallback_concepts
        raise RuntimeError("Mid concept provider returned invalid JSON: concepts array is required")
    by_packet = {str(item.get("packet_id")): item for item in raw_concepts if isinstance(item, dict)}
    concepts: list[dict[str, Any]] = []
    missing_packet_ids: list[str] = []
    for packet, fallback in zip(packets, fallback_concepts, strict=False):
        packet_id = str(packet.get("packet_id"))
        concept = by_packet.get(packet_id)
        if concept is None:
            if get_settings().enable_model_fallback:
                concept = fallback
            else:
                missing_packet_ids.append(packet_id)
                continue
        concepts.append(concept)
    if missing_packet_ids:
        raise RuntimeError(f"Mid concept provider omitted packet ids: {', '.join(missing_packet_ids[:8])}")
    return concepts


async def define_mid_concept(packet: dict[str, Any]) -> dict[str, Any]:
    return (await define_mid_concepts_batch([packet]))[0]


def concept_packet_for_cluster(db: Session, cluster: RQPrefix) -> dict[str, Any]:
    membership_rows = list(
        db.scalars(
            select(RQPrefixMembership)
            .where(RQPrefixMembership.rq_prefix_id == cluster.id)
        ).all()
    )
    membership_rows = sorted(membership_rows, key=lambda row: float(row.residual_norm or -1.0), reverse=True)[:6]
    residual_outlier_ids = [row.chunk_id for row in membership_rows if row.residual_norm is not None][:3]
    representative_ids = list(dict.fromkeys(list(cluster.representative_chunk_ids_json or []) + residual_outlier_ids))
    chunk_ids = list(representative_ids or cluster.support_chunk_ids_json or [])[:6]
    chunks = list(db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()) if chunk_ids else []
    support_set = set(cluster.support_chunk_ids_json or [])
    support_chunk_edges = (
        list(
            db.scalars(
                select(ChunkRelationEdge)
                .where(
                    ChunkRelationEdge.graph_state_id == cluster.graph_state_id,
                    (ChunkRelationEdge.source_chunk_id.in_(support_set) | ChunkRelationEdge.target_chunk_id.in_(support_set)),
                )
                .limit(120)
            ).all()
        )
        if support_set
        else []
    )
    support_chunk_edge_ids = [edge.id for edge in support_chunk_edges if edge.id]
    boundary_chunk_ids = list(
        dict.fromkeys(
            [
                edge.target_chunk_id if edge.source_chunk_id in support_set else edge.source_chunk_id
                for edge in support_chunk_edges
                if (edge.source_chunk_id in support_set) != (edge.target_chunk_id in support_set)
            ]
        )
    )
    rq_paths = [
        row.rq_path
        for row in membership_rows
        if row.rq_path
    ]
    return {
        "packet_id": stable_hash({"cluster": cluster.id, "chunks": chunk_ids})[:16],
        "support_rq_prefix_node_ids": [cluster.id],
        "support_chunk_edge_ids": support_chunk_edge_ids[:60],
        "bridge_chunk_ids": list(cluster.bridge_chunk_ids_json or [])[:20],
        "boundary_chunk_ids": boundary_chunk_ids[:24],
        "candidate_labels": [cluster.label],
        "representative_chunk_ids": [chunk.id for chunk in chunks],
        "support_chunk_count": len(cluster.support_chunk_ids_json or []),
        "support_chunk_ids": list(cluster.support_chunk_ids_json or [])[:30],
        "bridge_chunk_count": len(cluster.bridge_chunk_ids_json or []),
        "residual_outlier_chunk_ids": residual_outlier_ids,
        "rq_sampling": {
            "used": bool(cluster.rq_level),
            "level": cluster.rq_level,
            "prefix_depth": len(cluster.rq_path_prefix or []),
            "residual_outlier_chunk_ids": residual_outlier_ids,
            "residual_quantization_coverage": rq_paths[:12],
        },
        "source_spans": support_spans_for_chunks(db, list(cluster.support_chunk_ids_json or [])[:12]),
        "chunk_excerpts": [
            {
                "chunk_id": chunk.id,
                "section_path": chunk.section_path,
                "page_range": [chunk.page_start, chunk.page_end],
                "text": chunk.text[:480],
            }
            for chunk in chunks
        ],
        "grounding_hash": stable_hash({"cluster": cluster.id, "support": cluster.support_chunk_ids_json}),
    }


def support_spans_for_chunks(db: Session, chunk_ids: list[str]) -> list[dict[str, Any]]:
    chunks = list(db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()) if chunk_ids else []
    return [
        {
            "chunk_id": chunk.id,
            "document_version_id": chunk.document_version_id,
            "char_span": [chunk.char_start, chunk.char_end],
            "page_range": [chunk.page_start, chunk.page_end],
            "section_path": chunk.section_path,
        }
        for chunk in chunks
    ]


def build_mid_concept_edges(db: Session, state: MidConceptState, concepts: list[MidConcept]) -> None:
    if len(concepts) < 2:
        return
    chunk_edges = list(
        db.scalars(
            select(ChunkRelationEdge).where(ChunkRelationEdge.graph_state_id == state.chunk_relation_graph_state_id)
        ).all()
    )
    chunk_edges_by_id = {edge.id: edge for edge in chunk_edges if edge.id}
    edge_index = {(edge.source_chunk_id, edge.target_chunk_id, edge.edge_type): edge for edge in chunk_edges}
    chunks_by_concept = {concept.id: set(concept.support_chunk_ids_json or []) for concept in concepts}
    prefixes_by_concept = {concept.id: set(concept.support_rq_prefix_ids_json or []) for concept in concepts}
    for index, left in enumerate(concepts):
        for right in concepts[index + 1 :]:
            left_chunks = chunks_by_concept[left.id]
            right_chunks = chunks_by_concept[right.id]
            support_chunk_edge_ids = support_chunk_edge_ids_between(left_chunks, right_chunks, edge_index)
            supporting_edges = [chunk_edges_by_id[edge_id] for edge_id in support_chunk_edge_ids if edge_id in chunk_edges_by_id]
            if not supporting_edges:
                continue
            support_chunk_ids = list(
                dict.fromkeys(
                    chunk_id
                    for edge in supporting_edges
                    for chunk_id in (edge.source_chunk_id, edge.target_chunk_id)
                )
            )[:24]
            distances = [float(edge.distance if edge.distance is not None else distance_from_strength(edge.raw_strength or edge.weight)) for edge in supporting_edges]
            raw_strengths = [normalized_strength(float(edge.raw_strength or edge.weight or 0.1)) for edge in supporting_edges]
            raw_strength = max(raw_strengths) if raw_strengths else 0.1
            distance = min(distances) if distances else distance_from_strength(raw_strength)
            support_rq_prefix_ids = sorted(prefixes_by_concept[left.id] | prefixes_by_concept[right.id])
            db.add(
                MidConceptEdge(
                    concept_state_id=state.id,
                    source_concept_id=left.id,
                    target_concept_id=right.id,
                    edge_type="chunk_edge_projection",
                    weight=raw_strength,
                    distance=distance,
                    projected_distance_raw=distance,
                    projected_strength_raw=raw_strength,
                    raw_strength_summary_json={
                        "max_raw_strength": raw_strength,
                        "mean_raw_strength": round(sum(raw_strengths) / max(len(raw_strengths), 1), 6),
                        "min_distance": distance,
                        "support_chunk_edge_count": len(support_chunk_edge_ids),
                        "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
                    },
                    projection_normalization_stats_json={
                        "normalization": "identity_distance_v1",
                        "support_edge_count": len(support_chunk_edge_ids),
                    },
                    edge_projection_protocol_hash=edge_projection_protocol_hash(),
                    network_evidence_score=raw_strength,
                    llm_confidence=0.7,
                    support_rq_prefix_ids_json=support_rq_prefix_ids,
                    support_chunk_ids_json=support_chunk_ids,
                    support_chunk_edge_ids_json=support_chunk_edge_ids[:80],
                    support_relation_edge_ids_json=support_chunk_edge_ids[:80],
                    support_rq_prefix_node_ids_json=support_rq_prefix_ids,
                    explanation="Concept edge admitted only from projected bottom chunk relation edges.",
                    diagnostics_json={
                        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                        "support_chunk_edge_count": len(support_chunk_edge_ids),
                        "support_edge_types": dict(Counter(edge.edge_type for edge in supporting_edges)),
                    },
                )
            )


async def build_coarse_concept_graph(
    db: Session,
    knowledge_base_id: str,
    mid_state: MidConceptState,
    *,
    batch_id: str | None = None,
    state_scope: str = "active",
    shadow_metadata: dict[str, Any] | None = None,
) -> CoarseConceptState:
    mid_concepts = list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == mid_state.id, MidConcept.state == state_scope)).all())
    mid_edges = list(db.scalars(select(MidConceptEdge).where(MidConceptEdge.concept_state_id == mid_state.id)).all())
    l2_prefixes = list(
        db.scalars(
            select(RQPrefix)
            .where(
                RQPrefix.graph_state_id == mid_state.chunk_relation_graph_state_id,
                RQPrefix.state == state_scope,
                RQPrefix.rq_level == 2,
            )
            .order_by(RQPrefix.rq_prefix_key.asc())
        ).all()
    )
    l2_prefixes = [prefix for prefix in l2_prefixes if prefix.support_chunk_ids_json]
    child_l3_prefix_ids_by_l2: dict[str, list[str]] = defaultdict(list)
    for child in db.scalars(
        select(RQPrefix).where(
            RQPrefix.graph_state_id == mid_state.chunk_relation_graph_state_id,
            RQPrefix.state == state_scope,
            RQPrefix.rq_level == 3,
            RQPrefix.parent_rq_prefix_id.is_not(None),
        )
    ).all():
        if child.parent_rq_prefix_id:
            child_l3_prefix_ids_by_l2[child.parent_rq_prefix_id].append(child.id)
    grounding_hash = stable_hash([prefix.id for prefix in l2_prefixes] + [concept.id for concept in mid_concepts] + [mid_state.state_hash])
    state = CoarseConceptState(
        knowledge_base_id=knowledge_base_id,
        mid_concept_state_id=mid_state.id,
        state_hash=stable_hash({"grounding": grounding_hash, "prompt": COARSE_CONCEPT_PROMPT_VERSION}),
        grounding_hash=grounding_hash,
        prompt_protocol_version=COARSE_CONCEPT_PROMPT_VERSION,
        stats_json={},
        diagnostics_json={**(shadow_metadata or {}), **({"shadow_scope": True} if state_scope == "shadow" else {})},
        state=state_scope,
    )
    db.add(state)
    db.flush()
    communities = coarse_rq_l2_communities(mid_concepts, l2_prefixes)
    coarse_work_items: list[dict[str, Any]] = []
    coarse_concepts: list[CoarseConcept] = []
    used_coarse_labels: set[str] = set()
    for index, (l2_prefix, community) in enumerate(communities, start=1):
        mid_ids = [concept.id for concept in community]
        mid_id_set = set(mid_ids)
        support_mid_edges = [
            edge
            for edge in mid_edges
            if edge.source_concept_id in mid_id_set or edge.target_concept_id in mid_id_set
        ]
        support_chunks = list(
            dict.fromkeys(
                [chunk_id for chunk_id in (l2_prefix.support_chunk_ids_json or [])]
                + [
                    chunk_id
                    for concept in community
                    for chunk_id in (concept.support_chunk_ids_json or [])
                ]
            )
        )[:80]
        support_chunk_edge_ids = list(
            dict.fromkeys(
                chunk_edge_id
                for concept in community
                for chunk_edge_id in (concept.support_chunk_edge_ids_json or [])
            )
        )[:50]
        parent_l2_id = l2_prefix.id
        parent_l1_id = l2_prefix.parent_rq_prefix_id
        child_l3_ids = list(
            dict.fromkeys(
                child_l3_prefix_ids_by_l2.get(l2_prefix.id, [])
                + [
                    prefix_id
                    for concept in community
                    for prefix_id in ([concept.support_rq_l3_prefix_id] if concept.support_rq_l3_prefix_id else (concept.support_rq_prefix_ids_json or []))
                    if prefix_id
                ]
            )
        )
        packet = {
            "community_id": index,
            "support_rq_l2_prefix_id": parent_l2_id,
            "parent_rq_l1_prefix_id": parent_l1_id,
            "child_rq_l3_prefix_ids": child_l3_ids,
            "mid_concepts": [
                {
                    "id": concept.id,
                    "label": concept.canonical_label,
                    "definition": concept.definition,
                    "support_chunk_ids": concept.support_chunk_ids_json,
                }
                for concept in community
            ],
            "bridge_concepts": [concept.id for concept in community if len(concept.support_chunk_ids_json or []) > 2],
            "support_mid_edge_ids": [edge.id for edge in support_mid_edges],
            "support_chunk_ids": support_chunks,
            "support_chunk_edge_ids": support_chunk_edge_ids,
            "community_diagnostics": {"candidate_source": "rq_l2_prefix_projection"},
            "grounding_hash": stable_hash([l2_prefix.id, *mid_ids, *support_chunks]),
        }
        coarse_work_items.append(
            {
                "index": index,
                "l2_prefix": l2_prefix,
                "community": community,
                "mid_ids": mid_ids,
                "support_chunks": support_chunks,
                "support_chunk_edge_ids": support_chunk_edge_ids,
                "parent_l2_id": parent_l2_id,
                "parent_l1_id": parent_l1_id,
                "child_l3_ids": child_l3_ids,
                "packet": packet,
            }
        )

    settings = get_settings()
    model_concurrency = settings.model_request_concurrency
    context_graph_batch_heartbeat(
        batch_id,
        "coarse_concepts",
        {"llm_batches": len(coarse_work_items), "projected_rq_l2_prefixes": len(l2_prefixes), "model_concurrency": model_concurrency},
    )

    async def define_coarse_work_item(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return item, await define_coarse_concept(item["packet"])

    completed_coarse_batches = 0

    def coarse_batch_progress(_result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        nonlocal completed_coarse_batches
        completed_coarse_batches += 1
        total_batches = len(coarse_work_items)
        if completed_coarse_batches == 1 or completed_coarse_batches == total_batches or completed_coarse_batches % 5 == 0:
            context_graph_batch_heartbeat(
                batch_id,
                "coarse_concepts",
                {
                    "completed_llm_batches": completed_coarse_batches,
                    "llm_batches": total_batches,
                    "projected_rq_l2_prefixes": len(l2_prefixes),
                    "model_concurrency": model_concurrency,
                },
            )

    coarse_outputs = await gather_bounded(coarse_work_items, model_concurrency, define_coarse_work_item, on_result=coarse_batch_progress)
    for item, output in sorted(coarse_outputs, key=lambda pair: int(pair[0]["index"])):
        l2_prefix = item["l2_prefix"]
        community = item["community"]
        mid_ids = item["mid_ids"]
        support_chunks = item["support_chunks"]
        support_chunk_edge_ids = item["support_chunk_edge_ids"]
        parent_l2_id = item["parent_l2_id"]
        parent_l1_id = item["parent_l1_id"]
        child_l3_ids = item["child_l3_ids"]
        packet = item["packet"]
        confidence, confidence_diagnostics = coerce_confidence(output.get("confidence"), default=0.72)
        raw_node_weight = float(len(mid_ids)) + 0.15 * float(len(support_chunks)) + 0.25 * float(len(support_chunk_edge_ids))
        coarse_label = unique_concept_label(
            str(output.get("coarse_label") or output.get("canonical_label") or l2_prefix.label),
            l2_prefix.label,
            used_coarse_labels,
        )
        coarse = CoarseConcept(
            coarse_state_id=state.id,
            knowledge_base_id=knowledge_base_id,
            canonical_label=coarse_label,
            aliases_json=list(output.get("aliases") or []),
            support_rq_l2_prefix_id=parent_l2_id,
            parent_rq_l1_prefix_id=parent_l1_id,
            child_rq_l3_prefix_ids_json=child_l3_ids,
            definition=str(output.get("definition") or "A high-level topic area grounded in mid-level concepts."),
            summary=str(output.get("summary") or output.get("definition") or "A high-level topic area grounded in mid-level concepts."),
            scope_note=str(output.get("scope_note") or ""),
            inclusion_criteria_json=list(output.get("inclusion_criteria") or []),
            exclusion_criteria_json=list(output.get("exclusion_criteria") or []),
            display_terms_json=list(dict.fromkeys([coarse_label, *list(output.get("aliases") or [])]))[:12],
            internal_state_json={
                "support_rq_l2_prefix_id": parent_l2_id,
                "parent_rq_l1_prefix_id": parent_l1_id,
                "child_rq_l3_prefix_ids": child_l3_ids,
            },
            included_mid_concept_ids_json=mid_ids,
            boundary_mid_concept_ids_json=list(output.get("boundary_concepts") or [])[:10],
            bridge_mid_concept_ids_json=list(output.get("bridge_concepts") or packet["bridge_concepts"])[:10],
            outlier_mid_concept_ids_json=[],
            support_chunk_ids_json=support_chunks,
            support_chunk_edge_ids_json=support_chunk_edge_ids,
            cross_community_weak_ties_json=list(output.get("cross_community_weak_ties") or []),
            raw_node_weight=raw_node_weight,
            node_weight=raw_node_weight,
            node_weight_diagnostics_json={
                "normalization_pending": True,
                "included_mid_concept_count": len(mid_ids),
                "support_chunk_count": len(support_chunks),
                "support_chunk_edge_count": len(support_chunk_edge_ids),
            },
            confidence=confidence,
            llm_audit_json={
                "prompt_protocol_version": COARSE_CONCEPT_PROMPT_VERSION,
                "packet": packet,
                "raw_output": output,
                "confidence": confidence_diagnostics,
            },
            grounding_hash=stable_hash({"rq_l2_prefix": l2_prefix.id, "mid": mid_ids, "chunks": support_chunks}),
            state=state.state,
        )
        db.add(coarse)
        db.flush()
        db.add(
            CoarseConceptDefinition(
                coarse_concept_id=coarse.id,
                definition_json={"canonical_label": coarse.canonical_label, "definition": coarse.definition},
                llm_audit_json=coarse.llm_audit_json,
            )
        )
        for concept in community:
            db.add(
                CoarseConceptMembership(
                    coarse_concept_id=coarse.id,
                    mid_concept_id=concept.id,
                    membership_score=1.0,
                    role="bridge" if concept.id in (coarse.bridge_mid_concept_ids_json or []) else "included",
                )
        )
        coarse_concepts.append(coarse)
    concept_i18n_metrics = {
        "translation_phase": "concept_i18n",
        "translation_items": len(coarse_concepts),
        "translation_enabled": settings.concept_i18n_enabled,
    }
    if settings.concept_i18n_enabled:
        context_graph_batch_heartbeat(batch_id, "coarse_concepts", concept_i18n_metrics)
        concept_i18n_stats = await enrich_concepts_i18n(coarse_concepts, layer="coarse")
    else:
        context_graph_batch_heartbeat(batch_id, "coarse_concepts", {**concept_i18n_metrics, "translation_status": "disabled"})
        concept_i18n_stats = disabled_concept_i18n_stats(len(coarse_concepts), layer="coarse", target="concept")
    normalize_concept_node_weights(coarse_concepts, "coarse_concept_state")
    chunk_edges = list(
        db.scalars(select(ChunkRelationEdge).where(ChunkRelationEdge.graph_state_id == mid_state.chunk_relation_graph_state_id)).all()
    )
    chunk_edges_by_id = {edge.id: edge for edge in chunk_edges if edge.id}
    chunk_edge_index = {(edge.source_chunk_id, edge.target_chunk_id, edge.edge_type): edge for edge in chunk_edges}
    for index, left in enumerate(coarse_concepts):
        for right in coarse_concepts[index + 1 :]:
            left_mid_ids = set(left.included_mid_concept_ids_json or [])
            right_mid_ids = set(right.included_mid_concept_ids_json or [])
            projected_mid_edges = [
                edge
                for edge in mid_edges
                if (
                    edge.source_concept_id in left_mid_ids
                    and edge.target_concept_id in right_mid_ids
                )
                or (
                    edge.source_concept_id in right_mid_ids
                    and edge.target_concept_id in left_mid_ids
                )
            ]
            left_chunks = set(left.support_chunk_ids_json or [])
            right_chunks = set(right.support_chunk_ids_json or [])
            support_chunk_edge_ids = support_chunk_edge_ids_between(left_chunks, right_chunks, chunk_edge_index)
            supporting_chunk_edges = [chunk_edges_by_id[edge_id] for edge_id in support_chunk_edge_ids if edge_id in chunk_edges_by_id]
            if not supporting_chunk_edges:
                continue
            raw_strengths = [
                normalized_strength(float(edge.raw_strength or edge.weight or 0.1))
                for edge in supporting_chunk_edges
            ]
            distances = [
                float(edge.distance if edge.distance is not None else distance_from_strength(raw_strengths[idx]))
                for idx, edge in enumerate(supporting_chunk_edges)
            ]
            raw_strength = max(raw_strengths) if raw_strengths else 0.1
            distance = min(distances) if distances else distance_from_strength(raw_strength)
            support_mid_edge_ids = [edge.id for edge in projected_mid_edges]
            support_chunk_ids = list(
                dict.fromkeys(
                    chunk_id
                    for edge in supporting_chunk_edges
                    for chunk_id in (edge.source_chunk_id, edge.target_chunk_id)
                )
            )
            weak_ties = [
                {
                    "mid_edge_id": edge.id,
                    "source_mid_concept_id": edge.source_concept_id,
                    "target_mid_concept_id": edge.target_concept_id,
                    "distance": edge.distance,
                    "raw_strength_summary": edge.raw_strength_summary_json or {},
                }
                for edge in projected_mid_edges
            ]
            db.add(
                CoarseConceptEdge(
                    coarse_state_id=state.id,
                    source_concept_id=left.id,
                    target_concept_id=right.id,
                    edge_type="chunk_edge_projection",
                    weight=raw_strength,
                    distance=distance,
                    projected_distance_raw=distance,
                    projected_strength_raw=raw_strength,
                    raw_strength_summary_json={
                        "max_raw_strength": raw_strength,
                        "mean_raw_strength": round(sum(raw_strengths) / max(len(raw_strengths), 1), 6),
                        "min_distance": distance,
                        "support_mid_edge_count": len(support_mid_edge_ids),
                        "support_chunk_edge_count": len(support_chunk_edge_ids),
                        "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
                    },
                    projection_normalization_stats_json={
                        "normalization": "identity_distance_v1",
                        "support_mid_edge_count": len(support_mid_edge_ids),
                        "support_chunk_edge_count": len(support_chunk_edge_ids),
                    },
                    edge_projection_protocol_hash=edge_projection_protocol_hash(),
                    support_mid_concept_ids_json=list(left_mid_ids | right_mid_ids)[:20],
                    support_mid_edge_ids_json=support_mid_edge_ids,
                    support_chunk_ids_json=support_chunk_ids[:60],
                    support_chunk_edge_ids_json=support_chunk_edge_ids[:120],
                    cross_community_weak_ties_json=weak_ties,
                    explanation="Coarse edge admitted only from projected bottom chunk relation edges.",
                    diagnostics_json={
                        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                        "support_mid_edge_count": len(support_mid_edge_ids),
                        "support_chunk_edge_count": len(support_chunk_edge_ids),
                        "support_edge_types": dict(Counter(edge.edge_type for edge in supporting_chunk_edges)),
                    },
                )
            )
    db.flush()
    coarse_edges = list(db.scalars(select(CoarseConceptEdge).where(CoarseConceptEdge.coarse_state_id == state.id)).all())
    edge_i18n_metrics = {
        "translation_phase": "edge_i18n",
        "translation_items": len(coarse_edges),
        "translation_enabled": settings.concept_i18n_enabled,
    }
    if settings.concept_i18n_enabled:
        context_graph_batch_heartbeat(batch_id, "coarse_concepts", edge_i18n_metrics)
        edge_i18n_stats = await enrich_concept_edges_i18n(coarse_edges, {concept.id: concept for concept in coarse_concepts}, layer="coarse")
    else:
        context_graph_batch_heartbeat(batch_id, "coarse_concepts", {**edge_i18n_metrics, "translation_status": "disabled"})
        edge_i18n_stats = disabled_concept_i18n_stats(len(coarse_edges), layer="coarse", target="edge")
    db.flush()
    supported_coarse_edges = sum(1 for edge in coarse_edges if edge.support_chunk_edge_ids_json)
    stats = {
        "coarse_concept_count": len(coarse_concepts),
        "mid_concept_count": len(mid_concepts),
        "rq_l2_prefix_candidates": len(l2_prefixes),
        "projected_rq_l2_prefixes": len(coarse_concepts),
        "rq_l2_to_coarse_projection_coverage": round(len(coarse_concepts) / max(len(l2_prefixes), 1), 6) if l2_prefixes else 1.0,
        "coarse_edge_count": len(coarse_edges),
        "coarse_edge_support_chunk_edge_coverage": round(supported_coarse_edges / max(len(coarse_edges), 1), 6) if coarse_edges else 1.0,
        "bridge_concept_count": sum(len(item.bridge_mid_concept_ids_json or []) for item in coarse_concepts),
        "singleton_rate": round(sum(1 for item in coarse_concepts if len(item.included_mid_concept_ids_json or []) <= 1) / max(len(coarse_concepts), 1), 6),
        "concept_i18n_enabled": bool(settings.concept_i18n_enabled),
        "concept_i18n_translated_count": concept_i18n_stats.get("translated_count", 0),
        "edge_i18n_translated_count": edge_i18n_stats.get("translated_count", 0),
    }
    community_diagnostics = coarse_community_diagnostics(mid_concepts, mid_edges, [community for _prefix, community in communities])
    state.stats_json = stats
    state.diagnostics_json = {
        **(state.diagnostics_json or {}),
        "community_detection": "rq_l2_prefix_projection_v1",
        "legacy_label_bucket_active": False,
        "connected_components_used_as_final": False,
        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
        "projected_rq_l2_prefix_ids": [prefix.id for prefix, _community in communities],
        **community_diagnostics,
        "bridge_density": round(stats["bridge_concept_count"] / max(stats["mid_concept_count"], 1), 6),
        "concept_i18n": concept_i18n_stats,
        "edge_i18n": edge_i18n_stats,
    }
    state.state_hash = stable_hash({"coarse": [concept.id for concept in coarse_concepts], "stats": stats, "diagnostics": state.diagnostics_json})
    return state


def coarse_rq_l2_communities(mid_concepts: list[MidConcept], l2_prefixes: list[RQPrefix]) -> list[tuple[RQPrefix, list[MidConcept]]]:
    if not l2_prefixes:
        return []
    groups: dict[str, list[MidConcept]] = defaultdict(list)
    for concept in mid_concepts:
        if concept.parent_rq_l2_prefix_id:
            groups[concept.parent_rq_l2_prefix_id].append(concept)
    return [
        (prefix, sorted(groups.get(prefix.id, []), key=lambda item: item.canonical_label))
        for prefix in sorted(l2_prefixes, key=lambda item: (tuple(item.rq_path_prefix or []), item.rq_prefix_key or "", item.id))
    ]


def coarse_communities(mid_concepts: list[MidConcept], mid_edges: list[MidConceptEdge] | None = None) -> list[list[MidConcept]]:
    if not mid_concepts:
        return []
    if len(mid_concepts) <= 3:
        return [mid_concepts]
    concept_by_id = {concept.id: concept for concept in mid_concepts}
    edges = [
        edge
        for edge in (mid_edges or [])
        if edge.source_concept_id in concept_by_id and edge.target_concept_id in concept_by_id
    ]
    if not edges:
        return [mid_concepts]
    try:
        import networkx as nx

        graph = nx.Graph()
        for concept in mid_concepts:
            graph.add_node(concept.id)
        for edge in edges:
            raw_strength = normalized_strength(
                float((edge.raw_strength_summary_json or {}).get("max_raw_strength") or edge.weight or 0.1)
            )
            distance = float(edge.distance if edge.distance is not None else distance_from_strength(raw_strength))
            graph.add_edge(edge.source_concept_id, edge.target_concept_id, weight=1.0 / max(distance, 1e-6))
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
        grouped = [
            [concept_by_id[concept_id] for concept_id in sorted(group) if concept_id in concept_by_id]
            for group in communities
        ]
        grouped = [group for group in grouped if group]
        return grouped or [mid_concepts]
    except Exception:
        return [mid_concepts]


def coarse_community_diagnostics(
    mid_concepts: list[MidConcept],
    mid_edges: list[MidConceptEdge],
    communities: list[list[MidConcept]],
) -> dict[str, Any]:
    concept_ids = {concept.id for concept in mid_concepts}
    if not concept_ids:
        return {
            "modularity": 0.0,
            "conductance": 0.0,
            "betweenness_proxy": 0.0,
            "bridge_density": 0.0,
            "community_stability": 1.0,
            "singleton_rate": 0.0,
        }
    community_by_concept: dict[str, int] = {}
    for index, community in enumerate(communities):
        for concept in community:
            community_by_concept[concept.id] = index
    weighted_edges = []
    for edge in mid_edges:
        if edge.source_concept_id not in concept_ids or edge.target_concept_id not in concept_ids:
            continue
        raw_strength = normalized_strength(
            float((edge.raw_strength_summary_json or {}).get("max_raw_strength") or edge.weight or 0.0)
        )
        if edge.distance is not None:
            raw_strength = max(raw_strength, 1.0 / (1.0 + float(edge.distance)))
        weighted_edges.append((edge.source_concept_id, edge.target_concept_id, raw_strength))
    total_weight = sum(weight for _source, _target, weight in weighted_edges)
    if total_weight <= 0:
        singleton_rate = sum(1 for community in communities if len(community) <= 1) / max(len(communities), 1)
        return {
            "modularity": 0.0,
            "conductance": 0.0,
            "betweenness_proxy": 0.0,
            "bridge_density": 0.0,
            "community_stability": round(1.0 - singleton_rate, 6),
            "singleton_rate": round(singleton_rate, 6),
        }
    degree: Counter[str] = Counter()
    internal_weight = 0.0
    cut_weight_by_community: Counter[int] = Counter()
    volume_by_community: Counter[int] = Counter()
    cross_edges = 0
    for source, target, weight in weighted_edges:
        degree[source] += weight
        degree[target] += weight
        source_comm = community_by_concept.get(source, -1)
        target_comm = community_by_concept.get(target, -1)
        volume_by_community[source_comm] += weight
        volume_by_community[target_comm] += weight
        if source_comm == target_comm:
            internal_weight += weight
        else:
            cross_edges += 1
            cut_weight_by_community[source_comm] += weight
            cut_weight_by_community[target_comm] += weight
    modularity = 0.0
    m2 = 2.0 * total_weight
    for source, target, weight in weighted_edges:
        if community_by_concept.get(source) == community_by_concept.get(target):
            modularity += weight - (float(degree[source]) * float(degree[target]) / max(m2, 1e-9))
    modularity = modularity / max(m2, 1e-9)
    conductance_values = []
    for community_id, volume in volume_by_community.items():
        if community_id < 0 or volume <= 0:
            continue
        cut = float(cut_weight_by_community.get(community_id, 0.0))
        conductance_values.append(cut / max(volume, 1e-9))
    singleton_rate = sum(1 for community in communities if len(community) <= 1) / max(len(communities), 1)
    return {
        "modularity": round(modularity, 6),
        "conductance": round(sum(conductance_values) / max(len(conductance_values), 1), 6),
        "betweenness_proxy": round(cross_edges / max(len(weighted_edges), 1), 6),
        "bridge_density": round(cross_edges / max(len(mid_concepts), 1), 6),
        "community_stability": round(max(0.0, min(1.0, (internal_weight / total_weight) * (1.0 - singleton_rate))), 6),
        "singleton_rate": round(singleton_rate, 6),
    }


async def define_coarse_concept(packet: dict[str, Any]) -> dict[str, Any]:
    labels = [item.get("label") for item in packet.get("mid_concepts", []) if item.get("label")]
    fallback = {
        "coarse_label": " / ".join(labels[:2]) or "Grounded Topic Area",
        "definition": "A high-level topic area grounded in the listed mid-level concepts.",
        "included_mid_concepts": [item.get("id") for item in packet.get("mid_concepts", [])],
        "boundary_concepts": [],
        "bridge_concepts": packet.get("bridge_concepts") or [],
        "cross_community_weak_ties": [],
        "confidence": 0.72,
    }
    system = (
        "You define coarse topic areas for a Four-Layer Context Graph RAG system. "
        "Use only the supplied mid concept community. Return strict JSON with coarse_label, definition, "
        "included_mid_concepts, boundary_concepts, bridge_concepts, cross_community_weak_ties, and confidence."
    )
    try:
        output = await ChatProvider(purpose="graph").classify_json(system_prompt=system, user_prompt=str(packet), fallback=fallback)
    except FallbackDisabledError:
        raise
    required = {
        "coarse_label",
        "definition",
        "included_mid_concepts",
        "boundary_concepts",
        "bridge_concepts",
        "cross_community_weak_ties",
        "confidence",
    }
    if not isinstance(output, dict) or not required.issubset(output):
        if get_settings().enable_model_fallback:
            return fallback
        missing = sorted(required.difference(output or {})) if isinstance(output, dict) else sorted(required)
        raise RuntimeError(f"Coarse concept provider returned invalid JSON; missing: {', '.join(missing)}")
    return output


def write_context_graph_state(
    db: Session,
    knowledge_base_id: str,
    relation_state: ChunkRelationGraphState,
    mid_state: MidConceptState,
    coarse_state: CoarseConceptState,
    chunks: list[Chunk],
    *,
    state_scope: str = "active",
    shadow_metadata: dict[str, Any] | None = None,
) -> ContextGraphState:
    structure_hash = stable_hash(
        [
            row
            for row in db.execute(
                select(ChunkStructureMapping.chunk_id, ChunkStructureMapping.structure_node_id, ChunkStructureMapping.coverage_ratio)
                .where(ChunkStructureMapping.chunk_id.in_([chunk.id for chunk in chunks]))
            ).all()
        ]
    )
    rq_prefix_hash = stable_hash([cluster.id for cluster in db.scalars(select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id)).all()])
    chunk_scope_hash = compute_chunk_scope_hash(chunks)
    context_hash = stable_hash(
        {
            "chunk_scope": chunk_scope_hash,
            "structure": structure_hash,
            "relation": relation_state.state_hash,
            "rq_membership": rq_prefix_hash,
            "mid": mid_state.state_hash,
            "coarse": coarse_state.state_hash,
            "runtime_settings": runtime_settings_state_hash(),
            "agent_operating_envelope": agent_operating_envelope_state_hash(),
            "state_scope": state_scope,
            "shadow_metadata": shadow_metadata or {},
        }
    )
    state = ContextGraphState(
        knowledge_base_id=knowledge_base_id,
        chunk_relation_graph_state_id=relation_state.id,
        mid_concept_state_id=mid_state.id,
        coarse_concept_state_id=coarse_state.id,
        chunk_scope_hash=chunk_scope_hash,
        structure_graph_hash=structure_hash,
        chunk_relation_graph_hash=relation_state.state_hash,
        rq_membership_hash=rq_prefix_hash,
        mid_concept_hash=mid_state.state_hash,
        coarse_concept_hash=coarse_state.state_hash,
        context_graph_hash=context_hash,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        prompt_protocol_hash=stable_hash([MID_CONCEPT_PROMPT_VERSION, COARSE_CONCEPT_PROMPT_VERSION, ANSWER_PROMPT_PROTOCOL_VERSION]),
        stats_json={
            "chunks": len(chunks),
            "relation_edges": db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state.id)) or 0,
            "rq_prefixes": db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.graph_state_id == relation_state.id)) or 0,
            "mid_concepts": db.scalar(select(func.count(MidConcept.id)).where(MidConcept.concept_state_id == mid_state.id)) or 0,
            "coarse_concepts": db.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.coarse_state_id == coarse_state.id)) or 0,
        },
        diagnostics_json={**(shadow_metadata or {}), "protocol": CONTEXT_GRAPH_PROTOCOL_VERSION, **({"shadow_scope": True} if state_scope == "shadow" else {})},
        state=state_scope,
    )
    db.add(state)
    db.flush()
    for layer, layer_hash in {
        "chunk_structure": structure_hash,
        "chunk_relation": relation_state.state_hash,
        "mid_concepts": mid_state.state_hash,
        "coarse_concepts": coarse_state.state_hash,
        "context_graph": context_hash,
    }.items():
        db.add(
            ContextGraphFreshness(
                knowledge_base_id=knowledge_base_id,
                context_graph_state_id=state.id,
                layer=layer,
                state_hash=layer_hash,
                is_stale=False,
                stale_reasons_json=[],
            )
        )
    return state


@dataclass
class LayeredSearchResult:
    results: list[dict[str, Any]]
    trace: RetrievalTrace
    audit: dict[str, Any]


async def layered_search(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int | None,
    *,
    query_facets: dict[str, Any] | None = None,
    retrieval_granularity: RetrievalGranularity = "mid",
) -> LayeredSearchResult:
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    query_facets = query_facets_for_search(query, query_facets)
    result_top_k = resolve_result_top_k(top_k)
    if not chunks:
        trace = RetrievalTrace(
            knowledge_base_id=knowledge_base_id,
            query=query,
            filters_json=filters.model_dump(),
            result_chunk_ids_json=[],
            query_facets_json=query_facets,
            convergence_json={"reason": "no_active_chunks"},
            diagnostics_json={
                "reason": "no_active_chunks",
                "result_top_k": result_top_k,
                "retrieval_granularity": retrieval_granularity,
                "cache_key_components": {"retrieval_granularity": retrieval_granularity},
            },
            edge_distance_protocol_hash=edge_distance_protocol_hash(),
            edge_projection_protocol_hash=edge_projection_protocol_hash(),
            traversal_protocol_hash=traversal_protocol_hash(),
        )
        db.add(trace)
        db.flush()
        return LayeredSearchResult(
            [],
            trace,
            {
                "retrieval_pipeline": "layered_context_graph",
                "retrieval_granularity": retrieval_granularity,
                "reason": "no_active_chunks",
                "result_top_k": result_top_k,
            },
        )
    context_state = latest_context_graph_state(db, knowledge_base_id)
    relation_state = latest_relation_state(db, knowledge_base_id)
    query_vector = (await EmbeddingProvider().embed_texts([query], text_type="query"))[0]
    query_rq = encode_query_rq(relation_state, query_vector)
    coarse_entries = select_coarse_entries(db, knowledge_base_id, query_facets) if retrieval_granularity == "coarse" else {}
    mid_entries = select_mid_entries(db, knowledge_base_id, query_facets, coarse_entries)
    rq_membership_entries = select_rq_membership_entries(db, knowledge_base_id, query_vector, query_rq, mid_entries)
    dense_entries = dense_chunk_entries(db, knowledge_base_id, query_vector)
    traversal = execute_priority_queue_traversal(
        db,
        knowledge_base_id=knowledge_base_id,
        chunks=chunks,
        filters=filters,
        query_facets=query_facets,
        coarse_entries=coarse_entries,
        mid_entries=mid_entries,
        rq_membership_entries=rq_membership_entries,
        dense_entries=dense_entries,
        query_rq=query_rq,
        top_k=result_top_k,
        retrieval_granularity=retrieval_granularity,
    )
    results = traversal["results"]
    trace = write_retrieval_trace(
        db,
        knowledge_base_id,
        query,
        filters,
        results,
        context_state,
        traversal,
        query_rq,
        result_top_k,
        retrieval_granularity,
    )
    for item in results:
        item["metadata"]["retrieval_trace_id"] = trace.id
        for citation in item["citations"]:
            citation["retrieval_trace_id"] = trace.id
            if isinstance(citation.get("source_span"), dict):
                citation["source_span"]["retrieval_trace_id"] = trace.id
    audit = {
        "retrieval_pipeline": "layered_context_graph",
        "retrieval_granularity": retrieval_granularity,
        "retrieval_trace_id": trace.id,
        "context_graph_state_id": context_state.id if context_state else None,
        "degraded_mode": is_degraded_mode(),
        "result_top_k": result_top_k,
        "coarse_entries": len(coarse_entries),
        "mid_entries": len(mid_entries),
        "rq_membership_entries": len(rq_membership_entries),
        "frontier_pops": len(traversal["frontier_pops"]),
        "stage_queue_count": sum(len((value or {}).get("selected_ids") or []) for value in (traversal.get("stage_queues") or {}).values() if isinstance(value, dict)),
        "mid_topk_selected": len(((traversal.get("topk_selection") or {}).get("mid") or {}).get("selected_ids") or []),
        "chunk_topk_selected": len(((traversal.get("topk_selection") or {}).get("chunk") or {}).get("selected_ids") or []),
        "dominance_pruned_count": traversal["convergence"]["dominance_pruned_count"],
        "hard_stop_pruned_count": traversal["convergence"].get("hard_stop_pruned_count", 0),
        "red_zone_pruned_count": traversal["convergence"].get("red_zone_pruned_count", 0),
        "gray_zone_decision_count": traversal["convergence"].get("gray_zone_decision_count", 0),
        "query_rq_path": query_rq.get("rq_path") if query_rq else [],
        "coarse_skipped_reason": (traversal.get("retrieval_granularity_audit") or {}).get("coarse_skipped_reason"),
        "mid_direct_entry_count": (traversal.get("retrieval_granularity_audit") or {}).get("mid_direct_entry_count"),
    }
    return LayeredSearchResult(results, trace, audit)


def _normalize_query_facet_text(value: Any, *, max_length: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_length]


def _facet_dedupe_key(value: str) -> str:
    return _normalize_query_facet_text(value).casefold()


def _dedupe_query_values(values: list[Any], *, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = _normalize_query_facet_text(value)
        if not text:
            continue
        key = _facet_dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def _query_term_allowed(term: Any) -> bool:
    text = _normalize_query_facet_text(term, max_length=48)
    if not text:
        return False
    lowered = text.casefold()
    if lowered in QUERY_FACET_STOP_TERMS:
        return False
    if len(text) == 1 and not re.fullmatch(r"[A-Za-z0-9]|[=<>+\-*/^(),.;:_]+", text):
        return False
    return True


def _query_terms_from_values(values: list[Any]) -> list[str]:
    terms: list[str] = []
    for value in values:
        text = _normalize_query_facet_text(value)
        if not text:
            continue
        tokenized_terms = tokenize_for_search_terms(text)
        if not tokenized_terms and _query_term_allowed(text):
            terms.append(text.casefold())
        for term in tokenized_terms:
            if _query_term_allowed(term):
                terms.append(term.casefold())
    return _dedupe_query_values(terms, max_items=QUERY_FACET_MAX_TERMS)


def _coerce_facet_group(raw: Any, *, role: str) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        facet = _normalize_query_facet_text(
            raw.get("facet")
            or raw.get("label")
            or raw.get("name")
            or raw.get("term")
            or raw.get("value")
        )
        aliases = raw.get("aliases") or raw.get("alias") or raw.get("search_terms") or raw.get("terms") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        elif not isinstance(aliases, list):
            aliases = []
        raw_role = _normalize_query_facet_text(raw.get("role") or role, max_length=32) or role
        source = _normalize_query_facet_text(raw.get("source") or "llm", max_length=32) or "llm"
        confidence = coerce_confidence(raw.get("confidence"), default=0.7)
    else:
        facet = _normalize_query_facet_text(raw)
        aliases = []
        raw_role = role
        source = "llm"
        confidence = 0.7
    if not facet or not _query_term_allowed(facet):
        return None
    return {
        "facet": facet,
        "role": raw_role,
        "aliases": _dedupe_query_values(list(aliases), max_items=QUERY_FACET_MAX_ALIASES),
        "source": source,
        "confidence": confidence,
    }


def _facet_groups_from_llm_payload(llm_facets: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(llm_facets, dict):
        return []
    groups: list[dict[str, Any]] = []
    raw_groups = llm_facets.get("facet_groups")
    if isinstance(raw_groups, list):
        for item in raw_groups:
            group = _coerce_facet_group(item, role="domain")
            if group is not None:
                groups.append(group)
    keyed_roles = {
        "domain_facets": "domain",
        "entity_facets": "domain",
        "procedure_facets": "procedure",
        "algorithm_facets": "procedure",
        "constraint_facets": "constraint",
        "alias_facets": "alias",
        "required_facets": "required",
    }
    for key, role in keyed_roles.items():
        values = llm_facets.get(key)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for item in values:
            group = _coerce_facet_group(item, role=role)
            if group is not None:
                groups.append(group)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        key = _facet_dedupe_key(group["facet"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(group)
        if len(deduped) >= QUERY_FACET_MAX_GROUPS:
            break
    return deduped


def query_facets_for_search(query: str, llm_facets: dict[str, Any] | None = None, query_intent: dict[str, Any] | None = None) -> dict[str, Any]:
    lexical_terms = _query_terms_from_values(tokenize_for_search_terms(query))
    facet_groups = _facet_groups_from_llm_payload(llm_facets)
    if not facet_groups:
        facet_groups = [
            {
                "facet": term,
                "role": "lexical",
                "aliases": [],
                "source": "deterministic_tokenizer",
                "confidence": 0.55,
            }
            for term in lexical_terms[:QUERY_FACET_MAX_REQUIRED]
        ]
    raw_drop_terms = (llm_facets or {}).get("drop_terms") or []
    if isinstance(raw_drop_terms, str):
        raw_drop_terms = [raw_drop_terms]
    if not isinstance(raw_drop_terms, list):
        raw_drop_terms = []
    drop_terms = _dedupe_query_values(list(raw_drop_terms) + sorted(QUERY_FACET_STOP_TERMS), max_items=64)
    dropped = {_facet_dedupe_key(term) for term in drop_terms}
    required_facets = [
        group["facet"]
        for group in facet_groups
        if _facet_dedupe_key(group["facet"]) not in dropped and group.get("role") != "alias"
    ][:QUERY_FACET_MAX_REQUIRED]
    if not required_facets:
        required_facets = lexical_terms[:QUERY_FACET_MAX_REQUIRED]
    term_sources: list[Any] = [*lexical_terms]
    for group in facet_groups:
        term_sources.append(group["facet"])
        term_sources.extend(group.get("aliases") or [])
    terms = [
        term
        for term in _query_terms_from_values(term_sources)
        if _facet_dedupe_key(term) not in dropped
    ][:QUERY_FACET_MAX_TERMS]
    if not terms:
        terms = lexical_terms[:QUERY_FACET_MAX_TERMS]
    lower_query = query.lower()
    intent = str((query_intent or {}).get("intent") or (llm_facets or {}).get("intent") or "")
    if not intent:
        intent = "formula_table_lookup" if any(term in lower_query for term in ("formula", "equation", "table", "\u516c\u5f0f", "\u8868\u683c")) else "semantic"
    return {
        "query": query,
        "protocol_version": QUERY_FACET_PROTOCOL_VERSION,
        "terms": terms,
        "required_facets": _dedupe_query_values(required_facets, max_items=QUERY_FACET_MAX_REQUIRED),
        "facet_groups": facet_groups,
        "drop_terms": drop_terms,
        "answer_shape": _normalize_query_facet_text((llm_facets or {}).get("answer_shape") or (query_intent or {}).get("intent") or "grounded_answer", max_length=48),
        "intent": intent,
        "diagnostics": {
            "source": "llm_validated" if isinstance(llm_facets, dict) and llm_facets else "deterministic_tokenizer",
            "lexical_terms": lexical_terms,
            "dropped_query_terms": [term for term in tokenize_for_search_terms(query) if _facet_dedupe_key(term) in dropped or not _query_term_allowed(term)],
            "llm_keys": sorted(llm_facets.keys()) if isinstance(llm_facets, dict) else [],
        },
    }


def matched_required_facets_for_text(text: str, query_facets: dict[str, Any]) -> list[str]:
    required_facets = list(query_facets.get("required_facets") or [])
    if not required_facets:
        return []
    text_value = str(text or "")
    text_lower = text_value.casefold()
    text_terms = set(term.casefold() for term in tokenize_for_search_terms(text_value))
    groups_by_facet = {
        _facet_dedupe_key(group.get("facet")): group
        for group in (query_facets.get("facet_groups") or [])
        if isinstance(group, dict) and group.get("facet")
    }
    matched: list[str] = []
    for facet in required_facets:
        facet_text = _normalize_query_facet_text(facet)
        if not facet_text:
            continue
        group = groups_by_facet.get(_facet_dedupe_key(facet_text)) or {"facet": facet_text, "aliases": []}
        candidates = [group.get("facet"), *(group.get("aliases") or [])]
        for candidate in candidates:
            candidate_text = _normalize_query_facet_text(candidate)
            if not candidate_text:
                continue
            candidate_lower = candidate_text.casefold()
            candidate_terms = [term.casefold() for term in tokenize_for_search_terms(candidate_text) if _query_term_allowed(term)]
            if candidate_lower in text_lower or (candidate_terms and set(candidate_terms).issubset(text_terms)):
                matched.append(facet_text)
                break
    return _dedupe_query_values(matched)


def select_coarse_entries(db: Session, knowledge_base_id: str, query_facets: dict[str, Any]) -> dict[str, float]:
    state = latest_coarse_state(db, knowledge_base_id)
    if state is None:
        return {}
    concepts = db.scalars(select(CoarseConcept).where(CoarseConcept.coarse_state_id == state.id, CoarseConcept.state == "active")).all()
    envelope = agent_operating_envelope()
    scores = _text_entry_score({concept.id: concept_searchable_text(concept) for concept in concepts}, query_facets["terms"], top_n=envelope["agent_coarse_total_budget"])
    if scores or not concepts:
        return scores
    supported = sorted(
        concepts,
        key=lambda concept: (len(concept.included_mid_concept_ids_json or []), len(concept.bridge_mid_concept_ids_json or [])),
        reverse=True,
    )
    return {concept.id: 0.35 for concept in supported[: envelope["agent_coarse_total_budget"]]}


def select_mid_entries(db: Session, knowledge_base_id: str, query_facets: dict[str, Any], coarse_entries: dict[str, float]) -> dict[str, float]:
    state = latest_mid_state(db, knowledge_base_id)
    if state is None:
        return {}
    concepts = list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == state.id, MidConcept.state == "active")).all())
    envelope = agent_operating_envelope()
    scores = _text_entry_score({concept.id: concept_searchable_text(concept) for concept in concepts}, query_facets["terms"], top_n=envelope["agent_mid_top_k"])
    coarse_ids = set(coarse_entries)
    if coarse_ids:
        coarse_concepts = db.scalars(select(CoarseConcept).where(CoarseConcept.id.in_(coarse_ids))).all()
        boosted_mid_ids = {mid_id for coarse in coarse_concepts for mid_id in (coarse.included_mid_concept_ids_json or [])}
        for concept in concepts:
            if concept.id in boosted_mid_ids:
                scores[concept.id] = max(scores.get(concept.id, 0.0), 0.45)
    if not scores and concepts:
        supported = sorted(concepts, key=lambda concept: len(concept.support_rq_prefix_ids_json or []), reverse=True)
        scores = {concept.id: 0.35 for concept in supported[: envelope["agent_mid_top_k"]]}
    return scores


def select_rq_membership_entries(db: Session, knowledge_base_id: str, query_vector: list[float], query_rq: dict[str, Any] | None, mid_entries: dict[str, float]) -> dict[str, float]:
    relation_state = latest_relation_state(db, knowledge_base_id)
    if relation_state is None:
        return {}
    clusters = list(db.scalars(select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id, RQPrefix.state == "active")).all())
    scores: dict[str, float] = {}
    for cluster in clusters:
        centroid_score = max(0.0, cosine_similarity(query_vector, [float(value) for value in (cluster.centroid_json or [])]))
        scores[cluster.id] = centroid_score
        if query_rq and cluster.rq_path_prefix:
            query_path = [int(value) for value in (query_rq.get("rq_path") or [])]
            prefix = [int(value) for value in (cluster.rq_path_prefix or [])]
            levels = max(len(query_path), len(prefix), 1)
            lcp = lcp_depth(query_path, prefix)
            residual_mean = (cluster.diagnostics_json or {}).get("residual_mean_vector") or []
            residual_width = max(len(query_rq.get("residual_vector") or []), len(residual_mean))
            distance = _vector_norm(
                _vector_sub(
                    _fit_width(query_rq.get("residual_vector") or [], residual_width),
                    _fit_width(residual_mean, residual_width),
                )
            )
            residual_score = math.exp(-distance / max(float(query_rq.get("tau_r") or rq_tau()), 1e-6))
            rq_score = max(0.0, min(1.0, 0.70 * (lcp / levels) + 0.30 * residual_score))
            scores[cluster.id] = max(scores[cluster.id], rq_score)
    for concept_id, score in mid_entries.items():
        rows = db.scalars(select(MidConceptMembership).where(MidConceptMembership.mid_concept_id == concept_id)).all()
        for row in rows:
            scores[row.rq_prefix_id] = max(scores.get(row.rq_prefix_id, 0.0), score * row.membership_score)
    envelope = agent_operating_envelope()
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(envelope["agent_mid_top_k"], envelope["agent_chunk_per_mid_budget"])])


def dense_chunk_entries(db: Session, knowledge_base_id: str, query_vector: list[float]) -> dict[str, float]:
    records = list(db.scalars(select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base_id, VectorRecord.vector_status == "ready")).all())
    scored: list[tuple[str, float]] = []
    for record in records:
        vector = [float(value) for value in ((record.diagnostics_json or {}).get("embedding_vector") or []) if isinstance(value, (int, float))]
        if vector:
            scored.append((record.chunk_id, max(0.0, cosine_similarity(query_vector, vector))))
    return dict(sorted(scored, key=lambda item: item[1], reverse=True)[:80])


def _text_entry_score(text_by_id: dict[str, str], query_terms: list[str], *, top_n: int) -> dict[str, float]:
    if not text_by_id or not query_terms:
        return {}
    scored: list[tuple[str, float]] = []
    query_set = set(query_terms)
    for item_id, text in text_by_id.items():
        terms = set(tokenize_for_search_terms(text))
        overlap = len(query_set.intersection(terms))
        if overlap:
            scored.append((item_id, min(1.0, overlap / max(len(query_set), 1))))
    return dict(sorted(scored, key=lambda item: item[1], reverse=True)[:top_n])


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    width = max(1, int(size or 1))
    return [items[index : index + width] for index in range(0, len(items), width)]


def _language_map(raw: Any, fallback_text: str = "") -> dict[str, str]:
    payload = raw if isinstance(raw, dict) else {}
    return {
        "zh": str(payload.get("zh") or fallback_text or "").strip(),
        "en": str(payload.get("en") or fallback_text or "").strip(),
    }


def _language_list_map(raw: Any, fallback_items: list[str] | None = None) -> dict[str, list[str]]:
    payload = raw if isinstance(raw, dict) else {}

    def values_for(key: str) -> list[str]:
        values = payload.get(key)
        if not isinstance(values, list):
            values = fallback_items or []
        return [str(value).strip() for value in values if str(value).strip()]

    return {"zh": values_for("zh"), "en": values_for("en")}


def concept_i18n_payload(concept: Any) -> dict[str, Any]:
    payload = (getattr(concept, "llm_audit_json", None) or {}).get("concept_i18n")
    return payload if isinstance(payload, dict) else {}


def concept_searchable_text(concept: Any, *, include_i18n: bool | None = None) -> str:
    if include_i18n is None:
        include_i18n = bool(get_settings().concept_i18n_enabled)
    fields: list[str] = [
        str(getattr(concept, "canonical_label", "") or ""),
        str(getattr(concept, "definition", "") or ""),
        str(getattr(concept, "summary", "") or ""),
        str(getattr(concept, "scope_note", "") or ""),
        *[str(item) for item in (getattr(concept, "aliases_json", None) or [])],
        *[str(item) for item in (getattr(concept, "display_terms_json", None) or [])],
    ]
    i18n = concept_i18n_payload(concept)
    if include_i18n and i18n.get("status") == "ok":
        for key in ("label_i18n", "definition_i18n", "summary_i18n", "scope_note_i18n"):
            value = i18n.get(key)
            if isinstance(value, dict):
                fields.extend(str(item) for item in value.values())
        for key in ("aliases_i18n", "search_terms_i18n"):
            value = i18n.get(key)
            if isinstance(value, dict):
                for items in value.values():
                    if isinstance(items, list):
                        fields.extend(str(item) for item in items)
    return " ".join(field for field in fields if field)


def disabled_concept_i18n_stats(item_count: int, *, layer: str, target: str) -> dict[str, Any]:
    return {
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "enabled": False,
        "status": "disabled",
        "layer": layer,
        f"{target}_count": item_count,
        "translated_count": 0,
        "fallback_count": 0,
    }


def _concept_i18n_fallback(concept: Any, layer: str) -> dict[str, Any]:
    label = str(getattr(concept, "canonical_label", "") or "")
    definition = str(getattr(concept, "definition", "") or "")
    summary = str(getattr(concept, "summary", "") or definition)
    scope_note = str(getattr(concept, "scope_note", "") or "")
    aliases = [str(item) for item in (getattr(concept, "aliases_json", None) or [])]
    return {
        "id": getattr(concept, "id", None),
        "layer": layer,
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "status": "original_text_fallback",
        "label_i18n": {"zh": label, "en": label},
        "aliases_i18n": {"zh": aliases, "en": aliases},
        "definition_i18n": {"zh": definition, "en": definition},
        "summary_i18n": {"zh": summary, "en": summary},
        "scope_note_i18n": {"zh": scope_note, "en": scope_note},
        "search_terms_i18n": {"zh": [], "en": []},
    }


def _edge_i18n_fallback(edge: Any, source_label: str, target_label: str, layer: str) -> dict[str, Any]:
    explanation = str(getattr(edge, "explanation", "") or "")
    relation = f"{source_label} -> {target_label}"
    return {
        "id": getattr(edge, "id", None),
        "layer": layer,
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "status": "original_text_fallback",
        "relation_label_i18n": {"zh": relation, "en": relation},
        "explanation_i18n": {"zh": explanation, "en": explanation},
        "summary_i18n": {"zh": explanation, "en": explanation},
        "search_terms_i18n": {"zh": [], "en": []},
    }


def _normalize_concept_i18n_item(raw: dict[str, Any], concept: Any, layer: str, *, fallback_status: str | None = None) -> dict[str, Any]:
    fallback = _concept_i18n_fallback(concept, layer)
    status = str(raw.get("status") or fallback_status or "ok")
    return {
        "id": getattr(concept, "id", None),
        "layer": layer,
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "status": status,
        "label_i18n": _language_map(raw.get("label_i18n"), str(getattr(concept, "canonical_label", "") or "")),
        "aliases_i18n": _language_list_map(raw.get("aliases_i18n"), list(getattr(concept, "aliases_json", None) or [])),
        "definition_i18n": _language_map(raw.get("definition_i18n"), str(getattr(concept, "definition", "") or "")),
        "summary_i18n": _language_map(raw.get("summary_i18n"), str(getattr(concept, "summary", "") or getattr(concept, "definition", "") or "")),
        "scope_note_i18n": _language_map(raw.get("scope_note_i18n"), str(getattr(concept, "scope_note", "") or "")),
        "search_terms_i18n": _language_list_map(raw.get("search_terms_i18n"), []),
        "fallback_source": fallback["status"] if status != "ok" else None,
    }


def _normalize_edge_i18n_item(raw: dict[str, Any], edge: Any, source_label: str, target_label: str, layer: str, *, fallback_status: str | None = None) -> dict[str, Any]:
    fallback = _edge_i18n_fallback(edge, source_label, target_label, layer)
    status = str(raw.get("status") or fallback_status or "ok")
    return {
        "id": getattr(edge, "id", None),
        "layer": layer,
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "status": status,
        "relation_label_i18n": _language_map(raw.get("relation_label_i18n"), f"{source_label} -> {target_label}"),
        "explanation_i18n": _language_map(raw.get("explanation_i18n"), str(getattr(edge, "explanation", "") or "")),
        "summary_i18n": _language_map(raw.get("summary_i18n"), str(getattr(edge, "explanation", "") or "")),
        "search_terms_i18n": _language_list_map(raw.get("search_terms_i18n"), []),
        "fallback_source": fallback["status"] if status != "ok" else None,
    }


async def enrich_concepts_i18n(concepts: list[Any], *, layer: str, batch_size: int = 12) -> dict[str, Any]:
    if not concepts:
        return {"protocol_version": CONCEPT_I18N_PROTOCOL_VERSION, "enabled": True, "concept_count": 0, "translated_count": 0}
    settings = get_settings()
    batches = _chunked(concepts, batch_size)

    async def translate_batch(batch: list[Any]) -> list[dict[str, Any]]:
        items = [
            {
                "id": concept.id,
                "label": concept.canonical_label,
                "aliases": concept.aliases_json or [],
                "definition": concept.definition,
                "summary": concept.summary,
                "scope_note": concept.scope_note,
            }
            for concept in batch
        ]
        fallback = {"items": [_concept_i18n_fallback(concept, layer) for concept in batch]}
        system = (
            "You translate derived concept metadata for a grounded Four-Layer Context Graph RAG system. "
            "Return strict JSON with an items array. For every input item, preserve id and provide: "
            "label_i18n {zh,en}, aliases_i18n {zh,en arrays}, definition_i18n {zh,en}, summary_i18n {zh,en}, "
            "scope_note_i18n {zh,en}, search_terms_i18n {zh,en arrays}. "
            "Translate technical terms accurately, keep formulas/symbols unchanged, and do not add facts beyond the source text."
        )
        output = await ChatProvider(purpose="graph").classify_json(system_prompt=system, user_prompt=str({"layer": layer, "items": items}), fallback=fallback)
        output_items = output.get("items") if isinstance(output, dict) else None
        if not isinstance(output_items, list):
            if settings.enable_model_fallback:
                output_items = fallback["items"]
            else:
                raise RuntimeError(f"{layer} concept i18n provider returned invalid JSON; missing items array")
        by_id = {str(item.get("id")): item for item in output_items if isinstance(item, dict) and item.get("id")}
        normalized: list[dict[str, Any]] = []
        for concept in batch:
            raw = by_id.get(str(concept.id))
            if raw is None:
                if not settings.enable_model_fallback:
                    raise RuntimeError(f"{layer} concept i18n provider omitted concept {concept.id}")
                raw = _concept_i18n_fallback(concept, layer)
            normalized.append(_normalize_concept_i18n_item(raw, concept, layer))
        return normalized

    translated_batches = await gather_bounded(batches, settings.model_request_concurrency, translate_batch)
    payloads = [payload for batch in translated_batches for payload in batch]
    payload_by_id = {payload["id"]: payload for payload in payloads}
    for concept in concepts:
        concept.llm_audit_json = {
            **(concept.llm_audit_json or {}),
            "concept_i18n": payload_by_id.get(concept.id, _concept_i18n_fallback(concept, layer)),
        }
    translated_count = sum(1 for payload in payloads if payload.get("status") == "ok")
    return {
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "enabled": True,
        "concept_count": len(concepts),
        "translated_count": translated_count,
        "fallback_count": len(concepts) - translated_count,
        "hash": stable_hash(payloads),
    }


async def enrich_concept_edges_i18n(edges: list[Any], concepts_by_id: dict[str, Any], *, layer: str, batch_size: int = 16) -> dict[str, Any]:
    if not edges:
        return {"protocol_version": CONCEPT_I18N_PROTOCOL_VERSION, "enabled": True, "edge_count": 0, "translated_count": 0}
    settings = get_settings()
    batches = _chunked(edges, batch_size)

    def edge_labels(edge: Any) -> tuple[str, str]:
        source = concepts_by_id.get(getattr(edge, "source_concept_id", ""))
        target = concepts_by_id.get(getattr(edge, "target_concept_id", ""))
        return str(getattr(source, "canonical_label", "") or ""), str(getattr(target, "canonical_label", "") or "")

    async def translate_batch(batch: list[Any]) -> list[dict[str, Any]]:
        items = []
        fallbacks = []
        for edge in batch:
            source_label, target_label = edge_labels(edge)
            items.append(
                {
                    "id": edge.id,
                    "source_label": source_label,
                    "target_label": target_label,
                    "edge_type": edge.edge_type,
                    "explanation": edge.explanation,
                }
            )
            fallbacks.append(_edge_i18n_fallback(edge, source_label, target_label, layer))
        fallback = {"items": fallbacks}
        system = (
            "You translate derived concept-edge metadata for a grounded Four-Layer Context Graph RAG system. "
            "Return strict JSON with an items array. For every input item, preserve id and provide: "
            "relation_label_i18n {zh,en}, explanation_i18n {zh,en}, summary_i18n {zh,en}, search_terms_i18n {zh,en arrays}. "
            "Translate only the relationship wording; keep evidence meaning, formulas, and technical symbols unchanged."
        )
        output = await ChatProvider(purpose="graph").classify_json(system_prompt=system, user_prompt=str({"layer": layer, "items": items}), fallback=fallback)
        output_items = output.get("items") if isinstance(output, dict) else None
        if not isinstance(output_items, list):
            if settings.enable_model_fallback:
                output_items = fallback["items"]
            else:
                raise RuntimeError(f"{layer} edge i18n provider returned invalid JSON; missing items array")
        by_id = {str(item.get("id")): item for item in output_items if isinstance(item, dict) and item.get("id")}
        normalized: list[dict[str, Any]] = []
        for edge in batch:
            source_label, target_label = edge_labels(edge)
            raw = by_id.get(str(edge.id))
            if raw is None:
                if not settings.enable_model_fallback:
                    raise RuntimeError(f"{layer} edge i18n provider omitted edge {edge.id}")
                raw = _edge_i18n_fallback(edge, source_label, target_label, layer)
            normalized.append(_normalize_edge_i18n_item(raw, edge, source_label, target_label, layer))
        return normalized

    translated_batches = await gather_bounded(batches, settings.model_request_concurrency, translate_batch)
    payloads = [payload for batch in translated_batches for payload in batch]
    payload_by_id = {payload["id"]: payload for payload in payloads}
    for edge in edges:
        source_label, target_label = edge_labels(edge)
        edge.diagnostics_json = {
            **(edge.diagnostics_json or {}),
            "edge_i18n": payload_by_id.get(edge.id, _edge_i18n_fallback(edge, source_label, target_label, layer)),
        }
    translated_count = sum(1 for payload in payloads if payload.get("status") == "ok")
    return {
        "protocol_version": CONCEPT_I18N_PROTOCOL_VERSION,
        "enabled": True,
        "edge_count": len(edges),
        "translated_count": translated_count,
        "fallback_count": len(edges) - translated_count,
        "hash": stable_hash(payloads),
    }


def _edge_raw_strength(edge: Any) -> float:
    value = getattr(edge, "raw_strength", None)
    if value is not None:
        return normalized_strength(float(value))
    summary = getattr(edge, "raw_strength_summary_json", None) or {}
    if summary.get("max_raw_strength") is not None:
        return normalized_strength(float(summary.get("max_raw_strength")))
    return normalized_strength(float(getattr(edge, "weight", 0.1) or 0.1))


def _edge_distance(edge: Any) -> float:
    value = getattr(edge, "distance", None)
    if value is not None:
        return float(value)
    return distance_from_strength(_edge_raw_strength(edge))


def _edge_support_refs(edge: Any) -> dict[str, Any]:
    refs: dict[str, Any] = {"edge_id": getattr(edge, "id", None), "edge_type": getattr(edge, "edge_type", None)}
    for attr in (
        "support_chunk_ids_json",
        "support_chunk_edge_ids_json",
        "support_relation_edge_ids_json",
        "support_rq_prefix_ids_json",
        "support_rq_prefix_node_ids_json",
        "support_mid_edge_ids_json",
        "support_mid_concept_ids_json",
    ):
        value = getattr(edge, attr, None)
        if value:
            refs[attr.removesuffix("_json")] = value
    return refs


def _build_adjacency(edges: list[Any], source_attr: str, target_attr: str) -> dict[str, list[Any]]:
    adjacency: dict[str, list[Any]] = defaultdict(list)
    for edge in edges:
        source_id = getattr(edge, source_attr, None)
        target_id = getattr(edge, target_attr, None)
        if not source_id or not target_id:
            continue
        adjacency[source_id].append(edge)
        adjacency[target_id].append(edge)
    return adjacency


def _edge_neighbor(edge: Any, node_id: str, source_attr: str, target_attr: str) -> str:
    source_id = getattr(edge, source_attr)
    target_id = getattr(edge, target_attr)
    return target_id if source_id == node_id else source_id


def _path_distance_zone(distance: float, envelope: dict[str, Any]) -> str:
    green = float(envelope.get("path_distance_green_threshold") or 0.0)
    gray = float(envelope.get("path_distance_gray_threshold") or green)
    hard = float(envelope.get("path_distance_hard_threshold") or gray)
    if distance <= green:
        return "green"
    if distance <= gray:
        return "gray"
    if hard and distance <= hard:
        return "red"
    return "hard_stop"


def layer_frontier_budget(layer: str, envelope: dict[str, Any]) -> int:
    if layer == "coarse":
        return int(envelope.get("agent_coarse_total_budget") or 1)
    if layer == "mid":
        return max(1, int(envelope.get("agent_mid_per_coarse_budget") or 1) * max(1, int(envelope.get("agent_mid_top_k") or 1)))
    if layer == "chunk":
        return max(1, int(envelope.get("agent_chunk_per_mid_budget") or 1) * max(1, int(envelope.get("agent_chunk_top_k") or 1)))
    return 1


def _cycle_reward_for_next_state(
    *,
    state: dict[str, Any],
    neighbor_id: str,
    edge_id: str,
    edge_distance: float,
    edge_strength: float,
    envelope: dict[str, Any],
) -> tuple[float, dict[str, Any] | None]:
    previous_path = list(state.get("path") or [])
    if neighbor_id not in previous_path:
        return 0.0, None
    cap_remaining = max(0.0, float(envelope.get("max_cycle_reward_per_path") or 0.0) - float(state.get("reward_so_far") or 0.0))
    if cap_remaining <= 0:
        return 0.0, {
            "cycle_edges": [edge_id],
            "cycle_distance": edge_distance,
            "reward_before_cap": 0.0,
            "reward_after_cap": 0.0,
            "cap_reason": "max_cycle_reward_per_path_exhausted",
        }
    previous_index = len(previous_path) - 1 - list(reversed(previous_path)).index(neighbor_id)
    prior_edge_distances = [float(value) for value in (state.get("path_edge_distances") or [])]
    prior_edge_ids = list(state.get("path_edge_ids") or [])
    cycle_distance = sum(prior_edge_distances[previous_index:]) + edge_distance
    threshold = float(envelope.get("cycle_reward_distance_threshold") or 0.0)
    cycle_edges = prior_edge_ids[previous_index:] + [edge_id]
    if threshold <= 0 or cycle_distance > threshold:
        return 0.0, {
            "cycle_edges": cycle_edges,
            "cycle_distance": round(cycle_distance, 6),
            "support_delta": 0,
            "reward_before_cap": 0.0,
            "reward_after_cap": 0.0,
            "cap_reason": "cycle_distance_above_threshold",
        }
    reward_before_cap = 0.04 * normalized_strength(edge_strength) * math.exp(-cycle_distance / max(threshold, 1e-6))
    reward_after_cap = min(cap_remaining, reward_before_cap)
    return reward_after_cap, {
        "cycle_edges": cycle_edges,
        "cycle_distance": round(cycle_distance, 6),
        "support_delta": 1,
        "reward_before_cap": round(reward_before_cap, 6),
        "reward_after_cap": round(reward_after_cap, 6),
        "cap_reason": "within_cap" if reward_after_cap == reward_before_cap else "max_cycle_reward_per_path",
    }


def execute_layer_priority_walk(
    *,
    layer: str,
    entry_scores: dict[str, float],
    node_text_by_id: dict[str, str],
    adjacency: dict[str, list[Any]],
    query_facets: dict[str, Any],
    source_attr: str,
    target_attr: str,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    required_facets = set(query_facets.get("required_facets") or [])

    def covered_facets_for_node(node_id: str) -> set[str]:
        return set(matched_required_facets_for_text(node_text_by_id.get(node_id, ""), query_facets))

    entry_nodes = [
        {
            "layer": layer,
            "node_id": node_id,
            "entry_strength": normalized_strength(score),
            "roles": [f"{layer}_entry"],
            "metadata": {},
        }
        for node_id, score in sorted(entry_scores.items(), key=lambda item: item[1], reverse=True)
        if node_id in node_text_by_id
    ]
    frontier: list[tuple[tuple[float, float, int, int], int, dict[str, Any]]] = []
    serial = 0
    for entry in entry_nodes:
        node_id = entry["node_id"]
        covered = covered_facets_for_node(node_id)
        distance = distance_from_strength(float(entry["entry_strength"]))
        state = {
            "layer": layer,
            "node_id": node_id,
            "path": [node_id],
            "path_edge_ids": [],
            "path_edge_distances": [],
            "distance_so_far": distance,
            "reward_so_far": 0.0,
            "covered_facets": sorted(covered),
            "evidence_roles": list(entry["roles"]),
            "depth": 0,
            "visit_counts": {node_id: 1},
            "support_refs": {"entry_strength": entry["entry_strength"]},
        }
        key = (len(required_facets - covered), distance, 0, -len(state["evidence_roles"]))
        heapq.heappush(frontier, (key, serial, state))
        serial += 1

    frontier_pops: list[dict[str, Any]] = []
    frontier_snapshots: list[dict[str, Any]] = []
    path_labels: list[dict[str, Any]] = []
    gray_zone_decisions: list[dict[str, Any]] = []
    accepted_by_node: dict[str, dict[str, Any]] = {}
    dominance_labels: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    dominance_pruned_count = 0
    hard_stop_pruned_count = 0
    red_zone_pruned_count = 0
    expansion_count = 0
    max_expansions = layer_frontier_budget(layer, envelope)
    max_depth = int(envelope.get("max_depth_per_layer") or 1)
    max_labels = int(envelope.get("max_labels_per_node") or 1)
    stop_reason = "no_entry_nodes" if not frontier else "frontier_empty"

    while frontier and expansion_count < max_expansions:
        key, _serial, state = heapq.heappop(frontier)
        label_key = (key[0], round(key[1], 6), key[2], key[3])
        labels = dominance_labels[state["node_id"]]
        if len(labels) >= max_labels and any(existing <= label_key for existing in labels):
            dominance_pruned_count += 1
            continue
        labels.append(label_key)
        labels.sort()
        del labels[max_labels:]
        frontier_pops.append(state)
        frontier_snapshots.append({"layer": layer, "popped": state, "queue_size_after_pop": len(frontier), "key": list(key)})
        current = accepted_by_node.get(state["node_id"])
        if current is None or key < current["queue_key"]:
            accepted_by_node[state["node_id"]] = {"state": state, "queue_key": key}
        if int(state["depth"]) >= max_depth:
            path_labels.append(
                {
                    "layer": layer,
                    "node_id": state["node_id"],
                    "path": state["path"],
                    "path_edge_ids": state["path_edge_ids"],
                    "covered_facets": state["covered_facets"],
                    "distance_so_far": state["distance_so_far"],
                    "reward_so_far": state["reward_so_far"],
                    "expanded_edge_ids": [],
                }
            )
            continue
        expanded_edge_ids: list[str] = []
        for edge in sorted(adjacency.get(state["node_id"], []), key=_edge_distance):
            neighbor_id = _edge_neighbor(edge, state["node_id"], source_attr, target_attr)
            if neighbor_id not in node_text_by_id:
                continue
            edge_distance = _edge_distance(edge)
            next_distance = round(float(state["distance_so_far"]) + edge_distance, 6)
            zone = _path_distance_zone(next_distance, envelope)
            if zone in {"red", "hard_stop"}:
                hard_stop_pruned_count += 1
                if zone == "red":
                    red_zone_pruned_count += 1
                gray_zone_decisions.append(
                    {
                        "layer": layer,
                        "edge_id": edge.id,
                        "from_node_id": state["node_id"],
                        "to_node_id": neighbor_id,
                        "path_distance": next_distance,
                        "distance_zone": zone,
                        "decision": "red_zone_pruned" if zone == "red" else "hard_stop_pruned",
                    }
                )
                continue
            if zone == "gray":
                gray_zone_decisions.append(
                    {
                        "layer": layer,
                        "edge_id": edge.id,
                        "from_node_id": state["node_id"],
                        "to_node_id": neighbor_id,
                        "path_distance": next_distance,
                        "distance_zone": zone,
                        "covered_facets": state.get("covered_facets") or [],
                        "support_refs": _edge_support_refs(edge),
                        "decision": "continue_path",
                    }
                )
            visit_counts = dict(state["visit_counts"])
            previous_visits = int(visit_counts.get(neighbor_id, 0))
            visit_counts[neighbor_id] = previous_visits + 1
            reward_increment, cycle_diagnostics = _cycle_reward_for_next_state(
                state=state,
                neighbor_id=neighbor_id,
                edge_id=edge.id,
                edge_distance=edge_distance,
                edge_strength=_edge_raw_strength(edge),
                envelope=envelope,
            )
            reward_so_far = round(max(0.0, float(state["reward_so_far"]) + max(0.0, reward_increment)), 6)
            covered = set(state["covered_facets"]) | covered_facets_for_node(neighbor_id)
            roles = set(state["evidence_roles"])
            roles.add(str(getattr(edge, "edge_type", "edge")))
            next_state = {
                "layer": layer,
                "node_id": neighbor_id,
                "path": list(state["path"]) + [neighbor_id],
                "path_edge_ids": list(state["path_edge_ids"]) + [edge.id],
                "path_edge_distances": list(state.get("path_edge_distances") or []) + [edge_distance],
                "distance_so_far": next_distance,
                "reward_so_far": reward_so_far,
                "cycle_distance_rewards": list(state.get("cycle_distance_rewards") or []) + ([cycle_diagnostics] if cycle_diagnostics else []),
                "distance_zone": zone,
                "covered_facets": sorted(covered),
                "evidence_roles": sorted(roles),
                "depth": int(state["depth"]) + 1,
                "visit_counts": visit_counts,
                "support_refs": _edge_support_refs(edge),
            }
            next_key = (
                len(required_facets - covered),
                round(float(next_state["distance_so_far"]) - reward_so_far, 6),
                int(next_state["depth"]),
                -len(roles),
            )
            heapq.heappush(frontier, (next_key, serial, next_state))
            serial += 1
            expanded_edge_ids.append(edge.id)
            expansion_count += 1
            if expansion_count >= max_expansions:
                stop_reason = "hard_budget_hit"
                break
        path_labels.append(
            {
                "layer": layer,
                "node_id": state["node_id"],
                "path": state["path"],
                "path_edge_ids": state["path_edge_ids"],
                "covered_facets": state["covered_facets"],
                "distance_so_far": state["distance_so_far"],
                "reward_so_far": state["reward_so_far"],
                "expanded_edge_ids": expanded_edge_ids,
            }
        )

    accepted = sorted(accepted_by_node.values(), key=lambda item: item["queue_key"])
    return {
        "entry_nodes": entry_nodes,
        "accepted_nodes": [item["state"]["node_id"] for item in accepted],
        "accepted_states": [item["state"] for item in accepted],
        "frontier_pops": frontier_pops,
        "frontier_json": frontier_snapshots,
        "path_labels": path_labels,
        "gray_zone_path_decisions": gray_zone_decisions,
        "convergence": {
            "reason": stop_reason,
            "frontier_expansion_count": expansion_count,
            "dominance_pruned_count": dominance_pruned_count,
            "hard_stop_pruned_count": hard_stop_pruned_count,
            "red_zone_pruned_count": red_zone_pruned_count,
            "cycle_distance_reward_bounded": True,
            "path_distance_thresholds": {
                "green": envelope.get("path_distance_green_threshold"),
                "gray": envelope.get("path_distance_gray_threshold"),
                "hard": envelope.get("path_distance_hard_threshold"),
            },
            "frontier_remaining": len(frontier),
            "accepted_node_count": len(accepted),
        },
    }


def execute_priority_queue_traversal(
    db: Session,
    *,
    knowledge_base_id: str,
    chunks: list[Chunk],
    filters: SearchFilters,
    query_facets: dict[str, Any],
    coarse_entries: dict[str, float],
    mid_entries: dict[str, float],
    rq_membership_entries: dict[str, float],
    dense_entries: dict[str, float],
    query_rq: dict[str, Any] | None,
    top_k: int,
    retrieval_granularity: RetrievalGranularity = "mid",
) -> dict[str, Any]:
    envelope = agent_operating_envelope()
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    layer_walks: dict[str, dict[str, Any]] = {}

    coarse_concepts: list[CoarseConcept] = []
    if retrieval_granularity == "coarse":
        coarse_state = latest_coarse_state(db, knowledge_base_id)
        coarse_concepts = (
            list(db.scalars(select(CoarseConcept).where(CoarseConcept.coarse_state_id == coarse_state.id, CoarseConcept.state == "active")).all())
            if coarse_state
            else []
        )
        coarse_text = {concept.id: concept_searchable_text(concept) for concept in coarse_concepts}
        coarse_edges = list(db.scalars(select(CoarseConceptEdge).where(CoarseConceptEdge.coarse_state_id == coarse_state.id)).all()) if coarse_state else []
        layer_walks["coarse"] = execute_layer_priority_walk(
            layer="coarse",
            entry_scores=coarse_entries,
            node_text_by_id=coarse_text,
            adjacency=_build_adjacency(coarse_edges, "source_concept_id", "target_concept_id"),
            query_facets=query_facets,
            source_attr="source_concept_id",
            target_attr="target_concept_id",
            envelope=envelope,
        )
        stage_queues: dict[str, Any] = {
            "coarse": {
                "entry_ids": list(coarse_entries.keys()),
                "accepted_ids": list(layer_walks["coarse"].get("accepted_nodes") or []),
                "frontier_pop_count": len(layer_walks["coarse"].get("frontier_pops") or []),
            }
        }
        coarse_skipped_reason = None
    else:
        coarse_skipped_reason = "skipped_by_granularity=mid"
        layer_walks["coarse"] = {
            "entry_nodes": [],
            "accepted_nodes": [],
            "accepted_states": [],
            "frontier_pops": [],
            "frontier_json": [],
            "path_labels": [],
            "gray_zone_path_decisions": [],
            "convergence": {
                "reason": coarse_skipped_reason,
                "skipped_by_granularity": "mid",
                "active_traversal_layer": False,
            },
        }
        stage_queues = {
            "coarse": {
                "entry_ids": [],
                "accepted_ids": [],
                "frontier_pop_count": 0,
                "skipped_by_granularity": "mid",
                "reason": coarse_skipped_reason,
            }
        }
    candidate_pools: dict[str, Any] = {"mid_by_coarse": [], "chunk_by_mid": []}
    topk_selection: dict[str, Any] = {}

    mid_state = latest_mid_state(db, knowledge_base_id)
    mid_concepts = (
        list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == mid_state.id, MidConcept.state == "active")).all())
        if mid_state
        else []
    )
    mid_by_id = {concept.id: concept for concept in mid_concepts}
    coarse_by_id = {concept.id: concept for concept in coarse_concepts}
    mid_text = {concept.id: concept_searchable_text(concept) for concept in mid_concepts}
    mid_candidate_scores: dict[str, float] = {}
    if retrieval_granularity == "coarse":
        accepted_coarse_ids = list(layer_walks["coarse"].get("accepted_nodes") or []) or list(coarse_entries.keys())[: int(envelope["agent_coarse_total_budget"])]
        for coarse_id in accepted_coarse_ids:
            coarse = coarse_by_id.get(coarse_id)
            if coarse is None:
                continue
            coarse_strength = max(coarse_entries.get(coarse_id, 0.35), 0.35)
            candidates: list[dict[str, Any]] = []
            for mid_id in coarse.included_mid_concept_ids_json or []:
                if mid_id not in mid_by_id:
                    continue
                score = normalized_strength(max(mid_entries.get(mid_id, 0.0), coarse_strength * 0.9))
                candidates.append({"id": mid_id, "score": score, "parent_layer": "coarse", "parent_node_id": coarse_id})
            candidates.sort(key=lambda item: item["score"], reverse=True)
            selected = candidates[: int(envelope["agent_mid_per_coarse_budget"])]
            for item in selected:
                mid_candidate_scores[item["id"]] = max(mid_candidate_scores.get(item["id"], 0.0), float(item["score"]))
            candidate_pools["mid_by_coarse"].append(
                {
                    "parent_layer": "coarse",
                    "parent_node_id": coarse_id,
                    "candidate_ids": [item["id"] for item in candidates],
                    "selected_ids": [item["id"] for item in selected],
                    "per_parent_budget_status": {
                        "budget": int(envelope["agent_mid_per_coarse_budget"]),
                        "candidate_count": len(candidates),
                        "selected_count": len(selected),
                        "stop_reason": "per_parent_budget_hit" if len(candidates) > len(selected) else "parent_candidates_exhausted",
                    },
                }
            )
    else:
        mid_candidate_scores = dict(mid_entries)
        candidate_pools["mid_direct_entries"] = {
            "candidate_ids": list(mid_entries.keys()),
            "selected_ids": list(mid_entries.keys())[: int(envelope["agent_mid_top_k"])],
            "top_k": int(envelope["agent_mid_top_k"]),
            "coarse_skipped_reason": coarse_skipped_reason,
        }
    if not mid_candidate_scores:
        mid_candidate_scores = dict(mid_entries)
    mid_entries = dict(sorted(mid_candidate_scores.items(), key=lambda item: item[1], reverse=True)[: int(envelope["agent_mid_top_k"])])
    topk_selection["mid"] = {
        "top_k": int(envelope["agent_mid_top_k"]),
        "candidate_count": len(mid_candidate_scores),
        "selected_ids": list(mid_entries.keys()),
        "stop_reason": "layer_top_k_cut" if len(mid_candidate_scores) > len(mid_entries) else "candidate_pool_exhausted",
        "entry_mode": retrieval_granularity,
    }
    stage_queues["mid"] = {
        "selected_ids": list(mid_entries.keys()),
        "top_k": int(envelope["agent_mid_top_k"]),
        "entry_mode": "direct_mid" if retrieval_granularity == "mid" else "coarse_drilldown",
    }
    mid_edges = list(db.scalars(select(MidConceptEdge).where(MidConceptEdge.concept_state_id == mid_state.id)).all()) if mid_state else []
    layer_walks["mid"] = execute_layer_priority_walk(
        layer="mid",
        entry_scores=mid_entries,
        node_text_by_id=mid_text,
        adjacency=_build_adjacency(mid_edges, "source_concept_id", "target_concept_id"),
        query_facets=query_facets,
        source_attr="source_concept_id",
        target_attr="target_concept_id",
        envelope=envelope,
    )
    selected_mid_ids = list(layer_walks["mid"].get("accepted_nodes") or []) or list(mid_entries.keys())
    stage_queues["mid"]["accepted_ids"] = selected_mid_ids
    rq_membership_entries = {}
    for mid_id in selected_mid_ids:
        rows = db.scalars(select(MidConceptMembership).where(MidConceptMembership.mid_concept_id == mid_id)).all()
        mid_strength = max(mid_entries.get(mid_id, 0.35), 0.35)
        for row in rows:
            rq_membership_entries[row.rq_prefix_id] = max(rq_membership_entries.get(row.rq_prefix_id, 0.0), mid_strength * float(row.membership_score or 1.0))

    relation_state = latest_relation_state(db, knowledge_base_id)
    rq_prefixes = (
        list(db.scalars(select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id, RQPrefix.state == "active")).all())
        if relation_state
        else []
    )
    rq_prefix_by_id = {cluster.id: cluster for cluster in rq_prefixes}
    rq_membership_entry_nodes = [
        {
            "layer": "rq_membership",
            "node_id": rq_prefix_id,
            "rq_prefix_id": rq_prefix_id,
            "entry_strength": normalized_strength(strength),
            "roles": ["semantic_address_seed"],
            "metadata": {
                "label": rq_prefix_by_id[rq_prefix_id].label,
                "node_type": rq_prefix_by_id[rq_prefix_id].node_type,
                "rq_path_prefix": rq_prefix_by_id[rq_prefix_id].rq_path_prefix,
                "representative_terms": (rq_prefix_by_id[rq_prefix_id].diagnostics_json or {}).get("representative_terms") or [],
            },
        }
        for rq_prefix_id, strength in sorted(rq_membership_entries.items(), key=lambda item: item[1], reverse=True)
        if rq_prefix_id in rq_prefix_by_id
    ]
    layer_walks["rq_membership"] = {
        "entry_nodes": rq_membership_entry_nodes,
        "accepted_nodes": [item["node_id"] for item in rq_membership_entry_nodes],
        "accepted_states": [],
        "frontier_pops": [],
        "frontier_json": [],
        "path_labels": [],
        "gray_zone_path_decisions": [],
        "convergence": {
            "reason": "rq_membership_seed_selection",
            "seed_count": len(rq_membership_entry_nodes),
            "active_traversal_layer": False,
        },
    }
    stage_queues["rq_membership"] = {"selected_ids": [item["node_id"] for item in rq_membership_entry_nodes]}

    seed_strengths: dict[str, float] = defaultdict(float)
    seed_roles: dict[str, set[str]] = defaultdict(set)
    seed_metadata: dict[str, dict[str, Any]] = defaultdict(dict)

    def add_seed(chunk_id: str, strength: float, role: str, metadata: dict[str, Any] | None = None) -> None:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None or not passes_filters(db, chunk, filters):
            return
        raw_strength = normalized_strength(strength)
        entry_strength = calibrated_entry_seed_strength(raw_strength, role)
        seed_strengths[chunk_id] = max(seed_strengths[chunk_id], entry_strength)
        seed_roles[chunk_id].add(role)
        current = seed_metadata[chunk_id]
        entry_strengths = current.setdefault("entry_strengths", {})
        raw_entry_strengths = current.setdefault("raw_entry_strengths", {})
        entry_strengths[role] = max(float(entry_strengths.get(role, 0.0) or 0.0), entry_strength)
        raw_entry_strengths[role] = max(float(raw_entry_strengths.get(role, 0.0) or 0.0), raw_strength)
        current["entry_strength"] = max(float(current.get("entry_strength", 0.0) or 0.0), entry_strength)
        current["entry_distance"] = distance_from_strength(float(current["entry_strength"]))
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, list):
                    current.setdefault(key, [])
                    current[key].extend(item for item in value if item not in current[key])
                else:
                    current[key] = value

    chunk_candidate_scores: dict[str, float] = {}
    chunk_candidate_metadata: dict[str, dict[str, Any]] = defaultdict(dict)
    chunk_candidate_roles: dict[str, set[str]] = defaultdict(set)
    rq_rows_by_prefix: dict[str, list[RQPrefixMembership]] = defaultdict(list)
    if rq_membership_entries:
        for row in db.scalars(select(RQPrefixMembership).where(RQPrefixMembership.rq_prefix_id.in_(list(rq_membership_entries.keys())))).all():
            rq_rows_by_prefix[row.rq_prefix_id].append(row)

    def add_chunk_candidate(chunk_id: str, score: float, role: str, metadata: dict[str, Any] | None = None) -> None:
        if chunk_id not in chunk_by_id or not passes_filters(db, chunk_by_id[chunk_id], filters):
            return
        normalized = normalized_strength(score)
        chunk_candidate_scores[chunk_id] = max(chunk_candidate_scores.get(chunk_id, 0.0), normalized)
        chunk_candidate_roles[chunk_id].add(role)
        current = chunk_candidate_metadata[chunk_id]
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, list):
                    current.setdefault(key, [])
                    current[key].extend(item for item in value if item not in current[key])
                else:
                    current[key] = value

    for mid_id in selected_mid_ids:
        concept = mid_by_id.get(mid_id)
        if concept is None:
            continue
        mid_strength = max(mid_entries.get(mid_id, 0.35), 0.35)
        candidates: list[dict[str, Any]] = []
        support_ids = list(dict.fromkeys(concept.support_chunk_ids_json or []))
        for chunk_id in support_ids:
            score = normalized_strength(max(mid_strength * 0.85, dense_entries.get(chunk_id, 0.0) * 0.97))
            candidates.append({"id": chunk_id, "score": score, "source": "mid_support"})
        for prefix_id in concept.support_rq_prefix_ids_json or []:
            prefix_strength = rq_membership_entries.get(prefix_id, mid_strength)
            for row in rq_rows_by_prefix.get(prefix_id, []):
                score = normalized_strength(max(prefix_strength * float(row.membership_score or 1.0), dense_entries.get(row.chunk_id, 0.0) * 0.97))
                candidates.append({"id": row.chunk_id, "score": score, "source": "rq_membership", "rq_prefix_id": prefix_id})
        deduped: dict[str, dict[str, Any]] = {}
        for item in candidates:
            current = deduped.get(item["id"])
            if current is None or float(item["score"]) > float(current["score"]):
                deduped[item["id"]] = item
        ranked = sorted(deduped.values(), key=lambda item: item["score"], reverse=True)
        selected = ranked[: int(envelope["agent_chunk_per_mid_budget"])]
        candidate_pools["chunk_by_mid"].append(
            {
                "parent_layer": "mid",
                "parent_node_id": mid_id,
                "candidate_ids": [item["id"] for item in ranked],
                "selected_ids": [item["id"] for item in selected],
                "per_parent_budget_status": {
                    "budget": int(envelope["agent_chunk_per_mid_budget"]),
                    "candidate_count": len(ranked),
                    "selected_count": len(selected),
                    "stop_reason": "per_parent_budget_hit" if len(ranked) > len(selected) else "parent_candidates_exhausted",
                },
            }
        )
        for item in selected:
            metadata = {"mid_concept_ids": [mid_id], "chunk_candidate_source": item.get("source")}
            if item.get("rq_prefix_id"):
                metadata["rq_prefix_ids"] = [item["rq_prefix_id"]]
            add_chunk_candidate(item["id"], float(item["score"]), "mid_drilldown_entry", metadata)

    selected_chunk_candidates = sorted(chunk_candidate_scores.items(), key=lambda item: item[1], reverse=True)[: int(envelope["agent_chunk_top_k"])]
    topk_selection["chunk"] = {
        "top_k": int(envelope["agent_chunk_top_k"]),
        "candidate_count": len(chunk_candidate_scores),
        "selected_ids": [chunk_id for chunk_id, _score in selected_chunk_candidates],
        "stop_reason": "layer_top_k_cut" if len(chunk_candidate_scores) > len(selected_chunk_candidates) else "candidate_pool_exhausted",
    }
    stage_queues["chunk"] = {"selected_ids": [chunk_id for chunk_id, _score in selected_chunk_candidates], "top_k": int(envelope["agent_chunk_top_k"])}
    for chunk_id, strength in selected_chunk_candidates:
        metadata = dict(chunk_candidate_metadata.get(chunk_id) or {})
        rq_prefix_ids = metadata.get("rq_prefix_ids") or []
        if query_rq and rq_prefix_ids:
            rows = [
                row
                for prefix_id in rq_prefix_ids
                for row in rq_rows_by_prefix.get(prefix_id, [])
                if row.chunk_id == chunk_id and row.rq_path
            ]
            if rows:
                metadata["rq"] = rq_candidate_score(query_rq, rows[0])
        for role in sorted(chunk_candidate_roles.get(chunk_id) or {"mid_drilldown_entry"}):
            add_seed(chunk_id, strength, role, metadata)

    required_facets = set(query_facets.get("required_facets") or [])

    def covered_facets_for_chunk(chunk: Chunk) -> set[str]:
        return set(matched_required_facets_for_text(chunk.text, query_facets))

    adjacency: dict[str, list[ChunkRelationEdge]] = defaultdict(list)
    if relation_state:
        for edge in db.scalars(select(ChunkRelationEdge).where(ChunkRelationEdge.graph_state_id == relation_state.id)).all():
            adjacency[edge.source_chunk_id].append(edge)
            adjacency[edge.target_chunk_id].append(edge)

    def neighbor_for(edge: ChunkRelationEdge, chunk_id: str) -> str:
        return edge.target_chunk_id if edge.source_chunk_id == chunk_id else edge.source_chunk_id

    entry_nodes = [
        {
            "layer": "chunk",
            "node_id": chunk_id,
            "entry_strength": strength,
            "roles": sorted(seed_roles[chunk_id]),
            "metadata": seed_metadata.get(chunk_id, {}),
        }
        for chunk_id, strength in sorted(seed_strengths.items(), key=lambda item: item[1], reverse=True)
    ]
    layer_walks["chunk"] = {
        "entry_nodes": entry_nodes,
        "accepted_nodes": [],
        "accepted_states": [],
        "frontier_pops": [],
        "frontier_json": [],
        "path_labels": [],
        "gray_zone_path_decisions": [],
        "convergence": {},
    }
    frontier: list[tuple[tuple[float, float, int, int], int, dict[str, Any]]] = []
    serial = 0
    for chunk_id, strength in seed_strengths.items():
        chunk = chunk_by_id[chunk_id]
        covered = covered_facets_for_chunk(chunk)
        distance = distance_from_strength(strength)
        state = {
            "layer": "chunk",
            "node_id": chunk_id,
            "path": [chunk_id],
            "path_edge_ids": [],
            "path_edge_distances": [],
            "distance_so_far": distance,
            "reward_so_far": 0.0,
            "covered_facets": sorted(covered),
            "evidence_roles": sorted(seed_roles[chunk_id]),
            "depth": 0,
            "visit_counts": {chunk_id: 1},
            "support_refs": seed_metadata.get(chunk_id, {}),
        }
        key = (len(required_facets - covered), distance, 0, -len(state["evidence_roles"]))
        heapq.heappush(frontier, (key, serial, state))
        serial += 1

    accepted_by_chunk: dict[str, dict[str, Any]] = {}
    path_labels: list[dict[str, Any]] = []
    frontier_pops: list[dict[str, Any]] = []
    frontier_snapshots: list[dict[str, Any]] = []
    gray_zone_decisions: list[dict[str, Any]] = []
    dominance_labels: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    dominance_pruned_count = 0
    hard_stop_pruned_count = 0
    red_zone_pruned_count = 0
    expansion_count = 0
    stop_reason = "frontier_empty"
    max_expansions = layer_frontier_budget("chunk", envelope)
    max_depth = int(envelope["max_depth_per_layer"])
    max_labels = int(envelope["max_labels_per_node"])

    while frontier and expansion_count < max_expansions:
        key, _serial, state = heapq.heappop(frontier)
        label_key = (key[0], round(key[1], 6), key[2], key[3])
        labels = dominance_labels[state["node_id"]]
        if len(labels) >= max_labels and any(existing <= label_key for existing in labels):
            dominance_pruned_count += 1
            continue
        labels.append(label_key)
        labels.sort()
        del labels[max_labels:]

        chunk = chunk_by_id.get(state["node_id"])
        if chunk is None or not passes_filters(db, chunk, filters):
            continue
        frontier_pops.append(state)
        frontier_snapshots.append(
            {
                "popped": state,
                "queue_size_after_pop": len(frontier),
                "key": list(key),
            }
        )
        current = accepted_by_chunk.get(chunk.id)
        if current is None or key < current["queue_key"]:
            accepted_by_chunk[chunk.id] = {"state": state, "queue_key": key}
        if len(accepted_by_chunk) >= top_k and set(state.get("covered_facets") or []) >= required_facets:
            stop_reason = "all_required_facets_covered"
            break
        if int(state["depth"]) >= max_depth:
            continue

        expanded_edge_ids: list[str] = []
        for edge in sorted(adjacency.get(chunk.id, []), key=lambda item: float(item.distance if item.distance is not None else distance_from_strength(item.raw_strength or 1e-6))):
            neighbor_id = neighbor_for(edge, chunk.id)
            neighbor = chunk_by_id.get(neighbor_id)
            if neighbor is None or not passes_filters(db, neighbor, filters):
                continue
            edge_distance = float(edge.distance if edge.distance is not None else distance_from_strength(edge.raw_strength or 1e-6))
            next_distance = round(float(state["distance_so_far"]) + edge_distance, 6)
            zone = _path_distance_zone(next_distance, envelope)
            if zone in {"red", "hard_stop"}:
                hard_stop_pruned_count += 1
                if zone == "red":
                    red_zone_pruned_count += 1
                gray_zone_decisions.append(
                    {
                        "layer": "chunk",
                        "edge_id": edge.id,
                        "from_chunk_id": chunk.id,
                        "to_chunk_id": neighbor_id,
                        "path_distance": next_distance,
                        "distance_zone": zone,
                        "decision": "red_zone_pruned" if zone == "red" else "hard_stop_pruned",
                    }
                )
                continue
            if zone == "gray":
                gray_zone_decisions.append(
                    {
                        "layer": "chunk",
                        "edge_id": edge.id,
                        "from_chunk_id": chunk.id,
                        "to_chunk_id": neighbor_id,
                        "path_distance": next_distance,
                        "distance_zone": zone,
                        "covered_facets": state.get("covered_facets") or [],
                        "rq_membership_diagnostics": seed_metadata.get(chunk.id, {}),
                        "support_refs": {"edge_id": edge.id, "edge_type": edge.edge_type},
                        "decision": "continue_path",
                    }
                )
            visit_counts = dict(state["visit_counts"])
            previous_visits = int(visit_counts.get(neighbor_id, 0))
            visit_counts[neighbor_id] = previous_visits + 1
            reward_increment, cycle_diagnostics = _cycle_reward_for_next_state(
                state=state,
                neighbor_id=neighbor_id,
                edge_id=edge.id,
                edge_distance=edge_distance,
                edge_strength=normalized_strength(edge.raw_strength or 1e-6),
                envelope=envelope,
            )
            reward_so_far = round(max(0.0, float(state["reward_so_far"]) + max(0.0, reward_increment)), 6)
            covered = set(state["covered_facets"]) | covered_facets_for_chunk(neighbor)
            roles = set(state["evidence_roles"])
            roles.add(edge.edge_type)
            next_state = {
                "layer": "chunk",
                "node_id": neighbor_id,
                "path": list(state["path"]) + [neighbor_id],
                "path_edge_ids": list(state["path_edge_ids"]) + [edge.id],
                "path_edge_distances": list(state.get("path_edge_distances") or []) + [edge_distance],
                "distance_so_far": next_distance,
                "reward_so_far": reward_so_far,
                "cycle_distance_rewards": list(state.get("cycle_distance_rewards") or []) + ([cycle_diagnostics] if cycle_diagnostics else []),
                "distance_zone": zone,
                "covered_facets": sorted(covered),
                "evidence_roles": sorted(roles),
                "depth": int(state["depth"]) + 1,
                "visit_counts": visit_counts,
                "support_refs": {"edge_id": edge.id, "edge_type": edge.edge_type},
            }
            next_key = (
                len(required_facets - covered),
                round(float(next_state["distance_so_far"]) - reward_so_far, 6),
                int(next_state["depth"]),
                -len(roles),
            )
            heapq.heappush(frontier, (next_key, serial, next_state))
            serial += 1
            expanded_edge_ids.append(edge.id)
            expansion_count += 1
            if expansion_count >= max_expansions:
                stop_reason = "hard_budget_hit"
                break
        path_labels.append(
            {
                "chunk_id": chunk.id,
                "path": state["path"],
                "path_edge_ids": state["path_edge_ids"],
                "covered_facets": state["covered_facets"],
                "distance_so_far": state["distance_so_far"],
                "reward_so_far": state["reward_so_far"],
                "expanded_edge_ids": expanded_edge_ids,
            }
        )

    accepted_items = sorted(accepted_by_chunk.values(), key=lambda item: item["queue_key"])[:top_k]
    accepted_chunk_ids = [item["state"]["node_id"] for item in accepted_items]
    rq_membership_by_chunk: dict[str, RQPrefixMembership] = {}
    if query_rq and accepted_chunk_ids:
        rq_rows = list(
            db.scalars(
                select(RQPrefixMembership).where(
                    RQPrefixMembership.chunk_id.in_(accepted_chunk_ids),
                    RQPrefixMembership.rq_path.is_not(None),
                )
            ).all()
        )
        for row in rq_rows:
            if not row.rq_path:
                continue
            current = rq_membership_by_chunk.get(row.chunk_id)
            row_rank = (
                1 if row.membership_reason == "rq_leaf" else 0,
                len(row.rq_path or []),
                float(row.membership_score or 0.0),
            )
            current_rank = (
                1 if current and current.membership_reason == "rq_leaf" else 0,
                len((current.rq_path if current else []) or []),
                float(current.membership_score or 0.0) if current else 0.0,
            )
            if current is None or row_rank > current_rank:
                rq_membership_by_chunk[row.chunk_id] = row
    results: list[dict[str, Any]] = []
    for item in accepted_items:
        state = item["state"]
        chunk = chunk_by_id[state["node_id"]]
        traversal_score = round(1.0 / (1.0 + max(0.0, float(state["distance_so_far"]) - float(state["reward_so_far"]))), 6)
        metadata = {
            **seed_metadata.get(chunk.id, {}),
            "traversal": {
                "path": state["path"],
                "path_edge_ids": state["path_edge_ids"],
                "covered_facets": state["covered_facets"],
                "distance_so_far": state["distance_so_far"],
                "reward_so_far": state["reward_so_far"],
                "evidence_roles": state["evidence_roles"],
                "why_selected": "accepted_by_priority_queue_graph_traversal",
            },
        }
        rq_membership = rq_membership_by_chunk.get(chunk.id)
        if query_rq and rq_membership is not None:
            metadata["rq"] = rq_candidate_score(query_rq, rq_membership)
        results.append(search_payload_for_chunk(db, chunk, traversal_score, {"traversal_score": traversal_score}, metadata))

    convergence = {
        "reason": stop_reason,
        "frontier_expansion_count": expansion_count,
        "dominance_pruned_count": dominance_pruned_count,
        "hard_stop_pruned_count": hard_stop_pruned_count,
        "red_zone_pruned_count": red_zone_pruned_count,
        "cycle_distance_reward_bounded": True,
        "path_distance_thresholds": {
            "green": envelope.get("path_distance_green_threshold"),
            "gray": envelope.get("path_distance_gray_threshold"),
            "hard": envelope.get("path_distance_hard_threshold"),
        },
        "accepted_chunk_count": len(results),
        "frontier_remaining": len(frontier),
    }
    chunk_convergence = dict(convergence)
    layer_walks["chunk"] = {
        "entry_nodes": entry_nodes,
        "accepted_nodes": accepted_chunk_ids,
        "accepted_states": [item["state"] for item in accepted_items],
        "frontier_pops": frontier_pops,
        "frontier_json": frontier_snapshots,
        "path_labels": path_labels,
        "gray_zone_path_decisions": gray_zone_decisions,
        "convergence": chunk_convergence,
    }
    all_entry_nodes = [
        entry
        for layer_name in ("coarse", "mid", "rq_membership", "chunk")
        for entry in (layer_walks.get(layer_name, {}).get("entry_nodes") or [])
    ]
    all_frontier = [
        snapshot
        for layer_name in ("coarse", "mid", "chunk")
        for snapshot in (layer_walks.get(layer_name, {}).get("frontier_json") or [])
    ]
    all_path_labels = [
        label
        for layer_name in ("coarse", "mid", "chunk")
        for label in (layer_walks.get(layer_name, {}).get("path_labels") or [])
    ]
    all_gray_zone_decisions = [
        decision
        for layer_name in ("coarse", "mid", "chunk")
        for decision in (layer_walks.get(layer_name, {}).get("gray_zone_path_decisions") or [])
    ]
    layer_convergence = {layer_name: (layer_walks.get(layer_name, {}).get("convergence") or {}) for layer_name in ("coarse", "mid", "rq_membership", "chunk")}
    convergence["layers"] = layer_convergence
    convergence["hard_stop_pruned_count"] = sum(int((layer.get("hard_stop_pruned_count") or 0)) for layer in layer_convergence.values())
    convergence["red_zone_pruned_count"] = sum(int((layer.get("red_zone_pruned_count") or 0)) for layer in layer_convergence.values())
    convergence["gray_zone_decision_count"] = len(all_gray_zone_decisions)
    granularity_audit = {
        "retrieval_granularity": retrieval_granularity,
        "coarse_skipped_reason": coarse_skipped_reason,
        "mid_direct_entry_count": len(mid_entries) if retrieval_granularity == "mid" else 0,
        "mid_entry_mode": "direct_mid" if retrieval_granularity == "mid" else "coarse_drilldown",
    }
    convergence["retrieval_granularity"] = retrieval_granularity
    convergence["granularity_audit"] = granularity_audit
    return {
        "retrieval_granularity": retrieval_granularity,
        "retrieval_granularity_audit": granularity_audit,
        "query_facets": query_facets,
        "entry_nodes": all_entry_nodes,
        "frontier_pops": frontier_pops,
        "frontier_json": all_frontier,
        "stage_queues": stage_queues,
        "candidate_pools": candidate_pools,
        "topk_selection": topk_selection,
        "path_labels": all_path_labels,
        "convergence": convergence,
        "gray_zone_path_decisions": all_gray_zone_decisions,
        "results": results,
        "coarse_entries": coarse_entries,
        "mid_entries": mid_entries,
        "rq_membership_entries": rq_membership_entries,
        "layer_walks": layer_walks,
    }


def passes_filters(db: Session, chunk: Chunk, filters: SearchFilters) -> bool:
    document = db.get(Document, chunk.document_id)
    if document is None:
        return False
    if filters.source_type and document.source_type != filters.source_type:
        return False
    if filters.tags and not set(filters.tags).intersection(set(document.tags or [])):
        return False
    if filters.partition and filters.partition not in (document.tags or []) and filters.partition != (chunk.metadata_json or {}).get("partition"):
        return False
    return True


def chunk_source_span(
    db: Session,
    chunk: Chunk,
    *,
    context_package_id: str | None = None,
    retrieval_trace_id: str | None = None,
    verification_id: str | None = None,
) -> dict[str, Any]:
    coordinate = db.scalar(
        select(ChunkCoordinate)
        .where(ChunkCoordinate.chunk_id == chunk.id)
        .order_by(ChunkCoordinate.confidence.desc())
    )
    structure_rows = db.execute(
        select(ChunkStructureMapping, ChunkStructureNode)
        .join(ChunkStructureNode, ChunkStructureMapping.structure_node_id == ChunkStructureNode.id)
        .where(ChunkStructureMapping.chunk_id == chunk.id)
        .order_by(ChunkStructureMapping.coverage_ratio.desc(), ChunkStructureNode.depth.desc())
        .limit(8)
    ).all()
    section_path = chunk.section_path or next(
        (
            node.path or node.title
            for _mapping, node in structure_rows
            if node.node_type in {"section", "document"}
        ),
        None,
    )
    bbox = dict((coordinate.bbox_json or {}) if coordinate else {})
    if not bbox:
        bbox = next((dict(node.bbox_json or {}) for _mapping, node in structure_rows if node.bbox_json), {})
    page_range = (coordinate.page_range_json if coordinate and coordinate.page_range_json else {"start": chunk.page_start, "end": chunk.page_end})
    return {
        "document_version_id": chunk.document_version_id,
        "chunk_id": chunk.id,
        "char_span": [chunk.char_start, chunk.char_end],
        "page_range": [page_range.get("start"), page_range.get("end")],
        "section_path": section_path,
        "structure_path": section_path,
        "structure_node_ids": [node.id for _mapping, node in structure_rows],
        "bbox": bbox,
        "context_package_id": context_package_id,
        "retrieval_trace_id": retrieval_trace_id,
        "verification_id": verification_id,
    }


def search_payload_for_chunk(db: Session, chunk: Chunk, score: float, scores: dict[str, float], metadata: dict[str, Any]) -> dict[str, Any]:
    document = db.get(Document, chunk.document_id)
    snippet = re.sub(r"\s+", " ", chunk.text).strip()[:280]
    source_span = chunk_source_span(db, chunk, retrieval_trace_id=metadata.get("retrieval_trace_id"))
    citation = {
        "chunk_id": chunk.id,
        "document_id": document.id if document else chunk.document_id,
        "document_title": document.title if document else "",
        "source_path": document.source_path if document else "",
        "partition": (document.tags or [None])[0] if document and document.tags else None,
        "section": chunk.section_path,
        "page_number": chunk.page_start,
        "snippet": snippet,
        "source_span": source_span,
    }
    return {
        "chunk_id": chunk.id,
        "active_chunk_id": None,
        "document_id": document.id if document else chunk.document_id,
        "document_version_id": chunk.document_version_id,
        "title": document.title if document else None,
        "text": chunk.text,
        "snippet": snippet,
        "score": round(float(score), 6),
        "citations": [citation],
        "metadata": {**metadata, "scores": {key: round(float(value), 6) for key, value in scores.items()}},
        "content": chunk.text,
        "document_title": document.title if document else None,
        "source_path": document.source_path if document else None,
        "partition": (document.tags or [None])[0] if document and document.tags else None,
        "source_type": document.source_type if document else None,
    }


def write_retrieval_trace(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    results: list[dict[str, Any]],
    context_state: ContextGraphState | None,
    traversal: dict[str, Any],
    query_rq: dict[str, Any] | None = None,
    result_top_k: int | None = None,
    retrieval_granularity: RetrievalGranularity = "mid",
) -> RetrievalTrace:
    candidate_rq = {
        item["chunk_id"]: (item.get("metadata") or {}).get("rq")
        for item in results
        if (item.get("metadata") or {}).get("rq")
    }
    path_labels = traversal.get("path_labels") or []
    concept_path = [
        {"layer": "coarse", "ids": list((traversal.get("coarse_entries") or {}).keys())[:8]},
        {"layer": "mid", "ids": list((traversal.get("mid_entries") or {}).keys())[:12]},
        {"layer": "rq_membership", "ids": list((traversal.get("rq_membership_entries") or {}).keys())[:12]},
        {"layer": "chunk", "ids": [item["chunk_id"] for item in results]},
    ]
    conversation_hash = stable_hash({"conversation_state": "none"})
    cache_components = context_graph_cache_key_components(
        knowledge_base_id=knowledge_base_id,
        query=query,
        filters=filters,
        context_state=context_state,
        retrieval_mode="layered_context_graph",
        retrieval_granularity=retrieval_granularity,
        result_top_k=result_top_k,
        conversation_state_scope_hash=conversation_hash,
        query_facets=traversal.get("query_facets") or {},
    )
    cache_key = stable_hash(cache_components)
    trace = RetrievalTrace(
        knowledge_base_id=knowledge_base_id,
        query=query,
        filters_json=filters.model_dump(),
        retrieval_mode="layered_context_graph",
        chunk_scope_hash=context_state.chunk_scope_hash if context_state else None,
        structure_graph_hash=context_state.structure_graph_hash if context_state else None,
        chunk_relation_graph_hash=context_state.chunk_relation_graph_hash if context_state else None,
        rq_membership_hash=context_state.rq_membership_hash if context_state else None,
        mid_concept_hash=context_state.mid_concept_hash if context_state else None,
        coarse_concept_hash=context_state.coarse_concept_hash if context_state else None,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        prompt_protocol_hash=context_state.prompt_protocol_hash if context_state else None,
        result_chunk_ids_json=[item["chunk_id"] for item in results],
        concept_path_json=concept_path,
        scores_json={},
        query_facets_json=traversal.get("query_facets") or {},
        entry_nodes_json=traversal.get("entry_nodes") or [],
        frontier_json=traversal.get("frontier_json") or [],
        stage_queues_json=traversal.get("stage_queues") or {},
        candidate_pools_json=traversal.get("candidate_pools") or {},
        topk_selection_json=traversal.get("topk_selection") or {},
        path_labels_json=path_labels,
        convergence_json=traversal.get("convergence") or {},
        edge_distance_protocol_hash=edge_distance_protocol_hash(),
        edge_projection_protocol_hash=edge_projection_protocol_hash(),
        traversal_protocol_hash=traversal_protocol_hash(),
        conversation_state_scope_hash=conversation_hash,
        diagnostics_json={
            "context_graph_state_id": context_state.id if context_state else None,
            "retrieval_granularity": retrieval_granularity,
            "cache_key": cache_key,
            "cache_key_components": cache_components,
            "coarse_skipped_reason": (traversal.get("retrieval_granularity_audit") or {}).get("coarse_skipped_reason"),
            "mid_direct_entry_audit": {
                "entry_mode": (traversal.get("retrieval_granularity_audit") or {}).get("mid_entry_mode"),
                "direct_entry_count": (traversal.get("retrieval_granularity_audit") or {}).get("mid_direct_entry_count"),
            },
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope": agent_operating_envelope(),
            "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
            "result_top_k": result_top_k,
            "scores_json_retired_as_primary_audit": True,
            "traversal_protocol": TRAVERSAL_PROTOCOL_VERSION,
            "rq": {
                "query_rq_path": (query_rq or {}).get("rq_path") or [],
                "query_residual_norm": (query_rq or {}).get("residual_norm"),
                "index_protocol": (query_rq or {}).get("index_protocol"),
            },
        },
    )
    db.add(trace)
    db.flush()
    layer_walks = traversal.get("layer_walks") or {}
    steps = []
    for layer in ("coarse", "mid"):
        walk = layer_walks.get(layer) or {}
        if layer == "coarse":
            input_json = {"entry_nodes": walk.get("entry_nodes") or [], "query_facets": traversal.get("query_facets") or {}}
            output_json = {"accepted_nodes": walk.get("accepted_nodes") or [], "convergence": walk.get("convergence") or {}}
            steps.append(
                (
                    layer,
                    "select_entry_nodes",
                    {"query_facets": traversal.get("query_facets") or {}, "retrieval_granularity": retrieval_granularity},
                    {"selected_entry_nodes": walk.get("entry_nodes") or [], "skipped_by_granularity": "mid" if retrieval_granularity == "mid" else None},
                    walk,
                )
            )
            steps.append((layer, "staged_priority_queue_walk", input_json, output_json, walk))
        else:
            input_json = {"entry_nodes": walk.get("entry_nodes") or [], "coarse_entry_ids": list((traversal.get("coarse_entries") or {}).keys())}
            output_json = {"accepted_nodes": walk.get("accepted_nodes") or [], "convergence": walk.get("convergence") or {}}
            steps.append((layer, "drill_down_each_coarse_or_direct_mid_entry", input_json, output_json, walk))
    seed_walk = layer_walks.get("rq_membership") or {}
    steps.append(
        (
            "chunk",
            "select_seeds_from_mid_rq_membership",
            {
                "rq_membership_entries": seed_walk.get("entry_nodes") or [],
                "mid_entry_ids": list((traversal.get("mid_entries") or {}).keys()),
                "query_rq_path": (query_rq or {}).get("rq_path") or [],
            },
            {
                "selected_rq_memberships": seed_walk.get("accepted_nodes") or [],
                "candidate_rq": candidate_rq,
                "convergence": seed_walk.get("convergence") or {},
            },
            seed_walk,
        )
    )
    chunk_walk = layer_walks.get("chunk") or {}
    steps.append(
        (
            "chunk",
            "walk_graph_frontier",
            {"entry_nodes": chunk_walk.get("entry_nodes") or [], "query_rq_path": (query_rq or {}).get("rq_path") or []},
            {
                "accepted_chunks": {item["chunk_id"]: {"score": item["score"], "rq": (item.get("metadata") or {}).get("rq")} for item in results},
                "convergence": chunk_walk.get("convergence") or traversal.get("convergence") or {},
            },
            chunk_walk,
        )
    )
    structure_spans = [
        {
            "chunk_id": item.get("chunk_id"),
            "source_span": (((item.get("citations") or [{}])[0] or {}).get("source_span") or {}),
        }
        for item in results
    ]
    steps.append(
        (
            "structure",
            "restore_context_package",
            {"result_chunk_ids": [item["chunk_id"] for item in results]},
            {
                "restored_chunk_count": len(structure_spans),
                "source_spans": structure_spans,
                "convergence": {"reason": "structure_restored"},
            },
            {"convergence": {"reason": "structure_restored"}, "path_labels": []},
        )
    )
    for index, (layer, action, input_json, output_json, walk) in enumerate(steps):
        popped = walk.get("frontier_pops") or []
        popped_state = popped[0] if popped else {}
        layer_path_labels = walk.get("path_labels") or []
        expanded_edge_ids = list(dict.fromkeys(edge_id for label in layer_path_labels for edge_id in (label.get("expanded_edge_ids") or [])))
        cycle_distance_reward = max([float(label.get("reward_so_far") or 0.0) for label in layer_path_labels] + [0.0])
        convergence = walk.get("convergence") or {}
        parent_layer = None
        parent_node_id = None
        candidate_pool_ids: list[str] = []
        selected_topk_ids: list[str] = []
        per_parent_budget_status: dict[str, Any] = {}
        candidate_pools = traversal.get("candidate_pools") or {}
        topk_selection = traversal.get("topk_selection") or {}
        if layer == "mid":
            pools = candidate_pools.get("mid_by_coarse") or []
            parent_layer = "coarse"
            parent_node_id = (pools[0] or {}).get("parent_node_id") if pools else None
            candidate_pool_ids = list(dict.fromkeys(item for pool in pools for item in (pool.get("candidate_ids") or [])))
            selected_topk_ids = list((topk_selection.get("mid") or {}).get("selected_ids") or [])
            per_parent_budget_status = {pool.get("parent_node_id"): pool.get("per_parent_budget_status") for pool in pools if pool.get("parent_node_id")}
        elif action == "select_seeds_from_mid_rq_membership":
            pools = candidate_pools.get("chunk_by_mid") or []
            parent_layer = "mid"
            parent_node_id = (pools[0] or {}).get("parent_node_id") if pools else None
            candidate_pool_ids = list(dict.fromkeys(item for pool in pools for item in (pool.get("candidate_ids") or [])))
            selected_topk_ids = list((topk_selection.get("chunk") or {}).get("selected_ids") or [])
            per_parent_budget_status = {pool.get("parent_node_id"): pool.get("per_parent_budget_status") for pool in pools if pool.get("parent_node_id")}
            output_json = {**output_json, "candidate_count": len(candidate_pool_ids), "selected_topk_count": len(selected_topk_ids)}
        elif layer == "chunk":
            selected_topk_ids = list((topk_selection.get("chunk") or {}).get("selected_ids") or [])
        db.add(
            GraphRetrievalStep(
                retrieval_trace_id=trace.id,
                knowledge_base_id=knowledge_base_id,
                step_index=index,
                layer=layer,
                action=action,
                action_type=action,
                parent_layer=parent_layer,
                parent_node_id=parent_node_id,
                input_json=input_json,
                output_json=output_json,
                score_json={},
                popped_frontier_state_json=popped_state,
                expanded_edge_ids_json=expanded_edge_ids,
                candidate_pool_ids_json=candidate_pool_ids,
                selected_topk_ids_json=selected_topk_ids,
                dominance_pruned_count=int(convergence.get("dominance_pruned_count") or 0),
                cycle_distance_reward=cycle_distance_reward,
                gray_zone_path_decisions_json=walk.get("gray_zone_path_decisions") or [],
                per_parent_budget_status_json=per_parent_budget_status,
                stop_reason=str(convergence.get("reason") or ""),
                diagnostics_json={
                    "retrieval_granularity": retrieval_granularity,
                    "traversal_protocol": TRAVERSAL_PROTOCOL_VERSION,
                    "scores_json_retired_as_primary_audit": True,
                    "frontier_json": walk.get("frontier_json") or [],
                    "path_labels": layer_path_labels,
                    "stage_queues": traversal.get("stage_queues") or {},
                    "candidate_pools": candidate_pools,
                    "topk_selection": topk_selection,
                },
            )
        )
    db.flush()
    return trace


def build_context_package(
    db: Session,
    *,
    knowledge_base_id: str,
    query: str,
    trace: RetrievalTrace,
    results: list[dict[str, Any]],
    token_budget: int | None = None,
) -> ContextPackage:
    from app.services.strategy_profiles import active_profile_hash

    token_budget = token_budget or int(get_settings().context_package_token_budget or 2400)
    profile_hash = active_profile_hash(db, knowledge_base_id)
    hit_ids = [item["chunk_id"] for item in results]
    hit_id_set = set(hit_ids)
    selected_ids: list[str] = []
    bridge_ids: set[str] = set()
    graph_path_chunk_ids: set[str] = set()
    parent_node_ids: set[str] = set()
    chunks_by_id: dict[str, Chunk] = {}
    structure_by_chunk_id: dict[str, dict[str, Any]] = {}

    def add_chunk_id(chunk_id: str | None) -> None:
        if not chunk_id or chunk_id in selected_ids:
            return
        chunk = db.get(Chunk, chunk_id)
        if chunk is None or chunk.knowledge_base_id != knowledge_base_id or chunk.state != "active":
            return
        chunks_by_id[chunk.id] = chunk
        selected_ids.append(chunk.id)

    for item in results:
        traversal = (item.get("metadata") or {}).get("traversal") or {}
        for path_chunk_id in traversal.get("path") or []:
            if path_chunk_id and path_chunk_id not in hit_id_set:
                graph_path_chunk_ids.add(path_chunk_id)

    for chunk_id in hit_ids:
        chunk = db.get(Chunk, chunk_id)
        add_chunk_id(chunk_id)
        if chunk:
            add_chunk_id(chunk.previous_chunk_id)
            add_chunk_id(chunk.next_chunk_id)
            for mapping in db.scalars(select(ChunkStructureMapping).where(ChunkStructureMapping.chunk_id == chunk.id)).all():
                parent_node_ids.add(mapping.structure_node_id)
        for edge in db.scalars(
            select(ChunkRelationEdge)
            .where(
                ChunkRelationEdge.knowledge_base_id == knowledge_base_id,
                ChunkRelationEdge.is_bridge.is_(True),
                (ChunkRelationEdge.source_chunk_id == chunk_id) | (ChunkRelationEdge.target_chunk_id == chunk_id),
            )
            .order_by(ChunkRelationEdge.distance.asc())
            .limit(2)
        ).all():
            other_id = edge.target_chunk_id if edge.source_chunk_id == chunk_id else edge.source_chunk_id
            bridge_ids.add(other_id)
            add_chunk_id(other_id)

    for chunk_id in sorted(graph_path_chunk_ids):
        add_chunk_id(chunk_id)
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            for mapping in db.scalars(select(ChunkStructureMapping).where(ChunkStructureMapping.chunk_id == chunk.id)).all():
                parent_node_ids.add(mapping.structure_node_id)

    def structure_context(chunk: Chunk) -> dict[str, Any]:
        if chunk.id in structure_by_chunk_id:
            return structure_by_chunk_id[chunk.id]
        rows = db.execute(
            select(ChunkStructureMapping, ChunkStructureNode)
            .join(ChunkStructureNode, ChunkStructureMapping.structure_node_id == ChunkStructureNode.id)
            .where(ChunkStructureMapping.chunk_id == chunk.id)
            .order_by(ChunkStructureMapping.coverage_ratio.desc(), ChunkStructureNode.depth.desc())
        ).all()
        nodes = [
            {
                "node_id": node.id,
                "node_type": node.node_type,
                "title": node.title,
                "path": node.path,
                "depth": node.depth,
                "page_number": node.page_number,
                "bbox": node.bbox_json or {},
                "layout": node.layout_json or {},
                "mapping_role": mapping.mapping_role,
                "coverage_ratio": mapping.coverage_ratio,
            }
            for mapping, node in rows
        ]
        section_nodes = [
            node
            for node in nodes
            if node.get("node_type") in {"section", "heading", "document"}
        ]
        parent_section = section_nodes[0] if section_nodes else (nodes[0] if nodes else None)
        path = (
            chunk.section_path
            or (parent_section or {}).get("path")
            or (parent_section or {}).get("title")
            or (nodes[0].get("path") if nodes else None)
        )
        value = {
            "structure_path": path,
            "structure_node_ids": [node["node_id"] for node in nodes],
            "structure_nodes": nodes,
            "parent_section": parent_section,
            "same_page_region": [node for node in nodes if node.get("node_type") in {"page", "region"}],
            "table_formula_caption": [node for node in nodes if node.get("node_type") in {"table", "formula", "caption"}],
            "code_blocks": [node for node in nodes if node.get("node_type") == "code_block"],
        }
        structure_by_chunk_id[chunk.id] = value
        return value

    path_labels = list(trace.path_labels_json or [])
    traversal_by_chunk: dict[str, dict[str, Any]] = {}
    for item in results:
        traversal = (item.get("metadata") or {}).get("traversal") or {}
        for path_chunk_id in traversal.get("path") or []:
            if path_chunk_id and path_chunk_id not in traversal_by_chunk:
                traversal_by_chunk[path_chunk_id] = {
                    **traversal,
                    "why_selected": "restored_from_selected_graph_path",
                }
        traversal_by_chunk[item["chunk_id"]] = traversal

    def why_selected_for_chunk(chunk_id: str, role: str) -> dict[str, Any]:
        traversal = traversal_by_chunk.get(chunk_id) or {}
        return {
            "roles": traversal.get("evidence_roles") or ([role] if role else []),
            "path_edge_ids": traversal.get("path_edge_ids") or [],
            "covered_facets": traversal.get("covered_facets") or [],
            "reason": traversal.get("why_selected") or role or "selected_for_structure_restoration",
        }

    package_chunks: list[dict[str, Any]] = []
    token_count = 0
    for chunk_id in selected_ids:
        chunk = chunks_by_id[chunk_id]
        chunk_tokens = rough_token_count(chunk.text)
        if package_chunks and token_count + chunk_tokens > token_budget:
            continue
        document = db.get(Document, chunk.document_id)
        structure = structure_context(chunk)
        role = "hit" if chunk.id in hit_id_set else "bridge" if chunk.id in bridge_ids else "graph_path" if chunk.id in graph_path_chunk_ids else "restored_context"
        source_span = chunk_source_span(db, chunk, retrieval_trace_id=trace.id)
        dedupe_key = f"{chunk.id}:{[chunk.char_start, chunk.char_end]}"
        package_chunks.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "document_title": document.title if document else "",
                "source_path": document.source_path if document else "",
                "content": chunk.text,
                "section_path": chunk.section_path,
                "structure_path": structure["structure_path"],
                "structure_node_ids": structure["structure_node_ids"],
                "structure_nodes": structure["structure_nodes"],
                "parent_section": structure["parent_section"],
                "page_range": [chunk.page_start, chunk.page_end],
                "char_span": [chunk.char_start, chunk.char_end],
                "bbox": source_span.get("bbox") or {},
                "source_span": source_span,
                "structure_closure": {
                    "previous_chunk_id": chunk.previous_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                    "parent_section": structure["parent_section"],
                    "same_page_region": structure["same_page_region"],
                    "table_formula_caption": structure["table_formula_caption"],
                    "code_blocks": structure["code_blocks"],
                    "bridge_chunk_ids": sorted(bridge_ids),
                },
                "why_selected": why_selected_for_chunk(chunk.id, role),
                "dedupe_key": dedupe_key,
                "role": role,
            }
        )
        token_count += chunk_tokens
    graph_path_ids = list(
        dict.fromkeys(
            edge_id
            for label in path_labels
            for edge_id in (label.get("path_edge_ids") or [])
        )
    )
    why_selected = {
        chunk_id: why_selected_for_chunk(chunk_id, "hit" if chunk_id in hit_ids else "bridge" if chunk_id in bridge_ids else "restored_context")
        for chunk_id in selected_ids
    }
    dedupe_keys = [
        item.get("dedupe_key") or f"{item['chunk_id']}:{item.get('char_span')}"
        for item in package_chunks
    ]
    covered_facets = sorted(
        {
            facet
            for item in why_selected.values()
            for facet in (item.get("covered_facets") or [])
        }
    )
    cycle_convergence_score = round(
        sum(float((traversal_by_chunk.get(chunk_id) or {}).get("reward_so_far") or 0.0) for chunk_id in hit_ids),
        6,
    )
    package = ContextPackage(
        knowledge_base_id=knowledge_base_id,
        retrieval_trace_id=trace.id,
        query=query,
        hit_chunk_ids_json=hit_ids,
        restored_chunk_ids_json=[chunk_id for chunk_id in selected_ids if chunk_id not in hit_ids],
        bridge_chunk_ids_json=sorted(bridge_ids),
        parent_structure_node_ids_json=sorted(parent_node_ids),
        concept_path_json=trace.concept_path_json or [],
        package_json={"chunks": package_chunks},
        graph_path_ids_json=graph_path_ids,
        why_selected_json=why_selected,
        cycle_convergence_score=cycle_convergence_score,
        dedupe_keys_json=dedupe_keys,
        covered_facets_json=covered_facets,
        token_budget=token_budget,
        token_count=token_count,
        runtime_settings_hash=runtime_settings_state_hash(),
        profile_hash=profile_hash,
        citation_spans_json=[],
        diagnostics_json={
            "context_restoration_protocol": "previous_next_structure_bridge_v1",
            "retrieval_granularity": (trace.diagnostics_json or {}).get("retrieval_granularity", "mid"),
            "runtime_settings_hash": runtime_settings_state_hash(),
            "profile_hash": profile_hash,
            "path_summary": {
                "distinct_path_count": len({tuple(label.get("path") or []) for label in path_labels}),
                "distinct_edge_type_count": len(
                    {
                        role
                        for item in traversal_by_chunk.values()
                        for role in (item.get("evidence_roles") or [])
                    }
                ),
                "covered_facets": covered_facets,
                "cycle_convergence_score": cycle_convergence_score,
            },
            "dedupe_keys": dedupe_keys,
            "restore_counts": {
                "hit_chunks": len(hit_ids),
                "restored_chunks": len([chunk_id for chunk_id in selected_ids if chunk_id not in hit_ids]),
                "bridge_chunks": len(bridge_ids),
                "graph_path_chunks": len(graph_path_chunk_ids),
                "parent_structure_nodes": len(parent_node_ids),
            },
        },
    )
    db.add(package)
    db.flush()
    for item in package_chunks:
        source_span = item.get("source_span")
        if isinstance(source_span, dict):
            source_span["context_package_id"] = package.id
        item["context_package_id"] = package.id
    citation_spans = [
        {
            **(item.get("source_span") or {}),
            "document_id": item["document_id"],
            "document_title": item.get("document_title") or "",
            "source_path": item.get("source_path") or "",
            "section_path": item.get("section_path"),
            "structure_path": item.get("structure_path"),
            "structure_node_ids": item.get("structure_node_ids") or [],
            "structure_closure": item.get("structure_closure") or {},
        }
        for item in package_chunks
        if item["chunk_id"] in hit_ids
    ]
    package.package_json = {"chunks": package_chunks}
    package.citation_spans_json = citation_spans
    structure_output = {
        "context_package_id": package.id,
        "restored_chunk_ids": list(package.restored_chunk_ids_json or []),
        "bridge_chunk_ids": list(package.bridge_chunk_ids_json or []),
        "parent_structure_node_ids": list(package.parent_structure_node_ids_json or []),
        "citation_spans": citation_spans,
        "graph_path_ids": graph_path_ids,
        "why_selected": why_selected,
        "dedupe_keys": dedupe_keys,
        "token_count": token_count,
        "token_budget": token_budget,
    }
    structure_diagnostics = {
        **(package.diagnostics_json or {}),
        "token_count": token_count,
        "token_budget": token_budget,
        "scores_json_retired_as_primary_audit": True,
    }
    structure_step = db.scalar(
        select(GraphRetrievalStep)
        .where(
            GraphRetrievalStep.retrieval_trace_id == trace.id,
            GraphRetrievalStep.layer == "structure",
            GraphRetrievalStep.action_type == "restore_context_package",
        )
        .order_by(GraphRetrievalStep.step_index.asc())
    )
    if structure_step is None:
        next_index = (
            db.scalar(select(func.max(GraphRetrievalStep.step_index)).where(GraphRetrievalStep.retrieval_trace_id == trace.id))
            or -1
        ) + 1
        structure_step = GraphRetrievalStep(
            retrieval_trace_id=trace.id,
            knowledge_base_id=knowledge_base_id,
            step_index=next_index,
            layer="structure",
            action="restore_context_package",
            action_type="restore_context_package",
            score_json={},
        )
        db.add(structure_step)
    structure_step.input_json = {"hit_chunk_ids": hit_ids, "token_budget": token_budget}
    structure_step.output_json = structure_output
    structure_step.popped_frontier_state_json = {}
    structure_step.expanded_edge_ids_json = graph_path_ids
    structure_step.selected_topk_ids_json = hit_ids
    structure_step.dominance_pruned_count = int((trace.convergence_json or {}).get("dominance_pruned_count") or 0)
    structure_step.cycle_distance_reward = cycle_convergence_score
    structure_step.gray_zone_path_decisions_json = trace.convergence_json.get("gray_zone_path_decisions", []) if isinstance(trace.convergence_json, dict) else []
    structure_step.stop_reason = "context_package_built"
    structure_step.diagnostics_json = structure_diagnostics
    db.flush()
    return package


def context_package_to_contexts(package: ContextPackage) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["chunk_id"],
            "document_title": item.get("document_title") or "",
            "content": item.get("content") or "",
            "snippet": (item.get("content") or "")[:280],
            "metadata": {
                "section_path": item.get("section_path"),
                "structure_path": item.get("structure_path"),
                "parent_section": item.get("parent_section"),
                "structure_node_ids": item.get("structure_node_ids") or [],
                "page_range": item.get("page_range"),
                "bbox": item.get("bbox") or (item.get("source_span") or {}).get("bbox") or {},
                "source_span": item.get("source_span") or {},
                "structure_closure": item.get("structure_closure") or {},
                "why_selected": item.get("why_selected") or {},
                "dedupe_key": item.get("dedupe_key"),
                "role": item.get("role"),
                "context_package_id": package.id,
            },
        }
        for item in (package.package_json or {}).get("chunks", [])
    ]


def _state_bound_layers(
    db: Session,
    knowledge_base_id: str,
    state: ContextGraphState | None,
) -> tuple[ChunkRelationGraphState | None, MidConceptState | None, CoarseConceptState | None]:
    relation_state = db.get(ChunkRelationGraphState, state.chunk_relation_graph_state_id) if state and state.chunk_relation_graph_state_id else None
    mid_state = db.get(MidConceptState, state.mid_concept_state_id) if state and state.mid_concept_state_id else None
    coarse_state = db.get(CoarseConceptState, state.coarse_concept_state_id) if state and state.coarse_concept_state_id else None
    return (
        relation_state if relation_state and relation_state.state == "active" else latest_relation_state(db, knowledge_base_id),
        mid_state if mid_state and mid_state.state == "active" else latest_mid_state(db, knowledge_base_id),
        coarse_state if coarse_state and coarse_state.state == "active" else latest_coarse_state(db, knowledge_base_id),
    )


def _active_document_version_ids(chunks: list[Chunk]) -> list[str]:
    return sorted({chunk.document_version_id for chunk in chunks if chunk.document_version_id})


def _layer_full_counts(counts: dict[str, int], layer: str) -> dict[str, int]:
    if layer == "chunk-structure":
        return {"nodes": counts.get("structure_nodes", 0), "edges": counts.get("structure_edges", 0)}
    if layer == "chunk-relation":
        return {
            "nodes": counts.get("active_chunks", 0) + counts.get("rq_prefixes", 0),
            "edges": counts.get("chunk_relation_edges", 0) + counts.get("rq_prefix_memberships", 0),
        }
    if layer == "mid-concepts":
        return {
            "nodes": counts.get("mid_concepts", 0),
            "edges": counts.get("mid_concept_edges", 0) + counts.get("mid_concept_memberships", 0),
        }
    if layer == "coarse-concepts":
        return {
            "nodes": counts.get("coarse_concepts", 0),
            "edges": counts.get("coarse_concept_edges", 0) + counts.get("coarse_concept_memberships", 0),
        }
    return {"nodes": 0, "edges": 0}


def context_graph_stats(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    state = latest_context_graph_state(db, knowledge_base_id)
    relation_state, mid_state, coarse_state = _state_bound_layers(db, knowledge_base_id, state)
    document_version_ids = _active_document_version_ids(chunks)
    structure_node_count = (
        db.scalar(select(func.count(ChunkStructureNode.id)).where(ChunkStructureNode.document_version_id.in_(document_version_ids)))
        if document_version_ids
        else 0
    ) or 0
    structure_edge_count = (
        db.scalar(select(func.count(ChunkStructureEdge.id)).where(ChunkStructureEdge.document_version_id.in_(document_version_ids)))
        if document_version_ids
        else 0
    ) or 0
    structure_mapping_count = (
        db.scalar(select(func.count(ChunkStructureMapping.id)).where(ChunkStructureMapping.document_version_id.in_(document_version_ids)))
        if document_version_ids
        else 0
    ) or 0
    relation_state_id = relation_state.id if relation_state else None
    mid_state_id = mid_state.id if mid_state else None
    coarse_state_id = coarse_state.id if coarse_state else None
    counts = {
        "chunks": len(chunks),
        "active_chunks": len(chunks),
        "structure_nodes": structure_node_count,
        "structure_edges": structure_edge_count,
        "structure_mappings": structure_mapping_count,
        "chunk_relation_edges": (
            db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id))
            if relation_state_id
            else 0
        ) or 0,
        "rq_prefixes": (
            db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.graph_state_id == relation_state_id, RQPrefix.state == "active"))
            if relation_state_id
            else 0
        ) or 0,
        "rq_prefix_memberships": (
            db.scalar(
                select(func.count(RQPrefixMembership.id))
                .join(RQPrefix, RQPrefixMembership.rq_prefix_id == RQPrefix.id)
                .where(RQPrefix.graph_state_id == relation_state_id)
            )
            if relation_state_id
            else 0
        ) or 0,
        "rq_relation_edges": (
            db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id, ChunkRelationEdge.edge_type.like("rq_%")))
            if relation_state_id
            else 0
        ) or 0,
        "mid_concepts": (
            db.scalar(select(func.count(MidConcept.id)).where(MidConcept.concept_state_id == mid_state_id, MidConcept.state == "active"))
            if mid_state_id
            else 0
        ) or 0,
        "mid_concept_edges": (
            db.scalar(select(func.count(MidConceptEdge.id)).where(MidConceptEdge.concept_state_id == mid_state_id))
            if mid_state_id
            else 0
        ) or 0,
        "mid_concept_memberships": (
            db.scalar(
                select(func.count(MidConceptMembership.id))
                .join(MidConcept, MidConceptMembership.mid_concept_id == MidConcept.id)
                .where(MidConcept.concept_state_id == mid_state_id)
            )
            if mid_state_id
            else 0
        ) or 0,
        "coarse_concepts": (
            db.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.coarse_state_id == coarse_state_id, CoarseConcept.state == "active"))
            if coarse_state_id
            else 0
        ) or 0,
        "coarse_concept_edges": (
            db.scalar(select(func.count(CoarseConceptEdge.id)).where(CoarseConceptEdge.coarse_state_id == coarse_state_id))
            if coarse_state_id
            else 0
        ) or 0,
        "coarse_concept_memberships": (
            db.scalar(
                select(func.count(CoarseConceptMembership.id))
                .join(CoarseConcept, CoarseConceptMembership.coarse_concept_id == CoarseConcept.id)
                .where(CoarseConcept.coarse_state_id == coarse_state_id)
            )
            if coarse_state_id
            else 0
        ) or 0,
    }
    return {
        "knowledge_base_id": knowledge_base_id,
        "context_graph_state_id": state.id if state else None,
        "active_state_ids": {
            "context_graph_state_id": state.id if state else None,
            "chunk_relation_graph_state_id": relation_state_id,
            "mid_concept_state_id": mid_state_id,
            "coarse_concept_state_id": coarse_state_id,
        },
        "counts": counts,
        "full_counts_by_layer": {layer: _layer_full_counts(counts, layer) for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts")},
        "freshness": freshness_payload(db, knowledge_base_id, state),
        "hashes": {
            "chunk_scope_hash": state.chunk_scope_hash if state else compute_chunk_scope_hash(chunks) if chunks else None,
            "structure_graph_hash": state.structure_graph_hash if state else None,
            "chunk_relation_graph_hash": state.chunk_relation_graph_hash if state else relation_state.state_hash if relation_state else None,
            "mid_concept_hash": state.mid_concept_hash if state else mid_state.state_hash if mid_state else None,
            "coarse_concept_hash": state.coarse_concept_hash if state else coarse_state.state_hash if coarse_state else None,
            "context_graph_hash": state.context_graph_hash if state else None,
            "runtime_settings_hash": state.runtime_settings_hash if state else runtime_settings_state_hash(),
            "agent_operating_envelope_hash": state.agent_operating_envelope_hash if state else agent_operating_envelope_state_hash(),
            "policy_state_hash": state.policy_state_hash if state else None,
            "prompt_protocol_hash": state.prompt_protocol_hash if state else None,
        },
        "diagnostics": {
            "relation": relation_diagnostics_summary(relation_state.diagnostics_json if relation_state else {}),
            "mid": mid_state.diagnostics_json if mid_state else {},
            "coarse": coarse_state.diagnostics_json if coarse_state else {},
        },
    }


def relation_diagnostics_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    summary = dict(diagnostics or {})
    rq_model = summary.get("rq_kmeans")
    if isinstance(rq_model, dict):
        summary["rq_kmeans"] = {
            "enabled": bool(rq_model.get("enabled")),
            "levels": rq_model.get("levels"),
            "codebook_sizes": rq_model.get("codebook_sizes"),
            "embedding_dimensions": rq_model.get("embedding_dimensions"),
            "tau_r": rq_model.get("tau_r"),
            "index_protocol": rq_model.get("index_protocol"),
            "chunk_count": rq_model.get("chunk_count"),
        }
    return summary


def freshness_payload(db: Session, knowledge_base_id: str, state: ContextGraphState | None) -> dict[str, Any]:
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    current_hash = compute_chunk_scope_hash(chunks) if chunks else None
    stale = bool(state and current_hash and state.chunk_scope_hash != current_hash)
    reasons = ["chunk_scope_changed"] if stale else []
    if state is None and chunks:
        stale = True
        reasons = ["context_graph_missing"]
    return {
        "is_stale": stale,
        "stale_reasons": reasons,
        "current_chunk_scope_hash": current_hash,
        "context_graph_state_id": state.id if state else None,
        "context_graph_hash": state.context_graph_hash if state else None,
    }


def graph_layer_payload(db: Session, knowledge_base_id: str, layer: str, *, limit: int = 200) -> dict[str, Any]:
    stats = context_graph_stats(db, knowledge_base_id)
    context_state = latest_context_graph_state(db, knowledge_base_id)
    relation_state, mid_state, coarse_state = _state_bound_layers(db, knowledge_base_id, context_state)
    active_chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    active_document_versions = _active_document_version_ids(active_chunks)
    full_counts = stats["full_counts_by_layer"].get(layer, _layer_full_counts(stats["counts"], layer))
    if layer == "chunk-structure":
        nodes = [
            {
                "id": node.id,
                "label": node.title or node.node_type,
                "type": node.node_type,
                "name": node.title or node.node_type,
                "category": node.node_type,
                "document_id": node.document_id,
                "document_version_id": node.document_version_id,
                "page_number": node.page_number,
                "snippet": node.path,
            }
            for node in db.scalars(
                select(ChunkStructureNode)
                .where(ChunkStructureNode.document_version_id.in_(active_document_versions))
                .limit(limit)
            ).all()
        ]
        edges = [
            {"id": edge.id, "source": edge.source_node_id, "target": edge.target_node_id, "label": edge.edge_type, "type": edge.edge_type, "weight": edge.weight, "category": edge.edge_type}
            for edge in db.scalars(
                select(ChunkStructureEdge)
                .where(ChunkStructureEdge.document_version_id.in_(active_document_versions))
                .limit(limit * 2)
            ).all()
        ]
    elif layer == "chunk-relation":
        chunk_rows = list(db.scalars(active_chunks_query(knowledge_base_id).limit(limit)).all())
        cluster_rows: list[RQPrefix] = []
        membership_edge_rows: list[RQPrefixMembership] = []
        rq_membership_by_chunk: dict[str, RQPrefixMembership] = {}
        if relation_state:
            cluster_rows = list(
                db.scalars(
                    select(RQPrefix)
                    .where(RQPrefix.graph_state_id == relation_state.id, RQPrefix.state == "active")
                    .order_by(RQPrefix.rq_level.is_(None).asc(), RQPrefix.rq_level.asc(), RQPrefix.rq_prefix_key.asc())
                    .limit(limit)
                ).all()
            )
            cluster_ids = [cluster.id for cluster in cluster_rows]
            if chunk_rows and cluster_ids:
                membership_edge_rows = list(
                    db.scalars(
                        select(RQPrefixMembership)
                        .where(
                            RQPrefixMembership.chunk_id.in_([chunk.id for chunk in chunk_rows]),
                            RQPrefixMembership.rq_prefix_id.in_(cluster_ids),
                        )
                        .limit(limit * 2)
                    ).all()
                )
        if relation_state and chunk_rows:
            rows = db.scalars(
                select(RQPrefixMembership)
                .join(RQPrefix, RQPrefixMembership.rq_prefix_id == RQPrefix.id)
                .where(
                    RQPrefix.graph_state_id == relation_state.id,
                    RQPrefixMembership.chunk_id.in_([chunk.id for chunk in chunk_rows]),
                )
            ).all()
            for row in rows:
                if row.rq_path and (row.chunk_id not in rq_membership_by_chunk or row.membership_reason == "rq_leaf"):
                    rq_membership_by_chunk[row.chunk_id] = row
        chunk_nodes = [
            {
                "id": chunk.id,
                "label": chunk.section_path or f"Chunk {chunk.chunk_index}",
                "type": "chunk",
                "name": chunk.section_path or f"Chunk {chunk.chunk_index}",
                "category": "chunk",
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "snippet": chunk.text[:180],
                "page_number": chunk.page_start,
                "metadata": {
                    "rq_path": chunk.rq_path or (rq_membership_by_chunk.get(chunk.id).rq_path if rq_membership_by_chunk.get(chunk.id) else []),
                    "residual_norm": (
                        chunk.rq_residual_norm
                        if chunk.rq_residual_norm is not None
                        else (rq_membership_by_chunk.get(chunk.id).residual_norm if rq_membership_by_chunk.get(chunk.id) else None)
                    ),
                },
            }
            for chunk in chunk_rows
        ]
        cluster_nodes = [
            {
                "id": cluster.id,
                "label": cluster.label,
                "type": "rq_prefix",
                "name": cluster.label,
                "category": cluster.node_type or ("rq_prefix" if cluster.rq_level is not None else "rq_prefix"),
                "snippet": f"{len(cluster.support_chunk_ids_json or [])} support chunks",
                "metadata": {
                    "rq_prefix_key": cluster.rq_prefix_key,
                    "rq_level": cluster.rq_level,
                    "rq_path_prefix": cluster.rq_path_prefix or [],
                    "representative_chunk_ids": cluster.representative_chunk_ids_json or [],
                    "support_chunk_ids": cluster.support_chunk_ids_json or [],
                    "bridge_chunk_ids": cluster.bridge_chunk_ids_json or [],
                    "stats": cluster.stats_json or {},
                    "diagnostics": cluster.diagnostics_json or {},
                },
            }
            for cluster in cluster_rows
        ]
        nodes = chunk_nodes + cluster_nodes
        chunk_edge_rows = (
            list(
                db.scalars(
                    select(ChunkRelationEdge)
                    .where(ChunkRelationEdge.graph_state_id == relation_state.id)
                    .limit(limit * 2)
                ).all()
            )
            if relation_state
            else []
        )
        chunk_relation_edges = [
            {
                "id": edge.id,
                "source": edge.source_chunk_id,
                "target": edge.target_chunk_id,
                "label": edge.edge_type,
                "type": edge.edge_type,
                "weight": edge.weight,
                "distance": edge.distance,
                "raw_strength": edge.raw_strength,
                "category": edge.edge_type,
                "is_bridge": edge.is_bridge,
                "metadata": {
                    **(edge.features_json or {}),
                    "distance": edge.distance,
                    "raw_strength": edge.raw_strength,
                    "raw_strength_summary": edge.raw_strength_summary_json or {},
                    "source_algorithm": edge.source_algorithm,
                    "protocol_version": edge.protocol_version,
                    "graph_state_hash": edge.graph_state_hash,
                    "diagnostics": edge.diagnostics_json or {},
                },
            }
            for edge in chunk_edge_rows
        ]
        membership_edges = [
            {
                "id": f"membership:{row.id}",
                "source": row.chunk_id,
                "target": row.rq_prefix_id,
                "label": row.membership_reason,
                "type": "rq_prefix_membership",
                "weight": row.membership_score,
                "category": "rq_membership" if row.rq_path else "rq_prefix_membership",
                "metadata": {
                    "membership_role": row.membership_role,
                    "support_chunk_edge_ids": row.support_chunk_edge_ids_json or [],
                    "rq_path": row.rq_path or [],
                    "residual_norm": row.residual_norm,
                    "diagnostics": row.diagnostics_json or {},
                },
            }
            for row in membership_edge_rows
        ]
        edges = chunk_relation_edges + membership_edges
    elif layer == "mid-concepts":
        nodes = [
            {
                "id": concept.id,
                "label": concept.canonical_label,
                "type": "mid_concept",
                "name": concept.canonical_label,
                "category": "mid_concept",
                "summary": concept.definition,
                "support_active_chunk_ids": concept.support_chunk_ids_json,
                "confidence": concept.confidence,
            }
            for concept in db.scalars(
                select(MidConcept)
                .where(MidConcept.concept_state_id == (mid_state.id if mid_state else None), MidConcept.state == "active")
                .limit(limit)
            ).all()
        ]
        edges = []
        if mid_state:
            edges = [
                {
                    "source": edge.source_concept_id,
                    "target": edge.target_concept_id,
                    "label": edge.edge_type,
                    "weight": edge.weight,
                    "distance": edge.distance,
                    "category": edge.edge_type,
                    "metadata": {
                        "raw_strength_summary": edge.raw_strength_summary_json or {},
                        "support_rq_prefix_ids": edge.support_rq_prefix_ids_json or [],
                        "support_chunk_ids": edge.support_chunk_ids_json or [],
                        "support_chunk_edge_ids": edge.support_chunk_edge_ids_json or edge.support_relation_edge_ids_json or [],
                        "diagnostics": edge.diagnostics_json or {},
                    },
                }
                | {"id": edge.id, "type": edge.edge_type}
                for edge in db.scalars(select(MidConceptEdge).where(MidConceptEdge.concept_state_id == mid_state.id).limit(limit * 2)).all()
            ]
    elif layer == "coarse-concepts":
        nodes = [
            {
                "id": concept.id,
                "label": concept.canonical_label,
                "type": "coarse_concept",
                "name": concept.canonical_label,
                "category": "coarse_concept",
                "summary": concept.definition,
                "support_active_chunk_ids": concept.included_mid_concept_ids_json,
                "confidence": concept.confidence,
            }
            for concept in db.scalars(
                select(CoarseConcept)
                .where(CoarseConcept.coarse_state_id == (coarse_state.id if coarse_state else None), CoarseConcept.state == "active")
                .limit(limit)
            ).all()
        ]
        edges = []
        if coarse_state:
            edges = [
                {
                    "source": edge.source_concept_id,
                    "target": edge.target_concept_id,
                    "label": edge.edge_type,
                    "weight": edge.weight,
                    "distance": edge.distance,
                    "category": edge.edge_type,
                    "metadata": {
                        "raw_strength_summary": edge.raw_strength_summary_json or {},
                        "support_mid_edge_ids": edge.support_mid_edge_ids_json or [],
                        "support_chunk_ids": edge.support_chunk_ids_json or [],
                        "support_chunk_edge_ids": edge.support_chunk_edge_ids_json or [],
                        "cross_community_weak_ties": edge.cross_community_weak_ties_json or [],
                        "diagnostics": edge.diagnostics_json or {},
                    },
                }
                | {"id": edge.id, "type": edge.edge_type}
                for edge in db.scalars(select(CoarseConceptEdge).where(CoarseConceptEdge.coarse_state_id == coarse_state.id).limit(limit * 2)).all()
            ]
    else:
        raise ValueError(f"Unknown graph layer: {layer}")
    return {
        "knowledge_base_id": knowledge_base_id,
        "graph_type": layer,
        "schema_version": "context_graph_v1",
        "view": "overview",
        "nodes": nodes,
        "edges": edges,
        "counts": stats["counts"],
        "full_counts": full_counts,
        "sampled_counts": {"nodes": len(nodes), "edges": len(edges)},
        "node_counts": {"sampled": len(nodes), "full": full_counts.get("nodes", 0)},
        "edge_counts": {"sampled": len(edges), "full": full_counts.get("edges", 0)},
        "freshness": stats["freshness"],
        "hash": stats["hashes"].get("context_graph_hash"),
        "stale_reason": ", ".join(stats["freshness"].get("stale_reasons") or []) or None,
        "grounding": grounding_stats(db, knowledge_base_id),
        "retrieval_contribution": retrieval_contribution_stats(db, knowledge_base_id),
        "diagnostics": {
            "full_counts": stats["counts"],
            "layer_full_counts": full_counts,
            "sampled_counts": {"nodes": len(nodes), "edges": len(edges)},
            "active_state_ids": stats.get("active_state_ids", {}),
            "hashes": stats["hashes"],
            "grounding": grounding_stats(db, knowledge_base_id),
            "retrieval_contribution": retrieval_contribution_stats(db, knowledge_base_id),
            **stats.get("diagnostics", {}),
        },
    }


def grounding_stats(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    mid_total = db.scalar(select(func.count(MidConcept.id)).where(MidConcept.knowledge_base_id == knowledge_base_id, MidConcept.state == "active")) or 0
    grounded_mid = (
        db.scalar(
            select(func.count(MidConcept.id)).where(
                MidConcept.knowledge_base_id == knowledge_base_id,
                MidConcept.state == "active",
                func.coalesce(func.json_array_length(MidConcept.support_chunk_ids_json), 0) > 0,
            )
        )
        or 0
    )
    coarse_total = db.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.knowledge_base_id == knowledge_base_id, CoarseConcept.state == "active")) or 0
    grounded_coarse = (
        db.scalar(
            select(func.count(CoarseConcept.id)).where(
                CoarseConcept.knowledge_base_id == knowledge_base_id,
                CoarseConcept.state == "active",
                func.coalesce(func.json_array_length(CoarseConcept.included_mid_concept_ids_json), 0) > 0,
            )
        )
        or 0
    )
    return {
        "mid_grounded_rate": round(grounded_mid / max(mid_total, 1), 6),
        "mid_total": mid_total,
        "coarse_grounded_rate": round(grounded_coarse / max(coarse_total, 1), 6),
        "coarse_total": coarse_total,
    }


def retrieval_contribution_stats(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    traces = list(db.scalars(select(RetrievalTrace).where(RetrievalTrace.knowledge_base_id == knowledge_base_id).order_by(RetrievalTrace.created_at.desc()).limit(50)).all())
    edge_role_totals: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    frontier_pops = 0
    dominance_pruned = 0
    for trace in traces:
        frontier_pops += len(trace.frontier_json or [])
        convergence = trace.convergence_json or {}
        stop_reason = str(convergence.get("reason") or "unknown")
        stop_reasons[stop_reason] += 1
        dominance_pruned += int(convergence.get("dominance_pruned_count") or 0)
        for label in trace.path_labels_json or []:
            for edge_id in label.get("expanded_edge_ids") or []:
                edge_role_totals[str(edge_id)] += 1
    total_edges = sum(edge_role_totals.values()) or 1
    return {
        "frontier_pops": frontier_pops,
        "dominance_pruned_count": dominance_pruned,
        "expanded_edge_contribution": {key: round(value / total_edges, 6) for key, value in edge_role_totals.most_common(20)},
        "convergence_reasons": dict(stop_reasons),
        "scores_json_primary": False,
    }
