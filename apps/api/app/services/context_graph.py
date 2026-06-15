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

from rank_bm25 import BM25Okapi
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    BM25Record,
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
    FineCluster,
    FineClusterEdge,
    FineClusterMembership,
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
from app.schemas import SearchFilters
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


RELATION_PROTOCOL_VERSION = "chunk_relation_graph_rq_v2"
MID_CONCEPT_PROMPT_VERSION = "mid_concept_definition_v1"
COARSE_CONCEPT_PROMPT_VERSION = "coarse_concept_definition_v1"
CONTEXT_GRAPH_PROTOCOL_VERSION = "context_graph_v1"
ANSWER_PROMPT_PROTOCOL_VERSION = "context_graph_answer_v1"
EDGE_DISTANCE_PROTOCOL_VERSION = "edge_distance_log_raw_strength_v1"
EDGE_PROJECTION_PROTOCOL_VERSION = "edge_projection_support_ids_v1"
TRAVERSAL_PROTOCOL_VERSION = "priority_queue_layered_traversal_v1"


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
        "coarse_entry_budget": int(settings.agent_coarse_entry_budget),
        "coarse_jump_budget": int(settings.agent_coarse_jump_budget),
        "mid_entry_budget": int(settings.agent_mid_entry_budget),
        "mid_expansion_radius_cap": int(settings.agent_mid_expansion_radius_cap),
        "fine_entry_budget": int(settings.agent_fine_entry_budget),
        "frontier_expansion_budget": int(settings.agent_frontier_expansion_budget),
        "max_depth_per_layer": int(settings.agent_max_depth_per_layer),
        "max_labels_per_node": int(settings.agent_max_labels_per_node),
        "max_edge_reuse": int(settings.agent_max_edge_reuse),
        "max_cycle_reward_per_path": float(settings.agent_max_cycle_reward_per_path),
        "ambiguous_edge_distance_low": float(settings.agent_ambiguous_edge_distance_low),
        "ambiguous_edge_distance_high": float(settings.agent_ambiguous_edge_distance_high),
        "drilldown_budget_per_layer": int(settings.agent_drilldown_budget_per_layer),
        "chunk_candidate_budget": int(settings.agent_chunk_candidate_budget),
        "structure_restore_budget": int(settings.agent_structure_restore_budget),
        "context_package_token_budget": int(settings.context_package_token_budget),
        "context_path_summary_budget": int(settings.context_path_summary_budget),
        "planning_round_budget": int(settings.agent_planning_round_budget),
        "max_typed_actions_per_round": int(settings.agent_max_typed_actions_per_round),
        "repair_round_budget": int(settings.agent_repair_round_budget),
        "verification_budget": int(settings.agent_verification_budget),
        "allowed_relation_types": [
            "dense_knn",
            "bm25_overlap",
            "structure_adjacent",
        "same_section",
        "same_page_region",
        "same_table_formula_context",
        "co_retrieved",
        "fine_cluster_bridge",
        "centroid_near",
            "rq_hierarchy_near",
            "rq_prefix_sibling",
            "rq_residual_near",
        ],
        "required_restore_modes": ["previous_next", "parent_structure", "bridge_chunks"],
    }


def agent_operating_envelope_state_hash() -> str:
    return stable_hash(agent_operating_envelope())


def qdrant_collection_name(*, embedding_model: str, embedding_text_version: str, chunk_schema_version: str) -> str:
    raw = f"symbograph_{embedding_model}_{embedding_text_version}_{chunk_schema_version}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_").lower()[:180]


def tokenize_for_bm25(text: str) -> list[str]:
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
    return round(-math.log(max(1e-6, normalized_strength(raw_strength))), 6)


def edge_distance_protocol_hash() -> str:
    return stable_hash({"protocol": EDGE_DISTANCE_PROTOCOL_VERSION, "formula": "distance=-log(max(epsilon,raw_strength))"})


def edge_projection_protocol_hash() -> str:
    return stable_hash({"protocol": EDGE_PROJECTION_PROTOCOL_VERSION, "support": ["chunk_edges", "fine_edges", "mid_edges"]})


def traversal_protocol_hash() -> str:
    return stable_hash({"protocol": TRAVERSAL_PROTOCOL_VERSION, "queue_key": ["uncovered_facets", "distance_minus_reward", "depth", "negative_evidence_roles"]})


def context_graph_cache_key_components(
    *,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters | dict[str, Any] | None,
    context_state: ContextGraphState | None,
    retrieval_mode: str,
    conversation_state_scope_hash: str | None = None,
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
        "fine_cluster_hash": context_state.fine_cluster_hash if context_state else None,
        "mid_concept_hash": context_state.mid_concept_hash if context_state else None,
        "coarse_concept_hash": context_state.coarse_concept_hash if context_state else None,
        "edge_distance_protocol_hash": edge_distance_protocol_hash(),
        "edge_projection_protocol_hash": edge_projection_protocol_hash(),
        "traversal_protocol_hash": traversal_protocol_hash(),
        "runtime_settings_hash": runtime_settings_state_hash(),
        "policy_state_hash": context_state.policy_state_hash if context_state else None,
        "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
        "conversation_state_scope_hash": conversation_state_scope_hash or stable_hash({"conversation_state": "none"}),
        "prompt_protocol_hash": context_state.prompt_protocol_hash if context_state else None,
        "retrieval_mode": retrieval_mode,
    }


def context_graph_cache_key(
    *,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters | dict[str, Any] | None,
    context_state: ContextGraphState | None,
    retrieval_mode: str,
    conversation_state_scope_hash: str | None = None,
) -> str:
    return stable_hash(
        context_graph_cache_key_components(
            knowledge_base_id=knowledge_base_id,
            query=query,
            filters=filters,
            context_state=context_state,
            retrieval_mode=retrieval_mode,
            conversation_state_scope_hash=conversation_state_scope_hash,
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
        return {"chunks": 0, "vectors": 0, "bm25_records": 0}
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

    for chunk, context_text in zip(chunks, context_texts):
        terms = tokenize_for_bm25(context_text)
        frequencies = Counter(terms)
        db.add(
            BM25Record(
                knowledge_base_id=knowledge_base.id,
                chunk_id=chunk.id,
                embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
                text_hash=contextual_text_hash(context_text),
                token_count=len(terms),
                term_frequencies_json=dict(frequencies),
                document_length=len(terms),
                state="ready",
            )
        )
    db.flush()
    return {"chunks": len(chunks), "vectors": len(points), "bm25_records": len(chunks), "collection_name": collection_name}


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


async def rebuild_context_graph(db: Session, knowledge_base_id: str, *, batch_id: str | None = None) -> ContextGraphState:
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    if not chunks:
        raise RuntimeError("Cannot rebuild context graph without current chunks")
    context_graph_batch_heartbeat(batch_id, "starting", {"chunks": len(chunks)})
    deactivate_derived_states(db, knowledge_base_id)
    context_graph_batch_heartbeat(batch_id, "chunk_relation", {"chunks": len(chunks)})
    relation_state = build_chunk_relation_graph(db, knowledge_base_id, chunks, batch_id=batch_id)
    context_graph_batch_heartbeat(batch_id, "mid_concepts", dict(relation_state.stats_json or {}))
    mid_state = await build_mid_concept_graph(db, knowledge_base_id, relation_state, batch_id=batch_id)
    context_graph_batch_heartbeat(batch_id, "coarse_concepts", dict(mid_state.diagnostics_json or {}))
    coarse_state = await build_coarse_concept_graph(db, knowledge_base_id, mid_state)
    context_graph_batch_heartbeat(batch_id, "context_state", dict(coarse_state.diagnostics_json or {}))
    context_state = write_context_graph_state(db, knowledge_base_id, relation_state, mid_state, coarse_state, chunks)
    db.flush()
    context_graph_batch_heartbeat(batch_id, "completed", dict(context_state.stats_json or {}))
    return context_state


def context_graph_batch_heartbeat(batch_id: str | None, phase: str, metrics: dict[str, Any] | None = None) -> None:
    if not batch_id:
        return
    from app.db import SessionLocal
    from app.models import IngestionBatch

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
            stats["context_graph_metrics"] = metrics
        batch.stats = stats
        batch.heartbeat_at = now
        session.commit()


def build_chunk_relation_graph(db: Session, knowledge_base_id: str, chunks: list[Chunk], *, batch_id: str | None = None) -> ChunkRelationGraphState:
    scope_hash = compute_chunk_scope_hash(chunks)
    vectors = {chunk.id: vector_for_chunk(db, chunk.id) for chunk in chunks}
    graph_state = ChunkRelationGraphState(
        knowledge_base_id=knowledge_base_id,
        chunk_version=max(chunk.chunk_version for chunk in chunks),
        scope_hash=scope_hash,
        state_hash=stable_hash({"scope": scope_hash, "protocol": RELATION_PROTOCOL_VERSION}),
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        relation_protocol_version=RELATION_PROTOCOL_VERSION,
        active_chunk_ids_json=[chunk.id for chunk in chunks],
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db.add(graph_state)
    db.flush()

    edges: dict[tuple[str, str, str], ChunkRelationEdge] = {}
    ensure_not_cancelled(db, batch_id)
    context_graph_batch_heartbeat(batch_id, "chunk_relation:chunk_edges", {"chunks": len(chunks)})
    add_relation_edges(db, graph_state, chunks, vectors, edges)
    ensure_not_cancelled(db, batch_id)
    context_graph_batch_heartbeat(batch_id, "chunk_relation:fine_clusters", {"chunk_edges": len(edges)})
    clusters = build_fine_clusters(db, graph_state, chunks, vectors, edges)
    ensure_not_cancelled(db, batch_id)
    context_graph_batch_heartbeat(batch_id, "chunk_relation:fine_edges", {"fine_clusters": len(clusters), "chunk_edges": len(edges)})
    build_fine_cluster_edges(db, graph_state, clusters, edges, batch_id=batch_id)
    ensure_not_cancelled(db, batch_id)
    stats = relation_graph_stats(chunks, list(edges.values()), clusters)
    graph_state.stats_json = stats
    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        "orphan_chunk_rate": stats["orphan_chunk_rate"],
        "singleton_rate": stats["singleton_rate"],
        "bridge_edge_count": stats["bridge_edges"],
        "protocol": RELATION_PROTOCOL_VERSION,
    }
    graph_state.state_hash = stable_hash({"scope": scope_hash, "stats": stats, "clusters": [cluster.id for cluster in clusters]})
    for edge in edges.values():
        edge.graph_state_hash = graph_state.state_hash
    return graph_state


def relation_edge_source_algorithm(edge_type: str) -> str:
    if edge_type.startswith("rq_"):
        return "rq_kmeans"
    if edge_type.startswith("bm25"):
        return "bm25"
    if edge_type.startswith("dense"):
        return "embedding_knn"
    if edge_type.startswith("structure") or edge_type.startswith("same_"):
        return "structure_graph"
    if edge_type.startswith("fine") or edge_type.startswith("centroid"):
        return "fine_clustering"
    return edge_type


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
        edge.raw_strength_summary_json = {
            "max_raw_strength": edge.raw_strength,
            "distance": edge.distance,
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
        }
        edge.protocol_version = graph_state.relation_protocol_version
        edge.source_algorithm = relation_edge_source_algorithm(edge_type)
        edge.graph_state_hash = graph_state.state_hash
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
        confidence=raw_strength,
        features_json=features or {},
        support_json={"source": edge_type},
        source_algorithm=relation_edge_source_algorithm(edge_type),
        protocol_version=graph_state.relation_protocol_version,
        graph_state_hash=graph_state.state_hash,
        is_bridge=is_bridge,
        diagnostics_json={
            "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
            "weight_is_compatibility_copy": True,
        },
    )
    db.add(edge)
    edges[key] = edge
    return edge


def add_relation_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> None:
    def add(source: Chunk, target: Chunk, edge_type: str, weight: float, features: dict | None = None, *, is_bridge: bool = False) -> None:
        add_chunk_relation_edge(db, graph_state, source.id, target.id, edge_type, weight, features, edges, is_bridge=is_bridge)

    for chunk in chunks:
        if chunk.previous_chunk_id:
            target = next((item for item in chunks if item.id == chunk.previous_chunk_id), None)
            if target:
                add(chunk, target, "structure_adjacent", 1.0)
        if chunk.next_chunk_id:
            target = next((item for item in chunks if item.id == chunk.next_chunk_id), None)
            if target:
                add(chunk, target, "structure_adjacent", 1.0)
    by_section: dict[str, list[Chunk]] = defaultdict(list)
    by_page: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.section_path:
            by_section[chunk.section_path].append(chunk)
        if chunk.page_start is not None:
            by_page[f"{chunk.document_id}:{chunk.page_start}"].append(chunk)
    for group in by_section.values():
        for left, right in _sliding_pairs(group, max_pairs=6):
            add(left, right, "same_section", 0.72)
    for group in by_page.values():
        for left, right in _sliding_pairs(group, max_pairs=8):
            add(left, right, "same_page_region", 0.68)
    add_same_table_formula_context_edges(db, graph_state, chunks, edges)

    bm25_records = {record.chunk_id: record for record in db.scalars(select(BM25Record).where(BM25Record.knowledge_base_id == graph_state.knowledge_base_id)).all()}
    for i, left in enumerate(chunks):
        dense_scores: list[tuple[float, Chunk]] = []
        lexical_scores: list[tuple[float, Chunk]] = []
        dense_by_chunk: dict[str, float] = {}
        lexical_by_chunk: dict[str, float] = {}
        left_vector = vectors.get(left.id) or []
        left_terms = set((bm25_records.get(left.id).term_frequencies_json or {}).keys()) if bm25_records.get(left.id) else set()
        for right in chunks[i + 1 :]:
            right_vector = vectors.get(right.id) or []
            if left_vector and right_vector:
                dense = cosine_similarity(left_vector, right_vector)
                dense_scores.append((dense, right))
                dense_by_chunk[right.id] = dense
            right_terms = set((bm25_records.get(right.id).term_frequencies_json or {}).keys()) if bm25_records.get(right.id) else set()
            if left_terms and right_terms:
                overlap = len(left_terms.intersection(right_terms)) / max(len(left_terms.union(right_terms)), 1)
                lexical_scores.append((overlap, right))
                lexical_by_chunk[right.id] = overlap
        for score, right in sorted(dense_scores, key=lambda item: item[0], reverse=True)[:5]:
            if score > 0.3:
                add(left, right, "dense_knn", score, {"cosine": round(score, 6)}, is_bridge=(left.section_path != right.section_path and score > 0.52))
        for score, right in sorted(lexical_scores, key=lambda item: item[0], reverse=True)[:5]:
            if score > 0.08:
                add(left, right, "bm25_overlap", min(1.0, score * 3.0), {"term_jaccard": round(score, 6)}, is_bridge=(left.section_path != right.section_path and score > 0.18))
        co_retrieved_candidates: list[tuple[float, Chunk, float, float]] = []
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        for right_id in set(dense_by_chunk).intersection(lexical_by_chunk):
            dense = dense_by_chunk[right_id]
            lexical = lexical_by_chunk[right_id]
            if dense <= 0.24 or lexical <= 0.05:
                continue
            score = normalized_strength(0.55 * max(0.0, dense) + 0.45 * min(1.0, lexical * 3.0))
            co_retrieved_candidates.append((score, chunk_by_id[right_id], dense, lexical))
        for score, right, dense, lexical in sorted(co_retrieved_candidates, key=lambda item: item[0], reverse=True)[:3]:
            add(
                left,
                right,
                "co_retrieved",
                score,
                {
                    "co_activation_source": "dense_and_bm25_candidate_overlap",
                    "dense_cosine": round(dense, 6),
                    "term_jaccard": round(lexical, 6),
                },
                is_bridge=(left.section_path != right.section_path and score > 0.35),
            )
    db.flush()
    edge_types = {edge.edge_type for edge in edges.values()}
    missing_reasons = {}
    if "co_retrieved" not in edge_types:
        missing_reasons["co_retrieved"] = "No dense+BM25 co-activated chunk pair passed the protocol thresholds."
    if "same_table_formula_context" not in edge_types:
        missing_reasons["same_table_formula_context"] = "No shared table/formula/caption/code structure closure spanned multiple chunks."
    if missing_reasons:
        graph_state.diagnostics_json = {
            **(graph_state.diagnostics_json or {}),
            "missing_target_relation_edge_type_reasons": missing_reasons,
        }


def add_same_table_formula_context_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> None:
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    chunk_ids = set(chunk_by_id)
    rows = db.execute(
        select(ChunkStructureMapping, ChunkStructureNode)
        .join(ChunkStructureNode, ChunkStructureMapping.structure_node_id == ChunkStructureNode.id)
        .where(
            ChunkStructureMapping.chunk_id.in_(chunk_ids),
            ChunkStructureNode.node_type.in_(["table", "formula", "caption", "code_block"]),
        )
        .order_by(ChunkStructureNode.node_type.asc(), ChunkStructureMapping.coverage_ratio.desc())
    ).all()
    chunks_by_structure_node: dict[str, dict[str, Any]] = defaultdict(lambda: {"node": None, "chunk_ids": []})
    for mapping, node in rows:
        chunks_by_structure_node[node.id]["node"] = node
        if mapping.chunk_id not in chunks_by_structure_node[node.id]["chunk_ids"]:
            chunks_by_structure_node[node.id]["chunk_ids"].append(mapping.chunk_id)

    def add_edge_for_pair(left: Chunk, right: Chunk, strength: float, features: dict[str, Any]) -> None:
        add_chunk_relation_edge(
            db,
            graph_state,
            left.id,
            right.id,
            "same_table_formula_context",
            strength,
            features,
            edges,
            is_bridge=left.section_path != right.section_path,
        )

    for structure_node_id, payload in chunks_by_structure_node.items():
        node = payload.get("node")
        mapped_chunks = [chunk_by_id[chunk_id] for chunk_id in payload.get("chunk_ids", []) if chunk_id in chunk_by_id]
        if len(mapped_chunks) < 2:
            continue
        for left, right in _sliding_pairs(mapped_chunks, max_pairs=10):
            add_edge_for_pair(
                left,
                right,
                0.86,
                {
                    "structure_node_id": structure_node_id,
                    "structure_node_type": getattr(node, "node_type", None),
                    "closure": "shared_structure_node",
                },
            )

    flagged_by_section: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        metadata = chunk.metadata_json or {}
        if metadata.get("has_table") or metadata.get("has_formula") or metadata.get("has_caption") or metadata.get("content_kind") in {"table", "formula", "code"}:
            flagged_by_section[f"{chunk.document_id}:{chunk.section_path or ''}"].append(chunk)
    for group in flagged_by_section.values():
        if len(group) < 2:
            continue
        for left, right in _sliding_pairs(group, max_pairs=8):
            add_edge_for_pair(
                left,
                right,
                0.74,
                {
                    "closure": "chunk_metadata_table_formula_context",
                    "left_flags": chunk_relation_content_flags(left),
                    "right_flags": chunk_relation_content_flags(right),
                },
            )


def chunk_relation_content_flags(chunk: Chunk) -> list[str]:
    metadata = chunk.metadata_json or {}
    flags = []
    for key in ("has_table", "has_formula", "has_caption"):
        if metadata.get(key):
            flags.append(key)
    content_kind = metadata.get("content_kind")
    if content_kind and content_kind != "text":
        flags.append(f"content_kind:{content_kind}")
    return flags


def _sliding_pairs(chunks: list[Chunk], *, max_pairs: int) -> list[tuple[Chunk, Chunk]]:
    pairs: list[tuple[Chunk, Chunk]] = []
    ordered = sorted(chunks, key=lambda item: item.chunk_index)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 : index + 3]:
            pairs.append((left, right))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


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
    if direct:
        return list(dict.fromkeys(direct))
    return list(dict.fromkeys(support_chunk_edge_ids_for_chunks(left_chunk_ids | right_chunk_ids, edges)))


def fine_community_groups(chunks: list[Chunk], edges: dict[tuple[str, str, str], ChunkRelationEdge]) -> list[tuple[str, list[Chunk], dict[str, Any]]]:
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


def build_fine_clusters(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> list[FineCluster]:
    edge_bridge_chunks = {edge.source_chunk_id for edge in edges.values() if edge.is_bridge} | {edge.target_chunk_id for edge in edges.values() if edge.is_bridge}
    support_edges_by_chunk: dict[str, list[str]] = defaultdict(list)
    for edge in edges.values():
        support_edges_by_chunk[edge.source_chunk_id].append(edge.id)
        support_edges_by_chunk[edge.target_chunk_id].append(edge.id)
    clusters: list[FineCluster] = []
    community_groups = fine_community_groups(chunks, edges)
    graph_state.diagnostics_json = {
        **(graph_state.diagnostics_json or {}),
        "fine_seed_community_detection": {
            "algorithm": community_groups[0][2]["community_detection"] if community_groups else "none",
            "community_count": len(community_groups),
            "label_terms_are_diagnostics_only": True,
        },
    }
    for label, members, community_diagnostics in community_groups:
        member_vectors = [vectors.get(chunk.id) for chunk in members if vectors.get(chunk.id)]
        centroid = _centroid([vector for vector in member_vectors if vector])
        representatives = sorted(members, key=lambda item: len(item.text), reverse=True)[:3]
        cluster = FineCluster(
            graph_state_id=graph_state.id,
            knowledge_base_id=graph_state.knowledge_base_id,
            cluster_key=stable_hash({"label": label, "chunks": [chunk.id for chunk in members]})[:24],
            label=label,
            node_type="fine_seed",
            centroid_json=centroid,
            representative_chunk_ids_json=[chunk.id for chunk in representatives],
            support_chunk_ids_json=[chunk.id for chunk in members],
            bridge_chunk_ids_json=[chunk.id for chunk in members if chunk.id in edge_bridge_chunks],
            stats_json={"member_count": len(members), **community_diagnostics},
            diagnostics_json=community_diagnostics,
        )
        db.add(cluster)
        db.flush()
        clusters.append(cluster)
        for chunk in members:
            own_score = 1.0
            db.add(
                FineClusterMembership(
                    fine_cluster_id=cluster.id,
                    chunk_id=chunk.id,
                    membership_score=own_score,
                    membership_role="seed_member",
                    membership_reason="graph_community_seed",
                    support_chunk_edge_ids_json=list(dict.fromkeys(support_edges_by_chunk.get(chunk.id, [])))[:16],
                )
            )
        for chunk in members:
            if chunk.id in edge_bridge_chunks:
                for other in clusters:
                    if other.id == cluster.id:
                        continue
                    db.add(
                        FineClusterMembership(
                            fine_cluster_id=other.id,
                            chunk_id=chunk.id,
                            membership_score=0.35,
                            membership_role="bridge_member",
                            membership_reason="bridge_fuzzy_membership",
                            support_chunk_edge_ids_json=list(dict.fromkeys(support_edges_by_chunk.get(chunk.id, [])))[:16],
                            diagnostics_json={"source_cluster_id": cluster.id},
                        )
                    )
                    break
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
    clusters: list[FineCluster],
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

    chunk_by_id = {chunk.id: chunk for chunk, _vector in normalized_items}
    base_clusters = list(clusters)
    rq_clusters_by_key: dict[tuple[int, tuple[int, ...]], FineCluster] = {}
    prefix_groups: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for chunk_id, encoded in assignments.items():
        path = encoded["rq_path"]
        for level in range(1, len(path) + 1):
            prefix_groups[(level, tuple(path[:level]))].append(chunk_id)

    for (level, prefix), member_ids in sorted(prefix_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        prefix_vector = rq_prefix_vector(rq_model, list(prefix))
        representatives = sorted(
            member_ids,
            key=lambda chunk_id: _sq_distance(_fit_width(vectors.get(chunk_id) or [], len(prefix_vector)), prefix_vector),
        )[:3]
        residual_norms = [float(assignments[chunk_id]["residual_norm"]) for chunk_id in member_ids]
        residual_mean = _centroid([assignments[chunk_id]["residual_vector"] for chunk_id in member_ids])
        cluster = FineCluster(
            graph_state_id=graph_state.id,
            knowledge_base_id=graph_state.knowledge_base_id,
            cluster_key=f"rq:L{level}:{'-'.join(str(item) for item in prefix)}",
            label=f"RQ L{level} {'/'.join(str(item) for item in prefix)}",
            node_type="rq_prefix",
            centroid_json=prefix_vector,
            rq_level=level,
            rq_path_prefix=list(prefix),
            centroid_vector_ref=f"rq:{graph_state.id}:L{level}:{'-'.join(str(item) for item in prefix)}",
            representative_chunk_ids_json=representatives,
            support_chunk_ids_json=member_ids,
            bridge_chunk_ids_json=[chunk_id for chunk_id in member_ids if chunk_id in bridge_chunk_ids],
            stats_json={
                "member_count": len(member_ids),
                "residual_norm_mean": round(sum(residual_norms) / max(len(residual_norms), 1), 6),
                "residual_norm_max": round(max(residual_norms or [0.0]), 6),
            },
            diagnostics_json={"source": "rq_kmeans", "residual_mean_vector": residual_mean},
        )
        db.add(cluster)
        db.flush()
        clusters.append(cluster)
        rq_clusters_by_key[(level, tuple(prefix))] = cluster
        for chunk_id in member_ids:
            encoded = assignments[chunk_id]
            db.add(
                FineClusterMembership(
                    fine_cluster_id=cluster.id,
                    chunk_id=chunk_id,
                    membership_score=rq_membership_score(float(encoded["residual_norm"]), tau_r=float(rq_config["tau_r"])),
                    membership_role="rq_prefix_member",
                    membership_reason="rq_prefix" if level < len(encoded["rq_path"]) else "rq_leaf",
                    rq_path=encoded["rq_path"],
                    residual_norm=float(encoded["residual_norm"]),
                    support_chunk_edge_ids_json=support_chunk_edge_ids_for_chunks({chunk_id}, edges)[:16],
                    diagnostics_json={
                        "residual_vector": encoded["residual_vector"],
                        "reconstructed_vector": encoded["reconstructed_vector"],
                        "rq_level": level,
                        "rq_path_prefix": list(prefix),
                    },
                )
            )

    add_rq_relation_edges(db, graph_state, chunk_by_id, assignments, edges)
    db.flush()
    add_rq_cluster_edges(db, graph_state, rq_clusters_by_key, base_clusters, bridge_chunk_ids, edges)


def add_rq_cluster_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    rq_clusters_by_key: dict[tuple[int, tuple[int, ...]], FineCluster],
    base_clusters: list[FineCluster],
    bridge_chunk_ids: set[str],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> None:
    seen: set[tuple[str, str, str]] = set()
    added_types: set[str] = set()

    def representative_support_chunks(source: FineCluster, target: FineCluster, explicit_support_ids: list[str] | None = None) -> list[str]:
        support_ids = list(dict.fromkeys(explicit_support_ids or []))
        if support_ids:
            return support_ids[:16]
        source_ids = list(source.support_chunk_ids_json or [])
        target_ids = list(target.support_chunk_ids_json or [])
        representatives = []
        if source_ids:
            representatives.append(source_ids[0])
        if target_ids:
            representatives.append(target_ids[0])
        for chunk_id in source_ids[1:4] + target_ids[1:4]:
            if chunk_id not in representatives:
                representatives.append(chunk_id)
        return representatives[:16]

    def ensure_support_chunk_edge_ids(
        source: FineCluster,
        target: FineCluster,
        edge_type: str,
        raw_strength: float,
        diagnostics: dict[str, Any],
        support_chunk_ids: list[str],
    ) -> list[str]:
        existing = support_chunk_edge_ids_for_chunks(support_chunk_ids, edges)[:32]
        if existing:
            return existing
        left_candidates = [chunk_id for chunk_id in (source.support_chunk_ids_json or []) if chunk_id in support_chunk_ids] or list(source.support_chunk_ids_json or [])
        right_candidates = [chunk_id for chunk_id in (target.support_chunk_ids_json or []) if chunk_id in support_chunk_ids] or list(target.support_chunk_ids_json or [])
        if not left_candidates or not right_candidates:
            return []
        left_chunk_id = left_candidates[0]
        right_chunk_id = next((chunk_id for chunk_id in right_candidates if chunk_id != left_chunk_id), right_candidates[0])
        if left_chunk_id == right_chunk_id:
            return []
        chunk_edge_type = "rq_prefix_sibling" if edge_type == "rq_sibling" else "rq_residual_near" if edge_type == "rq_centroid_near" else "rq_hierarchy_near"
        edge = add_chunk_relation_edge(
            db,
            graph_state,
            left_chunk_id,
            right_chunk_id,
            chunk_edge_type,
            max(0.1, raw_strength * 0.75),
            {
                "materialized_for_fine_edge_support": True,
                "fine_edge_type": edge_type,
                "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                **diagnostics,
            },
            edges,
            is_bridge=True,
        )
        db.flush()
        return [edge.id] if edge is not None and edge.id else []

    def add_edge(source: FineCluster, target: FineCluster, edge_type: str, weight: float, support_ids: list[str], diagnostics: dict[str, Any]) -> None:
        if source.id == target.id:
            return
        left, right = (source, target) if source.id <= target.id else (target, source)
        key = (left.id, right.id, edge_type)
        if key in seen:
            return
        seen.add(key)
        raw_strength = normalized_strength(weight)
        support_chunk_ids = representative_support_chunks(source, target, support_ids)
        support_chunk_edge_ids = ensure_support_chunk_edge_ids(source, target, edge_type, raw_strength, diagnostics, support_chunk_ids)
        if not support_chunk_edge_ids:
            return
        added_types.add(edge_type)
        db.add(
            FineClusterEdge(
                graph_state_id=graph_state.id,
                source_cluster_id=left.id,
                target_cluster_id=right.id,
                edge_type=edge_type,
                weight=raw_strength,
                distance=distance_from_strength(raw_strength),
                raw_strength=raw_strength,
                raw_strength_summary_json={
                    "max_raw_strength": raw_strength,
                    "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
                },
                features_json=diagnostics,
                source_algorithm="rq_kmeans",
                protocol_version=graph_state.relation_protocol_version,
                support_chunk_ids_json=support_chunk_ids,
                support_chunk_edge_ids_json=support_chunk_edge_ids,
                diagnostics_json={
                    "source_algorithm": "rq_kmeans",
                    "protocol_version": graph_state.relation_protocol_version,
                    "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                    "support_materialized": bool((diagnostics or {}).get("fallback_pair")) or bool((diagnostics or {}).get("materialized_for_fine_edge_support")),
                    "support_chunk_edge_count": len(support_chunk_edge_ids),
                    **diagnostics,
                },
            )
        )

    for (level, prefix), child in rq_clusters_by_key.items():
        if level <= 1:
            continue
        parent = rq_clusters_by_key.get((level - 1, prefix[:-1]))
        if parent is None:
            continue
        add_edge(
            parent,
            child,
            "rq_parent_child",
            1.0,
            child.support_chunk_ids_json or [],
            {"parent_prefix": list(prefix[:-1]), "child_prefix": list(prefix), "rq_level": level},
        )

    children_by_parent: dict[tuple[int, ...], list[FineCluster]] = defaultdict(list)
    for (_level, prefix), cluster in rq_clusters_by_key.items():
        children_by_parent[prefix[:-1]].append(cluster)
    for parent_prefix, siblings in children_by_parent.items():
        if len(siblings) < 2:
            continue
        pairs: list[tuple[float, FineCluster, FineCluster]] = []
        for index, left in enumerate(siblings):
            for right in siblings[index + 1 :]:
                distance = math.sqrt(_sq_distance(left.centroid_json or [], right.centroid_json or []))
                pairs.append((distance, left, right))
        for distance, left, right in sorted(pairs, key=lambda item: item[0])[:8]:
            add_edge(
                left,
                right,
                "rq_sibling",
                math.exp(-distance / max(rq_tau(), 1e-6)),
                list(set(left.support_chunk_ids_json or []) & set(right.support_chunk_ids_json or [])),
                {"parent_prefix": list(parent_prefix), "centroid_distance": round(distance, 6)},
            )

    clusters_by_level: dict[int, list[FineCluster]] = defaultdict(list)
    for (level, _prefix), cluster in rq_clusters_by_key.items():
        clusters_by_level[level].append(cluster)
    for level, level_clusters in clusters_by_level.items():
        pairs = []
        for index, left in enumerate(level_clusters):
            for right in level_clusters[index + 1 :]:
                distance = math.sqrt(_sq_distance(left.centroid_json or [], right.centroid_json or []))
                pairs.append((distance, left, right))
        for distance, left, right in sorted(pairs, key=lambda item: item[0])[: max(2, len(level_clusters))]:
            add_edge(
                left,
                right,
                "rq_centroid_near",
                math.exp(-distance / max(rq_tau(), 1e-6)),
                list(set(left.support_chunk_ids_json or []) & set(right.support_chunk_ids_json or [])),
                {"rq_level": level, "centroid_distance": round(distance, 6)},
            )

    for rq_cluster in rq_clusters_by_key.values():
        rq_support = set(rq_cluster.support_chunk_ids_json or [])
        scored: list[tuple[float, FineCluster, list[str], list[str]]] = []
        for base_cluster in base_clusters:
            base_support = set(base_cluster.support_chunk_ids_json or [])
            overlap = sorted(rq_support & base_support)
            bridge_overlap = sorted((rq_support & set(bridge_chunk_ids)) | (base_support & set(bridge_chunk_ids)))
            if not overlap and not bridge_overlap:
                continue
            score = min(1.0, (len(overlap) + 1.5 * len(bridge_overlap)) / max(len(rq_support | base_support), 1))
            scored.append((score, base_cluster, overlap, bridge_overlap))
        for score, base_cluster, overlap, bridge_overlap in sorted(scored, key=lambda item: item[0], reverse=True)[:3]:
            add_edge(
                rq_cluster,
                base_cluster,
                "rq_overlap_bridge",
                max(0.15, score),
                overlap + bridge_overlap,
                {
                    "rq_prefix": rq_cluster.rq_path_prefix or [],
                    "base_cluster_key": base_cluster.cluster_key,
                    "overlap_count": len(overlap),
                    "bridge_overlap_count": len(bridge_overlap),
                },
            )

    all_rq_clusters = list(rq_clusters_by_key.values())
    if "rq_sibling" not in added_types and len(all_rq_clusters) >= 2:
        left, right = all_rq_clusters[0], all_rq_clusters[1]
        distance = math.sqrt(_sq_distance(left.centroid_json or [], right.centroid_json or []))
        add_edge(
            left,
            right,
            "rq_sibling",
            max(0.1, math.exp(-distance / max(rq_tau(), 1e-6))),
            list(set(left.support_chunk_ids_json or []) & set(right.support_chunk_ids_json or [])),
            {"fallback_pair": True, "centroid_distance": round(distance, 6)},
        )
    if "rq_centroid_near" not in added_types and len(all_rq_clusters) >= 2:
        left, right = all_rq_clusters[0], all_rq_clusters[1]
        distance = math.sqrt(_sq_distance(left.centroid_json or [], right.centroid_json or []))
        add_edge(
            left,
            right,
            "rq_centroid_near",
            max(0.1, math.exp(-distance / max(rq_tau(), 1e-6))),
            list(set(left.support_chunk_ids_json or []) & set(right.support_chunk_ids_json or [])),
            {"fallback_pair": True, "centroid_distance": round(distance, 6)},
        )
    if "rq_overlap_bridge" not in added_types and all_rq_clusters and base_clusters:
        rq_cluster = all_rq_clusters[0]
        base_cluster = base_clusters[0]
        support_ids = list(dict.fromkeys((rq_cluster.support_chunk_ids_json or []) + (base_cluster.support_chunk_ids_json or [])))[:8]
        add_edge(
            rq_cluster,
            base_cluster,
            "rq_overlap_bridge",
            0.15,
            support_ids,
            {"fallback_pair": True, "rq_prefix": rq_cluster.rq_path_prefix or [], "base_cluster_key": base_cluster.cluster_key},
        )


def add_rq_relation_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunk_by_id: dict[str, Chunk],
    assignments: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> None:
    items = sorted(assignments.items(), key=lambda item: item[0])
    levels = max((len(data["rq_path"]) for _chunk_id, data in items), default=1)
    added_types: set[str] = set()

    def add_rq_edge(left_id: str, right_id: str, edge_type: str, weight: float, features: dict[str, Any]) -> None:
        edge = add_chunk_relation_edge(db, graph_state, left_id, right_id, edge_type, weight, features, edges)
        if edge is not None:
            added_types.add(edge_type)

    by_leaf: dict[tuple[int, ...], list[str]] = defaultdict(list)
    by_parent: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for chunk_id, encoded in items:
        path = tuple(encoded["rq_path"])
        by_leaf[path].append(chunk_id)
        by_parent[path[:-1]].append(chunk_id)

    for member_ids in by_leaf.values():
        for left_id, right_id in nearest_rq_pairs(member_ids, assignments, max_pairs=6):
            features = rq_pair_features(assignments[left_id], assignments[right_id], levels)
            add_rq_edge(
                left_id,
                right_id,
                "rq_hierarchy_near",
                rq_edge_weight(features["lcp_depth"], levels, features["residual_distance"]),
                features,
            )

    for member_ids in by_parent.values():
        if len(member_ids) < 2:
            continue
        sibling_pairs = [
            (left_id, right_id)
            for index, left_id in enumerate(member_ids)
            for right_id in member_ids[index + 1 :]
            if assignments[left_id]["rq_path"] != assignments[right_id]["rq_path"]
        ]
        for left_id, right_id in sorted(sibling_pairs, key=lambda pair: residual_distance(assignments[pair[0]], assignments[pair[1]]))[:4]:
            features = rq_pair_features(assignments[left_id], assignments[right_id], levels)
            add_rq_edge(
                left_id,
                right_id,
                "rq_prefix_sibling",
                max(0.18, rq_edge_weight(features["lcp_depth"], levels, features["residual_distance"])),
                features,
            )

    residual_pairs = [
        (residual_distance(left_data, right_data), left_id, right_id)
        for index, (left_id, left_data) in enumerate(items)
        for right_id, right_data in items[index + 1 :]
    ]
    for _distance, left_id, right_id in sorted(residual_pairs, key=lambda item: item[0])[: max(4, len(items))]:
        features = rq_pair_features(assignments[left_id], assignments[right_id], levels)
        add_rq_edge(
            left_id,
            right_id,
            "rq_residual_near",
            max(0.12, rq_edge_weight(features["lcp_depth"], levels, features["residual_distance"])),
            features,
        )
    fallback_pairs = sorted(residual_pairs, key=lambda item: item[0])
    for missing_type in {"rq_hierarchy_near", "rq_prefix_sibling", "rq_residual_near"} - added_types:
        if not fallback_pairs:
            break
        _distance, left_id, right_id = fallback_pairs[0]
        features = rq_pair_features(assignments[left_id], assignments[right_id], levels)
        add_rq_edge(
            left_id,
            right_id,
            missing_type,
            max(0.10, rq_edge_weight(features["lcp_depth"], levels, features["residual_distance"])),
            {**features, "fallback_pair": True},
        )


def nearest_rq_pairs(member_ids: list[str], assignments: dict[str, dict[str, Any]], *, max_pairs: int) -> list[tuple[str, str]]:
    pairs = [
        (residual_distance(assignments[left_id], assignments[right_id]), left_id, right_id)
        for index, left_id in enumerate(member_ids)
        for right_id in member_ids[index + 1 :]
    ]
    return [(left_id, right_id) for _distance, left_id, right_id in sorted(pairs, key=lambda item: item[0])[:max_pairs]]


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


def rq_pair_features(left: dict[str, Any], right: dict[str, Any], levels: int) -> dict[str, Any]:
    distance = residual_distance(left, right)
    lcp = lcp_depth(left.get("rq_path") or [], right.get("rq_path") or [])
    tau = float(left.get("tau_r") or right.get("tau_r") or rq_tau())
    return {
        "lcp_depth": lcp,
        "residual_distance": round(distance, 6),
        "rq_weight": rq_edge_weight(lcp, levels, distance, tau_r=tau),
        "source_rq_path": left.get("rq_path") or [],
        "target_rq_path": right.get("rq_path") or [],
    }


def rq_edge_weight(lcp: int, levels: int, distance: float, *, tau_r: float | None = None) -> float:
    return round((lcp / max(levels, 1)) * math.exp(-distance / max(float(tau_r or rq_tau()), 1e-6)), 6)


def residual_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _vector_norm(_vector_sub(left.get("residual_vector") or [], right.get("residual_vector") or []))


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


def rq_candidate_score(query_rq: dict[str, Any], membership: FineClusterMembership) -> dict[str, Any]:
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


def build_fine_cluster_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    clusters: list[FineCluster],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
    *,
    batch_id: str | None = None,
) -> None:
    if len(clusters) < 2:
        return

    support_by_cluster = {cluster.id: set(cluster.support_chunk_ids_json or []) for cluster in clusters}
    bridge_by_cluster = {cluster.id: set(cluster.bridge_chunk_ids_json or []) for cluster in clusters}
    support_index: dict[str, set[str]] = defaultdict(set)
    bridge_index: dict[str, set[str]] = defaultdict(set)
    incident_edges_by_chunk: dict[str, list[ChunkRelationEdge]] = defaultdict(list)
    for cluster in clusters:
        for chunk_id in support_by_cluster[cluster.id]:
            support_index[chunk_id].add(cluster.id)
        for chunk_id in bridge_by_cluster[cluster.id]:
            bridge_index[chunk_id].add(cluster.id)
    for edge in edges.values():
        incident_edges_by_chunk[edge.source_chunk_id].append(edge)
        incident_edges_by_chunk[edge.target_chunk_id].append(edge)

    cluster_by_id = {cluster.id: cluster for cluster in clusters}
    candidate_pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def pair_key(left_id: str, right_id: str) -> tuple[str, str]:
        return (left_id, right_id) if left_id <= right_id else (right_id, left_id)

    def merge_candidate(left_id: str, right_id: str, edge_type: str, sim: float, bridge_chunks: set[str] | None = None) -> None:
        if left_id == right_id:
            return
        key = pair_key(left_id, right_id)
        current = candidate_pairs.get(key)
        if current is None or sim > float(current.get("sim") or 0.0) or edge_type == "overlap_bridge":
            candidate_pairs[key] = {
                "edge_type": edge_type if edge_type == "overlap_bridge" or current is None else current["edge_type"],
                "sim": max(sim, float((current or {}).get("sim") or 0.0)),
                "bridge_chunks": sorted((set((current or {}).get("bridge_chunks") or []) | set(bridge_chunks or set()))),
            }

    for chunk_id, bridge_cluster_ids in bridge_index.items():
        for left_id in bridge_cluster_ids:
            for right_id in support_index.get(chunk_id, set()):
                merge_candidate(left_id, right_id, "overlap_bridge", 0.42, {chunk_id})

    centroid_ids = [cluster.id for cluster in clusters if cluster.centroid_json]
    if len(centroid_ids) >= 2:
        try:
            import numpy as np

            width = min(len(cluster_by_id[cluster_id].centroid_json or []) for cluster_id in centroid_ids)
            matrix = np.array(
                [
                    [float(value) for value in (cluster_by_id[cluster_id].centroid_json or [])[:width]]
                    for cluster_id in centroid_ids
                ],
                dtype=float,
            )
            norms = np.linalg.norm(matrix, axis=1)
            valid = norms > 0
            matrix[valid] = matrix[valid] / norms[valid, None]
            similarities = matrix @ matrix.T
            np.fill_diagonal(similarities, -1.0)
            neighbor_k = min(12, len(centroid_ids) - 1)
            for row_index, left_id in enumerate(centroid_ids):
                if row_index % 100 == 0:
                    ensure_not_cancelled(db, batch_id)
                    context_graph_batch_heartbeat(
                        batch_id,
                        "chunk_relation:fine_edges",
                        {"fine_clusters": len(clusters), "candidate_pairs": len(candidate_pairs), "centroid_rows": row_index},
                    )
                top_indexes = np.argpartition(similarities[row_index], -neighbor_k)[-neighbor_k:]
                for col_index in top_indexes:
                    sim = float(similarities[row_index, col_index])
                    if sim > 0.45:
                        merge_candidate(left_id, centroid_ids[int(col_index)], "centroid_near", sim)
        except Exception:
            for index, left in enumerate(clusters):
                ensure_not_cancelled(db, batch_id)
                scored: list[tuple[float, str]] = []
                for right in clusters[index + 1 :]:
                    sim = cosine_similarity(left.centroid_json or [], right.centroid_json or [])
                    if sim > 0.45:
                        scored.append((sim, right.id))
                for sim, right_id in sorted(scored, key=lambda item: item[0], reverse=True)[:12]:
                    merge_candidate(left.id, right_id, "centroid_near", sim)

    def support_edge_ids(left_chunks: set[str], right_chunks: set[str]) -> list[str]:
        direct: list[str] = []
        smaller, larger = (left_chunks, right_chunks) if len(left_chunks) <= len(right_chunks) else (right_chunks, left_chunks)
        for chunk_id in smaller:
            for edge in incident_edges_by_chunk.get(chunk_id, []):
                other_id = edge.target_chunk_id if edge.source_chunk_id == chunk_id else edge.source_chunk_id
                if edge.id and other_id in larger:
                    direct.append(edge.id)
        if direct:
            return list(dict.fromkeys(direct))[:32]
        incident: list[str] = []
        for chunk_id in list(left_chunks | right_chunks)[:32]:
            incident.extend(edge.id for edge in incident_edges_by_chunk.get(chunk_id, []) if edge.id)
        return list(dict.fromkeys(incident))[:32]

    for index, ((left_id, right_id), candidate) in enumerate(sorted(candidate_pairs.items())):
        if index % 250 == 0:
            ensure_not_cancelled(db, batch_id)
            context_graph_batch_heartbeat(
                batch_id,
                "chunk_relation:fine_edges",
                {"fine_clusters": len(clusters), "candidate_pairs": len(candidate_pairs), "written_pairs": index},
            )
        left = cluster_by_id[left_id]
        right = cluster_by_id[right_id]
        left_chunks = support_by_cluster[left_id]
        right_chunks = support_by_cluster[right_id]
        bridge = set(candidate.get("bridge_chunks") or [])
        sim = float(candidate.get("sim") or 0.0)
        raw_strength = normalized_strength(max(sim, 0.42 if bridge else 0.0))
        support_chunk_edge_ids = support_edge_ids(left_chunks, right_chunks)
        support_chunk_ids = sorted(bridge) or sorted((left_chunks | right_chunks))[:16]
        if not support_chunk_edge_ids and left_chunks and right_chunks:
            left_rep = sorted(left_chunks)[0]
            right_rep = next((chunk_id for chunk_id in sorted(right_chunks) if chunk_id != left_rep), sorted(right_chunks)[0])
            if left_rep != right_rep:
                support_edge = add_chunk_relation_edge(
                    db,
                    graph_state,
                    left_rep,
                    right_rep,
                    "fine_cluster_bridge" if candidate["edge_type"] == "overlap_bridge" else "centroid_near",
                    max(0.1, raw_strength * 0.72),
                    {
                        "materialized_for_fine_edge_support": True,
                        "fine_edge_type": candidate["edge_type"],
                        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                    },
                    edges,
                    is_bridge=True,
                )
                db.flush()
                if support_edge is not None and support_edge.id:
                    support_chunk_edge_ids = [support_edge.id]
        if not support_chunk_edge_ids:
            continue
        db.add(
            FineClusterEdge(
                graph_state_id=graph_state.id,
                source_cluster_id=left.id,
                target_cluster_id=right.id,
                edge_type=candidate["edge_type"],
                weight=raw_strength,
                distance=distance_from_strength(raw_strength),
                raw_strength=raw_strength,
                raw_strength_summary_json={
                    "max_raw_strength": raw_strength,
                    "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
                },
                features_json={"centroid_similarity": round(sim, 6), "candidate_generation": "bridge_index_and_centroid_topk_v1"},
                source_algorithm="fine_graph_projection",
                protocol_version=graph_state.relation_protocol_version,
                support_chunk_ids_json=support_chunk_ids,
                support_chunk_edge_ids_json=support_chunk_edge_ids,
                diagnostics_json={
                    "centroid_similarity": round(sim, 6),
                    "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                    "support_chunk_edge_count": len(support_chunk_edge_ids),
                    "candidate_generation": "bridge_index_and_centroid_topk_v1",
                },
            )
        )


def relation_graph_stats(chunks: list[Chunk], edges: list[ChunkRelationEdge], clusters: list[FineCluster]) -> dict[str, Any]:
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
        "fine_cluster_count": len(clusters),
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
    terms = [term for term in tokenize_for_bm25(text.lower()) if len(term) > 2 and term not in STOP_TERMS]
    if not terms:
        return "General context"
    counts = Counter(terms)
    return " ".join(term for term, _ in counts.most_common(3)).title()


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


async def build_mid_concept_graph(db: Session, knowledge_base_id: str, relation_state: ChunkRelationGraphState, *, batch_id: str | None = None) -> MidConceptState:
    settings = get_settings()
    clusters = list(db.scalars(select(FineCluster).where(FineCluster.graph_state_id == relation_state.id, FineCluster.state == "active")).all())
    grounding_hash = stable_hash([cluster.id for cluster in clusters] + [relation_state.state_hash])
    state = MidConceptState(
        knowledge_base_id=knowledge_base_id,
        chunk_relation_graph_state_id=relation_state.id,
        state_hash=stable_hash({"grounding": grounding_hash, "prompt": MID_CONCEPT_PROMPT_VERSION}),
        grounding_hash=grounding_hash,
        prompt_protocol_version=MID_CONCEPT_PROMPT_VERSION,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db.add(state)
    db.flush()
    concepts: list[MidConcept] = []
    packet_batches = mid_concept_packet_batches(db, clusters, settings)
    total_batches = len(packet_batches)
    for batch_index, packet_batch in enumerate(packet_batches, start=1):
        context_graph_batch_heartbeat(
            batch_id,
            "mid_concepts",
            {"llm_batch": batch_index, "llm_batches": total_batches, "packet_count": len(packet_batch), "selected_fine_clusters": sum(len(batch) for batch in packet_batches)},
        )
        outputs = await define_mid_concepts_batch([packet for _, packet in packet_batch])
        for (cluster, packet), output in zip(packet_batch, outputs, strict=False):
            concept = write_mid_concept_from_output(db, state, knowledge_base_id, cluster, packet, output)
            concepts.append(concept)
        context_graph_batch_heartbeat(
            batch_id,
            "mid_concepts",
            {"completed_llm_batches": batch_index, "llm_batches": total_batches, "created_mid_concepts": len(concepts)},
        )
    build_mid_concept_edges(db, state, concepts)
    db.flush()
    mid_edges = list(db.scalars(select(MidConceptEdge).where(MidConceptEdge.concept_state_id == state.id)).all())
    supported_mid_edges = sum(1 for edge in mid_edges if edge.support_fine_edge_ids_json)
    stats = {
        "mid_concept_count": len(concepts),
        "grounded_concept_rate": 1.0 if concepts else 0.0,
        "mid_edge_count": len(mid_edges),
        "mid_edge_support_fine_edge_coverage": round(supported_mid_edges / max(len(mid_edges), 1), 6) if mid_edges else 1.0,
        "fine_cluster_candidates": len(clusters),
        "selected_fine_clusters": sum(len(batch) for batch in packet_batches),
        "llm_batches": len(packet_batches),
    }
    state.stats_json = stats
    state.diagnostics_json = {
        "candidate_keep_threshold": settings.mid_concept_candidate_keep_threshold,
        "max_model_batches": settings.mid_concept_extraction_max_model_batches,
        "max_candidates_per_batch": settings.mid_concept_extraction_max_candidates_per_batch,
        "max_tokens_per_batch": settings.mid_concept_extraction_max_tokens_per_batch,
        "selected_cluster_ids": [cluster.id for batch in packet_batches for cluster, _ in batch],
        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
    }
    state.state_hash = stable_hash({"concepts": [concept.id for concept in concepts], "stats": stats, "diagnostics": state.diagnostics_json})
    return state


def write_mid_concept_from_output(
    db: Session,
    state: MidConceptState,
    knowledge_base_id: str,
    cluster: FineCluster,
    packet: dict[str, Any],
    output: dict[str, Any],
) -> MidConcept:
    support_chunks = [chunk_id for chunk_id in output.get("support_chunk_ids", []) if chunk_id in set(cluster.support_chunk_ids_json or [])]
    if not support_chunks:
        support_chunks = list(cluster.support_chunk_ids_json or [])[:5]
    confidence, confidence_diagnostics = coerce_confidence(output.get("confidence"), default=0.72)
    concept = MidConcept(
        concept_state_id=state.id,
        knowledge_base_id=knowledge_base_id,
        canonical_label=str(output.get("canonical_label") or cluster.label)[:255],
        aliases_json=list(output.get("aliases") or []),
        definition=str(output.get("definition") or f"Grounded concept around {cluster.label}."),
        scope_note=str(output.get("scope_note") or ""),
        inclusion_criteria_json=list(output.get("inclusion_criteria") or []),
        exclusion_criteria_json=list(output.get("exclusion_criteria") or []),
        representative_chunk_ids_json=list(output.get("representative_chunk_ids") or cluster.representative_chunk_ids_json or [])[:5],
        support_fine_cluster_ids_json=[cluster.id],
        support_chunk_ids_json=support_chunks,
        confidence=confidence,
        llm_audit_json={
            "prompt_protocol_version": MID_CONCEPT_PROMPT_VERSION,
            "packet": packet,
            "raw_output": output,
            "confidence": confidence_diagnostics,
        },
        grounding_hash=stable_hash({"cluster": cluster.id, "chunks": support_chunks}),
    )
    db.add(concept)
    db.flush()
    db.add(
        MidConceptMembership(
            mid_concept_id=concept.id,
            fine_cluster_id=cluster.id,
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


def mid_concept_packet_batches(db: Session, clusters: list[FineCluster], settings: Any) -> list[list[tuple[FineCluster, dict[str, Any]]]]:
    if not clusters:
        return []
    max_batches = max(0, int(settings.mid_concept_extraction_max_model_batches))
    if max_batches <= 0:
        return []
    max_candidates = max(1, int(settings.mid_concept_extraction_max_candidates_per_batch))
    max_tokens = max(500, int(settings.mid_concept_extraction_max_tokens_per_batch))
    max_total = max_batches * max_candidates
    scored = sorted(((mid_concept_candidate_score(cluster, clusters), cluster) for cluster in clusters), key=lambda item: item[0], reverse=True)
    selected = [cluster for score, cluster in scored if score >= settings.mid_concept_candidate_keep_threshold][:max_total]
    if not selected:
        selected = [cluster for _, cluster in scored[: min(max_candidates, len(scored))]]
    batches: list[list[tuple[FineCluster, dict[str, Any]]]] = []
    current: list[tuple[FineCluster, dict[str, Any]]] = []
    current_tokens = 0
    for cluster in selected:
        packet = concept_packet_for_cluster(db, cluster)
        packet_tokens = max(1, rough_token_count(str(packet)))
        if current and (len(current) >= max_candidates or current_tokens + packet_tokens > max_tokens):
            batches.append(current)
            if len(batches) >= max_batches:
                return batches
            current = []
            current_tokens = 0
        current.append((cluster, packet))
        current_tokens += packet_tokens
    if current and len(batches) < max_batches:
        batches.append(current)
    return batches


def mid_concept_candidate_score(cluster: FineCluster, clusters: list[FineCluster]) -> float:
    max_support = max((len(item.support_chunk_ids_json or []) for item in clusters), default=1)
    support_score = len(cluster.support_chunk_ids_json or []) / max(max_support, 1)
    bridge_score = min(1.0, len(cluster.bridge_chunk_ids_json or []) / 5)
    rq_score = 0.15 if cluster.rq_level is not None else 0.0
    return round(min(1.0, 0.75 * support_score + 0.1 * bridge_score + rq_score), 6)


def mid_concept_fallback(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet.get("packet_id"),
        "canonical_label": (packet.get("candidate_labels") or ["Grounded Concept"])[0],
        "aliases": [],
        "definition": f"A grounded concept supported by fine cluster evidence for {(packet.get('candidate_labels') or ['this topic'])[0]}.",
        "scope_note": "Generated from chunk and fine cluster support.",
        "inclusion_criteria": ["Supported by the listed chunks."],
        "exclusion_criteria": ["Claims without supporting chunks."],
        "representative_chunk_ids": packet.get("representative_chunk_ids") or [],
        "support_chunk_ids": packet.get("support_chunk_ids") or [],
        "confidence": 0.72,
        "why_this_concept_exists": "The supporting chunks form a fine cluster.",
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
    output = await ChatProvider().classify_json(system_prompt=system, user_prompt=str({"concept_packets": packets}), fallback={"concepts": fallback_concepts})
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


def concept_packet_for_cluster(db: Session, cluster: FineCluster) -> dict[str, Any]:
    membership_rows = list(
        db.scalars(
            select(FineClusterMembership)
            .where(FineClusterMembership.fine_cluster_id == cluster.id)
        ).all()
    )
    membership_rows = sorted(membership_rows, key=lambda row: float(row.residual_norm or -1.0), reverse=True)[:6]
    residual_outlier_ids = [row.chunk_id for row in membership_rows if row.residual_norm is not None][:3]
    representative_ids = list(dict.fromkeys(list(cluster.representative_chunk_ids_json or []) + residual_outlier_ids))
    chunk_ids = list(representative_ids or cluster.support_chunk_ids_json or [])[:6]
    chunks = list(db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()) if chunk_ids else []
    support_fine_edges = list(
        db.scalars(
            select(FineClusterEdge).where(
                FineClusterEdge.graph_state_id == cluster.graph_state_id,
                (FineClusterEdge.source_cluster_id == cluster.id) | (FineClusterEdge.target_cluster_id == cluster.id),
            )
        ).all()
    )
    support_fine_edge_ids = [edge.id for edge in support_fine_edges if edge.support_chunk_edge_ids_json]
    related_fine_node_ids = list(
        dict.fromkeys(
            [
                edge.target_cluster_id if edge.source_cluster_id == cluster.id else edge.source_cluster_id
                for edge in support_fine_edges
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
        "support_fine_node_ids": [cluster.id],
        "support_fine_edge_ids": support_fine_edge_ids[:30],
        "bridge_fine_node_ids": related_fine_node_ids[:12] if cluster.bridge_chunk_ids_json else [],
        "boundary_fine_node_ids": related_fine_node_ids[:12],
        "candidate_labels": [cluster.label],
        "representative_chunk_ids": [chunk.id for chunk in chunks],
        "support_chunk_count": len(cluster.support_chunk_ids_json or []),
        "support_chunk_ids": list(cluster.support_chunk_ids_json or [])[:30],
        "bridge_chunk_count": len(cluster.bridge_chunk_ids_json or []),
        "bridge_chunk_ids": list(cluster.bridge_chunk_ids_json or [])[:20],
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
    clusters_by_concept = {concept.id: set(concept.support_fine_cluster_ids_json or []) for concept in concepts}
    for index, left in enumerate(concepts):
        for right in concepts[index + 1 :]:
            left_clusters = clusters_by_concept[left.id]
            right_clusters = clusters_by_concept[right.id]
            fine_edges = list(
                db.scalars(
                    select(FineClusterEdge).where(
                        FineClusterEdge.graph_state_id == state.chunk_relation_graph_state_id,
                        (
                            (FineClusterEdge.source_cluster_id.in_(left_clusters) & FineClusterEdge.target_cluster_id.in_(right_clusters))
                            | (FineClusterEdge.source_cluster_id.in_(right_clusters) & FineClusterEdge.target_cluster_id.in_(left_clusters))
                        ),
                    )
                ).all()
            )
            fine_edges = [edge for edge in fine_edges if edge.support_chunk_edge_ids_json]
            if not fine_edges:
                continue
            support_fine_edge_ids = [edge.id for edge in fine_edges]
            support_chunk_ids = list(
                dict.fromkeys(
                    chunk_id
                    for edge in fine_edges
                    for chunk_id in (edge.support_chunk_ids_json or [])
                )
            )[:24]
            distances = [float(edge.distance if edge.distance is not None else distance_from_strength(edge.raw_strength or edge.weight)) for edge in fine_edges]
            raw_strengths = [normalized_strength(float(edge.raw_strength or edge.weight or 0.1)) for edge in fine_edges]
            raw_strength = max(raw_strengths) if raw_strengths else 0.1
            distance = min(distances) if distances else distance_from_strength(raw_strength)
            db.add(
                MidConceptEdge(
                    concept_state_id=state.id,
                    source_concept_id=left.id,
                    target_concept_id=right.id,
                    edge_type="fine_edge_projection",
                    weight=raw_strength,
                    distance=distance,
                    raw_strength_summary_json={
                        "max_raw_strength": raw_strength,
                        "mean_raw_strength": round(sum(raw_strengths) / max(len(raw_strengths), 1), 6),
                        "min_distance": distance,
                        "support_fine_edge_count": len(support_fine_edge_ids),
                        "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
                    },
                    network_evidence_score=raw_strength,
                    llm_confidence=0.7,
                    support_chunk_ids_json=support_chunk_ids,
                    support_relation_edge_ids_json=list(
                        dict.fromkeys(
                            chunk_edge_id
                            for edge in fine_edges
                            for chunk_edge_id in (edge.support_chunk_edge_ids_json or [])
                        )
                    )[:40],
                    support_fine_edge_ids_json=support_fine_edge_ids,
                    support_fine_node_ids_json=sorted(left_clusters | right_clusters),
                    explanation="Concept edge admitted from fine-edge projection support; LLM may explain but cannot create it without bottom evidence.",
                    diagnostics_json={
                        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                        "support_fine_edge_count": len(support_fine_edge_ids),
                        "support_chunk_edge_count": sum(len(edge.support_chunk_edge_ids_json or []) for edge in fine_edges),
                    },
                )
            )


async def build_coarse_concept_graph(db: Session, knowledge_base_id: str, mid_state: MidConceptState) -> CoarseConceptState:
    mid_concepts = list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == mid_state.id, MidConcept.state == "active")).all())
    mid_edges = list(db.scalars(select(MidConceptEdge).where(MidConceptEdge.concept_state_id == mid_state.id)).all())
    grounding_hash = stable_hash([concept.id for concept in mid_concepts] + [mid_state.state_hash])
    state = CoarseConceptState(
        knowledge_base_id=knowledge_base_id,
        mid_concept_state_id=mid_state.id,
        state_hash=stable_hash({"grounding": grounding_hash, "prompt": COARSE_CONCEPT_PROMPT_VERSION}),
        grounding_hash=grounding_hash,
        prompt_protocol_version=COARSE_CONCEPT_PROMPT_VERSION,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db.add(state)
    db.flush()
    communities = coarse_communities(mid_concepts, mid_edges)
    support_mid_edge_ids_by_community: dict[int, list[str]] = {}
    coarse_concepts: list[CoarseConcept] = []
    for index, community in enumerate(communities, start=1):
        mid_ids = [concept.id for concept in community]
        mid_id_set = set(mid_ids)
        support_mid_edges = [
            edge
            for edge in mid_edges
            if edge.source_concept_id in mid_id_set or edge.target_concept_id in mid_id_set
        ]
        support_mid_edge_ids_by_community[index] = [edge.id for edge in support_mid_edges]
        support_chunks = list(
            dict.fromkeys(
                chunk_id
                for concept in community
                for chunk_id in (concept.support_chunk_ids_json or [])
            )
        )[:50]
        packet = {
            "community_id": index,
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
            "support_mid_edge_ids": support_mid_edge_ids_by_community[index],
            "support_chunk_ids": support_chunks,
            "community_diagnostics": {"candidate_source": "mid_distance_graph_community"},
            "grounding_hash": stable_hash([concept.id for concept in community]),
        }
        output = await define_coarse_concept(packet)
        confidence, confidence_diagnostics = coerce_confidence(output.get("confidence"), default=0.72)
        coarse = CoarseConcept(
            coarse_state_id=state.id,
            knowledge_base_id=knowledge_base_id,
            canonical_label=str(output.get("coarse_label") or output.get("canonical_label") or f"Topic Area {index}")[:255],
            aliases_json=list(output.get("aliases") or []),
            definition=str(output.get("definition") or "A high-level topic area grounded in mid-level concepts."),
            included_mid_concept_ids_json=mid_ids,
            boundary_mid_concept_ids_json=list(output.get("boundary_concepts") or [])[:10],
            bridge_mid_concept_ids_json=list(output.get("bridge_concepts") or packet["bridge_concepts"])[:10],
            cross_community_weak_ties_json=list(output.get("cross_community_weak_ties") or []),
            confidence=confidence,
            llm_audit_json={
                "prompt_protocol_version": COARSE_CONCEPT_PROMPT_VERSION,
                "packet": packet,
                "raw_output": output,
                "confidence": confidence_diagnostics,
            },
            grounding_hash=stable_hash({"community": index, "mid": mid_ids}),
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
            if not projected_mid_edges:
                continue
            raw_strengths = [
                normalized_strength(float((edge.raw_strength_summary_json or {}).get("max_raw_strength") or edge.weight or 0.1))
                for edge in projected_mid_edges
            ]
            distances = [
                float(edge.distance if edge.distance is not None else distance_from_strength(raw_strengths[idx]))
                for idx, edge in enumerate(projected_mid_edges)
            ]
            raw_strength = max(raw_strengths) if raw_strengths else 0.1
            distance = min(distances) if distances else distance_from_strength(raw_strength)
            support_mid_edge_ids = [edge.id for edge in projected_mid_edges]
            support_fine_edge_ids = list(
                dict.fromkeys(
                    fine_edge_id
                    for edge in projected_mid_edges
                    for fine_edge_id in (edge.support_fine_edge_ids_json or [])
                )
            )
            support_chunk_ids = list(
                dict.fromkeys(
                    chunk_id
                    for edge in projected_mid_edges
                    for chunk_id in (edge.support_chunk_ids_json or [])
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
                    edge_type="mid_edge_projection",
                    weight=raw_strength,
                    distance=distance,
                    raw_strength_summary_json={
                        "max_raw_strength": raw_strength,
                        "mean_raw_strength": round(sum(raw_strengths) / max(len(raw_strengths), 1), 6),
                        "min_distance": distance,
                        "support_mid_edge_count": len(support_mid_edge_ids),
                        "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_VERSION,
                    },
                    support_mid_concept_ids_json=list(left_mid_ids | right_mid_ids)[:20],
                    support_mid_edge_ids_json=support_mid_edge_ids,
                    support_fine_edge_ids_json=support_fine_edge_ids[:60],
                    support_chunk_ids_json=support_chunk_ids[:60],
                    cross_community_weak_ties_json=weak_ties,
                    explanation="Coarse edge admitted from projected mid edges; weak ties are retained for traversal.",
                    diagnostics_json={
                        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
                        "support_mid_edge_count": len(support_mid_edge_ids),
                        "support_fine_edge_count": len(support_fine_edge_ids),
                    },
                )
            )
    db.flush()
    coarse_edges = list(db.scalars(select(CoarseConceptEdge).where(CoarseConceptEdge.coarse_state_id == state.id)).all())
    supported_coarse_edges = sum(1 for edge in coarse_edges if edge.support_mid_edge_ids_json)
    stats = {
        "coarse_concept_count": len(coarse_concepts),
        "mid_concept_count": len(mid_concepts),
        "coarse_edge_count": len(coarse_edges),
        "coarse_edge_support_mid_edge_coverage": round(supported_coarse_edges / max(len(coarse_edges), 1), 6) if coarse_edges else 1.0,
        "bridge_concept_count": sum(len(item.bridge_mid_concept_ids_json or []) for item in coarse_concepts),
        "singleton_rate": round(sum(1 for item in coarse_concepts if len(item.included_mid_concept_ids_json or []) <= 1) / max(len(coarse_concepts), 1), 6),
    }
    community_diagnostics = coarse_community_diagnostics(mid_concepts, mid_edges, communities)
    state.stats_json = stats
    state.diagnostics_json = {
        "community_detection": "mid_distance_graph_greedy_modularity_v1",
        "legacy_label_bucket_active": False,
        "connected_components_used_as_final": False,
        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_VERSION,
        **community_diagnostics,
        "bridge_density": round(stats["bridge_concept_count"] / max(stats["mid_concept_count"], 1), 6),
    }
    state.state_hash = stable_hash({"coarse": [concept.id for concept in coarse_concepts], "stats": stats})
    return state


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
        output = await ChatProvider().classify_json(system_prompt=system, user_prompt=str(packet), fallback=fallback)
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
    fine_hash = stable_hash([cluster.id for cluster in db.scalars(select(FineCluster).where(FineCluster.graph_state_id == relation_state.id)).all()])
    chunk_scope_hash = compute_chunk_scope_hash(chunks)
    context_hash = stable_hash(
        {
            "chunk_scope": chunk_scope_hash,
            "structure": structure_hash,
            "relation": relation_state.state_hash,
            "fine": fine_hash,
            "mid": mid_state.state_hash,
            "coarse": coarse_state.state_hash,
            "runtime_settings": runtime_settings_state_hash(),
            "agent_operating_envelope": agent_operating_envelope_state_hash(),
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
        fine_cluster_hash=fine_hash,
        mid_concept_hash=mid_state.state_hash,
        coarse_concept_hash=coarse_state.state_hash,
        context_graph_hash=context_hash,
        runtime_settings_hash=runtime_settings_state_hash(),
        agent_operating_envelope_hash=agent_operating_envelope_state_hash(),
        prompt_protocol_hash=stable_hash([MID_CONCEPT_PROMPT_VERSION, COARSE_CONCEPT_PROMPT_VERSION, ANSWER_PROMPT_PROTOCOL_VERSION]),
        stats_json={
            "chunks": len(chunks),
            "relation_edges": db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state.id)) or 0,
            "fine_clusters": db.scalar(select(func.count(FineCluster.id)).where(FineCluster.graph_state_id == relation_state.id)) or 0,
            "mid_concepts": db.scalar(select(func.count(MidConcept.id)).where(MidConcept.concept_state_id == mid_state.id)) or 0,
            "coarse_concepts": db.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.coarse_state_id == coarse_state.id)) or 0,
        },
        diagnostics_json={"protocol": CONTEXT_GRAPH_PROTOCOL_VERSION},
        state="active",
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
    top_k: int,
) -> LayeredSearchResult:
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    if not chunks:
        query_facets = query_facets_for_search(query)
        trace = RetrievalTrace(
            knowledge_base_id=knowledge_base_id,
            query=query,
            filters_json=filters.model_dump(),
            result_chunk_ids_json=[],
            query_facets_json=query_facets,
            convergence_json={"reason": "no_active_chunks"},
            diagnostics_json={"reason": "no_active_chunks"},
            edge_distance_protocol_hash=edge_distance_protocol_hash(),
            edge_projection_protocol_hash=edge_projection_protocol_hash(),
            traversal_protocol_hash=traversal_protocol_hash(),
        )
        db.add(trace)
        db.flush()
        return LayeredSearchResult([], trace, {"retrieval_pipeline": "layered_context_graph", "reason": "no_active_chunks"})
    context_state = latest_context_graph_state(db, knowledge_base_id)
    relation_state = latest_relation_state(db, knowledge_base_id)
    query_vector = (await EmbeddingProvider().embed_texts([query], text_type="query"))[0]
    query_rq = encode_query_rq(relation_state, query_vector)
    query_facets = query_facets_for_search(query)
    coarse_entries = select_coarse_entries(db, knowledge_base_id, query_facets)
    mid_entries = select_mid_entries(db, knowledge_base_id, query_facets, coarse_entries)
    fine_entries = select_fine_entries(db, knowledge_base_id, query_vector, query_rq, mid_entries)
    dense_entries = dense_chunk_entries(db, knowledge_base_id, query_vector)
    lexical_entries = lexical_chunk_entries(db, knowledge_base_id, query_facets["terms"])
    traversal = execute_priority_queue_traversal(
        db,
        knowledge_base_id=knowledge_base_id,
        chunks=chunks,
        filters=filters,
        query_facets=query_facets,
        coarse_entries=coarse_entries,
        mid_entries=mid_entries,
        fine_entries=fine_entries,
        dense_entries=dense_entries,
        lexical_entries=lexical_entries,
        query_rq=query_rq,
        top_k=top_k,
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
    )
    for item in results:
        item["metadata"]["retrieval_trace_id"] = trace.id
        for citation in item["citations"]:
            citation["retrieval_trace_id"] = trace.id
            if isinstance(citation.get("source_span"), dict):
                citation["source_span"]["retrieval_trace_id"] = trace.id
    audit = {
        "retrieval_pipeline": "layered_context_graph",
        "retrieval_trace_id": trace.id,
        "context_graph_state_id": context_state.id if context_state else None,
        "degraded_mode": is_degraded_mode(),
        "coarse_entries": len(coarse_entries),
        "mid_entries": len(mid_entries),
        "fine_entries": len(fine_entries),
        "frontier_pops": len(traversal["frontier_pops"]),
        "dominance_pruned_count": traversal["convergence"]["dominance_pruned_count"],
        "query_rq_path": query_rq.get("rq_path") if query_rq else [],
    }
    return LayeredSearchResult(results, trace, audit)


def query_facets_for_search(query: str) -> dict[str, Any]:
    terms = tokenize_for_bm25(query)
    intent = "formula_table_lookup" if any(term in query.lower() for term in ("formula", "equation", "table", "公式", "表格")) else "semantic"
    return {
        "query": query,
        "terms": terms,
        "required_facets": list(dict.fromkeys(terms[:8])),
        "intent": intent,
    }


def select_coarse_entries(db: Session, knowledge_base_id: str, query_facets: dict[str, Any]) -> dict[str, float]:
    state = latest_coarse_state(db, knowledge_base_id)
    if state is None:
        return {}
    concepts = db.scalars(select(CoarseConcept).where(CoarseConcept.coarse_state_id == state.id, CoarseConcept.state == "active")).all()
    envelope = agent_operating_envelope()
    scores = _text_entry_score({concept.id: f"{concept.canonical_label} {concept.definition}" for concept in concepts}, query_facets["terms"], top_n=envelope["coarse_entry_budget"])
    if scores or not concepts:
        return scores
    supported = sorted(
        concepts,
        key=lambda concept: (len(concept.included_mid_concept_ids_json or []), len(concept.bridge_mid_concept_ids_json or [])),
        reverse=True,
    )
    return {concept.id: 0.35 for concept in supported[: envelope["coarse_entry_budget"]]}


def select_mid_entries(db: Session, knowledge_base_id: str, query_facets: dict[str, Any], coarse_entries: dict[str, float]) -> dict[str, float]:
    state = latest_mid_state(db, knowledge_base_id)
    if state is None:
        return {}
    concepts = list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == state.id, MidConcept.state == "active")).all())
    envelope = agent_operating_envelope()
    scores = _text_entry_score({concept.id: f"{concept.canonical_label} {concept.definition}" for concept in concepts}, query_facets["terms"], top_n=envelope["mid_entry_budget"])
    coarse_ids = set(coarse_entries)
    if coarse_ids:
        coarse_concepts = db.scalars(select(CoarseConcept).where(CoarseConcept.id.in_(coarse_ids))).all()
        boosted_mid_ids = {mid_id for coarse in coarse_concepts for mid_id in (coarse.included_mid_concept_ids_json or [])}
        for concept in concepts:
            if concept.id in boosted_mid_ids:
                scores[concept.id] = max(scores.get(concept.id, 0.0), 0.45)
    if not scores and concepts:
        supported = sorted(concepts, key=lambda concept: len(concept.support_fine_cluster_ids_json or []), reverse=True)
        scores = {concept.id: 0.35 for concept in supported[: envelope["mid_entry_budget"]]}
    return scores


def select_fine_entries(db: Session, knowledge_base_id: str, query_vector: list[float], query_rq: dict[str, Any] | None, mid_entries: dict[str, float]) -> dict[str, float]:
    relation_state = latest_relation_state(db, knowledge_base_id)
    if relation_state is None:
        return {}
    clusters = list(db.scalars(select(FineCluster).where(FineCluster.graph_state_id == relation_state.id, FineCluster.state == "active")).all())
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
            scores[row.fine_cluster_id] = max(scores.get(row.fine_cluster_id, 0.0), score * row.membership_score)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[: agent_operating_envelope()["fine_entry_budget"]])


def dense_chunk_entries(db: Session, knowledge_base_id: str, query_vector: list[float]) -> dict[str, float]:
    records = list(db.scalars(select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base_id, VectorRecord.vector_status == "ready")).all())
    scored: list[tuple[str, float]] = []
    for record in records:
        vector = [float(value) for value in ((record.diagnostics_json or {}).get("embedding_vector") or []) if isinstance(value, (int, float))]
        if vector:
            scored.append((record.chunk_id, max(0.0, cosine_similarity(query_vector, vector))))
    return dict(sorted(scored, key=lambda item: item[1], reverse=True)[:80])


def lexical_chunk_entries(db: Session, knowledge_base_id: str, query_terms: list[str]) -> dict[str, float]:
    records = list(db.scalars(select(BM25Record).where(BM25Record.knowledge_base_id == knowledge_base_id, BM25Record.state == "ready")).all())
    if not query_terms or not records:
        return {}
    corpus = []
    for record in records:
        terms: list[str] = []
        for term, count in (record.term_frequencies_json or {}).items():
            terms.extend([term] * int(count))
        corpus.append(terms)
    bm25 = BM25Okapi(corpus)
    raw_scores = bm25.get_scores(query_terms)
    max_score = max([float(score) for score in raw_scores] + [1.0])
    scored = [(record.chunk_id, float(score) / max_score) for record, score in zip(records, raw_scores) if float(score) > 0]
    return dict(sorted(scored, key=lambda item: item[1], reverse=True)[:80])


def _text_entry_score(text_by_id: dict[str, str], query_terms: list[str], *, top_n: int) -> dict[str, float]:
    if not text_by_id or not query_terms:
        return {}
    scored: list[tuple[str, float]] = []
    query_set = set(query_terms)
    for item_id, text in text_by_id.items():
        terms = set(tokenize_for_bm25(text))
        overlap = len(query_set.intersection(terms))
        if overlap:
            scored.append((item_id, min(1.0, overlap / max(len(query_set), 1))))
    return dict(sorted(scored, key=lambda item: item[1], reverse=True)[:top_n])


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
        "support_fine_edge_ids_json",
        "support_fine_node_ids_json",
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
        text_terms = set(tokenize_for_bm25(node_text_by_id.get(node_id, "")))
        return {facet for facet in required_facets if facet in text_terms}

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
    ambiguous_decisions: list[dict[str, Any]] = []
    accepted_by_node: dict[str, dict[str, Any]] = {}
    dominance_labels: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    dominance_pruned_count = 0
    expansion_count = 0
    max_expansions = int(envelope.get("frontier_expansion_budget") or 0)
    max_depth = int(envelope.get("max_depth_per_layer") or 1)
    max_labels = int(envelope.get("max_labels_per_node") or 1)
    cycle_reward_cap = float(envelope.get("max_cycle_reward_per_path") or 0.0)
    ambiguous_low = float(envelope.get("ambiguous_edge_distance_low") or 0.0)
    ambiguous_high = float(envelope.get("ambiguous_edge_distance_high") or 0.0)
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
            if ambiguous_low <= edge_distance <= ambiguous_high:
                ambiguous_decisions.append(
                    {
                        "layer": layer,
                        "edge_id": edge.id,
                        "from_node_id": state["node_id"],
                        "to_node_id": neighbor_id,
                        "distance": edge_distance,
                        "decision": "follow_within_budget",
                    }
                )
            visit_counts = dict(state["visit_counts"])
            previous_visits = int(visit_counts.get(neighbor_id, 0))
            visit_counts[neighbor_id] = previous_visits + 1
            reward_increment = min(cycle_reward_cap - float(state["reward_so_far"]), 0.04 * _edge_raw_strength(edge)) if previous_visits else 0.0
            reward_so_far = round(max(0.0, float(state["reward_so_far"]) + max(0.0, reward_increment)), 6)
            covered = set(state["covered_facets"]) | covered_facets_for_node(neighbor_id)
            roles = set(state["evidence_roles"])
            roles.add(str(getattr(edge, "edge_type", "edge")))
            next_state = {
                "layer": layer,
                "node_id": neighbor_id,
                "path": list(state["path"]) + [neighbor_id],
                "path_edge_ids": list(state["path_edge_ids"]) + [edge.id],
                "distance_so_far": round(float(state["distance_so_far"]) + edge_distance, 6),
                "reward_so_far": reward_so_far,
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
        "ambiguous_edge_decisions": ambiguous_decisions,
        "convergence": {
            "reason": stop_reason,
            "frontier_expansion_count": expansion_count,
            "dominance_pruned_count": dominance_pruned_count,
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
    fine_entries: dict[str, float],
    dense_entries: dict[str, float],
    lexical_entries: dict[str, float],
    query_rq: dict[str, Any] | None,
    top_k: int,
) -> dict[str, Any]:
    envelope = agent_operating_envelope()
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    layer_walks: dict[str, dict[str, Any]] = {}

    coarse_state = latest_coarse_state(db, knowledge_base_id)
    coarse_concepts = (
        list(db.scalars(select(CoarseConcept).where(CoarseConcept.coarse_state_id == coarse_state.id, CoarseConcept.state == "active")).all())
        if coarse_state
        else []
    )
    coarse_text = {concept.id: f"{concept.canonical_label} {concept.definition}" for concept in coarse_concepts}
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
    for coarse_id in layer_walks["coarse"]["accepted_nodes"]:
        coarse = next((item for item in coarse_concepts if item.id == coarse_id), None)
        if coarse is None:
            continue
        coarse_strength = max(coarse_entries.get(coarse_id, 0.35), 0.35)
        for mid_id in coarse.included_mid_concept_ids_json or []:
            mid_entries[mid_id] = max(mid_entries.get(mid_id, 0.0), coarse_strength * 0.9)

    mid_state = latest_mid_state(db, knowledge_base_id)
    mid_concepts = (
        list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == mid_state.id, MidConcept.state == "active")).all())
        if mid_state
        else []
    )
    mid_text = {concept.id: f"{concept.canonical_label} {concept.definition} {concept.scope_note}" for concept in mid_concepts}
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
    for mid_id in layer_walks["mid"]["accepted_nodes"]:
        rows = db.scalars(select(MidConceptMembership).where(MidConceptMembership.mid_concept_id == mid_id)).all()
        mid_strength = max(mid_entries.get(mid_id, 0.35), 0.35)
        for row in rows:
            fine_entries[row.fine_cluster_id] = max(fine_entries.get(row.fine_cluster_id, 0.0), mid_strength * float(row.membership_score or 1.0))

    relation_state = latest_relation_state(db, knowledge_base_id)
    fine_clusters = (
        list(db.scalars(select(FineCluster).where(FineCluster.graph_state_id == relation_state.id, FineCluster.state == "active")).all())
        if relation_state
        else []
    )
    fine_text = {
        cluster.id: " ".join(
            str(value)
            for value in [
                cluster.label,
                cluster.node_type,
                cluster.cluster_key,
                (cluster.diagnostics_json or {}).get("source"),
                " ".join((cluster.diagnostics_json or {}).get("representative_terms") or []),
            ]
            if value
        )
        for cluster in fine_clusters
    }
    fine_edges = list(db.scalars(select(FineClusterEdge).where(FineClusterEdge.graph_state_id == relation_state.id)).all()) if relation_state else []
    layer_walks["fine"] = execute_layer_priority_walk(
        layer="fine",
        entry_scores=fine_entries,
        node_text_by_id=fine_text,
        adjacency=_build_adjacency(fine_edges, "source_cluster_id", "target_cluster_id"),
        query_facets=query_facets,
        source_attr="source_cluster_id",
        target_attr="target_cluster_id",
        envelope=envelope,
    )
    for fine_id in layer_walks["fine"]["accepted_nodes"]:
        fine_entries[fine_id] = max(fine_entries.get(fine_id, 0.0), 0.34)

    seed_strengths: dict[str, float] = defaultdict(float)
    seed_roles: dict[str, set[str]] = defaultdict(set)
    seed_metadata: dict[str, dict[str, Any]] = defaultdict(dict)

    def add_seed(chunk_id: str, strength: float, role: str, metadata: dict[str, Any] | None = None) -> None:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None or not passes_filters(db, chunk, filters):
            return
        seed_strengths[chunk_id] = max(seed_strengths[chunk_id], normalized_strength(strength))
        seed_roles[chunk_id].add(role)
        if metadata:
            current = seed_metadata[chunk_id]
            for key, value in metadata.items():
                if isinstance(value, list):
                    current.setdefault(key, [])
                    current[key].extend(item for item in value if item not in current[key])
                else:
                    current[key] = value

    for chunk_id, strength in dense_entries.items():
        add_seed(chunk_id, strength, "dense_entry")
    for chunk_id, strength in lexical_entries.items():
        add_seed(chunk_id, strength, "bm25_entry")

    for fine_id, strength in fine_entries.items():
        member_rows = list(db.scalars(select(FineClusterMembership).where(FineClusterMembership.fine_cluster_id == fine_id)).all())
        for row in member_rows:
            metadata: dict[str, Any] = {"fine_cluster_ids": [fine_id]}
            if query_rq and row.rq_path:
                metadata["rq"] = rq_candidate_score(query_rq, row)
            add_seed(row.chunk_id, strength * float(row.membership_score or 1.0), "fine_membership_entry", metadata)

    for concept_id, strength in mid_entries.items():
        concept = db.get(MidConcept, concept_id)
        if concept is None:
            continue
        for chunk_id in concept.support_chunk_ids_json or []:
            add_seed(chunk_id, strength, "mid_drilldown_entry", {"mid_concept_ids": [concept_id]})

    for concept_id, strength in coarse_entries.items():
        coarse = db.get(CoarseConcept, concept_id)
        if coarse is None:
            continue
        mids = db.scalars(select(MidConcept).where(MidConcept.id.in_(coarse.included_mid_concept_ids_json or []))).all()
        for mid in mids:
            for chunk_id in mid.support_chunk_ids_json or []:
                add_seed(chunk_id, strength, "coarse_to_mid_drilldown_entry", {"coarse_concept_ids": [concept_id], "mid_concept_ids": [mid.id]})

    required_facets = set(query_facets.get("required_facets") or [])

    def covered_facets_for_chunk(chunk: Chunk) -> set[str]:
        text_terms = set(tokenize_for_bm25(chunk.text))
        return {facet for facet in required_facets if facet in text_terms}

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
        "ambiguous_edge_decisions": [],
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
    ambiguous_decisions: list[dict[str, Any]] = []
    dominance_labels: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    dominance_pruned_count = 0
    expansion_count = 0
    stop_reason = "frontier_empty"
    max_expansions = int(envelope["frontier_expansion_budget"])
    max_depth = int(envelope["max_depth_per_layer"])
    max_labels = int(envelope["max_labels_per_node"])
    cycle_reward_cap = float(envelope["max_cycle_reward_per_path"])
    ambiguous_low = float(envelope["ambiguous_edge_distance_low"])
    ambiguous_high = float(envelope["ambiguous_edge_distance_high"])

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
            decision = "follow"
            if ambiguous_low <= edge_distance <= ambiguous_high:
                decision = "follow_within_budget"
                ambiguous_decisions.append(
                    {
                        "edge_id": edge.id,
                        "from_chunk_id": chunk.id,
                        "to_chunk_id": neighbor_id,
                        "distance": edge_distance,
                        "decision": decision,
                    }
                )
            visit_counts = dict(state["visit_counts"])
            previous_visits = int(visit_counts.get(neighbor_id, 0))
            visit_counts[neighbor_id] = previous_visits + 1
            reward_increment = min(cycle_reward_cap - float(state["reward_so_far"]), 0.04 * normalized_strength(edge.raw_strength or 1e-6)) if previous_visits else 0.0
            reward_so_far = round(max(0.0, float(state["reward_so_far"]) + max(0.0, reward_increment)), 6)
            covered = set(state["covered_facets"]) | covered_facets_for_chunk(neighbor)
            roles = set(state["evidence_roles"])
            roles.add(edge.edge_type)
            next_state = {
                "layer": "chunk",
                "node_id": neighbor_id,
                "path": list(state["path"]) + [neighbor_id],
                "path_edge_ids": list(state["path_edge_ids"]) + [edge.id],
                "distance_so_far": round(float(state["distance_so_far"]) + edge_distance, 6),
                "reward_so_far": reward_so_far,
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
    rq_membership_by_chunk: dict[str, FineClusterMembership] = {}
    if query_rq and accepted_chunk_ids:
        rq_rows = list(
            db.scalars(
                select(FineClusterMembership).where(
                    FineClusterMembership.chunk_id.in_(accepted_chunk_ids),
                    FineClusterMembership.rq_path.is_not(None),
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
        "cycle_reward_bounded": True,
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
        "ambiguous_edge_decisions": ambiguous_decisions,
        "convergence": chunk_convergence,
    }
    all_entry_nodes = [
        entry
        for layer_name in ("coarse", "mid", "fine", "chunk")
        for entry in (layer_walks.get(layer_name, {}).get("entry_nodes") or [])
    ]
    all_frontier = [
        snapshot
        for layer_name in ("coarse", "mid", "fine", "chunk")
        for snapshot in (layer_walks.get(layer_name, {}).get("frontier_json") or [])
    ]
    all_path_labels = [
        label
        for layer_name in ("coarse", "mid", "fine", "chunk")
        for label in (layer_walks.get(layer_name, {}).get("path_labels") or [])
    ]
    all_ambiguous_decisions = [
        decision
        for layer_name in ("coarse", "mid", "fine", "chunk")
        for decision in (layer_walks.get(layer_name, {}).get("ambiguous_edge_decisions") or [])
    ]
    layer_convergence = {layer_name: (layer_walks.get(layer_name, {}).get("convergence") or {}) for layer_name in ("coarse", "mid", "fine", "chunk")}
    convergence["layers"] = layer_convergence
    return {
        "query_facets": query_facets,
        "entry_nodes": all_entry_nodes,
        "frontier_pops": frontier_pops,
        "frontier_json": all_frontier,
        "path_labels": all_path_labels,
        "convergence": convergence,
        "ambiguous_edge_decisions": all_ambiguous_decisions,
        "results": results,
        "coarse_entries": coarse_entries,
        "mid_entries": mid_entries,
        "fine_entries": fine_entries,
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
        {"layer": "fine", "ids": list((traversal.get("fine_entries") or {}).keys())[:12]},
        {"layer": "chunk", "ids": [item["chunk_id"] for item in results]},
    ]
    conversation_hash = stable_hash({"conversation_state": "none"})
    cache_components = context_graph_cache_key_components(
        knowledge_base_id=knowledge_base_id,
        query=query,
        filters=filters,
        context_state=context_state,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash=conversation_hash,
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
        fine_cluster_hash=context_state.fine_cluster_hash if context_state else None,
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
        path_labels_json=path_labels,
        convergence_json=traversal.get("convergence") or {},
        edge_distance_protocol_hash=edge_distance_protocol_hash(),
        edge_projection_protocol_hash=edge_projection_protocol_hash(),
        traversal_protocol_hash=traversal_protocol_hash(),
        conversation_state_scope_hash=conversation_hash,
        diagnostics_json={
            "context_graph_state_id": context_state.id if context_state else None,
            "cache_key": cache_key,
            "cache_key_components": cache_components,
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope": agent_operating_envelope(),
            "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
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
    for layer in ("coarse", "mid", "fine", "chunk"):
        walk = layer_walks.get(layer) or {}
        if layer == "coarse":
            input_json = {"entry_nodes": walk.get("entry_nodes") or [], "query_facets": traversal.get("query_facets") or {}}
            output_json = {"accepted_nodes": walk.get("accepted_nodes") or [], "convergence": walk.get("convergence") or {}}
        elif layer == "mid":
            input_json = {"entry_nodes": walk.get("entry_nodes") or [], "coarse_entry_ids": list((traversal.get("coarse_entries") or {}).keys())}
            output_json = {"accepted_nodes": walk.get("accepted_nodes") or [], "convergence": walk.get("convergence") or {}}
        elif layer == "fine":
            input_json = {
                "entry_nodes": walk.get("entry_nodes") or [],
                "mid_entry_ids": list((traversal.get("mid_entries") or {}).keys()),
                "query_rq_path": (query_rq or {}).get("rq_path") or [],
            }
            output_json = {"accepted_nodes": walk.get("accepted_nodes") or [], "candidate_rq": candidate_rq, "convergence": walk.get("convergence") or {}}
        else:
            input_json = {"entry_nodes": walk.get("entry_nodes") or [], "query_rq_path": (query_rq or {}).get("rq_path") or []}
            output_json = {
                "accepted_chunks": {item["chunk_id"]: {"score": item["score"], "rq": (item.get("metadata") or {}).get("rq")} for item in results},
                "convergence": walk.get("convergence") or traversal.get("convergence") or {},
            }
        steps.append((layer, "walk_graph_frontier", input_json, output_json, walk))
    for index, (layer, action, input_json, output_json, walk) in enumerate(steps):
        popped = walk.get("frontier_pops") or []
        popped_state = popped[0] if popped else {}
        layer_path_labels = walk.get("path_labels") or []
        expanded_edge_ids = list(dict.fromkeys(edge_id for label in layer_path_labels for edge_id in (label.get("expanded_edge_ids") or [])))
        cycle_reward = max([float(label.get("reward_so_far") or 0.0) for label in layer_path_labels] + [0.0])
        convergence = walk.get("convergence") or {}
        db.add(
            GraphRetrievalStep(
                retrieval_trace_id=trace.id,
                knowledge_base_id=knowledge_base_id,
                step_index=index,
                layer=layer,
                action=action,
                input_json=input_json,
                output_json=output_json,
                score_json={},
                popped_frontier_state_json=popped_state,
                expanded_edge_ids_json=expanded_edge_ids,
                dominance_pruned_count=int(convergence.get("dominance_pruned_count") or 0),
                cycle_reward=cycle_reward,
                ambiguous_edge_decisions_json=walk.get("ambiguous_edge_decisions") or [],
                stop_reason=str(convergence.get("reason") or ""),
                diagnostics_json={
                    "traversal_protocol": TRAVERSAL_PROTOCOL_VERSION,
                    "scores_json_retired_as_primary_audit": True,
                    "frontier_json": walk.get("frontier_json") or [],
                    "path_labels": layer_path_labels,
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
    from app.services.strategy_profiles import active_profile_json

    token_budget = token_budget or int(get_settings().context_package_token_budget or 2400)
    hit_ids = [item["chunk_id"] for item in results]
    selected_ids: list[str] = []
    bridge_ids: set[str] = set()
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
    traversal_by_chunk = {
        item["chunk_id"]: (item.get("metadata") or {}).get("traversal") or {}
        for item in results
    }

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
        role = "hit" if chunk.id in hit_ids else "bridge" if chunk.id in bridge_ids else "restored_context"
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
        profile_hash=(active_profile_json(db, knowledge_base_id) or {}).get("profile_hash"),
        citation_spans_json=[],
        diagnostics_json={
            "context_restoration_protocol": "previous_next_structure_bridge_v1",
            "runtime_settings_hash": runtime_settings_state_hash(),
            "profile_hash": (active_profile_json(db, knowledge_base_id) or {}).get("profile_hash"),
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
    next_index = (
        db.scalar(select(func.max(GraphRetrievalStep.step_index)).where(GraphRetrievalStep.retrieval_trace_id == trace.id))
        or -1
    ) + 1
    db.add(
        GraphRetrievalStep(
            retrieval_trace_id=trace.id,
            knowledge_base_id=knowledge_base_id,
            step_index=next_index,
            layer="structure",
            action="restore_context_package",
            input_json={"hit_chunk_ids": hit_ids, "token_budget": token_budget},
            output_json={
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
            },
            score_json={},
            popped_frontier_state_json={},
            expanded_edge_ids_json=graph_path_ids,
            dominance_pruned_count=int((trace.convergence_json or {}).get("dominance_pruned_count") or 0),
            cycle_reward=cycle_convergence_score,
            ambiguous_edge_decisions_json=trace.convergence_json.get("ambiguous_edge_decisions", []) if isinstance(trace.convergence_json, dict) else [],
            stop_reason="context_package_built",
            diagnostics_json={
                **(package.diagnostics_json or {}),
                "token_count": token_count,
                "token_budget": token_budget,
                "scores_json_retired_as_primary_audit": True,
            },
        )
    )
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
            "nodes": counts.get("active_chunks", 0) + counts.get("fine_clusters", 0),
            "edges": counts.get("chunk_relation_edges", 0) + counts.get("fine_cluster_memberships", 0) + counts.get("fine_cluster_edges", 0),
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
        "fine_clusters": (
            db.scalar(select(func.count(FineCluster.id)).where(FineCluster.graph_state_id == relation_state_id, FineCluster.state == "active"))
            if relation_state_id
            else 0
        ) or 0,
        "fine_cluster_memberships": (
            db.scalar(
                select(func.count(FineClusterMembership.id))
                .join(FineCluster, FineClusterMembership.fine_cluster_id == FineCluster.id)
                .where(FineCluster.graph_state_id == relation_state_id)
            )
            if relation_state_id
            else 0
        ) or 0,
        "fine_cluster_edges": (
            db.scalar(select(func.count(FineClusterEdge.id)).where(FineClusterEdge.graph_state_id == relation_state_id))
            if relation_state_id
            else 0
        ) or 0,
        "rq_clusters": (
            db.scalar(
                select(func.count(FineCluster.id)).where(
                    FineCluster.graph_state_id == relation_state_id,
                    FineCluster.state == "active",
                    FineCluster.rq_level.is_not(None),
                )
            )
            if relation_state_id
            else 0
        ) or 0,
        "rq_memberships": (
            db.scalar(
                select(func.count(FineClusterMembership.id))
                .join(FineCluster, FineClusterMembership.fine_cluster_id == FineCluster.id)
                .where(FineCluster.graph_state_id == relation_state_id, FineClusterMembership.residual_norm.is_not(None))
            )
            if relation_state_id
            else 0
        ) or 0,
        "rq_edges": (
            db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id, ChunkRelationEdge.edge_type.like("rq_%")))
            if relation_state_id
            else 0
        ) or 0,
        "rq_cluster_edges": (
            db.scalar(select(func.count(FineClusterEdge.id)).where(FineClusterEdge.graph_state_id == relation_state_id, FineClusterEdge.edge_type.like("rq_%")))
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
        cluster_rows: list[FineCluster] = []
        cluster_edge_rows: list[FineClusterEdge] = []
        membership_edge_rows: list[FineClusterMembership] = []
        rq_memberships: dict[str, FineClusterMembership] = {}
        if relation_state:
            cluster_rows = list(
                db.scalars(
                    select(FineCluster)
                    .where(FineCluster.graph_state_id == relation_state.id, FineCluster.state == "active")
                    .order_by(FineCluster.rq_level.is_(None).asc(), FineCluster.rq_level.asc(), FineCluster.cluster_key.asc())
                    .limit(limit)
                ).all()
            )
            cluster_ids = [cluster.id for cluster in cluster_rows]
            if cluster_ids:
                cluster_edge_rows = list(
                    db.scalars(
                        select(FineClusterEdge)
                        .where(
                            FineClusterEdge.graph_state_id == relation_state.id,
                            FineClusterEdge.source_cluster_id.in_(cluster_ids),
                            FineClusterEdge.target_cluster_id.in_(cluster_ids),
                        )
                        .limit(limit * 2)
                    ).all()
                )
            if chunk_rows and cluster_ids:
                membership_edge_rows = list(
                    db.scalars(
                        select(FineClusterMembership)
                        .where(
                            FineClusterMembership.chunk_id.in_([chunk.id for chunk in chunk_rows]),
                            FineClusterMembership.fine_cluster_id.in_(cluster_ids),
                        )
                        .limit(limit * 2)
                    ).all()
                )
        if relation_state and chunk_rows:
            rows = db.scalars(
                select(FineClusterMembership)
                .join(FineCluster, FineClusterMembership.fine_cluster_id == FineCluster.id)
                .where(
                    FineCluster.graph_state_id == relation_state.id,
                    FineClusterMembership.chunk_id.in_([chunk.id for chunk in chunk_rows]),
                )
            ).all()
            for row in rows:
                if row.rq_path and (row.chunk_id not in rq_memberships or row.membership_reason == "rq_leaf"):
                    rq_memberships[row.chunk_id] = row
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
                    "rq_path": chunk.rq_path or (rq_memberships.get(chunk.id).rq_path if rq_memberships.get(chunk.id) else []),
                    "residual_norm": (
                        chunk.rq_residual_norm
                        if chunk.rq_residual_norm is not None
                        else (rq_memberships.get(chunk.id).residual_norm if rq_memberships.get(chunk.id) else None)
                    ),
                },
            }
            for chunk in chunk_rows
        ]
        cluster_nodes = [
            {
                "id": cluster.id,
                "label": cluster.label,
                "type": "fine_cluster",
                "name": cluster.label,
                "category": cluster.node_type or ("rq_prefix" if cluster.rq_level is not None else "fine_cluster"),
                "snippet": f"{len(cluster.support_chunk_ids_json or [])} support chunks",
                "metadata": {
                    "cluster_key": cluster.cluster_key,
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
                "target": row.fine_cluster_id,
                "label": row.membership_reason,
                "type": "fine_cluster_membership",
                "weight": row.membership_score,
                "category": "rq_membership" if row.rq_path else "fine_cluster_membership",
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
        cluster_edges = [
            {
                "id": edge.id,
                "source": edge.source_cluster_id,
                "target": edge.target_cluster_id,
                "label": edge.edge_type,
                "type": edge.edge_type,
                "weight": edge.weight,
                "distance": edge.distance,
                "raw_strength": edge.raw_strength,
                "category": edge.edge_type,
                "metadata": {
                    "distance": edge.distance,
                    "raw_strength": edge.raw_strength,
                    "raw_strength_summary": edge.raw_strength_summary_json or {},
                    "support_chunk_ids": edge.support_chunk_ids_json or [],
                    "support_chunk_edge_ids": edge.support_chunk_edge_ids_json or [],
                    "source_algorithm": edge.source_algorithm,
                    "protocol_version": edge.protocol_version,
                    **(edge.diagnostics_json or {}),
                },
            }
            for edge in cluster_edge_rows
        ]
        edges = chunk_relation_edges + membership_edges + cluster_edges
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
                        "support_fine_edge_ids": edge.support_fine_edge_ids_json or [],
                        "support_fine_node_ids": edge.support_fine_node_ids_json or [],
                        "support_chunk_ids": edge.support_chunk_ids_json or [],
                        "support_chunk_edge_ids": edge.support_relation_edge_ids_json or [],
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
                        "support_fine_edge_ids": edge.support_fine_edge_ids_json or [],
                        "support_chunk_ids": edge.support_chunk_ids_json or [],
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
