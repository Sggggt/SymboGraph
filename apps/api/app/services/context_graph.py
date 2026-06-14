from __future__ import annotations

import asyncio
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
from app.services.embeddings import ChatProvider, EmbeddingProvider, is_degraded_mode
from app.services.parsers import ParsedSection
from app.services.vector_store import VectorStore


RELATION_PROTOCOL_VERSION = "chunk_relation_graph_rq_v2"
MID_CONCEPT_PROMPT_VERSION = "mid_concept_definition_v1"
COARSE_CONCEPT_PROMPT_VERSION = "coarse_concept_definition_v1"
CONTEXT_GRAPH_PROTOCOL_VERSION = "context_graph_v1"
ANSWER_PROMPT_PROTOCOL_VERSION = "context_graph_answer_v1"


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
        "coarse_activation_budget": int(settings.agent_coarse_activation_budget),
        "coarse_jump_budget": int(settings.agent_coarse_jump_budget),
        "mid_activation_budget": int(settings.agent_mid_activation_budget),
        "mid_expansion_radius_cap": int(settings.agent_mid_expansion_radius_cap),
        "fine_cluster_budget": int(settings.agent_fine_cluster_budget),
        "chunk_candidate_budget": int(settings.agent_chunk_candidate_budget),
        "structure_restore_budget": int(settings.agent_structure_restore_budget),
        "context_package_token_budget": int(settings.context_package_token_budget),
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
    previous_section: ChunkStructureNode | None = None
    for item in prepared.section_offsets:
        section_node = ChunkStructureNode(
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="section",
            parent_id=document_node.id,
            previous_sibling_id=previous_section.id if previous_section else None,
            depth=1,
            title=str(item.get("section_path") or item.get("title") or document.title),
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            page_number=item.get("page_number"),
            path=f"{document.title} / {item.get('section_path') or item.get('title')}",
            layout_json=item.get("metadata") or {},
        )
        db.add(section_node)
        db.flush()
        if previous_section is not None:
            previous_section.next_sibling_id = section_node.id
            db.add(
                ChunkStructureEdge(
                    knowledge_base_id=knowledge_base.id,
                    document_version_id=version.id,
                    source_node_id=previous_section.id,
                    target_node_id=section_node.id,
                    edge_type="prev_next",
                    weight=1.0,
                )
            )
        db.add(
            ChunkStructureEdge(
                knowledge_base_id=knowledge_base.id,
                document_version_id=version.id,
                source_node_id=document_node.id,
                target_node_id=section_node.id,
                edge_type="parent_child",
                weight=1.0,
            )
        )
        if item.get("page_number") is not None:
            db.add(
                ChunkStructureEdge(
                    knowledge_base_id=knowledge_base.id,
                    document_version_id=version.id,
                    source_node_id=section_node.id,
                    target_node_id=section_node.id,
                    edge_type="same_page",
                    weight=1.0,
                    metadata_json={"page_number": item.get("page_number")},
                )
            )
        nodes.append(section_node)
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
                mapping_role="parent" if node.node_type == "document" else "overlap",
            )
        )


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
    await VectorStore(knowledge_base_name=knowledge_base.name, collection_name=collection_name).async_upsert(points)
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
    relation_state = build_chunk_relation_graph(db, knowledge_base_id, chunks)
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


def build_chunk_relation_graph(db: Session, knowledge_base_id: str, chunks: list[Chunk]) -> ChunkRelationGraphState:
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
    add_relation_edges(db, graph_state, chunks, vectors, edges)
    clusters = build_fine_clusters(db, graph_state, chunks, vectors, edges)
    build_fine_cluster_edges(db, graph_state, clusters)
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
    normalized_weight = round(float(weight), 6)
    if key in edges:
        edge = edges[key]
        edge.weight = max(edge.weight, normalized_weight)
        edge.confidence = max(edge.confidence, max(0.0, min(1.0, normalized_weight)))
        edge.is_bridge = edge.is_bridge or is_bridge
        edge.features_json = {**(edge.features_json or {}), **(features or {})}
        edge.protocol_version = graph_state.relation_protocol_version
        edge.source_algorithm = relation_edge_source_algorithm(edge_type)
        edge.graph_state_hash = graph_state.state_hash
        return edge
    edge = ChunkRelationEdge(
        graph_state_id=graph_state.id,
        knowledge_base_id=graph_state.knowledge_base_id,
        source_chunk_id=left,
        target_chunk_id=right,
        edge_type=edge_type,
        weight=normalized_weight,
        confidence=max(0.0, min(1.0, normalized_weight)),
        features_json=features or {},
        support_json={"source": edge_type},
        source_algorithm=relation_edge_source_algorithm(edge_type),
        protocol_version=graph_state.relation_protocol_version,
        graph_state_hash=graph_state.state_hash,
        is_bridge=is_bridge,
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

    bm25_records = {record.chunk_id: record for record in db.scalars(select(BM25Record).where(BM25Record.knowledge_base_id == graph_state.knowledge_base_id)).all()}
    for i, left in enumerate(chunks):
        dense_scores: list[tuple[float, Chunk]] = []
        lexical_scores: list[tuple[float, Chunk]] = []
        left_vector = vectors.get(left.id) or []
        left_terms = set((bm25_records.get(left.id).term_frequencies_json or {}).keys()) if bm25_records.get(left.id) else set()
        for right in chunks[i + 1 :]:
            right_vector = vectors.get(right.id) or []
            if left_vector and right_vector:
                dense = cosine_similarity(left_vector, right_vector)
                dense_scores.append((dense, right))
            right_terms = set((bm25_records.get(right.id).term_frequencies_json or {}).keys()) if bm25_records.get(right.id) else set()
            if left_terms and right_terms:
                overlap = len(left_terms.intersection(right_terms)) / max(len(left_terms.union(right_terms)), 1)
                lexical_scores.append((overlap, right))
        for score, right in sorted(dense_scores, key=lambda item: item[0], reverse=True)[:5]:
            if score > 0.3:
                add(left, right, "dense_knn", score, {"cosine": round(score, 6)}, is_bridge=(left.section_path != right.section_path and score > 0.52))
        for score, right in sorted(lexical_scores, key=lambda item: item[0], reverse=True)[:5]:
            if score > 0.08:
                add(left, right, "bm25_overlap", min(1.0, score * 3.0), {"term_jaccard": round(score, 6)}, is_bridge=(left.section_path != right.section_path and score > 0.18))
    db.flush()


def _sliding_pairs(chunks: list[Chunk], *, max_pairs: int) -> list[tuple[Chunk, Chunk]]:
    pairs: list[tuple[Chunk, Chunk]] = []
    ordered = sorted(chunks, key=lambda item: item.chunk_index)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 : index + 3]:
            pairs.append((left, right))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_fine_clusters(
    db: Session,
    graph_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
) -> list[FineCluster]:
    by_label: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        label = concept_label_from_text(chunk.section_path or chunk.text)
        by_label[label].append(chunk)
    if len(by_label) == len(chunks) and len(chunks) > 4:
        by_label = defaultdict(list)
        for index, chunk in enumerate(chunks):
            by_label[f"local cluster {index // 4 + 1}"].append(chunk)
    edge_bridge_chunks = {edge.source_chunk_id for edge in edges.values() if edge.is_bridge} | {edge.target_chunk_id for edge in edges.values() if edge.is_bridge}
    clusters: list[FineCluster] = []
    for label, members in sorted(by_label.items(), key=lambda item: item[0]):
        member_vectors = [vectors.get(chunk.id) for chunk in members if vectors.get(chunk.id)]
        centroid = _centroid([vector for vector in member_vectors if vector])
        representatives = sorted(members, key=lambda item: len(item.text), reverse=True)[:3]
        cluster = FineCluster(
            graph_state_id=graph_state.id,
            knowledge_base_id=graph_state.knowledge_base_id,
            cluster_key=stable_hash({"label": label, "chunks": [chunk.id for chunk in members]})[:24],
            label=label,
            centroid_json=centroid,
            representative_chunk_ids_json=[chunk.id for chunk in representatives],
            support_chunk_ids_json=[chunk.id for chunk in members],
            bridge_chunk_ids_json=[chunk.id for chunk in members if chunk.id in edge_bridge_chunks],
            stats_json={"member_count": len(members)},
        )
        db.add(cluster)
        db.flush()
        clusters.append(cluster)
        for chunk in members:
            own_score = 1.0
            db.add(FineClusterMembership(fine_cluster_id=cluster.id, chunk_id=chunk.id, membership_score=own_score, membership_reason="section_or_term"))
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
                            membership_reason="bridge_fuzzy_membership",
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
                    membership_reason="rq_prefix" if level < len(encoded["rq_path"]) else "rq_leaf",
                    rq_path=encoded["rq_path"],
                    residual_norm=float(encoded["residual_norm"]),
                    diagnostics_json={
                        "residual_vector": encoded["residual_vector"],
                        "reconstructed_vector": encoded["reconstructed_vector"],
                        "rq_level": level,
                        "rq_path_prefix": list(prefix),
                    },
                )
            )

    add_rq_cluster_edges(db, graph_state, rq_clusters_by_key, base_clusters, bridge_chunk_ids)
    add_rq_relation_edges(db, graph_state, chunk_by_id, assignments, edges)


def add_rq_cluster_edges(
    db: Session,
    graph_state: ChunkRelationGraphState,
    rq_clusters_by_key: dict[tuple[int, tuple[int, ...]], FineCluster],
    base_clusters: list[FineCluster],
    bridge_chunk_ids: set[str],
) -> None:
    seen: set[tuple[str, str, str]] = set()
    added_types: set[str] = set()

    def add_edge(source: FineCluster, target: FineCluster, edge_type: str, weight: float, support_ids: list[str], diagnostics: dict[str, Any]) -> None:
        if source.id == target.id:
            return
        left, right = (source, target) if source.id <= target.id else (target, source)
        key = (left.id, right.id, edge_type)
        if key in seen:
            return
        seen.add(key)
        added_types.add(edge_type)
        db.add(
            FineClusterEdge(
                graph_state_id=graph_state.id,
                source_cluster_id=left.id,
                target_cluster_id=right.id,
                edge_type=edge_type,
                weight=round(max(0.0, min(1.0, float(weight))), 6),
                support_chunk_ids_json=list(dict.fromkeys(support_ids))[:16],
                diagnostics_json={
                    "source_algorithm": "rq_kmeans",
                    "protocol_version": graph_state.relation_protocol_version,
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


def build_fine_cluster_edges(db: Session, graph_state: ChunkRelationGraphState, clusters: list[FineCluster]) -> None:
    for index, left in enumerate(clusters):
        for right in clusters[index + 1 :]:
            left_chunks = set(left.support_chunk_ids_json or [])
            right_chunks = set(right.support_chunk_ids_json or [])
            left_bridges = set(left.bridge_chunk_ids_json or [])
            right_bridges = set(right.bridge_chunk_ids_json or [])
            bridge = left_bridges.intersection(right_chunks) or right_bridges.intersection(left_chunks)
            sim = cosine_similarity(left.centroid_json or [], right.centroid_json or [])
            if bridge or sim > 0.45:
                db.add(
                    FineClusterEdge(
                        graph_state_id=graph_state.id,
                        source_cluster_id=left.id,
                        target_cluster_id=right.id,
                        edge_type="overlap_bridge" if bridge else "centroid_near",
                        weight=round(max(sim, 0.42 if bridge else 0.0), 6),
                        support_chunk_ids_json=sorted(bridge),
                        diagnostics_json={"centroid_similarity": round(sim, 6)},
                    )
                )


def relation_graph_stats(chunks: list[Chunk], edges: list[ChunkRelationEdge], clusters: list[FineCluster]) -> dict[str, Any]:
    connected = Counter()
    for edge in edges:
        connected[edge.source_chunk_id] += 1
        connected[edge.target_chunk_id] += 1
    orphan_count = sum(1 for chunk in chunks if connected[chunk.id] == 0)
    singleton_count = sum(1 for cluster in clusters if len(cluster.support_chunk_ids_json or []) <= 1)
    return {
        "chunk_count": len(chunks),
        "edge_count": len(edges),
        "fine_cluster_count": len(clusters),
        "bridge_edges": sum(1 for edge in edges if edge.is_bridge),
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
    stats = {
        "mid_concept_count": len(concepts),
        "grounded_concept_rate": 1.0 if concepts else 0.0,
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
        confidence=float(output.get("confidence") or 0.72),
        llm_audit_json={"prompt_protocol_version": MID_CONCEPT_PROMPT_VERSION, "packet": packet, "raw_output": output},
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
        return fallback_concepts
    by_packet = {str(item.get("packet_id")): item for item in raw_concepts if isinstance(item, dict)}
    return [by_packet.get(str(packet.get("packet_id")), fallback) for packet, fallback in zip(packets, fallback_concepts, strict=False)]


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
    return {
        "packet_id": stable_hash({"cluster": cluster.id, "chunks": chunk_ids})[:16],
        "fine_cluster_ids": [cluster.id],
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
        },
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
    support_by_concept = {concept.id: set(concept.support_chunk_ids_json or []) for concept in concepts}
    for index, left in enumerate(concepts):
        for right in concepts[index + 1 :]:
            left_support = support_by_concept[left.id]
            right_support = support_by_concept[right.id]
            shared = left_support.intersection(right_support)
            lexical = len(set(tokenize_for_bm25(left.definition)).intersection(tokenize_for_bm25(right.definition)))
            score = min(1.0, (len(shared) * 0.3) + (lexical * 0.05))
            if score < 0.1:
                continue
            db.add(
                MidConceptEdge(
                    concept_state_id=state.id,
                    source_concept_id=left.id,
                    target_concept_id=right.id,
                    edge_type="bridge_to" if not shared else "co_occurs_with",
                    weight=round(max(score, 0.32), 6),
                    network_evidence_score=round(max(score, 0.32), 6),
                    llm_confidence=0.7,
                    support_chunk_ids_json=sorted(shared) or sorted((left_support | right_support))[:4],
                    explanation="Concept edge admitted from bottom network evidence and lexical/cluster overlap.",
                    diagnostics_json={"shared_support_chunks": len(shared), "lexical_overlap_terms": lexical},
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
    communities = coarse_communities(mid_concepts)
    coarse_concepts: list[CoarseConcept] = []
    for index, community in enumerate(communities, start=1):
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
            "grounding_hash": stable_hash([concept.id for concept in community]),
        }
        output = await define_coarse_concept(packet)
        mid_ids = [concept.id for concept in community]
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
            confidence=float(output.get("confidence") or 0.72),
            llm_audit_json={"prompt_protocol_version": COARSE_CONCEPT_PROMPT_VERSION, "packet": packet, "raw_output": output},
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
            db.add(
                CoarseConceptEdge(
                    coarse_state_id=state.id,
                    source_concept_id=left.id,
                    target_concept_id=right.id,
                    edge_type="bridge_to",
                    weight=0.35,
                    support_mid_concept_ids_json=(left.bridge_mid_concept_ids_json or [])[:2] + (right.bridge_mid_concept_ids_json or [])[:2],
                    explanation="Weak tie retained between coarse communities for multi-hop retrieval.",
                )
            )
    stats = {
        "coarse_concept_count": len(coarse_concepts),
        "mid_concept_count": len(mid_concepts),
        "bridge_concept_count": sum(len(item.bridge_mid_concept_ids_json or []) for item in coarse_concepts),
        "singleton_rate": round(sum(1 for item in coarse_concepts if len(item.included_mid_concept_ids_json or []) <= 1) / max(len(coarse_concepts), 1), 6),
    }
    community_diagnostics = coarse_community_diagnostics(mid_concepts, mid_edges, communities)
    state.stats_json = stats
    state.diagnostics_json = {
        "community_detection": "bridge_aware_label_bucket_v1",
        "connected_components_used_as_final": False,
        **community_diagnostics,
        "bridge_density": round(stats["bridge_concept_count"] / max(stats["mid_concept_count"], 1), 6),
    }
    state.state_hash = stable_hash({"coarse": [concept.id for concept in coarse_concepts], "stats": stats})
    return state


def coarse_communities(mid_concepts: list[MidConcept]) -> list[list[MidConcept]]:
    if not mid_concepts:
        return []
    if len(mid_concepts) <= 3:
        return [mid_concepts]
    buckets: dict[str, list[MidConcept]] = defaultdict(list)
    for concept in mid_concepts:
        token = (tokenize_for_bm25(concept.canonical_label) or ["general"])[0]
        buckets[token[:1].lower()].append(concept)
    communities = [items for items in buckets.values() if items]
    if len(communities) == len(mid_concepts):
        communities = [mid_concepts[index : index + 3] for index in range(0, len(mid_concepts), 3)]
    return communities


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
    weighted_edges = [
        (edge.source_concept_id, edge.target_concept_id, max(float(edge.weight or 0.0), 0.0))
        for edge in mid_edges
        if edge.source_concept_id in concept_ids and edge.target_concept_id in concept_ids
    ]
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
    return await ChatProvider().classify_json(system_prompt=system, user_prompt=str(packet), fallback=fallback)


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
        trace = RetrievalTrace(knowledge_base_id=knowledge_base_id, query=query, filters_json=filters.model_dump(), result_chunk_ids_json=[], diagnostics_json={"reason": "no_active_chunks"})
        db.add(trace)
        db.flush()
        return LayeredSearchResult([], trace, {"retrieval_pipeline": "layered_context_graph", "reason": "no_active_chunks"})
    context_state = latest_context_graph_state(db, knowledge_base_id)
    relation_state = latest_relation_state(db, knowledge_base_id)
    query_vector = (await EmbeddingProvider().embed_texts([query], text_type="query"))[0]
    query_rq = encode_query_rq(relation_state, query_vector)
    query_terms = tokenize_for_bm25(query)
    candidate_scores: dict[str, dict[str, float]] = defaultdict(dict)
    candidate_metadata: dict[str, dict[str, Any]] = defaultdict(dict)
    coarse_hits = coarse_activation(db, knowledge_base_id, query_terms)
    mid_hits = mid_activation(db, knowledge_base_id, query_terms, coarse_hits)
    fine_hits = fine_activation(db, knowledge_base_id, query_vector, mid_hits)
    vector_hits = vector_activation(db, knowledge_base_id, query_vector, filters)
    lexical_hits = lexical_activation(db, knowledge_base_id, query_terms)

    for chunk_id, score in vector_hits.items():
        candidate_scores[chunk_id]["dense"] = score
    for chunk_id, score in lexical_hits.items():
        candidate_scores[chunk_id]["bm25"] = score
    for fine_id, score in fine_hits.items():
        member_rows = db.scalars(select(FineClusterMembership).where(FineClusterMembership.fine_cluster_id == fine_id)).all()
        for row in member_rows:
            candidate_scores[row.chunk_id]["fine_cluster"] = max(candidate_scores[row.chunk_id].get("fine_cluster", 0.0), score * row.membership_score)
            candidate_metadata[row.chunk_id].setdefault("fine_cluster_ids", []).append(fine_id)
            if query_rq and row.rq_path:
                rq_diag = rq_candidate_score(query_rq, row)
                candidate_scores[row.chunk_id]["rq_fine"] = max(candidate_scores[row.chunk_id].get("rq_fine", 0.0), rq_diag["rq_score"])
                candidate_metadata[row.chunk_id]["rq"] = rq_diag
    for concept_id, score in mid_hits.items():
        concept = db.get(MidConcept, concept_id)
        if concept is None:
            continue
        for chunk_id in concept.support_chunk_ids_json or []:
            candidate_scores[chunk_id]["mid_concept"] = max(candidate_scores[chunk_id].get("mid_concept", 0.0), score)
            candidate_metadata[chunk_id].setdefault("mid_concept_ids", []).append(concept_id)
    for concept_id, score in coarse_hits.items():
        concept = db.get(CoarseConcept, concept_id)
        if concept is None:
            continue
        mid_ids = concept.included_mid_concept_ids_json or []
        mids = db.scalars(select(MidConcept).where(MidConcept.id.in_(mid_ids))).all() if mid_ids else []
        for mid in mids:
            for chunk_id in mid.support_chunk_ids_json or []:
                candidate_scores[chunk_id]["coarse_concept"] = max(candidate_scores[chunk_id].get("coarse_concept", 0.0), score)
                candidate_metadata[chunk_id].setdefault("coarse_concept_ids", []).append(concept_id)

    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    results: list[dict[str, Any]] = []
    for chunk_id, scores in candidate_scores.items():
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None or not passes_filters(db, chunk, filters):
            continue
        graph_score = graph_path_score(db, chunk_id)
        scores["graph_path"] = graph_score
        scores["structure"] = 0.12 if chunk.previous_chunk_id or chunk.next_chunk_id else 0.0
        scores["bridge_bonus"] = bridge_bonus(db, chunk_id)
        scores["redundancy_penalty"] = 0.0
        rq_drift = float((candidate_metadata.get(chunk_id) or {}).get("rq", {}).get("rq_drift_penalty", 0.0))
        scores["drift_penalty"] = (0.0 if context_state else 0.08) + rq_drift
        total = (
            0.30 * scores.get("dense", 0.0)
            + 0.22 * scores.get("bm25", 0.0)
            + 0.12 * scores.get("fine_cluster", 0.0)
            + 0.13 * scores.get("mid_concept", 0.0)
            + 0.08 * scores.get("coarse_concept", 0.0)
            + 0.07 * scores.get("graph_path", 0.0)
            + 0.07 * scores.get("rq_fine", 0.0)
            + 0.03 * scores.get("structure", 0.0)
            + 0.04 * scores.get("bridge_bonus", 0.0)
            - 0.02 * scores.get("drift_penalty", 0.0)
        )
        results.append(search_payload_for_chunk(db, chunk, total, scores, candidate_metadata.get(chunk_id) or {}))
    results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    trace = write_retrieval_trace(db, knowledge_base_id, query, filters, results, context_state, coarse_hits, mid_hits, fine_hits, query_rq)
    for item in results:
        item["metadata"]["retrieval_trace_id"] = trace.id
        for citation in item["citations"]:
            citation["retrieval_trace_id"] = trace.id
    audit = {
        "retrieval_pipeline": "layered_context_graph",
        "retrieval_trace_id": trace.id,
        "context_graph_state_id": context_state.id if context_state else None,
        "degraded_mode": is_degraded_mode(),
        "coarse_hits": len(coarse_hits),
        "mid_hits": len(mid_hits),
        "fine_hits": len(fine_hits),
        "query_rq_path": query_rq.get("rq_path") if query_rq else [],
    }
    return LayeredSearchResult(results, trace, audit)


def coarse_activation(db: Session, knowledge_base_id: str, query_terms: list[str]) -> dict[str, float]:
    state = latest_coarse_state(db, knowledge_base_id)
    if state is None:
        return {}
    concepts = db.scalars(select(CoarseConcept).where(CoarseConcept.coarse_state_id == state.id, CoarseConcept.state == "active")).all()
    return _text_activation({concept.id: f"{concept.canonical_label} {concept.definition}" for concept in concepts}, query_terms, top_n=8)


def mid_activation(db: Session, knowledge_base_id: str, query_terms: list[str], coarse_hits: dict[str, float]) -> dict[str, float]:
    state = latest_mid_state(db, knowledge_base_id)
    if state is None:
        return {}
    concepts = list(db.scalars(select(MidConcept).where(MidConcept.concept_state_id == state.id, MidConcept.state == "active")).all())
    scores = _text_activation({concept.id: f"{concept.canonical_label} {concept.definition}" for concept in concepts}, query_terms, top_n=16)
    coarse_ids = set(coarse_hits)
    if coarse_ids:
        coarse_concepts = db.scalars(select(CoarseConcept).where(CoarseConcept.id.in_(coarse_ids))).all()
        boosted_mid_ids = {mid_id for coarse in coarse_concepts for mid_id in (coarse.included_mid_concept_ids_json or [])}
        for concept in concepts:
            if concept.id in boosted_mid_ids:
                scores[concept.id] = max(scores.get(concept.id, 0.0), 0.45)
    return scores


def fine_activation(db: Session, knowledge_base_id: str, query_vector: list[float], mid_hits: dict[str, float]) -> dict[str, float]:
    relation_state = latest_relation_state(db, knowledge_base_id)
    if relation_state is None:
        return {}
    query_rq = encode_query_rq(relation_state, query_vector)
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
    for concept_id, score in mid_hits.items():
        rows = db.scalars(select(MidConceptMembership).where(MidConceptMembership.mid_concept_id == concept_id)).all()
        for row in rows:
            scores[row.fine_cluster_id] = max(scores.get(row.fine_cluster_id, 0.0), score * row.membership_score)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:16])


def vector_activation(db: Session, knowledge_base_id: str, query_vector: list[float], filters: SearchFilters) -> dict[str, float]:
    records = list(db.scalars(select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base_id, VectorRecord.vector_status == "ready")).all())
    scored: list[tuple[str, float]] = []
    for record in records:
        vector = [float(value) for value in ((record.diagnostics_json or {}).get("embedding_vector") or []) if isinstance(value, (int, float))]
        if vector:
            scored.append((record.chunk_id, max(0.0, cosine_similarity(query_vector, vector))))
    return dict(sorted(scored, key=lambda item: item[1], reverse=True)[:80])


def lexical_activation(db: Session, knowledge_base_id: str, query_terms: list[str]) -> dict[str, float]:
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


def _text_activation(text_by_id: dict[str, str], query_terms: list[str], *, top_n: int) -> dict[str, float]:
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


def graph_path_score(db: Session, chunk_id: str) -> float:
    relation_state = db.scalar(select(ChunkRelationGraphState).order_by(ChunkRelationGraphState.created_at.desc()))
    if relation_state is None:
        return 0.0
    degree = db.scalar(
        select(func.count(ChunkRelationEdge.id)).where(
            ChunkRelationEdge.graph_state_id == relation_state.id,
            (ChunkRelationEdge.source_chunk_id == chunk_id) | (ChunkRelationEdge.target_chunk_id == chunk_id),
        )
    ) or 0
    return min(1.0, float(degree) / 8.0)


def bridge_bonus(db: Session, chunk_id: str) -> float:
    relation_state = db.scalar(select(ChunkRelationGraphState).order_by(ChunkRelationGraphState.created_at.desc()))
    if relation_state is None:
        return 0.0
    count = db.scalar(
        select(func.count(ChunkRelationEdge.id)).where(
            ChunkRelationEdge.graph_state_id == relation_state.id,
            ChunkRelationEdge.is_bridge.is_(True),
            (ChunkRelationEdge.source_chunk_id == chunk_id) | (ChunkRelationEdge.target_chunk_id == chunk_id),
        )
    ) or 0
    return min(1.0, float(count) / 4.0)


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


def search_payload_for_chunk(db: Session, chunk: Chunk, score: float, scores: dict[str, float], metadata: dict[str, Any]) -> dict[str, Any]:
    document = db.get(Document, chunk.document_id)
    snippet = re.sub(r"\s+", " ", chunk.text).strip()[:280]
    citation = {
        "chunk_id": chunk.id,
        "document_id": document.id if document else chunk.document_id,
        "document_title": document.title if document else "",
        "source_path": document.source_path if document else "",
        "partition": (document.tags or [None])[0] if document and document.tags else None,
        "section": chunk.section_path,
        "page_number": chunk.page_start,
        "snippet": snippet,
        "source_span": {
            "document_version_id": chunk.document_version_id,
            "char_span": [chunk.char_start, chunk.char_end],
            "page_range": [chunk.page_start, chunk.page_end],
            "section_path": chunk.section_path,
        },
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
    coarse_hits: dict[str, float],
    mid_hits: dict[str, float],
    fine_hits: dict[str, float],
    query_rq: dict[str, Any] | None = None,
) -> RetrievalTrace:
    candidate_rq = {
        item["chunk_id"]: (item.get("metadata") or {}).get("rq")
        for item in results
        if (item.get("metadata") or {}).get("rq")
    }
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
        concept_path_json=[
            {"layer": "coarse", "ids": list(coarse_hits)[:8]},
            {"layer": "mid", "ids": list(mid_hits)[:12]},
            {"layer": "fine", "ids": list(fine_hits)[:12]},
        ],
        scores_json={item["chunk_id"]: (item.get("metadata") or {}).get("scores", {}) for item in results},
        diagnostics_json={
            "context_graph_state_id": context_state.id if context_state else None,
            "runtime_settings_hash": runtime_settings_state_hash(),
            "agent_operating_envelope": agent_operating_envelope(),
            "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
            "rq": {
                "query_rq_path": (query_rq or {}).get("rq_path") or [],
                "query_residual_norm": (query_rq or {}).get("residual_norm"),
                "index_protocol": (query_rq or {}).get("index_protocol"),
            },
        },
    )
    db.add(trace)
    db.flush()
    steps = [
        ("coarse", "activate_coarse_concepts", {}, coarse_hits),
        ("mid", "route_mid_concepts", {"coarse_hits": list(coarse_hits)}, mid_hits),
        (
            "fine",
            "route_fine_clusters",
            {"mid_hits": list(mid_hits), "query_rq_path": (query_rq or {}).get("rq_path") or []},
            {"fine_hits": fine_hits, "candidate_rq": candidate_rq},
        ),
        (
            "chunk",
            "recall_chunks",
            {"fine_hits": list(fine_hits), "query_rq_path": (query_rq or {}).get("rq_path") or []},
            {item["chunk_id"]: {"score": item["score"], "rq": (item.get("metadata") or {}).get("rq")} for item in results},
        ),
    ]
    for index, (layer, action, input_json, output_json) in enumerate(steps):
        db.add(
            GraphRetrievalStep(
                retrieval_trace_id=trace.id,
                knowledge_base_id=knowledge_base_id,
                step_index=index,
                layer=layer,
                action=action,
                input_json=input_json,
                output_json=output_json,
                score_json=output_json,
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
            .order_by(ChunkRelationEdge.weight.desc())
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
        }
        structure_by_chunk_id[chunk.id] = value
        return value

    package_chunks: list[dict[str, Any]] = []
    token_count = 0
    for chunk_id in selected_ids:
        chunk = chunks_by_id[chunk_id]
        chunk_tokens = rough_token_count(chunk.text)
        if package_chunks and token_count + chunk_tokens > token_budget:
            continue
        document = db.get(Document, chunk.document_id)
        structure = structure_context(chunk)
        package_chunks.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
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
                "role": "hit" if chunk.id in hit_ids else "bridge" if chunk.id in bridge_ids else "restored_context",
            }
        )
        token_count += chunk_tokens
    citation_spans = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "document_version_id": chunks_by_id[item["chunk_id"]].document_version_id,
            "char_span": item["char_span"],
            "page_range": item["page_range"],
            "section_path": item["section_path"],
            "structure_path": item.get("structure_path"),
            "structure_node_ids": item.get("structure_node_ids") or [],
        }
        for item in package_chunks
        if item["chunk_id"] in hit_ids
    ]
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
        token_budget=token_budget,
        token_count=token_count,
        runtime_settings_hash=runtime_settings_state_hash(),
        profile_hash=(active_profile_json(db, knowledge_base_id) or {}).get("profile_hash"),
        citation_spans_json=citation_spans,
        diagnostics_json={
            "context_restoration_protocol": "previous_next_structure_bridge_v1",
            "runtime_settings_hash": runtime_settings_state_hash(),
            "profile_hash": (active_profile_json(db, knowledge_base_id) or {}).get("profile_hash"),
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
            },
            score_json={"token_count": token_count, "token_budget": token_budget},
            diagnostics_json=package.diagnostics_json,
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
                "category": "rq_prefix" if cluster.rq_level is not None else "fine_cluster",
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
                "category": edge.edge_type,
                "is_bridge": edge.is_bridge,
                "metadata": {
                    **(edge.features_json or {}),
                    "source_algorithm": edge.source_algorithm,
                    "protocol_version": edge.protocol_version,
                    "graph_state_hash": edge.graph_state_hash,
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
                "category": edge.edge_type,
                "metadata": {
                    "support_chunk_ids": edge.support_chunk_ids_json or [],
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
                {"source": edge.source_concept_id, "target": edge.target_concept_id, "label": edge.edge_type, "weight": edge.weight, "category": edge.edge_type}
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
                {"source": edge.source_concept_id, "target": edge.target_concept_id, "label": edge.edge_type, "weight": edge.weight, "category": edge.edge_type}
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
    score_totals: Counter[str] = Counter()
    for trace in traces:
        for scores in (trace.scores_json or {}).values():
            if isinstance(scores, dict):
                for key, value in scores.items():
                    try:
                        score_totals[key] += float(value)
                    except (TypeError, ValueError):
                        pass
    total = sum(score_totals.values()) or 1.0
    return {key: round(value / total, 6) for key, value in score_totals.items()}
