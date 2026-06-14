from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Chunk, ContextPackage, Document, IngestionBatch, KnowledgeBase, RetrievalTrace
from app.schemas import SearchFilters
from app.services.context_graph import (
    build_context_package,
    context_graph_stats,
    graph_layer_payload,
    layered_search,
    latest_context_graph_state,
    context_package_to_contexts,
)
from app.services.embeddings import is_degraded_mode
from app.services.ingestion import collect_source_documents, get_job_status, list_knowledge_base_files
from app.services.parsers import derive_partition


TERMINAL_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled", "cancel_failed"}


async def search_chunks_with_audit(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int,
) -> tuple[list[dict], dict]:
    result = await layered_search(db, knowledge_base_id, query, filters, top_k)
    return result.results, result.audit


async def layered_context_search_chunks_with_audit(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int,
    route: str = "multi_hop_research",
) -> tuple[list[dict], dict]:
    result = await layered_search(db, knowledge_base_id, query, filters, top_k)
    audit = {**result.audit, "route": route}
    return result.results, audit


async def search_chunks(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int,
) -> list[dict]:
    return (await search_chunks_with_audit(db, knowledge_base_id, query, filters, top_k))[0]


def get_context_package(db: Session, package_id: str) -> dict | None:
    package = db.get(ContextPackage, package_id)
    if package is None:
        return None
    package_json = package.package_json or {}
    graph_expansion_paths = [
        {"kind": "concept_path", "path": package.concept_path_json or []},
        {"kind": "restored_chunks", "chunk_ids": package.restored_chunk_ids_json or []},
        {"kind": "bridge_chunks", "chunk_ids": package.bridge_chunk_ids_json or []},
        {"kind": "parent_structure_nodes", "node_ids": package.parent_structure_node_ids_json or []},
    ]
    package_hash = hashlib.sha256(
        json.dumps(
            {
                "package": package_json,
                "citation_spans": package.citation_spans_json or [],
                "graph_expansion_paths": graph_expansion_paths,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "id": package.id,
        "knowledge_base_id": package.knowledge_base_id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "package_hash": package_hash,
        "query": package.query,
        "hit_chunk_ids": package.hit_chunk_ids_json or [],
        "restored_chunk_ids": package.restored_chunk_ids_json or [],
        "bridge_chunk_ids": package.bridge_chunk_ids_json or [],
        "parent_structure_node_ids": package.parent_structure_node_ids_json or [],
        "concept_path": package.concept_path_json or [],
        "package": package_json,
        "contexts": context_package_to_contexts(package),
        "token_budget": package.token_budget,
        "token_count": package.token_count,
        "citation_spans": package.citation_spans_json or [],
        "graph_expansion_paths": graph_expansion_paths,
        "diagnostics": package.diagnostics_json or {},
        "created_at": package.created_at,
    }


def get_retrieval_trace_steps(db: Session, trace_id: str) -> dict | None:
    from app.models import GraphRetrievalStep

    trace = db.get(RetrievalTrace, trace_id)
    if trace is None:
        return None
    steps = db.scalars(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == trace_id).order_by(GraphRetrievalStep.step_index.asc())).all()
    return {
        "trace_id": trace.id,
        "query": trace.query,
        "retrieval_mode": trace.retrieval_mode,
        "concept_path": trace.concept_path_json or [],
        "result_chunk_ids": trace.result_chunk_ids_json or [],
        "steps": [
            {
                "id": step.id,
                "step_index": step.step_index,
                "layer": step.layer,
                "action": step.action,
                "input": step.input_json or {},
                "output": step.output_json or {},
                "scores": step.score_json or {},
                "diagnostics": step.diagnostics_json or {},
                "created_at": step.created_at,
            }
            for step in steps
        ],
    }


def get_dashboard_snapshot(db: Session, knowledge_base_id: str, *, include_graph: bool = True) -> dict:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        return empty_dashboard()
    documents = db.scalars(select(Document).where(Document.knowledge_base_id == knowledge_base.id, Document.is_active.is_(True))).all()
    file_items = list_knowledge_base_files(db, knowledge_base.id)
    chunk_count = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
    latest_batch = db.scalar(
        select(IngestionBatch)
        .where(IngestionBatch.knowledge_base_id == knowledge_base.id, IngestionBatch.status.notin_(TERMINAL_BATCH_STATES))
        .order_by(IngestionBatch.created_at.desc())
        .limit(1)
    )
    tree = tree_payload(knowledge_base, file_items)
    graph = graph_layer_payload(db, knowledge_base.id, "chunk-relation") if include_graph else empty_graph("chunk-relation")
    active_profile = knowledge_base.active_profile
    paths = get_settings().knowledge_base_paths_for_name(knowledge_base.name)
    return {
        "knowledge_base": {
            "id": knowledge_base.id,
            "name": knowledge_base.name,
            "description": knowledge_base.description,
            "source_root": str(paths["storage_root"]),
            "storage_root": str(paths["storage_root"]),
            "document_count": len(file_items),
            "chunk_count": chunk_count,
            "current_chunk_version": knowledge_base.current_chunk_version or 0,
            "has_parsed_chunks": chunk_count > 0,
            "can_full_reparse": chunk_count > 0,
            "degraded_mode": is_degraded_mode(),
            "active_profile_id": active_profile.id if active_profile else None,
            "active_profile_name": active_profile.name if active_profile else None,
            "active_profile_hash": active_profile.profile_hash if active_profile else None,
        },
        "tree": tree,
        "graph": graph,
        "batch_status": None if latest_batch is None else summarize_batch_for_dashboard(latest_batch),
        "ingested_document_count": len(documents),
        "chunk_count": chunk_count,
        "graph_relation_count": (graph.get("node_counts") or {}).get("chunk_relation_edges", 0),
        "coverage_by_source_type": dict(Counter(item.get("source_type") or "unknown" for item in file_items)),
        "degraded_mode": is_degraded_mode(),
        "context_graph": context_graph_stats(db, knowledge_base.id),
    }


def empty_dashboard() -> dict:
    return {
        "knowledge_base": {
            "id": "empty",
            "name": "KnowledgeBase Workspace",
            "description": None,
            "source_root": "",
            "storage_root": "",
            "document_count": 0,
            "chunk_count": 0,
            "current_chunk_version": 0,
            "has_parsed_chunks": False,
            "can_full_reparse": False,
            "degraded_mode": is_degraded_mode(),
            "active_profile_id": None,
            "active_profile_name": None,
            "active_profile_hash": None,
        },
        "tree": [],
        "graph": empty_graph("chunk-relation"),
        "batch_status": None,
        "ingested_document_count": 0,
        "chunk_count": 0,
        "graph_relation_count": 0,
        "coverage_by_source_type": {},
        "degraded_mode": is_degraded_mode(),
    }


def empty_graph(layer: str) -> dict:
    return {
        "graph_type": layer,
        "schema_version": "context_graph_v1",
        "view": "overview",
        "nodes": [],
        "edges": [],
        "node_counts": {},
        "edge_counts": {},
        "freshness": {"is_stale": False, "stale_reasons": []},
        "diagnostics": {},
    }


def tree_payload(knowledge_base: KnowledgeBase, file_items: list[dict]) -> list[dict]:
    partition_map: dict[str, list[dict]] = defaultdict(list)
    for item in file_items:
        partition = item.get("partition") or "General"
        partition_map[partition].append(item)
    return [
        {
            "id": f"partition:{partition}",
            "title": partition,
            "type": "partition",
            "children": [
                {"id": item["document_id"] or item["id"], "title": item["title"], "type": "document", "children": []}
                for item in sorted(entries, key=lambda row: row["title"].lower())
            ],
        }
        for partition, entries in sorted(partition_map.items())
    ]


def summarize_batch_for_dashboard(batch: IngestionBatch) -> dict:
    stats = dict(batch.stats or {})
    return {
        "batch_id": batch.id,
        "state": batch.status,
        "trigger_source": batch.trigger_source,
        "source_root": batch.source_root,
        "total_files": batch.total_files,
        "processed_files": batch.processed_files,
        "success_count": batch.success_count,
        "failure_count": batch.failure_count,
        "skipped_count": batch.skipped_count,
        "coverage_by_source_type": stats.get("coverage_by_source_type", {}),
        "errors": stats.get("errors", []),
        "graph_stats": stats.get("graph_stats", {}),
        "phase": stats.get("phase"),
        "parse_committed": bool(stats.get("parse_committed")),
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
    }
