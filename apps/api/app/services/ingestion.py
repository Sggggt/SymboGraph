from __future__ import annotations

import asyncio
import os
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.utils import source_type_from_path
from app.models import (
    Chunk,
    Document,
    DocumentVersion,
    IngestionBatch,
    IngestionJob,
    KnowledgeBase,
    ParseJob,
    SourceFile,
)
from app.services.context_graph import context_graph_stats, rebuild_context_graph, write_chunks_and_structure, write_contextual_indexes
from app.services.embeddings import is_degraded_mode
from app.services.ingestion_logs import emit_ingestion_log
from app.services.parsers import derive_partition, parse_document, sections_to_json
from app.services.runtime_settings import refresh_runtime_settings_if_needed
from app.services.storage import compute_checksum, copy_source_file


ALLOWED_SUFFIXES = {".pdf", ".ipynb", ".md", ".markdown", ".txt", ".docx", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".bmp", ".html", ".htm"}
EXCLUDED_PARTS = {"output", "scripts", ".ipynb_checkpoints", "__pycache__"}
IGNORED_NAMES = {".ds_store"}
TERMINAL_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled", "cancel_failed"}


def current_worker_id() -> str:
    return os.getenv("WORKER_ID") or os.getenv("HOSTNAME") or socket.gethostname()


def exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or f"{exc.__class__.__name__}: {exc!r}"


def should_include_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() in IGNORED_NAMES or path.name.startswith("~$"):
        return False
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    return not any(part.lower() in EXCLUDED_PARTS for part in path.parts)


def collect_source_documents(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = sorted((path for path in root.rglob("*") if should_include_file(path)), key=lambda item: str(item).lower())
    deduped: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        key = (path.name.lower(), compute_checksum(path))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def resolve_knowledge_base(db: Session, knowledge_base_id: str | None = None) -> KnowledgeBase:
    if knowledge_base_id:
        knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    else:
        settings = get_settings()
        knowledge_base = db.scalar(select(KnowledgeBase).where(KnowledgeBase.name == settings.knowledge_base_name))
        if knowledge_base is None:
            knowledge_base = db.scalar(select(KnowledgeBase).order_by(KnowledgeBase.created_at.asc()))
    if knowledge_base is None:
        raise LookupError("Knowledge base not found")
    return knowledge_base


def create_knowledge_base_space(db: Session, name: str, description: str | None = None) -> KnowledgeBase:
    existing = db.scalar(select(KnowledgeBase).where(KnowledgeBase.name == name))
    if existing is not None:
        raise ValueError(f"Knowledge base already exists: {name}")
    paths = get_settings().knowledge_base_paths_for_name(name)
    paths["storage_root"].mkdir(parents=True, exist_ok=True)
    paths["ingestion_root"].mkdir(parents=True, exist_ok=True)
    knowledge_base = KnowledgeBase(name=name, description=description, source_root=str(paths["storage_root"]))
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


def list_knowledge_base_summaries(db: Session) -> list[dict]:
    return [summarize_knowledge_base(db, item) for item in db.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at.asc())).all()]


def summarize_knowledge_base(db: Session, knowledge_base: KnowledgeBase) -> dict:
    paths = get_settings().knowledge_base_paths_for_name(knowledge_base.name)
    document_count = db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base.id, Document.is_active.is_(True))) or 0
    graph_stats = context_graph_stats(db, knowledge_base.id)
    graph_counts = graph_stats.get("counts") or {}
    freshness = graph_stats.get("freshness") or {}
    hashes = graph_stats.get("hashes") or {}
    chunk_count = int(graph_counts.get("chunks") or 0)
    active_chunk_count = int(graph_counts.get("active_chunks") or 0)
    stale_reason = ", ".join(freshness.get("stale_reasons") or []) or None
    return {
        "id": knowledge_base.id,
        "name": knowledge_base.name,
        "description": knowledge_base.description,
        "source_root": str(paths["storage_root"]),
        "storage_root": str(paths["storage_root"]),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "active_chunk_count": active_chunk_count,
        "current_chunk_version": knowledge_base.current_chunk_version or 0,
        "context_graph_state_id": graph_stats.get("context_graph_state_id"),
        "context_graph_hash": hashes.get("context_graph_hash"),
        "stale_reason": stale_reason,
        "has_parsed_chunks": active_chunk_count > 0,
        "can_full_reparse": active_chunk_count > 0,
        "degraded_mode": is_degraded_mode(),
        "active_profile_id": knowledge_base.active_profile_id,
        "active_profile_name": knowledge_base.active_profile.name if knowledge_base.active_profile else None,
        "active_profile_hash": knowledge_base.active_profile.profile_hash if knowledge_base.active_profile else None,
    }


def create_job(
    db: Session,
    *,
    knowledge_base_id: str,
    document_id: str | None,
    trigger_source: str,
    batch_id: str | None = None,
    source_path: str | None = None,
) -> IngestionJob:
    job = IngestionJob(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        trigger_source=trigger_source,
        batch_id=batch_id,
        source_path=source_path,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def register_uploaded_file(db: Session, knowledge_base: KnowledgeBase, stored_path: Path) -> tuple[Document, IngestionJob]:
    checksum = compute_checksum(stored_path)
    document = upsert_document(db, knowledge_base, stored_path, checksum=checksum, source_type=source_type_from_path(stored_path))
    job = create_job(
        db,
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        trigger_source="upload",
        source_path=str(stored_path),
    )
    return document, job


def upsert_document(db: Session, knowledge_base: KnowledgeBase, path: Path, *, checksum: str, source_type: str) -> Document:
    document = db.scalar(select(Document).where(Document.knowledge_base_id == knowledge_base.id, Document.source_path == str(path)))
    partition = derive_partition(path, knowledge_base_name=knowledge_base.name)
    if document is None:
        document = Document(
            knowledge_base_id=knowledge_base.id,
            title=path.stem or path.name,
            source_path=str(path),
            source_type=source_type,
            tags=[partition] if partition else [],
            checksum=checksum,
            is_active=True,
        )
        db.add(document)
        db.flush()
    else:
        document.title = path.stem or path.name
        document.source_type = source_type
        document.checksum = checksum
        document.tags = [partition] if partition else []
        document.is_active = True
    return document


def create_sync_batch(db: Session, knowledge_base_id: str, root: Path, trigger_source: str = "storage") -> IngestionBatch:
    files = collect_source_documents(root)
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base_id,
        trigger_source=trigger_source,
        source_root=str(root),
        total_files=len(files),
        status="queued",
        stats={"file_paths": [str(path) for path in files], "phase": "queued"},
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def create_uploaded_files_batch(db: Session, knowledge_base_id: str, file_paths: list[Path], *, force: bool = False, full_reparse: bool = False) -> IngestionBatch:
    source_root = str(file_paths[0].parent if file_paths else get_settings().data_root)
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base_id,
        trigger_source="upload",
        source_root=source_root,
        total_files=len(file_paths),
        status="queued",
        stats={"file_paths": [str(path) for path in file_paths], "force": force, "full_reparse": full_reparse, "phase": "queued"},
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def create_context_graph_rebuild_batch(db: Session, knowledge_base_id: str, *, layers: list[str] | None = None) -> IngestionBatch:
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base_id,
        trigger_source="graph_rebuild",
        source_root=str(get_settings().data_root),
        total_files=1,
        status="queued",
        stats={
            "phase": "queued",
            "maintenance_task": "context_graph_rebuild",
            "layers": layers or ["chunk-relation", "mid-concepts", "coarse-concepts", "context-graph"],
            "parse_committed": False,
        },
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def mark_batch_task_started(batch_id: str, task_id: str | None, task_name: str) -> None:
    from app.db import SessionLocal

    with SessionLocal() as db:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            return
        task_ids = set((batch.stats or {}).get("batch_task_ids") or [])
        if task_id:
            task_ids.add(task_id)
        batch.worker_id = current_worker_id()
        batch.heartbeat_at = datetime.utcnow()
        batch.stats = {**(batch.stats or {}), "celery_task_id": task_id, "celery_task_name": task_name, "batch_task_ids": sorted(task_ids)}
        db.commit()


def mark_batch_worker_heartbeat(db: Session, batch: IngestionBatch, *, phase: str | None = None) -> None:
    batch.worker_id = current_worker_id()
    batch.heartbeat_at = datetime.utcnow()
    stats = dict(batch.stats or {})
    if phase:
        stats["phase"] = phase
    stats["heartbeat_at"] = batch.heartbeat_at.isoformat()
    batch.stats = stats


def get_batch_status(db: Session, batch_id: str) -> dict | None:
    batch = db.get(IngestionBatch, batch_id)
    if batch is None:
        return None
    return summarize_batch(batch)


def summarize_batch(batch: IngestionBatch | None) -> dict:
    if batch is None:
        return {}
    stats = dict(batch.stats or {})
    context_graph_phase = stats.get("context_graph_phase")
    current_phase = f"context_graph:{context_graph_phase}" if context_graph_phase else stats.get("phase")
    return {
        "batch_id": batch.id,
        "knowledge_base_id": batch.knowledge_base_id,
        "state": batch.status,
        "mode": stats.get("ingestion_execution_mode"),
        "trigger_source": batch.trigger_source,
        "source_root": batch.source_root,
        "total_files": batch.total_files,
        "processed_files": batch.processed_files,
        "success_count": batch.success_count,
        "failure_count": batch.failure_count,
        "skipped_count": batch.skipped_count,
        "current_file": stats.get("current_file"),
        "current_phase": current_phase,
        "stats": stats,
        "coverage_by_source_type": stats.get("coverage_by_source_type", {}),
        "errors": stats.get("errors", []),
        "graph_stats": stats.get("graph_stats", {}),
        "phase": stats.get("phase"),
        "parse_committed": bool(stats.get("parse_committed")),
        "cancellation_status": stats.get("cancellation_status"),
        "cancel_failure_reason": stats.get("cancel_failure_reason"),
        "manual_review_required": bool(stats.get("manual_review_required", False)),
        "celery_task_id": stats.get("celery_task_id"),
        "celery_task_name": stats.get("celery_task_name"),
        "batch_task_ids": stats.get("batch_task_ids", []),
        "batch_worker_ids": [batch.worker_id] if batch.worker_id else [],
        "worker_id": batch.worker_id,
        "heartbeat_at": batch.heartbeat_at,
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
    }


def request_batch_cancel_control(db: Session, batch_id: str, knowledge_base_id: str) -> dict | None:
    batch = db.get(IngestionBatch, batch_id)
    if batch is None:
        return None
    if batch.knowledge_base_id != knowledge_base_id:
        raise PermissionError("Batch belongs to another knowledge base")
    if batch.status not in TERMINAL_STATES:
        batch.status = "cancelled"
        batch.completed_at = datetime.utcnow()
        batch.stats = {**(batch.stats or {}), "cancellation_status": "cancelled_by_control_plane", "phase": "cancelled"}
        for job in batch.jobs:
            if job.status not in TERMINAL_STATES:
                job.status = "cancelled"
        db.commit()
        emit_ingestion_log(batch.id, "batch_cancelled", "Batch cancelled before the next stage boundary", state="cancelled")
    return summarize_batch(batch)


async def ingest_file(
    db: Session,
    source_path: Path,
    *,
    trigger_source: str = "upload",
    existing_job_id: str | None = None,
    batch_id: str | None = None,
    knowledge_base_id: str | None = None,
    rebuild_graph: bool = True,
    force: bool = False,
    target_version: int | None = None,
) -> dict:
    refresh_runtime_settings_if_needed()
    settings = get_settings()
    knowledge_base = resolve_knowledge_base(db, knowledge_base_id)
    path = source_path.resolve()
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise RuntimeError(f"Unsupported file type: {path.suffix}")
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"File not found: {path}")
    checksum = compute_checksum(path)
    source_type = source_type_from_path(path)
    document = upsert_document(db, knowledge_base, path, checksum=checksum, source_type=source_type)
    job = db.get(IngestionJob, existing_job_id) if existing_job_id else None
    if job is None:
        job = create_job(db, knowledge_base_id=knowledge_base.id, document_id=document.id, trigger_source=trigger_source, batch_id=batch_id, source_path=str(path))
    runtime_snapshot = {
        "fixed_chunk_size_tokens": settings.fixed_chunk_size_tokens,
        "fixed_chunk_overlap_tokens": settings.fixed_chunk_overlap_tokens,
        "context_package_token_budget": settings.context_package_token_budget,
    }
    job.status = "parsing"
    job.document_id = document.id
    job.source_path = str(path)
    job.stats = {**(job.stats or {}), "phase": "parsing", "runtime_settings_snapshot": runtime_snapshot}
    db.flush()

    source_file = db.scalar(select(SourceFile).where(SourceFile.knowledge_base_id == knowledge_base.id, SourceFile.source_path == str(path)))
    if source_file is None:
        source_file = SourceFile(
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            source_path=str(path),
            checksum=checksum,
            source_type=source_type,
            size_bytes=path.stat().st_size,
            metadata_json={},
        )
        db.add(source_file)
    else:
        source_file.document_id = document.id
        source_file.checksum = checksum
        source_file.source_type = source_type
        source_file.size_bytes = path.stat().st_size
        source_file.state = "active"
    db.flush()

    existing_active = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
    chunk_version = target_version or knowledge_base.current_chunk_version or 1
    if existing_active == 0 and (knowledge_base.current_chunk_version or 0) == 0:
        chunk_version = 1
    storage_path = copy_source_file(path, knowledge_base.name)
    version_number = chunk_version
    for old_version in db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == document.id, DocumentVersion.is_active.is_(True))).all():
        old_version.is_active = False
    version = DocumentVersion(
        document_id=document.id,
        version=version_number,
        checksum=checksum,
        storage_path=str(storage_path),
        is_active=True,
    )
    db.add(version)
    db.flush()

    parse_job = ParseJob(
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        ingestion_job_id=job.id,
        source_file_id=source_file.id,
        parser_protocol_version="parser_v1",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(parse_job)
    db.flush()
    refresh_runtime_settings_if_needed()
    settings = get_settings()
    parsed_source_type, sections = parse_document(path)
    parse_job.status = "completed"
    parse_job.completed_at = datetime.utcnow()
    parse_job.stats_json = {"section_count": len(sections), "source_type": parsed_source_type}
    parse_job.diagnostics_json = {"sections": sections_to_json(sections)[:20]}
    db.flush()

    job.status = "chunking"
    job.stats = {
        **(job.stats or {}),
        "phase": "chunking",
        "section_count": len(sections),
        "runtime_settings_snapshot": {
            **runtime_snapshot,
            "fixed_chunk_size_tokens": settings.fixed_chunk_size_tokens,
            "fixed_chunk_overlap_tokens": settings.fixed_chunk_overlap_tokens,
        },
    }
    chunks = write_chunks_and_structure(
        db,
        knowledge_base=knowledge_base,
        document=document,
        version=version,
        sections=sections,
        chunk_version=chunk_version,
        chunk_size=settings.fixed_chunk_size_tokens,
        chunk_overlap=settings.fixed_chunk_overlap_tokens,
    )
    db.flush()

    refresh_runtime_settings_if_needed()
    job.status = "embedding"
    job.stats = {**(job.stats or {}), "phase": "embedding", "chunk_count": len(chunks)}
    index_stats = await write_contextual_indexes(db, knowledge_base=knowledge_base, chunks=chunks)
    job.status = "completed"
    job.stats = {**(job.stats or {}), "phase": "indexed", "index_stats": index_stats, "chunk_count": len(chunks)}
    db.flush()
    if rebuild_graph:
        refresh_runtime_settings_if_needed()
        await rebuild_context_graph(db, knowledge_base.id, batch_id=batch_id)
    if knowledge_base.current_chunk_version < chunk_version:
        knowledge_base.current_chunk_version = chunk_version
    db.commit()
    if batch_id:
        emit_ingestion_log(batch_id, "file_indexed", f"{path.name} indexed into {len(chunks)} chunks", source_path=str(path), chunk_count=len(chunks), **index_stats)
    return {"status": "completed", "document_id": document.id, "chunk_count": len(chunks), "source_type": source_type, "stats": {"chunks": len(chunks), **index_stats}}


def finalize_interrupted_batches() -> None:
    from app.db import SessionLocal

    with SessionLocal() as db:
        batches = db.scalars(select(IngestionBatch).where(IngestionBatch.status.notin_(TERMINAL_STATES))).all()
        for batch in batches:
            stats = dict(batch.stats or {})
            if batch_is_worker_owned(stats):
                stats["api_restart_observed_at"] = datetime.utcnow().isoformat()
                batch.stats = stats
                continue
            batch.status = "failed"
            batch.completed_at = datetime.utcnow()
            batch.last_error = "API restarted while batch was in progress"
            batch.stats = {**stats, "manual_review_required": True, "phase": "interrupted"}
        if batches:
            db.commit()


def batch_is_worker_owned(stats: dict) -> bool:
    return stats.get("ingestion_execution_mode") == "celery" or bool(stats.get("celery_task_id"))


async def run_batch_ingestion(batch_id: str) -> dict:
    from app.db import SessionLocal

    with SessionLocal() as db:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        file_paths = [Path(path) for path in (batch.stats or {}).get("file_paths", [])]
    return await run_uploaded_files_ingestion(batch_id, [str(path) for path in file_paths], execution_mode="celery")


async def run_context_graph_rebuild_batch(batch_id: str, *, execution_mode: str = "inline") -> dict:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        knowledge_base = resolve_knowledge_base(db, batch.knowledge_base_id)
        batch.status = "extracting_graph"
        batch.started_at = datetime.utcnow()
        batch.completed_at = None
        batch.total_files = 1
        batch.processed_files = 0
        batch.success_count = 0
        batch.failure_count = 0
        batch.skipped_count = 0
        batch.worker_id = current_worker_id()
        batch.heartbeat_at = datetime.utcnow()
        batch.stats = {
            **(batch.stats or {}),
            "phase": "context_graph",
            "ingestion_execution_mode": execution_mode,
            "maintenance_task": "context_graph_rebuild",
            "errors": [],
        }
        db.commit()
        emit_ingestion_log(batch.id, "batch_started", "Context graph rebuild started", maintenance_task="context_graph_rebuild")
        emit_ingestion_log(batch.id, "context_graph_started", "Building four-layer context graph")
        context_state = await rebuild_context_graph(db, knowledge_base.id, batch_id=batch.id)
        graph_stats = dict(context_state.stats_json or {})
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} disappeared")
        batch.status = "completed"
        batch.processed_files = 1
        batch.success_count = 1
        batch.failure_count = 0
        batch.completed_at = datetime.utcnow()
        batch.worker_id = None
        batch.heartbeat_at = None
        batch.stats = {
            **(batch.stats or {}),
            "phase": "completed",
            "graph_stats": graph_stats,
            "parse_committed": False,
            "graph_rebuild_committed": True,
            "context_graph_state_id": context_state.id,
        }
        db.commit()
        emit_ingestion_log(batch.id, "context_graph_completed", "Four-layer context graph is active", **graph_stats)
        emit_ingestion_log(batch.id, "batch_completed", "Context graph rebuild completed", **graph_stats)
        return summarize_batch(batch)
    except Exception as exc:
        db.rollback()
        message = exception_message(exc)
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            stats = dict(batch.stats or {})
            errors = list(stats.get("errors") or [])
            errors.append({"phase": stats.get("phase") or batch.status, "message": message})
            batch.status = "failed"
            batch.last_error = message
            batch.completed_at = datetime.utcnow()
            batch.worker_id = None
            batch.heartbeat_at = None
            batch.stats = {
                **stats,
                "phase": "failed",
                "errors": errors,
                "manual_review_required": True,
                "failed_at": datetime.utcnow().isoformat(),
            }
            db.commit()
            emit_ingestion_log(batch.id, "batch_failed", f"Context graph rebuild failed: {message}", error=message)
        raise
    finally:
        db.close()


async def run_uploaded_files_ingestion(
    batch_id: str,
    file_paths: list[str],
    *,
    force: bool = False,
    full_reparse: bool = False,
    execution_mode: str = "inline",
) -> dict:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        knowledge_base = resolve_knowledge_base(db, batch.knowledge_base_id)
        paths = [Path(path).resolve() for path in file_paths]
        current_version = knowledge_base.current_chunk_version or 0
        active_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
        active_max_version = db.scalar(select(func.max(Chunk.chunk_version)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
        if full_reparse and active_chunks <= 0:
            raise RuntimeError("Full reparse is unavailable before the first successful parse")
        target_version = target_chunk_version(current_version=current_version, active_max_version=active_max_version, full_reparse=full_reparse)
        batch.status = "parsing"
        batch.started_at = datetime.utcnow()
        batch.completed_at = None
        batch.total_files = len(paths)
        batch.processed_files = 0
        batch.success_count = 0
        batch.failure_count = 0
        batch.skipped_count = 0
        batch.worker_id = current_worker_id()
        batch.heartbeat_at = datetime.utcnow()
        batch.stats = {
            **(batch.stats or {}),
            "phase": "parsing",
            "target_version": target_version,
            "current_chunk_version_before": current_version,
            "active_max_chunk_version_before": active_max_version,
            "full_reparse": full_reparse,
            "force": force,
            "ingestion_execution_mode": execution_mode,
            "errors": [],
        }
        db.commit()
        emit_ingestion_log(batch.id, "batch_started", f"Parsing {len(paths)} files into context graph", total_files=len(paths), full_reparse=full_reparse)
        coverage: Counter[str] = Counter()
        errors: list[dict[str, str]] = []
        for index, path in enumerate(paths, start=1):
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                break
            mark_batch_worker_heartbeat(db, batch, phase="parsing")
            db.commit()
            emit_ingestion_log(batch.id, "file_started", f"[{index}/{len(paths)}] Parsing {path.name}", source_path=str(path))
            try:
                result = await ingest_file(
                    db,
                    path,
                    trigger_source=batch.trigger_source,
                    batch_id=batch.id,
                    knowledge_base_id=knowledge_base.id,
                    rebuild_graph=False,
                    force=force,
                    target_version=target_version,
                )
                coverage[result["source_type"]] += 1
                batch.success_count += 1
                emit_ingestion_log(batch.id, "file_completed", f"{path.name} parsed successfully", source_path=str(path), stats=result.get("stats", {}))
            except Exception as exc:
                db.rollback()
                message = exception_message(exc)
                errors.append({"source_path": str(path), "message": message})
                batch = db.get(IngestionBatch, batch_id)
                if batch is not None:
                    batch.failure_count += 1
                    batch.last_error = message
                    batch.stats = {**(batch.stats or {}), "errors": errors}
                emit_ingestion_log(batch_id, "file_failed", f"{path.name} parse failed: {message}", source_path=str(path), error=message)
                db.commit()
            finally:
                batch = db.get(IngestionBatch, batch_id)
                if batch is not None:
                    batch.processed_files += 1
                    batch.stats = {**(batch.stats or {}), "coverage_by_source_type": dict(coverage), "errors": errors}
                    db.commit()
                    emit_ingestion_log(
                        batch.id,
                        "batch_progress",
                        f"Progress {batch.processed_files}/{batch.total_files}",
                        processed_files=batch.processed_files,
                        total_files=batch.total_files,
                        success_count=batch.success_count,
                        failure_count=batch.failure_count,
                    )
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} disappeared")
        if batch.success_count > 0:
            batch.status = "extracting_graph"
            batch.stats = {**(batch.stats or {}), "phase": "context_graph"}
            mark_batch_worker_heartbeat(db, batch, phase="context_graph")
            db.commit()
            emit_ingestion_log(batch.id, "context_graph_started", "Building four-layer context graph")
            context_state = await rebuild_context_graph(db, knowledge_base.id, batch_id=batch.id)
            knowledge_base.current_chunk_version = max(knowledge_base.current_chunk_version or 0, target_version)
            graph_stats = dict(context_state.stats_json or {})
            batch.stats = {**(batch.stats or {}), "graph_stats": graph_stats, "phase": "completed", "parse_committed": True}
            emit_ingestion_log(batch.id, "context_graph_completed", "Four-layer context graph is active", **graph_stats)
        else:
            graph_stats = {}
            batch.stats = {**(batch.stats or {}), "graph_stats": graph_stats, "phase": "completed", "parse_committed": False}
        if batch.success_count == 0 and batch.total_files > 0:
            batch.status = "failed"
            terminal_event = "batch_failed"
        elif batch.failure_count > 0:
            batch.status = "partial_failed"
            terminal_event = "batch_partial_failed"
        else:
            batch.status = "completed"
            terminal_event = "batch_completed"
        batch.completed_at = datetime.utcnow()
        batch.worker_id = None
        batch.heartbeat_at = None
        db.commit()
        emit_ingestion_log(batch.id, terminal_event, f"Batch {batch.status}: success={batch.success_count}, failed={batch.failure_count}, skipped={batch.skipped_count}")
        return summarize_batch(batch)
    except Exception as exc:
        db.rollback()
        message = exception_message(exc)
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            stats = dict(batch.stats or {})
            errors = list(stats.get("errors") or [])
            errors.append({"phase": stats.get("phase") or batch.status, "message": message})
            batch.status = "failed"
            batch.last_error = message
            batch.completed_at = datetime.utcnow()
            batch.worker_id = None
            batch.heartbeat_at = None
            batch.stats = {
                **stats,
                "phase": "failed",
                "errors": errors,
                "manual_review_required": True,
                "failed_at": datetime.utcnow().isoformat(),
            }
            db.commit()
            emit_ingestion_log(batch.id, "batch_failed", f"Batch failed during {(stats.get('phase') or batch.status)}: {message}", error=message)
        raise
    finally:
        db.close()


def target_chunk_version(*, current_version: int, active_max_version: int, full_reparse: bool) -> int:
    effective_current_version = max(current_version or 0, active_max_version or 0)
    return effective_current_version + 1 if full_reparse else (effective_current_version or 1)


async def run_ingestion_job(job_id: str, source_path: Path, trigger_source: str = "upload") -> dict:
    from app.db import SessionLocal

    with SessionLocal() as db:
        job = db.get(IngestionJob, job_id)
        knowledge_base_id = job.knowledge_base_id if job else None
        return await ingest_file(db, source_path, trigger_source=trigger_source, existing_job_id=job_id, knowledge_base_id=knowledge_base_id)


def list_knowledge_base_files(db: Session, knowledge_base_id: str) -> list[dict]:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        return []
    storage_root = get_settings().knowledge_base_paths_for_name(knowledge_base.name)["storage_root"]
    storage_paths = {str(path) for path in collect_source_documents(storage_root)}
    documents = list(db.scalars(select(Document).where(Document.knowledge_base_id == knowledge_base_id, Document.is_active.is_(True))).all())
    documents_by_path = {document.source_path: document for document in documents}
    chunk_counts = Counter(
        {
            document_id: count
            for document_id, count in db.execute(
                select(Chunk.document_id, func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active").group_by(Chunk.document_id)
            ).all()
        }
    )
    items: list[dict] = []
    for path_string in sorted(storage_paths | set(documents_by_path), key=str.lower):
        path = Path(path_string)
        document = documents_by_path.get(path_string)
        latest_job = db.scalar(select(IngestionJob).where(IngestionJob.knowledge_base_id == knowledge_base_id, IngestionJob.source_path == path_string).order_by(IngestionJob.updated_at.desc()))
        chunk_count = chunk_counts.get(document.id, 0) if document else 0
        status = "parsed" if chunk_count else "pending"
        if latest_job and latest_job.status in {"queued", "parsing", "chunking", "embedding", "extracting_graph", "processing"}:
            status = "parsing"
        elif latest_job and latest_job.status == "failed":
            status = "failed"
        items.append(
            {
                "id": document.id if document else path_string,
                "document_id": document.id if document else None,
                "title": document.title if document else path.stem,
                "source_path": path_string,
                "source_type": document.source_type if document else source_type_from_path(path),
                "partition": (document.tags or [None])[0] if document and document.tags else derive_partition(path, knowledge_base_name=knowledge_base.name),
                "status": status,
                "job_state": latest_job.status if latest_job else None,
                "batch_id": latest_job.batch_id if latest_job else None,
                "error": latest_job.error_message if latest_job and latest_job.status == "failed" else None,
                "chunk_count": chunk_count,
                "chunk_version": knowledge_base.current_chunk_version or None,
                "updated_at": document.updated_at if document else latest_job.updated_at if latest_job else None,
            }
        )
    return items


def remove_knowledge_base_file(db: Session, knowledge_base: KnowledgeBase, source_path: str) -> bool:
    document = db.scalar(select(Document).where(Document.knowledge_base_id == knowledge_base.id, Document.source_path == source_path))
    path = Path(source_path)
    removed = False
    if document is not None:
        document.is_active = False
        for chunk in db.scalars(select(Chunk).where(Chunk.document_id == document.id, Chunk.state == "active")).all():
            chunk.state = "deleted"
        removed = True
    try:
        storage_root = get_settings().knowledge_base_paths_for_name(knowledge_base.name)["storage_root"].resolve()
        resolved = path.resolve()
        if resolved == storage_root or storage_root in resolved.parents:
            resolved.unlink(missing_ok=True)
            removed = True
    except OSError:
        pass
    db.commit()
    return removed


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
        "stats": job.stats or {},
    }
