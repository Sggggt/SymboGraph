from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

OUTPUT_ROOT = REPO_ROOT / "output"
KNOWLEDGE_BASE_RESOLUTION_SAMPLE_LIMIT = 8


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{name}_{utc_stamp()}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def prepare_runtime_for_model_io() -> dict:
    from app.services.runtime_settings import refresh_runtime_settings_if_needed, sync_model_bridge_runtime_config

    refresh_runtime_settings_if_needed(force=True)
    return sync_model_bridge_runtime_config()


def session_scope():
    from app.db import SessionLocal

    return SessionLocal()


def resolve_knowledge_base(db, *, knowledge_base_id: str | None = None, knowledge_base_name: str | None = None):
    from sqlalchemy import func, select

    from app.models import Chunk, KnowledgeBase

    if knowledge_base_id:
        knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise SystemExit(f"Knowledge base not found: {knowledge_base_id}")
        return knowledge_base
    if knowledge_base_name:
        knowledge_base = db.scalar(select(KnowledgeBase).where(KnowledgeBase.name == knowledge_base_name))
        if knowledge_base is None:
            raise SystemExit(f"Knowledge base not found: {knowledge_base_name}")
        return knowledge_base
    active_counts = (
        select(
            Chunk.knowledge_base_id.label("knowledge_base_id"),
            func.count(Chunk.id).label("active_chunk_count"),
        )
        .where(Chunk.state == "active")
        .group_by(Chunk.knowledge_base_id)
        .subquery()
    )
    active_chunk_count_expr = func.coalesce(
        active_counts.c.active_chunk_count,
        0,
    )
    changed_at_expr = func.coalesce(
        KnowledgeBase.updated_at,
        KnowledgeBase.created_at,
    )
    bounded_rows = list(
        db.execute(
            select(
                KnowledgeBase,
                active_chunk_count_expr.label("active_chunk_count"),
                changed_at_expr.label("changed_at"),
            )
            .outerjoin(
                active_counts,
                active_counts.c.knowledge_base_id == KnowledgeBase.id,
            )
            .where(active_chunk_count_expr > 0)
            .order_by(
                active_chunk_count_expr.desc(),
                changed_at_expr.desc(),
                KnowledgeBase.id.asc(),
            )
            .limit(KNOWLEDGE_BASE_RESOLUTION_SAMPLE_LIMIT + 1)
        )
    )
    if bounded_rows:
        best_knowledge_base, best_count, best_changed_at = bounded_rows[0]
        tied_rows = [
            row
            for row in bounded_rows
            if int(row[1]) == int(best_count) and row[2] == best_changed_at
        ]
        if len(tied_rows) > 1:
            sampled_rows = tied_rows[:KNOWLEDGE_BASE_RESOLUTION_SAMPLE_LIMIT]
            sample = [
                {
                    "id": str(row[0].id),
                    "name": str(row[0].name),
                    "active_chunk_count": int(row[1]),
                    "changed_at": row[2].isoformat() if row[2] else None,
                }
                for row in sampled_rows
            ]
            sample_truncated = (
                len(tied_rows) > KNOWLEDGE_BASE_RESOLUTION_SAMPLE_LIMIT
            )
            raise SystemExit(
                "Knowledge base selection is ambiguous; specify --knowledge-base-id or "
                f"--knowledge-base-name. candidate_sample={json.dumps(sample, ensure_ascii=False)}; "
                f"sample_truncated={str(sample_truncated).lower()}"
            )
        return best_knowledge_base
    from app.services.ingestion import resolve_knowledge_base as resolve_default

    try:
        return resolve_default(db, None)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc


def active_chunk_count(db, knowledge_base_id: str) -> int:
    from sqlalchemy import func, select

    from app.models import Chunk

    return db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active")) or 0


def active_chunk_max_version(db, knowledge_base_id: str) -> int:
    from sqlalchemy import func, select

    from app.models import Chunk

    return int(
        db.scalar(
            select(func.max(Chunk.chunk_version)).where(
                Chunk.knowledge_base_id == knowledge_base_id,
                Chunk.state == "active",
            )
        )
        or 0
    )


def knowledge_base_storage_root(knowledge_base_source_root: str | Path) -> Path:
    from app.core.config import get_settings

    return Path(
        get_settings().knowledge_base_paths_for_source_root(
            knowledge_base_source_root
        )["storage_root"]
    )


def storage_files(knowledge_base_source_root: str | Path) -> list[Path]:
    from app.services.ingestion import collect_source_documents

    root = knowledge_base_storage_root(knowledge_base_source_root)
    return collect_source_documents(root)
