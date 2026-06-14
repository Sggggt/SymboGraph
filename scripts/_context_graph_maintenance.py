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


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{name}_{utc_stamp()}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def session_scope():
    from app.db import SessionLocal

    return SessionLocal()


def resolve_knowledge_base(db, *, knowledge_base_id: str | None = None, knowledge_base_name: str | None = None):
    from sqlalchemy import select

    from app.models import KnowledgeBase
    from app.services.ingestion import resolve_knowledge_base as resolve_default

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
    try:
        return resolve_default(db, None)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc


def active_chunk_count(db, knowledge_base_id: str) -> int:
    from sqlalchemy import func, select

    from app.models import Chunk

    return db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active")) or 0


def storage_files(knowledge_base_name: str) -> list[Path]:
    from app.core.config import get_settings
    from app.services.ingestion import collect_source_documents

    root = get_settings().knowledge_base_paths_for_name(knowledge_base_name)["storage_root"]
    return collect_source_documents(root)
