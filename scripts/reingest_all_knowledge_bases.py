#!/usr/bin/env python3
"""Reingest knowledge base storage files through the normal Docker API service code.

Run inside the API container:
    python /app/scripts/reingest_all_knowledge_bases.py --knowledge-base-name "Knowledge Base Name" --cleanup-stale

The script scans DATA_ROOT knowledge base directories, creates a sync ingestion batch,
and runs the full parse -> chunk -> embed -> graph pipeline. It refuses to run
when model or database fallback is enabled.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

# 将项目根加入路径
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "apps" / "api"))

from app.db import SessionLocal
from app.models import KnowledgeBase
from app.core.config import get_settings
from app.services.ingestion import active_chunk_count, create_sync_batch, run_batch_ingestion
from app.services.maintenance import cleanup_stale_data

EXCLUDE_DIRS = {"ingestion", "postgres", "qdrant", "redis", "storage", "models"}


def get_all_knowledge_bases(knowledge_base_name: str | None = None) -> list[Path]:
    data_root = get_settings().data_root
    if not data_root.exists():
        return []
    knowledge_bases = [
        d for d in data_root.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DIRS and (d / "storage").exists()
    ]
    if knowledge_base_name:
        knowledge_bases = [item for item in knowledge_bases if item.name == knowledge_base_name]
    return knowledge_bases


def ensure_knowledge_base_record(db, knowledge_base_name: str) -> KnowledgeBase:
    """Get or create a KnowledgeBase record for a DATA_ROOT knowledge base directory."""
    knowledge_base = db.query(KnowledgeBase).filter(KnowledgeBase.name == knowledge_base_name).first()
    if knowledge_base is None:
        from app.services.ingestion import create_knowledge_base_space
        knowledge_base = create_knowledge_base_space(db, knowledge_base_name)
    else:
        paths = get_settings().knowledge_base_paths_for_name(knowledge_base.name)
        paths["knowledge_base_root"].mkdir(parents=True, exist_ok=True)
        paths["storage_root"].mkdir(parents=True, exist_ok=True)
        paths["ingestion_root"].mkdir(parents=True, exist_ok=True)
    return knowledge_base


async def reingest_knowledge_base(knowledge_base_name: str, cleanup_stale: bool) -> dict:
    db = SessionLocal()
    try:
        knowledge_base = ensure_knowledge_base_record(db, knowledge_base_name)
        paths = get_settings().knowledge_base_paths_for_name(knowledge_base.name)
        storage_root = paths["storage_root"]

        if not storage_root.exists():
            return {"knowledge_base": knowledge_base_name, "status": "skipped", "reason": "no storage directory"}

        batch = create_sync_batch(db, knowledge_base.id, storage_root, trigger_source="reingest")
        print(f"  [batch {batch.id}] created for knowledge base '{knowledge_base_name}'")

        result = await run_batch_ingestion(batch.id, force=True, full_reparse=active_chunk_count(db, knowledge_base.id) > 0)
        cleanup_stats = None
        if cleanup_stale and result.get("state") in {"completed", "partial_failed"}:
            cleanup_stats = cleanup_stale_data(db, knowledge_base.id, knowledge_base.name)
        return {
            "knowledge_base": knowledge_base_name,
            "status": result.get("state", "unknown"),
            "batch_id": batch.id,
            "cleanup": cleanup_stats,
        }
    except Exception as exc:
        return {"knowledge_base": knowledge_base_name, "status": "error", "error": str(exc)}
    finally:
        db.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reingest knowledge base storage files through the Docker API code path.")
    parser.add_argument("--knowledge-base-name", dest="knowledge_base_name", default=None, help="Limit to one knowledge base directory name.")
    parser.add_argument("--cleanup-stale", action="store_true", help="Delete inactive DB rows and stale Qdrant vectors after each successful knowledge base run.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise SystemExit("Refusing to reingest while fallback is enabled.")
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for real no-fallback reingestion.")

    knowledge_bases = get_all_knowledge_bases(args.knowledge_base_name)
    if not knowledge_bases:
        print("No matching knowledge base directories found in DATA_ROOT")
        return

    print(f"Found {len(knowledge_bases)} knowledge base(s) to re-ingest:")
    for knowledge_base_dir in knowledge_bases:
        print(f"  - {knowledge_base_dir.name}")

    for knowledge_base_dir in knowledge_bases:
        print(f"\nRe-ingesting: {knowledge_base_dir.name}")
        result = await reingest_knowledge_base(knowledge_base_dir.name, args.cleanup_stale)
        print(f"  -> {result['status']}")
        if result.get("cleanup") is not None:
            print(f"     cleanup={result['cleanup']}")
        if "error" in result:
            print(f"     ERROR: {result['error']}")

    print("\nAll knowledge bases re-ingestion completed.")


if __name__ == "__main__":
    asyncio.run(main())
