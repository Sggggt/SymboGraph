from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

from _context_graph_maintenance import active_chunk_count, active_chunk_max_version, knowledge_base_storage_root, resolve_knowledge_base, session_scope, storage_files, write_report


SOURCE_MANIFEST_PROTOCOL_VERSION = "rebuild_chunks_source_manifest_v1"
VERSION_IMPACT_PROTOCOL_VERSION = "rebuild_chunks_version_impact_v1"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_manifest(storage_root: Path, files: list[Path]) -> dict[str, Any]:
    """Build a relocation-stable, content-bound inventory without writes."""

    from app.services.storage import compute_checksum

    resolved_root = storage_root.resolve()
    cards: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for source in files:
        if source.is_symlink():
            raise RuntimeError(f"Source manifest rejects symbolic links: {source}")
        resolved = source.resolve(strict=True)
        try:
            relative_path = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"Source manifest path is outside the knowledge-base storage root: {source}"
            ) from exc
        normalized_path = unicodedata.normalize("NFKC", relative_path).casefold()
        if normalized_path in seen_paths:
            raise RuntimeError(
                f"Source manifest contains a casefold/NFKC path collision: {relative_path}"
            )
        seen_paths.add(normalized_path)
        before = resolved.stat()
        checksum = compute_checksum(resolved)
        after = resolved.stat()
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if before_identity != after_identity:
            raise RuntimeError(
                f"Source manifest file identity changed during checksum: {relative_path}"
            )
        cards.append(
            {
                "relative_path": relative_path,
                "size_bytes": int(after.st_size),
                "sha256": checksum,
            }
        )
    cards.sort(
        key=lambda card: (
            unicodedata.normalize("NFKC", str(card["relative_path"])).casefold(),
            str(card["relative_path"]),
        )
    )
    hash_payload = {
        "protocol_version": SOURCE_MANIFEST_PROTOCOL_VERSION,
        "files": cards,
    }
    return {
        **hash_payload,
        "file_count": len(cards),
        "total_bytes": sum(int(card["size_bytes"]) for card in cards),
        "manifest_hash": _canonical_hash(hash_payload),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full source-file reparse into fixed token chunks and all four graph layers.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true", help="Start the reparse. Omit for dry-run.")
    parser.add_argument("--full-reparse", action="store_true", help="Create current_chunk_version + 1 when parsed chunks already exist.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    from app.services.ingestion import create_uploaded_files_batch, run_uploaded_files_ingestion, target_chunk_version

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        files = storage_files(knowledge_base.source_root)
        storage_root = knowledge_base_storage_root(knowledge_base.source_root)
        source_manifest = build_source_manifest(storage_root, files)
        chunks = active_chunk_count(db, knowledge_base.id)
        current_version = int(knowledge_base.current_chunk_version or 0)
        active_max_version = active_chunk_max_version(db, knowledge_base.id)
        effective_current_version = max(current_version, active_max_version)
        target_version = target_chunk_version(
            current_version=current_version,
            active_max_version=active_max_version,
            full_reparse=args.full_reparse,
        )
        if chunks > 0 and not args.full_reparse:
            raise SystemExit("Existing parsed chunks require --full-reparse so the destructive version bump is explicit.")
        version_impact = {
            "protocol_version": VERSION_IMPACT_PROTOCOL_VERSION,
            "knowledge_base_current_chunk_version_before": current_version,
            "active_max_chunk_version_before": active_max_version,
            "effective_current_chunk_version_before": effective_current_version,
            "target_chunk_version": target_version,
            "chunk_version_incremented": target_version
            > effective_current_version,
            "durable_before_image_frozen": False,
            "durable_before_image_boundary": (
                "execution_only_after_knowledge_base_resource_lock"
            ),
        }
        payload = {
            "script": "rebuild_chunks",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "file_count": len(files),
            "active_chunks_before": chunks,
            "full_reparse": args.full_reparse,
            "execute": args.execute,
            "files": [str(path) for path in files],
            "source_manifest": source_manifest,
            "version_impact": version_impact,
            "impact": "parse files, write chunks/structure/contextual indexes, rebuild all graph layers" if args.execute else "no writes",
        }
        if args.execute:
            if not files:
                raise SystemExit("No source files found in knowledge base storage.")
            batch = create_uploaded_files_batch(db, knowledge_base.id, files, force=args.force, full_reparse=args.full_reparse)
            result = await run_uploaded_files_ingestion(batch.id, [str(path) for path in files], force=args.force, full_reparse=args.full_reparse, execution_mode="script")
            payload["batch"] = result
        report = write_report("rebuild_chunks", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
