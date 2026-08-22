from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest


class _OwnedScopeVectorStore:
    collections: dict[str, dict[str, dict]] = {}

    def __init__(
        self,
        _knowledge_base_name: str | None = None,
        collection_name: str | None = None,
        **_kwargs,
    ) -> None:
        self.collection = collection_name or "default"

    def list_collection_names_bounded(self, **_kwargs) -> dict:
        names = sorted(type(self).collections)
        return {
            "collection_names": names,
            "collection_count": len(names),
            "truncated": False,
            "max_collections": 512,
            "complete_backend_inventory": True,
        }

    def list_owned_ids_complete(self, knowledge_base_id: str, **_kwargs) -> dict:
        points = type(self).collections.get(self.collection, {})
        ids = sorted(
            point_id
            for point_id, point in points.items()
            if (point.get("payload") or {}).get("knowledge_base_id")
            == knowledge_base_id
        )
        return {
            "ids": ids,
            "point_count": len(ids),
            "page_count": 1 if ids else 0,
            "truncated": False,
            "max_points": 1_000_000,
        }

    def get_points_batched(self, ids: list[str], **_kwargs) -> list[dict]:
        points = type(self).collections.get(self.collection, {})
        return [dict(points[point_id]) for point_id in ids if point_id in points]

    def delete_if_payload_matches(self, expected_points: list[dict]) -> None:
        points = type(self).collections.setdefault(self.collection, {})
        for expected in expected_points:
            point_id = str(expected["id"])
            current = points.get(point_id)
            if current is not None and current.get("payload") == expected.get("payload"):
                points.pop(point_id)


def test_verified_storage_tree_delete_rejects_drift_and_is_durable(
    tmp_path: Path,
) -> None:
    from app.services.storage import (
        SourceSnapshotError,
        durable_delete_storage_tree,
        inventory_storage_tree,
    )

    root = tmp_path / "owned"
    snapshot = root / "ingestion" / "source_snapshots" / "aa" / "snapshot.md"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"immutable")
    os.chmod(snapshot, stat.S_IMODE(snapshot.stat().st_mode) & ~stat.S_IWUSR)
    inventory = inventory_storage_tree(root, authorized_parent=tmp_path)

    replacement = snapshot.with_suffix(".replacement")
    replacement.write_bytes(b"immutable")
    os.chmod(snapshot, stat.S_IMODE(snapshot.stat().st_mode) | stat.S_IWUSR)
    os.replace(replacement, snapshot)
    try:
        try:
            durable_delete_storage_tree(inventory, authorized_parent=tmp_path)
        except SourceSnapshotError:
            pass
        else:
            raise AssertionError("same-content identity drift must fail closed")
        assert snapshot.exists()
    finally:
        os.chmod(snapshot, stat.S_IMODE(snapshot.stat().st_mode) | stat.S_IWUSR)

    replay = inventory_storage_tree(root, authorized_parent=tmp_path)
    result = durable_delete_storage_tree(replay, authorized_parent=tmp_path)
    assert result["deleted_files"] == 1
    assert root.exists() is False


def test_full_kb_delete_tombstone_survives_facts_and_catches_qdrant_orphan(
    monkeypatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.models import KnowledgeBase, StorageMaintenanceIntent
    from app.services import cache_manager, maintenance, storage_maintenance

    class _Cache:
        def invalidate_knowledge_base(self, knowledge_base_id: str, *, strict: bool):
            assert knowledge_base_id == sample_knowledge_base.id
            assert strict is True
            return True

    monkeypatch.setattr(
        cache_manager,
        "get_cache_manager",
        lambda: _Cache(),
    )
    monkeypatch.setattr(
        storage_maintenance,
        "VectorStore",
        _OwnedScopeVectorStore,
    )
    _OwnedScopeVectorStore.collections = {
        "orphan-only-collection": {
            "orphan-point": {
                "id": "orphan-point",
                "vector": [1.0, 0.0],
                "payload": {
                    "knowledge_base_id": sample_knowledge_base.id,
                    "chunk_id": "orphan-point",
                },
            }
        }
    }
    paths = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )
    source = paths["storage_root"] / "source_slots" / "orphan.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"owned bytes")

    stats = maintenance.delete_knowledge_base_data(
        db_session,
        sample_knowledge_base,
    )

    assert stats["qdrant_points"] == 1
    assert stats["qdrant_orphan_points"] == 1
    assert paths["knowledge_base_root"].exists() is False
    assert (
        db_session.get(KnowledgeBase, sample_knowledge_base.id) is None
    )
    tombstone = db_session.query(StorageMaintenanceIntent).one()
    assert tombstone.status == "completed"
    assert tombstone.knowledge_base_id == sample_knowledge_base.id
    assert tombstone.payload_json["phase"] == "completed"
    assert _OwnedScopeVectorStore.collections["orphan-only-collection"] == {}


@pytest.mark.asyncio
async def test_full_kb_delete_cache_failure_is_recovered_from_surviving_tombstone(
    monkeypatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.models import KnowledgeBase, StorageMaintenanceIntent
    from app.services import cache_manager, maintenance, storage_maintenance

    class _Cache:
        fail = True

        def invalidate_knowledge_base(self, knowledge_base_id: str, *, strict: bool):
            assert knowledge_base_id == sample_knowledge_base.id
            assert strict is True
            if type(self).fail:
                raise ConnectionError("test Redis outage")
            return True

    monkeypatch.setattr(cache_manager, "get_cache_manager", lambda: _Cache())
    monkeypatch.setattr(
        storage_maintenance,
        "VectorStore",
        _OwnedScopeVectorStore,
    )
    _OwnedScopeVectorStore.collections = {}
    paths = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )
    source = paths["storage_root"] / "source_slots" / "cache-recovery.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"cache recovery")

    with pytest.raises(
        maintenance.MaintenanceConflict,
        match="cache invalidation remains pending",
    ):
        maintenance.delete_knowledge_base_data(
            db_session,
            sample_knowledge_base,
        )

    assert db_session.get(KnowledgeBase, sample_knowledge_base.id) is None
    tombstone = db_session.query(StorageMaintenanceIntent).one()
    assert tombstone.status == "cache_invalidation_pending"
    assert tombstone.payload_json["phase"] == "facts_deleted"

    _Cache.fail = False
    summary = (
        await storage_maintenance.reconcile_pending_storage_maintenance_startup()
    )
    db_session.expire_all()
    tombstone = db_session.get(StorageMaintenanceIntent, tombstone.id)
    assert summary["selected"] == 1
    assert summary["completed"] == 1
    assert summary["pending"] == 0
    assert summary["cache_pending"] == 0
    assert summary["external_pending"] == 0
    assert summary["status"] == "healthy"
    assert tombstone is not None
    assert tombstone.status == "completed"
    assert tombstone.payload_json["phase"] == "completed"


@pytest.mark.asyncio
async def test_startup_health_exposes_external_delete_phase_as_pending(
    monkeypatch,
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import KnowledgeBase, StorageMaintenanceIntent
    from app.services import maintenance, storage_maintenance

    class _FailingDeleteStore(_OwnedScopeVectorStore):
        def delete_if_payload_matches(self, expected_points: list[dict]) -> None:
            del expected_points
            raise OSError("test Qdrant outage")

    monkeypatch.setattr(
        storage_maintenance,
        "VectorStore",
        _FailingDeleteStore,
    )
    _FailingDeleteStore.collections = {
        "pending-delete": {
            "point": {
                "id": "point",
                "vector": [1.0, 0.0],
                "payload": {
                    "knowledge_base_id": sample_knowledge_base.id,
                    "chunk_id": "point",
                },
            }
        }
    }

    with pytest.raises(OSError, match="Qdrant outage"):
        maintenance.delete_knowledge_base_data(
            db_session,
            sample_knowledge_base,
        )

    db_session.expire_all()
    knowledge_base = db_session.get(KnowledgeBase, sample_knowledge_base.id)
    tombstone = db_session.query(StorageMaintenanceIntent).one()
    assert knowledge_base is not None
    assert knowledge_base.lifecycle_status == "deleting"
    assert tombstone.status == "external_deleting"

    summary = (
        await storage_maintenance.reconcile_pending_storage_maintenance_startup()
    )
    assert summary["selected"] == 0
    assert summary["cache_pending"] == 0
    assert summary["external_pending"] == 1
    assert summary["pending"] == 1
    assert summary["status"] == "degraded"


@pytest.mark.asyncio
async def test_snapshot_gc_dry_run_exact_confirmation_and_durable_intent(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.models import Document, DocumentVersion, StorageMaintenanceIntent
    from app.services.ingestion_resource_lock import (
        knowledge_base_ingestion_resource_lock,
    )
    from app.services.storage import (
        compute_checksum,
        protect_immutable_file,
        snapshot_source_file,
    )
    from app.services.storage_maintenance import (
        SOURCE_SNAPSHOT_GC_OPERATION,
        StorageMaintenanceIntegrityError,
        run_source_snapshot_gc,
    )

    paths = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )
    logical = paths["storage_root"] / "source_slots" / "retained.md"
    logical.parent.mkdir(parents=True, exist_ok=True)
    logical.write_bytes(b"retained")
    frozen_snapshot = snapshot_source_file(
        logical,
        knowledge_base_source_root=sample_knowledge_base.source_root,
    )
    retained = frozen_snapshot.canonical_path
    retained_checksum = frozen_snapshot.checksum
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="retained",
        source_path=str(logical),
        source_type="markdown",
        checksum=retained_checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    db_session.add(
        DocumentVersion(
            document_id=document.id,
            version=1,
            checksum=retained_checksum,
            storage_path=str(retained),
            is_active=True,
        )
    )
    db_session.commit()

    snapshot_root = paths["ingestion_root"] / "source_snapshots"
    orphan = snapshot_root / "ff" / ("f" * 64) / "orphan.md"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    old = time.time() - 120
    os.utime(orphan, (old, old))
    protect_immutable_file(orphan)
    assert len(compute_checksum(orphan)) == 64

    preview = run_source_snapshot_gc(
        db_session,
        sample_knowledge_base,
        retention_seconds=0,
    )
    inventory = preview["inventory"]
    assert inventory["counts"]["active_reference"] == 1
    assert inventory["counts"]["gc_candidate"] == 1, inventory["files"]
    assert orphan.exists()

    with pytest.raises(
        StorageMaintenanceIntegrityError,
        match="resource lock",
    ):
        run_source_snapshot_gc(
            db_session,
            sample_knowledge_base,
            execute=True,
            confirm_knowledge_base_id=sample_knowledge_base.id,
            confirm_inventory_hash=inventory["inventory_hash"],
            retention_seconds=0,
        )

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=SOURCE_SNAPSHOT_GC_OPERATION,
    ):
        executed = run_source_snapshot_gc(
            db_session,
            sample_knowledge_base,
            execute=True,
            confirm_knowledge_base_id=sample_knowledge_base.id,
            confirm_inventory_hash=inventory["inventory_hash"],
            retention_seconds=0,
        )

    assert executed["deleted_files"] == 1
    assert orphan.exists() is False
    assert retained.exists() is True
    intent = db_session.get(StorageMaintenanceIntent, executed["intent_id"])
    assert intent is not None
    assert intent.status == "completed"
