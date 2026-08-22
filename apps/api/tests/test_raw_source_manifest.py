from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mount contract")
def test_raw_manifest_uses_readonly_identity_without_mutation_durability_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from app.services import storage
    from app.services.ingestion import load_raw_source_manifest
    from app.services.storage import SourceSnapshotError

    source = tmp_path / "source.md"
    source.write_text("checksum-bound operator source", encoding="utf-8")
    (tmp_path / "raw-manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "symbograph_raw_source_manifest_v1",
                "raw_root": ".",
                "files": [{"path": source.name, "sha256": _sha256(source)}],
            }
        ),
        encoding="utf-8",
    )

    mount = {
        "mount_options": "ro,noatime",
        "super_options": "rw",
    }
    monkeypatch.setattr(storage, "_normalized_mount_info", lambda *_args: mount)
    with storage._without_test_namespace_durability_adapter():
        loaded = load_raw_source_manifest(tmp_path)
    assert loaded["paths"] == [source]
    assert loaded["file_count"] == 1

    mount["mount_options"] = "rw,noatime"
    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(SourceSnapshotError, match="mounted read-only"):
            with storage.open_verified_readonly_import_file(source, tmp_path):
                pass


def test_raw_manifest_is_complete_allowlist_not_recursive_discovery(
    tmp_path: Path,
) -> None:
    from app.services.ingestion import (
        RAW_SOURCE_MANIFEST_FILENAME,
        RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
        collect_source_documents,
        load_raw_source_manifest,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    included = raw / "included.md"
    included.write_text("included", encoding="utf-8")
    (raw / "unlisted.md").write_text("must not import", encoding="utf-8")
    derived = raw / "ingestion" / "source_snapshots" / "aa"
    derived.mkdir(parents=True)
    (derived / "snapshot.md").write_text("derived", encoding="utf-8")
    manifest = {
        "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
        "raw_root": "raw",
        "files": [{"path": "included.md", "sha256": _sha256(included)}],
    }
    (tmp_path / RAW_SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    loaded = load_raw_source_manifest(tmp_path)
    assert loaded["paths"] == [included]
    assert collect_source_documents(tmp_path, raw_manifest=True) == [included]
    assert loaded["file_count"] == 1
    assert len(loaded["manifest_hash"]) == 64


def test_raw_manifest_rejects_checksum_drift_and_normalized_duplicates(
    tmp_path: Path,
) -> None:
    from app.services.ingestion import (
        RAW_SOURCE_MANIFEST_FILENAME,
        RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
        load_raw_source_manifest,
    )
    from app.services.storage import UploadValidationError

    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8")
    manifest_path = tmp_path / RAW_SOURCE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
                "raw_root": ".",
                "files": [{"path": "source.md", "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UploadValidationError, match="checksum mismatch"):
        load_raw_source_manifest(tmp_path)

    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
                "raw_root": ".",
                "files": [
                    {"path": "source.md", "sha256": _sha256(source)},
                    {"path": "source.md", "sha256": _sha256(source)},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UploadValidationError, match="casefold/NFKC"):
        load_raw_source_manifest(tmp_path)


def test_raw_manifest_rejects_derived_paths_and_upload_slot_collisions(
    tmp_path: Path,
) -> None:
    from app.services.ingestion import (
        RAW_SOURCE_MANIFEST_FILENAME,
        RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
        load_raw_source_manifest,
    )
    from app.services.storage import UploadValidationError

    first = tmp_path / "first" / "same.md"
    second = tmp_path / "second" / "same.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest_path = tmp_path / RAW_SOURCE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
                "raw_root": ".",
                "files": [
                    {"path": "first/same.md", "sha256": _sha256(first)},
                    {"path": "second/same.md", "sha256": _sha256(second)},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UploadValidationError, match="upload-slot"):
        load_raw_source_manifest(tmp_path)

    derived = tmp_path / "ingestion" / "source_snapshots" / "derived.md"
    derived.parent.mkdir(parents=True)
    derived.write_text("derived", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
                "raw_root": ".",
                "files": [
                    {
                        "path": "ingestion/source_snapshots/derived.md",
                        "sha256": _sha256(derived),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UploadValidationError, match="derived/reserved"):
        load_raw_source_manifest(tmp_path)


@pytest.mark.skipif(
    os.name != "posix",
    reason="readonly import contract requires POSIX dirfd/openat primitives",
)
def test_raw_manifest_freezes_one_import_root_across_all_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import ingestion, storage
    from app.services.storage import UploadValidationError

    root = tmp_path / "root"
    displaced = tmp_path / "displaced"
    replacement = tmp_path / "replacement"
    root.mkdir()
    replacement.mkdir()
    original = root / "source.md"
    substitute = replacement / "source.md"
    original.write_bytes(b"same checksum-bound bytes")
    substitute.write_bytes(b"same checksum-bound bytes")
    (root / "raw-manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "symbograph_raw_source_manifest_v1",
                "raw_root": ".",
                "files": [
                    {
                        "path": "source.md",
                        "sha256": _sha256(original),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def mount_info(path: Path, root_stat: os.stat_result) -> dict[str, str]:
        return {
            "mount_id": "owner-manifest-regression",
            "parent_id": "owner-parent",
            "major_minor": (
                f"{os.major(root_stat.st_dev)}:{os.minor(root_stat.st_dev)}"
            ),
            "mount_root": "/",
            "mount_point": str(path),
            "mount_options": "ro,noatime",
            "filesystem_type": "ext4",
            "mount_source": "/dev/owner-manifest-regression",
            "super_options": "rw",
            "mount_signature": "owner-manifest-regression",
        }

    monkeypatch.setattr(storage, "_normalized_mount_info", mount_info)
    original_checksum = ingestion.verified_readonly_import_checksum
    swapped = False

    def swap_root_then_checksum(source_path: Path, authorized_root: Path):
        nonlocal swapped
        if not swapped:
            os.rename(root, displaced)
            os.rename(replacement, root)
            swapped = True
        return original_checksum(source_path, authorized_root)

    monkeypatch.setattr(
        ingestion,
        "verified_readonly_import_checksum",
        swap_root_then_checksum,
    )
    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(
            UploadValidationError,
            match="root identity changed",
        ):
            ingestion.load_raw_source_manifest(root)
