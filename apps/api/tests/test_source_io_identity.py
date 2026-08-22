from __future__ import annotations

import asyncio
from io import BytesIO
import os
from pathlib import Path
from threading import Lock, get_ident
import time

import pytest
from fastapi import UploadFile


@pytest.fixture
def isolated_source_io_concurrency(
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep this concurrency probe independent from the shared workspace .env."""

    from app.core import config as config_module
    from app.core.config import get_settings

    monkeypatch.setenv("SOURCE_IO_CONCURRENCY", "2")
    monkeypatch.setattr(
        config_module,
        "_read_workspace_env",
        lambda: {key.upper(): value for key, value in os.environ.items()},
    )
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def test_verified_source_open_binds_regular_leaf_identity(
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.services.storage import (
        open_verified_source_file,
        verified_source_checksum,
    )

    root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    root.mkdir(parents=True, exist_ok=True)
    source = root / "verified.md"
    source.write_bytes(b"verified bytes")

    with open_verified_source_file(source, root) as (handle, identity):
        assert handle.read() == b"verified bytes"
        assert identity.size_bytes == len(b"verified bytes")
        assert identity.inode == source.stat().st_ino

    checksum, replay_identity = verified_source_checksum(source, root)
    assert len(checksum) == 64
    assert replay_identity == identity


def test_verified_source_open_rejects_symlink_leaf(
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.services.storage import SourceSnapshotError, open_verified_source_file

    root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    root.mkdir(parents=True, exist_ok=True)
    victim = root / "victim.md"
    victim.write_bytes(b"victim")
    link = root / "link.md"
    try:
        link.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(SourceSnapshotError, match="symbolic link|reparse point"):
        with open_verified_source_file(link, root):
            raise AssertionError("symlink leaf must not be opened")


@pytest.mark.skipif(
    os.name != "posix",
    reason="readonly import contract requires POSIX dirfd/openat primitives",
)
def test_verified_readonly_import_replays_authorized_root_path_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import storage

    root = tmp_path / "root"
    displaced = tmp_path / "displaced"
    replacement = tmp_path / "replacement"
    root.mkdir()
    replacement.mkdir()
    source = root / "source.md"
    source.write_bytes(b"pinned original")
    (replacement / "source.md").write_bytes(b"replacement root")

    def mount_info(path: Path, root_stat: os.stat_result) -> dict[str, str]:
        return {
            "mount_id": "owner-regression",
            "parent_id": "owner-parent",
            "major_minor": (
                f"{os.major(root_stat.st_dev)}:{os.minor(root_stat.st_dev)}"
            ),
            "mount_root": "/",
            "mount_point": str(path),
            "mount_options": "ro,noatime",
            "filesystem_type": "ext4",
            "mount_source": "/dev/owner-regression",
            "super_options": "rw",
            "mount_signature": "owner-regression",
        }

    monkeypatch.setattr(storage, "_normalized_mount_info", mount_info)
    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(
            storage.SourceSnapshotError,
            match="root identity changed",
        ):
            with storage.open_verified_readonly_import_file(
                source,
                root,
            ) as (handle, _identity):
                assert handle.read() == b"pinned original"
                os.rename(root, displaced)
                os.rename(replacement, root)


@pytest.mark.skipif(
    os.name != "posix",
    reason="readonly import contract requires POSIX dirfd/openat primitives",
)
def test_verified_readonly_import_replays_full_root_stat_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import storage

    root = tmp_path / "root"
    root.mkdir()
    source = root / "source.md"
    source.write_bytes(b"pinned root stat")

    monkeypatch.setattr(
        storage,
        "_normalized_mount_info",
        lambda path, root_stat: {
            "mount_id": "owner-root-stat",
            "parent_id": "owner-parent",
            "major_minor": (
                f"{os.major(root_stat.st_dev)}:{os.minor(root_stat.st_dev)}"
            ),
            "mount_root": "/",
            "mount_point": str(path),
            "mount_options": "ro,noatime",
            "filesystem_type": "ext4",
            "mount_source": "/dev/owner-root-stat",
            "super_options": "rw",
            "mount_signature": "owner-root-stat",
        },
    )
    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(
            storage.SourceSnapshotError,
            match="root identity changed",
        ):
            with storage.open_verified_readonly_import_file(
                source,
                root,
            ) as (handle, _identity):
                assert handle.read() == b"pinned root stat"
                # A child directory changes the pinned root's link count,
                # size, mtime and ctime without changing its device/inode.
                (root / "identity-drift").mkdir()


def test_durable_unlink_rejects_same_content_leaf_identity_swap(
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.services.storage import (
        DirectoryDurabilityError,
        durable_unlink,
        verified_source_checksum,
    )

    root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    root.mkdir(parents=True, exist_ok=True)
    target = root / "identity-swap.md"
    replacement = root / "identity-swap.replacement"
    target.write_bytes(b"same content")
    _checksum, verified_identity = verified_source_checksum(target, root)
    replacement.write_bytes(b"same content")
    os.replace(replacement, target)

    with pytest.raises(DirectoryDurabilityError, match="identity changed"):
        durable_unlink(
            target,
            expected_identity=verified_identity,
        )

    assert target.read_bytes() == b"same content"


@pytest.mark.asyncio
async def test_bounded_source_io_enforces_configured_concurrency(
    isolated_source_io_concurrency,
) -> None:
    from app.services.storage import run_bounded_source_io

    guard = Lock()
    active = 0
    peak = 0

    def blocking_probe(value: int) -> int:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return value

    values = await asyncio.gather(
        *(run_bounded_source_io(blocking_probe, value) for value in range(8))
    )

    assert values == list(range(8))
    assert peak == 2


@pytest.mark.asyncio
async def test_upload_candidate_hash_and_fsync_run_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
    sample_knowledge_base,
) -> None:
    from app.core.config import get_settings
    from app.services import storage

    root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / ".bounded.candidate"
    event_loop_thread = get_ident()
    observed_thread: list[int] = []
    original = storage._write_upload_candidate_sync

    def observe(*args, **kwargs):
        observed_thread.append(get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "_write_upload_candidate_sync", observe)
    checksum, size_bytes = await storage.write_upload_candidate(
        UploadFile(filename="bounded.md", file=BytesIO(b"bounded")),
        candidate,
        max_bytes=1024,
    )

    assert size_bytes == 7
    assert len(checksum) == 64
    assert observed_thread and observed_thread[0] != event_loop_thread
