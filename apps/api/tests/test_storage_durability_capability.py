from __future__ import annotations

import os
import select
import signal
from pathlib import Path
from threading import Event, Thread

import httpx
import pytest
from fastapi import FastAPI


def test_settings_loading_has_no_directory_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.core import config

    absent_data_root = tmp_path / "settings-must-not-create-data"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATA_ROOT", str(absent_data_root))
    monkeypatch.setattr(
        config,
        "_read_workspace_env",
        lambda: {key.upper(): value for key, value in os.environ.items()},
    )
    config.get_settings.cache_clear()
    try:
        settings = config.get_settings()
        assert settings.data_root == absent_data_root
        assert not absent_data_root.exists()
        assert not settings.knowledge_base_data_root_path.exists()
        assert not settings.storage_root_path.exists()
        assert not settings.ingestion_root_path.exists()
    finally:
        config.get_settings.cache_clear()


def test_unsupported_data_mount_fails_before_child_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.core.config import get_settings
    from app.services import storage

    settings = get_settings()
    assert settings.data_root == no_fallback_env
    assert not settings.knowledge_base_data_root_path.exists()
    monkeypatch.setattr(storage, "_is_native_windows", lambda: False)
    monkeypatch.setattr(
        storage,
        "_mount_info_for_path",
        lambda root: {
            "filesystem_type": "virtiofs",
            "mount_source": "docker-desktop-bind",
            "mount_point": str(root),
        },
    )
    storage._clear_storage_durability_capability_cache_for_test()

    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(storage.StorageDurabilityCapabilityError):
            storage.ensure_storage_durability_ready(settings=settings)

    assert no_fallback_env.is_dir()
    assert not settings.knowledge_base_data_root_path.exists()
    assert not settings.storage_root_path.exists()
    assert not settings.ingestion_root_path.exists()


@pytest.mark.asyncio
async def test_runtime_capability_error_route_returns_sanitized_structured_503(
    no_fallback_env: Path,
) -> None:
    from app.main import storage_durability_capability_error_response
    from app.services.storage import StorageDurabilityCapabilityError

    test_app = FastAPI()
    test_app.add_exception_handler(
        StorageDurabilityCapabilityError,
        storage_durability_capability_error_response,
    )

    @test_app.get("/mutation")
    async def blocked_mutation():
        raise StorageDurabilityCapabilityError(
            "sensitive raw failure",
            diagnostics={
                "reason": "probe_root_not_trusted",
                "root": "C:/secret/private-data",
                "mount_source": "secret-volume-source",
                "action": "unsafe dynamic action",
            },
        )

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/mutation")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == {
        "code": "storage_durability_capability_unavailable",
        "title": "Storage durability capability unavailable",
        "message": "Storage mutation is disabled because its durability contract is not proven.",
        "reason": "probe_root_not_trusted",
        "action": "Provision a validated Linux managed volume for DATA_ROOT, then restart API and workers.",
        "issues": [],
        "fix_commands": [],
        "retryable": False,
    }
    assert "secret" not in response.text.lower()


def test_pytest_environment_variable_cannot_enable_fake_durability(
    monkeypatch: pytest.MonkeyPatch,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "forged-production-value")
    monkeypatch.setattr(storage, "_is_native_windows", lambda: True)
    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(storage.StorageDurabilityCapabilityError) as raised:
            storage.namespace_durability_protocol()

    assert raised.value.diagnostics["reason"] == "native_windows_namespace_barrier_unproven"


def test_native_windows_gate_fails_before_mkdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    target = tmp_path / "must-not-be-created"
    monkeypatch.setattr(storage, "_is_native_windows", lambda: True)
    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(storage.StorageDurabilityCapabilityError) as raised:
            storage.durable_ensure_directory(target)

    assert raised.value.diagnostics["mutation_started"] is False
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires real POSIX dirfd/openat primitives")
def test_posix_probe_is_bounded_auditable_and_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    fsync_descriptors: list[int] = []
    monkeypatch.setattr(storage, "_is_native_windows", lambda: False)
    monkeypatch.setattr(
        storage,
        "_mount_info_for_path",
        lambda root: {
            "filesystem_type": "ext4",
            "mount_source": "/dev/test-managed-volume",
            "mount_point": str(tmp_path),
        },
    )
    monkeypatch.setattr(storage.os, "fsync", lambda descriptor: fsync_descriptors.append(descriptor))
    storage._clear_storage_durability_capability_cache_for_test()

    with storage._without_test_namespace_durability_adapter():
        first = storage.require_storage_durability_capability(tmp_path, force_probe=True)
        second = storage.require_storage_durability_capability(tmp_path)
        child = tmp_path / "child"
        child.mkdir()
        sync_count_before_child = len(fsync_descriptors)
        child_capability = storage.require_storage_durability_capability(child)

    assert first.supported is True
    assert first.probe_version == storage.STORAGE_DURABILITY_PROBE_VERSION
    assert first.filesystem_type == "ext4"
    assert first.mount_signature
    assert first.process_id == os.getpid()
    assert first.cache_ttl_seconds == 30.0
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert child_capability.cache_hit is True
    assert child_capability.inode != first.inode
    assert len(fsync_descriptors) == sync_count_before_child
    assert len(fsync_descriptors) >= 5
    assert not list(tmp_path.glob(".symbograph-durability-probe-*"))


@pytest.mark.skipif(os.name != "posix", reason="requires real POSIX dirfd/openat primitives")
def test_known_windows_shared_mount_fails_closed_before_probe_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    monkeypatch.setattr(storage, "_is_native_windows", lambda: False)
    monkeypatch.setattr(
        storage,
        "_mount_info_for_path",
        lambda root: {
            "filesystem_type": "virtiofs",
            "mount_source": "docker-desktop-bind",
            "mount_point": str(root),
        },
    )
    storage._clear_storage_durability_capability_cache_for_test()

    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(storage.StorageDurabilityCapabilityError) as raised:
            storage.require_storage_durability_capability(tmp_path, force_probe=True)

    assert raised.value.diagnostics["filesystem_type"] == "virtiofs"
    assert not list(tmp_path.glob(".symbograph-durability-probe-*"))


@pytest.mark.skipif(os.name != "posix", reason="requires real POSIX dirfd/openat primitives")
def test_posix_component_symlink_is_rejected_before_child_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    trusted = tmp_path / "trusted"
    outside = tmp_path / "outside"
    trusted.mkdir()
    outside.mkdir()
    (trusted / "redirect").symlink_to(outside, target_is_directory=True)

    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(storage.StorageDurabilityCapabilityError) as raised:
            storage.durable_ensure_directory(trusted / "redirect" / "must-not-exist")

    assert raised.value.diagnostics["reason"] == "path_identity_check_failed"
    assert not (outside / "must-not-exist").exists()


@pytest.mark.skipif(os.name != "posix", reason="requires Linux mount identity")
def test_missing_mount_identity_fails_closed_before_probe_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    monkeypatch.setattr(
        storage,
        "_mount_info_for_path",
        lambda _root: {"filesystem_type": None, "mount_source": None, "mount_point": None},
    )
    storage._clear_storage_durability_capability_cache_for_test()

    with storage._without_test_namespace_durability_adapter():
        with pytest.raises(storage.StorageDurabilityCapabilityError) as raised:
            storage.require_storage_durability_capability(tmp_path, force_probe=True)

    assert raised.value.diagnostics["reason"] == "mount_identity_unavailable"
    assert not list(tmp_path.glob(".symbograph-durability-probe-*"))


def test_shared_filesystem_family_and_source_are_rejected() -> None:
    from app.services import storage

    base = {
        "mount_id": "1",
        "parent_id": "0",
        "major_minor": "8:1",
        "mount_root": "/",
        "mount_point": "/app/data",
        "mount_options": "rw",
        "super_options": "rw",
        "mount_signature": "digest",
    }
    assert storage._unsupported_shared_mount(
        {**base, "filesystem_type": "nfs4", "mount_source": "server:/data"}
    )
    assert storage._unsupported_shared_mount(
        {**base, "filesystem_type": "ext4", "mount_source": "/run/desktop/mnt/host/c/data"}
    )
    assert not storage._unsupported_shared_mount(
        {**base, "filesystem_type": "ext4", "mount_source": "/dev/vdb1"}
    )


def test_startup_worker_and_compose_storage_gate_order_is_explicit() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    main_source = (repository_root / "apps/api/app/main.py").read_text(encoding="utf-8")
    celery_source = (repository_root / "apps/worker/worker_app/celery_app.py").read_text(encoding="utf-8")
    tasks_source = (repository_root / "apps/worker/worker_app/tasks.py").read_text(encoding="utf-8")
    watcher_source = (repository_root / "apps/worker/worker_app/watcher.py").read_text(encoding="utf-8")
    compose_source = (repository_root / "infra/docker-compose.yml").read_text(encoding="utf-8")
    infra_readme = (repository_root / "infra/README.md").read_text(encoding="utf-8")

    early_gate = main_source.index("_EARLY_STORAGE_DURABILITY_CAPABILITY = ensure_storage_durability_ready")
    assert early_gate < main_source.index("from app.api import router")
    assert early_gate < main_source.index("from app.db import ensure_schema")
    worker_gate = celery_source.index("WORKER_STORAGE_DURABILITY_CAPABILITY = ensure_storage_durability_ready")
    assert worker_gate < celery_source.index('celery_app = Celery("knowledge_base_worker"')
    assert tasks_source.index("from worker_app.celery_app import celery_app") < tasks_source.index(
        "from app.db import SessionLocal"
    )
    task_gate = tasks_source.index("ensure_storage_durability_ready(settings=before_refresh)")
    assert task_gate < tasks_source.index("refresh_runtime_settings_if_needed(force=True)")
    assert "storage_root.mkdir" not in watcher_source
    assert "symbograph-data:/app/data" in compose_source
    assert "../data:/app/data" not in compose_source
    assert compose_source.count("source: ${SAMPLE_IMPORT_PATH:-./sample-import}") == 2
    assert "target: /app/import/sample" in compose_source
    assert "read_only: true" in compose_source
    assert (repository_root / "infra/sample-import/.gitkeep").is_file()
    assert "set `SAMPLE_IMPORT_PATH` explicitly" in infra_readme
    assert "../sample-raw-import" not in infra_readme
    assert 'SAMPLE_IMPORT_PATH = "../data/Sample/storage"' not in infra_readme
    assert "symbograph_raw_source_manifest_v1" in infra_readme
    assert "Never point\nit at `../data/Sample`" in infra_readme


@pytest.mark.skipif(
    not hasattr(os, "fork") or not hasattr(os, "register_at_fork"),
    reason="requires POSIX fork callbacks",
)
def test_fork_child_replaces_inherited_locked_capability_state() -> None:
    """A child must not inherit a cache lock held by a vanished parent thread."""

    from app.services import storage

    old_lock = storage._CAPABILITY_CACHE_LOCK
    lock_acquired = Event()
    release_lock = Event()

    def hold_old_lock() -> None:
        with old_lock:
            lock_acquired.set()
            release_lock.wait(timeout=10)

    holder = Thread(target=hold_old_lock, daemon=True)
    holder.start()
    assert lock_acquired.wait(timeout=2), "test thread did not acquire the pre-fork cache lock"

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions are reported through the pipe
        try:
            os.close(read_fd)
            storage._clear_storage_durability_capability_cache_for_test()
            valid = (
                storage._CAPABILITY_CACHE_LOCK is not old_lock
                and storage._CAPABILITY_CACHE == {}
                and storage._CAPABILITY_CACHE_PROCESS_ID == os.getpid()
            )
            os.write(write_fd, b"ok" if valid else b"invalid")
        except BaseException as exc:
            os.write(write_fd, f"error:{type(exc).__name__}".encode("ascii", errors="replace"))
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 3.0)
        if not readable:
            os.kill(child_pid, signal.SIGKILL)
            pytest.fail("fork child blocked on an inherited storage capability lock")
        assert os.read(read_fd, 64) == b"ok"
    finally:
        os.close(read_fd)
        os.waitpid(child_pid, 0)
        release_lock.set()
        holder.join(timeout=2)

    assert not holder.is_alive()


def test_explicit_fixture_adapter_supports_tests_but_not_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    from app.services import storage

    monkeypatch.setattr(storage, "_is_native_windows", lambda: True)
    target = tmp_path / "fixture-created"
    assert storage.namespace_durability_protocol() == "windows_pytest_adapter_v1"
    storage.durable_ensure_directory(target)
    assert target.is_dir()

    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(storage.StorageDurabilityCapabilityError) as raised:
        storage.namespace_durability_protocol()
    assert raised.value.diagnostics["reason"] == "test_adapter_forbidden_in_production"


def test_explicit_fixture_adapter_covers_verified_posix_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_fallback_env: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX descriptor contract")
    from app.services import storage

    root = tmp_path / "verified-root"
    root.mkdir()

    def forbidden_probe(*_args, **_kwargs):
        raise AssertionError("real durability probe must not run in fixture namespace")

    monkeypatch.setattr(
        storage,
        "_run_posix_storage_durability_probe",
        forbidden_probe,
    )
    with storage._authorized_posix_directory_fd(root) as (
        lexical_root,
        descriptor,
        capability,
    ):
        assert lexical_root == root
        assert descriptor >= 0
        assert capability.supported is True
        assert capability.adapter == "explicit_pytest_fixture_adapter_v1"
