import hashlib
import errno
import asyncio
import codecs
import ctypes
import io
import json
import os
import re
import stat
import unicodedata
import zipfile
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import RLock
from time import monotonic
from typing import Any, BinaryIO, Iterator
from uuid import uuid4

from weakref import WeakKeyDictionary

from fastapi import UploadFile

from app.core.config import get_settings


ALLOWED_UPLOAD_SUFFIXES = frozenset(
    {".pdf", ".ipynb", ".md", ".markdown", ".txt", ".docx", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".bmp", ".html", ".htm"}
)
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')
UPLOAD_FILENAME_VALIDATION_PROTOCOL_VERSION = "nfkc_closed_path_filename_v2"
UPLOAD_MULTIPART_FILENAME_PROTOCOL_VERSION = (
    "raw_content_disposition_filename_preservation_v1"
)
UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION = "upload_content_signature_v1"
UPLOAD_CONTENT_SIGNATURE_MAX_HEADER_BYTES = 8 * 1024
UPLOAD_CONTENT_SIGNATURE_MAX_ZIP_ENTRIES = 4096
UPLOAD_CONTENT_SIGNATURE_MAX_ZIP_NAME_BYTES = 1024 * 1024
UPLOAD_CONTENT_SIGNATURE_MAX_METADATA_BYTES = 2 * 1024 * 1024
UPLOAD_CONTENT_SIGNATURE_MAX_NOTEBOOK_BYTES = 32 * 1024 * 1024
UPLOAD_CONTENT_SIGNATURE_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
UPLOAD_CONTENT_SIGNATURE_MAX_EXPANSION_RATIO = 200
UPLOAD_CONTENT_SIGNATURE_ALLOWED_ZIP_METHODS = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)
# This explicit Unicode 17 / UTS #39 direct slash-or-backslash skeleton set
# freezes the decision across Python/Unicode database upgrades. NFKC catches
# compatibility forms; the remaining lookalikes need a closed denylist.
PATH_SEPARATOR_CONFUSABLES = frozenset(
    {
        "\u1735",  # PHILIPPINE SINGLE PUNCTUATION
        "\u2041",  # CARET INSERTION POINT
        "\u2044",  # FRACTION SLASH
        "\u2215",  # DIVISION SLASH
        "\u2216",  # SET MINUS
        "\u244a",  # OCR DOUBLE BACKSLASH
        "\u2571",  # BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT
        "\u2572",  # BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT
        "\u27c8",  # REVERSE SOLIDUS PRECEDING SUBSET
        "\u27cb",  # MATHEMATICAL RISING DIAGONAL
        "\u27cd",  # MATHEMATICAL FALLING DIAGONAL
        "\u29f5",  # REVERSE SOLIDUS OPERATOR
        "\u29f6",  # SOLIDUS WITH OVERBAR
        "\u29f8",  # BIG SOLIDUS
        "\u29f9",  # BIG REVERSE SOLIDUS
        "\u2cc6",  # COPTIC CAPITAL LETTER OLD COPTIC ESH
        "\u2cc7",  # COPTIC SMALL LETTER OLD COPTIC ESH
        "\u2cf9",  # COPTIC OLD NUBIAN FULL STOP
        "\u2afb",  # TRIPLE SOLIDUS BINARY RELATION
        "\u2afd",  # DOUBLE SOLIDUS OPERATOR
        "\u2f02",  # KANGXI RADICAL DOT
        "\u2f03",  # KANGXI RADICAL SLASH
        "\u3033",  # VERTICAL KANA REPEAT MARK UPPER HALF
        "\u30ce",  # KATAKANA LETTER NO
        "\u31d3",  # CJK STROKE SP
        "\u31d4",  # CJK STROKE D
        "\u4e36",  # CJK UNIFIED IDEOGRAPH-4E36
        "\u4e3f",  # CJK UNIFIED IDEOGRAPH-4E3F
        "\ufe68",  # SMALL REVERSE SOLIDUS
        "\uff0f",  # FULLWIDTH SOLIDUS
        "\uff3c",  # FULLWIDTH REVERSE SOLIDUS
        "\U0001d20f",  # GREEK VOCAL NOTATION SYMBOL-16
        "\U0001d23a",  # GREEK INSTRUMENTAL NOTATION SYMBOL-47
        "\U0001d23b",  # GREEK INSTRUMENTAL NOTATION SYMBOL-48
    }
)
WINDOWS_RESERVED_FILENAME_STEMS = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_CONTENT_KIND_BY_SUFFIX = {
    ".pdf": "pdf",
    ".ipynb": "jupyter_notebook",
    ".md": "utf8_text",
    ".markdown": "utf8_text",
    ".txt": "utf8_text",
    ".docx": "ooxml_word",
    ".pptx": "ooxml_presentation",
    ".ppt": "ole_presentation",
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".bmp": "bmp",
    ".html": "html",
    ".htm": "html",
}
SOURCE_SNAPSHOT_DIRECTORY = "source_snapshots"
SOURCE_SNAPSHOT_PROTOCOL_VERSION = "sha256_content_addressed_readonly_source_snapshot_v2"
SOURCE_SLOT_DIRECTORY = "source_slots"
LOGICAL_SOURCE_SLOT_PROTOCOL_VERSION = "nfkc_casefold_relative_slot_v1"
UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION = "nfkc_casefold_upload_filename_v1"
NAMESPACE_DURABILITY_PROTOCOLS = frozenset(
    {
        "posix_parent_directory_fsync_v1",
        # Decoder compatibility only. New production intents never select
        # either Windows protocol: native Windows has no proven ordinary-
        # service-account namespace barrier, and the test adapter is injected
        # explicitly by pytest fixtures.
        "windows_volume_flush_v1",
        "windows_pytest_adapter_v1",
    }
)
STORAGE_DURABILITY_PROBE_VERSION = "bounded_file_rename_dual_parent_fsync_unlink_v1"
KNOWN_UNSUPPORTED_SHARED_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "drvfs",
        "fuse.grpcfuse",
        "fuse.osxfs",
        "fuse.sshfs",
        "fuseblk",
        "gcsfuse",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "overlay",
        "ramfs",
        "smb3",
        "tmpfs",
        "virtiofs",
    }
)
_UNSUPPORTED_FILESYSTEM_PREFIXES = ("fuse.", "nfs", "smb")
_UNSUPPORTED_MOUNT_SOURCE_MARKERS = (
    "docker-desktop",
    "grpcfuse",
    "host_mnt",
    "mnt/host",
    "osxfs",
    "virtiofs",
    "wsl",
)
_MOUNTINFO_MAX_LINES = 16_384
_MOUNTINFO_MAX_BYTES = 4 * 1024 * 1024
_CAPABILITY_CACHE_TTL_SECONDS = 30.0
_CAPABILITY_CACHE_MAX_ENTRIES = 128
_PROBE_BYTES = b"symbograph-storage-durability-probe-v1\n"


class UploadValidationError(ValueError):
    """Raised before an untrusted upload is committed to knowledge-base storage."""


class UploadChecksumMismatchError(UploadValidationError):
    """Raised when admitted bytes do not match a caller-bound SHA-256 digest."""


class UploadTooLargeError(UploadValidationError):
    """Raised when the streamed upload exceeds the configured hard byte limit."""


@dataclass(frozen=True)
class ValidatedUploadContent:
    filename: str
    suffix: str
    checksum: str
    size_bytes: int
    content_kind: str
    filename_protocol_version: str = UPLOAD_FILENAME_VALIDATION_PROTOCOL_VERSION
    multipart_filename_protocol_version: str = (
        UPLOAD_MULTIPART_FILENAME_PROTOCOL_VERSION
    )
    content_signature_protocol_version: str = (
        UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION
    )

    def audit_card(self) -> dict[str, object]:
        return {
            "filename_protocol_version": self.filename_protocol_version,
            "filename_protocol_hash": (
                upload_filename_validation_protocol_hash()
            ),
            "multipart_filename_protocol_version": (
                self.multipart_filename_protocol_version
            ),
            "content_signature_protocol_version": (
                self.content_signature_protocol_version
            ),
            "content_signature_protocol_hash": (
                upload_content_signature_protocol_hash()
            ),
            "suffix": self.suffix,
            "content_kind": self.content_kind,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }


class SourceSnapshotError(RuntimeError):
    """Raised when immutable attempt source bytes cannot be proven durable."""


class SourceSnapshotNotFoundError(SourceSnapshotError):
    """Raised when a verified final-open proves the requested snapshot is absent."""


class DirectoryDurabilityError(OSError):
    """Raised after a namespace mutation lacks a proven directory barrier."""


class StorageDurabilityCapabilityError(RuntimeError):
    """Raised before mutation when the active filesystem contract is unproven."""

    code = "storage_durability_capability_unavailable"

    def __init__(self, message: str, *, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = {"code": self.code, **diagnostics}


def storage_deployment_contract() -> dict[str, object]:
    """Public, non-sensitive support boundary for storage mutation."""

    return {
        "protocol_version": "storage_deployment_support_boundary_v1",
        "native_windows_production_supported": False,
        "required_runtime": "linux_docker_managed_volume",
        "current_runtime_family": "native_windows" if _is_native_windows() else "posix",
        "fail_closed": True,
    }


@dataclass(frozen=True)
class VerifiedSourceIdentity:
    protocol_version: str
    root_device_id: int
    root_inode: int
    device_id: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    link_count: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "protocol_version": self.protocol_version,
            "root_device_id": self.root_device_id,
            "root_inode": self.root_inode,
            "device_id": self.device_id,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "link_count": self.link_count,
        }


FROZEN_SOURCE_SNAPSHOT_PROTOCOL_VERSION = "bounded_frozen_source_snapshot_v1"


@dataclass(frozen=True)
class FrozenSourceSnapshot:
    """One bounded immutable parser input backed by a verified snapshot card.

    ``content_bytes`` is read with the configured upload hard limit. It is
    deliberately excluded from repr/comparison so content cannot leak through
    diagnostics and identity comparisons stay card-only.
    """

    canonical_path: Path
    checksum: str
    size_bytes: int
    content_kind: str
    suffix: str
    identity: VerifiedSourceIdentity
    content_bytes: bytes = field(repr=False, compare=False)
    protocol_version: str = FROZEN_SOURCE_SNAPSHOT_PROTOCOL_VERSION

    @property
    def filename(self) -> str:
        return self.canonical_path.name

    def identity_card(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "canonical_path": str(self.canonical_path),
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "content_kind": self.content_kind,
            "suffix": self.suffix,
            "verified_source_identity": self.identity.as_dict(),
        }


def validate_frozen_source_snapshot(
    snapshot: FrozenSourceSnapshot,
) -> FrozenSourceSnapshot:
    """Strictly validate every frozen parser-input card field and its bytes."""

    if type(snapshot) is not FrozenSourceSnapshot:
        raise TypeError("FrozenSourceSnapshot is required")
    if type(snapshot.protocol_version) is not str or (
        snapshot.protocol_version != FROZEN_SOURCE_SNAPSHOT_PROTOCOL_VERSION
    ):
        raise TypeError("Frozen source snapshot protocol is invalid")
    if not isinstance(snapshot.canonical_path, Path) or (
        not snapshot.canonical_path.is_absolute()
    ):
        raise TypeError("Frozen source snapshot path must be absolute")
    if type(snapshot.checksum) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", snapshot.checksum
    ):
        raise TypeError("Frozen source snapshot checksum is invalid")
    if type(snapshot.size_bytes) is not int or snapshot.size_bytes < 0:
        raise TypeError("Frozen source snapshot size is invalid")
    if type(snapshot.content_kind) is not str or not snapshot.content_kind:
        raise TypeError("Frozen source snapshot content kind is invalid")
    if type(snapshot.suffix) is not str or (
        snapshot.suffix != snapshot.canonical_path.suffix.casefold()
    ):
        raise TypeError("Frozen source snapshot suffix is invalid")
    if type(snapshot.content_bytes) is not bytes or (
        len(snapshot.content_bytes) != snapshot.size_bytes
        or hashlib.sha256(snapshot.content_bytes).hexdigest() != snapshot.checksum
    ):
        raise TypeError("Frozen source snapshot bytes/card validation failed")
    identity = snapshot.identity
    if type(identity) is not VerifiedSourceIdentity:
        raise TypeError("Frozen source snapshot identity is invalid")
    allowed_identity_protocols = {
        "posix_openat_nofollow_fstat_v1",
        "explicit_test_adapter_lstat_fstat_v1",
        "explicit_parser_test_freeze_v1",
    }
    if type(identity.protocol_version) is not str or (
        identity.protocol_version not in allowed_identity_protocols
    ):
        raise TypeError("Frozen source snapshot identity protocol is invalid")
    for field_name in (
        "root_device_id",
        "root_inode",
        "device_id",
        "inode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "link_count",
    ):
        if type(getattr(identity, field_name)) is not int:
            raise TypeError("Frozen source snapshot identity field is invalid")
    if identity.size_bytes != snapshot.size_bytes or identity.link_count != 1:
        raise TypeError("Frozen source snapshot identity does not bind its bytes")
    validated = _validate_seekable_upload_content(
        io.BytesIO(snapshot.content_bytes),
        filename=snapshot.filename,
        max_bytes=max(snapshot.size_bytes, 1),
        expected_checksum=snapshot.checksum,
    )
    if (
        validated.size_bytes != snapshot.size_bytes
        or validated.suffix != snapshot.suffix
        or validated.content_kind != snapshot.content_kind
    ):
        raise TypeError("Frozen source snapshot content contract drifted")
    return snapshot


@dataclass(frozen=True)
class _FrozenReadonlyImportRoot:
    lexical_root: Path
    descriptor: int | None
    stat_identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class StorageDurabilityCapability:
    root: str
    protocol: str
    probe_version: str
    supported: bool
    checked_at: str
    duration_ms: float
    device_id: int | None
    inode: int | None
    filesystem_type: str | None
    mount_source: str | None
    mount_point: str | None
    mount_signature: str | None
    process_id: int
    cache_ttl_seconds: float
    adapter: str | None = None
    failure: str | None = None
    cache_hit: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "protocol": self.protocol,
            "probe_version": self.probe_version,
            "supported": self.supported,
            "checked_at": self.checked_at,
            "duration_ms": self.duration_ms,
            "device_id": self.device_id,
            "inode": self.inode,
            "filesystem_type": self.filesystem_type,
            "mount_source": self.mount_source,
            "mount_point": self.mount_point,
            "mount_signature": self.mount_signature,
            "process_id": self.process_id,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "adapter": self.adapter,
            "failure": self.failure,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class _InjectedTestNamespaceDurabilityAdapter:
    """No-op namespace barrier available only through the pytest fixture hook."""

    protocol: str = "windows_pytest_adapter_v1"
    name: str = "explicit_pytest_fixture_adapter_v1"

    def sync_directory(self, _directory: Path) -> None:
        return

    def capability(self, root: Path) -> StorageDurabilityCapability:
        return StorageDurabilityCapability(
            root=os.path.abspath(os.fspath(root)),
            protocol=self.protocol,
            probe_version="explicit_test_adapter_v1",
            supported=True,
            checked_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=0.0,
            device_id=None,
            inode=None,
            filesystem_type="test_adapter",
            mount_source=None,
            mount_point=None,
            mount_signature="explicit_test_adapter_v1",
            process_id=os.getpid(),
            cache_ttl_seconds=0.0,
            adapter=self.name,
        )


_TEST_DURABILITY_ADAPTER: ContextVar[_InjectedTestNamespaceDurabilityAdapter | None] = ContextVar(
    "symbograph_test_storage_durability_adapter",
    default=None,
)
_FROZEN_READONLY_IMPORT_ROOT: ContextVar[_FrozenReadonlyImportRoot | None] = (
    ContextVar(
        "symbograph_frozen_readonly_import_root",
        default=None,
    )
)
_CAPABILITY_CACHE: dict[
    tuple[int, str, int, int, str, str],
    tuple[float, StorageDurabilityCapability],
] = {}
_CAPABILITY_CACHE_LOCK = RLock()
_CAPABILITY_CACHE_PROCESS_ID = os.getpid()
_FILE_CONFIGURED_APP_ENV: str | None = None
_SOURCE_IO_SEMAPHORES: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    tuple[int, asyncio.Semaphore],
] = WeakKeyDictionary()
_SOURCE_IO_SEMAPHORES_LOCK = RLock()


def _reset_storage_durability_state_after_fork() -> None:
    global _CAPABILITY_CACHE, _CAPABILITY_CACHE_LOCK, _CAPABILITY_CACHE_PROCESS_ID
    global _SOURCE_IO_SEMAPHORES, _SOURCE_IO_SEMAPHORES_LOCK
    _CAPABILITY_CACHE = {}
    _CAPABILITY_CACHE_LOCK = RLock()
    _CAPABILITY_CACHE_PROCESS_ID = os.getpid()
    _SOURCE_IO_SEMAPHORES = WeakKeyDictionary()
    _SOURCE_IO_SEMAPHORES_LOCK = RLock()
    _TEST_DURABILITY_ADAPTER.set(None)
    _FROZEN_READONLY_IMPORT_ROOT.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_storage_durability_state_after_fork)


@contextmanager
def _use_explicit_test_namespace_durability_adapter() -> Iterator[None]:
    """Private pytest-fixture hook; production configuration cannot enable it."""

    token = _TEST_DURABILITY_ADAPTER.set(_InjectedTestNamespaceDurabilityAdapter())
    try:
        yield
    finally:
        _TEST_DURABILITY_ADAPTER.reset(token)


async def run_bounded_source_io(function, /, *args, **kwargs):
    """Run blocking source/storage I/O behind a hot-reloadable semaphore."""

    limit = int(get_settings().source_io_concurrency)
    loop = asyncio.get_running_loop()
    with _SOURCE_IO_SEMAPHORES_LOCK:
        cached = _SOURCE_IO_SEMAPHORES.get(loop)
        if cached is None or cached[0] != limit:
            cached = (limit, asyncio.Semaphore(limit))
            _SOURCE_IO_SEMAPHORES[loop] = cached
        semaphore = cached[1]
    async with semaphore:
        return await asyncio.to_thread(function, *args, **kwargs)


@contextmanager
def _without_test_namespace_durability_adapter() -> Iterator[None]:
    """Private test helper for exercising the real capability gate."""

    token = _TEST_DURABILITY_ADAPTER.set(None)
    try:
        yield
    finally:
        _TEST_DURABILITY_ADAPTER.reset(token)


def _clear_storage_durability_capability_cache_for_test() -> None:
    global _CAPABILITY_CACHE_PROCESS_ID
    with _CAPABILITY_CACHE_LOCK:
        _CAPABILITY_CACHE.clear()
        _CAPABILITY_CACHE_PROCESS_ID = os.getpid()


def _is_native_windows() -> bool:
    return os.name == "nt"


def _configured_app_env_without_directory_mutation() -> str:
    # Read the model directly so the earliest native-Windows gate remains
    # independent from runtime settings refresh and other service bootstrap.
    direct = os.getenv("APP_ENV")
    if direct is not None:
        return direct.strip().lower()
    global _FILE_CONFIGURED_APP_ENV
    if _FILE_CONFIGURED_APP_ENV is None:
        from app.core.config import Settings

        _FILE_CONFIGURED_APP_ENV = Settings().app_env.strip().lower()
    return _FILE_CONFIGURED_APP_ENV


def fail_closed_native_windows_production_before_settings() -> None:
    """Reject native-Windows production before settings can create data dirs."""

    if not _is_native_windows() or _configured_app_env_without_directory_mutation() != "production":
        return
    raise StorageDurabilityCapabilityError(
        "Native Windows production storage mutation is unsupported: no ordinary-service-account "
        "directory-entry durability barrier has been proven. Run the API/worker on Linux with a "
        "validated managed volume; do not grant raw-volume privileges as a workaround.",
        diagnostics={
            "reason": "native_windows_namespace_barrier_unproven",
            "platform": "windows",
            "app_env": "production",
            "mutation_started": False,
            "action": "Use Docker/Linux with a validated managed volume and restart the service.",
        },
    )


def _raise_native_windows_capability_error() -> None:
    raise StorageDurabilityCapabilityError(
        "Native Windows filesystem mutation is fail-closed because the namespace durability "
        "contract is unproven. Use Docker/Linux with a validated managed volume.",
        diagnostics={
            "reason": "native_windows_namespace_barrier_unproven",
            "platform": "windows",
            "mutation_started": False,
            "action": "Use Docker/Linux with a validated managed volume; raw-volume access is not supported.",
        },
    )


def namespace_durability_protocol() -> str:
    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        if _configured_app_env_without_directory_mutation() == "production":
            raise StorageDurabilityCapabilityError(
                "A test-only storage durability adapter cannot authorize a production upload intent.",
                diagnostics={
                    "reason": "test_adapter_forbidden_in_production",
                    "adapter": adapter.name,
                    "mutation_started": False,
                    "action": "Remove test injection and validate the real DATA_ROOT filesystem.",
                },
            )
        return adapter.protocol
    if _is_native_windows():
        _raise_native_windows_capability_error()
    return "posix_parent_directory_fsync_v1"


def compute_checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _verified_stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_nlink),
    )


def _verified_core_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int]:
    """Fields whose Windows adapter lstat/fstat representation is stable."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _raise_verified_root_identity_changed() -> None:
    raise SourceSnapshotError(
        "Authorized root identity changed during verified read"
    )


def _assert_posix_verified_root_identity(
    lexical_root: Path,
    descriptor: int,
    expected_identity: tuple[int, int, int, int, int, int],
) -> None:
    """Replay the pinned descriptor against the current no-follow root path."""

    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or _verified_stat_identity(descriptor_stat) != expected_identity
        ):
            _raise_verified_root_identity_changed()
        with _open_posix_directory_nofollow(lexical_root) as (
            _replayed_root,
            replayed_descriptor,
        ):
            path_stat = os.fstat(replayed_descriptor)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or _verified_stat_identity(path_stat) != expected_identity
        ):
            _raise_verified_root_identity_changed()
    except SourceSnapshotError:
        raise
    except (OSError, StorageDurabilityCapabilityError):
        _raise_verified_root_identity_changed()


@contextmanager
def _borrow_frozen_readonly_import_root(
    frozen: _FrozenReadonlyImportRoot,
    lexical_root: Path,
) -> Iterator[tuple[Path, int, None]]:
    if frozen.lexical_root != lexical_root or frozen.descriptor is None:
        _raise_verified_root_identity_changed()
    descriptor = os.dup(frozen.descriptor)
    try:
        yield lexical_root, descriptor, None
    finally:
        with suppress(OSError):
            os.close(descriptor)


@contextmanager
def freeze_verified_readonly_import_root(
    authorized_root: Path,
) -> Iterator[None]:
    """Pin one read-only import root across manifest and all file reads."""

    lexical_root = Path(os.path.abspath(authorized_root))
    existing = _FROZEN_READONLY_IMPORT_ROOT.get()
    if existing is not None:
        if existing.lexical_root != lexical_root:
            _raise_verified_root_identity_changed()
        yield
        return

    if os.name == "posix":
        with _readonly_import_posix_directory_fd(lexical_root) as (
            _root,
            descriptor,
            _capability,
        ):
            root_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise SourceSnapshotError(
                    "Read-only import root is not one authorized directory"
                )
            root_identity = _verified_stat_identity(root_stat)
            _assert_posix_verified_root_identity(
                lexical_root,
                descriptor,
                root_identity,
            )
            token = _FROZEN_READONLY_IMPORT_ROOT.set(
                _FrozenReadonlyImportRoot(
                    lexical_root=lexical_root,
                    descriptor=descriptor,
                    stat_identity=root_identity,
                )
            )
            try:
                yield
            finally:
                try:
                    _assert_posix_verified_root_identity(
                        lexical_root,
                        descriptor,
                        root_identity,
                    )
                finally:
                    _FROZEN_READONLY_IMPORT_ROOT.reset(token)
        return

    # Native Windows reaches this branch only through the explicit pytest
    # adapter. Production readonly imports require the POSIX descriptor
    # protocol above.
    require_storage_durability_capability(lexical_root)
    try:
        root_stat = lexical_root.lstat()
    except OSError:
        _raise_verified_root_identity_changed()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SourceSnapshotError(
            "Read-only import root is not one authorized directory"
        )
    root_identity = _verified_stat_identity(root_stat)
    token = _FROZEN_READONLY_IMPORT_ROOT.set(
        _FrozenReadonlyImportRoot(
            lexical_root=lexical_root,
            descriptor=None,
            stat_identity=root_identity,
        )
    )
    try:
        yield
    finally:
        try:
            try:
                root_after = lexical_root.lstat()
            except OSError:
                _raise_verified_root_identity_changed()
            if (
                not stat.S_ISDIR(root_after.st_mode)
                or _verified_stat_identity(root_after) != root_identity
            ):
                _raise_verified_root_identity_changed()
        finally:
            _FROZEN_READONLY_IMPORT_ROOT.reset(token)


@contextmanager
def _open_verified_source_file(
    source_path: Path,
    authorized_root: Path,
    *,
    readonly_import: bool,
) -> Iterator[tuple[BinaryIO, VerifiedSourceIdentity]]:
    """Final-open one regular file under one explicit source-root contract."""

    lexical_root = Path(os.path.abspath(authorized_root))
    lexical_source = Path(os.path.abspath(source_path))
    try:
        relative = lexical_source.relative_to(lexical_root)
    except ValueError as exc:
        raise SourceSnapshotError(
            "Source file is outside its authorized root"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SourceSnapshotError("Source file has an invalid relative identity")

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if _is_native_windows() and adapter is None:
        _raise_native_windows_capability_error()

    if os.name == "posix":
        parent_descriptor: int | None = None
        file_handle: BinaryIO | None = None
        try:
            frozen_root = (
                _FROZEN_READONLY_IMPORT_ROOT.get()
                if readonly_import
                else None
            )
            if frozen_root is not None:
                root_context = _borrow_frozen_readonly_import_root(
                    frozen_root,
                    lexical_root,
                )
            else:
                root_context = (
                    _readonly_import_posix_directory_fd(lexical_root)
                    if readonly_import
                    else _authorized_posix_directory_fd(lexical_root)
                )
            with root_context as (
                _root,
                root_descriptor,
                _capability,
            ):
                root_stat = os.fstat(root_descriptor)
                root_identity = _verified_stat_identity(root_stat)
                if (
                    frozen_root is not None
                    and root_identity != frozen_root.stat_identity
                ):
                    _raise_verified_root_identity_changed()
                _assert_posix_verified_root_identity(
                    lexical_root,
                    root_descriptor,
                    root_identity,
                )
                parent_descriptor = os.dup(root_descriptor)
                directory_flags = (
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                for part in relative.parts[:-1]:
                    next_descriptor = os.open(
                        part,
                        directory_flags,
                        dir_fd=parent_descriptor,
                    )
                    os.close(parent_descriptor)
                    parent_descriptor = next_descriptor
                    if os.fstat(parent_descriptor).st_dev != root_stat.st_dev:
                        raise SourceSnapshotError(
                            "Source path crossed an unauthorized mounted filesystem"
                        )
                leaf = relative.parts[-1]
                descriptor = os.open(
                    leaf,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                descriptor_stat = os.fstat(descriptor)
                path_stat = os.stat(
                    leaf,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(descriptor_stat.st_mode)
                    or int(descriptor_stat.st_nlink) != 1
                    or _verified_stat_identity(descriptor_stat)
                    != _verified_stat_identity(path_stat)
                    or descriptor_stat.st_dev != root_stat.st_dev
                ):
                    os.close(descriptor)
                    raise SourceSnapshotError(
                        "Final-open source identity is not one authorized regular file"
                    )
                identity = VerifiedSourceIdentity(
                    protocol_version=(
                        "posix_readonly_import_openat_nofollow_fstat_v1"
                        if readonly_import
                        else "posix_openat_nofollow_fstat_v1"
                    ),
                    root_device_id=int(root_stat.st_dev),
                    root_inode=int(root_stat.st_ino),
                    device_id=int(descriptor_stat.st_dev),
                    inode=int(descriptor_stat.st_ino),
                    size_bytes=int(descriptor_stat.st_size),
                    mtime_ns=int(descriptor_stat.st_mtime_ns),
                    ctime_ns=int(descriptor_stat.st_ctime_ns),
                    link_count=int(descriptor_stat.st_nlink),
                )
                file_handle = os.fdopen(descriptor, "rb", closefd=True)
                try:
                    yield file_handle, identity
                finally:
                    try:
                        descriptor_after = os.fstat(file_handle.fileno())
                        path_after = os.stat(
                            leaf,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        if (
                            not stat.S_ISREG(descriptor_after.st_mode)
                            or not stat.S_ISREG(path_after.st_mode)
                            or _verified_stat_identity(descriptor_after)
                            != _verified_stat_identity(descriptor_stat)
                            or _verified_stat_identity(path_after)
                            != _verified_stat_identity(descriptor_stat)
                        ):
                            raise SourceSnapshotError(
                                "Source identity changed while its verified handle was open"
                            )
                    finally:
                        _assert_posix_verified_root_identity(
                            lexical_root,
                            root_descriptor,
                            root_identity,
                        )
        except SourceSnapshotError:
            raise
        except FileNotFoundError:
            raise SourceSnapshotNotFoundError(
                "Verified source snapshot is absent"
            ) from None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SourceSnapshotError(
                    "Source path traverses a symbolic link or reparse point"
                ) from None
            raise SourceSnapshotError(
                f"Final-open source verification failed: {_sanitized_os_failure(exc)}"
            ) from None
        finally:
            if file_handle is not None:
                file_handle.close()
            if parent_descriptor is not None:
                with suppress(OSError):
                    os.close(parent_descriptor)
        return

    # Native Windows production is rejected above.  This branch exists only
    # for the explicit test adapter, so Windows CI can exercise higher-level
    # recovery state machines without pretending to prove a reparse-safe
    # service-account handle contract.
    require_storage_durability_capability(lexical_root)
    cursor = lexical_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise SourceSnapshotError(
                "Source path traverses a symbolic link or reparse point"
            )
    try:
        root_stat = lexical_root.stat()
        root_identity = _verified_stat_identity(root_stat)
        frozen_root = (
            _FROZEN_READONLY_IMPORT_ROOT.get()
            if readonly_import
            else None
        )
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or (
                frozen_root is not None
                and (
                    frozen_root.lexical_root != lexical_root
                    or frozen_root.stat_identity != root_identity
                )
            )
        ):
            _raise_verified_root_identity_changed()
        before = lexical_source.lstat()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        descriptor = os.open(lexical_source, flags)
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor_before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or _verified_core_identity(before)
            != _verified_core_identity(descriptor_before)
            or descriptor_before.st_dev != root_stat.st_dev
        ):
            handle.close()
            raise SourceSnapshotError(
                "Final-open source identity is not one authorized regular file"
            )
        identity = VerifiedSourceIdentity(
            protocol_version="explicit_test_adapter_lstat_fstat_v1",
            root_device_id=int(root_stat.st_dev),
            root_inode=int(root_stat.st_ino),
            device_id=int(before.st_dev),
            inode=int(before.st_ino),
            size_bytes=int(before.st_size),
            mtime_ns=int(before.st_mtime_ns),
            ctime_ns=int(before.st_ctime_ns),
            link_count=int(before.st_nlink),
        )
        try:
            yield handle, identity
        finally:
            try:
                try:
                    descriptor_after = os.fstat(handle.fileno())
                    path_after = lexical_source.lstat()
                    if (
                        not stat.S_ISREG(descriptor_after.st_mode)
                        or not stat.S_ISREG(path_after.st_mode)
                        or _verified_core_identity(descriptor_after)
                        != _verified_core_identity(descriptor_before)
                        or _verified_stat_identity(path_after)
                        != _verified_stat_identity(before)
                    ):
                        raise SourceSnapshotError(
                            "Source identity changed while its verified handle was open"
                        )
                finally:
                    root_after = lexical_root.lstat()
                    if (
                        not stat.S_ISDIR(root_after.st_mode)
                        or _verified_stat_identity(root_after) != root_identity
                    ):
                        _raise_verified_root_identity_changed()
            finally:
                handle.close()
    except SourceSnapshotError:
        raise
    except FileNotFoundError:
        raise SourceSnapshotNotFoundError(
            "Verified source snapshot is absent"
        ) from None
    except OSError as exc:
        raise SourceSnapshotError(
            f"Final-open source verification failed: {_sanitized_os_failure(exc)}"
        ) from None


@contextmanager
def open_verified_source_file(
    source_path: Path,
    authorized_root: Path,
) -> Iterator[tuple[BinaryIO, VerifiedSourceIdentity]]:
    """Open a source on a root that passed the mutation-durability contract."""

    with _open_verified_source_file(
        source_path,
        authorized_root,
        readonly_import=False,
    ) as opened:
        yield opened


@contextmanager
def open_verified_readonly_import_file(
    source_path: Path,
    authorized_root: Path,
) -> Iterator[tuple[BinaryIO, VerifiedSourceIdentity]]:
    """Open an allowlisted source from an explicitly read-only import mount.

    Read-only operator input does not need rename/unlink durability, but it
    must still use pinned no-follow descriptors and remain byte/checksum
    stable for the complete read. Writable production mounts are rejected so
    this entry point cannot bypass the normal DATA_ROOT durability gate.
    """

    with _open_verified_source_file(
        source_path,
        authorized_root,
        readonly_import=True,
    ) as opened:
        yield opened


def verified_source_checksum(
    source_path: Path,
    authorized_root: Path,
) -> tuple[str, VerifiedSourceIdentity]:
    digest = hashlib.sha256()
    with open_verified_source_file(source_path, authorized_root) as (
        handle,
        identity,
    ):
        for chunk in iter(lambda: handle.read(UPLOAD_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest(), identity


def verified_readonly_import_checksum(
    source_path: Path,
    authorized_root: Path,
) -> tuple[str, VerifiedSourceIdentity]:
    digest = hashlib.sha256()
    with open_verified_readonly_import_file(source_path, authorized_root) as (
        handle,
        identity,
    ):
        for chunk in iter(lambda: handle.read(UPLOAD_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest(), identity


def upload_filename_validation_protocol_hash() -> str:
    payload = {
        "protocol_version": UPLOAD_FILENAME_VALIDATION_PROTOCOL_VERSION,
        "multipart_protocol_version": UPLOAD_MULTIPART_FILENAME_PROTOCOL_VERSION,
        "unicode_normalization": "NFKC",
        "separator_confusable_codepoints": sorted(
            f"U+{ord(character):04X}"
            for character in PATH_SEPARATOR_CONFUSABLES
        ),
        "invalid_ascii_characters": sorted(INVALID_FILENAME_CHARS),
        "unicode_category_deny_prefix": "C",
        "path_grammars": ["posix", "windows_drive_root_unc_ads"],
        "windows_reserved_stems": sorted(WINDOWS_RESERVED_FILENAME_STEMS),
        "character_limit": 255,
        "utf8_byte_limit": 1024,
        "boundary_whitespace": "reject_before_and_after_normalization",
        "trailing_dot_or_space": "reject",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upload_content_signature_protocol_hash() -> str:
    payload = {
        "protocol_version": UPLOAD_CONTENT_SIGNATURE_PROTOCOL_VERSION,
        "suffix_kind_allowlist": sorted(_CONTENT_KIND_BY_SUFFIX.items()),
        "text_decode": "strict_utf8_optional_bom_no_binary_controls_v1",
        "html_contract": "bounded_utf8_html_marker_v1",
        "notebook_contract": "bounded_json_notebook_shape_v1",
        "pdf_contract": "header_version_and_terminal_eof_v1",
        "png_contract": "signature_ihdr_terminal_iend_v1",
        "jpeg_contract": "soi_marker_and_terminal_eoi_v1",
        "bmp_contract": "bitmap_header_size_offset_dib_v1",
        "ole_contract": "cfbf_header_geometry_v1",
        "ooxml_contract": "bounded_safe_zip_required_parts_content_type_v1",
        "zip_entry_limit": UPLOAD_CONTENT_SIGNATURE_MAX_ZIP_ENTRIES,
        "zip_name_byte_limit": UPLOAD_CONTENT_SIGNATURE_MAX_ZIP_NAME_BYTES,
        "zip_metadata_byte_limit": UPLOAD_CONTENT_SIGNATURE_MAX_METADATA_BYTES,
        "zip_uncompressed_byte_limit": (
            UPLOAD_CONTENT_SIGNATURE_MAX_UNCOMPRESSED_BYTES
        ),
        "zip_expansion_ratio_limit": (
            UPLOAD_CONTENT_SIGNATURE_MAX_EXPANSION_RATIO
        ),
        "zip_method_allowlist": sorted(
            UPLOAD_CONTENT_SIGNATURE_ALLOWED_ZIP_METHODS
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_content_disposition_segments(raw_value: bytes) -> list[str]:
    if (
        not raw_value
        or len(raw_value) > UPLOAD_CONTENT_SIGNATURE_MAX_HEADER_BYTES
        or b"\x00" in raw_value
        or b"\r" in raw_value
        or b"\n" in raw_value
    ):
        raise UploadValidationError(
            "Upload Content-Disposition header is malformed"
        )
    value = raw_value.decode("latin-1")
    segments: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for character in value:
        if quoted and character == "\\" and not escaped:
            escaped = True
            current.append(character)
            continue
        if character == '"' and not escaped:
            quoted = not quoted
            current.append(character)
            continue
        if character == ";" and not quoted:
            segments.append("".join(current).strip())
            current = []
        else:
            current.append(character)
        escaped = False
    if quoted or escaped:
        raise UploadValidationError(
            "Upload Content-Disposition header has an unterminated quote"
        )
    segments.append("".join(current).strip())
    return segments


def _decode_content_disposition_filename(value: str) -> str:
    if value.startswith('"'):
        if len(value) < 2 or not value.endswith('"'):
            raise UploadValidationError(
                "Upload Content-Disposition filename is malformed"
            )
        inner = value[1:-1]
        decoded: list[str] = []
        index = 0
        while index < len(inner):
            character = inner[index]
            if (
                character == "\\"
                and index + 1 < len(inner)
                and inner[index + 1] in {'"', "\\"}
            ):
                decoded.append(inner[index + 1])
                index += 2
                continue
            decoded.append(character)
            index += 1
        value = "".join(decoded)
    elif not re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", value):
        raise UploadValidationError(
            "Upload Content-Disposition filename must be a quoted value"
        )
    raw_bytes = value.encode("latin-1")
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")


def _original_multipart_filename(upload: UploadFile) -> str | None:
    headers = getattr(upload, "headers", None)
    raw_headers = list(getattr(headers, "raw", ()) or ())
    values = [
        bytes(value)
        for key, value in raw_headers
        if bytes(key).lower() == b"content-disposition"
    ]
    if not values:
        return None
    if len(values) != 1:
        raise UploadValidationError(
            "Upload must contain one Content-Disposition header"
        )
    segments = _split_content_disposition_segments(values[0])
    if not segments or segments[0].casefold() != "form-data":
        raise UploadValidationError(
            "Upload Content-Disposition must be form-data"
        )
    parameters: dict[str, str] = {}
    for segment in segments[1:]:
        key, separator, value = segment.partition("=")
        normalized_key = key.strip().casefold()
        if (
            not separator
            or not normalized_key
            or normalized_key in parameters
        ):
            raise UploadValidationError(
                "Upload Content-Disposition parameters are ambiguous"
            )
        if normalized_key.startswith("filename") and normalized_key != "filename":
            raise UploadValidationError(
                "Extended multipart filename parameters are not accepted"
            )
        parameters[normalized_key] = value.strip()
    if "filename" not in parameters:
        raise UploadValidationError(
            "Upload Content-Disposition filename is required"
        )
    return _decode_content_disposition_filename(parameters["filename"])


def normalize_upload_filename(filename: str | None) -> str:
    if not isinstance(filename, str):
        raise UploadValidationError("Upload filename is required")
    if filename != filename.strip():
        raise UploadValidationError(
            "Upload filename must not have leading or trailing whitespace"
        )
    if any(character in PATH_SEPARATOR_CONFUSABLES for character in filename):
        raise UploadValidationError(
            "Upload filename contains a path-separator confusable"
        )
    normalized = unicodedata.normalize("NFKC", filename)
    if not normalized or normalized in {".", ".."}:
        raise UploadValidationError("Upload filename is required")
    if len(normalized) > 255:
        raise UploadValidationError("Upload filename is too long")
    if len(normalized.encode("utf-8")) > 1024:
        raise UploadValidationError("Upload filename encoding is too long")
    if normalized != normalized.strip():
        raise UploadValidationError(
            "Upload filename normalization produced boundary whitespace"
        )
    if normalized != normalized.rstrip(" ."):
        raise UploadValidationError("Upload filename has an invalid trailing character")
    if any(character in PATH_SEPARATOR_CONFUSABLES for character in normalized):
        raise UploadValidationError(
            "Upload filename contains a path-separator confusable"
        )
    if any(
        unicodedata.category(character).startswith("C")
        or character in INVALID_FILENAME_CHARS
        for character in normalized
    ):
        raise UploadValidationError("Upload filename contains an invalid character")
    windows_path = PureWindowsPath(normalized)
    if (
        PurePosixPath(normalized).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or len(windows_path.parts) != 1
    ):
        raise UploadValidationError("Upload filename must not contain a path")
    if Path(normalized).name != normalized:
        raise UploadValidationError("Upload filename must not contain a path")
    reserved_stem = normalized.split(".", 1)[0].casefold()
    if reserved_stem in WINDOWS_RESERVED_FILENAME_STEMS:
        raise UploadValidationError(
            "Upload filename uses a reserved platform name"
        )
    return normalized


def validated_upload_filename(upload: UploadFile) -> str:
    original = _original_multipart_filename(upload)
    parser_filename = normalize_upload_filename(upload.filename)
    if original is None:
        return parser_filename
    original_filename = normalize_upload_filename(original)
    if original_filename != parser_filename:
        raise UploadValidationError(
            "Multipart parser rewrote the upload filename"
        )
    return original_filename


def normalize_upload_source_slot_key(filename: str | None) -> str:
    """Return the cross-platform logical identity of one mutable upload slot."""

    safe_filename = normalize_upload_filename(filename)
    return f"upload/{unicodedata.normalize('NFKC', safe_filename).casefold()}"


def normalize_relative_source_slot_key(relative_path: str | Path) -> str:
    """Normalize an importer-relative source address without filesystem I/O."""

    raw = unicodedata.normalize("NFKC", Path(relative_path).as_posix()).strip()
    if not raw or raw in {".", ".."} or raw.startswith("/") or "\x00" in raw:
        raise UploadValidationError("Relative source slot path is invalid")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UploadValidationError("Relative source slot path must not escape its root")
    return f"relative/{'/'.join(parts).casefold()}"


def source_slot_key_for_path(source_path: Path, storage_root: Path) -> str:
    lexical_source = Path(os.path.abspath(source_path))
    lexical_root = Path(os.path.abspath(storage_root))
    try:
        relative = lexical_source.relative_to(lexical_root)
    except ValueError as exc:
        raise UploadValidationError(
            "Source slot path is outside knowledge-base storage"
        ) from exc
    return normalize_relative_source_slot_key(relative)


def _knowledge_base_paths(
    *,
    knowledge_base_id: str | None,
    knowledge_base_name: str | None,
    knowledge_base_source_root: str | Path | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    if knowledge_base_source_root is not None:
        return settings.knowledge_base_paths_for_source_root(
            knowledge_base_source_root
        )
    if knowledge_base_id is not None:
        return settings.knowledge_base_paths_for_id(knowledge_base_id)
    return settings.knowledge_base_paths_for_name(
        knowledge_base_name or settings.knowledge_base_name
    )


def _contained_path(candidate: Path, storage_root: Path) -> Path:
    resolved_root = storage_root.resolve()
    lexical_candidate = Path(os.path.abspath(candidate))
    try:
        relative = lexical_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise UploadValidationError("Upload target is outside knowledge base storage") from exc
    cursor = resolved_root
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise UploadValidationError("Upload target has a symbolic-link parent")
    resolved_parent = lexical_candidate.parent.resolve()
    if resolved_parent == resolved_root or resolved_root in resolved_parent.parents:
        # Preserve the leaf identity. Resolving it would turn an attacker-
        # controlled candidate/backup symlink into its referenced victim path.
        return lexical_candidate
    raise UploadValidationError("Upload target is outside knowledge base storage")


def contained_path(candidate: Path, storage_root: Path) -> Path:
    """Normalize without following the leaf and reject storage-root escapes."""

    return _contained_path(candidate, storage_root)


def _decode_mountinfo_field(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _capability_preflight_error(
    reason: str,
    *,
    failure: str | None = None,
) -> StorageDurabilityCapabilityError:
    diagnostics: dict[str, object] = {
        "reason": reason,
        "protocol": "posix_parent_directory_fsync_v1",
        "mutation_started": False,
        "action": "Provision a validated Linux managed volume for DATA_ROOT and restart API and workers.",
    }
    if failure is not None:
        diagnostics["failure"] = failure
    return StorageDurabilityCapabilityError(
        "The storage durability capability could not be proven safely.",
        diagnostics=diagnostics,
    )


def _sanitized_os_failure(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno or 0, "OS_ERROR")


def _require_posix_dirfd_primitives() -> None:
    required_dir_fd = (os.open, os.mkdir, os.rename, os.rmdir, os.stat, os.unlink)
    required_follow_symlinks = (os.stat,)
    if (
        os.name != "posix"
        or not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or any(function not in os.supports_follow_symlinks for function in required_follow_symlinks)
    ):
        raise _capability_preflight_error("posix_dirfd_capability_unavailable")


@contextmanager
def _open_posix_directory_nofollow(directory: Path) -> Iterator[tuple[Path, int]]:
    """Pin every component of one absolute directory without following links."""

    _require_posix_dirfd_primitives()
    lexical = Path(os.path.abspath(os.fspath(directory)))
    if lexical.anchor != "/":
        raise _capability_preflight_error("path_identity_check_failed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open("/", flags)
            for part in lexical.parts[1:]:
                if part in {"", ".", ".."}:
                    raise _capability_preflight_error(
                        "path_identity_check_failed"
                    )
                next_descriptor = os.open(
                    part,
                    flags,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
        except StorageDurabilityCapabilityError:
            raise
        except OSError as exc:
            raise _capability_preflight_error(
                "path_identity_check_failed",
                failure=_sanitized_os_failure(exc),
            ) from None

        # Keep caller/body failures outside the setup translation block.  A
        # leaf O_NOFOLLOW failure, for example, belongs to source identity
        # verification and must not be relabelled as a root durability error
        # merely because it crosses this context manager's ``yield``.
        yield lexical, descriptor
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _mount_signature(details: dict[str, str]) -> str:
    fields = (
        details["mount_id"],
        details["parent_id"],
        details["major_minor"],
        details["mount_root"],
        details["mount_point"],
        details["mount_options"],
        details["filesystem_type"],
        details["mount_source"],
        details["super_options"],
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def _mount_info_for_path(path: Path) -> dict[str, str]:
    """Return a complete, bounded Linux mount identity or fail closed."""

    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise _capability_preflight_error("mount_identity_unavailable")
    best: tuple[int, dict[str, str]] | None = None
    total_bytes = 0
    try:
        with mountinfo.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle):
                total_bytes += len(line.encode("utf-8"))
                if line_number >= _MOUNTINFO_MAX_LINES or total_bytes > _MOUNTINFO_MAX_BYTES:
                    raise _capability_preflight_error("mount_identity_truncated")
                if not line.endswith("\n"):
                    raise _capability_preflight_error("mount_identity_truncated")
                left, separator, right = line.rstrip("\n").partition(" - ")
                if not separator:
                    continue
                left_fields = left.split()
                right_fields = right.split()
                if len(left_fields) < 6 or len(right_fields) < 3:
                    continue
                mount_point = Path(_decode_mountinfo_field(left_fields[4]))
                try:
                    path.relative_to(mount_point)
                except ValueError:
                    continue
                details = {
                    "mount_id": left_fields[0],
                    "parent_id": left_fields[1],
                    "major_minor": left_fields[2],
                    "mount_root": _decode_mountinfo_field(left_fields[3]),
                    "mount_point": str(mount_point),
                    "mount_options": left_fields[5],
                    "filesystem_type": right_fields[0].lower(),
                    "mount_source": _decode_mountinfo_field(right_fields[1]),
                    "super_options": right_fields[2],
                }
                match_length = len(mount_point.parts)
                if best is None or match_length > best[0]:
                    best = (match_length, details)
    except StorageDurabilityCapabilityError:
        raise
    except (OSError, UnicodeError) as exc:
        failure = _sanitized_os_failure(exc) if isinstance(exc, OSError) else "INVALID_UTF8"
        raise _capability_preflight_error("mount_identity_unavailable", failure=failure) from None
    if best is None:
        raise _capability_preflight_error("mount_identity_unavailable")
    details = best[1]
    details["mount_signature"] = _mount_signature(details)
    return details


def _normalized_mount_info(path: Path, root_stat: os.stat_result) -> dict[str, str]:
    try:
        raw = _mount_info_for_path(path)
    except StorageDurabilityCapabilityError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        failure = _sanitized_os_failure(exc) if isinstance(exc, OSError) else type(exc).__name__
        raise _capability_preflight_error("mount_identity_unavailable", failure=failure) from None
    filesystem_type = str(raw.get("filesystem_type") or "").strip().lower()
    mount_source = str(raw.get("mount_source") or "").strip()
    mount_point = str(raw.get("mount_point") or "").strip()
    if not filesystem_type or not mount_source or not mount_point:
        raise _capability_preflight_error("mount_identity_unavailable")
    expected_device = f"{os.major(root_stat.st_dev)}:{os.minor(root_stat.st_dev)}"
    major_minor = str(raw.get("major_minor") or expected_device)
    if major_minor != expected_device:
        raise _capability_preflight_error("mount_identity_mismatch")
    details = {
        "mount_id": str(raw.get("mount_id") or "test-mount"),
        "parent_id": str(raw.get("parent_id") or "test-parent"),
        "major_minor": major_minor,
        "mount_root": str(raw.get("mount_root") or "/"),
        "mount_point": mount_point,
        "mount_options": str(raw.get("mount_options") or "unknown"),
        "filesystem_type": filesystem_type,
        "mount_source": mount_source,
        "super_options": str(raw.get("super_options") or "unknown"),
    }
    supplied_signature = str(raw.get("mount_signature") or "").strip()
    details["mount_signature"] = supplied_signature or _mount_signature(details)
    return details


def _unsupported_shared_mount(mount: dict[str, str]) -> bool:
    filesystem_type = mount["filesystem_type"].casefold()
    source = mount["mount_source"].casefold().replace("\\", "/")
    return (
        filesystem_type in KNOWN_UNSUPPORTED_SHARED_FILESYSTEMS
        or filesystem_type.startswith(_UNSUPPORTED_FILESYSTEM_PREFIXES)
        or source.startswith("//")
        or any(marker in source for marker in _UNSUPPORTED_MOUNT_SOURCE_MARKERS)
    )


def _posix_fsync_directory(directory: Path) -> None:
    with _open_posix_directory_nofollow(directory) as (_lexical, descriptor):
        os.fsync(descriptor)


def _probe_capability_error(capability: StorageDurabilityCapability) -> StorageDurabilityCapabilityError:
    diagnostics = capability.as_dict()
    raw_root = str(diagnostics.pop("root", ""))
    diagnostics.pop("mount_source", None)
    diagnostics.pop("mount_point", None)
    diagnostics["root_identity_hash"] = hashlib.sha256(raw_root.encode("utf-8")).hexdigest()
    return StorageDurabilityCapabilityError(
        "The storage filesystem did not prove the required file-fsync, atomic-rename, "
        "dual-parent-directory-fsync, and durable-unlink contract.",
        diagnostics={
            **diagnostics,
            "reason": "filesystem_durability_probe_failed",
            "mutation_started": False,
            "action": "Move DATA_ROOT to a validated Linux managed volume and restart API and workers.",
        },
    )


def _run_posix_storage_durability_probe(
    root: Path,
    root_descriptor: int,
    root_stat: os.stat_result,
    mount: dict[str, str],
) -> StorageDurabilityCapability:
    started = monotonic()
    checked_at = datetime.now(timezone.utc).isoformat()
    filesystem_type = mount["filesystem_type"]
    if _unsupported_shared_mount(mount):
        return StorageDurabilityCapability(
            root=str(root),
            protocol="posix_parent_directory_fsync_v1",
            probe_version=STORAGE_DURABILITY_PROBE_VERSION,
            supported=False,
            checked_at=checked_at,
            duration_ms=round((monotonic() - started) * 1000, 3),
            device_id=int(root_stat.st_dev),
            inode=int(root_stat.st_ino),
            filesystem_type=filesystem_type,
            mount_source=mount["mount_source"],
            mount_point=mount["mount_point"],
            mount_signature=mount["mount_signature"],
            process_id=os.getpid(),
            cache_ttl_seconds=_CAPABILITY_CACHE_TTL_SECONDS,
            failure="unsupported_shared_filesystem_family",
        )

    probe_id = uuid4().hex
    source_dir_name = f".symbograph-durability-probe-{probe_id}-source"
    target_dir_name = f".symbograph-durability-probe-{probe_id}-target"
    source_descriptor: int | None = None
    target_descriptor: int | None = None
    failure: str | None = None
    try:
        os.mkdir(source_dir_name, dir_fd=root_descriptor)
        os.mkdir(target_dir_name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        source_descriptor = os.open(source_dir_name, flags, dir_fd=root_descriptor)
        target_descriptor = os.open(target_dir_name, flags, dir_fd=root_descriptor)
        file_descriptor = os.open(
            "candidate",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=source_descriptor,
        )
        try:
            remaining = memoryview(_PROBE_BYTES)
            while remaining:
                written = os.write(file_descriptor, remaining)
                if written <= 0:
                    raise OSError(errno.EIO, "probe write did not progress")
                remaining = remaining[written:]
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.rename(
            "candidate",
            "published",
            src_dir_fd=source_descriptor,
            dst_dir_fd=target_descriptor,
        )
        os.fsync(target_descriptor)
        os.fsync(source_descriptor)
        read_descriptor = os.open("published", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=target_descriptor)
        try:
            observed = os.read(read_descriptor, len(_PROBE_BYTES) + 1)
        finally:
            os.close(read_descriptor)
        if observed != _PROBE_BYTES:
            raise OSError("Durability probe bytes changed across atomic rename")
        os.unlink("published", dir_fd=target_descriptor)
        os.fsync(target_descriptor)
        os.close(source_descriptor)
        source_descriptor = None
        os.close(target_descriptor)
        target_descriptor = None
        os.rmdir(source_dir_name, dir_fd=root_descriptor)
        os.rmdir(target_dir_name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
    except OSError as exc:
        failure = f"probe_operation_failed:{_sanitized_os_failure(exc)}"
    finally:
        if source_descriptor is not None:
            with suppress(OSError):
                os.unlink("candidate", dir_fd=source_descriptor)
            with suppress(OSError):
                os.unlink("published", dir_fd=source_descriptor)
            with suppress(OSError):
                os.close(source_descriptor)
        if target_descriptor is not None:
            with suppress(OSError):
                os.unlink("candidate", dir_fd=target_descriptor)
            with suppress(OSError):
                os.unlink("published", dir_fd=target_descriptor)
            with suppress(OSError):
                os.close(target_descriptor)
        for directory_name in (source_dir_name, target_dir_name):
            with suppress(OSError):
                os.rmdir(directory_name, dir_fd=root_descriptor)
        with suppress(OSError):
            os.fsync(root_descriptor)

    return StorageDurabilityCapability(
        root=str(root),
        protocol="posix_parent_directory_fsync_v1",
        probe_version=STORAGE_DURABILITY_PROBE_VERSION,
        supported=failure is None,
        checked_at=checked_at,
        duration_ms=round((monotonic() - started) * 1000, 3),
        device_id=int(root_stat.st_dev),
        inode=int(root_stat.st_ino),
        filesystem_type=filesystem_type,
        mount_source=mount["mount_source"],
        mount_point=mount["mount_point"],
        mount_signature=mount["mount_signature"],
        process_id=os.getpid(),
        cache_ttl_seconds=_CAPABILITY_CACHE_TTL_SECONDS,
        failure=failure,
    )


def _capability_for_open_directory(
    root: Path,
    root_descriptor: int,
    *,
    force_probe: bool,
) -> StorageDurabilityCapability:
    global _CAPABILITY_CACHE_PROCESS_ID
    try:
        root_stat = os.fstat(root_descriptor)
    except OSError as exc:
        raise _capability_preflight_error(
            "path_identity_check_failed",
            failure=_sanitized_os_failure(exc),
        ) from None
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _capability_preflight_error("probe_root_not_trusted")
    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        if _configured_app_env_without_directory_mutation() == "production":
            namespace_durability_protocol()
        return adapter.capability(root)
    mount = _normalized_mount_info(root, root_stat)
    process_id = os.getpid()
    cache_key = (
        process_id,
        str(root),
        int(root_stat.st_dev),
        int(root_stat.st_ino),
        mount["mount_signature"],
        STORAGE_DURABILITY_PROBE_VERSION,
    )
    now = monotonic()
    with _CAPABILITY_CACHE_LOCK:
        if _CAPABILITY_CACHE_PROCESS_ID != process_id:
            _CAPABILITY_CACHE.clear()
            _CAPABILITY_CACHE_PROCESS_ID = process_id
        expired = [key for key, (expiry, _value) in _CAPABILITY_CACHE.items() if expiry <= now]
        for key in expired:
            _CAPABILITY_CACHE.pop(key, None)
        cached_entry = None if force_probe else _CAPABILITY_CACHE.get(cache_key)
        if cached_entry is not None:
            cached_result = replace(cached_entry[1], cache_hit=True)
            if not cached_result.supported:
                raise _probe_capability_error(cached_result)
            return cached_result

        # Durability is a mount contract. Once an ancestor root has completed
        # the destructive probe, a newly created descendant can inherit only
        # the remaining TTL of that proof after its own no-follow descriptor,
        # device/inode, and mount signature have been checked. This avoids a
        # probe-file storm for date/checksum directories without extending the
        # authorization or allowing sibling/out-of-root inheritance.
        if not force_probe:
            inherited_entry: tuple[float, StorageDurabilityCapability] | None = None
            for expiry, ancestor_capability in _CAPABILITY_CACHE.values():
                if (
                    expiry <= now
                    or not ancestor_capability.supported
                    or ancestor_capability.process_id != process_id
                    or ancestor_capability.device_id != int(root_stat.st_dev)
                    or ancestor_capability.mount_signature != mount["mount_signature"]
                    or ancestor_capability.probe_version != STORAGE_DURABILITY_PROBE_VERSION
                ):
                    continue
                try:
                    root.relative_to(Path(ancestor_capability.root))
                except ValueError:
                    continue
                inherited = replace(
                    ancestor_capability,
                    root=str(root),
                    device_id=int(root_stat.st_dev),
                    inode=int(root_stat.st_ino),
                    filesystem_type=mount["filesystem_type"],
                    mount_source=mount["mount_source"],
                    mount_point=mount["mount_point"],
                    mount_signature=mount["mount_signature"],
                    cache_hit=True,
                )
                inherited_entry = (expiry, inherited)
                break
            if inherited_entry is not None:
                if len(_CAPABILITY_CACHE) >= _CAPABILITY_CACHE_MAX_ENTRIES:
                    oldest_key = min(_CAPABILITY_CACHE, key=lambda key: _CAPABILITY_CACHE[key][0])
                    _CAPABILITY_CACHE.pop(oldest_key, None)
                _CAPABILITY_CACHE[cache_key] = inherited_entry
                return inherited_entry[1]

    capability = _run_posix_storage_durability_probe(root, root_descriptor, root_stat, mount)
    with _CAPABILITY_CACHE_LOCK:
        if len(_CAPABILITY_CACHE) >= _CAPABILITY_CACHE_MAX_ENTRIES:
            oldest_key = min(_CAPABILITY_CACHE, key=lambda key: _CAPABILITY_CACHE[key][0])
            _CAPABILITY_CACHE.pop(oldest_key, None)
        _CAPABILITY_CACHE[cache_key] = (now + _CAPABILITY_CACHE_TTL_SECONDS, capability)
    if not capability.supported:
        raise _probe_capability_error(capability)
    return capability


def require_storage_durability_capability(
    root: Path,
    *,
    force_probe: bool = False,
) -> StorageDurabilityCapability:
    """Prove and cache the durability contract for one actual mutation root."""

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        if _configured_app_env_without_directory_mutation() == "production":
            namespace_durability_protocol()  # raises the structured production error
        return adapter.capability(root)
    if _is_native_windows():
        _raise_native_windows_capability_error()

    try:
        with _open_posix_directory_nofollow(root) as (lexical_root, descriptor):
            return _capability_for_open_directory(
                lexical_root,
                descriptor,
                force_probe=force_probe,
            )
    except StorageDurabilityCapabilityError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        failure = _sanitized_os_failure(exc) if isinstance(exc, OSError) else type(exc).__name__
        raise _capability_preflight_error("storage_capability_inspection_failed", failure=failure) from None


def _lexically_contained_root(child: Path, parent: Path) -> Path:
    lexical_parent = Path(os.path.abspath(os.fspath(parent)))
    lexical_child = Path(os.path.abspath(os.fspath(child)))
    try:
        lexical_child.relative_to(lexical_parent)
    except ValueError:
        raise _capability_preflight_error("storage_root_outside_data_root") from None
    return lexical_child


@contextmanager
def _authorized_posix_directory_fd(
    directory: Path,
    *,
    force_probe: bool = False,
) -> Iterator[tuple[Path, int, StorageDurabilityCapability]]:
    with _open_posix_directory_nofollow(directory) as (lexical, descriptor):
        capability = _capability_for_open_directory(
            lexical,
            descriptor,
            force_probe=force_probe,
        )
        yield lexical, descriptor, capability


@contextmanager
def _readonly_import_posix_directory_fd(
    directory: Path,
) -> Iterator[tuple[Path, int, None]]:
    """Pin a source root and prove production imports cannot mutate it."""

    with _open_posix_directory_nofollow(directory) as (lexical, descriptor):
        if _TEST_DURABILITY_ADAPTER.get() is None:
            root_stat = os.fstat(descriptor)
            mount = _normalized_mount_info(lexical, root_stat)
            mount_options = {
                option.strip().casefold()
                for option in (
                    str(mount.get("mount_options") or "").split(",")
                    + str(mount.get("super_options") or "").split(",")
                )
                if option.strip()
            }
            if "ro" not in mount_options:
                raise SourceSnapshotError(
                    "Raw import root must be mounted read-only"
                )
        yield lexical, descriptor, None


def ensure_storage_durability_ready(
    *,
    settings=None,
    force_probe: bool = False,
) -> dict[str, object]:
    """Probe pre-provisioned DATA_ROOT, then durably bootstrap child roots."""

    configured = settings or get_settings()
    if _is_native_windows() and _TEST_DURABILITY_ADAPTER.get() is None:
        _raise_native_windows_capability_error()
    data_root = Path(configured.data_root)
    # DATA_ROOT is a deployment prerequisite. No child mkdir may run until
    # this exact mount has passed the real capability probe.
    capabilities = [
        require_storage_durability_capability(
            data_root,
            force_probe=force_probe,
        ).as_dict()
    ]
    roots = (
        Path(configured.knowledge_base_data_root_path),
        Path(configured.storage_root_path),
        Path(configured.ingestion_root_path),
    )
    seen = {os.path.abspath(os.fspath(data_root))}
    for root in roots:
        root = _lexically_contained_root(root, data_root)
        identity = os.path.abspath(os.fspath(root))
        if identity in seen:
            continue
        seen.add(identity)
        durable_ensure_directory(root)
        capabilities.append(require_storage_durability_capability(root).as_dict())
    return {
        "protocol": namespace_durability_protocol(),
        "probe_version": STORAGE_DURABILITY_PROBE_VERSION,
        "capabilities": capabilities,
    }


def ensure_knowledge_base_storage_durability_ready(
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
    create_missing: bool = False,
) -> dict[str, object]:
    """Prove the actual per-KB storage/ingestion roots before mutation."""

    settings = get_settings()
    capabilities = [require_storage_durability_capability(settings.data_root).as_dict()]
    paths = _knowledge_base_paths(
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_source_root=knowledge_base_source_root,
    )
    for key in ("storage_root", "ingestion_root"):
        root = _lexically_contained_root(paths[key], settings.data_root)
        if create_missing:
            durable_ensure_directory(root)
        capabilities.append(require_storage_durability_capability(root).as_dict())
    return {
        "protocol": namespace_durability_protocol(),
        "probe_version": STORAGE_DURABILITY_PROBE_VERSION,
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "capabilities": capabilities,
    }


def durable_sync_directory(directory: Path) -> None:
    """Make a directory-entry mutation durable, otherwise raise."""

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        if _configured_app_env_without_directory_mutation() == "production":
            namespace_durability_protocol()
        resolved = directory.resolve()
        adapter.sync_directory(resolved)
        return
    if _is_native_windows():
        _raise_native_windows_capability_error()
    try:
        with _authorized_posix_directory_fd(directory) as (_lexical, descriptor, _capability):
            os.fsync(descriptor)
    except StorageDurabilityCapabilityError:
        raise
    except OSError as exc:
        raise DirectoryDurabilityError(
            f"Directory durability barrier failed ({_sanitized_os_failure(exc)})"
        ) from exc


def durable_replace(source: Path, target: Path) -> None:
    """Atomically replace ``target`` and durably publish the rename."""

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        if _configured_app_env_without_directory_mutation() == "production":
            namespace_durability_protocol()
        source_parent = source.parent.resolve()
        target_parent = target.parent.resolve()
        require_storage_durability_capability(source_parent)
        if target_parent != source_parent:
            require_storage_durability_capability(target_parent)
        os.replace(source, target)
        durable_sync_directory(target_parent)
        if source_parent != target_parent:
            durable_sync_directory(source_parent)
        return
    if _is_native_windows():
        _raise_native_windows_capability_error()
    with _authorized_posix_directory_fd(source.parent) as (
        _source_parent,
        source_descriptor,
        _source_capability,
    ):
        with _authorized_posix_directory_fd(target.parent) as (
            _target_parent,
            target_descriptor,
            _target_capability,
        ):
            source_stat = os.stat(source.name, dir_fd=source_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(source_stat.st_mode):
                raise DirectoryDurabilityError("Durable replace source must be a regular file")
            os.rename(
                source.name,
                target.name,
                src_dir_fd=source_descriptor,
                dst_dir_fd=target_descriptor,
            )
            os.fsync(target_descriptor)
            if (
                source_descriptor != target_descriptor
                or source.parent != target.parent
            ):
                os.fsync(source_descriptor)


def _durable_publish_noreplace(
    source: Path,
    target: Path,
    *,
    expected_source_identity: VerifiedSourceIdentity,
    expected_source_mode: int,
) -> tuple[bool, VerifiedSourceIdentity | None]:
    """Durably publish one prepared regular file without clobbering a peer.

    Returns ``False`` when another writer already published ``target``. Linux
    production uses ``renameat2(RENAME_NOREPLACE)`` against pinned parents.
    The explicit test adapter uses an atomic hard-link/unlink sequence only so
    non-production CI can exercise the state machine without weakening the
    production primitive.
    """

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        require_storage_durability_capability(source.parent)
        require_storage_durability_capability(target.parent)
        source_stat = source.lstat()
        if (
            not stat.S_ISREG(source_stat.st_mode)
            # Windows may advance ctime when the last writable handle closes
            # after chmod.  The explicit pytest adapter therefore binds the
            # stable file identity fields here; production POSIX continues to
            # require the complete fstat/lstat identity, including ctime.
            or (
                int(source_stat.st_dev),
                int(source_stat.st_ino),
                int(source_stat.st_size),
                int(source_stat.st_mtime_ns),
                int(source_stat.st_nlink),
            )
            != (
                expected_source_identity.device_id,
                expected_source_identity.inode,
                expected_source_identity.size_bytes,
                expected_source_identity.mtime_ns,
                expected_source_identity.link_count,
            )
            or stat.S_IMODE(source_stat.st_mode) != expected_source_mode
        ):
            raise DirectoryDurabilityError(
                "No-clobber publish source identity changed before publication"
            )
        try:
            os.link(source, target, follow_symlinks=False)
        except FileExistsError:
            target_stat = target.lstat()
            parent_stat = target.parent.lstat()
            return False, VerifiedSourceIdentity(
                protocol_version="no_clobber_existing_target_v1",
                root_device_id=int(parent_stat.st_dev),
                root_inode=int(parent_stat.st_ino),
                device_id=int(target_stat.st_dev),
                inode=int(target_stat.st_ino),
                size_bytes=int(target_stat.st_size),
                mtime_ns=int(target_stat.st_mtime_ns),
                ctime_ns=int(target_stat.st_ctime_ns),
                link_count=int(target_stat.st_nlink),
            )
        durable_sync_directory(target.parent)
        linked_stat = source.lstat()
        parent_stat = source.parent.lstat()
        linked_identity = VerifiedSourceIdentity(
            protocol_version="test_adapter_linked_publish_source_v1",
            root_device_id=int(parent_stat.st_dev),
            root_inode=int(parent_stat.st_ino),
            device_id=int(linked_stat.st_dev),
            inode=int(linked_stat.st_ino),
            size_bytes=int(linked_stat.st_size),
            mtime_ns=int(linked_stat.st_mtime_ns),
            ctime_ns=int(linked_stat.st_ctime_ns),
            link_count=int(linked_stat.st_nlink),
        )
        if (
            linked_identity.device_id != expected_source_identity.device_id
            or linked_identity.inode != expected_source_identity.inode
            or linked_identity.size_bytes != expected_source_identity.size_bytes
            or linked_identity.mtime_ns != expected_source_identity.mtime_ns
            or linked_identity.link_count != 2
        ):
            raise DirectoryDurabilityError(
                "Test no-clobber publish source identity changed after link"
            )
        if _is_native_windows():
            # The readonly DOS attribute is inode-wide across hard links and
            # Windows refuses to unlink the prepared name while it is set.
            # This branch is reachable only through the explicit pytest
            # adapter; production never uses hard-link publication here.
            os.chmod(source, stat.S_IWRITE | stat.S_IREAD)
            try:
                durable_unlink(source)
            finally:
                if target.exists():
                    os.chmod(target, stat.S_IREAD)
        else:
            durable_unlink(source, expected_identity=linked_identity)
        return True, None

    if _is_native_windows():
        _raise_native_windows_capability_error()
    with _authorized_posix_directory_fd(source.parent) as (
        _source_parent,
        source_descriptor,
        _source_capability,
    ):
        with _authorized_posix_directory_fd(target.parent) as (
            _target_parent,
            target_descriptor,
            _target_capability,
        ):
            source_stat = os.stat(
                source.name,
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(source_stat.st_mode)
                or not _matches_verified_source_identity(source_stat, expected_source_identity)
                or stat.S_IMODE(source_stat.st_mode) != expected_source_mode
            ):
                raise DirectoryDurabilityError(
                    "No-clobber publish source identity changed before publication"
                )
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                raise DirectoryDurabilityError(
                    "Atomic no-clobber publication is unavailable"
                )
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                source_descriptor,
                os.fsencode(source.name),
                target_descriptor,
                os.fsencode(target.name),
                1,  # RENAME_NOREPLACE
            )
            if result != 0:
                error_number = ctypes.get_errno()
                if error_number == errno.EEXIST:
                    target_stat = os.stat(
                        target.name,
                        dir_fd=target_descriptor,
                        follow_symlinks=False,
                    )
                    parent_stat = os.fstat(target_descriptor)
                    return False, VerifiedSourceIdentity(
                        protocol_version="no_clobber_existing_target_v1",
                        root_device_id=int(parent_stat.st_dev),
                        root_inode=int(parent_stat.st_ino),
                        device_id=int(target_stat.st_dev),
                        inode=int(target_stat.st_ino),
                        size_bytes=int(target_stat.st_size),
                        mtime_ns=int(target_stat.st_mtime_ns),
                        ctime_ns=int(target_stat.st_ctime_ns),
                        link_count=int(target_stat.st_nlink),
                    )
                raise DirectoryDurabilityError(
                    "Atomic no-clobber publication failed "
                    f"({_sanitized_os_failure(OSError(error_number, os.strerror(error_number)))})"
                )
            target_stat = os.stat(
                target.name,
                dir_fd=target_descriptor,
                follow_symlinks=False,
            )
            published_identity = (
                int(target_stat.st_dev),
                int(target_stat.st_ino),
                int(target_stat.st_size),
                int(target_stat.st_mtime_ns),
                int(target_stat.st_nlink),
                stat.S_IMODE(target_stat.st_mode),
            )
            expected_published_identity = (
                expected_source_identity.device_id,
                expected_source_identity.inode,
                expected_source_identity.size_bytes,
                expected_source_identity.mtime_ns,
                expected_source_identity.link_count,
                expected_source_mode,
            )
            if (
                not stat.S_ISREG(target_stat.st_mode)
                or published_identity != expected_published_identity
            ):
                raise DirectoryDurabilityError(
                    "No-clobber publication identity changed during rename"
                )
            os.fsync(target_descriptor)
            if source.parent != target.parent:
                os.fsync(source_descriptor)
            return True, None


def _matches_verified_source_identity(
    target_stat: os.stat_result,
    expected_identity: VerifiedSourceIdentity,
) -> bool:
    return _verified_stat_identity(target_stat) == (
        expected_identity.device_id,
        expected_identity.inode,
        expected_identity.size_bytes,
        expected_identity.mtime_ns,
        expected_identity.ctime_ns,
        expected_identity.link_count,
    )


def durable_unlink(
    path: Path,
    *,
    missing_ok: bool = False,
    expected_identity: VerifiedSourceIdentity | None = None,
) -> None:
    """Delete a file and durably publish the directory-entry removal."""

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        if _configured_app_env_without_directory_mutation() == "production":
            namespace_durability_protocol()
        require_storage_durability_capability(path.parent)
        try:
            target_stat = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if not stat.S_ISREG(target_stat.st_mode):
            raise DirectoryDurabilityError("Durable unlink target must be a regular file")
        identity_matches = (
            expected_identity is None
            or _matches_verified_source_identity(target_stat, expected_identity)
        )
        if (
            expected_identity is not None
            and _is_native_windows()
            and adapter is not None
        ):
            # Closing the last writable Windows handle after chmod may advance
            # ctime without changing the prepared inode.  This exception is
            # confined to the explicit non-production durability adapter.
            identity_matches = (
                int(target_stat.st_dev),
                int(target_stat.st_ino),
                int(target_stat.st_size),
                int(target_stat.st_mtime_ns),
                int(target_stat.st_nlink),
            ) == (
                expected_identity.device_id,
                expected_identity.inode,
                expected_identity.size_bytes,
                expected_identity.mtime_ns,
                expected_identity.link_count,
            )
        if not identity_matches:
            raise DirectoryDurabilityError(
                "Durable unlink target identity changed after final-open verification"
            )
        if _is_native_windows() and not (
            stat.S_IMODE(target_stat.st_mode)
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            # Windows denies unlink for a readonly file.  The explicit test
            # adapter may clear that inode attribute only after the verified
            # identity comparison, then restores it if unlink itself fails.
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            try:
                path.unlink(missing_ok=missing_ok)
            finally:
                if path.exists():
                    os.chmod(path, stat.S_IREAD)
        else:
            path.unlink(missing_ok=missing_ok)
        durable_sync_directory(path.parent)
        return
    if _is_native_windows():
        _raise_native_windows_capability_error()
    with _authorized_posix_directory_fd(path.parent) as (
        _parent,
        parent_descriptor,
        _capability,
    ):
        try:
            target_stat = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if not stat.S_ISREG(target_stat.st_mode):
            raise DirectoryDurabilityError("Durable unlink target must be a regular file")
        if (
            expected_identity is not None
            and not _matches_verified_source_identity(target_stat, expected_identity)
        ):
            raise DirectoryDurabilityError(
                "Durable unlink target identity changed after final-open verification"
            )
        os.unlink(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def durable_rmdir(
    path: Path,
    *,
    missing_ok: bool = False,
    expected_device_id: int | None = None,
    expected_inode: int | None = None,
) -> None:
    """Remove one empty directory with an exact identity replay and parent fsync."""

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        require_storage_durability_capability(path.parent)
        try:
            target_stat = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if not stat.S_ISDIR(target_stat.st_mode):
            raise DirectoryDurabilityError("Durable rmdir target must be a directory")
        if (
            expected_device_id is not None
            and expected_inode is not None
            and (
                int(target_stat.st_dev) != int(expected_device_id)
                or int(target_stat.st_ino) != int(expected_inode)
            )
        ):
            raise DirectoryDurabilityError(
                "Durable rmdir target identity changed after inventory"
            )
        path.rmdir()
        durable_sync_directory(path.parent)
        return
    if _is_native_windows():
        _raise_native_windows_capability_error()
    with _authorized_posix_directory_fd(path.parent) as (
        _parent,
        parent_descriptor,
        _capability,
    ):
        try:
            target_stat = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if not stat.S_ISDIR(target_stat.st_mode):
            raise DirectoryDurabilityError("Durable rmdir target must be a directory")
        if (
            expected_device_id is not None
            and expected_inode is not None
            and (
                int(target_stat.st_dev) != int(expected_device_id)
                or int(target_stat.st_ino) != int(expected_inode)
            )
        ):
            raise DirectoryDurabilityError(
                "Durable rmdir target identity changed after inventory"
            )
        os.rmdir(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def _storage_tree_inventory_hash(payload: dict[str, Any]) -> str:
    canonical = {
        "protocol_version": payload["protocol_version"],
        "root_path": payload["root_path"],
        "exists": bool(payload["exists"]),
        "root_identity": payload.get("root_identity"),
        "entries": payload.get("entries") or [],
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def inventory_storage_tree(
    root: Path,
    *,
    authorized_parent: Path,
    max_entries: int = 100_000,
) -> dict[str, Any]:
    """Freeze a no-follow, content-verified inventory for one DB-owned tree."""

    if type(max_entries) is not int or max_entries < 1:
        raise ValueError("Storage tree inventory max_entries must be positive")
    lexical_parent = Path(os.path.abspath(authorized_parent))
    lexical_root = contained_path(Path(os.path.abspath(root)), lexical_parent)
    if lexical_root == lexical_parent:
        raise SourceSnapshotError("Storage tree root cannot equal its authorization root")
    if lexical_root.is_symlink():
        raise SourceSnapshotError("Storage tree root must not be a symbolic link")
    if not lexical_root.exists():
        payload = {
            "protocol_version": "verified_storage_tree_inventory_v1",
            "root_path": str(lexical_root),
            "exists": False,
            "root_identity": None,
            "entries": [],
        }
        payload["inventory_hash"] = _storage_tree_inventory_hash(payload)
        return payload

    root_stat = lexical_root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SourceSnapshotError("Storage tree root must be a directory")
    root_identity = {
        "device_id": int(root_stat.st_dev),
        "inode": int(root_stat.st_ino),
    }
    entries: list[dict[str, Any]] = []

    def add_entry(card: dict[str, Any]) -> None:
        entries.append(card)
        if len(entries) > max_entries:
            raise SourceSnapshotError(
                f"Storage tree inventory exceeds the hard limit of {max_entries} entries"
            )

    def walk(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                child_names = sorted(
                    (item.name for item in iterator),
                    key=lambda item: unicodedata.normalize("NFKC", item).casefold(),
                )
        except OSError as exc:
            raise SourceSnapshotError(
                f"Storage tree inventory failed: {_sanitized_os_failure(exc)}"
            ) from None
        for child_name in child_names:
            child_path = directory / child_name
            child_stat = child_path.lstat()
            relative = child_path.relative_to(lexical_root).as_posix()
            if stat.S_ISLNK(child_stat.st_mode):
                raise SourceSnapshotError(
                    f"Storage tree inventory rejects symbolic links: {relative}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                add_entry(
                    {
                        "relative_path": relative,
                        "kind": "directory",
                        "device_id": int(child_stat.st_dev),
                        "inode": int(child_stat.st_ino),
                        "mode": stat.S_IMODE(child_stat.st_mode),
                    }
                )
                walk(child_path)
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise SourceSnapshotError(
                    f"Storage tree inventory rejects non-regular entries: {relative}"
                )
            if int(child_stat.st_nlink) != 1:
                raise SourceSnapshotError(
                    f"Storage tree inventory rejects hard-link aliases: {relative}"
                )
            checksum, identity = verified_source_checksum(child_path, lexical_root)
            if (
                int(child_stat.st_dev),
                int(child_stat.st_ino),
                int(child_stat.st_size),
                int(child_stat.st_mtime_ns),
                int(child_stat.st_ctime_ns),
                int(child_stat.st_nlink),
            ) != (
                identity.device_id,
                identity.inode,
                identity.size_bytes,
                identity.mtime_ns,
                identity.ctime_ns,
                identity.link_count,
            ):
                raise SourceSnapshotError(
                    f"Storage tree entry changed during inventory: {relative}"
                )
            add_entry(
                {
                    "relative_path": relative,
                    "kind": "file",
                    "device_id": identity.device_id,
                    "inode": identity.inode,
                    "size_bytes": identity.size_bytes,
                    "mtime_ns": identity.mtime_ns,
                    "ctime_ns": identity.ctime_ns,
                    "link_count": identity.link_count,
                    "mode": stat.S_IMODE(child_stat.st_mode),
                    "checksum": checksum,
                    "final_open_protocol_version": identity.protocol_version,
                    "role": (
                        "immutable_snapshot"
                        if relative.startswith(f"ingestion/{SOURCE_SNAPSHOT_DIRECTORY}/")
                        else "logical_source_slot"
                        if relative.startswith(f"storage/{SOURCE_SLOT_DIRECTORY}/")
                        else "owned_storage_artifact"
                    ),
                }
            )

    walk(lexical_root)
    root_after = lexical_root.lstat()
    if (
        int(root_after.st_dev) != root_identity["device_id"]
        or int(root_after.st_ino) != root_identity["inode"]
    ):
        raise SourceSnapshotError("Storage tree root identity changed during inventory")
    entries.sort(key=lambda card: (str(card["relative_path"]), str(card["kind"])))
    payload = {
        "protocol_version": "verified_storage_tree_inventory_v1",
        "root_path": str(lexical_root),
        "exists": True,
        "root_identity": root_identity,
        "entries": entries,
    }
    payload["inventory_hash"] = _storage_tree_inventory_hash(payload)
    return payload


def _verified_identity_from_inventory(
    card: dict[str, Any],
    root: Path,
) -> VerifiedSourceIdentity:
    root_stat = root.lstat()
    return VerifiedSourceIdentity(
        protocol_version=str(card["final_open_protocol_version"]),
        root_device_id=int(root_stat.st_dev),
        root_inode=int(root_stat.st_ino),
        device_id=int(card["device_id"]),
        inode=int(card["inode"]),
        size_bytes=int(card["size_bytes"]),
        mtime_ns=int(card["mtime_ns"]),
        ctime_ns=int(card["ctime_ns"]),
        link_count=int(card["link_count"]),
    )


def durable_delete_storage_tree(
    inventory: dict[str, Any],
    *,
    authorized_parent: Path,
) -> dict[str, Any]:
    """Replay an exact tree inventory, reject drift, and durably remove it."""

    if (
        inventory.get("protocol_version") != "verified_storage_tree_inventory_v1"
        or inventory.get("inventory_hash") != _storage_tree_inventory_hash(inventory)
    ):
        raise SourceSnapshotError("Storage tree inventory hash or protocol is invalid")
    root = contained_path(Path(str(inventory["root_path"])), authorized_parent)
    if not inventory.get("exists"):
        if root.exists() or root.is_symlink():
            raise SourceSnapshotError("An unplanned storage tree appeared after inventory")
        return {"deleted_files": 0, "deleted_directories": 0, "already_absent": True}
    if root.is_symlink():
        raise SourceSnapshotError("Storage tree root became a symbolic link")
    if not root.exists():
        return {"deleted_files": 0, "deleted_directories": 0, "already_absent": True}
    root_stat = root.lstat()
    expected_root = inventory.get("root_identity") or {}
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or int(root_stat.st_dev) != int(expected_root.get("device_id", -1))
        or int(root_stat.st_ino) != int(expected_root.get("inode", -1))
    ):
        raise SourceSnapshotError("Storage tree root identity changed after intent commit")

    entries = list(inventory.get("entries") or [])
    directory_by_relative = {
        str(item["relative_path"]): item
        for item in entries
        if item.get("kind") == "directory"
    }

    def verify_ancestor_identities(relative_path: str) -> None:
        cursor = Path(relative_path).parent
        ancestry: list[Path] = []
        while cursor != Path("."):
            ancestry.append(cursor)
            cursor = cursor.parent
        for relative_directory in reversed(ancestry):
            key = relative_directory.as_posix()
            card = directory_by_relative.get(key)
            if card is None:
                raise SourceSnapshotError(
                    f"Storage tree inventory has no ancestor card: {key}"
                )
            candidate = contained_path(root / relative_directory, root)
            observed = candidate.lstat()
            if (
                not stat.S_ISDIR(observed.st_mode)
                or int(observed.st_dev) != int(card["device_id"])
                or int(observed.st_ino) != int(card["inode"])
            ):
                raise SourceSnapshotError(
                    f"Storage tree ancestor identity changed after intent commit: {key}"
                )

    deleted_files = 0
    for card in sorted(
        (item for item in entries if item.get("kind") == "file"),
        key=lambda item: str(item["relative_path"]),
    ):
        target = contained_path(root / str(card["relative_path"]), root)
        if not target.exists() and not target.is_symlink():
            continue
        verify_ancestor_identities(str(card["relative_path"]))
        before = target.lstat()
        expected = _verified_identity_from_inventory(card, root)
        if not _matches_verified_source_identity(before, expected):
            raise SourceSnapshotError(
                f"Storage tree file identity changed after intent commit: {card['relative_path']}"
            )
        current_mode = stat.S_IMODE(before.st_mode)
        if not current_mode & stat.S_IWUSR:
            if os.name == "posix":
                os.chmod(
                    target,
                    current_mode | stat.S_IWUSR,
                    follow_symlinks=False,
                )
            else:
                # Native Windows reaches this branch only through the explicit
                # pytest durability adapter.
                os.chmod(target, current_mode | stat.S_IWUSR)
        checksum, observed = verified_source_checksum(target, root)
        if (
            checksum != card.get("checksum")
            or observed.device_id != expected.device_id
            or observed.inode != expected.inode
            or observed.size_bytes != expected.size_bytes
            or observed.mtime_ns != expected.mtime_ns
            or observed.link_count != expected.link_count
        ):
            raise SourceSnapshotError(
                f"Storage tree file identity changed during controlled unprotect: {card['relative_path']}"
            )
        verify_ancestor_identities(str(card["relative_path"]))
        durable_unlink(target, expected_identity=observed)
        deleted_files += 1

    deleted_directories = 0
    directory_cards = sorted(
        (item for item in entries if item.get("kind") == "directory"),
        key=lambda item: (
            -len(Path(str(item["relative_path"])).parts),
            str(item["relative_path"]),
        ),
    )
    for card in directory_cards:
        target = contained_path(root / str(card["relative_path"]), root)
        if not target.exists() and not target.is_symlink():
            continue
        durable_rmdir(
            target,
            expected_device_id=int(card["device_id"]),
            expected_inode=int(card["inode"]),
        )
        deleted_directories += 1
    durable_rmdir(
        root,
        expected_device_id=int(expected_root["device_id"]),
        expected_inode=int(expected_root["inode"]),
    )
    deleted_directories += 1
    if root.exists() or root.is_symlink():
        raise SourceSnapshotError("Storage tree deletion postcondition failed")
    return {
        "deleted_files": deleted_files,
        "deleted_directories": deleted_directories,
        "already_absent": False,
    }


def durable_ensure_directory(directory: Path) -> Path:
    """Create a directory tree and flush every newly published parent entry."""

    if _is_native_windows() and _TEST_DURABILITY_ADAPTER.get() is None:
        # This branch intentionally precedes path normalization and mkdir.
        _raise_native_windows_capability_error()
    lexical = Path(os.path.abspath(os.fspath(directory)))
    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        missing: list[Path] = []
        cursor = lexical
        while not cursor.exists():
            if cursor.is_symlink():
                raise UploadValidationError(f"Storage directory is a symbolic link: {cursor}")
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        if not cursor.is_dir() or cursor.is_symlink():
            raise UploadValidationError(f"Storage directory parent is not a trusted directory: {cursor}")
        require_storage_durability_capability(cursor)
        for item in reversed(missing):
            try:
                item.mkdir()
            except FileExistsError:
                if not item.is_dir() or item.is_symlink():
                    raise UploadValidationError(f"Storage directory is not trusted: {item}")
            durable_sync_directory(item.parent)
        if lexical.parent != lexical:
            durable_sync_directory(lexical.parent)
        return lexical

    _require_posix_dirfd_primitives()
    if lexical.anchor != "/":
        raise _capability_preflight_error("path_identity_check_failed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    current = Path("/")
    try:
        for part in lexical.parts[1:]:
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                _capability_for_open_directory(current, descriptor, force_probe=False)
                try:
                    os.mkdir(part, mode=0o750, dir_fd=descriptor)
                    os.fsync(descriptor)
                except OSError as exc:
                    raise DirectoryDurabilityError(
                        f"Durable directory creation failed ({_sanitized_os_failure(exc)})"
                    ) from exc
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise _capability_preflight_error(
                    "path_identity_check_failed",
                    failure=_sanitized_os_failure(exc),
                ) from None
            os.close(descriptor)
            descriptor = next_descriptor
            current /= part
        _capability_for_open_directory(lexical, descriptor, force_probe=False)
        return lexical
    finally:
        with suppress(OSError):
            os.close(descriptor)


@contextmanager
def _open_new_file_for_durable_write(path: Path) -> Iterator[BinaryIO]:
    """Create one regular leaf relative to its pinned, authorized parent."""

    adapter = _TEST_DURABILITY_ADAPTER.get()
    if adapter is not None:
        require_storage_durability_capability(path.parent)
        try:
            with path.open("xb") as handle:
                yield handle
        finally:
            durable_sync_directory(path.parent)
        return
    if _is_native_windows():
        _raise_native_windows_capability_error()
    with _authorized_posix_directory_fd(path.parent) as (
        _parent,
        parent_descriptor,
        _capability,
    ):
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                yield handle
        finally:
            os.fsync(parent_descriptor)


def protect_immutable_file(path: Path) -> None:
    """Remove every ordinary write bit and verify the immutable-object guard."""

    if path.is_symlink() or not path.is_file():
        raise SourceSnapshotError(f"Immutable source snapshot is not a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise SourceSnapshotError(f"Immutable source snapshot must not have hard-link aliases: {path}")
    current_mode = stat.S_IMODE(path.stat().st_mode)
    readonly_mode = current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    if current_mode != readonly_mode:
        with path.open("r+b") as handle:
            os.chmod(path, readonly_mode)
            handle.flush()
            os.fsync(handle.fileno())
    if stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SourceSnapshotError(f"Source snapshot write protection could not be verified: {path}")


def verify_immutable_file_protection(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SourceSnapshotError(f"Immutable source snapshot is not a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise SourceSnapshotError(f"Immutable source snapshot has hard-link aliases: {path}")
    if stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SourceSnapshotError(f"Source snapshot is writable and cannot be used as immutable evidence: {path}")


def _make_file_owner_writable(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SourceSnapshotError(f"Snapshot maintenance target is not a regular file: {path}")
    os.chmod(path, stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def normalize_expected_upload_checksum(expected_checksum: str | None) -> str | None:
    normalized = (
        expected_checksum.strip().lower()
        if isinstance(expected_checksum, str)
        else None
    )
    if normalized is not None and (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise UploadValidationError(
            "Expected upload checksum must be one lowercase SHA-256 value"
        )
    return normalized


def _rewind_upload_stream(source: BinaryIO) -> None:
    try:
        source.seek(0)
    except (AttributeError, OSError, ValueError) as exc:
        raise UploadValidationError(
            "Upload stream must support deterministic rewind"
        ) from exc


def _hash_bounded_upload_stream(
    source: BinaryIO,
    *,
    max_bytes: int,
) -> tuple[str, int]:
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
    ):
        raise UploadValidationError("Upload byte limit is invalid")
    _rewind_upload_stream(source)
    digest = hashlib.sha256()
    total_bytes = 0
    while chunk := source.read(UPLOAD_READ_CHUNK_BYTES):
        if not isinstance(chunk, bytes):
            raise UploadValidationError("Upload stream returned non-byte content")
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise UploadTooLargeError(
                f"Upload exceeds the {max_bytes}-byte limit"
            )
        digest.update(chunk)
    return digest.hexdigest(), total_bytes


def _validate_text_characters(text: str) -> None:
    for character in text:
        if character in {"\t", "\n", "\r", "\f"}:
            continue
        if unicodedata.category(character) == "Cc":
            raise UploadValidationError(
                "Text upload contains binary control characters"
            )


def _validate_utf8_text_stream(
    source: BinaryIO,
    *,
    require_html_marker: bool,
) -> None:
    _rewind_upload_stream(source)
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="strict")
    prefix_parts: list[str] = []
    prefix_length = 0
    try:
        while chunk := source.read(UPLOAD_READ_CHUNK_BYTES):
            text = decoder.decode(chunk, final=False)
            _validate_text_characters(text)
            if require_html_marker and prefix_length < 64 * 1024:
                selected = text[: 64 * 1024 - prefix_length]
                prefix_parts.append(selected)
                prefix_length += len(selected)
        tail = decoder.decode(b"", final=True)
        _validate_text_characters(tail)
        if require_html_marker and prefix_length < 64 * 1024:
            selected = tail[: 64 * 1024 - prefix_length]
            prefix_parts.append(selected)
    except UnicodeDecodeError as exc:
        raise UploadValidationError(
            "Text upload must be strict UTF-8 with an optional BOM"
        ) from exc
    if require_html_marker:
        prefix = "".join(prefix_parts).lstrip().casefold()
        if not re.search(
            r"(?:<!doctype\s+html\b|<html(?:\s|>)|"
            r"<(?:head|body|title|p|h[1-6]|div|span|article|section|main|"
            r"table|ul|ol|li|a)(?:\s|>))",
            prefix,
        ):
            raise UploadValidationError(
                "HTML upload does not satisfy the versioned HTML content contract"
            )


def _validate_notebook_stream(source: BinaryIO, *, size_bytes: int) -> None:
    if size_bytes > UPLOAD_CONTENT_SIGNATURE_MAX_NOTEBOOK_BYTES:
        raise UploadValidationError(
            "Notebook upload exceeds the structured-content validation bound"
        )
    _rewind_upload_stream(source)
    try:
        raw = source.read(size_bytes + 1)
        if len(raw) != size_bytes:
            raise UploadValidationError(
                "Notebook upload size changed during validation"
            )
        text = raw.decode("utf-8-sig", errors="strict")
        _validate_text_characters(text)
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UploadValidationError(
            "Notebook upload must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise UploadValidationError("Notebook upload root must be an object")
    nbformat = payload.get("nbformat")
    nbformat_minor = payload.get("nbformat_minor")
    if (
        not isinstance(nbformat, int)
        or isinstance(nbformat, bool)
        or nbformat < 1
        or nbformat > 4
        or not isinstance(nbformat_minor, int)
        or isinstance(nbformat_minor, bool)
        or nbformat_minor < 0
        or not isinstance(payload.get("metadata"), dict)
        or not isinstance(payload.get("cells"), list)
    ):
        raise UploadValidationError(
            "Notebook upload does not satisfy the versioned notebook contract"
        )


def _validate_pdf_stream(source: BinaryIO, *, size_bytes: int) -> None:
    if size_bytes < 14:
        raise UploadValidationError("PDF upload is too short")
    _rewind_upload_stream(source)
    header = source.read(16)
    if not re.match(rb"%PDF-(?:1\.[0-7]|2\.0)(?:[\x00\t\n\f\r %])", header):
        raise UploadValidationError(
            "PDF upload does not have an accepted PDF header"
        )
    source.seek(max(0, size_bytes - 4096), os.SEEK_SET)
    tail = source.read(4096)
    marker = tail.rfind(b"%%EOF")
    if marker < 0 or tail[marker + 5 :].strip(b"\x00\t\n\f\r ") != b"":
        raise UploadValidationError(
            "PDF upload does not have a terminal PDF EOF marker"
        )


def _validate_png_stream(source: BinaryIO, *, size_bytes: int) -> None:
    if size_bytes < 45:
        raise UploadValidationError("PNG upload is too short")
    _rewind_upload_stream(source)
    header = source.read(33)
    if (
        header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[8:12] != b"\x00\x00\x00\r"
        or header[12:16] != b"IHDR"
        or int.from_bytes(header[16:20], "big") < 1
        or int.from_bytes(header[20:24], "big") < 1
    ):
        raise UploadValidationError(
            "PNG upload does not have a valid PNG/IHDR signature"
        )
    source.seek(size_bytes - 12, os.SEEK_SET)
    if source.read(12) != b"\x00\x00\x00\x00IEND\xaeB`\x82":
        raise UploadValidationError(
            "PNG upload does not have a terminal IEND chunk"
        )


def _validate_jpeg_stream(source: BinaryIO, *, size_bytes: int) -> None:
    if size_bytes < 4:
        raise UploadValidationError("JPEG upload is too short")
    _rewind_upload_stream(source)
    if source.read(3)[:3] != b"\xff\xd8\xff":
        raise UploadValidationError(
            "JPEG upload does not have a JPEG SOI marker"
        )
    source.seek(size_bytes - 2, os.SEEK_SET)
    if source.read(2) != b"\xff\xd9":
        raise UploadValidationError(
            "JPEG upload does not have a terminal JPEG EOI marker"
        )


def _validate_bmp_stream(source: BinaryIO, *, size_bytes: int) -> None:
    if size_bytes < 26:
        raise UploadValidationError("BMP upload is too short")
    _rewind_upload_stream(source)
    header = source.read(30)
    declared_size = int.from_bytes(header[2:6], "little")
    pixel_offset = int.from_bytes(header[10:14], "little")
    dib_size = int.from_bytes(header[14:18], "little")
    if (
        header[:2] != b"BM"
        or declared_size != size_bytes
        or pixel_offset < 14 + dib_size
        or pixel_offset > size_bytes
        or dib_size not in {12, 40, 52, 56, 64, 108, 124}
    ):
        raise UploadValidationError(
            "BMP upload does not have a valid bitmap header"
        )


def _validate_ole_presentation_stream(
    source: BinaryIO,
    *,
    size_bytes: int,
) -> None:
    if size_bytes < 512:
        raise UploadValidationError("Legacy PowerPoint upload is too short")
    _rewind_upload_stream(source)
    header = source.read(512)
    major_version = int.from_bytes(header[26:28], "little")
    sector_shift = int.from_bytes(header[30:32], "little")
    if (
        header[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        or header[28:30] != b"\xfe\xff"
        or major_version not in {3, 4}
        or sector_shift != (9 if major_version == 3 else 12)
        or int.from_bytes(header[32:34], "little") != 6
    ):
        raise UploadValidationError(
            "Legacy PowerPoint upload does not have a valid CFBF header"
        )


def _validate_zip_entry_name(name: str) -> str:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
    ):
        raise UploadValidationError("OOXML upload contains an unsafe ZIP entry")
    candidate = name[:-1] if name.endswith("/") else name
    if not candidate:
        raise UploadValidationError("OOXML upload contains an empty ZIP entry")
    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UploadValidationError("OOXML upload contains an unsafe ZIP path")
    normalized = PurePosixPath(candidate).as_posix()
    if normalized != candidate:
        raise UploadValidationError(
            "OOXML upload contains a non-canonical ZIP path"
        )
    return candidate


def _validate_ooxml_stream(
    source: BinaryIO,
    *,
    size_bytes: int,
    presentation: bool,
) -> None:
    if size_bytes < 4:
        raise UploadValidationError("OOXML upload is too short")
    _rewind_upload_stream(source)
    if source.read(4) != b"PK\x03\x04":
        raise UploadValidationError("OOXML upload does not have a ZIP signature")
    _rewind_upload_stream(source)
    required_part = (
        "ppt/presentation.xml" if presentation else "word/document.xml"
    )
    required_mime = (
        b"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
        if presentation
        else b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    )
    try:
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if (
                not entries
                or len(entries) > UPLOAD_CONTENT_SIGNATURE_MAX_ZIP_ENTRIES
            ):
                raise UploadValidationError(
                    "OOXML upload has an invalid ZIP entry count"
                )
            names: set[str] = set()
            casefold_names: set[str] = set()
            name_bytes = 0
            total_uncompressed = 0
            for entry in entries:
                name = _validate_zip_entry_name(entry.filename)
                folded = name.casefold()
                if name in names or folded in casefold_names:
                    raise UploadValidationError(
                        "OOXML upload has duplicate or case-ambiguous ZIP entries"
                    )
                names.add(name)
                casefold_names.add(folded)
                name_bytes += len(name.encode("utf-8"))
                total_uncompressed += int(entry.file_size)
                if (
                    name_bytes > UPLOAD_CONTENT_SIGNATURE_MAX_ZIP_NAME_BYTES
                    or total_uncompressed
                    > UPLOAD_CONTENT_SIGNATURE_MAX_UNCOMPRESSED_BYTES
                    or entry.compress_type
                    not in UPLOAD_CONTENT_SIGNATURE_ALLOWED_ZIP_METHODS
                    or bool(entry.flag_bits & 0x1)
                    or (
                        entry.file_size > 0
                        and (
                            entry.compress_size <= 0
                            or entry.file_size
                            > entry.compress_size
                            * UPLOAD_CONTENT_SIGNATURE_MAX_EXPANSION_RATIO
                        )
                    )
                ):
                    raise UploadValidationError(
                        "OOXML upload exceeds its closed ZIP safety contract"
                    )
            required = {
                "[Content_Types].xml",
                "_rels/.rels",
                required_part,
            }
            if not required.issubset(names):
                raise UploadValidationError(
                    "OOXML upload is missing required container parts"
                )
            content_type_info = archive.getinfo("[Content_Types].xml")
            if (
                content_type_info.file_size
                > UPLOAD_CONTENT_SIGNATURE_MAX_METADATA_BYTES
            ):
                raise UploadValidationError(
                    "OOXML content-type metadata exceeds its hard bound"
                )
            content_types = archive.read("[Content_Types].xml")
            if required_mime not in content_types:
                raise UploadValidationError(
                    "OOXML content type does not match the upload suffix"
                )
            corrupt_entry = archive.testzip()
            if corrupt_entry is not None:
                raise UploadValidationError(
                    "OOXML upload contains a corrupt ZIP entry"
                )
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise UploadValidationError(
            "OOXML upload is not a valid bounded ZIP container"
        ) from exc


def _validate_content_signature(
    source: BinaryIO,
    *,
    suffix: str,
    size_bytes: int,
) -> str:
    content_kind = _CONTENT_KIND_BY_SUFFIX.get(suffix)
    if content_kind is None or suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadValidationError(
            f"Unsupported upload file type: {suffix or '[none]'}"
        )
    if content_kind == "utf8_text":
        _validate_utf8_text_stream(source, require_html_marker=False)
    elif content_kind == "html":
        _validate_utf8_text_stream(source, require_html_marker=True)
    elif content_kind == "jupyter_notebook":
        _validate_notebook_stream(source, size_bytes=size_bytes)
    elif content_kind == "pdf":
        _validate_pdf_stream(source, size_bytes=size_bytes)
    elif content_kind == "png":
        _validate_png_stream(source, size_bytes=size_bytes)
    elif content_kind == "jpeg":
        _validate_jpeg_stream(source, size_bytes=size_bytes)
    elif content_kind == "bmp":
        _validate_bmp_stream(source, size_bytes=size_bytes)
    elif content_kind == "ole_presentation":
        _validate_ole_presentation_stream(source, size_bytes=size_bytes)
    elif content_kind == "ooxml_word":
        _validate_ooxml_stream(
            source,
            size_bytes=size_bytes,
            presentation=False,
        )
    elif content_kind == "ooxml_presentation":
        _validate_ooxml_stream(
            source,
            size_bytes=size_bytes,
            presentation=True,
        )
    else:  # pragma: no cover - the closed map and branch list must co-evolve.
        raise UploadValidationError(
            "Upload content signature protocol is not implemented"
        )
    return content_kind


def _validate_seekable_upload_content(
    source: BinaryIO,
    *,
    filename: str,
    max_bytes: int,
    expected_checksum: str | None = None,
) -> ValidatedUploadContent:
    safe_filename = normalize_upload_filename(filename)
    suffix = Path(safe_filename).suffix.casefold()
    normalized_expected_checksum = normalize_expected_upload_checksum(
        expected_checksum
    )
    try:
        checksum, size_bytes = _hash_bounded_upload_stream(
            source,
            max_bytes=max_bytes,
        )
        if (
            normalized_expected_checksum is not None
            and checksum != normalized_expected_checksum
        ):
            raise UploadChecksumMismatchError(
                "Uploaded bytes do not match the manifest-bound expected checksum"
            )
        content_kind = _validate_content_signature(
            source,
            suffix=suffix,
            size_bytes=size_bytes,
        )
        return ValidatedUploadContent(
            filename=safe_filename,
            suffix=suffix,
            checksum=checksum,
            size_bytes=size_bytes,
            content_kind=content_kind,
        )
    finally:
        _rewind_upload_stream(source)


def validate_upload_admission(
    upload: UploadFile,
    *,
    max_bytes: int,
    expected_checksum: str | None = None,
) -> ValidatedUploadContent:
    """Validate one request spool without creating application-owned state."""

    filename = validated_upload_filename(upload)
    return _validate_seekable_upload_content(
        upload.file,
        filename=filename,
        max_bytes=max_bytes,
        expected_checksum=expected_checksum,
    )


def validate_source_content_path(
    source_path: Path,
    authorized_root: Path,
    *,
    max_bytes: int | None = None,
    expected_checksum: str | None = None,
) -> ValidatedUploadContent:
    """Replay the upload content contract on a final-open contained file."""

    limit = int(max_bytes or get_settings().upload_max_bytes)
    safe_filename = normalize_upload_filename(Path(source_path).name)
    with open_verified_source_file(source_path, authorized_root) as (
        handle,
        identity,
    ):
        if int(identity.size_bytes) > limit:
            raise UploadTooLargeError(
                f"Upload exceeds the {limit}-byte limit"
            )
        validated = _validate_seekable_upload_content(
            handle,
            filename=safe_filename,
            max_bytes=limit,
            expected_checksum=expected_checksum,
        )
        if validated.size_bytes != int(identity.size_bytes):
            raise UploadValidationError(
                "Source file size changed during content validation"
            )
        return validated


def _write_upload_candidate_sync(
    source: BinaryIO,
    candidate_path: Path,
    *,
    max_bytes: int,
    expected_checksum: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[str, int]:
    _rewind_upload_stream(source)
    digest = hashlib.sha256()
    total_bytes = 0
    with _open_new_file_for_durable_write(candidate_path) as handle:
        while chunk := source.read(UPLOAD_READ_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise UploadTooLargeError(f"Upload exceeds the {max_bytes}-byte limit")
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    checksum = digest.hexdigest()
    if (
        expected_checksum is not None
        and checksum != expected_checksum
    ) or (
        expected_size_bytes is not None
        and total_bytes != expected_size_bytes
    ):
        raise UploadValidationError(
            "Upload bytes changed after pre-admission validation"
        )
    return checksum, total_bytes


async def write_upload_candidate(
    upload: UploadFile,
    candidate_path: Path,
    *,
    max_bytes: int,
    expected_checksum: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[str, int]:
    """Stream, hash and fsync untrusted bytes off the async event loop."""

    return await run_bounded_source_io(
        _write_upload_candidate_sync,
        upload.file,
        candidate_path,
        max_bytes=max_bytes,
        expected_checksum=expected_checksum,
        expected_size_bytes=expected_size_bytes,
    )


def _planned_storage_path_parts(
    filename: str,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    settings = get_settings()
    safe_filename = normalize_upload_filename(filename)
    logical_slot_key = normalize_upload_source_slot_key(safe_filename)
    slot_digest = hashlib.sha256(logical_slot_key.encode("utf-8")).hexdigest()
    storage_root = _knowledge_base_paths(
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_source_root=knowledge_base_source_root,
    )["storage_root"]
    require_storage_durability_capability(settings.data_root)
    storage_root = Path(os.path.abspath(storage_root))
    target_dir = _contained_path(
        storage_root / SOURCE_SLOT_DIRECTORY / slot_digest[:2],
        storage_root,
    )
    target = _contained_path(
        target_dir / f"{slot_digest}{Path(safe_filename).suffix.casefold()}",
        storage_root,
    )
    return storage_root, target_dir, target


def plan_storage_path(
    filename: str,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
) -> Path:
    """Freeze an upload target without creating its namespace directories."""

    _storage_root, _target_dir, target = _planned_storage_path_parts(
        filename,
        knowledge_base_name,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_source_root=knowledge_base_source_root,
    )
    return target


def build_storage_path(
    filename: str,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
) -> Path:
    storage_root, target_dir, target = _planned_storage_path_parts(
        filename,
        knowledge_base_name,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_source_root=knowledge_base_source_root,
    )
    materialized_root = durable_ensure_directory(storage_root).resolve()
    if materialized_root != storage_root.resolve():
        raise UploadValidationError("Knowledge-base storage root identity changed")
    require_storage_durability_capability(materialized_root)
    durable_ensure_directory(target_dir)
    return target


def validate_knowledge_base_source_path(
    source_path: Path,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
) -> Path:
    """Validate an executor source path without creating or following links."""

    if not source_path.is_absolute():
        raise UploadValidationError("Ingestion source path must be absolute")
    settings = get_settings()
    configured_root = _knowledge_base_paths(
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_source_root=knowledge_base_source_root,
    )["storage_root"]
    lexical_root = Path(os.path.abspath(configured_root))
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise UploadValidationError("Knowledge-base storage root is not a trusted directory")
    resolved_root = lexical_root.resolve(strict=True)
    if resolved_root != lexical_root:
        raise UploadValidationError("Knowledge-base storage root must not traverse symbolic links")

    lexical_source = Path(os.path.abspath(source_path))
    try:
        relative = lexical_source.relative_to(lexical_root)
    except ValueError as exc:
        raise UploadValidationError("Ingestion source path is outside knowledge-base storage") from exc
    cursor = lexical_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise UploadValidationError("Ingestion source path must not traverse symbolic links")
    if not lexical_source.is_file():
        raise UploadValidationError("Ingestion source path is not a regular file")
    resolved_source = lexical_source.resolve(strict=True)
    if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
        raise UploadValidationError("Resolved ingestion source path is outside knowledge-base storage")
    return resolved_source


def source_snapshot_path(
    filename: str,
    checksum: str,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
    create_parents: bool = False,
) -> Path:
    normalized_checksum = checksum.strip().lower()
    if len(normalized_checksum) != 64 or any(char not in "0123456789abcdef" for char in normalized_checksum):
        raise SourceSnapshotError("Source snapshot checksum must be a SHA-256 hex digest")
    settings = get_settings()
    paths = _knowledge_base_paths(
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_source_root=knowledge_base_source_root,
    )
    ingestion_root = paths["ingestion_root"]
    if create_parents:
        require_storage_durability_capability(settings.data_root)
        durable_ensure_directory(ingestion_root)
    ingestion_root = ingestion_root.resolve()
    safe_filename = normalize_upload_filename(filename)
    target_dir = _contained_path(
        ingestion_root / SOURCE_SNAPSHOT_DIRECTORY / normalized_checksum[:2] / normalized_checksum,
        ingestion_root,
    )
    if create_parents:
        durable_ensure_directory(target_dir)
    return _contained_path(target_dir / safe_filename, ingestion_root)


def _read_exact_bounded_source_bytes(
    source: BinaryIO,
    *,
    expected_size_bytes: int,
    max_bytes: int,
) -> bytes:
    if type(expected_size_bytes) is not int or expected_size_bytes < 0:
        raise UploadValidationError("Verified source size is invalid")
    if type(max_bytes) is not int or max_bytes < 1:
        raise UploadValidationError("Upload byte limit is invalid")
    if expected_size_bytes > max_bytes:
        raise UploadTooLargeError(f"Upload exceeds the {max_bytes}-byte limit")
    _rewind_upload_stream(source)
    content = source.read(expected_size_bytes + 1)
    if len(content) != expected_size_bytes:
        raise UploadValidationError(
            "Verified source size changed while bounded bytes were read"
        )
    if source.read(1):
        raise UploadTooLargeError(f"Upload exceeds the {max_bytes}-byte limit")
    _rewind_upload_stream(source)
    return content


def _freeze_snapshot_target(
    target: Path,
    *,
    ingestion_root: Path,
    validated: ValidatedUploadContent,
    expected_bytes: bytes,
    expected_identity: VerifiedSourceIdentity | None = None,
) -> FrozenSourceSnapshot:
    with open_verified_source_file(target, ingestion_root) as (handle, identity):
        if expected_identity is not None and not (
            identity.device_id == expected_identity.device_id
            and identity.inode == expected_identity.inode
            and identity.size_bytes == expected_identity.size_bytes
            and identity.mtime_ns == expected_identity.mtime_ns
            and identity.ctime_ns == expected_identity.ctime_ns
            and identity.link_count == expected_identity.link_count
        ):
            raise SourceSnapshotError(
                "Existing source snapshot identity changed during no-clobber reuse"
            )
        descriptor_stat = os.fstat(handle.fileno())
        if stat.S_IMODE(descriptor_stat.st_mode) & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise SourceSnapshotError(
                "Immutable source snapshot is writable"
            )
        observed_bytes = _read_exact_bounded_source_bytes(
            handle,
            expected_size_bytes=validated.size_bytes,
            max_bytes=validated.size_bytes or 1,
        )
        if observed_bytes != expected_bytes:
            raise SourceSnapshotError(
                "Immutable source snapshot bytes differ from the verified parser input"
            )
        checksum = hashlib.sha256(observed_bytes).hexdigest()
        if checksum != validated.checksum:
            raise SourceSnapshotError(
                "Immutable source snapshot checksum verification failed"
            )
        return FrozenSourceSnapshot(
            canonical_path=Path(os.path.abspath(target)),
            checksum=checksum,
            size_bytes=validated.size_bytes,
            content_kind=validated.content_kind,
            suffix=validated.suffix,
            identity=identity,
            content_bytes=observed_bytes,
        )


def freeze_existing_source_snapshot(
    snapshot_path: Path,
    *,
    authorized_root: Path,
    expected_checksum: str,
    max_bytes: int | None = None,
) -> FrozenSourceSnapshot:
    """Freeze an already committed snapshot through one verified descriptor."""

    limit = int(max_bytes or get_settings().upload_max_bytes)
    with open_verified_source_file(snapshot_path, authorized_root) as (
        handle,
        identity,
    ):
        descriptor_stat = os.fstat(handle.fileno())
        if stat.S_IMODE(descriptor_stat.st_mode) & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise SourceSnapshotError(
                "Existing source snapshot is writable"
            )
        content = _read_exact_bounded_source_bytes(
            handle,
            expected_size_bytes=identity.size_bytes,
            max_bytes=limit,
        )
        validated = _validate_seekable_upload_content(
            io.BytesIO(content),
            filename=snapshot_path.name,
            max_bytes=limit,
            expected_checksum=expected_checksum,
        )
        return FrozenSourceSnapshot(
            canonical_path=Path(os.path.abspath(snapshot_path)),
            checksum=validated.checksum,
            size_bytes=validated.size_bytes,
            content_kind=validated.content_kind,
            suffix=validated.suffix,
            identity=identity,
            content_bytes=content,
        )


def replay_frozen_source_snapshot(
    snapshot: FrozenSourceSnapshot,
    *,
    authorized_root: Path,
) -> None:
    """Replay one frozen snapshot card before any durable or external effect."""

    snapshot = validate_frozen_source_snapshot(snapshot)
    with open_verified_source_file(
        snapshot.canonical_path,
        authorized_root,
    ) as (handle, identity):
        if identity != snapshot.identity:
            raise SourceSnapshotError(
                "Frozen source snapshot identity changed before parser commit"
            )
        descriptor_stat = os.fstat(handle.fileno())
        if stat.S_IMODE(descriptor_stat.st_mode) & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise SourceSnapshotError(
                "Frozen source snapshot protection changed before parser commit"
            )
        observed = _read_exact_bounded_source_bytes(
            handle,
            expected_size_bytes=snapshot.size_bytes,
            max_bytes=snapshot.size_bytes or 1,
        )
        if observed != snapshot.content_bytes:
            raise SourceSnapshotError(
                "Frozen source snapshot bytes changed before parser commit"
            )
        if hashlib.sha256(observed).hexdigest() != snapshot.checksum:
            raise SourceSnapshotError(
                "Frozen source snapshot checksum changed before parser commit"
            )


def snapshot_source_file(
    source_path: Path,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
    expected_checksum: str | None = None,
    max_bytes: int | None = None,
) -> FrozenSourceSnapshot:
    """Copy one parse input into an immutable checksum-addressed attempt path.

    User-visible storage is a mutable upload slot. A ``DocumentVersion`` must
    never point at that slot because a later same-name upload may replace its
    bytes before parsing succeeds.
    """

    settings = get_settings()
    paths = _knowledge_base_paths(
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_source_root=knowledge_base_source_root,
    )
    storage_root = paths["storage_root"]
    ingestion_root = paths["ingestion_root"]
    require_storage_durability_capability(settings.data_root)
    durable_ensure_directory(ingestion_root)
    ingestion_root = ingestion_root.resolve()
    require_storage_durability_capability(ingestion_root)
    limit = int(max_bytes or settings.upload_max_bytes)
    safe_filename = normalize_upload_filename(source_path.name)
    snapshot_root = _contained_path(ingestion_root / SOURCE_SNAPSHOT_DIRECTORY, ingestion_root)
    pending_root = _contained_path(snapshot_root / ".pending", ingestion_root)
    durable_ensure_directory(pending_root)
    temporary_target = _contained_path(
        pending_root / f".{safe_filename}.{uuid4().hex}.snapshotting",
        ingestion_root,
    )
    prepared_identity: VerifiedSourceIdentity | None = None
    prepared_mode: int | None = None
    published = False
    try:
        with open_verified_source_file(source_path, storage_root) as (
            source,
            source_identity,
        ):
            source_bytes = _read_exact_bounded_source_bytes(
                source,
                expected_size_bytes=source_identity.size_bytes,
                max_bytes=limit,
            )
            validated = _validate_seekable_upload_content(
                io.BytesIO(source_bytes),
                filename=safe_filename,
                max_bytes=limit,
                expected_checksum=expected_checksum,
            )

        with _open_new_file_for_durable_write(temporary_target) as target_handle:
            for offset in range(0, len(source_bytes), UPLOAD_READ_CHUNK_BYTES):
                target_handle.write(
                    source_bytes[offset : offset + UPLOAD_READ_CHUNK_BYTES]
                )
            target_handle.flush()
            os.fsync(target_handle.fileno())
            if os.name == "posix":
                current_mode = stat.S_IMODE(os.fstat(target_handle.fileno()).st_mode)
                os.fchmod(
                    target_handle.fileno(),
                    current_mode
                    & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH),
                )
                os.fsync(target_handle.fileno())
            else:
                # Native Windows production is rejected before this branch;
                # the explicit test adapter still needs the readonly marker.
                os.chmod(temporary_target, stat.S_IREAD)
            prepared_stat = os.fstat(target_handle.fileno())
            if (
                not stat.S_ISREG(prepared_stat.st_mode)
                or int(prepared_stat.st_nlink) != 1
                or stat.S_IMODE(prepared_stat.st_mode)
                & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise SourceSnapshotError(
                    "Prepared source snapshot is not one protected regular file"
                )
            pending_root_stat = pending_root.lstat()
            prepared_identity = VerifiedSourceIdentity(
                protocol_version="prepared_frozen_source_snapshot_v1",
                root_device_id=int(pending_root_stat.st_dev),
                root_inode=int(pending_root_stat.st_ino),
                device_id=int(prepared_stat.st_dev),
                inode=int(prepared_stat.st_ino),
                size_bytes=int(prepared_stat.st_size),
                mtime_ns=int(prepared_stat.st_mtime_ns),
                ctime_ns=int(prepared_stat.st_ctime_ns),
                link_count=int(prepared_stat.st_nlink),
            )
            prepared_mode = stat.S_IMODE(prepared_stat.st_mode)

        target = source_snapshot_path(
            safe_filename,
            validated.checksum,
            knowledge_base_name,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_source_root=knowledge_base_source_root,
            create_parents=True,
        )
        assert prepared_identity is not None and prepared_mode is not None
        published, existing_identity = _durable_publish_noreplace(
            temporary_target,
            target,
            expected_source_identity=prepared_identity,
            expected_source_mode=prepared_mode,
        )
        frozen = _freeze_snapshot_target(
            target,
            ingestion_root=ingestion_root,
            validated=validated,
            expected_bytes=source_bytes,
            expected_identity=existing_identity,
        )
        if published and (
            frozen.identity.device_id != prepared_identity.device_id
            or frozen.identity.inode != prepared_identity.inode
            or frozen.identity.size_bytes != prepared_identity.size_bytes
            or frozen.identity.mtime_ns != prepared_identity.mtime_ns
            or frozen.identity.link_count != prepared_identity.link_count
        ):
            raise SourceSnapshotError(
                "Published source snapshot identity differs from its prepared descriptor"
            )
        if not published:
            durable_unlink(
                temporary_target,
                expected_identity=prepared_identity,
            )
        return frozen
    finally:
        if prepared_identity is not None:
            with suppress(OSError, SourceSnapshotError):
                durable_unlink(
                    temporary_target,
                    missing_ok=True,
                    expected_identity=prepared_identity,
                )


def copy_source_file(
    source_path: Path,
    knowledge_base_name: str | None = None,
    *,
    knowledge_base_id: str | None = None,
    knowledge_base_source_root: str | Path | None = None,
    expected_checksum: str | None = None,
) -> Path:
    snapshot = snapshot_source_file(
        source_path,
        knowledge_base_name,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_source_root=knowledge_base_source_root,
        expected_checksum=expected_checksum,
    )
    return snapshot.canonical_path
