from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


RUNTIME_SETTINGS_FILE_PROTOCOL = "symbograph_runtime_settings_v1"


class RuntimeSettingsFileError(ValueError):
    pass


def runtime_settings_path(*, workspace_root: Path, env_path: Path | None = None) -> Path:
    configured = str(os.environ.get("RUNTIME_SETTINGS_FILE") or "").strip()
    if configured:
        return Path(configured)
    if env_path is not None:
        return Path(env_path).with_name("settings.json")
    return workspace_root / "settings.json"


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeSettingsFileError("runtime settings JSON contains a duplicate key")
        result[key] = value
    return result


def _finite_json_scalar(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    return type(value) is float and math.isfinite(value)


def parse_runtime_settings_bytes(
    content: bytes | None,
    *,
    allowed_keys: Iterable[str],
    allow_missing: bool = True,
) -> dict[str, Any]:
    if content is None:
        if allow_missing:
            return {}
        raise RuntimeSettingsFileError("repository-root settings.json is missing")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeSettingsFileError("runtime settings JSON must be UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, RuntimeSettingsFileError) as exc:
        raise RuntimeSettingsFileError("runtime settings JSON is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {"protocol_version", "settings"}:
        raise RuntimeSettingsFileError("runtime settings JSON must use the closed root shape")
    if payload["protocol_version"] != RUNTIME_SETTINGS_FILE_PROTOCOL:
        raise RuntimeSettingsFileError("runtime settings JSON protocol mismatch")
    values = payload["settings"]
    if not isinstance(values, dict):
        raise RuntimeSettingsFileError("runtime settings JSON settings must be an object")
    allowed = frozenset(allowed_keys)
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise RuntimeSettingsFileError(
            "runtime settings JSON contains unsupported keys: " + ", ".join(unexpected)
        )
    if any(not _finite_json_scalar(value) for value in values.values()):
        raise RuntimeSettingsFileError("runtime settings JSON values must be finite JSON scalars")
    return dict(values)


def read_runtime_settings_values(
    path: Path,
    *,
    allowed_keys: Iterable[str],
    allow_missing: bool = True,
) -> dict[str, Any]:
    content = path.read_bytes() if path.exists() else None
    return parse_runtime_settings_bytes(
        content,
        allowed_keys=allowed_keys,
        allow_missing=allow_missing,
    )


def runtime_settings_bytes(values: Mapping[str, Any], *, allowed_keys: Iterable[str]) -> bytes:
    allowed = frozenset(allowed_keys)
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise RuntimeSettingsFileError(
            "runtime settings update contains unsupported keys: " + ", ".join(unexpected)
        )
    if any(not _finite_json_scalar(value) for value in values.values()):
        raise RuntimeSettingsFileError("runtime settings update values must be finite JSON scalars")
    payload = {
        "protocol_version": RUNTIME_SETTINGS_FILE_PROTOCOL,
        "settings": {key: values[key] for key in sorted(values)},
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def runtime_settings_file_identity(path: Path) -> dict[str, Any]:
    content = path.read_bytes() if path.exists() else None
    return {
        "protocol_version": "runtime_settings_file_identity_v1",
        "exists": content is not None,
        "size": len(content) if content is not None else 0,
        "sha256": hashlib.sha256(content or b"").hexdigest(),
    }


def atomic_replace_runtime_settings(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        with path.open("r+b") as handle:
            if handle.read() != content:
                raise OSError("runtime settings publication replay mismatch")
            os.fsync(handle.fileno())
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
