from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ResourceExhaustedError(RuntimeError):
    def __init__(self, message: str, *, snapshot: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.snapshot = snapshot or {}


@dataclass(frozen=True)
class MemoryPressureSnapshot:
    source: str
    current_bytes: int | None
    limit_bytes: int | None
    ratio: float | None
    process_rss_bytes: int | None
    level: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "current_bytes": self.current_bytes,
            "limit_bytes": self.limit_bytes,
            "ratio": round(self.ratio, 6) if self.ratio is not None else None,
            "process_rss_bytes": self.process_rss_bytes,
            "level": self.level,
        }


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int(path: str) -> int | None:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) * 1024
    except OSError:
        pass
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss * 1024
    except Exception:
        return None


def _cgroup_memory() -> tuple[int | None, int | None, str | None]:
    current = _read_int("/sys/fs/cgroup/memory.current")
    limit = _read_int("/sys/fs/cgroup/memory.max")
    if current is not None:
        return current, limit, "cgroup_v2"
    current = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if current is not None:
        if limit is not None and limit > 10**15:
            limit = None
        return current, limit, "cgroup_v1"
    return None, None, None


def _system_memory() -> tuple[int | None, int | None, str | None]:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return int(mem.total - mem.available), int(mem.total), "psutil"
    except Exception:
        return None, None, None


def memory_pressure_snapshot() -> MemoryPressureSnapshot:
    current, limit, source = _cgroup_memory()
    if current is None or limit is None:
        current, limit, source = _system_memory()
    ratio = (float(current) / float(limit)) if current is not None and limit else None
    soft = _env_float("INGESTION_MEMORY_SOFT_LIMIT_RATIO", 0.78)
    hard = _env_float("INGESTION_MEMORY_HARD_LIMIT_RATIO", 0.88)
    critical = _env_float("INGESTION_MEMORY_CRITICAL_LIMIT_RATIO", 0.94)
    if ratio is None:
        level = "unknown"
    elif ratio >= critical:
        level = "critical"
    elif ratio >= hard:
        level = "hard"
    elif ratio >= soft:
        level = "soft"
    else:
        level = "normal"
    return MemoryPressureSnapshot(
        source=source or "unavailable",
        current_bytes=current,
        limit_bytes=limit,
        ratio=ratio,
        process_rss_bytes=_process_rss_bytes(),
        level=level,
    )


def collect_memory() -> None:
    gc.collect()


def effective_embedding_batch_size(configured_batch_size: int) -> int:
    batch_size = max(1, int(configured_batch_size or 1))
    snapshot = memory_pressure_snapshot()
    if snapshot.level == "critical":
        return 1
    if snapshot.level == "hard":
        return 1
    if snapshot.level == "soft":
        return max(1, batch_size // 2)
    return batch_size


def enforce_memory_budget(stage: str, *, batch_id: str | None = None, sleep_seconds: float = 0.2) -> MemoryPressureSnapshot:
    snapshot = memory_pressure_snapshot()
    if snapshot.level in {"soft", "hard", "critical"}:
        collect_memory()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        snapshot = memory_pressure_snapshot()
    if snapshot.level == "critical":
        payload = snapshot.as_dict()
        if batch_id:
            payload["batch_id"] = batch_id
        payload["stage"] = stage
        raise ResourceExhaustedError(
            f"Memory pressure is critical during {stage}; aborting before the worker is OOM-killed",
            snapshot=payload,
        )
    return snapshot
