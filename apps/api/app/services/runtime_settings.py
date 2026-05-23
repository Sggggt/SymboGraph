from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy import text

from app.core.config import WORKSPACE_ROOT, get_settings


ENV_PATH = WORKSPACE_ROOT / ".env"
ENV_EXAMPLE_PATH = WORKSPACE_ROOT / ".env.example"


def read_env_str(key: str, default: str = "") -> str:
    """直接从 .env 文件读取字符串值（热加载，绕过 os.environ 缓存）。"""
    return _env_entries(ENV_PATH).get(key.upper(), default)


def read_env_bool(key: str, default: bool = False) -> bool:
    """直接从 .env 文件读取布尔值（热加载，绕过 os.environ 缓存）。"""
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def read_env_int(key: str, default: int = 0) -> int:
    """直接从 .env 文件读取整数值（热加载，绕过 os.environ 缓存）。"""
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def model_settings_payload() -> dict:
    settings = get_settings()
    env_entries = _env_entries(ENV_PATH)
    chat_base_url = env_entries.get("CHAT_BASE_URL", settings.chat_base_url)
    model_bridge_enabled = env_entries.get("MODEL_BRIDGE_ENABLED", "false").lower() == "true"
    return {
        "provider": "openai_compatible",
        "chat_base_url": chat_base_url,
        "model_bridge_enabled": model_bridge_enabled,
        "chat_resolve_ip": settings.chat_resolve_ip,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "graph_extraction_strategy": settings.graph_extraction_strategy,
        "graph_extraction_soft_start_budget": settings.graph_extraction_soft_start_budget or 120,
        "graph_extraction_max_input_tokens_per_run": settings.graph_extraction_max_input_tokens_per_run,
        "graph_extraction_max_model_calls_per_run": settings.graph_extraction_max_model_calls_per_run,
        "graph_extraction_min_marginal_gain": settings.graph_extraction_min_marginal_gain,
        "graph_extraction_stall_rounds": settings.graph_extraction_stall_rounds,
        "graph_extraction_concurrency": settings.graph_extraction_concurrency,
        "graph_extraction_resume_batch_size": settings.graph_extraction_resume_batch_size,
        "reranker_enabled": read_env_bool("RERANKER_ENABLED", settings.reranker_enabled),
        "reranker_model": read_env_str("RERANKER_MODEL", settings.reranker_model),
        "reranker_max_length": read_env_int("RERANKER_MAX_LENGTH", settings.reranker_max_length),
        "reranker_device": "cpu",
        "reranker_url": "",
        "semantic_chunking_enabled": settings.semantic_chunking_enabled,
        "semantic_chunking_min_length": settings.semantic_chunking_min_length,
        "enable_auto_hpo": read_env_bool("ENABLE_AUTO_HPO", settings.enable_auto_hpo),
        "has_api_key": bool(settings.openai_api_key),
        "degraded_mode": not settings.openai_api_key or not settings.embedding_api_key or not settings.embedding_base_url,
        "embedding_base_url": settings.embedding_base_url,
        "embedding_resolve_ip": settings.embedding_resolve_ip,
        "has_embedding_api_key": bool(settings.embedding_api_key),
    }


def _serialize_env_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if not text:
        return ""
    if any(char.isspace() for char in text) or any(char in text for char in ['"', "#", "="]):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _update_env_file(updates: dict[str, str | int | float | bool | None]) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    remaining = {key.upper(): value for key, value in updates.items()}
    next_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip().lstrip("\ufeff").upper()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if key not in remaining:
            value = line.split("=", 1)[1]
            next_lines.append(f"{key}={value}")
            continue
        value = remaining.pop(key)
        if value is not None:
            next_lines.append(f"{key}={_serialize_env_value(value)}")

    if remaining:
        if next_lines and next_lines[-1].strip():
            next_lines.append("")
        for key, value in remaining.items():
            if value is not None:
                next_lines.append(f"{key}={_serialize_env_value(value)}")

    ENV_PATH.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")


def _apply_runtime_env(updates: dict[str, str | int | float | bool | None]) -> None:
    for key, value in updates.items():
        env_key = key.upper()
        if value is None:
            os.environ.pop(env_key, None)
        elif isinstance(value, bool):
            os.environ[env_key] = "true" if value else "false"
        else:
            os.environ[env_key] = str(value)


def _env_keys(path: Path) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    bom_keys: list[str] = []
    if not path.exists():
        return keys, bom_keys
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue
        raw_key = raw_line.split("=", 1)[0].strip()
        clean_key = raw_key.lstrip("\ufeff").upper()
        if raw_key.startswith("\ufeff"):
            bom_keys.append(clean_key)
        keys.add(clean_key)
    return keys, bom_keys


def _env_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue
        raw_key, value = raw_line.split("=", 1)
        key = raw_key.strip().lstrip("\ufeff").upper()
        entries.setdefault(key, value)
    return entries


def normalize_env_file() -> None:
    """清理 .env 文件中的 BOM 前缀，不再与 .env.example 合并。

    之前的实现会重写 .env 文件并依赖 .env.example，这会丢失用户注释、
    空行和原有顺序，且 .env.example 是示例文件不应被运行时依赖。
    现在只移除 BOM 前缀，保留文件原貌。
    """
    if not ENV_PATH.exists():
        return
    content = ENV_PATH.read_text(encoding="utf-8")
    # 移除行首的 BOM 前缀
    cleaned_lines = []
    changed = False
    for line in content.splitlines():
        if line.startswith("\ufeff"):
            cleaned_lines.append(line.lstrip("\ufeff"))
            changed = True
        else:
            cleaned_lines.append(line)
    if changed:
        ENV_PATH.write_text("\n".join(cleaned_lines).rstrip() + "\n", encoding="utf-8")


def env_sync_status() -> dict:
    """检查 .env 与 .env.example 的参数列表是否一致，并检测 BOM 前缀。"""
    actual_keys, bom_keys = _env_keys(ENV_PATH)
    example_keys, _ = _env_keys(ENV_EXAMPLE_PATH)
    missing_keys = sorted(example_keys - actual_keys)
    extra_keys = sorted(actual_keys - example_keys)
    return {
        "synced": not bom_keys and not missing_keys and not extra_keys,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "bom_keys": sorted(set(bom_keys)),
    }


def _runtime_issue(code: str, title: str, message: str, fix_commands: list[str] | None = None) -> dict:
    return {"code": code, "title": title, "message": message, "fix_commands": fix_commands or []}


def _check_postgres() -> bool:
    with suppress(Exception):
        import app.db as db

        with db.SessionLocal() as session:
            session.execute(text("SELECT 1"))
            return True
    return False


def _check_qdrant() -> bool:
    settings = get_settings()
    with suppress(Exception):
        response = httpx.get(f"{settings.qdrant_url.rstrip('/')}/collections", timeout=2.0)
        return response.status_code < 500
    return False


def _check_redis() -> bool:
    settings = get_settings()
    parsed = urlparse(settings.redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    with suppress(Exception):
        import socket

        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(b"PING\r\n")
            return sock.recv(16).startswith(b"+PONG")
    return False


def _check_model_bridge() -> bool | None:
    settings = get_settings()
    parsed = urlparse(settings.chat_base_url)
    if (parsed.hostname or "").lower() != "host.docker.internal":
        return None
    with suppress(Exception):
        response = httpx.get(f"{settings.chat_base_url.rstrip('/')}/health", timeout=3.0)
        return response.status_code == 200
    return False


def _reranker_runtime_status() -> dict:
    """检测 CrossEncoder 重排序器的运行时状态（使用单例缓存，不重复加载模型）。"""
    from app.services.reranker import get_reranker_status
    return get_reranker_status()


def runtime_check_payload() -> dict:
    env_sync = env_sync_status()
    blocking_issues: list[dict] = []
    warnings: list[dict] = []
    if env_sync["bom_keys"]:
        blocking_issues.append(
            _runtime_issue(
                "env_bom_keys",
                ".env contains BOM-prefixed keys",
                "One or more .env keys contain a UTF-8 BOM prefix and must be normalized before saving settings.",
                ["Open the Settings page and save once, or rewrite the affected key without the BOM prefix."],
            )
        )
    if env_sync["missing_keys"] or env_sync["extra_keys"]:
        blocking_issues.append(
            _runtime_issue(
                "env_key_mismatch",
                ".env and .env.example keys differ",
                "The runtime .env parameter list must match .env.example. Values are not compared or exposed.",
                ["Compare .env and .env.example and add/remove only key names as needed."],
            )
        )
    infrastructure = {
        "postgres": _check_postgres(),
        "qdrant": _check_qdrant(),
        "redis": _check_redis(),
        "model_bridge": _check_model_bridge(),
    }
    for key, ok in infrastructure.items():
        if ok is None:
            continue
        if not ok:
            warnings.append(
                _runtime_issue(
                    f"{key}_unreachable",
                    f"{key} is not reachable",
                    f"The {key} infrastructure check failed from the API process.",
                    [".\\start-app.ps1"],
                )
            )
    return {
        "env_sync": env_sync,
        "reranker": _reranker_runtime_status(),
        "infrastructure": infrastructure,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }


def update_model_settings(payload: dict) -> dict:
    normalize_env_file()
    updates: dict[str, str | int | float | bool | None] = {}
    key_map = {
        "chat_base_url": "chat_base_url",
        "embedding_base_url": "embedding_base_url",
        "model_bridge_enabled": "model_bridge_enabled",
        "chat_resolve_ip": "chat_resolve_ip",
        "embedding_resolve_ip": "embedding_resolve_ip",
        "embedding_model": "embedding_model",
        "chat_model": "chat_model",
        "embedding_dimensions": "embedding_dimensions",
        "graph_extraction_strategy": "graph_extraction_strategy",
        "graph_extraction_soft_start_budget": "graph_extraction_soft_start_budget",
        "graph_extraction_max_input_tokens_per_run": "graph_extraction_max_input_tokens_per_run",
        "graph_extraction_max_model_calls_per_run": "graph_extraction_max_model_calls_per_run",
        "graph_extraction_min_marginal_gain": "graph_extraction_min_marginal_gain",
        "graph_extraction_stall_rounds": "graph_extraction_stall_rounds",
        "graph_extraction_concurrency": "graph_extraction_concurrency",
        "graph_extraction_resume_batch_size": "graph_extraction_resume_batch_size",
        "reranker_enabled": "reranker_enabled",
        "reranker_model": "reranker_model",
        "reranker_max_length": "reranker_max_length",
        "semantic_chunking_enabled": "semantic_chunking_enabled",
        "semantic_chunking_min_length": "semantic_chunking_min_length",
        "enable_auto_hpo": "enable_auto_hpo",
    }
    nullable_setting_keys = {
        "graph_extraction_max_input_tokens_per_run",
        "graph_extraction_max_model_calls_per_run",
    }
    for key, env_key in key_map.items():
        if key not in payload:
            continue
        value = payload.get(key)
        if key in {"chat_resolve_ip", "embedding_resolve_ip"} and (value is None or (isinstance(value, str) and not value.strip())):
            updates[env_key] = ""
        elif key in nullable_setting_keys and value is None:
            updates[env_key] = None
        elif value is not None:
            updates[env_key] = value.strip() if isinstance(value, str) else value

    api_key = payload.get("api_key")
    if payload.get("clear_api_key"):
        updates["openai_api_key"] = ""
    elif isinstance(api_key, str) and api_key.strip():
        updates["openai_api_key"] = api_key.strip()

    embedding_api_key = payload.get("embedding_api_key")
    if payload.get("clear_embedding_api_key"):
        updates["embedding_api_key"] = ""
    elif isinstance(embedding_api_key, str) and embedding_api_key.strip():
        updates["embedding_api_key"] = embedding_api_key.strip()

    if updates:
        _update_env_file(updates)
        _apply_runtime_env(updates)
        get_settings.cache_clear()
        get_settings()
        # 如果 reranker 模型配置发生变化，清除单例缓存以强制重新加载
        if "reranker_model" in updates or "reranker_max_length" in updates:
            from app.services.reranker import clear_reranker_cache
            clear_reranker_cache()
    return model_settings_payload()
