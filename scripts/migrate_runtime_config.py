from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _plan() -> dict:
    from app.core.config import RUNTIME_JSON_SETTINGS, Settings
    from app.core.runtime_config import (
        read_runtime_settings_values,
        runtime_settings_bytes,
    )
    from app.services.runtime_settings import (
        DEPRECATED_ENV_KEYS,
        ENV_PATH,
        SETTINGS_PATH,
        _runtime_env_bytes_with_updates,
    )

    if not ENV_PATH.exists():
        raise RuntimeError("repository-root .env is missing")
    configured = Settings()
    existing_values = read_runtime_settings_values(
        SETTINGS_PATH,
        allowed_keys=RUNTIME_JSON_SETTINGS,
        allow_missing=True,
    )
    values = {
        key: existing_values.get(key, getattr(configured, key))
        for key in sorted(RUNTIME_JSON_SETTINGS)
    }
    settings_bytes = runtime_settings_bytes(values, allowed_keys=RUNTIME_JSON_SETTINGS)
    env_before = ENV_PATH.read_bytes()
    removals = {
        key: None
        for key in sorted(
            {name.upper() for name in RUNTIME_JSON_SETTINGS} | set(DEPRECATED_ENV_KEYS)
        )
    }
    env_after = _runtime_env_bytes_with_updates(env_before, removals)
    return {
        "env_path": ENV_PATH,
        "settings_path": SETTINGS_PATH,
        "env_before": env_before,
        "env_after": env_after,
        "settings_bytes": settings_bytes,
        "settings_before": SETTINGS_PATH.read_bytes() if SETTINGS_PATH.exists() else None,
        "moved_keys": sorted(RUNTIME_JSON_SETTINGS),
        "removed_deprecated_keys": sorted(DEPRECATED_ENV_KEYS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute the one-time .env/settings.json authority split."
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = _plan()
    summary = {
        "execute": bool(args.execute),
        "env_file": plan["env_path"].name,
        "settings_file": plan["settings_path"].name,
        "moved_key_count": len(plan["moved_keys"]),
        "removed_deprecated_key_count": len(plan["removed_deprecated_keys"]),
        "moved_keys": plan["moved_keys"],
        "removed_deprecated_keys": plan["removed_deprecated_keys"],
        "secret_values_printed": False,
    }
    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    from app.core.runtime_config import atomic_replace_runtime_settings
    from app.services.runtime_settings import (
        _atomic_replace_runtime_env_bytes,
        runtime_env_file_lock,
    )

    settings_path = plan["settings_path"]
    with runtime_env_file_lock(path=plan["env_path"]):
        atomic_replace_runtime_settings(settings_path, plan["settings_bytes"])
        try:
            _atomic_replace_runtime_env_bytes(
                plan["env_after"],
                path=plan["env_path"],
            )
        except BaseException:
            if plan["settings_before"] is None:
                settings_path.unlink(missing_ok=True)
            else:
                atomic_replace_runtime_settings(
                    settings_path,
                    plan["settings_before"],
                )
            raise
    summary["executed"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
