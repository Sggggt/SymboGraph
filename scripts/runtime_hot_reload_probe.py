from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe runtime settings hot-reload publication and local singleton refresh.")
    parser.add_argument("--execute", action="store_true", help="Publish a probe runtime settings version. Omit for dry-run.")
    return parser.parse_args()


def main() -> None:
    from app.services.runtime_settings import current_runtime_settings_version, publish_runtime_settings_version, refresh_runtime_settings_if_needed

    args = parse_args()
    before = current_runtime_settings_version()
    message = None
    refresh = None
    if args.execute:
        message = publish_runtime_settings_version(changed_keys=["runtime_hot_reload_probe"], source="runtime_hot_reload_probe")
        refresh = refresh_runtime_settings_if_needed(force=True)
    after = current_runtime_settings_version()
    checks = {
        "redis_version_readable": after is not None if args.execute else True,
        "version_changed_when_executed": (before != after) if args.execute else True,
        "local_refresh_ran": bool(refresh and refresh.get("refreshed")) if args.execute else True,
    }
    payload = {
        "script": "runtime_hot_reload_probe",
        "execute": args.execute,
        "impact": "publish runtime settings probe version and clear local singletons" if args.execute else "no writes",
        "before_version": before,
        "after_version": after,
        "published_message": message,
        "refresh": refresh,
        "checks": checks,
        "pass": all(checks.values()),
    }
    report = write_report("runtime_hot_reload_probe", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], **payload}, ensure_ascii=False, default=str))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
