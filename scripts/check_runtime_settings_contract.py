from __future__ import annotations

import json
import re
from typing import Any

from _context_graph_maintenance import REPO_ROOT, write_report


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, evidence: dict[str, Any] | None = None) -> None:
    issues.append({"severity": severity, "code": code, "message": message, "evidence": evidence or {}})


def ts_interface_keys(interface_name: str) -> set[str]:
    text = (REPO_ROOT / "packages/shared/src/index.ts").read_text(encoding="utf-8")
    match = re.search(rf"export interface {re.escape(interface_name)} \{{(?P<body>.*?)\n\}}", text, flags=re.S)
    if not match:
        return set()
    return set(re.findall(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\??:", match.group("body"), flags=re.M))


def env_example_keys() -> set[str]:
    path = REPO_ROOT / ".env.example"
    if not path.exists():
        return set()
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue
        keys.add(raw_line.split("=", 1)[0].strip().upper())
    return keys


def main() -> None:
    from app.core.config import HOT_RELOAD_SETTINGS, Settings
    from app.schemas import ModelSettingsUpdate
    from app.services.runtime_settings import model_settings_payload

    issues: list[dict[str, Any]] = []
    settings_fields = set(Settings.model_fields)
    hot_reload = set(HOT_RELOAD_SETTINGS)
    runtime_payload = set(model_settings_payload())
    update_schema = set(ModelSettingsUpdate.model_fields)
    shared_response = ts_interface_keys("ModelSettingsResponse")
    shared_update = ts_interface_keys("ModelSettingsUpdate")
    env_keys = env_example_keys()
    runtime_payload_exemptions = {"openai_api_key", "embedding_api_key", "model_bridge_port"}
    update_schema_exemptions = {"openai_api_key", "embedding_api_key", "model_bridge_port", "enable_model_fallback"}

    for key in sorted(hot_reload):
        if key not in settings_fields:
            add_issue(issues, "blocker", "hot_reload_unknown_setting", f"{key} is hot_reloadable but not a Settings field.")
        if key not in runtime_payload and key not in runtime_payload_exemptions:
            add_issue(issues, "blocker", "hot_reload_missing_runtime_payload", f"{key} is hot_reloadable but missing from runtime payload.")
        if key not in update_schema and key not in update_schema_exemptions:
            add_issue(issues, "warning", "hot_reload_missing_update_schema", f"{key} is hot_reloadable but missing from API update schema.")
        if key not in shared_update and key not in update_schema_exemptions:
            add_issue(issues, "warning", "hot_reload_missing_shared_update_type", f"{key} is hot_reloadable but missing from shared ModelSettingsUpdate.")

    required_runtime_keys = {
        "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens",
        "context_package_token_budget",
        "rq_kmeans_levels",
        "rq_kmeans_max_k",
        "rq_residual_tau",
        "agent_planning_round_budget",
        "agent_max_typed_actions_per_round",
        "agent_repair_round_budget",
        "agent_verification_budget",
        "enable_model_fallback",
        "enable_database_fallback",
    }
    for key in sorted(required_runtime_keys):
        if key not in runtime_payload:
            add_issue(issues, "blocker", "required_runtime_key_missing", f"{key} is missing from runtime payload.")
        if key not in settings_fields:
            add_issue(issues, "blocker", "required_setting_missing", f"{key} is missing from Settings.")

    env_aliases = {
        "FIXED_CHUNK_SIZE_TOKENS",
        "FIXED_CHUNK_OVERLAP_TOKENS",
        "CONTEXT_PACKAGE_TOKEN_BUDGET",
        "RQ_KMEANS_LEVELS",
        "RQ_KMEANS_MAX_K",
        "RQ_RESIDUAL_TAU",
        "AGENT_PLANNING_ROUND_BUDGET",
        "AGENT_MAX_TYPED_ACTIONS_PER_ROUND",
        "AGENT_REPAIR_ROUND_BUDGET",
        "AGENT_VERIFICATION_BUDGET",
    }
    missing_env_examples = sorted(env_aliases - env_keys)
    if missing_env_examples:
        add_issue(issues, "warning", "env_example_missing_runtime_keys", "Some runtime keys are missing from .env.example.", {"missing": missing_env_examples})

    blockers = [issue for issue in issues if issue["severity"] == "blocker"]
    payload = {
        "script": "check_runtime_settings_contract",
        "pass": not blockers,
        "blocker_count": len(blockers),
        "warning_count": len([issue for issue in issues if issue["severity"] == "warning"]),
        "issues": issues,
        "sets": {
            "settings_fields": sorted(settings_fields),
            "hot_reload_settings": sorted(hot_reload),
            "runtime_payload": sorted(runtime_payload),
            "api_update_schema": sorted(update_schema),
            "shared_model_settings_response": sorted(shared_response),
            "shared_model_settings_update": sorted(shared_update),
        },
    }
    report = write_report("runtime_settings_contract", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "blocker_count": payload["blocker_count"], "warning_count": payload["warning_count"]}, ensure_ascii=False))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
