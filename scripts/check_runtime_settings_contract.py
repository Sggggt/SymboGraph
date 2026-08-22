from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the runtime-settings lifecycle and shared contract. "
            "This diagnostic is read-only apart from its output report."
        )
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    from app.core.config import (
        EDGE_DISTANCE_PROTOCOL_ALLOWLIST,
        EDGE_PROJECTION_PROTOCOL_ALLOWLIST,
        EDGE_TYPE_CALIBRATION_PROTOCOL_ALLOWLIST,
        EMBEDDING_API_PROTOCOL_ALLOWLIST,
        HOT_RELOAD_SETTINGS,
        MODEL_API_PROTOCOL_ALLOWLIST,
        RQ_MEMBERSHIP_PROTOCOL_ALLOWLIST,
        Settings,
    )
    from app.schemas import ModelSettingsUpdate
    from app.services.runtime_settings import model_settings_payload, runtime_lifecycle_payload

    issues: list[dict[str, Any]] = []
    settings_fields = set(Settings.model_fields)
    hot_reload = set(HOT_RELOAD_SETTINGS)
    runtime_payload = set(model_settings_payload())
    update_schema = set(ModelSettingsUpdate.model_fields)
    shared_response = ts_interface_keys("ModelSettingsResponse")
    shared_update = ts_interface_keys("ModelSettingsUpdate")
    env_keys = env_example_keys()
    lifecycle = runtime_lifecycle_payload()
    rebuild_lifecycle = set(lifecycle.get("rebuild_required") or [])
    hot_lifecycle = set(lifecycle.get("hot_reloadable") or [])
    runtime_payload_exemptions = {"chat_api_key", "graph_api_key", "embedding_api_key", "model_bridge_admin_token", "model_bridge_port"}
    update_schema_exemptions = {"chat_api_key", "graph_api_key", "embedding_api_key", "model_bridge_admin_token", "model_bridge_port", "enable_model_fallback"}

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
        "chat_api_protocol",
        "graph_api_protocol",
        "embedding_api_protocol",
        "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens",
        "context_package_token_budget",
        "concept_i18n_enabled",
        "query_facet_bilingual_enabled",
        "rq_kmeans_levels",
        "rq_kmeans_max_k",
        "rq_residual_tau",
        "agent_planning_round_budget",
        "agent_max_typed_actions_per_round",
        "agent_repair_round_budget",
        "agent_verification_budget",
        "traversal_observation_budget",
        "edge_distance_protocol",
        "rq_membership_protocol",
        "edge_projection_protocol",
        "edge_type_calibration_protocol",
        "rq_membership_temperature",
        "rq_membership_top_m",
        "rq_membership_probability_threshold",
        "enable_auto_tpe",
        "tpe_trial_budget",
        "tpe_startup_random_trials",
        "tpe_good_quantile_gamma",
        "tpe_probe_query_budget",
        "tpe_trial_timeout_seconds",
        "tpe_candidate_pool_size",
        "operating_point_hard_gate_max_edge_density",
        "operating_point_hard_gate_max_isolated_ratio",
        "operating_point_hard_gate_max_hubness_ratio",
        "operating_point_hard_gate_min_structure_recovery_rate",
        "operating_point_hard_gate_max_candidate_latency_p95_ms",
        "enable_model_fallback",
        "enable_database_fallback",
    }
    for key in sorted(required_runtime_keys):
        if key not in runtime_payload:
            add_issue(issues, "blocker", "required_runtime_key_missing", f"{key} is missing from runtime payload.")
        if key not in settings_fields:
            add_issue(issues, "blocker", "required_setting_missing", f"{key} is missing from Settings.")

    env_aliases = {
        "CHAT_API_PROTOCOL",
        "GRAPH_API_PROTOCOL",
        "EMBEDDING_API_PROTOCOL",
        "FIXED_CHUNK_SIZE_TOKENS",
        "FIXED_CHUNK_OVERLAP_TOKENS",
        "CONTEXT_PACKAGE_TOKEN_BUDGET",
        "SOURCE_IO_CONCURRENCY",
        "CONCEPT_I18N_ENABLED",
        "QUERY_FACET_BILINGUAL_ENABLED",
        "RQ_KMEANS_LEVELS",
        "RQ_KMEANS_MAX_K",
        "RQ_RESIDUAL_TAU",
        "AGENT_COARSE_INITIAL_BUDGET",
        "AGENT_COARSE_TOP_K",
        "AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET",
        "AGENT_MID_INITIAL_BUDGET",
        "AGENT_CHUNK_INITIAL_BUDGET",
        "AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET",
        "AGENT_PLANNING_ROUND_BUDGET",
        "AGENT_MAX_TYPED_ACTIONS_PER_ROUND",
        "AGENT_REPAIR_ROUND_BUDGET",
        "AGENT_VERIFICATION_BUDGET",
        "TRAVERSAL_OBSERVATION_BUDGET",
        "EDGE_DISTANCE_PROTOCOL",
        "RQ_MEMBERSHIP_PROTOCOL",
        "EDGE_PROJECTION_PROTOCOL",
        "EDGE_TYPE_CALIBRATION_PROTOCOL",
        "RQ_MEMBERSHIP_TEMPERATURE",
        "RQ_MEMBERSHIP_TOP_M",
        "RQ_MEMBERSHIP_PROBABILITY_THRESHOLD",
        "ENABLE_AUTO_TPE",
        "TPE_TRIAL_BUDGET",
        "TPE_STARTUP_RANDOM_TRIALS",
        "TPE_GOOD_QUANTILE_GAMMA",
        "TPE_PROBE_QUERY_BUDGET",
        "TPE_TRIAL_TIMEOUT_SECONDS",
        "TPE_CANDIDATE_POOL_SIZE",
        "OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY",
        "OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO",
        "OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO",
        "OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE",
        "OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS",
    }
    missing_env_examples = sorted(env_aliases - env_keys)
    if missing_env_examples:
        add_issue(issues, "warning", "env_example_missing_runtime_keys", "Some runtime keys are missing from .env.example.", {"missing": missing_env_examples})

    rebuild_graph_settings = {
        "edge_distance_protocol",
        "rq_membership_protocol",
        "edge_projection_protocol",
        "edge_type_calibration_protocol",
        "rq_membership_temperature",
        "rq_membership_top_m",
        "rq_membership_probability_threshold",
    }
    for key in sorted(rebuild_graph_settings):
        if key not in rebuild_lifecycle:
            add_issue(
                issues,
                "blocker",
                "graph_setting_missing_rebuild_lifecycle",
                f"{key} must be classified rebuild_required.",
            )
        if key in hot_lifecycle or key in hot_reload:
            add_issue(
                issues,
                "blocker",
                "graph_setting_misclassified_hot_reload",
                f"{key} changes active graph semantics and cannot be hot_reloadable.",
            )
        if key not in update_schema:
            add_issue(issues, "blocker", "graph_setting_missing_update_schema", f"{key} is missing from ModelSettingsUpdate.")
        if key not in shared_response or key not in shared_update:
            add_issue(
                issues,
                "blocker",
                "graph_setting_missing_shared_contract",
                f"{key} is missing from the shared response/update contract.",
            )

    protocol_allowlists = {
        "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_ALLOWLIST,
        "rq_membership_protocol": RQ_MEMBERSHIP_PROTOCOL_ALLOWLIST,
        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_ALLOWLIST,
        "edge_type_calibration_protocol": EDGE_TYPE_CALIBRATION_PROTOCOL_ALLOWLIST,
    }
    for key, allowlist in protocol_allowlists.items():
        if len(allowlist) != 1:
            add_issue(
                issues,
                "blocker",
                "graph_protocol_allowlist_not_closed",
                f"{key} must use a closed local implementation allowlist.",
                {"allowlist": sorted(allowlist)},
            )
        for protocol in allowlist:
            lowered = protocol.lower()
            if any(forbidden in lowered for forbidden in ("prompt", "model", "llm", "expression")):
                add_issue(
                    issues,
                    "blocker",
                    "graph_protocol_contains_forbidden_dynamic_language",
                    f"{key} contains a forbidden dynamic protocol token.",
                    {"protocol": protocol},
                )

    if MODEL_API_PROTOCOL_ALLOWLIST != frozenset({"openai", "anthropic"}):
        add_issue(
            issues,
            "blocker",
            "model_api_protocol_allowlist_not_closed",
            "Model API protocols must be exactly openai and anthropic.",
            {"allowlist": sorted(MODEL_API_PROTOCOL_ALLOWLIST)},
        )
    if EMBEDDING_API_PROTOCOL_ALLOWLIST != frozenset({"openai"}):
        add_issue(
            issues,
            "blocker",
            "embedding_api_protocol_allowlist_not_closed",
            "Embedding API protocols must currently be exactly openai; Anthropic Messages has no embedding contract.",
            {"allowlist": sorted(EMBEDDING_API_PROTOCOL_ALLOWLIST)},
        )
    for key, expected_lifecycle in (
        ("chat_api_protocol", "hot_reloadable"),
        ("graph_api_protocol", "rebuild_required"),
        ("embedding_api_protocol", "rebuild_required"),
    ):
        if key not in settings_fields or key not in runtime_payload:
            add_issue(
                issues,
                "blocker",
                "model_api_protocol_missing_runtime_contract",
                f"{key} must exist in Settings and the public runtime payload.",
            )
        if key not in update_schema or key not in shared_response or key not in shared_update:
            add_issue(
                issues,
                "blocker",
                "model_api_protocol_missing_typed_contract",
                f"{key} must exist in the API and shared typed contracts.",
            )
        lifecycle_set = hot_lifecycle if expected_lifecycle == "hot_reloadable" else rebuild_lifecycle
        opposite_set = rebuild_lifecycle if expected_lifecycle == "hot_reloadable" else hot_lifecycle
        if key not in lifecycle_set or key in opposite_set:
            add_issue(
                issues,
                "blocker",
                "model_api_protocol_lifecycle_mismatch",
                f"{key} must be classified only as {expected_lifecycle}.",
            )

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
            "rebuild_required": sorted(rebuild_lifecycle),
            "hot_reloadable": sorted(hot_lifecycle),
        },
    }
    report = write_report("runtime_settings_contract", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "blocker_count": payload["blocker_count"], "warning_count": payload["warning_count"]}, ensure_ascii=False))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
