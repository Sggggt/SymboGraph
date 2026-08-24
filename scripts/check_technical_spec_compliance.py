from __future__ import annotations

import argparse
import ast
import json
from typing import Any

from _context_graph_maintenance import REPO_ROOT, resolve_knowledge_base, session_scope, write_report


ACTIVE_SOURCE_ROOTS = ("apps/api/app", "apps/web/src", "packages/shared/src")
HISTORICAL_SOURCE_PREFIXES: tuple[str, ...] = ()
LEGACY_TOKEN_PATH_EXEMPTIONS: dict[str, set[str]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check technical-spec closure gates and write output/spec_compliance_*.json.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    return parser.parse_args()


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, evidence: dict[str, Any] | None = None) -> None:
    issues.append({"severity": severity, "code": code, "message": message, "evidence": evidence or {}})


def _edge_value(edge: Any, field: str) -> Any:
    return edge.get(field) if isinstance(edge, dict) else getattr(edge, field, None)


def _sha256_hex(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def cross_language_edge_identity_audit(
    edges: list[Any],
    identities_by_chunk: dict[str, dict[str, Any]],
    *,
    expected_scope_hash: str,
    detail_limit: int = 50,
) -> dict[str, Any]:
    """Replay endpoint language identity and channel/calibration facts per edge."""

    invalid_details: list[dict[str, Any]] = []
    detection_hashes: set[str] = set()
    protocol_versions: set[str] = set()
    scope_hashes: set[str] = set()
    calibration_stats_hashes: set[str] = set()
    language_pair_counts: dict[str, int] = {}
    quota_reason_counts: dict[str, int] = {}
    model_call_count = 0
    valid_count = 0
    for edge in edges:
        features = dict(_edge_value(edge, "features_json") or {})
        raw_summary = dict(_edge_value(edge, "raw_strength_summary_json") or {})
        normalization = dict(_edge_value(edge, "normalization_stats_json") or {})
        support = dict(_edge_value(edge, "support_json") or {})
        source_identity = identities_by_chunk.get(
            str(_edge_value(edge, "source_chunk_id")), {}
        )
        target_identity = identities_by_chunk.get(
            str(_edge_value(edge, "target_chunk_id")), {}
        )
        endpoint_identities = [source_identity, target_identity]
        errors: list[str] = []
        if not all(
            identity.get("valid") and identity.get("known")
            for identity in endpoint_identities
        ):
            errors.append("endpoint_identity_not_known_valid")
        endpoint_languages = sorted(
            str(identity.get("language") or "")
            for identity in endpoint_identities
        )
        if len(set(endpoint_languages)) != 2 or "" in endpoint_languages:
            errors.append("endpoint_languages_not_distinct")
        declared_languages = sorted(
            [
                str(features.get("source_language") or ""),
                str(features.get("target_language") or ""),
            ]
        )
        column_languages = sorted(
            [
                str(_edge_value(edge, "source_language") or ""),
                str(_edge_value(edge, "target_language") or ""),
            ]
        )
        if declared_languages != endpoint_languages:
            errors.append("feature_language_pair_mismatch")
        if column_languages != endpoint_languages:
            errors.append("column_language_pair_mismatch")
        endpoint_hashes = sorted(
            str(identity.get("detection_hash") or "")
            for identity in endpoint_identities
        )
        declared_hashes = sorted(
            [
                str(features.get("source_language_detection_hash") or ""),
                str(features.get("target_language_detection_hash") or ""),
            ]
        )
        if declared_hashes != endpoint_hashes or not all(
            _sha256_hex(value) for value in endpoint_hashes
        ):
            errors.append("endpoint_detection_hash_mismatch")
        endpoint_protocols = sorted(
            str(identity.get("protocol_version") or "")
            for identity in endpoint_identities
        )
        declared_protocols = sorted(
            [
                str(features.get("source_language_detection_protocol_version") or ""),
                str(features.get("target_language_detection_protocol_version") or ""),
            ]
        )
        if declared_protocols != endpoint_protocols or "" in endpoint_protocols:
            errors.append("endpoint_detection_protocol_mismatch")
        scope_hash = str(features.get("language_identity_scope_hash") or "")
        if scope_hash != expected_scope_hash or not _sha256_hex(scope_hash):
            errors.append("language_identity_scope_hash_mismatch")
        candidate_channels = {
            str(value) for value in (features.get("candidate_channels") or [])
        }
        all_candidate_channels = {
            str(value)
            for value in (features.get("all_candidate_channels") or [])
        }
        if "cross_language_candidates" not in (
            candidate_channels | all_candidate_channels
        ):
            errors.append("cross_language_candidate_channel_missing")
        if not bool(features.get("is_cross_language")) or not bool(
            _edge_value(edge, "is_cross_language")
        ):
            errors.append("cross_language_flag_missing")
        if not bool(_edge_value(edge, "is_bridge")):
            errors.append("bridge_flag_missing")
        quota_reasons = {
            str(features.get("bridge_quota_reason") or ""),
            str(_edge_value(edge, "bridge_quota_reason") or ""),
        }
        if quota_reasons != {"cross_language_dense_quota"}:
            errors.append("cross_language_quota_reason_mismatch")
        calibration = dict(normalization.get("edge_type_calibration") or {})
        calibration_stats_hash = str(
            raw_summary.get("edge_type_calibration_stats_hash") or ""
        )
        if (
            str(calibration.get("edge_type") or "")
            != "dense_cross_language_bridge"
            or str(calibration.get("stats_hash") or "")
            != calibration_stats_hash
            or not _sha256_hex(calibration_stats_hash)
        ):
            errors.append("edge_type_calibration_mismatch")
        edge_model_calls = support.get("model_call_count")
        if type(edge_model_calls) is not int or edge_model_calls != 0:
            errors.append("support_model_call_count_nonzero")
        else:
            model_call_count += edge_model_calls
        for value in endpoint_hashes:
            if _sha256_hex(value):
                detection_hashes.add(value)
        protocol_versions.update(value for value in endpoint_protocols if value)
        if scope_hash:
            scope_hashes.add(scope_hash)
        if calibration_stats_hash:
            calibration_stats_hashes.add(calibration_stats_hash)
        language_pair = "<->".join(endpoint_languages)
        language_pair_counts[language_pair] = (
            language_pair_counts.get(language_pair, 0) + 1
        )
        quota_reason = str(features.get("bridge_quota_reason") or "")
        quota_reason_counts[quota_reason] = (
            quota_reason_counts.get(quota_reason, 0) + 1
        )
        if errors:
            if len(invalid_details) < detail_limit:
                invalid_details.append(
                    {
                        "edge_id": str(_edge_value(edge, "id") or ""),
                        "errors": sorted(set(errors)),
                    }
                )
        else:
            valid_count += 1
    audit = {
        "protocol_version": "cross_language_edge_identity_replay_v1",
        "edge_count": len(edges),
        "valid_edge_count": valid_count,
        "invalid_edge_count": len(edges) - valid_count,
        "invalid_details": invalid_details,
        "invalid_details_truncated": max(
            len(edges) - valid_count - len(invalid_details), 0
        ),
        "language_pair_counts": dict(sorted(language_pair_counts.items())),
        "detection_hash_count": len(detection_hashes),
        "detection_hashes": sorted(detection_hashes),
        "language_detection_protocol_versions": sorted(protocol_versions),
        "language_identity_scope_hashes": sorted(scope_hashes),
        "edge_type_calibration_stats_hashes": sorted(calibration_stats_hashes),
        "quota_reason_counts": dict(sorted(quota_reason_counts.items())),
        "model_call_count": model_call_count,
    }
    audit["pass"] = bool(edges) and audit["invalid_edge_count"] == 0
    return audit


def source_text(path: str) -> str:
    target = REPO_ROOT / path
    return target.read_text(encoding="utf-8") if target.exists() else ""


def active_source_items() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for root in ACTIVE_SOURCE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx"}:
                continue
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            if any(
                relative_path.startswith(prefix)
                for prefix in HISTORICAL_SOURCE_PREFIXES
            ):
                continue
            items.append(
                (
                    relative_path,
                    path.read_text(encoding="utf-8", errors="ignore"),
                )
            )
    return items


def assigned_string_collection(source: str, name: str) -> set[str] | None:
    """Read a literal module-level action allow/deny list without importing app code."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        target_name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        if target_name != name or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
        if isinstance(literal, (set, list, tuple)) and all(isinstance(item, str) for item in literal):
            return set(literal)
        return None
    return None


def static_checks() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    forbidden_files = [
        "apps/api/app/services/evidence_graph.py",
        "apps/api/app/services/evidence_graph_payload.py",
        "apps/api/app/services/evidence_signal_projection.py",
        "apps/api/tests/test_rq_fuzzy_membership.py",
        "apps/api/migrations/versions/20260821_0042_retire_development_artifacts.py",
        "scripts/cleanup_orphan_mid_rq_memberships.py",
        "scripts/destroy_legacy_derived_data.py",
    ]
    for rel_path in forbidden_files:
        if (REPO_ROOT / rel_path).exists():
            add_issue(issues, "blocker", "legacy_active_file", f"Legacy evidence/signal file still exists: {rel_path}")
    source_items = active_source_items()
    app_sources = "\n".join(source for _path, source in source_items)
    if "strategy_profile_v3" in app_sources:
        add_issue(issues, "blocker", "legacy_profile_contract", "Active source still references strategy_profile_v3.")
    forbidden_active_tokens = {
        "BM25Record": "BM25 records were removed from active storage and contracts.",
        "rank_bm25": "BM25 libraries must not be imported by active source.",
        "reconcile_bm25_records": "Legacy BM25 reconciliation is no longer an active script entry.",
        "agent_rq_prefix_entry_budget": "rq-prefix-entry budget was replaced by rq-prefix-address seed budget.",
        "agent_coarse_entry_budget": "old coarse entry budget was replaced by staged traversal budgets.",
        "agent_chunk_candidate_budget": "old chunk candidate budget was replaced by agent_chunk_top_k and per-parent budgets.",
        "agent_ambiguous_edge_distance": "ambiguous-edge thresholds were replaced by path distance green/gray/hard thresholds.",
        "follow_ambiguous_edge": (
            "ambiguous edge actions were replaced by per-path deterministic_support_progress_v1 decisions "
            "inside the traversal executor; evaluate_gray_zone_path is not an allowed planner/evaluator action."
        ),
        "ambiguous_edge_decisions": "ambiguous edge decisions were replaced by gray_zone_path_decisions.",
    }
    for token, message in forbidden_active_tokens.items():
        exempt_paths = LEGACY_TOKEN_PATH_EXEMPTIONS.get(token, set())
        matched_paths = sorted(
            path
            for path, source in source_items
            if token in source and path not in exempt_paths
        )
        if matched_paths:
            add_issue(
                issues,
                "blocker",
                "legacy_active_token",
                f"Active source still references {token}: {message}",
                {"paths": matched_paths[:20]},
            )
    models = source_text("apps/api/app/models.py")
    for class_name in ("AgentPlan", "AgentAction", "AgentObservation"):
        if f"class {class_name}" not in models:
            add_issue(issues, "blocker", "missing_agent_table", f"{class_name} is not defined in models.py.")
    context_graph = source_text("apps/api/app/services/context_graph.py")
    for token in ("full_counts", "agent_operating_envelope_hash", "runtime_settings_hash", "restore_context_package"):
        if token not in context_graph:
            add_issue(issues, "blocker", "context_graph_contract_gap", f"context_graph.py is missing {token}.")
    language_metadata = source_text("apps/api/app/services/language_metadata.py")
    for token in (
        "document_language_unicode_script_v1",
        "language_detection_hash",
        "load_chunk_language_identities",
        "document_version_hash_mismatch",
    ):
        if token not in language_metadata:
            add_issue(
                issues,
                "blocker",
                "language_identity_contract_gap",
                f"language_metadata.py is missing {token}.",
            )
    for token in (
        "source_language_detection_hash",
        "target_language_detection_hash",
        "language_identity_scope_diagnostics",
    ):
        if token not in context_graph:
            add_issue(
                issues,
                "blocker",
                "cross_language_audit_contract_gap",
                f"context_graph.py is missing {token}.",
            )
    agent_graph = source_text("apps/api/app/services/agent_graph.py")
    for token in ("propose_agent_plan", "validate_typed_actions", "verify_answer_against_context", "repair_round_budget", "update_policy_state_from_reward"):
        if token not in agent_graph:
            add_issue(issues, "blocker", "agent_closure_gap", f"agent_graph.py is missing {token}.")
    allowed_actions = assigned_string_collection(agent_graph, "ALLOWED_TYPED_ACTIONS")
    forbidden_gray_outputs = assigned_string_collection(agent_graph, "FORBIDDEN_GRAY_PLANNER_OUTPUTS")
    if allowed_actions is None:
        add_issue(issues, "blocker", "typed_action_allowlist_unreadable", "ALLOWED_TYPED_ACTIONS must be a static local literal allowlist.")
    elif "evaluate_gray_zone_path" in allowed_actions:
        add_issue(
            issues,
            "blocker",
            "gray_zone_llm_action_allowed",
            "evaluate_gray_zone_path must not be exposed to the planner/evidence evaluator; gray paths are executor-only local decisions.",
        )
    if forbidden_gray_outputs is None or "evaluate_gray_zone_path" not in forbidden_gray_outputs:
        add_issue(
            issues,
            "blocker",
            "gray_zone_llm_action_not_forbidden",
            "FORBIDDEN_GRAY_PLANNER_OUTPUTS must explicitly reject evaluate_gray_zone_path.",
        )
    return issues


def _check_policy_reward_consumption(
    db: Any,
    knowledge_base_id: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify audit-only rewards separately from trainable Agent rewards.

    A missing ``agent_run_id`` is a valid citation/answer audit only when the
    server-owned eligibility diagnostics say exactly that it is ineligible.
    Conversely, merely setting the eligibility flag is not enough to make a
    reward trainable: the AgentRun must exist in the same knowledge base and
    the complete persisted reward replay must validate.
    """

    from sqlalchemy import select

    from app.models import AgentRun, PolicyState, RewardEvent
    from app.services.chunking import stable_hash
    from app.services.policy_reward import replay_policy_reward_event

    rewards = list(
        db.scalars(
            select(RewardEvent)
            .where(RewardEvent.knowledge_base_id == knowledge_base_id)
            .order_by(RewardEvent.created_at.asc(), RewardEvent.id.asc())
        ).all()
    )
    detail_limit = 100
    corrupt_count = 0
    corrupt_details: list[dict[str, Any]] = []
    audit_only_ids: list[str] = []
    eligible_ids: list[str] = []
    consumed_ids: list[str] = []
    unconsumed_ids: list[str] = []

    def record_corruption(
        reward: Any,
        reason: str,
        **evidence: Any,
    ) -> None:
        nonlocal corrupt_count
        corrupt_count += 1
        if len(corrupt_details) < detail_limit:
            corrupt_details.append(
                {
                    "reward_event_id": str(reward.id),
                    "reason": reason,
                    **evidence,
                }
            )

    for reward in rewards:
        context = reward.context_json
        diagnostics = reward.diagnostics_json
        if not isinstance(context, dict) or not isinstance(diagnostics, dict):
            record_corruption(
                reward,
                "reward eligibility payload must use JSON objects",
                context_type=type(context).__name__,
                diagnostics_type=type(diagnostics).__name__,
            )
            continue

        run_ref_present = "agent_run_id" in context
        run_ref = context.get("agent_run_id")
        eligibility_present = "policy_reward_training_eligible" in diagnostics
        eligibility = diagnostics.get("policy_reward_training_eligible")
        reason_present = (
            "policy_reward_training_ineligible_reason" in diagnostics
        )
        ineligible_reason = diagnostics.get(
            "policy_reward_training_ineligible_reason"
        )
        source = diagnostics.get("source")

        if run_ref is None:
            if (
                eligibility_present
                and eligibility is False
                and reason_present
                and ineligible_reason == "missing_agent_run"
                and source == "context_graph_agent_v1"
                and reward.policy_state_id is None
            ):
                audit_only_ids.append(str(reward.id))
                continue
            record_corruption(
                reward,
                "missing AgentRun reference is not a canonical audit-only reward",
                agent_run_id_field_present=run_ref_present,
                eligibility_field_present=eligibility_present,
                eligibility_value=eligibility,
                ineligible_reason_field_present=reason_present,
                ineligible_reason=ineligible_reason,
                source=source,
                policy_state_id=(
                    str(reward.policy_state_id)
                    if reward.policy_state_id is not None
                    else None
                ),
            )
            continue

        if not isinstance(run_ref, str) or not run_ref.strip():
            record_corruption(
                reward,
                "agent_run_id must be a non-empty string or null",
                agent_run_id_type=type(run_ref).__name__,
            )
            continue
        if (
            eligibility_present
            and eligibility is False
            and reason_present
            and ineligible_reason == "retired_protocol_lineage"
            and source == "context_graph_agent_v1"
            and reward.policy_state_id is None
        ):
            run = db.get(AgentRun, run_ref)
            if (
                run is not None
                and str(run.knowledge_base_id) == str(knowledge_base_id)
            ):
                audit_only_ids.append(str(reward.id))
                continue
            record_corruption(
                reward,
                "audit-only reward references an invalid AgentRun",
                agent_run_id=run_ref,
            )
            continue
        if (
            not eligibility_present
            or eligibility is not True
            or not reason_present
            or ineligible_reason is not None
            or source != "context_graph_agent_v1"
        ):
            record_corruption(
                reward,
                "Agent-linked reward has contradictory eligibility diagnostics",
                agent_run_id=run_ref,
                eligibility_field_present=eligibility_present,
                eligibility_value=eligibility,
                ineligible_reason_field_present=reason_present,
                ineligible_reason=ineligible_reason,
                source=source,
            )
            continue

        run = db.get(AgentRun, run_ref)
        if run is None:
            record_corruption(
                reward,
                "agent_run_id references a missing AgentRun",
                agent_run_id=run_ref,
            )
            continue
        if str(run.knowledge_base_id) != str(knowledge_base_id):
            record_corruption(
                reward,
                "agent_run_id references an AgentRun in another knowledge base",
                agent_run_id=run_ref,
                agent_run_knowledge_base_id=str(run.knowledge_base_id),
            )
            continue

        try:
            replay_policy_reward_event(db, reward)
        except Exception as exc:  # noqa: BLE001 - convert replay failure to an auditable blocker
            record_corruption(
                reward,
                "persisted Agent reward replay failed",
                agent_run_id=run_ref,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            continue

        eligible_ids.append(str(reward.id))
        if reward.policy_state_id is None:
            unconsumed_ids.append(str(reward.id))
            continue
        policy_state = db.get(PolicyState, reward.policy_state_id)
        if policy_state is None:
            record_corruption(
                reward,
                "policy_state_id references a missing PolicyState",
                policy_state_id=str(reward.policy_state_id),
            )
            continue
        if str(policy_state.knowledge_base_id) != str(knowledge_base_id):
            record_corruption(
                reward,
                "policy_state_id references a PolicyState in another knowledge base",
                policy_state_id=str(reward.policy_state_id),
                policy_state_knowledge_base_id=str(
                    policy_state.knowledge_base_id
                ),
            )
            continue
        reward_summary = policy_state.reward_summary_json
        if (
            not isinstance(reward_summary, dict)
            or str(reward_summary.get("last_reward_event_id") or "")
            != str(reward.id)
        ):
            record_corruption(
                reward,
                "RewardEvent and PolicyState do not have a reciprocal consumption binding",
                policy_state_id=str(policy_state.id),
                policy_last_reward_event_id=(
                    str(reward_summary.get("last_reward_event_id"))
                    if isinstance(reward_summary, dict)
                    and reward_summary.get("last_reward_event_id") is not None
                    else None
                ),
            )
            continue
        consumed_ids.append(str(reward.id))

    if corrupt_count:
        add_issue(
            issues,
            "blocker",
            "policy_reward_eligibility_or_binding_corrupt",
            (
                "RewardEvent eligibility, AgentRun provenance, persisted replay, "
                "or PolicyState binding is corrupt; audit-only rewards cannot be "
                "promoted by a forged eligibility flag."
            ),
            {
                "corrupt_count": corrupt_count,
                "records": corrupt_details,
                "records_truncated": corrupt_count > len(corrupt_details),
            },
        )
    if unconsumed_ids:
        add_issue(
            issues,
            "blocker",
            "policy_not_updated_from_reward",
            (
                "Validated, explicitly training-eligible Agent rewards exist "
                "without a consuming PolicyState."
            ),
            {
                "unconsumed_reward_event_count": len(unconsumed_ids),
                "reward_event_ids": unconsumed_ids[:detail_limit],
                "records_truncated": len(unconsumed_ids) > detail_limit,
            },
        )

    return {
        "reward_event_count": len(rewards),
        "audit_only_reward_count": len(audit_only_ids),
        "training_eligible_reward_count": len(eligible_ids),
        "consumed_training_reward_count": len(consumed_ids),
        "unconsumed_training_reward_count": len(unconsumed_ids),
        "corrupt_reward_count": corrupt_count,
        "audit_only_reward_event_ids": audit_only_ids[:detail_limit],
        "training_eligible_reward_event_ids": eligible_ids[:detail_limit],
        "consumed_training_reward_event_ids": consumed_ids[:detail_limit],
        "detail_limit": detail_limit,
    }


def db_checks(knowledge_base_id: str | None, knowledge_base_name: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if not knowledge_base_id and not knowledge_base_name:
        add_issue(issues, "warning", "db_checks_skipped", "No knowledge base was supplied; only static checks ran.")
        return summary, issues

    from sqlalchemy import func, inspect, select

    from app.models import (
        AgentAction,
        AgentObservation,
        AgentPlan,
        ChunkRelationEdge,
        ChunkRelationGraphState,
        ChunkStructureNode,
        CoarseConcept,
        CoarseConceptState,
        CitationVerification,
        RQPrefix,
        RQPrefixMembership,
        GraphRetrievalStep,
        MidConcept,
        MidConceptState,
        RetrievalTrace,
    )
    from app.services.context_graph import (
        CONCEPT_NODE_ELIGIBILITY_PROTOCOL_VERSION,
        active_chunks_query,
        context_graph_stats,
        graph_layer_payload,
    )
    from app.services.language_metadata import (
        language_identity_scope_diagnostics,
        load_chunk_language_identities,
    )

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=knowledge_base_id, knowledge_base_name=knowledge_base_name)
        stats = context_graph_stats(db, knowledge_base.id)
        summary["knowledge_base_id"] = knowledge_base.id
        summary["knowledge_base_name"] = knowledge_base.name
        summary["stats"] = stats
        active_chunks = list(db.scalars(active_chunks_query(knowledge_base.id)).all())
        identities_by_chunk = load_chunk_language_identities(db, active_chunks)
        language_diagnostics = language_identity_scope_diagnostics(
            identities_by_chunk
        )
        summary["language_identity"] = language_diagnostics
        if language_diagnostics["invalid_identity_count"]:
            add_issue(
                issues,
                "blocker",
                "active_chunk_language_identity_invalid",
                "Active chunks must reference an active DocumentVersion whose language identity/hash matches the Document snapshot.",
                language_diagnostics,
            )
        if language_diagnostics["unknown_language_count"]:
            add_issue(
                issues,
                "warning",
                "active_chunk_language_unknown",
                "Some active chunks have an audited unknown language and correctly cannot enter the cross-language channel.",
                language_diagnostics,
            )
        counts = stats.get("counts") or {}
        state_ids = stats.get("active_state_ids") or {}
        relation_state_id = state_ids.get("chunk_relation_graph_state_id")
        mid_state_id = state_ids.get("mid_concept_state_id")
        coarse_state_id = state_ids.get("coarse_concept_state_id")
        if counts.get("active_chunks", 0) <= 0:
            add_issue(issues, "blocker", "no_active_chunks", "Knowledge base has no active chunks.")
        if not relation_state_id:
            add_issue(issues, "blocker", "missing_relation_state", "No active relation state is bound to the context graph.")
        for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
            payload = graph_layer_payload(db, knowledge_base.id, layer, limit=25)
            full_counts = payload.get("full_counts") or {}
            sampled_counts = payload.get("sampled_counts") or {}
            if "nodes" not in full_counts or "edges" not in full_counts:
                add_issue(issues, "blocker", "missing_full_counts", f"{layer} payload does not expose node/edge full_counts.")
            if sampled_counts.get("nodes", 0) > full_counts.get("nodes", 0):
                add_issue(issues, "blocker", "sampled_exceeds_full", f"{layer} sampled nodes exceed full nodes.", {"payload": payload})
        if relation_state_id:
            active_dense_edge_types = {"dense_semantic", "dense_cross_document_bridge", "dense_cross_language_bridge"}
            forbidden_relation_edge_types = {
                row[0]
                for row in db.execute(
                    select(ChunkRelationEdge.edge_type)
                    .where(ChunkRelationEdge.graph_state_id == relation_state_id)
                    .distinct()
                ).all()
                if row[0] not in active_dense_edge_types
            }
            dense_edge_counts = {
                edge_type: db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id, ChunkRelationEdge.edge_type == edge_type)) or 0
                for edge_type in sorted(active_dense_edge_types)
            }
            bridge_edges = db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id, ChunkRelationEdge.is_bridge.is_(True))) or 0
            rq_prefix_count = db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.graph_state_id == relation_state_id, RQPrefix.rq_level.is_not(None))) or 0
            rq_l3_prefix_count = db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.graph_state_id == relation_state_id, RQPrefix.rq_level == 3, RQPrefix.state == "active")) or 0
            rq_l2_prefix_count = db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.graph_state_id == relation_state_id, RQPrefix.rq_level == 2, RQPrefix.state == "active")) or 0
            mid_concept_count = (
                db.scalar(select(func.count(MidConcept.id)).where(MidConcept.concept_state_id == mid_state_id, MidConcept.state == "active"))
                if mid_state_id
                else 0
            ) or 0
            coarse_concept_count = (
                db.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.coarse_state_id == coarse_state_id, CoarseConcept.state == "active"))
                if coarse_state_id
                else 0
            ) or 0
            rq_memberships = (
                db.scalar(
                    select(func.count(RQPrefixMembership.id))
                    .join(RQPrefix, RQPrefixMembership.rq_prefix_id == RQPrefix.id)
                    .where(RQPrefix.graph_state_id == relation_state_id, RQPrefixMembership.residual_norm.is_not(None))
                )
                or 0
            )
            relation_state = db.get(ChunkRelationGraphState, relation_state_id)
            rq_membership_diagnostics = dict(
                ((relation_state.diagnostics_json or {}).get(
                    "rq_membership"
                ) or {})
                if relation_state is not None
                else {}
            )
            primary_membership = dict(
                rq_membership_diagnostics.get("primary_membership") or {}
            )
            summary["rq"] = {
                "active_pair_edges": False,
                "prefixes": rq_prefix_count,
                "memberships": rq_memberships,
                "membership_protocol_version": (
                    rq_membership_diagnostics.get(
                        "membership_protocol_version"
                    )
                ),
                "primary_membership": primary_membership,
            }
            summary["dense_relation_edge_counts"] = dense_edge_counts
            cross_language_edges = list(
                db.scalars(
                    select(ChunkRelationEdge).where(
                        ChunkRelationEdge.graph_state_id == relation_state_id,
                        ChunkRelationEdge.edge_type
                        == "dense_cross_language_bridge",
                    )
                ).all()
            )
            cross_language_audit = cross_language_edge_identity_audit(
                cross_language_edges,
                identities_by_chunk,
                expected_scope_hash=str(language_diagnostics["scope_hash"]),
            )
            summary["cross_language_edge_identity_audit"] = (
                cross_language_audit
            )
            summary["forbidden_relation_edge_types"] = sorted(forbidden_relation_edge_types)
            summary["concept_projection_coverage"] = {
                "rq_l3_prefixes": rq_l3_prefix_count,
                "mid_concepts": mid_concept_count,
                "rq_l2_prefixes": rq_l2_prefix_count,
                "coarse_concepts": coarse_concept_count,
            }
            mid_state = (
                db.get(MidConceptState, mid_state_id)
                if mid_state_id
                else None
            )
            coarse_state = (
                db.get(CoarseConceptState, coarse_state_id)
                if coarse_state_id
                else None
            )
            mid_eligibility = dict(
                ((mid_state.stats_json or {}).get(
                    "concept_node_eligibility"
                ) or {})
                if mid_state is not None
                else {}
            )
            coarse_eligibility = dict(
                ((coarse_state.stats_json or {}).get(
                    "concept_node_eligibility"
                ) or {})
                if coarse_state is not None
                else {}
            )
            summary["concept_semantic_compression"] = {
                "protocol_version": (
                    CONCEPT_NODE_ELIGIBILITY_PROTOCOL_VERSION
                ),
                "mid": mid_eligibility,
                "coarse": coarse_eligibility,
            }
            summary["bridge_edges"] = bridge_edges
            if forbidden_relation_edge_types:
                add_issue(issues, "blocker", "forbidden_active_relation_edge_type", "Active chunk relation graph contains non-dense edge types.", {"edge_types": sorted(forbidden_relation_edge_types)})
            if dense_edge_counts.get("dense_semantic", 0) <= 0:
                add_issue(issues, "blocker", "dense_semantic_edges_missing", "Active chunk relation graph must retain dense_semantic edges.", {"edge_counts": dense_edge_counts})
            if (
                len(language_diagnostics.get("language_counts") or {}) >= 2
                and dense_edge_counts.get("dense_cross_language_bridge", 0) <= 0
            ):
                add_issue(
                    issues,
                    "blocker",
                    "cross_language_channel_unreachable",
                    "The active scope contains at least two audited languages but no cross-language relation edge.",
                    {
                        "language_identity": language_diagnostics,
                        "edge_counts": dense_edge_counts,
                    },
                )
            if cross_language_edges and not cross_language_audit["pass"]:
                add_issue(
                    issues,
                    "blocker",
                    "cross_language_edge_identity_replay_failed",
                    "Cross-language relation edges must replay exact endpoint detection hashes, channel/quota facts and type-local calibration.",
                    cross_language_audit,
                )
            if rq_prefix_count <= 0 or rq_memberships <= 0:
                add_issue(issues, "blocker", "rq_graph_incomplete", "RQ is enabled but required RQ nodes/memberships are incomplete.", summary["rq"])
            primary_membership_valid = (
                rq_membership_diagnostics.get(
                    "membership_protocol_version"
                )
                == "rq_primary_chain_v1"
                and int(primary_membership.get("chunk_count") or -1)
                == len(active_chunks)
                and int(
                    primary_membership.get("primary_membership_count")
                    or -1
                )
                == 3 * len(active_chunks)
                and int(primary_membership.get("membership_count") or -1)
                == int(rq_memberships)
                and int(rq_memberships) == 3 * len(active_chunks)
                and int(
                    primary_membership.get(
                        "non_primary_membership_count"
                    )
                    or 0
                )
                == 0
                and int(
                    primary_membership.get(
                        "observed_max_memberships_per_chunk"
                    )
                    or -1
                )
                <= 3
                and primary_membership.get(
                    "all_primary_chains_complete"
                )
                is True
                and primary_membership.get(
                    "all_chunk_cardinalities_exact"
                )
                is True
                and primary_membership.get(
                    "cartesian_expansion_used"
                )
                is False
                and primary_membership.get("model_call_count") == 0
            )
            if not primary_membership_valid:
                add_issue(
                    issues,
                    "blocker",
                    "rq_primary_membership_invalid",
                    (
                        "Active RQ membership must preserve exactly three "
                        "primary prefixes per chunk, materialize exactly "
                        "three memberships per chunk, and prohibit all "
                        "second address-chain expansion."
                    ),
                    summary["rq"],
                )
            concept_compression_valid = (
                mid_eligibility.get("protocol_version")
                == CONCEPT_NODE_ELIGIBILITY_PROTOCOL_VERSION
                and coarse_eligibility.get("protocol_version")
                == CONCEPT_NODE_ELIGIBILITY_PROTOCOL_VERSION
                and mid_eligibility.get(
                    "primary_only_selection_authority"
                )
                is True
                and coarse_eligibility.get(
                    "primary_only_selection_authority"
                )
                is True
                and mid_eligibility.get("model_call_count") == 0
                and coarse_eligibility.get("model_call_count") == 0
                and int(mid_eligibility.get("eligible_count") or -1)
                == int(mid_concept_count)
                and int(coarse_eligibility.get("eligible_count") or -1)
                == int(coarse_concept_count)
                and 0 < int(mid_concept_count) <= len(active_chunks)
                and 0 < int(coarse_concept_count) <= int(mid_concept_count)
                and (
                    len(active_chunks) <= 1
                    or int(mid_concept_count) < len(active_chunks)
                )
                and (
                    int(mid_concept_count) <= 1
                    or int(coarse_concept_count)
                    < int(mid_concept_count)
                )
            )
            if not concept_compression_valid:
                add_issue(
                    issues,
                    "blocker",
                    "concept_semantic_compression_invalid",
                    (
                        "Active Mid/Coarse concepts must use deterministic "
                        "primary-only eligibility and satisfy "
                        "Coarse<=Mid<=chunks with "
                        "strict non-trivial compression."
                    ),
                    summary["concept_semantic_compression"],
                )
            if bridge_edges <= 0:
                add_issue(issues, "blocker", "bridge_edges_missing", "No retained bridge edges were found in the active relation graph.")
            missing_edge_metrics = db.scalar(
                select(func.count(ChunkRelationEdge.id)).where(
                    ChunkRelationEdge.graph_state_id == relation_state_id,
                    (ChunkRelationEdge.distance.is_(None))
                    | (ChunkRelationEdge.raw_strength.is_(None))
                    | (ChunkRelationEdge.edge_distance_protocol_hash.is_(None)),
                )
            ) or 0
            if missing_edge_metrics:
                add_issue(issues, "blocker", "relation_edge_distance_raw_strength_missing", "Active relation edges must carry distance, raw_strength and edge distance protocol hash.", {"count": missing_edge_metrics})
            if "rq_prefix_edges" in inspect(db.bind).get_table_names():
                add_issue(issues, "blocker", "legacy_rq_prefix_edges_table_present", "rq_prefix_edges must be removed from the active schema; run migrations/cleanup.", {"table_present": True})
            structure_node_types = {
                row[0]
                for row in db.execute(
                    select(ChunkStructureNode.node_type)
                    .where(ChunkStructureNode.knowledge_base_id == knowledge_base.id)
                    .distinct()
                ).all()
            }
            required_structure_types = {"document", "section", "page", "region", "paragraph"}
            missing_structure_types = sorted(required_structure_types - structure_node_types)
            if missing_structure_types:
                add_issue(issues, "blocker", "structure_closure_node_type_missing", "Structure graph is missing required closure node types.", {"missing": missing_structure_types, "present": sorted(structure_node_types)})
        latest_trace = db.scalar(select(RetrievalTrace).where(RetrievalTrace.knowledge_base_id == knowledge_base.id).order_by(RetrievalTrace.created_at.desc()))
        if latest_trace:
            if not latest_trace.stage_queues_json or not latest_trace.candidate_pools_json or not latest_trace.topk_selection_json:
                add_issue(issues, "blocker", "trace_staged_selection_missing", "Latest retrieval trace must persist stage_queues_json, candidate_pools_json and topk_selection_json.")
            step_layers = set(
                db.scalars(select(GraphRetrievalStep.layer).where(GraphRetrievalStep.retrieval_trace_id == latest_trace.id)).all()
            )
            if "structure" not in step_layers:
                add_issue(issues, "blocker", "trace_missing_structure_restore", "Latest retrieval trace does not include structure restoration.")
            for required_layer in ("coarse", "mid", "chunk"):
                step = db.scalar(
                    select(GraphRetrievalStep)
                    .where(
                        GraphRetrievalStep.retrieval_trace_id == latest_trace.id,
                        GraphRetrievalStep.layer == required_layer,
                        GraphRetrievalStep.action_type.in_(
                            (
                                "walk_graph_frontier",
                                "select_entry_nodes",
                                "select_seeds_from_mid_rq_membership",
                                "staged_priority_queue_walk",
                                "drill_down_each_coarse_or_direct_mid_entry",
                                "collect_node_queue",
                                "merge_dedupe_rank_top_k",
                            )
                        ),
                    )
                    .order_by(GraphRetrievalStep.step_index.asc())
                )
                if step is None:
                    add_issue(issues, "blocker", "trace_layer_missing", f"Latest retrieval trace is missing {required_layer} traversal step.")
                    continue
                if not step.popped_frontier_state_json and (step.input_json or {}).get("entry_nodes"):
                    add_issue(issues, "blocker", "trace_frontier_pop_missing", f"{required_layer} traversal step has entries but no popped frontier state.")
                if not step.stop_reason:
                    add_issue(issues, "blocker", "trace_convergence_missing", f"{required_layer} traversal step does not record convergence stop reason.")
            rq_prefix_step = db.scalar(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == latest_trace.id, GraphRetrievalStep.layer == "fine"))
            if rq_prefix_step is not None:
                add_issue(issues, "blocker", "rq_prefix_active_traversal_present", "Latest retrieval trace still includes RQ prefix as an active traversal layer.")
            seed_step = db.scalar(
                select(GraphRetrievalStep).where(
                    GraphRetrievalStep.retrieval_trace_id == latest_trace.id,
                    GraphRetrievalStep.layer == "chunk",
                    GraphRetrievalStep.action_type == "select_seeds_from_mid_rq_membership",
                )
            )
            if seed_step is None:
                add_issue(issues, "blocker", "rq_membership_seed_step_missing", "Trace is missing chunk/select_seeds_from_mid_rq_membership.")
        else:
            add_issue(issues, "warning", "no_retrieval_trace", "No retrieval traces exist yet.")
        if db.scalar(select(func.count(AgentPlan.id)).where(AgentPlan.knowledge_base_id == knowledge_base.id)) or 0:
            if (db.scalar(select(func.count(AgentAction.id)).join(AgentPlan, AgentAction.plan_id == AgentPlan.id).where(AgentPlan.knowledge_base_id == knowledge_base.id)) or 0) <= 0:
                add_issue(issues, "blocker", "agent_actions_missing", "Agent plans exist without typed actions.")
            if (db.scalar(select(func.count(AgentObservation.id)).where(AgentObservation.run_id.is_not(None))) or 0) <= 0:
                add_issue(issues, "blocker", "agent_observations_missing", "Agent observations are missing.")
        else:
            add_issue(issues, "warning", "no_agent_plan", "No Agent plans exist yet; run QA before evaluating Agent coverage.")
        latest_verification = db.scalar(select(CitationVerification).where(CitationVerification.knowledge_base_id == knowledge_base.id).order_by(CitationVerification.created_at.desc()))
        if latest_verification and (latest_verification.diagnostics_json or {}).get("verification_method") == "context_package_span_presence_v1":
            add_issue(issues, "blocker", "old_citation_verification", "Latest citation verification still uses span-presence-only method.")
        if latest_verification:
            source_span = latest_verification.source_span_json or {}
            required_span_keys = {"document_version_id", "chunk_id", "char_span", "page_range", "section_path", "bbox", "context_package_id", "retrieval_trace_id", "verification_id"}
            missing_span_keys = sorted(
                key
                for key in required_span_keys
                if key not in source_span or source_span.get(key) is None
            )
            if missing_span_keys:
                add_issue(issues, "blocker", "citation_source_span_incomplete", "CitationVerification.source_span_json is missing required raw span fields.", {"missing": missing_span_keys, "source_span": source_span})
            if (latest_verification.diagnostics_json or {}).get("verification_method") != "claim_structure_plus_llm_entailment_v2":
                add_issue(issues, "blocker", "citation_verification_protocol_not_strict", "Latest citation verification must use claim_structure_plus_llm_entailment_v2.")
        summary["policy_reward_compliance"] = (
            _check_policy_reward_consumption(
                db,
                knowledge_base.id,
                issues,
            )
        )
    return summary, issues


def main() -> None:
    args = parse_args()
    issues = static_checks()
    db_summary, db_issues = db_checks(args.knowledge_base_id, args.knowledge_base_name)
    issues.extend(db_issues)
    blockers = [issue for issue in issues if issue["severity"] == "blocker"]
    payload = {
        "script": "check_technical_spec_compliance",
        "pass": not blockers,
        "blocker_count": len(blockers),
        "warning_count": len([issue for issue in issues if issue["severity"] == "warning"]),
        "issues": issues,
        "db_summary": db_summary,
    }
    report = write_report("spec_compliance", payload)
    print(json.dumps({"output": str(report), "pass": payload["pass"], "blocker_count": payload["blocker_count"], "warning_count": payload["warning_count"]}, ensure_ascii=False))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
