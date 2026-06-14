from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _context_graph_maintenance import REPO_ROOT, resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check technical-spec closure gates and write output/spec_compliance_*.json.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    return parser.parse_args()


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, evidence: dict[str, Any] | None = None) -> None:
    issues.append({"severity": severity, "code": code, "message": message, "evidence": evidence or {}})


def source_text(path: str) -> str:
    target = REPO_ROOT / path
    return target.read_text(encoding="utf-8") if target.exists() else ""


def static_checks() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    forbidden_files = [
        "apps/api/app/services/evidence_graph.py",
        "apps/api/app/services/evidence_graph_payload.py",
        "apps/api/app/services/evidence_signal_projection.py",
    ]
    for rel_path in forbidden_files:
        if (REPO_ROOT / rel_path).exists():
            add_issue(issues, "blocker", "legacy_active_file", f"Legacy evidence/signal file still exists: {rel_path}")
    app_sources = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in ("apps/api/app", "apps/web/src", "packages/shared/src")
        for path in (REPO_ROOT / root).rglob("*")
        if path.is_file() and path.suffix in {".py", ".ts", ".tsx"}
    )
    if "strategy_profile_v3" in app_sources:
        add_issue(issues, "blocker", "legacy_profile_contract", "Active source still references strategy_profile_v3.")
    models = source_text("apps/api/app/models.py")
    for class_name in ("AgentPlan", "AgentAction", "AgentObservation"):
        if f"class {class_name}" not in models:
            add_issue(issues, "blocker", "missing_agent_table", f"{class_name} is not defined in models.py.")
    context_graph = source_text("apps/api/app/services/context_graph.py")
    for token in ("full_counts", "agent_operating_envelope_hash", "runtime_settings_hash", "restore_context_package"):
        if token not in context_graph:
            add_issue(issues, "blocker", "context_graph_contract_gap", f"context_graph.py is missing {token}.")
    agent_graph = source_text("apps/api/app/services/agent_graph.py")
    for token in ("propose_agent_plan", "validate_typed_actions", "verify_answer_against_context", "repair_round_budget", "update_policy_state_from_reward"):
        if token not in agent_graph:
            add_issue(issues, "blocker", "agent_closure_gap", f"agent_graph.py is missing {token}.")
    return issues


def db_checks(knowledge_base_id: str | None, knowledge_base_name: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sqlalchemy import func, select

    from app.models import (
        AgentAction,
        AgentObservation,
        AgentPlan,
        ChunkRelationEdge,
        CitationVerification,
        FineCluster,
        FineClusterEdge,
        FineClusterMembership,
        GraphRetrievalStep,
        PolicyState,
        RewardEvent,
        RetrievalTrace,
    )
    from app.services.context_graph import context_graph_stats, graph_layer_payload

    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if not knowledge_base_id and not knowledge_base_name:
        add_issue(issues, "warning", "db_checks_skipped", "No knowledge base was supplied; only static checks ran.")
        return summary, issues
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=knowledge_base_id, knowledge_base_name=knowledge_base_name)
        stats = context_graph_stats(db, knowledge_base.id)
        summary["knowledge_base_id"] = knowledge_base.id
        summary["knowledge_base_name"] = knowledge_base.name
        summary["stats"] = stats
        counts = stats.get("counts") or {}
        state_ids = stats.get("active_state_ids") or {}
        relation_state_id = state_ids.get("chunk_relation_graph_state_id")
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
            rq_edge_types = {
                edge_type: db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id, ChunkRelationEdge.edge_type == edge_type)) or 0
                for edge_type in ("rq_hierarchy_near", "rq_prefix_sibling", "rq_residual_near")
            }
            rq_cluster_edges = {
                edge_type: db.scalar(select(func.count(FineClusterEdge.id)).where(FineClusterEdge.graph_state_id == relation_state_id, FineClusterEdge.edge_type == edge_type)) or 0
                for edge_type in ("rq_parent_child", "rq_sibling", "rq_centroid_near", "rq_overlap_bridge")
            }
            bridge_edges = db.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.graph_state_id == relation_state_id, ChunkRelationEdge.is_bridge.is_(True))) or 0
            rq_memberships = (
                db.scalar(
                    select(func.count(FineClusterMembership.id))
                    .join(FineCluster, FineClusterMembership.fine_cluster_id == FineCluster.id)
                    .where(FineCluster.graph_state_id == relation_state_id, FineClusterMembership.residual_norm.is_not(None))
                )
                or 0
            )
            summary["rq"] = {"edge_types": rq_edge_types, "cluster_edge_types": rq_cluster_edges, "memberships": rq_memberships}
            summary["bridge_edges"] = bridge_edges
            if any(count <= 0 for count in rq_edge_types.values()) or any(count <= 0 for count in rq_cluster_edges.values()) or rq_memberships <= 0:
                add_issue(issues, "blocker", "rq_graph_incomplete", "RQ is enabled but required RQ graph nodes/edges/memberships are incomplete.", summary["rq"])
            if bridge_edges <= 0:
                add_issue(issues, "blocker", "bridge_edges_missing", "No retained bridge edges were found in the active relation graph.")
        latest_trace = db.scalar(select(RetrievalTrace).where(RetrievalTrace.knowledge_base_id == knowledge_base.id).order_by(RetrievalTrace.created_at.desc()))
        if latest_trace:
            step_layers = set(
                db.scalars(select(GraphRetrievalStep.layer).where(GraphRetrievalStep.retrieval_trace_id == latest_trace.id)).all()
            )
            if "structure" not in step_layers:
                add_issue(issues, "blocker", "trace_missing_structure_restore", "Latest retrieval trace does not include structure restoration.")
        else:
            add_issue(issues, "warning", "no_retrieval_trace", "No retrieval traces exist yet.")
        if db.scalar(select(func.count(AgentPlan.id)).where(AgentPlan.knowledge_base_id == knowledge_base.id)) or 0:
            if (db.scalar(select(func.count(AgentAction.id)).join(AgentPlan, AgentAction.plan_id == AgentPlan.id).where(AgentPlan.knowledge_base_id == knowledge_base.id)) or 0) <= 0:
                add_issue(issues, "blocker", "agent_actions_missing", "Agent plans exist without typed actions.")
            if (db.scalar(select(func.count(AgentObservation.id)).where(AgentObservation.run_id.is_not(None))) or 0) <= 0:
                add_issue(issues, "blocker", "agent_observations_missing", "Agent observations are missing.")
        else:
            add_issue(issues, "warning", "no_agent_plan", "No Agent plans exist yet; run QA before final acceptance.")
        latest_verification = db.scalar(select(CitationVerification).where(CitationVerification.knowledge_base_id == knowledge_base.id).order_by(CitationVerification.created_at.desc()))
        if latest_verification and (latest_verification.diagnostics_json or {}).get("verification_method") == "context_package_span_presence_v1":
            add_issue(issues, "blocker", "old_citation_verification", "Latest citation verification still uses span-presence-only method.")
        if (db.scalar(select(func.count(RewardEvent.id)).where(RewardEvent.knowledge_base_id == knowledge_base.id)) or 0) > 0:
            latest_policy = db.scalar(select(PolicyState).where(PolicyState.knowledge_base_id == knowledge_base.id).order_by(PolicyState.created_at.desc()))
            if not latest_policy or not (latest_policy.reward_summary_json or {}).get("last_reward_event_id"):
                add_issue(issues, "blocker", "policy_not_updated_from_reward", "Reward events exist but latest policy state does not summarize the last reward.")
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
