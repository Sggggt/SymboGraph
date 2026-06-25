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
    forbidden_active_tokens = {
        "BM25Record": "BM25 records were removed from active storage and contracts.",
        "rank_bm25": "BM25 libraries must not be imported by active source.",
        "reconcile_bm25_records": "Legacy BM25 reconciliation is no longer an active script entry.",
        "agent_rq_prefix_entry_budget": "rq-prefix-entry budget was replaced by rq-prefix-address seed budget.",
        "agent_coarse_entry_budget": "old coarse entry budget was replaced by staged traversal budgets.",
        "agent_chunk_candidate_budget": "old chunk candidate budget was replaced by agent_chunk_top_k and per-parent budgets.",
        "agent_ambiguous_edge_distance": "ambiguous-edge thresholds were replaced by path distance green/gray/hard thresholds.",
        "follow_ambiguous_edge": "ambiguous edge actions were replaced by evaluate_gray_zone_path.",
        "ambiguous_edge_decisions": "ambiguous edge decisions were replaced by gray_zone_path_decisions.",
    }
    for token, message in forbidden_active_tokens.items():
        if token in app_sources:
            add_issue(issues, "blocker", "legacy_active_token", f"Active source still references {token}: {message}")
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
    from sqlalchemy import func, inspect, select

    from app.models import (
        AgentAction,
        AgentObservation,
        AgentPlan,
        ChunkRelationEdge,
        ChunkRelationGraphState,
        ChunkStructureNode,
        CoarseConcept,
        CitationVerification,
        RQPrefix,
        RQPrefixMembership,
        GraphRetrievalStep,
        MidConcept,
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
            summary["rq"] = {
                "active_pair_edges": False,
                "prefixes": rq_prefix_count,
                "memberships": rq_memberships,
            }
            summary["dense_relation_edge_counts"] = dense_edge_counts
            summary["forbidden_relation_edge_types"] = sorted(forbidden_relation_edge_types)
            summary["concept_projection_coverage"] = {
                "rq_l3_prefixes": rq_l3_prefix_count,
                "mid_concepts": mid_concept_count,
                "rq_l2_prefixes": rq_l2_prefix_count,
                "coarse_concepts": coarse_concept_count,
            }
            summary["bridge_edges"] = bridge_edges
            if forbidden_relation_edge_types:
                add_issue(issues, "blocker", "forbidden_active_relation_edge_type", "Active chunk relation graph contains non-dense edge types.", {"edge_types": sorted(forbidden_relation_edge_types)})
            if dense_edge_counts.get("dense_semantic", 0) <= 0:
                add_issue(issues, "blocker", "dense_semantic_edges_missing", "Active chunk relation graph must retain dense_semantic edges.", {"edge_counts": dense_edge_counts})
            if rq_prefix_count <= 0 or rq_memberships <= 0:
                add_issue(issues, "blocker", "rq_graph_incomplete", "RQ is enabled but required RQ nodes/memberships are incomplete.", summary["rq"])
            if mid_concept_count != rq_l3_prefix_count:
                add_issue(issues, "blocker", "rq_l3_mid_projection_incomplete", "Active mid concepts must be a one-to-one projection of active RQ L3 prefixes.", summary["concept_projection_coverage"])
            if coarse_concept_count != rq_l2_prefix_count:
                add_issue(issues, "blocker", "rq_l2_coarse_projection_incomplete", "Active coarse concepts must be a one-to-one projection of active RQ L2 prefixes.", summary["concept_projection_coverage"])
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
            add_issue(issues, "warning", "no_agent_plan", "No Agent plans exist yet; run QA before final acceptance.")
        latest_verification = db.scalar(select(CitationVerification).where(CitationVerification.knowledge_base_id == knowledge_base.id).order_by(CitationVerification.created_at.desc()))
        if latest_verification and (latest_verification.diagnostics_json or {}).get("verification_method") == "context_package_span_presence_v1":
            add_issue(issues, "blocker", "old_citation_verification", "Latest citation verification still uses span-presence-only method.")
        if latest_verification:
            source_span = latest_verification.source_span_json or {}
            required_span_keys = {"document_version_id", "chunk_id", "char_span", "page_range", "section_path", "bbox", "context_package_id", "retrieval_trace_id", "verification_id"}
            missing_span_keys = sorted(key for key in required_span_keys if not source_span.get(key))
            if missing_span_keys:
                add_issue(issues, "blocker", "citation_source_span_incomplete", "CitationVerification.source_span_json is missing required raw span fields.", {"missing": missing_span_keys, "source_span": source_span})
            if (latest_verification.diagnostics_json or {}).get("verification_method") != "structure_plus_llm_entailment_v1":
                add_issue(issues, "blocker", "citation_verification_protocol_not_strict", "Latest citation verification must use structure_plus_llm_entailment_v1.")
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
