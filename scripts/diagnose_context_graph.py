from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write four-layer context graph diagnostics to output/.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    return parser.parse_args()


def main() -> None:
    from sqlalchemy import func, select

    from app.models import ChunkRelationEdge, ChunkRelationGraphState, ChunkStructureNode, FineCluster, FineClusterEdge, FineClusterMembership
    from app.services.context_graph import context_graph_stats, graph_layer_payload

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        layers = {}
        for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
            layers[layer] = graph_layer_payload(db, knowledge_base.id, layer, limit=50)
        relation_state = db.scalar(
            select(ChunkRelationGraphState)
            .where(ChunkRelationGraphState.knowledge_base_id == knowledge_base.id, ChunkRelationGraphState.state == "active")
            .order_by(ChunkRelationGraphState.created_at.desc())
        )
        rq_edge_counts = {
            edge_type: db.scalar(
                select(func.count(ChunkRelationEdge.id)).where(
                    ChunkRelationEdge.knowledge_base_id == knowledge_base.id,
                    ChunkRelationEdge.edge_type == edge_type,
                )
            )
            or 0
            for edge_type in ("rq_hierarchy_near", "rq_prefix_sibling", "rq_residual_near")
        }
        rq_cluster_edge_counts = {
            edge_type: db.scalar(
                select(func.count(FineClusterEdge.id))
                .join(FineCluster, FineClusterEdge.source_cluster_id == FineCluster.id)
                .where(
                    FineCluster.knowledge_base_id == knowledge_base.id,
                    FineClusterEdge.edge_type == edge_type,
                )
            )
            or 0
            for edge_type in ("rq_parent_child", "rq_sibling", "rq_centroid_near", "rq_overlap_bridge")
        }
        rq_model = ((relation_state.diagnostics_json or {}).get("rq_kmeans") or {}) if relation_state else {}
        rq = {
            "enabled": bool(rq_model.get("enabled")),
            "model": {
                "levels": rq_model.get("levels"),
                "codebook_sizes": rq_model.get("codebook_sizes"),
                "embedding_dimensions": rq_model.get("embedding_dimensions"),
                "tau_r": rq_model.get("tau_r"),
                "index_protocol": rq_model.get("index_protocol"),
            },
            "cluster_count": db.scalar(select(func.count(FineCluster.id)).where(FineCluster.knowledge_base_id == knowledge_base.id, FineCluster.rq_level.is_not(None))) or 0,
            "membership_count": db.scalar(
                select(func.count(FineClusterMembership.id))
                .join(FineCluster, FineClusterMembership.fine_cluster_id == FineCluster.id)
                .where(FineCluster.knowledge_base_id == knowledge_base.id, FineClusterMembership.residual_norm.is_not(None))
            )
            or 0,
            "edge_counts": rq_edge_counts,
            "cluster_edge_counts": rq_cluster_edge_counts,
            "pass": bool(relation_state)
            and bool(rq_model.get("enabled"))
            and all(count > 0 for count in rq_edge_counts.values())
            and all(count > 0 for count in rq_cluster_edge_counts.values()),
        }
        relation_state_id = relation_state.id if relation_state else None
        target_relation_edge_counts = {
            edge_type: (
                db.scalar(
                    select(func.count(ChunkRelationEdge.id)).where(
                        ChunkRelationEdge.graph_state_id == relation_state_id,
                        ChunkRelationEdge.edge_type == edge_type,
                    )
                )
                if relation_state_id
                else 0
            )
            or 0
            for edge_type in ("co_retrieved", "same_table_formula_context")
        }
        missing_reasons = ((relation_state.diagnostics_json or {}).get("missing_target_relation_edge_type_reasons") or {}) if relation_state else {}
        unsupported_fine_edges = (
            db.scalar(
                select(func.count(FineClusterEdge.id)).where(
                    FineClusterEdge.graph_state_id == relation_state_id,
                    (FineClusterEdge.support_chunk_edge_ids_json.is_(None)) | (func.json_array_length(FineClusterEdge.support_chunk_edge_ids_json) <= 0),
                )
            )
            if relation_state_id
            else 0
        ) or 0
        missing_relation_metrics = (
            db.scalar(
                select(func.count(ChunkRelationEdge.id)).where(
                    ChunkRelationEdge.graph_state_id == relation_state_id,
                    (ChunkRelationEdge.distance.is_(None)) | (ChunkRelationEdge.raw_strength.is_(None)),
                )
            )
            if relation_state_id
            else 0
        ) or 0
        structure_node_types = {
            row[0]
            for row in db.execute(
                select(ChunkStructureNode.node_type)
                .where(ChunkStructureNode.knowledge_base_id == knowledge_base.id)
                .distinct()
            ).all()
        }
        required_structure_types = {"document", "section", "page", "region", "paragraph"}
        checks = {
            "rq_complete": rq["pass"],
            "target_relation_edges_present_or_blocked": all(count > 0 or missing_reasons.get(edge_type) for edge_type, count in target_relation_edge_counts.items()),
            "fine_edges_supported": unsupported_fine_edges == 0,
            "relation_metrics_present": missing_relation_metrics == 0,
            "structure_closure_types_present": required_structure_types.issubset(structure_node_types),
        }
        payload = {
            "script": "diagnose_context_graph",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "stats": context_graph_stats(db, knowledge_base.id),
            "rq_kmeans": rq,
            "checks": checks,
            "pass": all(checks.values()),
            "target_relation_edge_counts": target_relation_edge_counts,
            "missing_target_relation_edge_type_reasons": missing_reasons,
            "unsupported_fine_edge_count": unsupported_fine_edges,
            "missing_relation_metric_count": missing_relation_metrics,
            "structure_node_types": sorted(structure_node_types),
            "layers": layers,
        }
        report = write_report("diagnose_context_graph", payload)
        print(json.dumps({"output": str(report), "pass": payload["pass"], "knowledge_base_id": knowledge_base.id, "stats": payload["stats"], "checks": checks}, ensure_ascii=False, default=str))
        if not payload["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
