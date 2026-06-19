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
    from sqlalchemy import func, inspect, select

    from app.models import ChunkRelationEdge, ChunkRelationGraphState, ChunkStructureNode, RQPrefix, RQPrefixMembership
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
            "cluster_count": db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == knowledge_base.id, RQPrefix.rq_level.is_not(None))) or 0,
            "membership_count": db.scalar(
                select(func.count(RQPrefixMembership.id))
                .join(RQPrefix, RQPrefixMembership.rq_prefix_id == RQPrefix.id)
                .where(RQPrefix.knowledge_base_id == knowledge_base.id, RQPrefixMembership.residual_norm.is_not(None))
            )
            or 0,
            "pass": bool(relation_state)
            and bool(rq_model.get("enabled"))
            and (db.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == knowledge_base.id, RQPrefix.rq_level.is_not(None))) or 0) > 0,
        }
        relation_state_id = relation_state.id if relation_state else None
        legacy_rq_prefix_edges_table_present = "rq_prefix_edges" in inspect(db.bind).get_table_names()
        dense_edge_counts = {
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
            for edge_type in ("dense_semantic", "dense_cross_document_bridge", "dense_cross_language_bridge")
        }
        forbidden_relation_edge_types = []
        if relation_state_id:
            active_dense_edge_types = set(dense_edge_counts)
            forbidden_relation_edge_types = sorted(
                row[0]
                for row in db.execute(
                    select(ChunkRelationEdge.edge_type)
                    .where(ChunkRelationEdge.graph_state_id == relation_state_id)
                    .distinct()
                ).all()
                if row[0] not in active_dense_edge_types
            )
        missing_relation_metrics = (
            db.scalar(
                select(func.count(ChunkRelationEdge.id)).where(
                    ChunkRelationEdge.graph_state_id == relation_state_id,
                    (ChunkRelationEdge.distance.is_(None)) | (ChunkRelationEdge.raw_strength.is_(None)) | (ChunkRelationEdge.edge_distance_protocol_hash.is_(None)),
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
            "dense_semantic_edges_present": dense_edge_counts.get("dense_semantic", 0) > 0,
            "forbidden_relation_edge_types_absent": not forbidden_relation_edge_types,
            "rq_prefix_edges_removed": not legacy_rq_prefix_edges_table_present,
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
            "dense_relation_edge_counts": dense_edge_counts,
            "forbidden_relation_edge_types": forbidden_relation_edge_types,
            "missing_relation_metric_count": missing_relation_metrics,
            "legacy_rq_prefix_edges_table_present": legacy_rq_prefix_edges_table_present,
            "structure_node_types": sorted(structure_node_types),
            "layers": layers,
        }
        report = write_report("diagnose_context_graph", payload)
        print(json.dumps({"output": str(report), "pass": payload["pass"], "knowledge_base_id": knowledge_base.id, "stats": payload["stats"], "checks": checks}, ensure_ascii=False, default=str))
        if not payload["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
