from __future__ import annotations

import argparse
import json

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report
from _quality_gate import audit_graph_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write four-layer context graph diagnostics to output/.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sqlalchemy import func, inspect, select

    from app.models import (
        Chunk,
        ChunkRelationEdge,
        ChunkRelationGraphState,
        ChunkStructureNode,
        CoarseConcept,
        CoarseConceptEdge,
        MidConcept,
        MidConceptEdge,
        RQPrefix,
        RQPrefixMembership,
    )
    from app.services.context_graph import context_graph_stats, graph_layer_payload
    from app.services.graph_state_hashes import chunk_business_references

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        layers = {}
        for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
            layers[layer] = graph_layer_payload(db, knowledge_base.id, layer, limit=50)
        stats = context_graph_stats(db, knowledge_base.id)
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
        relation_edges = (
            list(
                db.scalars(
                    select(ChunkRelationEdge).where(
                        ChunkRelationEdge.graph_state_id == relation_state_id
                    )
                ).all()
            )
            if relation_state_id
            else []
        )
        memberships = (
            list(
                db.scalars(
                    select(RQPrefixMembership)
                    .join(RQPrefix, RQPrefixMembership.rq_prefix_id == RQPrefix.id)
                    .where(RQPrefix.graph_state_id == relation_state_id)
                ).all()
            )
            if relation_state_id
            else []
        )
        active_state_ids = stats.get("active_state_ids") or {}
        mid_state_id = active_state_ids.get("mid_concept_state_id")
        coarse_state_id = active_state_ids.get("coarse_concept_state_id")
        mid_edges = (
            list(
                db.scalars(
                    select(MidConceptEdge).where(
                        MidConceptEdge.concept_state_id == mid_state_id
                    )
                ).all()
            )
            if mid_state_id
            else []
        )
        coarse_edges = (
            list(
                db.scalars(
                    select(CoarseConceptEdge).where(
                        CoarseConceptEdge.coarse_state_id == coarse_state_id
                    )
                ).all()
            )
            if coarse_state_id
            else []
        )

        relation_chunk_ids = {
            str(chunk_id)
            for edge in relation_edges
            for chunk_id in (edge.source_chunk_id, edge.target_chunk_id)
        }
        relation_chunks = (
            list(
                db.scalars(
                    select(Chunk).where(Chunk.id.in_(sorted(relation_chunk_ids)))
                ).all()
            )
            if relation_chunk_ids
            else []
        )
        chunk_business_keys = (
            chunk_business_references(db, relation_chunks).key_by_id
            if relation_chunks
            else {}
        )
        mid_concept_ids = {
            str(concept_id)
            for edge in mid_edges
            for concept_id in (edge.source_concept_id, edge.target_concept_id)
        }
        mid_concepts = {
            str(row.id): row
            for row in (
                db.scalars(
                    select(MidConcept).where(
                        MidConcept.id.in_(sorted(mid_concept_ids))
                    )
                ).all()
                if mid_concept_ids
                else []
            )
        }
        coarse_concept_ids = {
            str(concept_id)
            for edge in coarse_edges
            for concept_id in (edge.source_concept_id, edge.target_concept_id)
        }
        coarse_concepts = {
            str(row.id): row
            for row in (
                db.scalars(
                    select(CoarseConcept).where(
                        CoarseConcept.id.in_(sorted(coarse_concept_ids))
                    )
                ).all()
                if coarse_concept_ids
                else []
            )
        }
        projection_prefix_ids = {
            str(prefix_id)
            for concept in [*mid_concepts.values(), *coarse_concepts.values()]
            for prefix_id in (
                getattr(concept, "support_rq_l3_prefix_id", None),
                getattr(concept, "support_rq_l2_prefix_id", None),
            )
            if prefix_id
        }
        projection_prefix_keys = {
            str(row.id): str(row.rq_prefix_key or "")
            for row in (
                db.scalars(
                    select(RQPrefix).where(
                        RQPrefix.id.in_(sorted(projection_prefix_ids))
                    )
                ).all()
                if projection_prefix_ids
                else []
            )
        }

        def projected_edge_payload(edge, layer: str) -> dict:
            concept_by_id = (
                mid_concepts if layer == "mid" else coarse_concepts
            )
            source_concept = concept_by_id.get(str(edge.source_concept_id))
            target_concept = concept_by_id.get(str(edge.target_concept_id))
            scope_field = (
                "support_rq_l3_prefix_id"
                if layer == "mid"
                else "support_rq_l2_prefix_id"
            )
            source_prefix_id = getattr(source_concept, scope_field, None)
            target_prefix_id = getattr(target_concept, scope_field, None)
            return {
                "id": edge.id,
                "source_concept_id": edge.source_concept_id,
                "target_concept_id": edge.target_concept_id,
                "source_scope_key": projection_prefix_keys.get(
                    str(source_prefix_id), ""
                ),
                "target_scope_key": projection_prefix_keys.get(
                    str(target_prefix_id), ""
                ),
                "source_rq_prefix_id": (
                    str(source_prefix_id) if source_prefix_id else ""
                ),
                "target_rq_prefix_id": (
                    str(target_prefix_id) if target_prefix_id else ""
                ),
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "distance": edge.distance,
                "projected_strength_raw": edge.projected_strength_raw,
                "projected_distance_raw": edge.projected_distance_raw,
                "raw_strength_summary": edge.raw_strength_summary_json or {},
                "projection_normalization_stats": edge.projection_normalization_stats_json or {},
                "edge_projection_protocol_hash": edge.edge_projection_protocol_hash,
                "source_algorithm": edge.source_algorithm,
                "protocol_version": edge.protocol_version,
                "state_hash": edge.state_hash,
                "support_chunk_ids": edge.support_chunk_ids_json or [],
                "support_chunk_edge_ids": edge.support_chunk_edge_ids_json or [],
                "support_mid_edge_ids": (
                    edge.support_mid_edge_ids_json or []
                    if layer == "coarse"
                    else []
                ),
                "diagnostics": edge.diagnostics_json or {},
            }

        graph_quality = audit_graph_quality(
            {
                "declared_counts": stats.get("counts") or {},
                "chunk_business_keys": chunk_business_keys,
                "rq_prefix_scope_keys": projection_prefix_keys,
                "relation_edges": [
                    {
                        "id": edge.id,
                        "source_chunk_id": edge.source_chunk_id,
                        "target_chunk_id": edge.target_chunk_id,
                        "edge_type": edge.edge_type,
                        "weight": edge.weight,
                        "distance": edge.distance,
                        "raw_strength": edge.raw_strength,
                        "raw_strength_summary": edge.raw_strength_summary_json or {},
                        "normalization_stats": edge.normalization_stats_json or {},
                        "support": edge.support_json or {},
                        "features": edge.features_json or {},
                        "source_algorithm": edge.source_algorithm,
                        "protocol_version": edge.protocol_version,
                        "edge_distance_protocol_hash": edge.edge_distance_protocol_hash,
                        "graph_state_hash": edge.graph_state_hash,
                    }
                    for edge in relation_edges
                ],
                "relation_calibration": (
                    (relation_state.diagnostics_json or {}).get("edge_type_calibration")
                    if relation_state
                    else {}
                )
                or {},
                "rq_memberships": [
                    {
                        "id": membership.id,
                        "rq_prefix_id": membership.rq_prefix_id,
                        "chunk_id": membership.chunk_id,
                        "membership_score": membership.membership_score,
                        "membership_role": membership.membership_role,
                        "membership_entropy": membership.membership_entropy,
                        "residual_norm": membership.residual_norm,
                        "rank": membership.rank,
                        "rq_path": membership.rq_path or [],
                        "role_evaluation": (
                            (membership.diagnostics_json or {}).get(
                                "membership_role_evaluation"
                            )
                            or {}
                        ),
                    }
                    for membership in memberships
                ],
                "rq_membership_diagnostics": (
                    (relation_state.diagnostics_json or {}).get("rq_membership")
                    if relation_state
                    else {}
                )
                or {},
                "projected_edges": {
                    "mid": [
                        projected_edge_payload(edge, "mid") for edge in mid_edges
                    ],
                    "coarse": [
                        projected_edge_payload(edge, "coarse")
                        for edge in coarse_edges
                    ],
                },
                "projection_calibration": {
                    "mid": ((stats.get("diagnostics") or {}).get("mid") or {}).get(
                        "projection_calibration"
                    )
                    or {},
                    "coarse": ((stats.get("diagnostics") or {}).get("coarse") or {}).get(
                        "projection_calibration"
                    )
                    or {},
                },
            }
        )
        checks = {
            "rq_complete": rq["pass"],
            "dense_semantic_edges_present": dense_edge_counts.get("dense_semantic", 0) > 0,
            "forbidden_relation_edge_types_absent": not forbidden_relation_edge_types,
            "rq_prefix_edges_removed": not legacy_rq_prefix_edges_table_present,
            "relation_metrics_present": missing_relation_metrics == 0,
            "structure_closure_types_present": required_structure_types.issubset(structure_node_types),
            "versioned_graph_quality_gate": bool(graph_quality["pass"]),
        }
        payload = {
            "script": "diagnose_context_graph",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "stats": stats,
            "rq_kmeans": rq,
            "checks": checks,
            "pass": all(checks.values()),
            "dense_relation_edge_counts": dense_edge_counts,
            "forbidden_relation_edge_types": forbidden_relation_edge_types,
            "missing_relation_metric_count": missing_relation_metrics,
            "legacy_rq_prefix_edges_table_present": legacy_rq_prefix_edges_table_present,
            "structure_node_types": sorted(structure_node_types),
            "quality_gate": graph_quality,
            "layers": layers,
        }
        report = write_report("diagnose_context_graph", payload)
        print(json.dumps({"output": str(report), "pass": payload["pass"], "knowledge_base_id": knowledge_base.id, "stats": payload["stats"], "checks": checks}, ensure_ascii=False, default=str))
        if not payload["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
