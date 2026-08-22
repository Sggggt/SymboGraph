from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from typing import Any, Iterator, Mapping, Sequence

from sqlalchemy import event, func, select, update
from sqlalchemy.inspection import inspect as sqlalchemy_inspect
from sqlalchemy.orm import Session

from app.models import (
    Chunk,
    ChunkRelationEdge,
    ChunkRelationGraphState,
    CoarseConceptState,
    ContextGraphState,
    IngestionCompensationLog,
    MidConcept,
    MidConceptState,
    RQPrefix,
    RQPrefixMembership,
    RQPrefixPairDiagnostic,
)
from app.services import context_graph
from app.services.chunking import stable_hash
from app.services.ingestion_resource_lock import (
    knowledge_base_ingestion_resource_lock,
)


SCOPED_CONTEXT_GRAPH_REBUILD_PROTOCOL_VERSION = (
    "scoped_context_graph_rebuild_v1"
)
SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_PROTOCOL_VERSION = (
    "scoped_context_graph_cache_invalidation_v1"
)
SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_OPERATION = (
    "scoped_context_graph_cache_invalidation"
)
SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_SESSION_KEY = (
    "_scoped_context_graph_cache_invalidation_intents"
)
CONTEXT_PROTOCOL_IDENTITY_REFRESH_PROTOCOL_VERSION = (
    "context_protocol_identity_refresh_v1"
)
CONTEXT_PROTOCOL_IDENTITY_REFRESH_ALLOWED_REASONS = frozenset(
    {
        "context_graph_canonical_protocol_identities_stale",
        "context_graph_canonical_traversal_envelope_mismatch",
    }
)

SCOPED_REBUILD_CONTRACT: dict[str, dict[str, list[str] | str]] = {
    "rq_membership": {
        "reused_layers": [
            "chunk_structure_graph",
            "chunk_relation_graph",
        ],
        "rebuilt_layers": [
            "rq_membership_graph",
            "mid_concept_graph",
            "coarse_concept_graph",
            "context_graph_state",
        ],
    },
    "mid_concept": {
        "reused_layers": [
            "chunk_structure_graph",
            "chunk_relation_graph",
            "rq_membership_graph",
        ],
        "rebuilt_layers": [
            "mid_concept_graph",
            "coarse_concept_graph",
            "context_graph_state",
        ],
    },
    "coarse_concept": {
        "reused_layers": [
            "chunk_structure_graph",
            "chunk_relation_graph",
            "rq_membership_graph",
            "mid_concept_graph",
        ],
        "rebuilt_layers": [
            "coarse_concept_graph",
            "context_graph_state",
        ],
    },
}


class ScopedContextGraphRebuildError(RuntimeError):
    """A scoped rebuild failed before its active lineage could be switched."""


def _strict_json_copy(value: Any) -> Any:
    return deepcopy(value)


def _replace_identifiers(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_identifiers(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_identifiers(item, replacements) for item in value)
    if isinstance(value, dict):
        return {
            str(key): _replace_identifiers(item, replacements)
            for key, item in value.items()
        }
    return value


def _model_column_values(
    row: Any,
    *,
    exclude: Sequence[str],
) -> dict[str, Any]:
    excluded = set(exclude)
    return {
        attribute.key: _strict_json_copy(getattr(row, attribute.key))
        for attribute in sqlalchemy_inspect(type(row)).mapper.column_attrs
        if attribute.key not in excluded
    }


def _active_chunks(db: Session, knowledge_base_id: str) -> list[Chunk]:
    chunks = list(
        db.scalars(
            context_graph.active_chunks_query(knowledge_base_id)
        ).all()
    )
    if not chunks:
        raise ScopedContextGraphRebuildError(
            "Cannot run a scoped graph rebuild without active chunks"
        )
    return chunks


def plan_context_protocol_identity_refresh(
    db: Session,
    knowledge_base_id: str,
) -> dict[str, Any]:
    """Plan a wrapper-only refresh without changing graph business facts."""

    chunks = _active_chunks(db, knowledge_base_id)
    current = context_graph.latest_context_graph_state(
        db,
        knowledge_base_id,
    )
    if current is None:
        return {
            "protocol_version": (
                CONTEXT_PROTOCOL_IDENTITY_REFRESH_PROTOCOL_VERSION
            ),
            "eligible": False,
            "reason": "active_context_graph_state_missing",
            "admission_reasons": ["context_graph_state_missing"],
            "identity_difference_keys": [],
            "model_call_count": 0,
            "qdrant_write_count": 0,
        }
    try:
        context_graph.active_graph_admission_gate(
            db,
            knowledge_base_id,
            chunks=chunks,
        )
    except context_graph.ActiveContextGraphAdmissionError as exc:
        admission_reasons = list(exc.reasons)
    else:
        admission_reasons = []
    diagnostics = dict(current.diagnostics_json or {})
    envelope = diagnostics.get("canonical_agent_operating_envelope")
    stored_identities = diagnostics.get("canonical_protocol_identities")
    identity_differences: dict[str, dict[str, Any]] = {}
    identity_error: str | None = None
    if not isinstance(envelope, dict) or not isinstance(
        stored_identities,
        dict,
    ):
        identity_error = "canonical_protocol_identity_packet_missing"
    else:
        try:
            expected_identities = context_graph.graph_state_protocol_identities(
                envelope,
                scope="context",
                frozen_protocol_envelope=True,
            )
        except (TypeError, ValueError):
            identity_error = "canonical_protocol_identity_packet_invalid"
        else:
            identity_differences = {
                key: {
                    "stored": stored_identities.get(key),
                    "expected": expected_identities.get(key),
                }
                for key in sorted(
                    set(stored_identities) | set(expected_identities)
                )
                if stored_identities.get(key)
                != expected_identities.get(key)
            }
    allowed_reason_set = set(
        CONTEXT_PROTOCOL_IDENTITY_REFRESH_ALLOWED_REASONS
    )
    actual_reason_set = set(admission_reasons)
    exact_traversal_only_difference = (
        list(identity_differences) == ["traversal_protocol_hash"]
    )
    eligible = bool(
        admission_reasons
        and actual_reason_set.issubset(allowed_reason_set)
        and exact_traversal_only_difference
        and identity_error is None
        and int(
            (stored_identities or {}).get("gray_zone_model_call_count")
            or 0
        )
        == 0
    )
    if not admission_reasons:
        reason = "active_context_graph_already_admitted"
    elif not actual_reason_set.issubset(allowed_reason_set):
        reason = "active_context_graph_has_non_wrapper_staleness"
    elif identity_error is not None:
        reason = identity_error
    elif not exact_traversal_only_difference:
        reason = "protocol_identity_drift_is_not_traversal_only"
    elif not eligible:
        reason = "gray_zone_model_call_count_nonzero"
    else:
        reason = "traversal_protocol_wrapper_refresh_required"
    return {
        "protocol_version": (
            CONTEXT_PROTOCOL_IDENTITY_REFRESH_PROTOCOL_VERSION
        ),
        "eligible": eligible,
        "reason": reason,
        "knowledge_base_id": str(knowledge_base_id),
        "active_context_graph_state_id": str(current.id),
        "active_graph_state_ids": {
            "relation": str(current.chunk_relation_graph_state_id or ""),
            "mid": str(current.mid_concept_state_id or ""),
            "coarse": str(current.coarse_concept_state_id or ""),
            "context": str(current.id),
        },
        "admission_reasons": admission_reasons,
        "allowed_admission_reasons": sorted(allowed_reason_set),
        "identity_difference_keys": list(identity_differences),
        "identity_differences": identity_differences,
        "active_chunk_count": len(chunks),
        "reused_layers": [
            "chunk_structure_graph",
            "chunk_relation_graph",
            "rq_membership_graph",
            "mid_concept_graph",
            "coarse_concept_graph",
            "contextual_index",
            "qdrant_vectors",
        ],
        "rewritten_layers": ["context_graph_state_wrapper"],
        "model_call_count": 0,
        "qdrant_write_count": 0,
        "gray_zone_rule_inputs_modified": False,
        "gray_zone_rule_protocol_modified": False,
    }


def _bound_active_lineage(
    db: Session,
    knowledge_base_id: str,
    *,
    chunks: list[Chunk],
) -> tuple[
    ContextGraphState,
    ChunkRelationGraphState,
    MidConceptState,
    CoarseConceptState,
]:
    current = context_graph.active_graph_admission_gate(
        db,
        knowledge_base_id,
        chunks=chunks,
    )
    if current is None:
        raise ScopedContextGraphRebuildError(
            "Scoped graph rebuild requires an admitted active context graph"
        )
    relation = (
        db.get(
            ChunkRelationGraphState,
            current.chunk_relation_graph_state_id,
        )
        if current.chunk_relation_graph_state_id
        else None
    )
    mid = (
        db.get(MidConceptState, current.mid_concept_state_id)
        if current.mid_concept_state_id
        else None
    )
    coarse = (
        db.get(CoarseConceptState, current.coarse_concept_state_id)
        if current.coarse_concept_state_id
        else None
    )
    if relation is None or mid is None or coarse is None:
        raise ScopedContextGraphRebuildError(
            "Scoped graph rebuild requires a complete active relation/RQ/mid/coarse lineage"
        )
    if (
        str(relation.knowledge_base_id) != str(knowledge_base_id)
        or str(mid.knowledge_base_id) != str(knowledge_base_id)
        or str(coarse.knowledge_base_id) != str(knowledge_base_id)
    ):
        raise ScopedContextGraphRebuildError(
            "Scoped graph rebuild lineage escaped the requested knowledge base"
        )
    return current, relation, mid, coarse


def _relation_card(relation: ChunkRelationGraphState) -> dict[str, Any]:
    card = dict(
        (relation.diagnostics_json or {}).get("canonical_state_hash_card")
        or {}
    )
    if (
        not context_graph.verify_state_hash_card(card)
        or str(card.get("state_hash") or "") != str(relation.state_hash or "")
    ):
        raise ScopedContextGraphRebuildError(
            "Scoped graph rebuild requires a valid active relation hash card"
        )
    return card


def _count(
    db: Session,
    model: type[Any],
    *conditions: Any,
) -> int:
    return int(
        db.scalar(select(func.count(model.id)).where(*conditions))
        or 0
    )


def _upstream_snapshot(
    db: Session,
    *,
    requested_scope: str,
    context_state: ContextGraphState,
    relation_state: ChunkRelationGraphState,
    mid_state: MidConceptState,
) -> dict[str, Any]:
    relation_card = _relation_card(relation_state)
    relation_components = dict(relation_card.get("component_hashes") or {})
    hashes: dict[str, str] = {
        "chunk_structure_graph": str(
            context_state.structure_graph_hash or ""
        ),
        "chunk_relation_graph": str(
            relation_components.get("edge_facts") or ""
        ),
    }
    counts: dict[str, int] = {
        "chunk_relation_edges": _count(
            db,
            ChunkRelationEdge,
            ChunkRelationEdge.graph_state_id == relation_state.id,
        )
    }
    if requested_scope in {"mid_concept", "coarse_concept"}:
        rq_card = dict(relation_card.get("rq_state_card") or {})
        rq_pair_card = dict(
            relation_card.get("rq_pair_state_card") or {}
        )
        if not context_graph.verify_state_hash_card(rq_card):
            raise ScopedContextGraphRebuildError(
                "Scoped graph rebuild requires a valid reused RQ hash card"
            )
        if not context_graph.verify_state_hash_card(rq_pair_card):
            raise ScopedContextGraphRebuildError(
                "Scoped graph rebuild requires a valid reused RQ-pair hash card"
            )
        hashes["rq_membership_graph"] = str(rq_card["state_hash"])
        hashes["rq_prefix_pair_aggregate"] = str(
            rq_pair_card["state_hash"]
        )
        counts.update(
            {
                "rq_prefixes": _count(
                    db,
                    RQPrefix,
                    RQPrefix.graph_state_id == relation_state.id,
                ),
                "rq_memberships": _count(
                    db,
                    RQPrefixMembership,
                    RQPrefixMembership.rq_prefix_id.in_(
                        select(RQPrefix.id).where(
                            RQPrefix.graph_state_id == relation_state.id
                        )
                    ),
                ),
                "rq_prefix_pair_diagnostics": _count(
                    db,
                    RQPrefixPairDiagnostic,
                    RQPrefixPairDiagnostic.graph_state_id
                    == relation_state.id,
                ),
            }
        )
    if requested_scope == "coarse_concept":
        hashes["mid_concept_graph"] = str(mid_state.state_hash or "")
        counts["mid_concepts"] = _count(
            db,
            MidConcept,
            MidConcept.concept_state_id == mid_state.id,
        )
    return {
        "hashes": hashes,
        "row_counts": counts,
        "snapshot_hash": stable_hash(
            {"hashes": hashes, "row_counts": counts}
        ),
    }


def _clone_relation_edges(
    db: Session,
    *,
    source_state: ChunkRelationGraphState,
    target_state: ChunkRelationGraphState,
) -> dict[tuple[str, str, str], ChunkRelationEdge]:
    source_edges = list(
        db.scalars(
            select(ChunkRelationEdge)
            .where(ChunkRelationEdge.graph_state_id == source_state.id)
            .order_by(
                ChunkRelationEdge.source_chunk_id.asc(),
                ChunkRelationEdge.target_chunk_id.asc(),
                ChunkRelationEdge.edge_type.asc(),
                ChunkRelationEdge.id.asc(),
            )
        ).all()
    )
    cloned_edges: list[ChunkRelationEdge] = []
    for source_edge in source_edges:
        values = _model_column_values(
            source_edge,
            exclude=(
                "id",
                "graph_state_id",
                "graph_state_hash",
                "created_at",
            ),
        )
        cloned = ChunkRelationEdge(
            **values,
            graph_state_id=target_state.id,
            graph_state_hash=target_state.state_hash,
        )
        db.add(cloned)
        cloned_edges.append(cloned)
    db.flush()
    replacements = {
        str(source.id): str(cloned.id)
        for source, cloned in zip(source_edges, cloned_edges, strict=True)
    }
    for cloned in cloned_edges:
        for field_name in (
            "raw_strength_summary_json",
            "normalization_stats_json",
            "features_json",
            "support_json",
            "diagnostics_json",
        ):
            setattr(
                cloned,
                field_name,
                _replace_identifiers(
                    getattr(cloned, field_name) or {},
                    replacements,
                ),
            )
    db.flush()
    result: dict[tuple[str, str, str], ChunkRelationEdge] = {}
    for edge in cloned_edges:
        key = (
            str(edge.source_chunk_id),
            str(edge.target_chunk_id),
            str(edge.edge_type),
        )
        if key in result:
            raise ScopedContextGraphRebuildError(
                "Scoped RQ rebuild found duplicate bottom relation edge identity"
            )
        result[key] = edge
    return result


def _clone_relation_state_for_rq(
    db: Session,
    *,
    source_state: ChunkRelationGraphState,
) -> ChunkRelationGraphState:
    values = _model_column_values(
        source_state,
        exclude=("id", "created_at"),
    )
    diagnostics = dict(values.get("diagnostics_json") or {})
    diagnostics["scoped_relation_reuse"] = {
        "protocol_version": SCOPED_CONTEXT_GRAPH_REBUILD_PROTOCOL_VERSION,
        "source_relation_state_id": str(source_state.id),
        "source_relation_state_hash": str(source_state.state_hash),
        "bottom_relation_rows_copied_without_recalibration": True,
        "tpe_operating_point_reused": True,
    }
    values.update(
        {
            "state": "active",
            "diagnostics_json": diagnostics,
        }
    )
    cloned = ChunkRelationGraphState(**values)
    db.add(cloned)
    db.flush()
    return cloned


def _finalize_rq_relation_state(
    db: Session,
    *,
    relation_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    edges: dict[tuple[str, str, str], ChunkRelationEdge],
    rq_prefixes: list[RQPrefix],
    vector_runtime_target: Any,
) -> None:
    pair_diagnostics = dict(
        (relation_state.diagnostics_json or {}).get(
            "rq_prefix_pair_diagnostics"
        )
        or {}
    )
    pair_hash = str(pair_diagnostics.get("diagnostic_hash") or "")
    if len(pair_hash) != 64:
        raise ScopedContextGraphRebuildError(
            "Scoped RQ rebuild did not produce a canonical prefix-pair diagnostic hash"
        )
    stats = {
        **context_graph.relation_graph_stats(
            chunks,
            list(edges.values()),
            rq_prefixes,
        ),
        "rq_prefix_pair_diagnostic_count": int(
            pair_diagnostics.get("diagnostic_count") or 0
        ),
        "rq_prefix_pair_diagnostic_count_by_type": dict(
            pair_diagnostics.get("diagnostic_count_by_type") or {}
        ),
    }
    relation_state.stats_json = stats
    relation_state.diagnostics_json = {
        **dict(relation_state.diagnostics_json or {}),
        "orphan_chunk_rate": stats["orphan_chunk_rate"],
        "singleton_rate": stats["singleton_rate"],
        "bridge_edge_count": stats["bridge_edges"],
        "protocol": context_graph.RELATION_PROTOCOL_VERSION,
    }
    db.flush()
    relation_hash_card = context_graph.build_relation_state_hash_card(
        db,
        relation_state,
        chunks,
        protocol_identities=context_graph.graph_state_protocol_identities(
            scope="relation"
        ),
        vector_identity=context_graph.canonical_vector_identity_card(
            vector_runtime_target
        ),
    )
    rq_state_card = dict(relation_hash_card["rq_state_card"])
    rq_pair_state_card = dict(
        relation_hash_card["rq_pair_state_card"]
    )
    rq_prefix_facts_hash = str(
        (rq_state_card.get("component_hashes") or {}).get("prefix_facts")
        or rq_state_card.get("prefix_facts_hash")
        or ""
    )
    rq_membership_hash = str(rq_state_card["state_hash"])
    rq_membership_diagnostics = dict(
        (relation_state.diagnostics_json or {}).get("rq_membership")
        or {}
    )
    rq_kmeans_diagnostics = dict(
        (relation_state.diagnostics_json or {}).get("rq_kmeans")
        or {}
    )
    relation_state.diagnostics_json = {
        **dict(relation_state.diagnostics_json or {}),
        "rq_prefix_facts_hash": rq_prefix_facts_hash,
        "rq_membership_hash": rq_membership_hash,
        "rq_membership": {
            **rq_membership_diagnostics,
            "membership_hash": rq_membership_hash,
            "canonical_state_hash_protocol_version": rq_state_card[
                "protocol_version"
            ],
        },
        "rq_kmeans": {
            **rq_kmeans_diagnostics,
            "membership_hash": rq_membership_hash,
        },
        "rq_prefix_pair_diagnostics_hash": pair_hash,
        "rq_prefix_pair_diagnostics_protocol_hash": (
            context_graph.rq_prefix_pair_diagnostic_protocol_hash()
        ),
        "rq_prefix_pair_aggregate_hash": rq_pair_state_card["state_hash"],
        "canonical_state_hash_card": relation_hash_card,
        "chunk_business_scope_hash": relation_hash_card[
            "chunk_business_scope_hash"
        ],
    }
    relation_state.state_hash = str(relation_hash_card["state_hash"])
    for edge in edges.values():
        edge.graph_state_hash = relation_state.state_hash
    db.flush()


def _rebuild_rq_on_reused_relation(
    db: Session,
    *,
    source_relation_state: ChunkRelationGraphState,
    chunks: list[Chunk],
    vector_runtime_target: Any,
) -> ChunkRelationGraphState:
    relation_state = _clone_relation_state_for_rq(
        db,
        source_state=source_relation_state,
    )
    edges = _clone_relation_edges(
        db,
        source_state=source_relation_state,
        target_state=relation_state,
    )
    vectors = {
        str(chunk.id): context_graph.vector_for_chunk(
            db,
            str(chunk.id),
            vector_runtime_target=vector_runtime_target,
        )
        for chunk in chunks
    }
    missing = sorted(
        chunk_id for chunk_id, vector in vectors.items() if not vector
    )
    if missing:
        raise ScopedContextGraphRebuildError(
            "Scoped RQ rebuild cannot reuse relation rows with missing active vectors: "
            + ", ".join(missing[:8])
        )
    rq_prefixes = context_graph.build_rq_prefixes(
        db,
        relation_state,
        chunks,
        vectors,
        edges,
    )
    context_graph.annotate_relation_edges_with_rq_boundaries(
        db,
        relation_state,
        edges,
    )
    _finalize_rq_relation_state(
        db,
        relation_state=relation_state,
        chunks=chunks,
        edges=edges,
        rq_prefixes=rq_prefixes,
        vector_runtime_target=vector_runtime_target,
    )
    return relation_state


def _deactivate_superseded_states(
    db: Session,
    *,
    requested_scope: str,
    relation_state: ChunkRelationGraphState,
    mid_state: MidConceptState,
    coarse_state: CoarseConceptState,
    context_state: ContextGraphState,
) -> None:
    targets: list[tuple[type[Any], str]] = []
    if requested_scope == "rq_membership":
        targets.append((ChunkRelationGraphState, str(relation_state.id)))
    if requested_scope in {"rq_membership", "mid_concept"}:
        targets.append((MidConceptState, str(mid_state.id)))
    targets.extend(
        [
            (CoarseConceptState, str(coarse_state.id)),
            (ContextGraphState, str(context_state.id)),
        ]
    )
    for model, keep_id in targets:
        db.execute(
            update(model)
            .where(
                model.knowledge_base_id == context_state.knowledge_base_id,
                model.state == "active",
                model.id != keep_id,
            )
            .values(state="inactive")
        )
    db.flush()


def _stage_cache_invalidation_intent(
    db: Session,
    *,
    knowledge_base_id: str,
    context_state: ContextGraphState,
    requested_scope: str,
) -> IngestionCompensationLog:
    payload = {
        "protocol_version": (
            SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_PROTOCOL_VERSION
        ),
        "knowledge_base_id": str(knowledge_base_id),
        "context_graph_state_id": str(context_state.id),
        "context_graph_hash": str(context_state.context_graph_hash),
        "requested_scope": requested_scope,
        "phase": "database_commit_pending",
        "attempt_count": 0,
        "completed_at": None,
    }
    intent = IngestionCompensationLog(
        knowledge_base_id=knowledge_base_id,
        operation=SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_OPERATION,
        target_ids_json=[str(context_state.id)],
        payload_json={**payload, "payload_hash": stable_hash(payload)},
        status="cache_invalidation_pending",
    )
    db.add(intent)
    db.flush()
    pending = db.info.setdefault(
        SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_SESSION_KEY,
        {},
    )
    pending[str(intent.id)] = {
        "intent_id": str(intent.id),
        "knowledge_base_id": str(knowledge_base_id),
    }
    return intent


def _validate_cache_intent_payload(
    intent: IngestionCompensationLog,
) -> dict[str, Any]:
    if intent.operation != SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_OPERATION:
        raise ScopedContextGraphRebuildError(
            "Scoped rebuild cache intent operation mismatch"
        )
    payload = dict(intent.payload_json or {})
    supplied_hash = str(payload.pop("payload_hash", ""))
    if supplied_hash != stable_hash(payload):
        raise ScopedContextGraphRebuildError(
            "Scoped rebuild cache intent payload hash mismatch"
        )
    if (
        payload.get("protocol_version")
        != SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_PROTOCOL_VERSION
        or str(payload.get("knowledge_base_id") or "")
        != str(intent.knowledge_base_id)
    ):
        raise ScopedContextGraphRebuildError(
            "Scoped rebuild cache intent provenance mismatch"
        )
    return payload


def deliver_scoped_rebuild_cache_invalidation(
    db: Session,
    intent_id: str,
) -> dict[str, Any]:
    intent = db.get(IngestionCompensationLog, intent_id)
    if intent is None:
        raise ScopedContextGraphRebuildError(
            f"Scoped rebuild cache invalidation intent not found: {intent_id}"
        )
    payload = _validate_cache_intent_payload(intent)
    if intent.status == "completed":
        return {
            "intent_id": str(intent.id),
            "status": "completed",
            "delivered": True,
            "idempotent_replay": True,
        }
    attempt_count = int(payload.get("attempt_count") or 0) + 1
    attempted_at = datetime.utcnow().isoformat()
    try:
        delivered = context_graph.invalidate_context_graph_cache_after_commit(
            str(intent.knowledge_base_id),
            strict=True,
        )
        if delivered is not True:
            raise RuntimeError("strict cache invalidation was not delivered")
    except Exception as exc:
        failed_payload = {
            **payload,
            "phase": "cache_invalidation_pending",
            "attempt_count": attempt_count,
            "last_attempt_at": attempted_at,
            "completed_at": None,
        }
        intent.payload_json = {
            **failed_payload,
            "payload_hash": stable_hash(failed_payload),
        }
        intent.status = "cache_invalidation_pending"
        intent.error_message = (
            "Scoped cache invalidation delivery failed: "
            f"{type(exc).__name__}"
        )
        db.flush()
        return {
            "intent_id": str(intent.id),
            "status": intent.status,
            "delivered": False,
            "error_type": type(exc).__name__,
        }
    completed_payload = {
        **payload,
        "phase": "completed",
        "attempt_count": attempt_count,
        "last_attempt_at": attempted_at,
        "completed_at": datetime.utcnow().isoformat(),
    }
    intent.payload_json = {
        **completed_payload,
        "payload_hash": stable_hash(completed_payload),
    }
    intent.status = "completed"
    intent.error_message = None
    db.flush()
    return {
        "intent_id": str(intent.id),
        "status": intent.status,
        "delivered": True,
        "idempotent_replay": False,
    }


def reconcile_scoped_rebuild_cache_invalidations(
    db: Session,
    *,
    knowledge_base_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    bounded_limit = max(1, min(int(limit), 100))
    conditions = [
        IngestionCompensationLog.operation
        == SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_OPERATION,
        IngestionCompensationLog.status == "cache_invalidation_pending",
    ]
    if knowledge_base_id is not None:
        conditions.append(
            IngestionCompensationLog.knowledge_base_id
            == str(knowledge_base_id)
        )
    rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(*conditions)
            .order_by(IngestionCompensationLog.created_at.asc())
            .limit(bounded_limit)
        ).all()
    )
    results = [
        deliver_scoped_rebuild_cache_invalidation(db, str(row.id))
        for row in rows
    ]
    return {
        "protocol_version": (
            SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_PROTOCOL_VERSION
        ),
        "checked": len(rows),
        "completed": sum(
            int(item.get("status") == "completed") for item in results
        ),
        "pending": sum(
            int(item.get("status") != "completed") for item in results
        ),
        "results": results,
    }


def _consume_cache_invalidation_intents(
    session: Session,
    *,
    committed: bool,
) -> None:
    pending = dict(
        session.info.pop(
            SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_SESSION_KEY,
            {},
        )
        or {}
    )
    if not committed or not pending:
        return
    from app.db import SessionLocal

    for item in pending.values():
        with SessionLocal() as delivery_db:
            deliver_scoped_rebuild_cache_invalidation(
                delivery_db,
                str(item["intent_id"]),
            )
            delivery_db.commit()


@event.listens_for(Session, "after_commit")
def _scoped_rebuild_cache_after_commit(session: Session) -> None:
    if session.in_nested_transaction():
        return
    _consume_cache_invalidation_intents(session, committed=True)


@event.listens_for(Session, "after_rollback")
def _scoped_rebuild_cache_after_rollback(session: Session) -> None:
    _consume_cache_invalidation_intents(session, committed=False)


@event.listens_for(Session, "after_transaction_end")
def _scoped_rebuild_cache_after_transaction_end(
    session: Session,
    transaction: Any,
) -> None:
    if transaction.parent is not None:
        return
    session.info.pop(
        SCOPED_CONTEXT_GRAPH_CACHE_INVALIDATION_SESSION_KEY,
        None,
    )


@contextmanager
def _candidate_savepoint(db: Session) -> Iterator[None]:
    """Keep candidate writes rollbackable without committing caller facts."""

    with db.begin_nested():
        yield


async def refresh_context_protocol_identity(
    db: Session,
    knowledge_base_id: str,
) -> ContextGraphState:
    """Atomically replace only a stale canonical context-state wrapper."""

    async with knowledge_base_ingestion_resource_lock(
        db,
        knowledge_base_id,
        operation="context_graph_rebuild",
        batch_id=None,
    ) as resource_lock:
        plan = plan_context_protocol_identity_refresh(
            db,
            knowledge_base_id,
        )
        if not plan.get("eligible"):
            raise ScopedContextGraphRebuildError(
                "Context protocol identity refresh is not eligible: "
                + str(plan.get("reason") or "unknown")
            )
        previous_context = db.get(
            ContextGraphState,
            str(plan["active_context_graph_state_id"]),
        )
        if previous_context is None or previous_context.state != "active":
            raise ScopedContextGraphRebuildError(
                "Context protocol identity refresh lost the active wrapper"
            )
        relation_state = db.get(
            ChunkRelationGraphState,
            previous_context.chunk_relation_graph_state_id,
        )
        mid_state = db.get(
            MidConceptState,
            previous_context.mid_concept_state_id,
        )
        coarse_state = db.get(
            CoarseConceptState,
            previous_context.coarse_concept_state_id,
        )
        if (
            relation_state is None
            or mid_state is None
            or coarse_state is None
            or relation_state.state != "active"
            or mid_state.state != "active"
            or coarse_state.state != "active"
        ):
            raise ScopedContextGraphRebuildError(
                "Context protocol identity refresh requires the exact active "
                "relation/RQ/mid/coarse lineage"
            )
        chunks = _active_chunks(db, knowledge_base_id)
        vector_target = context_graph._vector_runtime_target_for_kb(
            db,
            knowledge_base_id,
        )
        inherited_contextual_index_maintenance = _strict_json_copy(
            (previous_context.diagnostics_json or {}).get(
                "contextual_index_maintenance"
            )
            or {}
        )
        plan_hash = stable_hash(plan)
        metadata = {
            "context_protocol_identity_refresh_parent_state_id": str(
                previous_context.id
            ),
            "context_protocol_identity_refresh_parent_hash": str(
                previous_context.context_graph_hash
            ),
            "context_protocol_identity_refresh_plan_hash": plan_hash,
            "contextual_index_maintenance": (
                inherited_contextual_index_maintenance
            ),
        }
        with _candidate_savepoint(db):
            context_state = context_graph.write_context_graph_state(
                db,
                knowledge_base_id,
                relation_state,
                mid_state,
                coarse_state,
                chunks,
                state_scope="active",
                shadow_metadata=metadata,
                vector_runtime_target=vector_target,
            )
            db.execute(
                update(ContextGraphState)
                .where(
                    ContextGraphState.knowledge_base_id
                    == knowledge_base_id,
                    ContextGraphState.state == "active",
                    ContextGraphState.id != context_state.id,
                )
                .values(state="inactive")
            )
            from app.services.vector_shadow_lifecycle import (
                bind_active_vector_runtime_graph,
            )

            active_target = bind_active_vector_runtime_graph(
                db,
                target=vector_target,
                context_state=context_state,
            )
            intent = _stage_cache_invalidation_intent(
                db,
                knowledge_base_id=knowledge_base_id,
                context_state=context_state,
                requested_scope="context_protocol_identity_refresh",
            )
            audit = {
                "protocol_version": (
                    CONTEXT_PROTOCOL_IDENTITY_REFRESH_PROTOCOL_VERSION
                ),
                "plan_hash": plan_hash,
                "previous_context_graph_state_id": str(
                    previous_context.id
                ),
                "context_graph_state_id": str(context_state.id),
                "reused_graph_state_ids": {
                    "relation": str(relation_state.id),
                    "mid": str(mid_state.id),
                    "coarse": str(coarse_state.id),
                },
                "identity_difference_keys": list(
                    plan["identity_difference_keys"]
                ),
                "model_call_count": 0,
                "qdrant_write_count": 0,
                "provider_response_persisted": False,
                "gray_zone_rule_inputs_modified": False,
                "gray_zone_thresholds_modified": False,
                "gray_zone_rule_protocol_modified": False,
                "gray_zone_model_call_count": 0,
                "cache_invalidation_queued_after_commit": True,
                "cache_invalidation_intent_id": str(intent.id),
                "cache_invalidation_failure_recovery": (
                    "reconcile_scoped_rebuild_cache_invalidations"
                ),
                "active_vector_runtime_state_hash": (
                    active_target.runtime_state_hash
                ),
                "transaction_boundary": (
                    "caller_owned_outer_transaction_with_candidate_savepoint"
                ),
                "ingestion_resource_lock": resource_lock.diagnostics(
                    reentrant=resource_lock.operation
                    != "context_graph_rebuild"
                ),
            }
            audit["audit_hash"] = stable_hash(audit)
            context_state.diagnostics_json = {
                **dict(context_state.diagnostics_json or {}),
                "context_protocol_identity_refresh_audit": audit,
            }
            db.flush()
            admitted = context_graph.active_graph_admission_gate(
                db,
                knowledge_base_id,
                chunks=chunks,
            )
            if admitted is None or str(admitted.id) != str(
                context_state.id
            ):
                raise ScopedContextGraphRebuildError(
                    "Context protocol identity refresh candidate failed "
                    "active graph admission"
                )
        return context_state


async def _rebuild_scoped_context_graph(
    db: Session,
    knowledge_base_id: str,
    *,
    requested_scope: str,
    batch_id: str | None = None,
    require_provider_semantic_reuse: bool | None = None,
    provider_request_budget: int | None = None,
) -> ContextGraphState:
    contract = SCOPED_REBUILD_CONTRACT.get(requested_scope)
    if contract is None:
        raise ValueError(
            f"Unsupported scoped context graph rebuild: {requested_scope}"
        )
    semantic_reuse_required = (
        requested_scope in {"mid_concept", "coarse_concept"}
        if require_provider_semantic_reuse is None
        else bool(require_provider_semantic_reuse)
    )
    shared_provider_request_budget = (
        context_graph.ConceptProviderRequestBudget(
            max_requests=provider_request_budget
        )
        if provider_request_budget is not None
        else None
    )
    async with knowledge_base_ingestion_resource_lock(
        db,
        knowledge_base_id,
        operation="context_graph_rebuild",
        batch_id=batch_id,
    ) as resource_lock:
        chunks = _active_chunks(db, knowledge_base_id)
        (
            previous_context,
            previous_relation,
            previous_mid,
            previous_coarse,
        ) = _bound_active_lineage(
            db,
            knowledge_base_id,
            chunks=chunks,
        )
        before = _upstream_snapshot(
            db,
            requested_scope=requested_scope,
            context_state=previous_context,
            relation_state=previous_relation,
            mid_state=previous_mid,
        )
        vector_target = context_graph._vector_runtime_target_for_kb(
            db,
            knowledge_base_id,
        )
        inherited_contextual_index_maintenance = _strict_json_copy(
            (previous_context.diagnostics_json or {}).get(
                "contextual_index_maintenance"
            )
            or {}
        )
        metadata = {
            "scoped_rebuild_parent_context_graph_state_id": str(
                previous_context.id
            ),
            "scoped_rebuild_parent_context_graph_hash": str(
                previous_context.context_graph_hash
            ),
            "contextual_index_maintenance": (
                inherited_contextual_index_maintenance
            ),
        }
        with _candidate_savepoint(db):
            relation_state = previous_relation
            mid_state = previous_mid
            if requested_scope == "rq_membership":
                relation_state = _rebuild_rq_on_reused_relation(
                    db,
                    source_relation_state=previous_relation,
                    chunks=chunks,
                    vector_runtime_target=vector_target,
                )
            profile = context_graph.get_active_profile_record(
                db,
                knowledge_base_id,
            )
            with context_graph.use_strategy_profile(profile.profile_json):
                if requested_scope in {"rq_membership", "mid_concept"}:
                    mid_state = await context_graph.build_mid_concept_graph(
                        db,
                        knowledge_base_id,
                        relation_state,
                        batch_id=batch_id,
                        state_scope="active",
                        shadow_metadata=metadata,
                        provider_semantic_reuse_source_state=previous_mid,
                        require_provider_semantic_reuse=(
                            semantic_reuse_required
                        ),
                        provider_request_budget=(
                            shared_provider_request_budget
                        ),
                    )
                coarse_state = await context_graph.build_coarse_concept_graph(
                    db,
                    knowledge_base_id,
                    mid_state,
                    batch_id=batch_id,
                    state_scope="active",
                    shadow_metadata=metadata,
                    provider_semantic_reuse_source_state=previous_coarse,
                    require_provider_semantic_reuse=(
                        semantic_reuse_required
                    ),
                    provider_request_budget=(
                        shared_provider_request_budget
                    ),
                )
            context_state = context_graph.write_context_graph_state(
                db,
                knowledge_base_id,
                relation_state,
                mid_state,
                coarse_state,
                chunks,
                state_scope="active",
                shadow_metadata=metadata,
                vector_runtime_target=vector_target,
            )
            after = _upstream_snapshot(
                db,
                requested_scope=requested_scope,
                context_state=context_state,
                relation_state=relation_state,
                mid_state=mid_state,
            )
            hashes_unchanged = before["hashes"] == after["hashes"]
            row_counts_unchanged = (
                before["row_counts"] == after["row_counts"]
            )
            if not hashes_unchanged or not row_counts_unchanged:
                raise ScopedContextGraphRebuildError(
                    "Scoped rebuild changed a declared reused upstream layer"
                )
            _deactivate_superseded_states(
                db,
                requested_scope=requested_scope,
                relation_state=relation_state,
                mid_state=mid_state,
                coarse_state=coarse_state,
                context_state=context_state,
            )
            from app.services.vector_shadow_lifecycle import (
                bind_active_vector_runtime_graph,
            )

            active_target = bind_active_vector_runtime_graph(
                db,
                target=vector_target,
                context_state=context_state,
            )
            intent = _stage_cache_invalidation_intent(
                db,
                knowledge_base_id=knowledge_base_id,
                context_state=context_state,
                requested_scope=requested_scope,
            )
            audit = {
                "protocol_version": (
                    SCOPED_CONTEXT_GRAPH_REBUILD_PROTOCOL_VERSION
                ),
                "requested_scope": requested_scope,
                "reused_layers": list(contract["reused_layers"]),
                "rebuilt_layers": list(contract["rebuilt_layers"]),
                "upstream_business_hashes_unchanged": hashes_unchanged,
                "upstream_row_counts_unchanged": row_counts_unchanged,
                "upstream_before_snapshot_hash": before["snapshot_hash"],
                "upstream_after_snapshot_hash": after["snapshot_hash"],
                "upstream_business_hashes": dict(after["hashes"]),
                "upstream_row_counts": dict(after["row_counts"]),
                "cache_invalidation_queued_after_commit": True,
                "cache_invalidation_intent_id": str(intent.id),
                "cache_invalidation_failure_recovery": (
                    "reconcile_scoped_rebuild_cache_invalidations"
                ),
                "transaction_boundary": (
                    "caller_owned_outer_transaction_with_candidate_savepoint"
                ),
                "failure_semantics": (
                    "candidate_rows_and_pointer_switch_rollback_together"
                ),
                "ingestion_resource_lock": resource_lock.diagnostics(
                    reentrant=resource_lock.operation
                    != "context_graph_rebuild"
                ),
                "active_vector_runtime_state_hash": (
                    active_target.runtime_state_hash
                ),
                "gray_zone_rule_inputs_modified": False,
                "gray_zone_thresholds_modified": False,
                "gray_zone_rule_protocol_modified": False,
                "gray_zone_model_call_count": 0,
                "provider_semantic_reuse_protocol_version": (
                    context_graph.CONCEPT_DEFINITION_SEMANTIC_REUSE_PROTOCOL_VERSION
                ),
                "mid_provider_semantic_reuse_hit_count": int(
                    (mid_state.stats_json or {}).get(
                        "provider_semantic_reuse_hit_count"
                    )
                    or 0
                ) if requested_scope in {"rq_membership", "mid_concept"} else 0,
                "mid_provider_semantic_reuse_miss_count": int(
                    (mid_state.stats_json or {}).get(
                        "provider_semantic_reuse_miss_count"
                    )
                    or 0
                ) if requested_scope in {"rq_membership", "mid_concept"} else 0,
                "mid_provider_request_count": int(
                    (mid_state.stats_json or {}).get(
                        "provider_request_count"
                    )
                    or 0
                ) if requested_scope in {"rq_membership", "mid_concept"} else 0,
                "coarse_provider_semantic_reuse_hit_count": int(
                    (coarse_state.stats_json or {}).get(
                        "provider_semantic_reuse_hit_count"
                    )
                    or 0
                ),
                "coarse_provider_semantic_reuse_miss_count": int(
                    (coarse_state.stats_json or {}).get(
                        "provider_semantic_reuse_miss_count"
                    )
                    or 0
                ),
                "coarse_provider_request_count": int(
                    (coarse_state.stats_json or {}).get(
                        "provider_request_count"
                    )
                    or 0
                ),
                "concept_i18n_enabled": bool(
                    (mid_state.stats_json or {}).get("concept_i18n_enabled")
                    or (coarse_state.stats_json or {}).get(
                        "concept_i18n_enabled"
                    )
                ),
                "provider_semantic_reuse_required": bool(
                    semantic_reuse_required
                ),
                "provider_request_budget": (
                    shared_provider_request_budget.diagnostics()
                    if shared_provider_request_budget is not None
                    else None
                ),
            }
            audit["provider_request_count"] = int(
                audit["mid_provider_request_count"]
                + audit["coarse_provider_request_count"]
            )
            audit["audit_hash"] = stable_hash(audit)
            context_state.diagnostics_json = {
                **dict(context_state.diagnostics_json or {}),
                "scoped_rebuild_audit": audit,
            }
            db.flush()
            admitted = context_graph.active_graph_admission_gate(
                db,
                knowledge_base_id,
                chunks=chunks,
            )
            if admitted is None or str(admitted.id) != str(context_state.id):
                raise ScopedContextGraphRebuildError(
                    "Scoped rebuild candidate failed active graph admission"
                )
        return context_state


async def rebuild_rq_membership_graph(
    db: Session,
    knowledge_base_id: str,
    *,
    batch_id: str | None = None,
    require_provider_semantic_reuse: bool = False,
    provider_request_budget: int | None = None,
) -> ContextGraphState:
    return await _rebuild_scoped_context_graph(
        db,
        knowledge_base_id,
        requested_scope="rq_membership",
        batch_id=batch_id,
        require_provider_semantic_reuse=(
            require_provider_semantic_reuse
        ),
        provider_request_budget=provider_request_budget,
    )


async def rebuild_mid_concept_graph(
    db: Session,
    knowledge_base_id: str,
    *,
    batch_id: str | None = None,
) -> ContextGraphState:
    return await _rebuild_scoped_context_graph(
        db,
        knowledge_base_id,
        requested_scope="mid_concept",
        batch_id=batch_id,
    )


async def rebuild_coarse_concept_graph(
    db: Session,
    knowledge_base_id: str,
    *,
    batch_id: str | None = None,
) -> ContextGraphState:
    return await _rebuild_scoped_context_graph(
        db,
        knowledge_base_id,
        requested_scope="coarse_concept",
        batch_id=batch_id,
    )
