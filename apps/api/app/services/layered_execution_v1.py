"""Frozen three-layer Dense/RQ/BM25 entry, descent and distance traversal."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable, Sequence

import numpy as np
from sqlalchemy import select

from app.intent_contracts import AcceptedPlan, CapabilityManifest, ChannelWeights
from app.models import (
    Chunk,
    ChunkRelationEdge,
    CoarseConcept,
    CoarseConceptEdge,
    CoarseConceptMembership,
    ContextGraphState,
    GraphRetrievalStep,
    LexicalIndexState,
    MidConcept,
    MidConceptEdge,
    RQPrefix,
    RetrievalTrace,
)
from app.retrieval_control_contracts import control_hash
from app.services.chunking import stable_hash
from app.services.embeddings import EmbeddingProvider
from app.services.entry_ranking import EntryScore, FusedEntry, fuse_entry_candidates
from app.services.lexical_storage import search_lexical_index_many
from app.services.qa_performance import qa_stage, qa_sync_timed
from app.services.retrieval_adjacency import CompleteChunkAdjacency
from app.services.retrieval_corpus import RetrievalCorpus


PROTOCOL = "intent_execution_retrieval_v1"
TRAVERSAL_PROTOCOL = "layered_distance_traversal_v2"
GRAY_ZONE_PROTOCOL = "deterministic_support_progress_v2"


@dataclass(frozen=True)
class TraversedNode:
    node_id: str
    root_node_id: str
    distance: float
    depth: int
    node_path: tuple[str, ...]
    edge_path: tuple[str, ...]
    support_ids: tuple[str, ...]
    entry_score: float
    entry_channels: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LayeredExecutionResult:
    results: list[dict[str, Any]]
    trace: RetrievalTrace
    audit: dict[str, Any]
    scope_target_chunk_ids: tuple[str, ...] = ()
    scope_execution_audit: dict[str, Any] | None = None
    cache_components: dict[str, Any] | None = None
    cache_payload: dict[str, Any] | None = None
    cache_audit: dict[str, Any] | None = None


def _intent_cache_components(
    db,
    *,
    plan: AcceptedPlan,
    filters,
    top_k: int | None,
    capabilities,
    context_state,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    from app.core.config import get_settings
    from app.services import agent_graph
    from app.services.cache_manager import INTENT_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION
    from app.services.entry_ranking import FUSION_PROTOCOL, RQ_PROTOCOL
    from app.services.strategy_profiles import active_profile_hash

    lexical_enabled = any(
        plan.strategy.layer_weights.for_layer(layer).bm25 > 0
        for layer in plan.strategy.layer_weights.enabled_layers()
    )
    diagnostics = dict(context_state.diagnostics_json or {})
    vector_identity = diagnostics.get("canonical_vector_identity")
    if not isinstance(vector_identity, dict) or not vector_identity:
        raise ValueError("intent_retrieval_cache_vector_identity_missing")
    runtime_hash = agent_graph.runtime_settings_state_hash()
    profile_hash = control_hash(
        {
            "active_profile_hash": active_profile_hash(
                db, plan.task.knowledge_base_id
            )
        }
    )
    graph_snapshot = {
        "context_graph_state_id": context_state.id,
        "chunk_scope_hash": context_state.chunk_scope_hash,
        "structure_graph_hash": context_state.structure_graph_hash,
        "chunk_relation_graph_hash": context_state.chunk_relation_graph_hash,
        "rq_membership_hash": context_state.rq_membership_hash,
        "mid_concept_hash": context_state.mid_concept_hash,
        "coarse_concept_hash": context_state.coarse_concept_hash,
        "context_graph_hash": context_state.context_graph_hash,
        "diagnostics_hash": control_hash(diagnostics),
    }
    traversal_identity = {
        "protocol_version": TRAVERSAL_PROTOCOL,
        "gray_zone_protocol_version": GRAY_ZONE_PROTOCOL,
        "green": float(envelope["path_distance_green_threshold"]),
        "gray": float(envelope["path_distance_gray_threshold"]),
        "hard": float(envelope["path_distance_hard_threshold"]),
        "max_depth": plan.effective_budget.max_depth,
    }
    return {
        "cache_key_protocol_version": INTENT_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION,
        "knowledge_base_id": plan.task.knowledge_base_id,
        "conversation_identity_hash": plan.task.conversation_identity_hash,
        "conversation_scope_hash": plan.task.conversation_scope_hash,
        "question_hash": control_hash(plan.task.question),
        "filters_hash": control_hash(filters.model_dump(mode="json")),
        "task_hash": plan.task.identity,
        "intent_hash": plan.intent.identity,
        "strategy_hash": plan.strategy.identity,
        "accepted_plan_hash": plan.identity,
        "capability_hash": plan.capability_hash,
        "effective_budget_hash": plan.effective_budget.identity,
        "entry_layer": plan.strategy.entry_layer,
        "semantic_query_hash": control_hash(plan.strategy.semantic_query),
        "lexical_groups_hash": control_hash(
            [item.model_dump(mode="json") for item in plan.strategy.lexical_groups]
        ),
        "generate_lexical": plan.strategy.generate_lexical,
        "hybrid": plan.strategy.hybrid,
        "layer_weights_hash": plan.strategy.layer_weights.identity,
        "graph_state_id": context_state.id,
        "graph_identity": capabilities.graph_identity,
        "graph_snapshot_hash": control_hash(graph_snapshot),
        "source_manifest_hash": control_hash(
            {
                "chunk_scope_hash": context_state.chunk_scope_hash,
                "structure_graph_hash": context_state.structure_graph_hash,
                "context_graph_hash": context_state.context_graph_hash,
            }
        ),
        "embedding_identity": vector_identity,
        "lexical_identity": capabilities.lexical_identity if lexical_enabled else None,
        "ranking_protocol_hash": control_hash(
            {
                "fusion": FUSION_PROTOCOL,
                "rq_entry": RQ_PROTOCOL,
                "projection": "primary_support_parent_projection_v1",
                "candidate_nomination": "independent_channel_nomination_v1",
                "source_scope_filter": "structure_scope_resolver_v1",
            }
        ),
        "traversal_protocol_hash": control_hash(traversal_identity),
        "runtime_settings_hash": runtime_hash,
        "profile_hash": profile_hash,
        "result_top_k": min(
            50,
            int(top_k or get_settings().retrieval_result_top_k_default),
        ),
    }


def _cache_audit(
    *,
    status: str,
    key_digest: str,
    ttl_seconds_remaining: int | None,
    reason: str,
    source_trace_id: str | None = None,
    deletion_attempted: bool = False,
    deleted: bool = False,
    write_after_commit: bool = False,
) -> dict[str, Any]:
    return {
        "protocol_version": "intent_execution_retrieval_cache_v1",
        "status": status,
        "cache_hit": status == "hit",
        "cache_miss": status != "hit",
        "cache_key": key_digest,
        "redis_key_digest": key_digest,
        "ttl_seconds": 300,
        "ttl_seconds_remaining": ttl_seconds_remaining,
        "reason": reason,
        "deletion_attempted": deletion_attempted,
        "deleted": deleted,
        "source_retrieval_trace_id": source_trace_id,
        "postgresql_replay_required": True,
        "redis_payload_used_as_evidence": False,
        "query_embedding_model_call_count": 0 if status == "hit" else None,
        "traversal_execution_count": 0 if status == "hit" else None,
        "retrieval_fact_insert_count": 1 if status == "hit" else None,
        "gray_zone_input_modified": False,
        "gray_zone_model_call_count": 0,
        "write_scheduled_after_commit": write_after_commit,
    }


def _clone_cached_trace(db, *, source: RetrievalTrace, cache_audit: dict[str, Any]) -> RetrievalTrace:
    fields = (
        "knowledge_base_id",
        "query",
        "filters_json",
        "retrieval_mode",
        "chunk_scope_hash",
        "structure_graph_hash",
        "chunk_relation_graph_hash",
        "rq_membership_hash",
        "mid_concept_hash",
        "coarse_concept_hash",
        "runtime_settings_hash",
        "agent_operating_envelope_hash",
        "policy_state_hash",
        "prompt_protocol_hash",
        "result_chunk_ids_json",
        "concept_path_json",
        "scores_json",
        "query_facets_json",
        "entry_nodes_json",
        "frontier_json",
        "stage_queues_json",
        "candidate_pools_json",
        "topk_selection_json",
        "path_labels_json",
        "convergence_json",
        "edge_distance_protocol_hash",
        "edge_projection_protocol_hash",
        "traversal_protocol_hash",
        "conversation_state_scope_hash",
        "diagnostics_json",
    )
    values = {field: getattr(source, field) for field in fields}
    values["scores_json"] = {
        **dict(source.scores_json or {}),
        "retrieval_cache": cache_audit,
    }
    values["diagnostics_json"] = {
        **dict(source.diagnostics_json or {}),
        "retrieval_cache": cache_audit,
        "cache_replay_source_trace_id": source.id,
    }
    trace = RetrievalTrace(**values)
    db.add(trace)
    db.flush()
    rows = list(
        db.scalars(
            select(GraphRetrievalStep)
            .where(GraphRetrievalStep.retrieval_trace_id == source.id)
            .order_by(GraphRetrievalStep.step_index, GraphRetrievalStep.id)
        )
    )
    step_fields = (
        "knowledge_base_id",
        "step_index",
        "layer",
        "action",
        "action_type",
        "parent_layer",
        "parent_node_id",
        "input_json",
        "output_json",
        "score_json",
        "popped_frontier_state_json",
        "expanded_edge_ids_json",
        "candidate_pool_ids_json",
        "selected_topk_ids_json",
        "dominance_pruned_count",
        "cycle_distance_reward",
        "gray_zone_path_decisions_json",
        "per_parent_budget_status_json",
        "stop_reason",
        "diagnostics_json",
    )
    for row in rows:
        values = {field: getattr(row, field) for field in step_fields}
        values["retrieval_trace_id"] = trace.id
        values["diagnostics_json"] = {
            **dict(row.diagnostics_json or {}),
            "cache_replay_source_step_id": row.id,
        }
        db.add(GraphRetrievalStep(**values))
    db.flush()
    return trace


def _enabled_channels(weights: ChannelWeights) -> tuple[str, ...]:
    values = weights.effective()
    return tuple(name for name in ("dense", "rq", "bm25") if values[name] > 0)


def _entry(
    *,
    candidate_id: str,
    layer: str,
    score: float,
    witnesses: Iterable[str] = (),
) -> EntryScore:
    return EntryScore(
        candidate_id=candidate_id,
        business_key=f"{layer}:{candidate_id}",
        score=float(score),
        witness_ids=tuple(dict.fromkeys(str(item) for item in witnesses if str(item))),
    )


def _bounded_channel(rows: Iterable[EntryScore], limit: int) -> tuple[EntryScore, ...]:
    return tuple(
        sorted(rows, key=lambda item: (-item.score, item.business_key))[:limit]
    )


def _prefix_score(query: Sequence[float], prefix: RQPrefix) -> float:
    centroid = prefix.centroid_json or []
    if len(centroid) != len(query) or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in centroid
    ):
        raise ValueError("entry_rq_prefix_vector_invalid")
    try:
        score = -math.fsum((float(left) - float(right)) ** 2 for left, right in zip(query, centroid))
    except (OverflowError, ValueError):
        raise ValueError("entry_rq_numeric_overflow") from None
    if not math.isfinite(score):
        raise ValueError("entry_rq_numeric_overflow")
    return score


def _edge_support_ids(edge: Any, layer: str) -> tuple[str, ...]:
    if layer == "chunk":
        payload = dict(edge.support_json or {})
        values = [
            *payload.get("support_chunk_ids", []),
            *payload.get("support_relation_edge_ids", []),
            edge.source_chunk_id,
            edge.target_chunk_id,
        ]
    elif layer == "mid":
        values = [
            *(edge.support_chunk_ids_json or []),
            *(edge.support_chunk_edge_ids_json or []),
            *(edge.support_rq_prefix_ids_json or []),
            *(edge.support_relation_edge_ids_json or []),
        ]
    else:
        values = [
            *(edge.support_chunk_ids_json or []),
            *(edge.support_chunk_edge_ids_json or []),
            *(edge.support_rq_prefix_ids_json or []),
            *(edge.support_mid_concept_ids_json or []),
            *(edge.support_mid_edge_ids_json or []),
        ]
    return tuple(sorted({str(item) for item in values if str(item)}))


def _edge_endpoints(edge: Any, layer: str) -> tuple[str, str]:
    if layer == "chunk":
        return str(edge.source_chunk_id), str(edge.target_chunk_id)
    return str(edge.source_concept_id), str(edge.target_concept_id)


def _gray_decision(
    *,
    layer: str,
    distance: float,
    support_ids: tuple[str, ...],
    green: float,
    gray: float,
    hard: float,
) -> dict[str, Any]:
    if distance > hard:
        zone, decision, rule = "hard_stop", "stop", "distance_above_hard_threshold"
    elif distance > gray:
        zone, decision, rule = "red", "stop", "red_paths_do_not_expand"
    elif distance > green and not support_ids:
        zone, decision, rule = "gray", "stop", "gray_path_has_no_new_support"
    elif distance > green:
        zone, decision, rule = "gray", "continue", "gray_path_has_grounded_support"
    else:
        zone, decision, rule = "green", "continue", "green_supported_path"
    inputs = {
        "layer": layer,
        "distance": round(distance, 15),
        "zone": zone,
        "support_ids": list(support_ids),
        "query_anchor_preserved": True,
        "model_call_count": 0,
    }
    return {
        "protocol_version": GRAY_ZONE_PROTOCOL,
        "input_hash": control_hash(inputs),
        "inputs": inputs,
        "matched_rule": rule,
        "decision": decision,
        "model_call_count": 0,
    }


@qa_sync_timed("graph_traversal")
def _traverse(
    *,
    layer: str,
    roots: Sequence[FusedEntry],
    edges: Sequence[Any] = (),
    adjacency_reader: CompleteChunkAdjacency | None = None,
    limit: int,
    max_depth: int,
    green: float,
    gray: float,
    hard: float,
) -> tuple[tuple[TraversedNode, ...], dict[str, Any]]:
    adjacency: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for edge in edges:
        left, right = _edge_endpoints(edge, layer)
        distance = float(edge.distance)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("traversal_edge_distance_invalid")
        if left == right:
            continue
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))
    root_by_id = {item.candidate_id: item for item in roots}
    queue: list[tuple[float, int, tuple[str, ...], TraversedNode]] = []
    for root in roots:
        state = TraversedNode(
            node_id=root.candidate_id,
            root_node_id=root.candidate_id,
            distance=0.0,
            depth=0,
            node_path=(root.candidate_id,),
            edge_path=(),
            support_ids=(),
            entry_score=root.score,
            entry_channels=tuple(item.model_dump(mode="json") for item in root.channels),
        )
        heapq.heappush(queue, (0.0, 0, state.node_path, state))
    accepted: list[TraversedNode] = []
    best: dict[tuple[str, str], tuple[float, int, tuple[str, ...]]] = {}
    decisions: list[dict[str, Any]] = []
    pruned = 0
    while queue and len(accepted) < limit:
        _distance, _depth, _path, state = heapq.heappop(queue)
        key = (state.root_node_id, state.node_id)
        priority = (state.distance, state.depth, state.node_path)
        if key in best and best[key] <= priority:
            pruned += 1
            continue
        best[key] = priority
        accepted.append(state)
        if (
            state.depth >= max_depth
            or state.distance > gray
            or (state.distance > green and not state.support_ids)
        ):
            continue
        if adjacency_reader is None:
            incident = adjacency.get(state.node_id, ())
        else:
            incident = []
            for edge in adjacency_reader.get(state.node_id, ()):
                left, right = _edge_endpoints(edge, layer)
                if left == right:
                    continue
                neighbor = right if left == state.node_id else left
                distance = float(edge.distance)
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("traversal_edge_distance_invalid")
                incident.append((neighbor, edge))
        for neighbor, edge in sorted(
            incident,
            key=lambda item: (float(item[1].distance), str(item[1].id), item[0]),
        ):
            if neighbor in state.node_path:
                pruned += 1
                continue
            support = _edge_support_ids(edge, layer)
            next_distance = state.distance + float(edge.distance)
            decision = _gray_decision(
                layer=layer,
                distance=next_distance,
                support_ids=support,
                green=green,
                gray=gray,
                hard=hard,
            )
            decisions.append(
                {
                    **decision,
                    "edge_id": str(edge.id),
                    "source_node_id": state.node_id,
                    "target_node_id": neighbor,
                }
            )
            if decision["decision"] == "stop" and next_distance > hard:
                continue
            next_state = TraversedNode(
                node_id=neighbor,
                root_node_id=state.root_node_id,
                distance=next_distance,
                depth=state.depth + 1,
                node_path=(*state.node_path, neighbor),
                edge_path=(*state.edge_path, str(edge.id)),
                support_ids=tuple(sorted(set(state.support_ids) | set(support))),
                entry_score=root_by_id[state.root_node_id].score,
                entry_channels=state.entry_channels,
            )
            heapq.heappush(
                queue,
                (next_state.distance, next_state.depth, next_state.node_path, next_state),
            )
    return tuple(accepted), {
        "layer": layer,
        "root_ids": [item.candidate_id for item in roots],
        "visited_count": len(accepted),
        "frontier_remaining": len(queue),
        "dominance_or_cycle_pruned_count": pruned,
        "gray_zone_decisions": decisions,
        "stop_reason": "layer_budget_hit" if queue else "frontier_exhausted",
        "adjacency_query_count": (
            adjacency_reader.query_count if adjacency_reader is not None else 0
        ),
        "adjacency_rows_read": (
            adjacency_reader.rows_read if adjacency_reader is not None else len(edges)
        ),
        "adjacency_complete_for_loaded_nodes": adjacency_reader is not None,
    }


def _unique_traversed_nodes(
    traversed: Sequence[TraversedNode],
) -> tuple[TraversedNode, ...]:
    """Keep the first, highest-priority traversed path for each graph node."""

    selected: list[TraversedNode] = []
    seen: set[str] = set()
    for item in traversed:
        if item.node_id in seen:
            continue
        seen.add(item.node_id)
        selected.append(item)
    return tuple(selected)


def _merge_parent_entries(
    pools: list[tuple[str, tuple[FusedEntry, ...]]],
    *,
    broad: bool,
    limit: int,
) -> tuple[tuple[FusedEntry, ...], dict[str, list[str]]]:
    parents: dict[str, list[str]] = defaultdict(list)
    selected: list[FusedEntry] = []
    by_id: dict[str, FusedEntry] = {}
    if broad:
        max_length = max((len(rows) for _parent, rows in pools), default=0)
        ordered = [
            (parent, rows[index])
            for index in range(max_length)
            for parent, rows in pools
            if index < len(rows)
        ]
    else:
        ordered = sorted(
            ((parent, item) for parent, rows in pools for item in rows),
            key=lambda pair: (-pair[1].score, pair[1].business_key, pair[0]),
        )
    for parent, item in ordered:
        parents[item.candidate_id].append(parent)
        current = by_id.get(item.candidate_id)
        if current is None:
            by_id[item.candidate_id] = item
            selected.append(item)
        elif item.score > current.score:
            replacement = selected.index(current)
            selected[replacement] = item
            by_id[item.candidate_id] = item
        if len(selected) >= limit:
            break
    return tuple(selected), {key: list(dict.fromkeys(value)) for key, value in parents.items()}


@qa_sync_timed("entry_fusion")
def _fuse_scope(
    channel_rows: dict[str, dict[str, EntryScore]],
    *,
    candidate_ids: Iterable[str],
    weights: ChannelWeights,
    limit: int,
) -> tuple[FusedEntry, ...]:
    scope = set(candidate_ids)
    lists = {
        channel: tuple(row for key, row in channel_rows[channel].items() if key in scope)
        for channel in _enabled_channels(weights)
    }
    return fuse_entry_candidates(lists, weights, limit=limit)


def _concept_channel_rows(
    *,
    layer: str,
    concepts: Sequence[Any],
    dense_scores: dict[str, float],
    rq_prefixes: dict[str, RQPrefix],
    query_vector: Sequence[float],
    bm25_scores: dict[str, float],
    dense_limit: int,
    rq_limit: int,
    bm25_limit: int,
    enabled_channels: tuple[str, ...],
) -> tuple[dict[str, dict[str, EntryScore]], dict[str, int]]:
    from app.services.context_graph import semantic_concept_entry_candidates

    with qa_stage("dense_entry", item_count=len(concepts)):
        dense_cards = (
            semantic_concept_entry_candidates(concepts, dense_scores, layer=layer)
            if "dense" in enabled_channels
            else {}
        )
        dense = _bounded_channel(
            (
                _entry(
                    candidate_id=str(concept.id),
                    layer=layer,
                    score=dense_cards[str(concept.id)]["semantic_score"],
                    witnesses=dense_cards[str(concept.id)]["semantic_support_chunk_ids"],
                )
                for concept in concepts
                if dense_cards[str(concept.id)]["semantic_candidate"]
            ),
            dense_limit,
        ) if "dense" in enabled_channels else ()
    rq_rows: list[EntryScore] = []
    bm25_rows: list[EntryScore] = []
    unprojected = set(bm25_scores)
    if "rq" in enabled_channels:
        with qa_stage("rq_entry", item_count=len(concepts)):
            for concept in concepts:
                prefix_id = (
                    concept.support_rq_l3_prefix_id
                    if layer == "mid"
                    else concept.support_rq_l2_prefix_id
                )
                if prefix_id is None or str(prefix_id) not in rq_prefixes:
                    raise ValueError("entry_rq_concept_mapping_invalid")
                prefix = rq_prefixes[str(prefix_id)]
                rq_rows.append(
                    _entry(
                        candidate_id=str(concept.id),
                        layer=layer,
                        score=_prefix_score(query_vector, prefix),
                        witnesses=(str(prefix.id),),
                    )
                )
    if "bm25" in enabled_channels:
        with qa_stage("bm25_entry", item_count=len(concepts)):
            for concept in concepts:
                support = {str(item) for item in (concept.support_chunk_ids_json or [])}
                matched = [(chunk_id, bm25_scores[chunk_id]) for chunk_id in support if chunk_id in bm25_scores]
                if matched:
                    winner = min(matched, key=lambda item: (-item[1], item[0]))
                    bm25_rows.append(
                        _entry(
                            candidate_id=str(concept.id),
                            layer=layer,
                            score=winner[1],
                            witnesses=(winner[0],),
                        )
                    )
                    unprojected -= support
    rows = {
        "dense": {item.candidate_id: item for item in dense},
        "rq": {item.candidate_id: item for item in _bounded_channel(rq_rows, rq_limit)},
        "bm25": {item.candidate_id: item for item in _bounded_channel(bm25_rows, bm25_limit)},
    }
    return rows, {"unprojected_bm25_hit_count": len(unprojected)}


def _chunk_channel_rows(
    *,
    chunk_ids: Sequence[str],
    chunk_rq_paths: dict[str, tuple[int, ...]],
    prefix_by_path: dict[tuple[int, ...], RQPrefix],
    query_vector: Sequence[float],
    dense_scores: dict[str, float],
    bm25_scores: dict[str, float],
    bm25_witnesses: dict[str, tuple[str, ...]],
    dense_limit: int,
    rq_limit: int,
    bm25_limit: int,
    enabled_channels: tuple[str, ...],
) -> dict[str, dict[str, EntryScore]]:
    with qa_stage("dense_entry", item_count=len(chunk_ids)):
        dense = _bounded_channel(
            (
                _entry(candidate_id=chunk_id, layer="chunk", score=dense_scores[chunk_id], witnesses=(chunk_id,))
                for chunk_id in chunk_ids
                if dense_scores.get(chunk_id, 0.0) > 0
            ),
            dense_limit,
        ) if "dense" in enabled_channels else ()
    rq_rows = []
    if "rq" in enabled_channels:
        with qa_stage("rq_entry", item_count=len(chunk_ids)):
            for chunk_id in chunk_ids:
                path = chunk_rq_paths.get(chunk_id)
                if path is None or len(path) != 3 or path not in prefix_by_path:
                    raise ValueError("entry_rq_chunk_primary_mapping_invalid")
                prefix = prefix_by_path[path]
                rq_rows.append(
                    _entry(
                        candidate_id=chunk_id,
                        layer="chunk",
                        score=_prefix_score(query_vector, prefix),
                        witnesses=(str(prefix.id),),
                    )
                )
    with qa_stage("bm25_entry", item_count=len(bm25_scores)):
        chunk_scope = set(chunk_ids)
        lexical = _bounded_channel(
            (
                _entry(
                    candidate_id=chunk_id,
                    layer="chunk",
                    score=score,
                    witnesses=bm25_witnesses.get(chunk_id, (chunk_id,)),
                )
                for chunk_id, score in bm25_scores.items()
                if chunk_id in chunk_scope
            ),
            bm25_limit,
        ) if "bm25" in enabled_channels else ()
    return {
        "dense": {item.candidate_id: item for item in dense},
        "rq": {item.candidate_id: item for item in _bounded_channel(rq_rows, rq_limit)},
        "bm25": {item.candidate_id: item for item in lexical},
    }


def _concept_children(
    *,
    coarse_ids: Sequence[str],
    memberships: Sequence[CoarseConceptMembership],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {item: [] for item in coarse_ids}
    for row in memberships:
        parent = str(row.coarse_concept_id)
        if parent in result:
            result[parent].append(str(row.mid_concept_id))
    return {key: tuple(sorted(set(value))) for key, value in result.items()}


def _chunk_children(mid_concepts: dict[str, MidConcept], mid_ids: Sequence[str], eligible: set[str]) -> dict[str, tuple[str, ...]]:
    return {
        mid_id: tuple(
            sorted(
                eligible
                & {str(item) for item in (mid_concepts[mid_id].support_chunk_ids_json or [])}
            )
        )
        for mid_id in mid_ids
    }


def _replay_cached_execution(
    db,
    *,
    plan: AcceptedPlan,
    filters,
    context_state,
    cache_components: dict[str, Any],
    cache_read,
    cache_manager,
) -> tuple[LayeredExecutionResult | None, dict[str, Any]]:
    from app.services.context_graph import passes_filters, search_payload_for_chunk

    base_audit = _cache_audit(
        status=cache_read.status,
        key_digest=cache_read.key_digest,
        ttl_seconds_remaining=cache_read.ttl_seconds_remaining,
        reason=(cache_read.poison_reason or cache_read.status),
        deletion_attempted=cache_read.deletion_attempted,
        deleted=cache_read.deleted,
    )
    if cache_read.status != "hit" or not isinstance(cache_read.payload, dict):
        return None, base_audit
    payload = cache_read.payload
    expected_fields = {
        "protocol_version",
        "cache_identity_hash",
        "source_retrieval_trace_id",
        "result_chunk_ids",
        "result_identity_hash",
    }
    source = db.get(RetrievalTrace, payload.get("source_retrieval_trace_id"))
    result_ids = [str(item) for item in payload.get("result_chunk_ids") or []]
    valid = (
        set(payload) == expected_fields
        and payload.get("protocol_version")
        == "intent_execution_retrieval_cache_payload_v1"
        and payload.get("cache_identity_hash") == cache_read.key_digest
        and payload.get("result_identity_hash")
        == control_hash(result_ids)
        and len(result_ids) == len(set(result_ids))
        and source is not None
        and source.knowledge_base_id == plan.task.knowledge_base_id
        and source.retrieval_mode == PROTOCOL
        and source.result_chunk_ids_json == result_ids
        and source.prompt_protocol_hash == plan.identity
        and (source.diagnostics_json or {}).get(
            "intent_retrieval_cache_identity_hash"
        )
        == cache_read.key_digest
        and source.chunk_scope_hash == context_state.chunk_scope_hash
        and source.structure_graph_hash == context_state.structure_graph_hash
        and source.chunk_relation_graph_hash
        == context_state.chunk_relation_graph_hash
        and source.rq_membership_hash == context_state.rq_membership_hash
        and source.mid_concept_hash == context_state.mid_concept_hash
        and source.coarse_concept_hash == context_state.coarse_concept_hash
    )
    chunks = {
        str(item.id): item
        for item in db.scalars(select(Chunk).where(Chunk.id.in_(result_ids)))
    } if valid and result_ids else {}
    valid = valid and set(chunks) == set(result_ids) and all(
        item.knowledge_base_id == plan.task.knowledge_base_id
        and item.state == "active"
        and passes_filters(db, item, filters)
        for item in chunks.values()
    )
    if not valid:
        deleted = cache_manager.delete_intent_retrieval(
            plan.task.knowledge_base_id,
            cache_components=cache_components,
        )
        return None, _cache_audit(
            status="poison",
            key_digest=cache_read.key_digest,
            ttl_seconds_remaining=cache_read.ttl_seconds_remaining,
            reason="postgresql_replay_rejected",
            source_trace_id=source.id if source is not None else None,
            deletion_attempted=True,
            deleted=deleted,
        )
    hit_audit = _cache_audit(
        status="hit",
        key_digest=cache_read.key_digest,
        ttl_seconds_remaining=cache_read.ttl_seconds_remaining,
        reason="postgresql_trace_replay_passed",
        source_trace_id=source.id,
    )
    trace = _clone_cached_trace(db, source=source, cache_audit=hit_audit)
    labels = {
        str(item.get("chunk_id") or item.get("node_id") or ""): dict(item)
        for item in trace.path_labels_json or []
    }
    results: list[dict[str, Any]] = []
    for chunk_id in result_ids:
        label = labels.get(chunk_id)
        if label is None:
            raise ValueError("intent_retrieval_cache_path_label_missing")
        distance = float(label.get("distance_so_far") or 0.0)
        traversal = {
            "layer": "chunk",
            "node_id": chunk_id,
            "root_node_id": label.get("root_node_id") or chunk_id,
            "path": list(label.get("path") or [chunk_id]),
            "path_edge_ids": list(label.get("path_edge_ids") or []),
            "path_edge_types": list(label.get("path_edge_types") or []),
            "distance_so_far": distance,
            "reward_so_far": 0,
            "cycle_reward_so_far": 0,
            "covered_facets": list(label.get("covered_facets") or []),
            "evidence_roles": list(label.get("evidence_roles") or ["retrieval_hit"]),
            "support_refs": dict(label.get("support_refs") or {}),
            "entry_parent_refs": list(label.get("entry_parent_refs") or []),
            "why_selected": "accepted_by_layered_distance_traversal_v2",
        }
        results.append(
            search_payload_for_chunk(
                db,
                chunks[chunk_id],
                1.0 / (1.0 + distance),
                {"path_distance": distance},
                {
                    "retrieval_trace_id": trace.id,
                    "retrieval_protocol_version": PROTOCOL,
                    "root_node_id": traversal["root_node_id"],
                    "entry_score": (label.get("support_refs") or {}).get(
                        "entry_strength"
                    ),
                    "entry_channels": list(label.get("entry_channels") or []),
                    "path_edge_ids": traversal["path_edge_ids"],
                    "support_ids": list(
                        (label.get("support_refs") or {}).get("support_ids") or []
                    ),
                    "parent_node_ids": [
                        str(item.get("parent_node_id"))
                        for item in traversal["entry_parent_refs"]
                        if item.get("parent_node_id")
                    ],
                    "traversal": traversal,
                    "why_selected": "layered_distance_traversal_v2_cache_replay",
                },
            )
        )
    scope_audit = (trace.diagnostics_json or {}).get("source_scope_execution")
    target_ids = tuple(
        str(item)
        for item in (
            (scope_audit or {}).get("target_plan") or {}
        ).get("packing_target_chunk_ids")
        or (
            (scope_audit or {}).get("target_plan") or {}
        ).get("target_chunk_ids")
        or []
    )
    audit = {
        "protocol_version": PROTOCOL,
        "retrieval_trace_id": trace.id,
        "accepted_plan_hash": plan.identity,
        "entry_layer": plan.strategy.entry_layer,
        "result_count": len(results),
        "post_retrieval_model_call_count": 0,
        "reward_call_count": 0,
        "source_scope_execution": scope_audit,
        "retrieval_cache": hit_audit,
    }
    return LayeredExecutionResult(
        results=results,
        trace=trace,
        audit=audit,
        scope_target_chunk_ids=target_ids,
        scope_execution_audit=scope_audit,
        cache_components=cache_components,
        cache_audit=hit_audit,
    ), hit_audit


def publish_intent_retrieval_cache(execution: LayeredExecutionResult) -> bool:
    """Publish a trace pointer only after its caller commits PostgreSQL."""

    if execution.cache_components is None or execution.cache_payload is None:
        return False
    from app.services.cache_manager import get_cache_manager

    manager = get_cache_manager()
    if not manager.shared_cache_available:
        return False
    manager.set_intent_retrieval(
        execution.trace.knowledge_base_id,
        execution.cache_payload,
        ttl=300,
        cache_components=execution.cache_components,
    )
    return True


async def execute_layered_retrieval(
    db,
    *,
    plan: AcceptedPlan,
    filters,
    top_k: int | None,
    capabilities: CapabilityManifest | None = None,
    context_state: ContextGraphState | None = None,
) -> LayeredExecutionResult:
    """Execute one accepted plan without result-driven strategy changes."""

    from app.core.config import get_settings
    from app.services import agent_graph
    from app.services.context_graph import (
        EDGE_TYPE_CALIBRATION_EDGE_TYPES,
        agent_operating_envelope,
        gray_zone_runtime_settings_hash,
        search_payload_for_chunk,
    )
    from app.services.intent_planning import retrieval_capability_snapshot

    if capabilities is None or context_state is None:
        current_capabilities, context_state = retrieval_capability_snapshot(
            db,
            plan.task.knowledge_base_id,
        )
    else:
        current_capabilities = capabilities
        live_context = db.scalar(
            select(ContextGraphState).where(
                ContextGraphState.knowledge_base_id == plan.task.knowledge_base_id,
                ContextGraphState.state == "active",
            )
        )
        if (
            live_context is None
            or live_context.id != context_state.id
            or live_context.context_graph_hash != context_state.context_graph_hash
        ):
            raise ValueError("strategy_graph_identity_changed")
        live_lexical = db.scalar(
            select(LexicalIndexState).where(
                LexicalIndexState.knowledge_base_id == plan.task.knowledge_base_id,
                LexicalIndexState.state == "active",
            )
        )
        live_lexical_identity = (
            live_lexical.state_hash if live_lexical is not None else None
        )
        if live_lexical_identity != current_capabilities.lexical_identity:
            raise ValueError("strategy_lexical_identity_changed")
    if current_capabilities.identity != plan.capability_hash:
        raise ValueError("strategy_capability_identity_changed")
    budget = plan.effective_budget
    envelope = agent_operating_envelope()
    from app.services.cache_manager import get_cache_manager

    cache_manager = get_cache_manager()
    has_source_scope = any(
        item.source_scope is not None for item in plan.task.requirements
    )
    effective_filters = filters
    cache_components: dict[str, Any] | None = None
    retrieval_cache_audit: dict[str, Any] | None = None
    if not has_source_scope:
        cache_components = _intent_cache_components(
            db,
            plan=plan,
            filters=effective_filters,
            top_k=top_k,
            capabilities=current_capabilities,
            context_state=context_state,
            envelope=envelope,
        )
        cache_read = cache_manager.read_intent_retrieval(
            plan.task.knowledge_base_id,
            cache_components=cache_components,
        )
        cached, retrieval_cache_audit = _replay_cached_execution(
            db,
            plan=plan,
            filters=effective_filters,
            context_state=context_state,
            cache_components=cache_components,
            cache_read=cache_read,
            cache_manager=cache_manager,
        )
        if cached is not None:
            return cached
    corpus = await agent_graph.run_bounded_source_io(
        RetrievalCorpus.load,
        db,
        knowledge_base_id=plan.task.knowledge_base_id,
        filters=effective_filters,
    )
    scope_index = None
    resolved_scope_filter_audit = None
    if has_source_scope:
        from app.services.evidence_scope import (
            StructureScopeIndex,
            resolved_scope_filters,
        )

        scope_index = await agent_graph.run_bounded_source_io(
            StructureScopeIndex.load,
            db,
            corpus=corpus,
            task=plan.task,
        )
        effective_filters, resolved_scope_filter_audit = (
            await agent_graph.run_bounded_source_io(
                resolved_scope_filters,
                task=plan.task,
                index=scope_index,
                filters=filters,
            )
        )
        if resolved_scope_filter_audit is not None:
            corpus = await agent_graph.run_bounded_source_io(
                RetrievalCorpus.load,
                db,
                knowledge_base_id=plan.task.knowledge_base_id,
                filters=effective_filters,
            )
            scope_index = await agent_graph.run_bounded_source_io(
                StructureScopeIndex.load,
                db,
                corpus=corpus,
                task=plan.task,
            )
        cache_components = _intent_cache_components(
            db,
            plan=plan,
            filters=effective_filters,
            top_k=top_k,
            capabilities=current_capabilities,
            context_state=context_state,
            envelope=envelope,
        )
        cache_read = cache_manager.read_intent_retrieval(
            plan.task.knowledge_base_id,
            cache_components=cache_components,
        )
        cached, retrieval_cache_audit = _replay_cached_execution(
            db,
            plan=plan,
            filters=effective_filters,
            context_state=context_state,
            cache_components=cache_components,
            cache_read=cache_read,
            cache_manager=cache_manager,
        )
        if cached is not None:
            return cached
    if cache_components is None or retrieval_cache_audit is None:
        raise RuntimeError("intent_retrieval_cache_identity_not_frozen")
    retrieval_cache_audit = {
        **retrieval_cache_audit,
        "write_scheduled_after_commit": cache_manager.shared_cache_available,
    }
    filters = effective_filters
    provider = EmbeddingProvider().for_embedding_identity(
        embedding_model=corpus.target.schema.embedding_model,
        embedding_dimensions=corpus.target.schema.embedding_dimension,
    )
    semantic_queries = [plan.strategy.semantic_query] + [
        requirement.text for requirement in plan.task.requirements
    ]
    vectors = await provider.embed_texts(semantic_queries, text_type="query")
    if (
        len(vectors) != len(semantic_queries)
        or any(
            len(vector) != corpus.target.schema.embedding_dimension
            for vector in vectors
        )
    ):
        raise ValueError("entry_query_vector_shape_invalid")
    query_vector = tuple(float(value) for value in vectors[0])
    facet_query_vectors = {
        requirement.id: tuple(float(value) for value in vectors[index + 1])
        for index, requirement in enumerate(plan.task.requirements)
    }
    if (
        any(not math.isfinite(value) for vector in vectors for value in vector)
        or not any(query_vector)
        or any(not any(vector) for vector in facet_query_vectors.values())
    ):
        raise ValueError("entry_query_vector_invalid")
    with qa_stage("dense_entry", item_count=len(corpus.sources)):
        dense_values = corpus.cosine_scores(np.asarray(query_vector, dtype=np.float64))[0]
    dense_scores = {
        source.chunk_id: float(dense_values[index])
        for index, source in enumerate(corpus.sources)
    }
    chunk_ids = tuple(source.chunk_id for source in corpus.sources)
    lexical_terms = tuple(item.text for item in plan.strategy.lexical_terms)
    global_bm25_enabled = any(
        plan.strategy.layer_weights.for_layer(layer).bm25 > 0
        for layer in plan.strategy.layer_weights.enabled_layers()
    )
    lexical = None
    scope_target_chunk_ids: tuple[str, ...] = ()
    scope_execution_audit: dict[str, Any] | None = None
    if has_source_scope:
        from app.services.evidence_scope import scope_target_plan

        if scope_index is None:
            raise RuntimeError("intent_source_scope_index_missing")
        with qa_stage("dense_entry", item_count=len(plan.task.requirements)):
            facet_dense_scores = {
                requirement.id: {
                    source.chunk_id: float(values[index])
                    for index, source in enumerate(corpus.sources)
                }
                for requirement, values in zip(
                    plan.task.requirements,
                    corpus.cosine_scores(
                        np.asarray(
                            [
                                facet_query_vectors[requirement.id]
                                for requirement in plan.task.requirements
                            ],
                            dtype=np.float64,
                        )
                    ),
                    strict=True,
                )
            }
        chunk_weights = plan.effective_weights("chunk")
        lexical_queries: list[tuple[str, tuple[str, ...]]] = []
        if chunk_weights["bm25"] > 0:
            for requirement in plan.task.requirements:
                terms = tuple(
                    item.text
                    for item in plan.strategy.lexical_terms
                    if requirement.id in item.requirement_ids
                )
                if terms:
                    lexical_queries.append((requirement.id, terms))
        if global_bm25_enabled:
            lexical_queries.append(("__global__", lexical_terms))
        lexical_results = {}
        if lexical_queries:
            with qa_stage("bm25_entry", item_count=len(lexical_terms)):
                batch = search_lexical_index_many(
                    db,
                    plan.task.knowledge_base_id,
                    tuple(terms for _, terms in lexical_queries),
                    expected_identity=current_capabilities.lexical_identity,
                    eligible_chunk_ids=frozenset(chunk_ids),
                    limit=max(budget.bm25_candidates, plan.effective_budget.bm25_candidates),
                )
            lexical_results = {
                key: result for (key, _), result in zip(lexical_queries, batch, strict=True)
            }
            lexical = lexical_results.get("__global__")
        facet_affinities: dict[str, dict[str, float]] = {}
        for requirement in plan.task.requirements:
            dense_order = sorted(
                chunk_ids,
                key=lambda chunk_id: (
                    -facet_dense_scores[requirement.id][chunk_id],
                    chunk_id,
                ),
            )
            dense_affinity = {
                chunk_id: 1.0 / (rank + 1)
                for rank, chunk_id in enumerate(dense_order)
            }
            lexical_affinity: dict[str, float] = {}
            scoped_lexical = lexical_results.get(requirement.id)
            if scoped_lexical is not None:
                lexical_affinity = {
                    item.chunk_id: 1.0 / (rank + 1)
                    for rank, item in enumerate(
                        scoped_lexical.hits[: budget.bm25_candidates]
                    )
                }
            channel_total = chunk_weights["dense"] + (
                chunk_weights["bm25"] if lexical_affinity else 0.0
            )
            facet_affinities[requirement.id] = {
                chunk_id: (
                    chunk_weights["dense"] * dense_affinity[chunk_id]
                    + chunk_weights["bm25"] * lexical_affinity.get(chunk_id, 0.0)
                )
                / channel_total
                for chunk_id in chunk_ids
            }
        scope_target_chunk_ids, target_plan = await agent_graph.run_bounded_source_io(
            scope_target_plan,
            index=scope_index,
            task=plan.task,
            token_budget=int(envelope["context_package_token_budget"]),
            target_limit=min(256, budget.layer_entries),
            candidate_ids=set(chunk_ids),
            affinities=facet_affinities,
            overlap_target_budget=min(4, budget.per_parent_entries),
        )
        scope_execution_audit = {
            "protocol_version": "intent_source_scope_execution_v1",
            "task_hash": plan.task.identity,
            "source_scope_hash": corpus.scope_hash,
            "source_index_identity": scope_index.identity,
            "source_chunk_ids": list(chunk_ids),
            "scope_bindings": [
                item.model_dump(mode="json") for item in scope_index.bind(plan.task)
            ],
            "resolved_scope_filter": resolved_scope_filter_audit,
            "target_plan": target_plan,
            "model_call_count": 0,
        }
        scope_execution_audit["audit_hash"] = control_hash(scope_execution_audit)
    chunk_rows = {
        str(chunk.id): chunk
        for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids)))
    }
    if set(chunk_rows) != set(chunk_ids):
        raise ValueError("entry_chunk_scope_changed")
    graph_state_id = context_state.chunk_relation_graph_state_id
    prefixes = list(
        db.scalars(
            select(RQPrefix).where(
                RQPrefix.graph_state_id == graph_state_id,
                RQPrefix.state == "active",
            )
        )
    )
    prefix_by_id = {str(item.id): item for item in prefixes}
    prefix_by_path = {tuple(int(value) for value in item.rq_path_prefix): item for item in prefixes}
    chunk_rq_paths = {
        chunk_id: tuple(int(value) for value in (chunk_rows[chunk_id].rq_path or []))
        for chunk_id in chunk_ids
    }
    bm25_scores: dict[str, float] = {}
    bm25_witnesses: dict[str, tuple[str, ...]] = {}
    lexical_audit: dict[str, Any] = {"enabled": False}
    if global_bm25_enabled:
        if lexical is None:
            with qa_stage("bm25_entry", item_count=len(lexical_terms)):
                lexical = search_lexical_index_many(
                    db,
                    plan.task.knowledge_base_id,
                    (lexical_terms,),
                    expected_identity=current_capabilities.lexical_identity,
                    eligible_chunk_ids=frozenset(chunk_ids),
                    limit=plan.effective_budget.bm25_candidates,
                )[0]
        if lexical is None:
            raise RuntimeError("lexical_entry_result_missing")
        bm25_scores = {item.chunk_id: item.score for item in lexical.hits}
        bm25_witnesses = {
            item.chunk_id: tuple(
                "bm25:" + control_hash(
                    {
                        "chunk_id": item.chunk_id,
                        "term": witness.term,
                        "positions": witness.positions,
                    }
                )
                for witness in item.witnesses
            )
            for item in lexical.hits
        }
        lexical_audit = {
            "enabled": True,
            "index_state_id": lexical.index_state_id,
            "index_identity": lexical.index_identity,
            "statistics_hash": lexical.statistics_hash,
            "matched_documents": lexical.matched_documents,
            "postings_read": lexical.postings_read,
            "output_truncated": lexical.output_truncated,
        }
    mid_concepts = {
        str(item.id): item
        for item in db.scalars(
            select(MidConcept).where(
                MidConcept.concept_state_id == context_state.mid_concept_state_id,
                MidConcept.state == "active",
            )
        )
    }
    coarse_concepts = {
        str(item.id): item
        for item in db.scalars(
            select(CoarseConcept).where(
                CoarseConcept.coarse_state_id == context_state.coarse_concept_state_id,
                CoarseConcept.state == "active",
            )
        )
    }
    chunk_channels = _chunk_channel_rows(
        chunk_ids=chunk_ids,
        chunk_rq_paths=chunk_rq_paths,
        prefix_by_path=prefix_by_path,
        query_vector=query_vector,
        dense_scores=dense_scores,
        bm25_scores=bm25_scores,
        bm25_witnesses=bm25_witnesses,
        dense_limit=budget.dense_candidates,
        rq_limit=budget.rq_candidates,
        bm25_limit=budget.bm25_candidates,
        enabled_channels=_enabled_channels(plan.strategy.layer_weights.chunk),
    )
    mid_enabled = (
        _enabled_channels(plan.strategy.layer_weights.mid)
        if plan.strategy.layer_weights.mid is not None
        else ()
    )
    mid_channels, mid_projection = _concept_channel_rows(
        layer="mid",
        concepts=tuple(mid_concepts.values()),
        dense_scores=dense_scores,
        rq_prefixes=prefix_by_id,
        query_vector=query_vector,
        bm25_scores=bm25_scores,
        dense_limit=budget.dense_candidates,
        rq_limit=budget.rq_candidates,
        bm25_limit=budget.bm25_candidates,
        enabled_channels=mid_enabled,
    )
    coarse_enabled = (
        _enabled_channels(plan.strategy.layer_weights.coarse)
        if plan.strategy.layer_weights.coarse is not None
        else ()
    )
    coarse_channels, coarse_projection = _concept_channel_rows(
        layer="coarse",
        concepts=tuple(coarse_concepts.values()),
        dense_scores=dense_scores,
        rq_prefixes=prefix_by_id,
        query_vector=query_vector,
        bm25_scores=bm25_scores,
        dense_limit=budget.dense_candidates,
        rq_limit=budget.rq_candidates,
        bm25_limit=budget.bm25_candidates,
        enabled_channels=coarse_enabled,
    )
    channel_rows = {
        "coarse": coarse_channels,
        "mid": mid_channels,
        "chunk": chunk_channels,
    }
    green = float(envelope["path_distance_green_threshold"])
    gray = float(envelope["path_distance_gray_threshold"])
    hard = float(envelope["path_distance_hard_threshold"])
    if not 0 <= green <= gray <= hard:
        raise ValueError("traversal_threshold_order_invalid")
    layer_audits: list[dict[str, Any]] = []
    parent_refs: dict[str, list[str]] = defaultdict(list)
    strategy = plan.strategy
    if strategy.entry_layer == "coarse":
        coarse_roots = _fuse_scope(
            coarse_channels,
            candidate_ids=coarse_concepts,
            weights=strategy.layer_weights.coarse,
            limit=budget.root_entries,
        )
        coarse_edges = list(
            db.scalars(
                select(CoarseConceptEdge).where(
                    CoarseConceptEdge.coarse_state_id == context_state.coarse_concept_state_id
                )
            )
        )
        traversed_coarse, coarse_audit = _traverse(
            layer="coarse",
            roots=coarse_roots,
            edges=coarse_edges,
            limit=budget.layer_entries,
            max_depth=budget.max_depth,
            green=green,
            gray=gray,
            hard=hard,
        )
        layer_audits.append(coarse_audit)
        coarse_ids = [item.node_id for item in traversed_coarse]
        memberships = list(
            db.scalars(
                select(CoarseConceptMembership).where(
                    CoarseConceptMembership.coarse_concept_id.in_(coarse_ids)
                )
            )
        ) if coarse_ids else []
        children = _concept_children(coarse_ids=coarse_ids, memberships=memberships)
        pools = [
            (
                parent,
                _fuse_scope(
                    mid_channels,
                    candidate_ids=children[parent],
                    weights=strategy.layer_weights.mid,
                    limit=budget.per_parent_entries,
                ),
            )
            for parent in coarse_ids
        ]
        mid_roots, merged_refs = _merge_parent_entries(
            pools,
            broad=strategy.selection_scope == "broad",
            limit=budget.layer_entries,
        )
        for child, parents in merged_refs.items():
            parent_refs[child].extend(parents)
    else:
        mid_roots = (
            _fuse_scope(
                mid_channels,
                candidate_ids=mid_concepts,
                weights=strategy.layer_weights.mid,
                limit=budget.root_entries,
            )
            if strategy.entry_layer == "mid"
            else ()
        )
    if strategy.entry_layer in {"coarse", "mid"}:
        mid_edges = list(
            db.scalars(
                select(MidConceptEdge).where(
                    MidConceptEdge.concept_state_id == context_state.mid_concept_state_id
                )
            )
        )
        traversed_mid, mid_audit = _traverse(
            layer="mid",
            roots=mid_roots,
            edges=mid_edges,
            limit=budget.layer_entries,
            max_depth=budget.max_depth,
            green=green,
            gray=gray,
            hard=hard,
        )
        layer_audits.append(mid_audit)
        mid_ids = [item.node_id for item in traversed_mid]
        children = _chunk_children(mid_concepts, mid_ids, set(chunk_ids))
        pools = [
            (
                parent,
                _fuse_scope(
                    chunk_channels,
                    candidate_ids=children[parent],
                    weights=strategy.layer_weights.chunk,
                    limit=budget.per_parent_entries,
                ),
            )
            for parent in mid_ids
        ]
        chunk_roots, merged_refs = _merge_parent_entries(
            pools,
            broad=strategy.selection_scope == "broad",
            limit=budget.layer_entries,
        )
        for child, parents in merged_refs.items():
            parent_refs[child].extend(parents)
    else:
        chunk_roots = _fuse_scope(
            chunk_channels,
            candidate_ids=chunk_ids,
            weights=strategy.layer_weights.chunk,
            limit=budget.root_entries,
        )
    chunk_adjacency = CompleteChunkAdjacency(
        db,
        graph_state_id=graph_state_id,
        allowed_types=EDGE_TYPE_CALIBRATION_EDGE_TYPES,
    )
    chunk_adjacency.preload(item.candidate_id for item in chunk_roots)
    traversed_chunks, chunk_audit = _traverse(
        layer="chunk",
        roots=chunk_roots,
        adjacency_reader=chunk_adjacency,
        limit=budget.layer_entries,
        max_depth=budget.max_depth,
        green=green,
        gray=gray,
        hard=hard,
    )
    chunk_edges = list(chunk_adjacency.loaded_edges())
    chunk_edge_by_id = {str(edge.id): edge for edge in chunk_edges}
    chunk_path_visit_count = len(traversed_chunks)
    traversed_chunks = _unique_traversed_nodes(traversed_chunks)
    chunk_audit = {
        **chunk_audit,
        "path_visit_count": chunk_path_visit_count,
        "unique_node_count": len(traversed_chunks),
        "duplicate_node_visit_count": (
            chunk_path_visit_count - len(traversed_chunks)
        ),
    }
    layer_audits.append(chunk_audit)
    result_limit = min(
        50,
        int(top_k or get_settings().retrieval_result_top_k_default),
        len(traversed_chunks),
    )
    selected_chunks = list(traversed_chunks[:result_limit])
    result_channel_floor_ids = list(
        dict.fromkeys(
            rows[0].candidate_id
            for channel in ("dense", "rq", "bm25")
            if (
                rows := sorted(
                    chunk_channels.get(channel, {}).values(),
                    key=lambda item: (-item.score, item.business_key),
                )
            )
            and any(item.node_id == rows[0].candidate_id for item in traversed_chunks)
        )
    )
    if result_limit >= len(result_channel_floor_ids):
        leader_set = set(result_channel_floor_ids)
        selected_ids = {item.node_id for item in selected_chunks}
        traversed_by_id = {item.node_id: item for item in traversed_chunks}
        for leader in result_channel_floor_ids:
            if leader in selected_ids:
                continue
            replacement = next(
                (
                    index
                    for index in range(len(selected_chunks) - 1, -1, -1)
                    if selected_chunks[index].node_id not in leader_set
                ),
                None,
            )
            if replacement is None:
                break
            selected_ids.discard(selected_chunks[replacement].node_id)
            selected_chunks[replacement] = traversed_by_id[leader]
            selected_ids.add(leader)
        traversal_index = {
            item.node_id: index for index, item in enumerate(traversed_chunks)
        }
        selected_chunks.sort(key=lambda item: traversal_index[item.node_id])
    selected_chunks = tuple(selected_chunks)
    traversal_hash = control_hash(
        traversal_identity := {
            "protocol_version": TRAVERSAL_PROTOCOL,
            "green": green,
            "gray": gray,
            "hard": hard,
            "max_depth": budget.max_depth,
        }
    )
    runtime_settings_hash = agent_graph.runtime_settings_state_hash()
    operating_envelope_hash = stable_hash(envelope)
    gray_runtime_hash = gray_zone_runtime_settings_hash(envelope)
    trace = RetrievalTrace(
        knowledge_base_id=plan.task.knowledge_base_id,
        query=plan.task.question,
        filters_json=filters.model_dump(mode="json"),
        retrieval_mode=PROTOCOL,
        chunk_scope_hash=context_state.chunk_scope_hash,
        structure_graph_hash=context_state.structure_graph_hash,
        chunk_relation_graph_hash=context_state.chunk_relation_graph_hash,
        rq_membership_hash=context_state.rq_membership_hash,
        mid_concept_hash=context_state.mid_concept_hash,
        coarse_concept_hash=context_state.coarse_concept_hash,
        runtime_settings_hash=runtime_settings_hash,
        agent_operating_envelope_hash=operating_envelope_hash,
        policy_state_hash=None,
        prompt_protocol_hash=plan.identity,
        result_chunk_ids_json=[item.node_id for item in selected_chunks],
        concept_path_json=[
            {
                "layer": "chunk",
                "ids": [item.node_id for item in selected_chunks],
            },
            *(
                [
                    {
                        "layer": "mid",
                        "ids": sorted(
                            {
                                parent_id
                                for item in selected_chunks
                                for parent_id in parent_refs.get(item.node_id, ())
                            }
                        ),
                    }
                ]
                if any(parent_refs.get(item.node_id) for item in selected_chunks)
                else []
            ),
        ],
        scores_json={
            "protocol_version": PROTOCOL,
            "accepted_plan_hash": plan.identity,
            "capability_hash": plan.capability_hash,
            "lexical": lexical_audit,
            "projection": {"mid": mid_projection, "coarse": coarse_projection},
            "reward_call_count": 0,
            "post_retrieval_model_call_count": 0,
            "retrieval_cache": retrieval_cache_audit,
        },
        query_facets_json={
            "protocol_version": plan.task.protocol_version,
            "required_facets": [item.id for item in plan.task.requirements],
            "requirements": [item.model_dump(mode="json") for item in plan.task.requirements],
        },
        entry_nodes_json=[
            {
                "layer": strategy.entry_layer,
                **item.model_dump(mode="json"),
            }
            for item in (
                coarse_roots
                if strategy.entry_layer == "coarse"
                else mid_roots
                if strategy.entry_layer == "mid"
                else chunk_roots
            )
        ],
        frontier_json=[
            {"layer": item["layer"], "frontier_remaining": item["frontier_remaining"]}
            for item in layer_audits
        ],
        stage_queues_json={
            item["layer"]: {"root_ids": item["root_ids"], "stop_reason": item["stop_reason"]}
            for item in layer_audits
        },
        candidate_pools_json={
            layer: {
                channel: [row.model_dump(mode="json") for row in rows.values()]
                for channel, rows in channels.items()
                if channel in _enabled_channels(strategy.layer_weights.for_layer(layer))
            }
            for layer, channels in channel_rows.items()
            if layer in strategy.layer_weights.enabled_layers()
        },
        topk_selection_json={
            "top_k": result_limit,
            "selected_chunk_ids": [item.node_id for item in selected_chunks],
            "candidate_count": len(traversed_chunks),
            "path_candidate_count": chunk_path_visit_count,
            "duplicate_path_candidate_count": (
                chunk_path_visit_count - len(traversed_chunks)
            ),
            "truncated": len(traversed_chunks) > result_limit,
            "channel_floor_protocol_version": "entry_channel_floor_v1",
            "channel_floor_chunk_ids": result_channel_floor_ids,
        },
        path_labels_json=[
            {
                "layer": "chunk",
                "node_id": item.node_id,
                "chunk_id": item.node_id,
                "root_node_id": item.root_node_id,
                "path": list(item.node_path),
                "path_edge_ids": list(item.edge_path),
                "path_edge_types": [
                    str(chunk_edge_by_id[edge_id].edge_type)
                    for edge_id in item.edge_path
                    if edge_id in chunk_edge_by_id
                ],
                "path_edge_distances": [
                    float(chunk_edge_by_id[edge_id].distance)
                    for edge_id in item.edge_path
                    if edge_id in chunk_edge_by_id
                ],
                "path_edge_strengths": [
                    float(chunk_edge_by_id[edge_id].raw_strength)
                    for edge_id in item.edge_path
                    if edge_id in chunk_edge_by_id
                ],
                "cycle_distance_rewards": [],
                "expanded_edge_ids": list(item.edge_path),
                "covered_facets": [],
                "evidence_roles": ["retrieval_hit"],
                "distance_so_far": item.distance,
                "reward_so_far": 0,
                "cycle_reward_so_far": 0,
                "parent_layer": "mid" if parent_refs.get(item.node_id) else None,
                "parent_node_id": (
                    next(iter(parent_refs.get(item.node_id, ())), None)
                ),
                "stop_reason": "selected_for_context_package",
                "support_refs": {
                    "edge_ids": list(item.edge_path),
                    "edge_types": [
                        str(chunk_edge_by_id[edge_id].edge_type)
                        for edge_id in item.edge_path
                        if edge_id in chunk_edge_by_id
                    ],
                    "support_ids": list(item.support_ids),
                    "support_chunk_ids": [
                        support_id for support_id in item.support_ids if support_id in chunk_rows
                    ],
                    "entry_strength": item.entry_score,
                },
                "entry_channels": list(item.entry_channels),
                "entry_parent_refs": [
                    {
                        "parent_layer": "mid",
                        "parent_node_id": parent_id,
                        "edge_type": "mid_chunk_primary_support",
                        "support_refs": {"support_chunk_ids": [item.node_id]},
                    }
                    for parent_id in dict.fromkeys(parent_refs.get(item.node_id, ()))
                ],
                "path_edge_type_multiset": {
                    edge_type: sum(
                        1
                        for edge_id in item.edge_path
                        if edge_id in chunk_edge_by_id
                        and str(chunk_edge_by_id[edge_id].edge_type) == edge_type
                    )
                    for edge_type in {
                        str(chunk_edge_by_id[edge_id].edge_type)
                        for edge_id in item.edge_path
                        if edge_id in chunk_edge_by_id
                    }
                },
                "edge_reuse_counts": {edge_id: 1 for edge_id in item.edge_path},
            }
            for item in selected_chunks
        ],
        convergence_json={
            "protocol_version": TRAVERSAL_PROTOCOL,
            "agent_operating_envelope_hash": operating_envelope_hash,
            "layers": layer_audits,
            "cycle_reward": 0,
            "model_call_count": 0,
            "gray_zone_decision_count": 0,
            "gray_zone_rule_evaluation_count": 0,
            "red_zone_pruned_count": 0,
            "hard_stop_pruned_count": 0,
            "gray_zone_model_call_count": 0,
            "dominance_pruned_count": sum(
                item["dominance_or_cycle_pruned_count"] for item in layer_audits
            ),
        },
        edge_distance_protocol_hash=context_state.diagnostics_json.get("edge_distance_protocol_hash"),
        traversal_protocol_hash=traversal_hash,
        conversation_state_scope_hash=plan.task.conversation_scope_hash,
        diagnostics_json={
            "protocol_version": PROTOCOL,
            "task_hash": plan.task.identity,
            "intent_hash": plan.intent.identity,
            "strategy_hash": plan.strategy.identity,
            "entry_layer": strategy.entry_layer,
            "proposal_hash": plan.proposal_hash,
            "capability_hash": plan.capability_hash,
            "gray_zone_protocol_version": GRAY_ZONE_PROTOCOL,
            "traversal_identity": traversal_identity,
            "gray_zone_model_call_count": 0,
            "gray_zone_runtime_settings_hash": gray_runtime_hash,
            "agent_operating_envelope": envelope,
            "agent_operating_envelope_hash": operating_envelope_hash,
            "result_reflection_enabled": False,
            "generation_sufficiency_model_enabled": False,
            "policy_update_eligible": False,
            "source_scope_execution": scope_execution_audit,
            "intent_retrieval_cache_identity_hash": control_hash(cache_components),
            "retrieval_cache": retrieval_cache_audit,
        },
    )
    db.add(trace)
    db.flush()
    results = []
    for item in selected_chunks:
        chunk = chunk_rows[item.node_id]
        traversal = {
            "layer": "chunk",
            "node_id": item.node_id,
            "root_node_id": item.root_node_id,
            "path": list(item.node_path),
            "path_edge_ids": list(item.edge_path),
            "path_edge_types": [
                str(chunk_edge_by_id[edge_id].edge_type)
                for edge_id in item.edge_path
                if edge_id in chunk_edge_by_id
            ],
            "distance_so_far": item.distance,
            "reward_so_far": 0,
            "cycle_reward_so_far": 0,
            "covered_facets": [],
            "evidence_roles": ["retrieval_hit"],
            "support_refs": {
                "support_chunk_ids": [
                    support_id for support_id in item.support_ids if support_id in chunk_rows
                ],
            },
            "entry_parent_refs": [
                {"parent_layer": "mid", "parent_node_id": parent_id}
                for parent_id in dict.fromkeys(parent_refs.get(item.node_id, ()))
            ],
            "why_selected": "accepted_by_layered_distance_traversal_v2",
        }
        results.append(
            search_payload_for_chunk(
                db,
                chunk,
                1.0 / (1.0 + item.distance),
                {"path_distance": item.distance},
                {
                    "retrieval_trace_id": trace.id,
                    "retrieval_protocol_version": PROTOCOL,
                    "root_node_id": item.root_node_id,
                    "entry_score": item.entry_score,
                    "entry_channels": list(item.entry_channels),
                    "path_edge_ids": list(item.edge_path),
                    "support_ids": list(item.support_ids),
                    "parent_node_ids": list(dict.fromkeys(parent_refs.get(item.node_id, ()))),
                    "traversal": traversal,
                    "why_selected": "layered_distance_traversal_v2",
                },
            )
        )
    for index, item in enumerate(layer_audits):
        db.add(
            GraphRetrievalStep(
                retrieval_trace_id=trace.id,
                knowledge_base_id=plan.task.knowledge_base_id,
                step_index=index,
                layer=item["layer"],
                action="fuse_and_traverse",
                action_type="deterministic_layer_execution",
                input_json={
                    "accepted_plan_hash": plan.identity,
                    "weights": strategy.layer_weights.for_layer(item["layer"]).model_dump(mode="json"),
                },
                output_json=item,
                score_json={"fusion_protocol": "weighted_rrf_entry_v2"},
                candidate_pool_ids_json=item["root_ids"],
                selected_topk_ids_json=[
                    label.node_id for label in selected_chunks
                ] if item["layer"] == "chunk" else [],
                dominance_pruned_count=item["dominance_or_cycle_pruned_count"],
                cycle_distance_reward=0,
                gray_zone_path_decisions_json=[],
                stop_reason=item["stop_reason"],
                diagnostics_json={
                    "protocol_version": TRAVERSAL_PROTOCOL,
                    "model_call_count": 0,
                    "reward_call_count": 0,
                    "intent_execution_gray_zone_decisions": item["gray_zone_decisions"],
                    "path_labels": (
                        list(trace.path_labels_json or [])
                        if item["layer"] == "chunk"
                        else []
                    ),
                },
            )
        )
    audit = {
        "protocol_version": PROTOCOL,
        "retrieval_trace_id": trace.id,
        "accepted_plan_hash": plan.identity,
        "entry_layer": strategy.entry_layer,
        "result_count": len(results),
        "lexical": lexical_audit,
        "layers": layer_audits,
        "post_retrieval_model_call_count": 0,
        "reward_call_count": 0,
        "source_scope_execution": scope_execution_audit,
        "retrieval_cache": retrieval_cache_audit,
    }
    cache_payload = {
        "protocol_version": "intent_execution_retrieval_cache_payload_v1",
        "cache_identity_hash": control_hash(cache_components),
        "source_retrieval_trace_id": trace.id,
        "result_chunk_ids": list(trace.result_chunk_ids_json or []),
        "result_identity_hash": control_hash(list(trace.result_chunk_ids_json or [])),
    }
    db.flush()
    return LayeredExecutionResult(
        results=results,
        trace=trace,
        audit=audit,
        scope_target_chunk_ids=scope_target_chunk_ids,
        scope_execution_audit=scope_execution_audit,
        cache_components=cache_components,
        cache_payload=cache_payload,
        cache_audit=retrieval_cache_audit,
    )
