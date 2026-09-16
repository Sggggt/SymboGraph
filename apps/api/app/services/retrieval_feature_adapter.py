"""Build feature inputs from persisted graph choices and the actual packed text."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from sqlalchemy import select

from app.models import ChunkStructureMapping, ChunkStructureNode, GraphRetrievalStep, MidConcept, CoarseConcept
from app.retrieval_control_contracts import (
    DecisionPanel, FacetOpportunity, PathEvaluationIdentity, PathEvaluationParameters,
    PathFeatureCandidate, ScoreBounds, control_hash,
)
from app.services.context_graph import matched_query_facet_term_witnesses_for_text, query_facet_ordered_window_protocol_hash, stable_hash
from app.services.retrieval_path_features import compute_path_features
from app.services.retrieval_constraints import formal_literals_match
from app.services.source_use import PROTOCOL as SOURCE_USE_PROTOCOL, analyze_source_use, source_use_decision
from app.services.qa_performance import qa_sync_timed
from app.services.structure_roles import structure_roles as _roles


def _opportunities(task, source, text, roles, scores, *, clipped=False, use_decisions=None):
    result = []
    source_use = analyze_source_use(text)
    for index, facet in enumerate(task.requirements):
        score = float(scores[index])
        matched_roles = all(role in roles for role in facet.source_roles)
        has_quantity = facet.role != "quantity" or re.search(r"\d", text) is not None
        # Only exact identifiers/numbers are literal source constraints.
        # Natural-language topic identity remains the fixed semantic proxy.
        literal_match = formal_literals_match(facet, source.title, text)
        use = source_use_decision(task, facet, source_use, roles=roles)
        if use_decisions is not None:
            use_decisions.append(use)
        upper = score if matched_roles and has_quantity and literal_match and use.allowed else 0.0
        result.append(FacetOpportunity(facet_id=facet.id, value=ScoreBounds(
            lower=0 if clipped else upper, upper=upper)))
    return tuple(result)


def _structure_roles(db, selected_scope):
    grouped = defaultdict(list)
    for row in db.execute(select(ChunkStructureMapping.chunk_id, ChunkStructureNode.title, ChunkStructureNode.node_type)
        .join(ChunkStructureNode, ChunkStructureNode.id == ChunkStructureMapping.structure_node_id)
        .where(ChunkStructureMapping.chunk_id.in_(sorted(selected_scope)))):
        grouped[row.chunk_id].append(row)
    return {cid: _roles(nodes) for cid, nodes in grouped.items()}


@qa_sync_timed("feature_preparation")
def build_feature_snapshot(db, *, task, strategy, corpus, vectors, package, trace, source_audit, include_panels=True,
                           source_use_audit=None, scope_index=None):
    if corpus.knowledge_base_id != task.knowledge_base_id or package.knowledge_base_id != task.knowledge_base_id:
        raise ValueError("feature_adapter_knowledge_scope_changed")
    if len(vectors) != len(task.requirements) + 1:
        raise ValueError("feature_adapter_canonical_vector_scope_invalid")
    scores = corpus.cosine_scores(vectors)
    canonical_scores, facet_scores = scores[0], scores[1:]
    steps = list(db.scalars(select(GraphRetrievalStep).where(
        GraphRetrievalStep.retrieval_trace_id == trace.id).order_by(GraphRetrievalStep.step_index)))
    pools = next(((step.diagnostics_json or {}).get("candidate_pools") for step in steps
                  if (step.diagnostics_json or {}).get("candidate_pools")), {}) or {}
    chunk_pools = (pools.get("chunk_by_mid") or []) if include_panels else []
    if len(chunk_pools) > 64:
        raise ValueError("feature_adapter_panel_budget_exceeded")
    packed = {item["chunk_id"]: item for item in package.package_json["chunks"]}
    selected_scope = set(packed)
    for pool in chunk_pools:
        selected_scope.update(pool.get("candidate_ids") or [])
    if len(selected_scope) > 2048 or not selected_scope.issubset(corpus.by_id):
        raise ValueError("feature_adapter_candidate_scope_invalid")
    structure_roles = _structure_roles(db, selected_scope)
    scope_inputs = ()
    packed_scope_intervals = ()
    if any(facet.source_scope is not None for facet in task.requirements):
        from app.services.evidence_scope import StructureScopeIndex, package_scope_intervals
        scope_index = scope_index or StructureScopeIndex.load(db, corpus=corpus, task=task)
        if scope_index.corpus.scope_hash != corpus.scope_hash:
            raise ValueError('feature_scope_source_identity_changed')
        scope_inputs = scope_index.bind(task)
        packed_scope_intervals = package_scope_intervals(task, package)
    bound_by_facet = {item.facet_id: item for item in scope_inputs}
    terms_by_surface = defaultdict(list)
    groups = []
    for facet in task.requirements:
        terms = [term for term in strategy.terms if term.facet_id == facet.id]
        groups.append({"facet": facet.text, "aliases": [term.surface for term in terms]})
        for term in terms:
            terms_by_surface[(facet.text, term.surface.casefold().strip())].append(term.id)
    routing_packet = {"required_facets": [f.text for f in task.requirements], "facet_groups": groups,
                      "diagnostics": {"lexical_only_aliases": True}}
    labels = defaultdict(list)
    for label in trace.path_labels_json or []:
        labels[(label.get("layer"), label.get("node_id"))].append(label)
    root_scores = {}
    def root_score(layer, node_id):
        key = layer, node_id
        if key in root_scores:
            return root_scores[key]
        if layer == "chunk":
            index = corpus.positions.get(node_id)
            value = float(canonical_scores[index]) if index is not None else 0
        else:
            model = MidConcept if layer == "mid" else CoarseConcept
            node = db.get(model, node_id)
            ids = list(node.support_chunk_ids_json or []) if node is not None else []
            values = sorted((float(canonical_scores[corpus.positions[cid]]) for cid in ids if cid in corpus.positions),
                            reverse=True)[:4]
            value = .75 * values[0] + .25 * sum(values) / len(values) if values else 0
        root_scores[key] = max(1e-6, value)
        return root_scores[key]
    def path_values(layer, node_id, visited=()):
        if (layer, node_id) in visited or len(visited) > 3:
            raise ValueError("feature_path_parent_cycle")
        candidates = []
        for label in labels.get((layer, node_id), []):
            distances = [0.0, 0.0, 0.0]
            distances[{"coarse": 0, "mid": 1, "chunk": 2}[layer]] = sum(
                float(value) for value in label.get("path_edge_distances") or [])
            parents = [ref for ref in label.get("entry_parent_refs") or []
                       if ref.get("parent_layer") in {"coarse", "mid"}
                       and ref.get("parent_layer") != layer]
            for parent in parents:
                ancestor = path_values(parent["parent_layer"], parent["parent_node_id"], (*visited, (layer, node_id)))
                if ancestor is not None:
                    candidates.append((ancestor[0], tuple(a + b for a, b in zip(ancestor[1], distances))))
            if not parents:
                start = (label.get("path") or [node_id])[0]
                candidates.append((-math.log(root_score(layer, start)), tuple(distances)))
        return min(candidates, key=lambda item: item[0] + sum(item[1])) if candidates else None
    facts = {}
    packed_use_decisions = []
    def make(cid, *, parent=None, routing_cost=0, role_rank=0, candidate_id=None, packed_item=None):
        source = corpus.by_id[cid]
        text = packed_item["content"] if packed_item else source.text
        clipped = bool(packed_item and packed_item.get("content_clipped"))
        key = cid, text, clipped
        if key not in facts:
            matched = []
            for item in matched_query_facet_term_witnesses_for_text(text, routing_packet):
                for surface in item["matched_terms"]:
                    matched.extend(terms_by_surface.get((item["facet"], surface.casefold().strip()), []))
            use_decisions = []
            opportunities = _opportunities(task, source, text, structure_roles.get(cid, set()),
                                        facet_scores[:, corpus.positions[cid]], clipped=clipped, use_decisions=use_decisions)
            if bound_by_facet:
                from app.services.source_use import _merge_scope_intervals, _intersect_scope_intervals
                from app.services.evidence_scope import evaluate_scope_obligation
                from app.retrieval_control_contracts import EvidenceInterval
                span = packed_item.get('char_span') if packed_item else [source.char_start, source.char_end]
                current_interval = EvidenceInterval(knowledge_base_id=task.knowledge_base_id,
                    document_version_id=source.document_version_id, start=span[0], end=span[1])
                adjusted = []
                for ordinal, facet in enumerate(task.requirements):
                    if facet.id not in bound_by_facet:
                        adjusted.append(opportunities[ordinal])
                        continue
                    bound = bound_by_facet[facet.id]
                    status = evaluate_scope_obligation(task=task, facet=facet, bound=bound, packed=(current_interval,))
                    ranges = _merge_scope_intervals((item.knowledge_base_id, item.document_version_id, item.start, item.end)
                        for coverage in status.coverage for item in coverage.usable_intervals)
                    scoped_text = '\n'.join(text[start-span[0]:end-span[0]] for _, _, start, end in ranges)
                    # Every textual slot uses the actual scoped intersection.
                    scoped_task = task.model_copy(update={'requirements': (facet,)})
                    value = _opportunities(scoped_task, source, scoped_text, structure_roles.get(cid, set()),
                        (float(facet_scores[ordinal, corpus.positions[cid]]),), clipped=False)[0]
                    if not scoped_text:
                        upper = (float(facet_scores[ordinal, corpus.positions[cid]])
                            if any(binding.reason != 'resolved' for binding in bound.bindings) else 0)
                        value = value.model_copy(update={'value': ScoreBounds(lower=0, upper=upper)})
                    adjusted.append(value)
                opportunities = tuple(adjusted)
            facts[key] = (opportunities,
                          tuple(sorted(set(matched))), tuple(use_decisions))
        opportunities, matched, use_decisions = facts[key]
        if packed_item is not None:
            packed_use_decisions.append(use_decisions)
        actual = path_values("chunk", cid)
        if actual is None and parent:
            actual = path_values("mid", parent)
        path_observed = actual is not None
        if actual is None:
            # A source address does not prove a traversed path. Its quality
            # remains [0, 1], including for restored or retained sources.
            actual = (0, (0, 0, 0))
        span = packed_item.get("char_span") if packed_item else [source.char_start, source.char_end]
        return PathFeatureCandidate(id=candidate_id or cid, source_id=f"{cid}:{span[0]}:{span[1]}",
            topic_group=source.document_id, source_valid=bool(source_audit["all_valid"]), path_observed=path_observed,
            opportunities=opportunities, canonical_entry_distance=actual[0], physical_distances=actual[1],
            matched_term_ids=matched, routing_cost=routing_cost, role_rank=role_rank)
    panels = []
    for pool in chunk_pools:
        snapshot = pool.get('query_facet_posterior_snapshot') or {}
        posterior = snapshot.get('posterior') or {}
        if (snapshot.get('protocol_version') != 'query_facet_posterior_calibration_v1'
            or snapshot.get('snapshot_hash') != stable_hash({key: value for key, value in snapshot.items() if key != 'snapshot_hash'})
            or set(posterior) != {facet.text for facet in task.requirements}
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in posterior.values())):
            raise ValueError('feature_routing_weight_snapshot_invalid')
        routing_weights = {facet.id: float(posterior[facet.text]) for facet in task.requirements}
        candidates = []
        cards = pool.get("rq_chunk_seed_cards") or {}
        for cid in pool.get("candidate_ids") or []:
            seed_cards = cards.get(cid, []) if isinstance(cards, dict) else []
            rank = min((int(card.get("membership_role_tie_break_rank", 6)) for card in seed_cards), default=6)
            candidates.append(make(cid, parent=pool["parent_node_id"],
                                   routing_cost=-float(pool["candidate_scores"][cid]), role_rank=rank))
        panels.append(DecisionPanel(id=pool["parent_node_id"], candidates=tuple(candidates),
            selected_ids=tuple(pool.get("ranked_selected_ids") or pool["selected_ids"]),
            limit=int(pool["per_parent_budget_status"]["budget"]), routing_facet_weights=routing_weights))
    parameters = PathEvaluationParameters(protocol_version="canonical_task_path_quality_v4",
        source_use_protocol=SOURCE_USE_PROTOCOL,
        scope_index_protocol='task_structure_index_v1' if scope_inputs and scope_index.task_hash else 'complete_structure_index_v1',
        scope_inputs=scope_inputs, packed_scope_intervals=packed_scope_intervals,
        scope_source_chunk_ids=tuple(sorted(corpus.by_id)) if scope_inputs else (), identity=PathEvaluationIdentity(
        graph_scope_hash=control_hash({key: getattr(trace, key, None) for key in (
            "knowledge_base_id", "chunk_scope_hash", "chunk_relation_graph_hash", "mid_concept_hash", "coarse_concept_hash")}),
        vector_runtime_hash=corpus.vector_identity_hash, source_scope_hash=corpus.scope_hash,
        canonical_vectors_hash=control_hash(vectors),
        match_protocol_hash=query_facet_ordered_window_protocol_hash()),
        scope_selection=scope_index.semantic_selection if scope_inputs else None)
    packaged = tuple(make(cid, packed_item=item) for cid, item in packed.items())
    features = compute_path_features(task=task, strategy=strategy, panels=tuple(panels),
        packaged_candidates=packaged, parameters=parameters)
    replay_input = {"task": task.model_dump(mode="json"), "strategy": strategy.model_dump(mode="json"),
        "panels": [panel.model_dump(mode="json") for panel in panels],
        "package": [item.model_dump(mode="json") for item in packaged],
        "parameters": parameters.model_dump(mode="json")}
    if control_hash(replay_input) != features.input_hash:
        raise ValueError("feature_adapter_replay_identity_invalid")
    if source_use_audit is not None:
        source_use_audit.clear()
        source_use_audit.update(protocol_version=SOURCE_USE_PROTOCOL, feature_input_hash=features.input_hash,
            packed_source_count=len(packed_use_decisions),
            metadata_dominant_source_count=sum(any(d.metadata_dominant for d in row) for row in packed_use_decisions),
            rejected_facet_count=sum(not d.allowed for row in packed_use_decisions for d in row),
            reasons=dict(Counter(d.reason for row in packed_use_decisions for d in row)), model_call_count=0)
        source_use_audit["audit_hash"] = control_hash(source_use_audit)
    return features, parameters, facet_scores, replay_input
