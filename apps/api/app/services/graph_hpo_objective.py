from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np

from app.models import Chunk
from app.services.embeddings import ChatProvider
from app.services.graph_algorithms import GraphHyperparameters


HPO_OBJECTIVE_SCHEMA_VERSION = "graph_hpo_judge_objective_v1"
HPO_JUDGE_PROMPT_VERSION = "graph_hpo_pairwise_judge_v1"

FEATURE_NAMES = [
    "edge_evidence_coverage",
    "node_evidence_coverage",
    "multi_chunk_edge_support_rate",
    "multi_doc_concept_support_rate",
    "single_chunk_relation_rate",
    "evidence_source_entropy",
    "evidence_to_edge_ratio",
    "concept_support_gini",
    "relation_support_gini",
    "top_concept_dominance",
    "low_support_node_rate",
    "low_confidence_edge_rate",
    "confidence_entropy",
    "isolated_node_rate",
    "connected_component_count_norm",
    "giant_component_ratio",
    "mean_component_size_norm",
    "edge_density",
    "average_degree_norm",
    "degree_gini",
    "bridge_edge_rate",
    "modularity",
    "community_count_norm",
    "community_size_entropy",
    "largest_community_ratio",
    "inter_community_edge_ratio",
    "intra_community_edge_ratio",
    "conductance_mean",
    "conductance_p90",
    "average_clustering",
    "triangle_rate",
    "local_closure_rate",
    "chapter_coherence_rate",
    "section_mixing_entropy",
    "semantic_separation_margin",
    "cross_community_high_similarity_rate",
    "same_community_low_similarity_rate",
    "node_jaccard_bootstrap",
    "edge_jaccard_bootstrap",
]


@dataclass(frozen=True)
class CandidateGraph:
    graph: nx.Graph
    payload_chunk_ids: list[str]


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    params: GraphHyperparameters
    features: dict[str, float]
    hard_failures: list[str]
    summary: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _entropy(values: list[Any], *, normalize: bool = True) -> float:
    if not values:
        return 0.0
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values() if count)
    if normalize and len(counts) > 1:
        entropy /= math.log(len(counts))
    return float(max(0.0, min(1.0, entropy)))


def _gini(values: list[float]) -> float:
    cleaned = sorted(max(0.0, float(value)) for value in values if math.isfinite(float(value)))
    if not cleaned or sum(cleaned) == 0:
        return 0.0
    n = len(cleaned)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(cleaned))
    return float((2 * weighted_sum) / (n * sum(cleaned)) - (n + 1) / n)


def payload_fingerprint(payloads: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(payloads, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def build_candidate_graph(
    payloads: dict[str, dict[str, Any]],
    hyperparameters: GraphHyperparameters,
    *,
    chunks: list[Chunk] | None = None,
) -> CandidateGraph:
    params = hyperparameters.normalized()
    graph = nx.Graph()
    chunk_by_id = {str(chunk.id): chunk for chunk in chunks or []}
    alias_to_name: dict[str, str] = {}

    for chunk_id, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        for item in payload.get("concepts", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("canonical_name") or item.get("name") or "").strip()
            if not name:
                continue
            chunk = chunk_by_id.get(str(chunk_id))
            graph.add_node(name)
            node = graph.nodes[name]
            node.setdefault("chunk_ids", set()).add(str(chunk_id))
            if chunk and getattr(chunk, "document_id", None):
                node.setdefault("document_ids", set()).add(str(chunk.document_id))
            if chunk and getattr(chunk, "chapter", None):
                node.setdefault("chapters", set()).add(str(chunk.chapter))
            if chunk and getattr(chunk, "section", None):
                node.setdefault("sections", set()).add(str(chunk.section))
            node["confidence"] = max(_safe_float(item.get("confidence"), _safe_float(item.get("importance_score"), 0.0)), _safe_float(node.get("confidence"), 0.0))
            alias_to_name[name.lower()] = name
            for alias in item.get("aliases", []) or []:
                if isinstance(alias, str) and alias.strip():
                    alias_to_name[alias.strip().lower()] = name

    for chunk_id, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        chunk = chunk_by_id.get(str(chunk_id))
        for item in payload.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            raw_source = str(item.get("source") or "").strip()
            raw_target = str(item.get("target") or "").strip()
            if not raw_source or not raw_target or raw_source == raw_target:
                continue
            source = alias_to_name.get(raw_source.lower(), raw_source)
            target = alias_to_name.get(raw_target.lower(), raw_target)
            confidence = _safe_float(item.get("confidence"), 0.0)
            weight = _safe_float(item.get("weight"), confidence)
            if confidence < params.min_relation_confidence or weight < params.min_accepted_relation_weight:
                continue
            graph.add_node(source)
            graph.add_node(target)
            if graph.has_edge(source, target):
                edge = graph.edges[source, target]
                edge["confidence"] = max(float(edge.get("confidence", 0.0)), confidence)
                edge["weight"] = max(float(edge.get("weight", 0.0)), weight)
                edge.setdefault("confidences", []).append(confidence)
            else:
                graph.add_edge(source, target, confidence=confidence, weight=weight, confidences=[confidence])
                edge = graph.edges[source, target]
            edge.setdefault("evidence_chunk_ids", set()).add(str(chunk_id))
            edge.setdefault("relation_types", set()).add(str(item.get("relation_type") or "related_to"))
            if chunk and getattr(chunk, "document_id", None):
                edge.setdefault("document_ids", set()).add(str(chunk.document_id))
            if chunk and getattr(chunk, "chapter", None):
                edge.setdefault("chapters", set()).add(str(chunk.chapter))
            if chunk and getattr(chunk, "section", None):
                edge.setdefault("sections", set()).add(str(chunk.section))

    return CandidateGraph(graph=graph, payload_chunk_ids=sorted(str(chunk_id) for chunk_id in payloads))


def _node_chunk_count(graph: nx.Graph, node: str) -> int:
    return len(graph.nodes[node].get("chunk_ids", set()) or set())


def _edge_chunk_count(edge_data: dict[str, Any]) -> int:
    return len(edge_data.get("evidence_chunk_ids", set()) or set())


def _communities(graph: nx.Graph) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes]
    try:
        return [set(item) for item in nx.community.greedy_modularity_communities(graph, weight="weight")]
    except Exception:
        return [set(component) for component in nx.connected_components(graph)]


def _community_map(communities: list[set[str]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, community in enumerate(communities):
        for node in community:
            mapping[node] = index
    return mapping


def _conductance_values(graph: nx.Graph, communities: list[set[str]]) -> list[float]:
    values: list[float] = []
    if graph.number_of_edges() == 0:
        return values
    nodes = set(graph.nodes)
    for community in communities:
        if not community or community == nodes:
            continue
        cut_edges = 0
        volume = 0
        complement_volume = 0
        for source, target in graph.edges:
            in_source = source in community
            in_target = target in community
            if in_source != in_target:
                cut_edges += 1
            if in_source:
                volume += 1
            if in_target:
                volume += 1
            if not in_source:
                complement_volume += 1
            if not in_target:
                complement_volume += 1
        denom = min(volume, complement_volume)
        if denom:
            values.append(_ratio(cut_edges, denom))
    return values


def _community_source_entropy(graph: nx.Graph, communities: list[set[str]], key: str) -> float:
    entropies: list[float] = []
    weights: list[int] = []
    for community in communities:
        values: list[str] = []
        for node in community:
            values.extend(sorted(graph.nodes[node].get(key, set()) or []))
        if values:
            entropies.append(_entropy(values))
            weights.append(len(community))
    if not entropies:
        return 0.0
    return float(np.average(entropies, weights=weights or None))


def _community_coherence(graph: nx.Graph, communities: list[set[str]], key: str) -> float:
    scores: list[float] = []
    weights: list[int] = []
    for community in communities:
        values: list[str] = []
        for node in community:
            values.extend(sorted(graph.nodes[node].get(key, set()) or []))
        if values:
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            scores.append(max(counts.values()) / len(values))
            weights.append(len(community))
    if not scores:
        return 0.0
    return float(np.average(scores, weights=weights or None))


def _bootstrap_stability(payloads: dict[str, dict[str, Any]], params: GraphHyperparameters, chunks: list[Chunk] | None, full_graph: nx.Graph) -> tuple[float, float]:
    chunk_ids = sorted(payloads)
    if len(chunk_ids) < 2 or full_graph.number_of_nodes() == 0:
        return 1.0, 1.0
    full_nodes = set(full_graph.nodes)
    full_edges = {frozenset(edge) for edge in full_graph.edges}
    node_scores: list[float] = []
    edge_scores: list[float] = []
    for offset in range(min(3, len(chunk_ids))):
        sampled_ids = [chunk_id for index, chunk_id in enumerate(chunk_ids) if (index + offset) % 2 == 0]
        if not sampled_ids:
            sampled_ids = chunk_ids[:1]
        sampled = {chunk_id: payloads[chunk_id] for chunk_id in sampled_ids}
        graph = build_candidate_graph(sampled, params, chunks=chunks).graph
        nodes = set(graph.nodes)
        edges = {frozenset(edge) for edge in graph.edges}
        node_scores.append(_ratio(len(nodes & full_nodes), len(nodes | full_nodes)))
        edge_scores.append(_ratio(len(edges & full_edges), len(edges | full_edges)) if (edges or full_edges) else 1.0)
    return float(np.mean(node_scores)), float(np.mean(edge_scores))


def extract_candidate_graph_features(
    payloads: dict[str, dict[str, Any]],
    params: GraphHyperparameters,
    *,
    chunks: list[Chunk] | None = None,
) -> CandidateEvaluation:
    candidate = build_candidate_graph(payloads, params, chunks=chunks)
    graph = candidate.graph
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    max_edges = node_count * (node_count - 1) / 2
    components = [set(component) for component in nx.connected_components(graph)] if node_count else []
    largest_component = max((len(component) for component in components), default=0)
    degrees = [float(degree) for _node, degree in graph.degree]
    edge_data = [data for _source, _target, data in graph.edges(data=True)]
    node_chunk_counts = [_node_chunk_count(graph, node) for node in graph.nodes]
    edge_chunk_counts = [_edge_chunk_count(data) for data in edge_data]
    edge_docs = [document_id for data in edge_data for document_id in sorted(data.get("document_ids", set()) or [])]
    node_docs = [document_id for node in graph.nodes for document_id in sorted(graph.nodes[node].get("document_ids", set()) or [])]
    confidences = [float(data.get("confidence", 0.0) or 0.0) for data in edge_data]
    communities = _communities(graph)
    community_by_node = _community_map(communities)
    community_sizes = [len(community) for community in communities]
    inter_edges = 0
    intra_edges = 0
    for source, target in graph.edges:
        if community_by_node.get(source) == community_by_node.get(target):
            intra_edges += 1
        else:
            inter_edges += 1
    conductance = _conductance_values(graph, communities)
    bridge_count = len(list(nx.bridges(graph))) if edge_count else 0
    triangles = sum(nx.triangles(graph).values()) / 3 if node_count else 0.0
    possible_triangles = sum(degree * (degree - 1) / 2 for degree in degrees)
    try:
        modularity = nx.community.modularity(graph, communities, weight="weight") if edge_count and communities else 0.0
    except Exception:
        modularity = 0.0
    node_jaccard, edge_jaccard = _bootstrap_stability(payloads, params, chunks, graph)
    chapter_entropy = _community_source_entropy(graph, communities, "chapters")
    section_entropy = _community_source_entropy(graph, communities, "sections")
    chapter_coherence = _community_coherence(graph, communities, "chapters")
    semantic_margin = max(0.0, chapter_coherence - inter_edges / max(edge_count, 1))

    features = {
        "edge_evidence_coverage": _ratio(sum(1 for count in edge_chunk_counts if count > 0), edge_count),
        "node_evidence_coverage": _ratio(sum(1 for count in node_chunk_counts if count > 0), node_count),
        "multi_chunk_edge_support_rate": _ratio(sum(1 for count in edge_chunk_counts if count > 1), edge_count),
        "multi_doc_concept_support_rate": _ratio(sum(1 for node in graph.nodes if len(graph.nodes[node].get("document_ids", set()) or set()) > 1), node_count),
        "single_chunk_relation_rate": _ratio(sum(1 for count in edge_chunk_counts if count == 1), edge_count),
        "evidence_source_entropy": _entropy(edge_docs + node_docs),
        "evidence_to_edge_ratio": min(1.0, _ratio(sum(edge_chunk_counts), max(edge_count, 1) * 3)),
        "concept_support_gini": _gini([float(count) for count in node_chunk_counts]),
        "relation_support_gini": _gini([float(count) for count in edge_chunk_counts]),
        "top_concept_dominance": _ratio(max(node_chunk_counts, default=0), max(sum(node_chunk_counts), 1)),
        "low_support_node_rate": _ratio(sum(1 for count in node_chunk_counts if count <= 1), node_count),
        "low_confidence_edge_rate": _ratio(sum(1 for confidence in confidences if confidence < 0.65), edge_count),
        "confidence_entropy": _entropy([round(confidence, 1) for confidence in confidences]),
        "isolated_node_rate": _ratio(sum(1 for degree in degrees if degree == 0), node_count),
        "connected_component_count_norm": min(1.0, _ratio(len(components), max(node_count, 1))),
        "giant_component_ratio": _ratio(largest_component, node_count),
        "mean_component_size_norm": min(1.0, _ratio(float(np.mean([len(component) for component in components])) if components else 0.0, max(node_count, 1))),
        "edge_density": _ratio(edge_count, max_edges),
        "average_degree_norm": min(1.0, _ratio(float(np.mean(degrees)) if degrees else 0.0, max(node_count - 1, 1))),
        "degree_gini": _gini(degrees),
        "bridge_edge_rate": _ratio(bridge_count, edge_count),
        "modularity": max(0.0, min(1.0, float(modularity))),
        "community_count_norm": min(1.0, _ratio(len(communities), max(node_count, 1))),
        "community_size_entropy": _entropy([index for index, size in enumerate(community_sizes) for _ in range(size)]),
        "largest_community_ratio": _ratio(max(community_sizes, default=0), node_count),
        "inter_community_edge_ratio": _ratio(inter_edges, edge_count),
        "intra_community_edge_ratio": _ratio(intra_edges, edge_count),
        "conductance_mean": float(np.mean(conductance)) if conductance else 0.0,
        "conductance_p90": float(np.percentile(conductance, 90)) if conductance else 0.0,
        "average_clustering": nx.average_clustering(graph, weight="weight") if node_count > 1 else 0.0,
        "triangle_rate": min(1.0, _ratio(triangles, possible_triangles)),
        "local_closure_rate": nx.transitivity(graph) if node_count > 2 else 0.0,
        "chapter_coherence_rate": chapter_coherence,
        "section_mixing_entropy": section_entropy,
        "semantic_separation_margin": max(0.0, min(1.0, semantic_margin)),
        "cross_community_high_similarity_rate": max(0.0, min(1.0, inter_edges / max(edge_count, 1) * chapter_entropy)),
        "same_community_low_similarity_rate": max(0.0, min(1.0, intra_edges / max(edge_count, 1) * (1.0 - chapter_coherence))),
        "node_jaccard_bootstrap": node_jaccard,
        "edge_jaccard_bootstrap": edge_jaccard,
    }
    cleaned = {name: round(float(features.get(name, 0.0)), 6) for name in FEATURE_NAMES}
    summary = {
        "node_count": node_count,
        "edge_count": edge_count,
        "component_count": len(components),
        "community_count": len(communities),
        "feature_schema_version": HPO_OBJECTIVE_SCHEMA_VERSION,
        "top_features": {
            "edge_evidence_coverage": cleaned["edge_evidence_coverage"],
            "isolated_node_rate": cleaned["isolated_node_rate"],
            "giant_component_ratio": cleaned["giant_component_ratio"],
            "modularity": cleaned["modularity"],
            "average_clustering": cleaned["average_clustering"],
        },
    }
    candidate_id = hashlib.sha256(json.dumps(asdict(params.normalized()), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return CandidateEvaluation(
        candidate_id=candidate_id,
        params=params.normalized(),
        features=cleaned,
        hard_failures=hard_constraint_failures(cleaned, node_count=node_count, edge_count=edge_count),
        summary=summary,
    )


def hard_constraint_failures(features: dict[str, float], *, node_count: int, edge_count: int) -> list[str]:
    failures: list[str] = []
    if node_count <= 0:
        failures.append("empty_graph")
    if edge_count <= 0:
        failures.append("no_edges")
    if edge_count > 0 and features.get("edge_evidence_coverage", 0.0) < 0.5:
        failures.append("low_edge_evidence_coverage")
    if node_count >= 8 and features.get("isolated_node_rate", 0.0) > 0.75:
        failures.append("excessive_isolated_nodes")
    if node_count >= 12 and features.get("edge_density", 0.0) > 0.45:
        failures.append("over_dense_graph")
    if edge_count >= 4 and features.get("low_confidence_edge_rate", 0.0) > 0.75:
        failures.append("excessive_low_confidence_edges")
    return failures


def sample_candidate_pair_context(
    evaluation_a: CandidateEvaluation,
    evaluation_b: CandidateEvaluation,
    *,
    chunks: list[Chunk],
    max_tokens: int,
) -> dict[str, Any]:
    snippets = []
    budget = max(600, max_tokens)
    used = 0
    for chunk in chunks[:8]:
        text = (getattr(chunk, "snippet", None) or getattr(chunk, "content", "") or "")[:500]
        item = {
            "chunk_id": str(chunk.id),
            "chapter": getattr(chunk, "chapter", None),
            "section": getattr(chunk, "section", None),
            "snippet": text,
        }
        encoded = len(json.dumps(item, ensure_ascii=False))
        if used + encoded > budget:
            break
        snippets.append(item)
        used += encoded
    return {
        "candidate_a": {
            "id": evaluation_a.candidate_id,
            "features": evaluation_a.features,
            "hard_failures": evaluation_a.hard_failures,
            "summary": evaluation_a.summary,
        },
        "candidate_b": {
            "id": evaluation_b.candidate_id,
            "features": evaluation_b.features,
            "hard_failures": evaluation_b.hard_failures,
            "summary": evaluation_b.summary,
        },
        "evidence_snippets": snippets,
        "instructions": "Prefer the graph with stronger evidence grounding, coherent communities, low overmerge/fragmentation risk, and stable structure.",
    }


async def judge_candidate_pair(
    evaluation_a: CandidateEvaluation,
    evaluation_b: CandidateEvaluation,
    *,
    chunks: list[Chunk],
    max_tokens: int,
    provider: ChatProvider | None = None,
) -> dict[str, Any]:
    chat = provider or ChatProvider()
    context = sample_candidate_pair_context(evaluation_a, evaluation_b, chunks=chunks, max_tokens=max_tokens)
    system_prompt = (
        "You are a strict pairwise judge for knowledge graph HPO. "
        "Choose the candidate graph that is more reliable for evidence-grounded course RAG. "
        "Return strict JSON with keys: winner ('A', 'B', or 'tie'), confidence (0..1), reasons (list), safety_flags (list). "
        "Do not infer from hidden hyperparameter values; judge only the supplied graph quality evidence."
    )
    user_prompt = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
    result = await chat.classify_json(system_prompt, user_prompt, fallback=None)
    if not isinstance(result, dict):
        raise RuntimeError("Invalid HPO judge response")
    winner = str(result.get("winner", "")).upper()
    if winner not in {"A", "B", "TIE"}:
        raise RuntimeError("Invalid HPO judge winner")
    confidence = max(0.0, min(1.0, _safe_float(result.get("confidence"), 0.0)))
    return {
        "winner": "tie" if winner == "TIE" else winner,
        "confidence": confidence,
        "reasons": list(result.get("reasons", []))[:8] if isinstance(result.get("reasons"), list) else [str(result.get("reasons", ""))],
        "safety_flags": list(result.get("safety_flags", []))[:8] if isinstance(result.get("safety_flags"), list) else [],
        "prompt_version": HPO_JUDGE_PROMPT_VERSION,
        "raw_response": result,
    }


def normalisation_stats(evaluations: list[CandidateEvaluation]) -> dict[str, dict[str, float]]:
    matrix = np.array([[evaluation.features.get(name, 0.0) for name in FEATURE_NAMES] for evaluation in evaluations], dtype=float)
    if matrix.size == 0:
        matrix = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return {name: {"mean": float(means[index]), "std": float(stds[index])} for index, name in enumerate(FEATURE_NAMES)}


def _normalised_vector(features: dict[str, float], stats: dict[str, dict[str, float]]) -> np.ndarray:
    return np.array(
        [
            (float(features.get(name, 0.0)) - stats.get(name, {}).get("mean", 0.0)) / max(stats.get(name, {}).get("std", 1.0), 1e-6)
            for name in FEATURE_NAMES
        ],
        dtype=float,
    )


def learn_surrogate_objective(
    labels: list[dict[str, Any]],
    evaluations: dict[str, CandidateEvaluation],
) -> dict[str, Any]:
    effective = [
        label
        for label in labels
        if label.get("winner") in {"A", "B"} and float(label.get("confidence", 0.0) or 0.0) > 0.0
    ]
    if not effective:
        raise RuntimeError("No effective HPO judge labels")
    stats = normalisation_stats(list(evaluations.values()))
    accumulator = np.zeros(len(FEATURE_NAMES), dtype=float)
    total_weight = 0.0
    for label in effective:
        eval_a = evaluations[str(label["candidate_a_id"])]
        eval_b = evaluations[str(label["candidate_b_id"])]
        vec_a = _normalised_vector(eval_a.features, stats)
        vec_b = _normalised_vector(eval_b.features, stats)
        confidence = max(0.0, min(1.0, float(label.get("confidence", 0.0) or 0.0)))
        if label["winner"] == "A":
            accumulator += confidence * (vec_a - vec_b)
        else:
            accumulator += confidence * (vec_b - vec_a)
        total_weight += confidence
    if total_weight <= 0:
        raise RuntimeError("HPO judge labels have zero total confidence")
    weights = accumulator / total_weight
    norm = float(np.linalg.norm(weights))
    if norm < 1e-9:
        raise RuntimeError("HPO judge labels produced a zero objective vector")
    weights = weights / norm
    return {
        "schema_version": HPO_OBJECTIVE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "weights": {name: round(float(weights[index]), 8) for index, name in enumerate(FEATURE_NAMES)},
        "normalization": stats,
        "label_count": len(effective),
        "training_audit": {
            "raw_label_count": len(labels),
            "effective_label_count": len(effective),
            "objective_norm": round(norm, 8),
        },
    }


def score_with_learned_objective(features: dict[str, float], objective_model: dict[str, Any]) -> float:
    stats = objective_model.get("normalization") or {}
    weights = objective_model.get("weights") or {}
    vector = _normalised_vector(features, stats)
    weight_vector = np.array([float(weights.get(name, 0.0) or 0.0) for name in FEATURE_NAMES], dtype=float)
    return float(np.dot(vector, weight_vector))


def feature_summary(features: dict[str, float]) -> dict[str, float]:
    keys = [
        "edge_evidence_coverage",
        "node_evidence_coverage",
        "isolated_node_rate",
        "giant_component_ratio",
        "edge_density",
        "modularity",
        "average_clustering",
        "semantic_separation_margin",
        "node_jaccard_bootstrap",
        "edge_jaccard_bootstrap",
    ]
    return {key: round(float(features.get(key, 0.0)), 4) for key in keys}


def select_pair_indices(candidate_count: int, max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for distance in range(1, candidate_count):
        for index in range(candidate_count - distance):
            pairs.append((index, index + distance))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


JudgeFunction = Callable[[CandidateEvaluation, CandidateEvaluation, list[Chunk], int], Any]
