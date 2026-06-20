from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import AutoTpeRun, AutoTpeTrial, Chunk, Document, KnowledgeBase
from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
from app.services.context_graph import (
    RELATION_PROTOCOL_VERSION,
    RelationEdgeCandidate,
    compute_chunk_scope_hash,
    dense_graph_operating_point,
    edge_distance_protocol_hash,
    edge_type_calibration_protocol_hash,
    relation_edge_candidates,
    runtime_settings_snapshot,
    stable_hash,
)
from app.services.ingestion_logs import emit_ingestion_log
from app.services.runtime_settings import refresh_runtime_settings_if_needed


AUTO_TPE_PROTOCOL = "auto_tpe_lightweight_graph_operating_point_v1"
TPE_THETA_KEYS = (
    "dense_knn_k_min",
    "dense_knn_k_max",
    "dense_reverse_b_min_base",
    "dense_reverse_b_max_base",
    "dense_reverse_b_min_doc",
    "dense_reverse_b_max_doc",
    "dense_reverse_b_min_lang",
    "dense_reverse_b_max_lang",
    "dense_min_cosine",
    "dense_strong_cosine",
    "cross_doc_out_quota_min",
    "cross_doc_out_quota_max",
    "cross_doc_min_cosine",
    "cross_language_out_quota_min",
    "cross_language_out_quota_max",
    "cross_language_min_cosine",
)
TPE_TERMINAL_TRIAL_STATES = {"completed", "blocked", "failed"}


def _now() -> datetime:
    return datetime.utcnow()


def _protocol_hash() -> str:
    return stable_hash(
        {
            "auto_tpe_protocol": AUTO_TPE_PROTOCOL,
            "relation_protocol": RELATION_PROTOCOL_VERSION,
            "edge_distance_protocol_hash": edge_distance_protocol_hash(),
            "edge_type_calibration_protocol_hash": edge_type_calibration_protocol_hash(),
        }
    )


def _runtime_hash() -> str:
    return stable_hash(runtime_settings_snapshot())


def _theta_hash(theta: dict[str, Any]) -> str:
    return stable_hash({key: theta.get(key) for key in sorted(theta)})


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _bounded_float(value: Any, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def normalize_theta(theta: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(theta)
    int_bounds = {
        "dense_knn_k_min": (1, 32),
        "dense_knn_k_max": (1, 64),
        "dense_reverse_b_min_base": (1, 32),
        "dense_reverse_b_max_base": (1, 64),
        "dense_reverse_b_min_doc": (0, 32),
        "dense_reverse_b_max_doc": (1, 64),
        "dense_reverse_b_min_lang": (0, 32),
        "dense_reverse_b_max_lang": (1, 64),
        "cross_doc_out_quota_min": (0, 32),
        "cross_doc_out_quota_max": (1, 64),
        "cross_language_out_quota_min": (0, 32),
        "cross_language_out_quota_max": (1, 64),
    }
    float_bounds = {
        "dense_min_cosine": (0.05, 0.95),
        "dense_strong_cosine": (0.06, 0.99),
        "cross_doc_min_cosine": (0.05, 0.95),
        "cross_language_min_cosine": (0.05, 0.95),
    }
    for key, (lower, upper) in int_bounds.items():
        normalized[key] = _bounded_int(normalized.get(key, lower), lower, upper)
    for key, (lower, upper) in float_bounds.items():
        normalized[key] = round(_bounded_float(normalized.get(key, lower), lower, upper), 4)
    for min_key, max_key in (
        ("dense_knn_k_min", "dense_knn_k_max"),
        ("dense_reverse_b_min_base", "dense_reverse_b_max_base"),
        ("dense_reverse_b_min_doc", "dense_reverse_b_max_doc"),
        ("dense_reverse_b_min_lang", "dense_reverse_b_max_lang"),
        ("cross_doc_out_quota_min", "cross_doc_out_quota_max"),
        ("cross_language_out_quota_min", "cross_language_out_quota_max"),
    ):
        if int(normalized[min_key]) > int(normalized[max_key]):
            normalized[max_key] = normalized[min_key]
    min_threshold = max(
        float(normalized["dense_min_cosine"]),
        float(normalized["cross_doc_min_cosine"]),
        float(normalized["cross_language_min_cosine"]),
    )
    if float(normalized["dense_strong_cosine"]) <= min_threshold:
        normalized["dense_strong_cosine"] = round(min(0.99, min_threshold + 0.08), 4)
    normalized["protocol"] = "dense_dynamic_knn_bridge_quota_v1"
    normalized["optimizer"] = "auto_tpe_lightweight"
    return normalized


def preflight_theta(theta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in TPE_THETA_KEYS:
        if key not in theta:
            reasons.append(f"missing:{key}")
    for min_key, max_key in (
        ("dense_knn_k_min", "dense_knn_k_max"),
        ("dense_reverse_b_min_base", "dense_reverse_b_max_base"),
        ("dense_reverse_b_min_doc", "dense_reverse_b_max_doc"),
        ("dense_reverse_b_min_lang", "dense_reverse_b_max_lang"),
        ("cross_doc_out_quota_min", "cross_doc_out_quota_max"),
        ("cross_language_out_quota_min", "cross_language_out_quota_max"),
    ):
        if int(theta.get(min_key, 0)) > int(theta.get(max_key, 0)):
            reasons.append(f"invalid_range:{min_key}:{max_key}")
    strong = float(theta.get("dense_strong_cosine", 0.0))
    for key in ("dense_min_cosine", "cross_doc_min_cosine", "cross_language_min_cosine"):
        value = float(theta.get(key, -1.0))
        if value <= 0.0 or value >= 1.0:
            reasons.append(f"invalid_threshold:{key}")
        if strong <= value:
            reasons.append(f"strong_threshold_not_above:{key}")
    if strong <= 0.0 or strong >= 1.0:
        reasons.append("invalid_threshold:dense_strong_cosine")
    if theta.get("protocol") != "dense_dynamic_knn_bridge_quota_v1":
        reasons.append("invalid_protocol")
    return reasons


def _random_theta(base: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    theta = dict(base)
    int_ranges = {
        "dense_knn_k_min": (1, 12),
        "dense_knn_k_max": (4, 32),
        "dense_reverse_b_min_base": (1, 12),
        "dense_reverse_b_max_base": (4, 32),
        "dense_reverse_b_min_doc": (0, 8),
        "dense_reverse_b_max_doc": (2, 24),
        "dense_reverse_b_min_lang": (0, 8),
        "dense_reverse_b_max_lang": (2, 24),
        "cross_doc_out_quota_min": (0, 8),
        "cross_doc_out_quota_max": (2, 24),
        "cross_language_out_quota_min": (0, 8),
        "cross_language_out_quota_max": (2, 24),
    }
    float_ranges = {
        "dense_min_cosine": (0.18, 0.72),
        "dense_strong_cosine": (0.42, 0.92),
        "cross_doc_min_cosine": (0.18, 0.72),
        "cross_language_min_cosine": (0.18, 0.72),
    }
    for key, (lower, upper) in int_ranges.items():
        theta[key] = rng.randint(lower, upper)
    for key, (lower, upper) in float_ranges.items():
        theta[key] = round(rng.uniform(lower, upper), 4)
    return normalize_theta(theta)


def _sampler_state_hash(run: AutoTpeRun, completed_trials: list[AutoTpeTrial], next_index: int) -> str:
    return stable_hash(
        {
            "run_id": run.id,
            "scope": {
                "knowledge_base_id": run.knowledge_base_id,
                "chunk_version": run.chunk_version,
                "chat_model": run.chat_model,
                "embedding_model": run.embedding_model,
                "embedding_text_version": run.embedding_text_version,
            },
            "next_index": next_index,
            "completed": [
                {
                    "trial_index": trial.trial_index,
                    "theta_hash": trial.theta_hash,
                    "status": trial.status,
                    "objective_score": trial.objective_score,
                    "failure_code": trial.failure_code,
                }
                for trial in sorted(completed_trials, key=lambda item: item.trial_index)
            ],
        }
    )


def _mean_for(trials: list[AutoTpeTrial], key: str, base: dict[str, Any]) -> float:
    values = [float((trial.sampled_theta_json or {}).get(key, base.get(key, 0))) for trial in trials]
    return sum(values) / max(len(values), 1)


def _sample_tpe_theta(run: AutoTpeRun, completed_trials: list[AutoTpeTrial], trial_index: int, base_theta: dict[str, Any]) -> tuple[dict[str, Any], str]:
    refresh_runtime_settings_if_needed(force=True)
    settings = get_settings()
    sampler_hash = _sampler_state_hash(run, completed_trials, trial_index)
    startup = int(settings.tpe_startup_random_trials)
    valid = [trial for trial in completed_trials if trial.status == "completed" and trial.objective_score is not None]
    if trial_index <= startup or not valid:
        return _random_theta(base_theta, int(stable_hash({"run": run.id, "trial": trial_index})[:8], 16)), sampler_hash

    ordered = sorted(valid, key=lambda trial: float(trial.objective_score or 0.0), reverse=True)
    split = max(1, int(math.ceil(len(ordered) * float(settings.tpe_good_quantile_gamma))))
    good = ordered[:split]
    bad = ordered[split:] or ordered[-1:]
    rng = random.Random(int(sampler_hash[:8], 16))
    candidate_pool_size = max(1, int(settings.tpe_candidate_pool_size))
    best_candidate: dict[str, Any] | None = None
    best_ratio = -1.0
    for _ in range(candidate_pool_size):
        candidate = dict(base_theta)
        for key in TPE_THETA_KEYS:
            good_mean = _mean_for(good, key, base_theta)
            bad_mean = _mean_for(bad, key, base_theta)
            spread = max(abs(good_mean - bad_mean), 0.05 if "cosine" in key else 1.0)
            candidate[key] = good_mean + rng.uniform(-0.35, 0.35) * spread
        candidate = normalize_theta(candidate)
        if preflight_theta(candidate):
            continue
        ratio = 0.0
        for key in TPE_THETA_KEYS:
            value = float(candidate[key])
            good_distance = abs(value - _mean_for(good, key, base_theta)) + 1e-6
            bad_distance = abs(value - _mean_for(bad, key, base_theta)) + 1e-6
            ratio += bad_distance / good_distance
        if ratio > best_ratio:
            best_candidate = candidate
            best_ratio = ratio
    return best_candidate or _random_theta(base_theta, int(stable_hash({"fallback": sampler_hash})[:8], 16)), sampler_hash


def _probe_chunks(chunks: list[Chunk], budget: int) -> tuple[list[Chunk], str]:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index, chunk.id))
    if len(ordered) <= budget:
        probes = ordered
    else:
        step = max(1, len(ordered) // budget)
        probes = [ordered[index] for index in range(0, len(ordered), step)][:budget]
    return probes, stable_hash(
        [
            {
                "chunk_id": chunk.id,
                "text_hash": chunk.text_hash,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "span": [chunk.char_start, chunk.char_end],
            }
            for chunk in probes
        ]
    )


def _candidate_adjacency_hash(candidates: dict[tuple[str, str, str], RelationEdgeCandidate]) -> str:
    return stable_hash(
        [
            {
                "source": candidate.source_chunk_id,
                "target": candidate.target_chunk_id,
                "type": candidate.edge_type,
                "strength": round(float(candidate.raw_strength), 6),
            }
            for candidate in sorted(candidates.values(), key=lambda item: (item.source_chunk_id, item.target_chunk_id, item.edge_type))
        ]
    )


def _adjacency(candidates: dict[tuple[str, str, str], RelationEdgeCandidate]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates.values():
        adjacency[candidate.source_chunk_id].add(candidate.target_chunk_id)
        adjacency[candidate.target_chunk_id].add(candidate.source_chunk_id)
    return adjacency


def _reachable_within_two(source_id: str, target_ids: set[str], adjacency: dict[str, set[str]]) -> bool:
    if not target_ids:
        return False
    seen = {source_id}
    queue: deque[tuple[str, int]] = deque([(source_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= 2:
            continue
        for neighbor_id in adjacency.get(node_id, set()):
            if neighbor_id in target_ids:
                return True
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
    return False


def _entropy_ratio(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = float(len(values))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return max(0.0, min(1.0, entropy / math.log(len(counts))))


def _component(name: str, value: float, numerator: float, denominator: float, probe_hash: str | None) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "numerator": round(float(numerator), 6),
        "denominator": round(float(denominator), 6),
        "sample_count": round(float(denominator), 6),
        "probe_set_hash": probe_hash,
        "protocol": AUTO_TPE_PROTOCOL,
        "component": name,
    }


def _degree_stats(chunks: list[Chunk], candidates: dict[tuple[str, str, str], RelationEdgeCandidate]) -> dict[str, float]:
    degree_by_id = {chunk.id: 0 for chunk in chunks}
    for candidate in candidates.values():
        degree_by_id[candidate.source_chunk_id] = degree_by_id.get(candidate.source_chunk_id, 0) + 1
        degree_by_id[candidate.target_chunk_id] = degree_by_id.get(candidate.target_chunk_id, 0) + 1
    degrees = sorted(degree_by_id.values())
    if not degrees:
        return {"isolated": 0.0, "isolated_ratio": 0.0, "median": 0.0, "p95": 0.0, "hubness_ratio": 0.0}
    p95_index = min(len(degrees) - 1, int(math.ceil(len(degrees) * 0.95)) - 1)
    median = degrees[len(degrees) // 2]
    p95 = degrees[p95_index]
    isolated = sum(1 for value in degrees if value <= 0)
    return {
        "isolated": float(isolated),
        "isolated_ratio": isolated / max(len(degrees), 1),
        "median": float(median),
        "p95": float(p95),
        "hubness_ratio": p95 / max(float(median), 1.0),
    }


def evaluate_candidate_trial(
    db: Session,
    chunks: list[Chunk],
    candidates: dict[tuple[str, str, str], RelationEdgeCandidate],
    diagnostics: dict[str, Any],
    elapsed_ms: float,
) -> tuple[dict[str, Any], dict[str, Any], float, str | None, str]:
    settings = get_settings()
    probes, probe_hash = _probe_chunks(chunks, int(settings.tpe_probe_query_budget))
    if len(probes) < 2 or len(chunks) < 2:
        return {}, {}, 0.0, "insufficient_evaluation", probe_hash
    adjacency = _adjacency(candidates)
    degree_stats = _degree_stats(chunks, candidates)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    edge_count = len(candidates)
    chunk_count = max(1, len(chunks))
    edge_density = edge_count / chunk_count

    evidence_hits = sum(1 for chunk in probes if adjacency.get(chunk.id))
    structure_denominator = 0
    structure_hits = 0
    for chunk in probes:
        neighbors = {
            other.id
            for other in chunks
            if other.id != chunk.id
            and other.document_id == chunk.document_id
            and (abs(int(other.chunk_index) - int(chunk.chunk_index)) == 1 or (chunk.section_path and other.section_path == chunk.section_path))
        }
        if not neighbors:
            continue
        structure_denominator += 1
        if _reachable_within_two(chunk.id, neighbors, adjacency):
            structure_hits += 1
    structure_recovery_rate = structure_hits / max(structure_denominator, 1)

    non_isolated_ids = {chunk.id for chunk in chunks if adjacency.get(chunk.id)}
    covered_documents = {chunk_by_id[chunk_id].document_id for chunk_id in non_isolated_ids if chunk_id in chunk_by_id}
    all_documents = {chunk.document_id for chunk in chunks}
    component_coverage = 0.55 * (len(non_isolated_ids) / chunk_count) + 0.45 * (len(covered_documents) / max(len(all_documents), 1))
    precision_hits = sum(
        1
        for candidate in candidates.values()
        if bool((candidate.features_json.get("normalization_stats") or {}).get("mutual"))
        or bool((candidate.features_json.get("normalization_stats") or {}).get("reverse_accepted"))
        or float(candidate.features_json.get("cosine") or 0.0) >= float((candidate.features_json.get("normalization_stats") or {}).get("strong_threshold") or 1.0)
    )
    edge_precision_proxy = precision_hits / max(edge_count, 1)
    bridge_edges = [candidate for candidate in candidates.values() if candidate.edge_type in {"dense_cross_document_bridge", "dense_cross_language_bridge"}]
    possible_bridge_sources = {
        chunk.id
        for chunk in chunks
        if any(other.document_id != chunk.document_id for other in chunks) or any(_language(db, other) != _language(db, chunk) for other in chunks)
    }
    bridge_opportunity_coverage = len({edge.source_chunk_id for edge in bridge_edges} | {edge.target_chunk_id for edge in bridge_edges}) / max(len(possible_bridge_sources), 1)
    path_diversity = (
        0.34 * _entropy_ratio([candidate.edge_type for candidate in candidates.values()])
        + 0.33 * _entropy_ratio([str(candidate.features_json.get("source_language") or "unknown") for candidate in candidates.values()])
        + 0.33 * _entropy_ratio([str(candidate.features_json.get("bridge_quota_reason") or "base") for candidate in candidates.values()])
    )
    evidence_recall_proxy = evidence_hits / max(len(probes), 1)

    max_edge_density = float(settings.operating_point_hard_gate_max_edge_density)
    max_isolated_ratio = float(settings.operating_point_hard_gate_max_isolated_ratio)
    max_hubness_ratio = float(settings.operating_point_hard_gate_max_hubness_ratio)
    min_structure_recovery = float(settings.operating_point_hard_gate_min_structure_recovery_rate)
    max_latency = float(settings.operating_point_hard_gate_max_candidate_latency_p95_ms)
    hard_gate = {
        "edge_density": {"value": round(edge_density, 6), "threshold": max_edge_density, "passed": edge_density <= max_edge_density, "direction": "max"},
        "isolated_ratio": {
            "value": round(float(degree_stats["isolated_ratio"]), 6),
            "threshold": max_isolated_ratio,
            "passed": float(degree_stats["isolated_ratio"]) <= max_isolated_ratio,
            "direction": "max",
        },
        "hubness_ratio": {
            "value": round(float(degree_stats["hubness_ratio"]), 6),
            "threshold": max_hubness_ratio,
            "passed": float(degree_stats["hubness_ratio"]) <= max_hubness_ratio,
            "direction": "max",
        },
        "structure_recovery_rate": {
            "value": round(structure_recovery_rate, 6),
            "threshold": min_structure_recovery,
            "passed": structure_recovery_rate >= min_structure_recovery,
            "direction": "min",
        },
        "candidate_latency_p95_ms": {
            "value": round(float(elapsed_ms), 3),
            "threshold": max_latency,
            "passed": float(elapsed_ms) <= max_latency,
            "direction": "max",
        },
    }
    failed = [name for name, gate in hard_gate.items() if not gate["passed"]]
    components = {
        "evidence_recall_proxy": _component("evidence_recall_proxy", evidence_recall_proxy, evidence_hits, len(probes), probe_hash),
        "structure_recovery_rate": _component("structure_recovery_rate", structure_recovery_rate, structure_hits, max(structure_denominator, 1), probe_hash),
        "component_coverage": _component("component_coverage", component_coverage, len(non_isolated_ids), chunk_count, probe_hash),
        "edge_precision_proxy": _component("edge_precision_proxy", edge_precision_proxy, precision_hits, max(edge_count, 1), probe_hash),
        "bridge_opportunity_coverage": _component(
            "bridge_opportunity_coverage",
            bridge_opportunity_coverage,
            len({edge.source_chunk_id for edge in bridge_edges} | {edge.target_chunk_id for edge in bridge_edges}),
            max(len(possible_bridge_sources), 1),
            probe_hash,
        ),
        "path_diversity": _component("path_diversity", path_diversity, path_diversity, 1, probe_hash),
        "hubness_penalty": _component("hubness_penalty", min(1.0, float(degree_stats["hubness_ratio"]) / max(max_hubness_ratio, 1e-6)), degree_stats["hubness_ratio"], max_hubness_ratio, probe_hash),
        "density_penalty": _component("density_penalty", min(1.0, edge_density / max(max_edge_density, 1e-6)), edge_density, max_edge_density, probe_hash),
        "latency_penalty": _component("latency_penalty", min(1.0, float(elapsed_ms) / max(max_latency, 1e-6)), elapsed_ms, max_latency, probe_hash),
    }
    score = (
        0.26 * evidence_recall_proxy
        + 0.18 * structure_recovery_rate
        + 0.16 * component_coverage
        + 0.12 * edge_precision_proxy
        + 0.10 * bridge_opportunity_coverage
        + 0.08 * path_diversity
        - 0.12 * components["hubness_penalty"]["value"]
        - 0.10 * components["density_penalty"]["value"]
        - 0.06 * components["latency_penalty"]["value"]
    )
    failure_code = "hard_gate_failed:" + ",".join(failed) if failed else None
    components["candidate_diagnostics"] = {"value": 1.0, "protocol": AUTO_TPE_PROTOCOL, "candidate_count": edge_count, "diagnostics": diagnostics}
    return hard_gate, components, round(score, 6), failure_code, probe_hash


def _language(db: Session, chunk: Chunk) -> str:
    document = db.get(Document, chunk.document_id)
    return (document.language if document and document.language else "unknown").lower()


def _latest_completed_theta(db: Session, knowledge_base_id: str, chat_model: str, embedding_model: str, embedding_text_version: str) -> dict[str, Any] | None:
    run = db.scalar(
        select(AutoTpeRun)
        .where(
            AutoTpeRun.knowledge_base_id == knowledge_base_id,
            AutoTpeRun.status == "completed",
            AutoTpeRun.chat_model == chat_model,
            AutoTpeRun.embedding_model == embedding_model,
            AutoTpeRun.embedding_text_version == embedding_text_version,
        )
        .order_by(AutoTpeRun.chunk_version.desc(), AutoTpeRun.completed_at.desc())
    )
    if run and run.selected_theta_json:
        return dict(run.selected_theta_json)
    return None


def select_auto_tpe_operating_point(
    db: Session,
    knowledge_base_id: str,
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    *,
    fallback_operating_point: dict[str, Any],
    batch_id: str | None,
    chunk_version_incremented: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    refresh_runtime_settings_if_needed(force=True)
    settings = get_settings()
    chunk_version = max(chunk.chunk_version for chunk in chunks)
    chunk_scope_hash = compute_chunk_scope_hash(chunks)
    fallback_theta = dict(fallback_operating_point or dense_graph_operating_point())
    base_theta = normalize_theta(fallback_theta)
    if fallback_theta.get("optimizer"):
        base_theta["optimizer"] = str(fallback_theta["optimizer"])
    previous_theta = _latest_completed_theta(db, knowledge_base_id, settings.chat_model, settings.embedding_model, CURRENT_EMBEDDING_TEXT_VERSION)
    if previous_theta:
        base_theta = normalize_theta(previous_theta)

    if not settings.enable_auto_tpe:
        emit_ingestion_log(batch_id, "auto_tpe_skipped", "自动 TPE 未开启，沿用上一版或默认工作点", reason="disabled", chunk_version=chunk_version)
        return base_theta, {"status": "skipped", "reason": "disabled", "chunk_version": chunk_version}
    if not chunk_version_incremented:
        emit_ingestion_log(batch_id, "auto_tpe_skipped", "chunk 最高版本号未递增，跳过自动 TPE", reason="chunk_version_not_incremented", chunk_version=chunk_version)
        return base_theta, {"status": "skipped", "reason": "chunk_version_not_incremented", "chunk_version": chunk_version}

    run = AutoTpeRun(
        knowledge_base_id=knowledge_base_id,
        batch_id=batch_id,
        chunk_version=chunk_version,
        chunk_scope_hash=chunk_scope_hash,
        graph_operating_point_protocol="dense_dynamic_knn_bridge_quota_v1",
        protocol_hash=_protocol_hash(),
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        status="running",
        trigger_reason="chunk_version_incremented",
        trial_budget=int(settings.tpe_trial_budget),
        startup_random_trials=int(settings.tpe_startup_random_trials),
        good_quantile_gamma=float(settings.tpe_good_quantile_gamma),
        probe_query_budget=int(settings.tpe_probe_query_budget),
        candidate_pool_size=int(settings.tpe_candidate_pool_size),
        runtime_settings_hash=_runtime_hash(),
        diagnostics_json={
            "protocol": AUTO_TPE_PROTOCOL,
            "no_llm": True,
            "no_embedding": True,
            "no_mid_coarse": True,
            "no_shadow_graph": True,
        },
        started_at=_now(),
    )
    db.add(run)
    db.flush()
    emit_ingestion_log(
        batch_id,
        "auto_tpe_started",
        "自动 TPE 轻量调参开始",
        run_id=run.id,
        chunk_version=chunk_version,
        trial_budget=run.trial_budget,
        embedding_model=run.embedding_model,
        chat_model=run.chat_model,
    )

    completed_trials: list[AutoTpeTrial] = []
    for trial_index in range(1, int(run.trial_budget) + 1):
        refresh_runtime_settings_if_needed(force=True)
        theta, sampler_hash = _sample_tpe_theta(run, completed_trials, trial_index, base_theta)
        invalid = preflight_theta(theta)
        trial = AutoTpeTrial(
            run_id=run.id,
            knowledge_base_id=knowledge_base_id,
            trial_index=trial_index,
            sampled_theta_json=theta,
            theta_hash=_theta_hash(theta),
            sampler_state_hash=sampler_hash,
            status="blocked" if invalid else "running",
            failure_code="invalid_theta_preflight" if invalid else None,
            diagnostics_json={"invalid_theta_reasons": invalid, "sampler": "tpe_l_over_g_lightweight_v1"},
            started_at=_now(),
        )
        db.add(trial)
        run.sampler_state_hash = sampler_hash
        db.flush()
        emit_ingestion_log(batch_id, "auto_tpe_trial_started", f"自动 TPE trial {trial_index} 开始", run_id=run.id, trial_id=trial.id, trial_index=trial_index)
        if invalid:
            trial.finished_at = _now()
            emit_ingestion_log(batch_id, "auto_tpe_trial_blocked", f"自动 TPE trial {trial_index} 参数预检失败", run_id=run.id, trial_id=trial.id, reasons=invalid)
            completed_trials.append(trial)
            continue
        started = time.perf_counter()
        try:
            candidates, diagnostics = relation_edge_candidates(db, chunks, vectors, theta)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate_hash = _candidate_adjacency_hash(candidates)
            if elapsed_ms > float(get_settings().tpe_trial_timeout_seconds) * 1000.0:
                trial.status = "failed"
                trial.failure_code = "trial_timeout"
                trial.candidate_adjacency_hash = candidate_hash
                trial.diagnostics_json = {**(trial.diagnostics_json or {}), "elapsed_ms": round(elapsed_ms, 3), "candidate_count": len(candidates)}
            else:
                hard_gate, components, score, failure_code, probe_hash = evaluate_candidate_trial(db, chunks, candidates, diagnostics, elapsed_ms)
                trial.candidate_adjacency_hash = candidate_hash
                trial.probe_set_hash = probe_hash
                trial.hard_gate_json = hard_gate
                trial.objective_components_json = components
                trial.objective_score = score
                trial.status = "blocked" if failure_code else "completed"
                trial.failure_code = failure_code
                trial.diagnostics_json = {**(trial.diagnostics_json or {}), "elapsed_ms": round(elapsed_ms, 3), "candidate_count": len(candidates)}
            trial.finished_at = _now()
            event = "auto_tpe_trial_blocked" if trial.status != "completed" else "auto_tpe_trial_completed"
            emit_ingestion_log(
                batch_id,
                event,
                f"自动 TPE trial {trial_index} {'完成' if trial.status == 'completed' else '被阻断'}",
                run_id=run.id,
                trial_id=trial.id,
                trial_index=trial_index,
                objective_score=trial.objective_score,
                failure_code=trial.failure_code,
                hard_gate=trial.hard_gate_json,
            )
        except Exception as exc:
            trial.status = "failed"
            trial.failure_code = "candidate_simulation_failed"
            trial.finished_at = _now()
            trial.diagnostics_json = {**(trial.diagnostics_json or {}), "error": str(exc)}
            emit_ingestion_log(batch_id, "auto_tpe_trial_blocked", f"自动 TPE trial {trial_index} 模拟失败：{exc}", run_id=run.id, trial_id=trial.id, error=str(exc))
        completed_trials.append(trial)
        db.flush()

    valid_trials = [trial for trial in completed_trials if trial.status == "completed" and trial.objective_score is not None]
    if not valid_trials:
        failure_codes = {trial.failure_code for trial in completed_trials}
        if failure_codes == {"insufficient_evaluation"}:
            run.status = "skipped"
            run.completed_at = _now()
            run.failure_code = "insufficient_evaluation"
            run.blocking_reasons_json = ["insufficient_evaluation"]
            run.selected_theta_json = base_theta
            run.selected_theta_hash = _theta_hash(base_theta)
            db.flush()
            emit_ingestion_log(
                batch_id,
                "auto_tpe_skipped",
                "自动 TPE probe 不足，沿用上一版或默认工作点",
                run_id=run.id,
                chunk_version=chunk_version,
                reason="insufficient_evaluation",
            )
            return base_theta, {
                "status": "skipped",
                "reason": "insufficient_evaluation",
                "run_id": run.id,
                "selected_theta_hash": run.selected_theta_hash,
                "chunk_version": chunk_version,
            }
        run.status = "failed"
        run.completed_at = _now()
        run.failure_code = "no_trial_passed_hard_gate"
        run.blocking_reasons_json = ["no_trial_passed_hard_gate"]
        run.last_error = "自动 TPE 没有 trial 通过 hard gate"
        db.flush()
        emit_ingestion_log(batch_id, "auto_tpe_failed", run.last_error, run_id=run.id, chunk_version=chunk_version, blocking_reasons=run.blocking_reasons_json)
        raise RuntimeError(run.last_error)

    best = max(valid_trials, key=lambda trial: float(trial.objective_score or 0.0))
    run.status = "completed"
    run.best_trial_id = best.id
    run.best_objective_score = best.objective_score
    run.selected_theta_json = dict(best.sampled_theta_json or {})
    run.selected_theta_hash = best.theta_hash
    run.probe_set_hash = best.probe_set_hash
    run.hard_gate_json = dict(best.hard_gate_json or {})
    run.objective_components_json = dict(best.objective_components_json or {})
    run.completed_at = _now()
    db.flush()
    emit_ingestion_log(
        batch_id,
        "auto_tpe_best_theta_selected",
        "自动 TPE 已选择最佳底层关系图工作点",
        run_id=run.id,
        trial_id=best.id,
        objective_score=best.objective_score,
        theta_hash=best.theta_hash,
        chunk_version=chunk_version,
    )
    return dict(run.selected_theta_json or base_theta), {
        "status": run.status,
        "run_id": run.id,
        "best_trial_id": best.id,
        "selected_theta_hash": run.selected_theta_hash,
        "objective_score": run.best_objective_score,
        "chunk_version": chunk_version,
        "protocol_hash": run.protocol_hash,
    }


def mark_auto_tpe_relation_state(db: Session, run_id: str, relation_state_id: str) -> None:
    run = db.get(AutoTpeRun, run_id)
    if run is not None:
        run.chunk_relation_graph_state_id = relation_state_id
        db.flush()


def summarize_auto_tpe_trial(trial: AutoTpeTrial) -> dict[str, Any]:
    return {
        "trial_id": trial.id,
        "run_id": trial.run_id,
        "trial_index": trial.trial_index,
        "status": trial.status,
        "theta_hash": trial.theta_hash,
        "sampler_state_hash": trial.sampler_state_hash,
        "candidate_adjacency_hash": trial.candidate_adjacency_hash,
        "probe_set_hash": trial.probe_set_hash,
        "objective_score": trial.objective_score,
        "hard_gate": trial.hard_gate_json or {},
        "objective_components": trial.objective_components_json or {},
        "failure_code": trial.failure_code,
        "diagnostics": trial.diagnostics_json or {},
        "started_at": trial.started_at,
        "finished_at": trial.finished_at,
    }


def summarize_auto_tpe_run(db: Session, run: AutoTpeRun) -> dict[str, Any]:
    trials = list(db.scalars(select(AutoTpeTrial).where(AutoTpeTrial.run_id == run.id).order_by(AutoTpeTrial.trial_index.asc())).all())
    return {
        "run_id": run.id,
        "knowledge_base_id": run.knowledge_base_id,
        "batch_id": run.batch_id,
        "chunk_relation_graph_state_id": run.chunk_relation_graph_state_id,
        "chunk_version": run.chunk_version,
        "chunk_scope_hash": run.chunk_scope_hash,
        "graph_operating_point_protocol": run.graph_operating_point_protocol,
        "protocol_hash": run.protocol_hash,
        "chat_model": run.chat_model,
        "embedding_model": run.embedding_model,
        "embedding_text_version": run.embedding_text_version,
        "status": run.status,
        "trigger_reason": run.trigger_reason,
        "trial_budget": run.trial_budget,
        "startup_random_trials": run.startup_random_trials,
        "good_quantile_gamma": run.good_quantile_gamma,
        "probe_query_budget": run.probe_query_budget,
        "candidate_pool_size": run.candidate_pool_size,
        "best_trial_id": run.best_trial_id,
        "best_objective_score": run.best_objective_score,
        "selected_theta_hash": run.selected_theta_hash,
        "selected_theta": run.selected_theta_json or {},
        "sampler_state_hash": run.sampler_state_hash,
        "probe_set_hash": run.probe_set_hash,
        "hard_gate": run.hard_gate_json or {},
        "objective_components": run.objective_components_json or {},
        "last_error": run.last_error,
        "failure_code": run.failure_code,
        "blocking_reasons": run.blocking_reasons_json or [],
        "runtime_settings_hash": run.runtime_settings_hash,
        "diagnostics": run.diagnostics_json or {},
        "trials": [summarize_auto_tpe_trial(trial) for trial in trials],
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


def latest_auto_tpe_status(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise LookupError("Knowledge base not found")
    run = db.scalar(select(AutoTpeRun).where(AutoTpeRun.knowledge_base_id == knowledge_base_id).order_by(AutoTpeRun.created_at.desc()))
    return {
        "knowledge_base_id": knowledge_base_id,
        "current_chunk_version": knowledge_base.current_chunk_version or 0,
        "enabled": bool(get_settings().enable_auto_tpe),
        "latest_run": summarize_auto_tpe_run(db, run) if run else None,
    }
