from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.quality.signals import FORMULA_RE, build_quality_signals


CHUNK_ADAPTATION_VERSION = "continuous_chunk_ablation_v1"


@dataclass(frozen=True)
class ChunkingProfile:
    chunk_size: int
    chunk_overlap: int
    strategy: str
    score: float
    feature_summary: dict[str, float]
    spectral_summary: dict[str, Any]
    candidates: list[dict[str, Any]]

    def metadata(self) -> dict[str, Any]:
        return {
            "version": CHUNK_ADAPTATION_VERSION,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.strategy,
            "score": round(self.score, 4),
            "feature_summary": self.feature_summary,
            "spectral_summary": self.spectral_summary,
            "candidates": self.candidates,
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_text(text: str) -> str:
    text = (text or "").replace("\x00", "").replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def rough_token_count(text: str) -> int:
    from app.services.chinese_text import estimate_tokens

    return max(1, estimate_tokens(text))


def semantic_units(text: str) -> list[str]:
    normalized = normalize_text(text)
    units = re.split(r"(?<=[。！？])|(?<=[.?!])\s+|\n{2,}", normalized)
    return [unit.strip() for unit in units if unit.strip()]


def _sentence_lengths(text: str) -> list[int]:
    units = semantic_units(text)
    return [rough_token_count(unit) for unit in units if unit.strip()]


def _code_marker_ratio(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    markers = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ", "const ", "let ", "var ", "function ")):
            markers += 1
        elif re.search(r"[{};]|=>|</?[A-Za-z][^>]*>", stripped):
            markers += 1
    return markers / len(lines)


def document_feature_vector(
    text: str,
    *,
    title: str | None = None,
    section: str | None = None,
    content_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    normalized = normalize_text(text)
    token_count = rough_token_count(normalized) if normalized else 0
    lengths = _sentence_lengths(normalized)
    mean_sentence = float(np.mean(lengths)) if lengths else 0.0
    sentence_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    signals = build_quality_signals(
        target_type="chunk",
        text=normalized,
        title=title,
        section=section,
        content_kind=content_kind,
        metadata=metadata or {},
        version=CHUNK_ADAPTATION_VERSION,
    )
    symbol_count = sum(not char.isalnum() and not char.isspace() for char in normalized)
    formula_markers = len(FORMULA_RE.findall(normalized))
    return {
        "token_count": round(min(1.0, token_count / 1800.0), 4),
        "mean_sentence_tokens": round(min(1.0, mean_sentence / 80.0), 4),
        "sentence_length_cv": round(min(1.0, sentence_std / max(mean_sentence, 1.0)), 4),
        "definition_density": round(float(signals.semantic_density.definition_score), 4),
        "entity_density": round(float(signals.semantic_density.entity_density), 4),
        "term_density": round(float(signals.semantic_density.term_density), 4),
        "unique_token_ratio": round(float(signals.semantic_density.unique_token_ratio), 4),
        "formula_signal": round(min(1.0, _safe_ratio(formula_markers, max(token_count, 1)) * 12.0 + float(signals.semantic_density.has_formula) * 0.35), 4),
        "table_signal": round(float(signals.semantic_density.has_table), 4),
        "code_marker_ratio": round(_code_marker_ratio(normalized), 4),
        "symbol_ratio": round(min(1.0, _safe_ratio(symbol_count, max(len(normalized), 1)) * 4.0), 4),
        "structural_noise": round(max(float(signals.structural_role.structural_score), float(signals.text_quality.mojibake_ratio) * 20.0), 4),
        "repeated_line_ratio": round(float(signals.text_quality.repeated_line_ratio), 4),
    }


def spectral_document_shape(text: str) -> dict[str, Any]:
    units = semantic_units(text)
    if len(units) < 2:
        return {"eigenvalues": [0.0], "spectral_gap": 0.0, "principal_direction": [0.0], "semantic_curvature": 0.0}
    vectors = [
        list(document_feature_vector(unit).values())
        for unit in units[:48]
        if unit.strip()
    ]
    if len(vectors) < 2:
        return {"eigenvalues": [0.0], "spectral_gap": 0.0, "principal_direction": [0.0], "semantic_curvature": 0.0}
    matrix = np.asarray(vectors, dtype=float)
    centered = matrix - matrix.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return {"eigenvalues": [0.0], "spectral_gap": 0.0, "principal_direction": [0.0], "semantic_curvature": 0.0}
    order = np.argsort(eigenvalues)[::-1]
    values = [float(max(eigenvalues[index], 0.0)) for index in order[:4]]
    total = sum(values) or 1.0
    norm_values = [round(value / total, 4) for value in values]
    principal = eigenvectors[:, order[0]] if len(order) else np.zeros(matrix.shape[1])
    gap = norm_values[0] - (norm_values[1] if len(norm_values) > 1 else 0.0)
    deltas = np.diff(matrix, axis=0)
    curvature = float(np.mean(np.linalg.norm(deltas, axis=1))) if len(deltas) else 0.0
    return {
        "eigenvalues": norm_values,
        "spectral_gap": round(float(gap), 4),
        "principal_direction": [round(float(value), 4) for value in principal[:6]],
        "semantic_curvature": round(min(1.0, curvature), 4),
    }


def _candidate_profiles() -> list[tuple[int, int, str]]:
    return [
        (512, 64, "sentence_aware"),
        (640, 96, "sentence_aware"),
        (800, 120, "semantic_or_sentence"),
        (960, 144, "semantic_or_sentence"),
        (700, 160, "recursive_structure_preserving"),
    ]


def _score_candidate(features: dict[str, float], spectral: dict[str, Any], chunk_size: int, overlap: int, strategy: str) -> float:
    curvature = float(spectral.get("semantic_curvature", 0.0) or 0.0)
    spectral_gap = float(spectral.get("spectral_gap", 0.0) or 0.0)
    complexity = min(
        1.0,
        0.22 * features["sentence_length_cv"]
        + 0.18 * features["formula_signal"]
        + 0.16 * features["table_signal"]
        + 0.18 * features["code_marker_ratio"]
        + 0.16 * features["symbol_ratio"]
        + 0.10 * curvature,
    )
    density = min(1.0, 0.35 * features["term_density"] + 0.30 * features["entity_density"] + 0.20 * features["definition_density"] + 0.15 * features["unique_token_ratio"])
    target_size = 920 - 380 * complexity + 160 * density + 100 * spectral_gap
    target_overlap = 80 + 130 * complexity + 40 * curvature
    size_fit = 1.0 - min(1.0, abs(chunk_size - target_size) / 700.0)
    overlap_fit = 1.0 - min(1.0, abs(overlap - target_overlap) / 220.0)
    strategy_bonus = 0.0
    if strategy == "semantic_or_sentence":
        strategy_bonus += 0.08 * max(curvature, density)
    if strategy == "recursive_structure_preserving":
        strategy_bonus += 0.10 * features["code_marker_ratio"] + 0.05 * features["formula_signal"]
    return max(0.0, min(1.0, 0.54 * size_fit + 0.26 * overlap_fit + 0.14 * density - 0.10 * features["structural_noise"] + strategy_bonus))


def choose_chunking_profile(
    text: str,
    *,
    title: str | None = None,
    section: str | None = None,
    content_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChunkingProfile:
    features = document_feature_vector(text, title=title, section=section, content_kind=content_kind, metadata=metadata)
    spectral = spectral_document_shape(text)
    candidates: list[dict[str, Any]] = []
    for chunk_size, overlap, strategy in _candidate_profiles():
        score = _score_candidate(features, spectral, chunk_size, overlap, strategy)
        candidates.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "strategy": strategy,
                "score": round(score, 4),
            }
        )
    best = max(candidates, key=lambda item: (item["score"], item["chunk_size"]))
    return ChunkingProfile(
        chunk_size=int(best["chunk_size"]),
        chunk_overlap=int(best["chunk_overlap"]),
        strategy=str(best["strategy"]),
        score=float(best["score"]),
        feature_summary=features,
        spectral_summary=spectral,
        candidates=sorted(candidates, key=lambda item: item["score"], reverse=True),
    )
