from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
import re
from typing import Any

from app.services.quality.signals import QualitySignals


@dataclass(frozen=True)
class QualityDecision:
    target_type: str
    action: str
    score: float
    reasons: list[str] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(1.0, value)), 4)


def _chunk_route(action: str, *, retain: bool = True) -> dict[str, bool]:
    if not retain or action == "discard":
        return {
            "embed": False,
            "retrieval": False,
            "evidence_graph": False,
            "summary": False,
            "evidence_only": False,
        }
    if action == "graph_candidate":
        return {"embed": True, "retrieval": True, "evidence_graph": True, "summary": True, "evidence_only": False}
    if action == "retrieval_candidate":
        return {"embed": True, "retrieval": True, "evidence_graph": False, "summary": True, "evidence_only": False}
    if action == "summary_only":
        return {"embed": True, "retrieval": False, "evidence_graph": False, "summary": True, "evidence_only": True}
    if action == "evidence_only":
        return {"embed": True, "retrieval": True, "evidence_graph": False, "summary": False, "evidence_only": True}
    return {"embed": True, "retrieval": False, "evidence_graph": False, "summary": False, "evidence_only": False}


class ChunkQualityPolicy:
    def decide(self, signals: QualitySignals, *, section_name: str | None = None, section_title: str | None = None) -> QualityDecision:
        text = signals.text_quality
        semantic = signals.semantic_density
        structural = signals.structural_role
        reasons: list[str] = []

        if text.normalized_length == 0:
            reasons.append("empty_chunk")
        elif text.normalized_length < 40:
            reasons.append("too_short_for_chunk")
        if text.mojibake_ratio > 0.08:
            reasons.append("severe_mojibake_noise")
        elif text.mojibake_ratio > 0.01:
            reasons.append("mojibake_noise")
        if text.control_char_count >= max(8, int(max(text.length, 1) * 0.05)):
            reasons.append("control_char_noise")
        if text.repeated_line_ratio >= 0.92 and text.normalized_length >= 120:
            reasons.append("repeated_extraction_noise")
        if text.toc_like:
            reasons.append("toc_layout_noise")
        if "output" in structural.roles:
            reasons.append("notebook_output")

        score = (
            0.30 * min(1.0, text.normalized_length / 600)
            + 0.25 * semantic.term_density
            + 0.20 * semantic.unique_token_ratio
            + 0.15 * semantic.definition_score
            + 0.05 * float(semantic.has_formula)
            + 0.05 * float(semantic.has_table)
        )
        score -= 0.35 * float(text.toc_like)
        score -= 0.40 * min(1.0, text.mojibake_ratio * 20)

        hard_discard_reasons = {"empty_chunk", "severe_mojibake_noise", "control_char_noise", "repeated_extraction_noise"}
        retain = not bool(set(reasons).intersection(hard_discard_reasons))
        retention_reason = "retained_for_downstream_routing" if retain else "mechanical_noise"

        if not retain:
            action = "discard"
        elif text.toc_like or ("structural_label" in structural.roles and not (semantic.has_formula or semantic.has_table)):
            action = "summary_only"
        elif "output" in structural.roles:
            action = "evidence_only"
        elif text.normalized_length < 40 and not (semantic.has_formula or semantic.has_table or semantic.definition_score):
            action = "evidence_only"
        elif "code" in structural.roles and not _is_kept_code_section(section_name, section_title):
            action = "embed_only"
            reasons.append("code_without_traceable_context")
        elif semantic.definition_score or semantic.entity_density >= 0.08 or semantic.term_density >= 0.20:
            action = "graph_candidate"
        elif semantic.has_formula or semantic.has_table:
            action = "retrieval_candidate"
        else:
            action = "retrieval_candidate" if score >= 0.25 else "embed_only"

        return QualityDecision(
            target_type="chunk",
            action=action,
            score=_rounded(score),
            reasons=reasons or ["policy_passed"],
            audit={
                "signals": signals.model_dump(),
                "section_name": section_name,
                "section_title": section_title,
                "retention_decision": {"retain": retain, "reason": retention_reason},
                "route_eligibility": _chunk_route(action, retain=retain),
            },
        )

def _is_kept_code_section(section_name: str | None, section_title: str | None) -> bool:
    text = f"{section_name or ''} {section_title or ''}".strip()
    if not text:
        return False
    tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", text)
    generic_tokens = {
        "code",
        "cell",
        "generic",
        "utility",
        "utils",
        "script",
        "example",
        "sample",
        "output",
    }
    specific_tokens = [token for token in tokens if token.lower() not in generic_tokens]
    has_specific_label = len(specific_tokens) >= 2 or any(len(token) >= 8 for token in specific_tokens)
    has_symbol_context = bool(re.search(r"[A-Z_][A-Za-z0-9_]{2,}|[=(){}\[\].:/\\-]", text))
    return has_specific_label or (has_symbol_context and bool(specific_tokens))
