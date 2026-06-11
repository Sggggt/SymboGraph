#!/usr/bin/env python3
"""Export evidence chunk quality decision distributions and review samples.

Run inside the API container:
    python /app/scripts/evaluate_quality_decisions.py --KnowledgeBase-name "KnowledgeBase Name"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import select

from app.db import SessionLocal, ensure_schema
from app.models import ActiveChunk, ChunkCandidate, ChunkDecision, EvidenceGraphState, KnowledgeBase, QualityDecision


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _sample(candidate: ChunkCandidate, decision: QualityDecision) -> dict[str, Any]:
    diagnostics = decision.diagnostics_json or {}
    return {
        "candidate_id": candidate.id,
        "decision_id": decision.id,
        "generator": candidate.generator_name,
        "generator_version": candidate.generator_version,
        "atom_count": len(candidate.atom_ids_json or []),
        "token_count": candidate.token_count,
        "gate_passed": decision.gate_passed,
        "action": decision.decision_action,
        "confidence": decision.confidence,
        "risk_flags": decision.risk_flags_json or [],
        "diagnostics": _jsonable(diagnostics),
    }


def build_report(db, knowledge_base: KnowledgeBase, sample_limit: int) -> dict[str, Any]:
    graph_states = db.scalars(
        select(EvidenceGraphState).where(EvidenceGraphState.knowledge_base_id == knowledge_base.id)
    ).all()
    graph_state_ids = {state.id for state in graph_states}
    candidates = db.scalars(
        select(ChunkCandidate).where(ChunkCandidate.graph_state_id.in_(graph_state_ids or {"__none__"}))
    ).all()
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    decisions = db.scalars(
        select(QualityDecision).where(QualityDecision.candidate_id.in_(candidate_by_id.keys() or {"__none__"}))
    ).all()
    chunk_decisions = db.scalars(
        select(ChunkDecision).where(ChunkDecision.knowledge_base_id == knowledge_base.id)
    ).all()
    active_chunks = db.scalars(
        select(ActiveChunk).where(ActiveChunk.knowledge_base_id == knowledge_base.id, ActiveChunk.state == "active")
    ).all()

    action_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    generator_counts: Counter[str] = Counter()
    feedback_counts: Counter[str] = Counter()
    review_samples: list[dict[str, Any]] = []

    for decision in decisions:
        candidate = candidate_by_id.get(decision.candidate_id)
        if candidate is None:
            continue
        action_counts[str(decision.decision_action or "missing")] += 1
        gate_counts["passed" if decision.gate_passed else "rejected"] += 1
        generator_counts[str(candidate.generator_name or "missing")] += 1
        for flag in decision.risk_flags_json or ["none"]:
            risk_counts[str(flag)] += 1
        feedback = decision.feedback_json or {}
        for key, value in feedback.items():
            if value:
                feedback_counts[str(key)] += 1
        if (not decision.gate_passed or decision.risk_flags_json) and len(review_samples) < sample_limit:
            review_samples.append(_sample(candidate, decision))

    return {
        "knowledge_base_id": knowledge_base.id,
        "knowledge_base_name": knowledge_base.name,
        "counts": {
            "graph_states": len(graph_states),
            "chunk_candidates": len(candidates),
            "quality_decisions": len(decisions),
            "chunk_decisions": len(chunk_decisions),
            "active_chunks": len(active_chunks),
        },
        "distributions": {
            "actions": _counter_dict(action_counts),
            "gate": _counter_dict(gate_counts),
            "risks": _counter_dict(risk_counts),
            "generators": _counter_dict(generator_counts),
            "feedback": _counter_dict(feedback_counts),
        },
        "review_samples": review_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evidence quality decision audit report.")
    parser.add_argument("--KnowledgeBase-name", "--knowledge-base-name", dest="knowledge_base_name", default=None)
    parser.add_argument("--KnowledgeBase-id", "--knowledge-base-id", dest="knowledge_base_id", default=None)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    ensure_schema()
    with SessionLocal() as db:
        kb_query = select(KnowledgeBase)
        if args.knowledge_base_id:
            kb_query = kb_query.where(KnowledgeBase.id == args.knowledge_base_id)
        if args.knowledge_base_name:
            kb_query = kb_query.where(KnowledgeBase.name == args.knowledge_base_name)
        knowledge_bases = db.scalars(kb_query.order_by(KnowledgeBase.name.asc())).all()
        if not knowledge_bases:
            raise SystemExit("No matching knowledge_bases found.")
        report = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "knowledge_bases": [build_report(db, knowledge_base, args.sample_limit) for knowledge_base in knowledge_bases],
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"quality_decisions_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "knowledge_bases": len(report["knowledge_bases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
