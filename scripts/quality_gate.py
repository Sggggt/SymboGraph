#!/usr/bin/env python3
"""No-fallback quality gate for evidence graph and vector health.

Run inside the API container:
    python /app/scripts/quality_gate.py --KnowledgeBase-name "KnowledgeBase Name"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db import SessionLocal, ensure_schema
from app.models import (
    ActiveChunk,
    ChunkCandidate,
    ChunkDecision,
    CommunityMembership,
    CommunityState,
    CommunitySummary,
    EvidenceAtom,
    EvidenceEdge,
    EvidenceGraphState,
    KnowledgeBase,
    QualityDecision,
    VectorRecord,
)
from app.services.vector_store import VectorStore


REMOVED_ACTIVE_ENTRYPOINTS = (
    "/concepts",
    "/search/semantic-graph",
    "/maintenance/cleanup-stale-graph",
    "rebuild_graph_mode",
    "confirm_destructive_graph_rebuild",
    "QuerySemanticGraphRequest",
    "cleanupStaleGraph",
    "fetchConcepts",
    "rebuild_knowledge_base_graph_task",
)


def assert_removed_entrypoints_absent(repo_root: Path) -> None:
    scan_targets = [
        repo_root / "apps" / "api" / "app" / "api.py",
        repo_root / "apps" / "api" / "app" / "schemas.py",
        repo_root / "apps" / "api" / "app" / "services" / "ingestion.py",
        repo_root / "apps" / "api" / "app" / "services" / "retrieval.py",
        repo_root / "apps" / "api" / "app" / "services" / "agent_graph.py",
        repo_root / "apps" / "api" / "app" / "services" / "runtime_settings.py",
        repo_root / "apps" / "web" / "src",
        repo_root / "apps" / "worker" / "worker_app" / "tasks.py",
        repo_root / "packages" / "shared" / "src",
    ]
    matches: list[str] = []
    for target in scan_targets:
        paths = [target] if target.is_file() else target.rglob("*") if target.exists() else []
        for path in paths:
            if path.suffix.lower() not in {".py", ".ts", ".tsx"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for symbol in REMOVED_ACTIVE_ENTRYPOINTS:
                if symbol in text:
                    matches.append(f"{path.relative_to(repo_root)}: {symbol}")
    if matches:
        raise SystemExit("Removed legacy active entrypoints remain:\n  " + "\n  ".join(matches))


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check evidence graph and vector health.")
    parser.add_argument("--KnowledgeBase-name", "--knowledge-base-name", dest="knowledge_base_name", default=None)
    parser.add_argument("--KnowledgeBase-id", "--knowledge-base-id", dest="knowledge_base_id", default=None)
    parser.add_argument("--delete-orphan-zero-vectors", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if settings.enable_model_fallback or settings.enable_database_fallback:
        raise SystemExit("Quality gate must run with ENABLE_MODEL_FALLBACK=false and ENABLE_DATABASE_FALLBACK=false.")
    repo_root = Path(__file__).resolve().parents[1]
    assert_removed_entrypoints_absent(repo_root)

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

        ok = True
        for knowledge_base in knowledge_bases:
            active_chunks = db.scalars(
                select(ActiveChunk).where(ActiveChunk.knowledge_base_id == knowledge_base.id, ActiveChunk.state == "active")
            ).all()
            active_chunk_ids = {chunk.id for chunk in active_chunks}
            vector_records = db.scalars(
                select(VectorRecord).where(
                    VectorRecord.knowledge_base_id == knowledge_base.id,
                    VectorRecord.vector_status == "ready",
                )
            ).all()
            expected_vector_ids = {
                record.qdrant_point_id
                for record in vector_records
                if record.active_chunk_id in active_chunk_ids
            }
            broken_vector_records = [
                record.id
                for record in vector_records
                if record.active_chunk_id not in active_chunk_ids
            ]

            active_atoms = db.scalars(
                select(EvidenceAtom).where(EvidenceAtom.knowledge_base_id == knowledge_base.id, EvidenceAtom.state == "active")
            ).all()
            active_atom_ids = {atom.id for atom in active_atoms}
            graph_states = db.scalars(
                select(EvidenceGraphState).where(EvidenceGraphState.knowledge_base_id == knowledge_base.id, EvidenceGraphState.state == "active")
            ).all()
            graph_state_ids = {state.id for state in graph_states}
            bad_graph_states = [
                state.id
                for state in graph_states
                if not set(str(item) for item in (state.active_atom_ids or [])).issubset(active_atom_ids)
            ]
            edges = db.scalars(select(EvidenceEdge).where(EvidenceEdge.graph_state_id.in_(graph_state_ids or {"__none__"}))).all()
            bad_edges = [
                edge.id
                for edge in edges
                if edge.source_atom_id not in active_atom_ids or edge.target_atom_id not in active_atom_ids
            ]
            broken_active_chunks = [
                chunk.id
                for chunk in active_chunks
                if not (chunk.atom_ids_json or [])
                or not set(str(item) for item in (chunk.atom_ids_json or [])).issubset(active_atom_ids)
                or not chunk.quality_decision_id
                or not chunk.graph_state_hash
            ]

            candidate_count = db.scalar(
                select(func.count(ChunkCandidate.id)).where(ChunkCandidate.graph_state_id.in_(graph_state_ids or {"__none__"}))
            )
            decision_count = db.scalar(
                select(func.count(ChunkDecision.id)).where(ChunkDecision.knowledge_base_id == knowledge_base.id)
            )
            quality_count = db.scalar(
                select(func.count(QualityDecision.id)).join(ChunkCandidate, QualityDecision.candidate_id == ChunkCandidate.id).where(
                    ChunkCandidate.graph_state_id.in_(graph_state_ids or {"__none__"})
                )
            )
            community_states = db.scalars(
                select(CommunityState).where(CommunityState.knowledge_base_id == knowledge_base.id, CommunityState.state == "active")
            ).all()
            community_state_ids = {state.id for state in community_states}
            community_memberships = db.scalar(
                select(func.count(CommunityMembership.id)).where(CommunityMembership.community_state_id.in_(community_state_ids or {"__none__"}))
            )
            community_summaries = db.scalar(
                select(func.count(CommunitySummary.id)).where(CommunitySummary.community_state_id.in_(community_state_ids or {"__none__"}))
            )

            vector_store = VectorStore(knowledge_base_name=knowledge_base.name)
            qdrant_ids = set(vector_store.list_ids(knowledge_base.id))
            missing = sorted(expected_vector_ids - qdrant_ids)
            orphan = sorted(qdrant_ids - expected_vector_ids)

            zero_ids: list[str] = []
            checked = 0
            for index in range(0, len(qdrant_ids), 100):
                batch_ids = sorted(qdrant_ids)[index : index + 100]
                points = vector_store.get_points(batch_ids)
                checked += len(points)
                for point in points:
                    if vector_norm(point.get("vector") or []) <= 1e-12:
                        zero_ids.append(str(point["id"]))

            orphan_zero_ids = sorted(set(zero_ids).intersection(orphan))
            if args.delete_orphan_zero_vectors and orphan_zero_ids:
                vector_store.delete(orphan_zero_ids)
                orphan = sorted(set(orphan) - set(orphan_zero_ids))
                zero_ids = sorted(set(zero_ids) - set(orphan_zero_ids))

            print(f"\nKnowledgeBase: {knowledge_base.name}")
            print(f"  active_chunks={len(active_chunks)} active_atoms={len(active_atoms)} evidence_edges={len(edges)}")
            print(f"  graph_states={len(graph_states)} chunk_candidates={candidate_count or 0} chunk_decisions={decision_count or 0} quality_decisions={quality_count or 0}")
            print(f"  vector_records={len(vector_records)} expected_vectors={len(expected_vector_ids)} qdrant_vectors={len(qdrant_ids)} checked_vectors={checked}")
            print(f"  missing_vectors={len(missing)} orphan_vectors={len(orphan)} zero_vectors={len(zero_ids)}")
            print(f"  communities={len(community_states)} memberships={community_memberships or 0} summaries={community_summaries or 0}")
            print(f"  broken_active_chunks={len(broken_active_chunks)} broken_vector_records={len(broken_vector_records)} bad_graph_states={len(bad_graph_states)} bad_edges={len(bad_edges)}")

            if orphan_zero_ids:
                print(f"  orphan_zero_vectors={', '.join(orphan_zero_ids)}")
            if missing or orphan or zero_ids or broken_active_chunks or broken_vector_records or bad_graph_states or bad_edges:
                ok = False

        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
