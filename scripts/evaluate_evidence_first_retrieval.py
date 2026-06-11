#!/usr/bin/env python3
"""Compare base-only vs evidence-first retrieval proxies.

Run inside the API container:
    python /app/scripts/evaluate_evidence_first_retrieval.py --KnowledgeBase-name "KnowledgeBase Name"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import select

from app.db import SessionLocal, ensure_schema
from app.models import EvidenceAtom, EvidenceEdge, EvidenceGraphState, KnowledgeBase
from app.schemas import SearchFilters
from app.services.retrieval import evidence_first_search_chunks_with_audit, hybrid_search_chunks_with_audit


def build_queries(db, knowledge_base_id: str, limit: int) -> list[str]:
    atoms = db.scalars(
        select(EvidenceAtom)
        .where(EvidenceAtom.knowledge_base_id == knowledge_base_id, EvidenceAtom.state == "active")
        .order_by(EvidenceAtom.created_at.desc(), EvidenceAtom.atom_index.asc())
        .limit(limit)
    ).all()
    queries = [f"Find evidence about {atom.text[:90]}" for atom in atoms[: max(1, limit // 2)] if atom.text.strip()]
    graph_state_ids = db.scalars(
        select(EvidenceGraphState.id).where(EvidenceGraphState.knowledge_base_id == knowledge_base_id, EvidenceGraphState.state == "active")
    ).all()
    edges = db.scalars(
        select(EvidenceEdge)
        .where(EvidenceEdge.graph_state_id.in_(set(graph_state_ids) or {"__none__"}))
        .order_by(EvidenceEdge.weight.desc(), EvidenceEdge.confidence.desc())
        .limit(limit)
    ).all()
    for edge in edges[: max(1, limit - len(queries))]:
        source = db.get(EvidenceAtom, edge.source_atom_id)
        target = db.get(EvidenceAtom, edge.target_atom_id)
        if source is None or target is None:
            continue
        queries.append(f"How are these evidence atoms connected: {source.text[:70]} / {target.text[:70]}?")
    return list(dict.fromkeys(queries))[:limit]


def precision_proxy(results: list[dict]) -> float:
    if not results:
        return 0.0
    accepted = 0
    for item in results:
        metadata = item.get("metadata") or {}
        scores = metadata.get("scores") or {}
        if float(scores.get("rerank", 0.0) or scores.get("fused", 0.0) or item.get("score", 0.0) or 0.0) >= 0.3:
            accepted += 1
    return accepted / len(results)


async def evaluate_course(KnowledgeBase: KnowledgeBase, query_limit: int, top_k: int) -> dict:
    rows = []
    with SessionLocal() as db:
        queries = build_queries(db, KnowledgeBase.id, query_limit)
    for query in queries:
        filters = SearchFilters()
        with SessionLocal() as db:
            start = time.perf_counter()
            base_results, base_audit = await hybrid_search_chunks_with_audit(db, KnowledgeBase.id, query, filters, top_k)
            base_ms = int((time.perf_counter() - start) * 1000)
        with SessionLocal() as db:
            start = time.perf_counter()
            evidence_results, evidence_audit = await evidence_first_search_chunks_with_audit(
                db,
                KnowledgeBase.id,
                query,
                filters,
                top_k,
                route="multi_hop_research" if "relate" in query.lower() else "retrieve_sources",
            )
            evidence_ms = int((time.perf_counter() - start) * 1000)
        base_ids = {item["chunk_id"] for item in base_results}
        evidence_ids = {item["chunk_id"] for item in evidence_results}
        graph_chunks = sum(1 for item in evidence_results if (item.get("metadata") or {}).get("evidence_role") in {"evidence_neighbor", "community_summary"})
        rows.append(
            {
                "query": query,
                "base_count": len(base_results),
                "evidence_first_count": len(evidence_results),
                "base_precision_proxy": round(precision_proxy(base_results), 4),
                "evidence_precision_proxy": round(precision_proxy(evidence_results), 4),
                "recall_proxy_overlap": round(len(base_ids.intersection(evidence_ids)) / max(len(base_ids), 1), 4),
                "graph_expansion_ratio": round(graph_chunks / max(len(evidence_results), 1), 4),
                "base_latency_ms": base_ms,
                "evidence_first_latency_ms": evidence_ms,
                "audit": evidence_audit,
                "base_audit": base_audit,
            }
        )
    return {
        "knowledge_base_id": KnowledgeBase.id,
        "knowledge_base_name": KnowledgeBase.name,
        "query_count": len(rows),
        "summary": {
            "base_latency_ms_avg": round(statistics.mean([row["base_latency_ms"] for row in rows]), 2) if rows else 0,
            "evidence_first_latency_ms_avg": round(statistics.mean([row["evidence_first_latency_ms"] for row in rows]), 2) if rows else 0,
            "base_precision_proxy_avg": round(statistics.mean([row["base_precision_proxy"] for row in rows]), 4) if rows else 0,
            "evidence_precision_proxy_avg": round(statistics.mean([row["evidence_precision_proxy"] for row in rows]), 4) if rows else 0,
            "graph_expansion_ratio_avg": round(statistics.mean([row["graph_expansion_ratio"] for row in rows]), 4) if rows else 0,
        },
        "queries": rows,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence-first retrieval proxies.")
    parser.add_argument("--KnowledgeBase-name", "--knowledge-base-name", dest="knowledge_base_name", default=None)
    parser.add_argument("--KnowledgeBase-id", "--knowledge-base-id", dest="knowledge_base_id", default=None)
    parser.add_argument("--query-limit", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    ensure_schema()
    with SessionLocal() as db:
        query = select(KnowledgeBase)
        if args.knowledge_base_id:
            query = query.where(KnowledgeBase.id == args.knowledge_base_id)
        if args.knowledge_base_name:
            query = query.where(KnowledgeBase.name == args.knowledge_base_name)
        knowledge_bases = db.scalars(query.order_by(KnowledgeBase.name.asc())).all()
    if not knowledge_bases:
        raise SystemExit("No matching knowledge_bases found.")

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "top_k": args.top_k,
        "knowledge_bases": [await evaluate_course(KnowledgeBase, args.query_limit, args.top_k) for KnowledgeBase in knowledge_bases],
    }
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evidence_first_retrieval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "knowledge_bases": len(output["knowledge_bases"])}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
