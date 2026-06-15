from __future__ import annotations

import argparse
import asyncio
import json

from sqlalchemy import select

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real layered retrieval queries and write entry/frontier/path diagnostics.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--query", action="append", dest="queries", default=[])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


async def main() -> None:
    from app.models import Chunk
    from app.schemas import SearchFilters
    from app.services.context_graph import layered_search

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        queries = list(args.queries)
        if not queries:
            chunks = db.scalars(select(Chunk).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active").order_by(Chunk.created_at.desc()).limit(args.limit)).all()
            queries = [chunk.text[:120] for chunk in chunks if chunk.text.strip()]
        rows = []
        for query in queries[: args.limit]:
            result = await layered_search(db, knowledge_base.id, query, SearchFilters(), args.top_k)
            rq_results = [
                (item.get("metadata") or {}).get("rq")
                for item in result.results
                if (item.get("metadata") or {}).get("rq")
            ]
            rows.append(
                {
                    "query": query,
                    "result_count": len(result.results),
                    "top_chunks": [item["chunk_id"] for item in result.results[:5]],
                    "path_priorities": [item["score"] for item in result.results[:5]],
                    "frontier_pops": result.audit.get("frontier_pops", 0),
                    "dominance_pruned_count": result.audit.get("dominance_pruned_count", 0),
                    "rq": {
                        "query_rq_path": result.audit.get("query_rq_path") or [],
                        "candidate_count": len(rq_results),
                        "sample": rq_results[:5],
                    },
                    "audit": result.audit,
                    "trace_id": result.trace.id,
                }
            )
        payload = {
            "script": "evaluate_layered_retrieval",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "query_count": len(rows),
            "rows": rows,
            "pass": bool(rows)
            and all(
                row["result_count"] > 0
                and row["trace_id"]
                and row["frontier_pops"] > 0
                and row["rq"]["query_rq_path"]
                and row["rq"]["candidate_count"] > 0
                for row in rows
            ),
        }
        report = write_report("evaluate_layered_retrieval", payload)
        print(json.dumps({"output": str(report), "pass": payload["pass"], "query_count": len(rows)}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
