from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import prepare_runtime_for_model_io, resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild coarse concept graph and active context graph state.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true", help="Write new coarse/context states. Omit for dry-run.")
    return parser.parse_args()


async def main() -> None:
    from app.services.context_graph import context_graph_stats, rebuild_context_graph

    args = parse_args()
    model_io_runtime = prepare_runtime_for_model_io()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        payload = {
            "script": "rebuild_coarse_concept_graph",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "model_io_runtime": model_io_runtime,
            "execute": args.execute,
            "impact": "replace coarse concept/community state and active context graph state" if args.execute else "no writes",
        }
        if args.execute:
            state = await rebuild_context_graph(db, knowledge_base.id)
            db.commit()
            payload.update({"context_graph_state_id": state.id, "stats": state.stats_json or {}})
        else:
            payload["current_stats"] = context_graph_stats(db, knowledge_base.id)
        report = write_report("rebuild_coarse_concept_graph", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
