from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import (
    resolve_knowledge_base,
    session_scope,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh only a stale active context-state protocol wrapper when "
            "all relation/RQ/mid/coarse/vector facts remain current. The "
            "default is read-only."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Atomically create one canonical context-state wrapper, rebind "
            "the active PostgreSQL vector pointer, retire the old wrapper, "
            "and queue Redis invalidation. No provider or Qdrant writes."
        ),
    )
    return parser.parse_args()


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    from app.services import scoped_context_graph_rebuild

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        plan = (
            scoped_context_graph_rebuild.plan_context_protocol_identity_refresh(
                db,
                knowledge_base.id,
            )
        )
        payload = {
            "script": "refresh_context_protocol_identity",
            "knowledge_base_id": str(knowledge_base.id),
            "knowledge_base_name": str(knowledge_base.name),
            "execute": bool(args.execute),
            "mode": "execute" if args.execute else "dry_run",
            "plan": plan,
            "impact": (
                "write one PostgreSQL context-state wrapper, switch the "
                "active PostgreSQL vector pointer, retire the previous "
                "wrapper, and queue bounded Redis invalidation; zero model "
                "calls and zero Qdrant writes"
                if args.execute
                else "no PostgreSQL, Redis, Qdrant, provider, or file-data writes"
            ),
        }
        if args.execute:
            if not plan.get("eligible"):
                payload["status"] = "refused"
                payload["transaction"] = {
                    "postgresql_committed": False,
                    "rolled_back": False,
                }
                report = write_report(
                    "refresh_context_protocol_identity",
                    payload,
                )
                print(
                    json.dumps(
                        {"output": str(report), **payload},
                        ensure_ascii=False,
                        default=str,
                    )
                )
                raise SystemExit(1)
            try:
                state = (
                    await scoped_context_graph_rebuild.refresh_context_protocol_identity(
                        db,
                        knowledge_base.id,
                    )
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
            audit = dict(
                (state.diagnostics_json or {}).get(
                    "context_protocol_identity_refresh_audit"
                )
                or {}
            )
            payload.update(
                {
                    "status": "passed",
                    "context_graph_state_id": str(state.id),
                    "context_graph_hash": str(state.context_graph_hash),
                    "audit": audit,
                    "transaction": {
                        "postgresql_committed": True,
                        "candidate_savepoint": True,
                        "cache_invalidation_queued_after_commit": bool(
                            audit.get(
                                "cache_invalidation_queued_after_commit"
                            )
                        ),
                    },
                }
            )
        else:
            payload["status"] = "dry_run"
            payload["transaction"] = {
                "postgresql_committed": False,
                "candidate_savepoint": False,
                "cache_invalidation_queued_after_commit": False,
            }
        report = write_report(
            "refresh_context_protocol_identity",
            payload,
        )
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
