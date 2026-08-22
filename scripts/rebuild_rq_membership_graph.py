from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import (
    active_chunk_count,
    prepare_runtime_for_model_io,
    resolve_knowledge_base,
    session_scope,
    write_report,
)


REUSED_LAYERS = [
    "chunk_structure_graph",
    "chunk_relation_graph",
]
REBUILT_LAYERS = [
    "rq_membership_graph",
    "mid_concept_graph",
    "coarse_concept_graph",
    "context_graph_state",
]
WRITTEN_TABLES = [
    "chunk_relation_graph_states",
    "chunk_relation_edges",
    "rq_prefixes",
    "rq_prefix_memberships",
    "rq_prefix_diagnostics",
    "rq_prefix_pair_diagnostics",
    "mid_concept_states",
    "mid_concepts",
    "mid_concept_edges",
    "mid_concept_memberships",
    "coarse_concept_states",
    "coarse_concepts",
    "coarse_concept_edges",
    "coarse_concept_memberships",
    "context_graph_states",
    "context_graph_freshness",
    "ingestion_compensation_logs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild RQ membership and its mid/coarse/context dependents while "
            "reusing the admitted chunk structure graph, calibrated bottom "
            "relation facts, active vectors, and TPE operating point. Omit "
            "--execute for a read-only report."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Atomically switch only the RQ -> mid -> coarse -> context lineage. "
            "Without this flag the command performs no writes or model calls."
        ),
    )
    parser.add_argument(
        "--require-provider-zero",
        action="store_true",
        help=(
            "Require exact admitted Mid/Coarse semantic reuse. Any miss "
            "fails before provider network I/O and rolls back the candidate."
        ),
    )
    parser.add_argument(
        "--provider-request-budget",
        type=int,
        help=(
            "Hard upper bound for provider attempts shared by Mid and Coarse. "
            "Each semantic-miss group reserves two schema attempts before "
            "network I/O; valid range: 2..128."
        ),
    )
    return parser.parse_args()


def _rollback_if_supported(db) -> None:
    rollback = getattr(db, "rollback", None)
    if callable(rollback):
        rollback()


def _failure_diagnostics(
    exc: BaseException,
    context_graph,
) -> tuple[dict | None, dict | None]:
    semantic_reuse: dict | None = None
    provider_budget: dict | None = None
    current: BaseException | None = exc
    visited: set[int] = set()
    for _depth in range(8):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        if isinstance(
            current,
            context_graph.ConceptDefinitionSemanticReuseMissError,
        ):
            semantic_reuse = dict(current.diagnostics)
        if isinstance(
            current,
            context_graph.ConceptProviderRequestBudgetExceeded,
        ):
            provider_budget = dict(current.diagnostics)
        next_error = current.__cause__
        if next_error is None or id(next_error) in visited:
            next_error = current.__context__
        current = next_error
    return semantic_reuse, provider_budget


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    require_provider_zero = bool(
        getattr(args, "require_provider_zero", False)
    )
    provider_request_budget = getattr(
        args,
        "provider_request_budget",
        None,
    )
    from app.services import context_graph
    from app.services import scoped_context_graph_rebuild

    model_io_runtime = (
        prepare_runtime_for_model_io() if args.execute else None
    )
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        payload = {
            "script": "rebuild_rq_membership_graph",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "active_chunks": active_chunk_count(db, knowledge_base.id),
            "model_io_runtime": model_io_runtime,
            "execute": args.execute,
            "mode": "execute" if args.execute else "dry_run",
            "provider_request_budget": provider_request_budget,
            "scope_semantics": "scoped_dependency_cascade",
            "targets": {
                "requested_scope": "rq_membership",
                "reused_layers": list(REUSED_LAYERS),
                "rebuilt_layers": list(REBUILT_LAYERS),
                "tables_written_on_execute": list(WRITTEN_TABLES),
                "knowledge_base_id": knowledge_base.id,
            },
            "impact": (
                "reuses chunk structure and calibrated bottom relation facts; "
                "rebuilds RQ membership, mid concepts, coarse concepts, and "
                "the active context graph state"
                if args.execute
                else "no writes and no model/provider calls"
            ),
        }
        if args.execute:
            try:
                state = (
                    await scoped_context_graph_rebuild.rebuild_rq_membership_graph(
                        db,
                        knowledge_base.id,
                        require_provider_semantic_reuse=(
                            require_provider_zero
                        ),
                        provider_request_budget=(
                            provider_request_budget
                        ),
                    )
                )
                db.commit()
            except Exception as exc:
                _rollback_if_supported(db)
                failure = {
                    "error_type": type(exc).__name__[:128],
                    "provider_request_count": 0,
                    "provider_response_persisted": False,
                }
                reuse_failure, budget_failure = _failure_diagnostics(
                    exc,
                    context_graph,
                )
                if reuse_failure is not None:
                    failure["semantic_reuse_diagnostics"] = reuse_failure
                if budget_failure is not None:
                    failure["provider_request_budget"] = budget_failure
                    failure["provider_request_count"] = int(
                        budget_failure.get("observed_requests") or 0
                    )
                payload["failure"] = failure
                payload["transaction"] = {
                    "database_committed": False,
                    "candidate_savepoint": True,
                    "rolled_back": True,
                    "cache_invalidation_queued_after_commit": False,
                }
                report = write_report(
                    "rebuild_rq_membership_graph",
                    payload,
                )
                print(json.dumps({
                    "output": str(report),
                    "execute": True,
                    "status": "failed",
                    "failure": failure,
                }, ensure_ascii=False, default=str))
                raise SystemExit(1) from None
            audit = dict(
                (state.diagnostics_json or {}).get(
                    "scoped_rebuild_audit"
                )
                or {}
            )
            payload.update(
                {
                    "context_graph_state_id": state.id,
                    "stats": state.stats_json or {},
                    "scoped_rebuild_audit": audit,
                    "transaction": {
                        "database_committed": True,
                        "candidate_savepoint": True,
                        "cache_invalidation_queued_after_commit": bool(
                            audit.get(
                                "cache_invalidation_queued_after_commit"
                            )
                        ),
                    },
                    "provider_zero_required": bool(
                        require_provider_zero
                    ),
                    "provider_request_budget": (
                        provider_request_budget
                    ),
                }
            )
        else:
            payload["current_stats"] = context_graph.context_graph_stats(
                db,
                knowledge_base.id,
            )
            payload["transaction"] = {
                "database_committed": False,
                "candidate_savepoint": False,
                "cache_invalidation_queued_after_commit": False,
            }
        report = write_report("rebuild_rq_membership_graph", payload)
        print(
            json.dumps(
                {
                    "output": str(report),
                    "mode": payload["mode"],
                    "execute": args.execute,
                    "status": "succeeded",
                    "impact": payload["impact"],
                    "active_chunks": payload["active_chunks"],
                    **(
                        {
                            "context_graph_state_id": payload[
                                "context_graph_state_id"
                            ]
                        }
                        if "context_graph_state_id" in payload
                        else {}
                    ),
                    "provider_zero_required": bool(
                        require_provider_zero
                    ),
                    "provider_request_budget": (
                        (payload.get("scoped_rebuild_audit") or {}).get(
                            "provider_request_budget"
                        )
                        or provider_request_budget
                    ),
                    "provider_request_count": int(
                        (payload.get("scoped_rebuild_audit") or {}).get(
                            "provider_request_count"
                        )
                        or 0
                    ),
                    "gray_zone_model_call_count": int(
                        (payload.get("scoped_rebuild_audit") or {}).get(
                            "gray_zone_model_call_count"
                        )
                        or 0
                    ),
                    "database_committed": bool(
                        (payload.get("transaction") or {}).get(
                            "database_committed"
                        )
                    ),
                    "transaction": payload["transaction"],
                    "scope_semantics": payload["scope_semantics"],
                    "targets": payload["targets"],
                    "scoped_rebuild_audit": payload.get(
                        "scoped_rebuild_audit"
                    ),
                    **(
                        {"current_stats": payload["current_stats"]}
                        if "current_stats" in payload
                        else {}
                    ),
                },
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
