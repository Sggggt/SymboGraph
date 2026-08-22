from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import (
    prepare_runtime_for_model_io,
    session_scope,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry only the graph half of one exact compensated, "
            "version-incrementing ingestion batch."
        )
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the sealed graph-only plan. Omit for dry-run.",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    from app.services.context_graph import concept_provider_output_failure_card
    from app.services.error_sanitizer import external_failure_classification
    from app.services.ingestion import (
        plan_versioned_graph_retry,
        run_versioned_graph_retry,
    )

    with session_scope() as db:
        plan = plan_versioned_graph_retry(db, batch_id=str(args.batch_id))
        db.rollback()
    payload = {
        "script": "retry_versioned_graph",
        "execute": bool(args.execute),
        "plan": plan,
        "impact": (
            "retry the exact retained versioned TPE + Four-Layer graph transaction; "
            "no parse, chunk-version increment, embedding, or Qdrant writes"
            if args.execute
            else "no writes and no provider calls"
        ),
        "model_io_runtime": (
            prepare_runtime_for_model_io() if args.execute else None
        ),
    }
    try:
        if args.execute:
            payload["batch"] = await run_versioned_graph_retry(
                batch_id=str(args.batch_id),
                expected_plan_hash=str(plan["plan_hash"]),
                execution_mode="script",
            )
            cache_state = dict(
                (payload["batch"].get("stats") or {}).get(
                    "post_commit_cache_invalidation"
                )
                or {}
            )
            payload["status"] = (
                "passed"
                if cache_state.get("status") == "dispatched"
                else "failed_cache_invalidation_pending"
            )
        else:
            payload["status"] = "dry_run"
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "failure_type": type(exc).__name__[:128],
                "external_failure": external_failure_classification(exc),
                "concept_provider_failure": (
                    concept_provider_output_failure_card(exc)
                ),
                "provider_response_persisted": False,
            }
        )
        report = write_report("retry_versioned_graph", payload)
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )
        raise SystemExit(1) from None
    report = write_report("retry_versioned_graph", payload)
    print(
        json.dumps(
            {"output": str(report), **payload},
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
