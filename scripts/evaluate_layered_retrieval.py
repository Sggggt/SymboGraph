from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import (
    prepare_runtime_for_model_io,
    resolve_knowledge_base,
    session_scope,
    write_report,
)
from _gray_zone_audit import audit_gray_zone_trace, audit_gray_zone_traces
from _quality_gate import audit_retrieval_quality, retrieval_snapshot_from_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay persisted layered-retrieval traces in read-only mode. "
            "Creating new retrieval traces requires --execute and at least "
            "one explicit --query."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=[],
        help=(
            "Exact persisted query to replay in dry-run mode, or an exact new "
            "query to execute when --execute is present. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--trace-id",
        action="append",
        dest="trace_ids",
        default=[],
        help="Replay an exact persisted retrieval trace id (dry-run only).",
    )
    parser.add_argument(
        "--typed-replay-source-trace-id",
        help=(
            "Use one persisted Agent retrieval trace as the immutable source "
            "for an acceptance-only typed-control replay. Without --execute "
            "this is read-only; with --execute it creates independent traces "
            "through the production typed-action executor without replanning."
        ),
    )
    parser.add_argument(
        "--typed-replay-count",
        type=int,
        default=2,
        help="Number of independent typed-control executions (2-5).",
    )
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--retrieval-granularity",
        choices=("coarse", "mid"),
        default="mid",
        help=(
            "Active retrieval entry mode for --execute. The whitepaper and "
            "shared API contract currently support only coarse and mid; "
            "chunk is rejected rather than silently remapped."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the supplied queries through the same audited retrieval and "
            "Context Package builder used by POST /search, then commit the new "
            "artifacts. Omit for persisted replay."
        ),
    )
    parser.add_argument(
        "--require-gray-coverage",
        action="store_true",
        help="Fail unless the evaluated persisted traces include at least one deterministic gray local-rule decision.",
    )
    return parser.parse_args()


def _raw_trace_payload(db, trace) -> dict:
    from sqlalchemy import select

    from app.models import GraphRetrievalStep

    steps = db.scalars(
        select(GraphRetrievalStep)
        .where(GraphRetrievalStep.retrieval_trace_id == trace.id)
        .order_by(
            GraphRetrievalStep.step_index.asc(),
            GraphRetrievalStep.id.asc(),
        )
    ).all()
    return retrieval_snapshot_from_records(trace, steps)


def _document_coverage(db, chunk_ids: list[str]) -> dict:
    from sqlalchemy import select

    from app.models import Chunk, Document

    ordered_chunk_ids = list(dict.fromkeys(str(item) for item in chunk_ids if item))
    if not ordered_chunk_ids:
        return {
            "chunk_count": 0,
            "document_count": 0,
            "document_ids": [],
            "documents": [],
            "missing_chunk_ids": [],
            "missing_document_ids": [],
            "orphan_chunk_ids": [],
            "complete": True,
        }
    chunk_rows = list(
        db.scalars(select(Chunk).where(Chunk.id.in_(ordered_chunk_ids))).all()
    )
    chunks_by_id = {str(chunk.id): chunk for chunk in chunk_rows}
    document_ids = sorted(
        {str(chunk.document_id) for chunk in chunk_rows if chunk.document_id}
    )
    document_rows = list(
        db.scalars(select(Document).where(Document.id.in_(document_ids))).all()
    )
    documents_by_id = {str(document.id): document for document in document_rows}
    chunks_by_document: dict[str, list[str]] = {
        document_id: [] for document_id in document_ids
    }
    for chunk_id in ordered_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None and chunk.document_id:
            chunks_by_document[str(chunk.document_id)].append(chunk_id)
    missing_chunk_ids = [
        chunk_id for chunk_id in ordered_chunk_ids if chunk_id not in chunks_by_id
    ]
    missing_document_ids = [
        document_id
        for document_id in document_ids
        if document_id not in documents_by_id
    ]
    orphan_chunk_ids = [
        chunk_id
        for chunk_id in ordered_chunk_ids
        if chunk_id in chunks_by_id and not chunks_by_id[chunk_id].document_id
    ]
    return {
        "chunk_count": len(ordered_chunk_ids),
        "document_count": len(document_ids),
        "document_ids": document_ids,
        "documents": [
            {
                "document_id": document_id,
                "title": (
                    str(documents_by_id[document_id].title)
                    if document_id in documents_by_id
                    else None
                ),
                "chunk_ids": chunks_by_document[document_id],
            }
            for document_id in document_ids
        ],
        "missing_chunk_ids": missing_chunk_ids,
        "missing_document_ids": missing_document_ids,
        "orphan_chunk_ids": orphan_chunk_ids,
        "complete": not missing_chunk_ids
        and not missing_document_ids
        and not orphan_chunk_ids,
    }


def _persisted_context_package(db, knowledge_base_id: str, trace):
    from sqlalchemy import select

    from app.models import ContextPackage

    package = db.scalar(
        select(ContextPackage)
        .where(
            ContextPackage.knowledge_base_id == knowledge_base_id,
            ContextPackage.retrieval_trace_id == trace.id,
        )
        .order_by(
            ContextPackage.created_at.desc(),
            ContextPackage.id.asc(),
        )
        .limit(1)
    )
    if package is None:
        raise RuntimeError(
            "Persisted retrieval trace has no ContextPackage in the selected "
            f"knowledge base: {trace.id}"
        )
    trace_hit_ids = list(trace.result_chunk_ids_json or [])
    package_hit_ids = list(package.hit_chunk_ids_json or [])
    if package_hit_ids != trace_hit_ids:
        raise RuntimeError(
            "Persisted ContextPackage hit ids do not match its RetrievalTrace: "
            f"trace={trace.id}, package={package.id}"
        )
    return package


def _trace_row(
    db,
    trace,
    *,
    source: str,
    context_package=None,
) -> tuple[dict, dict]:
    raw_trace = _raw_trace_payload(db, trace)
    row_gray_audit = audit_gray_zone_trace(
        raw_trace,
        require_gray_coverage=False,
    )
    retrieval_quality = audit_retrieval_quality(
        raw_trace,
        gray_zone_audit=row_gray_audit,
    )
    steps = list(raw_trace.get("steps") or [])
    result_chunk_ids = list(trace.result_chunk_ids_json or [])
    document_coverage = _document_coverage(db, result_chunk_ids)
    return (
        {
            "source": source,
            "query": str(trace.query),
            "trace_id": str(trace.id),
            "retrieval_mode": str(trace.retrieval_mode),
            "created_at": (
                trace.created_at.isoformat()
                if trace.created_at is not None
                else None
            ),
            "retrieval_granularity": (
                getattr(trace, "diagnostics_json", None) or {}
            ).get("retrieval_granularity"),
            "result_count": len(result_chunk_ids),
            "top_chunk_ids": result_chunk_ids,
            # Backward-compatible report key; unlike the legacy evaluator this
            # is the complete returned top-k, not a five-item preview.
            "top_chunks": result_chunk_ids,
            "document_coverage": document_coverage,
            "context_package_id": (
                str(context_package.id) if context_package is not None else None
            ),
            "context_package_hit_chunk_ids": (
                list(context_package.hit_chunk_ids_json or [])
                if context_package is not None
                else None
            ),
            "context_package_persisted": (
                True
                if context_package is not None
                else (False if source == "new_execution" else None)
            ),
            "context_package_binding_complete": bool(
                context_package is not None
                and str(context_package.retrieval_trace_id or "")
                == str(trace.id)
                and list(context_package.hit_chunk_ids_json or [])
                == result_chunk_ids
            ),
            "frontier_step_count": sum(
                1
                for step in steps
                if step.get("action") == "walk_graph_frontier"
            ),
            "dominance_pruned_count": sum(
                int(step.get("dominance_pruned_count") or 0)
                for step in steps
            ),
            "quality_gate": retrieval_quality,
        },
        raw_trace,
    )


def _persisted_traces(db, knowledge_base_id: str, args: argparse.Namespace):
    from sqlalchemy import select

    from app.models import RetrievalTrace

    limit = max(1, min(int(args.limit), 100))
    requested_trace_ids = sorted({str(item) for item in args.trace_ids if item})
    requested_queries = sorted({str(item) for item in args.queries if item})
    base_query = select(RetrievalTrace).where(
        RetrievalTrace.knowledge_base_id == knowledge_base_id
    )
    if requested_trace_ids:
        traces = list(
            db.scalars(
                base_query.where(RetrievalTrace.id.in_(requested_trace_ids))
                .order_by(
                    RetrievalTrace.created_at.desc(),
                    RetrievalTrace.id.asc(),
                )
                .limit(limit)
            ).all()
        )
    elif requested_queries:
        # Resolve one newest trace for every exact requested query. A single
        # query can have many historical traces; applying one global LIMIT to
        # an IN query could otherwise silently omit another requested target.
        traces = []
        for requested_query in requested_queries:
            trace = db.scalar(
                base_query.where(RetrievalTrace.query == requested_query)
                .order_by(
                    RetrievalTrace.created_at.desc(),
                    RetrievalTrace.id.asc(),
                )
                .limit(1)
            )
            if trace is not None:
                traces.append(trace)
    else:
        traces = list(
            db.scalars(
                base_query.order_by(
                    RetrievalTrace.created_at.desc(),
                    RetrievalTrace.id.asc(),
                ).limit(limit)
            ).all()
        )
    if requested_trace_ids:
        found = {str(trace.id) for trace in traces}
        missing = sorted(set(requested_trace_ids) - found)
        if missing:
            raise SystemExit(
                "Persisted retrieval trace ids were not found in the selected "
                f"knowledge base: {missing}"
            )
    if requested_queries:
        found_queries = {str(trace.query) for trace in traces}
        missing_queries = sorted(set(requested_queries) - found_queries)
        if missing_queries:
            raise SystemExit(
                "Persisted retrieval queries were not found in the selected "
                f"knowledge base: {missing_queries}"
            )
    return traces


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    if int(args.limit) < 1 or int(args.limit) > 100:
        raise SystemExit("--limit must be between 1 and 100")
    bounded_limit = int(args.limit)
    typed_replay_source_trace_id = str(
        getattr(args, "typed_replay_source_trace_id", None) or ""
    )
    typed_replay_count = int(getattr(args, "typed_replay_count", 2))
    if typed_replay_count < 2 or typed_replay_count > 5:
        raise SystemExit("--typed-replay-count must be between 2 and 5")
    if typed_replay_source_trace_id and (args.queries or args.trace_ids):
        raise SystemExit(
            "--typed-replay-source-trace-id cannot be combined with --query or --trace-id"
        )
    if args.execute and args.trace_ids:
        raise SystemExit("--trace-id is read-only and cannot be combined with --execute")
    if args.execute and not args.queries and not typed_replay_source_trace_id:
        raise SystemExit(
            "--execute requires at least one explicit --query or "
            "--typed-replay-source-trace-id"
        )
    if len(args.queries) > bounded_limit or len(args.trace_ids) > bounded_limit:
        raise SystemExit(
            "Explicit query/trace targets exceed --limit; split the evaluation "
            "instead of silently truncating its write/read scope"
        )
    if int(args.top_k) < 1 or int(args.top_k) > 50:
        raise SystemExit(
            "--top-k must be between 1 and 50, matching SearchRequest"
        )
    retrieval_granularity = getattr(args, "retrieval_granularity", "mid")
    if retrieval_granularity not in {"coarse", "mid"}:
        raise SystemExit(
            "--retrieval-granularity must be coarse or mid; chunk is not an "
            "active mode in docs/technical-spec.md or the shared API contract"
        )

    mode = (
        "typed_replay_execute"
        if args.execute and typed_replay_source_trace_id
        else (
            "typed_replay_dry_run"
            if typed_replay_source_trace_id
            else ("execute" if args.execute else "dry_run")
        )
    )
    model_io_runtime = (
        prepare_runtime_for_model_io() if args.execute else None
    )
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        rows: list[dict] = []
        raw_traces: list[dict] = []
        if args.execute:
            from app.models import ContextPackage, RetrievalTrace
            from app.schemas import SearchFilters
            from app.services.agent_graph import execute_typed_retrieval_plan
            from app.services.context_graph import (
                QueryEmbeddingRequestMemo,
                build_context_package,
            )
            from app.services.retrieval import search_chunks_with_audit
            from app.services.storage import run_bounded_source_io
            from sqlalchemy.orm.attributes import flag_modified

            try:
                if typed_replay_source_trace_id:
                    source_trace = db.get(
                        RetrievalTrace, typed_replay_source_trace_id
                    )
                    if (
                        source_trace is None
                        or str(source_trace.knowledge_base_id)
                        != str(knowledge_base.id)
                    ):
                        raise RuntimeError(
                            "typed replay source trace was not found in the selected knowledge base"
                        )
                    source_diagnostics = dict(
                        source_trace.diagnostics_json or {}
                    )
                    controls = dict(
                        source_diagnostics.get("typed_action_controls") or {}
                    )
                    control_hash = str(
                        source_diagnostics.get("typed_action_control_hash")
                        or ""
                    )
                    if not controls or controls.get("control_hash") != control_hash:
                        raise RuntimeError(
                            "typed replay source has no exact persisted control-card binding"
                        )
                    query_facets = dict(source_trace.query_facets_json or {})
                    conversation_state_audit = dict(
                        source_diagnostics.get("conversation_state") or {}
                    )
                    source_gray_audit = audit_gray_zone_trace(
                        _raw_trace_payload(db, source_trace),
                        require_gray_coverage=False,
                    )
                    if (
                        not source_gray_audit.get("pass")
                        or not source_gray_audit.get("gray_zone_coverage")
                    ):
                        raise RuntimeError(
                            "typed replay source does not contain a complete deterministic gray decision"
                        )
                    request_memo = QueryEmbeddingRequestMemo()
                    for replay_index in range(typed_replay_count):
                        search_result = await execute_typed_retrieval_plan(
                            db,
                            knowledge_base_id=knowledge_base.id,
                            query=str(source_trace.query),
                            filters=SearchFilters.model_validate(
                                source_trace.filters_json or {}
                            ),
                            query_facets=query_facets,
                            controls=controls,
                            conversation_state_scope_hash=str(
                                source_trace.conversation_state_scope_hash or ""
                            ),
                            conversation_state_audit=conversation_state_audit,
                            policy_identity_frozen=True,
                            frozen_policy_state_hash=source_trace.policy_state_hash,
                            allow_cache_read=False,
                            query_embedding_request_memo=request_memo,
                        )
                        if (
                            search_result is None
                            or search_result.context_package is not None
                        ):
                            raise RuntimeError(
                                "typed replay executor did not create an independent retrieval trace"
                            )
                        trace = search_result.trace
                        trace.diagnostics_json = {
                            **(trace.diagnostics_json or {}),
                            "typed_action_control_hash": control_hash,
                            "typed_replay_acceptance": {
                                "protocol_version": "persisted_typed_control_independent_replay_v1",
                                "source_trace_id": typed_replay_source_trace_id,
                                "replay_index": replay_index,
                                "replanning_model_call_count": 0,
                                "gray_zone_model_call_count": 0,
                            },
                        }
                        flag_modified(trace, "diagnostics_json")
                        context_package = await run_bounded_source_io(
                            build_context_package,
                            db,
                            knowledge_base_id=knowledge_base.id,
                            query=str(source_trace.query),
                            trace=trace,
                            results=search_result.results,
                            token_budget=int(
                                controls["context_package_token_budget"]
                            ),
                            restore_per_chunk_budget=int(
                                controls["structure_restore_per_chunk_budget"]
                            ),
                            snapshot_verifier=search_result.snapshot_verifier,
                        )
                        row, raw_trace = _trace_row(
                            db,
                            trace,
                            source="typed_control_independent_replay",
                            context_package=context_package,
                        )
                        query_embedding_audit = dict(
                            (trace.diagnostics_json or {}).get(
                                "query_embedding_execution"
                            )
                            or {}
                        )
                        row["typed_replay_execution_audit"] = {
                            "protocol_version": "persisted_typed_control_independent_replay_v1",
                            "replanning_model_call_count": 0,
                            "gray_zone_model_call_count": int(
                                (trace.convergence_json or {}).get(
                                    "gray_zone_model_call_count"
                                )
                                or 0
                            ),
                            "query_embedding_model_call_count": int(
                                query_embedding_audit.get(
                                    "query_embedding_model_call_count"
                                )
                                or 0
                            ),
                            "query_embedding_request_memo_hit": bool(
                                query_embedding_audit.get("request_memo_hit")
                            ),
                        }
                        rows.append(row)
                        raw_traces.append(raw_trace)
                else:
                    for query in list(args.queries):
                        _results, model_audit = await search_chunks_with_audit(
                            db,
                            knowledge_base.id,
                            query,
                            SearchFilters(),
                            args.top_k,
                            retrieval_granularity=retrieval_granularity,
                        )
                        trace_id = str(model_audit.get("retrieval_trace_id") or "")
                        context_package_id = str(
                            model_audit.get("context_package_id") or ""
                        )
                        if not trace_id or not context_package_id:
                            raise RuntimeError(
                                "Audited retrieval did not produce both a RetrievalTrace "
                                "and ContextPackage identity"
                            )
                        trace = db.get(RetrievalTrace, trace_id)
                        context_package = db.get(ContextPackage, context_package_id)
                        if trace is None or context_package is None:
                            raise RuntimeError(
                                "Audited retrieval artifacts were not persisted in the "
                                "current transaction"
                            )
                        if str(context_package.retrieval_trace_id or "") != trace_id:
                            raise RuntimeError(
                                "Persisted ContextPackage is not bound to the emitted "
                                "RetrievalTrace"
                            )
                        if list(context_package.hit_chunk_ids_json or []) != list(
                            trace.result_chunk_ids_json or []
                        ):
                            raise RuntimeError(
                                "Persisted ContextPackage hit ids do not match the "
                                "emitted RetrievalTrace result ids"
                            )
                        row, raw_trace = _trace_row(
                            db,
                            trace,
                            source="new_execution",
                            context_package=context_package,
                        )
                        rows.append(row)
                        raw_traces.append(raw_trace)
                db.commit()
            except Exception:
                db.rollback()
                raise
        else:
            if typed_replay_source_trace_id:
                from app.models import RetrievalTrace

                source_trace = db.get(
                    RetrievalTrace, typed_replay_source_trace_id
                )
                if (
                    source_trace is None
                    or str(source_trace.knowledge_base_id)
                    != str(knowledge_base.id)
                ):
                    raise RuntimeError(
                        "typed replay source trace was not found in the selected knowledge base"
                    )
                persisted_traces = [source_trace]
            else:
                persisted_traces = _persisted_traces(
                    db, knowledge_base.id, args
                )
            for trace in persisted_traces:
                context_package = _persisted_context_package(
                    db,
                    knowledge_base.id,
                    trace,
                )
                row, raw_trace = _trace_row(
                    db,
                    trace,
                    source="persisted_replay",
                    context_package=context_package,
                )
                rows.append(row)
                raw_traces.append(raw_trace)

        gray_zone_trace_audit = audit_gray_zone_traces(
            raw_traces,
            require_gray_coverage=bool(args.require_gray_coverage),
        )
        retrieval_checks_pass = bool(rows) and all(
            row["trace_id"]
            and row["result_count"] > 0
            and row["quality_gate"]["pass"]
            and row.get("document_coverage", {}).get("complete", True)
            and row.get("context_package_binding_complete") is True
            for row in rows
        )
        typed_replay_contract_pass = True
        if typed_replay_source_trace_id and args.execute:
            typed_replay_contract_pass = (
                len(rows) == typed_replay_count
                and len({str(row["trace_id"]) for row in rows})
                == typed_replay_count
                and all(
                    int(
                        (row.get("typed_replay_execution_audit") or {}).get(
                            "replanning_model_call_count"
                        )
                        or 0
                    )
                    == 0
                    for row in rows
                )
                and sum(
                    int(
                        (row.get("typed_replay_execution_audit") or {}).get(
                            "query_embedding_model_call_count"
                        )
                        or 0
                    )
                    for row in rows
                )
                <= 1
            )
        payload = {
            "script": "evaluate_layered_retrieval",
            "mode": mode,
            "execute": bool(args.execute),
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "model_io_runtime": model_io_runtime,
            "targets": {
                "persisted_trace_ids": list(args.trace_ids),
                "queries": list(args.queries),
                "typed_replay_source_trace_id": (
                    typed_replay_source_trace_id or None
                ),
                "typed_replay_count": (
                    typed_replay_count
                    if typed_replay_source_trace_id
                    else None
                ),
                "limit": bounded_limit,
                "top_k_on_execute": int(args.top_k),
                "retrieval_granularity_on_execute": (
                    retrieval_granularity
                ),
            },
            "impact": (
                "replay one PostgreSQL-frozen typed-action control card through "
                "the production executor with retrieval cache reads disabled, "
                "commit independent RetrievalTrace and ContextPackage artifacts, "
                "reuse one request-scoped query embedding memo, and perform zero "
                "planner or gray-zone model calls"
                if args.execute and typed_replay_source_trace_id
                else (
                    "run search_chunks_with_audit for the exact supplied queries, "
                    "use the same audited retrieval/context-package service as "
                    "POST /search, and commit new RetrievalTrace and ContextPackage "
                    "artifacts"
                    if args.execute
                    else (
                        "read only the exact typed replay source trace and print "
                        "the bounded execution plan; no model, Qdrant, Redis, "
                        "cache, or production-state writes"
                        if typed_replay_source_trace_id
                        else "read only existing PostgreSQL retrieval traces and graph steps; no model, Qdrant, Redis, cache, or production-state writes"
                    )
                )
            ),
            "query_count": len(rows),
            "rows": rows,
            "gray_zone_trace_audit": gray_zone_trace_audit,
            "checks": {
                "retrieval_results": retrieval_checks_pass,
                "gray_zone_trace_audit": bool(gray_zone_trace_audit["pass"]),
                "gray_zone_coverage": bool(gray_zone_trace_audit["gray_zone_coverage"]),
                "gray_zone_coverage_required": bool(args.require_gray_coverage),
                "versioned_retrieval_quality_gate": bool(rows)
                and all(row["quality_gate"]["pass"] for row in rows),
                "production_state_write_requires_execute": True,
                "all_top_k_ids_reported": bool(rows)
                and all(
                    len(row.get("top_chunk_ids") or []) == row["result_count"]
                    for row in rows
                ),
                "document_coverage_complete": bool(rows)
                and all(
                    bool((row.get("document_coverage") or {}).get("complete"))
                    for row in rows
                ),
                "deterministic_gray_model_call_count": 0,
                "deterministic_gray_replay_provider_free": (
                    type(gray_zone_trace_audit.get("raw_record_count")) is int
                    and type(
                        gray_zone_trace_audit.get(
                            "explicit_zero_model_call_record_count"
                        )
                    )
                    is int
                    and gray_zone_trace_audit[
                        "explicit_zero_model_call_record_count"
                    ]
                    == gray_zone_trace_audit["raw_record_count"]
                ),
                "typed_replay_replanning_model_call_count_zero": (
                    typed_replay_contract_pass
                    if typed_replay_source_trace_id and args.execute
                    else None
                ),
                "typed_replay_query_embedding_request_count_bounded": (
                    sum(
                        int(
                            (row.get("typed_replay_execution_audit") or {}).get(
                                "query_embedding_model_call_count"
                            )
                            or 0
                        )
                        for row in rows
                    )
                    <= 1
                    if typed_replay_source_trace_id and args.execute
                    else None
                ),
                "execute_context_packages_persisted": (
                    all(
                        bool(row.get("context_package_persisted"))
                        for row in rows
                    )
                    if args.execute
                    else None
                ),
                "all_context_package_bindings_complete": bool(rows)
                and all(
                    row.get("context_package_binding_complete") is True
                    for row in rows
                ),
            },
            "transaction": {
                "database_committed": bool(args.execute),
                "default_persisted_replay": not args.execute,
            },
            "pass": retrieval_checks_pass
            and bool(gray_zone_trace_audit["pass"])
            and typed_replay_contract_pass,
        }
        report = write_report("evaluate_layered_retrieval", payload)
        print(
            json.dumps(
                {
                    "output": str(report),
                    "pass": payload["pass"],
                    "mode": mode,
                    "query_count": len(rows),
                },
                ensure_ascii=False,
                default=str,
            )
        )
        if not payload["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
