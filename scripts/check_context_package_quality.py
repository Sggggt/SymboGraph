from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import prepare_runtime_for_model_io, resolve_knowledge_base, session_scope, write_report
from _quality_gate import (
    _source_fact_hash,
    audit_context_package_quality,
)


_FRONTIER_ACTIONS_BY_LAYER = {
    "coarse": ("staged_priority_queue_walk", "walk_graph_frontier"),
    "mid": (
        "drill_down_each_coarse_or_direct_mid_entry",
        "walk_graph_frontier",
    ),
    "chunk": ("walk_graph_frontier",),
}


def audit_retrieval_frontier_convergence(public_trace: dict | None) -> dict:
    trace = public_trace or {}
    retrieval_granularity = trace.get("retrieval_granularity")
    steps = list(trace.get("steps") or [])
    step_by_layer = {
        layer: next(
            (
                step
                for step in steps
                if step.get("layer") == layer
                and step.get("action") in actions
            ),
            None,
        )
        for layer, actions in _FRONTIER_ACTIONS_BY_LAYER.items()
    }
    participating_layers = (
        ["mid", "chunk"]
        if retrieval_granularity == "mid"
        else ["coarse", "mid", "chunk"]
        if retrieval_granularity == "coarse"
        else []
    )
    checks = {
        "retrieval_granularity_is_supported": retrieval_granularity
        in {"coarse", "mid"},
        "coarse_skip_matches_retrieval_granularity": True,
        "each_participating_layer_has_frontier_path_convergence": all(
            bool(
                step
                and (step.get("popped_frontier_state") or {}).get(
                    "node_id"
                )
                and (step.get("popped_frontier_state") or {}).get("path")
                and step.get("stop_reason")
            )
            for layer in participating_layers
            for step in [step_by_layer.get(layer)]
        ),
        "gray_zone_model_call_count_zero": trace.get(
            "gray_zone_model_call_count"
        )
        == 0,
    }
    if retrieval_granularity == "mid":
        coarse_queue = (trace.get("stage_queues") or {}).get("coarse") or {}
        trace_diagnostics = trace.get("trace_diagnostics") or {}
        checks["coarse_skip_matches_retrieval_granularity"] = bool(
            coarse_queue.get("skipped_by_granularity") == "mid"
            and coarse_queue.get("reason") == "skipped_by_granularity=mid"
            and trace_diagnostics.get("coarse_skipped_reason")
            == "skipped_by_granularity=mid"
        )
    return {
        "retrieval_granularity": retrieval_granularity,
        "participating_layers": participating_layers,
        "checks": checks,
        "pass": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval and verify context package closure, citation spans, and graph expansion paths.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--query")
    target.add_argument(
        "--context-package-id",
        help="Read and audit one already-persisted context package and its trace without running retrieval.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run retrieval and persist its trace/context package. Omit for a read-only execution plan.",
    )
    args = parser.parse_args()
    if args.context_package_id and args.execute:
        parser.error("--context-package-id cannot be combined with --execute")
    return args


def persisted_context_package_quality_snapshot(
    db,
    package_id: str,
) -> dict:
    from app.models import Chunk, Document, DocumentVersion, RetrievalTrace
    from app.services.context_graph import SnapshotIntegrityVerifier
    from app.services.retrieval import get_context_package

    public_package = get_context_package(db, package_id)
    if public_package is None:
        raise RuntimeError(
            "context package disappeared before quality audit: "
            f"{package_id}"
        )
    trace_id = public_package.get("retrieval_trace_id")
    trace = db.get(RetrievalTrace, trace_id) if trace_id else None
    trace_path_edge_ids = list(
        dict.fromkeys(
            edge_id
            for label in (
                (trace.path_labels_json or []) if trace else []
            )
            for edge_id in (label.get("path_edge_ids") or [])
        )
    )
    public_chunks = (
        (public_package.get("package") or {}).get("chunks") or []
    )
    source_facts = []
    snapshot_verifier = SnapshotIntegrityVerifier()
    for public_chunk in public_chunks:
        chunk_id = str(public_chunk.get("chunk_id") or "")
        chunk = db.get(Chunk, chunk_id)
        if chunk is None:
            raise RuntimeError(
                "context package references a missing PostgreSQL chunk: "
                f"{chunk_id}"
            )
        document = db.get(Document, chunk.document_id)
        document_version = db.get(
            DocumentVersion, chunk.document_version_id
        )
        if document is None or document_version is None:
            raise RuntimeError(
                "context package source PostgreSQL facts are incomplete: "
                f"chunk_id={chunk_id}"
            )
        verification = snapshot_verifier.verify(
            db,
            chunk=chunk,
            document=document,
            document_version=document_version,
        )
        source_fact = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_version_id": chunk.document_version_id,
            "stored_chunk_text": chunk.text,
            "stored_chunk_text_hash": chunk.text_hash,
            "stored_char_span": [
                int(chunk.char_start),
                int(chunk.char_end),
            ],
            "document_version_checksum": document_version.checksum,
            "document_version_storage_path": (
                document_version.storage_path
            ),
            "logical_source_path": document.source_path,
            "snapshot_protocol_version": verification.get(
                "protocol_version"
            ),
            "snapshot_observed_checksum": verification.get(
                "checksum"
            ),
            "snapshot_size_bytes": verification.get("size_bytes"),
            "snapshot_verified": verification.get("verified"),
        }
        source_fact["fact_hash"] = _source_fact_hash(source_fact)
        source_facts.append(source_fact)
    return {
        "id": public_package["id"],
        "retrieval_trace_id": trace_id,
        "package_hash": public_package.get("package_hash"),
        "hit_chunk_ids": public_package.get("hit_chunk_ids") or [],
        "restored_chunk_ids": (
            public_package.get("restored_chunk_ids") or []
        ),
        "bridge_chunk_ids": (
            public_package.get("bridge_chunk_ids") or []
        ),
        "parent_structure_node_ids": public_package.get(
            "parent_structure_node_ids"
        )
        or [],
        "graph_path_ids": public_package.get("graph_path_ids") or [],
        "trace_path_edge_ids": trace_path_edge_ids,
        "dedupe_keys": public_package.get("dedupe_keys") or [],
        "token_budget": public_package.get("token_budget"),
        "token_count": public_package.get("token_count"),
        "chunks": public_chunks,
        "source_facts": source_facts,
        "citation_spans": [
            item.get("source_span") or {}
            for item in public_package.get("citation_spans") or []
        ],
        "diagnostics": public_package.get("diagnostics") or {},
    }


def audit_persisted_context_package(
    db,
    *,
    package_id: str,
    expected_knowledge_base_id: str,
) -> dict:
    from sqlalchemy import select

    from app.models import (
        ContextPackage,
        GraphRetrievalStep,
        RetrievalTrace,
    )
    from app.services.retrieval import (
        get_context_package,
        get_retrieval_trace_steps,
    )

    package = db.get(ContextPackage, package_id)
    if package is None:
        raise SystemExit(f"Context package not found: {package_id}")
    if str(package.knowledge_base_id) != str(expected_knowledge_base_id):
        raise SystemExit(
            "Context package does not belong to the selected knowledge base: "
            f"package_id={package_id}, "
            f"package_knowledge_base_id={package.knowledge_base_id}, "
            f"selected_knowledge_base_id={expected_knowledge_base_id}"
        )

    public_package = get_context_package(db, package.id)
    if public_package is None:
        raise RuntimeError(
            f"context package disappeared before quality audit: {package.id}"
        )
    trace = (
        db.get(RetrievalTrace, package.retrieval_trace_id)
        if package.retrieval_trace_id
        else None
    )
    if trace and str(trace.knowledge_base_id) != str(expected_knowledge_base_id):
        raise RuntimeError(
            "context package retrieval trace crosses knowledge-base scope: "
            f"package_id={package.id}, trace_id={trace.id}"
        )
    seed_step = (
        db.scalar(
            select(GraphRetrievalStep).where(
                GraphRetrievalStep.retrieval_trace_id
                == package.retrieval_trace_id,
                GraphRetrievalStep.layer == "chunk",
                GraphRetrievalStep.action
                == "select_seeds_from_mid_rq_membership",
            )
        )
        if package.retrieval_trace_id
        else None
    )
    rq_candidates = (
        ((seed_step.output_json or {}).get("candidate_rq") or {})
        if seed_step
        else {}
    )
    public_trace = get_retrieval_trace_steps(
        db, package.retrieval_trace_id
    )
    retrieval_frontier_audit = audit_retrieval_frontier_convergence(
        public_trace
    )
    quality_gate = audit_context_package_quality(
        persisted_context_package_quality_snapshot(db, package.id)
    )
    contexts = (package.package_json or {}).get("chunks") or []
    citation_spans = package.citation_spans_json or []
    graph_paths = package.concept_path_json or []
    checks = {
        "has_retrieval_trace": bool(package.retrieval_trace_id),
        "has_contexts": bool(contexts),
        "has_previous_or_next_context": any(
            item.get("role") in {"restored_context", "bridge"}
            for item in contexts
        ),
        "has_parent_structure": any(
            item.get("structure_path") for item in contexts
        ),
        "has_citation_spans": bool(citation_spans),
        "citation_spans_have_raw_addresses": all(
            span.get("document_version_id")
            and span.get("chunk_id")
            and span.get("char_span")
            and span.get("page_range") is not None
            and span.get("section_path")
            and span.get("bbox") is not None
            and span.get("context_package_id")
            and span.get("retrieval_trace_id")
            for span in citation_spans
        ),
        "contexts_have_source_span_and_closure": all(
            item.get("source_span")
            and item.get("structure_closure")
            and item.get("why_selected")
            and item.get("dedupe_key")
            for item in contexts
        ),
        "has_graph_path": bool(graph_paths),
        "has_rq_query_path": bool(
            (((trace.diagnostics_json or {}).get("rq") or {}).get(
                "query_rq_path"
            ))
            if trace
            else False
        ),
        "has_rq_membership_seed_step": bool(seed_step),
        "has_rq_candidate_metrics": any(
            "lcp_depth" in candidate and "residual_distance" in candidate
            for candidate in rq_candidates.values()
            if isinstance(candidate, dict)
        ),
        **retrieval_frontier_audit["checks"],
        "token_budget_recorded": package.token_budget > 0
        and package.token_count >= 0,
        "token_budget_not_exceeded": package.token_count
        <= package.token_budget,
        "versioned_context_package_quality_gate": bool(
            quality_gate["pass"]
        ),
    }
    return {
        "context_package_id": package.id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "query": package.query,
        "checks": checks,
        "pass": all(checks.values()),
        "diagnostics": package.diagnostics_json or {},
        "retrieval_frontier_audit": retrieval_frontier_audit,
        "quality_gate": quality_gate,
    }


async def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    context_package_id = getattr(args, "context_package_id", None)
    if context_package_id and args.execute:
        raise SystemExit(
            "--context-package-id cannot be combined with --execute"
        )
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        if context_package_id:
            audit = audit_persisted_context_package(
                db,
                package_id=context_package_id,
                expected_knowledge_base_id=knowledge_base.id,
            )
            payload = {
                "script": "check_context_package_quality",
                "execute": False,
                "mode": "persisted_context_package_read_only",
                "knowledge_base_id": knowledge_base.id,
                "knowledge_base_name": knowledge_base.name,
                **audit,
                "targets": {
                    "context_package_id": audit["context_package_id"],
                    "retrieval_trace_id": audit["retrieval_trace_id"],
                    "knowledge_base_id": knowledge_base.id,
                },
                "impact": (
                    "read-only PostgreSQL and immutable source snapshot audit; "
                    "no retrieval, model-provider, Qdrant, Redis, or database writes"
                ),
            }
            report = write_report("check_context_package_quality", payload)
            print(
                json.dumps(
                    {
                        "output": str(report),
                        "pass": payload["pass"],
                        "mode": payload["mode"],
                        "checks": payload["checks"],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
            if not payload["pass"]:
                raise SystemExit(1)
            return
        if not args.execute:
            payload = {
                "script": "check_context_package_quality",
                "execute": False,
                "mode": "dry_run",
                "knowledge_base_id": knowledge_base.id,
                "knowledge_base_name": knowledge_base.name,
                "query": args.query,
                "top_k": args.top_k,
                "targets": {
                    "operations": ["layered_search", "retrieval_trace", "context_package"],
                    "knowledge_base_id": knowledge_base.id,
                },
                "impact": "no PostgreSQL/Qdrant/Redis/model-provider writes or calls",
                "next_step": "rerun with --execute after reviewing the target and query",
            }
            report = write_report("check_context_package_quality", payload)
            print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))
            return

        from app.schemas import SearchFilters
        from app.services.context_graph import (
            build_context_package,
            layered_search,
        )

        model_io_runtime = prepare_runtime_for_model_io()
        retrieval = await layered_search(db, knowledge_base.id, args.query, SearchFilters(), args.top_k)
        package = build_context_package(
            db,
            knowledge_base_id=knowledge_base.id,
            query=args.query,
            trace=retrieval.trace,
            results=retrieval.results,
            token_budget=args.top_k * 800,
        )
        audit = audit_persisted_context_package(
            db,
            package_id=package.id,
            expected_knowledge_base_id=knowledge_base.id,
        )
        payload = {
            "script": "check_context_package_quality",
            "execute": True,
            "mode": "execute",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "query": audit["query"],
            "model_io_runtime": model_io_runtime,
            **audit,
            "targets": {
                "context_package_id": audit["context_package_id"],
                "retrieval_trace_id": audit["retrieval_trace_id"],
            },
            "impact": "persisted retrieval trace and context package for the selected knowledge base",
        }
        db.commit()
        report = write_report("check_context_package_quality", payload)
        print(json.dumps({"output": str(report), "pass": payload["pass"], "checks": audit["checks"]}, ensure_ascii=False, default=str))
        if not payload["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
