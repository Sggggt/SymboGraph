from __future__ import annotations

import argparse
import asyncio
import json

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval and verify context package closure, citation spans, and graph expansion paths.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


async def main() -> None:
    from sqlalchemy import select

    from app.models import GraphRetrievalStep, RetrievalTrace
    from app.schemas import SearchFilters
    from app.services.context_graph import build_context_package, layered_search

    args = parse_args()
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        retrieval = await layered_search(db, knowledge_base.id, args.query, SearchFilters(), args.top_k)
        package = build_context_package(
            db,
            knowledge_base_id=knowledge_base.id,
            query=args.query,
            trace=retrieval.trace,
            results=retrieval.results,
            token_budget=args.top_k * 800,
        )
        contexts = (package.package_json or {}).get("chunks") or []
        citation_spans = package.citation_spans_json or []
        graph_paths = package.concept_path_json or []
        trace = db.get(RetrievalTrace, package.retrieval_trace_id) if package.retrieval_trace_id else None
        seed_step = (
            db.scalar(
                select(GraphRetrievalStep).where(
                    GraphRetrievalStep.retrieval_trace_id == package.retrieval_trace_id,
                    GraphRetrievalStep.layer == "chunk",
                    GraphRetrievalStep.action == "select_seeds_from_mid_rq_membership",
                )
            )
            if package.retrieval_trace_id
            else None
        )
        rq_candidates = ((seed_step.output_json or {}).get("candidate_rq") or {}) if seed_step else {}
        checks = {
            "has_retrieval_trace": bool(package.retrieval_trace_id),
            "has_contexts": bool(contexts),
            "has_previous_or_next_context": any((item.get("role") in {"restored_context", "bridge"}) for item in contexts),
            "has_parent_structure": any(item.get("structure_path") for item in contexts),
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
            "has_rq_query_path": bool((((trace.diagnostics_json or {}).get("rq") or {}).get("query_rq_path")) if trace else False),
            "has_rq_membership_seed_step": bool(seed_step),
            "has_rq_candidate_metrics": any(
                "lcp_depth" in candidate and "residual_distance" in candidate
                for candidate in rq_candidates.values()
                if isinstance(candidate, dict)
            ),
            "each_layer_has_frontier_path_convergence": all(
                bool(step and step.popped_frontier_state_json and step.stop_reason)
                for layer, actions in {
                    "coarse": ("staged_priority_queue_walk", "walk_graph_frontier"),
                    "mid": ("drill_down_each_coarse_or_direct_mid_entry", "walk_graph_frontier"),
                    "chunk": ("walk_graph_frontier",),
                }.items()
                for step in [
                    db.scalar(
                        select(GraphRetrievalStep).where(
                            GraphRetrievalStep.retrieval_trace_id == package.retrieval_trace_id,
                            GraphRetrievalStep.layer == layer,
                            GraphRetrievalStep.action.in_(actions),
                        )
                    )
                ]
            ),
            "token_budget_recorded": package.token_budget > 0 and package.token_count >= 0,
        }
        payload = {
            "script": "check_context_package_quality",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "query": args.query,
            "context_package_id": package.id,
            "retrieval_trace_id": package.retrieval_trace_id,
            "checks": checks,
            "pass": all(checks.values()),
            "diagnostics": package.diagnostics_json or {},
        }
        db.commit()
        report = write_report("check_context_package_quality", payload)
        print(json.dumps({"output": str(report), "pass": payload["pass"], "checks": checks}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
