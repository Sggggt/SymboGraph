"""Immutable retrieval snapshot cloning for reflection context restoration."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, ContextPackage, GraphRetrievalStep, RetrievalTrace
from app.services.agent_reflection import ReflectionContractError, reflection_hash
from app.services.context_graph import build_context_package, context_package_to_contexts
from app.services.reflection_sources import retain_reflection_sources, retention_origin
from app.services.reflection_expansion import EXPANSION_PROTOCOL, select_source_expansions, persist_source_expansions, copy_source_expansions


def restore_reflection_context(
    db: Session, *, source_package: ContextPackage, target_chunk_ids: list[str],
    preserve_chunk_ids: list[str], token_budget: int, restore_per_chunk_budget: int,
    query_facets: dict[str, Any] | None = None,
    reserved_token_budget: int = 0,
    restoration_focus: list[str] | None = None,
    packing_priority_chunk_ids: list[str] | None = None,
) -> tuple[ContextPackage, list[dict[str, Any]]]:
    source_trace = db.get(RetrievalTrace, source_package.retrieval_trace_id)
    if source_trace is None or source_trace.knowledge_base_id != source_package.knowledge_base_id:
        raise ReflectionContractError("reflection_source_trace_missing")
    source_items = {item["chunk_id"]: item for item in (source_package.package_json or {}).get("chunks", [])}
    priorities = list(packing_priority_chunk_ids or [])
    if (len(priorities) > 256 or len(set(priorities)) != len(priorities)
            or not set(priorities).issubset(set(source_trace.result_chunk_ids_json or ()) | set(source_items))):
        raise ReflectionContractError('scope_packing_priority_outside_authorized_sources')
    targets = set(target_chunk_ids) | set(preserve_chunk_ids)
    if not target_chunk_ids or not targets.issubset(source_items) or len(targets) > 256:
        raise ReflectionContractError("reflection_restore_target_outside_package")
    chunks = list(db.scalars(select(Chunk).where(Chunk.id.in_(targets))))
    if len(chunks) != len(targets) or any(
        chunk.knowledge_base_id != source_package.knowledge_base_id or chunk.state != "active"
        or chunk.document_version_id != source_items[chunk.id]["document_version_id"]
        for chunk in chunks
    ):
        raise ReflectionContractError("reflection_restore_source_scope_changed")
    if token_budget <= 0 or restore_per_chunk_budget < 0:
        raise ReflectionContractError("reflection_restore_budget_invalid")
    from app.services.citation_provenance import audit_citation_provenance
    from app.services.reflection_sources import source_citation
    anchor_audit = audit_citation_provenance(db, knowledge_base_id=source_package.knowledge_base_id,
        package=source_package, contexts=context_package_to_contexts(source_package),
        citations=[source_citation(source_items[cid], source_package) for cid in sorted(targets)])
    if not anchor_audit["all_valid"]:
        raise ReflectionContractError("reflection_restore_anchor_provenance_invalid")
    # Retained sources keep their original graph authority. Restore that
    # authority first, then carry its validated raw spans into this package.
    retained_targets = {cid: retention_origin(db, package=source_package, chunk_id=cid)
                        for cid in target_chunk_ids if source_items[cid]["role"] == "preserved_source"}
    if retained_targets:
        if priorities:
            raise ReflectionContractError('scope_packing_priority_requires_local_anchor')
        package = source_package
        local_targets = [cid for cid in target_chunk_ids if cid not in retained_targets]
        if local_targets:
            package, _ = restore_reflection_context(db, source_package=source_package,
                target_chunk_ids=local_targets, preserve_chunk_ids=preserve_chunk_ids,
                token_budget=token_budget, restore_per_chunk_budget=restore_per_chunk_budget, query_facets=query_facets,
                restoration_focus=restoration_focus)
        for cid, origin in retained_targets.items():
            if origin is None:
                raise ReflectionContractError("reflection_retained_restore_origin_missing")
            # Full ancestry is replayed before recursion, including its bound.
            from app.services.citation_provenance import audit_citation_provenance
            from app.services.reflection_sources import source_citation
            replay = audit_citation_provenance(db, knowledge_base_id=source_package.knowledge_base_id,
                package=source_package, contexts=context_package_to_contexts(source_package),
                citations=[source_citation(source_items[cid], source_package)])
            if not replay["all_valid"]:
                raise ReflectionContractError("reflection_retained_restore_origin_invalid")
            restored, _ = restore_reflection_context(db, source_package=origin[1], target_chunk_ids=[cid],
                preserve_chunk_ids=[cid], token_budget=token_budget,
                restore_per_chunk_budget=restore_per_chunk_budget, query_facets=query_facets, restoration_focus=restoration_focus)
            original_ids = {item["chunk_id"] for item in origin[1].package_json["chunks"]}
            package, _ = retain_reflection_sources(db, candidate_package=package, source_package=restored,
                preserve_chunk_ids=[item["chunk_id"] for item in restored.package_json["chunks"]
                    if item["chunk_id"] == cid or item["chunk_id"] not in original_ids], token_budget=token_budget)
        return package, context_package_to_contexts(package)
    witnesses = select_source_expansions(db, source_package=source_package, anchor_ids=target_chunk_ids,
        query_facets=query_facets or source_trace.query_facets_json or {}, per_anchor_budget=restore_per_chunk_budget,
        restoration_focus=restoration_focus)
    source_fields = {
        column.name: deepcopy(getattr(source_trace, column.name))
        for column in RetrievalTrace.__table__.columns if column.name not in {"id", "created_at"}
    }
    source_fact_hash = reflection_hash(source_fields)
    trace = RetrievalTrace(**source_fields)
    trace.diagnostics_json = {
        **{key: value for key, value in source_fields["diagnostics_json"].items() if key != "reflection_source_retention"},
        "reflection_restoration": {
            "protocol_version": EXPANSION_PROTOCOL,
            "source_retrieval_trace_id": source_trace.id,
            "source_context_package_id": source_package.id,
            "source_trace_fact_hash": source_fact_hash,
            "retrieval_executed": False, "gray_zone_model_call_count": 0,
            **({"restoration_focus": deepcopy(restoration_focus)} if restoration_focus else {}),
            **({'packing_priority_chunk_ids':priorities,'packing_priority_protocol':'scope_interval_repacking_v1'} if priorities else {}),
        },
    }
    db.add(trace)
    db.flush()
    for step in db.scalars(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == source_trace.id).order_by(GraphRetrievalStep.step_index)):
        if step.layer == "structure" and step.action_type == "restore_context_package":
            continue
        values = {
            column.name: deepcopy(getattr(step, column.name))
            for column in GraphRetrievalStep.__table__.columns
            if column.name not in {"id", "created_at", "retrieval_trace_id"}
        }
        db.add(GraphRetrievalStep(retrieval_trace_id=trace.id, **values))
    db.flush()
    results = []
    for chunk_id in source_trace.result_chunk_ids_json:
        item = source_items.get(chunk_id)
        why = (item or {}).get("why_selected") or {}
        summary = next((deepcopy(value) for value in source_package.diagnostics_json.get("node_contributions", []) if value.get("node_id") == chunk_id), None)
        if summary is None:
            # A budget-skipped hit is still authorized by its original trace.
            # Restore its exact accepted path facts without inventing a new hit.
            from app.services.context_graph import aggregate_node_path_contributions
            summaries = aggregate_node_path_contributions([deepcopy(label) for label in source_trace.path_labels_json
                if label.get("layer") == "chunk" and label.get("node_id") == chunk_id])
            summary = next((value for value in summaries if value["node_id"] == chunk_id), None)
            if summary is None:
                raise ReflectionContractError("reflection_source_path_summary_missing")
        results.append({
            "chunk_id": chunk_id,
            "metadata": {"traversal": {
                "path": [chunk_id], "path_edge_ids": deepcopy(why.get("path_edge_ids") or []),
                "covered_facets": deepcopy(why.get("covered_facets") or []),
                "evidence_roles": deepcopy(why.get("evidence_roles") or []),
                "contribution_summary": summary, "why_selected": "reflection_original_hit_replay",
            }},
        })
    package = build_context_package(
        db, knowledge_base_id=source_package.knowledge_base_id, query=source_package.query,
        trace=trace, results=results, token_budget=token_budget, restore_per_chunk_budget=restore_per_chunk_budget,
        reflection_source_package=source_package,
        reflection_restore_chunk_ids=target_chunk_ids,
        reflection_preserve_chunk_ids=[cid for cid in preserve_chunk_ids if source_items[cid]["role"] != "preserved_source"],
        reflection_expansion_chunk_ids=[item["chunk_id"] for item in witnesses],
        reserved_token_budget=reserved_token_budget,
        packing_priority_chunk_ids=priorities,
    )
    copy_source_expansions(db, source=source_package, target=package)
    persist_source_expansions(db, source=source_package, target=package, witnesses=witnesses)
    package, contexts = retain_reflection_sources(db, candidate_package=package, source_package=source_package,
        preserve_chunk_ids=preserve_chunk_ids, token_budget=token_budget)
    final = audit_citation_provenance(db, knowledge_base_id=package.knowledge_base_id, package=package, contexts=contexts,
        citations=[source_citation(item, package) for item in package.package_json["chunks"]], for_update=True)
    if not final["all_valid"]:
        raise ReflectionContractError("reflection_restore_provenance_invalid")
    return package, contexts
