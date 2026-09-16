"""Retain bound raw spans without inventing new retrieval hits or paths."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ContextPackage, ContextPackageSourceRetention, GraphRetrievalStep, RetrievalTrace, generate_uuid
from app.services.agent_reflection import ReflectionContractError, reflection_hash
from app.services.chunking import rough_token_count


RETENTION_PROTOCOL = "reflection_bound_source_retention_v1"
MAX_SOURCE_ANCESTORS = 32
MAX_CONTEXT_SOURCES = 256


def retention_identity(row: ContextPackageSourceRetention) -> dict[str, Any]:
    return {key: getattr(row, key) for key in (
        "knowledge_base_id", "target_context_package_id", "source_context_package_id",
        "source_retrieval_trace_id", "chunk_id", "protocol_version", "source_item_hash",
    )}


def physical_source_identity(item: dict[str, Any]) -> dict[str, Any]:
    identity = {key: deepcopy(value) for key, value in item.items()
                if key not in {"context_package_id", "retrieval_trace_id", "role", "why_selected", "dedupe_key"}}
    span = identity.get("source_span") or {}
    for key in ("context_package_id", "retrieval_trace_id", "verification_id", "source_binding_id"):
        span.pop(key, None)
    # This list describes the owning package, while all other closure facts
    # belong to the raw source and must remain identical across retention.
    (identity.get("structure_closure") or {}).pop("bridge_chunk_ids", None)
    return identity


def source_citation(item: dict[str, Any], package: ContextPackage) -> dict[str, Any]:
    span = deepcopy(item["source_span"])
    return {"chunk_id": item["chunk_id"], "document_id": item["document_id"],
        "document_version_id": item["document_version_id"], "source_span": span,
        "context_package_id": package.id, "retrieval_trace_id": package.retrieval_trace_id,
        "source_path": span["source_path"], "logical_source_path": span["logical_source_path"],
        "section": span.get("section_path"), "page_number": span["page_range"][0]}


def source_covers(candidate: dict[str, Any], required: dict[str, Any]) -> bool:
    if candidate.get("document_version_id") != required.get("document_version_id"):
        return False
    current = candidate["source_span"]["char_span"]
    old = required["source_span"]["char_span"]
    return (current[0] <= old[0] <= old[1] <= current[1]
            and str(candidate.get("content") or "")[old[0] - current[0]:old[1] - current[0]] == required.get("content"))


def retention_origin(db: Session, *, package: ContextPackage, chunk_id: str, for_update: bool = False, row=None):
    query = select(ContextPackageSourceRetention).where(
        ContextPackageSourceRetention.target_context_package_id == package.id,
        ContextPackageSourceRetention.chunk_id == chunk_id,
    )
    if for_update:
        query = query.with_for_update()
    row = row if row is not None else db.scalar(query)
    if row is None:
        return None
    source_query = select(ContextPackage).where(ContextPackage.id == row.source_context_package_id)
    if for_update:
        source_query = source_query.with_for_update()
    source = db.scalar(source_query)
    source_items = (source.package_json or {}).get("chunks", []) if source is not None else []
    original = next((item for item in source_items if item.get("chunk_id") == chunk_id), None)
    target = next((item for item in (package.package_json or {}).get("chunks", []) if item.get("chunk_id") == chunk_id), None)
    if (source is None or original is None or target is None
        or row.knowledge_base_id != package.knowledge_base_id or source.knowledge_base_id != package.knowledge_base_id
        or source.id == package.id or source.retrieval_trace_id != row.source_retrieval_trace_id
        or row.protocol_version != RETENTION_PROTOCOL
        or row.source_item_hash != reflection_hash(original)
        or row.retention_hash != reflection_hash(retention_identity(row))
        or physical_source_identity(original) != physical_source_identity(target)):
        raise ReflectionContractError("reflection_retained_source_identity_mismatch")
    return row, source, original


def audit_retained_sources(db: Session, *, package: ContextPackage, citations: list[dict], for_update: bool, ancestors: tuple[str, ...]):
    from app.services.citation_provenance import audit_citation_provenance
    from app.services.context_graph import context_package_to_contexts

    query = select(ContextPackageSourceRetention).where(ContextPackageSourceRetention.target_context_package_id == package.id)
    if for_update:
        query = query.with_for_update()
    rows = list(db.scalars(query))
    declared = (package.diagnostics_json or {}).get("source_retention") or {}
    global_reasons = []
    trace = db.get(RetrievalTrace, package.retrieval_trace_id)
    lineage = ((trace.diagnostics_json or {}).get("reflection_source_retention") or {}) if trace is not None else {}
    if bool(declared) != bool(lineage) or (declared and any(declared.get(key) != value for key, value in lineage.items())):
        global_reasons.append("context_retention_trace_lineage_mismatch")
    if set(declared.get("retained_chunk_ids") or []) != {row.chunk_id for row in rows}:
        global_reasons.append("context_retention_row_scope_mismatch")
    preserved_ids = {item.get("chunk_id") for item in (package.package_json or {}).get("chunks", []) if item.get("role") == "preserved_source"}
    if set(declared.get("preserved_chunk_ids") or []) != preserved_ids:
        global_reasons.append("context_retention_preserved_role_mismatch")
    if rows and declared.get("protocol_version") != RETENTION_PROTOCOL:
        global_reasons.append("context_retention_protocol_mismatch")
    wanted = {str(item.get("chunk_id") or "") for item in citations if isinstance(item, dict)}
    supports = {}
    grouped = {}
    for row in rows:
        if row.chunk_id not in wanted:
            continue
        try:
            _row, origin, item = retention_origin(db, package=package, chunk_id=row.chunk_id, for_update=for_update, row=row)
        except ReflectionContractError:
            supports[row.chunk_id] = {"valid": False, "reason": "retained_source_identity_invalid"}
            continue
        group = grouped.setdefault(origin.id, {"package": origin, "items": [], "rows": {}})
        group["items"].append(item)
        group["rows"][row.chunk_id] = row
    for origin_id in sorted(grouped):
        group = grouped[origin_id]
        origin = group["package"]
        try:
            audit = audit_citation_provenance(db, knowledge_base_id=package.knowledge_base_id, package=origin,
                citations=[source_citation(item, origin) for item in sorted(group["items"], key=lambda item: item["chunk_id"])],
                contexts=context_package_to_contexts(origin), for_update=for_update, _retention_ancestors=(*ancestors, package.id))
        except ReflectionContractError:
            for cid in group["rows"]:
                supports[cid] = {"valid": False, "reason": "retained_source_ancestry_invalid"}
            continue
        for source_audit in audit["audits"]:
            cid = source_audit["chunk_id"]
            row = group["rows"][cid]
            supports[cid] = {"valid": source_audit["valid"], "retention_hash": row.retention_hash,
                "source_context_package_id": origin.id, "source_retrieval_trace_id": origin.retrieval_trace_id,
                "source_provenance_hash": source_audit["provenance_hash"]}
    return supports, global_reasons


def _retention_inputs(
    *, candidate_package: ContextPackage, source_package: ContextPackage,
    preserve_chunk_ids: list[str], token_budget: int,
):
    if candidate_package.knowledge_base_id != source_package.knowledge_base_id:
        raise ReflectionContractError("reflection_retention_cross_knowledge_base")
    if type(token_budget) is not int or token_budget <= 0:
        raise ReflectionContractError("reflection_retention_budget_invalid")
    current_items = deepcopy(candidate_package.package_json["chunks"])
    previous_items = {item["chunk_id"]: item for item in source_package.package_json["chunks"]}
    requested = set(preserve_chunk_ids)
    if len(requested) > MAX_CONTEXT_SOURCES or not requested.issubset(previous_items):
        raise ReflectionContractError("reflection_retention_source_scope_invalid")
    current_by_id = {item["chunk_id"]: item for item in current_items}
    missing = {cid for cid in requested if cid not in current_by_id or not source_covers(current_by_id[cid], previous_items[cid])}
    return current_items, previous_items, requested, current_by_id, missing


def pack_and_retain_reflection_sources(
    db: Session, *, candidate_package: ContextPackage, source_package: ContextPackage,
    preserve_chunk_ids: list[str], token_budget: int,
) -> tuple[ContextPackage, list[dict[str, Any]]]:
    current, previous, requested, current_by_id, missing = _retention_inputs(
        candidate_package=candidate_package, source_package=source_package,
        preserve_chunk_ids=preserve_chunk_ids, token_budget=token_budget)
    cap = min(token_budget, int(candidate_package.token_budget))
    combined_tokens = sum(rough_token_count((previous[item["chunk_id"]] if item["chunk_id"] in missing else item)["content"])
        for item in current) + sum(rough_token_count(previous[cid]["content"]) for cid in missing if cid not in current_by_id)
    if combined_tokens > cap:
        reserve = sum(rough_token_count(previous[cid]["content"]) for cid in requested)
        if reserve >= cap or not candidate_package.hit_chunk_ids_json:
            raise ReflectionContractError("reflection_retention_capacity_exceeded")
        from app.services.reflection_context import restore_reflection_context
        candidate_package, _ = restore_reflection_context(db, source_package=candidate_package,
            target_chunk_ids=list(candidate_package.hit_chunk_ids_json), preserve_chunk_ids=[],
            token_budget=cap, restore_per_chunk_budget=0, reserved_token_budget=reserve)
    return retain_reflection_sources(db, candidate_package=candidate_package, source_package=source_package,
        preserve_chunk_ids=preserve_chunk_ids, token_budget=cap)


def retain_reflection_sources(
    db: Session, *, candidate_package: ContextPackage, source_package: ContextPackage,
    preserve_chunk_ids: list[str], token_budget: int,
) -> tuple[ContextPackage, list[dict[str, Any]]]:
    from app.schemas import ContextSelectionReason
    from app.services.citation_provenance import audit_citation_provenance
    from app.services.context_graph import context_package_to_contexts
    current_items, previous_items, requested, current_by_id, missing = _retention_inputs(
        candidate_package=candidate_package, source_package=source_package,
        preserve_chunk_ids=preserve_chunk_ids, token_budget=token_budget)
    if not missing:
        return candidate_package, context_package_to_contexts(candidate_package)
    old_audit = audit_citation_provenance(db, knowledge_base_id=source_package.knowledge_base_id,
        package=source_package, contexts=context_package_to_contexts(source_package),
        citations=[source_citation(previous_items[cid], source_package) for cid in sorted(requested)])
    if not old_audit["all_valid"]:
        raise ReflectionContractError("reflection_retention_source_scope_changed")
    base_origins = {row.chunk_id for row in db.scalars(select(ContextPackageSourceRetention).where(
        ContextPackageSourceRetention.target_context_package_id == candidate_package.id))}
    origins = {cid: candidate_package for cid in base_origins}
    for cid in sorted(missing):
        replacement = deepcopy(previous_items[cid])
        replacement["role"] = current_by_id[cid]["role"] if cid in current_by_id else "preserved_source"
        if replacement["role"] == "preserved_source":
            replacement["why_selected"] = ContextSelectionReason(
                reason="preserved_prior_bound_source", roles=["preserved_source"]
            ).model_dump(mode="json")
        elif cid in current_by_id:
            replacement["why_selected"] = deepcopy(current_by_id[cid]["why_selected"])
        if cid in current_by_id:
            current_items = [replacement if item["chunk_id"] == cid else item for item in current_items]
        else:
            current_items.append(replacement)
        origins[cid] = source_package
    cap = min(token_budget, int(candidate_package.token_budget))
    tokens = sum(rough_token_count(item["content"]) for item in current_items)
    if len(current_items) > MAX_CONTEXT_SOURCES or tokens > cap:
        raise ReflectionContractError("reflection_retention_capacity_exceeded")
    base_trace = db.get(RetrievalTrace, candidate_package.retrieval_trace_id)
    if base_trace is None or base_trace.knowledge_base_id != candidate_package.knowledge_base_id:
        raise ReflectionContractError("reflection_retention_base_trace_missing")
    trace_fields = {column.name: deepcopy(getattr(base_trace, column.name))
                    for column in RetrievalTrace.__table__.columns if column.name not in {"id", "created_at"}}
    trace = RetrievalTrace(id=generate_uuid(), **trace_fields)
    lineage = {"protocol_version": RETENTION_PROTOCOL, "base_context_package_id": candidate_package.id,
               "base_retrieval_trace_id": base_trace.id, "source_context_package_id": source_package.id,
               "retrieval_executed": False, "gray_zone_model_call_count": 0}
    trace.diagnostics_json = {**trace.diagnostics_json, "reflection_source_retention": lineage}
    db.add(trace)
    db.flush()
    fields = {column.name: deepcopy(getattr(candidate_package, column.name))
              for column in ContextPackage.__table__.columns if column.name not in {"id", "created_at"}}
    fields.update(retrieval_trace_id=trace.id, token_count=tokens, token_budget=cap)
    package = ContextPackage(id=generate_uuid(), **fields)
    for item in current_items:
        item["context_package_id"] = package.id
        item["source_span"] = {**item["source_span"], "context_package_id": package.id, "retrieval_trace_id": trace.id}
        item["structure_closure"] = {**item["structure_closure"],
            "bridge_chunk_ids": list(package.bridge_chunk_ids_json or [])}
    package.package_json = {"chunks": current_items}
    package.why_selected_json = {item["chunk_id"]: item["why_selected"] for item in current_items}
    package.dedupe_keys_json = [item["dedupe_key"] for item in current_items]
    package.citation_spans_json = [{**item["source_span"], **{key: item.get(key) for key in (
        "document_id", "document_title", "source_path", "logical_source_path", "section_path", "structure_path", "structure_node_ids", "structure_closure"
    )}} for item in current_items]
    token_audit = {**package.diagnostics_json["token_budget_audit"], "token_budget": cap, "token_count": tokens, "within_budget": True,
        "clipped_chunk_ids": [item["chunk_id"] for item in current_items if item.get("content_clipped")],
        "skipped_chunk_ids": [cid for cid in package.diagnostics_json["token_budget_audit"].get("skipped_chunk_ids", [])
                              if cid not in {item["chunk_id"] for item in current_items}]}
    package.diagnostics_json = {**package.diagnostics_json, "token_budget_audit": token_audit, "dedupe_keys": package.dedupe_keys_json,
        "snapshot_integrity": {**package.diagnostics_json["snapshot_integrity"],
            "verified_document_version_count": len({item["document_version_id"] for item in current_items})},
        "source_retention": {**lineage, "retained_chunk_ids": sorted(origins),
                             "preserved_chunk_ids": [item["chunk_id"] for item in current_items if item["role"] == "preserved_source"]}}
    db.add(package)
    db.flush()
    from app.services.reflection_expansion import copy_source_expansions
    copy_source_expansions(db, source=candidate_package, target=package)
    for cid, origin in origins.items():
        original = next(item for item in origin.package_json["chunks"] if item["chunk_id"] == cid)
        row = ContextPackageSourceRetention(knowledge_base_id=package.knowledge_base_id,
            target_context_package_id=package.id, source_context_package_id=origin.id,
            source_retrieval_trace_id=origin.retrieval_trace_id, chunk_id=cid,
            protocol_version=RETENTION_PROTOCOL, source_item_hash=reflection_hash(original))
        row.retention_hash = reflection_hash(retention_identity(row))
        db.add(row)
    original_steps = list(db.scalars(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == base_trace.id).order_by(GraphRetrievalStep.step_index)))
    step_ids = {step.id: generate_uuid() for step in original_steps}
    for original in original_steps:
        values = {column.name: deepcopy(getattr(original, column.name)) for column in GraphRetrievalStep.__table__.columns
                  if column.name not in {"id", "created_at", "retrieval_trace_id"}}
        if original.layer == "structure" and original.action_type == "restore_context_package":
            values["output_json"] = {**values["output_json"], "context_package_id": package.id,
                "citation_spans": package.citation_spans_json, "why_selected": package.why_selected_json,
                "dedupe_keys": package.dedupe_keys_json, "token_count": tokens, "token_budget": cap}
            values["diagnostics_json"] = {**values["diagnostics_json"], **package.diagnostics_json, "token_count": tokens, "token_budget": cap,
                                          "scores_json_retired_as_primary_audit": True}
            values["diagnostics_json"]["gray_zone_decision_refs"] = [
                {**ref, "source_step_id": step_ids[ref["source_step_id"]]}
                for ref in values["diagnostics_json"].get("gray_zone_decision_refs", [])]
        db.add(GraphRetrievalStep(id=step_ids[original.id], retrieval_trace_id=trace.id, **values))
    db.flush()
    contexts = context_package_to_contexts(package)
    final = audit_citation_provenance(db, knowledge_base_id=package.knowledge_base_id, package=package, contexts=contexts,
        citations=[source_citation(item, package) for item in current_items], for_update=True)
    if not final["all_valid"]:
        raise ReflectionContractError("reflection_retention_provenance_failed")
    return package, contexts
