"""Replanning cannot silently discard already bound, replayable raw sources."""
from copy import deepcopy

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified

from app.models import ContextPackage, ContextPackageSourceRetention, RetrievalTrace
from app.schemas import ContextPackageChunk, ContextPackageDiagnostics
from app.services.agent_reflection import ReflectionContractError, reflection_hash
from app.services.citation_provenance import audit_citation_provenance
from app.services.context_graph import context_package_to_contexts
from app.services.reflection_sources import retain_reflection_sources, retention_identity, source_citation
from test_citation_provenance import _build_package


async def source_pair(db, kb, tmp_path):
    old = (await _build_package(db, kb, tmp_path))[4]
    new = (await _build_package(db, kb, tmp_path, source_name="unit-test-second-source.md",
        source_text="Conditional independence assumptions determine which conditional factors a model requires."))[4]
    return old, new


def audit_package(db, package):
    return audit_citation_provenance(db, knowledge_base_id=package.knowledge_base_id, package=package,
        contexts=context_package_to_contexts(package), citations=[source_citation(item, package) for item in package.package_json["chunks"]])


@pytest.mark.asyncio
async def test_replan_retains_missing_source_without_new_hit_or_path(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    old, new = await source_pair(db_session, sample_knowledge_base, tmp_path)
    snapshots = {pkg.id: deepcopy(pkg.package_json) for pkg in (old, new)}
    trace_facts = {pkg.retrieval_trace_id: deepcopy(db_session.get(RetrievalTrace, pkg.retrieval_trace_id).path_labels_json) for pkg in (old, new)}
    retained, contexts = retain_reflection_sources(db_session, candidate_package=new, source_package=old,
        preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=new.token_budget)
    assert retained.id not in {old.id, new.id}
    assert retained.hit_chunk_ids_json == new.hit_chunk_ids_json
    assert retained.restored_chunk_ids_json == new.restored_chunk_ids_json
    assert retained.graph_path_ids_json == new.graph_path_ids_json
    assert len(contexts) == 2
    item = next(row for row in retained.package_json["chunks"] if row["chunk_id"] == old.hit_chunk_ids_json[0])
    assert item["role"] == "preserved_source" and item["content"] == old.package_json["chunks"][0]["content"]
    assert item["why_selected"]["reached_by_paths"] == []
    audit = audit_package(db_session, retained)
    assert audit["all_valid"], [row["reasons"] for row in audit["audits"]]
    for raw in retained.package_json["chunks"]:
        ContextPackageChunk.model_validate(raw)
    assert ContextPackageDiagnostics.model_validate({key: value for key, value in retained.diagnostics_json.items()
        if key in ContextPackageDiagnostics.model_fields}).source_retention.retained_chunk_ids == old.hit_chunk_ids_json
    assert {pkg.id: pkg.package_json for pkg in (old, new)} == snapshots
    assert {tid: db_session.get(RetrievalTrace, tid).path_labels_json for tid in trace_facts} == trace_facts
    db_session.commit()
    db_session.expire_all()
    assert audit_package(db_session, db_session.get(ContextPackage, retained.id))["all_valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("target_has_bridge", [False, True])
async def test_retention_rebinds_package_bridge_scope_without_changing_raw_identity(
    db_session, sample_knowledge_base, tmp_path, fake_model_stack, target_has_bridge,
):
    from app.models import ChunkRelationEdge, ChunkRelationGraphState, GraphRetrievalStep
    from app.services.context_graph import build_context_package
    from app.services.reflection_sources import physical_source_identity, retention_origin

    old, candidate = await source_pair(db_session, sample_knowledge_base, tmp_path)
    bridge_source = (await _build_package(db_session, sample_knowledge_base, tmp_path,
        source_name="unit-test-original-bridge.md", source_text="A synthetic bridge explains prior probability."))[4]
    state = ChunkRelationGraphState(knowledge_base_id=sample_knowledge_base.id, chunk_version=1,
        scope_hash="a" * 64, state_hash="b" * 64, embedding_text_version="unit-test-vector-v1")
    db_session.add(state)
    db_session.flush()

    def add_bridge(source, bridge):
        edge = ChunkRelationEdge(graph_state_id=state.id, knowledge_base_id=sample_knowledge_base.id,
            source_chunk_id=source.hit_chunk_ids_json[0], target_chunk_id=bridge.hit_chunk_ids_json[0],
            edge_type="dense_cross_document_bridge", is_bridge=True, is_cross_document=True)
        db_session.add(edge)
        db_session.flush()
        base_trace = db_session.get(RetrievalTrace, source.retrieval_trace_id)
        trace = RetrievalTrace(**{column.name: deepcopy(getattr(base_trace, column.name))
            for column in RetrievalTrace.__table__.columns if column.name not in {"id", "created_at"}})
        db_session.add(trace)
        db_session.flush()
        chunk_step = db_session.scalar(select(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == base_trace.id, GraphRetrievalStep.layer == "chunk"))
        db_session.add(GraphRetrievalStep(retrieval_trace_id=trace.id,
            **{column.name: deepcopy(getattr(chunk_step, column.name))
               for column in GraphRetrievalStep.__table__.columns if column.name not in {"id", "created_at", "retrieval_trace_id"}}))
        db_session.flush()
        result = build_context_package(db_session, knowledge_base_id=source.knowledge_base_id,
            query=source.query, trace=trace, results=[{"chunk_id": source.hit_chunk_ids_json[0],
                "metadata": {"traversal": deepcopy(base_trace.path_labels_json[0])}}],
            token_budget=source.token_budget, restore_per_chunk_budget=1)
        assert result.bridge_chunk_ids_json == bridge.hit_chunk_ids_json
        assert audit_package(db_session, result)["all_valid"]
        return result, edge

    old, original_edge = add_bridge(old, bridge_source)
    if target_has_bridge:
        target_bridge = (await _build_package(db_session, sample_knowledge_base, tmp_path,
            source_name="unit-test-target-bridge.md", source_text="A separate bridge explains conditional factors."))[4]
        candidate, _ = add_bridge(candidate, target_bridge)
    assert old.bridge_chunk_ids_json != candidate.bridge_chunk_ids_json
    original_snapshots = {pkg.id: deepcopy(pkg.package_json) for pkg in (old, candidate)}
    original_items = {item["chunk_id"]: item for item in old.package_json["chunks"]}
    package, _ = retain_reflection_sources(db_session, candidate_package=candidate, source_package=old,
        preserve_chunk_ids=list(original_items), token_budget=candidate.token_budget)
    assert package.bridge_chunk_ids_json == candidate.bridge_chunk_ids_json
    for item in package.package_json["chunks"]:
        assert item["structure_closure"]["bridge_chunk_ids"] == candidate.bridge_chunk_ids_json
        span = next(span for span in package.citation_spans_json if span["chunk_id"] == item["chunk_id"])
        assert span["structure_closure"] == item["structure_closure"]
        if item["chunk_id"] in original_items:
            original = original_items[item["chunk_id"]]
            assert physical_source_identity(item) == physical_source_identity(original)
            row, _, _ = retention_origin(db_session, package=package, chunk_id=item["chunk_id"])
            assert row.source_item_hash == reflection_hash(original)
    assert {pkg.id: pkg.package_json for pkg in (old, candidate)} == original_snapshots
    db_session.commit()
    db_session.expire_all()
    assert audit_package(db_session, package)["all_valid"]

    retained_item = next(item for item in package.package_json["chunks"] if item["chunk_id"] == old.hit_chunk_ids_json[0])
    target_closure = deepcopy(retained_item["structure_closure"])
    retained_item["structure_closure"]["bridge_chunk_ids"] = old.bridge_chunk_ids_json
    flag_modified(package, "package_json")
    audit = audit_package(db_session, package)
    assert "context_package_bridge_closure_mismatch" in {reason for item in audit["audits"] for reason in item["reasons"]}
    retained_item["structure_closure"] = target_closure
    retained_item["structure_closure"]["previous_chunk_id"] = "unit-test-forged-neighbor"
    with pytest.raises(ReflectionContractError, match="retained_source_identity_mismatch"):
        retention_origin(db_session, package=package, chunk_id=retained_item["chunk_id"])
    retained_item["structure_closure"] = deepcopy(original_items[retained_item["chunk_id"]]["structure_closure"])
    retained_item["structure_closure"]["bridge_chunk_ids"] = candidate.bridge_chunk_ids_json
    flag_modified(package, "package_json")
    db_session.delete(original_edge)
    db_session.flush()
    assert not audit_package(db_session, package)["all_valid"], "Original bridge support must still be replayed."


@pytest.mark.asyncio
async def test_retention_capacity_failure_does_not_write_partial_snapshot(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    old, new = await source_pair(db_session, sample_knowledge_base, tmp_path)
    before = [db_session.scalar(select(func.count()).select_from(model)) for model in (ContextPackage, RetrievalTrace, ContextPackageSourceRetention)]
    with pytest.raises(ReflectionContractError, match="capacity_exceeded"):
        retain_reflection_sources(db_session, candidate_package=new, source_package=old,
            preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=1)
    assert [db_session.scalar(select(func.count()).select_from(model)) for model in (ContextPackage, RetrievalTrace, ContextPackageSourceRetention)] == before
    assert db_session.is_active


@pytest.mark.asyncio
async def test_packing_reservation_cannot_exceed_the_existing_candidate_cap(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.chunking import rough_token_count
    from app.services.reflection_sources import pack_and_retain_reflection_sources
    old, candidate = await source_pair(db_session, sample_knowledge_base, tmp_path)
    reserve = sum(rough_token_count(item["content"]) for item in old.package_json["chunks"])
    candidate.token_budget = reserve
    before = [db_session.scalar(select(func.count()).select_from(model))
        for model in (ContextPackage, RetrievalTrace, ContextPackageSourceRetention)]
    with pytest.raises(ReflectionContractError, match="capacity_exceeded"):
        pack_and_retain_reflection_sources(db_session, candidate_package=candidate, source_package=old,
            preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=2400)
    assert [db_session.scalar(select(func.count()).select_from(model))
        for model in (ContextPackage, RetrievalTrace, ContextPackageSourceRetention)] == before


@pytest.mark.asyncio
async def test_overlapping_bound_sources_do_not_reserve_duplicate_space(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.reflection_sources import pack_and_retain_reflection_sources
    source = (await _build_package(db_session, sample_knowledge_base, tmp_path))[4]
    original = deepcopy(source.package_json)
    result, _ = pack_and_retain_reflection_sources(db_session, candidate_package=source, source_package=source,
        preserve_chunk_ids=source.hit_chunk_ids_json, token_budget=source.token_budget)
    assert result is source and source.package_json == original


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["text", "row", "origin", "declaration"])
async def test_retained_sources_fail_closed_on_tamper(db_session, sample_knowledge_base, tmp_path, fake_model_stack, tamper):
    old, new = await source_pair(db_session, sample_knowledge_base, tmp_path)
    package, _ = retain_reflection_sources(db_session, candidate_package=new, source_package=old,
        preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=new.token_budget)
    if tamper == "text":
        package.package_json["chunks"][-1]["content"] = package.package_json["chunks"][-1]["content"].replace("Bayesian", "Counterfeit")
        flag_modified(package, "package_json")
    elif tamper == "row":
        row = db_session.scalar(select(ContextPackageSourceRetention).where(ContextPackageSourceRetention.target_context_package_id == package.id))
        row.retention_hash = "0" * 64
    elif tamper == "origin":
        old.package_json["chunks"][0]["why_selected"]["reason"] = "Changed source audit."
        flag_modified(old, "package_json")
    else:
        package.diagnostics_json["source_retention"]["retained_chunk_ids"] = []
        flag_modified(package, "diagnostics_json")
    db_session.flush()
    assert not audit_package(db_session, package)["all_valid"]


@pytest.mark.asyncio
async def test_retained_source_cycle_is_rejected_before_recursive_overflow(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.reflection_sources import RETENTION_PROTOCOL
    old, new = await source_pair(db_session, sample_knowledge_base, tmp_path)
    package, _ = retain_reflection_sources(db_session, candidate_package=new, source_package=old,
        preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=new.token_budget)
    cid = old.hit_chunk_ids_json[0]
    item = next(item for item in package.package_json["chunks"] if item["chunk_id"] == cid)
    row = ContextPackageSourceRetention(knowledge_base_id=old.knowledge_base_id, target_context_package_id=old.id,
        source_context_package_id=package.id, source_retrieval_trace_id=package.retrieval_trace_id, chunk_id=cid,
        protocol_version=RETENTION_PROTOCOL, source_item_hash=reflection_hash(item))
    row.retention_hash = reflection_hash(retention_identity(row))
    db_session.add(row)
    lineage = {"protocol_version": RETENTION_PROTOCOL, "base_context_package_id": new.id,
        "base_retrieval_trace_id": new.retrieval_trace_id, "source_context_package_id": package.id,
        "retrieval_executed": False, "gray_zone_model_call_count": 0}
    old.diagnostics_json = {**old.diagnostics_json, "source_retention": {**lineage, "retained_chunk_ids": [cid], "preserved_chunk_ids": []}}
    trace = db_session.get(RetrievalTrace, old.retrieval_trace_id)
    trace.diagnostics_json = {**trace.diagnostics_json, "reflection_source_retention": lineage}
    db_session.flush()
    assert not audit_package(db_session, package)["all_valid"]


@pytest.mark.asyncio
async def test_retained_source_restores_through_original_authority(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.reflection_context import restore_reflection_context
    old, new = await source_pair(db_session, sample_knowledge_base, tmp_path)
    package, _ = retain_reflection_sources(db_session, candidate_package=new, source_package=old,
        preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=new.token_budget)
    result, _ = restore_reflection_context(db_session, source_package=package,
        target_chunk_ids=old.hit_chunk_ids_json, preserve_chunk_ids=old.hit_chunk_ids_json,
        token_budget=package.token_budget, restore_per_chunk_budget=1)
    assert result.hit_chunk_ids_json == new.hit_chunk_ids_json
    assert audit_package(db_session, result)["all_valid"]


@pytest.mark.asyncio
async def test_shorter_same_chunk_prefix_cannot_replace_bound_span(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.models import GraphRetrievalStep
    from app.services.context_graph import build_context_package
    old = (await _build_package(db_session, sample_knowledge_base, tmp_path))[4]
    source_trace = db_session.get(RetrievalTrace, old.retrieval_trace_id)
    trace = RetrievalTrace(**{column.name: deepcopy(getattr(source_trace, column.name))
        for column in RetrievalTrace.__table__.columns if column.name not in {"id", "created_at"}})
    db_session.add(trace)
    db_session.flush()
    original_step = db_session.scalar(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == source_trace.id, GraphRetrievalStep.layer == "chunk"))
    step = GraphRetrievalStep(retrieval_trace_id=trace.id, **{column.name: deepcopy(getattr(original_step, column.name))
        for column in GraphRetrievalStep.__table__.columns if column.name not in {"id", "created_at", "retrieval_trace_id"}})
    db_session.add(step)
    db_session.flush()
    candidate = build_context_package(db_session, knowledge_base_id=old.knowledge_base_id, query=old.query, trace=trace,
        results=[{"chunk_id": old.hit_chunk_ids_json[0], "metadata": {"traversal": deepcopy(source_trace.path_labels_json[0])}}],
        token_budget=5, restore_per_chunk_budget=0)
    assert candidate.package_json["chunks"][0]["content_clipped"] is True
    assert audit_package(db_session, candidate)["all_valid"]
    with pytest.raises(ReflectionContractError, match="capacity_exceeded"):
        retain_reflection_sources(db_session, candidate_package=candidate, source_package=old,
            preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=5)

