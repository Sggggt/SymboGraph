import copy

import pytest
from sqlalchemy import func, select

from app.models import AnswerSession, AnswerSourceBinding, CitationVerification, GraphRetrievalStep, RetrievalTrace
from app.reflection_contracts import AnswerDraft
from app.services.agent_reflection import ReflectionContractError, render_answer_units
from app.services.answer_sources import (
    audit_answer_sources, build_answer_evidence_manifest, persist_answer_source_bindings, source_binding_citations,
)
from test_citation_provenance import _build_package


def answer_draft(text):
    return AnswerDraft.model_validate({
        "protocol_version": "structured_answer_self_assessment_v1",
        "answer_units": [
            {"kind": "framing", "text": "The source explains:", "source_handles": []},
            {"kind": "factual", "text": text, "source_handles": ["src_1"]},
        ],
        "self_assessment": {
            "question_relevance": 0.9, "context_relevance": 0.9, "coverage": 0.9,
            "needs_reflection": False, "issue_types": [], "summary": "Source directly addresses the question.",
        },
    })


def new_answer(db, kb, package, draft):
    answer, _units = render_answer_units(draft)
    row = AnswerSession(
        knowledge_base_id=kb.id, question=package.query, answer=answer,
        context_package_id=package.id, retrieval_trace_id=package.retrieval_trace_id,
    )
    db.add(row)
    db.flush()
    return row


@pytest.mark.asyncio
async def test_path_metrics_use_replayed_non_hit_chunk_labels(db_session, populated_context_graph):
    from app.schemas import SearchFilters
    from app.services.answer_sources import answer_source_path_metrics
    from app.services.context_graph import build_context_package, context_package_to_contexts, layered_search

    kb = populated_context_graph["knowledge_base"]
    result = await layered_search(db_session, kb.id, "Bayes theorem prior posterior", SearchFilters(), 1,
        allow_cache_read=False)
    package = build_context_package(db_session, knowledge_base_id=kb.id, query=result.trace.query,
        trace=result.trace, results=result.results, token_budget=2400, restore_per_chunk_budget=8)
    original_labels = copy.deepcopy(result.trace.path_labels_json)
    labels = [label for label in original_labels if label.get("layer") == "chunk"]
    known = {label["node_id"] for label in labels}
    item = next(item for item in package.package_json["chunks"]
        if item["role"] != "hit" and item["chunk_id"] in known and not any(
            path.get("layer") == "chunk" and path.get("node_id") == item["chunk_id"]
            for path in item["why_selected"].get("reached_by_paths", [])))
    contexts = context_package_to_contexts(package)
    evidence = build_answer_evidence_manifest(package, contexts)
    handle = next(key for key, source in evidence.by_handle().items() if source["chunk_id"] == item["chunk_id"])
    payload = answer_draft(item["content"]).model_dump(mode="json")
    payload["answer_units"][1]["source_handles"] = [handle]
    draft = AnswerDraft.model_validate(payload)
    _, audit = audit_answer_sources(db_session, knowledge_base_id=kb.id, package=package, contexts=contexts,
        draft=draft, evidence=evidence, unit_limit=12)
    assert audit["all_valid"], [entry["reasons"] for entry in audit["audits"]]
    metrics = answer_source_path_metrics(db_session, package=package, draft=draft, evidence=evidence, source_audit=audit)
    expected_distance = min(max(0, label["distance_so_far"] - label["reward_so_far"])
        for label in labels if label["node_id"] == item["chunk_id"])
    assert metrics.coverage == 1.0
    assert metrics.sources[0].effective_distance == pytest.approx(expected_distance, abs=1e-6)
    assert metrics.path_score == pytest.approx(1 / (1 + expected_distance), abs=1e-6)
    assert result.trace.path_labels_json == original_labels


@pytest.mark.parametrize("missing", ["own_label", "chunk_layer", "distance", "reward"])
def test_path_reader_keeps_unscored_sources_unknown(missing):
    from types import SimpleNamespace
    from app.services.answer_sources import answer_source_path_metrics

    label = {"layer": "chunk", "node_id": "unit-test-source", "path": ["unit-test-source"],
        "distance_so_far": 2.0, "reward_so_far": 0.5}
    if missing == "own_label":
        label.update(node_id="unit-test-other", path=["unit-test-source", "unit-test-other"])
    elif missing == "chunk_layer":
        label["layer"] = "mid"
    elif missing == "distance":
        label.pop("distance_so_far")
    else:
        label.pop("reward_so_far")
    trace = SimpleNamespace(id="unit-test-trace", knowledge_base_id="unit-test-kb",
        path_labels_json=[label], result_chunk_ids_json=[])
    package = SimpleNamespace(id="unit-test-package", retrieval_trace_id=trace.id,
        knowledge_base_id=trace.knowledge_base_id, restored_chunk_ids_json=[])
    db = SimpleNamespace(scalar=lambda _query: None, get=lambda _model, _key: trace)
    evidence = SimpleNamespace(by_handle=lambda: {"src_1": {"chunk_id": "unit-test-source"}})
    metrics = answer_source_path_metrics(db, package=package, draft=answer_draft("Synthetic answer"),
        evidence=evidence, source_audit={"all_valid": True})
    assert metrics.coverage == 0.0 and metrics.path_score is None
    assert metrics.sources[0].effective_distance is None


def test_invalid_source_audit_prevents_any_path_read():
    from app.services.answer_sources import answer_source_path_metrics
    metrics = answer_source_path_metrics(None, package=None, draft=answer_draft("Synthetic answer"),
        evidence=None, source_audit={"all_valid": False})
    assert metrics.coverage == 0.0 and metrics.path_score is None


@pytest.mark.parametrize("protocol", ["", "unsupported", False, []])
def test_historical_path_reader_rejects_unknown_protocol_before_source_access(protocol):
    from app.services.answer_sources import answer_source_path_metrics
    with pytest.raises(ReflectionContractError, match="reflection_path_protocol_unknown"):
        answer_source_path_metrics(None, package=None, draft=answer_draft("Synthetic answer"),
            evidence=None, source_audit={"all_valid": False}, _replay_protocol=protocol)


@pytest.mark.asyncio
async def test_new_source_binding_replays_provenance_without_semantic_judge(
    db_session, sample_knowledge_base, tmp_path, fake_model_stack, monkeypatch,
):
    from app.services import agent_graph
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("The old semantic citation judge must not run")
    monkeypatch.setattr(agent_graph, "verify_answer_against_context", forbidden)
    kb, _doc, _version, _chunks, package, contexts, text, _legacy_citations = await _build_package(db_session, sample_knowledge_base, tmp_path)
    draft = answer_draft(text)
    manifest = build_answer_evidence_manifest(package, contexts)
    answer = new_answer(db_session, kb, package, draft)
    candidates, audit = audit_answer_sources(
        db_session, knowledge_base_id=kb.id, package=package, contexts=contexts,
        draft=draft, evidence=manifest, unit_limit=12,
    )
    assert audit["all_valid"], [item["reasons"] for item in audit["audits"]]
    assert len(candidates) == 1
    rows = persist_answer_source_bindings(
        db_session, answer_session=answer, package=package, contexts=contexts,
        draft=draft, evidence=manifest, unit_limit=12, reflection_audit_hash="a" * 64,
    )
    assert len(rows) == 1 and rows[0].unit_index == 1
    row = rows[0]
    assert row.unit_text == answer.answer[row.answer_char_start:row.answer_char_end]
    assert row.diagnostics_json["status"] == "source_bound"
    assert row.diagnostics_json["semantic_entailment_claimed"] is False
    assert row.diagnostics_json["transactional_replay"] is True
    assert db_session.scalar(select(func.count()).select_from(CitationVerification)) == 0
    public = source_binding_citations(answer_session=answer, package=package, rows=rows, reflection_audit_hash="a" * 64)
    assert len(public) == 1
    assert public[0]["contract_version"] == "citation_public_v2"
    assert public[0]["verification"] is None
    assert public[0]["source_binding"]["semantic_entailment_claimed"] is False
    assert public[0]["source_span"]["source_binding_id"] == row.id
    row.unit_text = "Tampered persisted unit."
    with pytest.raises(ReflectionContractError, match="identity_mismatch"):
        source_binding_citations(answer_session=answer, package=package, rows=rows, reflection_audit_hash="a" * 64)


@pytest.mark.asyncio
async def test_source_bindings_and_answer_rollback_together(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    kb, _doc, _version, _chunks, package, contexts, text, _legacy_citations = await _build_package(db_session, sample_knowledge_base, tmp_path)
    db_session.commit()
    draft = answer_draft(text)
    answer = new_answer(db_session, kb, package, draft)
    persist_answer_source_bindings(
        db_session, answer_session=answer, package=package, contexts=contexts,
        draft=draft, evidence=build_answer_evidence_manifest(package, contexts),
        unit_limit=12, reflection_audit_hash="a" * 64,
    )
    db_session.rollback()
    assert db_session.scalar(select(func.count()).select_from(AnswerSourceBinding)) == 0
    assert db_session.scalar(select(func.count()).select_from(AnswerSession)) == 0


@pytest.mark.asyncio
async def test_manifest_rejects_changed_text_scope_and_mutation(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    _kb, _doc, _version, _chunks, package, contexts, _text, _legacy_citations = await _build_package(db_session, sample_knowledge_base, tmp_path)
    changed = copy.deepcopy(contexts)
    changed[0]["content"] += " A fact not in the frozen package."
    with pytest.raises(ReflectionContractError, match="context_text_mismatch"):
        build_answer_evidence_manifest(package, changed)
    with pytest.raises(ReflectionContractError, match="scope_mismatch"):
        build_answer_evidence_manifest(package, contexts + contexts)
    labels = build_answer_evidence_manifest(package, contexts)
    model_source = labels.model_sources()[0]
    assert set(model_source) == {"source_handle", "text", "source_label"}
    assert set(model_source["source_label"]) == {"document_title", "section", "pages"}
    assert model_source["text"] == contexts[0]["content"]
    labels.sources[0]["package_item"]["document_title"] = "A different location label."
    with pytest.raises(ReflectionContractError, match="manifest_mutated"):
        labels.model_sources()
    manifest = build_answer_evidence_manifest(package, contexts)
    manifest.sources[0]["text"] = "Modified after freezing."
    with pytest.raises(ReflectionContractError, match="manifest_mutated"):
        manifest.model_sources()


@pytest.mark.asyncio
async def test_source_binding_rejects_changed_answer_before_writing(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    kb, _doc, _version, _chunks, package, contexts, text, _legacy_citations = await _build_package(db_session, sample_knowledge_base, tmp_path)
    draft = answer_draft(text)
    answer = new_answer(db_session, kb, package, draft)
    answer.answer += " Added after review."
    with pytest.raises(ReflectionContractError, match="session_binding_mismatch"):
        persist_answer_source_bindings(
            db_session, answer_session=answer, package=package, contexts=contexts,
            draft=draft, evidence=build_answer_evidence_manifest(package, contexts),
            unit_limit=12, reflection_audit_hash="a" * 64,
        )
    assert db_session.scalar(select(func.count()).select_from(AnswerSourceBinding)) == 0


@pytest.mark.asyncio
async def test_reflection_restoration_preserves_original_trace_and_source_addresses(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.reflection_context import restore_reflection_context
    from app.services.chunking import stable_hash
    from app.services.context_graph import gray_zone_runtime_settings_hash
    kb, _doc, _version, chunks, package, contexts, text, _legacy_citations = await _build_package(db_session, sample_knowledge_base, tmp_path)
    original_step = db_session.scalar(select(GraphRetrievalStep).where(
        GraphRetrievalStep.retrieval_trace_id == package.retrieval_trace_id,
        GraphRetrievalStep.layer == "structure",
    ))
    original_output = copy.deepcopy(original_step.output_json)
    source_trace = db_session.get(RetrievalTrace, package.retrieval_trace_id)
    source_trace.agent_operating_envelope_hash = stable_hash(source_trace.diagnostics_json["agent_operating_envelope"])
    source_trace.diagnostics_json = {
        **source_trace.diagnostics_json,
        "gray_zone_runtime_settings_hash": gray_zone_runtime_settings_hash(source_trace.diagnostics_json["agent_operating_envelope"]),
    }
    restored, restored_contexts = restore_reflection_context(
        db_session, source_package=package, target_chunk_ids=[chunks[0].id], preserve_chunk_ids=[chunks[0].id],
        token_budget=package.token_budget, restore_per_chunk_budget=2,
    )
    assert restored.retrieval_trace_id != package.retrieval_trace_id
    assert original_step.output_json == original_output
    assert original_step.output_json["context_package_id"] == package.id
    for current, current_contexts in ((package, contexts), (restored, restored_contexts)):
        _candidates, audit = audit_answer_sources(
            db_session, knowledge_base_id=kb.id, package=current, contexts=current_contexts,
            draft=answer_draft(text), evidence=build_answer_evidence_manifest(current, current_contexts), unit_limit=12,
        )
        assert audit["all_valid"], [item["reasons"] for item in audit["audits"]]
