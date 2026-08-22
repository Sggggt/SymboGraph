from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sqlalchemy.orm.attributes import flag_modified

from sqlalchemy import select

from app.models import (
    AnswerSession,
    CitationVerification,
    Document,
    DocumentVersion,
    GraphRetrievalStep,
    QASession,
    RetrievalTrace,
    RewardEvent,
)
from app.services import agent_graph
from app.services.citation_provenance import (
    CITATION_PROVENANCE_PROTOCOL_VERSION,
    audit_citation_provenance,
    replay_citation_provenance_for_persistence,
)
from app.services.context_graph import (
    RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION,
    build_context_package,
    context_package_to_contexts,
    runtime_settings_state_hash,
    write_chunks_and_structure,
)
from app.services.parsers import ParsedSection
from app.services.storage import snapshot_source_file


def _exact_answer_model_audit(question: str) -> dict[str, object]:
    from app.services.embeddings import ChatProvider

    prompt_bundle = ChatProvider()._answer_prompt_bundle(
        question,
        context_quality="normal",
    )
    metadata = dict(prompt_bundle["protocol_metadata"])
    return {
        "provider": "unit_chat",
        "model": "unit-chat",
        "prompt_protocol_version": metadata["protocol_version"],
        "prompt_protocol_hash": metadata["prompt_protocol_hash"],
        "grounding_envelope_protocol_version": metadata["protocol_version"],
        "grounding_envelope_hash": metadata["envelope_hash"],
        "profile_hash": metadata["profile_hash"],
    }


async def _build_package(db_session, sample_knowledge_base, tmp_path):
    logical_source = (
        Path(sample_knowledge_base.source_root)
        / "source_slots"
        / "citation-provenance.md"
    )
    logical_source.parent.mkdir(parents=True, exist_ok=True)
    logical_source.write_text(
        "# Factorization\n"
        "Bayesian networks factorize a joint distribution into local conditional probabilities.\n",
        encoding="utf-8",
    )
    frozen_snapshot = snapshot_source_file(
        logical_source,
        knowledge_base_source_root=sample_knowledge_base.source_root,
    )
    snapshot_path = frozen_snapshot.canonical_path
    checksum = frozen_snapshot.checksum
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Citation provenance",
        source_path=str(logical_source),
        source_type="markdown",
        checksum=checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=checksum,
        storage_path=str(snapshot_path),
        parse_protocol_version="parser_native_layout_v2",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunks = write_chunks_and_structure(
        db_session,
        knowledge_base=sample_knowledge_base,
        document=document,
        version=version,
        sections=[
            ParsedSection(
                title="Factorization",
                text=(
                    "Bayesian networks factorize a joint distribution into "
                    "local conditional probabilities."
                ),
                page_number=1,
                section="Factorization",
            )
        ],
        chunk_version=1,
        chunk_size=64,
        chunk_overlap=4,
    )
    sample_knowledge_base.current_chunk_version = 1
    path_labels = [
        {
            "layer": "chunk",
            "node_id": chunks[0].id,
            "chunk_id": chunks[0].id,
            "path": [chunks[0].id],
            "path_edge_ids": [],
            "covered_facets": ["factorization"],
            "evidence_roles": ["definition"],
        }
    ]
    trace = RetrievalTrace(
        knowledge_base_id=sample_knowledge_base.id,
        query="How do Bayesian networks factorize a joint distribution?",
        retrieval_mode="layered_context_graph",
        runtime_settings_hash=runtime_settings_state_hash(),
        result_chunk_ids_json=[chunks[0].id],
        path_labels_json=path_labels,
        convergence_json={
            "gray_zone_decision_count": 0,
            "gray_zone_rule_evaluation_count": 0,
            "red_zone_pruned_count": 0,
            "hard_stop_pruned_count": 0,
            "gray_zone_model_call_count": 0,
        },
        diagnostics_json={"agent_operating_envelope": {}},
    )
    db_session.add(trace)
    db_session.flush()
    package = build_context_package(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        query="How do Bayesian networks factorize a joint distribution?",
        trace=trace,
        results=[
            {
                "chunk_id": chunks[0].id,
                "metadata": {
                    "traversal": {
                        "path": [chunks[0].id],
                        "path_edge_ids": [],
                        "covered_facets": ["factorization"],
                        "evidence_roles": ["definition"],
                        "why_selected": "direct_test_hit",
                    }
                },
            }
        ],
    )
    structure_step = db_session.scalar(
        select(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == trace.id,
            GraphRetrievalStep.layer == "structure",
        )
    )
    structure_step.step_index = 1
    db_session.add(
        GraphRetrievalStep(
            retrieval_trace_id=trace.id,
            knowledge_base_id=sample_knowledge_base.id,
            step_index=0,
            layer="chunk",
            action="walk_chunk_graph",
            action_type="walk_chunk_graph",
            selected_topk_ids_json=[chunks[0].id],
            diagnostics_json={"path_labels": path_labels},
        )
    )
    db_session.flush()
    contexts = context_package_to_contexts(package)
    answer = str(contexts[0]["content"])
    citations = agent_graph.citation_payloads_from_package(
        package,
        retrieval_trace_id=package.retrieval_trace_id,
        answer=answer,
        question="How do Bayesian networks factorize a joint distribution?",
    )
    assert citations
    return sample_knowledge_base, document, version, chunks, package, contexts, answer, citations


@pytest.mark.asyncio
async def test_db_citation_provenance_gate_binds_complete_source_identity(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    tmp_path,
    fake_model_stack,
):
    del fake_model_stack
    knowledge_base, _document, _version, _chunks, package, contexts, answer, citations = await _build_package(
        db_session,
        sample_knowledge_base,
        tmp_path,
    )

    gate = audit_citation_provenance(
        db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
        citations=citations,
        contexts=contexts,
    )
    assert gate["protocol_version"] == CITATION_PROVENANCE_PROTOCOL_VERSION
    assert gate["all_valid"] is True, [item["reasons"] for item in gate["audits"]]
    assert gate["invalid_count"] == 0
    assert len(gate["provenance_session_hash"]) == 64
    audit = gate["audits"][0]
    assert audit["valid"] is True
    assert audit["reasons"] == []
    assert len(audit["provenance_hash"]) == 64
    span = citations[0]["source_span"]
    assert span["raw_span_text_hash_protocol_version"] == RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION
    assert len(span["raw_span_text_hash"]) == 64
    assert span["chunk_text_hash"]
    assert span["source_snapshot_verification"]["verified"] is True
    assert span["context_package_id"] == package.id
    assert span["retrieval_trace_id"] == package.retrieval_trace_id

    empty_gate = audit_citation_provenance(
        db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
        citations=[],
        contexts=contexts,
    )
    assert empty_gate["citation_set_present"] is False
    assert empty_gate["all_valid"] is False

    results = await agent_graph.verify_answer_against_context(
        answer,
        citations,
        contexts,
        verification_budget=4,
        db=db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
    )
    assert results
    assert results[0]["diagnostics"]["citation_provenance_valid"] is True
    assert results[0]["diagnostics"]["citation_provenance_llm_override_allowed"] is False

    class EmptyEntailmentJudge:
        async def classify_json(self, **_kwargs):
            return {"verifications": []}

    monkeypatch.setattr(agent_graph, "ChatProvider", EmptyEntailmentJudge)
    missing_judgment = await agent_graph.verify_answer_against_context(
        answer,
        citations,
        contexts,
        verification_budget=4,
        db=db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
    )
    assert missing_judgment[0]["verdict"] == "unsupported"
    assert missing_judgment[0]["failure_type"] == "entailment_result_missing"
    assert missing_judgment[0]["diagnostics"][
        "llm_entailment_result_present"
    ] is False
    assert missing_judgment[1]["verdict"] == "supported"
    assert missing_judgment[1]["failure_type"] == "none"
    assert missing_judgment[1]["diagnostics"]["llm_entailment_judge"] == (
        "skipped_deterministic_exact_span"
    )


@pytest.mark.asyncio
async def test_answer_audit_persists_only_after_locked_provenance_replay(
    db_session,
    sample_knowledge_base,
    tmp_path,
    fake_model_stack,
):
    del fake_model_stack
    knowledge_base, _document, _version, _chunks, package, contexts, answer, _citations = await _build_package(
        db_session,
        sample_knowledge_base,
        tmp_path,
    )
    qa_session = QASession(
        knowledge_base_id=knowledge_base.id,
        title="Citation provenance audit",
    )
    db_session.add(qa_session)
    db_session.flush()

    question = "How do Bayesian networks factorize a joint distribution?"
    answer_model_audit = _exact_answer_model_audit(question)
    answer_session = await agent_graph.record_answer_audit(
        db_session,
        knowledge_base_id=knowledge_base.id,
        qa_session_id=qa_session.id,
        question=question,
        answer=answer,
        package=package,
        contexts=contexts,
        answer_model_audit=answer_model_audit,
    )

    gate = answer_session.diagnostics_json["citation_provenance_persistence_gate"]
    assert answer_session.prompt_protocol_version == "context_package_only_answer_grounding_envelope_v2"
    assert answer_session.model_json["prompt_protocol_hash"] == answer_model_audit[
        "prompt_protocol_hash"
    ]
    assert answer_session.diagnostics_json["answer_prompt_audit"] == {
        "prompt_protocol_version": "context_package_only_answer_grounding_envelope_v2",
        "prompt_protocol_hash": answer_model_audit["prompt_protocol_hash"],
        "grounding_envelope_protocol_version": "context_package_only_answer_grounding_envelope_v2",
        "grounding_envelope_hash": answer_model_audit["grounding_envelope_hash"],
        "profile_hash": answer_model_audit["profile_hash"],
        "context_quality": "normal",
        "server_recomputed": True,
        "exact_prompt_audit_verified": True,
        "missing_fields": [],
    }
    assert gate["passed"] is True
    assert gate["session_hash_matches"] is True
    assert gate["invalid_count"] == 0
    assert gate["transactional_replay"] is True
    assert gate["lock_backend"] == "sqlite"
    assert gate["rows_locked"] is False
    assert len(answer_session.diagnostics_json["citation_provenance_session_hash"]) == 64
    binding_hash = answer_session.diagnostics_json[
        "citation_answer_session_binding_hash"
    ]
    assert len(binding_hash) == 64
    verification = db_session.scalar(
        select(CitationVerification).where(
            CitationVerification.answer_session_id == answer_session.id
        )
    )
    assert verification is not None
    assert verification.verdict == "supported"
    assert verification.diagnostics_json[
        "citation_provenance_persistence_gate_passed"
    ] is True
    assert verification.diagnostics_json[
        "citation_provenance_session_hash_matches"
    ] is True
    assert verification.diagnostics_json[
        "citation_answer_session_binding_hash"
    ] == binding_hash
    assert verification.diagnostics_json[
        "citation_answer_session_binding_protocol_version"
    ] == "citation_answer_session_binding_v1"
    reward = db_session.scalar(
        select(RewardEvent).where(RewardEvent.answer_session_id == answer_session.id)
    )
    assert reward is not None
    assert reward.policy_state_id is None
    assert reward.diagnostics_json["policy_reward_training_eligible"] is False
    assert (
        reward.diagnostics_json["policy_reward_training_ineligible_reason"]
        == "missing_agent_run"
    )
    assert reward.action_json["prompt_protocol_hash"] == answer_model_audit[
        "prompt_protocol_hash"
    ]
    assert reward.action_json["grounding_envelope_hash"] == answer_model_audit[
        "grounding_envelope_hash"
    ]
    assert reward.diagnostics_json["answer_prompt_audit"][
        "profile_hash"
    ] == answer_model_audit["profile_hash"]

    package.package_json["chunks"][0]["source_span"]["raw_span_text_hash"] = (
        "0" * 64
    )
    flag_modified(package, "package_json")
    db_session.flush()
    tampered_contexts = context_package_to_contexts(package)
    rejected_session = await agent_graph.record_answer_audit(
        db_session,
        knowledge_base_id=knowledge_base.id,
        qa_session_id=qa_session.id,
        question="How do Bayesian networks factorize a joint distribution?",
        answer=answer,
        package=package,
        contexts=tampered_contexts,
        answer_model_audit={"provider": "unit_chat", "model": "unit-chat"},
    )
    rejected = db_session.scalar(
        select(CitationVerification).where(
            CitationVerification.answer_session_id == rejected_session.id
        )
    )
    assert rejected is not None
    assert rejected.verdict == "structure_context_missing"
    assert rejected.chunk_id is None
    assert rejected.diagnostics_json["authoritative_chunk_link_persisted"] is False
    assert rejected.diagnostics_json["attempted_chunk_id"]
    assert rejected_session.diagnostics_json[
        "citation_provenance_persistence_gate"
    ]["passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tampered_field",
    ["prompt_protocol_hash", "grounding_envelope_hash"],
)
async def test_answer_audit_rejects_forged_exact_prompt_metadata_before_persistence(
    tampered_field,
    db_session,
    sample_knowledge_base,
    tmp_path,
    fake_model_stack,
):
    del fake_model_stack
    (
        knowledge_base,
        _document,
        _version,
        _chunks,
        package,
        contexts,
        answer,
        _citations,
    ) = await _build_package(db_session, sample_knowledge_base, tmp_path)
    qa_session = QASession(
        knowledge_base_id=knowledge_base.id,
        title="Prompt isolation forged prompt audit",
    )
    db_session.add(qa_session)
    db_session.flush()
    question = "How do Bayesian networks factorize a joint distribution?"
    answer_model_audit = _exact_answer_model_audit(question)
    answer_model_audit[tampered_field] = "0" * 64
    answer_session_count = len(
        db_session.scalars(
            select(AnswerSession).where(
                AnswerSession.qa_session_id == qa_session.id
            )
        ).all()
    )

    with pytest.raises(ValueError, match="prompt|envelope"):
        await agent_graph.record_answer_audit(
            db_session,
            knowledge_base_id=knowledge_base.id,
            qa_session_id=qa_session.id,
            question=question,
            answer=answer,
            package=package,
            contexts=contexts,
            answer_model_audit=answer_model_audit,
        )

    assert len(
        db_session.scalars(
            select(AnswerSession).where(
                AnswerSession.qa_session_id == qa_session.id
            )
        ).all()
    ) == answer_session_count


@pytest.mark.asyncio
async def test_citation_provenance_tampering_fails_before_entailment(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    tmp_path,
):
    knowledge_base, _document, _version, _chunks, package, contexts, answer, citations = await _build_package(
        db_session,
        sample_knowledge_base,
        tmp_path,
    )
    class ForbiddenJudge:
        async def classify_json(self, **_kwargs):
            raise AssertionError("LLM entailment must not receive invalid provenance")

    monkeypatch.setattr(agent_graph, "ChatProvider", ForbiddenJudge)
    cases = {
        "out_of_bounds": "citation_source_span_mismatch",
        "cross_package": "citation_context_package_id_mismatch",
        "wrong_version": "citation_document_version_identity_mismatch",
        "wrong_page": "citation_source_span_mismatch",
        "wrong_section": "citation_source_span_mismatch",
        "wrong_bbox": "citation_source_span_mismatch",
        "wrong_raw_hash": "citation_source_span_mismatch",
        "wrong_structure": "citation_source_span_mismatch",
        "wrong_trace": "citation_retrieval_trace_id_mismatch",
    }
    for mutation, expected_reason in cases.items():
        tampered = copy.deepcopy(citations)
        for citation in tampered:
            span = citation["source_span"]
            if mutation == "out_of_bounds":
                span["char_span"] = [-1, 999999]
            elif mutation == "cross_package":
                span["context_package_id"] = "forged-package"
            elif mutation == "wrong_version":
                citation["document_version_id"] = "forged-version"
                span["document_version_id"] = "forged-version"
            elif mutation == "wrong_page":
                span["page_range"] = [999, 999]
            elif mutation == "wrong_section":
                span["section_path"] = "forged / section"
            elif mutation == "wrong_bbox":
                span["bbox"] = {
                    "x0": -1,
                    "y0": -1,
                    "x1": 999,
                    "y1": 999,
                }
            elif mutation == "wrong_raw_hash":
                span["raw_span_text_hash"] = "f" * 64
            elif mutation == "wrong_structure":
                span["structure_node_ids"] = ["forged-structure-node"]
            elif mutation == "wrong_trace":
                citation["retrieval_trace_id"] = "forged-trace"
                span["retrieval_trace_id"] = "forged-trace"

        provenance_gate = audit_citation_provenance(
            db_session,
            knowledge_base_id=knowledge_base.id,
            package=package,
            citations=tampered,
            contexts=contexts,
        )
        assert provenance_gate["invalid_count"] == len(tampered), mutation
        pre_judge_results = agent_graph.verify_answer_against_context_rules(
            answer,
            tampered,
            contexts,
            verification_budget=4,
            provenance_gate=provenance_gate,
        )
        assert not [
            result
            for result in pre_judge_results
            if (result.get("diagnostics") or {}).get(
                "citation_provenance_valid"
            )
        ], mutation

        results = await agent_graph.verify_answer_against_context(
            answer,
            tampered,
            contexts,
            verification_budget=4,
            db=db_session,
            knowledge_base_id=knowledge_base.id,
            package=package,
        )

        assert results[0]["verdict"] == "structure_context_missing", mutation
        assert results[0]["failure_type"] == "structure_context_missing", mutation
        assert expected_reason in results[0]["diagnostics"]["citation_provenance_reasons"], mutation
        assert results[0]["diagnostics"]["llm_entailment_judge"] == "skipped_provenance_failed", mutation


@pytest.mark.asyncio
async def test_stale_document_version_and_forged_structure_path_fail_closed(
    db_session,
    sample_knowledge_base,
    tmp_path,
):
    knowledge_base, _document, version, _chunks, package, contexts, _answer, citations = await _build_package(
        db_session,
        sample_knowledge_base,
        tmp_path,
    )
    chunk_id = citations[0]["chunk_id"]
    item = next(
        item
        for item in package.package_json["chunks"]
        if item["chunk_id"] == chunk_id
    )
    item["role"] = "graph_path"
    item["why_selected"] = {
        **(item.get("why_selected") or {}),
        "path_edge_ids": ["forged-edge"],
    }
    item["structure_nodes"][0]["mapping_weight"] = 999.0
    package.restored_chunk_ids_json = list(package.restored_chunk_ids_json or []) + [
        chunk_id
    ]
    package.why_selected_json = {
        **(package.why_selected_json or {}),
        chunk_id: item["why_selected"],
    }
    contexts = context_package_to_contexts(package)
    version.is_active = False
    flag_modified(package, "package_json")
    flag_modified(package, "restored_chunk_ids_json")
    flag_modified(package, "why_selected_json")
    db_session.flush()

    gate = audit_citation_provenance(
        db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
        citations=citations,
        contexts=contexts,
    )

    reasons = gate["audits"][0]["reasons"]
    assert "citation_document_version_inactive" in reasons
    assert "citation_role_scope_overlap" in reasons
    assert "citation_graph_edge_support_outside_package" in reasons
    assert "context_package_structure_node_forged" in reasons
    assert gate["all_valid"] is False


@pytest.mark.asyncio
async def test_consistently_forged_package_trace_path_lacks_executor_proof(
    db_session,
    sample_knowledge_base,
    tmp_path,
):
    knowledge_base, _document, _version, _chunks, package, contexts, _answer, citations = await _build_package(
        db_session,
        sample_knowledge_base,
        tmp_path,
    )
    trace = db_session.get(RetrievalTrace, package.retrieval_trace_id)
    structure_step = db_session.scalar(
        select(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == trace.id,
            GraphRetrievalStep.layer == "structure",
        )
    )
    item = package.package_json["chunks"][0]
    forged_edge_id = "forged-edge-without-executor-proof"
    trace.path_labels_json[0]["path_edge_ids"] = [forged_edge_id]
    package.graph_path_ids_json = [forged_edge_id]
    item["why_selected"] = {
        **item["why_selected"],
        "path_edge_ids": [forged_edge_id],
    }
    package.why_selected_json[item["chunk_id"]] = item["why_selected"]
    structure_step.output_json["graph_path_ids"] = [forged_edge_id]
    structure_step.expanded_edge_ids_json = [forged_edge_id]
    flag_modified(trace, "path_labels_json")
    flag_modified(package, "package_json")
    flag_modified(package, "graph_path_ids_json")
    flag_modified(package, "why_selected_json")
    flag_modified(structure_step, "output_json")
    flag_modified(structure_step, "expanded_edge_ids_json")
    db_session.flush()
    contexts = context_package_to_contexts(package)

    gate = audit_citation_provenance(
        db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
        citations=citations,
        contexts=contexts,
    )

    assert gate["all_valid"] is False
    assert "retrieval_trace_executor_path_mismatch" in gate["audits"][0][
        "reasons"
    ]


@pytest.mark.asyncio
async def test_persistence_replay_rejects_provenance_changed_after_entailment(
    db_session,
    sample_knowledge_base,
    tmp_path,
):
    knowledge_base, _document, _version, _chunks, package, contexts, _answer, citations = await _build_package(
        db_session,
        sample_knowledge_base,
        tmp_path,
    )
    before = audit_citation_provenance(
        db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
        citations=citations,
        contexts=contexts,
    )
    assert before["all_valid"] is True

    package.package_json["chunks"][0]["source_span"]["raw_span_text_hash"] = "0" * 64
    flag_modified(package, "package_json")
    db_session.flush()
    replay = replay_citation_provenance_for_persistence(
        db_session,
        knowledge_base_id=knowledge_base.id,
        package=package,
        citations=citations,
        contexts=contexts,
        expected_session_hash=before["provenance_session_hash"],
    )

    assert replay["matches_pre_entailment_session_hash"] is False
    assert replay["persistence_gate_passed"] is False
    assert replay["all_valid"] is False
    assert replay["transactional_replay"] is True
    assert replay["lock_backend"] == "sqlite"
    assert replay["rows_locked"] is False
