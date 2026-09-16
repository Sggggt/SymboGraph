import copy

import pytest
from sqlalchemy import select

from app.models import AgentObservation, AgentRun, AnswerSourceBinding
from app.retrieval_control_contracts import GroundedAnswerDraft, SourceGateAdmission
from app.services.agent_reflection import ReflectionContractError
from app.services.answer_sources import (
    build_answer_evidence_manifest, persist_answer_source_bindings, source_binding_citations,
)
from test_answer_sources import new_answer
from test_citation_provenance import _build_package
from test_reflection_sources import audit_package


async def prepared_case(db, kb, tmp_path):
    _, _, _, _, package, contexts, text, _ = await _build_package(db, kb, tmp_path)
    manifest = build_answer_evidence_manifest(package, contexts)
    source_audit = audit_package(db, package)
    assert source_audit["all_valid"]
    run = AgentRun(knowledge_base_id=kb.id, question=package.query, status="running",
                   metadata_json={"retrieval_control": {"task_hash": "a" * 64}})
    db.add(run)
    db.flush()
    admission = SourceGateAdmission(run_id=run.id, knowledge_base_id=kb.id,
        context_package_id=package.id, retrieval_trace_id=package.retrieval_trace_id,
        task_hash="a" * 64, strategy_hash="b" * 64, feature_hash="c" * 64, outcome="ready_full",
        evidence_manifest_hash=manifest.manifest_hash,
        provenance_session_hash=source_audit["provenance_session_hash"],
        source_chunk_ids=tuple(item["chunk_id"] for item in package.package_json["chunks"]))
    gate = AgentObservation(run_id=run.id, observation_type="retrieval_gate", verdict="ready_full",
        observation_json={"source_admission": admission.model_dump(mode="json"),
                          "source_admission_hash": admission.identity})
    db.add(gate)
    db.flush()
    draft = GroundedAnswerDraft.model_validate({"answer_units": [
        {"kind": "factual", "text": text, "source_handles": ["src_1"]}]})
    answer = new_answer(db, kb, package, draft)
    return run, gate, package, contexts, manifest, draft, answer


@pytest.mark.asyncio
async def test_new_binding_uses_persisted_retrieval_gate_without_self_assessment(
    db_session, sample_knowledge_base, tmp_path, fake_model_stack,
):
    run, gate, package, contexts, manifest, draft, answer = await prepared_case(
        db_session, sample_knowledge_base, tmp_path)
    assert not hasattr(draft, "self_assessment")
    rows = persist_answer_source_bindings(db_session, answer_session=answer, package=package,
        contexts=contexts, draft=draft, evidence=manifest, unit_limit=12, retrieval_gate=gate)
    assert rows and all(row.retrieval_gate_observation_id == gate.id for row in rows)
    assert all(row.protocol_version == "answer_source_binding_v2" for row in rows)
    assert all("reflection_audit_hash" not in row.diagnostics_json["binding_identity"] for row in rows)
    db_session.commit()
    db_session.expire_all()
    rows = list(db_session.scalars(select(AnswerSourceBinding).where(AnswerSourceBinding.answer_session_id == answer.id)))
    citations = source_binding_citations(answer_session=answer, package=package, rows=rows, retrieval_gate=gate)
    public = citations[0]["source_binding"]
    assert public["contract_version"] == "answer_source_binding_public_v2"
    assert public["retrieval_gate_observation_id"] == gate.id
    assert public["reflection_audit_hash"] is None
    assert public["semantic_entailment_claimed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["task", "manifest", "outcome"])
async def test_new_binding_rejects_wrong_task_source_or_unaccepted_gate(
    db_session, sample_knowledge_base, tmp_path, fake_model_stack, tamper,
):
    run, gate, package, contexts, manifest, draft, answer = await prepared_case(
        db_session, sample_knowledge_base, tmp_path)
    if tamper == "task":
        answer.question = "A different current question."
    elif tamper == "manifest":
        payload = copy.deepcopy(gate.observation_json)
        payload["source_admission"]["evidence_manifest_hash"] = "f" * 64
        payload["source_admission_hash"] = SourceGateAdmission.model_validate(payload["source_admission"]).identity
        gate.observation_json = payload
    else:
        gate.verdict = "repairable"
    with pytest.raises(ReflectionContractError):
        persist_answer_source_bindings(db_session, answer_session=answer, package=package,
            contexts=contexts, draft=draft, evidence=manifest, unit_limit=12, retrieval_gate=gate)
    assert not list(db_session.scalars(select(AnswerSourceBinding).where(AnswerSourceBinding.answer_session_id == answer.id)))
