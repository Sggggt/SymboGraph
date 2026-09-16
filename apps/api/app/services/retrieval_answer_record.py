"""Read current source bindings and historical v1 records without model calls."""
from sqlalchemy import select

from app.models import AgentObservation, AgentRun, AnswerSourceBinding
from app.retrieval_control_contracts import GroundedAnswerDraft, control_hash
from app.services.agent_reflection import render_answer_units, reflection_hash
from app.services.answer_sources import source_binding_citations, verify_sufficiency_owner


def replay_answer_bindings(db, *, answer, package):
    rows = list(db.scalars(select(AnswerSourceBinding).where(AnswerSourceBinding.answer_session_id == answer.id)))
    if sorted(row.id for row in rows) != sorted(answer.citation_ids_json or []):
        raise ValueError("answer_source_binding_row_scope_changed")
    diagnostics = answer.diagnostics_json or {}
    source_admission = diagnostics.get("source_integrity_admission")
    if source_admission:
        if (
            source_admission.get("protocol_version") != "source_integrity_admission_v1"
            or source_admission.get("outcome") != "passed"
            or source_admission.get("context_package_id") != package.id
            or source_admission.get("retrieval_trace_id") != package.retrieval_trace_id
            or source_admission.get("model_call_count") != 0
            or source_admission.get("score_fields_used_for_admission") != []
            or source_admission.get("audit_hash") != control_hash(
                {key: value for key, value in source_admission.items() if key != "audit_hash"}
            )
        ):
            raise ValueError("source_integrity_answer_audit_invalid")
        observation = db.get(AgentObservation, source_admission.get("observation_id"))
        if (
            observation is None
            or observation.observation_type != "source_integrity_admission"
            or observation.verdict != "passed"
            or observation.observation_json != source_admission
        ):
            raise ValueError("source_integrity_answer_observation_missing")
        return source_binding_citations(
            answer_session=answer,
            package=package,
            rows=rows,
            source_integrity_admission=observation,
        )
    current = diagnostics.get("retrieval_control")
    if current:
        if (current.get("protocol_version") != "retrieval_answer_v1"
            or current.get("audit_hash") != control_hash({key: value for key, value in current.items() if key != "audit_hash"})
            or current.get("generation_call_count") != 1 or current.get("post_generation_review_count") != 0):
            raise ValueError("retrieval_answer_audit_invalid")
        draft = GroundedAnswerDraft.model_validate(diagnostics.get("structured_answer"))
        if render_answer_units(draft)[0] != answer.answer:
            raise ValueError("retrieval_answer_text_identity_changed")
        gate = db.get(AgentObservation, current.get("gate_observation_id"))
        if gate is None or gate.run_id != current.get("run_id"):
            raise ValueError("retrieval_answer_gate_missing")
        if gate.observation_json.get("source_admission_hash") != current.get("source_admission_hash"):
            raise ValueError("retrieval_answer_gate_identity_changed")
        verify_sufficiency_owner(db,owner=db.get(AgentRun,current['run_id']),retrieval_gate=gate,
            required=current.get('evidence_sufficiency_protocol') is not None)
        inputs=gate.observation_json.get('feature_input') or {}
        if (inputs.get('parameters',{}).get('scope_selection') is not None
                or (gate.observation_json.get('source_admission') or {}).get('source_addressed_assessment_hash') is not None):
            from app.retrieval_control_contracts import TaskContract,PathEvaluationParameters
            from app.services.evidence_scope import replay_scope_inputs
            replay_scope_inputs(db,task=TaskContract.model_validate(inputs['task']),
                parameters=PathEvaluationParameters.model_validate(inputs['parameters']),package=package)
        return source_binding_citations(answer_session=answer, package=package, rows=rows, retrieval_gate=gate)
    legacy = diagnostics.get("answer_reflection") or {}
    if (legacy.get("protocol_version") != "agent_answer_reflection_v1"
        or legacy.get("audit_hash") != reflection_hash({key: value for key, value in legacy.items() if key != "audit_hash"})):
        raise ValueError("historical_answer_reflection_audit_invalid")
    return source_binding_citations(answer_session=answer, package=package, rows=rows,
                                   reflection_audit_hash=legacy["audit_hash"])
