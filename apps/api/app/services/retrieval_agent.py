"""The active retrieval-end gate, bounded lexical revision and one answer."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import re
import time

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import get_settings
from app.models import AgentObservation, AgentRun, AgentTraceEvent, AnswerSession, ContextPackage, RetrievalTrace
from app.retrieval_control_contracts import (
    GateThresholds, RetrievalAnswerSummary, SourceGateAdmission, control_hash,
)
from app.services.agent_reflection import history_summary_projection, render_answer_units
from app.services.answer_sources import build_answer_evidence_manifest, persist_answer_source_bindings, source_binding_citations
from app.services.citation_provenance import audit_citation_provenance
from app.services.context_graph import (
    agent_operating_envelope, build_context_package, context_package_to_contexts,
    schedule_layered_retrieval_cache_write,
)
from app.services.embeddings import EmbeddingProvider
from app.services.error_sanitizer import external_failure_classification
from app.services.lexical_patch import compile_lexical_patch, LexicalPatchNoProgress
from app.services.qa_performance import QAPerformance, current_qa_performance, qa_stage
from app.services.reflection_sources import source_citation, pack_and_retain_reflection_sources
from app.services.retrieval_corpus import RetrievalCorpus, EmptyFilteredRetrievalScope
from app.services.retrieval_query_vectors import prepare_query_vectors
from app.services.retrieval_execution import compile_retrieval_execution, persist_retrieval_execution
from app.services.retrieval_feature_adapter import build_feature_snapshot
from app.services.retrieval_fsm import advance_control, initialize_control, RetrievalControlState
from app.services.retrieval_models import RetrievalModels, compile_task_plan, routing_facets, normalize_source_roles, project_task_perception, lexical_repair_packet
from app.services.retrieval_path_features import decide_retrieval_gate
from app.services.retrieval_reward import read_lexical_policy, record_retrieval_reward
from app.services.retrieval_packing import plan_packing_repair, plan_scope_packing_repair,required_scope_retention
from app.services.reflection_context import restore_reflection_context
from app.services.retrieval_sufficiency import (
    PROTOCOL as SUFFICIENCY_PROTOCOL, READY, constrain_gate, repair_feedback,
    sufficiency_packet, replay_sufficiency,
    sufficiency_model_limits,
)
from app.services.source_addressed_assessment import (
    PROTOCOL as SOURCE_ADDRESSED_PROTOCOL, ELIGIBLE_OUTCOMES, build_assessment,
    assessment_guidance, replay_assessment_guidance,
)


def _save_repair_request(db, *, run, task, strategy, packet, witnesses, timeout_seconds, max_tokens):
    payload = {"protocol_version": "retrieval_repair_request_v1", "task_hash": task.identity,
        "strategy_hash": strategy.identity, "input_hash": control_hash(packet), "packet": packet,
        "witnesses": [asdict(witness) for witness in witnesses],
        "timeout_seconds": timeout_seconds, "max_tokens": max_tokens,
        "provider_response_persisted": False}
    payload['audit_hash'] = control_hash(payload)
    row = AgentObservation(run_id=run.id, observation_type='retrieval_repair_request',
        verdict='prepared', observation_json=payload)
    db.add(row)
    with qa_stage('database_commit'):
        db.commit()
    return row


def _save_location_request(db,*,run,task,request,timeout_seconds,max_tokens):
    from app.services.source_location import location_packet
    payload={'protocol_version':'source_location_call_v1','status':'prepared','run_id':run.id,
        'task_hash':task.identity,'request':request.model_dump(mode='json'),
        'input_hash':control_hash(location_packet(task,request)),
        'control_sequence_index':run.metadata_json['retrieval_control']['sequence_index'],
        'timeout_seconds':timeout_seconds,'max_tokens':max_tokens,'provider_response_persisted':False,
        'semantic_identity_proven':False,'policy_update_eligible':False}
    payload['audit_hash']=control_hash(payload)
    row=AgentObservation(run_id=run.id,observation_type='retrieval_scope_resolution',verdict='prepared',observation_json=payload)
    db.add(row)
    with qa_stage('database_commit'):
        db.commit()
    return row


def _source_audit(db, package):
    return audit_citation_provenance(db, knowledge_base_id=package.knowledge_base_id, package=package,
        contexts=context_package_to_contexts(package),
        citations=[source_citation(item, package) for item in package.package_json["chunks"]])


def _record_materialized_package(db, *, run, action, package, source_audit):
    if action.run_id != run.id or action.action_type != 'build_context_package' or package.knowledge_base_id != run.knowledge_base_id:
        raise ValueError('retrieval_materialization_action_identity_invalid')
    payload = {'protocol_version':'retrieval_package_materialization_v1','context_package_id':package.id,
        'retrieval_trace_id':package.retrieval_trace_id,'source_integrity':source_audit['all_valid'],
        'source_audit_hash':source_audit['provenance_session_hash'],'model_call_count':0}
    payload['audit_hash'] = control_hash(payload)
    action.status = 'completed' if source_audit['all_valid'] else 'failed'
    action.output_json = {**action.output_json,**payload}
    db.add(AgentObservation(run_id=run.id,action_id=action.id,observation_type='context_package_built',
        verdict='materialized',observation_json=payload,evidence_chunk_ids_json=list(package.hit_chunk_ids_json)))
    db.commit()


def _save_gate(db, *, run, task, strategy, package, features, decision, source_audit, replay_input,
               source_use_audit=None, sufficiency=None, path_decision=None):
    if features.scope_statuses:
        from app.retrieval_control_contracts import PathEvaluationParameters
        from app.services.evidence_scope import replay_scope_inputs
        parameters=PathEvaluationParameters.model_validate(replay_input['parameters'])
        if parameters.scope_selection is not None and parameters.scope_selection.run_id!=run.id:
            raise ValueError('source_location_run_identity_changed')
        replay_scope_inputs(db,task=task,parameters=parameters,package=package)
    manifest = (build_answer_evidence_manifest(package,context_package_to_contexts(package))
                if decision.outcome in READY or sufficiency is not None else None)
    payload = {"protocol_version": "retrieval_gate_observation_v1", "task_hash": task.identity,
               "strategy_hash": strategy.identity, "features": features.model_dump(mode="json"),
               "decision": decision.model_dump(mode="json"), "feature_input": replay_input,
               "context_package_id": package.id,
               "control_sequence_index": run.metadata_json["retrieval_control"]["sequence_index"]}
    if source_use_audit is not None:
        if (source_use_audit.get('feature_input_hash') != features.input_hash
            or source_use_audit.get('audit_hash') != control_hash({k: v for k, v in source_use_audit.items() if k != 'audit_hash'})):
            raise ValueError('retrieval_source_use_audit_invalid')
        payload['source_use_audit'] = dict(source_use_audit)
    if control_hash(replay_input) != features.input_hash:
        raise ValueError("retrieval_gate_feature_input_identity_invalid")
    if path_decision is not None:
        payload['path_decision'] = path_decision.model_dump(mode='json')
    if sufficiency is not None:
        assessment, guidance = replay_assessment_guidance(sufficiency, task=task, features=features,
            path_decision=path_decision or decision, replay_input=replay_input, evidence=manifest, source_audit=source_audit)
        result = replay_sufficiency(sufficiency,task=task,strategy_hash=strategy.identity,evidence=manifest,source_scopes=guidance)
        state = RetrievalControlState.model_validate(run.metadata_json['retrieval_control'])
        expected = constrain_gate(path_decision or decision,task=task,result=result,
            remaining_repairs=state.repair_limit-state.repairs_used,actionable_ids=decision.proposed_action_ids,assessment=assessment)
        if expected != decision:
            raise ValueError('retrieval_sufficiency_gate_decision_changed')
        payload['evidence_sufficiency'] = sufficiency
    pending = (run.metadata_json.get('evidence_sufficiency_protocol') == SUFFICIENCY_PROTOCOL and sufficiency is None)
    if pending and decision.outcome in READY:
        payload['admission_pending'] = True
    if decision.outcome in READY and not pending:
        if not source_audit["all_valid"]:
            raise ValueError("retrieval_gate_source_provenance_failed")
        admission = SourceGateAdmission(run_id=run.id, knowledge_base_id=run.knowledge_base_id,
            context_package_id=package.id, retrieval_trace_id=package.retrieval_trace_id,
            task_hash=task.identity, strategy_hash=strategy.identity,
            feature_hash=control_hash(features.model_dump(mode="json")), outcome=decision.outcome,
            evidence_manifest_hash=manifest.manifest_hash,
            provenance_session_hash=source_audit["provenance_session_hash"],
            source_chunk_ids=tuple(item["chunk_id"] for item in package.package_json["chunks"]),
            evidence_sufficiency_hash=sufficiency['audit_hash'] if sufficiency else None,
            source_addressed_assessment_hash=sufficiency.get('source_addressed_assessment_hash') if sufficiency else None,
            resolved_scope_filter_hash=(run.metadata_json.get('resolved_scope_filter') or {}).get('audit_hash'))
        payload.update(source_admission=admission.model_dump(mode="json"), source_admission_hash=admission.identity)
    row = AgentObservation(run_id=run.id, observation_type="retrieval_gate", verdict=decision.outcome,
                           observation_json=payload, evidence_chunk_ids_json=list(package.hit_chunk_ids_json))
    db.add(row)
    with qa_stage("database_commit"):
        db.commit()
    return row


def _assessment_candidate(*, task, features, decision, replay_input, package, source_audit, thresholds):
    if (decision.outcome not in ELIGIBLE_OUTCOMES or not source_audit['all_valid']
            or any(f.source_scope is None for f in task.requirements)):
        return None
    evidence = build_answer_evidence_manifest(package, context_package_to_contexts(package))
    return build_assessment(task=task, features=features, decision=decision, replay_input=replay_input,
        evidence=evidence, source_audit=source_audit, thresholds=thresholds)


async def _assess_ready_package(db, *, run, task, strategy, package, features, decision,
        source_audit, replay_input, source_use_audit, corpus, vectors, scope_index,
        thresholds, settings, model, advance, reused, assessment=None):
    """Assess the whole bounded package; numeric maxima cannot justify deleting complementary facts."""
    from app.services import agent_graph as ag
    from app.retrieval_control_contracts import PathEvaluationParameters
    from app.services.evidence_scope import generation_scope_guidance
    pre_gate = await ag.run_bounded_source_io(_save_gate,db,run=run,task=task,strategy=strategy,package=package,
        features=features,decision=decision,source_audit=source_audit,replay_input=replay_input,source_use_audit=source_use_audit)
    from app.services.generation_packing import plan_scope_generation_packing, verify_generation_packing
    packing_plan = plan_scope_generation_packing(task=task,package=package,features=features,replay_input=replay_input)
    if packing_plan is not None:
        packing_payload={**packing_plan.model_dump(mode='json'),'before_gate_observation_id':pre_gate.id,
            'control_sequence_index':run.metadata_json['retrieval_control']['sequence_index']}
        packing_payload['audit_hash']=control_hash(packing_payload)
        packing_row=AgentObservation(run_id=run.id,observation_type='retrieval_generation_packing',
            verdict='prepared',observation_json=packing_payload)
        db.add(packing_row)
        db.commit()
        previous_features,previous_decision=features,decision
        previous_items={item['chunk_id']:item for item in package.package_json['chunks']}
        with qa_stage('generation_packing'):
            package,_=await ag.run_bounded_source_io(restore_reflection_context,db,source_package=package,
                target_chunk_ids=list(packing_plan.selected_chunk_ids),preserve_chunk_ids=list(packing_plan.selected_chunk_ids),
                packing_priority_chunk_ids=list(packing_plan.selected_chunk_ids),
                token_budget=package.token_budget,reserved_token_budget=packing_plan.reserved_token_budget,
                restore_per_chunk_budget=0)
            actual=package.package_json['chunks']
            if (tuple(item['chunk_id'] for item in actual)!=packing_plan.selected_chunk_ids or any(
                    any(item.get(key)!=previous_items[item['chunk_id']].get(key)
                        for key in ('content','document_version_id','char_span','raw_chunk_char_span','content_clipped'))
                    for item in actual)):
                raise ValueError('scope_generation_packing_materialization_changed')
            source_audit=await ag.run_bounded_source_io(_source_audit,db,package)
            actual_trace=await ag.run_bounded_source_io(db.get,RetrievalTrace,package.retrieval_trace_id)
            features,_,_,replay_input=await ag.run_bounded_source_io(build_feature_snapshot,db,
                task=task,strategy=strategy,corpus=corpus,vectors=vectors,package=package,trace=actual_trace,
                source_audit=source_audit,include_panels=not reused,source_use_audit=source_use_audit,scope_index=scope_index)
            verify_generation_packing(previous_features,features)
        state=RetrievalControlState.model_validate(run.metadata_json['retrieval_control'])
        decision=decide_retrieval_gate(task=task,features=features,thresholds=thresholds,
            source_integrity=source_audit['all_valid'],remaining_repairs=state.repair_limit-state.repairs_used)
        if not source_audit['all_valid'] or decision.outcome!=previous_decision.outcome:
            raise ValueError('scope_generation_packing_gate_changed')
        refreshed=await ag.run_bounded_source_io(_assessment_candidate,task=task,features=features,decision=decision,
            replay_input=replay_input,package=package,source_audit=source_audit,thresholds=thresholds)
        if assessment is not None and refreshed is None:
            raise ValueError('scope_generation_packing_assessment_lost')
        assessment=refreshed
        packing_payload={k:v for k,v in packing_payload.items() if k!='audit_hash'}
        packing_payload.update(target_context_package_id=package.id,feature_input_hash_after=features.input_hash,
            actual_source_count=len(actual),actual_text_characters=sum(len(item['content']) for item in actual),
            coverage_bounds_preserved=True,source_integrity_passed=True,repair_count_unchanged=True)
        packing_payload['audit_hash']=control_hash(packing_payload)
        packing_row.verdict,packing_row.observation_json='completed',packing_payload
        # Commit the materialized package, completion audit and matching gate
        # together. A failure leaves the earlier prepared intent recoverable.
        pre_gate=await ag.run_bounded_source_io(_save_gate,db,run=run,task=task,strategy=strategy,package=package,
            features=features,decision=decision,source_audit=source_audit,replay_input=replay_input,source_use_audit=source_use_audit)
    state = RetrievalControlState.model_validate(run.metadata_json['retrieval_control'])
    evidence = build_answer_evidence_manifest(package,context_package_to_contexts(package))
    guidance = assessment_guidance(assessment) if assessment else generation_scope_guidance(task=task,
        parameters=PathEvaluationParameters.model_validate(replay_input['parameters']),evidence=evidence)
    await advance('evaluating')
    timeout, tokens = sufficiency_model_limits(settings)
    payload = {'protocol_version':'retrieval_sufficiency_call_v1','status':'prepared',
        'task_hash':task.identity,'strategy_hash':strategy.identity,'context_package_id':package.id,
        'evidence_manifest_hash':evidence.manifest_hash,
        'input_hash':control_hash(sufficiency_packet(task=task,evidence=evidence,source_scopes=guidance)),
        'control_sequence_index':run.metadata_json['retrieval_control']['sequence_index'],
        'timeout_seconds':timeout,'max_tokens':tokens,'provider_response_persisted':False}
    if assessment is not None:
        payload.update(protocol_version='retrieval_sufficiency_call_v2',
            source_addressed_gate_observation_id=pre_gate.id,
            source_addressed_assessment=assessment.model_dump(mode='json'),source_addressed_assessment_hash=assessment.identity)
        replay_assessment_guidance(payload, task=task, features=features, path_decision=decision,
            replay_input=replay_input, evidence=evidence, source_audit=source_audit)
    payload['audit_hash'] = control_hash(payload)
    row = AgentObservation(run_id=run.id,observation_type='retrieval_sufficiency',verdict='prepared',observation_json=payload)
    db.add(row)
    db.commit()
    result, model_audit = await model.assess_evidence(task=task,evidence=evidence,source_scopes=guidance,
        timeout_seconds=timeout,max_tokens=tokens)
    if model_audit['input_hash'] != payload['input_hash']:
        raise ValueError('retrieval_sufficiency_model_input_changed')
    ag.ensure_agent_run_not_cancelled(db,run)
    payload = {k:v for k,v in payload.items() if k!='audit_hash'}
    payload.update(status='completed',observation_id=row.id,result=result.model_dump(mode='json'),model_audit=model_audit)
    payload['audit_hash'] = control_hash(payload)
    row.verdict,row.observation_json = 'completed',payload
    db.commit()
    await advance('diagnosing')
    effective = constrain_gate(decision,task=task,result=result,remaining_repairs=state.repair_limit-state.repairs_used,assessment=assessment)
    return package,features,effective,source_audit,replay_input,payload,decision


def _finish_answer(db, *, request, session, run, task, strategy, package, gate_row,
                   features, decision, draft, model_audit, reused):
    from app.services import agent_graph as ag
    from app.services.storage import raise_if_source_io_cancelled
    row = db.scalar(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
    if row.status == "cancelled":
        raise asyncio.CancelledError("cancelled_by_user")
    state = RetrievalControlState.model_validate(row.metadata_json["retrieval_control"])
    answer, units = render_answer_units(draft)
    contexts = context_package_to_contexts(package)
    from app.models import Chunk
    from app.services.context_graph import passes_filters
    from app.schemas import SearchFilters
    filter_audit=run.metadata_json.get('resolved_scope_filter')
    final_filters=request.filters
    if filter_audit is not None:
        if (filter_audit.get('task_hash')!=task.identity
            or filter_audit.get('original_filters')!=request.filters.model_dump(mode='json')
            or filter_audit.get('audit_hash')!=control_hash({k:v for k,v in filter_audit.items() if k!='audit_hash'})):
            raise ValueError('source_scope_filter_identity_changed')
        final_filters=SearchFilters.model_validate(filter_audit['effective_filters'])
    if any(not passes_filters(db,db.get(Chunk,item['chunk_id']),final_filters) for item in package.package_json['chunks']):
        raise ValueError('retrieval_final_source_filter_scope_invalid')
    manifest = build_answer_evidence_manifest(package, contexts)
    route = "direct_answer" if reused else "layered_context_graph"
    direct_mode = "verified_context_reuse" if reused else None
    audit = {"protocol_version": "retrieval_answer_v1", "run_id": run.id,
             "evidence_sufficiency_protocol": SUFFICIENCY_PROTOCOL,
             "task_hash": task.identity, "strategy_hash": strategy.identity,
             "gate_observation_id": gate_row.id, "source_admission_hash": gate_row.observation_json["source_admission_hash"],
             "gate_outcome": decision.outcome, "repairs_used": state.repairs_used,
             "generation_call_count": 1, "post_generation_review_count": 0,
             "answer_hash": hashlib.sha256(answer.encode()).hexdigest(),
             "draft_hash": control_hash(draft.model_dump(mode="json"))}
    audit["audit_hash"] = control_hash(audit)
    public_audit = {"provider": model_audit["provider"], "model": model_audit["model"],
        "external_called": True, "answer_model_called": True, "provider_call": model_audit.get("provider_call"),
        "prompt_protocol_version": model_audit["protocol_version"], "prompt_protocol_hash": model_audit["prompt_protocol_hash"],
        "profile_hash": model_audit["profile_hash"], "context_package_id": package.id,
        "retrieval_trace_id": package.retrieval_trace_id, "retrieval_granularity": request.retrieval_granularity,
        "conversation_state_scope_hash": package.diagnostics_json["conversation_state_scope_hash"],
        "exact_answer_hash": audit["answer_hash"], "grounding_outcome": decision.outcome,
        "policy_update_eligible": False, "direct_answer_mode": direct_mode,
        "output_token_budget": model_audit["output_token_budget"]}
    answer_row = AnswerSession(knowledge_base_id=run.knowledge_base_id, qa_session_id=session.id,
        question=request.question, answer=answer, context_package_id=package.id,
        retrieval_trace_id=package.retrieval_trace_id, prompt_protocol_version=draft.protocol_version,
        chunk_ids_json=list(package.hit_chunk_ids_json), model_json=public_audit,
        diagnostics_json={"retrieval_control": audit, "structured_answer": draft.model_dump(mode="json"),
            "answer_units": units, "route": route, "direct_answer_mode": direct_mode,
            "source_binding_protocol_version": "answer_source_binding_v2"})
    db.add(answer_row)
    db.flush()
    bindings = persist_answer_source_bindings(db, answer_session=answer_row, package=package,
        contexts=contexts, draft=draft, evidence=manifest, unit_limit=get_settings().agent_answer_unit_limit,
        retrieval_gate=gate_row)
    citations = source_binding_citations(answer_session=answer_row, package=package, rows=bindings, retrieval_gate=gate_row)
    answer_row.citation_ids_json = [binding.id for binding in bindings]
    summary = RetrievalAnswerSummary(gate_outcome=decision.outcome, repairs_used=state.repairs_used,
        generation_call_count=1, source_binding_count=len(bindings),
        source_binding_pass_rate=1 if bindings else 0, feature_utility=features.utility, audit_hash=audit["audit_hash"])
    public_audit.update(answer_session_id=answer_row.id, returned_citation_count=len(citations),
        source_binding_pass_rate=summary.source_binding_pass_rate, retrieval_control=summary.model_dump(mode="json"))
    answer_row.model_json = public_audit
    flag_modified(answer_row,'model_json')
    row.metadata_json = {**row.metadata_json, "answer_session_id": answer_row.id, "policy_update_eligible": False}
    flag_modified(row, "metadata_json")
    db.flush()
    final_state = ag.append_session_turn(db, session, request.question, answer, run.id, citations,
        answer_session_id=answer_row.id, retrieval_trace_id=package.retrieval_trace_id,
        task_status="active", route=route, direct_answer_mode=direct_mode, commit=False)
    advance_control(db, run=row, target="completed", commit=False)
    row.final_answer, row.route = answer, route
    raise_if_source_io_cancelled()
    with qa_stage("database_commit"):
        db.commit()
    events = list(db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.sequence_index)))
    return {"run_id": run.id, "session_id": session.id, "answer": answer, "citations": citations,
        "used_chunks": contexts, "route": route, "direct_answer_mode": direct_mode,
        "trace": [ag.trace_event_to_payload(event) for event in events], "degraded_mode": ag.is_degraded_mode(),
        "context_package_id": package.id, "retrieval_trace_id": package.retrieval_trace_id,
        "retrieval_granularity": request.retrieval_granularity, "model_audit": public_audit,
        "answer_model_audit": public_audit, "conversation_state": final_state.public_payload()}


def _finish_gap(db, *, request, session, run, task, decision, features=None):
    from app.services import agent_graph as ag
    missing = {facet.id: facet.text for facet in task.requirements}
    labels = [missing[key] for key in decision.missing_facet_ids if key in missing]
    if decision.outcome in {"technical_failure", "source_incomplete"}:
        raise ValueError("retrieval_source_integrity_failed")
    if decision.outcome == "scope_ambiguous":
        answer = "需要进一步明确问题涉及的对象或资料范围：" + "、".join(labels or [task.question]) + "。"
    elif 'filtered_source_scope_empty' in decision.reason_codes:
        answer = '本次过滤范围没有可用的已索引材料。可以调整文档、页码或内容类型范围后再查询；这不代表整个资料库没有相关信息。'
    elif decision.outcome == 'source_unresolved':
        answer = "尚未能可靠定位你指定的文档、章节或对象。请补充更准确的来源名称或位置；这不表示原文没有相关内容。"
    elif decision.outcome == 'representation_incomplete':
        answer = "当前可读取的资料表示不足以核验你指定的来源范围，需要补全解析或提供可读取的内容后再回答。"
    else:
        answer = "在本次已检查的资料范围内，尚未找到足够证据支持：" + "、".join(labels or [task.question]) + "。这不代表整个原文中不存在相关信息。"
    state = RetrievalControlState.model_validate(run.metadata_json["retrieval_control"])
    gap_audit = {"protocol_version": "retrieval_gap_response_v1", "run_id": run.id,
        "task_hash": task.identity, "strategy_hash": state.strategy_hash,
        "decision": decision.model_dump(mode="json"), "repairs_used": state.repairs_used,
        "answer_hash": hashlib.sha256(answer.encode()).hexdigest(), "generation_call_count": 0,
        "post_generation_review_count": 0}
    gap_audit["audit_hash"] = control_hash(gap_audit)
    row = AnswerSession(knowledge_base_id=run.knowledge_base_id, qa_session_id=session.id,
        question=request.question, answer=answer, prompt_protocol_version="retrieval_gap_response_v1",
        model_json={"answer_model_called": False, "insufficient_evidence": True,
                    "grounding_outcome": decision.outcome, "policy_update_eligible": False},
        diagnostics_json={"agent_run_id": run.id, "gate_decision": decision.model_dump(mode="json"),
                          "retrieval_gap": gap_audit})
    db.add(row)
    db.flush()
    summary = RetrievalAnswerSummary(gate_outcome=decision.outcome, repairs_used=state.repairs_used,
        generation_call_count=0, source_binding_count=0, source_binding_pass_rate=0,
        feature_utility=features.utility if features else None, audit_hash=gap_audit["audit_hash"])
    row.model_json = {**row.model_json, "answer_session_id": row.id,
                      "retrieval_control": summary.model_dump(mode="json")}
    run.metadata_json = {**run.metadata_json, "answer_session_id": row.id, "policy_update_eligible": False}
    flag_modified(run, "metadata_json")
    final_state = ag.append_session_turn(db, session, request.question, answer, run.id, [],
        answer_session_id=row.id, task_status="waiting_user", route="layered_context_graph", commit=False)
    advance_control(db, run=run, target="insufficient", commit=False)
    run.final_answer, run.route = answer, "layered_context_graph"
    with qa_stage("database_commit"):
        db.commit()
    events = list(db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.sequence_index)))
    return {"run_id": run.id, "session_id": session.id, "answer": answer, "citations": [], "used_chunks": [],
            "route": "layered_context_graph", "trace": [ag.trace_event_to_payload(event) for event in events],
            "degraded_mode": ag.is_degraded_mode(), "retrieval_granularity": request.retrieval_granularity,
            "model_audit": row.model_json, "answer_model_audit": row.model_json,
            "conversation_state": final_state.public_payload()}


async def _execute(db, request, session, run):
    from app.services import agent_graph as ag
    settings = get_settings().model_copy(deep=True)
    runtime_hash = ag.runtime_settings_state_hash()
    await ag.run_bounded_source_io(initialize_control, db, run=run, runtime_hash=runtime_hash,
                                  repair_limit=settings.retrieval_repair_round_limit)
    async def advance(target, *, task=None, strategy=None):
        ag.ensure_agent_run_not_cancelled(db, run)
        if ag.runtime_settings_state_hash() != runtime_hash:
            raise ValueError("retrieval_runtime_identity_changed")
        state = await ag.run_bounded_source_io(advance_control, db, run=run, target=target, task=task, strategy=strategy)
        await ag.run_bounded_source_io(ag.trace, db, run.id, "retrieval_control",
            output_summary={"searching": "正在检索图谱", "packing": "正在恢复原文上下文",
                "diagnosing": "正在检查检索覆盖", "discovering": "正在查找原文表达",
                "evaluating": "正在核对材料是否覆盖问题",
                "scope_resolving": "正在确认文档和章节位置",
                "patching": "正在修正检索词面", "generating": "正在生成回答",
                "binding": "正在绑定原文来源", "reuse_check": "正在检查既有证据",
                "ready": "证据已就绪"}.get(target, "正在准备检索"),
            scores={"stage": target, "repairs_used": state.repairs_used})
        return state
    history, _ = history_summary_projection([item.model_dump() for item in request.history],
                                           max_characters=settings.agent_history_summary_max_chars)
    model = RetrievalModels(ag.ChatProvider)
    proposal, planning_audit = await model.plan(question=request.question, history_summary=history,
        timeout_seconds=settings.model_request_timeout_seconds,
        max_tokens=min(settings.chat_json_max_tokens, settings.retrieval_planning_max_tokens))
    _, perception_projection = project_task_perception(proposal, request.question)
    planning_audit = {**planning_audit, 'perception_projection': perception_projection}
    from app.services.retrieval_models import source_reference_role_audit
    source_roles=source_reference_role_audit(proposal,request.question)
    if source_roles is not None:
        planning_audit={**planning_audit,'source_reference_roles':source_roles}
    perception, task, strategy, initial_facets = compile_task_plan(proposal, question=request.question,
        knowledge_base_id=run.knowledge_base_id, conversation_scope_hash=run.metadata_json["conversation_state_scope_hash"],
        retrieval_granularity=request.retrieval_granularity)
    if task is not None:
        from app.services.task_constraints import split_response_requirements
        evidence_requirements, _, partition_audit = split_response_requirements(request.question, proposal.requirements)
        _, role_audit = normalize_source_roles(request.question, evidence_requirements)
        planning_audit = {**planning_audit, "source_role_normalization": role_audit,
                          "task_requirement_partition": partition_audit}
    await ag.run_bounded_source_io(ag.trace, db, run.id, "retrieval_control",
        output_summary="已识别问题和检索要点", scores={"stage": "planning", "model_call_count": 1})
    if perception["intent"] == "direct_answer":
        matched = ag.build_system_direct_answer(question=request.question, query_intent=perception)
        def finish_capability():
            advance_control(db, run=run, target="completed", system_capability=True, commit=False)
            return ag._execute_system_capability_direct_answer(db, request=request, session=session,
                run=run, matched=matched, conversation_state_scope_hash=run.metadata_json["conversation_state_scope_hash"])
        return await ag.run_bounded_source_io(finish_capability)
    await advance("task_ready", task=task, strategy=strategy)
    run.metadata_json = {**run.metadata_json,'evidence_sufficiency_protocol':SUFFICIENCY_PROTOCOL,
        'source_addressed_assessment_protocol':SOURCE_ADDRESSED_PROTOCOL,
        'source_filter_protocol':'active_source_filters_v2','retrieval_planning_audit':planning_audit}
    flag_modified(run,'metadata_json')
    db.commit()
    # layered_search admits the complete active graph before consuming it.
    # Only history reuse needs a separate admission because it skips search.
    try:
        corpus = await ag.run_bounded_source_io(RetrievalCorpus.load,db,knowledge_base_id=run.knowledge_base_id,filters=request.filters)
    except EmptyFilteredRetrievalScope as exc:
        from app.retrieval_control_contracts import RetrievalGateDecision
        scope = {'protocol_version':'active_source_filters_v2','filters':request.filters.model_dump(mode='json'),
            'checked_source_count':exc.checked_source_count,'eligible_source_count':0,'corpus_absence_proven':False}
        run.metadata_json = {**run.metadata_json,'empty_filtered_scope':scope}
        flag_modified(run,'metadata_json')
        decision = RetrievalGateDecision(outcome='scoped_not_found',missing_facet_ids=tuple(item.id for item in task.requirements),
            reason_codes=('filtered_source_scope_empty',),feature_hash=control_hash(scope),threshold_hash=control_hash({'no_path_evaluation':True}))
        return await ag.run_bounded_source_io(_finish_gap,db,request=request,session=session,run=run,task=task,decision=decision)
    embedding = EmbeddingProvider().for_embedding_identity(embedding_model=corpus.target.schema.embedding_model,
                                                           embedding_dimensions=corpus.target.schema.embedding_dimension)
    async with asyncio.timeout(settings.model_request_timeout_seconds):
        vectors, memo, vector_batch_audit = await prepare_query_vectors(task=task, strategy=strategy,
            initial_facets=initial_facets, target=corpus.target, provider=embedding)
    run.metadata_json = {**run.metadata_json, "retrieval_measurement_vectors": vectors,
                         "retrieval_planning_audit": planning_audit,
                         "retrieval_query_vector_batch": vector_batch_audit.model_dump(mode="json")}
    flag_modified(run, "metadata_json")
    db.commit()
    envelope = agent_operating_envelope()
    scope_index, scope_targets = None, ()
    effective_filters=request.filters
    if any(facet.source_scope is not None for facet in task.requirements):
        from app.services.evidence_scope import StructureScopeIndex, scope_target_plan
        from app.services.source_location import build_location_request,complete_location_call
        with qa_stage('source_scope_resolution'):
            scope_index = await ag.run_bounded_source_io(StructureScopeIndex.load, db, corpus=corpus, task=task)
            scores=await ag.run_bounded_source_io(corpus.cosine_scores,vectors)
            if any(binding.reason in {'no_verified_match','ambiguous'} for facet in scope_index.bind(task) for binding in facet.bindings):
                location_request=await ag.run_bounded_source_io(build_location_request,index=scope_index,task=task,
                    query_scores=scores[0],facet_scores={facet.id:scores[ordinal+1] for ordinal,facet in enumerate(task.requirements)})
                if location_request is not None:
                    await advance('scope_resolving')
                    timeout=min(settings.model_request_timeout_seconds,settings.retrieval_repair_timeout_seconds)
                    tokens=min(settings.chat_json_max_tokens,settings.retrieval_repair_max_tokens)
                    location_row=await ag.run_bounded_source_io(_save_location_request,db,run=run,task=task,
                        request=location_request,timeout_seconds=timeout,max_tokens=tokens)
                    selection,model_audit=await model.locate_sources(task=task,request=location_request,timeout_seconds=timeout,max_tokens=tokens)
                    ag.ensure_agent_run_not_cancelled(db,run)
                    await ag.run_bounded_source_io(complete_location_call,db,row=location_row,task=task,index=scope_index,
                        output=selection.model_dump(mode='json',by_alias=True),model_audit=model_audit)
                    await advance('task_ready')
            scope_targets, scope_plan = await ag.run_bounded_source_io(scope_target_plan,
                index=scope_index, task=task, token_budget=envelope['context_package_token_budget'],
                affinities={facet.id:{source.chunk_id:float(scores[ordinal+1,pos]) for pos,source in enumerate(corpus.sources)}
                    for ordinal,facet in enumerate(task.requirements)},
                target_limit=min(ag.TYPED_ACTION_TARGET_ID_LIMIT, envelope['agent_chunk_initial_budget'],
                    envelope['agent_chunk_top_k'], ag.resolve_result_top_k(request.top_k)))
        db.add(AgentObservation(run_id=run.id, observation_type='retrieval_scope_targets',
            verdict=scope_plan['status'], observation_json=scope_plan))
        db.commit()
        from app.services.evidence_scope import resolved_scope_filters
        effective_filters,filter_audit=await ag.run_bounded_source_io(resolved_scope_filters,task=task,index=scope_index,filters=request.filters)
        if filter_audit is not None:
            run.metadata_json={**run.metadata_json,'resolved_scope_filter':filter_audit}
            flag_modified(run,'metadata_json')
            db.commit()
    permitted_ids={source.chunk_id for source in corpus.sources
        if not effective_filters.document_ids or source.document_id in effective_filters.document_ids}
    thresholds = GateThresholds(coverage=settings.retrieval_gate_coverage_threshold,
        path_quality=settings.retrieval_gate_path_threshold, calibration_id="bounded_source_proxy_seed_v1")
    run.metadata_json = {**run.metadata_json, 'retrieval_gate_thresholds':thresholds.model_dump(mode='json')}
    flag_modified(run, 'metadata_json')
    db.commit()
    policy = await ag.run_bounded_source_io(read_lexical_policy, db, run.knowledge_base_id)
    candidate = await ag.run_bounded_source_io(ag._latest_verified_context_reuse_candidate, db,
        knowledge_base_id=run.knowledge_base_id,
        conversation_planner_context=run.metadata_json.get("conversation_state_planner_context") or {})
    if candidate and not {item['chunk_id'] for item in candidate['context_package'].package_json['chunks']}.issubset(permitted_ids):
        run.metadata_json = {**run.metadata_json,'retrieval_reuse_scope':{
            'protocol_version':'active_source_filters_v2','decision':'retrieval_required',
            'reason':'historical_package_outside_current_source_scope','source_scope_hash':corpus.scope_hash}}
        flag_modified(run,'metadata_json')
        db.commit()
        candidate = None
    reused = False
    features = baseline = None
    package = gate_row = decision = source_audit = None
    operations = ()
    previous_package = None
    preserve_ids = []
    packing_attempts = set()
    source_use_audit = {}
    sufficiency_audit = path_decision = None
    started = time.perf_counter()
    if candidate:
        await advance("reuse_check")
        package = candidate["context_package"]
        source_audit = await ag.run_bounded_source_io(_source_audit, db, package)
        features, _, facet_scores, replay_input = await ag.run_bounded_source_io(build_feature_snapshot, db,
            task=task, strategy=strategy, corpus=corpus, vectors=vectors, package=package,
            trace=candidate["retrieval_trace"], source_audit=source_audit, include_panels=False, source_use_audit=source_use_audit,
            scope_index=scope_index)
        await advance("diagnosing")
        decision = decide_retrieval_gate(task=task, features=features, thresholds=thresholds,
            source_integrity=source_audit["all_valid"], remaining_repairs=settings.retrieval_repair_round_limit)
        assessment = await ag.run_bounded_source_io(_assessment_candidate, task=task, features=features,
            decision=decision, replay_input=replay_input, package=package, source_audit=source_audit, thresholds=thresholds)
        if decision.outcome == "ready_full" or assessment is not None:
            # Coverage precheck grants no generation authority. A rejected
            # candidate enters layered_search, which performs its own full
            # admission; a successful reuse must pass that same gate here.
            await ag.run_bounded_source_io(ag.active_graph_admission_gate, db, run.knowledge_base_id)
            package,features,decision,source_audit,replay_input,sufficiency_audit,path_decision = await _assess_ready_package(
                db,run=run,task=task,strategy=strategy,package=package,features=features,decision=decision,
                source_audit=source_audit,replay_input=replay_input,source_use_audit=source_use_audit,
                corpus=corpus,vectors=vectors,scope_index=scope_index,thresholds=thresholds,
                settings=settings,model=model,advance=advance,reused=True,assessment=assessment)
            reused = decision.outcome == 'ready_full'
            gate_row = await ag.run_bounded_source_io(_save_gate, db, run=run, task=task, strategy=strategy,
                package=package, features=features, decision=decision, source_audit=source_audit, replay_input=replay_input,
                source_use_audit=source_use_audit,sufficiency=sufficiency_audit,path_decision=path_decision)
    while not reused:
        sufficiency_audit = path_decision = None
        state = await advance("searching", strategy=strategy)
        facets = routing_facets(task, strategy, initial_facets)
        target_ids = tuple(dict.fromkeys(corpus.locators[wid].chunk_id for wid in strategy.locator_ids))
        # Attested alias candidates also carry concrete source locations.
        target_ids = tuple(dict.fromkeys((*target_ids, *(
            corpus.locators[term.witness_id].chunk_id for term in strategy.terms
            if term.source == "attested" and term.witness_id in corpus.locators))))
        if strategy.revision == 0:
            target_ids = tuple(dict.fromkeys((*target_ids, *scope_targets)))
        actions, validation, controls = await ag.run_bounded_source_io(compile_retrieval_execution, db,
            task=task, strategy=strategy, envelope=envelope, top_k=ag.resolve_result_top_k(request.top_k),
            lexical_policy_hash=policy["state_hash"], chunk_targets=target_ids)
        plan, action_rows = await ag.run_bounded_source_io(persist_retrieval_execution, db, run=run,
            task=task, strategy=strategy, envelope=envelope, actions=actions, validation=validation, controls=controls)
        db.commit()
        with qa_stage("retrieval", round_index=state.repairs_used):
            result = await ag.execute_typed_retrieval_plan(db, knowledge_base_id=run.knowledge_base_id,
                query=request.question, filters=effective_filters, query_facets=facets, controls=controls,
                conversation_state_scope_hash=run.metadata_json["conversation_state_scope_hash"],
                conversation_state_audit=run.metadata_json["conversation_state"],
                policy_identity_frozen=True, frozen_policy_state_hash=None, query_embedding_request_memo=memo)
        if result is None:
            raise ValueError("retrieval_execution_returned_no_result")
        plan.retrieval_trace_id = result.trace.id
        plan.status = "executed"
        for action in action_rows:
            action.status = "executing" if action.action_type == 'build_context_package' else "completed"
            action.output_json = {"retrieval_trace_id": result.trace.id, "control_hash": controls["control_hash"]}
        await advance("packing")
        with qa_stage("packing"):
            package = result.context_package or await ag.run_bounded_source_io(build_context_package, db,
                knowledge_base_id=run.knowledge_base_id, query=request.question, trace=result.trace,
                results=result.results, token_budget=controls["context_package_token_budget"],
                restore_per_chunk_budget=controls["structure_restore_per_chunk_budget"])
            if previous_package is not None and preserve_ids:
                package, _ = await ag.run_bounded_source_io(pack_and_retain_reflection_sources, db,
                    candidate_package=package, source_package=previous_package,
                    preserve_chunk_ids=preserve_ids, token_budget=controls["context_package_token_budget"])
            source_audit = await ag.run_bounded_source_io(_source_audit, db, package)
            await ag.run_bounded_source_io(_record_materialized_package,db,run=run,
                action=next(action for action in action_rows if action.action_type=='build_context_package'),
                package=package,source_audit=source_audit)
        packing_repaired = False
        while True:
            await advance("diagnosing")
            actual_trace = await ag.run_bounded_source_io(db.get, RetrievalTrace, package.retrieval_trace_id)
            features, parameters, facet_scores, replay_input = await ag.run_bounded_source_io(build_feature_snapshot, db,
                task=task, strategy=strategy, corpus=corpus, vectors=vectors, package=package,
                trace=actual_trace, source_audit=source_audit, source_use_audit=source_use_audit, scope_index=scope_index)
            state = RetrievalControlState.model_validate(run.metadata_json["retrieval_control"])
            remaining_repairs = state.repair_limit - state.repairs_used
            decision = decide_retrieval_gate(task=task, features=features, thresholds=thresholds,
                source_integrity=source_audit["all_valid"], remaining_repairs=remaining_repairs)
            packing_plan = (plan_packing_repair(task=task, package=package, features=features,
                replay_input=replay_input, thresholds=thresholds) if remaining_repairs and source_audit["all_valid"] else None)
            if packing_plan is None and remaining_repairs and source_audit['all_valid']:
                packing_plan = await ag.run_bounded_source_io(plan_scope_packing_repair,task=task,package=package,trace=actual_trace,
                    features=features,scope_index=scope_index)
            if packing_plan and packing_plan.input_signature in packing_attempts:
                packing_plan = None
            if packing_plan:
                decision = decision.model_copy(update={"outcome": "repairable", "reason_codes": ("observed_packing_loss",),
                    "proposed_action_ids": (packing_plan.input_signature,)})
            assessment = (await ag.run_bounded_source_io(_assessment_candidate, task=task, features=features,
                decision=decision, replay_input=replay_input, package=package, source_audit=source_audit, thresholds=thresholds)
                if packing_plan is None else None)
            if packing_plan is None and (decision.outcome in READY or assessment is not None):
                package,features,decision,source_audit,replay_input,sufficiency_audit,path_decision = await _assess_ready_package(
                    db,run=run,task=task,strategy=strategy,package=package,features=features,decision=decision,
                    source_audit=source_audit,replay_input=replay_input,source_use_audit=source_use_audit,
                    corpus=corpus,vectors=vectors,scope_index=scope_index,thresholds=thresholds,
                    settings=settings,model=model,advance=advance,reused=False,assessment=assessment)
            gate_row = await ag.run_bounded_source_io(_save_gate, db, run=run, task=task, strategy=strategy,
                package=package, features=features, decision=decision, source_audit=source_audit, replay_input=replay_input,
                source_use_audit=source_use_audit,sufficiency=sufficiency_audit,path_decision=path_decision)
            await ag.run_bounded_source_io(record_retrieval_reward, db, run=run, package=package,
                features=features, baseline=baseline, attempt_index=state.repairs_used, operations=operations,
                extra_elapsed_seconds=time.perf_counter() - started, source_integrity=source_audit["all_valid"])
            db.commit()
            if packing_plan is None:
                break
            started = time.perf_counter()
            packing_attempts.add(packing_plan.input_signature)
            packing_row = AgentObservation(run_id=run.id, observation_type="retrieval_packing_repair", verdict="planned",
                observation_json=packing_plan.model_dump(mode="json"))
            db.add(packing_row)
            with qa_stage("database_commit"):
                db.commit()
            baseline, operations = features, ("restore_context",)
            await advance("restoring")
            with qa_stage("packing", round_index=state.repairs_used + 1):
                package, _ = await ag.run_bounded_source_io(restore_reflection_context, db, source_package=package,
                    target_chunk_ids=list(packing_plan.target_chunk_ids), preserve_chunk_ids=list(packing_plan.preserve_chunk_ids),
                    token_budget=controls["context_package_token_budget"],
                    restore_per_chunk_budget=0 if packing_plan.priority_chunk_ids else controls["structure_restore_per_chunk_budget"],
                    query_facets=facets, restoration_focus=list(packing_plan.focus),
                    packing_priority_chunk_ids=list(packing_plan.priority_chunk_ids))
                source_audit = await ag.run_bounded_source_io(_source_audit, db, package)
                packing_row.verdict = 'completed'
                packing_row.observation_json = {**packing_plan.model_dump(mode='json'), 'execution':{
                    'protocol_version':'retrieval_packing_execution_v1','context_package_id':package.id,
                    'retrieval_trace_id':package.retrieval_trace_id,'source_audit_hash':source_audit['provenance_session_hash'],
                    'source_integrity':source_audit['all_valid'],'additional_retrieval_count':0,'model_call_count':0}}
                db.commit()
            packing_repaired = True
            await advance("packing")
        if decision.outcome in {"ready_full", "ready_partial"}:
            if not packing_repaired and package.retrieval_trace_id == result.trace.id:
                schedule_layered_retrieval_cache_write(db, result=result, package=package)
                db.commit()
            break
        if state.repairs_used >= state.repair_limit or decision.outcome in {
                "source_incomplete", "technical_failure", "source_unresolved", "representation_incomplete", "scope_ambiguous"}:
            if state.repairs_used >= state.repair_limit and decision.outcome == "scoped_not_found":
                decision = decision.model_copy(update={"outcome": "budget_exhausted", "reason_codes": ("repair_budget_exhausted",)})
            return await ag.run_bounded_source_io(_finish_gap, db, request=request, session=session,
                                                  run=run, task=task, decision=decision, features=features)
        started = time.perf_counter()
        await advance("discovering")
        missing = tuple(decision.missing_facet_ids[:2])
        candidates = await ag.run_bounded_source_io(corpus.discover, task=task, missing_facet_ids=missing,
            facet_scores=facet_scores, excluded_chunk_ids=tuple(sorted((set(corpus.by_id)-permitted_ids)|
                {item['chunk_id'] for item in package.package_json['chunks']})))
        if not candidates:
            return await ag.run_bounded_source_io(_finish_gap, db, request=request, session=session,
                                                  run=run, task=task, decision=decision, features=features)
        decision = decide_retrieval_gate(task=task, features=features, thresholds=thresholds,
            source_integrity=source_audit["all_valid"], remaining_repairs=state.repair_limit - state.repairs_used,
            actionable_ids=tuple(candidate.id for candidate in candidates))
        if sufficiency_audit:
            from app.services.retrieval_sufficiency import EvidenceSufficiency
            path_decision = decision
            evidence = build_answer_evidence_manifest(package, context_package_to_contexts(package))
            assessment, _ = replay_assessment_guidance(sufficiency_audit, task=task, features=features,
                path_decision=path_decision, replay_input=replay_input, evidence=evidence, source_audit=source_audit)
            decision = constrain_gate(decision,task=task,
                result=EvidenceSufficiency.model_validate(sufficiency_audit['result']),
                remaining_repairs=state.repair_limit-state.repairs_used,
                actionable_ids=tuple(candidate.id for candidate in candidates),assessment=assessment)
        await ag.run_bounded_source_io(_save_gate, db, run=run, task=task, strategy=strategy,
            package=package, features=features, decision=decision, source_audit=source_audit, replay_input=replay_input,
            source_use_audit=source_use_audit,sufficiency=sufficiency_audit,path_decision=path_decision)
        await advance("patching")
        diagnosis = {"missing_facet_ids": list(missing), "source_use_rejections": source_use_audit.get('rejected_facet_count', 0),
                "source_use_reasons": source_use_audit.get('reasons', {}),
                "operation_priors": policy["operation_priors"], "terms": [
                {"term_id": term.term_id, "hits": term.observed_source_hits, "diagnosis": term.diagnosis,
                 "damage_lower": term.routing_damage.lower, "damage_upper": term.routing_damage.upper}
                for term in features.terms if term.facet_id in missing]}
        if sufficiency_audit:
            diagnosis['evidence_gap'] = repair_feedback(EvidenceSufficiency.model_validate(sufficiency_audit['result']),missing)
        repair_timeout = min(settings.model_request_timeout_seconds, settings.retrieval_repair_timeout_seconds)
        repair_tokens = min(settings.chat_json_max_tokens, settings.retrieval_repair_max_tokens)
        request_row = await ag.run_bounded_source_io(_save_repair_request, db, run=run, task=task,
            strategy=strategy, packet=lexical_repair_packet(task=task, strategy=strategy, candidates=candidates, diagnosis=diagnosis),
            witnesses=tuple(corpus.locators[wid] for wid in dict.fromkeys(candidate.witness_id for candidate in candidates)),
            timeout_seconds=repair_timeout, max_tokens=repair_tokens)
        patch, patch_audit = await model.repair(task=task, strategy=strategy, candidates=candidates,
            diagnosis=diagnosis, timeout_seconds=repair_timeout, max_tokens=repair_tokens)
        if patch_audit['input_hash'] != request_row.observation_json['input_hash']:
            raise ValueError('lexical_repair_request_input_changed')
        no_progress=False
        try:
            after = compile_lexical_patch(task=task, before=strategy, patch=patch, candidates=candidates)
        except LexicalPatchNoProgress:
            after,no_progress=None,True
        scope_preserve,retention_audit=(await ag.run_bounded_source_io(required_scope_retention,
            task=task,package=package,features=features,replay_input=replay_input,scope_index=scope_index)) if after is not None else ((),None)
        patch_record = {"protocol_version": "retrieval_patch_observation_v1", "task_hash": task.identity,
            "request_observation_id": request_row.id, "request_input_hash": patch_audit['input_hash'],
            "before_strategy_hash": strategy.identity, "after_strategy_hash": after.identity if after else None,
            "patch": patch.model_dump(mode="json"), "candidates": [item.model_dump(mode="json") for item in candidates],
            "model_audit": patch_audit, "attribution_scope": "patch", "actual_retrieval_executed": False,
            "compilation_outcome": "no_progress" if no_progress else "ready" if after else "model_stop"}
        if retention_audit is not None:
            patch_record['required_scope_retention']=retention_audit
        patch_record["audit_hash"] = control_hash(patch_record)
        db.add(AgentObservation(run_id=run.id, observation_type="retrieval_lexical_patch",
            verdict=patch.outcome, observation_json=patch_record))
        with qa_stage("database_commit"):
            db.commit()
        if after is None:
            decision = decision.model_copy(update={
                "outcome": "scope_ambiguous" if patch.outcome == "need_scope_clarification" else "scoped_not_found",
                "reason_codes": ("lexical_patch_no_new_information" if no_progress else patch.outcome,), "proposed_action_ids": ()})
            return await ag.run_bounded_source_io(_finish_gap, db, request=request, session=session,
                                                  run=run, task=task, decision=decision, features=features)
        preserve_ids = sorted({item.best_source_id.split(":", 1)[0] for item in features.facets
            if item.best_source_id and item.coverage.lower >= thresholds.coverage
            and item.path_quality.lower >= thresholds.path_quality}|set(scope_preserve))
        previous_package = package
        baseline, strategy = features, after
        operations = tuple(item.operation for item in patch.patches)
    await advance("ready")
    evidence = build_answer_evidence_manifest(package, context_package_to_contexts(package))
    _, guidance = replay_assessment_guidance(sufficiency_audit, task=task, features=features,
        path_decision=path_decision, replay_input=replay_input, evidence=evidence, source_audit=source_audit)
    replay_sufficiency(sufficiency_audit,task=task,strategy_hash=strategy.identity,evidence=evidence,source_scopes=guidance)
    await advance("generating")
    draft, model_audit = await model.generate(task=task, evidence=evidence, history_summary=history,
        missing_facets=decision.missing_facet_ids, timeout_seconds=min(settings.model_request_timeout_seconds,
        settings.retrieval_generation_timeout_seconds), max_tokens=min(settings.chat_json_max_tokens,
        settings.retrieval_generation_max_tokens), unit_limit=settings.agent_answer_unit_limit,source_scopes=guidance)
    await advance("binding")
    with qa_stage("source_binding"):
        return await ag.run_bounded_source_io(_finish_answer, db, request=request, session=session, run=run,
            task=task, strategy=strategy, package=package, gate_row=gate_row, features=features,
            decision=decision, draft=draft, model_audit=model_audit, reused=reused)


async def execute_retrieval_agent(db, request, session, run):
    from app.services import agent_graph as ag
    performance = current_qa_performance() or QAPerformance()
    response = None
    with performance.activate():
        try:
            remaining = get_settings().retrieval_total_timeout_seconds - performance.snapshot().elapsed_ms / 1000
            async with asyncio.timeout(max(0, remaining)):
                with qa_stage("request"):
                    response = await _execute(db, request, session, run)
        except BaseException as exc:
            db.rollback()
            db.refresh(run)
            persisted_control = (run.metadata_json or {}).get("retrieval_control") or {}
            if run.status == "cancelled" and persisted_control and persisted_control.get("state") not in {
                "completed", "insufficient", "failed", "cancelled"
            }:
                await ag.run_bounded_source_io(advance_control, db, run=run, target="cancelled")
            if run.status not in ag.TERMINAL_AGENT_RUN_STATUSES:
                admission_code = (exc.args[0] if isinstance(exc, asyncio.CancelledError) and exc.args
                    and type(exc.args[0]) is str
                    and exc.args[0] in {"agent_admission_unavailable", "agent_admission_lease_lost"} else None)
                target = "cancelled" if isinstance(exc, asyncio.CancelledError) and not admission_code else "failed"
                if (run.metadata_json or {}).get("retrieval_control"):
                    await ag.run_bounded_source_io(advance_control, db, run=run, target=target)
                else:
                    ag.set_run_state(db, run, target)
                code = admission_code or getattr(exc, "code", None)
                if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code):
                    code = str(exc) if type(exc) is ValueError and re.fullmatch(
                        r"(?:retrieval|feature|path|lexical|counterfactual|source_use|source_scope|scope_packing|generation_scope)_[a-z0-9_]{1,64}", str(exc)) else type(exc).__name__
                run.error_message = code
                run.metadata_json = {**run.metadata_json, "policy_update_eligible": False,
                    "technical_failure": {"stage": "agent_admission" if admission_code else getattr(exc, "stage", "retrieval_control"),
                        "code": code, "cause_type": getattr(exc, "cause_type", None),
                        "http_status": getattr(exc, "status_code", None),
                        "output_shape": getattr(exc, "provider_shape", None),
                        "model_call_count": getattr(exc, "model_call_count", 0),
                        "external_failure": getattr(exc, "external_failure", None) or external_failure_classification(exc)}}
                flag_modified(run, "metadata_json")
                db.commit()
            raise
        finally:
            summary = performance.snapshot().model_dump(mode="json")
            def persist():
                db.refresh(run)
                run.metadata_json = {**run.metadata_json, "qa_performance": summary}
                flag_modified(run, "metadata_json")
                db.commit()
            with qa_stage("audit_persistence"):
                await ag.run_bounded_source_io(persist)
        if response is not None:
            summary = performance.snapshot().model_dump(mode="json")
            for field in ("model_audit", "answer_model_audit"):
                response[field] = {**response.get(field, {}), "qa_performance": summary}
    return response
