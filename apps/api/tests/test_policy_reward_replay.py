from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def reward_factory(db_session, sample_knowledge_base):
    from app.services import cache_manager
    from app.models import (
        AgentAction,
        AgentObservation,
        AgentPlan,
        AgentRun,
        AgentTraceEvent,
        AnswerSession,
        Chunk,
        CitationVerification,
        CoarseConcept,
        CoarseConceptMembership,
        CoarseConceptState,
        ContextPackage,
        Document,
        DocumentVersion,
        KnowledgeBase,
        MidConcept,
        MidConceptState,
        PolicyState,
        RetrievalTrace,
        RewardEvent,
    )
    from app.services.agent_repair import (
        REPAIR_EXECUTOR_MECHANISMS,
        TYPED_REPAIR_PROTOCOL_VERSION,
        canonical_repair_hash,
        claim_grounding_gate,
        claim_rows,
    )
    from app.services.agent_graph import (
        AGENT_PLANNER_AUDIT_PROTOCOL_VERSION,
        AGENT_PLANNER_PROTOCOL_VERSION,
        REQUIRED_TYPED_ACTIONS,
        TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION,
        TYPED_ACTION_SCHEMA_PROTOCOL_VERSION,
        _repair_progress_for_bundle,
        compile_typed_action_execution_controls,
        validate_typed_actions,
    )
    from app.services.chunking import rough_token_count, stable_hash, text_hash
    from app.services.context_graph import (
        GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION,
        GRAY_ZONE_RULE_PROTOCOL_VERSION,
        TRAVERSAL_OBSERVATION_BUDGET_PROTOCOL_VERSION,
        agent_operating_envelope,
        agent_operating_envelope_state_hash,
        gray_zone_runtime_settings_hash,
        runtime_settings_state_hash,
    )
    from app.services.graph_state_hashes import (
        canonical_graph_hash,
        canonical_policy_state_hash,
        chunk_business_references,
    )
    from app.services.policy_reward import (
        build_policy_reward_replay,
        freeze_policy_reward_replay,
    )
    from app.services.policy import (
        POLICY_ARMS,
        POLICY_FAMILY,
        POLICY_VERSION,
        read_policy_operating_prior,
    )

    counter = 0

    def factory(*, second_database_identity: bool = False):
        nonlocal counter
        counter += 1
        if counter == 1 and not second_database_identity:
            kb = sample_knowledge_base
        else:
            kb = KnowledgeBase(
                name=f"Policy reward KB {counter}",
                description="same business facts, different row identity",
                source_root=f"{sample_knowledge_base.source_root}-reward-{counter}",
            )
            db_session.add(kb)
            db_session.flush()

        document = Document(
            knowledge_base_id=kb.id,
            title="Stable source",
            source_path="library/stable.txt",
            logical_source_slot_key="logical:library/stable.txt",
            source_slot_protocol_version="logical_source_slot_v1",
            source_type="txt",
            language="en",
            language_source="explicit_metadata",
            checksum="a" * 64,
        )
        db_session.add(document)
        db_session.flush()
        version = DocumentVersion(
            document_id=document.id,
            version=1,
            checksum="a" * 64,
            storage_path="snapshots/stable.txt",
            parse_protocol_version="parser_test_v1",
            language="en",
            language_source="explicit_metadata",
            is_active=True,
        )
        db_session.add(version)
        db_session.flush()
        chunk_text = "Alpha is supported by evidence."
        chunk = Chunk(
            knowledge_base_id=kb.id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=0,
            token_start=0,
            token_end=rough_token_count(chunk_text),
            char_start=0,
            char_end=len(chunk_text),
            text=chunk_text,
            text_hash=text_hash(chunk_text),
            section_path="Alpha",
            page_start=1,
            page_end=1,
            metadata_json={
                "chunk_schema_version": "chunk_schema_v1",
                "tokenizer_version": "symbograph_regex_tokenizer_v1",
                "chunk_size": 512,
                "chunk_overlap": 80,
            },
            state="active",
        )
        db_session.add(chunk)
        db_session.flush()
        chunk_refs = chunk_business_references(db_session, [chunk])

        mid_state = MidConceptState(
            knowledge_base_id=kb.id,
            state_hash="b" * 64,
            grounding_hash="c" * 64,
            prompt_protocol_version="mid_prompt_v1",
            state="active",
        )
        db_session.add(mid_state)
        db_session.flush()
        mid = MidConcept(
            concept_state_id=mid_state.id,
            knowledge_base_id=kb.id,
            canonical_label="Alpha detail",
            definition="The supported alpha detail.",
            summary="Alpha detail.",
            support_chunk_ids_json=[chunk.id],
            grounding_hash="d" * 64,
            state="active",
        )
        db_session.add(mid)
        db_session.flush()
        coarse_state = CoarseConceptState(
            knowledge_base_id=kb.id,
            mid_concept_state_id=mid_state.id,
            community_protocol_version="community_v1",
            state_hash="e" * 64,
            grounding_hash="f" * 64,
            prompt_protocol_version="coarse_prompt_v1",
            state="active",
        )
        db_session.add(coarse_state)
        db_session.flush()
        coarse = CoarseConcept(
            coarse_state_id=coarse_state.id,
            knowledge_base_id=kb.id,
            canonical_label="Alpha domain",
            definition="The alpha domain.",
            summary="Alpha.",
            included_mid_concept_ids_json=[mid.id],
            support_chunk_ids_json=[chunk.id],
            grounding_hash="1" * 64,
            state="active",
        )
        db_session.add(coarse)
        db_session.flush()
        db_session.add(
            CoarseConceptMembership(
                coarse_concept_id=coarse.id,
                mid_concept_id=mid.id,
                membership_score=1.0,
                role="included",
            )
        )
        db_session.flush()

        runtime_hash = runtime_settings_state_hash()
        envelope = agent_operating_envelope()
        envelope_hash = agent_operating_envelope_state_hash()
        gray_runtime_hash = gray_zone_runtime_settings_hash(envelope)
        assert envelope_hash == stable_hash(envelope)
        assert gray_runtime_hash != runtime_hash
        policy_weights = {arm: 1.0 for arm in POLICY_ARMS}
        safe_arms = list(POLICY_ARMS)
        policy_constraints = {
            "fallback_disabled": True,
            "citation_verification_required": True,
            "agent_operating_envelope": envelope,
            "runtime_settings_hash": runtime_hash,
            "planner_replacement": False,
            "gray_zone_decision_authority": False,
            "gray_zone_rule_inputs_modified": False,
            "gray_zone_model_call_count": 0,
        }
        policy_exploration = {
            "epsilon": 0.05,
            "safe_arms": safe_arms,
            "threshold_suggestions_runtime_lifecycle_accepted": False,
            "threshold_suggestions_applied": False,
            "gray_zone_decision_authority": False,
            "gray_zone_model_call_count": 0,
            "path_distance_green_threshold": envelope[
                "path_distance_green_threshold"
            ],
            "path_distance_gray_threshold": envelope[
                "path_distance_gray_threshold"
            ],
            "path_distance_hard_threshold": envelope[
                "path_distance_hard_threshold"
            ],
        }
        policy_summary = {
            "origin": "seed",
            "previous_policy_state_hash": None,
            "safe_arms": safe_arms,
            "posterior": policy_weights,
            "policy_version": POLICY_VERSION,
            "reward_history_tail": [],
            "runtime_settings_hash": runtime_hash,
            "agent_operating_envelope_hash": envelope_hash,
        }
        policy_hash = canonical_policy_state_hash(
            policy_family=POLICY_FAMILY,
            policy_version=POLICY_VERSION,
            profile_objective_hash=None,
            weights=policy_weights,
            constraints=policy_constraints,
            exploration=policy_exploration,
            reward_summary=policy_summary,
        )
        input_policy = PolicyState(
            knowledge_base_id=kb.id,
            policy_family=POLICY_FAMILY,
            policy_version=POLICY_VERSION,
            weights_json=policy_weights,
            constraints_json=policy_constraints,
            exploration_json=policy_exploration,
            reward_summary_json=policy_summary,
            state_hash=policy_hash,
        )
        db_session.add(input_policy)
        db_session.flush()
        policy_operating_prior = read_policy_operating_prior(
            db_session,
            kb.id,
            runtime_settings_hash=runtime_hash,
            agent_operating_envelope_hash=envelope_hash,
            agent_operating_envelope=envelope,
        )
        traversal_hash = "5" * 64
        edge_distance_hash = "6" * 64
        edge_projection_hash = "7" * 64
        question = "What supports alpha?"
        required_facets = {"required_facets": ["alpha"]}
        observation_budget = int(envelope["traversal_observation_budget"])
        empty_observation_budget_audit = {
            "protocol_version": TRAVERSAL_OBSERVATION_BUDGET_PROTOCOL_VERSION,
            "scope": "retrieval_traversal",
            "limit": observation_budget,
            "local_rule_evaluation_count": 0,
            "expanded_request_count": 0,
            "expanded_observation_count": 0,
            "cadence_compacted_count": 0,
            "budget_compacted_count": 0,
            "compacted_observation_count": 0,
            "hard_interrupt_count": 0,
            "budget_hit": False,
            "traversal_expanded_observation_count": 0,
            "remaining": observation_budget,
            "model_call_count": 0,
        }
        cache_components = {
            "knowledge_base_id": kb.id,
            "query": question,
            "filters": {},
            "chunk_business_scope_hash": chunk_refs.scope_hash,
            "structure_graph_hash": "8" * 64,
            "chunk_relation_graph_hash": "9" * 64,
            "rq_membership_hash": "a" * 64,
            "mid_concept_hash": "b" * 64,
            "coarse_concept_hash": "c" * 64,
            "runtime_settings_hash": runtime_hash,
            "policy_state_hash": policy_hash,
            "agent_operating_envelope_hash": canonical_graph_hash(
                "agent_operating_envelope_state_v1",
                envelope,
            ),
            "traversal_protocol_hash": traversal_hash,
            "edge_distance_protocol_hash": edge_distance_hash,
            "edge_projection_protocol_hash": edge_projection_hash,
            "query_facets_hash": stable_hash(required_facets),
            "retrieval_mode": "layered_context_graph",
            "retrieval_granularity": "coarse",
            "result_top_k": 1,
            "typed_action_allowed_relation_types": ["dense_semantic"],
        }
        trace = RetrievalTrace(
            knowledge_base_id=kb.id,
            query=question,
            filters_json={},
            retrieval_mode="layered_context_graph",
            runtime_settings_hash=runtime_hash,
            agent_operating_envelope_hash=envelope_hash,
            policy_state_hash=policy_hash,
            convergence_json={
                "gray_zone_decision_count": 0,
                "gray_zone_rule_evaluation_count": 0,
                "gray_zone_rule_stop_count": 0,
                "gray_zone_observation_compacted_count": 0,
                "red_zone_pruned_count": 0,
                "hard_stop_pruned_count": 0,
                "path_distance_partition_event_count": 0,
                "gray_zone_model_call_count": 0,
                "gray_zone_rule_protocol_version": (
                    GRAY_ZONE_RULE_PROTOCOL_VERSION
                ),
                "gray_zone_observation_cadence": int(
                    envelope["gray_zone_observation_cadence"]
                ),
                "traversal_observation_budget": observation_budget,
                "traversal_observation_expanded_count": 0,
                "traversal_observation_budget_compacted_count": 0,
                "traversal_observation_cadence_compacted_count": 0,
                "traversal_observation_hard_interrupt_count": 0,
                "traversal_observation_budget_hit": False,
                "traversal_observation_budget_audit": (
                    empty_observation_budget_audit
                ),
                "runtime_settings_hash": runtime_hash,
                "agent_operating_envelope_hash": envelope_hash,
                "traversal_protocol_hash": traversal_hash,
            },
            result_chunk_ids_json=[chunk.id],
            concept_path_json=[
                {"layer": "coarse", "ids": [coarse.id]},
                {"layer": "mid", "ids": [mid.id]},
                {"layer": "chunk", "ids": [chunk.id]},
            ],
            query_facets_json=required_facets,
            candidate_pools_json={
                "mid_by_coarse": [
                    {
                        "parent_node_id": coarse.id,
                        "candidate_ids": [mid.id],
                        "selected_ids": [mid.id],
                    }
                ],
                "chunk_by_mid": [
                    {
                        "parent_node_id": mid.id,
                        "candidate_ids": [chunk.id],
                        "selected_ids": [chunk.id],
                    }
                ],
            },
            topk_selection_json={
                "coarse": {"selected_ids": [coarse.id]},
                "mid": {"selected_ids": [mid.id]},
                "chunk": {"selected_ids": [chunk.id]},
            },
            path_labels_json=[
                {
                    "layer": "coarse",
                    "node_id": coarse.id,
                    "path": [coarse.id],
                    "covered_facets": ["alpha"],
                    "evidence_roles": ["coarse_entry"],
                    "distance_so_far": 0.1,
                    "reward_so_far": 0.0,
                },
                {
                    "layer": "mid",
                    "node_id": mid.id,
                    "path": [mid.id],
                    "covered_facets": ["alpha"],
                    "evidence_roles": ["mid_drilldown"],
                    "distance_so_far": 0.2,
                    "reward_so_far": 0.0,
                },
                {
                    "layer": "chunk",
                    "chunk_id": chunk.id,
                    "path": [chunk.id],
                    "covered_facets": ["alpha"],
                    "evidence_roles": ["chunk_recall"],
                    "distance_so_far": 0.3,
                    "reward_so_far": 0.0,
                },
            ],
            edge_distance_protocol_hash=edge_distance_hash,
            edge_projection_protocol_hash=edge_projection_hash,
            traversal_protocol_hash=traversal_hash,
            diagnostics_json={
                "retrieval_granularity": "coarse",
                "cache_key": stable_hash(cache_components),
                "cache_key_components": cache_components,
                "runtime_settings_hash": runtime_hash,
                "gray_zone_runtime_settings_identity_protocol_version": (
                    GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION
                ),
                "gray_zone_runtime_settings_hash": gray_runtime_hash,
                "agent_operating_envelope_hash": envelope_hash,
                "agent_operating_envelope": envelope,
                "effective_traversal_protocol_hash": traversal_hash,
                "policy_state_hash": policy_hash,
                "policy_operating_prior": policy_operating_prior,
            },
        )
        db_session.add(trace)
        db_session.flush()

        package = ContextPackage(
            knowledge_base_id=kb.id,
            retrieval_trace_id=trace.id,
            query=question,
            hit_chunk_ids_json=[chunk.id],
            restored_chunk_ids_json=[],
            bridge_chunk_ids_json=[],
            concept_path_json=deepcopy(trace.concept_path_json),
            package_json={"chunks": []},
            graph_path_ids_json=[],
            why_selected_json={},
            covered_facets_json=["alpha"],
            token_budget=100,
            token_count=rough_token_count(chunk_text),
            runtime_settings_hash=runtime_hash,
            citation_spans_json=[],
            diagnostics_json={},
        )
        db_session.add(package)
        db_session.flush()
        raw_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        source_span = {
            "document_version_id": version.id,
            "chunk_id": chunk.id,
            "source_path": document.source_path,
            "logical_source_path": document.source_path,
            "source_snapshot_verification": {
                "protocol_version": "sha256_content_addressed_readonly_source_snapshot_v2",
                "checksum": version.checksum,
                "verified": True,
                "size_bytes": len(chunk_text.encode("utf-8")),
            },
            "chunk_text_hash_protocol_version": "chunk_text_sha256_normalized_v1",
            "chunk_text_hash": chunk.text_hash,
            "raw_span_text_hash_protocol_version": "raw_chunk_span_utf8_sha256_v1",
            "raw_span_text_hash": raw_hash,
            "char_span": [0, len(chunk_text)],
            "raw_chunk_char_span": [0, len(chunk_text)],
            "page_range": [1, 1],
            "section_path": "Alpha",
            "structure_path": "Alpha",
            "structure_node_ids": [],
            "bbox": {},
            "context_package_id": package.id,
            "retrieval_trace_id": trace.id,
            "source_checksum": version.checksum,
        }
        package_item = {
            "chunk_id": chunk.id,
            "document_id": document.id,
            "document_version_id": version.id,
            "document_title": document.title,
            "context_package_id": package.id,
            "source_path": document.source_path,
            "logical_source_path": document.source_path,
            "content": chunk_text,
            "content_clipped": False,
            "content_token_count": rough_token_count(chunk_text),
            "raw_chunk_char_span": [0, len(chunk_text)],
            "chunk_text_hash_protocol_version": "chunk_text_sha256_normalized_v1",
            "chunk_text_hash": chunk.text_hash,
            "raw_span_text_hash_protocol_version": "raw_chunk_span_utf8_sha256_v1",
            "raw_span_text_hash": raw_hash,
            "section_path": "Alpha",
            "structure_path": "Alpha",
            "structure_node_ids": [],
            "structure_closure": {},
            "char_span": [0, len(chunk_text)],
            "source_span": deepcopy(source_span),
            "role": "hit",
        }
        package.package_json = {"chunks": [package_item]}
        package.citation_spans_json = [
            {
                **deepcopy(source_span),
                "document_id": document.id,
                "document_title": document.title,
                "source_path": document.source_path,
                "logical_source_path": document.source_path,
                "section_path": "Alpha",
                "structure_path": "Alpha",
                "structure_node_ids": [],
                "structure_closure": {},
            }
        ]
        package.why_selected_json = {
            chunk.id: {"covered_facets": ["alpha"]}
        }
        package.dedupe_keys_json = [f"{chunk.id}:[0,{len(chunk_text)}]"]
        package.diagnostics_json = {
            "token_budget_audit": {
                "token_budget": 100,
                "token_count": rough_token_count(chunk_text),
                "within_budget": True,
            }
        }
        db_session.flush()

        answer_text = "Alpha is supported."
        run = AgentRun(
            knowledge_base_id=kb.id,
            question=question,
            status="completed",
            route="layered_context_graph",
            current_node=None,
            final_answer=answer_text,
            metadata_json={
                "policy_operating_prior": policy_operating_prior,
                "retrieval_granularity": "coarse",
            },
        )
        db_session.add(run)
        db_session.flush()
        proposed_actions = [
            {
                "action_type": action_type,
                "target_ids": [],
                "reason": "Exercise the persisted four-layer Agent contract.",
                "budget_request": {},
                "expected_evidence": {
                    "source": "context_graph",
                    "requires_chunk_spans": True,
                },
                "stop_condition": {"required_action_complete": True},
            }
            for action_type in REQUIRED_TYPED_ACTIONS
        ]
        typed_actions, validation = validate_typed_actions(
            deepcopy(proposed_actions),
            envelope,
            db=db_session,
            knowledge_base_id=kb.id,
            retrieval_granularity="coarse",
        )
        validation = {
            **validation,
            "plan_index": 0,
            "retrieval_granularity_locked": "coarse",
            "unsupported_retrieval_granularity_rewrites_rejected": True,
        }
        assert validation["valid"] is True
        controls = compile_typed_action_execution_controls(
            typed_actions,
            envelope,
            requested_result_top_k=1,
            retrieval_granularity="coarse",
            validation_diagnostics=validation,
        )
        trace.diagnostics_json = {
            **trace.diagnostics_json,
            "typed_action_executor_protocol_version": (
                TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION
            ),
            "typed_action_schema_protocol_version": (
                TYPED_ACTION_SCHEMA_PROTOCOL_VERSION
            ),
            "typed_action_control_hash": controls["control_hash"],
        }
        plan = AgentPlan(
            run_id=run.id,
            knowledge_base_id=kb.id,
            retrieval_trace_id=trace.id,
            plan_index=0,
            planner_model_json={
                "planner_protocol": AGENT_PLANNER_PROTOCOL_VERSION,
                "typed_action_schema_protocol": (
                    TYPED_ACTION_SCHEMA_PROTOCOL_VERSION
                ),
                "planner_audit_protocol": (
                    AGENT_PLANNER_AUDIT_PROTOCOL_VERSION
                ),
                "provider_response_recorded": False,
                "provider_output_hash": cache_manager.strict_json_sha256(
                    {
                        "provider_output": {
                            "typed_actions": deepcopy(proposed_actions)
                        }
                    }
                ),
                "proposed_typed_actions": deepcopy(proposed_actions),
            },
            query_intent_json={"intent": "layered_context_graph"},
            envelope_json=deepcopy(envelope),
            typed_actions_json=typed_actions,
            validation_json=validation,
            diagnostics_json={
                "runtime_settings_hash": runtime_hash,
                "agent_operating_envelope_hash": envelope_hash,
                "policy_operating_prior": policy_operating_prior,
                "typed_action_executor_protocol_version": (
                    TYPED_ACTION_EXECUTOR_PROTOCOL_VERSION
                ),
                "typed_action_control_hash": controls["control_hash"],
                "execution_controls": controls,
            },
            status="validated",
        )
        db_session.add(plan)
        db_session.flush()
        trace.diagnostics_json = {
            **(trace.diagnostics_json or {}),
            "agent_plan_id": plan.id,
            "agent_plan_index": int(plan.plan_index),
        }
        accepted_by_index = {
            int(item["accepted_index"]): dict(item["validation"])
            for item in validation["accepted"]
        }
        primary_actions = []
        for index, typed_action in enumerate(typed_actions):
            action = AgentAction(
                run_id=run.id,
                plan_id=plan.id,
                action_index=index,
                action_type=typed_action["action_type"],
                target_ids_json=typed_action["target_ids"],
                reason=typed_action["reason"],
                budget_request_json=typed_action["budget_request"],
                expected_evidence_json=typed_action["expected_evidence"],
                stop_condition_json=typed_action["stop_condition"],
                validation_json={
                    **accepted_by_index[index],
                    "plan_valid": True,
                    "typed_action_schema_protocol_version": validation[
                        "typed_action_schema_protocol_version"
                    ],
                    "typed_action_schema_protocol_hash": validation[
                        "typed_action_schema_protocol_hash"
                    ],
                },
                status="completed",
                output_json={"action_type": typed_action["action_type"]},
            )
            db_session.add(action)
            primary_actions.append(action)
        db_session.flush()
        repair_actions = []
        progress = _repair_progress_for_bundle(package, {})
        verify_action = next(
            item
            for item in primary_actions
            if item.action_type == "verify_citations"
        )
        repair_claim = claim_rows(answer_text)[0]
        prior_fixture_repair_output_hashes = []
        for index, repair_type in enumerate(
            ("repair_concept_gap", "repair_missing_citation")
        ):
            remaining_before = 2 - index
            failure_card = {
                "repair_round_index": index,
                "remaining_repair_budget": remaining_before,
                "answer_hash": repair_claim["answer_hash"],
                "context_package_id": package.id,
                "retrieval_trace_id": trace.id,
                "claim_id": repair_claim["claim_id"],
                "claim_text": repair_claim["claim_text"],
                "claim_index": repair_claim["claim_index"],
                "citation_index": 0,
                "verdict": "unsupported",
                "failure_type": "concept_gap",
                "chunk_id": chunk.id,
                "source_span": deepcopy(source_span),
                "structure_closure_status": {},
                "covered_facets": ["alpha"],
                "missing_evidence_roles": ["concept_gap"],
                "prior_repair_action_output_hashes": list(
                    prior_fixture_repair_output_hashes
                ),
            }
            failure_card["failure_card_hash"] = canonical_repair_hash(
                TYPED_REPAIR_PROTOCOL_VERSION,
                failure_card,
            )
            stable_failure_source_span = {
                key: source_span.get(key)
                for key in (
                    "document_id",
                    "document_version_id",
                    "chunk_id",
                    "char_span",
                    "raw_chunk_char_span",
                    "page_range",
                    "section_path",
                    "structure_node_ids",
                    "bbox",
                    "source_checksum",
                    "chunk_text_hash",
                    "raw_span_text_hash",
                    "raw_span_text_hash_protocol_version",
                )
                if source_span.get(key) is not None
            }
            failure_card["semantic_failure_hash"] = canonical_repair_hash(
                TYPED_REPAIR_PROTOCOL_VERSION,
                {
                    "answer_hash": failure_card["answer_hash"],
                    "claim_id": failure_card["claim_id"],
                    "claim_text": failure_card["claim_text"],
                    "verdict": failure_card["verdict"],
                    "failure_type": failure_card["failure_type"],
                    "chunk_id": failure_card["chunk_id"],
                    "source_span": stable_failure_source_span,
                    "structure_closure_status": failure_card[
                        "structure_closure_status"
                    ],
                    "covered_facets": failure_card["covered_facets"],
                    "missing_evidence_roles": failure_card[
                        "missing_evidence_roles"
                    ],
                },
            )
            failure_set_hash = canonical_repair_hash(
                TYPED_REPAIR_PROTOCOL_VERSION,
                [failure_card["semantic_failure_hash"]],
            )
            input_hash = canonical_repair_hash(
                TYPED_REPAIR_PROTOCOL_VERSION,
                {
                    "action_type": repair_type,
                    "failure_set_hash": failure_set_hash,
                    "failure_card_hashes": [
                        failure_card["semantic_failure_hash"]
                    ],
                },
            )
            executor_mechanism = REPAIR_EXECUTOR_MECHANISMS[repair_type]
            canonical_target_refs = {
                "claim_ids": [repair_claim["claim_id"]],
                "source_chunk_ids": [chunk.id],
                "source_context_package_id": package.id,
                "source_retrieval_trace_id": trace.id,
                "mid_concept_ids": [mid.id],
            }
            canonical_target_refs["target_refs_hash"] = stable_hash(
                canonical_target_refs
            )
            raw_repair_action = {
                "action_type": repair_type,
                "target_ids": (
                    [mid.id]
                    if repair_type == "repair_concept_gap"
                    else [chunk.id]
                ),
                "reason": "Exercise a persisted typed repair round.",
                "budget_request": {"repair_round_budget": 1},
                "expected_evidence": {
                    "protocol_version": TYPED_REPAIR_PROTOCOL_VERSION,
                    "executor_mechanism": executor_mechanism,
                    "failure_types": ["concept_gap"],
                    "failure_card_hashes": [
                        failure_card["failure_card_hash"]
                    ],
                    "action_input_hash": input_hash,
                    "canonical_target_refs": canonical_target_refs,
                },
                "stop_condition": {
                    "all_claims_supported": True,
                    "no_semantic_progress": True,
                },
            }
            normalized_repairs, repair_validation = validate_typed_actions(
                [deepcopy(raw_repair_action)],
                envelope,
                db=db_session,
                knowledge_base_id=kb.id,
                require_required_actions=False,
                retrieval_granularity="coarse",
            )
            assert repair_validation["valid"] is True
            normalized_repair = normalized_repairs[0]
            repair_witness = repair_validation["accepted"][0]["validation"]
            directive_hash = stable_hash(
                {"fixture_directive": repair_type, "round_index": index}
            )
            after_answer_hash = repair_claim["answer_hash"]
            after_gate_hash = stable_hash(
                {"fixture_gate": repair_type, "round_index": index}
            )
            output_hash = stable_hash(
                {
                    "repair_protocol_version": TYPED_REPAIR_PROTOCOL_VERSION,
                    "action_input_hash": input_hash,
                    "directive_hash": directive_hash,
                    "before_progress_hash": progress["progress_hash"],
                    "after_progress_hash": progress["progress_hash"],
                    "after_answer_hash": after_answer_hash,
                    "after_gate_hash": after_gate_hash,
                }
            )
            fixture_validated_targets = {
                "action_target_ids": list(normalized_repair["target_ids"]),
                "canonical_target_refs": deepcopy(canonical_target_refs),
                "supported_source_chunk_ids": [chunk.id],
                "carry_forward_supported_chunk_ids": [],
                "bridge_seed_chunk_ids": [],
                "excluded_mid_ids": [],
                "excluded_result_chunk_ids": [],
            }
            action = AgentAction(
                run_id=run.id,
                plan_id=plan.id,
                parent_action_id=verify_action.id,
                action_index=len(primary_actions) + index,
                action_type=normalized_repair["action_type"],
                target_ids_json=normalized_repair["target_ids"],
                reason=normalized_repair["reason"],
                budget_request_json=normalized_repair["budget_request"],
                expected_evidence_json=normalized_repair[
                    "expected_evidence"
                ],
                stop_condition_json=normalized_repair["stop_condition"],
                validation_json={
                    **repair_witness,
                    "typed_action_schema_protocol_version": (
                        repair_validation[
                            "typed_action_schema_protocol_version"
                        ]
                    ),
                    "typed_action_schema_protocol_hash": repair_validation[
                        "typed_action_schema_protocol_hash"
                    ],
                    "repair_protocol_version": TYPED_REPAIR_PROTOCOL_VERSION,
                    "repair_budget_checked": True,
                    "repair_round_index": index,
                    "remaining_repair_budget_before": 2 - index,
                    "action_input_hash": input_hash,
                    "repair_directive_validator_protocol_version": (
                        "typed_repair_directive_validator_v1"
                    ),
                    "repair_directive_validator_result": "accepted",
                    "repair_directive_hash": directive_hash,
                    "validated_directive_hash": stable_hash(
                        {"validated_fixture_directive": directive_hash}
                    ),
                    "validated_targets": deepcopy(fixture_validated_targets),
                },
                diagnostics_json={
                    "failure_cards": [deepcopy(failure_card)],
                    "before_answer_hash": after_answer_hash,
                    "before_gate_hash": after_gate_hash,
                    "after_answer_hash": after_answer_hash,
                    "after_gate_hash": after_gate_hash,
                },
                status="no_progress",
            )
            db_session.add(action)
            db_session.flush()
            payload = {
                "protocol_version": TYPED_REPAIR_PROTOCOL_VERSION,
                "repair_round_index": index,
                "remaining_repair_budget_before": 2 - index,
                "remaining_repair_budget_after": 1 - index,
                "action_type": action.action_type,
                "executor_mechanism": executor_mechanism,
                "action_input_hash": input_hash,
                "action_output_hash": output_hash,
                "failure_card_hashes": [
                    failure_card["failure_card_hash"]
                ],
                "before_failure_types": ["concept_gap"],
                "after_failure_types": ["concept_gap"],
                "before_context_package_id": package.id,
                "repaired_context_package_id": package.id,
                "before_retrieval_trace_id": trace.id,
                "repaired_retrieval_trace_id": trace.id,
                "before_progress": deepcopy(progress),
                "after_progress": deepcopy(progress),
                "before_progress_hash": progress["progress_hash"],
                "after_progress_hash": progress["progress_hash"],
                "made_semantic_progress": False,
                "convergence_reason": "no_progress_try_alternate_direction",
                "retrieval_granularity": "coarse",
                "result_top_k": 1,
                "global_top_k_increased": False,
                "gray_zone_model_call_count": 0,
                "gray_zone_decision_authority": "deterministic_executor_only",
                "validated_targets": deepcopy(fixture_validated_targets),
            }
            action.output_json = deepcopy(payload)
            observation = AgentObservation(
                run_id=run.id,
                action_id=action.id,
                observation_type="typed_repair_round",
                observation_json=deepcopy(payload),
                evidence_chunk_ids_json=[chunk.id],
                verdict="no_progress",
            )
            db_session.add(observation)
            repair_actions.append((action, observation))
            prior_fixture_repair_output_hashes.append(output_hash)
        identical_trace_timestamp = datetime(2026, 1, 1, 0, 0, 0)
        db_session.add_all(
            [
                AgentTraceEvent(
                    run_id=run.id,
                    sequence_index=0,
                    node="agent_planner",
                    status="completed",
                    duration_ms=100,
                    created_at=identical_trace_timestamp,
                ),
                AgentTraceEvent(
                    run_id=run.id,
                    sequence_index=1,
                    node="citation_verification",
                    status="completed",
                    duration_ms=200,
                    created_at=identical_trace_timestamp,
                ),
            ]
        )
        db_session.flush()

        answer = AnswerSession(
            knowledge_base_id=kb.id,
            retrieval_trace_id=trace.id,
            context_package_id=package.id,
            question=question,
            answer=answer_text,
            chunk_ids_json=[chunk.id],
            diagnostics_json={},
        )
        db_session.add(answer)
        db_session.flush()
        claim = claim_rows(answer_text)[0]
        citation = CitationVerification(
            knowledge_base_id=kb.id,
            answer_session_id=answer.id,
            retrieval_trace_id=trace.id,
            context_package_id=package.id,
            chunk_id=chunk.id,
            claim_text=claim["claim_text"],
            source_span_json=deepcopy(source_span),
            verdict="supported",
            confidence=1.0,
            diagnostics_json={
                "failure_type": "none",
                "claim_id": claim["claim_id"],
                "claim_index": claim["claim_index"],
                "answer_hash": claim["answer_hash"],
                "citation_provenance_valid": True,
                "citation_provenance_persistence_gate_passed": True,
                "authoritative_chunk_link_persisted": True,
            },
        )
        db_session.add(citation)
        db_session.flush()
        citation.source_span_json = {
            **citation.source_span_json,
            "verification_id": citation.id,
        }
        answer.citation_ids_json = [citation.id]
        verification_result = {
            "citation_index": 1,
            "claim_id": claim["claim_id"],
            "claim_index": claim["claim_index"],
            "claim_text": claim["claim_text"],
            "answer_hash": claim["answer_hash"],
            "chunk_id": chunk.id,
            "source_span": deepcopy(citation.source_span_json),
            "verdict": "supported",
            "confidence": 1.0,
            "failure_type": "none",
            "diagnostics": deepcopy(citation.diagnostics_json),
        }
        gate = claim_grounding_gate(
            answer_text,
            [verification_result],
            require_persistence_replay=True,
        )
        evidence_gap = {
            "original_claim_count": 1,
            "original_supported_claim_count": 1,
            "original_unsupported_claim_count": 0,
            "repair_rounds_used": 2,
        }
        acceptance = {
            "accepted": True,
            "claim_grounding_rejected": False,
            "prompt_grounding_rejected": False,
            "exact_prompt_audit_verified": True,
        }
        answer.diagnostics_json = {
            "claim_grounded_gate": gate,
            "evidence_gap": evidence_gap,
            "grounding_outcome": "grounded_answer",
            "answer_acceptance_gate": acceptance,
        }
        reward = RewardEvent(
            knowledge_base_id=kb.id,
            retrieval_trace_id=trace.id,
            answer_session_id=answer.id,
            chunk_ids_json=[chunk.id],
            context_json={
                "context_package_id": package.id,
                "question_length": len(question),
                "context_token_count": package.token_count,
                "agent_run_id": run.id,
            },
            action_json={
                "route": "layered_context_graph",
                "policy_operating_prior": policy_operating_prior,
            },
            reward_json={"answer_acceptance_gate_pass": 1.0},
            propensity=0.5,
            diagnostics_json={
                "claim_grounded_gate": gate,
                "evidence_gap": evidence_gap,
                "grounding_outcome": "grounded_answer",
                "answer_acceptance_gate": acceptance,
                "policy_operating_prior": policy_operating_prior,
            },
        )
        db_session.add(reward)
        db_session.flush()
        # The replay cutoff must be strictly after every referenced audit row.
        # Explicit fixture ordering avoids wall-clock adjustments during long
        # full-suite runs from making an otherwise valid replay flaky.
        referenced_rows = [package, trace, run, plan, answer, citation]
        referenced_rows.extend(
            row
            for action_and_observation in repair_actions
            for row in action_and_observation
        )
        reward.created_at = max(
            row.created_at
            for row in referenced_rows
            if row.created_at is not None
        ) + timedelta(microseconds=1)
        db_session.flush()
        replay = build_policy_reward_replay(db_session, reward)
        freeze_policy_reward_replay(reward, replay)
        db_session.commit()
        return {
            "knowledge_base": kb,
            "document": document,
            "version": version,
            "chunk": chunk,
            "coarse": coarse,
            "mid": mid,
            "trace": trace,
            "package": package,
            "run": run,
            "plan": plan,
            "repair_actions": repair_actions,
            "answer": answer,
            "citation": citation,
            "reward": reward,
            "replay": replay,
        }

    return factory


def test_reward_replay_uses_frozen_context_package_document_identity(
    db_session,
    reward_factory,
) -> None:
    from app.services.policy_reward import replay_policy_reward_event

    bundle = reward_factory()
    original = replay_policy_reward_event(db_session, bundle["reward"])

    # Source-slot migrations and selected-file reparses may update the mutable
    # Document presentation row.  Historical reward identity remains bound to
    # the immutable DocumentVersion and the ContextPackage source snapshot.
    bundle["document"].title = "migrated-source-slot-name"
    bundle["document"].source_path = "storage/source_slots/migrated.txt"
    bundle["document"].checksum = "9" * 64
    db_session.flush()

    replayed = replay_policy_reward_event(db_session, bundle["reward"])
    assert replayed["content_card"] == original["content_card"]
    assert replayed["evidence_hash"] == original["evidence_hash"]
    assert replayed["reward_fact_hash"] == original["reward_fact_hash"]


def test_reward_replay_uses_frozen_target_layers_after_chunk_retirement(
    db_session,
    reward_factory,
) -> None:
    from app.services.policy_reward import replay_policy_reward_event

    bundle = reward_factory()
    original = replay_policy_reward_event(db_session, bundle["reward"])

    # A selected-file reparse retires the old active chunk UUID.  Historical
    # typed repairs remain bound to their persisted validator witness instead
    # of querying the mutable active graph for that old UUID.
    bundle["chunk"].state = "deleted"
    db_session.flush()

    replayed = replay_policy_reward_event(db_session, bundle["reward"])
    assert replayed["content_card"] == original["content_card"]
    assert replayed["evidence_hash"] == original["evidence_hash"]
    assert replayed["reward_fact_hash"] == original["reward_fact_hash"]


def test_reward_replay_uses_trace_time_concept_admission_after_graph_retirement(
    db_session,
    reward_factory,
) -> None:
    from app.services.policy_reward import replay_policy_reward_event

    bundle = reward_factory()
    original = replay_policy_reward_event(db_session, bundle["reward"])

    # Runtime Settings/vector promotion retires the prior graph bundle.  The
    # historical trace selected these concepts while they were active, so the
    # mutable lifecycle state cannot rewrite the frozen UUID-free reward key.
    bundle["mid"].state = "inactive"
    bundle["coarse"].state = "inactive"
    db_session.flush()

    replayed = replay_policy_reward_event(db_session, bundle["reward"])
    assert replayed["content_card"] == original["content_card"]
    assert replayed["evidence_hash"] == original["evidence_hash"]
    assert replayed["reward_fact_hash"] == original["reward_fact_hash"]


def test_reward_trace_time_admission_rejects_never_promoted_rows() -> None:
    from app.services.policy_reward import (
        PolicyRewardReplayError,
        _trace_time_admitted_lifecycle_state,
    )

    assert _trace_time_admitted_lifecycle_state(
        "active", field="concept"
    ) == "active"
    assert _trace_time_admitted_lifecycle_state(
        "inactive", field="concept"
    ) == "active"
    with pytest.raises(PolicyRewardReplayError, match="trace-time active admission"):
        _trace_time_admitted_lifecycle_state("shadow", field="concept")


def test_uuid_free_reward_hash_is_stable_and_business_change_changes_hash(
    db_session,
    reward_factory,
) -> None:
    from app.services.policy_reward import (
        build_policy_reward_replay,
        replay_policy_reward_event,
    )
    from app.services.agent_graph import _repair_progress_for_bundle
    from app.services.agent_repair import (
        TYPED_REPAIR_PROTOCOL_VERSION,
        canonical_repair_hash,
    )
    from app.services.chunking import stable_hash

    first = reward_factory()
    second = reward_factory(second_database_identity=True)

    replayed_first = replay_policy_reward_event(db_session, first["reward"])
    replayed_second = replay_policy_reward_event(db_session, second["reward"])
    assert first["knowledge_base"].id != second["knowledge_base"].id
    assert first["chunk"].id != second["chunk"].id
    assert replayed_first["refs"] != replayed_second["refs"]
    def first_differences(left, right, path="content_card"):
        if type(left) is not type(right):
            return [(path, left, right)]
        if isinstance(left, dict):
            if set(left) != set(right):
                return [(f"{path}.keys", sorted(left), sorted(right))]
            rows = []
            for key in sorted(left):
                rows.extend(first_differences(left[key], right[key], f"{path}.{key}"))
                if len(rows) >= 8:
                    break
            return rows
        if isinstance(left, list):
            if len(left) != len(right):
                return [(f"{path}.length", len(left), len(right))]
            rows = []
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                rows.extend(first_differences(left_item, right_item, f"{path}[{index}]"))
                if len(rows) >= 8:
                    break
            return rows
        return [] if left == right else [(path, left, right)]

    differences = first_differences(
        replayed_first["content_card"],
        replayed_second["content_card"],
    )
    assert not differences, differences
    assert replayed_first["evidence_hash"] == replayed_second["evidence_hash"]
    assert replayed_first["reward_fact_hash"] == replayed_second["reward_fact_hash"]
    serialized_card = json.dumps(
        replayed_first["content_card"],
        ensure_ascii=False,
        sort_keys=True,
    )
    for database_id in (
        first["knowledge_base"].id,
        first["document"].id,
        first["version"].id,
        first["chunk"].id,
        first["coarse"].id,
        first["mid"].id,
        first["trace"].id,
        first["package"].id,
        first["run"].id,
        first["plan"].id,
        first["answer"].id,
        first["citation"].id,
        first["reward"].id,
    ):
        assert database_id not in serialized_card

    package = second["package"]
    package.covered_facets_json = ["beta"]
    why_selected = deepcopy(package.why_selected_json)
    why_selected[second["chunk"].id]["covered_facets"] = ["beta"]
    package.why_selected_json = why_selected
    updated_progress = _repair_progress_for_bundle(package, {})
    mutated_prior_repair_output_hashes = []
    for action, observation in second["repair_actions"]:
        diagnostics = deepcopy(action.diagnostics_json)
        failure_card = deepcopy(diagnostics["failure_cards"][0])
        failure_card["prior_repair_action_output_hashes"] = list(
            mutated_prior_repair_output_hashes
        )
        failure_card["failure_card_hash"] = canonical_repair_hash(
            TYPED_REPAIR_PROTOCOL_VERSION,
            {
                key: value
                for key, value in failure_card.items()
                if key not in {"failure_card_hash", "semantic_failure_hash"}
            },
        )
        diagnostics["failure_cards"] = [failure_card]
        action.diagnostics_json = diagnostics
        expected_evidence = deepcopy(action.expected_evidence_json)
        expected_evidence["failure_card_hashes"] = [
            failure_card["failure_card_hash"]
        ]
        action.expected_evidence_json = expected_evidence
        payload = deepcopy(observation.observation_json)
        payload["failure_card_hashes"] = [
            failure_card["failure_card_hash"]
        ]
        payload["before_progress"] = deepcopy(updated_progress)
        payload["after_progress"] = deepcopy(updated_progress)
        payload["before_progress_hash"] = updated_progress["progress_hash"]
        payload["after_progress_hash"] = updated_progress["progress_hash"]
        payload["action_output_hash"] = stable_hash(
            {
                "repair_protocol_version": payload["protocol_version"],
                "action_input_hash": payload["action_input_hash"],
                "directive_hash": action.validation_json[
                    "repair_directive_hash"
                ],
                "before_progress_hash": updated_progress["progress_hash"],
                "after_progress_hash": updated_progress["progress_hash"],
                "after_answer_hash": action.diagnostics_json[
                    "after_answer_hash"
                ],
                "after_gate_hash": action.diagnostics_json["after_gate_hash"],
            }
        )
        action.output_json = deepcopy(payload)
        observation.observation_json = deepcopy(payload)
        mutated_prior_repair_output_hashes.append(
            payload["action_output_hash"]
        )
    db_session.flush()
    changed = build_policy_reward_replay(db_session, second["reward"])
    assert changed["evidence_hash"] != replayed_second["evidence_hash"]
    assert changed["metrics"]["drift_rate"] == 1.0


def test_policy_reward_replays_allowlisted_historical_typed_schema_identity(
    db_session,
    reward_factory,
) -> None:
    from app.services.agent_graph import (
        HISTORICAL_TYPED_ACTION_SCHEMA_REQUIRED_ACTIONS_BY_HASH,
        REQUIRED_TYPED_ACTIONS,
    )
    from app.services.policy_reward import (
        PolicyRewardReplayError,
        build_policy_reward_replay,
    )

    fixture = reward_factory()
    plan = fixture["plan"]
    historical_version = "typed_action_schema_v3"
    historical_hash = next(
        protocol_hash
        for protocol_hash, required_actions in (
            HISTORICAL_TYPED_ACTION_SCHEMA_REQUIRED_ACTIONS_BY_HASH.items()
        )
        if tuple(required_actions) == tuple(REQUIRED_TYPED_ACTIONS)
    )
    validation = deepcopy(plan.validation_json)
    validation["typed_action_schema_protocol_version"] = historical_version
    validation["typed_action_schema_protocol_hash"] = historical_hash
    plan.validation_json = validation
    db_session.flush()

    replay = build_policy_reward_replay(db_session, fixture["reward"])
    assert replay["protocol_version"]

    validation = deepcopy(plan.validation_json)
    validation["typed_action_schema_protocol_hash"] = "f" * 64
    plan.validation_json = validation
    db_session.flush()
    with pytest.raises(PolicyRewardReplayError, match="typed-action validator replay"):
        build_policy_reward_replay(db_session, fixture["reward"])


def test_replay_rejects_forged_hash_and_synchronized_card_copy(
    db_session,
    reward_factory,
) -> None:
    from app.services.graph_state_hashes import canonical_graph_hash
    from app.services.policy_reward import (
        POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        PolicyRewardReplayError,
        replay_policy_reward_event,
    )

    fixture = reward_factory()
    reward = fixture["reward"]
    diagnostics = deepcopy(reward.diagnostics_json)
    diagnostics["policy_reward_metric_evidence_hash"] = "f" * 64
    reward.diagnostics_json = diagnostics
    with pytest.raises(PolicyRewardReplayError, match="forged hash"):
        replay_policy_reward_event(db_session, reward)

    diagnostics = deepcopy(fixture["replay"])
    stored = deepcopy(fixture["reward"].diagnostics_json)
    card = deepcopy(diagnostics["content_card"])
    card["query"]["required_facets"] = ["forged"]
    stored["policy_reward_metric_evidence_card"] = card
    stored["policy_reward_metric_evidence_hash"] = canonical_graph_hash(
        POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        card,
    )
    reward.diagnostics_json = stored
    with pytest.raises(PolicyRewardReplayError, match="stored/rebuilt"):
        replay_policy_reward_event(db_session, reward)


def test_replay_rejects_every_nonfinite_or_out_of_domain_metric(
    db_session,
    reward_factory,
) -> None:
    from app.services.policy_reward import (
        PolicyRewardReplayError,
        replay_policy_reward_event,
    )

    fixture = reward_factory()
    reward = fixture["reward"]
    original = deepcopy(reward.reward_json)
    ratio_fields = (
        "retrieval_hit",
        "context_precision",
        "context_recall",
        "concept_path_accuracy",
        "citation_pass_rate",
        "answer_groundedness",
        "answer_completeness",
        "repair_success_rate",
        "agent_typed_action_validation_pass_rate",
        "drift_rate",
        "answer_acceptance_gate_pass",
    )
    attacks = [
        *((field, value) for field in ratio_fields for value in (
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.01,
            1.01,
        )),
        ("claim_count", -1),
        ("supported_claim_count", 0.5),
        ("unsupported_claim_count", float("nan")),
        ("latency_cost", -1.0),
        ("latency_ms", float("inf")),
        ("task_token_cost", float("nan")),
    ]
    for field, value in attacks:
        forged = deepcopy(original)
        forged[field] = value
        forged["policy_reward_metrics"][field] = value
        reward.reward_json = forged
        with pytest.raises(PolicyRewardReplayError):
            replay_policy_reward_event(db_session, reward)
        reward.reward_json = deepcopy(original)


def test_replay_rejects_cross_kb_and_broken_reciprocal_links(
    db_session,
    reward_factory,
) -> None:
    from app.services.policy_reward import (
        PolicyRewardReplayError,
        replay_policy_reward_event,
    )

    first = reward_factory()
    second = reward_factory(second_database_identity=True)
    first["citation"].knowledge_base_id = second["knowledge_base"].id
    db_session.flush()
    with pytest.raises(PolicyRewardReplayError, match="crosses"):
        replay_policy_reward_event(db_session, first["reward"])

    db_session.rollback()
    first = reward_factory()
    first["answer"].context_package_id = None
    db_session.flush()
    with pytest.raises(PolicyRewardReplayError, match="reciprocal links"):
        replay_policy_reward_event(db_session, first["reward"])


@pytest.mark.parametrize(
    ("replacement_index", "message"),
    [(0, "duplicate round"), (2, "missing or non-contiguous")],
)
def test_replay_rejects_duplicate_or_missing_repair_round(
    db_session,
    reward_factory,
    replacement_index,
    message,
) -> None:
    from app.services.policy_reward import (
        PolicyRewardReplayError,
        replay_policy_reward_event,
    )

    fixture = reward_factory()
    action, observation = fixture["repair_actions"][1]
    payload = deepcopy(observation.observation_json)
    payload["repair_round_index"] = replacement_index
    observation.observation_json = payload
    action.output_json = deepcopy(payload)
    db_session.flush()
    with pytest.raises(PolicyRewardReplayError, match=message):
        replay_policy_reward_event(db_session, fixture["reward"])
