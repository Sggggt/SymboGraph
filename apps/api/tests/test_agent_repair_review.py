from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.agent_repair import (
    canonical_failure_cards,
    claim_grounding_gate,
    claim_rows,
    exact_answer_hash,
    repair_made_progress,
    repair_semantic_progress_signature,
    select_repair_direction,
)


def _supported_verification_for(claim: dict) -> dict:
    return {
        "claim_id": claim["claim_id"],
        "claim_index": claim["claim_index"],
        "claim_text": claim["claim_text"],
        "answer_hash": claim["answer_hash"],
        "citation_index": claim["claim_index"] + 1,
        "chunk_id": "chunk-1",
        "source_span": {
            "document_version_id": "version-1",
            "char_span": [0, 12],
            "raw_span_text_hash": "a" * 64,
        },
        "verdict": "supported",
        "failure_type": "none",
        "diagnostics": {
            "claim_id": claim["claim_id"],
            "claim_index": claim["claim_index"],
            "citation_provenance_valid": True,
        },
    }


def test_claim_gate_rejects_verification_bound_to_another_exact_answer() -> None:
    answer = "Alpha is supported."
    claim = claim_rows(answer)[0]
    verification = _supported_verification_for(claim)
    verification["answer_hash"] = "0" * 64
    verification["diagnostics"]["answer_hash"] = "0" * 64

    gate = claim_grounding_gate(answer, [verification])

    assert gate["supported_claim_count"] == 0
    assert gate["all_claims_supported"] is False


def test_claim_gate_fail_closes_conflicting_explicit_claim_fields() -> None:
    answer = "Alpha is supported. Beta is separate."
    claims = claim_rows(answer)
    verification = _supported_verification_for(claims[0])
    verification["claim_index"] = claims[1]["claim_index"]
    verification["claim_text"] = claims[1]["claim_text"]

    gate = claim_grounding_gate(answer, [verification])
    cards = canonical_failure_cards(
        answer=answer,
        verification_results=[verification],
        repair_round_index=0,
        remaining_repair_budget=2,
        context_package_id="package-1",
        retrieval_trace_id="trace-1",
        structure_closure_status={},
        covered_facets=[],
        missing_evidence_roles=[],
        prior_repair_action_output_hashes=[],
    )

    assert gate["supported_claim_count"] == 0
    assert gate["unbound_verification_count"] == 1
    assert cards[0]["failure_type"] == "claim_binding_invalid"


def test_failure_cards_cover_exact_claim_with_no_verification_row() -> None:
    answer = "This exact claim has no citation result."
    claim = claim_rows(answer)[0]

    cards = canonical_failure_cards(
        answer=answer,
        verification_results=[],
        repair_round_index=1,
        remaining_repair_budget=1,
        context_package_id="package-empty",
        retrieval_trace_id="trace-empty",
        structure_closure_status={},
        covered_facets=["missing"],
        missing_evidence_roles=["citation"],
        prior_repair_action_output_hashes=["a" * 64],
    )

    assert len(cards) == 1
    assert cards[0]["claim_id"] == claim["claim_id"]
    assert cards[0]["claim_text"] == claim["claim_text"]
    assert cards[0]["citation_index"] == 0
    assert cards[0]["chunk_id"] is None
    assert cards[0]["failure_type"] == "citation_missing"


def test_semantic_failure_hash_ignores_only_persistence_row_addresses() -> None:
    answer = "Alpha needs concept support."
    claim = claim_rows(answer)[0]
    first = {
        **_supported_verification_for(claim),
        "verdict": "unsupported",
        "failure_type": "concept_gap",
    }
    first["source_span"] = {
        **first["source_span"],
        "chunk_id": first["chunk_id"],
        "context_package_id": "package-1",
        "retrieval_trace_id": "trace-1",
        "verification_id": "verification-1",
    }
    second = {
        **first,
        "source_span": {
            **first["source_span"],
            "context_package_id": "package-2",
            "retrieval_trace_id": "trace-2",
            "verification_id": "verification-2",
        },
    }

    def cards_for(result: dict, package_id: str, trace_id: str) -> list[dict]:
        return canonical_failure_cards(
            answer=answer,
            verification_results=[result],
            repair_round_index=0,
            remaining_repair_budget=2,
            context_package_id=package_id,
            retrieval_trace_id=trace_id,
            structure_closure_status={"has_formula_table_caption": False},
            covered_facets=["factorization"],
            missing_evidence_roles=["concept_support"],
            prior_repair_action_output_hashes=[],
        )

    first_card = cards_for(first, "package-1", "trace-1")[0]
    second_card = cards_for(second, "package-2", "trace-2")[0]

    assert first_card["failure_card_hash"] != second_card["failure_card_hash"]
    assert first_card["semantic_failure_hash"] == second_card[
        "semantic_failure_hash"
    ]


def test_progress_rejects_supported_claim_regression_even_with_new_evidence() -> None:
    before = repair_semantic_progress_signature(
        result_chunk_ids=["chunk-1"],
        package_chunk_spans=[
            {
                "chunk_id": "chunk-1",
                "document_version_id": "version-1",
                "char_span": [0, 12],
                "raw_span_text_hash": "a" * 64,
            }
        ],
        covered_facets=["alpha"],
        evidence_roles=["definition"],
        graph_path_ids=["edge-1"],
        supported_claim_ids=["claim-supported"],
        unsupported_claim_ids=["claim-gap"],
    )
    after = repair_semantic_progress_signature(
        result_chunk_ids=["chunk-2", "chunk-3"],
        package_chunk_spans=[
            {
                "chunk_id": "chunk-2",
                "document_version_id": "version-1",
                "char_span": [13, 24],
                "raw_span_text_hash": "b" * 64,
            }
        ],
        covered_facets=["alpha", "beta"],
        evidence_roles=["definition", "bridge"],
        graph_path_ids=["edge-1", "edge-2"],
        supported_claim_ids=["claim-gap"],
        unsupported_claim_ids=["claim-supported"],
    )

    assert repair_made_progress(before, after) is False


def test_partial_answer_reward_preserves_original_claim_completeness() -> None:
    from app.services.agent_graph import reward_metrics_from_verifications

    answer = "Alpha is supported."
    claim = claim_rows(answer)[0]
    verification = _supported_verification_for(claim)
    verification["diagnostics"][
        "citation_provenance_persistence_gate_passed"
    ] = True
    package = SimpleNamespace(
        hit_chunk_ids_json=["chunk-1"],
        restored_chunk_ids_json=["chunk-1"],
        concept_path_json=[],
        package_json={"chunks": [{"chunk_id": "chunk-1"}]},
        token_count=12,
    )

    metrics = reward_metrics_from_verifications(
        package,
        [verification],
        answer,
        evidence_gap={
            "original_claim_count": 2,
            "original_supported_claim_count": 1,
        },
    )

    assert metrics["citation_pass_rate"] == 1.0
    assert metrics["answer_groundedness"] == 1.0
    assert metrics["answer_completeness"] == 0.5
    assert metrics["repair_success_rate"] == 0.5


def test_no_progress_does_not_invent_unobserved_bridge_or_structure_direction() -> None:
    answer = "A concept claim."
    claim = claim_rows(answer)[0]
    verification = {
        **_supported_verification_for(claim),
        "verdict": "unsupported",
        "failure_type": "concept_gap",
    }
    cards = canonical_failure_cards(
        answer=answer,
        verification_results=[verification],
        repair_round_index=0,
        remaining_repair_budget=4,
        context_package_id="package-1",
        retrieval_trace_id="trace-1",
        structure_closure_status={},
        covered_facets=[],
        missing_evidence_roles=["concept_support"],
        prior_repair_action_output_hashes=[],
    )
    concept = select_repair_direction(cards)
    assert concept is not None
    assert concept["action_type"] == "repair_concept_gap"
    missing = select_repair_direction(
        cards,
        attempted_input_hashes_by_action={
            "repair_concept_gap": {concept["input_hash"]},
        },
    )
    assert missing is not None
    assert missing["action_type"] == "repair_missing_citation"

    exhausted = select_repair_direction(
        cards,
        attempted_input_hashes_by_action={
            "repair_concept_gap": {concept["input_hash"]},
            "repair_missing_citation": {missing["input_hash"]},
        },
    )

    assert exhausted is None


def test_gray_support_ids_exclude_repair_planner_control_metadata() -> None:
    from app.services.context_graph import (
        _bridge_repair_seed_metadata,
        _support_ref_ids,
    )

    repair_directive = {
        "repair_action_type": "repair_bridge_gap",
        "repair_directive_hash": "b" * 64,
        "direct_context_package_injection": False,
        "bridge_support_edge_ids": ["supported-edge-1"],
    }
    refs = _bridge_repair_seed_metadata(repair_directive)

    assert _support_ref_ids(refs) == ["supported-edge-1"]
    assert refs["support_relation_edge_ids"] == ["supported-edge-1"]
    assert "bridge_support_edge_ids" not in refs
    assert _support_ref_ids(repair_directive) == []


def test_cached_repair_policy_prior_presence_absence_tamper_fails_closed() -> None:
    from app.services.agent_graph import (
        _freeze_or_validate_typed_repair_policy_prior,
    )

    frozen_prior = {
        "protocol_version": "policy_operating_prior_v1",
        "policy_state_hash": "a" * 64,
    }
    source_with_prior = SimpleNamespace(
        diagnostics_json={"policy_operating_prior": frozen_prior}
    )
    cached_without_prior = SimpleNamespace(diagnostics_json={})
    with pytest.raises(
        RuntimeError,
        match="changed the frozen Policy prior",
    ):
        _freeze_or_validate_typed_repair_policy_prior(
            None,
            source_trace=source_with_prior,
            repaired_trace=cached_without_prior,
            cache_package_reused=True,
        )

    source_without_prior = SimpleNamespace(diagnostics_json={})
    cached_with_prior = SimpleNamespace(
        diagnostics_json={"policy_operating_prior": frozen_prior}
    )
    with pytest.raises(
        RuntimeError,
        match="introduced a Policy prior",
    ):
        _freeze_or_validate_typed_repair_policy_prior(
            None,
            source_trace=source_without_prior,
            repaired_trace=cached_with_prior,
            cache_package_reused=True,
        )


def test_executor_only_repair_role_cannot_change_gray_decision_or_hash() -> None:
    from app.services.chunking import stable_hash
    from app.services.context_graph import deterministic_gray_zone_decision

    observation = {
        "current_layer": "chunk",
        "path_distance": 0.75,
        "distance_zone": "gray",
        "covered_facets_before": ["factorization"],
        "covered_facets_after": ["factorization", "formula"],
        "required_facets": ["factorization", "formula"],
        "candidate_facets": ["formula"],
        "evidence_roles_before": ["mid_drilldown_entry"],
        "evidence_roles_after": [
            "dense_semantic",
            "mid_drilldown_entry",
        ],
        "support_ids_before": ["support-0"],
        "support_ids_after": ["support-0", "support-1"],
        "support_ids_before_count": 1,
        "support_ids_after_count": 2,
        "support_ids_before_hash": stable_hash(["support-0"]),
        "support_ids_after_hash": stable_hash(
            ["support-0", "support-1"]
        ),
        "support_id_gain": True,
        "independent_path_contribution_gain": False,
        "path_contribution_key": "f" * 64,
        "support_refs": {"support_chunk_ids": ["chunk-1"]},
        "active_edge_support_gate_pass": True,
        "support_backed_to_covered_path": True,
        "validated_entry_semantic_anchor": True,
        "semantic_uncertain_edge": False,
        "crossing_rq_boundary": False,
        "bridge_or_boundary_reason": [],
        "supported_raw_span_hit": True,
        "structure_context_available": True,
        "drilldown_eligible": False,
        "edge_type": "dense_semantic",
        "rq_membership_diagnostics": {},
        "candidate_chunk_span_summary": {},
        "structure_context_status": {"available": True},
        "hard_interrupt_state": {
            "max_edge_reuse": 1,
            "edge_reuse_count": 1,
        },
        "path_distance_green_threshold": 0.5,
        "path_distance_gray_threshold": 1.0,
        "path_distance_hard_threshold": 1.5,
        "gray_zone_rule_protocol_version": (
            "deterministic_support_progress_v1"
        ),
    }
    executor_polluted = {
        **observation,
        "evidence_roles_before": [
            *observation["evidence_roles_before"],
            "support_backed_bridge_repair_seed",
        ],
        "evidence_roles_after": [
            *observation["evidence_roles_after"],
            "support_backed_bridge_repair_seed",
        ],
    }

    clean = deterministic_gray_zone_decision(observation)
    polluted = deterministic_gray_zone_decision(executor_polluted)

    assert polluted == clean
    assert polluted["input_hash"] == clean["input_hash"]
    assert polluted["decision"] == clean["decision"]
    assert polluted["matched_rule"] == clean["matched_rule"]
    assert "support_backed_bridge_repair_seed" not in json.dumps(
        polluted.get("observation") or {},
        sort_keys=True,
    )


async def _initial_package(db_session, populated_context_graph):
    from app.schemas import SearchFilters
    from app.services.context_graph import build_context_package, layered_search

    knowledge_base = populated_context_graph["knowledge_base"]
    question = "Explain Bayesian network factorization."
    initial = await layered_search(
        db_session,
        knowledge_base.id,
        question,
        SearchFilters(),
        4,
        retrieval_granularity="mid",
    )
    package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query=question,
        trace=initial.trace,
        results=initial.results,
        snapshot_verifier=initial.snapshot_verifier,
    )
    return knowledge_base, question, initial, package


def _single_claim_answer_from_package(package) -> str:
    chunks = list((package.package_json or {}).get("chunks", []))
    source = next(
        (
            item
            for item in chunks
            if "bayesian networks represent" in str(
                item.get("content") or ""
            ).lower()
        ),
        chunks[0],
    )
    content = " ".join(str(source.get("content") or "").split())
    first_sentence = content.split(".", 1)[0].strip()
    return f"{first_sentence}." if first_sentence else "Supported evidence."


def _bound_supported_verification(package, *, answer: str, question: str):
    from app.services import agent_graph

    citation = agent_graph.citation_payloads_from_package(
        package,
        retrieval_trace_id=package.retrieval_trace_id,
        answer=answer,
        question=question,
    )[0]
    verification = {
        **citation,
        "source_span": dict(citation.get("source_span") or {}),
        "verdict": "supported",
        "failure_type": "none",
        "confidence": 0.95,
        "diagnostics": {
            "claim_id": citation.get("claim_id"),
            "claim_index": citation.get("claim_index"),
            "answer_hash": citation.get("answer_hash"),
            "citation_provenance_valid": True,
            "citation_provenance_session_hash": "a" * 64,
        },
    }
    return citation, verification


def _exact_verification_bundle(
    package,
    *,
    answer: str,
    citations: list[dict],
    verifications: list[dict],
) -> dict:
    from app.services.agent_repair import (
        CLAIM_GROUNDED_GATE_PROTOCOL_VERSION,
    )
    from app.services.chunking import stable_hash

    answer_hash = exact_answer_hash(answer)
    gate = claim_grounding_gate(answer, verifications)
    binding = {
        "answer_hash": answer_hash,
        "context_package_id": package.id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "citation_indexes": [
            int(item.get("citation_index") or 0) for item in citations
        ],
        "claim_ids": [row["claim_id"] for row in gate["claims"]],
        "gate_hash": gate["gate_hash"],
    }
    return {
        "answer": answer,
        "answer_hash": answer_hash,
        "citations": citations,
        "verifications": verifications,
        "gate": gate,
        "binding": binding,
        "binding_hash": stable_hash(
            {
                "protocol_version": CLAIM_GROUNDED_GATE_PROTOCOL_VERSION,
                **binding,
            }
        ),
    }


def _gray_walk_records(db_session, retrieval_trace_id: str) -> list[dict]:
    from sqlalchemy import select

    from app.models import GraphRetrievalStep

    steps = list(
        db_session.scalars(
            select(GraphRetrievalStep)
            .where(
                GraphRetrievalStep.retrieval_trace_id
                == retrieval_trace_id
            )
            .order_by(GraphRetrievalStep.step_index)
        ).all()
    )
    return [
        {
            "layer": step.layer,
            "action": step.action,
            "entry_nodes": list((step.input_json or {}).get("entry_nodes") or []),
            "frontier_json": list(
                (step.diagnostics_json or {}).get("frontier_json") or []
            ),
            "gray_zone_path_decisions": list(
                step.gray_zone_path_decisions_json or []
            ),
        }
        for step in steps
        if step.action
        in {
            "staged_priority_queue_walk",
            "drill_down_each_coarse_or_direct_mid_entry",
            "walk_graph_frontier",
        }
    ]


@pytest.mark.asyncio
async def test_repair_replays_source_trace_gray_identity_across_runtime_drift(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import SearchFilters
    from app.services import context_graph
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.chunking import stable_hash

    knowledge_base, question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    source_envelope = dict(
        (initial.trace.diagnostics_json or {}).get(
            "agent_operating_envelope"
        )
        or {}
    )
    source_threshold_hash = context_graph.path_distance_threshold_hash(
        source_envelope
    )
    assert stable_hash(source_envelope) == (
        initial.trace.agent_operating_envelope_hash
    )

    runtime_drift = dict(context_graph.agent_operating_envelope())
    runtime_drift.update(
        {
            "path_distance_green_threshold": 0.01,
            "path_distance_gray_threshold": 0.02,
            "path_distance_hard_threshold": 0.03,
            "agent_chunk_initial_budget": max(
                1,
                int(runtime_drift["agent_chunk_initial_budget"]) + 7,
            ),
        }
    )
    assert context_graph.path_distance_threshold_hash(runtime_drift) != (
        source_threshold_hash
    )
    monkeypatch.setattr(
        context_graph,
        "agent_operating_envelope",
        lambda settings=None: dict(runtime_drift),
    )

    query_facets = dict(initial.trace.query_facets_json or {})
    directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="7" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
    )
    repaired = await context_graph.layered_search(
        db_session,
        knowledge_base.id,
        question,
        SearchFilters(),
        4,
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        conversation_state_audit=dict(
            (initial.trace.diagnostics_json or {}).get(
                "conversation_state"
            )
            or {}
        ),
        repair_directive=directive,
    )

    repaired_diagnostics = dict(repaired.trace.diagnostics_json or {})
    initial_diagnostics = dict(initial.trace.diagnostics_json or {})
    effective_directive = dict(
        repaired_diagnostics.get("repair_directive") or {}
    )
    assert repaired.trace.agent_operating_envelope_hash == (
        initial.trace.agent_operating_envelope_hash
    )
    assert repaired.trace.traversal_protocol_hash == (
        initial.trace.traversal_protocol_hash
    )
    assert repaired.trace.runtime_settings_hash == (
        initial.trace.runtime_settings_hash
    )
    assert repaired_diagnostics["agent_operating_envelope"] == (
        source_envelope
    )
    assert repaired_diagnostics[
        "gray_zone_query_facet_protocol_version"
    ] == "deterministic_gray_query_tokenizer_v1"
    assert repaired_diagnostics["gray_zone_query_facet_hash"] == (
        initial_diagnostics["gray_zone_query_facet_hash"]
    ) == context_graph.stable_hash(
        context_graph.deterministic_gray_query_facets_for_search(question)
    )
    assert repaired_diagnostics[
        "gray_zone_external_routing_packet_used"
    ] is False
    assert repaired_diagnostics[
        "gray_zone_request_scoped_budget_in_identity"
    ] is False
    assert effective_directive[
        "frozen_agent_operating_envelope_hash"
    ] == initial.trace.agent_operating_envelope_hash
    assert effective_directive["frozen_traversal_protocol_hash"] == (
        initial.trace.traversal_protocol_hash
    )
    assert effective_directive[
        "frozen_path_distance_threshold_hash"
    ] == source_threshold_hash
    assert effective_directive["runtime_threshold_drift_observed"] is True
    assert repaired.trace.convergence_json["gray_zone_model_call_count"] == 0

    gray_records = [
        decision
        for walk in _gray_walk_records(db_session, repaired.trace.id)
        for decision in walk["gray_zone_path_decisions"]
    ]
    # Sparse graphs can legitimately produce no gray candidate.  The frozen
    # trace/directive identities above still prove that runtime drift cannot
    # change the decision authority.  When gray candidates exist, every
    # persisted decision must replay the source identities exactly.
    if gray_records:
        assert {
            str(record.get("threshold_hash") or "")
            for record in gray_records
        } == {source_threshold_hash}
        assert {
            str(record.get("traversal_protocol_hash") or "")
            for record in gray_records
        } == {str(initial.trace.traversal_protocol_hash)}
        assert {
            str(record.get("agent_operating_envelope_hash") or "")
            for record in gray_records
        } == {str(initial.trace.agent_operating_envelope_hash)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("operating_envelope", "operating-envelope identity mismatch"),
        ("traversal_protocol", "traversal identity mismatch"),
    ],
)
async def test_repair_rejects_tampered_source_trace_gray_identity(
    tamper,
    expected_error,
    db_session,
    populated_context_graph,
) -> None:
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.context_graph import validate_typed_repair_directive

    knowledge_base, _question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    if tamper == "operating_envelope":
        diagnostics = dict(initial.trace.diagnostics_json or {})
        envelope = dict(diagnostics.get("agent_operating_envelope") or {})
        envelope["agent_chunk_initial_budget"] = (
            int(envelope["agent_chunk_initial_budget"]) + 1
        )
        diagnostics["agent_operating_envelope"] = envelope
        initial.trace.diagnostics_json = diagnostics
    else:
        initial.trace.traversal_protocol_hash = "0" * 64
    db_session.flush()

    query_facets = dict(initial.trace.query_facets_json or {})
    directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="8" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
    )
    with pytest.raises(ValueError, match=expected_error):
        validate_typed_repair_directive(
            db_session,
            knowledge_base_id=knowledge_base.id,
            query_facets=query_facets,
            retrieval_granularity="mid",
            conversation_state_scope_hash=(
                initial.trace.conversation_state_scope_hash
            ),
            repair_directive=directive,
        )


@pytest.mark.asyncio
async def test_supported_carry_forward_is_post_traversal_not_gray_authority(
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import SearchFilters
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.context_graph import layered_search

    knowledge_base, question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    answer = _single_claim_answer_from_package(package)
    citation, supported_verification = _bound_supported_verification(
        package,
        answer=answer,
        question=question,
    )
    source_chunk_id = str(supported_verification["chunk_id"])
    query_facets = dict(initial.trace.query_facets_json or {})
    conversation_audit = dict(
        (initial.trace.diagnostics_json or {}).get("conversation_state")
        or {}
    )
    unsupported_verification = {
        **supported_verification,
        "verdict": "unsupported",
        "failure_type": "concept_gap",
    }
    unsupported_bundle = _exact_verification_bundle(
        package,
        answer=answer,
        citations=[citation],
        verifications=[unsupported_verification],
    )
    supported_bundle = _exact_verification_bundle(
        package,
        answer=answer,
        citations=[citation],
        verifications=[supported_verification],
    )
    no_carry_directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="9" * 64,
        package=package,
        verification_results=[unsupported_verification],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        verification_bundle=unsupported_bundle,
    )
    carry_directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="9" * 64,
        package=package,
        verification_results=[supported_verification],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        verification_bundle=supported_bundle,
    )
    assert no_carry_directive["supported_source_chunk_ids"] == (
        carry_directive["supported_source_chunk_ids"]
    )
    assert no_carry_directive["carry_forward_supported_chunk_ids"] == []
    assert carry_directive["carry_forward_supported_chunk_ids"] == [
        source_chunk_id
    ]

    async def run_repair(directive):
        return await layered_search(
            db_session,
            knowledge_base.id,
            question,
            SearchFilters(),
            4,
            query_facets=query_facets,
            retrieval_granularity="mid",
            conversation_state_scope_hash=(
                initial.trace.conversation_state_scope_hash
            ),
            conversation_state_audit=conversation_audit,
            repair_directive=directive,
        )

    no_carry = await run_repair(no_carry_directive)
    with_carry = await run_repair(carry_directive)
    no_carry_walks = _gray_walk_records(db_session, no_carry.trace.id)
    carry_walks = _gray_walk_records(db_session, with_carry.trace.id)

    # Carry-forward is appended after traversal.  It may change only final
    # result retention, never entries/frontiers or gray decision identities.
    assert [
        (walk["layer"], walk["action"], walk["entry_nodes"])
        for walk in carry_walks
    ] == [
        (walk["layer"], walk["action"], walk["entry_nodes"])
        for walk in no_carry_walks
    ]
    assert [walk["frontier_json"] for walk in carry_walks] == [
        walk["frontier_json"] for walk in no_carry_walks
    ]
    assert [
        walk["gray_zone_path_decisions"] for walk in carry_walks
    ] == [
        walk["gray_zone_path_decisions"] for walk in no_carry_walks
    ]
    assert source_chunk_id not in set(
        no_carry.trace.result_chunk_ids_json or []
    )
    assert source_chunk_id in set(
        with_carry.trace.result_chunk_ids_json or []
    )
    assert len(with_carry.trace.result_chunk_ids_json or []) <= 4
    assert (
        (with_carry.trace.topk_selection_json or {})["chunk"][
            "global_top_k_increased"
        ]
        is False
    )
    carried_result = next(
        item
        for item in with_carry.results
        if str(item.get("chunk_id")) == source_chunk_id
    )
    assert (
        (carried_result.get("metadata") or {})[
            "repair_evidence_retention"
        ]["gray_zone_input"]
        is False
    )

    serialized_gray = json.dumps(
        [
            decision
            for walk in carry_walks
            for decision in walk["gray_zone_path_decisions"]
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    for forbidden in (
        "carry_forward_supported_chunk_ids",
        "prior_supported_claim_carry_forward",
        "repair_evidence_retention",
        "source_context_package_id",
        "source_retrieval_trace_id",
    ):
        assert forbidden not in serialized_gray


@pytest.mark.asyncio
async def test_missing_citation_rebinds_current_package_before_graph_expansion(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import AgentRequest, SearchFilters
    from app.services import agent_graph

    knowledge_base, question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    package_chunks = list((package.package_json or {}).get("chunks", []))
    assert package_chunks
    source = next(
        item for item in package_chunks if len(str(item.get("content") or "")) >= 48
    )
    claim_text = str(source["content"])[:80].strip()
    claim = claim_rows(claim_text)[0]

    async def forbidden_layered_search(*_args, **_kwargs):
        raise AssertionError(
            "current-package claim/span rebind must run before graph expansion"
        )

    monkeypatch.setattr(agent_graph, "layered_search", forbidden_layered_search)
    execution = await agent_graph.execute_typed_repair_round(
        db_session,
        run=SimpleNamespace(knowledge_base_id=knowledge_base.id),
        request=AgentRequest(
            knowledge_base_id=knowledge_base.id,
            question=question,
            filters=SearchFilters(),
            top_k=4,
            retrieval_granularity="mid",
        ),
        result_top_k=4,
        query_facets=dict(initial.trace.query_facets_json or {}),
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        conversation_state_audit=dict(
            (initial.trace.diagnostics_json or {}).get("conversation_state")
            or {}
        ),
        package=package,
        verification_bundle={
            "verifications": [
                {
                    "claim_id": claim["claim_id"],
                    "claim_index": claim["claim_index"],
                    "claim_text": claim_text,
                    "answer_hash": claim["answer_hash"],
                    "citation_index": 0,
                    "chunk_id": None,
                    "source_span": {},
                    "verdict": "missing_citation",
                    "failure_type": "citation_missing",
                    "diagnostics": {
                        "claim_id": claim["claim_id"],
                        "claim_index": claim["claim_index"],
                    },
                }
            ]
        },
        action_type="repair_missing_citation",
        action_input_hash="c" * 64,
    )

    assert execution["repair_audit"]["layered_search_called"] is False
    assert any(
        str(item.get("chunk_id")) == str(source["chunk_id"])
        for item in (execution["package"].package_json or {}).get("chunks", [])
    )


@pytest.mark.asyncio
async def test_structure_repair_requires_provenance_valid_supported_source(
    db_session,
    populated_context_graph,
) -> None:
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.context_graph import validate_typed_repair_directive

    knowledge_base, _question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    query_facets = dict(initial.trace.query_facets_json or {})
    directive = _repair_directive_for_action(
        action_type="repair_structure_context",
        action_input_hash="d" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
    )

    with pytest.raises(ValueError, match="supported source"):
        validate_typed_repair_directive(
            db_session,
            knowledge_base_id=knowledge_base.id,
            query_facets=query_facets,
            retrieval_granularity="mid",
            conversation_state_scope_hash=(
                initial.trace.conversation_state_scope_hash
            ),
            repair_directive=directive,
        )


@pytest.mark.asyncio
async def test_bridge_repair_rejects_package_chunk_without_supported_bridge_edge(
    db_session,
    populated_context_graph,
) -> None:
    from sqlalchemy import select

    from app.models import ChunkRelationEdge
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.chunking import stable_hash
    from app.services.context_graph import (
        latest_relation_state,
        validate_typed_repair_directive,
    )

    knowledge_base, _question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    source_chunk_id = str(
        next(iter((package.package_json or {}).get("chunks", [])))["chunk_id"]
    )
    relation_state = latest_relation_state(db_session, knowledge_base.id)
    assert relation_state is not None
    incident_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id,
                (
                    (ChunkRelationEdge.source_chunk_id == source_chunk_id)
                    | (ChunkRelationEdge.target_chunk_id == source_chunk_id)
                ),
            )
        ).all()
    )
    for edge in incident_edges:
        edge.is_bridge = False
    db_session.flush()

    query_facets = dict(initial.trace.query_facets_json or {})
    directive = _repair_directive_for_action(
        action_type="repair_bridge_gap",
        action_input_hash="e" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
    )
    directive["bridge_seed_chunk_ids"] = [source_chunk_id]
    directive["directive_hash"] = stable_hash(
        {
            key: value
            for key, value in directive.items()
            if key != "directive_hash"
        }
    )

    with pytest.raises(ValueError, match="bridge seed|support-backed"):
        validate_typed_repair_directive(
            db_session,
            knowledge_base_id=knowledge_base.id,
            query_facets=query_facets,
            retrieval_granularity="mid",
            conversation_state_scope_hash=(
                initial.trace.conversation_state_scope_hash
            ),
            repair_directive=directive,
        )


@pytest.mark.asyncio
async def test_preverified_persistence_requires_explicit_answer_package_trace_binding(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.models import QASession
    from app.services import agent_graph
    from app.services.embeddings import ChatProvider as TrustedChatProvider

    knowledge_base, question, _initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    context = next(
        item
        for item in (package.package_json or {}).get("chunks", [])
        if len(str(item.get("content") or "")) >= 48
    )
    answer = str(context["content"])[:80].strip()
    claim = claim_rows(answer)[0]
    citation = agent_graph.citation_payloads_from_package(
        package,
        retrieval_trace_id=package.retrieval_trace_id,
        answer=answer,
        question=question,
    )[0]
    citation.pop("answer_hash", None)
    citation.pop("context_package_id", None)
    citation.pop("retrieval_trace_id", None)
    citation["source_span"] = {
        key: value
        for key, value in dict(citation.get("source_span") or {}).items()
        if key not in {"context_package_id", "retrieval_trace_id"}
    }
    verification = {
        **citation,
        "claim_id": claim["claim_id"],
        "claim_index": claim["claim_index"],
        "claim_text": claim["claim_text"],
        "verdict": "supported",
        "failure_type": "none",
        "confidence": 0.95,
        "diagnostics": {
            "claim_id": claim["claim_id"],
            "claim_index": claim["claim_index"],
            "citation_provenance_valid": True,
            "citation_provenance_session_hash": "a" * 64,
        },
    }
    verification.pop("answer_hash", None)
    qa_session = QASession(
        knowledge_base_id=knowledge_base.id,
        title="Explicit citation binding review",
    )
    db_session.add(qa_session)
    db_session.flush()
    prompt_metadata = dict(
        TrustedChatProvider()._answer_prompt_bundle(
            question,
            context_quality="normal",
        )["protocol_metadata"]
    )
    answer_model_audit = {
        "context_quality": "normal",
        "prompt_protocol_version": prompt_metadata["protocol_version"],
        "prompt_protocol_hash": prompt_metadata["prompt_protocol_hash"],
        "grounding_envelope_protocol_version": prompt_metadata[
            "protocol_version"
        ],
        "grounding_envelope_hash": prompt_metadata["envelope_hash"],
        "profile_hash": prompt_metadata["profile_hash"],
    }
    monkeypatch.setattr(
        agent_graph,
        "replay_citation_provenance_for_persistence",
        lambda *_args, **_kwargs: {
            "persistence_gate_passed": True,
            "matches_pre_entailment_session_hash": True,
            "provenance_session_hash": "a" * 64,
            "valid_count": 1,
            "invalid_count": 0,
            "transactional_replay": True,
            "lock_backend": "sqlite",
            "rows_locked": False,
            "audits": [
                {
                    "citation_index": int(
                        citation.get("citation_index") or 1
                    ),
                    "valid": True,
                    "reasons": [],
                    "provenance_hash": "b" * 64,
                }
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="preverified .*answer hash|explicit preverified binding",
    ):
        await agent_graph.record_answer_audit(
            db_session,
            knowledge_base_id=knowledge_base.id,
            qa_session_id=qa_session.id,
            question=question,
            answer=answer,
            package=package,
            contexts=agent_graph.context_package_to_contexts(package),
            answer_model_audit=answer_model_audit,
            preverified_citations=[citation],
            preverified_results=[verification],
            preverified_answer_hash=agent_graph.exact_answer_hash(answer),
            grounding_gate_audit=claim_grounding_gate(
                answer, [verification]
            ),
        )


@pytest.mark.asyncio
async def test_preverified_result_must_exactly_match_its_citation_source_span(
    db_session,
    populated_context_graph,
) -> None:
    from app.models import QASession
    from app.services import agent_graph
    from app.services.embeddings import ChatProvider as TrustedChatProvider

    knowledge_base, question, _initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    answer = _single_claim_answer_from_package(package)
    citation, verification = _bound_supported_verification(
        package,
        answer=answer,
        question=question,
    )
    verification["source_span"] = dict(verification["source_span"])
    original_span = list(verification["source_span"]["char_span"])
    verification["source_span"]["char_span"] = [
        original_span[0],
        original_span[1] + 1,
    ]
    qa_session = QASession(
        knowledge_base_id=knowledge_base.id,
        title="Exact citation/result binding review",
    )
    db_session.add(qa_session)
    db_session.flush()
    prompt_metadata = dict(
        TrustedChatProvider()._answer_prompt_bundle(
            question,
            context_quality="normal",
        )["protocol_metadata"]
    )
    answer_model_audit = {
        "context_quality": "normal",
        "prompt_protocol_version": prompt_metadata["protocol_version"],
        "prompt_protocol_hash": prompt_metadata["prompt_protocol_hash"],
        "grounding_envelope_protocol_version": prompt_metadata[
            "protocol_version"
        ],
        "grounding_envelope_hash": prompt_metadata["envelope_hash"],
        "profile_hash": prompt_metadata["profile_hash"],
    }

    with pytest.raises(ValueError, match="does not exactly match"):
        await agent_graph.record_answer_audit(
            db_session,
            knowledge_base_id=knowledge_base.id,
            qa_session_id=qa_session.id,
            question=question,
            answer=answer,
            package=package,
            contexts=agent_graph.context_package_to_contexts(package),
            answer_model_audit=answer_model_audit,
            preverified_citations=[citation],
            preverified_results=[verification],
            preverified_answer_hash=exact_answer_hash(answer),
            grounding_gate_audit=claim_grounding_gate(
                answer, [verification]
            ),
        )


@pytest.mark.asyncio
async def test_supported_evidence_carry_forward_preserves_cap_and_gray_identity(
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import SearchFilters
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.context_graph import build_context_package, layered_search
    from app.services.retrieval import (
        get_context_package,
        get_retrieval_trace_steps,
    )

    knowledge_base, question, initial, _full_package = await _initial_package(
        db_session, populated_context_graph
    )
    package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query=question,
        trace=initial.trace,
        results=initial.results[:1],
        snapshot_verifier=initial.snapshot_verifier,
    )
    answer = _single_claim_answer_from_package(package)
    citation, verification = _bound_supported_verification(
        package,
        answer=answer,
        question=question,
    )
    verification_bundle = _exact_verification_bundle(
        package,
        answer=answer,
        citations=[citation],
        verifications=[verification],
    )
    query_facets = dict(initial.trace.query_facets_json or {})
    naked_directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="e" * 64,
        package=package,
        verification_results=[verification],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
    )
    directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="f" * 64,
        package=package,
        verification_results=[verification],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        verification_bundle=verification_bundle,
    )
    carry_id = str(verification["chunk_id"])

    assert naked_directive["supported_source_chunk_ids"] == []
    assert naked_directive["carry_forward_supported_chunk_ids"] == []
    assert naked_directive["repair_source_binding"]["bundle_bound"] is False

    repaired = await layered_search(
        db_session,
        knowledge_base.id,
        question,
        SearchFilters(),
        4,
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        conversation_state_audit=dict(
            (initial.trace.diagnostics_json or {}).get("conversation_state")
            or {}
        ),
        repair_directive=directive,
    )

    result_ids = [str(item["chunk_id"]) for item in repaired.results]
    carry_result = next(
        item for item in repaired.results if str(item["chunk_id"]) == carry_id
    )
    effective = dict(
        (repaired.trace.diagnostics_json or {}).get("repair_directive") or {}
    )
    assert carry_id in directive["carry_forward_supported_chunk_ids"]
    assert carry_id in result_ids
    assert len(result_ids) <= 4
    # A sparse graph may have no novel concept after the prior Mid/package is
    # excluded.  In that case carry-only preserves already verified evidence;
    # the repair semantic-progress gate must stop instead of inventing a new
    # result or repeating the same expansion.
    assert repaired.audit["result_top_k"] == 4
    assert repaired.audit["repair_global_top_k_modified"] is False
    assert repaired.trace.agent_operating_envelope_hash == (
        initial.trace.agent_operating_envelope_hash
    )
    assert repaired.trace.traversal_protocol_hash == (
        initial.trace.traversal_protocol_hash
    )
    assert repaired.trace.convergence_json["gray_zone_model_call_count"] == 0
    assert repaired.trace.convergence_json[
        "repair_carry_forward_supported_chunk_count"
    ] == 1
    assert effective["path_distance_thresholds_modified"] is False
    assert effective["gray_zone_rule_inputs_modified"] is False
    assert (
        carry_result["metadata"]["repair_evidence_retention"][
            "gray_zone_input"
        ]
        is False
    )
    repaired_package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query=question,
        trace=repaired.trace,
        results=repaired.results,
        snapshot_verifier=repaired.snapshot_verifier,
    )
    db_session.flush()
    assert get_context_package(db_session, repaired_package.id) is not None
    public_trace = get_retrieval_trace_steps(
        db_session,
        repaired.trace.id,
    )
    assert public_trace is not None
    carry_label = next(
        label
        for label in public_trace["path_labels"]
        if label.get("chunk_id") == carry_id
        and label.get("stop_reason")
        == "repair_supported_evidence_carry_forward"
    )
    assert carry_label["path"] == [carry_id]
    assert carry_label["path_edge_ids"] == []
    assert (
        carry_label["repair_evidence_retention_protocol_version"]
        == "repair_supported_evidence_carry_forward_v1"
    )
    assert carry_label["source_context_package_id"] == str(package.id)
    assert carry_label["source_retrieval_trace_id"] == str(initial.trace.id)
    assert carry_label["repair_directive_hash"] == effective[
        "validated_directive_hash"
    ]
    chunk_selection = (repaired.trace.topk_selection_json or {})["chunk"]
    carry_rank_fact = next(
        fact
        for fact in chunk_selection["candidate_rank_facts"]
        if fact.get("candidate_id") == carry_id
        and fact.get("repair_evidence_retention_protocol_version")
        == "repair_supported_evidence_carry_forward_v1"
    )
    assert carry_rank_fact["source_context_package_id"] == str(package.id)
    assert carry_rank_fact["source_retrieval_trace_id"] == str(
        initial.trace.id
    )
    assert carry_rank_fact["repair_directive_hash"] == effective[
        "validated_directive_hash"
    ]


@pytest.mark.asyncio
async def test_stale_cross_answer_support_is_not_carried_into_repair(
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import AgentRequest, RepairExecutionAudit, SearchFilters
    from app.services import agent_graph

    knowledge_base, question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    stale_answer = _single_claim_answer_from_package(package)
    stale_citation, stale_supported = _bound_supported_verification(
        package,
        answer=stale_answer,
        question=question,
    )
    current_answer = "This exact current claim still lacks concept support."
    current_claim = claim_rows(current_answer)[0]
    current_failure = {
        "claim_id": current_claim["claim_id"],
        "claim_index": current_claim["claim_index"],
        "claim_text": current_claim["claim_text"],
        "answer_hash": current_claim["answer_hash"],
        "citation_index": 0,
        "chunk_id": None,
        "source_span": {},
        "verdict": "unsupported",
        "failure_type": "concept_gap",
        "diagnostics": {
            "claim_id": current_claim["claim_id"],
            "claim_index": current_claim["claim_index"],
            "answer_hash": current_claim["answer_hash"],
            "citation_provenance_valid": False,
        },
    }
    verification_results = [stale_supported, current_failure]
    verification_bundle = _exact_verification_bundle(
        package,
        answer=current_answer,
        citations=[stale_citation],
        verifications=verification_results,
    )
    execution = await agent_graph.execute_typed_repair_round(
        db_session,
        run=SimpleNamespace(knowledge_base_id=knowledge_base.id),
        request=AgentRequest(
            knowledge_base_id=knowledge_base.id,
            question=question,
            filters=SearchFilters(),
            top_k=4,
            retrieval_granularity="mid",
        ),
        result_top_k=4,
        query_facets=dict(initial.trace.query_facets_json or {}),
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        conversation_state_audit=dict(
            (initial.trace.diagnostics_json or {}).get("conversation_state")
            or {}
        ),
        package=package,
        verification_bundle=verification_bundle,
        action_type="repair_concept_gap",
        action_input_hash="1" * 64,
    )

    assert str(stale_supported["chunk_id"]) not in set(
        execution["directive"].get("carry_forward_supported_chunk_ids") or []
    )
    public_repair_audit = RepairExecutionAudit.model_validate(
        execution["repair_audit"]
    ).model_dump(mode="json", exclude_none=True)
    assert public_repair_audit["layered_search_called"] is True
    assert public_repair_audit["search_audit"]["contract_version"] == (
        "model_audit_public_v1"
    )
    nested_cache_audit = public_repair_audit["search_audit"][
        "retrieval_cache"
    ]
    assert nested_cache_audit["context_package_reused"] is False
    assert nested_cache_audit["write_scheduled_after_commit"] is False
    assert "retrieval_cache_hit" not in public_repair_audit
    assert "context_package_reused" not in public_repair_audit
    assert "cache_write_scheduled_after_commit" not in public_repair_audit


@pytest.mark.asyncio
async def test_repair_validator_freezes_source_gray_thresholds_under_runtime_drift(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.services.agent_graph import _repair_directive_for_action
    from app.services import context_graph

    knowledge_base, _question, initial, package = await _initial_package(
        db_session, populated_context_graph
    )
    query_facets = dict(initial.trace.query_facets_json or {})
    directive = _repair_directive_for_action(
        action_type="repair_missing_citation",
        action_input_hash="2" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
    )
    original_normalize = context_graph.normalize_agent_operating_envelope

    def drifted_normalize(payload=None):
        envelope = dict(original_normalize(payload))
        if payload is None:
            envelope["path_distance_green_threshold"] = max(
                0.0,
                float(envelope["path_distance_green_threshold"]) - 0.05,
            )
        return envelope

    monkeypatch.setattr(
        context_graph,
        "normalize_agent_operating_envelope",
        drifted_normalize,
    )
    validated = context_graph.validate_typed_repair_directive(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        repair_directive=directive,
    )

    assert validated is not None
    assert validated["runtime_threshold_drift_observed"] is True
    assert validated["frozen_agent_operating_envelope_hash"] == (
        initial.trace.agent_operating_envelope_hash
    )
    assert validated["frozen_traversal_protocol_hash"] == (
        initial.trace.traversal_protocol_hash
    )
    assert validated["path_distance_thresholds_modified"] is False
    assert validated["gray_zone_rule_inputs_modified"] is False
