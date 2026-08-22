from __future__ import annotations

from app.services.agent_repair import (
    CLAIM_GROUNDED_GATE_PROTOCOL_VERSION,
    TYPED_REPAIR_PROTOCOL_VERSION,
    canonical_failure_cards,
    claim_grounding_gate,
    claim_rows,
    exact_answer_hash,
    repair_made_progress,
    repair_semantic_progress_signature,
    select_repair_direction,
    split_answer_claims,
    supported_partial_answer,
)
from app.services.agent_graph import verify_answer_against_context_rules
import pytest


def _verification(
    claim: dict,
    *,
    verdict: str,
    failure_type: str = "none",
    chunk_id: str = "chunk-1",
) -> dict:
    return {
        "claim_id": claim["claim_id"],
        "claim_index": claim["claim_index"],
        "claim_text": claim["claim_text"],
        "answer_hash": claim["answer_hash"],
        "citation_index": claim["claim_index"] + 1,
        "chunk_id": chunk_id,
        "source_span": {
            "document_version_id": "version-1",
            "char_span": [0, 12],
            "raw_span_text_hash": "a" * 64,
        },
        "verdict": verdict,
        "failure_type": failure_type,
        "diagnostics": {
            "claim_id": claim["claim_id"],
            "claim_index": claim["claim_index"],
            "answer_hash": claim["answer_hash"],
            "citation_provenance_valid": True,
        },
    }


def test_split_answer_claims_drops_standalone_punctuation() -> None:
    assert split_answer_claims("Alpha is supported.\n\u201d") == [
        "Alpha is supported."
    ]


def test_claim_grounded_gate_does_not_promote_sibling_claims() -> None:
    answer = "The supported fact is alpha. The invented fact is omega."
    claims = claim_rows(answer)
    results = [
        _verification(claims[0], verdict="supported"),
        _verification(
            claims[1],
            verdict="unsupported",
            failure_type="unsupported_claim",
            chunk_id="chunk-2",
        ),
    ]

    gate = claim_grounding_gate(answer, results)

    assert gate["protocol_version"] == CLAIM_GROUNDED_GATE_PROTOCOL_VERSION
    assert gate["answer_hash"] == exact_answer_hash(answer)
    assert gate["claim_count"] == 2
    assert gate["supported_claim_count"] == 1
    assert gate["claim_pass_rate"] == 0.5
    assert gate["all_claims_supported"] is False
    assert gate["claims"][0]["supported"] is True
    assert gate["claims"][1]["supported"] is False

    partial = supported_partial_answer(answer, gate)
    assert partial["answer"] == claims[0]["claim_text"]
    assert "omega" not in partial["answer"]
    assert partial["evidence_gap"]["dropped_claim_count"] == 1


def test_claim_grounded_gate_requires_raw_provenance() -> None:
    answer = "Alpha is supported."
    claim = claim_rows(answer)[0]
    result = _verification(claim, verdict="supported")
    result["diagnostics"]["citation_provenance_valid"] = False

    gate = claim_grounding_gate(answer, [result])

    assert gate["supported_claim_count"] == 0
    assert gate["all_claims_supported"] is False


def test_failure_cards_and_repair_directions_are_typed_and_non_repeating() -> None:
    answer = "A formula claim."
    claim = claim_rows(answer)[0]
    result = _verification(
        claim,
        verdict="formula_table_context_missing",
        failure_type="formula_context_missing",
    )
    cards = canonical_failure_cards(
        answer=answer,
        verification_results=[result],
        repair_round_index=0,
        remaining_repair_budget=2,
        context_package_id="package-1",
        retrieval_trace_id="trace-1",
        structure_closure_status={"formula": False},
        covered_facets=["formula"],
        missing_evidence_roles=["formula_table"],
        prior_repair_action_output_hashes=[],
    )

    first = select_repair_direction(cards)
    assert first is not None
    assert first["action_type"] == "repair_structure_context"
    assert first["executor_mechanism"] == "supported_chunk_structure_closure_v1"

    second = select_repair_direction(
        cards,
        attempted_input_hashes_by_action={
            "repair_structure_context": {first["input_hash"]}
        },
    )
    assert second is not None
    assert second["action_type"] == "repair_missing_citation"
    assert cards[0]["failure_card_hash"]
    assert TYPED_REPAIR_PROTOCOL_VERSION == "typed_repair_loop_v1"


def test_exhausted_direction_is_not_reused_when_next_package_changes_input() -> None:
    answer = "A concept claim."
    claim = claim_rows(answer)[0]
    result = _verification(
        claim,
        verdict="unsupported",
        failure_type="concept_gap",
    )
    first_cards = canonical_failure_cards(
        answer=answer,
        verification_results=[result],
        repair_round_index=0,
        remaining_repair_budget=2,
        context_package_id="package-1",
        retrieval_trace_id="trace-1",
        structure_closure_status={"chunk_count": 1},
        covered_facets=["concept-a"],
        missing_evidence_roles=["concept_support"],
        prior_repair_action_output_hashes=[],
    )
    first = select_repair_direction(first_cards)
    assert first is not None
    assert first["action_type"] == "repair_concept_gap"

    second_cards = canonical_failure_cards(
        answer=answer,
        verification_results=[result],
        repair_round_index=1,
        remaining_repair_budget=1,
        context_package_id="package-2",
        retrieval_trace_id="trace-2",
        structure_closure_status={"chunk_count": 2},
        covered_facets=["concept-a", "concept-b"],
        missing_evidence_roles=["concept_support"],
        prior_repair_action_output_hashes=["output-1"],
    )
    changed_input = select_repair_direction(second_cards)
    assert changed_input is not None
    assert changed_input["action_type"] == "repair_concept_gap"
    assert changed_input["input_hash"] != first["input_hash"]

    alternate = select_repair_direction(
        second_cards,
        attempted_input_hashes_by_action={
            "repair_concept_gap": {first["input_hash"]},
        },
        exhausted_action_types={"repair_concept_gap"},
    )
    assert alternate is not None
    assert alternate["action_type"] == "repair_missing_citation"


def test_progress_ignores_new_database_ids_without_new_evidence() -> None:
    kwargs = {
        "result_chunk_ids": ["chunk-1"],
        "package_chunk_spans": [
            {
                "chunk_id": "chunk-1",
                "document_version_id": "version-1",
                "char_span": [0, 12],
                "raw_span_text_hash": "a" * 64,
            }
        ],
        "covered_facets": ["alpha"],
        "evidence_roles": ["definition"],
        "graph_path_ids": ["edge-business-key-1"],
        "supported_claim_ids": [],
        "unsupported_claim_ids": ["claim-1"],
    }
    before = repair_semantic_progress_signature(**kwargs)
    after = repair_semantic_progress_signature(**kwargs)

    assert repair_made_progress(before, after) is False

    improved = repair_semantic_progress_signature(
        **{
            **kwargs,
            "supported_claim_ids": ["claim-1"],
            "unsupported_claim_ids": [],
        }
    )
    assert repair_made_progress(before, improved) is True


def test_rule_verifier_binds_each_claim_to_its_own_citation() -> None:
    answer = "Alpha is present. Omega is invented."
    claims = claim_rows(answer)
    citations = [
        {
            "citation_index": 1,
            "claim_id": claims[0]["claim_id"],
            "claim_index": 0,
            "claim_text": claims[0]["claim_text"],
            "chunk_id": "chunk-alpha",
            "source_span": {
                "document_version_id": "version-1",
                "char_span": [0, 17],
                "raw_span_text_hash": "a" * 64,
            },
        },
        {
            "citation_index": 2,
            "claim_id": claims[1]["claim_id"],
            "claim_index": 1,
            "claim_text": claims[1]["claim_text"],
            "chunk_id": "chunk-alpha",
            "source_span": {
                "document_version_id": "version-1",
                "char_span": [0, 17],
                "raw_span_text_hash": "a" * 64,
            },
        },
    ]
    results = verify_answer_against_context_rules(
        answer,
        citations,
        [{"chunk_id": "chunk-alpha", "content": "Alpha is present."}],
        verification_budget=2,
        provenance_gate={
            "provenance_session_hash": "c" * 64,
            "audits": [
                {
                    "citation_index": 1,
                    "valid": True,
                    "reasons": [],
                    "provenance_hash": "d" * 64,
                },
                {
                    "citation_index": 2,
                    "valid": True,
                    "reasons": [],
                    "provenance_hash": "e" * 64,
                },
            ],
        },
    )

    assert [item["claim_id"] for item in results] == [
        claims[0]["claim_id"],
        claims[1]["claim_id"],
    ]
    assert results[0]["verdict"] == "supported"
    assert results[1]["verdict"] == "unsupported"
    gate = claim_grounding_gate(answer, results)
    assert gate["all_claims_supported"] is False
    assert gate["claim_pass_rate"] == 0.5


@pytest.mark.asyncio
async def test_repair_directive_is_audited_without_changing_gray_authority(
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import SearchFilters
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.context_graph import build_context_package, layered_search

    knowledge_base = populated_context_graph["knowledge_base"]
    initial = await layered_search(
        db_session,
        knowledge_base.id,
        "Explain Bayesian network factorization.",
        SearchFilters(),
        4,
        retrieval_granularity="mid",
    )
    package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query="Explain Bayesian network factorization.",
        trace=initial.trace,
        results=initial.results,
        snapshot_verifier=initial.snapshot_verifier,
    )
    query_facets = dict(initial.trace.query_facets_json or {})
    conversation_audit = dict(
        (initial.trace.diagnostics_json or {}).get("conversation_state") or {}
    )
    directive = _repair_directive_for_action(
        action_type="repair_missing_citation",
        action_input_hash="a" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=initial.trace.conversation_state_scope_hash,
    )

    repaired = await layered_search(
        db_session,
        knowledge_base.id,
        "Explain Bayesian network factorization.",
        SearchFilters(),
        4,
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=initial.trace.conversation_state_scope_hash,
        conversation_state_audit=conversation_audit,
        repair_directive=directive,
    )

    assert repaired.audit["result_top_k"] == 4
    assert repaired.audit["repair_global_top_k_modified"] is False
    assert repaired.audit["repair_action_type"] == "repair_missing_citation"
    assert repaired.trace.conversation_state_scope_hash == initial.trace.conversation_state_scope_hash
    assert repaired.trace.convergence_json["gray_zone_model_call_count"] == 0
    assert (repaired.trace.diagnostics_json or {})["repair_gray_zone_decision_authority"] is False
    assert (repaired.trace.diagnostics_json or {})["repair_directive"]["query_facets_hash"] == directive["query_facets_hash"]
    assert not set(repaired.trace.result_chunk_ids_json or []).intersection(
        set(directive["prior_package_chunk_ids"])
    )


@pytest.mark.asyncio
async def test_repair_directive_rejects_gray_authority_tampering(
    db_session,
    populated_context_graph,
) -> None:
    from app.schemas import SearchFilters
    from app.services.agent_graph import _repair_directive_for_action
    from app.services.chunking import stable_hash
    from app.services.context_graph import build_context_package, layered_search

    knowledge_base = populated_context_graph["knowledge_base"]
    initial = await layered_search(
        db_session,
        knowledge_base.id,
        "Explain Bayesian network factorization.",
        SearchFilters(),
        4,
        retrieval_granularity="mid",
    )
    package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query="Explain Bayesian network factorization.",
        trace=initial.trace,
        results=initial.results,
        snapshot_verifier=initial.snapshot_verifier,
    )
    query_facets = dict(initial.trace.query_facets_json or {})
    directive = _repair_directive_for_action(
        action_type="repair_concept_gap",
        action_input_hash="b" * 64,
        package=package,
        verification_results=[],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=initial.trace.conversation_state_scope_hash,
    )
    directive["gray_zone_decision_authority"] = True
    directive["directive_hash"] = stable_hash(
        {key: value for key, value in directive.items() if key != "directive_hash"}
    )

    with pytest.raises(ValueError, match="gray-zone authority"):
        await layered_search(
            db_session,
            knowledge_base.id,
            "Explain Bayesian network factorization.",
            SearchFilters(),
            4,
            query_facets=query_facets,
            retrieval_granularity="mid",
            conversation_state_scope_hash=initial.trace.conversation_state_scope_hash,
            conversation_state_audit=(initial.trace.diagnostics_json or {})[
                "conversation_state"
            ],
            repair_directive=directive,
        )


@pytest.mark.asyncio
async def test_structure_repair_executes_supported_formula_table_closure(
    db_session,
    populated_context_graph,
) -> None:
    from types import SimpleNamespace

    from app.schemas import AgentRequest, SearchFilters
    from app.services.agent_graph import (
        citation_payloads_from_package,
        execute_typed_repair_round,
    )
    from app.services.chunking import stable_hash
    from app.services.context_graph import build_context_package, layered_search

    knowledge_base = populated_context_graph["knowledge_base"]
    question = "Explain the factorization formula and its variable table."
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
    source_item = next(
        item
        for item in (package.package_json or {}).get("chunks", [])
        if "table and formula" in str(item.get("content") or "").lower()
    )
    source_span = dict(source_item["source_span"])
    source_chunk_id = str(source_item["chunk_id"])
    source_content = " ".join(
        str(source_item.get("content") or "").split()
    )
    answer = f"{source_content.split('.', 1)[0].strip()}."
    citation = next(
        item
        for item in citation_payloads_from_package(
            package,
            retrieval_trace_id=package.retrieval_trace_id,
            answer=answer,
            question=question,
        )
        if str(item.get("chunk_id")) == source_chunk_id
    )
    verification = {
        **citation,
        "source_span": dict(citation.get("source_span") or {}),
        "verdict": "formula_table_context_missing",
        "failure_type": "formula_context_missing",
        "confidence": 0.2,
        "diagnostics": {
            "claim_id": citation.get("claim_id"),
            "claim_index": citation.get("claim_index"),
            "answer_hash": citation.get("answer_hash"),
            "citation_provenance_valid": True,
        },
    }
    gate = claim_grounding_gate(answer, [verification])
    binding = {
        "answer_hash": exact_answer_hash(answer),
        "context_package_id": package.id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "citation_indexes": [int(citation.get("citation_index") or 0)],
        "claim_ids": [row["claim_id"] for row in gate["claims"]],
        "gate_hash": gate["gate_hash"],
    }
    verification_bundle = {
        "answer": answer,
        "answer_hash": exact_answer_hash(answer),
        "citations": [citation],
        "verifications": [verification],
        "gate": gate,
        "binding": binding,
        "binding_hash": stable_hash(
            {
                "protocol_version": (
                    CLAIM_GROUNDED_GATE_PROTOCOL_VERSION
                ),
                **binding,
            }
        ),
    }
    conversation_audit = dict(
        (initial.trace.diagnostics_json or {}).get("conversation_state") or {}
    )
    execution = await execute_typed_repair_round(
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
        conversation_state_audit=conversation_audit,
        package=package,
        verification_bundle=verification_bundle,
        action_type="repair_structure_context",
        action_input_hash="c" * 64,
    )

    repaired_package = execution["package"]
    repaired_chunks = list(
        (repaired_package.package_json or {}).get("chunks", [])
    )
    restored_ids = set(repaired_package.restored_chunk_ids_json or [])
    assert repaired_package.hit_chunk_ids_json == [source_chunk_id]
    assert restored_ids
    assert any(
        str(item.get("chunk_id")) in restored_ids
        and (
            any(
                str(node.get("node_type")) in {"formula", "table", "caption"}
                for node in item.get("structure_nodes") or []
            )
            or "prod" in str(item.get("content") or "").lower()
            or "| variable |" in str(item.get("content") or "").lower()
        )
        for item in repaired_chunks
    )
    diagnostics = repaired_package.diagnostics_json or {}
    assert diagnostics["repair_action_type"] == "repair_structure_context"
    assert diagnostics["repair_executor_mechanism"] == (
        "supported_chunk_structure_closure_v1"
    )
    assert diagnostics["repair_gray_zone_model_call_count"] == 0
    assert diagnostics["repair_gray_zone_decision_authority"] is False
    assert execution["retrieval_trace"].id == initial.trace.id
    assert execution["repair_audit"]["layered_search_called"] is False
    assert execution["repair_audit"]["answer_regenerated"] is False
    assert execution["repair_audit"]["global_top_k_increased"] is False


@pytest.mark.asyncio
async def test_bridge_repair_uses_supported_edges_as_normal_traversal_seeds(
    db_session,
    populated_context_graph,
) -> None:
    from collections import defaultdict

    from sqlalchemy import select

    from app.models import ChunkRelationEdge, GraphRetrievalStep
    from app.schemas import SearchFilters
    from app.services.agent_graph import (
        _repair_directive_for_action,
        citation_payloads_from_package,
    )
    from app.services.chunking import stable_hash
    from app.services.context_graph import (
        build_context_package,
        layered_search,
        latest_relation_state,
    )

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
    relation_state = latest_relation_state(db_session, knowledge_base.id)
    assert relation_state is not None
    edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id
            )
        ).all()
    )
    incident: dict[str, list[ChunkRelationEdge]] = defaultdict(list)
    for edge in edges:
        incident[str(edge.source_chunk_id)].append(edge)
        incident[str(edge.target_chunk_id)].append(edge)
    initial_by_chunk = {
        str(item["chunk_id"]): item for item in initial.results
    }
    source_chunk_id = next(
        chunk_id
        for chunk_id in initial_by_chunk
        if len(
            {
                (
                    str(edge.target_chunk_id)
                    if str(edge.source_chunk_id) == chunk_id
                    else str(edge.source_chunk_id)
                )
                for edge in incident[chunk_id]
            }
            - {chunk_id}
        )
        >= 3
    )
    source_result = dict(initial_by_chunk[source_chunk_id])
    package = build_context_package(
        db_session,
        knowledge_base_id=knowledge_base.id,
        query=question,
        trace=initial.trace,
        results=[source_result],
        restore_per_chunk_budget=0,
        snapshot_verifier=initial.snapshot_verifier,
    )
    prior_chunk_ids = {
        str(item.get("chunk_id"))
        for item in (package.package_json or {}).get("chunks", [])
    }
    # Freeze the pre-repair package before promoting candidate relation edges
    # to bridges. Otherwise normal context restoration correctly follows the
    # newly marked bridges and the fixture erases the bridge gap it is meant to
    # repair.
    selected_edges: list[ChunkRelationEdge] = []
    target_ids: set[str] = set()
    for edge in incident[source_chunk_id]:
        target_id = (
            str(edge.target_chunk_id)
            if str(edge.source_chunk_id) == source_chunk_id
            else str(edge.source_chunk_id)
        )
        if target_id in target_ids or target_id in prior_chunk_ids:
            continue
        edge.is_bridge = True
        selected_edges.append(edge)
        target_ids.add(target_id)
        if len(selected_edges) == 2:
            break
    assert len(selected_edges) == 2
    db_session.flush()

    novel_bridge_targets = target_ids - prior_chunk_ids
    assert novel_bridge_targets
    query_facets = dict(initial.trace.query_facets_json or {})
    conversation_audit = dict(
        (initial.trace.diagnostics_json or {}).get("conversation_state") or {}
    )
    source_content = " ".join(
        str(
            next(
                item
                for item in (package.package_json or {}).get("chunks", [])
                if str(item.get("chunk_id")) == source_chunk_id
            ).get("content")
            or ""
        ).split()
    )
    answer = f"{source_content.split('.', 1)[0].strip()}."
    citation = next(
        item
        for item in citation_payloads_from_package(
            package,
            retrieval_trace_id=package.retrieval_trace_id,
            answer=answer,
            question=question,
        )
        if str(item.get("chunk_id")) == source_chunk_id
    )
    verification = {
        **citation,
        "source_span": dict(citation.get("source_span") or {}),
        "verdict": "unsupported",
        "failure_type": "bridge_gap",
        "confidence": 0.2,
        "diagnostics": {
            "claim_id": citation.get("claim_id"),
            "claim_index": citation.get("claim_index"),
            "answer_hash": citation.get("answer_hash"),
            "citation_provenance_valid": True,
        },
    }
    gate = claim_grounding_gate(answer, [verification])
    binding = {
        "answer_hash": exact_answer_hash(answer),
        "context_package_id": package.id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "citation_indexes": [int(citation.get("citation_index") or 0)],
        "claim_ids": [row["claim_id"] for row in gate["claims"]],
        "gate_hash": gate["gate_hash"],
    }
    verification_bundle = {
        "answer": answer,
        "answer_hash": exact_answer_hash(answer),
        "citations": [citation],
        "verifications": [verification],
        "gate": gate,
        "binding": binding,
        "binding_hash": stable_hash(
            {
                "protocol_version": (
                    CLAIM_GROUNDED_GATE_PROTOCOL_VERSION
                ),
                **binding,
            }
        ),
    }
    directive = _repair_directive_for_action(
        action_type="repair_bridge_gap",
        action_input_hash="d" * 64,
        package=package,
        verification_results=[verification],
        query_facets=query_facets,
        retrieval_granularity="mid",
        conversation_state_scope_hash=(
            initial.trace.conversation_state_scope_hash
        ),
        verification_bundle=verification_bundle,
    )
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
        conversation_state_audit=conversation_audit,
        repair_directive=directive,
    )

    effective = dict(
        (repaired.trace.diagnostics_json or {})["repair_directive"]
    )
    assert novel_bridge_targets.issubset(
        set(effective["bridge_seed_chunk_ids"])
    )
    assert {str(edge.id) for edge in selected_edges}.issubset(
        set(effective["bridge_support_edge_ids"])
    )
    assert effective["executor_mechanism"] == (
        "support_backed_bridge_seed_then_layered_traversal_v1"
    )
    assert effective["gray_zone_rule_inputs_modified"] is False
    assert effective["path_distance_thresholds_modified"] is False
    assert effective["global_top_k_modified"] is False
    assert repaired.trace.convergence_json["repair_bridge_seed_count"] >= 2
    assert repaired.trace.convergence_json["gray_zone_model_call_count"] == 0
    assert not set(repaired.trace.result_chunk_ids_json or []).intersection(
        prior_chunk_ids
    )
    chunk_step = db_session.scalar(
        select(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == repaired.trace.id,
            GraphRetrievalStep.layer == "chunk",
            GraphRetrievalStep.action == "walk_graph_frontier",
        )
    )
    assert chunk_step is not None
    serialized_entry_nodes = str(
        (chunk_step.input_json or {}).get("entry_nodes") or []
    )
    assert any(
        target_id in serialized_entry_nodes
        for target_id in novel_bridge_targets
    )
    gray_records = list(chunk_step.gray_zone_path_decisions_json or [])
    assert gray_records
    serialized_gray_records = str(gray_records)
    for executor_only_value in (
        "support_backed_bridge_repair_seed",
        "repair_action_type",
        "repair_directive_hash",
        "direct_context_package_injection",
    ):
        assert executor_only_value not in serialized_gray_records
