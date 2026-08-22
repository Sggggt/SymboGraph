from __future__ import annotations

from copy import deepcopy

import pytest


def _facets() -> dict:
    return {
        "protocol_version": "query_facet_packet_v2",
        "query": "技术必须被什么夹在中间？",
        "intent": "definition",
        "answer_shape": "definition",
        "terms": ["技术", "夹在中间"],
        "required_facets": ["技术", "技术必须被什么夹在中间"],
        "facet_groups": [
            {
                "facet": "技术",
                "role": "domain",
                "aliases": ["technology"],
                "source": "llm",
                "confidence": 0.7,
            },
            {
                "facet": "技术必须被什么夹在中间",
                "role": "constraint",
                "aliases": ["技术被夹在中间", "夹在中间"],
                "source": "llm",
                "confidence": 0.7,
            },
        ],
        "drop_terms": [],
        "diagnostics": {"source": "llm_validated"},
    }


def test_ordered_window_rejects_document_wide_scatter_but_accepts_local_order():
    from app.services.context_graph import matched_required_facets_for_text

    facets = _facets()
    scattered = (
        "技术系统先介绍访问控制。后续章节说接口必须鉴权。"
        + "无关内容 " * 40
        + "最后讨论中间层。"
    )
    local = "图示说明：技术必须被需求和验证夹在中间。"

    assert matched_required_facets_for_text(scattered, facets) == ["技术"]
    assert matched_required_facets_for_text(local, facets) == [
        "技术",
        "技术必须被什么夹在中间",
    ]


def test_lossy_single_token_alias_does_not_become_document_wide_generic_hit():
    from app.services.context_graph import matched_required_facets_for_text

    facets = _facets()
    assert matched_required_facets_for_text("这是一段中间层说明。", facets) == []
    assert matched_required_facets_for_text("技术被夹在中间。", facets) == [
        "技术",
        "技术必须被什么夹在中间",
    ]


def test_chunk_facet_priority_card_is_boolean_audit_not_relevance_or_authority():
    from app.schemas import RetrievalChunkFacetPriorityCard
    from app.services.context_graph import chunk_facet_priority_card

    card = chunk_facet_priority_card(
        chunk_id="target",
        text="技术必须被需求和验证夹在中间。",
        query_facets=_facets(),
    )
    parsed = RetrievalChunkFacetPriorityCard.model_validate(card, strict=True)

    assert parsed.priority_prefix == [0]
    assert parsed.matched_required_facet_count == 2
    assert parsed.lexical_overlap_used_as_numeric_relevance is False
    assert parsed.is_evidence is False
    assert parsed.citation_authority is False
    assert parsed.gray_zone_decision_authority is False
    assert parsed.model_call_count == 0

    tampered = deepcopy(card)
    tampered["uncovered_required_facet_count"] = 1
    with pytest.raises(ValueError):
        RetrievalChunkFacetPriorityCard.model_validate(tampered, strict=True)


def test_query_facet_posterior_replays_prior_likelihood_aliases_and_zero_model_calls():
    from app.schemas import QueryFacetPosteriorCalibration
    from app.services.context_graph import QueryFacetPosteriorCalibrator, stable_hash

    facets = {
        "required_facets": ["alpha", "beta"],
        "facet_groups": [
            {"facet": "alpha", "aliases": ["first"]},
            {"facet": "beta", "aliases": ["second"]},
        ],
    }
    candidates = [
        {
            "candidate_id": "one",
            "text": "first only",
            "scope": "test",
            "candidate_business_input_hash": stable_hash({"id": "one"}),
        },
        {
            "candidate_id": "two",
            "text": "unrelated",
            "scope": "test",
            "candidate_business_input_hash": stable_hash({"id": "two"}),
        },
    ]
    first = QueryFacetPosteriorCalibrator(
        query_facets=facets,
        enabled=True,
        observation_budget=4,
        round_budget=2,
        convergence_epsilon=0.0,
    )
    first.observe("dense_entry_candidates", candidates)
    first.observe("merged_chunk_candidates", candidates)
    card = first.card()
    parsed = QueryFacetPosteriorCalibration.model_validate(card, strict=True)

    assert parsed.prior == {"alpha": 0.5, "beta": 0.5}
    assert parsed.rounds[0].likelihood == {"alpha": 0.5, "beta": 0.25}
    assert parsed.rounds[0].posterior == {"alpha": 0.666667, "beta": 0.333333}
    assert parsed.rounds[0].alias_posterior["alpha"] == {
        "alpha": 0.333333,
        "first": 0.666667,
    }
    assert parsed.model_call_count == 0
    assert parsed.llm_sample_budget == 0
    assert parsed.gray_zone_decision_authority is False
    assert parsed.is_evidence is False

    replay = QueryFacetPosteriorCalibrator(
        query_facets=facets,
        enabled=True,
        observation_budget=4,
        round_budget=2,
        convergence_epsilon=0.0,
    )
    replay.observe("dense_entry_candidates", candidates)
    replay.observe("merged_chunk_candidates", candidates)
    assert replay.card() == card


def test_query_facet_posterior_hard_observation_budget_stops_without_second_round():
    from app.services.context_graph import QueryFacetPosteriorCalibrator, stable_hash

    calibrator = QueryFacetPosteriorCalibrator(
        query_facets={
            "required_facets": ["alpha", "beta"],
            "facet_groups": [
                {"facet": "alpha", "aliases": []},
                {"facet": "beta", "aliases": []},
            ],
        },
        enabled=True,
        observation_budget=1,
        round_budget=2,
        convergence_epsilon=0.0,
    )
    candidates = [
        {
            "candidate_id": "one",
            "text": "alpha",
            "scope": "test",
            "candidate_business_input_hash": stable_hash({"id": "one"}),
        },
        {
            "candidate_id": "two",
            "text": "beta",
            "scope": "test",
            "candidate_business_input_hash": stable_hash({"id": "two"}),
        },
    ]
    calibrator.observe("dense_entry_candidates", candidates)
    calibrator.observe("merged_chunk_candidates", candidates)
    card = calibrator.card()

    assert card["observations_used"] == 1
    assert card["rounds_used"] == 1
    assert card["stop_reason"] == "observation_budget_exhausted"
    assert card["model_call_count"] == 0


def test_preposterior_build_envelope_replays_without_graph_rebuild_or_default_fill():
    from app.schemas import RetrievalAgentOperatingEnvelope
    from app.services import context_graph

    current = context_graph.agent_operating_envelope()
    historical = {
        key: value
        for key, value in current.items()
        if not key.startswith("query_facet_posterior_")
    }
    roundtrip = RetrievalAgentOperatingEnvelope.model_validate(
        historical
    ).model_dump(mode="json", exclude_unset=True)

    assert roundtrip == historical
    assert (
        context_graph.frozen_traversal_protocol_hash(historical)
        != context_graph.traversal_protocol_hash(current)
    )
    assert (
        context_graph.frozen_traversal_protocol_hash(historical)
        == context_graph._traversal_protocol_hash_from_effective_envelope(
            historical,
            chunk_facet_priority_protocol_version=(
                context_graph.LEGACY_CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
            ),
            chunk_facet_priority_protocol_hash_value=(
                context_graph.LEGACY_CHUNK_FACET_PRIORITY_PROTOCOL_HASH
            ),
        )
    )


def test_candidate_pool_replays_uncovered_facet_prefix_before_existing_score():
    from app.schemas import RetrievalCandidatePool
    from app.services.context_graph import (
        CHUNK_FACET_PRIORITY_PROTOCOL_VERSION,
        QueryFacetPosteriorCalibrator,
        chunk_facet_priority_card,
        chunk_facet_priority_protocol_hash,
        stable_hash,
    )

    facets = _facets()
    complete = chunk_facet_priority_card(
        chunk_id="complete",
        text="技术必须被需求和验证夹在中间。",
        query_facets=facets,
    )
    generic = chunk_facet_priority_card(
        chunk_id="generic",
        text="技术平台。",
        query_facets=facets,
    )
    posterior = QueryFacetPosteriorCalibrator(
        query_facets=facets,
        enabled=True,
        observation_budget=4,
        round_budget=2,
        convergence_epsilon=0.0,
    )
    observations = [
        {
            "candidate_id": "complete",
            "text": "技术必须被需求和验证夹在中间。",
            "scope": "test",
            "candidate_business_input_hash": stable_hash({"id": "complete"}),
        },
        {
            "candidate_id": "generic",
            "text": "技术平台。",
            "scope": "test",
            "candidate_business_input_hash": stable_hash({"id": "generic"}),
        },
    ]
    posterior.observe("dense_entry_candidates", observations)
    posterior_snapshot = posterior.snapshot()
    posterior.observe("merged_chunk_candidates", observations)
    payload = {
        "candidate_ids": ["complete", "generic"],
        "candidate_scores": {"complete": 0.2, "generic": 0.9},
        "chunk_facet_priority_cards": {
            "complete": complete,
            "generic": generic,
        },
        "facet_priority_protocol_version": (
            CHUNK_FACET_PRIORITY_PROTOCOL_VERSION
        ),
        "facet_priority_protocol_hash": chunk_facet_priority_protocol_hash(),
        "query_facet_posterior_snapshot": posterior_snapshot,
        "covered_posterior_mass_by_candidate": {
            "complete": 1.0,
            "generic": posterior_snapshot["posterior"]["技术"],
        },
        "rq_seed_cards": {},
        "rq_chunk_seed_cards": {},
        "ranking_protocol_version": None,
        "ranking_protocol_hash": None,
        "forced_candidate_ids": [],
        "selected_ids": ["complete"],
        "ranked_selected_ids": ["complete"],
        "candidate_count": 2,
    }
    RetrievalCandidatePool.model_validate(payload, strict=True)

    reversed_payload = deepcopy(payload)
    reversed_payload["candidate_ids"] = ["generic", "complete"]
    with pytest.raises(ValueError, match="ranking order"):
        RetrievalCandidatePool.model_validate(reversed_payload, strict=True)


def test_concept_walk_keeps_validated_queue_facets_out_of_gray_observation(
    monkeypatch,
):
    from types import SimpleNamespace

    from app.services import context_graph

    edge = SimpleNamespace(
        id="edge-1",
        source_concept_id="entry",
        target_concept_id="next",
        distance=0.1,
        weight=0.9,
        edge_type="bridge_to",
        features_json={"semantic_uncertain": True},
        diagnostics_json={"crossing_rq_boundary": True},
        support_chunk_ids_json=["support-1"],
        support_chunk_edge_ids_json=["edge-support-1"],
        support_relation_edge_ids_json=["edge-support-1"],
        support_rq_prefix_ids_json=["rq-1"],
        support_rq_prefix_node_ids_json=["rq-1"],
    )
    captured: list[dict] = []

    def fake_decision(observation: dict, *, include_observation: bool = True):
        captured.append(deepcopy(observation))
        return {
            "decision": "continue_path",
            "matched_rule": "test",
            "input_hash": "a" * 64,
            "decision_hash": "b" * 64,
            "model_call_count": 0,
            "decision_source": "deterministic_local_rule",
            "observation_compacted": not include_observation,
        }

    monkeypatch.setattr(context_graph, "deterministic_gray_zone_decision", fake_decision)
    envelope = context_graph.normalize_agent_operating_envelope(
        {
            **context_graph.agent_operating_envelope(),
            "path_distance_green_threshold": 0.01,
            "path_distance_gray_threshold": 10.0,
            "path_distance_hard_threshold": 20.0,
        }
    )
    validated = {
        "required_facets": ["provider-facet"],
        "facet_groups": [
            {"facet": "provider-facet", "aliases": ["validated-only"]}
        ],
    }
    local_gray = {
        "required_facets": ["raw-local"],
        "facet_groups": [
            {"facet": "raw-local", "aliases": ["raw-anchor"]}
        ],
    }
    context_graph.execute_layer_priority_walk(
        layer="mid",
        entry_scores={"entry": 0.9},
        node_text_by_id={
            "entry": "validated-only raw-anchor",
            "next": "provider-facet",
        },
        adjacency={"entry": [edge], "next": [edge]},
        query_facets=validated,
        gray_query_facets=local_gray,
        source_attr="source_concept_id",
        target_attr="target_concept_id",
        envelope=envelope,
    )

    assert captured
    assert captured[0]["required_facets"] == ["raw-local"]
    assert captured[0]["covered_facets_before"] == ["raw-local"]
    assert "provider-facet" not in captured[0]["covered_facets_after"]
