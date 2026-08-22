from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest


def _two_way_rq_model(*, threshold: float = 0.0) -> dict:
    from app.services.context_graph import (
        RQ_INDEX_PROTOCOL_VERSION,
        RQ_MEMBERSHIP_PROTOCOL_VERSION,
        rq_membership_protocol_hash,
    )
    from app.services.chunking import stable_hash

    codebooks = [
        [[-1.0, 0.0], [1.0, 0.0]],
        [[0.0, -1.0], [0.0, 1.0]],
        [[-0.5, 0.0], [0.5, 0.0]],
    ]
    config = {
        "levels": 3,
        "max_k": 2,
        "tau_r": 0.65,
        "tau_l": 0.35,
        "top_m": 2,
        "probability_threshold": threshold,
        "membership_protocol": RQ_MEMBERSHIP_PROTOCOL_VERSION,
    }
    return {
        "levels": 3,
        "max_k": 2,
        "codebooks": codebooks,
        "codebook_sizes": [2, 2, 2],
        "embedding_dimensions": 2,
        "tau_r": 0.65,
        "tau_l": 0.35,
        "membership_top_m": 2,
        "membership_probability_threshold": threshold,
        "membership_protocol": RQ_MEMBERSHIP_PROTOCOL_VERSION,
        "membership_protocol_hash": rq_membership_protocol_hash(config),
        "codebook_hash": stable_hash(
            {
                "index_protocol": RQ_INDEX_PROTOCOL_VERSION,
                "embedding_dimensions": 2,
                "codebooks": codebooks,
            }
        ),
        "index_protocol": RQ_INDEX_PROTOCOL_VERSION,
    }


def test_rq_encoder_materializes_sparse_fuzzy_prefixes_from_full_softmax():
    from app.services.context_graph import encode_rq_vector

    encoded = encode_rq_vector([0.0, 0.0], _two_way_rq_model())

    assert encoded["rq_path"] == [1, 1, 2]
    assert all(
        assignment["probability_sum"] == pytest.approx(1.0, abs=1e-12)
        for assignment in encoded["level_assignments"]
    )
    assert all(len(assignment["candidate_codes"]) == 2 for assignment in encoded["level_assignments"])
    counts_by_depth = Counter(
        membership["depth"] for membership in encoded["prefix_memberships"]
    )
    assert counts_by_depth == {1: 2, 2: 4, 3: 8}
    assert sum(
        1
        for membership in encoded["prefix_memberships"]
        if membership["depth"] == 3 and membership["is_primary_prefix"]
    ) == 1

    for membership in encoded["prefix_memberships"]:
        probability_product = 1.0
        for level, code in enumerate(membership["prefix"]):
            probability_product *= encoded["level_assignments"][level]["probabilities"][code - 1]
        assert membership["assignment_probability_product"] == pytest.approx(
            probability_product,
            rel=1e-12,
            abs=1e-15,
        )
        assert membership["membership_score"] == pytest.approx(
            encoded["gamma"] * probability_product,
            rel=1e-12,
            abs=1e-15,
        )


def test_rq_sparsification_keeps_primary_without_renormalizing_or_flooring():
    from app.services.context_graph import encode_rq_vector, rq_membership_score

    encoded = encode_rq_vector([0.0, 0.0], _two_way_rq_model(threshold=0.99))

    assert [
        len(assignment["candidate_codes"])
        for assignment in encoded["level_assignments"]
    ] == [1, 1, 1]
    assert len(encoded["prefix_memberships"]) == 3
    assert encoded["level_assignments"][0]["candidate_codes"][0]["probability"] == pytest.approx(0.5)
    assert encoded["prefix_memberships"][0]["membership_score"] == pytest.approx(
        encoded["gamma"] * 0.5
    )
    assert rq_membership_score(10.0, tau_r=0.1) < 0.2
    assert all(
        membership["membership_score"] < 0.2
        for membership in encoded["prefix_memberships"]
    )


def test_rq_encoding_hash_is_deterministic_and_parameter_sensitive():
    from app.services.context_graph import encode_rq_vector, encode_rq_vectors_batch

    first = encode_rq_vector([0.0, 0.0], _two_way_rq_model(threshold=0.0))
    second = encode_rq_vector([0.0, 0.0], _two_way_rq_model(threshold=0.0))
    pruned = encode_rq_vector([0.0, 0.0], _two_way_rq_model(threshold=0.99))

    assert first["membership_encoding_hash"] == second["membership_encoding_hash"]
    assert first["membership_encoding_hash"] != pruned["membership_encoding_hash"]
    batched, batch_count = encode_rq_vectors_batch(
        [("chunk-b", [0.1, 0.0]), ("chunk-a", [0.0, 0.0])],
        _two_way_rq_model(threshold=0.0),
        batch_size=1,
    )
    assert list(batched) == ["chunk-a", "chunk-b"]
    assert batch_count == 2
    with pytest.raises(ValueError, match="duplicate chunk ids"):
        encode_rq_vectors_batch(
            [("chunk-a", [0.0, 0.0]), ("chunk-a", [0.1, 0.0])],
            _two_way_rq_model(threshold=0.0),
        )


def test_rq_candidate_score_uses_exact_fuzzy_prefix_not_hard_lcp_as_score():
    from app.services.context_graph import encode_rq_vector, rq_candidate_score

    encoded = encode_rq_vector([0.0, 0.0], _two_way_rq_model())
    query_rq = {
        **encoded,
        "prefix_memberships": encoded["prefix_memberships"],
    }
    exact_prefix = max(
        (
            membership
            for membership in encoded["prefix_memberships"]
            if membership["depth"] == 3
        ),
        key=lambda membership: membership["membership_score"],
    )
    absent_prefix = [9, 9, 9]
    candidate_residual = encoded["residual_vector"]
    exact = SimpleNamespace(
        rq_path=exact_prefix["prefix"],
        membership_score=exact_prefix["membership_score"],
        residual_norm=encoded["residual_norm"],
        membership_reason="rq_leaf",
        membership_role="primary_member",
        membership_entropy=exact_prefix["membership_entropy"],
        rank=1,
        diagnostics_json={"residual_vector": candidate_residual},
    )
    absent = SimpleNamespace(
        rq_path=absent_prefix,
        membership_score=exact_prefix["membership_score"],
        residual_norm=encoded["residual_norm"],
        membership_reason="rq_leaf",
        membership_role="fuzzy_member",
        membership_entropy=exact_prefix["membership_entropy"],
        rank=1,
        diagnostics_json={"residual_vector": candidate_residual},
    )

    exact_score = rq_candidate_score(query_rq, exact)
    absent_score = rq_candidate_score(query_rq, absent)

    assert exact_score["query_prefix_membership_score"] > 0.0
    assert absent_score["query_prefix_membership_score"] == 0.0
    assert exact_score["rq_score"] > absent_score["rq_score"]
    assert exact_score["hard_path_lcp_used_as_score"] is False


@pytest.mark.parametrize(
    ("overrides", "expected_role"),
    [
        ({"is_primary_leaf": True}, "primary_member"),
        ({}, "fuzzy_member"),
        ({"membership_entropy": 0.9}, "boundary_member"),
        ({"boundary_distance": 0.01}, "boundary_member"),
        ({"is_bridge_chunk": True}, "bridge_member"),
        ({"gamma": 0.3}, "low_confidence_member"),
        (
            {"gamma": 0.2, "residual_norm": 3.0, "residual_outlier_threshold": 2.0},
            "outlier_member",
        ),
        ({"membership_score": 1e-10}, "noise_candidate"),
    ],
)
def test_rq_membership_role_protocol_covers_every_role(overrides, expected_role):
    from app.services.context_graph import (
        RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION,
        classify_rq_membership_role,
        rq_membership_role_protocol_hash,
    )

    inputs = {
        "membership_score": 0.8,
        "rank": 1,
        "membership_entropy": 0.1,
        "residual_norm": 0.1,
        "gamma": 0.9,
        "boundary_probability_margin": 0.5,
        "boundary_distance": 0.5,
        "residual_outlier_threshold": 2.0,
        "is_primary_leaf": False,
        "is_bridge_chunk": False,
    }
    evaluation = classify_rq_membership_role(**{**inputs, **overrides})

    assert evaluation["role"] == expected_role
    assert evaluation["protocol_version"] == RQ_MEMBERSHIP_ROLE_PROTOCOL_VERSION
    assert evaluation["protocol_hash"] == rq_membership_role_protocol_hash()
    assert evaluation["model_call_count"] == 0


def test_rq_membership_role_precedence_retains_all_simultaneous_matches():
    from app.services.context_graph import classify_rq_membership_role

    evaluation = classify_rq_membership_role(
        membership_score=0.0,
        rank=2,
        membership_entropy=0.9,
        residual_norm=3.0,
        gamma=0.2,
        boundary_probability_margin=0.01,
        boundary_distance=0.01,
        residual_outlier_threshold=2.0,
        is_primary_leaf=True,
        is_bridge_chunk=True,
    )

    assert evaluation["role"] == "noise_candidate"
    assert evaluation["matched_flags"] == [
        "noise_candidate",
        "outlier_member",
        "bridge_member",
        "low_confidence_member",
        "boundary_member",
        "primary_member",
        "fuzzy_member",
    ]
    assert evaluation["inputs"]["rank"] == 2
    assert evaluation["model_call_count"] == 0


@pytest.mark.asyncio
async def test_rq_build_persists_multi_memberships_and_mass(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import func, select

    from app.models import (
        ChunkRelationGraphState,
        MidConcept,
        RQPrefix,
        RQPrefixDiagnostic,
        RQPrefixMembership,
    )
    from app.services.context_graph import concept_packet_for_cluster

    knowledge_base = populated_context_graph["knowledge_base"]
    relation_state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert relation_state is not None
    diagnostics = (relation_state.diagnostics_json or {})["rq_membership"]
    assert diagnostics["membership_protocol_version"] == "rq_fuzzy_softmax_gamma_product_v1"
    assert diagnostics["artificial_membership_floor"] is False
    assert diagnostics["renormalized_after_sparsification"] is False
    assert diagnostics["membership_write_batch_count"] == 1
    assert diagnostics["full_softmax_normalization_max_error"] < 1e-12
    assert diagnostics["membership_hash"]

    membership_counts = list(
        db_session.execute(
            select(RQPrefixMembership.chunk_id, func.count(RQPrefixMembership.id))
            .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
            .where(RQPrefix.graph_state_id == relation_state.id)
            .group_by(RQPrefixMembership.chunk_id)
        ).all()
    )
    assert membership_counts
    assert any(count > 3 for _chunk_id, count in membership_counts)
    rows = list(
        db_session.scalars(
            select(RQPrefixMembership)
            .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
            .where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    assert all((row.diagnostics_json or {}).get("artificial_membership_floor") is False for row in rows)
    assert all(0.0 <= row.membership_score <= 1.0 for row in rows)
    assert any(
        row.membership_role != "primary_member" and len(row.rq_path or []) == 3
        for row in rows
    )
    assert diagnostics["membership_role_protocol_hash"]
    assert sum(diagnostics["membership_role_counts"].values()) == len(rows)
    assert diagnostics["membership_entropy_distribution"]["count"] == len(rows)
    assert diagnostics["boundary_probability_margin_distribution"]["count"] == len(rows)
    assert all(
        (row.diagnostics_json or {}).get("membership_role_evaluation", {}).get(
            "model_call_count"
        )
        == 0
        for row in rows
    )

    diagnostic_mass = {
        row.rq_prefix_id: row.support_membership_mass
        for row in db_session.scalars(
            select(RQPrefixDiagnostic).where(
                RQPrefixDiagnostic.graph_state_id == relation_state.id,
                RQPrefixDiagnostic.diagnostic_type == "membership_mass",
            )
        ).all()
    }
    for prefix_id, expected_mass in db_session.execute(
        select(
            RQPrefixMembership.rq_prefix_id,
            func.sum(RQPrefixMembership.membership_score),
        )
        .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
        .where(RQPrefix.graph_state_id == relation_state.id)
        .group_by(RQPrefixMembership.rq_prefix_id)
    ).all():
        assert diagnostic_mass[prefix_id] == pytest.approx(expected_mass)

    mid_concept = db_session.scalar(
        select(MidConcept)
        .where(
            MidConcept.knowledge_base_id == knowledge_base.id,
            MidConcept.state == "active",
            MidConcept.support_rq_l3_prefix_id.is_not(None),
        )
        .order_by(MidConcept.id)
    )
    assert mid_concept is not None
    l3_prefix = db_session.get(
        RQPrefix,
        mid_concept.support_rq_l3_prefix_id,
    )
    assert l3_prefix is not None and l3_prefix.rq_level == 3
    packet = concept_packet_for_cluster(db_session, l3_prefix)
    l3_membership_count = db_session.scalar(
        select(func.count(RQPrefixMembership.id)).where(
            RQPrefixMembership.rq_prefix_id == l3_prefix.id
        )
    )
    assert sum(packet["membership_role_distribution"].values()) == l3_membership_count
    assert packet["membership_role_protocol_hash"] == diagnostics[
        "membership_role_protocol_hash"
    ]
    assert packet["membership_model_call_count"] == 0
    assert mid_concept.internal_state_json["membership_role_distribution"] == packet[
        "membership_role_distribution"
    ]
    assert mid_concept.internal_state_json["membership_role_protocol_hash"] == packet[
        "membership_role_protocol_hash"
    ]


@pytest.mark.asyncio
async def test_rq_membership_and_relation_hashes_are_rebuild_deterministic(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import Chunk, ChunkRelationGraphState, RQPrefix
    from app.services.context_graph import build_chunk_relation_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    active_state = db_session.scalar(
        select(ChunkRelationGraphState).where(
            ChunkRelationGraphState.knowledge_base_id == knowledge_base.id,
            ChunkRelationGraphState.state == "active",
        )
    )
    assert active_state is not None
    chunks = list(
        db_session.scalars(
            select(Chunk)
            .where(
                Chunk.knowledge_base_id == knowledge_base.id,
                Chunk.state == "active",
            )
            .order_by(Chunk.id)
        ).all()
    )
    first = build_chunk_relation_graph(
        db_session,
        knowledge_base.id,
        chunks,
        state_scope="active",
        operating_point=dict(active_state.graph_operating_point_json or {}),
        shadow_metadata={"determinism_probe": "first"},
    )
    db_session.flush()
    first_prefix_ids = {
        prefix.id
        for prefix in db_session.scalars(
            select(RQPrefix).where(RQPrefix.graph_state_id == first.id)
        ).all()
    }
    second = build_chunk_relation_graph(
        db_session,
        knowledge_base.id,
        chunks,
        state_scope="active",
        operating_point=dict(active_state.graph_operating_point_json or {}),
        shadow_metadata={"determinism_probe": "second"},
    )
    db_session.flush()
    second_prefix_ids = {
        prefix.id
        for prefix in db_session.scalars(
            select(RQPrefix).where(RQPrefix.graph_state_id == second.id)
        ).all()
    }

    assert first.id != second.id
    assert first_prefix_ids.isdisjoint(second_prefix_ids)
    assert first.state_hash == second.state_hash
    assert first.diagnostics_json["rq_prefix_facts_hash"] == second.diagnostics_json["rq_prefix_facts_hash"]
    assert first.diagnostics_json["rq_membership_hash"] == second.diagnostics_json["rq_membership_hash"]
    assert (
        first.diagnostics_json["rq_prefix_pair_diagnostics_hash"]
        == second.diagnostics_json["rq_prefix_pair_diagnostics_hash"]
    )
    assert (
        first.diagnostics_json["rq_prefix_pair_diagnostics"][
            "diagnostic_count_by_type"
        ]
        == second.diagnostics_json["rq_prefix_pair_diagnostics"][
            "diagnostic_count_by_type"
        ]
    )
