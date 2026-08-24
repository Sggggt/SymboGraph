from app.services.context_graph import (
    _select_concept_eligibility_cards,
    concept_node_eligibility_budget,
)


def _card(
    index: int,
    *,
    primary: list[str],
    mass: float,
) -> dict[str, object]:
    return {
        "prefix_id": f"prefix-{index}",
        "prefix_key": f"rq:L3:{index}",
        "primary_support_ids": primary,
        "membership_mass": mass,
        "primary_membership_mass": mass if primary else 0.0,
        "support_edge_count": len(primary),
        "primary_support_edge_count": len(primary),
        "admissible": True,
        "ineligible_reasons": [],
        "business_fact": {
            "prefix_key": f"rq:L3:{index}",
            "primary": primary,
            "mass": mass,
        },
    }


def test_mid_eligibility_compresses_primary_candidates_without_llm():
    cards = [
        _card(
            index,
            primary=[f"chunk-{index % 17}"],
            mass=1.0 / (index + 1),
        )
        for index in range(52)
    ]

    selected, audit = _select_concept_eligibility_cards(
        cards,
        layer="mid",
        source_count=17,
    )

    assert concept_node_eligibility_budget("mid", 17) == 5
    assert len(selected) == 5
    assert audit["candidate_count"] == 52
    assert audit["eligible_count"] == 5
    assert audit["compression_rate"] == 1.0 - 5 / 17
    assert audit["llm_eligibility_authority"] is False
    assert audit["primary_only_selection_authority"] is True
    assert audit["model_call_count"] == 0


def test_candidate_without_primary_support_is_hard_ineligible():
    unsupported = _card(
        1,
        primary=[],
        mass=100.0,
    )
    primary = _card(
        2,
        primary=["chunk-1"],
        mass=0.01,
    )

    selected, audit = _select_concept_eligibility_cards(
        [unsupported, primary],
        layer="mid",
        source_count=3,
    )

    assert selected == ["prefix-2"]
    assert audit["admissible_candidate_count"] == 1
    assert audit["ineligible_reason_counts"] == {"no_primary_support": 1}


def test_coarse_eligibility_is_deterministic_and_never_exceeds_mid_count():
    cards = [
        _card(
            index,
            primary=[f"mid-{index}"],
            mass=float(10 - index),
        )
        for index in range(5)
    ]

    first, first_audit = _select_concept_eligibility_cards(
        cards,
        layer="coarse",
        source_count=5,
    )
    second, second_audit = _select_concept_eligibility_cards(
        list(reversed(cards)),
        layer="coarse",
        source_count=5,
    )

    assert concept_node_eligibility_budget("coarse", 5) == 3
    assert first == second
    assert first_audit["audit_hash"] == second_audit["audit_hash"]
    assert len(first) == 3 < 5
    assert first_audit["model_call_count"] == 0
