from __future__ import annotations

from itertools import permutations


def _label(*, distance, facets, roles, depth, edge_types):
    from app.services.context_graph import _traversal_label_from_state

    return _traversal_label_from_state(
        {
            "distance_so_far": distance,
            "covered_facets": facets,
            "evidence_roles": roles,
            "depth": depth,
            "path_edge_types": edge_types,
        }
    )


def test_multi_label_dominance_requires_actual_facet_and_role_supersets():
    from app.services.context_graph import _traversal_label_dominates

    facet_a = _label(distance=0.5, facets=["facet-a"], roles=["role-a"], depth=1, edge_types=["edge-a"])
    facet_b = _label(distance=0.5, facets=["facet-b"], roles=["role-a"], depth=1, edge_types=["edge-a"])
    role_b = _label(distance=0.5, facets=["facet-a"], roles=["role-b"], depth=1, edge_types=["edge-a"])

    assert not _traversal_label_dominates(facet_a, facet_b)
    assert not _traversal_label_dominates(facet_b, facet_a)
    assert not _traversal_label_dominates(facet_a, role_b)
    assert not _traversal_label_dominates(role_b, facet_a)


def test_multi_label_dominance_requires_path_edge_type_multiset_superset():
    from app.services.context_graph import _traversal_label_dominates

    edge_a = _label(distance=0.5, facets=["facet"], roles=["role", "edge-a"], depth=1, edge_types=["edge-a"])
    edge_b = _label(distance=0.5, facets=["facet"], roles=["role", "edge-b"], depth=1, edge_types=["edge-b"])
    stronger = _label(
        distance=0.4,
        facets=["facet", "extra-facet"],
        roles=["role", "edge-a", "extra-role"],
        depth=1,
        edge_types=["edge-a", "edge-a"],
    )

    assert not _traversal_label_dominates(edge_a, edge_b)
    assert not _traversal_label_dominates(edge_b, edge_a)
    assert _traversal_label_dominates(stronger, edge_a)
    assert not _traversal_label_dominates(edge_a, stronger)


def test_multi_label_dominance_uses_raw_path_distance_and_depth():
    from app.services.context_graph import _traversal_label_dominates

    baseline = _label(distance=0.5, facets=["facet"], roles=["role"], depth=1, edge_types=[])
    farther = _label(distance=0.6, facets=["facet"], roles=["role"], depth=1, edge_types=[])
    deeper = _label(distance=0.5, facets=["facet"], roles=["role"], depth=2, edge_types=[])

    assert _traversal_label_dominates(baseline, farther)
    assert _traversal_label_dominates(baseline, deeper)
    assert not _traversal_label_dominates(farther, baseline)
    assert not _traversal_label_dominates(deeper, baseline)


def test_equal_pareto_vectors_do_not_dominate_distinct_physical_paths():
    from app.services.context_graph import (
        _traversal_label_dominates,
    )

    left = _label(
        distance=0.5,
        facets=["facet"],
        roles=["role"],
        depth=1,
        edge_types=["same-type"],
    )
    right = _label(
        distance=0.5,
        facets=["facet"],
        roles=["role"],
        depth=1,
        edge_types=["same-type"],
    )

    assert not _traversal_label_dominates(left, right)
    assert not _traversal_label_dominates(right, left)


def test_label_cap_counts_every_real_eviction_in_all_arrival_orders():
    from app.services.context_graph import (
        _admit_non_dominated_traversal_state,
        _traversal_path_identity,
    )

    states = [
        {
            "layer": "mid",
            "node_id": "shared",
            "root_node_id": f"root-{name}",
            "path": [f"root-{name}", "shared"],
            "path_edge_ids": [f"edge-{name}"],
            "path_edge_types": ["same-type"],
            "distance_so_far": distance,
            "reward_so_far": 0.0,
            "covered_facets": [facet],
            "evidence_roles": [role],
            "depth": 1,
            "entry_parent_refs": [],
        }
        for name, distance, facet, role in (
            ("a", 0.1, "f1", "r1"),
            ("b", 0.2, "f2", "r2"),
            ("c", 0.3, "f3", "r3"),
            ("d", 0.4, "f4", "r4"),
        )
    ]
    expected_retained: list[str] | None = None
    for ordered_states in permutations(states):
        labels_by_node: dict[str, list[dict]] = {
            "shared": []
        }
        dominance_count = 0
        cap_count = 0
        for state in ordered_states:
            admitted, dominance_delta, cap_delta = (
                _admit_non_dominated_traversal_state(
                    labels_by_node,
                    state=state,
                    queue_key=(
                        len(
                            {"f1", "f2", "f3", "f4"}
                            - set(state["covered_facets"])
                        ),
                        state["distance_so_far"],
                        state["depth"],
                        -len(state["evidence_roles"]),
                    ),
                    required_facets={
                        "f1",
                        "f2",
                        "f3",
                        "f4",
                    },
                    max_labels=2,
                )
            )
            assert isinstance(admitted, bool)
            dominance_count += dominance_delta
            cap_count += cap_delta
        retained = sorted(
            _traversal_path_identity(entry["state"])
            for entry in labels_by_node["shared"]
        )
        assert dominance_count == 0
        assert cap_count == 2
        if expected_retained is None:
            expected_retained = retained
        assert retained == expected_retained
