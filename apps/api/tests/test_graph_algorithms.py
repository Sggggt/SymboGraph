from __future__ import annotations

from types import SimpleNamespace


def signal(concept_id: str, evidence_count: int, refs: tuple[str, ...] = ("L1",), vector: tuple[float, ...] | None = None):
    from app.services.graph_algorithms import ConceptSignal

    return ConceptSignal(
        concept_id=concept_id,
        importance=0.7,
        evidence_count=evidence_count,
        chapter_refs=refs,
        vector=vector,
    )


def test_dynamic_rnn_knn_edges_stay_sparse():
    from app.services.graph_algorithms import build_sparse_edges, dynamic_k

    concepts = [
        signal(f"c{index}", index % 6 + 1, vector=(1.0, index / 30, 0.0))
        for index in range(30)
    ]

    edges = build_sparse_edges(concepts, [])

    assert len(edges) <= sum(dynamic_k(concept.evidence_count) for concept in concepts)
    assert len(edges) < 30 * 12
    assert all(0.0 <= edge.weight <= 1.0 for edge in edges)


def test_analyze_graph_assigns_communities_and_centrality():
    from app.services.graph_algorithms import RelationSignal, analyze_graph, build_sparse_edges

    concepts = [
        signal("a", 4, ("L1",), (1.0, 0.0)),
        signal("b", 3, ("L1",), (0.98, 0.02)),
        signal("c", 3, ("L2",), (0.0, 1.0)),
        signal("d", 2, ("L2",), (0.02, 0.98)),
    ]
    relations = [
        RelationSignal("a", "b", "used_for", 0.9, evidence_chunk_id="ab", metadata={"evidence_support": True}),
        RelationSignal("c", "d", "part_of", 0.9, evidence_chunk_id="cd", metadata={"evidence_support": True}),
        RelationSignal("b", "c", "prerequisite_of", 0.86, support_count=2, evidence_chunk_id="bc", metadata={"evidence_support": True}),
    ]

    edges = build_sparse_edges(concepts, relations)
    metrics, graph = analyze_graph(concepts, edges)

    assert graph.number_of_edges() >= 3
    assert set(metrics) == {"a", "b", "c", "d"}
    assert all("community_louvain" in item for item in metrics.values())
    assert all("centrality_score" in item["centrality"] for item in metrics.values())


def test_hard_gate_rejects_weak_unsupported_typed_relation():
    from app.services.graph_algorithms import RelationSignal, build_sparse_edges

    concepts = [
        signal("source", 2, ("L1",), (1.0, 0.0)),
        signal("target", 2, ("L1",), (0.0, 1.0)),
    ]
    relations = [RelationSignal("source", "target", "prerequisite_of", 0.6)]

    edges = build_sparse_edges(concepts, relations)

    assert edges == []


def test_hard_gate_accepts_direct_evidence_relation():
    from app.services.graph_algorithms import RelationSignal, build_sparse_edges

    concepts = [
        signal("source", 2, ("L1",), (1.0, 0.0)),
        signal("target", 2, ("L1",), (0.0, 1.0)),
    ]
    relations = [
        RelationSignal(
            "source",
            "target",
            "prerequisite_of",
            0.74,
            evidence_chunk_id="chunk-1",
            metadata={"evidence_source_match": True, "evidence_target_match": True},
        )
    ]

    edges = build_sparse_edges(concepts, relations)

    assert len(edges) == 1
    assert edges[0].metadata["hard_gate"] == "accepted"
    assert edges[0].metadata["evidence_source_match"] is True
    assert edges[0].metadata["evidence_target_match"] is True


def test_adaptive_relation_threshold_accepts_course_calibrated_confidence():
    from app.services.graph_algorithms import RelationSignal, adaptive_graph_quality_thresholds, relation_hard_gate

    concepts = [
        signal("source", 3, ("L1",), (1.0, 0.0)),
        signal("target", 3, ("L1",), (0.98, 0.02)),
    ]
    relations = [
        RelationSignal(
            "source",
            "target",
            "prerequisite_of",
            0.65,
            evidence_chunk_id=f"chunk-{index}",
            metadata={"evidence_source_match": True, "evidence_target_match": True},
        )
        for index in range(20)
    ]

    thresholds = adaptive_graph_quality_thresholds(concepts, relations)
    passed, reason = relation_hard_gate(
        relations[0],
        concepts[0],
        concepts[1],
        semantic=0.96,
        weight=0.88,
        cooccurrence=0.9,
        thresholds=thresholds,
    )

    assert thresholds.audit["relation_confidence"]["enabled"] is True
    assert thresholds.relation_confidence == 0.65
    assert passed is True
    assert reason == "passed"


def test_adaptive_concept_keep_bounds_scales_past_static_cap():
    from app.services.graph_algorithms import adaptive_graph_quality_thresholds, analyze_graph, select_concepts_to_keep

    concepts = [
        signal(f"n{index}", 2, (f"L{index % 20}",), (1.0, index / 500))
        for index in range(500)
    ]
    metrics, graph = analyze_graph(concepts, [])
    thresholds = adaptive_graph_quality_thresholds(concepts, [])
    keep = select_concepts_to_keep(concepts, metrics, graph, thresholds=thresholds)

    assert thresholds.concept_keep_max > 360
    assert len(keep) > 360


def test_dijkstra_infers_hidden_relation_and_marks_source():
    from app.services.graph_algorithms import WeightedEdge, analyze_graph, infer_dijkstra_edges

    concepts = [
        signal("a", 6, ("L1",), (1.0, 0.0, 0.0)),
        signal("b", 6, ("L1",), (0.96, 0.04, 0.0)),
        signal("c", 6, ("L1",), (0.93, 0.07, 0.0)),
    ]
    edges = [
        WeightedEdge("a", "b", "used_for", 0.85, 0.9, 2, 0.9, None, "llm", False, {}),
        WeightedEdge("b", "c", "used_for", 0.85, 0.9, 2, 0.9, None, "llm", False, {}),
    ]
    _metrics, graph = analyze_graph(concepts, edges)

    inferred = infer_dijkstra_edges(graph, concepts, edges)

    assert inferred
    assert inferred[0].relation_source == "dijkstra_inferred"
    assert inferred[0].is_inferred is True
    assert inferred[0].metadata["path"] == ["a", "b", "c"]


def test_dijkstra_respects_course_hyperparameter_threshold():
    from app.services.graph_algorithms import GraphHyperparameters, WeightedEdge, analyze_graph, infer_dijkstra_edges

    concepts = [
        signal("a", 6, ("L1",), (1.0, 0.0, 0.0)),
        signal("b", 6, ("L1",), (0.96, 0.04, 0.0)),
        signal("c", 6, ("L1",), (0.93, 0.07, 0.0)),
    ]
    edges = [
        WeightedEdge("a", "b", "used_for", 0.85, 0.9, 2, 0.9, None, "llm", False, {}),
        WeightedEdge("b", "c", "used_for", 0.85, 0.9, 2, 0.9, None, "llm", False, {}),
    ]
    params = GraphHyperparameters(dijkstra_semantic_threshold=1.0)
    _metrics, graph = analyze_graph(concepts, edges, hyperparameters=params)

    assert infer_dijkstra_edges(graph, concepts, edges, hyperparameters=params) == []


def test_analyze_graph_uses_supplied_rank_weights():
    from app.services.graph_algorithms import GraphHyperparameters, RelationSignal, analyze_graph, build_sparse_edges

    concepts = [
        signal("hub", 1, ("L1",), (1.0, 0.0)),
        signal("leaf", 9, ("L1",), (0.98, 0.02)),
        signal("other", 1, ("L1",), (0.97, 0.03)),
    ]
    relations = [
        RelationSignal("hub", "leaf", "used_for", 0.95, evidence_chunk_id="hl", metadata={"evidence_support": True}),
        RelationSignal("hub", "other", "used_for", 0.95, evidence_chunk_id="ho", metadata={"evidence_support": True}),
    ]
    edges = build_sparse_edges(concepts, relations)
    evidence_only = GraphHyperparameters(w_centrality=0.0, w_llm_importance=0.0, w_evidence=1.0, source="unit").normalized()
    metrics, _graph = analyze_graph(concepts, edges, hyperparameters=evidence_only)

    assert metrics["leaf"]["graph_rank_score"] > metrics["hub"]["graph_rank_score"]
    assert metrics["leaf"]["centrality"]["hyperparameter_source"] == "unit"


def test_load_course_graph_hyperparameters_uses_llm_and_embedding_key(db_session, sample_course):
    from app.models import CourseModelHyperparameter
    from app.services.graph_algorithms import build_course_model_hyperparameter_key, load_course_graph_hyperparameters

    model_key = build_course_model_hyperparameter_key("unit-llm", "unit-embedding", "unit-text-v1")
    db_session.add(
        CourseModelHyperparameter(
            course_id=sample_course.id,
            llm_model_name="unit-llm",
            embedding_model_name="unit-embedding",
            embedding_text_version="unit-text-v1",
            model_name=model_key,
            min_relation_confidence=0.51,
            dijkstra_semantic_threshold=0.66,
        )
    )
    db_session.commit()

    params = load_course_graph_hyperparameters(
        db_session,
        sample_course.id,
        llm_model_name="unit-llm",
        embedding_model_name="unit-embedding",
        embedding_text_version="unit-text-v1",
    )

    assert params.min_relation_confidence == 0.51
    assert params.dijkstra_semantic_threshold == 0.66
    assert params.source == f"course_model_hyperparameters:{model_key}"


def test_pruning_keeps_minimum_available_nodes():
    from app.services.graph_algorithms import analyze_graph, select_concepts_to_keep

    concepts = [signal(f"n{index}", 1, (), (1.0, 0.0)) for index in range(12)]
    metrics, graph = analyze_graph(concepts, [])

    keep = select_concepts_to_keep(concepts, metrics, graph)

    assert len(keep) == 12


def test_completion_relation_signals_ignores_bad_concept_shape():
    from app.services.graph_algorithms import completion_relation_signals

    degree = SimpleNamespace(id="c1", canonical_name="Degree Centrality")
    closeness = SimpleNamespace(id="c2", canonical_name="Closeness Centrality")
    signals = completion_relation_signals(
        {
            "concepts": ["Degree Centrality", "Closeness Centrality"],
            "relations": [
                {
                    "source": "Degree Centrality",
                    "target": "Closeness Centrality",
                    "relation_type": "compares_with",
                    "confidence": "0.82",
                }
            ],
        },
        {degree.canonical_name: degree, closeness.canonical_name: closeness},
    )

    assert len(signals) == 1
    assert signals[0].source_id == "c1"
    assert signals[0].target_id == "c2"
    assert signals[0].confidence == 0.82


def test_relation_evidence_metadata_matches_aliases():
    from app.services.graph_algorithms import relation_evidence_metadata

    source = SimpleNamespace(
        id="c1",
        canonical_name="Bayesian Inference",
        normalized_name="bayesian inference",
        aliases=[SimpleNamespace(alias="Posterior updating")],
    )
    target = SimpleNamespace(
        id="c2",
        canonical_name="Posterior Distribution",
        normalized_name="posterior distribution",
        aliases=[],
    )
    relation = SimpleNamespace(source_concept_id="c1", target_concept_id="c2", evidence_chunk_id="chunk-1")
    chunk = SimpleNamespace(
        section="Lecture",
        snippet="Posterior updating gives the posterior distribution after observing evidence.",
        content="",
    )

    metadata = relation_evidence_metadata(relation, {"c1": source, "c2": target}, {"chunk-1": chunk})

    assert metadata["evidence_source_match"] is True
    assert metadata["evidence_target_match"] is True
    assert metadata["evidence_endpoint_match_count"] == 2


def test_semantic_sparse_edges_write_candidate_store_not_default_graph(db_session, sample_course):
    from app.models import Concept, ConceptRelation, GraphRelationCandidate
    from app.services.graph_algorithms import WeightedEdge, upsert_weighted_edges

    source = Concept(
        course_id=sample_course.id,
        canonical_name="Residual Network",
        normalized_name="residual network",
        evidence_count=2,
        importance_score=0.8,
    )
    target = Concept(
        course_id=sample_course.id,
        canonical_name="Augmenting Path",
        normalized_name="augmenting path",
        evidence_count=2,
        importance_score=0.8,
    )
    db_session.add_all([source, target])
    db_session.commit()

    changed = upsert_weighted_edges(
        db_session,
        sample_course.id,
        [
            WeightedEdge(
                source_id=source.id,
                target_id=target.id,
                relation_type="related_to",
                weight=0.9,
                semantic_similarity=0.92,
                support_count=2,
                confidence=0.9,
                evidence_chunk_id=None,
                relation_source="semantic_sparse",
                is_inferred=True,
                metadata={"semantic_sparse": True, "candidate_only": True},
            )
        ],
    )
    db_session.commit()

    assert changed == 1
    assert db_session.query(ConceptRelation).filter(ConceptRelation.course_id == sample_course.id).count() == 0
    candidate = db_session.query(GraphRelationCandidate).filter(GraphRelationCandidate.course_id == sample_course.id).one()
    assert candidate.relation_source == "semantic_sparse"
    assert candidate.decision_json["action"] == "candidate_only"


def test_write_concept_metrics_prunes_candidate_edges_before_concepts(db_session, sample_course):
    from app.models import Concept, GraphRelationCandidate
    from app.services.graph_algorithms import write_concept_metrics

    kept = Concept(
        course_id=sample_course.id,
        canonical_name="Kept Concept",
        normalized_name="kept concept::concept",
        evidence_count=3,
        importance_score=0.8,
    )
    pruned = Concept(
        course_id=sample_course.id,
        canonical_name="Weak Concept",
        normalized_name="weak concept::concept",
        evidence_count=1,
        importance_score=0.1,
    )
    db_session.add_all([kept, pruned])
    db_session.flush()
    pruned_id = pruned.id
    db_session.add(
        GraphRelationCandidate(
            course_id=sample_course.id,
            source_concept_id=kept.id,
            target_concept_id=pruned.id,
            target_name=pruned.canonical_name,
            relation_type="related_to",
            relation_source="semantic_sparse",
            confidence=0.5,
            weight=0.5,
            decision_json={"action": "candidate_only"},
        )
    )
    db_session.commit()

    updated = write_concept_metrics(
        db_session,
        [kept, pruned],
        [signal(kept.id, 3), signal(pruned.id, 1)],
        {kept.id: {"centrality": {}, "graph_rank_score": 0.9}, pruned.id: {"centrality": {}, "graph_rank_score": 0.1}},
        {kept.id},
    )
    db_session.commit()
    db_session.expire_all()

    assert updated == 1
    assert db_session.get(Concept, pruned_id) is None
    assert db_session.query(GraphRelationCandidate).filter_by(course_id=sample_course.id).count() == 0
