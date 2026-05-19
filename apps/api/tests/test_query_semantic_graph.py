from app.models import Concept, ConceptRelation
from app.services.concept_graph import get_query_semantic_graph_payload


def test_query_semantic_graph_uses_relations_grounded_to_retrieved_chunks(db_session, sample_course, indexed_chunks):
    _document, chunks = indexed_chunks
    source = Concept(
        course_id=sample_course.id,
        canonical_name="DAG",
        normalized_name="dag",
        concept_type="concept",
        chapter_refs=["L3"],
        importance_score=0.8,
        evidence_count=1,
    )
    target = Concept(
        course_id=sample_course.id,
        canonical_name="Topological sort",
        normalized_name="topological sort",
        concept_type="algorithm",
        chapter_refs=["L3"],
        importance_score=0.7,
        evidence_count=1,
    )
    db_session.add_all([source, target])
    db_session.flush()
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=source.id,
            target_concept_id=target.id,
            target_name=target.canonical_name,
            relation_type="implemented_by",
            evidence_chunk_id=chunks[0].id,
            confidence=0.91,
            weight=0.82,
            relation_source="llm",
            metadata_json={"hard_gate": "accepted"},
        )
    )
    db_session.commit()

    graph = get_query_semantic_graph_payload(db_session, sample_course.id, [chunks[0].id])

    assert {node["id"] for node in graph["nodes"]} >= {
        f"semantic:{source.id}",
        f"semantic:{target.id}",
        f"evidence_chunk:{chunks[0].id}",
    }
    assert any(edge["category"] == "semantic" and edge["label"] == "implemented_by" for edge in graph["edges"])
    assert any(edge["category"] == "evidence" and edge["label"] == "evidenced_by" for edge in graph["edges"])


def test_query_semantic_graph_expands_from_query_concept_when_retrieved_chunk_has_no_relation(db_session, sample_course, indexed_chunks):
    _document, chunks = indexed_chunks
    source = Concept(
        course_id=sample_course.id,
        canonical_name="DAG",
        normalized_name="dag",
        concept_type="concept",
        chapter_refs=["L3"],
        importance_score=0.8,
        evidence_count=1,
    )
    target = Concept(
        course_id=sample_course.id,
        canonical_name="Topological sort",
        normalized_name="topological sort",
        concept_type="algorithm",
        chapter_refs=["L3"],
        importance_score=0.7,
        evidence_count=1,
    )
    db_session.add_all([source, target])
    db_session.flush()
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=source.id,
            target_concept_id=target.id,
            target_name=target.canonical_name,
            relation_type="implemented_by",
            evidence_chunk_id=chunks[0].id,
            confidence=0.91,
            weight=0.82,
            relation_source="llm",
            metadata_json={"hard_gate": "accepted"},
        )
    )
    db_session.commit()

    graph = get_query_semantic_graph_payload(db_session, sample_course.id, [chunks[1].id], query="DAG")

    assert f"evidence_chunk:{chunks[0].id}" in {node["id"] for node in graph["nodes"]}
    assert any(edge["category"] == "semantic" and edge["label"] == "implemented_by" for edge in graph["edges"])
