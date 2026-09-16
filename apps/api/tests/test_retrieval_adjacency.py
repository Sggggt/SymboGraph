from collections import defaultdict

import pytest
from sqlalchemy import select

from app.models import ChunkRelationEdge
from app.services.retrieval_adjacency import CompleteChunkAdjacency
from app.services.qa_performance import QAPerformance


@pytest.mark.asyncio
async def test_complete_incident_reads_match_full_edge_domain_and_do_not_cache_partial_neighbors(
    db_session, populated_context_graph,
):
    edges = list(db_session.scalars(select(ChunkRelationEdge)))
    assert edges
    state_id = edges[0].graph_state_id
    allowed = {edges[0].edge_type}
    expected = defaultdict(list)
    for edge in edges:
        if edge.graph_state_id == state_id and edge.edge_type in allowed:
            expected[edge.source_chunk_id].append(edge.id)
            expected[edge.target_chunk_id].append(edge.id)
    adjacency = CompleteChunkAdjacency(db_session, graph_state_id=state_id, allowed_types=allowed)
    first = next(iter(expected))
    performance = QAPerformance()
    with performance.activate():
        adjacency.preload([first])
        assert set(adjacency._complete) == {first}
        for node, ids in expected.items():
            assert sorted(edge.id for edge in adjacency.get(node)) == sorted(ids)
        queries = adjacency.query_count
        adjacency.preload(expected)
        assert adjacency.query_count == queries
    assert performance.snapshot().stages["graph_edge_read"].success_count == queries
    assert sum(span.fields.item_count for span in performance.snapshot().spans) == adjacency.rows_read
    assert CompleteChunkAdjacency(db_session, graph_state_id="unit-test-absent", allowed_types=allowed).get(first) == ()
    assert CompleteChunkAdjacency(db_session, graph_state_id=state_id, allowed_types=()).get(first) == ()


def test_cancelled_read_never_marks_incomplete_batch_as_complete(monkeypatch):
    from app.services import retrieval_adjacency as module
    adjacency = CompleteChunkAdjacency(None, graph_state_id="unit-test-graph", allowed_types=["unit-test-edge"])
    def cancel():
        raise InterruptedError("unit-test cancelled")
    monkeypatch.setattr(module, "raise_if_source_io_cancelled", cancel)
    with pytest.raises(InterruptedError):
        adjacency.get("unit-test-node")
    assert not adjacency._complete and adjacency.query_count == 0
