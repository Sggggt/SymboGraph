from collections import defaultdict

import pytest
from sqlalchemy import event, select

from app.models import ChunkStructureMapping, ChunkStructureNode
from app.services.retrieval_feature_adapter import _roles, _structure_roles
from app.services.qa_performance import QAPerformance, qa_stage


@pytest.mark.asyncio
async def test_role_projection_matches_complete_structure_rows(db_session, populated_context_graph):
    full = defaultdict(list)
    for mapping, node in db_session.execute(select(ChunkStructureMapping, ChunkStructureNode)
        .join(ChunkStructureNode, ChunkStructureNode.id == ChunkStructureMapping.structure_node_id)):
        full[mapping.chunk_id].append(node)
    assert full
    expected = {cid: _roles(nodes) for cid, nodes in full.items()}
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(db_session.bind, 'before_cursor_execute', capture)
    try:
        actual = _structure_roles(db_session, list(full))
    finally:
        event.remove(db_session.bind, 'before_cursor_execute', capture)
    assert actual == expected
    assert len(statements) == 1
    columns = statements[0].split('FROM', 1)[0]
    assert 'chunk_structure_nodes.title' in columns and 'chunk_structure_nodes.node_type' in columns
    assert 'metadata_json' not in columns and 'text_content' not in columns
    assert _structure_roles(db_session, []) == {}


def test_feature_preparation_contains_kernel_without_double_counting():
    now = [0.0]
    recorder = QAPerformance(clock=lambda: now[0])
    with recorder.activate(), qa_stage('feature_preparation'):
        now[0] = 1.0
        with qa_stage('path_features'):
            now[0] = 3.0
        now[0] = 4.0
    summary = recorder.snapshot()
    assert summary.stages['feature_preparation'].active_wall_ms == 4000
    assert summary.stages['feature_preparation'].exclusive_ms == 2000
    assert summary.spans[1].parent_sequence == summary.spans[0].sequence
