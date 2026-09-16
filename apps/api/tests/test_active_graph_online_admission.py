import pytest
from sqlalchemy import delete, select

from app.models import ChunkRelationEdge, ContextGraphFreshness
from app.services.context_graph import (
    ActiveContextGraphAdmissionError,
    active_graph_admission_gate,
    active_graph_online_admission_gate,
)


def test_online_admission_replays_the_same_active_state_without_full_fact_decode(
    db_session,
    populated_context_graph,
):
    knowledge_base_id = populated_context_graph["knowledge_base"].id
    deep = active_graph_admission_gate(db_session, knowledge_base_id)
    online = active_graph_online_admission_gate(db_session, knowledge_base_id)
    assert online.id == deep.id
    assert online.context_graph_hash == deep.context_graph_hash


def test_online_admission_rejects_a_stale_freshness_row(
    db_session,
    populated_context_graph,
):
    state = populated_context_graph["state"]
    row = db_session.scalar(
        select(ContextGraphFreshness).where(
            ContextGraphFreshness.context_graph_state_id == state.id
        ).limit(1)
    )
    row.is_stale = True
    row.stale_reasons_json = ["unit_test_stale"]
    db_session.commit()

    with pytest.raises(ActiveContextGraphAdmissionError):
        active_graph_online_admission_gate(
            db_session,
            populated_context_graph["knowledge_base"].id,
        )


def test_online_admission_rejects_relation_row_count_drift(
    db_session,
    populated_context_graph,
):
    state = populated_context_graph["state"]
    edge = db_session.scalar(
        select(ChunkRelationEdge).where(
            ChunkRelationEdge.graph_state_id == state.chunk_relation_graph_state_id
        ).limit(1)
    )
    assert edge is not None
    db_session.execute(delete(ChunkRelationEdge).where(ChunkRelationEdge.id == edge.id))
    db_session.commit()

    with pytest.raises(
        ActiveContextGraphAdmissionError,
        match="online_count_mismatch",
    ):
        active_graph_online_admission_gate(
            db_session,
            populated_context_graph["knowledge_base"].id,
        )
