from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError


def test_agent_trace_sequence_is_contiguous_public_and_unique(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import AgentRun, AgentTraceEvent
    from app.services.agent_graph import trace, trace_event_to_payload

    run = AgentRun(
        knowledge_base_id=sample_knowledge_base.id,
        question="sequence audit",
        status="running",
        route="layered_context_graph",
    )
    db_session.add(run)
    db_session.commit()

    first = trace(db_session, run.id, "query_understanding")
    second = trace(db_session, run.id, "query_facet_extraction")
    assert first["sequence_index"] == 0
    assert second["sequence_index"] == 1

    rows = list(
        db_session.scalars(
            select(AgentTraceEvent)
            .where(AgentTraceEvent.run_id == run.id)
            .order_by(AgentTraceEvent.sequence_index.asc())
        ).all()
    )
    assert [row.sequence_index for row in rows] == [0, 1]
    assert [trace_event_to_payload(row)["sequence_index"] for row in rows] == [
        0,
        1,
    ]

    db_session.add(
        AgentTraceEvent(
            run_id=run.id,
            sequence_index=1,
            node="agent_planner",
            status="completed",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()
