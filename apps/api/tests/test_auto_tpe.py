from __future__ import annotations

import pytest


def _active_chunks_and_vectors(db_session, knowledge_base_id: str):
    from sqlalchemy import select

    from app.models import Chunk
    from app.services.context_graph import vector_for_chunk

    chunks = list(db_session.scalars(select(Chunk).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active")).all())
    vectors = {chunk.id: vector_for_chunk(db_session, chunk.id) for chunk in chunks}
    return chunks, vectors


def test_auto_tpe_skips_when_disabled_or_same_chunk_version(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import AutoTpeRun
    from app.services.auto_tpe import select_auto_tpe_operating_point
    from app.services.context_graph import dense_graph_operating_point

    kb = populated_context_graph["knowledge_base"]
    chunks, vectors = _active_chunks_and_vectors(db_session, kb.id)
    base_theta = dense_graph_operating_point()

    selected, context = select_auto_tpe_operating_point(
        db_session,
        knowledge_base_id=kb.id,
        batch_id="unit-batch-disabled",
        chunks=chunks,
        vectors=vectors,
        fallback_operating_point=base_theta,
        chunk_version_incremented=True,
    )
    assert selected == base_theta
    assert context["status"] == "skipped"
    assert context["reason"] == "disabled"
    assert db_session.scalar(select(func.count(AutoTpeRun.id))) == 0


@pytest.mark.asyncio
async def test_auto_tpe_runs_once_before_active_relation_graph_write(db_session, populated_context_graph, monkeypatch):
    from sqlalchemy import func, select

    from app.core.config import get_settings
    from app.models import AutoTpeRun, AutoTpeTrial, ChunkRelationGraphState, ContextGraphState
    from app.services.context_graph import rebuild_context_graph

    monkeypatch.setenv("ENABLE_AUTO_TPE", "true")
    monkeypatch.setenv("TPE_TRIAL_BUDGET", "2")
    monkeypatch.setenv("TPE_STARTUP_RANDOM_TRIALS", "1")
    monkeypatch.setenv("TPE_PROBE_QUERY_BUDGET", "2")
    monkeypatch.setenv("TPE_TRIAL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("TPE_CANDIDATE_POOL_SIZE", "4")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY", "1000")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO", "1000")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE", "0")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS", "600000")
    get_settings.cache_clear()

    kb = populated_context_graph["knowledge_base"]
    previous_relation_state_count = db_session.scalar(select(func.count(ChunkRelationGraphState.id)))
    state = await rebuild_context_graph(db_session, kb.id, batch_id="unit-batch-auto-tpe", chunk_version_incremented=True)
    db_session.flush()

    run = db_session.scalar(select(AutoTpeRun).where(AutoTpeRun.knowledge_base_id == kb.id).order_by(AutoTpeRun.created_at.desc()))
    assert run is not None
    assert run.status == "completed"
    assert run.selected_theta_hash
    assert run.best_trial_id
    assert run.chunk_relation_graph_state_id == state.chunk_relation_graph_state_id
    assert db_session.scalar(select(func.count(AutoTpeTrial.id)).where(AutoTpeTrial.run_id == run.id)) == 2

    relation_state = db_session.get(ChunkRelationGraphState, state.chunk_relation_graph_state_id)
    assert relation_state is not None
    assert relation_state.auto_tpe_run_id == run.id
    assert relation_state.auto_tpe_best_trial_id == run.best_trial_id
    assert relation_state.graph_operating_point_hash == run.selected_theta_hash
    assert db_session.scalar(select(func.count(ChunkRelationGraphState.id))) == previous_relation_state_count + 1
    assert db_session.scalar(select(func.count(ContextGraphState.id)).where(ContextGraphState.state == "shadow")) == 0


def test_auto_tpe_hard_gate_failure_blocks_instead_of_silent_success(db_session, populated_context_graph, monkeypatch):
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.models import AutoTpeRun
    from app.services.auto_tpe import select_auto_tpe_operating_point
    from app.services.context_graph import dense_graph_operating_point

    monkeypatch.setenv("ENABLE_AUTO_TPE", "true")
    monkeypatch.setenv("TPE_TRIAL_BUDGET", "1")
    monkeypatch.setenv("TPE_STARTUP_RANDOM_TRIALS", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY", "0.0001")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO", "1")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO", "1000")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE", "0")
    monkeypatch.setenv("OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS", "600000")
    get_settings.cache_clear()

    kb = populated_context_graph["knowledge_base"]
    chunks, vectors = _active_chunks_and_vectors(db_session, kb.id)

    with pytest.raises(RuntimeError, match="自动 TPE 没有 trial 通过 hard gate"):
        select_auto_tpe_operating_point(
            db_session,
            knowledge_base_id=kb.id,
            batch_id="unit-batch-auto-tpe-fail",
            chunks=chunks,
            vectors=vectors,
            fallback_operating_point=dense_graph_operating_point(),
            chunk_version_incremented=True,
        )

    run = db_session.scalar(select(AutoTpeRun).where(AutoTpeRun.knowledge_base_id == kb.id).order_by(AutoTpeRun.created_at.desc()))
    assert run is not None
    assert run.status == "failed"
    assert run.failure_code == "no_trial_passed_hard_gate"
    assert "no_trial_passed_hard_gate" in (run.blocking_reasons_json or [])
