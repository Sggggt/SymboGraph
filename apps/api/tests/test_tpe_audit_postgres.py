from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select


def _simulation_diagnostics(theta: dict[str, object]) -> dict[str, object]:
    from app.services.auto_tpe import theta_calibration_audit

    audit = theta_calibration_audit(theta)
    return {
        "relation_quota_signals": {
            "language_identity": {
                "protocol_version": "active_chunk_language_identity_scope_v1",
                "scope_hash": "5" * 64,
            },
            "signal_scope_hash": "6" * 64,
        },
        "edge_type_calibration": {
            "protocol_version": audit["edge_type_calibration_protocol"],
            "protocol_hash": audit["edge_type_calibration_protocol_hash"],
            "edge_distance_protocol_version": audit["edge_distance_protocol"],
            "edge_distance_protocol_hash": audit["edge_distance_protocol_hash"],
            "params": dict(audit["calibration_params"]),
            "calibration_params_hash": audit["calibration_params_hash"],
            "edge_type_calibration_config_hash": audit[
                "edge_type_calibration_config_hash"
            ],
        },
    }


def _gate_profile() -> dict[str, object]:
    return {
        "protocol": "tpe_hard_gate_profile_v2",
        "tpe_probe_query_budget": 1,
        "tpe_trial_timeout_seconds": 30.0,
        "hard_gate_thresholds": {
            "edge_density": {"direction": "max", "threshold": 1.0},
            "sparse_edge_budget_ratio": {
                "direction": "max",
                "threshold": 1.0,
            },
            "isolated_ratio": {"direction": "max", "threshold": 1.0},
            "hubness_ratio": {"direction": "max", "threshold": 1000.0},
            "structure_recovery_rate": {"direction": "min", "threshold": 0.0},
            "candidate_latency_p95_ms": {
                "direction": "max",
                "threshold": 600000.0,
            },
        },
    }


def _passing_hard_gate() -> dict[str, dict[str, object]]:
    return {
        "edge_density": {
            "value": 1.0,
            "threshold": 1.0,
            "direction": "max",
            "passed": True,
        },
        "sparse_edge_budget_ratio": {
            "value": 1.0,
            "threshold": 1.0,
            "direction": "max",
            "passed": True,
        },
        "isolated_ratio": {
            "value": 0.0,
            "threshold": 1.0,
            "direction": "max",
            "passed": True,
        },
        "hubness_ratio": {
            "value": 1.0,
            "threshold": 1000.0,
            "direction": "max",
            "passed": True,
        },
        "structure_recovery_rate": {
            "value": 1.0,
            "threshold": 0.0,
            "direction": "min",
            "passed": True,
        },
        "candidate_latency_p95_ms": {
            "value": 1.0,
            "threshold": 600000.0,
            "direction": "max",
            "passed": True,
        },
    }


@pytest.fixture
def postgres_tpe_scope():
    """Create an isolated, committed PostgreSQL scope for durable TPE tests."""

    from app.db import SessionLocal
    from app.models import Document, DocumentVersion, IngestionBatch, KnowledgeBase

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")

    knowledge_base = KnowledgeBase(
        name=f"tpe_audit-{uuid4()}",
        description="durable TPE audit fixture",
        source_root=f"tpe_audit/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.flush()
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base.id,
        trigger_source="tpe_audit-postgres-test",
        source_root=knowledge_base.source_root,
        status="running",
    )
    setup.add(batch)
    document = Document(
        knowledge_base_id=knowledge_base.id,
        title="Durable TPE audit fixture",
        source_path="tpe_audit/durable-tpe-fixture.md",
        logical_source_slot_key=f"tpe_audit:{uuid4()}",
        source_slot_protocol_version="logical_source_slot_v1",
        source_type="markdown",
        checksum="d" * 64,
    )
    setup.add(document)
    setup.flush()
    document_version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path="tpe_audit/durable-tpe-fixture.md",
        parse_protocol_version="parser_v1",
        is_active=True,
    )
    setup.add(document_version)
    setup.commit()
    scope = {
        "knowledge_base_id": knowledge_base.id,
        "batch_id": batch.id,
        "document_id": document.id,
        "document_version_id": document_version.id,
    }
    setup.close()

    try:
        yield scope
    finally:
        cleanup = SessionLocal()
        try:
            row = cleanup.get(KnowledgeBase, scope["knowledge_base_id"])
            if row is not None:
                cleanup.delete(row)
                cleanup.commit()
        except Exception:
            cleanup.rollback()
            raise
        finally:
            cleanup.close()


def _new_run(scope: dict[str, str], *, diagnostics: dict | None = None):
    from app.models import AutoTpeRun
    from app.services.auto_tpe import _protocol_hash, tpe_search_space_hash
    from app.services.context_graph import dense_graph_operating_point

    operating_point = dense_graph_operating_point()

    return AutoTpeRun(
        id=str(uuid4()),
        knowledge_base_id=scope["knowledge_base_id"],
        batch_id=scope["batch_id"],
        chunk_version=1,
        chunk_scope_hash="1" * 64,
        graph_operating_point_protocol=operating_point[
            "graph_operating_point_protocol"
        ],
        protocol_hash=_protocol_hash(),
        tpe_search_space_hash=tpe_search_space_hash(),
        chat_model="tpe_audit-chat-model",
        embedding_model="tpe_audit-embedding-model",
        embedding_text_version="contextualized_embedding_text_v1",
        status="running",
        trigger_reason="tpe_audit-postgres-test",
        trial_budget=1,
        startup_random_trials=1,
        good_quantile_gamma=0.25,
        probe_query_budget=1,
        candidate_pool_size=1,
        runtime_settings_hash="3" * 64,
        diagnostics_json=dict(diagnostics or {}),
        started_at=datetime.utcnow(),
    )


def _new_trial(run, *, diagnostics: dict | None = None):
    from app.models import AutoTpeTrial
    from app.services.auto_tpe import (
        _theta_hash,
        theta_calibration_audit,
        tpe_gate_profile_hash,
        tpe_search_space_hash,
    )
    from app.services.context_graph import dense_graph_operating_point

    sampled_theta = dense_graph_operating_point()
    calibration_audit = theta_calibration_audit(sampled_theta)
    gate_profile = _gate_profile()
    simulated_calibration = {
        "protocol_version": calibration_audit["edge_type_calibration_protocol"],
        "protocol_hash": calibration_audit["edge_type_calibration_protocol_hash"],
        "edge_distance_protocol_version": calibration_audit[
            "edge_distance_protocol"
        ],
        "edge_distance_protocol_hash": calibration_audit[
            "edge_distance_protocol_hash"
        ],
        "params": dict(calibration_audit["calibration_params"]),
        "calibration_params_hash": calibration_audit["calibration_params_hash"],
        "edge_type_calibration_config_hash": calibration_audit[
            "edge_type_calibration_config_hash"
        ],
    }

    return AutoTpeTrial(
        id=str(uuid4()),
        run_id=run.id,
        knowledge_base_id=run.knowledge_base_id,
        build_batch_id=run.batch_id,
        chunk_scope_hash=run.chunk_scope_hash,
        embedding_model=run.embedding_model,
        embedding_text_version=run.embedding_text_version,
        trial_index=1,
        sampled_theta_json=sampled_theta,
        theta_hash=_theta_hash(sampled_theta),
        tpe_search_space_hash=tpe_search_space_hash(),
        edge_distance_protocol=calibration_audit["edge_distance_protocol"],
        edge_distance_protocol_hash=calibration_audit[
            "edge_distance_protocol_hash"
        ],
        edge_type_calibration_protocol=calibration_audit[
            "edge_type_calibration_protocol"
        ],
        edge_type_calibration_protocol_hash=calibration_audit[
            "edge_type_calibration_protocol_hash"
        ],
        calibration_params_json=dict(calibration_audit["calibration_params"]),
        calibration_params_hash=calibration_audit["calibration_params_hash"],
        edge_type_calibration_config_hash=calibration_audit[
            "edge_type_calibration_config_hash"
        ],
        sampler_state_hash="5" * 64,
        runtime_settings_hash=run.runtime_settings_hash,
        gate_profile_hash=tpe_gate_profile_hash(gate_profile),
        gate_profile_json=gate_profile,
        status="running",
        diagnostics_json={
            "theta_calibration_audit": calibration_audit,
            "simulated_edge_type_calibration": simulated_calibration,
            **dict(diagnostics or {}),
        },
        started_at=datetime.utcnow(),
    )


def _persist_selected_run(
    db,
    scope: dict[str, str],
    *,
    register_promotion: bool,
    promotion_lease_expires_at: str | None = None,
):
    from app.services.tpe_audit import (
        TPE_SELECTED_PENDING_STATUS,
        persist_tpe_run,
        persist_tpe_trial,
        register_tpe_graph_promotion,
        transition_tpe_run,
        transition_tpe_trial,
    )

    run = _new_run(scope)
    persist_tpe_run(db, run)
    trial = _new_trial(run)
    persist_tpe_trial(db, trial)
    transition_tpe_trial(
        db,
        trial,
        "completed",
        details={
            "phase": "candidate_evaluation",
            "blocking_reasons": [],
            "retry_boundary": "none",
        },
        objective_score=0.75,
        candidate_adjacency_hash="8" * 64,
        probe_set_hash="9" * 64,
        hard_gate_json=_passing_hard_gate(),
        finished_at=datetime.utcnow(),
    )
    transition_tpe_run(
        db,
        run,
        TPE_SELECTED_PENDING_STATUS,
        details={
            "phase": "best_theta_selected",
            "promotion_status": "awaiting_active_graph_transaction",
            "retry_boundary": "graph_transaction_reconciliation",
        },
        best_trial_id=trial.id,
        best_objective_score=0.75,
        selected_theta_json=dict(trial.sampled_theta_json),
        selected_theta_hash=trial.theta_hash,
        selected_edge_distance_protocol=trial.edge_distance_protocol,
        selected_edge_distance_protocol_hash=trial.edge_distance_protocol_hash,
        selected_edge_type_calibration_protocol=trial.edge_type_calibration_protocol,
        selected_edge_type_calibration_protocol_hash=trial.edge_type_calibration_protocol_hash,
        selected_calibration_params_json=dict(trial.calibration_params_json),
        selected_calibration_params_hash=trial.calibration_params_hash,
        selected_edge_type_calibration_config_hash=trial.edge_type_calibration_config_hash,
        probe_set_hash=trial.probe_set_hash,
        hard_gate_json=dict(trial.hard_gate_json),
        runtime_settings_hash=trial.runtime_settings_hash,
        selected_graph_runtime_settings_hash="4" * 64,
        selected_gate_profile_hash=trial.gate_profile_hash,
        selected_gate_profile_json=dict(trial.gate_profile_json),
        diagnostics_json={
            **dict(run.diagnostics_json or {}),
            "promotion_lease_expires_at": promotion_lease_expires_at
            or (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "promotion_status": "awaiting_active_graph_transaction",
            "selected_runtime_settings_hash": trial.runtime_settings_hash,
            "selected_graph_runtime_settings_hash": "4" * 64,
            "selected_gate_profile_hash": trial.gate_profile_hash,
        },
        completed_at=None,
        failure_code=None,
        blocking_reasons_json=[],
        last_error=None,
    )
    if register_promotion:
        register_tpe_graph_promotion(
            db,
            run_id=run.id,
            knowledge_base_id=run.knowledge_base_id,
            best_trial_id=trial.id,
            selected_theta_hash=trial.theta_hash,
            selected_graph_runtime_settings_hash="4" * 64,
        )
    return run, trial


def _add_relation_state(
    db,
    scope: dict[str, str],
    run,
    trial,
    *,
    run_id: str | None = None,
    best_trial_id: str | None = None,
    theta_hash: str | None = None,
    chunk_version: int | None = None,
    scope_hash: str | None = None,
    objective_score: float | None = None,
    runtime_settings_hash: str | None = None,
    gate_profile_hash: str | None = None,
    gate_profile: dict[str, object] | None = None,
):
    from app.models import ChunkRelationGraphState
    from app.services.context_graph import RELATION_PROTOCOL_VERSION

    effective_run_id = run_id if run_id is not None else run.id
    effective_best_trial_id = best_trial_id if best_trial_id is not None else trial.id
    effective_theta_hash = theta_hash if theta_hash is not None else trial.theta_hash
    effective_chunk_version = chunk_version if chunk_version is not None else run.chunk_version
    effective_runtime_settings_hash = (
        runtime_settings_hash
        if runtime_settings_hash is not None
        else run.selected_graph_runtime_settings_hash
    )
    effective_gate_profile_hash = (
        gate_profile_hash
        if gate_profile_hash is not None
        else run.selected_gate_profile_hash
    )
    effective_gate_profile = (
        dict(gate_profile)
        if gate_profile is not None
        else dict(run.selected_gate_profile_json or {})
    )
    state = ChunkRelationGraphState(
        id=str(uuid4()),
        knowledge_base_id=scope["knowledge_base_id"],
        chunk_version=effective_chunk_version,
        scope_hash=scope_hash if scope_hash is not None else run.chunk_scope_hash,
        state_hash="7" * 64,
        graph_operating_point_hash=effective_theta_hash,
        graph_operating_point_json=dict(trial.sampled_theta_json or {}),
        embedding_text_version=run.embedding_text_version,
        relation_protocol_version=RELATION_PROTOCOL_VERSION,
        edge_distance_protocol_hash=run.selected_edge_distance_protocol_hash,
        edge_type_calibration_protocol_hash=run.selected_edge_type_calibration_protocol_hash,
        runtime_settings_hash=effective_runtime_settings_hash,
        auto_tpe_run_id=effective_run_id,
        auto_tpe_best_trial_id=effective_best_trial_id,
        active_chunk_ids_json=[],
        stats_json={},
        diagnostics_json={
            "fixture": "tpe_audit-postgres",
            "graph_operating_point_protocol": run.graph_operating_point_protocol,
            "graph_operating_point_hash": effective_theta_hash,
            "calibration_params_hash": run.selected_calibration_params_hash,
            "edge_type_calibration_config_hash": run.selected_edge_type_calibration_config_hash,
            "auto_tpe": {
                "status": "selected_pending_graph_commit",
                "run_id": effective_run_id,
                "best_trial_id": effective_best_trial_id,
                "selected_theta_hash": effective_theta_hash,
                "objective_score": (
                    objective_score
                    if objective_score is not None
                    else run.best_objective_score
                ),
                "chunk_version": effective_chunk_version,
                "protocol_hash": run.protocol_hash,
                "tpe_search_space_hash": run.tpe_search_space_hash,
                "edge_distance_protocol": run.selected_edge_distance_protocol,
                "edge_distance_protocol_hash": run.selected_edge_distance_protocol_hash,
                "edge_type_calibration_protocol": run.selected_edge_type_calibration_protocol,
                "edge_type_calibration_protocol_hash": run.selected_edge_type_calibration_protocol_hash,
                "calibration_params": dict(run.selected_calibration_params_json or {}),
                "calibration_params_hash": run.selected_calibration_params_hash,
                "edge_type_calibration_config_hash": run.selected_edge_type_calibration_config_hash,
                "runtime_settings_hash": effective_runtime_settings_hash,
                "optimizer_runtime_settings_hash": run.runtime_settings_hash,
                "gate_profile_hash": effective_gate_profile_hash,
                "gate_profile": effective_gate_profile,
            },
        },
        state="active",
    )
    db.add(state)
    db.flush()
    return state


def test_run_and_trial_audit_survive_outer_graph_rollback(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial, ChunkRelationGraphState
    from app.services.tpe_audit import bind_tpe_graph_promotion_state

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(
            main,
            postgres_tpe_scope,
            register_promotion=True,
        )
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)

        audit_before_rollback = SessionLocal()
        try:
            assert audit_before_rollback.get(AutoTpeRun, run.id).status == "selected_pending_graph_commit"
            assert audit_before_rollback.get(AutoTpeTrial, trial.id).status == "completed"
            assert audit_before_rollback.get(ChunkRelationGraphState, state.id) is None
        finally:
            audit_before_rollback.close()

        main.rollback()

        audit_after_rollback = SessionLocal()
        try:
            durable_run = audit_after_rollback.get(AutoTpeRun, run.id)
            durable_trial = audit_after_rollback.get(AutoTpeTrial, trial.id)
            assert durable_run is not None
            assert durable_run.status == "failed"
            assert durable_run.failure_code == "active_relation_graph_transaction_rolled_back"
            assert durable_run.blocking_reasons_json == ["active_relation_graph_transaction_rolled_back"]
            assert durable_run.chunk_relation_graph_state_id is None
            assert durable_run.diagnostics_json["audit_last_transition"]["details"]["phase"] == (
                "graph_transaction_rolled_back"
            )
            assert durable_trial is not None
            assert durable_trial.status == "completed"
            assert audit_after_rollback.get(ChunkRelationGraphState, state.id) is None
        finally:
            audit_after_rollback.close()
    finally:
        main.close()


def test_durable_tpe_state_machine_rejects_terminal_rewrite(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial
    from app.services.tpe_audit import (
        TPE_SELECTED_PENDING_STATUS,
        TpeAuditError,
        persist_tpe_run,
        persist_tpe_trial,
        transition_tpe_run,
        transition_tpe_trial,
        update_tpe_run,
    )

    origin = SessionLocal()
    try:
        run = _new_run(postgres_tpe_scope)
        persist_tpe_run(origin, run)
        trial = _new_trial(run)
        persist_tpe_trial(origin, trial)
        transition_tpe_trial(
            origin,
            trial,
            "failed",
            details={"phase": "state_machine_probe"},
            failure_code="probe_failure",
            finished_at=datetime.utcnow(),
        )
        transition_tpe_run(
            origin,
            run,
            "failed",
            details={"phase": "state_machine_probe"},
            failure_code="probe_failure",
            blocking_reasons_json=["probe_failure"],
            completed_at=datetime.utcnow(),
        )

        with pytest.raises(TpeAuditError, match="failed -> completed"):
            transition_tpe_trial(
                origin,
                trial,
                "completed",
                details={"phase": "illegal_rewrite"},
            )
        with pytest.raises(TpeAuditError, match="failed -> selected_pending_graph_commit"):
            transition_tpe_run(
                origin,
                run,
                TPE_SELECTED_PENDING_STATUS,
                details={"phase": "illegal_revival"},
            )
        with pytest.raises(TpeAuditError, match="must use the transition API"):
            update_tpe_run(origin, run, status="running")
    finally:
        origin.close()

    audit = SessionLocal()
    try:
        assert audit.get(AutoTpeRun, run.id).status == "failed"
        assert audit.get(AutoTpeTrial, trial.id).status == "failed"
    finally:
        audit.close()


def test_savepoint_release_does_not_finalize_or_consume_outer_promotion(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, ChunkRelationGraphState, KnowledgeBase
    from app.services.tpe_audit import (
        TPE_PROMOTION_SESSION_KEY,
        bind_tpe_graph_promotion_state,
    )

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(
            main,
            postgres_tpe_scope,
            register_promotion=True,
        )
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)

        with main.begin_nested():
            assert main.scalar(
                select(KnowledgeBase.id).where(KnowledgeBase.id == postgres_tpe_scope["knowledge_base_id"])
            ) == postgres_tpe_scope["knowledge_base_id"]

        audit_after_savepoint = SessionLocal()
        try:
            status_after_savepoint = audit_after_savepoint.get(AutoTpeRun, run.id).status
        finally:
            audit_after_savepoint.close()
        handle_survived_savepoint = run.id in dict(main.info.get(TPE_PROMOTION_SESSION_KEY, {}))

        main.rollback()

        audit_after_outer_rollback = SessionLocal()
        try:
            durable_run = audit_after_outer_rollback.get(AutoTpeRun, run.id)
            graph_state = audit_after_outer_rollback.get(ChunkRelationGraphState, state.id)
            final_status = durable_run.status
            final_failure_code = durable_run.failure_code
        finally:
            audit_after_outer_rollback.close()

        assert status_after_savepoint == "selected_pending_graph_commit"
        assert handle_survived_savepoint is True
        assert final_status == "failed"
        assert final_failure_code == "active_relation_graph_transaction_rolled_back"
        assert graph_state is None
    finally:
        main.close()


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        "run",
        "best_trial",
        "theta",
        "objective",
        "chunk_version",
        "chunk_scope",
        "runtime_settings",
        "gate_profile_hash",
        "gate_profile_payload",
    ],
)
def test_outer_commit_requires_exact_run_trial_theta_association(postgres_tpe_scope, mismatch):
    from app.db import SessionLocal
    from app.models import AutoTpeRun
    from app.services import auto_tpe
    from app.services.tpe_audit import (
        bind_tpe_graph_promotion_state,
        persist_tpe_run,
        persist_tpe_trial,
    )

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(
            main,
            postgres_tpe_scope,
            register_promotion=True,
        )
        alternate_run = None
        alternate_trial = None
        if mismatch in {"run", "best_trial"}:
            alternate_run = _new_run(postgres_tpe_scope)
            persist_tpe_run(main, alternate_run)
            alternate_trial = _new_trial(alternate_run)
            persist_tpe_trial(main, alternate_trial)
        state = _add_relation_state(
            main,
            postgres_tpe_scope,
            run,
            trial,
            run_id=alternate_run.id if mismatch == "run" else None,
            best_trial_id=alternate_trial.id if mismatch == "best_trial" else None,
            theta_hash="f" * 64 if mismatch == "theta" else None,
            objective_score=0.5 if mismatch == "objective" else None,
            chunk_version=run.chunk_version + 1 if mismatch == "chunk_version" else None,
            scope_hash="f" * 64 if mismatch == "chunk_scope" else None,
            runtime_settings_hash="f" * 64 if mismatch == "runtime_settings" else None,
            gate_profile_hash="f" * 64 if mismatch == "gate_profile_hash" else None,
            gate_profile={"tampered": True} if mismatch == "gate_profile_payload" else None,
        )
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)
        main.commit()

        audit = SessionLocal()
        try:
            durable_run = audit.get(AutoTpeRun, run.id)
            if mismatch is None:
                assert durable_run.status == "completed"
                assert durable_run.failure_code is None
                assert durable_run.chunk_relation_graph_state_id == state.id
                details = durable_run.diagnostics_json["audit_last_transition"]["details"]
                assert details["phase"] == "graph_transaction_committed"
                assert details["relation_state_id"] == state.id
                assert details["promotion_integrity_ok"] is True
                assert details["retry_boundary"] == "none"
                assert all(details["promotion_integrity_checks"].values())
                assert auto_tpe._latest_completed_theta(
                    audit,
                    run.knowledge_base_id,
                    run.chat_model,
                    run.embedding_model,
                    run.embedding_text_version,
                    run.chunk_scope_hash,
                ) == trial.sampled_theta_json
                assert auto_tpe._latest_completed_theta(
                    audit,
                    run.knowledge_base_id,
                    run.chat_model,
                    run.embedding_model,
                    run.embedding_text_version,
                    "0" * 64,
                ) is None
                summary = auto_tpe.summarize_auto_tpe_run(audit, durable_run)
                assert summary["runtime_settings_hash"] == trial.runtime_settings_hash
                assert summary["selected_graph_runtime_settings_hash"] == "4" * 64
                assert (
                    summary["runtime_settings_hash"]
                    != summary["selected_graph_runtime_settings_hash"]
                )
                assert summary["selected_gate_profile_hash"] == trial.gate_profile_hash
                assert summary["selected_gate_profile"] == trial.gate_profile_json
                assert summary["trials"][0]["runtime_settings_hash"] == trial.runtime_settings_hash
                assert summary["trials"][0]["gate_profile_hash"] == trial.gate_profile_hash
                assert summary["trials"][0]["gate_profile"] == trial.gate_profile_json
                from app.schemas import AutoTpeStatusResponse

                public_status = AutoTpeStatusResponse.model_validate(
                    {
                        "knowledge_base_id": durable_run.knowledge_base_id,
                        "current_chunk_version": durable_run.chunk_version,
                        "enabled": True,
                        "latest_run": summary,
                    }
                )
                assert public_status.latest_run is not None
                assert (
                    public_status.latest_run.runtime_settings_hash
                    == trial.runtime_settings_hash
                )
                assert (
                    public_status.latest_run.selected_graph_runtime_settings_hash
                    == "4" * 64
                )
            else:
                assert durable_run.status == "failed"
                assert durable_run.failure_code == "active_relation_graph_promotion_integrity_failed"
                assert durable_run.blocking_reasons_json == [
                    "active_relation_graph_promotion_integrity_failed"
                ]
                assert durable_run.chunk_relation_graph_state_id is None
                assert durable_run.diagnostics_json["audit_last_transition"]["details"][
                    "promotion_integrity_ok"
                ] is False
        finally:
            audit.close()
    finally:
        main.close()


@pytest.mark.parametrize(
    "trial_corruption",
    [
        "status",
        "objective_missing",
        "objective_mismatch",
        "failure_code",
        "hard_gate",
        "hard_gate_missing",
        "knowledge_base",
        "chunk_scope",
        "runtime_settings",
        "gate_profile_hash",
        "gate_profile_payload",
        "theta",
    ],
)
def test_outer_commit_rejects_corrupt_best_trial_audit(postgres_tpe_scope, trial_corruption):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial, KnowledgeBase
    from app.services.tpe_audit import bind_tpe_graph_promotion_state

    alternate_kb_id = None
    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(
            main,
            postgres_tpe_scope,
            register_promotion=True,
        )
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)

        corrupt = SessionLocal()
        try:
            durable_trial = corrupt.get(AutoTpeTrial, trial.id)
            if trial_corruption == "status":
                durable_trial.status = "blocked"
            elif trial_corruption == "objective_missing":
                durable_trial.objective_score = None
            elif trial_corruption == "objective_mismatch":
                durable_trial.objective_score = 0.5
            elif trial_corruption == "failure_code":
                durable_trial.failure_code = "tampered_failure"
            elif trial_corruption == "hard_gate":
                durable_trial.hard_gate_json = {
                    **_passing_hard_gate(),
                    "edge_density": {"passed": False},
                }
            elif trial_corruption == "hard_gate_missing":
                durable_trial.hard_gate_json = {
                    key: value
                    for key, value in _passing_hard_gate().items()
                    if key != "candidate_latency_p95_ms"
                }
            elif trial_corruption == "knowledge_base":
                alternate_kb = KnowledgeBase(
                    name=f"tpe_audit-alternate-{uuid4()}",
                    description="temporary mismatched TPE trial owner",
                    source_root=f"tpe_audit/alternate/{uuid4()}",
                )
                corrupt.add(alternate_kb)
                corrupt.flush()
                alternate_kb_id = alternate_kb.id
                durable_trial.knowledge_base_id = alternate_kb.id
            elif trial_corruption == "chunk_scope":
                durable_trial.chunk_scope_hash = "d" * 64
            elif trial_corruption == "runtime_settings":
                durable_trial.runtime_settings_hash = "d" * 64
            elif trial_corruption == "gate_profile_hash":
                durable_trial.gate_profile_hash = "d" * 64
            elif trial_corruption == "gate_profile_payload":
                durable_trial.gate_profile_json = {"tampered": True}
            else:
                durable_trial.theta_hash = "e" * 64
            corrupt.commit()
        finally:
            corrupt.close()

        main.commit()

        audit = SessionLocal()
        try:
            durable_run = audit.get(AutoTpeRun, run.id)
            assert durable_run.status == "failed"
            assert durable_run.failure_code == "active_relation_graph_promotion_integrity_failed"
            checks = durable_run.diagnostics_json["audit_last_transition"]["details"][
                "promotion_integrity_checks"
            ]
            expected_check = {
                "status": "best_trial_completed",
                "objective_missing": "best_trial_completed",
                "objective_mismatch": "best_trial_objective_matches",
                "failure_code": "best_trial_failure_clear",
                "hard_gate": "best_trial_hard_gate_passed",
                "hard_gate_missing": "best_trial_hard_gate_passed",
                "knowledge_base": "best_trial_knowledge_base_matches",
                "chunk_scope": "best_trial_chunk_scope_matches",
                "runtime_settings": "best_trial_runtime_settings_matches",
                "gate_profile_hash": "best_trial_gate_profile_matches",
                "gate_profile_payload": "best_trial_gate_profile_matches",
                "theta": "best_trial_theta_matches",
            }[trial_corruption]
            assert checks[expected_check] is False
        finally:
            audit.close()
    finally:
        main.close()
        if alternate_kb_id:
            cleanup = SessionLocal()
            try:
                alternate_kb = cleanup.get(KnowledgeBase, alternate_kb_id)
                if alternate_kb is not None:
                    cleanup.delete(alternate_kb)
                    cleanup.commit()
            finally:
                cleanup.close()


def test_outer_commit_rechecks_durable_run_instead_of_stale_handle(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun
    from app.services.tpe_audit import bind_tpe_graph_promotion_state, persist_tpe_trial

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(main, postgres_tpe_scope, register_promotion=True)
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)
        replacement_trial = _new_trial(run)
        replacement_trial.trial_index = 2
        persist_tpe_trial(main, replacement_trial)

        concurrent = SessionLocal()
        try:
            durable = concurrent.get(AutoTpeRun, run.id)
            durable.best_trial_id = replacement_trial.id
            durable.selected_theta_hash = "f" * 64
            concurrent.commit()
        finally:
            concurrent.close()

        main.commit()

        audit = SessionLocal()
        try:
            durable = audit.get(AutoTpeRun, run.id)
            assert durable.status == "failed"
            assert durable.failure_code == "active_relation_graph_promotion_integrity_failed"
            assert durable.chunk_relation_graph_state_id is None
        finally:
            audit.close()
    finally:
        main.close()


@pytest.mark.parametrize(
    ("run_corruption", "expected_check"),
    [
        ("failure_code", "run_failure_clear"),
        ("blocking_reason", "run_blocking_reasons_clear"),
        ("hard_gate", "run_hard_gate_passed"),
        ("optimizer_runtime_settings", "best_trial_runtime_settings_matches"),
        ("graph_runtime_settings", "state_runtime_settings_matches"),
        ("gate_profile_hash", "run_gate_profile_valid"),
    ],
)
def test_outer_commit_rejects_corrupt_selected_run_audit(
    postgres_tpe_scope,
    run_corruption,
    expected_check,
):
    from app.db import SessionLocal
    from app.models import AutoTpeRun
    from app.services.tpe_audit import bind_tpe_graph_promotion_state

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(main, postgres_tpe_scope, register_promotion=True)
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)

        corrupt = SessionLocal()
        try:
            durable_run = corrupt.get(AutoTpeRun, run.id)
            if run_corruption == "failure_code":
                durable_run.failure_code = "tampered_failure"
            elif run_corruption == "blocking_reason":
                durable_run.blocking_reasons_json = ["tampered_blocker"]
            elif run_corruption == "hard_gate":
                durable_run.hard_gate_json = {
                    **_passing_hard_gate(),
                    "edge_density": {"passed": False},
                }
            elif run_corruption == "optimizer_runtime_settings":
                durable_run.runtime_settings_hash = "f" * 64
            elif run_corruption == "graph_runtime_settings":
                durable_run.selected_graph_runtime_settings_hash = "f" * 64
            else:
                durable_run.selected_gate_profile_hash = "f" * 64
            corrupt.commit()
        finally:
            corrupt.close()

        main.commit()

        audit = SessionLocal()
        try:
            durable_run = audit.get(AutoTpeRun, run.id)
            assert durable_run.status == "failed"
            checks = durable_run.diagnostics_json["audit_last_transition"]["details"][
                "promotion_integrity_checks"
            ]
            assert checks[expected_check] is False
        finally:
            audit.close()
    finally:
        main.close()


def test_expired_lease_cannot_kill_a_live_outer_graph_transaction(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun
    from app.services.tpe_audit import bind_tpe_graph_promotion_state, reconcile_tpe_audit

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(
            main,
            postgres_tpe_scope,
            register_promotion=True,
            promotion_lease_expires_at=(datetime.utcnow() - timedelta(seconds=1)).isoformat(),
        )
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)

        status_session = SessionLocal()
        try:
            stats = reconcile_tpe_audit(
                status_session,
                knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
            )
        finally:
            status_session.close()
        assert stats["skipped_active_owner"] == 1

        main.commit()

        audit = SessionLocal()
        try:
            durable = audit.get(AutoTpeRun, run.id)
            assert durable.status == "completed"
            assert durable.chunk_relation_graph_state_id == state.id
        finally:
            audit.close()
    finally:
        main.close()


def test_running_trial_lease_is_not_masked_by_longer_run_lease(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial
    from app.services.tpe_audit import persist_tpe_run, persist_tpe_trial, reconcile_tpe_audit

    origin = SessionLocal()
    try:
        run = _new_run(
            postgres_tpe_scope,
            diagnostics={
                "run_lease_expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "retry_boundary": "next_trial_boundary",
            },
        )
        persist_tpe_run(origin, run)
        trial = _new_trial(
            run,
            diagnostics={
                "trial_lease_expires_at": (datetime.utcnow() - timedelta(seconds=1)).isoformat(),
                "retry_boundary": "next_trial_boundary",
            },
        )
        persist_tpe_trial(origin, trial)
    finally:
        origin.close()

    reconcile = SessionLocal()
    try:
        reconcile_tpe_audit(
            reconcile,
            knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
        )
    finally:
        reconcile.close()

    audit = SessionLocal()
    try:
        durable_run = audit.get(AutoTpeRun, run.id)
        durable_trial = audit.get(AutoTpeTrial, trial.id)
        assert durable_run.status == "failed"
        assert durable_run.failure_code == "tpe_process_interrupted"
        assert durable_trial.status == "failed"
        assert durable_trial.failure_code == "trial_process_interrupted"
    finally:
        audit.close()


def test_reconcile_only_fails_each_expired_running_trial(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial
    from app.services.tpe_audit import persist_tpe_run, persist_tpe_trial, reconcile_tpe_audit

    origin = SessionLocal()
    try:
        run = _new_run(
            postgres_tpe_scope,
            diagnostics={
                "run_lease_expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "retry_boundary": "next_graph_build",
            },
        )
        persist_tpe_run(origin, run)
        expired_trial = _new_trial(
            run,
            diagnostics={
                "trial_lease_expires_at": (datetime.utcnow() - timedelta(seconds=1)).isoformat(),
                "retry_boundary": "next_trial_boundary",
            },
        )
        persist_tpe_trial(origin, expired_trial)
        live_trial = _new_trial(
            run,
            diagnostics={
                "trial_lease_expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "retry_boundary": "next_trial_boundary",
            },
        )
        live_trial.trial_index = 2
        persist_tpe_trial(origin, live_trial)
    finally:
        origin.close()

    reconcile = SessionLocal()
    try:
        stats = reconcile_tpe_audit(
            reconcile,
            knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
        )
    finally:
        reconcile.close()

    audit = SessionLocal()
    try:
        durable_run = audit.get(AutoTpeRun, run.id)
        durable_expired = audit.get(AutoTpeTrial, expired_trial.id)
        durable_live = audit.get(AutoTpeTrial, live_trial.id)
        assert stats == {
            "checked": 1,
            "completed": 0,
            "failed": 0,
            "skipped_unexpired": 1,
            "skipped_active_owner": 0,
            "skipped_active_resource": 0,
        }
        assert durable_run.status == "running"
        assert durable_expired.status == "failed"
        assert durable_expired.failure_code == "trial_process_interrupted"
        assert durable_live.status == "running"
        details = durable_expired.diagnostics_json["audit_last_transition"]["details"]
        assert details["retry_boundary"] == "next_trial_boundary"
        assert details["remaining_live_trial_ids"] == [live_trial.id]
    finally:
        audit.close()


def test_shadow_relation_state_cannot_complete_or_seed_tpe(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun
    from app.services import auto_tpe
    from app.services.tpe_audit import bind_tpe_graph_promotion_state

    main = SessionLocal()
    try:
        run, trial = _persist_selected_run(main, postgres_tpe_scope, register_promotion=True)
        state = _add_relation_state(main, postgres_tpe_scope, run, trial)
        state.state = "shadow"
        bind_tpe_graph_promotion_state(main, run_id=run.id, relation_state_id=state.id)

        audit_update = SessionLocal()
        try:
            durable = audit_update.get(AutoTpeRun, run.id)
            durable.protocol_hash = auto_tpe._protocol_hash()
            audit_update.commit()
        finally:
            audit_update.close()

        main.commit()

        lookup = SessionLocal()
        try:
            durable = lookup.get(AutoTpeRun, run.id)
            assert durable.status == "failed"
            checks = durable.diagnostics_json["audit_last_transition"]["details"][
                "promotion_integrity_checks"
            ]
            assert checks["state_active"] is False
            selected = auto_tpe._latest_completed_theta(
                lookup,
                postgres_tpe_scope["knowledge_base_id"],
                run.chat_model,
                run.embedding_model,
                run.embedding_text_version,
                run.chunk_scope_hash,
            )
            assert selected is None
        finally:
            lookup.close()
    finally:
        main.close()


@pytest.mark.parametrize(
    ("scenario", "raised_error", "trial_status", "failure_code", "retry_boundary", "run_status"),
    [
        ("timeout", None, "failed", "trial_timeout", "next_trial_boundary", "failed"),
        (
            "evaluation_timeout",
            None,
            "failed",
            "trial_timeout",
            "next_trial_boundary",
            "failed",
        ),
        (
            "exception",
            None,
            "failed",
            "candidate_simulation_failed",
            "next_trial_boundary",
            "failed",
        ),
        (
            "cancel",
            None,
            "cancelled",
            "batch_cancelled",
            "next_graph_build",
            "cancelled",
        ),
    ],
)
def test_auto_tpe_trial_terminal_diagnostics_are_durable(
    postgres_tpe_scope,
    monkeypatch,
    scenario,
    raised_error,
    trial_status,
    failure_code,
    retry_boundary,
    run_status,
):
    from app.core.config import get_settings
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial, Chunk
    from app.services import auto_tpe
    from app.services.cancellation import IngestionCancelled

    real_settings = get_settings()
    settings_payload = real_settings.model_dump()
    settings_payload.update(
        {
            "enable_auto_tpe": True,
            "tpe_trial_budget": 1,
            "tpe_startup_random_trials": 1,
            "tpe_good_quantile_gamma": 0.25,
            "tpe_probe_query_budget": 1,
            "tpe_candidate_pool_size": 1,
            "tpe_trial_timeout_seconds": 0.001,
            "graph_model": "tpe_audit-chat-model",
            "embedding_model": "tpe_audit-embedding-model",
        }
    )
    fake_settings = SimpleNamespace(**settings_payload)
    fixed_theta = auto_tpe.normalize_theta(auto_tpe.dense_graph_operating_point())

    monkeypatch.setattr(auto_tpe, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(auto_tpe, "refresh_runtime_settings_if_needed", lambda **_kwargs: None)
    monkeypatch.setattr(auto_tpe, "emit_ingestion_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(auto_tpe, "_protocol_hash", lambda: "8" * 64)
    monkeypatch.setattr(auto_tpe, "_runtime_hash", lambda: "9" * 64)
    monkeypatch.setattr(
        auto_tpe,
        "_sample_tpe_theta",
        lambda *_args, **_kwargs: (dict(fixed_theta), "a" * 64),
    )
    monkeypatch.setattr(auto_tpe, "ensure_not_cancelled", lambda *_args, **_kwargs: None)

    if scenario == "timeout":
        # Candidate-RQ precomputation consumes the first pair; the trial
        # boundary then observes a timeout after candidate simulation.
        clock = iter([0.0, 0.0, 0.0, 1.0])
        monkeypatch.setattr(auto_tpe.time, "perf_counter", lambda: next(clock))
        monkeypatch.setattr(
            auto_tpe,
            "relation_edge_candidates",
            lambda *_args, **_kwargs: ({}, _simulation_diagnostics(_args[-1])),
        )
    elif scenario == "evaluation_timeout":
        # Candidate-RQ precomputation consumes the first pair; candidate
        # simulation remains in-budget and evaluation crosses the deadline.
        clock = iter([0.0, 0.0, 0.0, 0.0, 1.0])
        monkeypatch.setattr(auto_tpe.time, "perf_counter", lambda: next(clock))
        monkeypatch.setattr(
            auto_tpe,
            "relation_edge_candidates",
            lambda *_args, **_kwargs: ({}, _simulation_diagnostics(_args[-1])),
        )
    elif scenario == "exception":
        monkeypatch.setattr(auto_tpe.time, "perf_counter", lambda: 0.0)

        def fail_candidate_simulation(*_args, **_kwargs):
            raise ValueError("synthetic candidate simulation failure")

        monkeypatch.setattr(auto_tpe, "relation_edge_candidates", fail_candidate_simulation)
    else:
        raised_error = IngestionCancelled
        monkeypatch.setattr(auto_tpe.time, "perf_counter", lambda: 0.0)

        def cancel_candidate_simulation(*_args, **_kwargs):
            raise IngestionCancelled("synthetic trial cancellation")

        monkeypatch.setattr(auto_tpe, "relation_edge_candidates", cancel_candidate_simulation)

    if scenario == "evaluation_timeout":
        monkeypatch.setattr(
            auto_tpe,
            "evaluate_candidate_trial",
            lambda *_args, **_kwargs: (
                _passing_hard_gate(),
                {"probe": {"value": 1.0}},
                0.75,
                None,
                "c" * 64,
            ),
        )
    else:
        monkeypatch.setattr(
            auto_tpe,
            "evaluate_candidate_trial",
            lambda *_args, **_kwargs: pytest.fail(
                "terminal failure scenarios must not evaluate a candidate"
            ),
        )

    chunk = Chunk(
        id=str(uuid4()),
        knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
        document_id=postgres_tpe_scope["document_id"],
        document_version_id=postgres_tpe_scope["document_version_id"],
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=4,
        char_start=0,
        char_end=16,
        text="TPE audit fixture",
        text_hash="b" * 64,
        state="active",
    )

    main = SessionLocal()
    try:
        if raised_error is not None:
            with pytest.raises(raised_error):
                auto_tpe.select_auto_tpe_operating_point(
                    main,
                    postgres_tpe_scope["knowledge_base_id"],
                    [chunk],
                    {chunk.id: [1.0, 0.0]},
                    fallback_operating_point=fixed_theta,
                    batch_id=postgres_tpe_scope["batch_id"],
                    chunk_version_incremented=True,
                )
        else:
            selected, context = auto_tpe.select_auto_tpe_operating_point(
                main,
                postgres_tpe_scope["knowledge_base_id"],
                [chunk],
                {chunk.id: [1.0, 0.0]},
                fallback_operating_point=fixed_theta,
                batch_id=postgres_tpe_scope["batch_id"],
                chunk_version_incremented=True,
            )
            assert selected == fixed_theta
            assert context["status"] == "failed"
            assert context["auto_tpe_status"] == "failed_or_skipped"
        main.rollback()

        audit = SessionLocal()
        try:
            durable_run = audit.scalar(
                select(AutoTpeRun).where(
                    AutoTpeRun.knowledge_base_id == postgres_tpe_scope["knowledge_base_id"]
                )
            )
            durable_trial = audit.scalar(
                select(AutoTpeTrial).where(AutoTpeTrial.run_id == durable_run.id)
            )
            assert durable_run.status == run_status
            assert durable_trial is not None
            assert durable_trial.status == trial_status
            assert durable_trial.failure_code == failure_code
            assert durable_trial.finished_at is not None
            assert durable_trial.diagnostics_json["audit_durability_mode"] == (
                "independent_postgresql_transaction"
            )
            transition_details = durable_trial.diagnostics_json["audit_last_transition"]["details"]
            assert transition_details["blocking_reasons"] == [failure_code]
            assert transition_details["retry_boundary"] == retry_boundary
            if scenario == "exception":
                assert durable_trial.diagnostics_json["error"] == (
                    "synthetic candidate simulation failure"
                )
            if scenario == "cancel":
                assert durable_run.failure_code == "batch_cancelled"
                assert durable_run.blocking_reasons_json == ["batch_cancelled"]
        finally:
            audit.close()
    finally:
        main.close()


def test_reconcile_selected_pending_after_process_interruption(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun
    from app.services.tpe_audit import reconcile_tpe_audit

    no_handle_session = SessionLocal()
    try:
        graph_run, graph_trial = _persist_selected_run(
            no_handle_session,
            postgres_tpe_scope,
            register_promotion=False,
        )
        missing_graph_run, _ = _persist_selected_run(
            no_handle_session,
            postgres_tpe_scope,
            register_promotion=False,
            promotion_lease_expires_at=(datetime.utcnow() - timedelta(seconds=1)).isoformat(),
        )
    finally:
        no_handle_session.close()

    graph_session = SessionLocal()
    try:
        state = _add_relation_state(
            graph_session,
            postgres_tpe_scope,
            graph_run,
            graph_trial,
        )
        graph_session.commit()
        state_id = state.id
    finally:
        graph_session.close()

    reconcile_session = SessionLocal()
    try:
        stats = reconcile_tpe_audit(
            reconcile_session,
            knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
        )
    finally:
        reconcile_session.close()

    audit = SessionLocal()
    try:
        completed = audit.get(AutoTpeRun, graph_run.id)
        failed = audit.get(AutoTpeRun, missing_graph_run.id)
        assert stats == {
            "checked": 2,
            "completed": 1,
            "failed": 1,
            "skipped_unexpired": 0,
            "skipped_active_owner": 0,
            "skipped_active_resource": 0,
        }
        assert completed.status == "completed"
        assert completed.chunk_relation_graph_state_id == state_id
        assert completed.failure_code is None
        assert completed.diagnostics_json["audit_last_transition"]["details"]["phase"] == (
            "promotion_reconciled_after_process_interruption"
        )
        assert failed.status == "failed"
        assert failed.chunk_relation_graph_state_id is None
        assert failed.failure_code == "active_relation_graph_process_interrupted"
        assert failed.blocking_reasons_json == ["active_relation_graph_process_interrupted"]
    finally:
        audit.close()


def test_reconcile_expired_running_run_and_trial_as_failed(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import AutoTpeRun, AutoTpeTrial
    from app.services.tpe_audit import persist_tpe_run, persist_tpe_trial, reconcile_tpe_audit

    expired_at = (datetime.utcnow() - timedelta(seconds=1)).isoformat()
    origin = SessionLocal()
    try:
        run = _new_run(
            postgres_tpe_scope,
            diagnostics={
                "run_lease_expires_at": expired_at,
                "retry_boundary": "next_trial_boundary",
            },
        )
        persist_tpe_run(origin, run)
        trial = _new_trial(
            run,
            diagnostics={
                "trial_lease_expires_at": expired_at,
                "retry_boundary": "next_trial_boundary",
            },
        )
        persist_tpe_trial(origin, trial)
    finally:
        origin.close()

    reconcile_session = SessionLocal()
    try:
        stats = reconcile_tpe_audit(
            reconcile_session,
            knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
        )
    finally:
        reconcile_session.close()

    audit = SessionLocal()
    try:
        durable_run = audit.get(AutoTpeRun, run.id)
        durable_trial = audit.get(AutoTpeTrial, trial.id)
        assert stats == {
            "checked": 1,
            "completed": 0,
            "failed": 1,
            "skipped_unexpired": 0,
            "skipped_active_owner": 0,
            "skipped_active_resource": 0,
        }
        assert durable_run.status == "failed"
        assert durable_run.failure_code == "tpe_process_interrupted"
        assert durable_run.blocking_reasons_json == ["tpe_process_interrupted"]
        assert durable_run.completed_at is not None
        assert durable_trial.status == "failed"
        assert durable_trial.failure_code == "trial_process_interrupted"
        assert durable_trial.finished_at is not None
        trial_details = durable_trial.diagnostics_json["audit_last_transition"]["details"]
        assert trial_details["phase"] == "trial_reconciled_after_process_interruption"
        assert trial_details["blocking_reasons"] == ["trial_process_interrupted"]
        assert trial_details["retry_boundary"] == "next_graph_build"
    finally:
        audit.close()


def test_reconcile_releases_resource_and_owner_locks_on_its_physical_connection(
    postgres_tpe_scope,
):
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool

    from app.db import SessionLocal
    from app.services.ingestion_resource_lock import advisory_lock_key, knowledge_base_resource_key
    from app.services.tpe_audit import (
        _owner_lock_key,
        persist_tpe_run,
        persist_tpe_trial,
        reconcile_tpe_audit,
    )

    expired_at = (datetime.utcnow() - timedelta(seconds=1)).isoformat()
    origin = SessionLocal()
    try:
        run = _new_run(
            postgres_tpe_scope,
            diagnostics={
                "run_lease_expires_at": expired_at,
                "retry_boundary": "next_graph_build",
            },
        )
        persist_tpe_run(origin, run)
        trial = _new_trial(
            run,
            diagnostics={
                "trial_lease_expires_at": expired_at,
                "retry_boundary": "next_trial_boundary",
            },
        )
        persist_tpe_trial(origin, trial)
        source_engine = origin.get_bind()
    finally:
        origin.close()

    reconcile_session = SessionLocal()
    try:
        stats = reconcile_tpe_audit(
            reconcile_session,
            knowledge_base_id=postgres_tpe_scope["knowledge_base_id"],
        )
        assert stats["failed"] == 1
    finally:
        reconcile_session.close()

    resource_lock_key = advisory_lock_key(
        knowledge_base_resource_key(postgres_tpe_scope["knowledge_base_id"])
    )
    owner_lock_key = _owner_lock_key(run.id)
    probe_engine = create_engine(source_engine.url, future=True, poolclass=NullPool)
    acquired_resource = False
    acquired_owner = False
    try:
        with probe_engine.connect() as connection:
            acquired_resource = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:lock_key)"),
                    {"lock_key": resource_lock_key},
                ).scalar_one()
            )
            acquired_owner = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:lock_key)"),
                    {"lock_key": owner_lock_key},
                ).scalar_one()
            )
            assert acquired_resource is True
            assert acquired_owner is True
            if acquired_owner:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": owner_lock_key},
                )
            if acquired_resource:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": resource_lock_key},
                )
            connection.commit()
    finally:
        probe_engine.dispose()
