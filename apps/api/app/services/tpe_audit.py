from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import event, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, SessionTransaction

from app.models import AutoTpeRun, AutoTpeTrial, ChunkRelationGraphState
from app.services.chunking import stable_hash
from app.services.ingestion_resource_lock import advisory_lock_key, knowledge_base_resource_key


TPE_AUDIT_PROTOCOL_VERSION = "tpe_durable_audit_v5"
TPE_OWNER_FENCE_PROTOCOL_VERSION = "postgres_advisory_tpe_run_owner_v1"
TPE_GATE_PROFILE_PROTOCOL_VERSION = "tpe_hard_gate_profile_v2"
TPE_SELECTED_PENDING_STATUS = "selected_pending_graph_commit"
TPE_PROMOTION_SESSION_KEY = "tpe_graph_promotion_handles_v1"
TPE_AUDIT_TEST_HISTORY_KEY = "tpe_audit_sqlite_test_history_v1"
TPE_PROMOTION_LEASE_SECONDS = 300
TPE_REQUIRED_HARD_GATES = frozenset(
    {
        "edge_density",
        "sparse_edge_budget_ratio",
        "isolated_ratio",
        "hubness_ratio",
        "structure_recovery_rate",
        "candidate_latency_p95_ms",
    }
)

_RUN_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "running": {TPE_SELECTED_PENDING_STATUS, "failed", "cancelled", "skipped"},
    TPE_SELECTED_PENDING_STATUS: {"completed", "failed"},
}
_TRIAL_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "failed", "cancelled"},
    "running": {"completed", "blocked", "failed", "cancelled"},
}


class TpeAuditError(RuntimeError):
    pass


@dataclass
class TpePromotionHandle:
    run_id: str
    knowledge_base_id: str
    best_trial_id: str
    selected_theta_hash: str
    selected_graph_runtime_settings_hash: str
    relation_state_id: str | None = None
    owner_lock_key: int | None = None
    owner_connection: Connection | None = None


def _now() -> datetime:
    return datetime.utcnow()


def _running_under_pytest() -> bool:
    return bool(
        os.getenv("PYTEST_CURRENT_TEST")
        or "pytest" in sys.modules
        or any("pytest" in str(argument).lower() for argument in sys.argv)
    )


def _dialect_name(db: Session) -> str:
    return str(db.get_bind().dialect.name)


def _audit_session(origin_db: Session) -> Session:
    return Session(bind=origin_db.get_bind(), autoflush=False, expire_on_commit=False, future=True)


def _owner_lock_key(run_id: str) -> int:
    digest = hashlib.sha256(
        f"symbograph:tpe-audit-owner:v1:{run_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def _engine_for(db: Session) -> Engine:
    bind = db.get_bind()
    return bind if isinstance(bind, Engine) else bind.engine


def _acquire_owner_connection(db: Session, run_id: str) -> tuple[int | None, Connection | None]:
    if _dialect_name(db) != "postgresql":
        return None, None
    # Ensure the promotion handle is tied to a real root Session transaction;
    # Session.close/rollback can then fence and finalize an abandoned handle.
    db.connection()
    lock_key = _owner_lock_key(run_id)
    connection = _engine_for(db).connect()
    try:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": lock_key},
            ).scalar_one()
        )
        connection.commit()
        if not acquired:
            raise TpeAuditError(f"TPE audit owner lock is already held for run {run_id}")
        return lock_key, connection
    except Exception:
        connection.close()
        raise


def _release_owner_connection(handle: TpePromotionHandle) -> None:
    connection = handle.owner_connection
    lock_key = handle.owner_lock_key
    handle.owner_connection = None
    handle.owner_lock_key = None
    if connection is None:
        return
    try:
        if lock_key is not None:
            released = bool(
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key},
                ).scalar_one()
            )
            connection.commit()
            if not released:
                raise TpeAuditError(f"TPE audit owner lock was not held for run {handle.run_id}")
    except Exception:
        connection.invalidate()
        raise
    finally:
        connection.close()


def _try_reconcile_owner_lock(db: Session, run_id: str) -> tuple[int, bool]:
    lock_key = _owner_lock_key(run_id)
    acquired = bool(
        db.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": lock_key},
        ).scalar_one()
    )
    return lock_key, acquired


def _try_reconcile_resource_lock(db: Session, knowledge_base_id: str) -> tuple[int, bool]:
    lock_key = advisory_lock_key(knowledge_base_resource_key(knowledge_base_id))
    acquired = bool(
        db.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": lock_key},
        ).scalar_one()
    )
    return lock_key, acquired


def _release_reconcile_owner_lock(db: Session | Connection, lock_key: int) -> None:
    released = bool(
        db.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": lock_key},
        ).scalar_one()
    )
    if not released:
        raise TpeAuditError(f"TPE reconcile advisory lock was not held: {lock_key}")


def _annotate_owner_fence(db: Session, run_id: str, lock_key: int | None) -> None:
    if lock_key is None or _dialect_name(db) != "postgresql":
        return
    with (
        _engine_for(db).connect() as connection,
        Session(
            bind=connection,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        ) as audit,
    ):
        row = audit.get(AutoTpeRun, run_id, with_for_update=True)
        if row is None or row.status != TPE_SELECTED_PENDING_STATUS:
            raise TpeAuditError(f"Cannot fence missing or non-pending TPE run {run_id}")
        row.diagnostics_json = {
            **dict(row.diagnostics_json or {}),
            "owner_fence_protocol_version": TPE_OWNER_FENCE_PROTOCOL_VERSION,
            "owner_lock_key": lock_key,
            "owner_fence_acquired_at": _now().isoformat(),
        }
        audit.commit()


def _durability_mode(db: Session) -> str:
    if _dialect_name(db) == "postgresql":
        return "independent_postgresql_transaction"
    if _dialect_name(db) == "sqlite" and _running_under_pytest():
        return "sqlite_pytest_non_durable_adapter_v1"
    raise TpeAuditError(
        "Durable TPE audit records require PostgreSQL; only the explicit SQLite pytest adapter is supported"
    )


def _transition_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    status: str,
    durability_mode: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(diagnostics or {})
    history = list(payload.get("audit_status_history") or [])
    transition: dict[str, Any] = {
        "status": status,
        "at": _now().isoformat(),
    }
    if details:
        transition["details"] = dict(details)
    history.append(transition)
    payload.update(
        {
            "audit_protocol_version": TPE_AUDIT_PROTOCOL_VERSION,
            "audit_durability_mode": durability_mode,
            "audit_status_history": history,
            "audit_last_transition": transition,
        }
    )
    return payload


def _record_sqlite_test_history(
    db: Session,
    *,
    object_type: str,
    object_id: str,
    status: str,
    diagnostics: dict[str, Any],
) -> None:
    db.info.setdefault(TPE_AUDIT_TEST_HISTORY_KEY, []).append(
        {
            "object_type": object_type,
            "object_id": object_id,
            "status": status,
            "diagnostics": dict(diagnostics),
            "at": _now().isoformat(),
            "adapter": "sqlite_pytest_non_durable_adapter_v1",
        }
    )


def _validate_status_transition(
    model: type[AutoTpeRun] | type[AutoTpeTrial],
    current_status: str,
    target_status: str,
) -> None:
    transitions = _RUN_STATUS_TRANSITIONS if model is AutoTpeRun else _TRIAL_STATUS_TRANSITIONS
    if target_status not in transitions.get(current_status, set()):
        object_type = "run" if model is AutoTpeRun else "trial"
        raise TpeAuditError(
            f"Invalid durable TPE {object_type} transition: {current_status} -> {target_status}"
        )


def persist_tpe_run(db: Session, run: AutoTpeRun) -> bool:
    mode = _durability_mode(db)
    if run.status != "running":
        raise TpeAuditError(f"A durable TPE run must be created as running, got {run.status}")
    run.id = run.id or str(uuid4())
    run.diagnostics_json = _transition_diagnostics(
        run.diagnostics_json,
        status=run.status,
        durability_mode=mode,
        details={"phase": "run_created"},
    )
    if mode == "sqlite_pytest_non_durable_adapter_v1":
        db.add(run)
        db.flush()
        _record_sqlite_test_history(
            db,
            object_type="run",
            object_id=run.id,
            status=run.status,
            diagnostics=run.diagnostics_json,
        )
        return False
    with _audit_session(db) as audit:
        audit.add(run)
        audit.commit()
    return True


def persist_tpe_trial(db: Session, trial: AutoTpeTrial) -> bool:
    mode = _durability_mode(db)
    if trial.status not in {"queued", "running"}:
        raise TpeAuditError(
            f"A durable TPE trial must be created as queued or running, got {trial.status}"
        )
    trial.id = trial.id or str(uuid4())
    trial.diagnostics_json = _transition_diagnostics(
        trial.diagnostics_json,
        status=trial.status,
        durability_mode=mode,
        details={"phase": "trial_created"},
    )
    if mode == "sqlite_pytest_non_durable_adapter_v1":
        db.add(trial)
        db.flush()
        _record_sqlite_test_history(
            db,
            object_type="trial",
            object_id=trial.id,
            status=trial.status,
            diagnostics=trial.diagnostics_json,
        )
        return False
    with _audit_session(db) as audit:
        audit.add(trial)
        audit.commit()
    return True


def _update_tpe_object(
    db: Session,
    local_object: AutoTpeRun | AutoTpeTrial,
    model: type[AutoTpeRun] | type[AutoTpeTrial],
    updates: dict[str, Any],
) -> None:
    if "status" in updates:
        raise TpeAuditError("Durable TPE status changes must use the transition API")
    mode = _durability_mode(db)
    if mode == "sqlite_pytest_non_durable_adapter_v1":
        for field, value in updates.items():
            setattr(local_object, field, value)
        db.flush()
        _record_sqlite_test_history(
            db,
            object_type="run" if model is AutoTpeRun else "trial",
            object_id=local_object.id,
            status=str(local_object.status),
            diagnostics=dict(local_object.diagnostics_json or {}),
        )
        return
    with _audit_session(db) as audit:
        durable_object = audit.get(model, local_object.id, with_for_update=True)
        if durable_object is None:
            raise TpeAuditError(f"Durable TPE audit object is missing: {local_object.id}")
        for field, value in updates.items():
            setattr(durable_object, field, value)
        audit.commit()
    for field, value in updates.items():
        setattr(local_object, field, value)


def _transition_tpe_object(
    db: Session,
    local_object: AutoTpeRun | AutoTpeTrial,
    model: type[AutoTpeRun] | type[AutoTpeTrial],
    status: str,
    *,
    details: dict[str, Any] | None,
    updates: dict[str, Any],
) -> None:
    if "status" in updates:
        raise TpeAuditError("Transition target status must be passed positionally")
    mode = _durability_mode(db)
    explicit_diagnostics = updates.pop("diagnostics_json", None)
    if mode == "sqlite_pytest_non_durable_adapter_v1":
        current_status = str(local_object.status)
        _validate_status_transition(model, current_status, status)
        diagnostics = _transition_diagnostics(
            explicit_diagnostics if explicit_diagnostics is not None else local_object.diagnostics_json,
            status=status,
            durability_mode=mode,
            details=details,
        )
        applied_updates = {**updates, "status": status, "diagnostics_json": diagnostics}
        for field, value in applied_updates.items():
            setattr(local_object, field, value)
        db.flush()
        _record_sqlite_test_history(
            db,
            object_type="run" if model is AutoTpeRun else "trial",
            object_id=local_object.id,
            status=status,
            diagnostics=diagnostics,
        )
        return

    with _audit_session(db) as audit:
        durable_object = audit.get(model, local_object.id, with_for_update=True)
        if durable_object is None:
            raise TpeAuditError(f"Durable TPE audit object is missing: {local_object.id}")
        current_status = str(durable_object.status)
        _validate_status_transition(model, current_status, status)
        diagnostics = _transition_diagnostics(
            explicit_diagnostics if explicit_diagnostics is not None else durable_object.diagnostics_json,
            status=status,
            durability_mode=mode,
            details=details,
        )
        applied_updates = {**updates, "status": status, "diagnostics_json": diagnostics}
        for field, value in applied_updates.items():
            setattr(durable_object, field, value)
        audit.commit()
    for field, value in applied_updates.items():
        setattr(local_object, field, value)


def update_tpe_run(db: Session, run: AutoTpeRun, **updates: Any) -> None:
    _update_tpe_object(db, run, AutoTpeRun, updates)


def update_tpe_trial(db: Session, trial: AutoTpeTrial, **updates: Any) -> None:
    _update_tpe_object(db, trial, AutoTpeTrial, updates)


def transition_tpe_run(
    db: Session,
    run: AutoTpeRun,
    status: str,
    *,
    details: dict[str, Any] | None = None,
    **updates: Any,
) -> None:
    _transition_tpe_object(
        db,
        run,
        AutoTpeRun,
        status,
        details=details,
        updates=updates,
    )


def transition_tpe_trial(
    db: Session,
    trial: AutoTpeTrial,
    status: str,
    *,
    details: dict[str, Any] | None = None,
    **updates: Any,
) -> None:
    _transition_tpe_object(
        db,
        trial,
        AutoTpeTrial,
        status,
        details=details,
        updates=updates,
    )


def register_tpe_graph_promotion(
    db: Session,
    *,
    run_id: str,
    knowledge_base_id: str,
    best_trial_id: str,
    selected_theta_hash: str,
    selected_graph_runtime_settings_hash: str,
) -> None:
    handles: dict[str, TpePromotionHandle] = db.info.setdefault(TPE_PROMOTION_SESSION_KEY, {})
    existing = handles.get(run_id)
    if existing is not None:
        existing.knowledge_base_id = knowledge_base_id
        existing.best_trial_id = best_trial_id
        existing.selected_theta_hash = selected_theta_hash
        existing.selected_graph_runtime_settings_hash = (
            selected_graph_runtime_settings_hash
        )
        return
    lock_key, owner_connection = _acquire_owner_connection(db, run_id)
    handle = TpePromotionHandle(
        run_id=run_id,
        knowledge_base_id=knowledge_base_id,
        best_trial_id=best_trial_id,
        selected_theta_hash=selected_theta_hash,
        selected_graph_runtime_settings_hash=selected_graph_runtime_settings_hash,
        owner_lock_key=lock_key,
        owner_connection=owner_connection,
    )
    try:
        _annotate_owner_fence(db, run_id, lock_key)
    except Exception:
        _release_owner_connection(handle)
        raise
    handles[run_id] = handle


def bind_tpe_graph_promotion_state(db: Session, *, run_id: str, relation_state_id: str) -> None:
    handles: dict[str, TpePromotionHandle] = db.info.setdefault(TPE_PROMOTION_SESSION_KEY, {})
    handle = handles.get(run_id)
    if handle is None:
        run = db.get(AutoTpeRun, run_id)
        if run is None or run.status != TPE_SELECTED_PENDING_STATUS or not run.best_trial_id or not run.selected_theta_hash:
            return
        lock_key, owner_connection = _acquire_owner_connection(db, run.id)
        handle = TpePromotionHandle(
            run_id=run.id,
            knowledge_base_id=run.knowledge_base_id,
            best_trial_id=run.best_trial_id,
            selected_theta_hash=run.selected_theta_hash,
            selected_graph_runtime_settings_hash=str(
                run.selected_graph_runtime_settings_hash or ""
            ),
            owner_lock_key=lock_key,
            owner_connection=owner_connection,
        )
        try:
            _annotate_owner_fence(db, run.id, lock_key)
        except Exception:
            _release_owner_connection(handle)
            raise
        handles[run_id] = handle
    handle.relation_state_id = relation_state_id


def _update_durable_run_row(
    row: AutoTpeRun,
    *,
    status: str,
    details: dict[str, Any],
    relation_state_id: str | None = None,
    failure_code: str | None = None,
    blocking_reasons: list[str] | None = None,
    last_error: str | None = None,
) -> None:
    _validate_status_transition(AutoTpeRun, str(row.status), status)
    row.status = status
    row.chunk_relation_graph_state_id = relation_state_id
    row.failure_code = failure_code
    row.blocking_reasons_json = list(blocking_reasons or [])
    row.last_error = last_error
    row.completed_at = _now()
    row.diagnostics_json = _transition_diagnostics(
        row.diagnostics_json,
        status=status,
        durability_mode="independent_postgresql_transaction",
        details=details,
    )


def _hard_gate_all_passed(hard_gate: dict[str, Any] | None) -> bool:
    gates = dict(hard_gate or {})
    return TPE_REQUIRED_HARD_GATES.issubset(gates) and all(
        isinstance(gate, dict) and gate.get("passed") is True
        for gate in gates.values()
    )


def _gate_profile_is_valid(
    gate_profile: dict[str, Any] | None,
    gate_profile_hash: str | None,
) -> bool:
    profile = dict(gate_profile or {})
    if (
        not gate_profile_hash
        or profile.get("protocol") != TPE_GATE_PROFILE_PROTOCOL_VERSION
        or stable_hash(profile) != gate_profile_hash
    ):
        return False
    thresholds = dict(profile.get("hard_gate_thresholds") or {})
    return TPE_REQUIRED_HARD_GATES.issubset(thresholds) and all(
        isinstance(thresholds.get(name), dict)
        and thresholds[name].get("direction") in {"min", "max"}
        and isinstance(thresholds[name].get("threshold"), (int, float))
        for name in TPE_REQUIRED_HARD_GATES
    )


def _hard_gate_matches_profile(
    hard_gate: dict[str, Any] | None,
    gate_profile: dict[str, Any] | None,
) -> bool:
    gates = dict(hard_gate or {})
    thresholds = dict((gate_profile or {}).get("hard_gate_thresholds") or {})
    if not TPE_REQUIRED_HARD_GATES.issubset(gates) or not TPE_REQUIRED_HARD_GATES.issubset(
        thresholds
    ):
        return False
    for name in TPE_REQUIRED_HARD_GATES:
        gate = gates.get(name)
        profile_gate = thresholds.get(name)
        if not isinstance(gate, dict) or not isinstance(profile_gate, dict):
            return False
        try:
            threshold_matches = float(gate.get("threshold")) == float(
                profile_gate.get("threshold")
            )
        except (TypeError, ValueError):
            return False
        if not threshold_matches or gate.get("direction") != profile_gate.get("direction"):
            return False
    return True


def _strict_theta_identity(theta: dict[str, Any] | None) -> dict[str, Any] | None:
    # Local import avoids the auto_tpe -> tpe_audit module cycle while keeping
    # one canonical preflight/search-space/calibration implementation.
    from app.services.auto_tpe import (
        _protocol_hash,
        preflight_theta,
        theta_calibration_audit,
        tpe_search_space_hash,
    )

    payload = dict(theta or {})
    if not payload or preflight_theta(payload):
        return None
    try:
        calibration = theta_calibration_audit(payload)
    except (TypeError, ValueError):
        return None
    return {
        "theta_hash": stable_hash(
            {key: payload.get(key) for key in sorted(payload)}
        ),
        "tpe_protocol_hash": _protocol_hash(),
        "tpe_search_space_hash": tpe_search_space_hash(),
        "calibration": calibration,
    }


def _trial_calibration_identity_matches(
    trial: AutoTpeTrial,
    identity: dict[str, Any],
) -> bool:
    calibration = dict(identity["calibration"])
    diagnostics = dict(trial.diagnostics_json or {})
    diagnostic_audit = dict(diagnostics.get("theta_calibration_audit") or {})
    simulated = dict(diagnostics.get("simulated_edge_type_calibration") or {})
    return bool(
        trial.tpe_search_space_hash == identity["tpe_search_space_hash"]
        and trial.edge_distance_protocol == calibration["edge_distance_protocol"]
        and trial.edge_distance_protocol_hash
        == calibration["edge_distance_protocol_hash"]
        and trial.edge_type_calibration_protocol
        == calibration["edge_type_calibration_protocol"]
        and trial.edge_type_calibration_protocol_hash
        == calibration["edge_type_calibration_protocol_hash"]
        and dict(trial.calibration_params_json or {})
        == dict(calibration["calibration_params"])
        and trial.calibration_params_hash == calibration["calibration_params_hash"]
        and trial.edge_type_calibration_config_hash
        == calibration["edge_type_calibration_config_hash"]
        and diagnostic_audit == calibration
        and simulated.get("protocol_version")
        == calibration["edge_type_calibration_protocol"]
        and simulated.get("protocol_hash")
        == calibration["edge_type_calibration_protocol_hash"]
        and simulated.get("edge_distance_protocol_version")
        == calibration["edge_distance_protocol"]
        and simulated.get("edge_distance_protocol_hash")
        == calibration["edge_distance_protocol_hash"]
        and dict(simulated.get("params") or {})
        == dict(calibration["calibration_params"])
        and simulated.get("calibration_params_hash")
        == calibration["calibration_params_hash"]
        and simulated.get("edge_type_calibration_config_hash")
        == calibration["edge_type_calibration_config_hash"]
    )


def _run_calibration_identity_matches(
    run: AutoTpeRun,
    identity: dict[str, Any],
) -> bool:
    calibration = dict(identity["calibration"])
    return bool(
        run.protocol_hash == identity["tpe_protocol_hash"]
        and run.tpe_search_space_hash == identity["tpe_search_space_hash"]
        and run.graph_operating_point_protocol
        == calibration["graph_operating_point_protocol"]
        and run.selected_edge_distance_protocol
        == calibration["edge_distance_protocol"]
        and run.selected_edge_distance_protocol_hash
        == calibration["edge_distance_protocol_hash"]
        and run.selected_edge_type_calibration_protocol
        == calibration["edge_type_calibration_protocol"]
        and run.selected_edge_type_calibration_protocol_hash
        == calibration["edge_type_calibration_protocol_hash"]
        and dict(run.selected_calibration_params_json or {})
        == dict(calibration["calibration_params"])
        and run.selected_calibration_params_hash
        == calibration["calibration_params_hash"]
        and run.selected_edge_type_calibration_config_hash
        == calibration["edge_type_calibration_config_hash"]
    )


def tpe_trial_is_valid(trial: AutoTpeTrial | None) -> bool:
    if trial is None or trial.status != "completed" or trial.failure_code is not None:
        return False
    if not all(
        (
            trial.chunk_scope_hash,
            trial.embedding_model,
            trial.embedding_text_version,
            trial.candidate_adjacency_hash,
            trial.runtime_settings_hash,
            trial.gate_profile_hash,
            trial.finished_at,
        )
    ):
        return False
    try:
        objective_is_finite = trial.objective_score is not None and math.isfinite(
            float(trial.objective_score)
        )
    except (TypeError, ValueError):
        return False
    identity = _strict_theta_identity(trial.sampled_theta_json)
    return bool(
        objective_is_finite
        and identity is not None
        and identity["theta_hash"] == trial.theta_hash
        and _trial_calibration_identity_matches(trial, identity)
        and _hard_gate_all_passed(trial.hard_gate_json)
        and _gate_profile_is_valid(trial.gate_profile_json, trial.gate_profile_hash)
        and _hard_gate_matches_profile(trial.hard_gate_json, trial.gate_profile_json)
    )


def _promotion_integrity_checks(
    audit: Session,
    row: AutoTpeRun,
    state: ChunkRelationGraphState | None,
    *,
    handle: TpePromotionHandle | None = None,
    expected_run_status: str = TPE_SELECTED_PENDING_STATUS,
) -> dict[str, bool]:
    trial = audit.get(AutoTpeTrial, row.best_trial_id) if row.best_trial_id else None
    trial_theta_identity = (
        _strict_theta_identity(trial.sampled_theta_json)
        if trial is not None
        else None
    )
    run_theta_identity = _strict_theta_identity(row.selected_theta_json)
    state_theta_identity = (
        _strict_theta_identity(state.graph_operating_point_json)
        if state is not None
        else None
    )
    trial_theta_hash = (
        stable_hash(
            {
                key: (trial.sampled_theta_json or {}).get(key)
                for key in sorted(trial.sampled_theta_json or {})
            }
        )
        if trial is not None
        else None
    )
    run_theta_hash = stable_hash(
        {
            key: (row.selected_theta_json or {}).get(key)
            for key in sorted(row.selected_theta_json or {})
        }
    )
    state_theta_hash = (
        stable_hash(dict(state.graph_operating_point_json or {}))
        if state is not None
        else None
    )
    state_tpe_audit = (
        dict((state.diagnostics_json or {}).get("auto_tpe") or {})
        if state is not None
        else {}
    )
    state_diagnostics = dict(state.diagnostics_json or {}) if state is not None else {}
    run_diagnostics = dict(row.diagnostics_json or {})
    run_graph_runtime_settings_hash = str(
        row.selected_graph_runtime_settings_hash or ""
    )
    run_calibration = dict((run_theta_identity or {}).get("calibration") or {})
    run_gate_profile = dict(row.selected_gate_profile_json or {})
    state_gate_profile = dict(state_tpe_audit.get("gate_profile") or {})
    state_objective = state_tpe_audit.get("objective_score")
    try:
        state_objective_is_finite = state_objective is not None and math.isfinite(
            float(state_objective)
        )
    except (TypeError, ValueError):
        state_objective_is_finite = False
    run_objective = row.best_objective_score
    trial_objective = trial.objective_score if trial is not None else None
    run_objective_is_finite = bool(
        run_objective is not None and math.isfinite(float(run_objective))
    )
    trial_objective_is_finite = bool(
        trial_objective is not None and math.isfinite(float(trial_objective))
    )
    checks = {
        "run_status_matches": row.status == expected_run_status,
        "run_failure_clear": row.failure_code is None,
        "run_blocking_reasons_clear": not list(row.blocking_reasons_json or []),
        "run_best_trial_present": bool(row.best_trial_id),
        "run_best_objective_present": run_objective_is_finite,
        "run_runtime_settings_hash_present": bool(row.runtime_settings_hash),
        "run_graph_runtime_settings_hash_present": (
            len(run_graph_runtime_settings_hash) == 64
        ),
        "run_diagnostics_optimizer_runtime_settings_matches": bool(
            row.runtime_settings_hash
            and run_diagnostics.get("selected_runtime_settings_hash")
            == row.runtime_settings_hash
        ),
        "run_diagnostics_graph_runtime_settings_matches": bool(
            run_graph_runtime_settings_hash
            and run_diagnostics.get("selected_graph_runtime_settings_hash")
            == run_graph_runtime_settings_hash
        ),
        "run_gate_profile_valid": _gate_profile_is_valid(
            run_gate_profile,
            row.selected_gate_profile_hash,
        ),
        "run_hard_gate_passed": _hard_gate_all_passed(row.hard_gate_json),
        "run_theta_hash_present": bool(row.selected_theta_hash),
        "run_theta_preflight_valid": run_theta_identity is not None,
        "run_tpe_protocol_current": bool(
            run_theta_identity is not None
            and row.protocol_hash == run_theta_identity["tpe_protocol_hash"]
        ),
        "run_search_space_current": bool(
            run_theta_identity is not None
            and row.tpe_search_space_hash
            == run_theta_identity["tpe_search_space_hash"]
        ),
        "run_calibration_identity_matches": bool(
            run_theta_identity is not None
            and _run_calibration_identity_matches(row, run_theta_identity)
        ),
        "run_theta_payload_hash_matches": bool(
            row.selected_theta_json and run_theta_hash == row.selected_theta_hash
        ),
        "state_present": state is not None,
        "state_active": bool(state is not None and state.state == "active"),
        "state_knowledge_base_matches": bool(
            state is not None and state.knowledge_base_id == row.knowledge_base_id
        ),
        "state_chunk_version_matches": bool(
            state is not None and state.chunk_version == row.chunk_version
        ),
        "state_chunk_scope_matches": bool(
            state is not None and state.scope_hash == row.chunk_scope_hash
        ),
        "state_embedding_text_version_matches": bool(
            state is not None
            and state.embedding_text_version == row.embedding_text_version
        ),
        "state_runtime_settings_matches": bool(
            state is not None
            and state.runtime_settings_hash
            and state.runtime_settings_hash == run_graph_runtime_settings_hash
        ),
        "state_run_matches": bool(state is not None and state.auto_tpe_run_id == row.id),
        "state_best_trial_matches": bool(
            state is not None and state.auto_tpe_best_trial_id == row.best_trial_id
        ),
        "state_theta_matches": bool(
            state is not None and state.graph_operating_point_hash == row.selected_theta_hash
        ),
        "state_theta_preflight_valid": state_theta_identity is not None,
        "state_calibration_identity_matches": bool(
            state is not None
            and state_theta_identity is not None
            and run_theta_identity is not None
            and state_theta_identity["calibration"] == run_calibration
            and state.edge_distance_protocol_hash
            == run_calibration.get("edge_distance_protocol_hash")
            and state.edge_type_calibration_protocol_hash
            == run_calibration.get("edge_type_calibration_protocol_hash")
            and state_diagnostics.get("graph_operating_point_protocol")
            == run_calibration.get("graph_operating_point_protocol")
            and state_diagnostics.get("graph_operating_point_hash")
            == row.selected_theta_hash
            and state_diagnostics.get("calibration_params_hash")
            == run_calibration.get("calibration_params_hash")
            and state_diagnostics.get("edge_type_calibration_config_hash")
            == run_calibration.get("edge_type_calibration_config_hash")
        ),
        "state_theta_payload_hash_matches": bool(
            state is not None
            and state.graph_operating_point_json
            and state_theta_hash == state.graph_operating_point_hash
        ),
        "state_tpe_audit_present": bool(state_tpe_audit),
        "state_tpe_status_matches": bool(
            state_tpe_audit.get("status") == TPE_SELECTED_PENDING_STATUS
        ),
        "state_tpe_run_matches": bool(state_tpe_audit.get("run_id") == row.id),
        "state_tpe_best_trial_matches": bool(
            state_tpe_audit.get("best_trial_id") == row.best_trial_id
        ),
        "state_tpe_theta_matches": bool(
            state_tpe_audit.get("selected_theta_hash") == row.selected_theta_hash
        ),
        "state_tpe_objective_matches": bool(
            state_objective_is_finite
            and run_objective_is_finite
            and float(state_objective) == float(run_objective)
        ),
        "state_tpe_chunk_version_matches": bool(
            state_tpe_audit.get("chunk_version") == row.chunk_version
        ),
        "state_tpe_protocol_matches": bool(
            state_tpe_audit.get("protocol_hash") == row.protocol_hash
        ),
        "state_tpe_search_space_matches": bool(
            state_tpe_audit.get("tpe_search_space_hash")
            == row.tpe_search_space_hash
        ),
        "state_tpe_calibration_identity_matches": bool(
            state_tpe_audit.get("edge_distance_protocol")
            == row.selected_edge_distance_protocol
            and state_tpe_audit.get("edge_distance_protocol_hash")
            == row.selected_edge_distance_protocol_hash
            and state_tpe_audit.get("edge_type_calibration_protocol")
            == row.selected_edge_type_calibration_protocol
            and state_tpe_audit.get("edge_type_calibration_protocol_hash")
            == row.selected_edge_type_calibration_protocol_hash
            and dict(state_tpe_audit.get("calibration_params") or {})
            == dict(row.selected_calibration_params_json or {})
            and state_tpe_audit.get("calibration_params_hash")
            == row.selected_calibration_params_hash
            and state_tpe_audit.get("edge_type_calibration_config_hash")
            == row.selected_edge_type_calibration_config_hash
        ),
        "state_tpe_runtime_settings_matches": bool(
            state_tpe_audit.get("runtime_settings_hash")
            == run_graph_runtime_settings_hash
        ),
        "state_tpe_optimizer_runtime_settings_matches": bool(
            state_tpe_audit.get("optimizer_runtime_settings_hash")
            == row.runtime_settings_hash
        ),
        "state_tpe_gate_profile_hash_matches": bool(
            state_tpe_audit.get("gate_profile_hash") == row.selected_gate_profile_hash
        ),
        "state_tpe_gate_profile_payload_matches": bool(
            state_gate_profile
            and stable_hash(state_gate_profile) == row.selected_gate_profile_hash
            and state_gate_profile == run_gate_profile
        ),
        "best_trial_present": trial is not None,
        "best_trial_valid": tpe_trial_is_valid(trial),
        "best_trial_theta_preflight_valid": trial_theta_identity is not None,
        "best_trial_calibration_identity_matches": bool(
            trial is not None
            and trial_theta_identity is not None
            and _trial_calibration_identity_matches(trial, trial_theta_identity)
            and trial_theta_identity.get("calibration") == run_calibration
            and trial.tpe_search_space_hash == row.tpe_search_space_hash
        ),
        "best_trial_run_matches": bool(trial is not None and trial.run_id == row.id),
        "best_trial_knowledge_base_matches": bool(
            trial is not None and trial.knowledge_base_id == row.knowledge_base_id
        ),
        "best_trial_build_batch_matches": bool(
            trial is not None and trial.build_batch_id == row.batch_id
        ),
        "best_trial_chunk_scope_matches": bool(
            trial is not None and trial.chunk_scope_hash == row.chunk_scope_hash
        ),
        "best_trial_embedding_model_matches": bool(
            trial is not None and trial.embedding_model == row.embedding_model
        ),
        "best_trial_embedding_text_version_matches": bool(
            trial is not None
            and trial.embedding_text_version == row.embedding_text_version
        ),
        "best_trial_runtime_settings_matches": bool(
            trial is not None and trial.runtime_settings_hash == row.runtime_settings_hash
        ),
        "best_trial_gate_profile_matches": bool(
            trial is not None
            and trial.gate_profile_hash == row.selected_gate_profile_hash
            and dict(trial.gate_profile_json or {}) == run_gate_profile
        ),
        "best_trial_completed": bool(
            trial is not None
            and trial.status == "completed"
            and trial_objective_is_finite
        ),
        "best_trial_failure_clear": bool(
            trial is not None and trial.failure_code is None
        ),
        "best_trial_hard_gate_passed": bool(
            trial is not None and _hard_gate_all_passed(trial.hard_gate_json)
        ),
        "best_trial_hard_gate_matches_run": bool(
            trial is not None
            and stable_hash(dict(trial.hard_gate_json or {}))
            == stable_hash(dict(row.hard_gate_json or {}))
        ),
        "best_trial_candidate_adjacency_present": bool(
            trial is not None and trial.candidate_adjacency_hash
        ),
        "best_trial_finished": bool(trial is not None and trial.finished_at is not None),
        "best_trial_objective_matches": bool(
            run_objective_is_finite
            and trial_objective_is_finite
            and float(run_objective) == float(trial_objective)
        ),
        "best_trial_theta_matches": bool(
            trial is not None and trial.theta_hash == row.selected_theta_hash
        ),
        "best_trial_theta_payload_hash_matches": bool(
            trial is not None
            and trial.sampled_theta_json
            and trial_theta_hash == trial.theta_hash
        ),
    }
    if handle is not None:
        checks.update(
            {
                "handle_run_matches": handle.run_id == row.id,
                "handle_knowledge_base_matches": handle.knowledge_base_id == row.knowledge_base_id,
                "handle_best_trial_matches": handle.best_trial_id == row.best_trial_id,
                "handle_theta_matches": handle.selected_theta_hash == row.selected_theta_hash,
                "handle_graph_runtime_settings_matches": (
                    handle.selected_graph_runtime_settings_hash
                    == run_graph_runtime_settings_hash
                ),
                "handle_state_matches": handle.relation_state_id == (state.id if state else None),
            }
        )
    return checks


def tpe_run_has_valid_active_promotion(db: Session, run: AutoTpeRun) -> bool:
    if run.status != "completed" or not run.chunk_relation_graph_state_id:
        return False
    state = db.get(ChunkRelationGraphState, run.chunk_relation_graph_state_id)
    checks = _promotion_integrity_checks(
        db,
        run,
        state,
        expected_run_status="completed",
    )
    return all(checks.values())


def _finalize_promotion(origin_db: Session, handle: TpePromotionHandle, *, committed: bool) -> None:
    if _dialect_name(origin_db) != "postgresql":
        # SQLite is an explicitly non-durable unit-test adapter. Once its main
        # transaction commits, a separate session can still finalize the row;
        # after rollback the row itself correctly disappears with the fixture.
        if committed and _dialect_name(origin_db) == "sqlite" and _running_under_pytest():
            with _audit_session(origin_db) as audit:
                row = audit.get(AutoTpeRun, handle.run_id, with_for_update=True)
                state = audit.get(ChunkRelationGraphState, handle.relation_state_id) if handle.relation_state_id else None
                if row is not None:
                    integrity_checks = _promotion_integrity_checks(
                        audit,
                        row,
                        state,
                        handle=handle,
                    )
                    integrity_ok = all(integrity_checks.values())
                    target_status = "completed" if integrity_ok else "failed"
                    _validate_status_transition(AutoTpeRun, str(row.status), target_status)
                    row.status = target_status
                    row.chunk_relation_graph_state_id = state.id if integrity_ok and state is not None else None
                    row.failure_code = None if integrity_ok else "active_relation_graph_promotion_integrity_failed"
                    row.blocking_reasons_json = [] if integrity_ok else ["active_relation_graph_promotion_integrity_failed"]
                    row.last_error = None if integrity_ok else "Committed relation graph does not match the selected TPE run/trial/theta"
                    row.completed_at = _now()
                    row.diagnostics_json = _transition_diagnostics(
                        row.diagnostics_json,
                        status=row.status,
                        durability_mode="sqlite_pytest_non_durable_adapter_v1",
                        details={
                            "phase": "graph_transaction_committed",
                            "relation_state_id": state.id if state is not None else None,
                            "promotion_integrity_ok": integrity_ok,
                            "promotion_integrity_checks": integrity_checks,
                        },
                    )
                    audit.commit()
        return
    with _audit_session(origin_db) as audit:
        row = audit.get(AutoTpeRun, handle.run_id, with_for_update=True)
        if row is None or row.status != TPE_SELECTED_PENDING_STATUS:
            return
        if not committed:
            _update_durable_run_row(
                row,
                status="failed",
                details={
                    "phase": "graph_transaction_rolled_back",
                    "blocking_reasons": ["active_relation_graph_transaction_rolled_back"],
                    "retry_boundary": "next_graph_build",
                },
                failure_code="active_relation_graph_transaction_rolled_back",
                blocking_reasons=["active_relation_graph_transaction_rolled_back"],
                last_error="Active relation graph transaction rolled back after TPE selection",
            )
            audit.commit()
            return
        state = audit.get(ChunkRelationGraphState, handle.relation_state_id) if handle.relation_state_id else None
        integrity_checks = _promotion_integrity_checks(
            audit,
            row,
            state,
            handle=handle,
        )
        integrity_ok = all(integrity_checks.values())
        if not integrity_ok:
            _update_durable_run_row(
                row,
                status="failed",
                details={
                    "phase": "graph_transaction_committed",
                    "relation_state_id": handle.relation_state_id,
                    "promotion_integrity_ok": False,
                    "promotion_integrity_checks": integrity_checks,
                    "blocking_reasons": ["active_relation_graph_promotion_integrity_failed"],
                    "retry_boundary": "next_graph_build",
                },
                failure_code="active_relation_graph_promotion_integrity_failed",
                blocking_reasons=["active_relation_graph_promotion_integrity_failed"],
                last_error="Committed relation graph does not match the selected TPE run/trial/theta",
            )
        else:
            _update_durable_run_row(
                row,
                status="completed",
                details={
                    "phase": "graph_transaction_committed",
                    "relation_state_id": state.id,
                    "promotion_integrity_ok": True,
                    "promotion_integrity_checks": integrity_checks,
                    "retry_boundary": "none",
                },
                relation_state_id=state.id,
            )
        audit.commit()


def _consume_promotion_handles(session: Session, *, committed: bool) -> None:
    handles = dict(session.info.pop(TPE_PROMOTION_SESSION_KEY, {}) or {})
    for handle in handles.values():
        try:
            _finalize_promotion(session, handle, committed=committed)
        except Exception:
            # The fact transaction has already committed or rolled back. A
            # selected_pending row is deliberately recoverable by reconciliation.
            pass
        finally:
            try:
                _release_owner_connection(handle)
            except Exception:
                pass


@event.listens_for(Session, "after_commit")
def _tpe_audit_after_commit(session: Session) -> None:
    if session.in_nested_transaction():
        return
    _consume_promotion_handles(session, committed=True)


@event.listens_for(Session, "after_rollback")
def _tpe_audit_after_rollback(session: Session) -> None:
    if session.in_nested_transaction():
        return
    _consume_promotion_handles(session, committed=False)


@event.listens_for(Session, "after_transaction_end")
def _tpe_audit_after_transaction_end(
    session: Session,
    transaction: SessionTransaction,
) -> None:
    if transaction.parent is not None:
        return
    if session.info.get(TPE_PROMOTION_SESSION_KEY):
        _consume_promotion_handles(session, committed=False)


def promotion_lease_expires_at() -> str:
    return (_now() + timedelta(seconds=TPE_PROMOTION_LEASE_SECONDS)).isoformat()


def trial_lease_expires_at(timeout_seconds: float) -> str:
    return (_now() + timedelta(seconds=max(1.0, float(timeout_seconds)) + 30.0)).isoformat()


def _lease_expired(diagnostics: dict[str, Any] | None, key: str) -> bool:
    raw = (diagnostics or {}).get(key)
    if not isinstance(raw, str) or not raw:
        return True
    try:
        return datetime.fromisoformat(raw) <= _now()
    except ValueError:
        return True


def reconcile_tpe_audit(
    db: Session,
    *,
    knowledge_base_id: str | None = None,
    include_unexpired: bool = False,
) -> dict[str, int]:
    empty_stats = {
        "checked": 0,
        "completed": 0,
        "failed": 0,
        "skipped_unexpired": 0,
        "skipped_active_owner": 0,
        "skipped_active_resource": 0,
    }
    if _dialect_name(db) != "postgresql":
        return empty_stats
    stats = dict(empty_stats)
    with (
        _engine_for(db).connect() as connection,
        Session(
            bind=connection,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        ) as audit,
    ):
        acquired_owner_locks: list[int] = []
        resource_locks: dict[str, tuple[int, bool]] = {}
        try:
            query = select(AutoTpeRun).where(
                AutoTpeRun.status.in_({"running", TPE_SELECTED_PENDING_STATUS})
            )
            if knowledge_base_id:
                query = query.where(AutoTpeRun.knowledge_base_id == knowledge_base_id)
            runs = list(audit.scalars(query.order_by(AutoTpeRun.created_at.asc())).all())
            for row in runs:
                stats["checked"] += 1
                resource_lock = resource_locks.get(row.knowledge_base_id)
                if resource_lock is None:
                    resource_lock = _try_reconcile_resource_lock(
                        audit,
                        row.knowledge_base_id,
                    )
                    resource_locks[row.knowledge_base_id] = resource_lock
                if not resource_lock[1]:
                    stats["skipped_active_resource"] += 1
                    continue
                owner_lock_key, owner_available = _try_reconcile_owner_lock(audit, row.id)
                if not owner_available:
                    stats["skipped_active_owner"] += 1
                    continue
                acquired_owner_locks.append(owner_lock_key)
                locked_row = audit.get(
                    AutoTpeRun,
                    row.id,
                    populate_existing=True,
                    with_for_update=True,
                )
                if locked_row is None or locked_row.status not in {
                    "running",
                    TPE_SELECTED_PENDING_STATUS,
                }:
                    continue
                row = locked_row

                if row.status == TPE_SELECTED_PENDING_STATUS:
                    state = audit.scalar(
                        select(ChunkRelationGraphState)
                        .where(ChunkRelationGraphState.auto_tpe_run_id == row.id)
                        .order_by(ChunkRelationGraphState.created_at.desc())
                    )
                    if state is not None:
                        integrity_checks = _promotion_integrity_checks(
                            audit,
                            row,
                            state,
                        )
                        integrity_ok = all(integrity_checks.values())
                        if integrity_ok:
                            _update_durable_run_row(
                                row,
                                status="completed",
                                details={
                                    "phase": "promotion_reconciled_after_process_interruption",
                                    "relation_state_id": state.id,
                                    "promotion_integrity_ok": True,
                                    "promotion_integrity_checks": integrity_checks,
                                    "retry_boundary": "none",
                                },
                                relation_state_id=state.id,
                            )
                            stats["completed"] += 1
                        else:
                            _update_durable_run_row(
                                row,
                                status="failed",
                                details={
                                    "phase": "promotion_reconciled_after_process_interruption",
                                    "relation_state_id": state.id,
                                    "promotion_integrity_ok": False,
                                    "promotion_integrity_checks": integrity_checks,
                                    "blocking_reasons": ["active_relation_graph_promotion_integrity_failed"],
                                    "retry_boundary": "next_graph_build",
                                },
                                failure_code="active_relation_graph_promotion_integrity_failed",
                                blocking_reasons=["active_relation_graph_promotion_integrity_failed"],
                                last_error="Recovered relation graph does not match the selected TPE run",
                            )
                            stats["failed"] += 1
                        continue
                    if not include_unexpired and not _lease_expired(
                        row.diagnostics_json,
                        "promotion_lease_expires_at",
                    ):
                        stats["skipped_unexpired"] += 1
                        continue
                    _update_durable_run_row(
                        row,
                        status="failed",
                        details={
                            "phase": "promotion_reconciled_without_committed_graph",
                            "blocking_reasons": ["active_relation_graph_process_interrupted"],
                            "retry_boundary": "next_graph_build",
                        },
                        failure_code="active_relation_graph_process_interrupted",
                        blocking_reasons=["active_relation_graph_process_interrupted"],
                        last_error="TPE selection survived but no committed relation graph was found",
                    )
                    stats["failed"] += 1
                    continue

                running_trials = list(
                    audit.scalars(
                        select(AutoTpeTrial).where(
                            AutoTpeTrial.run_id == row.id,
                            AutoTpeTrial.status == "running",
                        ).with_for_update()
                    ).all()
                )
                expired_trials = [
                    trial
                    for trial in running_trials
                    if include_unexpired
                    or _lease_expired(trial.diagnostics_json, "trial_lease_expires_at")
                ]
                unexpired_trials = [
                    trial for trial in running_trials if trial not in expired_trials
                ]
                remaining_live_trial_ids = sorted(trial.id for trial in unexpired_trials)
                for trial in expired_trials:
                    _validate_status_transition(AutoTpeTrial, str(trial.status), "failed")
                    trial.status = "failed"
                    trial.failure_code = "trial_process_interrupted"
                    trial.finished_at = _now()
                    trial.diagnostics_json = _transition_diagnostics(
                        trial.diagnostics_json,
                        status="failed",
                        durability_mode="independent_postgresql_transaction",
                        details={
                            "phase": "trial_reconciled_after_process_interruption",
                            "retry_boundary": (
                                "next_trial_boundary"
                                if remaining_live_trial_ids
                                else "next_graph_build"
                            ),
                            "blocking_reasons": ["trial_process_interrupted"],
                            "remaining_live_trial_ids": remaining_live_trial_ids,
                        },
                    )
                if unexpired_trials:
                    stats["skipped_unexpired"] += 1
                    continue
                if (
                    not running_trials
                    and not include_unexpired
                    and not _lease_expired(row.diagnostics_json, "run_lease_expires_at")
                ):
                    stats["skipped_unexpired"] += 1
                    continue
                _update_durable_run_row(
                    row,
                    status="failed",
                    details={
                        "phase": "run_reconciled_after_process_interruption",
                        "blocking_reasons": ["tpe_process_interrupted"],
                        "retry_boundary": "next_graph_build",
                    },
                    failure_code="tpe_process_interrupted",
                    blocking_reasons=["tpe_process_interrupted"],
                    last_error="TPE process stopped before reaching a terminal run state",
                )
                stats["failed"] += 1
            audit.commit()
        except BaseException:
            audit.rollback()
            raise
        finally:
            release_keys = list(acquired_owner_locks)
            release_keys.extend(
                resource_lock_key
                for resource_lock_key, resource_available in resource_locks.values()
                if resource_available
            )
            for lock_key in release_keys:
                try:
                    _release_reconcile_owner_lock(connection, lock_key)
                except Exception:
                    connection.invalidate()
                    break
            if not connection.invalidated:
                connection.commit()
    return stats
