from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from typing import Any

from _context_graph_maintenance import (
    resolve_knowledge_base,
    session_scope,
    write_report,
)
from _destructive_cleanup_guard import (
    build_legacy_cleanup_inventory,
    build_stale_cleanup_inventory,
    combine_inventories,
    require_exact_confirmation,
)


LEGACY_PROFILE_KEYS = {
    "schema_pack",
    "concept_induction_policy",
    "parsing_strategy",
    "graph_strategy",
    "retrieval_strategy",
    "quality_policy",
    "signal_induction_policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory legacy derived state in read-only mode. Destructive "
            "execution requires the legacy acknowledgement plus exact KB id "
            "and complete inventory hash copied from the latest dry-run."
        )
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the exact confirmed legacy cleanup. Omit for dry-run.",
    )
    parser.add_argument(
        "--confirm-destroy-legacy",
        action="store_true",
        help="Additional destructive acknowledgement required with --execute.",
    )
    parser.add_argument(
        "--confirm-knowledge-base-id",
        help="With --execute, exactly repeat the resolved dry-run KB id.",
    )
    parser.add_argument(
        "--confirm-inventory-hash",
        help="With --execute, exactly repeat the complete dry-run inventory hash.",
    )
    parser.add_argument(
        "--delete-inactive-chunks",
        action="store_true",
        help="Include exact inactive chunk/version/dependent rows.",
    )
    parser.add_argument(
        "--clear-legacy-score-audit",
        action="store_true",
        help="Clear legacy score payloads only for the selected knowledge base.",
    )
    parser.add_argument(
        "--normalize-env",
        action="store_true",
        help=(
            "Remove deprecated keys/BOM only from the exact env identity/hash "
            "bound by the confirmed inventory and shared writer lock."
        ),
    )
    return parser.parse_args()


def legacy_profile_key_hits(profile_json: dict[str, Any]) -> list[str]:
    if not isinstance(profile_json, dict):
        return []
    return sorted(
        key for key in profile_json if key in LEGACY_PROFILE_KEYS
    )


def _legacy_blockers(legacy_inventory: dict[str, Any]) -> list[str]:
    profile_scope = legacy_inventory["scope_inventory"][
        "selected_active_profile_legacy_keys"
    ]
    profile_bindings = legacy_inventory["scope_inventory"][
        "selected_active_profile_bindings"
    ]
    blockers: list[str] = []
    if (
        int(profile_scope["count"]) > 0
        and int(profile_bindings["count"]) > 1
    ):
        blockers.append(
            "selected active profile is shared by another knowledge base; "
            "clone/bind a KB-local profile before destructive legacy scrub"
        )
    return blockers


def _exact_env_identity(
    legacy_inventory: dict[str, Any],
    *,
    normalize_env: bool,
) -> dict[str, Any] | None:
    env_scope = legacy_inventory["scope_inventory"]["shared_env_file"]
    if not normalize_env:
        if int(env_scope["count"]) != 0:
            raise SystemExit(
                "Shared env scope must be empty unless --normalize-env is set"
            )
        return None
    sample = list(env_scope.get("sample") or [])
    if (
        int(env_scope["count"]) != 1
        or bool(env_scope.get("sample_truncated"))
        or len(sample) != 1
        or not isinstance(sample[0].get("identity"), dict)
        or sample[0].get("identity_hash")
        != sample[0]["identity"].get("identity_hash")
    ):
        raise SystemExit(
            "Shared env exact identity/hash is missing from the confirmed "
            "inventory"
        )
    return dict(sample[0]["identity"])


def _combined_inventory(
    db,
    knowledge_base,
    args: argparse.Namespace,
    *,
    lock_legacy_rows: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cleanup_inventory = build_stale_cleanup_inventory(
        db,
        knowledge_base,
        delete_inactive_chunks=args.delete_inactive_chunks,
    )
    legacy_inventory = build_legacy_cleanup_inventory(
        db,
        knowledge_base,
        legacy_profile_keys=LEGACY_PROFILE_KEYS,
        clear_legacy_score_audit=args.clear_legacy_score_audit,
        normalize_env=args.normalize_env,
        lock_rows=lock_legacy_rows,
    )
    options = {
        "delete_inactive_chunks": bool(args.delete_inactive_chunks),
        "clear_legacy_score_audit": bool(
            args.clear_legacy_score_audit
        ),
        "normalize_env": bool(args.normalize_env),
    }
    impact = (
        "execute the exact stale-data cleanup inventory, scrub legacy keys "
        "only from the selected KB active profile, optionally clear score "
        "payloads only for that KB, and optionally normalize the exact "
        "identity/hash-bound shared env file"
    )
    combined = combine_inventories(
        operation="destroy_legacy_derived_data",
        knowledge_base_id=knowledge_base.id,
        knowledge_base_name=knowledge_base.name,
        options=options,
        impact=impact,
        components={
            "stale_cleanup": cleanup_inventory,
            "legacy_scope": legacy_inventory,
        },
    )
    return combined, cleanup_inventory, legacy_inventory


def main() -> None:
    args = parse_args()
    if args.execute and (
        not args.confirm_destroy_legacy
        or not args.confirm_knowledge_base_id
        or not args.confirm_inventory_hash
    ):
        raise SystemExit(
            "--execute requires --confirm-destroy-legacy, "
            "--confirm-knowledge-base-id, and --confirm-inventory-hash from "
            "the latest dry-run"
        )

    from sqlalchemy import select

    from app.models import (
        GraphRetrievalStep,
        RetrievalTrace,
        StrategyProfile,
    )
    from app.services.maintenance import (
        cleanup_stale_data,
        cleanup_stale_data_lock,
    )
    from app.services.runtime_settings import (
        RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION,
        RUNTIME_ENV_FILE_LOCK_PROTOCOL_VERSION,
        RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION,
        commit_runtime_env_file_mutation,
        env_sync_status,
        normalize_env_file_exact,
        require_runtime_env_file_identity,
        restore_runtime_env_file_mutation,
        runtime_env_file_lock,
    )
    from app.services.strategy_profiles import validate_profile_payload

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        cleanup_preview = None
        env_sync_snapshot = None
        env_lock_audit = None
        if args.execute:
            inventory, cleanup_inventory, legacy_inventory = (
                _combined_inventory(db, knowledge_base, args)
            )
        else:
            with cleanup_stale_data_lock(db, knowledge_base.id):
                inventory, cleanup_inventory, legacy_inventory = (
                    _combined_inventory(db, knowledge_base, args)
                )
                cleanup_preview = cleanup_stale_data(
                    db,
                    knowledge_base.id,
                    knowledge_base.name,
                    dry_run=True,
                    delete_inactive_chunks=args.delete_inactive_chunks,
                    lock_already_held=True,
                )
                env_sync_snapshot = env_sync_status()

        initial_env_identity = _exact_env_identity(
            legacy_inventory,
            normalize_env=bool(args.normalize_env),
        )
        payload: dict[str, Any] = {
            "script": "destroy_legacy_derived_data",
            "mode": "execute" if args.execute else "dry_run",
            "execute": bool(args.execute),
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "delete_inactive_chunks": bool(args.delete_inactive_chunks),
            "clear_legacy_score_audit": bool(
                args.clear_legacy_score_audit
            ),
            "normalize_env": bool(args.normalize_env),
            "inventory": inventory,
            "impact": inventory["impact"],
            "confirmation": {
                "required_on_execute": [
                    "--confirm-destroy-legacy",
                    "--confirm-knowledge-base-id",
                    "--confirm-inventory-hash",
                ],
                "knowledge_base_id_matches": (
                    args.confirm_knowledge_base_id == knowledge_base.id
                    if args.execute
                    else None
                ),
                "inventory_hash_matches": (
                    args.confirm_inventory_hash
                    == inventory["inventory_hash"]
                    if args.execute
                    else None
                ),
                "env_identity_hash_bound_by_inventory": (
                    initial_env_identity.get("identity_hash")
                    if initial_env_identity is not None
                    else None
                ),
            },
            "env_writer_protocol": {
                "lock_protocol_version": (
                    RUNTIME_ENV_FILE_LOCK_PROTOCOL_VERSION
                ),
                "cas_protocol_version": (
                    RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION
                ),
                "recovery_protocol_version": (
                    RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION
                ),
                "same_lock_as_runtime_settings_writer": True,
            },
        }
        blockers = _legacy_blockers(legacy_inventory)
        payload["blockers"] = blockers

        if not args.execute:
            payload["cleanup_stale_data"] = cleanup_preview
            payload.update(
                {
                    "scrubbed_profiles": 0,
                    "cleared_trace_scores": 0,
                    "cleared_step_scores": 0,
                    "env_normalized": False,
                    "env_sync": env_sync_snapshot,
                    "transaction": {
                        "database_committed": False,
                        "toctou_inventory_revalidated_under_lock": False,
                        "env_file_lock_acquired": bool(env_lock_audit),
                    },
                }
            )
        else:
            if blockers:
                raise SystemExit(
                    "Legacy cleanup blocked: " + "; ".join(blockers)
                )
            require_exact_confirmation(
                actual_knowledge_base_id=knowledge_base.id,
                actual_inventory_hash=inventory["inventory_hash"],
                confirmed_knowledge_base_id=args.confirm_knowledge_base_id,
                confirmed_inventory_hash=args.confirm_inventory_hash,
            )

            scrubbed_profiles = 0
            cleared_trace_scores = 0
            cleared_step_scores = 0
            env_normalized = False
            with cleanup_stale_data_lock(db, knowledge_base.id):
                env_lock_context = (
                    runtime_env_file_lock()
                    if args.normalize_env
                    else nullcontext(None)
                )
                with env_lock_context as env_lock_audit:
                    (
                        locked_inventory,
                        _locked_cleanup_inventory,
                        locked_legacy_inventory,
                    ) = _combined_inventory(
                        db,
                        knowledge_base,
                        args,
                        lock_legacy_rows=True,
                    )
                    require_exact_confirmation(
                        actual_knowledge_base_id=knowledge_base.id,
                        actual_inventory_hash=locked_inventory[
                            "inventory_hash"
                        ],
                        confirmed_knowledge_base_id=(
                            args.confirm_knowledge_base_id
                        ),
                        confirmed_inventory_hash=(
                            args.confirm_inventory_hash
                        ),
                    )
                    locked_blockers = _legacy_blockers(
                        locked_legacy_inventory
                    )
                    if locked_blockers:
                        raise SystemExit(
                            "Legacy cleanup blocked: "
                            + "; ".join(locked_blockers)
                        )
                    locked_env_identity = _exact_env_identity(
                        locked_legacy_inventory,
                        normalize_env=bool(args.normalize_env),
                    )
                    if locked_env_identity is not None:
                        require_runtime_env_file_identity(
                            locked_env_identity
                        )

                    locked_cleanup_preview = cleanup_stale_data(
                        db,
                        knowledge_base.id,
                        knowledge_base.name,
                        dry_run=True,
                        delete_inactive_chunks=args.delete_inactive_chunks,
                        lock_already_held=True,
                    )

                    profile_updates: list[
                        tuple[Any, dict[str, Any]]
                    ] = []
                    if knowledge_base.active_profile_id:
                        profiles = list(
                            db.scalars(
                                select(StrategyProfile)
                                .where(
                                    StrategyProfile.id
                                    == knowledge_base.active_profile_id
                                )
                                .with_for_update()
                            ).all()
                        )
                        for profile in profiles:
                            if not legacy_profile_key_hits(
                                profile.profile_json or {}
                            ):
                                continue
                            candidate, _warnings = validate_profile_payload(
                                profile.profile_json or {}
                            )
                            profile_updates.append((profile, candidate))

                    trace_rows: list[Any] = []
                    step_rows: list[Any] = []
                    if args.clear_legacy_score_audit:
                        trace_rows = list(
                            db.scalars(
                                select(RetrievalTrace)
                                .where(
                                    RetrievalTrace.knowledge_base_id
                                    == knowledge_base.id
                                )
                                .order_by(RetrievalTrace.id.asc())
                                .with_for_update()
                            ).all()
                        )
                        step_rows = list(
                            db.scalars(
                                select(GraphRetrievalStep)
                                .join(
                                    RetrievalTrace,
                                    GraphRetrievalStep.retrieval_trace_id
                                    == RetrievalTrace.id,
                                )
                                .where(
                                    RetrievalTrace.knowledge_base_id
                                    == knowledge_base.id
                                )
                                .order_by(GraphRetrievalStep.id.asc())
                                .with_for_update(of=GraphRetrievalStep)
                            ).all()
                        )

                    payload["blockers"] = locked_blockers
                    payload["locked_inventory_hash"] = locked_inventory[
                        "inventory_hash"
                    ]
                    payload["locked_cleanup_preview"] = (
                        locked_cleanup_preview
                    )
                    payload["confirmation"][
                        "locked_env_identity_hash"
                    ] = (
                        locked_env_identity.get("identity_hash")
                        if locked_env_identity is not None
                        else None
                    )
                    authorized_report = write_report(
                        "destroy_legacy_derived_data_authorized",
                        {
                            **payload,
                            "authorization_state": (
                                "all_components_confirmed_under_kb_and_env_fences_before_mutation"
                            ),
                        },
                    )
                    payload["authorized_plan_output"] = str(
                        authorized_report
                    )

                    env_receipt = None
                    try:
                        if locked_env_identity is not None:
                            env_receipt = normalize_env_file_exact(
                                expected_identity=locked_env_identity,
                                lock_already_held=True,
                            )
                            env_normalized = bool(
                                env_receipt.changed
                            )

                        for profile, candidate in profile_updates:
                            profile.profile_json = candidate
                            profile.profile_hash = candidate["profile_hash"]
                            scrubbed_profiles += 1
                        for trace in trace_rows:
                            if trace.scores_json:
                                trace.scores_json = {}
                                cleared_trace_scores += 1
                        for step in step_rows:
                            if step.score_json:
                                step.score_json = {}
                                cleared_step_scores += 1

                        payload["cleanup_stale_data"] = cleanup_stale_data(
                            db,
                            knowledge_base.id,
                            knowledge_base.name,
                            dry_run=False,
                            delete_inactive_chunks=(
                                args.delete_inactive_chunks
                            ),
                            lock_already_held=True,
                        )
                    except BaseException:
                        rollback = getattr(db, "rollback", None)
                        if callable(rollback):
                            rollback()
                        if env_receipt is not None:
                            restore_runtime_env_file_mutation(
                                env_receipt,
                                lock_already_held=True,
                            )
                        raise
                    else:
                        if env_receipt is not None:
                            commit_runtime_env_file_mutation(
                                env_receipt,
                                lock_already_held=True,
                            )
                    env_sync_snapshot = env_sync_status()

            payload.update(
                {
                    "scrubbed_profiles": scrubbed_profiles,
                    "cleared_trace_scores": cleared_trace_scores,
                    "cleared_step_scores": cleared_step_scores,
                    "env_normalized": env_normalized,
                    "env_sync": env_sync_snapshot,
                    "transaction": {
                        "database_committed": True,
                        "toctou_inventory_revalidated_under_lock": True,
                        "revalidated_inventory_hash": locked_inventory[
                            "inventory_hash"
                        ],
                        "legacy_scope_revalidated_with_row_locks": True,
                        "all_component_blockers_checked_before_mutation": True,
                        "env_file_lock_acquired": bool(env_lock_audit),
                        "env_exact_cas_precondition_used": bool(
                            args.normalize_env
                        ),
                        "env_rollback_capability_available": bool(
                            args.normalize_env
                        ),
                        "env_commit_point_crossed_after_database_commit": bool(
                            args.normalize_env
                        ),
                        "single_database_commit_owner": (
                            "cleanup_stale_data"
                        ),
                    },
                }
            )

        report = write_report("destroy_legacy_derived_data", payload)
        print(
            json.dumps(
                {"output": str(report), **payload},
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
