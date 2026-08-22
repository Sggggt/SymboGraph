from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from _context_graph_maintenance import (
    prepare_runtime_for_model_io,
    session_scope,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operate rebuild-required Runtime Settings through dry-run, bounded "
            "shadow build, measured evaluation, atomic promotion and rollback."
        )
    )
    parser.add_argument(
        "action",
        choices=(
            "stage",
            "build",
            "evaluate",
            "promote",
            "rollback",
            "status",
            "reconcile-activation",
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--knowledge-base-id", action="append", default=[])
    parser.add_argument("--settings-json", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--build-id")
    parser.add_argument("--reason")
    return parser.parse_args()


def _require(value, message: str):
    if value is None or value == "" or value == []:
        raise SystemExit(message)
    return value


def _read_settings(path: Path | None) -> dict:
    settings_path = _require(path, "stage requires --settings-json")
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--settings-json must contain one JSON object")
    return payload


async def _main_async(args: argparse.Namespace) -> dict:
    from app.services.runtime_settings_lifecycle import (
        apply_runtime_settings_activation_intent,
        build_runtime_settings_shadow,
        evaluate_runtime_settings_shadow,
        preview_runtime_settings_candidate,
        promote_runtime_settings_candidate,
        reconcile_runtime_settings_activation_intents,
        record_runtime_settings_build_failure,
        rollback_runtime_settings_candidate,
        runtime_settings_candidate_payload,
        stage_runtime_settings_candidate,
    )

    if args.action == "reconcile-activation":
        return reconcile_runtime_settings_activation_intents(
            candidate_id=args.candidate_id,
            dry_run=not args.execute,
        )

    with session_scope() as db:
        if args.action == "stage":
            knowledge_base_ids = _require(
                args.knowledge_base_id,
                "stage requires at least one --knowledge-base-id",
            )
            requested = _read_settings(args.settings_json)
            preview = preview_runtime_settings_candidate(
                db,
                knowledge_base_ids=knowledge_base_ids,
                requested_settings=requested,
            )
            if not args.execute:
                db.rollback()
                return {"preview": preview, "staged": False, "active_mutated": False}
            candidate, builds = stage_runtime_settings_candidate(
                db,
                knowledge_base_ids=knowledge_base_ids,
                requested_settings=requested,
                source="manage_runtime_settings_candidate.py",
            )
            db.commit()
            return {
                "preview": preview,
                "candidate": runtime_settings_candidate_payload(db, candidate.id),
                "build_ids": [build.id for build in builds],
                "active_mutated": False,
            }

        candidate_id = _require(
            args.candidate_id,
            f"{args.action} requires --candidate-id",
        )
        status = runtime_settings_candidate_payload(db, candidate_id)

        if args.action == "status":
            return {"candidate": status, "active_mutated": False}

        if args.action in {"build", "evaluate"}:
            build_ids = [
                str(build["id"])
                for build in status.get("builds") or []
                if args.build_id is None or str(build["id"]) == args.build_id
            ]
            if not build_ids:
                raise SystemExit("No matching Runtime Settings shadow build")
            if not args.execute:
                return {
                    "candidate": status,
                    "would_action": args.action,
                    "build_ids": build_ids,
                    "active_mutated": False,
                }
            if args.action == "build":
                prepare_runtime_for_model_io()
                for build_id in build_ids:
                    try:
                        await build_runtime_settings_shadow(db, build_id=build_id)
                        db.commit()
                    except Exception as exc:
                        db.rollback()
                        record_runtime_settings_build_failure(
                            db,
                            build_id=build_id,
                            error_type=type(exc).__name__,
                        )
                        db.commit()
                        raise RuntimeError(
                            "Runtime Settings shadow build failed; "
                            f"build_id={build_id}; error_type={type(exc).__name__}"
                        ) from None
            else:
                for build_id in build_ids:
                    evaluate_runtime_settings_shadow(db, build_id=build_id)
                db.commit()
            return {
                "candidate": runtime_settings_candidate_payload(db, candidate_id),
                "completed_action": args.action,
                "build_ids": build_ids,
                "active_mutated": False,
            }

        if args.action == "promote":
            if not args.execute:
                return {
                    "candidate": status,
                    "would_promote": status.get("status") == "evaluation_passed",
                    "active_mutated": False,
                }
            promotion = promote_runtime_settings_candidate(db, candidate_id)
            db.commit()
            activation = None
            if promotion.get("promoted") and promotion.get("activation_intent_id"):
                activation = apply_runtime_settings_activation_intent(
                    str(promotion["activation_intent_id"])
                )
                db.expire_all()
            return {
                "candidate": runtime_settings_candidate_payload(db, candidate_id),
                "promotion": promotion,
                "activation": activation,
            }

        reason = _require(
            args.reason,
            "rollback requires --reason (and --execute to mutate)",
        )
        if not args.execute:
            return {
                "candidate": status,
                "would_rollback": status.get("status")
                in {
                    "staged",
                    "building",
                    "evaluating",
                    "evaluation_passed",
                    "promotion_blocked",
                    "failed",
                    "promoted",
                },
                "reason": reason,
                "active_mutated": False,
            }
        candidate = rollback_runtime_settings_candidate(
            db,
            candidate_id,
            reason=reason,
        )
        db.commit()
        payload = runtime_settings_candidate_payload(db, candidate.id)
        rollback_intents = [
            intent
            for intent in payload.get("activation_intents") or []
            if intent.get("direction") == "rollback"
        ]
        activation = None
        if rollback_intents:
            activation = apply_runtime_settings_activation_intent(
                str(rollback_intents[-1]["id"])
            )
            db.expire_all()
        elif not (payload.get("diagnostics") or {}).get("unpromoted_abandon"):
            raise RuntimeError("Rollback committed without a durable activation intent")
        return {
            "candidate": runtime_settings_candidate_payload(db, candidate.id),
            "activation": activation,
        }


def main() -> None:
    args = parse_args()
    if args.action == "status" and args.execute:
        raise SystemExit("status is read-only; omit --execute")
    result = asyncio.run(_main_async(args))
    payload = {
        "script": "manage_runtime_settings_candidate",
        "action": args.action,
        "mode": "execute" if args.execute else "dry_run",
        "execute": bool(args.execute),
        "result": result,
        "impact": (
            "explicitly advances only the selected durable lifecycle transition; "
            "promotion/rollback apply a post-commit shared-env/version intent"
            if args.execute
            else "read-only validation/status; no PostgreSQL/Qdrant/Redis/env writes"
        ),
        "gray_zone_decision_authority": "deterministic_local_rule_only",
        "gray_zone_model_call_count": 0,
    }
    report = write_report("manage_runtime_settings_candidate", payload)
    print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
