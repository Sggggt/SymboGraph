from __future__ import annotations

import argparse
import json
from typing import Any

from sqlalchemy import select

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


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
    parser = argparse.ArgumentParser(description="Destroy or scrub legacy derived state that must not drive the active four-layer path.")
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--knowledge-base-name")
    parser.add_argument("--execute", action="store_true", help="Apply destructive legacy cleanup. Omit for dry-run.")
    parser.add_argument("--confirm-destroy-legacy", action="store_true", help="Required with --execute.")
    parser.add_argument("--delete-inactive-chunks", action="store_true", help="Delete inactive chunk versions and their derived rows.")
    parser.add_argument("--clear-legacy-score-audit", action="store_true", help="Clear old non-empty scores_json/score_json audit payloads.")
    parser.add_argument("--normalize-env", action="store_true", help="Remove deprecated legacy env keys from .env when executing.")
    return parser.parse_args()


def legacy_profile_key_hits(profile_json: dict[str, Any]) -> list[str]:
    if not isinstance(profile_json, dict):
        return []
    return sorted(key for key in profile_json if key in LEGACY_PROFILE_KEYS)


def main() -> None:
    from app.models import GraphRetrievalStep, RetrievalTrace, StrategyProfile
    from app.services.maintenance import cleanup_stale_data
    from app.services.runtime_settings import env_sync_status, normalize_env_file
    from app.services.strategy_profiles import validate_profile_payload

    args = parse_args()
    if args.execute and not args.confirm_destroy_legacy:
        raise SystemExit("--execute requires --confirm-destroy-legacy")

    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id=args.knowledge_base_id, knowledge_base_name=args.knowledge_base_name)
        cleanup = cleanup_stale_data(
            db,
            knowledge_base.id,
            knowledge_base.name,
            dry_run=not args.execute,
            delete_inactive_chunks=args.delete_inactive_chunks,
        )
        profiles = db.scalars(select(StrategyProfile)).all()
        legacy_profiles = [
            {"profile_id": profile.id, "name": profile.name, "legacy_keys": legacy_profile_key_hits(profile.profile_json or {})}
            for profile in profiles
            if legacy_profile_key_hits(profile.profile_json or {})
        ]
        traces_with_scores = [trace.id for trace in db.scalars(select(RetrievalTrace)).all() if trace.scores_json]
        steps_with_scores = [step.id for step in db.scalars(select(GraphRetrievalStep)).all() if step.score_json]

        scrubbed_profiles = 0
        cleared_trace_scores = 0
        cleared_step_scores = 0
        env_normalized = False

        if args.execute:
            for profile in profiles:
                if not legacy_profile_key_hits(profile.profile_json or {}):
                    continue
                payload, _warnings = validate_profile_payload(profile.profile_json or {})
                profile.profile_json = payload
                profile.profile_hash = payload["profile_hash"]
                scrubbed_profiles += 1
            if args.clear_legacy_score_audit:
                for trace in db.scalars(select(RetrievalTrace)).all():
                    if trace.scores_json:
                        trace.scores_json = {}
                        cleared_trace_scores += 1
                for step in db.scalars(select(GraphRetrievalStep)).all():
                    if step.score_json:
                        step.score_json = {}
                        cleared_step_scores += 1
            db.commit()
            if args.normalize_env:
                env_normalized = normalize_env_file()

        payload = {
            "script": "destroy_legacy_derived_data",
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "execute": args.execute,
            "confirm_destroy_legacy": args.confirm_destroy_legacy,
            "delete_inactive_chunks": args.delete_inactive_chunks,
            "clear_legacy_score_audit": args.clear_legacy_score_audit,
            "normalize_env": args.normalize_env,
            "impact": (
                "delete inactive derived rows, scrub legacy profile keys, optionally clear legacy score audits and deprecated env keys"
                if args.execute
                else "no writes"
            ),
            "cleanup_stale_data": cleanup,
            "legacy_profiles": legacy_profiles,
            "retrieval_traces_with_legacy_scores": len(traces_with_scores),
            "graph_steps_with_legacy_scores": len(steps_with_scores),
            "scrubbed_profiles": scrubbed_profiles,
            "cleared_trace_scores": cleared_trace_scores,
            "cleared_step_scores": cleared_step_scores,
            "env_normalized": env_normalized,
            "env_sync": env_sync_status(),
        }
        report = write_report("destroy_legacy_derived_data", payload)
        print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
