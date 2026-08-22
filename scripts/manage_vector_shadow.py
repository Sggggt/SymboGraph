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

MAX_STATUS_BUILD_SCAN = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operate the PostgreSQL-authoritative vector shadow lifecycle."
    )
    parser.add_argument(
        "action",
        choices=(
            "stage",
            "build",
            "evaluate",
            "promote",
            "rollback",
            "abandon",
            "reconcile-cache",
            "status",
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-id")
    parser.add_argument("--build-id")
    parser.add_argument("--knowledge-base-id", action="append", default=[])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--embedding-text-version")
    parser.add_argument("--chunk-schema-version")
    parser.add_argument(
        "--concept-provider-request-budget",
        type=int,
        default=4,
        help=(
            "Shared Mid+Coarse provider request hard limit for build; "
            "semantic-reuse hits consume zero requests (2..128)."
        ),
    )
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--reason")
    parser.add_argument(
        "--disposition",
        choices=("rejected", "superseded"),
        default="rejected",
    )
    return parser.parse_args()


def _require(value, message: str):
    if value is None or value == "" or value == []:
        raise SystemExit(message)
    return value


def _candidate_status(db, candidate_id: str) -> dict:
    from sqlalchemy import select

    from app.models import RuntimeSettingsCandidate, VectorShadowBuild

    candidate = db.get(RuntimeSettingsCandidate, candidate_id)
    if candidate is None:
        raise SystemExit(f"Runtime settings candidate not found: {candidate_id}")
    builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(VectorShadowBuild.runtime_settings_candidate_id == candidate.id)
            .order_by(VectorShadowBuild.knowledge_base_id.asc())
            .limit(MAX_STATUS_BUILD_SCAN + 1)
        ).all()
    )
    if len(builds) > MAX_STATUS_BUILD_SCAN:
        raise RuntimeError("Vector shadow status refused an unbounded build scan")
    return {
        "candidate": {
            "id": candidate.id,
            "candidate_hash": candidate.candidate_hash,
            "status": candidate.status,
            "changed_keys": list(candidate.changed_keys_json or []),
            "blocking_reasons": list(candidate.blocking_reasons_json or []),
        },
        "builds": [
            {
                "id": build.id,
                "knowledge_base_id": build.knowledge_base_id,
                "status": build.status,
                "collection_name": build.collection_name,
                "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
                "expected_point_count": build.expected_point_count,
                "ready_point_count": build.ready_point_count,
                "qdrant_proof_hash": build.qdrant_proof_hash,
                "evaluation_result_hash": build.evaluation_result_hash,
            }
            for build in builds
        ],
    }


async def _main_async(args: argparse.Namespace) -> dict:
    from app.models import ContextGraphState, VectorShadowBuild
    from app.services.chunking import CHUNK_SCHEMA_VERSION, CURRENT_EMBEDDING_TEXT_VERSION
    from app.services.vector_shadow_lifecycle import (
        VectorShadowEvaluation,
        abandon_vector_shadow_candidate,
        build_vector_shadow_artifacts,
        frozen_vector_schema,
        promote_vector_shadow_candidate,
        reconcile_vector_runtime_cache_invalidations,
        record_vector_shadow_build_attempt_failure,
        record_vector_shadow_evaluation,
        rollback_vector_shadow_candidate,
        stage_vector_runtime_candidate,
        vector_shadow_evaluation_input_hash,
        vector_shadow_promotion_preflight,
    )

    with session_scope() as db:
        if args.action == "stage":
            knowledge_base_ids = _require(
                args.knowledge_base_id,
                "stage requires at least one --knowledge-base-id",
            )
            embedding_model = _require(
                args.embedding_model,
                "stage requires --embedding-model",
            )
            embedding_dimension = _require(
                args.embedding_dimension,
                "stage requires --embedding-dimension",
            )
            schema = frozen_vector_schema(
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                embedding_text_version=(
                    args.embedding_text_version or CURRENT_EMBEDDING_TEXT_VERSION
                ),
                chunk_schema_version=(
                    args.chunk_schema_version or CHUNK_SCHEMA_VERSION
                ),
            )
            result = {
                "target_knowledge_base_ids": sorted(set(knowledge_base_ids)),
                "vector_schema": schema.model_dump(mode="json"),
                "note": "dry-run validates only the requested identity; execute performs locked scope/live-build checks",
            }
            if args.execute:
                candidate, builds = stage_vector_runtime_candidate(
                    db,
                    knowledge_base_ids=knowledge_base_ids,
                    embedding_model=schema.embedding_model,
                    embedding_dimension=schema.embedding_dimension,
                    embedding_text_version=schema.embedding_text_version,
                    chunk_schema_version=schema.chunk_schema_version,
                    source="manage_vector_shadow.py",
                )
                db.commit()
                result = _candidate_status(db, candidate.id)
                result["build_ids"] = [build.id for build in builds]
            return result

        if args.action == "build":
            build_id = _require(args.build_id, "build requires --build-id")
            build = db.get(VectorShadowBuild, build_id)
            if build is None:
                raise SystemExit(f"Vector shadow build not found: {build_id}")
            result = {
                "build_id": build.id,
                "status": build.status,
                "knowledge_base_id": build.knowledge_base_id,
                "collection_name": build.collection_name,
                "would_initialize_model_and_qdrant_io": bool(args.execute),
            }
            if args.execute:
                prepare_runtime_for_model_io()
                try:
                    build = await build_vector_shadow_artifacts(
                        db,
                        build_id=build.id,
                        concept_provider_request_budget=(
                            args.concept_provider_request_budget
                        ),
                    )
                except Exception as exc:
                    # Remove partial PostgreSQL facts first, then durably record
                    # only the exception type. Provider response/body text must
                    # never be copied into lifecycle diagnostics.
                    db.rollback()
                    error_type = type(exc).__name__
                    try:
                        record_vector_shadow_build_attempt_failure(
                            db,
                            build_id,
                            error_type=error_type,
                            failure=exc,
                        )
                        db.commit()
                    except Exception:
                        db.rollback()
                    raise RuntimeError(
                        "Vector shadow build failed; "
                        f"error_type={error_type}; build_id={build_id}; "
                        "inspect the persisted safe failure audit and outbox reconciliation"
                    ) from None
                db.commit()
                result = _candidate_status(db, build.runtime_settings_candidate_id)
            return result

        if args.action == "evaluate":
            build_id = _require(args.build_id, "evaluate requires --build-id")
            evaluation_path = _require(
                args.evaluation_json,
                "evaluate requires --evaluation-json",
            )
            build = db.get(VectorShadowBuild, build_id)
            if build is None or not build.shadow_context_graph_state_id:
                raise SystemExit("evaluate requires one attested shadow-ready build")
            context_state = db.get(ContextGraphState, build.shadow_context_graph_state_id)
            if context_state is None:
                raise SystemExit("shadow context graph state is missing")
            raw = json.loads(evaluation_path.read_text(encoding="utf-8"))
            raw["evaluation_input_hash"] = vector_shadow_evaluation_input_hash(
                build,
                context_state,
            )
            evaluation = VectorShadowEvaluation.model_validate(raw)
            result = {
                "build_id": build.id,
                "evaluation_input_hash": evaluation.evaluation_input_hash,
                "hard_gates": evaluation.hard_gates,
                "evidence_hashes": evaluation.evidence_hashes,
            }
            if args.execute:
                record_vector_shadow_evaluation(
                    db,
                    build_id=build.id,
                    evaluation=evaluation,
                )
                db.commit()
                result = _candidate_status(db, build.runtime_settings_candidate_id)
            return result

        if args.action == "promote":
            candidate_id = _require(args.candidate_id, "promote requires --candidate-id")
            result = vector_shadow_promotion_preflight(db, candidate_id)
            if args.execute:
                result = promote_vector_shadow_candidate(db, candidate_id)
                db.commit()
                cache_result = reconcile_vector_runtime_cache_invalidations(
                    db,
                    candidate_id=candidate_id,
                    dry_run=False,
                )
                db.commit()
                result = {"promotion": result, "cache_invalidation": cache_result}
            return result

        if args.action == "rollback":
            candidate_id = _require(args.candidate_id, "rollback requires --candidate-id")
            result = _candidate_status(db, candidate_id)
            result["reason"] = args.reason
            if args.execute:
                reason = _require(args.reason, "rollback --execute requires --reason")
                rollback_vector_shadow_candidate(db, candidate_id, reason=reason)
                db.commit()
                cache_result = reconcile_vector_runtime_cache_invalidations(
                    db,
                    candidate_id=candidate_id,
                    dry_run=False,
                )
                db.commit()
                result = {
                    **_candidate_status(db, candidate_id),
                    "cache_invalidation": cache_result,
                }
            return result

        if args.action == "abandon":
            candidate_id = _require(args.candidate_id, "abandon requires --candidate-id")
            result = _candidate_status(db, candidate_id)
            result["would_disposition"] = args.disposition
            result["reason"] = args.reason
            if args.execute:
                reason = _require(args.reason, "abandon --execute requires --reason")
                abandon_vector_shadow_candidate(
                    db,
                    candidate_id,
                    disposition=args.disposition,
                    reason=reason,
                )
                db.commit()
                result = _candidate_status(db, candidate_id)
            return result

        if args.action == "reconcile-cache":
            result = reconcile_vector_runtime_cache_invalidations(
                db,
                candidate_id=args.candidate_id,
                dry_run=not args.execute,
            )
            if args.execute:
                db.commit()
            return result

        candidate_id = _require(args.candidate_id, "status requires --candidate-id")
        return _candidate_status(db, candidate_id)


def main() -> None:
    args = parse_args()
    if args.action == "status" and args.execute:
        raise SystemExit("status is read-only; omit --execute")
    result = asyncio.run(_main_async(args))
    payload = {
        "script": "manage_vector_shadow",
        "action": args.action,
        "mode": "execute" if args.execute else "dry_run",
        "execute": bool(args.execute),
        "result": result,
        "impact": (
            "writes only the selected PostgreSQL-authoritative lifecycle transition; "
            "build may write a candidate Qdrant collection through the durable outbox"
            if args.execute
            else "read-only lifecycle inspection/validation; no PostgreSQL/Qdrant/Redis writes"
        ),
    }
    report = write_report("manage_vector_shadow", payload)
    print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
