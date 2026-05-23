from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune graph hyperparameters for one course/model stack using local TPE simulation.")
    parser.add_argument("--course-id", required=True, help="Course id to tune.")
    parser.add_argument("--llm-model-name", help="Graph extraction/chat model name, e.g. qwen-plus. Defaults to runtime settings.")
    parser.add_argument("--model-name", help="Deprecated alias for --llm-model-name.")
    parser.add_argument("--embedding-model-name", help="Embedding model name. Defaults to runtime settings.")
    parser.add_argument("--embedding-text-version", help="Embedding text version. Defaults to CURRENT_EMBEDDING_TEXT_VERSION.")
    parser.add_argument("--trials", type=int, default=30, help="Number of TPE trials.")
    parser.add_argument("--dry-run", action="store_true", help="Run tuning without writing course_model_hyperparameters.")
    return parser.parse_args()


async def main() -> None:
    from app.core.config import get_settings
    from app.db import SessionLocal, ensure_schema
    from app.models import Chunk
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
    from app.services.concept_graph import (
        GraphExtractionPlan,
        choose_graph_probe_chunks,
        create_graph_extraction_run_from_plan,
        execute_graph_extraction_run,
    )
    from app.services.hpo_engine import HyperparameterTuningService
    from sqlalchemy import select

    args = parse_args()
    settings = get_settings()
    llm_model_name = args.llm_model_name or args.model_name or settings.chat_model
    embedding_model_name = args.embedding_model_name or settings.embedding_model
    embedding_text_version = args.embedding_text_version or CURRENT_EMBEDDING_TEXT_VERSION
    ensure_schema()
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        chunks = db.scalars(
            select(Chunk)
            .where(Chunk.course_id == args.course_id, Chunk.is_active.is_(True))
            .order_by(Chunk.created_at.asc())
        ).all()
        probe_chunks = choose_graph_probe_chunks(chunks, limit=5)
        payloads = await HyperparameterTuningService.pre_extract_probes(db, probe_chunks)
        missing_probe_chunks = [chunk for chunk in probe_chunks if str(chunk.id) not in payloads]
        if missing_probe_chunks:
            selected_reasons = {
                str(chunk.id): {
                    "priority": 1.0,
                    "source": "hpo_probe",
                    "reason": "missing_reusable_graph_extraction_payload",
                }
                for chunk in missing_probe_chunks
            }
            plan = GraphExtractionPlan(
                selected_chunk_ids=[str(chunk.id) for chunk in missing_probe_chunks],
                selected_reasons=selected_reasons,
                skipped_reasons={},
                coverage={"selected_chunks": len(missing_probe_chunks), "source": "hpo_probe"},
                budget={"strategy": "hpo_probe", "max_model_calls_per_run": len(missing_probe_chunks)},
                stop_reason="hpo_probe_payload_backfill",
            )
            run = create_graph_extraction_run_from_plan(
                db,
                course_id=args.course_id,
                batch_id=None,
                chunks=missing_probe_chunks,
                plan=plan,
                profile_version="hpo_probe",
            )
            run.strategy = "hpo_probe"
            db.commit()
            fresh_payloads, errors, _stats = await execute_graph_extraction_run(
                db,
                run=run,
                chunks=missing_probe_chunks,
                batch_id=None,
            )
            payloads.update(fresh_payloads)
            missing_errors = {chunk_id: error for chunk_id, error in errors.items() if chunk_id not in payloads}
            if missing_errors:
                raise RuntimeError(f"HPO probe extraction failed: {next(iter(missing_errors.values()))}")
        result = await HyperparameterTuningService.tune_corpus_parameters(
            db,
            args.course_id,
            llm_model_name,
            probe_chunks=probe_chunks,
            payloads=payloads,
            baseline_context_chunks=chunks,
            embedding_model_name=embedding_model_name,
            embedding_text_version=embedding_text_version,
            n_trials=args.trials,
            dry_run=args.dry_run,
        )
    report_path = output_dir / f"graph_hpo_{args.course_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
