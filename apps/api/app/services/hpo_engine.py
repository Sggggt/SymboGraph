from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
import optuna

from app.models import (
    Chunk,
    CourseModelHyperparameter,
    GraphExtractionChunkTask,
    GraphHpoJudgeSample,
    GraphHpoObjectiveModel,
)
from app.services.graph_algorithms import GraphHyperparameters, build_course_model_hyperparameter_key
from app.services.graph_hpo_objective import (
    HPO_JUDGE_PROMPT_VERSION,
    HPO_OBJECTIVE_SCHEMA_VERSION,
    CandidateEvaluation,
    extract_candidate_graph_features,
    feature_summary,
    judge_candidate_pair,
    learn_surrogate_objective,
    payload_fingerprint,
    score_with_learned_objective,
    select_pair_indices,
)


HPO_SCHEMA_VERSION = "graph_hpo_judge_learned_v1"


def _normalise_best_params(params: dict[str, float]) -> GraphHyperparameters:
    return GraphHyperparameters(
        min_relation_confidence=params["min_relation_confidence"],
        min_accepted_relation_weight=params["min_accepted_relation_weight"],
        dijkstra_semantic_threshold=params["dijkstra_semantic_threshold"],
        w_degree=params["w_degree"],
        w_weighted_degree=params["w_weighted_degree"],
        w_pagerank=params["w_pagerank"],
        w_betweenness=params["w_betweenness"],
        w_closeness=params["w_closeness"],
        w_centrality=params["w_centrality"],
        w_llm_importance=params["w_llm_importance"],
        w_evidence=params["w_evidence"],
        source="hpo_candidate",
    ).normalized()


def _emit_hpo_log(batch_id: str | None, event: str, message: str, **payload: Any) -> None:
    if not batch_id:
        return
    from app.services.ingestion_logs import emit_ingestion_log

    emit_ingestion_log(batch_id, event, message, **payload)


class HyperparameterTuningService:
    @staticmethod
    def build_mock_graph_with_params(payloads: dict[str, dict[str, Any]], params: GraphHyperparameters):
        from app.services.graph_hpo_objective import build_candidate_graph

        return build_candidate_graph(payloads, params).graph

    @staticmethod
    def _cached_probe_payloads(db: Session, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        tasks = db.scalars(
            select(GraphExtractionChunkTask)
            .where(GraphExtractionChunkTask.chunk_id.in_(chunk_ids), GraphExtractionChunkTask.status == "completed")
            .order_by(GraphExtractionChunkTask.updated_at.desc())
        ).all()
        payloads: dict[str, dict[str, Any]] = {}
        for task in tasks:
            if task.chunk_id in payloads or not isinstance(task.payload_json, dict):
                continue
            payloads[task.chunk_id] = task.payload_json
        return payloads

    @classmethod
    async def pre_extract_probes(
        cls,
        db: Session,
        probe_chunks: list[Chunk],
        *,
        batch_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        chunk_ids = [str(chunk.id) for chunk in probe_chunks]
        return cls._cached_probe_payloads(db, chunk_ids)

    @staticmethod
    def _suggest_params(trial: Any) -> dict[str, float]:
        return {
            "min_relation_confidence": trial.suggest_float("min_relation_confidence", 0.50, 0.85),
            "min_accepted_relation_weight": trial.suggest_float("min_accepted_relation_weight", 0.45, 0.78),
            "dijkstra_semantic_threshold": trial.suggest_float("dijkstra_semantic_threshold", 0.65, 0.88),
            "w_degree": trial.suggest_float("w_degree", 0.05, 0.50),
            "w_weighted_degree": trial.suggest_float("w_weighted_degree", 0.05, 0.50),
            "w_pagerank": trial.suggest_float("w_pagerank", 0.05, 0.50),
            "w_betweenness": trial.suggest_float("w_betweenness", 0.05, 0.50),
            "w_closeness": trial.suggest_float("w_closeness", 0.05, 0.30),
            "w_centrality": trial.suggest_float("w_centrality", 0.10, 0.80),
            "w_llm_importance": trial.suggest_float("w_llm_importance", 0.05, 0.60),
            "w_evidence": trial.suggest_float("w_evidence", 0.05, 0.60),
        }

    @classmethod
    def _seed_candidate_params(cls, max_candidates: int) -> list[GraphHyperparameters]:
        seeds = [
            {
                "min_relation_confidence": 0.62,
                "min_accepted_relation_weight": 0.56,
                "dijkstra_semantic_threshold": 0.74,
                "w_degree": 0.25,
                "w_weighted_degree": 0.25,
                "w_pagerank": 0.20,
                "w_betweenness": 0.20,
                "w_closeness": 0.10,
                "w_centrality": 0.50,
                "w_llm_importance": 0.25,
                "w_evidence": 0.25,
            },
            {
                "min_relation_confidence": 0.70,
                "min_accepted_relation_weight": 0.64,
                "dijkstra_semantic_threshold": 0.78,
                "w_degree": 0.20,
                "w_weighted_degree": 0.35,
                "w_pagerank": 0.15,
                "w_betweenness": 0.20,
                "w_closeness": 0.10,
                "w_centrality": 0.40,
                "w_llm_importance": 0.25,
                "w_evidence": 0.35,
            },
            {
                "min_relation_confidence": 0.78,
                "min_accepted_relation_weight": 0.70,
                "dijkstra_semantic_threshold": 0.84,
                "w_degree": 0.15,
                "w_weighted_degree": 0.40,
                "w_pagerank": 0.15,
                "w_betweenness": 0.20,
                "w_closeness": 0.10,
                "w_centrality": 0.35,
                "w_llm_importance": 0.20,
                "w_evidence": 0.45,
            },
        ]
        params = [_normalise_best_params(seed) for seed in seeds]
        for index in range(max(0, max_candidates - len(params))):
            offset = index / max(max_candidates - len(params), 1)
            params.append(
                GraphHyperparameters(
                    min_relation_confidence=0.54 + 0.30 * offset,
                    min_accepted_relation_weight=0.50 + 0.25 * ((index * 3) % 7) / 6,
                    dijkstra_semantic_threshold=0.68 + 0.18 * ((index * 5) % 9) / 8,
                    w_degree=0.08 + 0.30 * ((index + 1) % 5) / 4,
                    w_weighted_degree=0.10 + 0.35 * ((index + 2) % 5) / 4,
                    w_pagerank=0.08 + 0.30 * ((index + 3) % 5) / 4,
                    w_betweenness=0.08 + 0.30 * ((index + 4) % 5) / 4,
                    w_closeness=0.05 + 0.20 * ((index + 2) % 4) / 3,
                    w_centrality=0.20 + 0.50 * ((index + 1) % 6) / 5,
                    w_llm_importance=0.10 + 0.45 * ((index + 3) % 6) / 5,
                    w_evidence=0.10 + 0.45 * ((index + 5) % 6) / 5,
                    source="hpo_seed",
                ).normalized()
            )
        unique: dict[str, GraphHyperparameters] = {}
        for item in params:
            unique[str(item.audit())] = item
        return list(unique.values())[:max_candidates]

    @classmethod
    async def _judge_pair(
        cls,
        evaluation_a: CandidateEvaluation,
        evaluation_b: CandidateEvaluation,
        chunks: list[Chunk],
        max_tokens: int,
    ) -> dict[str, Any]:
        return await judge_candidate_pair(evaluation_a, evaluation_b, chunks=chunks, max_tokens=max_tokens)

    @classmethod
    async def _build_judge_objective(
        cls,
        *,
        db: Session,
        course_id: str,
        model_key: str,
        llm_model_name: str,
        embedding_model_name: str,
        embedding_text_version: str,
        payloads: dict[str, dict[str, Any]],
        chunks: list[Chunk],
        batch_id: str | None,
        dry_run: bool,
    ) -> tuple[dict[str, Any], dict[str, CandidateEvaluation], list[dict[str, Any]], str | None]:
        from app.core.config import get_settings

        settings = get_settings()
        if settings.hpo_objective_mode != "judge_learned":
            raise RuntimeError(f"Unsupported HPO objective mode: {settings.hpo_objective_mode}")
        fingerprint = payload_fingerprint(payloads)
        candidates = cls._seed_candidate_params(settings.hpo_judge_max_candidates)
        _emit_hpo_log(
            batch_id,
            "hpo_objective_features_started",
            "Extracting graph HPO candidate features",
            stage="features",
            objective_mode="judge_learned",
            candidate_count=len(candidates),
        )
        evaluations: dict[str, CandidateEvaluation] = {}
        for params in candidates:
            evaluation = extract_candidate_graph_features(payloads, params, chunks=chunks)
            evaluations[evaluation.candidate_id] = evaluation
        hard_failures = {
            evaluation.candidate_id: evaluation.hard_failures
            for evaluation in evaluations.values()
            if evaluation.hard_failures
        }
        _emit_hpo_log(
            batch_id,
            "hpo_objective_features_completed",
            "Graph HPO candidate features extracted",
            stage="features",
            objective_mode="judge_learned",
            candidate_count=len(evaluations),
            feature_summary=feature_summary(next(iter(evaluations.values())).features) if evaluations else {},
            hard_constraint_failures=hard_failures,
        )

        ordered = list(evaluations.values())
        pair_indices = select_pair_indices(len(ordered), settings.hpo_judge_max_pairs)
        labels: list[dict[str, Any]] = []
        _emit_hpo_log(
            batch_id,
            "hpo_judge_started",
            "Starting pairwise graph HPO judge",
            stage="judge",
            objective_mode="judge_learned",
            candidate_count=len(ordered),
            pair_count=len(pair_indices),
            effective_labels=0,
            min_labels=settings.hpo_judge_min_labels,
        )
        for pair_number, (left, right) in enumerate(pair_indices, start=1):
            evaluation_a = ordered[left]
            evaluation_b = ordered[right]
            try:
                judge_result = await cls._judge_pair(evaluation_a, evaluation_b, chunks, settings.hpo_judge_max_tokens_per_pair)
            except Exception as exc:
                judge_result = {
                    "winner": "tie",
                    "confidence": 0.0,
                    "reasons": [str(exc)],
                    "safety_flags": ["judge_error"],
                    "prompt_version": HPO_JUDGE_PROMPT_VERSION,
                    "raw_response": {"error": str(exc)},
                }
            label = {
                "candidate_a_id": evaluation_a.candidate_id,
                "candidate_b_id": evaluation_b.candidate_id,
                **judge_result,
            }
            labels.append(label)
            effective_labels = len([item for item in labels if item.get("winner") in {"A", "B"} and float(item.get("confidence", 0.0) or 0.0) > 0])
            _emit_hpo_log(
                batch_id,
                "hpo_judge_progress",
                f"HPO judge progress {pair_number}/{len(pair_indices)}",
                stage="judge",
                objective_mode="judge_learned",
                candidate_count=len(ordered),
                pair_count=len(pair_indices),
                processed_pairs=pair_number,
                effective_labels=effective_labels,
                min_labels=settings.hpo_judge_min_labels,
            )
            if not dry_run:
                db.add(
                    GraphHpoJudgeSample(
                        course_id=course_id,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        embedding_text_version=embedding_text_version,
                        model_key=model_key,
                        payload_fingerprint=fingerprint,
                        prompt_version=str(judge_result.get("prompt_version") or HPO_JUDGE_PROMPT_VERSION),
                        judge_model=get_settings().chat_model,
                        candidate_a_params=evaluation_a.params.audit(),
                        candidate_b_params=evaluation_b.params.audit(),
                        candidate_a_features=evaluation_a.features,
                        candidate_b_features=evaluation_b.features,
                        winner=str(judge_result.get("winner") or "tie"),
                        confidence=float(judge_result.get("confidence", 0.0) or 0.0),
                        reasons=list(judge_result.get("reasons") or []),
                        safety_flags=list(judge_result.get("safety_flags") or []),
                        raw_response=dict(judge_result.get("raw_response") or {}),
                    )
                )
                db.flush()

        effective_labels = len([item for item in labels if item.get("winner") in {"A", "B"} and float(item.get("confidence", 0.0) or 0.0) > 0])
        if effective_labels < settings.hpo_judge_min_labels:
            _emit_hpo_log(
                batch_id,
                "hpo_judge_failed",
                "Insufficient effective HPO judge labels",
                stage="judge",
                objective_mode="judge_learned",
                pair_count=len(pair_indices),
                effective_labels=effective_labels,
                min_labels=settings.hpo_judge_min_labels,
            )
            raise RuntimeError(f"Insufficient HPO judge labels: {effective_labels}/{settings.hpo_judge_min_labels}")
        _emit_hpo_log(
            batch_id,
            "hpo_judge_completed",
            "Pairwise graph HPO judge completed",
            stage="judge",
            objective_mode="judge_learned",
            pair_count=len(pair_indices),
            effective_labels=effective_labels,
            min_labels=settings.hpo_judge_min_labels,
        )
        _emit_hpo_log(
            batch_id,
            "hpo_objective_training_started",
            "Training judge-learned HPO objective",
            stage="objective_training",
            objective_mode="judge_learned",
            effective_labels=effective_labels,
            min_labels=settings.hpo_judge_min_labels,
        )
        try:
            objective_model = learn_surrogate_objective(labels, evaluations)
        except Exception as exc:
            _emit_hpo_log(
                batch_id,
                "hpo_objective_training_failed",
                "Judge-learned HPO objective training failed",
                stage="objective_training",
                objective_mode="judge_learned",
                effective_labels=effective_labels,
                min_labels=settings.hpo_judge_min_labels,
                error=str(exc),
            )
            raise
        objective_model_id: str | None = None
        if not dry_run:
            row = GraphHpoObjectiveModel(
                course_id=course_id,
                llm_model_name=llm_model_name,
                embedding_model_name=embedding_model_name,
                embedding_text_version=embedding_text_version,
                model_key=model_key,
                objective_version=HPO_OBJECTIVE_SCHEMA_VERSION,
                payload_fingerprint=fingerprint,
                feature_names=list(objective_model["feature_names"]),
                weights_json=dict(objective_model["weights"]),
                normalization_json=dict(objective_model["normalization"]),
                label_count=int(objective_model["label_count"]),
                training_audit=dict(objective_model["training_audit"]),
                status="completed",
            )
            db.add(row)
            db.flush()
            objective_model_id = row.id
        _emit_hpo_log(
            batch_id,
            "hpo_objective_training_completed",
            "Judge-learned HPO objective trained",
            stage="objective_training",
            objective_mode="judge_learned",
            effective_labels=effective_labels,
            objective_model_id=objective_model_id,
            feature_summary={"top_weights": dict(list(objective_model["weights"].items())[:8])},
        )
        return objective_model, evaluations, labels, objective_model_id

    @classmethod
    def optimize_payloads(
        cls,
        payloads: dict[str, dict[str, Any]],
        objective_model: dict[str, Any],
        *,
        chunks: list[Chunk],
        n_trials: int = 30,
        batch_id: str | None = None,
    ) -> tuple[GraphHyperparameters, dict[str, Any]]:
        trials_seen = 0

        def objective(trial: Any) -> float:
            nonlocal trials_seen
            params = _normalise_best_params(cls._suggest_params(trial))
            evaluation = extract_candidate_graph_features(payloads, params, chunks=chunks)
            trials_seen += 1
            if batch_id and (trials_seen == 1 or trials_seen == n_trials or trials_seen % 5 == 0):
                _emit_hpo_log(
                    batch_id,
                    "hpo_tpe_progress",
                    f"HPO TPE progress {trials_seen}/{n_trials}",
                    stage="tpe",
                    objective_mode="judge_learned",
                    trial_count=trials_seen,
                    feature_summary=feature_summary(evaluation.features),
                )
            if evaluation.hard_failures:
                return -100.0 - len(evaluation.hard_failures)
            return score_with_learned_objective(evaluation.features, objective_model)

        _emit_hpo_log(
            batch_id,
            "hpo_tpe_started",
            "Starting TPE optimization with judge-learned objective",
            stage="tpe",
            objective_mode="judge_learned",
            trial_count=n_trials,
        )
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        params = _normalise_best_params(study.best_params)
        best_evaluation = extract_candidate_graph_features(payloads, params, chunks=chunks)
        _emit_hpo_log(
            batch_id,
            "hpo_tpe_completed",
            "TPE optimization completed",
            stage="tpe",
            objective_mode="judge_learned",
            trial_count=len(study.trials),
            best_value=float(study.best_value),
            feature_summary=feature_summary(best_evaluation.features),
        )
        return params, {
            "best_value": float(study.best_value),
            "trials_count": len(study.trials),
            "history": [trial.value for trial in study.trials],
            "optimizer": "optuna_tpe",
            "objective_mode": "judge_learned",
            "best_feature_summary": feature_summary(best_evaluation.features),
            "best_hard_failures": best_evaluation.hard_failures,
        }

    @classmethod
    async def tune_corpus_parameters(
        cls,
        db: Session,
        course_id: str,
        llm_model_name: str,
        *,
        probe_chunks: list[Chunk] | None = None,
        payloads: dict[str, dict[str, Any]] | None = None,
        baseline_context_chunks: list[Chunk] | None = None,
        embedding_model_name: str | None = None,
        embedding_text_version: str | None = None,
        n_trials: int = 30,
        dry_run: bool = False,
        commit: bool = True,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        from app.core.config import get_settings
        from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
        from app.services.concept_graph import choose_graph_probe_chunks

        settings = get_settings()
        selected_embedding = embedding_model_name or settings.embedding_model
        selected_text_version = embedding_text_version or CURRENT_EMBEDDING_TEXT_VERSION
        model_key = build_course_model_hyperparameter_key(llm_model_name, selected_embedding, selected_text_version)
        chunks = list(
            probe_chunks
            or db.scalars(
                select(Chunk)
                .where(Chunk.course_id == course_id, Chunk.is_active.is_(True))
                .order_by(Chunk.created_at.asc())
            ).all()
        )
        selected_probe_chunks = choose_graph_probe_chunks(chunks, limit=5)
        if not selected_probe_chunks:
            raise RuntimeError("No active chunks available for HPO probes")
        if payloads is None:
            payloads = await cls.pre_extract_probes(db, selected_probe_chunks)
        else:
            selected_ids = {str(chunk.id) for chunk in selected_probe_chunks}
            payloads = {
                str(chunk_id): payload
                for chunk_id, payload in payloads.items()
                if str(chunk_id) in selected_ids and isinstance(payload, dict)
            }
        if not payloads:
            raise RuntimeError("No graph extraction payloads available for HPO")
        context_chunks = list(baseline_context_chunks or chunks)
        objective_model, candidate_evaluations, labels, objective_model_id = await cls._build_judge_objective(
            db=db,
            course_id=course_id,
            model_key=model_key,
            llm_model_name=llm_model_name,
            embedding_model_name=selected_embedding,
            embedding_text_version=selected_text_version,
            payloads=payloads,
            chunks=context_chunks,
            batch_id=batch_id,
            dry_run=dry_run,
        )
        params, history = cls.optimize_payloads(
            payloads,
            objective_model,
            chunks=context_chunks,
            n_trials=n_trials,
            batch_id=batch_id,
        )
        if not dry_run:
            record = db.get(
                CourseModelHyperparameter,
                {
                    "course_id": course_id,
                    "llm_model_name": llm_model_name,
                    "embedding_model_name": selected_embedding,
                    "embedding_text_version": selected_text_version,
                },
            )
            if record is None:
                record = db.scalar(
                    select(CourseModelHyperparameter).where(
                        CourseModelHyperparameter.course_id == course_id,
                        CourseModelHyperparameter.model_name == model_key,
                    )
                )
            if record is None:
                record = CourseModelHyperparameter(
                    course_id=course_id,
                    llm_model_name=llm_model_name,
                    embedding_model_name=selected_embedding,
                    embedding_text_version=selected_text_version,
                )
                db.add(record)
            record.llm_model_name = llm_model_name
            record.embedding_model_name = selected_embedding
            record.embedding_text_version = selected_text_version
            record.model_name = model_key
            for key, value in asdict(params).items():
                if key == "source":
                    continue
                setattr(record, key, round(float(value), 4))
            record.graph_version = "active"
            record.hpo_status = "completed"
            record.last_optimized_at = datetime.utcnow()
            effective_labels = len([item for item in labels if item.get("winner") in {"A", "B"}])
            hard_failures = {
                candidate_id: evaluation.hard_failures
                for candidate_id, evaluation in candidate_evaluations.items()
                if evaluation.hard_failures
            }
            record.optuna_history = {
                **history,
                "schema_version": HPO_SCHEMA_VERSION,
                "objective_schema_version": HPO_OBJECTIVE_SCHEMA_VERSION,
                "objective_mode": "judge_learned",
                "objective_model_id": objective_model_id,
                "judge_pairs_requested": len(labels),
                "effective_labels": effective_labels,
                "feature_weights": objective_model.get("weights", {}),
                "hard_constraint_failures": hard_failures,
                "probe_chunk_ids": [str(chunk.id) for chunk in selected_probe_chunks],
            }
            if commit:
                db.commit()
            else:
                db.flush()
        return {
            "course_id": course_id,
            "llm_model_name": llm_model_name,
            "embedding_model_name": selected_embedding,
            "embedding_text_version": selected_text_version,
            "model_key": model_key,
            "dry_run": dry_run,
            "hyperparameters": params.audit(),
            "history": history,
            "objective": {
                "mode": "judge_learned",
                "schema_version": HPO_OBJECTIVE_SCHEMA_VERSION,
                "objective_model_id": objective_model_id,
                "judge_pairs_requested": len(labels),
                "effective_labels": len([item for item in labels if item.get("winner") in {"A", "B"}]),
                "feature_weights": objective_model.get("weights", {}),
                "training_audit": objective_model.get("training_audit", {}),
            },
            "probe_chunk_ids": [str(chunk.id) for chunk in selected_probe_chunks],
        }
