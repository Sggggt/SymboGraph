from __future__ import annotations

import pytest


def _set_fast_hpo_env(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("HPO_JUDGE_MAX_CANDIDATES", "4")
    monkeypatch.setenv("HPO_JUDGE_MAX_PAIRS", "3")
    monkeypatch.setenv("HPO_JUDGE_MIN_LABELS", "2")
    monkeypatch.setenv("HPO_JUDGE_MAX_TOKENS_PER_PAIR", "2000")
    get_settings.cache_clear()


def _payloads(chunks):
    return {
        chunks[0].id: {
            "concepts": [
                {"name": "Degree Centrality", "aliases": [], "confidence": 0.9},
                {"name": "Graph Node", "aliases": [], "confidence": 0.8},
            ],
            "relations": [
                {"source": "Degree Centrality", "target": "Graph Node", "relation_type": "defined_by", "confidence": 0.82, "weight": 0.82}
            ],
        },
        chunks[1].id: {
            "concepts": [
                {"name": "Betweenness Centrality", "aliases": [], "confidence": 0.9},
                {"name": "Shortest Path", "aliases": [], "confidence": 0.85},
            ],
            "relations": [
                {"source": "Betweenness Centrality", "target": "Shortest Path", "relation_type": "used_for", "confidence": 0.68, "weight": 0.68}
            ],
        },
    }


@pytest.mark.asyncio
async def test_hpo_tune_writes_judge_learned_hyperparameters(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.models import CourseModelHyperparameter, GraphHpoJudgeSample, GraphHpoObjectiveModel
    from app.services.hpo_engine import HyperparameterTuningService

    _set_fast_hpo_env(monkeypatch)
    _document, chunks = indexed_chunks
    payloads = _payloads(chunks)

    async def fake_judge(cls, evaluation_a, evaluation_b, chunks, max_tokens):
        score_a = evaluation_a.summary["edge_count"] - evaluation_a.features["isolated_node_rate"]
        score_b = evaluation_b.summary["edge_count"] - evaluation_b.features["isolated_node_rate"]
        return {
            "winner": "A" if score_a >= score_b else "B",
            "confidence": 0.9,
            "reasons": ["higher evidence-grounded structure score"],
            "safety_flags": [],
            "prompt_version": "unit_judge",
            "raw_response": {"unit": True},
        }

    monkeypatch.setattr(HyperparameterTuningService, "_judge_pair", classmethod(fake_judge))

    result = await HyperparameterTuningService.tune_corpus_parameters(
        db_session,
        sample_course.id,
        "unit-llm",
        probe_chunks=chunks,
        payloads=payloads,
        baseline_context_chunks=chunks,
        embedding_model_name="unit-embedding",
        embedding_text_version="unit-text-v1",
        n_trials=8,
    )

    record = db_session.get(
        CourseModelHyperparameter,
        {
            "course_id": sample_course.id,
            "llm_model_name": "unit-llm",
            "embedding_model_name": "unit-embedding",
            "embedding_text_version": "unit-text-v1",
        },
    )
    assert record is not None
    assert record.model_name == "llm:unit-llm|embedding:unit-embedding|text:unit-text-v1"
    assert result["model_key"] == record.model_name
    assert record.hpo_status == "completed"
    assert record.optuna_history["objective_mode"] == "judge_learned"
    assert record.optuna_history["trials_count"] == 8
    assert "golden_graph_source" not in record.optuna_history
    assert "golden_graph" not in result
    assert result["objective"]["effective_labels"] >= 2
    assert db_session.query(GraphHpoJudgeSample).filter_by(course_id=sample_course.id).count() >= 2
    assert db_session.query(GraphHpoObjectiveModel).filter_by(course_id=sample_course.id).count() == 1
    assert abs(record.w_degree + record.w_weighted_degree + record.w_pagerank + record.w_betweenness + record.w_closeness - 1.0) < 0.001


@pytest.mark.asyncio
async def test_hpo_pre_extract_probes_only_reads_cached_payloads(db_session, sample_course, indexed_chunks):
    from app.models import GraphExtractionChunkTask, GraphExtractionRun
    from app.services.concept_graph import _chunk_hash
    from app.services.hpo_engine import HyperparameterTuningService

    _document, chunks = indexed_chunks
    cached_payload = {
        "concepts": [{"name": "Cached Node", "confidence": 0.92}],
        "relations": [],
    }
    run = GraphExtractionRun(course_id=sample_course.id, strategy="hpo_probe", status="completed")
    db_session.add(run)
    db_session.flush()
    db_session.add(
        GraphExtractionChunkTask(
            run_id=run.id,
            course_id=sample_course.id,
            chunk_id=chunks[0].id,
            chunk_hash=_chunk_hash(chunks[0]),
            status="completed",
            payload_json=cached_payload,
        )
    )
    db_session.commit()

    payloads = await HyperparameterTuningService.pre_extract_probes(db_session, chunks)

    assert payloads == {chunks[0].id: cached_payload}
    assert chunks[1].id not in payloads


@pytest.mark.asyncio
async def test_hpo_missing_payloads_fails_without_completed_record(db_session, sample_course):
    from app.models import CourseModelHyperparameter
    from app.services.hpo_engine import HyperparameterTuningService

    with pytest.raises(RuntimeError, match="No active chunks available|No graph extraction payloads"):
        await HyperparameterTuningService.tune_corpus_parameters(
            db_session,
            sample_course.id,
            "unit-llm-empty",
            embedding_model_name="unit-embedding",
            embedding_text_version="unit-text-v1",
            n_trials=2,
        )

    record = db_session.get(
        CourseModelHyperparameter,
        {
            "course_id": sample_course.id,
            "llm_model_name": "unit-llm-empty",
            "embedding_model_name": "unit-embedding",
            "embedding_text_version": "unit-text-v1",
        },
    )
    assert record is None or record.hpo_status != "completed"


def test_candidate_feature_extraction_covers_evidence_and_structure(indexed_chunks):
    from app.services.graph_algorithms import GraphHyperparameters
    from app.services.graph_hpo_objective import FEATURE_NAMES, extract_candidate_graph_features

    _document, chunks = indexed_chunks
    evaluation = extract_candidate_graph_features(
        _payloads(chunks),
        GraphHyperparameters(min_relation_confidence=0.7, min_accepted_relation_weight=0.7),
        chunks=chunks,
    )

    for key in [
        "edge_evidence_coverage",
        "node_evidence_coverage",
        "isolated_node_rate",
        "giant_component_ratio",
        "modularity",
        "average_clustering",
        "node_jaccard_bootstrap",
        "edge_jaccard_bootstrap",
    ]:
        assert key in evaluation.features
    assert set(FEATURE_NAMES).issuperset(evaluation.features)
    assert evaluation.features["edge_evidence_coverage"] == 1.0
    assert evaluation.summary["node_count"] >= 4


def test_surrogate_objective_learns_from_pairwise_labels(indexed_chunks):
    from app.services.graph_algorithms import GraphHyperparameters
    from app.services.graph_hpo_objective import extract_candidate_graph_features, learn_surrogate_objective, score_with_learned_objective

    _document, chunks = indexed_chunks
    strong = extract_candidate_graph_features(
        _payloads(chunks),
        GraphHyperparameters(min_relation_confidence=0.5, min_accepted_relation_weight=0.5),
        chunks=chunks,
    )
    strict = extract_candidate_graph_features(
        _payloads(chunks),
        GraphHyperparameters(min_relation_confidence=0.9, min_accepted_relation_weight=0.9),
        chunks=chunks,
    )
    evaluations = {strong.candidate_id: strong, strict.candidate_id: strict}
    model = learn_surrogate_objective(
        [{"candidate_a_id": strong.candidate_id, "candidate_b_id": strict.candidate_id, "winner": "A", "confidence": 0.9}],
        evaluations,
    )

    assert model["label_count"] == 1
    assert score_with_learned_objective(strong.features, model) > score_with_learned_objective(strict.features, model)


@pytest.mark.asyncio
async def test_hpo_insufficient_judge_labels_does_not_write_completed(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.models import CourseModelHyperparameter
    from app.services.hpo_engine import HyperparameterTuningService

    _set_fast_hpo_env(monkeypatch)
    _document, chunks = indexed_chunks

    async def tie_judge(cls, evaluation_a, evaluation_b, chunks, max_tokens):
        return {"winner": "tie", "confidence": 0.0, "reasons": ["ambiguous"], "safety_flags": [], "raw_response": {}}

    monkeypatch.setattr(HyperparameterTuningService, "_judge_pair", classmethod(tie_judge))

    with pytest.raises(RuntimeError, match="Insufficient HPO judge labels"):
        await HyperparameterTuningService.tune_corpus_parameters(
            db_session,
            sample_course.id,
            "unit-llm-tie",
            probe_chunks=chunks,
            payloads=_payloads(chunks),
            baseline_context_chunks=chunks,
            embedding_model_name="unit-embedding",
            embedding_text_version="unit-text-v1",
            n_trials=2,
        )

    record = db_session.get(
        CourseModelHyperparameter,
        {
            "course_id": sample_course.id,
            "llm_model_name": "unit-llm-tie",
            "embedding_model_name": "unit-embedding",
            "embedding_text_version": "unit-text-v1",
        },
    )
    assert record is None or record.hpo_status != "completed"


def test_hpo_build_mock_graph_filters_by_candidate_thresholds():
    from app.services.graph_algorithms import GraphHyperparameters
    from app.services.hpo_engine import HyperparameterTuningService

    payloads = {
        "chunk-1": {
            "concepts": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "relations": [
                {"source": "A", "target": "B", "confidence": 0.9, "weight": 0.9},
                {"source": "B", "target": "C", "confidence": 0.51, "weight": 0.51},
            ],
        }
    }
    params = GraphHyperparameters(min_relation_confidence=0.7, min_accepted_relation_weight=0.7)

    graph = HyperparameterTuningService.build_mock_graph_with_params(payloads, params)

    assert graph.has_edge("A", "B")
    assert not graph.has_edge("B", "C")


def test_hpo_engine_requires_optuna_without_fallback(monkeypatch):
    import importlib
    import sys

    class BlockOptunaImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "optuna" or fullname.startswith("optuna."):
                raise ModuleNotFoundError("blocked optuna for test")
            return None

    saved_hpo = sys.modules.pop("app.services.hpo_engine", None)
    saved_optuna = sys.modules.pop("optuna", None)
    blocker = BlockOptunaImport()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ModuleNotFoundError, match="blocked optuna"):
            importlib.import_module("app.services.hpo_engine")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop("app.services.hpo_engine", None)
        if saved_hpo is not None:
            sys.modules["app.services.hpo_engine"] = saved_hpo
        if saved_optuna is not None:
            sys.modules["optuna"] = saved_optuna
