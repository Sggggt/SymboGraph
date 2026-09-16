"""Offline operating-point selection; never mutates production settings."""
from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata

from app.retrieval_control_contracts import (
    DecisionPanel, LexicalStrategy, PathEvaluationParameters, PathFeatureCandidate,
    TaskContract, control_hash,
)
from app.services.retrieval_path_features import compute_path_features


PROTOCOL = "retrieval_gate_development_calibration_v1"


@dataclass(frozen=True)
class ScoredSample:
    family: str
    split: str
    ready: bool
    coverage: float
    path: float
    integrity: bool = True
    scope_satisfied: bool = True


def require(value, code):
    if not value:
        raise ValueError(code)


def normalized_question(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def replay_dataset(dataset, *, excluded_questions):
    require(dataset.get("protocol_version") == PROTOCOL, "calibration_dataset_protocol_invalid")
    require(dataset.get("source_reviewed") is True and dataset.get("independence_reviewed") is True,
        "calibration_labels_require_source_and_independence_review")
    examples = dataset.get("examples", [])
    require(12 <= len(examples) <= 128, "calibration_sample_capacity_invalid")
    excluded = {normalized_question(question) for question in excluded_questions}
    seen_ids, seen_inputs, seen_questions, signatures, samples = set(), set(), {}, set(), []
    for example in examples:
        require(example.get("id") and example["id"] not in seen_ids, "calibration_duplicate_example")
        require(example.get("label_source") == "source_review" and type(example.get("ready")) is bool,
            "calibration_label_invalid")
        require(type(example.get("source_integrity")) is bool, "calibration_integrity_missing")
        inputs = example["feature_input"]
        require(set(inputs) == {"task", "strategy", "panels", "package", "parameters"},
            "calibration_feature_input_shape_invalid")
        task = TaskContract.model_validate(inputs["task"])
        require(not task.allow_partial, "calibration_requires_full_readiness_labels")
        question = normalized_question(task.question)
        group = (example["family"], example["split"])
        require(question not in excluded and seen_questions.get(question, group) == group,
            "calibration_question_leakage")
        strategy = LexicalStrategy.model_validate(inputs["strategy"])
        parameters = PathEvaluationParameters.model_validate(inputs["parameters"])
        features = compute_path_features(task=task, strategy=strategy,
            panels=tuple(DecisionPanel.model_validate(value) for value in inputs["panels"]),
            packaged_candidates=tuple(PathFeatureCandidate.model_validate(value) for value in inputs["package"]),
            parameters=parameters)
        require(features.input_hash == control_hash(inputs)
            and example.get("feature_hash") == control_hash(features.model_dump(mode="json")),
            "calibration_feature_replay_mismatch")
        require(features.input_hash not in seen_inputs, "calibration_duplicate_observation")
        signatures.add(control_hash({"parameters": parameters.model_dump(mode="json", exclude={"identity",
            'scope_inputs','packed_scope_intervals','scope_source_chunk_ids','scope_index_protocol'}),
            "vector_runtime": parameters.identity.vector_runtime_hash,
            "matcher": parameters.identity.match_protocol_hash}))
        samples.append(ScoredSample(family=example["family"], split=example["split"], ready=example["ready"],
            coverage=min(f.coverage.lower for f in features.facets),
            path=min(f.path_quality.lower for f in features.facets),
            integrity=example["source_integrity"] and features.invalid_packaged_source_count == 0,
            scope_satisfied=all(item.state == 'satisfied' for item in features.scope_statuses)))
        seen_ids.add(example["id"])
        seen_questions[question] = group
        seen_inputs.add(features.input_hash)
    require(len(signatures) == 1, "calibration_evaluation_protocol_mixed")
    return samples


def confusion(samples, coverage, path):
    counts = {"true_ready": 0, "false_ready": 0, "false_insufficient": 0, "true_insufficient": 0}
    for sample in samples:
        predicted = sample.integrity and sample.scope_satisfied and sample.coverage >= coverage and sample.path >= path
        key = ("true_ready" if sample.ready else "false_ready") if predicted else (
            "false_insufficient" if sample.ready else "true_insufficient")
        counts[key] += 1
    return counts


def passing(counts):
    positives = counts["true_ready"] + counts["false_insufficient"]
    return positives > 0 and counts["false_ready"] == 0 and counts["true_ready"] * 5 >= positives * 4


def calibrate(samples, *, baseline_coverage, baseline_path):
    require(12 <= len(samples) <= 128, "calibration_sample_capacity_invalid")
    require(all(math.isfinite(value) and 0 < value <= 1 for value in (baseline_coverage, baseline_path)),
        "calibration_baseline_invalid")
    require(all(sample.family and sample.split in {"train", "validation"}
        and type(sample.ready) is bool and type(sample.integrity) is bool and type(sample.scope_satisfied) is bool
        and all(math.isfinite(value) and 0 <= value <= 1 for value in (sample.coverage, sample.path))
        for sample in samples), "calibration_sample_invalid")
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    families = {split: {sample.family for sample in samples if sample.split == split}
        for split in ("train", "validation")}
    require(not families["train"] & families["validation"], "calibration_family_leakage")
    require(all(len(families[split]) >= 2 for split in families), "calibration_independent_families_insufficient")
    require(all(sum(sample.ready == label for sample in rows) >= minimum
        for rows, minimum in ((train, 4), (validation, 2)) for label in (False, True)),
        "calibration_label_counts_insufficient")

    positives = [sample for sample in train if sample.ready and sample.integrity and sample.scope_satisfied
        and sample.coverage >= baseline_coverage and sample.path >= baseline_path]
    negatives = [sample for sample in train if not sample.ready and sample.integrity and sample.scope_satisfied]
    best, evaluated = None, 0
    seen = set()
    for coverage in sorted({sample.coverage for sample in positives}):
        for path in sorted({sample.path for sample in positives}):
            evaluated += 1
            retained = [sample for sample in positives if sample.coverage >= coverage and sample.path >= path]
            if not retained:
                continue
            c, p = min(sample.coverage for sample in retained), min(sample.path for sample in retained)
            if (c,p) in seen:
                continue
            seen.add((c,p))
            margin = min(c-baseline_coverage,p-baseline_path,
                min((max(c-sample.coverage,p-sample.path)/2 for sample in negatives),default=1.0))
            if margin < 0:
                continue
            proposed_c, proposed_p = max(baseline_coverage,c-margin),max(baseline_path,p-margin)
            counts = confusion(train, proposed_c, proposed_p)
            if passing(counts):
                candidate = (counts["true_ready"], margin, proposed_c, proposed_p)
                if best is None or candidate > best:
                    best = candidate
    report = {"protocol_version": PROTOCOL, "sample_count": len(samples),
        "train_count": len(train), "validation_count": len(validation),
        "train_family_count": len(families["train"]), "validation_family_count": len(families["validation"]),
        "evaluated_operating_points": evaluated, "validation_evaluations": 0,
        'selection_protocol':'positive_frontier_margin_v1',
        "baseline": {"coverage": baseline_coverage, "path_quality": baseline_path},
        "thresholds_lowered": False, "production_settings_changed": False,
        "label_authority": "declared_source_review", "feature_replay_required": True,
        "small_sample_probability_guarantee": False, "acceptance_replacement": False}
    if best is None:
        return {**report, "status": "not_separable_at_current_boundaries", "suggestion": None}
    _, margin, coverage, path = best
    train_counts = confusion(train, coverage, path)
    validation_counts = confusion(validation, coverage, path)
    return {**report, "status": "development_validation_passed" if passing(validation_counts) else "validation_failed",
        "validation_evaluations": 1, "train": train_counts, "validation": validation_counts,
        'training_numeric_margin':margin,
        "suggestion": {"coverage": coverage, "path_quality": path} if passing(validation_counts) else None}
