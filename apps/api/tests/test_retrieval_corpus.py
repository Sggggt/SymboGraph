from types import SimpleNamespace

import numpy as np
import pytest

from app.services.retrieval_corpus import CorpusSource, RetrievalCorpus
from test_retrieval_path_features import task_fixture


def corpus():
    texts = ("The controller has a display setting.", "Maximum queueing delay\nWaiting before execution starts.")
    sources = [CorpusSource(f"source-{i}", f"doc-{i}", f"version-{i}", "Synthetic source", text,
                            10, 10 + len(text), "a" * 64) for i, text in enumerate(texts)]
    return RetrievalCorpus(knowledge_base_id="unit-test-kb", sources=sources,
        vectors=np.array([[1, 0], [0, 1]], dtype=np.float64), scope_hash="b" * 64, threads=1,
        target=SimpleNamespace(runtime_state_hash="c" * 64, activation_generation=1, vector_schema_hash="d" * 64))


def test_independent_facet_discovery_can_leave_the_failed_whole_query_route():
    view = corpus()
    old_scores = view.cosine_scores([1, 0])
    facet_scores = view.cosine_scores([0, 1])
    assert old_scores[0].argmax() == 0 and facet_scores[0].argmax() == 1
    candidates = view.discover(task=task_fixture(), missing_facet_ids=("f1",), facet_scores=facet_scores,
                               excluded_chunk_ids=("source-0",))
    assert candidates
    locator = view.locators[candidates[0].witness_id]
    assert locator.chunk_id == "source-1" and locator.document_version_id == "version-1"
    source = view.by_id[locator.chunk_id]
    assert source.text[locator.char_start - source.char_start:locator.char_end - source.char_start] == locator.surface
    assert candidates[0].scope_kind == "new_scope_proposal"


def test_text_scope_miss_never_claims_whole_corpus_fact_absence():
    lookup = corpus().literal_lookup("does not occur")
    assert lookup["matched_chunk_ids"] == ()
    assert lookup["observed_source_count"] == 2
    assert lookup["corpus_fact_absence_proven"] is False
    assert lookup["representation"] == "active_retrievable_text"


def test_candidate_keeps_decimal_and_source_location_without_inventing_roles():
    source = CorpusSource("source-0", "doc-0", "version-0", "Unit test memory report",
        "Budget is 1.25 GiB.\nA separate note", 0, 38, "a" * 64,
        section_path="Specification / Memory")
    view = RetrievalCorpus(knowledge_base_id="unit-test-kb", sources=[source], vectors=np.array([[1., 0.]]),
        scope_hash="b" * 64, threads=1, target=corpus().target)
    candidate = view.discover(task=task_fixture(), missing_facet_ids=("f1",),
        facet_scores=view.cosine_scores([1., 0.]))[0]
    assert "1.25" in candidate.surface
    assert candidate.source_title == source.title and candidate.source_section == source.section_path
    locator = view.locators[candidate.witness_id]
    assert source.text[locator.char_start:locator.char_end] == candidate.surface


@pytest.mark.parametrize("vector", [[0, 0], [1, 2, 3], [float("nan"), 1]])
def test_invalid_query_vectors_fail_instead_of_creating_fake_similarity(vector):
    with pytest.raises(ValueError):
        corpus().cosine_scores(vector)
