import asyncio
import math

import pytest

from app.services.lexical_index import (LexicalSource, bm25_term_score, prepare_bm25_snapshot,
    query_terms, search_bm25_snapshot, tokenize_source)


def corpus():
    return prepare_bm25_snapshot("unit-test-kb", (
        LexicalSource("a", "v1", "queue_limit sets the queue size. queue_limit is 12.", 100),
        LexicalSource("b", "v2", "Timeout describes how long execution may take."),
        LexicalSource("c", "v3", "queue_limit is distinct from timeout."),
    ))


def test_bm25_matches_an_independent_hand_calculation():
    actual = bm25_term_score(tf=3, length=20, document_count=10, document_frequency=2,
        average_length=15, k1=1.2, b=.75)
    expected = math.log(1 + 8.5 / 2.5) * 3 * 2.2 / (3 + 1.2 * (.25 + .75 * 20 / 15))
    assert actual == pytest.approx(expected, rel=1e-14)


def test_filtering_keeps_the_same_kb_statistics_and_query_repetition_has_no_extra_weight():
    snapshot = corpus()
    all_hits = search_bm25_snapshot(snapshot, ["queue_limit"])
    filtered = search_bm25_snapshot(snapshot, ["queue_limit", "queue_limit"], eligible_chunk_ids=frozenset({"a"}))
    assert len(filtered) == 1
    assert filtered[0].score == next(hit.score for hit in all_hits if hit.chunk_id == "a")
    assert snapshot.document_count == 3
    assert dict(snapshot.terms)["queue_limit"] == 2


def test_raw_token_offsets_survive_normalization_and_identifier_punctuation():
    source = "中文检索：QUEUE_LIMIT = 3.5%; Cafe\u0301 and ﬀ."
    tokens = tokenize_source(source)
    import unicodedata
    for token in tokens:
        assert unicodedata.normalize("NFKC", source[token.start:token.end]).casefold() == token.term
    assert "queue_limit" in {t.term for t in tokens}
    assert "3.5" in {t.term for t in tokens}
    assert "ff" in {t.term for t in tokens}


def test_chinese_and_english_queries_retrieve_their_actual_source_with_raw_witnesses():
    sources = (LexicalSource("zh", "zv", "知识图谱支持中文检索和关系查询。", 20),
               LexicalSource("en", "ev", "Identifiers include queue_limit and execution_timeout."))
    snapshot = prepare_bm25_snapshot("unit-test-multilingual", sources)
    assert search_bm25_snapshot(snapshot, ["知识图谱"])[0].chunk_id == "zh"
    assert search_bm25_snapshot(snapshot, ["execution_timeout"])[0].chunk_id == "en"
    hit = search_bm25_snapshot(snapshot, ["知识图谱"])[0]
    for witness in hit.witnesses:
        for start, end in witness.positions:
            assert sources[0].text[start-20:end-20].strip()


def test_identical_source_order_does_not_change_snapshot_but_source_or_kb_change_does():
    sources = [LexicalSource("a", "v1", "alpha beta"), LexicalSource("b", "v2", "gamma alpha")]
    first = prepare_bm25_snapshot("unit-test-a", sources)
    assert first == prepare_bm25_snapshot("unit-test-a", list(reversed(sources)))
    assert first.identity != prepare_bm25_snapshot("unit-test-b", sources).identity
    assert first.identity != prepare_bm25_snapshot("unit-test-a", [sources[0], LexicalSource("b", "v3", "gamma alpha")]).identity
    assert first.identity != prepare_bm25_snapshot("unit-test-a", [sources[0], LexicalSource("b", "v2", "gamma beta")]).identity


def test_bm25_scoring_parameters_are_part_of_the_frozen_index_identity():
    sources = (
        LexicalSource("unit-test-a", "unit-test-v1", "alpha beta"),
        LexicalSource("unit-test-b", "unit-test-v2", "beta gamma"),
    )
    default = prepare_bm25_snapshot("unit-test-kb", sources)
    changed = prepare_bm25_snapshot("unit-test-kb", sources, k1=1.6, b=.5)
    assert default.identity != changed.identity
    assert default.scoring_hash != changed.scoring_hash
    with pytest.raises(ValueError, match="scoring_identity_changed"):
        search_bm25_snapshot(default, ["beta"], k1=1.6, b=.5)


def test_empty_index_is_distinct_from_invalid_empty_query_and_healthy_zero_hit():
    empty = prepare_bm25_snapshot("unit-test-empty", [])
    assert empty.document_count == empty.total_length == 0
    assert search_bm25_snapshot(empty, ["validterm"]) == ()
    assert search_bm25_snapshot(corpus(), ["unobservedterm"]) == ()
    with pytest.raises(ValueError, match="no_tokens"): query_terms(["...", " "])
    with pytest.raises(ValueError, match="outside_snapshot"):
        search_bm25_snapshot(corpus(), ["queue"], eligible_chunk_ids=frozenset({"foreign"}))


@pytest.mark.parametrize("kwargs", [{"tf": 0}, {"tf": True}, {"length": 0}, {"document_frequency": 3},
    {"document_count": 0}, {"average_length": 0}, {"k1": -1}, {"b": 2}, {"b": math.nan}])
def test_invalid_statistics_cannot_produce_a_rankable_score(kwargs):
    values = dict(tf=1, length=5, document_count=2, document_frequency=1, average_length=5)
    values.update(kwargs)
    with pytest.raises(ValueError): bm25_term_score(**values)


def test_build_and_query_check_cancellation_without_partial_result():
    def cancelled(): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        prepare_bm25_snapshot("unit-test", [], check_cancelled=cancelled)
    with pytest.raises(asyncio.CancelledError):
        search_bm25_snapshot(corpus(), ["queue"], check_cancelled=cancelled)


def test_bounded_build_does_not_silently_publish_an_incomplete_index():
    sources = [LexicalSource("a", "v", "alpha beta")]
    with pytest.raises(ValueError, match="posting_budget"):
        prepare_bm25_snapshot("unit-test", sources, max_postings=1)
    with pytest.raises(ValueError, match="duplicate_source"):
        prepare_bm25_snapshot("unit-test", sources * 2)
    with pytest.raises(ValueError, match="source_text_invalid"):
        prepare_bm25_snapshot("unit-test", [LexicalSource("a", "v", "bad\x00text")])


def test_unicode_canonical_forms_and_non_ascii_identifiers_share_terms_without_losing_offsets():
    assert query_terms(["Café naïve Δ_value"]) == query_terms(["Cafe\u0301 nai\u0308ve δ_value"])
    assert "delta_value" in query_terms(["ＤＥＬＴＡ_value"])
    assert query_terms(["queue_limit中文"]) == tuple(sorted(set(query_terms(["queue_limit"]) + query_terms(["中文"]))))
