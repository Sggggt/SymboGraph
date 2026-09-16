import math

import pytest
from pydantic import ValidationError

from app.intent_contracts import ChannelWeights
from app.services.entry_ranking import EntryScore, fuse_entry_candidates, rank_channel, rq_reconstruction_score


def row(key, score):
    return EntryScore(candidate_id=key, business_key="unit-test-" + key, score=score, witness_ids=("source-" + key,))


def test_each_channel_can_nominate_a_candidate_missing_from_the_dense_list():
    result = fuse_entry_candidates({"dense": [row("a", .9)], "rq": [row("b", -8)],
        "bm25": [row("b", 20), row("c", 15)]}, ChannelWeights(dense=.2, rq=.3, bm25=.5), limit=3)
    assert [x.candidate_id for x in result] == ["b", "c", "a"]
    assert result[0].score == pytest.approx(.8)
    assert [c.channel for c in result[0].channels] == ["rq", "bm25"]
    assert result[0].channels[1].witness_ids == ("source-b",)


def test_healthy_empty_channel_does_not_redistribute_its_weight():
    result = fuse_entry_candidates({"dense": [row("a", .9)], "bm25": []},
        ChannelWeights(dense=.25, rq=0, bm25=.75), limit=1)
    assert result[0].score == .25
    with pytest.raises(ValueError, match="scope_incomplete"):
        fuse_entry_candidates({"dense": [row("a", .9)]}, ChannelWeights(dense=.25, rq=0, bm25=.75), limit=1)


def test_each_healthy_channel_keeps_its_first_nominee_when_budget_allows():
    result = fuse_entry_candidates(
        {
            "dense": [row("dense-a", 10), row("dense-b", 9)],
            "bm25": [row("lexical", 100)],
        },
        ChannelWeights(dense=0.9, rq=0, bm25=0.1),
        limit=2,
    )
    assert {item.candidate_id for item in result} == {"dense-a", "lexical"}


def test_fusion_ignores_score_units_but_preserves_ranks_and_uses_stable_ties():
    weights = ChannelWeights(dense=.5, rq=0, bm25=.5)
    first = fuse_entry_candidates({"dense": [row("b", .5), row("a", .5), row("c", .1)],
        "bm25": [row("a", 1), row("c", .5)]}, weights, limit=3)
    second = fuse_entry_candidates({"dense": [row("a", 500), row("c", 100), row("b", 500)],
        "bm25": [row("c", 50), row("a", 100)]}, weights, limit=3)
    assert [(x.candidate_id, x.score) for x in first] == [(x.candidate_id, x.score) for x in second]
    assert [(x.candidate_id, r) for x, r in rank_channel([row("c", .1), row("b", .5), row("a", .5)])] == [("a", 1), ("b", 1), ("c", 3)]


def test_dense_only_preserves_dense_order_without_lexical_or_rq_dependencies():
    result = fuse_entry_candidates({"dense": [row("a", -.8), row("b", -.1)]},
        ChannelWeights(dense=1, rq=0, bm25=0), limit=2)
    assert [x.candidate_id for x in result] == ["b", "a"]
    assert all([c.channel for c in x.channels] == ["dense"] for x in result)


@pytest.mark.parametrize("attack", ["duplicate", "identity_change", "business_collision", "inactive_channel"])
def test_corrupt_or_unplanned_candidates_fail_closed(attack):
    lists = {"dense": [row("a", .9)], "bm25": [row("b", 1)]}
    if attack == "duplicate": lists["dense"] *= 2
    elif attack == "identity_change": lists["bm25"] = [EntryScore(candidate_id="a", business_key="foreign", score=1)]
    elif attack == "business_collision": lists["bm25"] = [EntryScore(candidate_id="b", business_key="unit-test-a", score=1)]
    elif attack == "inactive_channel": lists["rq"] = []
    with pytest.raises(ValueError):
        fuse_entry_candidates(lists, ChannelWeights(dense=.5, rq=0, bm25=.5), limit=2)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "1", True])
def test_score_cannot_be_nonfinite_or_coerced(score):
    with pytest.raises(ValidationError): row("a", score)


def test_rq_uses_all_residual_centers_not_only_last_center_or_matching_code():
    books = [[[10., 0.]], [[-7., 0.]], [[-1., 0.], [1., 0.]]]
    assert rq_reconstruction_score([2., 0.], books, [0, 0, 0]) == 0
    assert rq_reconstruction_score([2., 0.], books, [0, 0, 1]) == -4
    assert rq_reconstruction_score([2., 0.], books, [0, 0]) == -1
    assert rq_reconstruction_score([2., 0.], books, [0]) == -64


@pytest.mark.parametrize("query,books,prefix", [
    ([1.], [[[0.]]], []), ([1.], [[[0.]]], [1]), ([1.], [[[0.]]], [True]),
    ([1.], [[[0., 1.]]], [0]), ([math.nan], [[[0.]]], [0]),
    ([1.], [[[math.inf]]], [0]), ([1e308], [[[-1e308]]], [0]),
])
def test_rq_invalid_or_overflowing_data_never_becomes_a_valid_zero_score(query, books, prefix):
    with pytest.raises(ValueError): rq_reconstruction_score(query, books, prefix)
