"""Bounded entry-only scoring. No path, edge, or answer-quality modification."""
from __future__ import annotations

import math
from typing import Sequence

from pydantic import Field, field_validator, model_validator

from app.intent_contracts import Channel, ChannelWeights, ClosedPlan

FUSION_PROTOCOL = "weighted_rrf_entry_v2"
RQ_PROTOCOL = "rq_reconstruction_entry_v1"
MAX_CANDIDATES_PER_CHANNEL = 4096


class EntryScore(ClosedPlan):
    candidate_id: str = Field(min_length=1, max_length=160)
    business_key: str = Field(min_length=1, max_length=512)
    score: float = Field(allow_inf_nan=False)
    witness_ids: tuple[str, ...] = Field(default=(), max_length=256)

    @field_validator("score", mode="before")
    @classmethod
    def numeric(cls, value):
        if type(value) not in (int, float):
            raise ValueError("entry_score_not_numeric")
        return value

    @model_validator(mode="after")
    def unique_witnesses(self):
        if len(set(self.witness_ids)) != len(self.witness_ids):
            raise ValueError("entry_duplicate_witness")
        return self


class ChannelContribution(ClosedPlan):
    channel: Channel
    raw_score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1, strict=True)
    weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    contribution: float = Field(ge=0, le=1, allow_inf_nan=False)
    witness_ids: tuple[str, ...] = ()


class FusedEntry(ClosedPlan):
    candidate_id: str
    business_key: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    channels: tuple[ChannelContribution, ...]


def rank_channel(rows: Sequence[EntryScore]) -> tuple[tuple[EntryScore, int], ...]:
    if len(rows) > MAX_CANDIDATES_PER_CHANNEL:
        raise ValueError("entry_candidate_budget_exceeded")
    if (len({row.candidate_id for row in rows}) != len(rows)
            or len({row.business_key for row in rows}) != len(rows)):
        raise ValueError("entry_duplicate_candidate")
    result, previous, rank = [], None, 0
    for index, row in enumerate(sorted(rows, key=lambda item: (-item.score, item.business_key)), 1):
        if previous is None or row.score != previous:
            rank = index
        result.append((row, rank))
        previous = row.score
    return tuple(result)


def fuse_entry_candidates(lists: dict[Channel, Sequence[EntryScore]], weights: ChannelWeights,
                          *, limit: int, k: int = 60) -> tuple[FusedEntry, ...]:
    if type(limit) is not int or not 1 <= limit <= MAX_CANDIDATES_PER_CHANNEL:
        raise ValueError("entry_output_budget_invalid")
    if type(k) is not int or not 1 <= k <= 10000:
        raise ValueError("entry_fusion_smoothing_invalid")
    effective = weights.effective()
    enabled = {name for name, value in effective.items() if value > 0}
    if set(lists) != enabled:
        raise ValueError("entry_channel_scope_incomplete")
    identities, business_ids, contributions = {}, {}, {}
    for channel in ("dense", "rq", "bm25"):
        if channel not in enabled:
            continue
        for row, rank in rank_channel(lists[channel]):
            if ((row.candidate_id in identities and identities[row.candidate_id] != row.business_key)
                    or (row.business_key in business_ids and business_ids[row.business_key] != row.candidate_id)):
                raise ValueError("entry_cross_channel_identity_conflict")
            identities[row.candidate_id] = row.business_key
            business_ids[row.business_key] = row.candidate_id
            contributions.setdefault(row.candidate_id, []).append(ChannelContribution(channel=channel,
                raw_score=row.score, rank=rank, weight=effective[channel],
                contribution=effective[channel] * ((k + 1) / (k + rank)), witness_ids=row.witness_ids))
    rows = [FusedEntry(candidate_id=key, business_key=identities[key],
                      score=min(1.0, math.fsum(c.contribution for c in parts)), channels=tuple(parts))
            for key, parts in contributions.items()]
    ranked = sorted(rows, key=lambda row: (-row.score, row.business_key))
    selected = list(ranked[:limit])
    leaders = list(
        dict.fromkeys(
            ranked_channel[0][0].candidate_id
            for channel in ("dense", "rq", "bm25")
            if channel in enabled
            and (ranked_channel := rank_channel(lists[channel]))
        )
    )
    if limit >= len(leaders):
        leader_set = set(leaders)
        selected_ids = {row.candidate_id for row in selected}
        for leader in leaders:
            if leader in selected_ids:
                continue
            replacement = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index].candidate_id not in leader_set
                ),
                None,
            )
            if replacement is None:
                break
            selected_ids.discard(selected[replacement].candidate_id)
            selected[replacement] = next(
                row for row in ranked if row.candidate_id == leader
            )
            selected_ids.add(leader)
        rank_index = {row.candidate_id: index for index, row in enumerate(ranked)}
        selected.sort(key=lambda row: rank_index[row.candidate_id])
    return tuple(selected)


def rq_reconstruction_score(query: Sequence[float], codebooks: Sequence[Sequence[Sequence[float]]],
                            prefix: Sequence[int]) -> float:
    """Compare the complete prefix reconstruction, not the last residual center."""
    if not 1 <= len(prefix) <= 3 or len(prefix) > len(codebooks) or not 1 <= len(query) <= 16384:
        raise ValueError("entry_rq_shape_invalid")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in query):
        raise ValueError("entry_rq_query_invalid")
    centers = []
    for level, key in enumerate(prefix):
        if type(key) is not int or not 0 <= key < len(codebooks[level]):
            raise ValueError("entry_rq_prefix_invalid")
        center = codebooks[level][key]
        if len(center) != len(query) or any(type(x) not in (int, float) or not math.isfinite(x) for x in center):
            raise ValueError("entry_rq_codebook_invalid")
        centers.append(center)
    try:
        score = -math.fsum((q - math.fsum(c[index] for c in centers)) ** 2 for index, q in enumerate(query))
    except (OverflowError, ValueError):
        raise ValueError("entry_rq_numeric_overflow") from None
    if not math.isfinite(score):
        raise ValueError("entry_rq_numeric_overflow")
    return score
