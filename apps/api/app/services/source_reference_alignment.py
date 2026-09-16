"""Bounded surface correspondence for locator cards; never document identity proof."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal

from pydantic import Field

from app.retrieval_control_contracts import ControlContract

PROTOCOL = 'source_reference_alignment_v1'
MAX_TOKENS = 128
MAX_INITIALISM = 12
_WORDS = re.compile(r'[+-]?\d+(?:[.,]\d+)*|[^\W_](?:[^\W_]|[\u0300-\u036f])*(?:[.,]\d+)*', re.UNICODE)
_INITIALISM = re.compile(rf'[A-Z]{{2,{MAX_INITIALISM}}}')


class InitialismMapping(ControlContract):
    reference_span: tuple[int, int]
    candidate_span: tuple[int, int]
    reference_text: str
    candidate_text: str


class ReferenceAlignment(ControlContract):
    protocol_version: Literal['source_reference_alignment_v1'] = PROTOCOL
    status: Literal['aligned', 'no_alignment', 'not_evaluated']
    semantic_identity_proven: Literal[False] = False
    candidate_span: tuple[int, int] | None = None
    reference_token_count: int = Field(ge=0)
    candidate_token_count: int = Field(ge=0)
    initialisms: tuple[InitialismMapping, ...] = ()


@dataclass(frozen=True)
class _Token:
    normalized: str
    start: int
    end: int
    initialism: str | None


def _tokens(text):
    result = []
    for match in _WORDS.finditer(text):
        value = unicodedata.normalize('NFKC', match.group())
        acronym = value.casefold() if _INITIALISM.fullmatch(value) else None
        result.append(_Token(value.casefold(), match.start(), match.end(), acronym))
    return result


def _expands(initialism, tokens, start):
    return (initialism is not None and start + len(initialism) <= len(tokens)
            and ''.join(token.normalized[0] for token in tokens[start:start + len(initialism)]) == initialism)


def align_reference(reference: str, candidate: str) -> ReferenceAlignment:
    """Shortest path in a token DAG; no token deletion or replacement enumeration."""
    left, right = _tokens(reference), _tokens(candidate)
    sizes = dict(reference_token_count=len(left), candidate_token_count=len(right))
    if len(reference) > 512 or len(candidate) > 160 or max(len(left), len(right)) > MAX_TOKENS:
        return ReferenceAlignment(status='not_evaluated', **sizes)
    if not left or not right:
        return ReferenceAlignment(status='no_alignment', **sizes)
    # A cell is (initialism count, start token, predecessor i, predecessor j).
    cells = [[None] * (len(right) + 1) for _ in range(len(left) + 1)]
    for j in range(len(right)):
        cells[0][j] = (0, j, None, None)
    for i, token in enumerate(left):
        for j, other in enumerate(right):
            current = cells[i][j]
            if current is None:
                continue
            targets = []
            if token.normalized == other.normalized:
                targets.append((i + 1, j + 1, 0))
            if _expands(token.initialism, right, j):
                targets.append((i + 1, j + len(token.initialism), 1))
            if _expands(other.initialism, left, i):
                targets.append((i + len(other.initialism), j + 1, 1))
            for ni, nj, cost in targets:
                proposal = (current[0] + cost, current[1], i, j)
                old = cells[ni][nj]
                if old is None or (proposal[0], -proposal[1]) < (old[0], -old[1]):
                    cells[ni][nj] = proposal
    endpoints = [j for j in range(1, len(right) + 1) if cells[len(left)][j] is not None]
    if not endpoints:
        return ReferenceAlignment(status='no_alignment', **sizes)
    def order(j):
        count, start, _, _ = cells[len(left)][j]
        return count, right[j - 1].end - right[start].start, right[start].start
    end = min(endpoints, key=order)
    start = cells[len(left)][end][1]
    mappings, i, j = [], len(left), end
    while i:
        _, _, pi, pj = cells[i][j]
        if i - pi != 1 or j - pj != 1:
            a, b = (left[pi].start, left[i - 1].end), (right[pj].start, right[j - 1].end)
            mappings.append(InitialismMapping(reference_span=a, candidate_span=b,
                reference_text=reference[a[0]:a[1]], candidate_text=candidate[b[0]:b[1]]))
        i, j = pi, pj
    return ReferenceAlignment(status='aligned', candidate_span=(right[start].start, right[end - 1].end),
                              initialisms=tuple(reversed(mappings)), **sizes)
