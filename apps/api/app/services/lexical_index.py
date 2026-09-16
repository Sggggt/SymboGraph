"""Source-only BM25 preparation and exact scoring with a frozen statistics domain."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version
import math
import re
from typing import Callable, Sequence
import unicodedata

from app.retrieval_control_contracts import control_hash

PROTOCOL = "source_chunk_bm25_v1"
TOKENIZER_PROTOCOL = "source_jieba_nfkc_identifiers_v1"
SCORING_PROTOCOL = "positive_idf_bm25_v1"
# Identifier and number spans are preserved before segmenting other text.
_WORD_START = r"(?![\u3400-\u9fff])[^\W\d]"
_WORD_PART = r"(?![\u3400-\u9fff])[\w\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff]"
_SPANS = re.compile(r"[\u3400-\u9fff]+|" + _WORD_START + r"(?:" + _WORD_PART + r")*(?:[-.](?:" + _WORD_PART + r")+)*|\d+(?:\.\d+)*|[%°]", re.UNICODE)
_CJK = re.compile(r"^[\u3400-\u9fff]+$")


@dataclass(frozen=True)
class LexicalToken:
    term: str
    start: int
    end: int


@lru_cache(maxsize=1)
def _tokenizer():
    import jieba
    tokenizer = jieba.Tokenizer()
    # The dictionary is package-owned and immutable for this tokenizer instance.
    # No user dictionaries, learned terms or HMM guessing enter this protocol.
    with tokenizer.get_dict_file() as file:
        dictionary_hash = hashlib.sha256(file.read()).hexdigest()
    tokenizer.initialize()
    identity = control_hash({"protocol": TOKENIZER_PROTOCOL, "jieba": version("jieba"),
        "dictionary": dictionary_hash, "hmm": False, "mode": "default", "stopwords": [],
        "normalization": "per-token NFKC/casefold; raw offsets unchanged"})
    return tokenizer, identity


def tokenizer_identity() -> str:
    return _tokenizer()[1]


def scoring_identity(*, k1: float, b: float) -> str:
    if (
        type(k1) not in (int, float)
        or type(b) not in (int, float)
        or not math.isfinite(k1)
        or not math.isfinite(b)
        or k1 <= 0
        or not 0 <= b <= 1
    ):
        raise ValueError("lexical_scoring_parameters_invalid")
    return control_hash(
        {"protocol": SCORING_PROTOCOL, "k1": float(k1), "b": float(b)}
    )


def tokenize_source(text: str) -> tuple[LexicalToken, ...]:
    if not isinstance(text, str) or "\x00" in text:
        raise ValueError("lexical_source_text_invalid")
    result = []
    for match in _SPANS.finditer(text):
        raw = match.group()
        spans = _tokenizer()[0].tokenize(raw, HMM=False) if _CJK.fullmatch(raw) else ((raw, 0, len(raw)),)
        for term, start, end in spans:
            normalized = unicodedata.normalize("NFKC", term).casefold()
            if normalized.strip():
                result.append(LexicalToken(normalized, match.start() + start, match.start() + end))
    return tuple(result)


@dataclass(frozen=True)
class LexicalSource:
    chunk_id: str
    document_version_id: str
    text: str
    char_start: int = 0


@dataclass(frozen=True)
class IndexedDocument:
    chunk_id: str
    document_version_id: str
    char_start: int
    char_end: int
    raw_text_hash: str
    length: int


@dataclass(frozen=True)
class Posting:
    term: str
    chunk_id: str
    positions: tuple[tuple[int, int], ...]

    @property
    def tf(self):
        return len(self.positions)


@dataclass(frozen=True)
class BM25Snapshot:
    knowledge_base_id: str
    source_scope_hash: str
    tokenizer_hash: str
    scoring_hash: str
    k1: float
    b: float
    documents: tuple[IndexedDocument, ...]
    terms: tuple[tuple[str, int], ...]
    postings: tuple[Posting, ...]
    total_length: int
    statistics_hash: str
    postings_hash: str
    identity: str

    @property
    def document_count(self):
        return len(self.documents)

    @property
    def average_length(self):
        return self.total_length / self.document_count if self.document_count else 0.0


def _check_cancelled():
    from app.services.storage import raise_if_source_io_cancelled
    raise_if_source_io_cancelled()


def prepare_bm25_snapshot(knowledge_base_id: str, sources: Sequence[LexicalSource], *,
                          max_documents: int = 100000, max_postings: int = 4000000,
                          k1: float = 1.2, b: float = .75,
                          check_cancelled: Callable[[], None] = _check_cancelled) -> BM25Snapshot:
    check_cancelled()
    if not knowledge_base_id or type(max_documents) is not int or type(max_postings) is not int or min(max_documents, max_postings) < 1:
        raise ValueError("lexical_build_budget_invalid")
    if len(sources) > max_documents:
        raise ValueError("lexical_document_budget_exceeded")
    if len({source.chunk_id for source in sources}) != len(sources):
        raise ValueError("lexical_duplicate_source")
    scoring_hash = scoring_identity(k1=k1, b=b)
    documents, postings, frequencies = [], [], Counter()
    for source in sorted(sources, key=lambda row: row.chunk_id):
        check_cancelled()
        if not source.chunk_id or not source.document_version_id or type(source.char_start) is not int or source.char_start < 0:
            raise ValueError("lexical_source_address_invalid")
        tokens = tokenize_source(source.text)
        positions = defaultdict(list)
        for index, token in enumerate(tokens):
            if index % 128 == 0:
                check_cancelled()
            positions[token.term].append((source.char_start + token.start, source.char_start + token.end))
        if len(postings) + len(positions) > max_postings:
            raise ValueError("lexical_posting_budget_exceeded")
        frequencies.update(positions.keys())
        documents.append(IndexedDocument(source.chunk_id, source.document_version_id, source.char_start,
            source.char_start + len(source.text), hashlib.sha256(source.text.encode()).hexdigest(), len(tokens)))
        postings.extend(Posting(term, source.chunk_id, tuple(spans)) for term, spans in sorted(positions.items()))
    th = tokenizer_identity()
    document_facts = [(d.chunk_id, d.document_version_id, d.char_start, d.char_end, d.raw_text_hash, d.length) for d in documents]
    terms = tuple(sorted(frequencies.items()))
    ordered_postings = tuple(sorted(postings, key=lambda p: (p.term, p.chunk_id)))
    total_length = sum(d.length for d in documents)
    source_hash = control_hash({"kb": knowledge_base_id, "documents": document_facts})
    stats_hash = control_hash({"N": len(documents), "total_length": total_length, "df": terms})
    postings_hash = control_hash([(p.term, p.chunk_id, p.positions) for p in ordered_postings])
    identity = control_hash({"protocol": PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
        "scoring_hash": scoring_hash,
        "source_scope": source_hash, "tokenizer": th, "statistics": stats_hash, "postings": postings_hash})
    return BM25Snapshot(knowledge_base_id, source_hash, th, scoring_hash, float(k1), float(b), tuple(documents), terms, ordered_postings,
        total_length, stats_hash, postings_hash, identity)


def bm25_term_score(*, tf: int, length: int, document_count: int, document_frequency: int,
                    average_length: float, k1: float = 1.2, b: float = .75) -> float:
    if (any(type(x) is not int for x in (tf, length, document_count, document_frequency))
            or not 0 < tf <= length or not 0 < document_frequency <= document_count):
        raise ValueError("lexical_statistics_invalid")
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (average_length, k1, b))
            or average_length <= 0 or k1 <= 0 or not 0 <= b <= 1):
        raise ValueError("lexical_scoring_parameters_invalid")
    inverse_frequency = math.log1p((document_count - document_frequency + .5) / (document_frequency + .5))
    denominator = tf + k1 * (1 - b + b * length / average_length)
    score = inverse_frequency * tf * (k1 + 1) / denominator
    if not math.isfinite(score) or score < 0:
        raise ValueError("lexical_scoring_nonfinite")
    return score


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float
    witnesses: tuple[Posting, ...]


def query_terms(lexical_terms: Sequence[str]) -> tuple[str, ...]:
    if len(lexical_terms) > 24 or any(not isinstance(term, str) or len(term) > 160 for term in lexical_terms):
        raise ValueError("lexical_query_budget_exceeded")
    result = tuple(sorted({token.term for term in lexical_terms for token in tokenize_source(term)}))
    if not result:
        raise ValueError("lexical_query_has_no_tokens")
    return result


def search_bm25_snapshot(snapshot: BM25Snapshot, lexical_terms: Sequence[str], *,
                         eligible_chunk_ids: frozenset[str] | None = None, limit: int = 64,
                         k1: float | None = None, b: float | None = None,
                         check_cancelled: Callable[[], None] = _check_cancelled) -> tuple[BM25Hit, ...]:
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError("lexical_query_limit_invalid")
    check_cancelled()
    k1 = snapshot.k1 if k1 is None else k1
    b = snapshot.b if b is None else b
    if k1 != snapshot.k1 or b != snapshot.b:
        raise ValueError("lexical_scoring_identity_changed")
    query = set(query_terms(lexical_terms))
    lengths = {document.chunk_id: document.length for document in snapshot.documents}
    if eligible_chunk_ids is not None and not eligible_chunk_ids <= lengths.keys():
        raise ValueError("lexical_filter_outside_snapshot")
    df = dict(snapshot.terms)
    scores, witnesses = defaultdict(list), defaultdict(list)
    for index, posting in enumerate(snapshot.postings):
        if index % 128 == 0:
            check_cancelled()
        if posting.term not in query or eligible_chunk_ids is not None and posting.chunk_id not in eligible_chunk_ids:
            continue
        scores[posting.chunk_id].append(bm25_term_score(tf=posting.tf, length=lengths[posting.chunk_id],
            document_count=snapshot.document_count, document_frequency=df[posting.term],
            average_length=snapshot.average_length, k1=k1, b=b))
        witnesses[posting.chunk_id].append(posting)
    result = [BM25Hit(key, math.fsum(parts), tuple(witnesses[key])) for key, parts in scores.items()]
    return tuple(sorted(result, key=lambda hit: (-hit.score, hit.chunk_id))[:limit])
