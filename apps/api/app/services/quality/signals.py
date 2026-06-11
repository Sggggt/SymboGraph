from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any


MOJIBAKE_MARKERS = ("\ufffd", "\u00c3", "\u00c2", "\u00e2", "\u9208", "\u9365", "\u9429", "\u95b3", "\u951f", "\u7d34", "\u6d93", "\u934f")
FORMULA_RE = re.compile(r"[\u2211\u222b\u2202\u221a\u221e\u2248\u2260\u2264\u2265\u00b1\u00d7\u00f7\u2208\u2209\u2282\u2286\u222a\u2229\u2192\u2190\u2194\u2200\u2203\u2207=<>^]")
FILENAME_RE = re.compile(r"^[\w .()\-\u4e00-\u9fff]+\.(?:pdf|pptx?|docx?|xlsx?|csv|txt|md|html?|ipynb|png|jpe?g|gif)$", re.IGNORECASE)
PATH_FRAGMENT_RE = re.compile(r"(?:[A-Za-z]:[\\/]|[/\\]|(?:^|[\s])\.\.?[/\\])")
DATE_OR_NUMBER_RE = re.compile(r"^(?:(?:19|20)\d{2}[-_/]?\d{1,2}[-_/]?\d{1,2}|\d{6,8}|p(?:age)?\s*\d+|\d+(?:\.\d+){1,4})$", re.IGNORECASE)
CAPITALIZED_TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4}\b")


@dataclass(frozen=True)
class TextQuality:
    length: int
    normalized_length: int
    mojibake_ratio: float
    control_char_count: int
    digit_ratio: float
    symbol_ratio: float
    alpha_ratio: float
    repeated_line_ratio: float
    toc_like: bool


@dataclass(frozen=True)
class StructuralRole:
    roles: tuple[str, ...] = ()
    structural_score: float = 0.0
    container_label: bool = False
    path_or_filename: bool = False


@dataclass(frozen=True)
class SemanticDensity:
    token_count: int
    unique_token_ratio: float
    definition_score: float
    entity_density: float
    term_density: float
    has_formula: bool = False
    has_table: bool = False


@dataclass(frozen=True)
class DomainSpecificity:
    local_idf: float | None = None
    corpus_frequency: int | None = None
    document_frequency: int | None = None
    entropy: float | None = None
    mutual_information: float | None = None
    chunk_support_count: int | None = None
    kg_degree: float | None = None
    kg_bridge_score: float | None = None
    genericity_score: float = 0.0
    specificity_score: float = 0.0


@dataclass(frozen=True)
class EvidenceGrounding:
    has_text_span: bool = False
    has_chunk: bool = False
    has_document: bool = False
    support_count: int = 0
    source_match: bool = False
    target_match: bool = False


@dataclass(frozen=True)
class ModelJudgment:
    verdict: str | None = None
    score: float | None = None
    reasons: tuple[str, ...] = ()
    cache_key: str | None = None


@dataclass(frozen=True)
class Provenance:
    knowledge_base_id: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    chunk_id: str | None = None
    extractor: str | None = None
    model: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class QualitySignals:
    target_type: str
    text: str
    normalized_text: str
    text_quality: TextQuality
    structural_role: StructuralRole
    semantic_density: SemanticDensity
    domain_specificity: DomainSpecificity
    evidence_grounding: EvidenceGrounding
    model_judgment: ModelJudgment = field(default_factory=ModelJudgment)
    provenance: Provenance = field(default_factory=Provenance)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text_for_quality(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_concept_text(text: str | None) -> str:
    value = normalize_text_for_quality(text)
    value = re.sub(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", lambda match: match.group(1) or match.group(2) or "", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[`~*#\[\]{}]", " ", value)
    for marker in ("鈥", "欌", "€", "榏"):
        value = value.replace(marker, "'")
    value = re.sub(r"\b([A-Za-z])'s\b", r"\1s", value)
    value = re.sub(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\u0370-\u03ff+\-/*^=<>()\s']", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    tokens: list[str] = []
    for token in value.split():
        if token == "max":
            token = "maximum"
        elif token == "min":
            token = "minimum"
        if re.fullmatch(r"[a-z]{4,}ies", token):
            token = token[:-3] + "y"
        elif re.fullmatch(r"[a-z]{4,}s", token) and not token.endswith(("ss", "ics")):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\u0370-\u03ff]+", text.lower())


def mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return marker_count / max(len(text), 1)


def is_toc_like(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    dotted = sum(1 for line in lines if re.search(r"\.{4,}\s*\d+$", line))
    numeric_short = sum(1 for line in lines if re.fullmatch(r"\d+(\.\d+)*", line) or len(line) <= 3)
    return dotted >= 4 or numeric_short / len(lines) > 0.45


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 3:
        return 0.0
    unique = len(set(lines))
    return max(0.0, 1.0 - unique / len(lines))


def ends_with_sentence_punctuation(text: str) -> bool:
    return (text or "").rstrip().endswith((".", "!", "?", "。", "！", "？"))


def structural_role_for_text(text: str, *, title: str | None = None, section: str | None = None, content_kind: str | None = None) -> StructuralRole:
    haystack = normalize_concept_text(" ".join(item for item in (title, section, text[:200]) if item))
    roles: set[str] = set()
    if content_kind:
        roles.add(str(content_kind))
    if is_toc_like(text):
        roles.add("toc")
    raw_text = normalize_text_for_quality(text)
    short_label = len(raw_text) <= 36 and len(tokenize(raw_text)) <= 3 and not ends_with_sentence_punctuation(raw_text)
    numeric_or_dated_label = DATE_OR_NUMBER_RE.fullmatch(raw_text) is not None
    formula_like = bool(FORMULA_RE.search(raw_text))
    mixed_short_number_label = short_label and not formula_like and any(char.isdigit() for char in raw_text) and len(tokenize(raw_text)) <= 4
    sparse_label = short_label and (
        numeric_or_dated_label
        or mixed_short_number_label
        or (text_quality_for_text(raw_text, raw_text).digit_ratio >= 0.25 and text_quality_for_text(raw_text, raw_text).alpha_ratio <= 0.65)
    )
    if sparse_label:
        roles.add("structural_label")
    if PATH_FRAGMENT_RE.search(text or "") or FILENAME_RE.fullmatch((text or "").strip()):
        roles.add("path_or_filename")
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    label_like_lines = sum(
        1
        for line in lines
        if len(line) <= 48 and len(tokenize(line)) <= 4 and not ends_with_sentence_punctuation(line)
    )
    if len(lines) >= 4 and label_like_lines / max(len(lines), 1) >= 0.75:
        roles.add("container_hint")
    score = 0.0
    if roles.intersection({"toc", "structural_label", "path_or_filename"}):
        score += 0.7
    if "container_hint" in roles:
        score += 0.25
    return StructuralRole(
        roles=tuple(sorted(roles)),
        structural_score=min(1.0, score),
        container_label="structural_label" in roles or "container_hint" in roles,
        path_or_filename="path_or_filename" in roles,
    )


def text_quality_for_text(text: str, normalized_text: str) -> TextQuality:
    length = len(text or "")
    alnum_count = sum(char.isalnum() for char in normalized_text)
    digit_count = sum(char.isdigit() for char in normalized_text)
    alpha_count = sum(char.isalpha() for char in normalized_text)
    symbol_count = sum(not char.isalnum() and not char.isspace() for char in normalized_text)
    return TextQuality(
        length=length,
        normalized_length=len(normalized_text),
        mojibake_ratio=round(mojibake_ratio(text), 4),
        control_char_count=sum(ord(char) < 32 and char not in "\n\r\t" for char in text or ""),
        digit_ratio=round(digit_count / max(len(normalized_text), 1), 4),
        symbol_ratio=round(symbol_count / max(len(normalized_text), 1), 4),
        alpha_ratio=round(alpha_count / max(len(normalized_text), 1), 4),
        repeated_line_ratio=round(repeated_line_ratio(text), 4),
        toc_like=is_toc_like(text),
    )


def semantic_density_for_text(text: str, *, content_kind: str | None = None, metadata: dict[str, Any] | None = None) -> SemanticDensity:
    normalized = normalize_text_for_quality(text)
    tokens = tokenize(normalized)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    definition_score = 1.0 if re.search(r"(^|\n)\s*.{2,120}?\s+(?:is|are|means|refers\s+to|defined\s+as|denotes|represents)\s+.{2,}", normalized, flags=re.IGNORECASE) or re.search(r"[\u4e00-\u9fffA-Za-z0-9 _/-]{2,80}(?:鏄寚|鎸囩殑鏄瘄瀹氫箟涓簗琛ㄧず涓簗绉颁负)[\u4e00-\u9fffA-Za-z0-9 _/-]{2,}", normalized) else 0.0
    capitalized_terms = CAPITALIZED_TERM_RE.findall(text or "")
    entity_density = min(1.0, len(capitalized_terms) / max(len(tokens), 1))
    long_terms = [token for token in tokens if len(token) >= 6 or re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", token)]
    term_density = min(1.0, len(set(long_terms)) / max(len(tokens), 1))
    has_formula = bool((metadata or {}).get("has_formula") or content_kind == "formula" or FORMULA_RE.search(text or ""))
    has_table = bool((metadata or {}).get("has_table") or content_kind == "table")
    return SemanticDensity(
        token_count=len(tokens),
        unique_token_ratio=round(unique_ratio, 4),
        definition_score=definition_score,
        entity_density=round(entity_density, 4),
        term_density=round(term_density, 4),
        has_formula=has_formula,
        has_table=has_table,
    )


def domain_specificity_for_text(normalized_text: str, *, corpus_texts: list[str] | None = None, genericity_hint: bool = False) -> DomainSpecificity:
    tokens = tokenize(normalized_text)
    compact = normalized_text.replace(" ", "")
    shape = {
        "token_count": len(tokens),
        "char_count": len(compact),
        "digit_ratio": sum(char.isdigit() for char in compact) / max(len(compact), 1),
        "symbol_ratio": sum((not char.isalnum()) for char in compact) / max(len(compact), 1),
    }
    shape_generic = (
        compact.isascii()
        and shape["token_count"] <= 1
        and shape["char_count"] <= 12
        and shape["digit_ratio"] == 0
        and shape["symbol_ratio"] == 0
    )
    genericity = 1.0 if genericity_hint or shape_generic else 0.0
    local_idf: float | None = None
    document_frequency: int | None = None
    if corpus_texts:
        document_frequency = sum(1 for text in corpus_texts if normalized_text and normalized_text in normalize_concept_text(text))
        total = len(corpus_texts)
        local_idf = max(0.0, min(1.0, math.log((total + 1) / (document_frequency + 1)) / math.log(total + 1))) if total > 1 else 1.0
    length_specificity = min(1.0, max(0.0, (len(tokens) - 1) / 3))
    specificity = local_idf if local_idf is not None else length_specificity
    specificity = max(0.0, min(1.0, specificity - 0.45 * genericity))
    return DomainSpecificity(
        local_idf=round(local_idf, 4) if local_idf is not None else None,
        corpus_frequency=None,
        document_frequency=document_frequency,
        entropy=None,
        mutual_information=None,
        chunk_support_count=None,
        kg_degree=None,
        kg_bridge_score=None,
        genericity_score=genericity,
        specificity_score=round(specificity, 4),
    )


def evidence_grounding_for_target(
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    evidence_text: str | None = None,
    source_name: str | None = None,
    target_name: str | None = None,
    support_count: int = 0,
) -> EvidenceGrounding:
    haystack = normalize_concept_text(evidence_text)
    source_norm = normalize_concept_text(source_name)
    target_norm = normalize_concept_text(target_name)
    return EvidenceGrounding(
        has_text_span=bool((evidence_text or "").strip()),
        has_chunk=bool(chunk_id),
        has_document=bool(document_id),
        support_count=max(0, int(support_count or 0)),
        source_match=bool(source_norm and source_norm in haystack),
        target_match=bool(target_norm and target_norm in haystack),
    )


def build_quality_signals(
    *,
    target_type: str,
    text: str,
    title: str | None = None,
    section: str | None = None,
    content_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    corpus_texts: list[str] | None = None,
    knowledge_base_id: str | None = None,
    document_id: str | None = None,
    document_version_id: str | None = None,
    chunk_id: str | None = None,
    evidence_text: str | None = None,
    source_name: str | None = None,
    target_name: str | None = None,
    support_count: int = 0,
    extractor: str | None = None,
    model: str | None = None,
    version: str | None = None,
) -> QualitySignals:
    normalized = normalize_concept_text(text) if target_type in {"concept", "relation"} else normalize_text_for_quality(text)
    return QualitySignals(
        target_type=target_type,
        text=text or "",
        normalized_text=normalized,
        text_quality=text_quality_for_text(text or "", normalized),
        structural_role=structural_role_for_text(text or "", title=title, section=section, content_kind=content_kind),
        semantic_density=semantic_density_for_text(text or "", content_kind=content_kind, metadata=metadata),
        domain_specificity=domain_specificity_for_text(normalized, corpus_texts=corpus_texts),
        evidence_grounding=evidence_grounding_for_target(
            chunk_id=chunk_id,
            document_id=document_id,
            evidence_text=evidence_text or text,
            source_name=source_name,
            target_name=target_name,
            support_count=support_count,
        ),
        provenance=Provenance(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            document_version_id=document_version_id,
            chunk_id=chunk_id,
            extractor=extractor,
            model=model,
            version=version,
        ),
    )
