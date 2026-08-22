from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentVersion


LANGUAGE_DETECTION_PROTOCOL_VERSION = "document_language_unicode_script_v1"
LANGUAGE_IDENTITY_SCOPE_PROTOCOL_VERSION = "active_chunk_language_identity_scope_v1"
LANGUAGE_DETECTION_SAMPLE_MAX_CHARS = 20_000
LANGUAGE_DETECTION_SUPPORTED_LANGUAGES = ("en", "ja", "ko", "zh")
LANGUAGE_IDENTITY_SOURCES = frozenset(
    {"explicit_metadata", "deterministic_detection", "unknown"}
)
LANGUAGE_CARD_FIELDS = frozenset(
    {
        "protocol_version",
        "status",
        "language",
        "normalized_language_tag",
        "source",
        "confidence",
        "input_hash",
        "sample_hash",
        "sample_char_count",
        "total_letter_count",
        "signals",
        "decision_reason",
    }
)
LANGUAGE_RECORD_FIELDS = (
    "language",
    "language_source",
    "language_confidence",
    "language_detection_protocol_version",
    "language_detection_hash",
    "language_metadata_json",
)

_LANGUAGE_TAG_RE = re.compile(
    r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$"
)
_PRIMARY_LANGUAGE_ALIASES = {
    "iw": "he",
    "in": "id",
    "ji": "yi",
}
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


class InvalidExplicitLanguageMetadata(ValueError):
    """Raised when user supplied language metadata is not a bounded BCP-47 tag."""


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_explicit_language_tag(value: str | None) -> str | None:
    """Validate and normalize an explicit language tag without guessing.

    The full normalized tag is retained in the audit card while the persisted
    relation bucket uses its canonical primary language.  ``und`` is accepted
    as an explicit request to keep the language unknown.
    """

    if value is None:
        return None
    normalized = value.strip().replace("_", "-").lower()
    if not normalized:
        return None
    if len(normalized) > 32 or not _LANGUAGE_TAG_RE.fullmatch(normalized):
        raise InvalidExplicitLanguageMetadata(
            "language must be a BCP-47 style tag with a 2-3 letter primary subtag"
        )
    return normalized


def canonical_primary_language(tag: str | None) -> str | None:
    normalized = normalize_explicit_language_tag(tag)
    if normalized is None:
        return None
    primary = normalized.split("-", 1)[0]
    primary = _PRIMARY_LANGUAGE_ALIASES.get(primary, primary)
    return None if primary == "und" else primary


def pending_language_metadata(explicit_language: str | None = None) -> dict[str, Any]:
    normalized_tag = normalize_explicit_language_tag(explicit_language)
    return {
        "protocol_version": LANGUAGE_DETECTION_PROTOCOL_VERSION,
        "status": "pending",
        "explicit_language_tag": normalized_tag,
        "supported_detection_languages": list(
            LANGUAGE_DETECTION_SUPPORTED_LANGUAGES
        ),
    }


def pending_language_record_fields(
    explicit_language: str | None = None,
) -> dict[str, Any]:
    return {
        "language": None,
        "language_source": None,
        "language_confidence": None,
        "language_detection_protocol_version": LANGUAGE_DETECTION_PROTOCOL_VERSION,
        "language_detection_hash": None,
        "language_metadata_json": pending_language_metadata(explicit_language),
    }


def explicit_language_from_pending_metadata(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    if metadata.get("status") != "pending":
        return None
    if metadata.get("protocol_version") != LANGUAGE_DETECTION_PROTOCOL_VERSION:
        return None
    value = metadata.get("explicit_language_tag")
    return normalize_explicit_language_tag(value) if value is not None else None


def _normalized_document_text(sections: Iterable[Any]) -> str:
    parts: list[str] = []
    for section in sections:
        title = str(getattr(section, "title", "") or "").strip()
        text = str(getattr(section, "text", "") or "").strip()
        if title:
            parts.append(title)
        if text:
            parts.append(text)
    normalized = unicodedata.normalize("NFKC", "\n".join(parts))
    return re.sub(r"\s+", " ", normalized).strip()


def _bounded_sample(text: str) -> str:
    if len(text) <= LANGUAGE_DETECTION_SAMPLE_MAX_CHARS:
        return text
    separator = "\n"
    content_budget = LANGUAGE_DETECTION_SAMPLE_MAX_CHARS - len(separator)
    head_chars = (content_budget + 1) // 2
    tail_chars = content_budget - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    return f"{text[:head_chars]}{separator}{tail}"


def _script_name(character: str) -> str:
    codepoint = ord(character)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    ):
        return "han"
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "katakana"
    if (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return "hangul"
    if "LATIN" in unicodedata.name(character, ""):
        return "latin"
    return "other"


def _auto_language_decision(sample: str) -> tuple[str | None, float, str, dict[str, Any]]:
    script_counts: Counter[str] = Counter()
    for character in sample:
        if unicodedata.category(character).startswith("L"):
            script_counts[_script_name(character)] += 1
    total_letters = sum(script_counts.values())
    latin_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", sample.casefold())
    english_stopword_hits = sum(
        1 for token in latin_tokens if token in _ENGLISH_STOPWORDS
    )
    signals = {
        "script_counts": {
            name: int(script_counts.get(name, 0))
            for name in ("han", "hiragana", "katakana", "hangul", "latin", "other")
        },
        "latin_token_count": len(latin_tokens),
        "english_stopword_hits": english_stopword_hits,
        "minimum_letter_count": 12,
        "dominant_script_ratio_threshold": 0.62,
    }
    if total_letters < 12:
        return None, 0.0, "insufficient_letter_evidence", signals

    han = script_counts.get("han", 0)
    kana = script_counts.get("hiragana", 0) + script_counts.get("katakana", 0)
    hangul = script_counts.get("hangul", 0)
    latin = script_counts.get("latin", 0)
    japanese_ratio = (han + kana) / total_letters
    korean_ratio = (han + hangul) / total_letters
    han_ratio = han / total_letters
    latin_ratio = latin / total_letters

    if kana >= 4 and japanese_ratio >= 0.62:
        confidence = round(min(0.99, 0.55 + 0.35 * japanese_ratio + 0.01 * kana), 6)
        return "ja", confidence, "kana_plus_han_dominance", signals
    if hangul >= 6 and korean_ratio >= 0.62:
        confidence = round(min(0.99, 0.55 + 0.35 * korean_ratio + 0.01 * hangul), 6)
        return "ko", confidence, "hangul_dominance", signals
    if han >= 6 and kana == 0 and hangul == 0 and han_ratio >= 0.62:
        confidence = round(min(0.99, 0.55 + 0.44 * han_ratio), 6)
        return "zh", confidence, "han_dominance_without_kana_or_hangul", signals
    if (
        latin >= 12
        and latin_ratio >= 0.80
        and len(latin_tokens) >= 6
        and english_stopword_hits >= 2
    ):
        confidence = round(
            min(0.99, 0.50 + 0.30 * latin_ratio + 0.025 * english_stopword_hits),
            6,
        )
        return "en", confidence, "latin_dominance_with_english_closed_lexicon", signals
    return None, 0.0, "mixed_unsupported_or_low_confidence", signals


def detect_document_language(
    sections: Iterable[Any],
    *,
    explicit_language: str | None = None,
) -> dict[str, Any]:
    normalized_tag = normalize_explicit_language_tag(explicit_language)
    normalized_text = _normalized_document_text(sections)
    sample = _bounded_sample(normalized_text)
    input_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    sample_hash = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    if normalized_tag is not None:
        language = canonical_primary_language(normalized_tag)
        source = "explicit_metadata" if language is not None else "unknown"
        confidence = 1.0 if language is not None else 0.0
        decision_reason = (
            "validated_explicit_bcp47_primary_language"
            if language is not None
            else "explicit_und_language"
        )
        _, _, _, signals = _auto_language_decision(sample)
        signals = {
            **signals,
            "detector_observation_only": True,
            "explicit_metadata_precedence": True,
        }
    else:
        language, confidence, decision_reason, signals = _auto_language_decision(sample)
        source = "deterministic_detection" if language is not None else "unknown"

    total_letter_count = sum(
        int(value)
        for value in dict(signals.get("script_counts") or {}).values()
    )
    card = {
        "protocol_version": LANGUAGE_DETECTION_PROTOCOL_VERSION,
        "status": "resolved",
        "language": language,
        "normalized_language_tag": normalized_tag,
        "source": source,
        "confidence": round(float(confidence), 6),
        "input_hash": input_hash,
        "sample_hash": sample_hash,
        "sample_char_count": len(sample),
        "total_letter_count": total_letter_count,
        "signals": signals,
        "decision_reason": decision_reason,
    }
    detection_hash = _canonical_hash(card)
    return {
        "language": language,
        "language_source": source,
        "language_confidence": card["confidence"],
        "language_detection_protocol_version": LANGUAGE_DETECTION_PROTOCOL_VERSION,
        "language_detection_hash": detection_hash,
        "language_metadata_json": card,
    }


def apply_language_identity(record: Document | DocumentVersion, identity: dict[str, Any]) -> None:
    for field_name in LANGUAGE_RECORD_FIELDS:
        setattr(record, field_name, identity.get(field_name))


def language_identity_summary(identity: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(identity.get("language_metadata_json") or {})
    return {
        "status": metadata.get("status") or (
            "resolved" if identity.get("language_detection_hash") else "pending"
        ),
        "language": identity.get("language"),
        "source": identity.get("language_source"),
        "confidence": identity.get("language_confidence"),
        "protocol_version": identity.get("language_detection_protocol_version"),
        "detection_hash": identity.get("language_detection_hash"),
        "explicit_language_tag": metadata.get("explicit_language_tag")
        or metadata.get("normalized_language_tag"),
        "decision_reason": metadata.get("decision_reason"),
    }


def language_identity_from_record(record: Document | DocumentVersion | None) -> dict[str, Any]:
    if record is None:
        return {
            "valid": False,
            "known": False,
            "language": None,
            "reason": "record_missing",
            "detection_hash": None,
            "protocol_version": None,
            "source": None,
            "confidence": None,
        }
    metadata = dict(getattr(record, "language_metadata_json", None) or {})
    stored_hash = getattr(record, "language_detection_hash", None)
    reasons: list[str] = []
    if set(metadata) != LANGUAGE_CARD_FIELDS:
        reasons.append("metadata_schema_invalid")
    if metadata.get("status") != "resolved":
        reasons.append("metadata_not_resolved")
    if metadata.get("protocol_version") != LANGUAGE_DETECTION_PROTOCOL_VERSION:
        reasons.append("unsupported_protocol")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        reasons.append("detection_hash_missing")
    elif _canonical_hash(metadata) != stored_hash:
        reasons.append("detection_hash_mismatch")
    source = getattr(record, "language_source", None)
    if source not in LANGUAGE_IDENTITY_SOURCES:
        reasons.append("source_invalid")
    language = getattr(record, "language", None)
    if language != metadata.get("language"):
        reasons.append("language_mirror_mismatch")
    if source != metadata.get("source"):
        reasons.append("source_mirror_mismatch")
    confidence = getattr(record, "language_confidence", None)
    if confidence != metadata.get("confidence"):
        reasons.append("confidence_mirror_mismatch")
    protocol = getattr(record, "language_detection_protocol_version", None)
    if protocol != metadata.get("protocol_version"):
        reasons.append("protocol_mirror_mismatch")
    if language is not None:
        try:
            if canonical_primary_language(language) != language:
                reasons.append("language_not_canonical_primary")
        except InvalidExplicitLanguageMetadata:
            reasons.append("language_invalid")
    valid = not reasons
    return {
        "valid": valid,
        "known": bool(valid and language),
        "language": language if valid else None,
        "reason": "ok" if valid else ",".join(sorted(set(reasons))),
        "detection_hash": stored_hash if valid else None,
        "protocol_version": protocol if valid else None,
        "source": source if valid else None,
        "confidence": confidence if valid else None,
    }


def load_chunk_language_identities(
    db: Session,
    chunks: list[Chunk],
    *,
    documents: dict[str, Document] | None = None,
) -> dict[str, dict[str, Any]]:
    document_ids = {str(chunk.document_id) for chunk in chunks}
    if documents is None:
        documents = {
            str(document.id): document
            for document in db.scalars(
                select(Document).where(Document.id.in_(document_ids))
            ).all()
        }
    else:
        documents = {str(key): value for key, value in documents.items()}
    version_ids = {str(chunk.document_version_id) for chunk in chunks}
    versions = {
        str(version.id): version
        for version in db.scalars(
            select(DocumentVersion).where(DocumentVersion.id.in_(version_ids))
        ).all()
    }
    identities: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.id)
        document = documents.get(str(chunk.document_id))
        version = versions.get(str(chunk.document_version_id))
        document_identity = language_identity_from_record(document)
        version_identity = language_identity_from_record(version)
        consistency_reasons: list[str] = []
        if str(chunk.state) != "active":
            consistency_reasons.append("chunk_not_active")
        if version is None:
            consistency_reasons.append("document_version_missing")
        else:
            if str(version.document_id) != str(chunk.document_id):
                consistency_reasons.append("document_version_document_mismatch")
            if not bool(version.is_active):
                consistency_reasons.append("document_version_not_active")
        if not document_identity["valid"]:
            consistency_reasons.append(
                f"document_identity:{document_identity['reason']}"
            )
        if not version_identity["valid"]:
            consistency_reasons.append(
                f"document_version_identity:{version_identity['reason']}"
            )
        if (
            document_identity["valid"]
            and version_identity["valid"]
            and document_identity["detection_hash"]
            != version_identity["detection_hash"]
        ):
            consistency_reasons.append("document_version_hash_mismatch")
        consistent = not consistency_reasons
        language = version_identity["language"] if consistent else None
        identities[chunk_id] = {
            "valid": consistent,
            "known": bool(consistent and language),
            "language": language,
            "reason": "ok" if consistent else ";".join(consistency_reasons),
            "detection_hash": (
                version_identity["detection_hash"] if consistent else None
            ),
            "protocol_version": (
                version_identity["protocol_version"] if consistent else None
            ),
            "source": version_identity["source"] if consistent else None,
            "confidence": (
                version_identity["confidence"] if consistent else None
            ),
            "document_id": str(chunk.document_id),
            "document_version_id": str(chunk.document_version_id),
        }
    return identities


def language_identity_scope_diagnostics(
    identities_by_chunk: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        {
            "chunk_id": str(chunk_id),
            "document_id": identity.get("document_id"),
            "document_version_id": identity.get("document_version_id"),
            "language": identity.get("language"),
            "valid": bool(identity.get("valid")),
            "known": bool(identity.get("known")),
            "detection_hash": identity.get("detection_hash"),
            "reason": identity.get("reason"),
        }
        for chunk_id, identity in sorted(identities_by_chunk.items())
    ]
    language_counts = Counter(
        str(identity.get("language"))
        for identity in identities_by_chunk.values()
        if identity.get("known") and identity.get("language")
    )
    return {
        "protocol_version": LANGUAGE_IDENTITY_SCOPE_PROTOCOL_VERSION,
        "language_detection_protocol_version": LANGUAGE_DETECTION_PROTOCOL_VERSION,
        "scope_hash": _canonical_hash(
            {
                "protocol_version": LANGUAGE_IDENTITY_SCOPE_PROTOCOL_VERSION,
                "rows": rows,
            }
        ),
        "chunk_count": len(rows),
        "valid_identity_count": sum(1 for row in rows if row["valid"]),
        "known_language_count": sum(1 for row in rows if row["known"]),
        "unknown_language_count": sum(1 for row in rows if row["valid"] and not row["known"]),
        "invalid_identity_count": sum(1 for row in rows if not row["valid"]),
        "language_counts": dict(sorted(language_counts.items())),
        "invalid_reasons": dict(
            sorted(
                Counter(
                    str(row["reason"])
                    for row in rows
                    if not row["valid"]
                ).items()
            )
        ),
    }
