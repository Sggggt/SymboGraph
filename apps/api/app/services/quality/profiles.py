from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Course, QualityProfile
from app.services.quality.signals import build_quality_signals, tokenize
from app.services.strategy_profiles import get_active_profile_record, profile_schema


QUALITY_PROFILE_SCHEMA_VERSION = "quality_profile_v1"


def _chunk_bucket(chunk: Chunk) -> tuple[str, str]:
    metadata = chunk.metadata_json or {}
    return (str(metadata.get("content_kind") or chunk.source_type or "unknown"), str(chunk.chapter or "unknown"))


def stratified_quality_sample(chunks: list[Chunk], limit: int = 32) -> list[Chunk]:
    if len(chunks) <= limit:
        return chunks[:]
    buckets: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        buckets[_chunk_bucket(chunk)].append(chunk)
    selected: list[Chunk] = []
    seen: set[str] = set()
    for bucket_chunks in buckets.values():
        ranked = sorted(bucket_chunks, key=lambda item: (len(item.content or ""), item.id))
        for candidate in (ranked[0], ranked[len(ranked) // 2], ranked[-1]):
            if candidate.id not in seen:
                selected.append(candidate)
                seen.add(candidate.id)
            if len(selected) >= limit:
                return selected
    for chunk in sorted(chunks, key=lambda item: len(item.content or ""), reverse=True):
        if chunk.id not in seen:
            selected.append(chunk)
            seen.add(chunk.id)
        if len(selected) >= limit:
            break
    return selected


def build_domain_quality_profile_payload(
    course: Course,
    chunks: list[Chunk],
    *,
    sample_limit: int = 32,
    strategy_profile: Any | None = None,
) -> dict[str, Any]:
    strategy_profile_json = getattr(strategy_profile, "profile_json", None) or {}
    strategy_schema = profile_schema(strategy_profile_json)
    samples = stratified_quality_sample(chunks, limit=sample_limit)
    token_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    positive_examples: list[dict[str, Any]] = []
    negative_examples: list[dict[str, Any]] = []
    for chunk in samples:
        metadata = chunk.metadata_json or {}
        signals = build_quality_signals(
            target_type="chunk",
            text=chunk.content or "",
            title=chunk.section,
            section=chunk.section,
            content_kind=metadata.get("content_kind"),
            metadata=metadata,
            course_id=chunk.course_id,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            chunk_id=chunk.id,
            version=QUALITY_PROFILE_SCHEMA_VERSION,
        )
        token_counter.update(token for token in tokenize(signals.normalized_text) if len(token) >= 5)
        role_counter.update(signals.structural_role.roles)
        item = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content_kind": metadata.get("content_kind"),
            "chapter": chunk.chapter,
            "section": chunk.section,
            "snippet": (chunk.snippet or chunk.content or "")[:280],
            "signals": signals.model_dump(),
        }
        if signals.text_quality.toc_like or signals.text_quality.mojibake_ratio > 0.01 or signals.structural_role.structural_score >= 0.7:
            negative_examples.append(item)
        elif signals.semantic_density.definition_score or signals.semantic_density.term_density >= 0.2:
            positive_examples.append(item)

    profile = {
        "schema_version": QUALITY_PROFILE_SCHEMA_VERSION,
        "course_id": course.id,
        "course_name": course.name,
        "strategy_profile_id": getattr(strategy_profile, "id", None),
        "strategy_profile_name": getattr(strategy_profile, "name", None),
        "strategy_profile_hash": getattr(strategy_profile, "profile_hash", None),
        "sample_chunk_ids": [chunk.id for chunk in samples],
        "common_terms": [term for term, _count in token_counter.most_common(40)],
        "structural_noise_types": [role for role, _count in role_counter.most_common(20)],
        "positive_examples": positive_examples[:8],
        "negative_examples": negative_examples[:8],
        "entity_type_hints": strategy_schema["entity_types"],
        "relation_schema_hints": strategy_schema["relation_types"],
    }
    profile["profile_hash"] = hashlib.sha256(repr(profile).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return profile


def get_active_quality_profile(db: Session, course_id: str) -> QualityProfile | None:
    return db.scalar(
        select(QualityProfile)
        .where(QualityProfile.course_id == course_id, QualityProfile.is_active.is_(True))
        .order_by(QualityProfile.created_at.desc())
    )


def rebuild_domain_quality_profile(db: Session, course_id: str, *, sample_limit: int = 32) -> QualityProfile:
    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"Course not found: {course_id}")
    chunks = db.scalars(select(Chunk).where(Chunk.course_id == course_id, Chunk.is_active.is_(True)).order_by(Chunk.created_at.asc())).all()
    strategy_profile = get_active_profile_record(db, course_id)
    payload = build_domain_quality_profile_payload(course, chunks, sample_limit=sample_limit, strategy_profile=strategy_profile)
    for profile in db.scalars(select(QualityProfile).where(QualityProfile.course_id == course_id, QualityProfile.is_active.is_(True))).all():
        profile.is_active = False
    version = f"{QUALITY_PROFILE_SCHEMA_VERSION}:{payload['profile_hash']}"
    profile = QualityProfile(course_id=course_id, version=version, profile_json=payload, sample_chunk_ids=payload["sample_chunk_ids"], is_active=True)
    db.add(profile)
    db.flush()
    return profile
