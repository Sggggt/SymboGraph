from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Chunk,
    ChunkRelationEdge,
    ChunkRelationGraphState,
    ChunkStructureEdge,
    ChunkStructureMapping,
    ChunkStructureNode,
    CoarseConcept,
    CoarseConceptDefinition,
    CoarseConceptEdge,
    CoarseConceptMembership,
    CoarseConceptState,
    Document,
    DocumentVersion,
    MidConcept,
    MidConceptDefinition,
    MidConceptEdge,
    MidConceptMembership,
    MidConceptState,
    PolicyState,
    RQPrefix,
    RQPrefixDiagnostic,
    RQPrefixMembership,
    RQPrefixPairDiagnostic,
)


CANONICAL_GRAPH_JSON_PROTOCOL_VERSION = "canonical_graph_business_facts_json_v1"
CHUNK_BUSINESS_KEY_PROTOCOL_VERSION = "chunk_business_key_v1"
STRUCTURE_STATE_HASH_PROTOCOL_VERSION = "chunk_structure_state_hash_v2"
STRUCTURE_MAPPING_FACT_DIGEST_PROTOCOL_VERSION = (
    "chunk_structure_mapping_business_fact_digest_v1"
)
STRUCTURE_MAPPING_MULTISET_HASH_PROTOCOL_VERSION = (
    "chunk_structure_mapping_fact_digest_multiset_v1"
)
STRUCTURE_MAPPING_QUERY_BATCH_SIZE = 512
STRUCTURE_MAPPING_SORT_RUN_SIZE = 2048
STRUCTURE_MAPPING_MERGE_FAN_IN = 32
RELATION_EDGE_FACT_HASH_PROTOCOL_VERSION = "chunk_relation_edge_facts_hash_v1"
RQ_STATE_HASH_PROTOCOL_VERSION = "rq_codebook_prefix_membership_role_state_hash_v1"
RQ_MEMBERSHIP_BUSINESS_FACT_HASH_PROTOCOL_VERSION = (
    "rq_membership_business_fact_hash_v1"
)
RQ_PAIR_AGGREGATE_HASH_PROTOCOL_VERSION = "rq_prefix_pair_aggregate_state_hash_v1"
RELATION_STATE_HASH_PROTOCOL_VERSION = "chunk_relation_state_hash_v1"
MID_STATE_HASH_PROTOCOL_VERSION = "mid_concept_state_hash_v2"
COARSE_STATE_HASH_PROTOCOL_VERSION = "coarse_concept_state_hash_v2"
PROVIDER_PROJECTION_BUSINESS_AUDIT_PROTOCOL_VERSION = (
    "provider_projection_business_audit_v1"
)
CONTEXT_STATE_HASH_PROTOCOL_VERSION = "four_layer_context_graph_state_hash_v1"
POLICY_STATE_HASH_PROTOCOL_VERSION = "policy_state_business_facts_hash_v1"


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
_DROP = object()
_EPHEMERAL_KEYS = {
    "id",
    "created_at",
    "updated_at",
    "checked_at",
    "started_at",
    "completed_at",
    "graph_state_hash",
    "raw_output",
    "provider",
    "provider_response",
    "provider_status",
    "provider_model",
    "provider_endpoint",
    "authorization",
    "api_key",
    "system_prompt",
    "user_prompt",
    "prompt_text",
    "last_error",
    "error_message",
    "traceback",
    # These legacy hashes were computed over UUID-bearing payloads. The
    # canonical aggregate below independently hashes the persisted facts.
    "membership_fact_hash",
    "prefix_fact_hash",
    "canonical_membership_fact_hash",
    "canonical_membership_fact_hash_protocol_version",
    # Persisted integrity checks retain these UUID-derived hashes so the
    # referenced rows can be audited in place.  They are address checks, not
    # business facts: the canonical cards independently bind the corresponding
    # chunk/edge business keys and full support facts.
    "support_chunk_ids_sample_hash",
    "support_chunk_ids_hash",
    "support_chunk_edge_ids_hash",
    "internal_chunk_edge_ids_hash",
    "cross_chunk_edge_ids_hash",
    "support_ids_hash",
    # Bounded raw-address samples remain available in diagnostics, while the
    # canonical edge fact independently binds complete bottom-edge business
    # support, predicate counts, and rollup decisions.
    "semantic_uncertain_support_edge_ids",
    "rq_boundary_support_edge_ids",
}
_UNORDERED_LIST_KEYS = {
    "aliases",
    "aliases_json",
    "display_terms_json",
    "inclusion_criteria_json",
    "exclusion_criteria_json",
    "safe_arms",
    "matched_flags",
    "candidate_channels",
    "all_candidate_channels",
    "support_chunk_ids",
    "support_chunk_ids_json",
    "support_chunk_ids_sample_json",
    "support_chunk_edge_ids",
    "support_chunk_edge_ids_json",
    "support_relation_edge_ids_json",
    "support_rq_prefix_ids_json",
    "support_rq_prefix_node_ids_json",
    "support_mid_concept_ids_json",
    "support_mid_edge_ids_json",
    "representative_chunk_ids_json",
    "bridge_chunk_ids_json",
    "boundary_chunk_ids_json",
    "core_chunk_ids_json",
    "outlier_chunk_ids_json",
    "included_mid_concept_ids_json",
    "boundary_mid_concept_ids_json",
    "bridge_mid_concept_ids_json",
    "outlier_mid_concept_ids_json",
    "child_rq_l3_prefix_ids_json",
    "child_rq_l3_prefix_ids",
    "low_confidence_mid_concept_ids",
    "cross_community_weak_ties",
    "cross_community_weak_ties_json",
}
_DATABASE_ADDRESS_FIELD_NAMES = {
    "chunk",
    "coarse_concept",
    "document",
    "mid_concept",
    "next_sibling",
    "parent",
    "parent_rq_l1_prefix",
    "previous_sibling",
    "rq_prefix",
    "source",
    "structure_node",
    "support_rq_l2_prefix",
    "target",
}
_DATABASE_ADDRESS_COLLECTION_SUFFIXES = (
    "_chunks",
    "_concepts",
    "_edges",
    "_nodes",
    "_prefixes",
)


def _is_database_address_field(value: str | None) -> bool:
    key = str(value or "").strip().lower()
    if not key:
        return False
    if key in _DATABASE_ADDRESS_FIELD_NAMES:
        return True
    if re.search(r"(?:^|_)(?:id|ids)(?:_json)?$", key):
        return True
    return key.endswith(_DATABASE_ADDRESS_COLLECTION_SUFFIXES)


def _normalize_number(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Canonical graph facts reject non-finite numbers")
    return 0.0 if result == 0.0 else result


def _canonical_value(
    value: Any,
    *,
    references: Mapping[str, str] | None = None,
    parent_key: str | None = None,
) -> Any:
    references = references or {}
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            if value in references and _is_database_address_field(parent_key):
                return references[value]
            if (
                _UUID_RE.fullmatch(value)
                and _is_database_address_field(parent_key)
            ):
                return _DROP
        return value
    if isinstance(value, float):
        return _normalize_number(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _EPHEMERAL_KEYS or lowered.endswith("_timestamp"):
                continue
            if key in references:
                key = references[key]
            elif (
                _UUID_RE.fullmatch(key)
                and _is_database_address_field(parent_key)
            ):
                continue
            item = _canonical_value(
                value[raw_key],
                references=references,
                parent_key=key,
            )
            if item is _DROP:
                continue
            if (
                (lowered.endswith("_id") or lowered.endswith("_ids"))
                and not item
            ):
                continue
            normalized[key] = item
        return normalized
    if isinstance(value, (set, frozenset)):
        items = [
            item
            for raw in value
            if (
                item := _canonical_value(
                    raw,
                    references=references,
                    parent_key=parent_key,
                )
            )
            is not _DROP
        ]
        return _sort_canonical(items)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [
            item
            for raw in value
            if (
                item := _canonical_value(
                    raw,
                    references=references,
                    parent_key=parent_key,
                )
            )
            is not _DROP
        ]
        if str(parent_key or "").lower() in _UNORDERED_LIST_KEYS:
            return _sort_canonical(items)
        return items
    raise TypeError(
        "Canonical graph facts accept only JSON primitives; "
        f"received {type(value).__name__}"
    )


def _provider_projection_business_audit(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Provider projection audit must be a mapping in graph business facts"
        )
    business_identity_card = value.get("business_identity_card")
    if not isinstance(business_identity_card, Mapping):
        raise ValueError(
            "Provider projection audit is missing its UUID-free business "
            "identity card"
        )
    required_strings = {
        "protocol_version": str(value.get("protocol_version") or "").strip(),
        "full_packet_business_hash": str(
            value.get("full_packet_business_hash") or ""
        ).strip(),
        "full_packet_business_hash_protocol_version": str(
            value.get("full_packet_business_hash_protocol_version") or ""
        ).strip(),
        "business_identity_card_hash": str(
            value.get("business_identity_card_hash") or ""
        ).strip(),
    }
    if any(not item for item in required_strings.values()):
        raise ValueError(
            "Provider projection graph business audit requires protocol, "
            "business packet hash and business identity hash"
        )
    authority = value.get("provider_authority")
    if not isinstance(authority, Mapping):
        raise ValueError(
            "Provider projection graph business audit requires authority bounds"
        )
    return {
        "business_audit_protocol_version": (
            PROVIDER_PROJECTION_BUSINESS_AUDIT_PROTOCOL_VERSION
        ),
        **required_strings,
        "business_identity_card": _business_fact_projection(
            business_identity_card
        ),
        "provider_authority": _business_fact_projection(authority),
    }


def _provider_identity_proposal_business_audit(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Provider identity proposal audit must be a mapping in graph "
            "business facts"
        )
    protocol_version = str(value.get("protocol_version") or "").strip()
    decision = str(value.get("decision") or "").strip()
    support_authority = value.get("support_authority")
    representative_authority = value.get("representative_authority")
    if (
        not protocol_version
        or decision != "ignore_provider_identity_proposals"
        or support_authority is not False
        or representative_authority is not False
    ):
        raise ValueError(
            "Provider identity proposal business audit must preserve the "
            "fixed deny-authority decision"
        )
    # Counts and hashes intentionally remain only in the persisted address
    # audit: provider-proposed UUID lists are non-authoritative and must not
    # rotate the concept business/state hash.
    return {
        "protocol_version": protocol_version,
        "support_authority": False,
        "representative_authority": False,
        "decision": decision,
    }


def _business_fact_projection(value: Any) -> Any:
    """Remove database-address projection audit fields from business facts.

    The complete persisted audit remains untouched.  Only the graph state hash
    view replaces `provider_projection_audit` with its explicitly UUID-free
    business subset, so consecutive same-fact rebuilds cannot inherit newly
    allocated Mid/Coarse database addresses.
    """

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key == "provider_projection_audit":
                projected[key] = _provider_projection_business_audit(raw_value)
            elif key == "provider_identity_proposal_audit":
                projected[key] = _provider_identity_proposal_business_audit(
                    raw_value
                )
            elif key in {
                "summary_grounded_reason",
                "definition_grounded_reason",
            }:
                # These fields explain whether the accepted final prose came
                # directly from the provider candidate or from the local
                # deterministic repair.  They are retained in PostgreSQL for
                # audit, but that execution path is not a final graph fact.
                continue
            elif key in {
                "summary_grounded_audit",
                "definition_grounded_audit",
            }:
                projected[key] = _grounded_audit_business_projection(
                    raw_value
                )
            else:
                projected[key] = _business_fact_projection(raw_value)
        return projected
    if isinstance(value, (set, frozenset)):
        return [_business_fact_projection(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_business_fact_projection(item) for item in value]
    return value


def _grounded_audit_business_projection(value: Any) -> dict[str, Any]:
    """Project a grounding audit onto the accepted final evidence verdict.

    Semantic reuse intentionally starts from the previously accepted final
    prose because raw provider responses are never persisted.  Revalidating
    that prose can therefore take a different candidate/repair route while
    producing the exact same final text and claim-level evidence verdict.
    The full route audit remains persisted, whereas the canonical graph hash
    binds only the final replayable evidence facts.
    """

    if not isinstance(value, Mapping):
        raise ValueError("Grounding business audit must be a mapping")
    allowed = {
        "protocol_version",
        "support_score_threshold",
        "claim_count",
        "grounded_claim_count",
        "ungrounded_claim_count",
        "grounded_rate",
        "summary_grounded",
        "claim_audits",
        "grounded_claim_texts",
        "ungrounded_claim_texts",
        "model_call_count",
        "final_text_hash",
        "final_replay_required",
    }
    projected = {
        str(key): _business_fact_projection(raw_value)
        for key, raw_value in value.items()
        if str(key) in allowed
    }
    required = {
        "protocol_version",
        "claim_count",
        "grounded_claim_count",
        "ungrounded_claim_count",
        "grounded_rate",
        "summary_grounded",
        "claim_audits",
        "grounded_claim_texts",
        "ungrounded_claim_texts",
        "model_call_count",
        "final_text_hash",
        "final_replay_required",
    }
    missing = sorted(required - set(projected))
    if missing:
        raise ValueError(
            "Grounding business audit is missing final replay fields: "
            + ", ".join(missing)
        )
    return projected


_CONCEPT_STATE_OPERATIONAL_STAT_KEYS = frozenset(
    {
        "llm_batches",
        "provider_semantic_reuse_hit_count",
        "provider_semantic_reuse_miss_count",
        "provider_request_count",
        "provider_semantic_reuse_required",
        "provider_request_budget",
        "max_live_packets",
        "concept_i18n_enabled",
        "concept_i18n_translated_count",
        "edge_i18n_translated_count",
    }
)


def _concept_state_stats_business_projection(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep graph measurements while excluding build execution telemetry."""

    return {
        str(key): _business_fact_projection(raw_value)
        for key, raw_value in dict(value or {}).items()
        if str(key) not in _CONCEPT_STATE_OPERATIONAL_STAT_KEYS
    }


def _canonical_bytes(value: Any) -> bytes:
    normalized = _canonical_value(value)
    if normalized is _DROP:
        raise ValueError("Canonical graph root cannot be a database UUID")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sort_canonical(values: Iterable[Any]) -> list[Any]:
    return sorted(values, key=_canonical_bytes)


def canonical_graph_hash(protocol_version: str, payload: Any) -> str:
    envelope = {
        "canonical_protocol_version": CANONICAL_GRAPH_JSON_PROTOCOL_VERSION,
        "protocol_version": str(protocol_version),
        "payload": payload,
    }
    return hashlib.sha256(_canonical_bytes(envelope)).hexdigest()


def canonical_fact_set_hash(protocol_version: str, facts: Iterable[Any]) -> str:
    return canonical_graph_hash(protocol_version, _sort_canonical(list(facts)))


@dataclass(frozen=True)
class StreamingFactMultisetHash:
    state_hash: str
    fact_count: int
    max_buffered_digests: int
    initial_run_count: int
    max_open_runs: int


def _write_digest_run(path: Path, digests: Iterable[str]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for digest in digests:
            stream.write(digest)
            stream.write("\n")


def _digest_lines(stream: Any) -> Iterator[str]:
    for line in stream:
        digest = line.rstrip("\n")
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise RuntimeError("Canonical fact digest run is corrupt")
        yield digest


def _merge_digest_runs(
    source_paths: Sequence[Path],
    target_path: Path,
) -> None:
    with ExitStack() as stack:
        sources = [
            stack.enter_context(path.open("r", encoding="ascii", newline=""))
            for path in source_paths
        ]
        _write_digest_run(
            target_path,
            heapq.merge(*(_digest_lines(source) for source in sources)),
        )


def _streaming_multiset_envelope_hash(
    *,
    protocol_version: str,
    fact_protocol_version: str,
    fact_count: int,
    sorted_digests: Iterable[str],
) -> str:
    """Hash the canonical multiset envelope without materializing its digest list."""

    protocol_json = _canonical_bytes(str(protocol_version))
    fact_protocol_json = _canonical_bytes(str(fact_protocol_version))
    hasher = hashlib.sha256()
    hasher.update(b'{"canonical_protocol_version":')
    hasher.update(_canonical_bytes(CANONICAL_GRAPH_JSON_PROTOCOL_VERSION))
    hasher.update(b',"payload":{"canonical_fact_digests":[')
    observed_count = 0
    for digest in sorted_digests:
        if observed_count:
            hasher.update(b",")
        hasher.update(_canonical_bytes(digest))
        observed_count += 1
    if observed_count != fact_count:
        raise RuntimeError(
            "Canonical fact digest count changed during external merge: "
            f"expected={fact_count}, observed={observed_count}"
        )
    hasher.update(b'],"fact_count":')
    hasher.update(str(fact_count).encode("ascii"))
    hasher.update(b',"fact_protocol_version":')
    hasher.update(fact_protocol_json)
    hasher.update(b'},"protocol_version":')
    hasher.update(protocol_json)
    hasher.update(b"}")
    return hasher.hexdigest()


def streaming_canonical_fact_digest_multiset_hash(
    *,
    protocol_version: str,
    fact_protocol_version: str,
    facts: Iterable[Any],
    sort_run_size: int = STRUCTURE_MAPPING_SORT_RUN_SIZE,
    merge_fan_in: int = STRUCTURE_MAPPING_MERGE_FAN_IN,
) -> StreamingFactMultisetHash:
    """Hash every canonical fact through bounded sorted runs, retaining multiplicity."""

    if sort_run_size < 1:
        raise ValueError("sort_run_size must be positive")
    if merge_fan_in < 2:
        raise ValueError("merge_fan_in must be at least two")

    fact_count = 0
    max_buffered_digests = 0
    max_open_runs = 0
    with tempfile.TemporaryDirectory(prefix="symbograph-canonical-multiset-") as temp:
        temp_path = Path(temp)
        run_paths: list[Path] = []
        buffer: list[str] = []

        def flush_buffer() -> None:
            if not buffer:
                return
            buffer.sort()
            path = temp_path / f"run-0-{len(run_paths):08d}.txt"
            _write_digest_run(path, buffer)
            run_paths.append(path)
            buffer.clear()

        for fact in facts:
            buffer.append(canonical_graph_hash(fact_protocol_version, fact))
            fact_count += 1
            max_buffered_digests = max(max_buffered_digests, len(buffer))
            if len(buffer) >= sort_run_size:
                flush_buffer()
        flush_buffer()
        initial_run_count = len(run_paths)

        generation = 1
        while len(run_paths) > merge_fan_in:
            next_paths: list[Path] = []
            for start in range(0, len(run_paths), merge_fan_in):
                group = run_paths[start : start + merge_fan_in]
                max_open_runs = max(max_open_runs, len(group))
                target = temp_path / f"run-{generation}-{len(next_paths):08d}.txt"
                _merge_digest_runs(group, target)
                next_paths.append(target)
                for path in group:
                    path.unlink()
            run_paths = next_paths
            generation += 1

        if not run_paths:
            sorted_digests: Iterable[str] = ()
            return StreamingFactMultisetHash(
                state_hash=_streaming_multiset_envelope_hash(
                    protocol_version=protocol_version,
                    fact_protocol_version=fact_protocol_version,
                    fact_count=0,
                    sorted_digests=sorted_digests,
                ),
                fact_count=0,
                max_buffered_digests=0,
                initial_run_count=0,
                max_open_runs=0,
            )

        max_open_runs = max(max_open_runs, len(run_paths))
        with ExitStack() as stack:
            sources = [
                stack.enter_context(path.open("r", encoding="ascii", newline=""))
                for path in run_paths
            ]
            state_hash = _streaming_multiset_envelope_hash(
                protocol_version=protocol_version,
                fact_protocol_version=fact_protocol_version,
                fact_count=fact_count,
                sorted_digests=heapq.merge(
                    *(_digest_lines(source) for source in sources)
                ),
            )
        return StreamingFactMultisetHash(
            state_hash=state_hash,
            fact_count=fact_count,
            max_buffered_digests=max_buffered_digests,
            initial_run_count=initial_run_count,
            max_open_runs=max_open_runs,
        )


def _finalize_state_card(protocol_version: str, card: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonical_value(dict(card))
    if normalized is _DROP or not isinstance(normalized, dict):
        raise ValueError("Canonical state card must be an object")
    payload = {
        "protocol_version": str(protocol_version),
        **normalized,
    }
    return {
        **payload,
        "state_hash": canonical_graph_hash(protocol_version, payload),
    }


def verify_state_hash_card(card: Mapping[str, Any]) -> bool:
    payload = dict(card or {})
    stored_hash = str(payload.pop("state_hash", ""))
    protocol_version = str(payload.get("protocol_version") or "")
    return bool(
        protocol_version
        and len(stored_hash) == 64
        and stored_hash == canonical_graph_hash(protocol_version, payload)
    )


def _document_business_fact(document: Document, version: DocumentVersion) -> dict[str, Any]:
    return {
        "source_path": str(document.source_path or ""),
        "source_type": str(document.source_type or ""),
        "title": str(document.title or ""),
        "document_checksum": str(document.checksum or ""),
        "document_version": int(version.version),
        "document_version_checksum": str(version.checksum or ""),
        "parse_protocol_version": str(version.parse_protocol_version or ""),
        "language": str(version.language or document.language or ""),
        "language_source": str(
            version.language_source or document.language_source or ""
        ),
        "language_detection_protocol_version": str(
            version.language_detection_protocol_version
            or document.language_detection_protocol_version
            or ""
        ),
        "language_detection_hash": str(
            version.language_detection_hash or document.language_detection_hash or ""
        ),
    }


def _chunk_protocol_descriptor(chunk: Chunk) -> dict[str, Any]:
    metadata = dict(chunk.metadata_json or {})
    descriptor = metadata.get("chunk_protocol_descriptor")
    descriptor = dict(descriptor) if isinstance(descriptor, dict) else {}
    return {
        "chunk_schema_version": str(
            descriptor.get("chunk_schema_version")
            or metadata.get("chunk_schema_version")
            or "missing"
        ),
        "tokenizer_version": str(
            descriptor.get("tokenizer_version")
            or metadata.get("tokenizer_version")
            or "missing"
        ),
        "chunk_size": descriptor.get("chunk_size", metadata.get("chunk_size")),
        "chunk_overlap": descriptor.get(
            "chunk_overlap", metadata.get("chunk_overlap")
        ),
    }


@dataclass(frozen=True)
class ChunkBusinessReferences:
    key_by_id: dict[str, str]
    fact_by_id: dict[str, dict[str, Any]]
    document_version_key_by_id: dict[str, str]
    scope_hash: str


def chunk_business_references(
    db: Session,
    chunks: Sequence[Chunk],
) -> ChunkBusinessReferences:
    document_ids = {str(chunk.document_id) for chunk in chunks}
    version_ids = {str(chunk.document_version_id) for chunk in chunks}
    documents = {
        str(row.id): row
        for row in db.scalars(select(Document).where(Document.id.in_(document_ids))).all()
    }
    versions = {
        str(row.id): row
        for row in db.scalars(
            select(DocumentVersion).where(DocumentVersion.id.in_(version_ids))
        ).all()
    }
    if len(documents) != len(document_ids) or len(versions) != len(version_ids):
        raise RuntimeError("Canonical chunk business facts require complete document provenance")

    document_version_key_by_id: dict[str, str] = {}
    key_by_id: dict[str, str] = {}
    fact_by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        document = documents[str(chunk.document_id)]
        version = versions[str(chunk.document_version_id)]
        document_fact = _document_business_fact(document, version)
        document_key = canonical_graph_hash("document_business_key_v1", document_fact)
        document_version_key_by_id[str(version.id)] = document_key
        raw_text_hash = hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()
        fact = {
            "protocol_version": CHUNK_BUSINESS_KEY_PROTOCOL_VERSION,
            "document": document_fact,
            "chunk_version": int(chunk.chunk_version),
            "chunk_index": int(chunk.chunk_index),
            "token_span": [int(chunk.token_start), int(chunk.token_end)],
            "char_span": [int(chunk.char_start), int(chunk.char_end)],
            "section_path": str(chunk.section_path or ""),
            "page_range": [chunk.page_start, chunk.page_end],
            "stored_text_hash": str(chunk.text_hash or ""),
            "raw_text_hash": raw_text_hash,
            "chunk_protocol_descriptor": _chunk_protocol_descriptor(chunk),
        }
        key = canonical_graph_hash(CHUNK_BUSINESS_KEY_PROTOCOL_VERSION, fact)
        key_by_id[str(chunk.id)] = key
        fact_by_id[str(chunk.id)] = fact
    scope_hash = canonical_fact_set_hash(
        "chunk_business_scope_hash_v1",
        fact_by_id.values(),
    )
    return ChunkBusinessReferences(
        key_by_id=key_by_id,
        fact_by_id=fact_by_id,
        document_version_key_by_id=document_version_key_by_id,
        scope_hash=scope_hash,
    )


def _reference_value(value: str | None, references: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    normalized = references.get(str(value))
    if normalized is None:
        raise RuntimeError(f"Canonical graph fact has an unresolved database reference: {value}")
    return normalized


def _structure_mapping_business_facts(
    db: Session,
    *,
    chunk_keys: Mapping[str, str],
    chunk_document_version_by_id: Mapping[str, str],
    node_keys: Mapping[str, str],
    node_document_version_by_id: Mapping[str, str],
    batch_size: int = STRUCTURE_MAPPING_QUERY_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Project and keyset-page all mapping facts without loading mapping entities."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunk_ids = list(chunk_keys)
    if not chunk_ids:
        return

    all_references = {**chunk_keys, **node_keys}
    last_mapping_id: str | None = None
    while True:
        statement = select(
            ChunkStructureMapping.id.label("mapping_id"),
            ChunkStructureMapping.chunk_id,
            ChunkStructureMapping.structure_node_id,
            ChunkStructureMapping.document_version_id,
            ChunkStructureMapping.overlap_chars,
            ChunkStructureMapping.overlap_tokens,
            ChunkStructureMapping.coverage_ratio,
            ChunkStructureMapping.span_overlap,
            ChunkStructureMapping.bbox_iou,
            ChunkStructureMapping.path_match,
            ChunkStructureMapping.mapping_weight,
            ChunkStructureMapping.mapping_protocol_version,
            ChunkStructureMapping.bbox_intersection_json,
            ChunkStructureMapping.mapping_role,
            ChunkStructureMapping.metadata_json,
        ).where(ChunkStructureMapping.chunk_id.in_(chunk_ids))
        if last_mapping_id is not None:
            statement = statement.where(ChunkStructureMapping.id > last_mapping_id)
        statement = (
            statement.order_by(ChunkStructureMapping.id)
            .limit(batch_size)
            .execution_options(yield_per=batch_size, stream_results=True)
        )

        page_count = 0
        for row in db.execute(statement):
            page_count += 1
            mapping_id = str(row.mapping_id)
            chunk_id = str(row.chunk_id)
            node_id = str(row.structure_node_id)
            mapping_document_version_id = str(row.document_version_id)
            chunk_document_version_id = chunk_document_version_by_id.get(chunk_id)
            node_document_version_id = node_document_version_by_id.get(node_id)
            if (
                chunk_document_version_id is None
                or node_document_version_id is None
            ):
                raise RuntimeError(
                    "Structure mapping references a missing active chunk or structure node"
                )
            if (
                mapping_document_version_id != chunk_document_version_id
                or mapping_document_version_id != node_document_version_id
            ):
                raise RuntimeError(
                    "Structure mapping document-version provenance does not match its "
                    "chunk and structure node"
                )
            mapping_protocol_version = str(row.mapping_protocol_version or "")
            if not mapping_protocol_version.strip():
                raise RuntimeError(
                    "Structure mapping is missing mapping protocol version"
                )

            yield {
                "chunk": _reference_value(chunk_id, chunk_keys),
                "structure_node": _reference_value(node_id, node_keys),
                "overlap_chars": int(row.overlap_chars or 0),
                "overlap_tokens": int(row.overlap_tokens or 0),
                "coverage_ratio": float(row.coverage_ratio or 0.0),
                "span_overlap": float(row.span_overlap or 0.0),
                "bbox_iou": row.bbox_iou,
                "path_match": row.path_match,
                "mapping_weight": float(row.mapping_weight or 0.0),
                "mapping_protocol_version": mapping_protocol_version,
                "bbox_intersection": row.bbox_intersection_json or {},
                "mapping_role": str(row.mapping_role or ""),
                "metadata": _canonical_value(
                    row.metadata_json or {}, references=all_references
                ),
            }
            last_mapping_id = mapping_id
        if page_count < batch_size:
            return


def build_structure_state_hash_card(
    db: Session,
    chunks: Sequence[Chunk],
    *,
    chunk_references: ChunkBusinessReferences | None = None,
) -> dict[str, Any]:
    refs = chunk_references or chunk_business_references(db, chunks)
    version_ids = sorted({str(chunk.document_version_id) for chunk in chunks})
    nodes = list(
        db.scalars(
            select(ChunkStructureNode).where(
                ChunkStructureNode.document_version_id.in_(version_ids)
            )
        ).all()
    )
    edges = list(
        db.scalars(
            select(ChunkStructureEdge).where(
                ChunkStructureEdge.document_version_id.in_(version_ids)
            )
        ).all()
    )
    node_base_by_id: dict[str, dict[str, Any]] = {}
    node_key_by_id: dict[str, str] = {}
    node_document_version_by_id: dict[str, str] = {}
    for node in nodes:
        document_key = refs.document_version_key_by_id.get(
            str(node.document_version_id)
        )
        if document_key is None:
            raise RuntimeError(
                "Structure node is outside the active document-version scope"
            )
        base = {
            "document": document_key,
            "node_type": str(node.node_type or ""),
            "depth": int(node.depth or 0),
            "title": str(node.title or ""),
            "char_span": [node.char_start, node.char_end],
            "page_number": node.page_number,
            "bbox": node.bbox_json or {},
            "layout": node.layout_json or {},
            "path": str(node.path or ""),
        }
        node_base_by_id[str(node.id)] = base
        node_document_version_by_id[str(node.id)] = str(node.document_version_id)
        node_key_by_id[str(node.id)] = canonical_graph_hash(
            "chunk_structure_node_business_key_v1", base
        )

    node_facts = [
        {
            **node_base_by_id[str(node.id)],
            "parent": _reference_value(node.parent_id, node_key_by_id),
            "previous_sibling": _reference_value(
                node.previous_sibling_id, node_key_by_id
            ),
            "next_sibling": _reference_value(node.next_sibling_id, node_key_by_id),
        }
        for node in nodes
    ]
    edge_facts = [
        {
            "source": _reference_value(edge.source_node_id, node_key_by_id),
            "target": _reference_value(edge.target_node_id, node_key_by_id),
            "edge_type": str(edge.edge_type or ""),
            "weight": float(edge.weight or 0.0),
            "confidence": float(edge.confidence or 0.0),
            "metadata": _canonical_value(
                edge.metadata_json or {}, references=node_key_by_id
            ),
        }
        for edge in edges
    ]
    chunk_document_version_by_id = {
        str(chunk.id): str(chunk.document_version_id) for chunk in chunks
    }
    expected_mapping_count = int(
        db.scalar(
            select(func.count(ChunkStructureMapping.id)).where(
                ChunkStructureMapping.chunk_id.in_(list(refs.key_by_id))
            )
        )
        or 0
    )
    mapping_hash = streaming_canonical_fact_digest_multiset_hash(
        protocol_version=STRUCTURE_MAPPING_MULTISET_HASH_PROTOCOL_VERSION,
        fact_protocol_version=STRUCTURE_MAPPING_FACT_DIGEST_PROTOCOL_VERSION,
        facts=_structure_mapping_business_facts(
            db,
            chunk_keys=refs.key_by_id,
            chunk_document_version_by_id=chunk_document_version_by_id,
            node_keys=node_key_by_id,
            node_document_version_by_id=node_document_version_by_id,
        ),
    )
    if mapping_hash.fact_count != expected_mapping_count:
        raise RuntimeError(
            "Structure mapping scope changed while computing its canonical hash: "
            f"expected={expected_mapping_count}, observed={mapping_hash.fact_count}"
        )
    component_hashes = {
        "nodes": canonical_fact_set_hash("chunk_structure_node_facts_v1", node_facts),
        "edges": canonical_fact_set_hash("chunk_structure_edge_facts_v1", edge_facts),
        "mappings": mapping_hash.state_hash,
    }
    return _finalize_state_card(
        STRUCTURE_STATE_HASH_PROTOCOL_VERSION,
        {
            "chunk_business_scope_hash": refs.scope_hash,
            "component_hashes": component_hashes,
            "component_hash_protocols": {
                "nodes": "chunk_structure_node_facts_v1",
                "edges": "chunk_structure_edge_facts_v1",
                "mapping_fact_digest": STRUCTURE_MAPPING_FACT_DIGEST_PROTOCOL_VERSION,
                "mappings": STRUCTURE_MAPPING_MULTISET_HASH_PROTOCOL_VERSION,
            },
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "mappings": mapping_hash.fact_count,
            },
        },
    )


def _relation_edge_maps(
    edges: Sequence[ChunkRelationEdge],
    chunk_keys: Mapping[str, str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    edge_key_by_id: dict[str, str] = {}
    identity_by_id: dict[str, dict[str, Any]] = {}
    for edge in edges:
        identity = {
            "source": _reference_value(edge.source_chunk_id, chunk_keys),
            "target": _reference_value(edge.target_chunk_id, chunk_keys),
            "edge_type": str(edge.edge_type or ""),
        }
        identity_by_id[str(edge.id)] = identity
        edge_key_by_id[str(edge.id)] = canonical_graph_hash(
            "chunk_relation_edge_business_key_v1", identity
        )
    references = {**chunk_keys, **edge_key_by_id}
    facts = []
    for edge in edges:
        facts.append(
            {
                **identity_by_id[str(edge.id)],
                "weight": float(edge.weight or 0.0),
                "distance": float(edge.distance or 0.0),
                "raw_strength": float(edge.raw_strength or 0.0),
                "raw_strength_summary": _canonical_value(
                    edge.raw_strength_summary_json or {}, references=references
                ),
                "normalization_stats": _canonical_value(
                    edge.normalization_stats_json or {}, references=references
                ),
                "confidence": float(edge.confidence or 0.0),
                "features": _canonical_value(
                    edge.features_json or {}, references=references
                ),
                "support": _canonical_value(
                    edge.support_json or {}, references=references
                ),
                "source_algorithm": str(edge.source_algorithm or ""),
                "protocol_version": str(edge.protocol_version or ""),
                "edge_distance_protocol_hash": str(
                    edge.edge_distance_protocol_hash or ""
                ),
                "source_language": str(edge.source_language or ""),
                "target_language": str(edge.target_language or ""),
                "is_cross_document": bool(edge.is_cross_document),
                "is_cross_language": bool(edge.is_cross_language),
                "bridge_quota_reason": str(edge.bridge_quota_reason or ""),
                "is_bridge": bool(edge.is_bridge),
                "diagnostics": _canonical_value(
                    edge.diagnostics_json or {}, references=references
                ),
            }
        )
    return edge_key_by_id, facts


def relation_edge_business_keys(
    edges: Sequence[ChunkRelationEdge],
    chunk_keys: Mapping[str, str],
) -> dict[str, str]:
    """Return UUID-free keys for persisted bottom-edge address references."""

    edge_keys, _facts = _relation_edge_maps(edges, chunk_keys)
    return edge_keys


def rq_membership_business_fact(
    membership: RQPrefixMembership,
    *,
    chunk_keys: Mapping[str, str],
    prefix_keys: Mapping[str, str],
    edge_keys: Mapping[str, str],
) -> dict[str, Any]:
    """Canonical persisted RQ membership fact without row/address hashes."""

    references = {**chunk_keys, **edge_keys, **prefix_keys}
    return {
        "rq_prefix_key": _reference_value(membership.rq_prefix_id, prefix_keys),
        "chunk": _reference_value(membership.chunk_id, chunk_keys),
        "membership_score": float(membership.membership_score or 0.0),
        "membership_role": str(membership.membership_role or ""),
        "membership_reason": str(membership.membership_reason or ""),
        "membership_entropy": membership.membership_entropy,
        "rq_path": [int(item) for item in (membership.rq_path or [])],
        "residual_norm": membership.residual_norm,
        "rank": int(membership.rank or 0),
        "support_chunk_edges": _canonical_value(
            membership.support_chunk_edge_ids_json or [],
            references=edge_keys,
            parent_key="support_chunk_edge_ids_json",
        ),
        "diagnostics": _canonical_value(
            membership.diagnostics_json or {}, references=references
        ),
    }


def rq_membership_business_fact_hash(
    membership: RQPrefixMembership,
    *,
    chunk_keys: Mapping[str, str],
    prefix_keys: Mapping[str, str],
    edge_keys: Mapping[str, str],
) -> str:
    return canonical_graph_hash(
        RQ_MEMBERSHIP_BUSINESS_FACT_HASH_PROTOCOL_VERSION,
        rq_membership_business_fact(
            membership,
            chunk_keys=chunk_keys,
            prefix_keys=prefix_keys,
            edge_keys=edge_keys,
        ),
    )


def _rq_codebook_fact(
    relation_state: ChunkRelationGraphState,
    chunk_keys: Mapping[str, str],
) -> dict[str, Any]:
    raw = dict((relation_state.diagnostics_json or {}).get("rq_kmeans") or {})
    path_by_chunk = raw.pop("path_by_chunk", {})
    encoding_by_chunk = raw.pop("encoding_hash_by_chunk", {})
    if isinstance(path_by_chunk, dict):
        raw["path_by_chunk_business_key"] = {
            chunk_keys[str(chunk_id)]: path
            for chunk_id, path in path_by_chunk.items()
            if str(chunk_id) in chunk_keys
        }
    if isinstance(encoding_by_chunk, dict):
        raw["encoding_hash_by_chunk_business_key"] = {
            chunk_keys[str(chunk_id)]: value
            for chunk_id, value in encoding_by_chunk.items()
            if str(chunk_id) in chunk_keys
        }
    # membership_hash was previously UUID-bearing.  It is replaced by the
    # independently recomputed canonical membership aggregate below.
    raw.pop("membership_hash", None)
    return _canonical_value(raw, references=chunk_keys)


def build_relation_state_hash_card(
    db: Session,
    relation_state: ChunkRelationGraphState,
    chunks: Sequence[Chunk],
    *,
    protocol_identities: Mapping[str, Any],
    vector_identity: Mapping[str, Any],
    chunk_references: ChunkBusinessReferences | None = None,
) -> dict[str, Any]:
    refs = chunk_references or chunk_business_references(db, chunks)
    edges = list(
        db.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id
            )
        ).all()
    )
    edge_key_by_id, edge_facts = _relation_edge_maps(edges, refs.key_by_id)
    prefixes = list(
        db.scalars(
            select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    prefix_key_by_id = {
        str(prefix.id): str(prefix.rq_prefix_key or "") for prefix in prefixes
    }
    all_references = {**refs.key_by_id, **edge_key_by_id, **prefix_key_by_id}
    prefix_facts = [
        {
            "rq_prefix_key": str(prefix.rq_prefix_key or ""),
            "label": str(prefix.label or ""),
            "node_type": str(prefix.node_type or ""),
            "centroid": prefix.centroid_json or [],
            "rq_level": int(prefix.rq_level or 0),
            "rq_path_prefix": [int(item) for item in (prefix.rq_path_prefix or [])],
            "parent_rq_prefix_key": _reference_value(
                prefix.parent_rq_prefix_id, prefix_key_by_id
            ),
            "codebook_version": str(prefix.codebook_version or ""),
            "representative_chunks": _canonical_value(
                prefix.representative_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="representative_chunk_ids_json",
            ),
            "support_chunks": _canonical_value(
                prefix.support_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_json",
            ),
            "bridge_chunks": _canonical_value(
                prefix.bridge_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="bridge_chunk_ids_json",
            ),
            "stats": _canonical_value(prefix.stats_json or {}),
            "diagnostics": _canonical_value(
                prefix.diagnostics_json or {}, references=all_references
            ),
            "state": str(prefix.state or ""),
        }
        for prefix in prefixes
    ]
    memberships = list(
        db.scalars(
            select(RQPrefixMembership)
            .join(RQPrefix, RQPrefixMembership.rq_prefix_id == RQPrefix.id)
            .where(RQPrefix.graph_state_id == relation_state.id)
        ).all()
    )
    membership_facts: list[dict[str, Any]] = []
    for membership in memberships:
        fact = rq_membership_business_fact(
            membership,
            chunk_keys=refs.key_by_id,
            prefix_keys=prefix_key_by_id,
            edge_keys=edge_key_by_id,
        )
        membership_facts.append(fact)
        membership.diagnostics_json = {
            **(membership.diagnostics_json or {}),
            "canonical_membership_fact_hash_protocol_version": (
                RQ_MEMBERSHIP_BUSINESS_FACT_HASH_PROTOCOL_VERSION
            ),
            "canonical_membership_fact_hash": canonical_graph_hash(
                RQ_MEMBERSHIP_BUSINESS_FACT_HASH_PROTOCOL_VERSION, fact
            ),
        }

    diagnostic_rows = list(
        db.scalars(
            select(RQPrefixDiagnostic).where(
                RQPrefixDiagnostic.graph_state_id == relation_state.id
            )
        ).all()
    )
    diagnostic_facts = [
        {
            "rq_prefix_key": _reference_value(
                row.rq_prefix_id, prefix_key_by_id
            ),
            "diagnostic_type": str(row.diagnostic_type or ""),
            "diagnostic_strength": float(row.diagnostic_strength or 0.0),
            "support_membership_mass": float(row.support_membership_mass or 0.0),
            "support_chunk_sample": _canonical_value(
                row.support_chunk_ids_sample_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_sample_json",
            ),
            "protocol_version": str(row.protocol_version or ""),
            "diagnostics": _canonical_value(
                row.diagnostics_json or {}, references=all_references
            ),
        }
        for row in diagnostic_rows
    ]
    pair_rows = list(
        db.scalars(
            select(RQPrefixPairDiagnostic).where(
                RQPrefixPairDiagnostic.graph_state_id == relation_state.id
            )
        ).all()
    )
    pair_facts = [
        {
            "source_rq_prefix_key": _reference_value(
                row.source_rq_prefix_id, prefix_key_by_id
            ),
            "target_rq_prefix_key": _reference_value(
                row.target_rq_prefix_id, prefix_key_by_id
            ),
            "edge_type": str(row.edge_type or ""),
            "diagnostic_strength": float(row.diagnostic_strength or 0.0),
            "support_membership_mass": float(row.support_membership_mass or 0.0),
            "support_chunk_sample": _canonical_value(
                row.support_chunk_ids_sample_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_sample_json",
            ),
            "support_chunk_edges": _canonical_value(
                row.support_chunk_edge_ids_json or [],
                references=edge_key_by_id,
                parent_key="support_chunk_edge_ids_json",
            ),
            "source_algorithm": str(row.source_algorithm or ""),
            "protocol_version": str(row.protocol_version or ""),
            "diagnostic_hash": str(row.diagnostic_hash or ""),
            "diagnostics": _canonical_value(
                row.diagnostics_json or {}, references=all_references
            ),
        }
        for row in pair_rows
    ]

    edge_facts_hash = canonical_fact_set_hash(
        RELATION_EDGE_FACT_HASH_PROTOCOL_VERSION, edge_facts
    )
    prefix_facts_hash = canonical_fact_set_hash("rq_prefix_business_facts_v1", prefix_facts)
    membership_facts_hash = canonical_fact_set_hash(
        "rq_membership_business_facts_v1", membership_facts
    )
    diagnostic_facts_hash = canonical_fact_set_hash(
        "rq_prefix_diagnostic_business_facts_v1", diagnostic_facts
    )
    codebook_hash = canonical_graph_hash(
        "rq_codebook_business_facts_v1",
        _rq_codebook_fact(relation_state, refs.key_by_id),
    )
    rq_state_card = _finalize_state_card(
        RQ_STATE_HASH_PROTOCOL_VERSION,
        {
            "codebook_hash": codebook_hash,
            "prefix_facts_hash": prefix_facts_hash,
            "membership_facts_hash": membership_facts_hash,
            "diagnostic_facts_hash": diagnostic_facts_hash,
            "counts": {
                "prefixes": len(prefixes),
                "memberships": len(memberships),
                "diagnostics": len(diagnostic_rows),
            },
            "protocol_identities": _canonical_value(dict(protocol_identities)),
        },
    )
    pair_state_card = _finalize_state_card(
        RQ_PAIR_AGGREGATE_HASH_PROTOCOL_VERSION,
        {
            "pair_facts_hash": canonical_fact_set_hash(
                "rq_prefix_pair_business_facts_v1", pair_facts
            ),
            "count": len(pair_rows),
            "protocol_identity": _canonical_value(
                {
                    key: value
                    for key, value in protocol_identities.items()
                    if "pair" in str(key)
                }
            ),
        },
    )
    relation_diagnostics = dict(relation_state.diagnostics_json or {})
    operating_point_card = {
        "graph_operating_point": relation_state.graph_operating_point_json or {},
        "graph_operating_point_hash": str(
            relation_state.graph_operating_point_hash or ""
        ),
        "edge_distance_protocol_hash": str(
            relation_state.edge_distance_protocol_hash or ""
        ),
        "edge_type_calibration_protocol_hash": str(
            relation_state.edge_type_calibration_protocol_hash or ""
        ),
        "calibration_params_hash": str(
            relation_diagnostics.get("calibration_params_hash") or ""
        ),
        "edge_type_calibration_config_hash": str(
            relation_diagnostics.get("edge_type_calibration_config_hash") or ""
        ),
        "auto_tpe": _canonical_value(
            relation_diagnostics.get("auto_tpe") or {},
            references=all_references,
        ),
    }
    operating_point_hash = canonical_graph_hash(
        "graph_operating_point_tpe_calibration_state_v1", operating_point_card
    )
    return _finalize_state_card(
        RELATION_STATE_HASH_PROTOCOL_VERSION,
        {
            "chunk_business_scope_hash": refs.scope_hash,
            "contextual_index_hash": str(
                relation_diagnostics.get("contextual_index_business_hash")
                or ""
            ),
            "edge_facts_hash": edge_facts_hash,
            "edge_stats": _canonical_value(relation_state.stats_json or {}),
            "rq_state_hash": rq_state_card["state_hash"],
            "rq_pair_aggregate_hash": pair_state_card["state_hash"],
            "operating_point_hash": operating_point_hash,
            "language_identity_scope_hash": str(
                (relation_diagnostics.get("language_identity") or {}).get(
                    "scope_hash"
                )
                or ""
            ),
            "protocol_identities": _canonical_value(dict(protocol_identities)),
            "vector_identity": _canonical_value(dict(vector_identity)),
            "counts": {
                "edges": len(edges),
                "prefixes": len(prefixes),
                "memberships": len(memberships),
                "rq_diagnostics": len(diagnostic_rows),
                "rq_pair_diagnostics": len(pair_rows),
            },
            "component_hashes": {
                "edge_facts": edge_facts_hash,
                "rq": rq_state_card["state_hash"],
                "rq_pair": pair_state_card["state_hash"],
                "operating_point": operating_point_hash,
            },
            "rq_state_card": rq_state_card,
            "rq_pair_state_card": pair_state_card,
        },
    )


def _concept_i18n_fact(audit: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict((audit or {}).get("concept_i18n") or {})
    payload.pop("id", None)
    return _canonical_value(payload)


def _edge_i18n_fact(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict((diagnostics or {}).get("edge_i18n") or {})
    payload.pop("id", None)
    return _canonical_value(payload)


def _mid_concept_key_maps(
    concepts: Sequence[MidConcept],
    prefix_keys: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for concept in concepts:
        base = {
            "support_rq_l3_prefix": _reference_value(
                concept.support_rq_l3_prefix_id, prefix_keys
            ),
            "canonical_label": str(concept.canonical_label or ""),
        }
        result[str(concept.id)] = canonical_graph_hash(
            "mid_concept_business_key_v1", base
        )
    return result


def build_mid_state_hash_card(
    db: Session,
    mid_state: MidConceptState,
    relation_state: ChunkRelationGraphState,
    chunks: Sequence[Chunk],
    *,
    relation_state_hash: str,
    profile_hash: str,
    prompt_protocol_hash: str,
    protocol_identities: Mapping[str, Any],
    chunk_references: ChunkBusinessReferences | None = None,
    relation_edges_override: Sequence[ChunkRelationEdge] | None = None,
    prefixes_override: Sequence[RQPrefix] | None = None,
    concepts_override: Sequence[MidConcept] | None = None,
    edges_override: Sequence[MidConceptEdge] | None = None,
) -> dict[str, Any]:
    refs = chunk_references or chunk_business_references(db, chunks)
    relation_edges = (
        list(relation_edges_override)
        if relation_edges_override is not None
        else list(
            db.scalars(
                select(ChunkRelationEdge).where(
                    ChunkRelationEdge.graph_state_id == relation_state.id
                )
            ).all()
        )
    )
    relation_edge_keys, _ = _relation_edge_maps(relation_edges, refs.key_by_id)
    prefixes = (
        list(prefixes_override)
        if prefixes_override is not None
        else list(
            db.scalars(
                select(RQPrefix).where(
                    RQPrefix.graph_state_id == relation_state.id
                )
            ).all()
        )
    )
    prefix_keys = {
        str(prefix.id): str(prefix.rq_prefix_key or "") for prefix in prefixes
    }
    concepts = (
        list(concepts_override)
        if concepts_override is not None
        else list(
            db.scalars(
                select(MidConcept).where(
                    MidConcept.concept_state_id == mid_state.id
                )
            ).all()
        )
    )
    concept_keys = _mid_concept_key_maps(concepts, prefix_keys)
    references = {
        **refs.key_by_id,
        **relation_edge_keys,
        **prefix_keys,
        **concept_keys,
    }
    concept_facts: list[dict[str, Any]] = []
    grounding_facts: list[dict[str, Any]] = []
    for concept in concepts:
        grounding_fact = {
            "support_rq_l3_prefix": _reference_value(
                concept.support_rq_l3_prefix_id, prefix_keys
            ),
            "support_rq_prefixes": _canonical_value(
                concept.support_rq_prefix_ids_json or [],
                references=prefix_keys,
                parent_key="support_rq_prefix_ids_json",
            ),
            "support_chunks": _canonical_value(
                concept.support_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_json",
            ),
            "support_chunk_edges": _canonical_value(
                concept.support_chunk_edge_ids_json or [],
                references=relation_edge_keys,
                parent_key="support_chunk_edge_ids_json",
            ),
        }
        grounding_hash = canonical_graph_hash(
            "mid_concept_grounding_business_facts_v1", grounding_fact
        )
        concept.grounding_hash = grounding_hash
        grounding_facts.append(grounding_fact)
        concept_facts.append(
            {
                "concept_key": concept_keys[str(concept.id)],
                "canonical_label": str(concept.canonical_label or ""),
                "aliases": _sort_canonical(list(concept.aliases_json or [])),
                "support_rq_l3_prefix": grounding_fact["support_rq_l3_prefix"],
                "parent_rq_l2_prefix": _reference_value(
                    concept.parent_rq_l2_prefix_id, prefix_keys
                ),
                "parent_rq_l1_prefix": _reference_value(
                    concept.parent_rq_l1_prefix_id, prefix_keys
                ),
                "definition": str(concept.definition or ""),
                "summary": str(concept.summary or ""),
                "scope_note": str(concept.scope_note or ""),
                "inclusion_criteria": _sort_canonical(
                    list(concept.inclusion_criteria_json or [])
                ),
                "exclusion_criteria": _sort_canonical(
                    list(concept.exclusion_criteria_json or [])
                ),
                "display_terms": _sort_canonical(
                    list(concept.display_terms_json or [])
                ),
                "internal_state": _canonical_value(
                    _business_fact_projection(
                        concept.internal_state_json or {}
                    ),
                    references=references,
                ),
                "representative_chunks": _canonical_value(
                    concept.representative_chunk_ids_json or [],
                    references=refs.key_by_id,
                    parent_key="representative_chunk_ids_json",
                ),
                "grounding": grounding_fact,
                "core_chunks": _canonical_value(
                    concept.core_chunk_ids_json or [],
                    references=refs.key_by_id,
                    parent_key="core_chunk_ids_json",
                ),
                "boundary_chunks": _canonical_value(
                    concept.boundary_chunk_ids_json or [],
                    references=refs.key_by_id,
                    parent_key="boundary_chunk_ids_json",
                ),
                "bridge_chunks": _canonical_value(
                    concept.bridge_chunk_ids_json or [],
                    references=refs.key_by_id,
                    parent_key="bridge_chunk_ids_json",
                ),
                "outlier_chunks": _canonical_value(
                    concept.outlier_chunk_ids_json or [],
                    references=refs.key_by_id,
                    parent_key="outlier_chunk_ids_json",
                ),
                "raw_node_weight": float(concept.raw_node_weight or 0.0),
                "node_weight": float(concept.node_weight or 0.0),
                "node_weight_normalization_scope": str(
                    concept.node_weight_normalization_scope or ""
                ),
                "node_weight_diagnostics": _canonical_value(
                    concept.node_weight_diagnostics_json or {}, references=references
                ),
                "confidence": float(concept.confidence or 0.0),
                "grounding_hash": grounding_hash,
                "i18n": _concept_i18n_fact(concept.llm_audit_json),
                "state": str(concept.state or ""),
            }
        )

    memberships = list(
        db.scalars(
            select(MidConceptMembership)
            .join(MidConcept, MidConceptMembership.mid_concept_id == MidConcept.id)
            .where(MidConcept.concept_state_id == mid_state.id)
        ).all()
    )
    membership_facts = [
        {
            "mid_concept": _reference_value(row.mid_concept_id, concept_keys),
            "rq_prefix": _reference_value(row.rq_prefix_id, prefix_keys),
            "membership_score": float(row.membership_score or 0.0),
            "support_chunks": _canonical_value(
                row.support_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_json",
            ),
            "diagnostics": _canonical_value(
                row.diagnostics_json or {}, references=references
            ),
        }
        for row in memberships
    ]
    edges = (
        list(edges_override)
        if edges_override is not None
        else list(
            db.scalars(
                select(MidConceptEdge).where(
                    MidConceptEdge.concept_state_id == mid_state.id
                )
            ).all()
        )
    )
    edge_facts = [
        {
            "source": _reference_value(edge.source_concept_id, concept_keys),
            "target": _reference_value(edge.target_concept_id, concept_keys),
            "edge_type": str(edge.edge_type or ""),
            "weight": float(edge.weight or 0.0),
            "distance": float(edge.distance or 0.0),
            "projected_distance_raw": float(edge.projected_distance_raw or 0.0),
            "projected_strength_raw": float(edge.projected_strength_raw or 0.0),
            "raw_strength_summary": _canonical_value(
                edge.raw_strength_summary_json or {}, references=references
            ),
            "projection_normalization_stats": _canonical_value(
                edge.projection_normalization_stats_json or {}, references=references
            ),
            "edge_projection_protocol_hash": str(
                edge.edge_projection_protocol_hash or ""
            ),
            "source_algorithm": str(edge.source_algorithm or ""),
            "protocol_version": str(edge.protocol_version or ""),
            "network_evidence_score": float(edge.network_evidence_score or 0.0),
            "support_rq_prefixes": _canonical_value(
                edge.support_rq_prefix_ids_json or [],
                references=prefix_keys,
                parent_key="support_rq_prefix_ids_json",
            ),
            "support_chunks": _canonical_value(
                edge.support_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_json",
            ),
            "support_chunk_edges": _canonical_value(
                edge.support_chunk_edge_ids_json or [],
                references=relation_edge_keys,
                parent_key="support_chunk_edge_ids_json",
            ),
            "support_relation_edges": _canonical_value(
                edge.support_relation_edge_ids_json or [],
                references=relation_edge_keys,
                parent_key="support_relation_edge_ids_json",
            ),
            "explanation": str(edge.explanation or ""),
            "diagnostics": _canonical_value(
                {
                    key: value
                    for key, value in dict(edge.diagnostics_json or {}).items()
                    if key != "edge_i18n"
                },
                references=references,
            ),
            "i18n": _edge_i18n_fact(edge.diagnostics_json),
        }
        for edge in edges
    ]
    definitions = list(
        db.scalars(
            select(MidConceptDefinition)
            .join(MidConcept, MidConceptDefinition.mid_concept_id == MidConcept.id)
            .where(MidConcept.concept_state_id == mid_state.id)
        ).all()
    )
    definition_facts = [
        {
            "mid_concept": _reference_value(row.mid_concept_id, concept_keys),
            "definition_version": str(row.definition_version or ""),
            "definition": _canonical_value(
                _business_fact_projection(row.definition_json or {}),
                references=references,
            ),
            "support_spans": _canonical_value(
                row.support_spans_json or [], references=references
            ),
        }
        for row in definitions
    ]
    component_hashes = {
        "concepts": canonical_fact_set_hash("mid_concept_business_facts_v2", concept_facts),
        "memberships": canonical_fact_set_hash(
            "mid_concept_membership_business_facts_v1", membership_facts
        ),
        "edges": canonical_fact_set_hash("mid_concept_edge_business_facts_v1", edge_facts),
        "definitions": canonical_fact_set_hash(
            "mid_concept_definition_business_facts_v2", definition_facts
        ),
    }
    grounding_hash = canonical_fact_set_hash(
        "mid_concept_grounding_state_v1", grounding_facts
    )
    mid_state.grounding_hash = grounding_hash
    card = _finalize_state_card(
        MID_STATE_HASH_PROTOCOL_VERSION,
        {
            "relation_state_hash": relation_state_hash,
            "grounding_hash": grounding_hash,
            "profile_hash": str(profile_hash),
            "prompt_protocol_hash": str(prompt_protocol_hash),
            "protocol_identities": _canonical_value(dict(protocol_identities)),
            "component_hashes": component_hashes,
            "stats": _canonical_value(
                _concept_state_stats_business_projection(
                    mid_state.stats_json
                )
            ),
            "counts": {
                "concepts": len(concepts),
                "memberships": len(memberships),
                "edges": len(edges),
                "definitions": len(definitions),
            },
        },
    )
    mid_state.state_hash = card["state_hash"]
    return card


def _coarse_concept_key_maps(
    concepts: Sequence[CoarseConcept],
    prefix_keys: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for concept in concepts:
        base = {
            "support_rq_l2_prefix": _reference_value(
                concept.support_rq_l2_prefix_id, prefix_keys
            ),
            "canonical_label": str(concept.canonical_label or ""),
        }
        result[str(concept.id)] = canonical_graph_hash(
            "coarse_concept_business_key_v1", base
        )
    return result


def build_coarse_state_hash_card(
    db: Session,
    coarse_state: CoarseConceptState,
    mid_state: MidConceptState,
    relation_state: ChunkRelationGraphState,
    chunks: Sequence[Chunk],
    *,
    mid_state_hash: str,
    profile_hash: str,
    prompt_protocol_hash: str,
    protocol_identities: Mapping[str, Any],
    chunk_references: ChunkBusinessReferences | None = None,
    relation_edges_override: Sequence[ChunkRelationEdge] | None = None,
    prefixes_override: Sequence[RQPrefix] | None = None,
    mid_concepts_override: Sequence[MidConcept] | None = None,
    mid_edges_override: Sequence[MidConceptEdge] | None = None,
    concepts_override: Sequence[CoarseConcept] | None = None,
    edges_override: Sequence[CoarseConceptEdge] | None = None,
) -> dict[str, Any]:
    refs = chunk_references or chunk_business_references(db, chunks)
    relation_edges = (
        list(relation_edges_override)
        if relation_edges_override is not None
        else list(
            db.scalars(
                select(ChunkRelationEdge).where(
                    ChunkRelationEdge.graph_state_id == relation_state.id
                )
            ).all()
        )
    )
    relation_edge_keys, _ = _relation_edge_maps(relation_edges, refs.key_by_id)
    prefixes = (
        list(prefixes_override)
        if prefixes_override is not None
        else list(
            db.scalars(
                select(RQPrefix).where(
                    RQPrefix.graph_state_id == relation_state.id
                )
            ).all()
        )
    )
    prefix_keys = {
        str(prefix.id): str(prefix.rq_prefix_key or "") for prefix in prefixes
    }
    mid_concepts = (
        list(mid_concepts_override)
        if mid_concepts_override is not None
        else list(
            db.scalars(
                select(MidConcept).where(
                    MidConcept.concept_state_id == mid_state.id
                )
            ).all()
        )
    )
    mid_keys = _mid_concept_key_maps(mid_concepts, prefix_keys)
    mid_edges = (
        list(mid_edges_override)
        if mid_edges_override is not None
        else list(
            db.scalars(
                select(MidConceptEdge).where(
                    MidConceptEdge.concept_state_id == mid_state.id
                )
            ).all()
        )
    )
    mid_edge_keys = {
        str(edge.id): canonical_graph_hash(
            "mid_concept_edge_business_key_v1",
            {
                "source": _reference_value(edge.source_concept_id, mid_keys),
                "target": _reference_value(edge.target_concept_id, mid_keys),
                "edge_type": str(edge.edge_type or ""),
            },
        )
        for edge in mid_edges
    }
    concepts = (
        list(concepts_override)
        if concepts_override is not None
        else list(
            db.scalars(
                select(CoarseConcept).where(
                    CoarseConcept.coarse_state_id == coarse_state.id
                )
            ).all()
        )
    )
    concept_keys = _coarse_concept_key_maps(concepts, prefix_keys)
    references = {
        **refs.key_by_id,
        **relation_edge_keys,
        **prefix_keys,
        **mid_keys,
        **mid_edge_keys,
        **concept_keys,
    }
    concept_facts: list[dict[str, Any]] = []
    grounding_facts: list[dict[str, Any]] = []
    for concept in concepts:
        grounding_fact = {
            "support_rq_l2_prefix": _reference_value(
                concept.support_rq_l2_prefix_id, prefix_keys
            ),
            "included_mid_concepts": _canonical_value(
                concept.included_mid_concept_ids_json or [],
                references=mid_keys,
                parent_key="included_mid_concept_ids_json",
            ),
            "boundary_mid_concepts": _canonical_value(
                concept.boundary_mid_concept_ids_json or [],
                references=mid_keys,
                parent_key="boundary_mid_concept_ids_json",
            ),
            "bridge_mid_concepts": _canonical_value(
                concept.bridge_mid_concept_ids_json or [],
                references=mid_keys,
                parent_key="bridge_mid_concept_ids_json",
            ),
            "outlier_mid_concepts": _canonical_value(
                concept.outlier_mid_concept_ids_json or [],
                references=mid_keys,
                parent_key="outlier_mid_concept_ids_json",
            ),
            "low_confidence_mid_concepts": _canonical_value(
                (
                    concept.internal_state_json or {}
                ).get("low_confidence_mid_concept_ids")
                or [],
                references=mid_keys,
                parent_key="low_confidence_mid_concept_ids",
            ),
            "support_chunks": _canonical_value(
                concept.support_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_json",
            ),
            "support_chunk_edges": _canonical_value(
                concept.support_chunk_edge_ids_json or [],
                references=relation_edge_keys,
                parent_key="support_chunk_edge_ids_json",
            ),
        }
        grounding_hash = canonical_graph_hash(
            "coarse_concept_grounding_business_facts_v1", grounding_fact
        )
        concept.grounding_hash = grounding_hash
        grounding_facts.append(grounding_fact)
        concept_facts.append(
            {
                "concept_key": concept_keys[str(concept.id)],
                "canonical_label": str(concept.canonical_label or ""),
                "aliases": _sort_canonical(list(concept.aliases_json or [])),
                "support_rq_l2_prefix": grounding_fact["support_rq_l2_prefix"],
                "parent_rq_l1_prefix": _reference_value(
                    concept.parent_rq_l1_prefix_id, prefix_keys
                ),
                "child_rq_l3_prefixes": _canonical_value(
                    concept.child_rq_l3_prefix_ids_json or [],
                    references=prefix_keys,
                    parent_key="child_rq_l3_prefix_ids_json",
                ),
                "definition": str(concept.definition or ""),
                "summary": str(concept.summary or ""),
                "scope_note": str(concept.scope_note or ""),
                "inclusion_criteria": _sort_canonical(
                    list(concept.inclusion_criteria_json or [])
                ),
                "exclusion_criteria": _sort_canonical(
                    list(concept.exclusion_criteria_json or [])
                ),
                "display_terms": _sort_canonical(
                    list(concept.display_terms_json or [])
                ),
                "internal_state": _canonical_value(
                    _business_fact_projection(
                        concept.internal_state_json or {}
                    ),
                    references=references,
                ),
                "grounding": grounding_fact,
                "boundary_mid_concepts": _canonical_value(
                    concept.boundary_mid_concept_ids_json or [],
                    references=mid_keys,
                    parent_key="boundary_mid_concept_ids_json",
                ),
                "bridge_mid_concepts": _canonical_value(
                    concept.bridge_mid_concept_ids_json or [],
                    references=mid_keys,
                    parent_key="bridge_mid_concept_ids_json",
                ),
                "outlier_mid_concepts": _canonical_value(
                    concept.outlier_mid_concept_ids_json or [],
                    references=mid_keys,
                    parent_key="outlier_mid_concept_ids_json",
                ),
                "cross_community_weak_ties": _canonical_value(
                    concept.cross_community_weak_ties_json or [],
                    references=references,
                ),
                "raw_node_weight": float(concept.raw_node_weight or 0.0),
                "node_weight": float(concept.node_weight or 0.0),
                "node_weight_normalization_scope": str(
                    concept.node_weight_normalization_scope or ""
                ),
                "node_weight_diagnostics": _canonical_value(
                    concept.node_weight_diagnostics_json or {}, references=references
                ),
                "confidence": float(concept.confidence or 0.0),
                "grounding_hash": grounding_hash,
                "i18n": _concept_i18n_fact(concept.llm_audit_json),
                "state": str(concept.state or ""),
            }
        )
    memberships = list(
        db.scalars(
            select(CoarseConceptMembership)
            .join(
                CoarseConcept,
                CoarseConceptMembership.coarse_concept_id == CoarseConcept.id,
            )
            .where(CoarseConcept.coarse_state_id == coarse_state.id)
        ).all()
    )
    membership_facts = [
        {
            "coarse_concept": _reference_value(row.coarse_concept_id, concept_keys),
            "mid_concept": _reference_value(row.mid_concept_id, mid_keys),
            "membership_score": float(row.membership_score or 0.0),
            "role": str(row.role or ""),
            "diagnostics": _canonical_value(
                row.diagnostics_json or {}, references=references
            ),
        }
        for row in memberships
    ]
    edges = (
        list(edges_override)
        if edges_override is not None
        else list(
            db.scalars(
                select(CoarseConceptEdge).where(
                    CoarseConceptEdge.coarse_state_id == coarse_state.id
                )
            ).all()
        )
    )
    edge_facts = [
        {
            "source": _reference_value(edge.source_concept_id, concept_keys),
            "target": _reference_value(edge.target_concept_id, concept_keys),
            "edge_type": str(edge.edge_type or ""),
            "weight": float(edge.weight or 0.0),
            "distance": float(edge.distance or 0.0),
            "projected_distance_raw": float(edge.projected_distance_raw or 0.0),
            "projected_strength_raw": float(edge.projected_strength_raw or 0.0),
            "raw_strength_summary": _canonical_value(
                edge.raw_strength_summary_json or {}, references=references
            ),
            "projection_normalization_stats": _canonical_value(
                edge.projection_normalization_stats_json or {}, references=references
            ),
            "edge_projection_protocol_hash": str(
                edge.edge_projection_protocol_hash or ""
            ),
            "source_algorithm": str(edge.source_algorithm or ""),
            "protocol_version": str(edge.protocol_version or ""),
            "support_rq_prefixes": _canonical_value(
                edge.support_rq_prefix_ids_json or [],
                references=prefix_keys,
                parent_key="support_rq_prefix_ids_json",
            ),
            "support_mid_concepts": _canonical_value(
                edge.support_mid_concept_ids_json or [],
                references=mid_keys,
                parent_key="support_mid_concept_ids_json",
            ),
            "support_mid_edges": _canonical_value(
                edge.support_mid_edge_ids_json or [],
                references=mid_edge_keys,
                parent_key="support_mid_edge_ids_json",
            ),
            "support_chunks": _canonical_value(
                edge.support_chunk_ids_json or [],
                references=refs.key_by_id,
                parent_key="support_chunk_ids_json",
            ),
            "support_chunk_edges": _canonical_value(
                edge.support_chunk_edge_ids_json or [],
                references=relation_edge_keys,
                parent_key="support_chunk_edge_ids_json",
            ),
            "cross_community_weak_ties": _canonical_value(
                edge.cross_community_weak_ties_json or [], references=references
            ),
            "explanation": str(edge.explanation or ""),
            "diagnostics": _canonical_value(
                {
                    key: value
                    for key, value in dict(edge.diagnostics_json or {}).items()
                    if key != "edge_i18n"
                },
                references=references,
            ),
            "i18n": _edge_i18n_fact(edge.diagnostics_json),
        }
        for edge in edges
    ]
    definitions = list(
        db.scalars(
            select(CoarseConceptDefinition)
            .join(
                CoarseConcept,
                CoarseConceptDefinition.coarse_concept_id == CoarseConcept.id,
            )
            .where(CoarseConcept.coarse_state_id == coarse_state.id)
        ).all()
    )
    definition_facts = [
        {
            "coarse_concept": _reference_value(row.coarse_concept_id, concept_keys),
            "definition_version": str(row.definition_version or ""),
            "definition": _canonical_value(
                _business_fact_projection(row.definition_json or {}),
                references=references,
            ),
            "support_spans": _canonical_value(
                row.support_spans_json or [], references=references
            ),
        }
        for row in definitions
    ]
    component_hashes = {
        "concepts": canonical_fact_set_hash(
            "coarse_concept_business_facts_v2", concept_facts
        ),
        "memberships": canonical_fact_set_hash(
            "coarse_concept_membership_business_facts_v1", membership_facts
        ),
        "edges": canonical_fact_set_hash(
            "coarse_concept_edge_business_facts_v1", edge_facts
        ),
        "definitions": canonical_fact_set_hash(
            "coarse_concept_definition_business_facts_v2", definition_facts
        ),
    }
    grounding_hash = canonical_fact_set_hash(
        "coarse_concept_grounding_state_v1", grounding_facts
    )
    coarse_state.grounding_hash = grounding_hash
    card = _finalize_state_card(
        COARSE_STATE_HASH_PROTOCOL_VERSION,
        {
            "mid_state_hash": mid_state_hash,
            "grounding_hash": grounding_hash,
            "profile_hash": str(profile_hash),
            "prompt_protocol_hash": str(prompt_protocol_hash),
            "protocol_identities": _canonical_value(dict(protocol_identities)),
            "component_hashes": component_hashes,
            "stats": _canonical_value(
                _concept_state_stats_business_projection(
                    coarse_state.stats_json
                )
            ),
            "counts": {
                "concepts": len(concepts),
                "memberships": len(memberships),
                "edges": len(edges),
                "definitions": len(definitions),
            },
        },
    )
    coarse_state.state_hash = card["state_hash"]
    return card


def build_context_state_hash_card(
    *,
    chunk_business_scope_hash: str,
    contextual_index_hash: str,
    structure_state_hash: str,
    relation_state_hash: str,
    rq_state_hash: str,
    rq_pair_aggregate_hash: str,
    mid_state_hash: str,
    coarse_state_hash: str,
    runtime_settings_hash: str,
    profile_hash: str,
    policy_state_hash: str | None,
    prompt_protocol_hash: str,
    agent_operating_envelope_hash: str,
    edge_distance_protocol_hash: str,
    edge_projection_protocol_hash: str,
    traversal_protocol_hash: str,
    graph_protocol_runtime_identity_hash: str,
    vector_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return _finalize_state_card(
        CONTEXT_STATE_HASH_PROTOCOL_VERSION,
        {
            "layer_hashes": {
                "chunk_business_scope": str(chunk_business_scope_hash),
                "contextual_index": str(contextual_index_hash),
                "structure": str(structure_state_hash),
                "relation": str(relation_state_hash),
                "rq": str(rq_state_hash),
                "rq_pair": str(rq_pair_aggregate_hash),
                "mid": str(mid_state_hash),
                "coarse": str(coarse_state_hash),
            },
            "runtime_profile_policy": {
                "runtime_settings_hash": str(runtime_settings_hash),
                "profile_hash": str(profile_hash),
                "policy_state_hash": str(policy_state_hash or "none"),
                "prompt_protocol_hash": str(prompt_protocol_hash),
                "agent_operating_envelope_hash": str(
                    agent_operating_envelope_hash
                ),
            },
            "protocol_hashes": {
                "edge_distance": str(edge_distance_protocol_hash),
                "edge_projection": str(edge_projection_protocol_hash),
                "traversal": str(traversal_protocol_hash),
                "graph_runtime_identity": str(
                    graph_protocol_runtime_identity_hash
                ),
            },
            "vector_identity": _canonical_value(dict(vector_identity)),
        },
    )


def canonical_policy_state_hash(
    *,
    policy_family: str,
    policy_version: str,
    profile_objective_hash: str | None,
    weights: Mapping[str, Any],
    constraints: Mapping[str, Any],
    exploration: Mapping[str, Any],
    reward_summary: Mapping[str, Any],
) -> str:
    summary = dict(reward_summary or {})
    summary.pop("last_reward_event_id", None)
    summary.pop("previous_policy_state_id", None)
    return canonical_graph_hash(
        POLICY_STATE_HASH_PROTOCOL_VERSION,
        {
            "policy_family": str(policy_family),
            "policy_version": str(policy_version),
            "profile_objective_hash": str(profile_objective_hash or ""),
            "weights": dict(weights or {}),
            "constraints": dict(constraints or {}),
            "exploration": dict(exploration or {}),
            "reward_summary": summary,
        },
    )


def canonical_policy_state_hash_for_row(row: PolicyState | None) -> str | None:
    if row is None:
        return None
    return canonical_policy_state_hash(
        policy_family=row.policy_family,
        policy_version=row.policy_version,
        profile_objective_hash=row.profile_objective_hash,
        weights=row.weights_json or {},
        constraints=row.constraints_json or {},
        exploration=row.exploration_json or {},
        reward_summary=row.reward_summary_json or {},
    )
