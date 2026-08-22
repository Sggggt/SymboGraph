from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    Chunk,
    ChunkRelationEdge,
    ChunkRelationGraphState,
    CoarseConceptEdge,
    CoarseConceptState,
    ContextPackage,
    Document,
    DocumentVersion,
    IngestionBatch,
    KnowledgeBase,
    MidConceptEdge,
    MidConceptState,
    RetrievalTrace,
)
from app.schemas import (
    CONTEXT_PACKAGE_PUBLIC_HASH_FIELDS,
    ContextCitationSpan,
    ContextConceptPathEntry,
    ContextGraphExpansionPath,
    ContextItem,
    ContextPackageChunk,
    ContextPackageDiagnostics,
    ContextPackageDocument,
    ContextPackageResponse,
    ContextSelectionReason,
    CycleDistanceRewardAudit,
    GrayRQScore,
    OrdinaryQueryPerceptionAudit,
    RetrievalEntryCandidateCard,
    RetrievalEntryParentRef,
    RetrievalFrontierSnapshot,
    RetrievalGranularity,
    RetrievalGrayZoneDecision,
    RetrievalNodeContributionSummary,
    RetrievalPathContribution,
    RetrievalPathLabel,
    RetrievalQueryRQSeedAudit,
    RetrievalRQChunkSeedCard,
    RetrievalSupportRefs,
    RetrievalTraversalState,
    SearchFilters,
)
from app.services.context_graph import (
    ENTRY_DENSE_REPLAY_PROTOCOL_VERSION,
    ENTRY_NEUTRAL_START_COST_PROTOCOL_VERSION,
    ENTRY_REPLAY_PROOF_PROTOCOL_VERSION,
    ENTRY_SELECTION_PROTOCOL_VERSION,
    ENTRY_TOPOLOGY_PROTOCOL_VERSION,
    EntrySelectionTraceInvariantError,
    build_context_package,
    context_graph_overview_stats,
    graph_layer_payload,
    layered_search,
    schedule_layered_retrieval_cache_write,
    entry_selection_protocol_hash,
    entry_topology_protocol_hash,
    validate_entry_candidate_card,
    validate_entry_dense_replay_against_database,
    validate_entry_selection_trace_audit,
    validate_entry_topology_replay_against_database,
    validate_gray_zone_decision_records_for_persistence,
    validate_semantic_entry_query_trace_packet,
)
from app.services import cache_manager
from app.services import context_graph as context_graph_service
from app.services.conversation_state import (
    CONVERSATION_PROMPT_HISTORY_PROTOCOL_VERSION,
    PROMPT_HISTORY_MAX_TOKENS,
    PROMPT_HISTORY_MAX_TURNS,
)
from app.services.embeddings import classify_json_with_budget, is_degraded_mode
from app.services.storage import run_bounded_source_io
from app.services.ingestion import list_knowledge_base_files
from app.services.strategy_profiles import (
    DEFAULT_QUERY_FACET_ALIAS_SUFFIX,
    DEFAULT_QUERY_FACET_BILINGUAL_SUFFIX,
    DEFAULT_QUERY_FACET_EXTRACTOR_SYSTEM,
    DEFAULT_QUESTION_PERCEPTION_SYSTEM,
    active_profile_hash,
    active_profile_json,
    profile_prompt,
    profile_prompt_template,
)


TERMINAL_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled", "cancel_failed"}
ORDINARY_QUERY_PERCEPTION_PROTOCOL_VERSION = (
    "bounded_query_perception_and_facet_proposal_v1"
)
ORDINARY_QUERY_PERCEPTION_MODEL_CALL_BUDGET = 2
ORDINARY_QUERY_REPLAY_POINTER_PROTOCOL_VERSION = (
    "ordinary_query_postgresql_facet_replay_v1"
)
ORDINARY_QUERY_REPLAY_POINTER_TTL_SECONDS = 300
ORDINARY_QUERY_INTENT_JSON_MAX_TOKENS = 4096
ORDINARY_QUERY_FACET_JSON_MAX_TOKENS = 4096
ORDINARY_QUERY_REPLAY_POINTER_FIELDS = frozenset(
    {
        "protocol_version",
        "pointer_key_protocol_version",
        "pointer_key_digest",
        "pointer_components",
        "knowledge_base_id",
        "source_retrieval_trace_id",
        "source_context_package_id",
        "query_facet_packet_hash",
        "query_perception_audit_hash",
        "full_cache_key",
        "full_redis_key_digest",
        "query_provider_protocol_hash",
        "ttl_seconds",
        "write_policy",
        "pointer_hash",
    }
)
ORDINARY_QUERY_INTENT_FIELDS = frozenset(
    {
        "intent",
        "entities",
        "sub_queries",
        "needs_graph",
        "suggested_strategy",
    }
)
ORDINARY_QUERY_INTENTS = frozenset(
    {
        "definition",
        "comparison",
        "application",
        "procedure",
        "analysis",
        "unknown",
    }
)
ORDINARY_QUERY_STRATEGIES = frozenset(
    {"global_dense", "local_graph", "hybrid", "community"}
)


class RetrievalTraceAuditError(RuntimeError):
    def __init__(self, audit: dict[str, Any]) -> None:
        self.audit = audit
        super().__init__(
            "retrieval gray-zone audit did not pass: "
            f"status={audit.get('status')}, conflicts={audit.get('conflict_count')}, "
            f"incomplete={audit.get('incomplete_record_count')}, "
            f"codes={[item.get('code') for item in (audit.get('conflicts') or audit.get('issues') or [])[:8]]}"
        )


class ContextPackagePublicIntegrityError(RuntimeError):
    """Raised when persisted package facts cannot produce one public proof."""


def _validated_frozen_trace_operating_envelope(
    trace: RetrievalTrace,
) -> dict[str, Any]:
    """Return one trace-frozen envelope only after all writer identities agree."""

    diagnostics = dict(trace.diagnostics_json or {})
    convergence = dict(trace.convergence_json or {})
    operating_envelope = dict(
        diagnostics.get("agent_operating_envelope") or {}
    )
    trace_envelope_hash = str(
        trace.agent_operating_envelope_hash or ""
    )
    if (
        len(trace_envelope_hash) != 64
        or str(
            diagnostics.get("agent_operating_envelope_hash") or ""
        )
        != trace_envelope_hash
        or str(
            convergence.get("agent_operating_envelope_hash") or ""
        )
        != trace_envelope_hash
        or context_graph_service.stable_hash(operating_envelope)
        != trace_envelope_hash
    ):
        raise ContextPackagePublicIntegrityError(
            "persisted cycle-reward envelope is not bound to the frozen "
            "retrieval trace identity"
        )
    return operating_envelope


def _gray_zone_protocol_from_persisted_trace(
    convergence: dict[str, Any], diagnostics: dict[str, Any]
) -> tuple[str | None, list[dict[str, Any]]]:
    candidates = [
        ("convergence.gray_zone_rule_protocol_version", convergence.get("gray_zone_rule_protocol_version")),
        ("convergence.gray_zone_protocol", convergence.get("gray_zone_protocol")),
        ("diagnostics.gray_zone_rule_protocol_version", diagnostics.get("gray_zone_rule_protocol_version")),
        ("diagnostics.gray_zone_protocol", diagnostics.get("gray_zone_protocol")),
    ]
    persisted = [(location, str(value).strip()) for location, value in candidates if str(value or "").strip()]
    values = sorted({value for _, value in persisted})
    if not values:
        return None, [
            {
                "code": "missing_persisted_gray_zone_protocol",
                "message": "trace does not persist the gray-zone rule protocol used by the request",
            }
        ]
    if len(values) > 1:
        return None, [
            {
                "code": "conflicting_persisted_gray_zone_protocol",
                "values": values,
                "locations": [location for location, _ in persisted],
            }
        ]
    return values[0], []


def validate_retrieval_gray_zone_trace(
    *,
    records: list[Any],
    convergence: dict[str, Any],
    diagnostics: dict[str, Any],
    traversal_protocol_hash: str | None,
    runtime_settings_hash: str | None,
    agent_operating_envelope_hash: str | None,
    operating_envelope: dict[str, Any],
) -> dict[str, Any]:
    """Validate raw persisted gray records without inferring missing audit facts."""

    issues: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    trace_hashes = {
        "traversal_protocol_hash": str(traversal_protocol_hash or "").strip(),
        "runtime_settings_hash": str(runtime_settings_hash or "").strip(),
        "agent_operating_envelope_hash": str(agent_operating_envelope_hash or "").strip(),
    }
    for field, value in trace_hashes.items():
        if not value:
            issues.append(
                {
                    "code": "missing_trace_identity_hash",
                    "field": field,
                    "message": "same-input determinism requires the request-frozen trace identity",
                }
            )

    persisted_protocol, protocol_issues = _gray_zone_protocol_from_persisted_trace(convergence, diagnostics)
    issues.extend(protocol_issues)
    validated_records: list[RetrievalGrayZoneDecision] = []
    checked_record_count = 0
    duplicate_reference_count = 0
    incomplete_record_count = 0

    for record_index, raw_record in enumerate(records):
        checked_record_count += 1
        if not isinstance(raw_record, dict):
            incomplete_record_count += 1
            issues.append(
                {
                    "code": "gray_zone_record_not_object",
                    "record_index": record_index,
                    "record_type": type(raw_record).__name__,
                }
            )
            continue
        try:
            validated_payload = validate_gray_zone_decision_records_for_persistence(
                [raw_record],
                traversal_hash=trace_hashes["traversal_protocol_hash"],
                runtime_settings_hash=trace_hashes["runtime_settings_hash"],
                operating_envelope_hash=trace_hashes["agent_operating_envelope_hash"],
                operating_envelope=operating_envelope,
            )[0]
            record = RetrievalGrayZoneDecision.model_validate(validated_payload)
        except (ValidationError, RuntimeError, ValueError) as exc:
            incomplete_record_count += 1
            validation_errors = (
                [
                    {
                        "location": ".".join(str(part) for part in error.get("loc") or []),
                        "message": str(error.get("msg") or "invalid persisted gray-zone audit field"),
                        "type": str(error.get("type") or "validation_error"),
                    }
                    for error in exc.errors(include_input=False, include_url=False)
                ]
                if isinstance(exc, ValidationError)
                else [
                    {
                        "location": "",
                        "message": str(exc),
                        "type": "gray_zone_audit_replay_error",
                    }
                ]
            )
            issues.append(
                {
                    "code": "gray_zone_record_incomplete",
                    "record_index": record_index,
                    "errors": validation_errors,
                }
            )
            continue

        identity_mismatches = {
            field: {"trace": trace_value, "record": getattr(record, field)}
            for field, trace_value in trace_hashes.items()
            if trace_value and getattr(record, field) != trace_value
        }
        if identity_mismatches:
            incomplete_record_count += 1
            issues.append(
                {
                    "code": "gray_zone_record_trace_identity_mismatch",
                    "record_index": record_index,
                    "mismatches": identity_mismatches,
                }
            )
            continue

        if record.decision_source == "deterministic_local_rule" and persisted_protocol:
            if record.protocol_version != persisted_protocol:
                incomplete_record_count += 1
                issues.append(
                    {
                        "code": "gray_zone_record_protocol_mismatch",
                        "record_index": record_index,
                        "trace_protocol": persisted_protocol,
                        "record_protocol": record.protocol_version,
                    }
                )
                continue

        validated_records.append(record)

    decision_hash_indices: dict[str, list[int]] = defaultdict(list)
    for record_index, record in enumerate(validated_records):
        decision_hash_indices[record.decision_hash].append(record_index)
    duplicate_groups = {
        decision_hash: indices
        for decision_hash, indices in decision_hash_indices.items()
        if len(indices) > 1
    }
    if duplicate_groups:
        duplicate_reference_count = sum(len(indices) - 1 for indices in duplicate_groups.values())
        incomplete_record_count += duplicate_reference_count
        issues.append(
            {
                "code": "duplicate_gray_zone_decision_event",
                "message": "raw retrieval steps must persist each gray-zone decision event exactly once",
                "duplicates": [
                    {"decision_hash": decision_hash, "record_indices": indices}
                    for decision_hash, indices in sorted(duplicate_groups.items())
                ],
            }
        )
    seen_decision_hashes: set[str] = set()
    unique_records: list[RetrievalGrayZoneDecision] = []
    for record in validated_records:
        if record.decision_hash in seen_decision_hashes:
            continue
        seen_decision_hashes.add(record.decision_hash)
        unique_records.append(record)
    local_records = [
        record for record in unique_records if record.decision_source == "deterministic_local_rule"
    ]
    partition_records = [
        record
        for record in unique_records
        if record.decision_source == "deterministic_distance_partition"
    ]
    red_records = [record for record in partition_records if record.distance_zone == "red"]
    hard_records = [record for record in partition_records if record.distance_zone == "hard_stop"]

    grouped_outcomes: dict[tuple[str, str, str], dict[tuple[str, str], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in local_records:
        key = (
            record.protocol_hash,
            record.runtime_settings_hash,
            record.input_hash,
        )
        grouped_outcomes[key][(record.decision, record.matched_rule)].append(record.decision_hash)
    for key, outcomes in grouped_outcomes.items():
        if len(outcomes) > 1:
            conflicts.append(
                {
                    "code": "same_input_outcome_conflict",
                    "group": {
                        "protocol_hash": key[0],
                        "runtime_settings_hash": key[1],
                        "input_hash": key[2],
                    },
                    "outcomes": [
                        {
                            "decision": outcome[0],
                            "matched_rule": outcome[1],
                            "decision_hashes": decision_hashes,
                        }
                        for outcome, decision_hashes in sorted(outcomes.items())
                    ],
                }
            )

    expected_counts = {
        "gray_zone_decision_count": len(local_records),
        "gray_zone_rule_evaluation_count": len(local_records),
        "red_zone_pruned_count": len(red_records),
        "hard_stop_pruned_count": len(hard_records),
    }
    for field, expected in expected_counts.items():
        if field not in convergence:
            issues.append(
                {
                    "code": "missing_gray_zone_convergence_count",
                    "field": field,
                    "expected": expected,
                }
            )
            continue
        value = convergence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            issues.append(
                {
                    "code": "gray_zone_convergence_count_mismatch",
                    "field": field,
                    "expected": expected,
                    "actual": value,
                }
            )
    if "gray_zone_model_call_count" not in convergence:
        issues.append(
            {
                "code": "missing_gray_zone_model_call_count",
                "message": "the persisted convergence audit must explicitly record zero model calls",
            }
        )
    elif convergence.get("gray_zone_model_call_count") != 0:
        conflicts.append(
            {
                "code": "nonzero_gray_zone_model_call_count",
                "actual": convergence.get("gray_zone_model_call_count"),
            }
        )

    status = "failed" if conflicts else ("incomplete" if issues else "passed")
    audit = {
        "status": status,
        "checked_record_count": checked_record_count,
        "unique_record_count": len(unique_records),
        "local_rule_record_count": len(local_records),
        "red_partition_record_count": len(red_records),
        "hard_stop_partition_record_count": len(hard_records),
        "duplicate_reference_count": duplicate_reference_count,
        "conflict_count": len(conflicts),
        "incomplete_record_count": incomplete_record_count,
        "conflicts": conflicts,
        "issues": issues,
        "persisted_protocol": persisted_protocol,
        "local_records": [record.model_dump(mode="json") for record in local_records],
        "partition_records": [record.model_dump(mode="json") for record in partition_records],
    }
    if status != "passed":
        raise RetrievalTraceAuditError(audit)
    return audit


def _ordinary_query_perception_protocol_hash(
    db: Session,
    knowledge_base_id: str,
) -> str:
    settings = get_settings()
    profile = active_profile_json(db, knowledge_base_id)
    bilingual_enabled = bool(settings.query_facet_bilingual_enabled)
    perception_system = profile_prompt_template(
        profile,
        "question_perception_system",
        DEFAULT_QUESTION_PERCEPTION_SYSTEM,
        {
            "perception_domain": profile_prompt(
                profile,
                "perception_domain",
                "context-graph-grounded knowledge-base search",
            ),
            "entity_label": profile_prompt(
                profile,
                "entity_label",
                "source-grounded concepts",
            ),
        },
    )
    facet_system = " ".join(
        part.strip()
        for part in (
            profile_prompt(
                profile,
                "query_facet_extractor_system",
                DEFAULT_QUERY_FACET_EXTRACTOR_SYSTEM,
            ),
            profile_prompt(
                profile,
                (
                    "query_facet_bilingual_suffix"
                    if bilingual_enabled
                    else "query_facet_alias_suffix"
                ),
                (
                    DEFAULT_QUERY_FACET_BILINGUAL_SUFFIX
                    if bilingual_enabled
                    else DEFAULT_QUERY_FACET_ALIAS_SUFFIX
                ),
            ),
        )
        if part and part.strip()
    )
    return context_graph_service.stable_hash(
        {
            "protocol_version": (
                ORDINARY_QUERY_PERCEPTION_PROTOCOL_VERSION
            ),
            "model_call_budget": (
                ORDINARY_QUERY_PERCEPTION_MODEL_CALL_BUDGET
            ),
            "intent_fields": sorted(ORDINARY_QUERY_INTENT_FIELDS),
            "intent_allowlist": sorted(ORDINARY_QUERY_INTENTS),
            "strategy_allowlist": sorted(ORDINARY_QUERY_STRATEGIES),
            "intent_system_prompt": perception_system,
            "facet_system_prompt": facet_system,
            "query_facet_protocol_hash": (
                context_graph_service.query_facet_protocol_hash()
            ),
            "query_facet_bilingual_enabled": bilingual_enabled,
            "active_profile_hash": active_profile_hash(
                db,
                knowledge_base_id,
            ),
            "chat_model": str(settings.chat_model),
            "chat_api_protocol": str(settings.chat_api_protocol),
            "bounded_history_protocol_version": (
                CONVERSATION_PROMPT_HISTORY_PROTOCOL_VERSION
            ),
            "bounded_history_turn_limit": PROMPT_HISTORY_MAX_TURNS,
            "bounded_history_token_limit": PROMPT_HISTORY_MAX_TOKENS,
            "conversation_history_is_evidence": False,
            "conversation_history_gray_zone_decision_authority": False,
            "gray_zone_decision_authority": False,
        }
    )


def _ordinary_query_intent_fallback(query: str) -> dict[str, Any]:
    lowered = str(query or "").casefold()
    if any(
        token in lowered
        for token in ("compare", "contrast", "difference", "对比", "比较", "区别")
    ):
        intent = "comparison"
        strategy = "hybrid"
    elif any(
        token in lowered
        for token in ("why", "how", "derive", "为什么", "如何", "推导")
    ):
        intent = "analysis"
        strategy = "local_graph"
    elif any(
        token in lowered
        for token in ("steps", "procedure", "algorithm", "步骤", "流程", "算法")
    ):
        intent = "procedure"
        strategy = "local_graph"
    else:
        intent = "definition"
        strategy = "global_dense"
    return {
        "intent": intent,
        "entities": [],
        "sub_queries": [str(query)],
        "needs_graph": True,
        "suggested_strategy": strategy,
    }


def _validate_ordinary_query_intent_payload(
    payload: Any,
    *,
    query: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("ordinary query intent payload must be an object")
    if frozenset(payload) != ORDINARY_QUERY_INTENT_FIELDS:
        missing = sorted(ORDINARY_QUERY_INTENT_FIELDS - frozenset(payload))
        unexpected = sorted(
            frozenset(payload) - ORDINARY_QUERY_INTENT_FIELDS
        )
        raise ValueError(
            "ordinary query intent schema mismatch: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    intent = str(payload.get("intent") or "").strip().casefold()
    if intent not in ORDINARY_QUERY_INTENTS:
        raise ValueError("ordinary query intent is outside the allowlist")
    strategy = (
        str(payload.get("suggested_strategy") or "")
        .strip()
        .casefold()
    )
    if strategy not in ORDINARY_QUERY_STRATEGIES:
        raise ValueError(
            "ordinary query strategy is outside the diagnostic allowlist"
        )
    entities = payload.get("entities")
    sub_queries = payload.get("sub_queries")
    if (
        not isinstance(entities, list)
        or len(entities) > 16
        or any(
            not isinstance(item, str)
            or not str(item).strip()
            or len(str(item)) > 128
            for item in entities
        )
    ):
        raise ValueError("ordinary query intent entities are invalid")
    if (
        not isinstance(sub_queries, list)
        or not sub_queries
        or len(sub_queries) > 8
        or any(
            not isinstance(item, str)
            or not str(item).strip()
            or len(str(item)) > 512
            for item in sub_queries
        )
    ):
        raise ValueError("ordinary query intent sub_queries are invalid")
    if type(payload.get("needs_graph")) is not bool:
        raise ValueError("ordinary query intent needs_graph must be boolean")
    return {
        "intent": intent,
        "entities": [str(item).strip() for item in entities],
        "sub_queries": [str(item).strip() for item in sub_queries],
        "needs_graph": bool(payload["needs_graph"]),
        # This field is retained for audit only. The user-selected retrieval
        # granularity and deterministic executor remain authoritative.
        "suggested_strategy": strategy,
        "strategy_is_executor_authority": False,
        "original_query_hash": context_graph_service.stable_hash(
            {"query": str(query)}
        ),
    }


def _validate_ordinary_prompt_history(
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if history is None:
        return []
    if not isinstance(history, list) or len(
        history
    ) > PROMPT_HISTORY_MAX_TURNS:
        raise ValueError(
            "ordinary query prompt history exceeds its bounded contract"
        )
    validated: list[dict[str, str]] = []
    for index, raw_item in enumerate(history):
        if (
            not isinstance(raw_item, dict)
            or frozenset(raw_item) != {"role", "content"}
        ):
            raise ValueError(
                "ordinary query prompt history item schema mismatch"
            )
        role = str(raw_item.get("role") or "")
        content = str(raw_item.get("content") or "").strip()
        if (
            role not in {"user", "assistant"}
            or not content
            or len(content) > 12_000
            or (
                index % 2 == 0
                and role != "user"
            )
            or (
                index % 2 == 1
                and role != "assistant"
            )
        ):
            raise ValueError(
                "ordinary query prompt history ordering or content is invalid"
            )
        validated.append({"role": role, "content": content})
    if len(validated) % 2 != 0:
        raise ValueError(
            "ordinary query prompt history must contain complete turns"
        )
    return validated


def _ordinary_query_facet_packet_hash(
    packet: dict[str, Any],
) -> str:
    payload = deepcopy(packet)
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics.pop("query_perception_audit", None)
    payload["diagnostics"] = diagnostics
    return context_graph_service.stable_hash(payload)


def _ordinary_query_perception_audit_from_packet(
    packet: dict[str, Any],
) -> dict[str, Any]:
    return dict(
        (packet.get("diagnostics") or {}).get(
            "query_perception_audit"
        )
        or {}
    )


def _ordinary_query_packet_is_replayable(
    packet: dict[str, Any],
    *,
    provider_protocol_hash: str,
) -> bool:
    try:
        raw_audit = (packet.get("diagnostics") or {}).get(
            "query_perception_audit"
        )
        if not isinstance(raw_audit, dict):
            return False
        validated_audit = OrdinaryQueryPerceptionAudit.model_validate(
            raw_audit,
            strict=True,
        )
        audit = validated_audit.model_dump(mode="json")
    except (AttributeError, TypeError, ValidationError, ValueError):
        return False
    model_call_count = audit["model_call_count"]
    return bool(
        audit.get("protocol_version")
        == ORDINARY_QUERY_PERCEPTION_PROTOCOL_VERSION
        and audit.get("provider_protocol_hash")
        == provider_protocol_hash
        and audit.get("model_call_budget")
        == ORDINARY_QUERY_PERCEPTION_MODEL_CALL_BUDGET
        and type(model_call_count) is int
        and 1 <= model_call_count <= ORDINARY_QUERY_PERCEPTION_MODEL_CALL_BUDGET
        and audit.get("budget_exhausted") is False
        and audit.get("intent_schema_validated") is True
        and audit.get("facet_schema_validated") is True
        and isinstance(
            audit.get("conversation_history_turn_count"),
            int,
        )
        and 0
        <= int(audit["conversation_history_turn_count"])
        <= PROMPT_HISTORY_MAX_TURNS
        and len(
            str(audit.get("conversation_history_hash") or "")
        )
        == 64
        and len(
            str(
                audit.get(
                    "conversation_history_audit_hash"
                )
                or ""
            )
        )
        == 64
        and audit.get("conversation_history_is_evidence") is False
        and audit.get(
            "conversation_history_gray_zone_decision_authority"
        )
        is False
        and audit.get("query_facet_packet_is_evidence") is False
        and audit.get("gray_zone_decision_authority") is False
        and audit.get("gray_zone_model_call_count") == 0
        and audit.get("query_facet_packet_hash")
        == _ordinary_query_facet_packet_hash(packet)
    )


def _ordinary_query_pointer_components(
    db: Session,
    *,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int | None,
    retrieval_granularity: RetrievalGranularity,
    conversation_state_scope_hash: str | None,
    provider_protocol_hash: str,
) -> dict[str, Any]:
    if conversation_state_scope_hash is None:
        conversation_state_scope_hash = (
            context_graph_service.anonymous_conversation_state_snapshot(
                knowledge_base_id
            ).scope_hash
        )
    context_state = context_graph_service.active_graph_admission_gate(
        db,
        knowledge_base_id,
    )
    (
        envelope,
        runtime_settings_hash,
    ) = context_graph_service.frozen_runtime_traversal_identity()
    provider_free_identity = (
        context_graph_service.context_graph_cache_key_components(
            knowledge_base_id=knowledge_base_id,
            query=query,
            filters=filters,
            context_state=context_state,
            retrieval_mode="layered_context_graph",
            conversation_state_scope_hash=(
                conversation_state_scope_hash
            ),
            retrieval_granularity=retrieval_granularity,
            result_top_k=(
                context_graph_service.resolve_result_top_k(top_k)
            ),
            query_facets={},
            profile_hash_value=active_profile_hash(
                db,
                knowledge_base_id,
            ),
            canonical_profile_hash_value=(
                context_graph_service.canonical_active_profile_state_hash(
                    db,
                    knowledge_base_id,
                )
            ),
            operating_envelope=envelope,
            runtime_settings_hash_value=runtime_settings_hash,
            cache_runtime_settings_hash_value=(
                context_graph_service.retrieval_cache_runtime_settings_hash()
            ),
            policy_state_hash_value=(
                context_graph_service.retrieval_policy_state_content_hash(
                    db,
                    knowledge_base_id,
                    policy_identity_frozen=False,
                    frozen_policy_state_hash=None,
                )
            ),
        )
    )
    return (
        cache_manager.validate_ordinary_query_replay_pointer_components(
            knowledge_base_id,
            {
                "pointer_key_protocol_version": (
                    cache_manager
                    .ORDINARY_QUERY_REPLAY_POINTER_KEY_PROTOCOL_VERSION
                ),
                "knowledge_base_id": str(knowledge_base_id),
                "provider_free_retrieval_identity": (
                    provider_free_identity
                ),
                "query_provider_protocol_hash": (
                    provider_protocol_hash
                ),
            },
        )
    )


def _ordinary_query_pointer_hash(
    pointer: dict[str, Any],
) -> str:
    payload = deepcopy(pointer)
    payload.pop("pointer_hash", None)
    return cache_manager.strict_json_sha256(payload)


def _delete_ordinary_query_pointer(
    *,
    knowledge_base_id: str,
    pointer_components: dict[str, Any],
) -> bool:
    manager = cache_manager.get_cache_manager()
    deleter = getattr(
        manager,
        "delete_ordinary_query_replay_pointer",
        None,
    )
    return bool(
        deleter(
            knowledge_base_id,
            pointer_components=pointer_components,
        )
    ) if callable(deleter) else False


def _read_ordinary_query_replay_packet(
    db: Session,
    *,
    knowledge_base_id: str,
    pointer_components: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    manager = cache_manager.get_cache_manager()
    reader = getattr(
        manager,
        "read_ordinary_query_replay_pointer",
        None,
    )
    pointer_digest = cache_manager.strict_json_sha256(
        pointer_components
    )
    if not callable(reader):
        return None, {
            "status": "unavailable",
            "reason": "ordinary_query_pointer_reader_unavailable",
            "pointer_key_digest": pointer_digest,
        }
    read = reader(
        knowledge_base_id,
        pointer_components=pointer_components,
    )
    if read.status != "hit" or read.payload is None:
        return None, {
            "status": read.status,
            "reason": (
                read.poison_reason
                or f"ordinary_query_pointer_{read.status}"
            ),
            "pointer_key_digest": read.key_digest,
            "ttl_seconds_remaining": read.ttl_seconds_remaining,
            "deletion_attempted": read.deletion_attempted,
            "deleted": read.deleted,
        }
    pointer = deepcopy(read.payload)
    rejection_reason: str | None = None
    try:
        if frozenset(pointer) != ORDINARY_QUERY_REPLAY_POINTER_FIELDS:
            raise ValueError("ordinary query pointer schema mismatch")
        if (
            pointer.get("protocol_version")
            != ORDINARY_QUERY_REPLAY_POINTER_PROTOCOL_VERSION
            or pointer.get("pointer_key_protocol_version")
            != cache_manager.ORDINARY_QUERY_REPLAY_POINTER_KEY_PROTOCOL_VERSION
            or pointer.get("pointer_key_digest") != pointer_digest
            or pointer.get("pointer_components")
            != pointer_components
            or str(pointer.get("knowledge_base_id") or "")
            != str(knowledge_base_id)
            or pointer.get("query_provider_protocol_hash")
            != pointer_components.get(
                "query_provider_protocol_hash"
            )
            or pointer.get("ttl_seconds")
            != ORDINARY_QUERY_REPLAY_POINTER_TTL_SECONDS
            or pointer.get("write_policy")
            != "postgresql_commit_then_redis_pointer"
            or pointer.get("pointer_hash")
            != _ordinary_query_pointer_hash(pointer)
            or not (
                0
                < int(read.ttl_seconds_remaining or 0)
                <= ORDINARY_QUERY_REPLAY_POINTER_TTL_SECONDS
            )
        ):
            raise ValueError("ordinary query pointer identity mismatch")
        trace = db.get(
            RetrievalTrace,
            str(pointer.get("source_retrieval_trace_id") or ""),
        )
        package = db.get(
            ContextPackage,
            str(pointer.get("source_context_package_id") or ""),
        )
        if (
            trace is None
            or package is None
            or str(trace.knowledge_base_id) != str(knowledge_base_id)
            or str(package.knowledge_base_id)
            != str(knowledge_base_id)
            or str(package.retrieval_trace_id or "") != str(trace.id)
        ):
            raise ValueError(
                "ordinary query pointer PostgreSQL binding mismatch"
            )
        packet = deepcopy(dict(trace.query_facets_json or {}))
        context_graph_service.validate_active_query_facet_packet(
            packet,
            query=str(trace.query),
        )
        if not _ordinary_query_packet_is_replayable(
            packet,
            provider_protocol_hash=str(
                pointer_components.get(
                    "query_provider_protocol_hash"
                )
                or ""
            ),
        ):
            raise ValueError(
                "ordinary query pointer packet replay audit mismatch"
            )
        packet_hash = _ordinary_query_facet_packet_hash(packet)
        audit = _ordinary_query_perception_audit_from_packet(
            packet
        )
        full_components = dict(
            (trace.diagnostics_json or {}).get(
                "cache_key_components"
            )
            or {}
        )
        if (
            pointer.get("query_facet_packet_hash")
            != packet_hash
            or pointer.get("query_perception_audit_hash")
            != cache_manager.strict_json_sha256(audit)
            or pointer.get("full_cache_key")
            != context_graph_service.stable_hash(full_components)
            or pointer.get("full_redis_key_digest")
            != cache_manager.strict_json_sha256(full_components)
        ):
            raise ValueError(
                "ordinary query pointer full-cache binding mismatch"
            )
        return packet, {
            "status": "hit",
            "reason": "postgresql_facet_replay_card_validated",
            "pointer_key_digest": pointer_digest,
            "ttl_seconds_remaining": read.ttl_seconds_remaining,
            "source_retrieval_trace_id": trace.id,
            "source_context_package_id": package.id,
        }
    except (TypeError, ValueError) as exc:
        rejection_reason = (
            f"{type(exc).__name__}:{str(exc)[:180]}"
        )
    deleted = _delete_ordinary_query_pointer(
        knowledge_base_id=knowledge_base_id,
        pointer_components=pointer_components,
    )
    return None, {
        "status": "poison",
        "reason": rejection_reason
        or "ordinary_query_pointer_validation_failed",
        "pointer_key_digest": pointer_digest,
        "ttl_seconds_remaining": read.ttl_seconds_remaining,
        "deletion_attempted": True,
        "deleted": deleted,
    }


def _schedule_ordinary_query_replay_pointer(
    db: Session,
    *,
    knowledge_base_id: str,
    result: Any,
    package: ContextPackage,
) -> dict[str, Any] | None:
    trace = result.trace
    packet = deepcopy(dict(trace.query_facets_json or {}))
    audit = _ordinary_query_perception_audit_from_packet(packet)
    provider_protocol_hash = str(
        audit.get("provider_protocol_hash") or ""
    )
    if not _ordinary_query_packet_is_replayable(
        packet,
        provider_protocol_hash=provider_protocol_hash,
    ):
        return None
    full_components = deepcopy(dict(result.cache_components or {}))
    provider_free_identity = deepcopy(full_components)
    provider_free_identity["query_facets_hash"] = (
        context_graph_service.query_facet_cache_identity_hash({})
    )
    provider_free_semantic_entry = (
        context_graph_service.semantic_entry_query_for_search(
            str(provider_free_identity.get("query") or ""),
            {},
        )
    )
    provider_free_identity[
        "semantic_entry_query_protocol_version"
    ] = provider_free_semantic_entry["protocol_version"]
    provider_free_identity["semantic_entry_query"] = (
        provider_free_semantic_entry["query"]
    )
    provider_free_identity["semantic_entry_query_hash"] = (
        provider_free_semantic_entry["packet_hash"]
    )
    pointer_components = (
        cache_manager.validate_ordinary_query_replay_pointer_components(
            knowledge_base_id,
            {
                "pointer_key_protocol_version": (
                    cache_manager
                    .ORDINARY_QUERY_REPLAY_POINTER_KEY_PROTOCOL_VERSION
                ),
                "knowledge_base_id": str(knowledge_base_id),
                "provider_free_retrieval_identity": (
                    provider_free_identity
                ),
                "query_provider_protocol_hash": (
                    provider_protocol_hash
                ),
            },
        )
    )
    pointer_digest = cache_manager.strict_json_sha256(
        pointer_components
    )
    pointer = {
        "protocol_version": (
            ORDINARY_QUERY_REPLAY_POINTER_PROTOCOL_VERSION
        ),
        "pointer_key_protocol_version": (
            cache_manager
            .ORDINARY_QUERY_REPLAY_POINTER_KEY_PROTOCOL_VERSION
        ),
        "pointer_key_digest": pointer_digest,
        "pointer_components": pointer_components,
        "knowledge_base_id": str(knowledge_base_id),
        "source_retrieval_trace_id": str(trace.id),
        "source_context_package_id": str(package.id),
        "query_facet_packet_hash": (
            _ordinary_query_facet_packet_hash(packet)
        ),
        "query_perception_audit_hash": (
            cache_manager.strict_json_sha256(audit)
        ),
        "full_cache_key": context_graph_service.stable_hash(
            full_components
        ),
        "full_redis_key_digest": (
            cache_manager.strict_json_sha256(full_components)
        ),
        "query_provider_protocol_hash": provider_protocol_hash,
        "ttl_seconds": ORDINARY_QUERY_REPLAY_POINTER_TTL_SECONDS,
        "write_policy": "postgresql_commit_then_redis_pointer",
        "pointer_hash": "",
    }
    pointer["pointer_hash"] = _ordinary_query_pointer_hash(pointer)
    cache_manager.schedule_ordinary_query_replay_pointer_write_after_commit(
        db,
        knowledge_base_id=knowledge_base_id,
        pointer_components=pointer_components,
        payload=pointer,
        ttl=ORDINARY_QUERY_REPLAY_POINTER_TTL_SECONDS,
    )
    return pointer


async def _bounded_ordinary_query_perception(
    db: Session,
    *,
    knowledge_base_id: str,
    query: str,
    provider_protocol_hash: str,
    conversation_prompt_history: (
        list[dict[str, str]] | None
    ) = None,
    conversation_prompt_history_audit: (
        dict[str, Any] | None
    ) = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    profile = active_profile_json(db, knowledge_base_id)
    provider = context_graph_service.ChatProvider()
    model_call_count = 0
    bounded_history = _validate_ordinary_prompt_history(
        conversation_prompt_history
    )
    bounded_history_audit = dict(
        conversation_prompt_history_audit or {}
    )
    perception_system = profile_prompt_template(
        profile,
        "question_perception_system",
        DEFAULT_QUESTION_PERCEPTION_SYSTEM,
        {
            "perception_domain": profile_prompt(
                profile,
                "perception_domain",
                "context-graph-grounded knowledge-base search",
            ),
            "entity_label": profile_prompt(
                profile,
                "entity_label",
                "source-grounded concepts",
            ),
        },
    )
    intent_fallback = _ordinary_query_intent_fallback(query)
    model_call_count += 1
    raw_intent = await classify_json_with_budget(
        provider,
        system_prompt=perception_system,
        user_prompt=str(
            {
                "question": query,
                "bounded_conversation_history": bounded_history,
                "allowed_output_fields": sorted(
                    ORDINARY_QUERY_INTENT_FIELDS
                ),
                "intent_allowlist": sorted(ORDINARY_QUERY_INTENTS),
                "strategy_is_diagnostic_only": True,
                "forbidden": [
                    "document ids",
                    "chunk ids",
                    "node ids",
                    "citations",
                    "facts",
                    "path decisions",
                    "gray-zone decisions",
                ],
            }
        ),
        fallback=intent_fallback,
        max_tokens=ORDINARY_QUERY_INTENT_JSON_MAX_TOKENS,
    )
    intent = _validate_ordinary_query_intent_payload(
        raw_intent,
        query=query,
    )
    bilingual_enabled = bool(settings.query_facet_bilingual_enabled)
    facet_system = " ".join(
        part.strip()
        for part in (
            profile_prompt(
                profile,
                "query_facet_extractor_system",
                DEFAULT_QUERY_FACET_EXTRACTOR_SYSTEM,
            ),
            profile_prompt(
                profile,
                (
                    "query_facet_bilingual_suffix"
                    if bilingual_enabled
                    else "query_facet_alias_suffix"
                ),
                (
                    DEFAULT_QUERY_FACET_BILINGUAL_SUFFIX
                    if bilingual_enabled
                    else DEFAULT_QUERY_FACET_ALIAS_SUFFIX
                ),
            ),
        )
        if part and part.strip()
    )
    facet_fallback = {"_ordinary_query_facet_fallback": True}
    model_call_count += 1
    raw_facets = await classify_json_with_budget(
        provider,
        system_prompt=facet_system,
        user_prompt=str(
            {
                "question": query,
                "bounded_conversation_history": bounded_history,
                "query_intent": intent,
                "bilingual_query_facets_enabled": bilingual_enabled,
                "required_json_shape": {
                    "facet_groups": [
                        {
                            "facet": "canonical query facet",
                            "role": (
                                "domain | procedure | constraint | alias | lexical"
                            ),
                            "aliases": ["bounded lexical aliases"],
                        }
                    ],
                    "answer_shape": "grounded answer shape",
                    "drop_terms": ["non-semantic query filler"],
                },
                "hard_output_limits": {
                    "facet_groups_max_items": (
                        context_graph_service.QUERY_FACET_MAX_GROUPS
                    ),
                    "facet_max_characters": 96,
                    "aliases_per_group_max_items": (
                        context_graph_service.QUERY_FACET_MAX_ALIASES
                    ),
                    "alias_max_characters": 96,
                    "drop_terms_max_items": 64,
                },
                "rejection_rules": [
                    "Do not output corpus facts or evidence.",
                    "Do not output document, chunk, node, citation, or path identifiers.",
                    "Do not decide traversal, gray-zone, thresholds, or stopping.",
                    "Never exceed any hard_output_limits value; prefer fewer canonical aliases.",
                ],
            }
        ),
        fallback=facet_fallback,
        max_tokens=ORDINARY_QUERY_FACET_JSON_MAX_TOKENS,
    )
    if (
        isinstance(raw_facets, dict)
        and raw_facets.get("_ordinary_query_facet_fallback")
    ):
        if not settings.enable_model_fallback:
            raise context_graph_service.FallbackDisabledError(
                "Ordinary query facet sampling is unavailable because "
                "ENABLE_MODEL_FALLBACK is false"
            )
        facets = context_graph_service.query_facets_for_search(
            query,
            None,
            intent,
        )
        facet_schema_validated = False
    else:
        validated_facets = (
            context_graph_service.validate_query_facet_llm_payload(
                raw_facets
            )
        )
        facets = context_graph_service.query_facets_for_search(
            query,
            validated_facets,
            intent,
        )
        facet_schema_validated = True
    facets.setdefault("diagnostics", {})[
        "bilingual_query_facets_enabled"
    ] = bilingual_enabled
    packet_hash = _ordinary_query_facet_packet_hash(facets)
    audit = {
        "protocol_version": (
            ORDINARY_QUERY_PERCEPTION_PROTOCOL_VERSION
        ),
        "provider_protocol_hash": provider_protocol_hash,
        "model_call_budget": (
            ORDINARY_QUERY_PERCEPTION_MODEL_CALL_BUDGET
        ),
        "model_call_count": model_call_count,
        "budget_exhausted": (
            model_call_count
            > ORDINARY_QUERY_PERCEPTION_MODEL_CALL_BUDGET
        ),
        "intent_schema_validated": True,
        "facet_schema_validated": facet_schema_validated,
        "query_intent_hash": context_graph_service.stable_hash(intent),
        "conversation_history_hash": (
            cache_manager.strict_json_sha256(
                {"history": bounded_history}
            )
        ),
        "conversation_history_turn_count": len(bounded_history),
        "conversation_history_audit_hash": (
            cache_manager.strict_json_sha256(
                {"history_audit": bounded_history_audit}
            )
        ),
        "conversation_history_is_evidence": False,
        "conversation_history_gray_zone_decision_authority": False,
        "query_facet_packet_hash": packet_hash,
        "query_facet_packet_is_evidence": False,
        "query_facet_packet_routing_only": True,
        "suggested_strategy_is_executor_authority": False,
        "retrieval_granularity_is_user_or_executor_locked": True,
        "gray_zone_decision_authority": False,
        "gray_zone_rule_inputs_modified": False,
        "gray_zone_model_call_count": 0,
    }
    if audit["budget_exhausted"]:
        raise RuntimeError(
            "ordinary query perception exceeded its hard model-call budget"
        )
    facets["diagnostics"]["query_perception_audit"] = deepcopy(
        audit
    )
    return facets, audit


async def _ordinary_layered_search(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int | None,
    *,
    retrieval_granularity: RetrievalGranularity,
    conversation_state_scope_hash: str | None,
    conversation_state_audit: dict[str, Any] | None,
    conversation_prompt_history: (
        list[dict[str, str]] | None
    ) = None,
    conversation_prompt_history_audit: (
        dict[str, Any] | None
    ) = None,
):
    provider_protocol_hash = _ordinary_query_perception_protocol_hash(
        db,
        knowledge_base_id,
    )
    pointer_components = _ordinary_query_pointer_components(
        db,
        knowledge_base_id=knowledge_base_id,
        query=query,
        filters=filters,
        top_k=top_k,
        retrieval_granularity=retrieval_granularity,
        conversation_state_scope_hash=conversation_state_scope_hash,
        provider_protocol_hash=provider_protocol_hash,
    )
    frozen_packet, pointer_audit = (
        _read_ordinary_query_replay_packet(
            db,
            knowledge_base_id=knowledge_base_id,
            pointer_components=pointer_components,
        )
    )
    prior_cache_lookup: dict[str, Any] | None = None
    if frozen_packet is not None:
        probe_audit: dict[str, Any] = {}
        replay = await layered_search(
            db,
            knowledge_base_id,
            query,
            filters,
            top_k,
            query_facets=frozen_packet,
            retrieval_granularity=retrieval_granularity,
            conversation_state_scope_hash=conversation_state_scope_hash,
            conversation_state_audit=conversation_state_audit,
            cache_only=True,
            cache_probe_audit=probe_audit,
        )
        if replay is not None:
            return replay
        if probe_audit.get("status") in {"poison", "unavailable"}:
            prior_cache_lookup = deepcopy(probe_audit)
        _delete_ordinary_query_pointer(
            knowledge_base_id=knowledge_base_id,
            pointer_components=pointer_components,
        )
        pointer_audit = {
            **pointer_audit,
            "status": "stale",
            "reason": (
                "full_retrieval_cache_"
                + str(probe_audit.get("status") or "miss")
            ),
            "full_cache_probe": deepcopy(probe_audit),
            "deleted": True,
        }
    query_facets, perception_audit = (
        await _bounded_ordinary_query_perception(
            db,
            knowledge_base_id=knowledge_base_id,
            query=query,
            provider_protocol_hash=provider_protocol_hash,
            conversation_prompt_history=conversation_prompt_history,
            conversation_prompt_history_audit=(
                conversation_prompt_history_audit
            ),
        )
    )
    perception_audit["provider_free_pointer"] = deepcopy(
        pointer_audit
    )
    query_facets["diagnostics"][
        "query_perception_audit"
    ] = deepcopy(perception_audit)
    return await layered_search(
        db,
        knowledge_base_id,
        query,
        filters,
        top_k,
        query_facets=query_facets,
        retrieval_granularity=retrieval_granularity,
        conversation_state_scope_hash=conversation_state_scope_hash,
        conversation_state_audit=conversation_state_audit,
        cache_lookup_prior=prior_cache_lookup,
    )


async def search_chunks_with_audit(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int | None,
    retrieval_granularity: RetrievalGranularity = "mid",
    *,
    conversation_state_scope_hash: str | None = None,
    conversation_state_audit: dict[str, Any] | None = None,
    conversation_prompt_history: (
        list[dict[str, str]] | None
    ) = None,
    conversation_prompt_history_audit: (
        dict[str, Any] | None
    ) = None,
) -> tuple[list[dict], dict]:
    result = await _ordinary_layered_search(
        db,
        knowledge_base_id,
        query,
        filters,
        top_k,
        retrieval_granularity=retrieval_granularity,
        conversation_state_scope_hash=conversation_state_scope_hash,
        conversation_state_audit=conversation_state_audit,
        conversation_prompt_history=conversation_prompt_history,
        conversation_prompt_history_audit=(
            conversation_prompt_history_audit
        ),
    )
    context_package = result.context_package
    cache_write_envelope = None
    pointer_write = None
    if context_package is None and result.results:
        context_package = await run_bounded_source_io(
            build_context_package,
            db,
            knowledge_base_id=knowledge_base_id,
            query=query,
            trace=result.trace,
            results=result.results,
            snapshot_verifier=result.snapshot_verifier,
        )
        cache_write_envelope = schedule_layered_retrieval_cache_write(
            db,
            result=result,
            package=context_package,
        )
        if cache_write_envelope:
            pointer_write = _schedule_ordinary_query_replay_pointer(
                db,
                knowledge_base_id=knowledge_base_id,
                result=result,
                package=context_package,
            )
    retrieval_cache_audit = dict(
        (result.audit or {}).get("retrieval_cache") or {}
    )
    retrieval_cache_audit["context_package_reused"] = (
        result.context_package is not None
    )
    retrieval_cache_audit["write_scheduled_after_commit"] = bool(
        cache_write_envelope
    )
    retrieval_cache_audit[
        "ordinary_query_pointer_write_scheduled_after_commit"
    ] = bool(pointer_write)
    audit = {
        **result.audit,
        "retrieval_cache": retrieval_cache_audit,
        "retrieval_trace_id": result.trace.id,
        "context_package_id": context_package.id if context_package else None,
    }
    return result.results, audit


async def layered_context_search_chunks_with_audit(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int | None,
    route: str = "multi_hop_research",
    retrieval_granularity: RetrievalGranularity = "mid",
    *,
    conversation_state_scope_hash: str | None = None,
    conversation_state_audit: dict[str, Any] | None = None,
    conversation_prompt_history: (
        list[dict[str, str]] | None
    ) = None,
    conversation_prompt_history_audit: (
        dict[str, Any] | None
    ) = None,
) -> tuple[list[dict], dict]:
    result = await _ordinary_layered_search(
        db,
        knowledge_base_id,
        query,
        filters,
        top_k,
        retrieval_granularity=retrieval_granularity,
        conversation_state_scope_hash=conversation_state_scope_hash,
        conversation_state_audit=conversation_state_audit,
        conversation_prompt_history=conversation_prompt_history,
        conversation_prompt_history_audit=(
            conversation_prompt_history_audit
        ),
    )
    context_package = result.context_package
    cache_write_envelope = None
    pointer_write = None
    if context_package is None and result.results:
        context_package = await run_bounded_source_io(
            build_context_package,
            db,
            knowledge_base_id=knowledge_base_id,
            query=query,
            trace=result.trace,
            results=result.results,
            snapshot_verifier=result.snapshot_verifier,
        )
        cache_write_envelope = schedule_layered_retrieval_cache_write(
            db,
            result=result,
            package=context_package,
        )
        if cache_write_envelope:
            pointer_write = _schedule_ordinary_query_replay_pointer(
                db,
                knowledge_base_id=knowledge_base_id,
                result=result,
                package=context_package,
            )
    retrieval_cache_audit = dict(
        (result.audit or {}).get("retrieval_cache") or {}
    )
    retrieval_cache_audit["context_package_reused"] = (
        result.context_package is not None
    )
    retrieval_cache_audit["write_scheduled_after_commit"] = bool(
        cache_write_envelope
    )
    retrieval_cache_audit[
        "ordinary_query_pointer_write_scheduled_after_commit"
    ] = bool(pointer_write)
    audit = {
        **result.audit,
        "retrieval_cache": retrieval_cache_audit,
        "route": route,
        "retrieval_trace_id": result.trace.id,
        "context_package_id": context_package.id if context_package else None,
    }
    return result.results, audit


async def search_chunks(
    db: Session,
    knowledge_base_id: str,
    query: str,
    filters: SearchFilters,
    top_k: int | None,
    retrieval_granularity: RetrievalGranularity = "mid",
) -> list[dict]:
    return (
        await search_chunks_with_audit(
            db,
            knowledge_base_id,
            query,
            filters,
            top_k,
            retrieval_granularity=retrieval_granularity,
        )
    )[0]


def _public_structure_closure(value: dict | None) -> dict:
    closure = dict(value or {})
    parent = closure.get("parent_section") or {}

    def node_ids(key: str) -> list[str]:
        return [
            str(item.get("node_id"))
            for item in (closure.get(key) or [])
            if isinstance(item, dict) and item.get("node_id")
        ]

    return {
        "previous_chunk_id": closure.get("previous_chunk_id"),
        "next_chunk_id": closure.get("next_chunk_id"),
        "parent_section_node_id": (
            str(parent.get("node_id"))
            if isinstance(parent, dict) and parent.get("node_id")
            else None
        ),
        "same_page_region_node_ids": node_ids("same_page_region"),
        "table_formula_caption_node_ids": node_ids("table_formula_caption"),
        "code_block_node_ids": node_ids("code_blocks"),
        "bridge_chunk_ids": [str(item) for item in (closure.get("bridge_chunk_ids") or [])],
        "parent_section": parent if isinstance(parent, dict) and parent else None,
        "same_page_region": list(closure.get("same_page_region") or []),
        "table_formula_caption": list(
            closure.get("table_formula_caption") or []
        ),
        "code_blocks": list(closure.get("code_blocks") or []),
    }


def _public_source_span(value: dict | None) -> dict:
    source = dict(value or {})
    projected = {
        key: source.get(key)
        for key in (
            "contract_version",
            "document_version_id",
            "chunk_id",
            "source_path",
            "source_checksum",
            "logical_source_path",
            "source_snapshot_verification",
            "chunk_text_hash_protocol_version",
            "chunk_text_hash",
            "raw_span_text_hash_protocol_version",
            "raw_span_text_hash",
            "char_span",
            "raw_chunk_char_span",
            "page_range",
            "section_path",
            "structure_path",
            "structure_node_ids",
            "bbox",
            "context_package_id",
            "retrieval_trace_id",
            "verification_id",
            "content_clipped",
            "content_token_count",
        )
        if key in source
    }
    if not projected.get("bbox"):
        projected["bbox"] = None
    return projected


def _public_context_package_chunk(item: dict, *, package_id: str) -> dict:
    return {
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "document_version_id": item.get("document_version_id"),
        "document_title": item.get("document_title") or "",
        "source_path": item.get("source_path") or "",
        "logical_source_path": item.get("logical_source_path") or "",
        "content": item.get("content") or "",
        "content_clipped": bool(item.get("content_clipped")),
        "content_token_count": int(item.get("content_token_count") or 0),
        "original_token_count": int(item.get("original_token_count") or 0),
        "raw_chunk_char_span": item.get("raw_chunk_char_span") or item.get("char_span") or [0, 0],
        "chunk_text_hash_protocol_version": item.get("chunk_text_hash_protocol_version"),
        "chunk_text_hash": item.get("chunk_text_hash"),
        "raw_span_text_hash_protocol_version": item.get("raw_span_text_hash_protocol_version"),
        "raw_span_text_hash": item.get("raw_span_text_hash"),
        "section_path": item.get("section_path"),
        "structure_path": item.get("structure_path"),
        "structure_node_ids": list(item.get("structure_node_ids") or []),
        "structure_nodes": list(item.get("structure_nodes") or []),
        "parent_section": item.get("parent_section"),
        "page_range": item.get("page_range") or [None, None],
        "char_span": item.get("char_span") or [0, 0],
        "bbox": item.get("bbox") or None,
        "source_span": _public_source_span(item.get("source_span")),
        "structure_closure": _public_structure_closure(item.get("structure_closure")),
        "why_selected": dict(item.get("why_selected") or {}),
        "dedupe_key": item.get("dedupe_key") or f"{item.get('chunk_id')}:{item.get('char_span')}",
        "role": item.get("role") or "restored_context",
        "context_package_id": package_id,
    }


def _public_context_item(item: dict, *, package_id: str) -> dict:
    public_chunk = _public_context_package_chunk(item, package_id=package_id)
    return {
        "chunk_id": public_chunk["chunk_id"],
        "document_title": public_chunk["document_title"],
        "source_path": public_chunk["source_path"],
        "content": public_chunk["content"],
        "snippet": str(public_chunk["content"])[:280],
        "metadata": {
            "source_path": public_chunk["source_path"],
            "logical_source_path": public_chunk["logical_source_path"],
            "section_path": public_chunk["section_path"],
            "structure_path": public_chunk["structure_path"],
            "parent_section_node_id": public_chunk["structure_closure"]["parent_section_node_id"],
            "parent_section": public_chunk.get("parent_section"),
            "structure_node_ids": public_chunk["structure_node_ids"],
            "page_range": public_chunk["page_range"],
            "char_span": public_chunk["char_span"],
            "bbox": public_chunk["bbox"],
            "source_span": public_chunk["source_span"],
            "structure_closure": public_chunk["structure_closure"],
            "why_selected": public_chunk["why_selected"],
            "dedupe_key": public_chunk["dedupe_key"],
            "role": public_chunk["role"],
            "content_clipped": public_chunk["content_clipped"],
            "content_token_count": public_chunk["content_token_count"],
            "original_token_count": public_chunk["original_token_count"],
            "raw_chunk_char_span": public_chunk["raw_chunk_char_span"],
            "context_package_id": package_id,
        },
    }


def _closed_canonical_entry_parent_refs(
    raw: Any,
) -> list[RetrievalEntryParentRef]:
    """Validate that persisted entry-parent references are already canonical.

    Writers deduplicate, merge and deterministically order these references.
    Public replay must reject persisted drift instead of silently repairing it.
    """

    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace entry-parent references are not a list"
        )
    try:
        normalized = [
            RetrievalEntryParentRef.model_validate(item) for item in raw
        ]
    except (TypeError, ValidationError) as exc:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace entry-parent reference is not a "
            "closed contract"
        ) from exc

    normalized_payload = [
        item.model_dump(mode="json") for item in normalized
    ]
    canonical = [
        RetrievalEntryParentRef.model_validate(item)
        for item in context_graph_service.canonical_entry_parent_refs(
            normalized_payload
        )
    ]
    canonical_payload = [
        item.model_dump(mode="json") for item in canonical
    ]
    if normalized_payload != canonical_payload:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace entry-parent references are not "
            "canonical, unique, and deterministically ordered"
        )
    return canonical


def _support_refs_include(
    container: RetrievalSupportRefs,
    required: RetrievalSupportRefs,
) -> bool:
    """Return whether package evidence contains every trace-parent fact."""

    container_payload = container.model_dump(mode="json")
    required_payload = required.model_dump(mode="json")
    for field, required_value in required_payload.items():
        container_value = container_payload.get(field)
        if isinstance(required_value, list):
            if not set(required_value).issubset(set(container_value or [])):
                return False
        elif isinstance(required_value, dict):
            if not isinstance(container_value, dict) or any(
                key not in container_value
                or container_value[key] != value
                for key, value in required_value.items()
            ):
                return False
        elif required_value is not None and container_value != required_value:
            return False
    return True


_TRACE_PATH_LABEL_INTERNAL_FIELDS = frozenset(
    {
        "path_edge_distances",
        "path_edge_strengths",
        "cycle_distance_rewards",
    }
)
_TRACE_PATH_LABEL_INTERNAL_SUPPORT_FIELDS = frozenset(
    {
        "query_rq_seed_cards",
        "rq",
    }
)


def _strict_finite_number_list(
    raw: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float | None = None,
) -> list[float]:
    if not isinstance(raw, list):
        raise ContextPackagePublicIntegrityError(
            f"persisted retrieval trace {field_name} is not a list"
        )
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ContextPackagePublicIntegrityError(
                f"persisted retrieval trace {field_name} contains a "
                "non-number"
            )
        normalized = float(value)
        if (
            not math.isfinite(normalized)
            or normalized < minimum
            or (maximum is not None and normalized > maximum)
        ):
            raise ContextPackagePublicIntegrityError(
                f"persisted retrieval trace {field_name} is outside its "
                "finite range"
            )
        values.append(normalized)
    return values


def _validate_cycle_reward_replay(
    *,
    rewards: list[CycleDistanceRewardAudit],
    path: list[str],
    path_edge_ids: list[str],
    path_edge_distances: list[float],
    path_edge_strengths: list[float],
    reward_so_far: Any,
    operating_envelope: dict[str, Any] | None,
) -> None:
    if len(path) != len(path_edge_ids) + 1:
        raise ContextPackagePublicIntegrityError(
            "persisted cycle reward path cannot be replayed"
        )
    expected_cycles: list[dict[str, Any]] = []
    for edge_index, edge_id in enumerate(path_edge_ids):
        neighbor_id = path[edge_index + 1]
        previous_path = path[: edge_index + 1]
        if neighbor_id not in previous_path:
            continue
        previous_index = max(
            index
            for index, node_id in enumerate(previous_path)
            if node_id == neighbor_id
        )
        expected_cycles.append(
            {
                "cycle_edges": path_edge_ids[
                    previous_index : edge_index + 1
                ],
                "cycle_distance": round(
                    sum(
                        path_edge_distances[
                            previous_index : edge_index + 1
                        ]
                    ),
                    6,
                ),
                "current_edge_id": edge_id,
                "current_edge_distance": path_edge_distances[edge_index],
                "current_edge_strength": path_edge_strengths[edge_index],
            }
        )
    if len(rewards) != len(expected_cycles):
        raise ContextPackagePublicIntegrityError(
            "persisted cycle rewards are duplicated or incomplete"
        )
    if not rewards:
        if (
            isinstance(reward_so_far, bool)
            or not isinstance(reward_so_far, (int, float))
            or not math.isfinite(float(reward_so_far))
            or not math.isclose(
                float(reward_so_far), 0.0, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted path reward has no replayable cycle event"
            )
        return
    envelope = operating_envelope or {}
    cap = envelope.get("max_cycle_reward_per_path")
    threshold = envelope.get("cycle_reward_distance_threshold")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (cap, threshold)
    ):
        raise ContextPackagePublicIntegrityError(
            "persisted cycle reward has no frozen finite envelope"
        )

    total = 0.0
    for reward, expected in zip(rewards, expected_cycles):
        cap_remaining = max(0.0, float(cap) - total)
        if cap_remaining <= 0.0:
            expected_edges = [expected["current_edge_id"]]
            expected_distance = expected["current_edge_distance"]
            expected_before = 0.0
            expected_after = 0.0
            expected_reason = "max_cycle_reward_per_path_exhausted"
            expected_support_delta = 0
        else:
            expected_edges = expected["cycle_edges"]
            expected_distance = expected["cycle_distance"]
            if float(threshold) <= 0.0 or expected_distance > float(
                threshold
            ):
                expected_before = 0.0
                expected_after = 0.0
                expected_reason = "cycle_distance_above_threshold"
                expected_support_delta = 0
            else:
                expected_before = 0.04 * expected[
                    "current_edge_strength"
                ] * math.exp(
                    -expected_distance / max(float(threshold), 1e-6)
                )
                expected_after = min(cap_remaining, expected_before)
                expected_reason = (
                    "within_cap"
                    if math.isclose(
                        expected_after,
                        expected_before,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    else "max_cycle_reward_per_path"
                )
                expected_support_delta = 1
        if not (
            reward.cycle_edges == expected_edges
            and math.isclose(
                reward.cycle_distance,
                expected_distance,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                reward.edge_strength,
                expected["current_edge_strength"],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                reward.reward_before_cap,
                round(expected_before, 6),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                reward.reward_after_cap,
                round(expected_after, 6),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and reward.cap_reason == expected_reason
            and reward.support_delta == expected_support_delta
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted cycle reward does not replay writer formula"
            )
        total = round(total + max(0.0, reward.reward_after_cap), 6)
    if (
        isinstance(reward_so_far, bool)
        or not isinstance(reward_so_far, (int, float))
        or not math.isfinite(float(reward_so_far))
        or not math.isclose(
            float(reward_so_far), total, rel_tol=0.0, abs_tol=1e-6
        )
        or total > float(cap) + 1e-6
    ):
        raise ContextPackagePublicIntegrityError(
            "persisted path reward does not equal replayed cycle total"
        )


def _validate_trace_path_label_internal_fields(
    raw: dict[str, Any],
    *,
    operating_envelope: dict[str, Any] | None = None,
) -> None:
    required_writer_fields = {
        "path_edge_distances",
        "path_edge_strengths",
        "cycle_distance_rewards",
        "path_edge_ids",
        "path_edge_types",
        "path",
    }
    missing_writer_fields = required_writer_fields.difference(raw)
    if missing_writer_fields:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path label is missing writer facts: "
            + ", ".join(sorted(missing_writer_fields))
        )
    internal_path_fields_present = True
    if internal_path_fields_present:
        distances = _strict_finite_number_list(
            raw["path_edge_distances"],
            field_name="path_edge_distances",
            minimum=0.0,
        )
        strengths = _strict_finite_number_list(
            raw["path_edge_strengths"],
            field_name="path_edge_strengths",
            minimum=0.0,
            maximum=1.0,
        )
        edge_ids = raw["path_edge_ids"]
        edge_types = raw["path_edge_types"]
        path = raw["path"]
        if not all(
            isinstance(items, list)
            for items in (edge_ids, edge_types, path)
        ) or any(
            not isinstance(value, str) or not value.strip()
            for items in (edge_ids, edge_types, path)
            for value in items
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path identities are invalid"
            )
        edge_count = len(edge_ids)
        if not (
            edge_count
            == len(edge_types)
            == len(distances)
            == len(strengths)
            == max(len(path) - 1, 0)
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path arrays are not parallel"
            )
    raw_rewards = raw["cycle_distance_rewards"]
    if not isinstance(raw_rewards, list):
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace cycle rewards are not a list"
        )
    try:
        rewards = [
            CycleDistanceRewardAudit.model_validate(
                item,
                strict=True,
            )
            for item in raw_rewards
        ]
    except (TypeError, ValidationError, ValueError) as exc:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace cycle rewards are not a "
            "closed contract"
        ) from exc
    _validate_cycle_reward_replay(
        rewards=rewards,
        path=list(raw.get("path") or []),
        path_edge_ids=edge_ids,
        path_edge_distances=distances,
        path_edge_strengths=strengths,
        reward_so_far=raw.get("reward_so_far", 0.0),
        operating_envelope=operating_envelope,
    )


def _validate_trace_path_edge_writer_facts(
    db: Session,
    *,
    trace: RetrievalTrace,
    raw_labels: list[Any],
) -> None:
    """Replay every persisted path hop against its frozen graph-state row.

    ``path_edge_distances`` and ``path_edge_strengths`` are writer facts that
    are intentionally not exposed by the public path-label schema.  Shape and
    finite-range checks cannot establish their provenance: an acyclic trace
    otherwise permits both arrays to be synchronously rewritten.  The graph
    edge rows are the PostgreSQL authority, while the trace's layer hashes
    freeze which historical graph states may be used for replay.
    """

    layer_configs: dict[str, dict[str, Any]] = {
        "chunk": {
            "edge_model": ChunkRelationEdge,
            "state_model": ChunkRelationGraphState,
            "state_id_field": "graph_state_id",
            "source_field": "source_chunk_id",
            "target_field": "target_chunk_id",
            "trace_hash": trace.chunk_relation_graph_hash,
            "edge_hash_field": "graph_state_hash",
        },
        "mid": {
            "edge_model": MidConceptEdge,
            "state_model": MidConceptState,
            "state_id_field": "concept_state_id",
            "source_field": "source_concept_id",
            "target_field": "target_concept_id",
            "trace_hash": trace.mid_concept_hash,
            "edge_hash_field": "state_hash",
        },
        "coarse": {
            "edge_model": CoarseConceptEdge,
            "state_model": CoarseConceptState,
            "state_id_field": "coarse_state_id",
            "source_field": "source_concept_id",
            "target_field": "target_concept_id",
            "trace_hash": trace.coarse_concept_hash,
            "edge_hash_field": "state_hash",
        },
    }
    raw_entry_nodes = trace.entry_nodes_json
    if not isinstance(raw_entry_nodes, list):
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace entry nodes are not a closed list"
        )
    validate_semantic_entry_query_trace_packet(
        (trace.diagnostics_json or {}).get("semantic_entry_query"),
        query=str(trace.query),
        query_facets=dict(trace.query_facets_json or {}),
    )
    replayed_entry_selection_audit: dict[str, Any] = {}
    if any(
        isinstance(entry, dict)
        and str(entry.get("layer") or "") in {"coarse", "mid"}
        for entry in raw_entry_nodes
    ):
        replayed_entry_selection_audit = (
            validate_entry_topology_replay_against_database(
                db,
                knowledge_base_id=str(trace.knowledge_base_id),
                raw_audit=validate_entry_selection_trace_diagnostics(
                    dict(trace.diagnostics_json or {})
                ),
                query_facets=dict(trace.query_facets_json or {}),
            )
        )
        replayed_entry_selection_audit = (
            validate_entry_dense_replay_against_database(
                db,
                knowledge_base_id=str(trace.knowledge_base_id),
                query=str(trace.query),
                query_facets=dict(trace.query_facets_json or {}),
                raw_audit=replayed_entry_selection_audit,
                raw_dense_replay_input=dict(
                    (trace.diagnostics_json or {}).get(
                        "entry_dense_replay_input"
                    )
                    or {}
                ),
            )
        )
    entry_strengths: dict[tuple[str, str], float] = {}
    for raw_entry in raw_entry_nodes:
        if not isinstance(raw_entry, dict):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace entry node is not an object"
            )
        entry_layer = str(raw_entry.get("layer") or "")
        entry_node_id = str(raw_entry.get("node_id") or "")
        entry_strength = raw_entry.get("entry_strength")
        if entry_layer not in layer_configs:
            # RQ membership is a deterministic drill-down layer, not a
            # physical edge-walk layer covered by this replay.
            continue
        if (
            not entry_node_id
            or isinstance(entry_strength, bool)
            or not isinstance(entry_strength, (int, float))
            or not math.isfinite(float(entry_strength))
            or not 0.0 < float(entry_strength) <= 1.0
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace entry strength is not a writer fact"
            )
        if entry_layer == "chunk":
            metadata = raw_entry.get("metadata")
            if not isinstance(metadata, dict):
                raise ContextPackagePublicIntegrityError(
                    "persisted chunk entry has no writer metadata authority"
                )
            metadata_strength = metadata.get("entry_strength")
            metadata_distance = metadata.get("entry_distance")
            if (
                isinstance(metadata_strength, bool)
                or not isinstance(metadata_strength, (int, float))
                or not math.isfinite(float(metadata_strength))
                or isinstance(metadata_distance, bool)
                or not isinstance(metadata_distance, (int, float))
                or not math.isfinite(float(metadata_distance))
                or not math.isclose(
                    float(metadata_strength),
                    float(entry_strength),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(metadata_distance),
                    context_graph_service.distance_from_strength(
                        float(entry_strength)
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ContextPackagePublicIntegrityError(
                    "persisted chunk entry strength/distance drifts from its "
                    "writer seed facts"
                )
            entry_roles = raw_entry.get("roles")
            role_entry_strengths = metadata.get("entry_strengths")
            role_raw_entry_strengths = metadata.get("raw_entry_strengths")
            if (
                not isinstance(entry_roles, list)
                or any(
                    not isinstance(role, str) or not role.strip()
                    for role in entry_roles
                )
                or len(entry_roles) != len(set(entry_roles))
                or not isinstance(role_entry_strengths, dict)
                or not isinstance(role_raw_entry_strengths, dict)
                or set(role_entry_strengths)
                != set(role_raw_entry_strengths)
                or set(role_entry_strengths) != set(entry_roles)
                or not role_entry_strengths
            ):
                raise ContextPackagePublicIntegrityError(
                    "persisted chunk entry role seed facts are incomplete"
                )
            calibrated_values: list[float] = []
            for role in entry_roles:
                calibrated_value = role_entry_strengths.get(role)
                raw_value = role_raw_entry_strengths.get(role)
                if (
                    isinstance(calibrated_value, bool)
                    or not isinstance(calibrated_value, (int, float))
                    or not math.isfinite(float(calibrated_value))
                    or isinstance(raw_value, bool)
                    or not isinstance(raw_value, (int, float))
                    or not math.isfinite(float(raw_value))
                    or not 0.0 < float(calibrated_value) <= 1.0
                    or not 0.0 < float(raw_value) <= 1.0
                    or not math.isclose(
                        float(calibrated_value),
                        context_graph_service.calibrated_entry_seed_strength(
                            float(raw_value),
                            role,
                        ),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise ContextPackagePublicIntegrityError(
                        "persisted chunk entry role seed does not replay its "
                        "calibration writer formula"
                    )
                calibrated_values.append(float(calibrated_value))
            if not math.isclose(
                max(calibrated_values),
                float(entry_strength),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ContextPackagePublicIntegrityError(
                    "persisted chunk entry strength does not replay its role "
                    "seed maximum"
                )
        elif entry_layer in {"coarse", "mid"}:
            metadata = raw_entry.get("metadata")
            candidate_card = (
                metadata.get("candidate_card")
                if isinstance(metadata, dict)
                else None
            )
            expected_replay_proof = dict(
                (
                    (
                        replayed_entry_selection_audit.get("layers") or {}
                    ).get(entry_layer)
                    or {}
                ).get("frozen_replay_proofs_by_node")
                or {}
            ).get(entry_node_id)
            if not isinstance(candidate_card, dict) or not isinstance(
                expected_replay_proof,
                dict,
            ) or not expected_replay_proof:
                raise EntrySelectionTraceInvariantError(
                    "public entry candidate is missing its separately "
                    "persisted replay proof: "
                    f"layer={entry_layer!r}, node_id={entry_node_id!r}"
                )
            validate_entry_candidate_card(
                candidate_card,
                expected_entry_strength=float(entry_strength),
                expected_node_id=entry_node_id,
                expected_layer=entry_layer,
                expected_replay_proof=expected_replay_proof,
            )
        entry_key = (entry_layer, entry_node_id)
        normalized_entry_strength = float(entry_strength)
        if (
            entry_key in entry_strengths
            and not math.isclose(
                entry_strengths[entry_key],
                normalized_entry_strength,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace has conflicting entry strengths"
            )
        entry_strengths[entry_key] = normalized_entry_strength
    labels_by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_labels:
        if not isinstance(raw, dict):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path label is not an object"
            )
        edge_ids = raw.get("path_edge_ids")
        if not isinstance(edge_ids, list):
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path edge identities are invalid"
            )
        if not edge_ids:
            continue
        layer = str(raw.get("layer") or "")
        if layer not in layer_configs:
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path layer has no writer replay "
                f"authority: {layer!r}"
            )
        labels_by_layer[layer].append(raw)

    for layer, layer_labels in labels_by_layer.items():
        config = layer_configs[layer]
        frozen_hash = str(config["trace_hash"] or "")
        if len(frozen_hash) != 64:
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path has no frozen graph-state "
                f"identity for layer {layer!r}"
            )
        requested_edge_ids = {
            str(edge_id)
            for label in layer_labels
            for edge_id in label["path_edge_ids"]
        }
        edge_model = config["edge_model"]
        edges = list(
            db.scalars(
                select(edge_model).where(
                    edge_model.id.in_(requested_edge_ids)
                )
            ).all()
        )
        edges_by_id = {str(edge.id): edge for edge in edges}
        if set(edges_by_id) != requested_edge_ids:
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path references an unavailable "
                f"{layer} edge writer fact"
            )
        state_id_field = str(config["state_id_field"])
        requested_state_ids = {
            str(getattr(edge, state_id_field)) for edge in edges
        }
        state_model = config["state_model"]
        states = list(
            db.scalars(
                select(state_model).where(
                    state_model.id.in_(requested_state_ids)
                )
            ).all()
        )
        states_by_id = {str(state.id): state for state in states}
        if set(states_by_id) != requested_state_ids:
            raise ContextPackagePublicIntegrityError(
                "persisted retrieval trace path graph-state authority is "
                "unavailable"
            )

        for label in layer_labels:
            path = list(label["path"])
            edge_ids = [str(value) for value in label["path_edge_ids"]]
            edge_types = [str(value) for value in label["path_edge_types"]]
            distances = [float(value) for value in label["path_edge_distances"]]
            strengths = [float(value) for value in label["path_edge_strengths"]]
            root_node_id = str(label.get("root_node_id") or path[0])
            if root_node_id != path[0]:
                raise ContextPackagePublicIntegrityError(
                    "persisted retrieval trace path root drifts from its "
                    "writer entry"
                )
            entry_strength = entry_strengths.get((layer, root_node_id))
            if entry_strength is None:
                raise ContextPackagePublicIntegrityError(
                    "persisted retrieval trace path has no matching writer "
                    "entry strength"
                )
            for index, edge_id in enumerate(edge_ids):
                edge = edges_by_id[edge_id]
                state_id = str(getattr(edge, state_id_field))
                state = states_by_id[state_id]
                edge_state_hash = str(
                    getattr(edge, str(config["edge_hash_field"]), "") or ""
                )
                if (
                    str(state.knowledge_base_id) != str(trace.knowledge_base_id)
                    or str(state.state_hash or "") != frozen_hash
                    or edge_state_hash != frozen_hash
                    or (
                        layer == "chunk"
                        and str(edge.knowledge_base_id)
                        != str(trace.knowledge_base_id)
                    )
                ):
                    raise ContextPackagePublicIntegrityError(
                        "persisted retrieval trace path edge drifts from its "
                        "frozen graph-state writer authority"
                    )
                persisted_endpoints = {
                    str(getattr(edge, str(config["source_field"]))),
                    str(getattr(edge, str(config["target_field"]))),
                }
                trace_endpoints = {path[index], path[index + 1]}
                if (
                    persisted_endpoints != trace_endpoints
                    or str(edge.edge_type) != edge_types[index]
                ):
                    raise ContextPackagePublicIntegrityError(
                        "persisted retrieval trace path hop does not replay "
                        "its edge endpoints and type"
                    )
                writer_distance = context_graph_service._edge_distance(edge)
                writer_strength = (
                    context_graph_service._edge_calibrated_strength(edge)
                )
                if not (
                    math.isclose(
                        distances[index],
                        writer_distance,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        strengths[index],
                        writer_strength,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise ContextPackagePublicIntegrityError(
                        "persisted retrieval trace path distance or strength "
                        "does not replay its PostgreSQL edge writer fact"
                    )
            expected_distance_so_far = (
                context_graph_service.distance_from_strength(entry_strength)
            )
            for edge_distance in distances:
                expected_distance_so_far = round(
                    expected_distance_so_far + edge_distance,
                    6,
                )
            distance_so_far = label.get("distance_so_far")
            if (
                isinstance(distance_so_far, bool)
                or not isinstance(distance_so_far, (int, float))
                or not math.isclose(
                    float(distance_so_far),
                    expected_distance_so_far,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise ContextPackagePublicIntegrityError(
                    "persisted retrieval trace cumulative path distance does "
                    "not replay its edge writer facts"
                )


def _validate_trace_path_label_internal_support_refs(
    raw_support_refs: dict[str, Any],
) -> None:
    if "query_rq_seed_cards" in raw_support_refs:
        raw_cards = raw_support_refs["query_rq_seed_cards"]
        if not isinstance(raw_cards, list):
            raise ContextPackagePublicIntegrityError(
                "persisted query RQ seed cards are not a list"
            )
        try:
            cards = [
                RetrievalRQChunkSeedCard.model_validate(
                    card,
                    strict=True,
                )
                for card in raw_cards
            ]
        except (TypeError, ValidationError, ValueError) as exc:
            raise ContextPackagePublicIntegrityError(
                "persisted query RQ seed card is not a closed contract"
            ) from exc
        card_hashes = [card.card_hash for card in cards]
        if card_hashes != sorted(card_hashes) or len(card_hashes) != len(
            set(card_hashes)
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted query RQ seed cards are not unique and ordered"
            )
    if "rq" in raw_support_refs:
        try:
            GrayRQScore.model_validate(
                raw_support_refs["rq"],
                strict=True,
            )
        except (TypeError, ValidationError, ValueError) as exc:
            raise ContextPackagePublicIntegrityError(
                "persisted query RQ score is not a closed contract"
            ) from exc


def _closed_trace_path_label_payload(
    raw: Any,
    *,
    operating_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deserialize one persisted path label without dropping unknown facts."""

    if not isinstance(raw, dict):
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path label is not an object"
        )
    if "entry_parent_refs" not in raw or raw["entry_parent_refs"] is None:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path label has no explicit "
            "entry-parent authority"
        )
    unknown_fields = set(raw).difference(
        RetrievalPathLabel.model_fields,
        _TRACE_PATH_LABEL_INTERNAL_FIELDS,
    )
    if unknown_fields:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path label contains unknown fields: "
            + ", ".join(sorted(str(field) for field in unknown_fields))
        )

    _validate_trace_path_label_internal_fields(
        raw,
        operating_envelope=operating_envelope,
    )

    projected = {
        field: raw[field]
        for field in RetrievalPathLabel.model_fields
        if field in raw
    }
    try:
        raw_support_refs = projected.get("support_refs") or {}
        if not isinstance(raw_support_refs, dict):
            raise TypeError("path-label support references are not an object")
        unknown_support_fields = set(raw_support_refs).difference(
            RetrievalSupportRefs.model_fields,
            _TRACE_PATH_LABEL_INTERNAL_SUPPORT_FIELDS,
        )
        if unknown_support_fields:
            raise ValueError(
                "unknown path-label support-reference fields: "
                + ", ".join(
                    sorted(str(field) for field in unknown_support_fields)
                )
            )
        _validate_trace_path_label_internal_support_refs(
            raw_support_refs
        )
        projected["support_refs"] = RetrievalSupportRefs.model_validate(
            {
                field: raw_support_refs[field]
                for field in RetrievalSupportRefs.model_fields
                if field in raw_support_refs
            }
        ).model_dump(mode="json")
    except (TypeError, ValidationError, ValueError) as exc:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path-label support references are "
            "not a closed contract"
        ) from exc
    projected["entry_parent_refs"] = [
        item.model_dump(mode="json")
        for item in _closed_canonical_entry_parent_refs(
            projected["entry_parent_refs"]
        )
    ]
    try:
        return RetrievalPathLabel.model_validate(projected).model_dump(
            mode="json"
        )
    except (TypeError, ValidationError) as exc:
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path label is not a closed contract"
        ) from exc


def _validate_context_package_trace_path_binding(
    db: Session,
    *,
    package: ContextPackage,
    diagnostics: dict[str, Any],
) -> None:
    """Replay package path facts against their PostgreSQL trace authority."""

    trace_id = str(package.retrieval_trace_id or "").strip()
    if not trace_id:
        raise ContextPackagePublicIntegrityError(
            "persisted context package has no retrieval trace binding"
        )
    trace = db.get(RetrievalTrace, trace_id)
    if trace is None:
        raise ContextPackagePublicIntegrityError(
            "persisted context package retrieval trace is unavailable"
        )
    if trace.knowledge_base_id != package.knowledge_base_id:
        raise ContextPackagePublicIntegrityError(
            "persisted context package identity drifts from its retrieval trace"
        )

    raw_labels = trace.path_labels_json
    if not isinstance(raw_labels, list):
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path labels are not a closed list"
        )
    operating_envelope = _validated_frozen_trace_operating_envelope(
        trace
    )
    labels = [
        _closed_trace_path_label_payload(
            label,
            operating_envelope=operating_envelope,
        )
        for label in raw_labels
    ]
    _validate_trace_path_edge_writer_facts(
        db,
        trace=trace,
        raw_labels=raw_labels,
    )
    expected_graph_path_ids = list(
        dict.fromkeys(
            str(edge_id)
            for label in labels
            for edge_id in (label.get("path_edge_ids") or [])
            if str(edge_id)
        )
    )
    if list(package.graph_path_ids_json or []) != expected_graph_path_ids:
        raise ContextPackagePublicIntegrityError(
            "persisted context package graph path ids drift from the "
            "retrieval trace physical edge scope"
        )

    raw_contributions = diagnostics.get("reached_by_paths") or []
    if not isinstance(raw_contributions, list):
        raise ContextPackagePublicIntegrityError(
            "persisted context package path contributions are not a list"
        )
    try:
        contributions = [
            RetrievalPathContribution.model_validate(item)
            for item in raw_contributions
        ]
    except (TypeError, ValidationError) as exc:
        raise ContextPackagePublicIntegrityError(
            "persisted context package path contribution cannot be replayed"
        ) from exc

    for contribution in contributions:
        matching_labels: list[dict[str, Any]] = []
        contribution_trace_variants: list[
            tuple[list[str], list[str], tuple[str, str, str | None] | None]
        ] = []
        if contribution.origin_parent_node_id:
            if not contribution.origin_parent_layer:
                raise ContextPackagePublicIntegrityError(
                    "persisted context package entry-parent prefix has no "
                    "origin layer"
                )
            contribution_trace_variants.append(
                (
                    list(contribution.path[1:]),
                    list(contribution.path_edge_types[1:]),
                    (
                        contribution.origin_parent_layer,
                        contribution.origin_parent_node_id,
                        (
                            contribution.path_edge_types[0]
                            if contribution.path_edge_types
                            else None
                        ),
                    ),
                )
            )
        else:
            contribution_trace_variants.append(
                (
                    list(contribution.path),
                    list(contribution.path_edge_types),
                    None,
                )
            )
        for label in labels:
            label_path = [str(value) for value in (label.get("path") or [])]
            label_edge_ids = [
                str(value)
                for value in (label.get("path_edge_ids") or [])
            ]
            label_node_id = str(
                label.get("node_id")
                or label.get("chunk_id")
                or (label_path[-1] if label_path else "")
            )
            label_root_node_id = str(
                label.get("root_node_id")
                or (label_path[0] if label_path else "")
            )
            label_layer = str(label.get("layer") or contribution.layer)
            label_edge_types = [
                str(value) for value in (label.get("path_edge_types") or [])
            ]
            label_entry_parent_refs = _closed_canonical_entry_parent_refs(
                label.get("entry_parent_refs")
            )
            if (
                label_layer != contribution.layer
                or label_node_id != contribution.node_id
                or label_root_node_id != contribution.root_node_id
                or label_edge_ids != contribution.path_edge_ids
            ):
                continue
            for (
                trace_path,
                trace_edge_types,
                required_entry_parent,
            ) in contribution_trace_variants:
                if (
                    label_path != trace_path
                    or label_edge_types != trace_edge_types
                ):
                    continue
                if (
                    required_entry_parent is None
                    and label_entry_parent_refs
                ):
                    continue
                if required_entry_parent is not None:
                    (
                        required_parent_layer,
                        required_parent_node_id,
                        required_edge_type,
                    ) = required_entry_parent
                    if not any(
                        ref.parent_layer == required_parent_layer
                        and ref.parent_node_id == required_parent_node_id
                        and ref.edge_type == required_edge_type
                        and _support_refs_include(
                            contribution.support_refs,
                            ref.support_refs,
                        )
                        for ref in label_entry_parent_refs
                    ):
                        continue
                matching_labels.append(label)
                break
        if not matching_labels:
            raise ContextPackagePublicIntegrityError(
                "persisted context package contribution is not backed by "
                "a retrieval trace physical path"
            )

        expected_facets = sorted(
            {
                str(value)
                for label in matching_labels
                for value in (label.get("covered_facets") or [])
                if str(value)
            }
        )
        expected_roles = sorted(
            {
                str(value)
                for label in matching_labels
                for value in (label.get("evidence_roles") or [])
                if str(value)
            }
        )
        if (
            contribution.covered_facets != expected_facets
            or contribution.evidence_roles != expected_roles
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package contribution semantics drift "
                "from its retrieval trace path"
            )
        if any("distance_so_far" in label for label in matching_labels):
            expected_distance = round(
                min(
                    max(0.0, float(label.get("distance_so_far") or 0.0))
                    for label in matching_labels
                ),
                6,
            )
            if contribution.distance_so_far != expected_distance:
                raise ContextPackagePublicIntegrityError(
                    "persisted context package contribution distance drifts "
                    "from its retrieval trace path"
                )
        if any("reward_so_far" in label for label in matching_labels):
            expected_reward = round(
                max(
                    max(0.0, float(label.get("reward_so_far") or 0.0))
                    for label in matching_labels
                ),
                6,
            )
            if contribution.reward_so_far != expected_reward:
                raise ContextPackagePublicIntegrityError(
                    "persisted context package contribution reward drifts "
                    "from its retrieval trace path"
                )


def _context_identity_equal(left: Any, right: Any) -> bool:
    return cache_manager.strict_json_sha256(
        {"value": left}
    ) == cache_manager.strict_json_sha256({"value": right})


def _validate_persisted_context_package_ownership(
    db: Session,
    *,
    package: ContextPackage,
    package_json: Any,
    persisted_citation_spans: Any,
) -> list[dict[str, Any]]:
    """Bind every raw evidence address to the outer package/trace authority."""

    if (
        not isinstance(package_json, dict)
        or set(package_json) != {"chunks"}
        or not isinstance(package_json.get("chunks"), list)
    ):
        raise ContextPackagePublicIntegrityError(
            "persisted context package document is not a closed contract"
        )
    trace_id = str(package.retrieval_trace_id or "").strip()
    if not trace_id:
        raise ContextPackagePublicIntegrityError(
            "persisted context package has no retrieval trace identity"
        )
    trace = db.get(RetrievalTrace, trace_id)
    if trace is None or str(trace.knowledge_base_id) != str(
        package.knowledge_base_id
    ):
        raise ContextPackagePublicIntegrityError(
            "persisted context package trace is missing or cross-KB"
        )
    if not isinstance(persisted_citation_spans, list):
        raise ContextPackagePublicIntegrityError(
            "persisted context package citation spans are not a list"
        )

    raw_chunks = list(package_json["chunks"])
    validated_chunks: list[dict[str, Any]] = []
    for index, raw_chunk in enumerate(raw_chunks):
        try:
            chunk = ContextPackageChunk.model_validate(
                raw_chunk,
                strict=True,
            ).model_dump(mode="json")
        except (TypeError, ValidationError, ValueError) as exc:
            raise ContextPackagePublicIntegrityError(
                "persisted context package raw chunk is not a closed "
                f"contract at index {index}"
            ) from exc
        source_span = chunk["source_span"]
        if (
            chunk["context_package_id"] != str(package.id)
            or source_span["context_package_id"] != str(package.id)
            or source_span["retrieval_trace_id"] != trace_id
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package raw span ownership mismatch"
            )
        identity_pairs = (
            (chunk["chunk_id"], source_span["chunk_id"]),
            (
                chunk["document_version_id"],
                source_span["document_version_id"],
            ),
            (chunk["source_path"], source_span["source_path"]),
            (
                chunk["logical_source_path"],
                source_span["logical_source_path"],
            ),
            (
                chunk["chunk_text_hash_protocol_version"],
                source_span["chunk_text_hash_protocol_version"],
            ),
            (chunk["chunk_text_hash"], source_span["chunk_text_hash"]),
            (
                chunk["raw_span_text_hash_protocol_version"],
                source_span["raw_span_text_hash_protocol_version"],
            ),
            (
                chunk["raw_span_text_hash"],
                source_span["raw_span_text_hash"],
            ),
            (chunk["char_span"], source_span["char_span"]),
            (
                chunk["raw_chunk_char_span"],
                source_span["raw_chunk_char_span"],
            ),
            (chunk["page_range"], source_span["page_range"]),
            (chunk["section_path"], source_span["section_path"]),
            (chunk["structure_path"], source_span["structure_path"]),
            (
                chunk["structure_node_ids"],
                source_span["structure_node_ids"],
            ),
            (chunk["bbox"], source_span["bbox"]),
            (chunk["content_clipped"], source_span["content_clipped"]),
            (
                chunk["content_token_count"],
                source_span["content_token_count"],
            ),
        )
        if any(
            not _context_identity_equal(left, right)
            for left, right in identity_pairs
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package raw chunk address drifts from "
                "its source span"
            )
        persisted_chunk = db.get(Chunk, chunk["chunk_id"])
        persisted_document = db.get(Document, chunk["document_id"])
        persisted_version = db.get(
            DocumentVersion,
            chunk["document_version_id"],
        )
        if (
            persisted_chunk is None
            or persisted_document is None
            or persisted_version is None
            or str(persisted_chunk.knowledge_base_id)
            != str(package.knowledge_base_id)
            or str(persisted_document.knowledge_base_id)
            != str(package.knowledge_base_id)
            or str(persisted_chunk.document_id) != chunk["document_id"]
            or str(persisted_chunk.document_version_id)
            != chunk["document_version_id"]
            or str(persisted_version.document_id) != chunk["document_id"]
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package chunk escapes its PostgreSQL "
                "knowledge-base provenance"
            )
        snapshot = source_span["source_snapshot_verification"]
        database_pairs = (
            (chunk["document_title"], persisted_document.title),
            (source_span["source_path"], persisted_version.storage_path),
            (source_span["source_checksum"], persisted_version.checksum),
            (
                source_span["logical_source_path"],
                persisted_document.source_path,
            ),
            (source_span["chunk_text_hash"], persisted_chunk.text_hash),
            (snapshot["storage_path"], persisted_version.storage_path),
            (snapshot["checksum"], persisted_version.checksum),
            (
                source_span["raw_chunk_char_span"],
                [persisted_chunk.char_start, persisted_chunk.char_end],
            ),
        )
        if any(
            not _context_identity_equal(left, right)
            for left, right in database_pairs
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package source span drifts from its "
                "PostgreSQL chunk/document provenance"
            )
        span_start, span_end = source_span["char_span"]
        relative_start = int(span_start) - int(persisted_chunk.char_start)
        relative_end = int(span_end) - int(persisted_chunk.char_start)
        expected_content = str(persisted_chunk.text or "")[
            relative_start:relative_end
        ]
        if (
            relative_start < 0
            or relative_end < relative_start
            or int(span_end) > int(persisted_chunk.char_end)
            or chunk["content"] != expected_content
            or source_span["raw_span_text_hash"]
            != hashlib.sha256(
                expected_content.encode("utf-8")
            ).hexdigest()
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package content does not replay its raw "
                "PostgreSQL chunk span"
            )
        # Keep the already-validated writer JSON byte-shape for the persisted
        # citation projection comparison.  Pydantic's model dump would add
        # optional default keys (for example an empty bbox object), turning a
        # valid writer row into a different persistence fact.
        validated_chunks.append(deepcopy(raw_chunk))

    if len(persisted_citation_spans) != len(validated_chunks):
        raise ContextPackagePublicIntegrityError(
            "persisted context package citation cardinality mismatch"
        )
    for citation in persisted_citation_spans:
        if not isinstance(citation, dict):
            raise ContextPackagePublicIntegrityError(
                "persisted context package citation span is not an object"
            )
        if (
            citation.get("context_package_id") != str(package.id)
            or citation.get("retrieval_trace_id") != trace_id
        ):
            raise ContextPackagePublicIntegrityError(
                "persisted context package citation ownership mismatch"
            )
    return validated_chunks


def get_context_package(db: Session, package_id: str) -> dict | None:
    package = db.get(ContextPackage, package_id)
    if package is None:
        return None
    package_json = package.package_json or {}
    persisted_citation_spans = package.citation_spans_json
    raw_chunks = _validate_persisted_context_package_ownership(
        db,
        package=package,
        package_json=package_json,
        persisted_citation_spans=persisted_citation_spans,
    )
    expected_persisted_citation_spans = [
        {
            **(item.get("source_span") or {}),
            "document_id": item["document_id"],
            "document_title": item.get("document_title") or "",
            "source_path": item.get("source_path") or "",
            "logical_source_path": (
                item.get("logical_source_path") or ""
            ),
            "section_path": item.get("section_path"),
            "structure_path": item.get("structure_path"),
            "structure_node_ids": (
                item.get("structure_node_ids") or []
            ),
            "structure_closure": (
                item.get("structure_closure") or {}
            ),
        }
        for item in raw_chunks
    ]
    if cache_manager.strict_json_sha256(
        {"citation_spans": persisted_citation_spans}
    ) != cache_manager.strict_json_sha256(
        {"citation_spans": expected_persisted_citation_spans}
    ):
        raise ContextPackagePublicIntegrityError(
            "persisted citation spans drift from the canonical package "
            "chunk projection"
        )

    raw_graph_expansion_paths = [
        {"kind": "concept_path", "path": package.concept_path_json or []},
        {"kind": "graph_path_ids", "edge_ids": package.graph_path_ids_json or []},
        {"kind": "restored_chunks", "chunk_ids": package.restored_chunk_ids_json or []},
        {"kind": "bridge_chunks", "chunk_ids": package.bridge_chunk_ids_json or []},
        {"kind": "parent_structure_nodes", "node_ids": package.parent_structure_node_ids_json or []},
    ]
    raw_diagnostics = dict(package.diagnostics_json or {})
    _validate_context_package_trace_path_binding(
        db,
        package=package,
        diagnostics=raw_diagnostics,
    )
    try:
        public_chunks = [
            ContextPackageChunk.model_validate(
                _public_context_package_chunk(
                    item,
                    package_id=package.id,
                )
            ).model_dump(mode="json")
            for item in raw_chunks
        ]
        public_package = ContextPackageDocument.model_validate(
            {"chunks": public_chunks}
        ).model_dump(mode="json")
        public_contexts = [
            ContextItem.model_validate(
                _public_context_item(
                    item,
                    package_id=package.id,
                )
            ).model_dump(mode="json")
            for item in raw_chunks
        ]
        public_citation_spans = [
            ContextCitationSpan.model_validate(
                {
                    "document_id": item["document_id"],
                    "document_title": item["document_title"],
                    "source_path": item["source_path"],
                    "logical_source_path": (
                        item["logical_source_path"]
                    ),
                    "section_path": item["section_path"],
                    "structure_path": item["structure_path"],
                    "structure_node_ids": (
                        item["structure_node_ids"]
                    ),
                    "structure_closure": (
                        item["structure_closure"]
                    ),
                    "source_span": item["source_span"],
                }
            ).model_dump(mode="json")
            for item in public_chunks
        ]
        public_concept_path = TypeAdapter(
            list[ContextConceptPathEntry]
        ).dump_python(
            TypeAdapter(
                list[ContextConceptPathEntry]
            ).validate_python(package.concept_path_json or []),
            mode="json",
        )
        graph_expansion_adapter = TypeAdapter(
            list[ContextGraphExpansionPath]
        )
        graph_expansion_paths = (
            graph_expansion_adapter.dump_python(
                graph_expansion_adapter.validate_python(
                    raw_graph_expansion_paths
                ),
                mode="json",
            )
        )
        public_why_selected = {
            str(chunk_id): ContextSelectionReason.model_validate(
                value
            ).model_dump(mode="json")
            for chunk_id, value in (
                package.why_selected_json or {}
            ).items()
        }
        public_diagnostics = ContextPackageDiagnostics.model_validate(
            {
                key: raw_diagnostics.get(key)
                for key in (
                    "context_restoration_protocol",
                    "repair_protocol_version",
                    "repair_action_type",
                    "repair_executor_mechanism",
                    "repair_gray_zone_model_call_count",
                    "repair_gray_zone_decision_authority",
                    "retrieval_granularity",
                    "conversation_state_scope_hash",
                    "conversation_state_is_evidence",
                    "runtime_settings_hash",
                    "profile_hash",
                    "path_summary",
                    "dedupe_keys",
                    "restore_counts",
                    "token_budget_audit",
                    "snapshot_integrity",
                )
            }
        ).model_dump(mode="json")
    except (KeyError, TypeError, ValidationError) as exc:
        raise ContextPackagePublicIntegrityError(
            "persisted context package cannot satisfy the public "
            f"contract: {type(exc).__name__}:{str(exc)[:240]}"
        ) from exc

    public_payload = {
        "contract_version": "context_package_public_v1",
        "id": package.id,
        "knowledge_base_id": package.knowledge_base_id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "query": package.query,
        "hit_chunk_ids": package.hit_chunk_ids_json or [],
        "restored_chunk_ids": package.restored_chunk_ids_json or [],
        "bridge_chunk_ids": package.bridge_chunk_ids_json or [],
        "parent_structure_node_ids": package.parent_structure_node_ids_json or [],
        "concept_path": public_concept_path,
        "graph_path_ids": package.graph_path_ids_json or [],
        "reached_by_paths": [
            _public_path_contribution(item)
            for item in (raw_diagnostics.get("reached_by_paths") or [])
        ],
        "node_contributions": [
            _public_node_contribution(item)
            for item in (raw_diagnostics.get("node_contributions") or [])
        ],
        "why_selected": public_why_selected,
        "cycle_convergence_score": package.cycle_convergence_score,
        "dedupe_keys": package.dedupe_keys_json or [],
        "covered_facets": package.covered_facets_json or [],
        "package": public_package,
        "contexts": public_contexts,
        "token_budget": package.token_budget,
        "token_count": package.token_count,
        "citation_spans": public_citation_spans,
        "graph_expansion_paths": graph_expansion_paths,
        "diagnostics": public_diagnostics,
    }
    public_projection = {
        field: public_payload[field]
        for field in CONTEXT_PACKAGE_PUBLIC_HASH_FIELDS
    }
    hash_card = {
        "protocol_version": "context_package_public_hash_v1",
        "canonicalization": "json_utf8_sort_keys_compact_v1",
        "hashed_public_fields": list(
            CONTEXT_PACKAGE_PUBLIC_HASH_FIELDS
        ),
        "public_payload_hash": cache_manager.strict_json_sha256(
            public_projection
        ),
        "public_citation_spans_hash": (
            cache_manager.strict_json_sha256(
                {"citation_spans": public_citation_spans}
            )
        ),
        "citation_spans_consistency": (
            "persisted_equals_public_projection"
        ),
        "chunk_count": len(public_chunks),
        "citation_span_count": len(public_citation_spans),
        "graph_expansion_path_count": len(
            graph_expansion_paths
        ),
    }
    response_payload = {
        **public_payload,
        "package_hash": cache_manager.strict_json_sha256(hash_card),
        "package_hash_card": hash_card,
        "created_at": package.created_at,
    }
    try:
        return ContextPackageResponse.model_validate(
            response_payload
        ).model_dump(mode="json")
    except ValidationError as exc:
        raise ContextPackagePublicIntegrityError(
            "context package public hash proof failed validation: "
            f"{str(exc)[:320]}"
        ) from exc


def _public_entry_node(
    raw: Any,
    *,
    expected_replay_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {}
    metadata = dict(item.get("metadata") or {})
    candidate_card = None
    if isinstance(metadata.get("candidate_card"), dict):
        raw_candidate_card = validate_entry_candidate_card(
            dict(metadata["candidate_card"]),
            expected_entry_strength=item.get("entry_strength"),
            expected_node_id=str(item.get("node_id") or ""),
            expected_layer=str(item.get("layer") or ""),
            expected_replay_proof=expected_replay_proof,
        )
        candidate_card = RetrievalEntryCandidateCard.model_validate(
            {
                key: raw_candidate_card[key]
                for key in RetrievalEntryCandidateCard.model_fields
                if key in raw_candidate_card
            }
        ).model_dump(mode="json")
    return {
        "layer": str(item.get("layer") or ""),
        "node_id": str(item.get("node_id") or ""),
        "entry_strength": item.get("entry_strength"),
        "roles": list(item.get("roles") or []),
        "rq_prefix_id": item.get("rq_prefix_id"),
        "metadata": {
            **{
                key: metadata.get(key)
                for key in (
                    "label",
                    "node_type",
                    "rq_path_prefix",
                    "representative_terms",
                )
                if metadata.get(key) is not None
            },
            **(
                {"candidate_card": candidate_card}
                if candidate_card is not None
                else {}
            ),
        },
    }


def _closed_contract_fields(model: Any, raw: Any) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {}
    return {key: item[key] for key in model.model_fields if key in item}


def _public_support_refs(raw: Any) -> dict[str, Any]:
    """Project internal traversal bookkeeping onto the evidence-only public contract."""

    return RetrievalSupportRefs.model_validate(
        _closed_contract_fields(RetrievalSupportRefs, raw)
    ).model_dump(mode="json")


def _public_entry_parent_ref(raw: Any) -> dict[str, Any]:
    projected = _closed_contract_fields(RetrievalEntryParentRef, raw)
    projected["support_refs"] = _public_support_refs(projected.get("support_refs"))
    return RetrievalEntryParentRef.model_validate(projected).model_dump(mode="json")


def _public_traversal_state(raw: Any) -> dict[str, Any]:
    projected = _closed_contract_fields(RetrievalTraversalState, raw)
    projected["support_refs"] = _public_support_refs(projected.get("support_refs"))
    projected["entry_support_refs"] = _public_support_refs(projected.get("entry_support_refs"))
    projected["entry_parent_refs"] = [
        _public_entry_parent_ref(item) for item in (projected.get("entry_parent_refs") or [])
    ]
    return RetrievalTraversalState.model_validate(projected).model_dump(mode="json")


def _public_frontier_snapshot(raw: Any) -> dict[str, Any]:
    projected = _closed_contract_fields(RetrievalFrontierSnapshot, raw)
    projected["popped"] = _public_traversal_state(projected.get("popped"))
    return RetrievalFrontierSnapshot.model_validate(projected).model_dump(mode="json")


def _public_path_label(
    raw: Any,
    *,
    operating_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _closed_trace_path_label_payload(
        raw,
        operating_envelope=operating_envelope,
    )


def _public_path_contribution(raw: Any) -> dict[str, Any]:
    projected = _closed_contract_fields(RetrievalPathContribution, raw)
    projected["support_refs"] = _public_support_refs(projected.get("support_refs"))
    return RetrievalPathContribution.model_validate(projected).model_dump(mode="json")


def _public_node_contribution(raw: Any) -> dict[str, Any]:
    projected = _closed_contract_fields(RetrievalNodeContributionSummary, raw)
    projected["reached_by_paths"] = [
        _public_path_contribution(item) for item in (projected.get("reached_by_paths") or [])
    ]
    return RetrievalNodeContributionSummary.model_validate(projected).model_dump(mode="json")


def _public_step_input(raw: Any) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {}

    def ids(rows: Any) -> list[str]:
        return [
            str(row.get("node_id"))
            for row in (rows or [])
            if isinstance(row, dict) and row.get("node_id")
        ]

    query_rq_seed_audit = None
    if isinstance(item.get("query_rq_seed_audit"), dict):
        query_rq_seed_audit = (
            RetrievalQueryRQSeedAudit.model_validate(
                item["query_rq_seed_audit"]
            ).model_dump(mode="json")
        )
    return {
        "entry_node_ids": ids(item.get("entry_nodes")),
        "coarse_entry_ids": [str(value) for value in (item.get("coarse_entry_ids") or [])],
        "mid_entry_ids": [str(value) for value in (item.get("mid_entry_ids") or [])],
        "rq_membership_entry_ids": ids(item.get("rq_membership_entries")),
        "query_rq_path": [int(value) for value in (item.get("query_rq_path") or [])],
        "result_chunk_ids": [str(value) for value in (item.get("result_chunk_ids") or [])],
        "hit_chunk_ids": [str(value) for value in (item.get("hit_chunk_ids") or [])],
        "token_budget": item.get("token_budget"),
        "query_rq_seed_audit": query_rq_seed_audit,
    }


def _public_step_output(raw: Any) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {}

    def node_ids(rows: Any) -> list[str]:
        result: list[str] = []
        for row in rows or []:
            if isinstance(row, dict):
                value = row.get("node_id") or row.get("chunk_id")
            else:
                value = row
            if value:
                result.append(str(value))
        return result

    accepted_chunks = item.get("accepted_chunks") or {}
    convergence = dict(item.get("convergence") or {})
    return {
        "accepted_node_ids": node_ids(item.get("accepted_nodes")),
        "selected_node_ids": node_ids(
            item.get("selected_entry_nodes") or item.get("selected_rq_memberships")
        ),
        "accepted_chunk_ids": (
            [str(value) for value in accepted_chunks]
            if isinstance(accepted_chunks, dict)
            else node_ids(accepted_chunks)
        ),
        "restored_chunk_count": item.get("restored_chunk_count"),
        "context_package_id": item.get("context_package_id"),
        "source_span_count": len(item.get("source_spans") or []),
        "convergence_reason": convergence.get("reason"),
    }


def _public_trace_diagnostics(
    raw: Any, *, retrieval_granularity: RetrievalGranularity
) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {}
    validate_entry_selection_trace_diagnostics(item)
    allowed = (
        "context_graph_state_id",
        "active_profile_hash",
        "canonical_active_profile_hash",
        "repair_protocol_version",
        "repair_action_type",
        "repair_executor_mechanism",
        "repair_directive_hash",
        "repair_gray_zone_decision_authority",
        "repair_gray_zone_model_call_count",
        "coarse_skipped_reason",
        "runtime_settings_hash",
        "gray_zone_runtime_settings_identity_protocol_version",
        "gray_zone_runtime_settings_hash",
        "gray_zone_query_facet_protocol_version",
        "gray_zone_query_facet_hash",
        "gray_zone_external_routing_packet_used",
        "gray_zone_request_scoped_budget_in_identity",
        "agent_operating_envelope",
        "agent_operating_envelope_hash",
        "effective_traversal_protocol_hash",
        "result_top_k",
        "traversal_protocol",
        "rank_score_protocol_version",
        "rank_score_protocol_hash",
        "raw_strength_protocol_version",
        "raw_strength_protocol_hash",
        "chunk_node_quality_protocol_version",
        "chunk_node_quality_protocol_hash",
        "out_evidence_mass_protocol_version",
        "out_evidence_mass_protocol_hash",
        "in_acceptance_capacity_protocol_version",
        "in_acceptance_capacity_protocol_hash",
        "relation_quota_protocol_version",
        "relation_quota_protocol_hash",
        "edge_type_calibration_protocol_version",
        "edge_type_calibration_protocol_hash",
        "graph_operating_point_hash",
        "calibration_params_hash",
        "edge_type_calibration_config_hash",
        "edge_distance_protocol_version",
        "edge_distance_protocol_hash",
        "entry_selection_protocol_version",
        "entry_selection_protocol_hash",
        "entry_topology_protocol_version",
        "entry_topology_protocol_hash",
        "entry_replay_proof_protocol_version",
        "entry_dense_replay_protocol_version",
        "entry_dense_replay_input_hash",
        "entry_neutral_start_cost_protocol_version",
        "entry_selection_model_call_count",
        "entry_selection_lexical_overlap_used_as_query_relevance",
        "entry_selection_topology_used_as_path_distance",
        "entry_selection_node_weight_used_as_query_relevance",
        "entry_neutral_start_cost_is_query_relevance",
        "entry_selection_gray_zone_rule_inputs_modified",
        "query_rq_seed_audit",
    )
    return {
        **{key: item.get(key) for key in allowed if item.get(key) is not None},
        "retrieval_granularity": retrieval_granularity,
        "scores_json_retired_as_primary_audit": True,
    }


def validate_entry_selection_trace_diagnostics(
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed before projecting persisted entry-selection authority."""

    item = dict(raw or {})
    expected = {
        "entry_selection_protocol_version": ENTRY_SELECTION_PROTOCOL_VERSION,
        "entry_selection_protocol_hash": entry_selection_protocol_hash(),
        "entry_topology_protocol_version": ENTRY_TOPOLOGY_PROTOCOL_VERSION,
        "entry_topology_protocol_hash": entry_topology_protocol_hash(),
        "entry_replay_proof_protocol_version": ENTRY_REPLAY_PROOF_PROTOCOL_VERSION,
        "entry_neutral_start_cost_protocol_version": (
            ENTRY_NEUTRAL_START_COST_PROTOCOL_VERSION
        ),
        "entry_selection_model_call_count": 0,
        "entry_selection_lexical_overlap_used_as_query_relevance": False,
        "entry_selection_topology_used_as_path_distance": False,
        "entry_selection_node_weight_used_as_query_relevance": False,
        "entry_neutral_start_cost_is_query_relevance": False,
        "entry_selection_gray_zone_rule_inputs_modified": False,
        "repair_gray_zone_decision_authority": False,
        "repair_gray_zone_model_call_count": 0,
        "typed_action_gray_zone_decision_authority": False,
        "typed_action_gray_zone_model_call_count": 0,
    }
    failures = [
        field_name
        for field_name, expected_value in expected.items()
        if item.get(field_name) != expected_value
    ]
    try:
        audit = validate_entry_selection_trace_audit(
            dict(item.get("entry_selection_audit") or {})
        )
    except EntrySelectionTraceInvariantError:
        failures.append("entry_selection_audit")
        audit = {}
    has_frozen_proofs = any(
        bool(
            (((audit.get("layers") or {}).get(layer) or {}).get(
                "frozen_replay_proofs_by_node"
            ))
        )
        for layer in ("coarse", "mid")
    )
    if has_frozen_proofs:
        dense_packet = item.get("entry_dense_replay_input")
        if not isinstance(dense_packet, dict):
            failures.append("entry_dense_replay_input")
            dense_packet = {}
        if (
            item.get("entry_dense_replay_protocol_version")
            != ENTRY_DENSE_REPLAY_PROTOCOL_VERSION
            or dense_packet.get("protocol_version")
            != ENTRY_DENSE_REPLAY_PROTOCOL_VERSION
        ):
            failures.append("entry_dense_replay_protocol_version")
        dense_input_hash = str(item.get("entry_dense_replay_input_hash") or "")
        if (
            len(dense_input_hash) != 64
            or dense_packet.get("input_hash") != dense_input_hash
            or audit.get("dense_replay_input_hash") != dense_input_hash
        ):
            failures.append("entry_dense_replay_input_hash")
    if failures:
        raise EntrySelectionTraceInvariantError(
            "persisted entry-selection diagnostics are non-replayable: "
            + ", ".join(sorted(set(failures)))
        )
    return audit


def get_retrieval_trace_steps(db: Session, trace_id: str) -> dict | None:
    from app.models import GraphRetrievalStep

    trace = db.get(RetrievalTrace, trace_id)
    if trace is None:
        return None
    context_package = db.scalar(
        select(ContextPackage)
        .where(ContextPackage.retrieval_trace_id == trace_id)
        .order_by(ContextPackage.created_at.desc())
        .limit(1)
    )
    diagnostics = trace.diagnostics_json or {}
    convergence = trace.convergence_json or {}
    operating_envelope = _validated_frozen_trace_operating_envelope(
        trace
    )
    raw_path_labels = trace.path_labels_json or []
    if not isinstance(raw_path_labels, list):
        raise ContextPackagePublicIntegrityError(
            "persisted retrieval trace path labels are not a closed list"
        )
    public_path_labels = [
        _public_path_label(
            item,
            operating_envelope=operating_envelope,
        )
        for item in raw_path_labels
    ]
    _validate_trace_path_edge_writer_facts(
        db,
        trace=trace,
        raw_labels=raw_path_labels,
    )
    steps = db.scalars(
        select(GraphRetrievalStep)
        .where(GraphRetrievalStep.retrieval_trace_id == trace_id)
        .order_by(GraphRetrievalStep.step_index.asc())
    ).all()
    path_distance_decisions = [
        decision
        for step in steps
        for decision in (step.gray_zone_path_decisions_json or [])
    ]
    gray_zone_audit = validate_retrieval_gray_zone_trace(
        records=path_distance_decisions,
        convergence=convergence,
        diagnostics=diagnostics,
        traversal_protocol_hash=trace.traversal_protocol_hash,
        runtime_settings_hash=str(
            diagnostics.get("gray_zone_runtime_settings_hash") or ""
        ),
        agent_operating_envelope_hash=trace.agent_operating_envelope_hash,
        operating_envelope=operating_envelope,
    )
    gray_zone_decisions = list(gray_zone_audit.pop("local_records"))
    path_distance_threshold_hits = list(gray_zone_audit.pop("partition_records"))
    persisted_gray_zone_protocol = str(gray_zone_audit.pop("persisted_protocol"))
    gray_zone_model_call_count = convergence["gray_zone_model_call_count"]
    authority_audit = diagnostics.get("gray_zone_authority_audit")
    if not isinstance(authority_audit, dict):
        raise RetrievalTraceAuditError(
            {"status": "incomplete", "issues": [{"code": "missing_gray_zone_authority_audit"}]}
        )
    if (
        authority_audit.get("gray_zone_decision_authority")
        != "deterministic_executor_only"
        or authority_audit.get("gray_zone_model_call_count") != 0
        or authority_audit.get("gray_zone_rule_protocol_version")
        != persisted_gray_zone_protocol
        or authority_audit.get("external_routing_packet_used") is not False
        or authority_audit.get("request_scoped_budget_used_by_gray_identity") is not False
    ):
        raise RetrievalTraceAuditError(
            {"status": "failed", "conflicts": [{"code": "conflicting_gray_zone_authority_audit"}]}
        )
    gray_zone_decision_authority = "executor_local_deterministic_only"
    retrieval_granularity: RetrievalGranularity = (
        (trace.diagnostics_json or {}).get("retrieval_granularity", "mid")
    )
    validate_semantic_entry_query_trace_packet(
        diagnostics.get("semantic_entry_query"),
        query=str(trace.query),
        query_facets=dict(trace.query_facets_json or {}),
    )
    entry_selection_audit = validate_entry_topology_replay_against_database(
        db,
        knowledge_base_id=str(trace.knowledge_base_id),
        raw_audit=validate_entry_selection_trace_diagnostics(
            dict(diagnostics)
        ),
        query_facets=dict(trace.query_facets_json or {}),
    )
    entry_selection_audit = validate_entry_dense_replay_against_database(
        db,
        knowledge_base_id=str(trace.knowledge_base_id),
        query=str(trace.query),
        query_facets=dict(trace.query_facets_json or {}),
        raw_audit=entry_selection_audit,
        raw_dense_replay_input=dict(
            diagnostics.get("entry_dense_replay_input") or {}
        ),
    )

    def public_entry_node(raw_entry: Any) -> dict[str, Any]:
        entry = dict(raw_entry or {}) if isinstance(raw_entry, dict) else {}
        layer = str(entry.get("layer") or "")
        node_id = str(entry.get("node_id") or "")
        expected_replay_proof = dict(
            (
                (
                    (entry_selection_audit.get("layers") or {}).get(layer)
                    or {}
                ).get("frozen_replay_proofs_by_node")
                or {}
            ).get(node_id)
            or {}
        )
        if isinstance((entry.get("metadata") or {}).get("candidate_card"), dict):
            if not expected_replay_proof:
                raise EntrySelectionTraceInvariantError(
                    "public entry candidate is missing its separately persisted replay proof: "
                    f"layer={layer!r}, node_id={node_id!r}"
                )
        return _public_entry_node(
            entry,
            expected_replay_proof=(expected_replay_proof or None),
        )

    return {
        "contract_version": "layered_retrieval_trace_public_v1",
        "trace_id": trace.id,
        "context_package_id": context_package.id if context_package else None,
        "query": trace.query,
        "retrieval_mode": trace.retrieval_mode,
        "retrieval_granularity": retrieval_granularity,
        "conversation_state_scope_hash": trace.conversation_state_scope_hash,
        "concept_path": trace.concept_path_json or [],
        "result_chunk_ids": trace.result_chunk_ids_json or [],
        "query_facets": trace.query_facets_json or {},
        "entry_nodes": [
            public_entry_node(item) for item in (trace.entry_nodes_json or [])
        ],
        "frontier": [
            _public_frontier_snapshot(item) for item in (trace.frontier_json or [])
        ],
        "stage_queues": trace.stage_queues_json or {},
        "candidate_pools": trace.candidate_pools_json or {},
        "topk_selection": trace.topk_selection_json or {},
        "path_labels": public_path_labels,
        "node_contributions": [
            _public_node_contribution(item)
            for item in (diagnostics.get("node_contributions") or [])
        ],
        "convergence": convergence,
        "trace_diagnostics": _public_trace_diagnostics(
            diagnostics, retrieval_granularity=retrieval_granularity
        ),
        "rq_diagnostics": diagnostics.get("rq") or {},
        "gray_zone_protocol": persisted_gray_zone_protocol,
        "gray_zone_decision_authority": gray_zone_decision_authority,
        "gray_zone_model_call_count": gray_zone_model_call_count,
        "gray_zone_determinism": gray_zone_audit,
        "gray_zone_path_decisions": gray_zone_decisions,
        "path_distance_threshold_hits": path_distance_threshold_hits,
        "steps": [
            {
                "id": step.id,
                "step_index": step.step_index,
                "layer": step.layer,
                "action": step.action,
                "action_type": step.action_type,
                "parent_layer": step.parent_layer,
                "parent_node_id": step.parent_node_id,
                "input": _public_step_input(step.input_json),
                "output": _public_step_output(step.output_json),
                "candidate_pool_ids": step.candidate_pool_ids_json or [],
                "selected_topk_ids": step.selected_topk_ids_json or [],
                "per_parent_budget_status": step.per_parent_budget_status_json or {},
                "popped_frontier_state": _public_traversal_state(
                    step.popped_frontier_state_json
                ),
                "expanded_edge_ids": step.expanded_edge_ids_json or [],
                "dominance_pruned_count": step.dominance_pruned_count,
                "cycle_distance_reward": step.cycle_distance_reward,
                "gray_zone_path_decisions": step.gray_zone_path_decisions_json or [],
                "stop_reason": step.stop_reason,
                "diagnostics": {
                    "retrieval_granularity": retrieval_granularity,
                    "traversal_protocol": diagnostics.get("traversal_protocol"),
                    "scores_json_retired_as_primary_audit": True,
                },
                "created_at": step.created_at,
            }
            for step in steps
        ],
    }


def get_dashboard_snapshot(db: Session, knowledge_base_id: str, *, include_graph: bool = True) -> dict:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        return empty_dashboard()
    documents = db.scalars(select(Document).where(Document.knowledge_base_id == knowledge_base.id, Document.is_active.is_(True))).all()
    file_items = list_knowledge_base_files(db, knowledge_base.id)
    chunk_count = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
    latest_batch = db.scalar(
        select(IngestionBatch)
        .where(IngestionBatch.knowledge_base_id == knowledge_base.id, IngestionBatch.status.notin_(TERMINAL_BATCH_STATES))
        .order_by(IngestionBatch.created_at.desc())
        .limit(1)
    )
    tree = tree_payload(knowledge_base, file_items)
    graph = graph_layer_payload(db, knowledge_base.id, "chunk-relation") if include_graph else empty_graph("chunk-relation")
    active_profile = knowledge_base.active_profile
    paths = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )
    return {
        "knowledge_base": {
            "id": knowledge_base.id,
            "name": knowledge_base.name,
            "description": knowledge_base.description,
            "source_root": str(paths["storage_root"]),
            "storage_root": str(paths["storage_root"]),
            "document_count": len(file_items),
            "chunk_count": chunk_count,
            "current_chunk_version": knowledge_base.current_chunk_version or 0,
            "has_parsed_chunks": chunk_count > 0,
            "can_full_reparse": chunk_count > 0,
            "degraded_mode": is_degraded_mode(),
            "active_profile_id": active_profile.id if active_profile else None,
            "active_profile_name": active_profile.name if active_profile else None,
            "active_profile_hash": active_profile.profile_hash if active_profile else None,
        },
        "tree": tree,
        "graph": graph,
        "batch_status": None if latest_batch is None else summarize_batch_for_dashboard(latest_batch),
        "ingested_document_count": len(documents),
        "chunk_count": chunk_count,
        "graph_relation_count": (graph.get("node_counts") or {}).get("chunk_relation_edges", 0),
        "coverage_by_source_type": dict(Counter(item.get("source_type") or "unknown" for item in file_items)),
        "degraded_mode": is_degraded_mode(),
        "context_graph": context_graph_overview_stats(db, knowledge_base.id),
    }


def empty_dashboard() -> dict:
    return {
        "knowledge_base": {
            "id": "empty",
            "name": "KnowledgeBase Workspace",
            "description": None,
            "source_root": "",
            "storage_root": "",
            "document_count": 0,
            "chunk_count": 0,
            "current_chunk_version": 0,
            "has_parsed_chunks": False,
            "can_full_reparse": False,
            "degraded_mode": is_degraded_mode(),
            "active_profile_id": None,
            "active_profile_name": None,
            "active_profile_hash": None,
        },
        "tree": [],
        "graph": empty_graph("chunk-relation"),
        "batch_status": None,
        "ingested_document_count": 0,
        "chunk_count": 0,
        "graph_relation_count": 0,
        "coverage_by_source_type": {},
        "degraded_mode": is_degraded_mode(),
    }


def empty_graph(layer: str) -> dict:
    return {
        "graph_type": layer,
        "schema_version": "context_graph_v1",
        "view": "overview",
        "nodes": [],
        "edges": [],
        "node_counts": {},
        "edge_counts": {},
        "freshness": {"is_stale": False, "stale_reasons": []},
        "diagnostics": {},
    }


def tree_payload(knowledge_base: KnowledgeBase, file_items: list[dict]) -> list[dict]:
    partition_map: dict[str, list[dict]] = defaultdict(list)
    for item in file_items:
        partition = item.get("partition") or "General"
        partition_map[partition].append(item)
    return [
        {
            "id": f"partition:{partition}",
            "title": partition,
            "type": "partition",
            "children": [
                {"id": item["document_id"] or item["id"], "title": item["title"], "type": "document", "children": []}
                for item in sorted(entries, key=lambda row: row["title"].lower())
            ],
        }
        for partition, entries in sorted(partition_map.items())
    ]


def summarize_batch_for_dashboard(batch: IngestionBatch) -> dict:
    stats = dict(batch.stats or {})
    return {
        "batch_id": batch.id,
        "state": batch.status,
        "trigger_source": batch.trigger_source,
        "source_root": batch.source_root,
        "total_files": batch.total_files,
        "processed_files": batch.processed_files,
        "success_count": batch.success_count,
        "failure_count": batch.failure_count,
        "skipped_count": batch.skipped_count,
        "coverage_by_source_type": stats.get("coverage_by_source_type", {}),
        "errors": stats.get("errors", []),
        "graph_stats": stats.get("graph_stats", {}),
        "phase": stats.get("phase"),
        "parse_committed": bool(stats.get("parse_committed")),
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
    }
