from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AgentAction,
    AgentObservation,
    AgentPlan,
    AgentRun,
    AgentTraceEvent,
    AnswerSession,
    Chunk,
    CitationVerification,
    CoarseConcept,
    CoarseConceptMembership,
    CoarseConceptState,
    ContextPackage,
    DocumentVersion,
    GraphRetrievalStep,
    MidConcept,
    MidConceptState,
    IngestionJob,
    PolicyState,
    RetrievalTrace,
    RewardEvent,
    RQPrefix,
)
from app.services.agent_repair import (
    REPAIR_EXECUTOR_MECHANISMS,
    TYPED_REPAIR_PROTOCOL_VERSION,
    canonical_repair_hash,
    claim_grounding_gate,
    claim_rows,
    exact_answer_hash,
    repair_gate_semantic_card,
    repair_made_progress,
)
from app.services.chunking import rough_token_count, stable_hash, text_hash
from app.services.graph_state_hashes import (
    CHUNK_BUSINESS_KEY_PROTOCOL_VERSION,
    canonical_fact_set_hash,
    canonical_graph_hash,
    chunk_business_references,
)


POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION = "policy_reward_metric_evidence_v4"
POLICY_REWARD_FACT_PROTOCOL_VERSION = "policy_reward_fact_v1"
POLICY_REWARD_STORAGE_PROTOCOL_VERSION = "policy_reward_persisted_replay_v1"

MAX_REWARD_ROWS = 100_000
MAX_REWARD_CLAIMS = 10_000
MAX_REWARD_TOKENS = 10_000_000
MAX_EVENT_DURATION_MS = 86_400_000
MAX_TOTAL_EVENT_DURATION_MS = 604_800_000
MAX_REWARD_CONTEXT_PACKAGE_SNAPSHOT_SCAN = 4096

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def _compact_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

_DATABASE_ADDRESS_DERIVED_HASH_FIELDS = {
    "action_input_hash",
    "action_output_hash",
    "after_gate_hash",
    "after_progress_hash",
    "before_gate_hash",
    "before_progress_hash",
    "binding_hash",
    "directive_hash",
    "failure_card_hash",
    "failure_set_hash",
    "gate_hash",
    "progress_hash",
    "rebind_input_hash",
    "semantic_failure_hash",
    "target_refs_hash",
    "validated_directive_hash",
}
_DATABASE_ADDRESS_DERIVED_HASH_LIST_FIELDS = {
    "failure_card_hashes",
    "prior_repair_action_output_hashes",
    "semantic_failure_hashes",
}

_METRIC_FIELDS = (
    "retrieval_hit",
    "context_precision",
    "context_recall",
    "concept_path_accuracy",
    "citation_pass_rate",
    "answer_groundedness",
    "answer_completeness",
    "claim_count",
    "supported_claim_count",
    "unsupported_claim_count",
    "repair_success_rate",
    "agent_typed_action_validation_pass_rate",
    "latency_cost",
    "latency_ms",
    "task_token_cost",
    "drift_rate",
    "answer_acceptance_gate_pass",
)


class PolicyRewardReplayError(RuntimeError):
    """Raised when persisted reward evidence cannot be replayed exactly."""


def _fail(message: str) -> None:
    raise PolicyRewardReplayError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    if minimum is not None and result < minimum:
        _fail(f"{field} is below its lower bound")
    if maximum is not None and result > maximum:
        _fail(f"{field} exceeds its upper bound")
    return 0.0 if result == 0.0 else result


def _bounded_int(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = MAX_REWARD_ROWS,
) -> int:
    if not _is_int(value) or not minimum <= value <= maximum:
        _fail(f"{field} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _require_hash(value: Any, *, field: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    digest = str(value or "")
    if not _SHA256_RE.fullmatch(digest):
        _fail(f"{field} must be a canonical lowercase SHA-256 digest")
    return digest


def _assert_finite_json(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{field} contains a non-string object key")
            _assert_finite_json(item, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_finite_json(item, field=f"{field}[{index}]")
        return
    _fail(f"{field} contains a non-JSON value")


def _strict_json(value: Any, *, field: str) -> str:
    _assert_finite_json(value, field=field)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PolicyRewardReplayError(f"{field} is not strict JSON") from exc


def _exact_json(left: Any, right: Any, *, field: str) -> None:
    if _strict_json(left, field=f"{field}.left") != _strict_json(
        right, field=f"{field}.right"
    ):
        _fail(f"{field} does not match its persisted reciprocal fact")


def _require_unique_ids(
    value: Any,
    *,
    field: str,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        _fail(f"{field} must be an array")
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            _fail(f"{field} must contain non-empty string references")
        rows.append(item)
    if len(rows) != len(set(rows)):
        _fail(f"{field} contains duplicate references")
    if not allow_empty and not rows:
        _fail(f"{field} must not be empty")
    if len(rows) > MAX_REWARD_ROWS:
        _fail(f"{field} exceeds the bounded replay row limit")
    return rows


def _require_id_sequence(
    value: Any,
    *,
    field: str,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        _fail(f"{field} must be an array")
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            _fail(f"{field} must contain non-empty string references")
        rows.append(item)
    if not allow_empty and not rows:
        _fail(f"{field} must not be empty")
    if len(rows) > MAX_REWARD_ROWS:
        _fail(f"{field} exceeds the bounded replay row limit")
    return rows


def _uuid_free_json(value: Any, *, field: str) -> Any:
    """Copy a small semantic card while rejecting database-looking UUIDs.

    This is used only for caller-independent semantic payloads such as
    ``evidence_gap``. Database address payloads are kept in the separate refs
    card and are never passed through this helper.
    """

    _assert_finite_json(value, field=field)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _UUID_RE.fullmatch(value):
            _fail(f"{field} contains a database UUID in a content fact")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key in sorted(value):
            key = str(raw_key)
            if _UUID_RE.fullmatch(key):
                _fail(f"{field} contains a database UUID object key")
            result[key] = _uuid_free_json(value[raw_key], field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _uuid_free_json(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(f"{field} is not JSON")


def _address_free_json(value: Any, *, field: str) -> Any:
    """Project persisted Agent payloads into database-address-free facts.

    Exact payload hashes and row addresses are retained in ``refs``.  The
    content card replaces UUID values (and UUID object keys) with typed
    placeholders so the same business execution in another database keeps
    the same evidence hash without making persisted action/observation rows
    unaudited.
    """

    _assert_finite_json(value, field=field)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {"database_reference": True} if _UUID_RE.fullmatch(value) else value
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {}
        database_keyed_values: list[Any] = []
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            raw_item = value[raw_key]
            if key in _DATABASE_ADDRESS_DERIVED_HASH_FIELDS and isinstance(
                raw_item,
                str,
            ) and _SHA256_RE.fullmatch(raw_item):
                projected = {"database_address_derived_hash": True}
            elif key in _DATABASE_ADDRESS_DERIVED_HASH_LIST_FIELDS and isinstance(
                raw_item,
                list,
            ):
                projected = [
                    {"database_address_derived_hash": True}
                    if isinstance(item, str) and _SHA256_RE.fullmatch(item)
                    else _address_free_json(
                        item,
                        field=f"{field}.{key}[{index}]",
                    )
                    for index, item in enumerate(raw_item)
                ]
            else:
                projected = _address_free_json(
                    raw_item,
                    field=f"{field}.{key}",
                )
            if _UUID_RE.fullmatch(key):
                database_keyed_values.append(projected)
            else:
                fields[key] = projected
        if database_keyed_values:
            return {
                "fields": fields,
                "database_keyed_values": sorted(
                    database_keyed_values,
                    key=lambda item: _strict_json(
                        item,
                        field=f"{field}.database_keyed_value",
                    ),
                ),
            }
        return fields
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _address_free_json(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(f"{field} is not JSON")


def _address_free_token_count(value: Any, *, field: str) -> int:
    projected = _address_free_json(value, field=field)
    return rough_token_count(
        _strict_json(projected, field=f"{field}.address_free_token_payload")
    )


def _typed_action_payload(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} is not an object")
    allowed = {
        "action_type",
        "target_ids",
        "reason",
        "budget_request",
        "expected_evidence",
        "stop_condition",
    }
    extra = set(value).difference(allowed)
    if extra:
        _fail(f"{field} has unknown typed action fields: {sorted(extra)}")
    action_type = str(value.get("action_type") or "")
    if not action_type:
        _fail(f"{field} has no action_type")
    target_ids = _require_id_sequence(
        list(value.get("target_ids") or []),
        field=f"{field}.target_ids",
    )
    if len(target_ids) != len(set(target_ids)):
        _fail(f"{field}.target_ids contains duplicates")
    reason = str(value.get("reason") or "")
    if len(reason) > 2_000:
        _fail(f"{field}.reason exceeds the bounded action contract")
    objects: dict[str, dict[str, Any]] = {}
    for key in ("budget_request", "expected_evidence", "stop_condition"):
        raw = value.get(key) or {}
        if not isinstance(raw, Mapping):
            _fail(f"{field}.{key} is not an object")
        objects[key] = dict(raw)
        _assert_finite_json(objects[key], field=f"{field}.{key}")
    return {
        "action_type": action_type,
        "target_ids": target_ids,
        "reason": reason,
        **objects,
    }


def _assert_no_gray_authority(
    value: Any,
    *,
    field: str,
    _gray_context: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.strip().lower().replace("-", "_")
            nested_gray_context = _gray_context or normalized_key in {
                "gray",
                "gray_zone",
                "gray_zone_audit",
                "gray_zone_decision",
            }
            if normalized_key == "gray_zone_model_call_count" and item != 0:
                _fail(f"{field} assigns a nonzero gray-zone model call count")
            if (
                nested_gray_context
                and normalized_key in {
                    "model_call_count",
                    "model_calls",
                    "llm_call_count",
                    "llm_calls",
                }
                and item != 0
            ):
                _fail(f"{field} assigns a nonzero gray-zone model call count")
            if normalized_key in {
                "gray_zone_decision_authority",
                "gray_zone_rule_inputs_modified",
            } and item not in {False, "deterministic_executor_only"}:
                _fail(f"{field} attempts to assign gray-zone authority")
            if (
                nested_gray_context
                and normalized_key
                in {"decision_authority", "authority", "rule_inputs_modified"}
                and item not in {False, "deterministic_executor_only"}
            ):
                _fail(f"{field} attempts to assign gray-zone authority")
            if normalized_key in {
                "llm_gray_zone_decision",
                "profile_gray_zone_decision",
                "conversation_gray_zone_decision",
                "provider_gray_zone_decision",
                "policy_gray_zone_decision",
            }:
                _fail(f"{field} contains a forbidden gray-zone decision")
            if normalized_key in {
                "policy_override",
                "profile_override",
                "provider_override",
            }:
                _fail(f"{field} contains a forbidden gray-zone override")
            _assert_no_gray_authority(
                item,
                field=f"{field}.{key}",
                _gray_context=nested_gray_context,
            )
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, item in enumerate(value):
            _assert_no_gray_authority(
                item,
                field=f"{field}[{index}]",
                _gray_context=_gray_context,
            )


def _structured_summary_payload(value: str, *, field: str) -> Any | None:
    """Return a JSON object/array embedded in a persisted trace summary.

    Human-readable summaries remain ordinary text.  A syntactically valid
    structured summary is an Agent payload surface and therefore receives
    the same finite-value and gray-authority checks as typed JSON columns.
    """

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, (Mapping, list)):
        return None
    _assert_finite_json(parsed, field=field)
    _assert_no_gray_authority(parsed, field=field)
    return parsed


_REPAIR_FAILURE_CARD_KEYS = {
    "repair_round_index",
    "remaining_repair_budget",
    "answer_hash",
    "context_package_id",
    "retrieval_trace_id",
    "claim_id",
    "claim_text",
    "claim_index",
    "citation_index",
    "verdict",
    "failure_type",
    "chunk_id",
    "source_span",
    "structure_closure_status",
    "covered_facets",
    "missing_evidence_roles",
    "prior_repair_action_output_hashes",
    "failure_card_hash",
    "semantic_failure_hash",
}
_REPAIR_SEMANTIC_SOURCE_SPAN_KEYS = (
    "document_id",
    "document_version_id",
    "chunk_id",
    "char_span",
    "raw_chunk_char_span",
    "page_range",
    "section_path",
    "structure_node_ids",
    "bbox",
    "source_checksum",
    "chunk_text_hash",
    "raw_span_text_hash",
    "raw_span_text_hash_protocol_version",
)


def _replay_repair_failure_cards(
    value: Any,
    *,
    action_type: str,
    repair_round_index: int,
    remaining_repair_budget: int,
    before_answer_hash: str,
    prior_repair_action_output_hashes: Sequence[str],
) -> dict[str, Any]:
    """Replay failure-card and repair-input hashes from persisted semantics."""

    if not isinstance(value, list) or not value:
        _fail("repair action has no persisted failure cards")
    if len(value) > MAX_REWARD_CLAIMS:
        _fail("repair failure-card count exceeds the bounded reward limit")
    semantic_hashes: list[str] = []
    card_hashes: list[str] = []
    failure_types: list[str] = []
    claim_ids: list[str] = []
    source_chunk_ids: list[str] = []
    cards: list[dict[str, Any]] = []
    expected_prior_outputs = list(prior_repair_action_output_hashes)
    for index, raw_card in enumerate(value):
        if not isinstance(raw_card, Mapping):
            _fail("repair failure card is not an object")
        card = dict(raw_card)
        _assert_finite_json(card, field=f"repair.failure_cards[{index}]")
        _assert_no_gray_authority(
            card,
            field=f"repair.failure_cards[{index}]",
        )
        if set(card) != _REPAIR_FAILURE_CARD_KEYS:
            _fail("repair failure card does not use the closed persisted schema")
        if (
            card.get("repair_round_index") != repair_round_index
            or card.get("remaining_repair_budget") != remaining_repair_budget
        ):
            _fail("repair failure card round/budget lineage diverges")
        if card.get("answer_hash") != before_answer_hash:
            _fail("repair failure card answer hash diverges from persisted input")
        if list(card.get("prior_repair_action_output_hashes") or []) != (
            expected_prior_outputs
        ):
            _fail("repair failure card prior-output lineage diverges")
        source_span = card.get("source_span")
        if not isinstance(source_span, Mapping):
            _fail("repair failure card source span is not an object")
        raw_for_card_hash = {
            key: item
            for key, item in card.items()
            if key not in {"failure_card_hash", "semantic_failure_hash"}
        }
        replayed_card_hash = canonical_repair_hash(
            TYPED_REPAIR_PROTOCOL_VERSION,
            raw_for_card_hash,
        )
        if card.get("failure_card_hash") != replayed_card_hash:
            _fail("repair failure card hash failed persisted semantic replay")
        stable_source_span = {
            key: source_span.get(key)
            for key in _REPAIR_SEMANTIC_SOURCE_SPAN_KEYS
            if source_span.get(key) is not None
        }
        replayed_semantic_hash = canonical_repair_hash(
            TYPED_REPAIR_PROTOCOL_VERSION,
            {
                "answer_hash": card["answer_hash"],
                "claim_id": card["claim_id"],
                "claim_text": card["claim_text"],
                "verdict": card["verdict"],
                "failure_type": card["failure_type"],
                "chunk_id": card["chunk_id"],
                "source_span": stable_source_span,
                "structure_closure_status": card["structure_closure_status"],
                "covered_facets": card["covered_facets"],
                "missing_evidence_roles": card["missing_evidence_roles"],
            },
        )
        if card.get("semantic_failure_hash") != replayed_semantic_hash:
            _fail("repair semantic failure hash failed persisted-fact replay")
        card_hashes.append(replayed_card_hash)
        semantic_hashes.append(replayed_semantic_hash)
        failure_types.append(str(card.get("failure_type") or "unsupported_claim"))
        if card.get("claim_id"):
            claim_ids.append(str(card["claim_id"]))
        if card.get("chunk_id"):
            source_chunk_ids.append(str(card["chunk_id"]))
        cards.append(card)
    failure_set_hash = canonical_repair_hash(
        TYPED_REPAIR_PROTOCOL_VERSION,
        semantic_hashes,
    )
    action_input_hash = canonical_repair_hash(
        TYPED_REPAIR_PROTOCOL_VERSION,
        {
            "action_type": action_type,
            "failure_set_hash": failure_set_hash,
            "failure_card_hashes": semantic_hashes,
        },
    )
    return {
        "cards": cards,
        "failure_card_hashes": card_hashes,
        "semantic_failure_hashes": semantic_hashes,
        "failure_set_hash": failure_set_hash,
        "action_input_hash": action_input_hash,
        "failure_types": sorted(set(failure_types)),
        "claim_ids": sorted(set(claim_ids)),
        "source_chunk_ids": sorted(set(source_chunk_ids)),
    }


def _gray_decision_content_fact(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one validated gray record into UUID-free reward evidence.

    Address-dependent input/decision hashes remain in replay refs.  The
    content card retains every decision-bearing semantic field so a changed
    rule, source, predicate, observation, or hard-interrupt fact changes the
    cross-database-stable reward evidence hash.
    """

    fields = (
        "protocol_version",
        "protocol_hash",
        "matched_rule",
        "decision",
        "decision_source",
        "layer",
        "path_distance",
        "distance_zone",
        "threshold_hash",
        "observation_compacted",
        "model_call_count",
        "predicates",
        "minimum_audit",
        "observation",
        "support_refs",
        "hard_interrupt_state",
        "semantic_uncertain_edge",
        "crossing_rq_boundary",
        "gray_candidate_reasons",
    )
    fact = {field: record.get(field) for field in fields if field in record}
    _assert_no_gray_authority(fact, field="gray decision content fact")
    return _address_free_json(fact, field="gray decision content fact")


def _load_row(db: Session, model: type[Any], row_id: Any, *, field: str) -> Any:
    if not isinstance(row_id, str) or not row_id:
        _fail(f"{field} is missing")
    row = db.get(model, row_id)
    if row is None:
        _fail(f"{field} references a missing persisted row")
    return row


def _require_kb(row: Any, knowledge_base_id: str, *, field: str) -> None:
    if str(getattr(row, "knowledge_base_id", "") or "") != knowledge_base_id:
        _fail(f"{field} crosses the reward knowledge-base boundary")


def _ordered_rows_by_ids(
    db: Session,
    model: type[Any],
    row_ids: list[str],
    *,
    field: str,
) -> list[Any]:
    if not row_ids:
        return []
    rows = list(db.scalars(select(model).where(model.id.in_(row_ids))).all())
    by_id = {str(row.id): row for row in rows}
    if len(by_id) != len(row_ids) or set(by_id) != set(row_ids):
        _fail(f"{field} contains missing persisted references")
    return [by_id[row_id] for row_id in row_ids]


def _business_key(
    protocol: str,
    payload: Mapping[str, Any],
) -> str:
    _assert_finite_json(payload, field=protocol)
    return canonical_graph_hash(protocol, dict(payload))


def _trace_time_admitted_lifecycle_state(value: Any, *, field: str) -> str:
    """Replay the immutable admission fact, not the mutable graph lifecycle."""

    current = str(value or "")
    if current not in {"active", "inactive"}:
        _fail(
            f"{field} has no valid trace-time active admission; "
            f"current lifecycle state is {current or 'missing'}"
        )
    # A persisted RetrievalTrace can only select rows admitted from an active
    # graph.  Promotion later retires those rows to ``inactive``; that mutable
    # lifecycle transition must not rewrite a historical reward business key.
    # Keep the v1 field/value shape so already frozen reward cards replay
    # exactly without a lossy migration.
    return "active"


def _concept_business_keys(
    db: Session,
    *,
    model: type[Any],
    row_ids: Iterable[str],
    knowledge_base_id: str,
    layer: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    ids = sorted(set(str(row_id) for row_id in row_ids if str(row_id)))
    rows = _ordered_rows_by_ids(
        db,
        model,
        ids,
        field=f"trace.concept_path.{layer}",
    )
    key_by_id: dict[str, str] = {}
    row_by_id: dict[str, Any] = {}
    for row in rows:
        _require_kb(row, knowledge_base_id, field=f"{layer} concept")
        state_id = (
            getattr(row, "coarse_state_id", None)
            if layer == "coarse"
            else getattr(row, "concept_state_id", None)
        )
        state = _load_row(
            db,
            CoarseConceptState if layer == "coarse" else MidConceptState,
            str(state_id or ""),
            field=f"{layer}.concept_state",
        )
        _require_kb(state, knowledge_base_id, field=f"{layer} concept state")
        fact = {
            "layer": layer,
            "canonical_label": str(row.canonical_label or ""),
            "definition_hash": hashlib.sha256(
                str(row.definition or "").encode("utf-8")
            ).hexdigest(),
            "summary_hash": hashlib.sha256(
                str(row.summary or "").encode("utf-8")
            ).hexdigest(),
            "grounding_hash": _require_hash(
                row.grounding_hash,
                field=f"{layer}.grounding_hash",
            ),
            "concept_state_hash": _require_hash(
                state.state_hash,
                field=f"{layer}.concept_state_hash",
            ),
            "concept_state_grounding_hash": _require_hash(
                state.grounding_hash,
                field=f"{layer}.concept_state_grounding_hash",
            ),
            "prompt_protocol_version": str(
                state.prompt_protocol_version or ""
            ),
            "state": _trace_time_admitted_lifecycle_state(
                row.state,
                field=f"{layer} concept",
            ),
        }
        key_by_id[str(row.id)] = _business_key(
            f"{layer}_concept_reward_business_key_v1",
            fact,
        )
        row_by_id[str(row.id)] = row
    if len(set(key_by_id.values())) != len(key_by_id):
        _fail(f"{layer} concept business-key collision")
    return key_by_id, row_by_id


def _rq_business_keys(
    db: Session,
    *,
    row_ids: Iterable[str],
    knowledge_base_id: str,
) -> dict[str, str]:
    ids = sorted(set(str(row_id) for row_id in row_ids if str(row_id)))
    rows = _ordered_rows_by_ids(
        db,
        RQPrefix,
        ids,
        field="trace.concept_path.rq_membership",
    )
    result: dict[str, str] = {}
    for row in rows:
        _require_kb(row, knowledge_base_id, field="RQ prefix")
        result[str(row.id)] = _business_key(
            "rq_prefix_reward_business_key_v1",
            {
                "rq_prefix_key": str(row.rq_prefix_key or ""),
                "rq_level": row.rq_level,
                "rq_path_prefix": list(row.rq_path_prefix or []),
                "codebook_version": str(row.codebook_version or ""),
                "state": _trace_time_admitted_lifecycle_state(
                    row.state,
                    field="RQ prefix",
                ),
            },
        )
    if len(set(result.values())) != len(result):
        _fail("RQ prefix business-key collision")
    return result


def _canonical_source_span(
    span: Mapping[str, Any],
    *,
    chunk: Chunk,
    chunk_key: str,
    version: DocumentVersion,
    content: str,
    context_package_id: str,
    retrieval_trace_id: str,
    verification_id: str | None = None,
) -> dict[str, Any]:
    raw = dict(span or {})
    if str(raw.get("chunk_id") or "") != str(chunk.id):
        _fail("citation source span is not bound to its persisted chunk")
    if str(raw.get("document_version_id") or "") != str(chunk.document_version_id):
        _fail("citation source span has a broken document-version reciprocal link")
    if str(raw.get("context_package_id") or "") != context_package_id:
        _fail("citation source span has a broken Context Package link")
    if str(raw.get("retrieval_trace_id") or "") != retrieval_trace_id:
        _fail("citation source span has a broken Retrieval Trace link")
    if verification_id is not None and str(raw.get("verification_id") or "") != str(
        verification_id
    ):
        _fail("citation source span has a broken verification reciprocal link")
    if str(raw.get("source_checksum") or "") != str(version.checksum or ""):
        _fail("citation source checksum does not match DocumentVersion")
    if str(raw.get("chunk_text_hash") or "") != str(chunk.text_hash or ""):
        _fail("citation chunk text hash does not match Chunk")
    if str(chunk.text_hash or "") != text_hash(str(chunk.text or "")):
        _fail("persisted Chunk text hash failed deterministic replay")

    char_span = raw.get("char_span")
    raw_chunk_span = raw.get("raw_chunk_char_span")
    if (
        not isinstance(char_span, list)
        or len(char_span) != 2
        or not all(_is_int(item) for item in char_span)
        or char_span[0] < int(chunk.char_start)
        or char_span[1] < char_span[0]
        or char_span[1] > int(chunk.char_end)
    ):
        _fail("citation char span is outside the persisted raw Chunk")
    if raw_chunk_span != [int(chunk.char_start), int(chunk.char_end)]:
        _fail("citation raw chunk span does not match the persisted Chunk")
    start = int(char_span[0]) - int(chunk.char_start)
    end = int(char_span[1]) - int(chunk.char_start)
    expected_content = str(chunk.text or "")[start:end]
    if content != expected_content:
        _fail("citation content does not replay from the persisted raw Chunk span")
    raw_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if str(raw.get("raw_span_text_hash") or "") != raw_hash:
        _fail("citation raw-span text hash failed replay")

    page_range = raw.get("page_range")
    if not isinstance(page_range, list) or len(page_range) != 2:
        _fail("citation page range is not a canonical pair")
    bbox = dict(raw.get("bbox") or {})
    _assert_finite_json(bbox, field="citation.bbox")
    return {
        "chunk_business_key": chunk_key,
        "char_span": [int(char_span[0]), int(char_span[1])],
        "raw_chunk_char_span": [
            int(raw_chunk_span[0]),
            int(raw_chunk_span[1]),
        ],
        "page_range": list(page_range),
        "section_path": str(raw.get("section_path") or ""),
        "source_path_hash": hashlib.sha256(
            str(raw.get("source_path") or "").encode("utf-8")
        ).hexdigest(),
        "logical_source_path_hash": hashlib.sha256(
            str(raw.get("logical_source_path") or "").encode("utf-8")
        ).hexdigest(),
        "source_checksum": str(version.checksum or ""),
        "chunk_text_hash": str(chunk.text_hash or ""),
        "raw_span_text_hash": raw_hash,
        "content_hash": raw_hash,
        "bbox": bbox,
    }


def _context_package_document_snapshots(
    package: ContextPackage,
) -> dict[str, dict[str, str]]:
    raw_chunks = list((package.package_json or {}).get("chunks") or [])
    snapshots: dict[str, dict[str, str]] = {}
    for index, raw_item in enumerate(raw_chunks):
        if not isinstance(raw_item, Mapping):
            _fail(f"Context Package chunk[{index}] is not an object")
        item = dict(raw_item)
        document_id = str(item.get("document_id") or "")
        logical_source_path = str(item.get("logical_source_path") or "")
        if not document_id or not logical_source_path:
            _fail(
                f"Context Package chunk[{index}] has no frozen document identity"
            )
        snapshot = {
            "source_path": logical_source_path,
            "title": str(item.get("document_title") or ""),
        }
        prior = snapshots.setdefault(document_id, snapshot)
        if prior != snapshot:
            _fail("Context Package has divergent frozen document identities")
    return snapshots


def _build_chunk_facts(
    db: Session,
    *,
    knowledge_base_id: str,
    package: ContextPackage,
    trace: RetrievalTrace,
    extra_chunk_ids: Iterable[str],
    allow_empty_results: bool = False,
    additional_document_snapshots: Mapping[str, Mapping[str, str]] | None = None,
    replay_cutoff: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Chunk], dict[str, str], dict[str, dict[str, Any]]]:
    package_json = dict(package.package_json or {})
    raw_package_chunks = package_json.get("chunks")
    if not isinstance(raw_package_chunks, list):
        _fail("Context Package chunks are missing")
    if len(raw_package_chunks) > MAX_REWARD_ROWS:
        _fail("Context Package chunk count exceeds the replay limit")
    package_ids: list[str] = []
    document_snapshot_by_id: dict[str, dict[str, str]] = {
        str(document_id): {
            "source_path": str(snapshot.get("source_path") or ""),
            "title": str(snapshot.get("title") or ""),
        }
        for document_id, snapshot in dict(
            additional_document_snapshots or {}
        ).items()
    }
    for index, item in enumerate(raw_package_chunks):
        if not isinstance(item, Mapping):
            _fail(f"Context Package chunk[{index}] is not an object")
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            _fail(f"Context Package chunk[{index}] has no chunk reference")
        package_ids.append(chunk_id)
        document_id = str(item.get("document_id") or "")
        logical_source_path = str(item.get("logical_source_path") or "")
        document_title = str(item.get("document_title") or "")
        if not document_id or not logical_source_path:
            _fail(
                f"Context Package chunk[{index}] has no frozen document identity"
            )
        snapshot = {"source_path": logical_source_path, "title": document_title}
        prior_snapshot = document_snapshot_by_id.setdefault(
            document_id, snapshot
        )
        if prior_snapshot != snapshot:
            _fail("Context Package has divergent frozen document identities")
    if len(package_ids) != len(set(package_ids)):
        _fail("Context Package contains duplicate chunks")

    hit_ids = _require_unique_ids(
        list(package.hit_chunk_ids_json or []),
        field="context_package.hit_chunk_ids",
    )
    restored_ids = _require_unique_ids(
        list(package.restored_chunk_ids_json or []),
        field="context_package.restored_chunk_ids",
    )
    bridge_ids = _require_unique_ids(
        list(package.bridge_chunk_ids_json or []),
        field="context_package.bridge_chunk_ids",
    )
    result_ids = _require_unique_ids(
        list(trace.result_chunk_ids_json or []),
        field="retrieval_trace.result_chunk_ids",
        allow_empty=allow_empty_results,
    )
    all_ids = sorted(
        set(
            [
                *package_ids,
                *hit_ids,
                *restored_ids,
                *bridge_ids,
                *result_ids,
                *(str(item) for item in extra_chunk_ids if str(item)),
            ]
        )
    )
    chunks = _ordered_rows_by_ids(db, Chunk, all_ids, field="reward chunk scope")
    chunks_by_id = {str(chunk.id): chunk for chunk in chunks}
    for chunk in chunks:
        _require_kb(chunk, knowledge_base_id, field="reward chunk")
    missing_document_ids = {
        str(chunk.document_id)
        for chunk in chunks
        if str(chunk.document_id) not in document_snapshot_by_id
    }
    if missing_document_ids:
        if replay_cutoff is None:
            _fail("reward document snapshot recovery has no replay cutoff")
        expected_version_checksums_by_document: dict[str, set[str]] = defaultdict(set)
        for chunk in chunks:
            version = db.get(DocumentVersion, str(chunk.document_version_id))
            if version is None:
                _fail("reward chunk references a missing DocumentVersion")
            expected_version_checksums_by_document[str(chunk.document_id)].add(
                str(version.checksum or "")
            )
        historical_jobs = list(
            db.scalars(
                select(IngestionJob)
                .where(
                    IngestionJob.knowledge_base_id == knowledge_base_id,
                    IngestionJob.document_id.in_(missing_document_ids),
                    IngestionJob.created_at <= replay_cutoff,
                )
                .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
                .limit(MAX_REWARD_CONTEXT_PACKAGE_SNAPSHOT_SCAN)
            ).all()
        )
        for historical_job in historical_jobs:
            document_id = str(historical_job.document_id or "")
            if document_id not in missing_document_ids:
                continue
            intent = dict(
                (historical_job.stats or {}).get("document_metadata_intent")
                or {}
            )
            candidate_state = dict(intent.get("candidate_state") or {})
            metadata = dict(candidate_state.get("metadata") or {})
            pending_payload = {
                key: value
                for key, value in intent.items()
                if key
                not in {
                    "pending_payload_hash",
                    "applied_at",
                    "apply_verification",
                }
            }
            pending_payload["status"] = "pending"
            apply_verification = dict(intent.get("apply_verification") or {})
            try:
                applied_at = datetime.fromisoformat(str(intent.get("applied_at") or ""))
            except ValueError:
                continue
            if (
                intent.get("protocol_version") != "document_metadata_intent_v2"
                or intent.get("status") != "applied"
                or str(intent.get("knowledge_base_id") or "")
                != knowledge_base_id
                or str(intent.get("document_id") or "") != document_id
                or intent.get("pending_payload_hash")
                != _compact_json_hash(pending_payload)
                or intent.get("candidate_state_hash")
                != _compact_json_hash(candidate_state)
                or apply_verification.get("ok") is not True
                or apply_verification.get("metadata_hash")
                != _compact_json_hash(metadata)
                or applied_at > replay_cutoff
                or str(metadata.get("checksum") or "")
                not in expected_version_checksums_by_document[document_id]
                or str(metadata.get("source_path") or "")
                != str(intent.get("source_path") or "")
                or not str(metadata.get("title") or "")
            ):
                continue
            document_snapshot_by_id[document_id] = {
                "source_path": str(metadata["source_path"]),
                "title": str(metadata["title"]),
            }
            missing_document_ids.remove(document_id)
            if not missing_document_ids:
                break
    if missing_document_ids:
        historical_packages = list(
            db.scalars(
                select(ContextPackage)
                .where(
                    ContextPackage.knowledge_base_id == knowledge_base_id,
                    ContextPackage.created_at <= replay_cutoff,
                )
                .order_by(ContextPackage.created_at.desc(), ContextPackage.id.desc())
                .limit(MAX_REWARD_CONTEXT_PACKAGE_SNAPSHOT_SCAN)
            ).all()
        )
        for historical_package in historical_packages:
            for document_id, snapshot in _context_package_document_snapshots(
                historical_package
            ).items():
                if document_id in missing_document_ids:
                    document_snapshot_by_id[document_id] = snapshot
                    missing_document_ids.remove(document_id)
            if not missing_document_ids:
                break
    if missing_document_ids:
        _fail(
            "reward chunk scope has no bounded historical source snapshot "
            "for every referenced document"
        )
    refs = chunk_business_references(db, chunks)
    snapshot_fact_by_id: dict[str, dict[str, Any]] = {}
    snapshot_key_by_id: dict[str, str] = {}
    for chunk in chunks:
        chunk_id = str(chunk.id)
        snapshot = document_snapshot_by_id.get(str(chunk.document_id))
        if snapshot is None:
            _fail(
                "reward chunk scope has no Context Package document snapshot "
                f"for document {chunk.document_id} (chunk {chunk.id})"
            )
        current_fact = refs.fact_by_id[chunk_id]
        document_fact = {
            **dict(current_fact["document"]),
            "source_path": snapshot["source_path"],
            "title": snapshot["title"],
            # Document.checksum is mutable across selected-file reparses; the
            # referenced DocumentVersion checksum is the immutable historical
            # source identity and was equal at package construction time.
            "document_checksum": str(
                current_fact["document"]["document_version_checksum"]
            ),
        }
        snapshot_fact = {**current_fact, "document": document_fact}
        snapshot_fact_by_id[chunk_id] = snapshot_fact
        snapshot_key_by_id[chunk_id] = canonical_graph_hash(
            CHUNK_BUSINESS_KEY_PROTOCOL_VERSION,
            snapshot_fact,
        )
    snapshot_scope_hash = canonical_fact_set_hash(
        "chunk_business_scope_hash_v1",
        snapshot_fact_by_id.values(),
    )
    if len(set(snapshot_key_by_id.values())) != len(snapshot_key_by_id):
        _fail("reward chunk scope has a canonical business-key collision")

    package_facts: list[dict[str, Any]] = []
    package_fact_by_id: dict[str, dict[str, Any]] = {}
    token_count = 0
    for index, raw_item in enumerate(raw_package_chunks):
        item = dict(raw_item)
        chunk_id = str(item["chunk_id"])
        chunk = chunks_by_id[chunk_id]
        version = _load_row(
            db,
            DocumentVersion,
            str(chunk.document_version_id),
            field=f"context_package.chunk[{index}].document_version",
        )
        content = str(item.get("content") or "")
        span = dict(item.get("source_span") or {})
        source_fact = _canonical_source_span(
            span,
            chunk=chunk,
            chunk_key=snapshot_key_by_id[chunk_id],
            version=version,
            content=content,
            context_package_id=str(package.id),
            retrieval_trace_id=str(trace.id),
        )
        if str(item.get("document_id") or "") != str(chunk.document_id):
            _fail("Context Package chunk has a broken document link")
        if str(item.get("document_version_id") or "") != str(
            chunk.document_version_id
        ):
            _fail("Context Package chunk has a broken document-version link")
        if str(item.get("context_package_id") or "") != str(package.id):
            _fail("Context Package chunk has a broken package reciprocal link")
        if list(item.get("char_span") or []) != source_fact["char_span"]:
            _fail("Context Package chunk and source-span coordinates diverge")
        if str(item.get("raw_span_text_hash") or "") != source_fact[
            "raw_span_text_hash"
        ]:
            _fail("Context Package raw-span hash copies diverge")
        if str(item.get("chunk_text_hash") or "") != str(chunk.text_hash or ""):
            _fail("Context Package chunk-text hash copies diverge")
        item_tokens = _bounded_int(
            item.get("content_token_count"),
            field=f"context_package.chunk[{index}].content_token_count",
            maximum=MAX_REWARD_TOKENS,
        )
        if item_tokens != rough_token_count(content):
            _fail("Context Package content token count failed deterministic replay")
        token_count += item_tokens
        role = str(item.get("role") or "")
        expected_role = "hit" if chunk_id in set(hit_ids) else None
        if expected_role is not None and role != expected_role:
            _fail("Context Package hit chunk is not labeled as a hit")
        fact = {
            "chunk_business_key": snapshot_key_by_id[chunk_id],
            "role": role,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content_token_count": item_tokens,
            "content_clipped": bool(item.get("content_clipped")),
            "source_span": source_fact,
            "section_path": str(item.get("section_path") or ""),
            "structure_path": str(item.get("structure_path") or ""),
        }
        package_facts.append(fact)
        package_fact_by_id[chunk_id] = fact

    package_token_count = _bounded_int(
        package.token_count,
        field="context_package.token_count",
        maximum=MAX_REWARD_TOKENS,
    )
    package_token_budget = _bounded_int(
        package.token_budget,
        field="context_package.token_budget",
        maximum=MAX_REWARD_TOKENS,
    )
    if token_count != package_token_count or token_count > package_token_budget:
        _fail("Context Package token budget/count failed replay")
    token_audit = dict(
        (package.diagnostics_json or {}).get("token_budget_audit") or {}
    )
    if (
        token_audit.get("token_count") != package_token_count
        or token_audit.get("token_budget") != package_token_budget
        or token_audit.get("within_budget") is not True
    ):
        _fail("Context Package token audit is not reciprocal with stored counts")
    why_selected = dict(package.why_selected_json or {})
    if set(why_selected).difference(
        set(package_ids) | set(hit_ids) | set(restored_ids)
    ):
        _fail("Context Package why-selected map contains an out-of-scope chunk")
    covered_from_selection = sorted(
        {
            str(facet)
            for value in why_selected.values()
            if isinstance(value, Mapping)
            for facet in (value.get("covered_facets") or [])
            if str(facet)
        }
    )
    covered_facets = sorted(
        set(
            _require_unique_ids(
                list(package.covered_facets_json or []),
                field="context_package.covered_facets",
            )
        )
    )
    if covered_from_selection and covered_from_selection != covered_facets:
        _fail("Context Package covered facets do not replay from selection facts")

    citation_spans = list(package.citation_spans_json or [])
    if len(citation_spans) != len(raw_package_chunks):
        _fail("Context Package citation-span count does not match package chunks")
    for index, span in enumerate(citation_spans):
        if not isinstance(span, Mapping):
            _fail("Context Package citation span is not an object")
        package_item = dict(raw_package_chunks[index])
        # ``citation_spans_json`` is the canonical source span enriched with
        # package-level document and structure metadata.  ``document_id`` is
        # deliberately stored on the package item (not CitationSourceSpan),
        # so replay must compare the exact persisted construction rather than
        # incorrectly require it in both source-span copies.
        expected_span = {
            **dict(package_item.get("source_span") or {}),
            "document_id": package_item.get("document_id"),
            "document_title": package_item.get("document_title") or "",
            "source_path": package_item.get("source_path") or "",
            "logical_source_path": package_item.get("logical_source_path") or "",
            "section_path": package_item.get("section_path"),
            "structure_path": package_item.get("structure_path"),
            "structure_node_ids": package_item.get("structure_node_ids") or [],
            "structure_closure": package_item.get("structure_closure") or {},
        }
        if dict(span) != expected_span:
            _fail("Context Package citation span copies diverge")

    facts = {
        "chunk_business_key_protocol_version": CHUNK_BUSINESS_KEY_PROTOCOL_VERSION,
        "chunk_business_scope_hash": snapshot_scope_hash,
        "package_chunks": sorted(
            package_facts,
            key=lambda item: (
                item["chunk_business_key"],
                _strict_json(item["source_span"]["char_span"], field="package.span"),
            ),
        ),
        "hit_chunk_business_keys": sorted(snapshot_key_by_id[row_id] for row_id in hit_ids),
        "restored_chunk_business_keys": sorted(
            snapshot_key_by_id[row_id] for row_id in restored_ids
        ),
        "bridge_chunk_business_keys": sorted(
            snapshot_key_by_id[row_id] for row_id in bridge_ids
        ),
        "result_chunk_business_keys": [
            snapshot_key_by_id[row_id] for row_id in result_ids
        ],
        "covered_facets": covered_facets,
        "token_budget": package_token_budget,
        "token_count": package_token_count,
        "query_hash": hashlib.sha256(str(package.query or "").encode("utf-8")).hexdigest(),
    }
    return facts, chunks_by_id, snapshot_key_by_id, package_fact_by_id


def _build_path_facts(
    db: Session,
    *,
    knowledge_base_id: str,
    trace: RetrievalTrace,
    chunk_by_id: Mapping[str, Chunk],
    chunk_key_by_id: Mapping[str, str],
    allow_empty_results: bool = False,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    raw_concept_path = list(trace.concept_path_json or [])
    layer_ids: dict[str, list[str]] = {}
    for item in raw_concept_path:
        if not isinstance(item, Mapping):
            _fail("Retrieval Trace concept path contains a non-object")
        layer = str(item.get("layer") or "")
        if not layer or layer in layer_ids:
            _fail("Retrieval Trace concept path has missing/duplicate layers")
        layer_ids[layer] = _require_unique_ids(
            list(item.get("ids") or []),
            field=f"retrieval_trace.concept_path.{layer}",
        )

    coarse_ids = layer_ids.get("coarse", [])
    mid_ids = layer_ids.get("mid", [])
    path_chunk_ids = layer_ids.get("chunk", [])
    topk = dict(trace.topk_selection_json or {})
    candidate_pools = dict(trace.candidate_pools_json or {})
    all_coarse_ids = set(coarse_ids)
    all_mid_ids = set(mid_ids)
    for item in (topk.get("coarse") or {}).get("selected_ids") or []:
        all_coarse_ids.add(str(item))
    for item in (topk.get("mid") or {}).get("selected_ids") or []:
        all_mid_ids.add(str(item))
    for raw_pool in list(candidate_pools.get("mid_by_coarse") or []):
        if isinstance(raw_pool, Mapping):
            if raw_pool.get("parent_node_id"):
                all_coarse_ids.add(str(raw_pool["parent_node_id"]))
            all_mid_ids.update(str(item) for item in raw_pool.get("candidate_ids") or [])
            all_mid_ids.update(str(item) for item in raw_pool.get("selected_ids") or [])
    for raw_pool in list(candidate_pools.get("chunk_by_mid") or []):
        if isinstance(raw_pool, Mapping) and raw_pool.get("parent_node_id"):
            all_mid_ids.add(str(raw_pool["parent_node_id"]))
    for label in list(trace.path_labels_json or []):
        if not isinstance(label, Mapping):
            continue
        layer = str(label.get("layer") or ("chunk" if label.get("chunk_id") else ""))
        ids = {
            str(label.get("node_id") or label.get("chunk_id") or ""),
            *(str(item) for item in label.get("path") or []),
        }
        ids.discard("")
        if layer == "coarse":
            all_coarse_ids.update(ids)
        elif layer == "mid":
            all_mid_ids.update(ids)
    coarse_keys, coarse_rows = _concept_business_keys(
        db,
        model=CoarseConcept,
        row_ids=all_coarse_ids,
        knowledge_base_id=knowledge_base_id,
        layer="coarse",
    )
    mid_keys, mid_rows = _concept_business_keys(
        db,
        model=MidConcept,
        row_ids=all_mid_ids,
        knowledge_base_id=knowledge_base_id,
        layer="mid",
    )
    rq_keys = _rq_business_keys(
        db,
        row_ids=layer_ids.get("rq_membership", []),
        knowledge_base_id=knowledge_base_id,
    )
    if set(path_chunk_ids).difference(chunk_by_id):
        _fail("Retrieval Trace concept path references an unloaded chunk")

    selected_coarse = _require_unique_ids(
        list((topk.get("coarse") or {}).get("selected_ids") or []),
        field="retrieval_trace.topk.coarse",
    )
    selected_mid = _require_unique_ids(
        list((topk.get("mid") or {}).get("selected_ids") or []),
        field="retrieval_trace.topk.mid",
    )
    selected_chunk = _require_unique_ids(
        list((topk.get("chunk") or {}).get("selected_ids") or []),
        field="retrieval_trace.topk.chunk",
        allow_empty=allow_empty_results,
    )
    if selected_coarse != coarse_ids or selected_mid != mid_ids:
        _fail("Retrieval Trace concept path does not match staged top-k selections")
    if selected_chunk != list(trace.result_chunk_ids_json or []) or path_chunk_ids != selected_chunk:
        _fail("Retrieval Trace chunk path does not match final selected results")

    mid_pools = list(candidate_pools.get("mid_by_coarse") or [])
    chunk_pools = list(candidate_pools.get("chunk_by_mid") or [])
    coarse_to_mid: list[dict[str, str]] = []
    mid_to_chunk: list[dict[str, str]] = []
    coarse_parent_ids: set[str] = set()
    mid_parent_ids: set[str] = set()

    memberships = list(
        db.scalars(
            select(CoarseConceptMembership).where(
                CoarseConceptMembership.coarse_concept_id.in_(selected_coarse)
            )
        ).all()
    ) if selected_coarse else []
    membership_pairs = {
        (str(row.coarse_concept_id), str(row.mid_concept_id)) for row in memberships
    }
    for raw_pool in mid_pools:
        if not isinstance(raw_pool, Mapping):
            _fail("mid-by-coarse candidate pool is not an object")
        parent_id = str(raw_pool.get("parent_node_id") or "")
        if parent_id not in coarse_keys:
            _fail("mid-by-coarse candidate pool has an unknown coarse parent")
        coarse_parent_ids.add(parent_id)
        candidates = _require_unique_ids(
            list(raw_pool.get("candidate_ids") or []),
            field="mid_by_coarse.candidate_ids",
        )
        selected = _require_unique_ids(
            list(raw_pool.get("selected_ids") or []),
            field="mid_by_coarse.selected_ids",
        )
        if not set(selected).issubset(candidates) or not set(selected).issubset(mid_keys):
            _fail("mid-by-coarse selected children are outside the candidate/path scope")
        for child_id in selected:
            if (parent_id, child_id) not in membership_pairs:
                _fail("coarse-to-mid path lacks persisted concept membership")
            coarse_to_mid.append(
                {
                    "coarse_concept_business_key": coarse_keys[parent_id],
                    "mid_concept_business_key": mid_keys[child_id],
                }
            )

    for raw_pool in chunk_pools:
        if not isinstance(raw_pool, Mapping):
            _fail("chunk-by-mid candidate pool is not an object")
        parent_id = str(raw_pool.get("parent_node_id") or "")
        if parent_id not in mid_keys:
            _fail("chunk-by-mid candidate pool has an unknown mid parent")
        mid_parent_ids.add(parent_id)
        candidates = _require_unique_ids(
            list(raw_pool.get("candidate_ids") or []),
            field="chunk_by_mid.candidate_ids",
        )
        selected = _require_unique_ids(
            list(raw_pool.get("selected_ids") or []),
            field="chunk_by_mid.selected_ids",
        )
        if not set(selected).issubset(candidates) or not set(selected).issubset(
            chunk_key_by_id
        ):
            _fail("chunk-by-mid selected children are outside the candidate/path scope")
        supported_ids = set(mid_rows[parent_id].support_chunk_ids_json or [])
        for child_id in selected:
            if child_id not in supported_ids:
                _fail("mid-to-chunk path lacks persisted concept support")
            mid_to_chunk.append(
                {
                    "mid_concept_business_key": mid_keys[parent_id],
                    "chunk_business_key": chunk_key_by_id[child_id],
                }
            )

    granularity = str(
        (trace.diagnostics_json or {}).get("retrieval_granularity") or ""
    )
    if granularity not in {"coarse", "mid"}:
        _fail("Retrieval Trace has an invalid retrieval granularity")
    if set(selected_mid) != mid_parent_ids:
        _fail("not every selected mid concept has a persisted chunk drilldown pool")
    if granularity == "coarse":
        if set(selected_coarse) != coarse_parent_ids:
            _fail("not every selected coarse concept has a persisted mid drilldown pool")
        expected_layers = ["coarse", "mid", "chunk"]
    else:
        if selected_coarse or coarse_parent_ids:
            _fail("mid-granularity trace unexpectedly contains a coarse drilldown")
        expected_layers = ["mid", "chunk"]

    path_labels: list[dict[str, Any]] = []
    for index, raw_label in enumerate(list(trace.path_labels_json or [])):
        if not isinstance(raw_label, Mapping):
            _fail("Retrieval Trace path label is not an object")
        label = dict(raw_label)
        layer = str(label.get("layer") or ("chunk" if label.get("chunk_id") else ""))
        node_id = str(label.get("node_id") or label.get("chunk_id") or "")
        mapper: Mapping[str, str]
        if layer == "coarse":
            mapper = coarse_keys
        elif layer == "mid":
            mapper = mid_keys
        elif layer == "chunk":
            mapper = chunk_key_by_id
        else:
            _fail(f"Retrieval Trace path label[{index}] has an unknown layer")
        if node_id not in mapper:
            _fail("Retrieval Trace path label points outside its persisted layer")
        path = _require_id_sequence(
            list(label.get("path") or []),
            field=f"retrieval_trace.path_label[{index}].path",
            allow_empty=False,
        )
        if any(item not in mapper for item in path):
            _fail("Retrieval Trace path label crosses an unbound layer")
        path_labels.append(
            {
                "layer": layer,
                "node_business_key": mapper[node_id],
                "path_business_keys": [mapper[item] for item in path],
                "covered_facets": sorted(
                    set(str(item) for item in label.get("covered_facets") or [] if str(item))
                ),
                "evidence_roles": sorted(
                    set(str(item) for item in label.get("evidence_roles") or [] if str(item))
                ),
                "distance_so_far": _finite_float(
                    label.get("distance_so_far") or 0.0,
                    field="path_label.distance_so_far",
                    minimum=0.0,
                ),
                "reward_so_far": _finite_float(
                    label.get("reward_so_far") or 0.0,
                    field="path_label.reward_so_far",
                    minimum=0.0,
                ),
                "stop_reason": str(label.get("stop_reason") or ""),
            }
        )

    layer_valid = {
        "coarse": bool(selected_coarse and coarse_to_mid)
        if granularity == "coarse"
        else True,
        "mid": bool(selected_mid and mid_to_chunk),
        "chunk": bool(selected_chunk),
    }
    path_accuracy = sum(1 for layer in expected_layers if layer_valid[layer]) / len(
        expected_layers
    )
    facts = {
        "retrieval_granularity": granularity,
        "expected_layers": expected_layers,
        "selected_coarse_concept_business_keys": [
            coarse_keys[row_id] for row_id in selected_coarse
        ],
        "selected_mid_concept_business_keys": [
            mid_keys[row_id] for row_id in selected_mid
        ],
        "selected_rq_prefix_business_keys": [
            rq_keys[row_id] for row_id in layer_ids.get("rq_membership", [])
        ],
        "selected_chunk_business_keys": [
            chunk_key_by_id[row_id] for row_id in selected_chunk
        ],
        "coarse_to_mid_links": sorted(
            coarse_to_mid,
            key=lambda row: (
                row["coarse_concept_business_key"],
                row["mid_concept_business_key"],
            ),
        ),
        "mid_to_chunk_links": sorted(
            mid_to_chunk,
            key=lambda row: (
                row["mid_concept_business_key"],
                row["chunk_business_key"],
            ),
        ),
        "path_labels": sorted(
            path_labels,
            key=lambda row: (
                row["layer"],
                row["node_business_key"],
                _strict_json(row, field="path_label"),
            ),
        ),
        "layer_validity": layer_valid,
    }
    path_refs = {
        "coarse_concept_ids": selected_coarse,
        "mid_concept_ids": selected_mid,
        "rq_prefix_ids": layer_ids.get("rq_membership", []),
        "chunk_ids": selected_chunk,
    }
    return facts, path_refs, {"concept_path_accuracy": round(path_accuracy, 6)}


def _build_citation_facts(
    db: Session,
    *,
    knowledge_base_id: str,
    answer: AnswerSession,
    package: ContextPackage,
    trace: RetrievalTrace,
    chunks_by_id: Mapping[str, Chunk],
    chunk_key_by_id: Mapping[str, str],
    package_fact_by_id: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    citation_ids = _require_unique_ids(
        list(answer.citation_ids_json or []),
        field="answer_session.citation_ids",
    )
    citations = _ordered_rows_by_ids(
        db,
        CitationVerification,
        citation_ids,
        field="answer_session.citation_ids",
    )
    all_for_answer = list(
        db.scalars(
            select(CitationVerification).where(
                CitationVerification.answer_session_id == answer.id
            )
        ).all()
    )
    if {str(row.id) for row in all_for_answer} != set(citation_ids):
        _fail("Answer Session citation refs omit or add persisted verifications")
    if len(citations) > MAX_REWARD_CLAIMS:
        _fail("citation count exceeds the bounded reward replay limit")

    exact_claims = claim_rows(str(answer.answer or ""))
    if len(exact_claims) > MAX_REWARD_CLAIMS:
        _fail("answer claim count exceeds the bounded reward replay limit")
    claims_by_id = {str(row["claim_id"]): row for row in exact_claims}
    verification_results: list[dict[str, Any]] = []
    citation_facts: list[dict[str, Any]] = []
    supported_chunk_ids: set[str] = set()
    for citation_index, row in enumerate(citations, start=1):
        _require_kb(row, knowledge_base_id, field="CitationVerification")
        if (
            str(row.answer_session_id or "") != str(answer.id)
            or str(row.retrieval_trace_id or "") != str(trace.id)
            or str(row.context_package_id or "") != str(package.id)
        ):
            _fail("CitationVerification reciprocal links are broken")
        confidence = _finite_float(
            row.confidence,
            field="citation.confidence",
            minimum=0.0,
            maximum=1.0,
        )
        diagnostics = dict(row.diagnostics_json or {})
        claim_id = str(diagnostics.get("claim_id") or "")
        claim_index = diagnostics.get("claim_index")
        claim = claims_by_id.get(claim_id)
        if (
            claim is None
            or not _is_int(claim_index)
            or claim_index != claim["claim_index"]
            or str(row.claim_text or "") != str(claim["claim_text"])
            or str(diagnostics.get("answer_hash") or "") != exact_answer_hash(
                str(answer.answer or "")
            )
        ):
            _fail("CitationVerification is not bound to an exact answer claim")
        failure_type = str(diagnostics.get("failure_type") or "")
        source_fact: dict[str, Any] | None = None
        chunk_id = str(row.chunk_id or "")
        if row.verdict == "supported":
            if chunk_id not in chunks_by_id or chunk_id not in package_fact_by_id:
                _fail("supported citation is outside the Context Package")
            if (
                diagnostics.get("citation_provenance_valid") is not True
                or diagnostics.get(
                    "citation_provenance_persistence_gate_passed"
                )
                is not True
                or diagnostics.get("authoritative_chunk_link_persisted") is not True
            ):
                _fail("supported citation lacks persisted provenance authority")
            chunk = chunks_by_id[chunk_id]
            version = _load_row(
                db,
                DocumentVersion,
                str(chunk.document_version_id),
                field="citation.document_version",
            )
            package_content = next(
                (
                    str(item.get("content") or "")
                    for item in (package.package_json or {}).get("chunks") or []
                    if str(item.get("chunk_id") or "") == chunk_id
                ),
                None,
            )
            if package_content is None:
                _fail("supported citation has no package content")
            source_fact = _canonical_source_span(
                dict(row.source_span_json or {}),
                chunk=chunk,
                chunk_key=chunk_key_by_id[chunk_id],
                version=version,
                content=package_content,
                context_package_id=str(package.id),
                retrieval_trace_id=str(trace.id),
                verification_id=str(row.id),
            )
            supported_chunk_ids.add(chunk_id)
        elif chunk_id:
            _fail("unsupported citation persisted an authoritative chunk link")

        result = {
            "citation_index": citation_index,
            "claim_id": claim_id,
            "claim_index": claim_index,
            "claim_text": str(row.claim_text or ""),
            "answer_hash": exact_answer_hash(str(answer.answer or "")),
            "chunk_id": chunk_id or None,
            "source_span": dict(row.source_span_json or {}),
            "verdict": str(row.verdict or ""),
            "confidence": confidence,
            "failure_type": failure_type,
            "diagnostics": diagnostics,
        }
        verification_results.append(result)
        citation_facts.append(
            {
                "citation_index": citation_index,
                "claim_id": claim_id,
                "claim_index": int(claim_index),
                "claim_text_hash": hashlib.sha256(
                    str(row.claim_text or "").encode("utf-8")
                ).hexdigest(),
                "verdict": str(row.verdict or ""),
                "failure_type": failure_type,
                "confidence": confidence,
                "source_span": source_fact,
            }
        )

    replayed_gate = claim_grounding_gate(
        str(answer.answer or ""),
        verification_results,
        require_persistence_replay=True,
    )
    answer_gate = dict(
        (answer.diagnostics_json or {}).get("claim_grounded_gate") or {}
    )
    reward_gate_placeholder: dict[str, Any] = {}
    if not answer_gate:
        _fail("Answer Session is missing its persisted claim-grounding gate")
    _exact_json(answer_gate, replayed_gate, field="answer claim-grounding gate")

    claims_facts = []
    results_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in citation_facts:
        results_by_claim[str(item["claim_id"])].append(item)
    for claim in exact_claims:
        claim_id = str(claim["claim_id"])
        rows = results_by_claim.get(claim_id, [])
        claims_facts.append(
            {
                "claim_id": claim_id,
                "claim_index": int(claim["claim_index"]),
                "claim_text_hash": hashlib.sha256(
                    str(claim["claim_text"]).encode("utf-8")
                ).hexdigest(),
                "supported": any(row["verdict"] == "supported" for row in rows),
                "support_chunk_business_keys": sorted(
                    {
                        str((row.get("source_span") or {}).get("chunk_business_key"))
                        for row in rows
                        if (row.get("source_span") or {}).get(
                            "chunk_business_key"
                        )
                    }
                ),
            }
        )
    facts = {
        "answer_hash": exact_answer_hash(str(answer.answer or "")),
        "claim_gate_protocol_version": str(replayed_gate.get("protocol_version") or ""),
        "claims": claims_facts,
        "citations": citation_facts,
        "claim_count": int(replayed_gate["claim_count"]),
        "supported_claim_count": int(replayed_gate["supported_claim_count"]),
        "unsupported_claim_count": int(replayed_gate["unsupported_claim_count"]),
        "all_claims_supported": bool(replayed_gate["all_claims_supported"]),
    }
    metrics = {
        "claim_count": int(replayed_gate["claim_count"]),
        "supported_claim_count": int(replayed_gate["supported_claim_count"]),
        "unsupported_claim_count": int(replayed_gate["unsupported_claim_count"]),
        "claim_pass_rate": _finite_float(
            replayed_gate["claim_pass_rate"],
            field="claim_pass_rate",
            minimum=0.0,
            maximum=1.0,
        ),
        "supported_chunk_ids": sorted(supported_chunk_ids),
    }
    return facts, metrics, reward_gate_placeholder, citation_ids


def _build_agent_facts(
    db: Session,
    *,
    knowledge_base_id: str,
    run: AgentRun,
    trace: RetrievalTrace,
    context_package: ContextPackage,
    claim_grounded_gate: Mapping[str, Any],
    answer_text: str,
    operating_envelope: Mapping[str, Any],
    chunk_key_by_id: Mapping[str, str],
    evidence_gap: Mapping[str, Any],
    grounding_outcome: str,
    reward_created_at: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_kb(run, knowledge_base_id, field="AgentRun")
    try:
        from app.services.agent_graph import (
            AGENT_EARLY_REPLAY_CARD_FIELDS,
            AGENT_EARLY_REPLAY_PROTOCOL_VERSION,
            _repair_progress_for_bundle,
            _source_agent_plan_binding_payload,
            compile_typed_action_execution_controls,
            historical_typed_action_required_actions_for_replay,
            rebind_historical_typed_action_validation_identity,
            validate_typed_actions,
        )
    except Exception as exc:
        raise PolicyRewardReplayError(
            "Agent typed-action replay implementation is unavailable"
        ) from exc

    def historical_target_layers_for_replay(
        actions: list[Any],
        persisted_validation: Mapping[str, Any],
    ) -> dict[str, list[str]]:
        """Recover the hash-bound target-layer witness without live graph reads.

        Selected-file reparses and graph promotion may retire database target
        identities while a RewardEvent must remain exactly replayable.  The
        validator card already persists the layer classification used at
        execution time.  Reusing only that allowlisted witness preserves the
        original validator decision; every other schema, budget, granularity,
        and action-layer rule is still replayed by ``validate_typed_actions``.
        """

        allowed_layers = {
            "coarse",
            "mid",
            "rq_membership",
            "chunk",
            "context_package",
        }
        target_layers: dict[str, list[str]] = {
            target_id: []
            for action in actions
            if isinstance(action, Mapping)
            and isinstance(action.get("target_ids"), list)
            for value in action["target_ids"]
            if isinstance(value, str)
            and (target_id := value.strip())
        }

        def merge_witness(value: Any, *, field: str) -> None:
            if value is None:
                return
            if not isinstance(value, Mapping):
                _fail(f"{field} is not an object")
            for raw_target_id, raw_layers in value.items():
                target_id = str(raw_target_id or "").strip()
                if target_id not in target_layers:
                    _fail(f"{field} contains an out-of-scope target")
                if not isinstance(raw_layers, list):
                    _fail(f"{field} contains a non-array layer witness")
                layers = [str(layer) for layer in raw_layers]
                if (
                    len(layers) != len(set(layers))
                    or any(layer not in allowed_layers for layer in layers)
                ):
                    _fail(f"{field} contains an invalid layer witness")
                normalized = sorted(layers)
                existing = target_layers[target_id]
                if existing and existing != normalized:
                    _fail(f"{field} conflicts with another layer witness")
                target_layers[target_id] = normalized

        merge_witness(
            persisted_validation.get("target_layers"),
            field="typed-action target-layer witness",
        )
        for card in list(persisted_validation.get("accepted") or []):
            if not isinstance(card, Mapping):
                _fail("typed-action accepted validator card is malformed")
            validation_card = card.get("validation")
            if not isinstance(validation_card, Mapping):
                _fail("typed-action accepted validator witness is malformed")
            merge_witness(
                validation_card.get("target_layers"),
                field="typed-action accepted target-layer witness",
            )
        for card in list(persisted_validation.get("rejected") or []):
            if not isinstance(card, Mapping):
                _fail("typed-action rejected validator card is malformed")
            if card.get("reason") != "target_layer_mismatch":
                continue
            details = card.get("details")
            if not isinstance(details, list):
                _fail("typed-action target-layer mismatch witness is malformed")
            for detail in details:
                if not isinstance(detail, Mapping):
                    _fail("typed-action target-layer mismatch detail is malformed")
                merge_witness(
                    {detail.get("target_id"): detail.get("layers")},
                    field="typed-action rejected target-layer witness",
                )
        return target_layers
    # ``operating_envelope`` is the *effective*, request-scoped envelope of
    # the reward's own (final/winning) RetrievalTrace: typed-action budget
    # requests may tighten specific fields for one search call, so this can
    # legitimately differ per round.  The Agent operating envelope B itself
    # is computed once per run and reused unchanged by every plan/replan
    # round (see ``execute_agent_run``); it must never be replayed as if a
    # later round's tightened control card had redefined B.  Recomputing
    # ``agent_operating_envelope()`` live would also be unsafe here, because
    # Runtime Settings may have changed since this reward was produced.  The
    # only historically stable source of the frozen B is therefore the
    # ``AgentPlan`` rows written at plan-creation time.
    _assert_finite_json(
        dict(operating_envelope),
        field="RetrievalTrace.effective_operating_envelope",
    )
    _assert_no_gray_authority(
        dict(operating_envelope),
        field="RetrievalTrace.effective_operating_envelope",
    )
    plans = list(
        db.scalars(
            select(AgentPlan)
            .where(
                AgentPlan.run_id == run.id,
                AgentPlan.created_at <= reward_created_at,
            )
            .order_by(AgentPlan.plan_index.asc(), AgentPlan.id.asc())
        ).all()
    )
    if len(plans) > MAX_REWARD_ROWS:
        _fail("Agent plan count exceeds the bounded reward replay limit")
    if [row.plan_index for row in plans] != list(range(len(plans))):
        _fail("Agent plan indexes are duplicate or non-contiguous")
    frozen_envelope = dict(plans[0].envelope_json or {}) if plans else dict(
        operating_envelope
    )
    _assert_finite_json(
        frozen_envelope,
        field="AgentRun.frozen_operating_envelope",
    )
    _assert_no_gray_authority(
        frozen_envelope,
        field="AgentRun.frozen_operating_envelope",
    )
    frozen_envelope_hash = stable_hash(frozen_envelope)
    persisted_actions = list(
        db.scalars(
            select(AgentAction)
            .where(
                AgentAction.run_id == run.id,
                AgentAction.created_at <= reward_created_at,
            )
            .order_by(
                AgentAction.plan_id.asc(),
                AgentAction.action_index.asc(),
                AgentAction.id.asc(),
            )
        ).all()
    )
    if len(persisted_actions) > MAX_REWARD_ROWS:
        _fail("AgentAction count exceeds the bounded reward replay limit")
    plan_ids = {str(row.id) for row in plans}
    actions_by_plan: dict[str, list[AgentAction]] = defaultdict(list)
    persisted_action_by_id: dict[str, AgentAction] = {}
    for row in persisted_actions:
        plan_id = str(row.plan_id or "")
        if str(row.run_id or "") != str(run.id) or plan_id not in plan_ids:
            _fail("AgentAction has a missing or cross-run AgentPlan")
        persisted_action_by_id[str(row.id)] = row
        actions_by_plan[plan_id].append(row)
    plan_facts: list[dict[str, Any]] = []
    action_refs: list[dict[str, Any]] = []
    appended_action_ids: set[str] = set()
    validator_input_attempt_count = 0
    validator_accepted_attempt_count = 0
    validator_inserted_action_count = 0
    terminal_plan_trace_id: str | None = None
    terminal_plan_id: str | None = None
    terminal_plan_index: int | None = None
    prior_repair_action_output_hashes: list[str] = []
    repair_input_audit_by_action_id: dict[str, dict[str, Any]] = {}
    non_context_task_token_count = (
        rough_token_count(str(run.question or ""))
        + rough_token_count(str(answer_text or ""))
    )
    for plan in plans:
        _require_kb(plan, knowledge_base_id, field="AgentPlan")
        _exact_json(
            dict(plan.envelope_json or {}),
            frozen_envelope,
            field=f"AgentPlan[{plan.plan_index}].frozen operating envelope",
        )
        plan_diagnostics = dict(plan.diagnostics_json or {})
        planner_model = dict(plan.planner_model_json or {})
        query_intent = dict(plan.query_intent_json or {})
        for field_name, payload in (
            ("diagnostics", plan_diagnostics),
            ("planner_model", planner_model),
            ("query_intent", query_intent),
        ):
            _assert_finite_json(
                payload,
                field=f"AgentPlan[{plan.plan_index}].{field_name}",
            )
            _assert_no_gray_authority(
                payload,
                field=f"AgentPlan[{plan.plan_index}].{field_name}",
            )
        plan_trace: RetrievalTrace | None = None
        if plan.retrieval_trace_id is not None:
            plan_trace = _load_row(
                db,
                RetrievalTrace,
                str(plan.retrieval_trace_id),
                field=f"AgentPlan[{plan.plan_index}].retrieval_trace_id",
            )
            _require_kb(
                plan_trace,
                knowledge_base_id,
                field=f"AgentPlan[{plan.plan_index}] RetrievalTrace",
            )
            if str(plan_trace.query or "") != str(run.question or ""):
                _fail("AgentPlan Retrieval Trace query diverges from AgentRun")
            plan_trace_diagnostics = dict(plan_trace.diagnostics_json or {})
            _assert_finite_json(
                plan_trace_diagnostics,
                field=f"AgentPlan[{plan.plan_index}] RetrievalTrace diagnostics",
            )
            _assert_no_gray_authority(
                plan_trace_diagnostics,
                field=f"AgentPlan[{plan.plan_index}] RetrievalTrace diagnostics",
            )
            trace_plan_id = str(
                plan_trace_diagnostics.get("agent_plan_id") or ""
            )
            trace_plan_index = plan_trace_diagnostics.get(
                "agent_plan_index"
            )
            if (
                trace_plan_id != str(plan.id)
                or trace_plan_index != int(plan.plan_index)
            ):
                if (
                    plan_diagnostics.get(
                        "agent_early_retrieval_cache_hit"
                    )
                    is not True
                ):
                    _fail(
                        "AgentPlan Retrieval Trace has a broken plan "
                        "id/index reciprocal link"
                    )
                source_plan_id = str(
                    plan_diagnostics.get(
                        "agent_early_replay_source_plan_id"
                    )
                    or ""
                )
                source_plan = _load_row(
                    db,
                    AgentPlan,
                    source_plan_id,
                    field=(
                        f"AgentPlan[{plan.plan_index}] early replay "
                        "source plan"
                    ),
                )
                _require_kb(
                    source_plan,
                    knowledge_base_id,
                    field="Agent early replay source AgentPlan",
                )
                source_plan_diagnostics = dict(
                    source_plan.diagnostics_json or {}
                )
                replay_card = plan_trace_diagnostics.get(
                    "agent_early_replay_card"
                )
                if (
                    not isinstance(replay_card, Mapping)
                    or set(replay_card)
                    != set(AGENT_EARLY_REPLAY_CARD_FIELDS)
                    or replay_card.get("protocol_version")
                    != AGENT_EARLY_REPLAY_PROTOCOL_VERSION
                    or str(source_plan.run_id or "") == str(run.id)
                    or int(source_plan.plan_index) != 0
                    or str(source_plan.retrieval_trace_id or "")
                    != str(plan_trace.id)
                    or trace_plan_id != str(source_plan.id)
                    or trace_plan_index != int(
                        source_plan.plan_index
                    )
                    or str(
                        replay_card.get(
                            "source_agent_plan_id"
                        )
                        or ""
                    )
                    != str(source_plan.id)
                    or replay_card.get(
                        "source_agent_plan_index"
                    )
                    != int(source_plan.plan_index)
                    or str(
                        replay_card.get(
                            "source_retrieval_trace_id"
                        )
                        or ""
                    )
                    != str(plan_trace.id)
                    or str(
                        replay_card.get(
                            "source_context_package_id"
                        )
                        or ""
                    )
                    != str(context_package.id)
                    or plan_diagnostics.get(
                        "agent_early_replay_protocol_version"
                    )
                    != AGENT_EARLY_REPLAY_PROTOCOL_VERSION
                    or str(
                        plan_diagnostics.get(
                            "agent_early_replay_source_trace_id"
                        )
                        or ""
                    )
                    != str(plan_trace.id)
                    or str(
                        plan_diagnostics.get(
                            "agent_early_replay_source_context_package_id"
                        )
                        or ""
                    )
                    != str(context_package.id)
                    or str(
                        plan_diagnostics.get(
                            "agent_early_replay_card_hash"
                        )
                        or ""
                    )
                    != str(replay_card.get("card_hash") or "")
                    or replay_card.get(
                        "gray_zone_rule_inputs_modified"
                    )
                    is not False
                    or replay_card.get(
                        "gray_zone_model_call_count"
                    )
                    != 0
                ):
                    _fail(
                        "AgentPlan early cache replay lineage is invalid"
                    )
                card_without_hash = {
                    key: value
                    for key, value in replay_card.items()
                    if key != "card_hash"
                }
                replayed_card_hash = hashlib.sha256(
                    _strict_json(
                        card_without_hash,
                        field="Agent early replay card",
                    ).encode("utf-8")
                ).hexdigest()
                if replayed_card_hash != str(
                    replay_card.get("card_hash") or ""
                ):
                    _fail(
                        "AgentPlan early cache replay card hash changed"
                    )
                source_binding_hash = hashlib.sha256(
                    _strict_json(
                        _source_agent_plan_binding_payload(
                            db,
                            source_plan,
                        ),
                        field="Agent early replay source plan binding",
                    ).encode("utf-8")
                ).hexdigest()
                if source_binding_hash != str(
                    replay_card.get(
                        "source_plan_binding_hash"
                    )
                    or ""
                ):
                    _fail(
                        "AgentPlan early cache source plan binding changed"
                    )
                for field_name, current_value, source_value in (
                    (
                        "query intent",
                        plan.query_intent_json or {},
                        source_plan.query_intent_json or {},
                    ),
                    (
                        "typed actions",
                        plan.typed_actions_json or [],
                        source_plan.typed_actions_json or [],
                    ),
                    (
                        "validation",
                        plan.validation_json or {},
                        source_plan.validation_json or {},
                    ),
                    (
                        "planner output",
                        plan.planner_model_json or {},
                        source_plan.planner_model_json or {},
                    ),
                    (
                        "frozen envelope",
                        plan.envelope_json or {},
                        source_plan.envelope_json or {},
                    ),
                    (
                        "execution controls",
                        plan_diagnostics.get(
                            "execution_controls"
                        )
                        or {},
                        source_plan_diagnostics.get(
                            "execution_controls"
                        )
                        or {},
                    ),
                ):
                    _exact_json(
                        current_value,
                        source_value,
                        field=(
                            "Agent early replay " + field_name
                        ),
                    )
            if (
                str(plan_trace.runtime_settings_hash or "")
                != str(trace.runtime_settings_hash or "")
                or str(plan_trace.policy_state_hash or "")
                != str(trace.policy_state_hash or "")
            ):
                _fail("AgentPlan Retrieval Trace runtime/policy lineage diverges")
            plan_trace_envelope_hash = _require_hash(
                plan_trace.agent_operating_envelope_hash,
                field=f"AgentPlan[{plan.plan_index}] RetrievalTrace envelope hash",
            )
            if (
                plan_trace_diagnostics.get("agent_operating_envelope_hash")
                != plan_trace_envelope_hash
            ):
                _fail("AgentPlan Retrieval Trace envelope identity is not reciprocal")
            terminal_plan_trace_id = str(plan_trace.id)
            terminal_plan_id = str(plan.id)
            terminal_plan_index = int(plan.plan_index)
        trace_bound = plan_trace is not None
        final_trace_bound = (
            plan_trace is not None and str(plan_trace.id) == str(trace.id)
        )
        pre_execution_rejection = (
            plan.retrieval_trace_id is None
            and str(plan.status or "")
            in {"validator_replan_requested", "invalid"}
        )
        if not trace_bound and not pre_execution_rejection:
            _fail("AgentPlan has a broken Retrieval Trace reciprocal link")
        validation = dict(plan.validation_json or {})
        _assert_finite_json(
            validation,
            field=f"AgentPlan[{plan.plan_index}].validation",
        )
        _assert_no_gray_authority(
            validation,
            field=f"AgentPlan[{plan.plan_index}].validation",
        )
        retrieval_granularity = validation.get(
            "retrieval_granularity_locked",
            validation.get("retrieval_granularity"),
        )
        if retrieval_granularity not in {"mid", "coarse"}:
            _fail("AgentPlan has no valid frozen retrieval granularity")
        proposed_actions_value = planner_model.get(
            "proposed_typed_actions"
        )
        if (
            planner_model.get("provider_response_recorded") is not False
            or "raw_output" in planner_model
            or not isinstance(proposed_actions_value, list)
            or not isinstance(
                planner_model.get("provider_output_hash"),
                str,
            )
            or len(planner_model["provider_output_hash"]) != 64
        ):
            _fail(
                "AgentPlan is missing its provider-safe proposed "
                "typed-action audit"
            )
        proposed_actions = list(proposed_actions_value)
        _assert_finite_json(
            proposed_actions,
            field=f"AgentPlan[{plan.plan_index}].proposed_typed_actions",
        )
        _assert_no_gray_authority(
            proposed_actions,
            field=f"AgentPlan[{plan.plan_index}].proposed_typed_actions",
        )
        replayed_actions, replayed_validation = validate_typed_actions(
            proposed_actions,
            frozen_envelope,
            db=db,
            knowledge_base_id=knowledge_base_id,
            require_required_actions=True,
            retrieval_granularity=retrieval_granularity,
            required_actions_override=(
                historical_typed_action_required_actions_for_replay(
                    validation
                )
            ),
            historical_target_layers_override=(
                historical_target_layers_for_replay(
                    proposed_actions,
                    validation,
                )
            ),
        )
        replayed_validation = {
            **replayed_validation,
            "plan_index": int(plan.plan_index),
            "retrieval_granularity_locked": retrieval_granularity,
            "unsupported_retrieval_granularity_rewrites_rejected": True,
        }
        replayed_validation = (
            rebind_historical_typed_action_validation_identity(
                validation,
                replayed_validation,
            )
        )
        _exact_json(
            validation,
            replayed_validation,
            field=f"AgentPlan[{plan.plan_index}] typed-action validator replay",
        )
        validator_input_attempt_count += int(
            replayed_validation.get("input_action_count") or 0
        )
        validator_accepted_attempt_count += sum(
            1
            for item in replayed_validation.get("accepted") or []
            if isinstance(item, Mapping) and item.get("index") is not None
        )
        validator_inserted_action_count += len(
            replayed_validation.get("inserted_required_actions") or []
        )
        actions = [
            _typed_action_payload(
                item,
                field=f"AgentPlan[{plan.plan_index}].typed_actions[{index}]",
            )
            for index, item in enumerate(list(plan.typed_actions_json or []))
        ]
        _exact_json(
            actions,
            replayed_actions,
            field=f"AgentPlan[{plan.plan_index}] normalized typed actions",
        )
        non_context_task_token_count += _address_free_token_count(
            {
                "query_intent": query_intent,
                "proposed_actions": proposed_actions,
                "validation": replayed_validation,
                "normalized_actions": actions,
            },
            field=f"AgentPlan[{plan.plan_index}].task_token_payload",
        )
        accepted = list(validation.get("accepted") or [])
        accepted_by_index: dict[int, Mapping[str, Any]] = {}
        for item in accepted:
            if (
                not isinstance(item, Mapping)
                or not _is_int(item.get("accepted_index"))
                or item["accepted_index"] in accepted_by_index
            ):
                _fail("AgentPlan accepted validator facts are malformed")
            accepted_by_index[int(item["accepted_index"])] = item
        if set(accepted_by_index) != set(range(len(actions))):
            _fail(
                "AgentPlan accepted indexes do not exactly cover its typed "
                "action scope"
            )
        # Every plan/replan round shares the same frozen envelope B, so this
        # must equal the run's one frozen-envelope hash -- not the reward's
        # own (possibly request-tightened, per-round effective) Retrieval
        # Trace envelope hash, which only the plan actually bound to that
        # trace is expected to derive from B via its own control card.
        if plan_diagnostics.get("agent_operating_envelope_hash") != frozen_envelope_hash:
            _fail(
                "AgentPlan operating-envelope hash diverges from the run's "
                "frozen operating envelope"
            )
        if plan_diagnostics.get("runtime_settings_hash") != str(
            trace.runtime_settings_hash or ""
        ):
            _fail("AgentPlan runtime-settings hash diverges from Retrieval Trace")
        replayed_controls: dict[str, Any] | None = None
        if validation["valid"]:
            if plan_trace is None:
                _fail("valid AgentPlan has no persisted Retrieval Trace")
            stored_controls = plan_diagnostics.get("execution_controls")
            if not isinstance(stored_controls, Mapping):
                _fail("valid AgentPlan is missing compiled execution controls")
            requested_result_top_k = _bounded_int(
                stored_controls.get("requested_result_top_k"),
                field=f"AgentPlan[{plan.plan_index}].requested_result_top_k",
                minimum=1,
                maximum=10_000,
            )
            replayed_controls = compile_typed_action_execution_controls(
                actions,
                frozen_envelope,
                requested_result_top_k=requested_result_top_k,
                retrieval_granularity=retrieval_granularity,
                validation_diagnostics=replayed_validation,
            )
            _exact_json(
                dict(stored_controls),
                replayed_controls,
                field=f"AgentPlan[{plan.plan_index}] execution controls",
            )
            control_hash = _require_hash(
                replayed_controls.get("control_hash"),
                field=f"AgentPlan[{plan.plan_index}].typed_action_control_hash",
            )
            if plan_diagnostics.get("typed_action_control_hash") != control_hash:
                _fail("AgentPlan compiled control hash is not reciprocal")
            if (
                (plan_trace.diagnostics_json or {}).get(
                    "typed_action_control_hash"
                )
                != control_hash
            ):
                _fail("AgentPlan and Retrieval Trace control hashes diverge")
        plan_action_rows = actions_by_plan.get(str(plan.id), [])
        primary_rows: dict[int, AgentAction] = {}
        appended_rows: list[AgentAction] = []
        for row in plan_action_rows:
            if not _is_int(row.action_index) or row.action_index < 0:
                _fail("AgentAction has an invalid action_index")
            if row.action_index < len(actions):
                if row.action_index in primary_rows:
                    _fail("AgentAction has a duplicate plan/action index")
                primary_rows[int(row.action_index)] = row
            else:
                appended_rows.append(row)
        if set(primary_rows) != set(range(len(actions))):
            _fail("AgentAction rows do not exactly cover AgentPlan actions")
        action_facts: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            action_type = str(action.get("action_type") or "")
            if not action_type:
                _fail("AgentPlan typed action has no action_type")
            accepted_card = accepted_by_index.get(index)
            validator = dict((accepted_card or {}).get("validation") or {})
            row = primary_rows[index]
            persisted_payload = {
                "action_type": str(row.action_type or ""),
                "target_ids": list(row.target_ids_json or []),
                "reason": str(row.reason or ""),
                "budget_request": dict(row.budget_request_json or {}),
                "expected_evidence": dict(row.expected_evidence_json or {}),
                "stop_condition": dict(row.stop_condition_json or {}),
            }
            _assert_no_gray_authority(
                action,
                field=f"AgentPlan[{plan.plan_index}].typed_action[{index}]",
            )
            _assert_no_gray_authority(
                persisted_payload,
                field=f"AgentAction[{index}].payload",
            )
            _exact_json(
                persisted_payload,
                action,
                field="AgentPlan typed action/AgentAction row",
            )
            persisted_validation = dict(row.validation_json or {})
            persisted_output = dict(row.output_json or {})
            persisted_diagnostics = dict(row.diagnostics_json or {})
            _assert_finite_json(
                persisted_diagnostics,
                field=f"AgentAction[{index}].diagnostics",
            )
            _assert_no_gray_authority(
                persisted_validation,
                field=f"AgentAction[{index}].validation",
            )
            _assert_no_gray_authority(
                persisted_output,
                field=f"AgentAction[{index}].output",
            )
            _assert_no_gray_authority(
                persisted_diagnostics,
                field=f"AgentAction[{index}].diagnostics",
            )
            for key, expected in validator.items():
                if persisted_validation.get(key) != expected:
                    _fail(
                        "AgentAction validator witness diverges from AgentPlan"
                    )
            if row.parent_action_id is not None:
                _fail("primary AgentAction unexpectedly has a parent action")
            if str(row.status or "") not in {
                "accepted",
                "completed",
                "rejected",
                "deferred",
                "no_progress",
            }:
                _fail("AgentAction status is not allowlisted")
            action_facts.append(
                {
                    "action_index": index,
                    "action_type": action_type,
                    "accepted": accepted_card is not None,
                    "schema_checked": validator.get("schema_checked") is True,
                    "budget_checked": validator.get("budget_checked") is True,
                    "target_ids_checked": validator.get("target_ids_checked") is True,
                    "target_scope_checked": validator.get("target_scope_checked") is True,
                    "valid": validator.get("valid") is True,
                    "typed_action": _address_free_json(
                        action,
                        field=f"AgentPlan[{plan.plan_index}].action[{index}]",
                    ),
                    "persisted_status": str(row.status or ""),
                    "persisted_validation": _address_free_json(
                        persisted_validation,
                        field=f"AgentAction[{index}].validation",
                    ),
                    "persisted_output": _address_free_json(
                        persisted_output,
                        field=f"AgentAction[{index}].output",
                    ),
                    "persisted_diagnostics": _address_free_json(
                        persisted_diagnostics,
                        field=f"AgentAction[{index}].diagnostics",
                    ),
                }
            )
            action_refs.append(
                {
                    "action_id": str(row.id),
                    "plan_id": str(plan.id),
                    "parent_action_id": None,
                    "target_ids": list(row.target_ids_json or []),
                    "payload_hash": stable_hash(persisted_payload),
                    "validation_hash": stable_hash(persisted_validation),
                    "output_hash": stable_hash(dict(row.output_json or {})),
                    "diagnostics_hash": stable_hash(persisted_diagnostics),
                }
            )
            non_context_task_token_count += _address_free_token_count(
                {
                    "payload": persisted_payload,
                    "validation": persisted_validation,
                    "output": persisted_output,
                    "diagnostics": persisted_diagnostics,
                },
                field=f"AgentAction[{index}].task_token_payload",
            )
        for row in appended_rows:
            appended_action_ids.add(str(row.id))
            validation_payload = dict(row.validation_json or {})
            if (
                validation_payload.get("repair_protocol_version") is None
                and validation_payload.get("repair_directive_validator_result")
                != "accepted"
            ):
                _fail("AgentAction exists outside the typed/repair action scope")
            parent_id = str(row.parent_action_id or "") or None
            if parent_id is not None:
                parent = persisted_action_by_id.get(parent_id)
                if (
                    parent is None
                    or str(parent.run_id) != str(run.id)
                    or str(parent.plan_id) != str(plan.id)
                ):
                    _fail("repair AgentAction parent is missing or cross-run")
            appended_payload = {
                "action_type": str(row.action_type or ""),
                "target_ids": list(row.target_ids_json or []),
                "reason": str(row.reason or ""),
                "budget_request": dict(row.budget_request_json or {}),
                "expected_evidence": dict(row.expected_evidence_json or {}),
                "stop_condition": dict(row.stop_condition_json or {}),
            }
            _assert_finite_json(
                appended_payload,
                field="repair AgentAction payload",
            )
            appended_output = dict(row.output_json or {})
            appended_diagnostics = dict(row.diagnostics_json or {})
            for field_name, payload in (
                ("payload", appended_payload),
                ("validation", validation_payload),
                ("output", appended_output),
                ("diagnostics", appended_diagnostics),
            ):
                _assert_finite_json(
                    payload,
                    field=f"repair AgentAction.{field_name}",
                )
                _assert_no_gray_authority(
                    payload,
                    field=f"repair AgentAction.{field_name}",
                )
            replayed_repairs, replayed_repair_validation = validate_typed_actions(
                [appended_payload],
                frozen_envelope,
                db=db,
                knowledge_base_id=knowledge_base_id,
                require_required_actions=False,
                retrieval_granularity=retrieval_granularity,
                historical_target_layers_override=(
                    historical_target_layers_for_replay(
                        [appended_payload],
                        validation_payload,
                    )
                ),
            )
            replayed_repair_validation = (
                rebind_historical_typed_action_validation_identity(
                    validation_payload,
                    replayed_repair_validation,
                )
            )
            if (
                replayed_repair_validation.get("valid") is not True
                or len(replayed_repairs) != 1
            ):
                _fail("repair AgentAction failed typed-action validator replay")
            _exact_json(
                appended_payload,
                replayed_repairs[0],
                field="repair AgentAction normalized typed action",
            )
            accepted_repair_cards = list(
                replayed_repair_validation.get("accepted") or []
            )
            if len(accepted_repair_cards) != 1:
                _fail("repair AgentAction has no unique validator witness")
            replayed_repair_witness = dict(
                (accepted_repair_cards[0] or {}).get("validation") or {}
            )
            for key, expected in replayed_repair_witness.items():
                if validation_payload.get(key) != expected:
                    _fail("repair AgentAction validator witness failed replay")
            for key, expected in (
                (
                    "typed_action_schema_protocol_version",
                    replayed_repair_validation.get(
                        "typed_action_schema_protocol_version"
                    ),
                ),
                (
                    "typed_action_schema_protocol_hash",
                    replayed_repair_validation.get(
                        "typed_action_schema_protocol_hash"
                    ),
                ),
            ):
                if validation_payload.get(key) != expected:
                    _fail("repair AgentAction schema identity failed replay")
            repair_protocol_version = str(
                (appended_payload.get("expected_evidence") or {}).get(
                    "protocol_version"
                )
                or ""
            )
            if (
                repair_protocol_version != TYPED_REPAIR_PROTOCOL_VERSION
                or validation_payload.get("repair_protocol_version")
                != TYPED_REPAIR_PROTOCOL_VERSION
            ):
                _fail("repair AgentAction protocol identity failed replay")
            repair_round_index = _bounded_int(
                validation_payload.get("repair_round_index"),
                field="repair AgentAction repair_round_index",
                maximum=100,
            )
            remaining_repair_budget = _bounded_int(
                validation_payload.get("remaining_repair_budget_before"),
                field="repair AgentAction remaining_repair_budget_before",
                minimum=1,
                maximum=100,
            )
            before_answer_hash = _require_hash(
                appended_diagnostics.get("before_answer_hash"),
                field="repair AgentAction before_answer_hash",
            )
            failure_audit = _replay_repair_failure_cards(
                appended_diagnostics.get("failure_cards"),
                action_type=str(row.action_type or ""),
                repair_round_index=repair_round_index,
                remaining_repair_budget=remaining_repair_budget,
                before_answer_hash=str(before_answer_hash),
                prior_repair_action_output_hashes=(
                    prior_repair_action_output_hashes
                ),
            )
            expected_evidence = dict(
                appended_payload.get("expected_evidence") or {}
            )
            if (
                list(expected_evidence.get("failure_card_hashes") or [])
                != failure_audit["failure_card_hashes"]
                or sorted(
                    set(
                        str(item)
                        for item in expected_evidence.get("failure_types") or []
                    )
                )
                != failure_audit["failure_types"]
            ):
                _fail("repair AgentAction failure-card witnesses diverge")
            expected_input_hash = str(
                expected_evidence.get("action_input_hash")
                or ""
            )
            if (
                not _SHA256_RE.fullmatch(expected_input_hash)
                or expected_input_hash != failure_audit["action_input_hash"]
                or validation_payload.get("action_input_hash")
                != expected_input_hash
                or appended_output.get("action_input_hash")
                != expected_input_hash
            ):
                _fail(
                    "repair AgentAction input hash failed persisted "
                    "failure-card replay"
                )
            expected_executor = str(
                expected_evidence.get("executor_mechanism")
                or ""
            )
            if (
                expected_executor
                != REPAIR_EXECUTOR_MECHANISMS.get(str(row.action_type or ""))
                or appended_output.get("executor_mechanism")
                != expected_executor
            ):
                _fail("repair AgentAction executor mechanism is not reciprocal")
            canonical_target_refs = expected_evidence.get(
                "canonical_target_refs"
            )
            if not isinstance(canonical_target_refs, Mapping):
                _fail("repair AgentAction has no canonical target references")
            canonical_target_refs = dict(canonical_target_refs)
            if set(canonical_target_refs) != {
                "claim_ids",
                "source_chunk_ids",
                "source_context_package_id",
                "source_retrieval_trace_id",
                "mid_concept_ids",
                "target_refs_hash",
            }:
                _fail("repair AgentAction canonical target schema is not closed")
            replayed_target_refs_hash = stable_hash(
                {
                    key: item
                    for key, item in canonical_target_refs.items()
                    if key != "target_refs_hash"
                }
            )
            if (
                canonical_target_refs.get("target_refs_hash")
                != replayed_target_refs_hash
            ):
                _fail("repair AgentAction canonical target hash failed replay")
            target_claim_ids = _require_unique_ids(
                list(canonical_target_refs.get("claim_ids") or []),
                field="repair canonical target claim_ids",
            )
            if sorted(target_claim_ids) != failure_audit["claim_ids"]:
                _fail("repair canonical target claims diverge from failure cards")
            target_chunk_ids = _require_unique_ids(
                list(canonical_target_refs.get("source_chunk_ids") or []),
                field="repair canonical target source_chunk_ids",
            )
            source_package = _load_row(
                db,
                ContextPackage,
                canonical_target_refs.get("source_context_package_id"),
                field="repair canonical target source_context_package_id",
            )
            source_trace = _load_row(
                db,
                RetrievalTrace,
                canonical_target_refs.get("source_retrieval_trace_id"),
                field="repair canonical target source_retrieval_trace_id",
            )
            _require_kb(
                source_package,
                knowledge_base_id,
                field="repair canonical target ContextPackage",
            )
            _require_kb(
                source_trace,
                knowledge_base_id,
                field="repair canonical target RetrievalTrace",
            )
            if (
                str(source_package.retrieval_trace_id or "")
                != str(source_trace.id)
                or str(source_package.query or "") != str(run.question or "")
                or str(source_trace.query or "") != str(run.question or "")
            ):
                _fail("repair canonical target package/trace lineage is broken")
            target_chunks = _ordered_rows_by_ids(
                db,
                Chunk,
                target_chunk_ids,
                field="repair canonical target source_chunk_ids",
            )
            for target_chunk in target_chunks:
                _require_kb(
                    target_chunk,
                    knowledge_base_id,
                    field="repair canonical target Chunk",
                )
            package_chunk_scope = list(
                dict.fromkeys(
                    [
                        *list(source_package.hit_chunk_ids_json or []),
                        *list(source_package.restored_chunk_ids_json or []),
                        *list(source_package.bridge_chunk_ids_json or []),
                        *[
                            str(item.get("chunk_id"))
                            for item in (
                                (source_package.package_json or {}).get(
                                    "chunks",
                                    [],
                                )
                            )
                            if isinstance(item, Mapping)
                            and item.get("chunk_id")
                        ],
                    ]
                )
            )
            if set(target_chunk_ids).difference(package_chunk_scope):
                _fail("repair canonical target chunks cross the source package")
            target_mid_ids = _require_unique_ids(
                list(canonical_target_refs.get("mid_concept_ids") or []),
                field="repair canonical target mid_concept_ids",
            )
            target_mid_rows = _ordered_rows_by_ids(
                db,
                MidConcept,
                target_mid_ids,
                field="repair canonical target mid_concept_ids",
            )
            for target_mid in target_mid_rows:
                _require_kb(
                    target_mid,
                    knowledge_base_id,
                    field="repair canonical target MidConcept",
                )
            repair_input_audit_by_action_id[str(row.id)] = {
                "repair_round_index": repair_round_index,
                "remaining_repair_budget_before": remaining_repair_budget,
                "source_context_package_id": str(source_package.id),
                "source_retrieval_trace_id": str(source_trace.id),
                "source_chunk_ids": target_chunk_ids,
                "target_refs_hash": replayed_target_refs_hash,
                "failure_card_hashes": list(
                    failure_audit["failure_card_hashes"]
                ),
                "semantic_failure_hashes": list(
                    failure_audit["semantic_failure_hashes"]
                ),
                "failure_set_hash": str(failure_audit["failure_set_hash"]),
                "action_input_hash": expected_input_hash,
            }
            output_hash = _require_hash(
                appended_output.get("action_output_hash"),
                field="repair AgentAction action_output_hash",
            )
            prior_repair_action_output_hashes.append(str(output_hash))
            action_refs.append(
                {
                    "action_id": str(row.id),
                    "plan_id": str(plan.id),
                    "parent_action_id": parent_id,
                    "target_ids": list(row.target_ids_json or []),
                    "payload_hash": stable_hash(appended_payload),
                    "validation_hash": stable_hash(validation_payload),
                    "output_hash": stable_hash(appended_output),
                    "diagnostics_hash": stable_hash(appended_diagnostics),
                }
            )
            action_facts.append(
                {
                    "action_index": int(row.action_index),
                    "action_type": str(row.action_type or ""),
                    "accepted": True,
                    "repair_action": True,
                    "typed_action": _address_free_json(
                        appended_payload,
                        field="repair AgentAction payload",
                    ),
                    "persisted_status": str(row.status or ""),
                    "persisted_validation": _address_free_json(
                        validation_payload,
                        field="repair AgentAction validation",
                    ),
                    "persisted_output": _address_free_json(
                        appended_output,
                        field="repair AgentAction output",
                    ),
                    "persisted_diagnostics": _address_free_json(
                        appended_diagnostics,
                        field="repair AgentAction diagnostics",
                    ),
                    "repair_input_replay": _address_free_json(
                        repair_input_audit_by_action_id[str(row.id)],
                        field="repair AgentAction input replay",
                    ),
                }
            )
            non_context_task_token_count += _address_free_token_count(
                {
                    "payload": appended_payload,
                    "validation": validation_payload,
                    "output": appended_output,
                    "diagnostics": appended_diagnostics,
                },
                field="repair AgentAction.task_token_payload",
            )
        schema_hash = validation.get("typed_action_schema_protocol_hash")
        if schema_hash is not None:
            _require_hash(
                schema_hash,
                field="AgentPlan.typed_action_schema_protocol_hash",
            )
        rejected = list(validation.get("rejected") or [])
        inserted_required = list(
            validation.get("inserted_required_actions") or []
        )
        _assert_finite_json(rejected, field="AgentPlan.validator_rejected")
        _assert_finite_json(
            inserted_required,
            field="AgentPlan.inserted_required_actions",
        )
        plan_facts.append(
            {
                "plan_index": int(plan.plan_index),
                "status": str(plan.status or ""),
                "retrieval_trace_bound": trace_bound,
                "final_reward_trace_bound": final_trace_bound,
                "pre_execution_rejection": pre_execution_rejection,
                "valid": bool(validation["valid"]),
                "typed_action_schema_protocol_version": str(
                    validation.get("typed_action_schema_protocol_version") or ""
                ),
                "typed_action_schema_protocol_hash": schema_hash,
                "rejected_count": len(rejected),
                "inserted_required_action_count": len(inserted_required),
                "validator_rejected": _address_free_json(
                    rejected,
                    field="AgentPlan.validator_rejected",
                ),
                "inserted_required_actions": _address_free_json(
                    inserted_required,
                    field="AgentPlan.inserted_required_actions",
                ),
                "execution_control_hash": (
                    replayed_controls.get("control_hash")
                    if replayed_controls is not None
                    else None
                ),
                "actions": action_facts,
            }
        )

    events = list(
        db.scalars(
            select(AgentTraceEvent).where(
                AgentTraceEvent.run_id == run.id,
                AgentTraceEvent.created_at <= reward_created_at,
            ).order_by(AgentTraceEvent.sequence_index.asc())
        ).all()
    )
    if len(events) > MAX_REWARD_ROWS:
        _fail("Agent trace event count exceeds the bounded reward replay limit")
    if any(row.created_at is None for row in events):
        _fail("AgentTraceEvent is missing its durable replay cutoff timestamp")
    sequence_indexes = [row.sequence_index for row in events]
    if (
        any(not _is_int(value) or value < 0 for value in sequence_indexes)
        or sequence_indexes != list(range(len(events)))
    ):
        _fail("AgentTraceEvent sequence indexes are missing or non-contiguous")
    event_facts: list[dict[str, Any]] = []
    event_refs: list[dict[str, Any]] = []
    latency_ms = 0
    for event_index, event in enumerate(events):
        duration = _bounded_int(
            event.duration_ms,
            field="AgentTraceEvent.duration_ms",
            maximum=MAX_EVENT_DURATION_MS,
        )
        latency_ms += duration
        if latency_ms > MAX_TOTAL_EVENT_DURATION_MS:
            _fail("Agent trace total duration exceeds the bounded reward envelope")
        if not str(event.node or "") or not str(event.status or ""):
            _fail("AgentTraceEvent has an empty node or status")
        input_summary = str(event.input_summary or "")
        output_summary = str(event.output_summary or "")
        structured_input_summary = _structured_summary_payload(
            input_summary,
            field=f"AgentTraceEvent[{event_index}].input_summary",
        )
        structured_output_summary = _structured_summary_payload(
            output_summary,
            field=f"AgentTraceEvent[{event_index}].output_summary",
        )
        scores = dict(event.scores or {})
        _assert_finite_json(scores, field="AgentTraceEvent.scores")
        _assert_no_gray_authority(scores, field="AgentTraceEvent.scores")
        event_facts.append(
            {
                "sequence_index": event_index,
                "node": str(event.node),
                "status": str(event.status),
                "duration_ms": duration,
                "input_summary_hash": hashlib.sha256(
                    input_summary.encode("utf-8")
                ).hexdigest(),
                "output_summary_hash": hashlib.sha256(
                    output_summary.encode("utf-8")
                ).hexdigest(),
                "input_summary_token_count": rough_token_count(input_summary),
                "output_summary_token_count": rough_token_count(output_summary),
                "structured_input_summary": (
                    _address_free_json(
                        structured_input_summary,
                        field=f"AgentTraceEvent[{event_index}].input_summary",
                    )
                    if structured_input_summary is not None
                    else None
                ),
                "structured_output_summary": (
                    _address_free_json(
                        structured_output_summary,
                        field=f"AgentTraceEvent[{event_index}].output_summary",
                    )
                    if structured_output_summary is not None
                    else None
                ),
                "scores": _address_free_json(
                    scores,
                    field="AgentTraceEvent.scores",
                ),
            }
        )
        event_refs.append(
            {
                "event_id": str(event.id),
                "input_summary_hash": hashlib.sha256(
                    input_summary.encode("utf-8")
                ).hexdigest(),
                "output_summary_hash": hashlib.sha256(
                    output_summary.encode("utf-8")
                ).hexdigest(),
                "scores_hash": stable_hash(scores),
            }
        )
        non_context_task_token_count += (
            rough_token_count(input_summary)
            + rough_token_count(output_summary)
            + _address_free_token_count(
                scores,
                field=f"AgentTraceEvent[{event_index}].scores",
            )
        )

    observations = list(
        db.scalars(
            select(AgentObservation).where(
                AgentObservation.run_id == run.id,
                AgentObservation.created_at <= reward_created_at,
            )
        ).all()
    )
    if len(observations) > MAX_REWARD_ROWS:
        _fail("Agent observation count exceeds the bounded reward replay limit")
    latest_observation_by_action: dict[str, AgentObservation] = {}
    for observation in observations:
        action_id = str(observation.action_id or "")
        if not action_id:
            continue
        previous = latest_observation_by_action.get(action_id)
        if previous is None or (
            str(observation.created_at or ""),
            str(observation.id),
        ) > (
            str(previous.created_at or ""),
            str(previous.id),
        ):
            latest_observation_by_action[action_id] = observation
    action_ids = sorted(
        {
            str(row.action_id)
            for row in observations
            if str(row.action_id or "")
        }
    )
    actions = _ordered_rows_by_ids(
        db,
        AgentAction,
        action_ids,
        field="AgentObservation.action_id",
    )
    actions_by_id = {str(row.id): row for row in actions}
    for action_id, action in actions_by_id.items():
        if (
            action_id not in persisted_action_by_id
            or str(action.run_id or "") != str(run.id)
            or str(action.plan_id or "") not in plan_ids
        ):
            _fail(
                "AgentObservation references a missing, cross-run, or "
                "cross-plan AgentAction"
            )
    observation_facts: list[dict[str, Any]] = []
    repair_by_index: dict[int, dict[str, Any]] = {}
    repair_links_by_index: dict[int, dict[str, str]] = {}
    observation_refs: list[dict[str, Any]] = []
    for observation in observations:
        observation_payload = dict(observation.observation_json or {})
        observation_diagnostics = dict(observation.diagnostics_json or {})
        _assert_finite_json(
            observation_payload,
            field="AgentObservation.observation_json",
        )
        _assert_finite_json(
            observation_diagnostics,
            field="AgentObservation.diagnostics_json",
        )
        _assert_no_gray_authority(
            observation_payload,
            field="AgentObservation.observation_json",
        )
        _assert_no_gray_authority(
            observation_diagnostics,
            field="AgentObservation.diagnostics_json",
        )
        action_id = str(observation.action_id or "")
        if (
            action_id
            and str(latest_observation_by_action[action_id].id)
            == str(observation.id)
        ):
            action = actions_by_id.get(action_id)
            if action is None:
                _fail("AgentObservation has a missing/cross-run AgentAction")
            _exact_json(
                dict(action.output_json or {}),
                observation_payload,
                field="AgentAction/AgentObservation output",
            )
        evidence_ids = _require_unique_ids(
            list(observation.evidence_chunk_ids_json or []),
            field="AgentObservation.evidence_chunk_ids",
        )
        if set(evidence_ids).difference(chunk_key_by_id):
            _fail("AgentObservation evidence crosses the reward chunk scope")
        base_fact = {
            "observation_type": str(observation.observation_type or ""),
            "verdict": str(observation.verdict or ""),
            "evidence_chunk_business_keys": sorted(
                chunk_key_by_id[row_id] for row_id in evidence_ids
            ),
            "observation": _address_free_json(
                observation_payload,
                field="AgentObservation.observation_json",
            ),
            "diagnostics": _address_free_json(
                observation_diagnostics,
                field="AgentObservation.diagnostics_json",
            ),
        }
        observation_refs.append(
            {
                "observation_id": str(observation.id),
                "action_id": str(observation.action_id or "") or None,
                "observation_hash": stable_hash(observation_payload),
                "diagnostics_hash": stable_hash(observation_diagnostics),
            }
        )
        non_context_task_token_count += _address_free_token_count(
            {
                "observation": observation_payload,
                "diagnostics": observation_diagnostics,
            },
            field="AgentObservation.task_token_payload",
        )
        if observation.observation_type == "typed_repair_round":
            payload = dict(observation.observation_json or {})
            round_index = payload.get("repair_round_index")
            if not _is_int(round_index) or round_index < 0:
                _fail("persisted repair observation has no valid round index")
            if round_index in repair_by_index:
                _fail("persisted repair observations contain a duplicate round")
            action = actions_by_id.get(str(observation.action_id or ""))
            if action is None or str(action.run_id) != str(run.id):
                _fail("repair observation has a missing/cross-run AgentAction")
            repair_input_audit = repair_input_audit_by_action_id.get(
                str(action.id)
            )
            if repair_input_audit is None:
                _fail("repair observation has no persisted input replay")
            if str(action.plan_id or "") not in {str(row.id) for row in plans}:
                _fail("repair AgentAction has a broken AgentPlan reciprocal link")
            _exact_json(
                dict(action.output_json or {}),
                payload,
                field="repair action/observation output",
            )
            if str(action.action_type or "") != str(payload.get("action_type") or ""):
                _fail("repair AgentAction type does not match its observation")
            if int(round_index) != repair_input_audit["repair_round_index"]:
                if int(round_index) in repair_by_index:
                    _fail("persisted repair observations contain a duplicate round")
                _fail("persisted repair rounds are missing or non-contiguous")
            if (
                payload.get("remaining_repair_budget_before")
                != repair_input_audit["remaining_repair_budget_before"]
                or payload.get("action_input_hash")
                != repair_input_audit["action_input_hash"]
                or list(payload.get("failure_card_hashes") or [])
                != repair_input_audit["failure_card_hashes"]
            ):
                _fail("repair observation diverges from persisted input facts")
            before_package = _load_row(
                db,
                ContextPackage,
                payload.get("before_context_package_id"),
                field="repair.before_context_package_id",
            )
            repaired_package = _load_row(
                db,
                ContextPackage,
                payload.get("repaired_context_package_id"),
                field="repair.repaired_context_package_id",
            )
            before_trace = _load_row(
                db,
                RetrievalTrace,
                payload.get("before_retrieval_trace_id"),
                field="repair.before_retrieval_trace_id",
            )
            repaired_trace = _load_row(
                db,
                RetrievalTrace,
                payload.get("repaired_retrieval_trace_id"),
                field="repair.repaired_retrieval_trace_id",
            )
            for linked, field_name in (
                (before_package, "repair before ContextPackage"),
                (repaired_package, "repair after ContextPackage"),
                (before_trace, "repair before RetrievalTrace"),
                (repaired_trace, "repair after RetrievalTrace"),
            ):
                _require_kb(linked, knowledge_base_id, field=field_name)
            if (
                str(before_package.retrieval_trace_id or "")
                != str(before_trace.id)
                or str(repaired_package.retrieval_trace_id or "")
                != str(repaired_trace.id)
            ):
                _fail("repair ContextPackage/RetrievalTrace links are broken")
            if (
                str(before_package.id)
                != repair_input_audit["source_context_package_id"]
                or str(before_trace.id)
                != repair_input_audit["source_retrieval_trace_id"]
            ):
                _fail(
                    "repair canonical target package/trace is not bound to "
                    "the observed before-chain"
                )
            action_validation = dict(action.validation_json or {})
            validated_targets = action_validation.get("validated_targets")
            if not isinstance(validated_targets, Mapping):
                _fail("repair action has no persisted validated targets")
            output_validated_targets = payload.get("validated_targets")
            if not isinstance(output_validated_targets, Mapping):
                _fail("repair observation has no persisted validated targets")
            _exact_json(
                dict(validated_targets),
                dict(output_validated_targets),
                field="repair validated targets action/observation",
            )
            expected_target_refs = (
                (action.expected_evidence_json or {}).get(
                    "canonical_target_refs"
                )
                or {}
            )
            _exact_json(
                dict(validated_targets).get("canonical_target_refs"),
                expected_target_refs,
                field="repair canonical target validator witness",
            )
            for package_row, trace_row in (
                (before_package, before_trace),
                (repaired_package, repaired_trace),
            ):
                if (
                    str(package_row.query or "") != str(run.question or "")
                    or str(trace_row.query or "") != str(run.question or "")
                    or package_row.runtime_settings_hash
                    != trace_row.runtime_settings_hash
                ):
                    _fail("repair package/trace query or runtime identity diverges")
                _build_chunk_facts(
                    db,
                    knowledge_base_id=knowledge_base_id,
                    package=package_row,
                    trace=trace_row,
                    extra_chunk_ids=[],
                    allow_empty_results=True,
                    replay_cutoff=reward_created_at,
                )
            before_progress = _repair_progress_for_bundle(
                before_package,
                {},
            )
            after_progress = _repair_progress_for_bundle(
                repaired_package,
                {},
            )
            _exact_json(
                payload.get("before_progress"),
                before_progress,
                field="repair.before_progress",
            )
            _exact_json(
                payload.get("after_progress"),
                after_progress,
                field="repair.after_progress",
            )
            if (
                payload.get("before_progress_hash")
                != before_progress.get("progress_hash")
                or payload.get("after_progress_hash")
                != after_progress.get("progress_hash")
            ):
                _fail("repair progress hashes failed persisted package replay")
            before = _bounded_int(
                payload.get("remaining_repair_budget_before"),
                field="repair.remaining_budget_before",
                maximum=100,
            )
            after = _bounded_int(
                payload.get("remaining_repair_budget_after"),
                field="repair.remaining_budget_after",
                maximum=100,
            )
            if after != before - 1:
                _fail("repair round does not consume exactly one repair budget")
            made_progress = payload.get("made_semantic_progress")
            if not isinstance(made_progress, bool):
                _fail("repair semantic-progress fact is not a boolean")
            replayed_made_progress = repair_made_progress(
                before_progress,
                after_progress,
            )
            if made_progress is not replayed_made_progress:
                _fail("repair semantic progress failed package-fact replay")
            convergence_reason = str(payload.get("convergence_reason") or "")
            if convergence_reason not in {
                "all_claims_supported",
                "continue",
                "no_progress_try_alternate_direction",
            }:
                _fail("repair convergence reason is not allowlisted")
            if convergence_reason == "continue" and not made_progress:
                _fail("repair continue convergence requires semantic progress")
            if (
                convergence_reason == "no_progress_try_alternate_direction"
                and made_progress
            ):
                _fail("repair no-progress convergence conflicts with progress")
            if convergence_reason == "all_claims_supported":
                if (
                    str(repaired_package.id) != str(context_package.id)
                    or claim_grounded_gate.get("all_claims_supported") is not True
                ):
                    _fail("repair all-claims-supported convergence is not final-gate bound")
                stored_gate_semantic_card = (
                    action.diagnostics_json or {}
                ).get("after_gate_semantic_card")
                if not isinstance(stored_gate_semantic_card, Mapping):
                    _fail(
                        "repair all-claims-supported convergence has no "
                        "persisted semantic gate card"
                    )
                _exact_json(
                    dict(stored_gate_semantic_card),
                    repair_gate_semantic_card(dict(claim_grounded_gate)),
                    field="repair final semantic gate",
                )
            action_diagnostics = dict(action.diagnostics_json or {})
            directive_hash = _require_hash(
                action_validation.get("repair_directive_hash"),
                field="repair.repair_directive_hash",
            )
            expected_output_hash = stable_hash(
                {
                    "repair_protocol_version": payload.get("protocol_version"),
                    "action_input_hash": payload.get("action_input_hash"),
                    "directive_hash": directive_hash,
                    "before_progress_hash": before_progress["progress_hash"],
                    "after_progress_hash": after_progress["progress_hash"],
                    "after_answer_hash": action_diagnostics.get(
                        "after_answer_hash"
                    ),
                    "after_gate_hash": action_diagnostics.get("after_gate_hash"),
                }
            )
            if payload.get("action_output_hash") != expected_output_hash:
                _fail("repair action output hash failed deterministic replay")
            expected_verdict = (
                "sufficient"
                if convergence_reason == "all_claims_supported"
                else "observed"
                if made_progress
                else "no_progress"
            )
            if str(observation.verdict or "") != expected_verdict:
                _fail("repair observation verdict does not match progress facts")
            if (
                payload.get("global_top_k_increased") is not False
                or payload.get("gray_zone_model_call_count") != 0
                or payload.get("gray_zone_decision_authority")
                != "deterministic_executor_only"
            ):
                _fail("repair observation violates frozen graph-control authority")
            repair_fact = {
                **base_fact,
                "repair_round_index": int(round_index),
                "remaining_repair_budget_before": before,
                "remaining_repair_budget_after": after,
                "protocol_version": str(payload.get("protocol_version") or ""),
                "action_type": str(payload.get("action_type") or ""),
                "executor_mechanism": str(payload.get("executor_mechanism") or ""),
                "before_failure_types": sorted(
                    set(str(item) for item in payload.get("before_failure_types") or [])
                ),
                "after_failure_types": sorted(
                    set(str(item) for item in payload.get("after_failure_types") or [])
                ),
                "made_semantic_progress": made_progress,
                "convergence_reason": convergence_reason,
                "retrieval_granularity": str(
                    payload.get("retrieval_granularity") or ""
                ),
                "result_top_k": _bounded_int(
                    payload.get("result_top_k"),
                    field="repair.result_top_k",
                    maximum=10_000,
                ),
                "global_top_k_increased": False,
                "gray_zone_model_call_count": 0,
                "gray_zone_decision_authority": "deterministic_executor_only",
            }
            repair_by_index[int(round_index)] = repair_fact
            repair_links_by_index[int(round_index)] = {
                "before_context_package_id": str(before_package.id),
                "repaired_context_package_id": str(repaired_package.id),
                "before_retrieval_trace_id": str(before_trace.id),
                "repaired_retrieval_trace_id": str(repaired_trace.id),
            }
            observation_facts.append(repair_fact)
        elif observation.observation_type == "claim_level_final_grounded_gate":
            payload = dict(observation.observation_json or {})
            if (
                payload.get("deterministic_citation_guard") is not True
                or payload.get("gray_zone_model_call_count") != 0
                or str(payload.get("grounding_outcome") or "") != grounding_outcome
            ):
                _fail("final grounded gate observation is not deterministic/bound")
            _exact_json(
                dict(payload.get("evidence_gap") or {}),
                dict(evidence_gap),
                field="final grounded gate evidence_gap",
            )
            observation_facts.append(
                {
                    **base_fact,
                    "deterministic_citation_guard": True,
                    "grounding_outcome": grounding_outcome,
                    "evidence_gap": _uuid_free_json(
                        dict(evidence_gap),
                        field="final_grounded_gate.evidence_gap",
                    ),
                    "gray_zone_model_call_count": 0,
                }
            )
        else:
            observation_facts.append(base_fact)

    if appended_action_ids.difference(action_ids):
        _fail("repair AgentAction exists without a reciprocal observation")
    repair_indexes = sorted(repair_by_index)
    if repair_indexes != list(range(len(repair_indexes))):
        _fail("persisted repair rounds are missing or non-contiguous")
    if "repair_rounds_used" in evidence_gap and evidence_gap.get(
        "repair_rounds_used"
    ) != len(repair_indexes):
        _fail("evidence_gap repair-round count is not reciprocal")
    for index in repair_indexes[1:]:
        previous = repair_links_by_index[index - 1]
        current = repair_links_by_index[index]
        if (
            current["before_context_package_id"]
            != previous["repaired_context_package_id"]
            or current["before_retrieval_trace_id"]
            != previous["repaired_retrieval_trace_id"]
        ):
            _fail("repair rounds do not form a contiguous package/trace chain")
    if repair_indexes:
        first_link = repair_links_by_index[repair_indexes[0]]
        if terminal_plan_trace_id != first_link["before_retrieval_trace_id"]:
            _fail(
                "terminal AgentPlan is not bound to the first repair "
                "before-trace"
            )
        final_link = repair_links_by_index[repair_indexes[-1]]
        if (
            final_link["repaired_context_package_id"]
            != str(context_package.id)
            or final_link["repaired_retrieval_trace_id"]
            != str(context_package.retrieval_trace_id or "")
        ):
            _fail("repair chain does not terminate at the rewarded Context Package")
    elif terminal_plan_trace_id != str(trace.id):
        _fail("terminal AgentPlan is not bound to the rewarded Retrieval Trace")
    if (
        terminal_plan_trace_id is None
        or terminal_plan_id is None
        or terminal_plan_index is None
    ):
        _fail("rewarded Agent run has no terminal executed AgentPlan")
    repair_facts = [repair_by_index[index] for index in repair_indexes]
    repair_success_rate = (
        sum(
            1
            for row in repair_facts
            if row["made_semantic_progress"]
            or row["convergence_reason"] == "all_claims_supported"
        )
        / len(repair_facts)
        if repair_facts
        else None
    )
    validator_attempt_count = (
        validator_input_attempt_count + validator_inserted_action_count
    )
    validation_rate = (
        validator_accepted_attempt_count / validator_attempt_count
        if validator_attempt_count
        else 0.0
    )
    if non_context_task_token_count > MAX_REWARD_TOKENS:
        _fail("persisted Agent task-token accounting exceeds the reward limit")
    facts = {
        "run_route": str(run.route or ""),
        "question_hash": hashlib.sha256(str(run.question or "").encode("utf-8")).hexdigest(),
        "plans": plan_facts,
        "trace_events": event_facts,
        "observations": sorted(
            observation_facts,
            key=lambda row: (
                row["observation_type"],
                int(row.get("repair_round_index", -1)),
                _strict_json(row, field="observation_fact"),
            ),
        ),
        "repair_rounds": repair_facts,
        "typed_action_validation_accounting": {
            "protocol_version": "replayed_typed_action_attempt_accounting_v1",
            "input_attempt_count": validator_input_attempt_count,
            "accepted_input_attempt_count": validator_accepted_attempt_count,
            "inserted_required_action_count": validator_inserted_action_count,
            "failed_or_rejected_attempt_count": max(
                0,
                validator_attempt_count - validator_accepted_attempt_count,
            ),
            "attempt_count": validator_attempt_count,
        },
        "task_token_accounting": {
            "protocol_version": "persisted_agent_task_payload_tokens_v2",
            "non_context_task_token_count": non_context_task_token_count,
            "tokenizer": "rough_token_count",
        },
    }
    metrics = {
        "typed_action_validation_pass_rate": round(validation_rate, 6),
        "latency_ms": latency_ms,
        "non_context_task_token_count": non_context_task_token_count,
        "frozen_operating_envelope": frozen_envelope,
        "frozen_operating_envelope_hash": frozen_envelope_hash,
        "repair_success_rate": (
            round(repair_success_rate, 6)
            if repair_success_rate is not None
            else None
        ),
    }
    refs = {
        "agent_run_id": str(run.id),
        "agent_plan_ids": [str(row.id) for row in plans],
        "agent_actions": sorted(
            action_refs,
            key=lambda row: (str(row["plan_id"]), str(row["action_id"])),
        ),
        "agent_trace_events": event_refs,
        "agent_observations": sorted(
            observation_refs,
            key=lambda row: (
                str(row["observation_id"]),
                str(row["action_id"] or ""),
            ),
        ),
    }
    return facts, metrics, refs


def _cache_business_fact(
    trace: RetrievalTrace,
    *,
    chunk_business_scope_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics = dict(trace.diagnostics_json or {})
    cache_key = _require_hash(diagnostics.get("cache_key"), field="trace.cache_key")
    components = diagnostics.get("cache_key_components")
    if not isinstance(components, Mapping):
        _fail("Retrieval Trace is missing persisted cache-key components")
    components = dict(components)
    _assert_finite_json(components, field="trace.cache_key_components")
    if stable_hash(components) != cache_key:
        _fail("Retrieval Trace cache key failed exact component replay")
    reciprocal = {
        "knowledge_base_id": str(trace.knowledge_base_id),
        "query": str(trace.query or ""),
        "filters": dict(trace.filters_json or {}),
        "runtime_settings_hash": trace.runtime_settings_hash,
        "policy_state_hash": trace.policy_state_hash,
        "traversal_protocol_hash": trace.traversal_protocol_hash,
        "retrieval_mode": trace.retrieval_mode,
    }
    for key, expected in reciprocal.items():
        if components.get(key) != expected:
            _fail(f"Retrieval Trace cache component {key} is not reciprocal")
    envelope = dict(
        (trace.diagnostics_json or {}).get("agent_operating_envelope") or {}
    )
    expected_cache_envelope_hash = canonical_graph_hash(
        "agent_operating_envelope_state_v1",
        envelope,
    )
    if (
        components.get("agent_operating_envelope_hash")
        != expected_cache_envelope_hash
    ):
        _fail("Retrieval Trace cache envelope business hash failed replay")
    business_keys = (
        "embedding_text_version",
        "local_hint_protocol_version",
        "contextual_index_business_hash",
        "context_graph_hash",
        "chunk_business_scope_hash",
        "structure_graph_hash",
        "chunk_relation_graph_hash",
        "relation_rank_score_protocol_hash",
        "relation_raw_strength_protocol_hash",
        "chunk_node_quality_protocol_hash",
        "out_evidence_mass_protocol_hash",
        "in_acceptance_capacity_protocol_hash",
        "relation_quota_protocol_hash",
        "edge_type_calibration_protocol_hash",
        "graph_operating_point_hash",
        "calibration_params_hash",
        "edge_type_calibration_config_hash",
        "rq_membership_hash",
        "rq_prefix_pair_aggregate_hash",
        "rq_membership_protocol_hash",
        "mid_concept_hash",
        "coarse_concept_hash",
        "edge_distance_protocol_hash",
        "edge_projection_protocol_hash",
        "graph_protocol_runtime_identity_hash",
        "traversal_protocol_hash",
        "traversal_observation_budget",
        "query_facet_protocol_hash",
        "query_facets_hash",
        "runtime_settings_hash",
        "policy_state_hash",
        "agent_operating_envelope_hash",
        "prompt_protocol_hash",
        "profile_hash",
        "canonical_profile_state_hash",
        "canonical_graph_hash_protocol_version",
        "canonical_vector_identity",
        "retrieval_mode",
        "retrieval_granularity",
        "result_top_k",
        "typed_action_executor_protocol_version",
        "typed_action_allowed_relation_types",
        "repair_directive_protocol_version",
        "repair_action_type",
    )
    raw_business = {key: components.get(key) for key in business_keys}
    canonical_vector_identity = raw_business.get("canonical_vector_identity")
    if isinstance(canonical_vector_identity, (Mapping, list)):
        raw_business["canonical_vector_identity"] = canonical_graph_hash(
            "policy_reward_cache_vector_identity_v1",
            canonical_vector_identity,
        )
    business = {
        "query_hash": hashlib.sha256(str(trace.query or "").encode("utf-8")).hexdigest(),
        "filters_hash": canonical_graph_hash(
            "policy_reward_filters_v1",
            dict(trace.filters_json or {}),
        ),
        **raw_business,
    }
    if business["chunk_business_scope_hash"] is None:
        business["chunk_business_scope_hash"] = chunk_business_scope_hash
    return business, {"cache_key": cache_key, "cache_key_components": components}


def build_policy_reward_replay(
    db: Session,
    reward: RewardEvent | str,
) -> dict[str, Any]:
    """Rebuild UUID-free policy reward evidence from persisted PostgreSQL facts.

    The returned ``content_card`` and both hashes contain no database row
    addresses. ``refs`` carries those addresses and is still compared exactly
    during replay.
    """

    row = (
        _load_row(db, RewardEvent, reward, field="reward_event")
        if isinstance(reward, str)
        else reward
    )
    if not isinstance(row, RewardEvent) or not str(row.id or ""):
        _fail("RewardEvent must be flushed before reward replay")
    knowledge_base_id = str(row.knowledge_base_id or "")
    if not knowledge_base_id:
        _fail("RewardEvent has no knowledge-base reference")
    _assert_finite_json(row.reward_json or {}, field="reward.reward_json")
    _assert_finite_json(row.context_json or {}, field="reward.context_json")
    _assert_finite_json(row.action_json or {}, field="reward.action_json")
    _assert_finite_json(row.diagnostics_json or {}, field="reward.diagnostics_json")
    _finite_float(row.propensity, field="reward.propensity", minimum=0.0, maximum=1.0)

    trace = _load_row(
        db,
        RetrievalTrace,
        str(row.retrieval_trace_id or ""),
        field="reward.retrieval_trace_id",
    )
    answer = _load_row(
        db,
        AnswerSession,
        str(row.answer_session_id or ""),
        field="reward.answer_session_id",
    )
    context = dict(row.context_json or {})
    package = _load_row(
        db,
        ContextPackage,
        context.get("context_package_id"),
        field="reward.context_json.context_package_id",
    )
    run = _load_row(
        db,
        AgentRun,
        context.get("agent_run_id"),
        field="reward.context_json.agent_run_id",
    )
    for linked, field in (
        (trace, "RetrievalTrace"),
        (answer, "AnswerSession"),
        (package, "ContextPackage"),
        (run, "AgentRun"),
    ):
        _require_kb(linked, knowledge_base_id, field=field)
    if (
        str(package.retrieval_trace_id or "") != str(trace.id)
        or str(answer.retrieval_trace_id or "") != str(trace.id)
        or str(answer.context_package_id or "") != str(package.id)
    ):
        _fail("RewardEvent trace/package/answer reciprocal links are broken")
    if str(run.session_id or "") != str(answer.qa_session_id or ""):
        _fail("AgentRun and AnswerSession QA-session reciprocal links diverge")
    question = str(answer.question or "")
    if (
        question != str(trace.query or "")
        or question != str(package.query or "")
        or question != str(run.question or "")
    ):
        _fail("reward query/question facts diverge")
    if run.final_answer is not None and str(run.final_answer) != str(answer.answer or ""):
        _fail("AgentRun final answer diverges from AnswerSession")
    if context.get("question_length") != len(question):
        _fail("RewardEvent question-length fact is not reciprocal")
    if context.get("context_token_count") != package.token_count:
        _fail("RewardEvent context token count is not reciprocal")
    reward_chunk_ids = _require_unique_ids(
        list(row.chunk_ids_json or []),
        field="reward.chunk_ids",
    )
    answer_chunk_ids = _require_unique_ids(
        list(answer.chunk_ids_json or []),
        field="answer.chunk_ids",
    )
    if (
        reward_chunk_ids != list(package.hit_chunk_ids_json or [])
        or answer_chunk_ids != list(package.hit_chunk_ids_json or [])
        or reward_chunk_ids != list(trace.result_chunk_ids_json or [])
    ):
        _fail("reward/answer/package/trace chunk references diverge")

    concept_path_raw = list(trace.concept_path_json or [])
    path_extra_chunks = {
        str(item)
        for path in concept_path_raw
        if isinstance(path, Mapping) and path.get("layer") == "chunk"
        for item in (path.get("ids") or [])
    }
    for pool in list((trace.candidate_pools_json or {}).get("chunk_by_mid") or []):
        if isinstance(pool, Mapping):
            path_extra_chunks.update(str(item) for item in pool.get("candidate_ids") or [])
            path_extra_chunks.update(str(item) for item in pool.get("selected_ids") or [])
    for label in list(trace.path_labels_json or []):
        if isinstance(label, Mapping) and (
            label.get("layer") == "chunk" or label.get("chunk_id")
        ):
            path_extra_chunks.update(str(item) for item in label.get("path") or [])
            if label.get("chunk_id"):
                path_extra_chunks.add(str(label["chunk_id"]))
    if row.created_at is None:
        _fail("RewardEvent has no durable replay cutoff timestamp")
    persisted_observations = list(
        db.scalars(
            select(AgentObservation).where(
                AgentObservation.run_id == run.id,
                AgentObservation.created_at <= row.created_at,
            )
        ).all()
    )
    for observation in persisted_observations:
        path_extra_chunks.update(
            str(item) for item in observation.evidence_chunk_ids_json or []
        )

    related_package_ids = {str(package.id)}
    context_package_reference_fields = (
        "context_package_id",
        "cache_source_context_package_id",
        "before_context_package_id",
        "repaired_context_package_id",
    )
    for observation in persisted_observations:
        payload = dict(observation.observation_json or {})
        for field in context_package_reference_fields:
            value = str(payload.get(field) or "")
            if value:
                related_package_ids.add(value)
    related_packages = _ordered_rows_by_ids(
        db,
        ContextPackage,
        sorted(related_package_ids),
        field="reward AgentRun context package snapshots",
    )
    related_document_snapshots: dict[str, dict[str, str]] = {}
    for related_package in related_packages:
        _require_kb(
            related_package,
            knowledge_base_id,
            field="reward AgentRun Context Package snapshot",
        )
        if (
            related_package.created_at is None
            or related_package.created_at > row.created_at
        ):
            _fail("reward references a Context Package after its replay cutoff")
        if str(related_package.query or "") != question:
            _fail("reward AgentRun Context Package query diverges")
        for document_id, snapshot in _context_package_document_snapshots(
            related_package
        ).items():
            prior = related_document_snapshots.setdefault(document_id, snapshot)
            if prior != snapshot:
                _fail(
                    "reward AgentRun Context Packages have divergent frozen "
                    "document identities"
                )

    pre_replay_grounding_outcome = str(
        (answer.diagnostics_json or {}).get("grounding_outcome") or ""
    )
    allow_empty_reward_results = pre_replay_grounding_outcome in {
        "insufficient_evidence",
        "rejected_attempt",
    }
    chunk_facts, chunks_by_id, chunk_key_by_id, package_fact_by_id = (
        _build_chunk_facts(
            db,
            knowledge_base_id=knowledge_base_id,
            package=package,
            trace=trace,
            extra_chunk_ids=path_extra_chunks,
            allow_empty_results=allow_empty_reward_results,
            additional_document_snapshots=related_document_snapshots,
            replay_cutoff=row.created_at,
        )
    )
    path_facts, path_refs, path_metrics = _build_path_facts(
        db,
        knowledge_base_id=knowledge_base_id,
        trace=trace,
        chunk_by_id=chunks_by_id,
        chunk_key_by_id=chunk_key_by_id,
        allow_empty_results=allow_empty_reward_results,
    )
    citation_facts, citation_metrics, _, citation_ids = _build_citation_facts(
        db,
        knowledge_base_id=knowledge_base_id,
        answer=answer,
        package=package,
        trace=trace,
        chunks_by_id=chunks_by_id,
        chunk_key_by_id=chunk_key_by_id,
        package_fact_by_id=package_fact_by_id,
    )

    answer_diagnostics = dict(answer.diagnostics_json or {})
    reward_diagnostics = dict(row.diagnostics_json or {})
    evidence_gap = dict(answer_diagnostics.get("evidence_gap") or {})
    _exact_json(
        evidence_gap,
        dict(reward_diagnostics.get("evidence_gap") or {}),
        field="answer/reward evidence_gap",
    )
    uuid_free_evidence_gap = _uuid_free_json(
        evidence_gap,
        field="evidence_gap",
    )
    grounding_outcome = str(answer_diagnostics.get("grounding_outcome") or "")
    if grounding_outcome not in {
        "grounded_answer",
        "insufficient_evidence",
        "rejected_attempt",
    }:
        _fail("Answer Session grounding outcome is not allowlisted")
    if grounding_outcome == "insufficient_evidence":
        from app.services.agent_graph import evidence_insufficient_answer

        if (
            str(evidence_gap.get("kind") or "") != "no_supported_claims"
            or str(answer.answer or "")
            != evidence_insufficient_answer(
                str(answer.question or ""),
                "insufficient_corpus",
            )
        ):
            _fail(
                "insufficient-evidence reward is not the deterministic "
                "no-supported-claims response"
            )
    if grounding_outcome != str(reward_diagnostics.get("grounding_outcome") or ""):
        _fail("answer/reward grounding outcomes diverge")
    answer_acceptance = dict(answer_diagnostics.get("answer_acceptance_gate") or {})
    reward_acceptance = dict(
        reward_diagnostics.get("answer_acceptance_gate") or {}
    )
    _exact_json(
        answer_acceptance,
        reward_acceptance,
        field="answer/reward acceptance gate",
    )
    for key in (
        "accepted",
        "claim_grounding_rejected",
        "prompt_grounding_rejected",
        "exact_prompt_audit_verified",
    ):
        if not isinstance(answer_acceptance.get(key), bool):
            _fail("answer acceptance gate contains a non-boolean fact")
    if answer_acceptance["accepted"] != (
        not answer_acceptance["claim_grounding_rejected"]
        and not answer_acceptance["prompt_grounding_rejected"]
    ):
        _fail("answer acceptance gate is internally inconsistent")
    if answer_acceptance["accepted"] and not answer_acceptance[
        "exact_prompt_audit_verified"
    ]:
        _fail(
            "accepted answer is missing the exact prompt-grounding audit"
        )
    reward_gate = dict(reward_diagnostics.get("claim_grounded_gate") or {})
    _exact_json(
        reward_gate,
        dict(answer_diagnostics.get("claim_grounded_gate") or {}),
        field="answer/reward claim gate",
    )

    runtime_hash = _require_hash(
        trace.runtime_settings_hash,
        field="trace.runtime_settings_hash",
    )
    envelope_hash = _require_hash(
        trace.agent_operating_envelope_hash,
        field="trace.agent_operating_envelope_hash",
    )
    traversal_hash = _require_hash(
        trace.traversal_protocol_hash,
        field="trace.traversal_protocol_hash",
    )
    policy_hash = _require_hash(
        trace.policy_state_hash,
        field="trace.policy_state_hash",
        optional=True,
    )
    input_policy_state_ids: list[str] = []
    envelope = (trace.diagnostics_json or {}).get("agent_operating_envelope")
    if not isinstance(envelope, Mapping):
        _fail("Retrieval Trace is missing its persisted operating envelope")
    _assert_finite_json(envelope, field="trace.agent_operating_envelope")
    if stable_hash(dict(envelope)) != envelope_hash:
        _fail("Retrieval Trace operating envelope hash failed replay")
    if envelope.get("gray_zone_model_call_budget") != 0:
        _fail("Retrieval Trace operating envelope granted gray-zone model calls")

    agent_facts, agent_metrics, agent_refs = _build_agent_facts(
        db,
        knowledge_base_id=knowledge_base_id,
        run=run,
        trace=trace,
        context_package=package,
        claim_grounded_gate=reward_gate,
        answer_text=str(answer.answer or ""),
        operating_envelope=dict(envelope),
        chunk_key_by_id=chunk_key_by_id,
        evidence_gap=evidence_gap,
        grounding_outcome=grounding_outcome,
        reward_created_at=row.created_at,
    )
    cache_fact, cache_refs = _cache_business_fact(
        trace,
        chunk_business_scope_hash=str(chunk_facts["chunk_business_scope_hash"]),
    )

    trace_prior = (trace.diagnostics_json or {}).get(
        "policy_operating_prior"
    )
    if not isinstance(trace_prior, Mapping):
        _fail("Retrieval Trace is missing its frozen Policy operating prior")
    for field, copy_value in (
        (
            "RewardEvent.action_json.policy_operating_prior",
            (row.action_json or {}).get("policy_operating_prior"),
        ),
        (
            "RewardEvent.diagnostics_json.policy_operating_prior",
            reward_diagnostics.get("policy_operating_prior"),
        ),
        (
            "AgentRun.metadata_json.policy_operating_prior",
            (run.metadata_json or {}).get("policy_operating_prior"),
        ),
    ):
        _exact_json(trace_prior, copy_value, field=field)
    try:
        from app.services.policy import replay_policy_operating_prior_card

        frozen_policy_envelope = dict(
            agent_metrics["frozen_operating_envelope"]
        )
        frozen_policy_envelope_hash = str(
            agent_metrics["frozen_operating_envelope_hash"]
        )
        validated_prior, input_policy_state = replay_policy_operating_prior_card(
            db,
            dict(trace_prior),
            knowledge_base_id=knowledge_base_id,
            runtime_settings_hash=str(runtime_hash),
            agent_operating_envelope_hash=frozen_policy_envelope_hash,
            agent_operating_envelope=frozen_policy_envelope,
        )
    except Exception as exc:
        raise PolicyRewardReplayError(
            "Retrieval Trace PolicyState/prior failed strict admission replay"
        ) from exc
    if validated_prior.get("policy_state_hash") != policy_hash:
        _fail("Retrieval Trace policy hash diverges from its frozen prior")
    if input_policy_state is not None:
        input_policy_state_ids = [str(input_policy_state.id)]
    prior_content = dict(validated_prior)
    prior_content.pop("knowledge_base_id", None)
    prior_content.pop("policy_state_id", None)
    prior_content.pop("prior_hash", None)
    prior_content = _address_free_json(
        prior_content,
        field="policy operating-prior content",
    )

    step_rows = list(
        db.scalars(
            select(GraphRetrievalStep)
            .where(GraphRetrievalStep.retrieval_trace_id == trace.id)
            .order_by(
                GraphRetrievalStep.step_index.asc(),
                GraphRetrievalStep.id.asc(),
            )
        ).all()
    )
    gray_records: list[Any] = []
    gray_step_refs: list[dict[str, Any]] = []
    for step in step_rows:
        _require_kb(step, knowledge_base_id, field="GraphRetrievalStep")
        step_records = list(step.gray_zone_path_decisions_json or [])
        gray_records.extend(step_records)
        gray_step_refs.append(
            {
                "step_id": str(step.id),
                "step_index": int(step.step_index),
                "gray_decision_hashes": [
                    str(record.get("decision_hash") or "")
                    for record in step_records
                    if isinstance(record, Mapping)
                ],
            }
        )
    # RetrievalTrace has no mapped gray-record column.  Reject any transient
    # shadow copy too, because an in-session mutation must not evade replay.
    shadow_gray_records = list(
        getattr(trace, "gray_zone_path_decisions_json", None) or []
    )
    gray_records.extend(shadow_gray_records)
    gray_runtime_hash = str(
        (trace.diagnostics_json or {}).get(
            "gray_zone_runtime_settings_hash"
        )
        or ""
    )
    gray_runtime_protocol = str(
        (trace.diagnostics_json or {}).get(
            "gray_zone_runtime_settings_identity_protocol_version"
        )
        or ""
    )
    try:
        from app.services.context_graph import (
            GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION,
            validate_gray_zone_decision_records_for_persistence,
            validate_gray_zone_trace_aggregate_for_persistence,
        )

        if (
            gray_runtime_protocol
            != GRAY_ZONE_RUNTIME_SETTINGS_IDENTITY_PROTOCOL_VERSION
        ):
            raise ValueError(
                "Retrieval Trace gray-zone runtime identity protocol diverges"
            )

        validated_gray_records = (
            validate_gray_zone_decision_records_for_persistence(
                gray_records,
                traversal_hash=str(traversal_hash),
                runtime_settings_hash=gray_runtime_hash,
                operating_envelope_hash=str(envelope_hash),
                operating_envelope=dict(envelope),
            )
        )
        validate_gray_zone_trace_aggregate_for_persistence(
            validated_gray_records,
            convergence=dict(trace.convergence_json or {}),
            operating_envelope=dict(envelope),
        )
    except Exception as exc:
        raise PolicyRewardReplayError(
            "Retrieval Trace gray-zone records failed deterministic replay"
        ) from exc
    gray_content_facts = [
        _gray_decision_content_fact(record)
        for record in validated_gray_records
    ]
    convergence = dict(trace.convergence_json or {})
    if convergence.get("gray_zone_model_call_count") != 0:
        _fail("Retrieval Trace gray-zone model call count is not zero")
    for key, expected in (
        ("runtime_settings_hash", runtime_hash),
        ("agent_operating_envelope_hash", envelope_hash),
        ("traversal_protocol_hash", traversal_hash),
    ):
        if convergence.get(key) != expected:
            _fail(f"Retrieval Trace gray convergence {key} diverges")
    for key, expected in (
        ("runtime_settings_hash", runtime_hash),
        ("agent_operating_envelope_hash", envelope_hash),
        ("effective_traversal_protocol_hash", traversal_hash),
        ("policy_state_hash", policy_hash),
    ):
        diagnostic_value = (trace.diagnostics_json or {}).get(key)
        if key == "effective_traversal_protocol_hash":
            if diagnostic_value != expected:
                _fail("Retrieval Trace traversal hash copies diverge")
        elif diagnostic_value is not None and diagnostic_value != expected:
            _fail(f"Retrieval Trace {key} copies diverge")
    if package.runtime_settings_hash != runtime_hash:
        _fail("Context Package runtime settings hash diverges from trace")

    required_facets = sorted(
        set(
            str(item)
            for item in (trace.query_facets_json or {}).get("required_facets") or []
            if str(item)
        )
    )
    covered_facets = list(chunk_facts["covered_facets"])
    drift_rate = (
        1.0
        - len(set(required_facets).intersection(covered_facets))
        / len(required_facets)
        if required_facets
        else 0.0
    )
    package_business_keys = {
        str(item["chunk_business_key"]) for item in chunk_facts["package_chunks"]
    }
    supported_business_keys = {
        chunk_key_by_id[row_id]
        for row_id in citation_metrics["supported_chunk_ids"]
    }
    hit_business_keys = set(chunk_facts["hit_chunk_business_keys"])
    context_precision = (
        len(supported_business_keys.intersection(package_business_keys))
        / len(package_business_keys)
        if package_business_keys
        else 0.0
    )
    context_recall = (
        len(hit_business_keys.intersection(package_business_keys))
        / len(hit_business_keys)
        if hit_business_keys
        else 0.0
    )
    claim_count = int(citation_metrics["claim_count"])
    supported_claim_count = int(citation_metrics["supported_claim_count"])
    original_claim_count = evidence_gap.get("original_claim_count", claim_count)
    original_supported_count = evidence_gap.get(
        "original_supported_claim_count",
        supported_claim_count,
    )
    original_unsupported_count = evidence_gap.get(
        "original_unsupported_claim_count",
        int(citation_metrics["unsupported_claim_count"]),
    )
    original_claim_count = _bounded_int(
        original_claim_count,
        field="evidence_gap.original_claim_count",
        maximum=MAX_REWARD_CLAIMS,
    )
    original_supported_count = _bounded_int(
        original_supported_count,
        field="evidence_gap.original_supported_claim_count",
        maximum=MAX_REWARD_CLAIMS,
    )
    original_unsupported_count = _bounded_int(
        original_unsupported_count,
        field="evidence_gap.original_unsupported_claim_count",
        maximum=MAX_REWARD_CLAIMS,
    )
    if (
        original_supported_count + original_unsupported_count
        != original_claim_count
    ):
        _fail("evidence_gap original claim counts are inconsistent")
    if (
        grounding_outcome != "insufficient_evidence"
        and original_claim_count < claim_count
    ):
        _fail("evidence_gap original claim counts are inconsistent")
    metric_original_claim_count = (
        max(original_claim_count, claim_count)
        if grounding_outcome == "insufficient_evidence"
        else original_claim_count
    )
    original_completeness = min(
        float(citation_metrics["claim_pass_rate"]),
        original_supported_count / max(metric_original_claim_count, 1),
    )
    accepted = bool(answer_acceptance["accepted"])
    repair_success = agent_metrics["repair_success_rate"]
    if repair_success is None:
        repair_success = original_completeness
    metrics = {
        "reward_metrics_protocol_version": POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        "retrieval_hit": 1.0 if hit_business_keys else 0.0,
        "context_precision": round(context_precision, 6),
        "context_recall": round(context_recall, 6),
        "concept_path_accuracy": path_metrics["concept_path_accuracy"],
        "citation_pass_rate": round(
            float(citation_metrics["claim_pass_rate"]) if accepted else 0.0,
            6,
        ),
        "answer_groundedness": round(
            float(citation_metrics["claim_pass_rate"]) if accepted else 0.0,
            6,
        ),
        "answer_completeness": round(
            original_completeness if accepted else 0.0,
            6,
        ),
        "claim_count": claim_count,
        "supported_claim_count": supported_claim_count,
        "unsupported_claim_count": int(citation_metrics["unsupported_claim_count"]),
        "repair_success_rate": round(float(repair_success) if accepted else 0.0, 6),
        "agent_typed_action_validation_pass_rate": agent_metrics[
            "typed_action_validation_pass_rate"
        ],
        "latency_cost": round(float(agent_metrics["latency_ms"]) / 1000.0, 6),
        "latency_ms": int(agent_metrics["latency_ms"]),
        "task_token_cost": int(package.token_count)
        + int(agent_metrics["non_context_task_token_count"]),
        "drift_rate": round(drift_rate, 6),
        "answer_acceptance_gate_pass": 1.0 if accepted else 0.0,
    }
    for field in (
        "retrieval_hit",
        "context_precision",
        "context_recall",
        "concept_path_accuracy",
        "citation_pass_rate",
        "answer_groundedness",
        "answer_completeness",
        "repair_success_rate",
        "agent_typed_action_validation_pass_rate",
        "drift_rate",
        "answer_acceptance_gate_pass",
    ):
        _finite_float(metrics[field], field=f"metrics.{field}", minimum=0.0, maximum=1.0)
    _finite_float(
        metrics["latency_ms"],
        field="metrics.latency_ms",
        minimum=0.0,
        maximum=float(MAX_TOTAL_EVENT_DURATION_MS),
    )
    _finite_float(
        metrics["task_token_cost"],
        field="metrics.task_token_cost",
        minimum=0.0,
        maximum=float(MAX_REWARD_TOKENS),
    )

    content_card = {
        "protocol_version": POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        "query": {
            "query_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "filters_hash": canonical_graph_hash(
                "policy_reward_filters_v1",
                dict(trace.filters_json or {}),
            ),
            "required_facets": required_facets,
            "covered_facets": covered_facets,
        },
        "trace_identity": {
            "retrieval_mode": str(trace.retrieval_mode or ""),
            "runtime_settings_hash": runtime_hash,
            "policy_state_hash": policy_hash,
            "agent_operating_envelope_hash": envelope_hash,
            "traversal_protocol_hash": traversal_hash,
            "gray_zone_model_call_count": 0,
            "policy_operating_prior": prior_content,
            "gray_zone": {
                "decision_count": len(gray_content_facts),
                "decisions": gray_content_facts,
                "rule_protocol_version": convergence.get(
                    "gray_zone_rule_protocol_version"
                ),
                "observation_cadence": convergence.get(
                    "gray_zone_observation_cadence"
                ),
                "model_call_count": 0,
            },
            "edge_distance_protocol_hash": _require_hash(
                trace.edge_distance_protocol_hash,
                field="trace.edge_distance_protocol_hash",
            ),
            "edge_projection_protocol_hash": _require_hash(
                trace.edge_projection_protocol_hash,
                field="trace.edge_projection_protocol_hash",
            ),
            "cache_business_identity": cache_fact,
        },
        "context_package": chunk_facts,
        "graph_path": path_facts,
        "answer_and_citations": citation_facts,
        "agent": agent_facts,
        "evidence_gap": uuid_free_evidence_gap,
        "grounding_outcome": grounding_outcome,
        "answer_acceptance_gate": _uuid_free_json(
            answer_acceptance,
            field="answer_acceptance_gate",
        ),
    }
    _assert_finite_json(content_card, field="policy_reward.content_card")
    evidence_hash = canonical_graph_hash(
        POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        content_card,
    )
    metrics["reward_metric_evidence_hash"] = evidence_hash
    reward_fact = {
        "protocol_version": POLICY_REWARD_FACT_PROTOCOL_VERSION,
        "evidence_protocol_version": POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        "reward_metric_evidence_hash": evidence_hash,
        "metrics": metrics,
        "policy_inputs": {
            "runtime_settings_hash": runtime_hash,
            "policy_state_hash": policy_hash,
            "agent_operating_envelope_hash": envelope_hash,
            "traversal_protocol_hash": traversal_hash,
        },
        "answer_acceptance_gate_pass": metrics[
            "answer_acceptance_gate_pass"
        ],
    }
    reward_fact_hash = canonical_graph_hash(
        POLICY_REWARD_FACT_PROTOCOL_VERSION,
        reward_fact,
    )

    policy_state_ref = None
    if row.policy_state_id is not None:
        policy_state = _load_row(
            db,
            PolicyState,
            str(row.policy_state_id),
            field="reward.policy_state_id",
        )
        _require_kb(policy_state, knowledge_base_id, field="reward PolicyState")
        if str(
            (policy_state.reward_summary_json or {}).get("last_reward_event_id")
            or ""
        ) != str(row.id):
            _fail("reward PolicyState does not link back to this RewardEvent")
        policy_state_ref = str(policy_state.id)
    refs = {
        "knowledge_base_id": knowledge_base_id,
        "reward_event_id": str(row.id),
        "retrieval_trace_id": str(trace.id),
        "context_package_id": str(package.id),
        "answer_session_id": str(answer.id),
        "policy_state_id": policy_state_ref,
        "input_policy_state_ids": input_policy_state_ids,
        "policy_operating_prior": dict(validated_prior),
        "gray_zone_decision_records": validated_gray_records,
        "gray_zone_step_refs": gray_step_refs,
        "citation_verification_ids": citation_ids,
        "chunk_ids_by_business_key": sorted(
            (
                {"chunk_business_key": business_key, "chunk_id": chunk_id}
                for chunk_id, business_key in chunk_key_by_id.items()
            ),
            key=lambda item: (item["chunk_business_key"], item["chunk_id"]),
        ),
        "graph_path": path_refs,
        **agent_refs,
        **cache_refs,
    }
    return {
        "storage_protocol_version": POLICY_REWARD_STORAGE_PROTOCOL_VERSION,
        "protocol_version": POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        "refs": refs,
        "content_card": content_card,
        "evidence_hash": evidence_hash,
        "metrics": metrics,
        "reward_fact": reward_fact,
        "reward_fact_hash": reward_fact_hash,
    }


def freeze_policy_reward_replay(
    reward: RewardEvent,
    replay: Mapping[str, Any],
) -> None:
    """Attach a freshly rebuilt replay packet to an uncommitted RewardEvent."""

    if str((replay.get("refs") or {}).get("reward_event_id") or "") != str(
        reward.id or ""
    ):
        _fail("reward replay packet belongs to a different RewardEvent")
    metrics = dict(replay.get("metrics") or {})
    reward_json = dict(reward.reward_json or {})
    reward_json.update(metrics)
    reward_json.update(
        {
            "reward_metrics_protocol_version": (
                POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION
            ),
            "reward_metric_evidence_hash": replay.get("evidence_hash"),
            "policy_reward_metrics": metrics,
            "policy_reward_fact_protocol_version": (
                POLICY_REWARD_FACT_PROTOCOL_VERSION
            ),
            "policy_reward_fact_hash": replay.get("reward_fact_hash"),
        }
    )
    diagnostics = dict(reward.diagnostics_json or {})
    diagnostics.update(
        {
            "policy_reward_storage_protocol_version": (
                POLICY_REWARD_STORAGE_PROTOCOL_VERSION
            ),
            "policy_reward_metric_evidence_protocol_version": (
                POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION
            ),
            "policy_reward_metric_evidence_refs": dict(
                replay.get("refs") or {}
            ),
            "policy_reward_metric_evidence_card": dict(
                replay.get("content_card") or {}
            ),
            "policy_reward_metric_evidence_hash": replay.get("evidence_hash"),
            "policy_reward_fact_protocol_version": (
                POLICY_REWARD_FACT_PROTOCOL_VERSION
            ),
            "policy_reward_fact": dict(replay.get("reward_fact") or {}),
            "policy_reward_fact_hash": replay.get("reward_fact_hash"),
            # Keep the historical top-level aliases exact for readers that
            # have not yet switched to the v3 nested card.
            "reward_metrics_protocol_version": (
                POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION
            ),
            "reward_metric_evidence_hash": replay.get("evidence_hash"),
        }
    )
    reward.reward_json = reward_json
    reward.diagnostics_json = diagnostics


def replay_policy_reward_event(
    db: Session,
    reward: RewardEvent | str,
) -> dict[str, Any]:
    """Rebuild and exactly compare the stored v3 evidence, metrics and fact."""

    row = (
        _load_row(db, RewardEvent, reward, field="reward_event")
        if isinstance(reward, str)
        else reward
    )
    rebuilt = build_policy_reward_replay(db, row)
    reward_json = dict(row.reward_json or {})
    diagnostics = dict(row.diagnostics_json or {})
    if diagnostics.get("policy_reward_storage_protocol_version") != (
        POLICY_REWARD_STORAGE_PROTOCOL_VERSION
    ):
        _fail("RewardEvent has no supported persisted reward replay packet")
    if diagnostics.get(
        "policy_reward_metric_evidence_protocol_version"
    ) != POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION:
        _fail("RewardEvent evidence protocol is missing or stale")
    if diagnostics.get("policy_reward_fact_protocol_version") != (
        POLICY_REWARD_FACT_PROTOCOL_VERSION
    ):
        _fail("RewardEvent fact protocol is missing or stale")

    stored_card = diagnostics.get("policy_reward_metric_evidence_card")
    stored_refs = diagnostics.get("policy_reward_metric_evidence_refs")
    stored_evidence_hash = _require_hash(
        diagnostics.get("policy_reward_metric_evidence_hash"),
        field="stored policy reward evidence hash",
    )
    if canonical_graph_hash(
        POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION,
        stored_card,
    ) != stored_evidence_hash:
        _fail("stored policy reward evidence card has a forged hash")
    _exact_json(
        stored_card,
        rebuilt["content_card"],
        field="stored/rebuilt policy reward evidence card",
    )
    _exact_json(
        stored_refs,
        rebuilt["refs"],
        field="stored/rebuilt policy reward refs",
    )
    if stored_evidence_hash != rebuilt["evidence_hash"]:
        _fail("stored policy reward evidence hash does not match replay")

    stored_metrics = reward_json.get("policy_reward_metrics")
    _exact_json(
        stored_metrics,
        rebuilt["metrics"],
        field="stored/rebuilt policy reward metrics",
    )
    for field in _METRIC_FIELDS:
        _exact_json(
            reward_json.get(field),
            rebuilt["metrics"].get(field),
            field=f"reward metric {field}",
        )
    if (
        reward_json.get("reward_metrics_protocol_version")
        != POLICY_REWARD_EVIDENCE_PROTOCOL_VERSION
        or reward_json.get("reward_metric_evidence_hash")
        != rebuilt["evidence_hash"]
    ):
        _fail("RewardEvent top-level metric identity does not match replay")

    stored_fact = diagnostics.get("policy_reward_fact")
    stored_fact_hash = _require_hash(
        diagnostics.get("policy_reward_fact_hash"),
        field="stored policy reward fact hash",
    )
    if canonical_graph_hash(
        POLICY_REWARD_FACT_PROTOCOL_VERSION,
        stored_fact,
    ) != stored_fact_hash:
        _fail("stored policy reward fact has a forged hash")
    _exact_json(
        stored_fact,
        rebuilt["reward_fact"],
        field="stored/rebuilt policy reward fact",
    )
    if stored_fact_hash != rebuilt["reward_fact_hash"]:
        _fail("stored policy reward fact hash does not match replay")
    if (
        reward_json.get("policy_reward_fact_hash") != stored_fact_hash
        or reward_json.get("policy_reward_fact_protocol_version")
        != POLICY_REWARD_FACT_PROTOCOL_VERSION
    ):
        _fail("RewardEvent top-level policy reward fact identity diverges")
    return rebuilt
