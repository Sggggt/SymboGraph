from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from typing import Any

from pydantic import ValidationError
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.sensitive_fields import (
    SENSITIVE_FIELD_KEY_PROTOCOL_VERSION,
    semantic_key_segments,
    sensitive_field_key_reason,
)
from app.models import (
    AgentAction,
    AgentObservation,
    AgentPlan,
    AgentRun,
    ContextPackage,
    RetrievalTrace,
)
from app.schemas import AgentPEAuditResponse


AGENT_PE_AUDIT_PROTOCOL_VERSION = "agent_pe_audit_public_v1"
AGENT_PE_SENSITIVE_KEY_PROTOCOL_VERSION = SENSITIVE_FIELD_KEY_PROTOCOL_VERSION
PLAN_ORDER = "plan_index ASC, created_at ASC, id ASC"
ACTION_ORDER = (
    "plan_index ASC NULLS LAST, action_index ASC, created_at ASC, id ASC"
)
OBSERVATION_ORDER = "created_at ASC, id ASC"
REPAIR_ACTION_TYPES = {
    "repair_missing_citation",
    "repair_concept_gap",
    "repair_bridge_gap",
    "repair_structure_context",
}
_REDACTED = "[REDACTED]"
_DANGEROUS_STORAGE_SEGMENTS = frozenset(
    {"archive", "backup", "blob", "bundle", "copy", "snapshot", "value"}
)


class AgentPEAuditIntegrityError(RuntimeError):
    """The persisted P&E rows cannot be exposed as one coherent run audit."""


def _sensitive_key_kind(value: object) -> str | None:
    """Bind P&E redaction to the repository-wide semantic-key protocol.

    The small P&E adapter only assigns the two public exposure categories and
    retains the pre-existing raw-output/header container semantics.  All key
    normalization, compound credential recognition, safe token accounting,
    NFKC handling, and dangerous-storage classification come from
    ``app.core.sensitive_fields``.
    """

    segments = semantic_key_segments(value)
    reason = sensitive_field_key_reason(
        value,
        include_public_private=True,
    )
    if reason == "provider_response":
        return "provider_raw_response"
    if reason is not None:
        return "credentials"

    dangerous_storage = bool(
        _DANGEROUS_STORAGE_SEGMENTS.intersection(segments)
    )
    for left, right in (("raw", "output"), ("raw", "response")):
        for index in range(0, len(segments) - 1):
            if segments[index : index + 2] != (left, right):
                continue
            if index + 2 == len(segments) or dangerous_storage:
                return "provider_raw_response"

    header_indexes = [
        index
        for index, segment in enumerate(segments)
        if segment in {"header", "headers"}
    ]
    if header_indexes and (
        header_indexes[-1] == len(segments) - 1 or dangerous_storage
    ):
        return "credentials"
    return None


def _sensitive_key(value: object) -> bool:
    return _sensitive_key_kind(value) is not None


def _scan_sensitive_exposure(value: Any) -> tuple[bool, bool]:
    provider_exposed = False
    credentials_exposed = False
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            kind = _sensitive_key_kind(raw_key)
            if kind is not None and raw_value != _REDACTED:
                if kind == "provider_raw_response":
                    provider_exposed = True
                else:
                    credentials_exposed = True
                continue
            child_provider, child_credentials = _scan_sensitive_exposure(
                raw_value
            )
            provider_exposed = provider_exposed or child_provider
            credentials_exposed = (
                credentials_exposed or child_credentials
            )
    elif isinstance(value, list):
        for item in value:
            child_provider, child_credentials = _scan_sensitive_exposure(item)
            provider_exposed = provider_exposed or child_provider
            credentials_exposed = (
                credentials_exposed or child_credentials
            )
    return provider_exposed, credentials_exposed


def _sanitize_json(value: Any, *, path: str) -> tuple[Any, list[str]]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        redacted: list[str] = []
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if _sensitive_key(key):
                result[key] = _REDACTED
                redacted.append(child_path)
                continue
            child, child_redacted = _sanitize_json(raw_value, path=child_path)
            result[key] = child
            redacted.extend(child_redacted)
        return result, redacted
    if isinstance(value, list):
        result_list: list[Any] = []
        redacted: list[str] = []
        for index, item in enumerate(value):
            child, child_redacted = _sanitize_json(
                item,
                path=f"{path}[{index}]",
            )
            result_list.append(child)
            redacted.extend(child_redacted)
        return result_list, redacted
    return value, []


def _json_payload(value: Any, *, path: str) -> tuple[dict[str, Any], list[str]]:
    sanitized, redacted = _sanitize_json(value, path=path)
    try:
        encoded = json.dumps(
            sanitized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {path} is not finite canonical JSON"
        ) from exc
    return {
        "encoding": "canonical_json_v1",
        "canonical_json": encoded,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "redacted_fields": sorted(set(redacted)),
    }, redacted


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {field} must be a JSON object"
        )
    return dict(value)


def _object_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {field} must be an array of JSON objects"
        )
    return [dict(item) for item in value]


def _string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {field} must be an array of strings"
        )
    return list(value)


def _value(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _consistent_lineage_value(
    *candidates: Any,
    field: str,
) -> str | None:
    values: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, str) or not candidate:
            raise AgentPEAuditIntegrityError(
                f"Persisted P&E {field} hash lineage must contain non-empty strings"
            )
        values.append(candidate)
    if any(value != values[0] for value in values[1:]):
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E {field} hash lineage conflicts"
        )
    return values[0] if values else None


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {field} must be a non-negative integer"
        )
    return value


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {field} must be a non-empty string"
        )
    return value


def _sha256_string(value: Any, *, field: str) -> str:
    candidate = _required_string(value, field=field)
    if re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
        raise AgentPEAuditIntegrityError(
            f"Persisted P&E field {field} must be lowercase SHA-256"
        )
    return candidate


def _scan_canonical_payload_exposure(value: Any) -> tuple[bool, bool]:
    provider_exposed = False
    credentials_exposed = False
    if isinstance(value, dict):
        if (
            value.get("encoding") == "canonical_json_v1"
            and isinstance(value.get("canonical_json"), str)
        ):
            try:
                decoded = json.loads(value["canonical_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                return True, True
            return _scan_sensitive_exposure(decoded)
        for child in value.values():
            child_provider, child_credentials = (
                _scan_canonical_payload_exposure(child)
            )
            provider_exposed = provider_exposed or child_provider
            credentials_exposed = (
                credentials_exposed or child_credentials
            )
    elif isinstance(value, list):
        for child in value:
            child_provider, child_credentials = (
                _scan_canonical_payload_exposure(child)
            )
            provider_exposed = provider_exposed or child_provider
            credentials_exposed = (
                credentials_exposed or child_credentials
            )
    return provider_exposed, credentials_exposed


def _validator_payload(
    raw_validation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    sanitized, redacted = _sanitize_json(
        raw_validation,
        path="validation",
    )
    payload, _ = _json_payload(raw_validation, path="validation")
    return {
        "valid": sanitized.get("valid"),
        "plan_valid": sanitized.get("plan_valid"),
        "schema_checked": sanitized.get("schema_checked"),
        "budget_checked": sanitized.get("budget_checked"),
        "target_ids_checked": sanitized.get("target_ids_checked"),
        "target_scope_checked": sanitized.get("target_scope_checked"),
        "typed_action_schema_protocol_version": sanitized.get(
            "typed_action_schema_protocol_version"
        ),
        "typed_action_schema_protocol_hash": sanitized.get(
            "typed_action_schema_protocol_hash"
        ),
        "repair_protocol_version": sanitized.get("repair_protocol_version"),
        "repair_budget_checked": sanitized.get("repair_budget_checked"),
        "repair_round_index": sanitized.get("repair_round_index"),
        "remaining_repair_budget_before": sanitized.get(
            "remaining_repair_budget_before"
        ),
        "action_input_hash": sanitized.get("action_input_hash"),
        "repair_directive_validator_protocol_version": sanitized.get(
            "repair_directive_validator_protocol_version"
        ),
        "repair_directive_validator_result": sanitized.get(
            "repair_directive_validator_result"
        ),
        "repair_directive_hash": sanitized.get("repair_directive_hash"),
        "validated_directive_hash": sanitized.get("validated_directive_hash"),
        "payload": payload,
    }, redacted


def _validate_linkage(
    *,
    db: Session,
    run: AgentRun,
    plans: list[AgentPlan],
    actions: list[AgentAction],
    observations: list[AgentObservation],
) -> tuple[dict[str, AgentPlan], dict[str, AgentAction]]:
    plan_by_id = {str(row.id): row for row in plans}
    action_by_id = {str(row.id): row for row in actions}
    if len(plan_by_id) != len(plans) or len(action_by_id) != len(actions):
        raise AgentPEAuditIntegrityError("Duplicate P&E row identity detected")

    plan_indexes: set[int] = set()
    action_indexes_by_plan: dict[str, set[int]] = defaultdict(set)
    for plan in plans:
        if str(plan.run_id) != str(run.id):
            raise AgentPEAuditIntegrityError("Plan belongs to a different run")
        if str(plan.knowledge_base_id) != str(run.knowledge_base_id):
            raise AgentPEAuditIntegrityError(
                "Plan knowledge-base linkage does not match its run"
            )
        plan_index = _integer(plan.plan_index, field="plan.plan_index")
        if plan_index in plan_indexes:
            raise AgentPEAuditIntegrityError(
                "Duplicate plan_index detected within the run"
            )
        plan_indexes.add(plan_index)
    if [int(plan.plan_index) for plan in plans] != list(range(len(plans))):
        raise AgentPEAuditIntegrityError(
            "plan_index must be the complete canonical sequence 0..N-1"
        )

    for action in actions:
        if str(action.run_id) != str(run.id):
            raise AgentPEAuditIntegrityError(
                "Action belongs to a different run"
            )
        plan_id = str(action.plan_id or "")
        if not plan_id or plan_id not in plan_by_id:
            raise AgentPEAuditIntegrityError(
                "Action plan linkage is missing or crosses run boundaries"
            )
        action_index = _integer(
            action.action_index,
            field="action.action_index",
        )
        if action_index in action_indexes_by_plan[plan_id]:
            raise AgentPEAuditIntegrityError(
                "Duplicate action_index detected within a plan"
            )
        action_indexes_by_plan[plan_id].add(action_index)
        if action.parent_action_id is not None:
            parent = action_by_id.get(str(action.parent_action_id))
            if (
                parent is None
                or str(parent.plan_id or "") != plan_id
                or int(parent.action_index) >= action_index
            ):
                raise AgentPEAuditIntegrityError(
                    "Action parent linkage is missing, crosses a run/plan, or is not prior"
                )
    for plan_id, indexes in action_indexes_by_plan.items():
        if sorted(indexes) != list(range(len(indexes))):
            raise AgentPEAuditIntegrityError(
                "action_index must be the complete per-plan sequence 0..N-1"
            )

    for observation in observations:
        if str(observation.run_id) != str(run.id):
            raise AgentPEAuditIntegrityError(
                "Observation belongs to a different run"
            )
        payload = _object(
            observation.observation_json,
            field="observation.observation_json",
        )
        action = (
            action_by_id.get(str(observation.action_id))
            if observation.action_id is not None
            else None
        )
        if observation.action_id is not None and action is None:
            raise AgentPEAuditIntegrityError(
                "Observation action linkage is missing or crosses run boundaries"
            )
        if action is None:
            if observation.observation_type != "evidence_evaluator":
                raise AgentPEAuditIntegrityError(
                    "Only evidence-evaluator observations may omit action_id"
                )
            plan_id = str(payload.get("plan_id") or "")
            if plan_id not in plan_by_id:
                raise AgentPEAuditIntegrityError(
                    "Evaluator observation plan linkage is missing or crosses runs"
                )
        else:
            plan_id = str(action.plan_id)
            supplied_plan_id = payload.get("plan_id")
            if supplied_plan_id is not None and str(supplied_plan_id) != plan_id:
                raise AgentPEAuditIntegrityError(
                    "Observation payload plan linkage conflicts with its action"
                )
        supplied_plan_index = payload.get("plan_index")
        if supplied_plan_index is not None and (
            isinstance(supplied_plan_index, bool)
            or not isinstance(supplied_plan_index, int)
            or supplied_plan_index != int(plan_by_id[plan_id].plan_index)
        ):
            raise AgentPEAuditIntegrityError(
                "Observation payload plan_index conflicts with its canonical plan"
            )
    _validate_repair_replay(
        db=db,
        run=run,
        actions=actions,
        observations=observations,
        action_by_id=action_by_id,
    )
    return plan_by_id, action_by_id


def _validate_repair_replay(
    *,
    db: Session,
    run: AgentRun,
    actions: list[AgentAction],
    observations: list[AgentObservation],
    action_by_id: dict[str, AgentAction],
) -> None:
    typed_repair_by_action: dict[str, list[AgentObservation]] = defaultdict(
        list
    )
    for observation in observations:
        if observation.observation_type != "typed_repair_round":
            continue
        if observation.action_id is None:
            raise AgentPEAuditIntegrityError(
                "Typed repair observation must reference its repair action"
            )
        action_id = str(observation.action_id)
        typed_repair_by_action[action_id].append(observation)

    repair_rounds: list[tuple[int, int, int, int]] = []
    action_order = {
        str(action.id): order_index
        for order_index, action in enumerate(actions)
    }
    for action_id, repair_observations in typed_repair_by_action.items():
        if len(repair_observations) != 1:
            raise AgentPEAuditIntegrityError(
                "Executed repair action must have exactly one typed repair observation"
            )
        action = action_by_id.get(action_id)
        if action is None or str(action.action_type) not in REPAIR_ACTION_TYPES:
            raise AgentPEAuditIntegrityError(
                "Typed repair observation is not linked to a repair action"
            )
        observation = repair_observations[0]
        validation = _object(
            action.validation_json,
            field="repair_action.validation_json",
        )
        action_output = _object(
            action.output_json,
            field="repair_action.output_json",
        )
        observation_payload = _object(
            observation.observation_json,
            field="repair_observation.observation_json",
        )
        if action_output != observation_payload:
            raise AgentPEAuditIntegrityError(
                "Repair action output and observation payload differ"
            )

        action_type = _required_string(
            observation_payload.get("action_type"),
            field="repair.action_type",
        )
        if action_type != str(action.action_type):
            raise AgentPEAuditIntegrityError(
                "Repair action type conflicts with its observation"
            )
        protocol_version = _required_string(
            observation_payload.get("protocol_version"),
            field="repair.protocol_version",
        )
        if (
            _required_string(
                validation.get("repair_protocol_version"),
                field="repair.validation.repair_protocol_version",
            )
            != protocol_version
        ):
            raise AgentPEAuditIntegrityError(
                "Repair protocol conflicts with its validated action"
            )

        repair_round_index = _integer(
            observation_payload.get("repair_round_index"),
            field="repair.repair_round_index",
        )
        if (
            _integer(
                validation.get("repair_round_index"),
                field="repair.validation.repair_round_index",
            )
            != repair_round_index
        ):
            raise AgentPEAuditIntegrityError(
                "Repair round conflicts with its validated action"
            )
        remaining_before = _integer(
            observation_payload.get("remaining_repair_budget_before"),
            field="repair.remaining_repair_budget_before",
        )
        if remaining_before < 1:
            raise AgentPEAuditIntegrityError(
                "Repair budget before execution must be positive"
            )
        if (
            _integer(
                validation.get("remaining_repair_budget_before"),
                field=(
                    "repair.validation.remaining_repair_budget_before"
                ),
            )
            != remaining_before
        ):
            raise AgentPEAuditIntegrityError(
                "Repair budget conflicts with its validated action"
            )
        remaining_after = _integer(
            observation_payload.get("remaining_repair_budget_after"),
            field="repair.remaining_repair_budget_after",
        )
        if remaining_after != remaining_before - 1:
            raise AgentPEAuditIntegrityError(
                "Repair round must consume exactly one hard-budget unit"
            )

        validation_input_hash = _sha256_string(
            validation.get("action_input_hash"),
            field="repair.validation.action_input_hash",
        )
        action_input_hash = _sha256_string(
            action_output.get("action_input_hash"),
            field="repair.action_output.action_input_hash",
        )
        observation_input_hash = _sha256_string(
            observation_payload.get("action_input_hash"),
            field="repair.observation.action_input_hash",
        )
        if not (
            validation_input_hash
            == action_input_hash
            == observation_input_hash
        ):
            raise AgentPEAuditIntegrityError(
                "Repair input hash conflicts across action and observation"
            )
        action_output_hash = _sha256_string(
            action_output.get("action_output_hash"),
            field="repair.action_output.action_output_hash",
        )
        observation_output_hash = _sha256_string(
            observation_payload.get("action_output_hash"),
            field="repair.observation.action_output_hash",
        )
        if action_output_hash != observation_output_hash:
            raise AgentPEAuditIntegrityError(
                "Repair output hash conflicts across action and observation"
            )

        for field in (
            "before_context_package_id",
            "repaired_context_package_id",
        ):
            _required_string(
                observation_payload.get(field),
                field=f"repair.{field}",
            )
        for field in (
            "before_retrieval_trace_id",
            "repaired_retrieval_trace_id",
        ):
            if field not in observation_payload:
                raise AgentPEAuditIntegrityError(
                    f"Persisted P&E field repair.{field} is required"
                )
            value = observation_payload[field]
            if value is not None and (
                not isinstance(value, str) or not value
            ):
                raise AgentPEAuditIntegrityError(
                    f"Persisted P&E field repair.{field} must be null or a non-empty string"
                )
        for package_field, trace_field in (
            (
                "before_context_package_id",
                "before_retrieval_trace_id",
            ),
            (
                "repaired_context_package_id",
                "repaired_retrieval_trace_id",
            ),
        ):
            package_id = str(observation_payload[package_field])
            package = db.get(ContextPackage, package_id)
            if package is None:
                raise AgentPEAuditIntegrityError(
                    f"Repair package linkage {package_field} does not exist"
                )
            if str(package.knowledge_base_id) != str(
                run.knowledge_base_id
            ):
                raise AgentPEAuditIntegrityError(
                    f"Repair package linkage {package_field} crosses knowledge-base scope"
                )
            declared_trace_id = observation_payload[trace_field]
            package_trace_id = (
                str(package.retrieval_trace_id)
                if package.retrieval_trace_id is not None
                else None
            )
            if declared_trace_id != package_trace_id:
                raise AgentPEAuditIntegrityError(
                    f"Repair package/trace linkage {package_field}->{trace_field} conflicts"
                )
            if declared_trace_id is not None:
                trace = db.get(RetrievalTrace, declared_trace_id)
                if trace is None:
                    raise AgentPEAuditIntegrityError(
                        f"Repair trace linkage {trace_field} does not exist"
                    )
                if str(trace.knowledge_base_id) != str(
                    run.knowledge_base_id
                ):
                    raise AgentPEAuditIntegrityError(
                        f"Repair trace linkage {trace_field} crosses knowledge-base scope"
                    )
        repair_rounds.append(
            (
                action_order[action_id],
                repair_round_index,
                remaining_before,
                remaining_after,
            )
        )

    for action in actions:
        validation = _object(
            action.validation_json,
            field="action.validation_json",
        )
        has_executed_repair_intent = (
            validation.get("repair_budget_checked") is True
            or validation.get("repair_round_index") is not None
            or validation.get("remaining_repair_budget_before") is not None
            or validation.get("action_input_hash") is not None
        )
        linked_count = len(typed_repair_by_action.get(str(action.id), []))
        if has_executed_repair_intent and (
            str(action.action_type) not in REPAIR_ACTION_TYPES
            or linked_count != 1
        ):
            raise AgentPEAuditIntegrityError(
                "Validated executed repair action lacks one typed repair observation"
            )

    ordered_rounds = sorted(repair_rounds)
    if [item[1] for item in ordered_rounds] != list(
        range(len(ordered_rounds))
    ):
        raise AgentPEAuditIntegrityError(
            "Repair round indices must be the complete sequence 0..N-1"
        )
    for previous, current in zip(
        ordered_rounds,
        ordered_rounds[1:],
        strict=False,
    ):
        if current[2] != previous[3]:
            raise AgentPEAuditIntegrityError(
                "Repair hard budget does not conserve across rounds"
            )


def load_agent_pe_audit(
    db: Session,
    run_id: str,
) -> AgentPEAuditResponse:
    run = db.get(AgentRun, run_id)
    if run is None:
        raise LookupError("Agent run not found")

    plans = list(
        db.scalars(
            select(AgentPlan)
            .where(AgentPlan.run_id == run.id)
            .order_by(
                AgentPlan.plan_index.asc(),
                AgentPlan.created_at.asc(),
                AgentPlan.id.asc(),
            )
        ).all()
    )
    actions = list(
        db.scalars(
            select(AgentAction)
            .outerjoin(AgentPlan, AgentAction.plan_id == AgentPlan.id)
            .where(AgentAction.run_id == run.id)
            .order_by(
                case((AgentPlan.plan_index.is_(None), 1), else_=0).asc(),
                AgentPlan.plan_index.asc(),
                AgentAction.action_index.asc(),
                AgentAction.created_at.asc(),
                AgentAction.id.asc(),
            )
        ).all()
    )
    observations = list(
        db.scalars(
            select(AgentObservation)
            .where(AgentObservation.run_id == run.id)
            .order_by(
                AgentObservation.created_at.asc(),
                AgentObservation.id.asc(),
            )
        ).all()
    )
    expected_counts = {
        "plans": int(
            db.scalar(
                select(func.count(AgentPlan.id)).where(
                    AgentPlan.run_id == run.id
                )
            )
            or 0
        ),
        "actions": int(
            db.scalar(
                select(func.count(AgentAction.id)).where(
                    AgentAction.run_id == run.id
                )
            )
            or 0
        ),
        "observations": int(
            db.scalar(
                select(func.count(AgentObservation.id)).where(
                    AgentObservation.run_id == run.id
                )
            )
            or 0
        ),
    }
    if expected_counts != {
        "plans": len(plans),
        "actions": len(actions),
        "observations": len(observations),
    }:
        raise AgentPEAuditIntegrityError(
            "Canonical P&E row query did not return every persisted row"
        )

    plan_by_id, action_by_id = _validate_linkage(
        db=db,
        run=run,
        plans=plans,
        actions=actions,
        observations=observations,
    )
    action_ids_by_plan: dict[str, list[str]] = defaultdict(list)
    observation_ids_by_action: dict[str, list[str]] = defaultdict(list)
    for action in actions:
        action_ids_by_plan[str(action.plan_id)].append(str(action.id))
    for observation in observations:
        if observation.action_id is not None:
            observation_ids_by_action[str(observation.action_id)].append(
                str(observation.id)
            )

    plan_rows: list[dict[str, Any]] = []
    for order_index, plan in enumerate(plans):
        planner = _object(
            plan.planner_model_json,
            field="plan.planner_model_json",
        )
        query_intent = _object(
            plan.query_intent_json,
            field="plan.query_intent_json",
        )
        envelope = _object(
            plan.envelope_json,
            field="plan.envelope_json",
        )
        typed_actions = _object_list(
            plan.typed_actions_json,
            field="plan.typed_actions_json",
        )
        validation = _object(
            plan.validation_json,
            field="plan.validation_json",
        )
        diagnostics = _object(
            plan.diagnostics_json,
            field="plan.diagnostics_json",
        )
        planner_public, planner_redacted = _json_payload(
            planner,
            path="planner_model_metadata",
        )
        query_public, query_redacted = _json_payload(
            query_intent,
            path="query_intent",
        )
        envelope_public, envelope_redacted = _json_payload(
            envelope,
            path="operating_envelope",
        )
        actions_public, actions_redacted = _json_payload(
            typed_actions,
            path="typed_actions",
        )
        validation_public, validation_redacted = _json_payload(
            validation,
            path="validation",
        )
        diagnostics_public, diagnostics_redacted = _json_payload(
            diagnostics,
            path="diagnostics",
        )
        plan_rows.append(
            {
                "order_index": order_index,
                "id": str(plan.id),
                "run_id": str(plan.run_id),
                "knowledge_base_id": str(plan.knowledge_base_id),
                "retrieval_trace_id": plan.retrieval_trace_id,
                "plan_index": int(plan.plan_index),
                "planner_protocol_version": planner.get("planner_protocol"),
                "typed_action_schema_protocol_version": _value(
                    validation.get("typed_action_schema_protocol_version"),
                    planner.get("typed_action_schema_protocol"),
                ),
                "typed_action_schema_protocol_hash": validation.get(
                    "typed_action_schema_protocol_hash"
                ),
                "typed_action_executor_protocol_version": diagnostics.get(
                    "typed_action_executor_protocol_version"
                ),
                "input_hash": _value(
                    diagnostics.get("planner_input_hash"),
                    query_intent.get("input_hash"),
                ),
                "output_hash": _value(
                    diagnostics.get("planner_output_hash"),
                    planner.get("output_hash"),
                ),
                "control_hash": diagnostics.get(
                    "typed_action_control_hash"
                ),
                "query_intent": query_public,
                "operating_envelope": envelope_public,
                "typed_actions": actions_public,
                "validation": validation_public,
                "planner_model_metadata": planner_public,
                "status": str(plan.status),
                "diagnostics": diagnostics_public,
                "action_ids": action_ids_by_plan[str(plan.id)],
                "action_count": len(action_ids_by_plan[str(plan.id)]),
                "redacted_fields": sorted(
                    set(
                        planner_redacted
                        + query_redacted
                        + envelope_redacted
                        + actions_redacted
                        + validation_redacted
                        + diagnostics_redacted
                    )
                ),
                "created_at": plan.created_at,
            }
        )

    action_rows: list[dict[str, Any]] = []
    for order_index, action in enumerate(actions):
        plan = plan_by_id[str(action.plan_id)]
        target_ids = _string_list(
            action.target_ids_json,
            field="action.target_ids_json",
        )
        budget = _object(
            action.budget_request_json,
            field="action.budget_request_json",
        )
        expected = _object(
            action.expected_evidence_json,
            field="action.expected_evidence_json",
        )
        stop = _object(
            action.stop_condition_json,
            field="action.stop_condition_json",
        )
        validation = _object(
            action.validation_json,
            field="action.validation_json",
        )
        output = _object(
            action.output_json,
            field="action.output_json",
        )
        diagnostics = _object(
            action.diagnostics_json,
            field="action.diagnostics_json",
        )
        budget_public, budget_redacted = _json_payload(
            budget,
            path="budget_request",
        )
        expected_public, expected_redacted = _json_payload(
            expected,
            path="expected_evidence",
        )
        stop_public, stop_redacted = _json_payload(
            stop,
            path="stop_condition",
        )
        validator, validation_redacted = _validator_payload(validation)
        output_public, output_redacted = _json_payload(
            output,
            path="output",
        )
        diagnostics_public, diagnostics_redacted = _json_payload(
            diagnostics,
            path="diagnostics",
        )
        plan_diagnostics = _object(
            plan.diagnostics_json,
            field="plan.diagnostics_json",
        )
        action_rows.append(
            {
                "order_index": order_index,
                "id": str(action.id),
                "run_id": str(action.run_id),
                "plan_id": str(action.plan_id),
                "plan_index": int(plan.plan_index),
                "parent_action_id": action.parent_action_id,
                "action_index": int(action.action_index),
                "action_type": str(action.action_type),
                "target_ids": target_ids,
                "reason": str(action.reason or ""),
                "budget_request": budget_public,
                "expected_evidence": expected_public,
                "stop_condition": stop_public,
                "validator": validator,
                "status": str(action.status),
                "input_hash": _value(
                    validation.get("action_input_hash"),
                    expected.get("action_input_hash"),
                    diagnostics.get("action_input_hash"),
                    output.get("action_input_hash"),
                ),
                "output_hash": _value(
                    output.get("action_output_hash"),
                    output.get("observation_hash"),
                    diagnostics.get("action_output_hash"),
                ),
                "control_hash": _value(
                    output.get("typed_action_control_hash"),
                    diagnostics.get("typed_action_control_hash"),
                    plan_diagnostics.get("typed_action_control_hash"),
                ),
                "output": output_public,
                "diagnostics": diagnostics_public,
                "observation_ids": observation_ids_by_action[str(action.id)],
                "observation_count": len(
                    observation_ids_by_action[str(action.id)]
                ),
                "redacted_fields": sorted(
                    set(
                        budget_redacted
                        + expected_redacted
                        + stop_redacted
                        + validation_redacted
                        + output_redacted
                        + diagnostics_redacted
                    )
                ),
                "created_at": action.created_at,
            }
        )

    observation_rows: list[dict[str, Any]] = []
    for order_index, observation in enumerate(observations):
        raw_payload = _object(
            observation.observation_json,
            field="observation.observation_json",
        )
        raw_diagnostics = _object(
            observation.diagnostics_json,
            field="observation.diagnostics_json",
        )
        action = (
            action_by_id[str(observation.action_id)]
            if observation.action_id is not None
            else None
        )
        plan_id = (
            str(action.plan_id)
            if action is not None
            else str(raw_payload["plan_id"])
        )
        plan = plan_by_id[plan_id]
        evaluator_linkage: dict[str, Any] | None = None
        repair_linkage: dict[str, Any] | None = None
        evaluator = raw_payload.get("evaluator_verdict")
        bounded = raw_payload.get("bounded_graph_observation")
        bounded_object = bounded if isinstance(bounded, dict) else {}
        if observation.observation_type == "evidence_evaluator":
            if not isinstance(evaluator, dict):
                raise AgentPEAuditIntegrityError(
                    "Evidence-evaluator observation lacks its typed verdict"
                )
            evaluator_verdict = str(evaluator.get("verdict") or "")
            if evaluator_verdict != str(observation.verdict):
                raise AgentPEAuditIntegrityError(
                    "Evidence-evaluator verdict conflicts with its observation row"
                )
            convergence = bounded_object.get("convergence")
            convergence_object = (
                convergence if isinstance(convergence, dict) else {}
            )
            gray_zone_model_call_count = convergence_object.get(
                "gray_zone_model_call_count"
            )
            if (
                isinstance(gray_zone_model_call_count, bool)
                or not isinstance(gray_zone_model_call_count, int)
                or gray_zone_model_call_count != 0
            ):
                raise AgentPEAuditIntegrityError(
                    "Evidence-evaluator observation must explicitly prove "
                    "gray_zone_model_call_count=0"
                )
            evaluator_linkage = {
                "plan_id": plan_id,
                "plan_index": int(plan.plan_index),
                "protocol_version": evaluator.get("protocol_version"),
                "verdict": evaluator_verdict,
                "decision_hash": evaluator.get("decision_hash"),
                # The executor owns the bounded-round control decision.  An
                # ``insufficient_corpus`` verdict can therefore be persisted
                # on a plan whose terminal interpretation was deferred until
                # the remaining planning round ran.
                "replan_requested": str(plan.status)
                == "replan_requested",
                "gray_zone_model_call_count": (
                    gray_zone_model_call_count
                ),
                "schema_repair_attempted": bool(
                    evaluator.get("schema_repair_attempted", False)
                ),
            }
        if observation.observation_type == "typed_repair_round":
            if action is None or str(action.action_type) not in REPAIR_ACTION_TYPES:
                raise AgentPEAuditIntegrityError(
                    "Repair observation is not linked to a typed repair action"
                )
            repair_linkage = {
                "action_id": str(action.id),
                "parent_action_id": action.parent_action_id,
                "action_type": str(action.action_type),
                "repair_protocol_version": raw_payload.get(
                    "protocol_version"
                ),
                "repair_round_index": _integer(
                    raw_payload.get("repair_round_index"),
                    field="repair.repair_round_index",
                ),
                "remaining_repair_budget_before": _integer(
                    raw_payload.get("remaining_repair_budget_before"),
                    field="repair.remaining_repair_budget_before",
                ),
                "remaining_repair_budget_after": _integer(
                    raw_payload.get("remaining_repair_budget_after"),
                    field="repair.remaining_repair_budget_after",
                ),
                "action_input_hash": raw_payload.get("action_input_hash"),
                "action_output_hash": raw_payload.get("action_output_hash"),
                "before_context_package_id": raw_payload.get(
                    "before_context_package_id"
                ),
                "repaired_context_package_id": raw_payload.get(
                    "repaired_context_package_id"
                ),
                "before_retrieval_trace_id": raw_payload.get(
                    "before_retrieval_trace_id"
                ),
                "repaired_retrieval_trace_id": raw_payload.get(
                    "repaired_retrieval_trace_id"
                ),
            }
        payload_public, payload_redacted = _json_payload(
            raw_payload,
            path="observation",
        )
        diagnostics_public, diagnostics_redacted = _json_payload(
            raw_diagnostics,
            path="diagnostics",
        )
        evaluator_object = evaluator if isinstance(evaluator, dict) else {}
        if observation.observation_type == "evidence_evaluator":
            action_output = (
                _object(
                    action.output_json,
                    field="evaluator.action.output_json",
                )
                if action is not None
                else {}
            )
            action_output_evaluator = action_output.get(
                "evaluator_verdict"
            )
            action_output_evaluator_object = (
                action_output_evaluator
                if isinstance(action_output_evaluator, dict)
                else {}
            )
            observation_output_hash = _consistent_lineage_value(
                raw_payload.get("action_output_hash"),
                evaluator_object.get("decision_hash"),
                raw_payload.get("observation_hash"),
                action_output.get("action_output_hash"),
                action_output.get("observation_hash"),
                action_output_evaluator_object.get("decision_hash"),
                field="evaluator decision/output",
            )
        else:
            observation_output_hash = _value(
                raw_payload.get("action_output_hash"),
                evaluator_object.get("decision_hash"),
                raw_payload.get("observation_hash"),
            )
        observation_rows.append(
            {
                "order_index": order_index,
                "id": str(observation.id),
                "run_id": str(observation.run_id),
                "plan_id": plan_id,
                "plan_index": int(plan.plan_index),
                "action_id": (
                    str(observation.action_id)
                    if observation.action_id is not None
                    else None
                ),
                "action_index": (
                    int(action.action_index) if action is not None else None
                ),
                "parent_action_id": (
                    action.parent_action_id if action is not None else None
                ),
                "observation_type": str(observation.observation_type),
                "protocol_version": _value(
                    raw_payload.get("protocol_version"),
                    evaluator_object.get("protocol_version"),
                ),
                "input_hash": _value(
                    raw_payload.get("action_input_hash"),
                    bounded_object.get("observation_hash"),
                ),
                "output_hash": observation_output_hash,
                "control_hash": raw_payload.get(
                    "typed_action_control_hash"
                ),
                "evaluator_linkage": evaluator_linkage,
                "repair_linkage": repair_linkage,
                "evidence_chunk_ids": _string_list(
                    observation.evidence_chunk_ids_json,
                    field="observation.evidence_chunk_ids_json",
                ),
                "verdict": str(observation.verdict),
                "observation": payload_public,
                "diagnostics": diagnostics_public,
                "redacted_fields": sorted(
                    set(payload_redacted + diagnostics_redacted)
                ),
                "created_at": observation.created_at,
            }
        )

    provider_raw_response_exposed, credentials_exposed = (
        _scan_canonical_payload_exposure(
            {
                "plans": plan_rows,
                "actions": action_rows,
                "observations": observation_rows,
            }
        )
    )
    try:
        return AgentPEAuditResponse.model_validate(
            {
                "run_id": str(run.id),
                "knowledge_base_id": str(run.knowledge_base_id),
                "run_status": str(run.status),
                "counts": expected_counts,
                "ordering": {
                    "plans": PLAN_ORDER,
                    "actions": ACTION_ORDER,
                    "observations": OBSERVATION_ORDER,
                },
                "plans": plan_rows,
                "actions": action_rows,
                "observations": observation_rows,
                "redaction_protocol_version": (
                    AGENT_PE_SENSITIVE_KEY_PROTOCOL_VERSION
                ),
                "provider_raw_response_exposed": (
                    provider_raw_response_exposed
                ),
                "credentials_exposed": credentials_exposed,
            }
        )
    except ValidationError as exc:
        raise AgentPEAuditIntegrityError(
            "Persisted P&E rows do not satisfy the closed public audit contract"
        ) from exc
