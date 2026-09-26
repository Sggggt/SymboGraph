"""Deterministic working-memory planning for model tool sessions.

This module owns no facts.  It plans which semantic units may be presented to
the model and emits a body-free audit; authoritative text remains in the
Context Package or the corresponding navigation resource.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Literal, Sequence

from app.retrieval_control_contracts import control_hash


CONTEXT_PLAN_PROTOCOL = "agent_context_plan_v1"
CONTEXT_EVENT_PROTOCOL = "context_event_log_v1"
Priority = Literal["P0", "P1", "P2", "P3", "P4"]
_PRIORITY_ORDER: dict[Priority, int] = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
}


class ContextCapacityError(ValueError):
    pass


@dataclass(frozen=True)
class ContextTreeNode:
    key: str
    kind: Literal["task", "requirement", "semantic_node", "source", "raw_chunk"]
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextTree:
    root: str
    nodes: tuple[ContextTreeNode, ...]

    def audit(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for node in self.nodes:
            by_kind[node.kind] = by_kind.get(node.kind, 0) + 1
        return {
            "root": self.root,
            "node_count": len(self.nodes),
            "nodes_by_kind": dict(sorted(by_kind.items())),
            "tree_hash": control_hash(
                [
                    {"key": node.key, "kind": node.kind, "children": list(node.children)}
                    for node in self.nodes
                ]
            ),
        }


@dataclass(frozen=True)
class PriorityCandidate:
    key: str
    explicit_source_responsibility: bool = False
    uncovered_requirement: bool = False
    active_branch: bool = False
    retrieval_order: int = 0
    recent_dependency: bool = False
    estimated_tokens: int = 0
    stable_key: str = ""


def stable_priority_order(candidates: Iterable[PriorityCandidate]) -> tuple[str, ...]:
    """Return the server-owned stable semantic load order."""

    return tuple(
        item.key
        for item in sorted(
            candidates,
            key=lambda item: (
                not item.explicit_source_responsibility,
                not item.uncovered_requirement,
                not item.active_branch,
                max(0, int(item.retrieval_order)),
                not item.recent_dependency,
                max(0, int(item.estimated_tokens)),
                item.stable_key or item.key,
                item.key,
            ),
        )
    )


@dataclass(frozen=True)
class ContextUnit:
    key: str
    kind: str
    priority: Priority
    content: str
    atomic_group: str | None = None
    set_name: Literal["pinned", "working", "compressed", "evicted"] = "working"
    dedupe_key: str | None = None
    handle: str | None = None

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.content)


@dataclass(frozen=True)
class ContextPlanResult:
    kept_keys: tuple[str, ...]
    audit: dict[str, Any]


def estimate_tokens(text: str) -> int:
    """Versioned fallback used only when provider token accounting is absent."""

    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def _grouped(units: Sequence[ContextUnit]) -> list[list[ContextUnit]]:
    groups: list[list[ContextUnit]] = []
    positions: dict[str, int] = {}
    for unit in units:
        group_key = unit.atomic_group or f"unit:{unit.key}"
        position = positions.get(group_key)
        if position is None:
            positions[group_key] = len(groups)
            groups.append([unit])
        else:
            groups[position].append(unit)
    return groups


def plan_context(
    units: Sequence[ContextUnit],
    *,
    input_token_budget: int,
    reserved_output_tokens: int,
    stable_prefix: str,
    input_token_count: int | None = None,
) -> ContextPlanResult:
    """Select complete semantic units without splitting atomic tool pairs."""

    if input_token_budget <= 0 or reserved_output_tokens < 0:
        raise ValueError("context_budget_invalid")
    if len({unit.key for unit in units}) != len(units):
        raise ValueError("context_unit_key_duplicate")

    kept = list(units)
    removed: list[dict[str, Any]] = []
    seen_dedupe: set[str] = set()
    deduped: list[ContextUnit] = []
    for unit in kept:
        if (
            unit.dedupe_key
            and unit.atomic_group is None
            and unit.dedupe_key in seen_dedupe
        ):
            removed.append(
                {
                    "key": unit.key,
                    "kind": unit.kind,
                    "handle": unit.handle,
                    "priority": unit.priority,
                    "reason": "duplicate_semantic_unit",
                }
            )
            continue
        if unit.dedupe_key and unit.atomic_group is None:
            seen_dedupe.add(unit.dedupe_key)
        deduped.append(unit)
    kept = deduped

    def total_tokens(values: Sequence[ContextUnit]) -> int:
        return sum(unit.estimated_tokens for unit in values)

    groups = _grouped(kept)
    remove_group_ids: set[int] = set()
    for index, group in enumerate(groups):
        if all(unit.priority == "P4" for unit in group):
            remove_group_ids.add(index)
            for unit in group:
                removed.append(
                    {
                        "key": unit.key,
                        "kind": unit.kind,
                        "handle": unit.handle,
                        "priority": unit.priority,
                        "reason": "default_excluded_p4",
                    }
                )
    kept = [
        unit
        for index, group in enumerate(groups)
        if index not in remove_group_ids
        for unit in group
    ]

    for priority, reason in (("P2", "compressed_control_history"), ("P3", "evicted_on_budget")):
        if total_tokens(kept) <= input_token_budget:
            break
        groups = _grouped(kept)
        candidates = [
            (index, group)
            for index, group in enumerate(groups)
            if all(_PRIORITY_ORDER[unit.priority] >= _PRIORITY_ORDER[priority] for unit in group)
            and any(unit.priority == priority for unit in group)
        ]
        for index, group in candidates:
            if total_tokens(kept) <= input_token_budget:
                break
            group_keys = {unit.key for unit in group}
            kept = [unit for unit in kept if unit.key not in group_keys]
            for unit in group:
                removed.append(
                    {
                        "key": unit.key,
                        "kind": unit.kind,
                        "handle": unit.handle,
                        "priority": unit.priority,
                        "reason": reason,
                    }
                )

    estimated = total_tokens(kept)
    if estimated > input_token_budget:
        raise ContextCapacityError("context_p0_p1_capacity_exceeded")

    priority_counts = {key: 0 for key in _PRIORITY_ORDER}
    set_counts = {key: 0 for key in ("pinned", "working", "compressed", "evicted")}
    for unit in kept:
        priority_counts[unit.priority] += 1
        set_counts[unit.set_name] += 1
    set_counts["compressed"] += sum(
        item["reason"] == "compressed_control_history" for item in removed
    )
    set_counts["evicted"] += sum(
        item["reason"] in {"default_excluded_p4", "evicted_on_budget"}
        for item in removed
    )
    audit = {
        "protocol_version": CONTEXT_PLAN_PROTOCOL,
        "token_accounting_mode": (
            "provider_actual" if input_token_count is not None else "utf8_quarter_estimate_v1"
        ),
        "estimated_input_tokens": estimated,
        "input_token_count": input_token_count,
        "output_token_count": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "total_token_count": None,
        "input_token_budget": input_token_budget,
        "output_token_budget": reserved_output_tokens,
        "prompt_cache_prefix_hash": hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest(),
        "priority_units": priority_counts,
        "sets": set_counts,
        "compression_applied": any(
            item["reason"] == "compressed_control_history" for item in removed
        ),
        "truncation_applied": any(
            item["reason"] == "evicted_on_budget" for item in removed
        ),
        "removed_units": removed,
        "atomic_groups_preserved": True,
        "body_persisted": False,
    }
    audit["plan_hash"] = control_hash(audit)
    return ContextPlanResult(tuple(unit.key for unit in kept), audit)


def apply_provider_usage(
    audit: dict[str, Any],
    provider_call: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach provider-reported counters without inferring missing values."""

    usage = dict((provider_call or {}).get("usage") or {})
    mapping = {
        "input_tokens": "input_token_count",
        "output_tokens": "output_token_count",
        "cache_creation_input_tokens": "cache_creation_input_tokens",
        "cache_read_input_tokens": "cache_read_input_tokens",
        "total_tokens": "total_token_count",
    }
    value = dict(audit)
    observed = False
    for provider_key, audit_key in mapping.items():
        count = usage.get(provider_key)
        value[audit_key] = count if type(count) is int and count >= 0 else None
        observed = observed or value[audit_key] is not None
    if value["input_token_count"] is not None:
        value["token_accounting_mode"] = "provider_actual"
    elif observed:
        value["token_accounting_mode"] = "provider_partial"
    value["plan_hash"] = control_hash(
        {key: item for key, item in value.items() if key != "plan_hash"}
    )
    return value


def context_event(
    event_type: Literal[
        "user_task",
        "tool_call",
        "tool_result_ref",
        "context_compacted",
        "context_evicted",
        "evidence_committed",
        "final_generation",
    ],
    **payload: Any,
) -> dict[str, Any]:
    value = {
        "protocol_version": CONTEXT_EVENT_PROTOCOL,
        "event_type": event_type,
        **payload,
    }
    value["event_hash"] = control_hash(value)
    return value


def validate_tool_event_pairs(events: Sequence[dict[str, Any]]) -> None:
    """Require every persisted tool call to be followed by one result ref."""

    awaiting_result = False
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("context_event_invalid")
        unsigned = {key: value for key, value in event.items() if key != "event_hash"}
        if (
            event.get("protocol_version") != CONTEXT_EVENT_PROTOCOL
            or event.get("event_hash") != control_hash(unsigned)
        ):
            raise ValueError("context_event_hash_invalid")
        event_type = event.get("event_type")
        if event_type == "tool_call":
            if awaiting_result:
                raise ValueError("context_tool_pair_split")
            awaiting_result = True
        elif event_type == "tool_result_ref":
            if not awaiting_result:
                raise ValueError("context_tool_result_without_call")
            awaiting_result = False
        elif awaiting_result:
            raise ValueError("context_tool_pair_split")
    if awaiting_result:
        raise ValueError("context_tool_call_without_result")


def json_message(role: Literal["user", "assistant"], payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": role,
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    }
