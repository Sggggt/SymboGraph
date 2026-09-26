"""Mid-semantic, replayable evidence tool session over one admitted package."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Literal

from pydantic import Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.models import (
    AgentObservation,
    AgentRun,
    ContextGraphState,
    ContextPackage,
    MidConcept,
    RQPrefix,
    RQPrefixMembership,
    RetrievalTrace,
)
from app.retrieval_control_contracts import ControlContract, control_hash
from app.services.agent_context import (
    ContextCapacityError,
    ContextTree,
    ContextTreeNode,
    ContextUnit,
    PriorityCandidate,
    apply_provider_usage,
    context_event,
    estimate_tokens,
    json_message,
    plan_context,
    stable_priority_order,
    validate_tool_event_pairs,
)
from app.services.answer_sources import (
    AnswerEvidenceManifest,
    build_answer_evidence_manifest,
    build_generation_evidence_view,
    replay_generation_evidence_view,
)
from app.services.context_graph import context_package_to_contexts
from app.services.qa_performance import qa_stage
from app.services.reflection_models import AnswerReviewModelError


LOOP_PROTOCOL = "evidence_read_loop_v2"
TOOL_CALL_PROTOCOL = "evidence_tool_call_v1"
TOOL_RESULT_PROTOCOL = "evidence_tool_result_v1"
OBSERVATION_TYPE = "evidence_read_loop"
DIRECT_CONTEXT_TOKEN_LIMIT = 2048


class EvidenceReadArguments(ControlContract):
    mid_handles: tuple[str, ...]

    @model_validator(mode="after")
    def validate_handles(self):
        if not self.mid_handles:
            raise ValueError("evidence_read_mid_handles_empty")
        if any(not _valid_handle(handle, "mid_") for handle in self.mid_handles):
            raise ValueError("evidence_tool_handles_invalid")
        if len(set(self.mid_handles)) != len(self.mid_handles):
            raise ValueError("evidence_tool_handles_duplicate")
        return self


class EvidenceCommitArguments(ControlContract):
    source_handles: tuple[str, ...]

    @model_validator(mode="after")
    def validate_handles(self):
        if any(not _valid_handle(handle, "src_") for handle in self.source_handles):
            raise ValueError("evidence_tool_handles_invalid")
        if len(set(self.source_handles)) != len(self.source_handles):
            raise ValueError("evidence_tool_handles_duplicate")
        return self


class EvidenceToolCall(ControlContract):
    protocol_version: Literal["evidence_tool_call_v1"] = TOOL_CALL_PROTOCOL
    tool: Literal["evidence.read", "evidence.commit"]
    arguments: EvidenceReadArguments | EvidenceCommitArguments

    @model_validator(mode="after")
    def validate_arguments(self):
        if self.tool == "evidence.read" and not isinstance(
            self.arguments, EvidenceReadArguments
        ):
            raise ValueError("evidence_read_arguments_invalid")
        if self.tool == "evidence.commit" and not isinstance(
            self.arguments, EvidenceCommitArguments
        ):
            raise ValueError("evidence_commit_arguments_invalid")
        return self

    @property
    def argument_payload(self) -> dict[str, Any]:
        return self.arguments.model_dump(mode="json")

    @property
    def mid_handles(self) -> tuple[str, ...]:
        return self.arguments.mid_handles if isinstance(self.arguments, EvidenceReadArguments) else ()

    @property
    def source_handles(self) -> tuple[str, ...]:
        return self.arguments.source_handles if isinstance(self.arguments, EvidenceCommitArguments) else ()


# Compatibility import for callers that only referenced the former class name.
EvidenceReadDecision = EvidenceToolCall


class EvidenceDecisionStateError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MidDirectory:
    active_state_hash: str | None
    model_entries: tuple[dict[str, str], ...]
    mid_to_sources: dict[str, tuple[str, ...]]
    source_to_mids: dict[str, tuple[str, ...]]
    mandatory_sources: tuple[str, ...]
    server_identity: dict[str, Any]
    tree: ContextTree

    @property
    def mid_handles(self) -> tuple[str, ...]:
        return tuple(item["mid_handle"] for item in self.model_entries)

    @property
    def identity(self) -> str:
        return control_hash(self.server_identity)


@dataclass(frozen=True)
class EvidenceLoopResult:
    admitted_evidence: AnswerEvidenceManifest
    generation_evidence: AnswerEvidenceManifest
    generation_view: dict[str, Any]
    observation: AgentObservation
    selected_contexts: tuple[dict[str, Any], ...]


def evidence_tool_system_prompt() -> str:
    return "\n".join(
        [
            "EVIDENCE TOOL SESSION V1. Submit exactly one action through the provided structured interface; never emit prose, Markdown, an answer, rationale, or a JSON example as text.",
            "The admitted raw text returned by evidence.read is the only factual authority. Mid titles and summaries are navigation only.",
            "Call evidence.read with one or more unread mid_handles. It returns all currently admitted raw sources under those semantic nodes.",
            "Call evidence.commit with only source_handles returned by earlier reads. Empty commit is legal when nothing supports the request.",
            "Never invent, repeat, or expose internal identifiers. Do not output coverage, scores, reasoning, quotations, or user-visible prose.",
        ]
    )


def _state_hash(state: dict[str, Any]) -> str:
    return control_hash({key: value for key, value in state.items() if key != "state_hash"})


def _seal_state(state: dict[str, Any]) -> dict[str, Any]:
    value = dict(state)
    value["state_hash"] = _state_hash(value)
    return value


def _observation_payload(
    base: dict[str, Any],
    state: dict[str, Any],
    *,
    view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **base,
        "state": _seal_state(state),
        "source_text_persisted": False,
        "directory_text_persisted": False,
    }
    if view is not None:
        payload["generation_evidence_view"] = view
    unsigned = {key: value for key, value in payload.items() if key != "audit_hash"}
    return {**unsigned, "audit_hash": control_hash(unsigned)}


def _persist_observation(
    observation: AgentObservation,
    *,
    base: dict[str, Any],
    state: dict[str, Any],
    verdict: str,
    view: dict[str, Any] | None = None,
) -> None:
    observation.verdict = verdict
    observation.observation_json = _observation_payload(base, state, view=view)
    flag_modified(observation, "observation_json")


def _valid_handle(value: Any, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    suffix = value[len(prefix) :]
    return bool(suffix and suffix.isascii() and suffix.isdigit() and not suffix.startswith("0"))


def _validate_state(
    state: dict[str, Any],
    *,
    all_mids: tuple[str, ...],
    all_sources: tuple[str, ...],
    mandatory_sources: tuple[str, ...],
) -> None:
    if state.get("state_hash") != _state_hash(state):
        raise EvidenceDecisionStateError("evidence_loop_state_hash_invalid")
    remaining = tuple(state.get("remaining_mid_handles") or ())
    read_mids = tuple(state.get("read_mid_handles") or ())
    read_sources = tuple(state.get("read_source_handles") or ())
    committed = tuple(state.get("committed_source_handles") or ())
    events = list(state.get("events") or ())
    if (
        state.get("phase") not in {"reading", "finalized", "failed", "cancelled"}
        or len(set(remaining)) != len(remaining)
        or len(set(read_mids)) != len(read_mids)
        or len(set(read_sources)) != len(read_sources)
        or len(set(committed)) != len(committed)
        or set(remaining) & set(read_mids)
        or set(remaining) | set(read_mids) != set(all_mids)
        or any(handle not in all_sources for handle in (*read_sources, *committed))
        or any(handle not in all_sources for handle in mandatory_sources)
        or any(handle not in set(read_sources) | set(mandatory_sources) for handle in committed)
        or any(not _valid_handle(handle, "mid_") for handle in all_mids)
        or any(not _valid_handle(handle, "src_") for handle in all_sources)
    ):
        raise EvidenceDecisionStateError("evidence_loop_state_invalid")
    validate_tool_event_pairs(events)
    plans = list(state.get("context_plans") or ())
    if any(
        not isinstance(item, dict)
        or item.get("protocol_version") != "agent_context_plan_v1"
        or item.get("body_persisted") is not False
        or item.get("plan_hash")
        != control_hash({key: value for key, value in item.items() if key != "plan_hash"})
        for item in plans
    ):
        raise EvidenceDecisionStateError("evidence_context_plan_invalid")


def _validate_observation(
    observation: AgentObservation,
    *,
    base: dict[str, Any],
    directory: MidDirectory,
    all_sources: tuple[str, ...],
) -> dict[str, Any]:
    payload = dict(observation.observation_json or {})
    unsigned = {key: value for key, value in payload.items() if key != "audit_hash"}
    if (
        observation.observation_type != OBSERVATION_TYPE
        or any(payload.get(key) != value for key, value in base.items())
        or payload.get("source_text_persisted") is not False
        or payload.get("directory_text_persisted") is not False
        or payload.get("audit_hash") != control_hash(unsigned)
        or not isinstance(payload.get("state"), dict)
    ):
        raise EvidenceDecisionStateError("evidence_loop_observation_invalid")
    state = dict(payload["state"])
    _validate_state(
        state,
        all_mids=directory.mid_handles,
        all_sources=all_sources,
        mandatory_sources=directory.mandatory_sources,
    )
    return state


def _path_parent_ids(trace: RetrievalTrace, source: dict[str, Any]) -> tuple[str, ...]:
    chunk_id = str(source["chunk_id"])
    parent_ids: list[str] = []
    for label in trace.path_labels_json or []:
        if not isinstance(label, dict) or str(label.get("chunk_id") or label.get("node_id") or "") != chunk_id:
            continue
        if label.get("parent_layer") == "mid" and label.get("parent_node_id"):
            parent_ids.append(str(label["parent_node_id"]))
        for ref in label.get("entry_parent_refs") or []:
            if isinstance(ref, dict) and ref.get("parent_layer") == "mid" and ref.get("parent_node_id"):
                parent_ids.append(str(ref["parent_node_id"]))
    why = dict(source.get("package_item", {}).get("why_selected") or {})
    parent_ids.extend(str(value) for value in why.get("parent_node_ids") or [] if value)
    for path in why.get("reached_by_paths") or []:
        if isinstance(path, dict) and path.get("parent_node_id"):
            parent_ids.append(str(path["parent_node_id"]))
    return tuple(dict.fromkeys(parent_ids))


def build_mid_directory(
    db,
    *,
    plan,
    package: ContextPackage,
    trace: RetrievalTrace,
    evidence: AnswerEvidenceManifest,
) -> MidDirectory:
    """Project admitted sources to grounded active Mid concepts."""

    evidence.verify_integrity()
    state = db.scalar(
        select(ContextGraphState)
        .where(
            ContextGraphState.knowledge_base_id == package.knowledge_base_id,
            ContextGraphState.state == "active",
            ContextGraphState.mid_concept_hash == trace.mid_concept_hash,
        )
        .order_by(ContextGraphState.created_at.desc())
    )
    concepts: list[MidConcept] = []
    if state is not None and state.mid_concept_state_id:
        concepts = list(
            db.scalars(
                select(MidConcept)
                .where(
                    MidConcept.concept_state_id == state.mid_concept_state_id,
                    MidConcept.knowledge_base_id == package.knowledge_base_id,
                    MidConcept.state == "active",
                )
                .order_by(MidConcept.canonical_label, MidConcept.id)
            )
        )
    concepts = [
        concept
        for concept in concepts
        if str(concept.canonical_label or "").strip()
        and str(concept.summary or concept.definition or "").strip()
    ]
    by_id = {str(concept.id): concept for concept in concepts}
    source_to_concept_ids: dict[str, tuple[str, ...]] = {}
    unresolved: list[dict[str, Any]] = []
    for source in evidence.sources:
        candidates = tuple(
            concept_id
            for concept_id in _path_parent_ids(trace, source)
            if concept_id in by_id
            and str(source["chunk_id"]) in set(by_id[concept_id].support_chunk_ids_json or [])
        )
        candidates = tuple(dict.fromkeys(candidates))
        if candidates:
            source_to_concept_ids[source["source_handle"]] = candidates
        else:
            unresolved.append(source)

    if unresolved and state is not None and state.chunk_relation_graph_state_id:
        chunk_ids = [str(source["chunk_id"]) for source in unresolved]
        rows = list(
            db.execute(
                select(RQPrefixMembership.chunk_id, RQPrefixMembership.rq_prefix_id)
                .join(RQPrefix, RQPrefix.id == RQPrefixMembership.rq_prefix_id)
                .where(
                    RQPrefixMembership.chunk_id.in_(chunk_ids),
                    RQPrefix.rq_level == 3,
                    RQPrefix.graph_state_id == state.chunk_relation_graph_state_id,
                )
            )
        )
        prefixes_by_chunk: dict[str, list[str]] = {}
        for chunk_id, prefix_id in rows:
            prefixes_by_chunk.setdefault(str(chunk_id), []).append(str(prefix_id))
        concepts_by_prefix: dict[str, list[str]] = {}
        for concept in concepts:
            if concept.support_rq_l3_prefix_id:
                concepts_by_prefix.setdefault(str(concept.support_rq_l3_prefix_id), []).append(str(concept.id))
        for source in unresolved:
            prefixes = tuple(dict.fromkeys(prefixes_by_chunk.get(str(source["chunk_id"]), ())))
            candidates = tuple(
                dict.fromkeys(
                    concept_id
                    for prefix_id in prefixes
                    for concept_id in concepts_by_prefix.get(prefix_id, ())
                    if str(source["chunk_id"]) in set(by_id[concept_id].support_chunk_ids_json or [])
                )
            )
            if len(prefixes) == 1 and len(candidates) == 1:
                source_to_concept_ids[source["source_handle"]] = candidates

    concept_to_sources: dict[str, list[str]] = {}
    for source in evidence.sources:
        for concept_id in source_to_concept_ids.get(source["source_handle"], ()):
            concept_to_sources.setdefault(concept_id, []).append(source["source_handle"])
    has_source_scope = any(
        requirement.source_scope is not None or requirement.source_roles
        for requirement in plan.task.requirements
    )
    first_source_order = {
        source["source_handle"]: index for index, source in enumerate(evidence.sources)
    }
    ordered_concepts = stable_priority_order(
        PriorityCandidate(
            key=concept_id,
            explicit_source_responsibility=has_source_scope,
            uncovered_requirement=bool(plan.task.requirements),
            active_branch=True,
            retrieval_order=min(first_source_order[handle] for handle in handles),
            recent_dependency=False,
            estimated_tokens=sum(
                estimate_tokens(evidence.by_handle()[handle]["text"]) for handle in handles
            ),
            stable_key=str(by_id[concept_id].canonical_label).casefold(),
        )
        for concept_id, handles in concept_to_sources.items()
    )
    mid_by_concept = {
        concept_id: f"mid_{index}" for index, concept_id in enumerate(ordered_concepts, start=1)
    }
    model_entries = tuple(
        {
            "mid_handle": mid_by_concept[concept_id],
            "title": str(by_id[concept_id].canonical_label),
            "summary": str(by_id[concept_id].summary or by_id[concept_id].definition),
        }
        for concept_id in ordered_concepts
    )
    mid_to_sources = {
        mid_by_concept[concept_id]: tuple(dict.fromkeys(concept_to_sources[concept_id]))
        for concept_id in ordered_concepts
    }
    source_to_mids = {
        source["source_handle"]: tuple(
            mid_by_concept[concept_id]
            for concept_id in source_to_concept_ids.get(source["source_handle"], ())
            if concept_id in mid_by_concept
        )
        for source in evidence.sources
    }
    mandatory = tuple(
        source["source_handle"]
        for source in evidence.sources
        if not source_to_mids[source["source_handle"]]
    )
    requirement_keys = tuple(f"requirement:{item.id}" for item in plan.task.requirements)
    mid_keys = tuple(f"semantic:{item['mid_handle']}" for item in model_entries)
    nodes: list[ContextTreeNode] = [
        ContextTreeNode("task", "task", requirement_keys or mid_keys)
    ]
    nodes.extend(
        ContextTreeNode(key, "requirement", mid_keys) for key in requirement_keys
    )
    for item in model_entries:
        mid_handle = item["mid_handle"]
        source_keys = tuple(f"source:{handle}" for handle in mid_to_sources[mid_handle])
        nodes.append(ContextTreeNode(f"semantic:{mid_handle}", "semantic_node", source_keys))
        nodes.extend(ContextTreeNode(key, "source") for key in source_keys)
    for handle in mandatory:
        nodes.append(ContextTreeNode(f"source:{handle}", "source"))
    tree = ContextTree("task", tuple(dict.fromkeys(nodes)))
    server_identity = {
        "active_mid_state_hash": str(state.mid_concept_hash) if state is not None else None,
        "entries": [
            {
                "mid_handle": mid_by_concept[concept_id],
                "concept_id": concept_id,
                "source_handles": list(mid_to_sources[mid_by_concept[concept_id]]),
                "semantic_hash": control_hash(
                    {
                        "title": str(by_id[concept_id].canonical_label),
                        "summary": str(by_id[concept_id].summary or by_id[concept_id].definition),
                    }
                ),
            }
            for concept_id in ordered_concepts
        ],
        "mandatory_sources": list(mandatory),
        "tree_hash": tree.audit()["tree_hash"],
    }
    return MidDirectory(
        active_state_hash=str(state.mid_concept_hash) if state is not None else None,
        model_entries=model_entries,
        mid_to_sources=mid_to_sources,
        source_to_mids=source_to_mids,
        mandatory_sources=mandatory,
        server_identity=server_identity,
        tree=tree,
    )


def _initial_message(plan, directory: MidDirectory) -> dict[str, Any]:
    return {
        "protocol_version": "evidence_tool_session_v1",
        "task": {
            "question": plan.task.question,
            "requirements": [item.model_dump(mode="json") for item in plan.task.requirements],
            "response_constraints": [
                item.model_dump(mode="json") for item in plan.task.response_constraints
            ],
        },
        "semantic_directory": list(directory.model_entries),
        "directory_complete": True,
        "tools": {
            "evidence.read": {"required": ["mid_handles"]},
            "evidence.commit": {"required": ["source_handles"]},
        },
    }


def _read_result(
    *,
    groups: list[dict[str, Any]],
    remaining_mid_handles: list[str],
    directory: MidDirectory,
    evidence: AnswerEvidenceManifest,
) -> dict[str, Any]:
    by_handle = evidence.by_handle()
    return {
        "protocol_version": TOOL_RESULT_PROTOCOL,
        "tool": "evidence.read",
        "status": "ok",
        "groups": [
            {
                "mid_handle": group["mid_handle"],
                "sources": [
                    {"source_handle": handle, "text": by_handle[handle]["text"]}
                    for handle in group["source_handles"]
                ],
            }
            for group in groups
        ],
        "remaining_mid_handles": remaining_mid_handles,
    }


def _messages_and_units(
    *,
    plan,
    directory: MidDirectory,
    evidence: AnswerEvidenceManifest,
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[ContextUnit]]:
    validate_tool_event_pairs(events)
    initial = json_message("user", _initial_message(plan, directory))
    messages = [initial]
    units = [ContextUnit("message:0", "user_task", "P0", initial["content"], set_name="pinned")]
    tool_events = [
        event
        for event in events
        if event.get("event_type") in {"tool_call", "tool_result_ref"}
    ]
    pairs = [
        (tool_events[index], tool_events[index + 1])
        for index in range(0, len(tool_events), 2)
    ]
    for pair_index, (call_event, result_event) in enumerate(pairs, start=1):
        call_payload = {
            "protocol_version": TOOL_CALL_PROTOCOL,
            "tool": call_event["tool"],
            "arguments": dict(call_event.get("arguments") or {}),
        }
        if result_event.get("status") == "ok":
            result_payload = _read_result(
                groups=list(result_event.get("groups") or []),
                remaining_mid_handles=list(result_event.get("remaining_mid_handles") or []),
                directory=directory,
                evidence=evidence,
            )
        else:
            result_payload = {
                "protocol_version": TOOL_RESULT_PROTOCOL,
                "tool": call_event.get("tool"),
                "status": "error",
                "error": result_event.get("error"),
                "field": result_event.get("field"),
            }
        call_message = json_message("assistant", call_payload)
        result_message = json_message("user", result_payload)
        messages.extend((call_message, result_message))
        priority = "P1" if result_event.get("status") == "ok" or pair_index == len(pairs) else "P2"
        group_key = f"tool_pair:{pair_index}"
        units.extend(
            [
                ContextUnit(
                    f"message:{2 * pair_index - 1}",
                    "tool_call",
                    priority,
                    call_message["content"],
                    atomic_group=group_key,
                    set_name="working" if priority == "P1" else "compressed",
                ),
                ContextUnit(
                    f"message:{2 * pair_index}",
                    "tool_result",
                    priority,
                    result_message["content"],
                    atomic_group=group_key,
                    set_name="working" if priority == "P1" else "compressed",
                ),
            ]
        )
    return messages, units


def _selected_contexts(
    package: ContextPackage,
    evidence: AnswerEvidenceManifest,
    selected_handles: list[str],
) -> tuple[dict[str, Any], ...]:
    contexts = {item["chunk_id"]: item for item in context_package_to_contexts(package)}
    sources = evidence.by_handle()
    return tuple(contexts[sources[handle]["chunk_id"]] for handle in selected_handles)


def _validate_identity(
    db,
    *,
    run,
    package,
    trace,
    admission,
    plan,
    evidence_hash: str,
    directory_hash: str,
) -> tuple[AnswerEvidenceManifest, MidDirectory]:
    locked_run = db.scalar(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
    refreshed_package = db.get(ContextPackage, package.id)
    refreshed_trace = db.get(RetrievalTrace, trace.id)
    admission_row = db.get(AgentObservation, admission.observation_id)
    admission_payload = dict(admission_row.observation_json or {}) if admission_row is not None else {}
    if (
        locked_run is None
        or locked_run.status != "running"
        or refreshed_package is None
        or refreshed_trace is None
        or refreshed_package.retrieval_trace_id != refreshed_trace.id
        or refreshed_package.knowledge_base_id != locked_run.knowledge_base_id
        or admission_row is None
        or admission_row.run_id != locked_run.id
        or admission_row.verdict != "passed"
        or admission_payload.get("audit_hash") != admission.audit.get("audit_hash")
        or (locked_run.metadata_json or {}).get("accepted_task")
        != plan.task.model_dump(mode="json")
    ):
        raise EvidenceDecisionStateError("evidence_loop_identity_changed")
    manifest = build_answer_evidence_manifest(
        refreshed_package,
        context_package_to_contexts(refreshed_package),
    )
    if (
        manifest.manifest_hash != evidence_hash
        or admission_payload.get("evidence_manifest_hash") != manifest.manifest_hash
    ):
        raise EvidenceDecisionStateError("evidence_loop_manifest_changed")
    directory = build_mid_directory(
        db,
        plan=plan,
        package=refreshed_package,
        trace=refreshed_trace,
        evidence=manifest,
    )
    if directory.identity != directory_hash:
        raise EvidenceDecisionStateError("evidence_mid_directory_changed")
    return manifest, directory


def _append_error_pair(
    state: dict[str, Any],
    *,
    tool: str,
    arguments: dict[str, Any],
    error: str,
    field: str,
) -> None:
    state["events"].extend(
        [
            context_event("tool_call", tool=tool, arguments=arguments),
            context_event(
                "tool_result_ref",
                tool=tool,
                status="error",
                error=error,
                field=field,
            ),
        ]
    )
    state["error_count"] = int(state.get("error_count") or 0) + 1


def mark_evidence_loop_terminal(db, *, run_id: str, cancelled: bool, error_code: str) -> None:
    """Persist a body-free terminal marker after an in-process failure."""

    db.rollback()
    observation = db.scalar(
        select(AgentObservation).where(
            AgentObservation.run_id == run_id,
            AgentObservation.observation_type == OBSERVATION_TYPE,
        )
    )
    if observation is None or observation.verdict == "finalized":
        return
    payload = dict(observation.observation_json or {})
    state = dict(payload.get("state") or {})
    if not state:
        return
    state["phase"] = "cancelled" if cancelled else "failed"
    state["terminal_error_code"] = (
        error_code if error_code and len(error_code) <= 96 else "evidence_tool_failure"
    )
    base = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "state",
            "source_text_persisted",
            "directory_text_persisted",
            "generation_evidence_view",
            "audit_hash",
        }
    }
    _persist_observation(
        observation,
        base=base,
        state=_seal_state(state),
        verdict=state["phase"],
    )
    db.commit()


def record_final_generation_event(
    db,
    *,
    observation_id: str,
    generation_view_hash: str,
    context_plan_hash: str,
) -> None:
    """Append the body-free final-generation reference to the context log."""

    observation = db.get(AgentObservation, observation_id)
    if observation is None or observation.verdict != "finalized":
        raise EvidenceDecisionStateError("evidence_final_generation_owner_invalid")
    payload = dict(observation.observation_json or {})
    state = dict(payload.get("state") or {})
    state["events"] = [
        *list(state.get("events") or []),
        context_event(
            "final_generation",
            generation_evidence_view_hash=generation_view_hash,
            context_plan_hash=context_plan_hash,
        ),
    ]
    base = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "state",
            "source_text_persisted",
            "directory_text_persisted",
            "generation_evidence_view",
            "audit_hash",
        }
    }
    _persist_observation(
        observation,
        base=base,
        state=_seal_state(state),
        verdict="finalized",
        view=dict(payload.get("generation_evidence_view") or {}),
    )
    db.flush()


async def run_evidence_read_loop(
    db,
    *,
    agent_graph,
    run,
    plan,
    package,
    trace,
    admission,
    evidence: AnswerEvidenceManifest,
    model,
    remaining_seconds: Callable[[], float],
    per_call_timeout_seconds: float,
    max_tokens: int,
) -> EvidenceLoopResult:
    """Run or resume one continuous Mid-semantic evidence tool session."""

    evidence.verify_integrity()
    with qa_stage("evidence_directory", item_count=len(evidence.sources)):
        directory = build_mid_directory(
            db,
            plan=plan,
            package=package,
            trace=trace,
            evidence=evidence,
        )
    all_sources = tuple(source["source_handle"] for source in evidence.sources)
    deterministic_direct = (
        len(directory.mid_handles) <= 1
        or int(package.token_count or 0) <= DIRECT_CONTEXT_TOKEN_LIMIT
    )
    base = {
        "protocol_version": LOOP_PROTOCOL,
        "run_id": run.id,
        "accepted_plan_hash": plan.identity,
        "context_package_id": package.id,
        "retrieval_trace_id": trace.id,
        "source_integrity_admission_hash": admission.audit["audit_hash"],
        "admitted_evidence_manifest_hash": evidence.manifest_hash,
        "mid_directory_hash": directory.identity,
        "active_mid_state_hash": directory.active_state_hash,
        "context_tree_hash": directory.tree.audit()["tree_hash"],
        "mid_count": len(directory.mid_handles),
        "source_count": len(all_sources),
        "mandatory_source_count": len(directory.mandatory_sources),
        "requirement_count": len(plan.task.requirements),
    }
    rows = list(
        db.scalars(
            select(AgentObservation).where(
                AgentObservation.run_id == run.id,
                AgentObservation.observation_type == OBSERVATION_TYPE,
            )
        )
    )
    if len(rows) > 1:
        raise EvidenceDecisionStateError("evidence_loop_observation_not_unique")
    if rows:
        observation = rows[0]
        state = _validate_observation(
            observation,
            base=base,
            directory=directory,
            all_sources=all_sources,
        )
    else:
        state = _seal_state(
            {
                "phase": "reading",
                "remaining_mid_handles": list(directory.mid_handles),
                "read_mid_handles": [],
                "read_source_handles": [],
                "committed_source_handles": [],
                "decision_call_count": 0,
                "read_action_count": 0,
                "error_count": 0,
                "deterministic_direct": deterministic_direct,
                "events": [context_event("user_task", task_hash=control_hash(plan.task.model_dump(mode="json")))],
                "context_plans": [],
                "transitions": [],
            }
        )
        observation = AgentObservation(
            run_id=run.id,
            observation_type=OBSERVATION_TYPE,
            verdict="prepared",
            observation_json=_observation_payload(base, state),
            evidence_chunk_ids_json=[source["chunk_id"] for source in evidence.sources],
        )
        db.add(observation)
        db.flush()
        run.current_node = "evidence_directory"
        agent_graph.trace(
            db,
            run.id,
            "evidence_directory_ready",
            output_summary=f"prepared {len(directory.mid_handles)} semantic nodes",
            scores={
                "protocol_version": LOOP_PROTOCOL,
                "source_count": len(all_sources),
                "document_count": len(
                    {source["package_item"].get("document_id") for source in evidence.sources}
                ),
                "requirement_count": len(plan.task.requirements),
                "remaining_count": len(directory.mid_handles),
                "decision_call_count": 0,
                "mid_count": len(directory.mid_handles),
                "mandatory_source_count": len(directory.mandatory_sources),
                "deterministic_direct": deterministic_direct,
            },
        )
        db.commit()

    if state["phase"] == "finalized":
        replayed_evidence, _ = _validate_identity(
            db,
            run=run,
            package=package,
            trace=trace,
            admission=admission,
            plan=plan,
            evidence_hash=evidence.manifest_hash,
            directory_hash=directory.identity,
        )
        view = dict((observation.observation_json or {}).get("generation_evidence_view") or {})
        generated, _expected = build_generation_evidence_view(
            replayed_evidence,
            state["committed_source_handles"],
            coverage={},
            source_integrity_admission_hash=admission.audit["audit_hash"],
        )
        replay_generation_evidence_view(replayed_evidence, generated, view)
        return EvidenceLoopResult(
            replayed_evidence,
            generated,
            view,
            observation,
            _selected_contexts(package, replayed_evidence, state["committed_source_handles"]),
        )

    if state["phase"] in {"failed", "cancelled"}:
        raise EvidenceDecisionStateError("evidence_loop_terminal")

    if state.get("deterministic_direct"):
        committed = list(all_sources)
        with qa_stage("evidence_freeze", item_count=len(committed)):
            generated, view = build_generation_evidence_view(
                evidence,
                committed,
                coverage={},
                source_integrity_admission_hash=admission.audit["audit_hash"],
            )
        state.update(
            {
                "phase": "finalized",
                "remaining_mid_handles": [],
                "read_mid_handles": list(directory.mid_handles),
                "read_source_handles": list(all_sources),
                "committed_source_handles": committed,
                "transitions": [
                    *state["transitions"],
                    {"action": "deterministic_direct", "selected_count": len(committed)},
                ],
            }
        )
        state["events"] = [
            *state["events"],
            context_event("evidence_committed", source_handles=committed, deterministic=True),
        ]
        _persist_observation(
            observation,
            base=base,
            state=_seal_state(state),
            verdict="finalized",
            view=view,
        )
        agent_graph.trace(
            db,
            run.id,
            "evidence_finalized",
            output_summary=f"froze {len(committed)} sources without a model decision",
            scores={
                "protocol_version": LOOP_PROTOCOL,
                "source_count": len(all_sources),
                "selected_count": len(committed),
                "remaining_count": 0,
                "decision_call_count": 0,
                "read_action_count": 0,
                "mid_count": len(directory.mid_handles),
                "mandatory_source_count": len(directory.mandatory_sources),
                "deterministic_direct": True,
            },
        )
        db.commit()
        return EvidenceLoopResult(
            evidence,
            generated,
            view,
            observation,
            _selected_contexts(package, evidence, committed),
        )

    max_calls = len(directory.mid_handles) + 3
    while True:
        state = _validate_observation(
            observation,
            base=base,
            directory=directory,
            all_sources=all_sources,
        )
        if int(state["decision_call_count"]) >= max_calls:
            raise EvidenceDecisionStateError("evidence_tool_call_budget_exhausted")
        messages, units = _messages_and_units(
            plan=plan,
            directory=directory,
            evidence=evidence,
            events=list(state["events"]),
        )
        context_plan = plan_context(
            units,
            input_token_budget=max(int(package.token_budget or 0) + 8192, 8192),
            reserved_output_tokens=max_tokens,
            stable_prefix=evidence_tool_system_prompt(),
        )
        kept = set(context_plan.kept_keys)
        messages = [message for index, message in enumerate(messages) if f"message:{index}" in kept]
        state["decision_call_count"] = int(state["decision_call_count"]) + 1
        state["context_plans"] = [*state["context_plans"], context_plan.audit]
        if context_plan.audit["compression_applied"]:
            state["events"] = [
                *state["events"],
                context_event(
                    "context_compacted",
                    context_plan_hash=context_plan.audit["plan_hash"],
                    unit_count=sum(
                        item["reason"] == "compressed_control_history"
                        for item in context_plan.audit["removed_units"]
                    ),
                ),
            ]
        if context_plan.audit["truncation_applied"]:
            state["events"] = [
                *state["events"],
                context_event(
                    "context_evicted",
                    context_plan_hash=context_plan.audit["plan_hash"],
                    unit_count=sum(
                        item["reason"] == "evicted_on_budget"
                        for item in context_plan.audit["removed_units"]
                    ),
                ),
            ]
        _persist_observation(
            observation,
            base=base,
            state=_seal_state(state),
            verdict="reading",
        )
        run.current_node = "evidence_read"
        db.commit()
        timeout_seconds = min(per_call_timeout_seconds, remaining_seconds())
        if timeout_seconds <= 0:
            raise TimeoutError("agent_deadline_exhausted")
        model_started = time.monotonic()
        try:
            call, model_audit = await model.call_evidence_tool(
                messages=messages,
                compatibility_packet={
                    "phase": "tool_session",
                    "messages": messages,
                    "remaining_mid_handles": list(state["remaining_mid_handles"]),
                    "read_source_handles": list(state["read_source_handles"]),
                },
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
            )
        except AnswerReviewModelError as exc:
            if exc.code != "schema_invalid" or int(state.get("error_count") or 0) >= 2:
                raise
            evidence, directory = _validate_identity(
                db,
                run=run,
                package=package,
                trace=trace,
                admission=admission,
                plan=plan,
                evidence_hash=evidence.manifest_hash,
                directory_hash=directory.identity,
            )
            observation = db.get(AgentObservation, observation.id)
            state = dict((observation.observation_json or {})["state"])
            _append_error_pair(
                state,
                tool="invalid",
                arguments={},
                error="invalid_tool_call",
                field="tool_call",
            )
            _persist_observation(
                observation,
                base=base,
                state=_seal_state(state),
                verdict="reading",
            )
            db.commit()
            continue
        model_duration_ms = round((time.monotonic() - model_started) * 1000, 3)
        state["context_plans"][-1] = apply_provider_usage(
            state["context_plans"][-1],
            model_audit.get("provider_call"),
        )

        evidence, directory = _validate_identity(
            db,
            run=run,
            package=package,
            trace=trace,
            admission=admission,
            plan=plan,
            evidence_hash=evidence.manifest_hash,
            directory_hash=directory.identity,
        )
        observation = db.get(AgentObservation, observation.id)
        persisted_state = dict((observation.observation_json or {})["state"])
        persisted_state["context_plans"] = state["context_plans"]
        state = persisted_state
        if call.tool == "evidence.read":
            invalid = None
            if any(not _valid_handle(handle, "mid_") for handle in call.mid_handles):
                invalid = ("mid_handle_invalid", "arguments.mid_handles")
            elif any(handle not in state["remaining_mid_handles"] for handle in call.mid_handles):
                invalid = ("mid_handle_not_unread", "arguments.mid_handles")
            if invalid is not None:
                _append_error_pair(
                    state,
                    tool=call.tool,
                    arguments=call.argument_payload,
                    error=invalid[0],
                    field=invalid[1],
                )
                _persist_observation(
                    observation,
                    base=base,
                    state=_seal_state(state),
                    verdict="reading",
                )
                db.commit()
                continue
            with qa_stage("evidence_read", item_count=len(call.mid_handles)):
                before_sources = set(state["read_source_handles"])
                groups: list[dict[str, Any]] = []
                for mid_handle in call.mid_handles:
                    source_handles = [
                        handle
                        for handle in directory.mid_to_sources[mid_handle]
                        if handle not in before_sources
                    ]
                    before_sources.update(source_handles)
                    groups.append(
                        {"mid_handle": mid_handle, "source_handles": source_handles}
                    )
            remaining = [
                handle for handle in state["remaining_mid_handles"] if handle not in call.mid_handles
            ]
            state["events"].extend(
                [
                    context_event("tool_call", tool=call.tool, arguments=call.argument_payload),
                    context_event(
                        "tool_result_ref",
                        tool=call.tool,
                        status="ok",
                        groups=groups,
                        remaining_mid_handles=remaining,
                        source_handles=[
                            handle for group in groups for handle in group["source_handles"]
                        ],
                    ),
                ]
            )
            state["remaining_mid_handles"] = remaining
            state["read_mid_handles"] = [
                handle for handle in directory.mid_handles if handle not in remaining
            ]
            state["read_source_handles"] = [
                handle for handle in all_sources if handle in before_sources
            ]
            state["read_action_count"] = int(state["read_action_count"]) + 1
            state["transitions"] = [
                *state["transitions"],
                {
                    "action": "evidence.read",
                    "mid_count": len(call.mid_handles),
                    "source_count": sum(len(group["source_handles"]) for group in groups),
                    "remaining_mid_count": len(remaining),
                    "model_duration_ms": model_duration_ms,
                },
            ]
            _persist_observation(
                observation,
                base=base,
                state=_seal_state(state),
                verdict="reading",
            )
            agent_graph.trace(
                db,
                run.id,
                "evidence_read",
                output_summary=(
                    f"read {len(call.mid_handles)} semantic nodes and "
                    f"{sum(len(group['source_handles']) for group in groups)} sources"
                ),
                scores={
                    "protocol_version": LOOP_PROTOCOL,
                    "read_count": sum(len(group["source_handles"]) for group in groups),
                    "selected_count": len(state["read_source_handles"]),
                    "remaining_count": len(remaining),
                    "decision_call_count": state["decision_call_count"],
                    "read_action_count": state["read_action_count"],
                    "model_duration_ms": round(model_duration_ms),
                    "mid_count": len(directory.mid_handles),
                    "mandatory_source_count": len(directory.mandatory_sources),
                    "estimated_input_tokens": state["context_plans"][-1]["estimated_input_tokens"],
                    "input_token_count": state["context_plans"][-1].get("input_token_count"),
                    "compression_applied": state["context_plans"][-1]["compression_applied"],
                    "truncation_applied": state["context_plans"][-1]["truncation_applied"],
                    "deterministic_direct": False,
                },
                duration_ms=round(model_duration_ms),
            )
            db.commit()
            continue

        invalid = None
        if any(not _valid_handle(handle, "src_") for handle in call.source_handles):
            invalid = ("source_handle_invalid", "arguments.source_handles")
        elif any(handle not in state["read_source_handles"] for handle in call.source_handles):
            invalid = ("source_handle_not_read", "arguments.source_handles")
        if invalid is not None:
            _append_error_pair(
                state,
                tool=call.tool,
                arguments=call.argument_payload,
                error=invalid[0],
                field=invalid[1],
            )
            _persist_observation(
                observation,
                base=base,
                state=_seal_state(state),
                verdict="reading",
            )
            db.commit()
            continue
        selected_set = set(call.source_handles) | set(directory.mandatory_sources)
        committed = [handle for handle in all_sources if handle in selected_set]
        with qa_stage("evidence_freeze", item_count=len(committed)):
            generated, view = build_generation_evidence_view(
                evidence,
                committed,
                coverage={},
                source_integrity_admission_hash=admission.audit["audit_hash"],
            )
        state["phase"] = "finalized"
        state["committed_source_handles"] = committed
        state["events"] = [
            *state["events"],
            context_event("evidence_committed", source_handles=committed, deterministic=False),
        ]
        state["transitions"] = [
            *state["transitions"],
            {
                "action": "evidence.commit",
                "selected_count": len(committed),
                "model_duration_ms": model_duration_ms,
            },
        ]
        _persist_observation(
            observation,
            base=base,
            state=_seal_state(state),
            verdict="finalized",
            view=view,
        )
        agent_graph.trace(
            db,
            run.id,
            "evidence_finalized",
            output_summary=f"froze {len(committed)} generation sources",
            scores={
                "protocol_version": LOOP_PROTOCOL,
                "source_count": len(all_sources),
                "selected_count": len(committed),
                "remaining_count": len(state["remaining_mid_handles"]),
                "decision_call_count": state["decision_call_count"],
                "read_action_count": state["read_action_count"],
                "model_duration_ms": round(model_duration_ms),
                "mid_count": len(directory.mid_handles),
                "mandatory_source_count": len(directory.mandatory_sources),
                "estimated_input_tokens": state["context_plans"][-1]["estimated_input_tokens"],
                "input_token_count": state["context_plans"][-1].get("input_token_count"),
                "compression_applied": state["context_plans"][-1]["compression_applied"],
                "truncation_applied": state["context_plans"][-1]["truncation_applied"],
                "deterministic_direct": False,
            },
            duration_ms=round(model_duration_ms),
        )
        db.commit()
        return EvidenceLoopResult(
            evidence,
            generated,
            view,
            observation,
            _selected_contexts(package, evidence, committed),
        )
