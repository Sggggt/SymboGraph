from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.models import (
    AgentPlan,
    AgentRun,
    AnswerSession,
    CitationVerification,
    ContextPackage,
    QASession,
    RetrievalTrace,
)
from app.schemas import (
    ChatMessage,
    Citation,
    ConversationStateUpdate,
    SearchFilters,
)
from app.services.chinese_text import estimate_tokens


CONVERSATION_STATE_PROTOCOL_VERSION = "conversation_state_v1"
CONVERSATION_STATE_SCOPE_PROTOCOL_VERSION = "conversation_state_scope_v1"
CONVERSATION_HISTORY_MERGE_PROTOCOL_VERSION = "server_history_overlap_merge_v1"
CONVERSATION_REFERENCE_PROTOCOL_VERSION = "answer_context_citation_reference_v1"
CONVERSATION_PROMPT_HISTORY_PROTOCOL_VERSION = "conversation_prompt_projection_v1"
CONVERSATION_STATE_UNINITIALIZED_HASH = (
    "959a95f2683e19efd10a1296ed82dac31d14c77a74caac4f5e0c12cfd062bd5e"
)

CLIENT_HISTORY_MAX_TURNS = 32
CLIENT_HISTORY_MAX_TOKENS = 8_192
PROMPT_HISTORY_MAX_TURNS = 24
PROMPT_HISTORY_MAX_TOKENS = 4_096


class ConversationStateError(RuntimeError):
    """Base class for conversation-state admission failures."""


class ConversationStateNotFoundError(ConversationStateError, LookupError):
    pass


class ConversationStateConflictError(ConversationStateError):
    pass


class ConversationStateIntegrityError(ConversationStateError):
    pass


@dataclass(frozen=True)
class ConversationStateSnapshot:
    qa_session_id: str | None
    knowledge_base_id: str
    state_hash: str
    scope_hash: str
    revision: int
    active_user_constraints: dict[str, Any]
    task_state: dict[str, Any]
    history_references: list[dict[str, Any]]
    transcript_message_count: int
    prompt_history: list[dict[str, str]]
    prompt_history_audit: dict[str, Any]

    def public_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": CONVERSATION_STATE_PROTOCOL_VERSION,
            "scope_protocol_version": CONVERSATION_STATE_SCOPE_PROTOCOL_VERSION,
            "qa_session_id": self.qa_session_id,
            "knowledge_base_id": self.knowledge_base_id,
            "revision": self.revision,
            "state_hash": self.state_hash,
            "scope_hash": self.scope_hash,
            "active_user_constraints": self.active_user_constraints,
            "task_state": self.task_state,
            "history_references": self.history_references,
            "transcript_message_count": self.transcript_message_count,
            "prompt_history_audit": self.prompt_history_audit,
            "evidence_authority": False,
            "gray_zone_decision_authority": False,
        }

    def retrieval_audit(self) -> dict[str, Any]:
        return {
            "protocol_version": CONVERSATION_STATE_PROTOCOL_VERSION,
            "scope_protocol_version": CONVERSATION_STATE_SCOPE_PROTOCOL_VERSION,
            "qa_session_id": self.qa_session_id,
            "knowledge_base_id": self.knowledge_base_id,
            "revision": self.revision,
            "state_hash": self.state_hash,
            "scope_hash": self.scope_hash,
            "active_user_constraints_hash": _stable_hash(
                self.active_user_constraints
            ),
            "task_state_hash": _stable_hash(self.task_state),
            "history_reference_count": len(self.history_references),
            "history_reference_hash": _stable_hash(self.history_references),
            "transcript_message_count": self.transcript_message_count,
            "prompt_history_audit": self.prompt_history_audit,
            "conversation_text_is_evidence": False,
            "gray_zone_decision_authority": False,
            "gray_zone_model_call_count": 0,
        }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any, *, field_name: str) -> str:
    digest = str(value or "")
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ConversationStateIntegrityError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return digest


def canonical_uuid(value: Any, *, field_name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConversationStateIntegrityError(
            f"{field_name} must be a canonical UUID"
        ) from exc


def _canonical_string_list(values: Iterable[Any]) -> list[str]:
    if not isinstance(values, list):
        raise ConversationStateIntegrityError("expected a list of strings")
    if any(not isinstance(value, str) for value in values):
        raise ConversationStateIntegrityError("expected a list of strings")
    return sorted(
        {
            str(value).strip()
            for value in values
            if str(value).strip()
        }
    )


def _canonical_uuid_list(values: Iterable[Any], *, field_name: str) -> list[str]:
    if not isinstance(values, list):
        raise ConversationStateIntegrityError(f"{field_name} must be a list")
    if any(not isinstance(value, str) for value in values):
        raise ConversationStateIntegrityError(
            f"{field_name} must contain UUID strings"
        )
    return sorted(
        {
            canonical_uuid(value, field_name=field_name)
            for value in values
        }
    )


def canonical_active_user_constraints(value: Any) -> dict[str, Any]:
    if value is None:
        payload: dict[str, Any] = {}
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        raise ConversationStateIntegrityError(
            "active_user_constraints must be an object"
        )
    unexpected = set(payload) - {"instructions", "retrieval_filters"}
    if unexpected:
        raise ConversationStateIntegrityError(
            "active_user_constraints contains unsupported fields"
        )
    raw_filters = payload.get("retrieval_filters", {})
    if not isinstance(raw_filters, dict):
        raise ConversationStateIntegrityError(
            "active_user_constraints.retrieval_filters must be an object"
        )
    filters = dict(raw_filters)
    filter_fields = {
        "document_ids",
        "source_paths",
        "source_type",
        "partition",
        "tags",
        "page_range",
        "content_kinds",
        "chunk_version",
    }
    if set(filters) - filter_fields:
        raise ConversationStateIntegrityError(
            "active_user_constraints.retrieval_filters contains unsupported fields"
        )
    instructions = payload.get("instructions", [])
    if not isinstance(instructions, list):
        raise ConversationStateIntegrityError(
            "active_user_constraints.instructions must be a list"
        )
    if len(instructions) > 16:
        raise ConversationStateIntegrityError(
            "active_user_constraints.instructions exceeds 16 entries"
        )
    canonical_instructions = _canonical_string_list(instructions)
    if any(len(item) > 512 for item in canonical_instructions):
        raise ConversationStateIntegrityError(
            "active_user_constraints instruction exceeds 512 characters"
        )
    page_range = filters.get("page_range")
    if page_range is not None:
        if (
            not isinstance(page_range, (list, tuple))
            or len(page_range) != 2
            or any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int))
                for value in page_range
            )
        ):
            raise ConversationStateIntegrityError(
                "active conversation page_range must contain two integer bounds"
            )
        page_range = list(page_range)
        if (
            page_range[0] is not None
            and page_range[1] is not None
            and page_range[0] > page_range[1]
        ):
            raise ConversationStateIntegrityError(
                "active conversation page_range lower bound exceeds upper bound"
            )
    for field in ("source_type", "partition"):
        if filters.get(field) is not None and not isinstance(filters[field], str):
            raise ConversationStateIntegrityError(
                f"active conversation {field} must be a string"
            )
    chunk_version = filters.get("chunk_version")
    if chunk_version is not None and (
        isinstance(chunk_version, bool)
        or not isinstance(chunk_version, int)
        or chunk_version < 1
    ):
        raise ConversationStateIntegrityError(
            "active conversation chunk_version must be a positive integer"
        )
    canonical_filters = {
        "document_ids": _canonical_uuid_list(
            filters.get("document_ids", []),
            field_name="active_user_constraints.retrieval_filters.document_ids",
        ),
        "source_paths": _canonical_string_list(filters.get("source_paths", [])),
        "source_type": str(filters["source_type"]).strip()
        if filters.get("source_type") is not None
        else None,
        "partition": str(filters["partition"]).strip()
        if filters.get("partition") is not None
        else None,
        "tags": _canonical_string_list(filters.get("tags", [])),
        "page_range": page_range,
        "content_kinds": _canonical_string_list(filters.get("content_kinds", [])),
        "chunk_version": chunk_version,
    }
    return {
        "instructions": canonical_instructions,
        "retrieval_filters": canonical_filters,
    }


def canonical_task_state(value: Any) -> dict[str, Any]:
    if value is None:
        payload: dict[str, Any] = {}
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        raise ConversationStateIntegrityError("task_state must be an object")
    if set(payload) - {"status", "objective", "current_step"}:
        raise ConversationStateIntegrityError(
            "task_state contains unsupported fields"
        )
    status = payload.get("status", "active")
    if not isinstance(status, str):
        raise ConversationStateIntegrityError("task_state status must be text")
    if status not in {"active", "waiting_user", "completed", "cancelled"}:
        raise ConversationStateIntegrityError("task_state status is unsupported")
    objective = payload.get("objective")
    current_step = payload.get("current_step")
    if objective is not None and not isinstance(objective, str):
        raise ConversationStateIntegrityError("task_state objective must be text")
    if current_step is not None and not isinstance(current_step, str):
        raise ConversationStateIntegrityError("task_state current_step must be text")
    if isinstance(objective, str) and len(objective.strip()) > 2_000:
        raise ConversationStateIntegrityError("task_state objective is too long")
    if isinstance(current_step, str) and len(current_step.strip()) > 1_000:
        raise ConversationStateIntegrityError("task_state current_step is too long")
    return {
        "status": status,
        "objective": str(objective).strip() if objective is not None else None,
        "current_step": str(current_step).strip()
        if current_step is not None
        else None,
    }


def _canonical_citation_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationStateIntegrityError(
            "persisted transcript citations must be objects"
        )
    # Citations are presentation copies. Authoritative evidence is resolved
    # through history_references and the referenced PostgreSQL rows.
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ConversationStateIntegrityError(
            "persisted transcript citations must contain strict JSON values"
        ) from exc


def canonical_transcript(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConversationStateIntegrityError("persisted transcript must be a list")
    messages: list[dict[str, Any]] = []
    previous_role: str | None = None
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConversationStateIntegrityError(
                f"persisted transcript item {index} must be an object"
            )
        if set(item) - {
            "role",
            "content",
            "run_id",
            "route",
            "retrieval_trace_id",
            "citations",
            "source",
        }:
            raise ConversationStateIntegrityError(
                f"persisted transcript item {index} contains unsupported fields"
            )
        role = item.get("role")
        if not isinstance(role, str):
            raise ConversationStateIntegrityError(
                f"persisted transcript item {index} role must be text"
            )
        if role not in {"user", "assistant"}:
            raise ConversationStateIntegrityError(
                f"persisted transcript item {index} has an invalid role"
            )
        raw_content = item.get("content")
        if not isinstance(raw_content, str):
            raise ConversationStateIntegrityError(
                f"persisted transcript item {index} content must be text"
            )
        content = raw_content.strip()
        if not content:
            raise ConversationStateIntegrityError(
                f"persisted transcript item {index} has empty content"
            )
        if previous_role == role:
            raise ConversationStateIntegrityError(
                "persisted transcript roles must alternate"
            )
        canonical: dict[str, Any] = {"role": role, "content": content}
        if item.get("run_id") is not None:
            if not isinstance(item["run_id"], str):
                raise ConversationStateIntegrityError(
                    f"transcript[{index}].run_id must be a UUID string"
                )
            canonical["run_id"] = canonical_uuid(
                item["run_id"], field_name=f"transcript[{index}].run_id"
            )
        if item.get("route") is not None:
            if not isinstance(item["route"], str):
                raise ConversationStateIntegrityError(
                    f"transcript[{index}].route must be text"
                )
            canonical["route"] = str(item["route"])
        if item.get("retrieval_trace_id") is not None:
            if not isinstance(item["retrieval_trace_id"], str):
                raise ConversationStateIntegrityError(
                    f"transcript[{index}].retrieval_trace_id must be a UUID string"
                )
            canonical["retrieval_trace_id"] = canonical_uuid(
                item["retrieval_trace_id"],
                field_name=f"transcript[{index}].retrieval_trace_id",
            )
        if item.get("citations") is not None:
            if not isinstance(item["citations"], list):
                raise ConversationStateIntegrityError(
                    f"transcript[{index}].citations must be a list"
                )
            canonical["citations"] = [
                _canonical_citation_payload(citation)
                for citation in item["citations"]
            ]
        if item.get("source") is not None:
            if item["source"] != "client_history":
                raise ConversationStateIntegrityError(
                    f"transcript[{index}].source is unsupported"
                )
            canonical["source"] = "client_history"
        messages.append(canonical)
        previous_role = role
    if messages and messages[0]["role"] != "user":
        raise ConversationStateIntegrityError(
            "persisted transcript must start with a user message"
        )
    if messages and messages[-1]["role"] != "assistant":
        raise ConversationStateIntegrityError(
            "persisted transcript must contain completed user/assistant turns"
        )
    return messages


def canonical_history_references(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConversationStateIntegrityError(
            "history_references must be a list"
        )
    references: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConversationStateIntegrityError(
                f"history reference {index} must be an object"
            )
        allowed_fields = {
            "protocol_version",
            "turn_index",
            "run_id",
            "answer_session_id",
            "context_package_id",
            "retrieval_trace_id",
            "citation_verification_ids",
        }
        if set(item) - allowed_fields:
            raise ConversationStateIntegrityError(
                f"history reference {index} contains unsupported fields"
            )
        if item.get("protocol_version") != CONVERSATION_REFERENCE_PROTOCOL_VERSION:
            raise ConversationStateIntegrityError(
                f"history reference {index} uses an unsupported protocol"
            )
        turn_index = item.get("turn_index")
        if (
            isinstance(turn_index, bool)
            or not isinstance(turn_index, int)
            or turn_index != index
        ):
            raise ConversationStateIntegrityError(
                "history references must use contiguous stored order"
            )
        raw_citation_ids = item.get("citation_verification_ids", [])
        if not isinstance(raw_citation_ids, list) or any(
            not isinstance(citation_id, str)
            for citation_id in raw_citation_ids
        ):
            raise ConversationStateIntegrityError(
                f"history_references[{index}].citation_verification_ids "
                "must contain UUID strings"
            )
        for id_field in (
            "run_id",
            "answer_session_id",
            "context_package_id",
            "retrieval_trace_id",
        ):
            if not isinstance(item.get(id_field), str):
                raise ConversationStateIntegrityError(
                    f"history_references[{index}].{id_field} must be a UUID string"
                )
        citation_ids = sorted(
            {
                canonical_uuid(
                    citation_id,
                    field_name=f"history_references[{index}].citation_verification_ids",
                )
                for citation_id in raw_citation_ids
            }
        )
        references.append(
            {
                "protocol_version": CONVERSATION_REFERENCE_PROTOCOL_VERSION,
                "turn_index": index,
                "run_id": canonical_uuid(
                    item.get("run_id"),
                    field_name=f"history_references[{index}].run_id",
                ),
                "answer_session_id": canonical_uuid(
                    item.get("answer_session_id"),
                    field_name=f"history_references[{index}].answer_session_id",
                ),
                "context_package_id": canonical_uuid(
                    item.get("context_package_id"),
                    field_name=f"history_references[{index}].context_package_id",
                ),
                "retrieval_trace_id": canonical_uuid(
                    item.get("retrieval_trace_id"),
                    field_name=f"history_references[{index}].retrieval_trace_id",
                ),
                "citation_verification_ids": citation_ids,
            }
        )
    return references


def conversation_state_fact_payload(
    *,
    qa_session_id: str,
    knowledge_base_id: str,
    transcript: Any,
    active_user_constraints: Any,
    task_state: Any,
    history_references: Any,
) -> dict[str, Any]:
    return {
        "protocol_version": CONVERSATION_STATE_PROTOCOL_VERSION,
        "qa_session_id": canonical_uuid(
            qa_session_id, field_name="qa_session_id"
        ),
        "knowledge_base_id": canonical_uuid(
            knowledge_base_id, field_name="knowledge_base_id"
        ),
        "transcript": canonical_transcript(transcript),
        "active_user_constraints": canonical_active_user_constraints(
            active_user_constraints
        ),
        "task_state": canonical_task_state(task_state),
        "history_references": canonical_history_references(history_references),
    }


def conversation_state_hash_for_session(session: QASession) -> str:
    return _stable_hash(
        conversation_state_fact_payload(
            qa_session_id=session.id,
            knowledge_base_id=session.knowledge_base_id,
            transcript=session.transcript,
            active_user_constraints=session.active_user_constraints_json,
            task_state=session.task_state_json,
            history_references=session.history_references_json,
        )
    )


def conversation_state_scope_hash(
    *,
    knowledge_base_id: str,
    qa_session_id: str | None,
    state_hash: str,
) -> str:
    return _stable_hash(
        {
            "protocol_version": CONVERSATION_STATE_SCOPE_PROTOCOL_VERSION,
            "knowledge_base_id": canonical_uuid(
                knowledge_base_id, field_name="knowledge_base_id"
            ),
            "qa_session_id": canonical_uuid(
                qa_session_id, field_name="qa_session_id"
            )
            if qa_session_id is not None
            else "anonymous",
            "conversation_state_hash": _canonical_sha256(
                state_hash, field_name="conversation_state_hash"
            ),
        }
    )


def anonymous_conversation_state_snapshot(
    knowledge_base_id: str,
) -> ConversationStateSnapshot:
    constraints = canonical_active_user_constraints({})
    task_state = canonical_task_state({})
    state_hash = _stable_hash(
        {
            "protocol_version": CONVERSATION_STATE_PROTOCOL_VERSION,
            "knowledge_base_id": canonical_uuid(
                knowledge_base_id, field_name="knowledge_base_id"
            ),
            "qa_session_id": "anonymous",
            "transcript": [],
            "active_user_constraints": constraints,
            "task_state": task_state,
            "history_references": [],
        }
    )
    prompt_audit = {
        "protocol_version": CONVERSATION_PROMPT_HISTORY_PROTOCOL_VERSION,
        "stored_turn_count": 0,
        "selected_turn_count": 0,
        "stored_completed_turn_count": 0,
        "selected_completed_turn_count": 0,
        "omitted_turn_count": 0,
        "selected_token_count": 0,
        "turn_limit": PROMPT_HISTORY_MAX_TURNS,
        "token_limit": PROMPT_HISTORY_MAX_TOKENS,
        "transcript_truncated": False,
        "persisted_transcript_retained_in_full": True,
    }
    return ConversationStateSnapshot(
        qa_session_id=None,
        knowledge_base_id=canonical_uuid(
            knowledge_base_id, field_name="knowledge_base_id"
        ),
        state_hash=state_hash,
        scope_hash=conversation_state_scope_hash(
            knowledge_base_id=knowledge_base_id,
            qa_session_id=None,
            state_hash=state_hash,
        ),
        revision=0,
        active_user_constraints=constraints,
        task_state=task_state,
        history_references=[],
        transcript_message_count=0,
        prompt_history=[],
        prompt_history_audit=prompt_audit,
    )


def _validate_reference_provenance(
    db: Session,
    session: QASession,
    references: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> None:
    seen_answer_ids: set[str] = set()
    transcript_pairs = [
        (transcript[index], transcript[index + 1])
        for index in range(0, len(transcript), 2)
        if index + 1 < len(transcript)
    ]
    for reference in references:
        answer_id = reference["answer_session_id"]
        if answer_id in seen_answer_ids:
            raise ConversationStateIntegrityError(
                "history references may not reuse an answer session"
            )
        seen_answer_ids.add(answer_id)
        run = db.get(AgentRun, reference["run_id"])
        answer = db.get(AnswerSession, answer_id)
        package = db.get(ContextPackage, reference["context_package_id"])
        trace = db.get(RetrievalTrace, reference["retrieval_trace_id"])
        if run is None or answer is None or package is None or trace is None:
            raise ConversationStateIntegrityError(
                "conversation history reference targets are incomplete"
            )
        if run.session_id != session.id or run.knowledge_base_id != session.knowledge_base_id:
            raise ConversationStateIntegrityError(
                "conversation run provenance does not match session and knowledge base"
            )
        matching_pairs = [
            pair
            for pair in transcript_pairs
            if pair[0].get("run_id") == reference["run_id"]
            and pair[1].get("run_id") == reference["run_id"]
        ]
        if len(matching_pairs) != 1:
            raise ConversationStateIntegrityError(
                "conversation run reference does not map to exactly one transcript turn"
            )
        user_message, assistant_message = matching_pairs[0]
        if answer.qa_session_id != session.id or answer.knowledge_base_id != session.knowledge_base_id:
            raise ConversationStateIntegrityError(
                "answer session provenance does not match conversation session"
            )
        if (
            str(run.question).strip() != user_message["content"]
            or str(answer.question).strip() != user_message["content"]
            or str(answer.answer).strip() != assistant_message["content"]
        ):
            raise ConversationStateIntegrityError(
                "conversation transcript text does not match the referenced run and answer"
            )
        if (
            answer.context_package_id != package.id
            or answer.retrieval_trace_id != trace.id
            or package.retrieval_trace_id != trace.id
            or package.knowledge_base_id != session.knowledge_base_id
            or trace.knowledge_base_id != session.knowledge_base_id
        ):
            raise ConversationStateIntegrityError(
                "answer, context package, and retrieval trace provenance diverged"
            )
        trace_scope = str(trace.conversation_state_scope_hash or "")
        answer_scope = str((answer.model_json or {}).get("conversation_state_scope_hash") or "")
        package_scope = str(
            (package.diagnostics_json or {}).get("conversation_state_scope_hash")
            or ""
        )
        if (
            len(trace_scope) != 64
            or answer_scope != trace_scope
            or package_scope != trace_scope
        ):
            raise ConversationStateIntegrityError(
                "answer and context package conversation scopes do not match the retrieval trace"
            )
        expected_citation_ids = sorted(
            {
                canonical_uuid(value, field_name="answer_session.citation_ids")
                for value in (answer.citation_ids_json or [])
            }
        )
        if reference["citation_verification_ids"] != expected_citation_ids:
            raise ConversationStateIntegrityError(
                "conversation citation references do not match the answer session"
            )
        for citation_id in expected_citation_ids:
            verification = db.get(CitationVerification, citation_id)
            if verification is None:
                raise ConversationStateIntegrityError(
                    "conversation citation verification target is missing"
                )
            if (
                verification.knowledge_base_id != session.knowledge_base_id
                or verification.answer_session_id != answer.id
                or verification.context_package_id != package.id
                or verification.retrieval_trace_id != trace.id
            ):
                raise ConversationStateIntegrityError(
                    "citation verification provenance does not match conversation history"
                )


def verify_session_state(
    db: Session,
    session: QASession,
    *,
    validate_references: bool = True,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    str,
]:
    transcript = canonical_transcript(session.transcript)
    constraints = canonical_active_user_constraints(
        session.active_user_constraints_json
    )
    task_state = canonical_task_state(session.task_state_json)
    references = canonical_history_references(session.history_references_json)
    expected_hash = _stable_hash(
        conversation_state_fact_payload(
            qa_session_id=session.id,
            knowledge_base_id=session.knowledge_base_id,
            transcript=transcript,
            active_user_constraints=constraints,
            task_state=task_state,
            history_references=references,
        )
    )
    if (
        session.conversation_state_protocol_version
        != CONVERSATION_STATE_PROTOCOL_VERSION
    ):
        raise ConversationStateIntegrityError(
            "unsupported persisted conversation state protocol"
        )
    revision = session.conversation_state_revision
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ConversationStateIntegrityError(
            "persisted conversation state revision is invalid"
        )
    stored_hash = str(session.conversation_state_hash or "")
    if stored_hash == CONVERSATION_STATE_UNINITIALIZED_HASH:
        raise ConversationStateIntegrityError(
            "conversation state was not initialized through the canonical service"
        )
    elif stored_hash != expected_hash:
        raise ConversationStateIntegrityError(
            "persisted conversation state hash does not match canonical state"
        )
    if validate_references:
        _validate_reference_provenance(db, session, references, transcript)
    return transcript, constraints, task_state, references, expected_hash


def _history_messages(history: Iterable[ChatMessage | dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    previous_role: str | None = None
    token_count = 0
    for index, item in enumerate(history):
        payload = item.model_dump() if isinstance(item, ChatMessage) else dict(item)
        if set(payload) - {"role", "content"}:
            raise ConversationStateConflictError(
                f"client history item {index} contains unsupported fields"
            )
        role = str(payload.get("role") or "")
        content = str(payload.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            raise ConversationStateConflictError(
                f"client history item {index} violates role/content schema"
            )
        if previous_role == role:
            raise ConversationStateConflictError(
                "client history roles must alternate"
            )
        messages.append({"role": role, "content": content})
        previous_role = role
        token_count += estimate_tokens(content)
    if len(messages) > CLIENT_HISTORY_MAX_TURNS:
        raise ConversationStateConflictError(
            f"client history exceeds {CLIENT_HISTORY_MAX_TURNS} turns"
        )
    if token_count > CLIENT_HISTORY_MAX_TOKENS:
        raise ConversationStateConflictError(
            f"client history exceeds {CLIENT_HISTORY_MAX_TOKENS} estimated tokens"
        )
    if messages and messages[0]["role"] != "user":
        raise ConversationStateConflictError(
            "client history must start with a user message"
        )
    if messages and messages[-1]["role"] != "assistant":
        raise ConversationStateConflictError(
            "client history must end with an assistant message before a new question"
        )
    return messages


def _merge_client_history(
    transcript: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not history:
        return transcript, {
            "protocol_version": CONVERSATION_HISTORY_MERGE_PROTOCOL_VERSION,
            "client_turn_count": 0,
            "overlap_turn_count": 0,
            "imported_turn_count": 0,
        }
    stored_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in transcript
    ]
    if not stored_messages:
        imported = [
            {**item, "source": "client_history"}
            for item in history
        ]
        return canonical_transcript(imported), {
            "protocol_version": CONVERSATION_HISTORY_MERGE_PROTOCOL_VERSION,
            "client_turn_count": len(history),
            "overlap_turn_count": 0,
            "imported_turn_count": len(history),
        }
    overlap = 0
    maximum = min(len(stored_messages), len(history))
    for width in range(maximum, 0, -1):
        if stored_messages[-width:] == history[:width]:
            overlap = width
            break
    if overlap == 0:
        if history == stored_messages or (
            len(history) <= len(stored_messages)
            and history == stored_messages[-len(history) :]
        ):
            overlap = len(history)
        else:
            raise ConversationStateConflictError(
                "client history does not overlap the authoritative session transcript"
            )
    imported = [
        {**item, "source": "client_history"}
        for item in history[overlap:]
    ]
    merged = canonical_transcript([*transcript, *imported])
    return merged, {
        "protocol_version": CONVERSATION_HISTORY_MERGE_PROTOCOL_VERSION,
        "client_turn_count": len(history),
        "overlap_turn_count": overlap,
        "imported_turn_count": len(imported),
    }


def _clip_text_to_token_limit(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    if estimate_tokens(text) <= token_limit:
        return text
    low = 1
    high = len(text)
    best = ""
    while low <= high:
        width = (low + high) // 2
        candidate = text[-width:]
        if estimate_tokens(candidate) <= token_limit:
            best = candidate
            low = width + 1
        else:
            high = width - 1
    return best


def bounded_prompt_history(
    transcript: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    selected_pairs_reversed: list[list[dict[str, str]]] = []
    token_count = 0
    transcript_pairs = [
        (transcript[index], transcript[index + 1])
        for index in range(0, len(transcript), 2)
    ]
    for user_item, assistant_item in reversed(transcript_pairs):
        selected_message_count = len(selected_pairs_reversed) * 2
        if selected_message_count + 2 > PROMPT_HISTORY_MAX_TURNS:
            break
        remaining = PROMPT_HISTORY_MAX_TOKENS - token_count
        if remaining < 2:
            break
        user_tokens = estimate_tokens(user_item["content"])
        assistant_tokens = estimate_tokens(assistant_item["content"])
        if user_tokens + assistant_tokens <= remaining:
            selected_pairs_reversed.append(
                [
                    {"role": "user", "content": user_item["content"]},
                    {
                        "role": "assistant",
                        "content": assistant_item["content"],
                    },
                ]
            )
            token_count += user_tokens + assistant_tokens
            continue
        if selected_pairs_reversed:
            break
        user_limit = max(1, min(user_tokens, remaining // 2))
        assistant_limit = max(1, remaining - user_limit)
        user_content = _clip_text_to_token_limit(
            user_item["content"], user_limit
        )
        assistant_content = _clip_text_to_token_limit(
            assistant_item["content"], assistant_limit
        )
        if not user_content or not assistant_content:
            break
        selected_pairs_reversed.append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )
        token_count = estimate_tokens(user_content) + estimate_tokens(
            assistant_content
        )
        break
    selected = [
        item
        for pair in reversed(selected_pairs_reversed)
        for item in pair
    ]
    audit = {
        "protocol_version": CONVERSATION_PROMPT_HISTORY_PROTOCOL_VERSION,
        "stored_turn_count": len(transcript),
        "selected_turn_count": len(selected),
        "stored_completed_turn_count": len(transcript_pairs),
        "selected_completed_turn_count": len(selected) // 2,
        "omitted_turn_count": len(transcript) - len(selected),
        "selected_token_count": token_count,
        "turn_limit": PROMPT_HISTORY_MAX_TURNS,
        "token_limit": PROMPT_HISTORY_MAX_TOKENS,
        "transcript_truncated": len(selected) < len(transcript)
        or any(
            selected[index]["content"]
            != transcript[len(transcript) - len(selected) + index]["content"]
            for index in range(len(selected))
        ),
        "persisted_transcript_retained_in_full": True,
    }
    return selected, audit


def _snapshot_from_verified(
    session: QASession,
    *,
    transcript: list[dict[str, Any]],
    constraints: dict[str, Any],
    task_state: dict[str, Any],
    references: list[dict[str, Any]],
    state_hash: str,
) -> ConversationStateSnapshot:
    prompt_history, prompt_audit = bounded_prompt_history(transcript)
    return ConversationStateSnapshot(
        qa_session_id=canonical_uuid(session.id, field_name="qa_session_id"),
        knowledge_base_id=canonical_uuid(
            session.knowledge_base_id, field_name="knowledge_base_id"
        ),
        state_hash=state_hash,
        scope_hash=conversation_state_scope_hash(
            knowledge_base_id=session.knowledge_base_id,
            qa_session_id=session.id,
            state_hash=state_hash,
        ),
        revision=int(session.conversation_state_revision or 0),
        active_user_constraints=constraints,
        task_state=task_state,
        history_references=references,
        transcript_message_count=len(transcript),
        prompt_history=prompt_history,
        prompt_history_audit=prompt_audit,
    )


def load_conversation_state(
    db: Session,
    *,
    knowledge_base_id: str,
    session_id: str | None,
    validate_references: bool = True,
    for_update: bool = False,
) -> tuple[QASession | None, ConversationStateSnapshot]:
    if session_id is None:
        return None, anonymous_conversation_state_snapshot(knowledge_base_id)
    canonical_session_id = canonical_uuid(session_id, field_name="session_id")
    statement = select(QASession).where(QASession.id == canonical_session_id)
    if for_update:
        statement = statement.with_for_update()
    session = db.scalar(statement)
    if session is None:
        raise ConversationStateNotFoundError("conversation session not found")
    if canonical_uuid(
        session.knowledge_base_id, field_name="session.knowledge_base_id"
    ) != canonical_uuid(knowledge_base_id, field_name="knowledge_base_id"):
        raise ConversationStateNotFoundError(
            "conversation session does not belong to the requested knowledge base"
        )
    verified = verify_session_state(
        db, session, validate_references=validate_references
    )
    return session, _snapshot_from_verified(
        session,
        transcript=verified[0],
        constraints=verified[1],
        task_state=verified[2],
        references=verified[3],
        state_hash=verified[4],
    )


def initialize_new_session_state(
    db: Session,
    session: QASession,
) -> None:
    db.flush()
    session.conversation_state_protocol_version = (
        CONVERSATION_STATE_PROTOCOL_VERSION
    )
    session.conversation_state_revision = 0
    session.transcript = canonical_transcript(session.transcript)
    session.active_user_constraints_json = canonical_active_user_constraints({})
    session.task_state_json = canonical_task_state({})
    session.history_references_json = []
    session.conversation_state_hash = conversation_state_hash_for_session(session)
    flag_modified(session, "transcript")
    flag_modified(session, "active_user_constraints_json")
    flag_modified(session, "task_state_json")
    flag_modified(session, "history_references_json")


def prepare_session_for_turn(
    db: Session,
    *,
    session: QASession,
    history: Iterable[ChatMessage | dict[str, Any]],
    question: str,
    state_update: ConversationStateUpdate | None,
) -> ConversationStateSnapshot:
    locked = db.scalar(
        select(QASession).where(QASession.id == session.id).with_for_update()
    )
    if locked is None:
        raise ConversationStateNotFoundError("conversation session not found")
    transcript, constraints, task_state, references, _state_hash = (
        verify_session_state(db, locked, validate_references=True)
    )
    client_history = _history_messages(history)
    merged_transcript, merge_audit = _merge_client_history(
        transcript, client_history
    )
    if state_update is not None:
        if state_update.active_user_constraints is not None:
            constraints = canonical_active_user_constraints(
                state_update.active_user_constraints.model_dump()
            )
        if state_update.task_state is not None:
            task_state = canonical_task_state(
                state_update.task_state.model_dump()
            )
    if not task_state.get("objective"):
        task_state = {
            **task_state,
            "objective": question.strip(),
        }
    if task_state.get("status") in {"completed", "cancelled"}:
        task_state = {**task_state, "status": "active"}
    task_state = {**task_state, "current_step": "answering"}
    locked.transcript = merged_transcript
    locked.active_user_constraints_json = constraints
    locked.task_state_json = task_state
    locked.history_references_json = references
    locked.conversation_state_revision = int(
        locked.conversation_state_revision or 0
    ) + 1
    locked.conversation_state_hash = conversation_state_hash_for_session(locked)
    flag_modified(locked, "transcript")
    flag_modified(locked, "active_user_constraints_json")
    flag_modified(locked, "task_state_json")
    flag_modified(locked, "history_references_json")
    db.flush()
    snapshot = _snapshot_from_verified(
        locked,
        transcript=merged_transcript,
        constraints=constraints,
        task_state=task_state,
        references=references,
        state_hash=locked.conversation_state_hash,
    )
    prompt_audit = {
        **snapshot.prompt_history_audit,
        "history_merge": merge_audit,
    }
    return ConversationStateSnapshot(
        **{
            **snapshot.__dict__,
            "prompt_history_audit": prompt_audit,
        }
    )


def merge_search_filters_with_conversation_constraints(
    filters: SearchFilters,
    constraints: dict[str, Any],
) -> SearchFilters:
    request_payload = filters.model_dump()
    state_payload = dict(
        (constraints.get("retrieval_filters") or {})
        if isinstance(constraints, dict)
        else {}
    )
    for field in ("document_ids", "source_paths", "tags", "content_kinds"):
        requested = _canonical_string_list(request_payload.get(field) or [])
        active = _canonical_string_list(state_payload.get(field) or [])
        if requested and active:
            intersection = sorted(set(requested) & set(active))
            if not intersection:
                raise ConversationStateConflictError(
                    f"request filter {field} conflicts with active conversation constraints"
                )
            request_payload[field] = intersection
        elif active:
            request_payload[field] = active
    for field in ("source_type", "partition", "chunk_version"):
        requested = request_payload.get(field)
        active = state_payload.get(field)
        if active is None:
            continue
        if requested is not None and requested != active:
            raise ConversationStateConflictError(
                f"request filter {field} conflicts with active conversation constraints"
            )
        request_payload[field] = active
    active_range = state_payload.get("page_range")
    requested_range = request_payload.get("page_range")
    if active_range is not None:
        if requested_range is None:
            request_payload["page_range"] = tuple(active_range)
        else:
            lower_values = [
                value
                for value in (requested_range[0], active_range[0])
                if value is not None
            ]
            upper_values = [
                value
                for value in (requested_range[1], active_range[1])
                if value is not None
            ]
            lower = max(lower_values) if lower_values else None
            upper = min(upper_values) if upper_values else None
            if lower is not None and upper is not None and lower > upper:
                raise ConversationStateConflictError(
                    "request page_range conflicts with active conversation constraints"
                )
            request_payload["page_range"] = (lower, upper)
    return SearchFilters.model_validate(request_payload)


def append_completed_turn(
    db: Session,
    *,
    session_id: str,
    question: str,
    answer: str,
    run_id: str,
    citations: list[dict[str, Any]],
    answer_session_id: str | None = None,
    retrieval_trace_id: str | None = None,
    task_status: str = "active",
) -> ConversationStateSnapshot:
    session = db.scalar(
        select(QASession).where(QASession.id == session_id).with_for_update()
    )
    if session is None:
        raise ConversationStateNotFoundError("conversation session not found")
    transcript, constraints, task_state, references, _state_hash = (
        verify_session_state(db, session, validate_references=True)
    )
    canonical_run_id = canonical_uuid(run_id, field_name="run_id")
    answer_row: AnswerSession | None = None
    if answer_session_id is not None:
        answer_row = db.get(
            AnswerSession,
            canonical_uuid(
                answer_session_id, field_name="answer_session_id"
            ),
        )
        if answer_row is None:
            raise ConversationStateIntegrityError("answer session is missing")
    canonical_retrieval_trace_id = (
        canonical_uuid(
            retrieval_trace_id,
            field_name="retrieval_trace_id",
        )
        if retrieval_trace_id is not None
        else answer_row.retrieval_trace_id
        if answer_row is not None
        else None
    )
    if (
        answer_row is not None
        and canonical_retrieval_trace_id != answer_row.retrieval_trace_id
    ):
        raise ConversationStateIntegrityError(
            "conversation retrieval trace diverges from answer provenance"
        )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": answer.strip(),
        "run_id": canonical_run_id,
        "citations": [
            _canonical_citation_payload(citation)
            for citation in citations
        ],
    }
    if canonical_retrieval_trace_id is not None:
        assistant_message["retrieval_trace_id"] = canonical_retrieval_trace_id
    transcript.extend(
        [
            {
                "role": "user",
                "content": question.strip(),
                "run_id": canonical_run_id,
            },
            assistant_message,
        ]
    )
    transcript = canonical_transcript(transcript)
    if answer_row is not None:
        reference = {
            "protocol_version": CONVERSATION_REFERENCE_PROTOCOL_VERSION,
            "turn_index": len(references),
            "run_id": canonical_run_id,
            "answer_session_id": answer_row.id,
            "context_package_id": answer_row.context_package_id,
            "retrieval_trace_id": answer_row.retrieval_trace_id,
            "citation_verification_ids": answer_row.citation_ids_json or [],
        }
        references = canonical_history_references([*references, reference])
        _validate_reference_provenance(db, session, references, transcript)
    task_steps = {
        "active": "awaiting_user",
        "waiting_user": "clarification_required",
        "completed": "completed",
        "cancelled": "cancelled",
    }
    if task_status not in task_steps:
        raise ConversationStateConflictError(
            "completed turn task status is unsupported"
        )
    task_state = {
        **task_state,
        "status": task_status,
        "current_step": task_steps[task_status],
    }
    session.transcript = transcript
    session.active_user_constraints_json = constraints
    session.task_state_json = task_state
    session.history_references_json = references
    session.last_question = question
    session.last_answer = answer
    session.conversation_state_revision = int(
        session.conversation_state_revision or 0
    ) + 1
    session.conversation_state_hash = conversation_state_hash_for_session(session)
    flag_modified(session, "transcript")
    flag_modified(session, "active_user_constraints_json")
    flag_modified(session, "task_state_json")
    flag_modified(session, "history_references_json")
    db.commit()
    db.refresh(session)
    verified = verify_session_state(db, session, validate_references=True)
    return _snapshot_from_verified(
        session,
        transcript=verified[0],
        constraints=verified[1],
        task_state=verified[2],
        references=verified[3],
        state_hash=verified[4],
    )


def session_transcript_public_payload(
    db: Session,
    session: QASession,
) -> list[dict[str, Any]]:
    """Expose persisted trace bindings without rewriting historical transcript facts."""

    transcript = canonical_transcript(session.transcript)
    public_transcript: list[dict[str, Any]] = []
    for item in transcript:
        raw_citations = list(item.get("citations") or [])
        public_item = dict(item)
        if not raw_citations:
            public_item["citation_replay_status"] = "not_present"
            public_item["citation_replay_reason"] = None
        else:
            try:
                public_item["citations"] = [
                    Citation.model_validate(citation).model_dump(mode="json")
                    for citation in raw_citations
                ]
            except ValueError:
                public_item["citations"] = []
                public_item["citation_replay_status"] = "unavailable"
                public_item[
                    "citation_replay_reason"
                ] = "persisted_citation_contract_mismatch"
            else:
                public_item["citation_replay_status"] = "valid"
                public_item["citation_replay_reason"] = None
        public_transcript.append(public_item)
    transcript = public_transcript
    missing_run_ids = {
        item["run_id"]
        for item in transcript
        if item["role"] == "assistant"
        and item.get("retrieval_trace_id") is None
        and item.get("run_id") is not None
    }
    if not missing_run_ids:
        return transcript
    plans = db.scalars(
        select(AgentPlan)
        .join(AgentRun, AgentRun.id == AgentPlan.run_id)
        .where(
            AgentPlan.run_id.in_(missing_run_ids),
            AgentPlan.knowledge_base_id == session.knowledge_base_id,
            AgentPlan.retrieval_trace_id.is_not(None),
            AgentRun.session_id == session.id,
        )
        .order_by(AgentPlan.run_id.asc(), AgentPlan.plan_index.desc())
    ).all()
    trace_by_run: dict[str, str] = {}
    for plan in plans:
        if plan.retrieval_trace_id is not None:
            trace_by_run.setdefault(plan.run_id, plan.retrieval_trace_id)
    return [
        {
            **item,
            **(
                {"retrieval_trace_id": trace_by_run[item["run_id"]]}
                if item["role"] == "assistant"
                and item.get("retrieval_trace_id") is None
                and item.get("run_id") in trace_by_run
                else {}
            ),
        }
        for item in transcript
    ]


def session_summary_payload(
    db: Session,
    session: QASession,
) -> dict[str, Any]:
    _session, snapshot = load_conversation_state(
        db,
        knowledge_base_id=session.knowledge_base_id,
        session_id=session.id,
        validate_references=True,
    )
    return {
        "id": session.id,
        "knowledge_base_id": session.knowledge_base_id,
        "title": session.title,
        "last_question": session.last_question,
        "last_answer": session.last_answer,
        "transcript": session_transcript_public_payload(db, session),
        "conversation_state": snapshot.public_payload(),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
