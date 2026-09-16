"""Carry useful verified excerpts through an otherwise-insufficient reuse route."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy.orm import Session

from app.models import AgentRun, AnswerSession, ContextPackage
from app.services.agent_reflection import ReflectionContractError, reflection_hash
from app.services.answer_sources import build_answer_evidence_manifest
from app.services.context_graph import context_package_to_contexts, runtime_settings_state_hash
from app.services.retrieval import get_context_package

PARTIAL_CONTEXT_PROTOCOL = "verified_partial_context_carry_v1"


class PartialContextIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    protocol_version: Literal["verified_partial_context_carry_v1"] = PARTIAL_CONTEXT_PROTOCOL
    run_id: str
    knowledge_base_id: str
    session_id: str
    question_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_package_id: str
    source_answer_id: str
    source_answer_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_trace_id: str
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_ids: list[str] = Field(min_length=1, max_length=256)
    evaluator_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("chunk_ids")
    @classmethod
    def unique_chunks(cls, values):
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("partial source ids must be unique and nonempty")
        return values


class PartialContextState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    intent: PartialContextIntent
    intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "applied"] = "pending"
    candidate_package_id: str | None = None
    target_package_id: str | None = None


def _answer_identity(answer: AnswerSession) -> str:
    return reflection_hash({key: getattr(answer, key) for key in
        ("id", "knowledge_base_id", "qa_session_id", "context_package_id", "retrieval_trace_id", "answer", "citation_ids_json", "diagnostics_json")})


def prepare_partial_context(db: Session, *, run: AgentRun, source: ContextPackage, source_answer: AnswerSession, evaluator: dict[str, Any]) -> None:
    selected = list(evaluator.get("referenced_chunk_ids") or [])
    if evaluator.get("verdict") != "insufficient" or not selected:
        return
    if (not run.session_id or source_answer.qa_session_id != run.session_id
        or source_answer.knowledge_base_id != run.knowledge_base_id or source_answer.context_package_id != source.id
        or source.knowledge_base_id != run.knowledge_base_id or not get_context_package(db, source.id)):
        raise ReflectionContractError("partial_context_source_scope_invalid")
    manifest = build_answer_evidence_manifest(source, context_package_to_contexts(source))
    if not set(selected).issubset({item["chunk_id"] for item in manifest.sources}):
        raise ReflectionContractError("partial_context_source_outside_manifest")
    intent = PartialContextIntent(run_id=run.id, knowledge_base_id=run.knowledge_base_id, session_id=run.session_id,
        question_hash=reflection_hash(run.question), runtime_hash=runtime_settings_state_hash(),
        source_package_id=source.id, source_answer_id=source_answer.id, source_answer_hash=_answer_identity(source_answer),
        source_trace_id=source.retrieval_trace_id, source_manifest_hash=manifest.manifest_hash,
        chunk_ids=selected, evaluator_input_hash=evaluator["input_hash"], evaluator_output_hash=evaluator["output_hash"])
    state = PartialContextState(intent=intent, intent_hash=reflection_hash(intent.model_dump(mode="json")))
    if (run.metadata_json or {}).get("partial_context_carry") is not None:
        raise ReflectionContractError("partial_context_intent_already_exists")
    run.metadata_json = {**run.metadata_json, "partial_context_carry": state.model_dump(mode="json")}
    db.commit()


def _load_source(db: Session, run: AgentRun) -> tuple[PartialContextState, ContextPackage] | None:
    value = (run.metadata_json or {}).get("partial_context_carry")
    if value is None:
        return None
    try:
        state = PartialContextState.model_validate(value)
    except ValidationError:
        raise ReflectionContractError("partial_context_intent_invalid") from None
    intent = state.intent
    if (state.intent_hash != reflection_hash(intent.model_dump(mode="json")) or intent.run_id != run.id
        or intent.knowledge_base_id != run.knowledge_base_id or intent.session_id != run.session_id
        or intent.question_hash != reflection_hash(run.question) or intent.runtime_hash != runtime_settings_state_hash()):
        raise ReflectionContractError("partial_context_intent_identity_changed")
    source = db.get(ContextPackage, intent.source_package_id)
    answer = db.get(AnswerSession, intent.source_answer_id)
    if (answer is None or answer.qa_session_id != run.session_id or answer.context_package_id != intent.source_package_id
        or _answer_identity(answer) != intent.source_answer_hash):
        raise ReflectionContractError("partial_context_history_identity_changed")
    if (source is None or source.knowledge_base_id != run.knowledge_base_id or source.retrieval_trace_id != intent.source_trace_id
        or not get_context_package(db, source.id)
        or build_answer_evidence_manifest(source, context_package_to_contexts(source)).manifest_hash != intent.source_manifest_hash):
        raise ReflectionContractError("partial_context_source_identity_changed")
    return state, source


def partial_context_observation(db: Session, *, run: AgentRun, query_facets: dict, granularity: str) -> list[dict]:
    loaded = _load_source(db, run)
    if loaded is None:
        return []
    state, source = loaded
    items = {item["chunk_id"]: item for item in source.package_json["chunks"]}
    summaries = [{"chunk_id": cid, "document_title": items[cid]["document_title"], "text_excerpt": items[cid]["content"][:360],
        "summary_hash": reflection_hash(items[cid]), "source_span_address": {key: items[cid]["source_span"].get(key)
            for key in ("chunk_id", "document_version_id", "char_span", "page_range")}} for cid in state.intent.chunk_ids]
    return [{"bounded_graph_observation": {"plan_index": 0, "retrieval_granularity": granularity,
        "required_facets": query_facets.get("required_facets") or [], "covered_facets": [], "result_chunk_ids": [],
        "result_count": 0, "citable_span_count": len(summaries), "candidate_chunk_span_summaries": summaries,
        "observation_hash": state.intent_hash}, "evidence_evaluator": {"verdict": "need_more_same_node",
            "reason": "Useful verified excerpts from this conversation are available; retrieve the missing parts of the current question.",
            "missing_facets": query_facets.get("required_facets") or []}}]


def apply_partial_context(db: Session, *, run: AgentRun, candidate: ContextPackage, token_budget: int):
    from app.services.chunking import rough_token_count
    from app.services.reflection_sources import retain_reflection_sources, source_covers
    from app.services.reflection_context import restore_reflection_context
    loaded = _load_source(db, run)
    if loaded is None:
        return candidate, context_package_to_contexts(candidate)
    state, source = loaded
    if state.status == "applied":
        if state.candidate_package_id != candidate.id:
            raise ReflectionContractError("partial_context_application_target_changed")
        target = db.get(ContextPackage, state.target_package_id)
        if target is None or not get_context_package(db, target.id):
            raise ReflectionContractError("partial_context_application_missing")
        return target, context_package_to_contexts(target)
    original_candidate_id = candidate.id
    originals = {item["chunk_id"]: item for item in source.package_json["chunks"]}
    current = {item["chunk_id"]: item for item in candidate.package_json["chunks"]}
    needed = [cid for cid in state.intent.chunk_ids if cid not in current or not source_covers(current[cid], originals[cid])]
    reserve = sum(rough_token_count(originals[cid]["content"]) for cid in state.intent.chunk_ids)
    if needed and candidate.token_count + sum(rough_token_count(originals[cid]["content"]) for cid in needed) > token_budget:
        cap = token_budget - reserve
        if cap <= 0 or not candidate.hit_chunk_ids_json:
            raise ReflectionContractError("partial_context_capacity_exceeded")
        # No answer has yet bound this new retrieval package. Repack its original
        # hit domain with the existing materializer, reserving verified excerpts.
        candidate, _ = restore_reflection_context(db, source_package=candidate,
            target_chunk_ids=list(candidate.hit_chunk_ids_json), preserve_chunk_ids=[],
            token_budget=token_budget, restore_per_chunk_budget=0, reserved_token_budget=reserve)
    target, contexts = retain_reflection_sources(db, candidate_package=candidate, source_package=source,
        preserve_chunk_ids=state.intent.chunk_ids, token_budget=token_budget)
    state.status, state.candidate_package_id, state.target_package_id = "applied", original_candidate_id, target.id
    run.metadata_json = {**run.metadata_json, "partial_context_carry": state.model_dump(mode="json")}
    db.commit()
    return target, contexts
