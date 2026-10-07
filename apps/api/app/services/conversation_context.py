"""Replayable, source-addressed dialogue memory; never answer evidence."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from pydantic import Field
from sqlalchemy import select

from app.core.config import get_settings
from app.intent_contracts import ConversationUserContext
from app.models import AgentObservation, AgentRun, QASession
from app.retrieval_control_contracts import ControlContract, control_hash
from app.services.agent_context import (ContextUnit, PriorityCandidate, stable_priority_order,
    estimate_tokens, json_message, plan_context, apply_provider_usage)
from app.services.qa_performance import qa_stage

PROTOCOL = "conversation_context_v1"
CHECKPOINT_PROTOCOL = "conversation_checkpoint_v1"


class ConversationReadArguments(ControlContract):
    keys: tuple[str, ...] = Field(min_length=1, max_length=8)


class CheckpointSelection(ControlContract):
    turn_key: str = Field(pattern=r"^turn_[1-9][0-9]*$")
    role: Literal["user", "assistant"]
    quote: str = Field(min_length=1, max_length=2000)


class CheckpointArguments(ControlContract):
    selections: tuple[CheckpointSelection, ...] = Field(default=(), max_length=24)


@dataclass
class ConversationContext:
    turns: tuple[dict[str, Any], ...] = ()
    instructions: tuple[str, ...] = ()
    selections: list[dict[str, Any]] = field(default_factory=list)
    archived_count: int = 0
    read_keys: set[str] = field(default_factory=set)
    checkpoint_calls: int = 0

    @classmethod
    def load(cls, db, run: AgentRun) -> ConversationContext:
        if not run.session_id:
            return cls()
        session = db.get(QASession, run.session_id)
        if session is None or session.knowledge_base_id != run.knowledge_base_id:
            raise ValueError("conversation_context_scope_mismatch")
        transcript = session.transcript or []
        count = len(transcript) // 2
        turns = tuple({"key": f"turn_{index + 1}",
                       "user": transcript[2 * index]["content"],
                       "assistant": transcript[2 * index + 1]["content"]} for index in range(count))
        context = cls(turns, tuple((session.active_user_constraints_json or {}).get("instructions") or ()))
        row = db.scalar(select(AgentObservation).join(AgentRun, AgentRun.id == AgentObservation.run_id)
                        .where(AgentRun.session_id == run.session_id,
                               AgentObservation.observation_type == "conversation_checkpoint")
                        .order_by(AgentObservation.created_at.desc()).limit(1))
        if row is not None:
            value = row.observation_json or {}
            archived = value.get("archived_turn_count", 0)
            if (value.get("protocol_version") == CHECKPOINT_PROTOCOL and value.get("status") == "completed" and type(archived) is int
                    and 0 < archived <= count and value.get("prefix_hash") == control_hash(turns[:archived])):
                selections = value.get("selections") or []
                context._quotes(selections)
                context.archived_count, context.selections = archived, selections
        return context

    def _by_key(self) -> dict[str, dict[str, Any]]:
        return {turn["key"]: turn for turn in self.turns}

    def _quotes(self, selections: list[dict[str, Any]]) -> list[dict[str, str]]:
        turns = self._by_key()
        values = []
        for item in selections:
            turn = turns.get(item.get("turn_key"))
            role, span = item.get("role"), item.get("char_span")
            if turn is None or role not in {"user", "assistant"} or not isinstance(span, list) or len(span) != 2:
                raise ValueError("conversation_checkpoint_reference_invalid")
            left, right = span
            if type(left) is not int or type(right) is not int or not 0 <= left < right <= len(turn[role]):
                raise ValueError("conversation_checkpoint_span_invalid")
            values.append({"turn_key": turn["key"], "role": role, "text": turn[role][left:right]})
        return values

    def navigation(self) -> dict[str, Any]:
        archive = self.turns[:self.archived_count]
        entries = {turn["key"]: {"key": turn["key"], "topic": turn["user"][:240],
                    "topic_excerpt": len(turn["user"]) > 240} for turn in archive}
        active = {item["turn_key"] for item in self.selections}
        order = stable_priority_order(PriorityCandidate(turn["key"], active_branch=turn["key"] in active,
            retrieval_order=len(archive) - index, estimated_tokens=estimate_tokens(json.dumps(entries[turn["key"]], ensure_ascii=False)),
            stable_key=turn["key"]) for index, turn in enumerate(archive))
        directory, cost = [], 0
        for key in order:
            item_cost = estimate_tokens(json.dumps(entries[key], ensure_ascii=False))
            if cost + item_cost <= get_settings().agent_context_window_tokens // 16:
                directory.append(entries[key]);cost += item_cost
        return {"is_evidence": False,
                "active_user_instructions": list(self.instructions),
                "checkpoint": self._quotes(self.selections),
                "archived_turns": directory, "directory_complete": len(directory) == len(archive),
                "turn_key_range": [self.turns[0]["key"], self.turns[-1]["key"]] if self.turns else []}

    def messages(self) -> list[dict[str, str]]:
        return [{"role": role, "content": f"Historical user {turn['key']}:\n{turn[role]}" if role == "user" else turn[role]}
                for turn in self.turns[self.archived_count:]
                for role in ("user", "assistant")]

    def units(self) -> list[ContextUnit]:
        return [ContextUnit(f"conversation:{turn['key']}:{role}", "dialogue_message", "P1",
                            f"Historical user {turn['key']}:\n{turn[role]}" if role == "user" else turn[role],
                            atomic_group=f"conversation:{turn['key']}", set_name="working")
                for turn in self.turns[self.archived_count:] for role in ("user", "assistant")]

    def unread_keys(self) -> set[str]:
        return {turn["key"] for turn in self.turns[:self.archived_count]} - self.read_keys

    def read(self, keys: tuple[str, ...]) -> dict[str, Any]:
        if len(set(keys)) != len(keys) or not set(keys) <= self.unread_keys():
            raise ValueError("conversation_read_not_unread")
        turns = self._by_key()
        result = {"tool": "conversation.read", "status": "ok", "is_evidence": False,
                  "turns": [{"key": key, "messages": [{"role": role, "content": turns[key][role]}
                                                      for role in ("user", "assistant")]} for key in keys]}
        self.read_keys.update(keys)
        return result

    def user_context(self, keys: tuple[str, ...]) -> tuple[ConversationUserContext, ...]:
        turns = self._by_key()
        if len(set(keys)) != len(keys) or not set(keys) <= set(turns):
            raise ValueError("conversation_context_key_invalid")
        return tuple(ConversationUserContext(turn_key=key, text=turns[key]["user"]) for key in keys)

    def _selection_references(self, raw: dict, end: int, quote_budget: int) -> list[dict[str, Any]]:
        selections = CheckpointArguments.model_validate(raw).selections
        if sum(len(note.quote) for note in selections) > quote_budget:
            raise ValueError("conversation_checkpoint_quote_budget_exceeded")
        allowed = {turn["key"]: turn for turn in self.turns[:end]}
        references = []
        for note in selections:
            turn = allowed.get(note.turn_key)
            if turn is None or note.quote not in turn[note.role]:
                raise ValueError("conversation_checkpoint_quote_not_witnessed")
            left = turn[note.role].index(note.quote)
            reference = {"turn_key": note.turn_key, "role": note.role, "char_span": [left, left + len(note.quote)]}
            if reference not in references:
                references.append(reference)
        return references

    async def prepare(self, db, run, provider, *, system: str, tools: list[dict],
                      task_packet: dict, max_tokens: int, timeout_seconds: float) -> None:
        settings = get_settings()
        envelope = estimate_tokens(system) + estimate_tokens(json.dumps(tools, ensure_ascii=False))
        envelope += estimate_tokens(json.dumps(task_packet, ensure_ascii=False)) + max_tokens + 256
        history_budget = settings.agent_context_window_tokens - envelope
        if history_budget <= 0:
            raise ValueError("conversation_envelope_capacity_exceeded")
        remaining = self.turns[self.archived_count:]
        if sum(estimate_tokens(turn["user"]) + estimate_tokens(turn["assistant"]) for turn in remaining) <= history_budget:
            return
        # Keep a complete recent tail. Older messages are selected semantically;
        # references preserve exact text and allow lazy restoration.
        tail_cost, cut = 0, len(self.turns)
        for index in range(len(self.turns) - 1, self.archived_count - 1, -1):
            cost = estimate_tokens(self.turns[index]["user"]) + estimate_tokens(self.turns[index]["assistant"])
            if tail_cost + cost > history_budget // 3:
                break
            tail_cost += cost
            cut = index
        if cut <= self.archived_count:
            return
        schema = CheckpointArguments.model_json_schema()
        summary_system = ("Call conversation_checkpoint once. Select exact original quotations preserving the user's goals, "
                          "referents, corrections, answer constraints, decisions and unfinished requirements. "
                          "Keep navigation clues from assistant messages, never treat them as answer evidence. "
                          "Do not rewrite quotes or emit reasoning. Prefer short complete semantic statements.")
        input_budget = settings.agent_context_window_tokens - max_tokens - estimate_tokens(summary_system) - estimate_tokens(json.dumps(schema)) - 256
        started = asyncio.get_running_loop().time()
        while self.archived_count < cut:
            end, cost = self.archived_count, 0
            while end < cut:
                item_cost = estimate_tokens(json.dumps(self.turns[end], ensure_ascii=False))
                if cost + item_cost > input_budget // 2:
                    break
                cost += item_cost
                end += 1
            if end == self.archived_count:
                raise ValueError("conversation_turn_capacity_exceeded")
            packet = {"current_task": task_packet, "quote_character_budget": settings.agent_history_summary_max_chars,
                      "previous_checkpoint": self._quotes(self.selections),
                      "messages": list(self.turns[self.archived_count:end])}
            budget = min(max_tokens, 4096)
            tool = {"name": "conversation_checkpoint", "result_tool": "conversation.checkpoint",
                    "description": "Retain source-addressed dialogue memory under window pressure.",
                    "input_schema": schema, "passthrough": True}
            context_plan = plan_context([ContextUnit("checkpoint", "dialogue_checkpoint", "P0",
                json_message("user", packet)["content"], set_name="pinned")],
                input_token_budget=settings.agent_context_window_tokens, reserved_output_tokens=budget,
                stable_prefix=summary_system, system_prompt=summary_system, tools=[tool],
                context_window_tokens=settings.agent_context_window_tokens)
            observation = AgentObservation(run_id=run.id, observation_type="conversation_checkpoint", verdict="prepared",
                observation_json={"protocol_version": CHECKPOINT_PROTOCOL, "status": "prepared",
                                  "context_plan": context_plan.audit, "source_text_persisted": False})
            db.add(observation)
            db.commit()
            messages = [json_message("user", packet)]
            attempts = 0
            try:
                while True:
                    with qa_stage("history_projection"):
                        async with asyncio.timeout(max(0.001, timeout_seconds - (asyncio.get_running_loop().time() - started))):
                            raw = await provider.classify_json_messages(summary_system, messages,
                                max_tokens=budget, response_schema=schema, native_tools=[tool])
                    attempts += 1
                    try:
                        references = self._selection_references(raw, end, settings.agent_history_summary_max_chars)
                        break
                    except ValueError:
                        if attempts >= 2:
                            raise
                        messages.extend((json_message("assistant", {"tool": "conversation.checkpoint", "arguments": raw}),
                                         json_message("user", {"tool": "conversation.checkpoint", "status": "error",
                                             "error": "Select exact quotes from provided turns within quote_character_budget."})))
            except BaseException as exc:
                observation.verdict = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                observation.observation_json = {**observation.observation_json, "status": observation.verdict,
                                               "failure_class": type(exc).__name__, "model_call_count": attempts}
                db.commit()
                raise
            self.selections, self.archived_count = references, end
            self.checkpoint_calls += attempts
            audit = {"protocol_version": CHECKPOINT_PROTOCOL, "archived_turn_count": end,
                     "prefix_hash": control_hash(self.turns[:end]), "selections": references,
                     "model_call_count": attempts, "provider_response_persisted": False,
                     "provider_call": provider.provider_call_audit(),
                     "context_plan": apply_provider_usage(context_plan.audit, provider.provider_call_audit())}
            observation.verdict, observation.observation_json = "completed", {**audit, "status": "completed"}
            db.commit()
