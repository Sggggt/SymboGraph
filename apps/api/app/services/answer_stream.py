"""Bounded provider-to-observer answer deltas for one grounded generation."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re
from typing import Literal


ANSWER_STREAM_PROTOCOL = "grounded_answer_provider_delta_v2"
GROUNDED_MARKDOWN_INLINE_CITATIONS_PROTOCOL = (
    "grounded_markdown_inline_citations_v1"
)
AnswerStreamUpdateKind = Literal["delta", "replace"]
AnswerStreamSink = Callable[[dict[str, str]], Awaitable[None]]

_ANSWER_STREAM_SINK: ContextVar[AnswerStreamSink | None] = ContextVar(
    "grounded_answer_stream_sink",
    default=None,
)
_ANSWER_UNITS = re.compile(r'"answer_units"\s*:\s*\[')
_TEXT_FIELD = re.compile(r'"text"\s*:\s*"')
_GFM_BLOCK_PREFIX = re.compile(
    r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|```|~~~|>|\|)",
)


@contextmanager
def use_answer_stream_sink(sink: AnswerStreamSink | None) -> Iterator[None]:
    token = _ANSWER_STREAM_SINK.set(sink)
    try:
        yield
    finally:
        _ANSWER_STREAM_SINK.reset(token)


def answer_streaming_enabled() -> bool:
    return _ANSWER_STREAM_SINK.get() is not None


async def publish_answer_stream_update(
    kind: AnswerStreamUpdateKind,
    text: str,
) -> bool:
    if not text:
        return False
    sink = _ANSWER_STREAM_SINK.get()
    if sink is None:
        return False
    from app.services.qa_performance import current_qa_performance

    recorder = current_qa_performance()
    if recorder is not None:
        recorder.mark_first_response(token=kind == "delta")
    await sink(
        {
            "protocol_version": ANSWER_STREAM_PROTOCOL,
            "type": kind,
            "text": text,
        }
    )
    return True


def _preceded_by_unescaped_backslash(value: str, index: int) -> bool:
    count = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == "\\":
        count += 1
        cursor -= 1
    return count % 2 == 1


def _decode_partial_json_string(value: str, start: int) -> tuple[str, int, bool]:
    result: list[str] = []
    cursor = start
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while cursor < len(value):
        character = value[cursor]
        if character == '"':
            return "".join(result), cursor + 1, True
        if character != "\\":
            result.append(character)
            cursor += 1
            continue
        if cursor + 1 >= len(value):
            break
        escaped = value[cursor + 1]
        if escaped == "u":
            digits = value[cursor + 2 : cursor + 6]
            if len(digits) != 4 or any(char not in "0123456789abcdefABCDEF" for char in digits):
                break
            result.append(chr(int(digits, 16)))
            cursor += 6
            continue
        replacement = escapes.get(escaped)
        if replacement is None:
            break
        result.append(replacement)
        cursor += 2
    return "".join(result), cursor, False


def grounded_answer_gfm_unit(text: str) -> tuple[str, bool]:
    """Add presentation-only GFM structure without changing source prose."""

    if not text or _GFM_BLOCK_PREFIX.match(text):
        return text, False
    return f"- {text}", True


def _answer_units_prefix(raw_json: str, start: int) -> str:
    """Return only the bounded answer_units array prefix.

    The provider response is untrusted until final schema validation.  A
    top-level or trailing field named ``text`` must never be projected merely
    because it follows the answer_units marker.
    """

    depth = 1
    in_string = False
    escaped = False
    cursor = start
    while cursor < len(raw_json):
        character = raw_json[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return raw_json[start:cursor]
        cursor += 1
    return raw_json[start:]


def partial_grounded_answer_text(raw_json: str) -> str:
    """Project the append-only prefix of ``answer_units[].text``.

    This is a presentation projection only. The complete provider response is
    still parsed and validated by the closed generation schema before it can be
    persisted or treated as an answer.
    """

    marker = _ANSWER_UNITS.search(raw_json)
    if marker is None:
        return ""
    answer_units = _answer_units_prefix(raw_json, marker.end())
    cursor = 0
    texts: list[str] = []
    while cursor < len(answer_units):
        match = _TEXT_FIELD.search(answer_units, cursor)
        if match is None:
            break
        if _preceded_by_unescaped_backslash(answer_units, match.start()):
            cursor = match.end()
            continue
        text, cursor, complete = _decode_partial_json_string(
            answer_units,
            match.end(),
        )
        texts.append(text)
        if not complete:
            break
    # Keep the provider prefix append-only while the JSON string is still
    # incomplete.  Presentation normalization runs once after the closed
    # schema has been parsed; applying it to a partial prefix can oscillate
    # when an initial ``1`` later becomes ``1. `` or ``#`` becomes ``## ``.
    return "\n\n".join(texts)


@dataclass
class GroundedAnswerDeltaProjector:
    raw_json: str = ""
    rendered_text: str = ""
    delta_count: int = 0
    replace_count: int = 0

    def feed(self, raw_delta: str) -> tuple[AnswerStreamUpdateKind, str] | None:
        if not raw_delta:
            return None
        self.raw_json += raw_delta
        projected = partial_grounded_answer_text(self.raw_json)
        if projected == self.rendered_text:
            return None
        if projected.startswith(self.rendered_text):
            delta = projected[len(self.rendered_text) :]
            self.rendered_text = projected
            if delta:
                self.delta_count += 1
                return "delta", delta
            return None
        self.rendered_text = projected
        self.replace_count += 1
        return "replace", projected

    def finalize(self, final_text: str) -> tuple[AnswerStreamUpdateKind, str] | None:
        if final_text == self.rendered_text:
            return None
        if final_text.startswith(self.rendered_text):
            delta = final_text[len(self.rendered_text) :]
            self.rendered_text = final_text
            if delta:
                self.delta_count += 1
                return "delta", delta
            return None
        self.rendered_text = final_text
        self.replace_count += 1
        return "replace", final_text

    def audit(self, *, provider_stream_used: bool) -> dict[str, object]:
        return {
            "protocol_version": ANSWER_STREAM_PROTOCOL,
            "provider_stream_used": provider_stream_used,
            "delta_count": self.delta_count,
            "replace_count": self.replace_count,
            "raw_response_characters": len(self.raw_json),
            "rendered_characters": len(self.rendered_text),
            "provider_response_persisted": False,
        }


class GroundedMarkdownStreamError(ValueError):
    """Content-free failure for the native Markdown transport."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


_SOURCE_HANDLE = re.compile(r"^src_[1-9][0-9]*$")


@dataclass(frozen=True)
class GroundedMarkdownResult:
    answer: str
    final_delta: str
    audit: dict[str, object]


@dataclass
class GroundedMarkdownAccumulator:
    """Tolerantly convert valid inline citations and preserve every other byte."""

    allowed_handles: frozenset[str]
    max_characters: int = 262_144
    _parts: list[str] | None = None
    _pending: str = ""
    _raw_characters: int = 0
    _rendered_characters: int = 0
    _delta_count: int = 0
    _citation_marker_count: int = 0
    _invalid_marker_count: int = 0

    def __post_init__(self) -> None:
        if not self.allowed_handles:
            raise GroundedMarkdownStreamError("answer_stream_sources_empty")
        if self.max_characters < 1:
            raise GroundedMarkdownStreamError("answer_stream_limit_invalid")
        self._parts = []

    @property
    def parts(self) -> list[str]:
        assert self._parts is not None
        return self._parts

    @staticmethod
    def _partial_prefix_length(value: str) -> int:
        prefix = "⟦cite:"
        maximum = min(len(value), len(prefix) - 1)
        for length in range(maximum, 0, -1):
            if value.endswith(prefix[:length]):
                return length
        return 0

    def _render_marker(self, raw_marker: str) -> str:
        payload = raw_marker[len("⟦cite:") : -1]
        handles = tuple(payload.split(",")) if payload else ()
        if (
            not handles
            or len(handles) > 64
            or len(set(handles)) != len(handles)
            or any(_SOURCE_HANDLE.fullmatch(handle) is None for handle in handles)
            or any(handle not in self.allowed_handles for handle in handles)
        ):
            self._invalid_marker_count += 1
            return raw_marker
        self._citation_marker_count += 1
        return " ".join(
            f"[{handle.removeprefix('src_')}](#source-{handle.removeprefix('src_')})"
            for handle in handles
        )

    def _emit(self, value: str) -> str:
        if value:
            self.parts.append(value)
            self._rendered_characters += len(value)
        return value

    def feed(self, raw_delta: str) -> str:
        if not raw_delta:
            return ""
        if "\x00" in raw_delta:
            raise GroundedMarkdownStreamError("answer_stream_contains_nul")
        if self._raw_characters + len(raw_delta) > self.max_characters:
            raise GroundedMarkdownStreamError("answer_stream_limit_exceeded")
        self._raw_characters += len(raw_delta)
        self._pending += raw_delta
        emitted: list[str] = []
        prefix = "⟦cite:"
        while self._pending:
            marker_start = self._pending.find(prefix)
            if marker_start < 0:
                hold = self._partial_prefix_length(self._pending)
                visible = self._pending[:-hold] if hold else self._pending
                self._pending = self._pending[-hold:] if hold else ""
                if visible:
                    emitted.append(self._emit(visible))
                break
            if marker_start > 0:
                emitted.append(self._emit(self._pending[:marker_start]))
                self._pending = self._pending[marker_start:]
            marker_end = self._pending.find("⟧", len(prefix))
            if marker_end < 0:
                if len(self._pending) <= 512:
                    break
                self._invalid_marker_count += 1
                emitted.append(self._emit(self._pending[0]))
                self._pending = self._pending[1:]
                continue
            raw_marker = self._pending[: marker_end + 1]
            emitted.append(self._emit(self._render_marker(raw_marker)))
            self._pending = self._pending[marker_end + 1 :]
        visible_delta = "".join(emitted)
        if visible_delta:
            self._delta_count += 1
        return visible_delta

    def finalize(
        self,
        *,
        provider_stream_used: bool,
        source_handle_count: int,
    ) -> GroundedMarkdownResult:
        final_delta = ""
        if self._pending:
            self._invalid_marker_count += int("⟦cite:" in self._pending)
            final_delta = self._emit(self._pending)
            if final_delta:
                self._delta_count += 1
            self._pending = ""
        answer = "".join(self.parts)
        if not answer.strip():
            raise GroundedMarkdownStreamError("answer_stream_empty")
        return GroundedMarkdownResult(
            answer=answer,
            final_delta=final_delta,
            audit={
                "protocol_version": GROUNDED_MARKDOWN_INLINE_CITATIONS_PROTOCOL,
                "provider_stream_used": provider_stream_used,
                "delta_count": self._delta_count,
                "replace_count": 0,
                "raw_response_characters": self._raw_characters,
                "rendered_characters": self._rendered_characters,
                "unit_count": 1,
                "source_handle_count": source_handle_count,
                "citation_marker_count": self._citation_marker_count,
                "invalid_marker_count": self._invalid_marker_count,
                "append_only": True,
                "provider_response_persisted": False,
            },
        )
