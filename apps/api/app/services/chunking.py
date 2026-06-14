from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.parsers import ParsedSection


DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80
TOKENIZER_VERSION = "symbograph_regex_tokenizer_v1"
CHUNK_SCHEMA_VERSION = "chunk_schema_v1"
CURRENT_EMBEDDING_TEXT_VERSION = "contextual_text_v1"


def normalize_text(text: str) -> str:
    text = (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def stable_hash(value: Any) -> str:
    import json

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str


@dataclass
class FixedChunk:
    chunk_index: int
    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    section_path: str | None
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_hash(self) -> str:
        return text_hash(self.text)


@dataclass
class PreparedDocument:
    text: str
    section_offsets: list[dict[str, Any]]
    protected_spans: list[ProtectedSpan]


class FixedTokenChunker:
    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer_version: str = TOKENIZER_VERSION,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.tokenizer_version = tokenizer_version

    def tokenize(self, text: str) -> list[Token]:
        return [Token(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text or "")]

    def prepare_document(self, sections: list[ParsedSection], *, title: str) -> PreparedDocument:
        parts: list[str] = []
        offsets: list[dict[str, Any]] = []
        protected: list[ProtectedSpan] = []
        cursor = 0
        for index, section in enumerate(sections):
            section_text = normalize_text(section.text)
            if not section_text:
                continue
            heading = normalize_text(section.section or section.title or title)
            prefix = f"\n\n# {heading}\n" if parts else f"# {heading}\n"
            parts.append(prefix)
            cursor += len(prefix)
            start = cursor
            parts.append(section_text)
            cursor += len(section_text)
            end = cursor
            metadata = dict(section.metadata or {})
            offsets.append(
                {
                    "index": index,
                    "title": section.title,
                    "section_path": heading,
                    "page_number": section.page_number,
                    "char_start": start,
                    "char_end": end,
                    "metadata": metadata,
                }
            )
            protected.extend(_protected_spans_for_section(section_text, base=start, metadata=metadata))
        return PreparedDocument(text="".join(parts).strip(), section_offsets=offsets, protected_spans=protected)

    def split_sections(self, sections: list[ParsedSection], *, title: str) -> tuple[list[FixedChunk], PreparedDocument]:
        prepared = self.prepare_document(sections, title=title)
        return self.split_prepared(prepared), prepared

    def split_prepared(self, prepared: PreparedDocument) -> list[FixedChunk]:
        tokens = self.tokenize(prepared.text)
        if not tokens:
            return []
        chunks: list[FixedChunk] = []
        token_start = 0
        index = 0
        while token_start < len(tokens):
            target_end = min(len(tokens), token_start + self.chunk_size)
            target_end = self._extend_to_protect_boundary(tokens, target_end, prepared.protected_spans)
            if target_end <= token_start:
                target_end = min(len(tokens), token_start + self.chunk_size)
            char_start = tokens[token_start].start
            char_end = tokens[target_end - 1].end
            text = prepared.text[char_start:char_end].strip()
            if text:
                sections = _sections_for_span(prepared.section_offsets, char_start, char_end)
                page_numbers = [item["page_number"] for item in sections if item.get("page_number") is not None]
                section_path = " / ".join(dict.fromkeys(str(item["section_path"]) for item in sections if item.get("section_path")))
                kinds = sorted(
                    {
                        kind
                        for span in prepared.protected_spans
                        if _overlap(char_start, char_end, span.start, span.end) > 0
                        for kind in str(span.kind).split(",")
                        if kind
                    }
                )
                content_kind = "code" if "code" in kinds else "formula" if "formula" in kinds else "table" if "table" in kinds else "text"
                chunks.append(
                    FixedChunk(
                        chunk_index=index,
                        text=text,
                        token_start=token_start,
                        token_end=target_end,
                        char_start=char_start,
                        char_end=char_end,
                        section_path=section_path or None,
                        page_start=min(page_numbers) if page_numbers else None,
                        page_end=max(page_numbers) if page_numbers else None,
                        metadata={
                            "tokenizer_version": self.tokenizer_version,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.overlap,
                            "protected_object_kinds": kinds,
                            "has_table": "table" in kinds,
                            "has_formula": "formula" in kinds,
                            "has_caption": "caption" in kinds,
                            "content_kind": content_kind,
                            "section_indices": [item["index"] for item in sections],
                        },
                    )
                )
                index += 1
            if target_end >= len(tokens):
                break
            token_start = max(token_start + 1, target_end - self.overlap)
        return chunks

    def _extend_to_protect_boundary(self, tokens: list[Token], token_end: int, protected_spans: list[ProtectedSpan]) -> int:
        if token_end >= len(tokens):
            return len(tokens)
        cut_char = tokens[token_end - 1].end
        for span in protected_spans:
            if span.start < cut_char < span.end:
                while token_end < len(tokens) and tokens[token_end - 1].end < span.end:
                    token_end += 1
                return min(token_end, len(tokens))
        return token_end


def _protected_spans_for_section(text: str, *, base: int, metadata: dict[str, Any]) -> list[ProtectedSpan]:
    spans: list[ProtectedSpan] = []
    for match in re.finditer(r"```.*?```", text, flags=re.DOTALL):
        spans.append(ProtectedSpan(base + match.start(), base + match.end(), "code"))
    for match in re.finditer(r"\$\$.*?\$\$", text, flags=re.DOTALL):
        spans.append(ProtectedSpan(base + match.start(), base + match.end(), "formula"))
    for match in re.finditer(r"(?m)^(?:\|.*\|\s*){2,}", text):
        spans.append(ProtectedSpan(base + match.start(), base + match.end(), "table"))
    for match in re.finditer(r"(?m)^#{1,6}\s+.+$", text):
        spans.append(ProtectedSpan(base + match.start(), base + match.end(), "heading"))
    for match in re.finditer(r"(?im)^(?:figure|fig\.|table|caption)\s*[:\d.-].+$", text):
        spans.append(ProtectedSpan(base + match.start(), base + match.end(), "caption"))
    if metadata.get("has_table"):
        spans.append(ProtectedSpan(base, base + len(text), "table"))
    if metadata.get("has_formula"):
        spans.append(ProtectedSpan(base, base + len(text), "formula"))
    if metadata.get("content_kind") == "code":
        spans.append(ProtectedSpan(base, base + len(text), "code"))
    return _merge_protected_spans(spans)


def _merge_protected_spans(spans: list[ProtectedSpan]) -> list[ProtectedSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    merged: list[ProtectedSpan] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            merged[-1] = ProtectedSpan(last.start, max(last.end, span.end), f"{last.kind},{span.kind}")
        else:
            merged.append(span)
    return merged


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _sections_for_span(section_offsets: list[dict[str, Any]], char_start: int, char_end: int) -> list[dict[str, Any]]:
    matched = [
        item
        for item in section_offsets
        if _overlap(char_start, char_end, int(item["char_start"]), int(item["char_end"])) > 0
    ]
    return matched or section_offsets[:1]


def rough_token_count(text: str) -> int:
    return len(FixedTokenChunker().tokenize(text))


def build_contextual_text(
    *,
    document_title: str,
    section_path: str | None,
    page_start: int | None,
    page_end: int | None,
    raw_text: str,
    local_hint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata = metadata or {}
    page_hint = ""
    if page_start is not None and page_end is not None:
        page_hint = f"Pages: {page_start}" if page_start == page_end else f"Pages: {page_start}-{page_end}"
    region_hint = ", ".join(str(item) for item in metadata.get("protected_object_kinds") or [])
    parts = [
        f"Document: {document_title}",
        f"Section path: {section_path or 'General'}",
    ]
    if page_hint:
        parts.append(page_hint)
    if region_hint:
        parts.append(f"Region hints: {region_hint}")
    if local_hint:
        parts.append(f"Local hint: {local_hint}")
    parts.extend(["Raw chunk:", raw_text])
    return "\n".join(parts)


def contextual_text_hash(contextual_text: str) -> str:
    return text_hash(contextual_text)
