from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from app.core.config import get_settings
from app.services.parsers import ParsedSection
from app.services.embeddings import vector_norm
from app.services.quality.policies import ChunkQualityPolicy, QualityDecision
from app.services.quality.signals import build_quality_signals

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
except Exception:  # pragma: no cover - optional fallback
    RecursiveCharacterTextSplitter = None
    MarkdownHeaderTextSplitter = None


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def sanitize_metadata(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, dict):
        return {normalize_text(key) if isinstance(key, str) else key: sanitize_metadata(item) for key, item in value.items()}
    return value


def rough_token_count(text: str) -> int:
    from app.services.chinese_text import estimate_tokens

    return max(1, estimate_tokens(text))


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
CODE_CHUNK_SIZE = 700
CODE_CHUNK_OVERLAP = 100
EMBEDDING_TEXT_VERSION = "metadata_enriched_v1"
CURRENT_EMBEDDING_TEXT_VERSION = "contextual_enriched_v3"
CODE_KEEP_MARKERS = ("centrality", "community", "random network", "configuration model")
MOJIBAKE_MARKERS = ("�", "鈥", "鐩", "绗", "鍥", "灏", "瀛", "凹", "鍫")


def build_splitter(chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    if RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
        )
    return None


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if splitter:
        return [chunk for chunk in splitter.split_text(normalize_text(text)) if chunk.strip()]

    normalized = normalize_text(text)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(start + 1, end - chunk_overlap)
    return [chunk for chunk in chunks if chunk]


def _smart_join(parts: list[str]) -> str:
    """Join sentence/paragraph parts with a space only when both neighbours are
    Latin-script (to avoid inserting spurious spaces inside Chinese text)."""
    if not parts:
        return ""
    result = [parts[0]]
    for prev, nxt in zip(parts, parts[1:]):
        # Need a space when both sides look like Latin words without intervening
        # punctuation/whitespace.
        prev_needs_space = prev and prev[-1].isalnum() and not _RE_CJK_CHAR.search(prev[-1])
        nxt_needs_space = nxt and nxt[0].isalnum() and not _RE_CJK_CHAR.search(nxt[0])
        if prev_needs_space and nxt_needs_space:
            result.append(" ")
        result.append(nxt)
    return "".join(result)


_RE_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def split_text_sentences(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """基于句子边界的文本切分，确保不会在句子中间截断。支持中英文标点。"""
    normalized = normalize_text(text)
    sentences = re.split(r"(?<=[。！？])|(?<=[.?!])\s+", normalized)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append(_smart_join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.extend(split_text(sentence, chunk_size, chunk_overlap))
            continue
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(_smart_join(current_chunk))
            overlap_chunk: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > chunk_overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s) + 1
            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    if current_chunk:
        chunks.append(_smart_join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def semantic_units(text: str) -> list[str]:
    normalized = normalize_text(text)
    units = re.split(r"(?<=[。！？])|(?<=[.?!])\s+|\n{2,}", normalized)
    return [unit.strip() for unit in units if unit.strip()]


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = vector_norm(left) or 1.0
    right_norm = vector_norm(right) or 1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


async def split_text_semantic_embeddings(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Embedding-based semantic splitter for long prose sections.

    It embeds sentence/paragraph units, inserts breakpoints at low adjacent
    similarity, and still enforces hard chunk_size bounds.
    """
    units = semantic_units(text)
    if len(units) < 3:
        return split_text_sentences(text, chunk_size, chunk_overlap)

    from app.services.embeddings import EmbeddingProvider

    embeddings = await EmbeddingProvider().embed_texts(units, text_type="document")
    similarities = [cosine(left, right) for left, right in zip(embeddings, embeddings[1:])]
    threshold = percentile(similarities, 0.25)
    break_after = {idx for idx, score in enumerate(similarities) if score <= threshold}

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    min_break_length = max(200, int(chunk_size * 0.45))

    def flush() -> None:
        nonlocal current, current_length
        if current:
            chunks.append(_smart_join(current).strip())
            overlap_units: list[str] = []
            overlap_length = 0
            for unit in reversed(current):
                if overlap_length + len(unit) > chunk_overlap:
                    break
                overlap_units.insert(0, unit)
                overlap_length += len(unit) + 1
            current = overlap_units
            current_length = overlap_length

    for idx, unit in enumerate(units):
        if len(unit) > chunk_size:
            flush()
            chunks.extend(split_text(unit, chunk_size, chunk_overlap))
            current = []
            current_length = 0
            continue
        if current and current_length + len(unit) + 1 > chunk_size:
            flush()
        current.append(unit)
        current_length += len(unit) + 1
        if idx in break_after and current_length >= min_break_length:
            flush()

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def build_markdown_splitter():
    if MarkdownHeaderTextSplitter:
        return MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
    return None


def split_text_semantic(text: str, source_type: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """根据 source_type 选择最佳切分策略。"""
    if source_type in ("md", "markdown"):
        splitter = build_markdown_splitter()
        if splitter:
            docs = splitter.split_text(text)
            result = []
            for doc in docs:
                header_parts = []
                for level, key in [("#", "h1"), ("##", "h2"), ("###", "h3")]:
                    if key in doc.metadata:
                        header_parts.append(f"{level} {doc.metadata[key]}")
                header = "\n".join(header_parts)
                if header:
                    result.append(f"{header}\n{doc.page_content}")
                else:
                    result.append(doc.page_content)
            return [r for r in result if r.strip()]

    if source_type in ("ipynb", "notebook", "code"):
        return split_text(text, chunk_size, chunk_overlap)

    return split_text_sentences(text, chunk_size, chunk_overlap)


async def split_text_semantic_async(
    text: str,
    source_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    settings = get_settings()
    if source_type in ("md", "markdown", "ipynb", "notebook", "code"):
        return split_text_semantic(text, source_type, chunk_size, chunk_overlap)
    if settings.semantic_chunking_enabled and len(normalize_text(text)) >= settings.semantic_chunking_min_length:
        return await split_text_semantic_embeddings(text, chunk_size, chunk_overlap)
    return split_text_sentences(text, chunk_size, chunk_overlap)


def infer_content_kind(content: str, fallback: str | None = None, metadata: dict | None = None) -> str:
    if fallback:
        return fallback
    if metadata and metadata.get("has_table"):
        return "table"
    if metadata and metadata.get("has_formula"):
        return "formula"
    stripped = content.lstrip()
    if stripped.startswith("[Code Cell]"):
        return "code"
    if stripped.startswith("[Output]"):
        return "output"
    return "text"


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE)).strip()


def mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return marker_count / len(text)


def is_toc_like(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    dotted = sum(1 for line in lines if re.search(r"\.{4,}\s*\d+$", line))
    numeric_short = sum(1 for line in lines if re.fullmatch(r"\d+(\.\d+)*", line) or len(line) <= 3)
    return dotted >= 4 or numeric_short / len(lines) > 0.45


def should_keep_code_chunk(section_name: str, section_title: str) -> bool:
    text = f"{section_name} {section_title}".lower()
    return any(marker in text for marker in CODE_KEEP_MARKERS)


def should_keep_chunk(content: str, content_kind: str, section_name: str, section_title: str) -> bool:
    return chunk_quality_decision(content, content_kind, section_name, section_title).action != "discard"


def chunk_quality_decision(content: str, content_kind: str, section_name: str, section_title: str, metadata: dict | None = None) -> QualityDecision:
    signals = build_quality_signals(
        target_type="chunk",
        text=content,
        title=section_title,
        section=section_name,
        content_kind=content_kind,
        metadata=metadata or {},
        version="chunk_quality_v1",
    )
    return ChunkQualityPolicy().decide(signals, section_name=section_name, section_title=section_title)


def quality_metadata(decision: QualityDecision) -> dict:
    retention = decision.audit.get("retention_decision", {})
    routes = decision.audit.get("route_eligibility", {})
    return {
        "quality_action": decision.action,
        "quality_score": decision.score,
        "quality_reasons": decision.reasons,
        "quality_policy": "chunk_quality_v1",
        "quality_profile_version": "chunk_quality_v1",
        "quality_retain": bool(retention.get("retain", decision.action != "discard")),
        "retention_decision": retention,
        "route_eligibility": routes,
        "quality_decision": {
            "target_type": decision.target_type,
            "action": decision.action,
            "score": decision.score,
            "reasons": decision.reasons,
            "retention_decision": retention,
            "route_eligibility": routes,
        },
        "quality_signals": decision.audit.get("signals", {}),
    }


def embedding_text(
    *,
    document_title: str,
    chapter: str | None,
    section: str | None,
    source_type: str | None,
    content_kind: str | None,
    content: str,
    summary: str | None = None,
    keywords: list[str] | None = None,
    has_table: bool = False,
    has_formula: bool = False,
) -> str:
    parts = [
        f"Document: {document_title}",
        f"Chapter: {chapter or ''}",
        f"Section: {section or ''}",
        f"Source Type: {source_type or ''}",
        f"Content Kind: {content_kind or ''}",
    ]
    if summary:
        parts.append(f"Summary: {summary}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if has_table:
        parts.append("Contains: table")
    if has_formula:
        parts.append("Contains: formula")
    parts.extend(["Content:", content])
    return "\n".join(parts)


def contextual_embedding_text(
    *,
    document_title: str,
    chapter: str | None,
    section: str | None,
    source_type: str | None,
    content_kind: str | None,
    content: str,
    parent_summary: str | None = None,
    prev_summary: str | None = None,
    next_summary: str | None = None,
    summary: str | None = None,
    keywords: list[str] | None = None,
    has_table: bool = False,
    has_formula: bool = False,
) -> str:
    parts = [
        f"Document: {document_title}",
        f"Chapter: {chapter or ''}",
        f"Section: {section or ''}",
        f"Source Type: {source_type or ''}",
        f"Content Kind: {content_kind or ''}",
    ]
    if summary:
        parts.append(f"Summary: {summary}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if has_table:
        parts.append("Contains: table")
    if has_formula:
        parts.append("Contains: formula")
    if parent_summary:
        parts.append(f"Context: This chunk belongs to a section about {parent_summary}.")
    if prev_summary:
        parts.append(f"Previous: {prev_summary}")
    if next_summary:
        parts.append(f"Next: {next_summary}")
    parts.extend(["Content:", content])
    return "\n".join(parts)


def chunk_sections_with_stats(sections: Iterable[ParsedSection], chapter: str, source_type: str) -> tuple[list[dict], dict[str, int]]:
    chunks: list[dict] = []
    stats = {"chunks_before_filter": 0, "chunks_filtered": 0}
    for section_index, section in enumerate(sections, start=1):
        section_text = normalize_text(section.text)
        section_title = normalize_text(section.title)
        section_name = normalize_text(section.section or section.title)
        content_kind = infer_content_kind(section_text, section.metadata.get("content_kind") if section.metadata else None, section.metadata)
        unit_size = CODE_CHUNK_SIZE if content_kind == "code" else DEFAULT_CHUNK_SIZE
        overlap = CODE_CHUNK_OVERLAP if content_kind == "code" else DEFAULT_CHUNK_OVERLAP
        for chunk_index, content in enumerate(split_text(section_text, unit_size, overlap), start=1):
            stats["chunks_before_filter"] += 1
            decision = chunk_quality_decision(content, content_kind, section_name, section_title, section.metadata)
            if decision.action == "discard":
                stats["chunks_filtered"] += 1
                continue
            snippet = content[:280].strip()
            chunks.append(
                {
                    "content": content,
                    "snippet": snippet,
                    "chapter": chapter,
                    "section": section_name,
                    "page_number": section.page_number,
                    "token_count": rough_token_count(content),
                    "metadata": sanitize_metadata({
                        "section_title": section_title,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                        "content_kind": content_kind,
                        "chunking_strategy": "chunk_800_metadata_enriched_v1",
                        **quality_metadata(decision),
                        **section.metadata,
                    }),
                }
            )
    return chunks, stats


def chunk_sections_semantic(sections: Iterable[ParsedSection], chapter: str, source_type: str) -> tuple[list[dict], dict[str, int]]:
    """语义感知切分：Markdown 用标题切分，其他用句子边界切分。"""
    chunks: list[dict] = []
    stats = {"chunks_before_filter": 0, "chunks_filtered": 0}
    for section_index, section in enumerate(sections, start=1):
        section_text = normalize_text(section.text)
        section_title = normalize_text(section.title)
        section_name = normalize_text(section.section or section.title)
        content_kind = infer_content_kind(section_text, section.metadata.get("content_kind") if section.metadata else None, section.metadata)
        unit_size = CODE_CHUNK_SIZE if content_kind == "code" else DEFAULT_CHUNK_SIZE
        overlap = CODE_CHUNK_OVERLAP if content_kind == "code" else DEFAULT_CHUNK_OVERLAP
        split_results = split_text_semantic(section_text, source_type, unit_size, overlap)
        for chunk_index, content in enumerate(split_results, start=1):
            stats["chunks_before_filter"] += 1
            decision = chunk_quality_decision(content, content_kind, section_name, section_title, section.metadata)
            if decision.action == "discard":
                stats["chunks_filtered"] += 1
                continue
            snippet = content[:280].strip()
            strategy = "markdown_header_v2" if source_type in ("md", "markdown") else "sentence_aware_v2"
            chunks.append(
                {
                    "content": content,
                    "snippet": snippet,
                    "chapter": chapter,
                    "section": section_name,
                    "page_number": section.page_number,
                    "token_count": rough_token_count(content),
                    "metadata": sanitize_metadata({
                        "section_title": section_title,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                        "content_kind": content_kind,
                        "chunking_strategy": strategy,
                        **quality_metadata(decision),
                        **section.metadata,
                    }),
                }
            )
    return chunks, stats


def chunk_sections_hierarchical(sections: Iterable[ParsedSection], chapter: str, source_type: str) -> tuple[list[dict], dict[str, int]]:
    """语义切分 + 父子块结构。返回的 payload 中 is_parent 标记 parent chunks。"""
    chunks: list[dict] = []
    stats = {"chunks_before_filter": 0, "chunks_filtered": 0, "parents_created": 0, "children_created": 0}
    for section_index, section in enumerate(sections, start=1):
        section_text = normalize_text(section.text)
        section_title = normalize_text(section.title)
        section_name = normalize_text(section.section or section.title)
        content_kind = infer_content_kind(section_text, section.metadata.get("content_kind") if section.metadata else None, section.metadata)
        parent_decision = chunk_quality_decision(section_text, content_kind, section_name, section_title, section.metadata)

        parent_payload = {
            "content": section_text,
            "snippet": section_text[:280].strip(),
            "chapter": chapter,
            "section": section_name,
            "page_number": section.page_number,
            "token_count": rough_token_count(section_text),
            "metadata": sanitize_metadata({
                "section_title": section_title,
                "section_index": section_index,
                "chunk_index": 0,
                "content_kind": content_kind,
                "chunking_strategy": "hierarchical_parent_v2",
                "is_parent": True,
                **quality_metadata(parent_decision),
                **section.metadata,
            }),
            "is_parent": True,
        }
        chunks.append(parent_payload)
        stats["parents_created"] += 1

        unit_size = CODE_CHUNK_SIZE if content_kind == "code" else DEFAULT_CHUNK_SIZE
        overlap = CODE_CHUNK_OVERLAP if content_kind == "code" else DEFAULT_CHUNK_OVERLAP
        split_results = split_text_semantic(section_text, source_type, unit_size, overlap)

        for chunk_index, content in enumerate(split_results, start=1):
            stats["chunks_before_filter"] += 1
            decision = chunk_quality_decision(content, content_kind, section_name, section_title, section.metadata)
            if decision.action == "discard":
                stats["chunks_filtered"] += 1
                continue
            snippet = content[:280].strip()
            strategy = "hierarchical_child_v2"
            if source_type in ("md", "markdown"):
                strategy = "hierarchical_child_markdown_v2"
            child_payload = {
                "content": content,
                "snippet": snippet,
                "chapter": chapter,
                "section": section_name,
                "page_number": section.page_number,
                "token_count": rough_token_count(content),
                "metadata": sanitize_metadata({
                    "section_title": section_title,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "content_kind": content_kind,
                    "chunking_strategy": strategy,
                    "is_parent": False,
                    **quality_metadata(decision),
                    **section.metadata,
                }),
                "is_parent": False,
                "parent_content": section_text,
            }
            chunks.append(child_payload)
            stats["children_created"] += 1

    return chunks, stats


async def chunk_sections_hierarchical_async(sections: Iterable[ParsedSection], chapter: str, source_type: str, batch_id: str | None = None) -> tuple[list[dict], dict[str, int]]:
    """Async hierarchical chunking that can use real embeddings for long semantic splits."""
    from app.services.chunk_adaptation import choose_chunking_profile

    chunks: list[dict] = []
    stats = {"chunks_before_filter": 0, "chunks_filtered": 0, "parents_created": 0, "children_created": 0, "semantic_embedding_splits": 0, "adaptive_profiles": 0}
    for section_index, section in enumerate(sections, start=1):
        section_text = normalize_text(section.text)
        section_title = normalize_text(section.title)
        section_name = normalize_text(section.section or section.title)
        content_kind = infer_content_kind(section_text, section.metadata.get("content_kind") if section.metadata else None, section.metadata)
        profile = choose_chunking_profile(
            section_text,
            title=section_title,
            section=section_name,
            content_kind=content_kind,
            metadata=section.metadata,
        )
        if batch_id:
            from app.services.ingestion_logs import emit_ingestion_log
            emit_ingestion_log(
                batch_id,
                "chunk_adaptive",
                f"章节 [{section_title}] 动态特征分析：自适应大小={profile.chunk_size}，重叠={profile.chunk_overlap}，策略={profile.strategy} (F1得分 {profile.score:.3f})"
            )
        stats["adaptive_profiles"] += 1
        parent_decision = chunk_quality_decision(section_text, content_kind, section_name, section_title, section.metadata)

        parent_payload = {
            "content": section_text,
            "snippet": section_text[:280].strip(),
            "chapter": chapter,
            "section": section_name,
            "page_number": section.page_number,
            "token_count": rough_token_count(section_text),
            "metadata": sanitize_metadata({
                "section_title": section_title,
                "section_index": section_index,
                "chunk_index": 0,
                "content_kind": content_kind,
                "chunking_strategy": "hierarchical_parent_v2",
                "chunking_profile": profile.metadata(),
                "is_parent": True,
                **quality_metadata(parent_decision),
                **section.metadata,
            }),
            "is_parent": True,
            "parent_key": section_index,
        }
        chunks.append(parent_payload)
        stats["parents_created"] += 1

        unit_size = profile.chunk_size
        overlap = profile.chunk_overlap
        used_embedding_split = (
            profile.strategy == "semantic_or_sentence"
            and source_type not in ("md", "markdown", "ipynb", "notebook", "code")
            and get_settings().semantic_chunking_enabled
            and len(section_text) >= get_settings().semantic_chunking_min_length
        )
        if profile.strategy == "recursive_structure_preserving":
            split_results = split_text(section_text, unit_size, overlap)
        else:
            split_results = await split_text_semantic_async(section_text, source_type, unit_size, overlap)
        if used_embedding_split:
            stats["semantic_embedding_splits"] += 1

        for chunk_index, content in enumerate(split_results, start=1):
            stats["chunks_before_filter"] += 1
            decision = chunk_quality_decision(content, content_kind, section_name, section_title, section.metadata)
            if decision.action == "discard":
                stats["chunks_filtered"] += 1
                continue
            snippet = content[:280].strip()
            strategy = "hierarchical_child_semantic_embedding_v2" if used_embedding_split else "hierarchical_child_adaptive_v1"
            if source_type in ("md", "markdown"):
                strategy = "hierarchical_child_markdown_adaptive_v1"
            child_payload = {
                "content": content,
                "snippet": snippet,
                "chapter": chapter,
                "section": section_name,
                "page_number": section.page_number,
                "token_count": rough_token_count(content),
                "metadata": sanitize_metadata({
                    "section_title": section_title,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "content_kind": content_kind,
                    "chunking_strategy": strategy,
                    "chunking_profile": profile.metadata(),
                    "is_parent": False,
                    **quality_metadata(decision),
                    **section.metadata,
                }),
                "is_parent": False,
                "parent_content": section_text,
                "parent_key": section_index,
            }
            chunks.append(child_payload)
            stats["children_created"] += 1

    return chunks, stats


def chunk_sections(sections: Iterable[ParsedSection], chapter: str, source_type: str) -> list[dict]:
    return chunk_sections_with_stats(sections, chapter, source_type)[0]
