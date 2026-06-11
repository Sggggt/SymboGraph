from __future__ import annotations

import os

import pytest

from app.services.chunking import (
    CURRENT_EMBEDDING_TEXT_VERSION,
    chunk_sections_hierarchical,
    chunk_sections_semantic,
    contextual_embedding_text,
    embedding_text,
    infer_content_kind,
    split_text_sentences,
    split_text_semantic,
)
from app.services.parsers import ParsedSection


def test_split_text_sentences_preserves_boundaries():
    """句子级切分不应在句子中间截断。"""
    text = "First sentence here. Second sentence follows. Third one ends."
    chunks = split_text_sentences(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 1
    for chunk in chunks:
        # 每个 chunk 应该包含完整的句子（不以非标点结尾）
        assert chunk.strip()[-1] in ".!?。！？" or len(chunk) < 50


def test_split_text_semantic_uses_markdown_splitter_for_md():
    """Markdown 文件应使用标题切分。"""
    text = "# Heading 1\nContent under h1.\n\n## Heading 2\nContent under h2."
    chunks = split_text_semantic(text, source_type="md")
    assert len(chunks) >= 2
    assert any("Heading 1" in c for c in chunks)
    assert any("Heading 2" in c for c in chunks)


def test_split_text_semantic_falls_back_to_sentences_for_pdf():
    """PDF 应使用句子级切分。"""
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = split_text_semantic(text, source_type="pdf")
    assert len(chunks) >= 1
    # 不应在句子中间截断
    for chunk in chunks:
        assert not chunk.strip().endswith("Sent")


def test_chunk_sections_hierarchical_creates_parents_and_children():
    """父子块结构：每个 section 产生 1 个 parent + N 个 children。"""
    sections = [
        ParsedSection(
            title="Intro",
            text="First paragraph. Second paragraph with more content. Third paragraph.",
            metadata={},
        ),
        ParsedSection(
            title="Methods",
            text="Method A description. Method B description.",
            metadata={},
        ),
    ]
    chunks, stats = chunk_sections_hierarchical(sections, partition="L1", source_type="pdf")

    assert stats["parents_created"] == 2
    assert stats["children_created"] >= 2

    parents = [c for c in chunks if c.get("is_parent")]
    children = [c for c in chunks if not c.get("is_parent")]

    assert len(parents) == 2
    assert len(children) >= 2

    # parent 应包含完整 section text
    for parent, section in zip(parents, sections):
        assert section.text in parent["content"]

    # child 应有 parent_content 引用
    for child in children:
        assert "parent_content" in child
        assert child["parent_content"] in [p["content"] for p in parents]


def test_chunk_sections_semantic_uses_different_strategies():
    """语义切分应对不同 source_type 使用不同策略。"""
    sections = [
        ParsedSection(
            title="Markdown Doc",
            text="# Title\n\nParagraph one with enough content to pass the filter. Paragraph two also has sufficient length for the chunking system.",
            metadata={},
        ),
    ]
    chunks_md, _ = chunk_sections_semantic(sections, partition="L1", source_type="md")
    chunks_pdf, _ = chunk_sections_semantic(sections, partition="L1", source_type="pdf")

    assert chunks_md[0]["metadata"]["chunking_strategy"] == "markdown_header_v2"
    assert chunks_pdf[0]["metadata"]["chunking_strategy"] == "sentence_aware_v2"


def test_contextual_embedding_text_includes_context():
    """上下文增强嵌入应包含 parent/prev/next 摘要。"""
    text = contextual_embedding_text(
        document_title="Networks",
        partition="L2",
        section="Centrality",
        source_type="pdf",
        content_kind="text",
        content="Degree centrality counts edges.",
        parent_summary="This section discusses centrality measures.",
        prev_summary="Previous chunk about graph basics.",
        next_summary="Next chunk about betweenness centrality.",
    )
    assert "Document: Networks" in text
    assert "Context:" in text
    assert "centrality measures" in text
    assert "Previous:" in text
    assert "Next:" in text
    assert text.endswith("Degree centrality counts edges.")


def test_embedding_text_enhanced_with_summary_keywords():
    """增强版 embedding_text 应包含 summary、keywords、table/formula 标记。"""
    text = embedding_text(
        document_title="Stats",
        partition="L3",
        section="Bayes",
        source_type="pdf",
        content_kind="text",
        content="Bayes theorem formula.",
        summary="Overview of Bayes theorem.",
        keywords=["Bayes", "probability"],
        has_table=True,
        has_formula=True,
    )
    assert "Summary: Overview of Bayes theorem." in text
    assert "Keywords: Bayes, probability" in text
    assert "Contains: table" in text
    assert "Contains: formula" in text


def test_infer_content_kind_detects_table_and_formula():
    """infer_content_kind 应识别表格和公式类型。"""
    assert infer_content_kind("text", metadata={"has_table": True}) == "table"
    assert infer_content_kind("text", metadata={"has_formula": True}) == "formula"
    assert infer_content_kind("text", metadata={}) == "text"
    assert infer_content_kind("[Code Cell]\nprint()") == "code"
    assert infer_content_kind("[Output]\n1") == "output"


@pytest.mark.skipif(
    os.getenv("RUN_NO_FALLBACK_E2E") != "1",
    reason="Requires Docker Compose with PostgreSQL, Qdrant, and real API endpoint",
)
def test_hierarchical_retrieval_integration():
    """端到端测试占位符：验证父子块检索在真实环境中的行为。"""
    pass
