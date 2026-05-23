from __future__ import annotations

from app.services.chunking import (
    CODE_CHUNK_OVERLAP,
    CODE_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_TEXT_VERSION,
    chunk_sections,
    chunk_sections_with_stats,
    embedding_text,
    infer_content_kind,
    normalize_text,
    split_text,
)
from app.services.parsers import ParsedSection


def test_normalize_and_split_text_overlap():
    text = "alpha\x00  beta\r\n\r\n\r\n" + ("word " * 80)
    normalized = normalize_text(text)
    assert "\x00" not in normalized
    assert "\r\n" not in normalized
    assert "\n\n\n" not in normalized
    assert "alpha beta" in normalized

    chunks = split_text("0123456789" * 20, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_content_kind_and_chunk_metadata():
    sections = [
        ParsedSection(
            title="Centrality Code",
            section="Degree centrality example",
            text="[Code Cell]\n# centrality example\nprint('degree centrality for network nodes')",
            metadata={},
        ),
        ParsedSection(
            title="Text\x00",
            text="Course\x00 note text about degree centrality and network representation.",
            metadata={"content_kind": "markdown", "raw": "a\x00b"},
        ),
    ]
    chunks = chunk_sections(sections, chapter="L1", source_type="notebook")

    assert infer_content_kind("[Output]\n1") == "output"
    assert chunks[0]["metadata"]["content_kind"] == "code"
    assert chunks[1]["metadata"]["content_kind"] == "markdown"
    assert "\x00" not in chunks[1]["content"]
    assert chunks[1]["metadata"]["raw"] == "ab"
    assert all(chunk["chapter"] == "L1" for chunk in chunks)
    assert all(chunk["metadata"]["chunking_strategy"] == "chunk_800_metadata_enriched_v1" for chunk in chunks)
    assert "route_eligibility" in chunks[0]["metadata"]
    assert "retention_decision" in chunks[0]["metadata"]


def test_default_chunk_sizes_are_best_eval_strategy():
    assert DEFAULT_CHUNK_SIZE == 800
    assert DEFAULT_CHUNK_OVERLAP == 120
    assert CODE_CHUNK_SIZE == 700
    assert CODE_CHUNK_OVERLAP == 100


def test_chunk_sections_routes_noise_and_keeps_relevant_code():
    sections = [
        ParsedSection(title="Output", text="[Output]\n1\n2\n3", metadata={"content_kind": "output"}),
        ParsedSection(title="Short", text="too short", metadata={"content_kind": "markdown"}),
        ParsedSection(title="TOC", text="\n".join(["1", "2", "3", "4", "5", "6", "7", "8"]), metadata={"content_kind": "pdf_page"}),
        ParsedSection(title="Bad", text="\ufffd" * 20, metadata={"content_kind": "pdf_page"}),
        ParsedSection(title="Generic Code", text="[Code Cell]\nprint('hello world from a utility cell')", metadata={}),
        ParsedSection(title="Community Code", section="community detection", text="[Code Cell]\nprint('community detection centrality network')", metadata={}),
        ParsedSection(title="Good", text="This paragraph explains degree centrality in complex networks with enough detail.", metadata={"content_kind": "markdown"}),
    ]

    chunks, stats = chunk_sections_with_stats(sections, chapter="L1", source_type="notebook")

    assert stats["chunks_before_filter"] == 7
    assert stats["chunks_filtered"] == 1
    assert [chunk["section"] for chunk in chunks] == ["Output", "Short", "TOC", "Generic Code", "community detection", "Good"]
    assert chunks[0]["metadata"]["quality_action"] == "evidence_only"
    assert chunks[0]["metadata"]["route_eligibility"]["graph_extraction"] is False
    assert chunks[1]["metadata"]["quality_retain"] is True
    assert chunks[2]["metadata"]["quality_action"] == "summary_only"
    assert chunks[3]["metadata"]["quality_action"] == "embed_only"
    assert chunks[4]["metadata"]["quality_action"] == "graph_candidate"


def test_embedding_text_adds_metadata_without_changing_content():
    text = embedding_text(
        document_title="Centralities",
        chapter="Lecture 3",
        section="Degree",
        source_type="notebook",
        content_kind="markdown",
        content="Degree centrality counts incident edges.",
    )

    assert "Document: Centralities" in text
    assert "Chapter: Lecture 3" in text
    assert "Content Kind: markdown" in text
    assert text.endswith("Degree centrality counts incident edges.")
    assert EMBEDDING_TEXT_VERSION == "metadata_enriched_v1"


def test_continuous_chunk_adaptation_does_not_emit_fixed_document_type():
    from app.services.chunk_adaptation import choose_chunking_profile

    proof_like = (
        "Let A be a matrix. A vector v is an eigenvector if Av = lambda v. "
        "The characteristic polynomial det(A - lambda I)=0 defines eigenvalues. "
    ) * 12
    prose_like = (
        "This section explains how a course project should be planned. "
        "It connects goals, milestones, evaluation criteria, and reflection in a coherent narrative. "
    ) * 12

    proof_profile = choose_chunking_profile(proof_like, title="Eigenvalues")
    prose_profile = choose_chunking_profile(prose_like, title="Project Notes")

    assert "document_type" not in proof_profile.metadata()
    assert "document_type" not in prose_profile.metadata()
    assert proof_profile.feature_summary != prose_profile.feature_summary
    assert (proof_profile.chunk_size, proof_profile.chunk_overlap, proof_profile.strategy) != (
        prose_profile.chunk_size,
        prose_profile.chunk_overlap,
        prose_profile.strategy,
    )
