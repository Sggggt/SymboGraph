from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select


@pytest.mark.parametrize(
    ("line", "title"),
    [
        ("# Heading", "Heading"),
        (" ## Heading", "Heading"),
        ("  ### Heading ###", "Heading"),
        ("   #### Heading ####   ", "Heading"),
        ("#####\tTabbed\t#####", "Tabbed"),
        ("######", ""),
        ("### ###", ""),
    ],
    ids=[
        "level-one",
        "level-two-one-space-indent",
        "level-three-closing",
        "level-four-three-space-indent",
        "tab-separator-and-closing",
        "empty-level-six",
        "closing-only-content",
    ],
)
def test_gfm_atx_heading_lexer_accepts_only_normative_openings(
    line: str,
    title: str,
):
    from app.services.parsers import _markdown_atx_heading_title

    assert _markdown_atx_heading_title(line) == title


@pytest.mark.parametrize(
    "line",
    [
        "#ID | QPS",
        "#hashtag",
        r"\# Escaped opening",
        "####### Seven hashes",
        "    # Four-space indent",
        "\t# Tab indent",
        "##No separator",
        "prose # fragment",
    ],
    ids=[
        "hash-identifier",
        "hashtag",
        "escaped-opening",
        "seven-hashes",
        "four-space-indent",
        "tab-indent",
        "no-separator",
        "prose-fragment",
    ],
)
def test_gfm_atx_heading_lexer_rejects_non_headings(line: str):
    from app.services.parsers import _markdown_atx_heading_title

    assert _markdown_atx_heading_title(line) is None


@pytest.mark.parametrize(
    "header",
    [
        "#ID | QPS",
        "#Issue | QPS",
        "#Tag | QPS",
        "#1 | QPS",
        "####### | QPS",
        r"\#Escaped | QPS",
    ],
    ids=[
        "id",
        "issue",
        "tag",
        "number",
        "seven-hashes",
        "escaped-hash",
    ],
)
def test_hash_content_remains_available_to_markdown_table_scanner(
    tmp_path: Path,
    header: str,
):
    from app.services.parsers import (
        MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION,
        parse_document,
    )

    source_path = tmp_path / "hash-table.md"
    source_path.write_text(
        f"{header}\n"
        "--- | ---:\n"
        "east | 1200\n",
        encoding="utf-8",
    )

    source_type, sections = parse_document(source_path)

    assert source_type == "markdown"
    assert len(sections) == 1
    assert [item.object_type for item in sections[0].structure_objects] == ["table"]
    table = sections[0].structure_objects[0]
    assert table.text.startswith(header)
    assert (
        table.metadata["structure_protocol_version"]
        == MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION
    )
    assert table.bbox == {}


def test_legal_atx_levels_and_closing_sequences_create_exact_sections(
    tmp_path: Path,
):
    from app.services.parsers import parse_document

    source_path = tmp_path / "headings.md"
    source_path.write_text(
        "# Alpha #\n"
        "alpha body\n\n"
        "  ## Beta ##   \n"
        "beta body\n\n"
        "###### Zeta ######\n"
        "zeta body\n",
        encoding="utf-8",
    )

    _source_type, sections = parse_document(source_path)

    assert [section.title for section in sections] == ["Alpha", "Beta", "Zeta"]
    assert [section.text for section in sections] == [
        "alpha body",
        "beta body",
        "zeta body",
    ]
    assert [
        item.path
        for section in sections
        for item in section.structure_objects
    ] == [
        "Alpha / paragraph:1",
        "Beta / paragraph:1",
        "Zeta / paragraph:1",
    ]


def test_markdown_section_scanner_preserves_three_vs_four_space_atx_boundary(
    tmp_path: Path,
):
    from app.services.parsers import parse_document

    source_path = tmp_path / "atx-indent-boundary.md"
    source_path.write_text(
        "   # Three-space heading\n"
        "three body\n"
        "    # Four-space non-heading\n"
        "four body\n"
        "####### Seven-hash non-heading\n"
        "seven body\n"
        "# Final heading\n"
        "final body\n",
        encoding="utf-8",
    )

    _source_type, sections = parse_document(source_path)

    assert [section.title for section in sections] == [
        "Three-space heading",
        "Final heading",
    ]
    assert "Four-space non-heading" in sections[0].text
    assert "Seven-hash non-heading" in sections[0].text
    assert sections[1].text == "final body"


def test_closing_sequence_does_not_strip_hash_content_or_escaped_hashes():
    from app.services.parsers import _markdown_atx_heading_title

    assert _markdown_atx_heading_title("# Literal #tag") == "Literal #tag"
    assert _markdown_atx_heading_title(r"# Escaped \###") == r"Escaped \###"
    assert _markdown_atx_heading_title("# Tight###") == "Tight###"
    assert _markdown_atx_heading_title("# Content # #") == "Content #"


def test_fence_precedes_heading_and_table_scanners_then_hash_table_survives(
    tmp_path: Path,
):
    from app.services.chunking import FixedTokenChunker
    from app.services.parsers import parse_document

    source_path = tmp_path / "fence-then-hash-table.md"
    source_path.write_text(
        "# Capacity\n\n"
        "````markdown\n"
        "# hidden heading\n"
        "#Fake | QPS\n"
        "--- | ---:\n"
        "east | 9999\n"
        "```\n"
        "still fenced\n"
        "````\n"
        "Table 1: grounded values\n"
        "#ID | QPS\n"
        "--- | ---:\n"
        "west | 900\n\n"
        "Grounded prose after the table.\n",
        encoding="utf-8",
    )

    _source_type, sections = parse_document(source_path)

    assert len(sections) == 1
    structures = sections[0].structure_objects
    assert [item.object_type for item in structures] == [
        "code_block",
        "caption",
        "table",
        "paragraph",
    ]
    assert "# hidden heading" in structures[0].text
    assert "#Fake | QPS" in structures[0].text
    assert structures[2].text.startswith("#ID | QPS")
    assert structures[2].path == "Capacity / table:3"
    assert all(item.bbox == {} for item in structures)

    prepared = FixedTokenChunker(chunk_size=128, overlap=8).prepare_document(
        sections,
        title="Fence and table",
    )
    assert [item.object_type for item in prepared.structure_objects] == [
        "code_block",
        "caption",
        "table",
        "paragraph",
    ]
    assert prepared.structure_objects[2].path == "Capacity / table:3"


def test_hash_prefixed_table_reaches_prepared_document_and_g0(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
):
    from app.models import (
        ChunkCoordinate,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )
    from app.services.chunking import FixedTokenChunker
    from app.services.context_graph import write_chunks_and_structure
    from app.services.parsers import (
        MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION,
        parse_document,
    )

    source_path = tmp_path / "hash-prefixed-table.md"
    source_path.write_text(
        "# Capacity\n\n"
        "#ID | QPS\n"
        "--- | ---:\n"
        "east | 1200\n",
        encoding="utf-8",
    )
    _source_type, sections = parse_document(source_path)
    parsed_table = sections[0].structure_objects[0]
    assert parsed_table.object_type == "table"
    assert parsed_table.path == "Capacity / table:1"
    assert parsed_table.bbox == {}

    prepared = FixedTokenChunker(chunk_size=64, overlap=8).prepare_document(
        sections,
        title="Hash table",
    )
    assert len(prepared.structure_objects) == 1
    prepared_table = prepared.structure_objects[0]
    assert prepared_table.object_type == "table"
    assert prepared_table.path == parsed_table.path
    assert prepared_table.bbox == {}
    assert prepared.text[
        prepared_table.char_start : prepared_table.char_end
    ] == prepared_table.text

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Hash table",
        source_path=str(source_path),
        source_type="markdown",
        checksum="markdown_table-atx-hash-table",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=str(source_path),
        parse_protocol_version="markdown_table_markdown_atx_heading_v1",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()

    chunks = write_chunks_and_structure(
        db_session,
        knowledge_base=sample_knowledge_base,
        document=document,
        version=version,
        sections=sections,
        chunk_version=1,
        chunk_size=64,
        chunk_overlap=8,
    )
    db_session.flush()

    nodes = list(
        db_session.scalars(
            select(ChunkStructureNode).where(
                ChunkStructureNode.document_version_id == version.id
            )
        ).all()
    )
    coordinates = list(
        db_session.scalars(
            select(ChunkCoordinate).where(
                ChunkCoordinate.document_version_id == version.id
            )
        ).all()
    )
    table_nodes = [node for node in nodes if node.node_type == "table"]
    assert chunks
    assert len(table_nodes) == 1
    assert table_nodes[0].path.endswith("/ table:1")
    assert table_nodes[0].layout_json["parser_path"] == parsed_table.path
    assert table_nodes[0].layout_json["metadata"][
        "structure_protocol_version"
    ] == MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION
    assert all(node.bbox_json == {} for node in nodes)
    assert all(row.bbox_json == {} for row in coordinates)
    assert all(row.coordinate_system == "text_flow_v1" for row in coordinates)
