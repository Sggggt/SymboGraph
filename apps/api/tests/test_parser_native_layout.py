from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select


def _native_bbox(*, page: int, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "page_number": page,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "coordinate_system": "normalized_page_v1",
        "synthetic": False,
        "raw_coordinate_system": "unit_fixture",
        "page_size": [1000.0, 1000.0],
    }


def test_two_column_pdf_preserves_parser_native_blocks(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    from app.services.parsers import PARSER_LAYOUT_PROTOCOL_VERSION, parse_pdf

    pdf_path = tmp_path / "two-column.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_textbox(fitz.Rect(36, 60, 260, 240), "LEFT COLUMN\nalpha evidence\nleft conclusion", fontsize=12)
    page.insert_textbox(fitz.Rect(340, 60, 564, 240), "RIGHT COLUMN\nbeta evidence\nright conclusion", fontsize=12)
    document.save(pdf_path)
    document.close()

    sections = parse_pdf(pdf_path)

    assert len(sections) == 1
    section = sections[0]
    assert section.parser_metadata["layout_protocol_version"] == PARSER_LAYOUT_PROTOCOL_VERSION
    assert section.parser_metadata["native_layout_available"] is True
    assert len(section.layout_items) >= 2
    assert len(section.structure_objects) >= 2
    native_boxes = [item.bbox for item in section.layout_items]
    assert all(box and box["synthetic"] is False for box in native_boxes)
    assert all(box["coordinate_system"] == "normalized_page_v1" for box in native_boxes)
    assert min(box["x0"] for box in native_boxes) < 0.2
    assert max(box["x0"] for box in native_boxes) > 0.5
    assert all(item.char_end > item.char_start for item in section.layout_items)


def test_scanned_image_uses_ocr_native_regions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from app.services import parsers
    from PIL import Image

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (2, 2), color="white").save(image_path, format="PNG")
    rows = [
        {
            "text": "upper scanned line",
            "bbox": _native_bbox(page=1, x0=0.1, y0=0.1, x1=0.8, y1=0.2),
            "confidence": 0.96,
        },
        {
            "text": "lower scanned line",
            "bbox": _native_bbox(page=1, x0=0.1, y0=0.6, x1=0.8, y1=0.7),
            "confidence": 0.91,
        },
    ]
    monkeypatch.setattr(
        parsers,
        "ocr_image_to_text",
        lambda _path, source_name: (
            "upper scanned line\nlower scanned line",
            {
                "ocr_engine": "unit_ocr",
                "ocr_confidence": 0.935,
                "ocr_layout_items": rows,
                "image_size": [1000, 1000],
                "source_name": source_name,
            },
        ),
    )

    section = parsers.parse_image(image_path)[0]

    assert section.parser_metadata["native_layout_available"] is True
    assert [item.reading_order for item in section.layout_items] == [0, 1]
    assert [item.bbox["y0"] for item in section.layout_items] == [0.1, 0.6]
    assert all(item.metadata["native_geometry"] is True for item in section.layout_items)
    assert len(section.structure_objects) == 2


def test_g0_persists_parser_coordinates_and_native_structure(db_session, sample_knowledge_base):
    from app.models import (
        ChunkCoordinate,
        ChunkStructureMapping,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )
    from app.services.context_graph import STRUCTURE_MAPPING_PROTOCOL_VERSION, write_chunks_and_structure
    from app.services.parsers import ParsedLayoutItem, ParsedSection, ParsedStructureObject

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Native layout fixture",
        source_path="native-layout.pdf",
        source_type="pdf",
        checksum="native-layout",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    text = "Left-column grounded statement.\n\nRight-column grounded statement."
    separator = text.index("Right-column")
    left_bbox = _native_bbox(page=1, x0=0.05, y0=0.1, x1=0.45, y1=0.4)
    right_bbox = _native_bbox(page=1, x0=0.55, y0=0.1, x1=0.95, y1=0.4)
    section = ParsedSection(
        title="Two columns",
        section="Two columns",
        text=text,
        page_number=1,
        layout_items=[
            ParsedLayoutItem("left-layout", text[: separator - 2], 0, separator - 2, 1, left_bbox, "normalized_page_v1", "paragraph", 0),
            ParsedLayoutItem("right-layout", text[separator:], separator, len(text), 1, right_bbox, "normalized_page_v1", "paragraph", 1),
        ],
        structure_objects=[
            ParsedStructureObject("left-structure", "paragraph", text[: separator - 2], 0, separator - 2, "Left", 1, left_bbox, "normalized_page_v1", 0, path="Two columns / Left"),
            ParsedStructureObject("right-structure", "paragraph", text[separator:], separator, len(text), "Right", 1, right_bbox, "normalized_page_v1", 1, path="Two columns / Right"),
        ],
        parser_metadata={"parser": "unit_native", "native_layout_available": True},
    )

    chunks = write_chunks_and_structure(
        db_session,
        knowledge_base=sample_knowledge_base,
        document=document,
        version=version,
        sections=[section],
        chunk_version=1,
        chunk_size=128,
        chunk_overlap=8,
    )
    db_session.flush()

    coordinates = db_session.scalars(select(ChunkCoordinate).where(ChunkCoordinate.chunk_id == chunks[0].id)).all()
    nodes = db_session.scalars(select(ChunkStructureNode).where(ChunkStructureNode.document_version_id == version.id)).all()
    mappings = db_session.scalars(select(ChunkStructureMapping).where(ChunkStructureMapping.chunk_id == chunks[0].id)).all()
    assert {row.page_range_json["layout_id"] for row in coordinates} == {
        "section:0:left-layout",
        "section:0:right-layout",
    }
    assert all(row.bbox_json and row.bbox_json["synthetic"] is False for row in coordinates)
    assert any(node.node_type == "region" and node.bbox_json for node in nodes)
    assert {node.layout_json.get("structure_id") for node in nodes if node.depth == 3} == {
        "section:0:left-structure",
        "section:0:right-structure",
    }
    assert all(not node.bbox_json.get("synthetic", False) for node in nodes)
    assert any(mapping.bbox_iou is not None for mapping in mappings)
    assert all(mapping.mapping_protocol_version == STRUCTURE_MAPPING_PROTOCOL_VERSION for mapping in mappings)
    assert any((mapping.metadata_json or {}).get("chunk_layout_ids") for mapping in mappings)


def test_mapping_weight_uses_bbox_and_path_not_only_span(db_session, sample_knowledge_base):
    from app.models import Chunk, ChunkStructureMapping, ChunkStructureNode, Document, DocumentVersion
    from app.services.chunking import text_hash
    from app.services.context_graph import STRUCTURE_MAPPING_PROTOCOL_VERSION, write_structure_mappings_for_chunk

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Mapping fixture",
        source_path="mapping.pdf",
        source_type="pdf",
        checksum="mapping",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version=1, checksum="mapping", storage_path="mapping.pdf", is_active=True)
    db_session.add(version)
    db_session.flush()
    left_bbox = _native_bbox(page=1, x0=0.05, y0=0.1, x1=0.45, y1=0.4)
    right_bbox = _native_bbox(page=1, x0=0.55, y0=0.1, x1=0.95, y1=0.4)
    chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=10,
        char_start=0,
        char_end=20,
        text="same character span",
        text_hash=text_hash("same character span"),
        section_path="Evidence / Left",
        page_start=1,
        page_end=1,
        metadata_json={
            "layout_coordinates": [
                {
                    "layout_id": "left-layout",
                    "page_number": 1,
                    "bbox": left_bbox,
                    "coordinate_system": "normalized_page_v1",
                }
            ]
        },
        state="active",
    )
    db_session.add(chunk)
    left = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="paragraph",
        depth=3,
        title="Left",
        char_start=0,
        char_end=20,
        page_number=1,
        bbox_json=left_bbox,
        layout_json={"coordinate_system": "normalized_page_v1", "structure_id": "left"},
        path="Mapping fixture / Evidence / Left",
    )
    right = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="paragraph",
        depth=3,
        title="Right",
        char_start=0,
        char_end=20,
        page_number=1,
        bbox_json=right_bbox,
        layout_json={"coordinate_system": "normalized_page_v1", "structure_id": "right"},
        path="Mapping fixture / Evidence / Right",
    )
    db_session.add_all([left, right])
    db_session.flush()

    write_structure_mappings_for_chunk(db_session, chunk=chunk, nodes=[left, right])
    db_session.flush()
    mappings = {
        row.structure_node_id: row
        for row in db_session.scalars(select(ChunkStructureMapping).where(ChunkStructureMapping.chunk_id == chunk.id)).all()
    }

    assert mappings[left.id].span_overlap == mappings[right.id].span_overlap == 1.0
    assert mappings[left.id].bbox_iou == 1.0
    assert mappings[right.id].bbox_iou == 0.0
    assert mappings[left.id].path_match == 1.0
    assert mappings[right.id].path_match == 0.5
    assert mappings[left.id].mapping_weight > mappings[right.id].mapping_weight
    assert mappings[left.id].mapping_protocol_version == STRUCTURE_MAPPING_PROTOCOL_VERSION
    assert mappings[right.id].bbox_iou is not None
    assert mappings[left.id].metadata_json["effective_weights"] == {
        "span_overlap": pytest.approx(0.55),
        "bbox_iou": pytest.approx(0.30),
        "path_match": pytest.approx(0.15),
    }


def _mapping_admission_fixture(db_session, sample_knowledge_base):
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Admission fixture",
        source_path="admission.pdf",
        source_type="pdf",
        checksum="admission",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="admission",
        storage_path="admission.pdf",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=10,
        char_start=0,
        char_end=20,
        text="grounded chunk text",
        text_hash=text_hash("grounded chunk text"),
        section_path="Parent / Exact section",
        page_start=1,
        page_end=1,
        metadata_json={},
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    return document, version, chunk


@pytest.mark.parametrize(
    "node_type",
    [
        "paragraph",
        "list",
        "table",
        "formula",
        "caption",
        "code_block",
        "page",
        "region",
        "future_leaf",
    ],
)
def test_mapping_admission_rejects_path_only_leaf_and_layout_nodes(
    db_session,
    sample_knowledge_base,
    node_type,
):
    from app.models import ChunkStructureMapping, ChunkStructureNode
    from app.services.context_graph import write_structure_mappings_for_chunk

    document, version, chunk = _mapping_admission_fixture(
        db_session,
        sample_knowledge_base,
    )
    node = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type=node_type,
        depth=3,
        title=node_type,
        char_start=100,
        char_end=120,
        path=f"Admission fixture / Parent / Exact section / {node_type}",
        bbox_json={},
        layout_json={"coordinate_system": "text_flow_v1"},
    )
    db_session.add(node)
    db_session.flush()

    write_structure_mappings_for_chunk(db_session, chunk=chunk, nodes=[node])
    db_session.flush()

    assert db_session.scalar(
        select(ChunkStructureMapping).where(
            ChunkStructureMapping.chunk_id == chunk.id,
            ChunkStructureMapping.structure_node_id == node.id,
        )
    ) is None


def test_mapping_admission_rejects_ambiguous_repeated_exact_section_path(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkStructureMapping, ChunkStructureNode
    from app.services.context_graph import write_structure_mappings_for_chunk

    document, version, chunk = _mapping_admission_fixture(
        db_session,
        sample_knowledge_base,
    )
    sections = [
        ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="section",
            depth=2,
            title="Exact section",
            char_start=None,
            char_end=None,
            path="Admission fixture / Parent / Exact section",
            bbox_json={},
            layout_json={"coordinate_system": "text_flow_v1"},
        )
        for _index in range(2)
    ]
    db_session.add_all(sections)
    db_session.flush()

    write_structure_mappings_for_chunk(db_session, chunk=chunk, nodes=sections)
    db_session.flush()

    assert db_session.scalars(
        select(ChunkStructureMapping).where(
            ChunkStructureMapping.chunk_id == chunk.id
        )
    ).all() == []


def test_mapping_admission_allows_unique_exact_section_only_when_span_and_bbox_unavailable(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkStructureMapping, ChunkStructureNode
    from app.services.context_graph import (
        STRUCTURE_MAPPING_ADMISSION_PROTOCOL_VERSION,
        STRUCTURE_MAPPING_PROTOCOL_VERSION,
        write_structure_mappings_for_chunk,
    )

    document, version, chunk = _mapping_admission_fixture(
        db_session,
        sample_knowledge_base,
    )
    unique = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="section",
        depth=2,
        title="Exact section",
        char_start=None,
        char_end=None,
        path="Admission fixture / Parent / Exact section",
        bbox_json={},
        layout_json={"coordinate_system": "text_flow_v1"},
    )
    numeric_zero = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="section",
        depth=2,
        title="Other exact section",
        char_start=100,
        char_end=120,
        path="Admission fixture / Parent / Other exact section",
        bbox_json={},
        layout_json={"coordinate_system": "text_flow_v1"},
    )
    db_session.add_all([unique, numeric_zero])
    db_session.flush()

    write_structure_mappings_for_chunk(
        db_session,
        chunk=chunk,
        nodes=[unique, numeric_zero],
    )
    db_session.flush()

    mappings = db_session.scalars(
        select(ChunkStructureMapping).where(
            ChunkStructureMapping.chunk_id == chunk.id
        )
    ).all()
    assert [mapping.structure_node_id for mapping in mappings] == [unique.id]
    mapping = mappings[0]
    assert mapping.mapping_protocol_version == STRUCTURE_MAPPING_PROTOCOL_VERSION
    assert mapping.mapping_weight == pytest.approx(1.0)
    assert mapping.metadata_json["available_components"] == ["path_match"]
    assert mapping.metadata_json["unavailable_components"] == [
        "bbox_iou",
        "span_overlap",
    ]
    assert mapping.metadata_json["mapping_admission_protocol_version"] == (
        STRUCTURE_MAPPING_ADMISSION_PROTOCOL_VERSION
    )
    assert mapping.metadata_json["admission"]["reason"] == (
        "unique_exact_section_path_fallback"
    )


def test_mapping_admission_rejects_cross_document_even_with_span_overlap(
    db_session,
    sample_knowledge_base,
):
    from app.models import (
        ChunkStructureMapping,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )
    from app.services.context_graph import write_structure_mappings_for_chunk

    _document, _version, chunk = _mapping_admission_fixture(
        db_session,
        sample_knowledge_base,
    )
    foreign_document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Foreign",
        source_path="foreign.pdf",
        source_type="pdf",
        checksum="foreign",
        tags=[],
        is_active=True,
    )
    db_session.add(foreign_document)
    db_session.flush()
    foreign_version = DocumentVersion(
        document_id=foreign_document.id,
        version=1,
        checksum="foreign",
        storage_path="foreign.pdf",
        is_active=True,
    )
    db_session.add(foreign_version)
    db_session.flush()
    foreign_node = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=foreign_document.id,
        document_version_id=foreign_version.id,
        node_type="document",
        depth=0,
        title="Foreign",
        char_start=0,
        char_end=20,
        path="Foreign",
        bbox_json={},
        layout_json={},
    )
    db_session.add(foreign_node)
    db_session.flush()

    write_structure_mappings_for_chunk(
        db_session,
        chunk=chunk,
        nodes=[foreign_node],
    )
    db_session.flush()

    assert db_session.scalar(
        select(ChunkStructureMapping).where(
            ChunkStructureMapping.chunk_id == chunk.id
        )
    ) is None


def test_mapping_admission_keeps_same_scope_document_container(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkStructureMapping, ChunkStructureNode
    from app.services.context_graph import write_structure_mappings_for_chunk

    document, version, chunk = _mapping_admission_fixture(
        db_session,
        sample_knowledge_base,
    )
    document_node = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="document",
        depth=0,
        title="Admission fixture",
        char_start=None,
        char_end=None,
        path="Admission fixture",
        bbox_json={},
        layout_json={},
    )
    db_session.add(document_node)
    db_session.flush()

    write_structure_mappings_for_chunk(
        db_session,
        chunk=chunk,
        nodes=[document_node],
    )
    db_session.flush()

    mapping = db_session.scalar(
        select(ChunkStructureMapping).where(
            ChunkStructureMapping.chunk_id == chunk.id
        )
    )
    assert mapping is not None
    assert mapping.metadata_json["admission"]["reason"] == (
        "same_scope_document_container"
    )


@pytest.mark.parametrize(
    ("table_text", "column_count", "data_row_count"),
    [
        (
            "| Region | QPS |\n| --- | --- |\n| east | 1200 |",
            2,
            1,
        ),
        (
            "Region | QPS\n:--- | ---:\neast | 1200\nwest | 900",
            2,
            2,
        ),
        (
            "| Name | Notes\n| :---: | ---:\neast | escaped \\| value |",
            2,
            1,
        ),
    ],
    ids=["leading-and-trailing", "no-edge-pipes-alignment", "mixed-edge-pipes-escaped"],
)
def test_markdown_pipe_table_is_one_versioned_flow_object(
    tmp_path: Path,
    table_text: str,
    column_count: int,
    data_row_count: int,
):
    from app.services.parsers import (
        FLOW_BLOCK_PROTOCOL_VERSION,
        MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION,
        parse_document,
    )

    source_path = tmp_path / "table.md"
    source_path.write_text(f"# Capacity\n\n{table_text}", encoding="utf-8")

    source_type, sections = parse_document(source_path)

    assert source_type == "markdown"
    assert len(sections) == 1
    section = sections[0]
    assert section.parser_metadata["flow_block_protocol_version"] == (
        FLOW_BLOCK_PROTOCOL_VERSION
    )
    assert len(section.layout_items) == 1
    assert len(section.structure_objects) == 1
    layout = section.layout_items[0]
    structure = section.structure_objects[0]
    assert layout.region_type == structure.object_type == "table"
    assert layout.text == structure.text == section.text
    assert (layout.char_start, layout.char_end) == (0, len(section.text))
    assert (structure.char_start, structure.char_end) == (0, len(section.text))
    assert layout.bbox == structure.bbox == {}
    assert layout.coordinate_system == structure.coordinate_system == "text_flow_v1"
    assert structure.path == "Capacity / table:1"
    assert structure.metadata["flow_block_protocol_version"] == FLOW_BLOCK_PROTOCOL_VERSION
    assert structure.metadata["structure_protocol_version"] == MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION
    assert structure.metadata["column_count"] == column_count
    assert structure.metadata["data_row_count"] == data_row_count


@pytest.mark.parametrize(
    ("body", "expected_type"),
    [
        ("The alternatives are east | west in ordinary prose.", "paragraph"),
        ("| Region | QPS |", "paragraph"),
        ("| Region | QPS |\n| east | 1200 |", "paragraph"),
        ("| Region | QPS |\n| --- | not-a-delimiter |", "paragraph"),
        ("Table 1: Capacity | QPS", "caption"),
        (
            "```markdown\n| Region | QPS |\n| --- | --- |\n| east | 1200 |\n```",
            "code_block",
        ),
        (
            "~~~markdown\n# not a section\n| Region | QPS |\n| --- | --- |\n| east | 1200 |\n~~~",
            "code_block",
        ),
    ],
    ids=[
        "pipe-prose",
        "single-row",
        "missing-delimiter",
        "invalid-delimiter",
        "caption",
        "backtick-fence",
        "tilde-fence-with-heading",
    ],
)
def test_markdown_non_tables_are_not_promoted_to_table(
    tmp_path: Path,
    body: str,
    expected_type: str,
):
    from app.services.parsers import parse_document

    source_path = tmp_path / "not-table.md"
    source_path.write_text(f"# Capacity\n\n{body}", encoding="utf-8")

    _source_type, sections = parse_document(source_path)

    assert len(sections) == 1
    object_types = [
        item.object_type
        for section in sections
        for item in section.structure_objects
    ]
    assert object_types == [expected_type]
    assert "table" not in object_types


def test_markdown_caption_table_and_prose_have_distinct_flow_spans(tmp_path: Path):
    from app.services.parsers import MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION, parse_document

    source_path = tmp_path / "adjacent-table.md"
    source_path.write_text(
        "# Capacity\n\n"
        "Table 1: regional capacity\n"
        "| Region | QPS |\n"
        "| :--- | ---: |\n"
        "| east | 1200 |\n\n"
        "The table is grounded in the source flow.",
        encoding="utf-8",
    )

    _source_type, sections = parse_document(source_path)

    assert len(sections) == 1
    structures = sections[0].structure_objects
    assert [item.object_type for item in structures] == [
        "caption",
        "table",
        "paragraph",
    ]
    table = structures[1]
    assert table.text == (
        "| Region | QPS |\n"
        "| :--- | ---: |\n"
        "| east | 1200 |"
    )
    assert table.metadata["structure_protocol_version"] == MARKDOWN_PIPE_TABLE_PROTOCOL_VERSION
    assert table.path == "Capacity / table:2"
    assert all(not item.bbox for item in sections[0].layout_items)
    assert all(not item.bbox for item in structures)


def test_markdown_table_path_reaches_prepared_document_and_g0_without_geometry(
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

    source_path = tmp_path / "flow-table.md"
    source_path.write_text(
        "# Capacity\n\n"
        "| Region | QPS |\n"
        "| --- | ---: |\n"
        "| east | 1200 |\n\n"
        "The table is addressable in source flow.",
        encoding="utf-8",
    )
    _source_type, sections = parse_document(source_path)
    parsed_table = next(
        item
        for section in sections
        for item in section.structure_objects
        if item.object_type == "table"
    )
    prepared = FixedTokenChunker(chunk_size=128, overlap=8).prepare_document(
        sections,
        title="Flow table",
    )
    prepared_table = next(
        item for item in prepared.structure_objects if item.object_type == "table"
    )
    assert prepared_table.path == parsed_table.path == "Capacity / table:1"
    assert prepared_table.bbox == {}

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Flow table",
        source_path=str(source_path),
        source_type="markdown",
        checksum="markdown_table-flow-table",
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
        parse_protocol_version="markdown_table_markdown_pipe_table_fixture_v1",
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
        chunk_size=128,
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
