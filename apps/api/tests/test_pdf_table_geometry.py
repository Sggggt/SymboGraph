from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.services import parsers


def ruled_pdf(tmp_path):
    import fitz
    path = tmp_path / "unit-test-ruled-table.pdf"
    with fitz.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 80), "Table 1: Synthetic calibration")
        for x in (40, 200, 280, 530):
            page.draw_line((x, 100), (x, 235))
        for y in (100, 135, 235):
            page.draw_line((40, y), (530, y))
        for x, title in ((45, "Target"), (205, "Hours"), (285, "Purpose")):
            page.insert_text((x, 120), title)
        for y, name, value in ((160, "Alpha", "2.5"), (195, "Beta", "4.0")):
            for x, text in ((45, name), (205, value), (285, "Calibration")):
                page.insert_text((x, y), text)
        page.insert_text((40, 270), "Outside paragraph is not tabular evidence.")
        expected_text = "\n\n".join(str(b[4]).strip() for b in page.get_text("blocks", sort=True)
            if len(b) > 6 and b[6] == 0 and str(b[4]).strip())
        document.save(path)
    return path, expected_text


def test_ruled_pdf_adds_table_address_without_changing_text_or_caption(tmp_path):
    from app.schemas import ContextStructureNativeMetadataAudit
    path, expected = ruled_pdf(tmp_path)
    original = path.read_bytes()
    section = parsers.parse_pdf(path)[0]
    tables = [obj for obj in section.structure_objects if obj.object_type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert "Alpha" in table.text and "2.5" in table.text and "Beta" in table.text
    assert "Table 1" not in table.text and "Outside paragraph" not in table.text
    assert section.text == expected
    assert section.text[table.char_start:table.char_end] == table.text
    assert path.read_bytes() == original
    assert table.bbox["synthetic"] is False
    audit = ContextStructureNativeMetadataAudit.model_validate(table.metadata)
    assert audit.table_geometry.column_count == 3
    assert audit.table_geometry.row_band_count >= 2


def test_table_fragments_do_not_absorb_interleaved_non_table_text():
    import fitz
    raw = "Alpha\n\nOutside\n\nBeta"
    layouts = []
    for order, (word, box) in enumerate((("Alpha", [10, 10, 30, 20]),
            ("Outside", [120, 10, 180, 20]), ("Beta", [10, 30, 30, 40]))):
        start = raw.index(word)
        layouts.append(parsers.ParsedLayoutItem(str(order), word, start, start + len(word),
            bbox={"raw_bbox": box}, reading_order=order,
            metadata={"parser_source": "pymupdf_text_block", "source_index": order}))
    table = SimpleNamespace(bbox=[0, 0, 100, 100], row_count=2, col_count=2)
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 200),
        find_tables=lambda **kwargs: SimpleNamespace(tables=[table]))
    objects = parsers._pdf_table_objects(page, raw, layouts, page_number=1)
    assert [item.text for item in objects] == ["Alpha", "Beta"]
    assert all(raw[item.char_start:item.char_end] == item.text for item in objects)


def test_table_detection_checks_cancellation_and_has_safe_errors(monkeypatch):
    from app.services import source_parse_pipeline
    calls = []
    def detect(**kwargs):
        calls.append(True)
        raise RuntimeError("unit-test source content must not be logged")
    page = SimpleNamespace(find_tables=detect)
    def cancelled():
        raise InterruptedError("unit-test cancellation")
    monkeypatch.setattr(source_parse_pipeline, "check_parse_cancellation", cancelled)
    with pytest.raises(InterruptedError):
        parsers._pdf_table_objects(page, "", [], page_number=1)
    assert not calls
    monkeypatch.setattr(source_parse_pipeline, "check_parse_cancellation", lambda: None)
    with pytest.raises(ValueError, match="^pdf_table_geometry_detection_failed$") as caught:
        parsers._pdf_table_objects(page, "", [], page_number=1)
    assert caught.value.__suppress_context__


def test_ruled_pdf_structure_mapping_persists_only_native_table_span(db_session, sample_knowledge_base, tmp_path):
    from app.models import ChunkStructureNode, ChunkStructureMapping, Document, DocumentVersion
    from app.services.context_graph import write_chunks_and_structure
    path, _ = ruled_pdf(tmp_path)
    sections = parsers.parse_pdf(path)
    document = Document(knowledge_base_id=sample_knowledge_base.id, title="Unit test table",
        source_path=str(path), source_type="pdf", checksum="unit-test-table", tags=[], is_active=True)
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version=1, checksum=document.checksum,
        storage_path=str(path), is_active=True)
    db_session.add(version)
    db_session.flush()
    chunks = write_chunks_and_structure(db_session, knowledge_base=sample_knowledge_base,
        document=document, version=version, sections=sections, chunk_version=1, chunk_size=128, chunk_overlap=8)
    db_session.flush()
    tables = list(db_session.scalars(select(ChunkStructureNode).where(
        ChunkStructureNode.document_version_id == version.id, ChunkStructureNode.node_type == "table")))
    assert len(tables) == 1
    table = tables[0]
    assert table.layout_json["metadata"]["table_geometry"]["protocol_version"] == parsers.PDF_TABLE_GEOMETRY_PROTOCOL_VERSION
    containing_chunk = next(chunk for chunk in chunks
        if chunk.char_start <= table.char_start and chunk.char_end >= table.char_end)
    assert containing_chunk.text[table.char_start - containing_chunk.char_start:table.char_end - containing_chunk.char_start] == next(
        obj.text for obj in sections[0].structure_objects if obj.object_type == "table")
    assert db_session.scalar(select(ChunkStructureMapping).where(
        ChunkStructureMapping.structure_node_id == table.id, ChunkStructureMapping.chunk_id.in_([c.id for c in chunks])))
    from app.retrieval_control_contracts import control_hash
    from app.services.evidence_scope import StructureScopeIndex
    from app.services.retrieval_corpus import CorpusSource
    from test_evidence_scope import request
    sources = tuple(CorpusSource(chunk.id,chunk.document_id,chunk.document_version_id,document.title,
        chunk.text,chunk.char_start,chunk.char_end,chunk.text_hash) for chunk in chunks)
    corpus = SimpleNamespace(knowledge_base_id=sample_knowledge_base.id,sources=sources,
        by_id={source.chunk_id:source for source in sources},scope_hash=control_hash('unit-pdf-scope'))
    binding = StructureScopeIndex.load(db_session,corpus=corpus).resolve(request('table','Table 1','label'))
    assert binding.reason == 'resolved'
    assert [(span.start,span.end) for span in binding.fact.intervals] == [(table.char_start,table.char_end)]
    from test_evidence_scope import task,obligation
    fixed=task(obligation(request('table','Table 1','label')),'Read Table 1').model_copy(update={'knowledge_base_id':sample_knowledge_base.id})
    narrow=StructureScopeIndex.load(db_session,corpus=corpus,task=fixed)
    replay=narrow.resolve(request('table','Table 1','label'))
    assert replay.fact == binding.fact and replay.node_ids == binding.node_ids
    with pytest.raises(ValueError,match='task_identity_changed'):
        narrow.bind(fixed.model_copy(update={'question':'A different question'}))
