from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import parsers
from app.services.parsers import clean_extracted_text, decode_text_bytes, derive_chapter, parse_document


def test_parse_markdown_html_notebook(tmp_path, no_fallback_env):
    markdown = tmp_path / "notes.md"
    markdown.write_text("# Centrality\nDegree centrality counts edges.", encoding="utf-8")
    html = tmp_path / "notes.html"
    html.write_text("<html><title>Networks</title><body><h1>Graphs</h1><p>Nodes and edges.</p></body></html>", encoding="utf-8")
    notebook = tmp_path / "lab.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Lab\n", "Centrality exercise"], "outputs": []},
                    {"cell_type": "code", "source": ["print('graph')"], "outputs": [{"text": ["graph\n"]}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert parse_document(markdown)[0] == "markdown"
    assert "Nodes and edges" in parse_document(html)[1][0].text
    _, notebook_sections = parse_document(notebook)
    assert {section.metadata["content_kind"] for section in notebook_sections} == {"markdown", "code"}


def test_decode_text_bytes_handles_common_encodings():
    utf8_bom, utf8_meta = decode_text_bytes("\ufeffBayesian inference".encode("utf-8-sig"))
    gb18030_text, gb18030_meta = decode_text_bytes("贝叶斯 inference".encode("gb18030"))
    latin_text, latin_meta = decode_text_bytes("café likelihood".encode("cp1252"))

    assert utf8_bom.lstrip("\ufeff") == "Bayesian inference"
    assert "encoding_used" in utf8_meta
    assert "贝叶斯" in gb18030_text
    assert "encoding_used" in gb18030_meta
    assert "café" in latin_text
    assert "encoding_used" in latin_meta


def test_clean_extracted_text_repairs_mojibake_and_preserves_math():
    cleaned, metadata = clean_extracted_text("cafÃ© posterior\x00\n\n\\(\\alpha + \\beta \\le \\gamma\\)\nE = mc^2")

    assert "café posterior" in cleaned
    assert "\\alpha" in cleaned
    assert "\\beta" in cleaned
    assert "E = mc^2" in cleaned
    assert metadata["mojibake_repaired"] is True
    assert "removed_control_chars" in metadata["text_cleaning_flags"]


def test_clean_extracted_text_normalizes_pdf_ocr_layout_noise():
    cleaned, metadata = clean_extracted_text("inter-\nnational\nnetwork\n\nnext paragraph", source_type="pdf")

    assert "international network" in cleaned
    assert "next paragraph" in cleaned
    assert "normalized_pdf_ocr_linebreaks" in metadata["text_cleaning_flags"]


def test_parse_docx_pptx_pdf(tmp_path, no_fallback_env):
    docx = pytest.importorskip("docx")
    pptx = pytest.importorskip("pptx")
    fitz = pytest.importorskip("fitz")

    doc_path = tmp_path / "chapter.docx"
    document = docx.Document()
    document.add_heading("Chapter", level=1)
    document.add_paragraph("Document text for parsing.")
    document.save(doc_path)

    ppt_path = tmp_path / "slides.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Slide Title"
    slide.placeholders[1].text = "Slide body text"
    presentation.save(ppt_path)

    pdf_path = tmp_path / "paper.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF parsing text")
    pdf.save(pdf_path)
    pdf.close()

    assert parse_document(doc_path)[1][0].metadata["content_kind"] == "doc_section"
    assert parse_document(ppt_path)[1][0].metadata["content_kind"] == "slide"
    assert "PDF parsing text" in parse_document(pdf_path)[1][0].text


def test_parse_pdf_with_image_appends_ocr_text(tmp_path, no_fallback_env, monkeypatch):
    fitz = pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "scan.png"
    image = Image.new("RGB", (120, 60), color="white")
    image.save(image_path)

    pdf_path = tmp_path / "scan.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=180, height=100)
    page.insert_image(fitz.Rect(10, 10, 130, 70), filename=str(image_path))
    pdf.save(pdf_path)
    pdf.close()

    monkeypatch.setattr(parsers, "ocr_image_to_text", lambda image, source_name: ("OCR centrality text", {"ocr_engine": "unit-ocr"}))

    _source_type, sections = parse_document(pdf_path)

    assert "OCR centrality text" in sections[0].text
    assert sections[0].metadata["ocr_applied"] is True
    assert sections[0].metadata["ocr_engine"] == "unit-ocr"
    assert sections[0].metadata["ocr_reason"] == "page_contains_images"


def test_parse_pdf_with_image_continues_when_ocr_fails(tmp_path, no_fallback_env, monkeypatch):
    fitz = pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (80, 40), color="white").save(image_path)
    pdf_path = tmp_path / "scan.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=120, height=80)
    page.insert_image(fitz.Rect(10, 10, 90, 50), filename=str(image_path))
    pdf.save(pdf_path)
    pdf.close()

    def fail_ocr(image, source_name):
        raise RuntimeError("OCR dependencies unavailable for scan.pdf p.1 img.1: install/enable the api OCR extra")

    monkeypatch.setattr(parsers, "ocr_image_to_text", fail_ocr)

    _source_type, sections = parse_document(pdf_path)
    assert sections == []


def test_parse_pdf_with_image_continues_when_ocr_fails_but_text_exists(tmp_path, no_fallback_env, monkeypatch):
    fitz = pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (80, 40), color="white").save(image_path)
    pdf_path = tmp_path / "scan.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=120, height=80)
    page.insert_text((10, 70), "Native PDF text")
    page.insert_image(fitz.Rect(10, 10, 90, 50), filename=str(image_path))
    pdf.save(pdf_path)
    pdf.close()

    def fail_ocr(image, source_name):
        raise RuntimeError("OCR dependencies unavailable for scan.pdf p.1 img.1: install/enable the api OCR extra")

    monkeypatch.setattr(parsers, "ocr_image_to_text", fail_ocr)

    _source_type, sections = parse_document(pdf_path)
    assert len(sections) == 1
    assert "Native PDF text" in sections[0].text
    assert sections[0].metadata.get("ocr_image_errors")
    assert "OCR dependencies unavailable" in sections[0].metadata["ocr_image_errors"][0]


def test_parser_does_not_call_unstructured_when_fallback_disabled(tmp_path, no_fallback_env, monkeypatch):
    broken = tmp_path / "broken.md"
    broken.write_text("# Broken", encoding="utf-8")
    called = False

    def fail_markdown(path):
        raise ValueError("forced parser failure")

    def fail_if_called(path):
        nonlocal called
        called = True
        raise AssertionError("unstructured fallback must not be called")

    monkeypatch.setattr(parsers, "parse_markdown", fail_markdown)
    monkeypatch.setattr(parsers, "parse_with_unstructured", fail_if_called)

    with pytest.raises(RuntimeError, match="Failed to parse broken.md"):
        parse_document(broken)
    assert called is False


def test_unsupported_type_requires_explicit_fallback(tmp_path, no_fallback_env, monkeypatch):
    unknown = tmp_path / "data.unknown"
    unknown.write_text("unknown", encoding="utf-8")
    monkeypatch.setattr(parsers, "parse_with_unstructured", lambda path: pytest.fail("fallback called"))

    with pytest.raises(RuntimeError, match="Unsupported source type"):
        parse_document(unknown)


def test_derive_chapter_uses_filename_before_date_or_course_folder():
    assert derive_chapter(Path("fixtures/course-a/storage/20260425/Chapter 1.pdf"), "Course A") == "Chapter 1"
    assert derive_chapter(Path("fixtures/course-b/storage/20260425/Lecture 5 - Slides.pdf"), "Course B") == "Lecture 5"
    assert derive_chapter(Path("fixtures/course-b/storage/20260425/Lecture 9 - Sildes.pdf"), "Course B") == "Lecture 9"
    assert derive_chapter(Path("fixtures/course-a/storage/20260425/Week 2 - Solutions.pdf"), "Course A") == "Week 2"
    assert derive_chapter(Path("fixtures/course-b/storage/20260425/Lab Questions.pdf"), "Course B") == "Lab Questions"
    assert derive_chapter(Path("fixtures/course-a/storage/20260425/Labs solutions.pdf"), "Course A") == "Lab Solutions"
    assert derive_chapter(Path("fixtures/course-b/storage/20260425/Written Coursework.pdf"), "Course B") == "Coursework"
    assert derive_chapter(Path("fixtures/course-b/storage/20260425/Z-Table.pdf"), "Course B") == "Reference"
    assert derive_chapter(Path("fixtures/course-a/storage/20260425/graph_algorithms_visualizer.html"), "Course A") == "Reference"
    assert derive_chapter(Path("fixtures/course-a/storage/20260425/参考资料.pdf"), "Course A") == "Reference"
    assert derive_chapter(Path("fixtures/course-b/storage/20260425/topic-notes.md"), "Course B") == "topic notes"
