"""Public parser metadata accepted by the closed QA context projection."""
import pytest
from pydantic import ValidationError

from app.schemas import ContextStructureLayoutAudit


def test_pdf_structure_and_caption_metadata_have_explicit_public_fields():
    layout = ContextStructureLayoutAudit.model_validate({
        "native_structure_block_count": 3,
        "metadata": {"source_block_index": 2, "style": "Heading"},
    }, strict=True)
    assert layout.native_structure_block_count == 3
    assert layout.metadata.source_block_index == 2
    assert layout.metadata.style == "Heading"

    with pytest.raises(ValidationError):
        ContextStructureLayoutAudit.model_validate({
            "native_structure_block_count": 3,
            "metadata": {"unregistered_parser_key": "x"},
        }, strict=True)
