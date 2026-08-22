from __future__ import annotations

from app.services.chunking import (
    CHUNK_TEXT_HASH_PROTOCOL_VERSION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    FixedTokenChunker,
    text_hash,
)
from app.services.parsers import ParsedSection


def test_fixed_token_chunker_preserves_protected_objects():
    sections = [
        ParsedSection(
            title="Protected objects",
            text=(
                "# Protected objects\n"
                "Intro text before the table.\n\n"
                "| variable | meaning |\n| X | node |\n| Pa(X) | parents |\n\n"
                "$$P(X)=\\prod_i P(X_i | Pa(X_i))$$\n\n"
                "```python\nprint('bayes')\n```\n\n"
                "Figure 1: Bayesian network with conditional dependencies."
            ),
            page_number=4,
            section="Protected objects",
        )
    ]
    chunks, _prepared = FixedTokenChunker(chunk_size=12, overlap=2).split_sections(sections, title="Protected objects")
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "| variable | meaning |" in joined
    assert "$$P(X)=" in joined
    assert "```python" in joined
    assert "Figure 1:" in joined
    assert any(chunk.metadata.get("has_table") for chunk in chunks)
    assert any(chunk.metadata.get("has_formula") for chunk in chunks)
    assert any(chunk.metadata.get("content_kind") == "code" for chunk in chunks)


def test_fixed_token_chunker_default_is_512_80():
    chunker = FixedTokenChunker()

    assert DEFAULT_CHUNK_SIZE == 512
    assert DEFAULT_CHUNK_OVERLAP == 80
    assert chunker.chunk_size == 512
    assert chunker.overlap == 80


def test_chunk_text_hash_uses_the_versioned_normalized_sha256_protocol():
    assert CHUNK_TEXT_HASH_PROTOCOL_VERSION == "chunk_text_sha256_normalized_v1"
    assert text_hash(" alpha\x00  beta\r\n\r\n\r\n\r\ngamma ") == text_hash("alpha beta\n\n\ngamma")
    assert text_hash("alpha beta") != text_hash("alpha gamma")
