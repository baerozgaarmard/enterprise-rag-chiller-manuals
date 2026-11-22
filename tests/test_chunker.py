"""Unit tests for deterministic chunker."""

import pytest

from chiller_rag.chunker import Chunker


def test_chunker_basic() -> None:
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    pages = [
        (1, "First paragraph about water chillers.\n\nSecond paragraph about cooling towers."),
        (2, "Third paragraph on electrical lockout tagout procedures."),
    ]

    chunks = chunker.chunk_pages(pages, document_id="doc_1", filename="manual.pdf")

    assert len(chunks) >= 2
    assert all(c.document_id == "doc_1" for c in chunks)
    assert all(c.filename == "manual.pdf" for c in chunks)
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
    assert chunks[0].char_end > chunks[0].char_start


def test_chunker_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=120)
