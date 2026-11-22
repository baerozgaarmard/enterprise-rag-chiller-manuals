"""Unit tests for PDF manual parser."""

from pathlib import Path

import pytest

from chiller_rag.pdf_parser import PdfIngestionError, parse_pdf_pages


def test_parse_valid_pdf(synthetic_chiller_manual_pdf: Path) -> None:
    pages = parse_pdf_pages(synthetic_chiller_manual_pdf)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "R-134a refrigerant" in pages[0][1]
    assert pages[1][0] == 2
    assert "High voltage" in pages[1][1]


def test_parse_corrupt_file(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"INVALID_HEADER_DATA")

    with pytest.raises(PdfIngestionError) as exc_info:
        parse_pdf_pages(bad_pdf)
    assert "Invalid PDF format" in str(exc_info.value)


def test_parse_empty_bytes() -> None:
    with pytest.raises(PdfIngestionError):
        parse_pdf_pages(b"")
