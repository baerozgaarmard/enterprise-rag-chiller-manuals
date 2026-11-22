"""Independent PDF parser extracting per-page text with page boundaries."""

from pathlib import Path

import pymupdf


class PdfIngestionError(Exception):
    """Raised when PDF manual ingestion or parsing fails."""


def parse_pdf_pages(
    source: bytes | Path | str, filename: str = "manual.pdf"
) -> list[tuple[int, str]]:
    """Extract page number and raw text tuples from PDF source."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise PdfIngestionError(f"Manual file not found at: {path}")
        raw_bytes = path.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    else:
        raise PdfIngestionError(f"Unsupported source type: {type(source)}")

    if not raw_bytes or not raw_bytes.startswith(b"%PDF-"):
        raise PdfIngestionError("Invalid PDF format: file lacks standard PDF header.")

    try:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfIngestionError(f"Failed to open PDF stream for {filename}: {exc}") from exc

    pages: list[tuple[int, str]] = []
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PdfIngestionError(f"Cannot parse encrypted manual: '{filename}'.")

        for idx, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            pages.append((idx + 1, page_text))
    finally:
        doc.close()

    return pages
