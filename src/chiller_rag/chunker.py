"""Deterministic, overlap-aware text chunker preserving page and document metadata."""

import re

from chiller_rag.models import TextChunk


class Chunker:
    """Chunks text documents with configurable size, overlap, and page-boundary preservation."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: list[tuple[int, str]],
        document_id: str,
        filename: str,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        global_chunk_idx = 0

        for page_num, text in pages:
            if not text.strip():
                continue

            # Split by paragraphs or double newlines first
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text.strip()]

            # Slide window across text blocks for this page
            buffer = ""
            start_offset = 0

            for para in paragraphs:
                if not buffer:
                    buffer = para
                elif len(buffer) + 1 + len(para) <= self.chunk_size:
                    buffer += "\n\n" + para
                else:
                    # Flush current buffer as a chunk
                    chunk_text = buffer.strip()
                    end_offset = start_offset + len(chunk_text)
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{document_id}_p{page_num}_c{global_chunk_idx}",
                            document_id=document_id,
                            filename=filename,
                            page_number=page_num,
                            text=chunk_text,
                            char_start=start_offset,
                            char_end=end_offset,
                        )
                    )
                    global_chunk_idx += 1

                    # Handle overlap
                    overlap_chars = (
                        buffer[-self.chunk_overlap :]
                        if len(buffer) > self.chunk_overlap
                        else buffer
                    )
                    buffer = (overlap_chars + "\n\n" + para).strip()
                    start_offset = max(0, end_offset - self.chunk_overlap)

            if buffer.strip():
                chunk_text = buffer.strip()
                end_offset = start_offset + len(chunk_text)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_p{page_num}_c{global_chunk_idx}",
                        document_id=document_id,
                        filename=filename,
                        page_number=page_num,
                        text=chunk_text,
                        char_start=start_offset,
                        char_end=end_offset,
                    )
                )
                global_chunk_idx += 1

        return chunks
