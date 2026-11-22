"""Unit tests for local vector store and metadata filtering."""

from pathlib import Path

import numpy as np

from chiller_rag.models import DocumentRecord, TextChunk
from chiller_rag.vector_store import LocalVectorStore


def test_vector_store_crud(tmp_path: Path) -> None:
    store_file = tmp_path / "store.json"
    store = LocalVectorStore(persistence_path=store_file)

    chunks = [
        TextChunk(
            chunk_id="doc1_c0",
            document_id="doc1",
            filename="manual1.pdf",
            page_number=1,
            text="Refrigerant charge details for model 500",
            char_start=0,
            char_end=40,
        ),
        TextChunk(
            chunk_id="doc2_c0",
            document_id="doc2",
            filename="manual2.pdf",
            page_number=1,
            text="Absorption chiller lithium bromide solution",
            char_start=0,
            char_end=43,
        ),
    ]

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.vstack([v1, v2])

    store.add_chunks(chunks, embeddings)
    store.add_document_record(
        DocumentRecord(
            document_id="doc1",
            filename="manual1.pdf",
            page_count=1,
            indexed_at="2026-11-22T00:00:00Z",
            chunk_count=1,
        )
    )

    # 1. Search without filter
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "doc1_c0"
    assert results[0].score == 1.0

    # 2. Document isolation: filter by doc2
    results_doc2 = store.search(query_vec, top_k=2, document_ids=["doc2"])
    assert len(results_doc2) == 1
    assert results_doc2[0].chunk_id == "doc2_c0"

    # 3. Persistence reload
    reloaded_store = LocalVectorStore(persistence_path=store_file)
    assert "doc1_c0" in reloaded_store.chunks
    assert "doc1" in reloaded_store.documents

    # 4. Deletion
    deleted = reloaded_store.delete_document("doc1")
    assert deleted == 1
    assert "doc1_c0" not in reloaded_store.chunks
    assert "doc1" not in reloaded_store.documents
