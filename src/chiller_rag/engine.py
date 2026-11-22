"""Unified RAG coordination engine connecting ingestion, chunking, storage, and retrieval."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from chiller_rag.chunker import Chunker
from chiller_rag.config import settings
from chiller_rag.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    LocalSemanticHashEmbeddingProvider,
)
from chiller_rag.generator import (
    FakeLlmProvider,
    GroundedRuleLlmProvider,
    LlmProvider,
)
from chiller_rag.models import (
    DocumentRecord,
    QueryRequest,
    QueryResponse,
    RetrievalSummary,
    RetrieveRequest,
    RetrieveResponse,
)
from chiller_rag.pdf_parser import parse_pdf_pages
from chiller_rag.vector_store import LocalVectorStore, VectorStore


class ChillerRagEngine:
    """Enterprise RAG engine for technical chiller manuals."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider: LlmProvider | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunker = Chunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )
        self.vector_store = vector_store or LocalVectorStore(
            persistence_path=settings.data_dir / "vector_store.json"
        )
        self.embedding_provider = embedding_provider or (
            FakeEmbeddingProvider()
            if settings.embedding_provider == "fake"
            else LocalSemanticHashEmbeddingProvider()
        )
        self.llm_provider = llm_provider or (
            FakeLlmProvider() if settings.llm_provider == "fake" else GroundedRuleLlmProvider()
        )

    def ingest_manual(
        self,
        source: bytes | Path | str,
        filename: str = "chiller_manual.pdf",
        document_id: str | None = None,
    ) -> DocumentRecord:
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:10]}"
        # Parse PDF into page-level text
        pages = parse_pdf_pages(source=source, filename=filename)

        # Chunk pages preserving page boundaries and offsets
        chunks = self.chunker.chunk_pages(pages, document_id=doc_id, filename=filename)

        # Compute embeddings for extracted chunks
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedding_provider.embed_texts(chunk_texts)
            self.vector_store.add_chunks(chunks, embeddings)

        record = DocumentRecord(
            document_id=doc_id,
            filename=filename,
            page_count=len(pages),
            indexed_at=datetime.now(UTC).isoformat(),
            chunk_count=len(chunks),
        )
        if hasattr(self.vector_store, "add_document_record"):
            self.vector_store.add_document_record(record)

        return record

    def query(self, request: QueryRequest) -> QueryResponse:
        top_k = request.top_k or settings.top_k
        query_vec = self.embedding_provider.embed_query(request.question)

        passages = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            similarity_threshold=settings.similarity_threshold,
            document_ids=request.document_ids,
        )

        answer, insufficient, citations = self.llm_provider.generate_answer(
            question=request.question,
            passages=passages,
        )

        max_score = passages[0].score if passages else 0.0
        return QueryResponse(
            answer=answer,
            insufficient_evidence=insufficient,
            citations=citations,
            retrieval=RetrievalSummary(result_count=len(passages), max_score=max_score),
        )

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        top_k = request.top_k or settings.top_k
        query_vec = self.embedding_provider.embed_query(request.query)

        passages = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            similarity_threshold=0.0,  # Return all ranked candidates for inspection
            document_ids=request.document_ids,
        )

        return RetrieveResponse(passages=passages, total_retrieved=len(passages))

    def delete_document(self, document_id: str) -> int:
        return self.vector_store.delete_document(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return self.vector_store.list_documents()
