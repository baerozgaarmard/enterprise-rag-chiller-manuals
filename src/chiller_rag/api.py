"""FastAPI REST application for manual ingestion, vector retrieval, and grounded Q&A."""

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from chiller_rag import __version__
from chiller_rag.engine import ChillerRagEngine
from chiller_rag.models import (
    DocumentRecord,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from chiller_rag.pdf_parser import PdfIngestionError

app = FastAPI(
    title="Enterprise RAG for Chiller Manuals API",
    version=__version__,
    description="Grounded technical question answering and citation service for industrial chiller manuals.",
)

engine = ChillerRagEngine()


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Service health verification."""
    return {"status": "ok", "version": __version__}


@app.post(
    "/v1/documents",
    response_model=DocumentRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["Manuals"],
)
async def ingest_document(file: UploadFile = File(...)) -> DocumentRecord:
    """Upload and index a PDF technical chiller manual."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF chiller manuals (ending with .pdf) are accepted.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplied manual file is empty (0 bytes).",
        )

    try:
        record = engine.ingest_manual(source=content, filename=file.filename)
        return record
    except PdfIngestionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest manual: {exc}",
        ) from exc


@app.get("/v1/documents", response_model=list[DocumentRecord], tags=["Manuals"])
def list_documents() -> list[DocumentRecord]:
    """List all indexed technical manuals."""
    return engine.list_documents()


@app.delete("/v1/documents/{document_id}", tags=["Manuals"])
def delete_document(document_id: str) -> dict[str, int | str]:
    """Delete a manual and purge its indexed chunks from the vector store."""
    deleted_count = engine.delete_document(document_id)
    if deleted_count == 0 and not engine.vector_store.get_document(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return {"document_id": document_id, "deleted_chunks": deleted_count}


@app.post("/v1/query", response_model=QueryResponse, tags=["RAG"])
def query_manuals(request: QueryRequest) -> QueryResponse:
    """Submit a technical question and receive a grounded answer with citations."""
    try:
        return engine.query(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc


@app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["RAG"])
def retrieve_passages(request: RetrieveRequest) -> RetrieveResponse:
    """Inspect raw vector search results without synthesis for debugging and evaluation."""
    try:
        return engine.retrieve(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc
