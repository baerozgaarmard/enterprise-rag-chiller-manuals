"""Pydantic data models for chunking, vector storage, retrieval, and grounded Q&A."""

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    page_count: int = Field(ge=0)
    indexed_at: str
    chunk_count: int = Field(ge=0)


class TextChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class RetrievedPassage(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    text: str
    score: float


class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    chunk_id: str
    quote: str


class RetrievalSummary(BaseModel):
    result_count: int = Field(ge=0)
    max_score: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[Citation] = Field(default_factory=list)
    retrieval: RetrievalSummary


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrieveResponse(BaseModel):
    passages: list[RetrievedPassage] = Field(default_factory=list)
    total_retrieved: int = Field(ge=0)
