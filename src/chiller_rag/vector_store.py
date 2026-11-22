"""Vector store abstraction with local persistence and metadata filtering."""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from chiller_rag.models import DocumentRecord, RetrievedPassage, TextChunk


class VectorStore(Protocol):
    """Protocol for vector persistence and similarity retrieval."""

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None: ...

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedPassage]: ...

    def delete_document(self, document_id: str) -> int: ...

    def list_documents(self) -> list[DocumentRecord]: ...

    def get_document(self, document_id: str) -> DocumentRecord | None: ...

    def clear(self) -> None: ...


class LocalVectorStore:
    """Local vector store using NumPy for cosine similarity and JSON for metadata persistence."""

    def __init__(self, persistence_path: Path | None = None) -> None:
        self.persistence_path = persistence_path
        self.chunks: dict[str, TextChunk] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.documents: dict[str, DocumentRecord] = {}

        if self.persistence_path and self.persistence_path.exists():
            self._load()

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})"
            )

        for chunk, emb in zip(chunks, embeddings, strict=True):
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = emb.astype(np.float32)

        self._save()

    def add_document_record(self, doc: DocumentRecord) -> None:
        self.documents[doc.document_id] = doc
        self._save()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        if not self.chunks:
            return []

        # Filter candidate chunk IDs
        candidate_ids = [
            cid
            for cid, chunk in self.chunks.items()
            if (document_ids is None or chunk.document_id in document_ids)
        ]
        if not candidate_ids:
            return []

        matrix = np.vstack([self.vectors[cid] for cid in candidate_ids])
        q_norm = np.linalg.norm(query_vector)
        if q_norm < 1e-6:
            return []

        q_unit = query_vector / q_norm
        # Cosine similarities
        scores = np.dot(matrix, q_unit)

        scored_candidates: list[tuple[str, float]] = []
        for cid, score in zip(candidate_ids, scores, strict=True):
            s = float(score)
            if s >= similarity_threshold:
                scored_candidates.append((cid, s))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        passages: list[RetrievedPassage] = []
        for cid, score in top_candidates:
            chunk = self.chunks[cid]
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=round(score, 4),
                )
            )
        return passages

    def delete_document(self, document_id: str) -> int:
        to_delete = [cid for cid, chunk in self.chunks.items() if chunk.document_id == document_id]
        for cid in to_delete:
            del self.chunks[cid]
            del self.vectors[cid]

        if document_id in self.documents:
            del self.documents[document_id]

        self._save()
        return len(to_delete)

    def list_documents(self) -> list[DocumentRecord]:
        return list(self.documents.values())

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def clear(self) -> None:
        self.chunks.clear()
        self.vectors.clear()
        self.documents.clear()
        self._save()

    def _save(self) -> None:
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": {k: v.model_dump() for k, v in self.documents.items()},
            "chunks": {k: v.model_dump() for k, v in self.chunks.items()},
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
        }
        self.persistence_path.write_text(json.dumps(data), encoding="utf-8")

    def _load(self) -> None:
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            data = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            self.documents = {
                k: DocumentRecord.model_validate(v) for k, v in data.get("documents", {}).items()
            }
            self.chunks = {
                k: TextChunk.model_validate(v) for k, v in data.get("chunks", {}).items()
            }
            self.vectors = {
                k: np.array(v, dtype=np.float32) for k, v in data.get("vectors", {}).items()
            }
        except Exception:
            # If storage is corrupted, reset safely
            self.documents.clear()
            self.chunks.clear()
            self.vectors.clear()
