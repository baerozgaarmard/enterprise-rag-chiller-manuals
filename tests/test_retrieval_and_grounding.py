"""Integration tests for retrieval grounding, citations, and safety boundaries."""

from pathlib import Path

from chiller_rag.engine import ChillerRagEngine
from chiller_rag.models import QueryRequest
from chiller_rag.vector_store import LocalVectorStore


def test_grounded_query_with_evidence(synthetic_chiller_manual_pdf: Path, tmp_path: Path) -> None:
    store = LocalVectorStore(persistence_path=tmp_path / "test_store.json")
    engine = ChillerRagEngine(vector_store=store)

    doc_record = engine.ingest_manual(
        source=synthetic_chiller_manual_pdf,
        filename=synthetic_chiller_manual_pdf.name,
        document_id="manual_x500",
    )
    assert doc_record.page_count == 2
    assert doc_record.chunk_count > 0

    # Query matching Page 1 content
    query_req = QueryRequest(question="What refrigerant does the Model X500 use?")
    response = engine.query(query_req)

    assert not response.insufficient_evidence
    assert "R-134a" in response.answer
    assert len(response.citations) > 0

    citation = response.citations[0]
    assert citation.document_id == "manual_x500"
    assert citation.page_number == 1
    assert "R-134a" in citation.quote


def test_safety_critical_question_includes_notice(
    synthetic_chiller_manual_pdf: Path, tmp_path: Path
) -> None:
    store = LocalVectorStore(persistence_path=tmp_path / "safety_store.json")
    engine = ChillerRagEngine(vector_store=store)
    engine.ingest_manual(source=synthetic_chiller_manual_pdf, document_id="manual_x500")

    query_req = QueryRequest(question="What electrical lockout procedure is required?")
    response = engine.query(query_req)

    assert not response.insufficient_evidence
    assert "SAFETY NOTICE" in response.answer
    assert any(c.page_number == 2 for c in response.citations)


def test_query_with_insufficient_evidence(
    synthetic_chiller_manual_pdf: Path, tmp_path: Path
) -> None:
    store = LocalVectorStore(persistence_path=tmp_path / "empty_store.json")
    engine = ChillerRagEngine(vector_store=store)
    engine.ingest_manual(source=synthetic_chiller_manual_pdf, document_id="manual_x500")

    # Unrelated query with zero evidence in manual
    query_req = QueryRequest(question="What is the recommended dosage for pediatric antibiotics?")
    response = engine.query(query_req)

    assert response.insufficient_evidence
    assert len(response.citations) == 0
    assert (
        "insufficient evidence" in response.answer.lower()
        or "not provide" in response.answer.lower()
    )
