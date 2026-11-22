"""REST API tests for Chiller RAG application."""

from pathlib import Path

from fastapi.testclient import TestClient

from chiller_rag.api import app, engine
from chiller_rag.vector_store import LocalVectorStore

client = TestClient(app)


def test_api_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "0.1.0"}


def test_api_full_rag_workflow(synthetic_chiller_manual_pdf: Path, tmp_path: Path) -> None:
    # Use isolated test store
    engine.vector_store = LocalVectorStore(persistence_path=tmp_path / "api_test_store.json")

    # 1. Ingest document
    with open(synthetic_chiller_manual_pdf, "rb") as f:
        ingest_resp = client.post(
            "/v1/documents",
            files={"file": (synthetic_chiller_manual_pdf.name, f, "application/pdf")},
        )
    assert ingest_resp.status_code == 201
    doc_data = ingest_resp.json()
    doc_id = doc_data["document_id"]
    assert doc_data["page_count"] == 2
    assert doc_data["chunk_count"] > 0

    # 2. List documents
    list_resp = client.get("/v1/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert any(d["document_id"] == doc_id for d in docs)

    # 3. Retrieve passages debug endpoint
    retrieve_resp = client.post(
        "/v1/retrieve",
        json={"query": "condenser water flow rate GPM", "top_k": 3},
    )
    assert retrieve_resp.status_code == 200
    ret_data = retrieve_resp.json()
    assert ret_data["total_retrieved"] > 0

    # 4. Grounded Query endpoint
    query_resp = client.post(
        "/v1/query",
        json={"question": "What is the condenser water flow rate?"},
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert not q_data["insufficient_evidence"]
    assert "3.0 GPM" in q_data["answer"]
    assert len(q_data["citations"]) > 0
    assert q_data["citations"][0]["page_number"] == 1

    # 5. Delete document
    del_resp = client.delete(f"/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_chunks"] > 0

    # 6. Verify deletion reflected in query
    query_after_del = client.post(
        "/v1/query",
        json={"question": "What is the condenser water flow rate?"},
    )
    assert query_after_del.status_code == 200
    assert query_after_del.json()["insufficient_evidence"]
