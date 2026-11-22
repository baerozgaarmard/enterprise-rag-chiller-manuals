# Engineering Session Log - Project 2: enterprise-rag-chiller-manuals

- **Date:** 2026-11-22
- **Goal:** Implement an enterprise retrieval-augmented generation (RAG) system for industrial chiller manuals with strict citations, safety rules, and local deterministic execution.

## Key Decisions Made
1. **Zero External Dependency**: Implemented self-contained `parse_pdf_pages` using PyMuPDF to ensure zero cross-repo runtime coupling.
2. **Chunking & Provenance**: Designed `Chunker` with sliding character window and overlap, tracking document ID, filename, page number, and offsets.
3. **Pluggable Vector Engine**: Built `LocalVectorStore` with NumPy cosine similarity and JSON persistence, enabling fast local retrieval and document purging.
4. **Factual Grounding & Safety**: Implemented `GroundedRuleLlmProvider` that enforces evidence retrieval thresholds, refuses unsupported questions (`insufficient_evidence: true`), extracts exact quotes, and appends mandatory LOTO safety disclaimers for hazardous procedures.

## Verification
- 13 automated unit & integration tests passing (`pytest -v`).
- Zero linting or formatting errors (`ruff check .`, `ruff format --check .`).
