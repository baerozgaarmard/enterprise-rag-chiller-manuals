"""Unit tests for embedding providers."""

import numpy as np

from chiller_rag.embeddings import FakeEmbeddingProvider, LocalSemanticHashEmbeddingProvider


def test_local_semantic_hash_embedding_provider() -> None:
    provider = LocalSemanticHashEmbeddingProvider(dimension=128)
    assert provider.dimension == 128

    text_a = "Centrifugal chiller compressor lubrication"
    text_b = "Centrifugal chiller compressor lubrication"
    text_c = "Solar panel electrical inverter output"

    vec_a = provider.embed_query(text_a)
    vec_b = provider.embed_query(text_b)
    vec_c = provider.embed_query(text_c)

    assert vec_a.shape == (128,)
    # Norm should be 1.0
    assert np.isclose(np.linalg.norm(vec_a), 1.0, atol=1e-4)

    # Identical texts yield identical vectors
    assert np.allclose(vec_a, vec_b)

    # Overlapping texts have positive cosine similarity
    sim_ac = float(np.dot(vec_a, vec_c))
    assert sim_ac < 1.0


def test_fake_embedding_provider() -> None:
    provider = FakeEmbeddingProvider(dimension=64)
    assert provider.dimension == 64

    matrix = provider.embed_texts(["sample text one", "sample text two"])
    assert matrix.shape == (2, 64)
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0, atol=1e-4)
