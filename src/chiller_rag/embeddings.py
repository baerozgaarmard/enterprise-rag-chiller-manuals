"""Embedding provider abstraction with deterministic local and test fakes."""

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Protocol for embedding computation engines."""

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class LocalSemanticHashEmbeddingProvider:
    """Deterministic local embedding provider using hashed n-gram token projections.

    Produces dense, unit-normalized vectors without downloading external model weights
    or calling remote APIs. Ideal for zero-dependency local evaluation and CI environments.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dim

    def _embed_single(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec

        # Bag of words and character tri-grams
        for word in words:
            idx = self._hash_token(word)
            vec[idx] += 1.0
            for i in range(max(1, len(word) - 2)):
                tri = word[i : i + 3]
                vec[self._hash_token(tri)] += 0.5

        # L2 normalization for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = [self._embed_single(t) for t in texts]
        return np.vstack(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_single(text)


class FakeEmbeddingProvider:
    """Fixed-dimension fake embedding provider for deterministic unit testing."""

    def __init__(self, dimension: int = 32) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 1000
            np.random.seed(seed)
            v = np.random.randn(self._dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-6
            vectors.append(v)
        return np.vstack(vectors) if vectors else np.empty((0, self._dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % 1000
        np.random.seed(seed)
        v = np.random.randn(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-6)
