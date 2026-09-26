"""A deterministic stand-in for the encoder, so tests never need Ollama."""

import hashlib

import numpy as np
import pytest

FAKE_DIMENSION = 12


class FakeEncoder:
    """Hashes text into a fixed vector. Same text, same vector, no network."""

    def __init__(self, dimension: int = FAKE_DIMENSION) -> None:
        self._dimension = dimension
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            raw = np.random.default_rng(seed).standard_normal(self._dimension)
            vectors.append(raw / np.linalg.norm(raw))
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()
