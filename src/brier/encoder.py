"""Turning words into vectors, with a cache because training reads them often."""

import hashlib
import json
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

REQUEST_TIMEOUT_SECONDS = 120.0


class EncoderError(RuntimeError):
    """Raised when the encoder cannot produce vectors."""


class Encoder(Protocol):
    def encode(self, texts: tuple[str, ...]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class OllamaEncoder:
    """A frozen embedding model, served locally.

    The weights here are not ours and are never trained. Everything the model
    learns lives in the head on top, which is what makes training cheap enough
    to run on a laptop.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._dimension = 0

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._dimension = int(self.encode(("probe",)).shape[1])
        return self._dimension

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            raise ValueError("nothing to encode")
        try:
            response = httpx.post(
                self._url,
                json={"model": self._model, "input": list(texts)},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            payload = response.raise_for_status().json()
        except httpx.HTTPError as error:
            raise EncoderError(f"{self._url} did not answer: {error}") from error

        vectors = payload.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise EncoderError(f"asked for {len(texts)} vectors, got {len(vectors or [])}")
        return _unit(np.asarray(vectors, dtype=np.float32))


class CachedEncoder:
    """Remembers vectors on disk, keyed by the text that produced them."""

    def __init__(self, inner: Encoder, cache_dir: Path) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._vectors_path = self._dir / "vectors.npy"
        self._index: dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._load()

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def _load(self) -> None:
        if self._index_path.exists() and self._vectors_path.exists():
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._vectors = np.load(self._vectors_path)

    def _save(self) -> None:
        self._index_path.write_text(json.dumps(self._index), encoding="utf-8")
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        # The same guard the inner encoder makes. A wrapper that quietly accepts
        # what the thing it wraps rejects is a wrapper the caller cannot reason
        # about, and here the empty case used to reach an assert instead.
        if not texts:
            raise ValueError("nothing to encode")

        missing = tuple(text for text in dict.fromkeys(texts) if _key(text) not in self._index)
        if missing:
            fresh = self._inner.encode(missing)
            self._vectors = fresh if self._vectors is None else np.vstack([self._vectors, fresh])
            for offset, text in enumerate(missing):
                self._index[_key(text)] = len(self._vectors) - len(missing) + offset
            self._save()

        if self._vectors is None:
            raise EncoderError(f"cache at {self._dir} has an index but no vectors")
        return self._vectors[[self._index[_key(text)] for text in texts]]


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)
